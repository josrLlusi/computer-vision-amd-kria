"""Float YOLO26 one2one head -> int8 program for the RTL, plus float reference + RTL runner.

Float head format (BN already folded into conv):
  head['box'][s] = [L, L, L]        L = dict(w=ndarray, b=ndarray, dw=bool, act=bool, k=int)
  head['cls'][s] = [L, L, L, L, L]  dense w:(cout,cin,k,k)   depthwise w:(c,k,k)
"""
import json, math, os, subprocess, shutil
import numpy as np
import golden as g

STRIDE_SHIFT = [3, 4, 5]          # strides 8, 16, 32


def silu(x):
    return x / (1.0 + np.exp(-x))


def fconv(x, L):
    k = L['k']; pad = k // 2
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    w = L['w']
    cout = w.shape[0]
    out = np.zeros((cout, H, W))
    for ky in range(k):
        for kx in range(k):
            win = xp[:, ky:ky + H, kx:kx + W]
            out += (win * w[:, ky, kx][:, None, None]) if L['dw'] else np.einsum('oc,chw->ohw', w[:, :, ky, kx], win)
    return out + L['b'][:, None, None]


def fchain(layers, x):
    outs = []
    for L in layers:
        pre = fconv(x, L)
        post = silu(pre) if L['act'] else pre
        outs.append((pre, post)); x = post
    return outs


def quantize_chain(layers, outs, H, W, s_in, s_final, final_sat16):
    qs = []
    for i, (L, (pre, post)) in enumerate(zip(layers, outs)):
        w = L['w']; cout = w.shape[0]
        sw = np.maximum(np.abs(w.reshape(cout, -1)).max(axis=1), 1e-12) / 127.0
        wq = np.clip(np.round(w / sw.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int64)
        acc_scale = s_in * sw
        bq = np.clip(np.round(L['b'] / acc_scale), -2**31 + 1, 2**31 - 1).astype(np.int64)
        lut = None
        if L['act']:
            s_out = max(np.abs(pre).max(), 1e-9) / 127.0
            s_post = max(np.abs(post).max(), 1e-9) / 127.0
            lut = g.silu_lut(s_out, s_post)
        else:
            s_out = s_final
        ratio = acc_scale / s_out
        shift = int(np.clip(math.floor(math.log2((2**30 - 1) / ratio.max())), 1, 31))
        mult = np.round(ratio * 2**shift).astype(np.int64)
        last = i == len(layers) - 1
        qs.append(g.LayerQ(w.shape[1] if not L['dw'] else w.shape[0], cout, L['k'], H, W, wq, bq, mult, shift,
                           dw=L['dw'], act=L['act'], sat16=(last and final_sat16), lut=lut))
        s_in = (s_post if L['act'] else None)
    return qs


def quantize_head(head, feats, nc, FB=5, pct=100.0):
    """feats: 3 float arrays (C,H,W). Returns (Program, info)."""
    fouts_box = [fchain(head['box'][s], feats[s]) for s in range(3)]
    fouts_cls = [fchain(head['cls'][s], feats[s]) for s in range(3)]
    s_cls = max(np.abs(fouts_cls[s][-1][1]).max() for s in range(3)) / 127.0
    s_cls = max(s_cls, 1e-9)
    inputs, box_chains, cls_chains = [], [], []
    for s in range(3):
        C, H, W = feats[s].shape
        s_in = max(np.percentile(np.abs(feats[s]), pct), 1e-9) / 127.0
        inputs.append(np.clip(np.round(feats[s] / s_in), -128, 127).astype(np.int64))
        box_chains.append(quantize_chain(head['box'][s], fouts_box[s], H, W, s_in, 2.0 ** -FB, True))
        cls_chains.append(quantize_chain(head['cls'][s], fouts_cls[s], H, W, s_in, s_cls, False))
    P = g.assemble(inputs, box_chains, cls_chains, nc, FB, STRIDE_SHIFT, g.sigmoid_lut(s_cls))
    info = dict(s_cls=s_cls, float_box=[fouts_box[s][-1][1] for s in range(3)],
                float_cls=[fouts_cls[s][-1][1] for s in range(3)])
    return P, info


def float_post(box, cls, max_det=300):
    """Float reference of the head post-processing. box[s]:(4,H,W) grid units, cls[s]:(nc,H,W) logits."""
    cells, best, L_all = [], [], []
    for s in range(3):
        _, H, W = cls[s].shape
        for y in range(H):
            for x in range(W):
                cells.append((s, y, x)); L_all.append(cls[s][:, y, x]); best.append(cls[s][:, y, x].max())
    k = min(max_det, len(cells))
    order1 = sorted(range(len(cells)), key=lambda i: (-best[i], i))[:k]
    nc = cls[0].shape[0]
    flat = [(L_all[order1[j]][c], j * nc + c) for j in range(len(order1)) for c in range(nc)]
    order2 = sorted(flat, key=lambda t: (-t[0], t[1]))[:k]
    rows = []
    for v, f in order2:
        j, c = divmod(f, nc); s, y, x = cells[order1[j]]
        d = box[s][:, y, x]; st = 2 ** STRIDE_SHIFT[s]
        rows.append([(x + .5 - d[0]) * st, (y + .5 - d[1]) * st, (x + .5 + d[2]) * st, (y + .5 + d[3]) * st,
                     1 / (1 + np.exp(-v)), c])
    return np.array(rows)


def run_rtl(P, workdir, max_det=300, simulator_dir=None):
    """Write hex files, compile + run the Verilog testbench, return rows (int) and cycle count."""
    here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
    os.makedirs(workdir, exist_ok=True)
    g.write_files(P, workdir)
    lg = lambda n, lo: max(lo, math.ceil(math.log2(n + 1)))
    params = dict(AAW=lg(P.act_words, 10), WAW=lg(len(P.wgt), 10), PAW=lg(len(P.par), 10),
                  LAW=lg(len(P.lut) + 256, 8), PRW=lg(len(P.prog), 6), K=max_det, OUTAW=lg(6 * max_det + 6, 8))
    srcs = [f'{root}/rtl/{f}' for f in ('ram_sync.v', 'topk_sorter.v', 'conv_engine.v', 'det_post.v', 'yolo26_head_top.v')] + [f'{root}/tb/tb_head.v']
    cmd = ['iverilog', '-g2012', '-o', f'{workdir}/tb.vvp'] + [f'-Ptb_head.{k}={v}' for k, v in params.items()] + srcs
    subprocess.run(cmd, check=True)
    r = subprocess.run(['vvp', '-n', 'tb.vvp', '+maxcyc=2000000000'], cwd=workdir, capture_output=True, text=True)
    cyc = [l for l in r.stdout.splitlines() if l.startswith('CYCLES')]
    if not cyc:
        raise RuntimeError('simulation did not finish:\n' + r.stdout[-500:])
    rows = [list(map(int, l.split())) for l in open(f'{workdir}/rows.txt') if l.strip()]
    return rows, int(cyc[0].split('=')[1])


def dequant_rows(rows):
    a = np.array(rows, dtype=float)
    if len(a) == 0:
        return np.zeros((0, 6))
    a[:, 0:4] /= 4.0; a[:, 4] /= 65535.0
    return a
