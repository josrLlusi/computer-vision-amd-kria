"""Float YOLO26 one2one head -> int8 program for the RTL (v2/Kria), plus float reference and RTL runner.

Float head format (BN already folded into conv):
  head['box'][s] = [L, L, L]        L = dict(w=ndarray, b=ndarray, dw=bool, act=bool, k=int)
  head['cls'][s] = [L, L, L, L, L]  dense w:(cout,cin,k,k)   depthwise w:(c,k,k)
Quantisation: int8 activations (per-tensor scale), int8 weights (per-output-channel scale), per-channel u16 multiplier
and 6-bit shift, SiLU as a 256-entry LUT, ONE shared logit scale for all 3 scales (so top-k compares across scales),
box distances int16 with FB fractional bits.
"""
import math, os, subprocess
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
    out = np.zeros((w.shape[0], H, W))
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


def calibrate(head, feats_list, pct=100.0):
    """Activation statistics over a calibration set. feats_list: list of [f3,f4,f5] float arrays (C,H,W).
    Per-tensor scales are the max over images of the per-image abs-max (or pct percentile for the inputs)."""
    st = dict(s_in=[0.0] * 3, box=[None] * 3, cls=[None] * 3, cls_max=0.0)
    for feats in feats_list:
        for s in range(3):
            st['s_in'][s] = max(st['s_in'][s], np.percentile(np.abs(feats[s]), pct) / 127.0)
            for kind in ('box', 'cls'):
                outs = fchain(head[kind][s], feats[s])
                cur = [(np.abs(pre).max(), np.abs(post).max()) for pre, post in outs]
                st[kind][s] = cur if st[kind][s] is None else [(max(a, c), max(b, d)) for (a, b), (c, d) in zip(st[kind][s], cur)]
                if kind == 'cls':
                    st['cls_max'] = max(st['cls_max'], cur[-1][1])
    st['s_cls'] = max(st['cls_max'] / 127.0, 1e-9)
    st['s_in'] = [max(v, 1e-9) for v in st['s_in']]
    return st


def quantize_chain(layers, stats, H, W, s_in, s_final, final_sat16):
    qs = []
    for i, (L, (pre_max, post_max)) in enumerate(zip(layers, stats)):
        w = L['w']; cout = w.shape[0]
        sw = np.maximum(np.abs(w.reshape(cout, -1)).max(axis=1), 1e-12) / 127.0
        wq = np.clip(np.round(w / sw.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int64)
        acc_scale = s_in * sw
        bq = np.clip(np.round(L['b'] / acc_scale), -2**31 + 1, 2**31 - 1).astype(np.int64)
        lut = None
        if L['act']:
            s_out = max(pre_max, 1e-9) / 127.0
            s_post = max(post_max, 1e-9) / 127.0
            lut = g.silu_lut(s_out, s_post)
        else:
            s_out = s_final
        ratio = acc_scale / s_out
        shift = np.clip(np.floor(np.log2(65535.0 / np.maximum(ratio, 1e-30))), 1, g.MAXSH).astype(np.int64)   # per channel
        mult = np.clip(np.round(ratio * 2.0 ** shift), 0, 65535).astype(np.int64)
        last = i == len(layers) - 1
        qs.append(g.LayerQ(w.shape[1] if not L['dw'] else w.shape[0], cout, L['k'], H, W, wq, bq, mult, shift,
                           dw=L['dw'], act=L['act'], sat16=(last and final_sat16), lut=lut))
        s_in = (s_post if L['act'] else None)
    return qs


def build_program(head, st, ref_feats, nc, FB=5, lanes=16):
    """Weights/params/LUTs from calibration stats `st`; ref_feats = any image (only provides shapes + initial inputs)."""
    inputs, box_chains, cls_chains = [], [], []
    for s in range(3):
        C, H, W = ref_feats[s].shape
        inputs.append(quantize_input(ref_feats[s], st['s_in'][s]))
        box_chains.append(quantize_chain(head['box'][s], st['box'][s], H, W, st['s_in'][s], 2.0 ** -FB, True))
        cls_chains.append(quantize_chain(head['cls'][s], st['cls'][s], H, W, st['s_in'][s], st['s_cls'], False))
    return g.assemble(inputs, box_chains, cls_chains, nc, FB, STRIDE_SHIFT, g.sigmoid_lut(st['s_cls']), L=lanes)


def quantize_input(f, s_in):
    return np.clip(np.round(f / s_in), -128, 127).astype(np.int64)


def run_int(P, st, feats):
    """Run one image (float neck features) through the int8 head program (golden == RTL). Returns act, box."""
    act0 = g.set_inputs(P, [quantize_input(feats[s], st['s_in'][s]) for s in range(3)])
    return g.golden_layers(P, act0)


def quantize_head(head, feats, nc, FB=5, pct=100.0, lanes=16):
    """Single-image convenience wrapper (calibrate on the image itself). Returns (Program, info)."""
    st = calibrate(head, [feats], pct)
    P = build_program(head, st, feats, nc, FB, lanes)
    info = dict(s_cls=st['s_cls'], st=st,
                float_box=[fchain(head['box'][s], feats[s])[-1][1] for s in range(3)],
                float_cls=[fchain(head['cls'][s], feats[s])[-1][1] for s in range(3)])
    return P, info


def int_outputs(P, info, act, box, FB):
    """Dequantised cls logits / box distances of the int8 head, per scale, in the float head's units."""
    cls, bx = [], []
    for s, (cb, bb, H, W, sh, cwn) in enumerate(P.scales):
        cls.append(g.read_tensor(act, cb, P.nc, H, W, P.L) * info['s_cls'])
        bx.append(box[bb:bb + H * W * 4].reshape(H * W, 4).T.reshape(4, H, W) / 2.0 ** FB)
    return cls, bx


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


def run_rtl(P, workdir, max_det=300):
    """Write hex files, compile + run the Verilog testbench, return rows (int) and cycle count."""
    here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
    os.makedirs(workdir, exist_ok=True)
    meta = g.write_files(P, workdir)
    lg = lambda n, lo: max(lo, math.ceil(math.log2(n + 1)))
    params = dict(LANES=P.L, LOGL=int(math.log2(P.L)), AAW=lg(meta['act_words'], 6), WAW=lg(meta['n_wgt'], 6),
                  PAW=lg(meta['n_par'], 6), LAW=lg(meta['n_lut'] + 256, 8), PRW=lg(meta['n_prog'], 6),
                  BAW=lg(meta['box_words'], 6), K=max_det, OUTAW=lg(6 * max_det + 6, 8))
    srcs = [f'{root}/rtl/{f}' for f in ('ram_tdp.v', 'topk_sorter.v', 'conv_engine.v', 'det_post.v', 'yolo26_head_core.v')] + [f'{root}/tb/tb_head.v']
    subprocess.run(['iverilog', '-g2012', '-o', f'{workdir}/tb.vvp'] + [f'-Ptb_head.{k}={v}' for k, v in params.items()] + srcs, check=True)
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
