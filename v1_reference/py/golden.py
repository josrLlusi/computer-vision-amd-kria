"""
Bit-exact integer model of the YOLO26 (one2one) detect head as implemented in rtl/.

Arithmetic contract (RTL must match this exactly):
  conv   : acc(int32) = sum(act_int8 * w_int8)           zero padding, stride 1
  requant: q = clamp( ((acc + bias) * mult + (1 << (shift-1))) >> shift )   (arithmetic shift, int64)
           clamp to int8, or int16 if sat16
  act    : if act: q = lut[q + 128]                       (SiLU as a 256-entry LUT, int8 -> int8)
  post   : see post()  (two-stage top-k on int8 logits, ties -> lower index, then box decode)
"""
import json
import numpy as np

HDR_WORDS = 32          # header words at start of prog.hex
LAYER_WORDS = 8         # descriptor words per layer


# ----------------------------------------------------------------------------- layers
class LayerQ:
    """One quantized layer. dense: w (cout,cin,k,k); depthwise: w (c,k,k)."""

    def __init__(self, cin, cout, k, H, W, w, bias, mult, shift,
                 dw=False, act=False, sat16=False, lut=None):
        self.cin, self.cout, self.k, self.H, self.W = cin, cout, k, H, W
        self.w = np.asarray(w, dtype=np.int64)
        self.bias = np.asarray(bias, dtype=np.int64)
        self.mult = np.asarray(mult, dtype=np.int64)
        self.shift = int(shift)
        self.dw, self.act, self.sat16 = dw, act, sat16
        self.lut = None if lut is None else np.asarray(lut, dtype=np.int64)
        if dw:
            assert cin == cout and self.w.shape == (cin, k, k)
        else:
            assert self.w.shape == (cout, cin, k, k)
        assert self.shift >= 1
        assert (self.mult >= 0).all() and (self.mult < 2 ** 31).all()
        if act:
            assert self.lut is not None and len(self.lut) == 256


def np_conv_acc(x, L):
    pad = L.k // 2
    H, W = L.H, L.W
    xp = np.pad(x.astype(np.int64), ((0, 0), (pad, pad), (pad, pad)))
    acc = np.zeros((L.cout, H, W), dtype=np.int64)
    for ky in range(L.k):
        for kx in range(L.k):
            win = xp[:, ky:ky + H, kx:kx + W]
            if L.dw:
                acc += win * L.w[:, ky, kx][:, None, None]
            else:
                acc += np.einsum('oc,chw->ohw', L.w[:, :, ky, kx], win)
    return acc


def clamp_bits(v, bits):
    return np.clip(v, -(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def np_requant(acc, L):
    t = (acc + L.bias[:, None, None]) * L.mult[:, None, None]
    t = (t + (1 << (L.shift - 1))) >> L.shift
    q = clamp_bits(t, 16 if L.sat16 else 8)
    if L.act:
        q = L.lut[q + 128]
    return q


def run_layer_np(x, L):
    return np_requant(np_conv_acc(x, L), L)


# ----------------------------------------------------------------------------- LUTs
def silu_lut(s_pre, s_post):
    q = np.arange(-128, 128)
    x = q * s_pre
    y = x / (1.0 + np.exp(-x))
    return np.clip(np.round(y / s_post), -128, 127).astype(np.int64)


def sigmoid_lut(s_cls):
    q = np.arange(-128, 128)
    return np.clip(np.round(65535.0 / (1.0 + np.exp(-q * s_cls))), 0, 65535).astype(np.int64)


# ----------------------------------------------------------------------------- program assembly
class Program:
    pass


def assemble(inputs, box_chains, cls_chains, nc, FB, stride_shifts, sig_lut):
    """Lay out memories and build the program for 3 scales.

    inputs        : 3 int arrays (Cin, H, W), int8 range  (neck outputs P3,P4,P5)
    box_chains[s] : list of LayerQ (last one sat16, cout=4)
    cls_chains[s] : list of LayerQ (last one cout=nc)
    """
    assert FB >= 2
    P = Program()
    P.nc, P.FB, P.sh = nc, FB, list(stride_shifts)
    HW = [(a.shape[1], a.shape[2]) for a in inputs]

    # ---- activation memory map
    cur = 0
    in_base = []
    for a in inputs:
        in_base.append(cur)
        cur += a.size
    cls_base, box_base = [], []
    for s, (H, W) in enumerate(HW):
        cls_base.append(cur); cur += nc * H * W
        box_base.append(cur); cur += 4 * H * W
    scratch = 0
    for s in range(3):
        for ch in (box_chains[s], cls_chains[s]):
            for L in ch[:-1]:
                scratch = max(scratch, L.cout * L.H * L.W)
    bufA, bufB = cur, cur + scratch
    cur += 2 * scratch
    P.act_words = cur

    # ---- assign bases per layer, build ordered list
    layers = []
    for s in range(3):
        for chain, final in ((box_chains[s], box_base[s]), (cls_chains[s], cls_base[s])):
            cin_base = in_base[s]
            for i, L in enumerate(chain):
                L.in_base = cin_base
                L.out_base = final if i == len(chain) - 1 else (bufA, bufB)[i % 2]
                cin_base = L.out_base
                layers.append(L)
    P.layers = layers

    # ---- weights / params / luts
    w_words, p_words, l_words = [], [], []
    prog = [0] * HDR_WORDS
    prog[0] = len(layers)
    prog[1] = nc
    prog[2] = FB
    for s in range(3):
        o = 4 + 5 * s
        prog[o:o + 5] = [cls_base[s], box_base[s], HW[s][0], HW[s][1], P.sh[s]]
    for L in layers:
        L.w_base = len(w_words)
        w_words.extend(L.w.reshape(-1).tolist())
        L.par_base = len(p_words)
        for co in range(L.cout):
            p_words.extend([int(L.bias[co]), int(L.mult[co])])
        if L.act:
            L.lut_base = len(l_words)
            l_words.extend(L.lut.tolist())
        else:
            L.lut_base = 0
        flags = (int(L.dw) | (int(L.act) << 1) | (int(L.sat16) << 2) | (L.k << 3) | (L.shift << 8))
        prog.extend([L.in_base, L.out_base, L.w_base, L.par_base, L.lut_base,
                     (L.H << 16) | L.W, (L.cin << 16) | L.cout, flags])
    P.prog, P.wgt, P.par, P.lut = prog, w_words, p_words, l_words
    P.sig = sig_lut.tolist()

    init = np.zeros(P.act_words, dtype=np.int64)
    for s, a in enumerate(inputs):
        init[in_base[s]:in_base[s] + a.size] = a.reshape(-1)
    P.act_init = init
    P.scales = [(cls_base[s], box_base[s], HW[s][0], HW[s][1], P.sh[s]) for s in range(3)]
    P.n_input_words = sum(a.size for a in inputs)
    return P


def golden_layers(P):
    act = P.act_init.copy()
    for L in P.layers:
        n_in = L.cin * L.H * L.W
        x = act[L.in_base:L.in_base + n_in].reshape(L.cin, L.H, L.W)
        y = run_layer_np(x, L)
        act[L.out_base:L.out_base + y.size] = y.reshape(-1)
    return act


# ----------------------------------------------------------------------------- post-processing
def to_i16(v):
    return int(np.clip(v, -32768, 32767))


def post(act, P, max_det):
    nc, FB = P.nc, P.FB
    cells = []   # (s, y, x) in global (flatten) order
    for s, (cb, bb, H, W, sh) in enumerate(P.scales):
        for y in range(H):
            for x in range(W):
                cells.append((s, y, x))
    total = len(cells)
    k = min(max_det, total)

    def logits(cell):
        s, y, x = cell
        cb, bb, H, W, sh = P.scales[s]
        return np.array([act[cb + c * H * W + y * W + x] for c in range(nc)], dtype=np.int64)

    L_all = [logits(c) for c in cells]
    best = [int(l.max()) for l in L_all]
    # stage 1: best class score per cell, keep top-k cells (ties -> lower global index)
    order1 = sorted(range(total), key=lambda i: (-best[i], i))[:k]
    # stage 2: flatten (rank j, class c), keep top-k (ties -> lower flat index)
    flat = [(int(L_all[order1[j]][c]), j * nc + c) for j in range(len(order1)) for c in range(nc)]
    order2 = sorted(flat, key=lambda t: (-t[0], t[1]))[:k]

    rows = []
    for v, f in order2:
        j, c = divmod(f, nc)
        s, y, x = cells[order1[j]]
        cb, bb, H, W, sh = P.scales[s]
        d = [int(act[bb + ch * H * W + y * W + x]) for ch in range(4)]
        ax = ((2 * x + 1) << (FB - 1))
        ay = ((2 * y + 1) << (FB - 1))
        corners = [ax - d[0], ay - d[1], ax + d[2], ay + d[3]]
        pix = [to_i16((cv << sh) >> (FB - 2)) for cv in corners]
        score = int(P.sig[v + 128])
        rows.append(pix + [score, c])
    return rows


# ----------------------------------------------------------------------------- file writers
def write_hex(path, arr, bits):
    mask = (1 << bits) - 1
    nd = bits // 4
    with open(path, 'w') as f:
        for v in arr:
            f.write(format(int(v) & mask, '0%dx' % nd) + '\n')


def write_files(P, outdir, rows=None, act_golden=None):
    import os
    os.makedirs(outdir, exist_ok=True)
    write_hex(f'{outdir}/prog.hex', P.prog, 32)
    write_hex(f'{outdir}/wgt.hex', P.wgt, 8)
    write_hex(f'{outdir}/par.hex', P.par, 32)
    write_hex(f'{outdir}/lut.hex', P.lut if P.lut else [0], 8)
    write_hex(f'{outdir}/sig.hex', P.sig, 16)
    write_hex(f'{outdir}/act_init.hex', P.act_init[:P.n_input_words], 16)
    meta = dict(act_words=int(P.act_words), nc=P.nc, FB=P.FB, n_layers=len(P.layers),
                n_prog=len(P.prog), n_wgt=len(P.wgt), n_par=len(P.par), n_lut=len(P.lut),
                scales=[list(map(int, s)) for s in P.scales])
    json.dump(meta, open(f'{outdir}/meta.json', 'w'), indent=1)
    if act_golden is not None:
        write_hex(f'{outdir}/act_golden.hex', act_golden[:P.act_words], 16)
    if rows is not None:
        with open(f'{outdir}/rows_golden.txt', 'w') as f:
            for r in rows:
                f.write(' '.join(str(int(v)) for v in r) + '\n')
    return meta
