"""
Bit-exact integer model of the YOLO26 (one2one) detect head, Kria/v2 architecture.

Arithmetic contract (RTL must match exactly):
  conv   : acc(int32) = sum(act_int8 * w_int8)                       zero padding, stride 1
  requant: q = clamp( ((acc + bias) * mult + (1 << (sh-1))) >> sh )  per OUTPUT CHANNEL mult (u16), sh (1..40)
           clamp to int8, or int16 if sat16            (arithmetic shift, int64)
  act    : if act: q = lut[q + 128]                    (SiLU as a 256-entry LUT, int8 -> int8)
Memory layout (v2): activations are HWC, channel-padded to a multiple of LANES, one RAM word = LANES int8 bytes
  (byte l = channel chunk*LANES + l).  word address = base + (y*W + x)*Cw + chunk,  Cw = ceil(C/LANES).
  box outputs (sat16 layers, 4 channels) live in a separate 16-bit RAM:  address = base + (y*W + x)*4 + c.
"""
import json
import numpy as np

HDR_WORDS = 32
LAYER_WORDS = 8
SCALE_WORDS = 6
MAXSH = 40


def cdiv(a, b):
    return -(-a // b)


# ----------------------------------------------------------------------------- layers
class LayerQ:
    """dense: w (cout,cin,k,k); depthwise: w (c,k,k). mult/shift are per output channel."""

    def __init__(self, cin, cout, k, H, W, w, bias, mult, shift,
                 dw=False, act=False, sat16=False, lut=None):
        self.cin, self.cout, self.k, self.H, self.W = cin, cout, k, H, W
        self.w = np.asarray(w, dtype=np.int64)
        self.bias = np.asarray(bias, dtype=np.int64)
        self.mult = np.asarray(mult, dtype=np.int64)
        self.shift = np.broadcast_to(np.asarray(shift, dtype=np.int64), (cout,)).copy()
        self.dw, self.act, self.sat16 = dw, act, sat16
        self.lut = None if lut is None else np.asarray(lut, dtype=np.int64)
        self.check()

    def check(self):
        if self.dw:
            assert self.cin == self.cout and self.w.shape == (self.cin, self.k, self.k)
        else:
            assert self.w.shape == (self.cout, self.cin, self.k, self.k)
        assert (self.mult >= 0).all() and (self.mult <= 65535).all(), 'mult must fit u16'
        assert (self.shift >= 1).all() and (self.shift <= MAXSH).all(), 'shift out of range'
        assert (np.abs(self.bias) < 2 ** 31).all()
        if self.sat16:
            assert self.cout == 4 and not self.act
        if self.act:
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
    sh = L.shift[:, None, None]
    t = (t + (np.int64(1) << (sh - 1))) >> sh
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


# ----------------------------------------------------------------------------- HWC memory helpers
def read_tensor(mem, base, C, H, W, L):
    cw = cdiv(C, L)
    blk = mem[base:base + H * W * cw].reshape(H * W, cw * L)[:, :C]
    return blk.T.reshape(C, H, W).copy()


def write_tensor(mem, base, t, L):
    C, H, W = t.shape
    cw = cdiv(C, L)
    buf = np.zeros((H * W, cw * L), dtype=np.int64)
    buf[:, :C] = t.reshape(C, H * W).T
    mem[base:base + H * W * cw] = buf.reshape(H * W * cw, L)


# ----------------------------------------------------------------------------- program assembly
class Program:
    pass


def assemble(inputs, box_chains, cls_chains, nc, FB, stride_shifts, sig_lut, L=16):
    assert FB >= 2 and L >= 4 and (L & (L - 1)) == 0
    P = Program()
    P.L, P.nc, P.FB, P.sh = L, nc, FB, list(stride_shifts)
    HW = [(a.shape[1], a.shape[2]) for a in inputs]
    cw = lambda c: cdiv(c, L)

    # ---- act RAM (words of L bytes) and box RAM (16-bit words) maps
    cur = 0
    in_base = []
    for a in inputs:
        in_base.append(cur); cur += a.shape[1] * a.shape[2] * cw(a.shape[0])
    cls_base = []
    for s, (H, W) in enumerate(HW):
        cls_base.append(cur); cur += H * W * cw(nc)
    box_base, bcur = [], 0
    for s, (H, W) in enumerate(HW):
        box_base.append(bcur); bcur += H * W * 4
    scratch = 0
    for s in range(3):
        for ch in (box_chains[s], cls_chains[s]):
            for Ly in ch[:-1]:
                scratch = max(scratch, Ly.H * Ly.W * cw(Ly.cout))
    bufA, bufB = cur, cur + scratch
    cur += 2 * scratch
    P.act_words, P.box_words = cur, bcur

    layers = []
    for s in range(3):
        for chain, final in ((box_chains[s], box_base[s]), (cls_chains[s], cls_base[s])):
            src = in_base[s]
            for i, Ly in enumerate(chain):
                Ly.in_base = src
                Ly.out_base = final if i == len(chain) - 1 else (bufA, bufB)[i % 2]
                src = Ly.out_base
                layers.append(Ly)
    P.layers = layers

    # ---- weights (words of L int8), params (64-bit), LUTs
    wwords, pwords, lwords = [], [], []
    prog = [0] * HDR_WORDS
    prog[0], prog[1], prog[2], prog[3] = len(layers), nc, FB, L
    for s in range(3):
        o = 4 + SCALE_WORDS * s
        prog[o:o + SCALE_WORDS] = [cls_base[s], box_base[s], HW[s][0], HW[s][1], P.sh[s], cw(nc)]
    for Ly in layers:
        k = Ly.k
        Ly.w_base = len(wwords)
        if Ly.dw:
            cwl = cw(Ly.cin)
            wp = np.zeros((cwl, k, k, L), dtype=np.int64)
            for c in range(Ly.cin):
                wp[c // L, :, :, c % L] = Ly.w[c]
            wwords.extend(wp.reshape(-1, L).tolist())
            npar = cwl * L
        else:
            cwl = cw(Ly.cin)
            wp = np.zeros((Ly.cout, k, k, cwl * L), dtype=np.int64)
            wp[:, :, :, :Ly.cin] = np.transpose(Ly.w, (0, 2, 3, 1))
            wwords.extend(wp.reshape(-1, L).tolist())
            npar = Ly.cout
        Ly.par_base = len(pwords)
        for c in range(npar):
            if c < Ly.cout:
                b, m, sh = int(Ly.bias[c]), int(Ly.mult[c]), int(Ly.shift[c])
            else:
                b, m, sh = 0, 0, 1
            pwords.append((b & 0xFFFFFFFF) | (m << 32) | (sh << 48))
        if Ly.act:
            Ly.lut_base = len(lwords); lwords.extend(Ly.lut.tolist())
        else:
            Ly.lut_base = 0
        flags = int(Ly.dw) | (int(Ly.act) << 1) | (int(Ly.sat16) << 2) | (k << 3)
        prog.extend([Ly.in_base, Ly.out_base, Ly.w_base, Ly.par_base, Ly.lut_base,
                     (Ly.H << 16) | Ly.W, (cwl << 16) | Ly.cout, flags])
    P.prog, P.wgt, P.par, P.lut, P.sig = prog, wwords, pwords, lwords, sig_lut.tolist()

    P.act_init = np.zeros((P.act_words, L), dtype=np.int64)
    for s, a in enumerate(inputs):
        write_tensor(P.act_init, in_base[s], a, L)
    P.in_base = in_base
    P.n_input_words = sum(a.shape[1] * a.shape[2] * cw(a.shape[0]) for a in inputs)
    P.scales = [(cls_base[s], box_base[s], HW[s][0], HW[s][1], P.sh[s], cw(nc)) for s in range(3)]
    return P


def set_inputs(P, inputs):
    """Return an act-RAM image with new int8 input feature maps (same program, new image)."""
    act = P.act_init.copy()
    for s, a in enumerate(inputs):
        write_tensor(act, P.in_base[s], a, P.L)
    return act


def golden_layers(P, act_init=None):
    act = (P.act_init if act_init is None else act_init).copy()
    box = np.zeros(P.box_words, dtype=np.int64)
    for Ly in P.layers:
        x = read_tensor(act, Ly.in_base, Ly.cin, Ly.H, Ly.W, P.L)
        y = run_layer_np(x, Ly)
        if Ly.sat16:
            box[Ly.out_base:Ly.out_base + Ly.H * Ly.W * 4] = y.reshape(4, -1).T.reshape(-1)
        else:
            write_tensor(act, Ly.out_base, y, P.L)
    return act, box


# ----------------------------------------------------------------------------- post-processing
def to_i16(v):
    return int(np.clip(v, -32768, 32767))


def post(act, box, P, max_det):
    nc, FB, L = P.nc, P.FB, P.L
    cells, L_all = [], []
    for s, (cb, bb, H, W, sh, cwn) in enumerate(P.scales):
        lg = read_tensor(act, cb, nc, H, W, L)
        for y in range(H):
            for x in range(W):
                cells.append((s, y, x)); L_all.append(lg[:, y, x])
    total = len(cells)
    k = min(max_det, total)
    best = [int(l.max()) for l in L_all]
    order1 = sorted(range(total), key=lambda i: (-best[i], i))[:k]
    flat = [(int(L_all[order1[j]][c]), j * nc + c) for j in range(len(order1)) for c in range(nc)]
    order2 = sorted(flat, key=lambda t: (-t[0], t[1]))[:k]
    rows = []
    for v, f in order2:
        j, c = divmod(f, nc)
        s, y, x = cells[order1[j]]
        cb, bb, H, W, sh, cwn = P.scales[s]
        d = [int(box[bb + (y * W + x) * 4 + ch]) for ch in range(4)]
        ax = ((2 * x + 1) << (FB - 1)); ay = ((2 * y + 1) << (FB - 1))
        corners = [ax - d[0], ay - d[1], ax + d[2], ay + d[3]]
        pix = [to_i16((cv << sh) >> (FB - 2)) for cv in corners]
        rows.append(pix + [int(P.sig[v + 128]), c])
    return rows


# ----------------------------------------------------------------------------- file writers
def pack_word(lanes):
    v = 0
    for i, b in enumerate(lanes):
        v |= (int(b) & 0xFF) << (8 * i)
    return v


def write_hex(path, ints, bits):
    nd = bits // 4
    mask = (1 << bits) - 1
    with open(path, 'w') as f:
        for v in ints:
            f.write(format(int(v) & mask, '0%dx' % nd) + '\n')


def write_files(P, outdir, rows=None, act_golden=None, box_golden=None):
    import os
    os.makedirs(outdir, exist_ok=True)
    L = P.L
    write_hex(f'{outdir}/prog.hex', P.prog, 32)
    write_hex(f'{outdir}/wgt.hex', [pack_word(w) for w in P.wgt], 8 * L)
    write_hex(f'{outdir}/par.hex', P.par, 64)
    write_hex(f'{outdir}/lut.hex', P.lut if P.lut else [0], 32)
    write_hex(f'{outdir}/sig.hex', P.sig, 32)
    write_hex(f'{outdir}/act_init.hex', [pack_word(w) for w in P.act_init[:P.n_input_words]], 8 * L)
    meta = dict(L=L, act_words=int(P.act_words), box_words=int(P.box_words), nc=P.nc, FB=P.FB,
                n_layers=len(P.layers), n_prog=len(P.prog), n_wgt=len(P.wgt), n_par=len(P.par),
                n_lut=len(P.lut), n_input_words=int(P.n_input_words),
                scales=[list(map(int, s)) for s in P.scales])
    json.dump(meta, open(f'{outdir}/meta.json', 'w'), indent=1)
    if act_golden is not None:
        write_hex(f'{outdir}/act_golden.hex', [pack_word(w) for w in act_golden[:P.act_words]], 8 * L)
    if box_golden is not None:
        write_hex(f'{outdir}/box_golden.hex', box_golden, 16)
    if rows is not None:
        with open(f'{outdir}/rows_golden.txt', 'w') as f:
            for r in rows:
                f.write(' '.join(str(int(v)) for v in r) + '\n')
    return meta
