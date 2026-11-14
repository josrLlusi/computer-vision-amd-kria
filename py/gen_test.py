"""Generate a small random-but-sane quantized YOLO26 head test + bit-exact golden outputs.
usage: python gen_test.py OUTDIR [--seed N] [--grid G] [--ch c0 c1 c2] [--nc N] [--maxdet K] [--fb F]
"""
import argparse, json, numpy as np
import golden as g

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--grid', type=int, default=8)          # P3 grid size; P4=grid/2, P5=grid/4
ap.add_argument('--ch', type=int, nargs=3, default=[8, 12, 16])
ap.add_argument('--nc', type=int, default=5)
ap.add_argument('--maxdet', type=int, default=16)
ap.add_argument('--fb', type=int, default=5)
ap.add_argument('--lanes', type=int, default=16)
ap.add_argument('--hfrac', type=float, default=1.0)   # image height / width (0.75 = 640x480)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

def rand_layer(x, cin, cout, k, H, W, dw, act, sat16, tgt):
    w = rng.integers(-6, 7, (cin, k, k) if dw else (cout, cin, k, k))
    L = g.LayerQ(cin, cout, k, H, W, w, np.zeros(cout, int), np.ones(cout, int), 1, dw=dw, act=act,
                 sat16=sat16, lut=np.zeros(256, int) if act else None)
    acc = g.np_conv_acc(x, L)
    std = acc.reshape(cout, -1).std(axis=1) + 1.0
    L.shift = (12 + rng.integers(0, 4, cout)).astype(np.int64)          # per-channel shift
    L.mult = np.clip(np.round((1 << L.shift) * tgt / std), 1, 65535).astype(np.int64)
    L.bias = rng.integers(-40, 41, cout).astype(np.int64)
    L.check()
    if act:
        s_pre = rng.uniform(0.05, 0.2)
        L.lut = g.silu_lut(s_pre, rng.uniform(0.05, 0.2))
    return L, g.run_layer_np(x, L)

GW = [a.grid, a.grid // 2, a.grid // 4]
h0 = int(round(a.grid * a.hfrac)); GH = [h0, h0 // 2, h0 // 4]
nc, fb = a.nc, a.fb
c2 = max(16, a.ch[0] // 4, 4)
c3 = max(a.ch[0], min(nc, 100))
inputs, box_chains, cls_chains = [], [], []
for s in range(3):
    H, W = GH[s], GW[s]; cin = a.ch[s]
    x0 = rng.integers(-70, 71, (cin, H, W))
    inputs.append(x0)
    # box chain: conv3 -> conv3 -> conv1(4, int16)
    x = x0; ch = []
    for (ci_, co_, k_, act, sat) in [(cin, c2, 3, True, False), (c2, c2, 3, True, False), (c2, 4, 1, False, True)]:
        L, x = rand_layer(x, ci_, co_, k_, H, W, False, act, sat, 80.0 if sat else 30.0); ch.append(L)
    box_chains.append(ch)
    # cls chain: dw3 -> pw1 -> dw3 -> pw1 -> pw1(nc)
    x = x0; ch = []
    for (ci_, co_, k_, dw, act, sat) in [(cin, cin, 3, True, True, False), (cin, c3, 1, False, True, False),
                                         (c3, c3, 3, True, True, False), (c3, c3, 1, False, True, False),
                                         (c3, nc, 1, False, False, False)]:
        L, x = rand_layer(x, ci_, co_, k_, H, W, dw, act, sat, 40.0 if co_ == nc and not dw else 30.0); ch.append(L)
    cls_chains.append(ch)

P = g.assemble(inputs, box_chains, cls_chains, nc, fb, [3, 4, 5], g.sigmoid_lut(0.08), L=a.lanes)
act, box = g.golden_layers(P)
rows = g.post(act, box, P, a.maxdet)
meta = g.write_files(P, a.out, rows=rows, act_golden=act, box_golden=box)
json.dump(dict(meta, maxdet=a.maxdet, n_rows=len(rows)), open(f'{a.out}/meta.json', 'w'), indent=1)
print(f'{a.out}: layers={len(P.layers)} act_words={P.act_words} wgt={len(P.wgt)} rows={len(rows)}')
