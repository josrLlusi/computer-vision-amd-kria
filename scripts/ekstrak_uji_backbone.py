"""
ekstrak_uji_backbone.py
----------------------------------------------------------------------------
Data uji + golden BIT-EXACT untuk:
  --mode conv   : satu conv_kxk generik (uji 1x1 / 3x3 s1 / 3x3 s2)
  --mode c3k2   : blok C3k2 layer 2 atau 4 yolo26n            (c3k2.sv)
  --mode chain  : rangkaian backbone layer 0..5 yolo26n       (backbone_l0_l5.sv)

Model integer meniru RTL persis (Q16.16, akumulator 64-bit, acc[47:16], LUT SiLU,
residual wrap 32-bit), sehingga RTL yang benar harus cocok 0 LSB.

Contoh (root repo, venv aktif):
  python scripts/ekstrak_uji_backbone.py --mode c3k2 --layer 2 --size 16                 # cepat, acak
  python scripts/ekstrak_uji_backbone.py --mode c3k2 --layer 2 --yolo yolo26n.pt --image assets/gambar_tes.jpeg
  python scripts/ekstrak_uji_backbone.py --mode chain --size 64 --yolo yolo26n.pt --image assets/gambar_tes.jpeg
  python scripts/ekstrak_uji_backbone.py --mode chain --yolo yolo26n.pt --image assets/gambar_tes.jpeg   # 640 penuh (lama)
  python scripts/ekstrak_uji_backbone.py --mode conv --kernel 3 --stride 1 --in-ch 8 --out-ch 4 --size 10
Lalu: vsim -c -do sim/run_sim_backbone.do   dan   python scripts/validasi_backbone.py
"""
import argparse
import numpy as np

Q_FRAC = 16
Q_SCALE = 1 << Q_FRAC
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
SIG_N = 2049

# Struktur Conv yolo26n layer 0..5: nama -> (OUT, IN, K, STRIDE)
STRUCT = {
    "L0": (16, 3, 3, 2), "L1": (32, 16, 3, 2),
    "L2.cv1": (32, 32, 1, 1), "L2.m0.cv1": (8, 16, 3, 1), "L2.m0.cv2": (16, 8, 3, 1), "L2.cv2": (64, 48, 1, 1),
    "L3": (64, 64, 3, 2),
    "L4.cv1": (64, 64, 1, 1), "L4.m0.cv1": (16, 32, 3, 1), "L4.m0.cv2": (32, 16, 3, 1), "L4.cv2": (128, 96, 1, 1),
    "L5": (128, 128, 3, 2),
}
# PAR (IC, OC) C3k2 standalone = sama dengan backbone_l0_l5.sv
C3K2_PAR = {2: dict(cv1=(8, 2), m1=(4, 1), m2=(4, 1), cv2=(16, 4)),
            4: dict(cv1=(4, 4), m1=(2, 1), m2=(2, 1), cv2=(8, 8))}
C3K2_DIM = {2: (32, 64, 16, 8, 160), 4: (64, 128, 32, 16, 80)}   # C1, C2, C, CB, ukuran


# ---------------------------------------------------------------------------
# Model integer bit-exact & referensi float
# ---------------------------------------------------------------------------
def to_q16(a, name=""):
    q = np.round(np.asarray(a, dtype=np.float64) * Q_SCALE)
    n = int(np.count_nonzero((q < I32_MIN) | (q > I32_MAX)))
    if n:
        print(f"  PERINGATAN: {n} nilai {name} di luar rentang Q16.16 -> disaturasi")
    return np.clip(q, I32_MIN, I32_MAX).astype(np.int64)


def wrap32(v):
    v = np.asarray(v, dtype=np.int64) & 0xFFFFFFFF
    return np.where(v >= (1 << 31), v - (1 << 32), v)


def sigmoid_lut():
    xs = -16 + np.arange(SIG_N) / 64.0
    return np.round(Q_SCALE / (1.0 + np.exp(-xs))).astype(np.int64)


LUT = sigmoid_lut()


def silu_fixed(r):
    r = np.asarray(r, dtype=np.int64)
    hi, lo = 16 << 16, -(16 << 16)
    mid = (r >= lo) & (r < hi)
    xo = np.where(mid, r + hi, 0)
    idx, frac = xo >> 10, xo & 1023
    s0, s1 = LUT[idx], LUT[np.minimum(idx + 1, SIG_N - 1)]
    sig = s0 + (((s1 - s0) * frac) >> 10)
    return np.where(r >= hi, r, np.where(mid, wrap32((r * sig) >> Q_FRAC), 0))


def _patches(x, k, s):
    """x [C,H,W] -> iterator (ky, kx, patch [C,OH,OW]) dengan padding k//2."""
    p = k // 2
    _, H, W = x.shape
    OH, OW = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    for ky in range(k):
        for kx in range(k):
            yield ky, kx, xp[:, ky:ky + s * (OH - 1) + 1:s, kx:kx + s * (OW - 1) + 1:s], OH, OW


def conv_fixed(x_q, w_q, b_q, k, s, silu=True):
    acc = None
    with np.errstate(over="ignore"):
        for ky, kx, patch, OH, OW in _patches(x_q, k, s):
            if acc is None:
                acc = np.broadcast_to((b_q << Q_FRAC)[:, None, None], (w_q.shape[0], OH, OW)).astype(np.int64).copy()
            acc += np.einsum("chw,oc->ohw", patch, w_q[:, :, ky, kx])
    r = wrap32(acc >> Q_FRAC)
    return silu_fixed(r) if silu else r


def conv_float(x, w, b, k, s, silu=True):
    y = None
    for ky, kx, patch, OH, OW in _patches(x.astype(np.float64), k, s):
        if y is None:
            y = np.broadcast_to(b[:, None, None], (w.shape[0], OH, OW)).astype(np.float64).copy()
        y += np.einsum("chw,oc->ohw", patch, w[:, :, ky, kx])
    return y / (1 + np.exp(-y)) if silu else y


def c3k2_run(x, P, pre, fixed):
    cv = (lambda x_, n, k, s: conv_fixed(x_, *P[n], k, s)) if fixed else \
         (lambda x_, n, k, s: conv_float(x_, *P[n], k, s))
    a = cv(x, f"{pre}.cv1", 1, 1)
    C = a.shape[0] // 2
    y0, y1 = a[:C], a[C:]
    u = cv(cv(y1, f"{pre}.m0.cv1", 3, 1), f"{pre}.m0.cv2", 3, 1)
    z = wrap32(y1 + u) if fixed else y1 + u
    return cv(np.concatenate([y0, y1, z]), f"{pre}.cv2", 1, 1)


def chain_run(x, P, fixed):
    cv = (lambda x_, n: conv_fixed(x_, *P[n], 3, 2)) if fixed else (lambda x_, n: conv_float(x_, *P[n], 3, 2))
    l4 = c3k2_run(cv(c3k2_run(cv(cv(x, "L0"), "L1"), P, "L2", fixed), "L3"), P, "L4", fixed)
    return l4, cv(l4, "L5")


def fold_bn(w, gamma, beta, mean, var, eps):
    s = gamma / np.sqrt(var + eps)
    return w * s[:, None, None, None], beta - mean * s


# ---------------------------------------------------------------------------
# Sumber bobot & input
# ---------------------------------------------------------------------------
def random_params(names, seed):
    rng = np.random.default_rng(seed)
    P = {}
    for n in names:
        o, i, k, _ = STRUCT[n] if n in STRUCT else n
        a = np.sqrt(3.0 / (i * k * k))
        P[n] = (rng.uniform(-a, a, (o, i, k, k)), rng.uniform(-0.5, 0.5, o))
    return P


def load_yolo(path):
    """Kembalikan (net, P) dengan BN dilebur. Butuh torch + ultralytics."""
    from ultralytics import YOLO
    net = YOLO(path).model.float().eval()

    def fold(m):
        w = m.conv.weight.detach().double().numpy()
        if hasattr(m, "bn"):
            bn = m.bn
            return fold_bn(w, bn.weight.detach().double().numpy(), bn.bias.detach().double().numpy(),
                           bn.running_mean.double().numpy(), bn.running_var.double().numpy(), bn.eps)
        b = m.conv.bias
        return w, (b.detach().double().numpy() if b is not None else np.zeros(w.shape[0]))

    P = {}
    for i in (0, 1, 3, 5):
        P[f"L{i}"] = fold(net.model[i])
    for i in (2, 4):
        L = net.model[i]
        if len(L.m) != 1 or type(L.m[0]).__name__ != "Bottleneck" or not L.m[0].add:
            raise SystemExit(f"Layer {i}: struktur C3k2 tidak sesuai (harus 1 Bottleneck dengan shortcut)")
        P[f"L{i}.cv1"], P[f"L{i}.cv2"] = fold(L.cv1), fold(L.cv2)
        P[f"L{i}.m0.cv1"], P[f"L{i}.m0.cv2"] = fold(L.m[0].cv1), fold(L.m[0].cv2)
    for n, (o, i_, k, _) in STRUCT.items():
        if P[n][0].shape != (o, i_, k, k):
            raise SystemExit(f"Bentuk bobot {n} = {P[n][0].shape}, diharapkan {(o, i_, k, k)} (bukan yolo26n?)")
    print(f"  Bobot  : {path} (BN dilebur, eps={net.model[0].bn.eps})")
    return net, P


def torch_forward(net, x, upto):
    """Jalankan net.model[0..upto] berurutan; kembalikan dict {indeks: keluaran [C,H,W]}."""
    import torch
    out, y = {}, torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        for i in range(upto + 1):
            y = net.model[i](y)
            out[i] = y[0].double().numpy()
    return out


def load_image(path, size):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    return (np.asarray(img, dtype=np.float64) / 255.0).transpose(2, 0, 1)


# ---------------------------------------------------------------------------
# Ekspor
# ---------------------------------------------------------------------------
def write_hex(values, fn, quiet=False):
    flat = np.asarray(values, dtype=np.int64).reshape(-1) & 0xFFFFFFFF
    with open(fn, "w") as f:
        f.write("\n".join(f"{int(v):08x}" for v in flat) + "\n")
    if not quiet:
        print(f"  Tersimpan: {fn:<22} ({flat.size} baris)")


def hwc(a):
    return np.ascontiguousarray(np.asarray(a).transpose(1, 2, 0))


def report(name, g_q, ref):
    g = g_q / Q_SCALE
    err = np.abs(g - ref)
    rel = np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-12)
    print(f"  {name:<10}: {g_q.shape[0]} ch x {g_q.shape[1]} x {g_q.shape[2]} | Q16.16 vs float: "
          f"err maks {err.max():.2e} ({err.max()*Q_SCALE:.0f} LSB), relatif {rel:.2e}")


def write_cfg(d):
    defaults = dict(MODE=0, IMG=16, IN_W=16, IN_H=16, IN_CH=3,
                    O1_W=8, O1_H=8, O1_CH=16, O2_W=1, O2_H=1, O2_CH=1,
                    K=3, S=2, PIC=1, POC=1, C1=32, C2=64, C=16, CB=8,
                    P1I=1, P1O=1, PM1I=1, PM1O=1, PM2I=1, PM2O=1, P2I=1, P2O=1)
    defaults.update(d)
    with open("tb_cfg.svh", "w") as f:
        f.write("// Dibangkitkan otomatis oleh ekstrak_uji_backbone.py - JANGAN diedit\n")
        f.write("// TB_MODE: 0 = conv_kxk, 1 = c3k2, 2 = backbone_l0_l5\n")
        for k, v in defaults.items():
            if k != "PREFIX":
                f.write(f"localparam int TB_{k:<6}= {v};\n")
        f.write(f'localparam TB_PREFIX = "{defaults.get("PREFIX", "L2")}";\n')
    print("  Tersimpan: tb_cfg.svh")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["conv", "c3k2", "chain"], default="c3k2")
    ap.add_argument("--layer", type=int, default=2, choices=[2, 4], help="layer C3k2 (mode c3k2)")
    ap.add_argument("--size", type=int, default=None, help="ukuran input blok (default: ukuran asli)")
    ap.add_argument("--yolo", type=str, default=None)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # khusus --mode conv
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--in-ch", type=int, default=8)
    ap.add_argument("--out-ch", type=int, default=4)
    ap.add_argument("--par-ic", type=int, default=1)
    ap.add_argument("--par-oc", type=int, default=1)
    args = ap.parse_args()

    print("=" * 66)
    print(f" Data uji backbone - mode {args.mode}" + (f" layer {args.layer}" if args.mode == "c3k2" else ""))
    print("=" * 66)
    rng = np.random.default_rng(args.seed)
    net = None

    # ---------------- mode conv ----------------
    if args.mode == "conv":
        k, s, ci, co, sz = args.kernel, args.stride, args.in_ch, args.out_ch, args.size or 12
        P = random_params([(co, ci, k, s)], args.seed)
        w, b = P[(co, ci, k, s)]
        x = rng.standard_normal((ci, sz, sz))
        x_q, w_q, b_q = to_q16(x), to_q16(w), to_q16(b)
        g = conv_fixed(x_q, w_q, b_q, k, s)
        report("keluaran", g, conv_float(x, w, b, k, s))
        write_hex(w_q, "U.w.hex"); write_hex(b_q, "U.b.hex")
        write_hex(hwc(x_q), "tb_input.hex"); write_hex(hwc(g), "gold_o1.hex")
        np.save("ref_o1.npy", hwc(conv_float(x, w, b, k, s)))
        write_hex(LUT, "sigmoid_lut.hex")
        write_cfg(dict(MODE=0, IN_W=sz, IN_H=sz, IN_CH=ci, O1_W=g.shape[2], O1_H=g.shape[1], O1_CH=co,
                       K=k, S=s, PIC=args.par_ic, POC=args.par_oc))
        return

    # ---------------- bobot ----------------
    if args.yolo:
        net, P = load_yolo(args.yolo)
    else:
        P = random_params(STRUCT.keys(), args.seed)
        print("  Bobot  : acak (bentuk = yolo26n)")
    Pq = {n: (to_q16(w, n), to_q16(b, n)) for n, (w, b) in P.items()}

    # ---------------- mode c3k2 ----------------
    if args.mode == "c3k2":
        L = args.layer
        C1, C2, C, CB, full = C3K2_DIM[L]
        sz = args.size or full
        if args.yolo and args.image:
            # Aktivasi ASLI: jalankan layer 0..L-1 pada citra 640, potong ke sz
            x = torch_forward(net, load_image(args.image, 640), L - 1)[L - 1][:, :sz, :sz]
            print(f"  Input  : aktivasi asli masuk layer {L} dari '{args.image}' -> {C1}x{sz}x{sz}")
        else:
            z = rng.standard_normal((C1, sz, sz)) * 1.5
            x = z / (1 + np.exp(-z))
            print(f"  Input  : acak (sebaran mirip keluaran SiLU) {C1}x{sz}x{sz}")
        x_q = to_q16(x, "input")
        g = c3k2_run(x_q, Pq, f"L{L}", True)
        ref = c3k2_run(x, P, f"L{L}", False)
        if net is not None:   # referensi PyTorch asli + cek peleburan BN
            import torch
            with torch.no_grad():
                tref = net.model[L](torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0))[0].double().numpy()
            print(f"  Cek BN-fold: numpy float vs PyTorch, selisih maks {np.abs(ref - tref).max():.2e}")
            ref = tref
        print()
        report(f"L{L} out", g, ref)
        print("\nMengekspor ...")
        for n in STRUCT:
            if n.startswith(f"L{L}."):
                write_hex(Pq[n][0], f"{n}.w.hex"); write_hex(Pq[n][1], f"{n}.b.hex")
        write_hex(LUT, "sigmoid_lut.hex")
        write_hex(hwc(x_q), "tb_input.hex")
        write_hex(hwc(g), "gold_o1.hex")
        np.save("ref_o1.npy", hwc(ref))
        p = C3K2_PAR[L]
        write_cfg(dict(MODE=1, IN_W=sz, IN_H=sz, IN_CH=C1, O1_W=sz, O1_H=sz, O1_CH=C2,
                       C1=C1, C2=C2, C=C, CB=CB, PREFIX=f"L{L}",
                       P1I=p["cv1"][0], P1O=p["cv1"][1], PM1I=p["m1"][0], PM1O=p["m1"][1],
                       PM2I=p["m2"][0], PM2O=p["m2"][1], P2I=p["cv2"][0], P2O=p["cv2"][1]))
        return

    # ---------------- mode chain ----------------
    sz = args.size or 640
    if sz % 32:
        raise SystemExit("--size untuk mode chain harus kelipatan 32 (mis. 64, 128, 640)")
    if args.image:
        x = load_image(args.image, sz)
        print(f"  Input  : citra '{args.image}' diubah ke {sz}x{sz}")
    else:
        x = rng.uniform(0, 1, (3, sz, sz))
        print(f"  Input  : acak U(0,1) 3x{sz}x{sz}")
    x_q = to_q16(x, "input")
    print("  Menghitung golden integer (bisa beberapa menit untuk 640) ...")
    g4, g5 = chain_run(x_q, Pq, True)
    r4, r5 = chain_run(x, P, False)
    if net is not None:
        t = torch_forward(net, x, 5)
        print(f"  Cek BN-fold: numpy float vs PyTorch, selisih maks L4 {np.abs(r4 - t[4]).max():.2e}, "
              f"L5 {np.abs(r5 - t[5]).max():.2e}")
        r4, r5 = t[4], t[5]
    print()
    report("P3 (L4)", g4, r4)
    report("L5", g5, r5)
    print("\nMengekspor ...")
    for n in STRUCT:
        write_hex(Pq[n][0], f"{n}.w.hex", quiet=True); write_hex(Pq[n][1], f"{n}.b.hex", quiet=True)
    print(f"  Tersimpan: {len(STRUCT)*2} file bobot/bias (L0.w.hex ... L5.b.hex)")
    write_hex(LUT, "sigmoid_lut.hex")
    write_hex(hwc(x_q), "tb_input.hex")
    write_hex(hwc(g4), "gold_o1.hex")
    write_hex(hwc(g5), "gold_o2.hex")
    np.save("ref_o1.npy", hwc(r4)); np.save("ref_o2.npy", hwc(r5))
    write_cfg(dict(MODE=2, IMG=sz, IN_W=sz, IN_H=sz, IN_CH=3,
                   O1_W=sz // 8, O1_H=sz // 8, O1_CH=128, O2_W=sz // 16, O2_H=sz // 16, O2_CH=128))


if __name__ == "__main__":
    main()
