"""
ekstrak_uji_multichannel.py  (v2)
----------------------------------------------------------------------------
Data uji + golden data untuk conv3x3_s2.sv v2 = blok "Conv" YOLO26:
    y = SiLU( BN( Conv2d(x, 3x3, stride 2, padding 1, tanpa bias) ) )
BatchNorm dilebur ke bobot & bias (w' = w*g/sqrt(var+eps), b' = beta - mean*g/sqrt(var+eps)).

Preset --layer mengikuti yolo26n (skala n: width 0.25, max 1024):
    layer 0  P1/2  :   3 ->  16, 640x640 -> 320x320   (gambar: "Conv 64")
    layer 1  P2/4  :  16 ->  32, 320x320 -> 160x160   (gambar: "Conv 128")
    layer 3  P3/8  :  64 ->  64, 160x160 ->  80x80    (gambar: "Conv 256")
    layer 5  P4/16 : 128 -> 128,  80x80  ->  40x40    (gambar: "Conv 512")
    layer 7  P5/32 : 128 -> 256,  40x40  ->  20x20    (gambar: "Conv 1024")
    (layer 17 & 20 di neck juga Conv 3x3/2: 64->64 @80, 128->128 @40)

File keluaran (di folder kerja):
    tb_input.hex        input Q16.16, urutan [y][x][ic]
    tb_weight.hex       bobot (BN-folded) Q16.16, urutan [oc][ic][ky][kx]
    tb_bias.hex         bias  (BN-folded) Q16.16, urutan [oc]
    sigmoid_lut.hex     LUT sigmoid 2049 titik, Q0.16
    tb_golden.hex       golden BIT-EXACT (model integer = RTL), urutan [oy][ox][oc]
    tb_golden_float.npy referensi float (PyTorch jika --yolo, numpy float64 jika acak)
    tb_params.svh       parameter testbench (otomatis, jangan diedit)

Contoh (dari root repo, venv aktif):
    python scripts/ekstrak_uji_multichannel.py --layer 0 --size 64            # cepat, bobot acak
    python scripts/ekstrak_uji_multichannel.py --layer 0 --yolo yolo26n.pt --image assets/gambar_tes.jpeg
    python scripts/ekstrak_uji_multichannel.py --layer 5 --yolo yolo26n.pt --image assets/gambar_tes.jpeg
      -> dengan --yolo + --image, input layer diambil dari aktivasi ASLI jaringan
         (hook PyTorch), dan golden float = keluaran layer PyTorch asli.
"""
import argparse
import numpy as np

Q_FRAC = 16
Q_SCALE = 1 << Q_FRAC
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1

# layer: (IN, OUT, ukuran input, PAR_IC, PAR_OC) -- PAR seimbang, lihat README
PRESETS = {
    0: (3, 16, 640, 3, 2),
    1: (16, 32, 320, 4, 2),
    3: (64, 64, 160, 8, 2),
    5: (128, 128, 80, 8, 2),
    7: (128, 256, 40, 4, 2),
    17: (64, 64, 80, 8, 2),
    20: (128, 128, 40, 8, 2),
}

SIG_LO, SIG_STEP_BITS, SIG_N = -16, 10, 2049   # langkah 2^-6 = 1/64 (2^10 LSB Q16.16)


# ---------------------------------------------------------------------------
# Q16.16 & model integer BIT-EXACT (meniru conv3x3_s2.sv)
# ---------------------------------------------------------------------------
def to_q16(arr, name=""):
    q = np.round(np.asarray(arr, dtype=np.float64) * Q_SCALE)
    n_sat = int(np.count_nonzero((q < I32_MIN) | (q > I32_MAX)))
    if n_sat:
        print(f"  PERINGATAN: {n_sat} nilai {name} di luar rentang Q16.16 -> disaturasi")
    return np.clip(q, I32_MIN, I32_MAX).astype(np.int64)


def wrap32(v):
    v = np.asarray(v, dtype=np.int64) & 0xFFFFFFFF
    return np.where(v >= (1 << 31), v - (1 << 32), v)


def make_sigmoid_lut():
    xs = SIG_LO + np.arange(SIG_N) / 64.0
    return np.round(Q_SCALE / (1.0 + np.exp(-xs))).astype(np.int64)


def silu_fixed(r, lut):
    """r: int64 Q16.16 (32-bit bertanda). Sama persis dengan silu_q16() di RTL."""
    r = np.asarray(r, dtype=np.int64)
    hi, lo = 16 << 16, -(16 << 16)
    mid = (r >= lo) & (r < hi)
    xo = np.where(mid, r + hi, 0)
    idx, frac = xo >> SIG_STEP_BITS, xo & ((1 << SIG_STEP_BITS) - 1)
    s0, s1 = lut[idx], lut[np.minimum(idx + 1, SIG_N - 1)]
    sig = s0 + (((s1 - s0) * frac) >> SIG_STEP_BITS)
    y = wrap32((r * sig) >> Q_FRAC)
    return np.where(r >= hi, r, np.where(mid, y, 0))


def golden_fixed(x_q, w_q, b_q, lut, use_silu=True):
    """
    x_q [IN,H,W], w_q [OUT,IN,3,3], b_q [OUT]  (semua int64 Q16.16)
    return [OH,OW,OUT] int64 Q16.16  (stride 2, padding 1)
    """
    _, H, W = x_q.shape
    OH, OW = H // 2, W // 2
    xp = np.pad(x_q, ((0, 0), (1, 1), (1, 1)))
    acc = np.broadcast_to(b_q << Q_FRAC, (OH, OW, w_q.shape[0])).astype(np.int64).copy()
    with np.errstate(over="ignore"):              # wrap 2^64 = register 64-bit
        for ky in range(3):
            for kx in range(3):
                patch = xp[:, ky:ky + 2 * OH - 1:2, kx:kx + 2 * OW - 1:2]   # [IN,OH,OW]
                acc += np.einsum("cyx,oc->yxo", patch, w_q[:, :, ky, kx])
    res = wrap32(acc >> Q_FRAC)                    # = acc[47:16]
    return silu_fixed(res, lut) if use_silu else res


def float_reference(x, w, b, use_silu=True):
    """Referensi float64 (numpy) - dipakai jika tidak ada layer PyTorch asli."""
    _, H, W = x.shape
    OH, OW = H // 2, W // 2
    xp = np.pad(x.astype(np.float64), ((0, 0), (1, 1), (1, 1)))
    y = np.broadcast_to(b.astype(np.float64), (OH, OW, w.shape[0])).copy()
    for ky in range(3):
        for kx in range(3):
            patch = xp[:, ky:ky + 2 * OH - 1:2, kx:kx + 2 * OW - 1:2]
            y += np.einsum("cyx,oc->yxo", patch, w[:, :, ky, kx].astype(np.float64))
    return y / (1.0 + np.exp(-y)) if use_silu else y


def fold_bn(w, gamma, beta, mean, var, eps):
    """Leburkan BatchNorm ke Conv: kembalikan (w', b') float64."""
    s = gamma.astype(np.float64) / np.sqrt(var.astype(np.float64) + eps)
    return w.astype(np.float64) * s[:, None, None, None], beta - mean * s


# ---------------------------------------------------------------------------
# Ekspor
# ---------------------------------------------------------------------------
def write_hex(values, filename):
    flat = np.asarray(values, dtype=np.int64).reshape(-1) & 0xFFFFFFFF
    with open(filename, "w") as f:
        f.write("\n".join(f"{int(v):08x}" for v in flat) + "\n")
    print(f"  Tersimpan: {filename:<20} ({flat.size} baris)")


def export_case(x_f, w_f, b_f, par_ic, par_oc, use_silu=True, ref_float=None):
    """
    x_f [IN,H,W], w_f [OUT,IN,3,3], b_f [OUT] (float, BN sudah dilebur)
    ref_float [OH,OW,OUT] opsional (keluaran layer PyTorch asli)
    """
    IN, H, W = x_f.shape
    OUT = w_f.shape[0]
    if H % 2 or W % 2:
        raise SystemExit("Ukuran citra harus genap.")
    if IN % par_ic or OUT % par_oc:
        raise SystemExit(f"PAR_IC={par_ic} harus membagi {IN} dan PAR_OC={par_oc} membagi {OUT}.")

    x_q, w_q, b_q = to_q16(x_f, "input"), to_q16(w_f, "bobot"), to_q16(b_f, "bias")
    lut = make_sigmoid_lut()
    g_q = golden_fixed(x_q, w_q, b_q, lut, use_silu)
    OH, OW = H // 2, W // 2

    if ref_float is None:   # referensi float dari nilai float ASLI (bukan terkuantisasi)
        ref_float = float_reference(x_f, w_f, b_f, use_silu)
    err = np.abs(g_q / Q_SCALE - ref_float)
    rel = np.linalg.norm(g_q / Q_SCALE - ref_float) / max(np.linalg.norm(ref_float), 1e-12)

    print(f"\n  Dimensi : {IN}x{H}x{W} -> {OUT}x{OH}x{OW}  (3x3, stride 2, padding 1"
          f"{', SiLU' if use_silu else ''})")
    print(f"  PAR     : PAR_IC={par_ic}, PAR_OC={par_oc} -> {par_ic*par_oc*9} pengali, "
          f"{(IN//par_ic)*(OUT//par_oc)} siklus/piksel keluaran")
    print(f"  Model Q16.16 vs float: err maks {err.max():.2e} ({err.max()*Q_SCALE:.1f} LSB), "
          f"err relatif L2 {rel:.2e}")

    print("\nMengekspor ...")
    write_hex(x_q.transpose(1, 2, 0), "tb_input.hex")
    write_hex(w_q, "tb_weight.hex")
    write_hex(b_q, "tb_bias.hex")
    write_hex(lut, "sigmoid_lut.hex")
    write_hex(g_q, "tb_golden.hex")
    np.save("tb_golden_float.npy", np.asarray(ref_float, dtype=np.float64))
    print("  Tersimpan: tb_golden_float.npy")

    with open("tb_params.svh", "w") as f:
        f.write("// Dibangkitkan otomatis oleh ekstrak_uji_multichannel.py - JANGAN diedit\n")
        for k, v in [("IMAGE_WIDTH", W), ("IMAGE_HEIGHT", H), ("IN_CHANNELS", IN),
                     ("OUT_CHANNELS", OUT), ("OUT_WIDTH", OW), ("OUT_HEIGHT", OH),
                     ("PAR_IC", par_ic), ("PAR_OC", par_oc), ("USE_SILU", int(use_silu))]:
            f.write(f"localparam int TB_{k:<13}= {v};\n")
    print("  Tersimpan: tb_params.svh")
    return g_q


# ---------------------------------------------------------------------------
# Sumber data (PyTorch / Ultralytics hanya di-import bila diperlukan)
# ---------------------------------------------------------------------------
def load_image(path, size):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    return (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)   # [3,H,W], RGB, [0,1]


def from_yolo(args):
    import torch
    from ultralytics import YOLO
    net = YOLO(args.yolo).model.float().eval()
    layer = net.model[args.layer]
    conv = layer.conv
    if tuple(conv.kernel_size) != (3, 3) or tuple(conv.stride) != (2, 2) or tuple(conv.padding) != (1, 1):
        raise SystemExit(f"Layer {args.layer} bukan Conv 3x3/2 pad 1: {conv}")
    if not isinstance(layer.act, torch.nn.SiLU):
        print(f"  PERINGATAN: aktivasi layer = {type(layer.act).__name__}, bukan SiLU")

    w = conv.weight.detach().double().numpy()
    if hasattr(layer, "bn"):
        bn = layer.bn
        w_f, b_f = fold_bn(w, bn.weight.detach().double().numpy(), bn.bias.detach().double().numpy(),
                           bn.running_mean.double().numpy(), bn.running_var.double().numpy(), bn.eps)
        print(f"  Bobot  : {args.yolo} layer[{args.layer}] {tuple(w.shape)}, BN dilebur (eps={bn.eps})")
    else:   # model sudah di-fuse
        w_f = w
        b_f = conv.bias.detach().double().numpy() if conv.bias is not None else np.zeros(w.shape[0])
        print(f"  Bobot  : {args.yolo} layer[{args.layer}] {tuple(w.shape)} (sudah fused)")

    size = args.size or PRESETS.get(args.layer, (0, 0, 640))[2]
    if args.image:
        img = torch.from_numpy(load_image(args.image, 640)).unsqueeze(0)
        cap = {}
        h = layer.register_forward_pre_hook(lambda m, inp: cap.setdefault("x", inp[0].detach()))
        with torch.no_grad():
            net(img)
        h.remove()
        x = cap["x"][0]
        print(f"  Input  : aktivasi ASLI masuk layer {args.layer} dari '{args.image}', bentuk {tuple(x.shape)}")
        if args.size and args.size < x.shape[1]:
            x = x[:, :args.size, :args.size]
            print(f"           dipotong ke {args.size}x{args.size}")
    else:
        torch.manual_seed(args.seed)
        x = torch.randn(conv.in_channels, size, size)
        print(f"  Input  : acak N(0,1), {tuple(x.shape)}")

    with torch.no_grad():
        ref = layer(x.unsqueeze(0).float())[0] if not args.no_silu else \
              torch.nn.functional.conv2d(x.unsqueeze(0).double(), torch.from_numpy(w_f), torch.from_numpy(b_f),
                                         stride=2, padding=1)[0]
    return x.double().numpy(), w_f, b_f, ref.double().permute(1, 2, 0).numpy()


def from_random(args):
    IN, OUT, size, _, _ = PRESETS.get(args.layer, (3, 16, 64, 3, 2))
    IN, OUT = args.in_ch or IN, args.out_ch or OUT
    size = args.size or size
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((IN, size, size))
    w = rng.uniform(-1, 1, (OUT, IN, 3, 3)) / np.sqrt(IN)      # skala mirip bobot terlatih
    b = rng.uniform(-1, 1, OUT)
    print(f"  Input  : acak N(0,1) {IN}x{size}x{size}; bobot & bias acak")
    return x, w, b, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, default=0, help="indeks layer Conv yolo26n: 0,1,3,5,7 (atau 17,20)")
    p.add_argument("--size", type=int, default=None, help="ukuran input (default: preset layer)")
    p.add_argument("--in-ch", type=int, default=None)
    p.add_argument("--out-ch", type=int, default=None)
    p.add_argument("--par-ic", type=int, default=None)
    p.add_argument("--par-oc", type=int, default=None)
    p.add_argument("--yolo", type=str, default=None, help="path yolo26n.pt (bobot asli + BN)")
    p.add_argument("--image", type=str, default=None, help="citra uji (dengan --yolo: aktivasi asli)")
    p.add_argument("--no-silu", action="store_true", help="uji konvolusi+bias saja")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("=" * 64)
    print(f" Data uji conv3x3_s2 v2 - layer {args.layer}")
    print("=" * 64)
    if args.yolo:
        x, w, b, ref = from_yolo(args)
    else:
        x, w, b, ref = from_random(args)
        if args.image:
            x = load_image(args.image, args.size or 640)
            w, b = w[:, :3], b
            print(f"  Input  : citra '{args.image}'")

    pre = PRESETS.get(args.layer, (0, 0, 0, 1, 1))
    par_ic = args.par_ic or (pre[3] if x.shape[0] % pre[3] == 0 else 1)
    par_oc = args.par_oc or (pre[4] if w.shape[0] % pre[4] == 0 else 1)
    export_case(x, w, b, par_ic, par_oc, not args.no_silu, ref)
    print("\nLanjut: vsim -c -do sim/run_sim_multich.do   lalu   python scripts/validasi_simulasi_multich.py")


if __name__ == "__main__":
    main()
