"""
validasi_simulasi_multich.py
----------------------------------------------------------------------------
Memvalidasi sim_output.hex (hasil ModelSim) terhadap:
  1) tb_golden.hex         : golden BIT-EXACT (target: 0 LSB)
  2) tb_golden_float.npy   : referensi PyTorch float (laporan akurasi)

Pemakaian (dari root repo):  python validasi_simulasi_multich.py [--tol 0]
"""
import argparse
import re
import numpy as np

Q_SCALE = 65536.0


def read_params(path="tb_params.svh"):
    params = {}
    with open(path) as f:
        for m in re.finditer(r"localparam int (\w+)\s*=\s*(\d+);", f.read()):
            params[m.group(1)] = int(m.group(2))
    return params


def read_hex(path):
    vals, bad = [], 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if any(ch in s.lower() for ch in "xz"):
                bad += 1
                vals.append(0)
                continue
            v = int(s, 16)
            vals.append(v - (1 << 32) if v >= (1 << 31) else v)
    return np.array(vals, dtype=np.int64), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=int, default=0, help="toleransi LSB vs golden bit-exact (default 0)")
    args = ap.parse_args()

    p = read_params()
    OH, OW, OC = p["TB_OUT_HEIGHT"], p["TB_OUT_WIDTH"], p["TB_OUT_CHANNELS"]
    n_exp = OH * OW * OC
    n_frame = n_exp

    sim, n_xz = read_hex("sim_output.hex")
    gold, _ = read_hex("tb_golden.hex")
    gflt = np.load("tb_golden_float.npy").reshape(-1)
    # Beberapa frame (+FRAMES=n): golden diulang
    if sim.size > n_exp and sim.size % n_exp == 0:
        reps = sim.size // n_exp
        n_exp *= reps
        gold, gflt = np.tile(gold, reps), np.tile(gflt, reps)
        print(f" (terdeteksi {reps} frame)")

    print("=" * 60)
    print(f" Validasi conv3x3_s2 v2: {OC} ch x {OH} x {OW} = {n_frame} nilai/frame"
          f"{' (dengan SiLU)' if p.get('TB_USE_SILU', 0) else ''}")
    print("=" * 60)
    print(f" Nilai dari RTL        : {sim.size}")
    print(f" Nilai X/Z             : {n_xz}")

    ok_count = sim.size == n_exp and n_xz == 0
    n = min(sim.size, n_exp)
    s, g, gf = sim[:n], gold[:n], gflt[:n]

    diff = np.abs(s - g)
    n_bad = int(np.count_nonzero(diff > args.tol))
    print(f" Mismatch (> {args.tol} LSB)    : {n_bad}")
    print(f" Selisih maks vs golden: {int(diff.max()) if n else 0} LSB")
    print(f" Bit-exact             : {100.0 * np.count_nonzero(diff == 0) / max(n, 1):.4f} %")

    if n_bad:
        idx = np.nonzero(diff > args.tol)[0][:10]
        print("\n Contoh mismatch pertama:")
        for i in idx:
            pix, oc = divmod(int(i), OC)
            oy, ox = divmod(pix, OW)
            print(f"   oy={oy:4d} ox={ox:4d} oc={oc:3d}  RTL={s[i]:>12d}  GOLD={g[i]:>12d}  beda={diff[i]}")
        # Ringkasan per saluran -> membantu melacak kesalahan indeks bobot/packing
        bad_ch = np.unique(np.nonzero((diff > args.tol))[0] % OC)
        print(f" Saluran keluaran yang salah: {bad_ch.tolist()[:20]}{' ...' if bad_ch.size > 20 else ''}")

    # Akurasi terhadap PyTorch float
    real = s / Q_SCALE
    err = np.abs(real - gf)
    rel = np.linalg.norm(real - gf) / max(np.linalg.norm(gf), 1e-12)
    print("\n Akurasi vs PyTorch float:")
    print(f"   Error absolut maks  : {err.max():.3e}  ({err.max() * Q_SCALE:.2f} LSB)")
    print(f"   Error absolut rerata: {err.mean():.3e}")
    print(f"   Error relatif (L2)  : {rel:.3e}  -> akurasi {100 * (1 - rel):.5f} %")
    print("   (selisih vs float = kuantisasi Q16.16 + aproksimasi LUT SiLU; yang menentukan LULUS adalah bit-exact)")

    print("-" * 60)
    if ok_count and n_bad == 0:
        print(" STATUS: LULUS VERIFIKASI! Conv 3x3/2 + bias + SiLU akurat (bit-exact).")
    else:
        if not ok_count:
            print(f" Jumlah/kualitas output tidak sesuai (diharapkan {n_exp}, X/Z={n_xz}).")
        print(" STATUS: GAGAL.")
    print("=" * 60)


if __name__ == "__main__":
    main()
