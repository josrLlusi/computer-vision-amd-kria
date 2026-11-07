"""
validasi_backbone.py
----------------------------------------------------------------------------
Memvalidasi hasil ModelSim tb_backbone.sv (mode conv / c3k2 / chain):
  sim_o1.hex (+ sim_o2.hex)  vs  gold_o1.hex (+ gold_o2.hex)   -> harus 0 LSB
  dan akurasi terhadap referensi float ref_o1.npy (+ ref_o2.npy).
Pemakaian (root repo): python validasi_backbone.py
"""
import re
import numpy as np

Q = 65536.0


def cfg():
    p = {}
    with open("tb_cfg.svh") as f:
        for m in re.finditer(r"localparam int TB_(\w+)\s*=\s*(-?\d+);", f.read()):
            p[m.group(1)] = int(m.group(2))
    return p


def read_hex(fn):
    v, bad = [], 0
    with open(fn) as f:
        for s in (l.strip() for l in f):
            if not s:
                continue
            if any(c in s.lower() for c in "xz"):
                bad += 1
                v.append(0)
                continue
            x = int(s, 16)
            v.append(x - (1 << 32) if x >= (1 << 31) else x)
    return np.array(v, dtype=np.int64), bad


def check(tag, n_frame, ch, w):
    import os
    if os.path.getmtime(f"sim_{tag}.hex") < os.path.getmtime(f"gold_{tag}.hex"):
        print(f" [{tag}] PERINGATAN: sim_{tag}.hex lebih lama dari gold_{tag}.hex"
              f" -> simulasi belum dijalankan ulang setelah ekstraksi!")
        return False
    sim, nxz = read_hex(f"sim_{tag}.hex")
    gold, _ = read_hex(f"gold_{tag}.hex")
    ref = np.load(f"ref_{tag}.npy").reshape(-1)
    reps = max(1, sim.size // n_frame) if sim.size % n_frame == 0 and sim.size else 1
    gold, ref = np.tile(gold, reps), np.tile(ref, reps)
    n_exp = n_frame * reps
    n = min(sim.size, n_exp)
    d = np.abs(sim[:n] - gold[:n])
    bad = int(np.count_nonzero(d))
    print(f" [{tag}] nilai {sim.size}/{n_exp} ({reps} frame), X/Z {nxz}, mismatch {bad}, "
          f"selisih maks {int(d.max()) if n else 0} LSB")
    if bad:
        for i in np.nonzero(d)[0][:5]:
            pix, c = divmod(int(i) % n_frame, ch)
            print(f"     y={pix // w} x={pix % w} ch={c}: RTL={sim[i]} GOLD={gold[i]}")
    err = np.abs(sim[:n] / Q - ref[:n])
    rel = np.linalg.norm(sim[:n] / Q - ref[:n]) / max(np.linalg.norm(ref[:n]), 1e-12)
    print(f"        vs float: err maks {err.max():.2e} ({err.max()*Q:.0f} LSB), relatif {rel:.2e} "
          f"-> akurasi {100*(1-rel):.4f} %")
    return bad == 0 and nxz == 0 and sim.size == n_exp


def main():
    p = cfg()
    mode = {0: "conv_kxk", 1: "c3k2", 2: "backbone_l0_l5"}[p["MODE"]]
    print("=" * 66)
    print(f" Validasi {mode}")
    print("=" * 66)
    ok = check("o1", p["O1_W"] * p["O1_H"] * p["O1_CH"], p["O1_CH"], p["O1_W"])
    if p["MODE"] == 2:
        ok &= check("o2", p["O2_W"] * p["O2_H"] * p["O2_CH"], p["O2_CH"], p["O2_W"])
    print("-" * 66)
    print(" STATUS: " + ("LULUS VERIFIKASI (bit-exact)" if ok else "GAGAL"))
    print("=" * 66)


if __name__ == "__main__":
    main()
