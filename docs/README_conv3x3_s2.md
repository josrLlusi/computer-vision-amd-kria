# conv3x3_s2 v2 — Blok `Conv` 3×3/2 Backbone YOLO26 (RTL SystemVerilog)

Blok biru **Conv 3×3 / 2** pada backbone YOLO26 di Ultralytics adalah:

```
y = SiLU( BatchNorm( Conv2d(x, kernel 3x3, stride 2, padding 1, bias=False) ) )
```

`conv3x3_s2.sv` mengimplementasikan blok ini secara utuh untuk FPGA (Q16.16):

| Fitur | Status |
|---|---|
| Konvolusi 3×3, stride 2, multi-channel | ya |
| Padding 1 (output 320×320 dari 640×640, sama dengan PyTorch) | ya |
| BatchNorm (dilebur ke bobot + bias oleh skrip Python) | ya |
| Aktivasi SiLU (LUT sigmoid 2049 titik + interpolasi linear) | ya |
| MAC time-multiplexed (`PAR_IC`, `PAR_OC`) supaya muat di DSP | ya |
| Handshake valid/ready dua arah (siap dirangkai / AXI-Stream) | ya |
| Multi-frame (koordinat kembali ke 0 tiap `IMAGE_HEIGHT` baris) | ya |

## Dimensi asli yolo26n (skala n: width 0.25)

Angka pada diagram (64, 128, 256, 512, 1024) adalah lebar **dasar**. Untuk yolo26n dikali 0.25:

| Layer | Diagram | IN → OUT | Input → Output | Preset PAR (IC×OC) | Pengali | Siklus/frame* |
|---|---|---|---|---|---|---|
| 0 (P1/2) | Conv 64 | 3 → 16 | 640² → 320² | 3×2 | 54 | 1,23 jt |
| 1 (P2/4) | Conv 128 | 16 → 32 | 320² → 160² | 4×2 | 72 | 1,74 jt |
| 3 (P3/8) | Conv 256 | 64 → 64 | 160² → 80² | 8×2 | 144 | 1,66 jt |
| 5 (P4/16) | Conv 512 | 128 → 128 | 80² → 40² | 8×2 | 144 | 1,65 jt |
| 7 (P5/32) | Conv 1024 | 128 → 256 | 40² → 20² | 4×2 | 72 | 1,64 jt |

\*Siklus/frame ≈ `H·W + OH·OW·(IN/PAR_IC)·(OUT/PAR_OC)`. PAR dipilih supaya kelima layer
seimbang (~1,7 juta siklus). Jika dirangkai sebagai dataflow pada 200 MHz, itu sekitar 8,7 ms/frame
untuk kelima Conv ini (frekuensi belum diverifikasi lewat sintesis).
Layer neck 17 & 20 juga Conv 3×3/2 dan bisa memakai modul yang sama.

## Cara menjalankan (dari root repo, venv aktif)

```bash
# 1) Data uji: bobot ASLI + aktivasi ASLI dari gambar (butuh ultralytics + torch)
python scripts/ekstrak_uji_multichannel.py --layer 0 --yolo yolo26n.pt --image assets/gambar_tes.jpeg
#    atau cepat dengan bobot acak & ukuran kecil:
python scripts/ekstrak_uji_multichannel.py --layer 5 --size 16

# 2) Simulasi ModelSim (ubah BUBBLE/BACKPRESSURE/FRAMES di run_sim_multich.do untuk uji stres)
vsim -c -do sim/run_sim_multich.do

# 3) Validasi
python scripts/validasi_simulasi_multich.py
```

Ganti `--layer` dengan 1, 3, 5, atau 7 untuk layer lain. Dengan `--yolo` + `--image`, input layer
diambil dari aktivasi asli jaringan (forward hook PyTorch), dan referensi float adalah
keluaran layer PyTorch itu sendiri.

## Antarmuka modul

```systemverilog
conv3x3_s2 #(.IMAGE_WIDTH(640), .IMAGE_HEIGHT(640), .IN_CHANNELS(3), .OUT_CHANNELS(16),
             .PAR_IC(3), .PAR_OC(2), .USE_SILU(1),
             .WEIGHT_FILE("tb_weight.hex"), .BIAS_FILE("tb_bias.hex"),
             .SIGMOID_FILE("sigmoid_lut.hex")) u_conv (
  .clk, .rst_n,
  .pixel_in,  .valid_in,  .ready_in,    // 1 beat = 1 piksel, semua saluran (ic0 di LSB)
  .pixel_out, .valid_out, .ready_out);  // 1 beat = 1 piksel keluaran (oc0 di LSB)
```

> Perubahan dari versi lama: `ready_out` sekarang adalah **input** (back-pressure dari hilir),
> dan ada `ready_in` (output) untuk memberi tahu hulu bahwa modul siap menerima piksel.

## Catatan sumber daya KV260 (perkiraan, perlu dikonfirmasi lewat sintesis Vivado)

- Satu DSP48E2 adalah pengali 27×18 (dua DSP cukup sampai 35×27). Jadi perkalian 32×32 (Q16.16)
  butuh sekitar 3–4 DSP. Preset di atas butuh ±496 pengali, atau ±1.500–2.000 DSP, sedangkan K26
  hanya punya **1.248 DSP**. Pada Q16.16, PAR harus dikurangi sekitar separuhnya (throughput
  ikut turun separuh).
- Bobot Q16.16 kelima layer = 484.272 × 32 bit ≈ 15,5 Mbit dari 26,6 Mbit SRAM on-chip.
- Rekomendasi: beralih ke 16-bit atau INT8 per-kanal (seperti Vitis AI). Satu pengali lalu cukup
  memakai 1 DSP (atau 2 MAC INT8 per DSP), dan memori bobot turun 2–4×.
- Datapath MAC saat ini adalah satu rantai kombinasional (PAR_IC×9 perkalian + penjumlahan +
  SiLU). Untuk mencapai ≥200 MHz perlu ditambah register pipeline (adder tree).
