# Backbone YOLO26n Layer 0–5 (RTL SystemVerilog, Q16.16)

Bagian backbone yang ditugaskan: **Conv 64 → Conv 128 → C3k2 256 → Conv 256 → C3k2 512 → Conv 512**
(layer 0–5 yolo26n). Keluaran layer 4 (**P3/8**) diteruskan ke neck, keluaran layer 5 (**P4/16**) ke layer 6.

## File

| File | Isi |
|---|---|
| `rtl/conv_kxk.sv` | Conv generik: 1×1, 3×3 stride 1, 3×3 stride 2 (+ BN-folded bias + SiLU) |
| `rtl/stream_fifo.sv` | FIFO valid/ready (penahan cabang di C3k2) |
| `rtl/c3k2.sv` | Blok C3k2 (cv1 → chunk → Bottleneck + residual → concat 3 bagian → cv2) |
| `rtl/backbone_l0_l5.sv` | Rangkaian layer 0–5, keluaran `p3_*` (layer 4) dan `l5_*` (layer 5) |
| `tb/tb_backbone.sv` | Testbench generik (mode conv / c3k2 / chain) |
| `scripts/ekstrak_uji_backbone.py` | Data uji + golden bit-exact + referensi PyTorch |
| `scripts/validasi_backbone.py` | Pemeriksa hasil simulasi |
| `sim/run_sim_backbone.do` | Skrip ModelSim |

`conv3x3_s2.sv` dan file `*_multich*` yang lama tetap valid untuk menguji Conv 3×3/2 sendiri.
`conv_kxk.sv` dengan `KERNEL=3, STRIDE=2` menghitung hal yang sama.

## Isi C3k2 pada yolo26n (dicek dari bobot asli)

```
Layer 2 (32→64 ch, 160×160)              Layer 4 (64→128 ch, 80×80)
  cv1  Conv 1×1  32→32                     cv1  Conv 1×1  64→64
  chunk → y0 (16), y1 (16)                 chunk → y0 (32), y1 (32)
  z = y1 + Conv3×3(Conv3×3(y1))            z = y1 + Conv3×3(Conv3×3(y1))
        (16→8→16)                                (32→16→32)
  cv2  Conv 1×1  cat[y0,y1,z] 48→64        cv2  Conv 1×1  cat[y0,y1,z] 96→128
```
Concat menggabungkan **tiga** bagian (y0, y1, z), bukan dua seperti tampak di diagram.

## Cara menjalankan (root repo, venv aktif)

```powershell
# C3k2 cepat (bobot acak)
python scripts/ekstrak_uji_backbone.py --mode c3k2 --layer 2 --size 16
# C3k2 dengan bobot & aktivasi ASLI
python scripts/ekstrak_uji_backbone.py --mode c3k2 --layer 2 --yolo yolo26n.pt --image assets/gambar_tes.jpeg
python scripts/ekstrak_uji_backbone.py --mode c3k2 --layer 4 --yolo yolo26n.pt --image assets/gambar_tes.jpeg
# Rangkaian layer 0–5 (mulai dari 64, lalu 640 penuh)
python scripts/ekstrak_uji_backbone.py --mode chain --size 64 --yolo yolo26n.pt --image assets/gambar_tes.jpeg
python scripts/ekstrak_uji_backbone.py --mode chain --yolo yolo26n.pt --image assets/gambar_tes.jpeg
```
Lalu di ModelSim `do sim/run_sim_backbone.do` (atau `vsim -c -do sim/run_sim_backbone.do`), dan
`python scripts/validasi_backbone.py`.

Dengan `--yolo`, skrip juga mencetak **"Cek BN-fold"**, yaitu selisih antara model float kita dan
PyTorch asli. Nilainya harus sangat kecil (orde 1e-5 atau lebih kecil). Itu bukti bahwa BatchNorm
dilebur dengan benar.

## Antarmuka `backbone_l0_l5`

```systemverilog
backbone_l0_l5 #(.IMG(640)) u_bb (
  .clk, .rst_n,
  .pixel_in, .valid_in, .ready_in,     // 3 ch/beat (R,G,B; ch0 di LSB), Q16.16, nilai 0..1
  .p3_data, .p3_valid, .p3_ready,      // layer 4: 128 ch/beat, 80×80  -> ke neck
  .l5_data, .l5_valid, .l5_ready);     // layer 5: 128 ch/beat, 40×40  -> ke layer 6
```
Satu beat adalah satu piksel dengan semua saluran, urutan raster (baris demi baris). Keluaran
layer 4 hanya maju bila `p3_ready` **dan** layer 5 siap, jadi tim neck wajib menerima P3 secara
streaming atau menyimpannya di buffer sendiri.

## Catatan desain

- **Padding streaming:** Conv 3×3 stride 1 berjalan pada grid (W+1)×(H+1) dengan kolom/baris
  "fantom" bernilai 0 yang disisipkan sendiri, sehingga padding kanan/bawah benar tanpa frame buffer.
- **FIFO C3k2:** cabang y0/y1 ditahan selama Bottleneck menghitung. Isi maksimum terukur =
  2·(W+2), dan kedalaman default = 2·(W+2)+16 (layer 2: 340 entri × 1024 bit; layer 4: 180 × 2048 bit).
- **Masih Q16.16:** untuk muat seluruh YOLO26 di KV260 perlu INT8/16-bit (lihat diskusi tim).
  Struktur RTL tetap sama; yang berubah lebar data dan aritmetika.
