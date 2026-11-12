# YOLO26n di FPGA AMD Kria

Implementasi RTL (SystemVerilog, fixed-point Q16.16) untuk bagian backbone YOLO26n, dengan
verifikasi bit-exact terhadap model integer di Python dan referensi PyTorch.

Status saat ini: backbone layer 0–5 (Conv → Conv → C3k2 → Conv → C3k2 → Conv) selesai dan
lolos simulasi. Layer berikutnya dan sintesis di Kria belum dikerjakan.

## Struktur

```
rtl/       modul RTL: conv_kxk, conv3x3_s2, c3k2, stream_fifo, backbone_l0_l5
tb/        testbench self-checking
sim/       skrip ModelSim/Questa (.do)
scripts/   ekspor model, pembuatan data uji + golden, validasi hasil simulasi
docs/      catatan per modul
hls/       kode Vitis HLS (demo AXI-Stream)
assets/    gambar uji
```

## Kebutuhan

- Python 3 dengan `ultralytics`, `torch`, `numpy`
- ModelSim atau Questa
- `yolo26n.pt` di root repo (diunduh otomatis oleh `python scripts/inferensi_dasar.py`)

## Cara pakai

Semua perintah dijalankan dari root repo.

```powershell
# 1. Buat data uji dan golden
python scripts/ekstrak_uji_backbone.py --mode chain --size 64 --yolo yolo26n.pt --image assets/gambar_tes.jpeg

# 2. Simulasi
vsim -c -do sim/run_sim_backbone.do

# 3. Bandingkan hasil simulasi dengan golden
python scripts/validasi_backbone.py
```

File `.hex`, `.npy`, dan `tb_*.svh` dibuat ulang oleh langkah 1 sehingga tidak ikut di git.

Detail tiap modul ada di [docs/README_backbone.md](docs/README_backbone.md) dan
[docs/README_conv3x3_s2.md](docs/README_conv3x3_s2.md).
