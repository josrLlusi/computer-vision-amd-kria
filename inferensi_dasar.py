from ultralytics import YOLO

# 1. Memuat model YOLO26 varian Nano
# Jika file yolo26n.pt belum ada, skrip ini akan mengunduhnya secara otomatis
model = YOLO("yolo26n.pt")

# 2. Menjalankan deteksi pada gambar contoh (gunakan gambar apapun di lokal Anda)
# Parameter 'save=True' akan menyimpan hasil gambar dengan bounding box yang sudah digambar
hasil = model.predict(source="gambar_tes.jpeg", save=True)

# 3. Mengintip isi dari output (Tensor)
for deteksi in hasil:
    print("Koordinat Bounding Box:", deteksi.boxes.xyxy) # Format [x_min, y_min, x_max, y_max]
    print("Tingkat Keyakinan (Confidence):", deteksi.boxes.conf)
    print("Kelas Objek:", deteksi.boxes.cls)

model.info(detailed=True)