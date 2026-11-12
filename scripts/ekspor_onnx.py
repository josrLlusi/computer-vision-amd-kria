from ultralytics import YOLO

# 1. Muat model yang sama
model = YOLO("yolo26n.pt")

# 2. Ekspor ke format ONNX
# imgsz=640 mengunci resolusi input menjadi statis (640x640 piksel)
# dynamic=False memastikan ukuran matriks benar-benar kaku (syarat mutlak untuk DPU FPGA)
berkas_onnx = model.export(format="onnx", imgsz=640, dynamic=False)

print(f"Model berhasil diekspor ke: {berkas_onnx}")