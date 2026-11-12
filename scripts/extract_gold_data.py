import torch
from ultralytics import YOLO
import numpy as np
import os

def float_to_q16_16_hex(tensor, filename):
    """Mengubah tensor float32 menjadi format hex Q16.16 (32-bit)."""
    # Ratakan tensor menjadi array 1 dimensi
    arr = tensor.detach().cpu().numpy().flatten()
    
    # Kalikan 2^16, ubah ke int32, lalu BACA memorinya sebagai uint32 (Two's Complement otomatis)
    q_arr = np.int32(arr * 65536.0).view(np.uint32)
    
    with open(filename, 'w') as f:
        for val in q_arr:
            # Format langsung sebagai hex 8 digit karena data sudah berwujud unsigned 32-bit
            f.write(f"{val:08x}\n")
    print(f"Tersimpan: {filename} ({len(q_arr)} baris)")

def main():
    print("Memuat model YOLO26n...")
    model = YOLO("yolo26n.pt")
    
    # Mengisolasi lapisan pertama: Conv 64, 3x3, stride 2
    conv_layer = model.model.model[0].conv
    
    # Membuat tensor gambar dummy berukuran 640x640 RGB (1, 3, 640, 640)
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 640, 640)
    
    print("Menjalankan inferensi pada lapisan pertama...")
    with torch.no_grad():
        golden_output = conv_layer(dummy_input)
    
    print("\nMengekspor Golden Data ke Q16.16 HEX...")
    float_to_q16_16_hex(dummy_input, "tb_input_640x640.hex")
    float_to_q16_16_hex(conv_layer.weight, "tb_weight_3x3.hex")
    
    if conv_layer.bias is not None:
        float_to_q16_16_hex(conv_layer.bias, "tb_bias.hex")
        
    float_to_q16_16_hex(golden_output, "tb_output_golden.hex")
    
    print("\nEkstraksi selesai. Data siap untuk $readmemh.")

if __name__ == "__main__":
    main()