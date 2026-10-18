import torch
import torch.nn as nn
import numpy as np

def float_to_q16_16_hex(tensor, filename):
    arr = tensor.detach().cpu().numpy().flatten()
    q_arr = np.int32(arr * 65536.0).view(np.uint32)
    with open(filename, 'w') as f:
        for val in q_arr:
            f.write(f"{val:08x}\n")
    print(f"Tersimpan: {filename} ({len(q_arr)} baris)")

def main():
    print("Menyiapkan Single-Channel Testbench untuk RTL...")
    # Buat dummy input 1 channel, 640x640
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 1, 640, 640)
    
    # Custom Conv2d: 1 channel, stride 2, TANPA padding, TANPA bias
    conv_test = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, 
                          stride=2, padding=0, bias=False)
    
    # Acak bobot antara -1.0 hingga 1.0
    nn.init.uniform_(conv_test.weight, -1.0, 1.0)
    
    print("Menjalankan inferensi tanpa padding...")
    with torch.no_grad():
        golden_output = conv_test(dummy_input)
        
    # Karena tidak ada padding, output pasti berukuran 319x319 (101.761 piksel)
    print(f"Bentuk Output PyTorch: {golden_output.shape}") 
    
    print("\nMengekspor Data Q16.16 HEX...")
    float_to_q16_16_hex(dummy_input, "tb_input_640x640.hex")
    float_to_q16_16_hex(conv_test.weight, "tb_weight_3x3.hex")
    float_to_q16_16_hex(golden_output, "tb_output_golden.hex")

if __name__ == "__main__":
    main()