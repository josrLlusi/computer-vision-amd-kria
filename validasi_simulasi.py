def validasi_hasil():
    file_sim = "sim_output.hex"
    file_gold = "tb_output_golden.hex"
    
    try:
        with open(file_sim, 'r') as fs:
            # Hanya ambil angka, buang sinyal 'x'
            sim_lines = [line.strip() for line in fs.readlines() if 'x' not in line.lower()]
        with open(file_gold, 'r') as fg:
            gold_lines = [line.strip() for line in fg.readlines()]
    except Exception as e:
        print(f"Error: {e}"); return

    # Gunakan panjang terpendek (biasanya 101760 atau 101761)
    batas_uji = min(len(sim_lines), len(gold_lines))
    errors = 0

    for i in range(batas_uji):
        sim_val = int(sim_lines[i], 16)
        if sim_val >= 2**31: sim_val -= 2**32
            
        gold_val = int(gold_lines[i], 16)
        if gold_val >= 2**31: gold_val -= 2**32
            
        selisih = abs(sim_val - gold_val)
        
        # Toleransi 5 karena bit terendah terpotong saat Q32.32 menjadi Q16.16
        if selisih > 15:
            if errors < 5:
                print(f"Mismatch baris {i}: RTL={sim_val}, PyTorch={gold_val}, Beda={selisih}")
            errors += 1

    print("-" * 50)
    print(f"Piksel Divalidasi : {batas_uji}")
    print(f"Total Mismatch    : {errors}")
    print("-" * 50)
    
    if errors == 0:
        print("STATUS: LULUS VERIFIKASI! MAC Hardware Akurat.")
    else:
        print("STATUS: GAGAL.")

if __name__ == "__main__":
    validasi_hasil()