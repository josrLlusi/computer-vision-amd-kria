# Hapus library kerja lama jika ada, lalu buat yang baru
if {[file exists work]} {
    vdel -lib work -all
}
vlib work

# Kompilasi desain utama dan testbench (menggunakan standar SystemVerilog)
vlog -sv conv3x3_s2.sv
vlog -sv tb_conv3x3.sv

# Muat testbench ke dalam simulator dengan optimasi yang mengizinkan pengamatan sinyal
vsim -voptargs="+acc" work.tb_conv3x3

# Konfigurasi Jendela Gelombang (Waveform)
view wave
delete wave *

# Tambahkan sinyal kontrol
add wave -color "Yellow" /tb_conv3x3/clk
add wave -color "Orange" /tb_conv3x3/rst_n

# Tambahkan sinyal input (Format Hexadecimal)
add wave -divider "Input Aliran Piksel"
add wave -color "Cyan" /tb_conv3x3/valid_in
add wave -radix hexadecimal /tb_conv3x3/pixel_in

# Tambahkan sinyal output (Format Hexadecimal)
add wave -divider "Output Konvolusi"
add wave -color "Green" /tb_conv3x3/valid_out
add wave -radix hexadecimal /tb_conv3x3/pixel_out

# (Opsional) Intip isi FSM di dalam modul untuk debugging
add wave -divider "Internal FSM"
add wave -radix unsigned /tb_conv3x3/dut/pixel_x
add wave -radix unsigned /tb_conv3x3/dut/pixel_y

# Jalankan simulasi selama 5 milidetik (cukup untuk memproses sebagian gambar)
run 5 ms

# Zoom penuh agar seluruh gelombang terlihat
wave zoom full