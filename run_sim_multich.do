# ==========================================================================
# run_sim_multich.do (v2) - Simulasi conv3x3_s2 v2 di ModelSim
# Jalankan dari root repo SETELAH: python ekstrak_uji_multichannel.py --layer N ...
#   GUI   : vsim -do run_sim_multich.do
#   Batch : vsim -c -do run_sim_multich.do
# Uji ketahanan handshake: ubah BUBBLE / BACKPRESSURE (persen) dan FRAMES di bawah.
# Rekomendasi uji stres: BUBBLE 20, BACKPRESSURE 50, FRAMES 2 (back-pressure ringan
# seperti 20% belum tentu memicu kasus buffer keluaran penuh).
# ==========================================================================
set BUBBLE       0
set BACKPRESSURE 0
set FRAMES       1

if {[file exists work]} {
    vdel -lib work -all
}
vlib work

vlog -sv +incdir+. conv3x3_s2.sv
vlog -sv +incdir+. tb_conv3x3_multich.sv

vsim -voptargs="+acc" work.tb_conv3x3_multich +BUBBLE=$BUBBLE +BACKPRESSURE=$BACKPRESSURE +FRAMES=$FRAMES

if {![batch_mode]} {
    view wave
    add wave -color "Yellow" /tb_conv3x3_multich/clk
    add wave -color "Orange" /tb_conv3x3_multich/rst_n

    add wave -divider "Input (ic0 di LSB)"
    add wave -color "Cyan" /tb_conv3x3_multich/valid_in
    add wave -color "Cyan" /tb_conv3x3_multich/ready_in
    add wave -radix hexadecimal /tb_conv3x3_multich/pixel_in

    add wave -divider "Koordinat & FSM"
    add wave -radix unsigned /tb_conv3x3_multich/dut/pixel_x
    add wave -radix unsigned /tb_conv3x3_multich/dut/pixel_y
    add wave /tb_conv3x3_multich/dut/state
    add wave -radix unsigned /tb_conv3x3_multich/dut/g_ic
    add wave -radix unsigned /tb_conv3x3_multich/dut/g_oc
    add wave /tb_conv3x3_multich/dut/mask_top
    add wave /tb_conv3x3_multich/dut/mask_left

    add wave -divider "Output (oc0 di LSB)"
    add wave -color "Green" /tb_conv3x3_multich/valid_out
    add wave -color "Green" /tb_conv3x3_multich/ready_out
    add wave -radix hexadecimal /tb_conv3x3_multich/pixel_out
    add wave -radix unsigned /tb_conv3x3_multich/out_pix_cnt
    add wave -radix unsigned /tb_conv3x3_multich/mismatches
}

run -all

if {![batch_mode]} {
    catch {wave zoom full}
} else {
    quit -f
}
