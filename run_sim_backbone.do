# ==========================================================================
# run_sim_backbone.do - Simulasi conv_kxk / c3k2 / backbone_l0_l5 di ModelSim
# Jalankan dari root repo SETELAH: python ekstrak_uji_backbone.py --mode ...
#   GUI   : do run_sim_backbone.do
#   Batch : vsim -c -do run_sim_backbone.do
# Uji stres: BUBBLE 20, BACKPRESSURE 50, FRAMES 2
# ==========================================================================
set BUBBLE       0
set BACKPRESSURE 0
set FRAMES       1

if {[file exists work]} {
    vdel -lib work -all
}
vlib work

vlog -sv +incdir+. conv_kxk.sv stream_fifo.sv c3k2.sv backbone_l0_l5.sv
vlog -sv +incdir+. tb_backbone.sv

vsim -voptargs="+acc" work.tb_backbone +BUBBLE=$BUBBLE +BACKPRESSURE=$BACKPRESSURE +FRAMES=$FRAMES

if {![batch_mode]} {
    view wave
    add wave -color "Yellow" /tb_backbone/clk
    add wave -color "Orange" /tb_backbone/rst_n
    add wave -divider "Input"
    add wave -color "Cyan" /tb_backbone/in_valid
    add wave -color "Cyan" /tb_backbone/in_ready
    add wave -radix hexadecimal /tb_backbone/in_data
    add wave -divider "Out1 (conv / c3k2 / P3)"
    add wave -color "Green" /tb_backbone/o1_valid
    add wave -color "Green" /tb_backbone/o1_ready
    add wave -radix hexadecimal /tb_backbone/o1_data
    add wave -radix unsigned /tb_backbone/o1_rcv
    add wave -radix unsigned /tb_backbone/o1_mis
    add wave -divider "Out2 (layer 5, mode chain)"
    add wave -color "Green" /tb_backbone/o2_valid
    add wave -radix unsigned /tb_backbone/o2_rcv
    add wave -radix unsigned /tb_backbone/o2_mis
}

run -all

if {![batch_mode]} {
    catch {wave zoom full}
} else {
    quit -f
}
