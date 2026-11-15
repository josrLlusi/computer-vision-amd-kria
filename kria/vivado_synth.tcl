# UNTESTED (no Vivado in the authoring sandbox). Out-of-context synth + place + route for the K26, 200 MHz target.
#   cd <project root>;  vivado -mode batch -source kria/vivado_synth.tcl
# Part: Kria K26 SOM = XCK26-SFVC784-2LV-C  (check against your board: KV260 and KR260 both use the K26).
set part  xck26-sfvc784-2LV-c
set top   yolo26_head_kria_top
set_part $part
read_verilog [list rtl/ram_tdp.v rtl/topk_sorter.v rtl/conv_engine.v rtl/det_post.v \
                   rtl/yolo26_head_core.v rtl/yolo26_head_axil.v rtl/yolo26_head_kria_top.v]
# activations default to UltraRAM; to force BRAM add:  -verilog_define {ACT_RAM_STYLE="block"}
synth_design -top $top -part $part -mode out_of_context
create_clock -period 5.000 -name aclk [get_ports aclk]
opt_design
place_design
route_design
file mkdir kria/reports
report_utilization    -file kria/reports/utilization.rpt
report_timing_summary -file kria/reports/timing.rpt -delay_type min_max -max_paths 20
report_ram_utilization -file kria/reports/ram.rpt
write_checkpoint -force kria/reports/post_route.dcp
puts "WNS/TNS are in kria/reports/timing.rpt. If WNS < 0 at 5 ns, first suspects: UltraRAM read path, 51-bit requant multiply, 300-entry sorter compare."
