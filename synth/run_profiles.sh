#!/bin/sh
# Footprint profiles on the K26 (Yosys estimate!). Needs sim/size40/meta.json (python py/gen_test.py sim/size40 --grid 40 --ch 64 128 256 --nc 80 --lanes 16)
cd "$(dirname "$0")/.."
run() {  # name lanes K yxw cw
  python3 py/make_top.py sim/size40/meta.json --lanes $2 --K $3 --yxw $4 --cw $5 --out synth/top_$1.v > /dev/null
  cat > synth/p_$1.ys <<EOT
read_verilog -sv rtl/ram_tdp.v rtl/topk_sorter.v rtl/conv_engine.v rtl/det_post.v rtl/yolo26_head_core.v rtl/yolo26_head_axil.v synth/top_$1.v
hierarchy -check -top yolo26_head_kria_top
synth_xilinx -top yolo26_head_kria_top -family xcup -flatten
tee -o synth/stat_$1.txt stat
EOT
  yosys -q synth/p_$1.ys > /dev/null 2>&1
  echo "$1 (LANES=$2 K=$3 YXW=$4 CW=$5): $(python3 py/parse_stat.py synth/stat_$1.txt)" >> synth/profiles.txt
}
rm -f synth/profiles.txt
run balanced 16 100 8 7
run lean      8 100 8 7
echo DONE >> synth/profiles.txt
