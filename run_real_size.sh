#!/bin/sh
# Real-size YOLO26n head dimensions (ch 64/128/256, nc 80, max_det 300), random weights, RTL vs golden.
#   ./run_real_size.sh [imgsz=160] [LANES=16]      (imgsz must be a multiple of 32; sim time grows ~ imgsz^2)
set -e
cd "$(dirname "$0")"
IMG=${1:-160}; LANES=${2:-16}; LOGL=$(python3 -c "import math;print(int(math.log2($LANES)))")
d=sim/real${IMG}_L${LANES}; K=300
lg() { python3 -c "import math;print(max($2,math.ceil(math.log2($1+1))))"; }
python3 py/gen_test.py $d --seed 7 --grid $((IMG/8)) --ch 64 128 256 --nc 80 --maxdet $K --lanes $LANES
j() { python3 -c "import json;print(json.load(open('$d/meta.json'))['$1'])"; }
iverilog -g2012 -o $d/tb.vvp -Ptb_head.LANES=$LANES -Ptb_head.LOGL=$LOGL \
  -Ptb_head.AAW=$(lg $(j act_words) 6) -Ptb_head.WAW=$(lg $(j n_wgt) 6) -Ptb_head.PAW=$(lg $(j n_par) 6) \
  -Ptb_head.LAW=$(lg $(( $(j n_lut) + 256 )) 8) -Ptb_head.PRW=$(lg $(j n_prog) 6) -Ptb_head.BAW=$(lg $(j box_words) 6) \
  -Ptb_head.K=$K -Ptb_head.OUTAW=$(lg $((6*K+6)) 8) \
  rtl/ram_tdp.v rtl/topk_sorter.v rtl/conv_engine.v rtl/det_post.v rtl/yolo26_head_core.v tb/tb_head.v
AW_=$(j act_words); BW_=$(j box_words)
(cd $d && vvp -n tb.vvp +actwords=$AW_ +boxwords=$BW_ +maxcyc=400000000 | grep -E "CYCLES|ROWS|TIMEOUT")
python3 py/check.py $d
