#!/bin/sh
# Self-checking regression: random quantised heads, RTL vs bit-exact golden model.
#   ./run_tests.sh            LANES=16 (default)   ./run_tests.sh 4     other LANES (power of 2, >= 4)
set -e
cd "$(dirname "$0")"
LANES=${1:-16}; LOGL=$(python3 -c "import math;print(int(math.log2($LANES)))")
mkdir -p sim
SRC="rtl/ram_tdp.v rtl/topk_sorter.v rtl/conv_engine.v rtl/det_post.v rtl/yolo26_head_core.v"
lg() { python3 -c "import math;print(max($2,math.ceil(math.log2($1+1))))"; }
run() {  # name K seed grid "ch" nc  (env HF = height/width ratio)
  name=$1; K=$2; seed=$3; grid=$4; ch=$5; nc=$6; d=sim/L${LANES}_$name
  python3 py/gen_test.py $d --seed $seed --grid $grid --ch $ch --nc $nc --maxdet $K --lanes $LANES --hfrac ${HF:-1.0}
  j() { python3 -c "import json;print(json.load(open('$d/meta.json'))['$1'])"; }
  iverilog -g2012 -o $d/tb.vvp -Ptb_head.LANES=$LANES -Ptb_head.LOGL=$LOGL \
      -Ptb_head.AAW=$(lg $(j act_words) 6) -Ptb_head.WAW=$(lg $(j n_wgt) 6) -Ptb_head.PAW=$(lg $(j n_par) 6) \
      -Ptb_head.LAW=$(lg $(( $(j n_lut) + 256 )) 8) -Ptb_head.PRW=$(lg $(j n_prog) 6) -Ptb_head.BAW=$(lg $(j box_words) 6) \
      -Ptb_head.K=$K -Ptb_head.OUTAW=$(lg $((6*K+6)) 8) $XP $SRC tb/tb_head.v
  AW_=$(j act_words); BW_=$(j box_words)
  (cd $d && vvp -n tb.vvp +actwords=$AW_ +boxwords=$BW_ +maxcyc=40000000 | grep -E "CYCLES|ROWS|TIMEOUT")
  python3 py/check.py $d
}
echo "== LANES=$LANES test A: K=16 (k < cells), nc=5";             run tA 16  1 8  "8 12 16" 5
echo "== LANES=$LANES test B: K=300 (k = all 84 cells), nc=3";    run tB 300 2 8  "8 16 16" 3
echo "== LANES=$LANES test C: K=40, nc=7, bigger grid, Cin%L!=0"; run tC 40  3 16 "12 16 24" 7
XP="-Ptb_head.YXW=5 -Ptb_head.CW=3 -Ptb_head.IW=6"
echo "== LANES=$LANES test D: compact sorter widths (YXW=5, CW=3, IW=6), K=40, nc=7"; run tD 40 4 16 "12 16 24" 7
XP=""
HF=0.75
echo "== LANES=$LANES test E: NON-SQUARE 4:3 grids (as 640x480), K=40, nc=7"; run tE 40 5 16 "12 16 24" 7
HF=1.0

# ---- PS-style test: same networks, but loaded/started/read purely through the AXI4-Lite slave
axil() {
  name=$1; K=$2; d=sim/L${LANES}_$name
  j() { python3 -c "import json;print(json.load(open('$d/meta.json'))['$1'])"; }
  iverilog -g2012 -o $d/tbax.vvp -Ptb_axil.LANES=$LANES -Ptb_axil.LOGL=$LOGL \
      -Ptb_axil.AAW=$(lg $(j act_words) 6) -Ptb_axil.WAW=$(lg $(j n_wgt) 6) -Ptb_axil.PAW=$(lg $(j n_par) 6) \
      -Ptb_axil.LAW=$(lg $(( $(j n_lut) + 256 )) 8) -Ptb_axil.PRW=$(lg $(j n_prog) 6) -Ptb_axil.BAW=$(lg $(j box_words) 6) \
      -Ptb_axil.K=$K -Ptb_axil.OUTAW=$(lg $((6*K+6)) 8) \
      rtl/ram_tdp.v rtl/topk_sorter.v rtl/conv_engine.v rtl/det_post.v rtl/yolo26_head_core.v rtl/yolo26_head_axil.v tb/tb_axil.v
  NP=$(j n_prog); NR=$(j n_par); NW=$(j n_wgt); NL=$(j n_lut); NI=$(j n_input_words); NA=$(j act_words)
  (cd $d && vvp -n tbax.vvp +nprog=$NP +npar=$NR +nwgt=$NW +nlut=$NL +nin=$NI +nact=$NA | grep -E "ID|CFG|loaded|irq|CYCLES|ROWS|TIMEOUT")
  python3 py/check.py $d --nobox
}
echo "== LANES=$LANES AXI4-Lite (PS-style) test on A"; axil tA 16
echo "== LANES=$LANES AXI4-Lite (PS-style) test on C"; axil tC 40
