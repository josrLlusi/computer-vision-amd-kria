#!/bin/sh
# Self-checking regression: random quantized heads, RTL vs bit-exact golden model.
set -e
cd "$(dirname "$0")"
mkdir -p sim
run() {  # name K seed grid "ch" nc
  name=$1; K=$2; seed=$3; grid=$4; ch=$5; nc=$6
  python3 py/gen_test.py sim/$name --seed $seed --grid $grid --ch $ch --nc $nc --maxdet $K > /dev/null
  AW=$(python3 -c "import json,math;m=json.load(open('sim/$name/meta.json'));print(max(10,math.ceil(math.log2(m['act_words']))))")
  WW=$(python3 -c "import json,math;m=json.load(open('sim/$name/meta.json'));print(max(10,math.ceil(math.log2(m['n_wgt']+1))))")
  PW=$(python3 -c "import json,math;m=json.load(open('sim/$name/meta.json'));print(max(10,math.ceil(math.log2(m['n_par']+1))))")
  LW=$(python3 -c "import json,math;m=json.load(open('sim/$name/meta.json'));print(max(8,math.ceil(math.log2(m['n_lut']+256))))")
  PR=$(python3 -c "import json,math;m=json.load(open('sim/$name/meta.json'));print(max(6,math.ceil(math.log2(m['n_prog']+1))))")
  iverilog -g2012 -o sim/$name/tb.vvp -Ptb_head.AAW=$AW -Ptb_head.WAW=$WW -Ptb_head.PAW=$PW -Ptb_head.LAW=$LW \
      -Ptb_head.PRW=$PR -Ptb_head.K=$K -Ptb_head.OUTAW=11 \
      rtl/ram_sync.v rtl/topk_sorter.v rtl/conv_engine.v rtl/det_post.v rtl/yolo26_head_top.v tb/tb_head.v
  ACTW=$(python3 -c "import json;print(json.load(open('sim/$name/meta.json'))['act_words'])")
  (cd sim/$name && vvp -n tb.vvp +actwords=$ACTW +maxcyc=30000000 | grep -E "CYCLES|ROWS|TIMEOUT") 
  python3 py/check.py sim/$name
}
echo "== test A: K=16 (k < cells), nc=5"; run tA 16 1 8 "8 12 16" 5
echo "== test B: K=300 (k = all 84 cells), nc=3"; run tB 300 2 8 "8 16 16" 3
echo "== test C: K=40, nc=7, bigger grid"; run tC 40 3 16 "12 16 24" 7
