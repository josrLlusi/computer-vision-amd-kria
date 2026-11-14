# YOLO26 Detect Head in RTL for the AMD Kria K26 (one2one / NMS-free path)

Verilog implementation of the YOLO26 detect head's inference path, with a bit-exact Python golden model, simulation
testbenches (direct and AXI4-Lite "PS-style"), a quantiser + image-in/image-out flow, an accuracy report, and
(untested) Vivado / PYNQ scripts, and a COCO mAP evaluator.  The backbone + neck run in software; **only the head is RTL**.

```
image -> [software] backbone+neck -> P3/P4/P5 feature maps -> [AXI4-Lite load] -> RTL head -> 300 rows -> [software] draw
```

## Architecture (v2, Kria-oriented)

| File | Role |
|---|---|
| `rtl/conv_engine.v` | Pipelined layer processor, `LANES` (default 16) int8 MACs/cycle. Dense layers: LANES input channels in parallel, 1 output byte per (pixel,cout) through one requantiser. Depthwise: LANES channels in parallel. Per-channel u16 multiplier + 6-bit shift, SiLU as LUT. |
| `rtl/det_post.v`, `rtl/topk_sorter.v` | decode + sigmoid + two-stage top-k (same algorithm as Ultralytics `get_topk_index`), 300-entry streaming sorters replace NMS. |
| `rtl/ram_tdp.v` | True dual-port byte-enable RAM (Vivado TDP template, exact depth). `ram_tdp_act` = activations, UltraRAM by default. |
| `rtl/yolo26_head_core.v` | engines + all RAMs + 32-bit host port (7 windows). |
| `rtl/yolo26_head_axil.v` | AXI4-Lite slave: CTRL/STATUS/N_ROWS/CYCLES/ID/CFG regs + memory windows (map in file header). |
| `rtl/yolo26_head_kria_top.v` | generated fixed-parameter top (`py/make_top.py`) for the block design / synthesis. |
| `kria/vivado_synth.tcl`, `kria/kria_host.py` | **untested** Vivado OOC flow and PYNQ driver. |
| `v1_reference/` | the first (1 MAC/cycle) design, kept for reference. |

Memory layout: activations HWC, channels padded to a multiple of LANES, one RAM word = LANES int8; box distances in a
separate 16-bit RAM. Number formats: int8 activations/weights, int16 box (FB=5 fractional bits), one logit scale shared
by all 3 scales, output coordinates in 1/4 pixel, score u16.

## Run

```sh
./run_tests.sh [LANES]        # 3 random heads direct + 2 through AXI4-Lite, RTL vs golden (iverilog, numpy)
./run_real_size.sh 160 16     # real YOLO26n dims (ch 64/128/256, nc 80, K=300), random weights  (slow in Icarus)
cd py
python run_image.py --synthetic --imgsz 128 --out ../out.png          # image flow, random head
python run_image.py --image cat.jpg --weights yolo26n.pt --imgsz 160  # real model (needs ultralytics; UNTESTED adapter)
python accuracy_report.py --trials 16                                  # -> accuracy_report.md
yosys synth/synth_yosys.ys                                             # resource estimate (UltraScale+)
```

## Checking an input image against the output image

```sh
cd py
python run_image.py --image cat.jpg --weights yolo26n.pt --imgsz 640 --sim golden --out out.png     # fast: numpy model
python run_image.py --image cat.jpg --weights yolo26n.pt --imgsz 160 --sim rtl    --out out.png     # real RTL in Icarus (slow)
```
Pass criteria, in order:
1. `RTL rows == bit-exact golden rows: YES` (only with `--sim rtl`) - the Verilog does exactly what the Python model says.
2. Per-scale lines: int8 logit/box correlation with the float head close to 1.
3. `CHECK vs Ultralytics predict()`: most RTL detections matched (same class, IoU >= 0.5), mean IoU high.
4. `out.png`: green boxes (RTL int8 head) on top of red boxes (Ultralytics) - they should coincide.
On the board, `kria/kria_host.py` returns the rows read over AXI-Lite; they must equal the golden rows for the same data files.
Only (1) is bit-exact; (2)-(4) measure quantisation error and need the real model. Whole-dataset accuracy = `coco_eval.py`.

## Non-square inputs (640x480, 480p)

The head accepts any feature-map shapes (H and W are independent in every descriptor; tested bit-exact with 4:3 grids). The
network input sides must be multiples of 32: **640x480 works** (feature maps 80x60, 40x30, 20x15). True 480p 854x480 does not:
use 864x480 (padded) or resize to e.g. 640x384. Pass `--imgsz 640x480` (WxH) to run_image.py / coco_eval.py (the adapter then
letterboxes to 640x480 instead of padding to a square). The adapter's rectangular path is UNTESTED with real Ultralytics.

| input WxH | activation RAM (16 lanes) | URAM288 | of 64 |
|---|---|---|---|
| 640x640 | 2356 KB | 74 | 116% (does not fit) |
| **640x480** | 1767 KB | 56 | 88% (leaves little room) |
| 640x384 | 1414 KB | 46 | 72% |
| 512x384 | 1131 KB | 36 | 56% |
| 416x320 | 766 KB | 24 | 38% |

## Footprint on the K26 (so the head leaves room for the backbone/neck)

Yosys estimates (`synth_xilinx -family xcup`, YOLO26n head @320 px). **Yosys is not Vivado**: expect different numbers after
place & route, and check timing there. Profiles are generated with `py/make_top.py` and measured by `synth/run_profiles.sh`.

| profile | LUT | FF | DSP | BRAM36 | URAM288 |
|---|---|---|---|---|---|
| full: K=300, 16 lanes | 41.5K (35%) | 20.8K (9%) | 40 (3%) | 57.5 (40%) | 36 (56%) |
| **balanced** (default top): K=100, 16 lanes, YXW=8, CW=7 | 18.1K (15%) | 8.0K (3%) | 40 (3%) | 56.5 (39%) | 36 (56%) |
| lean: K=100, 8 lanes | 14.5K (12%) | 7.5K (3%) | 32 (3%) | 36.5 (25%) | 18 (28%) |

* Logic is small; **memory is the cost**. Knobs: `--K` (COCO scores at most 100 detections/image, so K=100 should not change COCO mAP
  - the COCO report has a "float head, K=100" row to confirm; on random synthetic heads the top-100 rows were identical), `--yxw/--cw`
  (sorter payload bits), `--lanes`, and the image size.
* Weights are ~148 KB for the n head (~33 BRAM36, fixed). The s/m/l/x heads have ~4x or more weights: put them in URAM or DDR.
* Activation RAM (URAM288, estimate): 224 px 289 KB = 10 URAM (16%); 320 px 589 KB = 20 (31%); 416 px 996 KB = 32 (50%);
  512 px 1.5 MB = 48 (75%); 640 px 2.36 MB does not fit in 64 URAM -> would need DDR streaming (not implemented).
* Integration: the neck's P3/P4/P5 outputs should be written straight into the head's input region of the activation RAM
  (host port, while the head is idle) instead of a second copy; the head's scratch/output regions can then be the only extra storage.
  Not implemented here: you must wire the neck's writer to that port. The head runs after the neck finishes a frame.

## Verified (authoring sandbox)

* RTL == golden, bit for bit (every activation word, box word, output row): random networks x LANES = 4, 8, 16, 32, a compact-width
  test (YXW=5, CW=3, IW=6), and 2 networks loaded/started/read purely over AXI4-Lite. 7 injected bugs are each caught.
* Real-size run (`./run_real_size.sh 160 16`): YOLO26n head dims (ch 64/128/256, nc 80, K=300), random weights, 160 px: all memory
  words and 300 rows bit-exact; 1,411,849 cycles (~7.1 ms at an unproven 200 MHz).
* Channel counts from `yolo26.yaml`: Detect inputs n 64/128/256, s 128/256/512, m/l 256/512/512, x 384/768/768 (my reading of `parse_model`).

## NOT verified

* **Nothing has run in Vivado or on a Kria** (no Fmax, routed utilisation, UltraRAM timing, power). 200 MHz is unproven.
* `py/ultra_adapter.py`, `py/coco_eval.py` on real weights/COCO, `kria/*`: never run (no torch, no COCO in the sandbox).
  `coco_eval.py --selftest` only proves the evaluation plumbing works with pycocotools.
* **Real accuracy is unmeasured.** `accuracy_report.md` uses random weights and synthetic activations (I had to reshape them -
  a plain random head scores every class alike, so detection metrics were meaningless). It shows numerical behaviour only: int8
  per-tensor PTQ degrades when activations are sparse/heavy-tailed. COCO mAP needs `coco_eval.py` on the real model.
* Host access to the activation RAM works only while the core is idle; WSTRB ignored.

## COCO accuracy (what you will run)

```sh
pip install ultralytics pycocotools opencv-python
python py/coco_eval.py --weights yolo26n.pt --coco /data/coco --imgsz 640 --calib 128 --n 500 --lanes 16 --K 100 --ul-ref
```
Calibrates the int8 scales on 128 val2017 images taken from the END of the list, evaluates 500 DIFFERENT images, reports
mAP50-95/mAP50 for: float head (numpy, K=300), float head at K=100, int8 head (RTL-equivalent) and Ultralytics' own `predict`
(sanity: your float row should be close to it and to `yolo val`). Use more calibration images / all 5000 for final numbers.
If the int8 drop is too large, the levers are: calibration set size, percentile (`--pct`), more fractional bits, per-channel
activation scales, wider intermediates, or QAT.

## Memory vs image size, performance

At 160 px the n head takes ~1.4 M cycles (16 lanes). ~19 M MAC/frame @160 px, ~300 M @640 px; raise `LANES` for throughput
(costs URAM/BRAM width, not DSPs).
