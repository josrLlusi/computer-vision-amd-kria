# Accuracy of the int8 / fixed-point YOLO26 head vs float

Setup: real YOLO26n head dimensions (ch 64/128/256, nc 80, max_det 300), imgsz 160, 16 random heads/inputs, heavy-tailed (Student-t) features, sparse per-cell activity, class weights centred and scaled x40 (logit range about -15..+3), bias set so ~0.5% of cells exceed score 0.82. Float head = reference.
**Random weights: this measures numerical fidelity of the quantised pipeline, not detection mAP.**

## Main configuration (max-abs calibration, FB=5)

| metric | value |
|---|---|
| class-logit RMSE (logit units) | 0.0548 |
| class-logit max abs error | 0.440 |
| class-logit correlation (worst scale/trial) | 0.99903 |
| box RMSE (pixels) | 2.756 |
| box max abs error (pixels) | 24.83 |
| float top-50 detections present in int8 top-300 (same class, IoU>=0.5) | 0.511 |
| float top-1 detection present in int8 top-5 | 0.375 |
| score >= 0.7: float / int8 / matched | 578 / 552 / 317 |
| score >= 0.7: precision / recall / F1 vs float | 0.5743 / 0.5484 / 0.5611 |
| score >= 0.7: mean / min IoU of matched | 0.8806 / 0.5014 |
| score >= 0.7: mean abs score error | 0.0075 |
| score >= 0.25: matched / float (cutoff-churn dominated, see note) | 2296 / 4265 (0.538) |

## Ablations (first 6 inputs)

| config | logit RMSE | box RMSE px | P@0.7 | R@0.7 | F1@0.7 | mean IoU@0.7 | top50 in 300 |
|---|---|---|---|---|---|---|---|
| calib max-abs (default) | 0.0569 | 2.674 | 0.7579 | 0.7236 | 0.7404 | 0.9342 | 0.633 |
| calib 99.99 pct | 0.0586 | 2.922 | 0.7419 | 0.6935 | 0.7169 | 0.9309 | 0.603 |
| calib 99.9 pct | 0.0757 | 6.278 | 0.7824 | 0.6683 | 0.7209 | 0.8983 | 0.580 |
| box FB=4 | 0.0569 | 2.693 | 0.7579 | 0.7236 | 0.7404 | 0.9344 | 0.633 |
| box FB=6 | 0.0569 | 2.664 | 0.7579 | 0.7236 | 0.7404 | 0.9348 | 0.630 |

## Sensitivity to how heavy-tailed / sparse the activations are (8 inputs each)

| gain sigma | act max / p99 | logit RMSE | logit RMSE / logit std | box RMSE px | F1@0.7 | recall@0.7 |
|---|---|---|---|---|---|---|
| 0.0 | 2.7 | 0.0357 | 0.0158 | 0.399 | 0.899 | 0.899 |
| 0.3 | 2.7 | 0.0358 | 0.0159 | 0.474 | 0.895 | 0.895 |
| 0.6 | 3.1 | 0.0364 | 0.0161 | 0.746 | 0.863 | 0.858 |
| 0.9 | 3.6 | 0.0399 | 0.0176 | 1.388 | 0.743 | 0.749 |
| 1.2 | 4.3 | 0.0565 | 0.0245 | 2.939 | 0.424 | 0.440 |

Note: at score >= 0.25 every image has >= 300 candidate rows, so which rows land just inside the 300-row cutoff is decided by tiny score differences; that metric mostly measures cutoff churn, not quantisation error. The >= 0.7 and top-50-in-300 rows are the meaningful ones.
