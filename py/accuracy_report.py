"""Accuracy statistics of the int8 / fixed-point head (what the RTL implements) vs the float head.

The RTL is bit-exact with golden.py (see run_tests.sh), so the golden model is used here for speed.
Head = real YOLO26n dimensions (ch 64/128/256, nc 80, max_det 300) with RANDOM weights and heavy-tailed random features.
=> measures NUMERICAL FIDELITY of the quantised pipeline, NOT mAP. For mAP run ultra_adapter.py on real weights + COCO.
usage: python accuracy_report.py [--trials 8] [--imgsz 160] [--out accuracy_report.md]
"""
import argparse, json, sys, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import quantize as q, golden as g
from run_image import synthetic

N_CH = (64, 128, 256); NC = 80; THR = 0.25


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(rows_f, rows_q, THR=THR):
    """float rows = reference. greedy match by score, same class, IoU>=0.5."""
    F = [r for r in rows_f if r[4] >= THR]; Q = [r for r in rows_q if r[4] >= THR]
    used = set(); tp = 0; ious = []; serr = []
    for qr in sorted(Q, key=lambda r: -r[4]):
        best, bj = 0.0, -1
        for j, fr in enumerate(F):
            if j in used or int(fr[5]) != int(qr[5]):
                continue
            v = iou(qr[:4], fr[:4])
            if v > best: best, bj = v, j
        if best >= 0.5:
            used.add(bj); tp += 1; ious.append(best); serr.append(abs(qr[4] - F[bj][4]))
    return len(F), len(Q), tp, ious, serr


def make(seed, imgsz, frac=0.005, target=1.5, gain_sigma=1.2, w_scale=40.0):
    head, feats, nc, names, rgb, _ = synthetic(imgsz, NC, seed=seed, ch=N_CH, heavy=True)
    rng = np.random.default_rng(seed + 99)
    # real feature maps are sparse: a few "object" cells are strongly active, most are background.
    feats = [f * np.exp(rng.normal(0, gain_sigma, (1,) + f.shape[1:])) for f in feats]
    # lift the final class bias so that ~frac of cells have a confidently positive logit (a "realistic" score spread)
    for s in range(3):                       # remove the class-common mode so a cell's classes are distinct (like a real head)
        L = head['cls'][s][-1]
        L['w'] = w_scale * (L['w'] - L['w'].mean(axis=0, keepdims=True)); L['b'] = L['b'] - L['b'].mean()
    best = np.concatenate([q.fchain(head['cls'][s], feats[s])[-1][1].max(axis=0).ravel() for s in range(3)])
    delta = target - np.quantile(best, 1 - frac)
    for s in range(3):
        head['cls'][s][-1]['b'] = head['cls'][s][-1]['b'] + delta
    return head, feats


def trial(head, feats, FB=5, pct=100.0, lanes=16, max_det=300):
    P, info = q.quantize_head(head, feats, NC, FB=FB, pct=pct, lanes=lanes)
    act, box = g.golden_layers(P)
    cls_q, box_q = q.int_outputs(P, info, act, box, FB)
    res = dict(cls_rmse=[], cls_max=[], cls_corr=[], box_rmse_px=[], box_max_px=[])
    for s in range(3):
        fl, fb_ = info['float_cls'][s], info['float_box'][s]; st = 2 ** (3 + s)
        res['cls_rmse'].append(float(np.sqrt(np.mean((cls_q[s] - fl) ** 2)))); res['cls_max'].append(float(np.abs(cls_q[s] - fl).max()))
        res['cls_corr'].append(float(np.corrcoef(cls_q[s].ravel(), fl.ravel())[0, 1]))
        res['box_rmse_px'].append(float(np.sqrt(np.mean((box_q[s] - fb_) ** 2)) * st)); res['box_max_px'].append(float(np.abs(box_q[s] - fb_).max() * st))
    rows_f = q.float_post(info['float_box'], info['float_cls'], max_det)
    rows_q = q.dequant_rows(g.post(act, box, P, max_det))
    for tag, thr in (('', 0.25), ('_hi', 0.7)):
        nf, nq, tp, ious, serr = match(rows_f, rows_q, thr)
        res.update({'nF' + tag: nf, 'nQ' + tag: nq, 'tp' + tag: tp, 'iou' + tag: ious, 'serr' + tag: serr})
    found = lambda a, rows: any(int(a[5]) == int(b[5]) and iou(a[:4], b[:4]) >= 0.5 for b in rows)
    res['top1_in5'] = found(rows_f[0], rows_q[:5])
    res['top50_in300'] = sum(found(a, rows_q) for a in rows_f[:50]) / 50.0
    return res


def agg(rs):
    out = agg1(rs, '')
    hi = agg1(rs, '_hi')
    out.update({k + '_hi': v for k, v in hi.items() if k in ('nF', 'nQ', 'tp', 'precision', 'recall', 'f1', 'mean_iou', 'min_iou', 'score_err')})
    out['top1_in5'] = float(np.mean([r['top1_in5'] for r in rs])); out['top50_in300'] = float(np.mean([r['top50_in300'] for r in rs]))
    return out


def agg1(rs, t):
    tp = sum(r['tp' + t] for r in rs); nf = sum(r['nF' + t] for r in rs); nq = sum(r['nQ' + t] for r in rs)
    ious = [v for r in rs for v in r['iou' + t]]; serr = [v for r in rs for v in r['serr' + t]]
    prec = tp / nq if nq else float('nan'); rec = tp / nf if nf else float('nan')
    return dict(cls_rmse=np.mean([np.mean(r['cls_rmse']) for r in rs]), cls_max=np.max([max(r['cls_max']) for r in rs]),
                cls_corr=np.min([min(r['cls_corr']) for r in rs]), box_rmse_px=np.mean([np.mean(r['box_rmse_px']) for r in rs]),
                box_max_px=np.max([max(r['box_max_px']) for r in rs]), nF=nf, nQ=nq, tp=tp, precision=prec, recall=rec,
                f1=(2 * prec * rec / (prec + rec)) if tp else 0.0, mean_iou=float(np.mean(ious)) if ious else float('nan'),
                min_iou=float(np.min(ious)) if ious else float('nan'), score_err=float(np.mean(serr)) if serr else float('nan'),
                )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=8); ap.add_argument('--imgsz', type=int, default=160)
    ap.add_argument('--abl', type=int, default=3, help='trials used for each ablation config')
    ap.add_argument('--out', default='accuracy_report.md')
    a = ap.parse_args(); t0 = time.time()
    data = [make(1000 + i, a.imgsz) for i in range(a.trials)]
    main = agg([trial(h, f) for h, f in data]); print('main config done', round(time.time() - t0), 's', flush=True)
    abl = {}
    for name, kw in [('calib max-abs (default)', dict(pct=100.0)), ('calib 99.99 pct', dict(pct=99.99)), ('calib 99.9 pct', dict(pct=99.9)),
                     ('box FB=4', dict(FB=4)), ('box FB=6', dict(FB=6))]:
        abl[name] = agg([trial(h, f, **kw) for h, f in data[:a.abl]]); print(name, 'done', round(time.time() - t0), 's', flush=True)
    sens = []
    for gs in (0.0, 0.3, 0.6, 0.9, 1.2):
        rs = []; rel = []; ratio = []
        for i in range(8):
            h, f = make(2000 + i, a.imgsz, gain_sigma=gs)
            ratio.append(np.mean([np.abs(x).max() / np.percentile(np.abs(x), 99) for x in f]))
            r = trial(h, f); rs.append(r)
            lg = np.concatenate([c.ravel() for c in q.fchain(h['cls'][0], f[0])[-1][1:]]) if False else None
            P_, info_ = q.quantize_head(h, f, NC); lgs = np.concatenate([c.ravel() for c in info_['float_cls']])
            rel.append(np.mean(r['cls_rmse']) / lgs.std())
        ag = agg(rs); sens.append((gs, np.mean(ratio), ag['cls_rmse'], np.mean(rel), ag['box_rmse_px'], ag['f1_hi'], ag['recall_hi']))
    L = [f'# Accuracy of the int8 / fixed-point YOLO26 head vs float\n',
         f'Setup: real YOLO26n head dimensions (ch 64/128/256, nc 80, max_det 300), imgsz {a.imgsz}, {a.trials} random heads/inputs, '
         f'heavy-tailed (Student-t) features, sparse per-cell activity, class weights centred and scaled x40 (logit range about -15..+3), bias set so ~0.5% of cells exceed score 0.82. Float head = reference.',
         f'**Random weights: this measures numerical fidelity of the quantised pipeline, not detection mAP.**\n',
         '## Main configuration (max-abs calibration, FB=5)\n', '| metric | value |', '|---|---|',
         f"| class-logit RMSE (logit units) | {main['cls_rmse']:.4f} |", f"| class-logit max abs error | {main['cls_max']:.3f} |",
         f"| class-logit correlation (worst scale/trial) | {main['cls_corr']:.5f} |",
         f"| box RMSE (pixels) | {main['box_rmse_px']:.3f} |", f"| box max abs error (pixels) | {main['box_max_px']:.2f} |",
         f"| float top-50 detections present in int8 top-300 (same class, IoU>=0.5) | {main['top50_in300']:.3f} |",
         f"| float top-1 detection present in int8 top-5 | {main['top1_in5']:.3f} |",
         f"| score >= 0.7: float / int8 / matched | {main['nF_hi']} / {main['nQ_hi']} / {main['tp_hi']} |",
         f"| score >= 0.7: precision / recall / F1 vs float | {main['precision_hi']:.4f} / {main['recall_hi']:.4f} / {main['f1_hi']:.4f} |",
         f"| score >= 0.7: mean / min IoU of matched | {main['mean_iou_hi']:.4f} / {main['min_iou_hi']:.4f} |",
         f"| score >= 0.7: mean abs score error | {main['score_err_hi']:.4f} |",
         f"| score >= 0.25: matched / float (cutoff-churn dominated, see note) | {main['tp']} / {main['nF']} ({main['recall']:.3f}) |", '', f'## Ablations (first {a.abl} inputs)\n',
         '| config | logit RMSE | box RMSE px | P@0.7 | R@0.7 | F1@0.7 | mean IoU@0.7 | top50 in 300 |', '|---|---|---|---|---|---|---|---|']
    for k, v in abl.items():
        L.append(f"| {k} | {v['cls_rmse']:.4f} | {v['box_rmse_px']:.3f} | {v['precision_hi']:.4f} | {v['recall_hi']:.4f} | {v['f1_hi']:.4f} | {v['mean_iou_hi']:.4f} | {v['top50_in300']:.3f} |")
    L += ['', '## Sensitivity to how heavy-tailed / sparse the activations are (8 inputs each)\n',
          '| gain sigma | act max / p99 | logit RMSE | logit RMSE / logit std | box RMSE px | F1@0.7 | recall@0.7 |', '|---|---|---|---|---|---|---|']
    for gs, rt, rm, rl, bx, f1, rc in sens:
        L.append(f'| {gs:.1f} | {rt:.1f} | {rm:.4f} | {rl:.4f} | {bx:.3f} | {f1:.3f} | {rc:.3f} |')
    L.append('\nNote: at score >= 0.25 every image has >= 300 candidate rows, so which rows land just inside the 300-row cutoff is decided by tiny score differences; that metric mostly measures cutoff churn, not quantisation error. The >= 0.7 and top-50-in-300 rows are the meaningful ones.')
    open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
