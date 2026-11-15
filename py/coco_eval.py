"""COCO mAP of the float head vs the int8 head (= what the RTL computes; RTL is bit-exact with golden.py).

Real run (needs: pip install ultralytics pycocotools opencv-python; COCO val2017 + annotations):
  python coco_eval.py --weights yolo26n.pt --coco /data/coco --imgsz 640 --calib 128 --n 500 --lanes 16 --K 100
Plumbing self-test (no torch / COCO needed; random head; the 'ground truth' is the float head's own detections, so the
numbers only prove the evaluation code path works, they say NOTHING about real accuracy):
  python coco_eval.py --selftest

Protocol: weights quantised ONCE from statistics over --calib images (taken from the END of the image list, disjoint from the
--n evaluation images at the start). Each evaluation image goes: letterbox -> backbone/neck (Ultralytics) -> head. Detections
are mapped back to original-image pixels, score >= 0.001, top 100 per image (COCO's maxDets), as in `yolo val`.
UNTESTED on real data/weights (see README). Use the float row as the sanity anchor: it should be close to `yolo val`.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize as q, golden as g


def parse_size(s):
    """'640' -> 640 (square);  '640x480' (WxH) -> (480, 640) as (height, width). Sides must be multiples of 32."""
    s = str(s).lower()
    if 'x' in s:
        w, h = (int(v) for v in s.split('x'))
        assert w % 32 == 0 and h % 32 == 0, 'sides must be multiples of 32'
        return (h, w)
    return int(s)


def rows_to_coco(rows, image_id, meta, cat_ids, max_det=100, min_score=0.001):
    out = []
    h, w = meta['shape']
    for x1, y1, x2, y2, sc, c in sorted(rows, key=lambda r: -r[4])[:max_det]:
        if sc < min_score:
            continue
        x1 = np.clip((x1 - meta['left']) / meta['r'], 0, w); x2 = np.clip((x2 - meta['left']) / meta['r'], 0, w)
        y1 = np.clip((y1 - meta['top']) / meta['r'], 0, h);  y2 = np.clip((y2 - meta['top']) / meta['r'], 0, h)
        if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
            continue
        out.append(dict(image_id=int(image_id), category_id=int(cat_ids[int(c)]), bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        score=float(sc)))
    return out


def coco_map(coco_gt, dets, img_ids):
    from pycocotools.cocoeval import COCOeval
    if not dets:
        return 0.0, 0.0
    dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, dt, 'bbox'); ev.params.imgIds = list(img_ids)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def float_outputs(head, feats):
    return [q.fchain(head['box'][s], feats[s])[-1][1] for s in range(3)], [q.fchain(head['cls'][s], feats[s])[-1][1] for s in range(3)]


def main_real(a):
    import cv2
    from pycocotools.coco import COCO
    from ultra_adapter import UltraHead
    coco = COCO(os.path.join(a.coco, 'annotations', 'instances_val2017.json'))
    ids = sorted(coco.getImgIds()); cat_ids = sorted(coco.getCatIds())
    eval_ids, calib_ids = ids[:a.n], ids[-a.calib:]
    assert not set(eval_ids) & set(calib_ids), 'evaluation and calibration images overlap; lower --n or --calib'
    u = UltraHead(a.weights); assert u.nc == len(cat_ids), 'class count mismatch'
    path = lambda i: os.path.join(a.coco, 'images', 'val2017', coco.loadImgs(i)[0]['file_name'])
    t0 = time.time()
    cal = [u.features(cv2.imread(path(i)), a.imgsz)[0] for i in calib_ids]
    st = q.calibrate(u.head, cal, a.pct); P = q.build_program(u.head, st, cal[0], u.nc, a.fb, a.lanes)
    print(f'calibrated on {len(cal)} images, program built ({time.time()-t0:.0f}s)', flush=True)
    det_f, det_fk, det_q, det_u = [], [], [], []
    for k, i in enumerate(eval_ids):
        bgr = cv2.imread(path(i)); feats, meta = u.features(bgr, a.imgsz)
        bx, cl = float_outputs(u.head, feats)
        det_f += rows_to_coco(q.float_post(bx, cl, 300), i, meta, cat_ids)
        det_fk += rows_to_coco(q.float_post(bx, cl, a.K), i, meta, cat_ids)
        act, box = q.run_int(P, st, feats)
        det_q += rows_to_coco(q.dequant_rows(g.post(act, box, P, a.K)), i, meta, cat_ids)
        if a.ul_ref:
            det_u += rows_to_coco(u.predict_rows(meta['rgb'], a.imgsz), i, meta, cat_ids)
        if (k + 1) % 25 == 0:
            print(f'  {k+1}/{len(eval_ids)} images ({time.time()-t0:.0f}s)', flush=True)
    res = {'float head (numpy, K=300)': coco_map(coco, det_f, eval_ids), f'float head, K={a.K} (isolates the K effect)': coco_map(coco, det_fk, eval_ids), f'int8 head (RTL-equivalent, K={a.K})': coco_map(coco, det_q, eval_ids)}
    if a.ul_ref:
        res['ultralytics predict (sanity)'] = coco_map(coco, det_u, eval_ids)
    report(res, a, len(eval_ids), len(cal))


def report(res, a, n, ncal, selftest=False):
    base = list(res.values())[0]
    title = '# SELF-TEST ONLY (random head, synthetic images, pseudo ground truth): NOT an accuracy result' if selftest else '# COCO bbox mAP: float head vs int8 head'
    L = [title, '', f'{n} {"synthetic" if selftest else "val2017"} images, {ncal} calibration images, imgsz {a.imgsz}, '
         f'FB={a.fb}, calibration percentile {a.pct}, LANES={a.lanes}', '', '| model | mAP50-95 | mAP50 | delta mAP50-95 vs float |', '|---|---|---|---|']
    for k, (m, m50) in res.items():
        L.append(f'| {k} | {m:.4f} | {m50:.4f} | {m - base[0]:+.4f} |')
    open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


def selftest(a):
    from pycocotools.coco import COCO
    from run_image import synthetic
    imgsz, nc, nimg = 128, 8, 6
    head, f0, _, _, _, _ = synthetic(imgsz, nc, seed=0, ch=(16, 32, 48))
    feats_all = [f0] + [synthetic(imgsz, nc, seed=s, ch=(16, 32, 48))[1] for s in range(1, nimg)]
    cat_ids = list(range(1, nc + 1)); meta = dict(shape=(imgsz, imgsz), r=1.0, left=0, top=0)
    # pseudo ground truth = float head's top confident rows after a crude NMS
    gt = dict(images=[], annotations=[], categories=[dict(id=c, name=f'c{c}') for c in cat_ids]); aid = 1
    float_rows = []
    for i, feats in enumerate(feats_all):
        gt['images'].append(dict(id=i + 1, width=imgsz, height=imgsz))
        bx, cl = float_outputs(head, feats); rows = q.float_post(bx, cl, 300); float_rows.append(rows)
        kept = []
        for r in rows[:60]:
            if all(not (int(r[5]) == int(k[5]) and _iou(r, k) > 0.5) for k in kept): kept.append(r)
        for r in kept[:5]:
            x1, y1, x2, y2 = np.clip(r[:4], 0, imgsz)
            if x2 - x1 > 1 and y2 - y1 > 1:
                gt['annotations'].append(dict(id=aid, image_id=i + 1, category_id=cat_ids[int(r[5])], bbox=[x1, y1, x2 - x1, y2 - y1],
                                              area=float((x2 - x1) * (y2 - y1)), iscrowd=0)); aid += 1
    coco = COCO(); coco.dataset = gt; coco.createIndex()
    st = q.calibrate(head, feats_all[:3]); P = q.build_program(head, st, feats_all[0], nc, 5, 16)
    det_f, det_q = [], []
    for i, feats in enumerate(feats_all):
        det_f += rows_to_coco(float_rows[i], i + 1, meta, cat_ids)
        act, box = q.run_int(P, st, feats)
        det_q += rows_to_coco(q.dequant_rows(g.post(act, box, P, 100)), i + 1, meta, cat_ids)
    ids = [im['id'] for im in gt['images']]
    res = {'float head (== GT source)': coco_map(coco, det_f, ids), 'int8 head': coco_map(coco, det_q, ids)}
    a.out = a.out.replace('.md', '_selftest.md'); report(res, a, len(ids), 3, selftest=True)
    print('SELFTEST: plumbing only - pseudo ground truth from the float head; not an accuracy result.')


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolo26n.pt'); ap.add_argument('--coco'); ap.add_argument('--imgsz', default='640', help='N (square) or WxH, e.g. 640x480')
    ap.add_argument('--calib', type=int, default=128); ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--lanes', type=int, default=16); ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--fb', type=int, default=5); ap.add_argument('--pct', type=float, default=100.0)
    ap.add_argument('--ul-ref', action='store_true', help='also evaluate ultralytics predict() as a sanity anchor')
    ap.add_argument('--out', default='coco_report.md'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(); a.imgsz = parse_size(a.imgsz)
    selftest(a) if a.selftest else main_real(a)
