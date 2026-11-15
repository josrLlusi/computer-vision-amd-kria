"""Image in -> detections drawn on an image out, with the YOLO26 detect head executed by the Verilog RTL.

  real model :  python run_image.py --image cat.jpg --weights yolo26n.pt --imgsz 160 --out out.png
  synthetic  :  python run_image.py --synthetic --imgsz 64 --out out.png      (random head, tests the flow)

Backbone + neck run in software (Ultralytics). Only the Detect head runs in RTL.
Tip: simulation speed is ~0.3 M MAC-cycles/s; use --imgsz 160 for the real model, --sim golden to skip RTL.
"""
import argparse, os, sys, time
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize as q
import golden as g


def parse_size(s):
    """'640' -> 640 (square);  '640x480' (WxH) -> (480, 640) as (height, width). Sides must be multiples of 32."""
    s = str(s).lower()
    if 'x' in s:
        w, h = (int(v) for v in s.split('x'))
        assert w % 32 == 0 and h % 32 == 0, 'sides must be multiples of 32'
        return (h, w)
    return int(s)


def synthetic(imgsz, nc, seed=0, ch=(16, 32, 48), heavy=False):
    rng = np.random.default_rng(seed)
    img = np.full((imgsz, imgsz, 3), 114, np.uint8)
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([imgsz*.15, imgsz*.2, imgsz*.5, imgsz*.7], fill=(200, 60, 60))
    d.ellipse([imgsz*.55, imgsz*.45, imgsz*.9, imgsz*.85], fill=(60, 90, 210))
    img = np.array(im)
    ch = list(ch); G = [imgsz // 8, imgsz // 16, imgsz // 32]
    c2 = max(16, ch[0] // 4, 4); c3 = max(ch[0], min(nc, 100))
    def lay(cout, cin, k, dw=False, act=True, bias=0.0):
        w = rng.normal(0, 1 / np.sqrt(cin * k * k * (1 if not dw else 1)), (cout, k, k) if dw else (cout, cin, k, k))
        return dict(w=w, b=rng.normal(0, .1, cout) + bias, dw=dw, act=act, k=k)
    head = dict(box=[], cls=[]); feats = []
    for s in range(3):
        base = np.kron(img[:, :, 0] / 255.0, np.ones((1, 1)))[:: 8 * 2 ** s, :: 8 * 2 ** s][:G[s], :G[s]]
        f = (rng.standard_t(5, (ch[s], G[s], G[s])) if heavy else rng.normal(0, 1, (ch[s], G[s], G[s]))) + 3 * base[None]
        feats.append(f)
        cin = ch[s]
        head['box'].append([lay(c2, cin, 3), lay(c2, c2, 3), lay(4, c2, 1, act=False, bias=1.5)])
        head['cls'].append([lay(cin, cin, 3, dw=True), lay(c3, cin, 1), lay(c3, c3, 3, dw=True), lay(c3, c3, 1),
                            lay(nc, c3, 1, act=False, bias=-2.2)])
    return head, feats, nc, {i: f'cls{i}' for i in range(nc)}, img, np.zeros((0, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image'); ap.add_argument('--weights', default='yolo26n.pt')
    ap.add_argument('--imgsz', default='160', help='N (square) or WxH, e.g. 640x480'); ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--sim', choices=['rtl', 'golden'], default='rtl')
    ap.add_argument('--maxdet', type=int, default=300); ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--fb', type=int, default=5); ap.add_argument('--lanes', type=int, default=16); ap.add_argument('--nc', type=int, default=8)
    ap.add_argument('--out', default='out.png'); ap.add_argument('--work', default='sim/image_run')
    a = ap.parse_args(); a.imgsz = parse_size(a.imgsz)

    if a.synthetic:
        assert isinstance(a.imgsz, int), 'synthetic mode is square only'
        head, feats, nc, names, rgb, ref_rows = synthetic(a.imgsz, a.nc)
    else:
        import cv2, ultra_adapter
        bgr = cv2.imread(a.image); assert bgr is not None, 'cannot read image'
        head, feats, nc, names, rgb, ref_rows = ultra_adapter.load(a.weights, bgr, a.imgsz)

    t = time.time()
    P, info = q.quantize_head(head, feats, nc, FB=a.fb, lanes=a.lanes)
    act, box = g.golden_layers(P)
    print(f'quantized: {len(P.layers)} layers, {len(P.wgt)} weights, act mem {P.act_words} words ({time.time()-t:.1f}s)')

    if a.sim == 'rtl':
        rows, cyc = q.run_rtl(P, a.work, max_det=a.maxdet)
        print(f'RTL finished: {cyc} cycles, {len(rows)} rows')
        rows_q = q.dequant_rows(rows)
        gold = g.post(act, box, P, a.maxdet)
        print('RTL rows == bit-exact golden rows:', 'YES' if [list(r) for r in gold] == rows else 'NO  <-- mismatch')
    else:
        rows = g.post(act, box, P, a.maxdet)
        rows_q = q.dequant_rows(rows)

    # accuracy of the int8 head vs the float head (same weights, same features)
    cls_q, box_q = q.int_outputs(P, info, act, box, a.fb)
    for s_ in range(3):
        fl, fb_ = info['float_cls'][s_], info['float_box'][s_]
        print(f'  P{s_+3}: cls logit corr={np.corrcoef(cls_q[s_].ravel(), fl.ravel())[0,1]:.4f} max|err|={np.abs(cls_q[s_]-fl).max():.3f}'
              f' | box corr={np.corrcoef(box_q[s_].ravel(), fb_.ravel())[0,1]:.4f} max|err|={np.abs(box_q[s_]-fb_).max():.3f} grid units')
    rows_f = q.float_post(info['float_box'], info['float_cls'], a.maxdet)

    keep = rows_q[rows_q[:, 4] >= a.conf]
    note = ''
    if len(keep) == 0:
        keep = rows_q[:5]; note = f' (nothing >= {a.conf}; drawing top 5)'
    # reference = Ultralytics' own detections when available (real model), else the float head of this script
    if len(ref_rows):
        ref, ref_name = ref_rows[ref_rows[:, 4] >= a.conf], 'Ultralytics predict()'
    else:
        ref, ref_name = rows_f[rows_f[:, 4] >= a.conf], 'float head (numpy)'
    def iou(p, r):
        x1, y1, x2, y2 = max(p[0], r[0]), max(p[1], r[1]), min(p[2], r[2]), min(p[3], r[3])
        i_ = max(0, x2 - x1) * max(0, y2 - y1); u = (p[2]-p[0])*(p[3]-p[1]) + (r[2]-r[0])*(r[3]-r[1]) - i_
        return i_ / u if u > 0 else 0
    used, matched, ious = set(), 0, []
    for p in rows_q[rows_q[:, 4] >= a.conf]:
        best, bj = 0.0, -1
        for j, r in enumerate(ref):
            if j not in used and int(r[5]) == int(p[5]) and iou(p, r) > best: best, bj = iou(p, r), j
        if best >= 0.5: used.add(bj); matched += 1; ious.append(best)
    nq = int((rows_q[:, 4] >= a.conf).sum())
    print(f'drawing {len(keep)} RTL detections{note}')
    print(f'CHECK vs {ref_name}: RTL has {nq} detections >= {a.conf}, reference has {len(ref)}, '
          f'{matched} matched (same class, IoU>=0.5), mean IoU {np.mean(ious) if ious else float("nan"):.3f}')
    print('  RTL  top-3:', [(int(r[5]), round(float(r[4]), 3)) for r in rows_q[:3]])
    print('  ref  top-3:', [(int(r[5]), round(float(r[4]), 3)) for r in (ref if len(ref) else rows_f)[:3]])

    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    for x1, y1, x2, y2, sc, c in ref:                       # reference: thin red
        d.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=1)
    for x1, y1, x2, y2, sc, c in keep:                      # RTL: thick green + label
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        d.text((x1 + 2, max(0, y1 - 11)), f'{names.get(int(c), int(c))} {sc:.2f}', fill=(255, 255, 0))
    im.save(a.out); print('wrote', a.out, '(green = RTL int8 head, red = ' + ref_name + ')')


if __name__ == '__main__':
    main()
