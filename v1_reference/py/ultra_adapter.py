"""Pull the one2one head weights + the neck feature maps out of a real Ultralytics YOLO26 model.
NOT TESTED in the authoring sandbox (torch/ultralytics could not be installed there).
Needs: pip install ultralytics   (and a YOLO26 checkpoint, e.g. yolo26n.pt)
"""
import numpy as np


def _fold(m):
    """Conv (+BN) -> (w, b). Works on unfused (conv+bn) and fused (conv with bias) modules."""
    conv = m.conv
    w = conv.weight.detach().cpu().numpy().astype(np.float64)
    b = (conv.bias.detach().cpu().numpy().astype(np.float64) if conv.bias is not None else np.zeros(w.shape[0]))
    bn = getattr(m, 'bn', None)
    if bn is not None and hasattr(bn, 'running_mean'):
        s = bn.weight.detach().cpu().numpy() / np.sqrt(bn.running_var.detach().cpu().numpy() + bn.eps)
        w = w * s.reshape(-1, 1, 1, 1)
        b = (b - bn.running_mean.detach().cpu().numpy()) * s + bn.bias.detach().cpu().numpy()
    return w, b


def _layer(m, dw=False, act=True):
    w, b = _fold(m)
    k = w.shape[-1]
    if dw:
        assert w.shape[1] == 1, 'depthwise conv expected'
        w = w[:, 0]                       # (c,1,k,k) -> (c,k,k)
    return dict(w=w, b=b, dw=dw, act=act, k=k)


def load(weights, image_bgr, imgsz):
    """Returns (head, feats, nc, names, letterboxed_rgb_uint8, ref_detections)."""
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    y = YOLO(weights)
    net = y.model.eval()
    det = net.model[-1]
    assert getattr(det, 'one2one_cv2', None) is not None, 'checkpoint has no one2one head'
    nc = det.nc
    lb = LetterBox((imgsz, imgsz), auto=False, stride=32)(image=image_bgr)         # BGR uint8
    x = torch.from_numpy(np.ascontiguousarray(lb[..., ::-1].transpose(2, 0, 1))).float()[None] / 255.0
    cap = {}
    h = det.register_forward_pre_hook(lambda mod, inp: cap.__setitem__('x', [t.detach().cpu().numpy()[0] for t in inp[0]]))
    with torch.no_grad():
        net(x)
    h.remove()
    head = dict(box=[], cls=[])
    for i in range(det.nl):
        c2 = det.one2one_cv2[i]          # Sequential(Conv, Conv, Conv2d)
        head['box'].append([_layer(c2[0]), _layer(c2[1]),
                            dict(w=c2[2].weight.detach().cpu().numpy().astype(np.float64),
                                 b=c2[2].bias.detach().cpu().numpy().astype(np.float64), dw=False, act=False, k=1)])
        c3 = det.one2one_cv3[i]          # Sequential(Seq(DWConv,Conv), Seq(DWConv,Conv), Conv2d)
        head['cls'].append([_layer(c3[0][0], dw=True), _layer(c3[0][1]), _layer(c3[1][0], dw=True), _layer(c3[1][1]),
                            dict(w=c3[2].weight.detach().cpu().numpy().astype(np.float64),
                                 b=c3[2].bias.detach().cpu().numpy().astype(np.float64), dw=False, act=False, k=1)])
    ref = y.predict(lb, imgsz=imgsz, conf=0.25, verbose=False)[0]
    ref_rows = np.concatenate([ref.boxes.xyxy.cpu().numpy(), ref.boxes.conf.cpu().numpy()[:, None],
                               ref.boxes.cls.cpu().numpy()[:, None]], 1) if len(ref.boxes) else np.zeros((0, 6))
    return head, [f.astype(np.float64) for f in cap['x']], nc, net.names, lb[..., ::-1].copy(), ref_rows
