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


def _plain(m):
    return dict(w=m.weight.detach().cpu().numpy().astype(np.float64), b=m.bias.detach().cpu().numpy().astype(np.float64),
                dw=False, act=False, k=1)


class UltraHead:
    """Loads a checkpoint once; `features(bgr, imgsz)` returns the three neck outputs for one image."""

    def __init__(self, weights):
        from ultralytics import YOLO
        self.y = YOLO(weights)
        self.net = self.y.model.eval()
        det = self.net.model[-1]
        assert getattr(det, 'one2one_cv2', None) is not None, 'checkpoint has no one2one head'
        self.det, self.nc, self.names = det, det.nc, self.net.names
        self.head = dict(box=[], cls=[])
        for i in range(det.nl):
            c2 = det.one2one_cv2[i]          # Sequential(Conv, Conv, Conv2d)
            self.head['box'].append([_layer(c2[0]), _layer(c2[1]), _plain(c2[2])])
            c3 = det.one2one_cv3[i]          # Sequential(Seq(DWConv,Conv), Seq(DWConv,Conv), Conv2d)
            self.head['cls'].append([_layer(c3[0][0], dw=True), _layer(c3[0][1]), _layer(c3[1][0], dw=True),
                                     _layer(c3[1][1]), _plain(c3[2])])

    @staticmethod
    def hw(imgsz):
        """imgsz: int (square) or (height, width). Both must be multiples of 32 (640x480 -> (480, 640))."""
        h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        assert h % 32 == 0 and w % 32 == 0, 'network input sides must be multiples of 32'
        return h, w

    def letterbox(self, bgr, imgsz):
        """Same geometry as ultralytics LetterBox(auto=False): returns (rgb uint8 H x W, ratio, left, top)."""
        from ultralytics.data.augment import LetterBox
        nh_, nw_ = self.hw(imgsz)
        h, w = bgr.shape[:2]
        r = min(nh_ / h, nw_ / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (nw_ - nw) / 2, (nh_ - nh) / 2
        left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
        lb = LetterBox((nh_, nw_), auto=False, stride=32)(image=bgr)
        return lb[..., ::-1].copy(), r, left, top

    def features(self, bgr, imgsz):
        import torch
        rgb, r, left, top = self.letterbox(bgr, imgsz)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None] / 255.0
        cap = {}
        h = self.det.register_forward_pre_hook(lambda mod, inp: cap.__setitem__('x', [t.detach().cpu().numpy()[0] for t in inp[0]]))
        with torch.no_grad():
            self.net(x)
        h.remove()
        return [f.astype(np.float64) for f in cap['x']], dict(rgb=rgb, r=r, left=left, top=top, shape=bgr.shape[:2])

    def predict_rows(self, rgb, imgsz, conf=0.001):
        """Ultralytics' own detections on a letterboxed RGB image (pixels of that image) - sanity reference."""
        res = self.y.predict(rgb[..., ::-1], imgsz=list(self.hw(imgsz)), conf=conf, verbose=False)[0]
        if len(res.boxes) == 0:
            return np.zeros((0, 6))
        return np.concatenate([res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()[:, None],
                               res.boxes.cls.cpu().numpy()[:, None]], 1)


def load(weights, image_bgr, imgsz):
    """Back-compat for run_image.py: (head, feats, nc, names, letterboxed_rgb, ref_rows)."""
    u = UltraHead(weights)
    feats, meta = u.features(image_bgr, imgsz)
    return u.head, feats, u.nc, u.names, meta['rgb'], u.predict_rows(meta['rgb'], imgsz, 0.25)
