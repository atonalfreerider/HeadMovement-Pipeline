"""john_lula: pod-side top-down whole-body pose (RTMW-x, COCO-WholeBody 133 keypoints incl. 6 foot points, 68 face, 42 hand) of the two dancers (agent J3a).

    python jl_people_pose.py VIDEO FUSED.npz OUT.npz

FUSED.npz (written by dancecap.jl_people): box (F, 2, 4) xyxy of the fused SAM mask of John / Lula (NaN = absent), "m_<frame>_<obj>" PNG bytes of the masks.
Two variants per dancer and frame, both on the dancer's own mask box: `plain` (the frame as is) and `masked` (the partner's pixels, those inside the
partner's mask but not the dancer's own, painted grey: in an embrace the estimator then sees ONE body). Output kp (F, 2, 2, 133, 3) = [x, y, score]
in frame pixels (variant axis: plain, masked), NaN where the dancer is absent. The score is RTMW's SimCC peak value (0 .. ~8, NOT a probability).

rtmlib + onnxruntime-gpu 1.20.2 in the farm env (the CUDA libs come from torch's nvidia wheels: LD_LIBRARY_PATH must list site-packages/nvidia/*/lib, torch
imported first). The weights (rtmw-dw-x-l_simcc-cocktail14_270e-384x288) download once into the volume's torch_home.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"


def log(msg: str) -> None:
    print(f"[jl_pose {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None) -> int:
    import torch  # noqa: F401  (loads the CUDA libraries onnxruntime needs)
    import cv2
    from rtmlib import RTMPose
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("fused"); ap.add_argument("out")
    ap.add_argument("--pad", type=float, default=0.04)
    a = ap.parse_args(argv)
    z = np.load(a.fused)
    boxes = z["box"]
    F = len(boxes)
    pose = RTMPose(URL, model_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cuda")
    out = np.full((F, 2, 2, 133, 3), np.nan, np.float32)
    cap = cv2.VideoCapture(a.video)
    t0 = time.time()
    H = W = None
    for f in range(F):
        ok, img = cap.read()
        if not ok:
            break
        H, W = img.shape[:2]
        ms = []
        for o in range(2):
            k = f"m_{f}_{o}"
            ms.append(cv2.imdecode(z[k], cv2.IMREAD_GRAYSCALE) > 127 if k in z.files else None)
        for o in range(2):
            b = boxes[f, o]
            if not np.isfinite(b).all() or b[2] - b[0] < 8 or b[3] - b[1] < 8:
                continue
            px, py = a.pad * (b[2] - b[0]), a.pad * (b[3] - b[1])
            bb = [max(0, b[0] - px), max(0, b[1] - py), min(W, b[2] + px), min(H, b[3] + py)]
            kp, sc = pose(img, bboxes=[bb])
            out[f, o, 0, :, :2], out[f, o, 0, :, 2] = kp[0], sc[0]
            partner, own = ms[1 - o], ms[o]
            if partner is not None and own is not None:
                pm = (partner & ~own).astype(np.uint8)
                pm = cv2.dilate(pm, np.ones((5, 5), np.uint8)).astype(bool) & ~own
                im2 = img.copy()
                im2[pm] = 128
                kp, sc = pose(im2, bboxes=[bb])
            out[f, o, 1, :, :2], out[f, o, 1, :, 2] = kp[0], sc[0]
        if f % 200 == 0:
            log(f"frame {f}/{F}  {time.time() - t0:.0f} s")
    np.savez_compressed(a.out, kp=out, model=np.array(URL))
    log(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
