"""MediaPipe SelfieMulticlass on the head crops of jl_facing - a standalone script for the separate .venv-face env.

It imports only numpy, OpenCV and mediapipe (no dancecap / torch / scipy), so jl_facing runs it as a subprocess:

    .venv-face/Scripts/python.exe dancecap/dancecap/mono4d/jl_facing_mp.py --spec spec.json

spec.json = {"model": <selfie_multiclass_256x256.tflite>, "frames": [frame jpg paths, index = frame], "boxes": <(F, 2, 4)
int32 .npy, -1 = none>, "masks": <masks_01.npz, keys m_<frame>_<who> PNG bytes>, "out": <out .npz>}
Per crop (square box, zero padded, resized to 256 x 256): the mean confidence of the six classes (background, hair,
body-skin, face-skin, clothes, others) over the pixels of the dancer's OWN mask inside the crop, and the share of those
pixels whose argmax is each class. out npz: prob (F, 2, 6), share (F, 2, 6), n_px (F, 2).
Weights: local dancecap/weights/mediapipe/selfie_multiclass_256x256.tflite (Google MediaPipe, Apache-2.0); nothing is
downloaded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def square(img: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in box)
    H, W = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + img.shape[2:], img.dtype)
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if ax1 > ax0 and ay1 > ay0:
        out[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = img[ay0:ay1, ax0:ax1]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    spec = json.loads(Path(a.spec).read_text())
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(spec["model"])),
        output_confidence_masks=True, output_category_mask=False))
    boxes = np.load(spec["boxes"])
    masks = np.load(spec["masks"])
    keys = set(masks.files)
    F = boxes.shape[0]
    prob = np.full((F, 2, 6), np.nan, np.float32)
    share = np.full((F, 2, 6), np.nan, np.float32)
    npx = np.zeros((F, 2), np.int32)
    t0 = time.time()
    for f in range(F):
        if not (boxes[f] >= 0).any():
            continue
        img = None
        for o in range(2):
            b = boxes[f, o]
            k = f"m_{f}_{o}"
            if (b < 0).all() or b[2] - b[0] < 4 or k not in keys:
                continue
            if img is None:
                img = cv2.imread(spec["frames"][f], cv2.IMREAD_COLOR)
            m = cv2.imdecode(np.frombuffer(masks[k].tobytes(), np.uint8), cv2.IMREAD_UNCHANGED) > 0
            crop = cv2.resize(square(img, b), (256, 256), interpolation=cv2.INTER_CUBIC)
            mc = cv2.resize(square(m.astype(np.uint8), b), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
            if mc.sum() < 50:
                continue
            res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop[..., ::-1])))
            P = np.stack([np.asarray(c.numpy_view(), np.float32).reshape(256, 256) for c in res.confidence_masks], -1)
            sel = P[mc]
            prob[f, o] = sel.mean(0)
            share[f, o] = np.bincount(sel.argmax(1), minlength=6)[:6] / len(sel)
            npx[f, o] = int(mc.sum())
        if f % 300 == 0:
            print(f"[jl_facing_mp] {f}/{F} {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(spec["out"], prob=prob, share=share, n_px=npx)
    print(f"[jl_facing_mp] -> {spec['out']} ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
