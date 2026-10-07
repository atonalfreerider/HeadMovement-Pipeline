"""MediaPipe hair segmentation of image crops - a standalone script for the separate .venv-face environment.

It imports only numpy, OpenCV and mediapipe (no dancecap / torch), so hair_reference.py runs it as a subprocess:

    .venv-face/Scripts/python.exe dancecap/dancecap/hair_segment_mp.py --jobs jobs.json

jobs.json = {"models": {"hair": <hair_segmenter.tflite>, "multiclass": <selfie_multiclass_256x256.tflite>},
             "jobs": [{"image": path, "box": [x0, y0, x1, y1] | null, "out": path.png, "model": "hair"|"multiclass"}]}
Every job crops the box (clamped, zero-padded to a square so the model sees the true aspect), runs the model and writes
`out` at the crop's size: "hair" -> one uint8 PNG = P(hair) x 255; "multiclass" -> a 4-channel uint8 PNG
(B = hair, G = clothes, R = face-skin, A = body-skin, each P x 255) - the six SelfieMulticlass classes are background,
hair, body-skin, face-skin, clothes, others.

Weights (open, ungated; Google MediaPipe Solutions, framework Apache-2.0, model cards on developers.google.com):
  hair_segmenter.tflite              https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/
                                     float32/latest/hair_segmenter.tflite (781,618 B, 512x512 input)
  selfie_multiclass_256x256.tflite   https://storage.googleapis.com/mediapipe-models/image_segmenter/
                                     selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
                                     (16,371,837 B)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def square_crop(img: np.ndarray, box) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    """Crop box (clamped) padded with zeros to a square; returns (square, clamped box, (pad_x, pad_y))."""
    H, W = img.shape[:2]
    if box is None:
        box = (0, 0, W, H)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.zeros((s, s, 3), np.uint8)
    px, py = (s - w) // 2, (s - h) // 2
    sq[py:py + h, px:px + w] = crop
    return sq, (x0, y0, x1, y1), (px, py)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    a = ap.parse_args(argv)
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    spec = json.loads(Path(a.jobs).read_text())
    segs = {}
    for name, path in spec["models"].items():
        opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=str(path)),
                                            output_confidence_masks=True, output_category_mask=False)
        segs[name] = vision.ImageSegmenter.create_from_options(opts)
    done = 0
    for job in spec["jobs"]:
        img = cv2.imread(job["image"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"[mp] missing {job['image']}", file=sys.stderr)
            continue
        sq, (x0, y0, x1, y1), (px, py) = square_crop(img, job.get("box"))
        h, w = y1 - y0, x1 - x0
        if h < 8 or w < 8:
            continue
        model = job.get("model", "hair")
        rgb = np.ascontiguousarray(sq[:, :, ::-1])
        res = segs[model].segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        masks = [m.numpy_view().astype(np.float32) for m in res.confidence_masks]
        s = sq.shape[0]

        def back(m):
            m = cv2.resize(m, (s, s), interpolation=cv2.INTER_LINEAR)
            return np.clip(m[py:py + h, px:px + w] * 255 + 0.5, 0, 255).astype(np.uint8)
        if model == "hair":
            out = back(masks[1] if len(masks) > 1 else masks[0])
        else:  # background, hair, body-skin, face-skin, clothes, others
            out = np.dstack([back(masks[1]), back(masks[4]), back(masks[3]), back(masks[2])])
        Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(job["out"], out)
        done += 1
    print(json.dumps({"done": done, "jobs": len(spec["jobs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
