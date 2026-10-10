"""2D body keypoints of every person of a video (YOLO11 pose, COCO-17) with a torso colour for identity.

    python -m dancecap.cameras.people2d <video> --out <file.npz> [--stride 1] [--weights yolo11x-pose.pt]

The room / camera calibration of a two-camera take only needs the two dancers' points in synchronised frames; the
detector returns everybody (judges, audience), the dancers are picked afterwards (dancecap.cameras.twocam) by motion,
size and the torso colour stored here (mean RGB of the box between shoulders and hips).

Output arrays (D = total detections): frame (D,) video frame index, box (D,4) x0 y0 x1 y1, conf (D,), kp (D,17,3) x y score,
rgb (D,3) mean torso colour (0-255), n_frames, size.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import ptz_feats as F
from .ptz_run import probe


def torso_rgb(img_bgr: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Mean RGB of the torso box (shoulders to hips, shrunk) or NaN when the keypoints are unreliable."""
    idx = [5, 6, 11, 12]
    pts = kp[idx]
    if (pts[:, 2] < 0.3).sum() > 1:
        return np.full(3, np.nan, np.float32)
    ok = pts[:, 2] >= 0.3
    x0, x1 = pts[ok, 0].min(), pts[ok, 0].max()
    y0, y1 = pts[ok, 1].min(), pts[ok, 1].max()
    mx, my = 0.2 * (x1 - x0), 0.15 * (y1 - y0)
    x0, x1, y0, y1 = int(x0 + mx), int(x1 - mx), int(y0 + my), int(y1 - my)
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return np.full(3, np.nan, np.float32)
    return img_bgr[y0:y1, x0:x1, ::-1].reshape(-1, 3).mean(0).astype(np.float32)


def run(video: str, out: Path, stride: int = 1, frames: list[int] | None = None, weights: str = "yolo11x-pose.pt", batch: int = 8, imgsz: int = 1280,
        conf: float = 0.25, log=print) -> dict:
    from ultralytics import YOLO
    meta = probe(video)
    model = YOLO(weights)
    t0 = time.time()
    rows = {"frame": [], "box": [], "conf": [], "kp": [], "rgb": []}
    buf, idx = [], []

    def flush():
        res = model(buf, imgsz=imgsz, conf=conf, classes=[0], verbose=False)
        for fi, im, r in zip(idx, buf, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            box = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            kp = np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], -1)
            for b, c, k in zip(box, cf, kp):
                rows["frame"].append(fi); rows["box"].append(b); rows["conf"].append(c); rows["kp"].append(k)
                rows["rgb"].append(torso_rgb(im, k))
        buf.clear(); idx.clear()

    for fi, im in F.read_video_frames(video, meta["width"], meta["height"], stride, frames=frames):
        buf.append(im.copy()); idx.append(fi)
        if len(buf) >= batch:
            flush()
            if len(idx) == batch and (fi // max(stride, 1)) % (batch * 25) < stride:
                log(f"frame {fi}/{meta['frames']} {time.time() - t0:.0f} s")
    if buf:
        flush()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frame=np.array(rows["frame"], np.int32), box=np.array(rows["box"], np.float32),
                        conf=np.array(rows["conf"], np.float32), kp=np.array(rows["kp"], np.float32),
                        rgb=np.array(rows["rgb"], np.float32), n_frames=meta["frames"], fps=meta["fps"],
                        size=np.array([meta["width"], meta["height"]]), stride=stride)
    return {"detections": len(rows["frame"]), "seconds": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--weights", default="yolo11x-pose.pt")
    ap.add_argument("--frames", help="text/json file with the frame indices to process (instead of --stride)")
    a = ap.parse_args(argv)
    fr = None
    if a.frames:
        import json
        fr = [int(k) for k in json.loads(Path(a.frames).read_text())]
    print(run(a.video, Path(a.out), a.stride, fr, a.weights))


if __name__ == "__main__":
    main()
