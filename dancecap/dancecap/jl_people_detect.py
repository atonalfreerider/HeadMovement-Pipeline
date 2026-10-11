"""john_lula: pod-side 2D detection of EVERY person of the heat video + bib-number reading (agent J3a, 2026-10-11).

Runs on a Runpod pod inside the farm environment (numpy, opencv, torch, ultralytics; easyocr is pip-installed by the launcher,
dancecap.jl_people_pod). Never needs the repository: the file is copied to the pod and run as a script.

    python jl_people_detect.py pose  VIDEO OUT.npz [--weights yolo11x-pose.pt] [--imgsz 1280 1920]
    python jl_people_detect.py ocr   VIDEO POSE.npz OUT.json [--step 2] [--max-persons 6]

pose  every frame, every person (conf >= 0.15), at each of the given inference sizes (the video is 538x960 and the people are 100-750 px
      high: a larger inference size finds the small ones, a smaller one is better for the big ones; the local stage merges the passes).
      Per detection: box, conf, 17 COCO keypoints (x, y, conf), mean colours (torso / thigh / shin / head, RGB 0-255), a 32x16 colour
      thumbnail of the box (identity by appearance).
ocr   the bib numbers: per person crop (the shoulders-to-knees zone, enlarged), EasyOCR restricted to digits; every read is stored with
      its box in frame pixels and the person (frame, row) it belongs to; the local stage turns them into identity evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

CONF_MIN = 0.15
THUMB = (32, 16)   # rows, cols


def log(msg: str) -> None:
    print(f"[jl_detect {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _region(img: np.ndarray, pts: np.ndarray, conf: np.ndarray, shrink: float = 0.2, need: int = 1) -> np.ndarray:
    """Mean RGB of the bounding box of the keypoints with conf >= 0.3 (shrunk), NaN when fewer than `need` points or the box is tiny."""
    ok = conf >= 0.3
    if int(ok.sum()) < need:
        return np.full(3, np.nan, np.float32)
    x0, y0 = pts[ok].min(0)
    x1, y1 = pts[ok].max(0)
    mx, my = shrink * (x1 - x0), shrink * (y1 - y0)
    h, w = img.shape[:2]
    x0, x1, y0, y1 = int(max(0, x0 + mx)), int(min(w, x1 - mx)), int(max(0, y0 + my)), int(min(h, y1 - my))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return np.full(3, np.nan, np.float32)
    return img[y0:y1, x0:x1, ::-1].reshape(-1, 3).mean(0).astype(np.float32)


def colours(img: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """(4, 3) mean RGB of torso (shoulders-hips), thigh (hips-knees), shin (knees-ankles), head (around the nose)."""
    xy, c = kp[:, :2], kp[:, 2]
    out = np.full((4, 3), np.nan, np.float32)
    if (c[[5, 6, 11, 12]] >= 0.3).sum() >= 3:
        out[0] = _region(img, xy[[5, 6, 11, 12]], c[[5, 6, 11, 12]], 0.2)
    if (c[[11, 12, 13, 14]] >= 0.3).sum() >= 3:
        out[1] = _region(img, xy[[11, 12, 13, 14]], c[[11, 12, 13, 14]], 0.15)
    if (c[[13, 14, 15, 16]] >= 0.3).sum() >= 3:
        out[2] = _region(img, xy[[13, 14, 15, 16]], c[[13, 14, 15, 16]], 0.15)
    if c[0] >= 0.3 and (c[[5, 6]] >= 0.3).any():
        sw = np.abs(xy[5, 0] - xy[6, 0]) if (c[[5, 6]] >= 0.3).all() else 0.0
        r = max(6.0, 0.35 * sw)
        h, w = img.shape[:2]
        x0, x1, y0, y1 = int(max(0, xy[0, 0] - r)), int(min(w, xy[0, 0] + r)), int(max(0, xy[0, 1] - 1.4 * r)), int(min(h, xy[0, 1] + 0.4 * r))
        if x1 - x0 >= 3 and y1 - y0 >= 3:
            out[3] = img[y0:y1, x0:x1, ::-1].reshape(-1, 3).mean(0)
    return out


def thumb(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3]))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((*THUMB, 3), np.uint8)
    return cv2.resize(img[y0:y1, x0:x1, ::-1], (THUMB[1], THUMB[0]), interpolation=cv2.INTER_AREA)


def read_frames(video: str):
    import cv2
    cap = cv2.VideoCapture(video)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield i, f
        i += 1
    cap.release()


def pose(video: str, out: str, weights: str, sizes: list[int], batch: int = 6) -> None:
    import cv2
    from ultralytics import YOLO
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    model = YOLO(weights)
    rows: dict[str, list] = {k: [] for k in ("frame", "pass", "box", "conf", "kp", "col", "thumb")}
    t0 = time.time()
    buf, idx = [], []

    def flush():
        for ps, sz in enumerate(sizes):
            res = model.predict(buf, imgsz=sz, conf=CONF_MIN, iou=0.6, classes=[0], half=True, max_det=40, verbose=False)
            for fi, im, r in zip(idx, buf, res):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                box = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                kp = np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], -1)
                for b, c, k in zip(box, cf, kp):
                    rows["frame"].append(fi); rows["pass"].append(ps); rows["box"].append(b); rows["conf"].append(c); rows["kp"].append(k)
                    rows["col"].append(colours(im, k)); rows["thumb"].append(thumb(im, b))
        buf.clear(); idx.clear()

    n = 0
    for fi, im in read_frames(video):
        buf.append(im); idx.append(fi)
        n = fi + 1
        if len(buf) == batch:
            flush()
            if fi % 300 < batch:
                log(f"frame {fi}/{n_total}  {len(rows['frame'])} detections  {time.time() - t0:.0f} s")
    if buf:
        flush()
    np.savez_compressed(out, frame=np.array(rows["frame"], np.int32), npass=np.array(rows["pass"], np.int8), box=np.array(rows["box"], np.float32),
                        conf=np.array(rows["conf"], np.float32), kp=np.array(rows["kp"], np.float32), col=np.array(rows["col"], np.float32),
                        thumb=np.array(rows["thumb"], np.uint8), n_frames=n, fps=fps, size=np.array([W, H]), sizes=np.array(sizes))
    log(f"pose done: {len(rows['frame'])} detections over {n} frames in {time.time() - t0:.0f} s")


def bib_zone(box: np.ndarray, kp: np.ndarray, pad: float = 0.06) -> tuple[int, int, int, int]:
    """The crop the bibs are read from: the box's width, from just above the shoulders to the knees (John's bib is on his back between the
    shoulder blades and the waist, Lula's on her trousers at the hip / thigh), else the upper 70 % of the box."""
    x0, y0, x1, y1 = [float(v) for v in box]
    h = y1 - y0
    c = kp[:, 2]
    ys = []
    if (c[[5, 6]] >= 0.3).any():
        ys.append(kp[[5, 6], 1][c[[5, 6]] >= 0.3].min() - 0.08 * h)
    top = min(ys) if ys else y0
    kn = kp[[13, 14], 1][c[[13, 14]] >= 0.3] if (c[[13, 14]] >= 0.3).any() else None
    bot = float(kn.max()) + 0.02 * h if kn is not None else y0 + 0.75 * h
    top = max(y0, top)
    bot = min(y1, max(bot, top + 0.3 * h))
    px = pad * (x1 - x0)
    return int(x0 - px), int(top), int(x1 + px), int(bot)


def ocr(video: str, pose_npz: str, out: str, step: int, max_persons: int, min_h: float, zone_h: int) -> None:
    import cv2
    import easyocr
    z = np.load(pose_npz)
    fr, box, kp, cf = z["frame"], z["box"], z["kp"], z["conf"]
    hgt = box[:, 3] - box[:, 1]
    p0 = int(np.flatnonzero(z["sizes"] == 1280)[0]) if (z["sizes"] == 1280).any() else 0
    keep = np.flatnonzero(z["npass"] == p0)
    by_frame: dict[int, list[int]] = {}
    for i in keep:
        by_frame.setdefault(int(fr[i]), []).append(int(i))
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    reads = []
    t0 = time.time()
    n_crops = 0
    for fi, img in read_frames(video):
        if fi % step or fi not in by_frame:
            continue
        ids = sorted(by_frame[fi], key=lambda i: -hgt[i])[:max_persons]
        for i in ids:
            if hgt[i] < min_h:
                continue
            x0, y0, x1, y1 = bib_zone(box[i], kp[i])
            H, W = img.shape[:2]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            crop = img[y0:y1, x0:x1]
            s = float(np.clip(zone_h / crop.shape[0], 1.0, 5.0))
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            n_crops += 1
            try:
                res = reader.readtext(crop, allowlist="0123456789", detail=1, paragraph=False, text_threshold=0.5, low_text=0.3, mag_ratio=1.0,
                                      canvas_size=2560, width_ths=0.3)
            except Exception as e:  # noqa: BLE001
                log(f"frame {fi}: ocr failed {e}")
                continue
            for quad, text, c in res:
                q = np.array(quad, float) / s + np.array([x0, y0])
                reads.append({"frame": fi, "det": int(i), "text": text, "conf": round(float(c), 3), "quad": np.round(q, 1).tolist()})
        if fi % 200 < step:
            log(f"frame {fi}  crops {n_crops}  reads {len(reads)}  {time.time() - t0:.0f} s")
    with open(out, "w") as f:
        json.dump({"step": step, "reads": reads, "n_crops": n_crops, "seconds": time.time() - t0}, f)
    log(f"ocr done: {n_crops} crops, {len(reads)} reads in {time.time() - t0:.0f} s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pose")
    a.add_argument("video"); a.add_argument("out"); a.add_argument("--weights", default="yolo11x-pose.pt")
    a.add_argument("--imgsz", type=int, nargs="+", default=[1280, 1920])
    b = sub.add_parser("ocr")
    b.add_argument("video"); b.add_argument("pose"); b.add_argument("out")
    b.add_argument("--step", type=int, default=2); b.add_argument("--max-persons", type=int, default=6)
    b.add_argument("--min-h", type=float, default=150.0); b.add_argument("--zone-h", type=int, default=600)
    ns = ap.parse_args(argv)
    if ns.cmd == "pose":
        pose(ns.video, ns.out, ns.weights, ns.imgsz)
    else:
        ocr(ns.video, ns.pose, ns.out, ns.step, ns.max_persons, ns.min_h, ns.zone_h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
