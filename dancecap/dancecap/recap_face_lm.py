"""2D face landmarks of image crops for the FLAME identity fit - a standalone script for the separate .venv-face environment.

It imports only numpy, OpenCV, mediapipe and rtmlib (no dancecap / torch), so recap_avatars.py runs it as a subprocess:

    .venv-face/Scripts/python.exe dancecap/dancecap/recap_face_lm.py --jobs jobs.json --out landmarks.json

jobs.json = {"mp_task": <face_landmarker.task>, "rtmw_onnx": <rtmw end2end.onnx>,
             "images": [{"name": str, "file": path, "face": [cx, cy, side] | null}]}
face = a square (centre x, centre y, side in px of the image) around the face: the video frames' face boxes come from the 2D
pose keypoints (nose / eyes / ears); null = the whole image (a portrait). Every image: MediaPipe FaceLandmarker (478 points)
in two passes (the given crop, then a crop centred on the first answer), the facial transformation matrix (yaw / pitch /
roll), RTMW-x 384x288 whole-body iBUG-68 face points (top-down on a box around the face) and a sharpness value. The record
is exactly what dancecap.face_identity.fit_images reads (the schema of the 2026-10-08 face_shape_v4 landmarks.json):

    {name, file, ok, size, yaw, pitch, roll, mp478 (478x3 px of the FULL image), smirk_center, smirk_side, rtmw68,
     rtmw68_score, occ_px (outer-canthal distance), sharpness, blend}

Weights (open, ungated): MediaPipe face_landmarker.task (float16 v1, Apache-2.0, storage.googleapis.com/mediapipe-models),
RTMW-x 384x288 ONNX (Apache-2.0, openmmlab, via rtmlib). Likeness data: the outputs stay under work/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def landmarker(task: str):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    o = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(task)),
                                     running_mode=vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.3,
                                     min_face_presence_confidence=0.3, output_facial_transformation_matrixes=True,
                                     output_face_blendshapes=True)
    return mp, vision.FaceLandmarker.create_from_options(o)


def detect_crop(mp, lm, img, box=None, size=512):
    """-> (P (478,3) px of the FULL image with z in x-px scale, M 4x4, blend dict) or None. box = square (x0, y0, side)."""
    if box is None:
        x0, y0, s, crop = 0.0, 0.0, 1.0, img
    else:
        x0, y0, side = box
        s = size / side
        A = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
        crop = cv2.warpAffine(img, A, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    h, w = crop.shape[:2]
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop[:, :, ::-1])))
    if not r.face_landmarks:
        return None
    f = r.face_landmarks[0]
    P = np.array([[q.x * w, q.y * h, q.z * w] for q in f], np.float64)
    P[:, :2] = P[:, :2] / s + [x0, y0]
    P[:, 2] /= s
    M = np.asarray(r.facial_transformation_matrixes[0], float)
    bl = {b.category_name: float(b.score) for b in r.face_blendshapes[0]} if r.face_blendshapes else {}
    return P, M, bl


def detect(mp, lm, img, face=None):
    """Two passes: the given crop (or the whole image, then a centred square for a small face), then a crop centred on the
    first answer (1.6 x its extent)."""
    if face is not None:
        c, side = np.array(face[:2], float), float(face[2])
        d = detect_crop(mp, lm, img, (c[0] - side / 2, c[1] - side / 2, side))
    else:
        d = detect_crop(mp, lm, img)
        if d is None:
            H, W = img.shape[:2]
            side = min(H, W)
            d = detect_crop(mp, lm, img, ((W - side) / 2, (H - side) / 2, side))
    if d is None:
        return None
    P = d[0]
    c = P[:, :2].mean(0)
    side = 1.6 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    return detect_crop(mp, lm, img, (c[0] - side / 2, c[1] - side / 2, side)) or d


def pose(M):
    R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0, keepdims=True)
    yaw = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])))
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def sharpness(img, P) -> float:
    x0, y0 = np.maximum(P[:, :2].min(0), 0).astype(int)
    x1, y1 = P[:, :2].max(0).astype(int)
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()) if g.size else 0.0


class Rtmw:
    def __init__(self, onnx: str):
        from rtmlib import RTMPose
        self.m = RTMPose(onnx_model=str(onnx), model_input_size=(288, 384), backend="onnxruntime", device="cpu")

    def __call__(self, img, box=None):
        H, W = img.shape[:2]
        kp, sc = self.m(img, bboxes=[box or [0, 0, W, H]])
        return kp[0, 23:91].astype(float), sc[0, 23:91].astype(float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.jobs).read_text())
    mp, lm = landmarker(spec["mp_task"])
    rt = Rtmw(spec["rtmw_onnx"])
    res = []
    for job in spec["images"]:
        img = cv2.imread(job["file"])
        name = job["name"]
        if img is None:
            res.append({"name": name, "file": job["file"], "ok": False, "why": "unreadable"})
            print(name, "UNREADABLE", flush=True)
            continue
        H, W = img.shape[:2]
        d = detect(mp, lm, img, job.get("face"))
        if d is None:
            res.append({"name": name, "file": job["file"], "ok": False, "why": "no face"})
            print(name, "NO FACE", flush=True)
            continue
        P, M, bl = d
        yaw, pitch, roll = pose(M)
        lo, hi = P[:, :2].min(0), P[:, :2].max(0)
        sc = float((hi - lo).sum() / 2 * 1.4)
        cen = (lo + hi) / 2
        bb = [float(cen[0] - 1.6 * sc), float(cen[1] - 1.2 * sc), float(cen[0] + 1.6 * sc), float(cen[1] + 2.4 * sc)]
        bb = [max(0.0, bb[0]), max(0.0, bb[1]), min(float(W), bb[2]), min(float(H), bb[3])]
        K, ks = rt(img, bb if job.get("face") is not None else None)
        rec = {"name": name, "file": job["file"], "ok": True, "size": [W, H], "yaw": yaw, "pitch": pitch, "roll": roll,
               "mp478": P.round(3).tolist(), "smirk_center": cen.tolist(), "smirk_side": sc, "rtmw68": K.round(3).tolist(),
               "rtmw68_score": ks.round(3).tolist(), "occ_px": float(np.linalg.norm(P[33, :2] - P[263, :2])),
               "sharpness": sharpness(img, P),
               "blend": {k: round(v, 3) for k, v in bl.items() if k.startswith(("mouth", "jaw", "cheek", "eyeBlink"))}}
        res.append(rec)
        print(f"{name}: yaw {yaw:+.1f} pitch {pitch:+.1f} roll {roll:+.1f} occ {rec['occ_px']:.1f}px rtmw "
              f"{float(ks.mean()):.2f} sharp {rec['sharpness']:.0f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
