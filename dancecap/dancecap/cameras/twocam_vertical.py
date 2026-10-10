"""Architecture vertical of a panning camera, in the camera's fixed REFERENCE frame (dancecap.cameras.vertical machinery).

Vertical edges of the room (panel borders, door frames, pillars) are an independent gravity reference. A pan / tilt / zoom camera
sees them at a different place in every frame; with the per-frame rotation R(t) and focal f(t) of dancecap.cameras.ptz every
segment's interpretation-plane normal can be carried into the reference frame (n_ref = R^T n_cam) and ALL frames pool into one
robust fit  n_ref . v = 0  (vertical.solve_vertical: hypotheses from pairs of long segments, Tukey IRLS).

v is returned oriented UP (the camera's -y side), in the reference frame of the camera; `Track.at()` gives R(t), f(t).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import ptz as P
from . import ptz_feats as F
from . import vertical as V


def frame_normals(img: np.ndarray, ignore: np.ndarray | None, R: np.ndarray, f: float, lens: P.Lens,
                  min_len_frac: float = 0.035) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpretation-plane normals of the frame's line segments in the REFERENCE frame, their lengths (px) and the segments."""
    seg = V.detect_segments(img, ignore, min_len_frac=min_len_frac)
    if len(seg) == 0:
        return np.zeros((0, 3)), np.zeros(0), seg
    p = (seg.reshape(-1, 2) - lens.c) / f
    p = P.undistort(p, lens.k1, lens.k2).reshape(-1, 2, 2)
    Pn = np.concatenate([p, np.ones((len(p), 2, 1))], 2)
    n = np.cross(Pn[:, 0], Pn[:, 1])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return n @ R, L, seg          # row vector form of R^T n


def architecture_vertical(video: str, width: int, height: int, track, frames: list[int], rects=(), yolo: str = "yolo11x-seg.pt",
                          gate_deg: float = 15.0, c_deg: float = 2.0) -> dict:
    """Pool the segments of `frames` (video frame indices, a `ptz_run`-style Track gives R and f) into one vertical."""
    Ns, Ws, up0, per = [], [], [], []
    for fi, im in F.read_video_frames(video, width, height, frames=frames):
        R, f = track.at(np.array([float(fi)]))
        pm = F.person_masks_yolo([im], yolo)[0]
        ign = cv2.dilate(F.static_ignore(im.shape[:2], pm, list(rects)), np.ones((21, 21), np.uint8))
        n, L, seg = frame_normals(im, ign, R[0], float(f[0]), track.lens)
        if len(n) < 8:
            continue
        Ns.append(n)
        Ws.append(L)
        up0.append(-R[0].T @ np.array([0.0, 1.0, 0.0]))
        per.append((int(fi), int(len(n))))
    if not Ns:
        return {"ok": False}
    N, W = np.concatenate(Ns), np.concatenate(Ws)
    v0 = np.mean(up0, 0)
    v0 /= np.linalg.norm(v0)
    r = V.solve_vertical(N, W, v0, gate_deg=gate_deg, c_deg=c_deg)
    if r is None:
        return {"ok": False, "n_segments": int(len(N))}
    return {"ok": True, "v": r["v"], "n_frames": len(up0), "n_segments": int(len(N)), "n_inliers": r["n_inliers"],
            "sigma_deg": r["sigma_deg"], "resid_deg_median": r["resid_deg_median"],
            "angle_to_mean_camera_up_deg": float(np.degrees(np.arccos(np.clip(r["v"] @ v0, -1, 1))))}
