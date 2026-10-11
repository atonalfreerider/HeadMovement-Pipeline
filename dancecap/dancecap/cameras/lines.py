"""Straight line segments of the static background of a video frame (LSD), people masked, plus the geometry to carry them to rays.

A segment's two end rays (after undistortion with the lens of the solved camera) span a plane through the camera centre with unit
normal n = r1 x r2 / |r1 x r2|; a family of parallel 3D lines of direction d has n . d = 0 for all of them, so the normals are what
the vanishing-point / Manhattan-frame fits work on (dancecap.cameras.manhattan). Pure numpy apart from cv2's LSD.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect(gray: np.ndarray, ignore: np.ndarray | None = None, min_len: float = 28.0, border: int = 6, dilate: int = 9) -> np.ndarray:
    """(N,5) x1 y1 x2 y2 length of the LSD segments longer than min_len px whose both ends and midpoint are outside `ignore` (uint8 mask, 255 = ignore)."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    seg = lsd.detect(gray)[0]
    if seg is None:
        return np.zeros((0, 5), np.float32)
    seg = seg.reshape(-1, 4)
    ln = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    h, w = gray.shape
    keep = ln >= min_len
    for k in (0, 2):
        keep &= (seg[:, k] > border) & (seg[:, k] < w - border) & (seg[:, k + 1] > border) & (seg[:, k + 1] < h - border)
    if ignore is not None:
        m = cv2.dilate(ignore, np.ones((dilate, dilate), np.uint8)) if dilate else ignore
        for t in (0.0, 0.5, 1.0):
            x = np.clip((seg[:, 0] + t * (seg[:, 2] - seg[:, 0])).astype(int), 0, w - 1)
            y = np.clip((seg[:, 1] + t * (seg[:, 3] - seg[:, 1])).astype(int), 0, h - 1)
            keep &= m[y, x] == 0
    return np.concatenate([seg, ln[:, None]], 1)[keep].astype(np.float32)


def plane_normals(seg: np.ndarray, f: float, c: np.ndarray, k1: float = 0.0, k2: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Unit plane normals (N,3) of the segments and the unit direction (N,3) of the segment's middle ray, camera frame (x right, y down, z forward)."""
    from .ptz import Lens, pix_to_ray
    lens = Lens(int(2 * c[0]), int(2 * c[1]), k1, k2)
    r1 = pix_to_ray(seg[:, 0:2].astype(float), np.full(len(seg), f), lens)
    r2 = pix_to_ray(seg[:, 2:4].astype(float), np.full(len(seg), f), lens)
    n = np.cross(r1, r2)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    mid = r1 + r2
    mid /= np.maximum(np.linalg.norm(mid, axis=1, keepdims=True), 1e-12)
    return n, mid
