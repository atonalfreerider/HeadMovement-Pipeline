"""Draw a room model (floor grid, walls, axes, points) into a video frame through a calibrated camera - review images.

A camera here is the per-frame model of the carlos_aline rig: world -> camera  X_c = R X_w + tvec  (R = cam_from_world, rows are
the camera axes), pixels  p = c + f * x_d,  x_d = x_u (1 + k1 r_u^2),  x_u = X_c[:2] / X_c[2],  r_u in normalised units.
"""
from __future__ import annotations

import cv2
import numpy as np


def project(Xw: np.ndarray, R: np.ndarray, tvec: np.ndarray, f: float, c: tuple[float, float], k1: float = 0.0,
            k2: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(M,3) world points -> (M,2) pixels and the depth (M,) (negative / tiny = behind the camera)."""
    Xc = Xw @ R.T + tvec
    z = Xc[:, 2]
    zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
    xu = Xc[:, :2] / zz[:, None]
    r2 = (xu ** 2).sum(1, keepdims=True)
    xd = xu * (1 + k1 * r2 + k2 * r2 * r2)
    return xd * f + np.asarray(c), z


def _polyline(img: np.ndarray, pts: np.ndarray, z: np.ndarray, color, thickness: int, zmin: float = 0.2) -> None:
    """Draw a polyline, cutting it where it leaves the front half-space (so a floor grid running behind the camera is clipped)."""
    ok = z > zmin
    h, w = img.shape[:2]
    for i in range(len(pts) - 1):
        if not (ok[i] and ok[i + 1]):
            continue
        a, b = pts[i], pts[i + 1]
        if max(abs(a).max(), abs(b).max()) > 1e5:
            continue
        cv2.line(img, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), color, thickness, cv2.LINE_AA)


def draw_segment3d(img, A, B, cam, color, thickness=2, n=60):
    t = np.linspace(0, 1, n)[:, None]
    X = A[None] * (1 - t) + B[None] * t
    p, z = project(X, **cam)
    _polyline(img, p, z, color, thickness)


def draw_floor_grid(img: np.ndarray, cam: dict, x_range: tuple[float, float], z_range: tuple[float, float], step: float = 1.0,
                    y: float = 0.0, color=(0, 220, 255), thickness: int = 1, major: int = 5) -> None:
    xs = np.arange(np.ceil(x_range[0] / step) * step, x_range[1] + 1e-6, step)
    zs = np.arange(np.ceil(z_range[0] / step) * step, z_range[1] + 1e-6, step)
    for x in xs:
        major_line = abs(round(x / step)) % major == 0
        draw_segment3d(img, np.array([x, y, z_range[0]]), np.array([x, y, z_range[1]]), cam, color,
                       thickness + (1 if major_line else 0), n=max(20, int(60 * (z_range[1] - z_range[0]) / 10)))
    for z in zs:
        major_line = abs(round(z / step)) % major == 0
        draw_segment3d(img, np.array([x_range[0], y, z]), np.array([x_range[1], y, z]), cam, color,
                       thickness + (1 if major_line else 0), n=max(20, int(60 * (x_range[1] - x_range[0]) / 10)))


def draw_axes(img: np.ndarray, cam: dict, origin=(0.0, 0.0, 0.0), length: float = 1.0) -> None:
    o = np.asarray(origin, float)
    for axis, col, name in ((0, (0, 0, 255), "X"), (1, (0, 255, 0), "Y"), (2, (255, 80, 0), "Z")):
        e = np.zeros(3)
        e[axis] = length
        draw_segment3d(img, o, o + e, cam, col, 3, 20)
        p, z = project((o + e)[None], **cam)
        if z[0] > 0.2:
            cv2.putText(img, name, (int(p[0, 0]) + 4, int(p[0, 1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)


def draw_quad(img, corners, cam, color, thickness=3, label=None):
    C = np.asarray(corners, float)
    for i in range(len(C)):
        draw_segment3d(img, C[i], C[(i + 1) % len(C)], cam, color, thickness, 40)
    if label:
        p, z = project(C.mean(0)[None], **cam)
        if z[0] > 0.2:
            cv2.putText(img, label, (int(p[0, 0]), int(p[0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def draw_points(img, X, cam, color=(0, 255, 0), radius=4):
    p, z = project(np.asarray(X, float).reshape(-1, 3), **cam)
    h, w = img.shape[:2]
    for (x, y), zz in zip(p, z):
        if zz > 0.2 and 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def contact_sheet(tiles: list[np.ndarray], cols: int, tile_w: int, labels: list[str] | None = None) -> np.ndarray:
    ims = []
    for i, t in enumerate(tiles):
        s = tile_w / t.shape[1]
        im = cv2.resize(t, (tile_w, int(round(t.shape[0] * s))), interpolation=cv2.INTER_AREA)
        if labels:
            cv2.putText(im, labels[i], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(im, labels[i], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        ims.append(im)
    th = max(i.shape[0] for i in ims)
    rows = []
    for r in range(0, len(ims), cols):
        row = [np.pad(i, ((0, th - i.shape[0]), (0, 0), (0, 0))) for i in ims[r:r + cols]]
        while len(row) < cols:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)
