"""World frame and per-frame camera arrays of the two-camera rig (dancecap.cameras.twocam output -> CONTRACTS.md schemas).

Provisional frame = reference frame of camera 01 (C_01 = 0, |C_02| = 1). world_from_prov = (s, R, t): X_world = s R X_prov + t
(Y up, floor y = 0, metres), the Sim3 that dancecap.cameras.world.bootstrap returns.

  camera_arrays()   per-frame K (F,3,3), R (cam_from_world), tvec, dist, centre C for a camera with a FIXED centre and a per-frame
                    rotation / focal (the schema of cameras/<cam>.npz, CONTRACTS.md; the world IS the "SfM frame" of this take)
  room_box_planes() the room's planes in the world frame from the floor, the back wall line and the ceiling height
"""
from __future__ import annotations

import numpy as np


def camera_arrays(R_cam_from_prov: np.ndarray, f: np.ndarray, c_px: tuple[float, float], k1: np.ndarray | float,
                  C_prov: np.ndarray, s: float, Rw: np.ndarray, tw: np.ndarray) -> dict:
    """Per-frame arrays of one camera in the metric world frame.

    X_cam = R_cw (X_w - C_w) with R_cw = R_cp Rw^T (a rotation, metres) and C_w = s Rw C_prov + tw; tvec = -R_cw C_w."""
    Rcp = np.asarray(R_cam_from_prov, float)
    F = len(Rcp)
    R_cw = Rcp @ Rw.T
    C_w = s * (Rw @ np.asarray(C_prov, float)) + np.asarray(tw, float)
    tvec = -(R_cw @ C_w)
    K = np.zeros((F, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = np.asarray(f, float)
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = c_px[0], c_px[1], 1.0
    return {"K": K, "R": R_cw, "tvec": tvec, "C": np.broadcast_to(C_w, (F, 3)).copy(),
            "dist": np.broadcast_to(np.asarray(k1, float), (F,)).copy()}


def to_world(X_prov: np.ndarray, s: float, Rw: np.ndarray, tw: np.ndarray) -> np.ndarray:
    return s * (np.asarray(X_prov, float) @ np.asarray(Rw, float).T) + np.asarray(tw, float)


def yaw_about_y(angle_deg: float) -> np.ndarray:
    """Rotation matrix about the world Y axis (counter-clockwise seen from above, right-handed Y up)."""
    a = np.radians(angle_deg)
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])


def re_yaw(Rw: np.ndarray, tw: np.ndarray, angle_deg: float, about: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """New (Rw, tw) after turning the world about Y by angle_deg around the point `about` (world coordinates, default origin):
    X' = Ry (X - about) + about."""
    Ry = yaw_about_y(angle_deg)
    about = np.zeros(3) if about is None else np.asarray(about, float)
    return Ry @ Rw, Ry @ (np.asarray(tw, float) - about) + about


def ray_floor_hit(C: np.ndarray, d: np.ndarray, y: float = 0.0) -> np.ndarray:
    """Intersection of rays C + t d (d: (M,3)) with the plane Y = y; NaN where the ray does not go down to it."""
    d = np.asarray(d, float)
    t = (y - C[1]) / np.where(np.abs(d[:, 1]) < 1e-9, 1e-9, d[:, 1])
    out = C[None] + t[:, None] * d
    out[t <= 0] = np.nan
    return out


def pixel_rays_world(px: np.ndarray, K: np.ndarray, R_cw: np.ndarray, k1: float = 0.0, k2: float = 0.0) -> np.ndarray:
    """World-frame unit directions of pixels (M,2) of a camera with intrinsics K (3,3), world->camera rotation R_cw and the radial
    terms of the take's lens model (normalised coordinates)."""
    from .ptz import undistort
    f = K[0, 0]
    xd = (np.asarray(px, float) - K[:2, 2]) / f
    xu = undistort(xd, k1, k2)
    d = np.concatenate([xu, np.ones((len(xu), 1))], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d @ R_cw          # R_cw^T d for each row
