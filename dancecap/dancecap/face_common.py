"""Shared face geometry for the face stages (face_survey, face_eval, face_expr, avatar_face).

SMPL-X carries the iBUG-68 landmark embedding of its FLAME-compatible head: 51 static landmarks (iBUG 17-67: brows,
nose, eyes, mouth; `lmk_faces_idx` + `lmk_bary_coords`) and a yaw-dependent 17-point contour
(`dynamic_lmk_faces_idx` (79, 17), yaw -39..+39 deg). We use the 51 static points + the chin (contour point 8 of the
frontal contour, which barely moves with yaw) = LMK52 in iBUG numbering IBUG52 = (8, 17, 18, ..., 67).

Detector conventions mapped onto iBUG-68:
  * MediaPipe Face Landmarker (478 points): MP_TO_IBUG68 below (the widely used 68-point subset; verified by eye on
    the portraits, review/face_portrait_landmarks.jpg).
  * RTMW / COCO-WholeBody (133 points): face = keypoints 23..90 in iBUG-68 order.

Pixel / camera conventions are dancecap's (CONTRACTS.md): OpenCV x_cam = R X + t, K per frame, SIMPLE_RADIAL k1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROLES = ("lead", "follow")
PIDS = {"lead": 1, "follow": 2}
LABEL = {"lead": "LEAD (Kadu)", "follow": "FOLLOW (Larissa)"}

IBUG52 = np.array([8] + list(range(17, 68)))           # chin + 51 static landmarks
IBUG_EYE_OUTER = (36, 45)
IBUG_MOUTH = tuple(range(48, 68))
IBUG_INNER_LIPS = tuple(range(60, 68))

# MediaPipe 478 -> iBUG 68 (jaw 0-16, brows 17-26, nose 27-35, eyes 36-47, outer lips 48-59, inner lips 60-67)
MP_TO_IBUG68 = np.array([
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87])
RTMW_FACE = slice(23, 91)


def smplx_model_npz() -> Path:
    from .mamma_dyn import SMPLX_MODELS
    return Path(SMPLX_MODELS) / "smplx" / "SMPLX_NEUTRAL.npz"


class LandmarkEmbedding:
    """iBUG landmarks on the SMPL-X surface: (faces (L,3) vertex ids, bary (L,3)) for IBUG52."""

    def __init__(self, model_npz: Path | None = None):
        m = np.load(model_npz or smplx_model_npz(), allow_pickle=True)
        f = np.asarray(m["f"], np.int64)
        st_f = np.asarray(m["lmk_faces_idx"], np.int64)          # (51,)
        st_b = np.asarray(m["lmk_bary_coords"], float)           # (51, 3)
        dyn_f = np.asarray(m["dynamic_lmk_faces_idx"], np.int64)  # (79, 17)
        dyn_b = np.asarray(m["dynamic_lmk_bary_coords"], float)  # (79, 17, 3)
        mid = dyn_f.shape[0] // 2                                 # yaw 0
        self.faces = np.concatenate([f[dyn_f[mid, 8]][None], f[st_f]], 0)
        self.bary = np.concatenate([dyn_b[mid, 8][None], st_b], 0)
        self.ibug = IBUG52.copy()

    def points(self, verts: np.ndarray) -> np.ndarray:
        """(..., V, 3) -> (..., 52, 3)."""
        return np.einsum("...lkd,lk->...ld", verts[..., self.faces, :], self.bary)

    def index(self, ibug_ids) -> np.ndarray:
        pos = {int(k): n for n, k in enumerate(self.ibug)}
        return np.array([pos[int(k)] for k in ibug_ids])


def project(P: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, k1: float = 0.0):
    """World points (N,3) -> pixels (N,2), depth (N,)."""
    Xc = P @ R.T + t
    z = Xc[:, 2]
    zc = np.maximum(z, 1e-4)
    x, y = Xc[:, 0] / zc, Xc[:, 1] / zc
    d = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], 1), z


def kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t with B ~ R A + t (rows are points)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def head_frame(lmk_rest: np.ndarray, lmk_posed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation of the head (rest -> posed, from the rigid upper-face landmarks) and the face centre.
    SMPL-X rest pose looks along +z, so forward = R[:, 2]."""
    rigid = np.r_[1:11, 11:20, 20:32]   # brows, nose, eyes (positions in LMK52; mouth/chin excluded: they move)
    R, _ = kabsch(lmk_rest[rigid], lmk_posed[rigid])
    return R, lmk_posed[rigid].mean(0)


def frames_meta(work: Path, cam: str) -> dict:
    return json.loads((work / "frames" / cam / "frames.json").read_text())


def cameras_world(work: Path, cams: list[str]) -> dict:
    from .mamma_dyn import world_cameras
    sim3_p = work / "cameras" / "world_from_sfm.json"
    sim3 = json.loads(sim3_p.read_text()) if sim3_p.exists() else None
    out = {}
    for c in cams:
        z = np.load(work / "cameras" / f"{c}.npz")
        Rw, tw = world_cameras(z["R"], z["tvec"], sim3)
        out[c] = {"K": z["K"], "R": Rw, "t": tw, "k1": z["dist"] if "dist" in z.files else np.zeros(len(Rw)),
                  "ok": z["ok"] if "ok" in z.files else np.ones(len(Rw), bool)}
    return out


def take_cams(work: Path) -> list[str]:
    return sorted(p.stem for p in (work / "cameras").glob("*.npz") if p.stem.isdigit())


def square_box(center, side, W, H):
    """Integer square box (x0, y0, x1, y1), may extend past the image (crops pad)."""
    cx, cy = center
    h = side / 2
    return (int(round(cx - h)), int(round(cy - h)), int(round(cx + h)), int(round(cy + h)))


def crop_pad(img: np.ndarray, box, out: int | None = None, interp=None):
    import cv2
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    c = img[y0:y1, x0:x1]
    if out is not None and c.size:
        c = cv2.resize(c, (out, out), interpolation=interp if interp is not None else
                       (cv2.INTER_CUBIC if out > c.shape[0] else cv2.INTER_AREA))
    return c


def label(img, text, org, color=(255, 255, 255), scale=0.5, thick=1):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return img
