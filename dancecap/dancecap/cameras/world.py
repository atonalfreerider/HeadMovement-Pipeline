"""world_from_sfm bootstrap: gravity, floor, metric scale and origin from the cameras and the two dancers.

  gravity  mean camera up vector (phones are held upright: -row 1 of R), refined by the plane through the
           dancers' triangulated ankle keypoints (COCO 15/16, people/<cam>.npz, dancer boxes from
           dancers/roles.json) - feet are on the floor most of the time, so the LOW ankle points are used.
  floor    ankle plane lowered by the ankle-joint height (0.039 x stature + ~2.5 cm sneaker sole), then
           refined by a RANSAC plane through the sparse SfM points near it when the floor has enough
           texture (scuffs). Points far below the floor are mirror / glossy-floor reflections (counted).
  scale    eye-above-ankle height of each dancer (90th percentile over the window = upright moments)
           vs 0.897 x stature (Drillis & Contini: eye 0.936 H, ankle 0.039 H). Statures come from
           takes/<take>.toml [dancers] (user supplied) or default to lead 1.75 / follow 1.65 m ("assumed").
  axes     Y = up (floor normal), origin = dancers' mean floor position, X = horizontal projection of
           camera 06's mean right vector (reference clock camera), Z = X x Y (right-handed).
X_world = s R (X_sfm) + t  (contract: cameras/world_from_sfm.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EYE_MINUS_ANKLE = 0.936 - 0.039  # fraction of stature
ANKLE_FRAC = 0.039
SOLE_M = 0.025
DEFAULT_STATURE = {"lead": 1.75, "follow": 1.65}
KP_EYES, KP_ANKLES = (1, 2), (15, 16)


def undistort_pts(xy: np.ndarray, K: np.ndarray, k1: float, iters: int = 5) -> np.ndarray:
    """Pixel (OpenCV convention) -> normalised camera coordinates (SIMPLE_RADIAL inverse, fixed point)."""
    x = (xy[..., 0] - K[0, 2]) / K[0, 0]
    y = (xy[..., 1] - K[1, 2]) / K[1, 1]
    xu, yu = x.copy(), y.copy()
    for _ in range(iters):
        d = 1 + k1 * (xu * xu + yu * yu)
        xu, yu = x / d, y / d
    return np.stack([xu, yu], -1)


def triangulate(nxy: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation from normalised coordinates nxy (V,2) and poses R (V,3,3), t (V,3)."""
    A = []
    for (x, y), Ri, ti in zip(nxy, R, t):
        P = np.hstack([Ri, ti[:, None]])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    return X[:3] / X[3]


def triangulate_robust(xy: np.ndarray, K: np.ndarray, k1: np.ndarray, R: np.ndarray, t: np.ndarray,
                       max_px: float = 8.0, min_views: int = 2) -> tuple[np.ndarray | None, float, int]:
    """Triangulate one point seen in V views (pixels), dropping the worst view while its error > max_px."""
    idx = list(range(len(xy)))
    nxy = np.array([undistort_pts(xy[i], K[i], k1[i]) for i in idx])
    while len(idx) >= min_views:
        X = triangulate(nxy[idx], R[idx], t[idx])
        errs = []
        for i in idx:
            Xc = R[i] @ X + t[i]
            if Xc[2] <= 0:
                errs.append(1e6)
                continue
            u = Xc[:2] / Xc[2]
            errs.append(np.linalg.norm((u - nxy[i]) * K[i][0, 0]))
        errs = np.asarray(errs)
        if errs.max() <= max_px:
            return X, float(np.median(errs)), len(idx)
        if len(idx) == min_views:
            break
        idx.pop(int(np.argmax(errs)))
    return None, float("nan"), 0


def fit_plane(P: np.ndarray, thr: float, prior_n: np.ndarray | None = None, max_angle_deg: float = 20.0,
              iters: int = 2000, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane n.x = d (n unit, oriented like prior_n) + least-squares refit. Returns (n, d, inliers)."""
    rng = np.random.default_rng(seed)
    best = None
    cos_max = np.cos(np.radians(max_angle_deg))
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if prior_n is not None:
            if np.dot(n, prior_n) < 0:
                n = -n
            if np.dot(n, prior_n) < cos_max:
                continue
        d = np.dot(n, s[0])
        inl = np.abs(P @ n - d) < thr
        if best is None or inl.sum() > best[2].sum():
            best = (n, d, inl)
    if best is None:
        raise ValueError("no plane")
    n, d, inl = best
    for _ in range(3):  # LS refit on inliers
        c = P[inl].mean(0)
        _, _, vt = np.linalg.svd(P[inl] - c)
        n = vt[-1] * (1 if prior_n is None or np.dot(vt[-1], prior_n) >= 0 else -1)
        d = float(np.dot(n, c))
        inl = np.abs(P @ n - d) < thr
    return n, d, inl


def dancer_keypoints(work: Path, cam: str, n_frames: int) -> dict[str, np.ndarray]:
    """{'lead': (F,17,3), 'follow': (F,17,3)} COCO keypoints of the dancers (zeros where unknown).

    Dancer boxes come from dancers/roles.json; without it the two largest boxes near the image centre are used
    (role unknown -> both under 'lead'/'follow' by size)."""
    from ..detect import iou_matrix
    pp = np.load(work / "people" / f"{cam}.npz")
    boxes, valid, kpts = pp["boxes"], pp["valid"], pp["kpts"]
    out = {r: np.zeros((n_frames, 17, 3), np.float32) for r in ("lead", "follow")}
    roles_p = work / "dancers" / "roles.json"
    roles = json.loads(roles_p.read_text())["cams"].get(cam) if roles_p.exists() else None
    for i in range(min(n_frames, len(boxes))):
        b = boxes[i][valid[i]]
        k = kpts[i][valid[i]]
        if len(b) == 0:
            continue
        if roles is not None:
            fr = roles["frames"][i] if i < len(roles["frames"]) else None
            for role in ("lead", "follow"):
                rb = (fr or {}).get(role)
                if not rb or rb.get("box") is None:
                    continue
                iou = iou_matrix(np.asarray(rb["box"], float)[None], b)[0]
                j = int(np.argmax(iou))
                if iou[j] > 0.5:
                    out[role][i] = k[j]
        else:
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            for role, j in zip(("lead", "follow"), np.argsort(-area)[:2]):
                out[role][i] = k[j]
    return out


def triangulate_dancers(work: Path, tracks: dict, grid_dt: float = 1 / 30, max_dt: float = 0.02,
                        min_conf: float = 0.5, kps: tuple[int, ...] = KP_EYES + KP_ANKLES,
                        return_grid: bool = False):
    """Triangulated COCO keypoints `kps` of both dancers on a 30 Hz grid (SfM frame).

    tracks[cam] = dict(t, K, k1, R, tvec, ok). Returns {role: {kp: (T,3) NaN where unknown}} (+ grid)."""
    t0 = max(tr["t"][0] for tr in tracks.values())
    t1 = min(tr["t"][-1] for tr in tracks.values())
    grid = np.arange(t0, t1, grid_dt)
    kp = {cam: dancer_keypoints(work, cam, len(tr["t"])) for cam, tr in tracks.items()}
    out = {r: {k: np.full((len(grid), 3), np.nan) for k in kps} for r in ("lead", "follow")}
    for g, tg in enumerate(grid):
        sel = []
        for cam, tr in tracks.items():
            i = int(np.argmin(np.abs(tr["t"] - tg)))
            if abs(tr["t"][i] - tg) <= max_dt and tr["ok"][i]:
                sel.append((cam, i))
        if len(sel) < 2:
            continue
        for role in ("lead", "follow"):
            for k in kps:
                v = [(cam, i) for cam, i in sel if kp[cam][role][i, k, 2] >= min_conf]
                if len(v) < 2:
                    continue
                xy = np.array([kp[cam][role][i, k, :2] for cam, i in v], float)
                K = np.array([tracks[cam]["K"][i] for cam, i in v])
                k1 = np.array([tracks[cam]["k1"][i] for cam, i in v])
                R = np.array([tracks[cam]["R"][i] for cam, i in v])
                t = np.array([tracks[cam]["tvec"][i] for cam, i in v])
                X, _, nv = triangulate_robust(xy, K, k1, R, t)
                if X is not None and nv >= 2:
                    out[role][k][g] = X
    return (out, grid) if return_grid else out


def bootstrap(tracks: dict, dancers: dict, points: np.ndarray, statures: dict[str, float],
              stature_source: str, x_cam: str = "06", up_cams: list[str] | None = None) -> dict:
    """Compute the Sim3 (s, R, t) SfM -> world plus diagnostics."""
    ups = []
    for cam, tr in tracks.items():
        if up_cams is not None and cam not in up_cams:
            continue
        ups.append(-tr["R"][tr["ok"], 1, :])
    up_cam = np.concatenate(ups).mean(0)
    up_cam /= np.linalg.norm(up_cam)
    diag: dict = {"up_from_cameras": up_cam.tolist()}

    ank = np.concatenate([dancers[r][k] for r in dancers for k in KP_ANKLES])
    ank = ank[np.isfinite(ank).all(1)]
    diag["ankle_points"] = int(len(ank))
    if len(ank) < 20:
        raise ValueError(f"only {len(ank)} triangulated ankle points: cannot bootstrap the floor")
    h = ank @ up_cam
    low = ank[h <= np.percentile(h, 40)]
    spread = np.percentile(h, 90) - np.percentile(h, 10)
    n, d, inl = fit_plane(low, thr=max(0.05 * spread, 1e-6), prior_n=up_cam, max_angle_deg=25)
    diag["ankle_plane_inliers"] = int(inl.sum())
    diag["gravity_vs_cameras_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(n, up_cam), -1, 1))))
    gravity_source = "ankle plane (dancers' triangulated low ankle keypoints), init mean camera up"
    if diag["gravity_vs_cameras_deg"] > 15:  # implausible: phones are roughly upright
        n, gravity_source = up_cam, "mean camera up vector (ankle plane disagreed by > 15 deg)"
        d = float(np.percentile(h, 20))

    # scale from eye-above-ankle-plane heights
    scales = {}
    for role in ("lead", "follow"):
        eyes = np.nanmean(np.stack([dancers[role][k] for k in KP_EYES]), 0)
        he = eyes @ n - d
        he = he[np.isfinite(he)]
        if len(he) < 10:
            continue
        h90 = float(np.percentile(he, 90))
        scales[role] = {"eye_above_ankle_sfm": h90, "n": int(len(he)),
                        "s": EYE_MINUS_ANKLE * statures[role] / h90, "stature_m": statures[role]}
    if not scales:
        raise ValueError("no dancer eye heights: cannot set the metric scale")
    s = float(np.mean([v["s"] for v in scales.values()]))
    diag["scale_per_dancer"] = scales
    if len(scales) == 2:
        a, b = (v["s"] for v in scales.values())
        diag["scale_disagreement"] = float(abs(a - b) / s)

    # floor = ankle plane lowered by the ankle height; refine with sparse floor points
    ankle_h = ANKLE_FRAC * np.mean(list(statures.values())) + SOLE_M
    d_floor = d - ankle_h / s
    floor_source = "ankle plane - ankle height"
    if points is not None and len(points) > 50:
        dist = (points @ n - d_floor) * s  # metres above floor
        near = points[np.abs(dist) < 0.06]
        diag["sfm_points_near_floor"] = int(len(near))
        if len(near) > 100:
            n2, d2, inl2 = fit_plane(near, thr=0.015 / s, prior_n=n, max_angle_deg=6)
            off = (d2 - d_floor) * s
            diag["sfm_floor_plane"] = {"inliers": int(inl2.sum()), "tilt_vs_ankle_deg":
                                       float(np.degrees(np.arccos(np.clip(np.dot(n2, n), -1, 1)))),
                                       "offset_vs_ankle_m": float(off)}
            if inl2.sum() > 100 and abs(off) < 0.08:
                n, d_floor = n2, d2
                floor_source = "RANSAC plane through sparse SfM floor points (init: dancers' ankle plane)"
        below = (points @ n - d_floor) * s < -0.15
        diag["points_below_floor"] = int(below.sum())  # glossy floor / mirror reflections
        diag["points_total"] = int(len(points))

    # origin: dancers' mean floor position
    ank_low = ank[(ank @ n - d_floor) * s < 0.2]
    o = (ank_low if len(ank_low) else ank).mean(0)
    o = o - (np.dot(o, n) - d_floor) * n
    # X axis from camera x_cam's mean right vector
    good = [c for c in sorted(tracks) if tracks[c]["ok"].any()]
    cam = x_cam if x_cam in good else good[0]
    r = tracks[cam]["R"][tracks[cam]["ok"], 0, :].mean(0)
    x = r - np.dot(r, n) * n
    x /= np.linalg.norm(x)
    z = np.cross(x, n)
    Rw = np.stack([x, n, z])
    t = -s * Rw @ o
    return {"s": s, "R": Rw.tolist(), "t": t.tolist(), "gravity_source": gravity_source,
            "scale_source": f"dancers' eye-above-ankle height vs stature ({stature_source}: "
                            + ", ".join(f"{k} {v:.3f} m" for k, v in statures.items()) + ")",
            "floor_source": floor_source, "x_axis": f"camera {cam} mean right vector (horizontal)",
            "diagnostics": diag}


def statures_from_take(cfg: dict) -> tuple[dict[str, float], str]:
    d = cfg.get("dancers", {})
    st = {r: float(d[r]["stature_m"]) for r in ("lead", "follow") if r in d and "stature_m" in d[r]}
    if len(st) == 2:
        return st, "user-supplied statures (takes/<take>.toml)"
    return dict(DEFAULT_STATURE), "assumed statures"


def apply(sim3: dict, X: np.ndarray) -> np.ndarray:
    return float(sim3["s"]) * X @ np.asarray(sim3["R"]).T + np.asarray(sim3["t"])
