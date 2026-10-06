"""Attach a camera that did not register in the main SfM model, using the floor and the dancers.

Some phones see a part of the room that no other camera sees well (cam05: the shutter wall from up close,
cam06: the far wall), so static features cannot link them to the main model (weak links registered cam06 at a
degenerate, smeared position). Such a camera still forms its OWN sub-model (keyframes + window frames) in which
its per-frame poses and focal lengths are accurate up to an unknown similarity

    X_main = s Rs X_sub + T,  R_main = R_sub Rs^T,  t_main = s t_sub - R_main T.

The 7 numbers come from two independent sources:
  floor    the sub-model's floor plane (RANSAC through its sparse floor points) must map onto the main floor
           (from the world bootstrap): fixes tilt (2 dof) and, with the camera height, the scale;
  dancers  the dancers' COCO keypoints, triangulated in the main frame from the registered cameras, reprojected
           into this camera's window frames (Huber): fixes yaw and position.
Dancer keypoints alone are NOT enough: the dancers fill a small part of the view, so focal/depth and the scale
of the camera's own motion are unobservable (s collapses to 0). Frame-to-frame motion and zoom stay those of the
static sub-model.

ROTATION-ONLY variant (solve_rotonly): a phone that mostly rotates (cam05/06 on larissa_kadu) gives a sub-model
without parallax - points near infinity, no usable floor, camera "translations" meaningless - but its per-frame
ROTATIONS and focal lengths are still precise (self-calibration from rotation). Then the camera centre is held
constant (hand shake of a few cm is ignored) and only Rs (3) + C (3) are solved from the dancers:
    R_main(i) = R_sub(i) Rs^T,   t_main(i) = -R_main(i) C.
The pipeline tries both and keeps the one with the lower dancer reprojection error.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def compose(R_sub: np.ndarray, t_sub: np.ndarray, rv: np.ndarray, log_s: float, T: np.ndarray):
    Rs = Rotation.from_rotvec(rv).as_matrix()
    s = np.exp(log_s)
    R_m = R_sub @ Rs.T
    t_m = s * t_sub - np.einsum("fij,j->fi", R_m, T)
    return R_m, t_m


def _reproj(x, R_sub, t_sub, K, k1, fidx, X, uv):
    R_m, t_m = compose(R_sub, t_sub, x[:3], x[3], x[4:7])
    Xc = np.einsum("nij,nj->ni", R_m[fidx], X) + t_m[fidx]
    z = Xc[:, 2]
    vis = z > 0.05
    zs = np.where(vis, z, 1.0)
    xn, yn = Xc[:, 0] / zs, Xc[:, 1] / zs
    d = 1 + k1[fidx] * (xn * xn + yn * yn)
    Kf = K[fidx]
    r = np.stack([Kf[:, 0, 0] * xn * d + Kf[:, 0, 2], Kf[:, 1, 1] * yn * d + Kf[:, 1, 2]], 1) - uv
    r[~vis] = 200.0
    return r


def _residuals(x, R_sub, t_sub, K, k1, fidx, X, uv, w, floor, lam):
    r = (_reproj(x, R_sub, t_sub, K, k1, fidx, X, uv) * w[:, None]).ravel()
    if floor is None:
        return r
    n_s, h_s, n_m, d_m, C_s = floor
    Rs = Rotation.from_rotvec(x[:3]).as_matrix()
    s = np.exp(x[3])
    C_m = s * C_s @ Rs.T + x[4:7]
    r_n = lam["normal"] * (Rs @ n_s - n_m)  # sub floor normal -> main floor normal
    r_h = lam["height"] * (C_m @ n_m - d_m - s * h_s)  # camera height above floor agrees in both frames
    return np.concatenate([r, r_n, r_h])


def floor_in_model(points: np.ndarray, C: np.ndarray, up: np.ndarray, min_inliers: int = 100):
    """Floor plane of a (sub-)model: RANSAC plane through points below the cameras, normal near `up`.

    Returns (n (unit, pointing up), d, n_inliers) with n.X = d, or None. The glossy floor mirrors the room, but
    mirrored points are spread in depth, so the dense scuff-mark plane wins the RANSAC vote."""
    from .world import fit_plane
    c = C.mean(0)
    h = (points - c) @ up
    below = points[h < 0]
    if len(below) < min_inliers:
        return None
    scale = np.median(np.linalg.norm(points - c, axis=1))
    try:
        n, d, inl = fit_plane(below, thr=0.01 * scale, prior_n=up, max_angle_deg=25)
    except ValueError:
        return None
    if inl.sum() < min_inliers:
        return None
    return n, float(d), int(inl.sum())


def observations(kp2d: dict[str, np.ndarray], t_frames: np.ndarray, ok: np.ndarray, dancers3d: dict, grid: np.ndarray,
                 min_conf: float = 0.5, max_dt: float = 0.02):
    """Stack (frame index, 3D main-frame keypoint, 2D keypoint, weight) for frames with an ok sub-model pose."""
    fi, X, uv, w = [], [], [], []
    for role, kp in kp2d.items():
        for i in np.nonzero(ok)[0]:
            g = int(np.argmin(np.abs(grid - t_frames[i])))
            if abs(grid[g] - t_frames[i]) > max_dt:
                continue
            for k, P in dancers3d[role].items():
                if kp[i, k, 2] >= min_conf and np.isfinite(P[g]).all():
                    fi.append(i), X.append(P[g]), uv.append(kp[i, k, :2]), w.append(1.0)
    return (np.asarray(fi, int), np.asarray(X, float).reshape(-1, 3), np.asarray(uv, float).reshape(-1, 2),
            np.asarray(w, float))


def init_rotation(R_sub: np.ndarray, t_sub: np.ndarray, K: np.ndarray, fidx, X, uv) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame PnP on the dancer points (fixed intrinsics) -> chordal mean of Rs and main-frame centres."""
    import cv2
    Rs_list, C_m, C_s = [], [], []
    for f in np.unique(fidx):
        sel = fidx == f
        if sel.sum() < 10:
            continue
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X[sel], uv[sel], K[f], None, reprojectionError=12.0,
                                                 iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 8:
            continue
        R_m = cv2.Rodrigues(rvec)[0]
        Rs_list.append(R_m.T @ R_sub[f])  # R_m = R_sub Rs^T  ->  Rs = R_m^T R_sub
        C_m.append(-R_m.T @ tvec.ravel())
        C_s.append(-R_sub[f].T @ t_sub[f])
    if not Rs_list:
        raise ValueError("no frame with enough dancer keypoints for PnP")
    U, _, Vt = np.linalg.svd(np.mean(Rs_list, 0))  # chordal mean, projected back onto SO(3)
    Rs = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return Rs, np.asarray(C_m), np.asarray(C_s)


def _align_normal(Rs: np.ndarray, n_s: np.ndarray, n_m: np.ndarray) -> np.ndarray:
    """Smallest rotation that makes Rs n_s = n_m (keeps the yaw of Rs)."""
    a = Rs @ n_s
    v = np.cross(a, n_m)
    if np.linalg.norm(v) < 1e-12:
        return Rs if np.dot(a, n_m) > 0 else Rotation.from_rotvec([np.pi, 0, 0]).as_matrix() @ Rs
    ang = np.arctan2(np.linalg.norm(v), np.dot(a, n_m))
    return Rotation.from_rotvec(v / np.linalg.norm(v) * ang).as_matrix() @ Rs


def solve_sim3(R_sub, t_sub, K, k1, fidx, X, uv, w, floor_sub=None, floor_main=None, huber_px: float = 6.0,
               n_scales: int = 9, n_yaw: int = 8, lam_normal: float = 500.0, lam_height: float = 1000.0) -> dict:
    """Multi-start robust least squares for (Rs, s, T).

    floor_sub = (n_s, d_s) in the sub-model, floor_main = (n_m, d_m) in the main frame (both n.X = d, n up).
    lam_* convert the floor residuals (unit vector / main-frame units ~ metres) into pixel-equivalents."""
    C_s = -np.einsum("fji,fj->fi", R_sub, t_sub)
    floor = None
    lam = {"normal": lam_normal, "height": lam_height}
    if floor_sub is not None and floor_main is not None:
        n_s, d_s = floor_sub
        n_m, d_m = floor_main
        floor = (np.asarray(n_s, float), C_s @ n_s - d_s, np.asarray(n_m, float), float(d_m), C_s)
    try:
        Rs0, C_m0, C_s0 = init_rotation(R_sub, t_sub, K, fidx, X, uv)
    except ValueError:
        Rs0, C_m0, C_s0 = np.eye(3), np.zeros((1, 3)), C_s[:1]
    starts = []
    if floor is not None:
        Rs0 = _align_normal(Rs0, floor[0], floor[2])
        T0 = np.median(C_m0, 0)
        h_m = T0 @ floor[2] - floor[3]
        s0 = max(h_m, 0.5) / max(np.median(floor[1]), 1e-9)
        for yaw in np.linspace(0, 2 * np.pi, n_yaw, endpoint=False):  # PnP yaw can be off: multi-start
            Ry = Rotation.from_rotvec(floor[2] * yaw).as_matrix() @ Rs0
            for ls in np.log(s0) + np.linspace(-0.7, 0.7, 3):
                starts.append((Ry, ls))
    else:
        sp_m = np.linalg.norm(C_m0 - C_m0.mean(0), axis=1).mean()
        sp_s = np.linalg.norm(C_s0 - C_s0.mean(0), axis=1).mean()
        s_c = sp_m / sp_s if sp_s > 1e-9 and sp_m > 1e-9 else 1.0
        starts = [(Rs0, ls) for ls in np.log(s_c) + np.linspace(-3, 3, n_scales)]
    best = None
    args = (R_sub, t_sub, K, k1, fidx, X, uv, w, floor, lam)
    for Rs_i, ls in starts:
        T0 = np.median(C_m0 - np.exp(ls) * C_s0 @ Rs_i.T, axis=0)
        x0 = np.r_[Rotation.from_matrix(Rs_i).as_rotvec(), ls, T0]
        r = least_squares(_residuals, x0, args=args, loss="huber", f_scale=huber_px, max_nfev=300)
        if best is None or r.cost < best.cost:
            best = r
    e = np.linalg.norm(_reproj(best.x, R_sub, t_sub, K, k1, fidx, X, uv), axis=1)
    out = {"Rs": Rotation.from_rotvec(best.x[:3]).as_matrix(), "s": float(np.exp(best.x[3])), "T": best.x[4:7],
           "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)), "n_obs": int(len(e)),
           "inlier_frac_10px": float((e < 10).mean()), "frames_used": int(len(np.unique(fidx))),
           "floor_constraint": floor is not None, "starts": len(starts)}
    if floor is not None:
        rr = _residuals(best.x, *args)[-(3 + len(C_s)):]
        out["floor_normal_err_deg"] = float(np.degrees(np.linalg.norm(rr[:3]) / lam_normal))
        out["camera_height_err_median"] = float(np.median(np.abs(rr[3:])) / lam_height)
    return out


def _rot_reproj(x, R_sub, K, k1, fidx, X, uv):
    Rs = Rotation.from_rotvec(x[:3]).as_matrix()
    R_m = R_sub @ Rs.T
    Xc = np.einsum("nij,nj->ni", R_m[fidx], X - x[3:6])
    z = Xc[:, 2]
    vis = z > 0.05
    zs = np.where(vis, z, 1.0)
    xn, yn = Xc[:, 0] / zs, Xc[:, 1] / zs
    d = 1 + k1[fidx] * (xn * xn + yn * yn)
    Kf = K[fidx]
    r = np.stack([Kf[:, 0, 0] * xn * d + Kf[:, 0, 2], Kf[:, 1, 1] * yn * d + Kf[:, 1, 2]], 1) - uv
    r[~vis] = 200.0
    return r


def solve_rotonly(R_sub, K, k1, fidx, X, uv, w, up_main: np.ndarray | None = None, huber_px: float = 6.0,
                  n_yaw: int = 8) -> dict:
    """Rotation-only attach: (Rs, C) from the dancers; multi-start over yaw about the main up axis."""
    t0 = np.zeros((len(R_sub), 3))
    try:
        Rs0, C_m0, _ = init_rotation(R_sub, t0, K, fidx, X, uv)
        C0 = np.median(C_m0, 0)
    except ValueError:
        Rs0, C0 = np.eye(3), X.mean(0) + np.array([0, 0, -5.0])
    axis = up_main if up_main is not None else np.array([0, 1.0, 0])
    best = None
    fun = lambda x: (_rot_reproj(x, R_sub, K, k1, fidx, X, uv) * w[:, None]).ravel()  # noqa: E731
    for yaw in np.linspace(0, 2 * np.pi, n_yaw, endpoint=False):
        Ry = Rotation.from_rotvec(axis * yaw)
        # rotate the camera about the dancers' centre: orientation and centre move together
        cen = X.mean(0)
        Rs_i = Ry.as_matrix() @ Rs0
        C_i = cen + Ry.apply(C0 - cen)
        x0 = np.r_[Rotation.from_matrix(Rs_i).as_rotvec(), C_i]
        r = least_squares(fun, x0, loss="huber", f_scale=huber_px, max_nfev=300)
        if best is None or r.cost < best.cost:
            best = r
    e = np.linalg.norm(_rot_reproj(best.x, R_sub, K, k1, fidx, X, uv), axis=1)
    return {"Rs": Rotation.from_rotvec(best.x[:3]).as_matrix(), "C": best.x[3:6], "median_px": float(np.median(e)),
            "p90_px": float(np.percentile(e, 90)), "n_obs": int(len(e)), "inlier_frac_10px": float((e < 10).mean()),
            "frames_used": int(len(np.unique(fidx))), "starts": n_yaw}
