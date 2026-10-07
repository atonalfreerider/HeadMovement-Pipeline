"""Zoom-depth guard for per-frame camera poses, applied by stage_smooth before its light smoothing.

A zoomed phone (narrow field of view) that sees few static points cannot tell a move along its viewing axis from a
focal change: P4Pf solutions then scatter along the axis while the focal compensates, the static points still fit,
and the dancers (much closer than the room) project at the wrong size. larissa_kadu_38 cam09 50-56 s: 32-210 inliers
at f/maxdim 1.5-2.4, the exported centre jittered along the axis by ~0.1 m (median) with a 1.35 m spike at 50.78 s.

1. Per frame, the posterior SD of the camera centre ALONG ITS OPTICAL AXIS from the frame's inlier 2D-3D
   correspondences: Gauss-Newton covariance of (rotation, centre, log focal), k1 held, at the frame's robust pixel
   residual (>= 1 px). On larissa_kadu_38 it predicts the observed jitter (deviation from a 1 s running median):
   cam09 50-52 s 79 mm predicted / 102 mm observed (medians), 52-56 s 42 / 56 mm, elsewhere 7 / 4 mm; the wide
   static cameras 1-4 mm.
2. Frames whose axis SD exceeds max_axis_sd_rel x the camera's median static-point depth (0.003: ~25 mm at 8 m) or
   with fewer than min_inliers inliers are AMBIGUOUS. Their centre is replaced by an inverse-variance weighted robust
   local-linear smooth (sigma_s) of ALL ok frames' centres (weight min(1, (threshold / axis SD)^2)), and rotation +
   focal are re-solved against the same correspondences with that centre fixed (Huber 2 px, k1 held). A re-solved
   frame stays ok only if its median static reprojection is <= max_reproj_px.
   Why not hold the centre at the well-determined frames' low-pass: on cam09 that made the dancers WORSE (leave-one-
   out residual, sync-corrected dancers from the five other static cameras, 52-56 s: 23.8 -> 44 px median). The
   moderately determined frames there agree on a ~0.4 m forward move of the hand-held phone that the dancers
   confirm; the inverse-variance smooth keeps it and removes the noise and the spike (ambiguous frames: 25.9 -> 23.4 px
   median, 32.8 -> 27.5 px mean; static reprojection 1.96 px median). Most of cam09's remaining 23 px there (vs 13 px
   elsewhere) is the zoom: the same 3D error is ~1.8x more pixels at f/maxdim ~2.
3. Attached cameras (source 'attach', centre fixed by the attach solve) are skipped.

    python -m dancecap.cameras.depth_guard <take> [--cams 09]     # report only, no files written
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DepthGuardSettings:
    enabled: bool = True
    max_axis_sd_rel: float = 0.003   # x the camera's median static-point depth (SfM units cancel)
    min_inliers: int = 60            # fewer inliers -> ambiguous whatever the covariance says
    sigma_s: float = 0.2             # inverse-variance robust smooth of the centres (s)
    max_reproj_px: float = 4.0       # a re-solved pose must still fit its static points (stage_smooth's limit)
    max_pts: int = 1500              # correspondences per frame for the covariance / re-solve (subsampled)
    min_frames: int = 10             # fewer ok frames -> nothing done (report only)


def _rot(v: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(v).as_matrix()


def project(X: np.ndarray, R: np.ndarray, C: np.ndarray, f: float, cx: float, cy: float, k1: float) -> np.ndarray:
    """OpenCV pixel-centre projection with one radial term (cameras.report.project_cv), centre parametrisation."""
    Xc = (X - C) @ R.T
    z = np.where(np.abs(Xc[:, 2]) > 1e-9, Xc[:, 2], 1e-9)
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    d = 1 + k1 * (x * x + y * y)
    return np.stack([f * x * d + cx, f * y * d + cy], -1)


def _subsample(n: int, k: int, seed: int = 0) -> np.ndarray:
    return np.arange(n) if n <= k else np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def axis_sd(X: np.ndarray, x: np.ndarray, R: np.ndarray, C: np.ndarray, f: float, k1: float, cx: float,
            cy: float) -> tuple[float, float]:
    """(SD of the centre along the optical axis, robust pixel sigma used) from one frame's correspondences."""
    p0 = project(X, R, C, f, cx, cy, k1)
    sig = max(1.0, 1.4826 * float(np.median(np.linalg.norm(p0 - x, axis=1))))
    J = np.empty((2 * len(X), 7))
    for k in range(7):
        if k < 3:
            e = np.zeros(3)
            e[k] = 1e-6
            p1, h = project(X, _rot(e) @ R, C, f, cx, cy, k1), 1e-6
        elif k < 6:
            e = np.zeros(3)
            e[k - 3] = 1e-4
            p1, h = project(X, R, C + e, f, cx, cy, k1), 1e-4
        else:
            p1, h = project(X, R, C, f * np.exp(1e-5), cx, cy, k1), 1e-5
        J[:, k] = ((p1 - p0) / h).ravel()
    cov = np.linalg.pinv(J.T @ J) * sig ** 2
    a = R[2]  # optical axis in the map frame (row 2 of world->camera R)
    return float(np.sqrt(max(float(a @ cov[3:6, 3:6] @ a), 0.0))), sig


def resolve_fixed_centre(X: np.ndarray, x: np.ndarray, R0: np.ndarray, C: np.ndarray, f0: float, k1: float,
                         cx: float, cy: float) -> tuple[np.ndarray, float]:
    """Rotation + focal with the centre fixed (robust Huber, 2 px), starting from (R0, f0)."""
    from scipy.optimize import least_squares

    def res(p):
        return (project(X, _rot(p[:3]) @ R0, C, f0 * np.exp(p[3]), cx, cy, k1) - x).ravel()
    sol = least_squares(res, np.zeros(4), loss="huber", f_scale=2.0, x_scale=np.array([1e-3, 1e-3, 1e-3, 1e-2]),
                        max_nfev=200)
    return _rot(sol.x[:3]) @ R0, float(f0 * np.exp(sol.x[3]))


def apply(d: dict, width: int, height: int, s: DepthGuardSettings | None = None) -> tuple[dict, dict]:
    """Raw localisation (cameras/cache/loc_<cam>.npz contents) -> (copy with the ambiguous frames' R / tvec / f /
    reproj_px replaced and unrecoverable ones ok=False, plus 'recentred' (F,) bool and 'axis_sd' (F,) SfM units;
    report dict)."""
    from .smooth import robust_smooth
    s = s or DepthGuardSettings()
    F = len(d["t"])
    t = np.asarray(d["t"], float)
    ok = np.asarray(d["ok"], bool)
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    out["recentred"] = np.zeros(F, bool)
    out["axis_sd"] = np.full(F, np.nan)
    rep = {"settings": asdict(s), "frames": F}
    src = np.asarray(d["source"]) if "source" in d else np.array([""] * F)
    if not s.enabled or (src == "attach").any() or "corr_off" not in d or ok.sum() < s.min_frames:
        rep["skipped"] = ("disabled" if not s.enabled else "attached camera" if (src == "attach").any()
                          else "no correspondences" if "corr_off" not in d else f"only {int(ok.sum())} ok frames")
        return out, rep
    cx, cy = width / 2 - 0.5, height / 2 - 0.5
    off = d["corr_off"]
    C_raw = -np.einsum("fji,fj->fi", d["R"], d["tvec"])
    depth = np.full(F, np.nan)
    sd = np.full(F, np.nan)
    for i in np.flatnonzero(ok):
        a, b = int(off[i]), int(off[i + 1])
        if b - a < 6:
            continue
        sel = _subsample(b - a, s.max_pts, i)
        X, x = d["corr_xyz"][a:b][sel].astype(float), d["corr_xy"][a:b][sel].astype(float)
        depth[i] = float(np.median((X - C_raw[i]) @ d["R"][i][2]))
        sd[i], _ = axis_sd(X, x, d["R"][i], C_raw[i], float(d["f"][i]), float(d["k1"][i]), cx, cy)
    out["axis_sd"] = sd
    D = float(np.nanmedian(depth)) if np.isfinite(depth).any() else float("nan")
    thr = s.max_axis_sd_rel * D
    amb = ok & (~np.isfinite(sd) | (sd > thr) | (np.asarray(d["n_inliers"]) < s.min_inliers))
    rep.update({"median_point_depth": round(D, 3), "max_axis_sd": round(thr, 4),
                "axis_sd_median": round(float(np.nanmedian(sd)), 4) if np.isfinite(sd).any() else None,
                "axis_sd_p99": round(float(np.nanpercentile(sd, 99)), 4) if np.isfinite(sd).any() else None,
                "axis_sd_max": round(float(np.nanmax(sd)), 4) if np.isfinite(sd).any() else None,
                "ambiguous": int(amb.sum())})
    if not amb.any() or not np.isfinite(thr):
        rep.update(recentred=0, dropped=0)
        return out, rep
    w = np.where(ok, np.clip((thr / np.maximum(np.nan_to_num(sd, nan=10 * thr), 1e-9)) ** 2, 0.0, 1.0), 0.0)
    C_s, _ = robust_smooth(t, np.where(ok[:, None], C_raw, np.nan), w, s.sigma_s)
    shift, rp_before, rp_after, dropped = [], [], [], []
    for i in np.flatnonzero(amb):
        a, b = int(off[i]), int(off[i + 1])
        if b - a < 6:
            out["ok"][i] = False
            dropped.append(i)
            continue
        X, x = d["corr_xyz"][a:b].astype(float), d["corr_xy"][a:b].astype(float)
        sel = _subsample(len(X), s.max_pts, i)
        k1 = float(d["k1"][i])
        R1, f1 = resolve_fixed_centre(X[sel], x[sel], d["R"][i], C_s[i], float(d["f"][i]), k1, cx, cy)
        e0 = float(np.median(np.linalg.norm(project(X, d["R"][i], C_raw[i], float(d["f"][i]), cx, cy, k1) - x, axis=1)))
        e1 = float(np.median(np.linalg.norm(project(X, R1, C_s[i], f1, cx, cy, k1) - x, axis=1)))
        rp_before.append(e0)
        rp_after.append(e1)
        shift.append(float(np.linalg.norm(C_s[i] - C_raw[i])))
        out["R"][i], out["tvec"][i], out["f"][i] = R1, -R1 @ C_s[i], f1
        out["recentred"][i] = True
        if "reproj_px" in out:
            out["reproj_px"][i] = e1
        if e1 > s.max_reproj_px:
            out["ok"][i] = False
            dropped.append(i)
    rc = out["recentred"]
    rep.update({"recentred": int(rc.sum()), "dropped": len(dropped), "recentred_t": spans(t, rc),
                "dropped_t": spans(t, np.isin(np.arange(F), dropped)),
                "centre_shift_median": round(float(np.median(shift)), 4) if shift else None,
                "centre_shift_p90": round(float(np.percentile(shift, 90)), 4) if shift else None,
                "centre_shift_max": round(float(np.max(shift)), 4) if shift else None,
                "reproj_px_before_median": round(float(np.median(rp_before)), 3) if rp_before else None,
                "reproj_px_after_median": round(float(np.median(rp_after)), 3) if rp_after else None,
                "reproj_px_after_p90": round(float(np.percentile(rp_after, 90)), 3) if rp_after else None})
    return out, rep


def spans(t: np.ndarray, m: np.ndarray, join_s: float = 0.2) -> list[list[float]]:
    """Time spans [t0, t1] of a frame mask (gaps up to join_s merged)."""
    idx = np.flatnonzero(m)
    if not len(idx):
        return []
    sp = [[float(t[idx[0]]), float(t[idx[0]])]]
    for i in idx[1:]:
        if t[i] - sp[-1][1] <= join_s:
            sp[-1][1] = float(t[i])
        else:
            sp.append([float(t[i]), float(t[i])])
    return [[round(a, 3), round(b, 3)] for a, b in sp]


def main(argv=None):
    from ..take import Take
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.depth_guard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take")
    ap.add_argument("--cams")
    a = ap.parse_args(argv)
    work = Take.load(a.take).work.resolve()
    cams = a.cams.split(",") if a.cams else sorted(p.stem[4:] for p in (work / "cameras" / "cache").glob("loc_*.npz"))
    res = {}
    for cam in cams:
        d = dict(np.load(work / "cameras" / "cache" / f"loc_{cam}.npz"))
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        _, rep = apply(d, int(meta["width"]), int(meta["height"]))
        res[cam] = rep
        print(f"[depth_guard] {cam}: " + json.dumps({k: v for k, v in rep.items() if k != "settings"}))
    return res


if __name__ == "__main__":
    main()
