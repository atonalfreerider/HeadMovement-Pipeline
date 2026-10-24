"""Per-frame focal of an ATTACHED camera that zooms (cameras.pipeline.stage_attach, rotation-only attach).

An attached camera (cam05, cam06 of larissa_kadu) is localised against its own single-camera sub-model, which has ONE
focal (pipeline.stage_attach: fix_focal). The rotation-only attach then keeps only the per-frame ROTATIONS and holds
the camera centre constant. A phone that ZOOMS cannot show it that way: with the focal fixed the per-frame solver
slides the camera along its optical axis instead (dolly-zoom; larissa_kadu_full cam05 at 9.6-10.8 s: the sub-model
centre moves by 140 units against a point depth of 345 while the picture zooms x1.41) and the exported focal stays
constant, so the dancers project at the wrong size for the whole zoom (and for every segment, because the attach
centre is fitted to whatever zoom state dominates the window).

Here the zoom is recovered from the same 2D-3D correspondences, without the images: with the centre held at the
sub-model's reference centre C_ref (the mean centre of the sub-model's own images; they were taken at one zoom state,
the sub-model has one focal), each frame's rotation AND focal are re-solved (Huber 2 px, k1 held; the start is the
focal the slide implies, f0 x D / (D - slide)). The sub-model has no parallax (depth ~ 345 units, centre scatter ~
3 units), so C_ref pins the focal to ~1 %. The result is a per-frame (R, f) with the constant centre: exactly what
solve_rotonly (attach.py) expects, now with a per-frame K.

Only cameras the user declared as ZOOMING are re-solved (takes/<take>.toml [cameras."NN"] zoom = true): an attached
camera declared no-zoom keeps one focal per sub-model.

    python -m dancecap.cameras.attach_zoom <take> --cams 05        # re-attach with the per-frame focal, rewrite the export
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .depth_guard import project, resolve_fixed_centre


@dataclass
class ZoomSettings:
    min_corr: int = 30             # correspondences a frame needs
    max_pts: int = 1500            # subsample per frame (as depth_guard)
    f_range: tuple = (0.25, 4.0)   # accepted f / f0
    max_reproj_px: float = 6.0     # a re-solved frame must still fit its static points (stage_attach's ransac_px)
    min_free: int = 120            # inliers a frame needs to determine its own focal (fewer: the focal is interpolated)
    min_spread: float = 0.0        # ... optionally their spread (std / image size): real frames have a median of 0.11-0.16
    #                                (1 % 0.03-0.05), so it is off by default; the inlier COUNT carries the larissa_kadu_full case
    hampel_win: int = 10           # focal spikes: median over +-win well-determined frames
    hampel_k: float = 3.0
    hampel_floor: float = 0.02     # ... a frame whose log focal is off by max(k x MAD, floor) is a spike


def reference_centre(sub) -> np.ndarray:
    """Mean camera centre of the sub-model's registered images (sub-model frame)."""
    C = []
    for im in sub.images.values():
        if im.has_pose:
            T = im.cam_from_world()
            C.append(-T.rotation.matrix().T @ T.translation)
    if not C:
        raise ValueError("sub-model has no registered image")
    return np.mean(C, 0)


def _rot(v: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(v).as_matrix()


def resolve_rotation(X, x, R0, C, f, k1, cx, cy) -> np.ndarray:
    """Rotation only (centre and focal fixed), robust (Huber 2 px), from R0."""
    from scipy.optimize import least_squares
    sol = least_squares(lambda v: (project(X, _rot(v) @ R0, C, f, cx, cy, k1) - x).ravel(), np.zeros(3), loss="huber",
                        f_scale=2.0, x_scale=1e-3, max_nfev=100)
    return _rot(sol.x) @ R0


def hampel_spikes(t: np.ndarray, lf: np.ndarray, win: int, k: float, floor: float) -> np.ndarray:
    """Frames of a time-sorted log-focal track that are spikes against the median of their +-win neighbours. A monotone
    ramp (a zoom) is its own median: only isolated excursions are flagged."""
    n = len(lf)
    bad = np.zeros(n, bool)
    for i in range(n):
        w = lf[max(0, i - win):i + win + 1]
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        bad[i] = abs(lf[i] - med) > max(k * mad, floor)
    return bad


def resolve_zoom(poses: list, C_ref: np.ndarray, w: int, h: int, s: ZoomSettings | None = None,
                 t: np.ndarray | None = None) -> tuple[list, dict]:
    """Copy of the sub-model localisation `poses` (localize.FramePose, fixed focal f0, slid per-frame centres) with the
    centre of every ok frame held at C_ref and rotation + focal re-solved.

    A focal needs correspondences that see it: a frame with fewer than min_free inliers, or inliers clustered in a
    corner (a dancer fills the view: larissa_kadu_full cam05 130.9-131.5 s, 17-68 inliers -> f x0.81 for 0.6 s), and a
    frame whose focal is a spike against its neighbours get the focal INTERPOLATED (in time, from the well-determined
    frames) and only their rotation re-solved. t: frame times (default 30 fps). Returns (poses, report)."""
    import copy
    s = s or ZoomSettings()
    out = [copy.copy(p) for p in poses]
    n = len(out)
    t = np.arange(n) / 30.0 if t is None else np.asarray(t, float)
    cx, cy = w / 2 - 0.5, h / 2 - 0.5
    f0s = [p.f for p in poses if p.ok]
    f0 = float(np.median(f0s)) if f0s else 1.0
    n_ok = n_drop = 0
    before, after, ratios = [], [], []
    data: dict[int, tuple] = {}
    strong = np.zeros(n, bool)
    f_free = np.full(n, np.nan)
    for i, p in enumerate(out):
        if not p.ok:
            continue
        n_ok += 1
        if p.corr is None or len(p.corr.xyz) < s.min_corr:
            p.ok = False
            n_drop += 1
            continue
        X, x = np.asarray(p.corr.xyz, float), np.asarray(p.q_xy, float)
        if len(X) > s.max_pts:
            sel = np.sort(np.random.default_rng(0).choice(len(X), s.max_pts, replace=False))
            X, x = X[sel], x[sel]
        Ci = -p.R.T @ p.t
        depth = float(np.median((X - Ci) @ p.R[2]))
        slide = float((Ci - C_ref) @ p.R[2])
        # fixed-focal PnP slides toward the scene when the picture zooms in: the implied focal ratio D / (D - slide)
        f_init = f0 * depth / (depth - slide) if depth > 0 and (depth - slide) > 0.3 * depth else float(p.f)
        f_init = float(np.clip(f_init, s.f_range[0] * f0, s.f_range[1] * f0))
        e0 = float(np.median(np.linalg.norm(project(X, p.R, C_ref, float(p.f), cx, cy, float(p.k1)) - x, axis=1)))
        R1, f1 = resolve_fixed_centre(X, x, p.R, C_ref, f_init, float(p.k1), cx, cy)
        spread = float(min(np.std(x[:, 0]) / w, np.std(x[:, 1]) / h))
        strong[i] = len(p.corr.xyz) >= s.min_free and spread >= s.min_spread and s.f_range[0] <= f1 / f0 <= s.f_range[1]
        f_free[i] = f1
        data[i] = (X, x, R1, e0)
    if strong.sum() < 10:                      # a short or sparse camera: every frame decides its own focal
        strong = np.isfinite(f_free)
    idx = np.flatnonzero(strong)
    spikes = np.zeros(n, bool)
    if len(idx) >= 2 * s.hampel_win + 1:
        spikes[idx] = hampel_spikes(t[idx], np.log(f_free[idx]), s.hampel_win, s.hampel_k, s.hampel_floor)
    good = strong & ~spikes
    if not good.any():
        good = strong
    gi = np.flatnonzero(good)
    f_ref = np.exp(np.interp(t, t[gi], np.log(f_free[gi])))
    n_interp = n_fix = 0
    for i, (X, x, R1, e0) in data.items():
        p = out[i]
        k1 = float(p.k1)
        if good[i]:
            f1 = float(f_free[i])
        else:
            f1 = float(f_ref[i])
            R1 = resolve_rotation(X, x, R1, C_ref, f1, k1, cx, cy)
            n_interp += 1
        e1 = float(np.median(np.linalg.norm(project(X, R1, C_ref, f1, cx, cy, k1) - x, axis=1)))
        if not (s.f_range[0] <= f1 / f0 <= s.f_range[1]) or e1 > s.max_reproj_px:
            p.ok = False
            n_drop += 1
            continue
        p.R, p.t, p.f, p.reproj_px = R1, -R1 @ C_ref, f1, e1
        n_fix += 1
        before.append(e0), after.append(e1), ratios.append(f1 / f0)
    rep = {"frames_ok_in": n_ok, "frames_resolved": n_fix, "frames_dropped": n_drop, "frames_focal_interpolated": n_interp,
           "f0": round(f0, 3),
           "f_ratio_min": round(float(np.min(ratios)), 4) if ratios else None,
           "f_ratio_max": round(float(np.max(ratios)), 4) if ratios else None,
           "reproj_px_fixed_focal_at_C_ref_median": round(float(np.median(before)), 3) if before else None,
           "reproj_px_resolved_median": round(float(np.median(after)), 3) if after else None,
           "C_ref": [round(float(v), 4) for v in C_ref]}
    return out, rep


# ------------------------------------------------------------------------------------------------ offline driver

def zoom_cams_of(work: Path, policy: str | None = None) -> set[str]:
    """Cameras the user declared as zooming (source user/auto, zoom true): their attach gets a per-frame focal."""
    from .. import camera_types as ct
    try:
        types = ct.types_for_work(work, policy=policy)
    except ct.CameraTypeError:
        return set()
    return {c for c, t in types.items() if t.zoom and t.source in ("user", "auto")}


def align_sub_mtime(work: Path, cam: str, log=print) -> bool:
    """The sub-model copies of larissa_kadu_full_{b,c,d} were written a few seconds after the stored cache key's
    mtime (the pod's copy of the SAME file: same size, same model - the front session localised against it). The
    pickle cache (locsub_<cam>.pkl) would be recomputed from images that are not on this machine: re-stamp the file with
    the key's time when the size agrees and the cache's frame list is this take's. No content changes."""
    import os
    import pickle
    p = work / "cameras" / "sfm_work" / f"sub_{cam}" / "best" / "images.bin"
    c = work / "cameras" / "cache" / f"locsub_{cam}.pkl"
    if not p.exists() or not c.exists():
        return False
    key = pickle.loads(c.read_bytes()).get("key") or {}
    sub = (key.get("sub") or [[None, None, None]])[0]
    if sub[2] != p.stat().st_size or not sub[1]:
        return False
    stored = int(sub[1]) // 10 ** 9
    if int(p.stat().st_mtime) != stored:
        os.utime(p, (stored, stored))
        log(f"[attach_zoom] {work.name} sub_{cam}: images.bin mtime re-stamped to the cache key's ({stored})")
    return True


def reattach(work: Path, cams: list[str], backup: str = "s2_gate_backup", log=print) -> dict:
    """Re-run the attach of `cams` (cached sub-model localisation, no images needed) with the per-frame focal and
    re-export them. Files replaced (copied to cameras/<backup>/ first): cameras/<cam>.npz, cache/loc_<cam>.npz,
    cache/attach_<cam>.npz. Other cameras are untouched."""
    from ..take import Take
    from . import pipeline as P
    from .features import Features, FeatureSettings, list_images
    from .mapping import map_settings
    P.LOOSE_FILE_KEYS = True      # the cached sub-model localisations are keyed by mtimes that a pod download changed
    work = Path(work).resolve()
    take = Take.load(work.name)
    all_cams = [s.id for s in take.sources if (work / "cameras" / f"{s.id}.npz").exists()]
    main_cams = [c for c in all_cams if c in P.model_cams(P.load_model(work))]
    bdir = work / "cameras" / backup
    (bdir / "cache").mkdir(parents=True, exist_ok=True)
    for c in cams:
        for rel in (f"{c}.npz", f"cache/loc_{c}.npz", f"cache/attach_{c}.npz"):
            src = work / "cameras" / rel
            if src.exists() and not (bdir / rel).exists():
                shutil.copy2(src, bdir / rel)
    for c in cams:
        align_sub_mtime(work, c, log)
    recs = list_images(work, main_cams + list(cams))
    fs, ms, ls = FeatureSettings(), map_settings(take.cfg), P.loc_settings(take)
    F = Features(work, work / "cameras" / "cache", fs)
    from .world import scale_rule_from_take, statures_from_take
    from .. import camera_types as ct
    types = {c: t for c, t in ct.types_for_work(work).items() if not t.general}
    t0 = time.time()
    info = P.stage_attach(work, recs, F, ls, ms, main_cams + list(cams), 0.04, 4.0, statures_from_take(take.cfg),
                          scale_rule_from_take(take.cfg), types=types, zoom_cams=set(cams))
    log(f"[attach_zoom] attach {cams}: {time.time() - t0:.0f}s")
    P.stage_smooth(work, list(cams), 0.04, 4.0, types=types)  # writes cameras/<cam>.npz
    rp = work / "cameras" / "report.json"
    if rp.exists():
        rep = json.loads(rp.read_text())
        rep.setdefault("attach", {}).update({c: info[c] for c in cams if c in info})
        rep["attach_zoom"] = {"cams": list(cams), "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "note": "per-frame focal for attached zooming cameras (cameras/attach_zoom.py); "
                                      f"previous exports in cameras/{backup}/"}
        rp.write_text(json.dumps(rep, indent=1, default=P._np))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.attach_zoom", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take")
    ap.add_argument("--cams", default="05")
    a = ap.parse_args(argv)
    from ..take import Take
    reattach(Take.load(a.take).work, a.cams.split(","))


if __name__ == "__main__":
    main()
