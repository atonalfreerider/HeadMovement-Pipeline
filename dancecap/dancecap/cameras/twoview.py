"""Two-view gates, PTZ attach and video sync for a take filmed by TWO cameras: one tripod + one moving camera (lucas_luiza).

dancecap.sync needs >= 2 OTHER cameras per camera, report.json's G2a gates were built for >= 5 and the tripod's PTZ track
(rotation + zoom about one centre, solved on another take's cut) lives in its own frame: this module fills those gaps.
CPU only (numpy + pycolmap), < 1 GB: frames are never decoded; Atlas depth maps and people masks are read one keyframe
at a time; large npz members (the loc cache's correspondences) are never loaded.

    python -m dancecap.cameras.twoview <cmd> <take> [...]

  map-gate  cameras/sfm_work/map.json: registered / total map images per camera (cameras/sfm), main_cams,
            strong_link_groups, the verified link of every --need pair (verification.per_campair: pairs, max_inliers and
            link_pairs = pairs with >= link_min_inliers inliers; a map.json without link_pairs -> strong_link_groups as a
            yes/no), the refined mean reprojection, each camera's map focal and the doppelganger check of cameras/sfm
            (cameras.plausible: registered images outside their camera's band of centres AND outside the room; G1
            wants 0) ->
            cameras/qa/map_gate.json. Exit 3 only when a --need camera has no registered image in the main model (loc
            cannot serve it).
  ptz       --mode check: the tripod's loc track vs the PTZ track of --ptz-take (camera_work/final/track_s*.npz, R_i =
            reference -> camera i, ptz.py). The cut (ffmpeg -ss S -vf fps=30 of the 59.94 fps original) holds in frame n
            the source frame FFmpeg's fps filter picks (round=near): the last whose rounded output pts is <= n, i.e. the
            last before S + (n + 0.5) / cut_fps, so sample n sits at t = (ceil((S + (n + 0.5) / cut_fps) x src_fps) - 1)
            / src_fps on this take's clock: 0-16.7 ms AFTER S + n / 30, and not the nearest source frame (lucas_luiza_h:
            the cut's YOLO joints match source frame floor(x) + 1 in every phase of x = (S + n / 30) x 59.94; cut frame 0
            is source frame 840, the nearest rule's 839 lies before the seek point). --cut-rule nearest | nominal for
            comparison. Q = robust chordal mean of R_ptz(t_i)^T R_loc_i (IRLS, Huber 0.2 deg), residual angle per frame
            and its time-shift scan per half of the cut phase (frac(x) < 0.5: where the two rules pick different frames;
            a wrong rule shows as a one-source-frame shift in that half only, which neither Q nor the Q-free check can
            see), relative rotations over 1/6 s (Q-free) and the residual time shift that best aligns them, f_loc /
            f_ptz, the robust SD of the RAW loc centres (cameras/cache/loc_<cam>.npz) in metres, focal levels around the
            zoom. --mode attach: the camera's map images (cameras/sfm) give Q (chordal mean), the centre C (median), the
            focal scale s_f = median(f_j / f_ptz(t_j)) and one k1 (median of the map's) -> cameras/ptz/<cam>.npz in the
            cameras/<cam>.npz schema (R = R_ptz Q, tvec = -R C, ok = rms_px <= 3 & n_inl >= 50, camera_class tripod_zoom,
            type_model fixed_centre); outside cameras/*.npz, which mamma_dyn.load_tracks and others read as cameras.
            --adopt installs it as cameras/<cam>.npz and moves the loc track to cameras/orig/<cam>_loc.npz: that file is
            the adoption marker cameras.pipeline.adopted_track honours, so no later cameras stage (stage_smooth, loc-join)
            overwrites or deletes the adopted track; cameras/adopted.json records it. Both modes update their own section
            of cameras/qa/ptz_check.json.
  atlas     the Atlas cameras of --atlas (mono4d.atlas.from_atlas_camera: RUB -> OpenCV) vs the SfM images
            keyframes/<cam>/<file> (keyframes.json order): RANSAC Umeyama Sim3 on the centres, centre / rotation residuals,
            fx_atlas / f_sfm, per keyframe atlas.world_from_atlas on the SfM points (people masked) and the Atlas centres'
            heights in world metres -> cameras/qa/atlas_check.json, review/atlas_check.png. Atlas depth is NOT metric
            (lucas_luiza: its focal is 2.2-2.8 x what the dancers' sizes imply): the per-keyframe depth scale is a drift
            check along the orbit, never a scale or focal prior; the metric scale comes from the stature chain.
  sync      the other camera's video offset against the master (gauge: master = 0) from the dancers' COCO joints: for each
            master frame tau, the other camera at label tau - d (camera and joints interpolated). Cost = speed-weighted
            trimmed mean (20 %) of the symmetric epipolar distance (px) of the joints with conf >= 0.5 in both views and
            image speed >= 50 px/s in the other camera. dancecap.sync's steps (20 ms over +-range, 4 ms and 1 ms around the
            minimum), then the vertex of a least-squares parabola through the SQUARED cost (a hyperbola in d: a parabola
            through the cost itself is pulled to the window centre), refitted on a window centred on it; repeated per
            --sub-s window: delta(t) = a + b (t - t_ref). The rate is applied (use_rate) when the drift moves delta by
            >= 2 ms and |b| >= 3 SE; delta is then the line's a at t_ref (the whole-window minimum sits at the joints'
            weighted time, not at t_ref: delta_global_ms) and stable = the sub-window estimates within 10 ms of the line
            (a step still fails), else within 10 ms of each other (dancecap.sync's rule); spread_ms stays the raw spread.
            Epipolar geometry is similarity invariant, so the SfM-frame tracks suffice (no world frame needed). Writes
            sync/offsets.json in dancecap.sync's layout (method "two_view_epipolar"; the clocks block built exactly as
            recap_front.fixed_sync_doc does, so sync.offsets_check and mamma prepare accept it), sync/curves_2v.json,
            review/sync_2v.png. It NEVER writes "use": false (mamma_dyn would silently apply delta 0): an unstable estimate
            or one at the search edge exits 2 and leaves sync/offsets.json as it was.
  check     sync-corrected two-view DLT of the 17 COCO joints of both dancers (COCO-17: the ankles are the lowest joints,
            there are no toes) in the world frame: reprojection per camera, two-view coverage (>= 8 joints) over EVERY master
            frame of the window (a frame without a pose or a dancer counts as not covered) and per 10 s, bone lengths
            (CV = SD / mean; the robust 1.4826 MAD / median beside it), implied statures (world.py's chain rule), stance
            ankle heights, swap share, camera-to-couple distance -> cameras/qa/twoview_check[_<tag>].json,
            review/twoview_check[_<tag>].png. --track CAM=path (the attached PTZ track: cameras/ptz/<cam>.npz; its
            earlier name cameras/<cam>_ptz.npz reads that file). The implied statures sit near the TOML values BY
            CONSTRUCTION (the world scale came from the same chain and statures): only the lead / follow disagreement is
            informative. A consistency check, not a metric verification. Records which sync/offsets.json it used.
  gates     G1-G8 of the two-view runbook -> cameras/qa/twoview_gates.json {gate: {value, threshold, pass, status, checks}}
            and a table. Exit 0 also when gates fail; 1 only when an input is missing (the file is written first). G8
            fails on a sync/offsets.json that is not the last sync run's (a rejected re-run keeps the old file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import tomllib
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .. import mamma_dyn as md
from .. import sync as S
from ..take import PACKAGE_ROOT
from . import world as W

ROLES = ("lead", "follow")
MIN_CONF = 0.5  # a joint enters only with this confidence in BOTH views
SPEED_MIN_PX_S = 50.0  # sync: slower joints (other camera) carry no timing information (0.5 px per 10 ms)
SPEED_CAP_PX_S = 500.0  # ... and the speed weight saturates here: a fast mis-detection must not dominate
TRIM = 0.2  # sync cost: the largest 20 % (of the weight) of the epipolar distances are dropped
GAP_FRAMES = 1.5  # interpolation never bridges more than this many frame intervals
SUB_MIN_OBS = 300  # a sync sub-window needs this many joint observations at its minimum
DEPTH_MS = 40.0  # curve depth: the cost at delta +- 40 ms over the minimum (G8 >= 1.15)
DRIFT_USE_MS = 2.0  # use_rate when the fitted drift moves delta by >= 2 ms over the window and |b| >= 3 SE
PW_WIN_S, PW_STEP_S = 30.0, 10.0  # piecewise sync: a delta per 30 s window every 10 s ...
PW_HALF_MS = 15.0  # ... counted where its even / odd frame halves agree within this (= PW_REPRO_MAX_MS)
PW_REPRO_MEDIAN_MS, PW_REPRO_MAX_MS = 5.0, 15.0  # stable: the even-half and odd-half curves agree (median / max)
PW_MIN_SHARE = 0.8  # ... and this share of the windows counted
COVER_MIN_JOINTS = 8  # a dancer is two-view covered in a frame with >= 8 joints in both views
SWAP_MARGIN = 0.8  # a frame counts as swapped when the swapped pairing is cheaper by more than 20 %
BONE_MAX_PX = 15.0  # check: a joint reprojecting worse than this in either view enters no bone / stature / ankle
STANCE_V = 0.15  # check: an ankle moving slower than this (m/s, horizontal) is in stance
PTZ_OK_RMS, PTZ_OK_INL = 3.0, 50  # a PTZ frame is ok with rms_px <= 3 and n_inl >= 50
HUBER_DEG = 0.2  # chordal-mean IRLS
REL_DT_S = 1 / 6  # Q-free relative rotations over 1/6 s
SHIFT_SCAN_MS = 20.0  # ptz check: residual time shift scanned over +-20 ms in 1 ms steps
ZOOM_GAP = (115.0, 130.0)  # lucas_luiza: the PTZ shows a wide stretch here: focal statistics exclude it, levels flank it
SRC_FPS = 60000 / 1001  # the tripod's original (59.94 fps)
CUT_RULES = ("fps", "nearest", "nominal")  # ptz: which source frame a cut frame holds (ptz_sample_times)
ATTACH_MIN_IMAGES = 5
ATTACH_MIN_PAN_COVER = 0.5  # the map images must span half of the window's pan range
ROOM_EDGE_DEG, ROOM_WALL_DEG = 1.0, 2.0  # G6 room check: horizontal edges / walls vs world Y
KP_ANKLES = W.KP_ANKLES
BONES = {"thigh": ((11, 13), (12, 14)), "shank": ((13, 15), (14, 16)), "upper_arm": ((5, 7), (6, 8)),
         "forearm": ((7, 9), (8, 10)), "shoulder_width": ((5, 6),), "hip_width": ((11, 12),)}


# ------------------------------------------------------------------------------------------------ small helpers


def _need(p: Path) -> Path:
    """An input the command cannot run without: SystemExit (status 1) naming it."""
    if not Path(p).exists():
        raise SystemExit(f"[twoview] missing input {p}")
    return Path(p)


def _json(p: Path):
    return json.loads(Path(p).read_text())


def _f(x, nd: int = 4):
    """float rounded for JSON; None when missing or not finite."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_json(p: Path, doc) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, default=_jd))
    return p


def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def take_cfg(take: str) -> dict:
    """takes/<take>.toml without the alignment file (a CPU pod may run a command before every input is there)."""
    return tomllib.loads((PACKAGE_ROOT / "takes" / f"{take}.toml").read_text(encoding="utf-8"))


def _window(cfg: dict | None) -> tuple[float, float] | None:
    w = (cfg or {}).get("window")
    return (float(w[0]), float(w[1])) if w else None


# ------------------------------------------------------------------------------------------------ rotations


def rot_deg(R: np.ndarray) -> np.ndarray:
    """Rotation angle (deg) of rotation matrices (..., 3, 3)."""
    R = np.asarray(R, float)
    return np.degrees(Rot.from_matrix(R.reshape(-1, 3, 3)).magnitude()).reshape(R.shape[:-2])


def _so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def chordal_mean(M: np.ndarray, huber_deg: float = HUBER_DEG, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Robust chordal L2 mean of rotations M (N,3,3): IRLS with Huber weights (1 within huber_deg, huber / angle beyond).
    Returns (Q, residual angle (N,) deg = angle(Q^T M_i))."""
    M = np.asarray(M, float)
    w = np.ones(len(M))
    for _ in range(iters):
        Q = _so3(np.einsum("n,nij->ij", w, M))
        th = rot_deg(np.einsum("ji,njk->nik", Q, M))
        w_new = np.where(th <= huber_deg, 1.0, huber_deg / np.maximum(th, 1e-12))
        if np.allclose(w_new, w, atol=1e-9):
            break
        w = w_new
    return Q, th


def slerp_pairs(R0: np.ndarray, R1: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Geodesic interpolation R0 Exp(a Log(R0^T R1)) per row (M,3,3), a in [0, 1]."""
    rel = Rot.from_matrix(np.einsum("mji,mjk->mik", R0, R1)).as_rotvec()
    return np.einsum("mij,mjk->mik", R0, Rot.from_rotvec(rel * np.asarray(a, float)[:, None]).as_matrix())


def _bracket(ts: np.ndarray, t: np.ndarray):
    """Bracketing samples of query times t in sorted times ts: (i0, i1, a in [0,1], span, inside). a is snapped to 0 / 1
    within 1e-6 so that a query ON a sample never depends on its neighbour."""
    ts, t = np.asarray(ts, float), np.asarray(t, float)
    i1 = np.clip(np.searchsorted(ts, t, side="right"), 1, len(ts) - 1)
    i0 = i1 - 1
    span = ts[i1] - ts[i0]
    a = np.clip((t - ts[i0]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    a = np.where(a <= 1e-6, 0.0, np.where(a >= 1 - 1e-6, 1.0, a))
    inside = (t >= ts[0] - 1e-9) & (t <= ts[-1] + 1e-9)
    return i0, i1, a, span, inside


# ------------------------------------------------------------------------------------------------ tracks and joints


def load_track(work: Path, cam: str, path: Path | str | None = None, world: bool = True) -> dict:
    """cameras/<cam>.npz (or `path`, relative to work) -> track dict {t, K, R, tvec, C, f, pp, k1, ok, reproj_px, dt, frame,
    sim3}. world=True with cameras/world_from_sfm.json: world-frame poses in metres (mamma_dyn.world_cameras); else the
    SfM frame. Frames whose pose / intrinsics are not finite are filled (identity, median focal) and marked not ok."""
    p = Path(path) if path is not None else work / "cameras" / f"{cam}.npz"
    p = p if p.is_absolute() else work / p
    with np.load(_need(p)) as z:
        t, K, R, tv = (np.asarray(z[k], float) for k in ("t", "K", "R", "tvec"))
        dist = np.asarray(z["dist"], float) if "dist" in z.files else np.zeros(len(t))
        ok = np.asarray(z["ok"], bool)
        rp = np.asarray(z["reproj_px"], float) if "reproj_px" in z.files else np.full(len(t), np.nan)
    k1 = dist if dist.ndim == 1 else dist[:, 0]
    good = (np.isfinite(R).all((1, 2)) & np.isfinite(tv).all(1) & np.isfinite(K).all((1, 2)) & (K[:, 0, 0] > 0)
            & np.isfinite(k1))
    if not good.any():
        raise SystemExit(f"[twoview] {p}: no finite pose")
    R[~good], tv[~good], k1 = np.eye(3), 0.0, np.where(good, k1, 0.0)
    K[~good] = K[good][len(K[good]) // 2]
    sim3_p = work / "cameras" / "world_from_sfm.json"
    sim3 = _json(sim3_p) if world and sim3_p.exists() else None
    Rw, tw = md.world_cameras(R, tv, sim3)
    C = -np.einsum("fji,fj->fi", Rw, tw)
    return {"cam": cam, "path": str(p), "t": t, "K": K, "R": Rw, "tvec": tw, "C": C, "f": K[:, 0, 0].copy(),
            "pp": K[:, :2, 2].copy(), "k1": k1, "ok": ok & good, "reproj_px": rp,
            "dt": float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30, "frame": "world" if sim3 else "sfm",
            "sim3": sim3}


def interp_track(tr: dict, t: np.ndarray, max_gap: float = GAP_FRAMES) -> dict:
    """The camera at times t: R by slerp, centre linear, focal log-linear, k1 / principal point linear. ok = both
    bracketing frames ok (only the one hit when t is on a frame), at most max_gap frame intervals apart, t inside."""
    t = np.asarray(t, float)
    i0, i1, a, span, inside = _bracket(tr["t"], t)
    ok = tr["ok"]
    valid = inside & (span <= max_gap * tr["dt"]) & ((a >= 1) | ok[i0]) & ((a <= 0) | ok[i1])
    A = a[:, None]
    R = slerp_pairs(tr["R"][i0], tr["R"][i1], a)
    C = (1 - A) * tr["C"][i0] + A * tr["C"][i1]
    return {"R": R, "C": C, "tvec": -np.einsum("mij,mj->mi", R, C),
            "f": np.exp((1 - a) * np.log(tr["f"][i0]) + a * np.log(tr["f"][i1])),
            "pp": (1 - A) * tr["pp"][i0] + A * tr["pp"][i1], "k1": (1 - a) * tr["k1"][i0] + a * tr["k1"][i1],
            "ok": valid}


def role_kp(work: Path, cam: str) -> tuple[np.ndarray, np.ndarray]:
    """(t (F,), kp (F, 2, 17, 3)): COCO-17 x, y, conf of the lead (slot 0) and the follow (slot 1) in every frame of
    people/<cam>.npz, matched to dancers/roles.json's boxes exactly as world.dancer_keypoints does (IoU > 0.5); NaN where
    unknown. roles.json must label this camera (no 'two largest boxes' fallback: judges and crowd stand in a J&J heat)
    with one entry per people frame: dancer_keypoints reads roles.json's frames BY POSITION, so a shorter or shifted
    list would drop or mis-assign joints silently (an entry's t, where recorded, must be its people frame's +- 1 ms)."""
    rp = _need(work / "dancers" / "roles.json")
    rc = (_json(rp).get("cams") or {}).get(cam)
    if rc is None:
        raise SystemExit(f"[twoview] missing input: {rp} has no camera {cam}")
    with np.load(_need(work / "people" / f"{cam}.npz")) as z:
        t = np.asarray(z["t"], float)
    fr = rc.get("frames") or []
    if len(fr) != len(t):
        raise SystemExit(f"[twoview] {rp}: camera {cam} has {len(fr)} frames, people/{cam}.npz {len(t)} (read by position)")
    ft = np.array([float(f["t"]) if isinstance(f, dict) and f.get("t") is not None else np.nan for f in fr])
    bad = np.flatnonzero(np.isfinite(ft) & (np.abs(ft - t) > 1e-3))
    if len(bad):
        raise SystemExit(f"[twoview] {rp}: camera {cam}'s frame {bad[0]} is at t {ft[bad[0]]:.4f}, people/{cam}.npz "
                         f"{t[bad[0]]:.4f} ({len(bad)} frames off by > 1 ms)")
    d = W.dancer_keypoints(work, cam, len(t))
    kp = np.stack([d["lead"], d["follow"]], 1).astype(float)
    kp[kp[..., 2] <= 0] = np.nan
    return t, kp


def interp_kp(kt: np.ndarray, kp: np.ndarray, t: np.ndarray, max_gap: float = GAP_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Joints kp (F, ..., 3) at times t (M,): linear between the bracketing frames (conf = the smaller), NaN when either
    is missing (except for a query on a frame) or they are more than max_gap frame intervals apart (a gap is never
    bridged). -> (kp (M, ..., 3), image speed (M, ...) px/s between the bracketing frames, NaN where unknown)."""
    t = np.asarray(t, float)
    i0, i1, a, span, inside = _bracket(kt, t)
    dt = float(np.median(np.diff(kt))) if len(kt) > 1 else 1 / 30
    ext = (1,) * (kp.ndim - 2)
    A = a.reshape((-1,) + ext + (1,))
    k0, k1 = kp[i0], kp[i1]
    v0, v1 = np.isfinite(k0).all(-1), np.isfinite(k1).all(-1)
    near = (inside & (span <= max_gap * dt)).reshape((-1,) + ext)
    on0, on1 = (a <= 0).reshape((-1,) + ext), (a >= 1).reshape((-1,) + ext)
    both = near & v0 & v1
    with np.errstate(invalid="ignore"):
        out = (1 - A) * k0 + A * k1
        out[..., 2] = np.minimum(k0[..., 2], k1[..., 2])
        out = np.where((on0 & v0 & near)[..., None], k0, np.where((on1 & v1 & near)[..., None], k1, out))
        out[~(both | (on0 & v0 & near) | (on1 & v1 & near))] = np.nan
        spd = np.linalg.norm(k1[..., :2] - k0[..., :2], axis=-1) / np.where(span > 0, span, np.nan).reshape((-1,) + ext)
    return out, np.where(both, spd, np.nan)


def _normalise(xy: np.ndarray, f: np.ndarray, pp: np.ndarray, k1: np.ndarray, iters: int = 5) -> np.ndarray:
    """Pixels (M, ..., 2) of per-row cameras (f (M,), pp (M,2), k1 (M,)) -> undistorted normalised coordinates
    (SIMPLE_RADIAL inverse by fixed point, as world.undistort_pts)."""
    ext = (1,) * (xy.ndim - 2)
    f = np.asarray(f, float).reshape((-1,) + ext)
    k1 = np.asarray(k1, float).reshape((-1,) + ext)
    pp = np.asarray(pp, float).reshape((-1,) + ext + (2,))
    x = (xy - pp) / f[..., None]
    xu = x.copy()
    for _ in range(iters):
        xu = x / (1 + k1 * (xu ** 2).sum(-1))[..., None]
    return xu


def undistort(x: np.ndarray, K: np.ndarray, k1: np.ndarray) -> np.ndarray:
    """Pixels x (M, ..., 2) with per-row intrinsics K (M,3,3) and k1 (M,) -> normalised undistorted coordinates."""
    K = np.asarray(K, float)
    return _normalise(np.asarray(x, float), K[:, 0, 0], K[:, :2, 2], k1)


def _homog(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones(x.shape[:-1] + (1,))], -1)


def tri2(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of N points from two views: P (N,3,4) or (3,4) in normalised coordinates, x (N,2)
    normalised -> X (N,3)."""
    n = len(x1)
    P1, P2 = np.broadcast_to(P1, (n, 3, 4)), np.broadcast_to(P2, (n, 3, 4))
    A = np.stack([x1[:, :1] * P1[:, 2] - P1[:, 0], x1[:, 1:] * P1[:, 2] - P1[:, 1],
                  x2[:, :1] * P2[:, 2] - P2[:, 0], x2[:, 1:] * P2[:, 2] - P2[:, 1]], 1)
    X = np.linalg.svd(A)[2][:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return X[:, :3] / X[:, 3:4]


def reproj(X: np.ndarray, R: np.ndarray, tvec: np.ndarray, f: np.ndarray, pp: np.ndarray,
           k1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points X (N,3) with per-point cameras (R (N,3,3), tvec (N,3), f, pp (N,2), k1; SIMPLE_RADIAL as
    report.project_cv) -> (pixels (N,2), depth (N,))."""
    Xc = np.einsum("nij,nj->ni", R, X) + tvec
    z = Xc[:, 2]
    x = Xc[:, :2] / np.where(np.abs(z) > 1e-9, z, 1e-9)[:, None]
    d = 1 + k1 * (x ** 2).sum(1)
    return x * (d * f)[:, None] + pp, z


def _skew(v: np.ndarray) -> np.ndarray:
    o = np.zeros(v.shape[:-1] + (3, 3))
    o[..., 0, 1], o[..., 0, 2], o[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    o[..., 1, 0], o[..., 2, 0], o[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    return o


def sym_epi(xo: np.ndarray, xm: np.ndarray, R_om: np.ndarray, t_om: np.ndarray, fo: np.ndarray,
            fm: np.ndarray) -> np.ndarray:
    """Symmetric epipolar distance (px) of normalised homogeneous points xo (other view) and xm (master view), rows M
    first and broadcast over the other axes; X_other = R_om X_master + t_om per row; fo / fm the focal (px) per row:
    0.5 (distance of xo to the line E xm + distance of xm to the line E^T xo), E = [t_om]x R_om."""
    E = np.einsum("mij,mjk->mik", _skew(t_om), R_om)
    sh = (len(E),) + (1,) * (np.broadcast(xo, xm).ndim - 2)
    Eb = E.reshape(sh + (3, 3))
    lo = (Eb @ xm[..., None])[..., 0]
    lm = (np.swapaxes(Eb, -1, -2) @ xo[..., None])[..., 0]
    r = np.abs((xo * lo).sum(-1))
    do = r / np.maximum(np.hypot(lo[..., 0], lo[..., 1]), 1e-12) * np.asarray(fo, float).reshape(sh)
    dm = r / np.maximum(np.hypot(lm[..., 0], lm[..., 1]), 1e-12) * np.asarray(fm, float).reshape(sh)
    return 0.5 * (do + dm)


def trimmed_mean(x: np.ndarray, w: np.ndarray, trim: float = TRIM) -> float:
    """Weighted mean of x after dropping its largest values carrying `trim` of the total weight."""
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w)
    keep = cw <= (1 - trim) * cw[-1]
    keep[0] = True
    return float(np.sum(x[keep] * w[keep]) / np.sum(w[keep]))


def label_at(tau: np.ndarray, off: dict | None) -> np.ndarray:
    """The frame label L of a camera whose sync-corrected time (mamma_dyn.apply_offset: L + delta + rate (L - t_ref), or
    a piecewise delta) is tau: the inverse of apply_offset."""
    return md.offset_label(tau, off or {})


class TwoViewPair:
    """The master's frames (camera + normalised joints of both dancers) and the other camera (track + joints), the other
    evaluated at any label times: what sync and check share. Master rows = master frames that are ok, inside the
    window and have at least one joint."""

    def __init__(self, master: dict, other: dict, kp_m: tuple[np.ndarray, np.ndarray],
                 kp_o: tuple[np.ndarray, np.ndarray], window: tuple[float, float] | None = None):
        self.m, self.o = master, other
        self.kt_o, self.ko = kp_o
        t = master["t"]
        km, _ = interp_kp(kp_m[0], kp_m[1], t)  # the master's joints on its own frames
        sel = master["ok"] & np.isfinite(km[..., 0]).any((1, 2))
        if window is not None:
            sel &= (t >= window[0]) & (t <= window[1])
        self.rows = np.flatnonzero(sel)
        if len(self.rows) < 2:
            raise SystemExit(f"[twoview] camera {master['cam']}: fewer than 2 usable frames")
        r = self.rows
        self.t = t[r]
        self.R, self.tvec, self.C, self.f = master["R"][r], master["tvec"][r], master["C"][r], master["f"][r]
        self.pp, self.k1 = master["pp"][r], master["k1"][r]
        self.kp = km[r]
        self.conf = self.kp[..., 2]
        self.x = _homog(_normalise(self.kp[..., :2], self.f, self.pp, self.k1))

    def other_at(self, labels: np.ndarray) -> dict:
        co = interp_track(self.o, labels)
        ko, spd = interp_kp(self.kt_o, self.ko, labels)
        co.update(kp=ko, conf=ko[..., 2], spd=spd, x=_homog(_normalise(ko[..., :2], co["f"], co["pp"], co["k1"])))
        return co

    def cross(self, co: dict, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Epipolar distance (M, 2 other roles, 2 master roles, 17) of master rows idx against the other camera co (its
        rows aligned with idx), and the mask of joints with conf >= MIN_CONF in both views and both cameras ok."""
        R_om = np.einsum("mij,mkj->mik", co["R"], self.R[idx])
        t_om = co["tvec"] - np.einsum("mij,mj->mi", R_om, self.tvec[idx])
        xo, xm = co["x"][:, :, None], self.x[idx][:, None]
        with np.errstate(invalid="ignore"):
            d = sym_epi(xo, xm, R_om, t_om, co["f"], self.f[idx])
            v = ((co["conf"] >= MIN_CONF)[:, :, None] & (self.conf[idx] >= MIN_CONF)[:, None]
                 & np.isfinite(d) & co["ok"][:, None, None, None])
        return d, v

    def sync_cost_offsets(self, off: dict, idx: np.ndarray | None = None) -> tuple[float, int]:
        """sync_cost with the other camera's offset as a mamma_dyn offsets dict (piecewise too): label = the inverse
        of its offset at the master's times."""
        idx = np.arange(len(self.t)) if idx is None else idx
        return self._cost_at_labels(label_at(self.t[idx], off), idx)

    def sync_cost(self, d: float, idx: np.ndarray | None = None) -> tuple[float, int]:
        """Speed-weighted trimmed mean of the same-role epipolar distance with the other camera at label t - d."""
        idx = np.arange(len(self.t)) if idx is None else idx
        return self._cost_at_labels(self.t[idx] - d, idx)

    def _cost_at_labels(self, labels: np.ndarray, idx: np.ndarray) -> tuple[float, int]:
        co = self.other_at(labels)
        dist, v = self.cross(co, idx)
        dist, v = dist[:, [0, 1], [0, 1]], v[:, [0, 1], [0, 1]]
        with np.errstate(invalid="ignore"):
            v &= co["spd"] >= SPEED_MIN_PX_S
        n = int(v.sum())
        if n < 20:
            return float("nan"), n
        return trimmed_mean(dist[v], np.clip(co["spd"][v], SPEED_MIN_PX_S, SPEED_CAP_PX_S)), n

    def swap(self, co: dict, idx: np.ndarray | None = None) -> dict:
        """Share of frames where the lead/follow pairing across the views is better swapped (median epipolar distance
        per pairing, >= COVER_MIN_JOINTS joints in each of the four pairings)."""
        idx = np.arange(len(self.t)) if idx is None else idx
        dist, v = self.cross(co, idx)
        n = v.sum(-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(v, dist, np.nan), -1)  # (M, 2, 2)
        use = (n >= COVER_MIN_JOINTS).all((1, 2))
        same = 0.5 * (med[:, 0, 0] + med[:, 1, 1])
        swp = 0.5 * (med[:, 0, 1] + med[:, 1, 0])
        u = int(use.sum())
        sh = lambda m: _f(m[use].mean(), 5) if u else None  # noqa: E731
        return {"frames": u, "share": sh(swp < SWAP_MARGIN * same), "share_any": sh(swp < same),
                "lead_share": sh(med[:, 0, 1] < med[:, 0, 0]), "follow_share": sh(med[:, 1, 0] < med[:, 1, 1]),
                "margin": SWAP_MARGIN, "same_px_median": _f(np.median(same[use]), 3) if u else None,
                "swapped_px_median": _f(np.median(swp[use]), 3) if u else None,
                "what": "frames where the swapped lead/follow pairing across the views is cheaper than the labelled one by "
                        "more than (1 - margin) (share); share_any: cheaper at all"}


# ------------------------------------------------------------------------------------------------ sync


def scan_curve(cost, range_ms: float, centre_ms: float | None = None) -> dict:
    """dancecap.sync's search on cost(d_s) -> (value, n): 20 ms over +-range then 4 ms within +-40 ms of the minimum
    (centre_ms given: only the 4 ms steps around it, or the full scan when their minimum lands on their edge), 1 ms
    within +-8 ms and the vertex of a parabola through the squared cost (vertex()), once more centred on that vertex.
    -> {delta_ms, min_cost, n, curve, fine, at_edge}."""
    curve: dict[float, tuple] = {}

    def ev(dms: float):
        k = round(float(dms), 3)
        if k not in curve:
            curve[k] = cost(k / 1000.0)
        return curve[k]

    def argmin():
        ok = [(k, v[0]) for k, v in curve.items() if np.isfinite(v[0])]
        return min(ok, key=lambda x: x[1])[0] if ok else None

    def coarse(m):
        for d in np.arange(max(-range_ms, m - S.COARSE_HALF_MS), min(range_ms, m + S.COARSE_HALF_MS) + 1e-9, S.COARSE_MS):
            ev(d)

    if centre_ms is not None:
        coarse(centre_ms)
        m = argmin()
        if m is None or abs(m - centre_ms) >= S.COARSE_HALF_MS - 1e-6:
            return scan_curve(cost, range_ms)
    else:
        for d in np.arange(-range_ms, range_ms + 1e-9, S.SCAN_MS):
            ev(d)
        m = argmin()
        if m is None:
            return {"delta_ms": float("nan"), "min_cost": float("nan"), "n": 0, "at_edge": False,
                    "curve": sorted((k, *v) for k, v in curve.items()), "fine": []}
        coarse(m)
    centre = argmin()

    def vertex(c: float) -> tuple[float, list, list]:
        """1 ms steps within +-8 ms of c and the vertex of a least-squares parabola through the SQUARED cost: the
        epipolar cost is close to a hyperbola sqrt(c0^2 + k^2 (d - d0)^2), whose square is a parabola in d (a parabola
        through the cost itself is pulled towards the window centre by its V shape)."""
        fine = [(float(d), *cost(d / 1000.0)) for d in np.arange(c - S.FINE_HALF_MS, c + S.FINE_HALF_MS + 1e-9, S.FINE_MS)]
        okf = [x for x in fine if np.isfinite(x[1])]
        if not okf:
            return float("nan"), fine, okf
        best = min(okf, key=lambda x: x[1])[0]
        if len(okf) >= 5:
            xs, ys = np.array([x[0] for x in okf]), np.array([x[1] for x in okf])
            a2, a1, _ = np.polyfit(xs - best, ys ** 2, 2)
            if a2 > 0 and abs(-a1 / (2 * a2)) <= S.FINE_HALF_MS:
                best = best - a1 / (2 * a2)
        return float(best), fine, okf

    best, fine, okf = vertex(centre)
    if np.isfinite(best) and abs(best - centre) > 0.25 * S.FINE_MS:  # once more, the window centred on the vertex:
        b2, f2, o2 = vertex(best)  # a symmetric cost fitted off-centre is still biased towards the window centre
        if np.isfinite(b2):
            best, fine, okf = b2, f2, o2
    if not okf:
        return {"delta_ms": float("nan"), "min_cost": float("nan"), "n": 0, "at_edge": False,
                "curve": sorted((k, *v) for k, v in curve.items()), "fine": fine}
    j = int(np.argmin([x[1] for x in okf]))
    return {"delta_ms": float(best), "min_cost": float(okf[j][1]), "n": int(okf[j][2]),
            "at_edge": bool(abs(best) > range_ms - S.COARSE_MS),
            "curve": sorted((k, *v) for k, v in curve.items()), "fine": fine}


def _subwindows(t: np.ndarray, sub_s: float) -> list[tuple[float, np.ndarray]]:
    lo, hi = float(t[0]), float(t[-1])
    n = max(1, int(round((hi - lo) / sub_s)))
    edges = np.linspace(lo, hi + 1e-6, n + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        rows = np.flatnonzero((t >= a) & (t < b))
        if len(rows) >= 15:
            out.append((float((a + b) / 2), rows))
    return out


def drift_fit(tm: np.ndarray, d_ms: np.ndarray) -> dict:
    """delta(t) = a + b (t - t_ref) by least squares over the sub-window estimates (t_ref = their mean time)."""
    tm, d_ms = np.asarray(tm, float), np.asarray(d_ms, float)
    t_ref = float(tm.mean())
    if len(tm) < 2:
        return {"a_ms": float(d_ms[0]) if len(d_ms) else None, "b_ms_per_s": 0.0, "t_ref": t_ref, "se_b_ms_per_s": None}
    A = np.stack([tm - t_ref, np.ones_like(tm)], 1)
    (b, a), *_ = np.linalg.lstsq(A, d_ms, rcond=None)
    se = None
    if len(tm) >= 3:
        rss = float(np.sum((A @ np.array([b, a]) - d_ms) ** 2))
        se = float(np.sqrt(rss / (len(tm) - 2) / max(np.sum((tm - t_ref) ** 2), 1e-12)))
    return {"a_ms": float(a), "b_ms_per_s": float(b), "t_ref": t_ref, "se_b_ms_per_s": se}


def _median3(y: np.ndarray) -> np.ndarray:
    """Running median of 3 (ends kept: a step near an end stays a step): one odd window does not bend the curve."""
    y = np.asarray(y, float)
    if len(y) < 3:
        return y.copy()
    return np.r_[y[0], np.median(np.stack([y[:-2], y[1:-1], y[2:]]), 0), y[-1]]


def piecewise_measure(pair: "TwoViewPair", centre_ms: float, range_ms: float, log=print) -> dict:
    """delta(t) of the other camera: an estimate per PW_WIN_S window every PW_STEP_S (all master frames, the even ones,
    the odd ones), counted where the halves agree within PW_HALF_MS; the curve = the running median of 3 over the
    counted windows (linear between, flat outside). Reproducible = the even-half and odd-half curves (built the same way)
    agree; better = the whole window's cost with the curve is not above the constant's (module docstring: a moving
    camera's audio clock follows its distance to the speakers, ~3 ms per metre)."""
    t = pair.t
    lo, hi = float(t[0]), float(t[-1])
    cs = np.arange(lo + PW_WIN_S / 2, hi - PW_WIN_S / 2 + 1e-6, PW_STEP_S)
    if len(cs) == 0:
        cs = np.array([(lo + hi) / 2])
    rows_all = np.arange(len(t))
    out = []
    for c in cs:
        rows = rows_all[(t >= c - PW_WIN_S / 2) & (t < c + PW_WIN_S / 2)]
        est = {}
        for k, r in (("all", rows), ("even", rows[0::2]), ("odd", rows[1::2])):
            sc = scan_curve(lambda d, r=r: pair.sync_cost(d, r), range_ms, centre_ms=centre_ms)
            est[k] = (sc["delta_ms"], sc["n"], sc["at_edge"])
        a, e, o = est["all"], est["even"], est["odd"]
        ok = bool(np.isfinite(a[0]) and a[1] >= SUB_MIN_OBS and not a[2] and np.isfinite(e[0]) and np.isfinite(o[0])
                  and abs(e[0] - o[0]) <= PW_HALF_MS)
        out.append({"t": round(float(c), 2), "delta_ms": _f(a[0], 2), "even_ms": _f(e[0], 2), "odd_ms": _f(o[0], 2),
                    "n": int(a[1]), "counted": ok})
    good = [w for w in out if w["counted"]]
    if not good:
        return {"windows": out, "curve": [], "reproducible": False, "share": 0.0}
    tg = np.array([w["t"] for w in good])
    curve, ce, co = (_median3(np.array([w[k] for w in good], float)) for k in ("delta_ms", "even_ms", "odd_ms"))
    rep_d = np.abs(ce - co)
    share = len(good) / len(out)
    pw = {"windows": out, "curve": [[float(a), round(float(b), 2)] for a, b in zip(tg, curve)],
          "share": round(share, 3), "repro_median_ms": _f(np.median(rep_d), 2), "repro_max_ms": _f(rep_d.max(), 2),
          "range_ms": _f(curve.max() - curve.min(), 2)}
    off = {"piecewise": [[a, b / 1000.0] for a, b in pw["curve"]]}
    cost_pw = pair.sync_cost_offsets(off)[0]
    cost_c = pair.sync_cost(centre_ms / 1000.0)[0]
    pw.update(cost_px=_f(cost_pw, 4), cost_constant_px=_f(cost_c, 4),
              reproducible=bool(share >= PW_MIN_SHARE and np.median(rep_d) <= PW_REPRO_MEDIAN_MS
                                and rep_d.max() <= PW_REPRO_MAX_MS and cost_pw <= cost_c))
    log(f"[twoview] piecewise: {len(good)}/{len(out)} windows counted, delta {curve.min():+.1f}..{curve.max():+.1f} ms, "
        f"even vs odd median {pw['repro_median_ms']} / max {pw['repro_max_ms']} ms, cost {cost_pw:.3f} px vs constant "
        f"{cost_c:.3f} px -> {'reproducible' if pw['reproducible'] else 'not reproducible'}")
    return pw


def sync_measure(pair: TwoViewPair, range_ms: float = 300.0, sub_s: float = 30.0, log=print) -> dict:
    """The other camera's offset against the master: whole-window curve, sub-window curves, drift fit, curve depth and
    the swap test at the optimum (see the module docstring). No files."""
    g = scan_curve(pair.sync_cost, range_ms)
    log(f"[twoview] sync: whole window {g['delta_ms']:+.2f} ms ({g['min_cost']:.3f} px, {g['n']} joints)")
    subs = []
    for tm, rows in _subwindows(pair.t, sub_s):
        if not np.isfinite(g["delta_ms"]):
            break
        r = scan_curve(lambda d, rows=rows: pair.sync_cost(d, rows), range_ms, centre_ms=g["delta_ms"])
        ok = bool(np.isfinite(r["delta_ms"]) and r["n"] >= SUB_MIN_OBS and not r["at_edge"])
        subs.append({"t_mid": round(tm, 2), "delta_ms": _f(r["delta_ms"], 2), "min_px": _f(r["min_cost"], 3),
                     "n": r["n"], "n_frames": int(len(rows)), "valid": ok, "fine": r["fine"]})
        log(f"[twoview]   {tm:7.1f} s: {r['delta_ms']:+.2f} ms ({r['n']} joints){'' if ok else ' (not valid)'}")
    good = [s for s in subs if s["valid"]]
    ds = np.array([s["delta_ms"] for s in good], float)
    fit = drift_fit(np.array([s["t_mid"] for s in good], float), ds) if good else None
    d0 = g["delta_ms"] / 1000.0 if np.isfinite(g["delta_ms"]) else 0.0
    c0 = pair.sync_cost(0.0)[0]
    depth = {f"{s:+.0f}": _f(pair.sync_cost(d0 + s / 1000.0)[0] / g["min_cost"], 4) if np.isfinite(g["delta_ms"])
             and g["min_cost"] > 0 else None for s in (-DEPTH_MS, DEPTH_MS)}
    sw = pair.swap(pair.other_at(pair.t - d0)) if np.isfinite(g["delta_ms"]) else None
    span_s = (good[-1]["t_mid"] - good[0]["t_mid"]) if len(good) >= 2 else 0.0
    use_rate = bool(fit is not None and len(good) >= 4 and fit["se_b_ms_per_s"] is not None
                    and abs(fit["b_ms_per_s"]) * span_s >= DRIFT_USE_MS
                    and abs(fit["b_ms_per_s"]) >= 3 * fit["se_b_ms_per_s"])
    detr = None
    if fit is not None and len(ds) >= 2:  # the sub-window estimates about the fitted line (a step leaves them large)
        res_l = ds - (fit["a_ms"] + fit["b_ms_per_s"] * (np.array([s["t_mid"] for s in good], float) - fit["t_ref"]))
        detr = float(res_l.max() - res_l.min())
    pw = piecewise_measure(pair, g["delta_ms"], range_ms, log) if np.isfinite(g["delta_ms"]) else None
    return {"global": g, "sub_windows": subs, "n_valid": len(good),
            "spread_ms": float(ds.max() - ds.min()) if len(ds) >= 2 else None, "spread_detrended_ms": detr, "fit": fit,
            "use_rate": use_rate, "cost_at_zero_px": _f(c0, 4), "depth_ratio": depth, "swap": sw, "piecewise": pw}


def sync_doc(work: Path, master: str, other: str, res: dict, range_ms: float, sub_s: float) -> dict:
    """sync/offsets.json in dancecap.sync's layout; the clocks block exactly as recap_front.fixed_sync_doc builds it.
    With use_rate, delta = the drift line at t_ref and stability is judged about that line (module docstring)."""
    g, fit = res["global"], res["fit"] or {}
    spread, use_rate = res["spread_ms"], bool(res["use_rate"])
    judged = res["spread_detrended_ms"] if use_rate else spread
    stable = bool(res["n_valid"] >= 3 and judged is not None and judged < S.STABLE_MS)
    pw = res.get("piecewise") or {}
    piecewise = bool(not stable and pw.get("reproducible"))  # a constant that holds is kept; a delta(t) the halves reproduce else
    stable = stable or piecewise
    use_rate = use_rate and not piecewise
    d_ms = round(float(fit["a_ms"] if use_rate else g["delta_ms"]), 2)
    rate = float(fit.get("b_ms_per_s") or 0.0) / 1000.0
    o = {"delta": round(d_ms / 1000.0, 5), "delta_ms": d_ms, "delta_global_ms": round(float(g["delta_ms"]), 2),
         "stable": stable, "use": True, "spread_ms": None if spread is None else round(spread, 2),
         "spread_detrended_ms": _f(res["spread_detrended_ms"], 2),
         "stable_on": ("piecewise: the even- and odd-frame curves reproduce delta(t)" if piecewise else
                       "spread about the drift line (use_rate)" if use_rate else "spread"),
         **({"piecewise": [[a, round(b / 1000.0, 5)] for a, b in pw["curve"]],
             "piecewise_detail": {k: pw.get(k) for k in ("share", "repro_median_ms", "repro_max_ms", "range_ms", "cost_px",
                                                         "cost_constant_px")}} if piecewise else {}),
         # dancecap.sync's sub-window keys (delta_ms None where the window is not valid; its measurement kept)
         "sub_windows": [{**{k: v for k, v in s.items() if k != "fine"}, "delta_ms": s["delta_ms"] if s["valid"] else None,
                          "delta_ms_measured": s["delta_ms"]} for s in res["sub_windows"]],
         "n_sub_valid": res["n_valid"], "rate": rate, "t_ref": _f(fit.get("t_ref"), 3) or 0.0,
         "use_rate": use_rate,
         "fit": {"a_ms": _f(fit.get("a_ms"), 3), "b_ms_per_100s": _f(100 * (fit.get("b_ms_per_s") or 0.0), 3),
                 "se_b_ms_per_100s": _f(100 * fit["se_b_ms_per_s"], 3) if fit.get("se_b_ms_per_s") is not None else None},
         "residual_px": {"metric": "speed-weighted trimmed mean symmetric epipolar distance",
                         "at_delta": _f(g["min_cost"], 4), "at_zero": res["cost_at_zero_px"],
                         **{f"ratio_at_{k}ms": v for k, v in res["depth_ratio"].items()}},
         "at_search_edge": bool(g["at_edge"]), "attached": False, "swap": res["swap"]}
    m = {"delta": 0.0, "delta_ms": 0.0, "stable": True, "use": True, "use_rate": False, "rate": 0.0, "t_ref": 0.0,
         "attached": False, "master": True}
    return {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "what": "per-camera residual video offset: corrected reference time = t + delta (+ rate * (t - t_ref) only "
                    "where use_rate is true). TWO cameras: measured by cameras.twoview sync (dancecap.sync needs >= 3)",
            "gauge": f"camera {master} = 0", "method": "two_view_epipolar",
            "method_detail": {"joints": f"COCO-17 of both dancers (people/<cam>.npz + dancers/roles.json), conf >= "
                                        f"{MIN_CONF} in both views, image speed >= {SPEED_MIN_PX_S:g} px/s in camera {other}",
                              "cost": f"symmetric epipolar distance (px), speed-weighted (cap {SPEED_CAP_PX_S:g} px/s), "
                                      f"trimmed mean dropping {TRIM:.0%} of the weight",
                              "search_ms": {"range": range_ms, "scan": S.SCAN_MS, "coarse": S.COARSE_MS,
                                            "coarse_half": S.COARSE_HALF_MS, "fine": S.FINE_MS, "fine_half": S.FINE_HALF_MS},
                              "sub_window_s": sub_s, "stable_if_spread_below_ms": S.STABLE_MS,
                              "stable_with_use_rate": "the spread of the sub-window estimates about the drift line",
                              "use_rate_if": f"drift over the sub-windows >= {DRIFT_USE_MS:g} ms and |b| >= 3 SE"},
            "offsets": {other: o, master: m},
            # the audio clocks these deltas sit on top of (offsets_check refuses the file on any other clocks)
            "clocks": {"alignment": (S.take_clocks(work) or {}).get("alignment"), "sources": S.frame_clocks(work)},
            "host": "pod" if os.environ.get("DANCECAP_POD") == "1" else "local"}


def plot_sync(res: dict, doc: dict, other: str, out_png: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["global"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    cur = np.array([x[:2] for x in g["curve"]], float).reshape(-1, 2)
    fine = np.array([x[:2] for x in g["fine"]], float).reshape(-1, 2)
    ax[0].plot(cur[:, 0], cur[:, 1], color="#888", lw=1, label="scan 20 ms + coarse 4 ms")
    if len(fine):
        ax[0].plot(fine[:, 0], fine[:, 1], color="#1f77b4", lw=1.6, label="fine (1 ms)")
    ax[0].axvline(g["delta_ms"], color="k", lw=1)
    ax[0].set_xlabel("delta (ms)")
    ax[0].set_ylabel("trimmed epipolar distance (px)")
    o = doc["offsets"][other]
    ax[0].set_title(f"cam {other}: {o['delta_ms']:+.2f} ms ({'stable' if o['stable'] else 'UNSTABLE'})", fontsize=9)
    ax[0].legend(fontsize=7)
    for s in res["sub_windows"]:
        f = np.array([x[:2] for x in s["fine"]], float).reshape(-1, 2)
        if len(f) and np.isfinite(f[:, 1]).any():
            ax[1].plot(f[:, 0], f[:, 1] / np.nanmin(f[:, 1]), lw=0.9, ls="-" if s["valid"] else ":",
                       label=f"{s['t_mid']:.0f} s")
    ax[1].set_xlabel("delta (ms)")
    ax[1].set_ylabel("cost / its minimum")
    ax[1].set_title(f"sub-windows ({res['n_valid']} valid)", fontsize=9)
    ax[1].legend(fontsize=6, ncol=2)
    sv = [s for s in res["sub_windows"] if s["delta_ms"] is not None]
    ax[2].plot([s["t_mid"] for s in sv], [s["delta_ms"] for s in sv], "ko", ms=4)
    fit = res["fit"]
    if fit and sv:
        tt = np.array([sv[0]["t_mid"], sv[-1]["t_mid"]])
        ax[2].plot(tt, fit["a_ms"] + fit["b_ms_per_s"] * (tt - fit["t_ref"]), color="#d62728",
                   label=f"{fit['a_ms']:+.2f} ms {100 * fit['b_ms_per_s']:+.2f} ms/100 s")
        ax[2].legend(fontsize=7)
    pw = res.get("piecewise") or {}
    if pw.get("curve"):
        cv = np.array(pw["curve"], float)
        ax[2].plot(cv[:, 0], cv[:, 1], color="#2ca02c", lw=1.4,
                   label=f"delta(t) {'reproducible' if pw.get('reproducible') else 'not reproducible'}")
        wv = [w for w in pw["windows"] if w["counted"]]
        ax[2].plot([w["t"] for w in wv], [w["even_ms"] for w in wv], "v", color="#2ca02c", ms=3, alpha=0.6)
        ax[2].plot([w["t"] for w in wv], [w["odd_ms"] for w in wv], "^", color="#2ca02c", ms=3, alpha=0.6)
        ax[2].legend(fontsize=7)
    ax[2].set_xlabel("t (s)")
    ax[2].set_ylabel("delta (ms)")
    ax[2].set_title(f"spread {o['spread_ms']} ms (about the line {o['spread_detrended_ms']} ms)", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def cmd_sync(work: Path, master: str, other: str, range_ms: float = 300.0, sub_s: float = 30.0,
             window: tuple[float, float] | None = None, log=print) -> dict:
    """Measure, write curves_2v.json + the review plot, and sync/offsets.json only when the estimate is usable:
    SystemExit(2) when it is unstable, at the search edge or has no minimum (offsets.json is left as it was)."""
    for c in (master, other):  # the clocks block (sync.frame_clocks) comes from them: without it the file is unverifiable
        _need(work / "frames" / c / "frames.json")
    pair = TwoViewPair(load_track(work, master, world=False), load_track(work, other, world=False),
                       role_kp(work, master), role_kp(work, other), window)
    t0 = time.time()
    res = sync_measure(pair, range_ms, sub_s, log)
    g = res["global"]
    why = []
    if not np.isfinite(g["delta_ms"]):
        why.append("no finite minimum (too few joints with conf >= 0.5 in both views and image speed >= 50 px/s)")
        doc = None
    else:
        doc = sync_doc(work, master, other, res, range_ms, sub_s)
        doc["seconds"] = round(time.time() - t0, 1)
        o = doc["offsets"][other]
        if o["at_search_edge"]:
            why.append(f"at the search edge ({o['delta_ms']:+.1f} ms, range +-{range_ms:g} ms)")
        if not o["stable"]:
            sp = o["spread_detrended_ms"] if o["use_rate"] else o["spread_ms"]
            why.append(f"unstable: {res['n_valid']} valid {sub_s:g} s sub-windows, {o['stable_on']} {sp} ms "
                       f"(needs >= 3 and < {S.STABLE_MS:g} ms)")
    curves = {"master": master, "other": other, "accepted": not why, "why": why, "global": g,
              "sub_windows": res["sub_windows"], "fit": res["fit"], "depth_ratio": res["depth_ratio"],
              "cost_at_zero_px": res["cost_at_zero_px"], "swap": res["swap"], "piecewise": res.get("piecewise"), "doc": doc}
    _write_json(work / "sync" / "curves_2v.json", curves)
    if doc is not None:
        log(f"[twoview] -> {plot_sync(res, doc, other, work / 'review' / 'sync_2v.png')}")
    if why:
        log(f"[twoview] sync REJECTED, sync/offsets.json not written (an existing one is left as it was): {'; '.join(why)}")
        raise SystemExit(2)
    p = _write_json(work / "sync" / "offsets.json", doc)
    log(f"[twoview] -> {p}: camera {other} {doc['offsets'][other]['delta_ms']:+.2f} ms (gauge {master} = 0), "
        f"rate {100 * 1000 * doc['offsets'][other]['rate']:+.2f} ms/100 s (use_rate {doc['offsets'][other]['use_rate']})")
    return doc


# ------------------------------------------------------------------------------------------------ check


def _robust_cv(x: np.ndarray) -> float | None:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return None
    m = float(np.median(x))
    return 1.4826 * _mad(x) / m if m > 0 else None


def check_measure(pair: TwoViewPair, offsets: dict, statures: dict, soles: dict,
                  window: tuple[float, float] | None = None) -> dict:
    """Sync-corrected two-view triangulation of both dancers (world frame) and the G7 measures (module docstring)."""
    mc, oc = pair.m["cam"], pair.o["cam"]
    tau = md.apply_offset(pair.t, offsets.get(mc))
    co = pair.other_at(label_at(tau, offsets.get(oc)))
    idx = np.arange(len(pair.t))
    v = pair.cross(co, idx)[1][:, [0, 1], [0, 1]]  # (M, 2, 17): the same dancer's joint in both views
    M = len(pair.t)
    X = np.full((M, 2, 17, 3), np.nan)
    e_m, e_o = np.full((M, 2, 17), np.nan), np.full((M, 2, 17), np.nan)
    fi, ri, ji = np.nonzero(v)
    if len(fi):
        Pm = np.concatenate([pair.R, pair.tvec[:, :, None]], 2)
        Po = np.concatenate([co["R"], co["tvec"][:, :, None]], 2)
        Xn = tri2(Pm[fi], Po[fi], pair.x[fi, ri, ji, :2], co["x"][fi, ri, ji, :2])
        um, zm = reproj(Xn, pair.R[fi], pair.tvec[fi], pair.f[fi], pair.pp[fi], pair.k1[fi])
        uo, zo = reproj(Xn, co["R"][fi], co["tvec"][fi], co["f"][fi], co["pp"][fi], co["k1"][fi])
        front = (zm > 0) & (zo > 0)
        em = np.where(front, np.linalg.norm(um - pair.kp[fi, ri, ji, :2], axis=1), np.inf)
        eo = np.where(front, np.linalg.norm(uo - co["kp"][fi, ri, ji, :2], axis=1), np.inf)
        e_m[fi, ri, ji], e_o[fi, ri, ji] = em, eo
        good = front & (em <= BONE_MAX_PX) & (eo <= BONE_MAX_PX)
        X[fi[good], ri[good], ji[good]] = Xn[good]
    rep = {}
    for cam, e in ((mc, e_m), (oc, e_o)):
        x = e[v]
        x = np.where(np.isfinite(x), x, 1e3)
        rep[cam] = {"median": _f(np.median(x), 3) if len(x) else None, "p90": _f(np.percentile(x, 90), 3) if len(x) else None,
                    "n": int(len(x))}
    # coverage (>= COVER_MIN_JOINTS joints in both views) per role, per 10 s window, over EVERY master frame of the
    # window: frames outside pair.rows (master pose not ok, or no dancer seen) count as not covered, a 10 s window
    # without master frames as 0
    cov = {}
    nj = v.sum(-1)
    t_all = pair.m["t"]
    w0 = window[0] if window else float(t_all[0])
    w1 = window[1] if window else float(t_all[-1])
    inw = (t_all >= w0) & (t_all <= w1)
    for r, role in enumerate(ROLES):
        c_all = np.zeros(len(t_all), bool)
        c_all[pair.rows] = nj[:, r] >= COVER_MIN_JOINTS
        tw, c = t_all[inw], c_all[inw]
        low = []
        for a in np.arange(w0, w1, 10.0):
            b = min(a + 10.0, w1 + 1e-9)
            sel = (tw >= a) & (tw < b)
            sh = float(c[sel].mean()) if sel.any() else 0.0
            if sh < 0.5:
                low.append([round(float(a), 1), round(float(b), 1), round(sh, 3)])
        cov[role] = {"share": _f(c.mean(), 4) if len(c) else None, "frames": int(len(c)), "frames_used": int(M),
                     "windows_below_50pct": low}
    # bones, statures, stance ankles
    bones, lr, stat, ank = {}, {}, {}, {}
    for r, role in enumerate(ROLES):
        bones[role], lr[role] = {}, {}
        for name, pairs in BONES.items():
            per = []
            for side, (a, b) in zip(("l", "r") if len(pairs) == 2 else ("",), pairs):
                L = np.linalg.norm(X[:, r, a] - X[:, r, b], axis=1)
                key = f"{name}_{side}" if side else name
                ok = np.isfinite(L)
                bones[role][key] = {"median_m": _f(np.median(L[ok]), 4) if ok.any() else None,
                                    "cv": _f(np.std(L[ok]) / np.mean(L[ok]), 4) if ok.sum() >= 10 else None,
                                    "cv_robust": _f(_robust_cv(L), 4), "n": int(ok.sum())}
                per.append(bones[role][key]["median_m"])
            if len(per) == 2 and None not in per:
                lr[role][name] = _f(abs(per[0] - per[1]) / (0.5 * (per[0] + per[1])), 4)
        with warnings.catch_warnings():  # a dancer seen in one view only: all-NaN segments
            warnings.simplefilter("ignore", RuntimeWarning)
            cm = W.chain_measure({k: X[:, r, k] for k in range(17)})
        if cm is not None and role in statures:
            sole = float(soles.get(role, W.SOLE_M))
            implied = cm["chain"] / W.CHAIN_FRAC - sole + W.CHAIN_FRAC_SOLE_M  # the inverse of world.chain_stature
            stat[role] = {"chain_m": _f(cm["chain"], 4), "implied_stature_m": _f(implied, 4),
                          "toml_stature_m": statures[role], "sole_m": sole, "ratio": _f(implied / statures[role], 4),
                          "rel_se": _f(cm["rel_se"], 4), "halves_rel_diff": _f(cm["halves_rel_diff"], 4), "n": cm["n"]}
        ank[role] = {"expected_ankle_m": _f(W.ANKLE_FRAC * statures.get(role, W.DEFAULT_STATURE[role])
                                            + float(soles.get(role, W.SOLE_M)), 4)}
        for side, k in zip(("left", "right"), KP_ANKLES):
            P = X[:, r, k]
            if M < 3:
                continue
            dt2 = pair.t[2:] - pair.t[:-2]
            vh = np.linalg.norm((P[2:] - P[:-2])[:, [0, 2]], axis=1) / dt2
            st = np.isfinite(vh) & (dt2 <= 2 * GAP_FRAMES * pair.m["dt"]) & (vh < STANCE_V)
            y = P[1:-1][st, 1]
            ank[role][side] = {"stance_median_m": _f(np.median(y), 4) if len(y) else None, "n": int(len(y))}
        meds = [ank[role][s]["stance_median_m"] for s in ("left", "right") if (ank[role].get(s) or {}).get("stance_median_m")
                is not None]
        ank[role]["diff_m"] = _f(np.median(meds) - ank[role]["expected_ankle_m"], 4) if meds else None
    lf = None
    if all(stat.get(r, {}).get("ratio") for r in ROLES):
        lf = _f(stat["lead"]["ratio"] / stat["follow"]["ratio"], 4)
    # camera to couple (horizontal, world metres): couple = mean of both dancers' mid-hips
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        couple = np.nanmean(np.nanmean(X[:, :, [11, 12]], 2), 1)
    dist_c = {}
    for cam, C in ((mc, pair.C), (oc, co["C"])):
        h = np.linalg.norm((C - couple)[:, [0, 2]], axis=1)
        h = h[np.isfinite(h)]
        dist_c[cam] = _f(np.median(h), 3) if len(h) else None
    cv_max = lambda r, k: _f(max([b[k] for b in bones[r].values() if b[k] is not None], default=np.nan), 4)  # noqa: E731
    return {"master": mc, "other": oc, "frames": int(M), "reproj_px": rep, "coverage": cov, "bones": bones,
            "bones_lr": lr, "bone_cv_max": {r: cv_max(r, "cv") for r in ROLES},
            "bone_cv_robust_max": {r: cv_max(r, "cv_robust") for r in ROLES},
            "bone_cv_what": "cv = SD / mean (the runbook's CV, G7); cv_robust = 1.4826 MAD / median",
            "bone_lr_max": {r: _f(max(lr[r].values(), default=np.nan), 4) for r in ROLES},
            "stature": stat, "stature_lead_follow_ratio": lf, "stance_ankles": ank, "swap": pair.swap(co, idx),
            "cam_to_couple_m": dist_c, "bone_max_px": BONE_MAX_PX,
            "note": "the implied statures sit near the TOML ones by construction (the world scale came from the same chain "
                    "and statures): only the lead/follow disagreement is informative - a consistency check, not a metric "
                    "verification"}


def plot_check(doc: dict, out_png: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    cams = list(doc["reproj_px"])
    ax[0].bar(np.arange(len(cams)) - 0.2, [doc["reproj_px"][c]["median"] or 0 for c in cams], 0.4, label="median")
    ax[0].bar(np.arange(len(cams)) + 0.2, [doc["reproj_px"][c]["p90"] or 0 for c in cams], 0.4, label="p90")
    ax[0].set_xticks(np.arange(len(cams)), [f"cam {c}" for c in cams])
    ax[0].set_ylabel("reprojection (px)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("two-view triangulation", fontsize=9)
    for k, role in enumerate(ROLES):
        b = doc["bones"][role]
        names = list(b)
        ax[1].bar(np.arange(len(names)) + 0.4 * k - 0.2, [100 * (b[n]["cv"] or 0) for n in names], 0.4, label=role)
        ax[1].set_xticks(np.arange(len(names)), names, rotation=60, fontsize=6)
    ax[1].axhline(8, color="#d62728", lw=0.8)
    ax[1].set_ylabel("CV = SD / mean (%)")
    ax[1].legend(fontsize=7)
    ax[1].set_title("bone lengths", fontsize=9)
    st = doc["stature"]
    ax[2].bar(np.arange(2), [100 * ((st.get(r) or {}).get("ratio", 1.0) - 1) for r in ROLES], 0.5)
    ax[2].set_xticks(np.arange(2), [f"{r}\ncover {doc['coverage'][r]['share']}" for r in ROLES])
    ax[2].axhspan(-3, 3, color="#2ca02c", alpha=0.15)
    ax[2].set_ylabel("implied / TOML stature - 1 (%)")
    ax[2].set_title(f"statures (lead/follow {doc['stature_lead_follow_ratio']}), swap {doc['swap']['share']}", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def _track_arg(work: Path, cam: str, p: str | Path, log=print) -> Path:
    """check --track CAM=path (relative to work). The attached PTZ track's earlier name cameras/<cam>_ptz.npz (inside
    cameras/*.npz, which mamma_dyn.load_tracks and other readers take for a camera) reads cameras/ptz/<cam>.npz."""
    p = Path(p)
    q = p if p.is_absolute() else work / p
    if not q.exists() and q.resolve() == (work / "cameras" / f"{cam}_ptz.npz").resolve() and ptz_track_path(work, cam).exists():
        log(f"[twoview] --track {cam}={p}: the attached PTZ track is {ptz_track_path(work, cam).relative_to(work)}")
        return ptz_track_path(work, cam)
    return p


def cmd_check(work: Path, cfg: dict, master: str | None = None, tracks: dict | None = None, tag: str | None = None,
              log=print) -> dict:
    """check: needs sync/offsets.json (mamma_dyn.read_sync_offsets: refuses stale clocks) and world_from_sfm.json.
    An offsets file of another method than this module's sync is used with a warning; the doc records which file."""
    _need(work / "cameras" / "world_from_sfm.json")
    raw = _json(_need(work / "sync" / "offsets.json"))
    offsets = md.read_sync_offsets(work)
    cams = sorted(offsets)
    if len(cams) != 2:
        raise SystemExit(f"[twoview] check: sync/offsets.json holds {cams}, not two cameras")
    if raw.get("method") != "two_view_epipolar":
        log(f"[twoview] WARNING: sync/offsets.json is method {raw.get('method')!r}, not this module's two_view_epipolar sync")
    if master is None:
        master = next((c for c, v in raw.get("offsets", {}).items() if isinstance(v, dict) and v.get("master")), "02")
    other = next(c for c in cams if c != master)
    tracks = {c: _track_arg(work, c, p, log) for c, p in (tracks or {}).items()}
    mt, ot = load_track(work, master, tracks.get(master)), load_track(work, other, tracks.get(other))
    window = _window(cfg)
    pair = TwoViewPair(mt, ot, role_kp(work, master), role_kp(work, other), window)
    statures, st_src = W.statures_from_take(cfg)
    soles, _ = W.sole_from_take(cfg)
    doc = check_measure(pair, offsets, statures, soles, window)
    doc.update({"tag": tag, "tracks": {master: mt["path"], other: ot["path"]}, "stature_source": st_src,
                "offsets_ms": {c: _f(1000 * v["delta"], 2) for c, v in offsets.items()},
                "offsets_file": {"method": raw.get("method"), "created": raw.get("created")}, "window": window,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    sfx = f"_{tag}" if tag else ""
    p = _write_json(work / "cameras" / "qa" / f"twoview_check{sfx}.json", doc)
    png = plot_check(doc, work / "review" / f"twoview_check{sfx}.png")
    log(f"[twoview] -> {p}, {png}: reprojection " + ", ".join(f"{c} {v['median']} / {v['p90']} px"
                                                              for c, v in doc["reproj_px"].items())
        + f"; coverage {[doc['coverage'][r]['share'] for r in ROLES]}; swap {doc['swap']['share']}")
    return doc


# ------------------------------------------------------------------------------------------------ PTZ


def ptz_sample_times(n: np.ndarray, shift: float, cut_fps: float = 30.0, src_fps: float = SRC_FPS,
                     rule: str = "fps") -> np.ndarray:
    """This take's clock time of PTZ cut frame n (cut = ffmpeg -ss shift -vf fps=cut_fps of a src_fps original).
    'fps': the source frame FFmpeg's fps filter (round=near) puts in output frame n, the last whose output pts
    round((t - shift) x cut_fps) is <= n: k = ceil((shift + (n + 0.5) / cut_fps) x src_fps) - 1 (an exact half rounds
    up, away from frame n: the 1e-6 keeps it out); 'nearest': round((shift + n / cut_fps) x src_fps); t = k / src_fps.
    'nominal': shift + n / cut_fps."""
    n = np.asarray(n, float)
    if rule == "fps":
        return (np.ceil((shift + (n + 0.5) / cut_fps) * src_fps - 1e-6) - 1) / src_fps
    if rule == "nearest":
        return np.round((shift + n / cut_fps) * src_fps) / src_fps
    if rule == "nominal":
        return shift + n / cut_fps
    raise SystemExit(f"[twoview] ptz: unknown cut rule {rule!r} (one of {CUT_RULES})")


def cut_phase(n: np.ndarray, shift: float, cut_fps: float = 30.0, src_fps: float = SRC_FPS) -> np.ndarray:
    """frac((shift + n / cut_fps) x src_fps) of cut frame n: below 0.5 the 'fps' and 'nearest' rules pick different
    source frames (one apart), above they agree."""
    x = (shift + np.asarray(n, float) / cut_fps) * src_fps
    return x - np.floor(x)


def load_ptz(ptz_work: Path, shift: float, cut_fps: float = 30.0, src_fps: float = SRC_FPS, rule: str = "fps",
             track: Path | None = None) -> dict:
    """camera_work/final/track_s*.npz (frame, rv, f, rms_px, n_inl) + its skel_s*.npz lens of another take, on this take's
    clock. Only ok samples (rms_px <= 3, n_inl >= 50, finite) are kept for interpolation."""
    final = Path(ptz_work) / "camera_work" / "final"
    if track is None:
        cands = sorted(final.glob("track_s*.npz"))
        if len(cands) != 1:
            raise SystemExit(f"[twoview] missing input: want exactly one {final}/track_s*.npz, found {len(cands)}")
        track = cands[0]
    track = _need(Path(track))
    with np.load(track) as z:
        frame, rv, f = np.asarray(z["frame"], int), np.asarray(z["rv"], float), np.asarray(z["f"], float)
        rms, n_inl = np.asarray(z["rms_px"], float), np.asarray(z["n_inl"], int)
    sk = track.with_name(track.name.replace("track_", "skel_"))
    lens, size = np.zeros(4), None
    if sk.exists():
        with np.load(sk) as z:
            lens, size = np.asarray(z["lens"], float), np.asarray(z["size"], int).tolist()
    o = np.argsort(frame)
    frame, rv, f, rms, n_inl = frame[o], rv[o], f[o], rms[o], n_inl[o]
    with np.errstate(invalid="ignore"):
        good = np.isfinite(rv).all(1) & np.isfinite(f) & (f > 0) & (rms <= PTZ_OK_RMS) & (n_inl >= PTZ_OK_INL)
    return {"t": ptz_sample_times(frame[good], shift, cut_fps, src_fps, rule), "frame": frame[good],
            "phase": cut_phase(frame[good], shift, cut_fps, src_fps),
            "R": Rot.from_rotvec(rv[good]).as_matrix(), "f": f[good], "rms": rms[good], "n_inl": n_inl[good],
            "lens": lens.tolist(), "size": size, "path": str(track), "n_total": int(len(frame)), "n_ok": int(good.sum()),
            "shift": shift, "cut_fps": cut_fps, "src_fps": src_fps, "rule": rule}


def ptz_at(ptz: dict, t: np.ndarray, dt_s: float = 0.0) -> dict:
    """PTZ rotation (reference -> camera), focal, rms, n_inl at this take's times t (+ dt_s): slerp / log-linear between
    the bracketing ok samples; ok = t inside and the two samples adjacent cut frames (or t on a sample)."""
    t = np.asarray(t, float) + dt_s
    i0, i1, a, span, inside = _bracket(ptz["t"], t)
    adj = (ptz["frame"][i1] - ptz["frame"][i0]) <= 1
    return {"R": slerp_pairs(ptz["R"][i0], ptz["R"][i1], a),
            "f": np.exp((1 - a) * np.log(ptz["f"][i0]) + a * np.log(ptz["f"][i1])),
            "rms": np.maximum(ptz["rms"][i0], ptz["rms"][i1]), "n_inl": np.minimum(ptz["n_inl"][i0], ptz["n_inl"][i1]),
            "ok": inside & (adj | (a <= 0) | (a >= 1))}


def _scan_shift(fn, half_ms: float = SHIFT_SCAN_MS) -> tuple[float, float, float]:
    """fn(shift_s) over +-half_ms in 1 ms steps -> (best shift ms: vertex of a parabola through the 3 values around the
    minimum, the minimum, the value at 0)."""
    shifts = np.arange(-half_ms, half_ms + 1e-9, 1.0)
    vals = np.array([fn(s / 1000.0) for s in shifts])
    k = int(np.argmin(vals))
    best = float(shifts[k])
    if 0 < k < len(shifts) - 1:
        a2, a1, _ = np.polyfit(shifts[k - 1:k + 2] - best, vals[k - 1:k + 2], 2)
        best = best - a1 / (2 * a2) if a2 > 0 else best
    return best, float(vals.min()), float(vals[int(np.argmin(np.abs(shifts)))])


def pan_deg(R: np.ndarray) -> np.ndarray:
    """Pan angle (deg) of the optical axis in the PTZ reference frame (x right, y down, z forward of frame 0)."""
    d = np.asarray(R, float)[:, 2, :]  # R^T e_z: the optical axis in the reference frame
    return np.degrees(np.arctan2(d[:, 0], d[:, 2]))


def _levels(t: np.ndarray, f: np.ndarray, gap: tuple[float, float]) -> dict:
    lo, hi = f[t < gap[0]], f[t > gap[1]]
    mid = f[(t >= gap[0]) & (t <= gap[1])]
    a, b = (float(np.median(x)) if len(x) else None for x in (lo, hi))
    return {"before_px": _f(a, 2), "after_px": _f(b, 2), "ratio": _f(b / a, 4) if a and b else None,
            "inside_gap_px": [_f(np.min(mid), 2), _f(np.median(mid), 2), _f(np.max(mid), 2)] if len(mid) else None,
            "gap_s": list(gap)}


def ptz_compare(loc: dict, ptz: dict, raw_C: np.ndarray | None = None, s_world: float | None = None,
                zoom_gap: tuple[float, float] = ZOOM_GAP) -> dict:
    """--mode check on a loc track (SfM frame) and a PTZ track (see the module docstring)."""
    pa = ptz_at(ptz, loc["t"])
    sel = loc["ok"] & pa["ok"]
    if sel.sum() < 10:
        raise SystemExit(f"[twoview] ptz check: only {int(sel.sum())} frames with both tracks ok")
    t, Rl, fl = loc["t"][sel], loc["R"][sel], loc["f"][sel]
    Rp, fp = pa["R"][sel], pa["f"][sel]
    Q, th = chordal_mean(np.einsum("nji,njk->nik", Rp, Rl))
    out = {"frames": int(sel.sum()), "frames_loc_ok": int(loc["ok"].sum()), "Q": Q.tolist(),
           "residual_deg": {"median": _f(np.median(th), 4), "p95": _f(np.percentile(th, 95), 4),
                            "p99": _f(np.percentile(th, 99), 4), "max": _f(th.max(), 4)}}
    # the cut rule: per half of the cut phase (of each loc frame's nearest PTZ sample) the time shift that minimises the
    # residual after that half's own Q (a Q shared with a wrongly dated half would absorb part of its error); a wrong
    # rule dates the samples of one half one source frame off (16.7 ms)
    i0, i1, a, _, _ = _bracket(ptz["t"], t)
    ph = ptz["phase"][np.where(a < 0.5, i0, i1)]
    halves = {}
    for name, hs in (("phase_below_half", ph < 0.5), ("phase_above_half", ph >= 0.5)):
        if hs.sum() < 10:
            halves[name] = {"frames": int(hs.sum())}
            continue

        def resid(dt_s: float, hs=hs) -> float:
            q = ptz_at(ptz, t[hs], dt_s)["R"]
            return float(np.median(chordal_mean(np.einsum("nji,njk->nik", q, Rl[hs]), iters=10)[1]))

        best, vmin, v0 = _scan_shift(resid)
        halves[name] = {"frames": int(hs.sum()), "residual_median_deg": _f(v0, 4), "best_time_shift_ms": _f(best, 2),
                        "median_deg_at_best_shift": _f(vmin, 4)}
    b_lo, b_hi = (halves[k].get("best_time_shift_ms") for k in ("phase_below_half", "phase_above_half"))
    out["cut_phase"] = {"rule": ptz["rule"], **halves,
                        "shift_difference_ms": _f(b_lo - b_hi, 2) if b_lo is not None and b_hi is not None else None,
                        "note": "phase = frac((S + n / cut_fps) x src_fps) of the cut frame; below 0.5 the 'fps' and "
                                "'nearest' rules pick source frames one apart: the right rule gives both halves the same "
                                "best shift (~0), a wrong one a 1000 / src_fps ms difference"}
    # Q-free relative rotations over REL_DT_S, and the residual time shift that best aligns them
    j = np.searchsorted(t, t + REL_DT_S)
    jj = np.clip(j, 0, len(t) - 1)
    pr = (j < len(t)) & (np.abs(t[jj] - t - REL_DT_S) <= 0.5 * loc["dt"])
    i, j = np.flatnonzero(pr), jj[pr]
    Rl_rel = np.einsum("nij,nkj->nik", Rl[j], Rl[i])

    def agree(dt_s: float) -> np.ndarray:
        q = ptz_at(ptz, loc["t"][sel], dt_s)["R"]
        return rot_deg(np.einsum("nij,nkj->nik", Rl_rel, np.einsum("nij,nkj->nik", q[j], q[i])))

    if len(i) >= 10:
        a0 = agree(0.0)
        best, vmin, _ = _scan_shift(lambda s: float(np.median(agree(s))))
        out["relative"] = {"dt_s": REL_DT_S, "pairs": int(len(i)), "median_deg": _f(np.median(a0), 4),
                           "p95_deg": _f(np.percentile(a0, 95), 4),
                           "motion_median_deg": _f(np.median(rot_deg(Rl_rel)), 4),
                           "best_time_shift_ms": _f(best, 2), "median_deg_at_best_shift": _f(vmin, 4),
                           "note": "Q-free: R_loc(t+dt) R_loc(t)^T vs the same of the PTZ; best_time_shift_ms = the PTZ "
                                   "clock correction that aligns them best (0 = the re-timed source frames are right; "
                                   "blind to a per-phase error: cut_phase)"}
    ratio = fl / fp
    gap = (t >= zoom_gap[0]) & (t <= zoom_gap[1])
    r_out = ratio[~gap]
    out["focal_ratio"] = {"median": _f(np.median(ratio), 5), "median_outside_gap": _f(np.median(r_out), 5) if len(r_out) else None,
                          "mad_rel_outside_gap": _f(_mad(r_out) / np.median(r_out), 5) if len(r_out) else None,
                          "mad_rel_inside_gap": _f(_mad(ratio[gap]) / np.median(ratio[gap]), 5) if gap.any() else None,
                          "note": "f_loc / f_ptz; its median is the absolute focal correction of the PTZ (MoGe) hypothesis"}
    out["focal_levels"] = {"loc": _levels(loc["t"][loc["ok"]], loc["f"][loc["ok"]], zoom_gap),
                           "ptz": _levels(ptz["t"], ptz["f"], zoom_gap)}
    if raw_C is not None and len(raw_C) >= 10:
        sd = 1.4826 * np.median(np.abs(raw_C - np.median(raw_C, 0)), 0)
        out["centre"] = {"frames": int(len(raw_C)), "robust_sd_sfm": [_f(x, 6) for x in sd],
                         "robust_sd_norm_sfm": _f(np.linalg.norm(sd), 6),
                         "robust_sd_m": _f(np.linalg.norm(sd) * s_world, 5) if s_world else None,
                         "what": "1.4826 x MAD per axis of the RAW per-frame loc centres (cameras/cache), norm; x world s"}
    else:
        out["centre"] = {"robust_sd_m": None, "why": "no raw loc cache (cameras/cache/loc_<cam>.npz)"}
    return out


def ptz_attach_fit(ptz: dict, img: dict, t_all: np.ndarray) -> dict:
    """Q, centre, focal scale and k1 of the PTZ track in the SfM frame from the camera's map images img {t, R (cam from
    SfM), C, f, k1} (see the module docstring); ok False with a reason when they are too few or do not span the pan."""
    pa = ptz_at(ptz, img["t"])
    use = pa["ok"]
    n = int(use.sum())
    out = {"images": int(len(img["t"])), "images_used": n}
    if n < ATTACH_MIN_IMAGES:
        return {**out, "ok": False, "why": f"only {n} map images inside the PTZ track (< {ATTACH_MIN_IMAGES})"}
    Q, th = chordal_mean(np.einsum("nji,njk->nik", pa["R"][use], img["R"][use]))
    Cj = img["C"][use]
    C = np.median(Cj, 0)
    sf = img["f"][use] / pa["f"][use]
    pan_img = pan_deg(pa["R"][use])
    pa_all = ptz_at(ptz, t_all)
    pan_all = pan_deg(pa_all["R"][pa_all["ok"]])
    rng = float(np.ptp(pan_all)) if len(pan_all) else 0.0
    cover = float(np.ptp(pan_img) / rng) if rng > 1e-6 else 1.0
    out.update({"Q": Q.tolist(), "C": C.tolist(), "s_f": float(np.median(sf)), "k1": float(np.median(img["k1"][use])),
                "rotation_residual_deg": {"median": _f(np.median(th), 4), "p95": _f(np.percentile(th, 95), 4),
                                          "max": _f(th.max(), 4)},
                "centre_robust_sd_sfm": [_f(x, 6) for x in 1.4826 * np.median(np.abs(Cj - C), 0)],
                "s_f_mad_rel": _f(_mad(sf) / np.median(sf), 5),
                "pan_cover": _f(cover, 3), "pan_range_deg": _f(rng, 2)})
    if cover < ATTACH_MIN_PAN_COVER:
        return {**out, "ok": False, "why": f"the map images span {cover:.0%} of the window's pan range "
                                           f"(< {ATTACH_MIN_PAN_COVER:.0%})"}
    return {**out, "ok": True}


def ptz_track_arrays(ptz: dict, fit: dict, t: np.ndarray, w: int, h: int) -> dict:
    """cameras/<cam>.npz arrays of the attached PTZ track at this take's frame times t (SfM frame)."""
    pa = ptz_at(ptz, t)
    Q, C = np.asarray(fit["Q"], float), np.asarray(fit["C"], float)
    R = np.einsum("nij,jk->nik", pa["R"], Q)
    f = fit["s_f"] * pa["f"]
    K = np.zeros((len(t), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2 - 0.5, h / 2 - 0.5, 1.0  # cameras.pipeline._K convention
    ok = pa["ok"] & (pa["rms"] <= PTZ_OK_RMS) & (pa["n_inl"] >= PTZ_OK_INL)
    F = len(t)
    return {"t": np.asarray(t, float), "K": K, "R": R, "tvec": -np.einsum("nij,j->ni", R, C),
            "dist": np.full(F, float(fit["k1"])), "ok": ok, "reproj_px": np.where(ok, pa["rms"], np.nan),
            "n_inliers": np.where(ok, pa["n_inl"], 0).astype(np.int32), "attached": np.bool_(False),
            "axis_sd": np.full(F, np.nan, np.float32), "recentred": np.zeros(F, bool),
            "camera_class": np.str_("tripod_zoom"), "type_model": np.str_("fixed_centre"), "type_held": np.zeros(F, bool),
            "track_source": np.str_("ptz")}


def _frame_times(work: Path, cam: str) -> dict[str, float]:
    """COLMAP image name -> this take's time, for the camera's window frames and keyframes."""
    out = {}
    fj = work / "frames" / cam / "frames.json"
    if fj.exists():
        out.update({f"frames/{cam}/{f['file']}": float(f["t"]) for f in _json(fj)["frames"]})
    kj = work / "keyframes" / cam / "keyframes.json"
    if kj.exists():
        out.update({f"keyframes/{cam}/{f['file']}": float(f["t"]) for f in _json(kj)})
    return out


def _cam_from_world(im):
    T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    return np.asarray(T.rotation.matrix(), float), np.asarray(T.translation, float)


def map_images_of(work: Path, cam: str, model=None, max_obs: int = 300) -> dict:
    """The camera's registered images in cameras/sfm: {name, t, R, tvec, C, f, k1, pp, obs [(xy OpenCV px, xyz)]}."""
    import pycolmap
    rec = model or pycolmap.Reconstruction(str(_need(work / "cameras" / "sfm")))
    times = _frame_times(work, cam)
    rows = []
    rng = np.random.default_rng(0)
    for im in rec.images.values():
        if not im.has_pose or im.name.split("/")[1] != cam or im.name not in times:
            continue
        R, tv = _cam_from_world(im)
        c = rec.cameras[im.camera_id]
        p = np.asarray(c.params, float)
        xy, xyz = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                xy.append(np.asarray(p2.xy, float) - 0.5)  # COLMAP pixel corner origin -> OpenCV centres
                xyz.append(rec.points3D[p2.point3D_id].xyz)
        if len(xy) > max_obs:
            k = rng.choice(len(xy), max_obs, replace=False)
            xy, xyz = [xy[i] for i in k], [xyz[i] for i in k]
        rows.append({"name": im.name, "t": times[im.name], "R": R, "tvec": tv, "C": -R.T @ tv, "f": float(p[0]),
                     "pp": p[1:3] - 0.5, "k1": float(p[3]) if c.model.name == "SIMPLE_RADIAL" else 0.0,
                     "xy": np.asarray(xy).reshape(-1, 2), "xyz": np.asarray(xyz, float).reshape(-1, 3)})
    rows.sort(key=lambda r: r["t"])
    return {"name": [r["name"] for r in rows], "t": np.array([r["t"] for r in rows], float),
            "R": np.array([r["R"] for r in rows]).reshape(-1, 3, 3), "tvec": np.array([r["tvec"] for r in rows]).reshape(-1, 3),
            "C": np.array([r["C"] for r in rows]).reshape(-1, 3), "f": np.array([r["f"] for r in rows], float),
            "k1": np.array([r["k1"] for r in rows], float), "pp": np.array([r["pp"] for r in rows]).reshape(-1, 2),
            "xy": [r["xy"] for r in rows], "xyz": [r["xyz"] for r in rows]}


def map_points_px(img: dict, arrays: dict, t_all: np.ndarray) -> dict:
    """Static map points of the camera's map images reprojected with the attached track (nearest frame time) vs with
    the map's own camera of that image: median / p90 px."""
    e_new, e_map = [], []
    for k, (xy, xyz) in enumerate(zip(img["xy"], img["xyz"])):
        if not len(xy):
            continue
        i = int(np.argmin(np.abs(t_all - img["t"][k])))
        if abs(t_all[i] - img["t"][k]) > 1e-3 or not arrays["ok"][i]:
            continue
        n = len(xyz)
        b = lambda x: np.broadcast_to(x, (n,) + np.shape(x))  # noqa: E731
        uv, z = reproj(xyz, b(arrays["R"][i]), b(arrays["tvec"][i]), b(arrays["K"][i, 0, 0]), b(arrays["K"][i, :2, 2]),
                       b(arrays["dist"][i]))
        e_new.append(np.where(z > 0, np.linalg.norm(uv - xy, axis=1), np.inf))
        uv, z = reproj(xyz, b(img["R"][k]), b(img["tvec"][k]), b(img["f"][k]), b(img["pp"][k]), b(img["k1"][k]))
        e_map.append(np.where(z > 0, np.linalg.norm(uv - xy, axis=1), np.inf))
    if not e_new:
        return {"images": 0}
    a, b_ = np.concatenate(e_new), np.concatenate(e_map)
    return {"images": len(e_new), "points": int(len(a)), "attached_median_px": _f(np.median(a), 3),
            "attached_p90_px": _f(np.percentile(a, 90), 3), "map_median_px": _f(np.median(b_), 3),
            "map_p90_px": _f(np.percentile(b_, 90), 3)}


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def adopt_track(work: Path, cam: str, src: Path, why: str) -> dict:
    """Install src (cameras/ptz/<cam>.npz) as cameras/<cam>.npz. The loc track moves to cameras/orig/<cam>_loc.npz: the
    adoption marker cameras.pipeline.adopted_track honours (stage_smooth and loc-join then keep cameras/<cam>.npz, and
    the loc cache stays for the world / qa stages). A repeated adoption only replaces the installed copy. Recorded in
    cameras/adopted.json."""
    from .pipeline import adopted_track
    cdir = work / "cameras"
    cur, orig = cdir / f"{cam}.npz", cdir / "orig" / f"{cam}_loc.npz"
    first = not orig.exists()
    if first:
        _need(cur)
        orig.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cur), str(orig))
    shutil.copyfile(src, cur)
    if adopted_track(work, cam) != orig:  # the guard that keeps every later cameras stage off the adopted file
        raise SystemExit(f"[twoview] adopt: cameras.pipeline.adopted_track does not see {orig}")
    rec_p = cdir / "adopted.json"
    rec = _json(rec_p) if rec_p.exists() else {}
    rec[cam] = {"track": f"cameras/{cam}.npz", "from": str(Path(src).relative_to(work)), "sha1": _sha1(cur),
                "loc_track": f"cameras/orig/{cam}_loc.npz", "first_adoption": first,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "why": why,
                "undo": f"move cameras/orig/{cam}_loc.npz back to cameras/{cam}.npz",
                "note": "cameras.pipeline.adopted_track keeps cameras/<cam>.npz while cameras/orig/<cam>_loc.npz exists: "
                        "stage_smooth / loc-join neither rewrite it from cameras/cache/loc_<cam>.npz nor delete it"}
    _write_json(rec_p, rec)
    return rec[cam]


def ptz_track_path(work: Path, cam: str) -> Path:
    """The attached PTZ track: cameras/ptz/<cam>.npz, outside the cameras/*.npz that readers take for cameras."""
    return work / "cameras" / "ptz" / f"{cam}.npz"


_CLOCK = {"fps": "t = (ceil((shift + (n + 0.5) / cut_fps) x src_fps) - 1) / src_fps: the source frame FFmpeg's fps filter "
                 "(round=near) puts in cut frame n",
          "nearest": "t = round((shift + n / cut_fps) x src_fps) / src_fps: the nearest source frame (comparison)",
          "nominal": "t = shift + n / cut_fps (nominal, comparison)"}


def cmd_ptz(work: Path, cam: str, ptz_work: Path, shift: float, mode: str, adopt: bool = False, cut_fps: float = 30.0,
            src_fps: float = SRC_FPS, rule: str = "fps", zoom_gap: tuple[float, float] = ZOOM_GAP,
            track: Path | None = None, log=print) -> dict:
    """ptz --mode check | attach [--adopt]: each updates its own section of cameras/qa/ptz_check.json."""
    ptz = load_ptz(ptz_work, shift, cut_fps, src_fps, rule, track)
    qa_p = work / "cameras" / "qa" / "ptz_check.json"
    doc = _json(qa_p) if qa_p.exists() else {}
    doc["ptz"] = {k: ptz[k] for k in ("path", "n_total", "n_ok", "shift", "cut_fps", "src_fps", "rule", "lens", "size")}
    doc["ptz"]["clock"] = _CLOCK[rule]
    sim3_p = work / "cameras" / "world_from_sfm.json"
    s_world = float(_json(sim3_p)["s"]) if sim3_p.exists() else None
    if mode == "check":
        orig = work / "cameras" / "orig" / f"{cam}_loc.npz"  # after an adoption cameras/<cam>.npz IS the PTZ track
        loc = load_track(work, cam, orig if orig.exists() else None, world=False)
        raw_C = None
        cache = work / "cameras" / "cache" / f"loc_{cam}.npz"
        if cache.exists():
            with np.load(cache) as z:  # only these members: the correspondences stay on disk
                R, tv, ok = np.asarray(z["R"], float), np.asarray(z["tvec"], float), np.asarray(z["ok"], bool)
            ok &= np.isfinite(R).all((1, 2)) & np.isfinite(tv).all(1)
            raw_C = -np.einsum("fji,fj->fi", R[ok], tv[ok])
        res = ptz_compare(loc, ptz, raw_C, s_world, zoom_gap)
        res.update({"loc_track": loc["path"], "raw_cache": str(cache) if raw_C is not None else None,
                    "s_world": s_world, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
        doc["check"] = res
        log(f"[twoview] ptz check: residual after Q {res['residual_deg']}, relative {res.get('relative', {}).get('median_deg')} "
            f"deg (best shift {res.get('relative', {}).get('best_time_shift_ms')} ms), cut rule {rule}: best shift per "
            f"phase half {[res['cut_phase'][k].get('best_time_shift_ms') for k in ('phase_below_half', 'phase_above_half')]}"
            f" ms, f_loc/f_ptz {res['focal_ratio']['median']}, centre SD {res['centre'].get('robust_sd_m')} m")
    elif mode == "attach":
        meta = _json(_need(work / "frames" / cam / "frames.json"))
        t_all = np.array([f["t"] for f in meta["frames"]], float)
        img = map_images_of(work, cam)
        fit = ptz_attach_fit(ptz, img, t_all)
        doc["attach"] = {**fit, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if not fit["ok"]:
            _write_json(qa_p, doc)
            raise SystemExit(f"[twoview] ptz attach refused: {fit['why']}")
        arr = ptz_track_arrays(ptz, fit, t_all, int(meta["width"]), int(meta["height"]))
        out = ptz_track_path(work, cam)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arr)
        doc["attach"].update({"out": str(out), "frames": int(len(t_all)), "frames_ok": int(arr["ok"].sum()),
                              "map_points": map_points_px(img, arr, t_all),
                              "centre_robust_sd_m": (_f(np.linalg.norm(fit["centre_robust_sd_sfm"]) * s_world, 5)
                                                     if s_world else None)})
        log(f"[twoview] ptz attach -> {out}: {fit['images_used']} map images, s_f {fit['s_f']:.4f}, k1 {fit['k1']:.4f}, "
            f"rotation residual {fit['rotation_residual_deg']}, map points {doc['attach']['map_points']}")
        if adopt:
            doc["adopted"] = adopt_track(work, cam, out, "twoview ptz --mode attach --adopt")
            log(f"[twoview] adopted: cameras/{cam}.npz = the PTZ track; the loc track -> cameras/orig/{cam}_loc.npz")
    else:
        raise SystemExit(f"[twoview] ptz: unknown mode {mode!r}")
    _write_json(qa_p, doc)
    return doc


# ------------------------------------------------------------------------------------------------ Atlas


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity dst ~ s R src + t (least squares, Umeyama 1991)."""
    ms, md_ = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md_
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    E = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    s = float(np.trace(np.diag(D) @ E) / max((a ** 2).sum() / len(src), 1e-18))
    return s, R, md_ - s * R @ ms


def ransac_sim3(src: np.ndarray, dst: np.ndarray, thr: float, iters: int = 2000,
                seed: int = 0) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC over minimal 3-point Umeyama fits, then a least-squares refit on the inliers (twice)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        k = rng.choice(len(src), 3, replace=False)
        s, R, t = umeyama(src[k], dst[k])
        if not np.isfinite(s) or s <= 0:
            continue
        inl = np.linalg.norm(dst - (s * src @ R.T + t), axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    inl = best if best is not None and best.sum() >= 3 else np.ones(len(src), bool)
    for _ in range(2):
        s, R, t = umeyama(src[inl], dst[inl])
        new = np.linalg.norm(dst - (s * src @ R.T + t), axis=1) < thr
        if new.sum() < 3:
            break
        inl = new
    return s, R, t, inl


def atlas_compare(Ra: np.ndarray, Ca: np.ndarray, fa: np.ndarray, Rs: np.ndarray, Cs: np.ndarray, fs: np.ndarray,
                  inlier_rel: float = 0.05) -> dict:
    """Atlas cameras (world->camera R, centres, fx; Atlas frame) vs the same keyframes in the SfM frame: Sim3 on the
    centres (inliers within inlier_rel x the SfM path extent), centre residual / extent, rotation residual, fx ratio."""
    ext = float(max(np.linalg.norm(Cs[:, None] - Cs[None], axis=-1).max(), 1e-12))
    s, R, t, inl = ransac_sim3(Ca, Cs, inlier_rel * ext)
    res = np.linalg.norm(Cs - (s * Ca @ R.T + t), axis=1) / ext
    rot = rot_deg(np.einsum("nij,kj,nlk->nil", Ra, R, Rs))  # (R_a R^T) R_sfm^T
    fr = fa / fs
    return {"n": int(len(Ca)), "sim3": {"s": s, "R": R.tolist(), "t": t.tolist(), "inliers": int(inl.sum()),
                                        "inlier_threshold_rel": inlier_rel},
            "path_extent_sfm": ext, "centre_residual_rel": {"median": _f(np.median(res), 4), "p90": _f(np.percentile(res, 90), 4),
                                                            "median_inliers": _f(np.median(res[inl]), 4)},
            "rotation_residual_deg": {"median": _f(np.median(rot), 3), "p90": _f(np.percentile(rot, 90), 3)},
            "fx_ratio": {"median": _f(np.median(fr), 4), "p10": _f(np.percentile(fr, 10), 4), "p90": _f(np.percentile(fr, 90), 4),
                         "prior_usable": bool(0.9 <= np.median(fr) <= 1.1),
                         "note": "fx_atlas / f_sfm; outside [0.9, 1.1] Atlas is a verification only, never a focal or "
                                 "scale prior"},
            "inlier_mask": inl, "centre_residual": res, "rotation_residual": rot}


def cmd_atlas(work: Path, cam: str, atlas_json: Path, log=print) -> dict:
    """atlas: see the module docstring. Depth maps (view_<cam>_<kf>.npz) and people masks are read one keyframe at a time."""
    import cv2
    import pycolmap
    from ..mono4d import atlas as A
    doc = _json(_need(atlas_json))
    kfs = _json(_need(work / "keyframes" / cam / "keyframes.json"))
    rec = pycolmap.Reconstruction(str(_need(work / "cameras" / "sfm")))
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    sim3_p = work / "cameras" / "world_from_sfm.json"
    world = _json(sim3_p) if sim3_p.exists() else None
    rows, unreg = [], []
    for k, (row, acam) in enumerate(zip(doc["rows"], doc["cameras"])):
        kf = int(row.get("kf", k))
        f = kfs[kf]
        if abs(float(f["t"]) - float(row.get("t", f["t"]))) > 1e-3:
            raise SystemExit(f"[twoview] atlas row {k}: t {row.get('t')} != keyframes.json[{kf}] t {f['t']}")
        im = by_name.get(f"keyframes/{cam}/{f['file']}")
        if im is None:
            unreg.append(kf)
            continue
        Ka, Ra, ta, Wa, Ha = A.from_atlas_camera(acam)
        R, tv = _cam_from_world(im)
        p = np.asarray(rec.cameras[im.camera_id].params, float)
        rows.append({"kf": kf, "file": f["file"], "camera": acam, "Ra": Ra, "Ca": -Ra.T @ ta, "fa": float(Ka[0, 0]),
                     "R": R, "tvec": tv, "C": -R.T @ tv, "f": float(p[0]), "pp": p[1:3] - 0.5, "size": (Wa, Ha)})
    if len(rows) < 5:
        raise SystemExit(f"[twoview] atlas: only {len(rows)} Atlas keyframes are registered in cameras/sfm")
    st = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    cmp_ = atlas_compare(st("Ra"), st("Ca"), st("fa"), st("R"), st("C"), st("f"))
    inl, cres, rres = cmp_.pop("inlier_mask"), cmp_.pop("centre_residual"), cmp_.pop("rotation_residual")
    pts = np.array([p.xyz for p in rec.points3D.values()], float).reshape(-1, 3)
    scales = []
    meta = _json(_need(work / "frames" / cam / "frames.json"))
    W_, H_ = int(meta["width"]), int(meta["height"])
    for r in rows:
        vz = Path(atlas_json).parent / f"view_{cam}_{r['kf']:02d}.npz"
        e = {"kf": r["kf"]}
        if vz.exists():
            with np.load(vz) as z:
                view = A.PosedView(camera=r["camera"], image={}, z=np.asarray(z["z"], np.float32),
                                   conf=np.asarray(z["conf"], np.float32) if "conf" in z.files else None)
            keep = None
            mp = work / "people_masks_kf" / cam / f"{Path(r['file']).stem}.png"
            if mp.exists():
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m.shape != view.z.shape:
                    m = cv2.resize(m, view.z.shape[::-1], interpolation=cv2.INTER_NEAREST)
                keep = m < 128
            K = np.array([[r["f"], 0, r["pp"][0]], [0, r["f"], r["pp"][1]], [0, 0, 1.0]])
            try:
                wa = A.world_from_atlas(view, K, r["R"], r["tvec"], W_, H_, pts, keep)
                q1, q3 = wa["ratio_iqr"]
                e.update({"s_sfm_per_atlas": wa["s"], "n_points": wa["n_points"], "ratio_iqr_rel": (q3 - q1) / wa["s"],
                          "focal_ratio": wa["focal_ratio"]})
            except ValueError as ex:
                e["why"] = str(ex)
            del view
        else:
            e["why"] = f"no {vz.name}"
        scales.append(e)
    s_ok = np.array([e["s_sfm_per_atlas"] for e in scales if "s_sfm_per_atlas" in e], float)
    depth = {"keyframes": int(len(s_ok))}
    if len(s_ok):
        q1, q3 = np.percentile(s_ok, [25, 75])
        sw = float(world["s"]) if world else None
        depth.update({"s_sfm_per_atlas_median": _f(np.median(s_ok), 5), "iqr_rel": _f((q3 - q1) / np.median(s_ok), 4),
                      "m_per_atlas_unit_median": _f(np.median(s_ok) * sw, 5) if sw else None,
                      "within_frame_ratio_iqr_rel_median": _f(np.median([e["ratio_iqr_rel"] for e in scales
                                                                         if "ratio_iqr_rel" in e]), 4),
                      "note": "Atlas depth is NOT metric: the scale across keyframes only tests for drift along the orbit; "
                              "the metric scale always comes from the stature chain"})
    out = {"cam": cam, "atlas": str(atlas_json), "n_atlas": len(doc["cameras"]), "keyframes_matched": len(rows),
           "keyframes_unregistered": unreg, **cmp_, "depth_scale": depth,
           "per_keyframe": [{"kf": r["kf"], "inlier": bool(i), "centre_residual_rel": _f(c, 4), "rotation_deg": _f(a, 3),
                             **{k: (_f(v, 5) if isinstance(v, float) else v) for k, v in e.items() if k != "kf"}}
                            for r, i, c, a, e in zip(rows, inl, cres, rres, scales)]}
    sim = cmp_["sim3"]
    Ca_s = sim["s"] * st("Ca") @ np.asarray(sim["R"]).T + np.asarray(sim["t"])
    if world:
        hw_a, hw_s = W.apply(world, Ca_s)[:, 1], W.apply(world, st("C"))[:, 1]
        up = np.asarray(world["R"]) @ np.asarray(sim["R"]) @ np.array([0.0, 1.0, 0.0])
        out["heights_world_m"] = {"atlas": [_f(np.percentile(hw_a, q), 3) for q in (5, 50, 95)],
                                  "sfm": [_f(np.percentile(hw_s, q), 3) for q in (5, 50, 95)],
                                  "atlas_minus_sfm_median": _f(np.median(hw_a - hw_s), 4),
                                  "atlas_up_vs_world_y_deg": _f(np.degrees(np.arccos(np.clip(up[1] / np.linalg.norm(up), -1, 1))), 3)}
    p = _write_json(work / "cameras" / "qa" / "atlas_check.json", out)
    png = _plot_atlas(st("C"), Ca_s, inl, world, work / "review" / "atlas_check.png", out)
    log(f"[twoview] -> {p}, {png}: Sim3 inliers {sim['inliers']}/{len(rows)}, centre residual "
        f"{cmp_['centre_residual_rel']['median']} x extent, rotation {cmp_['rotation_residual_deg']} deg, fx ratio "
        f"{cmp_['fx_ratio']['median']}, depth scale IQR/median {depth.get('iqr_rel')}")
    return out


def _plot_atlas(Cs: np.ndarray, Ca: np.ndarray, inl: np.ndarray, world: dict | None, out_png: Path, doc: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if world:
        Cs, Ca = W.apply(world, Cs), W.apply(world, Ca)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Cs[:, 0], Cs[:, 2], "-o", color="#1f77b4", ms=3, lw=1, label="SfM keyframes")
    ax.plot(Ca[inl, 0], Ca[inl, 2], "o", color="#ff7f0e", ms=3, label="Atlas (Sim3), inliers")
    ax.plot(Ca[~inl, 0], Ca[~inl, 2], "x", color="#d62728", ms=5, label="Atlas outliers")
    for a, b in zip(Cs, Ca):
        ax.plot([a[0], b[0]], [a[2], b[2]], color="#999", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)" if world else "x (SfM)")
    ax.set_ylabel("z (m)" if world else "z (SfM)")
    ax.set_title(f"top view: {doc['sim3']['inliers']}/{doc['n']} inliers, rotation {doc['rotation_residual_deg']['median']} deg, "
                 f"fx ratio {doc['fx_ratio']['median']}", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------------------------------------ map gate


def _model_stats(work: Path) -> dict | None:
    """{cam: {registered, focal_px [min, median, max], colmap_cameras}} of cameras/sfm (+ "_doppelgangers": the
    cameras.plausible check of the model, "_reproj_px": its mean reprojection), or None without the model."""
    try:
        import pycolmap
    except ImportError:
        return None
    p = work / "cameras" / "sfm"
    if not (p / "images.bin").exists():
        return None
    rec = pycolmap.Reconstruction(str(p))
    out: dict = {}
    for im in rec.images.values():
        if im.has_pose:
            e = out.setdefault(im.name.split("/")[1], {"registered": 0, "f": [], "cams": set()})
            e["registered"] += 1
            e["f"].append(float(rec.cameras[im.camera_id].params[0]))
            e["cams"].add(im.camera_id)
    from .plausible import check_model
    dg = check_model(rec)
    return {**{c: {"registered": e["registered"], "focal_px": [_f(min(e["f"]), 2), _f(np.median(e["f"]), 2), _f(max(e["f"]), 2)],
                   "colmap_cameras": len(e["cams"])} for c, e in out.items()},
            "_doppelgangers": {"flagged": len(dg["flagged"]), "per_cam": dg["per_cam"], "kept_majority": dg["kept_majority"]},
            "_reproj_px": _f(rec.compute_mean_reprojection_error(), 4)}


def cmd_map_gate(work: Path, need: list[str], log=print) -> dict:
    """map-gate: see the module docstring. doc['exit'] = 3 when a needed camera has no registered main-model image."""
    info = _json(_need(work / "cameras" / "sfm_work" / "map.json"))
    total: dict[str, int] = {}
    for k in info.get("map_images", []):
        total[k.split("/")[0]] = total.get(k.split("/")[0], 0) + 1
    model = _model_stats(work)
    dg = model.pop("_doppelgangers", None) if model is not None else None
    rp_model = model.pop("_reproj_px", None) if model is not None else None  # the written model (after any clean)
    if model is not None:
        reg, reg_src = {c: v["registered"] for c, v in model.items()}, "cameras/sfm"
    else:
        reg, reg_src = dict((info.get("models") or [{}])[0].get("cams") or {}), "map.json models[0] (no cameras/sfm)"
    ver = info.get("verification") or {}
    per_cp = ver.get("per_campair") or {}
    groups = info.get("strong_link_groups") or []
    cams = sorted(set(total) | set(reg) | set(need))
    per = {c: {"registered": int(reg.get(c, 0)), "map_images": int(total.get(c, 0)),
               "share": _f(reg.get(c, 0) / total[c], 4) if total.get(c) else None,
               "intrinsics_mode": (info.get("intrinsics_mode") or {}).get(c),
               **({"focal_px": model[c]["focal_px"], "colmap_cameras": model[c]["colmap_cameras"]}
                  if model and c in model else {})} for c in cams}
    links = {}
    for i, a in enumerate(sorted(need)):
        for b in sorted(need)[i + 1:]:
            cp = f"{a}-{b}"
            e = per_cp.get(cp) or {}
            same = any(a in g and b in g for g in groups)
            links[cp] = {"pairs": e.get("pairs", 0), "max_inliers": e.get("max_inliers", 0),
                         "link_pairs": e.get("link_pairs"), "link_min_inliers": ver.get("link_min_inliers", 150),
                         "link_min_pairs": ver.get("link_min_pairs", 3),
                         "strong_link": bool(same),
                         "source": "verification.per_campair.link_pairs" if "link_pairs" in e
                         else "strong_link_groups (yes/no: map.json predates link_pairs)"}
    missing = [c for c in need if per[c]["registered"] == 0]
    doc = {"need": need, "cams": per, "registered_from": reg_src, "main_cams": info.get("main_cams"),
           "strong_link_groups": groups, "links": links,
           "mean_reproj_px": rp_model if rp_model is not None else _f((info.get("refine") or {}).get("mean_reproj_after"), 4),
           "intrinsics_mode": info.get("intrinsics_mode"), "missing_from_main_model": missing,
           "doppelgangers": dg, "doppelgangers_removed": len((info.get("doppelgangers") or {}).get("flagged") or {}),
           "exit": 3 if missing else 0, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    p = _write_json(work / "cameras" / "qa" / "map_gate.json", doc)
    log(f"[twoview] -> {p}: " + ", ".join(f"{c} {v['registered']}/{v['map_images']}" for c, v in per.items())
        + f"; links {[(k, v['link_pairs'], v['strong_link']) for k, v in links.items()]}"
        + (f"; NOT in the main model: {missing}" if missing else "")
        + (f"; {dg['flagged']} map image(s) outside their camera's band {dg['per_cam']} (cameras.plausible clean)"
           if dg and dg["flagged"] else ""))
    return doc


# ------------------------------------------------------------------------------------------------ gates

_RANK = {"pass": 0, "info": 0, "soft": 1, "fail": 2, "missing": 3}


def _chk(value, threshold, ok: bool | None, soft: bool = False) -> dict:
    """One check: status pass | soft | fail | missing (value None and ok None)."""
    if ok is None:
        st = "missing"
    else:
        st = "pass" if ok else ("soft" if soft else "fail")
    return {"value": value, "threshold": threshold, "pass": None if ok is None else bool(ok), "status": st}


def _info(value, note: str = "") -> dict:
    return {"value": value, "status": "info", **({"note": note} if note else {})}


def _gate(checks: dict, missing: list[str]) -> dict:
    st = max((c["status"] for c in checks.values()), key=lambda s: _RANK[s], default="missing")
    st = "missing" if missing else ("pass" if st == "info" else st)
    return {"status": st, "pass": st == "pass", "value": {k: c.get("value") for k, c in checks.items()},
            "threshold": {k: c.get("threshold") for k, c in checks.items() if "threshold" in c},
            "checks": checks, "missing_inputs": missing}


def _in(x, lo, hi):
    return None if x is None else bool(lo <= x <= hi)


def _le(x, thr):
    return None if x is None else bool(x <= thr)


def _ge(x, thr):
    return None if x is None else bool(x >= thr)


def _load(p: Path, missing: list[str]):
    if not p.exists():
        missing.append(str(p))
        return None
    return _json(p)


def _longest_gap_s(t: np.ndarray, ok: np.ndarray) -> float:
    best, i, n = 0.0, 0, len(t)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0
    while i < n:
        if ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not ok[j + 1]:
            j += 1
        best = max(best, float(t[j] - t[i] + dt))
        i = j + 1
    return best


def _camera_roles(cfg: dict) -> tuple[str, str]:
    """(tripod, moving camera) from takes/<take>.toml [cameras."NN"] mount; default ('02', '01') as in lucas_luiza."""
    cams = {k: v for k, v in (cfg.get("cameras") or {}).items() if isinstance(v, dict) and k.isdigit()}
    tri = sorted(c for c, v in cams.items() if v.get("mount") == "tripod")
    mov = sorted(c for c in cams if c not in tri)
    return (tri[0] if tri else "02", mov[0] if mov else "01")


def gates_eval(work: Path, cfg: dict, tripod: str, moving: str) -> dict:
    """G1-G8 (runbook step 9) from the files the cameras stages and this module write."""
    q = work / "cameras" / "qa"
    window = _window(cfg)
    out, miss_all = {}, []

    def need(rel: str, m: list):
        return _load(work / rel, m)

    # G1 map
    m: list = []
    mg, rep = need("cameras/qa/map_gate.json", m), need("cameras/report.json", m)
    ch = {}
    if mg:
        cams = mg["cams"]
        ch["both_in_main_model"] = _chk({c: cams.get(c, {}).get("registered") for c in (moving, tripod)}, "> 0 (hard)",
                                        all((cams.get(c, {}).get("registered") or 0) > 0 for c in (moving, tripod)))
        for c in (moving, tripod):
            sh = cams.get(c, {}).get("share")
            ch[f"registered_share_{c}"] = _chk(sh, ">= 0.8 (0.5-0.8 soft)", None if sh is None else sh >= 0.8,
                                               soft=sh is not None and sh >= 0.5)
        lk = (mg.get("links") or {}).get("-".join(sorted((moving, tripod)))) or {}
        if lk.get("link_pairs") is not None:
            ch["link"] = _chk({"link_pairs": lk["link_pairs"], "max_inliers": lk.get("max_inliers")},
                              f">= {lk.get('link_min_pairs', 3)} pairs with >= {lk.get('link_min_inliers', 150)} verified inliers",
                              lk["link_pairs"] >= lk.get("link_min_pairs", 3))
        else:
            ch["link"] = _chk({"strong_link": lk.get("strong_link")}, "same strong-link group (yes/no)", lk.get("strong_link"))
        rp = mg.get("mean_reproj_px")
        if rp is None and rep:
            rp = rep.get("map_mean_reproj_px")
        ch["map_mean_reproj_px"] = _chk(_f(rp, 4), "< 1.5", None if rp is None else rp < 1.5)
        fm = (cams.get(moving, {}).get("focal_px") or [None, None, None])[1]
        ch[f"focal_{moving}_px"] = _chk(fm, "700-3000 (report its value)", _in(fm, 700, 3000))
        ch[f"intrinsics_mode_{moving}"] = _info(cams.get(moving, {}).get("intrinsics_mode"))
        dg = mg.get("doppelgangers")
        ch["no_doppelgangers"] = _chk(None if dg is None else {"flagged": dg["flagged"], "per_cam": dg["per_cam"],
                                                               "removed_before": mg.get("doppelgangers_removed", 0)},
                                      "0 registered images outside their camera's band (cameras.plausible)",
                                      None if dg is None else dg["flagged"] == 0)
    out["G1_map"] = _gate(ch, m)
    miss_all += m
    # G2 loc
    m, ch = [], {}
    for c, (ok_min, gap_max) in ((tripod, (0.90, 1.0)), (moving, (0.85, 2.0))):
        orig = work / "cameras" / "orig" / f"{c}_loc.npz"
        p = orig if orig.exists() else work / "cameras" / f"{c}.npz"
        if not p.exists():
            m.append(str(p))
            continue
        with np.load(p) as z:
            t, ok = np.asarray(z["t"], float), np.asarray(z["ok"], bool)
            rp = np.asarray(z["reproj_px"], float) if "reproj_px" in z.files else np.full(len(t), np.nan)
        sel = (t >= window[0]) & (t <= window[1]) if window else np.ones(len(t), bool)
        t, ok, rp = t[sel], ok[sel], rp[sel]
        r = rp[ok & np.isfinite(rp)]
        ch[f"ok_share_{c}"] = _chk(_f(ok.mean(), 4), f">= {ok_min}", bool(ok.mean() >= ok_min))
        ch[f"reproj_median_{c}"] = _chk(_f(np.median(r), 3) if len(r) else None, "<= 1.5 px",
                                        _le(float(np.median(r)), 1.5) if len(r) else None)
        ch[f"reproj_p90_{c}"] = _chk(_f(np.percentile(r, 90), 3) if len(r) else None, "<= 3.0 px",
                                     _le(float(np.percentile(r, 90)), 3.0) if len(r) else None)
        g = _longest_gap_s(t, ok)
        ch[f"longest_gap_s_{c}"] = _chk(_f(g, 3), f"<= {gap_max} s", g <= gap_max)
        if orig.exists():
            ch[f"track_{c}"] = _info(str(p.relative_to(work)), "the loc track (cameras/<cam>.npz is an adopted PTZ track)")
    out["G2_loc"] = _gate(ch, m)
    miss_all += m
    # G3 types / tripod
    m, ch = [], {}
    rep = need("cameras/report.json", m)
    pz = need("cameras/qa/ptz_check.json", m)
    pc = (pz or {}).get("check")
    if pz is not None and pc is None:
        m.append(str(q / "ptz_check.json") + " [check]")
    if rep:
        rc = rep.get("cams") or {}
        for c, want in ((moving, "fixed_focal"), (tripod, "fixed_centre")):
            ct = (rc.get(c) or {}).get("camera_type") or {}
            if ct:
                okc = ct.get("model") == want and (ct.get("check") or {}).get("passed", True) is not False
                ch[f"camera_type_{c}"] = _chk({"model": ct.get("model"), "why": ct.get("why") or (ct.get("check") or {}).get("why"),
                                               "source": "report.json"}, f"{want} applied", okc)
                continue
            # report.json's entry is written by the qa step's probe B, which swallows its errors: the applied model is
            # also the track's type_model (stage_smooth; the loc track's when the PTZ track is adopted)
            orig = work / "cameras" / "orig" / f"{c}_loc.npz"
            tp = orig if orig.exists() else work / "cameras" / f"{c}.npz"
            if not tp.exists():
                m.append(f"report.json cams.{c}.camera_type / {tp}")
                continue
            with np.load(tp) as z:
                model = str(z["type_model"]) if "type_model" in z.files else None
            ch[f"camera_type_{c}"] = _chk({"model": model, "source": f"{tp.relative_to(work)} type_model",
                                           "why": "report.json has no camera_type (probe B failed?): check.passed unknown"},
                                          f"{want} applied", model == want)
        ev = ((rc.get(tripod) or {}).get("focal") or {}).get("zoom_events")
        if ev is not None:
            gap = ZOOM_GAP
            inside = [e for e in ev if e["t1"] >= gap[0] and e["t0"] <= gap[1]]
            other = [e for e in ev if e not in inside and abs(e["change_pct"]) > 3.0]
            ch["focal_steps_outside_gap"] = _chk(other, "no focal step > 3 % outside 115-130 s", not other)
            ch["focal_steps_115_130"] = _info(inside)
        else:
            ch["focal_steps_outside_gap"] = _chk(None, "report.json cams.<tripod>.focal.zoom_events", None)
    if pc:
        ch["tripod_centre_sd_m"] = _chk(pc.get("centre", {}).get("robust_sd_m"), "<= 0.03",
                                        _le(pc.get("centre", {}).get("robust_sd_m"), 0.03))
        zr = ((pc.get("focal_levels") or {}).get("loc") or {}).get("ratio")
        ch["zoom_ratio"] = _chk(zr, "median f(t > 130 s) / median f(t < 115 s) in [1.12, 1.20]", _in(zr, 1.12, 1.20))
    out["G3_types_tripod"] = _gate(ch, m)
    miss_all += m
    # G4 tripod vs PTZ
    m, ch = [], {}
    if pc is None:
        m.append(str(q / "ptz_check.json") + " [check]")
    else:
        rd, rel, fr = pc.get("residual_deg") or {}, pc.get("relative") or {}, pc.get("focal_ratio") or {}
        ch["residual_median_deg"] = _chk(rd.get("median"), "<= 0.10", _le(rd.get("median"), 0.10))
        ch["residual_p95_deg"] = _chk(rd.get("p95"), "<= 0.30", _le(rd.get("p95"), 0.30))
        ch["relative_median_deg"] = _chk(rel.get("median_deg"), "<= 0.05 (1/6 s, Q-free)", _le(rel.get("median_deg"), 0.05))
        ch["focal_ratio_mad_rel"] = _chk(fr.get("mad_rel_outside_gap"), "<= 0.015 outside 115-130 s",
                                         _le(fr.get("mad_rel_outside_gap"), 0.015))
        ch["focal_ratio_median"] = _info(fr.get("median"), "the absolute focal correction of the PTZ / MoGe hypothesis")
        ch["best_time_shift_ms"] = _info(rel.get("best_time_shift_ms"))
        cp = pc.get("cut_phase") or {}
        ch["cut_phase_shift_ms"] = _info({"rule": cp.get("rule"), **{k: (cp.get(k) or {}).get("best_time_shift_ms")
                                                                      for k in ("phase_below_half", "phase_above_half")}},
                                         "best PTZ time shift per half of the cut phase: equal (~0) = the cut rule is right, "
                                         "one source frame (16.7 ms) apart = it is not")
    out["G4_tripod_vs_ptz"] = _gate(ch, m)
    miss_all += m
    # G5 Atlas
    m, ch = [], {}
    at = need("cameras/qa/atlas_check.json", m)
    if at:
        n_m = at.get("keyframes_matched") or at.get("n") or 0
        n = at.get("n_atlas") or n_m  # inliers out of the Atlas cameras: an unregistered keyframe counts as an outlier
        need_inl = int(np.ceil(70 / 99 * n))
        ch["sim3_inliers"] = _chk(f"{at['sim3']['inliers']}/{n}", f">= {need_inl} (70 / 99 of the Atlas cameras"
                                  + ("" if at.get("n_atlas") else ": n_atlas not recorded, of the registered keyframes") + ")",
                                  at["sim3"]["inliers"] >= need_inl)
        ch["keyframes_registered"] = _info(f"{n_m}/{n}", f"unregistered keyframes {at.get('keyframes_unregistered')}"
                                           if at.get("keyframes_unregistered") else "")
        cr, rr = at.get("centre_residual_rel") or {}, at.get("rotation_residual_deg") or {}
        ch["centre_residual_rel"] = _chk(cr.get("median"), "median <= 0.05 x path extent", _le(cr.get("median"), 0.05))
        ch["rotation_median_deg"] = _chk(rr.get("median"), "<= 2", _le(rr.get("median"), 2.0))
        ch["rotation_p90_deg"] = _chk(rr.get("p90"), "<= 5", _le(rr.get("p90"), 5.0))
        ds = at.get("depth_scale") or {}
        ch["depth_scale_iqr_rel"] = _chk(ds.get("iqr_rel"), "<= 0.10 (no scale drift along the orbit)",
                                         _le(ds.get("iqr_rel"), 0.10))
        fx = at.get("fx_ratio") or {}
        ch["fx_ratio"] = _info(fx.get("median"), "outside [0.9, 1.1]: Atlas is a verification only, never a focal or scale "
                                                 "prior" + ("" if fx.get("prior_usable") else " (outside)"))
        ch["depth_metric"] = _info(False, "Atlas depth is not metric: the metric scale comes from the stature chain")
    out["G5_atlas"] = _gate(ch, m)
    miss_all += m
    # G6 world
    m, ch = [], {}
    sim3, rep = need("cameras/world_from_sfm.json", m), need("cameras/report.json", m)
    tvc = _load(q / "twoview_check.json", [])
    if sim3:
        d = sim3.get("diagnostics") or {}
        v = d.get("vertical") or {}
        mcd = v.get("max_cam_deg")
        gv = bool(v.get("adopted")) and (v.get("n_cams") or 0) >= 2 and mcd is not None and mcd <= 2.0
        ch["gravity"] = _chk({"source": sim3.get("gravity_source"), "max_cam_deg": v.get("max_cam_deg"), "n_cams": v.get("n_cams")},
                             "architecture vertical, both cameras within 2 deg (ankle-plane fallback: soft)", gv, soft=True)
        ch["x_axis"] = _chk(sim3.get("x_axis"), f"camera {tripod}", f"camera {tripod} " in str(sim3.get("x_axis")))
        roles = d.get("dancer_roles") or {}
        rv, rok = {}, True
        for c in (moving, tripod):  # dancer_keypoints reads roles.json's frames by position: one per people frame
            e = roles.get(c) or {}
            pp = work / "people" / f"{c}.npz"
            n_pp = None
            if pp.exists():
                with np.load(pp) as z:
                    n_pp = int(len(z["t"]))
            rv[c] = {"source": e.get("source"), "frames": e.get("frames"), "people_frames": n_pp}
            rok &= e.get("source") == "roles.json" and (n_pp is None or e.get("frames") == n_pp)
        ch["dancer_roles"] = _chk(rv, "roles.json for both cameras, one entry per people frame", rok)
        la = d.get("low_ankle_height_m") or {}
        dla = (la["median"] - la["expected_ankle_joint"]) if "median" in la and "expected_ankle_joint" in la else None
        ch["low_ankle_vs_expected_m"] = _chk(_f(dla, 4), "|median - expected ankle joint (sole incl.)| <= 0.02",
                                             None if dla is None else abs(dla) <= 0.02)
        if tvc:
            ch["stance_ankles_vs_expected_m"] = _info({r: (tvc.get("stance_ankles") or {}).get(r, {}).get("diff_m") for r in ROLES},
                                                      "twoview check (sync-corrected), per dancer")
        rc = d.get("room_metric_check") or {}
        e = [x.get("elevation_vs_world_y_deg") for x in rc.get("horizontal_edges") or []]
        wl = [x.get("tilt_vs_world_y_deg") for x in rc.get("walls") or []]
        if e or wl:
            ch["room_check"] = _chk({"edges_deg": e, "walls_deg": wl}, f"edges <= {ROOM_EDGE_DEG}, walls <= {ROOM_WALL_DEG} deg",
                                    all(x is not None and x <= ROOM_EDGE_DEG for x in e)
                                    and all(x is not None and x <= ROOM_WALL_DEG for x in wl))
        else:  # no edge and no wall measured: nothing passed
            ch["room_check"] = _chk(rc or None, "diagnostics.room_metric_check with edges or walls", None)
    if rep:
        rc = rep.get("cams") or {}
        for c, lo, hi in ((moving, 0.8, 2.0), (tripod, 1.2, 2.2)):
            hm = ((rc.get(c) or {}).get("height_m") or [None, None, None])[1]
            ch[f"height_m_{c}"] = _chk(hm, f"{lo}-{hi}", _in(hm, lo, hi))
        dist = ((tvc or {}).get("cam_to_couple_m") or {}).get(tripod)
        src = "twoview_check cam_to_couple_m"
        if dist is None:
            dist, src = (rc.get(tripod) or {}).get("dist_to_origin_m"), "report.json dist_to_origin_m"
        ch[f"distance_{tripod}_to_couple_m"] = _chk(dist, f"4-9 ({src})", _in(dist, 4.0, 9.0))
        g2a = rep.get("gates_G2a") or {}
        for k in ("world_scale", "world_scale_chain", "gravity_vs_building"):
            if k in g2a:
                ch[f"G2a_{k}"] = _chk(g2a[k].get("passed"), "passed", bool(g2a[k].get("passed")))
    out["G6_world"] = _gate(ch, m)
    miss_all += m
    # G7 triangulated dancers
    m, ch = [], {}
    tv = need("cameras/qa/twoview_check.json", m)
    if tv:
        for c, v in (tv.get("reproj_px") or {}).items():
            ch[f"reproj_median_{c}"] = _chk(v.get("median"), "<= 6 px", _le(v.get("median"), 6.0))
            ch[f"reproj_p90_{c}"] = _chk(v.get("p90"), "<= 15 px", _le(v.get("p90"), 15.0))
        for r in ROLES:
            cv = (tv.get("coverage") or {}).get(r) or {}
            ch[f"coverage_{r}"] = _chk(cv.get("share"), ">= 0.70 of the window's master frames (>= 8 joints in both views)",
                                       _ge(cv.get("share"), 0.70))
            ch[f"coverage_windows_below_50pct_{r}"] = _info(cv.get("windows_below_50pct"))
            bcv = (tv.get("bone_cv_max") or {}).get(r)
            ch[f"bone_cv_max_{r}"] = _chk(bcv, "<= 0.08 (CV = SD / mean)", _le(bcv, 0.08))
            ch[f"bone_cv_robust_max_{r}"] = _info((tv.get("bone_cv_robust_max") or {}).get(r),
                                                  "1.4826 MAD / median: well below the CV = a few outlying frames")
            ch[f"bone_lr_max_{r}"] = _chk((tv.get("bone_lr_max") or {}).get(r), "<= 0.05", _le((tv.get("bone_lr_max") or {}).get(r), 0.05))
            ra = ((tv.get("stature") or {}).get(r) or {}).get("ratio")
            ch[f"stature_ratio_{r}"] = _chk(ra, "implied / TOML within +-3 %", None if ra is None else abs(ra - 1) <= 0.03)
        lf = tv.get("stature_lead_follow_ratio")
        ch["stature_lead_follow"] = _chk(lf, "within +-2.5 %", None if lf is None else abs(lf - 1) <= 0.025)
        sw = tv.get("swap") or {}
        ch["swap_share"] = _chk(sw.get("share_any"), "<= 0.01 (frames where the swapped pairing is cheaper)",
                                _le(sw.get("share_any"), 0.01))
        ch["swap_share_margin"] = _info(sw.get("share"), f"swapped pairing cheaper by > {1 - SWAP_MARGIN:.0%}: well below "
                                                         "swap_share = near-ambiguous frames, not a swap")
        used = (tv.get("offsets_file") or {}).get("created")
        cur_off = _load(work / "sync" / "offsets.json", [])  # G8 lists it when missing
        ch["offsets_of_this_sync"] = _chk({"check_used": used, "offsets_json": (cur_off or {}).get("created")},
                                          "check ran on the current sync/offsets.json",
                                          None if cur_off is None else bool(used is not None and used == cur_off.get("created")))
        ch["note"] = _info(tv.get("note"))
    out["G7_dancers"] = _gate(ch, m)
    miss_all += m
    # G8 sync
    m, ch = [], {}
    off, cur = need("sync/offsets.json", m), need("sync/curves_2v.json", m)
    sc_s, sc_a = need("scale/chain_sync.json", m), need("scale/chain_audio_clocks.json", m)
    if off:
        o = (off.get("offsets") or {}).get(moving) or {}
        ch["method"] = _chk(off.get("method"), "two_view_epipolar", off.get("method") == "two_view_epipolar")
        dm = o.get("delta_ms")
        ch[f"delta_ms_{moving}"] = _chk(dm, "|delta| <= 80 ms, not at the search edge",
                                        None if dm is None else bool(abs(dm) <= 80 and not o.get("at_search_edge")))
        ch["sub_windows_valid"] = _chk(o.get("n_sub_valid"), ">= 5 (30 s)", _ge(o.get("n_sub_valid"), 5))
        if o.get("piecewise"):  # delta(t): judged by the reproducibility of the curve, not by one constant's spread
            pd = o.get("piecewise_detail") or {}
            ch["piecewise_reproducible"] = _chk(pd, f"even / odd curves within {PW_REPRO_MEDIAN_MS:g} ms (median), "
                                                    f"{PW_REPRO_MAX_MS:g} ms (max); >= {PW_MIN_SHARE:.0%} windows; cost <= constant",
                                                bool(o.get("stable")))
            ch["spread_ms"] = _info({"constant_spread_ms": o.get("spread_ms"), "piecewise_range_ms": pd.get("range_ms")})
        else:
            ch["spread_ms"] = _chk(o.get("spread_ms"), "<= 10", _le(o.get("spread_ms"), 10.0))
            b = (o.get("fit") or {}).get("b_ms_per_100s")
            ch["drift_ms_per_100s"] = _chk(b, "|b| <= 3", None if b is None else abs(b) <= 3.0)
    if cur:
        # a rejected re-run keeps the previous offsets.json (by design) but rewrites curves_2v.json: the file must be
        # the last run's, and that run accepted
        cd = (cur.get("doc") or {}).get("created")
        same = bool(off) and cd is not None and cd == off.get("created")
        ch["offsets_of_last_sync_run"] = _chk({"accepted": cur.get("accepted"), "why": cur.get("why"), "run_created": cd,
                                               "offsets_created": (off or {}).get("created")},
                                              "the last sync run was accepted and wrote sync/offsets.json",
                                              None if not off else bool(cur.get("accepted") is True and same))
        dr = cur.get("depth_ratio") or {}
        vals = [v for v in dr.values() if v is not None]
        ch["curve_depth"] = _chk(dr, "cost at +-40 ms >= 1.15 x the minimum", bool(len(vals) == 2 and min(vals) >= 1.15)
                                 if dr else None)
    if sc_s and sc_a:
        r = sc_s["s_chain"] / sc_a["s_chain"] - 1
        ch["chain_sync_vs_audio"] = _chk(_f(r, 5), "|s_chain(sync) / s_chain(audio) - 1| <= 0.01", abs(r) <= 0.01)
    out["G8_sync"] = _gate(ch, m)
    miss_all += m
    # the tripod-track decision (not a gate)
    ptc = _load(q / "twoview_check_ptz.json", [])
    dec = {"available": bool(tv and ptc)}
    if tv and ptc:
        a, b = (tv.get("reproj_px") or {}).get(tripod, {}).get("median"), (ptc.get("reproj_px") or {}).get(tripod, {}).get("median")
        gain = (a - b) / a if a and b is not None else None
        dec.update({"loc_median_px": a, "ptz_median_px": b, "gain": _f(gain, 4),
                    "adopt_ptz": bool(gain is not None and gain >= 0.10 and out["G3_types_tripod"]["pass"]
                                      and out["G4_tripod_vs_ptz"]["pass"]),
                    "rule": "adopt the PTZ track when G3 and G4 pass and it reprojects the dancers >= 10 % better"})
    return {"gates": out, "tripod_decision": dec, "cams": {"tripod": tripod, "moving": moving},
            "missing_inputs": sorted(set(miss_all)), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}


def cmd_gates(work: Path, cfg: dict, tripod: str | None = None, moving: str | None = None, log=print) -> int:
    """gates: writes cameras/qa/twoview_gates.json, prints the table; returns 1 when an input is missing, else 0."""
    t0, m0 = _camera_roles(cfg)
    doc = gates_eval(work, cfg, tripod or t0, moving or m0)
    _write_json(work / "cameras" / "qa" / "twoview_gates.json", {**doc["gates"], "_meta": {k: v for k, v in doc.items()
                                                                                          if k != "gates"}})
    log(f"{'gate':<20} {'status':<8} failing / soft / missing checks")
    for g, v in doc["gates"].items():
        bad = [f"{k}={c.get('value')}" for k, c in v["checks"].items() if c["status"] in ("fail", "soft", "missing")]
        log(f"{g:<20} {v['status']:<8} {'; '.join(bad)[:150]}")
    if doc["tripod_decision"].get("available"):
        log(f"tripod track: {doc['tripod_decision']}")
    if doc["missing_inputs"]:
        log("[twoview] missing inputs: " + ", ".join(doc["missing_inputs"]))
        return 1
    return 0


# ------------------------------------------------------------------------------------------------ CLI


def _fps(s: str) -> float:
    a, _, b = str(s).partition("/")
    return float(a) / float(b) if b else float(a)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.twoview")
    ap.add_argument("cmd", choices=("map-gate", "ptz", "atlas", "sync", "check", "gates"))
    ap.add_argument("take")
    ap.add_argument("--need", default="01,02", help="map-gate: cameras that must be in the main model")
    ap.add_argument("--cam", help="ptz / atlas: the camera")
    ap.add_argument("--ptz-take", help="ptz: the take whose camera_work/final/track_s*.npz is the PTZ track")
    ap.add_argument("--ptz-track", help="ptz: that track file instead")
    ap.add_argument("--clock-shift", type=float, default=0.0, help="ptz: this take's time of the cut's t = 0 (s)")
    ap.add_argument("--cut-fps", type=_fps, default=30.0, help="ptz: frame rate of the PTZ take's cut")
    ap.add_argument("--src-fps", type=_fps, default=SRC_FPS, help="ptz: frame rate of the original the cut was made from")
    ap.add_argument("--cut-rule", choices=CUT_RULES, default="fps",
                    help="ptz: the source frame a cut frame holds (fps: FFmpeg's fps filter; nearest / nominal: comparison)")
    ap.add_argument("--zoom-gap", default=f"{ZOOM_GAP[0]:g},{ZOOM_GAP[1]:g}", help="ptz: excluded zoom stretch (s,s)")
    ap.add_argument("--mode", choices=("check", "attach"), default="check")
    ap.add_argument("--adopt", action="store_true", help="ptz attach: install the PTZ track as cameras/<cam>.npz")
    ap.add_argument("--atlas", help="atlas: pose_<cam>_<n>.json")
    ap.add_argument("--master", help="sync / check: the gauge camera (sync default 02)")
    ap.add_argument("--other", default="01", help="sync: the camera whose offset is measured")
    ap.add_argument("--range-ms", type=float, default=300.0)
    ap.add_argument("--sub-s", type=float, default=30.0)
    ap.add_argument("--track", action="append", default=[], help="check: CAM=path (relative to work) instead of cameras/<cam>.npz")
    ap.add_argument("--tag", help="check: output suffix (twoview_check_<tag>.json)")
    ap.add_argument("--tripod", help="gates: the tripod camera (default from the TOML mount)")
    ap.add_argument("--moving", help="gates: the moving camera (default from the TOML mount)")
    a = ap.parse_args(argv)
    work = md.take_work(a.take)
    cfg = take_cfg(a.take)
    if a.cmd == "map-gate":
        doc = cmd_map_gate(work, [c.strip() for c in a.need.split(",") if c.strip()])
        if doc["exit"]:
            raise SystemExit(doc["exit"])
    elif a.cmd == "ptz":
        if not a.cam or not (a.ptz_take or a.ptz_track):
            raise SystemExit("ptz: --cam and --ptz-take (or --ptz-track) are required")
        gap = tuple(float(x) for x in a.zoom_gap.split(","))
        cmd_ptz(work, a.cam, md.take_work(a.ptz_take) if a.ptz_take else Path(a.ptz_track).parents[2], a.clock_shift, a.mode,
                a.adopt, a.cut_fps, a.src_fps, a.cut_rule, gap, Path(a.ptz_track) if a.ptz_track else None)
    elif a.cmd == "atlas":
        if not a.cam or not a.atlas:
            raise SystemExit("atlas: --cam and --atlas are required")
        p = Path(a.atlas)
        cmd_atlas(work, a.cam, p if p.exists() else PACKAGE_ROOT / p)
    elif a.cmd == "sync":
        cmd_sync(work, a.master or "02", a.other, a.range_ms, a.sub_s, _window(cfg))
    elif a.cmd == "check":
        tracks = dict(x.split("=", 1) for x in a.track)
        cmd_check(work, cfg, a.master, tracks, a.tag)
    else:
        rc = cmd_gates(work, cfg, a.tripod, a.moving)
        if rc:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
