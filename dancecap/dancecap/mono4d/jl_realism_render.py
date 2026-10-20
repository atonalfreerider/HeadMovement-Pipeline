r"""john_lula realism stage 3 (2026-10-20): the FINAL LHM++ avatars of John (lead) and Lula (follow), posed by the MAMMA +
SAM 3D Body fit with LHM++'s own animation, rendered for the WHOLE dance (0-88.07 s), and the tutorial videos.

Owner, 2026-10-19: "Yes: use skinned LHM++ splats posed by the MAMMA + SAM 3D Body fit for the john_lula tutorial video.
Run the full john_lula process now (SAM 3D Body on all frames, the prior refit, LHM++ with my whole photo mask,
renders)".

    python -m dancecap.mono4d.jl_realism_render prep [--fit DIR]   # input/ (refs, poses, views) + camera check sheets
    python -m dancecap.mono4d.jl_realism_render job                # remote/cand_lhmpp/pod_job_jl_final.json
    (DANCECAP_RUNPOD_CAP=... python -m dancecap remote run --jobs <that json> --max-cost 2 --gpus ...)
    python -m dancecap.mono4d.jl_realism_render import [--session SID]
    python -m dancecap.mono4d.jl_realism_render videos [--only front,sbs,orbit]
    python -m dancecap.mono4d.jl_realism_render sheets

POSE. Default driver: realism/refit/audit_patch/smplx_rf2_noslip_3db_patched/{lead,follow}.npz = the new fit (one-camera
MAMMA + SAM 3D Body prior w 0.5, stage 2 = smplx_3db) after the take's standard pose_refine (smplx_rf2_noslip: feet on the
floor, less skate / jitter), with John's 26.20-26.43 s taken from the current fit (the stage-2 audit's patch: both new
fits lurch John 30 deg back along the camera ray there, invisible in 2D, a lurch in every novel view). Same contract as
smplx/. --fit picks another folder (e.g. smplx_3db/).

AVATARS. The preview's lhmpp_video8_photofull inputs byte for byte (jl_lhmpp_preview: 8 clean video crops per dancer by
the w8 + white-patch rules; John: his WHOLE visible photo mask FIRST as the frontal reference; Lula: her photo crop
appended), shape mode 'ref' (betas[:10] of the fit; pinned, identical to smplx/), one LHM++ forward per dancer on the
pod (LHMPP-700M-PixelShuffle), 160k Gaussians each.

POSE KEYS (input/poses.npz). keys 0..2642 = the 2643 video frames at their pts (identity clock: t = pts), 2643..5285 =
the fit's 30 Hz grid k/30 (k 0..2642). Parameters at a pts: cand_lhmpp.interp_params (per-joint slerp, linear transl /
joints), vectorised (checked against it).

VIEWS (input/views.npz; OpenCV pinhole, world Y up, floor y = 0; the phone's k1 -0.012 is ignored as in the preview):
  front     TUTORIAL FRONT 1920 x 1080 at the grid times: camera 1.5 m high, 3.5 m (horizontal) from the couple centre
            (mean pelvis xz, Gaussian sigma 1 s) on the couple's smoothed mean facing direction (sum of both dancers'
            horizontal facing vectors, Gaussian sigma 8 s, then the unwrapped azimuth again with sigma 4 s: the viewer
            sits in front of the couple as an audience would, no jitter), looking horizontally at the centre, pitched
            to a fixed look-at height; ONE focal length for the whole dance = the largest that keeps every frame's
            couple points (joints 0-21 of both + a crown 0.20 m above the head joint + the ankles dropped to the floor)
            inside with a margin (FRONT['margin'] x H; checked per frame, reported).
  phone     the phone's own track (each frame's K, R, tvec from cameras/01.npz), 538 x 960, at the frame's pts.
  phone_hi  the same cameras at 1080 x 1920 (K x 2, principal point + 0.5 for the pixel-centre convention, 2 px pad
            each side: 1076 px of real width), for the side-by-side.
  orbit     ORBIT 1080 x 1080: a 360 deg orbit at 3 m (horizontal) around the couple centre (sigma 1.5 s), 1.6 m high,
            ORBIT['seconds'] long over the window with the most follow turning that holds a >= 360 deg spin, starting at
            the front camera's azimuth; one focal length for the window (same framing rule).
Passes (priority if the pod runs short): front, then phone_hi + phone, then orbit.

VIDEOS (local, ffmpeg libx264 CRF 18, yuv420p, 30 fps, the heat's own audio on the same clock): realism/videos/
  jl_tutorial_front.mp4 (1920 x 1080, studio grey wall + warm grey floor with soft contact shadows, caption),
  jl_side_by_side.mp4 (1920 x 1080: real phone | splats along the phone path, each 605 x 1080),
  jl_orbit.mp4 (1080 x 1080, audio of its window).
Likeness data (crops, splats, renders, videos) stays in work/ and our own pods; nothing here is committed.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE

TAKE, CAM = "john_lula", "01"
ROLES = ("lead", "follow")
NAME = {"lead": "John", "follow": "Lula"}
WORK = PACKAGE_ROOT / "work" / TAKE
REAL = WORK / "realism"
LH = REAL / "lhmpp"
INPUT = LH / "input"
POD_OUT = LH / "pod_out"
RENDERS = REAL / "renders"
VIDEOS = REAL / "videos"
SHEETS = REAL / "sheets"
FIT_DIR = REAL / "refit" / "audit_patch" / "smplx_rf2_noslip_3db_patched"
PREVIEW = WORK / "lhmpp_preview"
PREVIEW_VARIANT = "lhmpp_video8_photofull"
AUDIO_FALLBACK = Path("C:/Users/johnb/Videos/Zouk/john-lula/john_lula_jnj.mp4")
FPS = 30.0
FRONT = {"W": 1920, "H": 1080, "height_m": 1.5, "dist_m": 3.5, "face_sigma_s": 8.0, "angle_sigma_s": 4.0,
         "centre_sigma_s": 1.0, "margin": 0.05}
ORBIT = {"W": 1080, "H": 1080, "radius_m": 3.0, "height_m": 1.6, "seconds": 25.0, "centre_sigma_s": 1.5,
         "margin": 0.05, "spin_deg": 360.0, "spin_window_s": 1.2}
PHONE_HI = {"W": 1080, "H": 1920, "scale": 2.0}
PASSES = [["front"], ["phone_hi", "phone"], ["orbit"]]
CAPTION = "John & Lula - Jack & Jill (3D splats from one phone)"
JOB_NAME = "jl_lhmpp_final"
JOB_PATH = PACKAGE_ROOT / "remote" / "cand_lhmpp" / "pod_job_jl_final.json"
ROT_KEYS = (("global_orient", 1, "root"), ("body_pose", 21, "body"), ("jaw_pose", 1, "jaw"), ("leye_pose", 1, "leye"),
            ("reye_pose", 1, "reye"), ("left_hand_pose", 15, "lhand"), ("right_hand_pose", 15, "rhand"))
# background (sRGB 0..1): wall above the horizon, floor below, fog to the horizon colour
BG = {"top": (0.47, 0.48, 0.50), "horizon": (0.66, 0.66, 0.66), "floor": (0.60, 0.565, 0.52), "fog_m": 14.0,
      "pelvis_sigma_m": 0.22, "pelvis_strength": 0.22, "foot_sigma_m": 0.075, "foot_strength": 0.30,
      "foot_lift_m": 0.08, "shade_min": 0.45}


def log(msg: str) -> None:
    print(f"[jl_realism] {msg}", flush=True)


def _ws(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def audio_source() -> Path:
    try:
        from . import jl_lhmpp_preview as JP
        if JP.VIDEO.exists():
            return JP.VIDEO
    except Exception:  # noqa: BLE001
        pass
    return AUDIO_FALLBACK


# ================================================================================================ fit + poses
def load_fit(role: str, fit_dir: Path = FIT_DIR) -> dict:
    with np.load(Path(fit_dir) / f"{role}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def interp_many(d: dict, ts: np.ndarray) -> dict:
    """cand_lhmpp.interp_params for many times at once: per-joint slerp of the rotations, linear transl / joints."""
    from scipy.spatial.transform import Rotation as Rot
    T = np.asarray(d["t"], float)
    ts = np.asarray(ts, float)
    k = np.clip(np.searchsorted(T, ts) - 1, 0, len(T) - 2)
    a = np.clip((ts - T[k]) / (T[k + 1] - T[k]), 0.0, 1.0)
    out = {"t": ts, "k": k, "a": a}
    N = len(ts)
    for name, n, _ in ROT_KEYS:
        A = np.asarray(d[name], float).reshape(len(T), n, 3)
        r0 = Rot.from_rotvec(A[k].reshape(-1, 3))
        r1 = Rot.from_rotvec(A[k + 1].reshape(-1, 3))
        rel = (r0.inv() * r1).as_rotvec()
        out[name] = (r0 * Rot.from_rotvec(rel * np.repeat(a, n)[:, None])).as_rotvec().reshape(N, n * 3)
    for name in ("transl", "joints"):
        X = np.asarray(d[name], float)
        w = a.reshape((-1,) + (1,) * (X.ndim - 1))
        out[name] = (1 - w) * X[k] + w * X[k + 1]
    return out


def facing(d: dict) -> np.ndarray:
    """Horizontal unit forward vector of the body (SMPL-X canonical body faces +Z) per grid frame."""
    from scipy.spatial.transform import Rotation as Rot
    f = Rot.from_rotvec(np.asarray(d["global_orient"], float)).apply([0.0, 0.0, 1.0])
    f[:, 1] = 0.0
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)


def couple_points(J: dict) -> np.ndarray:
    """(F, 50, 3): joints 0-21 of both + a crown 0.20 m above the head joint + both ankles dropped to the floor."""
    P = []
    for r in ROLES:
        j = np.asarray(J[r], float)
        crown = j[:, 15] + np.array([0.0, 0.20, 0.0])
        P += [j[:, :22], crown[:, None], j[:, [7, 8]] * np.array([1.0, 0.0, 1.0])]
    return np.concatenate(P, 1)


def gsmooth(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.asarray(x, float), sigma_frames, axis=0, mode="nearest")


# ================================================================================================ cameras
def look_at_many(C: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(N,3,3) R of pinholes at C looking at target, world Y up (= appearance_compare.look_at, vectorised)."""
    z = target - C
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def fit_focal(C: np.ndarray, centre_xz: np.ndarray, P: np.ndarray, W: int, H: int, margin: float,
              y_grid=np.arange(0.40, 1.301, 0.02)) -> dict:
    """The look-at height and the ONE focal length (largest) that keep every frame's points inside the W x H image with
    margin x H px. P (N, M, 3); camera n at C[n] looks at (centre_xz[n, 0], y, centre_xz[n, 1])."""
    best = None
    m = margin * H
    for y in y_grid:
        tgt = np.stack([centre_xz[:, 0], np.full(len(C), y), centre_xz[:, 1]], 1)
        R = look_at_many(C, tgt)
        Xc = np.einsum("nij,nmj->nmi", R, P - C[:, None])
        z = Xc[..., 2]
        if (z <= 0.3).any():
            continue
        fx = (W / 2 - m) / np.abs(Xc[..., 0] / z).max(1)
        fy = (H / 2 - m) / np.abs(Xc[..., 1] / z).max(1)
        fpf = np.minimum(fx, fy)
        f = float(fpf.min())
        if best is None or f > best["f"]:
            best = {"f": f, "y_look": round(float(y), 3), "f_per_frame": fpf, "R": R}
    if best is None:
        raise SystemExit("framing: a point is behind / too near every candidate camera")
    fpf = best.pop("f_per_frame")
    best["limiting_frames"] = [int(i) for i in np.argsort(fpf)[:5]]
    best["f_per_frame_p1_p50"] = [round(float(np.percentile(fpf, 1)), 1), round(float(np.median(fpf)), 1)]
    best["vfov_deg"] = round(math.degrees(2 * math.atan(H / 2 / best["f"])), 2)
    return best


def frame_margins(K: np.ndarray, R: np.ndarray, tvec: np.ndarray, P: np.ndarray, W: int, H: int) -> np.ndarray:
    """Per frame: the smallest distance (px / H) of the points to the image border (negative = outside)."""
    Xc = np.einsum("nij,nmj->nmi", R, P) + tvec[:, None]
    z = Xc[..., 2]
    u = K[:, None, 0, 0] * Xc[..., 0] / z + K[:, None, 0, 2]
    v = K[:, None, 1, 1] * Xc[..., 1] / z + K[:, None, 1, 2]
    mm = np.minimum.reduce([u, W - 1 - u, v, H - 1 - v])
    mm[z <= 0.1] = -1e9
    return mm.min(1) / H


def front_design(fits: dict) -> dict:
    T = np.asarray(fits["lead"]["t"], float)
    F = len(T)
    J = {r: np.asarray(fits[r]["joints"], float) for r in ROLES}
    pel = 0.5 * (J["lead"][:, 0] + J["follow"][:, 0])
    cen = gsmooth(pel[:, [0, 2]], FRONT["centre_sigma_s"] * FPS)
    s = facing(fits["lead"]) + facing(fits["follow"])
    ss = gsmooth(s, FRONT["face_sigma_s"] * FPS)
    az = gsmooth(np.unwrap(np.arctan2(ss[:, 0], ss[:, 2])), FRONT["angle_sigma_s"] * FPS)
    u = np.stack([np.sin(az), np.cos(az)], 1)
    C = np.stack([cen[:, 0] + FRONT["dist_m"] * u[:, 0], np.full(F, FRONT["height_m"]),
                  cen[:, 1] + FRONT["dist_m"] * u[:, 1]], 1)
    P = couple_points(J)
    W, H = FRONT["W"], FRONT["H"]
    fr = fit_focal(C, cen, P, W, H, FRONT["margin"])
    tgt = np.stack([cen[:, 0], np.full(F, fr["y_look"]), cen[:, 1]], 1)
    R = look_at_many(C, tgt)
    K = np.array([[fr["f"], 0, (W - 1) / 2], [0, fr["f"], (H - 1) / 2], [0, 0, 1.0]])
    tvec = -np.einsum("nij,nj->ni", R, C)
    Kn = np.repeat(K[None], F, 0)
    mg = frame_margins(Kn, R, tvec, P, W, H)
    fd = {r: facing(fits[r]) for r in ROLES}
    dcam = np.stack([C[:, 0] - pel[:, 0], C[:, 2] - pel[:, 2]], 1)
    dcam /= np.linalg.norm(dcam, axis=1, keepdims=True)
    front_frac = {r: round(float(((fd[r][:, [0, 2]] * dcam).sum(1) > 0.3).mean()), 3) for r in ROLES}
    near = min(float(np.linalg.norm(J[r][:, :22] - C[:, None], axis=-1).min()) for r in ROLES)
    rate = np.degrees(np.abs(np.diff(az))) * FPS
    speed = np.linalg.norm(np.diff(C, axis=0), axis=1) * FPS
    info = {"F": F, "f_px": round(fr["f"], 2), "vfov_deg": fr["vfov_deg"], "y_look_m": fr["y_look"],
            "margin_min_frac_H": round(float(mg.min()), 4), "margin_p1_frac_H": round(float(np.percentile(mg, 1)), 4),
            "limiting_frames": fr["limiting_frames"], "f_per_frame_p1_p50": fr["f_per_frame_p1_p50"],
            "closest_dancer_joint_m": round(near, 3),
            "azimuth_deg_every_10s": [round(float(np.degrees(a)) % 360, 1) for a in az[::300]],
            "azimuth_total_turn_deg": round(float(np.degrees(az[-1] - az[0])), 1),
            "azimuth_rate_deg_s": {"p95": round(float(np.percentile(rate, 95)), 2), "max": round(float(rate.max()), 2)},
            "camera_speed_m_s": {"p95": round(float(np.percentile(speed, 95)), 3), "max": round(float(speed.max()), 3)},
            "dancer_faces_camera_frac(cos>0.3)": front_frac, **{k: FRONT[k] for k in FRONT}}
    return {"K": Kn, "R": R, "tvec": tvec, "C": C, "az": az, "cen": cen, "info": info}


def orbit_window(fits: dict) -> dict:
    """The ORBIT['seconds'] window with the most follow turning (sum |d yaw|) that holds a spin of >= spin_deg within
    spin_window_s (the follow's or the lead's)."""
    from scipy.spatial.transform import Rotation as Rot
    T = np.asarray(fits["lead"]["t"], float)
    yaw = {}
    for r in ROLES:
        f = Rot.from_rotvec(np.asarray(fits[r]["global_orient"], float)).apply([0.0, 0.0, 1.0])
        yaw[r] = np.degrees(np.unwrap(np.arctan2(f[:, 0], f[:, 2])))
    n = int(round(ORBIT["spin_window_s"] * FPS))
    spins = []
    for r in ROLES:
        dy = yaw[r][n:] - yaw[r][:-n]
        order = np.argsort(-np.abs(dy))
        for i in order:
            if abs(dy[i]) < ORBIT["spin_deg"]:
                break
            if all(abs(i - s["k"]) > 45 for s in spins if s["role"] == r):
                spins.append({"role": r, "k": int(i), "t": round(float(T[i]), 2), "deg": round(float(dy[i]), 1)})
    L = int(round(ORBIT["seconds"] * FPS))
    best = None
    for k0 in range(0, len(T) - L, 15):
        inside = [s for s in spins if k0 <= s["k"] and s["k"] + n <= k0 + L]
        if not inside:
            continue
        turn = float(np.abs(np.diff(yaw["follow"][k0:k0 + L])).sum())
        if best is None or turn > best["follow_turn_deg"]:
            best = {"k0": k0, "k1": k0 + L - 1, "t0": round(float(T[k0]), 4), "t1": round(float(T[k0 + L - 1]), 4),
                    "follow_turn_deg": round(turn, 1), "spins": sorted(inside, key=lambda s: s["k"])}
    if best is None:
        raise SystemExit("no orbit window with a spin")
    best["all_spins"] = sorted(spins, key=lambda s: s["k"])
    big = max(best["spins"], key=lambda s: abs(s["deg"]))
    best["scene_k"] = int(big["k"] + n // 2)
    return best


def orbit_design(fits: dict, front: dict, win: dict) -> dict:
    k0, k1 = win["k0"], win["k1"]
    ks = np.arange(k0, k1 + 1)
    N = len(ks)
    J = {r: np.asarray(fits[r]["joints"], float)[ks] for r in ROLES}
    pel = 0.5 * (J["lead"][:, 0] + J["follow"][:, 0])
    cen = gsmooth(pel[:, [0, 2]], ORBIT["centre_sigma_s"] * FPS)
    a0 = float(front["az"][k0])
    a = a0 + 2 * math.pi * np.arange(N) / N
    C = np.stack([cen[:, 0] + ORBIT["radius_m"] * np.sin(a), np.full(N, ORBIT["height_m"]),
                  cen[:, 1] + ORBIT["radius_m"] * np.cos(a)], 1)
    P = couple_points(J)
    W, H = ORBIT["W"], ORBIT["H"]
    fr = fit_focal(C, cen, P, W, H, ORBIT["margin"])
    tgt = np.stack([cen[:, 0], np.full(N, fr["y_look"]), cen[:, 1]], 1)
    R = look_at_many(C, tgt)
    K = np.array([[fr["f"], 0, (W - 1) / 2], [0, fr["f"], (H - 1) / 2], [0, 0, 1.0]])
    tvec = -np.einsum("nij,nj->ni", R, C)
    Kn = np.repeat(K[None], N, 0)
    mg = frame_margins(Kn, R, tvec, P, W, H)
    near = min(float(np.linalg.norm(J[r][:, :22] - C[:, None], axis=-1).min()) for r in ROLES)
    info = {"k0": k0, "k1": k1, "t0": win["t0"], "t1": win["t1"], "frames": N, "f_px": round(fr["f"], 2),
            "vfov_deg": fr["vfov_deg"], "y_look_m": fr["y_look"], "start_azimuth_deg": round(math.degrees(a0) % 360, 2),
            "deg_per_s": round(360.0 / (N / FPS), 2), "margin_min_frac_H": round(float(mg.min()), 4),
            "closest_dancer_joint_m": round(near, 3), "spins": win["spins"], "follow_turn_deg": win["follow_turn_deg"],
            **{k: ORBIT[k] for k in ORBIT}}
    return {"K": Kn, "R": R, "tvec": tvec, "C": C, "ks": ks, "info": info}


def phone_cams(tr) -> dict:
    K = np.asarray(tr.K, float)
    s = PHONE_HI["scale"]
    pad = (PHONE_HI["W"] - s * tr.width) / 2
    Kh = K.copy()
    Kh[:, 0, 0] *= s
    Kh[:, 1, 1] *= s
    Kh[:, 0, 2] = s * K[:, 0, 2] + (s - 1) / 2 + pad
    Kh[:, 1, 2] = s * K[:, 1, 2] + (s - 1) / 2 + (PHONE_HI["H"] - s * tr.height) / 2
    return {"K": K, "K_hi": Kh, "R": np.asarray(tr.R, float), "tvec": np.asarray(tr.tvec, float), "pad_px": pad}


# ================================================================================================ background
def background(K, R, tvec, W: int, H: int, feet: list | None = None, ds: int = 4) -> np.ndarray:
    """Studio backdrop as uint8 BGR: grey wall above the horizon, warm grey floor (y = 0) fogged to the horizon colour,
    soft contact shadows (pelvis + feet blobs on the floor, fainter as a foot lifts). feet: [(x, y, z, kind)]."""
    K, R, tvec = np.asarray(K, float), np.asarray(R, float), np.asarray(tvec, float)
    w, h = int(math.ceil(W / ds)), int(math.ceil(H / ds))
    u = (np.arange(w) + 0.5) * ds - 0.5
    v = (np.arange(h) + 0.5) * ds - 0.5
    uu, vv = np.meshgrid(u, v)
    dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    dw = dc @ R
    n = dw / np.linalg.norm(dw, axis=-1, keepdims=True)
    C = -R.T @ tvec
    top, hor, flo = (np.array(BG[k]) for k in ("top", "horizon", "floor"))
    wall = hor + (top - hor) * np.clip(n[..., 1:2] * 2.5, 0, 1)
    img = wall
    below = dw[..., 1] < -1e-6
    if below.any() and C[1] > 0:
        s = np.where(below, -C[1] / np.where(below, dw[..., 1], -1.0), 0.0)
        X = C + s[..., None] * dw
        dist = s * np.linalg.norm(dw, axis=-1)
        shade = np.ones((h, w))
        for (x, y, z, kind) in feet or []:
            if kind == "pelvis":
                sg, st = BG["pelvis_sigma_m"], BG["pelvis_strength"]
            else:
                sg = BG["foot_sigma_m"]
                st = BG["foot_strength"] * math.exp(-max(0.0, y - 0.06) / BG["foot_lift_m"])
            d2 = (X[..., 0] - x) ** 2 + (X[..., 2] - z) ** 2
            shade -= st * np.exp(-d2 / (2 * sg * sg))
        shade = np.clip(shade, BG["shade_min"], 1.0)
        fog = 1.0 - np.exp(-dist / BG["fog_m"])
        floor = flo * shade[..., None] * (1 - fog[..., None]) + hor * fog[..., None]
        img = np.where(below[..., None], floor, wall)
    low = np.ascontiguousarray(np.clip(img[..., ::-1], 0, 1).astype(np.float32))
    return cv2.convertScaleAbs(cv2.resize(low, (W, H), interpolation=cv2.INTER_LINEAR), alpha=255.0)


def shadow_points(J: dict) -> list:
    """J: role -> (55, 3) joints at one time."""
    out = []
    for r in ROLES:
        j = J[r]
        out.append((float(j[0, 0]), 0.0, float(j[0, 2]), "pelvis"))
        for a in (7, 8, 10, 11):
            out.append((float(j[a, 0]), float(j[a, 1]), float(j[a, 2]), "foot"))
    return out


def composite(bg: np.ndarray, rgba: np.ndarray | None) -> np.ndarray:
    if rgba is None:
        return bg
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return (bg.astype(np.float32) * (1 - a) + rgba[:, :, :3].astype(np.float32) * a + 0.5).astype(np.uint8)


# ================================================================================================ prep
def _draw_points(img, K, R, tvec, J: dict, cols=((60, 200, 60), (60, 60, 230))):
    from . import cand_lhmpp as CL
    bones = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
             (12, 15), (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]
    for r, col in zip(ROLES, cols):
        uv, z = CL.project(np.asarray(J[r][:22], float), K, R, tvec)
        for a_, b_ in bones:
            if z[a_] > 0 and z[b_] > 0:
                cv2.line(img, tuple(int(x) for x in uv[a_]), tuple(int(x) for x in uv[b_]), col, 3, cv2.LINE_AA)
    return img


def _label(img, text, org=(10, 30), s=0.8):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)


def prep(fit_dir: Path = FIT_DIR) -> dict:
    from .. import mamma_dyn as md
    from . import cand_lhmpp as CL
    t0 = time.time()
    fit_dir = Path(fit_dir)
    tr = md.load_tracks(WORK, [CAM])[CAM]
    fits = {r: load_fit(r, fit_dir) for r in ROLES}
    T = np.asarray(fits["lead"]["t"], float)
    F = len(T)
    NV = len(tr.t)
    assert np.allclose(fits["follow"]["t"], T) and F == 2643 and NV == 2643, (F, NV)
    if INPUT.exists():
        shutil.rmtree(INPUT)
    (INPUT / "refs").mkdir(parents=True)
    # ---- refs: the preview's photofull inputs, byte for byte
    prefs = json.loads((PREVIEW / "input" / "refs.json").read_text())
    refs = {}
    for r in ROLES:
        src = PREVIEW / "input" / "refs" / r / PREVIEW_VARIANT
        dst = INPUT / "refs" / r
        dst.mkdir(parents=True)
        files = sorted(src.glob("*.png"))
        for p in files:
            shutil.copy2(p, dst / p.name)
        refs[r] = {"from": _ws(src), "files": [{"file": p.name, "sha1": _sha1(p)} for p in files],
                   "slots": prefs[r]["files"][PREVIEW_VARIANT]}
    # ---- pose keys: video frames at pts, then the grid
    ts = np.concatenate([np.asarray(tr.t, float), T])
    poses = {"t": ts.astype(np.float64)}
    chk = {}
    for r in ROLES:
        p = interp_many(fits[r], ts)
        for name, n, short in ROT_KEYS:
            poses[f"{r}_{short}"] = p[name].astype(np.float32)
        poses[f"{r}_pelvis"] = p["joints"][:, 0].astype(np.float32)
        poses[f"{r}_joints22"] = p["joints"][:, :22].astype(np.float32)
        poses[f"{r}_transl"] = p["transl"].astype(np.float32)
        poses[f"{r}_betas"] = np.asarray(fits[r]["betas"], np.float32).reshape(-1)
        # vectorised interpolation == cand_lhmpp.interp_params (spot check), grid keys == the fit's own frames
        errs = []
        for q in np.linspace(0, len(ts) - 1, 25).round().astype(int):
            ref = CL.interp_params(fits[r], float(ts[q]))
            errs.append(max(float(np.abs(ref[name] - p[name][q]).max()) for name, _, _ in ROT_KEYS))
        grid_err = float(np.abs(p["body_pose"][NV:] - np.asarray(fits[r]["body_pose"], float).reshape(F, -1)).max())
        chk[r] = {"vs_interp_params_max_rad": float(max(errs)), "grid_vs_fit_body_pose_max_rad": grid_err,
                  "betas_equal_smplx": bool(np.array_equal(np.asarray(fits[r]["betas"]).ravel(),
                                                           np.asarray(load_fit(r, WORK / "smplx")["betas"]).ravel()))}
    np.savez_compressed(INPUT / "poses.npz", **poses)
    # ---- views
    front = front_design(fits)
    win = orbit_window(fits)
    orbit = orbit_design(fits, front, win)
    ph = phone_cams(tr)
    groups = {
        "front": {"W": FRONT["W"], "H": FRONT["H"], "K": front["K"], "R": front["R"], "tvec": front["tvec"],
                  "key": NV + np.arange(F), "index": np.arange(F), "t": T},
        "phone_hi": {"W": PHONE_HI["W"], "H": PHONE_HI["H"], "K": ph["K_hi"], "R": ph["R"], "tvec": ph["tvec"],
                     "key": np.arange(NV), "index": np.arange(NV), "t": np.asarray(tr.t, float)},
        "phone": {"W": tr.width, "H": tr.height, "K": ph["K"], "R": ph["R"], "tvec": ph["tvec"],
                  "key": np.arange(NV), "index": np.arange(NV), "t": np.asarray(tr.t, float)},
        "orbit": {"W": ORBIT["W"], "H": ORBIT["H"], "K": orbit["K"], "R": orbit["R"], "tvec": orbit["tvec"],
                  "key": NV + orbit["ks"], "index": np.arange(len(orbit["ks"])), "t": T[orbit["ks"]]},
    }
    arrs = {}
    for g, v in groups.items():
        for k in ("K", "R", "tvec"):
            arrs[f"{g}_{k}"] = np.asarray(v[k], np.float32)
        arrs[f"{g}_key"] = np.asarray(v["key"], np.int32)
        arrs[f"{g}_index"] = np.asarray(v["index"], np.int32)
        arrs[f"{g}_t"] = np.asarray(v["t"], np.float64)
    np.savez_compressed(INPUT / "views.npz", **arrs)
    vj = {"take": TAKE, "passes": PASSES, "scene_key": int(NV + win["scene_k"]),
          "groups": {g: {"W": int(v["W"]), "H": int(v["H"]), "n": int(len(v["key"]))} for g, v in groups.items()},
          "keys": {"video_frames": [0, NV - 1], "grid": [NV, NV + F - 1]}}
    (INPUT / "views.json").write_text(json.dumps(vj, indent=1))
    # ---- check sheets (LOOK before spending): skeletons through the planned cameras over the backdrop
    SHEETS.mkdir(parents=True, exist_ok=True)
    Jg = {r: np.asarray(fits[r]["joints"], float) for r in ROLES}
    tiles = []
    for k in list(np.linspace(30, F - 31, 7).round().astype(int)) + front["info"]["limiting_frames"][:1]:
        g = groups["front"]
        img = background(g["K"][k], g["R"][k], g["tvec"][k], g["W"], g["H"], shadow_points({r: Jg[r][k] for r in ROLES}))
        _draw_points(img, g["K"][k], g["R"][k], g["tvec"][k], {r: Jg[r][k] for r in ROLES})
        _label(img, f"front k{k} t={T[k]:.2f}s az {math.degrees(front['az'][k]) % 360:.0f}")
        tiles.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in (0, 4)], 0)
    cv2.imwrite(str(SHEETS / "check_front_cameras.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    tiles = []
    for n in np.linspace(0, len(orbit["ks"]) - 1, 8).round().astype(int):
        k = int(orbit["ks"][n])
        g = groups["orbit"]
        img = background(g["K"][n], g["R"][n], g["tvec"][n], g["W"], g["H"], shadow_points({r: Jg[r][k] for r in ROLES}))
        _draw_points(img, g["K"][n], g["R"][n], g["tvec"][n], {r: Jg[r][k] for r in ROLES})
        _label(img, f"orbit {n} t={T[k]:.2f}s")
        tiles.append(cv2.resize(img, (420, 420), interpolation=cv2.INTER_AREA))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in (0, 4)], 0)
    cv2.imwrite(str(SHEETS / "check_orbit_cameras.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    from . import jl_lhmpp_preview as JP
    tiles = []
    for i in np.linspace(40, NV - 41, 6).round().astype(int):
        pk = interp_many(fits["lead"], [tr.t[i]])["joints"][0], interp_many(fits["follow"], [tr.t[i]])["joints"][0]
        real = cv2.imread(str(JP.frame_path(i)))
        hi = np.full((PHONE_HI["H"], PHONE_HI["W"], 3), 40, np.uint8)
        pad = int(round(ph["pad_px"]))
        hi[:, pad:pad + 2 * tr.width] = cv2.resize(real, (2 * tr.width, 2 * tr.height), interpolation=cv2.INTER_LANCZOS4)
        _draw_points(hi, ph["K_hi"][i], ph["R"][i], ph["tvec"][i], {"lead": pk[0], "follow": pk[1]})
        _label(hi, f"phone_hi f{i} t={tr.t[i]:.2f}s", s=1.2)
        tiles.append(cv2.resize(hi, (360, 640), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(SHEETS / "check_phone_hi_cameras.jpg"), np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])
    info = {"take": TAKE, "fit_dir": _ws(fit_dir),
            "fit_sha1": {r: _sha1(fit_dir / f"{r}.npz") for r in ROLES},
            "pose_keys": int(len(ts)), "video_frames": NV, "grid_frames": F, "transfer_check": chk,
            "front": front["info"], "orbit": orbit["info"], "orbit_window": {k: win[k] for k in win if k != "all_spins"},
            "all_spins": win["all_spins"], "phone_hi": {**PHONE_HI, "pad_px": ph["pad_px"]},
            "camera_not_ok_frames": [int(i) for i in np.nonzero(~np.asarray(tr.ok, bool))[0]],
            "groups": vj["groups"], "passes": PASSES, "scene_key": vj["scene_key"], "refs": refs,
            "bytes": int(sum(p.stat().st_size for p in INPUT.rglob("*") if p.is_file())),
            "seconds": round(time.time() - t0, 1)}
    (INPUT / "prep.json").write_text(json.dumps(info, indent=1, default=float))
    log(f"prep: {json.dumps({k: info[k] for k in ('pose_keys', 'groups', 'bytes', 'seconds')})}")
    log(f"front: {json.dumps(front['info'], default=float)}")
    log(f"orbit: {json.dumps(orbit['info'], default=float)}")
    log(f"transfer: {json.dumps(chk)}")
    return info


# ================================================================================================ pod job
def job(max_minutes: float = 50) -> Path:
    rel = lambda p: _ws(p)[len("dancecap/"):]  # noqa: E731 - cwd of the commands is dancecap/
    groups = [g for p in PASSES for g in p]
    vj = json.loads((INPUT / "views.json").read_text())
    last = {g: f"{vj['groups'][g]['n'] - 1:05d}.png" for g in groups}
    d = {"schema": "dancecap.pod_job/1", "name": JOB_NAME, "take": TAKE, "max_minutes": max_minutes, "needs": [],
         "inputs": [_ws(INPUT) + "/"],
         "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
         "env": {"LHMPP_COMMIT": "906b5d9fb967ab42efb92f6fa55bf22cac86b653",
                 "LHMPP_ROOT": "/workspace/lhmpp_cand/LHM-plusplus", "LHMPP_VENV": "/workspace/lhmpp_cand/venv",
                 "LHMPP_MODEL": "LHMPP-700M-PixelShuffle", "TORCH_HOME": "/workspace/lhmpp_cand/torch_home",
                 "HF_HOME": "/workspace/lhmpp_cand/hf_home"},
         "commands": [
             {"name": "install", "timeout_min": 35, "cmd": "bash remote/cand_lhmpp/install.sh"},
             {"name": "render", "timeout_min": 40,
              "cmd": f"/workspace/lhmpp_cand/venv/bin/python remote/cand_lhmpp/lhmpp_render_pod.py --input "
                     f"{rel(INPUT)} --out {rel(POD_OUT)} --renders {rel(RENDERS)}"}],
         "outputs": [_ws(POD_OUT) + "/"] + [_ws(RENDERS / g) + "/" for g in groups],
         "verify": [_ws(POD_OUT / "lhmpp_render_pod.json"), _ws(POD_OUT / "gs" / "lead_cano.ply"),
                    _ws(POD_OUT / "gs" / "follow_cano.ply")] + [_ws(RENDERS / g / last[g]) for g in groups],
         "after_download": ["python -m dancecap.mono4d.jl_realism_render import --session <sid>"],
         "notes": ["LHM++ (LHMPP-700M-PixelShuffle, CC BY-NC 4.0 weights) avatars of John and Lula (take john_lula) from "
                   "the preview's lhmpp_video8_photofull crops (John: whole visible photo mask first + 8 video crops; "
                   "Lula: 8 video crops + photo crop), shape ref, posed with LHM++'s own animation by the MAMMA + SAM 3D "
                   "Body fit (refined, audit-patched) for the whole dance: tutorial front 1920x1080, the phone path "
                   "538x960 + 1080x1920, a 25 s orbit 1080x1080; weights stay on the pod; crops go to our own pod only "
                   "(owner consent 2026-10-19)"]}
    JOB_PATH.write_text(json.dumps(d, indent=1), newline="\n")
    log(f"job -> {JOB_PATH}")
    return JOB_PATH


# ================================================================================================ import
def _session_meta(sid: str | None) -> dict:
    root = PACKAGE_ROOT / "work" / "runpod" / "sessions"
    if sid is None:
        cands = sorted((p for p in root.glob("*") if (p / "result.json").exists()
                        and f'"{JOB_NAME}"' in (p / "result.json").read_text(errors="ignore")),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            return {}
        sid = cands[-1].name
    p = root / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text())
    pod = r.get("pod") or {}
    j = (r.get("jobs") or {}).get(JOB_NAME) or {}
    steps = {x["step"]: x["seconds"] for x in r.get("timings", []) if "step" in x}
    return {"session": sid, "pod_id": pod.get("id"), "gpu": pod.get("gpu"), "cost_per_hr": pod.get("cost_per_hr"),
            "cost_usd": pod.get("ledger_cost"), "pod_minutes": round((pod.get("lifetime_s") or 0) / 60, 1),
            "job_status": j.get("status"), "commands": j.get("commands"), "job_seconds": j.get("seconds"),
            "download_mb": j.get("download_mb"), "steps_s": steps}


def import_run(sid: str | None = None) -> dict:
    from PIL import Image
    vj = json.loads((INPUT / "views.json").read_text())
    prep_ = json.loads((INPUT / "prep.json").read_text())
    podj = POD_OUT / "lhmpp_render_pod.json"
    pod = json.loads(podj.read_text()) if podj.exists() else {}
    groups = {}
    for g, gi in vj["groups"].items():
        files = sorted((RENDERS / g).glob("*.png"))
        bad = []
        for p in files[:: max(1, len(files) // 200)] + files[-1:]:
            with Image.open(p) as im:
                if im.size != (gi["W"], gi["H"]) or im.mode != "RGBA":
                    bad.append(p.name)
        idx = {int(p.stem) for p in files}
        groups[g] = {"planned": gi["n"], "on_disk": len(files), "missing": [i for i in range(gi["n"]) if i not in idx][:20],
                     "bad_size_or_mode_sampled": bad[:10], "mb": round(sum(p.stat().st_size for p in files) / 1e6, 1),
                     "W": gi["W"], "H": gi["H"]}
    plys = {p.name: {"bytes": p.stat().st_size} for p in sorted((POD_OUT / "gs").glob("*.ply"))} \
        if (POD_OUT / "gs").exists() else {}
    for name in plys:
        with open(POD_OUT / "gs" / name, "rb") as f:
            head = f.read(4096).split(b"end_header")[0].decode("ascii", "replace")
        n = [int(l.split()[-1]) for l in head.splitlines() if l.startswith("element vertex")]
        plys[name]["gaussians"] = n[0] if n else None
    ses = _session_meta(sid)
    rep = {"what": "john_lula realism stage 3: LHM++ avatars (preview photofull inputs, shape ref) posed by the MAMMA + "
                   "SAM 3D Body fit with LHM++'s own animation, whole dance; RGBA renders", "fit_dir": prep_["fit_dir"],
           "fit_sha1": prep_["fit_sha1"], "session": ses, "groups": groups, "plys": plys,
           "dancers": pod.get("dancers"), "pod_groups": pod.get("groups"), "pod_timings": pod.get("timings"),
           "pod_errors": pod.get("errors"), "stopped_early": pod.get("stopped_early"), "versions": pod.get("versions"),
           "front": prep_["front"], "orbit": prep_["orbit"], "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    LH.mkdir(parents=True, exist_ok=True)
    (LH / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    counts = {g: f"{v['on_disk']}/{v['planned']}" for g, v in groups.items()}
    log(f"import: {json.dumps(counts)}")
    log(f"plys: {json.dumps(plys)}; session: {json.dumps(ses, default=str)}")
    return rep


# ================================================================================================ videos
@functools.lru_cache(maxsize=16)
def _font(px: int):
    from PIL import ImageFont
    for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(f, px)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _text(img: np.ndarray, lines: list[tuple[str, tuple[int, int], int]], col=(255, 255, 255)) -> np.ndarray:
    """Draw anti-aliased text (with a soft dark shadow) on a BGR uint8 image; lines = [(text, (x, y), px)]."""
    from PIL import Image, ImageDraw
    out = img.copy()
    H, W = img.shape[:2]
    for (s, (x, y), px) in lines:  # draw each line inside its own small box (whole-frame PIL round trips are slow)
        f = _font(px)
        x0, y0 = max(0, x - 4), max(0, y - 4)
        x1, y1 = min(W, int(x + 0.62 * px * len(s) + 12)), min(H, int(y + 1.5 * px + 8))
        im = Image.fromarray(np.ascontiguousarray(out[y0:y1, x0:x1, ::-1]))
        dr = ImageDraw.Draw(im)
        for dx, dy in ((1, 1), (2, 2)):
            dr.text((x - x0 + dx, y - y0 + dy), s, font=f, fill=(0, 0, 0))
        dr.text((x - x0, y - y0), s, font=f, fill=col)
        out[y0:y1, x0:x1] = np.asarray(im)[:, :, ::-1]
    return out


def _ffmpeg(out: Path, W: int, H: int, a0: float, dur: float, copy_audio: bool) -> subprocess.Popen:
    out.parent.mkdir(parents=True, exist_ok=True)
    src = audio_source()
    aud = ["-ss", f"{a0:.4f}", "-t", f"{dur:.4f}", "-i", str(src)] if a0 > 0 else ["-t", f"{dur:.4f}", "-i", str(src)]
    acodec = ["-c:a", "copy"] if copy_audio else ["-c:a", "aac", "-b:a", "160k"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-framerate",
           "30", "-i", "-", *aud, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-r", "30", *acodec, "-shortest", "-movflags", "+faststart",
           str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _read_rgba(p: Path) -> np.ndarray | None:
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return im if im is not None and im.ndim == 3 and im.shape[2] == 4 else None


def _encode(name: str, n_frames: int, make, W: int, H: int, a0: float, copy_audio: bool, workers: int = 3) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    out = VIDEOS / name
    t0 = time.time()
    p = _ffmpeg(out, W, H, a0, n_frames / FPS, copy_audio)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, fr in enumerate(ex.map(make, range(n_frames))):
            assert fr.shape == (H, W, 3), (name, n, fr.shape)
            p.stdin.write(fr.tobytes())
            if n % 300 == 0:
                log(f"{name}: frame {n}/{n_frames} ({time.time() - t0:.0f} s)")
    p.stdin.close()
    rc = p.wait()
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                         "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration,sample_rate",
                         "-of", "json", str(out)], capture_output=True, text=True)
    info = {"file": _ws(out), "rc": rc, "frames": n_frames, "seconds": round(time.time() - t0, 1),
            "audio_from_s": a0, "bytes": out.stat().st_size if out.exists() else None,
            "ffprobe": json.loads(pr.stdout or "{}").get("streams")}
    log(f"{name}: {json.dumps(info)}")
    return info


def videos(only: tuple[str, ...] = ("front", "sbs", "orbit")) -> dict:
    from . import jl_lhmpp_preview as JP
    with np.load(INPUT / "views.npz") as z:
        V = {k: z[k] for k in z.files}
    prep_ = json.loads((INPUT / "prep.json").read_text())
    fits = {r: load_fit(r, WORKSPACE / prep_["fit_dir"]) for r in ROLES}
    T = np.asarray(fits["lead"]["t"], float)
    Jg = {r: np.asarray(fits[r]["joints"], float) for r in ROLES}
    res = {}
    if "front" in only:
        W, H = FRONT["W"], FRONT["H"]
        F = len(V["front_key"])

        def make_front(k):
            bg = background(V["front_K"][k], V["front_R"][k], V["front_tvec"][k], W, H,
                            shadow_points({r: Jg[r][k] for r in ROLES}))
            fr = composite(bg, _read_rgba(RENDERS / "front" / f"{k:05d}.png"))
            return _text(fr, [(CAPTION, (28, H - 52), 26)])
        res["jl_tutorial_front.mp4"] = _encode("jl_tutorial_front.mp4", F, make_front, W, H, 0.0, True)
    if "orbit" in only:
        W, H = ORBIT["W"], ORBIT["H"]
        N = len(V["orbit_key"])
        ks = V["orbit_key"] - len(V["phone_key"])
        a0 = float(T[ks[0]])

        def make_orbit(n):
            k = int(ks[n])
            bg = background(V["orbit_K"][n], V["orbit_R"][n], V["orbit_tvec"][n], W, H,
                            shadow_points({r: Jg[r][k] for r in ROLES}))
            fr = composite(bg, _read_rgba(RENDERS / "orbit" / f"{n:05d}.png"))
            return _text(fr, [(CAPTION, (22, H - 46), 22), (f"360 deg orbit, {T[ks[0]]:.1f}-{T[ks[-1]]:.1f} s", (22, 18), 22)])
        res["jl_orbit.mp4"] = _encode("jl_orbit.mp4", N, make_orbit, W, H, a0, False)
    if "sbs" in only:
        Wc, Hc = 1920, 1080
        ts_v = np.asarray(V["phone_t"], float)
        NV = len(ts_v)
        n_out = len(T)
        pw = int(round(538 * Hc / 960))
        x_l, x_r = Wc // 2 - pw - 6, Wc // 2 + 6
        pad = int(round(prep_["phone_hi"]["pad_px"]))
        Kh, Rh, th = V["phone_hi_K"], V["phone_hi_R"], V["phone_hi_tvec"]
        Hh, Wh = PHONE_HI["H"], PHONE_HI["W"]

        def make_sbs(n):
            t = n / FPS
            i = int(np.clip(np.argmin(np.abs(ts_v - t)), 0, NV - 1))
            canvas = np.full((Hc, Wc, 3), 22, np.uint8)
            real = cv2.imread(str(JP.frame_path(i)))
            canvas[:, x_l:x_l + pw] = cv2.resize(real, (pw, Hc), interpolation=cv2.INTER_LANCZOS4)
            Jt = {r: interp_many(fits[r], [ts_v[i]])["joints"][0] for r in ROLES}
            bg = background(Kh[i], Rh[i], th[i], Wh, Hh, shadow_points(Jt))
            fr = composite(bg, _read_rgba(RENDERS / "phone_hi" / f"{i:05d}.png"))[:, pad:Wh - pad]
            canvas[:, x_r:x_r + pw] = cv2.resize(fr, (pw, Hc), interpolation=cv2.INTER_AREA)
            return _text(canvas, [("real phone video", (x_l + 12, 10), 24),
                                  ("3D splats, same camera path", (x_r + 12, 10), 24),
                                  ("John & Lula", (40, 60), 40), ("Jack & Jill", (40, 112), 30),
                                  ("3D splats from one phone", (40, 160), 26),
                                  ("LHM++ avatars posed by", (40, 230), 20), ("MAMMA + SAM 3D Body", (40, 258), 20),
                                  (f"t = {t:5.2f} s", (x_r + pw + 30, Hc - 60), 24)])
        res["jl_side_by_side.mp4"] = _encode("jl_side_by_side.mp4", n_out, make_sbs, Wc, Hc, 0.0, True)
    old = {}
    rp = VIDEOS / "videos.json"
    if rp.exists():
        old = json.loads(rp.read_text())
    old.update(res)
    rp.write_text(json.dumps(old, indent=1))
    return res


# ================================================================================================ sheets
def sheets() -> dict:
    """Contact sheets read back from the encoded MP4s + phone-path alignment (splats over the real frame): LOOK."""
    from . import jl_lhmpp_preview as JP
    SHEETS.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, cols, tw in (("jl_tutorial_front.mp4", 3, 640), ("jl_side_by_side.mp4", 3, 640), ("jl_orbit.mp4", 4, 420)):
        p = VIDEOS / name
        if not p.exists():
            continue
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tiles = []
        for q in np.linspace(5, n - 6, cols * 3).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(q))
            ok, fr = cap.read()
            if not ok:
                continue
            th = int(round(fr.shape[0] * tw / fr.shape[1]))
            fr = cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA)
            _label(fr, f"{q / FPS:.2f}s", (8, th - 10), 0.6)
            tiles.append(fr)
        cap.release()
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
        sp = SHEETS / f"sheet_{p.stem}.jpg"
        cv2.imwrite(str(sp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out[name] = {"sheet": _ws(sp), "frames": n}
    tiles = []
    for i in np.linspace(60, 2580, 8).round().astype(int):
        real = cv2.imread(str(JP.frame_path(i)))
        rg = _read_rgba(RENDERS / "phone" / f"{i:05d}.png")
        if rg is None:
            continue
        a = rg[:, :, 3:4] / 255.0 * 0.75
        ov = (real * (1 - a) + rg[:, :, :3] * a).astype(np.uint8)
        _label(ov, f"f{i}", (6, 22), 0.6)
        tiles.append(np.concatenate([real, ov], 1))
    if tiles:
        sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        sheet = cv2.resize(sheet, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(SHEETS / "align_phone_path.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out["align"] = _ws(SHEETS / "align_phone_path.jpg")
    tt = [cv2.imread(str(POD_OUT / "turntable" / f"{r}.png")) for r in ROLES]
    if all(x is not None for x in tt):
        cv2.imwrite(str(SHEETS / "canonical_turntables.jpg"), np.concatenate(tt, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out["turntables"] = _ws(SHEETS / "canonical_turntables.jpg")
    log(json.dumps(out))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.jl_realism_render")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--fit", default=str(FIT_DIR))
    p = sub.add_parser("job")
    p.add_argument("--max-minutes", type=float, default=50)
    p = sub.add_parser("import")
    p.add_argument("--session")
    p = sub.add_parser("videos")
    p.add_argument("--only", default="front,sbs,orbit")
    sub.add_parser("sheets")
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        prep(Path(a.fit))
    elif a.cmd == "job":
        job(a.max_minutes)
    elif a.cmd == "import":
        import_run(a.session)
    elif a.cmd == "videos":
        videos(tuple(x for x in a.only.split(",") if x))
    elif a.cmd == "sheets":
        sheets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
