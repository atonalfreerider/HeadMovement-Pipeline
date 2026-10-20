r"""Appearance comparison of the dancer renderers (2026-10-19): ONE camera / time spec that every renderer follows, so
the owner can look at picture side-by-sides and decide between a textured avatar and moving 3D Gaussian splats.

Owner (2026-10-19): "we are going for realism and we need to decide between avatar and 3d gauss splat moving. the pose
can still be MAMMA or better, but the rendering needs to be improved from the avatar models we are currently using. I
need to see picture side-by-sides".

Renderers that follow the spec (each in its own stage; this module only writes the spec and, later, composes sheets):
    avatar   the CURRENT shipped look, rendered offline: unlit SMPL-X + the footage-baked albedo (avatar_texture.render,
             avatar/<role>_*); the Unity build adds hair cards with tip bloom, a portrait face and 35 % scene shading
             (HM_AvatarLit _LightInfluence 0.35) - every sheet says "offline render of
             the shipped texture; Unity adds hair cards"
    lhmpp    LHM++ one-phone (06) Gaussian avatars posed with the reference fit (cand_lhmpp.py, remote/cand_lhmpp/)
    ags      our multi-view avatar-guided splats (dancer_splats/, trained on take larissa_kadu = the same videos on
             other clocks / another world frame: views.json cross_take says how to render the same instants)

    python -m dancecap.mono4d.appearance_compare spec [--take larissa_kadu_38] [--other larissa_kadu]
        -> work/<take>/mono4d/compare/views.json (+ README.md, spec_check/*.jpg: the reference joints drawn on every
           real view, the novel cameras' framing as skeletons)
    python -m dancecap.mono4d.appearance_compare compose [--steps sheets,metrics,jev,videos,check] [--allow-upload]
        -> compare/sheet_*.jpg, clip_vr_seat.mp4, clip_phone03.mp4, metrics.json/.md, jev.json, compose.json
           (appearance_compose.py; the Jev realism judge in appearance_jev.py)

views.json (schema dancecap.mono4d.appearance_compare.spec/1), all on the take's reference clock (sync-corrected
seconds = frames.json t + the phone's sync delta, the clock of smplx/ and mamma_dyn.load_tracks) and in its world
(Y-up metres, floor y = 0); cameras OpenCV (x_cam = R X + tvec, +x right, +y down, +z forward), K pinhole with pixel
centres at integers; k1 = the real phone's SIMPLE_RADIAL term (renders are pinhole; dancer_splats.data.undistort the real
frame, or ignore k1: <= ~7 px at the corners of phone 02); boxes [x0, y0, x1, y1] in native pixels of the canonical
frame, x1/y1 exclusive.
    instants      6 grid instants inside AGS's trained window (58.0-61.5 s), the AGS review moments mapped to this clock,
                  incl. the follow's three fast head turns (checked on the fit: head angular-speed peaks)
    phones        per phone: native size, fps, high_res, azimuth around the couple over the clip
    phone_survey  per instant x phone: both dancers' mask area / joint visibility / border cut (the high-res pick)
    real_views    per instant: phone 03 (held out by AGS and LHM++) + the high-res phones; the canonical frame nearest
                  the instant (t = that frame's reference time = the time to pose the bodies at), K, R, tvec, k1, size,
                  crop = union of both dancers' masks + 10 % of its longer side; seen_by (which renderer saw the frame /
                  phone); lk = the same physical frame in the other take (frame, its own reference time, its camera)
    novel         turntable (12 views, 1080 x 1080) and the fixed VR seat (1080 x 1080, 55 deg vertical FOV); each view
                  also mapped into the other take's world (lk) with the body-fitted Sim3
    clip          105 grid frames 58.0-61.5 s: the VR seat (static) and phone 03's own moving track (real frames)
    closeups      per instant on the high-res phones: a 32 cm face square per dancer, a 60 cm hair square from the
                  follow's crown down (AGS's evaluate.crop_box convention), best_cam = the better-facing phone
    cross_take    clock offset and Sim3 world_<take> -> world_<other> (Umeyama on camera centres AND on the two fits'
                  joints) with residuals
Likeness data (frames, renders) stays in work/; nothing here is committed.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT

SCHEMA = "dancecap.mono4d.appearance_compare.spec/1"
ROLES = ("lead", "follow")
PID = {"lead": "01", "follow": "02"}
GRID_HZ = 30.0
WINDOW = (58.0, 61.5)                                   # AGS's trained window (larissa_kadu reference seconds)
AGS_REVIEW_LK = (58.29107, 59.19104, 59.89102, 60.391, 60.89099, 61.4243)  # bundle views review=True (cam 03, lk clock)
HEAD_TURNS_LK = (59.18, 60.88, 61.41)                   # the follow's fast head turns (dancer_splats.review.TURNS)
HELDOUT_CAM = "03"
FACE_HALF_M = 0.16                                      # = dancer_splats.evaluate.FACE_HALF_M (32 cm square)
HAIR_HALF_M, HAIR_DY_M = 0.30, 0.18                     # = HAIR_HALF_M, the hair close-up's 0.18 m shift down
SKULL_OFFSET = np.array([0.0, 0.07, 0.01])              # skull centre = J15 + R_head @ this (evaluate.head_points)
CROP_PAD = 0.10
TURNTABLE = {"n": 12, "radius_m": 3.0, "height_m": 1.6, "look_y_m": 1.0, "size": 1080, "vfov_deg": 50.0}
VR_SEAT = {"dist_m": 2.5, "height_m": 1.6, "look_y_m": 1.0, "size": 1080, "vfov_deg": 55.0}
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
HEAD_CHAIN = (0, 3, 6, 9, 12, 15)
AVATAR_CAPTION = "offline render of the shipped texture; Unity adds hair cards"


def compare_dir(take: str) -> Path:
    return PACKAGE_ROOT / "work" / take / "mono4d" / "compare"


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        return str(p)


def _r(x, n=6):
    return np.round(np.asarray(x, float), n).tolist()


# ------------------------------------------------------------------------------------------------ geometry
def rodrigues(aa: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle -> (N, 3, 3)."""
    aa = np.asarray(aa, float).reshape(-1, 3)
    th = np.linalg.norm(aa, axis=1)
    k = aa / np.maximum(th, 1e-12)[:, None]
    Kx = np.zeros((len(aa), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx[:, 1, 0], Kx[:, 2, 0], Kx[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * Kx + (1 - c) * Kx @ Kx


def umeyama(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity Y ~ s R X + t (least squares)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = float(np.trace(np.diag(D) @ E) / (Xc ** 2).sum(1).mean())
    return s, R, my - s * R @ mx


def sim3_apply(sim: dict, X: np.ndarray) -> np.ndarray:
    return float(sim["s"]) * np.asarray(X, float) @ np.asarray(sim["R"]).T + np.asarray(sim["t"])


def map_camera(R: np.ndarray, tvec: np.ndarray, sim: dict) -> tuple[np.ndarray, np.ndarray]:
    """A camera of world A (x_cam = R X_A + tvec) expressed in world B = s Ra X_A + ta: same image, same K."""
    Rs, s, ts = np.asarray(sim["R"]), float(sim["s"]), np.asarray(sim["t"])
    R2 = np.asarray(R) @ Rs.T
    return R2, s * np.asarray(tvec) - R2 @ ts


def centre(R, tvec) -> np.ndarray:
    return -np.asarray(R).T @ np.asarray(tvec)


def look_at(C, target, f: float, W: int, H: int) -> dict:
    """Pinhole at C looking at target, world Y up, image x right / y down (= dancer_splats.review.look_at)."""
    C, target = np.asarray(C, float), np.asarray(target, float)
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    return {"K": K, "R": R, "tvec": -R @ C, "W": int(W), "H": int(H)}


def project(P, K, R, tvec, k1: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    from ..dancer_splats.data import project as _p
    return _p(np.asarray(P, float).reshape(-1, 3), np.asarray(K, float), np.asarray(R, float), np.asarray(tvec, float),
              float(k1))


def azimuth(P, c) -> float:
    """Azimuth about +Y of point P seen from centre c: atan2(dx, dz) (0 = +Z, 90 deg = +X), degrees in [0, 360)."""
    return float(math.degrees(math.atan2(P[0] - c[0], P[2] - c[2])) % 360.0)


# ------------------------------------------------------------------------------------------------ the reference fit
class Fit:
    """smplx/<role>.npz: grid t, 55 joints, head global rotation (FK chain 0-3-6-9-12-15)."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=True)
        self.path = Path(path)
        self.t = np.asarray(z["t"], float)
        self.J = np.asarray(z["joints"], float)
        go, bp = np.asarray(z["global_orient"], float), np.asarray(z["body_pose"], float).reshape(len(self.t), 21, 3)
        R = rodrigues(go).reshape(-1, 3, 3)
        for j in HEAD_CHAIN[1:]:
            R = R @ rodrigues(bp[:, j - 1]).reshape(-1, 3, 3)
        self.R_head = R

    def _lerp(self, t: float) -> tuple[int, float]:
        k = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        a = float(np.clip((t - self.t[k]) / (self.t[k + 1] - self.t[k]), 0.0, 1.0))
        return k, a

    def joints(self, t: float) -> np.ndarray:
        k, a = self._lerp(t)
        return (1 - a) * self.J[k] + a * self.J[k + 1]

    def head_rot(self, t: float) -> np.ndarray:
        k, a = self._lerp(t)
        U, _, Vt = np.linalg.svd((1 - a) * self.R_head[k] + a * self.R_head[k + 1])
        return U @ Vt

    def skull(self, t: float) -> np.ndarray:
        return self.joints(t)[15] + self.head_rot(t) @ SKULL_OFFSET

    def fk_check(self) -> dict:
        """The head FK must reproduce the stored eye joints rigidly: spread of (eyes - J15) in the head frame."""
        d = np.einsum("tji,tj->ti", self.R_head, 0.5 * (self.J[:, 23] + self.J[:, 24]) - self.J[:, 15])
        return {"eye_offset_head_frame_m": _r(np.median(d, 0), 4), "spread_mm": round(1e3 * float(np.std(d, 0).max()), 2)}


def head_turns(fit: Fit, window: tuple[float, float]) -> list[dict]:
    """Angular speed of the head (deg/s, central differences) and its local maxima inside the window."""
    R = fit.R_head
    dR = np.einsum("tij,tkj->tik", R[2:], R[:-2])
    ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    speed = np.r_[np.nan, ang / (fit.t[2:] - fit.t[:-2]), np.nan]
    sel = np.nonzero((fit.t >= window[0]) & (fit.t <= window[1]))[0]
    return [{"k": int(k), "t": float(fit.t[k]), "deg_s": float(speed[k])} for k in sel
            if np.isfinite(speed[k]) and speed[k] >= np.nanmax(speed[max(k - 2, 0):k + 3])], speed  # +-2 frames


# ------------------------------------------------------------------------------------------------ frames / tracks
def frames_meta(work: Path, cam: str) -> dict:
    return json.loads((work / "frames" / cam / "frames.json").read_text())


def load_take(take: str, cams: list[str] | None = None):
    from .. import mamma_dyn as md
    work = PACKAGE_ROOT / "work" / take
    tracks = md.load_tracks(work, cams)
    return work, tracks, md.read_sync_offsets(work)


def nearest_frame(tr, t: float) -> int:
    return int(np.argmin(np.abs(tr.t - t)))


def read_mask(work: Path, cam: str, i: int, role: str, shape) -> np.ndarray | None:
    p = work / "dancers" / cam / "masks" / f"mask_{i:04d}_{PID[role]}.png"
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != tuple(shape):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def _box(m: np.ndarray | None):
    if m is None or not m.any():
        return None
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def pad_box(b, W: int, H: int, pad: float = CROP_PAD):
    p = pad * max(b[2] - b[0], b[3] - b[1])
    return [int(max(0, math.floor(b[0] - p))), int(max(0, math.floor(b[1] - p))),
            int(min(W, math.ceil(b[2] + p))), int(min(H, math.ceil(b[3] + p)))]


def mask_stats(work: Path, tr, i: int, fits: dict, t: float) -> dict:
    """Both dancers in one canonical frame: mask area, projected body joints inside the own (dilated) mask, border
    cut, overlap with the partner's mask; union crop box."""
    W, H = tr.width, tr.height
    out, boxes = {}, []
    masks = {r: read_mask(work, tr.name, i, r, (H, W)) for r in ROLES}
    tol = max(3, int(0.015 * max(W, H)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    for r in ROLES:
        m = masks[r]
        b = _box(m)
        boxes.append(b)
        J = fits[r].joints(t)[:22]
        uv, z = project(J, tr.K[i], tr.R[i], tr.tvec[i], tr.k1[i])
        if m is None or b is None:
            out[r] = {"area_px": 0, "area_frac": 0.0, "vis": 0.0, "cut": True, "box": None}
            continue
        md = cv2.dilate(m.astype(np.uint8), ker) > 0
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inside = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(len(J), bool)
        hit[inside] = md[v[inside], u[inside]]
        other = masks["follow" if r == "lead" else "lead"]
        cut = b[0] <= 1 or b[1] <= 1 or b[2] >= W - 1 or b[3] >= H - 1
        out[r] = {"area_px": int(m.sum()), "area_frac": round(float(m.mean()), 4), "vis": round(float(hit.mean()), 3),
                  "joints_in_image": round(float(inside.mean()), 3), "cut": bool(cut), "box": b,
                  "partner_overlap": round(float((m & other).sum() / max(m.sum(), 1)), 3) if other is not None else 0.0}
    u = None
    for b in boxes:
        if b is not None:
            u = b if u is None else [min(u[0], b[0]), min(u[1], b[1]), max(u[2], b[2]), max(u[3], b[3])]
    both = all(out[r]["area_frac"] >= 0.002 and out[r]["vis"] >= 0.6 and not out[r]["cut"] for r in ROLES)
    score = min(out[r]["vis"] * (0.5 if out[r]["cut"] else 1.0) for r in ROLES) * min(
        1.0, min(out[r]["area_frac"] for r in ROLES) / 0.01)
    return {"roles": out, "union_box": u, "crop": pad_box(u, W, H) if u else None, "both_well": bool(both),
            "score": round(float(score), 3)}


def pts_index(meta: dict) -> dict:
    return {round(float(f["pts"]), 4): i for i, f in enumerate(meta["frames"])}


# ------------------------------------------------------------------------------------------------ cross-take
def cross_take(take: str, other: str, cams: list[str], fits: dict, tr38: dict,
               window: tuple[float, float]) -> tuple[dict, dict]:
    """The same source videos ingested as <other>: per phone the reference-time offset of identical frames (by pts),
    Sim3 world_take -> world_other from the camera centres of identical frames (Umeyama) and from the two fits' joints
    (time shift searched), and the analytic one from the two world_from_sfm.json (valid only if the SfM frames are
    identical). Returns (record, context with the other take's tracks / fits / pts maps)."""
    w38 = PACKAGE_ROOT / "work" / take
    wlk, trlk, _ = load_take(other, cams)
    per, A, B, per_res = {}, [], [], {}
    pmap = {}
    for c in cams:
        m38, mlk = frames_meta(w38, c), frames_meta(wlk, c)
        p38 = pts_index(m38)
        pmap[c] = {"38_to_lk": {}, "lk_pts": pts_index(mlk)}
        d, pairs = [], []
        for j, f in enumerate(mlk["frames"]):
            i = p38.get(round(float(f["pts"]), 4))
            if i is None:
                continue
            pmap[c]["38_to_lk"][i] = j
            d.append(tr38[c].t[i] - trlk[c].t[j])
            if tr38[c].ok[i] and trlk[c].ok[j]:
                pairs.append((centre(tr38[c].R[i], tr38[c].tvec[i]), centre(trlk[c].R[j], trlk[c].tvec[j])))
        d = np.asarray(d)
        per[c] = {"matched_frames": int(len(d)), "other_frames": len(mlk["frames"]),
                  "offset_ms_median": round(1e3 * float(np.median(d)), 2) if len(d) else None,
                  "offset_ms_range": [round(1e3 * float(d.min()), 2), round(1e3 * float(d.max()), 2)] if len(d) else None,
                  "same_raw_clock_ms": round(1e3 * float(np.median([m38["frames"][i]["t"] - mlk["frames"][j]["t"]
                                                                     for i, j in pmap[c]["38_to_lk"].items()])), 2)}
        per_res[c] = pairs
        A += [p[0] for p in pairs]
        B += [p[1] for p in pairs]
    A, B = np.asarray(A), np.asarray(B)
    s, R, t = umeyama(A, B)
    res = np.linalg.norm(s * A @ R.T + t - B, axis=1)
    cam_sim = {"s": s, "R": R.tolist(), "t": t.tolist(), "n": int(len(A)), "rms_m": round(float(np.sqrt((res ** 2).mean())), 4),
               "median_m": round(float(np.median(res)), 4), "p95_m": round(float(np.percentile(res, 95)), 4),
               "per_cam_median_m": {c: round(float(np.median([np.linalg.norm(s * R @ a + t - b) for a, b in per_res[c]])), 4)
                                    for c in cams if per_res[c]},
               "how": "Umeyama on the camera centres of the SAME source frames (matched by pts) of all phones, ok frames "
                      "of both takes, whole overlap"}
    # analytic (both from one SfM map?)
    a38 = json.loads((w38 / "cameras" / "world_from_sfm.json").read_text())
    alk = json.loads((wlk / "cameras" / "world_from_sfm.json").read_text())
    sa = alk["s"] / a38["s"]
    Ra = np.asarray(alk["R"]) @ np.asarray(a38["R"]).T
    ta = np.asarray(alk["t"]) - sa * Ra @ np.asarray(a38["t"])
    res_a = np.linalg.norm(sa * A @ Ra.T + ta - B, axis=1)
    # bodies: time shift + Sim3 on the two fits' 22 body joints inside the window
    flk = {r: Fit(wlk / "smplx" / f"{r}.npz") for r in ROLES}
    best = None
    for sh in np.arange(-0.040, 0.0401, 0.001):
        X, Y = [], []
        for r in ROLES:
            sel = (flk[r].t >= window[0]) & (flk[r].t <= window[1])
            for k in np.nonzero(sel)[0]:
                X.append(fits[r].joints(flk[r].t[k] + sh)[:22])
                Y.append(flk[r].J[k, :22])
        X, Y = np.concatenate(X), np.concatenate(Y)
        sb, Rb, tb = umeyama(X, Y)
        e = np.linalg.norm(sb * X @ Rb.T + tb - Y, axis=1)
        if best is None or e.mean() < best[0]:
            best = (float(e.mean()), float(sh), sb, Rb, tb, float(np.median(e)), float(np.percentile(e, 95)))
        if abs(sh) < 1e-9:
            e0 = float(e.mean())
    body_sim = {"s": best[2], "R": best[3].tolist(), "t": best[4].tolist(), "mean_m": round(best[0], 4),
                "median_m": round(best[5], 4), "p95_m": round(best[6], 4), "mean_m_at_zero_shift": round(e0, 4),
                "how": f"Umeyama on both fits' 22 body joints over {window[0]}-{window[1]} s of {other}'s grid, with "
                       f"{take}'s fit sampled at t_{other} + shift (shift searched -40..+40 ms in 1 ms)"}
    ang = lambda M: round(float(np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1)))), 3)  # noqa: E731
    offs = [v["offset_ms_median"] for v in per.values() if v["offset_ms_median"] is not None]
    rec = {
        "other": other,
        "same_sources": True,
        "clock": {
            "same_clock": False,
            "body_shift_ms": round(1e3 * best[1], 1),
            "rule": f"t_{other} = t_{take} - body_shift (novel views, the clip's VR seat); for a REAL frame use the "
                    f"other take's own reference time of the same frame (real_views[*].lk.t, clip.cam03.lk_t), "
                    f"which is what its fit and AGS were trained on",
            "per_phone": per,
            "per_phone_median_ms": round(float(np.median(offs)), 2),
            "why": f"{take} is on the studio-derived audio clocks (alignment_studio.json) + its own sync.py video offsets, "
                   f"{other} on alignment.json + its own: identical frames get reference times "
                   f"{min(offs):+.1f}..{max(offs):+.1f} ms apart per phone (median {np.median(offs):+.1f}); the two fits' "
                   f"motion lines up best at a {1e3 * best[1]:+.0f} ms shift (joint error {1e3 * best[0]:.1f} mm vs "
                   f"{1e3 * e0:.1f} at 0)",
        },
        "world": {
            "same_world": False,
            "sim3_take_to_other_cameras": {**cam_sim, "rot_deg": ang(R)},
            "sim3_take_to_other_bodies": {**body_sim, "rot_deg": ang(best[3])},
            "sim3_analytic_from_world_from_sfm": {"s": sa, "R": Ra.tolist(), "t": ta.tolist(), "rot_deg": ang(Ra),
                                                  "rms_on_camera_centres_m": round(float(np.sqrt((res_a ** 2).mean())), 4),
                                                  "note": "composition of the two world_from_sfm.json; its residual on "
                                                          "the camera centres says whether the SfM frames are identical"},
            "use": "novel cameras -> sim3_take_to_other_bodies (the couple lands where the other renderers put it); "
                   "real frames -> the other take's own camera of the same frame (real_views[*].lk)",
        },
    }
    ctx = {"work": wlk, "tracks": trlk, "fits": flk, "pmap": pmap, "body_shift": best[1],
           "body_sim": {"s": best[2], "R": best[3], "t": best[4]}}
    return rec, ctx


# ------------------------------------------------------------------------------------------------ seen-by
def seen_by_index(take: str, other: str) -> dict:
    """Which frames each renderer used: avatar bake views, LHM++ input crops, AGS bundle views (other take)."""
    w = PACKAGE_ROOT / "work"
    out = {"avatar": set(), "avatar_cams": [], "lhmpp_cam": None, "lhmpp": {}, "ags": {}}
    p = w / take / "avatar" / "report.json"
    if p.exists():
        rep = json.loads(p.read_text())
        out["avatar"] = {(v["cam"], int(v["i"])) for v in rep.get("views", [])}
        out["avatar_cams"] = rep.get("cams", [])
        out["avatar_step_s"] = (rep.get("config") or {}).get("step_s")
    p = w / take / "mono4d" / "lhmpp" / "report.json"
    if p.exists():
        out["lhmpp_cam"] = json.loads(p.read_text()).get("input_cam")
        out["lhmpp_holdout"] = json.loads(p.read_text()).get("holdout")
    p = w / take / "mono4d" / "lhmpp" / "input" / "refs.json"
    if p.exists():
        for role, sets in json.loads(p.read_text()).items():
            for s, d in sets.items():
                for f in d.get("frames", []):
                    out["lhmpp"].setdefault(int(f["i"]), []).append(f"{role}:{s}")
    p = w / other / "dancer_splats" / "bundle" / "views.json"
    if p.exists():
        out["ags"] = {(v["cam"], int(v["frame"])): {"bundle_view": int(v["i"]), "split": v["split"]}
                      for v in json.loads(p.read_text())}
    return out


def seen_by(idx: dict, cam: str, i: int, j_lk: int | None) -> dict:
    ags = idx["ags"].get((cam, j_lk)) if j_lk is not None else None
    ags_cams = {c for c, _ in idx["ags"]}
    return {
        "avatar": {"phone_in_bake": cam in idx["avatar_cams"], "frame_in_bake": (cam, i) in idx["avatar"]},
        "lhmpp": ("input phone" + (f", this frame is an input crop ({', '.join(idx['lhmpp'][i])})" if i in idx["lhmpp"]
                                   else ", not an input crop")) if cam == idx["lhmpp_cam"] else "never seen (held out)",
        "ags": (f"{ags['split']} (bundle view {ags['bundle_view']})" if ags else
                ("phone held out (no frame in the bundle)" if cam == HELDOUT_CAM else
                 ("frame not in the bundle (training phone, a neighbour frame trained)" if cam in ags_cams else "no bundle"))),
    }


# ------------------------------------------------------------------------------------------------ spec
def _cam_rec(K, R, tvec, W, H, k1=0.0) -> dict:
    return {"K": _r(K), "R": _r(R, 9), "tvec": _r(tvec), "k1": float(k1), "W": int(W), "H": int(H),
            "C": _r(centre(R, tvec), 4)}


def _fov_f(H: int, vfov_deg: float) -> float:
    return (H / 2) / math.tan(math.radians(vfov_deg) / 2)


def _couple_points(fits: dict, t: float) -> np.ndarray:
    """Joints of both dancers + a crown point 0.20 m above each head joint + each ankle dropped to the floor: a
    conservative extent of the couple for framing checks."""
    P = []
    for r in ROLES:
        J = fits[r].joints(t)
        P += list(J[:22]) + [J[15] + np.array([0, 0.20, 0])] + [J[a] * [1, 0, 1] for a in (7, 8)]
    return np.asarray(P)


def _frame_margin(P, cam: dict) -> float:
    uv, z = project(P, cam["K"], cam["R"], cam["tvec"])
    if (z <= 0.1).any():
        return -1.0
    m = np.minimum.reduce([uv[:, 0], cam["W"] - 1 - uv[:, 0], uv[:, 1], cam["H"] - 1 - uv[:, 1]])
    return float(m.min() / cam["H"])


def build_spec(take: str = "larissa_kadu_38", other: str = "larissa_kadu", log=print) -> dict:
    work = PACKAGE_ROOT / "work" / take
    cams = sorted(p.stem for p in (work / "cameras").glob("*.npz"))
    fits = {r: Fit(work / "smplx" / f"{r}.npz") for r in ROLES}
    grid = fits["lead"].t
    assert np.allclose(grid, fits["follow"].t), "lead / follow grids differ"
    _, tracks, offs = load_take(take, cams)
    log(f"[compare] {take}: phones {cams}; grid {grid[0]:.4f}..{grid[-1]:.4f} s ({len(grid)} frames)")

    xt, ctx = cross_take(take, other, cams, fits, tracks, WINDOW)
    sh = ctx["body_shift"]
    log(f"[compare] clock: body shift {1e3 * sh:+.1f} ms; per phone {xt['clock']['per_phone_median_ms']:+.1f} ms median; "
        f"Sim3 cameras s {xt['world']['sim3_take_to_other_cameras']['s']:.5f} rms "
        f"{xt['world']['sim3_take_to_other_cameras']['rms_m']:.3f} m; bodies s "
        f"{xt['world']['sim3_take_to_other_bodies']['s']:.5f} mean {xt['world']['sim3_take_to_other_bodies']['mean_m']:.4f} m")

    # ---- instants: AGS review moments (other clock) -> this clock -> snapped to the grid
    turns_peaks, speed = head_turns(fits["follow"], (WINDOW[0], WINDOW[1]))
    inst = []
    for n, t_lk in enumerate(AGS_REVIEW_LK):
        t_req = t_lk + sh
        k = int(np.argmin(np.abs(grid - t_req)))
        turn = next((x for x in HEAD_TURNS_LK if abs(x - t_lk) < 0.03), None)
        rec = {"id": f"i{n}", "k": k, "t": float(grid[k]), "requested_t": round(t_req, 5),
               "snap_ms": round(1e3 * (grid[k] - t_req), 1), "ags_review_t_other": t_lk,
               "t_other": round(float(grid[k]) - sh, 5), "head_turn": turn is not None,
               "follow_head_deg_s": round(float(speed[k]), 1)}
        if turn is not None:
            near = [p for p in turns_peaks if abs(p["t"] - grid[k]) <= 0.2]
            pk = max(near, key=lambda p: p["deg_s"]) if near else None
            rec["head_turn_check"] = ({"peak_t": round(pk["t"], 5), "peak_deg_s": round(pk["deg_s"], 1),
                                       "frames_from_peak": int(pk["k"] - k)} if pk else {"peak_t": None})
        inst.append(rec)
    if not all(WINDOW[0] <= x["t"] <= WINDOW[1] for x in inst):
        raise SystemExit(f"[compare] an instant left the window {WINDOW}")

    # ---- phones
    clip_k = np.nonzero((grid >= WINDOW[0] - 1e-9) & (grid <= WINDOW[1] + 1e-9))[0]
    clip_t = grid[clip_k]
    pel_clip = np.mean([np.mean([fits[r].joints(t)[0] for t in clip_t], 0) for r in ROLES], 0)
    phones = {}
    for c in cams:
        tr = tracks[c]
        m = frames_meta(work, c)
        ii = [nearest_frame(tr, t) for t in clip_t]
        ok = [i for i in ii if tr.ok[i]]
        Cm = np.mean([centre(tr.R[i], tr.tvec[i]) for i in (ok or ii)], 0)
        phones[c] = {"W": tr.width, "H": tr.height, "fps": round(float(m.get("fps_median") or 0), 3),
                     "source": m.get("source"), "sync_delta_ms": round(1e3 * float((offs.get(c) or {}).get("delta", 0)), 2),
                     "azimuth_deg": round(azimuth(Cm, pel_clip), 2),
                     "dist_m": round(float(np.linalg.norm((Cm - pel_clip)[[0, 2]])), 3),
                     "C_mean": _r(Cm, 3), "k1": round(float(np.median(tr.k1)), 5),
                     "focal_px_median": round(float(np.median(tr.K[:, 0, 0])), 1)}
    side = {c: max(v["W"], v["H"]) for c, v in phones.items()}
    hires_all = sorted([c for c in cams if side[c] >= 1000], key=lambda c: -side[c])

    # ---- phone survey (all phones x instants) -> high-res pick
    survey = {}
    for x in inst:
        survey[x["id"]] = {}
        for c in cams:
            tr = tracks[c]
            i = nearest_frame(tr, x["t"])
            st = mask_stats(work, tr, i, fits, float(tr.t[i]))
            survey[x["id"]][c] = {"frame": i, "dt_ms": round(1e3 * (tr.t[i] - x["t"]), 1), "ok": bool(tr.ok[i]),
                                  "both_well": st["both_well"], "score": st["score"],
                                  **{r: {k: v for k, v in st["roles"][r].items() if k != "box"} for r in ROLES}}
    well = {c: sum(survey[x["id"]][c]["both_well"] for x in inst) for c in cams}
    if len(hires_all) > 2:
        hires = sorted(hires_all, key=lambda c: (-well[c], -np.mean([survey[x["id"]][c]["score"] for x in inst])))[:2]
    else:
        hires = hires_all
    log(f"[compare] high-res phones (>= 1000 px): {hires_all} -> {hires}; both dancers well per phone: {well}")

    idx = seen_by_index(take, other)
    trlk, pmap = ctx["tracks"], ctx["pmap"]

    def lk_block(c, i):
        j = pmap[c]["38_to_lk"].get(i)
        if j is None:
            return None
        t2 = trlk[c]
        return {"frame": int(j), "t": round(float(t2.t[j]), 5), "ok": bool(t2.ok[j]),
                "file": _rel(Path(t2.files[j])), **_cam_rec(t2.K[j], t2.R[j], t2.tvec[j], t2.width, t2.height, t2.k1[j])}

    # ---- real views
    real = []
    for x in inst:
        for c in [HELDOUT_CAM] + [h for h in hires if h != HELDOUT_CAM]:
            tr = tracks[c]
            i = nearest_frame(tr, x["t"])
            st = mask_stats(work, tr, i, fits, float(tr.t[i]))
            j = pmap[c]["38_to_lk"].get(i)
            real.append({
                "id": f"{x['id']}_c{c}", "instant": x["id"], "cam": c, "kind": "heldout" if c == HELDOUT_CAM else "hires",
                "frame": i, "file": _rel(Path(tr.files[i])), "t": round(float(tr.t[i]), 5),
                "t_raw": round(float(frames_meta(work, c)["frames"][i]["t"]), 5),
                "dt_ms": round(1e3 * (tr.t[i] - x["t"]), 1), "ok": bool(tr.ok[i]),
                **_cam_rec(tr.K[i], tr.R[i], tr.tvec[i], tr.width, tr.height, tr.k1[i]),
                "crop": st["crop"], "union_box": st["union_box"],
                "masks": {r: _rel(work / "dancers" / c / "masks" / f"mask_{i:04d}_{PID[r]}.png") for r in ROLES},
                "mask_stats": st["roles"], "both_well": st["both_well"],
                "seen_by": seen_by(idx, c, i, j), "lk": lk_block(c, i)})

    # ---- novel cameras
    x_tt = min(inst, key=lambda x: abs(x["t"] - (HEAD_TURNS_LK[0] + sh)))
    t_tt = x_tt["t"]
    pel = np.mean([fits[r].joints(t_tt)[0] for r in ROLES], 0)
    target = np.array([pel[0], TURNTABLE["look_y_m"], pel[2]])
    v03 = next(v for v in real if v["instant"] == x_tt["id"] and v["cam"] == HELDOUT_CAM)
    a0 = azimuth(np.asarray(v03["C"]), pel)
    S = TURNTABLE["size"]
    f_tt = _fov_f(S, TURNTABLE["vfov_deg"])
    P_tt = _couple_points(fits, t_tt)
    bsim = ctx["body_sim"]
    tt_views = []
    for n in range(TURNTABLE["n"]):
        a = math.radians(a0 + 360.0 * n / TURNTABLE["n"])
        C = np.array([pel[0] + TURNTABLE["radius_m"] * math.sin(a), TURNTABLE["height_m"],
                      pel[2] + TURNTABLE["radius_m"] * math.cos(a)])
        cam = look_at(C, target, f_tt, S, S)
        R2, t2 = map_camera(cam["R"], cam["tvec"], bsim)
        near = min(cams, key=lambda c: abs((phones[c]["azimuth_deg"] - math.degrees(a) + 180) % 360 - 180))
        tt_views.append({"id": f"tt{n:02d}", "azimuth_from_cam03_deg": round(360.0 * n / TURNTABLE["n"], 1),
                         "azimuth_deg": round(math.degrees(a) % 360, 2), **_cam_rec(cam["K"], cam["R"], cam["tvec"], S, S),
                         "frame_margin": round(_frame_margin(P_tt, cam), 3), "nearest_phone": near,
                         "lk": {"t": round(t_tt - sh, 5), "R": _r(R2, 9), "tvec": _r(t2), "C": _r(centre(R2, t2), 4)}})
    # VR seat: the middle of the largest angular gap between the phones around the couple (clip mean pelvis)
    az = sorted((phones[c]["azimuth_deg"], c) for c in cams)
    gaps = [((az[(n + 1) % len(az)][0] - az[n][0]) % 360, az[n], az[(n + 1) % len(az)]) for n in range(len(az))]
    g, (a_lo, c_lo), (a_hi, c_hi) = max(gaps, key=lambda q: q[0])
    a_seat = (a_lo + g / 2) % 360
    Cs = np.array([pel_clip[0] + VR_SEAT["dist_m"] * math.sin(math.radians(a_seat)), VR_SEAT["height_m"],
                   pel_clip[2] + VR_SEAT["dist_m"] * math.cos(math.radians(a_seat))])
    S2 = VR_SEAT["size"]
    # look-at height: the one that centres the couple's vertical extent over the clip (1.0 m leaves the near feet out
    # of a 55 deg frame at 2.5 m); searched 0.40-1.40 m in 1 cm
    P_clip = [_couple_points(fits, t) for t in clip_t]
    best_y = None
    for y in np.arange(0.40, 1.401, 0.01):
        cam = look_at(Cs, [pel_clip[0], y, pel_clip[2]], _fov_f(S2, VR_SEAT["vfov_deg"]), S2, S2)
        m = min(_frame_margin(P, cam) for P in P_clip)
        if best_y is None or m > best_y[1] + 1e-9:
            best_y = (float(y), m)
    look_y = round(best_y[0], 2)
    seat = look_at(Cs, [pel_clip[0], look_y, pel_clip[2]], _fov_f(S2, VR_SEAT["vfov_deg"]), S2, S2)
    margins = [_frame_margin(P, seat) for P in P_clip]
    m_at_1m = min(_frame_margin(P, look_at(Cs, [pel_clip[0], VR_SEAT["look_y_m"], pel_clip[2]],
                                           _fov_f(S2, VR_SEAT["vfov_deg"]), S2, S2)) for P in P_clip)
    dmin = min(float(np.linalg.norm(_couple_points(fits, t) - Cs, axis=1).min()) for t in clip_t)
    R2, t2 = map_camera(seat["R"], seat["tvec"], bsim)
    vr = {"id": "vr_seat", **_cam_rec(seat["K"], seat["R"], seat["tvec"], S2, S2), "azimuth_deg": round(a_seat, 2),
          "vfov_deg": VR_SEAT["vfov_deg"], "gap": {"between": [c_lo, c_hi], "deg": round(g, 2),
                                                    "nearest_phone_deg": round(g / 2, 2)},
          "all_gaps_deg": {f"{p[1][1]}-{p[2][1]}": round(p[0], 1) for p in gaps},
          "centre_pelvis_clip": _r(pel_clip, 4), "look_at": _r([pel_clip[0], look_y, pel_clip[2]], 4),
          "look_at_rule": "height that centres the couple's vertical extent over the clip (max of the min frame "
                          f"margin); at {VR_SEAT['look_y_m']} m the margin would be {m_at_1m:.3f} (near feet cut)",
          "focal_px": round(_fov_f(S2, VR_SEAT["vfov_deg"]), 2),
          "frame_margin_min": round(min(margins), 3), "closest_dancer_point_m": round(dmin, 3),
          "lk": {"R": _r(R2, 9), "tvec": _r(t2), "C": _r(centre(R2, t2), 4), "t_rule": "t_other = t - body_shift"}}

    # ---- clip
    tr3 = tracks[HELDOUT_CAM]
    c3 = []
    for t in clip_t:
        i = nearest_frame(tr3, t)
        j = pmap[HELDOUT_CAM]["38_to_lk"].get(i)
        c3.append({"frame": i, "file": _rel(Path(tr3.files[i])), "t": round(float(tr3.t[i]), 5),
                   "dt_ms": round(1e3 * (tr3.t[i] - t), 1), "ok": bool(tr3.ok[i]),
                   "K": _r(tr3.K[i]), "R": _r(tr3.R[i], 9), "tvec": _r(tr3.tvec[i]), "k1": float(tr3.k1[i]),
                   "lk_frame": j, "lk_t": round(float(trlk[HELDOUT_CAM].t[j]), 5) if j is not None else None,
                   "lk_R": _r(trlk[HELDOUT_CAM].R[j], 9) if j is not None else None,
                   "lk_tvec": _r(trlk[HELDOUT_CAM].tvec[j]) if j is not None else None})
    frames_used = [r["frame"] for r in c3]
    clip = {"t0": float(clip_t[0]), "t1": float(clip_t[-1]), "n": int(len(clip_t)), "fps": GRID_HZ,
            "k": [int(clip_k[0]), int(clip_k[-1])], "t": _r(clip_t, 5), "t_other": _r(clip_t - sh, 5),
            "vr_seat": "novel.vr_seat (static; pose the bodies at clip.t)",
            "cam03": {"W": tr3.width, "H": tr3.height, "frames": c3,
                      "duplicate_frames": int(len(frames_used) - len(set(frames_used))),
                      "max_abs_dt_ms": round(max(abs(r["dt_ms"]) for r in c3), 1),
                      "pose_time": "frames[*].t (the photo's own reference time), so the render matches the real panel"},
            "render": "1080 x 1080 (VR seat); phone 03 at its native 480 x 848 (never upscaled past 1080p)"}

    # ---- close-ups (high-res phones)
    closeups = []
    for x in inst:
        cands = {}
        for c in hires:
            v = next(r for r in real if r["instant"] == x["id"] and r["cam"] == c)
            tr = tracks[c]
            i, t = v["frame"], v["t"]
            K, R, tv = np.asarray(v["K"]), np.asarray(v["R"]), np.asarray(v["tvec"])
            masks = {r: read_mask(work, c, i, r, (tr.height, tr.width)) for r in ROLES}
            boxes, score, hair_score = {}, 0.0, 0.0
            for kind, role, half, dy in [("face", "lead", FACE_HALF_M, 0.0), ("face", "follow", FACE_HALF_M, 0.0),
                                         ("hair", "follow", HAIR_HALF_M, HAIR_DY_M)]:
                sk = fits[role].skull(t)
                ctr = sk - np.array([0.0, dy, 0.0])
                uv, z = project(ctr, K, R, tv, v["k1"])
                h = float(K[0, 0]) * half / max(float(z[0]), 1e-6)
                bu = [int(math.floor(uv[0, 0] - h)), int(math.floor(uv[0, 1] - h)), int(math.ceil(uv[0, 0] + h)),
                      int(math.ceil(uv[0, 1] + h))]
                bc = [max(0, bu[0]), max(0, bu[1]), min(tr.width, bu[2]), min(tr.height, bu[3])]
                fwd = fits[role].head_rot(t)[:, 2]
                to_cam = centre(R, tv) - sk
                facing = float(fwd @ to_cam / np.linalg.norm(to_cam))
                m_own = masks[role]
                partner = masks["follow" if role == "lead" else "lead"]
                u0, v0 = int(round(uv[0, 0])), int(round(uv[0, 1]))
                hit = bool(m_own is not None and 0 <= u0 < tr.width and 0 <= v0 < tr.height and m_own[v0, u0]) \
                    if kind == "face" else None
                ov = 0.0
                if partner is not None and bc[2] > bc[0] and bc[3] > bc[1]:
                    ov = float(partner[bc[1]:bc[3], bc[0]:bc[2]].mean())
                valid = z[0] > 0.1 and bc[2] - bc[0] >= 6 and bc[3] - bc[1] >= 6
                own = float(m_own[bc[1]:bc[3], bc[0]:bc[2]].mean()) if m_own is not None and valid else 0.0
                boxes[f"{kind}_{role}"] = {
                    "role": role, "kind": kind, "centre_world": _r(ctr, 4), "head_joint": _r(fits[role].joints(t)[15], 4),
                    "half_m": half, "dy_m": dy, "box": bc if valid else None, "box_unclipped": bu,
                    "clipped": bu != bc, "side_px": round(2 * h, 1), "z_m": round(float(z[0]), 3),
                    "facing_cos": round(facing, 3), "own_mask_at_centre": hit, "own_mask_frac": round(own, 3),
                    "partner_mask_frac": round(ov, 3)}
                if kind == "face" and valid:
                    score += 2 * h * max(facing, 0.05) * (1.0 if hit else 0.3) * (1 - ov)
                if kind == "hair" and valid:
                    hair_score = 2 * h * own * (1 - ov)
            cands[c] = {"view_id": v["id"], "frame": i, "t": t, "score": round(score, 1),
                        "hair_score": round(hair_score if boxes["hair_follow"]["box"] else 0.0, 1), "boxes": boxes}
        best = max(cands, key=lambda c: cands[c]["score"])
        best_h = max(cands, key=lambda c: cands[c]["hair_score"])
        closeups.append({"instant": x["id"], "t": x["t"], "best_cam": best, "best_cam_hair": best_h, "cams": cands,
                         "rule": "best_cam: sum over both faces of side_px x max(facing, 0.05) x (own mask at the centre "
                                 "? 1 : 0.3) x (1 - partner_mask_frac); best_cam_hair: side_px x own_mask_frac x "
                                 "(1 - partner_mask_frac) of the hair square (hair reads from behind too)"})

    # ---- checks: the other take's renders land on the same pixels (its fit, its camera / the mapped camera, its clock)
    flk = ctx["fits"]

    def px_err(cam38, t38, cam_lk, t_lk, k1=0.0, k1_lk=0.0):
        e = []
        for r in ROLES:
            a, za = project(fits[r].joints(t38)[:22], cam38["K"], cam38["R"], cam38["tvec"], k1)
            b, zb = project(flk[r].joints(t_lk)[:22], cam_lk["K"], cam_lk["R"], cam_lk["tvec"], k1_lk)
            e += list(np.linalg.norm(a - b, axis=1)[(za > 0.1) & (zb > 0.1)])
        return round(float(np.median(e)), 1), round(float(np.percentile(e, 90)), 1)

    chk_real = {}
    for v in real:
        if v["lk"]:
            m, p90 = px_err(v, v["t"], v["lk"], v["lk"]["t"], v["k1"], v["lk"]["k1"])
            chk_real[v["id"]] = {"median_px": m, "p90_px": p90, "img_h": v["H"]}
    chk_novel = {}
    for v in tt_views + [vr]:
        t38 = t_tt if v["id"] != "vr_seat" else float(clip_t[len(clip_t) // 2])
        m, p90 = px_err(v, t38, {"K": v["K"], "R": v["lk"]["R"], "tvec": v["lk"]["tvec"]}, t38 - sh)
        chk_novel[v["id"]] = {"median_px": m, "p90_px": p90, "img_h": v["H"], "t": round(t38, 4)}
    bundle_cam = {}
    bp = PACKAGE_ROOT / "work" / other / "dancer_splats" / "bundle" / "views.json"
    if bp.exists():
        bv = json.loads(bp.read_text())
        for v in real:
            if v["lk"] and "bundle view" in v["seen_by"]["ags"]:
                b = bv[int(v["seen_by"]["ags"].split("bundle view ")[1].rstrip(")"))]
                bundle_cam[v["id"]] = {"dR": round(float(np.abs(np.asarray(b["R"]) - np.asarray(v["lk"]["R"])).max()), 6),
                                       "dt_m": round(float(np.abs(np.asarray(b["t_w"]) - np.asarray(v["lk"]["tvec"])).max()), 5),
                                       "dt_s": round(float(b["t"] - v["lk"]["t"]), 5), "bundle_frame": int(b["frame"])}
    checks = {
        "lk_real_views_px": chk_real,
        "lk_novel_views_px": chk_novel,
        "ags_bundle_vs_lk_track": bundle_cam,
        "how": "the reference joints of THIS take through its camera vs the other take's fit joints through its camera "
               "(real: its own track of the same frame at its own time; novel: the body-Sim3-mapped camera at t - "
               "body_shift); ags_bundle_vs_lk_track: AGS's training camera of that frame vs the other take's current "
               "cameras/ track (0 = AGS lives in that world)",
    }

    # ---- references (Unity screenshots of the current avatars)
    rev = work / "review"
    unity = [p for p in [rev / "unity_closeup_follow_t23.0s.png", rev / "unity_closeup_lead_t23.0s.png"] if p.exists()]

    spec = {
        "schema": SCHEMA, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "take": take,
        "code": "dancecap/dancecap/mono4d/appearance_compare.py spec",
        "goal": "owner 2026-10-19: realism; decide between a textured avatar and moving 3D Gaussian splats; pose = MAMMA "
                "or better; picture side-by-sides",
        "conventions": {
            "time": "reference seconds of this take = frames.json t + the phone's sync delta (candidates.frame_to_ref; "
                    "smplx/ and mamma_dyn.load_tracks' clock). Real views: pose the bodies at real_views[*].t (the "
                    "photo's own time), not at the instant",
            "world": "this take's world: right-handed, Y up, metres, floor y = 0",
            "camera": "OpenCV x_cam = R @ X + tvec (+x right, +y down, +z forward); K pinhole, pixel centres at "
                      "integers; C = camera centre; k1 = the real phone's SIMPLE_RADIAL term (render pinhole; undistort "
                      "the real frame with dancer_splats.data.undistort or ignore: <= ~7 px at phone 02's corners)",
            "boxes": "[x0, y0, x1, y1] native pixels of the canonical (distorted) frame, x1 / y1 exclusive",
            "azimuth": "about +Y around the couple's mean pelvis: atan2(dx, dz) in degrees (0 = +Z, 90 = +X)",
            "max_resolution": "renders and videos at most 1080p (never 4K); phone-native sizes are never upscaled",
        },
        "reference_fit": {"dir": _rel(work / "smplx"), "grid": {"t0": float(grid[0]), "hz": GRID_HZ, "n": int(len(grid))},
                          "joints": "SMPL-X 55: 0 pelvis, 15 head, 22 jaw, 23/24 eyes", "head_fk": {r: fits[r].fk_check()
                                                                                                   for r in ROLES}},
        "renderers": {
            "avatar": {"what": "unlit SMPL-X + footage-baked albedo (avatar_texture.render)", "caption": AVATAR_CAPTION,
                       "inputs": [_rel(work / "avatar" / n) for n in ("lead_albedo.png", "follow_albedo.png",
                                                                         "uv_mesh.npz", "lead_texels.npz", "follow_texels.npz")],
                       "saw": f"all phones {idx['avatar_cams']} (bake views every {idx.get('avatar_step_s')} s over the "
                              f"whole take): an IN-SAMPLE reference on every real view, phone 03 included"},
            "lhmpp": {"what": "LHM++ 700M Gaussian avatars from phone 06 crops, posed with the reference fit",
                      "outputs": _rel(work / "mono4d" / "lhmpp"), "input_cam": idx["lhmpp_cam"],
                      "saw": f"phone {idx['lhmpp_cam']} only (held out: {idx.get('lhmpp_holdout')}, and 02 / 04 / 07 / 09 "
                             f"were never used either)"},
            "ags": {"what": "avatar-guided dancer splats, 8 phones", "take": other,
                    "model": _rel(PACKAGE_ROOT / "work" / other / "dancer_splats" / "model.pt"),
                    "saw": "every phone but 03 (and every 10th grid instant of the others: split test_time), "
                           f"{other}'s clock and world: use cross_take"},
        },
        "cross_take": xt,
        "instants": inst,
        "turntable_instant": x_tt["id"],
        "phones": phones,
        "high_res": {"rule": "native long side >= 1000 px", "phones": hires_all, "picked": hires,
                     "both_dancers_well_count": well,
                     "note": "frames.json + the JPEGs: 02 and 06 are 1080 x 1920; 09 is 480 x 848 like 03 / 07 (the "
                             "task's '02/09 are high-res' is not what the frame store says); 06 is LHM++'s input phone"},
        "phone_survey": survey,
        "real_views": real,
        "novel": {"turntable": {"instant": x_tt["id"], "t": t_tt, "t_other": round(t_tt - sh, 5),
                                "centre_pelvis": _r(pel, 4), "look_at": _r(target, 4), **TURNTABLE,
                                "focal_px": round(f_tt, 2), "azimuth0_deg": round(a0, 2),
                                "azimuth0": f"phone {HELDOUT_CAM}'s direction at {x_tt['id']}",
                                "views": tt_views},
                  "vr_seat": vr},
        "clip": clip,
        "closeups": closeups,
        "checks": checks,
        "unity_reference": [{"file": _rel(p), "note": "real Unity screenshot of the current avatars (04_LarissaKaduDemo, "
                                                      "captured 2026-10-07 at scene 23.0 s = reference ~53.15 s, before "
                                                      "the 2026-10-08 face-shape edit; HUD overlays)"} for p in unity],
    }
    return spec, fits, tracks


# ------------------------------------------------------------------------------------------------ spec check images
COL = {"lead": (60, 220, 60), "follow": (60, 60, 240)}   # BGR


def _draw_skel(img, uv, z, col, th=2):
    for j, p in enumerate(SMPL_PARENTS):
        if p < 0 or z[j] <= 0.1 or z[p] <= 0.1:
            continue
        cv2.line(img, tuple(int(round(a)) for a in uv[j]), tuple(int(round(a)) for a in uv[p]), col, th, cv2.LINE_AA)
    for j in range(len(SMPL_PARENTS)):
        if z[j] > 0.1:
            cv2.circle(img, tuple(int(round(a)) for a in uv[j]), th + 1, col, -1, cv2.LINE_AA)


def _label(img, text, org=(8, 22), s=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)


def _fit_h(img, h):
    return cv2.resize(img, (int(round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


def spec_check(spec: dict, fits: dict, out: Path, log=print) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.jpg"):
        p.unlink()
    work = PACKAGE_ROOT / "work" / spec["take"]
    written = []
    by_inst = {}
    cu = {c["instant"]: c for c in spec["closeups"]}
    for v in spec["real_views"]:
        img = cv2.imread(str(PACKAGE_ROOT / v["file"]))
        s = max(1, int(round(max(v["W"], v["H"]) / 960)))
        th = 2 * s
        for r in ROLES:
            m = read_mask(work, v["cam"], v["frame"], r, (v["H"], v["W"]))
            if m is not None:
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, COL[r], max(1, s), cv2.LINE_AA)
            uv, z = project(fits[r].joints(v["t"])[:22], v["K"], v["R"], v["tvec"], v["k1"])
            _draw_skel(img, uv, z, COL[r], th)
        if v["crop"]:
            x0, y0, x1, y1 = v["crop"]
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 255), th)
        c = cu.get(v["instant"], {}).get("cams", {}).get(v["cam"])
        if c:
            for k, b in c["boxes"].items():
                if b["box"]:
                    col = (255, 200, 0) if b["kind"] == "face" else (0, 140, 255)
                    cv2.rectangle(img, tuple(b["box"][:2]), (b["box"][2] - 1, b["box"][3] - 1), col, max(1, s))
        lab = (f"{v['id']} cam {v['cam']} frame {v['frame']} t {v['t']:.3f} (instant dt {v['dt_ms']:+.1f} ms) "
               f"{'BOTH WELL' if v['both_well'] else 'check'}")
        _label(img, lab, (8 * s, 26 * s), 0.6 * s)
        _label(img, "green lead / red follow: reference joints + mask outlines; yellow crop; cyan face, orange hair",
               (8 * s, 52 * s), 0.5 * s)
        p = out / f"{v['id']}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        written.append(p.name)
        x0, y0, x1, y1 = v["crop"] or [0, 0, v["W"], v["H"]]
        by_inst.setdefault(v["instant"], []).append(_fit_h(img[y0:y1, x0:x1], 520))
    for k, tiles in by_inst.items():
        p = out / f"sheet_{k}.jpg"
        cv2.imwrite(str(p), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
        written.append(p.name)
    # novel cameras: skeletons on black
    tt = spec["novel"]["turntable"]
    tiles = []
    for v in tt["views"]:
        img = np.full((v["H"], v["W"], 3), 30, np.uint8)
        for r in ROLES:
            uv, z = project(fits[r].joints(tt["t"])[:22], v["K"], v["R"], v["tvec"])
            _draw_skel(img, uv, z, COL[r], 4)
        _label(img, f"{v['id']} az {v['azimuth_from_cam03_deg']:.0f} from cam03 margin {v['frame_margin']:.2f}",
               (12, 40), 1.1)
        tiles.append(cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA))
    p = out / "novel_turntable.jpg"
    cv2.imwrite(str(p), np.vstack([np.hstack(tiles[:6]), np.hstack(tiles[6:])]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    written.append(p.name)
    vr, clip = spec["novel"]["vr_seat"], spec["clip"]
    tiles = []
    for t in clip["t"][::15]:
        img = np.full((vr["H"], vr["W"], 3), 30, np.uint8)
        for r in ROLES:
            uv, z = project(fits[r].joints(t)[:22], vr["K"], vr["R"], vr["tvec"])
            _draw_skel(img, uv, z, COL[r], 4)
        _label(img, f"VR seat t {t:.2f} az {vr['azimuth_deg']:.0f}", (12, 40), 1.1)
        tiles.append(cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA))
    p = out / "novel_vr_seat.jpg"
    cv2.imwrite(str(p), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    written.append(p.name)
    # phone 03's own track through the clip (real frames + joints)
    tiles = []
    c3 = clip["cam03"]
    for r0 in c3["frames"][::15]:
        img = cv2.imread(str(PACKAGE_ROOT / r0["file"]))
        for r in ROLES:
            uv, z = project(fits[r].joints(r0["t"])[:22], r0["K"], r0["R"], r0["tvec"], r0["k1"])
            _draw_skel(img, uv, z, COL[r], 2)
        _label(img, f"cam03 f{r0['frame']} t {r0['t']:.2f}", (6, 20), 0.5)
        tiles.append(img)
    p = out / "clip_cam03.jpg"
    cv2.imwrite(str(p), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    written.append(p.name)
    # bird's-eye: phones, VR seat, turntable ring, couple
    S, sc = 900, 70.0
    pc = np.asarray(vr["centre_pelvis_clip"])
    img = np.full((S, S, 3), 25, np.uint8)
    to = lambda P: (int(S / 2 + sc * (P[0] - pc[0])), int(S / 2 + sc * (P[2] - pc[2])))  # noqa: E731  (from above)
    for c, ph in spec["phones"].items():
        cv2.circle(img, to(ph["C_mean"]), 6, (200, 200, 200), -1)
        _label(img, c, (to(ph["C_mean"])[0] + 8, to(ph["C_mean"])[1]), 0.55)
    for v in tt["views"]:
        cv2.circle(img, to(v["C"]), 4, (0, 200, 255), -1)
    cv2.circle(img, to(vr["C"]), 8, (255, 0, 255), -1)
    _label(img, "VR seat", (to(vr["C"])[0] + 10, to(vr["C"])[1]), 0.55)
    for r in ROLES:
        for t in clip["t"][::5]:
            cv2.circle(img, to(fits[r].joints(t)[0]), 2, COL[r], -1)
    _label(img, "seen from above (x right, z down): phones (grey), turntable ring (yellow), VR seat (magenta), pelvis tracks", (8, 22),
           0.55)
    p = out / "top_view.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    written.append(p.name)
    log(f"[compare] spec_check: {len(written)} images in {out}")
    return written


# ------------------------------------------------------------------------------------------------ README
def readme(spec: dict) -> str:
    xt = spec["cross_take"]
    cw, bw = xt["world"]["sim3_take_to_other_cameras"], xt["world"]["sim3_take_to_other_bodies"]
    inst = "\n".join(
        f"| {x['id']} | {x['t']:.4f} | {x['k']} | {x['ags_review_t_other']:.3f} | {x['t_other']:.4f} | "
        f"{'yes' if x['head_turn'] else ''}{(' (peak ' + format(x['head_turn_check']['peak_t'], '.3f') + ', ' + str(x['head_turn_check']['frames_from_peak']) + ' fr)') if x.get('head_turn_check', {}).get('peak_t') else ''} |"
        for x in spec["instants"])
    vr = spec["novel"]["vr_seat"]
    tt = spec["novel"]["turntable"]
    hr = spec["high_res"]
    nwell = {v["id"]: v["both_well"] for v in spec["real_views"]}
    bad = [k for k, ok in nwell.items() if not ok]
    return f"""# Appearance comparison spec - {spec['take']}

Written by `python -m dancecap.mono4d.appearance_compare spec` ({spec['created']}). Every renderer (current avatar,
LHM++, AGS splats) renders exactly the cameras and times in `views.json`; a later `compose` step builds the
side-by-sides. Sheets that show the avatar say: "{AVATAR_CAPTION}".

**Conventions:** reference seconds of {spec['take']} (frames.json t + the phone's sync delta); world Y-up metres, floor
y = 0; OpenCV cameras (`x_cam = R X + tvec`), pinhole K; boxes `[x0, y0, x1, y1]` in native pixels of the canonical
frame. Real views: pose the bodies at `real_views[*].t` (the photo's own time). At most 1080p.

## Instants (grid of smplx/, 30 Hz)

| id | t | grid k | AGS review t ({xt['other']} clock) | t on {xt['other']} clock | head turn |
|---|---|---|---|---|---|
{inst}

The turntable uses {tt['instant']} (t {tt['t']:.4f}).

## Cameras

- **Real:** phone 03 (held out by AGS and LHM++; the avatar bake saw it) and the high-res phones {hr['picked']}.
  Only 02 and 06 are high-res (1080 x 1920); 09 is 480 x 848 like 03 and 07, so "02/09 are high-res" was wrong. 06 is
  LHM++'s input phone and both 02 and 06 are AGS training phones. `seen_by` on each view says what saw what.
  Views where a dancer is not shown well (mask cut, small or joints outside the mask): {bad or 'none'}.
- **Turntable:** 12 views, 1080 x 1080, {tt['vfov_deg']:.0f} deg vertical FOV, radius {tt['radius_m']} m, height
  {tt['height_m']} m, looking at 1.0 m above the floor over the couple's mean pelvis; azimuth 0 = phone 03's direction.
- **VR seat:** 1080 x 1080, 55 deg vertical FOV, 2.5 m from the clip's mean pelvis, 1.6 m high, looking at
  {vr['look_at'][1]:.2f} m above the floor (the height that centres the couple over the clip; 1.0 m cut the near
  feet), azimuth {vr['azimuth_deg']:.1f} deg, in the middle of the largest gap between phones ({vr['gap']['between'][0]}-{vr['gap']['between'][1]},
  {vr['gap']['deg']:.1f} deg wide, so {vr['gap']['nearest_phone_deg']:.1f} deg from the nearest phone). The couple stays
  in frame over the whole clip (min margin {vr['frame_margin_min']:.2f} of the frame height); the closest dancer point
  is {vr['closest_dancer_point_m']:.2f} m away.
- **Clip:** {spec['clip']['n']} grid frames, {spec['clip']['t0']:.4f}-{spec['clip']['t1']:.4f} s, from the VR seat and
  from phone 03's own track (real frames; {spec['clip']['cam03']['duplicate_frames']} duplicate frames, max
  |dt| {spec['clip']['cam03']['max_abs_dt_ms']} ms).
- **Close-ups:** high-res phones only. Face: a 32 cm square around the skull centre (head joint + R_head [0, 0.07,
  0.01] m, AGS's convention). Hair: a 60 cm square from the follow's crown down (skull centre 0.18 m lower). `best_cam`
  per instant = the phone that sees both faces bigger and more frontally; `best_cam_hair` = the one that sees most of
  her hair square unoccluded: {', '.join(f"{c['instant']} {c['best_cam']}/{c['best_cam_hair']}" for c in spec['closeups'])}
  (face/hair).
- The turntable's FOV (50 deg) was not given: it keeps the couple (joints, crowns + 0.2 m, feet on the floor) inside
  every view with {min(v['frame_margin'] for v in tt['views']):.2f}-{max(v['frame_margin'] for v in tt['views']):.2f} of
  the frame height to spare.

## Same videos, other take ({xt['other']}, where AGS was trained)

- **Clock:** not the same. Identical frames are {min(v['offset_ms_median'] for v in xt['clock']['per_phone'].values()):+.1f}
  to {max(v['offset_ms_median'] for v in xt['clock']['per_phone'].values()):+.1f} ms apart per phone (two different
  sync estimates), median {xt['clock']['per_phone_median_ms']:+.1f} ms. The two fits' motion lines up at a
  **{xt['clock']['body_shift_ms']:+.0f} ms** shift: `t_{xt['other']} = t - {xt['clock']['body_shift_ms']:.0f} ms`. For real frames use the other
  take's own time of the same frame (`real_views[*].lk.t`).
- **World:** not the same. Sim3 world_{spec['take']} -> world_{xt['other']} from the camera centres: s {cw['s']:.5f},
  {cw['rot_deg']:.2f} deg, residual rms {100 * cw['rms_m']:.1f} cm (median {100 * cw['median_m']:.1f}, per phone
  {', '.join(f'{c} {100 * v:.1f}' for c, v in cw['per_cam_median_m'].items())} cm). From the two fits' joints: s
  {bw['s']:.5f}, {bw['rot_deg']:.2f} deg, mean residual {1000 * bw['mean_m']:.1f} mm. The novel cameras' `lk` blocks use the
  body Sim3; real views carry the other take's own camera of the same frame.

**Cross-take check** (`checks`): the other take's fit drawn through its own camera of the same frame lands
{min(v['median_px'] for v in spec['checks']['lk_real_views_px'].values()):.1f}-{max(v['median_px'] for v in spec['checks']['lk_real_views_px'].values()):.1f} px (median per view) from this take's fit through
this take's camera on the real views, and {min(v['median_px'] for v in spec['checks']['lk_novel_views_px'].values()):.1f}-{max(v['median_px'] for v in spec['checks']['lk_novel_views_px'].values()):.1f} px on the
novel views through the mapped cameras (1080 px frames). AGS's training cameras equal the other take's current tracks
exactly (`ags_bundle_vs_lk_track`).

## Check images (`spec_check/`)

The reference joints (green lead, red follow) and mask outlines on every real view, the crop (yellow), face (cyan) and
hair (orange) boxes, `sheet_<instant>.jpg` (the three crops side by side), the novel cameras as skeletons
(`novel_turntable.jpg`, `novel_vr_seat.jpg`), phone 03's clip track (`clip_cam03.jpg`) and a top view.

Unity reference (real screenshot of the current avatars, older face shape):
{chr(10).join('- `' + u['file'] + '`' for u in spec['unity_reference']) or '- none found'}
"""


def run_spec(take: str, other: str, log=print) -> Path:
    spec, fits, _ = build_spec(take, other, log)
    out = compare_dir(take)
    out.mkdir(parents=True, exist_ok=True)
    spec["spec_check"] = spec_check(spec, fits, out / "spec_check", log)
    p = out / "views.json"
    p.write_text(json.dumps(spec, indent=1))
    (out / "README.md").write_text(readme(spec), encoding="utf-8")
    log(f"[compare] wrote {p} ({p.stat().st_size / 1e3:.0f} kB) + README.md")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.appearance_compare")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec", help="write work/<take>/mono4d/compare/views.json + README.md + spec_check/")
    s.add_argument("--take", default="larissa_kadu_38")
    s.add_argument("--other", default="larissa_kadu", help="the take AGS was trained on (same source videos)")
    c = sub.add_parser("compose", help="side-by-side sheets, metrics, Jev judge, videos from compare/renders/ "
                                       "(appearance_compose.py, appearance_jev.py)")
    c.add_argument("--take", default="larissa_kadu_38")
    c.add_argument("--steps", default="sheets,metrics,videos,check",
                   help="comma list of sheets, metrics, jev, videos, check (jev needs --allow-upload)")
    c.add_argument("--allow-upload", action="store_true",
                   help="jev: send the real crops and renders (likeness data) to TypeSafe (owner consent 2026-10-19)")
    c.add_argument("--shuffles", type=int, default=3, help="jev: shuffled Choice questions per real view")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--best-splat", default=None, help="phone-03 video splat panel (default: Jev / metrics pick)")
    c.add_argument("--best-hybrid", default=None, help="phone-03 video hybrid panel (default: Jev / metrics pick)")
    a = ap.parse_args(argv)
    if a.cmd == "spec":
        run_spec(a.take, a.other)
    elif a.cmd == "compose":
        from .appearance_compose import run_compose
        run_compose(a.take, tuple(s.strip() for s in a.steps.split(",") if s.strip()), allow_upload=a.allow_upload,
                    best_splat=a.best_splat, best_hybrid=a.best_hybrid, shuffles=a.shuffles, workers=a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
