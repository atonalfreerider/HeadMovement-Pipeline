"""Dancer-anchored localisation of a MOVING camera beside a fixed one (two-view takes; lucas_luiza 2026-10-24).

Why: a room with repeated structure (identical double doors and EXIT signs on several walls, rows of audience) lets a
moving camera's frames - and its own map images - register on the WRONG wall: inside the room, at 1-2 px against the
static points, so neither the reprojection nor cameras.plausible sees it. lucas_luiza's Steadicam (01): 24 of the 65
map images whose dancers are measurable sat on another wall (the dancers' joints 40-680 px off the tripod's epipolar
lines, the right ones 3-15 px), the raw per-frame track jumped 2-15 map units between consecutive frames 592 times, and
only ~55 % of its frames agreed with the tripod: whole 5-20 s stretches were fitted from a camera on the wrong wall.

The fixed camera (a tripod: one centre, ~1000 static inliers per frame) is the judge. The dancers are seen by both
cameras at the same instant, and a moving-camera pose is RIGHT when
  epi       the dancers' joints lie on the epipolar lines of the fixed camera's joints (median same-role symmetric
            epipolar distance, joints with conf >= 0.5 in both views, >= 8), and
  transfer  each dancer's ankle, put on the floor by the fixed camera's ray (cameras/world_from_sfm.json: gravity, floor
            y = 0; the ankle joint's height above it), projects onto the same ankle in the moving camera (the smaller error
            of the two feet - the planted one -, median px over the dancers). The epipolar test is one-dimensional: two cameras at one height have near-horizontal epipolar
            lines, and a pose on the mirror-image side of the room can pass it (lucas_luiza 53.7-55.3 s and 68.7-71.0 s:
            epi 4-14 px, transfer 700-1050 px; right poses: transfer median 84 px, p95 256 px - a lifted foot). No world
            file: the epipolar test alone.

    python -m dancecap.cameras.anchored score <take> --moving 01 --fixed 02      # CPU: judge the raw track and the map images
    python -m dancecap.cameras.anchored run   <take> --moving 01 --fixed 02      # + re-localise (features: GPU) -> loc cache

run, on the raw localisation cameras/cache/loc_<moving>.npz (front_pod loc-part / loc-join; kept as loc_<moving>_raw.npz):
  1 judge   every raw pose (residual_px at the take's sync offset: --offset-ms, else sync/curves_2v.json's whole-window
            estimate, else 0) and every registered map image of the moving camera (window frames and keyframes)
  2 refs    the map images the judge rejects (> LOOSE x TAU) leave the localiser's references
  3 anchors raw poses with residual <= TAU
  4 grow    forward / backward from accepted frames through raw poses that are continuous with them (motion gate: the
            centre moves <= V_MAX_ROOM x the room radius per second, the rotation <= W_MAX_DEG per second, + slack) and
            whose residual is not > LOOSE x TAU
  5 prior   frames still open next to an accepted one (<= NEAR frames) are localised again with the neighbour's pose as a
            PRIOR: the references are the map images (either camera, refused ones excluded) whose pose is nearest the
            prior's (centre distance + view angle: they see the same part of the room; the localiser's nearest-in-time
            choice is what put whole stretches on the wrong wall) plus the nearest in time; only the 2D-3D matches that
            reproject within PRIOR_PX of their keypoint under the prior are kept (the wrong wall's matches land far away),
            P3P with the map focal (fix_focal), then the motion gate and the judge (Judge.refused); passes alternate
            direction until nothing changes
  6 write   cameras/cache/loc_<moving>.npz: ok and source per frame ('anchor', 'cont', 'prior'; not ok: 'dancers' = a
            raw pose the judge refused, 'unanchored' = a raw pose no accepted frame reached), cameras/qa/anchored_<moving>
            .json (counts, coverage per 8 s, excluded map images, residuals). stage_smooth / the typed refit / the world
            then read it as any loc cache.
The fixed camera is never changed. A frame whose dancers cannot be measured (a clinch, one dancer hidden) is judged by
continuity alone: it can only enter through steps 4-5, from a judged neighbour. Residual risk: a wrong pose on the
symmetric side of the room that still FACES the dancers can pass both tests; the motion gate (a jump between frames)
and the anchors around it are the defence there.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from . import twoview as tv

TAU_PX = 15.0  # an anchor: the dancers' median epipolar distance to the fixed camera's joints
LOOSE = 3.0  # a grown / prior pose may reach LOOSE x TAU where the dancers are measurable; map images above it leave the refs
XFER_ANCHOR = 0.25  # an anchor's ankle transfer error <= this x the moving image's height (lucas_luiza: 320 px)
XFER_LOOSE = 0.3  # a grown / prior pose or a map image is refused above this (right poses: p95 205 px = 0.16 x 1280)
ANKLE_H = 0.084  # the ankle joint above the floor (m) without a measured value (world diagnostics low_ankle_height_m)
MIN_JOINTS = 8
V_MAX_ROOM = 0.3  # motion gate: room radius per second (lucas_luiza: 12.4 map units -> 3.7 units/s)
W_MAX_DEG = 180.0  # rotation per second
SLACK_ROOM, SLACK_DEG = 0.01, 2.0
PRIOR_PX = 80.0
NEAR = 2
MAX_PASSES = 6
N_SPATIAL = 6  # map images nearest the prior pose ...
N_TIME = 2  # ... + the moving camera's nearest in time
ANGLE_UNIT_DEG = 20.0  # spatial distance = centre distance / room radius x 10 + view angle / this (deg)


# ------------------------------------------------------------------------------------------------ the judge


class Judge:
    """The fixed camera's track and both cameras' role joints; residual() of candidate moving-camera poses."""

    def __init__(self, work: Path, moving: str, fixed: str, offset_s: float):
        self.work, self.moving, self.fixed, self.offset_s = work, moving, fixed, float(offset_s)
        self.fx = tv.load_track(work, fixed, world=False)  # SfM frame, as the loc cache
        self.kt_f, self.kp_f = tv.role_kp(work, fixed)
        self.kt_m, self.kp_m = tv.role_kp(work, moving)
        meta = json.loads((work / "frames" / moving / "frames.json").read_text())
        self.w, self.h = int(meta["width"]), int(meta["height"])
        self.pp = np.array([(self.w - 1) / 2.0, (self.h - 1) / 2.0])  # OpenCV pixel centres, as the exported K
        wp = work / "cameras" / "world_from_sfm.json"
        self.world = None
        if wp.exists():
            W = json.loads(wp.read_text())
            ha = ((W.get("diagnostics") or {}).get("low_ankle_height_m") or {}).get("median")
            self.world = {"s": float(W["s"]), "R": np.asarray(W["R"], float), "t": np.asarray(W["t"], float),
                          "ankle_h": float(ha) if ha else ANKLE_H}

    def residual(self, t: np.ndarray, kp_m: np.ndarray, R: np.ndarray, tvec: np.ndarray, f: np.ndarray,
                 k1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(epi px (M,), joints (M,)) of moving-camera poses R (M,3,3), tvec (M,3), f (M,), k1 (M,) at its times t (M,)
        with its joints kp_m (M, 2, 17, 3): NaN where fewer than MIN_JOINTS joints are measurable."""
        return self.judge(t, kp_m, R, tvec, f, k1)[::2]

    def judge(self, t: np.ndarray, kp_m: np.ndarray, R: np.ndarray, tvec: np.ndarray, f: np.ndarray,
              k1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(epi px, transfer px, joints) per pose (module docstring); transfer NaN without a world file or ankles."""
        t = np.asarray(t, float)
        M = len(t)
        if M == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0, int)
        R, tvec, f, k1 = (np.asarray(x, float) for x in (R, tvec, f, k1))
        kp_m = np.asarray(kp_m, float)
        tau = t + self.offset_s  # the fixed camera is the gauge: its label = the corrected time
        co = tv.interp_track(self.fx, tau)
        kf, _ = tv.interp_kp(self.kt_f, self.kp_f, tau)
        xo = tv._homog(tv._normalise(kf[..., :2], co["f"], co["pp"], co["k1"]))
        xm = tv._homog(tv._normalise(kp_m[..., :2], f, np.broadcast_to(self.pp, (M, 2)), k1))
        R_om = np.einsum("mij,mkj->mik", co["R"], R)
        t_om = co["tvec"] - np.einsum("mij,mj->mi", R_om, tvec)
        with np.errstate(invalid="ignore"):
            d = tv.sym_epi(xo, xm, R_om, t_om, co["f"], f)  # (M, 2 roles, 17)
            v = ((kf[..., 2] >= tv.MIN_CONF) & (kp_m[..., 2] >= tv.MIN_CONF) & np.isfinite(d)
                 & co["ok"][:, None, None])
        n = v.sum((1, 2))
        with np.errstate(invalid="ignore"):
            med = np.array([np.median(d[i][v[i]]) if n[i] >= MIN_JOINTS else np.nan for i in range(M)])
        return med, self.transfer(co, kf, kp_m, R, tvec, f, k1), n

    def transfer(self, co: dict, kf: np.ndarray, kp_m: np.ndarray, R, tvec, f, k1) -> np.ndarray:
        """Median over the dancers of the smaller (over the two feet: the planted one) |moving camera's ankle - the same
        ankle of the fixed camera put on the floor and projected into the moving camera| (px); 1e4 when that point lies
        behind the moving camera; NaN without a world file or an ankle seen in both views (conf >= MIN_CONF)."""
        M = len(f)
        if self.world is None:
            return np.full(M, np.nan)
        W = self.world
        af, am = kf[:, :, 15:17], np.asarray(kp_m, float)[:, :, 15:17]  # (M, 2 roles, 2 ankles, 3)
        seen = (af[..., 2] >= tv.MIN_CONF) & (am[..., 2] >= tv.MIN_CONF)
        xn = tv._normalise(af[..., :2], co["f"], co["pp"], co["k1"])
        dc = np.concatenate([xn, np.ones(xn.shape[:-1] + (1,))], -1)
        d_sfm = np.einsum("mji,mraj->mrai", co["R"], dc)  # rays in the SfM frame
        o_w = W["s"] * co["C"] @ W["R"].T + W["t"]
        d_w = np.einsum("ij,mraj->mrai", W["R"], d_sfm)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (W["ankle_h"] - o_w[:, None, None, 1]) / d_w[..., 1]
            P_w = o_w[:, None, None, :] + lam[..., None] * d_w
            P = np.einsum("ji,mraj->mrai", W["R"], P_w - W["t"]) / W["s"]  # back to the SfM frame
            X = np.einsum("mij,mraj->mrai", R, P) + tvec[:, None, None, :]
            u = X[..., :2] / X[..., 2:3]
            px = f[:, None, None, None] * u * (1 + k1[:, None, None] * (u ** 2).sum(-1))[..., None] + self.pp
            e = np.linalg.norm(px - am[..., :2], axis=-1)  # (M, 2, 2)
        e = np.where(X[..., 2] > 0, e, 1e4)
        e = np.where(seen & (lam > 0) & np.isfinite(e) & co["ok"][:, None, None], e, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(np.nanmin(e, 2), 1)

    def anchor(self, epi: np.ndarray, xfer: np.ndarray) -> np.ndarray:
        """An anchor: the dancers measured and both tests well inside - with a world file the transfer MUST be measured
        (the epipolar test alone passes mirror poses: review of b829830, lucas_luiza 53.98-55.19 s)."""
        ok_x = (np.isfinite(xfer) & (xfer <= XFER_ANCHOR * self.h)) if self.world is not None else np.ones(len(epi), bool)
        return np.isfinite(epi) & (epi <= TAU_PX) & ok_x

    def refused(self, epi: np.ndarray, xfer: np.ndarray) -> np.ndarray:
        """Clearly off on either test (a grown / prior pose, a map image); unmeasurable is never refused."""
        return (np.isfinite(epi) & (epi > LOOSE * TAU_PX)) | (np.isfinite(xfer) & (xfer > XFER_LOOSE * self.h))

    def joints_at(self, t: np.ndarray) -> np.ndarray:
        """The moving camera's role joints at times t (its own frames: exact; keyframes: interpolated, NaN in a gap)."""
        return tv.interp_kp(self.kt_m, self.kp_m, np.asarray(t, float))[0]


def offset_ms(work: Path, given: float | None, moving: str = "01", fixed: str = "02") -> tuple[float, str]:
    """The moving camera's offset against the fixed one (ms; the fixed camera's corrected time = the moving camera's
    t + it): --offset-ms, else sync/offsets.json (gauged on either camera; a drift line is taken at its t_ref), else
    sync/curves_2v.json's whole-window minimum (master / other either way round), else 0."""
    if given is not None:
        return float(given), "--offset-ms"

    def from_offsets(d):
        o = d["offsets"]
        dm, df = float(o[moving].get("delta_ms") or 0.0), float(o[fixed].get("delta_ms") or 0.0)
        return dm - df  # each camera's corrected time is t + its delta: relative to the fixed one

    def from_curves(d):
        g = float(d["global"]["delta_ms"])
        if (d.get("master"), d.get("other")) == (fixed, moving):
            return g
        if (d.get("master"), d.get("other")) == (moving, fixed):
            return -g
        raise ValueError("curves_2v.json is about other cameras")

    for p, get in ((work / "sync" / "offsets.json", from_offsets), (work / "sync" / "curves_2v.json", from_curves)):
        if p.exists():
            try:
                v = float(get(json.loads(p.read_text())))
                if np.isfinite(v):
                    return v, str(p.relative_to(work))
            except (KeyError, StopIteration, TypeError, ValueError):
                continue
    return 0.0, "none (0)"


def rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    c = (np.einsum("...ij,...ij->...", Ra, Rb) - 1.0) / 2.0  # trace(Ra Rb^T)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def centres(R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return -np.einsum("...ji,...j->...i", R, tvec)


def motion_ok(Ca, Ra, ta, Cb, Rb, tb, room: float) -> bool:
    dt = abs(float(tb) - float(ta))
    return bool(np.linalg.norm(np.asarray(Cb) - np.asarray(Ca)) <= V_MAX_ROOM * room * dt + SLACK_ROOM * room
                and float(rot_deg(np.asarray(Ra), np.asarray(Rb))) <= W_MAX_DEG * dt + SLACK_DEG)


def grow(t, ok, C, R, refused, acc, room: float) -> np.ndarray:
    """Step 4: accepted frames extended through continuous raw poses, forward and backward until stable. A raw pose
    joins when the last accepted frame is <= NEAR frame intervals before it (in the pass direction), the motion gate
    holds against it and the judge does not refuse it (Judge.refused); a refused raw pose ends the chain."""
    acc = acc.copy()
    n = len(t)
    gap = (NEAR + 0.5) * float(np.median(np.diff(t))) if n > 1 else 1.0
    changed = True
    while changed:
        changed = False
        for order in (range(n), range(n - 1, -1, -1)):
            last = None
            for i in order:
                if acc[i]:
                    last = i
                    continue
                if not ok[i] or last is None:
                    continue
                if abs(t[i] - t[last]) > gap or refused[i]:  # a refused frame ends the chain: never hop over it
                    last = None
                    continue
                if motion_ok(C[last], R[last], t[last], C[i], R[i], t[i], room):
                    acc[i] = True
                    last = i
                    changed = True
    return acc


def prior_gate(xyz: np.ndarray, kp: np.ndarray, R: np.ndarray, tvec: np.ndarray, f: float, k1: float, w: int, h: int,
               px: float = PRIOR_PX) -> np.ndarray:
    """Matches (kp in COLMAP px, xyz) that reproject within px of their keypoint under the prior pose (SIMPLE_RADIAL,
    principal point (w/2, h/2) as solve_pose); points behind the prior camera never pass."""
    X = xyz @ np.asarray(R).T + np.asarray(tvec)
    z = X[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = X[:, :2] / z[:, None]
        r2 = (u ** 2).sum(1)
        p = f * u * (1 + k1 * r2)[:, None] + np.array([w / 2.0, h / 2.0])
        return (z > 0) & (np.linalg.norm(p - kp, axis=1) <= px)


# ------------------------------------------------------------------------------------------------ map images


def map_image_scores(work: Path, J: Judge) -> list[dict]:
    """The moving camera's registered map images (cameras/sfm) judged: [{name, t, residual_px, joints}]."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(work / "cameras" / "sfm"))
    times = {}
    for kind, sub in (("frames", "frames.json"), ("keyframes", "keyframes.json")):
        p = work / kind / J.moving / sub
        if p.exists():
            d = json.loads(p.read_text())
            for e in (d["frames"] if isinstance(d, dict) else d):
                times[f"{kind}/{J.moving}/{e['file']}"] = float(e["t"])
    rows, ts, Rs, tvs, fs, k1s = [], [], [], [], [], []
    for im in rec.images.values():
        if not im.has_pose or im.name not in times:
            continue
        T = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
        p = rec.cameras[im.camera_id].params
        rows.append(im.name), ts.append(times[im.name]), Rs.append(np.asarray(T.rotation.matrix())),
        tvs.append(np.asarray(T.translation)), fs.append(float(p[0])), k1s.append(float(p[3]) if len(p) > 3 else 0.0)
    if not rows:
        return []
    t = np.array(ts)
    epi, xf, n = J.judge(t, J.joints_at(t), np.array(Rs), np.array(tvs), np.array(fs), np.array(k1s))
    bad = J.refused(epi, xf)
    num = lambda x: None if not np.isfinite(x) else round(float(x), 2)  # noqa: E731
    return [{"name": nm, "t": round(float(a), 3), "residual_px": num(r), "transfer_px": num(x), "joints": int(k),
             "refused": bool(b)} for nm, a, r, x, k, b in sorted(zip(rows, t, epi, xf, n, bad), key=lambda z: z[1])]


# ------------------------------------------------------------------------------------------------ run


ANCHORED_SOURCES = ("anchor", "cont", "prior", "dancers", "unanchored")


def is_anchored(d: dict) -> bool:
    return bool(np.isin(np.asarray(d.get("source", [])).astype(str), ANCHORED_SOURCES).any())


def _load_raw(work: Path, cam: str, write: bool = False) -> tuple[dict, Path]:
    """The raw localisation: loc_<cam>.npz when it is one (no anchored source: a fresh loc / loc-join / map_extend -
    it then replaces loc_<cam>_raw.npz when write), else loc_<cam>_raw.npz (an anchored loc_<cam>.npz without it:
    SystemExit - its refused poses are gone)."""
    cache = work / "cameras" / "cache"
    cur_p, raw_p = cache / f"loc_{cam}.npz", cache / f"loc_{cam}_raw.npz"
    cur = dict(np.load(tv._need(cur_p)))
    if not is_anchored(cur):
        if write:
            shutil.copy2(cur_p, raw_p)
        return cur, cur_p
    if not raw_p.exists():
        raise SystemExit(f"[anchored] {cur_p} is an anchored output and {raw_p.name} is missing: localise {cam} again "
                         f"(front_pod loc-part / loc-join) or bring {raw_p.name} back")
    return dict(np.load(raw_p)), raw_p


def score(J: Judge, d: dict) -> dict:
    t, ok = np.asarray(d["t"], float), np.asarray(d["ok"], bool)
    epi, xf, n = J.judge(t, J.joints_at(t), d["R"], d["tvec"], np.asarray(d["f"], float), np.asarray(d["k1"], float))
    return {"residual": np.where(ok, epi, np.nan), "transfer": np.where(ok, xf, np.nan), "joints": n}


def coverage(t: np.ndarray, ok: np.ndarray, step: float = 8.0) -> list:
    out = []
    for a in np.arange(np.floor(t[0]), t[-1], step):
        m = (t >= a) & (t < a + step)
        if m.any():
            out.append([round(float(a), 1), round(float(ok[m].mean()), 3)])
    return out


def cmd_score(work: Path, moving: str, fixed: str, off_ms: float | None, log=print) -> dict:
    off, why = offset_ms(work, off_ms, moving, fixed)
    J = Judge(work, moving, fixed, off / 1000.0)
    d, src = _load_raw(work, moving)
    s = score(J, d)
    r = s["residual"][np.isfinite(s["residual"])]
    xf = s["transfer"][np.isfinite(s["transfer"])]
    maps = map_image_scores(work, J)
    bad = [m for m in maps if m["refused"]]
    doc = {"moving": moving, "fixed": fixed, "offset_ms": off, "offset_from": why, "raw": str(src.relative_to(work)),
           "frames": int(len(d["t"])), "raw_ok": int(np.sum(d["ok"])), "judged": int(len(r)),
           "residual_pct": {str(q): round(float(np.percentile(r, q)), 2) for q in (25, 50, 75, 90)} if len(r) else None,
           "share_le": {str(x): round(float(np.mean(r <= x)), 3) for x in (TAU_PX, LOOSE * TAU_PX)} if len(r) else None,
           "transfer_pct": {str(q): round(float(np.percentile(xf, q)), 1) for q in (50, 75, 90)} if len(xf) else None,
           "anchors": int(J.anchor(s["residual"], s["transfer"]).sum()), "world": J.world is not None,
           "map_images": {"judged": sum(m["residual_px"] is not None for m in maps), "total": len(maps), "bad": len(bad),
                          "rows": maps}}
    log(f"[anchored] {moving} vs {fixed} at {off:+.1f} ms ({why}): raw ok {doc['raw_ok']}/{doc['frames']}, judged "
        f"{doc['judged']}, residual {doc['residual_pct']}, transfer {doc['transfer_pct']}, anchors {doc['anchors']}; "
        f"map images {len(bad)} of {doc['map_images']['judged']} judged on another wall")
    return doc


def cmd_run(take_name: str, moving: str, fixed: str, off_ms: float | None, log=print) -> dict:
    from ..take import Take
    from ..front_pod import loc_ref_cams
    from .features import Features, FeatureSettings, list_images
    from .localize import FramePose, Localizer, LocSettings, Ref, _dedupe, _lift, solve_pose
    from .mapping import map_settings
    from .pipeline import _save_raw, load_model, loc_settings, map_images
    from .plausible import model_bands

    t0 = time.time()
    take = Take.load(take_name)
    work = take.work.resolve()
    doc = cmd_score(work, moving, fixed, off_ms, log)
    off = doc["offset_ms"] / 1000.0
    J = Judge(work, moving, fixed, off)
    d, src = _load_raw(work, moving, write=True)
    cache = work / "cameras" / "cache"
    t, ok = np.asarray(d["t"], float), np.asarray(d["ok"], bool)
    R, tvec = np.array(d["R"], float), np.array(d["tvec"], float)
    f, k1 = np.array(d["f"], float), np.array(d["k1"], float)
    C = centres(R, tvec)
    sc = score(J, d)
    res, xfer = sc["residual"], sc["transfer"]
    model = load_model(work)
    box = model_bands(model).box
    room = float(box.radius) if box is not None else 10.0
    anchors = ok & J.anchor(res, xfer)
    judged_out = ok & J.refused(res, xfer)  # raw poses the judge refuses (kept for the labels: res / xfer change below)
    ok_raw = ok.copy()
    acc = grow(t, ok, C, R, judged_out, anchors, room)
    how = np.where(anchors, "anchor", np.where(acc, "cont", ""))
    log(f"[anchored] anchors {int(anchors.sum())}, grown {int(acc.sum() - anchors.sum())} of {len(t)} frames")

    # the localiser without the map images the judge puts on another wall
    bad = {m["name"] for m in doc["map_images"]["rows"] if m["refused"]}
    recs = list_images(work, loc_ref_cams(take, moving))
    F = Features(work, cache, FeatureSettings())
    images, _ = map_images(work, recs, map_settings(take.cfg))
    ls = loc_settings(take)
    ls = LocSettings(**{**ls.__dict__, "fix_focal": True, "plausible": False})  # the map focal; the judge gates instead
    L = Localizer(F, model, [r for r in images if r.name not in bad], ls)
    ref_pose = {}  # map image key -> (centre, view direction) in the model
    by_name = {im.name: im for im in model.images.values() if im.has_pose}
    for k, ref in L.refs.items():
        im = by_name.get(ref.rec.name)
        if im is not None:
            T = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
            Rr = np.asarray(T.rotation.matrix())
            ref_pose[k] = (centres(Rr, np.asarray(T.translation)), Rr[2])
    keys = list(ref_pose)
    RC = np.array([ref_pose[k][0] for k in keys]).reshape(-1, 3)
    RZ = np.array([ref_pose[k][1] for k in keys]).reshape(-1, 3)

    def spatial_refs(Rp, Cp, q):
        """The map images nearest the prior pose (Rp, Cp) + the moving camera's nearest in time."""
        if not keys:
            return L.candidates(q, cross=False)
        ang = np.degrees(np.arccos(np.clip(RZ @ Rp[2], -1.0, 1.0)))
        dist = 10.0 * np.linalg.norm(RC - Cp, axis=1) / room + ang / ANGLE_UNIT_DEG
        out = [L.refs[keys[k]] for k in np.argsort(dist)[:N_SPATIAL]]
        seen = {r.rec.key for r in out}
        out += [r for r in L.candidates(q, cross=False) if r.rec.key not in seen][:N_TIME]
        return out

    frames = [r for r in recs[moving] if r.kind == "win"]
    if len(frames) != len(t):
        raise SystemExit(f"[anchored] {moving}: {len(frames)} window frames vs {len(t)} in {src.name}")
    meta = json.loads((work / "frames" / moving / "frames.json").read_text())
    w, h = int(meta["width"]), int(meta["height"])
    f_map = float(np.median(f[anchors])) if anchors.any() else float(np.nanmedian(f[ok]))
    k1_map = float(np.median(k1[anchors])) if anchors.any() else 0.0

    # every frame's correspondences from the raw cache (accepted raw poses keep theirs)
    off_c = np.asarray(d["corr_off"])
    corr = {i: (np.asarray(d["corr_xy"][off_c[i]:off_c[i + 1]]), np.asarray(d["corr_xyz"][off_c[i]:off_c[i + 1]]))
            for i in range(len(t)) if off_c[i + 1] > off_c[i]}
    n_inl, rp = np.array(d["n_inliers"]), np.array(d["reproj_px"], float)
    tried_from = np.full(len(t), -1)  # the neighbour a frame was last tried from: retried only from another one
    for k_pass in range(MAX_PASSES):
        added = 0
        order = range(len(t)) if k_pass % 2 == 0 else range(len(t) - 1, -1, -1)
        step = 1 if k_pass % 2 == 0 else -1
        for i in order:
            if acc[i]:
                continue
            nb = [i - step * k for k in range(1, NEAR + 1) if 0 <= i - step * k < len(t) and acc[i - step * k]]
            if not nb or tried_from[i] == nb[0]:
                continue
            j = nb[0]
            tried_from[i] = j
            q = frames[i]
            qi, xyz, pid = [], [], []
            for ref in spatial_refs(R[j], C[j], q):
                a, b, c = _lift(F, q, ref, L.match_cache)
                qi.append(a), xyz.append(b), pid.append(c)
            if not qi:
                continue
            qi, xyz, pid = _dedupe(np.concatenate(qi), np.concatenate(xyz), np.concatenate(pid))
            if len(qi) < ls.min_inliers:
                continue
            kp = F.get(q)["kpts"][qi] + 0.5  # COLMAP px
            g = prior_gate(xyz, kp, R[j], tvec[j], f[j], k1[j], w, h)
            if g.sum() < ls.min_inliers:
                continue
            sol = solve_pose(kp[g], xyz[g], w, h, f_map, k1_map, ls)
            if sol is None or sol["n"] < ls.min_inliers:
                continue
            Rn, tn = sol["R"], sol["t"]
            Cn = centres(Rn, tn)
            if not motion_ok(C[j], R[j], t[j], Cn, Rn, t[i], room):
                continue
            r_i, x_i, _ = J.judge(t[i:i + 1], J.joints_at(t[i:i + 1]), Rn[None], tn[None], np.array([sol["f"]]),
                                  np.array([k1_map]))
            if J.refused(r_i, x_i)[0]:
                continue
            inl = np.asarray(sol["inliers"], bool)
            e = tv_reproj(Rn, tn, sol["f"], k1_map, w, h, xyz[g][inl], kp[g][inl])
            R[i], tvec[i], C[i], f[i], k1[i] = Rn, tn, Cn, sol["f"], k1_map
            ok[i], acc[i], how[i], res[i], xfer[i] = True, True, "prior", r_i[0], x_i[0]
            n_inl[i], rp[i] = int(sol["n"]), float(np.median(e)) if len(e) else float("nan")
            corr[i] = (kp[g][inl] - 0.5, xyz[g][inl])
            added += 1
        log(f"[anchored] pass {k_pass + 1}: +{added} frames by prior ({time.time() - t0:.0f} s)")
        if not added and k_pass >= 1:
            break

    # write: accepted frames ok; raw poses the judge refused 'dancers', raw poses it never reached (no dancers measurable,
    # no continuous accepted neighbour) 'unanchored', both not ok; frames the raw cache had not localised as they were
    out_ok = acc.copy()
    refused = judged_out & ~acc
    unreached = ok_raw & ~acc & ~judged_out
    src_raw = np.asarray(d.get("source", np.array([""] * len(t))))
    source = np.where(acc, how, np.where(refused, "dancers", np.where(unreached, "unanchored", src_raw)))
    poses = []
    for i in range(len(t)):
        fp = FramePose(ok=bool(out_ok[i]), R=R[i], t=tvec[i], f=float(f[i]), k1=float(k1[i]), n_inliers=int(n_inl[i]),
                       reproj_px=float(rp[i]), source=str(source[i]))
        if out_ok[i] and i in corr:
            xy, xyz = corr[i]
            fp.corr, fp.q_xy = Ref(frames[i], np.zeros(len(xyz), int), xyz, -np.ones(len(xyz), np.int64)), xy
        poses.append(fp)
    from .typed import LOC_MARKER_KEYS
    _save_raw(cache / f"loc_{moving}.npz", frames, poses, {k: d[k] for k in LOC_MARKER_KEYS if k in d})
    rep = {**{k: v for k, v in doc.items() if k != "map_images"},
           "map_images": {k: v for k, v in doc["map_images"].items() if k != "rows"}, "map_images_excluded": sorted(bad),
           "params": {"tau_px": TAU_PX, "loose": LOOSE, "xfer_anchor": XFER_ANCHOR, "xfer_loose": XFER_LOOSE,
                      "world": J.world is not None, "min_joints": MIN_JOINTS, "v_max_room": V_MAX_ROOM,
                      "w_max_deg": W_MAX_DEG, "prior_px": PRIOR_PX, "near": NEAR, "room": round(room, 3),
                      "f_map": round(f_map, 2)},
           "counts": {"anchor": int((how == "anchor").sum()), "cont": int((how == "cont").sum()),
                      "prior": int((how == "prior").sum()), "refused": int(refused.sum()), "unanchored": int(unreached.sum()),
                      "ok": int(out_ok.sum()),
                      "frames": int(len(t)), "share": round(float(out_ok.mean()), 4)},
           "coverage_8s": coverage(t, out_ok), "residual_after_pct": None, "seconds": round(time.time() - t0, 1),
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    ra, xa = res[out_ok & np.isfinite(res)], xfer[out_ok & np.isfinite(xfer)]
    if len(ra):
        rep["residual_after_pct"] = {str(q): round(float(np.percentile(ra, q)), 2) for q in (50, 90, 99)}
    if len(xa):
        rep["transfer_after_pct"] = {str(q): round(float(np.percentile(xa, q)), 1) for q in (50, 90, 99)}
    p = tv._write_json(work / "cameras" / "qa" / f"anchored_{moving}.json", rep)
    log(f"[anchored] -> {p}: ok {rep['counts']['ok']}/{len(t)} ({rep['counts']['share']:.1%}): anchors "
        f"{rep['counts']['anchor']}, grown {rep['counts']['cont']}, prior {rep['counts']['prior']}, refused {rep['counts']['refused']}")
    return rep


def tv_reproj(R, t, f, k1, w, h, xyz, kp):
    from .localize import reproj
    return reproj(R, t, f, k1, w, h, xyz, kp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.anchored", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("score", "run"))
    ap.add_argument("take")
    ap.add_argument("--moving", default="01")
    ap.add_argument("--fixed", default="02")
    ap.add_argument("--offset-ms", type=float, default=None,
                    help="the moving camera's offset against the fixed one (default: sync/offsets.json, curves_2v.json, 0)")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        from ..take import Take
        work = Take.load(a.take).work.resolve()
        doc = cmd_score(work, a.moving, a.fixed, a.offset_ms)
        tv._write_json(work / "cameras" / "qa" / f"anchored_score_{a.moving}.json", doc)
        return 0
    cmd_run(a.take, a.moving, a.fixed, a.offset_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
