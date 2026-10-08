"""Counterbalance from the two dancers' SMPL-X fits (head-movement VIEWER_SPEC 3.5 / 3.7 / 3.9).

A counterbalance: the follower's weight is offset by the leader's - they lean away from each other and hold each other
in tension through the connection, so their shared centre of mass sits between them. Its pivot form (the user's
description of the Larissa/Kadu take): the leader anchors one of the follower's feet at a point on the floor and walks a
circle around her, pivoting her in place. Two tiers are detected:

  counterbalance  lean AND connection (static, travelling or pivoting)  -> the yellow floor dot + axis (VIEWER_SPEC 3.9)
  pivot           counterbalance AND anchor AND circling                 -> the pivot ring + leader arc (3.5)

    python -m dancecap.counterbalance detect --take larissa_kadu     -> work/<take>/counterbalance/counterbalance.json
                                                                        + work/<take>/review/counterbalance.jpg (scored
                                                                        against truth.json / counterbalance/truth_user.json
                                                                        when present)
    python -m dancecap.counterbalance synthetic                       -> work/synthetic_cb/ (ground truth) + capture
                                                                        03_SyntheticCounterbalance, detected + scored
    python -m dancecap.counterbalance export --take synthetic_cb      -> StreamingAssets/<capture>/counterbalance.json
    python -m dancecap.counterbalance review --take larissa_kadu --smplx-dir smplx_v1
                                                                      -> review/counterbalance.jpg (+ _cams.jpg)

Inputs: work/<take>/smplx/{lead,follow}.npz (else smplx_v1/; contract: Y-up right-handed world, metres, floor y=0,
`t` = reference seconds on the shared master grid), the beat grid (timing/timing.json or beats.json; 8 zouk eighths
per measure, so one beat = two grid entries), optional dancer masses in takes/<take>.toml ([dancers.<role>] mass_kg).
A time grid with gaps (> 1.5 frames) is analysed per contiguous segment; `coverage` lists the analysed ranges.

Per-frame criteria, each in 0..1 with linear ramps between the thresholds of CBConfig (rates per BEAT, so they follow
the tempo); anchor / circling use windows one beat long, centred; lean / connection are per frame:
  anchor      a follower foot in floor contact (lower of ankle / ball height above that dancer's floor estimate,
              4..9 cm) whose PIVOT POINT drifts < 3..8 cm over the window (length of a straight-line fit's
              displacement across the window: a slide or a step counts, frame-to-frame fit jitter averages out)
              while her pelvis yaw turns 8..20 deg/beat. The pivot point is the point on
              the heel..toe axis (ankle + s (ball - ankle), s in [-0.2, 1.2]) that moves least in the window - closed-form
              least squares - so a ball-of-foot pivot is found with the heel swinging.
  circling    the leader's pelvis sweeps an angle about that pivot point: 10..25 deg/beat, one sign (60..90 % of the
              frame steps), mean radius within 0.35..1.6 m (soft edges 0.25 / 1.8) and nearly constant (CV 8..25 %).
  lean        (lean_mode 'counterweight', the default since 2026-10-07) one dancer HANGS and the partner is the
              counterweight. Hang = standing (a foot in contact) with the COM BEYOND THE EDGE of their own support away
              from the partner - COM minus the furthest ankle / ball joint of the feet in contact along the line between
              the two COMs, 3..9 cm (0.5 at 6 cm ~ the sole edge: heel / toe tip lie 5-7 cm beyond those joints) - AND
              away from their own contact-weighted mid-foot by 4..12 cm (smooth while a foot lands; the support edge
              jumps). Counterweight = the partner's COM not ahead of their own mid-foot towards the hanger (-2..+2 cm).
              Moment balance: a heavier or bracing partner (centre of pressure towards the heels) holds the tension
              inside their own support, so the partner need not hang too; the better role assignment counts.
              Why not 'both' (both COMs >= 4..12 cm from their own mid-foot, the rule before 2026-10-07): on
              larissa_kadu_38 the user-labelled counterbalance (follower on one foot hanging ~30 cm out, leader braced,
              his lean 0-4 cm at the edges) scored 0 at its edges, while a wide lunge (COM 13 cm from the mid-foot but
              between the feet) passed it.
  connection  a hand-hand pair at 20..8 cm (hand centre = wrist + finger joints, as physics) with straight arms (elbow
              extension ratio |shoulder-wrist| / (|shoulder-elbow| + |elbow-wrist|) 0.85..0.97); the pair score is the
              lowest of distance and both arms' extension scores; or a hand on the partner's body (25..12 cm, weight
              0.85) with ANY elbow angle: an embrace frame (his arm around her back, her hand on his shoulder) carries
              the tension with a bent elbow (hand_body_extension=False since 2026-10-07; True = the old straight-arm
              requirement, which rejected the labelled close-embrace counterbalance). Tension vs a mere touch is decided
              by the lean criterion: in an ordinary embrace nobody hangs.
Tier scores are strict ANDs: counterbalance = min(lean, connection); pivot = min(counterbalance, anchor, circling), so a
weak criterion cannot be outvoted by strong ones. Intervals by hysteresis (on >= 0.5, held while >= 0.35), gaps up to
0.5 beat merged, runs < 0.5 beat dropped - and runs with fewer than 0.5 beat of frames AT the on level (a row of
blips bridged by the merge is not a held figure); pivot runs are also merged across gaps up to 1 beat when the pivot
point stays within 5 cm (fit jitter must not blink the pivot off). Every pivot lies inside a counterbalance interval.

Real-data check (2026-10-07): larissa_kadu_38, the user's label HUD 32.0-36.15 s = reference 62.151-66.301 s
(counterbalance/truth_user.json): before (both-lean, straight-arm frame) nothing was detected (max score 0.52); after,
one pivot interval 62.142-66.186 s (IoU 0.97, edges -0.3 / -3.4 frames, 700 deg cw about her left foot, leader radius
0.39 m) and nothing else in the 38 s; larissa_kadu (56.1-61.9 s) none before and after.

Edges (VIEWER_SPEC 14: the axis within +-1 frame of the truth): the window-based scores blur the edges by half a beat,
so they are refined after the hysteresis -
  counterbalance  t0 / t1 = where the per-frame counterbalance score crosses `on` (0.5), linearly interpolated between
                  frames (no window, so no lag).
  pivot           t0 / t1 = where the leader's angular speed about the pivot point (from the 5 Hz-smoothed pelvis)
                  crosses 10 % of its median inside the pivot (at least 3 deg/s) - or, when fit jitter keeps it above
                  that, where it bottoms out (first local minimum below twice that) - searched within +-0.75 beat of
                  the coarse edge and walked out from 0.5 beat inside it; clamped into the counterbalance interval.
Per pivot: the pivot foot is the anchor-weighted majority, the pivot point the median of that foot's least-moving
point over the coarse run (y = 0, the floor), swept_deg / direction the leader pelvis' unwrapped angle about it from t0
to t1 (ccw = counter-clockwise seen from above = positive about +Y in the right-handed world), leader_radius_m its
median distance.

Combined COM = mass-weighted COM of both bodies: each from the physics_body rigid mass elements (shaped SMPL-X,
voxelised, uniform density; mass = takes/<take>.toml mass_kg, else closed-mesh volume x 1010 kg/m^3, the physics
default), from the SMPL-X parameters (never from noisy joints); joint-based segment COM (de Leva-style weights,
stature-based masses) only when the parameters / body model are unavailable - then a RuntimeWarning is raised and the
reason is recorded in params.warnings / params.com_source.

Thresholds are documented engineering choices (one human label so far, above): drift 3..8 cm covers fit
jitter of a planted foot (~1-3 cm) while rejecting a stepping foot; 8..20 deg/beat separates a pivot from the ordinary
slow rotation of a couple in closed position only together with the lean and connection criteria (the synthetic
ordinary dancing rotates ~22 deg/beat with planted feet and is rejected by lean + connection); 4..12 cm lean is about
one foot length beyond normal postural sway (+-2-3 cm; the synthetic ordinary dancing stays within 2 cm).
docs/MOVES_RESEARCH.md covers move vocabulary, not these thresholds; revisit them against human labels once the move
timeline exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE

ROLES = ("lead", "follow")
FEET = {"left": (7, 10), "right": (8, 11)}        # (ankle, ball = SMPL-X 'foot') joints
ARMS = {"l": (16, 18, 20), "r": (17, 19, 21)}      # shoulder, elbow, wrist
BODY_POINTS = {"spine3": 9, "neck": 12, "l_shoulder": 16, "r_shoulder": 17, "pelvis": 0}
UNITY_STREAMING = WORKSPACE / "head-movement" / "Assets" / "StreamingAssets"
DEFAULT_CAPTURES = {"synthetic_cb": "03_SyntheticCounterbalance", "larissa_kadu": "02_LarissaKadu",
                    "larissa_kadu_38": "04_LarissaKaduDemo", "synthetic": "01_SyntheticSMPLX"}
CRITERIA = ("anchor", "circling", "lean", "connection")
KINDS = ("counterbalance", "pivot")
# de Leva (1996)-style segment weights onto SMPL-X joints for the joints-only COM fallback (segment = mean of joints)
JOINT_SEGMENTS = [(0.068, (15,)), (0.43, (0, 3, 6, 9)), (0.026, (16, 18)), (0.026, (17, 19)), (0.015, (18, 20)),
                  (0.015, (19, 21)), (0.006, (20,)), (0.006, (21,)), (0.145, (1, 4)), (0.145, (2, 5)),
                  (0.045, (4, 7)), (0.045, (5, 8)), (0.013, (7, 10)), (0.013, (8, 11))]


@dataclass
class CBConfig:
    cutoff_hz: float = 5.0                  # smoothing spline on joints / COM (as physics)
    window_beats: float = 1.0
    default_bpm: float = 76.4               # when the take has no beat grid
    floor_pct: float = 5.0                  # per-dancer floor = this percentile of the lower foot joint
    contact_h_m: tuple = (0.04, 0.09)       # foot height above floor: full contact .. none
    pivot_s_range: tuple = (-0.2, 1.2)      # pivot point along ankle(0) .. ball(1) axis
    drift_m: tuple = (0.03, 0.08)           # pivot point drift in the window: good .. bad
    drift_mode: str = "linear"              # 'linear': displacement of a straight-line fit across the window
                                            # (fit jitter averages out); 'p90': 90th pct distance to the median
    yaw_deg_per_beat: tuple = (8.0, 20.0)   # follower pelvis yaw rate: none .. full
    sweep_deg_per_beat: tuple = (10.0, 25.0)
    sign_consistency: tuple = (0.6, 0.9)
    radius_cv: tuple = (0.08, 0.25)         # good .. bad
    radius_m: tuple = (0.25, 0.35, 1.6, 1.8)
    lean_m: tuple = (0.04, 0.12)            # lean_mode 'both': COM away from the own mid-foot, none .. full
    lean_mode: str = "counterweight"        # 'counterweight' (default): one dancer HANGS (COM beyond the edge of their
                                            # own support, overhang_m) and the partner is the counterweight (not
                                            # leaning towards them, counter_lean_m); 'both': both dancers' COM >= lean_m
                                            # from their own mid-foot (the rule before 2026-10-07)
    overhang_m: tuple = (0.03, 0.09)        # COM beyond the furthest ankle / ball joint of the feet in contact: 0.5 at
                                            # 6 cm ~ the sole edge (heel / toe tip 5-7 cm beyond those joints)
    counter_lean_m: tuple = (-0.02, 0.02)   # the partner's COM vs own mid-foot: towards the hanger (bad) .. upright /
                                            # away (good)
    support_soft_m: float = 2.0             # a foot with contact weight w counts (1 - w) x this short of its position
                                            # (> any leg reach: a lifted, extended leg never widens the support)
    supported_w: tuple = (0.3, 0.7)         # the hanger must stand: max foot contact weight, none .. full
    hand_dist_m: tuple = (0.08, 0.20)       # good .. bad
    hand_body_dist_m: tuple = (0.12, 0.25)
    extension: tuple = (0.85, 0.97)
    hand_body_weight: float = 0.85
    hand_body_extension: bool = False       # True: a hand on the partner's body also needs a straight arm (pre-10-07);
                                            # False: an embrace frame carries the tension with a bent elbow
    on: float = 0.5
    off: float = 0.35
    min_beats: float = 0.5
    merge_beats: float = 0.5                # gaps up to this are bridged (both tiers)
    pivot_merge_beats: float = 1.0          # pivot tier: longer gaps bridged when the pivot point stays put
    pivot_merge_m: float = 0.05
    onset_frac: float = 0.1                 # pivot edges: leader angular speed crosses this x its median
    onset_min_deg_s: float = 3.0
    onset_search_beats: float = 0.75
    gap_frames: float = 1.5                 # time steps > this x the median step split the grid into segments
    com_margin_s: float = 0.3               # com_track extends this far beyond the interval (viewer fades)
    voxel_m: float = 0.007


# ----------------------------------------------------------------------------------------------- helpers

def _up(x, lo, hi):
    """0 at lo, 1 at hi (linear, clipped)."""
    return np.clip((np.asarray(x, float) - lo) / (hi - lo), 0.0, 1.0)


def _down(x, good, bad):
    """1 at good, 0 at bad (linear, clipped)."""
    return np.clip((bad - np.asarray(x, float)) / (bad - good), 0.0, 1.0)


def _bridge(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linearly interpolate non-finite frames of (T, ...) along time (nearest value at the ends)."""
    a = np.asarray(a, float).copy()
    flat = a.reshape(len(a), -1)
    good = np.isfinite(flat).all(1)
    if good.all():
        return a
    if good.sum() < 2:
        raise ValueError("counterbalance: fewer than 2 finite frames")
    for c in range(flat.shape[1]):
        flat[~good, c] = np.interp(t[~good], t[good], flat[good, c])
    return flat.reshape(a.shape)


def _r(x, nd=4):
    if x is None:
        return None
    a = np.round(np.asarray(x, float), nd)
    return np.where(np.isfinite(a), a, np.nan).tolist() if a.ndim else (float(a) if np.isfinite(a) else None)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def hand_center(joints: np.ndarray, side: str) -> np.ndarray:
    from .physics import _hand_center
    return _hand_center(joints, side)


def arm_extension(joints: np.ndarray, side: str) -> np.ndarray:
    """|shoulder - wrist| / (|shoulder - elbow| + |elbow - wrist|): 1 = straight arm."""
    s, e, w = (joints[:, j] for j in ARMS[side])
    return np.linalg.norm(s - w, axis=1) / np.maximum(np.linalg.norm(s - e, axis=1) + np.linalg.norm(e - w, axis=1),
                                                      1e-6)


def pelvis_yaw(joints: np.ndarray) -> np.ndarray:
    """Unwrapped yaw of the hip line (rad, + = CCW seen from above)."""
    v = joints[:, 1] - joints[:, 2]
    return np.unwrap(np.arctan2(-v[:, 2], v[:, 0]))


def floor_angle(v: np.ndarray) -> np.ndarray:
    """Angle (rad) of horizontal (x, z) vectors, + = CCW seen from above (right-handed about +Y)."""
    return np.arctan2(-v[..., 1], v[..., 0])


def pivot_s(a: np.ndarray, d: np.ndarray, lo: float, hi: float) -> float:
    """s minimising the spread of a + s d over frames (a, d: (n, 2)); 1 (the ball) when the foot does not turn."""
    ac, dc = a - a.mean(0), d - d.mean(0)
    den = float((dc ** 2).sum())
    if den < 1e-8:
        return 1.0
    return float(np.clip(-(ac * dc).sum() / den, lo, hi))


def point_drift(t: np.ndarray, p: np.ndarray, mode: str = "linear") -> float:
    """How far a floor point (n, 2) moves over a window: 'linear' = length of the least-squares straight-line
    displacement from the first to the last time (systematic slide / step; i.i.d. fit jitter averages out),
    'p90' = 90th percentile of the distances to the median point."""
    if mode == "p90" or len(t) < 3:
        return float(np.percentile(np.linalg.norm(p - np.median(p, 0), axis=1), 90))
    tc = t - t.mean()
    den = float((tc ** 2).sum())
    v = (tc[:, None] * (p - p.mean(0))).sum(0) / max(den, 1e-12)
    return float(np.linalg.norm(v) * (t[-1] - t[0]))


def joints_com(joints: np.ndarray) -> np.ndarray:
    w = np.array([s[0] for s in JOINT_SEGMENTS])
    c = np.stack([joints[:, list(s[1])].mean(1) for s in JOINT_SEGMENTS], 1)
    return (w[None, :, None] * c).sum(1) / w.sum()


def body_com(npz: dict, mass: float | None, cfg: CBConfig) -> tuple[np.ndarray, float, str, np.ndarray]:
    """(COM (T, 3), mass kg, mass source, FK joints (T, 55, 3)) from the SMPL-X parameters via physics_body."""
    from . import physics as ph
    from . import physics_body as pb
    pc = ph.PhysicsConfig(voxel_m=cfg.voxel_m)
    d, _, _, _ = ph.sanitize(npz, pc)
    body = ph._body(d, mass, cfg.voxel_m)
    R, p = pb.forward_kinematics(body, pb.full_pose(d), np.asarray(d["transl"], float))
    c, _ = pb.element_world(body, R, p)
    com = (body.elem_mass[None, :, None] * c).sum(1) / body.mass
    return com, body.mass, body.mass_source, np.einsum("tjab,jb->tja", R, body.j_rest) + p


# ----------------------------------------------------------------------------------------------- runs and edges

def _runs(score: np.ndarray, on: float, off: float) -> list[list[int]]:
    """Runs [i0, i1] (inclusive) where score >= off that reach `on` somewhere."""
    above = score >= off
    runs, i, T = [], 0, len(score)
    while i < T:
        if above[i]:
            j = i
            while j + 1 < T and above[j + 1]:
                j += 1
            if score[i:j + 1].max() >= on:
                runs.append([i, j])
            i = j + 1
        else:
            i += 1
    return runs


def _merge_runs(runs: list, gap: int, ok=None) -> list[list[int]]:
    """Merge consecutive runs separated by <= gap frames (and ok(prev, next) when given)."""
    merged: list[list[int]] = []
    for r in sorted([list(r) for r in runs]):
        if merged and r[0] - merged[-1][1] - 1 <= gap and (ok is None or ok(merged[-1], r)):
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return merged


def hysteresis(score: np.ndarray, on: float, off: float, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    """Runs (i0, i1 inclusive) where score >= off that reach on somewhere; gaps <= merge_gap merged; short runs dropped."""
    merged = _merge_runs(_runs(score, on, off), merge_gap)
    return [(a, b) for a, b in merged if b - a + 1 >= min_len]


def contiguous_segments(t: np.ndarray, gap_frames: float = 1.5) -> list[tuple[int, int]]:
    """[(i0, i1 inclusive)] of the time grid split where a step exceeds gap_frames x the median step."""
    if len(t) < 2:
        return [(0, len(t) - 1)]
    d = np.diff(t)
    cut = np.flatnonzero(d > gap_frames * float(np.median(d)))
    starts = np.concatenate([[0], cut + 1])
    ends = np.concatenate([cut, [len(t) - 1]])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def level_edges(t: np.ndarray, y: np.ndarray, level: float, i0: int, i1: int) -> tuple[float, float]:
    """Interpolated times where y first rises to `level` and last falls below it inside [i0, i1] (y >= level there)."""
    idx = np.flatnonzero(y[i0:i1 + 1] >= level) + i0
    if not len(idx):
        return float(t[i0]), float(t[i1])
    a, b = int(idx[0]), int(idx[-1])
    t0 = float(t[a]) if a == 0 or y[a - 1] >= level else \
        float(t[a - 1] + (level - y[a - 1]) / (y[a] - y[a - 1]) * (t[a] - t[a - 1]))
    t1 = float(t[b]) if b == len(t) - 1 or y[b + 1] >= level else \
        float(t[b] + (y[b] - level) / (y[b] - y[b + 1]) * (t[b + 1] - t[b]))
    return t0, t1


def _walk_crossing(t: np.ndarray, y: np.ndarray, thr: float, start: int, lo: int, hi: int,
                   direction: int, bottom: float | None = None) -> float | None:
    """From `start` (moved inwards = against `direction` until y >= thr), walk in `direction` while y >= thr; the
    interpolated time where y drops below thr, or - once y is below `bottom` - the first local minimum (the ramp
    bottomed out above thr: fit jitter / a slow tail), or the search limit (lo / hi)."""
    j = int(np.clip(start, lo, hi))
    while y[j] < thr:
        j -= direction
        if not lo <= j <= hi:
            return None
    while lo <= j + direction <= hi:
        k = j + direction
        if y[k] < thr:
            return float(t[j] + (y[j] - thr) / (y[j] - y[k]) * (t[k] - t[j]))
        if bottom is not None and y[j] < bottom and y[k] > y[j]:
            return float(t[j])
        j = k
    return float(t[j])


def circling_edges(t: np.ndarray, phi: np.ndarray, p0: int, p1: int, beat: float,
                   cfg: CBConfig, lo_i: int = 0, hi_i: int | None = None) -> tuple[float, float, dict]:
    """Pivot onset / offset: where the leader's angular speed about the pivot (deg/s, in the circling direction)
    crosses onset_frac x its median over the coarse run [p0, p1], searched within onset_search_beats of each edge."""
    dt = float(np.median(np.diff(t)))
    w = np.degrees(np.gradient(phi, t))
    sgn = float(np.sign(np.median(w[p0:p1 + 1]))) or 1.0
    wd = sgn * w
    plateau = float(np.median(wd[p0:p1 + 1]))
    thr = max(cfg.onset_frac * plateau, cfg.onset_min_deg_s)
    S = max(1, int(round(cfg.onset_search_beats * beat / dt)))
    h = max(1, int(round(0.5 * beat / dt)))
    mid = (p0 + p1) // 2
    T = len(t)
    hi_i = T - 1 if hi_i is None else hi_i
    on = _walk_crossing(t, wd, thr, min(p0 + h, mid), max(lo_i, p0 - S), mid, -1, 2 * thr)
    off = _walk_crossing(t, wd, thr, max(p1 - h, mid), mid, min(hi_i, p1 + S), +1, 2 * thr)
    info = {"method": f"leader angular speed about the pivot crosses {cfg.onset_frac:g} x its median "
                      f"(>= {cfg.onset_min_deg_s:g} deg/s), or bottoms out below twice that",
            "plateau_deg_s": round(plateau, 3), "threshold_deg_s": round(thr, 3),
            "refined": [on is not None, off is not None]}
    return (float(t[p0]) if on is None else on), (float(t[p1]) if off is None else off), info


# ----------------------------------------------------------------------------------------------- criteria

def foot_contacts(J: dict, cfg: CBConfig) -> tuple[dict, dict]:
    """Contact weight (T,) and mid-foot floor point (T, 2) per (role, foot) from the ankle / ball joint heights above
    that dancer's floor estimate (floor_pct percentile of the lower foot joint)."""
    contact, mid = {}, {}
    for role, Jr in J.items():
        fl_a = np.percentile(np.minimum(Jr[:, 7, 1], Jr[:, 8, 1]), cfg.floor_pct)
        fl_b = np.percentile(np.minimum(Jr[:, 10, 1], Jr[:, 11, 1]), cfg.floor_pct)
        for f, (ia, ib) in FEET.items():
            hgt = np.minimum(Jr[:, ia, 1] - fl_a, Jr[:, ib, 1] - fl_b)
            contact[(role, f)] = _down(hgt, *cfg.contact_h_m)
            mid[(role, f)] = 0.5 * (Jr[:, ia] + Jr[:, ib])[:, [0, 2]]
    return contact, mid


def lean_displacement(com: dict, contact: dict, mid: dict) -> dict:
    """Per role (T,): COM displacement (m) away from the partner relative to the contact-weighted centre of the own
    feet, along the horizontal line between the two COMs."""
    u = (com["follow"] - com["lead"])[:, [0, 2]]
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
    disp = {}
    for role, sgn in (("follow", 1.0), ("lead", -1.0)):
        w = np.stack([contact[(role, f)] for f in FEET], 1) + 1e-3
        bos = sum(w[:, k:k + 1] * mid[(role, f)] for k, f in enumerate(FEET)) / w.sum(1, keepdims=True)
        disp[role] = ((com[role][:, [0, 2]] - bos) * (sgn * u)).sum(1)
    return disp


def support_overhang(com: dict, contact: dict, J: dict, cfg: CBConfig) -> tuple[dict, dict]:
    """({role: (T,) overhang m}, {role: (T,) support weight}). Overhang = how far the COM's floor projection lies
    BEYOND the far edge of the dancer's own support, away from the partner (along the horizontal line between the two
    COMs; < 0 = inside): COM minus the furthest ankle / ball joint of the feet in contact. A foot with contact weight
    w counts (1 - w) x support_soft_m short of its position, so the edge moves continuously while a foot lifts or
    lands. Support weight = the larger foot contact weight (0 = airborne: nothing to hang from)."""
    u = (com["follow"] - com["lead"])[:, [0, 2]]
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
    over, sup = {}, {}
    for role, sgn in (("follow", 1.0), ("lead", -1.0)):
        ua = sgn * u
        c = (com[role][:, [0, 2]] * ua).sum(1)
        edge = np.full(len(c), -np.inf)
        for f, (ia, ib) in FEET.items():
            w = contact[(role, f)]
            for j in (ia, ib):
                edge = np.maximum(edge, (J[role][:, j][:, [0, 2]] * ua).sum(1) - (1.0 - w) * cfg.support_soft_m)
        over[role] = c - edge
        sup[role] = np.maximum(contact[(role, "left")], contact[(role, "right")])
    return over, sup


def lean_score(disp: dict, cfg: CBConfig, over: dict | None = None, sup: dict | None = None) -> np.ndarray:
    """Couple lean score (T,).
    'counterweight' (default): a dancer HANGS - standing (support weight, supported_w) with the COM beyond the edge of
    their own support away from the partner (overhang_m), i.e. they would fall without the partner, and away from
    their own mid-foot by lean_m (the contact-weighted centre lean moves smoothly while a foot lands; the support edge
    jumps, so the centre lean keeps a flickering landing contact from re-opening the hang) - AND the partner
    is the counterweight: not leaning towards the hanger (centre lean disp >= counter_lean_m). By moment balance a
    heavier or bracing partner (centre of pressure towards the heels) holds the tension inside their own support, so
    the partner need not hang too. The better of the two role assignments counts.
    'both': both dancers' COM >= lean_m from their own mid-foot (the lower of the two ramp scores; a wide stance
    passes it without hanging - why it is no longer the default)."""
    f, l = np.asarray(disp["follow"], float), np.asarray(disp["lead"], float)
    if cfg.lean_mode == "both":
        return np.minimum(_up(f, *cfg.lean_m), _up(l, *cfg.lean_m))
    if cfg.lean_mode != "counterweight":
        raise ValueError(f"lean_mode {cfg.lean_mode!r}: 'counterweight' or 'both'")
    if over is None or sup is None:
        raise ValueError("lean_mode 'counterweight' needs the support overhang (support_overhang)")
    out = np.zeros(len(f))
    for hanger, partner in (("follow", "lead"), ("lead", "follow")):
        hang = np.minimum(np.minimum(_up(over[hanger], *cfg.overhang_m), _up(disp[hanger], *cfg.lean_m)),
                          _up(sup[hanger], *cfg.supported_w))
        out = np.maximum(out, np.minimum(hang, _up(disp[partner], *cfg.counter_lean_m)))
    return out


def connection_scores(J: dict, cfg: CBConfig) -> tuple[list, np.ndarray, np.ndarray]:
    """(pairs [(name, kind)], score (T, P), distance (T, P)): hand-hand pairs score min(distance, both arms'
    extension); a hand on the partner's body scores hand_body_weight x min(distance, that arm's extension)."""
    hands = {r: {s: hand_center(J[r], s) for s in ARMS} for r in ROLES}
    ext = {r: {s: _up(arm_extension(J[r], s), *cfg.extension) for s in ARMS} for r in ROLES}
    pairs, pscore, pdist = [], [], []
    for ls in ARMS:
        for fs in ARMS:
            d = np.linalg.norm(hands["lead"][ls] - hands["follow"][fs], axis=1)
            pairs.append((f"lead_{ls}_hand-follow_{fs}_hand", "hand-hand"))
            pscore.append(np.minimum(_down(d, *cfg.hand_dist_m), np.minimum(ext["lead"][ls], ext["follow"][fs])))
            pdist.append(d)
    for r, other in (("lead", "follow"), ("follow", "lead")):
        for s in ARMS:
            for name, j in BODY_POINTS.items():
                d = np.linalg.norm(hands[r][s] - J[other][:, j], axis=1)
                pairs.append((f"{r}_{s}_hand-{other}_{name}", "hand-body"))
                sd = _down(d, *cfg.hand_body_dist_m)
                pscore.append(cfg.hand_body_weight * (np.minimum(sd, ext[r][s]) if cfg.hand_body_extension else sd))
                pdist.append(d)
    return pairs, np.stack(pscore, 1), np.stack(pdist, 1)


def best_pairs(pscore: np.ndarray, pdist: np.ndarray) -> np.ndarray:
    """Per frame the highest-scoring pair; ties (e.g. every pair 0 with bent arms) go to the nearest pair, not to
    pair 0."""
    return (pscore - 1e-6 * pdist).argmax(1)


def tier_scores(scores: dict) -> tuple[np.ndarray, np.ndarray]:
    """(counterbalance, pivot) tier scores: strict ANDs (minimum) of the criteria."""
    cbs = np.minimum(scores["lean"], scores["connection"])
    return cbs, np.minimum(cbs, np.minimum(scores["anchor"], scores["circling"]))


# ----------------------------------------------------------------------------------------------- detection

def detect(npzs: dict, cfg: CBConfig | None = None, beat_s: float | None = None, masses: dict | None = None,
           statures: dict | None = None, beat_times: np.ndarray | None = None) -> dict:
    """Per-frame criterion / tier scores and counterbalance intervals for {role: smplx npz dict}. Arrays on npz['t'];
    a grid with gaps is analysed per contiguous segment and the results concatenated."""
    cfg = cfg or CBConfig()
    t = np.asarray(npzs["lead"]["t"], float)
    tf = np.asarray(npzs["follow"]["t"], float)
    if len(t) != len(tf) or np.abs(t - tf).max() > 1e-6:
        raise ValueError("lead and follow must share the master time grid")
    segs = contiguous_segments(t, cfg.gap_frames)
    if len(segs) == 1:
        return _detect_segment(npzs, cfg, beat_s, masses, statures, beat_times)
    parts = [_detect_segment(_slice_npzs(npzs, a, b, len(t)), cfg, beat_s, masses, statures, beat_times)
             for a, b in segs if b - a + 1 >= 8]
    return _concat_results(parts)


def _slice_npzs(npzs: dict, a: int, b: int, T: int) -> dict:
    out = {}
    for role, npz in npzs.items():
        d = {}
        for k, v in npz.items():
            arr = np.asarray(v) if not isinstance(v, str) else v
            d[k] = arr[a:b + 1] if (k != "betas" and isinstance(arr, np.ndarray) and arr.ndim >= 1
                                    and arr.shape[0] == T) else v
        out[role] = d
    return out


def _cat(vals: list):
    v0 = vals[0]
    if isinstance(v0, np.ndarray) and v0.ndim >= 1:
        return np.concatenate(vals)
    if isinstance(v0, dict):
        return {k: _cat([v[k] for v in vals]) for k in v0}
    return v0


def _concat_results(parts: list[dict]) -> dict:
    keep = {"intervals", "coverage", "runs", "pivot_runs", "beat_times", "pairs", "cfg", "info"}
    res = {k: _cat([p[k] for p in parts]) for k in parts[0] if k not in keep}
    res.update({"intervals": [iv for p in parts for iv in p["intervals"]],
                "coverage": [c for p in parts for c in p["coverage"]],
                "beat_times": parts[0]["beat_times"], "pairs": parts[0]["pairs"], "cfg": parts[0]["cfg"],
                "info": {**parts[0]["info"], "segments": len(parts)}})
    off, runs, pruns = 0, [], []
    for p in parts:
        runs += [(a + off, b + off) for a, b in p["runs"]]
        pruns += [(a + off, b + off) for a, b in p["pivot_runs"]]
        off += len(p["t"])
    res["runs"], res["pivot_runs"] = runs, pruns
    return res


def _detect_segment(npzs: dict, cfg: CBConfig, beat_s: float | None, masses: dict | None, statures: dict | None,
                    beat_times: np.ndarray | None) -> dict:
    from .physics_spline import Smoother
    masses, statures = masses or {}, statures or {}
    t = np.asarray(npzs["lead"]["t"], float)
    T = len(t)
    dt = float(np.median(np.diff(t)))
    beat = float(beat_s) if beat_s else 60.0 / cfg.default_bpm
    S = Smoother(t, cfg.cutoff_hz)

    J, com, info = {}, {}, {"com_source": {}, "mass_kg": {}, "mass_source": {}, "warnings": []}
    for role in ROLES:
        npz = npzs[role]
        raw_j = np.asarray(npz["joints"], float)[:, :55] if "joints" in npz else None
        try:
            c, m, msrc, fkj = body_com(npz, masses.get(role), cfg)
            info["com_source"][role] = "physics_body mass elements (SMPL-X parameters)"
        except Exception as e:  # noqa: BLE001  (no parameters / body model: joints-only fallback)
            if raw_j is None:
                raise
            msg = (f"{role}: SMPL-X body COM unavailable ({type(e).__name__}: {e}); using the joint-segment COM "
                   "fallback (de Leva-style weights; less accurate)")
            warnings.warn(f"counterbalance: {msg}", RuntimeWarning, stacklevel=3)
            info["warnings"].append(msg)
            c = joints_com(_bridge(raw_j, t))
            st = statures.get(role)
            m = masses.get(role) or (22.0 * st * st if st else (75.0 if role == "lead" else 58.0))
            msrc = "user input" if masses.get(role) else ("stature x BMI 22" if st else "default")
            fkj = None
            info["com_source"][role] = f"joint segments (fallback: {type(e).__name__}: {e})"
        jj = raw_j if raw_j is not None and np.isfinite(raw_j).mean() > 0.5 else fkj
        J[role] = S(_bridge(jj, t))
        com[role] = S(_bridge(c, t))
        info["mass_kg"][role] = float(m)
        info["mass_source"][role] = msrc
    m_l, m_f = info["mass_kg"]["lead"], info["mass_kg"]["follow"]
    com_c = (m_l * com["lead"] + m_f * com["follow"]) / (m_l + m_f)

    h = max(1, int(round(0.5 * cfg.window_beats * beat / dt)))
    win = [(max(0, i - h), min(T, i + h + 1)) for i in range(T)]
    Jf, Jl = J["follow"], J["lead"]
    contact, mid = foot_contacts(J, cfg)

    # (a) anchor: least-moving point of a follower foot while she turns
    yaw_f = pelvis_yaw(Jf)
    feet = {}
    for f, (ia, ib) in FEET.items():
        a = Jf[:, ia][:, [0, 2]]
        d = Jf[:, ib][:, [0, 2]] - a
        piv = np.zeros((T, 2))
        drift, yawr, cont, ss = (np.zeros(T) for _ in range(4))
        for i, (lo, hi) in enumerate(win):
            s = pivot_s(a[lo:hi], d[lo:hi], *cfg.pivot_s_range)
            p = a[lo:hi] + s * d[lo:hi]
            piv[i] = np.median(p, 0)
            drift[i] = point_drift(t[lo:hi], p, cfg.drift_mode)
            nb = max((t[hi - 1] - t[lo]) / beat, 1e-6)
            yawr[i] = abs(math.degrees(yaw_f[hi - 1] - yaw_f[lo])) / nb
            cont[i] = contact[("follow", f)][lo:hi].mean()
            ss[i] = s
        sc = cont * _down(drift, *cfg.drift_m) * _up(yawr, *cfg.yaw_deg_per_beat)
        feet[f] = {"score": sc, "pivot": piv, "drift": drift, "yaw_rate": yawr, "contact": cont, "s": ss}
    fi = np.where(feet["left"]["score"] >= feet["right"]["score"], 0, 1)
    anchor = np.where(fi == 0, feet["left"]["score"], feet["right"]["score"])
    pivot = np.where(fi[:, None] == 0, feet["left"]["pivot"], feet["right"]["pivot"])

    # (b) circling: leader pelvis about the pivot point
    Lp = Jl[:, 0][:, [0, 2]]
    circ, sweep_rate, radius = np.zeros(T), np.zeros(T), np.zeros(T)
    lo_r, lo_r1, hi_r1, hi_r = cfg.radius_m
    for i, (lo, hi) in enumerate(win):
        v = Lp[lo:hi] - pivot[i]
        phi = np.unwrap(floor_angle(v))
        sw = phi[-1] - phi[0]
        nb = max((t[hi - 1] - t[lo]) / beat, 1e-6)
        rate = abs(math.degrees(sw)) / nb
        st = np.diff(phi)
        cons = float(np.mean(np.sign(st) == np.sign(sw))) if sw != 0 and len(st) else 0.0
        r = np.linalg.norm(v, axis=1)
        rm = float(r.mean())
        cv = float(r.std() / max(rm, 1e-6))
        circ[i] = float(_up(rate, *cfg.sweep_deg_per_beat) * _up(cons, *cfg.sign_consistency)
                        * _down(cv, *cfg.radius_cv) * _up(rm, lo_r, lo_r1) * _down(rm, hi_r1, hi_r))
        sweep_rate[i], radius[i] = rate, rm

    # (c) lean and (d) connection: per frame
    disp = lean_displacement(com, contact, mid)
    over, sup = support_overhang(com, contact, J, cfg)
    lean = lean_score(disp, cfg, over, sup)
    pairs, pscore, pdist = connection_scores(J, cfg)
    best = best_pairs(pscore, pdist)
    connection = pscore[np.arange(T), best]

    scores = {"anchor": anchor, "circling": circ, "lean": lean, "connection": connection}
    cbs, pvs = tier_scores(scores)
    min_len = max(1, int(math.ceil(cfg.min_beats * beat / dt)))
    gap = int(round(cfg.merge_beats * beat / dt))
    cb_runs = _merge_runs(_runs(cbs, cfg.on, cfg.off), gap)
    pv_runs = _merge_runs(_runs(pvs, cfg.on, cfg.off), gap)

    def same_point(r1, r2):
        p1 = np.median(pivot[r1[0]:r1[1] + 1], 0)
        p2 = np.median(pivot[r2[0]:r2[1] + 1], 0)
        return float(np.linalg.norm(p1 - p2)) <= cfg.pivot_merge_m
    pv_runs = _merge_runs(pv_runs, int(round(cfg.pivot_merge_beats * beat / dt)), same_point)
    # nesting: a pivot lies inside one counterbalance interval (a bridged pivot gap bridges the counterbalance too)
    for a, b in pv_runs:
        hit = [r for r in cb_runs if r[1] >= a and r[0] <= b]
        cb_runs = [r for r in cb_runs if r not in hit] + [[min([a] + [r[0] for r in hit]),
                                                          max([b] + [r[1] for r in hit])]]
    # held: at least min_beats of frames AT the on level (a run of brief blips bridged by the merge is not a figure)
    held = lambda y, r: int((y[r[0]:r[1] + 1] >= cfg.on).sum()) >= min_len   # noqa: E731
    cb_runs = [r for r in _merge_runs(cb_runs, -1) if r[1] - r[0] + 1 >= min_len and held(cbs, r)]
    pv_runs = [r for r in pv_runs if r[1] - r[0] + 1 >= min_len and held(pvs, r)
               and any(c[0] <= r[0] and r[1] <= c[1] for c in cb_runs)]

    res = {"t": t, "beat_s": beat, "window_frames": 2 * h + 1, "scores": scores, "counterbalance": cbs,
           "pivot_score": pvs, "combined": pvs, "com": com, "com_combined": com_c, "joints": J, "feet": feet,
           "foot_index": fi, "pivot": pivot, "sweep_rate": sweep_rate, "radius": radius, "lean_disp": disp,
           "overhang": over,
           "pairs": pairs, "pair_score": pscore, "pair_dist": pdist, "best_pair": best, "contact": contact,
           "info": info, "cfg": cfg, "runs": [tuple(r) for r in cb_runs], "pivot_runs": [tuple(r) for r in pv_runs],
           "coverage": [[float(t[0]), float(t[-1])]],
           "beat_times": None if beat_times is None else np.asarray(beat_times, float)}
    ivs = []
    for i0, i1 in cb_runs:
        t0, t1 = level_edges(t, cbs, cfg.on, i0, i1)
        inner = [r for r in pv_runs if i0 <= r[0] and r[1] <= i1]
        pivs = []
        for n, (p0, p1) in enumerate(inner):     # refined edges never cross into a neighbouring pivot's gap half
            lo_i = (inner[n - 1][1] + p0 + 1) // 2 if n else 0
            hi_i = (p1 + inner[n + 1][0]) // 2 if n + 1 < len(inner) else T - 1
            pivs.append(pivot_details(res, p0, p1, t0, t1, lo_i, hi_i))
        ivs.append(interval_details(res, i0, i1, t0, t1, pivs))
    res["intervals"] = ivs
    return res


def _beat_ticks(res: dict, t0: float, t1: float) -> list[dict]:
    bt = res.get("beat_times")
    if bt is None:
        return []
    t, Jl = res["t"], res["joints"]["lead"]
    bt = bt[(bt >= t0) & (bt <= t1)]
    lx, lz = np.interp(bt, t, Jl[:, 0, 0]), np.interp(bt, t, Jl[:, 0, 2])
    return [{"t": round(float(a), 4), "pos": [round(float(x), 4), 0.0, round(float(z), 4)]} for a, x, z in zip(bt, lx, lz)]


def _frames_in(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    k = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    if not k.any():
        k[int(np.argmin(np.abs(t - 0.5 * (t0 + t1))))] = True
    return k


def pivot_details(res: dict, p0: int, p1: int, lo_t: float, hi_t: float, lo_i: int = 0,
                  hi_i: int | None = None) -> dict:
    """One pivot (coarse run [p0, p1]): pivot point and foot, refined edges (searched within frames [lo_i, hi_i],
    clamped to [lo_t, hi_t]), the leader's arc (world, reference s)."""
    cfg: CBConfig = res["cfg"]
    t = res["t"]
    k = slice(p0, p1 + 1)
    Jf, Jl = res["joints"]["follow"], res["joints"]["lead"]
    an = res["scores"]["anchor"][k]
    votes = [float((an * (res["foot_index"][k] == n)).sum()) for n in (0, 1)]
    foot = "left" if votes[0] >= votes[1] else "right"
    ia, ib = FEET[foot]
    a = Jf[k, ia][:, [0, 2]]
    d = Jf[k, ib][:, [0, 2]] - a
    s = pivot_s(a, d, *cfg.pivot_s_range)
    p = a + s * d
    piv = np.median(p, 0)
    dist = np.linalg.norm(p - piv, axis=1)
    Lp = Jl[:, 0][:, [0, 2]]
    phi = np.unwrap(floor_angle(Lp - piv))
    pt0, pt1, einfo = circling_edges(t, phi, p0, p1, res["beat_s"], cfg, lo_i, hi_i)
    pt0, pt1 = max(pt0, lo_t), min(pt1, hi_t)
    kk = _frames_in(t, pt0, pt1)
    ph0, ph1 = float(np.interp(pt0, t, phi)), float(np.interp(pt1, t, phi))
    swept = math.degrees(ph1 - ph0)
    rad = np.linalg.norm(Lp[kk] - piv, axis=1)
    height = float(np.median(np.minimum(Jf[kk, ia, 1], Jf[kk, ib, 1])))
    tp = np.concatenate([[pt0], t[kk][(t[kk] > pt0) & (t[kk] < pt1)], [pt1]])
    path = np.stack([np.interp(tp, t, Lp[:, 0]), np.zeros(len(tp)), np.interp(tp, t, Lp[:, 1])], 1)
    return {
        "t0": round(pt0, 4), "t1": round(pt1, 4), "coarse": [round(float(t[p0]), 4), round(float(t[p1]), 4)],
        "duration_beats": round((pt1 - pt0) / res["beat_s"], 3),
        "confidence": round(float(res["pivot_score"][kk].mean()), 4),
        "pivot": [round(float(piv[0]), 4), 0.0, round(float(piv[1]), 4)],
        "pivot_foot": foot,
        "pivot_s": round(s, 3), "pivot_drift_m": round(float(dist.max()), 4),
        "pivot_drift_p90_m": round(float(np.percentile(dist, 90)), 4), "pivot_height_m": round(height, 4),
        "leader_radius_m": round(float(np.median(rad)), 4),
        "leader_radius_range_m": [round(float(rad.min()), 4), round(float(rad.max()), 4)],
        "swept_deg": round(abs(swept), 2), "direction": "ccw" if swept > 0 else "cw",
        "swept_signed_deg": round(swept, 2),
        "arc_start_deg": round(math.degrees(ph0) % 360.0, 2),
        "scores": {c: round(float(res["scores"][c][kk].mean()), 4) for c in CRITERIA},
        "leader_path": {"t": _r(tp, 4), "pos": _r(path, 4)},
        "beat_ticks": _beat_ticks(res, pt0, pt1),
        "edges": einfo,
    }


def interval_details(res: dict, i0: int, i1: int, t0: float, t1: float, pivots: list[dict]) -> dict:
    """Contract record of one counterbalance interval (world frame, reference seconds) with its pivots."""
    cfg: CBConfig = res["cfg"]
    t = res["t"]
    kk = _frames_in(t, t0, t1)
    k = np.flatnonzero(kk)
    bp = np.bincount(res["best_pair"][k], weights=res["scores"]["connection"][k] + 1e-6,
                     minlength=len(res["pairs"])).argmax()
    name, kind = res["pairs"][bp]
    sel = res["best_pair"][k] == bp
    conn = {"kind": kind, "pair": name,
            "mean_dist_m": round(float(res["pair_dist"][k][sel, bp].mean()), 4),
            "share_of_interval": round(float(sel.mean()), 3)}
    m = (t >= t0 - cfg.com_margin_s) & (t <= t1 + cfg.com_margin_s)
    cc = res["com_combined"][m]
    axis = np.unwrap(floor_angle((res["com"]["follow"] - res["com"]["lead"])[:, [0, 2]]))
    rot = math.degrees(float(np.interp(t1, t, axis) - np.interp(t0, t, axis)))
    main = max(pivots, key=lambda p: p["t1"] - p["t0"]) if pivots else None
    g = (lambda key: main[key]) if main else (lambda key: None)
    return {
        "kind": "pivot" if pivots else "counterbalance",
        "t0": round(t0, 4), "t1": round(t1, 4), "coarse": [round(float(t[i0]), 4), round(float(t[i1]), 4)],
        "duration_beats": round((t1 - t0) / res["beat_s"], 3),
        "confidence": round(float(res["counterbalance"][kk].mean()), 4),
        # the main (longest) pivot; null for a counterbalance without a pivot
        "pivot": g("pivot"), "pivot_foot": g("pivot_foot"), "leader_radius_m": g("leader_radius_m"),
        "swept_deg": g("swept_deg"), "direction": g("direction"), "swept_signed_deg": g("swept_signed_deg"),
        "arc_start_deg": g("arc_start_deg"), "pivot_t0": g("t0"), "pivot_t1": g("t1"),
        "pivot_confidence": g("confidence"),
        "connection": conn,
        "scores": {c: round(float(res["scores"][c][kk].mean()), 4) for c in CRITERIA},
        "lean_m": {r: round(float(np.median(res["lean_disp"][r][kk])), 4) for r in ROLES},
        "overhang_m": {r: round(float(np.median(res["overhang"][r][kk])), 4) for r in ROLES},
        "hanger": max(ROLES, key=lambda r: float(np.median(res["overhang"][r][kk]))),
        "couple_rotation_deg": round(rot, 2),
        "com_track": {"t": _r(t[m], 4), "com": _r(cc, 4), "margin_s": cfg.com_margin_s},
        "leader_path": g("leader_path"), "beat_ticks": main["beat_ticks"] if main else [],
        "pivots": pivots,
    }


# ----------------------------------------------------------------------------------------------- take I/O

def work_dir(take: str) -> Path:
    return PACKAGE_ROOT / "work" / take


def _sha1(p: Path) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def code_sha1() -> str:
    """sha1 of this module's source: a changed criterion / threshold makes every counterbalance.json stale."""
    return _sha1(Path(__file__))


def find_smplx_dir(take: str, smplx_dir: str | Path | None = None) -> Path:
    """work/<take>/smplx/ (current fits), else smplx_v1/ (the backup of the v1 fits), or an explicit directory."""
    work = work_dir(take)
    cands = [Path(smplx_dir) if Path(smplx_dir).is_absolute() else work / smplx_dir] if smplx_dir else \
        [work / "smplx", work / "smplx_v1"]
    for d in cands:
        if all((d / f"{r}.npz").exists() for r in ROLES):
            return d
    raise FileNotFoundError(f"no lead/follow npz in {[str(c) for c in cands]}")


def world_frame_note(d: Path) -> str:
    """What world frame the fits are in: smplx_v1/README.json, or 'identical to smplx_v1' by content hash."""
    readme = d / "README.json"
    if readme.exists():
        return str(json.loads(readme.read_text()).get("world_frame", "see README.json"))
    v1 = d.parent / "smplx_v1"
    if d.name != "smplx_v1" and (v1 / "lead.npz").exists() and _sha1(d / "lead.npz") == _sha1(v1 / "lead.npz"):
        note = json.loads((v1 / "README.json").read_text()).get("world_frame", "") if (v1 / "README.json").exists() else ""
        return f"identical to smplx_v1 (same file content): {note}"
    if not (d.parent / "cameras").exists():
        return "world frame of the fits (no cameras/ for this take, e.g. synthetic)"
    return "current world frame of work/<take>/cameras (no README)"


def take_cfg(take: str) -> dict:
    p = PACKAGE_ROOT / "takes" / f"{take}.toml"
    return tomllib.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def grid_file(work: Path) -> Path | None:
    """The file load_grid reads the beat grid from (timing/timing.json with a beats.grid, else the beats.json files)."""
    tj = work / "timing" / "timing.json"
    if tj.exists():
        try:
            if json.loads(tj.read_text()).get("beats", {}).get("grid"):
                return tj
        except ValueError:
            pass
    for p in (work / "timing" / "beats.json", work / "beats.json", work / "beats" / "beats.json"):
        if p.exists():
            return p
    return None


def grid_sha1(work: Path) -> str | None:
    g = grid_file(work)
    return _sha1(g) if g is not None else None


def load_grid(work: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Beat grid (times, types) in reference seconds: timing/timing.json, else beats.json (zouk 8-per-measure)."""
    tj = work / "timing" / "timing.json"
    rows = None
    if tj.exists():
        try:
            rows = json.loads(tj.read_text()).get("beats", {}).get("grid")
        except ValueError:
            rows = None
    if not rows:
        for p in (work / "timing" / "beats.json", work / "beats.json", work / "beats" / "beats.json"):
            if p.exists():
                rows = json.loads(p.read_text())
                break
    if not rows:
        return None
    t = np.array([float(r[0]) for r in rows])
    ty = np.array([int(r[1]) if len(r) > 1 else 3 for r in rows])
    return t, ty


def beats_from_grid(grid) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """(beat seconds, beat times, measure-start times) from a zouk eighth grid (two entries per beat; measure start =
    a type-1 entry whose entry +3 is type 2)."""
    if grid is None or len(grid[0]) < 4:
        return None, None, None
    t, ty = grid
    beat = 2.0 * float(np.median(np.diff(t)))
    ones = np.flatnonzero(ty == 1)
    par = int(ones[0] % 2) if len(ones) else 0
    meas = np.array([t[i] for i in ones if i + 3 < len(ty) and ty[i + 3] == 2])
    return beat, t[par::2], meas


def load_npzs(d: Path) -> dict:
    out = {}
    for r in ROLES:
        with np.load(d / f"{r}.npz", allow_pickle=False) as z:
            out[r] = {k: z[k] for k in z.files}
    return out


def lead_probe(npz: dict, n: int = 5) -> dict:
    """A few (t, transl) samples of the lead fit, so an export can check the capture holds the same motion."""
    t = np.asarray(npz["t"], float)
    idx = np.unique(np.linspace(0, len(t) - 1, n).round().astype(int))
    return {"t": _r(t[idx], 5), "transl": _r(np.asarray(npz["transl"], float)[idx], 5)}


def to_json(res: dict, take: str, source: dict) -> dict:
    cfg = res["cfg"]
    sc = res["scores"]
    return _clean({
        "version": 1, "take": take, "frame": "world", "time_base": "reference",
        "coverage": [[round(a, 4), round(b, 4)] for a, b in res["coverage"]],
        "params": {**asdict(cfg), "beat_s": round(res["beat_s"], 5), "window_frames": res["window_frames"],
                   **res["info"]},
        "source": source,
        "intervals": res["intervals"],
        "track": {"t": _r(res["t"], 4), "counterbalance": _r(res["counterbalance"], 3),
                  "pivot": _r(res["pivot_score"], 3), "combined": _r(res["pivot_score"], 3),
                  **{c: _r(sc[c], 3) for c in CRITERIA},
                  "lean_follow_m": _r(res["lean_disp"]["follow"], 3), "lean_lead_m": _r(res["lean_disp"]["lead"], 3),
                  "overhang_follow_m": _r(res["overhang"]["follow"], 3),
                  "overhang_lead_m": _r(res["overhang"]["lead"], 3),
                  "pivot_foot": ["left" if f == 0 else "right" for f in res["foot_index"]]},
        "notes": [
            "World: right-handed Y-up metres, floor y=0; Unity = (x, y, -z). Times = reference seconds (npz t).",
            "coverage = the reference-time ranges that were analysed (the fits' frames); outside them nothing is "
            "known - absence of an interval there does NOT mean 'no counterbalance'.",
            "intervals = counterbalance tier (lean AND connection) = the yellow floor dot + axis to the combined COM "
            "(VIEWER_SPEC 3.9). kind 'pivot': it contains a pivot (anchored follower foot, leader circling) - pivot, "
            "pivot_foot, leader_radius_m, swept_deg, direction, arc_start_deg, leader_path, beat_ticks describe the "
            "main (longest) pivot, shown from pivot_t0 to pivot_t1 (ring + arc, VIEWER_SPEC 3.5); all pivots are in "
            "'pivots'. kind 'counterbalance': no pivot, those fields are null.",
            "Edges: t0/t1 where the per-frame counterbalance score crosses 0.5 (interpolated); pivot_t0/pivot_t1 "
            "where the leader's angular speed about the pivot crosses 10 % of its median (pivots[].edges); 'coarse' "
            "= the hysteresis frames.",
            "direction / swept_signed_deg: seen from above; ccw = positive rotation about +Y (right-handed world). "
            "arc_start_deg: angle of the leader's pelvis about the pivot at pivot_t0, from +x towards -z (ccw from "
            "above). couple_rotation_deg: rotation of the lead->follow COM line over [t0, t1], same sign convention.",
            "pivot = the follower foot's least-moving point (ankle + pivot_s * (ball - ankle)) median over the pivot, "
            "y = 0 (floor); pivot_height_m = the foot joints' median height.",
            "com_track = mass-weighted COM of both dancers per frame over [t0 - margin_s, t1 + margin_s]; the yellow "
            "floor dot is (x, 0, z) of it.",
            "confidence = mean counterbalance-tier score over [t0, t1] (pivot_confidence: pivot tier over the pivot); "
            "< 0.6 = draw dashed (VIEWER_SPEC 3.9). Tier scores are strict ANDs (minimum) of their criteria; "
            "track.combined = track.pivot (all four criteria).",
            "lean_m = median COM displacement away from the partner relative to the own mid-foot; overhang_m = median "
            "COM beyond the own support edge (> 0: would fall without the partner); hanger = the role that hangs "
            "(params.lean_mode 'counterweight': one dancer hangs, the partner does not lean in).",
            "Thresholds are documented engineering choices (module docstring), checked against one human label "
            "(larissa_kadu_38 counterbalance/truth_user.json), not trained.",
        ],
    })


def run(take: str, smplx_dir: str | Path | None = None, cfg: CBConfig | None = None, masses: dict | None = None,
        review: bool = True, out_path: Path | None = None) -> dict:
    """Detect on a take and write work/<take>/counterbalance/counterbalance.json (+ the review sheet)."""
    cfg = cfg or CBConfig()
    work = work_dir(take)
    d = find_smplx_dir(take, smplx_dir)
    npzs = load_npzs(d)
    tc = take_cfg(take)
    dancers = tc.get("dancers", {})
    masses = dict(masses or {})
    for r in ROLES:
        if r not in masses and dancers.get(r, {}).get("mass_kg"):
            masses[r] = float(dancers[r]["mass_kg"])
    statures = {r: float(dancers[r]["stature_m"]) for r in ROLES if dancers.get(r, {}).get("stature_m")}
    grid = load_grid(work)
    beat, beat_t, _ = beats_from_grid(grid)
    res = detect(npzs, cfg, beat, masses, statures, beat_t)
    source = {"smplx_dir": str(d.relative_to(PACKAGE_ROOT)) if d.is_relative_to(PACKAGE_ROOT) else str(d),
              "world_frame": world_frame_note(d), "lead_sha1": _sha1(d / "lead.npz"),
              "lead_transl0": _r(np.asarray(npzs["lead"]["transl"], float)[0], 5),
              "lead_probe": lead_probe(npzs["lead"]),
              "beats": "take grid" if beat else f"none (default {cfg.default_bpm} BPM)",
              "beats_file": None if grid_file(work) is None else str(grid_file(work).relative_to(PACKAGE_ROOT)),
              "beats_sha1": grid_sha1(work),
              "frames": len(res["t"]), "t_range": _r([res["t"][0], res["t"][-1]], 4),
              "code_sha1": code_sha1()}
    doc = to_json(res, take, source)
    out_path = out_path or work / "counterbalance" / "counterbalance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1))
    res["_json"], res["_path"] = doc, out_path
    truth = load_truth(take)
    if truth is not None:
        res["_eval"] = evaluate(doc, truth)
    if review:
        res["_review"] = plot_review(res, doc, work / "review" / "counterbalance.jpg", take, grid, truth)
    return res


def is_current(take: str, smplx_dir: str | Path | None = None) -> bool:
    """True when work/<take>/counterbalance/counterbalance.json was detected on the current fits (lead sha1), beat
    grid (beats_sha1: e.g. timing.json's studio beats replacing beats.json) and detector (code_sha1: this module's
    source, so a changed criterion or threshold re-detects)."""
    p = work_dir(take) / "counterbalance" / "counterbalance.json"
    if not p.exists():
        return False
    d = find_smplx_dir(take, smplx_dir)
    src = json.loads(p.read_text()).get("source", {})
    return (src.get("lead_sha1") == _sha1(d / "lead.npz") and src.get("beats_sha1") == grid_sha1(work_dir(take))
            and src.get("code_sha1") == code_sha1())


def load_truth(take: str) -> dict | None:
    """Ground truth of a take: work/<take>/truth.json (synthetic generators) or the human label
    work/<take>/counterbalance/truth_user.json (reference seconds), when it holds a counterbalance list."""
    work = work_dir(take)
    for p in (work / "truth.json", work / "counterbalance" / "truth_user.json"):
        if p.exists():
            tr = json.loads(p.read_text())
            if "counterbalance" in tr:
                return tr
    return None


# ----------------------------------------------------------------------------------------------- truth scoring

def _iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def evaluate(doc: dict, truth: dict) -> dict:
    """Detected intervals vs truth.json's counterbalance list (one truth interval expected). Edge errors in frames
    (detected - truth): t0/t1 = the yellow axis (counterbalance tier), pivot_t0/pivot_t1 = the pivot."""
    tr = truth.get("counterbalance", [])
    out = {"n_detected": len(doc["intervals"]), "n_truth": len(tr)}
    if not tr:
        return out
    g = tr[0]
    fps = float(truth.get("fps") or 30.0)
    if not doc["intervals"]:
        out.update({"iou": 0.0})
        return out
    best = max(doc["intervals"], key=lambda iv: max(0.0, min(iv["t1"], g["t1"]) - max(iv["t0"], g["t0"])))
    out["iou"] = _iou(best["t0"], best["t1"], g["t0"], g["t1"])
    out["interval"] = [best["t0"], best["t1"]]
    out["kind"] = best.get("kind")
    out["edge_err_frames"] = {"t0": (best["t0"] - g["t0"]) * fps, "t1": (best["t1"] - g["t1"]) * fps}
    if g.get("pivot_t0") is None:
        return out
    if best.get("pivot_t0") is None:
        out["pivot_iou"] = 0.0
        return out
    out["pivot_interval"] = [best["pivot_t0"], best["pivot_t1"]]
    out["n_pivots"] = len(best.get("pivots", []))
    out["pivot_iou"] = _iou(best["pivot_t0"], best["pivot_t1"], g["pivot_t0"], g["pivot_t1"])
    out["pivot_edge_err_frames"] = {"t0": (best["pivot_t0"] - g["pivot_t0"]) * fps,
                                    "t1": (best["pivot_t1"] - g["pivot_t1"]) * fps}
    out["pivot_err_m"] = float(np.linalg.norm(np.array(best["pivot"])[[0, 2]] - np.array(g["pivot"])[[0, 2]]))
    out["swept_deg"] = best["swept_deg"]
    out["swept_err_deg"] = abs(best["swept_deg"] - abs(g["swept_deg"]))
    out["direction_ok"] = best["direction"] == g["direction"]
    out["pivot_foot_ok"] = best["pivot_foot"] == g["pivot_foot"]
    out["leader_radius_err_m"] = abs(best["leader_radius_m"] - g["leader_radius_m"])
    ct = np.array(best["com_track"]["t"])
    cc = np.array(best["com_track"]["com"])
    tt = np.array(truth["com"]["t"])
    tc = np.array(truth["com"]["combined"])
    k = (ct >= max(best["t0"], g["t0"])) & (ct <= min(best["t1"], g["t1"]))
    if k.any():
        ref = np.stack([np.interp(ct[k], tt, tc[:, j]) for j in range(3)], 1)
        e = np.linalg.norm(cc[k][:, [0, 2]] - ref[:, [0, 2]], axis=1)
        out["com_floor_err_m"] = {"median": float(np.median(e)), "max": float(e.max())}
        out["com_height_err_m"] = float(np.abs(cc[k][:, 1] - ref[:, 1]).max())
    return out


# ----------------------------------------------------------------------------------------------- review

LEAD_C, FOLLOW_C, CB_C, INK, MUTED = "#e34948", "#2a78d6", "#eda100", "#0b0b0b", "#8a8984"


def plot_review(res: dict, doc: dict, path: Path, take: str, grid=None, truth: dict | None = None) -> Path:
    """Top view (paths, pivot rings, leader arcs, combined-COM floor dots) + the tier and criterion scores on the beat
    grid (light yellow = counterbalance / yellow axis, dark yellow = pivot; dotted / dashed = truth)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    t = res["t"]
    J = res["joints"]
    fig = plt.figure(figsize=(17, 12.5), dpi=100)
    gs = fig.add_gridspec(8, 2, width_ratios=[1.0, 1.25], height_ratios=[1] * 6 + [0.45, 1.6], hspace=0.12,
                          wspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ivs = doc["intervals"]
    pivs = [p for iv in ivs for p in iv.get("pivots", [])]
    inside = np.zeros(len(t), bool)
    for iv in ivs:
        inside |= (t >= iv["t0"]) & (t <= iv["t1"])
    # top view: x right, -z up (= the view from above; also Unity x right / z up)
    for role, col in (("lead", LEAD_C), ("follow", FOLLOW_C)):
        P = J[role]
        for f, (ia, ib) in FEET.items():
            m = 0.5 * (P[:, ia] + P[:, ib])
            ax.plot(m[:, 0], -m[:, 2], color=col, lw=0.7, alpha=0.45, ls="--" if f == "right" else "-")
        ax.plot(P[:, 0, 0], -P[:, 0, 2], color=col, lw=2.0, label=f"{role} pelvis")
        ax.plot(P[0, 0, 0], -P[0, 0, 2], "o", color=col, ms=8, mec="white", mew=1.5)
        ax.plot(P[-1, 0, 0], -P[-1, 0, 2], "s", color=col, ms=8, mec="white", mew=1.5)
    cc = res["com_combined"]
    ax.plot(cc[:, 0], -cc[:, 2], color=MUTED, lw=1.0, label="combined COM (floor)")
    if inside.any():
        ax.scatter(cc[inside, 0], -cc[inside, 2], s=14, color=CB_C, edgecolor=INK, linewidth=0.4, zorder=5,
                   label="yellow floor dot (in counterbalance)")
    for pv in pivs:
        px, pz = pv["pivot"][0], pv["pivot"][2]
        ax.add_patch(Circle((px, -pz), 0.08, fill=False, color=CB_C, lw=3, zorder=6))
        ax.add_patch(Circle((px, -pz), 0.08, fill=False, color=INK, lw=0.6, zorder=7))
        r = 1.08 * pv["leader_radius_m"]          # just outside the leader's path so the path stays visible
        a0 = math.radians(pv["arc_start_deg"])
        sw = math.radians(pv["swept_signed_deg"])
        aa = a0 + np.linspace(0, sw, 200)
        rr = r * (1 + 0.04 * np.linspace(0, 1, 200) * (abs(sw) > 2 * math.pi))   # spiral out past one turn
        ax.plot(px + rr * np.cos(aa), -pz + rr * np.sin(aa), color=CB_C, lw=3.5, alpha=0.9, zorder=4)
        ax.annotate("", xy=(px + rr[-1] * np.cos(aa[-1]), -pz + rr[-1] * np.sin(aa[-1])),
                    xytext=(px + rr[-8] * np.cos(aa[-8]), -pz + rr[-8] * np.sin(aa[-8])),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2), zorder=8)
        n = pivs.index(pv)
        ax.text(0.98, 0.02 + 0.08 * n, f"pivot {n + 1}: {pv['swept_deg']:.0f}° {pv['direction']}, leader r "
                f"{pv['leader_radius_m']:.2f} m\n{pv['t0']:.2f}-{pv['t1']:.2f} s, conf {pv['confidence']:.2f}",
                transform=ax.transAxes, fontsize=9, color=INK, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=CB_C, lw=1))
        for tk in pv.get("beat_ticks", []):
            ax.plot(tk["pos"][0], -tk["pos"][2], "|", color=INK, ms=10, mew=1.5, zorder=8)
    if truth:
        for g in truth.get("counterbalance", []):
            if g.get("pivot"):
                ax.plot(g["pivot"][0], -g["pivot"][2], "x", color=INK, ms=10, mew=2, zorder=9, label="truth pivot")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("-z (m)   [view from above; Unity z]")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("top view: pelvis (thick), mid-foot (thin; right foot dashed), o start / square end", fontsize=10)

    beat, beat_t, meas_t = beats_from_grid(grid)
    rows = [("counterbalance\n(yellow axis)", res["counterbalance"]), ("pivot\n(ring + arc)", res["pivot_score"])] \
        + [(c, res["scores"][c]) for c in ("lean", "connection", "anchor", "circling")]
    axes = []
    for k, (name, y) in enumerate(rows):
        a = fig.add_subplot(gs[k, 1], sharex=axes[0] if axes else None)
        axes.append(a)
        if beat_t is not None:
            for bt in beat_t[(beat_t >= t[0]) & (beat_t <= t[-1])]:
                a.axvline(bt, color="#ecebe6", lw=0.8, zorder=0)
        if meas_t is not None:
            for mt in meas_t[(meas_t >= t[0]) & (meas_t <= t[-1])]:
                a.axvline(mt, color="#b9b8b2", lw=1.0, zorder=0)
        for iv in ivs:
            a.axvspan(iv["t0"], iv["t1"], color=CB_C, alpha=0.22, lw=0, zorder=1)
        for pv in pivs:
            a.axvspan(pv["t0"], pv["t1"], color=CB_C, alpha=0.35, lw=0, zorder=1)
        if truth:
            for g in truth.get("counterbalance", []):
                for key, ls in (("t0", ":"), ("t1", ":"), ("pivot_t0", "--"), ("pivot_t1", "--")):
                    if g.get(key) is not None:
                        a.axvline(g[key], color=INK, lw=1.0, ls=ls, zorder=2)
        a.plot(t, y, color=INK if k < 2 else "#2a78d6", lw=1.6, zorder=3)
        if k < 2:
            a.axhline(res["cfg"].on, color=INK, lw=0.8, ls="--")
            a.axhline(res["cfg"].off, color=MUTED, lw=0.8, ls="--")
        a.set_ylim(-0.05, 1.05)
        a.set_yticks([0, 0.5, 1])
        a.set_ylabel(name, fontsize=9)
        a.tick_params(labelsize=8)
        if k < len(rows) - 1:
            a.tick_params(labelbottom=False)
    axes[-1].set_xlabel("reference time (s)   grid: light = beats, dark = measure starts; light yellow = counterbalance, "
                        "dark = pivot" + ("; truth: dotted = counterbalance, dashed = pivot" if truth else ""))
    # zoom on the edges of the longest interval: tier scores per frame, detected vs truth edges (VIEWER_SPEC 14)
    if ivs:
        iv = max(ivs, key=lambda v: v["t1"] - v["t0"])
        g = None
        if truth and truth.get("counterbalance"):
            g = min(truth["counterbalance"], key=lambda x: abs((x.get("t0") or 0.0) - iv["t0"]))
        zs = gs[7, 1].subgridspec(1, 2, wspace=0.15)
        dt = float(np.median(np.diff(t)))
        for z, (key, pkey) in enumerate((("t0", "pivot_t0"), ("t1", "pivot_t1"))):
            a = fig.add_subplot(zs[0, z])
            c = iv[key] if iv.get(pkey) is None else 0.5 * (iv[key] + iv[pkey])
            m = (t >= c - 0.45) & (t <= c + 0.45)
            a.plot(t[m], res["counterbalance"][m], "o-", color=INK, ms=3, lw=1.2, label="counterbalance score")
            a.plot(t[m], res["pivot_score"][m], "s-", color=CB_C, ms=3, lw=1.2, label="pivot score")
            a.axvline(iv[key], color=INK, lw=2.0, label=f"axis {key} {iv[key]:.3f}")
            if iv.get(pkey) is not None:
                a.axvline(iv[pkey], color=CB_C, lw=2.0, label=f"pivot {key} {iv[pkey]:.3f}")
            if g is not None and g.get(key) is not None:
                a.axvline(g[key], color=INK, lw=1.2, ls=":", label=f"truth: axis {(iv[key] - g[key]) / dt:+.2f} fr")
            if g is not None and g.get(pkey) is not None and iv.get(pkey) is not None:
                a.axvline(g[pkey], color="#a86b00", lw=1.2, ls="--",
                          label=f"truth: pivot {(iv[pkey] - g[pkey]) / dt:+.2f} fr")
            a.set_ylim(-0.05, 1.05)
            a.tick_params(labelsize=8)
            a.legend(fontsize=7, loc="center right" if z == 0 else "center left", framealpha=0.9)
            a.set_title(f"{'start' if z == 0 else 'end'} of the {iv['kind']} interval (dots = frames)", fontsize=9)
            a.set_xlabel("reference time (s)", fontsize=8)
    src = doc.get("source", {})
    kinds = ", ".join(f"{iv['kind']} {iv['t0']:.2f}-{iv['t1']:.2f}" for iv in ivs) or "none"
    head = (f"{take}: counterbalance detection - {len(ivs)} interval(s) [{kinds}]  coverage "
            f"{', '.join(f'{a:.2f}-{b:.2f}' for a, b in doc.get('coverage', []))} s\n"
            f"[{src.get('smplx_dir', '')}: {src.get('world_frame', '')[:120]}]")
    fig.suptitle(head, fontsize=10.5, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)
    return path


def plot_cams(take: str, res: dict, doc: dict, path: Path, cams: list[str] | None = None, instants: int = 3) -> Path | None:
    """Reproject the yellow COM axis (floor dot -> combined COM), the pivot ring and both skeletons into camera frames.
    Fits in the OLD world frame (smplx_v1) use work/<take>/cameras_v1 (the cameras of that frame)."""
    import cv2
    from .review import SMPLX_EDGES, _frames, _label, _nearest
    work = work_dir(take)
    old = "OLD world frame" in doc.get("source", {}).get("world_frame", "")
    cam_root = work / "cameras_v1" if old and (work / "cameras_v1").exists() else work / "cameras"
    wfs = cam_root / "world_from_sfm.json"
    if not wfs.exists() or not (work / "frames").exists():
        return None
    w = json.loads(wfs.read_text())
    s, Rw, tw = float(w["s"]), np.array(w["R"]), np.array(w["t"])
    avail = sorted(p.stem for p in cam_root.glob("*.npz") if (work / "frames" / p.stem / "frames.json").exists())
    if not cams:
        med = {}
        for c in avail:
            z = np.load(cam_root / f"{c}.npz")
            ok = z["ok"] if "ok" in z.files else np.ones(len(z["t"]), bool)
            med[c] = float(np.median(z["reproj_px"][ok])) if "reproj_px" in z.files and ok.any() else 99.0
        cams = sorted(avail, key=lambda c: med[c])[:2]
    t = res["t"]
    ivs = doc["intervals"]
    if ivs:
        iv = max(ivs, key=lambda v: v["t1"] - v["t0"])
        times = np.linspace(iv["t0"], iv["t1"], instants + 2)[1:-1]
        note = f"{iv['kind']} {iv['t0']:.2f}-{iv['t1']:.2f} s"
    else:
        times = np.linspace(t[0] + 0.5, t[-1] - 0.5, instants)
        note = "NO counterbalance detected - axis shown at every instant only to check the COM placement"
    cc = res["com_combined"]

    def proj(Xw, K, R, tv):
        Xs = ((np.atleast_2d(Xw) - tw) @ Rw) / s
        Xc = Xs @ R.T + tv
        uv = Xc @ K.T
        return uv[:, :2] / uv[:, 2:3], Xc[:, 2]

    rows = []
    for tt in times:
        tiles = []
        i_res = int(np.argmin(np.abs(t - tt)))
        for c in cams:
            meta = _frames(work, c)
            i, dt_ = _nearest(meta, float(tt))
            img = cv2.imread(str(work / "frames" / c / meta["frames"][i]["file"]))
            z = np.load(cam_root / f"{c}.npz")
            K, R, tv = z["K"][i], z["R"][i], z["tvec"][i]
            for role, col in (("lead", (60, 60, 235)), ("follow", (235, 140, 40))):
                uv, _ = proj(res["joints"][role][i_res][:22], K, R, tv)
                for a_, b_ in SMPLX_EDGES:
                    cv2.line(img, tuple(int(v) for v in uv[a_]), tuple(int(v) for v in uv[b_]), col, 3, cv2.LINE_AA)
            com = cc[i_res]
            foot = np.array([com[0], 0.0, com[2]])
            uv, _ = proj(np.stack([foot, com]), K, R, tv)
            yel = (0, 200, 255)
            cv2.line(img, tuple(int(v) for v in uv[0]), tuple(int(v) for v in uv[1]), (0, 0, 0), 11, cv2.LINE_AA)
            cv2.line(img, tuple(int(v) for v in uv[0]), tuple(int(v) for v in uv[1]), yel, 7, cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in uv[0]), 14, yel, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in uv[1]), 12, yel, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in uv[1]), 12, (0, 0, 0), 2, cv2.LINE_AA)
            for v in ivs:
                for pv in v.get("pivots", []):
                    if pv["t0"] <= tt <= pv["t1"]:
                        ang = np.linspace(0, 2 * np.pi, 48)
                        ring = np.stack([pv["pivot"][0] + 0.1 * np.cos(ang), np.zeros(48),
                                         pv["pivot"][2] + 0.1 * np.sin(ang)], 1)
                        ru, _ = proj(ring, K, R, tv)
                        cv2.polylines(img, [ru.astype(np.int32)], True, yel, 5, cv2.LINE_AA)
            scale = 720 / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), 720), interpolation=cv2.INTER_AREA)
            _label(img, f"cam {c}  t={tt:.2f}s ({dt_ * 1000:+.0f} ms)", (8, 26), (255, 255, 255), 0.6)
            tiles.append(img)
        rows.append(np.concatenate(tiles, 1))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    sheet = np.concatenate(rows, 0)
    import textwrap
    lines = [(f"{take}: yellow = floor dot + vertical axis to the couple's combined COM", (255, 255, 255)),
             ("red = lead, blue = follow SMPL-X joints; yellow ring = pivot (in a pivot)", (255, 255, 255)),
             (f"fits {doc.get('source', {}).get('smplx_dir', '')}, cameras {cam_root.name}/ "
              f"({'OLD' if old else 'current'} world frame)", (200, 200, 200))]
    lines += [(ln, (120, 220, 255)) for ln in textwrap.wrap(note, 62)]
    head = np.full((14 + 26 * len(lines), sheet.shape[1], 3), 12, np.uint8)
    for n, (txt, col) in enumerate(lines):
        _label(head, txt, (10, 28 + 26 * n), col, 0.55)
    sheet = np.concatenate([head, sheet], 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


# ----------------------------------------------------------------------------------------------- Unity export

def capture_clock(cap_dir: Path):
    """The capture's clock: capture time = reference time + time_to_audio (capture.json; v2 captures: audio_offset -
    times[0]) - the viewer's CaptureTimeline.AudioTimes, shared with the moves Unity copies (moves.labels.CaptureClock)."""
    from .moves.labels import CaptureClock
    if not (Path(cap_dir) / "capture.json").exists():
        raise FileNotFoundError(f"{cap_dir}/capture.json missing (export the capture first)")
    return CaptureClock.from_capture(cap_dir)


def check_capture_motion(cap_dir: Path, doc: dict, tol_m: float = 0.01) -> str:
    """The capture must hold the same lead motion the detection ran on: the fit's lead transl at the probe times
    (source.lead_probe; old files: lead_transl0 at the first frame) vs the capture frame at the same reference time."""
    p = cap_dir / "smplx_lead.bin"
    src = doc.get("source", {})
    probe = src.get("lead_probe") or ({"t": [src["t_range"][0]], "transl": [src["lead_transl0"]]}
                                      if src.get("lead_transl0") is not None and src.get("t_range") else None)
    tj = cap_dir / "times.json"
    if not p.exists() or probe is None or not tj.exists():
        return "not checked (no smplx_lead.bin / times.json / lead probe)"
    from .export_unity import read_motion, to_unity_points
    _, frames = read_motion(p)
    times = np.asarray(json.loads(tj.read_text()), float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30
    errs = []
    for tp, xp in zip(probe["t"], probe["transl"]):
        i = int(np.argmin(np.abs(times - tp)))
        if abs(times[i] - tp) <= 0.5 * dt + 1e-6:
            errs.append(float(np.linalg.norm(frames[i, :3] - to_unity_points(np.asarray(xp, float)))))
    if not errs:
        return "not checked (no probe time inside the capture)"
    err = max(errs)
    if err > tol_m:
        raise ValueError(f"{cap_dir.name}: capture lead transl differs by {err:.3f} m from the fits the detection ran "
                         f"on ({src.get('smplx_dir')}) - re-run detect on the capture's smplx (--smplx-dir)")
    return f"ok ({err * 1000:.1f} mm over {len(errs)} probe frame(s))"


def capture_matches(cap_dir: Path, take: str, tol_m: float = 1e-3) -> bool:
    """True when the capture already holds the take's current fits: same frame times and the lead transl of every
    frame within tol_m (so a re-export, which wipes the capture folder, is not needed)."""
    from .export_unity import read_motion, to_unity_points
    try:
        times = np.asarray(json.loads((cap_dir / "times.json").read_text()), float)
        _, frames = read_motion(cap_dir / "smplx_lead.bin")
        npz = load_npzs(find_smplx_dir(take))["lead"]
    except (FileNotFoundError, OSError, ValueError, KeyError, AssertionError):
        return False
    t = np.asarray(npz["t"], float)
    if len(times) != len(t) or len(frames) != len(t) or np.abs(times - t).max() > 1e-4:
        return False
    return float(np.abs(frames[:, :3] - to_unity_points(np.asarray(npz["transl"], float))).max()) <= tol_m


def _overlap_s(ranges_a: list, a: float, b: float) -> float:
    return sum(max(0.0, min(r1, b) - max(r0, a)) for r0, r1 in ranges_a)


def unity_doc(doc: dict, time_to_audio: float, capture: str, motion_check: str) -> dict:
    """counterbalance.json in the capture's time base (capture = reference + time_to_audio) and Unity coordinates
    (x, y, -z). Every converted time keeps its reference value in <key>_reference."""
    from .export_unity import to_unity_points
    tta = float(time_to_audio)
    U = lambda p: None if p is None else _r(to_unity_points(np.asarray(p, float)), 4)  # noqa: E731
    T = lambda x: None if x is None else _r(np.asarray(x, float) + tta, 4)             # noqa: E731

    def times(d: dict, keys=("t0", "t1", "pivot_t0", "pivot_t1")) -> None:
        for key in keys:
            if key in d and d[key] is not None:
                d[f"{key}_reference"] = d[key]
                d[key] = round(d[key] + tta, 4)
        if d.get("coarse") is not None:
            d["coarse"] = T(d["coarse"])

    def arc(d: dict) -> None:
        if d.get("pivot") is not None:
            d["pivot"] = U(d["pivot"])
        if d.get("leader_path"):
            d["leader_path"] = {"t": T(d["leader_path"]["t"]), "pos": U(d["leader_path"]["pos"])}
        for tk in d.get("beat_ticks") or []:
            tk["t"], tk["pos"] = round(tk["t"] + tta, 4), U(tk["pos"])
        # the mirror keeps the picture seen from above; the signed angle about Unity's +Y (left-handed) flips sign
        sw = d.get("swept_signed_deg")
        d["swept_signed_deg_unity_up"] = None if sw is None else -sw
        a = d.get("arc_start_deg")
        d["arc_start_deg_unity"] = None if a is None else round(math.degrees(math.atan2(
            math.sin(math.radians(a)), math.cos(math.radians(a)))), 2)

    out = json.loads(json.dumps(doc))
    out.update({"time_base": "capture", "frame": "unity", "capture": capture, "time_to_audio": tta,
                "time_offset_s": tta, "motion_check": motion_check,
                "coverage_reference": doc.get("coverage"), "coverage": [T(c) for c in doc.get("coverage", [])]})
    for iv in out["intervals"]:
        times(iv)
        arc(iv)
        iv["com_track"]["t"] = T(iv["com_track"]["t"])
        iv["com_track"]["com"] = U(iv["com_track"]["com"])
        for pv in iv.get("pivots", []):
            times(pv, ("t0", "t1"))
            arc(pv)
    out["track"]["t"] = T(out["track"]["t"])
    out["notes"] = out["notes"] + [
        f"UNITY COPY of work/{doc['take']}/counterbalance/counterbalance.json: positions (x, y, -z) in the capture's "
        "Unity frame BEFORE the viewer's dance-origin offset (capture.json 'origin', VIEWER_SPEC 3.0) - apply it like "
        "every other capture position; times = capture time = reference + time_to_audio (capture.json; the clock of "
        "CaptureTimeline.AudioTimes, as the moves/*.json copies); *_reference keep the reference seconds.",
        "direction 'ccw'/'cw' is as seen from above and unchanged by the mirror (Unity top view: x right, z up). In "
        "Unity's left-handed frame a ccw sweep is a NEGATIVE angle about +Y: swept_signed_deg_unity_up. "
        "arc_start_deg_unity: angle in the Unity floor plane from +x towards +z (= the world angle from +x towards -z).",
    ]
    return out


def export(take: str, capture: str | None = None, streaming: Path = UNITY_STREAMING, force: bool = False,
           manifest: bool = True, src: Path | None = None, mirror: bool = True) -> dict:
    """Write StreamingAssets/<capture>/counterbalance.json (+ work/<take>/export/<capture>/ when present and mirror).
    Refuses (ValueError) when the analysed range does not overlap the capture or the capture's lead motion differs
    from the fits the detection ran on (force: write anyway, motion_check 'MISMATCH (forced)')."""
    capture = capture or DEFAULT_CAPTURES.get(take, take)
    src = Path(src) if src else work_dir(take) / "counterbalance" / "counterbalance.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: run `python -m dancecap.counterbalance detect --take {take}`")
    doc = json.loads(src.read_text())
    cov = doc.get("coverage") or [doc.get("source", {}).get("t_range") or [doc["track"]["t"][0], doc["track"]["t"][-1]]]
    report = {"capture": capture, "written": []}
    dirs = [Path(streaming) / capture] + ([work_dir(take) / "export" / capture] if mirror else [])
    for cap_dir in dirs:
        if not cap_dir.exists():
            continue
        clock = capture_clock(cap_dir)
        rr = clock.range_reference
        if rr is not None and _overlap_s(cov, rr[0], rr[1]) <= 0:
            raise ValueError(f"{cap_dir.name}: the analysed range {cov} (reference s) does not overlap the capture's "
                             f"frames {list(rr)} - wrong take or window")
        try:
            mc = check_capture_motion(cap_dir, doc)
        except ValueError:
            if not force:
                raise
            mc = "MISMATCH (forced)"
        out = unity_doc(doc, clock.time_to_audio, capture, mc)
        out["time_offset_source"] = "capture.json time_to_audio (moves.labels.CaptureClock = CaptureTimeline.AudioTimes)"
        out["capture_range_reference"] = None if rr is None else [round(rr[0], 4), round(rr[1], 4)]
        out["capture_range"] = None if rr is None else [round(clock.to_capture(rr[0]), 4),
                                                         round(clock.to_capture(rr[1]), 4)]
        (cap_dir / "counterbalance.json").write_text(json.dumps(out, indent=1))
        report["written"].append(str(cap_dir / "counterbalance.json"))
        report["motion_check"] = mc
        report["time_to_audio"] = clock.time_to_audio
        cj = cap_dir / "capture.json"
        if manifest and cj.exists():   # additive: tell the viewer the layer exists (capture.json v4 key)
            m = json.loads(cj.read_text())
            if m.get("counterbalance") != "counterbalance.json":
                m["counterbalance"] = "counterbalance.json"
                cj.write_text(json.dumps(m, indent=2))
    if not report["written"]:
        raise FileNotFoundError(f"capture {capture} not found in {streaming} or work/{take}/export/")
    return report


# ----------------------------------------------------------------------------------------------- CLI

def _summary(doc: dict) -> str:
    cov = ", ".join(f"{a:.2f}-{b:.2f}" for a, b in doc.get("coverage", []))
    lines = [f"{doc['take']}: {len(doc['intervals'])} counterbalance interval(s)  [{doc['source']['smplx_dir']}; "
             f"analysed {cov} s]"]
    for w in doc.get("params", {}).get("warnings", []):
        lines.append(f"  WARNING {w}")
    for iv in doc["intervals"]:
        lines.append(f"  {iv['kind']:>14} {iv['t0']:.3f}-{iv['t1']:.3f} s ({iv['duration_beats']:.1f} beats) "
                     f"conf {iv['confidence']:.2f} lean {iv['lean_m']} {iv['connection']['pair']} "
                     f"scores {iv['scores']}")
        for pv in iv.get("pivots", []):
            lines.append(f"           pivot {pv['t0']:.3f}-{pv['t1']:.3f} s conf {pv['confidence']:.2f}  "
                         f"{pv['pivot_foot']} foot ({pv['pivot'][0]:.3f}, {pv['pivot'][2]:.3f})  leader r "
                         f"{pv['leader_radius_m']:.2f} m swept {pv['swept_deg']:.0f} deg {pv['direction']}")
    tr = doc["track"]
    for c in ("counterbalance", "pivot") + CRITERIA:
        y = np.array([v if v is not None else np.nan for v in tr[c]], float)
        lines.append(f"  {c:>14}: median {np.nanmedian(y):.2f}  p90 {np.nanpercentile(y, 90):.2f}  max "
                     f"{np.nanmax(y):.2f}  frames>=0.5 {np.mean(y >= 0.5):.0%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.counterbalance", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("detect", help="detect intervals -> work/<take>/counterbalance/counterbalance.json")
    p.add_argument("--take", required=True)
    p.add_argument("--smplx-dir", default=None, help="default: work/<take>/smplx, else smplx_v1")
    p.add_argument("--mass-lead", type=float, default=None)
    p.add_argument("--mass-follow", type=float, default=None)
    p.add_argument("--no-review", action="store_true")
    p.add_argument("--cams", default=None, help="comma list for the camera reprojection sheet (default: best 2)")
    p = sub.add_parser("synthetic", help="write work/synthetic_cb (+ truth), capture 03_SyntheticCounterbalance, detect")
    p.add_argument("--no-unity", action="store_true")
    p.add_argument("--reexport", action="store_true",
                   help="re-export the capture even when it already holds the same motion (wipes its other layers)")
    p = sub.add_parser("export", help="Unity copy -> StreamingAssets/<capture>/counterbalance.json")
    p.add_argument("--take", required=True)
    p.add_argument("--capture", default=None)
    p.add_argument("--force", action="store_true", help="write even if the capture motion differs from the fits")
    p = sub.add_parser("review", help="re-run detection (rewrites counterbalance.json, deterministic) + review sheets")
    p.add_argument("--take", required=True)
    p.add_argument("--smplx-dir", default=None)
    p.add_argument("--cams", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "synthetic":
        from . import synthetic_counterbalance as sc
        out = sc.make()
        print(f"[counterbalance] synthetic take -> {out}")
        res = run(sc.TAKE)
        print(_summary(res["_json"]))
        ev = evaluate(res["_json"], json.loads((out / "truth.json").read_text()))
        print("[counterbalance] vs truth:", json.dumps(ev, default=float))
        if not a.no_unity:
            if a.reexport or not capture_matches(UNITY_STREAMING / sc.CAPTURE, sc.TAKE):
                rep = sc.export_capture()      # export_unity rewrites the whole capture folder
                print(f"[counterbalance] capture -> {rep.get('installed', rep['out'])}  (re-exported: re-run "
                      f"`python -m dancecap.moves export {sc.TAKE}` for its moves/ copies)")
            else:
                print(f"[counterbalance] capture {sc.CAPTURE} already holds this motion - not re-exported "
                      "(keeps the other layers, e.g. moves/); --reexport forces it")
            print("[counterbalance] unity:", export(sc.TAKE, sc.CAPTURE))
        return
    if a.cmd == "export":
        print(json.dumps(export(a.take, a.capture, force=a.force), indent=1))
        return
    masses = {}
    if a.cmd == "detect":
        if a.mass_lead:
            masses["lead"] = a.mass_lead
        if a.mass_follow:
            masses["follow"] = a.mass_follow
    res = run(a.take, a.smplx_dir, masses=masses, review=not getattr(a, "no_review", False))
    print(_summary(res["_json"]))
    if res.get("_eval"):
        print("[counterbalance] vs truth:", json.dumps(res["_eval"], default=float))
    print(f"[counterbalance] -> {res['_path']}")
    if res.get("_review"):
        print(f"[counterbalance] review -> {res['_review']}")
    if not getattr(a, "no_review", False):
        cams = a.cams.split(",") if a.cams else None
        p = plot_cams(a.take, res, res["_json"], work_dir(a.take) / "review" / "counterbalance_cams.jpg", cams)
        if p:
            print(f"[counterbalance] camera sheet -> {p}")


if __name__ == "__main__":
    main()
