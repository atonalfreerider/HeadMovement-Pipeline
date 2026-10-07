"""Interpretable per-frame motion features of the couple -> work/<take>/moves/features.npz + features_schema.json.

    python -m dancecap.moves features <take> [--smplx-dir smplx_v1] [--com physics|joints] [--no-review]

Stage A of the move-classification plan (docs/MOVES_RESEARCH.md section 8, decisions F1/F2/F7): CPU only, no
training. These descriptors feed the unlabelled boundary proposals (proposals.py) now and the training windows
(dataset.py) once the user has narrated / labelled moves. Nothing here says WHICH move is danced.

Inputs (F1)
  work/<take>/smplx/{lead,follow}.npz (the current fits), else smplx_v1/ (a warning is recorded: the v1 world frame
  is ~4.8 deg off gravity, ~0.08 m at hip height, which biases the lean features); the beat grid (timing/timing.json
  grid, else beats.json: 8 zouk eighths per measure, type 1 at eighths 0 and 4); optional
  work/<take>/counterbalance/counterbalance.json (criterion scores); optional work/<take>/timing/timing.json
  footsteps (touchdowns; else foot-contact onsets give the step rate); optional dancer masses in takes/<take>.toml.

Body model (--com physics, the default): physics.analyze_dancer per dancer (shaped SMPL-X, voxelised rigid mass
elements: COM, COM velocity, sole contacts from the posed soles) + forward kinematics for the joint rotations
(head / chest axes, element angular velocity for the rotational kinetic energy). --com joints: joint positions only
(de Leva-style segment weights, joint-height foot contacts) - fast, used by the unit tests and when the SMPL-X model
files are missing.

Couple frame (head-movement VIEWER_SPEC 3.6, the leader floor axis): origin = the leader's chest (spine3, joint 9)
projected to the floor; LONG axis = the leader's shoulder line (left shoulder 16 minus right shoulder 17) projected to
the floor, pointing to the leader's LEFT; SHORT axis = forward from the leader's chest = long x up (right-handed,
Y-up: forward_xz = (-long_z, long_x)). Angles about the vertical are + counter-clockwise seen from above.

Every feature (name, unit, group, description, scale floor) is listed in FEATURES and in features_schema.json. Per
frame they are on the fits' time grid (reference seconds). Per eighth note (F2) they are pooled to mean / min / max /
delta (last - first frame of the eighth) and robust-z-scored per take: z = (x - median) / max(1.4826 MAD, floor),
where the floor is a physically meaningful difference for that feature (a near-constant feature in a homogeneous
take must not be blown up to unit variance - that would invent boundaries).

Frame validity: a frame is pooled when it is inside a run of >= 8 frames, its fits are finite (non-finite frames are
bridged for the derivatives but never pooled) and its fit quality (min over the dancers) is > min_quality (0.1:
mamma_dyn caps interpolated frames at 0.1; 0.1 = a worst-part reprojection error of 23 px). eighth_valid = the
pooled fraction of each eighth's frames; proposals.py masks eighths below 50 % (a tracking dropout or a bad fit
becomes a gap in the proposals instead of silently emptying them).

Inputs are hashed into features_schema.json (fits, counterbalance.json, timing/timing.json, the beat-grid file);
`staleness(take)` / `features <take> --check` compare them with the files on disk, and proposals.json carries the
result as warnings ('stale').

features.npz (version 1; allow_pickle=False):
  t (T,) reference s | X (T, F) float32 | names / units / groups (F,) str | valid (T,) bool | quality (T,)
  run (T,) int (contiguous run index; derivatives never cross a gap)
  couple_centre (T, 3) (pelvis midpoint on the floor, VIEWER_SPEC 3.0) | frame_origin, frame_long, frame_forward (T, 2)
  com_lead, com_follow (T, 3) | grid_t, grid_type (beat grid) | measure_phase0 (index of the first downbeat)
  eighth_t0, eighth_t1, eighth_index, eighth_slot, eighth_type, eighth_coverage, eighth_frames (N,)
  eighth_valid (N,) pooled fraction of the eighth's frames | eighth_quality (N,) mean fit quality of its frames
  stats (4,) = mean, min, max, delta | pooled (N, F, 4) | pooled_z (N, F, 4) | z_center, z_scale (F, 4)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import moves_dir, work_dir

VERSION = 1
ROLES = ("lead", "follow")
STATS = ("mean", "min", "max", "delta")
DEFAULT_BPM = 76.4
FOOT_JOINTS = (7, 8, 10, 11)                                   # ankles, balls (SMPL-X 'foot')
FEET = {"left": (7, 10), "right": (8, 11)}
ARMS = {"l": (16, 18, 20), "r": (17, 19, 21)}                  # shoulder, elbow, wrist
BODY_POINTS = {"spine3": 9, "neck": 12, "l_shoulder": 16, "r_shoulder": 17, "pelvis": 0}
ZOUK_PATTERN = (1, 3, 3, 2, 1, 3, 2, 3)
# de Leva (1996)-style mass fractions onto SMPL-X joints (joints-only mode; segment = mean of its joints)
JOINT_SEGMENTS = [(0.068, (15,)), (0.43, (0, 3, 6, 9)), (0.026, (16, 18)), (0.026, (17, 19)), (0.015, (18, 20)),
                  (0.015, (19, 21)), (0.006, (20,)), (0.006, (21,)), (0.145, (1, 4)), (0.145, (2, 5)),
                  (0.045, (4, 7)), (0.045, (5, 8)), (0.013, (7, 10)), (0.013, (8, 11))]


@dataclass(frozen=True)
class Feature:
    name: str
    unit: str
    group: str            # couple | dancer | counterbalance | beat
    role: str | None      # lead | follow | None (couple)
    description: str
    floor: float          # minimum robust-z scale (a meaningful difference in `unit`)
    novelty: bool = True  # used by the boundary proposals


def _dancer_features(role: str) -> list[Feature]:
    r = role
    out = [
        Feature(f"{r}_com_height", "m", "dancer", r, "centre of mass height minus the standing height (rest pose) - "
                "negative in dips, drops, lunges", 0.03),
        Feature(f"{r}_com_speed", "m/s", "dancer", r, "horizontal speed of the centre of mass", 0.05),
        Feature(f"{r}_ke", "J/kg", "dancer", r, "kinetic energy per kg: translational + rotational energy of the "
                "body's rigid mass elements (joints mode: translational, de Leva segments)", 0.05),
        Feature(f"{r}_yaw_rate", "deg/s", "dancer", r, "pelvis (hip line) yaw rate, + = counter-clockwise from above",
                15.0),
        Feature(f"{r}_torsion", "deg", "dancer", r, "chest (shoulder line) yaw minus pelvis (hip line) yaw, wrapped "
                "to +-180 - torso torsion (the corredor signature)", 5.0),
        Feature(f"{r}_trunk_lean", "deg", "dancer", r, "angle of the pelvis->neck axis from vertical", 3.0),
        Feature(f"{r}_lean_away", "m", "dancer", r, "horizontal COM displacement AWAY from the partner relative to "
                "the contact-weighted centre of the own feet (counterbalance lean; + = leaning away)", 0.02),
        Feature(f"{r}_head_off_axis", "deg", "dancer", r, "angle between the head's up direction and the chest's up "
                "direction (head off the body axis: head movements)", 5.0),
        Feature(f"{r}_head_ang_speed", "deg/s", "dancer", r, "angular speed of the head's up direction", 20.0),
        Feature(f"{r}_arm_ext_l", "ratio", "dancer", r, "left arm extension |shoulder-wrist| / (upper arm + "
                "forearm); 1 = straight", 0.05),
        Feature(f"{r}_arm_ext_r", "ratio", "dancer", r, "right arm extension (as left)", 0.05),
        Feature(f"{r}_hand_height", "m", "dancer", r, "height of the higher hand above the dancer's floor estimate",
                0.05),
        Feature(f"{r}_foot_height", "m", "dancer", r, "height of the higher foot (its lower joint, ankle or ball) "
                "above the floor estimate - leg lifts, kicks", 0.03),
        Feature(f"{r}_contact_l", "0..1", "dancer", r, "left foot floor contact (physics: posed-sole height + "
                "speed; joints mode: soft joint-height ramp 4..9 cm)", 0.2, novelty=False),
        Feature(f"{r}_contact_r", "0..1", "dancer", r, "right foot floor contact (as left)", 0.2, novelty=False),
        Feature(f"{r}_step_rate", "steps/s", "dancer", r, "touchdown rate (timing.json touchdowns, else foot-contact "
                "onsets), Gaussian density with sigma = 1 beat", 0.3),
    ]
    if role == "follow":
        out.insert(4, Feature("follow_yaw_rel", "deg/s", "dancer", "follow", "follower pelvis yaw rate relative to "
                              "the couple frame (turns / spins inside the couple)", 15.0))
    return out


FEATURES: list[Feature] = [
    Feature("couple_speed", "m/s", "couple", None, "horizontal speed of the couple centre (midpoint of the two pelvis "
            "floor projections, VIEWER_SPEC 3.0) - the travel the graph view removes", 0.05),
    Feature("couple_yaw_rate", "deg/s", "couple", None, "yaw rate of the couple frame (the leader floor axis, "
            "VIEWER_SPEC 3.6), + = counter-clockwise from above", 15.0),
    Feature("axis_yaw_rate", "deg/s", "couple", None, "yaw rate of the horizontal leader->follower COM line "
            "(circling)", 15.0),
    Feature("com_distance", "m", "couple", None, "horizontal distance between the two centres of mass", 0.05),
    Feature("follow_lateral", "m", "couple", None, "follower COM along the couple frame's long axis (+ = the "
            "leader's left)", 0.05),
    Feature("follow_forward", "m", "couple", None, "follower COM along the couple frame's short axis (+ = in front "
            "of the leader's chest)", 0.05),
    Feature("facing_cos", "1", "couple", None, "cos of the follower's chest yaw minus the leader's (-1 = face to "
            "face, +1 = same way)", 0.1),
    Feature("facing_sin", "1", "couple", None, "sin of that angle (+ = the follower's chest turned "
            "counter-clockwise from the leader's)", 0.1),
    Feature("couple_ke", "J/kg", "couple", None, "kinetic energy of both bodies per kg of couple mass (the energy "
            "band estimate)", 0.05),
    Feature("hand_hand_min", "m", "couple", None, "smallest lead-hand to follow-hand distance (hand centre = wrist "
            "+ finger joints)", 0.03),
    Feature("n_hand_holds", "count", "couple", None, "hands joined (0, 1, 2): hand pairs closer than "
            "hand_contact_m, counting each hand once", 0.5),
    Feature("lead_hand_on_follow", "m", "couple", None, "smallest lead-hand distance to the follower's torso "
            "(spine3, neck, shoulders, pelvis)", 0.03),
    Feature("follow_hand_on_lead", "m", "couple", None, "smallest follow-hand distance to the leader's torso", 0.03),
    *_dancer_features("lead"),
    *_dancer_features("follow"),
    Feature("cb_score", "0..1", "counterbalance", None, "counterbalance tier score (counterbalance.json track; "
            "min(lean, connection) when the track has no tier column); NaN without counterbalance.json", 0.1),
    Feature("cb_lean", "0..1", "counterbalance", None, "counterbalance lean criterion", 0.1),
    Feature("cb_connection", "0..1", "counterbalance", None, "counterbalance connection criterion", 0.1),
    Feature("cb_anchor", "0..1", "counterbalance", None, "counterbalance anchor criterion (pivot foot)", 0.1),
    Feature("cb_circling", "0..1", "counterbalance", None, "counterbalance circling criterion (leader circling "
            "the pivot)", 0.1),
    Feature("cb_interval", "0/1", "counterbalance", None, "inside a detected counterbalance interval", 0.5),
    Feature("eighth_pos", "eighths", "beat", None, "continuous position on the eighth-note grid (grid index + "
            "fraction)", 1.0, novelty=False),
    Feature("beat_phase", "0..1", "beat", None, "phase inside the beat (0 = on the beat)", 0.25, novelty=False),
    Feature("measure_phase", "0..1", "beat", None, "phase inside the measure (0 = the downbeat)", 0.125,
            novelty=False),
]
NAMES = [f.name for f in FEATURES]
INDEX = {n: i for i, n in enumerate(NAMES)}


@dataclass
class FeatureConfig:
    com: str = "physics"                 # physics (body model) | joints
    cutoff_hz: float = 5.0               # quintic smoothing spline on joints / angles before differentiation
    physics_boot: int = 20               # physics.analyze_dancer bootstrap draws (its force bands are not used here)
    floor_pct: float = 5.0               # per-dancer floor = this percentile of the lowest foot joint
    contact_h_m: tuple = (0.04, 0.09)    # joints mode: foot height full contact .. none (as counterbalance)
    hand_contact_m: float = 0.15         # physics.CONTACT_DIST['hand']
    step_sigma_beats: float = 1.0
    onset_min_s: float = 0.15            # contact onsets closer than this are one touchdown
    gap_frames: float = 1.5              # a time step > this x the median step splits the grid into runs
    min_eighth_coverage: float = 0.5     # pooled eighths need this fraction of their span covered by frames
    min_quality: float = 0.1             # frames whose fit quality (min over the dancers) is <= this are not pooled:
                                         # mamma_dyn quality = exp(-worst part reprojection px / 10) and caps
                                         # interpolated frames at 0.1, so 0.1 = interpolated or >= 23 px; 0 disables


@dataclass
class FeatureSet:
    take: str
    t: np.ndarray
    X: np.ndarray
    valid: np.ndarray
    quality: np.ndarray
    run: np.ndarray
    frame: dict                          # couple_centre, frame_origin, frame_long, frame_forward, com_lead, com_follow
    grid_t: np.ndarray
    grid_type: np.ndarray
    phase0: int
    eighths: dict                        # t0, t1, index, slot, type, coverage, frames
    pooled: np.ndarray
    pooled_z: np.ndarray
    z_center: np.ndarray
    z_scale: np.ndarray
    meta: dict = field(default_factory=dict)

    def col(self, name: str) -> np.ndarray:
        return self.X[:, INDEX[name]]

    def pooled_col(self, name: str, stat: str = "mean", z: bool = False) -> np.ndarray:
        a = self.pooled_z if z else self.pooled
        return a[:, INDEX[name], STATS.index(stat)]


# ----------------------------------------------------------------------------------------------- small helpers

def floor_angle(v: np.ndarray) -> np.ndarray:
    """Angle (rad) of horizontal (x, z) vectors (..., 2), + = counter-clockwise seen from above (about +Y)."""
    return np.arctan2(-v[..., 1], v[..., 0])


def wrap_deg(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = (_unit(a) * _unit(b)).sum(-1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _ramp_down(x, good, bad):
    return np.clip((bad - np.asarray(x, float)) / (bad - good), 0.0, 1.0)


def hand_center(joints: np.ndarray, side: str) -> np.ndarray:
    """Wrist + mean finger joint (as physics._hand_center); wrist only when the fit has no finger joints."""
    wrist = 20 if side == "l" else 21
    if joints.shape[1] < 55:
        return joints[:, wrist]
    hand = range(25, 40) if side == "l" else range(40, 55)
    return 0.5 * joints[:, wrist] + 0.5 * joints[:, list(hand)].mean(1)


def arm_extension(joints: np.ndarray, side: str) -> np.ndarray:
    s, e, w = (joints[:, j] for j in ARMS[side])
    return np.linalg.norm(s - w, axis=1) / np.maximum(np.linalg.norm(s - e, axis=1) + np.linalg.norm(e - w, axis=1),
                                                      1e-6)


def joints_com(joints: np.ndarray) -> np.ndarray:
    w = np.array([s[0] for s in JOINT_SEGMENTS])
    c = np.stack([joints[:, list(s[1])].mean(1) for s in JOINT_SEGMENTS], 1)
    return (w[None, :, None] * c).sum(1) / w.sum()


def contiguous_runs(t: np.ndarray, gap_frames: float = 1.5) -> list[tuple[int, int]]:
    """[(i0, i1 inclusive)] of the time grid split where a step exceeds gap_frames x the median step."""
    if len(t) < 2:
        return [(0, len(t) - 1)]
    d = np.diff(t)
    cut = np.flatnonzero(d > gap_frames * float(np.median(d)))
    starts = np.concatenate([[0], cut + 1])
    ends = np.concatenate([cut, [len(t) - 1]])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _smoother(t: np.ndarray, cutoff_hz: float):
    from ..physics_spline import Smoother
    return Smoother(t, cutoff_hz)


def _bridge(a: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate non-finite frames of a (T, ...) array; returns (array, bad (T,) bool)."""
    a = np.asarray(a, float).copy()
    flat = a.reshape(len(a), -1)
    bad = ~np.isfinite(flat).all(1)
    if bad.any() and (~bad).sum() >= 2:
        good = np.flatnonzero(~bad)
        for c in range(flat.shape[1]):
            flat[bad, c] = np.interp(t[bad], t[good], flat[good, c])
    return flat.reshape(a.shape), bad


def _sha16(p: Path) -> str | None:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


# ----------------------------------------------------------------------------------------------- per dancer

def _dancer_physics(npz: dict, cfg: FeatureConfig, mass: float | None) -> dict:
    """Body-model quantities via physics.analyze_dancer (+ FK rotations)."""
    from .. import physics as ph
    from .. import physics_body as pb
    pc = ph.PhysicsConfig(n_boot=cfg.physics_boot, cutoff_hz=cfg.cutoff_hz)
    res = ph.analyze_dancer(npz, pc, mass=mass)
    t = res["t"]
    T = len(t)
    body = res["body"]
    d, _, _, _ = ph.sanitize(npz, pc)
    R, p = pb.forward_kinematics(body, pb.full_pose(d), np.asarray(d["transl"], float))
    S = _smoother(t, cfg.cutoff_hz)
    c, I = pb.element_world(body, R, p)
    vc = S(c.reshape(T, -1), 1).reshape(T, -1, 3)
    m = body.elem_mass
    ke_t = 0.5 * (m[None, :] * (vc ** 2).sum(-1)).sum(1)
    Rd = S(R.reshape(T, -1), 1).reshape(R.shape)
    W = Rd @ np.swapaxes(R, -1, -2)
    W = 0.5 * (W - np.swapaxes(W, -1, -2))
    omega = np.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], -1)          # (T, 55, 3) world angular velocity
    ke_r = 0.5 * np.einsum("tja,tjab,tjb->t", omega, I, omega)
    J = np.einsum("tjab,jb->tja", R, body.j_rest) + p
    rest_foot = float(body.j_rest[list(FOOT_JOINTS), 1].min())
    return {"t": t, "joints": J, "com": res["com"], "vcom": res["vcom"], "mass": float(body.mass),
            "mass_source": body.mass_source, "ke": (ke_t + ke_r) / body.mass,
            "contact": res["contact"].astype(float),                           # (T, 2) left, right
            "head_up": R[:, 15, :, 1], "chest_up": R[:, 9, :, 1],
            "standing_h": float(body.rest_com[1]) - rest_foot, "standing_source": "rest pose (body model)",
            "bad": res["bad"] | res["gap"], "contact_source": res["info"].get("contact_source", "physics"),
            "com_source": "physics.analyze_dancer (shaped SMPL-X mass elements)"}


def _dancer_joints(npz: dict, cfg: FeatureConfig, mass: float | None, stature: float | None) -> dict:
    """Joint-positions-only quantities (no body model)."""
    t = np.asarray(npz["t"], float)
    J, bad = _bridge(np.asarray(npz["joints"], float)[:, :55], t)
    S = _smoother(t, cfg.cutoff_hz)
    Js = S(J.reshape(len(t), -1)).reshape(J.shape)
    com_raw = joints_com(J)
    com, vcom = S(com_raw), S(com_raw, 1)
    M = float(mass) if mass else (22.0 * stature ** 2 if stature else 70.0)
    seg = np.stack([J[:, list(s[1])].mean(1) for s in JOINT_SEGMENTS], 1)
    vs = S(seg.reshape(len(t), -1), 1).reshape(seg.shape)
    w = np.array([s[0] for s in JOINT_SEGMENTS]) / sum(s[0] for s in JOINT_SEGMENTS)
    ke = 0.5 * (w[None, :] * (vs ** 2).sum(-1)).sum(1)                        # J/kg
    fl = float(np.percentile(Js[:, list(FOOT_JOINTS), 1].min(1), cfg.floor_pct))
    contact = np.stack([_ramp_down(Js[:, list(FEET[f]), 1].min(1) - fl, *cfg.contact_h_m) for f in FEET], 1)
    h = com[:, 1] - fl
    return {"t": t, "joints": J, "com": com, "vcom": vcom, "mass": M,
            "mass_source": "takes toml mass_kg" if mass else ("22 kg/m^2 x stature^2" if stature else "70 kg default"),
            "ke": ke, "contact": contact,
            "head_up": Js[:, 15] - Js[:, 12], "chest_up": Js[:, 12] - Js[:, 9],
            "standing_h": float(np.percentile(h, 95)), "standing_source": "95th percentile of the take (joints mode)",
            "bad": bad, "contact_source": "joint heights (soft ramp)",
            "com_source": "joints (de Leva-style segment weights)"}


def _run_features(npzs: dict, cfg: FeatureConfig, masses: dict, statures: dict) -> tuple[dict, dict, dict]:
    """Per-frame features of one contiguous run: (columns {name: (T,)}, frame arrays, info)."""
    D = {}
    for r in ROLES:
        if cfg.com == "physics":
            D[r] = _dancer_physics(npzs[r], cfg, masses.get(r))
        elif cfg.com == "joints":
            D[r] = _dancer_joints(npzs[r], cfg, masses.get(r), statures.get(r))
        else:
            raise ValueError(f"unknown com mode {cfg.com!r} (physics | joints)")
    t = D["lead"]["t"]
    T = len(t)
    S = _smoother(t, cfg.cutoff_hz)
    col: dict[str, np.ndarray] = {}
    Js = {r: S(D[r]["joints"].reshape(T, -1)).reshape(D[r]["joints"].shape) for r in ROLES}
    floor = {r: float(np.percentile(Js[r][:, list(FOOT_JOINTS), 1].min(1), cfg.floor_pct)) for r in ROLES}

    def rate(angle_rad: np.ndarray) -> np.ndarray:
        return np.degrees(S(np.unwrap(angle_rad), 1))

    # couple frame (VIEWER_SPEC 3.6)
    JL = Js["lead"]
    theta_l = np.unwrap(floor_angle((JL[:, 16] - JL[:, 17])[:, [0, 2]]))
    long_ = np.stack([np.cos(theta_l), -np.sin(theta_l)], 1)
    fwd = np.stack([-long_[:, 1], long_[:, 0]], 1)
    origin = JL[:, 9][:, [0, 2]]
    centre = 0.5 * (Js["lead"][:, 0] + Js["follow"][:, 0])
    centre[:, 1] = 0.0
    com = {r: D[r]["com"] for r in ROLES}
    d_lf = (com["follow"] - com["lead"])[:, [0, 2]]
    col["couple_speed"] = np.linalg.norm(S(centre[:, [0, 2]], 1), axis=1)
    col["couple_yaw_rate"] = rate(theta_l)
    col["axis_yaw_rate"] = rate(floor_angle(d_lf))
    col["com_distance"] = np.linalg.norm(d_lf, axis=1)
    rel = com["follow"][:, [0, 2]] - origin
    col["follow_lateral"] = (rel * long_).sum(1)
    col["follow_forward"] = (rel * fwd).sum(1)
    chest_yaw = {r: np.unwrap(floor_angle((Js[r][:, 16] - Js[r][:, 17])[:, [0, 2]])) for r in ROLES}
    pelvis_yaw = {r: np.unwrap(floor_angle((Js[r][:, 1] - Js[r][:, 2])[:, [0, 2]])) for r in ROLES}
    dyaw = chest_yaw["follow"] - chest_yaw["lead"]
    col["facing_cos"], col["facing_sin"] = np.cos(dyaw), np.sin(dyaw)
    M = D["lead"]["mass"] + D["follow"]["mass"]
    col["couple_ke"] = (D["lead"]["ke"] * D["lead"]["mass"] + D["follow"]["ke"] * D["follow"]["mass"]) / M
    hands = {r: {s: hand_center(Js[r], s) for s in ("l", "r")} for r in ROLES}
    hh = np.stack([[np.linalg.norm(hands["lead"][a] - hands["follow"][b], axis=1) for b in "lr"] for a in "lr"])
    # hh: (2 lead hands, 2 follow hands, T)
    col["hand_hand_min"] = hh.reshape(4, T).min(0)
    near = hh < cfg.hand_contact_m
    col["n_hand_holds"] = np.minimum(near.any(1).sum(0), near.any(0).sum(0)).astype(float)
    for r, other, name in (("lead", "follow", "lead_hand_on_follow"), ("follow", "lead", "follow_hand_on_lead")):
        ds = [np.linalg.norm(hands[r][s] - Js[other][:, j], axis=1) for s in "lr" for j in BODY_POINTS.values()]
        col[name] = np.min(np.stack(ds), 0)
    # per dancer
    for r in ROLES:
        other = "follow" if r == "lead" else "lead"
        J = Js[r]
        h = D[r]["com"][:, 1] - floor[r]
        col[f"{r}_com_height"] = h - D[r]["standing_h"]
        col[f"{r}_com_speed"] = np.linalg.norm(D[r]["vcom"][:, [0, 2]], axis=1)
        col[f"{r}_ke"] = D[r]["ke"]
        col[f"{r}_yaw_rate"] = rate(pelvis_yaw[r])
        if r == "follow":
            col["follow_yaw_rel"] = col["follow_yaw_rate"] - col["couple_yaw_rate"]
        col[f"{r}_torsion"] = wrap_deg(np.degrees(chest_yaw[r] - pelvis_yaw[r]))
        col[f"{r}_trunk_lean"] = _angle_between(J[:, 12] - J[:, 0], np.array([0.0, 1.0, 0.0]))
        u = _unit((com[r] - com[other])[:, [0, 2]])
        w = D[r]["contact"] + 1e-3
        mid = {f: 0.5 * (J[:, a] + J[:, b])[:, [0, 2]] for f, (a, b) in FEET.items()}
        base = (w[:, :1] * mid["left"] + w[:, 1:] * mid["right"]) / w.sum(1, keepdims=True)
        col[f"{r}_lean_away"] = ((com[r][:, [0, 2]] - base) * u).sum(1)
        hu, cu = S(D[r]["head_up"]), S(D[r]["chest_up"])
        col[f"{r}_head_off_axis"] = _angle_between(hu, cu)
        hun = _unit(hu)
        col[f"{r}_head_ang_speed"] = np.degrees(np.linalg.norm(np.cross(hun, S(hun, 1)), axis=1))
        col[f"{r}_arm_ext_l"], col[f"{r}_arm_ext_r"] = arm_extension(J, "l"), arm_extension(J, "r")
        col[f"{r}_hand_height"] = np.maximum(hands[r]["l"][:, 1], hands[r]["r"][:, 1]) - floor[r]
        col[f"{r}_foot_height"] = np.maximum(J[:, list(FEET["left"]), 1].min(1),
                                             J[:, list(FEET["right"]), 1].min(1)) - floor[r]
        col[f"{r}_contact_l"], col[f"{r}_contact_r"] = D[r]["contact"][:, 0], D[r]["contact"][:, 1]
    frame = {"couple_centre": centre, "frame_origin": origin, "frame_long": long_, "frame_forward": fwd,
             "com_lead": com["lead"], "com_follow": com["follow"]}
    bad = D["lead"]["bad"] | D["follow"]["bad"]
    info = {r: {"mass_kg": round(D[r]["mass"], 2), "mass_source": D[r]["mass_source"],
                "com_source": D[r]["com_source"], "contact_source": D[r]["contact_source"],
                "standing_com_height_m": round(D[r]["standing_h"], 4), "standing_source": D[r]["standing_source"],
                "floor_estimate_m": round(floor[r], 4)} for r in ROLES}
    contacts = {r: D[r]["contact"] for r in ROLES}
    return col, {**frame, "bad": bad, "contacts": contacts}, info


# ----------------------------------------------------------------------------------------------- time-based columns

def step_rate(t: np.ndarray, touchdowns: np.ndarray, sigma_s: float) -> np.ndarray:
    """Touchdowns per second: Gaussian kernel density, normalised by the kernel mass inside [t0, t1] (no edge drop)."""
    t = np.asarray(t, float)
    td = np.asarray(touchdowns, float)
    if not len(td):
        return np.zeros_like(t)
    k = np.exp(-0.5 * ((t[:, None] - td[None, :]) / sigma_s) ** 2).sum(1) / (sigma_s * math.sqrt(2 * math.pi))
    from scipy.special import ndtr
    inside = ndtr((t[-1] - t) / sigma_s) - ndtr((t[0] - t) / sigma_s)
    return k / np.maximum(inside, 0.5)


def contact_onsets(t: np.ndarray, contact: np.ndarray, min_s: float) -> np.ndarray:
    """Touchdown times = upward crossings of 0.5 per foot (contact (T, 2)); crossings closer than min_s merged."""
    out = []
    for k in range(contact.shape[1]):
        c = contact[:, k] >= 0.5
        idx = np.flatnonzero(c[1:] & ~c[:-1]) + 1
        last = -1e9
        for i in idx:
            if t[i] - last >= min_s:
                out.append(float(t[i]))
                last = t[i]
    return np.sort(np.asarray(out))


def beat_columns(t: np.ndarray, grid_t: np.ndarray, phase0: int) -> dict:
    """eighth_pos, beat_phase (0 = on the beat), measure_phase (0 = downbeat); NaN outside the grid."""
    g = np.asarray(grid_t, float)
    out = {k: np.full(len(t), np.nan) for k in ("eighth_pos", "beat_phase", "measure_phase")}
    if len(g) < 2:
        return out
    k = np.searchsorted(g, t, side="right") - 1
    ok = (k >= 0) & (k < len(g) - 1)
    kk = np.clip(k, 0, len(g) - 2)
    frac = (t - g[kk]) / np.maximum(g[kk + 1] - g[kk], 1e-9)
    pos = kk + frac
    slot = (pos - phase0) % 8.0
    out["eighth_pos"] = np.where(ok, pos, np.nan)
    out["beat_phase"] = np.where(ok, (slot % 2.0) / 2.0, np.nan)
    out["measure_phase"] = np.where(ok, slot / 8.0, np.nan)
    return out


def counterbalance_columns(t: np.ndarray, doc: dict | None) -> dict:
    """cb_* columns from counterbalance.json's per-frame track (interpolated; NaN outside its time range)."""
    names = ("cb_score", "cb_lean", "cb_connection", "cb_anchor", "cb_circling", "cb_interval")
    out = {n: np.full(len(t), np.nan) for n in names}
    tr = (doc or {}).get("track") or {}
    if not tr.get("t"):
        return out
    tt = np.asarray(tr["t"], float)
    inside = (t >= tt[0] - 1e-6) & (t <= tt[-1] + 1e-6)

    def get(key):
        if key not in tr:
            return None
        return np.where(inside, np.interp(t, tt, np.asarray(tr[key], float)), np.nan)

    for n, key in (("cb_lean", "lean"), ("cb_connection", "connection"), ("cb_anchor", "anchor"),
                   ("cb_circling", "circling")):
        v = get(key)
        if v is not None:
            out[n] = v
    tier = get("counterbalance")
    out["cb_score"] = tier if tier is not None else np.minimum(out["cb_lean"], out["cb_connection"])
    iv = np.zeros(len(t))
    for i in doc.get("intervals", []):
        iv[(t >= float(i["t0"])) & (t <= float(i["t1"]))] = 1.0
    out["cb_interval"] = np.where(inside, iv, np.nan)
    return out


# ----------------------------------------------------------------------------------------------- beat grid

def _grid_rows(work: Path) -> tuple[list | None, Path | None]:
    """(grid rows [[t, type], ...], the file they came from): timing/timing.json beats.grid, else a beats.json."""
    tj = Path(work) / "timing" / "timing.json"
    if tj.exists():
        try:
            rows = json.loads(tj.read_text()).get("beats", {}).get("grid")
        except ValueError:
            rows = None
        if rows:
            return rows, tj
    for p in (work / "timing" / "beats.json", work / "beats.json", work / "beats" / "beats.json"):
        if p.exists():
            d = json.loads(p.read_text())
            rows = d.get("grid") if isinstance(d, dict) else d
            if rows:
                return rows, p
    return None, None


def grid_source(work: Path) -> Path | None:
    """The file load_grid reads the beat grid from (None: no grid)."""
    return _grid_rows(Path(work))[1]


def load_grid(work: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Beat grid (times, types), reference seconds: timing/timing.json grid, else beats.json (zouk eighths)."""
    rows, _ = _grid_rows(Path(work))
    if not rows:
        return None
    return np.array([float(r[0]) for r in rows]), np.array([int(r[1]) if len(r) > 1 else 3 for r in rows])


def fallback_grid(t0: float, t1: float, bpm: float = DEFAULT_BPM) -> tuple[np.ndarray, np.ndarray]:
    """A constant-tempo eighth grid starting on a downbeat at t0 (used only when the take has no beat grid)."""
    step = 30.0 / bpm
    n = int(math.ceil((t1 - t0) / step)) + 9
    gt = t0 + step * np.arange(n)
    return gt, np.array([ZOUK_PATTERN[i % 8] for i in range(n)])


def grid_phase(grid_type: np.ndarray) -> int:
    from .placeholder import measure_phase
    return measure_phase([[0.0, int(x)] for x in grid_type])


def eighth_table(t: np.ndarray, grid_t: np.ndarray, phase0: int, grid_type: np.ndarray,
                 min_coverage: float = 0.5) -> dict:
    """The grid's eighths that the frames cover: [t0, t1), grid index, slot in the measure (0..7), type, the
    covered fraction of the eighth (by frame time span) and the frame index range [f0, f1)."""
    g = np.asarray(grid_t, float)
    rows = {k: [] for k in ("t0", "t1", "index", "slot", "type", "coverage", "f0", "f1")}
    if len(t) < 2 or len(g) < 2:
        return {k: np.asarray(v) for k, v in rows.items()}
    dt = float(np.median(np.diff(t)))
    for k in range(len(g) - 1):
        a, b = g[k], g[k + 1]
        if b <= t[0] or a > t[-1]:
            continue
        f0, f1 = int(np.searchsorted(t, a, "left")), int(np.searchsorted(t, b, "left"))
        if f1 - f0 < 1:
            continue
        cov = min(1.0, (min(t[f1 - 1] + dt, b) - max(t[f0], a)) / (b - a))
        if cov < min_coverage:
            continue
        for key, v in (("t0", a), ("t1", b), ("index", k), ("slot", (k - phase0) % 8), ("type", int(grid_type[k])),
                       ("coverage", round(cov, 4)), ("f0", f0), ("f1", f1)):
            rows[key].append(v)
    out = {k: np.asarray(v) for k, v in rows.items()}
    for k in ("index", "slot", "type", "f0", "f1"):
        out[k] = out[k].astype(int)
    return out


def pool(X: np.ndarray, eighths: dict) -> np.ndarray:
    """(N, F, 4) mean / min / max / delta of the frames inside each eighth (NaN-safe)."""
    N, F = len(eighths["t0"]), X.shape[1]
    P = np.full((N, F, len(STATS)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, (a, b) in enumerate(zip(eighths["f0"], eighths["f1"])):
            seg = X[a:b]
            P[i, :, 0] = np.nanmean(seg, 0)
            P[i, :, 1] = np.nanmin(seg, 0)
            P[i, :, 2] = np.nanmax(seg, 0)
            P[i, :, 3] = seg[-1] - seg[0] if len(seg) > 1 else 0.0
    return P


def _eighth_frame_stats(eighths: dict, valid: np.ndarray, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(valid-frame fraction, mean fit quality) of each pooled eighth's frames [f0, f1)."""
    vf, mq = [], []
    for a, b in zip(eighths["f0"], eighths["f1"]):
        a, b = int(a), int(b)
        vf.append(float(np.mean(valid[a:b])) if b > a else 0.0)
        mq.append(float(np.mean(quality[a:b])) if b > a else 0.0)
    return np.round(np.asarray(vf, float), 4), np.round(np.asarray(mq, float), 4)


def eighth_valid_fraction(fs: "FeatureSet") -> np.ndarray:
    """Fraction of each pooled eighth's frames that were pooled (finite fits, inside a run, fit quality above
    FeatureConfig.min_quality). Read from features.npz; recomputed from fs.valid for older files / test sets."""
    v = fs.eighths.get("valid")
    if v is not None and len(v) == len(fs.eighths["t0"]):
        return np.asarray(v, float)
    return _eighth_frame_stats(fs.eighths, np.asarray(fs.valid, bool), np.asarray(fs.quality, float))[0]


def eighth_low_quality_fraction(fs: "FeatureSet") -> np.ndarray:
    """Fraction of each pooled eighth's frames whose fit quality is <= the features' min_quality."""
    mq = float(((fs.meta or {}).get("config") or {}).get("min_quality", FeatureConfig.min_quality))
    q = np.asarray(fs.quality, float)
    low = (q <= mq) if mq > 0 else np.zeros(len(q), bool)
    return _eighth_frame_stats(fs.eighths, low, q)[0]


def robust_z(P: np.ndarray, floors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per take: z = (x - median) / max(1.4826 MAD, floor) per (feature, stat); returns (z, center, scale)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmedian(P, 0)
        mad = 1.4826 * np.nanmedian(np.abs(P - center[None]), 0)
    scale = np.maximum(np.nan_to_num(mad, nan=0.0), np.asarray(floors, float)[:, None])
    center = np.where(np.isfinite(center), center, 0.0)
    return (P - center[None]) / scale[None], center, scale


# ----------------------------------------------------------------------------------------------- main computation

def _slice(npz: dict, a: int, b: int, T: int) -> dict:
    out = {}
    for k, v in npz.items():
        arr = np.asarray(v)
        out[k] = arr[a:b + 1] if (k != "betas" and arr.ndim >= 1 and arr.shape[0] == T) else v
    return out


def compute_features(npzs: dict, cfg: FeatureConfig | None = None, grid: tuple | None = None,
                     cb_doc: dict | None = None, touchdowns: dict | None = None, masses: dict | None = None,
                     statures: dict | None = None, take: str = "") -> FeatureSet:
    """Features for {role: smplx npz dict} (lead and follow on the same time grid).

    grid: (times, types) beat grid in the fits' time base (None: a constant 76.4 BPM grid from the first frame,
    recorded as a warning). cb_doc: counterbalance.json (or None). touchdowns: {role: [t, ...]} (None: foot-contact
    onsets)."""
    cfg = cfg or FeatureConfig()
    masses, statures = dict(masses or {}), dict(statures or {})
    t = np.asarray(npzs["lead"]["t"], float)
    tf = np.asarray(npzs["follow"]["t"], float)
    if len(t) != len(tf) or np.abs(t - tf).max() > 1e-6:
        raise ValueError("lead and follow must share the time grid")
    T = len(t)
    warn: list[str] = []
    runs = contiguous_runs(t, cfg.gap_frames)
    cols, frames, infos = [], [], []
    run_id = np.full(T, -1, int)
    for k, (a, b) in enumerate(runs):
        if b - a + 1 < 8:
            warn.append(f"run [{t[a]:.2f}, {t[b]:.2f}] has < 8 frames: skipped (features NaN)")
            continue
        c, f, info = _run_features({r: _slice(npzs[r], a, b, T) for r in ROLES}, cfg, masses, statures)
        cols.append((a, b, c))
        frames.append((a, b, f))
        infos.append(info)
        run_id[a:b + 1] = k
    if not cols:
        raise ValueError("no contiguous run with >= 8 frames")
    X = np.full((T, len(FEATURES)), np.nan)
    frame = {k: np.full((T,) + np.shape(v)[1:], np.nan) for k, v in frames[0][2].items()
             if k not in ("bad", "contacts")}
    bad = np.ones(T, bool)
    contacts = {r: np.zeros((T, 2)) for r in ROLES}
    for (a, b, c), (_, _, f) in zip(cols, frames):
        for n, v in c.items():
            X[a:b + 1, INDEX[n]] = v
        for n in frame:
            frame[n][a:b + 1] = f[n]
        bad[a:b + 1] = f["bad"]
        for r in ROLES:
            contacts[r][a:b + 1] = f["contacts"][r]
    # beat grid and beat-time columns
    if grid is None or len(grid[0]) < 9:
        warn.append(f"no beat grid: a constant {DEFAULT_BPM} BPM grid from the first frame is used (pooling and "
                    "snapping are then NOT on the music)")
        grid = fallback_grid(float(t[0]), float(t[-1]))
    grid_t, grid_type = np.asarray(grid[0], float), np.asarray(grid[1], int)
    phase0 = grid_phase(grid_type)
    beat_s = 2.0 * float(np.median(np.diff(grid_t)))
    for n, v in beat_columns(t, grid_t, phase0).items():
        X[:, INDEX[n]] = v
    # step rate
    sources = {}
    for r in ROLES:
        td = None if touchdowns is None else touchdowns.get(r)
        if td is not None and len(td) and (np.min(td) <= t[-1] and np.max(td) >= t[0]):
            sources[r] = "timing.json touchdowns"
            tds = np.asarray(td, float)
        else:
            sources[r] = "foot-contact onsets"
            tds = contact_onsets(t, contacts[r], cfg.onset_min_s)
        sr = np.full(T, np.nan)
        for a, b, _ in cols:
            sr[a:b + 1] = step_rate(t[a:b + 1], tds, cfg.step_sigma_beats * beat_s)
        X[:, INDEX[f"{r}_step_rate"]] = sr
    # counterbalance
    for n, v in counterbalance_columns(t, cb_doc).items():
        X[:, INDEX[n]] = v
    if cb_doc is None:
        warn.append("no counterbalance.json: cb_* features are NaN (proposals ignore them)")
    q = np.ones(T)
    for r in ROLES:
        if "quality" in npzs[r]:
            q = np.minimum(q, np.nan_to_num(np.asarray(npzs[r]["quality"], float), nan=0.0))
    low_q = (q <= cfg.min_quality) if cfg.min_quality > 0 else np.zeros(T, bool)
    valid = (run_id >= 0) & ~bad & ~low_q
    if low_q.any():
        warn.append(f"{int(low_q.sum())} frame(s) with fit quality <= {cfg.min_quality:g} (interpolated, or a worst-part "
                    "reprojection error >= 23 px) are not pooled; eighths with < 50 % valid frames are masked in the "
                    "proposals")
    eighths = eighth_table(t, grid_t, phase0, grid_type, cfg.min_eighth_coverage)
    eighths["valid"], eighths["quality"] = _eighth_frame_stats(eighths, valid, q)
    Xv = np.where(valid[:, None], X, np.nan)
    P = pool(Xv, eighths)
    Z, center, scale = robust_z(P, np.array([f.floor for f in FEATURES]))
    meta = {"config": asdict(cfg), "runs": [[round(float(t[a]), 4), round(float(t[b]), 4)] for a, b, _ in cols],
            "dancers": infos[0], "step_rate_source": sources, "beat_s": round(beat_s, 5), "warnings": warn,
            "invalid_frames": int((~valid).sum()), "low_quality_frames": int(low_q.sum()),
            "bad_frames": int(((run_id < 0) | bad).sum())}
    return FeatureSet(take=take, t=t, X=X.astype(np.float32), valid=valid, quality=q, run=run_id, frame=frame,
                      grid_t=grid_t, grid_type=grid_type, phase0=phase0, eighths=eighths, pooled=P.astype(np.float32),
                      pooled_z=Z.astype(np.float32), z_center=center, z_scale=scale, meta=meta)


# ----------------------------------------------------------------------------------------------- take I/O

def find_smplx_dir(take: str, smplx_dir: str | Path | None = None, work: Path | None = None) -> Path:
    """work/<take>/smplx/ (current fits), else smplx_v1/, or an explicit directory (absolute or under work/<take>)."""
    work = work or work_dir(take)
    if smplx_dir:
        p = Path(smplx_dir)
        cands = [p if p.is_absolute() else work / p]
    else:
        cands = [work / "smplx", work / "smplx_v1"]
    for d in cands:
        if all((d / f"{r}.npz").exists() for r in ROLES):
            return d
    raise FileNotFoundError(f"no lead/follow npz in {[str(c) for c in cands]}")


def load_npzs(d: Path) -> dict:
    out = {}
    for r in ROLES:
        with np.load(d / f"{r}.npz", allow_pickle=False) as z:
            out[r] = {k: z[k] for k in z.files}
    return out


def world_frame_warning(d: Path) -> str | None:
    """The F1 warning when the fits are (identical to) smplx_v1."""
    v1 = d.parent / "smplx_v1"
    same = d.name == "smplx_v1" or ((v1 / "lead.npz").exists() and _sha16(d / "lead.npz") == _sha16(v1 / "lead.npz"))
    if same:
        return ("fits are smplx_v1 (OLD world frame, ~4.8 deg off gravity = ~0.08 m at hip height): lean, trunk lean "
                "and COM-height features are biased; re-run after the Phase 2b refit writes work/<take>/smplx/")
    return None


def take_dancers(take: str) -> tuple[dict, dict]:
    import tomllib
    from ..take import PACKAGE_ROOT
    p = PACKAGE_ROOT / "takes" / f"{take}.toml"
    dancers = tomllib.loads(p.read_text(encoding="utf-8")).get("dancers", {}) if p.exists() else {}
    masses = {r: float(dancers[r]["mass_kg"]) for r in ROLES if dancers.get(r, {}).get("mass_kg")}
    statures = {r: float(dancers[r]["stature_m"]) for r in ROLES if dancers.get(r, {}).get("stature_m")}
    return masses, statures


def load_touchdowns(work: Path) -> tuple[dict | None, Path | None]:
    p = work / "timing" / "timing.json"
    if not p.exists():
        return None, None
    try:
        fs = json.loads(p.read_text()).get("footsteps") or {}
    except ValueError:
        return None, None
    out = {r: [float(e["t"]) for e in fs.get(r, []) if "t" in e] for r in ROLES}
    return (out if any(out.values()) else None), p


def features_paths(take: str, out_dir: Path | None = None) -> tuple[Path, Path]:
    d = Path(out_dir) if out_dir else moves_dir(take)
    return d / "features.npz", d / "features_schema.json"


def schema(fs: FeatureSet, source: dict) -> dict:
    t = fs.t
    m = fs.meta
    cfg = m.get("config") or asdict(FeatureConfig())
    return {
        "version": VERSION, "kind": "features", "take": fs.take, "time_base": "reference", "frame": "world",
        "note": "Interpretable motion descriptors for move proposals and (later) training - NOT move "
                "classifications.",
        "couple_frame": "VIEWER_SPEC 3.6 leader floor axis: origin = leader spine3 (joint 9) on the floor; long axis = "
                        "leader shoulder line (joint 16 - joint 17) on the floor, + = the leader's left; short axis "
                        "= forward from the chest = long x up, forward_xz = (-long_z, long_x). Angles about +Y, "
                        "+ = counter-clockwise from above. World: right-handed, Y-up, metres, floor y = 0.",
        "source": source,
        "n_frames": int(len(t)), "t_range": [round(float(t[0]), 4), round(float(t[-1]), 4)],
        "fps": round(1.0 / float(np.median(np.diff(t))), 3) if len(t) > 1 else None,
        "runs": m.get("runs"), "invalid_frames": m.get("invalid_frames"),
        "beats": {"beat_s": m.get("beat_s"), "measure_phase0": int(fs.phase0),
                  "n_eighths": int(len(fs.eighths["t0"])),
                  "pooling": "per eighth note of the beat grid; eighths need >= "
                             f"{cfg.get('min_eighth_coverage')} of their span covered by frames"},
        "stats": list(STATS),
        "z_score": "robust per take: (x - median) / max(1.4826 MAD, floor) per feature and stat (floor = feature "
                   "'floor', in its unit)",
        "dancers": m.get("dancers"), "step_rate_source": m.get("step_rate_source"),
        "features": [{"index": i, **asdict(f)} for i, f in enumerate(FEATURES)],
        "config": cfg, "warnings": m.get("warnings", []),
    }


def save(fs: FeatureSet, source: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    npz_p, sch_p = features_paths(fs.take, out_dir)
    npz_p.parent.mkdir(parents=True, exist_ok=True)
    e = fs.eighths
    np.savez_compressed(
        npz_p, version=np.int64(VERSION), take=np.str_(fs.take), t=fs.t, X=fs.X,
        names=np.array(NAMES), units=np.array([f.unit for f in FEATURES]),
        groups=np.array([f.group for f in FEATURES]), valid=fs.valid, quality=fs.quality, run=fs.run,
        grid_t=fs.grid_t, grid_type=fs.grid_type, measure_phase0=np.int64(fs.phase0),
        eighth_t0=e["t0"], eighth_t1=e["t1"], eighth_index=e["index"], eighth_slot=e["slot"], eighth_type=e["type"],
        eighth_coverage=e["coverage"], eighth_frames=np.stack([e["f0"], e["f1"]], 1) if len(e["f0"]) else
        np.zeros((0, 2), int), eighth_valid=eighth_valid_fraction(fs),
        eighth_quality=np.asarray(e.get("quality", np.full(len(e["t0"]), np.nan)), float),
        stats=np.array(STATS), pooled=fs.pooled, pooled_z=fs.pooled_z, z_center=fs.z_center, z_scale=fs.z_scale,
        **{k: v for k, v in fs.frame.items()})
    doc = schema(fs, source)
    doc["files"] = {"npz": npz_p.name, "npz_sha256_16": _sha16(npz_p)}
    sch_p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return npz_p, sch_p


def load(take_or_path: str | Path) -> FeatureSet:
    """FeatureSet from a take name (work/<take>/moves/features.npz) or an npz path."""
    p = Path(take_or_path)
    if p.suffix != ".npz":
        p = features_paths(str(take_or_path))[0]
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    names = [str(x) for x in d["names"]]
    if names != NAMES:
        missing = sorted(set(NAMES) - set(names))
        raise ValueError(f"{p}: feature list differs from this code ({len(names)} vs {len(NAMES)} columns; missing "
                         f"{missing[:5]}): re-run `python -m dancecap.moves features <take>`")
    fr = d["eighth_frames"]
    eighths = {"t0": d["eighth_t0"], "t1": d["eighth_t1"], "index": d["eighth_index"], "slot": d["eighth_slot"],
               "type": d["eighth_type"], "coverage": d["eighth_coverage"], "f0": fr[:, 0], "f1": fr[:, 1]}
    for k in ("valid", "quality"):
        if f"eighth_{k}" in d:
            eighths[k] = d[f"eighth_{k}"]
    frame = {k: d[k] for k in ("couple_centre", "frame_origin", "frame_long", "frame_forward", "com_lead",
                               "com_follow") if k in d}
    sch = p.with_name("features_schema.json")
    meta = json.loads(sch.read_text(encoding="utf-8")) if sch.exists() else {}
    return FeatureSet(take=str(d["take"]), t=d["t"], X=d["X"], valid=d["valid"], quality=d["quality"], run=d["run"],
                      frame=frame, grid_t=d["grid_t"], grid_type=d["grid_type"], phase0=int(d["measure_phase0"]),
                      eighths=eighths, pooled=d["pooled"], pooled_z=d["pooled_z"], z_center=d["z_center"],
                      z_scale=d["z_scale"], meta={**meta, "path": str(p)})


def run(take: str, smplx_dir: str | Path | None = None, cfg: FeatureConfig | None = None,
        out_dir: Path | None = None, work: Path | None = None) -> tuple[FeatureSet, Path, Path]:
    """Compute and save the features of a take; returns (FeatureSet, npz path, schema path)."""
    cfg = cfg or FeatureConfig()
    work = work or work_dir(take)
    d = find_smplx_dir(take, smplx_dir, work)
    npzs = load_npzs(d)
    masses, statures = take_dancers(take)
    grid = load_grid(work)
    cbp = work / "counterbalance" / "counterbalance.json"   # = counterbalance_path(take) for the default work dir
    cb_doc = json.loads(cbp.read_text(encoding="utf-8")) if cbp.exists() else None
    td, tdp = load_touchdowns(work)
    fs = compute_features(npzs, cfg, grid, cb_doc, td, masses, statures, take)
    w = world_frame_warning(d)
    if w:
        fs.meta["warnings"].insert(0, w)
    if cb_doc is not None:
        m = counterbalance_fits_mismatch(cb_doc, d, take)
        if m:
            fs.meta["warnings"].append(m)
    gp = grid_source(work)
    tj = work / "timing" / "timing.json"
    source = {"smplx_dir": _rel(d), "lead_sha256_16": _sha16(d / "lead.npz"),
              "follow_sha256_16": _sha16(d / "follow.npz"),
              "beats": "take grid" if grid is not None else f"none (constant {DEFAULT_BPM} BPM)",
              "beats_file": _rel(gp) if gp is not None else None, "beats_sha256_16": _sha16(gp) if gp else None,
              "timing": _rel(tj) if tj.exists() else None, "timing_sha256_16": _sha16(tj),
              "counterbalance": _rel(cbp) if cb_doc is not None else None,
              "counterbalance_sha256_16": _sha16(cbp) if cb_doc is not None else None,
              "touchdowns": _rel(tdp) if td is not None else None, "com": cfg.com,
              "smplx_dir_explicit": bool(smplx_dir)}
    npz_p, sch_p = save(fs, source, out_dir)
    return fs, npz_p, sch_p


def _rel(p) -> str:
    from ..take import PACKAGE_ROOT
    p = Path(p)
    return str(p.relative_to(PACKAGE_ROOT)).replace("\\", "/") if p.is_relative_to(PACKAGE_ROOT) else str(p)


def _abs(p: str | Path) -> Path:
    from ..take import PACKAGE_ROOT
    p = Path(str(p).replace("\\", "/"))
    return p if p.is_absolute() else PACKAGE_ROOT / p


def counterbalance_fits_mismatch(cb_doc: dict, fits_dir: Path, take: str = "<take>") -> str | None:
    """A warning when counterbalance.json was computed from other fits than `fits_dir` (its source.lead_sha1 vs the
    sha1 of fits_dir/lead.npz; the directory name when the document records no sha1)."""
    src = (cb_doc or {}).get("source") or {}
    fix = f"re-run `python -m dancecap.counterbalance detect --take {take}` so the cb_* columns match the fits"
    sha = src.get("lead_sha1")
    if sha:
        lp = Path(fits_dir) / "lead.npz"
        if lp.exists() and hashlib.sha1(lp.read_bytes()).hexdigest() != sha:
            return (f"counterbalance.json was computed from other fits than {_rel(fits_dir)}/ (lead.npz sha1 "
                    f"{sha[:10]} != {hashlib.sha1(lp.read_bytes()).hexdigest()[:10]}): {fix}")
        return None
    cb_dir = str(src.get("smplx_dir", "")).replace("\\", "/")
    if cb_dir and Path(cb_dir).name != Path(fits_dir).name:
        return f"counterbalance.json was computed from {cb_dir}, the features from {Path(fits_dir).name}/: {fix}"
    return None


def staleness(take: str, work: Path | None = None, npz_path: Path | None = None) -> list[str]:
    """Why features.npz no longer matches its inputs ([] = current): the recorded sha of the fits, counterbalance.json,
    timing/timing.json and the beat-grid file vs the files now on disk, the default fits directory, and whether
    counterbalance.json was computed from the same fits. Re-run `python -m dancecap.moves features <take>` (and
    `proposals`) when this is not empty."""
    work = Path(work) if work else work_dir(take)
    npz_p = Path(npz_path) if npz_path else features_paths(take)[0]
    sch_p = npz_p.with_name("features_schema.json")
    if not npz_p.exists():
        return [f"no {npz_p.name}: run `python -m dancecap.moves features {take}`"]
    if not sch_p.exists():
        return [f"no {sch_p.name} next to {npz_p.name}: the inputs cannot be checked; re-run features"]
    doc = json.loads(sch_p.read_text(encoding="utf-8"))
    src = doc.get("source") or {}
    out: list[str] = []
    rec = (doc.get("files") or {}).get("npz_sha256_16")
    if rec and rec != _sha16(npz_p):
        out.append(f"{npz_p.name} was rewritten after its schema (sha mismatch)")
    fits = _abs(src["smplx_dir"]) if src.get("smplx_dir") else None
    if fits is not None:
        for r in ROLES:
            key = f"{r}_sha256_16"
            if key in src and _sha16(fits / f"{r}.npz") != src[key]:
                out.append(f"fits {src['smplx_dir']}/{r}.npz changed since the features were computed")
        if not src.get("smplx_dir_explicit"):
            try:
                now = find_smplx_dir(take, None, work)
            except FileNotFoundError:
                now = None
            if now is not None and now.resolve() != fits.resolve():
                out.append(f"features use {src['smplx_dir']}/, the default fits are now {_rel(now)}/")
    cbp = work / "counterbalance" / "counterbalance.json"
    if _sha16(cbp) != src.get("counterbalance_sha256_16"):
        out.append("counterbalance.json " + ("appeared" if src.get("counterbalance_sha256_16") is None else
                                             "disappeared" if not cbp.exists() else "changed")
                   + " since the features were computed (cb_* columns)")
    elif cbp.exists() and fits is not None:
        m = counterbalance_fits_mismatch(json.loads(cbp.read_text(encoding="utf-8")), fits, take)
        if m:
            out.append(m)
    if "timing_sha256_16" not in src:
        out.append("features_schema.json predates input tracking (no timing / beat-grid sha): re-run features")
    else:
        tj = work / "timing" / "timing.json"
        if _sha16(tj) != src["timing_sha256_16"]:
            out.append("timing/timing.json changed since the features were computed (beat grid / touchdowns)")
        gp = grid_source(work)
        same_file = gp is not None and src.get("beats_file") and _abs(src["beats_file"]).resolve() == gp.resolve()
        if (gp is None) != (src.get("beats_file") is None) or (gp is not None and not same_file):
            out.append(f"the beat grid now comes from {_rel(gp) if gp else 'nowhere'}, the features used "
                       f"{src.get('beats_file')}")
        elif gp is not None and gp.resolve() != tj.resolve() and _sha16(gp) != src.get("beats_sha256_16"):
            out.append(f"{_rel(gp)} (beat grid) changed since the features were computed")
    return out


# ----------------------------------------------------------------------------------------------- CLI

def cmd_features(args: argparse.Namespace) -> int:
    if args.check:
        st = staleness(args.take)
        for m in st:
            print(f"[moves] STALE {m}")
        print(f"[moves] features {args.take}: " + ("STALE - re-run `python -m dancecap.moves features "
                                                    f"{args.take}` and `proposals {args.take}`" if st else
                                                    "current (fits, counterbalance, timing and beat grid unchanged)"))
        return 1 if st else 0
    cfg = FeatureConfig(com=args.com, min_quality=args.min_quality)
    fs, npz_p, sch_p = run(args.take, args.smplx_dir, cfg)
    nz = int(np.isfinite(fs.pooled[:, :, 0]).all(0).sum())
    vf = eighth_valid_fraction(fs)
    print(f"[moves] wrote {npz_p}: {len(fs.t)} frames [{fs.t[0]:.2f}, {fs.t[-1]:.2f}] s, {len(FEATURES)} features, "
          f"{len(fs.eighths['t0'])} eighths pooled ({nz} features finite on every eighth; "
          f"{int((vf < 0.5).sum())} eighth(s) with < 50 % valid frames)")
    print(f"[moves] schema {sch_p}")
    for w in fs.meta["warnings"]:
        print(f"[moves] WARNING {w}")
    if not args.no_review:
        from .features_review import draw_features
        img = draw_features(args.take)
        print(f"[moves] review image {img}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("features", help="interpretable per-frame couple features -> work/<take>/moves/features.npz "
                                        "(+ features_schema.json, review/moves_features.png)")
    p.add_argument("take")
    p.add_argument("--smplx-dir", help="fits directory (default work/<take>/smplx, else smplx_v1)")
    p.add_argument("--com", choices=("physics", "joints"), default="physics",
                   help="physics = body model mass elements (default); joints = joint positions only (fast)")
    p.add_argument("--min-quality", type=float, default=FeatureConfig.min_quality,
                   help="frames with fit quality <= this are not pooled (0 disables)")
    p.add_argument("--check", action="store_true", help="only compare the recorded input hashes with the files on "
                                                        "disk (exit 1 when stale); computes nothing")
    p.add_argument("--no-review", action="store_true")
    p.set_defaults(func=cmd_features)
