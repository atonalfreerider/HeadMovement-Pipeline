"""Synthetic counterbalance take with ground truth: ordinary dancing, a counterbalance pivot, release, dancing.

    python -m dancecap.counterbalance synthetic     -> work/synthetic_cb/{smplx/{lead,follow}.npz, beats.json,
                                                       truth.json, audio.wav} + Unity capture 03_SyntheticCounterbalance

Two shaped SMPL-X bodies (the betas of synthetic.py), real FK (physics_body), Y-up right-handed world, metres,
floor y = 0 at the soles. Timeline at 76.4 BPM (beat B = 0.785 s, measure M = 4 B = 3.14 s), defaults of Spec:

  [0, 3M)          ordinary dancing: synthetic.body_pose stepping (time-scaled from 88 to 76.4 BPM so a step lands
                   on every beat), the couple rotating CCW about its centre once every 4 measures, elastic partner
                   gap ~0.62 m - as synthetic.make
  [3M, T1)         set-up (1 beat): both lean away, the arms reach to the hold, the follower's left foot settles on the
                   pivot point P and her right foot lifts; the couple's ordinary rotation slows to zero
  [T1, T2]         THE PIVOT (2 measures): the follower's left foot ball (SMPL-X joint 10 'left_foot') fixed at P on
                   the floor; she is a straight plank leaning 14 deg away from the leader about that foot (ankle
                   compensated so the sole stays flat), the free right leg lifted behind and swinging; the leader leans
                   10 deg away with a ONE-HAND hold (her left hand in his right, both arms straight, each turned 45 deg
                   so the holding shoulder leads; free arms lowered) and walks a full 360 deg circle around P (CCW seen
                   from above, constant radius 0.86 m, angular speed eased in / out over 1/4 beat inside [T1, T2]), so
                   she pivots in place facing him
  [T2, T2+B)       release (1 beat): back to ordinary dancing
  [T2+B, end)      ordinary dancing (2 measures)

Spec(hold="two") (both hands: her left - his right, her right - his left; turn_deg forced to 0 so both pairs meet)
reaches only ~0.68 m radius with straight arms. generate() refuses a take whose held hand pairs drift apart (> grip_gap
+ 3 cm) during the pivot.

Ground truth (truth.json, "counterbalance": [one interval]):
  pivot_t0 / pivot_t1   = [T1, T2], the generator's schedule (the leader circles exactly there) - with P, the pivot
                          foot, the leader's swept angle / direction / radius;
  t0 / t1               = the counterbalance (yellow axis) interval: the counterbalance-tier criteria of
                          dancecap.counterbalance (one dancer hangs beyond the edge of their support while the partner
                          does not lean in, AND a connection) at their 0.5 points, evaluated on the EXACT geometry
                          (exact joints, posed-mesh COM, no smoothing, no noise) - the lean starts during the set-up
                          when the follower's free foot leaves the floor, so it precedes the pivot by a few frames.
                          DEFINITIONAL: these edges come from the detector's own lean / connection functions, so the
                          synthetic test checks the detector's smoothing, edges and pivot logic, not whether its
                          definition of a counterbalance is right - only a human label can (the regression test on
                          work/larissa_kadu_38/counterbalance/truth_user.json);
  plus the masses and the per-frame combined centre of mass of the couple (each body's COM from the LBS-posed closed
  SMPL-X mesh with uniform density, physics_body.posed_mesh_com; mass = closed rest-mesh volume x 1010 kg/m^3).
Known limits (truth.json notes): the leader's feet glide along the circle while in contact (no planted stance feet),
and the moment balance about the follower's anchored ball only closes if her centre of pressure sits near mid-foot -
fine for detector tests, not a physically consistent gait.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import physics_body as pb
from . import synthetic as syn
from .physics import _hand_center
from .take import PACKAGE_ROOT, WORKSPACE

BPM = 76.4
BEAT = 60.0 / BPM
MEASURE = 4 * BEAT
TIME_SCALE = BPM / syn.BPM          # synthetic.body_pose(t * TIME_SCALE) steps on the 76.4 BPM beat
TAKE = "synthetic_cb"
CAPTURE = "03_SyntheticCounterbalance"
ROLES = ("lead", "follow")
BETAS = {"lead": np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]),
         "follow": np.array([-0.6, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.0])}   # as synthetic.make
LEG_JOINTS = (1, 2, 4, 5, 7, 8)     # hips, knees, ankles (SMPL-X body joint ids)


@dataclass
class Spec:
    fps: float = 30.0
    pre_measures: float = 3.0
    setup_beats: float = 1.0
    pivot_measures: float = 2.0
    release_beats: float = 1.0
    post_measures: float = 2.0
    ease_beats: float = 0.25          # angular speed ramps inside [T1, T2]
    turns: float = 1.0                # full circles of the leader around the pivot
    hold: str = "one"                 # 'one': her left hand in his right; 'two': both hands
    turn_deg: float = 45.0            # one-hand hold: each turns so the holding shoulder leads
    lean_follow_deg: float = 14.0
    lean_lead_deg: float = 10.0
    hand_height_m: float = 1.3        # joined hands (hand centres) above the floor
    hand_half_width_m: float = 0.17   # two-hand hold: lateral offset of each joined hand pair from the couple axis
    grip_gap_m: float = 0.05          # distance between the two joined hand centres
    gap_m: float = 0.62               # ordinary partner gap (pelvis to pelvis), as synthetic.make

    @property
    def ta(self) -> float:            # set-up starts
        return self.pre_measures * MEASURE

    @property
    def t1(self) -> float:            # counterbalance starts (truth)
        return self.ta + self.setup_beats * BEAT

    @property
    def t2(self) -> float:            # counterbalance ends (truth)
        return self.t1 + self.pivot_measures * MEASURE

    @property
    def tb(self) -> float:            # release done
        return self.t2 + self.release_beats * BEAT

    @property
    def duration(self) -> float:
        return self.tb + self.post_measures * MEASURE


# ----------------------------------------------------------------------------------------------- schedule

def cb_weight(t: np.ndarray, spec: Spec) -> np.ndarray:
    """0 = ordinary dancing, 1 = counterbalance pose (smoothstep set-up / release ramps outside [T1, T2])."""
    t = np.asarray(t, float)
    up = syn._smoothstep((t - spec.ta) / (spec.t1 - spec.ta))
    down = 1.0 - syn._smoothstep((t - spec.t2) / (spec.tb - spec.t2))
    return np.where(t < spec.t2, up, down)


def _circling_rate(t: np.ndarray, spec: Spec) -> np.ndarray:
    """Leader's angular speed about the pivot (rad/s): constant inside [T1, T2] with 1/4-beat eases at both ends,
    scaled so the integral over [T1, T2] is exactly turns x 2 pi (an ease of length e loses e/2 of area)."""
    e = spec.ease_beats * BEAT
    L = spec.t2 - spec.t1
    w = spec.turns * 2 * math.pi / (L - e)
    rise = syn._smoothstep((t - spec.t1) / e)
    fall = 1.0 - syn._smoothstep((t - (spec.t2 - e)) / e)
    return w * np.where((t >= spec.t1) & (t <= spec.t2), rise * fall, 0.0)


def couple_yaw(t: np.ndarray, spec: Spec) -> np.ndarray:
    """Yaw of the couple frame (rad, + = CCW seen from above = right-handed about +Y), integrated on a fine grid:
    the ordinary rotation (one turn per 4 measures) fades out with the set-up, the circling takes over."""
    tf = np.arange(0.0, float(np.max(t)) + 0.01, 1e-4)
    rate = 2 * math.pi / (4 * MEASURE) * (1.0 - cb_weight(tf, spec)) + _circling_rate(tf, spec)
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(tf))])
    return np.interp(t, tf, psi)


# ----------------------------------------------------------------------------------------------- geometry

def _rot_y(a) -> Rot:
    a = np.atleast_1d(np.asarray(a, float))
    return Rot.from_rotvec(np.stack([np.zeros_like(a), a, np.zeros_like(a)], -1))


def _rot_x(a) -> Rot:
    a = np.atleast_1d(np.asarray(a, float))
    return Rot.from_rotvec(np.stack([a, np.zeros_like(a), np.zeros_like(a)], -1))


def _min_rotvec(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Smallest rotation (axis-angle) taking unit vector u to unit vector v."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    ax = np.cross(u, v)
    s, c = np.linalg.norm(ax), float(np.dot(u, v))
    if s < 1e-9:
        return np.zeros(3)
    return ax / s * math.atan2(s, c)


def _fk(body: pb.BodyModel, pose: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """Joint positions (T, 55, 3) for local axis-angle pose (T, 55, 3) and root translation (T, 3)."""
    R, p = pb.forward_kinematics(body, pose, transl)
    return np.einsum("tjab,jb->tja", R, body.j_rest) + p


def _sole_height(body: pb.BodyModel, joint: int = 10) -> float:
    """Height of a foot joint above the sole in the rest pose (the joint height when standing flat on the floor)."""
    return float(body.j_rest[joint, 1] - body.v_rest[:, 1].min())


def _free_leg(pose: np.ndarray, t: np.ndarray) -> None:
    """Follower's free right leg during the pivot: thigh behind and out, knee bent, a slow swing (2 beats)."""
    swing = np.sin(2 * math.pi * t / (2 * BEAT))
    syn._set(pose, 2, 0, 0.30)                     # hip extension (positive x = thigh back)
    syn._set(pose, 2, 2, -0.30 - 0.12 * swing)    # abduction (negative z moves the right leg out)
    syn._set(pose, 5, 0, 0.80)                     # knee flexion
    syn._set(pose, 8, 0, 0.30)


def _lean_turn(role: str, spec: Spec, w: float | np.ndarray = 1.0) -> tuple[Rot, Rot]:
    """(lean, turn) of a dancer in the couple's local frame at counterbalance weight w: global = lean * turn.
    turn: yaw about +Y (follower faces +z, leader -z; with a one-hand hold each turns so the holding shoulder leads);
    lean: away from the partner along the couple axis (follower towards -z, leader towards +z), about the local x."""
    w = np.atleast_1d(np.asarray(w, float))
    if role == "follow":
        a, tau = -math.radians(spec.lean_follow_deg), -math.radians(spec.turn_deg)
        return _rot_x(w * a), _rot_y(w * tau)
    a, tau = math.radians(spec.lean_lead_deg), math.radians(spec.turn_deg)
    return _rot_x(w * a), _rot_y(np.pi + w * tau)


def _flat_ankle(lean: Rot, turn: Rot) -> np.ndarray:
    """Local ankle rotation that keeps the sole flat (foot world rotation = turn) under global = lean * turn."""
    return (turn.inv() * lean.inv() * turn).as_rotvec()[0]


def _arm_dir(body: pb.BodyModel, side: str) -> tuple[np.ndarray, float]:
    """Rest-pose shoulder -> hand-centre vector (unit) and length."""
    sh = 16 if side == "l" else 17
    hc = _hand_center(body.j_rest[None], side)[0] - body.j_rest[sh]
    L = float(np.linalg.norm(hc))
    return hc / L, L


def solve_hold(bodies: dict, spec: Spec) -> dict:
    """Counterbalance geometry in the couple's local frame (pivot at the origin, follower facing +z, leader on +z
    facing -z): shoulder rotations that put the held hands at hand_height with straight arms, and the leader's
    distance D (feet midpoint from the pivot) at which his straight arm(s) just reach hers.

    hold 'one': her left hand in his right (the holding shoulders turned forward by turn_deg), free arms lowered;
    hold 'two': both hands (her left - his right at +x, her right - his left at -x), turn_deg ignored.
    Returns {role: {"global", "arms": {joint: rotvec}, "ankle": rotvec, "transl_loc", ...}, "D", "radius"}."""
    if spec.hold == "two":
        spec = Spec(**{**asdict(spec), "turn_deg": 0.0})
    out: dict = {}
    H = spec.hand_height_m
    # follower: plank about the anchored left foot (ankle compensated: sole flat)
    bf = bodies["follow"]
    lean, turn = _lean_turn("follow", spec)
    Rf = (lean * turn)[0]
    ank_f = _flat_ankle(lean, turn)
    pose = np.zeros((1, 55, 3))
    pose[0, 0] = Rf.as_rotvec()
    pose[0, 7] = ank_f
    J = _fk(bf, pose, np.zeros((1, 3)))[0]
    tl_f = np.array([-J[10, 0], _sole_height(bf) - J[10, 1], -J[10, 2]])
    Jf = J + tl_f
    # leader at D = 0 (feet midpoint on the pivot), then pushed out along +z
    bl = bodies["lead"]
    lean_l, turn_l = _lean_turn("lead", spec)
    Rl = (lean_l * turn_l)[0]
    ank_l = _flat_ankle(lean_l, turn_l)
    pose = np.zeros((1, 55, 3))
    pose[0, 0] = Rl.as_rotvec()
    pose[0, 7] = pose[0, 8] = ank_l
    J = _fk(bl, pose, np.zeros((1, 3)))[0]
    mid = 0.5 * (J[7] + J[8])
    tl0 = np.array([-mid[0], _sole_height(bl) - min(J[10, 1], J[11, 1]), -mid[2]])
    J0 = J + tl0
    pairs = [("r", 17, "l", 16)] if spec.hold == "one" else [("r", 17, "l", 16), ("l", 16, "r", 17)]
    targets, Ds = {}, []
    for ls, lsh, fs, fsh in pairs:          # (leader side, his shoulder, follower side, her shoulder)
        x = 0.5 * (Jf[fsh, 0] + J0[lsh, 0]) if spec.hold == "one" else             (spec.hand_half_width_m if fs == "l" else -spec.hand_half_width_m)
        _, Lf = _arm_dir(bf, fs)
        S = Jf[fsh]
        Tf = np.array([x, H, S[2] + math.sqrt(max(Lf ** 2 - (x - S[0]) ** 2 - (H - S[1]) ** 2, 1e-6))])
        Tl = Tf + np.array([0.0, 0.0, spec.grip_gap_m])
        targets[(ls, fs)] = (Tf, Tl)
        _, Ll = _arm_dir(bl, ls)
        S0 = J0[lsh]
        Ds.append(Tl[2] - S0[2] + math.sqrt(max(Ll ** 2 - (Tl[0] - S0[0]) ** 2 - (Tl[1] - S0[1]) ** 2, 1e-6)))
    D = float(np.mean(Ds))
    tl_l = tl0 + np.array([0.0, 0.0, D])
    Jl = J0 + np.array([0.0, 0.0, D])
    arms_f, arms_l = {}, {}
    for ls, lsh, fs, fsh in pairs:
        Tf, Tl = targets[(ls, fs)]
        u, _ = _arm_dir(bf, fs)
        arms_f[fsh] = _min_rotvec(u, Rf.inv().apply((Tf - Jf[fsh]) / np.linalg.norm(Tf - Jf[fsh])))
        u, _ = _arm_dir(bl, ls)
        arms_l[lsh] = _min_rotvec(u, Rl.inv().apply((Tl - Jl[lsh]) / np.linalg.norm(Tl - Jl[lsh])))
    if spec.hold == "one":                 # free arms lowered ~60 deg out to the side
        arms_f[17] = np.array([0.0, 0.0, math.radians(60)])
        arms_l[16] = np.array([0.0, 0.0, -math.radians(60)])
    out["follow"] = {"global": Rf.as_rotvec(), "arms": arms_f, "ankle": ank_f, "transl_loc": tl_f}
    out["lead"] = {"global": Rl.as_rotvec(), "arms": arms_l, "ankle": ank_l, "transl_loc": tl_l}
    out["D"] = D
    out["radius"] = float(np.linalg.norm(Jl[0, [0, 2]]))
    out["pairs"] = [f"lead_{ls}_hand-follow_{fs}_hand" for ls, _, fs, _ in pairs]
    return out


# ----------------------------------------------------------------------------------------------- motion

def _poses(t: np.ndarray, spec: Spec, bodies: dict, hold: dict) -> dict:
    """Blended SMPL-X parameters per role (before the per-frame floor correction)."""
    psi = couple_yaw(t, spec)
    w = cb_weight(t, spec)
    P = pivot_point(spec, bodies)
    half_gap = 0.5 * (spec.gap_m + 0.12 * np.sin(2 * np.pi * t / MEASURE))
    out = {}
    for role, sign in (("lead", -1.0), ("follow", 1.0)):
        body = bodies[role]
        h = hold[role]
        # ordinary dancing (synthetic.make): pelvis on a circle about the origin, facing the partner
        bp_ord = syn.body_pose(t * TIME_SCALE, role)
        target = np.stack([-sign * half_gap * np.sin(psi), -sign * half_gap * np.cos(psi)], 1)
        tr_ord = np.zeros((len(t), 3))
        tr_ord[:, 0] = target[:, 0] - body.j_rest[0, 0]
        tr_ord[:, 2] = target[:, 1] - body.j_rest[0, 2]
        # counterbalance pose
        bp_cb = np.zeros((len(t), 63))
        if role == "follow":
            _free_leg(bp_cb, t)
            bp_cb[:, 18:21] = h["ankle"]                  # left ankle (joint 7): sole flat under the lean
        else:
            steps = syn.body_pose(t * TIME_SCALE, "lead")
            for j in LEG_JOINTS:
                bp_cb[:, (j - 1) * 3:(j - 1) * 3 + 3] = steps[:, (j - 1) * 3:(j - 1) * 3 + 3]
            bp_cb[:, 18:24] += np.tile(h["ankle"], 2)     # both ankles (joints 7, 8)
        for sh, rv in h["arms"].items():
            bp_cb[:, (sh - 1) * 3:(sh - 1) * 3 + 3] = rv
        # the local configuration rotated by psi about the vertical through P:
        # posed(x) = Ry(psi) R_loc (x - J0) + Ry(psi) (J0 + transl_loc) - J0 + transl ... with transl as below
        J0 = body.j_rest[0]
        tr_cb = _rot_y(psi).apply(J0 + h["transl_loc"]) - J0 + np.array([P[0], 0.0, P[2]])
        # blend: couple yaw psi, then the dancer's lean and turn scaled by w (w = 0: facing the partner, upright)
        lean, turn = _lean_turn(role, spec, w)
        go = (_rot_y(psi) * lean * turn).as_rotvec()
        bp = (1 - w)[:, None] * bp_ord + w[:, None] * bp_cb
        tr = (1 - w)[:, None] * tr_ord + w[:, None] * tr_cb
        out[role] = {"global_orient": go, "body_pose": bp, "transl": tr}
    return out


def pivot_point(spec: Spec, bodies: dict) -> np.ndarray:
    """P on the floor: where the follower's ordinary-dancing pelvis is at T1 (so the set-up barely travels)."""
    psi = float(couple_yaw(np.array([spec.t1]), spec)[0])
    hg = 0.5 * (spec.gap_m + 0.12 * math.sin(2 * math.pi * spec.t1 / MEASURE))
    return np.array([-hg * math.sin(psi), 0.0, -hg * math.cos(psi)])


def load_bodies() -> dict:
    return {r: pb.load_body({"betas": BETAS[r], "gender": "neutral"}) for r in ROLES}


def generate(spec: Spec | None = None) -> dict:
    """In-memory take: {"t", role: {params, joints, R, p}, "bodies", "hold", "P"}; feet on the floor per frame."""
    spec = spec or Spec()
    if spec.hold == "two" and spec.turn_deg:
        spec = replace(spec, turn_deg=0.0)      # both pairs only meet when the dancers face each other squarely
    bodies = load_bodies()
    hold = solve_hold(bodies, spec)
    t = np.arange(int(round(spec.duration * spec.fps))) / spec.fps
    params = _poses(t, spec, bodies, hold)
    out = {"t": t, "bodies": bodies, "hold": hold, "P": pivot_point(spec, bodies), "spec": spec}
    for role in ROLES:
        b = bodies[role]
        pr = params[role]
        pose = np.concatenate([pr["global_orient"], pr["body_pose"], np.zeros((len(t), 3 * 33))], 1).reshape(-1, 55, 3)
        R, p = pb.forward_kinematics(b, pose, pr["transl"])
        J = np.einsum("tjab,jb->tja", R, b.j_rest) + p
        # floor: the lower foot (ball joint) at its standing height -> the stance sole touches y = 0
        dy = _sole_height(b) - np.minimum(J[:, 10, 1], J[:, 11, 1])
        pr["transl"] = pr["transl"] + np.stack([np.zeros_like(dy), dy, np.zeros_like(dy)], 1)
        J[..., 1] += dy[:, None]
        p[..., 1] += dy[:, None]
        out[role] = {**pr, "joints": J, "R": R, "p": p}
    k = (t >= spec.t1) & (t <= spec.t2)
    for pair in hold["pairs"]:
        ls, fs = pair[5], pair.split("-")[1][7]
        d = np.linalg.norm(_hand_center(out["lead"]["joints"][k], ls) - _hand_center(out["follow"]["joints"][k], fs),
                           axis=1)
        if d.max() > spec.grip_gap_m + 0.03:
            raise ValueError(f"synthetic_counterbalance: held pair {pair} is {d.max():.3f} m apart during the pivot "
                             f"(grip gap {spec.grip_gap_m} m) - the hold geometry does not close")
    return out


def counterbalance_truth(g: dict, coms: dict) -> dict | None:
    """The counterbalance (yellow axis) interval on the exact geometry: dancecap.counterbalance's counterbalance-tier
    criteria (lean_score: one dancer hangs beyond their support, the partner not leaning in; connection_scores: a hand
    pair joined with straight arms or a hand on the partner's body) at their 0.5 points, from the exact joints and the
    posed-mesh COMs, no smoothing - the run around the middle of the pivot; None when there is none. Definitional (the
    detector's own criteria): it cannot catch a wrong definition, only the human label can."""
    from . import counterbalance as cb
    cfg = cb.CBConfig()
    spec: Spec = g["spec"]
    t = g["t"]
    J = {r: g[r]["joints"] for r in ROLES}
    contact, mid = cb.foot_contacts(J, cfg)
    disp = cb.lean_displacement(coms, contact, mid)
    over, sup = cb.support_overhang(coms, contact, J, cfg)
    _, ps, _ = cb.connection_scores(J, cfg)
    s = np.minimum(cb.lean_score(disp, cfg, over, sup), ps.max(1))
    i = int(np.argmin(np.abs(t - 0.5 * (spec.t1 + spec.t2))))
    if s[i] < cfg.on:
        return None
    i0, i1 = i, i
    while i0 > 0 and s[i0 - 1] >= cfg.on:
        i0 -= 1
    while i1 < len(t) - 1 and s[i1 + 1] >= cfg.on:
        i1 += 1
    t0, t1 = cb.level_edges(t, s, cfg.on, i0, i1)
    k = (t >= spec.t1) & (t <= spec.t2)
    return {"t0": t0, "t1": t1, "min_score_in_pivot": float(s[k].min()),
            "lean_m_median_in_pivot": {r: round(float(np.median(disp[r][k])), 4) for r in ROLES},
            "overhang_m_median_in_pivot": {r: round(float(np.median(over[r][k])), 4) for r in ROLES}}


def truth_for(g: dict) -> dict:
    """Ground truth of a generated take (see the module docstring)."""
    spec: Spec = g["spec"]
    t = g["t"]
    P = g["P"]
    coms, masses = {}, {}
    for role in ROLES:
        b = g["bodies"][role]
        coms[role] = pb.posed_mesh_com(b, g[role]["R"], g[role]["p"])
        masses[role] = b.volume * pb.DENSITY
    M = masses["lead"] + masses["follow"]
    com = (masses["lead"] * coms["lead"] + masses["follow"] * coms["follow"]) / M
    k = (t >= spec.t1) & (t <= spec.t2)
    anchor = g["follow"]["joints"][:, 10]
    drift = float(np.linalg.norm(anchor[k][:, [0, 2]] - P[[0, 2]], axis=1).max())
    lp = g["lead"]["joints"][:, 0]
    ang = np.unwrap(np.arctan2(-(lp[:, 2] - P[2]), lp[:, 0] - P[0]))
    psi = couple_yaw(np.array([spec.t1, spec.t2]), spec)
    swept_exact = math.degrees(float(psi[1] - psi[0]))
    radius = np.linalg.norm(lp[k][:, [0, 2]] - P[[0, 2]], axis=1)
    hands = {}
    for pair in g["hold"]["pairs"]:
        ls, fs = pair[5], pair.split("-")[1][7]
        d = np.linalg.norm(_hand_center(g["lead"]["joints"], ls) - _hand_center(g["follow"]["joints"], fs), axis=1)
        hands[pair] = round(float(np.median(d[k])), 4)
    free = g["follow"]["joints"][:, [8, 11], 1].min(1) - g["follow"]["joints"][:, 10, 1]
    # leader's feet: horizontal speed of a ball joint while it is within 2 cm of its standing height (gliding check)
    Jl = g["lead"]["joints"]
    hb = _sole_height(g["bodies"]["lead"])
    v = np.linalg.norm(np.gradient(Jl[:, [10, 11]][:, :, [0, 2]], t, axis=0), axis=2)
    on_floor = (Jl[:, [10, 11], 1] - hb) < 0.02
    glide = float(np.percentile(v[k][on_floor[k]], 90)) if on_floor[k].any() else 0.0
    cbt = counterbalance_truth(g, coms)
    pivot = spec.turns != 0
    iv = {
        "kind": "pivot" if pivot else "counterbalance",
        "t0": round(cbt["t0"], 6) if cbt else None, "t1": round(cbt["t1"], 6) if cbt else None,
        "definition": "t0/t1 = counterbalance (yellow axis): dancecap.counterbalance counterbalance-tier criteria "
                      "(both lean away >= 8 cm from their own feet, a hand pair <= 14 cm apart with arm extension "
                      ">= 0.91) on the exact joints + posed-mesh COM; pivot_t0/pivot_t1 = the generator's circling "
                      "schedule [T1, T2]",
        "counterbalance_check": cbt,
        "pivot_t0": round(spec.t1, 6) if pivot else None, "pivot_t1": round(spec.t2, 6) if pivot else None,
        "setup": [round(spec.ta, 6), round(spec.t1, 6)], "release": [round(spec.t2, 6), round(spec.tb, 6)],
        "pivot": [round(float(P[0]), 5), 0.0, round(float(P[2]), 5)] if pivot else None,
        "pivot_point": "follower left foot ball = SMPL-X joint 10 'left_foot' (horizontal position)",
        "pivot_foot": "left", "pivot_drift_max_m": round(drift, 6),
        "swept_deg": round(swept_exact, 4),
        "swept_deg_frames": round(math.degrees(float(ang[k][-1] - ang[k][0])), 3),
        "direction": "ccw" if swept_exact > 0 else "cw",
        "leader_radius_m": round(float(np.median(radius)), 4),
        "leader_radius_range_m": [round(float(radius.min()), 4), round(float(radius.max()), 4)],
        "lean_deg": {"follow": spec.lean_follow_deg, "lead": spec.lean_lead_deg},
        "connection": f"{spec.hold}-hand hold, straight arms: " + ", ".join(f"{k_} {v_} m" for k_, v_ in hands.items()),
        "hand_pairs": hands,
        "free_foot_clearance_min_m": round(float(free[k].min()), 4),
        "leader_foot_glide_p90_m_s": round(glide, 3),
    }
    return {
        "bpm": BPM, "beat_s": BEAT, "measure_s": MEASURE, "fps": spec.fps, "duration": float(t[-1] + 1 / spec.fps),
        "spec": asdict(spec),
        "counterbalance": [iv] if cbt else [],
        "masses_kg": {r: round(m, 3) for r, m in masses.items()},
        "mass_source": f"closed rest-mesh volume x {pb.DENSITY:.0f} kg/m^3 (physics_body default)",
        "com_method": "LBS-posed closed SMPL-X mesh, uniform density, divergence theorem (physics_body.posed_mesh_com)",
        "com": {"t": [round(float(x), 6) for x in t],
                "combined": np.round(com, 5).tolist(),
                "lead": np.round(coms["lead"], 5).tolist(), "follow": np.round(coms["follow"], 5).tolist()},
        "touchdowns": {},
        "notes": [
            "World: right-handed Y-up metres, floor y=0 at the soles. direction is seen from above (ccw = positive "
            "rotation about +Y). Ordinary dancing outside [setup0, release1]: no counterbalance.",
            "counterbalance[0].pivot_t0 / pivot_t1 = [T1, T2]: anchored foot + the leader circling (the generator's "
            "schedule, independent of any detector threshold).",
            "counterbalance[0].t0 / t1 = the counterbalance (yellow axis): a CONVENTION - the detector's documented "
            "counterbalance-tier criteria at their 0.5 points evaluated on the exact geometry. The lean ramps in "
            "during the set-up beat and the follower leans beyond her support once her free foot leaves the floor, "
            "so t0 precedes pivot_t0 (and t1 follows pivot_t1) by a few frames.",
            "Known limits: the leader's feet glide along the circle while in contact (leader_foot_glide_p90_m_s; "
            "no planted stance feet), and the moment balance about the follower's anchored ball only closes if her "
            "centre of pressure sits near mid-foot. Fine for detector tests; not a physically consistent gait.",
            "touchdowns are not modelled for this take (empty).",
        ],
    }


def make(out_dir: Path | None = None, spec: Spec | None = None) -> Path:
    """Write work/synthetic_cb/ (smplx npz contract files, beats.json, truth.json, audio.wav)."""
    spec = spec or Spec()
    out_dir = Path(out_dir) if out_dir is not None else PACKAGE_ROOT / "work" / TAKE
    (out_dir / "smplx").mkdir(parents=True, exist_ok=True)
    g = generate(spec)
    t = g["t"]
    T = len(t)
    for role in ROLES:
        r = g[role]
        np.savez_compressed(
            out_dir / "smplx" / f"{role}.npz", t=t, global_orient=r["global_orient"], body_pose=r["body_pose"],
            left_hand_pose=np.zeros((T, 45)), right_hand_pose=np.zeros((T, 45)), jaw_pose=np.zeros((T, 3)),
            transl=r["transl"], betas=BETAS[role], joints=r["joints"], gender="neutral", quality=np.ones(T),
            source="dancecap synthetic_counterbalance")
    truth = truth_for(g)
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1))
    sys.path.insert(0, str(WORKSPACE / "atlas_bridge"))
    from atlas_bridge.beats import zouk_grid  # noqa: E402  (shared beat format)
    (out_dir / "beats.json").write_text(json.dumps(zouk_grid(BPM, spec.duration + 1.0), indent=1))
    try:
        import atlas_bridge.demo as demo
        old = demo.BPM
        demo.BPM = BPM
        try:
            demo._write_groove(out_dir / "audio.wav", spec.duration + 1.0)
        finally:
            demo.BPM = old
    except Exception as e:  # noqa: BLE001  (audio is a convenience for the Unity capture)
        print(f"[synthetic_cb] audio skipped: {e}")
    return out_dir


def export_capture(capture: str = CAPTURE, install: bool = True) -> dict:
    """Unity capture of work/synthetic_cb via export_unity.export (unchanged)."""
    from . import export_unity
    return export_unity.export(TAKE, capture=capture, install=install)
