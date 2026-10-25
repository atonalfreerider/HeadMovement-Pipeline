"""Magnetic contact points: the leader's hands snap to standardized sites on the follower (the owner, 2026-10-25).

Owner (2026-10-25): "we need sticky standardized contact points: leader right/left hand on follower: upper left/right shoulder, upper
left/right back, forearm wrap around mid back and holding right/left mid back, top of shoulders, back of left/right tricep, left/right
elbow, left/right forearm, left/right hand. the hand should have magnetism to each of these points and gracefully snap to each one. use
the original 3d pose data and then magnetically snap. the transition in and out of a snap should be linear"

Pure numpy, deterministic, seconds. The ORIGINAL pose = the detected input fit (work/<take>/smplx_rf2_noslip); nothing else is read.

    python -m dancecap.magnet sites  --take larissa_kadu_recap  # her 16 sites on her shaped rest mesh -> magnet/sites.json, sites_rest.png
    python -m dancecap.magnet snap   --take larissa_kadu_recap  # sticky timeline, linear weights, two-bone IK -> magnet/timeline.json,
                                                                #   magnet/smplx_magnet/{lead,follow}.npz, magnet/measure.json
    python -m dancecap.magnet export --take larissa_kadu_recap  # side capture -> magnet/unity_motion/ (4 files, unity_motion.json with
                                                                #   sha256 + proof, install_motion.sh); never installs
    python -m dancecap.magnet sheets --take larissa_kadu_recap  # 6-phone sheets at film 0:32 0:57 1:30 2:30 2:43 4:50 -> magnet/sheets/

SITES (her left and right of each): a point + outward normal in the frame of the joint that carries it, found once on her shaped rest
mesh by a ray cast (zouk_contacts._ray_hit). The torso sites have one entry; the limb sites (tricep: the back half, elbow and forearm: all
round) a ring of entries; her hand a palm face and a back face (from her posed hand joints). An attachment keeps ONE entry (the one nearest
his palm most often during it).

SNAP, per frame and each of his hands, on the original pose: distance = his palm centre to the site's target (site point + his palm
thickness along its normal). Engage the nearest site within R_in for T_on; stay while within R_out; switch to another site only when it is
nearer by `margin` (and within R_in) for T_on; release beyond R_out for T_off; a minimum dwell before a switch or release; a site held by
his other hand is not a candidate.

WEIGHTS, linear in time: w ramps 0 -> 1 over T_blend from the engage frame and 1 -> 0 over T_blend before the release frame (the first
frame beyond R_out); a switch cross-fades the old and the new target over T_blend (their weights sum to 1). Palm target = lerp(original
palm, site target, w); palm orientation = the original wrist rotation turned toward "palm facing the site" (palm normal = -site normal,
at most turn_max_deg) by the same w.

ARM: analytic two-bone IK (shoulder -> elbow -> wrist): the elbow on the IK circle nearest the ORIGINAL elbow, the elbow's hinge normal
carried over (a pure change of flexion); out of reach (flexion outside [flex_min, flex_max], or the original's if outside), w is scaled
down until it is reachable. Only his shoulders, elbows and wrists (16-21) change; collar, spine, everything else of his, all of her
motion and every finger rotation stay exactly the original. No collision terms.

V2 (the 2026-10-25 check): STRENGTH per hand and site (MagnetCfg.strength_right / strength_left; effective distance = distance /
strength in every rule; a real-distance cap per hand reach_max_m); stand-alone attachments < min_hold_s dropped; the site target clamped
in speed and held through her tracking glitches (clamp_target); the out-of-reach scale smoothed in time; the elbow chosen with continuity
and damped (elbow_damp_s: the elbow's 3-D change vs the original smoothed, x w).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HZ = 30.0
ROLES = ("lead", "follow")
SIDES = ("left", "right")
SX = {"left": "L", "right": "R"}
PELVIS, SPINE1, SPINE2, SPINE3, NECK = 0, 3, 6, 9, 12
COLLAR = {"left": 13, "right": 14}
SHOULDER = {"left": 16, "right": 17}
ELBOW = {"left": 18, "right": 19}
WRIST = {"left": 20, "right": 21}
ARM_JOINTS = (16, 17, 18, 19, 20, 21)
HINGE_REST = {"left": np.array([0.0, -1.0, 0.0]), "right": np.array([0.0, 1.0, 0.0])}   # elbow flexion axis in the rest upper arm

FILM_CUT_S, FILM_SHIFT_S = 246.459, 7.2            # film = dance until the cut, then dance + 7.2 s (the recap's replay)
SHEET_FILM_S = (32.0, 40.0, 50.0, 57.0, 90.0, 150.0, 163.0, 290.0)
CAPTURE = "06_LarissaKaduRecap"
SIDE_CAPTURE = CAPTURE + "_magnet"
MOTION_FILES = ("smplx_lead.bin", "smplx_follow.bin", "figure1.json", "figure2.json")
INPUT_SHA = {"smplx_lead.bin": "86e6780b174085b1165aa51884f2c2b5631ec010303398df59457552d0faf34a",
             "smplx_follow.bin": "986577451dc1bb2667e8b650351349d211664d88d556c4be61f3e0e536e1e6dc"}
V44B_SHA = {"smplx_lead.bin": "dd0286c31e842f4bf1a79cc4c8f37402cc591db0483d495c4308edd2d038e817",
            "smplx_follow.bin": "f41d798369d13f7a02701c13eeb37f4f9c01c2ecfa059e80a78ccccc8d70b749"}
SITE_KINDS = ("upper_shoulder", "upper_back", "mid_back", "shoulder_top", "tricep", "elbow", "forearm", "hand")
SITE_BGR = {"upper_shoulder": (0, 200, 255), "upper_back": (60, 220, 60), "mid_back": (255, 160, 40), "shoulder_top": (40, 255, 255),
            "tricep": (255, 80, 200), "elbow": (200, 120, 255), "forearm": (255, 255, 120), "hand": (80, 80, 255)}


def log(*a) -> None:
    print("[magnet]", *a, flush=True)


@dataclass
class MagnetCfg:
    # tuned once on the recap (2026-10-25; starting values in brackets): the starting values switched his right hand between her left hand
    # and forearm every few seconds (148 attachments, elbow swings 9 > the original's 6, 1-5 Hz wobble 1.28 cm vs 0.71)
    r_in: float = 0.10            # m: engage within this ... [0.12]
    r_out: float = 0.22           # ... stay attached while within this
    t_on: float = 0.30            # s: within R_in this long to engage (or nearer by `margin` this long to switch) [0.15]
    t_off: float = 0.20           # s: beyond R_out this long to release
    dwell: float = 1.00           # s: minimum attachment before a switch / release [0.4]
    margin: float = 0.10          # m: a switch needs the other site nearer by this [0.04]
    dwell_pull_m: float = 0.05    # m: within the dwell a release still happens beyond r_out + this (the pull never grows past it)
    t_blend: float = 0.50         # s: the linear ramps in / out and the cross-fade of a switch [0.3]
    palm_m: float = 0.015         # his palm centre this far off the site's surface along its normal
    her_palm_m: float = 0.012     # her hand: palm face = her palm centre + this along her palm normal ...
    her_back_m: float = 0.010     # ... back face = her palm centre - this
    flex_min_deg: float = 8.0     # the elbow never straighter than this (or the original, when straighter): no hyperextension
    flex_max_deg: float = 150.0
    turn_max_deg: float = 90.0    # the palm turns at most this far toward facing the site
    # v2 (2026-10-25 check): MAGNET STRENGTH per hand and site, effective distance = distance / strength in every engage / stay / switch
    # rule (a site missing from the table: 1.0); a real-distance cap so a strong site never pulls the hand further than reach_max_m
    strength_right: dict = field(default_factory=lambda: dict(STRENGTH_RIGHT))
    strength_left: dict = field(default_factory=lambda: dict(STRENGTH_LEFT))
    # m, per hand: beyond this real distance a site is no candidate and an attachment counts as beyond r_out (his right hand reaches
    # round her to her back, ~35 cm in the closed hold; his left hand lets go of her hand by 25 cm: 0:54-0:57 the phones show it free)
    reach_max_m: dict = field(default_factory=lambda: {"right": 0.40, "left": 0.25})
    # TARGET SPEED CLAMP: the snapped site target moves at most max(his original palm's step + target_extra_mps / HZ, target_floor_m)
    # per frame; a site whose offset to his original palm changes by > jump_m within jump_s holds that offset until it settles
    target_extra_mps: float = 0.6
    target_floor_m: float = 0.03
    jump_m: float = 0.20
    jump_s: float = 0.20
    min_hold_s: float = 1.5       # s: an engage-to-release attachment shorter than this is dropped (drop_blips)
    scale_smooth_s: float = 0.25  # s: the out-of-reach scale of w, smoothed in time (min filter 2x this, box mean 1x, never above reach)
    elbow_damp_s: float = 0.6     # s (0 = off): inside a snap the elbow's change vs the original (3-D) is smoothed (Gaussian sigma, x w)
                                  # and the elbow turned about the shoulder-wrist line to the circle point nearest it: it swings like the original's
    # v3 (the owner 2026-10-25 on film 1:32: "the wrist should not bend in like that"): the WRIST RULE (wrist_rule). (1) both his wrists
    # clamped to these limits everywhere (flexion / extension / radial / ulnar deviation of the swing, the twist about the forearm), the
    # correction ramped linearly over wrist_ramp_s in and out of each clamped span; (2) a SUPPORTING wrist (by the weight of an attachment
    # to a back / shoulder site, and inside her counterbalance intervals): flexion / extension within support_flex_deg, deviation within
    # support_dev_deg, the palm turned onto her by the forearm twist; support_elbow_share of a twist change rolls the forearm at the
    # elbow (about the bone: no joint moves). The shoulders and the elbow / wrist positions stay as the magnet put them.
    wrist_rule: bool = True
    wrist_flex_max_deg: float = 60.0
    wrist_ext_max_deg: float = 60.0
    wrist_radial_max_deg: float = 25.0
    wrist_ulnar_max_deg: float = 35.0
    wrist_twist_max_deg: float = 90.0     # zouk_contacts.WRIST_LIMITS["twist"]
    wrist_ramp_s: float = 0.30
    support_flex_deg: float = 15.0        # flexion (bending IN, the palm toward the forearm) of a supporting wrist at most this ...
    support_ext_deg: float = 70.0         # ... but EXTENSION (bending back, a palm pushing on a wall) up to this: the palm lies flat on her
                                          # (v5, the owner 2026-10-25: a supporting hand carries her weight on its palm)
    support_dev_deg: float = 15.0
    # v5 SURFACE NUDGE: in supporting frames where his palm skin (palm centre + palm_m along the palm normal) or his wrist joint is more
    # than surface_in_m inside her (her shaped posed mesh: torso, collars, upper arms; nearest vertex, its normal), his elbow flexion is
    # changed about its own hinge axis (the shoulder and so the elbow joint stay put: same elbow side and plane) by the smallest angle
    # that puts the palm skin surface_target_m in (within [0, surface_in_m]) with the wrist outside; bending (shorter reach) wins ties;
    # one sign per span, linear ramps of surface_ramp_s, then the wrist rule re-aims the palm onto her local surface
    surface_nudge: bool = False
    surface_in_m: float = 0.010
    surface_target_m: float = 0.005
    surface_gap_m: float = 0.030
    surface_ramp_s: float = 0.40
    surface_max_deg: float = 45.0
    support_cb_ramp_s: float = 0.50       # the counterbalance weight ramps 0 -> 1 over this before t0 (1 -> 0 after t1)
    support_cb_extra_m: float = 0.10      # inside a counterbalance a free hand supports when the nearest back site is within reach_max_m (+ a
                                          # linear fade over this)
    support_elbow_share: float = 0.0     # 0: the whole twist in the wrist joint (SMPL-X carries pronation / supination there; zouk_contacts
                                          # wrist_angles "the forearm's twist"); > 0 rolls that share at the elbow (unwrapped in time: it can reach
                                          # 180 deg and push the wrist's own twist past its limit, 2026-10-25 check)


# the owner (2026-10-25 check): his right hand mostly rests on her upper-left back; his left hand holds her right hand
STRENGTH_RIGHT = {"upper_back_L": 3.0, "upper_shoulder_L": 3.0, "mid_back_L": 2.0, "mid_back_R": 2.0, "shoulder_top_L": 1.5,
                  **{f"{k}_{x}": 0.8 for k in ("tricep", "elbow", "forearm", "hand") for x in "LR"}}
STRENGTH_LEFT = {"hand_R": 2.5, "forearm_R": 1.0, "elbow_R": 1.0,
                 **{f"{k}_{x}": 1.2 for k in ("upper_shoulder", "upper_back", "mid_back") for x in "LR"}}


def strength_vector(cfg: MagnetCfg, side: str, names: list) -> np.ndarray:
    tab = cfg.strength_right if side == "right" else cfg.strength_left
    return np.array([float(tab.get(nm, 1.0)) for nm in names])


def effective_distances(D: np.ndarray, strength: np.ndarray, cfg: MagnetCfg, side: str = "right") -> np.ndarray:
    """(T,S) real distances -> distance / strength; beyond reach_max_m real distance: just past r_out + dwell_pull_m (never a
    candidate, and an attachment there is released after t_off even within the dwell)"""
    E = np.asarray(D, float) / strength[None]
    return np.where(np.asarray(D) > cfg.reach_max_m[side], np.maximum(E, cfg.r_out + cfg.dwell_pull_m + 1e-3), E)


def film_of(t: float) -> float:
    return float(t) if t < FILM_CUT_S else float(t) + FILM_SHIFT_S


def dance_of(f: float) -> float:
    return float(f) if f < FILM_CUT_S else float(f) - FILM_SHIFT_S


def mmss(f: float) -> str:
    return f"{int(f // 60)}:{f % 60:04.1f}"


# the base fit the magnets start from (work/<take>/<BASE_FIT>): the detected input, or a copy of it with whole-body
# fixes the owner kept (e.g. the recap's 3:19 spin splice); set by the CLI's --base
BASE_FIT = "smplx_rf2_noslip"


def paths(take: str) -> dict:
    from .take import PACKAGE_ROOT
    work = PACKAGE_ROOT / "work" / take
    out = work / "magnet"
    return {"root": PACKAGE_ROOT, "work": work, "out": out, "src_fit": work / BASE_FIT, "fit": out / "smplx_magnet",
            "unity": out / "unity_motion", "input_export": work / "v3" / "unity_motion" / "backup_installed_motion",
            "v44b_fit": work / "smplx_sides", "sheets": out / "sheets", "lock": PACKAGE_ROOT / "work" / "unity.lock"}


def write_json(p: Path, obj) -> Path:
    from .hand_contact import py_json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(py_json(obj), indent=1), encoding="utf-8")
    return p


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ small rotation helpers

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def rot_axis_angle(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """(n,3) unit axes, (n,) angles -> (n,3,3) (Rodrigues)"""
    axis = np.asarray(axis, float).reshape(-1, 3)
    ang = np.asarray(ang, float).reshape(-1)
    K = np.zeros((len(axis), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = axis[:, 2], -axis[:, 1], axis[:, 0]
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def rot_between(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """the smallest rotation taking unit a onto unit b (n,3): (axis (n,3), angle (n,)); opposite vectors turn about a perpendicular"""
    a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    c = np.cross(a, b)
    s = np.linalg.norm(c, axis=1)
    ang = np.arctan2(s, (a * b).sum(1))
    perp = np.cross(a, np.array([1.0, 0, 0]))
    perp = np.where(np.linalg.norm(perp, axis=1, keepdims=True) < 1e-6, np.cross(a, np.array([0, 1.0, 0])), perp)
    axis = np.where(s[:, None] > 1e-9, c / np.maximum(s, 1e-12)[:, None], _unit(perp))
    return axis, ang


def frame_of(a: np.ndarray, n: np.ndarray) -> np.ndarray:
    """(n,3,3) orthonormal columns: a, n made perpendicular to a, a x n"""
    a = _unit(a)
    n = _unit(n - (n * a).sum(-1, keepdims=True) * a)
    return np.stack([a, n, np.cross(a, n)], -1)


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Ro
    return Ro.from_matrix(np.asarray(R, float).reshape(-1, 3, 3)).as_rotvec().reshape(np.shape(R)[:-2] + (3,))


# ------------------------------------------------------------------------------------------------ 1. sites on her rest mesh

def _surface_specs(J: np.ndarray) -> list:
    """(name, kind, side, joint, anchor (rest, inside her), ray direction) of the one-entry torso sites"""
    out = []
    for s in SIDES:
        x, sg, sh = SX[s], (1.0 if s == "left" else -1.0), J[SHOULDER[s]]
        out += [
            # the back-top of the shoulder (trapezius / back of the acromion), from behind and above
            (f"upper_shoulder_{x}", "upper_shoulder", s, COLLAR[s], np.array([sh[0] * 0.8, sh[1], sh[2]]), _unit(np.array([0, -0.6, 0.8]))),
            # the shoulder blade: 10 cm off the spine, 6 cm above spine 3 (zouk_contacts shoulder_blade)
            (f"upper_back_{x}", "upper_back", s, SPINE3, np.array([sg * 0.10, J[SPINE3, 1] + 0.06, J[SPINE3, 2]]), np.array([0, 0, 1.0])),
            # the mid back, 11 cm off the spine at spine 2 (the forearm wrapped round her, the palm on her left or right mid back)
            (f"mid_back_{x}", "mid_back", s, SPINE2, np.array([sg * 0.11, J[SPINE2, 1] + 0.02, J[SPINE2, 2]]), np.array([0, 0, 1.0])),
            # the top of the shoulder, from above (zouk_contacts shoulder_top)
            (f"shoulder_top_{x}", "shoulder_top", s, COLLAR[s], np.array([sh[0] * 0.75, sh[1], sh[2]]), np.array([0, -1.0, 0])),
        ]
    return out


def _limb_specs() -> list:
    """(name, kind, side, carrying joint, (joint a, joint b, fraction), ring angles deg from her back (-z rest))"""
    out = []
    full = tuple(range(0, 360, 30))
    for s in SIDES:
        x = SX[s]
        out += [(f"tricep_{x}", "tricep", s, SHOULDER[s], (SHOULDER[s], ELBOW[s], 0.5), (-60, -30, 0, 30, 60)),
                (f"elbow_{x}", "elbow", s, ELBOW[s], (SHOULDER[s], ELBOW[s], 1.0), full),
                (f"forearm_{x}", "forearm", s, ELBOW[s], (ELBOW[s], WRIST[s], 0.5), full)]
    return out


def fit_sites(betas: np.ndarray) -> dict:
    """{name: {"kind", "side", "joint", "entries": [{"p" (offset from the joint, rest frame), "n" (outward normal), "rest_point"}],
    "default"}} on the shaped rest mesh (rest frame: x = her left, y up, z front); her hand sites carry no entries (posed hand joints)"""
    from . import zouk_contacts as zc
    rb = zc.rest_body(np.asarray(betas, float))
    V, J, F = rb["V"], rb["J"], rb["faces"]
    out = {}
    for name, kind, s, j, anchor, d in _surface_specs(J):
        P, n = zc._ray_hit(V, F, anchor - 0.5 * d, d)
        if P is None:
            raise RuntimeError(f"site {name}: the ray missed her rest mesh")
        out[name] = {"kind": kind, "side": s, "joint": int(j), "default": 0,
                     "entries": [{"p": (P - J[j]).tolist(), "n": n.tolist(), "rest_point": P.tolist()}]}
    for name, kind, s, j, (ja, jb, f), angs in _limb_specs():
        a = J[ja] + f * (J[jb] - J[ja])
        b = _unit(J[ELBOW[s]] - J[SHOULDER[s]]) if f >= 1.0 else _unit(J[jb] - J[ja])
        back = np.array([0, 0, -1.0])
        back = _unit(back - (back @ b) * b)
        up = np.cross(b, back)
        up = up if up[1] >= 0 else -up
        ents = []
        for ang in angs:
            u = np.cos(np.radians(ang)) * back + np.sin(np.radians(ang)) * up
            P, n = zc._ray_hit(V, F, a + 0.25 * u, -u)
            if P is None or not (0.01 < np.linalg.norm(P - a) < 0.10):
                continue
            ents.append({"p": (P - J[j]).tolist(), "n": n.tolist(), "rest_point": P.tolist(), "ring_deg": int(ang)})
        if not ents:
            raise RuntimeError(f"site {name}: no ring ray hit her rest mesh")
        dflt = int(np.argmin([abs(e["ring_deg"] if e["ring_deg"] <= 180 else e["ring_deg"] - 360) for e in ents]))
        out[name] = {"kind": kind, "side": s, "joint": int(j), "default": dflt, "axis_point_rest": a.tolist(), "entries": ents}
    for s in SIDES:
        out[f"hand_{SX[s]}"] = {"kind": "hand", "side": s, "joint": int(WRIST[s]), "default": 0, "entries": [],
                                "faces": ["palm (palm to palm)", "back of her hand"]}
    order = [f"{k}_{SX[s]}" for k in SITE_KINDS for s in SIDES]
    return {nm: out[nm] for nm in order}


def site_names(sites: dict) -> list:
    return list(sites)


def site_entries_world(sites: dict, J: np.ndarray, R: np.ndarray, cfg: MagnetCfg) -> dict:
    """{name: (P (T,E,3) surface points, N (T,E,3) outward normals)} of her posed body (J (T,55,3), R (T,55,3,3) global)"""
    from . import hand_contact as hc
    out = {}
    for nm, s in sites.items():
        if s["kind"] == "hand":
            pc, pn = hc.palm_centre(J, s["side"]), hc.palm_normal(J, s["side"])
            out[nm] = (np.stack([pc + cfg.her_palm_m * pn, pc - cfg.her_back_m * pn], 1), np.stack([pn, -pn], 1))
            continue
        j = s["joint"]
        p = np.array([e["p"] for e in s["entries"]], float)
        n = np.array([e["n"] for e in s["entries"]], float)
        out[nm] = (J[:, j, None] + np.einsum("tij,ej->tei", R[:, j], p), np.einsum("tij,ej->tei", R[:, j], n))
    return out


def site_distances(pc: np.ndarray, entries: dict, cfg: MagnetCfg) -> tuple[np.ndarray, np.ndarray]:
    """his palm centre (T,3) -> per site the distance to its nearest entry's target (T,S) and that entry (T,S)"""
    D, A = [], []
    for nm, (P, N) in entries.items():
        d = np.linalg.norm(pc[:, None] - (P + cfg.palm_m * N), axis=2)
        D.append(d.min(1))
        A.append(d.argmin(1))
    return np.stack(D, 1), np.stack(A, 1)


# ------------------------------------------------------------------------------------------------ 2. the sticky state machine

def frames_of(cfg: MagnetCfg, hz: float = HZ) -> dict:
    return {"on": max(1, int(round(cfg.t_on * hz))), "off": max(1, int(round(cfg.t_off * hz))), "dwell": int(round(cfg.dwell * hz)),
            "blend": max(1, int(round(cfg.t_blend * hz)))}


def sticky_timeline(D: list, cfg: MagnetCfg, hz: float = HZ) -> list:
    """D: per hand (T,S) distances palm -> site target. -> per hand the attachments [{"site", "k0", "k1", "via"}] (frames; the weight is
    0 at k0 and k1). Engage: the nearest site within r_in for t_on (k0 = that frame). Release: beyond r_out for t_off after the dwell
    (k1 = the first frame of that run beyond r_out, at least the dwell after k0; within the dwell a release only beyond r_out +
    dwell_pull_m). Switch: another site within r_in and nearer by margin for t_on after the dwell (the old one's k1
    = the switch frame + blend, the new one's k0 = the switch frame). A site held by the other hand is no candidate."""
    nf = frames_of(cfg, hz)
    H = len(D)
    T = len(D[0])
    st = [{"cur": -1, "since": 0, "cand": -1, "cc": 0, "out": 0, "sw": -1, "swc": 0} for _ in range(H)]
    att = [[] for _ in range(H)]
    for k in range(T):
        for h in range(H):
            s = st[h]
            d = np.array(D[h][k], float)
            for g in range(H):
                if g != h and st[g]["cur"] >= 0:
                    d[st[g]["cur"]] = np.inf
            near = int(np.argmin(d))
            if s["cur"] < 0:
                if d[near] < cfg.r_in:
                    s["cc"] = s["cc"] + 1 if s["cand"] == near else 1
                    s["cand"] = near
                else:
                    s["cand"], s["cc"] = -1, 0
                if s["cc"] >= nf["on"]:
                    att[h].append({"site": s["cand"], "k0": k, "k1": T - 1, "via": "engage"})
                    s.update(cur=s["cand"], since=k, cand=-1, cc=0, out=0, sw=-1, swc=0)
                continue
            dc = float(D[h][k][s["cur"]])
            s["out"] = s["out"] + 1 if dc > cfg.r_out else 0
            dwell_ok = k - s["since"] >= nf["dwell"]
            if s["out"] >= nf["off"] and (dwell_ok or dc > cfg.r_out + cfg.dwell_pull_m):
                k_out = k - s["out"] + 1                                                  # the first frame of the run beyond r_out
                att[h][-1]["k1"] = max(k_out, s["since"] + nf["dwell"]) if dwell_ok else k_out   # >= the dwell (unless pulled too far)
                s.update(cur=-1, cand=-1, cc=0, out=0, sw=-1, swc=0)
                continue
            if dwell_ok and near != s["cur"] and d[near] < cfg.r_in and d[near] < dc - cfg.margin:
                s["swc"] = s["swc"] + 1 if s["sw"] == near else 1
                s["sw"] = near
                if s["swc"] >= nf["on"]:
                    att[h][-1]["k1"] = min(T - 1, k + nf["blend"])
                    att[h].append({"site": near, "k0": k, "k1": T - 1, "via": "switch"})
                    s.update(cur=near, since=k, cand=-1, cc=0, out=0, sw=-1, swc=0)
            else:
                s["sw"], s["swc"] = -1, 0
    return att


def drop_blips(att: list, cfg: MagnetCfg, hz: float = HZ) -> list:
    """an engage-to-release attachment (no switch into or out of it) shorter than min_hold_s is dropped: two ramps of t_blend inside a
    second are a bump of his elbow, not a hold (1:09.5, his left hand 19-25 cm from her hand for 1.1 s)"""
    out = []
    for i, a in enumerate(att):
        solo = a["via"] == "engage" and (i + 1 >= len(att) or att[i + 1]["via"] != "switch")
        if solo and a["k1"] - a["k0"] < cfg.min_hold_s * hz:
            continue
        out.append(a)
    return out


def attachment_weights(att: list, T: int, blend: int) -> np.ndarray:
    """(T, n_att) linear weights: 0 at k0, 1 after `blend` frames, 1 -> 0 over the `blend` frames before k1 (a triangle when shorter)"""
    W = np.zeros((T, len(att)))
    k = np.arange(T, dtype=float)
    for i, a in enumerate(att):
        W[:, i] = np.clip(np.minimum((k - a["k0"]) / blend, (a["k1"] - k) / blend), 0.0, 1.0)
        W[:a["k0"], i] = 0.0
        W[a["k1"] + 1:, i] = 0.0
    return W


def hand_targets(pc0: np.ndarray, att: list, W: np.ndarray, entries: dict, names: list, arg: np.ndarray, cfg: MagnetCfg) -> dict:
    """per frame: the palm target X (T,3) = lerp(original palm, site targets, w), the aimed site normal (T,3), the total weight (T,);
    each attachment keeps one entry (its most frequent nearest entry) -> also returned per attachment"""
    T = len(pc0)
    acc = np.zeros((T, 3))
    nacc = np.zeros((T, 3))
    nback = np.zeros((T, 3))
    wback = np.zeros(T)
    whand = np.zeros(T)
    wsum = W.sum(1) if W.size else np.zeros(T)
    for i, a in enumerate(att):
        nm = names[a["site"]]
        P, N = entries[nm]
        sl = slice(a["k0"], a["k1"] + 1)
        e = int(np.bincount(arg[sl, a["site"]], minlength=P.shape[1]).argmax())
        a["entry"] = e
        tgt, nrm, info = clamp_target(P[:, e] + cfg.palm_m * N[:, e], N[:, e], pc0, a["k0"], a["k1"], cfg)
        a.update(info)
        acc += W[:, i, None] * tgt
        nacc += W[:, i, None] * nrm
        kind = site_kind(nm)
        if kind in BACK_KINDS:
            nback += W[:, i, None] * nrm
            wback += W[:, i]
        elif kind == "hand":
            whand += W[:, i]
    X = acc + (1.0 - wsum)[:, None] * pc0
    n_aim = _unit(nacc)
    return {"X": X, "n": n_aim, "w": wsum, "n_back": _unit(nback), "w_back": wback, "w_hand": whand}


# the sites a supporting (weight-bearing) hand rests on: her back and shoulders (not her arms, not her hand)
BACK_KINDS = ("upper_shoulder", "upper_back", "mid_back", "shoulder_top")


def site_kind(name: str) -> str:
    return name.rsplit("_", 1)[0]


def clamp_target(tgt: np.ndarray, nrm: np.ndarray, pc0: np.ndarray, k0: int, k1: int, cfg: MagnetCfg, hz: float = HZ):
    """one attachment's site target (T,3) and normal on k0..k1 -> followed smoothly, never copied: (1) where the site's offset to his
    ORIGINAL palm changes by more than jump_m within jump_s (her tracking glitch), the previous offset is held (the target moves with his
    original palm, the normal stays) until it settles; (2) the target steps at most max(|his palm step| + target_extra_mps / hz,
    target_floor_m) per frame. The weights stay linear in time (this only smooths what w blends toward). -> (tgt, nrm, info)"""
    out, nout = tgt.copy(), nrm.copy()
    rel = tgt - pc0
    n = max(1, int(round(cfg.jump_s * hz)))
    held = clamped = 0
    extra = cfg.target_extra_mps / hz
    for k in range(k0 + 1, k1 + 1):
        lo = max(k0, k - n)
        jump = np.linalg.norm(rel[lo:k] - rel[k], axis=1).max() > cfg.jump_m
        if jump:
            out[k] = out[k - 1] + (pc0[k] - pc0[k - 1])
            nout[k] = nout[k - 1]
            held += 1
            continue
        d = tgt[k] - out[k - 1]
        L = np.linalg.norm(d)
        lim = max(float(np.linalg.norm(pc0[k] - pc0[k - 1])) + extra, cfg.target_floor_m)
        if L > lim:
            out[k] = out[k - 1] + d * (lim / L)
            clamped += 1
        else:
            out[k] = tgt[k]
    return out, nout, {"frames_held": held, "frames_clamped": clamped}


# ------------------------------------------------------------------------------------------------ 3-4. the palm orientation and the arm

def flex_dist(L1, L2, phi):
    return np.sqrt(np.maximum(L1 ** 2 + L2 ** 2 + 2 * L1 * L2 * np.cos(phi), 1e-12))


def arm_ik(S, E0, W0, G14, G17, G19, G21, pc0, pn0, X, n_aim, w, side: str, cfg: MagnetCfg, e_target=None, sc_fixed=None,
           contig=None) -> dict:
    """one arm on n frames (only frames with w > 0): the palm centre to X, the palm normal turned toward -n_aim by the fraction w (at most
    turn_max_deg), the elbow on the IK circle nearest the original elbow E0 with the original hinge normal carried over; out of reach w
    is scaled down until reachable. G* global rotations of collar / shoulder / elbow / wrist. -> local R17, R19, R21 (n,3,3), the new
    elbow E and wrist W, the scale (1 = reached), the palm turn (deg), the flexion before / after (deg)"""
    n = len(S)
    L1 = np.linalg.norm(E0 - S, axis=1)
    L2 = np.linalg.norm(W0 - E0, axis=1)
    e10, f0 = _unit(E0 - S), _unit(W0 - E0)
    phi0 = np.arccos(np.clip((e10 * f0).sum(1), -1, 1))
    c_loc = np.einsum("nji,nj->ni", G21, pc0 - W0)                       # the palm centre in the wrist's frame
    axis, ang = rot_between(pn0, -n_aim)
    ang = np.minimum(ang, np.radians(cfg.turn_max_deg))
    lo_phi = np.minimum(np.radians(cfg.flex_min_deg), phi0)
    hi_phi = np.maximum(np.radians(cfg.flex_max_deg), phi0)
    dmax, dmin = flex_dist(L1, L2, lo_phi), flex_dist(L1, L2, hi_phi)

    def at(sc):
        G = rot_axis_angle(axis, sc * w * ang) @ G21
        Wt = pc0 + sc[:, None] * (X - pc0) - np.einsum("nij,nj->ni", G, c_loc)
        return G, Wt

    def ok(Wt):
        d = np.linalg.norm(Wt - S, axis=1)
        return (d <= dmax + 1e-9) & (d >= dmin - 1e-9)
    sc = np.ones(n) if sc_fixed is None else np.asarray(sc_fixed, float)
    G21n, Wt = at(sc)
    if sc_fixed is not None:                                             # a scale already smoothed in time (snap): reachable by construction
        n = 0                                                            # (no search)
    # the scale = the first unreachable point along the path from the original palm (a scan, then bisection): reachability need not be
    # monotone along the path (it can pass near his own shoulder), and a bisection over [0, 1] then jumps between frames (v2 fix)
    grid = np.linspace(0.0, 1.0, 21)
    okg = np.stack([ok(at(np.full(n, g))[1]) for g in grid], 0) if n else np.ones((len(grid), 0), bool)
    okg[0] = True
    first_bad = np.where((~okg).any(0), np.argmax(~okg, 0), len(grid))
    bad = first_bad < len(grid)
    if n and bad.any():
        lo = np.where(bad, grid[np.maximum(first_bad - 1, 0)], 1.0)
        hi = np.where(bad, grid[np.minimum(first_bad, len(grid) - 1)], 1.0)
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            g = ok(at(mid)[1])
            lo = np.where(g, mid, lo)
            hi = np.where(g, hi, mid)
        sc = np.where(bad, lo, 1.0)
        G21n, Wt = at(sc)
    d = np.clip(np.linalg.norm(Wt - S, axis=1), dmin, dmax)
    u = _unit(Wt - S)
    a = (L1 ** 2 - L2 ** 2 + d ** 2) / (2 * d)
    rr = np.sqrt(np.maximum(L1 ** 2 - a ** 2, 0.0))
    C0 = S + a[:, None] * u
    # the original hinge normal (the elbow's flexion axis): e1 x f, or near a straight arm the rest axis carried by the upper arm
    nx = np.cross(e10, f0)
    n_fb = _unit(np.einsum("nij,j->ni", G17, HINGE_REST[side]))
    wn = np.clip(np.linalg.norm(nx, axis=1) / np.sin(np.radians(10.0)), 0, 1)[:, None]
    n0 = _unit(wn * _unit(nx) + (1 - wn) * n_fb)
    n0 = _unit(n0 - (n0 * e10).sum(1, keepdims=True) * e10)
    # the elbow on the circle nearest the original elbow; where the original elbow lies on the shoulder-wrist line, the side the hinge gives
    hp = (E0 - C0) - ((E0 - C0) * u).sum(1, keepdims=True) * u
    q_fb = _unit(np.cross(u, n0))
    wq = np.clip(np.linalg.norm(hp, axis=1) / 0.03, 0, 1)[:, None]
    q = _unit(wq * _unit(hp) + (1 - wq) * q_fb)
    q = _unit(q - (q * u).sum(1, keepdims=True) * u)
    if contig is not None or e_target is not None:
        q = _elbow_dirs(q, u, C0, rr, E0, contig, e_target)
    E = C0 + rr[:, None] * q
    Wn = E + L2[:, None] * _unit(Wt - E)
    e1, f1 = _unit(E - S), _unit(Wn - E)
    n1 = _unit(np.cross(q, u))
    G17n = frame_of(e1, n1) @ np.transpose(frame_of(e10, n0), (0, 2, 1)) @ G17
    G19n = frame_of(f1, n1) @ np.transpose(frame_of(f0, n0), (0, 2, 1)) @ G19
    tr = lambda M: np.transpose(M, (0, 2, 1))
    return {"R17": tr(G14) @ G17n, "R19": tr(G17n) @ G19n, "R21": tr(G19n) @ G21n, "E": E, "W": Wn, "scale": sc,
            "turn_deg": np.degrees(sc * w * ang), "flex0_deg": np.degrees(phi0), "flex_deg": np.degrees(np.arccos(np.clip((e1 * f1).sum(1), -1, 1)))}


# ------------------------------------------------------------------------------------------------ 4b. the wrist rule (v3)

def wrist_compose(flex, dev, tw, side: str) -> np.ndarray:
    """the inverse of zouk_contacts.wrist_angles_R: degrees (n,) of flexion / deviation (the swing vector, its length = the bend) and the
    twist -> the wrist's local rotation R = T(a, twist) F (n,3,3), F the minimal rotation taking a_h = F^T a onto the forearm axis a"""
    from . import zouk_contacts as zc
    a, nv, tv = zc._rest_axes(side)
    f, d, w = (np.radians(np.asarray(x, float).reshape(-1)) for x in (flex, dev, tw))
    b = np.hypot(f, d)
    sinc = np.where(b > 1e-9, np.sin(b) / np.maximum(b, 1e-12), 1.0)
    ah = np.cos(b)[:, None] * a - sinc[:, None] * (f[:, None] * nv + d[:, None] * tv)
    ax = np.cross(ah, a)
    sn = np.linalg.norm(ax, axis=1)
    axu = np.where(sn[:, None] > 1e-12, ax / np.maximum(sn, 1e-12)[:, None], tv)
    F = rot_axis_angle(axu, np.where(sn > 1e-12, b, 0.0))
    return rot_axis_angle(np.tile(a, (len(w), 1)), w) @ F


def _wrap(x):
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0


def _spans(mask: np.ndarray) -> list:
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def ramp_correction(c: np.ndarray, n: int) -> np.ndarray:
    """a signed per-frame correction c (T,) (0 where nothing is clamped) -> E (T,): |E| = the max over s of |c(s)| (1 - |t - s| / n) with
    the sign of that s: linear ramps of n frames into and out of each clamped span, and |E| >= |c| everywhere (the limit still holds)"""
    c = np.asarray(c, float)
    T = len(c)
    best, sgn = np.abs(c).copy(), np.sign(c)
    for k in range(1, max(1, n)):
        f = 1.0 - k / n
        for sh in (k, -k):
            if abs(sh) >= T:
                continue
            src, ss = np.zeros(T), np.zeros(T)
            if sh > 0:
                src[sh:], ss[sh:] = np.abs(c[:-sh]) * f, np.sign(c[:-sh])
            else:
                src[:sh], ss[:sh] = np.abs(c[-sh:]) * f, np.sign(c[-sh:])
            m = src > best
            best[m], sgn[m] = src[m], ss[m]
    return sgn * best


def wrist_limits(cfg: MagnetCfg) -> dict:
    return {"flex": (-cfg.wrist_ext_max_deg, cfg.wrist_flex_max_deg), "dev": (-cfg.wrist_ulnar_max_deg, cfg.wrist_radial_max_deg),
            "tw": (-cfg.wrist_twist_max_deg, cfg.wrist_twist_max_deg)}


def clamp_wrist_angles(ang: dict, cfg: MagnetCfg, hz: float = HZ, gate: np.ndarray | None = None) -> dict:
    """zouk_contacts.wrist_angles (flexion, deviation, twist; degrees per frame) -> clamped to wrist_limits with linear ramps of
    wrist_ramp_s (ramp_correction). The twist: each span beyond the limit is held at the bound it left by (or, when it comes back on the
    other side through 180 deg, carried linearly from one bound to the other), so the hand never flips. gate (T,) in [0, 1] scales the
    per-frame correction before the ramps (wrist_rule: 1 where the detected input itself is beyond a limit, else 1 - the hand-hold
    weight: a palm-to-palm hold keeps the magnet's turn)"""
    n = max(2, int(round(cfg.wrist_ramp_s * hz)))
    gt = 1.0 if gate is None else np.asarray(gate, float)
    L = wrist_limits(cfg)
    out = {}
    for key, nm in (("flex", "flexion"), ("dev", "deviation")):
        lo, hi = L[key]
        x = np.asarray(ang[nm], float)
        out[key] = x + ramp_correction(gt * (np.clip(x, lo, hi) - x), n)
        if gate is None:
            out[key] = np.clip(out[key], lo, hi)
    tw = np.asarray(ang["twist"], float)
    out["tw"] = _wrap(tw + ramp_correction(gt * _wrap(twist_target(tw, L["tw"][1]) - tw), n))
    return out


def twist_target(tw: np.ndarray, lim: float) -> np.ndarray:
    """the twist (T,) clamped to +-lim with continuity: a span beyond the limit is held at the bound it left by, or (back on the other
    side through 180 deg) carried linearly from one bound to the other"""
    tw = np.asarray(tw, float)
    tgt = np.clip(tw, -lim, lim)
    T = len(tw)
    for k0, k1 in _spans(np.abs(tw) > lim):
        s0 = np.sign(tw[k0 - 1] if k0 > 0 else tw[k0]) or 1.0
        s1 = np.sign(tw[k1 + 1]) if k1 + 1 < T else s0
        tgt[k0:k1 + 1] = s0 * lim if s0 == s1 else np.linspace(s0 * lim, s1 * lim, k1 - k0 + 1)
    return tgt


def support_angles(Ge: np.ndarray, aim: np.ndarray, pn_h: np.ndarray, dev_now: np.ndarray, side: str, cfg: MagnetCfg) -> dict:
    """the weight-bearing wrist with the palm normal toward `aim` (n,3 world): Ge the forearm's (elbow's) global rotation (n,3,3), pn_h his
    palm normal in the hand's own frame (3,), dev_now the deviation it has now -> degrees: deviation clipped to support_dev_deg, flexion
    within [-support_ext_deg, support_flex_deg] (bent back up to 70, in at most 15) tilting the palm toward the aim along the forearm, then
    the twist (about the forearm) turning it onto the aim"""
    from . import zouk_contacts as zc
    a = zc._rest_axes(side)[0]
    n = len(aim)
    p = _unit(np.einsum("nji,nj->ni", Ge, aim))
    dev = np.clip(np.asarray(dev_now, float), -cfg.support_dev_deg, cfg.support_dev_deg)
    zero = np.zeros(n)
    lo, hi = np.full(n, -cfg.support_ext_deg), np.full(n, cfg.support_flex_deg)      # (negative = extension: allowed further)
    for _ in range(30):                                  # (F pn_h) . a falls with the flexion (the palm tilts back toward the elbow)
        mid = 0.5 * (lo + hi)
        g = np.einsum("nij,j->ni", wrist_compose(mid, dev, zero, side), pn_h) @ a - p @ a
        lo, hi = np.where(g > 0, mid, lo), np.where(g > 0, hi, mid)
    flex = 0.5 * (lo + hi)
    v = np.einsum("nij,j->ni", wrist_compose(flex, dev, zero, side), pn_h)
    vp, pp = v - (v @ a)[:, None] * a, p - (p @ a)[:, None] * a
    tw = np.degrees(np.arctan2(np.cross(vp, pp) @ a, (vp * pp).sum(1)))
    return {"flex": flex, "dev": dev, "tw": tw}                # (raw: wrist_rule clamps it with continuity in time)


def depop(Rn: np.ndarray, Rref: np.ndarray, max_step_deg: float = 12.0, half: int = 6, passes: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """local wrist rotations (T,3,3) -> where a frame-to-frame step exceeds max(max_step_deg, the reference's own step + 5 deg) (the
    clamp crossing the swing's 180-deg singularity, the twist the palm needs turning past 180 deg), the window +-half frames round it is
    replaced by a slerp between its ends; repeated (a window grows) -> (R, the frames replaced)"""
    from scipy.spatial.transform import Rotation as Ro, Slerp
    Rn = Rn.copy()
    T = len(Rn)

    def steps(R):
        return np.degrees(np.linalg.norm(mat_to_rotvec(np.transpose(R[:-1], (0, 2, 1)) @ R[1:]), axis=1))
    sref = steps(Rref)
    fixed = np.zeros(T, bool)
    for _ in range(passes):
        bad = steps(Rn) > np.maximum(max_step_deg, sref + 5.0)
        if not bad.any():
            break
        for k0, k1 in _spans(bad):
            a, b = max(0, k0 - half), min(T - 1, k1 + 1 + half)
            Rn[a:b + 1] = Slerp([0.0, 1.0], Ro.from_matrix(np.stack([Rn[a], Rn[b]])))(np.linspace(0, 1, b - a + 1)).as_matrix()
            fixed[a:b + 1] = True
    return Rn, fixed


def counterbalance_intervals(path: Path | None = None) -> list:
    """her counterbalance intervals (dance seconds) from the recap capture's counterbalance.json (read only)"""
    from .take import PACKAGE_ROOT
    p = Path(path) if path else PACKAGE_ROOT.parent / "head-movement" / "Assets" / "StreamingAssets" / CAPTURE / "counterbalance.json"
    if not p.exists():
        log(f"no {p}: no counterbalance intervals")
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return [(float(i["t0"]), float(i["t1"])) for i in d.get("intervals", []) if i.get("kind") == "counterbalance"]


def interval_weight(T: int, intervals: list, ramp_s: float, hz: float = HZ) -> np.ndarray:
    """1 inside each interval, linear ramps of ramp_s outside its ends"""
    t = np.arange(T) / hz
    w = np.zeros(T)
    for t0, t1 in intervals:
        w = np.maximum(w, np.clip(np.minimum((t - (t0 - ramp_s)) / ramp_s, ((t1 + ramp_s) - t) / ramp_s), 0.0, 1.0))
    return w


def wrist_rule(fit_lead: dict, Pose: np.ndarray, kl: dict, per_hand: dict, ents: dict, names: list, pcs: dict, cfg: MagnetCfg,
               intervals: list, Pose0: np.ndarray | None = None, aim_override: dict | None = None) -> tuple[np.ndarray, dict]:
    """v3: both wrists clamped (clamp_wrist_angles) and, by the support weight ws = max(w of a back / shoulder attachment, inside a
    counterbalance: (1 - w of a hand hold) x reach fade), blended in angle space toward support_angles; a twist change rolls the forearm
    at the elbow by support_elbow_share (about the elbow -> wrist bone: the wrist joint stays put). Only frames whose wrist changes are
    written. -> (Pose, info)"""
    from . import hand_contact as hc
    from . import zouk_contacts as zc
    Pose = Pose.copy()
    T = len(Pose)
    Jr = zc.rest_body(np.asarray(fit_lead["betas"], float))["J"]
    cb = interval_weight(T, intervals, cfg.support_cb_ramp_s)
    back_names = [nm for nm in names if site_kind(nm) in BACK_KINDS]
    info = {"counterbalance_intervals": [list(iv) for iv in intervals]}
    for s in SIDES:
        R = zc.rotvec_to_mat(Pose[:, WRIST[s]])
        ang = zc.wrist_angles_R(R, s)
        tg = per_hand[s]["tg"]
        w_back, n_back, w_hand = tg["w_back"], tg["n_back"], tg["w_hand"]
        L = wrist_limits(cfg)
        a0 = zc.wrist_angles_R(zc.rotvec_to_mat((Pose if Pose0 is None else Pose0)[:, WRIST[s]]), s)
        over0 = ((a0["flexion"] > L["flex"][1]) | (a0["flexion"] < L["flex"][0]) | (a0["deviation"] > L["dev"][1]) |
                 (a0["deviation"] < L["dev"][0]) | (np.abs(a0["twist"]) > L["tw"][1]))
        cl = clamp_wrist_angles(ang, cfg, gate=np.where(over0, 1.0, 1.0 - w_hand))
        # inside a counterbalance: the back / shoulder site entry nearest his original palm over the (ramped) interval
        n_cb, g = np.zeros((T, 3)), np.zeros(T)
        cb_sites = []
        for t0, t1 in intervals:
            ks = np.arange(max(0, int(np.floor((t0 - cfg.support_cb_ramp_s) * HZ))), min(T, int(np.ceil((t1 + cfg.support_cb_ramp_s) * HZ)) + 1))
            if not len(ks):
                continue
            best = None
            # the back / shoulder site his hand is attached to in the interval (most weight), else the nearest on average
            Wk = per_hand[s]["W"][ks] if per_hand[s]["W"].size else np.zeros((len(ks), 0))
            held = [(Wk[:, i].sum(), a) for i, a in enumerate(per_hand[s]["att"]) if site_kind(names[a["site"]]) in BACK_KINDS]
            held = [x for x in held if x[0] > 0]
            if held:
                a_ = max(held, key=lambda x: x[0])[1]
                nm_ = names[a_["site"]]
                P, N = ents[nm_]
                e_ = a_.get("entry", 0)
                d_ = np.linalg.norm(pcs[s][ks] - (P[ks, e_] + cfg.palm_m * N[ks, e_]), axis=1)
                best = (-1.0, nm_, e_, d_)
            for nm in ([] if held else back_names):
                P, N = ents[nm]
                d = np.linalg.norm(pcs[s][ks, None] - (P[ks] + cfg.palm_m * N[ks]), axis=2)
                for e in range(P.shape[1]):
                    if best is None or d[:, e].mean() < best[0]:
                        best = (d[:, e].mean(), nm, e, d[:, e])
            _, nm, e, d = best
            n_cb[ks] = ents[nm][1][ks, e]
            g[ks] = np.clip((cfg.reach_max_m[s] + cfg.support_cb_extra_m - d) / cfg.support_cb_extra_m, 0.0, 1.0)
            cb_sites.append({"t0": t0, "t1": t1, "site": nm, "entry": e, "mean_dist_m": round(float(d.mean()), 3)})
        w_cb = cb * (1.0 - w_hand) * g
        ws = np.maximum(w_back, w_cb)
        aim_n = _unit(w_back[:, None] * n_back + np.maximum(0.0, w_cb - w_back)[:, None] * n_cb)
        if aim_override and s in aim_override:            # v5: where the surface nudge moved the palm, her local surface normal there
            u_o, n_o = aim_override[s]
            aim_n = _unit((1.0 - u_o)[:, None] * aim_n + u_o[:, None] * n_o)
        flex, dev, tw = cl["flex"].copy(), cl["dev"].copy(), cl["tw"].copy()
        ks = np.flatnonzero(ws > 1e-9)
        if len(ks):
            Ge = zc.posed(fit_lead, ks, Pose[ks])["R"][:, ELBOW[s]]
            smp = np.arange(0, T, max(1, T // 400))
            pn_h = _unit(np.einsum("nji,nj->ni", kl["R"][smp, WRIST[s]], hc.palm_normal(kl["J"][smp], s)).mean(0))
            sup = support_angles(Ge, -aim_n[ks], pn_h, cl["dev"][ks], s, cfg)
            # the twist the palm needs can pass 180 deg (the palm turned right round): clamped with continuity (twist_target), never a flip
            tws = cl["tw"].copy()
            tws[ks] = sup["tw"]
            n_r = max(2, int(round(cfg.wrist_ramp_s * HZ)))
            tws = _wrap(tws + ramp_correction(_wrap(twist_target(tws, cfg.wrist_twist_max_deg) - tws), n_r))
            sup["tw"] = tws[ks]
            wk = ws[ks]
            flex[ks] += wk * (sup["flex"] - flex[ks])
            dev[ks] += wk * (sup["dev"] - dev[ks])
            tw[ks] = _wrap(tw[ks] + wk * _wrap(sup["tw"] - tw[ks]))
        if len(ks):                                       # (the support blend never leaves the limits; a supporting wrist may extend further)
            lo_f = -(cfg.wrist_ext_max_deg + ws[ks] * max(0.0, cfg.support_ext_deg - cfg.wrist_ext_max_deg))
            flex[ks], dev[ks] = np.clip(flex[ks], lo_f, L["flex"][1]), np.clip(dev[ks], *L["dev"])
        ch = (np.abs(flex - ang["flexion"]) > 1e-6) | (np.abs(dev - ang["deviation"]) > 1e-6) | (np.abs(_wrap(tw - ang["twist"])) > 1e-6)
        Rall = R.copy()
        Rall[ch] = wrist_compose(flex[ch], dev[ch], tw[ch], s)
        Rall, fixed = depop(Rall, R)
        ch = ch | fixed
        tw = zc.wrist_angles_R(Rall, s)["twist"]
        kc = np.flatnonzero(ch)
        if len(kc):
            Rn = Rall[kc]
            # the forearm's share of the twist change, continuous in time (the change unwrapped along the film)
            dtw = np.zeros(T)
            dtw[kc] = np.radians(_wrap(tw[kc] - ang["twist"][kc]))
            th = cfg.support_elbow_share * np.unwrap(dtw)[kc]
            b = _unit(Jr[WRIST[s]] - Jr[ELBOW[s]])
            Rb = rot_axis_angle(np.tile(b, (len(kc), 1)), th)
            roll = np.abs(th) > 1e-9
            Re = zc.rotvec_to_mat(Pose[kc, ELBOW[s]])
            Pose[kc[roll], ELBOW[s]] = mat_to_rotvec((Re @ Rb)[roll])
            Pose[kc, WRIST[s]] = mat_to_rotvec(np.transpose(Rb, (0, 2, 1)) @ Rn)
        info[s] = {"frames_changed": int(len(kc)), "spans": [[int(a), int(b_)] for a, b_ in _spans(ch)],
                   "frames_support": int((ws > 1e-9).sum()), "frames_support_full": int((ws >= 0.999).sum()),
                   "frames_over_limits_before": int(((ang["flexion"] > L["flex"][1]) | (ang["flexion"] < L["flex"][0]) |
                                                    (ang["deviation"] > L["dev"][1]) | (ang["deviation"] < L["dev"][0]) |
                                                    (np.abs(ang["twist"]) > L["tw"][1])).sum()),
                   "frames_over_limits_input": int(over0.sum()), "frames_depop": int(fixed.sum()),
                   "counterbalance_sites": cb_sites, "elbow_roll_deg_max": round(float(np.degrees(np.abs(th)).max()) if len(kc) else 0.0, 1)}
        per_hand[s]["ws"] = ws
    return Pose, info


# ------------------------------------------------------------------------------------------------ 4c. the surface nudge (v5)

HER_SURFACE_JOINTS = (PELVIS, SPINE1, SPINE2, SPINE3, NECK, 13, 14, 16, 17)     # her torso, collars, upper arms (dominant skinning joint)


def her_surface(fit_follow: dict, ks: np.ndarray, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """her shaped posed mesh on frames ks: the vertices of her torso / collars / upper arms and their normals (n, m, 3) each"""
    import torch
    from . import hand_contact as hc
    from .refine_hands import mesh_normals
    kin = hc.full_kin(fit_follow["betas"], "cpu")
    sel = np.flatnonzero(np.isin(kin.weights_np.argmax(1), HER_SURFACE_JOINTS))
    Pf, tr, F = hc.full_pose(fit_follow), np.asarray(fit_follow["transl"], float), np.asarray(kin.faces)
    Vo, No = [], []
    for i in range(0, len(ks), batch):
        kk = ks[i:i + batch]
        with torch.no_grad():
            V = kin.forward(torch.tensor(Pf[kk], dtype=torch.float64), torch.tensor(tr[kk]), want_verts=True)["verts"].numpy()
        Vo.append(V[:, sel])
        No.append(np.stack([mesh_normals(v, F)[sel] for v in V]))
    return np.concatenate(Vo), np.concatenate(No)


def surface_depth(Pt: np.ndarray, V: np.ndarray, N: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """points Pt (n,3) vs her surface V, N (n,m,3) -> depth (n,) (> 0: inside, by the nearest vertex's normal) and that normal (n,3)"""
    dep, nr = np.zeros(len(Pt)), np.zeros((len(Pt), 3))
    for i in range(0, len(Pt), chunk):
        sl = slice(i, i + chunk)
        j = ((V[sl] - Pt[sl, None]) ** 2).sum(2).argmin(1)
        r = np.arange(len(j))
        v, nn = V[sl][r, j], N[sl][r, j]
        dep[sl], nr[sl] = ((v - Pt[sl]) * nn).sum(1), nn
    return dep, nr


def hinge_axis(S, E, W, G17, side: str) -> np.ndarray:
    """the elbow's flexion axis (world, n x 3): e1 x f, near a straight arm the rest axis carried by the upper arm (as arm_ik)"""
    e1, f = _unit(E - S), _unit(W - E)
    nx = np.cross(e1, f)
    n_fb = _unit(np.einsum("nij,j->ni", G17, HINGE_REST[side]))
    wn = np.clip(np.linalg.norm(nx, axis=1) / np.sin(np.radians(10.0)), 0, 1)[:, None]
    n0 = _unit(wn * _unit(nx) + (1 - wn) * n_fb)
    return _unit(n0 - (n0 * e1).sum(1, keepdims=True) * e1)


def nudge_choice(dskin: np.ndarray, dwrist: np.ndarray, grid_deg: np.ndarray, allowed: np.ndarray, cfg: MagnetCfg,
                 sign: float | None = None) -> np.ndarray:
    """per frame (n,) the elbow change (deg, from grid_deg (g,); dskin / dwrist / allowed (n, g)): the smallest |change| that puts the palm
    skin within [-surface_gap_m, surface_in_m] of her surface and the wrist no deeper than surface_in_m (ties: + = bending); none: the
    least deep (if it is better by more than surface_in_m), else 0. sign: only changes of that sign (and 0)"""
    al = allowed if sign is None else allowed & ((np.sign(grid_deg) == sign) | (grid_deg == 0))[None]
    ok = al & (dskin <= cfg.surface_in_m) & (dskin >= -cfg.surface_gap_m) & (dwrist <= cfg.surface_in_m)
    key = np.abs(grid_deg) - 1e-3 * np.sign(grid_deg)                       # (bending first at equal |change|)
    big = np.where(ok, key[None], np.inf)
    j = big.argmin(1)
    r = np.arange(len(j))
    out = np.where(np.isfinite(big[r, j]), grid_deg[j], np.nan)
    bad = np.isnan(out)
    if bad.any():
        sc = np.where(al, np.maximum(np.maximum(dskin, dwrist), 0.0) + 1e-4 * np.abs(grid_deg)[None], np.inf)
        jb = sc.argmin(1)
        i0 = int(np.argmin(np.abs(grid_deg)))
        better = sc[r, jb] < np.maximum(np.maximum(dskin[:, i0], dwrist[:, i0]), 0.0) - cfg.surface_in_m
        out[bad] = np.where(better, grid_deg[jb], 0.0)[bad]
    return out


def surface_nudge(fit_lead: dict, fit_follow: dict, Pose: np.ndarray, per_hand: dict, cfg: MagnetCfg) -> tuple[dict, dict, dict]:
    """v5: on the supporting frames (ws > 0) of the wrist-ruled Pose: where his palm skin or wrist joint is more than surface_in_m inside
    her, the elbow change about its hinge (nudge_choice, one sign per span, linear ramps of surface_ramp_s) -> ({side: (T,) deg}, the
    wrist rule's aim override {side: (u (T,), her surface normal (T,3))}, info)"""
    from scipy.ndimage import gaussian_filter1d
    from . import hand_contact as hc
    from . import zouk_contacts as zc
    T = len(Pose)
    out, aim, info = {}, {}, {}
    sup = np.zeros(T, bool)
    for s in SIDES:
        sup |= per_hand[s].get("ws", np.zeros(T)) > 1e-9
    ks = np.flatnonzero(sup)
    if not len(ks):
        return out, aim, info
    V, N = her_surface(fit_follow, ks)
    kl = zc.posed(fit_lead, ks, Pose[ks])
    J, R = kl["J"], kl["R"]
    grid = np.arange(-cfg.surface_max_deg, cfg.surface_max_deg + 0.5, 1.0)
    n_r = max(2, int(round(cfg.surface_ramp_s * HZ)))
    for s in SIDES:
        ws = per_hand[s].get("ws", np.zeros(T))[ks]
        S, E, W, G17 = J[:, SHOULDER[s]], J[:, ELBOW[s]], J[:, WRIST[s]], R[:, SHOULDER[s]]
        pc, pn = hc.palm_centre(J, s), hc.palm_normal(J, s)
        skin = pc + cfg.palm_m * pn
        hn = hinge_axis(S, E, W, G17, s)
        flex0 = np.degrees(np.arccos(np.clip((_unit(E - S) * _unit(W - E)).sum(1), -1, 1)))
        ds0, _ = surface_depth(skin, V, N)
        dw0, _ = surface_depth(W, V, N)
        need = (ws > 1e-9) & ((ds0 > cfg.surface_in_m) | (dw0 > cfg.surface_in_m))
        dphi = np.zeros(T)
        u = np.zeros(T)
        if need.any():
            kn = np.flatnonzero(need)
            DS, DW = np.zeros((len(kn), len(grid))), np.zeros((len(kn), len(grid)))
            for gi, g in enumerate(grid):
                Rg = rot_axis_angle(hn[kn], np.full(len(kn), np.radians(g)))
                W2 = E[kn] + np.einsum("nij,nj->ni", Rg, W[kn] - E[kn])
                DW[:, gi] = surface_depth(W2, V[kn], N[kn])[0]
                DS[:, gi] = surface_depth(W2 + (skin[kn] - W[kn]), V[kn], N[kn])[0]   # (the hand's orientation kept: the rule re-aims it)
            fl = flex0[kn, None] + grid[None]
            allowed = (fl <= cfg.flex_max_deg) & (fl >= np.minimum(cfg.flex_min_deg, flex0[kn])[:, None])
            ch = nudge_choice(DS, DW, grid, allowed, cfg)
            # one sign per span (spans closer than two ramps merged): the majority of the frames, bending on a tie
            merged = []
            for a, b in _spans(need):
                if merged and a - merged[-1][1] <= 2 * n_r:
                    merged[-1][1] = b
                else:
                    merged.append([a, b])
            signs = []
            for a, b in merged:
                m = (kn >= a) & (kn <= b)
                sg = 1.0 if (ch[m] > 0).sum() >= (ch[m] < 0).sum() else -1.0
                signs.append(sg)
                wrong = np.flatnonzero(m & (np.sign(ch) == -sg))
                if len(wrong):
                    ch[wrong] = nudge_choice(DS[wrong], DW[wrong], grid, allowed[wrong], cfg, sign=sg)
            cT = np.zeros(T)
            cT[ks[kn]] = ch
            dphi = ramp_correction(cT, n_r)
            ind = np.zeros(T)
            ind[ks[kn[np.abs(ch) > 1e-9]]] = 1.0
            u = np.clip(ramp_correction(ind, n_r), 0.0, 1.0)
        # her surface normal at the nudged palm (smoothed in time): the aim of the re-run wrist rule where u > 0
        Rg = rot_axis_angle(hn, np.radians(dphi[ks]))
        W2 = E + np.einsum("nij,nj->ni", Rg, W - E)
        ds1, n1 = surface_depth(W2 + (skin - W), V, N)
        dw1, _ = surface_depth(W2, V, N)
        nT = np.zeros((T, 3))
        nT[ks] = _unit(gaussian_filter1d(n1, 2.0, axis=0, mode="nearest"))
        out[s] = dphi
        aim[s] = (u, nT)
        # the local elbow change: R19' = R19 Rot(G19^T hn, dphi) (the forearm and hand turn about the hinge through the elbow)
        per_hand[s]["nudge_axis_local"] = (ks, np.einsum("nji,nj->ni", R[:, ELBOW[s]], hn))
        sp = ws > 1e-9
        info[s] = {"frames_need": int(need.sum()), "frames_nudged": int((np.abs(dphi) > 1e-9).sum()),
                   "spans": [[int(a), int(b)] for a, b in _spans(np.abs(dphi) > 1e-9)],
                   "span_signs": signs if need.any() else [],
                   "dphi_deg_max_bend": round(float(dphi.max()), 1), "dphi_deg_max_straighten": round(float(-dphi.min()), 1),
                   "wrist_move_cm_max": round(float(np.linalg.norm(W2 - W, axis=1).max()) * 100, 1),
                   "skin_depth_cm_before_p95": round(float(np.percentile(ds0[need], 95)) * 100, 1) if need.any() else None,
                   "skin_depth_cm_after_est_p95": round(float(np.percentile(ds1[need], 95)) * 100, 1) if need.any() else None,
                   "wrist_depth_cm_before_max": round(float(dw0[sp].max()) * 100, 1) if sp.any() else None,
                   "wrist_depth_cm_after_est_max": round(float(dw1[sp].max()) * 100, 1) if sp.any() else None}
    return out, aim, info


def apply_elbow_nudge(Pose: np.ndarray, dphi: dict, per_hand: dict) -> np.ndarray:
    """Pose with each elbow's local rotation turned by dphi (deg, (T,)) about the hinge axis surface_nudge stored (the wrist's local
    rotation kept: the hand turns with the forearm until the wrist rule re-aims it)"""
    from . import zouk_contacts as zc
    Pose = Pose.copy()
    for s, d in dphi.items():
        ks, ax = per_hand[s]["nudge_axis_local"]
        m = np.abs(d[ks]) > 1e-9
        if not m.any():
            continue
        k = ks[m]
        Re = zc.rotvec_to_mat(Pose[k, ELBOW[s]])
        Pose[k, ELBOW[s]] = mat_to_rotvec(Re @ rot_axis_angle(_unit(ax[m]), np.radians(d[k])))
    return Pose


# ------------------------------------------------------------------------------------------------ the whole snap

def _elbow_dirs(q, u, C0, rr, E0, contig, e_target):
    """v2: the elbow direction on the IK circle chosen with continuity in time (contig[i]: frame i follows frame i-1): the circle point
    nearest the midpoint of the reference and the previous NEW elbow (the original elbow alone is ambiguous where it lies near the
    shoulder-wrist line: the elbow flipped 20-30 cm in one frame). Reference = the original elbow, or with elbow damping e_target"""
    n = len(q)
    contig = np.zeros(n, bool) if contig is None else np.asarray(contig, bool)
    ref = E0 if e_target is None else e_target
    out = q.copy()
    Eprev = None
    for i in range(n):
        if contig[i] and Eprev is not None:
            P = 0.5 * (ref[i] + Eprev) - C0[i]
        elif e_target is not None:
            P = ref[i] - C0[i]
        else:
            P = None
        if P is not None:
            hp = P - (P @ u[i]) * u[i]
            if np.linalg.norm(hp) > 1e-6:
                out[i] = hp / np.linalg.norm(hp)
        if e_target is not None:
            # then on that side of the circle to the target's height (clipped to the circle): of the two points at that height the one
            # nearest the side chosen (and the previous elbow)
            qi = out[i]
            qb = np.cross(u[i], qi)
            qb /= max(np.linalg.norm(qb), 1e-12)
            A = np.hypot(qi[1], qb[1])
            if rr[i] * A > 1e-4:
                al = np.arctan2(qb[1], qi[1])
                c = np.clip((e_target[i, 1] - C0[i, 1]) / (rr[i] * A), -1.0, 1.0)
                Ec = C0[i] + rr[i] * qi
                r0 = 0.5 * (Ec + Eprev) if (contig[i] and Eprev is not None) else Ec
                cands = [np.cos(t) * qi + np.sin(t) * qb for t in (al + np.arccos(c), al - np.arccos(c))]
                out[i] = min(cands, key=lambda v: np.linalg.norm(C0[i] + rr[i] * v - r0))
        Eprev = C0[i] + rr[i] * out[i]
    return out


def load_fit(d: Path) -> dict:
    out = {}
    for r in ROLES:
        with np.load(Path(d) / f"{r}.npz", allow_pickle=True) as z:
            out[r] = {k: z[k] for k in z.files}
    return out


def build_sites(take: str, force: bool = False) -> dict:
    P = paths(take)
    p = P["out"] / "sites.json"
    fit = load_fit(P["src_fit"])
    betas = np.round(np.asarray(fit["follow"]["betas"], float), 6).tolist()
    if p.exists() and not force:
        d = json.loads(p.read_text(encoding="utf-8"))
        if np.allclose(d.get("betas", []), betas, atol=2e-6):
            return d["sites"]
    sites = fit_sites(fit["follow"]["betas"])
    write_json(p, {"what": "magnet sites on the follower (dancecap.magnet.fit_sites): point + outward normal in the frame of the joint that "
                           "carries it (rest frame x = her left, y up, z front), ray cast on her shaped rest mesh; limb sites = a ring of "
                           "entries; her hand = palm / back face from her posed hand joints", "betas": betas, "sites": sites})
    sites_image(sites, fit["follow"]["betas"], P["out"] / "sites_rest.png")
    return sites


def snap(take: str, cfg: MagnetCfg | None = None, write: bool = True) -> dict:
    """steps 2-4 on the original pose: timeline -> weights -> targets -> IK -> the magnet fit (+ measure)"""
    from . import hand_contact as hc
    from . import zouk_contacts as zc
    cfg = cfg or MagnetCfg()
    P = paths(take)
    t0 = time.time()
    fit = load_fit(P["src_fit"])
    sites = build_sites(take)
    names = site_names(sites)
    T = len(fit["lead"]["t"])
    kl = zc.posed(fit["lead"])
    kf = zc.posed(fit["follow"])
    ents = site_entries_world(sites, kf["J"], kf["R"], cfg)
    nf = frames_of(cfg)
    D, A, pcs = [], [], {}
    for s in SIDES:
        pcs[s] = hc.palm_centre(kl["J"], s)
        d, a = site_distances(pcs[s], ents, cfg)
        D.append(d)
        A.append(a)
    D_eff = [effective_distances(D[h], strength_vector(cfg, s, names), cfg, s) for h, s in enumerate(SIDES)]
    att = [drop_blips(a, cfg) for a in sticky_timeline(D_eff, cfg)]
    Pose0 = hc.full_pose(fit["lead"])
    Pose = Pose0.copy()
    per_hand = {}
    for h, s in enumerate(SIDES):
        W = attachment_weights(att[h], T, nf["blend"])
        tg = hand_targets(pcs[s], att[h], W, ents, names, A[h], cfg)
        m = np.flatnonzero(tg["w"] > 1e-9)
        J, R = kl["J"], kl["R"]
        ik = None
        if len(m):
            args = (J[m, SHOULDER[s]], J[m, ELBOW[s]], J[m, WRIST[s]], R[m, COLLAR[s]], R[m, SHOULDER[s]], R[m, ELBOW[s]],
                    R[m, WRIST[s]], pcs[s][m], hc.palm_normal(J[m], s), tg["X"][m], tg["n"][m], tg["w"][m], s, cfg)
            contig = np.r_[False, np.diff(m) == 1]                     # frame i of m follows frame i - 1: the elbow keeps its side
            ik = arm_ik(*args, contig=contig)
            # out of reach the scale of w is per frame and can jump (a full arm whose palm turn grows): continuous in time, never above
            # any frame's own reachable scale (a min filter of 2r, then a box mean of r: each term of the mean is <= the frame's own)
            from scipy.ndimage import gaussian_filter1d, minimum_filter1d, uniform_filter1d
            r = max(1, int(round(cfg.scale_smooth_s * HZ)))
            sc = np.ones(T)
            sc[m] = ik["scale"]
            sc = uniform_filter1d(minimum_filter1d(sc, 4 * r + 1, mode="nearest"), 2 * r + 1, mode="nearest")
            sc = np.minimum(sc, 1.0)
            ik = arm_ik(*args, sc_fixed=sc[m], contig=contig)
            if cfg.elbow_damp_s > 0:
                dev = np.zeros((T, 3))
                dev[m] = ik["E"] - J[m, ELBOW[s]]
                dev_s = gaussian_filter1d(dev, cfg.elbow_damp_s * HZ, axis=0, mode="nearest")
                # x w: the smoothing spreads a deviation into the ramp frames, where w ~ 0 must stay the original elbow
                ik = arm_ik(*args, e_target=J[m, ELBOW[s]] + tg["w"][m, None] * dev_s[m], sc_fixed=sc[m], contig=contig)
            for j, key in ((SHOULDER[s], "R17"), (ELBOW[s], "R19"), (WRIST[s], "R21")):
                Pose[m, j] = mat_to_rotvec(ik[key])
        per_hand[s] = {"att": att[h], "W": W, "tg": tg, "m": m, "ik": ik, "D": D[h]}
    wr_info = None
    if cfg.wrist_rule:
        ivs = counterbalance_intervals()
        Pose_ik = Pose
        Pose, wr_info = wrist_rule(fit["lead"], Pose_ik, kl, per_hand, ents, names, pcs, cfg, ivs, Pose0=Pose0)
        if cfg.surface_nudge:                               # v5: the palm onto her surface (the elbow bent / straightened), then re-aimed
            dphi, aim, nudge_info = surface_nudge(fit["lead"], fit["follow"], Pose, per_hand, cfg)
            if any(np.abs(d).max() > 1e-9 for d in dphi.values()):
                Pose, wr_info = wrist_rule(fit["lead"], apply_elbow_nudge(Pose_ik, dphi, per_hand), kl, per_hand, ents, names, pcs, cfg,
                                           ivs, Pose0=Pose0, aim_override=aim)
            wr_info["surface_nudge"] = nudge_info
            for s in SIDES:
                if s in dphi:
                    per_hand[s]["dphi"] = dphi[s]
    changed = np.flatnonzero(np.abs(Pose - Pose0).max((1, 2)) > 0)
    J_new = np.asarray(fit["lead"]["joints"], float).copy()
    kn = None
    if len(changed):
        kn = zc.posed(fit["lead"], changed, Pose[changed])
        J_new[changed] = kn["J"]
    # the palm reached (FK of the written pose) on the frames where the scale stayed 1
    resid = {}
    for s in SIDES:
        ph = per_hand[s]
        m, ik = ph["m"], ph["ik"]
        if ik is None:
            resid[s] = None
            continue
        pc_new = hc.palm_centre(J_new[m], s)
        Xs = pcs[s][m] + ik["scale"][:, None] * (ph["tg"]["X"][m] - pcs[s][m])
        e = np.linalg.norm(pc_new - Xs, axis=1)
        resid[s] = {"max_mm": round(float(e.max()) * 1000, 2), "p99_mm": round(float(np.percentile(e, 99)) * 1000, 2),
                    "frames_reduced_w": int((ik["scale"] < 1 - 1e-6).sum()), "min_scale": round(float(ik["scale"].min()), 3),
                    "palm_turn_deg_max": round(float(ik["turn_deg"].max()), 1),
                    "flex_deg_range_new": [round(float(ik["flex_deg"].min()), 1), round(float(ik["flex_deg"].max()), 1)]}
    tl = timeline_json(take, cfg, att, per_hand, names, T)
    tl["palm_reached"] = resid
    if wr_info is not None:
        tl["wrist_rule"] = wr_info
    tl["seconds"] = round(time.time() - t0, 1)
    if write:
        fdir = P["fit"]
        fdir.mkdir(parents=True, exist_ok=True)
        d = dict(fit["lead"])
        d["global_orient"] = Pose[:, 0].copy()
        d["body_pose"] = Pose[:, 1:22].reshape(T, 63).copy()
        d["left_hand_pose"] = Pose[:, 25:40].reshape(T, 45).copy()
        d["right_hand_pose"] = Pose[:, 40:55].reshape(T, 45).copy()
        d["joints"] = J_new
        d["source"] = np.array(str(fit["lead"].get("source", "")) + " | dancecap.magnet 2026-10-25: his arms (16-21) snapped to her sites")
        np.savez(fdir / "lead.npz", **d)
        shutil.copy2(P["src_fit"] / "follow.npz", fdir / "follow.npz")              # her motion: the input file, byte for byte
        write_json(fdir / "magnet.json", {"what": "the input (smplx_rf2_noslip) + his shoulders / elbows / wrists snapped to her sites "
                                                   "(dancecap.magnet)", "base": str(P["src_fit"]), "frames_changed": int(len(changed)),
                                          "config": asdict(cfg)})
        write_json(P["out"] / "timeline.json", tl)
        np.savez_compressed(P["out"] / "timeline.npz", **{f"w_{s}": per_hand[s]["tg"]["w"] for s in SIDES},
                            **{f"X_{s}": per_hand[s]["tg"]["X"] for s in SIDES}, **{f"D_{s}": per_hand[s]["D"] for s in SIDES},
                            **{f"ws_{s}": per_hand[s]["ws"] for s in SIDES if "ws" in per_hand[s]},
                            **{f"dphi_{s}": per_hand[s]["dphi"] for s in SIDES if "dphi" in per_hand[s]})
        meas = measure(take, J_new=J_new, tl=tl)
        tl["measure"] = meas
    else:
        tl["joints_new"] = J_new                                                       # (write=False: for a comparison of settings)
    log(f"snap: {sum(len(a) for a in att)} attachments, {len(changed)} frames changed, {tl['seconds']} s")
    return tl


def timeline_json(take: str, cfg: MagnetCfg, att: list, per_hand: dict, names: list, T: int) -> dict:
    out = {"what": "the magnet timeline (dancecap.magnet.sticky_timeline) of his hands on the ORIGINAL pose: per hand the attachments and "
                   "per frame the site (index into `sites`, -1 free) and its weight; film = dance until 246.459 s, then + 7.2 s",
           "take": take, "config": asdict(cfg), "frames": frames_of(cfg), "hz": HZ, "sites": names, "hands": {}}
    for h, s in enumerate(SIDES):
        ph = per_hand[s]
        W = ph["W"]
        site_pf = np.full(T, -1, int)
        if W.size:
            dom = W.argmax(1)
            on = W.max(1) > 0
            site_pf[on] = np.array([a["site"] for a in att[h]])[dom[on]]
        rows = []
        for i, a in enumerate(att[h]):
            rows.append({"site": names[a["site"]], "entry": a.get("entry", 0), "via": a["via"], "k0": a["k0"], "k1": a["k1"],
                         "t0": round(a["k0"] / HZ, 3), "t1": round(a["k1"] / HZ, 3), "film0": mmss(film_of(a["k0"] / HZ)),
                         "film1": mmss(film_of(a["k1"] / HZ)), "seconds": round((a["k1"] - a["k0"]) / HZ, 2),
                         "peak_w": round(float(W[:, i].max()), 3), "frames_held": a.get("frames_held", 0),
                         "frames_clamped": a.get("frames_clamped", 0)})
        out["hands"][s] = {"attachments": rows, "site_per_frame": site_pf.tolist(),
                           "w_per_frame": np.round(ph["tg"]["w"], 3).tolist()}
    return out


# ------------------------------------------------------------------------------------------------ 6. measures

def measure(take: str, J_new: np.ndarray | None = None, tl: dict | None = None) -> dict:
    """his elbow height (vs his shoulder) swings >= 3 cm within 1 s in 0-90 s and the 1-5 Hz wobble p90 (0-100 s), the sides.report
    numbers, for the original, v4.4b and the magnet; time attached per site per hand; hand-hold frames"""
    from .sides.report import band_rms_windows, swings_within
    P = paths(take)
    tl = tl or json.loads((P["out"] / "timeline.json").read_text(encoding="utf-8"))
    fits = {"original": np.asarray(load_fit(P["src_fit"])["lead"]["joints"], float)}
    if (P["v44b_fit"] / "lead.npz").exists():
        with np.load(P["v44b_fit"] / "lead.npz", allow_pickle=True) as z:
            fits["v4.4b"] = np.asarray(z["joints"], float)
    fits["magnet"] = J_new if J_new is not None else np.asarray(load_fit(P["fit"])["lead"]["joints"], float)
    T = len(fits["original"])
    t = np.arange(T) / HZ
    out = {"swings_3cm_1s_0_90s": {}, "wobble_1_5hz_p90_cm_0_100s": {}}
    for nm, J in fits.items():
        row, rw = {}, {}
        for s in SIDES:
            x = J[:, ELBOW[s], 1] - J[:, SHOULDER[s], 1]
            row[s] = swings_within(x[t < 90.0], 0.03, 30)
            r = band_rms_windows(x[t < 100.0])
            rw[s] = round(float(np.nanpercentile(r, 90)) * 100, 2)
        out["swings_3cm_1s_0_90s"][nm] = row
        out["wobble_1_5hz_p90_cm_0_100s"][nm] = rw
    att = {}
    hold = {}
    for s in SIDES:
        h = tl["hands"][s]
        per = {}
        for a in h["attachments"]:
            per[a["site"]] = round(per.get(a["site"], 0.0) + a["seconds"], 2)
        att[s] = dict(sorted(per.items(), key=lambda kv: -kv[1]))
        sp = np.asarray(h["site_per_frame"])
        w = np.asarray(h["w_per_frame"])
        hand_idx = [i for i, n in enumerate(tl["sites"]) if n.startswith("hand_")]
        hold[s] = {"frames_attached_to_her_hand": int(np.isin(sp, hand_idx).sum()),
                   "frames_full_weight_on_her_hand": int((np.isin(sp, hand_idx) & (w >= 0.999)).sum()),
                   "frames_attached_any": int((sp >= 0).sum()), "share_attached": round(float((sp >= 0).mean()), 3)}
    out["seconds_attached_per_site"] = att
    out["hand_holds"] = hold
    out["attachments"] = {s: len(tl["hands"][s]["attachments"]) for s in SIDES}
    if J_new is not None or (P["fit"] / "lead.npz").exists():
        Jm = fits["magnet"]
        dpalm = {}
        from . import hand_contact as hc
        for s in SIDES:
            e = np.linalg.norm(hc.palm_centre(Jm, s) - hc.palm_centre(fits["original"], s), axis=1)
            dpalm[s] = {"max_cm": round(float(e.max()) * 100, 1), "p95_cm_moved_frames": round(float(np.percentile(e[e > 0], 95)) * 100, 1)
                        if (e > 0).any() else 0.0}
        out["palm_moved_vs_original"] = dpalm
    write_json(P["out"] / "measure.json", out)
    return out


# ------------------------------------------------------------------------------------------------ 5. export + proof + install script

def prove_export(udir: Path, ref_dir: Path, tl: dict, parents: np.ndarray) -> dict:
    """the magnet export against the input's export (the 2026-10-11 motion): follow bin byte for byte; lead rows differ only on frames
    inside the attachments (k0..k1 of either hand), and there only his shoulder / elbow / wrist rotations (16-21) and the positions of
    the elbows, wrists and the fingers that ride on them; transl and headers equal; figure2.json identical"""
    from . import export_unity as eu
    out = {"ref_dir": str(ref_dir)}
    out["follow_sha256"] = sha(udir / "smplx_follow.bin")
    out["follow_identical_to_input_export"] = (udir / "smplx_follow.bin").read_bytes() == (ref_dir / "smplx_follow.bin").read_bytes()
    out["follow_sha_is_98657745"] = out["follow_sha256"] == INPUT_SHA["smplx_follow.bin"]
    h1, m1 = eu.read_motion(udir / "smplx_lead.bin")
    h0, m0 = eu.read_motion(ref_dir / "smplx_lead.bin")
    T = len(m0)
    allowed = np.zeros(T, bool)
    for s in SIDES:
        for a in tl["hands"][s]["attachments"]:
            allowed[a["k0"]:a["k1"] + 1] = True
        for k0, k1 in tl.get("wrist_rule", {}).get(s, {}).get("spans", []):     # v3: the wrist rule's frames (both wrists, the whole film)
            allowed[k0:k1 + 1] = True
    rot1, rot0 = m1[:, 3:3 + 220].reshape(T, 55, 4), m0[:, 3:3 + 220].reshape(T, 55, 4)
    pos1, pos0 = m1[:, 223:].reshape(T, 55, 3), m0[:, 223:].reshape(T, 55, 3)
    rows = np.flatnonzero((m1 != m0).any(1))
    rot_j = np.flatnonzero((rot1 != rot0).any((0, 2)))
    pos_j = np.flatnonzero((pos1 != pos0).any((0, 2)))
    ride = set()
    for j in range(55):
        k = j
        while k >= 0:
            if k in (ELBOW["left"], ELBOW["right"]):
                ride.add(j)
                break
            k = int(parents[k])
    lead = {"frames": int(T), "changed_rows": int(len(rows)), "changed_rows_outside_attachments": int((~allowed[rows]).sum()),
            "attachment_frames": int(allowed.sum()), "transl_identical": bool((m1[:, :3] == m0[:, :3]).all()),
            "rotations_changed_joints": rot_j.tolist(), "positions_changed_joints": pos_j.tolist(),
            "positions_allowed_joints": sorted(ride), "header_equal": {k: h1.get(k) == h0.get(k) for k in sorted(set(h0) | set(h1))}}
    lead["ok"] = bool(lead["changed_rows_outside_attachments"] == 0 and lead["transl_identical"] and set(rot_j) <= set(ARM_JOINTS)
                      and set(pos_j) <= ride and all(lead["header_equal"].values()))
    out["lead"] = lead
    out["figure2_identical"] = (udir / "figure2.json").read_bytes() == (ref_dir / "figure2.json").read_bytes()
    out["ok"] = bool(out["follow_identical_to_input_export"] and out["follow_sha_is_98657745"] and lead["ok"] and out["figure2_identical"])
    return out


def write_install_script(udir: Path, capture: str = CAPTURE) -> Path:
    """the sides install_motion.sh (dancecap.sides.install.write_install_script: check sha256 + sizes, back up the installed motion once,
    --check / --restore, honours work/unity.lock), reworded for the magnet motion; its first install backs up v4.4b"""
    from .sides import install as ins
    p = ins.write_install_script(udir, capture)
    s = p.read_text(encoding="utf-8")
    reps = [("put the sides motion (dancecap.sides) into", "put the magnet motion (dancecap.magnet) into"),
            ("(python -m dancecap.sides export --take <take>)", "(python -m dancecap.magnet export --take <take>)"),
            ("# never overwrite the first backup: the motion installed before sides",
             "# never overwrite the first backup: the motion installed before magnet (v4.4b: lead dd0286c3, follow f41d7983)"),
            ('echo "installed the sides motion (dancecap.sides) into', 'echo "installed the magnet motion (dancecap.magnet) into')]
    for a, b in reps:
        if a not in s:
            raise RuntimeError(f"install_motion.sh: the sides script changed ({a!r} not found)")
        s = s.replace(a, b)
    p.write_bytes(s.replace("\r\n", "\n").encode("utf-8"))
    return p


def export(take: str) -> dict:
    from . import export_unity as eu
    from . import hand_contact as hc
    P = paths(take)
    t0 = time.time()
    rel = P["fit"].relative_to(P["work"]).as_posix()
    rep = eu.export(take, SIDE_CAPTURE, install=False, room=False, textures=False, faces=False, shape_edits=True, allow_no_audio=True,
                    smplx_dir=rel)
    side = P["work"] / "export" / SIDE_CAPTURE
    u = P["unity"]
    u.mkdir(parents=True, exist_ok=True)
    for f in MOTION_FILES:
        shutil.copy2(side / f, u / f)
    tl = json.loads((P["out"] / "timeline.json").read_text(encoding="utf-8"))
    parents = np.asarray(hc.Body(load_fit(P["fit"])["lead"], "cpu").kin.parents)
    out = {"what": "the magnet motion (dancecap.magnet): exported into the side capture, never installed by the export",
           "sha256": {f: sha(u / f) for f in MOTION_FILES},
           "side_export": str(side), "files": [str(u / f) for f in MOTION_FILES],
           "fk_max_err_mm": {r: d["fk_max_err_mm"] for r, d in rep["dancers"].items()}, "warnings": rep["warnings"],
           "fit": {r: {"path": str(P["fit"] / f"{r}.npz"), "sha256": sha(P["fit"] / f"{r}.npz")} for r in ROLES},
           "installed_before_this_install": [{"file": f, "sha256_prefix": h[:8]} for f, h in V44B_SHA.items()],
           "exported": time.strftime("%Y-%m-%d %H:%M:%S")}
    out["proof_vs_input_export"] = prove_export(u, P["input_export"], tl, parents)
    out["seconds"] = round(time.time() - t0, 1)
    write_json(u / "unity_motion.json", out)
    write_install_script(u)
    log("exported", {k: v[:8] for k, v in out["sha256"].items()}, "proof ok:", out["proof_vs_input_export"]["ok"])
    return out


# ------------------------------------------------------------------------------------------------ images

def _view(kind: str, centre: np.ndarray, dist: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """(R world->camera, t) of a pinhole camera looking at centre from her back / front / left / above-behind"""
    if kind == "back":
        R = np.array([[-1.0, 0, 0], [0, -1, 0], [0, 0, 1]])
        C = centre + np.array([0, 0, -dist])
    elif kind == "front":
        R = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
        C = centre + np.array([0, 0, dist])
    elif kind == "top_back":
        z = _unit(np.array([0, -0.75, 0.66]))
        y = _unit(np.cross(z, np.array([-1.0, 0, 0])))
        x = np.cross(y, z)
        R = np.stack([x, y, z])
        C = centre - dist * z
    else:                                              # her left side
        R = np.array([[0, 0, -1.0], [0, -1, 0], [-1, 0, 0]])
        C = centre + np.array([dist, 0, 0])
    return R, -R @ C


def sites_image(sites: dict, betas: np.ndarray, path: Path) -> Path:
    """her shaped rest mesh from behind, from above-behind and from the front with every site entry (dot + 4 cm normal; the default entry
    big, labelled) -> png"""
    import cv2
    from . import hand_contact as hc
    from . import zouk_contacts as zc
    rb = zc.rest_body(np.asarray(betas, float))
    V, F, J = rb["V"], rb["faces"], rb["J"]
    Wd, Hd = 900, 560
    centre = np.array([0.0, J[SPINE3, 1] + 0.02, J[SPINE3, 2]])
    K = np.array([[1500.0, 0, Wd / 2], [0, 1500.0, Hd / 2], [0, 0, 1]])
    tiles = []
    for kind in ("back", "top_back", "front"):
        R, t = _view(kind, centre)
        img, cov = hc.render_faces(V, F, np.tile([0.78, 0.72, 0.66], (len(F), 1)), K, R, t, Hd, Wd)
        im = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
        im[~cov] = 35
        Cc = -R.T @ t
        for nm, s in sites.items():
            col = SITE_BGR[s["kind"]]
            for i, e in enumerate(s["entries"]):
                p, n = np.asarray(e["rest_point"]), np.asarray(e["n"])
                if n @ (Cc - p) <= 0:
                    continue
                uv = K @ (R @ p + t)
                q = K @ (R @ (p + 0.04 * n) + t)
                a, b = (int(uv[0] / uv[2]), int(uv[1] / uv[2])), (int(q[0] / q[2]), int(q[1] / q[2]))
                big = i == s["default"]
                cv2.line(im, a, b, col, 1, cv2.LINE_AA)
                cv2.circle(im, a, 6 if big else 3, col, -1 if big else 1, cv2.LINE_AA)
                if big:
                    cv2.putText(im, nm, (a[0] + 6, a[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(im, nm, (a[0] + 6, a[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        if kind == "front":       # her hands: palm / back faces are posed sites; mark the rest palm centres
            for sd in SIDES:
                pc = hc.palm_centre(J[None], sd)[0]
                uv = K @ (R @ pc + t)
                a = (int(uv[0] / uv[2]), int(uv[1] / uv[2]))
                cv2.circle(im, a, 6, SITE_BGR["hand"], -1, cv2.LINE_AA)
                cv2.putText(im, f"hand_{SX[sd]}", (a[0] + 6, a[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, SITE_BGR["hand"], 1, cv2.LINE_AA)
        tiles.append(hc.label_tile(im, [f"her rest mesh, from {kind.replace('_', ' ')}  (big dot = default entry, line = 4 cm normal)"], 0.5))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(tiles, 0))
    return path


def sheets(take: str, film_times=SHEET_FILM_S, tile: int = 400, extent_m: float = 1.2, ref_fit: Path | None = None,
           out_dir: Path | None = None, tips: bool = False) -> list:
    """per film time one 3 x 2 sheet of the 6 phones: his arms original (white) and magnet (green: shoulder - elbow - wrist - palm), her
    sites (small dots; an attached one big with its name) -> magnet/sheets/sheet_<film>.jpg. ref_fit: the white arms from this fit
    instead of the original (e.g. the previous magnet); tips: the arm line runs on to the middle fingertip (shows the wrist bend)"""
    import cv2
    from . import hand_contact as hc
    from . import zouk_contacts as zc
    P = paths(take)
    cfg = MagnetCfg(**{k: v for k, v in json.loads((P["out"] / "timeline.json").read_text(encoding="utf-8"))["config"].items()})
    tl = json.loads((P["out"] / "timeline.json").read_text(encoding="utf-8"))
    sites = build_sites(take)
    names = site_names(sites)
    f0, fm = load_fit(P["src_fit"] if ref_fit is None else ref_fit), load_fit(P["fit"])
    obs, phones = hc.Obs(), hc.Phones()
    outs = []
    for ft in film_times:
        t = dance_of(ft)
        k = int(round(t * HZ))
        ks = np.array([k])
        kl0, klm, kf = zc.posed(f0["lead"], ks), zc.posed(fm["lead"], ks), zc.posed(fm["follow"], ks)
        ents = site_entries_world(sites, kf["J"], kf["R"], cfg)
        att_now = {}
        for s in SIDES:
            i = tl["hands"][s]["site_per_frame"][k]
            if i >= 0:
                a = [r for r in tl["hands"][s]["attachments"] if r["k0"] <= k <= r["k1"] and r["site"] == names[i]]
                att_now[s] = (names[i], a[0]["entry"] if a else 0, tl["hands"][s]["w_per_frame"][k])
        arms = []
        for J, tp, col in ((kl0["J"][0], kl0["tips"][0], (255, 255, 255)), (klm["J"][0], klm["tips"][0], (60, 255, 60))):
            for s in SIDES:
                pts = [J[SHOULDER[s]], J[ELBOW[s]], J[WRIST[s]], hc.palm_centre(J[None], s)[0]]
                if tips:                                   # tips: 5 per hand (left first), the middle finger third
                    pts.append(tp[(0 if s == "left" else 5) + 2])
                arms.append((np.stack(pts), col))
        centre = 0.5 * (kl0["J"][0, SPINE3] + kf["J"][0, SPINE3])
        tiles = []
        for c in obs.cams:
            z = obs.z[c]
            if not z["ok"][k]:
                tiles.append(np.full((tile, tile, 3), 30, np.uint8))
                continue
            img, k1, _ = phones.frame(c, k)
            K = z["K"][k]
            uv, d = obs.project(c, centre[None, None], ks)
            cx, cy = uv[0, 0]
            side = max(160.0, extent_m * K[0, 0] / max(d[0, 0], 0.3))
            raw = hc.crop_img(img, cx, cy, side, tile, K, k1) if img is not None else np.full((tile, tile, 3), 40, np.uint8)
            sc = tile / side

            def to_tile(u):
                return (int(round((u[0] - (cx - side / 2)) * sc)), int(round((u[1] - (cy - side / 2)) * sc)))
            held = {v[0]: (sd, v[1]) for sd, v in att_now.items()}
            for nm in names:
                Pw, Nw = ents[nm]
                e = held[nm][1] if nm in held else sites[nm]["default"]
                pu, _ = obs.project(c, Pw[:, e][:, None], ks)
                p = to_tile(pu[0, 0])
                col = SITE_BGR[sites[nm]["kind"]]
                if nm in held:
                    cv2.circle(raw, p, 8, (60, 255, 60), 2, cv2.LINE_AA)
                    cv2.circle(raw, p, 3, col, -1, cv2.LINE_AA)
                    cv2.putText(raw, nm, (p[0] + 9, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(raw, nm, (p[0] + 9, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 255, 60), 1, cv2.LINE_AA)
                else:
                    cv2.circle(raw, p, 3, col, 1, cv2.LINE_AA)
            for X, col in arms:
                pu, _ = obs.project(c, X[None], ks)
                q = [to_tile(x) for x in pu[0]]
                for a_, b_ in zip(q[:-1], q[1:]):
                    cv2.line(raw, a_, b_, col, 2 if col[1] == 255 and col[0] == 60 else 1, cv2.LINE_AA)
                cv2.circle(raw, q[-1], 4, col, -1, cv2.LINE_AA)
            lines = [f"cam {c}  film {mmss(ft)}  dance {t:.2f}"]
            for s in ("right", "left"):
                v = att_now.get(s)
                lines.append(f"his {s}: " + (f"{v[0]} w {v[2]:.2f}" if v else "free"))
            tiles.append(hc.label_tile(raw, lines, 0.45))
        sheet = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:6], 1)], 0)
        p = (P["sheets"] if out_dir is None else Path(out_dir)) / (f"sheet_{int(ft):03d}s.jpg" if ft == int(ft) else f"sheet_{ft:05.1f}s.jpg")
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        outs.append(p)
        log("sheet", p)
    return outs


# ------------------------------------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.magnet", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("sites", "snap", "export", "sheets"))
    ap.add_argument("--take", default="larissa_kadu_recap")
    ap.add_argument("--cfg", default=None, help="JSON overrides of MagnetCfg (snap), e.g. '{\"r_in\": 0.10}'")
    ap.add_argument("--film", default=None, help="sheets: comma-separated film seconds (default 32,57,90,150,163,290)")
    ap.add_argument("--base", default=None, help="the base fit under work/<take>/ (default smplx_rf2_noslip, the detected input)")
    a = ap.parse_args(argv)
    if a.base:
        global BASE_FIT
        BASE_FIT = a.base
    if a.cmd == "sites":
        s = build_sites(a.take, force=True)
        log(f"{len(s)} sites -> {paths(a.take)['out'] / 'sites.json'}, sites_rest.png")
    elif a.cmd == "snap":
        cfg = MagnetCfg(**json.loads(a.cfg)) if a.cfg else MagnetCfg()
        tl = snap(a.take, cfg)
        print(json.dumps({"palm_reached": tl["palm_reached"], "measure": tl.get("measure")}, indent=1))
    elif a.cmd == "export":
        out = export(a.take)
        print(json.dumps({"sha256": out["sha256"], "proof": out["proof_vs_input_export"]}, indent=1))
    else:
        ft = tuple(float(x) for x in a.film.split(",")) if a.film else SHEET_FILM_S
        for p in sheets(a.take, ft):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
