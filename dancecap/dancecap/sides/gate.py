"""The gates of the "sides" step (DESIGN 7.1, 7.2): a solved window is adopted only when every gate passes against the base; otherwise it
is reverted (and flagged). Everything is numpy on the PC except the exact mesh depth (G9), which the solve pod measures for the base and
the candidate of each window (`exact` arrays in the window's npz) and which falls back to zouk_contacts.exact_of here.

    window_gates(ctx, cand, win)          G1-G10 + trust for one window -> {"pass", "gates": {G1: {..., "ok"}}, "failed": [...]}
    take_gate(ctx, merged, adopted)       the whole take after the merge
    calibrate_gate(take)                  DESIGN 7.2 before any solve -> work/<take>/sides/gate_calibration.json

| G | condition (DESIGN 7.1) |
|---|---|
| G1 2D | per camera that sees the moved points on >= 20 % of the window frames (views where they are predicted visible): MAMMA 512 and COCO error of the points the candidate moved (> 1 cm), median <= base + max(1 px, 10 %), p90 <= base + max(3 px, 10 %); pooled AND per moved chain (each arm, each leg, torso + head: a chain's median worse in >= 2 cameras, uncontradicted by the other detector; cfg.gate_per_chain) |
| G2 3D | median distance to the triangulated joints (>= 3 views) <= base + 0.5 cm, pooled AND per moved chain (cfg.gate_per_chain; target: a `real`-only dancer's joints improve by >= 2 cm or 20 % and end <= 8 cm) |
| G3 sides | no more violated decided-relation frames than the base, no relation newly violated (target: <= 2 % of the decided frames) |
| G4 thru | no new GLI pass-through (4 arm pairings) in window + ramps, no more inside held runs than the base (target: none) |
| G5 swing | elbow / wrist reversals >= 3 cm within 1 s <= max(base, triangulated); 1-5 Hz RMS <= max(base, triangulated) + 0.3 cm; swivel acceleration and joint jerk p99 <= 1.2 x base |
| G6 limits | no new frames beyond the wrist / elbow gates (zouk_contacts.GATE) or the twist gates (DESIGN 3.4); no chain wrap |
| G7 speed | no new hand speed incident (hand_contact.speed_violations rules) and no whip (zouk_contacts.whip_frames_added) |
| G8 winding | spin windows: yaw vs psi* p90 <= 25 deg or <= base, no reversal > 90 deg against psi* beyond the base's, face contradictions with >= 2 views <= base, no new torso / foot winding (target: p90 <= 25, 0 contradictions) |
| G9 inside | exact: frames with his hand / forearm > 3 cm in her torso <= base; none > 5 cm unless the base had them (target: <= 5 % of the window, none > 5 cm) |

Every gate is "never worse than the base" (a no-op passes, DESIGN 7.2); the absolute thresholds of DESIGN 7.1 that a no-op cannot meet are
the window's targets (`targets_missed`: an adopted window that misses one is flagged in the report).
| G10 held-out | the held-out phone's COCO error of the moved joints: median <= base + 0.5 px, p90 <= base + 2 px |
| trust | trust-region hits on <= 10 % of the window frames |
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import body as bd
from . import geom
from .adapters import PIDX, adapter_for, project
from .config import SidesCfg, load_cfg
from .evidence import capsule_set, load_tri, load_views, posed_capsules, rest_joints, write_json, yaw_axes

ROLES = ("lead", "follow")
OTHER = {"lead": "follow", "follow": "lead"}
GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "trust")
ARM_J = {"L": {"sh": 16, "el": 18, "wr": 20}, "R": {"sh": 17, "el": 19, "wr": 21}}


def log(*a) -> None:
    print("[sides.gate]", *a, flush=True)


def _r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def descendants(parents, joints) -> np.ndarray:
    """(55,) bool: the joints and every joint below them"""
    out = np.zeros(len(parents), bool)
    js = set(int(j) for j in joints)
    for j in range(len(parents)):
        k = j
        while k >= 0:
            if k in js:
                out[j] = True
                break
            k = int(parents[k])
    return out


def nanpct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else np.nan


# ------------------------------------------------------------------------------------------------ the context (everything read once)

class GateCtx:
    """inputs of every gate: the input fit, the base (sides/cpu), the evidence, the relations, the yaw track, the obs"""

    def __init__(self, take: str, cfg: SidesCfg | None = None, base: dict | None = None):
        from .prepare import load_cpu, load_fit
        self.take = take
        self.cfg = cfg = cfg or load_cfg(take)
        self.ad = ad = adapter_for(take, cfg)
        self.work = ad.work
        self.sd = ad.work / "sides"
        self.T, self.hz, self.cams = ad.T, cfg.hz, ad.cams
        self.input = {r: load_fit(ad.base_dir / f"{r}.npz") for r in ROLES}
        self.base = base if base is not None else (load_cpu(self.work) if (self.sd / "cpu" / "lead.npz").exists() else self.input)
        for f in list(self.input.values()) + list(self.base.values()):
            if "joints" not in f:                       # the pods' slim copies carry no joints: FK them once
                f["joints"] = np.zeros((len(f["t"]), 55, 3))
                f["joints"][:] = bd.forward(f, points=False)["J"]
        self.tri = {r: load_tri(self.work, r) for r in ROLES}
        self.tri_ho = {r: load_tri(self.work, r, heldout=True) for r in ROLES}
        self.views = load_views(self.work, self.cams)
        self.obs = ad.obs()
        self.rel = dict(np.load(self.sd / "relations.npz"))
        self.yaw = dict(np.load(self.sd / "yaw.npz"))
        bj = json.loads((self.sd / "evidence" / "coco_bias.json").read_text(encoding="utf-8"))
        self.bias = {r: np.asarray(bj["bias"][r], float) for r in ROLES}
        betas = {r: np.asarray(self.input[r]["betas"], float) for r in ROLES}
        self.caps = capsule_set(betas, self.sd / "evidence" / "capsules.json")
        self.rest = rest_joints(betas, self.sd / "evidence" / "rest.json")
        self.parents = bd.model(self.input["lead"])["parents"]
        self.Bw = {r: bd.model(self.input[r])["Bw"] for r in ROLES}
        self._lm = {}
        self._pin = None
        self._speed_ref = None
        self.exact_cache = {}

    def lm(self, cam: str) -> np.ndarray:
        if cam not in self._lm:
            self._lm[cam] = self.ad.landmarks(cam)
        return self._lm[cam]

    def speed_ref(self) -> dict:
        """p99 of the input's hand / elbow / ankle speeds over the whole take (hand_contact.input_speed_ref, more points)"""
        if self._speed_ref is None:
            out = {}
            for r in ROLES:
                J = np.asarray(self.input[r]["joints"], float)
                for nm, X in track_points(J).items():
                    out[f"{r}_{nm}"] = float(np.percentile(np.linalg.norm(np.diff(X, axis=0), axis=1) * self.hz, 99))
            self._speed_ref = out
        return self._speed_ref


def track_points(J: np.ndarray) -> dict:
    """the points the speed / whip gate watches: wrists, palm centres, elbows, ankles, feet"""
    return {"wrist_L": J[:, 20], "wrist_R": J[:, 21], "palm_L": geom.palm_from_joints(J, "left"), "palm_R": geom.palm_from_joints(J, "right"),
            "elbow_L": J[:, 18], "elbow_R": J[:, 19], "ankle_L": J[:, 7], "ankle_R": J[:, 8], "foot_L": J[:, 10], "foot_R": J[:, 11]}


# ------------------------------------------------------------------------------------------------ candidate / fits

def fit_window(base: dict, P: np.ndarray, transl: np.ndarray | None, ks: np.ndarray) -> dict:
    """a shallow copy of one dancer's fit with pose55 P and translation of frames ks replaced and its joints recomputed there"""
    out = dict(base)
    for k in (*bd.POSE_KEYS, "transl", "joints"):
        if k in base:
            out[k] = np.array(base[k], copy=True)
    bd.write_pose55(out, P, ks)
    if transl is not None:
        out["transl"][ks] = transl
    J = bd.forward(out, ks, points=False)["J"]
    out["joints"][ks, :55] = J
    return out


# ------------------------------------------------------------------------------------------------ per-gate metrics

def coco_pred(ctx: GateCtx, role: str, J55: np.ndarray) -> np.ndarray:
    """(n,17,3) the COCO points a fit predicts: SMPL-X joints (geom.C2S) + the per-role bias in the dancer's yaw frame"""
    return J55[:, list(geom.C2S)] + np.einsum("kij,pj->kpi", yaw_axes(J55), ctx.bias[role])


CHAINS = ("arm_L", "arm_R", "leg_L", "leg_R", "torso")


def joint_chain() -> np.ndarray:
    """(55,) chain index (CHAINS) of each SMPL-X joint: each arm = shoulder, elbow, wrist + that hand's fingers; each leg = hip, knee,
    ankle, foot; torso = pelvis, spine, neck, collars, head, jaw, eyes"""
    ch = np.full(55, CHAINS.index("torso"))
    ch[[16, 18, 20, *range(25, 40)]] = CHAINS.index("arm_L")
    ch[[17, 19, 21, *range(40, 55)]] = CHAINS.index("arm_R")
    ch[[1, 4, 7, 10]] = CHAINS.index("leg_L")
    ch[[2, 5, 8, 11]] = CHAINS.index("leg_R")
    return ch


def point_chains(Bw: np.ndarray) -> np.ndarray:
    """(512,) chain index of each MAMMA point: the chain of its dominant skinning joint"""
    return joint_chain()[np.argmax(np.asarray(Bw), 1)]


def coco_chains() -> np.ndarray:
    """(17,) chain index of each COCO point (the face points belong to the torso / head)"""
    return joint_chain()[list(geom.C2S)]


def _frame_median(e: np.ndarray, min_n: int) -> np.ndarray:
    """(n,) per-frame nanmedian of (n, m) errors, NaN where fewer than min_n are finite"""
    import warnings
    if e.shape[1] == 0:
        return np.full(len(e), np.nan)
    n = np.isfinite(e).sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.where(n >= max(1, min_n), np.nanmedian(e, 1), np.nan)


def g1_compare(eb: dict, ec: dict, cfg: SidesCfg, cols: dict | None = None, n_ref: int | None = None) -> dict:
    """one camera's G1 verdict from per-(frame, point) errors. eb / ec {"mamma": (n, m), "coco": (n, q)} of the base / candidate;
    cols {"mamma": idx, "coco": idx} the points to pool (default all). Per frame the median over the visible points (MAMMA needs
    >= gate_g1_min_points), then median and p90 over the frames; a kind counts when the base has a value on >= gate_g1_min_share of the
    frames (n_ref: the frames the share is counted over, default all): median <= base + max(gate_g1_med_px, gate_g1_rel x base), p90 <=
    base + max(gate_g1_p90_px, gate_g1_rel x base)"""
    out = {}
    ok = True
    for kind, min_n in (("mamma", cfg.gate_g1_min_points), ("coco", 1)):
        c_ = None if cols is None else np.asarray(cols.get(kind, []), int)
        b = eb[kind] if c_ is None else eb[kind][:, c_]
        c = ec[kind] if c_ is None else ec[kind][:, c_]
        fb, fc = _frame_median(b, min_n), _frame_median(c, min_n)
        share = float(np.isfinite(fb).sum() / max(1, n_ref if n_ref is not None else len(fb)))
        out[f"share_{kind}"] = round(share, 2)
        if share < cfg.gate_g1_min_share:
            continue
        bm, cm, bp, cp = nanpct(fb, 50), nanpct(fc, 50), nanpct(fb, 90), nanpct(fc, 90)
        okk = (cm <= bm + max(cfg.gate_g1_med_px, cfg.gate_g1_rel * bm)) and (cp <= bp + max(cfg.gate_g1_p90_px, cfg.gate_g1_rel * bp))
        out[kind] = {"base": [_r(bm), _r(bp)], "cand": [_r(cm), _r(cp)], "ok": bool(okk)}
        ok &= okk
    out["ok"] = bool(ok)
    return out


def chain_cam_worse(oc: dict, cfg: SidesCfg) -> bool:
    """one camera's verdict on one moved chain (g1_compare output on the (frame, point) pairs that moved): worse when a detector's MEDIAN
    rises beyond max(gate_chain_med_px, gate_g1_rel x base) and no detector's median there falls beyond it (MAMMA and COCO contradicting
    each other in a view is not evidence). Medians only: a chain is a few points, its per-camera p90 is noise (the pooled check keeps
    the p90)"""
    worse = better = False
    for kind in ("mamma", "coco"):
        if kind not in oc:
            continue
        bm, cm = oc[kind]["base"][0], oc[kind]["cand"][0]
        if bm is None or cm is None:
            continue
        tol = max(cfg.gate_chain_med_px, cfg.gate_g1_rel * bm)
        worse |= cm > bm + tol
        better |= cm < bm - tol
    return bool(worse and not better)


R_HAND_J = (21, *range(40, 55))          # his right wrist + right-hand finger joints (SMPL-X)
COCO_WR_R = 10                           # COCO right wrist


def r4_classes(ctx: GateCtx, fits_r: dict, ks: np.ndarray) -> np.ndarray:
    """(n,) the palm class (decide.r4_class_base: 1 front, 2 flank, 3 back, 0 unknown) of his right palm under the fit pair fits_r"""
    from .decide import r4_class_base
    if not hasattr(ctx, "_hw"):
        ctx._hw = float(json.loads((ctx.sd / "relations.json").read_text(encoding="utf-8"))["relations"]["R4"].get("half_width_m", 0.14))
    M = margins_of(fits_r, ks, ctx._hw)
    return r4_class_base(np.asarray(M["R4_fb"]["m"], float), np.asarray(M["R4_fl"]["m"], float), ctx.cfg)


def r_hand_hidden(ctx: GateCtx, fits_r: dict, ks: np.ndarray) -> dict:
    """per frame and view (n, V) whether his right hand and his distal forearm are HIDDEN under the fit pair fits_r: the two-body ray test
    (evidence.visible, both dancers' body capsules posed on these fits, his own right forearm / hand excluded, cfg.occl_margin_m) at his
    palm centre and wrist ("hand": both hidden) and at 3/4 of his forearm ("forearm"); a camera without the frame is not "hidden"
    (it has no detection to drop) -> {"hand", "forearm"} bool (n, V)"""
    from .evidence import PALM_EXCL, exclusion, visible
    cfg = ctx.cfg
    JL, JF = (np.asarray(fits_r[r]["joints"], float)[ks] for r in ROLES)
    RgL = geom.global_rotations(np.asarray(fits_r["lead"]["global_orient"], float)[ks], np.asarray(fits_r["lead"]["body_pose"], float)[ks])
    RgF = geom.global_rotations(np.asarray(fits_r["follow"]["global_orient"], float)[ks], np.asarray(fits_r["follow"]["body_pose"], float)[ks])
    A1, B1, r1, pt1 = posed_capsules(ctx.caps["lead"], JL, RgL)
    A2, B2, r2, pt2 = posed_capsules(ctx.caps["follow"], JF, RgF)
    OA, OB = np.concatenate([A1, A2], 1), np.concatenate([B1, B2], 1)
    rad = np.concatenate([r1, r2]) * cfg.occl_r_scale
    X = np.stack([geom.palm_from_joints(JL, "right"), JL[:, 21], JL[:, 19] + 0.75 * (JL[:, 21] - JL[:, 19])], 1)
    ex = exclusion(pt1, pt2, [PALM_EXCL["right"]] * 3)
    vis = visible(ctx.obs, ctx.cams, X, OA, OB, rad, ex, cfg.occl_margin_m, frames=ks)          # (n, 3, V)
    ok = np.stack([np.asarray(ctx.obs[c]["ok"], bool)[ks] for c in ctx.cams], 1)
    return {"hand": ok & ~vis[:, 0] & ~vis[:, 1], "forearm": ok & ~vis[:, 2]}


def r4_proxy_hidden(base: dict, caps: dict, sites_f: dict, obs: dict, cams: list, cfg: SidesCfg, ks: np.ndarray, cls: np.ndarray) -> dict:
    """(n, V) the views the DECIDED class hides his right hand / distal forearm from, before any solve (the solver's evidence rule,
    cfg.solve_r4_vis): per decided frame his palm on her site of that class nearest the base palm (palm_lift_m off her surface, as
    decide.r4_visibility_test), the wrist palm_from_wrist_m back toward his base elbow, the distal forearm 3/4 of the way from the elbow;
    the two-body ray test on both dancers' base capsules (his right forearm / hand excluded) -> {"hand" (palm and wrist hidden),
    "forearm"} bool (n, V), False on undecided frames and where a camera has no frame"""
    from .. import zouk_contacts as zc
    from .decide import CLASSES, SITE_CLASS
    from .evidence import PALM_EXCL, exclusion, visible
    n, V = len(ks), len(cams)
    out = {"hand": np.zeros((n, V), bool), "forearm": np.zeros((n, V), bool)}
    cls = np.asarray(cls, int)
    if not (cls != 0).any():
        return out
    JL, JF = (np.asarray(base[r]["joints"], float)[ks] for r in ROLES)
    RgL = geom.global_rotations(np.asarray(base["lead"]["global_orient"], float)[ks], np.asarray(base["lead"]["body_pose"], float)[ks])
    RgF = geom.global_rotations(np.asarray(base["follow"]["global_orient"], float)[ks], np.asarray(base["follow"]["body_pose"], float)[ks])
    RF55 = np.zeros((n, 55, 3, 3))
    RF55[:, :22] = RgF
    RF55[:, 22:] = np.eye(3)
    A1, B1, r1, pt1 = posed_capsules(caps["lead"], JL, RgL)
    A2, B2, r2, pt2 = posed_capsules(caps["follow"], JF, RgF)
    OA, OB = np.concatenate([A1, A2], 1), np.concatenate([B1, B2], 1)
    rad = np.concatenate([r1, r2]) * cfg.occl_r_scale
    ex = exclusion(pt1, pt2, [PALM_EXCL["right"]] * 3)
    palm_b = geom.palm_from_joints(JL, "right")
    ok = np.stack([np.asarray(obs[c]["ok"], bool)[ks] for c in cams], 1)
    for h in (1, 2, 3):
        fr = np.flatnonzero(cls == h)
        if not len(fr):
            continue
        P, N = [], []
        for nm in SITE_CLASS[CLASSES[h - 1]]:
            if nm in sites_f:
                p, nn = zc.site_world(sites_f[nm], JF[fr], RF55[fr])
                P.append(p)
                N.append(nn)
        if not P:
            continue
        P, N = np.stack(P, 1), np.stack(N, 1)
        k = np.argmin(np.linalg.norm(P - palm_b[fr][:, None], axis=-1), 1)
        palm = P[np.arange(len(fr)), k] + cfg.palm_lift_m * N[np.arange(len(fr)), k]
        u = palm - JL[fr, 19]
        wr = palm - cfg.palm_from_wrist_m * u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        fo = JL[fr, 19] + 0.75 * (wr - JL[fr, 19])
        vis = visible(obs, cams, np.stack([palm, wr, fo], 1), OA[fr], OB[fr], rad, ex, cfg.occl_margin_m, frames=ks[fr])
        out["hand"][fr] = ok[fr] & ~vis[:, 0] & ~vis[:, 1]
        out["forearm"][fr] = ok[fr] & ~vis[:, 2]
    return out


def r4_visibility(ctx: GateCtx, fits: dict, ks: np.ndarray) -> dict | None:
    """the visibility-consistent evidence of a window with a DECIDED R4 class (cameras, crops or D1 blind): on the decided frames where the
    candidate's palm is ON the decided class, the views where the candidate predicts his right hand (palm + wrist) / distal forearm
    hidden (r_hand_hidden: behind her back, under her arm) - a detection there is a hallucination of the hidden hand and no evidence
    against the decided side, so G1 / G10 drop his hand / wrist / distal forearm detections of those view-frames and G2 his triangulated
    wrist where a hidden view is one of its inliers. Off-class or undecided frames keep every detection. None when the window has no
    decided R4 frame (or cfg.gate_vis_r4 is off) -> {"drop_hand", "drop_forearm", "cand_vis_hand" (n, V), "report"}"""
    cfg = ctx.cfg
    if not cfg.gate_vis_r4:
        return None
    cls = np.asarray(ctx.rel["R4__cls"], int)[ks]
    dec = cls != 0
    if not dec.any():
        return None
    cc = r4_classes(ctx, fits["cand"], ks)
    on = dec & (cc == cls)
    hc = r_hand_hidden(ctx, fits["cand"], ks)
    hb = r_hand_hidden(ctx, fits["base"], ks)
    dh = hc["hand"] & on[:, None]
    df = hc["forearm"] & on[:, None]
    rep = {"decided_frames": int(dec.sum()), "cand_on_class_frames": int(on.sum()),
           "base_on_class_frames": int((dec & (r4_classes(ctx, fits["base"], ks) == cls)).sum()),
           "hand_hidden_view_frames": {"cand": int(dh.sum()), "base_too": int((dh & hb["hand"]).sum())},
           "forearm_hidden_view_frames": {"cand": int(df.sum()), "base_too": int((df & hb["forearm"]).sum())},
           "per_cam_hand": {c: int(dh[:, v].sum()) for v, c in enumerate(ctx.cams)}}
    ok = np.stack([np.asarray(ctx.obs[c]["ok"], bool)[ks] for c in ctx.cams], 1)
    return {"drop_hand": dh, "drop_forearm": df, "cand_vis_hand": on[:, None] & ok & ~hc["hand"], "report": rep}


def r_arm_groups(Bw: np.ndarray, P: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(512,) masks of the MAMMA points of his right hand (dominant skinning joint: the right wrist / a right-hand finger) and of his
    DISTAL right forearm (dominant joint the right elbow, past half way to the wrist: the median over frames of the point's parameter
    along elbow -> wrist), from posed points P (n, 512, 3) and joints J (n, 55, 3) of the same frames"""
    dj = np.argmax(np.asarray(Bw), 1)
    hand = np.isin(dj, R_HAND_J)
    fore = np.zeros(len(dj), bool)
    idx = np.flatnonzero(dj == 19)
    if len(idx) and len(P):
        ax = J[:, 21] - J[:, 19]
        s = np.einsum("nkd,nd->nk", P[:, idx] - J[:, None, 19], ax) / np.maximum((ax ** 2).sum(-1), 1e-9)[:, None]
        fore[idx] = np.median(s, 0) > 0.5
    return hand, fore


def g1_g10(ctx: GateCtx, win: dict, fits: dict, ks: np.ndarray, moved: dict, vr: dict | None = None) -> dict:
    """per camera MAMMA / COCO errors of the affected points (G1) and the held-out phone's COCO errors (G10), base and candidate.
    G1 is checked pooled over every moved point (per camera and detector, median and p90) and (cfg.gate_per_chain) per moved chain
    (CHAINS): a chain fails when its moved points fit worse than the base's in >= gate_chain_min_cams cameras (chain_cam_worse), even
    when the pooled median hides it (a spine3 move pools the torso and both arms). vr (r4_visibility): on decided-R4 frames with the
    candidate on the decided class, the view-frames where the candidate predicts his right hand / distal forearm hidden contribute no
    detection of those points (to base and candidate alike); the COCO right wrist there counts where the candidate predicts it visible"""
    cfg = ctx.cfg
    g1 = {"cams": {}, "chains": {}, "failed_chains": [], "ok": True}
    g10 = {"ok": True}
    cc_coco = coco_chains()
    for role, js in moved.items():
        p = PIDX[role]
        ho = ctx.tri[role]["ho"][ks]
        hoe = {"base": [], "cand": []}
        full = {}
        Jb_ = None
        for nm in ("base", "cand"):
            o = bd.forward(fits[nm][role], ks)
            full[nm] = (o["pts"], coco_pred(ctx, role, o["J"]))
            if nm == "base":
                Jb_ = o["J"]
        # the points the candidate really moved (> gate_moved_m somewhere in the window): an unchanged point would only dilute the error
        pts_aff = np.flatnonzero(np.linalg.norm(full["cand"][0] - full["base"][0], axis=-1).max(0) > cfg.gate_moved_m)
        coco_aff = np.flatnonzero(np.linalg.norm(full["cand"][1] - full["base"][1], axis=-1).max(0) > cfg.gate_moved_m)
        if not len(pts_aff) and not len(coco_aff):
            g10[role] = {"n": 0, "ok": True, "note": "nothing moved"}
            continue
        pch = point_chains(ctx.Bw[role])[pts_aff]
        cch = cc_coco[coco_aff]
        # per frame: which of the affected points moved there (the chain check reads only those (frame, point) pairs: frames where a
        # point did not move carry the same error in base and candidate and would dilute the change)
        mv_p = np.linalg.norm(full["cand"][0][:, pts_aff] - full["base"][0][:, pts_aff], axis=-1) > cfg.gate_moved_m
        mv_c = np.linalg.norm(full["cand"][1][:, coco_aff] - full["base"][1][:, coco_aff], axis=-1) > cfg.gate_moved_m
        chain_cols = {CHAINS[k]: {"mamma": np.flatnonzero(pch == k), "coco": np.flatnonzero(cch == k)}
                      for k in sorted(set(pch.tolist()) | set(cch.tolist()))}
        res = {nm: (full[nm][0][:, pts_aff], full[nm][1][:, coco_aff]) for nm in ("base", "cand")}
        # decided R4 (his right hand): the hand / wrist / distal-forearm points of the view-frames where the candidate hides the hand
        vrr = vr if (vr is not None and role == "lead") else None
        if vrr is not None:
            hand_m, fore_m = r_arm_groups(ctx.Bw[role], full["base"][0], Jb_)
            hand_c, fore_c = hand_m[pts_aff], fore_m[pts_aff]
            wr_c = coco_aff == COCO_WR_R
            vrep = {"mamma_view_frames": 0, "coco_wrist_view_frames": 0, "coco_wrist_added": 0, "per_cam": {}}
        for ci, c in enumerate(ctx.cams):
            ob = ctx.obs[c]
            okc = ob["ok"][ks]
            lm = ctx.lm(c)[ks, p][:, pts_aff].astype(float)
            vis = (ob["vis"][ks, p][:, pts_aff].astype(float) > 0.5) & (np.abs(lm).sum(-1) > 0) & okc[:, None]
            vw = ctx.views[c]
            cu = vw["uv"][ks, p][:, coco_aff].astype(float)
            cc = vw["conf"][ks, p][:, coco_aff].astype(float)
            vp = vw["vis_pred"][ks, p][:, coco_aff].astype(bool)
            cd0 = (cc >= 0.5) & (vw["det"][ks, p].astype(float) >= cfg.det_conf)[:, None] & okc[:, None] & np.isfinite(cu[..., 0])
            cdet = cd0 & vp
            if vrr is not None:
                dh, dfo = vrr["drop_hand"][:, ci], vrr["drop_forearm"][:, ci]
                vis2 = vis & ~((dh[:, None] & hand_c[None]) | (dfo[:, None] & fore_c[None]))
                # on the on-class decided frames the wrist counts where the CANDIDATE predicts it visible (the base's vis_pred elsewhere)
                on_c = (dh | vrr["cand_vis_hand"][:, ci])[:, None] & wr_c[None]
                cdet2 = np.where(on_c, cd0 & vrr["cand_vis_hand"][:, ci][:, None], cdet)
                nm_ = int((vis & ~vis2).any(1).sum())
                nc_ = int((cdet & ~cdet2).any(1).sum())
                na_ = int((cdet2 & ~cdet).any(1).sum())
                vrep["mamma_view_frames"] += nm_
                vrep["coco_wrist_view_frames"] += nc_
                vrep["coco_wrist_added"] += na_
                if nm_ or nc_ or na_:
                    vrep["per_cam"][c] = {"mamma": nm_, "coco_wrist": nc_, "coco_wrist_added": na_}
                vis, cdet = vis2, cdet2
            err = {}
            for nm in ("base", "cand"):
                P3, C3 = res[nm]
                uv, dep = project(ob["K"][ks], ob["R"][ks], ob["t"][ks], P3)
                e = np.where(vis & (dep > 0.1), np.linalg.norm(uv - lm, axis=-1), np.nan)
                uvc, depc = project(ob["K"][ks], ob["R"][ks], ob["t"][ks], C3)
                ec = np.where(cdet & (depc > 0.1), np.linalg.norm(uvc - cu, axis=-1), np.nan)
                err[nm] = {"mamma": e, "coco": ec}
                hoe[nm].append(np.where(ho == ci, _frame_median(ec, 1), np.nan))
            out = g1_compare(err["base"], err["cand"], cfg)
            g1["cams"][f"{role}_{c}"] = out
            g1["ok"] &= out["ok"]
            if cfg.gate_per_chain_any or len(chain_cols) > 1:
                em = {nm: {"mamma": np.where(mv_p, err[nm]["mamma"], np.nan), "coco": np.where(mv_c, err[nm]["coco"], np.nan)}
                      for nm in ("base", "cand")}
                for chn, cols in chain_cols.items():
                    n_ref = int((mv_p[:, cols["mamma"]].any(1) | mv_c[:, cols["coco"]].any(1)).sum())
                    oc = g1_compare(em["base"], em["cand"], cfg, cols, n_ref=n_ref)
                    if "mamma" not in oc and "coco" not in oc:
                        continue
                    oc["worse"] = chain_cam_worse(oc, cfg)
                    g1["chains"].setdefault(f"{role}_{chn}", {})[c] = oc
        if vrr is not None:
            g1["vis_r4"] = {**vrr["report"], "dropped_detections": vrep}
        # a moved chain fails when its median error rises in >= gate_chain_min_cams cameras (chain_cam_worse)
        for chn in chain_cols:
            rows = g1["chains"].get(f"{role}_{chn}", {})
            worse = [c for c, oc in rows.items() if oc.get("worse")]
            if len(worse) >= cfg.gate_chain_min_cams:
                g1["failed_chains"].append(f"{role}_{chn}: " + ",".join(worse))
                if cfg.gate_per_chain:
                    g1["ok"] = False
        b = np.nanmin(np.stack(hoe["base"]), 0) if hoe["base"] else np.array([np.nan])
        c_ = np.nanmin(np.stack(hoe["cand"]), 0) if hoe["cand"] else np.array([np.nan])
        n = int(np.isfinite(b).sum())
        if n >= cfg.gate_g10_min_frames:
            bm, cm, bp, cp = nanpct(b, 50), nanpct(c_, 50), nanpct(b, 90), nanpct(c_, 90)
            okk = (cm <= bm + cfg.gate_g10_med_px) and (cp <= bp + cfg.gate_g10_p90_px)
            g10[role] = {"n": n, "base": [_r(bm), _r(bp)], "cand": [_r(cm), _r(cp)], "ok": bool(okk)}
            g10["ok"] &= bool(okk)
        else:
            g10[role] = {"n": n, "ok": True, "note": "held-out phone undefined / too few frames"}
    g1["ok"] = bool(g1["ok"])
    return {"G1": g1, "G10": g10}


def g2(ctx: GateCtx, win: dict, fits: dict, ks: np.ndarray, moved: dict, vr: dict | None = None) -> dict:
    """median distance of the moved COCO joints (>= 3 views) to the triangulated ones <= base + gate_g2_m, pooled and (cfg.gate_per_chain)
    per moved chain with >= gate_g2_chain_min_n samples. vr (r4_visibility): his triangulated right wrist does not count on a decided-R4
    frame (candidate on the class) where one of its inlier views is a view the candidate hides the hand from (a hallucinated detection
    pulled the triangulation)"""
    cfg = ctx.cfg
    out = {"ok": True, "failed_chains": []}
    cc_coco = coco_chains()
    for role, js in moved.items():
        Jb_, Jc_ = (np.asarray(fits[nm][role]["joints"], float)[ks] for nm in ("base", "cand"))
        coco_aff = [j for j in range(5, 17) if np.linalg.norm(Jc_[:, geom.C2S[j]] - Jb_[:, geom.C2S[j]], axis=-1).max() > cfg.gate_moved_m]
        tr = ctx.tri[role]
        ok3 = tr["nv"][ks][:, coco_aff] >= cfg.min_views_real
        if vr is not None and role == "lead" and COCO_WR_R in coco_aff:
            ci_ = coco_aff.index(COCO_WR_R)
            bad = (np.asarray(tr["inl"], bool)[ks][:, COCO_WR_R] & vr["drop_hand"]).any(1)
            out["vis_r4_wrist_dropped"] = int((ok3[:, ci_] & bad).sum())
            ok3[:, ci_] &= ~bad
        X = tr["Xd"][ks][:, coco_aff].astype(float)
        d = {}
        for nm in ("base", "cand"):
            J = np.asarray(fits[nm][role]["joints"], float)[ks]
            d[nm] = np.where(ok3, np.linalg.norm(J[:, [geom.C2S[j] for j in coco_aff]] - X, axis=-1), np.nan)
        bm, cm = (nanpct(d["base"], 50), nanpct(d["cand"], 50)) if coco_aff else (np.nan, np.nan)
        row = {"n": int(np.isfinite(d["base"]).sum()), "joints": [geom.COCO[j] for j in coco_aff], "base_cm": _r(bm * 100),
               "cand_cm": _r(cm * 100)}
        okk = (not np.isfinite(bm)) or cm <= bm + cfg.gate_g2_m
        # per moved chain, on the (frame, joint) pairs that moved (as G1's chain check)
        chains = {}
        cch = cc_coco[coco_aff] if coco_aff else np.zeros(0, int)
        mv = np.linalg.norm(Jc_[:, [geom.C2S[j] for j in coco_aff]] - Jb_[:, [geom.C2S[j] for j in coco_aff]], axis=-1) > cfg.gate_moved_m             if coco_aff else np.zeros((len(ks), 0), bool)
        if cfg.gate_per_chain_any or len(set(cch.tolist())) > 1:
            for k in sorted(set(cch.tolist())):
                cols = np.flatnonzero(cch == k)
                db, dc = np.where(mv[:, cols], d["base"][:, cols], np.nan), np.where(mv[:, cols], d["cand"][:, cols], np.nan)
                n = int(np.isfinite(db).sum())
                if n < cfg.gate_g2_chain_min_n:
                    continue
                b_, c__ = nanpct(db, 50), nanpct(dc, 50)
                okc = bool(c__ <= b_ + cfg.gate_g2_m)
                chains[CHAINS[k]] = {"n": n, "joints": [geom.COCO[coco_aff[i]] for i in cols], "base_cm": _r(b_ * 100),
                                     "cand_cm": _r(c__ * 100), "ok": okc}
                if not okc:
                    out["failed_chains"].append(f"{role}_{CHAINS[k]}")
                    if cfg.gate_per_chain:
                        okk = False
        row["chains"] = chains
        # real-only dancers: every target joint improves by >= 2 cm or 20 % and ends <= 8 cm
        dz = win["dancers"].get(role, {})
        items = []
        tgt = True
        if set(dz.get("reasons", [])) == {"real"}:
            for it in dz.get("real_items", []):
                j = geom.CI.get(it["joint"])
                if j is None:
                    continue
                kk = np.arange(max(it["k0"], ks[0]), min(it["k1"], ks[-1]) + 1)
                okj = tr["nv"][kk, j] >= cfg.min_views_real
                if okj.sum() < 3:
                    continue
                Xj = tr["Xd"][kk, j].astype(float)
                mb = float(np.median(np.linalg.norm(np.asarray(fits["base"][role]["joints"], float)[kk, geom.C2S[j]] - Xj, axis=1)[okj]))
                mc = float(np.median(np.linalg.norm(np.asarray(fits["cand"][role]["joints"], float)[kk, geom.C2S[j]] - Xj, axis=1)[okj]))
                good = ((mb - mc) >= min(cfg.gate_g2_real_gain_m, cfg.gate_g2_real_gain_rel * mb)) and mc <= cfg.gate_g2_real_end_m
                items.append({"joint": it["joint"], "base_cm": _r(mb * 100), "cand_cm": _r(mc * 100), "target": bool(good)})
                okk &= mc <= mb + cfg.gate_g2_m
                tgt &= good
        row["real_items"] = items
        row["ok"] = bool(okk)
        out[role] = row
        out["ok"] &= bool(okk)
        out["target"] = bool(out.get("target", True) and tgt)
    return out


def margins_of(fits_r: dict, ks: np.ndarray, hw: float) -> dict:
    JL = np.asarray(fits_r["lead"]["joints"], float)[ks]
    JF = np.asarray(fits_r["follow"]["joints"], float)[ks]
    return geom.margins(geom.points_from_smplx(JL), geom.points_from_smplx(JF), half_width_m=hw)


def g3(ctx: GateCtx, win: dict, fits: dict, ks: np.ndarray) -> dict:
    from .decide import r4_class_base
    cfg = ctx.cfg
    rel = ctx.rel
    hw = float(json.loads((ctx.sd / "relations.json").read_text(encoding="utf-8"))["relations"]["R4"].get("half_width_m", 0.14)) \
        if not hasattr(ctx, "_hw") else ctx._hw
    ctx._hw = hw
    M = {nm: margins_of(fits[nm], ks, hw) for nm in ("base", "cand")}
    per = {}
    n_dec = 0
    tot = {"base": 0, "cand": 0}
    new = []
    for rid, (kind, dz, _, _) in geom.REL_TABLE.items():
        s = rel[f"{rid}__s"][ks].astype(float)
        dec = s != 0
        if not dec.any():
            continue
        n_dec += int(dec.sum())
        v = {}
        for nm in ("base", "cand"):
            m = np.asarray(M[nm][rid]["m"], float)
            with np.errstate(invalid="ignore"):
                v[nm] = int((dec & (s * m < -dz)).sum())
            tot[nm] += v[nm]
        per[rid] = v
        if v["base"] == 0 and v["cand"] > 0:
            new.append(rid)
    cls = rel["R4__cls"][ks]
    if (cls != 0).any():
        n_dec += int((cls != 0).sum())
        v = {}
        for nm in ("base", "cand"):
            fb, fl = np.asarray(M[nm]["R4_fb"]["m"], float), np.asarray(M[nm]["R4_fl"]["m"], float)
            cb = r4_class_base(fb, fl, cfg)
            clear = ((cls == 1) & (cb == 3) & (fb < -cfg.r4_fb_dz_m)) | ((cls == 3) & (cb == 1) & (fb > cfg.r4_fb_dz_m)) \
                | ((cls == 2) & (fl < -2 * cfg.r4_fl_dz_m)) | (((cls == 1) | (cls == 3)) & (cb == 2) & (fl > 2 * cfg.r4_fl_dz_m))
            v[nm] = int(clear.sum())
            tot[nm] += v[nm]
        per["R4"] = v
        if v["base"] == 0 and v["cand"] > 0:
            new.append("R4")
    lim = cfg.gate_g3_share * max(n_dec, 1)
    ok = tot["cand"] <= tot["base"] and not new
    target = tot["cand"] <= lim and (tot["cand"] < tot["base"] or tot["cand"] == 0)
    return {"decided_frames": n_dec, "violated": tot, "per_relation": per, "newly_violated": new, "ok": bool(ok), "target": bool(target)}


def gli_series(fits_r: dict, ks: np.ndarray) -> dict:
    JL = np.asarray(fits_r["lead"]["joints"], float)[ks]
    JF = np.asarray(fits_r["follow"]["joints"], float)[ks]
    out = {}
    for rid, (sa, sb) in geom.R9_PAIRS.items():
        a = [17, 19, 21] if sa == "R" else [16, 18, 20]
        b = [17, 19, 21] if sb == "R" else [16, 18, 20]
        out[rid] = geom.gli_chain(JL[:, a], JF[:, b])
    return out


def g4(ctx: GateCtx, win: dict, fits: dict, ks: np.ndarray) -> dict:
    G = {nm: gli_series(fits[nm], ks) for nm in ("base", "cand")}
    out = {"ok": True}
    for rid in geom.R9_PAIRS:
        eb = geom.passthrough_events(G["base"][rid], ctx.cfg.thru_jump)
        ec = geom.passthrough_events(G["cand"][rid], ctx.cfg.thru_jump)
        s = ctx.rel[f"{rid}__s"][ks]
        held = [int(k) for k in ec if (s[max(0, k - 5):k + 5] != 0).any()]
        held_b = [int(k) for k in eb if (s[max(0, k - 5):k + 5] != 0).any()]
        ok = len(ec) <= len(eb) and len(held) <= len(held_b)
        out[rid] = {"base": len(eb), "cand": len(ec), "base_held": len(held_b), "cand_held": len(held),
                    "cand_t": [round(float(ks[k]) / ctx.hz, 2) for k in ec][:10], "ok": bool(ok)}
        out["ok"] &= bool(ok)
        out["target"] = bool(out.get("target", True) and not held)
    return out


def g5(ctx: GateCtx, win: dict, fits: dict, ks: np.ndarray, moved: dict, kw: np.ndarray) -> dict:
    """swing / smoothness on the frames ks (window + ramps + a margin), reversals and band energy on kw (window + ramps)"""
    from .report import band_rms_windows, swings_within
    cfg = ctx.cfg
    hz = ctx.hz
    out = {"ok": True}
    spin = {r for r, d in win["dancers"].items() if d.get("kind") == "spin" or "spin" in d.get("reasons", [])}
    for role, js in moved.items():
        aff = descendants(ctx.parents, js)
        Jb, Jc = (np.asarray(fits[nm][role]["joints"], float)[ks] for nm in ("base", "cand"))
        sel = np.isin(ks, kw)
        tr = ctx.tri[role]
        ratio = cfg.gate_g5_ratio_spin if role in spin else cfg.gate_g5_ratio
        for side in ("L", "R"):
            a = ARM_J[side]
            own = {13 if side == "L" else 14, a["sh"], a["el"], a["wr"]}
            if not own & set(int(j) for j in js):
                continue                        # the arm only rides along (a spin window's turn): G8 / G7 gate it
            hS = Jb[:, a["sh"], 1]
            for jn in ("el", "wr"):
                if role in spin:
                    continue
                j = a[jn]
                xb, xc = Jb[:, j, 1] - hS, Jc[:, j, 1] - hS
                # a new swing must clear the 3-cm rule by a margin the base / the real joint do not reach (a 2.9 -> 3.1 cm wobble of an
                # existing up-and-down is not a new swing): candidate at 3 + m, base and triangulated at 3 - m
                m_ = cfg.gate_g5_rev_margin_m
                rb, rc = swings_within(xb[sel], 0.03 - m_, 30), swings_within(xc[sel], 0.03 + m_, 30)
                bb, bc = band_rms_windows(xb), band_rms_windows(xc)
                ci = geom.CI[f"{jn}{side}"]
                yt = np.where(tr["nv"][ks, ci] >= 2, tr["Xd"][ks, ci, 1].astype(float), np.nan) - hS
                bt = band_rms_windows(yt)
                mb, mc, mt = np.nanmax(bb) if np.isfinite(bb).any() else 0.0, np.nanmax(bc) if np.isfinite(bc).any() else 0.0, \
                    (np.nanmax(bt) if np.isfinite(bt).any() else 0.0)
                # the triangulated joint's own reversals (lightly smoothed; only where it covers 80 % of the window): a candidate that follows
                # a real up-and-down the cameras show is not a new swing
                rt = 0
                okt = np.isfinite(yt[sel])
                if okt.mean() >= 0.8:
                    from scipy.ndimage import gaussian_filter1d
                    idx = np.flatnonzero(okt)
                    yi = np.interp(np.arange(okt.size), idx, yt[sel][idx])
                    rt = swings_within(gaussian_filter1d(yi, 1.5), 0.03 - m_, 30)
                ok = (rc <= max(rb, rt)) and (mc <= max(mb, mt) + cfg.gate_g5_band_m)
                out[f"{role}_{jn}{side}"] = {"reversals": [rb, rc, rt], "band_max_cm": [_r(mb * 100), _r(mc * 100), _r(mt * 100)], "ok": bool(ok)}
                out["ok"] &= bool(ok)
            # swivel acceleration
            pb, wb = geom.swivel(Jb[:, a["sh"]], Jb[:, a["el"]], Jb[:, a["wr"]])
            pc, wc = geom.swivel(Jc[:, a["sh"]], Jc[:, a["el"]], Jc[:, a["wr"]])
            vb = geom.swivel_valid(Jb[:, a["sh"]], Jb[:, a["wr"]])
            vc = geom.swivel_valid(Jc[:, a["sh"]], Jc[:, a["wr"]])

            def acc(phi, w, v):
                u = np.degrees(np.unwrap(np.radians(phi)))
                d2 = np.abs(np.diff(u, 2)) * hz * hz
                ok3 = (w[1:-1] > 0.1) & v[2:] & v[1:-1] & v[:-2]
                return d2[ok3]
            ab, ac = acc(pb, wb, vb), acc(pc, wc, vc)
            qb, qc = nanpct(ab, 99), nanpct(ac, 99)
            ok = (not np.isfinite(qc)) or (not np.isfinite(qb) and qc <= cfg.gate_g5_swivel_floor * 10) or \
                qc <= ratio * qb + cfg.gate_g5_swivel_floor
            out[f"{role}_swivel{side}"] = {"p99_deg_s2": [_r(qb, 0), _r(qc, 0)], "ok": bool(ok)}
            out["ok"] &= bool(ok)
        # jerk of the joints the window moves (positions below the moved rotations)
        idx = np.flatnonzero(aff[:22])
        if len(idx):
            jb = np.linalg.norm(np.diff(Jb[:, idx], 3, axis=0), axis=-1) * hz ** 3
            jc = np.linalg.norm(np.diff(Jc[:, idx], 3, axis=0), axis=-1) * hz ** 3
            qb, qc = nanpct(jb, 99), nanpct(jc, 99)
            ok = qc <= ratio * qb + cfg.gate_g5_jerk_floor
            out[f"{role}_jerk"] = {"p99_m_s3": [_r(qb, 1), _r(qc, 1)], "ok": bool(ok)}
            out["ok"] &= bool(ok)
    return out


def limit_masks(ctx: GateCtx, fit: dict, ks: np.ndarray, role: str) -> dict:
    """(n,) bool per limit: wrist (zouk_contacts limit_excess > 0, either hand), elbow (flexion > 150, hyperextension > 10, off-hinge > 35),
    the DESIGN 3.4 twist gates, chain wraps"""
    from .. import zouk_contacts as zc
    from .decide import CHAINS, TWIST_AXES, TWIST_GROUP, twists
    P = bd.pose55(fit, ks)
    out = {}
    for s, x in (("left", "L"), ("right", "R")):
        ex = zc.limit_excess(zc.wrist_angles(P, s))
        out[f"wrist_{x}"] = np.any(np.stack([v > 0 for v in ex.values()]), 0)
        ea = zc.elbow_angles(P, s)
        L = zc.ELBOW_LIMITS
        out[f"elbow_{x}"] = (ea["flexion"] > L["flexion"]) | (ea["flexion"] < -L["hyperextension"]) | (np.abs(ea["off_hinge"]) > L["off_hinge"])
    sub = {"global_orient": P[:, 0], "body_pose": P[:, 1:22].reshape(len(ks), 63)}
    tw = twists(sub, ctx.rest[role])
    g = ctx.cfg.twist_gate
    for j, v in tw.items():
        grp = TWIST_GROUP[j]
        out[f"twist_{j}"] = np.abs(v) > (g["spine"] if grp == "spine" else g[grp])
    out["spine_total"] = np.abs(tw[3] + tw[6] + tw[9]) > g["spine_total"]
    out["neck_head"] = np.abs(tw[12] + tw[15]) > g["neck_head"]
    Rl = geom.local_rotations(sub["global_orient"], sub["body_pose"])
    rj = ctx.rest[role]
    for ch, js in CHAINS.items():
        axes = []
        for j in js:
            c = TWIST_AXES[j]
            a = rj[c] - rj[j] if c != j else rj[15] - rj[12]
            axes.append(a / max(np.linalg.norm(a), 1e-9))
        ct = geom.chain_twist(Rl[:, list(js)], np.asarray(axes))
        out[f"wrap_{ch}"] = np.abs(ct["wrap"]) >= ctx.cfg.wrap_deg
    return out


def g6(ctx: GateCtx, win: dict, fits: dict, ks: np.ndarray, moved: dict) -> dict:
    out = {"ok": True}
    for role in moved:
        mb = limit_masks(ctx, fits["base"][role], ks, role)
        mc = limit_masks(ctx, fits["cand"][role], ks, role)
        bad = {}
        for k in mc:
            new = int((mc[k] & ~mb[k]).sum())
            if k.startswith("wrap_"):
                new = int(mc[k].sum())
            if new:
                bad[k] = new
        out[role] = {"new_frames": bad, "ok": not bad}
        out["ok"] &= not bad
    return out


def g7(ctx: GateCtx, win: dict, fits: dict, ks: np.ndarray, moved: dict) -> dict:
    """no new speed incident against the input (jump = added speed > 0.55 m/s; fast = faster than max(input p99, input) + 0.25 m/s)
    and no whip (zouk_contacts.whip_frames_added: base = the reference, the window's own start = the base)"""
    from .. import zouk_contacts as zc
    ref = ctx.speed_ref()
    hz = ctx.hz
    G = zc.GATE
    out = {"ok": True}
    spin = {r for r, d in win["dancers"].items() if d.get("kind") == "spin" or "spin" in d.get("reasons", [])}
    for role in moved:
        Ji = np.asarray(ctx.input[role]["joints"], float)[ks]
        Jb = np.asarray(fits["base"][role]["joints"], float)[ks]
        Jc = np.asarray(fits["cand"][role]["joints"], float)[ks]
        ti, tb, tc = track_points(Ji), track_points(Jb), track_points(Jc)
        bad = {}
        for nm in ti:
            sp = lambda X: np.linalg.norm(np.diff(X, axis=0), axis=1) * hz
            v_in, v0, v1 = sp(ti[nm]), sp(tb[nm]), sp(tc[nm])
            a0, a1 = sp(tb[nm] - ti[nm]), sp(tc[nm] - ti[nm])
            a_start = sp(tc[nm] - tb[nm])
            p99 = ref[f"{role}_{nm}"]
            jump_c = (a1 > G["add_mps"]) & (a_start > G["add_mps"])
            fast_c = v1 > np.maximum(p99, v_in) + G["fast_margin_mps"]
            fast_b = v0 > np.maximum(p99, v_in) + G["fast_margin_mps"]
            whip = zc.whip_frames_added(a0, a1, a_start, v_in, v0, v1, p99)
            if role in spin:
                # a spin correction turns the whole body: its own motion against the input is the correction (the frames of fast motion
                # move with it), so only a peak beyond anything the input / the base / the take shows here counts
                peak = max(p99, float(v_in.max()) if len(v_in) else 0.0, float(v0.max()) if len(v0) else 0.0) + G["fast_margin_mps"]
                n = int((v1 > peak).sum())
            else:
                n = int((jump_c | (fast_c & ~fast_b) | (whip & ~fast_b)).sum())
            if n:
                bad[nm] = {"frames": n, "max_added_mps": _r(a_start.max()), "max_speed_mps": _r(v1.max())}
        out[role] = {"incidents": bad, "ok": not bad}
        out["ok"] &= not bad
    return out


def g8(ctx: GateCtx, win: dict, fits: dict, ks: np.ndarray, spin_roles: list) -> dict:
    """spin windows: the candidate's hip-line heading against the camera yaw track psi*"""
    from .splice import face_contradictions
    from .prepare import azimuths, face_labels, head_heading
    cfg = ctx.cfg
    out = {"ok": True}
    win_s = int(round(1.0 * ctx.hz))
    for role in spin_roles:
        ps = np.asarray(ctx.yaw[f"{role}__psi_star"], float)[ks]
        conf = np.asarray(ctx.yaw[f"{role}__conf"], float)[ks]
        row = {}
        for nm in ("base", "cand"):
            f = fits[nm][role]
            J = np.asarray(f["joints"], float)[ks]
            h = geom.unwrap_deg(geom.line_heading(J[:, 1], J[:, 2]))
            e = np.abs(geom.wrap_deg(h - ps))
            # reversals against psi*: the winding difference over 1 s
            du = (h - ps)
            rev = np.abs(du[win_s:] - du[:-win_s]).max() if len(du) > win_s else 0.0
            lab = face_labels(ctx.views, ctx.cams, role, ks)
            az = azimuths(ctx.obs, ctx.cams, J, ks)
            hh = head_heading(f, ks)
            fc = face_contradictions(lab, az, hh)
            nl = (lab != 0).sum(1)
            contra = int(((fc >= 2) & (fc >= 0.5 * np.maximum(1, nl))).sum())
            seg = geom.wrap_deg(geom.line_heading(J[:, 16], J[:, 17]) - geom.line_heading(J[:, 1], J[:, 2]))
            fl = np.stack([geom.wrap_deg(geom.heading((J[:, f_] - J[:, a_]) * np.array([1, 0, 1.0]), min_h=0.02)
                                         - geom.line_heading(J[:, 1], J[:, 2])) for a_, f_ in ((7, 10), (8, 11))], 1)
            wind = (np.abs(seg) > cfg.gate_g8_torso_deg) | (np.nan_to_num(np.abs(fl), nan=0.0) > cfg.gate_g8_foot_deg).any(1)
            row[nm] = {"p90": nanpct(e[conf > 0.0], 90), "rev": float(rev), "contra": contra, "wind": wind}
        b, c = row["base"], row["cand"]
        new_wind = int((c["wind"] & ~b["wind"]).sum())
        ok = (c["p90"] <= cfg.gate_g8_p90_deg or c["p90"] <= b["p90"] + 1.0) and (c["rev"] <= cfg.gate_g8_rev_deg or c["rev"] <= b["rev"]) \
            and c["contra"] <= b["contra"] and new_wind == 0
        target = c["p90"] <= cfg.gate_g8_p90_deg and c["rev"] <= cfg.gate_g8_rev_deg and c["contra"] == 0
        out[role] = {"yaw_p90_deg": [_r(b["p90"], 1), _r(c["p90"], 1)], "reversal_deg": [_r(b["rev"], 1), _r(c["rev"], 1)],
                     "face_contra_frames": [b["contra"], c["contra"]], "new_winding_frames": new_wind, "ok": bool(ok), "target": bool(target)}
        out["ok"] &= bool(ok)
        out["target"] = bool(out.get("target", True) and target)
    return out


INSIDE_PARTS = ("torso", "shoulder_L", "shoulder_R")
ALLOW_PARTS = ("upper_arm_L", "upper_arm_R")


def inside_series(ctx: GateCtx, fits_r: dict, ks: np.ndarray, exact: dict | None = None) -> np.ndarray:
    """(n,) depth (m) of his right hand / forearm inside her: the exact hand depth in her torso / shoulders (upper arms less the
    allowance), and the capsule depth of his right forearm in her torso slabs"""
    from .. import zouk_contacts as zc
    from .decide import inside_depth
    if exact is None:
        exact = zc.exact_of({r: fits_r[r] for r in ROLES}, ks, ctx.cfg.gate_exact_device, 48)
    parts = list(zc.EXACT_PARTS)
    hp = np.asarray(exact["hand_pen_part"], float)
    d1 = hp[:, [parts.index(p) for p in INSIDE_PARTS]].max(1)
    d2 = hp[:, [parts.index(p) for p in ALLOW_PARTS]].max(1) - ctx.cfg.gate_g9_allow_m
    JL, JF = (np.asarray(fits_r[r]["joints"], float)[ks] for r in ROLES)
    RgL = geom.global_rotations(np.asarray(fits_r["lead"]["global_orient"], float)[ks], np.asarray(fits_r["lead"]["body_pose"], float)[ks])
    RgF = geom.global_rotations(np.asarray(fits_r["follow"]["global_orient"], float)[ks], np.asarray(fits_r["follow"]["body_pose"], float)[ks])
    d3 = inside_depth(ctx.caps["lead"], ctx.caps["follow"], posed_capsules(ctx.caps["lead"], JL, RgL), posed_capsules(ctx.caps["follow"], JF, RgF))
    return np.maximum.reduce([d1, np.maximum(d2, 0), d3])


def g9(ctx: GateCtx, win: dict, fits: dict, ks: np.ndarray, exact: dict | None = None) -> dict:
    cfg = ctx.cfg
    d = {nm: inside_series(ctx, fits[nm], ks, (exact or {}).get(nm)) for nm in ("base", "cand")}
    n = len(ks)
    b3, c3 = int((d["base"] > cfg.gate_g9_m).sum()), int((d["cand"] > cfg.gate_g9_m).sum())
    b5, c5 = int((d["base"] > cfg.gate_g9_hard_m).sum()), int((d["cand"] > cfg.gate_g9_hard_m).sum())
    ok = (c3 <= b3) and (c5 == 0 or c5 <= b5)
    target = (c3 <= cfg.gate_g9_share * n) and c5 == 0
    return {"frames_gt3": [b3, c3], "frames_gt5": [b5, c5], "max_cm": [_r(d["base"].max() * 100, 1), _r(d["cand"].max() * 100, 1)],
            "p99_cm": [_r(np.percentile(d["base"], 99) * 100, 1), _r(np.percentile(d["cand"], 99) * 100, 1)], "n": n, "ok": bool(ok),
            "target": bool(target)}


def trust_hits(ctx: GateCtx, fits: dict, ks: np.ndarray, win: dict) -> dict:
    """share of frames where a moved joint's change against the base reaches 0.98 of its trust radius (or the root its limit)"""
    cfg = ctx.cfg
    hit = np.zeros(len(ks), bool)
    worst = {}
    for role, d in win["dancers"].items():
        Pb = bd.pose55(fits["base"][role], ks)
        Pc = bd.pose55(fits["cand"][role], ks)
        Rb, Rc = bd.rotmat(Pb), bd.rotmat(Pc)
        for j_s, r in d["trust_deg"].items():
            j = int(j_s)
            dR = Rc[:, j] @ np.swapaxes(Rb[:, j], 1, 2)
            ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))
            hit |= ang >= cfg.gate_trust_frac * r
            worst[f"{role}_{j}"] = _r(ang.max(), 1)
        moved_all = set(int(j) for j in d["trust_deg"])
        other = [j for j in range(55) if j not in moved_all]
        if len(other):
            dd = np.abs(Pc[:, other] - Pb[:, other]).max()
            if dd > 1e-9:
                hit |= True
                worst[f"{role}_unmoved_changed"] = _r(float(dd), 6)
        dt = np.linalg.norm(np.asarray(fits["cand"][role]["transl"], float)[ks] - np.asarray(fits["base"][role]["transl"], float)[ks], axis=1)
        if d.get("root_m", 0.0) > 0:
            hit |= dt >= cfg.gate_trust_frac * d["root_m"]
        elif dt.max() > 1e-9:
            hit |= dt > 1e-9
        worst[f"{role}_root_cm"] = _r(dt.max() * 100, 2)
    share = float(hit.mean()) if len(hit) else 0.0
    return {"share": round(share, 3), "worst_deg": worst, "ok": share <= cfg.gate_trust_share}


# ------------------------------------------------------------------------------------------------ the window gate

def window_frames(win: dict, T: int, margin: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """(window + ramps, the same + a margin for derivatives / GLI neighbours)"""
    kw = np.arange(win["r0"], win["r1"] + 1)
    km = np.arange(max(0, win["r0"] - margin), min(T, win["r1"] + margin + 1))
    return kw, km


def window_gates(ctx: GateCtx, cand: dict, win: dict, base: dict | None = None, exact: dict | None = None,
                 trust: dict | None = None, skip: tuple = ()) -> dict:
    """all gates of one window. cand {role: fit} = the base with the window's frames solved (full arrays); base {role: fit} (default
    ctx.base); exact {"base": {...}, "cand": {...}} the exact-mesh arrays on the window + ramps frames (computed here when absent);
    trust: the solver's own trust report ({"share"}), else measured from the poses"""
    t0 = time.time()
    base = base or ctx.base
    fits = {"base": base, "cand": cand}
    kw, km = window_frames(win, ctx.T)
    moved = {r: d["joints"] for r, d in win["dancers"].items()}
    spin_roles = [r for r, d in win["dancers"].items() if d.get("kind") == "spin" or "spin" in d.get("reasons", [])]
    g = {}
    # a decided R4 class (his right hand): the views the candidate hides the hand from carry no hand evidence (r4_visibility)
    vr = r4_visibility(ctx, fits, kw) if ("lead" in moved and not {"G1", "G2"} <= set(skip)) else None
    g.update(g1_g10(ctx, win, fits, kw, moved, vr) if "G1" not in skip else {})
    if "G2" not in skip:
        g["G2"] = g2(ctx, win, fits, kw, moved, vr)
    if "G3" not in skip:
        g["G3"] = g3(ctx, win, fits, kw)
    if "G4" not in skip:
        g["G4"] = g4(ctx, win, fits, km)
    if "G5" not in skip:
        g["G5"] = g5(ctx, win, fits, km, moved, kw)
    if "G6" not in skip:
        g["G6"] = g6(ctx, win, fits, kw, moved)
    if "G7" not in skip:
        g["G7"] = g7(ctx, win, fits, km, moved)
    if spin_roles and "G8" not in skip:
        g["G8"] = g8(ctx, win, fits, kw, spin_roles)
    if "G9" not in skip:
        g["G9"] = g9(ctx, win, fits, kw, exact)
    if "trust" not in skip:
        g["trust"] = trust if trust is not None else trust_hits(ctx, fits, kw, win)
        g["trust"]["ok"] = bool(g["trust"].get("share", 0.0) <= ctx.cfg.gate_trust_share)
    failed = [k for k in GATES if k in g and not g[k]["ok"]]
    missed = [k for k in GATES if k in g and g[k].get("target") is False]
    return {"id": win.get("id"), "pass": not failed, "failed": failed, "targets_missed": missed, "gates": g,
            "seconds": round(time.time() - t0, 2)}


# ------------------------------------------------------------------------------------------------ calibration (DESIGN 7.2)

def _runs(m: np.ndarray, merge: int = 15, min_len: int = 6) -> list:
    from . import hmm
    return hmm.runs(m, merge=merge, min_len=min_len)


def calib_window(ctx: GateCtx, k0: int, k1: int, dancers: dict, pad: int = 15) -> dict:
    """a window record for a calibration candidate: core = the changed run padded, ramps 0 (the candidate carries its own blends)"""
    T = ctx.T
    a, b = max(0, k0 - pad), min(T - 1, k1 + pad)
    from .plan import trust_of
    return {"id": f"calib_{k0}", "k0": a, "k1": b, "r0": a, "r1": b, "ramp_in": 0, "ramp_out": 0, "t0": round(a / ctx.hz, 3),
            "t1": round(b / ctx.hz, 3), "reasons": sorted({x for d in dancers.values() for x in d["reasons"]}),
            "dancers": {r: {"joints": list(d["joints"]), "reasons": d["reasons"], "kind": d.get("kind", "arm"),
                            "trust_deg": {str(j): trust_of(j, "spin" if d.get("kind") == "spin" else "arm", ctx.cfg) for j in d["joints"]},
                            "root_m": d.get("root_m", 0.0), "real_items": []} for r, d in dancers.items()}}


def synthetic_swing(ctx: GateCtx, k0: int, n: int, amp_m: float = 0.04, hz_sw: float = 2.0) -> dict:
    """the base with his right elbow swung up and down by +-amp_m at hz_sw (the shoulder's rotation about the horizontal axis normal to
    the upper arm, cosine-ramped over 0.5 s at each end)"""
    lead = ctx.base["lead"]
    ks = np.arange(k0, k0 + n)
    P = bd.pose55(lead, ks)
    J = np.asarray(lead["joints"], float)[ks]
    ua = J[:, 19] - J[:, 17]
    L = np.linalg.norm(ua, axis=1)
    ax = np.cross(ua, np.array([0, 1.0, 0]))
    ax /= np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-9)
    tt = np.arange(n) / ctx.hz
    ramp = np.clip(np.minimum(tt, tt[::-1]) / 0.5, 0, 1)
    ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)
    ang = (amp_m / L) * np.sin(2 * np.pi * hz_sw * tt) * ramp            # small-angle: elbow height change ~ amp
    o = bd.forward(lead, ks, points=False)
    Rg = o["R"]
    # world rotation about ax applied to the shoulder: local' = R_parent^T Rw R_parent R_local
    Rw = bd.rotmat(ax * ang[:, None])
    Rp = Rg[:, 14]
    Rl = bd.rotmat(P[:, 17])
    P[:, 17] = bd.rotvec(np.swapaxes(Rp, 1, 2) @ Rw @ Rp @ Rl)
    return {"lead": fit_window(lead, P, None, ks), "follow": ctx.base["follow"]}


def _closest_pair(JL: np.ndarray, JF: np.ndarray) -> tuple:
    """per frame: the interior bone pair (his right arm x her left arm) with the smallest distance -> (d_s, dist, cA, n, his bone i)"""
    A, B = JL[:, [17, 19, 21]], JF[:, [16, 18, 20]]
    best = None
    for i in range(2):
        for j in range(2):
            ds, s_, t_, dist, inter = geom.signed_sep(A[:, i], A[:, i + 1], B[:, j], B[:, j + 1])
            u, v = A[:, i + 1] - A[:, i], B[:, j + 1] - B[:, j]
            n = geom.unit(np.cross(u, v))
            cA = A[:, i] + u * s_[:, None]
            dd = np.where(inter, dist, np.inf)
            if best is None:
                best = [ds, dd, cA, n, np.full(len(ds), i)]
            else:
                m = dd < best[1]
                for q, val in enumerate((ds, dd, cA, n, np.full(len(ds), i))):
                    best[q] = np.where(m[(...,) + (None,) * (np.ndim(val) - 1)], val, best[q])
    return tuple(best)


def synthetic_passthrough(ctx: GateCtx, n: int = 24, ramp: int = 7) -> tuple | None:
    """a broken base for the calibration: his right arm pushed THROUGH her left arm (a GLI pass-through event) inside a held stretch
    (the arms entangled, closest bones < 6 cm apart) where the base has none and his forearm is not > 3 cm inside her: his shoulder turned so the closest point of his bone moves across hers
    (along -sign(d_s) n by |d_s| + 3 cm), cosine-ramped in and out over `ramp` frames, in the first such stretch whose cameras support the
    true base over the broken one -> (broken {role: fit}, k0, n) or None"""
    from . import hmm
    T = ctx.T
    G = gli_series(ctx.base, np.arange(T))["R9_RL"]
    s = ctx.rel["R9_RL__s"]
    ev = geom.passthrough_events(G, ctx.cfg.thru_jump)
    inside = np.nan_to_num(ctx.rel["R4__inside_base"].astype(float))
    JL, JF = np.asarray(ctx.base["lead"]["joints"], float), np.asarray(ctx.base["follow"]["joints"], float)
    ds, dist, cA, nn, bone = _closest_pair(JL, JF)
    band = ctx.rel["R9_RL__band"].astype(bool)
    good = band & np.isfinite(dist) & (dist < 0.06) & (inside < ctx.cfg.gate_g9_m)
    lead = ctx.base["lead"]
    for a, b in hmm.runs(good, min_len=n):
        if any(a - 15 <= e <= b + 15 for e in ev):
            continue
        k0 = a + max(0, (b - a + 1 - n) // 2)
        ks = np.arange(k0, k0 + n)
        tt = np.arange(n)
        w = 0.5 - 0.5 * np.cos(np.pi * np.clip(np.minimum(tt, tt[::-1]) / ramp, 0, 1))
        o = bd.forward(lead, ks, points=False)
        S = o["J"][:, 17]
        k_mid = n // 2
        disp = -(ds[ks] + np.sign(ds[ks]) * 0.03)[:, None] * nn[ks]            # move his closest point across hers
        r = cA[ks] - S
        ax = np.cross(r, disp)
        ax = ax / np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-9)
        ang = np.linalg.norm(disp, axis=1) / np.maximum(np.linalg.norm(r, axis=1), 0.05)
        ang = np.full(n, ang[k_mid]) * w
        Rw = bd.rotmat(np.tile(ax[k_mid], (n, 1)) * ang[:, None])
        P = bd.pose55(lead, ks)
        Rp = o["R"][:, 14]
        P[:, 17] = bd.rotvec(np.swapaxes(Rp, 1, 2) @ Rw @ Rp @ bd.rotmat(P[:, 17]))
        f = fit_window(lead, P, None, ks)
        broken = {"lead": f, "follow": ctx.base["follow"]}
        km = np.arange(max(0, k0 - 15), min(T, k0 + n + 15))
        if not len(geom.passthrough_events(gli_series(broken, km)["R9_RL"], ctx.cfg.thru_jump)):
            continue
        # a valid plant is one the cameras contradict: on the moved points every camera that sees them fits the true base at least as
        # well as the broken one (in the closed hold the COCO arms fit the far side better - that stretch is no "correct base" to break)
        win = calib_window(ctx, k0, k0 + n - 1, {"lead": {"joints": [9, 14, 17, 19, 21], "reasons": ["thru"], "kind": "arm"}}, pad=0)
        g = g1_g10(ctx, win, {"base": broken, "cand": ctx.base}, np.arange(win["r0"], win["r1"] + 1), {"lead": [9, 14, 17, 19, 21]})["G1"]
        supports = all(v[kind]["cand"][0] <= v[kind]["base"][0] for v in g["cams"].values() for kind in ("mamma", "coco") if kind in v)
        if supports:
            return broken, int(k0), int(n)
    return None


def calibrate_gate(take: str, cfg: SidesCfg | None = None, write: bool = True, v43_dir: Path | None = None) -> dict:
    """DESIGN 7.2: every v4.3 changed run of his right arm (>= 15 of 16 rejected), the 199-s splice (accepted), a no-op (accepted), a
    synthetic 4-cm 2-Hz elbow swing (rejected), a synthetic pass-through removal (accepted) -> work/<take>/sides/gate_calibration.json"""
    t0 = time.time()
    from .prepare import load_fit
    ctx = GateCtx(take, cfg)
    cfg = ctx.cfg
    T = ctx.T
    hz = ctx.hz
    rep = {"schema": "dancecap.sides.gate_calibration/1", "take": take, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": {}}
    arm = {"joints": [9, 14, 17, 19, 21], "reasons": ["inside"], "kind": "arm"}
    # ---- v4.3 changed runs (vs the input): candidate = the input with v4.3's lead poses in the run
    v43 = Path(v43_dir) if v43_dir else ctx.work / "v4" / "smplx_contact_v43"
    rows = []
    if (v43 / "lead.npz").exists():
        f43 = load_fit(v43 / "lead.npz")
        J43, Ji = np.asarray(f43["joints"], float), np.asarray(ctx.input["lead"]["joints"], float)
        d = np.abs(J43[:, [17, 19, 21]] - Ji[:, [17, 19, 21]]).max((1, 2)) > 0.005
        runs = _runs(d)
        ex_in = dict(np.load(ctx.work / "v4" / "contacts" / "exact_input.npz")) if (ctx.work / "v4" / "contacts" / "exact_input.npz").exists() else None
        ex_43 = dict(np.load(ctx.work / "v4" / "contacts" / "exact_v43.npz")) if (ctx.work / "v4" / "contacts" / "exact_v43.npz").exists() else None
        P43 = bd.pose55(f43)
        base_in = {r: ctx.input[r] for r in ROLES}
        for a, b in runs:
            win = calib_window(ctx, a, b, {"lead": arm})
            kw = np.arange(win["r0"], win["r1"] + 1)
            cand_lead = fit_window(ctx.input["lead"], P43[kw], np.asarray(f43["transl"], float)[kw], kw)
            cand = {"lead": cand_lead, "follow": ctx.input["follow"]}
            ex = None
            if ex_in is not None and ex_43 is not None:
                ex = {"base": {k: ex_in[k][kw] for k in ("hand_pen", "arm_pen", "hand_gap", "hand_pen_part")},
                      "cand": {k: ex_43[k][kw] for k in ("hand_pen", "arm_pen", "hand_gap", "hand_pen_part")}}
            r = window_gates(ctx, cand, win, base=base_in, exact=ex)
            rows.append({"t": [round(a / hz, 2), round(b / hz, 2)], "pass": r["pass"], "failed": r["failed"], "gates": r["gates"]})
            log(f"v4.3 run {a / hz:.2f}-{b / hz:.2f}: {'ACCEPT' if r['pass'] else 'reject ' + ','.join(r['failed'])}")
    rej = sum(not x["pass"] for x in rows)
    rep["cases"]["v43_runs"] = {"n": len(rows), "rejected": rej, "need": "rejected >= n - 1", "ok": bool(len(rows) and rej >= len(rows) - 1),
                                "rows": rows}
    # ---- the splice (prepare): candidate = the cpu fit, base = the input, over each splice window
    prep = json.loads((ctx.sd / "cpu" / "prepare.json").read_text(encoding="utf-8")) if (ctx.sd / "cpu" / "prepare.json").exists() else {}
    srows = []
    for s in prep.get("splices", []):
        spin = {"joints": list(range(55)), "reasons": ["spin"], "kind": "spin", "root_m": 1.0}
        dz = {s["defect_role"]: spin, OTHER[s["defect_role"]]: dict(spin, reasons=["spin"])}
        win = calib_window(ctx, s["k0"], s["k1"] - 1, dz, pad=0)
        # the splice moves every body joint: its trust is the splice itself (no solver), so the trust gate measures nothing here
        for r in win["dancers"]:
            win["dancers"][r]["trust_deg"] = {str(j): 360.0 for j in range(55)}
            win["dancers"][r]["root_m"] = 10.0
        r = window_gates(ctx, ctx.base, win, base={k: ctx.input[k] for k in ROLES})
        srows.append({"t": [s["t0"], s["t1"]], "pass": r["pass"], "failed": r["failed"], "gates": r["gates"]})
        log(f"splice {s['t0']}-{s['t1']}: {'ACCEPT' if r['pass'] else 'reject ' + ','.join(r['failed'])}")
    rep["cases"]["splice"] = {"rows": srows, "ok": bool(srows) and all(x["pass"] for x in srows)}
    # ---- no-op
    k0 = int(round(57.0 * hz)) if T > int(round(60 * hz)) else T // 3
    win = calib_window(ctx, k0, k0 + 60, {"lead": arm})
    r = window_gates(ctx, ctx.base, win)
    rep["cases"]["noop"] = {"t": [win["t0"], win["t1"]], "pass": r["pass"], "failed": r["failed"], "ok": r["pass"], "gates": r["gates"]}
    log(f"no-op: {'ACCEPT' if r['pass'] else 'reject ' + ','.join(r['failed'])}")
    # ---- synthetic swing: 4 cm 2 Hz on his right elbow, over 3 s of the counterbalance
    ks0 = int(round(cfg.gate_calib_swing_t * hz))
    n = int(round(3.0 * hz))
    cand = synthetic_swing(ctx, ks0, n)
    win = calib_window(ctx, ks0, ks0 + n - 1, {"lead": dict(arm, reasons=["swing"])}, pad=0)
    r = window_gates(ctx, cand, win)
    rep["cases"]["synthetic_swing"] = {"t": [win["t0"], win["t1"]], "pass": r["pass"], "failed": r["failed"], "ok": not r["pass"],
                                       "gates": r["gates"]}
    log(f"synthetic swing: {'ACCEPT' if r['pass'] else 'reject ' + ','.join(r['failed'])}")
    # ---- synthetic pass-through removal: base = the base with a planted pass-through, candidate = the true base
    sp = synthetic_passthrough(ctx)
    if sp is not None:
        broken, k0p, n_p = sp
        win = calib_window(ctx, k0p, k0p + n_p - 1, {"lead": dict(arm, reasons=["thru"])}, pad=0)
        r = window_gates(ctx, ctx.base, win, base=broken)
        rep["cases"]["synthetic_passthrough_removal"] = {"t": [win["t0"], win["t1"]], "pass": r["pass"], "failed": r["failed"],
                                                         "ok": r["pass"], "gates": r["gates"]}
        log(f"synthetic pass-through removal {win['t0']}-{win['t1']}: {'ACCEPT' if r['pass'] else 'reject ' + ','.join(r['failed'])}")
    if "synthetic_passthrough_removal" not in rep["cases"]:
        rep["cases"]["synthetic_passthrough_removal"] = {"ok": False, "note": "no held stretch found to plant one"}
    rep["ok"] = all(c.get("ok") for c in rep["cases"].values())
    rep["cfg"] = {k: v for k, v in cfg.to_json().items() if k.startswith("gate_")}
    rep["seconds"] = round(time.time() - t0, 1)
    if write:
        write_json(ctx.sd / "gate_calibration.json", rep)
    log("calibration", "OK" if rep["ok"] else "FAILED", rep["seconds"], "s")
    return rep


# ------------------------------------------------------------------------------------------------ the whole take (DESIGN 7.1, last paragraph)

def mamma_frame_medians(ctx: GateCtx, fit: dict, role: str, ks: np.ndarray) -> np.ndarray:
    """(n, C) per-frame median px error of the dancer's 512 MAMMA points in each camera (NaN: fewer than 20 visible points)"""
    p = PIDX[role]
    out = np.full((len(ks), len(ctx.cams)), np.nan)
    for a in range(0, len(ks), 1500):
        kk = ks[a:a + 1500]
        pts = bd.forward(fit, kk)["pts"]
        for ci, c in enumerate(ctx.cams):
            o = ctx.obs[c]
            lm = ctx.lm(c)[kk, p].astype(float)
            vis = (o["vis"][kk, p].astype(float) > 0.5) & (np.abs(lm).sum(-1) > 0) & o["ok"][kk][:, None]
            uv, dep = project(o["K"][kk], o["R"][kk], o["t"][kk], pts)
            e = np.where(vis & (dep > 0.1), np.linalg.norm(uv - lm, axis=-1), np.nan)
            n = np.isfinite(e).sum(1)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[a:a + len(kk), ci] = np.where(n >= 20, np.nanmedian(e, 1), np.nan)
    return out


def sole_heights(fit: dict, ks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2) lowest sole point height and (n, 2, 3) sole centre of each foot (left, right)"""
    from .solve import sole_index
    si = sole_index(fit)
    o = bd.forward(fit, ks, points=False)
    R = o["R"]
    p = o["J"] - np.einsum("tjab,jb->tja", R, bd.model(fit)["j_rest"])
    low, cen = [], []
    for f in ("left", "right"):
        Wv, xv = si[f]
        Rb = np.einsum("vj,tjab->tvab", Wv, R)
        X = np.einsum("tvab,vb->tva", Rb, xv) + np.einsum("vj,tjd->tvd", Wv, p)
        low.append(X[..., 1].min(1))
        cen.append(X.mean(1))
    return np.stack(low, 1), np.stack(cen, 1)


def take_gate(ctx: GateCtx, merged: dict, adopted: list) -> dict:
    """no adopted window fails (checked by the merge); whole-take per-camera 2D median <= base + 0.2 px; stance slip <= base + 2 mm;
    no new floor penetration > 1 cm; every frame outside the adopted windows + ramps byte-identical to the base"""
    cfg = ctx.cfg
    T = ctx.T
    cover = np.zeros(T, bool)
    for w in adopted:
        cover[w["r0"]:w["r1"] + 1] = True
    out = {"ok": True}
    ident = {}
    for r in ROLES:
        keep = ~cover
        same = all(np.array_equal(np.asarray(merged[r][k])[keep], np.asarray(ctx.base[r][k])[keep])
                   for k in (*bd.POSE_KEYS, "transl", "joints") if k in ctx.base[r])
        ident[r] = bool(same)
    out["outside_identical"] = ident
    out["ok"] &= all(ident.values())
    two = {}
    for r in ROLES:
        ch = np.flatnonzero((np.abs(bd.pose55(merged[r]) - bd.pose55(ctx.base[r])).max((1, 2)) > 0)
                            | (np.abs(np.asarray(merged[r]["transl"]) - np.asarray(ctx.base[r]["transl"])).max(1) > 0))
        if not len(ch):
            two[r] = {"changed_frames": 0, "ok": True}
            continue
        eb = mamma_frame_medians(ctx, ctx.base[r], r, np.arange(T))
        em = eb.copy()
        em[ch] = mamma_frame_medians(ctx, merged[r], r, ch)
        rows = {}
        ok = True
        for ci, c in enumerate(ctx.cams):
            b, m = nanpct(eb[:, ci], 50), nanpct(em[:, ci], 50)
            okc = (not np.isfinite(b)) or m <= b + cfg.gate_take_2d_px
            rows[c] = [_r(b, 3), _r(m, 3)]
            ok &= okc
        # stance slip (the base's stance flags): p95 horizontal step of the sole centre on stance frames, mm
        st = np.asarray(ctx.base[r].get("stance", np.zeros((T, 2), bool)), bool)
        kk = np.unique(np.clip(np.concatenate([ch - 1, ch, ch + 1]), 0, T - 1))
        lb, cb = sole_heights(ctx.base[r], kk)
        lm_, cm_ = sole_heights(merged[r], kk)
        sl = {}
        for i, f in enumerate(("left", "right")):
            both = st[kk[1:], i] & st[kk[:-1], i] & (np.diff(kk) == 1)
            db = np.linalg.norm(np.diff(cb[:, i, [0, 2]], axis=0), axis=1)[both] * 1000
            dm = np.linalg.norm(np.diff(cm_[:, i, [0, 2]], axis=0), axis=1)[both] * 1000
            sl[f] = [_r(nanpct(db, 95), 2), _r(nanpct(dm, 95), 2)]
            if len(db) and nanpct(dm, 95) > nanpct(db, 95) + cfg.gate_take_slip_mm:
                ok = False
        pen_new = int(((lm_ < -cfg.gate_take_floor_m) & ~(lb < -cfg.gate_take_floor_m)).sum())
        ok &= pen_new == 0
        two[r] = {"changed_frames": int(len(ch)), "mamma_median_px": rows, "stance_slip_p95_mm": sl, "floor_pen_new_frames": pen_new, "ok": bool(ok)}
        out["ok"] &= bool(ok)
    out["roles"] = two
    return out
