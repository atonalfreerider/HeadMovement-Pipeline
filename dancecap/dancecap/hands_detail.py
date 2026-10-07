"""Hand detail for the avatars: contact-consistent fingers, wrists and arms, merged with the face into the 'detail' fit.

    python -m dancecap.refine_hands build <take> [--workers N] [--no-loo] [--no-review]
          -> hands/wrist_gate.{npz,json}, hands/smplx_fingers_build/, hands/contact.{json,npz}, hands/smplx/,
             smplx_hands/{lead,follow}.npz, smplx_detail/{lead,follow}.npz, hands/metrics.{json,npz},
             review/hands_metrics.png (+ review/hands_contact.jpg, hands_render.jpg unless --no-review)
    python -m dancecap.refine_hands review <take>        local and light: the review images from the build outputs
    python -m dancecap.hands_detail outputs <take>       re-write smplx_hands/ + smplx_detail/ (e.g. after a new
                                                         face_expr run) from the build's hands/smplx, no refitting
    python -m dancecap.hands_pod run <take> --build      the build in one Runpod session (no frames uploaded), then
                                                         `refine_hands review <take>` locally

Inputs (all made earlier by the hand pod session, `hands_pod run <take>`): smplx/{lead,follow}.npz (the adopted MAMMA
fit), hands/grid_obs.npz (RTMW / RTMPose / MediaPipe hand keypoints per view on the 30 Hz grid), hands/mano_<m>.npz +
mano_fused_<m>.npz (WiLoR / HaMeR per view and fused over views), mamma/ma_2d (MAMMA's own landmarks), cameras,
dancer masks; optional face/expr_<role>.npz (face_expr) for the detail fit. Parameterised by the take only, so the
same command runs on larissa_kadu_38 once its hand pod session has run.

Steps:
  1 episodes0   contact episodes on the MAMMA fit (the wrist gate must know the hand-hand holds)
  2 wrist gate  per hand and grid frame, the wrist rotation switches from MAMMA's to the mean of the MANO regressors'
                fused world rotations (WiLoR + HaMeR) where (a) the two regressors agree within 30 deg, (b) WiLoR's own
                hand detector found the hand in >= 2 views (+-2 frames), (c) the hand is not in a hand-hand hold (+-6
                frames: inside a held hand the regressors confuse whose hand is whose). The target is the regressors'
                mean Gaussian-smoothed over the gated frames; gaps <= 12 frames between gated runs are bridged by slerp
                between the runs' targets (never the regressors' disagreeing frames), runs < 10 frames dropped, 6-frame
                raised-cosine ramps inside each run. A hand is switched at all only if the independent keypoint judge
                (RTMW / RTMPose / MediaPipe, never seen by the regressors) improves by >= 5 % (median finger-shape
                error / hand length, wrist re-anchored) with >= 55 % of the views better. Since the 2026-10-07 review:
                the gate weight is cut where the gated hand would sink > max(MAMMA's sink, 1 cm) + 2 cm into the
                partner (winding-number test), bridged frames whose regressors are > 90 deg apart are dropped when
                the judge says they are not better there, and the bridged share is reported.
  3 fingers     refine_hands fingers: 45-D hand pose IK to the multi-view keypoints, MANO PCA prior, temporal
                smoothing, pulled towards the fused WiLoR fingers (hybrid route) and started from / regularised towards
                the gated wrists -> hands/smplx_fingers_build/ (+ leave-one-camera-out check)
  4 contact     refine_hands contact on that fit: episodes, then shoulder/elbow/wrist + finger IK that closes holds
                without interpenetration (corrections in a smooth temporal basis; exact distances + winding-number
                inside test, robust c^2 log(1 + (d/c)^2) penalty so a sink the arm cannot fix is left, not forced;
                upper arm / forearm penetration too; finger-finger collision terms; palm displacement capped at 3 cm,
                6 cm where the input hand is far from its keypoints and on bridged hold gaps), merged hand-hand holds
                attracted across short gaps the better record still sees, with the guard (MAMMA's arm + hand landmarks
                overall and per camera, shoulder / elbow turn <= 25 deg and the wrist only further when it turns back
                towards MAMMA's wrist, jitter, palm displacement, finger crossings, arm penetration);
                retried with stronger priors and the input's fingers, last sink-only; when no attempt passes the
                INPUT pose is kept and the episode flagged -> hands/smplx/
  5 outputs     smplx_hands/<role>.npz = the contract fit with refined fingers / wrists / arms + hand extras;
                smplx_detail/<role>.npz = smplx_hands + face_expr's jaw and expression (neck only when face_expr
                says adopt_neck) - same schema as smplx/, so export_unity can read either
  6 metrics     MAMMA vs fingers vs final: hand keypoint reprojection per camera (fused keypoints, HaMeR / WiLoR joints,
                MAMMA's own ma_2d hand landmarks), contact gap distribution inside the episodes, interpenetration,
                jitter -> hands/metrics.json, review/hands_metrics.png
  7 review      review/hands_metrics.png, hands_render.jpg (rendered close-ups of the contacts, sunk hand / arm
                vertices marked), hands_finger_crossing.jpg (worst finger-crossing frame per hand and stage) - on the
                pod; review/hands_contact.jpg (video crops at contact moments in the best 2-3 cameras, before / after
                meshes + hand skeletons + detections) where the frames are (`refine_hands review <take> --video-only`)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import mamma_dyn as md
from . import refine_hands as rh
from .face_common import LABEL, ROLES, crop_pad, label, project
from .hand_kin import FINGER_JOINTS, HAND_EDGES, SIDES, WRIST, hand_vertex_ids

STAGES = (("mamma", "MAMMA c8 fit"), ("fingers", "fingers + wrists"), ("final", "final (+ contact)"))
STAGE_COL = {"mamma": "#2a78d6", "fingers": "#eb6834", "final": "#1baf7a"}   # categorical slots 1-3 (validated set)
OTHER = rh.OTHER


def stage_dirs(work: Path) -> dict:
    return {"mamma": work / "smplx", "fingers": work / "hands" / "smplx_fingers_build", "final": work / "hands" / "smplx"}


# ------------------------------------------------------------------------------------------------ wrist gate

@dataclass
class GateConfig:
    models: tuple = ("wilor", "hamer")
    max_disagree_deg: float = 30.0   # the two regressors' fused world wrist rotations must agree within this
    min_det_views: int = 2           # WiLoR's own YOLO hand detector found the hand in >= this many views ...
    det_window: int = 2              # ... within +- this many grid frames
    handhand_margin: int = 6         # frames excluded around a hand-hand hold / touch (either hand of it)
    fill_gap: int = 12               # gaps up to this long between gated runs are bridged (slerp between the runs'
                                     # end targets, never the regressors' noisy disagreeing frames)
    min_run: int = 10
    ramp: int = 6                    # raised-cosine frames inside each run's ends
    smooth_frames: float = 3.0       # Gaussian sigma of the robust target smoothing over the gated frames
    judge_min_gain: float = 0.05     # the independent keypoint judge must improve by >= 5 % (median) ...
    judge_min_better: float = 0.55   # ... with >= 55 % of the judged views better ...
    judge_min_n: int = 20            # ... over >= 20 judged views
    # bridged frames (regressors disagree > max_disagree_deg) further apart than this are cut from the gate when the
    # independent judge says the bridged wrist is not better there than MAMMA's (2026-10-07 review: up to 180 deg)
    max_bridge_disagree_deg: float = 90.0
    # contact plausibility (2026-10-07 review: the gated wrist turned Kadu's right hand up to 11 cm into Larissa's back,
    # 2-8 cm deeper than MAMMA's hand, and the contact step then wrecked the arm to undo it): per frame the gate weight
    # is cut to the largest of contact_steps whose hand sinks no deeper into the partner than max(MAMMA's sink,
    # contact_free_m) + contact_margin_m (winding-number inside test), eroded over +-contact_erode frames, smoothed
    contact_check: bool = True
    contact_steps: tuple = (1.0, 0.75, 0.5, 0.25)
    contact_free_m: float = 0.01
    contact_margin_m: float = 0.02
    contact_erode: int = 3


def runs_of(g: np.ndarray) -> list[tuple[int, int]]:
    """[(a, b) inclusive] runs of True."""
    g = np.asarray(g, bool)
    if not g.any():
        return []
    d = np.diff(np.concatenate([[0], g.astype(int), [0]]))
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def clean_runs(g: np.ndarray, fill_gap: int, min_run: int) -> np.ndarray:
    """Fill gaps <= fill_gap frames between runs, then drop runs shorter than min_run."""
    g = np.asarray(g, bool).copy()
    rs = runs_of(g)
    for (a0, b0), (a1, b1) in zip(rs, rs[1:]):
        if a1 - b0 - 1 <= fill_gap:
            g[b0 + 1:a1] = True
    out = np.zeros_like(g)
    for a, b in runs_of(g):
        if b - a + 1 >= min_run:
            out[a:b + 1] = True
    return out


def ramp_inside(g: np.ndarray, R: int) -> np.ndarray:
    """(T,) weights: 1 inside each run, raised-cosine ramps of R frames INSIDE the run's ends (never beyond them);
    a run touching the take's start / end is not ramped there."""
    T = len(g)
    w = np.zeros(T)
    for a, b in runs_of(g):
        k = np.arange(a, b + 1)
        u = np.ones(len(k))
        if a > 0:
            u = np.minimum(u, (k - a + 1) / (R + 1))
        if b < T - 1:
            u = np.minimum(u, (b - k + 1) / (R + 1))
        w[a:b + 1] = 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))
    return w


def blend_rotvecs(r0: np.ndarray, r1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per frame geodesic blend of axis-angles r0 -> r1 by w in [0, 1] (r1 may be NaN where w == 0)."""
    from scipy.spatial.transform import Rotation as Rot
    r1 = np.where(np.isfinite(r1), r1, r0)
    R0 = Rot.from_rotvec(r0)
    d = (R0.inv() * Rot.from_rotvec(r1)).as_rotvec()
    return (R0 * Rot.from_rotvec(np.asarray(w, float)[:, None] * d)).as_rotvec()


def _robust_smooth(loc: np.ndarray, keys: np.ndarray, sigma: float, scale_deg: float = 20.0) -> dict:
    """{key: rotation (3,3)}: Gaussian-weighted (sigma frames, +-3 sigma) robust rotation mean (Cauchy reweighting)
    of loc over the other keys around each key."""
    out = {}
    for k in keys:
        nb = keys[np.abs(keys - k) <= 3 * sigma]
        wk = np.exp(-0.5 * ((nb - k) / sigma) ** 2)
        out[int(k)] = rh.robust_rotation_mean(loc[nb][:, None], wk, scale_deg)[0][0]
    return out


def gated_target(loc: np.ndarray, g_raw: np.ndarray, g: np.ndarray, sigma: float, outlier_deg: float = 35.0) -> np.ndarray:
    """(T,3) target local wrist axis-angles on the cleaned gate g (NaN elsewhere) from the regressors' rotations loc
    (T,3,3) on the raw-gate frames: robust Gaussian smoothing over those frames, frames further than outlier_deg from
    their smoothed neighbourhood dropped (isolated flips of a hidden hand), smoothed again, and slerped across the
    bridged frames between them."""
    from scipy.spatial.transform import Rotation as Rot
    from scipy.spatial.transform import Slerp
    T = len(loc)
    out = np.full((T, 3), np.nan)
    keys = np.flatnonzero(np.asarray(g_raw, bool) & np.isfinite(loc).all((1, 2)))
    if not len(keys):
        return out
    sm = _robust_smooth(loc, keys, sigma)
    keys = np.array([k for k in keys if rh._geo_deg(loc[k], sm[int(k)]) <= outlier_deg], int)
    if not len(keys):
        return out
    sm = _robust_smooth(loc, keys, sigma)
    for a, b in runs_of(g):
        kk = keys[(keys >= a) & (keys <= b)]
        if not len(kk):
            continue
        if len(kk) == 1:
            out[a:b + 1] = Rot.from_matrix(sm[int(kk[0])]).as_rotvec()
            continue
        sl = Slerp(kk, Rot.from_matrix(np.stack([sm[int(k)] for k in kk])))
        out[a:b + 1] = sl(np.clip(np.arange(a, b + 1), kk[0], kk[-1])).as_rotvec()
    return out


def smooth_rotvec_track(rv: np.ndarray, mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian (chordal-mean) smoothing of a (T,3) axis-angle track on the masked frames only."""
    from scipy.spatial.transform import Rotation as Rot
    Rm = Rot.from_rotvec(rv).as_matrix()
    out = rv.copy()
    idx = np.arange(len(rv))
    for k in np.flatnonzero(mask):
        nb = idx[np.abs(idx - k) <= int(np.ceil(3 * sigma))]
        wk = np.exp(-0.5 * ((nb - k) / sigma) ** 2)
        out[k] = Rot.from_matrix(rh._so3(np.einsum("n,nab->ab", wk, Rm[nb]))).as_rotvec()
    return out


def det_views_grid(work: Path, model: str, grid: np.ndarray) -> dict:
    """{(role, side): (T,) number of cameras whose job at that grid frame the regressor's OWN detector found}."""
    z = rh._npz(work / "hands" / f"mano_{model}.npz")
    jobs = json.loads((work / "hands" / "jobs.json").read_text())["jobs"]
    tmap = {(j["cam"], int(j["i"]), j["role"], j["side"]): float(j["t"]) for j in jobs}
    dt = float(np.median(np.diff(grid))) if len(grid) > 1 else 1 / 30
    T = len(grid)
    sets: dict = {}
    for n in range(len(z["job_cam"])):
        if not z["ok"][n] or int(z["box_src"][n]) != 1:
            continue
        key = (str(z["job_cam"][n]), int(z["job_i"][n]), str(z["job_role"][n]), str(z["job_side"][n]))
        t = tmap.get(key)
        if t is None:
            continue
        k = int(np.argmin(np.abs(grid - t)))
        if abs(grid[k] - t) > 0.5 * dt + 1e-3:
            continue
        sets.setdefault(key[2:], [set() for _ in range(T)])[k].add(key[0])
    return {rs: np.array([len(x) for x in v]) for rs, v in sets.items()}


def handhand_frames(episodes: list, T: int, margin: int) -> dict:
    """{(role, side): (T,) bool} frames in (or within margin of) a hand-hand hold / touch, for BOTH hands of it."""
    out = {(r, s): np.zeros(T, bool) for r in ROLES for s in SIDES}
    for e in episodes:
        if not e.get("hand_hand") or e["kind"] not in ("hold", "touch"):
            continue
        a, b = max(0, e["i0"] - margin), min(T - 1, e["i1"] + margin)
        out[(e["role"], e["side"])][a:b + 1] = True
        ps = "left" if str(e["region"]).startswith("left") else "right"
        out[(OTHER[e["role"]], ps)][a:b + 1] = True
    return out


def gate_frames(disagree_deg: np.ndarray, det_views: np.ndarray, handhand: np.ndarray, cfg: GateConfig) -> np.ndarray:
    """(T,) bool raw gate before cleaning: regressors agree, own detector sees the hand, no hand-hand hold."""
    from scipy.ndimage import maximum_filter1d
    dv = maximum_filter1d(np.asarray(det_views, float), size=2 * cfg.det_window + 1, mode="nearest")
    agree = np.isfinite(disagree_deg) & (np.nan_to_num(disagree_deg, nan=999.0) <= cfg.max_disagree_deg)
    return agree & (dv >= cfg.min_det_views) & ~np.asarray(handhand, bool)


def judge_errors(J: np.ndarray, obs_rs: dict, gc: dict, Jref: np.ndarray, mask: np.ndarray, min_pts: int = 10,
                 min_len_px: float = 12.0) -> dict:
    """{(frame, cam): finger-shape error / hand length} of hand joints J (T,21,3) against 2D keypoints obs_rs
    {cam: (kp (T,21,2), w (T,21))} with the wrist re-anchored on the detection (refine_hands cross_judge measure);
    hand length from Jref."""
    out = {}
    for c, (kp, w) in obs_rs.items():
        for k in np.flatnonzero(mask):
            ok = w[k] > 0
            if ok.sum() < min_pts or not gc[c]["ok"][k]:
                continue
            P = gc[c]["K"][k] @ np.concatenate([gc[c]["R"][k], gc[c]["t"][k][:, None]], 1)
            pm = rh._proj(P, Jref[k])
            L = float(np.linalg.norm(pm[0] - pm[12]))
            if L < min_len_px:
                continue
            okf = ok.copy()
            okf[0] = False
            p = rh._proj(P, J[k])
            out[(int(k), c)] = float(np.median(np.linalg.norm((p - p[0] + kp[k, 0])[okf] - kp[k, okf], axis=1))) / L
    return out


def hand_sink(dn: rh.Dancer, side: str, wrists: list, frames: np.ndarray, Vo: np.ndarray, device: str | None = None,
              topo=None) -> np.ndarray:
    """(len(wrists), len(frames)) depth (m, >= 0) of the deepest hand vertex inside the partner Vo (T,nv,3) for each
    candidate local wrist track (T,3), the rest of the pose being dn's (winding-number inside test)."""
    import torch
    from . import mesh_contact as mc
    hid = hand_vertex_ids(dn.mdir, side)
    kin = rh.SmplxKin(dn.mdir, np.asarray(dn.d["betas"], float).reshape(-1), "cpu", vert_subset=hid)
    topo = topo or mc.Topology(dn.kin.faces, len(dn.kin.weights_np))
    out = np.zeros((len(wrists), len(frames)))
    for n, wr in enumerate(wrists):
        P = dn.pose[frames].clone()
        P[:, WRIST[side]] = torch.tensor(np.asarray(wr, float)[frames])
        with torch.no_grad():
            V = kin.forward(P, dn.transl[frames], want_verts=True)["verts"].numpy()
        sd = mc.signed_distance_frames(V, Vo[frames], topo, device=device)["sd"]
        out[n] = np.maximum(0.0, -sd.min(1))
    return out


def contact_limit(dn: rh.Dancer, side: str, wr0: np.ndarray, target: np.ndarray, wg: np.ndarray, Vo: np.ndarray,
                  cfg: GateConfig, device: str | None = None) -> tuple[np.ndarray, dict]:
    """Cut the gate weight wg (T,) where the gated hand would sink into the partner Vo (T,nv,3) deeper than MAMMA's
    hand + margin (GateConfig.contact_*). -> (new weights, summary)."""
    from scipy.ndimage import gaussian_filter1d, minimum_filter1d
    fr = np.flatnonzero(wg > 0)
    steps = [float(c) for c in cfg.contact_steps]
    cands = [wr0] + [blend_rotvecs(wr0, target, c * wg) for c in steps]
    sink = hand_sink(dn, side, cands, fr, Vo, device)
    allow = np.maximum(sink[0], cfg.contact_free_m) + cfg.contact_margin_m
    c = np.zeros(len(wg))
    cf = np.zeros(len(fr))
    for i in range(len(steps) - 1, -1, -1):          # smallest step first, the largest allowed wins
        cf = np.where(sink[1 + i] <= allow, steps[i], cf)
    c[fr] = cf
    c[wg <= 0] = 1.0                                 # outside the gate the weight is 0 anyway
    c = minimum_filter1d(c, 2 * cfg.contact_erode + 1, mode="nearest")
    c = np.clip(gaussian_filter1d(c, 2.0, mode="nearest"), 0, 1)
    w = wg * c
    sink_on = sink[1] if len(steps) else sink[0]
    info = {"frames_tested": int(len(fr)), "frames_cut": int(((c < 0.999) & (wg > 0)).sum()),
            "weight_mean_before": round(float(wg[fr].mean()), 3) if len(fr) else None,
            "weight_mean_after": round(float(w[fr].mean()), 3) if len(fr) else None,
            "sink_cm_max_mamma": round(float(sink[0].max()) * 100, 2) if len(fr) else None,
            "sink_cm_max_full_gate": round(float(sink_on.max()) * 100, 2) if len(fr) else None,
            "sink_cm_max_after": round(float(hand_sink(dn, side, [blend_rotvecs(wr0, target, w)], fr, Vo,
                                                       device)[0].max()) * 100, 2) if len(fr) else None}
    return w, info


def _judge_stats(e_off: dict, e_on: dict) -> dict | None:
    keys = sorted(set(e_off) & set(e_on))
    if not keys:
        return None
    a_off = np.array([e_off[k] for k in keys])
    a_on = np.array([e_on[k] for k in keys])
    return {"n": len(keys), "mamma_wrist": round(float(np.median(a_off)), 3),
            "gated_wrist": round(float(np.median(a_on)), 3),
            "gain": round(1 - float(np.median(a_on)) / max(float(np.median(a_off)), 1e-9), 3),
            "better_frac": round(float((a_on < a_off).mean()), 3)}


def _gate_hand(dn, s: str, base: dict, wr0: np.ndarray, loc: np.ndarray, g_raw: np.ndarray, hh: np.ndarray,
               far: np.ndarray, cut_far: bool, partner, obs_rs: dict, gc: dict, cfg: GateConfig) -> dict:
    """One hand's gate from the raw gate: cleaned runs (optionally never bridging across `far` frames), regressor
    target, ramps, contact limit, smoothing; the independent keypoint judge on all gated frames, on the agreeing
    (measured) frames and on the far-bridged frames. -> dict(wg, wr_on, judge*, contact_check, wrist_change_deg_median)"""
    import torch
    from scipy.spatial.transform import Rotation as Rot
    g = clean_runs(g_raw, cfg.fill_gap, cfg.min_run) & ~hh
    if cut_far:
        g &= ~(far & ~g_raw)
    g = clean_runs(g, 0, cfg.min_run)
    wg = ramp_inside(g, cfg.ramp)
    target = gated_target(loc, g_raw, g, cfg.smooth_frames)
    wg = np.where(np.isfinite(target).all(1), wg, 0.0)
    out = {}
    if cfg.contact_check and (wg > 0).any():
        wg, out["contact_check"] = contact_limit(dn, s, wr0, target, wg, partner.verts(), cfg)
    wr_on = blend_rotvecs(wr0, target, wg)
    near = np.convolve((wg > 0).astype(float), np.ones(9), "same") > 0
    wr_on = smooth_rotvec_track(wr_on, near, 1.5)
    out.update({"wg": wg, "wr_on": wr_on, "judge": None, "judge_measured": None, "judge_far_bridged": None})
    on = wg > 0.5
    if on.any():
        Rm = Rot.from_rotvec(wr0[on]).as_matrix()
        Ro = Rot.from_rotvec(wr_on[on]).as_matrix()
        out["wrist_change_deg_median"] = round(float(np.median(rh._geo_deg(Rm, Ro))), 1)
        # independent judge: same (MAMMA) fingers, MAMMA wrist vs gated wrist
        f0 = FINGER_JOINTS[s][0]
        th = dn.pose[:, f0:f0 + 15]
        with torch.no_grad():
            J_on = dn.kin.hand_fk(base, torch.tensor(wr_on), th).numpy()
        J_off = dn.hands[s]
        e_off = judge_errors(J_off, obs_rs, gc, J_off, on)
        e_on = judge_errors(J_on, obs_rs, gc, J_off, on)
        out["judge"] = _judge_stats(e_off, e_on)
        for key, m in (("judge_measured", g_raw), ("judge_far_bridged", far & ~g_raw)):
            sub = {k: v for k, v in e_off.items() if m[k[0]]}
            out[key] = _judge_stats(sub, {k: v for k, v in e_on.items() if m[k[0]]})
    return out


def wrist_gate(work: Path, dancers: dict, gc: dict, episodes: list, cfg: GateConfig | None = None) -> dict:
    """Gated wrist rotations per hand -> hands/wrist_gate.npz (<role>_<side>_wrist (T,3) local axis-angle, _weight,
    _disagree_deg, _det_views) + wrist_gate.json. Without both regressors' fused files every weight is 0."""
    cfg = cfg or GateConfig()
    out = work / "hands"
    grid = dancers["lead"].t
    T = len(grid)
    have = [m for m in cfg.models if (out / f"mano_fused_{m}.npz").exists() and (out / f"mano_{m}.npz").exists()]
    fused = {m: rh._npz(out / f"mano_fused_{m}.npz") for m in have}
    dets = det_views_grid(work, cfg.models[0], grid) if cfg.models[0] in have else {}
    hh = handhand_frames(episodes, T, cfg.handhand_margin)
    obs = rh.load_grid_obs(work, "fused") if (out / "grid_obs.npz").exists() else {}
    arrays, summary = {"t": grid}, {"config": asdict(cfg), "models": have, "hands": {}}
    for r in ROLES:
        dn = dancers[r]
        for s in SIDES:
            wr0 = dn.pose[:, WRIST[s]].numpy().copy()
            e = {"adopted": False}
            w = np.zeros(T)
            wr = wr0.copy()
            dis = np.full(T, np.nan)
            dv = dets.get((r, s), np.zeros(T, int))
            if len(have) == len(cfg.models) and len(have) >= 2:
                Rw = [fused[m][f"{r}_{s}_wrist_world"] for m in have]
                fin = np.all([np.isfinite(x).all((1, 2)) for x in Rw], 0)
                dis[fin] = rh._geo_deg(Rw[0][fin], Rw[1][fin])
                g_raw = gate_frames(dis, dv, hh[(r, s)], cfg) & fin
                base = dn.kin.hand_base(dn.out, s)
                Rp = base["Rp"].numpy()
                loc = np.full((T, 3, 3), np.nan)
                if fin.any():   # mean of the regressors' world wrists -> local wrist rotation (relative to the elbow)
                    M = rh._so3(np.sum([x[fin] for x in Rw], 0))
                    loc[fin] = np.transpose(Rp[fin], (0, 2, 1)) @ M
                o = obs.get((r, s), {})
                far = np.isfinite(dis) & (np.nan_to_num(dis, nan=0.0) > cfg.max_bridge_disagree_deg)
                res = _gate_hand(dn, s, base, wr0, loc, g_raw, hh[(r, s)], far, False, dancers[OTHER[r]], o, gc, cfg)
                fb = res["judge_far_bridged"]
                if fb and fb["n"] >= 10 and fb["gain"] <= 0:
                    # the bridged wrist is no better than MAMMA's where the regressors are > 90 deg apart: cut those
                    res = _gate_hand(dn, s, base, wr0, loc, g_raw, hh[(r, s)], far, True, dancers[OTHER[r]], o, gc,
                                     cfg)
                    res["far_bridges_cut"] = {"why": "bridged frames with regressors > "
                                              f"{cfg.max_bridge_disagree_deg:g} deg apart: judge gain {fb['gain']}",
                                              "judge_before_cut": fb}
                wg, wr_on = res.pop("wg"), res.pop("wr_on")
                on = wg > 0
                bridged = on & ~(np.nan_to_num(dis, nan=999.0) <= cfg.max_disagree_deg)
                e.update({"frames_gated": int((wg > 0.5).sum()), "frames_raw_gate": int(g_raw.sum()),
                          "frames_weight_gt0": int(on.sum()),
                          # gated frames whose regressors do NOT agree within max_disagree_deg (or one is missing):
                          # their wrist is slerped between neighbouring agreeing frames, not measured
                          "frames_bridged": int(bridged.sum()),
                          "bridged_share": round(float(bridged.sum() / max(on.sum(), 1)), 3),
                          "frames_bridged_gt_far": int((bridged & far).sum()),
                          "bridged_disagree_deg_max": (round(float(np.nanmax(np.where(bridged, dis, np.nan))), 1)
                                                       if np.isfinite(np.where(bridged, dis, np.nan)).any() else None),
                          "frames_regressors_agree": int((np.nan_to_num(dis, nan=999) <= cfg.max_disagree_deg).sum()),
                          "frames_handhand_excluded": int(hh[(r, s)].sum()),
                          "regressor_disagreement_deg_median": round(float(np.nanmedian(dis)), 1) if fin.any() else None,
                          "det_views_median": float(np.median(dv)), **res})
                if (wg > 0.5).any() and e.get("judge"):
                    jd = e["judge"]
                    e["adopted"] = bool(jd["n"] >= cfg.judge_min_n and jd["gain"] >= cfg.judge_min_gain
                                        and jd["better_frac"] >= cfg.judge_min_better)
                    e["why"] = ("independent keypoint judge improves" if e["adopted"] else
                                "independent keypoint judge does not improve enough: MAMMA wrist kept")
                else:
                    e["why"] = "no frame passes the gate"
                if e["adopted"]:
                    w, wr = wg, wr_on
            else:
                e["why"] = f"needs fused wrists of {cfg.models} (have {have})"
            arrays[f"{r}_{s}_wrist"], arrays[f"{r}_{s}_weight"] = wr, w
            arrays[f"{r}_{s}_disagree_deg"], arrays[f"{r}_{s}_det_views"] = dis, dv
            summary["hands"].setdefault(r, {})[s] = e
            print(f"[wrist gate] {r} {s}: {json.dumps(e)}", flush=True)
    np.savez_compressed(out / "wrist_gate.npz", **arrays)
    (out / "wrist_gate.json").write_text(json.dumps(summary, indent=1))
    return {"arrays": arrays, "summary": summary}


# ------------------------------------------------------------------------------------------------ outputs

def merge_face(d: dict, z: dict, adopt_neck: bool | None = None) -> tuple[dict, str]:
    """Contract dict d + face_expr npz z -> (detail dict, note). jaw_pose and expression are taken over; the neck
    correction only when adopt_neck (default: z's own 'adopt_neck' flag, else False - face_expr's docstring says its
    neck correction overfits single views). Refuses a different time grid."""
    t, tz = np.asarray(d["t"], float), np.asarray(z["t"], float)
    if len(t) != len(tz) or np.abs(t - tz).max() > 1e-4:
        return dict(d), "face not merged: face/expr grid differs from the fit's grid"
    out = dict(d)
    out["jaw_pose"] = np.asarray(z["jaw_pose"], float)
    out["expression"] = np.asarray(z["expression"], np.float32)
    for k in ("expression_rest", "n_expr"):
        if k in z:
            out[k] = np.asarray(z[k])
    if "confidence" in z:
        out["face_confidence"] = np.asarray(z["confidence"], float)
    if adopt_neck is None:
        adopt_neck = bool(np.asarray(z.get("adopt_neck", False)))
    note = "face_expr jaw + expression"
    if adopt_neck and "neck_pose" in z:
        bp = np.asarray(out["body_pose"], float).copy()
        bp[:, 33:36] = np.asarray(z["neck_pose"], float)
        out["body_pose"] = bp
        note += " + neck"
    out["face_neck_adopted"] = np.array(bool(adopt_neck and "neck_pose" in z))
    return out, note


def contact_state(episodes: list, role: str, T: int) -> np.ndarray:
    """(T,2) int8 per hand of `role` (left, right): 0 free, 1 touch, 2 hold. A hand-hand hold merged from both hands'
    records (refine_hands.merge_hold_records: 'union', 'partner_side', 'attract_runs') gives BOTH hands the same state
    - hold on its attracted (incl. bridged) frames, touch on the rest of the union - and the partner's twin record
    ('merged_into') is not counted again, so a runtime contact lock never sees the two hands of one hold disagree."""
    st = np.zeros((T, 2), np.int8)
    for e in episodes:
        if e["kind"] not in ("hold", "touch") or e.get("merged_into") is not None:
            continue
        if "union" in e and e.get("partner_side"):
            hands = [(e["role"], e["side"]), (OTHER[e["role"]], e["partner_side"])]
            a, b = e["union"]
            v = np.zeros(T, np.int8)
            v[a:b + 1] = 1
            v[rh.attract_mask(e, T)] = 2 if e["kind"] == "hold" else 1
            for rr, ss in hands:
                if rr == role:
                    j = SIDES.index(ss)
                    st[:, j] = np.maximum(st[:, j], v)
            continue
        if e["role"] != role:
            continue
        v = 2 if e["kind"] == "hold" else 1
        j = SIDES.index(e["side"])
        st[e["i0"]:e["i1"] + 1, j] = np.maximum(st[e["i0"]:e["i1"] + 1, j], v)
    return st


STALE_KEYS = ("quality", "quality_parts", "reproj_parts_px", "reproj_px")


def recompute_quality(d: dict, mdir: Path, pid: int, ma2d: dict, gc: dict) -> dict:
    """MAMMA's per-frame quality fields (mamma_dyn.export_smplx definitions) for a changed fit d: reprojection of the
    fit's 512 points against ma_2d's visible landmarks (vis > 0.5), per frame the median over cameras and points,
    overall and per part; quality = exp(-worst of body / feet / head px / 10), repaired frames capped at 0.1."""
    import torch
    from .hand_kin import full_pose, model_npz
    nv = len(np.load(model_npz(mdir), allow_pickle=True)["weights"])
    kin = rh.SmplxKin(mdir, np.asarray(d["betas"], float).reshape(-1), "cpu", vert_subset=np.arange(nv))
    with torch.no_grad():
        V = kin.forward(torch.tensor(full_pose(d)), torch.tensor(np.asarray(d["transl"], float)),
                        want_verts=True)["verts"].numpy()
    D = md._load_verts512()
    pts = np.matmul(D, V)                                                             # (T,512,3)
    parts = md.part_index()
    T = len(pts)
    errs = {k: [[] for _ in range(T)] for k in ("all", *parts)}
    for c, (lm, _, vis) in ma2d.items():
        g = gc.get(c)
        if g is None:
            continue
        for f in range(T):
            if not g["ok"][f]:
                continue
            Xc = pts[f] @ g["R"][f].T + g["t"][f]
            px = Xc[:, :2] / np.maximum(Xc[:, 2:], 1e-6) @ g["K"][f][:2, :2].T + g["K"][f][:2, 2]
            obs = lm[f, pid]
            ok = (vis[f, pid] > 0.5) & (np.abs(obs).sum(-1) > 0) & (Xc[:, 2] > 0)
            e = np.linalg.norm(px - obs, axis=1)
            errs["all"][f].extend(e[ok].tolist())
            for k, idx in parts.items():
                errs[k][f].extend(e[idx][ok[idx]].tolist())
    rp = {k: np.array([np.median(x) if len(x) >= 3 else np.nan for x in v]) for k, v in errs.items()}
    names = [k for k in md.PARTS if k in rp]
    part_px = np.stack([rp[k] for k in names], 1)
    core = np.stack([rp[k] for k in names if "hand" not in k], 1)
    reproj = rp["all"]
    worst = np.nanmax(np.where(np.isfinite(core), core, -np.inf), 1)
    worst = np.where(np.isfinite(worst), np.maximum(worst, np.nan_to_num(reproj, nan=0.0)), reproj)
    quality = np.exp(-np.nan_to_num(worst, nan=1e3) / 10.0)
    bad = np.asarray(d.get("repaired", np.zeros(T, bool)), bool)
    quality[bad] = np.minimum(quality[bad], 0.1)
    return {"quality": quality, "quality_parts": np.exp(-np.nan_to_num(part_px, nan=1e3) / 10.0).astype(np.float32),
            "reproj_parts_px": part_px.astype(np.float32), "reproj_px": reproj, "parts": np.array(names)}


def write_outputs(take, gate: dict | None = None, metrics_arrays: dict | None = None) -> dict:
    """hands/smplx (contact result) -> smplx_hands/<role>.npz (+ extras) and smplx_detail/<role>.npz (+ face)."""
    import torch
    work = take.work
    src = work / "hands" / "smplx"
    cs = json.loads((work / "hands" / "contact.json").read_text())
    eps = cs["episodes"]
    if gate is None and (work / "hands" / "wrist_gate.npz").exists():
        gate = {"arrays": rh._npz(work / "hands" / "wrist_gate.npz")}
    if metrics_arrays is None and (work / "hands" / "metrics.npz").exists():
        metrics_arrays = rh._npz(work / "hands" / "metrics.npz")   # re-merge (e.g. a new face_expr) without metrics
    fj = work / "hands" / "fingers_build.json"
    fing = json.loads(fj.read_text()) if fj.exists() else {}
    res = {"smplx_hands": {}, "smplx_detail": {}, "quality": {}}
    ma2d, _ = rh.ma2d_obs(work)
    gc = None
    if ma2d:
        cams = rh.fit_cams(work)
        grid = np.asarray(rh.load_fit(work, "lead", src)["t"], float)
        gc = rh.grid_cameras(work, md.load_tracks(work, cams), grid)
    for d_ in ("smplx_hands", "smplx_detail"):
        (work / d_).mkdir(parents=True, exist_ok=True)
    for r in ROLES:
        d = rh.load_fit(work, r, src)
        T = len(d["t"])
        if gate is not None:
            d["wrist_gate_weight"] = np.stack([np.asarray(gate["arrays"][f"{r}_{s}_weight"], float) for s in SIDES], 1)
        d["contact_state"] = contact_state(eps, r, T)
        if metrics_arrays is not None and f"final|{r}|left|gap" in metrics_arrays:
            d["contact_gap_m"] = np.stack([metrics_arrays[f"final|{r}|{s}|gap"] for s in SIDES], 1)
        mine = [{k: e.get(k) for k in ("role", "side", "partner", "region", "kind", "t0", "t1", "i0", "i1",
                                       "gap_cm_median", "penetration_cm_max", "evidence_median", "refined",
                                       "attract_runs", "bridged_gaps", "union", "partner_side", "merged_into") if k in e}
                for e in eps if e["role"] == r or e.get("partner") == r]
        # MAMMA's per-frame quality fields describe the MAMMA fit: recomputed for this fit (same definitions)
        if ma2d and gc is not None:
            base = rh.load_fit(work, r)
            q_new = recompute_quality(d, rh.model_dir(d), ROLES.index(r), ma2d, gc)
            q_chk = recompute_quality(base, rh.model_dir(base), ROLES.index(r), ma2d, gc)
            stored = np.asarray(base.get("reproj_px", np.full(T, np.nan)), float)
            res["quality"][r] = {
                "recomputed": True,
                "check_vs_mamma_stored_reproj_px_median_absdiff": round(float(np.nanmedian(np.abs(
                    q_chk["reproj_px"] - stored))), 3),
                "reproj_px_median_mamma": round(float(np.nanmedian(q_chk["reproj_px"])), 2),
                "reproj_px_median_this": round(float(np.nanmedian(q_new["reproj_px"])), 2),
                "quality_median_mamma": round(float(np.median(q_chk["quality"])), 3),
                "quality_median_this": round(float(np.median(q_new["quality"])), 3)}
            d.update(q_new)
            d["quality_source"] = np.array("recomputed by dancecap.hands_detail for this fit (ma_2d landmarks, "
                                           "dancecap cameras); repaired / suspect / reproj_px_raw / contact / "
                                           "floor_contact are MAMMA's (evidence or raw-fit flags)")
        else:
            for k in STALE_KEYS:
                d.pop(k, None)
            d["quality_source"] = np.array("dropped: no ma_2d landmarks to recompute MAMMA's quality fields")
        d["contact_episodes"] = np.array(json.dumps(mine))
        d["hands_source"] = np.array(json.dumps({
            "fingers": ("hybrid IK: multi-view keypoints + fused " + fing["config"]["mano_model"]
                        if fing.get("config", {}).get("mano_model") else "keypoint IK")
                       + " (refine_hands fingers, tag build; wrists from hands/wrist_gate.npz)",
            "contact": "refine_hands contact (arm guard)", "base": "smplx/ (adopted MAMMA fit)"}))
        d["source"] = np.array(str(d.get("source", "")) + " + dancecap.hands_detail build")
        np.savez_compressed(work / "smplx_hands" / f"{r}.npz", **d)
        res["smplx_hands"][r] = str(work / "smplx_hands" / f"{r}.npz")
        # the face block face_expr.build adopted on held-out views (2026-10-07; face/expr_<role>.npz is the earlier
        # research fit those checks rejected - per-frame jaw / expression and neck do not generalise across cameras)
        fp = work / "smplx_face" / f"{r}.npz"
        if fp.exists():
            dd, note = merge_face(d, rh._npz(fp))
            if dd.get("face_neck_adopted", False):
                kin = rh.SmplxKin(rh.model_dir(dd), np.asarray(dd["betas"], float).reshape(-1))
                from .hand_kin import full_pose
                with torch.no_grad():
                    dd["joints"] = kin.forward(torch.tensor(full_pose(dd)),
                                               torch.tensor(np.asarray(dd["transl"], float)))["joints"].numpy()
        else:
            dd, note = dict(d), "no smplx_face/<role>.npz: jaw / expression stay MAMMA's (zero)"
        dd["face_source"] = np.array(note)
        dd["source"] = np.array(str(d["source"]) + (" + face_expr" if note.startswith("face_expr") else ""))
        np.savez_compressed(work / "smplx_detail" / f"{r}.npz", **dd)
        res["smplx_detail"][r] = {"path": str(work / "smplx_detail" / f"{r}.npz"), "face": note}
    return res


# ------------------------------------------------------------------------------------------------ metrics

def _hand_lm_ids(mdir: Path) -> dict:
    """MAMMA's 63 dense hand landmarks per side -> SMPL-X vertex ids (side checked against the LBS weights)."""
    from . import env
    parts = np.load(env.mamma_data() / "body_models" / "downsampled_verts" / "smplx_512_body_parts.npz")
    D = np.asarray(md._load_verts512().argmax(1))
    out = {}
    for s in SIDES:
        idx = np.asarray(parts[f"{s}_hand"])
        hv = set(hand_vertex_ids(mdir, s, thresh=0.3).tolist())
        frac = np.mean([v in hv for v in D[idx]])
        if frac < 0.5:      # MAMMA's part names would be mirrored: use the other list
            idx = np.asarray(parts[f"{'right' if s == 'left' else 'left'}_hand"])
        out[s] = (idx, D[idx])
    return out


def _stats(x: np.ndarray, scale: float = 1.0, nd: int = 2) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q = np.percentile(x, [5, 25, 50, 75, 95]) * scale
    return {"n": int(len(x)), "p5": round(q[0], nd), "p25": round(q[1], nd), "p50": round(q[2], nd),
            "p75": round(q[3], nd), "p95": round(q[4], nd), "max": round(float(x.max() * scale), nd)}


def gap_stats(gap: np.ndarray) -> dict:
    """Signed gap series (m, negative = inside the partner) of a contact episode -> cm percentiles, the share within
    1 cm of touching, floating (> 1 cm / > 2 cm above the partner) and sunk in (> 1 cm), the deepest penetration and the
    largest float. The gaps come from hand_partner_geometry (exact distance, winding-number inside test)."""
    g = np.asarray(gap, float)
    g = g[np.isfinite(g)]
    if not len(g):
        return {"n": 0}
    out = _stats(g, 100.0)
    out.update({"touching_within_1cm": round(float((np.abs(g) <= 0.01).mean()), 3),
                "float_gt_1cm": round(float((g > 0.01).mean()), 3),
                "apart_gt_2cm": round(float((g > 0.02).mean()), 3),
                "sunk_gt_1cm": round(float((g < -0.01).mean()), 3),
                "penetration_cm_max": round(float(max(0.0, -g.min()) * 100), 2),
                "penetration_cm_p95": round(float(max(0.0, -np.percentile(g, 5)) * 100), 2),
                "float_cm_max": round(float(max(0.0, g.max()) * 100), 2)})
    return out


def hf_rms(x: np.ndarray, window: int = 9, order: int = 3, edge: int = 3) -> float:
    """Jitter: RMS (over points and frames, edges dropped) of the residual of x (T, N, 3) from a local cubic
    (Savitzky-Golay, 9 frames = 0.3 s at 30 Hz). Smooth motion up to ~2-3 Hz passes the cubic almost untouched, so
    the residual is frame-to-frame noise; still, compare stages, not absolute values (fast real motion leaks in)."""
    from scipy.signal import savgol_filter
    x = np.asarray(x, float)
    if len(x) < max(window, 2 * edge + 3):
        return float("nan")
    r = x - savgol_filter(x, window, order, axis=0, mode="interp")
    r = r[edge:len(x) - edge]
    return float(np.sqrt(np.nanmean(np.sum(r.reshape(len(r), -1, x.shape[-1]) ** 2, -1))))


def _local_ang_acc_deg(rv: np.ndarray) -> np.ndarray:
    """(T,3) local axis-angles -> (T-2,) angular acceleration magnitude (deg / frame^2) of the rotation track."""
    from scipy.spatial.transform import Rotation as Rot
    R = Rot.from_rotvec(np.asarray(rv, float))
    w = (R[:-1].inv() * R[1:]).as_rotvec()                    # per-frame relative rotation (angular velocity)
    return np.degrees(np.linalg.norm(w[1:] - w[:-1], axis=1)) if len(w) > 1 else np.zeros(0)


def jitter(dn: rh.Dancer, side: str) -> dict:
    """Jitter of one hand (mm, residual RMS from a local cubic, hf_rms): fingers relative to the wrist, the wrist in
    the world, the elbow; finger joint angular acceleration p95 (deg/frame^2); shoulder / elbow local angular
    acceleration p95 (deg/frame^2); world wrist acceleration p95 / max and fingertip second difference p95 (mm/frame^2)
    - the adversarial review's measures."""
    H = dn.hands[side]
    f0 = FINGER_JOINTS[side][0]
    th = dn.pose[:, f0:f0 + 15].numpy()
    acc = np.degrees(np.linalg.norm(th[2:] - 2 * th[1:-1] + th[:-2], axis=-1)) if len(th) > 2 else np.zeros((0, 15))
    el = dn.joints[:, 18 if side == "left" else 19]
    P = dn.pose.numpy()
    sh_j, el_j = (16, 18) if side == "left" else (17, 19)
    wa = np.linalg.norm(H[2:, 0] - 2 * H[1:-1, 0] + H[:-2, 0], axis=-1) * 1000 if len(H) > 2 else np.zeros(0)
    tips = H[:, [4, 8, 12, 16, 20]]
    ta = np.linalg.norm(tips[2:] - 2 * tips[1:-1] + tips[:-2], axis=-1) * 1000 if len(H) > 2 else np.zeros(0)

    def p95(x):
        return round(float(np.percentile(x, 95)), 2) if np.size(x) else None
    return {"fingers_rel_wrist_mm": round(hf_rms(H[:, 1:] - H[:, :1]) * 1000, 2),
            "wrist_mm": round(hf_rms(H[:, :1]) * 1000, 2), "elbow_mm": round(hf_rms(el[:, None]) * 1000, 2),
            "finger_ang_acc_deg_p95": p95(acc),
            "shoulder_ang_acc_deg_p95": p95(_local_ang_acc_deg(P[:, sh_j])),
            "elbow_ang_acc_deg_p95": p95(_local_ang_acc_deg(P[:, el_j])),
            "wrist_ang_acc_deg_p95": p95(_local_ang_acc_deg(P[:, WRIST[side]])),
            "wrist_acc_mm_p95": p95(wa), "wrist_acc_mm_max": round(float(wa.max()), 2) if wa.size else None,
            "fingertip_acc_mm_p95": p95(ta)}


def kp_reprojection(dn: rh.Dancer, side: str, obs_rs: dict, gc: dict, cams: list[str], Jref: np.ndarray,
                    min_pts: int = 10, min_len_px: float = 12.0) -> dict:
    """Per camera: median |projected hand joint - detected keypoint| (px and / hand length, absolute) and the
    finger-shape error (wrist re-anchored, / hand length), over grid frames with >= min_pts confident keypoints."""
    J = dn.hands[side]
    out = {}
    allr = []
    for c in cams:
        if c not in obs_rs:
            continue
        kp, w = obs_rs[c]
        rows = []
        for k in range(len(J)):
            ok = w[k] > 0
            if ok.sum() < min_pts or not gc[c]["ok"][k]:
                continue
            P = gc[c]["K"][k] @ np.concatenate([gc[c]["R"][k], gc[c]["t"][k][:, None]], 1)
            pr = rh._proj(P, Jref[k])
            L = float(np.linalg.norm(pr[0] - pr[12]))
            if L < min_len_px:
                continue
            p = rh._proj(P, J[k])
            e = np.linalg.norm(p[ok] - kp[k, ok], axis=1)
            okf = ok.copy()
            okf[0] = False
            a = np.linalg.norm((p - p[0] + kp[k, 0])[okf] - kp[k, okf], axis=1)
            rows.append((float(np.median(e)), float(np.median(e)) / L, float(np.median(a)) / L, L))
        if rows:
            A = np.array(rows)
            out[c] = {"n": len(A), "px": round(float(np.median(A[:, 0])), 2), "len": round(float(np.median(A[:, 1])), 3),
                      "shape_len": round(float(np.median(A[:, 2])), 3), "hand_px": round(float(np.median(A[:, 3])), 1)}
            allr.append(A)
    if allr:
        A = np.concatenate(allr)
        out["all"] = {"n": len(A), "px": round(float(np.median(A[:, 0])), 2), "len": round(float(np.median(A[:, 1])), 3),
                      "shape_len": round(float(np.median(A[:, 2])), 3), "hand_px": round(float(np.median(A[:, 3])), 1)}
    return out


def ma2d_hand_reprojection(dn: rh.Dancer, side: str, lm_ids: tuple, ma2d: dict, gc: dict, cams: list[str]) -> dict:
    """Per camera: median px error of MAMMA's own 63 dense hand landmarks (vis > 0.5) - the evidence the c8 fit was
    made from; the refinement should not move far from it."""
    import torch
    idx512, vids = lm_ids
    kin = rh.SmplxKin(dn.mdir, np.asarray(dn.d["betas"], float).reshape(-1), "cpu", vert_subset=vids)
    with torch.no_grad():
        V = kin.forward(dn.pose, dn.transl, want_verts=True)["verts"].numpy()
    pid = ROLES.index(dn.role)
    out, allv = {}, []
    for c in cams:
        if c not in ma2d:
            continue
        lm, _, vis = ma2d[c]
        errs = []
        for k in range(len(V)):
            if not gc[c]["ok"][k]:
                continue
            ok = vis[k, pid, idx512] > 0.5
            if ok.sum() < 5:
                continue
            P = gc[c]["K"][k] @ np.concatenate([gc[c]["R"][k], gc[c]["t"][k][:, None]], 1)
            errs.append(float(np.median(np.linalg.norm(rh._proj(P, V[k][ok]) - lm[k, pid, idx512][ok], axis=1))))
        if errs:
            out[c] = {"n": len(errs), "px": round(float(np.median(errs)), 2)}
            allv += errs
    if allv:
        out["all"] = {"n": len(allv), "px": round(float(np.median(allv)), 2)}
    return out


def arm_vertex_ids(mdir: Path, side: str, stride: int = 2) -> np.ndarray:
    """Upper-arm + forearm vertices (dominant joint shoulder / elbow of `side`), every stride-th."""
    from .hand_kin import model_npz
    dom = np.asarray(np.load(model_npz(mdir), allow_pickle=True)["weights"], float).argmax(1)
    return np.flatnonzero(np.isin(dom, rh.UPPER_ARM_JOINTS[side]))[::max(1, stride)]


def hold_pairs(cs: dict) -> list[dict]:
    """Merged hand-hand holds (refine_hands.merge_hold_records): episode, both hands, attract mask, bridged gaps."""
    out = []
    for n, e in enumerate(cs.get("episodes", [])):
        if e.get("hand_hand") and e.get("partner_side") and e.get("union") and e["kind"] in ("hold", "touch"):
            out.append({"episode": n, "a": (e["role"], e["side"]), "b": (OTHER[e["role"]], e["partner_side"]),
                        "ep": e})
    return out


def hand_to_hand(fits: dict, a: tuple, b: tuple) -> np.ndarray:
    """(T,) closest distance (m) between the surfaces of hand a and hand b (vertex sets)."""
    from scipy.spatial import cKDTree
    ia = hand_vertex_ids(fits[a[0]].mdir, a[1])
    ib = hand_vertex_ids(fits[b[0]].mdir, b[1])
    Va, Vb = fits[a[0]].verts(), fits[b[0]].verts()
    return np.array([cKDTree(Vb[k][ib]).query(Va[k][ia])[0].min() for k in range(len(Va))])


def hand_metrics(take, stages: dict | None = None) -> tuple[dict, dict]:
    """Numbers for MAMMA vs fingers vs final -> hands/metrics.json + metrics.npz (per-stage series, keys
    '<stage>|<role>|<side>|gap' / '|pen' / '|arm_gap' / '|arm_pen' / '|cross' (finger-finger intersecting face pairs)).
    Gaps: exact distance + winding-number inside test (mesh_contact; GPU on a pod)."""
    from . import mesh_contact as mc
    work = take.work
    out = work / "hands"
    stages = stages or {k: v for k, v in stage_dirs(work).items() if (v / "lead.npz").exists()}
    cams = rh.fit_cams(work)
    tracks = md.load_tracks(work, cams)
    t0 = time.time()
    fits = {st: {r: rh.Dancer(work, r, d) for r in ROLES} for st, d in stages.items()}
    grid = fits["mamma"]["lead"].t
    gc = rh.grid_cameras(work, tracks, grid)
    judges = {}
    have = (out / "grid_obs.npz").exists()
    wil = rh.load_grid_obs(work, "wilor") if have else {}
    for jd in ("fused", "hamer", "wilor"):
        o = rh.load_grid_obs(work, jd) if have else {}
        if not o:
            continue
        if jd == "fused":
            judges[jd] = o
            continue
        # MANO judges: only views where WiLoR's own detector found the hand (HaMeR ran on those same boxes; elsewhere
        # both return a prior-shaped guess for a hidden hand). Detector hits have a grid weight > 0.35.
        jj = {}
        for rs, v in o.items():
            jj[rs] = {}
            for c, (kp, w) in v.items():
                ww = wil.get(rs, {}).get(c, (None, np.zeros_like(w)))[1]
                jj[rs][c] = (kp, np.where((w > 0) & (ww > 0.35), 1.0, 0.0))
        judges[jd] = jj
    ma2d, _ = rh.ma2d_obs(work)
    mdir = fits["mamma"]["lead"].mdir
    lm_ids = _hand_lm_ids(mdir) if ma2d else None
    hand_ids = {s: hand_vertex_ids(mdir, s) for s in SIDES}
    arm_ids = {s: arm_vertex_ids(mdir, s) for s in SIDES}
    device = mc.default_device()
    arrays = {"t": grid}
    geo = {}
    for st, ds in fits.items():
        g = rh.hand_partner_geometry(ds, hand_ids, arm_ids=arm_ids, device=device)
        geo[st] = g
        for (r, s), v in g.items():
            for k in ("gap", "pen", "arm_gap", "arm_pen"):
                arrays[f"{st}|{r}|{s}|{k}"] = v[k]
    print(f"[metrics] geometry {round(time.time() - t0, 1)} s ({device})", flush=True)
    kin = fits["mamma"]["lead"].kin
    hfs = {s: mc.HandFaces(kin.weights_np, kin.faces, s) for s in SIDES}
    cs = json.loads((out / "contact.json").read_text()) if (out / "contact.json").exists() else {"episodes": []}
    M = {"take": take.name, "stages": {k: str(v) for k, v in stages.items()}, "cams": cams, "device": device,
         "gap_method": "exact point-to-triangle distance; inside = generalized winding number of the partner's closed "
                       "body > 0.5 (mesh_contact)",
         "judges": {"fused": "RTMW / RTMPose / MediaPipe keypoints (in-sample for the finger IK)",
                    "hamer": "HaMeR's 2D joints where WiLoR's detector found the hand (not used by the finger IK; "
                             "its wrist is half of a gated wrist)",
                    "wilor": "WiLoR's 2D joints where its own detector found the hand (its fused fingers are the IK's "
                             "prior: partly in-sample)",
                    "ma2d": "MAMMA's 63 dense hand landmarks per hand (what the c8 fit was made from)"},
         "keypoints": {}, "ma2d": {}, "contact": [], "holds": [], "penetration": {}, "arm_penetration": {},
         "finger_crossings": {}, "palm_vs_mamma_cm": {}, "jitter": {}}
    for r in ROLES:
        for s in SIDES:
            Jref = fits["mamma"][r].hands[s]
            palm_m = Jref[:, list(rh.PALM21)].mean(1)
            for st, ds in fits.items():
                for jd, o in judges.items():
                    M["keypoints"].setdefault(f"{r}_{s}", {}).setdefault(jd, {})[st] = \
                        kp_reprojection(ds[r], s, o.get((r, s), {}), gc, cams, Jref)
                if ma2d:
                    M["ma2d"].setdefault(f"{r}_{s}", {})[st] = ma2d_hand_reprojection(ds[r], s, lm_ids[s], ma2d, gc,
                                                                                       cams)
                p = geo[st][(r, s)]["pen"]
                M["penetration"].setdefault(f"{r}_{s}", {})[st] = {
                    "cm_max": round(float(p.max()) * 100, 2), "cm_p95": round(float(np.percentile(p, 95)) * 100, 2),
                    "frames_gt_1cm": round(float((p > 0.01).mean()), 3)}
                pa = geo[st][(r, s)]["arm_pen"]
                M["arm_penetration"].setdefault(f"{r}_{s}", {})[st] = {
                    "cm_max": round(float(pa.max()) * 100, 2), "frames_gt_1cm": round(float((pa > 0.01).mean()), 3),
                    "frames_gt_2cm": round(float((pa > 0.02).mean()), 3),
                    "worst_t": round(float(grid[int(np.argmax(pa))]), 2)}
                ncr, tot = mc.crossing_series(ds[r].verts(), hfs[s])
                arrays[f"{st}|{r}|{s}|cross"] = ncr
                M["finger_crossings"].setdefault(f"{r}_{s}", {})[st] = {
                    "frames": int((ncr > 0).sum()), "max_pairs": int(ncr.max()),
                    "worst_t": round(float(grid[int(np.argmax(ncr))]), 2),
                    "top_pairs": sorted(tot.items(), key=lambda x: -x[1])[:4]}
                dpalm = np.linalg.norm(ds[r].hands[s][:, list(rh.PALM21)].mean(1) - palm_m, axis=1)
                M["palm_vs_mamma_cm"].setdefault(f"{r}_{s}", {})[st] = {
                    "median": round(float(np.median(dpalm)) * 100, 2), "p95": round(float(np.percentile(dpalm, 95)) * 100, 2),
                    "max": round(float(dpalm.max()) * 100, 2)}
                M["jitter"].setdefault(f"{r}_{s}", {})[st] = jitter(ds[r], s)
    print(f"[metrics] per hand {round(time.time() - t0, 1)} s", flush=True)
    T = len(grid)
    for n, e in enumerate(cs["episodes"]):
        if e["kind"] not in ("hold", "touch"):
            continue
        m = rh.attract_mask(e, T)
        rec = {"episode": n, **{k: e.get(k) for k in ("role", "side", "region", "kind", "t0", "t1", "refined",
                                                       "bridged_gaps", "merged_into")}, "frames": int(m.sum())}
        for st in fits:
            rec[st] = gap_stats(geo[st][(e["role"], e["side"])]["gap"][m])
        M["contact"].append(rec)
    for hp in hold_pairs(cs):
        e = hp["ep"]
        m = rh.attract_mask(e, T)
        gaps = np.zeros(T, bool)
        for a, b in e.get("bridged_gaps") or []:
            gaps[a:b + 1] = True
        rec = {"episode": hp["episode"], "hands": [list(hp["a"]), list(hp["b"])], "bridged_gaps": e.get("bridged_gaps"),
               "attracted_frames": int(m.sum()), "bridged_frames": int(gaps.sum())}
        for st, ds in fits.items():
            dd = hand_to_hand(ds, hp["a"], hp["b"])
            arrays[f"{st}|hold{hp['episode']}|hand_dist"] = dd
            rec[st] = {"attracted_cm": _stats(dd[m], 100.0), "attracted_gt_2cm": round(float((dd[m] > 0.02).mean()), 3),
                       "bridged_cm": _stats(dd[gaps], 100.0) if gaps.any() else None,
                       "bridged_gt_2cm": round(float((dd[gaps] > 0.02).mean()), 3) if gaps.any() else None}
        M["holds"].append(rec)
    M["runtime_s"] = round(time.time() - t0, 1)
    np.savez_compressed(out / "metrics.npz", **arrays)
    (out / "metrics.json").write_text(json.dumps(M, indent=1, default=float))
    return M, arrays


def _metrics_plot(M: dict, path: Path) -> Path:
    """Five rows, MAMMA / fingers / final side by side: keypoint reprojection per camera (fused keypoint judge),
    MAMMA's own hand landmarks per camera, contact gap distribution per episode (winding-number sign; sunk / floating
    shares written under each box), jitter per hand, finger crossings + arm-in-partner frames per hand."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    stages = [s for s, _ in STAGES if s in M["stages"]]
    names = dict(STAGES)
    hands = [f"{r}_{s}" for r in ROLES for s in SIDES]
    fig, axes = plt.subplots(5, 4, figsize=(22, 21))
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    bw = 0.8 / max(len(stages), 1)
    for j, h in enumerate(hands):
        r, s = h.split("_")
        for row, (src, key, ylab) in enumerate((("keypoints", "fused", "median keypoint error / hand length"),
                                               ("ma2d", None, "median px, MAMMA hand landmarks"))):
            ax = axes[row, j]
            d = M[src].get(h, {})
            d = d.get(key, {}) if key else d
            cams = sorted({c for st in stages for c in d.get(st, {}) if c != "all"})
            x = np.arange(len(cams) + 1)
            for i, st in enumerate(stages):
                v = d.get(st, {})
                y = [v.get(c, {}).get("len" if key else "px", np.nan) for c in cams] + \
                    [v.get("all", {}).get("len" if key else "px", np.nan)]
                ax.bar(x - 0.4 + bw * (i + 0.5), y, bw * 0.9, color=STAGE_COL[st], label=names[st])
            ax.set_xticks(x)
            ax.set_xticklabels([f"cam {c}" for c in cams] + ["all"], fontsize=7)
            ax.set_title(f"{LABEL[r]} {s} hand", fontsize=10)
            ax.grid(axis="y", color="#e6e6e3", lw=0.6)
            ax.legend(fontsize=7, frameon=False)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
        ax = axes[3, j]
        jd = M["jitter"].get(h, {})
        ks = [("fingers_rel_wrist_mm", "fingers\n(rel. wrist) mm"), ("wrist_mm", "wrist mm"), ("elbow_mm", "elbow mm"),
              ("wrist_acc_mm_p95", "wrist accel.\np95 mm/frame2"), ("elbow_ang_acc_deg_p95", "elbow ang. acc.\np95 deg/frame2")]
        x = np.arange(len(ks))
        for i, st in enumerate(stages):
            ax.bar(x - 0.4 + bw * (i + 0.5), [jd.get(st, {}).get(k) or np.nan for k, _ in ks], bw * 0.9,
                   color=STAGE_COL[st], label=names[st])
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in ks], fontsize=7)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.legend(fontsize=7, frameon=False)
        if j == 0:
            ax.set_ylabel("jitter (RMS off a local cubic) / accelerations", fontsize=9)
    # contact gaps: one panel spanning the third row
    for ax in axes[2]:
        ax.remove()
    ax = fig.add_subplot(5, 1, 3)
    eps = M.get("contact", [])
    pos, ticks = 0, []
    for e in eps:
        for i, st in enumerate(stages):
            g = e.get(st, {})
            if not g.get("n"):
                continue
            xx = pos + i * 0.8
            ax.plot([xx, xx], [g["p5"], g["p95"]], color=STAGE_COL[st], lw=2, solid_capstyle="round")
            ax.add_patch(plt.Rectangle((xx - 0.3, g["p25"]), 0.6, g["p75"] - g["p25"], color=STAGE_COL[st], alpha=0.85,
                                       lw=0))
            ax.plot([xx - 0.3, xx + 0.3], [g["p50"], g["p50"]], color="#0b0b0b", lw=1.5)
            ax.text(xx, -7.6, f"{g.get('sunk_gt_1cm', 0):.0%}\n{g.get('float_gt_1cm', 0):.0%}", ha="center", va="top",
                    fontsize=6.5, color="#333333")
        merged = " (merged hold, bridged)" if e.get("bridged_gaps") else (" (record merged)" if e.get("merged_into")
                                                                          is not None else "")
        lab = (f"{LABEL[e['role']].split(' ')[0]} {e['side']} -> {e['region']}\n{e['kind']} {e['t0']:.1f}-{e['t1']:.1f}s"
               f"{merged}\n{str(e.get('refined', ''))[:44]}")
        ticks.append((pos + 0.8 * (len(stages) - 1) / 2, lab))
        pos += 0.8 * len(stages) + 1.2
    ax.axhspan(-1, 1, color="#1baf7a", alpha=0.08, lw=0)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_ylim(-9.5, 8)
    ax.set_xticks([p for p, _ in ticks])
    ax.set_xticklabels([t for _, t in ticks], fontsize=7)
    ax.set_ylabel("hand-to-partner gap (cm), negative = sunk in\nbox p25-p75, line p5-p95, bar median", fontsize=9)
    ax.set_title("contact episodes: gap inside each episode (shaded = within 1 cm of touching); numbers under each box: "
                 "share sunk > 1 cm / share floating > 1 cm (exact distance, winding-number inside test)", fontsize=10)
    ax.legend(handles=[Patch(color=STAGE_COL[s], label=names[s]) for s in stages], fontsize=8, frameon=False,
              loc="upper right")
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    # finger crossings / arm in partner
    for ax in axes[4]:
        ax.remove()
    for col, (src, key, ylab, title, scale) in enumerate((
            ("finger_crossings", "frames", "frames with crossing fingers",
             "finger-finger self-intersections (exact triangle tests): frames per hand", 1.0),
            ("arm_penetration", "frames_gt_1cm", "share of frames",
             "upper arm / forearm inside the partner > 1 cm (winding number)", 1.0))):
        ax = fig.add_subplot(5, 2, 9 + col)
        x = np.arange(len(hands))
        for i, st in enumerate(stages):
            y = [M.get(src, {}).get(h, {}).get(st, {}).get(key, np.nan) for h in hands]
            ax.bar(x - 0.4 + bw * (i + 0.5), y, bw * 0.9, color=STAGE_COL[st], label=names[st])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{LABEL[h.split('_')[0]].split(' ')[0]} {h.split('_')[1]}" for h in hands], fontsize=8)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{M['take']}: hands before / after - keypoint reprojection per camera (top), MAMMA's own hand "
                 f"landmarks, contact gaps, jitter, finger crossings + arm in partner (bottom); lower is better except "
                 f"the gap (0 = touching)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=85)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------ build

def run_build(take, workers: int = 1, loo: bool = True, review: bool = True, gate_cfg: GateConfig | None = None,
              rcfg: rh.RefineConfig | None = None, finger_overrides: dict | None = None) -> dict:
    """Steps 1-7 of the module docstring; finger_overrides replace FingerConfig fields (tests / smoke runs)."""
    from dataclasses import replace
    work = take.work
    out = work / "hands"
    for d in ("hands", "review"):
        (work / d).mkdir(parents=True, exist_ok=True)
    if not (out / "grid_obs.npz").exists():
        raise SystemExit(f"build needs hands/grid_obs.npz (hand keypoint evidence): run `python -m dancecap.hands_pod "
                         f"run {take.name}` first")
    t0 = time.time()
    summary = {"take": take.name, "steps": {}}
    # 1. episodes on the adopted fit (the wrist gate excludes hand-hand holds)
    cams = rh.fit_cams(work)
    tracks = md.load_tracks(work, cams)
    dancers = {r: rh.Dancer(work, r) for r in ROLES}
    grid = dancers["lead"].t
    gc = rh.grid_cameras(work, tracks, grid)
    hand_ids = {s: hand_vertex_ids(dancers["lead"].mdir, s) for s in SIDES}
    ccfg = rh.ContactConfig()
    eps0, _ = rh.contact_episodes(rh.hand_partner_geometry(dancers, hand_ids),
                                  rh.view_evidence(work, dancers, tracks, cams, ccfg), grid, ccfg)
    summary["steps"]["episodes0"] = {"n": len(eps0), "seconds": round(time.time() - t0, 1)}
    # 2. wrist gate
    t1 = time.time()
    gate = wrist_gate(work, dancers, gc, eps0, gate_cfg)
    summary["steps"]["wrist_gate"] = {"hands": gate["summary"]["hands"], "seconds": round(time.time() - t1, 1)}
    # 3. fingers (hybrid when the fused WiLoR fingers exist)
    t1 = time.time()
    mano = "wilor" if (out / "mano_fused_wilor.npz").exists() else ""
    fcfg = replace(rh.FingerConfig(mano_model=mano, tag="build", wrist_init=str(out / "wrist_gate.npz")),
                   **(finger_overrides or {}))
    fs = rh.run_fingers(take, fcfg, loo=loo, workers=workers, sheet=review)
    summary["steps"]["fingers"] = {"route": f"hybrid keypoints + fused {mano}" if mano else "keypoint IK",
                                   "hands": fs["hands"], "seconds": round(time.time() - t1, 1)}
    # 4. contact from the new fingers
    t1 = time.time()
    cs = rh.run_contact(take, rcfg=rcfg, smplx_dir=out / "smplx_fingers_build", sheets=False)
    summary["steps"]["contact"] = {"refined": [{k: v for k, v in x.items() if not k.startswith("terms")}
                                               for x in cs.get("refined", [])], "seconds": round(time.time() - t1, 1)}
    # 5-6. metrics, outputs
    t1 = time.time()
    M, arrays = hand_metrics(take)
    summary["steps"]["metrics"] = {"seconds": round(time.time() - t1, 1)}
    summary["outputs"] = write_outputs(take, gate, arrays)
    # 7. review images: the mesh-only ones here (pod), the video crops where the frames are (review=True, local)
    t1 = time.time()
    summary["review"] = [str(p) for p in run_review(take, metrics=(M, arrays), mesh=True, video=review)]
    summary["steps"]["review"] = {"seconds": round(time.time() - t1, 1)}
    summary["runtime_s"] = round(time.time() - t0, 1)
    (out / "build.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


# ------------------------------------------------------------------------------------------------ review images

def _frames_available(work: Path, cams: list[str]) -> bool:
    return all((work / "frames" / c / "000000.jpg").exists() for c in cams)


def contact_moments(cs: dict, geo_before: dict, T: int, max_moments: int = 8, palm_move: dict | None = None,
                    min_move_m: float = 0.03, at: list | None = None, grid: np.ndarray | None = None) -> list[dict]:
    """Moments to show: per attempted hold / touch episode (longest first) the attracted frame where the hand was sunk
    deepest before (the mid frame when it never was) + for a merged hand-hand hold the middle of each bridged gap
    (else of its longest run) + the frame where the contact step moved the hand most (palm_move {(role, side): (T,)
    m}, when > min_move_m)."""
    eps = [(n, e) for n, e in enumerate(cs["episodes"]) if e["kind"] in ("hold", "touch")
           and (str(e.get("refined", "")).startswith("yes") or str(e.get("refined", "")).startswith("no"))]
    eps.sort(key=lambda x: -(x[1]["i1"] - x[1]["i0"]))
    out = []
    for n, e in eps:
        m = rh.attract_mask(e, T)
        idx = np.flatnonzero(m)
        if not len(idx):
            continue
        pen = geo_before.get((e["role"], e["side"]))
        deep = pen is not None and pen[idx].max() > 0.003
        k = int(idx[np.argmax(pen[idx])]) if deep else int(idx[len(idx) // 2])
        out.append({"episode": n, "k": k, "why": "deepest before" if deep else "mid", **e})
        if e.get("hand_hand"):
            gaps = e.get("bridged_gaps") or []
            if gaps:
                for a, b in gaps:
                    out.append({"episode": n, "k": (a + b) // 2, "why": "bridged gap", **e})
            else:
                a, b = max(runs_of(m), key=lambda x: x[1] - x[0])
                km = (a + b) // 2
                if abs(km - k) > 10:
                    out.append({"episode": n, "k": km, "why": "mid hold", **e})
        mv = (palm_move or {}).get((e["role"], e["side"]))
        if mv is not None and mv[idx].max() > min_move_m:
            km = int(idx[np.argmax(mv[idx])])
            if all(abs(km - o["k"]) > 5 for o in out if o["episode"] == n):
                out.append({"episode": n, "k": km, "why": f"largest move ({mv[km] * 100:.1f} cm)", **e})
    out = out[:max_moments]
    for ta in at or []:            # requested times (e.g. the review's evidence frames): every episode active then
        if grid is None:
            break
        k = int(np.argmin(np.abs(np.asarray(grid) - float(ta))))
        for n, e in eps:
            if e["i0"] <= k <= e["i1"] and all(abs(k - o["k"]) > 2 or o["episode"] != n for o in out):
                out.append({"episode": n, "k": k, "why": f"requested {float(ta):.2f}s", **e})
    return out


def _partner_side(e: dict) -> str | None:
    if not e.get("hand_hand"):
        return None
    return "left" if str(e["region"]).startswith("left") else "right"


def _best_cams(work, tracks, cams, fits_final, e, k, n: int = 3) -> list[tuple]:
    """Cameras where the contacting hand (and the partner's hand in a hand-hand hold) is large and not hidden."""
    r, s = e["role"], e["side"]
    ps = _partner_side(e)
    tk = fits_final[r].t[k]
    hid = {sd: hand_vertex_ids(fits_final[r].mdir, sd) for sd in SIDES}
    cand = []
    for c in cams:
        tr = tracks[c]
        i = int(md.nearest_index(tr.t, np.array([tk]))[0])
        if not tr.ok[i] or abs(tr.t[i] - tk) > 1.0 / 30:
            continue
        K, R, tt, k1 = tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i])
        pts = [fits_final[r].hands[s][k]] + ([fits_final[OTHER[r]].hands[ps][k]] if ps else [])
        P = np.concatenate(pts)
        px, z = project(P, K, R, tt, k1)
        if not (((px[:, 0] >= 0) & (px[:, 0] < tr.width) & (px[:, 1] >= 0) & (px[:, 1] < tr.height)).all()
                and (z > 0.1).all()):
            continue
        L = float(np.linalg.norm(px[0] - px[12]))
        occ = []
        for rr in ROLES:
            V = fits_final[rr].verts()[k]
            keep = np.ones(len(V), bool)
            if rr == r:
                keep[hid[s]] = False
            if ps and rr == OTHER[r]:
                keep[hid[ps]] = False
            if rr == OTHER[r]:   # the partner's surface at the contact (a hand resting on / sunk into it) hides nothing
                from scipy.spatial import cKDTree
                keep &= cKDTree(P).query(V)[0] > 0.06
            occ.append(V[keep])
        vis = float(rh._zbuffer_vis(np.concatenate(occ), P, K, R, tt, k1, tr.width, tr.height,
                                    float(np.clip(L / 6, 1.5, 6))).mean())
        cand.append((L * (0.25 + vis), L, vis, c, i))
    cand.sort(key=lambda x: -x[0])
    good = [x for x in cand if x[2] >= 0.4 and x[1] >= 10]
    return (good + [x for x in cand if x not in good])[:n]


def _native_det(nat: dict, r: str, s: str, c: str, i: int):
    """The best keypoint detection (RTMW / RTMPose / MediaPipe, >= 15 confident points) of a hand in a native
    frame -> (kp (21,2) distorted px, w (21,), detector) or None."""
    best = None
    for det in rh.DETECTORS:
        d = nat.get((det, r, s, c))
        if d is None:
            continue
        hit = np.flatnonzero(d["i"] == i)
        if not len(hit):
            continue
        w = d["w"][hit[0]]
        if (w > 0).sum() >= 15 and (best is None or w.mean() > best[1].mean()):
            best = (d["kp"][hit[0]], w, det)
    return best


def _kp_err(px: np.ndarray, det) -> float:
    if det is None:
        return float("nan")
    ok = det[1] > 0
    return float(np.median(np.linalg.norm(px[ok] - det[0][ok], axis=1)))


def _crop_K(K: np.ndarray, box, sc: float) -> np.ndarray:
    """Intrinsics of a crop (box in native px, scaled by sc) of a camera with intrinsics K."""
    Kc = K.copy()
    Kc[0, 0] *= sc
    Kc[1, 1] *= sc
    Kc[0, 2] = (K[0, 2] - box[0]) * sc
    Kc[1, 2] = (K[1, 2] - box[1]) * sc
    return Kc


def _overlay_tile(crop, fits, t_i, faces, K, R, tvec, k1, box, sc, hands, dets, title, tile):
    """Mesh overlay (both dancers, pinhole) + hand skeletons (with the camera's k1) + detections (yellow) on a crop
    of the native frame; returns the tile and each hand's median keypoint error (native px)."""
    import cv2
    col = {"lead": (0.95, 0.45, 0.35), "follow": (0.35, 0.6, 0.95)}
    meshes = [(fits[rr].at(fits[rr].verts(), t_i), faces, col[rr]) for rr in ROLES]
    ren, cov = rh.render_meshes(meshes, _crop_K(K, box, sc), R, tvec, tile, tile)
    ov = crop.astype(float) / 255
    ov[cov] = 0.6 * ov[cov] + 0.4 * ren[cov][:, ::-1]
    ov = (ov * 255).astype(np.uint8)
    errs = []
    for n, ((rr, ss), det) in enumerate(zip(hands, dets)):
        px, _ = project(fits[rr].at(fits[rr].hands[ss], t_i), K, R, tvec, k1)
        if det is not None:
            ok = det[1] > 0
            for q in np.round((det[0][ok] - [box[0], box[1]]) * sc).astype(int):
                cv2.circle(ov, tuple(int(v) for v in q), 2, (0, 255, 255), -1, cv2.LINE_AA)
        rh.draw_hand(ov, (px - [box[0], box[1]]) * sc, (255, 255, 255) if n == 0 else (255, 80, 255), 1, 1)
        errs.append(_kp_err(px, det))
    e_txt = " / ".join(f"{x:.1f}" if np.isfinite(x) else "-" for x in errs)
    label(ov, title, (3, 13), scale=0.4)
    label(ov, f"kp err px: {e_txt}", (3, tile - 6), scale=0.38)
    return ov, errs


def review_contact(work, take_name, cs, tracks, cams, fits, geo_before, path: Path, tile: int = 200,
                   palm_move: dict | None = None, at: list | None = None) -> Path | None:
    """hands_contact.jpg: per contact moment and best camera: video crop | before (MAMMA meshes, hand skeletons,
    detections in yellow, keypoint error) | after (final)."""
    import cv2
    T = len(fits["mamma"]["lead"].t)
    moments = contact_moments(cs, geo_before, T, max_moments=8, palm_move=palm_move, at=at,
                              grid=fits["mamma"]["lead"].t)
    if not moments:
        return None
    nat = rh.native_detections(work)
    faces = fits["mamma"]["lead"].kin.faces
    rows = []
    for mo in moments:
        r, s, k = mo["role"], mo["side"], mo["k"]
        ps = _partner_side(mo)
        hands = [(r, s)] + ([(OTHER[r], ps)] if ps else [])
        tiles, errs_all = [], []
        for _, L, vis, c, i in _best_cams(work, tracks, cams, fits["final"], mo, k):
            tr = tracks[c]
            img = cv2.imread(str(work / "frames" / c / f"{i:06d}.jpg"))
            if img is None:
                continue
            K, R, tt, k1 = tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i])
            t_i = float(tr.t[i])
            pts = np.concatenate([project(fits[st][rr].at(fits[st][rr].hands[ss], t_i), K, R, tt, k1)[0]
                                  for st in ("mamma", "final") for rr, ss in hands])
            if not ps:   # include the partner's surface point nearest the hand (the contact point)
                Vo = fits["final"][OTHER[r]].at(fits["final"][OTHER[r]].verts(), t_i)
                hc = fits["final"][r].at(fits["final"][r].hands[s], t_i).mean(0)
                q = Vo[np.argmin(np.linalg.norm(Vo - hc, axis=1))]
                pts = np.concatenate([pts, project(q[None], K, R, tt, k1)[0]])
            lo, hi = pts.min(0), pts.max(0)
            side = max(1.35 * float((hi - lo).max()), 3.0 * L, 64.0)
            cx, cy = (lo + hi) / 2
            box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
            sc = tile / (box[2] - box[0])
            crop = crop_pad(img, box, tile, cv2.INTER_CUBIC)
            dets = [_native_det(nat, rr, ss, c, i) for rr, ss in hands]
            b_tile, eb = _overlay_tile(crop, fits["mamma"], t_i, faces, K, R, tt, k1, box, sc, hands, dets,
                                       "before (MAMMA)", tile)
            a_tile, ea = _overlay_tile(crop, fits["final"], t_i, faces, K, R, tt, k1, box, sc, hands, dets,
                                       "after (final)", tile)
            v = crop.copy()
            label(v, f"cam {c} t={t_i:.2f}s", (3, 13), scale=0.4)
            label(v, f"hand {L:.0f}px vis {vis:.0%}", (3, tile - 6), scale=0.38)
            tiles += [v, b_tile, a_tile]
            errs_all.append((eb, ea))
        if not tiles:
            continue
        row = np.concatenate(tiles, 1)
        hdr = np.full((22, row.shape[1], 3), 25, np.uint8)
        who = f"{LABEL[r]} {s} hand -> {LABEL[OTHER[r]]} {mo['region']}"
        label(hdr, f"{who} ({mo['kind']}, {mo['t0']:.2f}-{mo['t1']:.2f}s), frame t={fits['mamma'][r].t[k]:.2f}s "
                   f"({mo['why']}; {str(mo.get('refined', ''))[:48]}); white skeleton = {r} {s}"
              + (f", magenta = {OTHER[r]} {ps}" if ps else "")
              + "; yellow = keypoint detections", (5, 15), scale=0.42)
        rows.append(np.concatenate([hdr, row], 0))
    if not rows:
        return None
    w = max(x.shape[1] for x in rows)
    sheet = np.concatenate([np.pad(x, ((0, 3), (0, w - x.shape[1]), (0, 0))) for x in rows], 0)
    head = np.full((46, w, 3), 10, np.uint8)
    label(head, f"{take_name}: hand contact before / after (lead red, follow blue mesh): per camera video | MAMMA c8 | "
                f"final (fingers + wrist gate + contact)", (6, 18), scale=0.48)
    label(head, "kp err = median px from the white / magenta hand skeleton to the yellow keypoint detections (RTMW / "
                "RTMPose / MediaPipe) in that frame; '-' = no detection", (6, 38), scale=0.45)
    cv2.imwrite(str(path), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def _signed_to(Vh: np.ndarray, Vo: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Signed distance (m) of points Vh to the closed body Vo (exact distance; negative inside by winding number)."""
    from . import mesh_contact as mc
    return mc.signed_distance(Vh, Vo, mc.Topology(faces, len(Vo)))["sd"]


def _render_faces(V, F, face_rgb, K, R, t, H, W, xray: np.ndarray | None = None):
    """Lambert (headlight) render of faces with per-face colours; xray (F,) bool faces (the contacting hands) are
    also drawn see-through (45 %) where another surface hides them, so a hand sunk into the partner stays visible."""
    import torch
    from .avatar_raster import rasterize
    Xc = V @ R.T + t
    z = Xc[:, 2]
    px = np.stack([K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-4) + K[0, 2],
                   K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-4) + K[1, 2]], 1)
    _, fid = rasterize(torch, torch.tensor(px), torch.tensor(z), torch.tensor(F), H, W, near=0.02)
    fid = fid.numpy()
    fx = None
    if xray is not None and xray.any():
        sub = np.flatnonzero(xray)
        _, fx = rasterize(torch, torch.tensor(px), torch.tensor(z), torch.tensor(F[sub]), H, W, near=0.02)
        fx = fx.numpy()
        fx = np.where(fx >= 0, sub[np.maximum(fx, 0)], -1)
    tri = Xc[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    view = -tri.mean(1)
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    shade = 0.3 + 0.7 * np.abs((n * view).sum(1))
    img = np.full((H, W, 3), 0.13)
    cov = fid >= 0
    img[cov] = face_rgb[fid[cov]] * shade[fid[cov], None]
    if fx is not None:
        hidden = (fx >= 0) & (fx != fid)
        img[hidden] = 0.55 * img[hidden] + 0.45 * face_rgb[fx[hidden]] * shade[fx[hidden], None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()


def review_render(take_name, cs, fits, geo_before, path: Path, tile: int = 240, radius: float = 0.2,
                  palm_move: dict | None = None) -> Path | None:
    """hands_render.jpg: per contact moment, two close-up viewpoints, before | after: only the surface within `radius`
    of the contact is drawn (no occluding bodies); hand vertices sunk > 3 mm into the partner are yellow, upper-arm /
    forearm vertices sunk > 1 cm orange (exact distance, winding-number inside test)."""
    import cv2
    from .avatar_review import _look_at
    T = len(fits["mamma"]["lead"].t)
    moments = contact_moments(cs, geo_before, T, max_moments=8, palm_move=palm_move)
    if not moments:
        return None
    faces = fits["mamma"]["lead"].kin.faces
    hid = {s: hand_vertex_ids(fits["mamma"]["lead"].mdir, s) for s in SIDES}
    base_col = {"lead": np.array([0.95, 0.45, 0.35]), "follow": np.array([0.35, 0.6, 0.95])}
    up = np.array([0.0, 1.0, 0.0])
    Kv = np.array([[tile * 1.5, 0, tile / 2], [0, tile * 1.5, tile / 2], [0, 0, 1.0]])
    rows = []
    for mo in moments:
        r, s, k = mo["role"], mo["side"], mo["k"]
        ps = _partner_side(mo)
        o = OTHER[r]
        c0 = fits["final"][r].hands[s][k].mean(0)
        if ps:
            c0 = 0.5 * (c0 + fits["final"][o].hands[ps][k].mean(0))
            dvec = fits["final"][o].joints[k, 9] - fits["final"][r].joints[k, 9]
            d1 = np.cross(up, dvec)
        else:
            d1 = c0 - fits["final"][o].joints[k, 9]
        d1[1] = 0
        d1 /= max(np.linalg.norm(d1), 1e-9)
        a = np.deg2rad(55)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        views = [c0 + 0.42 * d1 + 0.12 * up, c0 + 0.38 * (Ry @ d1) + 0.25 * up]
        tiles = []
        nums = {}
        for st in ("mamma", "final"):
            V = {rr: fits[st][rr].verts()[k].astype(float) for rr in ROLES}
            cols = {rr: np.tile(base_col[rr], (len(faces), 1)) for rr in ROLES}
            # sunk hand vertices -> yellow faces
            pen = []
            for rr, ss in [(r, s)] + ([(o, ps)] if ps else []):
                aid = arm_vertex_ids(fits[st][rr].mdir, ss, 1)
                sd = _signed_to(np.concatenate([V[rr][hid[ss]], V[rr][aid]]), V[OTHER[rr]], faces)
                sh, sa = sd[:len(hid[ss])], sd[len(hid[ss]):]
                bad = np.zeros(len(V[rr]), bool)
                bad[aid[sa < -0.01]] = True
                cols[rr][bad[faces].any(1)] = [1.0, 0.55, 0.1]
                bad = np.zeros(len(V[rr]), bool)
                bad[hid[ss][sh < -0.003]] = True
                fb = bad[faces].any(1)
                cols[rr][fb] = [1.0, 0.85, 0.1]
                pen.append(max(0.0, -float(sh.min())))
            nums[st] = max(pen)
            Vs, Fs, Cs, Xs, off = [], [], [], [], 0
            rad = 0.6 * radius if ps else radius
            for rr in ROLES:
                keep = np.linalg.norm(V[rr] - c0, axis=1) < rad
                fk = keep[faces].all(1)
                remap = -np.ones(len(V[rr]), int)
                remap[keep] = np.arange(keep.sum())
                Vs.append(V[rr][keep])
                Fs.append(remap[faces[fk]] + off)
                Cs.append(cols[rr][fk])
                inhand = np.zeros(len(V[rr]), bool)
                for ss in ([s] if rr == r else []) + ([ps] if ps and rr == o else []):
                    inhand[hid[ss]] = True
                Xs.append(inhand[faces[fk]].all(1))
                off += int(keep.sum())
            Vall, Fall, Call, Xall = np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Cs), np.concatenate(Xs)
            for vi, C in enumerate(views):
                Rv, tv = _look_at(C.copy(), c0.copy())
                im = _render_faces(Vall, Fall, Call, Kv, Rv, tv, tile, tile, xray=Xall)
                label(im, f"{'before (MAMMA)' if st == 'mamma' else 'after (final)'} view {vi + 1}", (3, 14), scale=0.42)
                label(im, f"deepest sink {nums[st] * 100:.1f} cm", (3, tile - 7), scale=0.4)
                tiles.append((vi, st, im))
        tiles.sort(key=lambda x: (x[0], x[1] != "mamma"))
        row = np.concatenate([x[2] for x in tiles], 1)
        hdr = np.full((22, row.shape[1], 3), 25, np.uint8)
        label(hdr, f"{LABEL[r]} {s} hand -> {LABEL[o]} {mo['region']} ({mo['kind']}), t={fits['mamma'][r].t[k]:.2f}s "
                   f"({mo['why']})", (5, 15), scale=0.42)
        rows.append(np.concatenate([hdr, row], 0))
    w = max(x.shape[1] for x in rows)
    sheet = np.concatenate([np.pad(x, ((0, 3), (0, w - x.shape[1]), (0, 0))) for x in rows], 0)
    head = np.full((46, w, 3), 10, np.uint8)
    label(head, f"{take_name}: hand contacts rendered (lead red, follow blue; only the surface near the contact): "
                f"view 1 before | after, view 2 before | after", (6, 18), scale=0.45)
    label(head, "yellow = hand sunk > 3 mm into the partner, orange = arm sunk > 1 cm (winding-number test); see-through "
                "ghost = the hidden part of a sunk hand", (6, 38), scale=0.45)
    cv2.imwrite(str(path), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def review_crossings(take_name, fits: dict, M: dict, path: Path, tile: int = 220) -> Path | None:
    """hands_finger_crossing.jpg: per hand, the frame with the most finger-finger crossings in any stage: close-ups of
    the hand (fingers coloured, intersecting faces red) from the palm and the back, MAMMA | fingers | final."""
    import cv2
    from . import mesh_contact as mc
    from .avatar_review import _look_at
    kin = fits["mamma"]["lead"].kin
    F = kin.faces
    dom = kin.weights_np.argmax(1)
    segc = {0: (0.8, 0.8, 0.8), 1: (0.95, 0.45, 0.35), 2: (0.35, 0.75, 0.35), 3: (0.35, 0.55, 0.95),
            4: (0.9, 0.8, 0.2), 5: (0.75, 0.4, 0.85)}
    Kv = np.array([[tile * 1.6, 0, tile / 2], [0, tile * 1.6, tile / 2], [0, 0, 1.0]])
    rows = []
    stages = [st for st in ("mamma", "fingers", "final") if st in fits]
    for r in ROLES:
        for s in SIDES:
            fc = M.get("finger_crossings", {}).get(f"{r}_{s}", {})
            if not fc:
                continue
            st_w = max(stages, key=lambda st: fc.get(st, {}).get("max_pairs", 0))
            tw = fc.get(st_w, {}).get("worst_t")
            if tw is None:
                continue
            k = int(np.argmin(np.abs(fits["mamma"][r].t - tw)))
            hf = mc.HandFaces(kin.weights_np, F, s)
            base, wr = (25, 20) if s == "left" else (40, 21)
            seg = {wr: 0, **{base + i: 1 + i // 3 for i in range(15)}}
            fcol = np.array([segc[seg[int(np.bincount(dom[f]).argmax())]] for f in hf.F])
            tiles = []
            for st in stages:
                Vk = fits[st][r].verts()[k]
                res = mc.self_intersections(Vk, hf, return_faces=True)
                col = fcol.copy()
                if len(res.get("faces", [])):
                    col[res["faces"]] = (1.0, 0.1, 0.1)
                vids = np.unique(hf.F)
                c0 = Vk[vids].mean(0)
                _, _, vt = np.linalg.svd(Vk[vids] - c0, full_matrices=False)
                for sgn in (1, -1):
                    C = c0 + sgn * 0.3 * vt[2] + 0.05 * vt[0]
                    Rv, tv = _look_at(C.copy(), c0.copy())
                    im = _render_faces(Vk, hf.F, col, Kv, Rv, tv, tile, tile)
                    label(im, f"{dict(STAGES)[st]}", (3, 14), scale=0.4)
                    label(im, f"{res['n']} crossing face pairs", (3, tile - 7), scale=0.4)
                    tiles.append(im)
            row = np.concatenate(tiles, 1)
            hdr = np.full((22, row.shape[1], 3), 25, np.uint8)
            frames_txt = ", ".join(f"{dict(STAGES)[st].split(' ')[0]} {fc.get(st, {}).get('frames', '-')}" for st in stages)
            label(hdr, f"{LABEL[r]} {s} hand, t={fits['mamma'][r].t[k]:.2f}s (worst frame of {st_w}); frames with "
                       f"crossings of {len(fits['mamma'][r].t)}: {frames_txt}", (5, 15), scale=0.42)
            rows.append(np.concatenate([hdr, row], 0))
    if not rows:
        return None
    w = max(x.shape[1] for x in rows)
    sheet = np.concatenate([np.pad(x, ((0, 3), (0, w - x.shape[1]), (0, 0))) for x in rows], 0)
    head = np.full((46, w, 3), 10, np.uint8)
    label(head, f"{take_name}: finger self-intersections, each stage from two sides (red = faces of one finger "
                f"passing through another finger)", (6, 18), scale=0.45)
    label(head, "fingers: index red, middle green, pinky blue, ring yellow, thumb purple, palm grey", (6, 38), scale=0.45)
    cv2.imwrite(str(path), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def palm_moves(fits: dict, ref: str = "fingers") -> dict:
    """{(role, side): (T,) m} palm-centre distance final vs the contact step's input (fingers stage)."""
    if "final" not in fits or ref not in fits:
        return {}
    return {(r, s): np.linalg.norm(fits["final"][r].hands[s][:, list(rh.PALM21)].mean(1)
                                   - fits[ref][r].hands[s][:, list(rh.PALM21)].mean(1), axis=1)
            for r in ROLES for s in SIDES}


def run_review(take, metrics: tuple | None = None, mesh: bool = True, video: bool = True,
               at: list | None = None) -> list[Path]:
    """The review images from the build outputs. mesh: hands_metrics.png, hands_render.jpg, hands_finger_crossing.jpg
    (no frames needed: the pod draws them); video: hands_contact.jpg (needs frames/: local)."""
    work = take.work
    out = work / "hands"
    review = work / "review"
    review.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        if (out / "metrics.json").exists() and (out / "metrics.npz").exists():
            metrics = (json.loads((out / "metrics.json").read_text()), rh._npz(out / "metrics.npz"))
        else:
            metrics = hand_metrics(take)
    M, arrays = metrics
    paths = [_metrics_plot(M, review / "hands_metrics.png")] if mesh else []
    stages = {k: v for k, v in stage_dirs(work).items() if (v / "lead.npz").exists()}
    fits = {st: {r: rh.Dancer(work, r, d) for r in ROLES} for st, d in stages.items()}
    cs = json.loads((out / "contact.json").read_text())
    geo_before = {(r, s): np.asarray(arrays[f"mamma|{r}|{s}|pen"]) for r in ROLES for s in SIDES
                  if f"mamma|{r}|{s}|pen" in arrays}
    pm = palm_moves(fits)
    cams = rh.fit_cams(work)
    if mesh:
        for p in (review_render(take.name, cs, fits, geo_before, review / "hands_render.jpg", palm_move=pm),
                  review_crossings(take.name, fits, M, review / "hands_finger_crossing.jpg")):
            if p:
                paths.append(p)
    if video:
        if _frames_available(work, cams):
            tracks = md.load_tracks(work, cams)
            p = review_contact(work, take.name, cs, tracks, cams, fits, geo_before, review / "hands_contact.jpg",
                               palm_move=pm, at=at)
            if p:
                paths.append(p)
        else:
            print("[review] frames/ missing: hands_contact.jpg skipped (run `refine_hands review` where the frames are)")
    return paths


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["build", "review", "gate", "outputs", "metrics"])
    ap.add_argument("take")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--video-only", action="store_true", help="review: only hands_contact.jpg (the pod drew the rest)")
    ap.add_argument("--mesh-only", action="store_true", help="review: no video crops")
    ap.add_argument("--at", default="", help="review: extra moments at these reference times (comma-separated s)")
    a = ap.parse_args(argv)
    take = rh.load_take(a.take)
    if a.stage == "build":
        s = run_build(take, workers=a.workers, loo=not a.no_loo, review=not a.no_review)
        print(json.dumps({k: s[k] for k in ("outputs", "review", "runtime_s")}, indent=1, default=str))
    elif a.stage == "review":
        at = [float(x) for x in a.at.split(",") if x.strip()]
        print("review:", [str(p) for p in run_review(take, mesh=not a.video_only, video=not a.mesh_only, at=at)])
    elif a.stage == "metrics":
        M, _ = hand_metrics(take)
        print(json.dumps({k: M[k] for k in ("contact", "penetration")}, indent=1))
    elif a.stage == "outputs":
        print(json.dumps(write_outputs(take), indent=1))
    else:
        work = take.work
        cams = rh.fit_cams(work)
        dancers = {r: rh.Dancer(work, r) for r in ROLES}
        gc = rh.grid_cameras(work, md.load_tracks(work, cams), dancers["lead"].t)
        cs = json.loads((work / "hands" / "contact.json").read_text())
        wrist_gate(work, dancers, gc, cs["episodes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
