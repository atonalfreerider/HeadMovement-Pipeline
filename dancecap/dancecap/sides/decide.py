"""Decide the true sides from the cameras (DESIGN 2.1-2.4): per relation and frame a log-odds from the evidence only (the fit being repaired
never votes), an HMM per relation ("decide on approach, hold while entangled"), the per-dancer yaw track, and the list of places where the
base fit breaks a decided side or the real joint positions (the violations; no motion is changed here).

    L_r(t) = clip(l_3D, +-4) + sum_c w_c,r l_c + clip(l_vis, +-3) + clip(l_crop, +-3) + clip(l_prior, +-cap)

  l_3D    sign(m) clip(1.6 (|m| - dz) / sigma_m, 0, 4) of the triangulated (visibility-consistent, debiased) margin; sigma_m^2 = sum_i
          a_i^T S_i a_i with a_i = dm/dX_i by finite differences on every COCO joint (so derived points - palm, head - are exact)
  l_c     the per-view crossing vote o_c sigma_c (R9 arm pairings), weight logit(q_c) self-calibrated on 3D-decisive frames (q < 0.55 -> 0)
  l_vis   R4 only: the generative visibility test (r4_visibility_test) over front / flank / back
  l_crop  crops.json (written by sides.crops; the contract is in `load_crops`)
  l_prior [sides.priors] (zouk holds, "his right hand usually rests on her upper-left back"), capped at prior_nats (0.5); in camera-blind
          R4 frames at blind_prior_nats (owner decision D1, default 1.5; 0.5 = decision (a))

Outputs (work/<take>/sides/): relations.npz, relations.json, yaw.npz, spins.json, violations.json; the keys are listed in REL_NPZ_KEYS /
YAW_KEYS below and in the notes of the step.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import geom, hmm
from .adapters import PIDX, adapter_for, project
from .config import SidesCfg, load_cfg
from .evidence import (capsule_set, crossing_votes, cue_reliability, exclusion, load_tri, load_views, posed_capsules, rest_joints,
                       site_set, visible, write_json)
from .filmtime import FilmMap, mmss

CLASSES = hmm.R4_CLASSES
REL_NPZ_KEYS = ("<rid>__m_base", "<rid>__m_tri", "<rid>__sigma", "<rid>__L", "<rid>__l3d", "<rid>__lc", "<rid>__post", "<rid>__s",
                "<rid>__gate", "<rid>__band", "R9_*__votes", "R9_*__G_base", "R9_*__G_tri", "R4__fb_base", "R4__fl_base", "R4__fb_tri",
                "R4__fl_tri", "R4__l3d", "R4__lvis", "R4__lprior", "R4__lcrop", "R4__L", "R4__post", "R4__cls", "R4__cls_base",
                "R4__gate", "R4__band", "R4__blind", "R4__inside_base", "R4__surf_base", "R10_<role>_<L|R>__phi_base", "..__w_base",
                "..__phi_tri", "t")
YAW_KEYS = ("<role>__psi_star", "<role>__conf", "<role>__psi_base", "<role>__err", "<role>__face_contra", "<role>__n_bits",
            "<role>__spin_cand", "<role>__defect", "t")
SIGN_IDS = tuple(geom.REL_TABLE)


def log(*a) -> None:
    print("[sides.decide]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ inputs

def tri_coco(tri: dict, min_views: int = 2, key: str = "Xd") -> np.ndarray:
    """(T,17,3) triangulated joints with >= min_views inlier views (NaN elsewhere)"""
    return np.where((tri["nv"] >= min_views)[..., None], tri[key], np.nan).astype(float)


def her_half_width(caps_f: dict) -> float:
    """her torso half-width from the ribs slab capsule: half its axis + its radius"""
    try:
        i = caps_f["names"].index("ribs")
        a, b = np.asarray(caps_f["A"][i][2], float), np.asarray(caps_f["B"][i][2], float)
        return float(0.5 * np.linalg.norm(b - a) + caps_f["r"][i])
    except (ValueError, IndexError, TypeError):
        return 0.14


def load_crops(path: Path | None) -> list:
    """crops.json contract (written by sides.crops, read here):
        {"items": [{"t": dance s, "cam": "00", "relation": "R4" | "<sign relation id>",
                    "p": {"front": p, "flank": p, "back": p}  (R4)  |  {"+": p}  (sign relations: P(the + side)),
                    "score": calibration score of the question (agent looks: 1.0), "source": "jev" | "agent"}]}
    Items with score < cfg.crop_min_score are ignored."""
    if not path or not Path(path).exists():
        return []
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(d.get("items", []))


def hold_weight(t: np.ndarray, t0: float, band: np.ndarray | None, cfg: SidesCfg) -> np.ndarray:
    """(T,) weight of a crop look at t0 held along the band run that contains it: 1 within crop_hold_full_s, linear to 0 at crop_hold_s,
    0 outside the run (a look says the side at its instant; while the limbs stay entangled / the palm stays on her, the side cannot
    change without a pass-through, which the HMM's band transitions price separately)"""
    if band is None or cfg.crop_hold_s <= 0:
        return np.zeros(len(t))
    b = np.asarray(band, bool)
    k0 = int(np.argmin(np.abs(t - t0)))
    if not b[k0]:
        return np.zeros(len(t))
    a = k0
    while a > 0 and b[a - 1]:
        a -= 1
    e = k0
    while e + 1 < len(b) and b[e + 1]:
        e += 1
    d = np.abs(t - t0)
    full, end = cfg.crop_hold_full_s, max(cfg.crop_hold_s, cfg.crop_hold_full_s + 1e-6)
    w = np.clip((end - d) / (end - full), 0.0, 1.0)
    out = np.zeros(len(t))
    out[a:e + 1] = w[a:e + 1]
    return out


def crop_terms(items: list, t: np.ndarray, cfg: SidesCfg, bands: dict | None = None) -> dict:
    """{relation: (T,) log-odds (sign relations) or (T,3) centred class log-likelihoods (R4)}, triangular weights over +-halfwidth
    (DESIGN), held along the relation's band run when `bands` {relation: (T,) bool} gives one (hold_weight)"""
    out = {}
    for it in items:
        if float(it.get("score", 0.0)) < cfg.crop_min_score:
            continue
        rid = str(it["relation"])
        w = np.clip(1.0 - np.abs(t - float(it["t"])) / cfg.crop_halfwidth_s, 0.0, 1.0)
        if bands is not None and rid in bands:
            w = np.maximum(w, hold_weight(t, float(it["t"]), bands[rid], cfg))
        p = it.get("p", {})
        if rid == "R4":
            v = np.log(np.clip([float(p.get(c, 1 / 3)) for c in CLASSES], 1e-3, 1.0))
            v = np.clip(v - v.mean(), -cfg.clip_crop, cfg.clip_crop)
            out.setdefault(rid, np.zeros((len(t), 3)))
            out[rid] += w[:, None] * v[None]
        else:
            q = float(np.clip(p.get("+", 0.5), 1e-3, 1 - 1e-3))
            out.setdefault(rid, np.zeros(len(t)))
            out[rid] += w * float(np.clip(np.log(q / (1 - q)), -cfg.clip_crop, cfg.clip_crop))
    for k, v in out.items():
        out[k] = np.clip(v, -cfg.clip_crop, cfg.clip_crop)
    return out


# ------------------------------------------------------------------------------------------------ margins and their noise

def all_margins(XL: np.ndarray, XF: np.ndarray, hw: float, ahead: float) -> dict:
    return geom.margins(geom.points_from_coco(XL, ahead), geom.points_from_coco(XF, ahead), half_width_m=hw)


def margin_sigmas(XL: np.ndarray, XF: np.ndarray, SL: np.ndarray, SF: np.ndarray, hw: float, ahead: float, h: float = 1e-3) -> tuple[dict, dict]:
    """(margins of the triangulated joints, {id: sigma_m (T,)}) with sigma_m^2 = sum_i a_i^T S_i a_i, a_i by forward differences"""
    M0 = all_margins(XL, XF, hw, ahead)
    T = len(XL)
    sig2 = {k: np.zeros(T) for k in M0}
    for who in ("L", "F"):
        X, S = (XL, SL) if who == "L" else (XF, SF)
        for j in range(17):
            if not np.isfinite(X[:, j, 0]).any():
                continue
            g = {k: np.zeros((T, 3)) for k in M0}
            for ax in range(3):
                Xp = X.copy()
                Xp[:, j, ax] += h
                M1 = all_margins(Xp, XF, hw, ahead) if who == "L" else all_margins(XL, Xp, hw, ahead)
                for k in M0:
                    with np.errstate(invalid="ignore"):
                        g[k][:, ax] = np.nan_to_num((np.asarray(M1[k]["m"], float) - np.asarray(M0[k]["m"], float)) / h)
            Sj = np.nan_to_num(S[:, j].astype(float))
            for k in M0:
                sig2[k] += np.einsum("ti,tij,tj->t", g[k], Sj, g[k])
    return M0, {k: np.sqrt(v) for k, v in sig2.items()}


def l3d(m: np.ndarray, sigma: np.ndarray, dz: float, clip: float, floor: float) -> np.ndarray:
    """sign(m) clip(1.6 (|m| - dz) / sigma, 0, clip); 0 where m is unknown or inside the dead zone"""
    s = np.sqrt(np.nan_to_num(sigma) ** 2 + floor ** 2)
    with np.errstate(invalid="ignore"):
        v = np.sign(m) * np.clip(1.6 * (np.abs(m) - dz) / s, 0.0, clip)
    return np.nan_to_num(v)


# ------------------------------------------------------------------------------------------------ crossing votes (R9)

COCO_ARM = {"L": (5, 7, 9), "R": (6, 8, 10)}


def r9_votes(views: dict, cams: list, sa: str, sb: str, cfg: SidesCfg) -> np.ndarray:
    """(T,V) per view the vote on the sign of G(his arm sa, her arm sb): the 2D crossings of the four segment pairs, each o_c sigma_c,
    summed and signed"""
    T = len(next(iter(views.values()))["uv"])
    out = np.zeros((T, len(cams)), np.int8)
    ja, jb = COCO_ARM[sa], COCO_ARM[sb]
    for v, c in enumerate(cams):
        z = views[c]
        uv, cf, det, ok = z["uv"].astype(float), z["conf"].astype(float), z["det"].astype(float), z["ok"].astype(bool)
        tot = np.zeros(T)
        for i in range(2):
            for j in range(2):
                a = [ja[i], ja[i + 1]]
                b = [jb[j], jb[j + 1]]
                uA, uB = uv[:, 0][:, a], uv[:, 1][:, b]
                cA, cB = cf[:, 0][:, a], cf[:, 1][:, b]
                val = ok & (det[:, 0] >= cfg.det_conf) & (det[:, 1] >= cfg.det_conf) & (cA >= 0.3).all(1) & (cB >= 0.3).all(1)
                val &= np.isfinite(uA).all((1, 2)) & np.isfinite(uB).all((1, 2))
                vt = crossing_votes(np.nan_to_num(uA), cA, np.nan_to_num(uB), cB, cfg.vote_conf_gap)
                tot += np.where(val, vt, 0)
        out[:, v] = np.sign(tot).astype(np.int8)
    return out


def vote_weights(votes: np.ndarray, labels: np.ndarray, cfg: SidesCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, n = cue_reliability(votes, labels)
    w = np.where(q >= cfg.q_min, np.log(q / (1 - q)), 0.0)
    return w, q, n


# ------------------------------------------------------------------------------------------------ R4: front / flank / back

SITE_CLASS = {"front": ("chest", "chest_L", "chest_R", "shoulder_front_L", "shoulder_front_R"),
              "flank": ("side_L", "waist_L", "side_R", "waist_R"),
              "back": tuple(f"back_{x}_{lv}" for x in ("L2", "L1", "C", "R1", "R2") for lv in ("hi", "mid", "lo"))
              + ("shoulder_blade_L", "shoulder_blade_R", "shoulder_back_L", "shoulder_back_R", "upper_back", "mid_back", "lower_back",
                 "back_side_L", "back_side_R")}


def r4_class_base(fb: np.ndarray, fl: np.ndarray, cfg: SidesCfg) -> np.ndarray:
    """the class a palm position says (1 front, 2 flank, 3 back, 0 unknown): flank = beyond her half-width (less the dead zone) and
    within 12 cm of her coronal plane"""
    out = np.zeros(len(fb), np.int8)
    fin = np.isfinite(fb) & np.isfinite(fl)
    fla = fin & (fl > -cfg.r4_fl_dz_m) & (np.abs(np.nan_to_num(fb)) < 0.12)
    out[fin & ~fla & (np.nan_to_num(fb) > 0)] = 1
    out[fin & ~fla & (np.nan_to_num(fb) <= 0)] = 3
    out[fla] = 2
    return out


def r4_l3d(fb, fl, sfb, sfl, cfg: SidesCfg) -> np.ndarray:
    """(T,3) centred class log-likelihoods (front, flank, back) from the triangulated palm: P(flank) = sigma(e_fl),
    P(front | not flank) = sigma(e_fb), relative to the no-evidence distribution (1/4, 1/2, 1/4)"""
    efb = l3d(fb, sfb, cfg.r4_fb_dz_m, cfg.clip_3d, 0.005)
    efl = l3d(fl, sfl, cfg.r4_fl_dz_m, cfg.clip_3d, 0.005)
    pfl = 1 / (1 + np.exp(-efl))
    pfr = 1 / (1 + np.exp(-efb))
    P = np.stack([(1 - pfl) * pfr, pfl, (1 - pfl) * (1 - pfr)], 1)
    L = np.log(np.maximum(P, 1e-9)) - np.log(np.array([0.25, 0.5, 0.25]))[None]
    known = np.isfinite(fb) & np.isfinite(fl)
    L[~known] = 0.0
    return L - L.mean(1, keepdims=True)


def surface_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray, r: np.ndarray) -> np.ndarray:
    """points (T,3) to a set of capsules (T,K,3) x 2 (r (K,)) -> (T,) signed clearance (min axis distance - r)"""
    from .. import zouk_contacts as zc
    d, _, _ = zc.seg_seg(P[:, None], P[:, None], A, B)
    return (d - r[None]).min(1)


def inside_depth(caps_l: dict, caps_f: dict, posed_l, posed_f) -> np.ndarray:
    """(T,) the deepest overlap (m, >= 0) of his right forearm / palm capsules with her torso slabs"""
    from .. import zouk_contacts as zc
    Al, Bl, rl, pl = posed_l
    Af, Bf, rf, pf = posed_f
    ia = [i for i, p in enumerate(pl) if p in ("forearm_R", "hand_R")]
    ib = [i for i, p in enumerate(pf) if p == "torso"]
    d, _, _ = zc.seg_seg(Al[:, ia][:, :, None], Bl[:, ia][:, :, None], Af[:, ib][:, None], Bf[:, ib][:, None])
    clr = d - rl[ia][None, :, None] - rf[ib][None, None, :]
    return np.maximum(0.0, -clr.reshape(len(d), -1).min(1))


def r4_visibility_test(frames: np.ndarray, palm_base: np.ndarray, elbow: np.ndarray, sites_f: dict, Jf: np.ndarray, Rf: np.ndarray,
                       occ: tuple, obs: dict, views: dict, cams: list, rates: dict, cfg: SidesCfg) -> dict:
    """the generative R4 test on `frames`: per class h (front, flank, back) put his palm on the nearest of her sites of that class to the
    base palm (lifted palm_lift_m off her surface), his wrist palm_from_wrist_m back toward the observed elbow (a simplification of
    zouk_contacts.arm_ik that keeps the elbow), predict per view whether the palm and wrist are visible, and score the views' observations:
    MAMMA right-hand visibility, the COCO right wrist's presence and position (a visible wrist near its predicted pixel; a hidden one is a
    hallucination, broad around it) -> {"l" (n,3) log-likelihoods, "vis" (n,3,V) palm visible, "palm" (n,3,3), "wrist" (n,3,3)}"""
    from .. import zouk_contacts as zc
    n = len(frames)
    V = len(cams)
    OA, OB, Orad = occ
    pts = {}
    for h in CLASSES:
        P, N = [], []
        for nm in SITE_CLASS[h]:
            if nm in sites_f:
                p, nn = zc.site_world(sites_f[nm], Jf[frames], Rf[frames])
                P.append(p)
                N.append(nn)
        P, N = np.stack(P, 1), np.stack(N, 1)                    # (n,S,3)
        k = np.argmin(np.linalg.norm(P - palm_base[frames][:, None], axis=-1), 1)
        pts[h] = (P[np.arange(n), k] + cfg.palm_lift_m * N[np.arange(n), k])
    L = np.zeros((n, 3))
    VIS = np.zeros((n, 3, V), bool)
    PALM = np.zeros((n, 3, 3))
    WR = np.zeros((n, 3, 3))
    for hi, h in enumerate(CLASSES):
        palm = pts[h]
        u = palm - elbow[frames]
        wr = palm - cfg.palm_from_wrist_m * u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        PALM[:, hi], WR[:, hi] = palm, wr
        X = np.stack([palm, wr], 1)
        vis = visible(obs, cams, X, OA[frames], OB[frames], Orad, None, cfg.occl_margin_m, frames=frames)   # (n,2,V)
        VIS[:, hi] = vis[:, 0]
        for v, c in enumerate(cams):
            o = obs[c]
            z = views[c]
            okc = o["ok"][frames]
            m_obs = z["part_vis"][frames, 0, 2].astype(float) > cfg.hand_vis_thr   # MAMMA: his right hand
            d_obs = (z["conf"][frames, 0, 10].astype(float) >= 0.5) & (z["det"][frames, 0].astype(float) >= cfg.det_conf)
            uv_obs = z["uv"][frames, 0, 10].astype(float)
            a, b, cc, e = (np.clip(rates[c][k], cfg.p_vis_floor, 1 - cfg.p_vis_floor) for k in ("hand_vis", "hand_hid", "wr_vis", "wr_hid"))
            vh, vw = vis[:, 0, v], vis[:, 1, v]
            pm = np.where(vh, a, b)
            lm = np.where(m_obs, np.log(pm), np.log(1 - pm))
            pd = np.where(vw, cc, e)
            ld = np.where(d_obs, np.log(pd), np.log(1 - pd))
            uvp, dep = project(o["K"][frames], o["R"][frames], o["t"][frames], wr[:, None])
            r = np.linalg.norm(uv_obs - uvp[:, 0], axis=-1)
            f = o["f"][frames]
            sv, sh = cfg.vis_sigma_px * f, cfg.hall_sigma_px * f
            g = lambda rr, s: np.exp(-0.5 * (rr / s) ** 2) / (2 * np.pi * s * s)
            with np.errstate(all="ignore"):
                lpv = np.log(0.9 * g(r, sv) + 0.1 * g(r, sh) + 1e-12)
                lph = np.log(0.7 * g(r, sh) + 0.3 / (np.pi * (3 * sh) ** 2) + 1e-12)
            lp = np.where(vw, lpv, lph)
            lp = np.where(d_obs & np.isfinite(r) & (dep[:, 0] > 0.1), lp, 0.0)
            L[:, hi] += np.where(okc, lm + ld + lp, 0.0)
    return {"l": L, "vis": VIS, "palm": PALM, "wrist": WR}


def r4_look_labels(items: list, t: np.ndarray, cfg: SidesCfg) -> list:
    """[(frame, class index 0..2)] of the agent's R4 looks (score >= crop_min_score) whose class probability is >= r4_cue_label_p"""
    out = []
    for it in items:
        if it.get("relation") != "R4" or it.get("source") != "agent" or float(it.get("score", 0.0)) < cfg.crop_min_score:
            continue
        p = it.get("p", {})
        v = [float(p.get(c, 0.0)) for c in CLASSES]
        h = int(np.argmax(v))
        if v[h] >= cfg.r4_cue_label_p:
            out.append((int(np.argmin(np.abs(t - float(it["t"])))), h))
    return out


def r4_cue_reliability(labels: list, cues: dict, cfg: SidesCfg) -> dict:
    """per R4 cue {name: (T,3) centred log-likelihoods}: its arg-max against the looks' classes on the labelled frames where it votes
    (spread > 0.05 nats) -> {name: {"n", "acc", "weight" (1 = evidence, 0 = check only), "confusion"}} (DESIGN 2.3's calibration rule:
    a cue is evidence only when it agrees with known answers; the votes' q < 0.55 -> w = 0, Jev < 0.8 -> check only)"""
    out = {}
    for name, L in cues.items():
        conf = np.zeros((3, 3), int)
        for k, h in labels:
            v = np.asarray(L[k], float)
            if np.ptp(v) <= 0.05:
                continue
            conf[h, int(np.argmax(v))] += 1
        n = int(conf.sum())
        acc = float(np.trace(conf) / n) if n else None
        w = 1.0 if (n < cfg.r4_cue_min_n or acc is None) else float(acc >= cfg.r4_cue_min_acc)
        # P(cue votes c | truth h): the counts + a Dirichlet prior that trusts the cue a little by default (diagonal alpha_diag,
        # off-diagonal 1); a truth class never labelled keeps the prior row
        A = np.ones((3, 3)) + (cfg.r4_cue_alpha_diag - 1.0) * np.eye(3)
        P = (conf + A) / (conf + A).sum(1, keepdims=True)
        out[name] = {"n": n, "acc": None if acc is None else round(acc, 3), "weight": w, "confusion_truth_x_cue": conf.tolist(),
                     "p_cue_given_truth": np.round(P, 3).tolist()}
    return out


def calibrated_cue(L: np.ndarray, rep: dict, cfg: SidesCfg) -> np.ndarray:
    """(T,3) a cue's centred log-likelihoods replaced by its calibrated vote: log P(cue's arg-max | truth h) (r4_cue_reliability), scaled
    by how decisive the cue is (its spread / clip_3d, at most 1); the cue as it is when it has no calibration (too few looks)"""
    if rep is None or rep.get("n", 0) < cfg.r4_cue_min_n:
        return L
    if cfg.r4_cue_mode == "weight":
        return L * rep["weight"]
    P = np.asarray(rep["p_cue_given_truth"], float)
    lp = np.log(P)                                           # (truth h, cue c)
    c = np.argmax(L, 1)
    v = lp[:, c].T                                           # (T, 3): log P(c_t | h)
    v = v - v.mean(1, keepdims=True)
    s = np.clip(np.ptp(L, 1) / cfg.clip_3d, 0.0, 1.0)
    s[np.ptp(L, 1) <= 0.05] = 0.0
    return v * s[:, None]


def hand_unseen(views: dict, cams: list, cfg: SidesCfg) -> np.ndarray:
    """(T,) no view's MAMMA sees his right hand (part visibility > hand_vis_thr), as a majority over r4_blind_window_s"""
    from scipy.ndimage import uniform_filter1d
    n = None
    for c in cams:
        z = views[c]
        s = (z["part_vis"][:, 0, 2].astype(float) > cfg.hand_vis_thr) & z["ok"].astype(bool)
        n = s.astype(int) if n is None else n + s
    w = max(1, int(round(cfg.r4_blind_window_s * cfg.hz)))
    return uniform_filter1d((n == 0).astype(float), w, mode="nearest") >= 0.5


def r4_prior(fcls: np.ndarray, blind: np.ndarray, cfg: SidesCfg) -> np.ndarray:
    """(T,3) prior class log-likelihoods (front, flank, back) per couple-facing class ([sides.priors] R4), capped at prior_nats; in
    camera-blind frames scaled up to blind_prior_nats (owner decision D1 (b); blind_prior_nats = prior_nats is decision (a))"""
    pri = cfg.priors.get("R4", {})
    T = len(fcls)
    Lp = np.zeros((T, 3))
    for fc in ("face", "side", "away"):
        m = np.asarray(fcls) == fc
        v = np.array([float(pri.get(fc, {}).get(c, 0.0)) for c in CLASSES])
        Lp[m] = v[None]
    blind = np.asarray(blind, bool)
    cap = np.where(blind, cfg.blind_prior_nats, cfg.prior_nats)
    scale = np.where(blind, cfg.blind_prior_nats / max(cfg.prior_nats, 1e-9), 1.0)
    return np.clip(Lp * scale[:, None], -cap[:, None], cap[:, None])


def r4_rates(views: dict, cams: list, free: np.ndarray, cfg: SidesCfg) -> dict:
    """per camera P(MAMMA sees his right hand | palm visible / hidden) and P(COCO right wrist | wrist visible / hidden) on free frames
    (pelvises apart), where the base's own visibility is trusted (Beta(1,1))"""
    out = {}
    for c in cams:
        z = views[c]
        ok = z["ok"].astype(bool) & free
        vh = z["vis_palm"][:, 0, 1].astype(bool)
        vw = z["vis_pred"][:, 0, 10].astype(bool)
        m = z["part_vis"][:, 0, 2].astype(float) > cfg.hand_vis_thr
        d = (z["conf"][:, 0, 10].astype(float) >= 0.5) & (z["det"][:, 0].astype(float) >= cfg.det_conf)
        def rate(x, cond):
            k = ok & cond
            return float((x[k].sum() + 1) / (k.sum() + 2)), int(k.sum())
        hv, nhv = rate(m, vh)
        hh, nhh = rate(m, ~vh)
        wv, nwv = rate(d, vw)
        wh, nwh = rate(d, ~vw)
        out[c] = {"hand_vis": hv, "hand_hid": hh, "wr_vis": wv, "wr_hid": wh, "n": [nhv, nhh, nwv, nwh]}
    return out


# ------------------------------------------------------------------------------------------------ yaw track (DESIGN 2.4)

def _halfplane_log(delta_deg: np.ndarray, bit: np.ndarray, q: float, soft: float) -> np.ndarray:
    """log P(bit | heading) for a front (+1) / back (-1) bit of a view: P(front seen) = q inside +-90 deg of the view, 1-q outside,
    softened over `soft` deg"""
    s = 1 / (1 + np.exp(-(90.0 - np.abs(delta_deg)) / soft))
    pf = (1 - q) + (2 * q - 1) * s
    return np.where(bit[..., None] > 0, np.log(pf), np.where(bit[..., None] < 0, np.log(1 - pf), 0.0))


def yaw_track(role: str, base_r: dict, Xd: np.ndarray, nv: np.ndarray, views: dict, obs: dict, cams: list, cfg: SidesCfg) -> dict:
    """the yaw HMM of one dancer: heading psi = the hip-line heading; emissions from the per-view face bits (head = psi + the base's
    head-pelvis twist, clipped), the per-view L/R order (torso), the triangulated hip / shoulder lines (calm-frame bias removed) and a weak
    anchor to the base; transitions around the base's angular velocity -> psi* (unwrapped), conf, err = wrap(psi_base - psi*),
    face contradictions of the base, spin candidates and defects"""
    hz = cfg.hz
    p = PIDX[role]
    J = np.asarray(base_r["joints"], float)
    T = len(J)
    Rg = geom.global_rotations(base_r["global_orient"], base_r["body_pose"])
    psi_b = geom.line_heading(J[:, 1], J[:, 2])
    psi_bu = geom.unwrap_deg(psi_b)
    head_f = geom.heading(Rg[:, geom.HEAD] @ np.array([0, 0, 1.0]))
    tau_h = np.clip(geom.wrap_deg(head_f - psi_b), -cfg.yaw_twist_clip_deg, cfg.yaw_twist_clip_deg)
    sh_b = geom.line_heading(J[:, 16], J[:, 17])
    tau_s = np.clip(geom.wrap_deg(sh_b - psi_b), -cfg.yaw_twist_clip_deg, cfg.yaw_twist_clip_deg)
    om = np.gradient(np.nan_to_num(psi_bu)) * hz
    from scipy.ndimage import median_filter
    om = median_filter(om, 5, mode="nearest")
    win = int(round(0.5 * hz))
    dpsi = np.abs(np.roll(psi_bu, -win) - np.roll(psi_bu, win))
    dpsi[:win] = dpsi[-win:] = 0
    cand_base = dpsi > cfg.spin_cand_dps
    B = int(cfg.yaw_bins)
    cb = np.arange(B) * 360.0 / B
    LL = np.zeros((T, B))
    n_bits = np.zeros(T, int)
    face_contra = np.zeros(T, int)
    n_face = np.zeros(T, int)
    qs = {}
    for c in cams:
        z = views[c]
        ok = z["ok"].astype(bool)
        C = obs[c]["C"]
        az_head = geom.heading(C - J[:, geom.HEAD])
        az_tor = geom.heading(C - J[:, geom.SPINE3])
        # a face bit counts only where the base predicts the head visible from this view (a head hidden by the partner reads as "back")
        fb = np.where(ok & z["vis_pred"][:, p, 0].astype(bool), z["face"][:, p], 0).astype(int)
        lr = z["lr"][:, p].astype(float)
        # an L/R bit counts only where the base predicts all four of shoulders and hips visible (a partly hidden torso swaps COCO's labels)
        vis4 = z["vis_pred"][:, p][:, [5, 6, 11, 12]].astype(bool).all(1)
        lb = np.where(ok & vis4 & np.isfinite(lr), np.where(lr > cfg.lr_dead, 1, np.where(lr < -cfg.lr_dead, -1, 0)), 0)
        # calibration on calm frames against the base (outside spin candidates): q = the share of bits agreeing
        d_head = np.abs(geom.wrap_deg(head_f - az_head))
        d_tor = np.abs(geom.wrap_deg(sh_b - az_tor))
        calm = ~cand_base
        def q_of(bit, d):
            m = calm & (bit != 0) & np.isfinite(d)
            agree = ((bit > 0) & (d < 90)) | ((bit < 0) & (d > 90))
            return float(np.clip((agree[m].sum() + 1) / (m.sum() + 2), 0.6, 0.97)), int(m.sum())
        qf, nf = q_of(fb, d_head)
        ql, nl = q_of(lb, d_tor)
        qs[c] = {"face": qf, "face_n": nf, "lr": ql, "lr_n": nl}
        dh = geom.wrap_deg(cb[None] + tau_h[:, None] - np.nan_to_num(az_head)[:, None])
        LL += _halfplane_log(dh, fb, qf, cfg.yaw_soft_deg)
        dt = geom.wrap_deg(cb[None] + tau_s[:, None] - np.nan_to_num(az_tor)[:, None])
        LL += _halfplane_log(dt, lb, ql, cfg.yaw_soft_deg)
        n_bits += (fb != 0).astype(int) + (lb != 0).astype(int)
        face_contra += (((fb > 0) & (d_head > 110)) | ((fb < 0) & (d_head < 70))).astype(int)
        n_face += (fb != 0).astype(int)
    # triangulated hip / shoulder lines
    def line_ok(a, b, ja, jb):
        w_t = np.linalg.norm((Xd[:, a] - Xd[:, b])[:, [0, 2]], axis=1)
        w_b = np.linalg.norm((J[:, ja] - J[:, jb])[:, [0, 2]], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rat = w_t / np.maximum(w_b, 1e-6)
        lo, hi = cfg.yaw_line_width
        return (nv[:, a] >= cfg.yaw_line_min_views) & (nv[:, b] >= cfg.yaw_line_min_views) & (rat >= lo) & (rat <= hi)
    ok_h = line_ok(11, 12, 1, 2)
    psi_t = np.where(ok_h, geom.line_heading(Xd[:, 11], Xd[:, 12]), np.nan)
    ok_s = line_ok(5, 6, 16, 17)
    psi_ts = np.where(ok_s, geom.line_heading(Xd[:, 5], Xd[:, 6]), np.nan)
    bias_h = float(np.nanmedian(geom.wrap_deg(psi_t - psi_b)[~cand_base])) if np.isfinite(psi_t[~cand_base]).any() else 0.0
    bias_s = float(np.nanmedian(geom.wrap_deg(psi_ts - sh_b)[~cand_base])) if np.isfinite(psi_ts[~cand_base]).any() else 0.0
    ks_ = 1.0 / np.radians(cfg.yaw_shoulder_sigma_deg) ** 2
    ds = cb[None] + tau_s[:, None] - (psi_ts - bias_s)[:, None]
    LL += np.where(np.isfinite(ds), ks_ * (np.cos(np.radians(np.nan_to_num(ds))) - 1), 0.0)
    if cfg.yaw_base_sigma_deg > 0:
        kb = 1.0 / np.radians(cfg.yaw_base_sigma_deg) ** 2
        LL += kb * (np.cos(np.radians(cb[None] - np.nan_to_num(psi_b)[:, None])) - 1)
    k = 1.0 / np.radians(cfg.yaw_line_sigma_deg) ** 2
    dl = cb[None] - (psi_t - bias_h)[:, None]
    LLh = np.where(np.isfinite(dl), k * (np.cos(np.radians(np.nan_to_num(dl))) - 1), 0.0)
    yv = hmm.yaw_viterbi(LL + LLh, om, cfg, hz, psi0=float(psi_bu[0]))
    # the same track without the hip lines: "COCO-3D hip lines alone never drive a spin repair" (DESIGN 2.4) - an error counts as a
    # defect only where both tracks disagree with the base
    yn = hmm.yaw_viterbi(LL, om, cfg, hz, psi0=float(psi_bu[0]))
    psi = yv["psi"]
    err = geom.wrap_deg(psi_bu - psi)
    err_n = geom.wrap_deg(psi_bu - yn["psi"])
    dps = np.abs(np.roll(psi, -win) - np.roll(psi, win))
    dps[:win] = dps[-win:] = 0
    cand = cand_base | (dps > cfg.spin_cand_dps)
    contra = (face_contra >= 2) & (face_contra >= 0.5 * np.maximum(1, n_face))
    defect = ((np.abs(err) > cfg.spin_err_deg) & (np.abs(err_n) > cfg.spin_err_deg)) | contra
    runs = hmm.runs(defect, merge=2, min_len=cfg.spin_min_frames)
    # winding: the unwrapped base falls a turn behind (or ahead of) the camera track within 2 s ("ends a turn behind")
    du = psi_bu - psi
    w2 = int(round(1.0 * hz))
    dj = np.zeros(T)
    dj[w2:-w2] = du[2 * w2:] - du[:-2 * w2]
    turns = np.round(dj / 360.0)
    dm = np.zeros(T, bool)
    for a, b in runs:
        dm[a:b + 1] = True
    return {"psi_star": psi, "conf": yv["conf"], "psi_base": psi_bu, "err": err, "err_nohip": err_n, "psi_star_nohip": yn["psi"],
            "turns": turns,
            "face_contra": face_contra, "n_bits": n_bits, "spin_cand": cand, "defect": dm, "runs": runs, "q": qs, "bias_hip_deg": bias_h,
            "bias_shoulder_deg": bias_s}


# ------------------------------------------------------------------------------------------------ violations

def _runs_items(mask: np.ndarray, t: np.ndarray, reason: str, min_frames: int, merge: int = 3, **extra) -> list:
    out = []
    for a, b in hmm.runs(mask, merge=merge, min_len=min_frames):
        out.append({"reason": reason, "k0": int(a), "k1": int(b), "t0": round(float(t[a]), 3), "t1": round(float(t[b]), 3),
                    "frames": int(b - a + 1), **extra})
    return out


def band_rms(y: np.ndarray, hz: float, band=(1.0, 5.0), win_s: float = 2.0) -> np.ndarray:
    """(T,) RMS of the band-passed signal over a centred window (NaN gaps interpolated; NaN where the window is mostly missing)"""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import butter, sosfiltfilt
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 30:
        return np.full(len(y), np.nan)
    idx = np.flatnonzero(ok)
    yi = np.interp(np.arange(len(y)), idx, y[idx])
    sos = butter(2, [band[0] / (hz / 2), band[1] / (hz / 2)], btype="band", output="sos")
    f = sosfiltfilt(sos, yi)
    w = max(3, int(round(win_s * hz)))
    rms = np.sqrt(uniform_filter1d(f * f, w, mode="nearest"))
    cov = uniform_filter1d(ok.astype(float), w, mode="nearest")
    return np.where(cov >= 0.8, rms, np.nan)


TWIST_AXES = {3: 6, 6: 9, 9: 12, 12: 15, 15: 15, 1: 4, 2: 5, 4: 7, 5: 8, 7: 10, 8: 11}
TWIST_GROUP = {3: "spine", 6: "spine", 9: "spine", 12: "neck_head", 15: "neck_head", 1: "hip", 2: "hip", 4: "knee", 5: "knee",
               7: "ankle", 8: "ankle"}
CHAINS = {"spine": (3, 6, 9), "neck": (12, 15), "leg_L": (1, 4, 7), "leg_R": (2, 5, 8)}


def twists(base_r: dict, rest_J: np.ndarray | None) -> dict:
    """{joint: twist (T,) deg} of the local rotations about each joint's rest bone axis (zouk_contacts.wrist_angles_R pattern)"""
    Rl = geom.local_rotations(base_r["global_orient"], base_r["body_pose"])
    out = {}
    for j, ch in TWIST_AXES.items():
        if rest_J is not None:
            a = rest_J[ch] - rest_J[j] if ch != j else rest_J[15] - rest_J[12]
        else:
            a = np.array([0, -1.0, 0]) if j in (1, 2, 4, 5) else np.array([0, 1.0, 0])
        a = a / max(np.linalg.norm(a), 1e-9)
        out[j] = geom.swing_twist(Rl[:, j], a)[0]
    return out


def violations(t: np.ndarray, rel: dict, base: dict, tri: dict, yaw: dict, rest: dict, cfg: SidesCfg, fm: FilmMap) -> list:
    """the places where the base breaks a decided side or the real joint positions (DESIGN 3.1 reasons: side, thru, inside, real, swing,
    spin, limit, wrap) -> [{reason, role, rel / joint, k0, k1, t0, t1, frames, worst, film0, film1, film_extra}]"""
    items = []
    hz = cfg.hz
    # side
    for rid in SIGN_IDS:
        s, mb = rel[f"{rid}__s"], rel[f"{rid}__m_base"]
        dz = geom.REL_TABLE[rid][1]
        with np.errstate(invalid="ignore"):
            bad = (s != 0) & (s * mb < -dz / 2)
        for it in _runs_items(bad, t, "side", cfg.side_min_frames, rel=rid):
            seg = slice(it["k0"], it["k1"] + 1)
            it["worst"] = round(float(np.nanmin(s[seg] * mb[seg])), 4)
            it["decided"] = int(np.sign(np.sum(s[seg])))
            items.append(it)
    cls, cb = rel["R4__cls"], rel["R4__cls_base"]
    fb, fl = rel["R4__fb_base"], rel["R4__fl_base"]
    clear = np.zeros(len(t), bool)
    clear |= (cls == 1) & ((cb == 3) & (fb < -cfg.r4_fb_dz_m))
    clear |= (cls == 3) & ((cb == 1) & (fb > cfg.r4_fb_dz_m))
    clear |= (cls == 2) & (fl < -2 * cfg.r4_fl_dz_m)
    clear |= ((cls == 1) | (cls == 3)) & (cb == 2) & (fl > 2 * cfg.r4_fl_dz_m)
    for it in _runs_items(clear, t, "side", cfg.side_min_frames, rel="R4"):
        seg = slice(it["k0"], it["k1"] + 1)
        vals, cnt = np.unique(cls[seg], return_counts=True)
        it["decided"] = CLASSES[int(vals[np.argmax(cnt)]) - 1]
        vb, cntb = np.unique(cb[seg], return_counts=True)
        it["base"] = CLASSES[int(vb[np.argmax(cntb)]) - 1] if vb[np.argmax(cntb)] > 0 else "?"
        it["blind_share"] = round(float(rel["R4__blind"][seg].mean()), 2)
        items.append(it)
    # thru: pass-through events of the base, 4 arm pairings
    for rid in geom.R9_PAIRS:
        G = rel[f"{rid}__G_base"]
        s = rel[f"{rid}__s"]
        band = rel[f"{rid}__band"]
        for k in geom.passthrough_events(G, cfg.thru_jump):
            lo, hi = max(0, k - 5), min(len(t), k + 5)
            held = bool((s[lo:hi] != 0).any())
            items.append({"reason": "thru", "rel": rid, "k0": int(k - 1), "k1": int(k), "t0": round(float(t[k - 1]), 3), "t1": round(float(t[k]), 3),
                          "frames": 2, "worst": round(float(G[k] - G[k - 1]), 3), "held": held, "in_band": bool(band[k])})
    # inside: his palm / forearm > 3 cm deep in her torso capsules
    dep = rel["R4__inside_base"]
    for it in _runs_items(dep > cfg.r4_inside_m, t, "inside", cfg.inside_min_frames, role="lead"):
        it["worst"] = round(float(dep[it["k0"]:it["k1"] + 1].max()), 4)
        items.append(it)
    # real: base joint far from the triangulated one
    from scipy.ndimage import uniform_filter1d
    for r in cfg.roles:
        J = np.asarray(base[r]["joints"], float)
        tr = tri[r]
        cj = set(cfg.contact_joints.get(r, ()))
        for j in range(5, 17):
            nm = geom.COCO[j]
            ok = tr["nv"][:, j] >= cfg.min_views_real
            X = tr["Xd"][:, j].astype(float)
            d = np.linalg.norm(J[:, geom.C2S[j]] - X, axis=1)
            S = np.nan_to_num(tr["S"][:, j].astype(float))
            sig = np.sqrt(np.maximum(np.linalg.eigvalsh(S)[:, -1], 0))
            tau = cfg.real_tau_contact_m if nm in cj else cfg.real_tau_m
            bad = ok & (d > np.maximum(3 * sig, tau))
            cnt = uniform_filter1d(bad.astype(float), cfg.real_win, mode="constant") * cfg.real_win
            m = (cnt >= cfg.real_min - 1e-6) & ok
            for it in _runs_items(m, t, "real", 3, merge=5, role=r, joint=nm):
                seg = slice(it["k0"], it["k1"] + 1)
                it["worst"] = round(float(np.nanmax(np.where(ok[seg], d[seg], np.nan))), 3)
                it["median"] = round(float(np.nanmedian(np.where(ok[seg], d[seg], np.nan))), 3)
                items.append(it)
    # swing: 1-5 Hz energy of elbow / wrist height beyond the triangulated one
    for r in cfg.roles:
        J = np.asarray(base[r]["joints"], float)
        tr = tri[r]
        for nm, sj, cj in (("elL", 18, 7), ("elR", 19, 8), ("wrL", 20, 9), ("wrR", 21, 10)):
            rb = band_rms(J[:, sj, 1], hz, cfg.swing_band_hz, cfg.swing_win_s)
            yt = np.where(tr["nv"][:, cj] >= 2, tr["Xd"][:, cj, 1].astype(float), np.nan)
            rt = band_rms(yt, hz, cfg.swing_band_hz, cfg.swing_win_s)
            thr = np.maximum(rt, cfg.swing_floor_m) + cfg.swing_slack_m
            m = np.isfinite(thr) & (rb > thr)            # only where the triangulated joint says how much it really moves
            for it in _runs_items(m, t, "swing", int(0.5 * hz), merge=int(0.5 * hz), role=r, joint=nm):
                seg = slice(it["k0"], it["k1"] + 1)
                it["worst"] = round(float(np.nanmax(rb[seg])), 4)
                it["tri_rms"] = round(float(np.nanmedian(rt[seg])), 4) if np.isfinite(rt[seg]).any() else None
                items.append(it)
    # spin: yaw defects
    for r in cfg.roles:
        y = yaw[r]
        for a, b in y["runs"]:
            seg = slice(a, b + 1)
            items.append({"reason": "spin", "role": r, "k0": int(a), "k1": int(b), "t0": round(float(t[a]), 3), "t1": round(float(t[b]), 3),
                          "frames": int(b - a + 1), "worst": round(float(np.max(np.abs(y["err"][seg]))), 1),
                          "face_contra_max": int(y["face_contra"][seg].max()),
                          "turns_off": int(y["turns"][max(0, a - 30):b + 31][np.argmax(np.abs(y["turns"][max(0, a - 30):b + 31]))])})
    # limit + wrap: swing-twist of the base
    for r in cfg.roles:
        tw = twists(base[r], rest.get(r))
        g = cfg.twist_gate
        over = {}
        for j, v in tw.items():
            grp = TWIST_GROUP[j]
            lim = g["spine"] if grp == "spine" else g[grp]
            over[j] = np.abs(v) > lim
        sp_tot = np.abs(tw[3] + tw[6] + tw[9]) > g["spine_total"]
        nh = np.abs(tw[12] + tw[15]) > g["neck_head"]
        masks = {"spine": over[3] | over[6] | over[9] | sp_tot, "neck_head": nh, "hip_L": over[1], "hip_R": over[2], "knee_L": over[4],
                 "knee_R": over[5], "ankle_L": over[7], "ankle_R": over[8]}
        for nm, m in masks.items():
            for it in _runs_items(m, t, "limit", cfg.limit_min_frames, role=r, joint=nm):
                items.append(it)
        Rl = geom.local_rotations(base[r]["global_orient"], base[r]["body_pose"])
        rj = rest.get(r)
        for ch, js in CHAINS.items():
            axes = []
            for j in js:
                c = TWIST_AXES[j]
                a = (rj[c] - rj[j] if c != j else rj[15] - rj[12]) if rj is not None else (np.array([0, -1.0, 0]) if j in (1, 2, 4, 5) else np.array([0, 1.0, 0]))
                axes.append(a / max(np.linalg.norm(a), 1e-9))
            ct = geom.chain_twist(Rl[:, list(js)], np.asarray(axes))
            for it in _runs_items(np.abs(ct["wrap"]) >= cfg.wrap_deg, t, "wrap", 1, role=r, joint=ch):
                seg = slice(it["k0"], it["k1"] + 1)
                it["worst"] = round(float(np.max(np.abs(ct["total"][seg]))), 1)
                items.append(it)
    for it in items:
        it["film0"] = fm.main(it["t0"])
        it["film1"] = fm.main(it["t1"])
        ex = fm.extra(it["t0"]) + fm.extra(it["t1"])
        if ex:
            it["film_extra"] = sorted({e[0] for e in ex})
    items.sort(key=lambda x: (x["t0"], x["reason"]))
    return items


# ------------------------------------------------------------------------------------------------ the decide step

def run(take: str, cfg: SidesCfg | None = None, crops: Path | None = None, out_dir: Path | None = None, ev_dir: Path | None = None) -> dict:
    t_start = time.time()
    cfg = cfg or load_cfg(take)
    ad = adapter_for(take, cfg)
    work = ad.work
    out_dir = Path(out_dir or work / "sides")
    out_dir.mkdir(parents=True, exist_ok=True)
    evd = Path(ev_dir or work / "sides" / "evidence")
    crops = Path(crops) if crops else (work / "sides" / "crops.json")
    cams = ad.cams
    base = ad.base()
    t = ad.t
    T = len(t)
    obs = ad.obs()
    tri = {r: dict(np.load(evd / f"tri_{r}.npz")) for r in cfg.roles}
    views = {}
    for c in cams:
        with np.load(evd / f"views_{c}.npz") as z:
            views[c] = {k: z[k] for k in z.files}
    betas = {r: np.asarray(base[r]["betas"], float) for r in cfg.roles}
    caps = capsule_set(betas, evd / "capsules.json")
    sites = site_set(betas, evd / "sites.json")
    rest = rest_joints(betas, evd / "rest.json")
    J55 = {r: np.asarray(base[r]["joints"], float) for r in cfg.roles}
    Rg = {r: geom.global_rotations(base[r]["global_orient"], base[r]["body_pose"]) for r in cfg.roles}
    posed = {r: posed_capsules(caps[r], J55[r], Rg[r]) for r in cfg.roles}
    hw = her_half_width(caps["follow"])
    log(f"{take}: inputs loaded {round(time.time() - t_start, 1)} s; her half-width {hw:.3f} m")
    # ---------------- margins: base (says "defined" only) and triangulated (evidence)
    PLb, PFb = geom.points_from_smplx(J55["lead"]), geom.points_from_smplx(J55["follow"])
    Mb = geom.margins(PLb, PFb, half_width_m=hw)
    XL, XF = tri_coco(tri["lead"]), tri_coco(tri["follow"])
    SL, SF = tri["lead"]["S"], tri["follow"]["S"]
    Mt, Sig = margin_sigmas(XL, XF, SL, SF, hw, cfg.palm_from_wrist_m)
    log("margins + sigmas", round(time.time() - t_start, 1), "s")
    crop_items = load_crops(crops)
    rel = {"t": t}
    rj = {"what": "sides relations (decide)", "take": take, "relations": {}, "cfg": cfg.to_json(), "crops": str(crops) if crops.exists() else None}

    def defined(rid):
        mt = np.asarray(Mt[rid]["m"], float)
        fin = np.isfinite(mt)
        return np.where(fin, np.asarray(Mt[rid]["gate"], bool) & fin, np.asarray(Mb[rid]["gate"], bool)), fin

    # bands
    bands = {}
    for rid in geom.R9_PAIRS:
        Gt, Gb = np.asarray(Mt[rid]["m"], float), np.asarray(Mb[rid]["m"], float)
        dt_, db_ = np.asarray(Mt[rid]["aux"]["dist"], float), np.asarray(Mb[rid]["aux"]["dist"], float)
        fin = np.isfinite(Gt) & np.isfinite(dt_)
        with np.errstate(invalid="ignore"):
            bt = (np.abs(Gt) > cfg.band_gli) | (dt_ < cfg.band_dist_m)
        bb = (np.abs(Gb) > cfg.band_gli) | (db_ < cfg.band_dist_m)
        bands[rid] = np.where(fin, bt, bb)

    def knee_dist(PL, PF):
        return np.min(np.stack([np.linalg.norm(PL[a] - PF[b], axis=-1) for a in ("knL", "knR") for b in ("knL", "knR")], 1), 1)
    kt = knee_dist(geom.points_from_coco(XL), geom.points_from_coco(XF))
    kb = knee_dist(PLb, PFb)
    with np.errstate(invalid="ignore"):
        bands["legs"] = np.where(np.isfinite(kt), kt < cfg.band_dist_m, kb < cfg.band_dist_m)
    # crop looks of the sign relations, held along each relation's band run
    sign_bands = {rid: bands[geom.REL_TABLE[rid][3]] for rid in SIGN_IDS if geom.REL_TABLE[rid][3] in bands}
    crop_l = crop_terms([it for it in crop_items if it.get("relation") != "R4"], t, cfg, sign_bands)
    # ---------------- sign relations
    for rid in SIGN_IDS:
        kind, dz, dzf, bkey = geom.REL_TABLE[rid]
        gate, fin = defined(rid)
        mt = np.asarray(Mt[rid]["m"], float)
        floor = 0.1 * dz
        l3 = l3d(np.where(gate, mt, np.nan), Sig[rid], dz, cfg.clip_3d, floor)
        lc = np.zeros(T)
        extra = {}
        if rid in geom.R9_PAIRS:
            sa, sb = geom.R9_PAIRS[rid]
            votes = r9_votes(views, cams, sa, sb, cfg)
            with np.errstate(invalid="ignore"):
                lab = np.where(fin & (np.abs(mt) > cfg.q_label_sigma * np.sqrt(Sig[rid] ** 2 + floor ** 2)) & (np.abs(mt) > dz), np.sign(mt), 0)
            w, q, n = vote_weights(votes, lab, cfg)
            lc = (votes * w[None]).sum(1)
            rel[f"{rid}__votes"] = votes
            rel[f"{rid}__G_base"] = np.asarray(Mb[rid]["m"], float)
            rel[f"{rid}__G_tri"] = mt
            extra = {"q": dict(zip(cams, np.round(q, 3).tolist())), "q_n": dict(zip(cams, n.tolist())), "w": dict(zip(cams, np.round(w, 3).tolist()))}
        lcr = crop_l.get(rid, np.zeros(T))
        pr = cfg.priors.get(rid)
        lp = np.full(T, float(np.clip(pr, -cfg.prior_nats, cfg.prior_nats))) if isinstance(pr, (int, float)) else np.zeros(T)
        L = l3 + lc + lcr + lp
        band = bands.get(bkey, np.zeros(T, bool)) if bkey else np.zeros(T, bool)
        d = hmm.decide_sign(L, gate, band, cfg, cfg.hz)
        rel.update({f"{rid}__m_base": np.asarray(Mb[rid]["m"], float), f"{rid}__m_tri": mt, f"{rid}__sigma": Sig[rid], f"{rid}__L": L,
                    f"{rid}__l3d": l3, f"{rid}__lc": lc, f"{rid}__post": d["post"].astype(np.float32), f"{rid}__s": d["s"],
                    f"{rid}__gate": gate, f"{rid}__band": band})
        runs = [(a, b, v) for a, b, v in hmm.label_runs(d["s"]) if v != 0]
        rj["relations"][rid] = {"kind": kind, "dz_decide": dz, "dz_fit": dzf, "band": bkey,
                                "decided_frames": int((d["s"] != 0).sum()), "defined_frames": int(gate.sum()),
                                "runs": [[round(float(t[a]), 3), round(float(t[b]), 3), int(v)] for a, b, v in runs], **extra}
    log("sign relations", round(time.time() - t_start, 1), "s")
    # ---------------- R4
    capf_A, capf_B, capf_r, capf_p = posed["follow"]
    it = [i for i, p in enumerate(capf_p) if p in ("torso", "shoulder_L", "shoulder_R")]
    palm_b = PLb["palmR"]
    PLt = geom.points_from_coco(XL, cfg.palm_from_wrist_m)
    palm_t = PLt["palmR"]
    surf_b = surface_distance(palm_b, capf_A[:, it], capf_B[:, it], capf_r[it])
    fin_t = np.isfinite(palm_t).all(1)
    surf_t = np.full(T, np.nan)
    if fin_t.any():
        surf_t[fin_t] = surface_distance(palm_t[fin_t], capf_A[fin_t][:, it], capf_B[fin_t][:, it], capf_r[it])
    surf = np.where(np.isfinite(surf_t), surf_t, surf_b)
    gate4 = surf < cfg.r4_gate_m
    # in contact by either estimate (Stage 2: the triangulated palm is pulled off her by the COCO wrist bias the crops show at the
    # counterbalances, 56.8-58.4 s; the base palm only ever says "defined", never a side)
    with np.errstate(invalid="ignore"):
        band4 = (surf < cfg.r4_band_m) | (surf_b < cfg.r4_band_m)
    fbb, flb = np.asarray(Mb["R4_fb"]["m"], float), np.asarray(Mb["R4_fl"]["m"], float)
    fbt, flt = np.asarray(Mt["R4_fb"]["m"], float), np.asarray(Mt["R4_fl"]["m"], float)
    L3 = r4_l3d(fbt, flt, Sig["R4_fb"], Sig["R4_fl"], cfg)
    # the visibility test on the gated frames
    frames = np.flatnonzero(gate4)
    Lv = np.zeros((T, 3))
    if len(frames):
        free = (np.hypot(*(J55["lead"][:, 0] - J55["follow"][:, 0])[:, [0, 2]].T) > cfg.bias_free_m)
        rates = r4_rates(views, cams, free, cfg)
        rj["r4_rates"] = rates
        Al, Bl, rl, pl = posed["lead"]
        of = [i for i, p in enumerate(capf_p) if p in ("torso", "neck_head", "upper_arm_L", "upper_arm_R", "forearm_L", "forearm_R",
                                                        "shoulder_L", "shoulder_R")]
        ol = [i for i, p in enumerate(pl) if p in ("torso", "neck_head", "upper_arm_L", "forearm_L", "shoulder_L")]
        occ = (np.concatenate([capf_A[:, of], Al[:, ol]], 1), np.concatenate([capf_B[:, of], Bl[:, ol]], 1),
               np.concatenate([capf_r[of], rl[ol]]) * cfg.occl_r_scale)
        el_t = np.where((tri["lead"]["nv"][:, 8] >= 2)[:, None], tri["lead"]["Xd"][:, 8].astype(float), np.nan)
        elbow = np.where(np.isfinite(el_t), el_t, J55["lead"][:, 19])
        Rf55 = np.zeros((T, 55, 3, 3))
        Rf55[:, :22] = Rg["follow"]
        Rf55[:, 22:] = np.eye(3)
        vt = r4_visibility_test(frames, palm_b, elbow, sites["follow"], J55["follow"], Rf55, occ, obs, views, cams, rates, cfg)
        lv = vt["l"] - vt["l"].mean(1, keepdims=True)
        Lv[frames] = np.clip(lv, -cfg.clip_vis, cfg.clip_vis)
    Lcr = crop_terms([it for it in crop_items if it.get("relation") == "R4"], t, cfg, {"R4": band4}).get("R4", np.zeros((T, 3)))
    # calibrate the 3D and visibility cues on the agent's looks (Stage 2: on the recap both say "front" where the crops show his palm
    # on her side; their arg-max agrees with the looks on 31 % / 40 % of the labelled frames -> check only)
    cue_rep = r4_cue_reliability(r4_look_labels(crop_items, t, cfg), {"l3d": L3, "lvis": Lv}, cfg) if cfg.r4_cue_calibrate else {}
    L3w = calibrated_cue(L3, cue_rep.get("l3d"), cfg)
    Lvw = calibrated_cue(Lv, cue_rep.get("lvis"), cfg)
    # prior per couple-facing class (R2 margin, tri where known)
    m2 = np.where(np.isfinite(np.asarray(Mt["R2"]["m"], float)), np.asarray(Mt["R2"]["m"], float), np.asarray(Mb["R2"]["m"], float))
    fcls = np.where(m2 > 0.35, "face", np.where(m2 < -0.35, "away", "side"))
    if cfg.r4_blind_rule == "no_evidence":
        spread_v = Lvw.max(1) - Lvw.min(1)
        spread_3 = L3w.max(1) - L3w.min(1)
        blind = gate4 & (spread_v < cfg.blind_vis_nats) & (spread_3 < cfg.blind_3d_nats) & (np.abs(Lcr).max(1) < 1e-9)
    else:
        # no camera sees his right hand, and no camera evidence favours another class over the prior's own class
        Lcam = L3w + Lvw + Lcr
        pb = np.argmax(r4_prior(fcls, np.zeros(T, bool), cfg), 1)
        own = Lcam[np.arange(T), pb]
        Lo = Lcam.copy()
        Lo[np.arange(T), pb] = -np.inf
        against = Lo.max(1) - own >= cfg.blind_vis_nats
        blind = gate4 & hand_unseen(views, cams, cfg) & ~against
    Lp = r4_prior(fcls, blind, cfg)
    Lc = L3w + Lvw + Lcr + Lp
    d4 = hmm.four_state_r4(Lc, gate4, band4, cfg, cfg.hz)
    cls_b = r4_class_base(fbb, flb, cfg)
    cls_b[~gate4] = 0
    dep = inside_depth(caps["lead"], caps["follow"], posed["lead"], posed["follow"])
    rel.update({"R4__fb_base": fbb, "R4__fl_base": flb, "R4__fb_tri": fbt, "R4__fl_tri": flt, "R4__sig_fb": Sig["R4_fb"],
                "R4__sig_fl": Sig["R4_fl"], "R4__l3d": L3, "R4__lvis": Lv, "R4__l3d_cal": L3w, "R4__lvis_cal": Lvw, "R4__lprior": Lp, "R4__lcrop": Lcr, "R4__L": Lc,
                "R4__post": d4["post"].astype(np.float32), "R4__cls": d4["cls"], "R4__cls_base": cls_b, "R4__gate": gate4,
                "R4__band": band4, "R4__blind": blind, "R4__inside_base": dep, "R4__surf_base": surf_b, "R4__surf": surf})
    r4runs = [(a, b, v) for a, b, v in hmm.label_runs(d4["cls"]) if v != 0]
    rj["relations"]["R4"] = {"kind": "four_state", "classes": list(CLASSES), "half_width_m": hw, "defined_frames": int(gate4.sum()),
                             "decided_frames": int((d4["cls"] != 0).sum()), "blind_frames": int(blind.sum()),
                             "decided_by_prior_frames": int(((d4["cls"] != 0) & blind).sum()),
                             "cue_calibration": cue_rep, "blind_rule": cfg.r4_blind_rule,
                             "enforce": bool(crops.exists()),
                             "note": "R4 is enforced by the solver only after the crop check (crops.json); D1: blind_prior_nats in camera-blind frames",
                             "runs": [[round(float(t[a]), 3), round(float(t[b]), 3), CLASSES[v - 1], round(float(blind[a:b + 1].mean()), 2)]
                                      for a, b, v in r4runs]}
    log("R4", round(time.time() - t_start, 1), "s; decided", rj["relations"]["R4"]["decided_frames"], "blind", int(blind.sum()))
    # ---------------- R10 swivel (self)
    for r, P, X in (("lead", PLb, XL), ("follow", PFb, XF)):
        Pt = geom.points_from_coco(X)
        for s in ("L", "R"):
            phi, w = geom.swivel(P[f"sh{s}"], P[f"el{s}"], P[f"wr{s}"])
            phit, _ = geom.swivel(Pt[f"sh{s}"], Pt[f"el{s}"], Pt[f"wr{s}"])
            rel[f"R10_{r}_{s}__phi_base"], rel[f"R10_{r}_{s}__w_base"], rel[f"R10_{r}_{s}__phi_tri"] = phi, w, phit
    # ---------------- yaw
    yaw = {}
    yz = {"t": t}
    sj = {"what": "spin candidates (|psi(t+0.5 s) - psi(t-0.5 s)| > spin_cand_dps deg) and defects (|wrap(psi_base - psi*)| > 45 deg for >= 3 "
                  "frames, or >= 2 views (and half the labelled) contradict the base's face direction)", "roles": {}}
    fm = FilmMap.for_work(work)
    for r in cfg.roles:
        y = yaw_track(r, base[r], tri[r]["Xd"].astype(float), tri[r]["nv"], views, obs, cams, cfg)
        yaw[r] = y
        for k in ("psi_star", "conf", "psi_base", "err", "err_nohip", "psi_star_nohip", "turns", "face_contra", "n_bits", "spin_cand", "defect"):
            yz[f"{r}__{k}"] = y[k]
        cand_runs = hmm.runs(y["spin_cand"], merge=3, min_len=5)
        sj["roles"][r] = {"q": y["q"], "bias_hip_deg": y["bias_hip_deg"], "bias_shoulder_deg": y["bias_shoulder_deg"],
                          "candidates": [[round(float(t[a]), 2), round(float(t[b]), 2), mmss(fm.main(float(t[a])))] for a, b in cand_runs],
                          "defects": [{"t0": round(float(t[a]), 3), "t1": round(float(t[b]), 3), "film": mmss(fm.main(float(t[a]))),
                                       "err_max_deg": round(float(np.max(np.abs(y["err"][a:b + 1]))), 1),
                                       "face_contra_max": int(y["face_contra"][a:b + 1].max()),
                                       "turns_off": int(y["turns"][max(0, a - 30):b + 31][np.argmax(np.abs(y["turns"][max(0, a - 30):b + 31]))])}
                                      for a, b in y["runs"]]}
    log("yaw", round(time.time() - t_start, 1), "s")
    # ---------------- violations
    viol = violations(t, rel, base, tri, yaw, rest, cfg, fm)
    log("violations", len(viol), round(time.time() - t_start, 1), "s")
    np.savez_compressed(out_dir / "relations.npz", **{k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 and k != "t" else v)
                                                      for k, v in rel.items()})
    np.savez_compressed(out_dir / "yaw.npz", **yz)
    rj["seconds"] = round(time.time() - t_start, 1)
    rj["film_map"] = fm.source
    write_json(out_dir / "relations.json", rj)
    write_json(out_dir / "spins.json", sj)
    counts = {}
    for v in viol:
        counts[v["reason"]] = counts.get(v["reason"], 0) + 1
    write_json(out_dir / "violations.json", {"what": "where the base breaks a decided side or the real joint positions (no motion changed)",
                                             "take": take, "base": str(ad.base_dir), "counts": counts, "items": viol})
    log("done", rj["seconds"], "s", counts)
    return {"counts": counts, "seconds": rj["seconds"]}
