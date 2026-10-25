"""The windowed solve of the "sides" step (DESIGN 3.2-3.5): per window, Gaussian B-spline knot deltas left-multiplied onto the base's
local rotations of the moved joints (and the root, in spin windows) inside a tanh trust region, against the camera evidence and the
decided sides. No term pushes two bodies apart except a hand / forearm more than 3 cm inside her (E_inside); a relation on its decided
side or undecided costs nothing.

    solve_window(problem, cfg)          one window, one attempt -> {"P": {role: (W,55,3)}, "transl", "trust", "terms", ...}
    solve_all(take, part, n_parts)      every window of the part (attempts: as planned -> lambda_prior x 2 -> the single worst chain ->
                                        revert), each gated (sides.gate) -> work/<take>/sides/solve/part_<i>/win_<id>.npz + .json

Terms (weights SidesCfg.w_*; DESIGN 3.3): E_2D (MAMMA 512, Cauchy 10 px f/1000; COCO-17 + bias, 14 px f/1000, the held-out phone left out),
E_tri (Mahalanobis, pseudo-Huber at 3 sigma, the held-out triangulation), E_prior (|log R R_base^T|^2 / (4 + 16 q)^2), E_side (decided relations,
one-sided softplus), E_sep / E_gli (held arm pass-throughs, F2), E_inside (> 3 cm), E_contact (R4 flank / back: palm within 2 cm of her
torso), E_lim (wrist / elbow limits, the 3.4 twist hinges), E_acc, E_swivel, E_hinge, E_band, E_yaw / E_wind / E_face (spin windows),
E_floor (stance soles; whenever legs or root move). E_pen is off.

Schedule (3.5): Adam lr 0.02; stage 1 = 2D, tri, prior, limits, smoothness; stage 2 adds side, sep, gli, inside, contact, yaw, wind,
face, floor; stage 3 polish. Fixed seeds; float32 forward, float64 for G and d_s.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import body as bd
from . import geom
from .adapters import PIDX
from .config import SidesCfg, load_cfg

ROLES = ("lead", "follow")
ARM = {"L": (13, 16, 18, 20), "R": (14, 17, 19, 21)}
COCO_ARM = {"L": (5, 7, 9), "R": (6, 8, 10)}
# the COCO point a moved joint mostly moves (its "seen by q_j" point)
CHILD_COCO = {0: 11, 3: 5, 6: 5, 9: 5, 12: 0, 15: 0, 13: 5, 14: 6, 16: 7, 17: 8, 18: 9, 19: 10, 20: 9, 21: 10, 1: 13, 2: 14, 4: 15, 5: 16,
              7: 15, 8: 16, 10: 15, 11: 16}
LIMB_TWIST = {3: (3, 6), 6: (6, 9), 9: (9, 12), 12: (12, 15), 15: (12, 15), 1: (1, 4), 2: (2, 5), 4: (4, 7), 5: (5, 8), 7: (7, 10), 8: (8, 11)}
TWIST_HINGE = {3: 20.0, 6: 20.0, 9: 20.0, 12: 25.0, 15: 25.0, 1: 40.0, 2: 40.0, 4: None, 5: None, 7: 30.0, 8: 30.0}


def log(*a) -> None:
    print("[sides.solve]", *a, flush=True)


def mcp(side: str) -> tuple:
    from .. import hand_contact as hc
    return hc.mcp_joints(side)


def palm_t(J, side: str):
    i, m, r, p = mcp(side)
    w = 20 if side == "left" else 21
    return 0.4 * J[:, w] + 0.15 * (J[:, i] + J[:, m] + J[:, r] + J[:, p])


def points_t(J) -> dict:
    """geom's named points of one dancer from torch joints (W,55,3)"""
    P = {"shL": J[:, 16], "shR": J[:, 17], "elL": J[:, 18], "elR": J[:, 19], "wrL": J[:, 20], "wrR": J[:, 21], "hipL": J[:, 1], "hipR": J[:, 2],
         "knL": J[:, 4], "knR": J[:, 5], "anL": J[:, 7], "anR": J[:, 8], "head": J[:, 15], "pelvis": J[:, 0], "neck": J[:, 12],
         "ftL": J[:, 10], "ftR": J[:, 11]}
    P["palmL"], P["palmR"] = palm_t(J, "left"), palm_t(J, "right")
    return P


def fir_band(hz: float, lo: float, hi: float, taps: int = 31) -> np.ndarray:
    from scipy.signal import firwin
    return firwin(taps, [lo, hi], pass_zero=False, fs=hz)


def band_rms_np(y: np.ndarray, h: np.ndarray) -> float:
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < len(h):
        return float("nan")
    idx = np.flatnonzero(ok)
    yi = np.interp(np.arange(len(y)), idx, y[idx])
    f = np.convolve(yi - yi.mean(), h, mode="valid")
    return float(np.sqrt((f * f).mean())) if len(f) else float("nan")


# ------------------------------------------------------------------------------------------------ pure term helpers (tested alone)

def huber(torch, x):
    """x^2 up to 1, then 2x - 1 (x >= 0): a hinge that stops dominating everything once it is far from met"""
    return torch.where(x <= 1.0, x * x, 2.0 * x - 1.0)


def cauchy(torch, e2, s2):
    """log(1 + e^2 / s^2): a robust 2D cost whose pull decays as 1/e (Geman-McClure's 1/e^3 lets a limb drift away unopposed)"""
    return torch.log1p(e2 / s2)


def pseudo_huber(torch, d2):
    """sqrt(1 + d^2) - 1: quadratic near 0, linear far out (a joint far from its triangulated place keeps a constant pull)"""
    return torch.sqrt(1.0 + d2) - 1.0


def ramp_weights(W: int, ramp_in: int, ramp_out: int) -> np.ndarray:
    """(W,) 1 on the core, a raised cosine on each ramp (zero slope at both ends of each ramp: C1)"""
    w = np.ones(W)
    if ramp_in:
        w[:ramp_in] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(1, ramp_in + 1) / (ramp_in + 1)))
    if ramp_out:
        w[W - ramp_out:] = (0.5 - 0.5 * np.cos(np.pi * (np.arange(1, ramp_out + 1) / (ramp_out + 1))))[::-1]
    return w


def project_t(torch, it: dict, X):
    """(W,n,3) world points -> (W,n,2) pixels and (W,n) depth with the item's per-frame K, R, t"""
    Xc = torch.einsum("kij,knj->kni", it["R"], X) + it["t"][:, None]
    uv = torch.einsum("kij,knj->kni", it["K"], Xc)
    return uv[..., :2] / uv[..., 2:].clamp_min(1e-3), Xc[..., 2]


def g1_hinge(torch, e, eb, m, min_pts: int, min_share: float, med_px: float, p90_px: float, rel: float, margin_px: float):
    """sides.gate.g1_g10 per camera in torch: per frame the median error of the moved, visible points (frames with >= min_pts of
    them), then the median and p90 over frames, candidate e vs base eb (both (W,n) pixel errors, m the (W,n) mask) -> the squared
    excess over the gate's tolerance minus a margin (0 when the camera passes with the margin to spare)"""
    n = m.sum(1)
    fr = n >= max(1, min_pts)
    if float(fr.float().mean()) < min_share or int(fr.sum()) < 2:
        return None
    nan = torch.full_like(e, float("nan"))
    fc = torch.nanmedian(torch.where(m, e, nan)[fr], dim=1).values
    with torch.no_grad():
        fb = torch.nanmedian(torch.where(m, eb, nan)[fr], dim=1).values
        bm, bp = torch.quantile(fb, 0.5), torch.quantile(fb, 0.9)
    cm, cp = torch.quantile(fc, 0.5), torch.quantile(fc, 0.9)
    tm = torch.clamp(bm * rel, min=med_px) - margin_px
    tp = torch.clamp(bp * rel, min=p90_px) - margin_px
    return torch.relu(cm - bm - tm) ** 2 + torch.relu(cp - bp - tp) ** 2


def trust_delta(torch, v, rad):
    """the tanh trust region: v (..., nJ, 3) -> delta = r_j tanh(|v|) v / |v| (|delta| < r_j always) and tanh(|v|) (..., nJ)"""
    n = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-12)
    th = torch.tanh(n)
    return v * (rad[..., :, None] * th / n), th[..., 0]


def hold_terms(torch, A, B, s, d0: float = 0.01, dist_max: float = 0.06, gli_scale: float = 0.02):
    """E_gli and E_sep of one held arm pairing (DESIGN 3.3, F2), float64: A, B (W, 3, 3) the two 2-segment chains, s (W,) the decided
    GLI sign (0 = not held) -> (e_gli, e_sep). Both are exactly 0 on correctly linked or unlinked chains."""
    G = geom.gli_chain(A, B, xp=torch)
    held = (s != 0).to(A.dtype)
    eg = ((torch.relu(-s * G) / gli_scale) ** 2 * held).sum()
    es = torch.zeros((), dtype=A.dtype, device=A.device)
    for i in range(2):
        for j in range(2):
            ds, _, _, dist, inter = geom.signed_sep(A[:, i], A[:, i + 1], B[:, j], B[:, j + 1], xp=torch)
            act = held * (inter & (dist < dist_max)).to(A.dtype)
            es = es + ((torch.relu(d0 - s * ds) / d0) ** 2 * act).sum()
    return eg, es


def band_rms_sliding_np(y: np.ndarray, h: np.ndarray, win: int = 60) -> np.ndarray:
    """(n,) RMS of the band-passed signal in moving windows of `win` frames (the G5 measure; NaN gaps interpolated)"""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < len(h) + 2:
        return np.zeros(0)
    idx = np.flatnonzero(ok)
    yi = np.interp(np.arange(len(y)), idx, y[idx])
    f = np.convolve(yi - yi.mean(), h, mode="valid")
    w = min(win, len(f))
    return np.sqrt(np.convolve(f * f, np.ones(w) / w, mode="valid"))


def band_rms_sliding_t(torch, y, fir_flipped, win: int = 60):
    f = torch.nn.functional.conv1d((y - y.mean())[None, None], fir_flipped)[0, 0]
    w = min(win, f.shape[0])
    box = torch.full((1, 1, w), 1.0 / w, dtype=y.dtype, device=y.device)
    return torch.sqrt(torch.nn.functional.conv1d((f * f)[None, None], box)[0, 0] + 1e-12)


def band_windows_np(win: int = 60, hz: float = 30.0, lo: float = 1.0, hi: float = 5.0) -> dict:
    """the constants of report.band_rms_windows (detrend projector, Hann window, band mask) for a torch twin"""
    t = np.arange(win, dtype=float)
    A = np.stack([t, np.ones(win)], 1)
    Pd = np.eye(win) - A @ np.linalg.pinv(A)
    w = np.hanning(win)
    f = np.fft.rfftfreq(win, 1 / hz)
    return {"P": Pd, "w": w, "band": ((f >= lo) & (f <= hi)).astype(float), "norm": 2.0 / (win * (w ** 2).sum()), "win": win}


def band_rms_windows_t(torch, x, c: dict, step: int = 15):
    """report.band_rms_windows in torch: sliding Hann windows (detrended), RMS amplitude in lo..hi Hz -> (n_windows,)"""
    win = c["win"]
    if x.shape[0] < win:
        return x.new_zeros(0)
    seg = x.unfold(0, win, step)                                    # (n, win)
    seg = seg @ c["P"].T * c["w"]
    X = torch.fft.rfft(seg, dim=-1)
    pw = (X.real ** 2 + X.imag ** 2) * c["band"]
    return torch.sqrt(c["norm"] * pw.sum(-1) + 1e-12)


def band_rms_t(torch, y, fir_flipped):
    """RMS of the band-passed signal y (W,) (fixed FIR, valid part only) - the swing energy E_band caps"""
    f = torch.nn.functional.conv1d((y - y.mean())[None, None], fir_flipped)[0, 0]
    return torch.sqrt((f * f).mean() + 1e-12)


def knee_twist_limit(torch, Rl):
    """DESIGN 3.4: the knee's twist hinge 5 + 30 sin(flexion) deg (flexion = the knee's local rotation angle)"""
    flex = torch.acos(((Rl[:, 0, 0] + Rl[:, 1, 1] + Rl[:, 2, 2] - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))
    return 5.0 + 30.0 * torch.sin(flex)


def twist_excess(torch, Rl, axis, lim):
    """degrees of the swing-twist twist of local rotations Rl (W, 3, 3) about `axis` beyond `lim` (scalar or (W,))"""
    tw, _ = geom.swing_twist(Rl, axis, xp=torch)
    return torch.relu(tw.abs() - lim)


# ------------------------------------------------------------------------------------------------ the problem of one window

class Evidence:
    """the take-wide evidence the solve reads (loaded once per pod part)"""

    def __init__(self, take: str, cfg: SidesCfg):
        from .adapters import adapter_for
        from .evidence import capsule_set, load_tri, load_views, rest_joints, site_set
        from .prepare import load_cpu, load_fit
        self.take, self.cfg = take, cfg
        self.ad = ad = adapter_for(take, cfg)
        self.work = ad.work
        self.sd = ad.work / "sides"
        self.cams = ad.cams
        self.T = ad.T
        self.base = load_cpu(self.work)
        for r in ROLES:
            if "joints" not in self.base[r]:
                self.base[r]["joints"] = np.zeros((self.T, 55, 3))
                self.base[r]["joints"][:] = bd.forward(self.base[r], points=False)["J"]
        self.tri = {r: load_tri(self.work, r) for r in ROLES}
        self.tri_ho = {r: load_tri(self.work, r, heldout=True) for r in ROLES}
        self.views = load_views(self.work, self.cams)
        self.obs = ad.obs()
        self._lm = {}
        self.rel = dict(np.load(self.sd / "relations.npz"))
        self.yaw = dict(np.load(self.sd / "yaw.npz"))
        bj = json.loads((self.sd / "evidence" / "coco_bias.json").read_text(encoding="utf-8"))
        self.bias = {r: np.asarray(bj["bias"][r], float) for r in ROLES}
        betas = {r: np.asarray(self.base[r]["betas"], float) for r in ROLES}
        self.caps = capsule_set(betas, self.sd / "evidence" / "capsules.json")
        self.sites = site_set(betas, self.sd / "evidence" / "sites.json")
        self.rest = rest_joints(betas, self.sd / "evidence" / "rest.json")
        rj = json.loads((self.sd / "relations.json").read_text(encoding="utf-8"))
        self.half_width = float(rj["relations"]["R4"].get("half_width_m", 0.14))
        self.fir = fir_band(cfg.hz, *cfg.swing_band_hz)

    def lm(self, cam: str) -> np.ndarray:
        if cam not in self._lm:
            self._lm[cam] = self.ad.landmarks(cam)
        return self._lm[cam]


def sole_index(fit: dict) -> dict:
    from .. import physics_body as pb
    key = bd.betas_key(fit["betas"])
    cache = sole_index.__dict__.setdefault("cache", {})
    if key not in cache:
        body = pb.load_body(fit)
        out = {}
        for f in ("left", "right"):
            idx = body.sole_vertices(f, 0.02)
            out[f] = (body.weights[idx], body.v_rest[idx])
        cache[key] = out
    return cache[key]


class Problem:
    """one window: the base, the moved joints, the evidence as torch tensors on `device`"""

    def __init__(self, ev: Evidence, win: dict, device: str = "cpu", lam_prior: float = 1.0, only_joints: dict | None = None,
                 side_scale: float = 1.0, r4_scale: float = 1.0):
        import torch
        from .. import hand_contact as hc
        from .. import zouk_contacts as zc
        self.torch, self.dev = torch, device
        self.ev, self.win, self.cfg = ev, win, ev.cfg
        cfg = ev.cfg
        f32 = dict(dtype=torch.float32, device=device)
        self.f32 = f32
        self.ks = ks = np.arange(win["r0"], win["r1"] + 1)
        self.W = W = len(ks)
        self.Wc = Wc = win["k1"] - win["k0"] + 1
        self.ri = int(win["k0"] - win["r0"])
        hz = cfg.hz
        # ramp weights (1 on the core, a raised cosine on each ramp) and the core index each frame takes its delta from (the ramp is a
        # pure blend of the core edge's correction: hand_contact.Actor._edge)
        w = ramp_weights(W, win["ramp_in"], win["ramp_out"])
        self.w_np = w
        self.w = torch.tensor(w, **f32)
        if cfg.solve_free_ramps:
            # the knots span the ramps too: a ramp frame's correction is w(t) x its own value, so the blend can take a path around her arm
            # (E_add / E_smooth keep it from overshooting: hand_contact.Actor._edge's lesson)
            self.cidx = torch.arange(W, device=device)
            self.B = torch.tensor(hc.temporal_basis(W, hc.KNOT_STEP), **f32)
        else:
            self.cidx = torch.as_tensor(np.clip(np.arange(W) - self.ri, 0, Wc - 1), device=device)
            self.B = torch.tensor(hc.temporal_basis(Wc, hc.KNOT_STEP), **f32)
        nk = self.B.shape[1]
        self.dancers = {}
        self.bodies = {}
        moved = {r: d for r, d in win["dancers"].items() if (only_joints is None or r in only_joints)}
        for r in ROLES:
            fit = ev.base[r]
            P = bd.pose55(fit, ks)
            self.bodies[r] = bd.TorchBody(fit, device)
            d = {"Rl0": torch.tensor(bd.rotmat(P), **f32), "tr0": torch.tensor(np.asarray(fit["transl"], float)[ks], **f32),
                 "P0": P, "J0": np.asarray(fit["joints"], float)[ks]}
            if r in moved:
                md = moved[r]
                js = [int(j) for j in (only_joints[r] if only_joints is not None else md["joints"])]
                d["js"] = js
                d["rad"] = torch.tensor([math.radians(md["trust_deg"][str(j)]) for j in js], **f32)
                d["x"] = torch.zeros((nk, len(js), 3), **f32, requires_grad=True)
                d["root_m"] = float(md.get("root_m", 0.0)) if only_joints is None or 0 in js else 0.0
                d["y"] = torch.zeros((nk, 3), **f32, requires_grad=True) if d["root_m"] > 0 else None
                d["kind"] = md.get("kind", "arm")
                d["reasons"] = md.get("reasons", [])
            self.dancers[r] = d
        self.moved = [r for r in ROLES if "x" in self.dancers[r]]
        self.lam_prior = lam_prior
        self.side_scale = side_scale             # the fallbacks approach a decided side / class as far as the evidence allows
        self.r4_scale = r4_scale                 # an attempt's own R4 class / contact multiplier (x cfg.w_r4_mult; build C)
        self._prep_evidence()

    # ---------------------------------------------------------------- evidence tensors
    def _prep_evidence(self):
        torch, ev, cfg, ks, f32 = self.torch, self.ev, self.cfg, self.ks, self.f32
        W = self.W
        self.mamma, self.coco, self.trit, self.qj, self.yawf = {}, {}, {}, {}, {}
        for r in self.moved:
            d = self.dancers[r]
            p = PIDX[r]
            par = ev.base[r]
            m = bd.model(par)
            from .gate import descendants
            aff = descendants(m["parents"], d["js"])
            d["aff"] = aff
            pts = np.flatnonzero(m["Bw"][:, aff].sum(1) > 0.5)
            d["pts_idx"] = torch.as_tensor(pts, device=self.dev)
            items = []
            for c in ev.cams:
                o = ev.obs[c]
                lm = np.nan_to_num(ev.lm(c)[ks, p][:, pts].astype(float))
                vis = (o["vis"][ks, p][:, pts].astype(float) > 0.5) & (np.abs(lm).sum(-1) > 0) & o["ok"][ks][:, None]
                if vis.sum() < 20:
                    continue
                items.append({"cam": c, "lm": torch.tensor(lm, **f32), "vis": torch.tensor(vis.astype(np.float32), **f32),
                              "K": torch.tensor(o["K"][ks], **f32), "R": torch.tensor(o["R"][ks], **f32), "t": torch.tensor(o["t"][ks], **f32),
                              "s": torch.tensor(np.maximum(cfg.s_mamma_px * o["f"][ks], 1.0), **f32)})
            self.mamma[r] = items
            # COCO-17 (+ bias in the base's yaw frame), the held-out phone and detections dropped as hidden left out
            coco_aff = np.array([j for j in range(17) if aff[geom.C2S[j]] or (j < 5 and aff[geom.HEAD])], int)
            d["coco_aff"] = coco_aff
            from .evidence import yaw_axes
            Fr = yaw_axes(d["J0"])
            d["bias_w"] = torch.tensor(np.einsum("kij,pj->kpi", Fr, ev.bias[r])[:, coco_aff], **f32)     # (W,n,3)
            ho = ev.tri[r]["ho"][ks]
            drop = ev.tri[r]["dropped"][ks] if "dropped" in ev.tri[r] else np.zeros((W, 17, len(ev.cams)), bool)
            citems = []
            for ci, c in enumerate(ev.cams):
                vw = ev.views[c]
                o = ev.obs[c]
                uv = vw["uv"][ks, p][:, coco_aff].astype(float)
                cf = vw["conf"][ks, p][:, coco_aff].astype(float)
                okm = (cf >= cfg.kp_conf) & (vw["det"][ks, p].astype(float) >= cfg.det_conf)[:, None] & o["ok"][ks][:, None] \
                    & np.isfinite(uv[..., 0]) & ~drop[:, coco_aff, ci] & (ho != ci)[:, None]
                if okm.sum() < 5:
                    continue
                citems.append({"cam": c, "uv": torch.tensor(np.nan_to_num(uv), **f32), "m": torch.tensor(okm.astype(np.float32), **f32),
                               "K": torch.tensor(o["K"][ks], **f32), "R": torch.tensor(o["R"][ks], **f32), "t": torch.tensor(o["t"][ks], **f32),
                               "s": torch.tensor(np.maximum(cfg.s_coco_px * o["f"][ks], 1.0), **f32)})
            self.coco[r] = citems
            # triangulated targets (held-out variant), >= min_views_real views, debiased
            th = ev.tri_ho[r]
            X = th["Xd"][ks].astype(float)
            S = np.nan_to_num(th["S"][ks].astype(float))
            ok = (th["nv"][ks] >= cfg.min_views_real) & np.isfinite(X).all(-1)
            sel = [j for j in range(5, 17) if aff[geom.C2S[j]]]
            Si = np.zeros((W, len(sel), 3, 3))
            for i, j in enumerate(sel):
                good = ok[:, j]
                if good.any():
                    Si[good, i] = np.linalg.inv(S[good, j] + np.eye(3)[None] * 1e-8)
            okt = ok[:, sel].copy()
            # a decided R4 class (his right hand): drop the hand / wrist / distal-forearm evidence of the views the decided class hides the
            # hand from (the gate's rule, gate.r4_visibility, on the proxy placement of gate.r4_proxy_hidden): hallucinated detections of
            # the hidden hand must not pull it back to where the cameras' detector put it
            cls4 = np.asarray(ev.rel["R4__cls"], int)[ks]
            if cfg.solve_r4_vis and r == "lead" and (cls4 != 0).any():
                from .gate import r4_proxy_hidden, r_arm_groups, COCO_WR_R
                ph = r4_proxy_hidden(ev.base, ev.caps, ev.sites["follow"], ev.obs, ev.cams, cfg, ks, cls4)
                o0 = bd.forward(par, ks)
                hand_m, fore_m = r_arm_groups(m["Bw"], o0["pts"], o0["J"])
                hp, fp = hand_m[pts], fore_m[pts]
                nd = {"mamma_view_frames": 0, "coco_wrist_view_frames": 0, "tri_wrist_frames": 0,
                      "proxy_hidden_view_frames": int(ph["hand"].sum()), "decided_frames": int((cls4 != 0).sum())}
                for it in items:
                    ci = ev.cams.index(it["cam"])
                    drop = (ph["hand"][:, ci][:, None] & hp[None]) | (ph["forearm"][:, ci][:, None] & fp[None])
                    v0 = it["vis"].cpu().numpy() > 0.5
                    nd["mamma_view_frames"] += int((v0 & drop).any(1).sum())
                    it["vis"] = it["vis"] * torch.tensor((~drop).astype(np.float32), **f32)
                wc = np.flatnonzero(coco_aff == COCO_WR_R)
                for it in citems:
                    if not len(wc):
                        break
                    ci = ev.cams.index(it["cam"])
                    mm = it["m"].cpu().numpy().copy()
                    hid = ph["hand"][:, ci]
                    nd["coco_wrist_view_frames"] += int((mm[:, wc[0]] > 0.5)[hid].sum())
                    mm[hid, wc[0]] = 0.0
                    it["m"] = torch.tensor(mm, **f32)
                if COCO_WR_R in sel:
                    i10 = sel.index(COCO_WR_R)
                    bad = (np.asarray(th["inl"], bool)[ks][:, COCO_WR_R] & ph["hand"]).any(1)
                    nd["tri_wrist_frames"] = int((okt[:, i10] & bad).sum())
                    okt[:, i10] &= ~bad
                d["r4_vis"] = nd
            self.trit[r] = {"sel": sel, "X": torch.tensor(np.nan_to_num(X[:, sel]), **f32), "Si": torch.tensor(Si, **f32),
                            "m": torch.tensor(okt.astype(np.float32), **f32), "C2S": [geom.C2S[j] for j in sel]}
            # q_j: share of views that detect the joint's child COCO point
            nvw = np.zeros(17)
            for c in ev.cams:
                vw = ev.views[c]
                nvw += ((vw["conf"][ks, p].astype(float) >= cfg.kp_conf) & ev.obs[c]["ok"][ks][:, None]).mean(0)
            q = nvw / max(1, len(ev.cams))
            sig = np.array([cfg.prior_sigma0_deg + cfg.prior_sigma_q_deg * q[CHILD_COCO.get(j, 0)] for j in d["js"]])
            # never wider than half the joint's trust radius: a small-radius joint (collar 10, spine3 5 deg) must not sit on its bound
            rad_deg = np.degrees(d["rad"].cpu().numpy())
            sig = np.minimum(sig, cfg.prior_sigma_trust_frac * rad_deg)
            d["sig_p"] = torch.tensor(np.radians(sig), **f32)
        # decided relations on the window frames
        rel = ev.rel
        self.sides = []
        for rid, (kind, dz, dzf, _) in geom.REL_TABLE.items():
            s = rel[f"{rid}__s"][ks].astype(float)
            if not (s != 0).any():
                continue
            post = rel[f"{rid}__post"][ks].astype(float)
            c = np.where(s > 0, post[:, 2], np.where(s < 0, post[:, 1], 0.0))
            self.sides.append({"rid": rid, "kind": kind, "s": torch.tensor(s, **f32), "c": torch.tensor(c, **f32), "dzf": max(dzf, 1e-3)})
        cls = rel["R4__cls"][ks]
        self.r4 = {"cls": torch.as_tensor(cls.astype(np.int64), device=self.dev), "any": bool((cls != 0).any())}
        # R4 decided classes come from the crops / D1: enforced only once the crop check exists (decide's "enforce")
        rj = json.loads((ev.sd / "relations.json").read_text(encoding="utf-8"))
        self.r4["enforce"] = bool(rj["relations"]["R4"].get("enforce", False)) and self.r4["any"]
        # R9 holds: the decided sign; and where undecided but entangled, the base's own sign where the base has no pass-through nearby
        # (a weaker hold: the solve must not create a pass-through - G4 - while it moves an arm for another reason)
        self.holds = []
        JLb, JFb = self.dancers["lead"]["J0"], self.dancers["follow"]["J0"]
        for rid, (sa, sb) in geom.R9_PAIRS.items():
            s = rel[f"{rid}__s"][ks].astype(float)
            a_ = [17, 19, 21] if sa == "R" else [16, 18, 20]
            b_ = [17, 19, 21] if sb == "R" else [16, 18, 20]
            Gb = geom.gli_chain(JLb[:, a_], JFb[:, b_])
            ev_b = geom.passthrough_events(Gb, cfg.thru_jump)
            near = np.zeros(len(ks), bool)
            for e in ev_b:
                near[max(0, e - 6):e + 6] = True
            soft = np.where((s == 0) & (np.abs(Gb) > cfg.band_gli) & ~near, np.sign(Gb), 0.0)
            if (s != 0).any():
                self.holds.append({"rid": rid, "sa": sa, "sb": sb, "s": torch.tensor(s, dtype=torch.float64, device=self.dev), "w": 1.0})
            if (soft != 0).any() and "lead" in self.moved:
                self.holds.append({"rid": rid, "sa": sa, "sb": sb, "s": torch.tensor(soft, dtype=torch.float64, device=self.dev),
                                   "w": cfg.w_hold_base})
        # swivel targets (triangulated S, E, W with >= 3 views) and band references for the moved arms
        self.arms = []
        for r in self.moved:
            d = self.dancers[r]
            for side in ("L", "R"):
                if not any(j in d["js"] for j in ARM[side]):          # an arm that only rides along (spine3) is not re-shaped
                    continue
                tr = ev.tri[r]
                S_, E_, W_ = (tr["Xd"][ks, j].astype(float) for j in COCO_ARM[side])
                okt = (tr["nv"][ks][:, list(COCO_ARM[side])] >= 3).all(1)
                phi_t, _ = geom.swivel(S_, E_, W_)
                J0 = d["J0"]
                phi_b, _ = geom.swivel(J0[:, ARM[side][1]], J0[:, ARM[side][2]], J0[:, ARM[side][3]])
                phi_obs = np.where(okt & np.isfinite(phi_t), phi_t, phi_b)
                hS = J0[:, ARM[side][1], 1]
                band = {}
                for jn, jj, cj in (("el", ARM[side][2], COCO_ARM[side][1]), ("wr", ARM[side][3], COCO_ARM[side][2])):
                    from .report import band_rms_windows
                    rb = band_rms_windows(J0[:, jj, 1] - hS)
                    yt = np.where(tr["nv"][ks, cj] >= 2, tr["Xd"][ks, cj, 1].astype(float), np.nan) - hS
                    rt = band_rms_windows(yt)
                    band[jn] = (jj, max(float(np.nanmax(rb)) if np.isfinite(rb).any() else 0.0,
                                        float(np.nanmax(rt)) if np.isfinite(rt).any() else 0.0))
                    # the base just outside the window (the gate's band windows reach 15 frames past the ramps)
                    mg = 15
                    kl = np.arange(max(0, ks[0] - mg), ks[0])
                    kr = np.arange(ks[-1] + 1, min(ev.T, ks[-1] + 1 + mg))
                    Jfull = np.asarray(ev.base[r]["joints"], float)
                    band[jn + "_pad"] = (Jfull[kl, jj, 1] - Jfull[kl, ARM[side][1], 1], Jfull[kr, jj, 1] - Jfull[kr, ARM[side][1], 1])
                self.arms.append({"role": r, "side": side, "phi_obs": torch.tensor(np.radians(phi_obs), **f32),
                                  "hS": torch.tensor(hS, **f32), "band": band})
        self.fir_t = torch.tensor(ev.fir, **f32).flip(0)[None, None]
        bc = band_windows_np(int(round(cfg.swing_win_s * cfg.hz)), cfg.hz, *cfg.swing_band_hz)
        self.bandc = {k: (torch.tensor(v, **f32) if isinstance(v, np.ndarray) else v) for k, v in bc.items()}
        # spin windows: psi*, face bits
        self.spin = []
        for r in self.moved:
            d = self.dancers[r]
            if d["kind"] != "spin" and "spin" not in d["reasons"]:
                continue
            ps = np.radians(np.asarray(ev.yaw[f"{r}__psi_star"], float)[ks])
            conf = np.clip(np.asarray(ev.yaw[f"{r}__conf"], float)[ks], 0, 1)
            from .prepare import face_labels
            lab = face_labels(ev.views, ev.cams, r, ks)
            C = np.stack([ev.obs[c]["C"][ks] for c in ev.cams], 1)
            self.spin.append({"role": r, "psi": torch.tensor(ps, **f32), "conf": torch.tensor(conf, **f32),
                              "lab": torch.tensor(lab.astype(np.float32), **f32), "C": torch.tensor(C, **f32),
                              "stance": torch.tensor(np.asarray(ev.base[r]["stance"], bool)[ks].astype(np.float32), **f32)
                              if "stance" in ev.base[r] else None})
        # soles of dancers whose legs or root move
        self.floor = []
        for r in self.moved:
            d = self.dancers[r]
            legs = any(d["aff"][j] for j in (7, 8, 10, 11))
            if not legs:
                continue
            si = sole_index(ev.base[r])
            st = np.asarray(ev.base[r]["stance"], bool)[ks] if "stance" in ev.base[r] else np.zeros((W, 2), bool)
            sol = {}
            for i, f in enumerate(("left", "right")):
                Wv, xv = si[f]
                sol[f] = (torch.tensor(Wv, **f32), torch.tensor(xv, **f32), torch.tensor(st[:, i].astype(np.float32), **f32))
            self.floor.append({"role": r, "soles": sol})
        # capsules for inside / contact
        from .. import zouk_contacts as zc
        from .evidence import body_capsule_index
        self.capt = {}
        for r in ROLES:
            caps = ev.caps[r]
            idx = body_capsule_index(caps)
            sub = {"A": [caps["A"][i] for i in idx], "B": [caps["B"][i] for i in idx]}
            self.capt[r] = {"sub": sub, "r": torch.tensor(np.asarray(caps["r"], float)[idx], **f32), "parts": [caps["parts"][i] for i in idx]}
        self.i_arm = [i for i, p in enumerate(self.capt["lead"]["parts"]) if p in ("forearm_R", "hand_R")]
        self.i_torso = [i for i, p in enumerate(self.capt["follow"]["parts"]) if p == "torso"]
        self.i_torso_sh = [i for i, p in enumerate(self.capt["follow"]["parts"]) if p in ("torso", "shoulder_L", "shoulder_R")]
        # base references for E_acc / E_floor / E_yaw
        self.ref = {}
        with torch.no_grad():
            for r in ROLES:
                Rr, pr, Jr = self.bodies[r].fk(self.dancers[r]["Rl0"], self.dancers[r]["tr0"])
                self.ref[r] = {"J": Jr, "R": Rr, "p": pr}

    # ---------------------------------------------------------------- the pose of the current variables
    def set_base(self, o: dict) -> None:
        """the base's per-point pixel errors and positions (the zero-delta forward): E_g1 compares against them"""
        torch = self.torch
        self._b2d = {}
        for r in self.moved:
            d = self.dancers[r]
            X0 = o[r]["pts"].detach() if "pts" in o[r] else None
            C0 = (o[r]["J"][:, [geom.C2S[j] for j in d["coco_aff"]]] + d["bias_w"]).detach()
            em = []
            for it in self.mamma.get(r, []):
                uv, _ = project_t(torch, it, X0)
                em.append(((uv - it["lm"]) ** 2).sum(-1).clamp_min(0).sqrt())
            ec = []
            for it in self.coco.get(r, []):
                uv, _ = project_t(torch, it, C0)
                ec.append(((uv - it["uv"]) ** 2).sum(-1).clamp_min(0).sqrt())
            self._b2d[r] = {"X0": X0, "C0": C0, "em": em, "ec": ec}

    def deltas(self, r: str):
        torch = self.torch
        d = self.dancers[r]
        v = torch.einsum("wk,kjd->wjd", self.B, d["x"])[self.cidx]               # (W, nJ, 3)
        dl, th = trust_delta(torch, v, d["rad"])
        return dl * self.w[:, None, None], th

    def forward(self, need_points: bool = True) -> dict:
        torch = self.torch
        out = {}
        for r in ROLES:
            d = self.dancers[r]
            if "x" in d:
                dl, th = self.deltas(r)
                Rl = d["Rl0"].clone()
                js = torch.as_tensor(d["js"], device=self.dev)
                Rl[:, js] = bd.rodrigues_t(torch, dl) @ d["Rl0"][:, js]
                tr = d["tr0"]
                if d.get("y") is not None:
                    dt = d["root_m"] * torch.tanh(torch.einsum("wk,kd->wd", self.B, d["y"])[self.cidx]) * self.w[:, None]
                    tr = tr + dt
                R, p, J = self.bodies[r].fk(Rl, tr)
                o = {"R": R, "p": p, "J": J, "Rl": Rl, "delta": dl, "tanh": th, "transl": tr}
                if need_points and self.mamma.get(r):
                    o["pts"] = self.bodies[r].points(R, p, d["pts_idx"])
                out[r] = o
            else:
                ref = self.ref[r]
                out[r] = {"R": ref["R"], "p": ref["p"], "J": ref["J"], "Rl": d["Rl0"], "transl": d["tr0"]}
        return out

    # ---------------------------------------------------------------- terms
    def terms(self, o: dict, stage: int) -> dict:
        torch, cfg = self.torch, self.cfg
        hz = cfg.hz
        T = {}
        z = torch.zeros((), **self.f32)

        def gm(e2, s2):
            return e2 / (e2 + s2)
        for r in self.moved:
            d = self.dancers[r]
            J = o[r]["J"]
            # E_2D MAMMA
            e2d = z
            eg1 = z
            b2 = getattr(self, "_b2d", {}).get(r) if cfg.w_g1 > 0 else None
            g1a = (cfg.gate_g1_min_share, cfg.gate_g1_med_px, cfg.gate_g1_p90_px, cfg.gate_g1_rel, cfg.g1_margin_px)
            if b2 is not None and b2["X0"] is not None:
                mv = ((o[r]["pts"] - b2["X0"]).norm(dim=-1).amax(0) > cfg.gate_moved_m).detach()
            for ii, it in enumerate(self.mamma.get(r, [])):
                X = o[r]["pts"]
                uv, dep = project_t(torch, it, X)
                e2 = ((uv - it["lm"]) ** 2).sum(-1)
                rho = cauchy(torch, e2, it["s"][:, None] ** 2) * it["vis"]
                n = it["vis"].sum(1).clamp_min(1.0)
                e2d = e2d + (rho.sum(1) / n).sum()
                if b2 is not None and b2["X0"] is not None and bool(mv.any()):
                    h = g1_hinge(torch, (e2 + 1e-6).sqrt(), b2["em"][ii], (it["vis"] > 0.5) & mv[None] & (dep.detach() > 0.1),
                                 cfg.gate_g1_min_points, *g1a)
                    if h is not None:
                        eg1 = eg1 + h
            Cc = J[:, [geom.C2S[j] for j in d["coco_aff"]]] + d["bias_w"]
            if b2 is not None:
                mvc = ((Cc - b2["C0"]).norm(dim=-1).amax(0) > cfg.gate_moved_m).detach()
            for ii, it in enumerate(self.coco.get(r, [])):
                uv, dep = project_t(torch, it, Cc)
                e2 = ((uv - it["uv"]) ** 2).sum(-1)
                rho = cauchy(torch, e2, it["s"][:, None] ** 2) * it["m"]
                n = it["m"].sum(1).clamp_min(1.0)
                e2d = e2d + (rho.sum(1) / n).sum()
                if b2 is not None and bool(mvc.any()):
                    h = g1_hinge(torch, (e2 + 1e-6).sqrt(), b2["ec"][ii], (it["m"] > 0.5) & mvc[None] & (dep.detach() > 0.1), 1, *g1a)
                    if h is not None:
                        eg1 = eg1 + h
            T[f"2d_{r}"] = cfg.w_2d * e2d
            if b2 is not None:
                T[f"g1_{r}"] = cfg.w_g1 * eg1
            # E_tri (Mahalanobis, Geman-McClure on d_M^2 / 9)
            tt = self.trit[r]
            if tt["sel"]:
                dx = J[:, tt["C2S"]] - tt["X"]
                dm2 = torch.einsum("wni,wnij,wnj->wn", dx, tt["Si"], dx)
                T[f"tri_{r}"] = cfg.w_tri * (pseudo_huber(torch, dm2 / 9.0) * tt["m"]).sum()
            # E_prior
            T[f"prior_{r}"] = cfg.w_prior * self.lam_prior * ((o[r]["delta"] ** 2).sum(-1) / d["sig_p"][None] ** 2).sum()
            # temporal smoothness of the correction itself (no new wobble: the side / contact terms switch on and off between frames)
            dl = o[r]["delta"]
            if self.W > 2:
                d2d = dl[2:] - 2 * dl[1:-1] + dl[:-2]
                T[f"smooth_{r}"] = cfg.w_smooth * ((d2d ** 2).sum(-1) / math.radians(cfg.smooth_deg) ** 2).sum()
            # the trust region's edge: a barrier before tanh saturates (a window that needs more than its radius yields, it is not forced)
            T[f"trust_{r}"] = cfg.w_trust * ((torch.relu(o[r]["tanh"] - cfg.trust_soft) / (1.0 - cfg.trust_soft)) ** 2).sum()
            if d.get("y") is not None:
                dt = o[r]["transl"] - d["tr0"]
                T[f"prior_root_{r}"] = cfg.w_prior * self.lam_prior * ((dt / cfg.prior_root_m) ** 2).sum()
            # E_lim: wrists / elbows of the moved arms, twist hinges of the moved spine / legs
            T[f"lim_{r}"] = self._limits(r, o[r])
            # E_acc: elbows, wrists, knees (x 2 for the knees)
            ja = [j for j in (18, 19, 20, 21, 4, 5) if d["aff"][j]]
            if ja and self.W > 3:
                a1 = (J[2:, ja] - 2 * J[1:-1, ja] + J[:-2, ja]).norm(dim=-1) * hz * hz
                Jr = self.ref[r]["J"]
                a0 = (Jr[2:, ja] - 2 * Jr[1:-1, ja] + Jr[:-2, ja]).norm(dim=-1) * hz * hz
                wj = torch.tensor([2.0 if j in (4, 5) else 1.0 for j in ja], **self.f32)
                T[f"acc_{r}"] = cfg.w_acc * ((torch.relu(a1 - a0 - cfg.acc_slack_mps2) ** 2) * wj).sum() / 100.0
            # E_add: the correction's own speed (the G7 "jump": the hand / elbow / knee / ankle moving against the base faster than
            # add_mps) - it also sizes the edge correction to what the ramps can blend in
            jd = [j for j in (18, 19, 20, 21, 4, 5, 7, 8) if d["aff"][j]]
            if jd and self.W > 2:
                Jr = self.ref[r]["J"]
                c = J[:, jd] - Jr[:, jd]
                for side, wj in (("left", 20), ("right", 21)):
                    if d["aff"][wj]:
                        c = torch.cat([c, (palm_t(J, side) - palm_t(Jr, side))[:, None]], 1)
                v = (c[1:] - c[:-1]).norm(dim=-1) * hz
                T[f"add_{r}"] = cfg.w_add * ((torch.relu(v - cfg.add_mps) / 0.1) ** 2).sum()
        # arms: swivel, hinge plane, band
        for a in self.arms:
            r, side = a["role"], a["side"]
            J = o[r]["J"]
            S_, E_, W_ = J[:, ARM[side][1]], J[:, ARM[side][2]], J[:, ARM[side][3]]
            phi, wt = geom.swivel(S_, E_, W_, xp=torch)
            wt = wt * geom.swivel_valid(S_, W_, xp=torch).float()          # an arm overhead has no stable swivel reference
            phi = phi * (math.pi / 180.0)
            dphi = torch.atan2(torch.sin(phi[1:] - phi[:-1]), torch.cos(phi[1:] - phi[:-1]))
            if len(dphi) > 1:
                d2 = dphi[1:] - dphi[:-1]
                T[f"swivel_{r}{side}"] = cfg.w_swivel * ((wt[1:-1] * (d2 / math.radians(5.0)) ** 2).sum()
                                                         + (wt * (torch.atan2(torch.sin(phi - a["phi_obs"]), torch.cos(phi - a["phi_obs"]))
                                                                  / math.radians(20.0)) ** 2).sum())
            # the swivel's acceleration beyond the base's own (G5: p99 <= 1.2 x base)
            Jr = self.ref[r]["J"]
            phb, _ = geom.swivel(Jr[:, ARM[side][1]], Jr[:, ARM[side][2]], Jr[:, ARM[side][3]], xp=torch)
            phb = phb * (math.pi / 180.0)
            if self.W > 3:
                dpb = torch.atan2(torch.sin(phb[1:] - phb[:-1]), torch.cos(phb[1:] - phb[:-1]))
                a1, a0 = (dphi[1:] - dphi[:-1]).abs(), (dpb[1:] - dpb[:-1]).abs()
                ex = torch.relu(a1 - a0 - math.radians(cfg.swivel_acc_slack_deg)) / math.radians(cfg.swivel_acc_slack_deg)
                T[f"swacc_{r}{side}"] = cfg.w_swivel_acc * (wt[1:-1] * ex ** 2).sum()
                # jerk of the arm's correction (G5: joint jerk p99 <= 1.2 x base)
                cj = torch.stack([E_, W_], 1) - torch.stack([Jr[:, ARM[side][2]], Jr[:, ARM[side][3]]], 1)
                j3 = (cj[3:] - 3 * cj[2:-1] + 3 * cj[1:-2] - cj[:-3]).norm(dim=-1)
                T[f"jerk_{r}{side}"] = cfg.w_jerk * ((torch.relu(j3 - cfg.jerk_slack_m) / cfg.jerk_slack_m) ** 2).sum()
            n = torch.cross(E_ - S_, W_ - E_, dim=-1)
            n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            if self.W > 2:
                d2n = (n[2:] - 2 * n[1:-1] + n[:-2]).norm(dim=-1) ** 2
                T[f"hinge_{r}{side}"] = cfg.w_hinge * (wt[1:-1] * d2n).sum()
            for jn, val in a["band"].items():
                if jn.endswith("_pad"):
                    continue
                jj, ref = val
                y = J[:, jj, 1] - a["hS"]
                pl, pr = a["band"][jn + "_pad"]
                y = torch.cat([torch.tensor(pl, **self.f32), y, torch.tensor(pr, **self.f32)])
                if len(y) > self.fir_t.shape[-1]:
                    rms = band_rms_windows_t(torch, y, self.bandc)
                    T[f"band_{r}{side}{jn}"] = cfg.w_band * ((torch.relu(rms - ref - cfg.swing_slack_m) * 100.0) ** 2).sum()
        self._floor_terms(o, T)
        self._stage2_terms(o, T, stage)
        return T

    def _limits(self, r: str, o: dict):
        torch, cfg = self.torch, self.cfg
        from .. import zouk_contacts as zc
        d = self.dancers[r]
        Rl = o["Rl"]
        tot = torch.zeros((), **self.f32)
        deg = 180.0 / math.pi
        for side, x in (("left", "L"), ("right", "R")):
            # never beyond the base's own excess (G6 counts new frames only; the detected wrist / hand pose stays as detected)
            if d["aff"][ARM[x][3]] and any(j in d["js"] for j in ARM[x][1:]):
                L = zc.WRIST_LIMITS

                def wex(R):
                    b_, f_, _ = zc.wrist_terms_t(torch, R, side)
                    return torch.stack([b_ * deg - L["bend"], f_ * deg - L["flexion"], -f_ * deg - L["extension"]], -1)
                key = f"wrist_{x}"
                if key not in d.setdefault("lim_base", {}):
                    with torch.no_grad():
                        d["lim_base"][key] = torch.relu(wex(d["Rl0"][:, ARM[x][3]]))
                tot = tot + (torch.relu(wex(Rl[:, ARM[x][3]]) - d["lim_base"][key]) ** 2).sum()
            if any(j in d["js"] for j in ARM[x][:3]):
                E = zc.ELBOW_LIMITS

                def eex(R):
                    fx_, of_ = zc.elbow_terms_t(torch, R, side)
                    return torch.stack([fx_ * deg - E["flexion"], -fx_ * deg - E["hyperextension"], of_.abs() * deg - cfg.elbow_off_soft_deg,
                                        of_.abs() * deg - E["off_hinge"]], -1)
                key = f"elbow_{x}"
                if key not in d.setdefault("lim_base", {}):
                    with torch.no_grad():
                        d["lim_base"][key] = torch.relu(eex(d["Rl0"][:, ARM[x][2]]))
                ex = torch.relu(eex(Rl[:, ARM[x][2]]) - d["lim_base"][key]) ** 2
                wts = torch.tensor([1.0, 1.0, cfg.w_hinge_soft, 10.0], **self.f32)
                tot = tot + (ex * wts).sum()
        rest = self.ev.rest[r]
        # the 3.4 twist hinges: absolute in spin / limit windows (the over-twist is what they repair), elsewhere never beyond the
        # base's own twist (a real / arm window must not start re-twisting MAMMA's knees everywhere)
        absolute = d["kind"] == "spin" or bool({"spin", "limit"} & set(d["reasons"]))
        if "tw_base" not in d:
            d["tw_base"] = {}
        for j in d["js"]:
            if j not in LIMB_TWIST:
                continue
            a_, b_ = LIMB_TWIST[j]
            ax = rest[b_] - rest[a_]
            ax = ax / max(np.linalg.norm(ax), 1e-9)
            axt = torch.tensor(ax, **self.f32)
            if TWIST_HINGE.get(j) is None:                                  # knee: 5 + 30 sin(flex)
                lim = knee_twist_limit(torch, Rl[:, j])
            else:
                lim = torch.full((self.W,), TWIST_HINGE[j], **self.f32)
            if not absolute:
                if j not in d["tw_base"]:
                    with torch.no_grad():
                        d["tw_base"][j] = geom.swing_twist(d["Rl0"][:, j], axt, xp=torch)[0].abs()
                lim = torch.maximum(lim, d["tw_base"][j])
            tot = tot + (twist_excess(torch, Rl[:, j], axt, lim) ** 2).sum()
        return cfg.w_lim * tot

    def _stage2_terms(self, o: dict, T: dict, stage: int = 2):
        """couple terms: the holds (E_gli / E_sep) and E_inside from stage 1 on (a limb pushed through in stage 1 cannot come back: G is
        flat once through); the decided sides, contact, yaw / winding / face from stage 2"""
        torch, cfg = self.torch, self.cfg
        from .. import zouk_contacts as zc
        # a dancer that moves only to meet her / his own triangulated joints (`real` riders) is not pushed around by the couple terms:
        # its pose enters them detached (the gates still check every relation on the result)
        oc = {}
        for r in ROLES:
            d = self.dancers[r]
            rider = "x" in d and set(d.get("reasons", [])) <= {"real"}
            oc[r] = {k: (v.detach() if rider and torch.is_tensor(v) else v) for k, v in o[r].items()}
        o = oc
        PL, PF = points_t(o["lead"]["J"]), points_t(o["follow"]["J"])
        cross = any(r in self.moved for r in ROLES)
        r4s = (1.0 if cfg.r4_keep_scale else self.side_scale) * cfg.w_r4_mult * self.r4_scale      # the R4 class / contact pull (build C)
        if cross and (self.sides or self.r4["enforce"]) and stage >= 2:
            M = geom.margins(PL, PF, xp=torch, half_width_m=self.ev.half_width)
            tot = torch.zeros((), **self.f32)
            tot4 = torch.zeros((), **self.f32)
            for sd in self.sides:
                if sd["kind"] == "hold":
                    continue
                m = M[sd["rid"]]["m"]
                dzf = sd["dzf"]
                k = 1.0 / (0.3 * dzf)
                h = torch.nn.functional.softplus(k * (dzf - sd["s"] * m)) / k
                tot = tot + (sd["c"] * (sd["s"] != 0).float() * huber(torch, h / dzf)).sum()
            # R4 classes (front: fb >= +dz, back: fb <= -dz, flank: fl >= -0.5 cm and |fb| <= 12 cm)
            if self.r4["enforce"]:
                fb, fl = M["R4_fb"]["m"], M["R4_fl"]["m"]
                cls = self.r4["cls"]
                dz = cfg.r4_fb_dz_m / 3.0
                sp = lambda x: torch.nn.functional.softplus(x / (0.3 * dz)) * (0.3 * dz)
                h = torch.where(cls == 1, sp(dz - fb), torch.zeros_like(fb))
                h = h + torch.where(cls == 3, sp(dz + fb), torch.zeros_like(fb))
                h = h + torch.where(cls == 2, sp(-0.005 - fl) + sp(fb.abs() - 0.12), torch.zeros_like(fb))
                tot4 = tot4 + huber(torch, h / dz).sum()
            T["side"] = cfg.w_side * (self.side_scale * tot + r4s * tot4)
        # held arm pass-throughs: E_gli (chain) + E_sep (bone pairs with interior closest points)
        if self.holds and "lead" in self.moved:
            JL64, JF64 = o["lead"]["J"].double(), o["follow"]["J"].double()
            eg = torch.zeros((), dtype=torch.float64, device=self.dev)
            es = torch.zeros((), dtype=torch.float64, device=self.dev)
            for hd in self.holds:
                a = [17, 19, 21] if hd["sa"] == "R" else [16, 18, 20]
                b = [17, 19, 21] if hd["sb"] == "R" else [16, 18, 20]
                g_, s_ = hold_terms(torch, JL64[:, a], JF64[:, b], hd["s"], cfg.sep_d0_m, cfg.sep_dist_m)
                eg = eg + hd["w"] * g_
                es = es + hd["w"] * s_
            T["gli"] = cfg.w_gli * eg.float()
            T["sep"] = cfg.w_sep * es.float()
        # inside (his right forearm / hand capsules in her torso beyond 3 cm) and contact (R4 flank / back: palm within 2 cm of her)
        if cross:
            cl, cf = self.capt["lead"], self.capt["follow"]
            tipsL = torch.zeros((self.W, 10, 3), **self.f32)
            AL, BL = zc.capsule_points(cl["sub"], o["lead"]["J"], o["lead"]["R"], tipsL, xp=torch)
            AF, BF = zc.capsule_points(cf["sub"], o["follow"]["J"], o["follow"]["R"], tipsL, xp=torch)
            ia, ib = self.i_arm, self.i_torso
            dd, _, _ = zc.seg_seg(AL[:, ia][:, :, None], BL[:, ia][:, :, None], AF[:, ib][:, None], BF[:, ib][:, None], torch)
            depth = cl["r"][ia][None, :, None] + cf["r"][ib][None, None, :] - dd
            dmax = depth.reshape(self.W, -1).max(1).values                  # the deepest overlap per frame (not a sum over capsule pairs)
            T["inside"] = cfg.w_inside * huber(torch, torch.relu(dmax - cfg.r4_inside_m) / 0.01).sum()
            if self.r4["enforce"] and stage >= 2:
                palm = PL["palmR"]
                ic = self.i_torso_sh
                dp, _, _ = zc.seg_seg(palm[:, None], palm[:, None], AF[:, ic], BF[:, ic], torch)
                surf = (dp - cf["r"][ic][None]).min(1).values
                want = ((self.r4["cls"] == 2) | (self.r4["cls"] == 3)).float()
                T["contact"] = cfg.w_contact * r4s * (huber(torch, torch.relu(surf - cfg.contact_m) / 0.01) * want).sum()
        # spin windows: yaw to psi*, winding, face bits
        for sp_ in (self.spin if stage >= 2 else []):
            r = sp_["role"]
            J = o[r]["J"]
            Rg = o[r]["R"]
            hl = J[:, 1] - J[:, 2]
            psi = torch.atan2(-hl[:, 2], hl[:, 0])                         # heading of (L - R) x up: atan2(x, z) of (-dz, 0, dx)
            psi = torch.atan2(-hl[:, 2], hl[:, 0])
            fwd = torch.stack([-hl[:, 2], torch.zeros_like(hl[:, 0]), hl[:, 0]], -1)
            psi = torch.atan2(fwd[:, 0], fwd[:, 2])
            e = torch.atan2(torch.sin(psi - sp_["psi"]), torch.cos(psi - sp_["psi"]))
            T[f"yaw_{r}"] = cfg.w_yaw * (sp_["conf"] * (torch.relu(e.abs() - math.radians(15.0)) / math.radians(10.0)) ** 2).sum()
            sl = J[:, 16] - J[:, 17]
            fs = torch.stack([-sl[:, 2], torch.zeros_like(sl[:, 0]), sl[:, 0]], -1)
            psi_s = torch.atan2(fs[:, 0], fs[:, 2])
            rel_s = torch.atan2(torch.sin(psi_s - psi), torch.cos(psi_s - psi))
            n = int(round(0.5 * cfg.hz))
            wind = torch.zeros((), **self.f32)
            if self.W > n:
                dr = rel_s[n:] - rel_s[:-n]
                wind = wind + (torch.relu(dr.abs() - math.radians(70.0)) / math.radians(10.0)).pow(2).sum()
                if sp_["stance"] is not None:
                    for i, (a_, f_) in enumerate(((7, 10), (8, 11))):
                        fv = J[:, f_] - J[:, a_]
                        psi_f = torch.atan2(fv[:, 0], fv[:, 2])
                        rel_f = torch.atan2(torch.sin(psi_f - psi), torch.cos(psi_f - psi))
                        dfv = rel_f[n:] - rel_f[:-n]
                        stn = sp_["stance"][n:, i] * sp_["stance"][:-n, i]
                        wind = wind + (stn * (torch.relu(dfv.abs() - math.radians(60.0)) / math.radians(10.0)) ** 2).sum()
            T[f"wind_{r}"] = cfg.w_wind * wind
            # face: the head's forward against each view's face bit, a 10-deg margin around +-90
            hf = Rg[:, 15][:, :, 2]
            psi_h = torch.atan2(hf[:, 0], hf[:, 2])
            Hp = J[:, 15]
            az = torch.atan2(sp_["C"][..., 0] - Hp[:, None, 0], sp_["C"][..., 2] - Hp[:, None, 2])
            dlt = torch.cos(az - psi_h[:, None])                            # > 0 facing the camera
            m10 = math.sin(math.radians(10.0))
            lab = sp_["lab"]
            face = (lab > 0).float() * torch.relu(m10 - dlt) ** 2 + (lab < 0).float() * torch.relu(dlt + m10) ** 2
            T[f"face_{r}"] = cfg.w_face * face.sum() / (m10 ** 2)
    def _floor_terms(self, o: dict, T: dict):
        """stance soles keep the base's height and place (the foot may pivot); every stage, whenever legs or root move"""
        torch, cfg = self.torch, self.cfg
        for fl in self.floor:
            r = fl["role"]
            R, p = o[r]["R"], o[r]["p"]
            Rr, pr = self.ref[r]["R"], self.ref[r]["p"]
            tot = torch.zeros((), **self.f32)
            for f, (Wv, xv, st) in fl["soles"].items():
                def soles(RR, pp):
                    Rb = torch.einsum("vj,wjab->wvab", Wv, RR)
                    return torch.einsum("wvab,vb->wva", Rb, xv) + torch.einsum("vj,wjd->wvd", Wv, pp)
                s1, s0 = soles(R, p), soles(Rr, pr)
                y1, y0 = s1[..., 1].min(1).values, s0[..., 1].min(1).values
                c1, c0 = s1.mean(1), s0.mean(1)
                dh = (c1 - c0)[:, [0, 2]].norm(dim=-1)
                tot = tot + (st * (((y1 - y0) / 0.005) ** 2 + (dh / 0.005) ** 2)).sum()
            T[f"floor_{r}"] = cfg.w_floor * tot

    # ---------------------------------------------------------------- the solve
    def params(self):
        out = []
        for r in self.moved:
            d = self.dancers[r]
            out.append(d["x"])
            if d.get("y") is not None:
                out.append(d["y"])
        return out


def solve_window(prob: Problem, seed: int = 0) -> dict:
    """the schedule of DESIGN 3.5 on one Problem -> {"P": {role: (W,55,3)}, "transl": {role: (W,3)}, "trust": {...}, "terms": {stage: {...}}}"""
    torch = prob.torch
    cfg = prob.cfg
    torch.manual_seed(seed)
    opt = torch.optim.Adam(prob.params(), lr=cfg.solve_lr)
    hist = {}
    t0 = time.time()
    with torch.no_grad():
        o = prob.forward()
        prob.set_base(o)
        T0 = {k: float(v) for k, v in prob.terms(o, 2).items()}
    hist["start"] = T0
    for stage, iters in ((1, cfg.solve_iters1), (2, cfg.solve_iters2), (3, cfg.solve_iters3)):
        st = 2 if stage >= 2 else 1
        for it in range(iters):
            opt.zero_grad()
            o = prob.forward()
            T = prob.terms(o, st)
            loss = sum(T.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"window {prob.win.get('id')}: non-finite loss at stage {stage} iteration {it}: "
                                         + json.dumps({k: float(v) for k, v in T.items()}))
            loss.backward()
            opt.step()
        with torch.no_grad():
            o = prob.forward()
            hist[f"stage{stage}"] = {k: float(v) for k, v in prob.terms(o, 2).items()}
    with torch.no_grad():
        o = prob.forward(need_points=False)
    out = {"P": {}, "transl": {}, "trust": {}, "terms": hist, "seconds": round(time.time() - t0, 2)}
    hit = np.zeros(prob.W, bool)
    for r in prob.moved:
        Rl = o[r]["Rl"].detach().cpu().numpy().astype(np.float64)
        P = prob.dancers[r]["P0"].copy()
        js = prob.dancers[r]["js"]
        P[:, js] = bd.rotvec(Rl[:, js])
        same = prob.w_np <= 0
        P[same] = prob.dancers[r]["P0"][same]
        out["P"][r] = P
        out["transl"][r] = o[r]["transl"].detach().cpu().numpy().astype(np.float64)
        th = o[r]["tanh"].detach().cpu().numpy()
        h = (th >= cfg.gate_trust_frac).any(1) & (prob.w_np > 0)
        hit |= h
        out["trust"][r] = {"max_tanh": round(float(th.max()), 3), "frames": int(h.sum())}
    out["trust"]["share"] = round(float(hit.mean()), 3)
    return out


# ------------------------------------------------------------------------------------------------ attempts, gate, outputs

def candidate_fits(ev: Evidence, res: dict, win: dict) -> dict:
    from .gate import fit_window
    ks = np.arange(win["r0"], win["r1"] + 1)
    out = {}
    for r in ROLES:
        if r in res["P"]:
            out[r] = fit_window(ev.base[r], res["P"][r], res["transl"][r], ks)
        else:
            out[r] = ev.base[r]
    return out


def worst_chain(win: dict, gate_res: dict) -> dict:
    """the single chain kept by the second fallback: the lead's right arm when his arm moved (the owner's case), else the first moved
    dancer's joints of its first reason"""
    d = win["dancers"]
    if "lead" in d and any(j in d["lead"]["joints"] for j in (14, 17, 19, 21)):
        return {"lead": [j for j in d["lead"]["joints"] if j in (9, 14, 17, 19, 21)]}
    r = next(iter(d))
    return {r: d[r]["joints"]}


def solve_and_gate(ev: Evidence, gctx, win: dict, device: str, seed: int = 0) -> dict:
    """attempts (cfg.solve_attempts; DESIGN 3.5: as planned -> lambda_prior x 2 -> the single worst chain, here the fallbacks also
    soften the decided-side / contact pull): the first that passes every gate is the window's result"""
    from .gate import window_gates, inside_series
    from .. import zouk_contacts as zc
    attempts = []
    chosen = None
    plans = [(a["name"], float(a.get("lam_prior", 1.0)), "chain" if a.get("chain") else None, float(a.get("side_scale", 1.0)),
              float(a.get("r4", 1.0))) for a in ev.cfg.solve_attempts]
    ks = np.arange(win["r0"], win["r1"] + 1)
    ex_base = None
    for name, lam, cut, ssc, r4m in plans:
        only = worst_chain(win, attempts[-1]["gate"] if attempts else {}) if cut else None
        prob = Problem(ev, win, device, lam_prior=lam, only_joints=only, side_scale=ssc, r4_scale=r4m)
        if not prob.moved:
            continue
        res = solve_window(prob, seed)
        cand = candidate_fits(ev, res, win)
        if ex_base is None:
            ex_base = zc.exact_of({r: gctx.base[r] for r in ROLES}, ks, device, 48)
        ex_c = zc.exact_of({r: cand[r] for r in ROLES}, ks, device, 48)
        trust = {"share": res["trust"]["share"], "per_role": {r: v for r, v in res["trust"].items() if r != "share"}}
        g = window_gates(gctx, cand, win if only is None else _cut_win(win, only), exact={"base": ex_base, "cand": ex_c}, trust=trust)
        attempts.append({"name": name, "lam_prior": lam, "side_scale": ssc, "r4": r4m, "only": only, "pass": g["pass"], "failed": g["failed"],
                         "targets_missed": g["targets_missed"], "gate": g, "terms": res["terms"], "seconds": res["seconds"],
                         "r4_vis": prob.dancers.get("lead", {}).get("r4_vis"),
                         "trust": res["trust"], "_res": res, "_exact": ex_c})
        log(f"window {win['id']} {win['t0']}-{win['t1']} s {name}: {'PASS' if g['pass'] else 'fail ' + ','.join(g['failed'])} "
            f"({res['seconds']} s)")
        if g["pass"]:
            chosen = attempts[-1]
            break
    return {"attempts": attempts, "chosen": chosen, "exact_base": ex_base}


def _cut_win(win: dict, only: dict) -> dict:
    w = json.loads(json.dumps(win))
    w["dancers"] = {r: dict(d, joints=list(only[r]), trust_deg={str(j): d["trust_deg"][str(j)] for j in only[r]})
                    for r, d in win["dancers"].items() if r in only}
    return w


def write_window(out_dir: Path, win: dict, sg: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = sg["chosen"]
    z = {"ks": np.arange(win["r0"], win["r1"] + 1), "w": np.ones(1)}
    rep = {"id": win["id"], "t0": win["t0"], "t1": win["t1"], "status": "adopt" if ch else "revert",
           "attempts": [{k: v for k, v in a.items() if not k.startswith("_")} for a in sg["attempts"]]}
    for i, a in enumerate(sg["attempts"]):
        for r, P in a["_res"]["P"].items():
            z[f"a{i}_{r}_pose"] = P.astype(np.float64)
            z[f"a{i}_{r}_transl"] = a["_res"]["transl"][r].astype(np.float64)
        for k, v in a["_exact"].items():
            z[f"a{i}_exact_{k}"] = np.asarray(v)
    if sg["exact_base"] is not None:
        for k, v in sg["exact_base"].items():
            z[f"base_exact_{k}"] = np.asarray(v)
    if ch is not None:
        i = sg["attempts"].index(ch)
        rep["chosen"] = i
        z["chosen"] = np.array(i)
    np.savez_compressed(out_dir / f"win_{win['id']:03d}.npz", **z)
    from .evidence import write_json
    write_json(out_dir / f"win_{win['id']:03d}.json", rep)
    return rep


def _solve_ids(take: str, ids: list, device: str, cfg: SidesCfg, od: Path, tag: str, threads: int | None = None) -> dict:
    """one worker: Evidence + GateCtx once, then each window id in turn -> its summary (written to od/summary_<tag>.json as it goes)"""
    t0 = time.time()
    if threads:
        import torch
        torch.set_num_threads(int(threads))
    from .evidence import write_json
    from .gate import GateCtx
    from .plan import load_plan
    ev = Evidence(take, cfg)
    wins = {w["id"]: w for w in load_plan(ev.work)["windows"]}
    gctx = GateCtx(take, cfg, base=ev.base)
    gctx._lm = ev._lm
    summary = {"windows": {}, "tag": tag}
    log(f"{take} [{tag}]: {len(ids)} windows on {device} (load {round(time.time() - t0, 1)} s)")
    for i in ids:
        w = wins[i]
        try:
            sg = solve_and_gate(ev, gctx, w, device, seed=cfg.solve_seed + i)
            rep = write_window(od, w, sg)
            summary["windows"][str(i)] = {"status": rep["status"], "attempts": [(a["name"], a["pass"], a["failed"]) for a in rep["attempts"]]}
        except Exception as e:  # noqa: BLE001 - one window's failure must not stop the part
            import traceback
            summary["windows"][str(i)] = {"status": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
            log(f"window {i}: ERROR {type(e).__name__}: {e}")
        summary["seconds"] = round(time.time() - t0, 1)
        write_json(od / f"summary_{tag}.json", summary)
    return summary


def _worker(args):
    take, ids, device, cfg_json, od, tag, threads = args
    from .config import apply_overrides, SidesCfg
    cfg = apply_overrides(SidesCfg(), {k: v for k, v in cfg_json.items()})
    return _solve_ids(take, ids, device, cfg, Path(od), tag, threads)


def solve_all(take: str, part: int = 0, n_parts: int = 1, device: str = "cuda", cfg: SidesCfg | None = None, only: list | None = None,
              out_dir: Path | None = None, workers: int | None = None) -> dict:
    """the windows of one part (plan.parts_of), the owner's first; `workers` processes share the device (the solve is launch-bound:
    a 24-GB GPU runs several windows at once) -> summary.json"""
    t0 = time.time()
    cfg = cfg or load_cfg(take)
    from .adapters import work_dir
    from .evidence import write_json
    from .plan import load_plan, parts_of
    work = work_dir(take)
    plan = load_plan(work)
    ids = parts_of(plan, n_parts)[part] if only is None else list(only)
    wins = {w["id"]: w for w in plan["windows"]}
    ids = sorted(ids, key=lambda i: (wins[i].get("tier", 2), i))          # the owner's windows first (a part cut short keeps them)
    od = Path(out_dir or work / "sides" / "solve" / f"part_{part}")
    od.mkdir(parents=True, exist_ok=True)
    nw = max(1, min(int(workers or cfg.solve_workers), len(ids)))
    summary = {"take": take, "part": part, "n_parts": n_parts, "device": device, "workers": nw, "windows": {},
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if nw == 1:
        res = [_solve_ids(take, ids, device, cfg, od, "w0")]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        subs = [ids[k::nw] for k in range(nw)]
        cj = cfg.to_json()
        args = [(take, sub, device, cj, str(od), f"w{k}", cfg.solve_threads) for k, sub in enumerate(subs)]
        log(f"{take}: part {part}/{n_parts}: {len(ids)} windows on {device} in {nw} worker processes")
        with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_worker, args))
    for r in res:
        summary["windows"].update(r["windows"])
    summary["seconds"] = round(time.time() - t0, 1)
    write_json(od / "summary.json", summary)
    log("part done", summary["seconds"], "s")
    return summary
