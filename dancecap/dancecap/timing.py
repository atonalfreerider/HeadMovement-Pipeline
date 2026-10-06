"""Timing: kinematic derivatives, footstep touchdowns, joint accents and their asynchrony to the zouk beat grid.

Inputs (work/<take>/): smplx/{lead,follow}.npz (`t`, `joints` (T,55,3) Y-up metres, floor y=0) and a beats json
in the zouk format ([[seconds, type]], 8 per measure, type 1 = eighths 0/4, 2 = eighths 3/6, 3 = other, exactly as
atlas_bridge/beats.py and beat_this_analyzer.py write it). Output: timing/timing.json (+ kinematics.npz, qa png).

Method (see `notes` in timing.json):
  * derivatives on the REAL timestamps with a quintic smoothing spline per coordinate, s = N * sigma^2 from a
    noise model (sigma estimated from 4th differences unless given); Butterworth filtfilt + central differences
    is computed as a cross-check only. No forward differences (they shift jerk ~1.5 frames early).
  * footstep touchdowns from a CONTACT MODEL per foot (ankle = heel proxy + foot/toe joint): stance height,
    vertical + horizontal speed with hysteresis. Sub-frame time: a hinge (stance plateau + polynomial descent;
    the hinge is where the height reaches stance and the vertical velocity reaches zero) on each step, refined
    by aligning all steps of a foot to their leave-one-out average (matched filter over every leg channel) and
    placing the touchdown at the hinge of the pooled, aligned descents. Sigma per event = parametric
    Monte-Carlo SD with the noise model (re-running the whole detector).
  * key-joint accents: local maxima of |acceleration| and |jerk| (spline, 4x oversampled) with prominence.
    Jerk peaks are less reliable than touchdowns at 30 fps; they are reported separately.
  * each event is assigned to the nearest beat of the allowed zouk counts; asynchrony = event - beat, with
    circular stats, Monte-Carlo CIs (pose noise + beat jitter + camera sync), a circular-shift surrogate for
    chance hit rate, and a permutation test for measures that deviate (e.g. a dancer stepping late).

    python -m dancecap timing synthetic                    -> work/synthetic/timing/timing.json
    python -m dancecap timing synthetic --validate         -> + validation.json against truth.json (gate G5)
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

ROLES = ("lead", "follow")
# SMPL-X body joints (first 22 of the 55 used by the contract)
JOINT_NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
               "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
               "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
FOOT_JOINTS = {"left": {"ankle": 7, "toe": 10}, "right": {"ankle": 8, "toe": 11}}
ACCENT_JOINTS = {"head": 15, "left_wrist": 20, "right_wrist": 21, "pelvis": 0, "left_foot": 10, "right_foot": 11}
MEASURE_LEN = 8  # zouk grid entries per measure
# hinge polynomials (powers of u = tau - t) for the approach onto the stance plateau: with a linear term = impact
# landing, without = soft landing; the cubic absorbs the swing curvature
HINGE_MODELS = ((0, 1, 2, 3), (0, 1, 3), (0, 2, 3), (0, 1, 2), (0, 1), (0, 2))


@dataclass
class TimingConfig:
    # noise model / differentiation
    noise_m: float | None = None  # joint position noise (m); None = robust estimate from 4th differences
    noise_floor_m: float = 0.0005
    spline_k: int = 5
    butter_cutoff_hz: float = 6.0
    butter_order: int = 4
    # contact model
    # contact/release heights grow by noise_k * sigma but are capped at 35% / 60% of the foot's clearance (p95)
    contact_height_m: float = 0.025  # foot part this close to its stance height may be in contact
    release_height_m: float = 0.045  # leave contact above this
    contact_vy: float = 0.25  # m/s |vertical speed| to enter contact
    contact_vxz: float = 1.0  # m/s horizontal speed to enter contact (stance feet pivot and slide in zouk)
    release_vxz: float = 1.6
    min_stance_s: float = 0.06
    merge_gap_s: float = 0.08  # contact flickers shorter than this are merged
    min_apex_m: float = 0.015  # swing clearance (and > release height) needed before a touchdown counts
    fit_upper_frac: float = 0.5  # hinge fit uses the descent below this fraction of the swing apex
    fit_after_s: float = 0.2
    touchdown_joint: str = "auto"  # auto: ankle (heel/foot-flat) unless the heel stays up, then toe
    noise_k: float = 1.0  # thresholds grow by noise_k * sigma (the spline is much smoother than the raw noise)
    template: bool = True  # refine with the per-foot step-averaged descent template (matched filter)
    template_pre_s: float = 0.3
    template_post_s: float = 0.15
    template_search_s: tuple = (0.08, 0.04, 0.02)
    template_min_steps: int = 4
    template_knot_s: float = 0.025
    template_scale_sd: float = 0.15  # prior SD of a step's descent amplitude relative to the template
    template_fit_frac: float = 0.7  # template hinge uses the descent below this fraction of the apex
    template_hinge_models: tuple = HINGE_MODELS
    # accents
    accent_joints: dict = field(default_factory=lambda: dict(ACCENT_JOINTS))
    accent_oversample: int = 4
    accent_min_sep_s: float = 0.15
    accent_prominence: float = 0.5  # x (p90 - p10) of the magnitude signal
    # beat grid
    allowed_types: tuple = (1, 2, 3)
    allowed_positions: tuple | None = None  # eighth positions 0..7 inside the measure; None = all
    refine_beats: bool = True
    coarsen_grid: int = 1  # 2 = undo beat_this split_measures halving (keep every other grid entry)
    hit_tol_s: float = 0.05
    dense_chance_warn: float = 0.35
    av_offset_s: float = 0.0  # added to event (video) times to put them on the audio clock
    av_sync_event: str | None = None  # description of the A/V sync event, if one was used
    # uncertainty
    mc_draws: int = 200
    workers: int = 4  # processes for the Monte-Carlo
    beat_jitter_s: float | None = None  # None = max(beat refinement residual RMS, 10 ms)
    sync_sigma_s: float = 0.005  # per-camera residual sync uncertainty
    n_cams: int = 8
    n_surrogates: int = 2000
    n_perm: int = 5000
    match_tol_s: float = 0.08
    seed: int = 0


# ----------------------------------------------------------------------------------------------- derivatives

def estimate_noise(x: np.ndarray) -> float:
    """Robust white-noise sigma (m) of time series x (T, ...): MAD of 4th differences / sqrt(70), median over series.

    The 4th difference of white noise has variance 70 sigma^2 while smooth motion contributes ~A (2 pi f dt)^4;
    impacts are sparse and the MAD ignores them."""
    d = np.diff(x.reshape(len(x), -1), 4, axis=0)
    mad = np.median(np.abs(d - np.median(d, 0)), 0) * 1.4826
    return float(np.median(mad) / math.sqrt(70.0))


@dataclass
class Kinematics:
    """pos/vel/acc/jerk (Te, J, 3) evaluated at times te; NaN for joints that were not fitted."""
    te: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    jerk: np.ndarray
    method: str


def _check_times(t: np.ndarray) -> None:
    if len(t) < 8 or not np.all(np.diff(t) > 0):
        raise ValueError("timestamps must be strictly increasing with >= 8 samples")


def spline_kinematics(t: np.ndarray, joints: np.ndarray, noise_m: float, idx=None, te=None, k: int = 5):
    """Quintic smoothing spline per coordinate on the real timestamps (s = N sigma^2), analytic derivatives.

    te: evaluation times (default t), or a tuple of arrays -> tuple of Kinematics from one fit."""
    from scipy.interpolate import UnivariateSpline
    _check_times(t)
    idx = range(joints.shape[1]) if idx is None else sorted(set(idx))
    tes = te if isinstance(te, tuple) else (t if te is None else te,)
    outs = [np.full((4, len(x)) + joints.shape[1:], np.nan) for x in tes]
    s = len(t) * noise_m ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in idx:
            for c in range(joints.shape[2]):
                spl = UnivariateSpline(t, joints[:, j, c], k=k, s=s)
                for x, out in zip(tes, outs):
                    for nu in range(4):
                        out[nu, :, j, c] = spl(x, nu)
    method = f"quintic smoothing spline k={k} s=N*({noise_m * 1000:.2f} mm)^2"
    res = tuple(Kinematics(x, *out, method=method) for x, out in zip(tes, outs))
    return res if isinstance(te, tuple) else res[0]


def butter_kinematics(t: np.ndarray, joints: np.ndarray, cutoff_hz: float = 6.0, order: int = 4, idx=None,
                      te=None) -> Kinematics:
    """Cross-check: resample to the median frame interval, zero-phase Butterworth (filtfilt), central differences."""
    from scipy.signal import butter, filtfilt
    _check_times(t)
    idx = list(range(joints.shape[1])) if idx is None else sorted(set(idx))
    te = t if te is None else te
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1] + 1e-9, dt)
    b, a = butter(order, cutoff_hz / (0.5 / dt))
    out = np.full((4, len(te)) + joints.shape[1:], np.nan)
    for j in idx:
        for c in range(joints.shape[2]):
            y = filtfilt(b, a, np.interp(tu, t, joints[:, j, c]), padtype="odd", padlen=min(3 * max(len(a), len(b)),
                                                                                             len(tu) - 1))
            d = [y]
            for _ in range(3):
                d.append(np.gradient(d[-1], dt))
            for nu in range(4):
                out[nu, :, j, c] = np.interp(te, tu, d[nu])
    return Kinematics(te, *out, method=f"butterworth order {order} {cutoff_hz} Hz filtfilt + central differences")


# ----------------------------------------------------------------------------------------------- contact model

def _hysteresis(H, vy, vxz, h_on, h_off, cfg: TimingConfig, dt: float) -> list[tuple[int, int]]:
    """Stance intervals [start, end) from foot height/speeds with enter/leave thresholds."""
    spans, start = [], None
    for i in range(len(H)):
        if start is None:
            if H[i] < h_on and abs(vy[i]) < cfg.contact_vy and vxz[i] < cfg.contact_vxz:
                start = i
        elif H[i] > h_off or vxz[i] > cfg.release_vxz:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(H)))
    merged = []
    for s, e in spans:
        if merged and (s - merged[-1][1]) * dt < cfg.merge_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if (e - s) * dt >= cfg.min_stance_s - 1e-9]


def hinge_touchdown(t: np.ndarray, y: np.ndarray, t_lo: float, t_hi: float, step: float = 0.001,
                    descent: bool = True, models=HINGE_MODELS) -> tuple[float, float]:
    """Fit y = a + b u + c u^2 + d u^3 with u = max(0, tau - t): a polynomial approach onto a plateau -> (tau, rms).

    tau is where the approach ends: the height reaches the stance plateau and the vertical velocity reaches zero
    (b = 0, soft landing) or jumps to zero (b != 0, impact); the cubic absorbs the swing's curvature change.
    y may be (n,) or (n, P): P channels share tau, each picks its best polynomial. descent=True requires b, c >= 0
    (coming down onto the plateau). Exhaustive `step` search + parabolic refinement of the SSE minimum."""
    y = y.reshape(len(t), -1)
    taus = np.arange(t_lo, t_hi + step / 2, step)
    if len(taus) < 3 or len(t) < 5:
        return float("nan"), float("nan")
    u = np.maximum(0.0, taus[:, None] - t[None, :])  # (Ntau, n)
    powers = np.stack([np.ones_like(u), u, u * u, u ** 3], -1)
    total = np.zeros(len(taus))
    for p in range(y.shape[1]):
        best = np.full(len(taus), np.inf)
        for cols in models:
            if len(cols) >= len(t):
                continue
            A = powers[..., cols]  # (Ntau, n, q)
            AtA = np.einsum("knp,knq->kpq", A, A) + 1e-12 * np.eye(len(cols))
            Aty = np.einsum("knp,n->kp", A, y[:, p])
            try:
                coef = np.linalg.solve(AtA, Aty[..., None])[..., 0]
            except np.linalg.LinAlgError:
                continue
            sse = ((np.einsum("knp,kp->kn", A, coef) - y[:, p]) ** 2).sum(1)
            if descent:
                lin = [i for i, c in enumerate(cols) if c in (1, 2)]
                ok = (coef[:, lin] >= 0).all(1) & (coef[:, lin].sum(1) > 0)
            else:
                ok = np.abs(coef[:, 1:]).sum(1) > 0
            best = np.where(ok & (sse < best), sse, best)
        total += best
    k = int(np.argmin(total))
    if not np.isfinite(total[k]):
        return float("nan"), float("nan")
    tau = taus[k]
    if 0 < k < len(taus) - 1 and np.isfinite(total[k - 1]) and np.isfinite(total[k + 1]):
        den = total[k - 1] - 2 * total[k] + total[k + 1]
        if den > 0:
            tau += step * 0.5 * (total[k - 1] - total[k + 1]) / den
    return float(tau), float(math.sqrt(total[k] / y.size))


def _lsq_template(U: np.ndarray, Y: np.ndarray, lo: float, hi: float, knot_s: float):
    """Least-squares cubic spline per column on fixed knots (a regression: it cannot chase one step's noise)."""
    from scipy.interpolate import LSQUnivariateSpline
    o = np.argsort(U, kind="stable")
    U, Y = U[o] + np.arange(len(U)) * 1e-9, Y[o].reshape(len(U), -1)
    inner = np.arange(lo + knot_s, hi - knot_s / 2, knot_s)
    inner = inner[(inner > U[0]) & (inner < U[-1])]
    return [LSQUnivariateSpline(U, Y[:, p], inner, k=3) for p in range(Y.shape[1])]


def _step_samples(t, Y, taus, spans, pre, post):
    """Rows of each step inside [-pre, post] around its touchdown, times relative to it, plateau-referenced."""
    rows, us, ys = [], [], []
    for i, (a, b) in enumerate(spans):
        k = np.arange(a, b)
        k = k[(t[k] - taus[i] >= -pre) & (t[k] - taus[i] <= post)]
        u = t[k] - taus[i]
        after = u > 0.03
        plateau = np.median(Y[k[after]], 0) if after.sum() >= 2 else (np.median(Y[k], 0) if len(k) else 0.0)
        rows.append(k)
        us.append(u)
        ys.append(Y[k] - plateau)
    return rows, us, ys


def align_steps(t: np.ndarray, Y: np.ndarray, taus: np.ndarray, spans: list[tuple[int, int]], cfg: TimingConfig,
                noise_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Matched-filter alignment of one foot's steps to their leave-one-out step-averaged template.

    Y (T, C): leg channels (heights + pelvis-frame positions of knee/ankle/toe); spans[i] = rows of step i (its
    swing + stance). Each pass pools the other steps aligned at their current times into a fixed-knot spline
    template and shifts step i (one time shift shared by all channels; per channel an offset and an amplitude
    with a prior around 1, since a free amplitude trades off against the shift on a monotone descent). Uses the
    whole descent and every leg channel, so noise averages down. Only RELATIVE times are set here; the absolute
    touchdown inside the template comes from template_touchdown(). Returns (times, per-step rms (m))."""
    pre, post = cfg.template_pre_s, cfg.template_post_s
    taus = taus.astype(float).copy()
    rms = np.full(len(taus), np.nan)
    lam = (max(noise_m, cfg.noise_floor_m) / max(cfg.template_scale_sd, 1e-6)) ** 2
    for r in cfg.template_search_s:
        rows, us, ys = _step_samples(t, Y, taus, spans, pre, post)
        good = [i for i in range(len(rows)) if len(rows[i]) >= 6]
        if len(good) < cfg.template_min_steps:
            break
        new = taus.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in good:
                k = rows[i]
                k = k[(t[k] - taus[i] >= -pre + r) & (t[k] - taus[i] <= post - r)]
                if len(k) < 6:
                    continue
                loo = [j for j in good if j != i]
                tm = _lsq_template(np.concatenate([us[j] for j in loo]), np.concatenate([ys[j] for j in loo]),
                                   -pre, post, cfg.template_knot_s)
                shifts = np.arange(-r, r + 5e-4, 0.001)
                sse = np.zeros(len(shifts))
                for p in range(Y.shape[1]):
                    M = tm[p]((t[k][None, :] - taus[i] - shifts[:, None]).ravel()).reshape(len(shifts), len(k))
                    Mc = M - M.mean(1, keepdims=True)
                    yc = Y[k, p] - Y[k, p].mean()
                    scale = np.maximum(((Mc * yc).sum(1) + lam) / ((Mc * Mc).sum(1) + lam), 0.0)
                    sse += ((yc[None] - scale[:, None] * Mc) ** 2).sum(1) + lam * (scale - 1) ** 2
                j = int(np.argmin(sse))
                sh = shifts[j]
                if 0 < j < len(shifts) - 1:
                    den = sse[j - 1] - 2 * sse[j] + sse[j + 1]
                    if den > 0:
                        sh += 0.001 * 0.5 * (sse[j - 1] - sse[j + 1]) / den
                new[i] = taus[i] + sh
                rms[i] = math.sqrt(sse[j] / (len(k) * Y.shape[1]))
        taus = new
    return taus, rms


def template_touchdown(t: np.ndarray, groups: list, cfg: TimingConfig) -> float:
    """Where the touchdown sits relative to the aligned steps: hinge (descent end / vertical-velocity zero) of the
    pooled reference heights of all groups [(y (T,), taus, spans)] (e.g. both feet of a dancer). Returns offset (s)."""
    U, YY = [], []
    for y, taus, spans in groups:
        _, us, ys = _step_samples(t, y[:, None], taus, spans, cfg.template_pre_s, cfg.template_post_s)
        U += us
        YY += [v[:, 0] for v in ys]
    U, YY = np.concatenate(U), np.concatenate(YY)
    if len(U) < 12:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tm = _lsq_template(U, YY, -cfg.template_pre_s, cfg.template_post_s, cfg.template_knot_s)[0]
    uu = np.linspace(-cfg.template_pre_s, 0, 200)
    tv = tm(uu)
    pk = int(np.argmax(tv))
    above = np.nonzero(tv[pk:] > cfg.template_fit_frac * tv[pk])[0]
    u_lo = uu[pk + int(above[-1])] if len(above) else -cfg.template_pre_s
    sel = U >= u_lo
    tau0, _ = hinge_touchdown(U[sel], YY[sel], max(u_lo, -0.15), min(0.1, cfg.template_post_s - 0.03),
                              models=cfg.template_hinge_models)
    return float(tau0) if np.isfinite(tau0) else 0.0


def leg_channels(joints: np.ndarray, kin: Kinematics, foot: str) -> np.ndarray:
    """(T, 9) channels of one leg: heights of knee/ankle/toe (cols 0-2) and their forward/lateral position in a
    pelvis-centred heading frame from the smoothed hips (cols 3-8)."""
    knee = 4 if foot == "left" else 5
    ids = (knee, FOOT_JOINTS[foot]["ankle"], FOOT_JOINTS[foot]["toe"])
    pel = kin.pos[:, 0]
    lat = (kin.pos[:, 1] - kin.pos[:, 2])[:, [0, 2]]
    lat /= np.maximum(np.linalg.norm(lat, axis=1, keepdims=True), 1e-9)
    fwd = np.stack([-lat[:, 1], lat[:, 0]], 1)
    ch = [joints[:, j, 1] for j in ids]
    for j in ids:
        d = (joints[:, j] - pel)[:, [0, 2]]
        ch += [(d * fwd).sum(1), (d * lat).sum(1)]
    return np.stack(ch, 1)


def _vy_zero(te, vy, i_lo, i_hi, eps: float) -> float:
    """Spline method: first time after the fastest descent where the vertical velocity rises above -eps."""
    seg = vy[i_lo:i_hi]
    if len(seg) < 3:
        return float("nan")
    m = int(np.argmin(seg))
    for i in range(i_lo + m, i_hi - 1):
        if vy[i] < -eps <= vy[i + 1]:
            return float(te[i] + (te[i + 1] - te[i]) * (-eps - vy[i]) / (vy[i + 1] - vy[i]))
    return float("nan")


def stance_heights(y: np.ndarray) -> float:
    """Contact height of a joint: median of its lowest quarter of samples (feet spend much time on the floor)."""
    q = np.percentile(y, 25)
    return float(np.median(y[y <= q]))


def detect_touchdowns(t: np.ndarray, joints: np.ndarray, kin: Kinematics, noise_m: float, cfg: TimingConfig,
                      fine: Kinematics | None = None) -> list[dict]:
    """Touchdowns for both feet: contact intervals with hysteresis, then sub-frame time per event.

    Per foot: stance height of ankle and toe (lowest quarter), foot height H = min over parts, contact when H is
    low AND the ankle's vertical speed is small AND the foot is not sweeping fast; leave contact on clear lift or
    fast horizontal motion. A touchdown is the start of a stance preceded by a swing with real clearance. Its time
    is the hinge (descent end) of the raw heights, refined by the per-foot descent template (matched filter)."""
    dt = float(np.median(np.diff(t)))
    events, by_foot = [], {}
    for foot, parts in FOOT_JOINTS.items():
        ja, jt = parts["ankle"], parts["toe"]
        ys = kin.pos[:, [ja, jt], 1]
        h0 = np.array([stance_heights(ys[:, 0]), stance_heights(ys[:, 1])])
        h = ys - h0
        H = h.min(1)
        vy = kin.vel[:, ja, 1]
        vxz = np.linalg.norm(kin.vel[:, [ja, jt]][:, :, [0, 2]], axis=-1).max(1)
        # thresholds grow with noise but stay below this foot's typical clearance (low zouk steps)
        clearance = float(np.percentile(H, 95))
        h_on = min(cfg.contact_height_m + cfg.noise_k * noise_m, 0.35 * clearance)
        h_off = max(min(cfg.release_height_m + cfg.noise_k * noise_m, 0.6 * clearance), h_on + 0.005)
        spans = _hysteresis(H, vy, vxz, h_on, h_off, cfg, dt)
        foot_ev, prev_end = [], 0
        for s, e in spans:
            sw0, prev_end = prev_end, e
            if s - sw0 < 3:  # stance already in progress at the window start, or no observable swing
                continue
            apex = H[sw0:s].max()
            if apex < max(cfg.min_apex_m, h_off):
                continue
            heel_up = np.median(h[s:e, 0]) > h_on  # releve: the heel stays up through stance -> time the toe
            which = cfg.touchdown_joint if cfg.touchdown_joint != "auto" else ("toe" if heel_up else "ankle")
            res = {}
            for name, col in (("ankle", 0), ("toe", 1)):
                hj = h[:, col]
                ia = sw0 + int(np.argmax(hj[sw0:s]))
                above = np.nonzero(hj[ia:s] > cfg.fit_upper_frac * hj[ia])[0]
                i_lo = ia + (int(above[-1]) if len(above) else 0)
                i_hi = min(e, s + max(2, int(round(cfg.fit_after_s / dt)))) - 1
                res[name] = hinge_touchdown(t[i_lo:i_hi + 1], kin.pos[i_lo:i_hi + 1, parts[name], 1], t[i_lo], t[i_hi])
            t_td = res[which][0]
            if not np.isfinite(t_td):
                continue
            ev = {"t": float(t_td), "foot": foot, "joint": which, "method": "hinge", "t_hinge": float(t_td),
                  "t_ankle": res["ankle"][0], "t_toe": res["toe"][0], "fit_rms_mm": 1000 * res[which][1],
                  "stance": [float(t[s]), float(t[e - 1])], "apex_m": float(apex), "_rows": (sw0, e)}
            if fine is not None:  # spline cross-check: vertical velocity of the ankle back to ~zero
                fs = int(np.searchsorted(fine.te, t[sw0]))
                fe = int(np.searchsorted(fine.te, min(t[e - 1], t[s] + cfg.fit_after_s)))
                ev["t_spline_vy0"] = _vy_zero(fine.te, fine.vel[:, ja, 1], fs, fe, eps=0.02 + 2 * noise_m)
            foot_ev.append(ev)
        by_foot[foot] = foot_ev
    if cfg.template:
        groups = []
        for foot, foot_ev in by_foot.items():
            if len(foot_ev) < cfg.template_min_steps:
                continue
            spans = [e["_rows"] for e in foot_ev]
            tt, rms = align_steps(t, leg_channels(joints, kin, foot), np.array([e["t"] for e in foot_ev]), spans, cfg,
                                  noise_m)
            ok = [i for i, (x, e) in enumerate(zip(tt, foot_ev)) if np.isfinite(rms[i])
                  and abs(x - e["t_hinge"]) < 2 * max(cfg.template_search_s)]
            for i in ok:
                foot_ev[i].update(_aligned=float(tt[i]), template_rms_mm=1000 * float(rms[i]))
            toe = sum(e["joint"] == "toe" for e in foot_ev) > len(foot_ev) / 2
            sel = [foot_ev[i] for i in ok]
            groups.append((joints[:, FOOT_JOINTS[foot]["toe" if toe else "ankle"], 1],
                           np.array([e["_aligned"] for e in sel]), [e["_rows"] for e in sel], sel))
        if groups:
            tau0 = 0.0
            for _ in range(2):  # second pass re-centres the window on the touchdown found by the first
                tau0 += template_touchdown(t, [(y, taus + tau0, sp) for y, taus, sp, _ in groups], cfg)
            for g in groups:
                for e in g[3]:
                    e.update(t=e.pop("_aligned") + tau0, method="template", template_offset_ms=1000 * tau0)
    for foot_ev in by_foot.values():
        for e in foot_ev:
            e.pop("_rows")
            e.pop("_aligned", None)
        events += foot_ev
    events.sort(key=lambda d: d["t"])
    return events


# ----------------------------------------------------------------------------------------------- accents

def detect_accents(fine: Kinematics, cfg: TimingConfig) -> dict:
    """Local maxima of |acc| and |jerk| per key joint on the oversampled spline, with prominence + separation."""
    from scipy.signal import find_peaks
    dte = float(np.median(np.diff(fine.te)))
    out = {}
    for name, j in cfg.accent_joints.items():
        out[name] = {}
        for kind, arr in (("acc", fine.acc), ("jerk", fine.jerk)):
            mag = np.linalg.norm(arr[:, j], axis=-1)
            spread = np.percentile(mag, 90) - np.percentile(mag, 10)
            pk, props = find_peaks(mag, prominence=max(cfg.accent_prominence * spread, 1e-9),
                                   distance=max(1, int(cfg.accent_min_sep_s / dte)))
            evs = []
            for p, prom in zip(pk, props["prominences"]):
                tp = fine.te[p]
                if 0 < p < len(mag) - 1:
                    den = mag[p - 1] - 2 * mag[p] + mag[p + 1]
                    if den < 0:
                        tp += dte * 0.5 * (mag[p - 1] - mag[p + 1]) / den
                evs.append({"t": float(tp), "value": float(mag[p]), "prominence": float(prom)})
            out[name][kind] = evs
    return out


# ----------------------------------------------------------------------------------------------- beat grid

@dataclass
class BeatGrid:
    t: np.ndarray
    type: np.ndarray
    measure: np.ndarray
    pos: np.ndarray

    def allowed(self, cfg: TimingConfig) -> np.ndarray:
        m = np.isin(self.type, cfg.allowed_types)
        if cfg.allowed_positions is not None:
            m &= np.isin(self.pos, cfg.allowed_positions)
        return m


def load_beats(path: Path) -> BeatGrid:
    arr = np.asarray(json.loads(Path(path).read_text()), float).reshape(-1, 2)
    i = np.arange(len(arr))
    return BeatGrid(arr[:, 0], arr[:, 1].astype(int), i // MEASURE_LEN, i % MEASURE_LEN)


def coarsen(grid: BeatGrid, factor: int) -> BeatGrid:
    """Undo beat_this split_measures (min_bpm=120) halving: keep every `factor`-th entry, recompute zouk types."""
    if factor <= 1:
        return grid
    t = grid.t[::factor]
    i = np.arange(len(t))
    pos = i % MEASURE_LEN
    typ = np.where(np.isin(pos, (0, 4)), 1, np.where(np.isin(pos, (3, 6)), 2, 3))
    return BeatGrid(t, typ, i // MEASURE_LEN, pos)


def refine_beats(grid: BeatGrid, half_window: int = MEASURE_LEN) -> tuple[BeatGrid, float]:
    """Local-linear (tempo) fit of beat time vs index over +-1 measure; returns refined grid + residual RMS (s)."""
    n = len(grid.t)
    i = np.arange(n, dtype=float)
    ref = grid.t.copy()
    for k in range(n):
        lo, hi = max(0, k - half_window), min(n, k + half_window + 1)
        w = 1.0 - np.abs(i[lo:hi] - k) / (half_window + 1)
        A = np.stack([np.ones(hi - lo), i[lo:hi] - k], 1) * w[:, None]
        coef = np.linalg.lstsq(A, grid.t[lo:hi] * w, rcond=None)[0]
        ref[k] = coef[0]
    rms = float(np.sqrt(np.mean((ref - grid.t) ** 2))) if n else 0.0
    return BeatGrid(ref, grid.type, grid.measure, grid.pos), rms


def grid_diagnostics(grid: BeatGrid, cfg: TimingConfig) -> dict:
    """Tempo, measure density and chance hit rate; warns when the grid is so dense that 'on beat' is likely."""
    al = grid.t[grid.allowed(cfg)]
    warns = []
    meas = np.diff(grid.t[::MEASURE_LEN]) if len(grid.t) > MEASURE_LEN else np.array([np.nan])
    measure_s = float(np.median(meas))
    grid_bpm = 240.0 / measure_s if measure_s > 0 else float("nan")
    gaps = np.diff(al)
    chance = float(np.minimum(gaps, 2 * cfg.hit_tol_s).sum() / gaps.sum()) if len(gaps) else float("nan")
    if grid_bpm > 120:
        warns.append(f"grid measures run at {grid_bpm:.0f} BPM (4 beats per 8 entries); zouk is ~60-110 BPM, so "
                     f"beat_this split_measures(min_bpm=120) probably halved each measure into a dense grid. "
                     f"Consider coarsen_grid=2.")
    if chance > cfg.dense_chance_warn:
        warns.append(f"allowed beats every {np.median(gaps) * 1000:.0f} ms: a random event is within "
                     f"+-{cfg.hit_tol_s * 1000:.0f} ms of one {chance:.0%} of the time; compare hit rates to "
                     f"the surrogate baseline, not to 0.")
    return {"n": int(len(grid.t)), "n_allowed": int(len(al)), "measure_s": measure_s, "grid_measure_bpm": grid_bpm,
            "allowed_spacing_ms": float(np.median(gaps) * 1000) if len(gaps) else None,
            "chance_hit_rate_analytic": chance, "warnings": warns}


def assign_beats(times: np.ndarray, grid: BeatGrid, cfg: TimingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Nearest allowed beat index (into grid) and its local cycle length (mean gap to allowed neighbours)."""
    ai = np.nonzero(grid.allowed(cfg))[0]
    bt = grid.t[ai]
    k = np.clip(np.searchsorted(bt, times), 1, len(bt) - 1)
    k = np.where(np.abs(times - bt[k - 1]) <= np.abs(times - bt[k]), k - 1, k)
    gaps = np.diff(bt)
    period = np.empty(len(bt))
    period[1:-1] = 0.5 * (gaps[1:] + gaps[:-1])
    period[0], period[-1] = gaps[0], gaps[-1]
    return ai[k], period[k]


# ----------------------------------------------------------------------------------------------- statistics

def circular(asyn: np.ndarray, period: np.ndarray) -> dict:
    """Circular mean asynchrony (s), resultant length R, V (on-beat locking), Rayleigh p."""
    if len(asyn) == 0:
        return {"n": 0}
    ph = 2 * np.pi * asyn / period
    C, S = np.cos(ph).mean(), np.sin(ph).mean()
    R = float(math.hypot(C, S))
    n = len(asyn)
    Z = n * R * R
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - n * n * R * R)) - (1 + 2 * n))  # Zar's Rayleigh approximation
    return {"circ_mean_ms": float(math.atan2(S, C) / (2 * np.pi) * np.median(period) * 1000), "R": R, "V": float(C),
            "circ_std_ms": float(math.sqrt(max(0.0, -2 * math.log(max(R, 1e-12)))) / (2 * np.pi)
                                 * np.median(period) * 1000), "rayleigh_Z": float(Z), "rayleigh_p": float(min(1.0, p))}


def _ci(x: np.ndarray, q=(2.5, 97.5)) -> list[float]:
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, q)] if len(x) else [float("nan")] * 2


def group_stats(nom: np.ndarray, mc: np.ndarray, period: np.ndarray, cfg: TimingConfig, rng) -> dict:
    """Mean asynchrony with a CI that combines measurement MC (rows of mc) and event bootstrap."""
    n = len(nom)
    if n == 0:
        return {"n": 0}
    D = mc.shape[0]
    boot = rng.integers(0, n, size=(D, n))
    means = np.nanmean(np.take_along_axis(mc, boot, 1), 1) * 1000
    hits = np.nanmean(np.abs(np.take_along_axis(mc, boot, 1)) <= cfg.hit_tol_s, 1)
    out = {"n": int(n), "mean_ms": float(nom.mean() * 1000), "ci95_ms": _ci(means),
           "sd_measurement_ms": float(np.nanstd(np.nanmean(mc, 1)) * 1000), "median_ms": float(np.median(nom) * 1000),
           "hit_rate": float(np.mean(np.abs(nom) <= cfg.hit_tol_s)), "hit_rate_ci95": _ci(hits)}
    out.update(circular(nom, period))
    return out


def surrogate(times: np.ndarray, grid: BeatGrid, cfg: TimingConfig, rng, observed: dict) -> dict:
    """Null baseline: circularly shift the whole event train by U(0, 1 measure) inside the beat span, reassign."""
    if len(times) == 0:
        return {}
    al = grid.t[grid.allowed(cfg)]
    lo, hi = al[0], al[-1]
    span = hi - lo
    meas = float(np.median(np.diff(grid.t[::MEASURE_LEN]))) if len(grid.t) > MEASURE_LEN else span
    ts = lo + (times[None, :] - lo + rng.uniform(0, meas, (cfg.n_surrogates, 1))) % span
    bi, per = assign_beats(ts, grid, cfg)
    a = ts - grid.t[bi]
    hits = np.mean(np.abs(a) <= cfg.hit_tol_s, 1)
    vs = np.mean(np.cos(2 * np.pi * a / per), 1)
    gaps = np.diff(al)
    floor = float(np.minimum(2 * cfg.hit_tol_s / np.median(gaps), 1.0)) if len(gaps) else float("nan")
    return {"hit_rate_mean": float(hits.mean()), "hit_rate_p95": float(np.percentile(hits, 95)),
            "p_floor_isochronous": floor,
            "hit_rate_p": float((1 + np.sum(hits >= observed["hit_rate"])) / (1 + len(hits))),
            "V_mean": float(vs.mean()), "V_p95": float(np.percentile(vs, 95)),
            "V_p": float((1 + np.sum(vs >= observed["V"])) / (1 + len(vs))),
            "note": "R is invariant to shifting a periodic train, so the surrogate tests hit rate and V = mean cos(phase). "
                    "For steady (isochronous) stepping, a shift by one allowed-beat spacing restores every hit, so p "
                    "cannot go below p_floor_isochronous = 2 tol / spacing however precise the dancer is; restrict "
                    "allowed_types/positions to the intended counts for a sharper test."}


def measure_effects(asyn: np.ndarray, measure: np.ndarray, cfg: TimingConfig, rng) -> dict:
    """Permutation test: does a measure's mean asynchrony differ from the dancer's other measures (BH-corrected)?"""
    ms = np.unique(measure)
    if len(ms) < 2 or len(asyn) < 4:
        return {"measures": [], "global_p": float("nan")}
    overall = asyn.mean()
    onehot = (measure[:, None] == ms[None, :]).astype(float)  # (n, M)
    cnts = onehot.sum(0)
    obs = asyn @ onehot / cnts - overall
    F_obs = float(np.sum(cnts * obs ** 2))
    perm = asyn[np.argsort(rng.random((cfg.n_perm, len(asyn))), axis=1)]  # permuting values == permuting labels
    d = perm @ onehot / cnts - overall  # (n_perm, M)
    cnt = (np.abs(d) >= np.abs(obs) - 1e-12).sum(0)
    F_cnt = int((np.sum(cnts * d ** 2, 1) >= F_obs - 1e-15).sum())
    p = (1 + cnt) / (1 + cfg.n_perm)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return {"measures": [{"measure": int(m), "dev_ms": float(d * 1000), "p": float(pp), "q_bh": float(min(1.0, qq))}
                         for m, d, pp, qq in zip(ms, obs, p, q)],
            "global_p": float((1 + F_cnt) / (1 + cfg.n_perm))}


def match(a: np.ndarray, b: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of sorted event times by |difference| <= tol."""
    if len(a) == 0 or len(b) == 0:
        return []
    d = np.abs(a[:, None] - b[None, :])
    pairs, ua, ub = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(d, None), d.shape)):
        if d[i, j] > tol:
            break
        if i not in ua and j not in ub:
            pairs.append((int(i), int(j)))
            ua.add(i)
            ub.add(j)
    return pairs


# ----------------------------------------------------------------------------------------------- pipeline

def _needed(cfg: TimingConfig) -> list[int]:
    """Joints whose derivatives the analysis needs: pelvis + hips (heading frame), knees, feet, accent joints."""
    legs = {0, 1, 2, 4, 5} | {j for p in FOOT_JOINTS.values() for j in p.values()}
    return sorted(legs | set(cfg.accent_joints.values()))


def events_for(t: np.ndarray, joints: np.ndarray, noise_m: float, cfg: TimingConfig) -> tuple[list, dict, Kinematics]:
    """Touchdowns + accents of one dancer (one spline fit, evaluated at the samples and 4x oversampled)."""
    te = np.linspace(t[0], t[-1], (len(t) - 1) * cfg.accent_oversample + 1)
    kin, fine = spline_kinematics(t, joints, noise_m, _needed(cfg), te=(t, te), k=cfg.spline_k)
    return detect_touchdowns(t, joints, kin, noise_m, cfg, fine), detect_accents(fine, cfg), kin


def _mc_chunk(t: np.ndarray, smooth: np.ndarray, noise_m: float, cfg: TimingConfig, seeds: list[int]) -> list:
    """Monte-Carlo draws (worker): Gaussian joint noise around the spline fit -> compact event times."""
    out = []
    for sd in seeds:
        jd = smooth + np.random.default_rng(sd).normal(0, noise_m, smooth.shape)
        tds, acc, _ = events_for(t, jd, noise_m, cfg)
        out.append(({f: np.array([e["t"] for e in tds if e["foot"] == f]) for f in FOOT_JOINTS},
                    {n: {k: np.array([e["t"] for e in v]) for k, v in kinds.items()} for n, kinds in acc.items()}))
    return out


def run_mc(t: np.ndarray, smooth: np.ndarray, noise_m: float, cfg: TimingConfig, rng) -> list:
    """cfg.mc_draws draws, spread over cfg.workers processes (spline fits are CPU bound)."""
    seeds = [int(x) for x in rng.integers(0, 2 ** 31 - 1, cfg.mc_draws)]
    if not seeds:
        return []
    workers = max(1, min(cfg.workers, len(seeds)))
    if workers == 1:
        return _mc_chunk(t, smooth, noise_m, cfg, seeds)
    from concurrent.futures import ProcessPoolExecutor
    chunks = [seeds[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_mc_chunk, *zip(*[(t, smooth, noise_m, cfg, c) for c in chunks])))
    return [d for part in parts for d in part]


def _mc_times(nominal: np.ndarray, draws: list[np.ndarray], tol: float) -> tuple[np.ndarray, np.ndarray]:
    """(D, n) matched MC event times (NaN when missed) and the per-event detection rate."""
    out = np.full((len(draws), len(nominal)), np.nan)
    for d, ts in enumerate(draws):
        for i, j in match(nominal, np.asarray(ts), tol):
            out[d, i] = ts[j]
    return out, np.mean(np.isfinite(out), 0) if len(draws) else np.ones(len(nominal))


def _attach_mc(e: dict, col: np.ndarray, rate: float) -> None:
    """sigma = MC spread; '_mc' = nominal time + MC deviation (the MC is used for spread only: its centre comes
    from the smoothed signal, which is slightly rounder than the data)."""
    ok = np.isfinite(col)
    e["sigma"] = float(max(np.std(col[ok]), 1e-4)) if ok.sum() > 1 else float("nan")
    e["detect_rate"] = float(rate)
    e["_mc"] = e["t"] + (col - np.median(col[ok])) if ok.sum() > 1 else np.full(len(col), e["t"])


def analyze_dancer(t: np.ndarray, joints: np.ndarray, cfg: TimingConfig, rng) -> dict:
    """Nominal events + Monte-Carlo event times (pose noise re-sampled around the spline fit)."""
    noise = cfg.noise_m if cfg.noise_m is not None else estimate_noise(joints[:, _needed(cfg)])
    noise = max(noise, cfg.noise_floor_m)
    tds, acc, kin = events_for(t, joints, noise, cfg)
    smooth = joints.copy()
    smooth[:, _needed(cfg)] = kin.pos[:, _needed(cfg)]
    draws = run_mc(t, smooth, noise, cfg, rng)
    out = {"noise_m": noise, "noise_estimated": cfg.noise_m is None, "touchdowns": [], "accents": {}}
    for foot in FOOT_JOINTS:
        nom = [e for e in tds if e["foot"] == foot]
        nt = np.array([e["t"] for e in nom])
        mc, rate = _mc_times(nt, [d[0][foot] for d in draws], cfg.match_tol_s)
        for e, col, r in zip(nom, mc.T, rate):
            _attach_mc(e, col, r)
        out["touchdowns"] += nom
    out["touchdowns"].sort(key=lambda d: d["t"])
    for name in acc:
        out["accents"][name] = {}
        for kind, evs in acc[name].items():
            nt = np.array([e["t"] for e in evs])
            mc, rate = _mc_times(nt, [d[1][name][kind] for d in draws], cfg.accent_min_sep_s / 2)
            for e, col, r in zip(evs, mc.T, rate):
                _attach_mc(e, col, r)
            out["accents"][name][kind] = evs
    # Butterworth cross-check on the nominal data
    bk = butter_kinematics(t, joints, cfg.butter_cutoff_hz, cfg.butter_order, _needed(cfg))
    te = np.linspace(t[0], t[-1], (len(t) - 1) * cfg.accent_oversample + 1)
    bf = butter_kinematics(t, joints, cfg.butter_cutoff_hz, cfg.butter_order, _needed(cfg), te=te)
    btd = detect_touchdowns(t, joints, bk, noise, cfg, bf)
    sp = np.array([e.get("t_spline_vy0", np.nan) for e in tds])
    bt = np.array([e.get("t_spline_vy0", np.nan) for e in btd])
    pairs = match(np.array([e["t"] for e in tds]), np.array([e["t"] for e in btd]), cfg.match_tol_s)
    dvy = [bt[j] - sp[i] for i, j in pairs if np.isfinite(sp[i]) and np.isfinite(bt[j])]
    jj = _needed(cfg)
    out["cross_check"] = {
        "butterworth": f"order {cfg.butter_order}, {cfg.butter_cutoff_hz} Hz, filtfilt",
        "vel_rms_diff_m_s": float(np.sqrt(np.nanmean((bk.vel[:, jj] - kin.vel[:, jj]) ** 2))),
        "acc_rms_diff_m_s2": float(np.sqrt(np.nanmean((bk.acc[:, jj] - kin.acc[:, jj]) ** 2))),
        "touchdowns_spline_vs_butter": [len(tds), len(btd), len(pairs)],
        "vy0_median_diff_ms": float(np.median(dvy) * 1000) if dvy else None,
        "vy0_median_absdiff_ms": float(np.median(np.abs(dvy)) * 1000) if dvy else None,
        "hinge_minus_spline_vy0_median_ms": float(np.nanmedian(np.array([e["t"] for e in tds]) - sp) * 1000)
        if np.isfinite(sp).any() else None,
    }
    out["_kin"] = kin
    return out


def _event_async(evs: list[dict], grid: BeatGrid, beats_mc: np.ndarray, sync_mc: np.ndarray, cfg: TimingConfig):
    """Assign events (nominal) to beats; asynchrony nominal (n,) and MC (D, n) with beat jitter + sync offset."""
    if not evs:
        return np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros((len(sync_mc), 0))
    t = np.array([e["t"] for e in evs]) + cfg.av_offset_s
    bi, per = assign_beats(t, grid, cfg)
    nom = t - grid.t[bi]
    mc = np.stack([e["_mc"] for e in evs], 1) + cfg.av_offset_s + sync_mc[:, None] - beats_mc[:, bi]
    for k, (e, b, a, p) in enumerate(zip(evs, bi, nom, per)):
        e.update(beat_index=int(b), beat_t=float(grid.t[b]), beat_type=int(grid.type[b]), measure=int(grid.measure[b]),
                 count=int(grid.pos[b]), async_ms=float(a * 1000), period_ms=float(p * 1000), _async_mc=mc[:, k])
    return bi, nom, per, mc


def _summarize(evs, grid, beats_mc, sync_mc, cfg, rng, with_surrogate=True) -> dict:
    bi, nom, per, mc = _event_async(evs, grid, beats_mc, sync_mc, cfg)
    if len(nom) == 0:
        return {"n": 0}
    s = group_stats(nom, mc, per, cfg, rng)
    if with_surrogate:
        s["surrogate"] = surrogate(np.array([e["t"] for e in evs]) + cfg.av_offset_s, grid, cfg, rng, s)
    meas = grid.measure[bi]
    s["per_measure"] = []
    for m in np.unique(meas):
        k = meas == m
        g = group_stats(nom[k], mc[:, k], per[k], cfg, rng)
        s["per_measure"].append({"measure": int(m), "t0": float(grid.t[m * MEASURE_LEN]) if m * MEASURE_LEN < len(
            grid.t) else None, **{x: g[x] for x in ("n", "mean_ms", "ci95_ms", "hit_rate")}})
    s["measure_effects"] = measure_effects(nom, meas, cfg, rng)
    s["_mc_by_measure"] = {int(m): mc[:, meas == m] for m in np.unique(meas)}
    return s


def analyze(t_by_role: dict, joints_by_role: dict, beats: BeatGrid, cfg: TimingConfig) -> dict:
    """Full timing analysis for both dancers -> json-able dict (+ private '_' entries for plotting)."""
    rng = np.random.default_rng(cfg.seed)
    raw_grid = coarsen(beats, cfg.coarsen_grid)
    grid, rms = refine_beats(raw_grid) if cfg.refine_beats else (raw_grid, 0.0)
    diag = grid_diagnostics(grid, cfg)
    jitter = cfg.beat_jitter_s if cfg.beat_jitter_s is not None else max(rms, 0.010)
    D = cfg.mc_draws
    beats_mc = grid.t[None, :] + rng.normal(0, jitter, (D, len(grid.t)))
    sync_mc = rng.normal(0, cfg.sync_sigma_s, (D, cfg.n_cams)).mean(1)
    res = {"version": 1, "config": _cfg_json(cfg), "beats": {
        **diag, "refined": cfg.refine_beats, "refine_residual_rms_ms": rms * 1000, "jitter_used_ms": jitter * 1000,
        "coarsen_grid": cfg.coarsen_grid, "tempo_bpm": 240.0 / diag["measure_s"] * (1 if diag["grid_measure_bpm"] <= 120
                                                                                     else 0.5),
        "grid": [[round(float(a), 6), int(b), int(m), int(p)] for a, b, m, p in
                 zip(grid.t, grid.type, grid.measure, grid.pos)], "grid_columns": ["t", "type", "measure", "count"]},
        "kinematics": {}, "footsteps": {}, "accents": {}, "asynchrony": {}, "_kin": {}, "_grid": grid}
    for role in ROLES:
        if role not in joints_by_role:
            continue
        d = analyze_dancer(t_by_role[role], joints_by_role[role], cfg, rng)
        res["_kin"][role] = d.pop("_kin")
        res["kinematics"][role] = {"noise_m": d["noise_m"], "noise_estimated": d["noise_estimated"],
                                   "cross_check": d["cross_check"]}
        tds = d["touchdowns"]
        res["asynchrony"][role] = {"footsteps": _summarize(tds, grid, beats_mc, sync_mc, cfg, rng), "accents": {}}
        for name, kinds in d["accents"].items():
            res["asynchrony"][role]["accents"][name] = {
                kind: _summarize(evs, grid, beats_mc, sync_mc, cfg, rng, with_surrogate=kind == "acc")
                for kind, evs in kinds.items()}
        res["footsteps"][role] = tds
        res["accents"][role] = d["accents"]
    if all(r in res["asynchrony"] for r in ROLES):
        res["asynchrony"]["follow_minus_lead"] = _relative(res, rng)
    res["absolute_timing"] = {
        "av_offset_s": cfg.av_offset_s, "av_sync_event": cfg.av_sync_event,
        "sync_sigma_s": cfg.sync_sigma_s, "n_cams": cfg.n_cams,
        "valid": cfg.av_sync_event is not None,
        "note": ("Event times are video reference seconds shifted by av_offset_s onto the beat (audio) clock. Absolute "
                 "on-beat claims (mean asynchrony, hit rate) need an audio-visual sync event (clap, slate, flash) "
                 "that pins the A/V offset; without one only RELATIVE timing is trustworthy: follow-minus-lead, "
                 "measure-to-measure changes, consistency (R, sd).")}
    res["notes"] = NOTES
    return res


def _relative(res: dict, rng) -> dict:
    """Follow minus lead mean footstep asynchrony, overall and per measure (cancels A/V offset and sync)."""
    fl, ld = res["asynchrony"]["follow"]["footsteps"], res["asynchrony"]["lead"]["footsteps"]
    if not fl.get("n") or not ld.get("n"):
        return {}
    out = []
    for m in sorted(set(fl["_mc_by_measure"]) & set(ld["_mc_by_measure"])):
        a, b = fl["_mc_by_measure"][m], ld["_mc_by_measure"][m]
        D = a.shape[0]
        ba = rng.integers(0, a.shape[1], (D, a.shape[1]))
        bb = rng.integers(0, b.shape[1], (D, b.shape[1]))
        diff = (np.nanmean(np.take_along_axis(a, ba, 1), 1) - np.nanmean(np.take_along_axis(b, bb, 1), 1)) * 1000
        nom = next(x["mean_ms"] for x in fl["per_measure"] if x["measure"] == m) - \
            next(x["mean_ms"] for x in ld["per_measure"] if x["measure"] == m)
        out.append({"measure": m, "diff_ms": nom, "ci95_ms": _ci(diff)})
    return {"overall_ms": fl["mean_ms"] - ld["mean_ms"], "per_measure": out,
            "note": "relative timing: independent of the A/V offset and of the common sync offset"}


def _cfg_json(cfg: TimingConfig) -> dict:
    d = asdict(cfg)
    d["allowed_types"] = list(cfg.allowed_types)
    d["allowed_positions"] = None if cfg.allowed_positions is None else list(cfg.allowed_positions)
    return d


NOTES = [
    "Derivatives: quintic smoothing spline per coordinate on the real timestamps, smoothing s = N*sigma^2 with sigma "
    "the joint noise (estimated from 4th differences unless configured). Butterworth filtfilt + central differences is "
    "a cross-check only; forward differences are never used (they shift jerk ~1.5 frames early).",
    "Touchdown = end of the foot's descent onto its stance height (heel/ankle by default = foot-flat / weight "
    "acceptance; toe if the heel stays up). Sub-frame time from a hinge fit (plateau + linear/quadratic descent) to "
    "the raw heights, i.e. the height reaching stance and the vertical velocity reaching zero. t_toe is the toe's own "
    "arrival; t_spline_vy0 the spline vertical-velocity cross-check. Sliding steps without clearance are not counted.",
    "sigma per event = SD over a parametric Monte-Carlo (Gaussian joint noise of the noise model around the spline "
    "fit). detect_rate = fraction of MC draws that found the event.",
    "Accents = local maxima of |acceleration| and |jerk| of key joints with prominence. At 30 fps jerk peaks are much "
    "less reliable than contact-model touchdowns (third derivative of a ~33 ms sampled signal); treat them as "
    "descriptive.",
    "Asynchrony = event - nearest allowed beat (allowed_types / allowed_positions). CIs combine Monte-Carlo pose noise, "
    "beat jitter, the per-camera sync uncertainty (mean over n_cams) and an event bootstrap. Chance level: circular "
    "shifts of the event train (surrogate) and the analytic +-tol coverage of the allowed grid.",
    "measure_effects: permutation test of each measure's mean asynchrony against the dancer's other measures, "
    "Benjamini-Hochberg q values; a late measure has dev_ms > 0 with small q.",
]


# ----------------------------------------------------------------------------------------------- io

def load_smplx(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return np.asarray(d["t"], float), np.asarray(d["joints"], float)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def find_beats(work: Path) -> Path:
    for p in (work / "timing" / "beats.json", work / "beats.json", work / "zouk-time-analysis.json"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no beats json in {work} (timing/beats.json, beats.json or zouk-time-analysis.json); "
                            f"pass --beats")


def run(work: Path, cfg: TimingConfig, beats_path: Path | None = None, out_dir: Path | None = None,
        noise_inject_m: float = 0.0, plot: bool = True) -> dict:
    """Read smplx/{lead,follow}.npz + beats -> out_dir/timing.json (+ kinematics.npz, qa_timing.png)."""
    out_dir = out_dir or work / "timing"
    ts, js = {}, {}
    rng = np.random.default_rng(cfg.seed + 1000)
    for role in ROLES:
        p = work / "smplx" / f"{role}.npz"
        if p.exists():
            ts[role], js[role] = load_smplx(p)
            if noise_inject_m > 0:
                js[role] = js[role] + rng.normal(0, noise_inject_m, js[role].shape)
    if not ts:
        raise FileNotFoundError(f"no smplx/{{lead,follow}}.npz in {work} (run the mamma stage first)")
    beats_path = beats_path or find_beats(work)
    sync = work / "sync" / "offsets.json"
    if sync.exists():
        try:
            cfg.n_cams = max(1, len(json.loads(sync.read_text()).get("offsets", json.loads(sync.read_text()))))
        except (AttributeError, TypeError, ValueError):
            pass
    res = analyze(ts, js, load_beats(beats_path), cfg)
    res["_joints"] = js
    res["inputs"] = {"smplx": {r: str(work / "smplx" / f"{r}.npz") for r in ts}, "beats": str(beats_path),
                     "noise_injected_m": noise_inject_m}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timing.json").write_text(json.dumps(_clean(res), indent=1))
    kin = {}
    for role in ts:  # full 55-joint derivatives for downstream stages (physics, Unity overlays)
        k = spline_kinematics(ts[role], js[role], res["kinematics"][role]["noise_m"], k=cfg.spline_k)
        kin.update({f"{role}_{n}": getattr(k, n).astype(np.float32) for n in ("pos", "vel", "acc", "jerk")})
        kin[f"{role}_t"] = ts[role]
    np.savez_compressed(out_dir / "kinematics.npz", **kin)
    if plot:
        plot_qa(res, js, ts, out_dir / "qa_timing.png")
    res["_out"] = out_dir
    return res


def plot_qa(res: dict, js: dict, ts: dict, path: Path, truth: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = res["_grid"]
    roles = [r for r in ROLES if r in js]
    fig, axs = plt.subplots(len(roles) + 1, 1, figsize=(18, 4.2 * (len(roles) + 1)), sharex=True)
    for ax, role in zip(axs, roles):
        t, J = ts[role], js[role]
        for foot, c in (("left", "tab:blue"), ("right", "tab:red")):
            ja = FOOT_JOINTS[foot]["ankle"]
            ax.plot(t, J[:, ja, 1], ".", color=c, ms=2, alpha=.6)
            ax.plot(t, res["_kin"][role].pos[:, ja, 1], "-", color=c, lw=.8, label=f"{foot} ankle y")
            for e in res["footsteps"][role]:
                if e["foot"] == foot:
                    ax.axvline(e["t"], color=c, lw=1.2, alpha=.8)
        if truth:
            y0 = float(np.nanmin(J[:, [7, 8], 1])) - 0.01
            ax.plot([e["t"] for e in truth["touchdowns"][role]], [y0] * len(truth["touchdowns"][role]), "k^", ms=6)
        for bt, ty in zip(grid.t, grid.type):
            if ty == 1:
                ax.axvline(bt, color="k", lw=.4, ls=":")
        ax.set_ylabel(f"{role}: ankle height (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{role}: touchdowns (vertical lines) | beat type 1 dotted | truth triangles" if truth else role)
    ax = axs[-1]
    for role, c in zip(roles, ("tab:green", "tab:purple")):
        evs = res["footsteps"][role]
        x = [e["t"] for e in evs]
        y = [e.get("async_ms", np.nan) for e in evs]
        ax.errorbar(x, y, yerr=[1000 * e.get("sigma", 0) for e in evs], fmt="o", color=c, ms=4, label=f"{role} steps")
        for pm in res["asynchrony"][role]["footsteps"].get("per_measure", []):
            if pm["t0"] is None:
                continue
            t0, t1 = pm["t0"], pm["t0"] + res["beats"]["measure_s"]
            ax.fill_between([t0, t1], pm["ci95_ms"][0], pm["ci95_ms"][1], color=c, alpha=.15)
            ax.plot([t0, t1], [pm["mean_ms"]] * 2, color=c, lw=1.5)
    ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("asynchrony (ms) = step - beat")
    ax.set_xlabel("reference time (s)")
    ax.legend(fontsize=8)
    ax.set_title("footstep asynchrony per event (+-sigma) and per-measure mean with 95% CI band")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- validation

def score_touchdowns(res: dict, truth: dict, tol: float = 0.1) -> dict:
    """Recall/precision/median |error| of touchdowns vs truth.json (per role and pooled)."""
    out, errs_all = {}, []
    for role in ROLES:
        det = res["footsteps"].get(role, [])
        tp, n_det, n_true, errs = 0, 0, 0, []
        for foot in FOOT_JOINTS:
            a = np.array([e["t"] for e in det if e["foot"] == foot])
            b = np.array([e["t"] for e in truth["touchdowns"][role] if e["foot"] == foot])
            pairs = match(a, b, tol)
            tp += len(pairs)
            n_det += len(a)
            n_true += len(b)
            errs += [a[i] - b[j] for i, j in pairs]
        errs = np.array(errs)
        errs_all += list(errs)
        out[role] = {"recall": tp / max(n_true, 1), "precision": tp / max(n_det, 1), "n_true": n_true,
                     "n_detected": n_det, "median_abs_err_ms": float(np.median(np.abs(errs)) * 1000) if len(errs) else None,
                     "mean_err_ms": float(np.mean(errs) * 1000) if len(errs) else None,
                     "p90_abs_err_ms": float(np.percentile(np.abs(errs), 90) * 1000) if len(errs) else None}
    e = np.abs(np.array(errs_all))
    out["pooled_median_abs_err_ms"] = float(np.median(e) * 1000) if len(e) else None
    return out


def group_contrast(evs: list[dict], in_group, cfg: TimingConfig, rng) -> dict:
    """Mean asynchrony of events in a group of measures vs the dancer's other events: difference with a CI
    (Monte-Carlo + bootstrap) and an event-label permutation p value (null: measures are exchangeable)."""
    a = np.array([e["async_ms"] for e in evs]) / 1000
    g = np.array([bool(in_group(e["measure"])) for e in evs])
    if g.sum() == 0 or (~g).sum() == 0:
        return {}
    mc = np.stack([e["_async_mc"] for e in evs], 1)
    D = mc.shape[0]
    gi, oi = np.nonzero(g)[0], np.nonzero(~g)[0]
    bg = gi[rng.integers(0, len(gi), (D, len(gi)))]
    bo = oi[rng.integers(0, len(oi), (D, len(oi)))]
    grp = np.nanmean(np.take_along_axis(mc, bg, 1), 1)
    diff = grp - np.nanmean(np.take_along_axis(mc, bo, 1), 1)
    obs = a[g].mean() - a[~g].mean()
    pg = g[np.argsort(rng.random((cfg.n_perm, len(g))), axis=1)]
    perm = (pg * a).sum(1) / g.sum() - (~pg * a).sum(1) / (~g).sum()
    return {"n_group": int(g.sum()), "n_rest": int((~g).sum()), "group_mean_ms": float(a[g].mean() * 1000),
            "group_mean_ci95_ms": _ci(grp * 1000), "diff_ms": float(obs * 1000), "diff_ci95_ms": _ci(diff * 1000),
            "perm_p": float((1 + np.sum(perm >= obs - 1e-12)) / (1 + cfg.n_perm))}


def score_late(res: dict, truth: dict, cfg: TimingConfig, tol: float = 0.1) -> dict:
    """Late-measure check against truth: the follow's late measures vs her other measures (and the lead's same
    measures as a control), per-measure flags, and the asynchrony of the detected truth-late steps themselves."""
    late_s = truth["late_s"] * 1000
    late_mod = set(truth["late_measures_mod8"])
    rng = np.random.default_rng(cfg.seed + 7)
    out = {}
    for role in ROLES:
        fs = res["asynchrony"][role]["footsteps"]
        evs = res["footsteps"][role]
        eff = {m["measure"]: m for m in fs["measure_effects"]["measures"]}
        rows = []
        for pm in fs["per_measure"]:
            m = pm["measure"]
            expect = late_s if (role == "follow" and m % 8 in late_mod) else 0.0
            rows.append({**pm, "expected_ms": expect, "perm_q": eff.get(m, {}).get("q_bh"),
                         "dev_ms": eff.get(m, {}).get("dev_ms")})
        # detections matched to truth steps that are late / on time
        det = np.array([e["t"] for e in evs])
        tr = truth["touchdowns"][role]
        pairs = match(det, np.array([e["t"] for e in tr]), tol)
        late_async = [evs[i]["async_ms"] for i, j in pairs if tr[j]["late_s"] > 0]
        on_async = [evs[i]["async_ms"] for i, j in pairs if tr[j]["late_s"] == 0]
        out[role] = {
            "per_measure": rows, "global_perm_p": fs["measure_effects"]["global_p"],
            "flagged_measures": [r["measure"] for r in rows if r["perm_q"] is not None and r["perm_q"] < .05],
            "late_measures_vs_rest": group_contrast(evs, lambda m: m % 8 in late_mod, cfg, rng),
            "truth_late_steps_async_ms": {"n": len(late_async), "mean": float(np.mean(late_async)) if late_async else None,
                                          "median": float(np.median(late_async)) if late_async else None},
            "truth_on_steps_async_ms": {"n": len(on_async), "mean": float(np.mean(on_async)) if on_async else None,
                                        "median": float(np.median(on_async)) if on_async else None}}
    return out


def validate_synthetic(cfg: TimingConfig | None = None, noise_levels=(0.0, 0.01), work: Path | None = None,
                       extra_noise_seeds: int = 4, extra_mc: int = 50) -> dict:
    """Gate G5 on work/synthetic: touchdown error clean / noisy (+ extra noise seeds for the spread), late
    measures significant vs the permutation null, surrogate chance baseline reported."""
    cfg = cfg or TimingConfig()
    work = work or PACKAGE_ROOT / "work" / "synthetic"
    truth = json.loads((work / "truth.json").read_text())
    report = {"truth": str(work / "truth.json"), "runs": {}, "noise_seeds": {}}
    ts = {r: load_smplx(work / "smplx" / f"{r}.npz")[0] for r in ROLES}
    for nz in noise_levels:
        c = TimingConfig(**asdict(cfg))
        tag = "clean" if nz == 0 else f"noise_{nz * 100:g}cm"
        out_dir = work / "timing" if nz == 0 else work / "timing" / tag
        res = run(work, c, out_dir=out_dir, noise_inject_m=nz, plot=False)
        plot_qa(res, res["_joints"], ts, out_dir / "qa_timing.png", truth)
        late = score_late(res, truth, c)
        report["runs"][tag] = {
            "noise_injected_m": nz, "noise_estimated_m": {r: res["kinematics"][r]["noise_m"] for r in ROLES},
            "touchdowns": score_touchdowns(res, truth), "late": late,
            "footsteps": {r: {k: res["asynchrony"][r]["footsteps"][k] for k in
                              ("n", "mean_ms", "ci95_ms", "R", "V", "hit_rate", "surrogate")} for r in ROLES},
            "follow_minus_lead": res["asynchrony"]["follow_minus_lead"],
            "beats": {k: res["beats"][k] for k in ("grid_measure_bpm", "chance_hit_rate_analytic", "warnings")},
            "cross_check": {r: res["kinematics"][r]["cross_check"] for r in ROLES},
            "out": str(out_dir)}
        if nz > 0:
            meds = [report["runs"][tag]["touchdowns"]["pooled_median_abs_err_ms"]]
            for k in range(1, extra_noise_seeds + 1):
                ck = TimingConfig(**{**asdict(cfg), "seed": cfg.seed + k, "mc_draws": extra_mc})
                rk = run(work, ck, out_dir=work / "timing" / f"{tag}_seed{k}", noise_inject_m=nz, plot=False)
                meds.append(score_touchdowns(rk, truth)["pooled_median_abs_err_ms"])
            report["noise_seeds"][tag] = {"median_abs_err_ms_per_seed": meds,
                                          "median_of_seeds_ms": float(np.median(meds)), "worst_ms": float(np.max(meds))}
    g = {}
    for tag, rr in report["runs"].items():
        td = rr["touchdowns"]
        limit = 20.0 if tag == "clean" else 33.0
        med = td["pooled_median_abs_err_ms"]
        seeds = report["noise_seeds"].get(tag)
        fl, ld = rr["late"]["follow"]["late_measures_vs_rest"], rr["late"]["lead"]["late_measures_vs_rest"]
        rel = [m for m in rr["follow_minus_lead"]["per_measure"] if m["measure"] % 8 in truth["late_measures_mod8"]]
        g[tag] = {
            "median_abs_err_ms": med, "limit_ms": limit, "median_over_noise_seeds_ms": seeds and seeds["median_of_seeds_ms"],
            "touchdown_ok": med is not None and med <= limit and (seeds is None or seeds["median_of_seeds_ms"] <= limit),
            "recall": min(td[r]["recall"] for r in ROLES), "precision": min(td[r]["precision"] for r in ROLES),
            "follow_late_minus_rest_ms": fl.get("diff_ms"), "follow_late_minus_rest_ci95_ms": fl.get("diff_ci95_ms"),
            "follow_late_perm_p": fl.get("perm_p"), "lead_same_measures_perm_p": ld.get("perm_p"),
            "lead_same_measures_diff_ci95_ms": ld.get("diff_ci95_ms"),
            "follow_truth_late_steps_mean_ms": rr["late"]["follow"]["truth_late_steps_async_ms"]["mean"],
            "follow_minus_lead_late_measures": rel,
            "late_ok": bool(fl and fl["perm_p"] < 0.05 and fl["diff_ci95_ms"][0] > 0 and ld and ld["perm_p"] >= 0.05),
            "surrogate_follow": rr["footsteps"]["follow"]["surrogate"]}
    g["G5"] = bool(all(v["touchdown_ok"] and v["late_ok"] for v in g.values()))
    report["gate"] = g
    (work / "timing" / "validation.json").write_text(json.dumps(_clean(report), indent=1))
    return report
