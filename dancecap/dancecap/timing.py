"""Timing: kinematic derivatives, footstep touchdowns, joint accents and their asynchrony to the zouk beat grid.

Inputs (work/<take>/): smplx/{lead,follow}.npz (`t`, `joints` (T,55,3) Y-up metres, floor y=0) and a beats json
in the zouk format ([[seconds, type]], 8 per measure, type 1 = eighths 0/4, 2 = eighths 3/6, 3 = other, exactly as
atlas_bridge/beats.py and beat_this_analyzer.py write it). Output: timing/timing.json (+ kinematics.npz, qa png).

Method (see `notes` in timing.json):
  * positions / velocities on the REAL timestamps with a quintic smoothing spline per coordinate, s = N sigma^2
    from a noise model (sigma from 4th differences unless given); acceleration and jerk from the penalised
    quintic smoother (physics_spline) at a cutoff chosen per derivative order from the noise (choose_cutoffs),
    with per-joint SNR / reliability. Butterworth filtfilt + central differences is a cross-check only. No
    forward differences (they shift jerk ~1.5 frames early).
  * footstep touchdowns from a CONTACT MODEL per foot (ankle = heel proxy + foot/toe joint): stance height,
    vertical + horizontal speed with hysteresis. Sub-frame time: the change point of the step's RAW leg channels
    (heights come down onto the plateau, pelvis-frame positions stop), refined by aligning all steps of a foot to
    their leave-one-out average (matched filter) and placing the touchdown at the change point of the foot's
    pooled, aligned steps, searched around the median per-step estimate. Sigma per event = Monte-Carlo SD with
    the noise model added to the raw data (re-running the whole detector, per-draw common shift kept).
  * key-joint accents: local maxima of |acceleration| and |jerk| (4x oversampled) with prominence, each with
    its joint's derivative SNR; at realistic pose noise most jerk accents are flagged unreliable.
  * each event is assigned to the nearest beat of the allowed zouk counts; asynchrony = event - beat, with
    circular stats (Rayleigh, V-test), Monte-Carlo CIs (pose noise + beat jitter + camera sync), a jitter null
    (independent uniform phase per event) and a circular-shift surrogate for chance hit rate, and two-sided
    permutation tests for measures that deviate (e.g. a dancer stepping late).

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
# leg channels (leg_channels): 3 heights (descent onto the stance plateau) + 6 pelvis-frame horizontal positions
# (any change of velocity / curvature; -1 = linear drift after the change point: a planted foot seen from a moving
# pelvis). Every channel shares one change point. (A symmetric piecewise-quadratic model was tried and biased soft
# landings early by 10-30 ms; the one-sided cubic is biased LATE when its search range is wide, because a cubic can
# absorb a kink placed before tau - hence the pooled search is anchored on the per-step estimates.)
LEG_DIM = 9
LEG_DESCENT = [True] * 3 + [False] * 6
LEG_MODELS = [HINGE_MODELS] * 3 + [((0, -1, 1, 2),)] * 6  # horizontal: quadratic approach, linear drift after


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
    fit_after_s: float = 0.35  # per-step fit window after the first stance frame (soft landings start stance early)
    touchdown_joint: str = "auto"  # auto: ankle (heel/foot-flat) unless the heel stays up, then toe
    noise_k: float = 1.0  # thresholds grow by noise_k * sigma (the spline is much smoother than the raw noise)
    template: bool = True  # refine with the per-foot step-averaged descent template (matched filter)
    template_pre_s: float = 0.3
    template_post_s: float = 0.2
    touchdown_post_s: float = 0.3  # pooled change-point fit: samples up to this long after the touchdown
    touchdown_search_s: tuple = ((-0.06, 0.06), (-0.03, 0.03))  # pooled offset search per pass, around the
    # median of the per-step estimates (pass 1) and then around the previous pass
    template_search_s: tuple = (0.08, 0.04, 0.02)
    template_min_steps: int = 3  # per foot (a 6 s window at 76-88 BPM has 3-5 steps per foot)
    template_apex_ratio: float = 2.0  # steps whose swing apex is outside [1/r, r] x the foot's median: no template
    # touchdown <= first stance frame + this (None = off). The contact model enters stance 45-70 ms BEFORE a soft
    # landing ends (foot low and slow), so a one-frame clamp biased clean touchdowns 20-35 ms early; 0.15 s only
    # catches events the template pushed well past the observed contact.
    clamp_after_stance_s: float | None = 0.15
    plausible_lag_ms: tuple = (-60.0, 160.0)  # per-foot median (touchdown - first stance frame); outside -> suspect
    template_knot_s: float = 0.033
    template_scale_sd: float = 0.15  # prior SD of a step's descent amplitude relative to the template
    # accents
    accent_joints: dict = field(default_factory=lambda: dict(ACCENT_JOINTS))
    accent_oversample: int = 4
    accent_min_sep_s: float = 0.15
    accent_prominence: float = 0.5  # x (p90 - p10) of the magnitude signal
    # higher derivatives (acc / jerk): penalised quintic smoother, cutoff = predicted-MSE optimum per order
    deriv_cutoff_range_hz: tuple = (1.5, 10.0)
    deriv_snr_min: float = 2.0  # reliable: signal RMS / noise RMS of the derivative >= this (noise <= 20 % power)
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
    # per-camera residual sync uncertainty (SD) and number of cameras averaged. None = from sync/offsets.json when
    # it exists (run()), else 5 ms / 8; explicit values always win and are never overwritten
    sync_sigma_s: float | None = None
    n_cams: int | None = None
    sync_source: str = "config"
    n_surrogates: int = 2000
    n_perm: int = 5000
    match_tol_s: float = 0.08
    mc_shift_tol_s: float = 0.2  # loose matching that measures each MC draw's common shift first
    # optional low-frequency pose error added in the MC on top of the white noise model: fused multi-view fits
    # (MAMMA) are smooth, so the 4th-difference estimate sees ~0.5-1 mm while the real error is cm-level and
    # correlated over ~0.1 s; 0 = off (synthetic validation), set it for real takes as a stated assumption
    mc_corr_noise_m: float = 0.0
    mc_corr_time_s: float = 0.1
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


def _spectrum(t: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean Hann-windowed periodogram over the columns of X (T, C) on the median frame interval, linear trend
    removed -> (f Hz, P) normalised so that white noise of SD sigma has P = sigma^2 in every bin."""
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1] + 1e-9, dt)
    Y = np.stack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])], 1)
    A = np.stack([np.ones(len(tu)), tu - tu.mean()], 1)
    Y = Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    w = np.hanning(len(tu))
    F = np.fft.rfft(Y * w[:, None], axis=0)
    return np.fft.rfftfreq(len(tu), dt), (np.abs(F) ** 2).mean(1) / (w ** 2).sum()


def choose_cutoffs(t: np.ndarray, joints: np.ndarray, idx, noise_m: float, cfg: "TimingConfig",
                   fixed: dict | None = None) -> dict:
    """Smoothing for the higher derivatives, chosen PER JOINT and per derivative order from the data and the noise.

    The position spline (s = N sigma^2, per coordinate) adapts its smoothness to each coordinate but can leave
    spurious acceleration / jerk peaks under realistic pose noise (synthetic, 1 cm: a near-still wrist got |jerk|
    peaks of 27-48 m/s^3 against 1 in truth). For acc (nu=2) and jerk (nu=3) each joint uses the penalised quintic
    Smoother (physics_spline: zero phase, linear, gain H = 1 / (1 + (f/fc)^6)) at the cutoff that minimises the
    predicted mean squared error of its nu-th derivative,
        MSE(fc) = sum_f (2 pi f)^(2 nu) [ (1 - H)^2 S(f) + H^2 sigma^2 ],
    S = (5-bin smoothed mean periodogram of the joint's 3 coordinates) - sigma^2 in the bins clearly above the
    white-noise floor that form the contiguous low-frequency signal band (gaps <= 3 bins), else 0; searched over cfg.deriv_cutoff_range_hz (a joint without detectable signal gets the lowest cutoff). Zero
    phase: peak TIMES are kept; peaks are attenuated / widened at low cutoffs (at 30 fps the sharp foot-landing jerk
    of the synthetic, 116-213 m/s^3, is not recoverable by any cutoff even at 3 mm noise).
    Reliability per joint: SNR = sqrt(signal power / noise power) of the smoothed derivative (noise power exact for
    the linear smoother: 3 sigma^2 mean(noise_gain^2); signal = total - noise); reliable = SNR >= deriv_snr_min.
    fixed = {"acc"|"jerk": {joint: fc}} skips the search. Returns {"acc"|"jerk": {"joints": idx, "cutoff_hz",
    "snr", "reliable" (lists over idx), "noise_sd" (T, len(idx)) per coordinate, "snr_median", "reliable_fraction",
    "cutoff_hz_median"}}."""
    from .physics_spline import Smoother
    idx = [int(j) for j in idx]
    lo, hi = cfg.deriv_cutoff_range_hz
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 40))
    cache = {}

    def smoother(fc):
        if fc not in cache:
            cache[fc] = Smoother(t, fc)
        return cache[fc]

    out = {}
    for name, nu in (("acc", 2), ("jerk", 3)):
        out[name] = {"joints": idx, "cutoff_hz": [], "snr": [], "reliable": [], "noise_sd": []}
    for j in idx:
        X = joints[:, j]
        f, P = _spectrum(t, X)
        # 5-bin running mean, then only bins clearly above the floor (~4 SD of that mean for white noise): the
        # (2 pi f)^(2 nu) weight would otherwise turn single floor fluctuations at high f into 'signal'
        Ps = np.convolve(np.pad(P, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        above = Ps > noise_m ** 2 * (1 + 4 * np.sqrt(1.5 / 15))
        # the signal band: contiguous from the lowest frequencies (gaps <= 3 bins); isolated high bins are ignored
        edge, gap = 0, 0
        for i in range(1, len(f)):
            if above[i]:
                edge, gap = i, 0
            else:
                gap += 1
                if gap > 3:
                    break
        Ssig = np.where(above & (np.arange(len(f)) <= edge), Ps - noise_m ** 2, 0.0)
        for name, nu in (("acc", 2), ("jerk", 3)):
            if fixed is not None:
                fc = float(fixed[name][j])
            else:
                wgt = (2 * np.pi * f) ** (2 * nu)
                H = 1.0 / (1.0 + (f[None, :] / grid[:, None]) ** 6)
                mse = ((1 - H) ** 2 * Ssig[None] + H ** 2 * noise_m ** 2) @ wgt
                fc = float(grid[int(np.argmin(mse))])
            S = smoother(fc)
            g = S.noise_gain(nu)
            noise_p = 3 * noise_m ** 2 * float(np.mean(g ** 2))
            tot = float((S(X, nu) ** 2).sum(-1).mean())
            snr = math.sqrt(max(tot - noise_p, 0.0) / max(noise_p, 1e-30))
            o = out[name]
            o["cutoff_hz"].append(round(fc, 3))
            o["snr"].append(snr)
            o["reliable"].append(bool(snr >= cfg.deriv_snr_min))
            o["noise_sd"].append(noise_m * g)
    for o in out.values():
        o["noise_sd"] = np.stack(o["noise_sd"], 1)
        o["snr_median"] = float(np.median(o["snr"]))
        o["reliable_fraction"] = float(np.mean(o["reliable"]))
        o["cutoff_hz_median"] = float(np.median(o["cutoff_hz"]))
    return out


_OP_CACHE: dict = {}


def _deriv_operator(t: np.ndarray, fc: float, nu: int, te: np.ndarray) -> np.ndarray:
    """(len(te), N) Smoother derivative operator, cached (the Monte-Carlo re-uses the same times and cutoffs)."""
    from .physics_spline import Smoother
    key = (hash(t.tobytes()), len(t), float(fc), nu, hash(te.tobytes()), len(te))
    op = _OP_CACHE.get(key)
    if op is None:
        if len(_OP_CACHE) > 64:
            _OP_CACHE.clear()
        op = _OP_CACHE[key] = Smoother(t, fc).operator(nu, te)
    return op


def apply_cutoffs(t: np.ndarray, joints: np.ndarray, cut: dict, kins: tuple) -> None:
    """Overwrite acc / jerk of the joints in `cut` in every Kinematics of `kins` (at their own te) with the
    per-joint Smoother derivatives."""
    for name, nu in (("acc", 2), ("jerk", 3)):
        c = cut[name]
        by_fc = {}
        for j, fc in zip(c["joints"], c["cutoff_hz"]):
            by_fc.setdefault(fc, []).append(j)
        for fc, js in by_fc.items():
            X = joints[:, js].reshape(len(t), -1)
            for K in kins:
                op = _deriv_operator(np.asarray(t, float), fc, nu, np.asarray(K.te, float))
                getattr(K, name)[:, js] = (op @ X).reshape(len(K.te), len(js), 3)
        for K in kins:
            K.method += f"; {name}: per-joint penalised quintic smoother (median fc {c['cutoff_hz_median']:g} Hz)"


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


def hinge_sse(t: np.ndarray, y: np.ndarray, taus: np.ndarray, descent=True, models=HINGE_MODELS) -> np.ndarray:
    """SSE (Ntau,) of the best hinge model per channel, summed over channels, for every candidate tau.

    Columns of a model are powers of u = max(0, tau - t) (0 = constant, 1..3 = the approach before tau) and -1 =
    v = max(0, t - tau), a linear drift AFTER tau (a planted foot seen in a moving frame). descent (bool or (P,)
    bools) requires the u and u^2 coefficients of that channel to be >= 0 (a height coming DOWN onto its plateau);
    channels without the constraint (horizontal positions) locate tau by a change of velocity or curvature."""
    y = y.reshape(len(t), -1)
    desc = np.broadcast_to(np.asarray(descent, bool), (y.shape[1],))
    u = np.maximum(0.0, taus[:, None] - t[None, :])  # (Ntau, n)
    v = np.maximum(0.0, t[None, :] - taus[:, None])
    cols_all = {0: np.ones_like(u), 1: u, 2: u * u, 3: u ** 3, -1: v, -2: v * v}
    total = np.zeros(len(taus))
    for p in range(y.shape[1]):
        best = np.full(len(taus), np.inf)
        for cols in (models[p] if isinstance(models, list) else models):
            if len(cols) >= len(t):
                continue
            A = np.stack([cols_all[c] for c in cols], -1)  # (Ntau, n, q)
            At = A.transpose(0, 2, 1)
            AtA = At @ A + 1e-12 * np.eye(len(cols))
            Aty = At @ y[:, p]
            try:
                coef = np.linalg.solve(AtA, Aty[..., None])[..., 0]
            except np.linalg.LinAlgError:
                continue
            sse = (((A @ coef[..., None])[..., 0] - y[:, p]) ** 2).sum(1)
            if desc[p]:  # one-sided models: u and u^2 >= 0; two-sided (with v terms): only u >= 0
                lin = [i for i, c in enumerate(cols) if c in ((1,) if min(cols) < 0 else (1, 2))]
                ok = (coef[:, lin] >= 0).all(1) & (coef[:, lin].sum(1) > 0)
            else:
                ok = np.abs(coef[:, 1:]).sum(1) > 0
            best = np.where(ok & (sse < best), sse, best)
        total += best
    return total


def _argmin_refined(taus: np.ndarray, total: np.ndarray) -> tuple[float, float]:
    """Grid argmin with parabolic refinement -> (tau, SSE at the grid minimum); NaN when nothing fitted."""
    if not np.isfinite(total).any():
        return float("nan"), float("nan")
    k = int(np.nanargmin(np.where(np.isfinite(total), total, np.nan)))
    tau = taus[k]
    if 0 < k < len(taus) - 1 and np.isfinite(total[k - 1]) and np.isfinite(total[k + 1]):
        den = total[k - 1] - 2 * total[k] + total[k + 1]
        if den > 0:
            tau += (taus[1] - taus[0]) * 0.5 * (total[k - 1] - total[k + 1]) / den
    return float(tau), float(total[k])


def hinge_touchdown(t: np.ndarray, y: np.ndarray, t_lo: float, t_hi: float, step: float = 0.001,
                    descent=True, models=HINGE_MODELS) -> tuple[float, float]:
    """Fit y = a + b u + c u^2 + d u^3 with u = max(0, tau - t): a polynomial approach onto a plateau -> (tau, rms).

    tau is where the approach ends: the height reaches the stance plateau and the vertical velocity reaches zero
    (b = 0, soft landing) or jumps to zero (b != 0, impact); the cubic absorbs the swing's curvature change.
    y may be (n,) or (n, P): P channels share tau, each picks its best polynomial (see hinge_sse; models may be a
    per-channel list). descent=True requires b, c >= 0 (coming down onto the plateau). Exhaustive `step` search +
    parabolic refinement of the SSE minimum."""
    y = y.reshape(len(t), -1)
    taus = np.arange(t_lo, t_hi + step / 2, step)
    if len(taus) < 3 or len(t) < 5:
        return float("nan"), float("nan")
    tau, sse = _argmin_refined(taus, hinge_sse(t, y, taus, descent, models))
    return tau, float(math.sqrt(sse / y.size)) if np.isfinite(sse) else float("nan")


def _lsq_template(U: np.ndarray, Y: np.ndarray, lo: float, hi: float, knot_s: float):
    """Least-squares cubic spline per column on fixed knots (a regression: it cannot chase one step's noise)."""
    from scipy.interpolate import LSQUnivariateSpline
    o = np.argsort(U, kind="stable")
    U, Y = U[o] + np.arange(len(U)) * 1e-9, Y[o].reshape(len(U), -1)
    # Few pooled steps (LOO with 2-3 steps, sparse 30 Hz samples) can leave knot intervals without data
    # (Schoenberg-Whitney fails in FITPACK, seen in the real-take MC with correlated noise): widen the knot spacing
    # until the fit is accepted; last resort one cubic over the window.
    for mult in (1, 2, 3, 4, 6, 8):
        ks = knot_s * mult
        inner = np.arange(lo + ks, hi - ks / 2, ks)
        inner = inner[(inner > U[0]) & (inner < U[-1])]
        try:
            return [LSQUnivariateSpline(U, Y[:, p], inner, k=3) for p in range(Y.shape[1])]
        except ValueError:
            continue
    return [LSQUnivariateSpline(U, Y[:, p], [], k=3) for p in range(Y.shape[1])]


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


def template_touchdown(t: np.ndarray, groups: list, cfg: TimingConfig, search=(-0.15, 0.12)) -> tuple[float, dict]:
    """Where the touchdown sits relative to the aligned steps: the shared change point of the pooled, aligned
    leg channels of all groups [(Y (T, 9) leg channels, taus, spans)] (both feet of a dancer).

    Per foot and channel one hinge polynomial is fitted to all of that foot's aligned, plateau-referenced steps;
    the offset minimises the summed SSE over feet and channels. Heights must come down onto their plateau (b, c
    >= 0); horizontal (pelvis-frame) channels may drift after the change point, so a foot that stops moving
    horizontally (velocity kink) locates the touchdown even when the vertical landing is soft (zero vertical
    velocity at contact makes the height corner nearly invisible under 1 cm noise). Returns (offset s, info)."""
    lo, hi = search
    grid = np.arange(lo, hi + 5e-4, 0.001)
    total = np.zeros(len(grid))
    n = 0
    for Y, taus, spans in groups:
        _, us, ys = _step_samples(t, Y, taus, spans, cfg.template_pre_s, cfg.touchdown_post_s)
        U, YY = np.concatenate(us), np.concatenate(ys)
        if len(U) < 12:
            continue
        total += hinge_sse(U, YY, grid, LEG_DESCENT, LEG_MODELS)
        n += len(U)
    if n == 0:
        return 0.0, {"n_samples": 0}
    tau0, _ = _argmin_refined(grid, total)
    # curvature of the SSE at the minimum -> a rough SD of the offset (SSE/n per residual variance)
    k = int(np.nanargmin(total)) if np.isfinite(total).any() else 0
    sd = float("nan")
    if 2 <= k < len(grid) - 2 and np.all(np.isfinite(total[k - 2:k + 3])):
        h = grid[1] - grid[0]
        curv = (total[k - 2] - 2 * total[k] + total[k + 2]) / (4 * h * h)
        s2 = total[k] / max(n * LEG_DIM - 1, 1)
        if curv > 0:
            sd = float(math.sqrt(2 * s2 / curv))
    at_edge = not (lo + 0.003 < tau0 < hi - 0.003) if np.isfinite(tau0) else True
    return (float(tau0) if np.isfinite(tau0) else 0.0), {"n_samples": int(n), "sd_ms": 1000 * sd,
                                                          "at_search_edge": bool(at_edge)}


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
                      fine: Kinematics | None = None, info: dict | None = None) -> list[dict]:
    """Touchdowns for both feet: contact intervals with hysteresis, then sub-frame time per event.

    Per foot: stance height of ankle and toe (lowest quarter), foot height H = min over parts, contact when H is
    low AND the ankle's vertical speed is small AND the foot is not sweeping fast; leave contact on clear lift or
    fast horizontal motion (all on the smoothing spline). A touchdown is the start of a stance preceded by a swing
    with real clearance. Its time: the shared change point of the RAW leg channels of that step (multi-channel
    hinge: heights come down onto their plateau, pelvis-frame positions stop / change velocity), then refined by
    aligning the foot's steps to their leave-one-out average (matched filter) and placing the touchdown at the
    change point of that foot's pooled aligned steps (template_touchdown, searched around the median of the
    per-step change points). Steps left out of the template get the dancer's median template-minus-hinge offset.
    A touchdown never lies more than clamp_after_stance_s after the first frame the contact model calls stance
    (flag clamped_to_stance). `info` (dict) receives the per-foot template diagnostics."""
    info = {} if info is None else info
    dt = float(np.median(np.diff(t)))
    events, by_foot, chans = [], {}, {}
    for foot, parts in FOOT_JOINTS.items():
        Yf = chans[foot] = leg_channels(joints, kin, foot)
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
            heel_up = np.median(h[s:e, 0]) > h_on  # releve: the heel stays up through stance -> toe joint
            which = cfg.touchdown_joint if cfg.touchdown_joint != "auto" else ("toe" if heel_up else "ankle")
            i_hi = min(e, s + max(2, int(round(cfg.fit_after_s / dt)))) - 1
            res = {}
            for name, col in (("ankle", 0), ("toe", 1)):  # per-joint height hinges on the RAW heights (diagnostic)
                hj = h[:, col]
                ia = sw0 + int(np.argmax(hj[sw0:s]))
                above = np.nonzero(hj[ia:s] > cfg.fit_upper_frac * hj[ia])[0]
                i_lo = ia + (int(above[-1]) if len(above) else 0)
                res[name] = hinge_touchdown(t[i_lo:i_hi + 1], joints[i_lo:i_hi + 1, parts[name], 1], t[i_lo], t[i_hi])
            # the step's own change point: all raw leg channels from the swing apex (<= 0.3 s before stance)
            ia = sw0 + int(np.argmax(H[sw0:s]))
            i_lo = max(ia, s - int(round(cfg.template_pre_s / dt)))
            lo = max(t[i_lo] + 0.05, t[s] - 0.15)
            hi = min(t[i_hi] - 0.05, t[s] + 0.1)
            t_td, rms = hinge_touchdown(t[i_lo:i_hi + 1], Yf[i_lo:i_hi + 1], lo, hi, descent=LEG_DESCENT,
                                        models=LEG_MODELS) if hi - lo > 0.01 else (float("nan"), float("nan"))
            if not np.isfinite(t_td):
                t_td, rms = res[which]
            if not np.isfinite(t_td):
                continue
            ev = {"t": float(t_td), "foot": foot, "joint": which, "method": "hinge", "t_hinge": float(t_td),
                  "t_ankle": res["ankle"][0], "t_toe": res["toe"][0], "fit_rms_mm": 1000 * rms,
                  "stance": [float(t[s]), float(t[e - 1])], "apex_m": float(apex), "_rows": (sw0, e)}
            if fine is not None:  # spline cross-check: vertical velocity of the ankle back to ~zero
                fs = int(np.searchsorted(fine.te, t[sw0]))
                fe = int(np.searchsorted(fine.te, min(t[e - 1], t[s] + cfg.fit_after_s)))
                ev["t_spline_vy0"] = _vy_zero(fine.te, fine.vel[:, ja, 1], fs, fe, eps=0.02 + 2 * noise_m)
            foot_ev.append(ev)
        by_foot[foot] = foot_ev
    info.update(n_template=0, n_fallback=0, template_feet={})
    if cfg.template:
        groups = []
        for foot, all_ev in by_foot.items():
            # the template assumes steps of one kind: a kick or a much smaller / larger step (swing apex outside
            # [1/r, r] x the foot's median) is left out (it gets the hinge + offset time)
            med = float(np.median([e["apex_m"] for e in all_ev])) if all_ev else 0.0
            r = cfg.template_apex_ratio
            foot_ev = [e for e in all_ev if med / r <= e["apex_m"] <= med * r]
            for e in all_ev:
                if e not in foot_ev:
                    e["template_excluded"] = "apex"
            if len(foot_ev) < cfg.template_min_steps:
                continue
            spans = [e["_rows"] for e in foot_ev]
            tt, rms = align_steps(t, chans[foot], np.array([e["t"] for e in foot_ev]), spans, cfg, noise_m)
            ok = [i for i, (x, e) in enumerate(zip(tt, foot_ev)) if np.isfinite(rms[i])
                  and abs(x - e["t_hinge"]) < 2 * max(cfg.template_search_s)]
            for i in ok:
                foot_ev[i].update(_aligned=float(tt[i]), template_rms_mm=1000 * float(rms[i]))
            sel = [foot_ev[i] for i in ok]
            if len(sel) >= 2:
                groups.append((foot, chans[foot], np.array([e["_aligned"] for e in sel]), [e["_rows"] for e in sel],
                               sel))
        # one offset PER FOOT: the alignment fixes only relative times inside a foot, so each foot's aligned train
        # carries its own arbitrary common shift (the mean of its initial per-step estimates)
        info["template_feet"] = {}
        for foot, Y, taus, sp, sel in groups:
            # anchor: the median of this foot's per-step change points (robust, ~20 ms SD at 1 cm noise for 4 steps);
            # the pooled fit refines it within touchdown_search_s (an unanchored search drifts late, see LEG_MODELS)
            anchor = float(np.median([e["t_hinge"] - e["_aligned"] for e in sel]))
            tau0, tinfo = anchor, {}
            for rng_s in cfg.touchdown_search_s:  # later passes re-centre on the touchdown found so far
                d, tinfo = template_touchdown(t, [(Y, taus + tau0, sp)], cfg, rng_s)
                tau0 += d
            if tinfo.get("at_search_edge"):  # no interior minimum: keep the per-step anchor
                tau0 = anchor
            info["template_feet"][foot] = {"offset_ms": 1000 * tau0, "anchor_ms": 1000 * anchor,
                                           "offset_sd_ms": tinfo.get("sd_ms"),
                                           "at_search_edge": tinfo.get("at_search_edge"), "n": len(sel)}
            for e in sel:
                e.update(t=e.pop("_aligned") + tau0, method="template", template_offset_ms=1000 * tau0)
    # steps left out of the template: same estimator family, shifted by the dancer's median template-minus-hinge
    allev = [e for fe in by_foot.values() for e in fe]
    tpl = [e["t"] - e["t_hinge"] for e in allev if e["method"] == "template"]
    off = float(np.median(tpl)) if tpl else 0.0
    for e in allev:
        if e["method"] != "template":
            e.update(t=e["t_hinge"] + off, method="hinge+offset" if tpl else "hinge", fallback_offset_ms=1000 * off)
    info.update(n_template=len(tpl), n_fallback=len(allev) - len(tpl))
    # never later than clamp_after_stance_s after the first frame the contact model already calls stance
    n_cl = 0
    for e in allev if cfg.clamp_after_stance_s is not None else []:
        lim = e["stance"][0] + cfg.clamp_after_stance_s
        if e["t"] > lim:
            e.update(t_unclamped=e["t"], t=lim, clamped_to_stance=True)
            n_cl += 1
    info["n_clamped"] = n_cl
    lag = [e["stance"][0] - e["t"] for e in allev]
    info["stance_start_minus_touchdown_ms"] = float(np.median(lag) * 1000) if lag else None
    # plausibility per foot: the contact model enters stance while a soft landing ends (touchdown ~0-100 ms after
    # the first stance frame on the synthetic take, clean or 1 cm noise). A foot whose touchdowns sit far outside
    # that is flagged: its times are not trustworthy. (at_search_edge alone is not: the per-step anchor it falls
    # back to is fine on clean data, e.g. the unit-test walker.)
    lo_ms, hi_ms = cfg.plausible_lag_ms
    info["suspect_feet"] = []
    for foot, fe in by_foot.items():
        if not fe:
            continue
        lag_ms = float(np.median([e["t"] - e["stance"][0] for e in fe]) * 1000)
        tf = info["template_feet"].get(foot)
        why = []
        if not lo_ms <= lag_ms <= hi_ms:
            why.append(f"median touchdown - stance start {lag_ms:+.0f} ms outside [{lo_ms:g}, {hi_ms:g}]")
        if tf is not None:
            tf["touchdown_minus_stance_ms"] = lag_ms
        if why:
            info["suspect_feet"].append({"foot": foot, "reasons": why})
            for e in fe:
                e["suspect"] = True
    for foot_ev in by_foot.values():
        for e in foot_ev:
            e.pop("_rows")
            e.pop("_aligned", None)
        events += foot_ev
    events.sort(key=lambda d: d["t"])
    return events


# ----------------------------------------------------------------------------------------------- accents

def detect_accents(fine: Kinematics, cfg: TimingConfig, cut: dict | None = None) -> dict:
    """Local maxima of |acc| and |jerk| per key joint on the oversampled smoother, with prominence + separation.
    With `cut` (choose_cutoffs over cfg.accent_joints) every event carries that joint's derivative SNR and
    `reliable` (SNR >= cfg.deriv_snr_min): peaks of a noise-dominated signal are mostly noise."""
    from scipy.signal import find_peaks
    dte = float(np.median(np.diff(fine.te)))
    out = {}
    for name, j in cfg.accent_joints.items():
        out[name] = {}
        for kind, arr in (("acc", fine.acc), ("jerk", fine.jerk)):
            snr = cut[kind]["snr"][cut[kind]["joints"].index(j)] if cut and j in cut[kind]["joints"] else None
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
                ev = {"t": float(tp), "value": float(mag[p]), "prominence": float(prom)}
                if snr is not None:
                    ev.update(snr=float(snr), reliable=bool(snr >= cfg.deriv_snr_min))
                evs.append(ev)
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
    """Circular mean asynchrony (s), resultant length R, V = mean cos(phase), Rayleigh test (any preferred phase)
    and V-test (phase locked AT the beat, one-sided). Phases are 2 pi asyn / local beat period."""
    if len(asyn) == 0:
        return {"n": 0}
    ph = 2 * np.pi * asyn / period
    C, S = np.cos(ph).mean(), np.sin(ph).mean()
    R = float(math.hypot(C, S))
    n = len(asyn)
    Z = n * R * R
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - n * n * R * R)) - (1 + 2 * n))  # Zar's Rayleigh approximation
    u = math.sqrt(2 * n) * C  # V-test (Durand & Greenwood): u ~ N(0, 1) under uniform phases
    return {"circ_mean_ms": float(math.atan2(S, C) / (2 * np.pi) * np.median(period) * 1000), "R": R, "V": float(C),
            "circ_std_ms": float(math.sqrt(max(0.0, -2 * math.log(max(R, 1e-12)))) / (2 * np.pi)
                                 * np.median(period) * 1000), "rayleigh_Z": float(Z), "rayleigh_p": float(min(1.0, p)),
            "v_test_u": float(u), "v_test_p": float(0.5 * math.erfc(u / math.sqrt(2)))}


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
    with warnings.catch_warnings():  # draws in which every resampled event was missed -> NaN, skipped by _ci
        warnings.simplefilter("ignore", RuntimeWarning)
        bm = np.take_along_axis(mc, boot, 1)
        means = np.nanmean(bm, 1) * 1000
        hits = np.nanmean(np.where(np.isfinite(bm), np.abs(bm) <= cfg.hit_tol_s, np.nan), 1)
        sd_meas = float(np.nanstd(np.nanmean(mc, 1)) * 1000)
    out = {"n": int(n), "mean_ms": float(nom.mean() * 1000), "ci95_ms": _ci(means),
           "sd_measurement_ms": sd_meas, "median_ms": float(np.median(nom) * 1000),
           "hit_rate": float(np.mean(np.abs(nom) <= cfg.hit_tol_s)), "hit_rate_ci95": _ci(hits)}
    out.update(circular(nom, period))
    return out


def surrogate(times: np.ndarray, grid: BeatGrid, cfg: TimingConfig, rng, observed: dict) -> dict:
    """Chance baselines for hit rate and V = mean cos(phase) (one-sided: more hits / more locked than chance).

    Primary (hit_rate_p, V_p): JITTER null - every event is moved independently and uniformly within the inter-beat
    cell of its assigned allowed beat (-gap_before/2 .. +gap_after/2), i.e. event phases independent and uniform
    relative to the grid (no relation to the music). It can reject chance for a precise on-beat dancer: n perfect
    hits have probability ~ (2 tol / spacing)^n under it (8 steps at 88 BPM, every eighth allowed: ~5e-5).
    Caveats: it assumes independent events; a dancer keeping her OWN steady tempo close to the music's would also
    reject over a short window (6 s: a 2 % tempo mismatch drifts only ~0.1 s), so a rejection means 'phase-locked
    to the grid in this window', not proof of entrainment; the common A/V offset is not part of it (use the
    absolute_timing caveat). Secondary (shift_*): circular shift of the whole event train by U(0, 1 measure),
    which keeps the train's own regularity; for steady stepping at the grid tempo a shift by one allowed spacing
    restores every hit, so its p cannot go below p_floor_isochronous = 2 tol / spacing (reported for reference)."""
    if len(times) == 0:
        return {}
    ai = np.nonzero(grid.allowed(cfg))[0]
    al = grid.t[ai]
    gaps = np.diff(al)
    n_s = cfg.n_surrogates
    # jitter null
    bi, _ = assign_beats(times, grid, cfg)
    k = np.searchsorted(ai, bi)
    edge_gap = (gaps[0], gaps[-1]) if len(gaps) else (1.0, 1.0)
    g_prev = np.where(k > 0, al[k] - al[np.maximum(k - 1, 0)], edge_gap[0])
    g_next = np.where(k < len(al) - 1, al[np.minimum(k + 1, len(al) - 1)] - al[k], edge_gap[1])
    a_j = rng.uniform(-g_prev / 2, g_next / 2, (n_s, len(times)))  # uniform over the beat's (Voronoi) cell
    per_j = 0.5 * (g_prev + g_next)
    hits_j = np.mean(np.abs(a_j) <= cfg.hit_tol_s, 1)
    v_j = np.mean(np.cos(2 * np.pi * a_j / per_j), 1)
    chance_i = np.minimum(cfg.hit_tol_s, g_prev / 2) / (0.5 * (g_prev + g_next))         + np.minimum(cfg.hit_tol_s, g_next / 2) / (0.5 * (g_prev + g_next))
    # circular-shift surrogate (secondary)
    lo, hi = al[0], al[-1]
    span = hi - lo
    meas = float(np.median(np.diff(grid.t[::MEASURE_LEN]))) if len(grid.t) > MEASURE_LEN else span
    ts = lo + (times[None, :] - lo + rng.uniform(0, meas, (n_s, 1))) % span
    bs, per = assign_beats(ts, grid, cfg)
    a = ts - grid.t[bs]
    hits = np.mean(np.abs(a) <= cfg.hit_tol_s, 1)
    vs = np.mean(np.cos(2 * np.pi * a / per), 1)
    floor = float(np.minimum(2 * cfg.hit_tol_s / np.median(gaps), 1.0)) if len(gaps) else float("nan")
    pv = lambda null, obs: float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null)))
    return {"null": "jitter: independent uniform phase per event (see note)",
            "hit_rate_mean": float(hits_j.mean()), "hit_rate_p95": float(np.percentile(hits_j, 95)),
            "hit_rate_p": pv(hits_j, observed["hit_rate"]), "chance_per_event_mean": float(chance_i.mean()),
            "V_mean": float(v_j.mean()), "V_p95": float(np.percentile(v_j, 95)), "V_p": pv(v_j, observed["V"]),
            "shift_hit_rate_mean": float(hits.mean()), "shift_hit_rate_p": pv(hits, observed["hit_rate"]),
            "shift_V_mean": float(vs.mean()), "shift_V_p": pv(vs, observed["V"]), "p_floor_isochronous": floor,
            "one_sided": True,
            "note": "hit_rate_p / V_p: jitter null (independent uniform phase per event inside its beat cell), "
                    "one-sided. shift_*: circular shift of the whole train, cannot go below p_floor_isochronous "
                    "for steady stepping. Rayleigh / V-test p values are in the parent dict. All tests assume "
                    "independent events and say nothing about the A/V offset."}


def measure_effects(asyn: np.ndarray, measure: np.ndarray, cfg: TimingConfig, rng) -> dict:
    """Permutation test (two-sided): does a measure's mean asynchrony differ from the mean of the dancer's OTHER
    measures (dev_ms = mean(measure) - mean(rest))? BH-corrected q per measure + a global F-like test."""
    ms = np.unique(measure)
    if len(ms) < 2 or len(asyn) < 4:
        return {"measures": [], "global_p": float("nan")}
    n = len(asyn)
    onehot = (measure[:, None] == ms[None, :]).astype(float)  # (n, M)
    cnts = onehot.sum(0)

    def dev(x):
        tot = x.sum(-1, keepdims=True)
        sm = x @ onehot
        return sm / cnts - (tot - sm) / np.maximum(n - cnts, 1)

    obs = dev(asyn)
    overall = asyn.mean()
    F_obs = float(np.sum(cnts * (asyn @ onehot / cnts - overall) ** 2))
    perm = asyn[np.argsort(rng.random((cfg.n_perm, n)), axis=1)]  # permuting values == permuting labels
    d = dev(perm)  # (n_perm, M)
    cnt = (np.abs(d) >= np.abs(obs) - 1e-12).sum(0)
    F_cnt = int((np.sum(cnts * (perm @ onehot / cnts - overall) ** 2, 1) >= F_obs - 1e-15).sum())
    p = (1 + cnt) / (1 + cfg.n_perm)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return {"measures": [{"measure": int(m), "n": int(c), "dev_ms": float(dd * 1000), "p": float(pp),
                          "q_bh": float(min(1.0, qq))} for m, c, dd, pp, qq in zip(ms, cnts, obs, p, q)],
            "global_p": float((1 + F_cnt) / (1 + cfg.n_perm)), "two_sided": True}


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


def events_for(t: np.ndarray, joints: np.ndarray, noise_m: float, cfg: TimingConfig, cut: dict | None = None):
    """Touchdowns + accents of one dancer (one position spline evaluated at the samples and 4x oversampled; acc /
    jerk from the Smoother at the cutoffs `cut`, chosen from the noise when None).
    Returns (touchdowns, accents, kin, touchdown info, cut)."""
    te = np.linspace(t[0], t[-1], (len(t) - 1) * cfg.accent_oversample + 1)
    kin, fine = spline_kinematics(t, joints, noise_m, _needed(cfg), te=(t, te), k=cfg.spline_k)
    if cut is None:
        cut = choose_cutoffs(t, joints, _needed(cfg), noise_m, cfg)
    apply_cutoffs(t, joints, cut, (kin, fine))
    info = {}
    tds = detect_touchdowns(t, joints, kin, noise_m, cfg, fine, info)
    return tds, detect_accents(fine, cfg, cut), kin, info, cut


def _mc_chunk(t: np.ndarray, base: np.ndarray, noise_m: float, cfg: TimingConfig, cut: dict, seeds: list[int]) -> list:
    """Monte-Carlo draws (worker): Gaussian joint noise of the noise model added to the RAW joints (noise
    addition: for a near-linear estimator the spread of theta(x + e) given x equals the sampling spread of theta,
    and the draws are centred on the data, not on the rounder smooth fit) -> compact event times."""
    out = []
    kern = None
    if cfg.mc_corr_noise_m > 0:  # low-frequency (temporally correlated) pose error, e.g. calibration / shape errors
        dt = float(np.median(np.diff(t)))
        sig = max(cfg.mc_corr_time_s / dt, 0.5)
        x = np.arange(-int(4 * sig) - 1, int(4 * sig) + 2)
        kern = np.exp(-0.5 * (x / sig) ** 2)
        kern /= np.sqrt((kern ** 2).sum())  # unit SD after filtering unit white noise
    for sd in seeds:
        g = np.random.default_rng(sd)
        jd = base + g.normal(0, noise_m, base.shape)
        if kern is not None:
            w = g.normal(0, 1, (len(t) + len(kern) - 1,) + base.shape[1:])
            c = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="valid"), 0, w)
            jd = jd + cfg.mc_corr_noise_m * c
        tds, acc, _, _, _ = events_for(t, jd, noise_m, cfg, cut)
        out.append(({f: np.array([e["t"] for e in tds if e["foot"] == f]) for f in FOOT_JOINTS},
                    {n: {k: np.array([e["t"] for e in v]) for k, v in kinds.items()} for n, kinds in acc.items()}))
    return out


def run_mc(t: np.ndarray, base: np.ndarray, noise_m: float, cfg: TimingConfig, rng, cut: dict) -> list:
    """cfg.mc_draws draws, spread over cfg.workers processes (spline fits are CPU bound)."""
    seeds = [int(x) for x in rng.integers(0, 2 ** 31 - 1, cfg.mc_draws)]
    if not seeds:
        return []
    workers = max(1, min(cfg.workers, len(seeds)))
    if workers == 1:
        return _mc_chunk(t, base, noise_m, cfg, cut, seeds)
    from concurrent.futures import ProcessPoolExecutor
    chunks = [seeds[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_mc_chunk, *zip(*[(t, base, noise_m, cfg, cut, c) for c in chunks])))
    return [d for part in parts for d in part]


def _mc_times(nominal: np.ndarray, draws: list[np.ndarray], tol: float, shift_tol: float | None = None):
    """(D, n) matched MC event times (NaN when missed), the per-event detection rate and the per-draw common
    shift (D,). With shift_tol, each draw is first matched loosely (shift_tol) to estimate its median shift, and
    the tol matching is done after removing it: a draw whose touchdowns ALL moved (the per-foot template offset is
    re-estimated in every draw) still finds its events, and the stored times keep the shift, so sigma includes it."""
    out = np.full((len(draws), len(nominal)), np.nan)
    shifts = np.zeros(len(draws))
    for d, ts in enumerate(draws):
        ts = np.asarray(ts, float)
        if shift_tol is not None and len(ts):
            pr = match(nominal, ts, shift_tol)
            shifts[d] = float(np.median([ts[j] - nominal[i] for i, j in pr])) if pr else 0.0
        for i, j in match(nominal, ts - shifts[d], tol):
            out[d, i] = ts[j]
    rate = np.mean(np.isfinite(out), 0) if len(draws) else np.ones(len(nominal))
    return out, rate, shifts


def _attach_mc(e: dict, col: np.ndarray, rate: float) -> None:
    """sigma = MC SD around the nominal time (includes the draw's common shift); '_mc' = nominal time + the draw's
    deviation from the MC median (the MC supplies the spread; missed draws stay NaN and are skipped by nan-stats)."""
    ok = np.isfinite(col)
    e["sigma"] = float(max(np.std(col[ok]), 1e-4)) if ok.sum() > 1 else float("nan")
    e["detect_rate"] = float(rate)
    e["_mc"] = e["t"] + (col - np.median(col[ok])) if ok.sum() > 1 else np.full(len(col), e["t"])


def analyze_dancer(t: np.ndarray, joints: np.ndarray, cfg: TimingConfig, rng) -> dict:
    """Nominal events + Monte-Carlo event times (noise of the noise model added to the raw joints)."""
    noise = cfg.noise_m if cfg.noise_m is not None else estimate_noise(joints[:, _needed(cfg)])
    noise = max(noise, cfg.noise_floor_m)
    tds, acc, kin, tinfo, cut = events_for(t, joints, noise, cfg)
    draws = run_mc(t, joints, noise, cfg, rng, cut)
    out = {"noise_m": noise, "noise_estimated": cfg.noise_m is None, "touchdowns": [], "accents": {},
           "touchdown_info": tinfo, "_cut": cut,
           "derivatives": {k: {n: {"cutoff_hz": v["cutoff_hz"][v["joints"].index(j)],
                                   "snr": v["snr"][v["joints"].index(j)],
                                   "reliable": v["reliable"][v["joints"].index(j)]}
                               for n, j in cfg.accent_joints.items()} for k, v in cut.items()}}
    common = {}
    for foot in FOOT_JOINTS:
        nom = [e for e in tds if e["foot"] == foot]
        nt = np.array([e["t"] for e in nom])
        mc, rate, shifts = _mc_times(nt, [d[0][foot] for d in draws], cfg.match_tol_s, cfg.mc_shift_tol_s)
        for e, col, r in zip(nom, mc.T, rate):
            _attach_mc(e, col, r)
        out["touchdowns"] += nom
        common[foot] = {"common_shift_sd_ms": float(np.std(shifts) * 1000) if len(draws) > 1 else None,
                        "common_shift_mean_ms": float(np.mean(shifts) * 1000) if len(draws) else None,
                        "detect_rate_median": float(np.median(rate)) if len(rate) else None}
    out["touchdowns"].sort(key=lambda d: d["t"])
    out["mc"] = {"draws": len(draws), "per_foot": common, "white_noise_m": noise,
                 "corr_noise_m": cfg.mc_corr_noise_m, "corr_time_s": cfg.mc_corr_time_s if cfg.mc_corr_noise_m > 0 else None,
                 "method": "noise addition: raw joints + N(0, noise_m) per draw, whole detector re-run (template "
                           "offsets re-estimated), matched to the nominal events after removing each draw's "
                           "median shift; sigma keeps that shift"}
    for name in acc:
        out["accents"][name] = {}
        for kind, evs in acc[name].items():
            nt = np.array([e["t"] for e in evs])
            mc, rate, _ = _mc_times(nt, [d[1][name][kind] for d in draws], cfg.accent_min_sep_s / 2)
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
    s["_mc_all"] = mc
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
    sync_mc = rng.normal(0, cfg.sync_sigma_s if cfg.sync_sigma_s is not None else DEFAULT_SYNC_SIGMA_S,
                         (D, cfg.n_cams or DEFAULT_N_CAMS)).mean(1)
    res = {"version": 2, "config": _cfg_json(cfg), "beats": {
        **diag, "refined": cfg.refine_beats, "refine_residual_rms_ms": rms * 1000, "jitter_used_ms": jitter * 1000,
        "coarsen_grid": cfg.coarsen_grid, "tempo_bpm": 240.0 / diag["measure_s"] * (1 if diag["grid_measure_bpm"] <= 120
                                                                                     else 0.5),
        "grid": [[round(float(a), 6), int(b), int(m), int(p)] for a, b, m, p in
                 zip(grid.t, grid.type, grid.measure, grid.pos)], "grid_columns": ["t", "type", "measure", "count"]},
        "kinematics": {}, "touchdown_method": {}, "footsteps": {}, "accents": {}, "asynchrony": {}, "_kin": {},
        "_cut": {}, "_grid": grid}
    for role in ROLES:
        if role not in joints_by_role:
            continue
        d = analyze_dancer(t_by_role[role], joints_by_role[role], cfg, rng)
        res["_kin"][role] = d.pop("_kin")
        res["_cut"][role] = d.pop("_cut")
        res["kinematics"][role] = {"noise_m": d["noise_m"], "noise_estimated": d["noise_estimated"],
                                   "derivatives": d["derivatives"], "cross_check": d["cross_check"]}
        res["touchdown_method"][role] = {**d["touchdown_info"], "mc": d["mc"]}
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
        "sync_sigma_s": cfg.sync_sigma_s, "n_cams": cfg.n_cams, "sync_source": cfg.sync_source,
        "valid": cfg.av_sync_event is not None,
        "note": ("Event times are video reference seconds shifted by av_offset_s onto the beat (audio) clock. Absolute "
                 "on-beat claims (mean asynchrony, hit rate) need an audio-visual sync event (clap, slate, flash) "
                 "that pins the A/V offset; without one only RELATIVE timing is trustworthy: follow-minus-lead, "
                 "measure-to-measure changes, consistency (R, sd).")}
    res["notes"] = NOTES
    return res


def _relative(res: dict, rng) -> dict:
    """Follow minus lead mean footstep asynchrony, overall and per measure, with paired-draw CIs.

    The MC rows share the beat-jitter and camera-sync draws between the dancers, so those cancel in the difference
    (as does the A/V offset). The pose-noise rows are independent per dancer and keep each dancer's re-estimated
    per-foot template offset, so the CI includes that uncertainty. What does NOT cancel and is not in the CI: a
    systematic difference between the dancers' touchdown definitions (e.g. a heel-first vs a toe-first landing
    style moves the change point differently); the synthetic validation bounds it for one landing style only."""
    fl, ld = res["asynchrony"]["follow"]["footsteps"], res["asynchrony"]["lead"]["footsteps"]
    if not fl.get("n") or not ld.get("n"):
        return {}

    def boot_diff(a, b):
        D = a.shape[0]
        ba = rng.integers(0, a.shape[1], (D, a.shape[1]))
        bb = rng.integers(0, b.shape[1], (D, b.shape[1]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return (np.nanmean(np.take_along_axis(a, ba, 1), 1) - np.nanmean(np.take_along_axis(b, bb, 1), 1)) * 1000

    out = []
    for m in sorted(set(fl["_mc_by_measure"]) & set(ld["_mc_by_measure"])):
        a, b = fl["_mc_by_measure"][m], ld["_mc_by_measure"][m]
        nom = next(x["mean_ms"] for x in fl["per_measure"] if x["measure"] == m) -             next(x["mean_ms"] for x in ld["per_measure"] if x["measure"] == m)
        out.append({"measure": m, "diff_ms": nom, "ci95_ms": _ci(boot_diff(a, b)), "n_follow": int(a.shape[1]),
                    "n_lead": int(b.shape[1])})
    return {"overall_ms": fl["mean_ms"] - ld["mean_ms"], "overall_ci95_ms": _ci(boot_diff(fl["_mc_all"], ld["_mc_all"])),
            "per_measure": out,
            "note": "relative timing: the A/V offset, beat jitter and camera sync cancel (paired draws). Each "
                    "dancer's touchdown estimator has its own noise-dependent offset (per-foot template offset): it "
                    "does NOT cancel; its noise part is propagated (MC re-estimates it per draw), a systematic "
                    "landing-style difference between the dancers is not."}


def _cfg_json(cfg: TimingConfig) -> dict:
    d = asdict(cfg)
    d["allowed_types"] = list(cfg.allowed_types)
    d["allowed_positions"] = None if cfg.allowed_positions is None else list(cfg.allowed_positions)
    return d


NOTES = [
    "Positions / velocities: quintic smoothing spline per coordinate on the real timestamps, s = N*sigma^2 with "
    "sigma the joint noise (4th differences unless configured). Acceleration / jerk: penalised quintic smoother "
    "(zero phase, linear) at the cutoff minimising the predicted MSE of that derivative (signal spectrum of the "
    "accent joints minus the white-noise floor), per derivative order (kinematics.derivatives: cutoff_hz, per-joint "
    "snr, reliable = snr >= deriv_snr_min). Low cutoffs keep peak times but flatten peaks; at ~1 cm pose noise most "
    "jerk is flagged unreliable. Butterworth is a cross-check only; forward "
    "differences are never used (they shift jerk ~1.5 frames early).",
    "Touchdown = the change point at which the foot's swing ends: the heights come down onto their stance plateau "
    "and the pelvis-frame positions of knee / ankle / toe stop changing velocity. Per step a multi-channel hinge on "
    "the RAW leg channels (t_hinge); per foot the steps are aligned to their leave-one-out average (matched "
    "filter) and the touchdown is placed at the change point of the pooled aligned steps (method 'template'), "
    "searched within +-60 ms of the median per-step estimate (touchdown_method.<role>.template_feet). Steps left "
    "out get the dancer's median template-minus-hinge offset (method 'hinge+offset'). t_ankle / t_toe are the "
    "per-joint raw-height hinges, t_spline_vy0 the spline vertical-velocity cross-check. A touchdown is never "
    "placed more than clamp_after_stance_s after the contact model's first stance frame (clamped_to_stance). A "
    "foot whose median touchdown-minus-first-stance-frame lies outside plausible_lag_ms (synthetic: -3..92 ms, clean "
    "and at 1 cm noise) is listed in touchdown_method.<role>."
    "suspect_feet and its events carry suspect=true.",
    "sigma per event = SD over a Monte-Carlo that adds the noise model to the RAW joints and re-runs the whole "
    "detector (template offsets re-estimated per draw); draws are matched to the nominal events after removing "
    "each draw's median shift, and sigma keeps that shift. detect_rate = fraction of draws that found the event. "
    "Optionally (mc_corr_noise_m > 0) a temporally correlated pose error is added too: smooth fused fits look "
    "nearly noise-free to the 4th-difference estimate, so the white model alone gives lower-bound sigmas.",
    "Accents = local maxima of |acceleration| and |jerk| of key joints with prominence; each carries its joint's "
    "derivative SNR and 'reliable'. Treat unreliable accents (and jerk generally at 30 fps) as descriptive.",
    "Asynchrony = event - nearest allowed beat (allowed_types / allowed_positions). CIs combine Monte-Carlo pose noise, "
    "beat jitter, the per-camera sync uncertainty (mean over n_cams) and an event bootstrap. Chance level: "
    "surrogate.hit_rate_p / V_p use the JITTER null (each event's phase independent and uniform within its beat "
    "cell: can reject chance for a precise on-beat dancer); shift_* is the circular shift of the whole train "
    "(floor p_floor_isochronous for steady stepping); rayleigh_p / v_test_p are the analytic circular tests. All "
    "assume independent events; a steady self-paced tempo close to the music's would also look locked over 6 s.",
    "measure_effects: two-sided permutation test of each measure's mean asynchrony against the mean of the dancer's "
    "OTHER measures, Benjamini-Hochberg q values; a late measure has dev_ms > 0 with small q. Late-vs-rest "
    "contrasts report perm_p_greater (directional) and perm_p_two_sided (controls).",
    "follow_minus_lead: paired MC draws cancel the A/V offset, beat jitter and camera sync. Each dancer's touchdown "
    "estimator has its own noise-dependent offset that does NOT cancel; its noise part is in the CI (re-estimated "
    "per draw), a systematic landing-style difference between the dancers is not.",
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
    """Beats in REFERENCE seconds: timing/beats.json, beats.json (copy of beats/beats.json) or beats/beats.json.
    zouk-time-analysis.json is NOT searched: per the contract it holds AUDIO-CLIP time (t + time_to_audio)."""
    for p in (work / "timing" / "beats.json", work / "beats.json", work / "beats" / "beats.json"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no beats json in {work} (timing/beats.json, beats.json or beats/beats.json); "
                            f"pass --beats")


DEFAULT_SYNC_SIGMA_S, DEFAULT_N_CAMS = 0.005, 8


def resolve_sync(cfg: TimingConfig, work: Path) -> TimingConfig:
    """Fill sync_sigma_s / n_cams that are None from work/sync/offsets.json (number of cameras; residual SD when
    the sync stage writes one: 'residual_sd_s' or per-camera {'sd_s'|'sigma_s'}), else the defaults. Explicit
    values are kept. Records where the values came from in cfg.sync_source."""
    if cfg.sync_sigma_s is not None and cfg.n_cams is not None:
        cfg.sync_source = "config"
        return cfg
    src = []
    p = work / "sync" / "offsets.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            offs = d.get("offsets", d) if isinstance(d, dict) else {}
            if cfg.n_cams is None and isinstance(offs, dict) and offs:
                cfg.n_cams = len(offs)
                src.append("n_cams: sync/offsets.json")
            if cfg.sync_sigma_s is None:
                sd = d.get("residual_sd_s") if isinstance(d, dict) else None
                if sd is None and isinstance(offs, dict):
                    per = [v.get("sd_s", v.get("sigma_s")) for v in offs.values() if isinstance(v, dict)]
                    per = [float(x) for x in per if x is not None]
                    sd = float(np.sqrt(np.mean(np.square(per)))) if per else None
                if sd is not None:
                    cfg.sync_sigma_s = float(sd)
                    src.append("sync_sigma_s: sync/offsets.json")
        except (AttributeError, TypeError, ValueError):
            src.append("sync/offsets.json unreadable")
    if cfg.n_cams is None:
        cfg.n_cams = DEFAULT_N_CAMS
        src.append(f"n_cams: default {DEFAULT_N_CAMS}")
    if cfg.sync_sigma_s is None:
        cfg.sync_sigma_s = DEFAULT_SYNC_SIGMA_S
        src.append(f"sync_sigma_s: default {DEFAULT_SYNC_SIGMA_S * 1000:g} ms")
    cfg.sync_source = "; ".join(src) or "config"
    return cfg


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
    cfg = resolve_sync(TimingConfig(**asdict(cfg)), work)  # a copy: the caller's config is never mutated
    res = analyze(ts, js, load_beats(beats_path), cfg)
    res["_joints"] = js
    for role in ts:  # smooth fused fits: the white-noise model alone gives optimistic sigmas / CIs
        nz = res["kinematics"][role]["noise_m"]
        if nz < 0.002 and cfg.mc_corr_noise_m <= 0:
            res["beats"]["warnings"].append(
                f"{role}: estimated white pose noise is only {nz * 1000:.1f} mm (a smooth fused fit); its real error is "
                f"larger and temporally correlated, so sigma / CIs from the white-noise MC are LOWER bounds - set "
                f"mc_corr_noise_m (e.g. 0.01 m, mc_corr_time_s 0.1) for an assumed low-frequency error")
    res["inputs"] = {"smplx": {r: str(work / "smplx" / f"{r}.npz") for r in ts}, "beats": str(beats_path),
                     "noise_injected_m": noise_inject_m}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timing.json").write_text(json.dumps(_clean(res), indent=1))
    kin = {}
    for role in ts:  # full 55-joint derivatives for downstream stages (physics, Unity overlays)
        noise = res["kinematics"][role]["noise_m"]
        k = spline_kinematics(ts[role], js[role], noise, k=cfg.spline_k)
        full = choose_cutoffs(ts[role], js[role], range(js[role].shape[1]), noise, cfg)  # per joint, all 55
        apply_cutoffs(ts[role], js[role], full, (k,))
        kin.update({f"{role}_{n}": getattr(k, n).astype(np.float32) for n in ("pos", "vel", "acc", "jerk")})
        kin[f"{role}_t"] = ts[role]
        kin[f"{role}_noise_m"] = np.float32(noise)
        for name in ("acc", "jerk"):  # reliability for consumers (Unity): per joint, plus the per-frame noise SD
            kin[f"{role}_{name}_cutoff_hz"] = np.asarray(full[name]["cutoff_hz"], np.float32)
            kin[f"{role}_{name}_snr"] = np.asarray(full[name]["snr"], np.float32)
            kin[f"{role}_{name}_reliable"] = np.asarray(full[name]["reliable"], bool)
            kin[f"{role}_{name}_noise_sd"] = np.asarray(full[name]["noise_sd"], np.float32)
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


def plot_review(res: dict, js: dict, ts: dict, path: Path, title: str = "", truth: dict | None = None) -> None:
    """Review sheet: per dancer the foot heights with every touchdown (+- sigma, async label) on the beat grid,
    the asynchrony per step and per measure (95 % CI), event phases on the beat circle, and the test summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = res["_grid"]
    roles = [r for r in ROLES if r in js]
    col = {"lead": "#1f4e9c", "follow": "#c0392b"}
    fig = plt.figure(figsize=(22, 4.3 * (len(roles) + 1)))
    gs = fig.add_gridspec(len(roles) + 1, 5)
    t_lo = min(float(ts[r][0]) for r in roles)
    t_hi = max(float(ts[r][-1]) for r in roles)
    in_win = (grid.t >= t_lo - 0.05) & (grid.t <= t_hi + 0.05)
    axs = []
    for k, role in enumerate(roles):
        ax = fig.add_subplot(gs[k, :4], sharex=axs[0] if axs else None)
        axs.append(ax)
        t, J = ts[role], js[role]
        for bt, ty, pos in zip(grid.t[in_win], grid.type[in_win], grid.pos[in_win]):
            ax.axvline(bt, color="0.35" if pos == 0 else "0.6", lw=1.6 if pos == 0 else (0.9 if ty == 1 else 0.4),
                       ls="-" if ty == 1 else ":", zorder=0)
        for foot, fc in (("left", "tab:blue"), ("right", "tab:orange")):
            ja, jt = FOOT_JOINTS[foot]["ankle"], FOOT_JOINTS[foot]["toe"]
            ax.plot(t, J[:, ja, 1], ".", color=fc, ms=3, alpha=.5)
            ax.plot(t, res["_kin"][role].pos[:, ja, 1], "-", color=fc, lw=1.2, label=f"{foot} ankle height")
            ax.plot(t, J[:, jt, 1], "-", color=fc, lw=0.6, alpha=.5)
            for e in res["footsteps"][role]:
                if e["foot"] != foot:
                    continue
                sg = e.get("sigma") or 0.0
                ax.axvspan(e["t"] - sg, e["t"] + sg, color=fc, alpha=.18, lw=0)
                ax.axvline(e["t"], color=fc, lw=1.6)
                ytop = float(np.nanmax(J[:, [7, 8], 1]))
                lab = f"{e.get('async_ms', float('nan')):+.0f}"
                if e.get("method") != "template":
                    lab += "*"
                ax.text(e["t"], ytop * 1.02, lab, color=fc, fontsize=8, ha="center", va="bottom")
        if truth:
            y0 = float(np.nanmin(J[:, [7, 8], 1])) - 0.01
            tt = [e["t"] for e in truth["touchdowns"][role] if t[0] <= e["t"] <= t[-1]]
            ax.plot(tt, [y0] * len(tt), "k^", ms=6, label="truth")
        tm = res.get("touchdown_method", {}).get(role, {})
        off = ", ".join(f"{f} {v['offset_ms']:+.0f} ms" for f, v in tm.get("template_feet", {}).items())
        ax.set_ylabel(f"{role}: height (m)")
        ax.set_title(f"{role.upper()}: ankle (dots raw, line spline) and toe heights; touchdowns +-1 sigma, label = "
                     f"asynchrony ms (* = fallback); grey lines = beats (thick: measure start, solid: counts 1/3)"
                     + (f" | template offsets {off}" if off else ""), fontsize=9, loc="left")
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        # phase circle
        pa = fig.add_subplot(gs[k, 4], projection="polar")
        evs = res["footsteps"][role]
        if evs:
            ph = np.array([2 * np.pi * e["async_ms"] / e["period_ms"] for e in evs if "async_ms" in e])
            pa.scatter(ph, np.ones(len(ph)) + np.linspace(-0.08, 0.08, len(ph)), color=col[role], s=25, alpha=.8)
            fs = res["asynchrony"][role]["footsteps"]
            if fs.get("n"):
                pa.annotate("", xy=(2 * np.pi * fs["circ_mean_ms"] / np.median([e["period_ms"] for e in evs]),
                                    fs["R"]), xytext=(0, 0), arrowprops=dict(color=col[role], width=2, headwidth=8))
                sur = fs.get("surrogate", {})
                pa.set_title(f"{role}: phase vs nearest beat (0 = on beat)\nR={fs['R']:.2f}  V-test p={fs['v_test_p']:.2g}"
                             f"  jitter-null p={sur.get('hit_rate_p', float('nan')):.2g}", fontsize=8)
        pa.set_theta_zero_location("N")
        pa.set_theta_direction(-1)
        pa.set_yticks([])
        pa.set_xticks(np.radians([0, 90, 180, 270]))
        pa.set_xticklabels(["on beat", "+1/4", "+-1/2", "-1/4"], fontsize=7)
    ax = fig.add_subplot(gs[len(roles), :4], sharex=axs[0])
    tol = res["config"]["hit_tol_s"] * 1000
    ax.axhspan(-tol, tol, color="0.9", zorder=0)
    lines = []
    for role in roles:
        evs = res["footsteps"][role]
        for foot, mk in (("left", "o"), ("right", "s")):
            ee = [e for e in evs if e["foot"] == foot and "async_ms" in e]
            ax.errorbar([e["t"] for e in ee], [e["async_ms"] for e in ee], yerr=[1000 * (e.get("sigma") or 0) for e in ee],
                        fmt=mk, color=col[role], ms=5, capsize=2, label=f"{role} {foot}")
        fs = res["asynchrony"][role]["footsteps"]
        for pm in fs.get("per_measure", []):
            if pm["t0"] is None:
                continue
            t0, t1 = max(pm["t0"], t_lo), min(pm["t0"] + res["beats"]["measure_s"], t_hi)
            ax.fill_between([t0, t1], pm["ci95_ms"][0], pm["ci95_ms"][1], color=col[role], alpha=.12)
            ax.plot([t0, t1], [pm["mean_ms"]] * 2, color=col[role], lw=1.5)
        if fs.get("n"):
            sur = fs.get("surrogate", {})
            lines.append(f"{role}: n={fs['n']}  mean {fs['mean_ms']:+.0f} ms [{fs['ci95_ms'][0]:+.0f}, {fs['ci95_ms'][1]:+.0f}]"
                         f"  hit {fs['hit_rate']:.0%} (chance {sur.get('hit_rate_mean', float('nan')):.0%},"
                         f" jitter-null p={sur.get('hit_rate_p', float('nan')):.2g}; shift p={sur.get('shift_hit_rate_p', float('nan')):.2g})")
    fml = res["asynchrony"].get("follow_minus_lead", {})
    if fml:
        ci = fml.get("overall_ci95_ms", [float("nan")] * 2)
        lines.append(f"follow - lead: {fml['overall_ms']:+.0f} ms [{ci[0]:+.0f}, {ci[1]:+.0f}] (A/V offset cancels; "
                     f"per-dancer touchdown definition offsets do not)")
    if not res.get("absolute_timing", {}).get("valid"):
        lines.append("ABSOLUTE asynchrony needs an A/V sync event (none given): read the means as relative only")
    ax.text(0.005, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8.5, family="monospace",
            bbox=dict(facecolor="white", alpha=.85, lw=0))
    ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("step - nearest beat (ms)")
    ax.set_xlabel("reference time (s)")
    ax.legend(fontsize=7, loc="lower right", ncol=4)
    ax.set_title(f"asynchrony per step (+-sigma) and per measure (mean, 95 % CI band); grey band = +-{tol:.0f} ms hit",
                 fontsize=9, loc="left")
    ax.set_xlim(t_lo - 0.1, t_hi + 0.1)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- validation

def score_touchdowns(res: dict, truth: dict, tol: float = 0.1, span: tuple | None = None, edge_s: float = 0.3) -> dict:
    """Recall/precision/|error| of touchdowns vs truth.json, PER DANCER (+ pooled for reference).

    Only truth steps and detections inside [span0 + edge_s, span1 - edge_s] are scored (a step at the very start
    has no observed swing, one at the very end no stance); span = the data's time range. Detections are matched
    one-to-one within tol against ALL truth steps, so an interior detection of an edge step is not a false one.
    sigma_coverage95 = fraction of matched events with |error| <= 1.96 sigma (MC sigma incl. common offset)."""
    out, errs_all = {}, []
    lo, hi = ((span[0] + edge_s, span[1] - edge_s) if span is not None else (-np.inf, np.inf))
    for role in ROLES:
        det = res["footsteps"].get(role, [])
        tp, n_det, n_fp, n_true, errs, ztd = 0, 0, 0, 0, [], []
        for foot in FOOT_JOINTS:
            de = [e for e in det if e["foot"] == foot]
            a = np.array([e["t"] for e in de])
            b = np.array([e["t"] for e in truth["touchdowns"][role] if e["foot"] == foot])
            pairs = match(a, b, tol)
            ina = (a >= lo) & (a <= hi)
            inb = (b >= lo) & (b <= hi)
            matched = {i for i, _ in pairs}
            n_det += int(ina.sum())
            n_fp += sum(1 for i in range(len(a)) if ina[i] and i not in matched)
            n_true += int(inb.sum())
            for i, j in pairs:
                if inb[j]:
                    tp += 1
                    errs.append(a[i] - b[j])
                    sg = de[i].get("sigma")
                    if sg is not None and np.isfinite(sg) and sg > 0:
                        ztd.append((a[i] - b[j]) / sg)
        errs = np.array(errs)
        errs_all += list(errs)
        ztd = np.array(ztd)
        out[role] = {"recall": tp / max(n_true, 1), "precision": 1 - n_fp / max(n_det, 1), "n_true": n_true,
                     "n_detected": n_det, "median_abs_err_ms": float(np.median(np.abs(errs)) * 1000) if len(errs) else None,
                     "mean_err_ms": float(np.mean(errs) * 1000) if len(errs) else None,
                     "sd_err_ms": float(np.std(errs) * 1000) if len(errs) > 1 else None,
                     "p90_abs_err_ms": float(np.percentile(np.abs(errs), 90) * 1000) if len(errs) else None,
                     "sigma_coverage95": float(np.mean(np.abs(ztd) <= 1.96)) if len(ztd) else None}
    e = np.abs(np.array(errs_all))
    out["pooled_median_abs_err_ms"] = float(np.median(e) * 1000) if len(e) else None
    return out


def group_contrast(evs: list[dict], in_group, cfg: TimingConfig, rng) -> dict:
    """Mean asynchrony of events in a group of measures vs the dancer's other events: difference with a CI
    (Monte-Carlo + bootstrap) and event-label permutation p values (null: measures are exchangeable):
    perm_p_greater (one-sided, group LATER than the rest - the directional hypothesis for a late measure) and
    perm_p_two_sided (|difference|; use it for a control, where early is as much a deviation as late)."""
    a = np.array([e["async_ms"] for e in evs]) / 1000
    g = np.array([bool(in_group(e["measure"])) for e in evs])
    if g.sum() == 0 or (~g).sum() == 0:
        return {}
    mc = np.stack([e["_async_mc"] for e in evs], 1)
    D = mc.shape[0]
    gi, oi = np.nonzero(g)[0], np.nonzero(~g)[0]
    bg = gi[rng.integers(0, len(gi), (D, len(gi)))]
    bo = oi[rng.integers(0, len(oi), (D, len(oi)))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grp = np.nanmean(np.take_along_axis(mc, bg, 1), 1)
        diff = grp - np.nanmean(np.take_along_axis(mc, bo, 1), 1)
    obs = a[g].mean() - a[~g].mean()
    pg = g[np.argsort(rng.random((cfg.n_perm, len(g))), axis=1)]
    perm = (pg * a).sum(1) / g.sum() - (~pg * a).sum(1) / (~g).sum()
    return {"n_group": int(g.sum()), "n_rest": int((~g).sum()), "group_mean_ms": float(a[g].mean() * 1000),
            "group_mean_ci95_ms": _ci(grp * 1000), "diff_ms": float(obs * 1000), "diff_ci95_ms": _ci(diff * 1000),
            "perm_p_greater": float((1 + np.sum(perm >= obs - 1e-12)) / (1 + cfg.n_perm)),
            "perm_p_two_sided": float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (1 + cfg.n_perm))}


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


def late_ok(late: dict) -> bool:
    """Follow's late measures later than her rest (one-sided p < 0.05, CI > 0) and the lead's same measures NOT
    different from his rest (two-sided p >= 0.05 and the CI contains 0: an early lead also fails the control)."""
    fl, ld = late["follow"]["late_measures_vs_rest"], late["lead"]["late_measures_vs_rest"]
    return bool(fl and ld and fl["perm_p_greater"] < 0.05 and fl["diff_ci95_ms"][0] > 0
                and ld["perm_p_two_sided"] >= 0.05 and ld["diff_ci95_ms"][0] <= 0 <= ld["diff_ci95_ms"][1])


def _grid_from(beats: list) -> BeatGrid:
    arr = np.asarray(beats, float).reshape(-1, 2)
    i = np.arange(len(arr))
    return BeatGrid(arr[:, 0], arr[:, 1].astype(int), i // MEASURE_LEN, i % MEASURE_LEN)


def validate_windows(cfg: TimingConfig, work: Path, noise_m: float = 0.01, window_s: float = 6.0,
                     starts=None, seeds: int = 4, mc: int = 40) -> dict:
    """Short-window validation (the real take is a 6 s window): 6 s slices of the synthetic take, iid Gaussian
    joint noise, several seeds; full analyze() (MC included) per slice. Per dancer: touchdown error, recall,
    precision, the per-dancer common offset (mean signed error) spread, MC sigma coverage and detect_rate, the
    jitter-null p for the (perfectly on-beat) lead, and the late-measure contrast in slices that hold both late and
    on-time measures."""
    from . import synthetic as syn
    starts = list(np.arange(0.0, 18.01, 3.0)) if starts is None else list(starts)
    rows = []
    for s0 in starts:
        for k in range(seeds if noise_m > 0 else 1):
            w = syn.window(work, float(s0), float(s0) + window_s, joint_noise_m=noise_m, seed=1000 * int(s0) + k)
            c = TimingConfig(**{**asdict(cfg), "mc_draws": mc, "seed": cfg.seed + 17 * k + int(s0)})
            c = resolve_sync(c, work)
            res = analyze({r: np.asarray(w["npz"][r]["t"], float) for r in ROLES},
                          {r: np.asarray(w["npz"][r]["joints"], float) for r in ROLES}, _grid_from(w["beats"]), c)
            sc = score_touchdowns(res, w["truth"], span=(float(s0), float(s0) + window_s))
            row = {"t0": float(s0), "seed": k, "noise_m": noise_m}
            for r in ROLES:
                fs = res["asynchrony"][r]["footsteps"]
                sur = fs.get("surrogate", {})
                row[r] = {**sc[r], "detect_rate_median": float(np.median([e["detect_rate"] for e in res["footsteps"][r]]))
                          if res["footsteps"][r] else None,
                          "hit_rate": fs.get("hit_rate"), "jitter_hit_p": sur.get("hit_rate_p"),
                          "jitter_V_p": sur.get("V_p"), "v_test_p": fs.get("v_test_p"),
                          "shift_hit_p": sur.get("shift_hit_rate_p"),
                          "template_feet": res["touchdown_method"][r].get("template_feet")}
            late = score_late(res, w["truth"], c)
            if late["follow"]["late_measures_vs_rest"] and late["lead"]["late_measures_vs_rest"]:
                row["late"] = {r: {k2: late[r]["late_measures_vs_rest"][k2] for k2 in
                                   ("n_group", "n_rest", "diff_ms", "diff_ci95_ms", "perm_p_greater", "perm_p_two_sided")}
                               for r in ROLES}
                row["late_ok"] = late_ok(late)
                row["follow_minus_lead_ci95_ms"] = res["asynchrony"]["follow_minus_lead"].get("overall_ci95_ms")
            rows.append(row)
    summ = {}
    nanf = lambda v: np.nan if v is None else v
    for r in ROLES:
        med = np.array([nanf(x[r]["median_abs_err_ms"]) for x in rows], float)
        mean = np.array([nanf(x[r]["mean_err_ms"]) for x in rows], float)
        cov = [x[r]["sigma_coverage95"] for x in rows if x[r]["sigma_coverage95"] is not None]
        summ[r] = {"n_windows": len(rows),
                   "median_abs_err_ms": {"median": float(np.nanmedian(med)), "p90": float(np.nanpercentile(med, 90)),
                                         "max": float(np.nanmax(med))},
                   "common_offset_ms": {"mean": float(np.nanmean(mean)), "sd": float(np.nanstd(mean)),
                                        "min": float(np.nanmin(mean)), "max": float(np.nanmax(mean))},
                   "recall": {"mean": float(np.mean([x[r]["recall"] for x in rows])),
                              "min": float(np.min([x[r]["recall"] for x in rows]))},
                   "precision": {"mean": float(np.mean([x[r]["precision"] for x in rows])),
                                 "min": float(np.min([x[r]["precision"] for x in rows]))},
                   "sigma_coverage95_mean": float(np.mean(cov)) if cov else None,
                   "detect_rate_median": float(np.nanmedian([nanf(x[r]["detect_rate_median"]) for x in rows])),
                   "jitter_hit_p_max": float(np.nanmax([nanf(x[r]["jitter_hit_p"]) for x in rows])),
                   "jitter_hit_p_median": float(np.nanmedian([nanf(x[r]["jitter_hit_p"]) for x in rows])),
                   "shift_hit_p_min": float(np.nanmin([nanf(x[r]["shift_hit_p"]) for x in rows]))}
    lw = [x for x in rows if "late_ok" in x]
    summ["late_windows"] = {"n": len(lw), "n_late_ok": int(sum(x["late_ok"] for x in lw)),
                            "follow_p_greater": [x["late"]["follow"]["perm_p_greater"] for x in lw],
                            "lead_p_two_sided": [x["late"]["lead"]["perm_p_two_sided"] for x in lw],
                            "follow_minus_rest_ms": [x["late"]["follow"]["diff_ms"] for x in lw]}
    return {"window_s": window_s, "noise_m": noise_m, "starts": starts, "seeds": seeds, "mc_draws": mc,
            "summary": summ, "rows": rows}


def plot_validation(report: dict, path: Path) -> None:
    """One-page summary of the synthetic validation: per dancer, the 6 s windows (median |error| per window, the
    common offset, MC sigma coverage, recall), the chance tests' power for the on-beat lead (jitter null vs the
    circular shift), and the late-measure contrast in the windows that contain one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wins = report.get("windows", {})
    tags = list(wins)
    col = {"lead": "#1f4e9c", "follow": "#c0392b"}
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for k, tag in enumerate(tags):
        rows = wins[tag]["rows"]
        for j, role in enumerate(ROLES):
            x = 2 * k + j + np.random.default_rng(0).uniform(-0.15, 0.15, len(rows))
            med = [r[role]["median_abs_err_ms"] for r in rows]
            off = [r[role]["mean_err_ms"] for r in rows]
            cov = [r[role]["sigma_coverage95"] for r in rows]
            rec = [r[role]["recall"] for r in rows]
            axs[0, 0].plot(x, med, "o", color=col[role], alpha=.7, label=role if k == 0 else None)
            axs[0, 1].plot(x, off, "o", color=col[role], alpha=.7)
            axs[0, 2].plot(x, [c if c is not None else np.nan for c in cov], "o", color=col[role], alpha=.7)
            axs[0, 2].plot(x, rec, "x", color=col[role], alpha=.7)
    ticks = [2 * k + j for k in range(len(tags)) for j in range(2)]
    labels = [f"{tag}\n{role}" for tag in tags for role in ROLES]
    for a in axs[0]:
        a.set_xticks(ticks)
        a.set_xticklabels(labels, fontsize=8)
        a.grid(alpha=.3)
    axs[0, 0].axhline(33, color="k", ls="--", lw=.8)
    axs[0, 0].axhline(20, color="k", ls=":", lw=.8)
    axs[0, 0].set_title("6 s windows: median |touchdown error| per window (ms)\n(dashed 33 ms noisy limit, dotted 20 ms clean)",
                        fontsize=9)
    axs[0, 0].legend(fontsize=8)
    axs[0, 1].axhline(0, color="k", lw=.6)
    axs[0, 1].set_title("6 s windows: common offset = mean signed error per dancer-window (ms)\n(does not cancel in "
                        "follow-minus-lead)", fontsize=9)
    axs[0, 2].axhline(0.95, color="k", ls="--", lw=.8)
    axs[0, 2].set_ylim(0, 1.05)
    axs[0, 2].set_title("o = MC sigma coverage (|err| <= 1.96 sigma), x = recall (interior steps)", fontsize=9)
    # chance tests on the on-beat lead
    a = axs[1, 0]
    for k, tag in enumerate(tags):
        rows = wins[tag]["rows"]
        jit = [r["lead"]["jitter_hit_p"] for r in rows]
        sh = [r["lead"]["shift_hit_p"] for r in rows]
        vt = [r["lead"]["v_test_p"] for r in rows]
        a.semilogy(np.full(len(rows), 3 * k) + 0.0, jit, "o", color="tab:green", label="jitter null (hit rate)" if k == 0 else None)
        a.semilogy(np.full(len(rows), 3 * k) + 0.8, vt, "^", color="tab:purple", label="V-test" if k == 0 else None)
        a.semilogy(np.full(len(rows), 3 * k) + 1.6, sh, "s", color="tab:gray", label="circular shift (old)" if k == 0 else None)
    a.axhline(0.05, color="k", ls="--", lw=.8)
    a.set_xticks([3 * k + 0.8 for k in range(len(tags))])
    a.set_xticklabels(tags)
    a.set_title("perfectly on-beat lead, 6 s windows: chance-test p values (one-sided)", fontsize=9)
    a.legend(fontsize=8)
    # late windows
    a = axs[1, 1]
    for k, tag in enumerate(tags):
        lw = [r for r in wins[tag]["rows"] if "late" in r]
        a.plot(np.full(len(lw), k) - 0.1, [r["late"]["follow"]["diff_ms"] for r in lw], "o", color=col["follow"],
               label="follow late-vs-rest" if k == 0 else None)
        a.plot(np.full(len(lw), k) + 0.1, [r["late"]["lead"]["diff_ms"] for r in lw], "o", color=col["lead"],
               label="lead same measures (control)" if k == 0 else None)
        ok = sum(r["late_ok"] for r in lw)
        a.text(k, 120, f"late_ok {ok}/{len(lw)}", ha="center", fontsize=8)
    a.axhline(90, color=col["follow"], ls=":", lw=.8)
    a.axhline(0, color="k", lw=.6)
    a.set_xticks(range(len(tags)))
    a.set_xticklabels(tags)
    a.set_title("windows holding late + on-time measures: difference (ms), truth +90 / 0", fontsize=9)
    a.legend(fontsize=8)
    # gate text
    a = axs[1, 2]
    a.axis("off")
    g = report.get("gate", {})
    lines = [f"G5 = {g.get('G5')}"]
    for tag in [t for t in g if t != "G5"]:
        v = g[tag]
        lines.append(f"[{tag}] touchdown_ok {v['touchdown_ok']}  late_ok {v['late_ok']}")
        for role in ROLES:
            pd = v["per_dancer"][role]
            wm = pd.get("window_median_abs_err_ms") or {}
            lines.append(f"  {role}: 24 s median {pd['median_abs_err_ms']:.1f} ms (worst seed {pd['worst_noise_seed_ms']:.1f}),"
                         f" recall {pd['recall']:.2f}")
            if wm:
                lines.append(f"     6 s: median {wm['median']:.1f} p90 {wm['p90']:.1f} max {wm['max']:.1f} ms,"
                             f" offset sd {pd['window_common_offset_ms']['sd']:.1f} ms")
        lines.append(f"  follow late-rest {v['follow_late_minus_rest_ms']:+.0f} ms p>{v['follow_late_perm_p_greater']:.3g};"
                     f" lead control 2-sided p {v['lead_same_measures_perm_p_two_sided']:.2f}")
        lines.append(f"  on-beat lead: jitter-null p {v['lead_on_beat_jitter_p']:.3g}, shift p {v['lead_on_beat_shift_p']:.2f}")
    a.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    fig.suptitle("timing: synthetic validation (gate G5, per dancer; 6 s windows like the real take)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def validate_synthetic(cfg: TimingConfig | None = None, noise_levels=(0.0, 0.01), work: Path | None = None,
                       extra_noise_seeds: int = 4, extra_mc: int = 50, windows: bool = True, window_seeds: int = 4,
                       window_mc: int = 40) -> dict:
    """Gate G5 on work/synthetic, PER DANCER: the full 24 s take clean / noisy (+ extra noise seeds), and 6 s
    windows like the real take (clean + noisy, several seeds); late measures significant vs the permutation null
    with a two-sided lead control; chance baselines reported."""
    cfg = cfg or TimingConfig()
    work = work or PACKAGE_ROOT / "work" / "synthetic"
    truth = json.loads((work / "truth.json").read_text())
    report = {"truth": str(work / "truth.json"), "runs": {}, "noise_seeds": {}, "windows": {}}
    ts = {r: load_smplx(work / "smplx" / f"{r}.npz")[0] for r in ROLES}
    span = (float(ts["lead"][0]), float(ts["lead"][-1]))
    for nz in noise_levels:
        c = TimingConfig(**asdict(cfg))
        tag = "clean" if nz == 0 else f"noise_{nz * 100:g}cm"
        out_dir = work / "timing" if nz == 0 else work / "timing" / tag
        res = run(work, c, out_dir=out_dir, noise_inject_m=nz, plot=False)
        plot_qa(res, res["_joints"], ts, out_dir / "qa_timing.png", truth)
        late = score_late(res, truth, c)
        report["runs"][tag] = {
            "noise_injected_m": nz, "noise_estimated_m": {r: res["kinematics"][r]["noise_m"] for r in ROLES},
            "touchdowns": score_touchdowns(res, truth, span=span), "late": late, "late_ok": late_ok(late),
            "footsteps": {r: {k: res["asynchrony"][r]["footsteps"][k] for k in
                              ("n", "mean_ms", "ci95_ms", "R", "V", "rayleigh_p", "v_test_p", "hit_rate", "surrogate")}
                          for r in ROLES},
            "touchdown_method": res["touchdown_method"],
            "follow_minus_lead": res["asynchrony"]["follow_minus_lead"],
            "derivatives": {r: res["kinematics"][r]["derivatives"] for r in ROLES},
            "beats": {k: res["beats"][k] for k in ("grid_measure_bpm", "chance_hit_rate_analytic", "warnings")},
            "cross_check": {r: res["kinematics"][r]["cross_check"] for r in ROLES},
            "out": str(out_dir)}
        if nz > 0:
            per = {r: [report["runs"][tag]["touchdowns"][r]["median_abs_err_ms"]] for r in ROLES}
            offs = {r: [report["runs"][tag]["touchdowns"][r]["mean_err_ms"]] for r in ROLES}
            for k in range(1, extra_noise_seeds + 1):
                ck = TimingConfig(**{**asdict(cfg), "seed": cfg.seed + k, "mc_draws": extra_mc})
                rk = run(work, ck, out_dir=work / "timing" / f"{tag}_seed{k}", noise_inject_m=nz, plot=False)
                sk = score_touchdowns(rk, truth, span=span)
                for r in ROLES:
                    per[r].append(sk[r]["median_abs_err_ms"])
                    offs[r].append(sk[r]["mean_err_ms"])
            report["noise_seeds"][tag] = {r: {"median_abs_err_ms_per_seed": per[r], "worst_ms": float(np.max(per[r])),
                                              "mean_err_ms_per_seed": offs[r]} for r in ROLES}
        if windows:
            report["windows"][tag] = validate_windows(cfg, work, nz, seeds=window_seeds, mc=window_mc)
    g = {}
    for tag, rr in report["runs"].items():
        td = rr["touchdowns"]
        limit = 20.0 if tag == "clean" else 33.0
        seeds = report["noise_seeds"].get(tag)
        wsum = report["windows"].get(tag, {}).get("summary")
        per_dancer = {}
        for r in ROLES:
            worst_seed = seeds[r]["worst_ms"] if seeds else td[r]["median_abs_err_ms"]
            w = wsum[r] if wsum else None
            per_dancer[r] = {
                "median_abs_err_ms": td[r]["median_abs_err_ms"], "worst_noise_seed_ms": worst_seed,
                "recall": td[r]["recall"], "precision": td[r]["precision"],
                "sigma_coverage95": td[r]["sigma_coverage95"],
                "window_median_abs_err_ms": w and w["median_abs_err_ms"], "window_recall": w and w["recall"],
                "window_precision": w and w["precision"], "window_common_offset_ms": w and w["common_offset_ms"],
                "window_sigma_coverage95": w and w["sigma_coverage95_mean"],
                "touchdown_ok": bool(td[r]["median_abs_err_ms"] is not None and td[r]["median_abs_err_ms"] <= limit
                                     and worst_seed <= limit and td[r]["recall"] >= 0.9
                                     and (w is None or (w["median_abs_err_ms"]["p90"] <= limit
                                                        and w["recall"]["mean"] >= 0.9)))}
        fl, ld = rr["late"]["follow"]["late_measures_vs_rest"], rr["late"]["lead"]["late_measures_vs_rest"]
        rel = [m for m in rr["follow_minus_lead"]["per_measure"] if m["measure"] % 8 in truth["late_measures_mod8"]]
        g[tag] = {
            "limit_ms": limit, "per_dancer": per_dancer,
            "touchdown_ok": all(v["touchdown_ok"] for v in per_dancer.values()),
            "follow_late_minus_rest_ms": fl.get("diff_ms"), "follow_late_minus_rest_ci95_ms": fl.get("diff_ci95_ms"),
            "follow_late_perm_p_greater": fl.get("perm_p_greater"),
            "lead_same_measures_perm_p_two_sided": ld.get("perm_p_two_sided"),
            "lead_same_measures_diff_ci95_ms": ld.get("diff_ci95_ms"),
            "follow_truth_late_steps_mean_ms": rr["late"]["follow"]["truth_late_steps_async_ms"]["mean"],
            "follow_minus_lead_late_measures": rel,
            "late_ok": rr["late_ok"],
            "window_late": wsum and wsum["late_windows"],
            "lead_on_beat_jitter_p": rr["footsteps"]["lead"]["surrogate"].get("hit_rate_p"),
            "lead_on_beat_shift_p": rr["footsteps"]["lead"]["surrogate"].get("shift_hit_rate_p"),
            "surrogate_follow": rr["footsteps"]["follow"]["surrogate"]}
    g["G5"] = bool(all(v["touchdown_ok"] and v["late_ok"] for v in g.values()))
    report["gate"] = g
    (work / "timing" / "validation.json").write_text(json.dumps(_clean(report), indent=1))
    try:
        plot_validation(report, work / "timing" / "validation.png")
    except Exception as e:  # the plot must never lose the validation numbers
        report["plot_error"] = repr(e)
    return report
