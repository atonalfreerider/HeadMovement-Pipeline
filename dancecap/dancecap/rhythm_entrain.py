"""Rhythmic ENTRAINMENT of the dancers' motion to the song's beat grid (agent S7, 2026-10-11): the measurement under the live rhythm score.

Why it exists. The first rhythm score (S6) asked "does a foot touchdown land on a pulse / 'and' slot?" and gave 5 to a professional heat. The controls
(docs/RHYTHM.md, "Controls") showed that the question, not the dancing, was the problem: the same score reads 6.5 on the demo (a professional couple on a studio song with
a careful alignment), because (a) zouk is flowing weight transfer, not footfalls on every beat, (b) a touchdown time is only good to ~0.05 to 0.2 s, and (c) a slot-exact
kernel asks for more than the data can give. The body does keep time, though: the centre of mass sways once per two pulses, locked to the grid for minutes,
in both couples. So this module asks the physical question: is the motion's rhythm **phase-locked** to the music's rhythm at some metrical level?

Method (all causal in the dancers' motion; a pure function of time).
  * Levels h (cycles per 4-pulse measure): 2 = once per two pulses (the half bar, zouk's "slow"), 4 = the pulse, 6 = three in two pulses, 8 = the eighth.
  * Streams: BODY = the centre of mass' horizontal speed (weight transfer) and its height (bounce), band-passed (+-25 %) around the level's frequency;
    STEPS = the foot touchdowns as an event train. Their phase against the grid is demodulated (u = x * exp(-2 pi i h phi / 4), phi = grid phase in pulses) and averaged with a
    causal exponential window (tau_long 45 s, or tau 15 s for the recent form): rho = |mean u| / mean |x_band| is the amplitude-normalised phase-locking value
    (1 = a locked oscillation, ~0 = unrelated). The level is the one with the highest lock so far (a causal choice; the baseline pays for it).
  * The noise floor of the estimator, the price of choosing a level and the start-up bias are removed with a PERMUTATION baseline: the identical pipeline on K metronomes at the
    wrong tempo (the grid stretched by 0.84 .. 1.19: the same regularity, no relation to the dancers). skill = clip((M - b) / (ref - b), 0, 1), b = the baseline's mean + 0.5 sd,
    with the skill the baseline runs themselves reach taken off again: 0 = what unrelated motion gives, 1 = the reference level `ref` (see EntrainConfig.ref_*).
  * Couple sync: the coherence of the lead's and the follow's locked oscillations (the grid phase cancels), against the follow replayed at the wrong rate.
A constant phase offset is invisible to all of this by design: the absolute alignment of the A/V clocks is +-30 ms unless a sync event pins it, and the convention of where in the
beat a weight transfer "should" land is not established; the score says so. The song's tempo (one number per song) and the grid are known in advance; nothing of the dancers' motion after
t enters the value at t (tests/test_rhythm.py checks it to 1e-9).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LEVEL_H = (2, 4, 6, 8)
LEVEL_LABELS = ("half bar (1 per 2 pulses)", "pulse", "3 in 2 pulses", "eighth")
LEVEL_NAMES = ("halfbar", "pulse", "triplet", "eighth")
NULL_SCALES = (0.84, 0.88, 0.92, 1.08, 1.13, 1.19)
PARTNER_SCALES = (0.72, 0.78, 0.82, 0.86, 0.90, 0.93)          # the follow's signals replayed slower (< 1: they are drawn from the past, so the baseline stays causal)
ROLES = ("lead", "follow")
BODY_STREAMS = ("move", "bounce")


@dataclass
class EntrainConfig:
    tau_s: float = 15.0               # causal exponential window of the 'recent' phase average
    tau_long_s: float = 45.0          # the long window (body, feet, couple): about everything since the start of the song, forgetting slowly so a drifting phase can be followed
    band: float = 0.25                # +-25 % band-pass around the level frequency
    warm_s: float = 6.0               # filter warm-up taken from the signal before a grid segment starts
    base_smooth_s: float = 20.0       # moving average of the null baseline (K is small)
    base_margin_sd: float = 0.5       # the baseline level is the wrong-tempo runs' mean plus this many of their standard deviations (a margin over chance, not just its average)
    ref_body: float = 0.48            # phase-locking value of the body streams that reads 1.0 (the 90th percentile on the demo's first half: docs/RHYTHM.md, 'Reference levels')
    ref_steps: float = 0.31           # phase concentration of the foot touchdowns that reads 1.0 (same)
    ref_couple: float = 0.37          # lead-follow coherence that reads 1.0 (same)


@dataclass
class Body:
    """body signals per role at a common time base (30 fps): `move` (horizontal centre-of-mass speed, m/s), `bounce` (centre-of-mass height, m)"""
    t: np.ndarray
    sig: dict                         # role -> {"move": x, "bounce": x}
    source: str = "physics"


def load_body(work, fallback_kinematics: bool = True) -> "Body | None":
    """work/<take>/physics/physics.npz (centre of mass) or else the pelvis of timing/kinematics.npz"""
    work = Path(work)
    p = work / "physics" / "physics.npz"
    if p.exists():
        z = np.load(p)
        if all(f"{r}_com" in z.files and f"{r}_vcom" in z.files for r in ROLES):
            t = np.asarray(z["t"], float)
            sig = {r: {"move": np.linalg.norm(z[f"{r}_vcom"][:, [0, 2]], axis=1).astype(float), "bounce": np.asarray(z[f"{r}_com"][:, 1], float)} for r in ROLES}
            return Body(t, sig, "physics centre of mass")
    k = work / "timing" / "kinematics.npz"
    if fallback_kinematics and k.exists():
        z = np.load(k)
        if all(f"{r}_pos" in z.files and f"{r}_vel" in z.files for r in ROLES):
            t = np.asarray(z["lead_t"], float)
            sig = {r: {"move": np.linalg.norm(z[f"{r}_vel"][:, 0][:, [0, 2]], axis=1).astype(float), "bounce": np.asarray(z[f"{r}_pos"][:, 0, 1], float)} for r in ROLES}
            return Body(t, sig, "pelvis")
    return None


# ---------------------------------------------------------------------------------------------------------------- grid phase and segments
def grid_phase(grid, t: np.ndarray, scale: float = 1.0, anchor: float | None = None) -> np.ndarray:
    """grid phase in pulses at times t (the grid's rows are eighth slots: row i = i / 2 pulses). `scale` stretches the tempo about `anchor` (the wrong-tempo metronome of the baseline)."""
    nodes = np.arange(len(grid.t)) / 2.0
    ph = np.interp(t, grid.t, nodes)
    if scale != 1.0:
        a = float(np.interp(anchor if anchor is not None else t[0], grid.t, nodes))
        ph = a + (ph - a) * scale
    return ph


def beat_segments(grid, t: np.ndarray) -> list:
    """[(i0, i1)] sample index ranges of the beat-covered stretches (state 0) of the time base t; the grid is continuous inside them"""
    state = np.array([grid.state_at(x) for x in t])
    ok = (state == 0) & (t >= grid.t[0]) & (t <= grid.t[-1])
    out, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j < len(ok) and ok[j]:
                j += 1
            if j - i > 5:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _grid_end(grid, a: float) -> float:
    """the end of the beat-covered grid stretch that starts at a: the next no-beat interval, else the grid's last row"""
    nxt = [x for x, _, _ in grid.no_beat if x > a + 1e-6]
    return float(min(nxt + [grid.t[-1]]))


def pulse_period(grid, a: float, b: float) -> float:
    m = (grid.slot % 2 == 0) & (grid.t >= a) & (grid.t <= b)
    d = np.diff(grid.t[m])
    d = d[(d > 0.2) & (d < 2.0)]
    return float(np.median(d)) if len(d) else 0.75


# ---------------------------------------------------------------------------------------------------------------- causal averaging
def _nan(fn, a, axis=None):
    """numpy's nan-reductions without the 'empty slice' warnings (an all-NaN slice stays NaN)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return fn(a, axis=axis)


def _ema(x: np.ndarray, tau_samples: float) -> np.ndarray:
    """causal exponential average (complex ok) of a 1-D array, normalised by its own weight (no start-up bias)"""
    from scipy.signal import lfilter
    a = math.exp(-1.0 / tau_samples)
    num = lfilter([1.0 - a], [1.0, -a], x)
    den = lfilter([1.0 - a], [1.0, -a], np.ones(len(x)))
    return num / np.maximum(den, 1e-12)


def _bandpass(x: np.ndarray, fs: float, f0: float, band: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    lo, hi = f0 * (1.0 - band), f0 * (1.0 + band)
    sos = butter(2, [lo / (fs / 2.0), hi / (fs / 2.0)], "band", output="sos")
    return sosfilt(sos, x)


def _phasor(ph_pulses: np.ndarray, h: int) -> np.ndarray:
    return np.exp(-2j * np.pi * h * ph_pulses / 4.0)


@dataclass
class Series:
    """what one analysis produces on the body time base (n samples). 'cum' = the long causal window (tau_long: about everything since the start of the song), 'ema' = the shorter causal window tau."""
    t: np.ndarray
    valid: np.ndarray                          # beat-covered samples
    elapsed: np.ndarray                        # seconds since the start of the sample's grid segment
    cum_body: dict                             # role -> (H, n) phase-locking of the body streams (mean over the streams)
    ema_body: dict                             # role -> (H, n) the same over the last ~tau seconds
    cum_steps: dict                            # role -> (H, n) phase concentration of the foot touchdowns (NaN before 3 steps)
    couple: np.ndarray                         # (n,) lead-follow coherence of the locked oscillations at the lead's chosen level
    best_body: dict                            # role -> (n,) index into LEVEL_H (the level with the highest cumulative body lock so far: causal)
    best_steps: dict
    phase_body: dict = field(default_factory=dict)   # role -> (H, n) locked phase in cycles (diagnostics, plots)
    n_steps: dict = field(default_factory=dict)


def analyse(grid, body: Body, steps: dict, cfg: EntrainConfig, scale: float = 1.0, partner_scale: float = 1.0) -> Series:
    """The causal entrainment series of one (possibly permuted) analysis. `scale` stretches the grid's tempo (the wrong-tempo metronome), `partner_scale` resamples the
    follow's signals about the middle of each segment (the null of the lead-follow coherence)."""
    t = body.t
    n = len(t)
    fs = 1.0 / float(np.median(np.diff(t)))
    tau = cfg.tau_s * fs
    tau_l = cfg.tau_long_s * fs
    H = len(LEVEL_H)
    valid = np.zeros(n, bool)
    elapsed = np.zeros(n)

    def nan_h():
        return {r: np.full((H, n), np.nan) for r in ROLES}

    cum_body, ema_body, cum_steps, phase_body = nan_h(), nan_h(), nan_h(), nan_h()
    couple = np.full(n, np.nan)
    best_body = {r: np.zeros(n, int) for r in ROLES}
    best_steps = {r: np.zeros(n, int) for r in ROLES}
    n_steps = {r: np.zeros(n) for r in ROLES}
    ev_t = {r: np.array(sorted(s["t"] for s in steps.get(r, [])), float) for r in ROLES}
    for (i0, i1) in beat_segments(grid, t):
        a_t, b_t = float(t[i0]), float(t[i1 - 1])
        valid[i0:i1] = True
        elapsed[i0:i1] = t[i0:i1] - a_t
        P = pulse_period(grid, a_t, _grid_end(grid, a_t))                  # the song's tempo: the music is known in advance
        mid = a_t                                                          # the wrong-tempo metronome and the replayed partner are anchored at the song's start: causal
        w0 = max(0, i0 - int(cfg.warm_s * fs))
        ts = t[w0:i1]
        ph = grid_phase(grid, ts, scale, anchor=mid)
        seg = slice(i0 - w0, None)
        L = i1 - i0
        lowpassed = {r: {} for r in ROLES}              # role -> level -> (L,) complex: the streams' baseband, short-lowpassed and amplitude-normalised (for the lead-follow coherence)
        for r in ROLES:
            for hi, h in enumerate(LEVEL_H):
                f0 = h * scale / (4.0 * P)
                if body.sig is None or not (0.05 < f0 < fs / 2.5):
                    continue
                rho_c = np.zeros(L)
                rho_e = np.zeros(L)
                zsum = np.zeros(L, complex)
                bb = np.zeros(L, complex)
                for sname in BODY_STREAMS:
                    x = body.sig[r][sname][w0:i1].astype(float)
                    if r == "follow" and partner_scale != 1.0:
                        x = np.interp((ts - mid) * partner_scale + mid, ts, x)
                    x = x - _ema(x, 10.0 * fs)                                # a causal high-pass (the band-pass removes the rest)
                    xb = _bandpass(x, fs, f0, cfg.band)[seg]
                    u = xb * _phasor(ph, h)[seg]
                    zc = _ema(u, tau_l) / np.maximum(_ema(np.abs(xb), tau_l), 1e-12) * (4.0 / math.pi)
                    ze = _ema(u, tau) / np.maximum(_ema(np.abs(xb), tau), 1e-12) * (4.0 / math.pi)
                    rho_c += np.abs(zc) / len(BODY_STREAMS)
                    rho_e += np.abs(ze) / len(BODY_STREAMS)
                    zsum += zc / len(BODY_STREAMS)
                    bb += _ema(u, 3.0 * fs) / np.maximum(_ema(np.abs(xb), tau_l), 1e-12)
                cum_body[r][hi, i0:i1] = rho_c
                ema_body[r][hi, i0:i1] = rho_e
                phase_body[r][hi, i0:i1] = np.angle(zsum) / (2 * np.pi)
                lowpassed[r][hi] = bb
            # steps: the cumulative phasor of the event train
            et = ev_t[r]
            m = (et >= t[i0]) & (et <= t[i1 - 1])
            cnt_t = np.zeros(L)
            idx = np.zeros(0, int)
            if m.any():
                idx = np.clip(np.searchsorted(t[i0:i1], et[m]), 0, L - 1)
                np.add.at(cnt_t, idx, 1.0)
            ncum = _ema(cnt_t, tau_l) * tau_l                      # steps in the long window (effective count)
            n_steps[r][i0:i1] = ncum
            for hi, h in enumerate(LEVEL_H):
                inj = np.zeros(L, complex)
                if m.any():
                    np.add.at(inj, idx, np.exp(-2j * np.pi * h * np.interp(et[m], ts, ph) / 4.0))
                rho = np.abs(_ema(inj, tau_l)) / np.maximum(_ema(cnt_t, tau_l), 1e-12)
                cum_steps[r][hi, i0:i1] = np.where(ncum >= 3, rho, np.nan)
        # the best level per role (causal): the level with the highest cumulative lock so far
        for r in ROLES:
            for src, dst in ((cum_body[r], best_body[r]), (cum_steps[r], best_steps[r])):
                dst[i0:i1] = np.argmax(np.nan_to_num(src[:, i0:i1], nan=-1.0), axis=0)
        # couple: both roles' baseband at the lead's chosen level; the grid phase cancels in the product
        zl = np.zeros(L, complex)
        zf = np.zeros(L, complex)
        for hi in lowpassed["lead"]:
            if hi in lowpassed["follow"]:
                mm = best_body["lead"][i0:i1] == hi
                zl = np.where(mm, lowpassed["lead"][hi], zl)
                zf = np.where(mm, lowpassed["follow"][hi], zf)
        num = _ema(zl * np.conj(zf), tau_l)
        den = np.sqrt(np.maximum(_ema(np.abs(zl) ** 2, tau_l) * _ema(np.abs(zf) ** 2, tau_l), 1e-12))
        couple[i0:i1] = np.abs(num) / den
    return Series(t, valid, elapsed, cum_body, ema_body, cum_steps, couple, best_body, best_steps, phase_body, n_steps)


def _smooth(x: np.ndarray, fs: float, secs: float) -> np.ndarray:
    """trailing (causal) moving average over the last `secs` seconds of a series that may hold NaN (invalid stretches are ignored)"""
    w = max(1, int(round(secs * fs)))
    ok = np.isfinite(x)
    k = np.ones(w)
    num = np.convolve(np.where(ok, x, 0.0), k)[:len(x)]
    den = np.convolve(ok.astype(float), k)[:len(x)]
    return np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)


def _at_best(a: np.ndarray, best: np.ndarray) -> np.ndarray:
    """the (H, n) statistic at the chosen level of every sample"""
    v = a[best, np.arange(a.shape[1])]
    return np.where(np.isfinite(v), v, np.nan)


def baseline_set(grid, body: Body, steps: dict, cfg: EntrainConfig) -> tuple:
    """the permutation baselines: the identical pipeline against metronomes at the wrong tempo (the grid stretched by NULL_SCALES: body, feet), and, for the lead-follow
    coherence, against the follow's signals resampled at the wrong rate (the grid unchanged)"""
    grid_nulls = [analyse(grid, body, steps, cfg, scale=s) for s in NULL_SCALES]
    partner_nulls = [analyse(grid, body, steps, cfg, partner_scale=s) for s in PARTNER_SCALES] if body.sig is not None else []
    return grid_nulls, partner_nulls


@dataclass
class Skills:
    """the normalised, causal skill series (0 = what the wrong-tempo metronomes give, 1 = the reference level) and the evidence behind them. All NaN outside the beat-covered stretches."""
    t: np.ndarray
    valid: np.ndarray
    body: np.ndarray                      # (n,) cumulative body entrainment, roles averaged
    recent: np.ndarray                    # (n,) body entrainment over the last ~tau seconds, roles averaged
    feet: np.ndarray                      # (n,) cumulative foot-touchdown entrainment, roles averaged (NaN until a few steps)
    couple: np.ndarray                    # (n,) lead-follow coherence
    feet_role: dict                       # role -> (n,)
    body_role: dict                       # role -> (n,)
    z_body: dict                          # role -> (n,) excess of the cumulative body lock over the baseline in baseline standard deviations
    raw: dict                             # the statistics before normalisation: ("body"|"recent"|"steps", role) -> (n,)
    baseline: dict                        # ("body"|"recent"|"steps", role) -> (b(t), sd(t)); "couple" -> (b, sd)
    noise_sd: dict                        # component -> (n,) 1-sigma noise of the skill under the baseline (sd of the wrong-tempo runs / span)


def calibrated(real: np.ndarray, nulls: list, ref: float, valid: np.ndarray, fs: float, smooth_s: float, min_span: float = 0.15, margin_sd: float = 0.0):
    """skill 0..1 of a statistic, calibrated by the permutation baseline: the baseline level b (mean of the wrong-tempo runs) is scaled to the reference level `ref`, and the skill the
    wrong-tempo runs themselves reach under the same scaling (the positive bias of a quantity clipped at 0) is taken off again, so that unrelated motion reads 0 on average.
    Returns (skill, mean level of the baseline runs, their sd, 1-sigma noise of the skill)."""
    N = np.array(nulls)
    b0 = _smooth(_nan(np.nanmean, N, 0), fs, smooth_s)
    sd = _smooth(_nan(np.nanstd, N, 0), fs, smooth_s)
    b = b0 + margin_sd * sd
    span = np.maximum(ref - b, min_span)
    s = np.clip((real - b) / span, 0.0, 1.0)
    sbar = _smooth(_nan(np.nanmean, np.clip((N - b) / span, 0.0, 1.0), 0), fs, smooth_s)
    out = np.clip((s - sbar) / np.maximum(1.0 - sbar, 0.2), 0.0, 1.0)
    return np.where(valid, out, np.nan), b0, sd, sd / span


def normalise(S: Series, baselines: tuple, cfg: EntrainConfig) -> Skills:
    nulls, pnulls = baselines
    t = S.t
    fs = 1.0 / float(np.median(np.diff(t)))
    valid = S.valid
    body, recent, feet, zb, raw, base, noise = {}, {}, {}, {}, {}, {}, {}
    for r in ROLES:
        # the null runs choose their own level (the same causal rule), so the price of choosing is in the baseline
        for key, ref, arr, store in (("body", cfg.ref_body, "cum_body", body), ("recent", cfg.ref_body, "ema_body", recent), ("steps", cfg.ref_steps, "cum_steps", feet)):
            sel = S.best_steps[r] if key == "steps" else S.best_body[r]
            M = np.where(valid, _at_best(getattr(S, arr)[r], sel), np.nan)
            Nk = [np.where(ns.valid, _at_best(getattr(ns, arr)[r], ns.best_steps[r] if key == "steps" else ns.best_body[r]), np.nan) for ns in nulls]
            sk, b, sd, nsd = calibrated(M, Nk, ref, valid, fs, cfg.base_smooth_s, margin_sd=cfg.base_margin_sd)
            store[r] = sk
            raw[(key, r)] = M
            base[(key, r)] = (b, sd)
            noise[(key, r)] = nsd
            if key == "body":
                zb[r] = np.where(valid, (M - b) / np.maximum(sd, 0.02), np.nan)
    couple, cb, csd, cn = calibrated(S.couple, [ns.couple for ns in (pnulls or nulls)], cfg.ref_couple, valid, fs, cfg.base_smooth_s, margin_sd=cfg.base_margin_sd)
    base["couple"] = (cb, csd)

    def mean2(d):
        return _nan(np.nanmean, np.vstack([d[r] for r in ROLES]), 0)

    def noise2(k):
        return np.sqrt(_nan(np.nansum, np.vstack([noise[(k, r)] ** 2 for r in ROLES]), 0)) / 2.0

    nz = {"body": noise2("body"), "recent": noise2("recent"), "feet": noise2("steps"), "couple": cn}
    return Skills(t, valid, mean2(body), mean2(recent), mean2(feet), couple, feet, body, zb, raw, base, nz)
