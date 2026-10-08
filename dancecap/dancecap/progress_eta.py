"""ETA, budget and rate history for the pipeline monitor (docs/PIPELINE_PROGRESS.md sections 6-9). Read-only.

    snapshot(session_dir)  -> {state, level, phases[], current, eta {p10, p50, p90, branches}, kill {..},
                              cost {so_far, p50, p90, worst}, pod {..}, warnings[]}       (one Runpod session)
    Tracker(session_dir)   -> incremental reader for a live session (stamps new runner.log lines with their receive
                              time, so live rates are exact even at L0); .snapshot()
    live_snapshot(dir)     -> what a long-lived monitor calls: one background-polled Tracker per watched session
    budget()               -> ledger spent (= remote.spent()) + live cost + projected p50/p90 + worst case + warnings
    launch_gate(p90_cost)  -> the monitor's stricter pre-launch check (counts running pods' worst case)
    estimate_session(...)  -> p50/p90 minutes and $ per phase for a session that has not started (plan dialog)
    rates_rebuild()        -> work/progress/rates.json from every past session (derived, rebuildable)
    camera_types(take)     -> the EFFECTIVE camera type per source video (dancecap.camera_types.stage_types under the
                              active policy) + confirmed/suggested; loc_units / loc_estimate / estimate_front price it

Everything here only reads session files (open, read, close; partial JSON retried) and writes only under
work/progress/. Estimates are (p10, p50, p90) seconds. Stdlib only (the monitor imports this without numpy).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

from . import progress as P
from .take import PACKAGE_ROOT

WORK = PACKAGE_ROOT / "work"
SESSIONS = WORK / "runpod" / "sessions"
LEDGER = WORK / "runpod" / "ledger.json"
RATES = WORK / "progress" / "rates.json"
Z90 = 1.2816
HIST_SIGMA_FLOOR = 0.15       # sigma floor of a history rate with < 10 samples (one take / host: overconfident)
DEFAULT_RESERVE = 2.0
CHEAPEST_80GB_PER_HR = 1.19   # A100 PCIe community, 2026-10-07
KILL_START_TIMEOUT_MIN = 6.0  # remote.pod_session start_timeout_min (budget worst case per pod)
# GPU speed relative to A100-SXM4 (placeholders until measured; docs/PIPELINE_PROGRESS.md 7)
GPU_FACTOR = {"A100-SXM4": 1.0, "A100-PCIe": 1.05, "H100": 0.7, "L40S": 1.1, "A40": 1.5}

# seeds (code constants and measured 2026-10-06/07) used until rates.json has samples for a key
SEED = {
    "session/upload/MBps": {"p10": 0.7, "p50": 2.5, "p90": 2.8},          # p10 = slow end
    "session/download/MBps": {"p10": 1.5, "p50": 12.0, "p90": 16.0},
    "session/pod_start/s": {"p10": 16, "p50": 34, "p90": 48},
    "session/bundle_scan/s": {"p10": 1, "p50": 2, "p90": 4},
    "session/bundle_build/s": {"p10": 0, "p50": 0, "p90": 140},
    "session/extract/s": {"p10": 10, "p50": 15, "p90": 25},
    "session/wheels_back/s": {"p10": 1, "p50": 1, "p90": 5},
    "session/archive/s": {"p10": 8, "p50": 15, "p90": 30},
    "session/pack/s": {"p10": 1, "p50": 3, "p90": 10},
    "session/job_extract/s": {"p10": 0, "p50": 1, "p90": 5},
    "session/delete/s": {"p10": 2, "p50": 4, "p90": 10},
    "texture/pass0/s_per_view": {"p50": 0.063, "sigma": 0.15},           # A100-SXM4 (avatar_pod 0.061)
    "texture/aligned/s_per_view": {"p50": 0.134, "sigma": 0.15},         # (avatar_pod 0.114: 18 % optimistic)
    # fixed (non-pass) time of a bake, measured on the live-recorded 2026-10-07 re-bake (20261007-110948-batch, 2292
    # views): pre = load + head fix + hair model + atlas 294 s, post = fusion + fills + review images 28 s
    "texture/pre/s_per_view": {"p50": 0.128, "sigma": 0.3},
    "texture/post/s_per_view": {"p50": 0.0122, "sigma": 0.4},
    "mamma/lane/ratio": {"p50": 1.0, "sigma": 0.12},                     # measured lane s / mamma_chunks model
    "mamma/fallback/ratio": {"p50": 0.80, "sigma": 0.2},                 # 3182 s measured / 66 min printed
    "mamma/gate_fail": {"p": 0.5, "n": 1},
    "room/train/s": {"p50": 1944, "sigma": 0.15, "N": 30000},
    # dancer splats (estimate for an A100: 0.15-0.17 s/step measured on the shared 2080 Ti at pod scale - 171k
    # Gaussians, 4 native-resolution views, launch-bound - / ~2.7); replaced by history (rates_rebuild) after a run
    "dancer_splats/train/s_per_step": {"p50": 0.06, "sigma": 0.35},
    "dancer_splats/eval/s_per_view": {"p50": 1.2, "sigma": 0.5},
}
BOOT_SEED_MIN = {"base": (1.4, 1.5, 6.5), "mamma": (0.35, 0.5, 3.7), "room": (0.3, 0.4, 3.0), "texture": (0, 0, 0.1),
                 "dancers": (1.5, 2.5, 5.2), "cameras": (0.2, 1.0, 3.8), "ma2d": (0.5, 0.8, 1.2),
                 "check": (0.1, 0.12, 0.3), "gsplat": (0.2, 0.3, 2.0)}
MAMMA_Q_USED = (0.4, 0.6, 1.0, 0.8)      # seed share of N_max actually run per optimisation run (early stopping)
# room gsplat time profile F(x) at x = 0, .05, .., 1 (2026-10-06 pod run, 30000 steps, 1 M cap)
ROOM_PROFILE = [0.0, 0.0335, 0.0752, 0.1183, 0.1655, 0.2153, 0.2697, 0.3237, 0.3787, 0.4313, 0.4857, 0.5401, 0.5931,
                0.6463, 0.6995, 0.7523, 0.8053, 0.8577, 0.9113, 0.9630, 1.0]


# ---------------------------------------------------------------------------------------------------- estimates


def est(p50: float, sigma: float = 0.3, floor: float = 0.0) -> tuple[float, float, float]:
    """(p10, p50, p90) seconds from a median and a log-normal spread."""
    p50 = max(floor, float(p50))
    return (p50 * math.exp(-Z90 * sigma), p50, p50 * math.exp(Z90 * sigma))


def add(*es) -> tuple[float, float, float]:
    """Independent steps in sequence: p50 = sum; band = p50 -/+ sqrt(sum of squared half-bands)."""
    es = [e for e in es if e]
    if not es:
        return (0.0, 0.0, 0.0)
    p50 = sum(e[1] for e in es)
    lo = math.sqrt(sum((e[1] - e[0]) ** 2 for e in es))
    hi = math.sqrt(sum((e[2] - e[1]) ** 2 for e in es))
    return (max(0.0, p50 - lo), p50, p50 + hi)


def add_p90(*es) -> float:
    """Conservative (budget) p90 of a sequence: the sum of the p90s."""
    return sum(e[2] for e in es if e)


def lanes_max(es) -> tuple[float, float, float]:
    """Parallel lanes: the slowest lane (per quantile; conservative, no speed-up when others finish)."""
    es = [e for e in es if e]
    if not es:
        return (0.0, 0.0, 0.0)
    return (max(e[0] for e in es), max(e[1] for e in es), max(e[2] for e in es))


def scale(e, f: float):
    return (e[0] * f, e[1] * f, e[2] * f)


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def _interp_pts(pts: list[tuple[float, float]], x: float) -> float:
    """Piecewise-linear y(x) through sorted (x, y) points, clamped at the ends."""
    if x <= pts[0][0]:
        return pts[0][1]
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if x <= xb:
            return ya if xb <= xa else ya + (yb - ya) * (x - xa) / (xb - xa)
    return pts[-1][1]


def _pass_profile_eta(prev_obs, prev_end: float | None, prev_s: float | None, obs, views: int, diag: dict,
                      rho_prior: float = 1.0):
    """Remaining seconds of a bake pass from the PREVIOUS pass's time profile F(x) (fraction of its seconds spent when
    it had done x of the views; the passes walk the views in the same order). rho = this pass's time to its last tick
    / the previous pass's time to the same view, shrunk to rho_prior while x is small; R = rho x prev_s x (1 - F(x))
    minus the time since the last tick (only STALL_K gaps of it once stalled, like remaining_linear). Returns
    (R, projected seconds of this whole pass) or None when the previous pass has no usable timeline."""
    if not prev_obs or not prev_s or prev_end is None or len(prev_obs) < 8 or not obs:
        return None
    t_start = prev_end - prev_s
    pts = [(0.0, 0.0)]
    for t, d in _advances(prev_obs):
        if t is None:
            return None
        f = (t - t_start) / prev_s
        if 0.0 <= f <= 1.0 and d / views > pts[-1][0] and f >= pts[-1][1]:
            pts.append((d / views, f))
    pts.append((1.0, 1.0))
    if len(pts) < 8:
        return None
    adv = _advances(obs)
    tl, dl = adv[-1]
    x = dl / views
    F = _interp_pts(pts, x)
    el = tl - prev_end  # this pass starts where the previous one ended
    el -= live_rate_info(obs, window_s=1e12)["excess_s"]  # a resumed pause is not this pass's speed
    if F < 0.03 or el <= 0:
        return None
    rho = el / (prev_s * F)
    w = x / (x + 0.05)
    rho_hat = math.exp(w * math.log(_clamp(rho, 0.2, 5.0)) + (1 - w) * math.log(max(1e-3, rho_prior)))
    R_tick = rho_hat * prev_s * (1.0 - F)
    since = float(diag.get("since_advance_s") or 0.0)
    gap = float(diag.get("med_gap_s") or 0.0)
    R = max(R_tick * 0.1, R_tick - (min(since, STALL_K * gap) if diag.get("stalled") else since))
    return R, rho_hat * prev_s


def blend_rate(s_live: float | None, s_hist: float | None, x: float, n_hist: int, sigma_hist: float = 0.3,
               sigma_live: float = 0.1, x0: float | None = None) -> tuple[float, float]:
    """Seconds per unit blended in log space, weighted by the fraction done x: w = x / (x + x0). Returns (s, sigma).
    x0 (optional) overrides the default 0.05 / 0.02: larger = trust the live rate later (estimated tick times)."""
    if s_live is None or s_live <= 0:
        return (s_hist or 0.0), sigma_hist
    if not s_hist or s_hist <= 0:
        return s_live, max(0.05, sigma_live)
    x0 = x0 if x0 is not None else (0.05 if n_hist >= 3 else 0.02)
    w = x / (x + x0) if x > 0 else 0.0
    s = math.exp(w * math.log(s_live) + (1 - w) * math.log(s_hist))
    return s, (1 - w) * sigma_hist + w * max(0.05, sigma_live)


PAUSE_K, PAUSE_MIN_S = 5.0, 30.0      # an interval > 5x the window's median s/unit and > 30 s is a pause (dropped)
STALL_K = 3.0                          # no advance for > 3x the median tick gap: stop counting the step down
STALL_WARN_MIN_S, STALL_WARN_K = 300.0, 10.0  # warn when a step with a progress signal is stuck max(5 min, 10x gap)


def _advances(obs) -> list[tuple[float, float]]:
    """(t, done) observations reduced to the moments `done` increased (repeated / heartbeat values carry no rate:
    their time is part of the next advance's interval). The first observation is kept as the origin."""
    out: list[tuple[float, float]] = []
    for t, d in obs:
        if t is None or d is None:
            continue
        if not out:
            out.append((t, d))
        elif d > out[-1][1]:
            if t <= out[-1][0] and len(out) > 1:  # same receive time (lines read in one batch): one observation
                out[-1] = (out[-1][0], d)
            else:
                out.append((max(t, out[-1][0]), d))
    return out


def live_rate_info(obs: list[tuple[float, float]], window_s: float = 120.0, min_ticks: int = 4) -> dict:
    """Robust seconds per unit over the latest window of WORKING time (>= window_s and >= min_ticks intervals).

    The observations are cut into intervals between advances of `done`. An interval slower than PAUSE_K x the median
    per-unit rate of the last 30 intervals and longer than PAUSE_MIN_S is a pause (a stall that resumed, a hiccup): it
    is left out of BOTH the mean and the spread, and the window reaches further back instead, so one resumed stall
    cannot inflate the rate (and the cost / kill-switch projections) while the band stays falsely narrow.
    s = sum(dt) / sum(dd) over the kept intervals; sigma = MAD of ln of their per-interval rates / sqrt(n) (floor 0.05;
    0.1 when a pause was dropped, 0.3 with fewer than 3 intervals).
    Returns {s (None when < 20 s or < 1 unit kept), sigma, n, dropped, dropped_s, excess_s (the pauses' own time),
    med_gap_s, unstable}."""
    adv = _advances(obs)
    out = {"s": None, "sigma": 0.0, "n": 0, "dropped": 0, "dropped_s": 0.0, "excess_s": 0.0, "med_gap_s": None,
           "unstable": True}
    if len(adv) < 2:
        return out
    iv = [(tb - ta, db - da) for (ta, da), (tb, db) in zip(adv, adv[1:]) if tb > ta and db > da][-400:]
    if not iv:
        return out
    tail = sorted(dt for dt, _ in iv[-30:])
    out["med_gap_s"] = tail[len(tail) // 2]
    recent = iv[-30:]
    pause = lambda dt, dd: False  # noqa: E731
    med = None
    if len(recent) >= 3:
        rs = sorted(dt / dd for dt, dd in recent)
        med = rs[len(rs) // 2]
        pause = lambda dt, dd: dt / dd > PAUSE_K * med and dt > PAUSE_MIN_S  # noqa: E731
    elif len(recent) == 2:  # two intervals: the slower one is a pause only when it is far slower and long
        a_, b_ = sorted(recent, key=lambda x: x[0] / x[1])
        if b_[0] / b_[1] > PAUSE_K * a_[0] / a_[1] and b_[0] > PAUSE_MIN_S:
            pause = lambda dt, dd: (dt, dd) == b_  # noqa: E731
            med = a_[0] / a_[1]
    keep: list[tuple[float, float]] = []
    T = 0.0
    for dt, dd in reversed(iv):
        if T >= window_s and len(keep) >= min_ticks:
            break
        if pause(dt, dd):
            out["dropped"] += 1
            out["dropped_s"] += dt
            out["excess_s"] += dt - dd * (med or 0.0)  # the pause itself (its units took the usual time)
            continue
        keep.append((dt, dd))
        T += dt
    out["dropped_s"] = round(out["dropped_s"], 1)
    out["excess_s"] = round(out["excess_s"], 1)
    D = sum(dd for _, dd in keep)
    out["n"] = len(keep)
    if D < 1 or T < 20:
        return out
    out["s"] = T / D
    logs = [math.log(dt / dd) for dt, dd in keep]
    if len(logs) >= 3:  # spread of the window's MEAN rate (per-interval jitter / sqrt(n)); receive-time quantisation
        med = sorted(logs)[len(logs) // 2]  # of L0 lines makes single intervals noisy
        sig = max(0.05, 1.4826 * sorted(abs(r - med) for r in logs)[len(logs) // 2] / math.sqrt(len(logs)))
    else:
        sig = 0.3
    if out["dropped"]:
        sig = max(sig, 0.1)
    out["sigma"] = sig
    # one interval dominating the window (few intervals, or a pause just dropped): not a basis for a red alarm
    out["unstable"] = len(keep) < 3 or (out["dropped"] > 0 and len(keep) < 5) or max(dt for dt, _ in keep) > 0.6 * T
    return out


def live_rate(obs: list[tuple[float, float]], window_s: float = 120.0, min_ticks: int = 4) -> tuple[float | None, float]:
    """(seconds per unit, sigma_live) of live_rate_info: (None, 0) when the window has < 20 s or < 1 unit."""
    i = live_rate_info(obs, window_s, min_ticks)
    return (i["s"], i["sigma"]) if i["s"] is not None else (None, 0.0)


def remaining_linear(N: float, done: float, obs, hist_s: float | None, n_hist: int = 1, sigma_hist: float = 0.3,
                     tail_s: float = 0.0, warmup_s: float = 0.0, started_at: float | None = None,
                     now: float | None = None, diag: dict | None = None, x0: float | None = None):
    """Section 6.2: remaining (p10, p50, p90) of a linear step with N units, done so far, observations (t, done).

    The time since the last ADVANCE of `done` (not the last event: a heartbeat or a repeated value is no progress)
    counts against R while it is within STALL_K x the typical gap between advances. Beyond that the step is stalled:
    R stops shrinking (only STALL_K gaps are credited) and p90 grows by the stall so far (it may last as long again).
    diag (optional dict) receives {s_live, sigma_live, dropped, med_gap_s, since_advance_s, stalled, unstable}."""
    now = time.time() if now is None else now
    if not N:
        return est(tail_s + warmup_s, sigma_hist)
    if done <= 0 or not obs:
        if started_at is not None:
            left_warm = max(0.0, warmup_s - (now - started_at))
            return est(left_warm + N * (hist_s or 0) + tail_s, sigma_hist)
        return est(warmup_s + N * (hist_s or 0) + tail_s, sigma_hist)
    x = min(1.0, done / N)
    info = live_rate_info(obs)
    s_live, sig_live = (info["s"], info["sigma"]) if info["s"] is not None else (None, 0.0)
    if x0 is not None and s_live is not None:  # estimated tick times: a wider live spread too
        sig_live = max(sig_live, 0.2)
    s, sig = blend_rate(s_live, hist_s, x, n_hist, sigma_hist, sig_live, x0)
    R0 = max(0.0, (N - done) * s) + tail_s
    adv = _advances(obs)
    t_adv = adv[-1][0] if adv else None
    gap = info["med_gap_s"]
    if gap is None and adv and started_at is not None and adv[-1][1] > 0:
        gap = max(1.0, (adv[-1][0] - started_at) / max(1, len(adv)))
    since = (now - t_adv) if (t_adv is not None and now > t_adv) else 0.0
    stalled = bool(gap) and since > max(STALL_K * gap, 10.0)
    if R0 <= 0:
        R = 0.0
    elif stalled:
        R = max(R0 * 0.1, R0 - STALL_K * gap)
    else:
        R = max(R0 * 0.1, R0 - since)
    lo, hi = R * math.exp(-Z90 * sig), R * math.exp(Z90 * sig)
    if stalled:
        hi += since - STALL_K * gap
    if diag is not None:
        diag.update({"s_live": s_live, "sigma_live": sig_live, "dropped": info["dropped"], "med_gap_s": gap,
                     "since_advance_s": round(since, 1), "stalled": stalled,
                     "unstable": bool((info["unstable"] or x0 is not None) and s_live is not None and x < 0.98)})
    return (lo, R, hi)


def remaining_profile(T_h: float, profile: list[float], x: float, elapsed: float, n_hist: int = 1,
                      sigma_hist: float = 0.15, tail_s: float = 0.0):
    """Section 6.3: non-linear step (gsplat densification): rho = elapsed / (T_h F_h(x)), shrunk to 1 while x is small."""
    F = _interp_profile(profile, x)
    if x <= 0 or F <= 0:
        return est(T_h + tail_s, sigma_hist)
    rho = elapsed / (T_h * F)
    x0 = 0.05 if n_hist >= 3 else 0.02
    w = x / (x + x0)
    rho_hat = math.exp(w * math.log(max(rho, 1e-6)))
    R = rho_hat * T_h * (1 - F) + tail_s
    sig = (1 - w) * sigma_hist + w * 0.05
    return (R * math.exp(-Z90 * sig), R, R * math.exp(Z90 * sig))


def _interp_profile(profile: list[float], x: float) -> float:
    if not profile:
        return x
    x = min(1.0, max(0.0, x))
    n = len(profile) - 1
    i = min(n - 1, int(x * n))
    f = x * n - i
    return profile[i] * (1 - f) + profile[i + 1] * f


def expected_total(n_max: int, done: int, q: float) -> int:
    """Section 6.4: early-stopped runs: E[N] = clip(q N_max, done + 1, N_max)."""
    return int(min(n_max, max(done + 1, round(q * n_max))))


def remaining_after_elapsed(dur, elapsed: float):
    """A step with no progress signal, running for `elapsed`: remaining = duration quantiles minus elapsed, floored
    so an overdue step never reads 0 (conditional tail: at least 15 % of the p50 or 10 s)."""
    floor = max(10.0, 0.15 * dur[1])
    r = [max(d - elapsed, 0.0) for d in dur]
    r50 = max(r[1], floor if elapsed > 0 else 0.0)
    if elapsed > dur[1]:  # overdue (a stall-prone, heavy-tailed step): what is left grows with what has been spent
        r50 = max(r50, 0.25 * elapsed)
    r90 = max(r[2], r50 * 2.0 if elapsed > dur[1] else (r50 * 1.5 if r[2] <= r50 else r[2]))
    r10 = min(max(r[0], 0.0), r50)
    return (r10, r50, r90)


def simulate_dag(tasks: dict, caps: dict | None = None, now: float = 0.0) -> float:
    """Front DAG list scheduling (section 6.5). tasks: name -> {kind, deps[], after[], state ('ok'|'running'|'pending'|
    'failed'|'skipped'), remaining (s, running), duration (s, pending)}. Returns the makespan from now (seconds).
    Priority: the longest path to the end (critical path first). caps: max running tasks per kind."""
    caps = caps or {}
    done_states = {"ok", "failed", "skipped", "killed", "done"}
    left = {k: dict(v) for k, v in tasks.items() if v.get("state") not in done_states}
    if not left:
        return 0.0
    dur = {k: float(v.get("remaining") if v.get("state") == "running" else v.get("duration") or 0.0)
           for k, v in left.items()}
    succ: dict[str, list[str]] = {k: [] for k in left}
    for k, v in left.items():
        for d in list(v.get("deps", [])) + list(v.get("after", [])):
            if d in left:
                succ[d].append(k)
    memo: dict[str, float] = {}

    def tail(k, depth=0):
        if k in memo:
            return memo[k]
        if depth > 200:
            return dur[k]
        memo[k] = dur[k] + max((tail(s, depth + 1) for s in succ[k]), default=0.0)
        return memo[k]
    t = 0.0
    running: dict[str, float] = {k: dur[k] for k, v in left.items() if v.get("state") == "running"}
    pending = [k for k in left if k not in running]
    finished: set[str] = set()
    guard = 0
    while (running or pending) and guard < 10000:
        guard += 1
        count = {}
        for k in running:
            kd = left[k].get("kind", "cpu")
            count[kd] = count.get(kd, 0) + 1
        ready = [k for k in pending if all(d not in left or d in finished
                                           for d in list(left[k].get("deps", [])) + list(left[k].get("after", [])))]
        ready.sort(key=lambda k: -tail(k))
        for k in ready:
            kd = left[k].get("kind", "cpu")
            if count.get(kd, 0) < caps.get(kd, 10 ** 6):
                running[k] = t + dur[k]
                pending.remove(k)
                count[kd] = count.get(kd, 0) + 1
        if not running:
            break  # unsatisfiable deps: stop
        k_next = min(running, key=running.get)
        t = running.pop(k_next)
        finished.add(k_next)
    return t


# ---------------------------------------------------------------------------------------------------- history

# mamma_chunks' cost model (ITER_FIXED_S, ITER_PER_FRAME_S, N_ITERS, PARALLEL_SLOWDOWN), copied so the monitor never
# imports numpy/mamma_dyn; tests/test_progress.py checks they still match mamma_chunks.
MAMMA_ITER_FIXED_S, MAMMA_ITER_PER_FRAME_S, MAMMA_N_ITERS, MAMMA_PARALLEL_SLOWDOWN = 0.40, 0.0048, 675, 2.1


def mamma_solo_s(frames: int) -> float:
    return (MAMMA_ITER_FIXED_S + MAMMA_ITER_PER_FRAME_S * frames) * MAMMA_N_ITERS


def mamma_lane_model_s(frames: int, lanes: int) -> float:
    """One chunk lane side by side with `lanes` - 1 others under MPS (mamma_chunks.plan's estimate incl. the held-out
    run's (n + 1) / n)."""
    lanes = max(1, int(lanes or 1))
    if lanes == 1:
        return mamma_solo_s(frames)
    return mamma_solo_s(frames) * MAMMA_PARALLEL_SLOWDOWN * lanes / max(1, lanes - 1)



# pod commands that keep the GPU busy (history keys of _cmd_key; prefix + "."): only these raise "GPU idle" (CPU-bound
# commands - timing, physics, pose_quality, the MANO fits, front_pod's localisation / sync - idle the GPU by design)
GPU_COMMANDS = ("texture", "mamma", "mamma_chunks.fit", "mamma_chunks.gate", "mamma_loo", "room", "avatar_face",
                "dancer_splats",
                "detect", "dancers", "dancers_geoveto", "hand_worker", "refine_hands.detectors", "pose_refine",
                "texdev")


def gpu_command(key: str | None) -> bool:
    k = key or ""
    return any(k == g or k.startswith(g + ".") for g in GPU_COMMANDS)


def gpu_class(gpu: str | None) -> str:
    g = (gpu or "").upper()
    if "A100" in g:
        return "A100-SXM4" if "SXM" in g else "A100-PCIe"
    for k in ("H100", "L40S", "A40", "L40", "RTX 4090", "A6000"):
        if k in g:
            return k.replace(" ", "")
    return g.replace("NVIDIA ", "").split()[0] if g else "?"


def _wq(xs: list[float], ws: list[float], f: float) -> float:
    """Weighted quantile (linear in cumulative weight)."""
    pairs = sorted(zip(xs, ws))
    tot = sum(w for _, w in pairs)
    acc = 0.0
    for x, w in pairs:
        acc += w
        if acc >= f * tot - 1e-12:
            return x
    return pairs[-1][0]


def _stats(samples: list[float], half_life: float = 3.0) -> dict:
    """Quantiles of the last 10 samples, recency-weighted (half-life 3 samples) so a new code path takes over within
    a couple of runs; sigma = MAD of ln over the positive samples."""
    xs = [float(x) for x in samples[-10:] if x is not None and x == x]
    if not xs:
        return {"n": 0}
    n = len(xs)
    ws = [0.5 ** ((n - 1 - i) / half_life) for i in range(n)]
    out = {"n": n, "p10": round(_wq(xs, ws, 0.1), 4), "p50": round(_wq(xs, ws, 0.5), 4),
           "p90": round(_wq(xs, ws, 0.9), 4)}
    pos = [math.log(x) for x in xs if x > 0]
    if len(pos) >= 2:
        med = sorted(pos)[len(pos) // 2]
        out["sigma"] = round(max(0.05, 1.4826 * sorted(abs(v - med) for v in pos)[len(pos) // 2]), 3)
    return out


def load_rates(path: Path = RATES) -> dict:
    d = P.read_json(path)
    return d if isinstance(d, dict) and "keys" in d else {"v": 1, "keys": {}}


def lookup(rates: dict | None, key: str, gpu: str | None = None, field: str = "p50"):
    """Exact key@gpu, then the same key on another GPU x GPU factor, then the key without GPU, then SEED.
    Returns (value, n, sigma, source)."""
    keys = (rates or {}).get("keys", {})
    gc = gpu_class(gpu) if gpu else None
    if gc and f"{key}@{gc}" in keys and keys[f"{key}@{gc}"].get("n"):
        r = keys[f"{key}@{gc}"]
        return r.get(field, r.get("p50")), r["n"], r.get("sigma", 0.3), "history"
    if gc:
        for k, r in keys.items():
            if k.startswith(key + "@") and r.get("n") and r.get(field) is not None:
                other = k.split("@", 1)[1]
                f = GPU_FACTOR.get(gc, 1.0) / GPU_FACTOR.get(other, 1.0) if key.endswith(("s_per_view", "/s",
                                                                                          "s_per_frame")) else 1.0
                return r[field] * f, r["n"], max(r.get("sigma", 0.3), 0.2), f"history ({other})"
    if key in keys and keys[key].get("n"):
        r = keys[key]
        return r.get(field, r.get("p50")), r["n"], r.get("sigma", 0.3), "history"
    s = SEED.get(key)
    if s:
        return s.get(field, s.get("p50")), 0, s.get("sigma", 0.3), "seed"
    return None, 0, 0.3, "none"


def _cmd_key(job: str, cmd: dict | None) -> str:
    """History key of a pod command from its command line: 'texture', 'avatar_face', 'mamma_chunks.fit', ..."""
    c = (cmd or {}).get("cmd", "") if isinstance(cmd, dict) else str(cmd or "")
    m = re.search(r"python[0-9.]* -m dancecap(?:\.(\w+))?(?: ([a-z][a-z0-9_-]{0,30})(?=[\s;&|)>]|$))?", c)
    if m:
        mod, verb = m.group(1), m.group(2)
        if mod and verb and "_" not in verb:
            return f"{mod}.{verb}"
        if mod:
            return mod
        if verb:
            return verb if "_" not in verb else "dancecap"
    name = (cmd or {}).get("name", "") if isinstance(cmd, dict) else ""
    return f"{re.sub(r'[0-9]+$', '', job)}.{re.sub(r'[0-9]+$', '', name)}"


def rates_rebuild(sessions: Path = SESSIONS, out: Path = RATES, ledger_rows: list | None = None) -> dict:
    """work/progress/rates.json from every session (state.json timings, L0 parses of runner.log, bootstrap timings) and
    the downloaded MAMMA logs. Derived and rebuildable; never touches the session dirs."""
    rows = ledger_rows if ledger_rows is not None else (P.read_json(LEDGER) or [])
    created = {r["pod_id"]: r for r in rows if isinstance(r, dict) and r.get("pod_id")}
    S: dict[str, list] = {}

    def put(key, v):
        if v is not None and v == v:
            S.setdefault(key, []).append(float(v))
    sess_dirs = sorted([d for d in Path(sessions).glob("*") if d.is_dir()]) if Path(sessions).exists() else []
    gates = []
    for d in sess_dirs:
        st = P.read_json(d / "state.json") or {}
        pod = st.get("pod") or {}
        gc = gpu_class(pod.get("gpu")) if pod.get("gpu") else None
        at = f"@{gc}" if gc else ""
        row = created.get(pod.get("id"))
        if row and pod.get("started"):
            put("session/pod_start/s", pod["started"] - row["created_ts"])
        specs = {j: (v.get("spec") or {}) for j, v in (st.get("jobs") or {}).items()}
        for t in st.get("timings", []):
            if not t.get("ok"):
                continue
            stp, sec = t.get("step", ""), t.get("seconds")
            if stp == "upload" and t.get("mb") and sec and sec > 30:
                put("session/upload/MBps", t["mb"] / sec)
            elif stp.endswith("/download") and (t.get("mb") or 0) > 30 and sec:
                put("session/download/MBps", t["mb"] / sec)
            elif stp in ("bundle_scan", "bundle_build", "extract", "wheels_back"):
                put(f"session/{stp}/s", sec)
            elif stp.endswith("/archive"):
                put("session/archive/s", sec)
            elif stp.endswith("/pack"):
                put("session/pack/s", sec)
            elif stp.endswith("/extract"):
                put("session/job_extract/s", sec)
            elif "/" in stp and t.get("job"):
                job, cname = stp.split("/", 1)
                cmd = next((c for c in specs.get(job, {}).get("commands", []) if c.get("name") == cname), None)
                rcs = ((st.get("jobs") or {}).get(job) or {}).get("commands") or {}
                if cmd and sec and sec > 5 and rcs.get(cname) == 0:
                    put(f"cmd/{_cmd_key(job, cmd)}/s{at}", sec)
        tsv = d / "bootstrap_timings.tsv"
        try:
            for line in tsv.read_text().splitlines():
                p = line.split("\t")
                if len(p) == 3 and p[2] == "0":
                    put(f"session/bootstrap/{p[0]}/s", float(p[1]))
        except OSError:
            pass
        evs, _, _ = P.parse_runner_log(d / "runner.log")
        gate_s = None
        views = None
        pass_s = {}
        for e in evs:
            src, kind, x = e.get("src"), e.get("kind"), e.get("extra") or {}
            if src == "texture":
                if kind == "plan":
                    views, pass_s = x.get("views"), {}
                elif kind == "end" and e["step"].startswith("bake/pass") and views:
                    k = int(e["step"][9:])
                    pass_s[k] = x.get("seconds")
                    put(f"texture/{'pass0' if k == 0 else 'aligned'}/s_per_view{at}", x["seconds"] / views)
                elif kind == "end" and e["step"] == "bake/finish" and views and pass_s:
                    fixed = x.get("seconds_total", 0) - sum(pass_s.values())
                    if fixed > 0:
                        put(f"texture/fixed/s_per_view{at}", fixed / views)
            elif src == "chunks":
                if kind == "end" and x.get("seconds") and x.get("frames"):
                    lanes = next(((y.get("extra") or {}).get("lanes") for y in evs if y.get("src") == "chunks"
                                  and y.get("kind") == "plan" and (y.get("extra") or {}).get("lanes")), 7)
                    put("mamma/lane/ratio", x["seconds"] / mamma_lane_model_s(x["frames"], lanes))
                elif kind == "note" and "gate" in (x or {}):
                    gate_s = 0 if x["gate"] else 1  # merge and gate both print it: one per session
                elif kind == "plan" and e.get("eta_s"):
                    pass
            elif src == "room" and kind == "end" and x.get("elapsed_s"):
                put(f"room/train/s{at}", x["elapsed_s"] * 30000 / max(1, e.get("total") or 30000))
            elif src == "splats" and kind == "end" and e.get("step", "").endswith("splats/train") and \
                    x.get("elapsed_s") and e.get("done"):
                put(f"dancer_splats/train/s_per_step{at}", x["elapsed_s"] / max(1, e["done"]))
            elif src == "hands" and kind == "tick" and x.get("elapsed_s") and e.get("done"):
                pass
        if gate_s is not None:
            gates.append(gate_s)
        # L1/L2: the pod's own events (bake/load and bake/finish split the texture's fixed time)
        pod_evs, _ = P.read_jsonl(d / "progress" / "pod.jsonl")
        pviews = None
        for e in pod_evs:
            x = e.get("extra") or {}
            if e.get("kind") == "plan" and e.get("step") == "bake":
                pviews = x.get("views")
            elif e.get("kind") == "end" and pviews and e.get("step") == "bake/load" and x.get("seconds"):
                put(f"texture/pre/s_per_view{at}", x["seconds"] / pviews)
            elif e.get("kind") == "end" and pviews and e.get("step") == "bake/finish" and x.get("seconds"):
                put(f"texture/post/s_per_view{at}", x["seconds"] / pviews)
        # gate fallback duration: the gate command of a session whose gate failed
        if any(e.get("src") == "chunks" and e.get("kind") == "plan" and (e.get("extra") or {}).get("branch") ==
               "fallback" for e in evs):
            fb = next((e for e in evs if e.get("src") == "chunks" and e.get("kind") == "plan" and e.get("eta_s")), None)
            gt = next((t for t in st.get("timings", []) if t.get("step", "").endswith("/gate") and t.get("ok")), None)
            if fb and gt:
                put("mamma/fallback/ratio", gt["seconds"] / fb["eta_s"])
    # front task seconds per frame, from downloaded front_pod.json files (tasks run in that session only)
    for fp in sorted(WORK.glob("*/front/front_pod.json")):
        doc = P.read_json(fp) or {}
        fr = take_frames(fp.parent.parent.name)
        tasks = doc.get("tasks") or {}
        total = sum(fr.values())
        for name, t in tasks.items():
            if t.get("state") != "ok" or not t.get("seconds"):
                continue
            parts = name.split(":")
            kind = parts[0]
            if kind in ("sam", "ma2d", "sub") and len(parts) == 2 and fr.get(parts[1]):
                put(f"front/{kind}/s_per_frame", t["seconds"] / fr[parts[1]])
            elif kind == "loc" and len(parts) >= 2 and fr.get(parts[1]) and t.get("kind") == "loc":
                n = fr[parts[1]] / (2 if len(parts) == 3 else 1)
                put("front/loc/s_per_frame", t["seconds"] / n)
            elif kind in ("cameras", "sync") and total:
                put(f"front/{kind}/s_per_frame", t["seconds"] / total)
    # MAMMA iterations actually run per optimisation run (early stopping), from downloaded ma_3d logs
    for log in sorted(WORK.glob("*/mamma/logs/ma_3d_*.log"))[-40:]:
        try:
            parser = P.LineParser(("mamma_iter",))
            with open(log, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    for e in parser.feed(line):
                        if e["kind"] == "end" and e["step"].startswith("run") and e.get("done") and e.get("total"):
                            r = int(e["step"][3:])
                            if 1 <= r <= 4:
                                put(f"mamma/run{r}/q_used", e["done"] / P.MAMMA_RUN_TOTALS[r - 1])
        except OSError:
            continue
    keys = {}
    for k, v in S.items():
        keys[k] = {**_stats(v), "samples": [round(x, 4) for x in v[-10:]]}
    if gates:
        keys["mamma/gate_fail"] = {"n": len(gates), "p": round((sum(gates) + 0.5) / (len(gates) + 1), 3)}
    keys["room/train/profile"] = {"n": 1, "profile": ROOM_PROFILE, "N": 30000}
    doc = {"v": 1, "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "sessions": len(sess_dirs), "keys": keys}
    P.write_json_atomic(out, doc)
    return doc


# ---------------------------------------------------------------------------------------------------- camera types
# ONE source of truth: dancecap.camera_types (stdlib at import) decides what the camera stages APPLY - policy `user`
# by default (confirmed answers only; DANCECAP_CAMERA_TYPES / --camera-types auto|off), everything else handheld +
# zoom. The estimates price p50 from that effective class and p90 from the worst case the stages can still fall into
# (an unconfirmed camera is today's general solver; a keyframe-localised tripod whose constant pose is rejected is
# localised again on every frame: cameras/typed.localize_typed).

DEFAULT_TYPE = {"mount": "handheld", "zoom": True, "pans": None, "source": "default", "class": "handheld_zoom",
                "confirmed": False}


def camera_types(take: str, takes_dir: Path | None = None, policy: str | None = None) -> dict:
    """{cam: {mount, zoom, pans, source, class, confirmed, state, set_at, answer, suggested}}: the EFFECTIVE type the camera
    stages apply to each source video (camera_types.stage_type under the policy, default DANCECAP_CAMERA_TYPES or
    'user'), `confirmed` (the user answered: state confirmed | conflict), `set_at` (when) and `suggested` (the
    automatic suggestion: what confirming would change). Unconfirmed / missing -> handheld + zoom (today's solver).
    takes_dir: read <takes_dir>/<take>.toml instead of the package's takes/ (tests; no probe files, no alignment)."""
    from . import camera_types as CT
    try:
        pol = CT.resolve_policy(policy)
    except CT.CameraTypeError:
        pol = "user"
    try:
        if takes_dir is None:
            cts, _ = CT.load(take)
        else:
            import tomllib
            cfg = tomllib.loads((Path(takes_dir) / f"{take}.toml").read_text(encoding="utf-8-sig"))
            cts = [CT.describe(c, t) for c, t in sorted(CT.cam_tables(cfg).items())]
    except Exception:  # no TOML / unreadable: today's general model everywhere (what the stages do too)
        return {}
    out = {}
    for ct in cts:
        st = CT.stage_type(ct, pol)
        sug = ct.get("suggestion") or None
        if sug is None and ct["state"] == "suggested" and ct.get("mount") in ("handheld", "tripod") and \
                isinstance(ct.get("zoom"), bool):  # source = "auto" in the TOML without a `suggested` table
            sug = {"mount": ct["mount"], "zoom": ct["zoom"], "pans": ct.get("pans"), "confidence": None}
        out[ct["cam"]] = {"mount": st.mount, "zoom": bool(st.zoom), "pans": st.pans, "source": st.source,
                          "class": st.cls, "confirmed": ct["state"] in ("confirmed", "conflict"), "state": ct["state"],
                          "set_at": ct.get("set_at"),
                          "answer": {"mount": ct["mount"], "zoom": ct["zoom"]} if ct["state"] in ("confirmed",
                                                                                                    "conflict") else None,
                          "suggested": None if not sug else {k: sug.get(k) for k in ("mount", "zoom", "pans",
                                                                                     "confidence")}}
    return out


def keyframe_class(t: dict | None) -> bool:
    """Localised on keyframes only: a tripod without zoom whose head does not pan (cameras/typed.localize_typed, used
    by stage_loc; front_pod runs such a camera as ONE loc task - never split into loc_parts, which always localise
    every frame). Every other class - tripod + zoom, a panning head, handheld - is localised on every frame."""
    return bool(t) and t.get("mount") == "tripod" and t.get("zoom") is False and not t.get("pans")


def loc_units(frames: int, mount: str = "handheld", zoom: bool = True, fps: float = 30.0, key_every_s: float = 2.0,
              map_frac: float = 0.02, pans: bool | None = None) -> int:
    """Localisation work units (frames localised) of one camera under the class the stage applies (section 9.1):
    keyframe class = one frame every key_every_s + first and last, each with its next frame (typed.keyframe_subset
    pairs=True: the pair difference measures the per-frame noise of the drift check), >= 5, + the map's own frames
    (~map_frac); every other class = every frame."""
    if not keyframe_class({"mount": mount, "zoom": zoom, "pans": pans}):
        return int(frames)
    k = max(5, 2 * (int(math.ceil(frames / max(1.0, fps * key_every_s))) + 1))
    return min(int(frames), k + int(math.ceil(frames * map_frac)))


def loc_units_p90(frames: int, t: dict | None) -> int:
    """Worst case the stage can fall into: a keyframe-class camera whose keyframes do not fit one constant pose is
    localised again on every frame (keyframes + all); anything else localises every frame anyway."""
    t = t or DEFAULT_TYPE
    if keyframe_class(t):
        return loc_units(frames, t["mount"], t["zoom"], pans=t.get("pans")) + int(frames)
    return int(frames)


def _confirmed(t: dict | None) -> bool:
    return bool(t) and bool(t.get("confirmed", t.get("source") == "user"))


def loc_estimate(frames_by_cam: dict, types: dict | None = None, s_per_frame: float = 0.7, sigma: float = 0.25):
    """p10/p50/p90 seconds of localisation for all cameras (sum; lanes run in parallel on the pod, the DAG decides
    the wall time). types: EFFECTIVE types (camera_types()); p50 from the class the stage applies, p90 with the
    keyframe fallback (section 9.1). `unconfirmed`: cameras running today's general model for want of an answer."""
    types = types or {}
    u50 = u90 = 0
    per = {}
    for cam, n in frames_by_cam.items():
        t = types.get(cam) or DEFAULT_TYPE
        a = loc_units(n, t["mount"], t["zoom"], pans=t.get("pans"))
        b = loc_units_p90(n, t)
        per[cam] = {"units_p50": a, "units_p90": b, **{k: t.get(k) for k in ("mount", "zoom", "pans", "source")},
                    "confirmed": _confirmed(t)}
        u50 += a
        u90 += b
    e = est(u50 * s_per_frame, sigma)
    return {"p10": e[0], "p50": e[1], "p90": max(e[2], u90 * s_per_frame * math.exp(Z90 * sigma)),
            "units_p50": u50, "units_p90": u90, "cams": per,
            "unconfirmed": sorted(c for c, v in per.items() if not v["confirmed"])}


def camera_type_flags(cam: str, declared: dict | None, m: dict, at: float | None = None) -> list[dict]:
    """A camera_motion metric vs the type the user declared -> monitor warnings (the user's type is never overridden).

    m must be probe B's measurement (source = "probe_b": dancecap.camera_types.track_metrics on today's UNCONSTRAINED
    per-frame track, + its suggestion); the verdict uses camera_types' own thresholds (suggest_from_metrics,
    MIN_CONFIDENCE), so the monitor and the camera_types gate never disagree. Anything else (the old metric on the
    type-constrained export: 0 cm / x1.00 by construction) is ignored. declared: a camera_types() entry; a metric
    measured before the user's latest answer (at <= set_at) was acknowledged and raises nothing."""
    from . import camera_types as CT
    if not m or m.get("source") != "probe_b" or m.get("constrained"):
        return []
    d = declared or DEFAULT_TYPE
    conf = _confirmed(d)
    if conf and d.get("answer"):  # the user's own answer (the effective type is handheld + zoom under policy `off`)
        d = {**d, **d["answer"]}
    if conf and d.get("set_at") and at is not None and at <= float(d["set_at"]):
        return []
    sug = m if (m.get("mount") is not None or m.get("zoom") is not None) else \
        (CT.suggest_from_metrics(m, mount_judgeable=m.get("mount_judgeable", True),
                                 zoom_judgeable=m.get("zoom_judgeable", not m.get("attached", False))) or {})
    mc = sug.get("mount_confidence") if sug.get("mount_confidence") is not None else sug.get("confidence")
    zc = sug.get("zoom_confidence") if sug.get("zoom_confidence") is not None else sug.get("confidence")
    mok, zok = (mc or 0.0) >= CT.MIN_CONFIDENCE, (zc or 0.0) >= CT.MIN_CONFIDENCE
    c95, hf, fr = m.get("centre_p95_m"), m.get("hf_rot_deg"), m.get("focal_ratio")
    where = (f"centre {c95 * 100:.1f} cm (p95)" if c95 is not None else "") + \
        (f", rotation jitter {hf:.2f} deg" if hf is not None else "")
    why = {"mount": where, "zoom": f"focal x{fr:.3f} (p95/p5)" if fr is not None else ""}
    out = []
    if conf:  # a confident measurement against the user's answer: exactly camera_types' conflict (its gate opens)
        for field, ok, c in (("mount", mok, mc), ("zoom", zok, zc)):
            meas = sug.get(field)
            if meas is None or meas == d.get(field) or not ok:
                continue
            what = (f"marked {d.get(field)} but measures {meas}" if field == "mount" else
                    f"marked {'zoom' if d.get(field) else 'no zoom'} but measures {'zoom' if meas else 'no zoom'}")
            harm = {("mount", "tripod"): "the rigid pose will be rejected",
                    ("zoom", False): "a fixed focal would slide the camera along its axis"}.get((field, d.get(field)),
                                                                                             "a free model is kept")
            out.append({"level": "amber", "cam": cam, "field": field,
                        "msg": f"camera {cam} {what}: {why[field]} (probe B, confidence {c:.2f}); {harm} - check the "
                               "type"})
    else:  # no answer yet: what confirming the measurement would give
        if sug.get("mount") == "tripod" and mok:
            out.append({"level": "info", "cam": cam, "field": "mount",
                        "msg": f"camera {cam} looks static ({where}): confirm 'tripod' to localise keyframes only "
                               "(one rigid pose)"})
        if sug.get("zoom") is False and zok:
            out.append({"level": "info", "cam": cam, "field": "zoom",
                        "msg": f"camera {cam} shows no zoom ({why['zoom']}): confirm 'no zoom' for one focal per "
                               "video"})
    return out


# front job (pod): seconds per frame per task kind, measured on larissa_kadu_38 (A100, 28 vCPU host, 2026-10-07;
# work/larissa_kadu_38/front/front_pod.json); rates.json keys front/<kind>/s_per_frame replace them when present
FRONT_SEED = {"sam": 0.34, "loc": 0.78, "sub": 0.75, "ma2d": 0.16, "cameras": 0.113, "sync": 0.151}
FRONT_FIXED_S = {"merge": 3, "labels_review": 1, "prepare": 6, "chain": 15, "loc_join": 6}
FRONT_CAPS = {"sam": 5, "loc": 5, "ma2d": 2, "cpu": 2}
FRONT_LOC_SPLIT = 2  # front_pod.build_tasks(loc_split=2): the long pole's localisation runs in halves side by side


def _front_rate(rates, kind):
    v, n, _, _ = lookup(rates, f"front/{kind}/s_per_frame")
    return v if v else FRONT_SEED[kind]


def sam_sizes(take: str, work: Path = WORK) -> dict:
    """front_pod._sam_mem_gb per camera (host RAM of one dancers worker, GB) from frames.json: front_pod sorts the
    cameras by it and splits the LARGEST one's localisation (its long pole)."""
    out = {}
    for f in sorted((Path(work) / take / "frames").glob("*/frames.json")):
        d = P.read_json(f) or {}
        fr = d.get("frames")
        n = len(fr) if isinstance(fr, list) else int(fr or d.get("frame_count") or 0)
        try:
            w, h = int(d["width"]), int(d["height"])
        except (KeyError, TypeError, ValueError):
            return {}  # one camera without its size: rank by frames (front_dag's fallback), never mixed units
        px = w * h
        work_px = min(1.0, 1280 / max(w, h)) ** 2 * px
        out[f.parent.name] = round((n * (3 * px + 2 * px + 2 * work_px + 960 * 540 * 3)) / 2 ** 30 * 1.1 + 3.0, 1)
    return out


def front_dag(frames: dict, types: dict | None = None, main: list | None = None, attached: list | None = None,
              rates: dict | None = None, conservative: bool = False, sizes: dict | None = None,
              loc_split: int = FRONT_LOC_SPLIT) -> dict:
    """The front job's task DAG, mirroring front_pod.build_tasks: the long pole (largest SAM size, else most frames,
    among ALL cameras) is localised in `loc_split` parts named by their first frame (loc:02:0, loc:02:1151) + a join,
    only when it is a main-model camera and not a locked tripod (keyframe class: one loc task, one decision). Durations from frames x rates; localisation units
    from the EFFECTIVE camera types (p50) or their worst case (conservative: the keyframe fallback, section 9.1)."""
    types = types or {}
    attached = [c for c in (attached or []) if c in frames]
    main = [c for c in (main or sorted(frames)) if c in frames and c not in attached]
    cams = sorted(main + attached)
    key = (lambda c: (sizes or {}).get(c, 0.0)) if sizes else (lambda c: frames[c])
    long_pole = max(cams, key=lambda c: (key(c), -cams.index(c))) if cams else None
    T: dict[str, dict] = {}

    def units(c):
        t = types.get(c) or DEFAULT_TYPE
        if conservative:
            return loc_units_p90(frames[c], t)
        return loc_units(frames[c], t["mount"], t["zoom"], pans=t.get("pans"))
    for c in main + attached:
        T[f"sam:{c}"] = {"kind": "sam", "duration": frames[c] * _front_rate(rates, "sam")}
        T[f"ma2d:{c}"] = {"kind": "ma2d", "deps": [f"sam:{c}"], "duration": frames[c] * _front_rate(rates, "ma2d")}
    loc_names = []
    for c in main:
        u = units(c)
        if c == long_pole and loc_split > 1 and not keyframe_class(types.get(c)):
            n = frames[c]
            bounds = [round(k * n / loc_split) for k in range(loc_split + 1)]
            parts = [f"loc:{c}:{a}" for a in bounds[:-1]]
            for h, (a, b) in zip(parts, zip(bounds, bounds[1:])):
                T[h] = {"kind": "loc", "duration": u * (b - a) / max(1, n) * _front_rate(rates, "loc")}
            T[f"loc:{c}"] = {"kind": "cpu", "deps": parts, "duration": FRONT_FIXED_S["loc_join"]}
        else:
            T[f"loc:{c}"] = {"kind": "loc", "duration": u * _front_rate(rates, "loc")}
        loc_names.append(f"loc:{c}")
    for c in attached:
        T[f"sub:{c}"] = {"kind": "loc", "duration": units(c) * _front_rate(rates, "sub")}
        loc_names.append(f"sub:{c}")
    sams = [f"sam:{c}" for c in main + attached]
    T["merge"] = {"kind": "cpu", "after": sams, "duration": FRONT_FIXED_S["merge"]}
    total = sum(frames[c] for c in main + attached)
    T["cameras"] = {"kind": "cpu", "deps": ["merge"], "after": loc_names,
                    "duration": total * _front_rate(rates, "cameras")}
    T["sync"] = {"kind": "cpu", "deps": ["cameras"], "after": [f"ma2d:{c}" for c in main + attached],
                 "duration": total * _front_rate(rates, "sync")}
    T["prepare"] = {"kind": "cpu", "deps": ["sync"], "duration": FRONT_FIXED_S["prepare"]}
    T["chain"] = {"kind": "cpu", "deps": ["sync"], "duration": FRONT_FIXED_S["chain"]}
    for v in T.values():
        v.setdefault("state", "pending")
        v.setdefault("deps", [])
        v.setdefault("after", [])
    return T


def _as_confirmed(types: dict) -> dict:
    """The types the stages would apply if every automatic suggestion (mount and zoom both given) were confirmed."""
    out = {}
    for c, t in (types or {}).items():
        s = (t or {}).get("suggested") or {}
        if not _confirmed(t) and s.get("mount") in ("handheld", "tripod") and isinstance(s.get("zoom"), bool):
            out[c] = {**t, "mount": s["mount"], "zoom": s["zoom"], "pans": s.get("pans"), "source": "user",
                      "confirmed": True}
        else:
            out[c] = t
    return out


def estimate_front(frames: dict, types: dict | None = None, main: list | None = None, attached: list | None = None,
                   rates: dict | None = None, caps: dict | None = None, sigma: float = 0.2,
                   sizes: dict | None = None) -> dict:
    """Front job wall time (DAG simulation). types: EFFECTIVE types (camera_types(): what the stages apply under the
    policy - an unconfirmed suggestion changes nothing under the default `user` policy). p50 from them; p90 from their
    worst case (keyframe fallback). Also the all-handheld baseline (saving_s) and, for the plan dialog, what
    confirming the automatic suggestions would give (if_confirmed_p50, potential_saving_s)."""
    caps = caps or FRONT_CAPS
    d50 = simulate_dag(front_dag(frames, types, main, attached, rates, sizes=sizes), caps)
    d90 = simulate_dag(front_dag(frames, types, main, attached, rates, conservative=True, sizes=sizes), caps)
    base = simulate_dag(front_dag(frames, {}, main, attached, rates, sizes=sizes), caps)
    dif = simulate_dag(front_dag(frames, _as_confirmed(types or {}), main, attached, rates, sizes=sizes), caps)
    le = loc_estimate(frames, types)
    return {"p10": round(d50 * math.exp(-Z90 * sigma), 1), "p50": round(d50, 1),
            "p90": round(max(d90, d50) * math.exp(Z90 * sigma), 1), "all_handheld_p50": round(base, 1),
            "saving_s": round(base - d50, 1), "if_confirmed_p50": round(dif, 1),
            "potential_saving_s": round(max(0.0, d50 - dif), 1), "loc_units_p50": le["units_p50"],
            "loc_units_p90": le["units_p90"], "unconfirmed": le["unconfirmed"]}


def take_frames(take: str, work: Path = WORK) -> dict:
    """Window frames per camera from work/<take>/frames/<cam>/frames.json (after ingest)."""
    out = {}
    for f in sorted((Path(work) / take / "frames").glob("*/frames.json")):
        d = P.read_json(f) or {}
        n = d.get("frames")
        n = len(n) if isinstance(n, list) else (n or d.get("frame_count"))
        if n:
            out[f.parent.name] = int(n)
    return out


# ---------------------------------------------------------------------------------------------------- session model


def parse_at(s: str, sid: str | None = None) -> float:
    """'HH:MM[:SS]' on the session's (local) date, or an epoch."""
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    d = P.sid_date(sid or "") or time.localtime()[:3]
    parts = [int(x) for x in s.split(":")] + [0, 0]
    return time.mktime((d[0], d[1], d[2], parts[0], parts[1], parts[2], 0, 0, -1))


def _fill_times(evs: list[dict], end_anchor: float | None, tex_pre_pv: float | None = None,
                last_line: int | None = None) -> None:
    """ts for events without one: rx when known, else interpolated by line number between anchored events (the
    file mtime anchors the file's LAST line - last_line, the parser's line count - which is the latest write; without
    it, the line after the last event). Texture passes are then spread over their measured seconds, and an unfinished
    pass is re-timed from the command start (_fill_texture_running)."""
    for e in evs:
        if e.get("ts") is None and e.get("rx") is not None:
            e["ts"], e["ts_est"] = e["rx"], "rx"
    idx = [i for i, e in enumerate(evs) if e.get("ts") is not None]
    if not evs:
        return
    if end_anchor is not None and (not idx or idx[-1] != len(evs) - 1):
        last = evs[-1]
        end_line = last.get("line", len(evs)) + 1
        if last_line is not None and last_line >= last.get("line", len(evs)):
            end_line = last_line
        anchors = [(evs[i].get("line", i), evs[i]["ts"]) for i in idx] + [(end_line, end_anchor)]
    else:
        anchors = [(evs[i].get("line", i), evs[i]["ts"]) for i in idx]
    if not anchors:
        return
    j = 0
    for i, e in enumerate(evs):
        if e.get("ts") is not None:
            continue
        ln = e.get("line", i)
        while j + 1 < len(anchors) and anchors[j + 1][0] < ln:
            j += 1
        a = anchors[j]
        b = anchors[j + 1] if j + 1 < len(anchors) else None
        if a[0] > ln:
            e["ts"] = a[1]
        elif b is None or b[0] == a[0]:
            e["ts"] = a[1]
        else:
            e["ts"] = a[1] + (b[1] - a[1]) * (ln - a[0]) / (b[0] - a[0])
        e["ts_est"] = "interp"
        e["ts"] = min(e["ts"], end_anchor) if end_anchor else e["ts"]
    # texture: ticks of a finished pass spread over its measured seconds, ending at the pass end
    by_pass: dict[tuple, list] = {}
    for e in evs:
        if e.get("src") == "texture" and e.get("step", "").startswith("bake/pass"):
            by_pass.setdefault((e.get("job"), e.get("cmd"), e["step"]), []).append(e)
    for es in by_pass.values():
        endev = next((e for e in es if e["kind"] == "end" and (e.get("extra") or {}).get("seconds")), None)
        if not endev or endev.get("ts_est") not in ("interp",) and endev.get("rx"):
            continue
        S = endev["extra"]["seconds"]
        t1 = endev["ts"]
        n = endev.get("total") or 1
        for e in es:
            if e["kind"] in ("tick", "start") and e.get("ts_est") == "interp":
                e["ts"] = t1 - S + S * (e.get("done") or 0) / n
    _fill_texture_running(evs, tex_pre_pv)


def _fill_texture_running(evs: list[dict], pre_pv: float | None) -> None:
    """Cold start on an unfinished bake pass whose view lines carry no clock (an L0 log read after the fact): line
    interpolation would spread the pass's ticks back over the load phase (~5 min with few lines), reading its first
    minute ~5-9x too slow. The pass start is rebuilt from what is exact - the command start ('[pod-runner] <ISO>Z
    start'), the finished passes' measured seconds - plus the load phase from history (pre_pv x views): the ticks are
    spread from that start to the last tick's time. ts_est = "interp_pass" (estimated: the ETA trusts it less)."""
    pre_pv = pre_pv if pre_pv else SEED["texture/pre/s_per_view"]["p50"]
    starts = {(e.get("job"), e.get("step")): e.get("ts") for e in evs
              if e.get("src") == "pod-runner" and e.get("kind") == "start" and e.get("ts") is not None}
    groups: dict[tuple, dict] = {}
    for e in evs:
        if e.get("src") != "texture":
            continue
        g = groups.setdefault((e.get("job"), e.get("cmd")), {"views": None, "end": {}, "ticks": {}, "fin": False})
        st = e.get("step", "")
        if e["kind"] == "plan" and st == "bake":
            g["views"] = (e.get("extra") or {}).get("views")
        elif st.startswith("bake/pass"):
            k = int(st[9:])
            if e["kind"] == "end":
                g["end"][k] = (e.get("extra") or {}).get("seconds")
            elif e["kind"] == "tick":
                g["ticks"].setdefault(k, []).append(e)
        elif st == "bake/finish" and e["kind"] == "end":
            g["fin"] = True
    for (job, cmd), g in groups.items():
        t_cmd = starts.get((job, cmd))
        open_k = [k for k in g["ticks"] if k not in g["end"]]
        if t_cmd is None or g["fin"] or not open_k:
            continue
        k = max(open_k)
        ticks = g["ticks"][k]
        if any(e.get("ts_est") != "interp" for e in ticks):
            continue  # receive-time or pod-clock stamps: exact already
        views = g["views"] or ticks[-1].get("total")
        if not views or any(g["end"].get(j) is None for j in range(k)):
            continue
        t0 = t_cmd + pre_pv * views + sum(g["end"][j] for j in range(k))
        t1 = ticks[-1]["ts"]
        d1 = ticks[-1].get("done") or 0
        if not d1 or t1 is None or t1 - t0 < 5:
            continue  # the load phase ran longer than history says: keep the line interpolation
        for e in ticks:
            e["ts"] = t0 + (t1 - t0) * (e.get("done") or 0) / d1
            e["ts_est"] = "interp_pass"
        # the finished passes on the same time line (their ends and ticks were line-interpolated too)
        te = t_cmd + pre_pv * views
        for j in range(k):
            sj = g["end"][j]
            te += sj
            for e in evs:
                if e.get("src") == "texture" and e.get("job") == job and e.get("cmd") == cmd and \
                        e.get("step") == f"bake/pass{j}" and e.get("ts_est") == "interp":
                    e["ts"] = te if e["kind"] == "end" else te - sj + sj * (e.get("done") or 0) / max(1, views)
                    e["ts_est"] = "interp_pass"


def _pre_pv(rates: dict | None) -> float | None:
    """History of a bake's load phase (s per view) for the cold-start timing of an unfinished pass."""
    try:
        v, n, _, _ = lookup(rates or {}, "texture/pre/s_per_view")
        return v
    except Exception:
        return None


def session_events(session_dir: Path, at: float | None = None, parsed=None, rates: dict | None = None) -> list[dict]:
    """Every event of a session: L0 parse of runner.log (+ L1 runner.jsonl, pod.jsonl when present), ts filled."""
    d = Path(session_dir)
    n_lines = None
    if parsed is None:
        evs, _, parser = P.parse_runner_log(d / "runner.log")
        n_lines = parser.n
    else:
        evs = parsed
    try:
        mt = (d / "runner.log").stat().st_mtime
    except OSError:
        mt = None
    _fill_times(evs, mt, _pre_pv(rates), n_lines)
    for f in ("runner.jsonl", "pod.jsonl"):
        more, _ = P.read_jsonl(d / "progress" / f)
        for e in more:
            e.setdefault("src", "l1" if f == "runner.jsonl" else "pod")
        evs = evs + more
    if at is not None:
        evs = [e for e in evs if e.get("ts") is not None and e["ts"] <= at]
    return evs


class Phase:
    __slots__ = ("name", "job", "kind", "status", "start", "end", "seconds", "remaining", "fraction", "detail", "cmd",
                 "optional", "p90_only", "unstable", "stalled")

    def __init__(self, name, job, kind, cmd=None, optional=False):
        self.name, self.job, self.kind, self.cmd, self.optional = name, job, kind, cmd, optional
        self.status, self.start, self.end, self.seconds = "pending", None, None, None
        self.remaining, self.fraction, self.detail, self.p90_only = None, None, None, False
        self.unstable, self.stalled = False, None  # live rate dominated by one interval / {since_s, gap_s} when stuck

    def as_dict(self, now):
        el = None
        if self.start is not None:
            el = (self.end if self.end is not None else now) - self.start
        rem = self.remaining
        return {"name": self.name, "job": self.job, "kind": self.kind, "status": self.status,
                "start": self.start, "end": self.end, "elapsed_s": round(el, 1) if el is not None else None,
                "fraction": None if self.fraction is None else round(self.fraction, 4),
                "remaining": None if rem is None else {"p10": round(rem[0], 1), "p50": round(rem[1], 1),
                                                       "p90": round(rem[2], 1)},
                "detail": self.detail, "optional": self.optional, "unsettled": self.unstable or None,
                "stalled": self.stalled}


def _job_phases(job: str, spec: dict) -> list[Phase]:
    out = []
    for i, _ in enumerate(spec.get("archives") or []):
        out.append(Phase(f"{job}/archive", job, "archive"))
    for i, _ in enumerate(spec.get("setup") or []):
        out.append(Phase(f"{job}/setup{i + 1}", job, "setup"))
    for c in spec.get("commands") or []:
        out.append(Phase(f"{job}/{c['name']}", job, "command", cmd=c, optional=bool(c.get("optional"))))
    out += [Phase(f"{job}/pack", job, "pack"), Phase(f"{job}/download", job, "download"),
            Phase(f"{job}/extract", job, "job_extract")]
    return out


def _attempt_events(evs: list[dict]) -> list[dict]:
    """Events of the last runner attempt (a resumed sid holds several; each starts with bundle_scan)."""
    starts = [i for i, e in enumerate(evs) if e.get("src") == "runner" and e.get("kind") == "start"
              and e.get("step") == "bundle_scan"]
    return evs[starts[-1]:] if starts else evs


class SessionModel:
    """Builds the phase list of a session's current attempt with status and remaining-time estimates."""

    def __init__(self, session_dir: Path, evs: list[dict], state: dict, now: float, rates: dict, ledger_rows: list,
                 at: float | None = None, puller: dict | None = None):
        self.dir, self.state, self.now, self.rates, self.at = Path(session_dir), state or {}, now, rates, at
        self.sid = self.dir.name
        self.all_evs = evs
        self.evs = _attempt_events(evs)
        self.rows = ledger_rows
        self.puller = puller or {}
        self.pod = dict(self.state.get("pod") or {})
        self.gpu = self.pod.get("gpu")
        self.warnings: list[dict] = []
        self.branches = None

    # ---- helpers
    def _hist(self, key, default, sigma=0.3):
        """History rate (value, n, sigma, source). Thin history (< 10 samples: few runs, usually one take on one host)
        keeps a sigma floor of HIST_SIGMA_FLOOR: its MAD says nothing about the next take / host / code version."""
        v, n, sig, src = lookup(self.rates, key, self.gpu)
        if v is None:
            return default, 0, sigma, "default"
        return v, n, max(sig or 0.0, HIST_SIGMA_FLOOR if n < 10 else 0.08), src

    def _flag(self, ph, diag: dict | None) -> None:
        """Copy a remaining_linear diagnosis onto the phase (unsettled live rate / stalled step)."""
        if not diag:
            return
        if diag.get("unstable"):
            ph.unstable = True
        if diag.get("stalled"):
            ph.stalled = {"since_s": diag.get("since_advance_s"), "gap_s": diag.get("med_gap_s"),
                          "step": diag.get("step")}

    def _dur(self, key, default_p50, sigma=0.3):
        """(p10, p50, p90) duration of a step from history samples (seconds)."""
        keys = (self.rates or {}).get("keys", {})
        r = keys.get(key)
        if r and r.get("n"):
            p10, p50, p90 = r["p10"], r["p50"], r["p90"]
            if r["n"] < 3:  # thin history: widen
                p10, p90 = min(p10, p50 * math.exp(-Z90 * 0.3)), max(p90, p50 * math.exp(Z90 * 0.3))
            return (p10, p50, p90)
        s = SEED.get(key)
        if s and "p10" in s:
            return (s["p10"], s["p50"], s["p90"])
        return est(default_p50, sigma)

    # ---- build
    def build(self) -> list[Phase]:
        st = self.state
        phases = [Phase("bundle_scan", "_session", "bundle_scan"), Phase("bundle_build", "_session", "bundle_build"),
                  Phase("pod_start", "_session", "pod_start"), Phase("upload", "_session", "upload"),
                  Phase("extract", "_session", "extract"), Phase("bootstrap", "_session", "bootstrap"),
                  Phase("wheels_back", "_session", "wheels_back")]
        jobs = st.get("jobs") or {}
        self.todo = []
        attempt_t0 = self.evs[0].get("ts") if self.evs else None
        for j, v in jobs.items():
            # jobs already ok from an earlier attempt (same fingerprint) are skipped by the runner
            if v.get("status") == "ok" and v.get("started") and attempt_t0 and v["started"] < attempt_t0 - 1:
                continue
            self.todo.append(j)
            phases += _job_phases(j, v.get("spec") or {})
        phases.append(Phase("delete", "_session", "delete"))
        self.phases = phases
        self._apply_runner_events()
        self._apply_pod_start()
        self._estimate()
        return phases

    def _find(self, name):
        for ph in self.phases:
            if ph.name == name and ph.status in ("pending", "running"):
                return ph
        for ph in self.phases:
            if ph.name == name:
                return ph
        return None

    def _apply_runner_events(self):
        by_name = {}
        for e in self.evs:
            if e.get("src") != "runner":  # L1 runner.jsonl duplicates these; runner.log stays the source of truth
                continue
            name = (e.get("extra") or {}).get("runner_step") or (e["step"] if e.get("job") in ("_session", None)
                                                                  else f"{e['job']}/{e['step']}")
            if name == "bootstrap_retry":
                name = "bootstrap"
            if e["kind"] == "start":
                ph = self._next_named(name, by_name)
                if ph is not None:
                    if ph.status == "done" and name == "bootstrap":
                        pass
                    ph.status, ph.start, ph.end = "running", e.get("ts"), None
            elif e["kind"] in ("end", "fail"):
                ph = by_name.get(name) or self._find(name)
                if ph is not None:
                    ph.status = "done" if e["kind"] == "end" else "failed"
                    ph.end = e.get("ts")
                    secs = (e.get("extra") or {}).get("seconds")
                    ph.seconds = secs
                    if ph.start is None and secs is not None and ph.end is not None:
                        ph.start = ph.end - secs
            elif e["kind"] == "fail" and e.get("step") == "session":
                self.warnings.append({"level": "red", "msg": f"session error: {e.get('msg')}"})
        # job status lines: skipped/failed jobs -> their pending phases will not run
        for e in self.evs:
            if e.get("src") == "runner" and e.get("step") == "_job":
                status = (e.get("extra") or {}).get("status") or e.get("msg") or ""
                for ph in self.phases:
                    if ph.job == e["job"] and ph.status == "pending":
                        ph.status = "skipped"
                if status.startswith(("failed", "skipped")):
                    self.warnings.append({"level": "amber", "msg": f"{e['job']}: {status}"[:200]})
            if e.get("src") == "runner" and e.get("kind") == "fail" and e.get("step") == "session":
                for ph in self.phases:
                    if ph.status in ("pending", "running") and ph.kind != "delete":
                        ph.status = "skipped" if ph.status == "pending" else "failed"
        # a failed non-optional command skips the job's later commands (pack/download still run)
        for j in self.todo:
            failed = False
            for ph in self.phases:
                if ph.job != j:
                    continue
                if ph.kind == "command" and failed and ph.status == "pending":
                    ph.status = "skipped"
                if ph.kind == "command" and ph.status == "done":
                    rc = self._pod_rc(j, ph.cmd["name"] if ph.cmd else None)
                    if rc not in (None, 0) and not ph.optional:
                        failed = True
                        ph.status = "failed"
                    elif rc not in (None, 0):
                        ph.detail = f"optional, rc={rc}"

    def _pod_rc(self, job, cmd):
        rc = None
        for e in self.evs:
            if e.get("src") == "pod-runner" and e.get("job") == job and e.get("step") == cmd and e["kind"] in ("end",
                                                                                                             "fail"):
                rc = e.get("rc", 0)
        return rc

    def _next_named(self, name, by_name):
        for ph in self.phases:
            if ph.name == name and ph.status == "pending":
                by_name[name] = ph
                return ph
        ph = self._find(name)
        if ph is not None:
            by_name[name] = ph
        return ph

    def _apply_pod_start(self):
        ph = self._find("pod_start")
        bb = self._find("bundle_build")
        up = self._find("upload")
        row = self._ledger_row()
        if row:
            ph.start = row["created_ts"]
        elif bb and bb.end and self.pod.get("started") is None and up.status == "pending":
            ph.start = bb.end
        started = self.pod.get("started")
        if started and (self.at is None or started <= self.at):
            ph.status, ph.end = "done", started
            if ph.start is None and up.start:
                ph.start = started - 34
        elif ph.start is not None and bb and bb.status == "done":
            ph.status = "running"
        if up.status in ("running", "done") and ph.status != "done":
            ph.status, ph.end = "done", up.start

    def _ledger_row(self):
        pid = self.pod.get("id")
        if not pid:
            return None
        row = next((r for r in self.rows if r.get("pod_id") == pid), None)
        if row and self.at is not None and row.get("created_ts", 0) > self.at:
            return None
        return row

    # ---- estimates
    def _estimate(self):
        now = self.now
        bundle_bytes = (self.state.get("bundle") or {}).get("bytes")
        feats = None
        for e in self.evs:
            if e.get("step") == "bundle" and e.get("kind") == "note":
                feats = (e.get("extra") or {}).get("features")
                bundle_bytes = bundle_bytes or ((e.get("extra") or {}).get("mb") or 0) * 1e6
        self.features = feats or ["base", "check"]
        for ph in self.phases:
            if ph.status in ("done", "failed", "skipped"):
                ph.remaining = (0.0, 0.0, 0.0)
                if ph.status == "done":
                    ph.fraction = 1.0
                continue
            elapsed = (now - ph.start) if (ph.status == "running" and ph.start is not None) else 0.0
            k = ph.kind
            if k == "upload":
                ph.remaining = self._upload(bundle_bytes, elapsed, ph)
            elif k == "bootstrap":
                ph.remaining = self._bootstrap(ph, elapsed)
            elif k == "command":
                ph.remaining = self._command(ph, elapsed)
            elif k == "download":
                ph.remaining = self._download(ph, elapsed)
            elif k == "pod_start":
                ph.remaining = remaining_after_elapsed(self._dur("session/pod_start/s", 34), elapsed)
            else:
                key = {"job_extract": "session/job_extract/s", "archive": "session/archive/s",
                       "setup": "session/setup/s"}.get(k, f"session/{k}/s")
                dur = self._dur(key, 10)
                ph.remaining = remaining_after_elapsed(dur, elapsed) if ph.status == "running" else dur

    def _upload(self, nbytes, elapsed, ph):
        mbps = self._mbps("session/upload/MBps", (0.7, 2.5, 2.8))
        part = (self.puller or {}).get("upload_part_bytes")
        if nbytes and part and ph.status == "running" and elapsed > 30:
            r_live = part / 1e6 / max(1.0, elapsed)
            left = (nbytes - part) / 1e6 / max(0.05, r_live)
            ph.fraction = part / nbytes
            ph.detail = f"{part / 1e6:.0f}/{nbytes / 1e6:.0f} MB at {r_live:.2f} MB/s (live)"
            return (left * 0.85, left, left * 1.3)
        if not nbytes:
            return est(600, 0.5)
        mb = nbytes / 1e6
        dur = (mb / mbps[2], mb / mbps[1], mb / mbps[0])
        if ph.status == "running":
            ph.fraction = min(0.99, elapsed / dur[1]) if dur[1] else None
            ph.detail = f"{mb:.0f} MB; ~{mb * min(1.0, elapsed / dur[1]):.0f} MB est. at {mbps[1]:.1f} MB/s (no byte " \
                        f"progress at L0)"
            return remaining_after_elapsed(dur, elapsed)
        return dur

    def _mbps(self, key, default):
        r = (self.rates or {}).get("keys", {}).get(key)
        if r and r.get("n", 0) >= 3:
            return (r["p10"], r["p50"], r["p90"])
        return default

    def _bootstrap(self, ph, elapsed):
        done, cur, cur_t0 = set(), None, None
        for e in self.evs:
            if e.get("src") == "bootstrap" and e.get("step", "").startswith("bootstrap/"):
                f = e["step"].split("/", 1)[1]
                if e["kind"] == "start":
                    cur, cur_t0 = f, e.get("ts")
                elif e["kind"] in ("end", "fail"):
                    done.add(f)
                    if cur == f:
                        cur = None
        parts = []
        for f in self.features:
            if f in done:
                continue
            d = self._dur(f"session/bootstrap/{f}/s", BOOT_SEED_MIN.get(f, (1, 2, 5))[1] * 60)
            if f in BOOT_SEED_MIN and not (self.rates or {}).get("keys", {}).get(f"session/bootstrap/{f}/s"):
                d = tuple(x * 60 for x in BOOT_SEED_MIN[f])
            if f == cur and cur_t0 is not None:
                d = remaining_after_elapsed(d, self.now - cur_t0)
            parts.append(d)
        nf = max(1, len(self.features))
        ph.fraction = len(done & set(self.features)) / nf if ph.status == "running" else None
        ph.detail = (f"feature {cur} running" if cur else None) if ph.status == "running" else None
        r = add(*parts)
        return r

    def _download(self, ph, elapsed):
        mb = None
        for t in reversed(self.state.get("timings", [])):
            if t.get("step") == ph.name and t.get("mb"):
                mb = t["mb"]
                break
        jobst = (self.state.get("jobs") or {}).get(ph.job, {})
        if jobst.get("pack_bytes"):
            mb = jobst["pack_bytes"] / 1e6
        mb = mb or 60.0
        r = self._mbps("session/download/MBps", (1.5, 12.0, 16.0))
        part = self.dir / "downloads" / f"{ph.job}.tar.gz.part"
        if ph.status == "running" and jobst.get("pack_bytes") and self.at is None:
            try:  # live byte progress from the local .part file (L1: pack_bytes recorded before the download)
                got = part.stat().st_size
                ph.fraction = min(0.999, got / jobst["pack_bytes"])
                rate = got / 1e6 / max(1.0, elapsed)
                left = (jobst["pack_bytes"] - got) / 1e6 / max(0.05, rate)
                ph.detail = f"{got / 1e6:.0f}/{jobst['pack_bytes'] / 1e6:.0f} MB at {rate:.1f} MB/s"
                return (left * 0.8, left, left * 1.5 + 5)
            except OSError:
                pass
        dur = (mb / r[2] + 1, mb / r[1] + 2, mb / r[0] + 5)
        return remaining_after_elapsed(dur, elapsed) if ph.status == "running" else dur

    # ---- commands
    def _cmd_events(self, ph):
        """Inner progress of one pod command. L2/L1 events (src pod / tail: exact pod clock) win over the L0 parse of
        the same steps in runner.log; L0 fills whatever the pod does not report."""
        job, name = ph.job, ph.cmd["name"] if ph.cmd else None
        evs = [e for e in self.evs if e.get("job") == job and e.get("cmd", name if e.get("src") == "tail" else None)
               == name and e.get("src") not in ("runner", "pod-runner")]
        pod_steps = {(e.get("step"), e.get("lane")) for e in evs if e.get("src") in ("pod", "tail")}
        if not pod_steps:
            return evs
        out = [e for e in evs if e.get("src") in ("pod", "tail") or (e.get("step"), e.get("lane")) not in pod_steps]
        return sorted(out, key=lambda e: e.get("ts") or 0)

    def _command(self, ph, elapsed):
        c = ph.cmd or {}
        key = _cmd_key(ph.job, c)
        to = float(c.get("timeout_min") or 60) * 60
        evs = self._cmd_events(ph)
        if "texture" == key or any(e.get("src") in ("texture", "pod") and str(e.get("step", "")).startswith("bake")
                                   for e in evs):
            return self._texture(ph, evs, elapsed)
        if key == "mamma_chunks.fit" or any(e.get("src") in ("chunks", "pod") and e.get("step") == "fit_chunks"
                                            for e in evs):
            return self._chunks(ph, evs, elapsed)
        if key == "mamma_chunks.gate":
            return self._gate(ph, evs, elapsed)
        if key == "front_pod.supervise" or key == "front_pod.run":
            r = self._front(ph, evs, elapsed)
            if r is not None:
                return r
        if key.startswith("dancer_splats") or any(e.get("src") == "splats" or str(e.get("step", "")).startswith(
                "splats/") for e in evs):
            return self._splats(ph, evs, elapsed)
        if any(e.get("src") == "room" or (e.get("src") == "pod" and e.get("step") == "train") for e in evs):
            return self._room(ph, evs, elapsed)
        if any(e.get("src") == "hands" and e.get("kind") == "tick" for e in evs) and ph.status == "running":
            last = [e for e in evs if e.get("src") == "hands" and e.get("kind") == "tick"][-1]
            steps = [e for e in evs if e.get("step") == last["step"] and e.get("kind") == "tick"]
            obs = [(e.get("ts"), e.get("done")) for e in steps]
            ph.fraction = last["done"] / max(1, last["total"])
            ph.detail = f"{last['step']} {last['done']}/{last['total']} frames"
            x = last["extra"].get("elapsed_s") or 0
            s_h = x / last["done"] if last["done"] else None
            diag = {"step": last["step"]}
            r = remaining_linear(last["total"], last["done"], obs, s_h, 1, 0.3, now=self.now, diag=diag)
            self._flag(ph, diag)
            return r
        # generic: history of this command kind, else the timeout
        v, n, sig, src = lookup(self.rates, f"cmd/{key}/s", self.gpu)
        if v is None:  # timeouts are generous: measured run/timeout 0.03-0.57 (median ~0.3) on 2026-10-06/07
            dur = (0.05 * to, 0.3 * to, 0.9 * to)
            ph.detail = (ph.detail or "") + " (no history: timeout-based)"
        else:
            dur = est(v, max(sig, 0.25))
            ph.detail = f"history {src} n={n}" if not ph.detail else ph.detail
        return remaining_after_elapsed(dur, elapsed) if ph.status == "running" else dur

    def _texture(self, ph, evs, elapsed):
        views, passes = None, None
        for e in evs:
            if e.get("src") in ("texture", "pod") and e["kind"] == "plan" and e.get("step") == "bake":
                views, passes = (e.get("extra") or {}).get("views"), (e.get("extra") or {}).get("passes")
        c = (ph.cmd or {}).get("cmd", "")
        if passes is None:
            pm = re.search(r"--preset (\w+)", c)
            passes = P.TEXTURE_PASSES.get(pm.group(1) if pm else "pod", 3)
        views = views or self._hist_views() or 2292
        s0, n0, g0, _ = self._hist("texture/pass0/s_per_view", 0.063, 0.15)
        sa, na, ga, _ = self._hist("texture/aligned/s_per_view", 0.134, 0.15)
        fixed_pv, nf, gf, _ = self._hist("texture/fixed/s_per_view", 0.137, 0.35)
        pre_pv, n_pre, _, src_pre = self._hist("texture/pre/s_per_view", None)
        post_pv, n_post, _, src_post = self._hist("texture/post/s_per_view", None)
        if n_pre and n_post:  # measured split (L2 bake/load + bake/finish, or a live-recorded run)
            pre_s, post_s = pre_pv * views, post_pv * views
        else:  # L0 history only knows pre + post together: split by the seed ratio
            f_pre = SEED["texture/pre/s_per_view"]["p50"] / (SEED["texture/pre/s_per_view"]["p50"]
                                                            + SEED["texture/post/s_per_view"]["p50"])
            pre_s, post_s = f_pre * fixed_pv * views, (1 - f_pre) * fixed_pv * views
        # current pass and live rate
        cur_k, cur_done, obs, done_passes, pass_s = None, 0, [], set(), {}
        pass_obs: dict[int, list] = {}
        pass_end: dict[int, float | None] = {}
        cur_est = False  # the current pass's tick times are estimated (cold L0 read), not received / pod-clocked
        first_tick_ts = None
        for e in evs:
            if e.get("src") not in ("texture", "pod") or not e.get("step", "").startswith("bake/pass"):
                continue
            k = int(e["step"][9:])
            if e["kind"] == "end":
                done_passes.add(k)
                pass_s[k] = (e.get("extra") or {}).get("seconds")
                pass_end[k] = e.get("ts")
            elif e["kind"] == "tick":
                if first_tick_ts is None:
                    first_tick_ts = e.get("ts")
                if cur_k != k:
                    cur_k, obs = k, []
                cur_done = e["done"]
                obs.append((e.get("ts"), e["done"]))
                pass_obs.setdefault(k, []).append((e.get("ts"), e["done"]))
                cur_est = (cur_est if obs[:-1] else True) and str(e.get("ts_est") or "").startswith("interp")
        # a finished pass's working speed: robust over its own ticks (a pause inside it is not the bake's speed),
        # else its measured seconds
        for k in list(pass_s):
            ri = live_rate_info(pass_obs.get(k) or [], window_s=1e12)
            if ri["s"] and ri["n"] >= 5 and ri["dropped"] and pass_s[k]:
                pass_s[k] = min(pass_s[k], ri["s"] * views)
        finished = any(e.get("src") in ("texture", "pod") and e.get("step") == "bake/finish" and e["kind"] == "end"
                       for e in evs)
        speed = 1.0
        parts = []
        if ph.status != "running" or (cur_k is None and not done_passes):
            # not started, or still loading (pre-phase)
            pre = est(pre_s, gf)
            if ph.status == "running":
                pre = remaining_after_elapsed(pre, elapsed)
                ph.detail = f"loading / head fix / hair model ({elapsed:.0f} s)"
                ph.fraction = 0.0
            parts.append(pre)
            for k in range(passes):
                parts.append(est(views * (s0 if k == 0 else sa), g0 if k == 0 else ga))
            parts.append(est(post_s, gf))
            return add(*parts)
        if finished:
            ph.detail = "finishing (writing outputs)"
            ph.fraction = 0.99
            return est(5, 0.5)
        # this run's speed vs history from its FINISHED aligned passes (stable; the live rate early in a pass is noisy)
        done_ratio = [pass_s[k] / (views * sa) for k in done_passes if k > 0 and pass_s.get(k)]
        prior = 1.0  # what is known about this run's speed before the current pass
        if done_ratio:
            speed = prior = _clamp(math.exp(sum(math.log(x) for x in done_ratio) / len(done_ratio)), 0.25, 4.0)
        elif 0 in done_passes and pass_s.get(0):  # pass 0 is another kernel: its speed carries at half weight
            speed = prior = math.exp(0.5 * math.log(_clamp(pass_s[0] / (views * s0), 0.5, 2.0)))
        if cur_k is not None and cur_k not in done_passes:
            s_h = (s0 if cur_k == 0 else sa * prior)
            diag = {"step": f"bake/pass{cur_k}"}
            r = remaining_linear(views, cur_done, obs, s_h, (n0 if cur_k == 0 else na) + len(done_ratio),
                                 g0 if cur_k == 0 else (ga if not done_ratio else 0.08), now=self.now, diag=diag,
                                 x0=0.25 if cur_est else None)
            self._flag(ph, diag)
            s_live = diag.get("s_live")
            # every pass walks the views in the same order (per-view cost varies ~4x along it): the previous pass's
            # time profile predicts this pass's far better than a 2-minute window (section 6.3, as gsplat's profile)
            pr = None
            if cur_k >= 1 and r[1] > 0:
                pr = _pass_profile_eta(pass_obs.get(cur_k - 1), pass_end.get(cur_k - 1), pass_s.get(cur_k - 1),
                                       obs, views, diag, (views * sa * prior) / pass_s[cur_k - 1]
                                       if cur_k == 1 and pass_s.get(0) else 1.0)
            if pr is not None:
                R_prof, proj = pr
                r = (r[0] / r[1] * R_prof, R_prof, r[2] / r[1] * R_prof)
                if not done_ratio:  # this pass's projected length tells the later passes this run's speed
                    x = cur_done / views
                    w = x / (x + 0.1)
                    speed = math.exp(w * math.log(_clamp(proj / (views * sa), 0.5, 2.0)) + (1 - w) * math.log(prior))
            elif s_live and not done_ratio:
                # an unfinished pass's live speed reaches the later passes only bounded (0.5x-2x) and weighted by how
                # far the pass is: one resumed stall or a slow start must not multiply the rest of the bake
                x = cur_done / views
                w = x / (x + 0.1)
                if cur_k > 0:
                    speed = math.exp(w * math.log(_clamp(s_live / sa, 0.5, 2.0)) + (1 - w) * math.log(prior))
                else:
                    speed = math.exp(0.5 * w * math.log(_clamp(s_live / s0, 0.5, 2.0)))
            parts.append(r)
            nxt = cur_k + 1
            ph.detail = f"pass {cur_k + 1}/{passes}: {cur_done}/{views} views" + (
                f" at {s_live:.3f} s/view (history {s_h:.3f})" if s_live else "")
        else:
            nxt = (max(done_passes) + 1) if done_passes else 0
            ph.detail = f"between passes ({len(done_passes)}/{passes} done)"
        for k in range(nxt, passes):
            parts.append(scale(est(views * (s0 if k == 0 else sa), g0 if k == 0 else (0.08 if done_ratio else ga)),
                               speed if k > 0 else 1.0))
        parts.append(est(post_s, gf))
        total_units = passes * views
        done_units = len(done_passes) * views + (cur_done if cur_k is not None and cur_k not in done_passes else 0)
        ph.fraction = min(0.99, done_units / total_units)
        return add(*parts)

    def _front(self, ph, evs, elapsed):
        """Front job: the task DAG from the take's frames and camera types, live task states from '[front]' lines."""
        take = ((self.state.get("jobs") or {}).get(ph.job) or {}).get("spec", {}).get("take")
        frames = take_frames(take) if take else {}
        if not frames:
            return None
        fp = P.read_json(WORK / take / "front" / "front_pod.json") or {}
        main, att = fp.get("main_cams"), fp.get("attached_cams")
        types = camera_types(take)  # EFFECTIVE types: what loc_part / stage_loc apply under the policy
        sizes = sam_sizes(take)
        T50 = front_dag(frames, types, main, att, self.rates, sizes=sizes)
        T90 = front_dag(frames, types, main, att, self.rates, conservative=True, sizes=sizes)
        n_done = 0
        running = []
        for e in evs:
            if e.get("src") not in ("front", "pod") or e.get("step") != "front" or not e.get("lane"):
                continue
            name = e["lane"]
            for T in (T50, T90):
                tgt = name
                if tgt not in T and tgt.count(":") == 2 and tgt.rsplit(":", 1)[0] in T:
                    # a localisation part the model did not split: it runs inside the camera's single loc task
                    tgt = tgt.rsplit(":", 1)[0]
                    if e["kind"] != "start":
                        continue  # one part ending does not end the camera (its join / last part does)
                if tgt not in T:
                    continue
                if e["kind"] == "start":
                    if T[tgt]["state"] == "pending":
                        T[tgt].update(state="running", t0=e.get("ts"))
                    # the model split this camera but the pod runs it whole: its parts are not separate work
                    for k in T[tgt].get("deps", []):
                        if k.startswith(tgt + ":") and T[k]["state"] == "pending":
                            T[k]["state"] = "skipped"
                elif e["kind"] in ("end", "fail"):
                    T[tgt]["state"] = "ok" if e["kind"] == "end" else "failed"
        for T in (T50, T90):
            for name, t in T.items():
                if t["state"] == "running":
                    el = self.now - (t.get("t0") or self.now)
                    t["remaining"] = max(t["duration"] - el, 0.15 * t["duration"], 5.0)
        n_done = sum(1 for t in T50.values() if t["state"] in ("ok", "failed"))
        running = [k for k, t in T50.items() if t["state"] == "running"]
        if ph.status != "running":
            d50, d90 = simulate_dag(T50, FRONT_CAPS), simulate_dag(T90, FRONT_CAPS)
            ph.detail = f"{len(T50)} tasks for {len(frames)} cameras" + (
                f" ({sum(1 for c in types.values() if c['mount'] == 'tripod')} tripod)" if types else "")
            return (d50 * math.exp(-Z90 * 0.2), d50, d90 * math.exp(Z90 * 0.2))
        d50, d90 = simulate_dag(T50, FRONT_CAPS), simulate_dag(T90, FRONT_CAPS)
        ph.fraction = n_done / max(1, len(T50))
        ph.detail = f"{n_done}/{len(T50)} tasks; running {', '.join(running[:4])}" + ("..." if len(running) > 4 else "")
        return (d50 * math.exp(-Z90 * 0.15), d50, d90 * math.exp(Z90 * 0.15))

    def _hist_views(self):
        for e in self.all_evs:
            if e.get("src") == "texture" and e["kind"] == "plan":
                return (e.get("extra") or {}).get("views")
        return None

    def _chunks(self, ph, evs, elapsed):
        plan = next((e for e in evs if e.get("src") in ("chunks", "pod") and e["kind"] == "plan"
                     and e.get("step") == "fit_chunks"), None)
        lanes = (plan.get("extra") or {}).get("lanes") if plan else None
        ratio, n, sig, _ = self._hist("mamma/lane/ratio", 1.0, 0.12)
        frames = 232
        ended = {e.get("lane"): (e.get("extra") or {}).get("seconds") for e in evs
                 if e.get("src") in ("chunks", "pod") and e.get("step") == "fit_chunks" and e["kind"] in ("end", "fail")
                 and e.get("lane")}
        for e in evs:
            if (e.get("extra") or {}).get("frames"):
                frames = e["extra"]["frames"]
        n_lanes = lanes or 7
        model = mamma_lane_model_s(frames, n_lanes)
        lane_dur = est(model * ratio, max(sig, 0.12))
        # in-lane iterations (L1 tails / L2 watch_log): fraction through runs 1..4
        it = {}
        for e in evs:
            if e.get("src") in ("mamma_iter", "pod", "tail") and str(e.get("step", "")).startswith(
                    "fit_chunks/run") and e.get("lane"):
                r = int(str(e["step"]).rsplit("run", 1)[1] or 1)
                it.setdefault(e["lane"], {})[r] = (e.get("done") or 0, e.get("total"), e["kind"])
        lane_rem = []
        n_running = max(0, n_lanes - len(ended))
        for lane, runs in it.items():
            if lane in ended:
                continue
            exp_total, done_u = 0.0, 0.0
            for r in range(1, 5):
                nmax = P.MAMMA_RUN_TOTALS[r - 1]
                q, _, _, _ = self._hist(f"mamma/run{r}/q_used", MAMMA_Q_USED[r - 1])
                if r in runs:
                    d, t, kind = runs[r]
                    tot = d if kind == "end" else expected_total(t or nmax, d, q)
                    exp_total += tot
                    done_u += d
                else:
                    exp_total += q * nmax
            x = done_u / exp_total if exp_total else 0
            lane_rem.append(remaining_profile(lane_dur[1], [], x, elapsed, 1, sig))
        if ph.status != "running":
            return lane_dur
        rem_l0 = remaining_after_elapsed(lane_dur, elapsed)
        ph.fraction = len(ended) / n_lanes if n_lanes else None
        ph.detail = f"{len(ended)}/{n_lanes} lanes done" + (" (in-lane iterations from tails)" if lane_rem
                                                             else " (L0: lane ends only)")
        if lane_rem:
            return lanes_max(lane_rem + ([rem_l0] if len(lane_rem) < n_running else []))
        return rem_l0 if n_running else est(5, 0.3)

    def _gate(self, ph, evs, elapsed):
        p_fail = ((self.rates or {}).get("keys", {}).get("mamma/gate_fail") or SEED["mamma/gate_fail"]).get("p", 0.5)
        fb = next((e for e in reversed(self.evs) if e.get("src") in ("chunks", "pod") and e["kind"] == "plan"
                   and (e.get("extra") or {}).get("branch") == "fallback"), None)
        gate_note = next((e for e in reversed(self.evs) if e.get("src") in ("chunks", "pod") and e["kind"] == "note"
                          and "gate" in (e.get("extra") or {})), None)
        ratio, _, sig, _ = self._hist("mamma/fallback/ratio", 0.80, 0.2)
        T = None
        for e in self.all_evs:
            if (e.get("extra") or {}).get("frames") and e.get("src") == "chunks" and e["kind"] == "plan":
                T = e["extra"]["frames"]
        printed = (fb.get("eta_s") or (fb.get("extra") or {}).get("eta_s")) if fb else mamma_solo_s(T or 1148)
        fallback = est(printed * ratio, max(sig, 0.2))
        passing = est(3, 0.3)
        if fb:
            self.branches = {"fallback": True}
            ph.detail = "gate FAILED: full-fit fallback running"
            return remaining_after_elapsed(fallback, elapsed) if ph.status == "running" else fallback
        if gate_note is not None and (gate_note.get("extra") or {}).get("gate") is True:
            return passing
        self.branches = {"p_fail": p_fail, "pass_s": passing[1], "fallback_s": fallback[1]}
        if p_fail >= 0.5:
            return (passing[0], fallback[1], fallback[2])
        return (passing[0], passing[1], fallback[2] if p_fail >= 0.1 else passing[2])

    def _room(self, ph, evs, elapsed):
        ticks = [e for e in evs if (e.get("src") == "room" or (e.get("src") == "pod" and e.get("step") == "train"))
                 and e["kind"] in ("tick", "end")]
        T, n, sig, _ = self._hist("room/train/s", 1944, 0.15)
        if not ticks:
            return remaining_after_elapsed(est(T + 600, 0.3), elapsed) if ph.status == "running" else est(T + 600)
        last = ticks[-1]
        x = (last.get("done") or 0) / max(1, last.get("total") or 30000)
        el = (last.get("extra") or {}).get("elapsed_s") or elapsed
        ph.fraction = x
        ph.detail = f"train {last.get('done')}/{last.get('total')} steps, PSNR {(last.get('extra') or {}).get('psnr')}"
        r = remaining_profile(T * (last.get("total") or 30000) / 30000, ROOM_PROFILE, x, el, n, sig, tail_s=480)
        return r

    def _splats(self, ph, evs, elapsed):
        """dancecap.dancer_splats: train = linear in steps over the (possibly shrunk) S1-S3 schedule (L2 events or the
        L0 '[dancer_splats] step N/M' lines), eval = linear in views; seeds until a run is in the history."""
        running = ph.status == "running"
        name = (ph.cmd or {}).get("name")
        if name == "eval":
            ticks = [e for e in evs if str(e.get("step", "")).endswith("splats/eval") and e.get("kind") in ("tick", "end")
                     and e.get("total")]
            s_h, n, sig, _ = self._hist("dancer_splats/eval/s_per_view", 1.2, 0.5)
            if not ticks:
                d = est(160 * s_h + 60, 0.5)
                return remaining_after_elapsed(d, elapsed) if running else d
            last = ticks[-1]
            ph.fraction = (last.get("done") or 0) / max(1, last["total"])
            ph.detail = f"eval {last.get('done')}/{last['total']} views"
            obs = [(e.get("ts") or e.get("rx"), e.get("done")) for e in ticks]
            return remaining_linear(last["total"], last.get("done") or 0, obs, s_h, n, sig, tail_s=60, now=self.now)
        ticks = [e for e in evs if str(e.get("step", "")).endswith("splats/train") and e.get("kind") in ("tick", "end")
                 and e.get("total")]
        s_h, n, sig, _ = self._hist("dancer_splats/train/s_per_step", 0.06, 0.35)
        if not ticks:
            d = est(25000 * s_h + 90, 0.4)
            return remaining_after_elapsed(d, elapsed) if running else d
        last = ticks[-1]
        x = last.get("extra") or {}
        ph.fraction = (last.get("done") or 0) / max(1, last["total"])
        ph.detail = (f"train {last.get('done')}/{last['total']} steps, stage {x.get('stage', '?')}, PSNR "
                     f"{x.get('psnr')}, {x.get('gaussians') or '?'} Gaussians")
        obs = [(e.get("ts") or e.get("rx"), e.get("done")) for e in ticks]
        diag = {"step": "splats/train"}
        r = remaining_linear(last["total"], last.get("done") or 0, obs, s_h, n, sig, now=self.now, diag=diag)
        self._flag(ph, diag)
        return r

    def stuck_steps(self) -> list[dict]:
        """Open steps of the running commands that HAVE a progress signal (>= 2 advances of `done`) and have not
        advanced for max(STALL_WARN_MIN_S, STALL_WARN_K x their median gap between advances). Heartbeats (the Writer
        says "alive" every 30 s while a step is open) never count as progress; they only tell a hung-but-alive process
        from a silent one. A step is closed by its end/fail, by done >= total, or by an end/fail on its lane."""
        out = []
        for ph in self.phases:
            if ph.status != "running" or ph.kind != "command":
                continue
            groups: dict[tuple, dict] = {}
            lane_end: dict = {}
            for e in self._cmd_events(ph):
                k = (e.get("step"), e.get("lane"))
                g = groups.setdefault(k, {"obs": [], "closed": False, "hb": None, "total": None})
                kind, ts = e.get("kind"), e.get("ts")
                if kind == "start":
                    g["closed"] = False
                    if ts is not None:
                        g["obs"].append((ts, 0))
                elif kind == "tick" and e.get("done") is not None and ts is not None:
                    g["obs"].append((ts, e["done"]))
                    g["total"] = e.get("total") or g["total"]
                    g["closed"] = False
                elif kind == "heartbeat" and ts is not None:
                    g["hb"] = ts
                elif kind in ("end", "fail"):
                    g["closed"] = True
                    if e.get("lane") is not None and ts is not None:
                        lane_end[e["lane"]] = max(lane_end.get(e["lane"], 0), ts)
            for (step, lane), g in groups.items():
                adv = _advances(g["obs"])
                if g["closed"] or len(adv) < 3 or (g["total"] and adv[-1][1] >= g["total"]):
                    continue
                if lane is not None and lane_end.get(lane, 0) >= adv[-1][0]:
                    continue
                gaps = sorted(b[0] - a[0] for a, b in zip(adv, adv[1:]) if b[0] > a[0])
                med = gaps[len(gaps) // 2] if gaps else 60.0
                since = self.now - adv[-1][0]
                if since > max(STALL_WARN_MIN_S, STALL_WARN_K * med):
                    out.append({"phase": ph.name, "step": step, "lane": lane, "since_s": round(since, 1),
                                "gap_s": round(med, 1), "done": adv[-1][1], "total": g["total"],
                                "alive": bool(g["hb"] and g["hb"] > adv[-1][0])})
        return out

    # ---- whole session
    def summary(self) -> dict:
        now = self.now
        phases = self.phases
        active = [p for p in phases if p.status == "running"]
        rem = [p.remaining for p in phases if p.status in ("running", "pending") and p.remaining]
        eta = add(*rem) if rem else (0.0, 0.0, 0.0)
        p90_budget = add_p90(*rem)
        # the pod bills until its last download ends (+ delete)
        pod_rem = [p.remaining for p in phases if p.status in ("running", "pending") and p.remaining
                   and p.kind not in ("bundle_scan", "bundle_build")]
        pod_eta = add(*pod_rem) if pod_rem else (0.0, 0.0, 0.0)
        pod_p90 = add_p90(*pod_rem)
        return {"eta": eta, "eta_p90_budget": p90_budget, "pod_eta": pod_eta, "pod_p90": pod_p90,
                "current": ", ".join(f"{p.name}" + (f" ({p.detail})" if p.detail else "") for p in active) or None}


def _state_at(state: dict, at: float | None) -> dict:
    if at is None or not state:
        return state
    s = dict(state)
    s["timings"] = [t for t in state.get("timings", []) if (t.get("start") or 0) + (t.get("seconds") or 0) <= at]
    pod = dict(state.get("pod") or {})
    if pod.get("started") and pod["started"] > at:
        pod.pop("started")
    s["pod"] = pod
    return s


def _ledger_at(rows: list, at: float | None) -> list:
    if at is None:
        return rows
    out = []
    for r in rows:
        if r.get("created_ts", 0) > at:
            continue
        r = dict(r)
        if r.get("deleted_ts") and r["deleted_ts"] > at:
            r.pop("deleted_ts", None)
            r.pop("cost", None)
            r.pop("deleted_reason", None)
        out.append(r)
    return out


def snapshot(session_dir, now: float | None = None, at: float | None = None, rates: dict | None = None,
             ledger_rows: list | None = None, events: list | None = None, runner_alive: bool | None = None) -> dict:
    """What the monitor shows for one session (section 6.6 + 8). `at` replays the session as of that epoch."""
    d = Path(session_dir)
    now = (at if at is not None else time.time()) if now is None else now
    rates = rates if rates is not None else load_rates()
    rows = ledger_rows if ledger_rows is not None else (P.read_json(LEDGER) or [])
    rows = _ledger_at(rows, at)
    state = _state_at(P.read_json(d / "state.json") or {}, at)
    evs = events if events is not None else session_events(d, at=at, rates=rates)
    if events is not None and at is not None:
        evs = [e for e in evs if e.get("ts") is not None and e["ts"] <= at]
    puller = P.read_json(d / "progress" / "puller.json") if at is None else None
    ri = P.read_json(d / "runner.json") if at is None else None
    if runner_alive is None and ri and ri.get("pid"):
        runner_alive = pid_alive(ri["pid"])
    level = "L1" if (d / "progress" / "runner.jsonl").exists() else "L0"
    if any(e.get("src") == "pod" for e in evs):
        level = "L2" if any(e.get("kind") == "plan" and e.get("src") == "pod" for e in evs) else "L1"
    m = SessionModel(d, evs, state, now, rates, rows, at=at, puller=puller)
    phases = m.build()
    s = m.summary()
    pod = m.pod
    row = m._ledger_row()
    price = (row or {}).get("cost_per_hr") or pod.get("cost_per_hr") or 0.0
    # state
    res = P.read_json(d / "result.json") if at is None else None
    last_ev = max((e.get("ts") or 0 for e in m.evs if e.get("kind") != "heartbeat"), default=0)  # alive != progress
    try:
        mtime = (d / "runner.log").stat().st_mtime if at is None else last_ev
    except OSError:
        mtime = last_ev
    ended = bool(m.evs) and all(p.status in ("done", "failed", "skipped") for p in phases if p.kind != "delete")
    dry = any(e.get("step") == "session" and "dry run" in (e.get("msg") or "") for e in m.evs)
    err = any(e.get("step") == "session" and e.get("kind") == "fail" for e in m.evs)
    open_row = bool(row) and not row.get("deleted_ts")
    if dry:
        state_s = "dry_run"
    elif (ended or err) and not open_row:
        state_s = "ended" if not err else "failed"
    elif res and (res.get("timings") and m.evs and (res.get("timings")[-1].get("start") or 0) >= (
            m.evs[0].get("ts") or 0) - 1) and not open_row:
        state_s = "ended"
    elif now - mtime > 1800 and not open_row:
        state_s = "stale"
    else:
        state_s = "running"
    if state_s in ("ended", "failed", "dry_run"):
        for ph in phases:
            if ph.status == "pending" and ph.kind == "delete":
                ph.status, ph.remaining = "done", (0.0, 0.0, 0.0)
    # kill switch
    kill = None
    ks = pod.get("killswitch") or {}
    if pod.get("started") and state_s == "running":
        kill_at = (puller or {}).get("kill_at") or (pod["started"] - 5 + float(ks.get("after_min") or
                                                                                 state.get("max_minutes") or 0) * 60)
        left = kill_at - now
        margin = kill_at - (now + s["pod_p90"])
        margin50 = kill_at - (now + s["pod_eta"][1])
        lvl = "red" if margin50 < 0 else ("amber" if margin < 300 else "green")
        # a live rate that one interval dominates (a just-resumed stall, the first ticks of a pass) is not evidence
        # enough for a red "outputs would be lost": amber until the rate settles
        unsettled = [p.name for p in phases if p.status == "running" and p.unstable]
        if lvl == "red" and unsettled:
            lvl = "amber"
        kill = {"kill_at": round(kill_at, 1), "left_s": round(left, 1), "margin_p90_s": round(margin, 1),
                "margin_p50_s": round(margin50, 1), "level": lvl, "armed": ks.get("armed"),
                "unsettled": unsettled or None,
                "source": "probe" if (puller or {}).get("kill_at") else "estimate (started + after_min)"}
        if lvl == "red":
            m.warnings.append({"level": "red", "msg": f"the kill switch fires ~{-margin50 / 60:.0f} min before the last "
                                                      "download: outputs would be lost"})
        elif lvl == "amber" and margin50 < 0:
            m.warnings.append({"level": "amber", "msg": f"the kill switch may fire ~{-margin50 / 60:.0f} min before the "
                                                        f"last download if the current rate holds (rate of "
                                                        f"{unsettled[0]} not settled yet)"})
        elif lvl == "amber":
            m.warnings.append({"level": "amber", "msg": f"kill-switch margin at p90 is {margin / 60:.0f} min"})
        if ks and not ks.get("armed"):
            m.warnings.append({"level": "red", "msg": "on-pod kill switch NOT armed: the local delete is the only guard"})
    # cost
    so_far = 0.0
    if row:
        end = row.get("deleted_ts") or now
        so_far = row.get("cost") if row.get("deleted_ts") and row.get("cost") is not None else \
            price * max(0.0, end - row["created_ts"]) / 3600
    if state_s == "running":
        pod_started = bool(row)
        rem50 = s["pod_eta"][1] if pod_started else s["pod_eta"][1] + 34
        cost = {"so_far": round(so_far, 4), "p50": round(so_far + price * rem50 / 3600, 4),
                "p90": round(so_far + price * s["pod_p90"] / 3600, 4),
                "worst": round(so_far + price * max(0.0, (kill["kill_at"] - now) if kill else
                                                     float(state.get("max_minutes") or 0) * 60) / 3600, 4),
                "price_per_hr": price}
    else:
        cost = {"so_far": round(so_far, 4), "p50": round(so_far, 4), "p90": round(so_far, 4),
                "worst": round(so_far, 4), "price_per_hr": price}
    # stalls
    if state_s == "running":
        running = [p for p in phases if p.status == "running"]
        # (a) a step with a progress signal that stopped advancing (heartbeats say "alive", not "progressing")
        stuck = m.stuck_steps()
        for x in stuck:
            where = x["step"] + (f"@{x['lane']}" if x.get("lane") else "")
            m.warnings.append({"level": "amber", "msg": f"no progress in {x['phase']} ({where}) for "
                               f"{x['since_s'] / 60:.0f} min at {x['done']}/{x['total'] or '?'}"
                               + (" - the process is alive (heartbeats) but not advancing" if x["alive"] else ""),
                               "stalled": x})
        # (b) phases without an in-step signal: quiet log (heartbeat events excluded from last_ev)
        ts_nonhb = [e.get("ts") or 0 for e in m.evs if e.get("kind") != "heartbeat"]
        gaps = sorted(b - a for a, b in zip(ts_nonhb, ts_nonhb[1:]) if b > a)
        med_gap = gaps[len(gaps) // 2] if gaps else 60
        quiet = now - max(mtime, last_ev)
        limit = max(300.0, 5 * med_gap)
        if running and quiet > limit and running[0].kind not in ("command",):
            m.warnings.append({"level": "amber", "msg": f"no progress for {quiet / 60:.0f} min (last: "
                                                        f"{running[0].name})"})
        elif running and not stuck and quiet > max(limit, 900):
            m.warnings.append({"level": "amber", "msg": f"no new log line for {quiet / 60:.0f} min during "
                                                        f"{running[0].name} (L0 sees no in-step progress)"})
        idle = (puller or {}).get("gpu_idle_since")
        if idle and now - idle > 300 and running and running[0].kind == "command" and \
                gpu_command(_cmd_key(running[0].job, running[0].cmd)):
            m.warnings.append({"level": "amber", "msg": f"GPU idle for {(now - idle) / 60:.0f} min during "
                                                        f"{running[0].name}"})
        if runner_alive is False and open_row:
            m.warnings.append({"level": "red", "msg": "pod alive but its runner process is gone: orphan (the kill "
                                                      "switch deletes it at its deadline)"})
    # QA: probe B's camera_motion metric (cameras stage qa step, today's unconstrained track) vs the user's camera
    # types, with camera_types' thresholds (section 9.2); the latest measurement per camera
    mets = {}
    for e in evs:
        x = e.get("extra") or {}
        if e.get("kind") == "metric" and x.get("metric") == "camera_motion" and x.get("source") == "probe_b":
            mets[str(e.get("lane"))] = e
    if mets:
        take = next((v.get("spec", {}).get("take") for v in (state.get("jobs") or {}).values()
                     if v.get("spec", {}).get("take")), None)
        types = camera_types(take) if take else {}
        for cam, e in sorted(mets.items()):
            for w in camera_type_flags(cam, types.get(cam), e.get("extra") or {}, at=e.get("ts")):
                m.warnings.append(w)
    eta = s["eta"]
    out = {
        "schema": "dancecap.progress.snapshot/1", "sid": d.name, "dir": str(d), "now": round(now, 1),
        "replay_at": at, "level": level, "state": state_s,
        "current": s["current"],
        "phases": [p.as_dict(now) for p in phases],
        "eta": {"p10": round(eta[0], 1), "p50": round(eta[1], 1), "p90": round(eta[2], 1),
                "p90_budget": round(s["eta_p90_budget"], 1),
                "finish_p50": round(now + eta[1], 1), "finish_p90": round(now + eta[2], 1),
                "branches": m.branches} if state_s == "running" else None,
        "kill": kill,
        "cost": cost,
        "pod": {"id": pod.get("id"), "gpu": pod.get("gpu"), "gpu_class": gpu_class(pod.get("gpu")) if pod.get("gpu")
                else None, "price_per_hr": price, "started": pod.get("started"), "open": open_row,
                "connection": _connection(state_s, now, mtime, puller, open_row)},
        "warnings": m.warnings,
        "runner": {"pid": (ri or {}).get("pid"), "alive": runner_alive, "attempt": (ri or {}).get("attempt")}
        if ri else None,
    }
    return out


def pid_alive(pid) -> bool | None:
    """Is a local process alive? Read-only (never os.kill on Windows: any signal there TERMINATES the process)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    try:
        import psutil
        return bool(psutil.pid_exists(pid))
    except ImportError:
        pass
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            code = ctypes.c_ulong()
            k32.GetExitCodeProcess(h, ctypes.byref(code))
            k32.CloseHandle(h)
            return code.value == 259  # STILL_ACTIVE
        except Exception:
            return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return None


def local_stage_snapshot(take: str, stage: str, now: float | None = None, work: Path = WORK) -> dict:
    """Progress of a local stage from work/<take>/progress/<stage>[.<pid>].jsonl (written by the helper when the
    launcher set DANCECAP_PROGRESS): open steps with fraction and a live-rate ETA, the last event time."""
    now = time.time() if now is None else now
    d = Path(work) / take / "progress"
    evs = []
    for f in sorted(d.glob(f"{stage}*.jsonl")) if d.exists() else []:
        more, _ = P.read_jsonl(f)
        evs += more
    evs.sort(key=lambda e: e.get("ts") or 0)
    steps: dict[tuple, dict] = {}
    for e in evs:
        k = (e.get("step"), e.get("lane"))
        st = steps.setdefault(k, {"step": k[0], "lane": k[1], "status": "pending", "obs": [], "total": None})
        if e["kind"] == "start":
            st.update(status="running", start=e.get("ts"), total=e.get("total") or st["total"])
        elif e["kind"] == "tick":
            st["status"] = "running"
            if e.get("done") is not None:
                st["obs"].append((e.get("ts"), e["done"]))
                st["total"] = e.get("total") or st["total"]
        elif e["kind"] == "heartbeat":  # alive, not progress: never an observation of `done`
            st["status"] = "running"
            st["alive_at"] = e.get("ts")
        elif e["kind"] in ("end", "fail"):
            st.update(status="done" if e["kind"] == "end" else "failed", end=e.get("ts"))
    out = []
    for st in steps.values():
        obs, N = st.pop("obs"), st.get("total")
        done = obs[-1][1] if obs else 0
        if st["status"] == "running" and N:
            diag: dict = {}
            r = remaining_linear(N, done, obs, None, 0, started_at=st.get("start"), now=now, diag=diag)
            st.update(done=done, fraction=round(done / N, 4), remaining={"p10": round(r[0], 1), "p50": round(r[1], 1),
                                                                       "p90": round(r[2], 1)})
            adv = _advances(obs)
            gap = diag.get("med_gap_s") or 60.0
            if len(adv) >= 3 and now - adv[-1][0] > max(STALL_WARN_MIN_S, STALL_WARN_K * gap):
                st["stalled"] = {"since_s": round(now - adv[-1][0], 1), "gap_s": round(gap, 1),
                                 "alive": bool(st.get("alive_at") and st["alive_at"] > adv[-1][0])}
        out.append(st)
    last = evs[-1].get("ts") if evs else None
    return {"take": take, "stage": stage, "steps": [s for s in out if s["step"] not in ("plan", "note")],
            "notes": [e.get("msg") for e in evs if e["kind"] == "note"][-5:],
            "metrics": [e for e in evs if e["kind"] == "metric"][-20:], "last_event": last,
            "quiet_s": None if last is None else round(now - last, 1)}


def _connection(state_s, now, mtime, puller, open_row) -> str:
    if state_s != "running":
        return "ended"
    if puller and puller.get("last_ok"):
        age = now - puller["last_ok"]
        iv = puller.get("interval_s", 15)
        return "connected" if age < 2 * iv else ("degraded" if age < 5 * iv else "lost")
    if not open_row:
        return "creating"
    return "connected" if now - mtime < 120 else ("degraded" if now - mtime < 600 else "quiet")


class Tracker:
    """Incremental, live reader of one session (what a long-lived monitor keeps per session): new runner.log lines are
    parsed from the stored offset and stamped with their receive time, so L0 live rates are exact.

    Receive time: lines that arrive within LIVE_BATCH_S of the previous poll get the file's mtime (the last write);
    a larger batch (nobody polled for a while) is spread linearly over (previous poll, mtime] by line number. Lines
    already in the file at the first poll are timed like snapshot() does (interpolation between clocked lines).
    Thread-safe (one lock per tracker): live_snapshot()'s background poller and request threads share it."""

    def __init__(self, session_dir, rates: dict | None = None):
        import threading
        self.dir = Path(session_dir)
        self.lock = threading.RLock()
        self._rates_given = rates is not None
        self.rates = rates if rates is not None else load_rates()
        self._rates_mt = _mtime(RATES)
        self._reset()

    def _reset(self) -> None:
        self.parser = P.LineParser(P.RUNNER_LOG_ADAPTERS, date0=P.sid_date(self.dir.name))
        self.offset = 0
        self.events: list[dict] = []
        self.first = True
        self.l1_off = {"runner.jsonl": 0, "pod.jsonl": 0}
        self.l1: list[dict] = []
        self.last_poll: float | None = None

    def feed_lines(self, lines: list[str], rx: float) -> None:
        with self.lock:
            for ln in lines:
                self.events += self.parser.feed(ln, rx=None if self.first else rx)

    def poll(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self.lock:
            log = self.dir / "runner.log"
            try:
                st = log.stat()
                size, mt = st.st_size, st.st_mtime
            except OSError:
                size = mt = None
            if size is not None and size < self.offset:  # rewritten / truncated: start over
                self._reset()
            n0 = self.parser.n
            evs, self.offset, self.parser = P.parse_runner_log(log, self.offset, self.parser,
                                                               rx=None if self.first else now)
            if evs and not self.first:
                t1 = min(now, mt) if mt else now
                t0 = self.last_poll
                if t0 is not None:
                    t1 = max(t1, t0)
                n1 = self.parser.n
                if t0 is not None and t1 - t0 > LIVE_BATCH_S and n1 > n0:
                    for e in evs:
                        e["rx"] = t0 + (t1 - t0) * (e.get("line", n1) - n0) / (n1 - n0)
                else:
                    for e in evs:
                        e["rx"] = t1
            self.events += evs
            if self.first:
                _fill_times(self.events, mt, _pre_pv(self.rates), self.parser.n)
                self.first = False
            for f in self.l1_off:
                more, self.l1_off[f] = P.read_jsonl(self.dir / "progress" / f, self.l1_off[f])
                for e in more:
                    e.setdefault("src", "l1" if f == "runner.jsonl" else "pod")
                self.l1 += more
            self.last_poll = now

    def snapshot(self, now: float | None = None, poll: bool = True) -> dict:
        now = time.time() if now is None else now
        with self.lock:
            if poll:
                self.poll(now)
            if not self._rates_given:  # rates.json is rebuilt at every session end
                mt = _mtime(RATES)
                if mt != self._rates_mt:
                    self.rates, self._rates_mt = load_rates(), mt
            for e in self.events:
                if e.get("ts") is None and e.get("rx") is not None:
                    e["ts"] = e["rx"]
            evs = self.events + self.l1
        return snapshot(self.dir, now=now, rates=self.rates, events=evs)


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


# live trackers of the monitor process (docs/PIPELINE_PROGRESS.md 10.1): one Tracker per session somebody looks at,
# polled in the background every LIVE_POLL_S so L0 lines get their receive time even between page refreshes.
LIVE_POLL_S = 3.0        # background poll interval (open, read from the offset, close: never holds a handle)
LIVE_BATCH_S = 10.0      # lines arriving within this of the previous poll are stamped with the file mtime
LIVE_IDLE_S = 1800.0     # a session nobody asked about for 30 min is dropped (a later request starts a fresh one)
LIVE_ENDED_S = 120.0     # an ended session's tracker is dropped 2 min after its end was seen
LIVE_MAX = 8
_live: dict[str, dict] = {}
_live_thread: list = [None]


def _live_lock():
    import threading
    lk = globals().get("_LIVE_LOCK")
    if lk is None:
        lk = globals().setdefault("_LIVE_LOCK", threading.Lock())
    return lk


def live_snapshot(session_dir, now: float | None = None, poll_thread: bool = True) -> dict:
    """snapshot() for a long-lived monitor: the session's live Tracker (created on first use, then polled in the
    background every LIVE_POLL_S while somebody keeps asking), so timestamp-less L0 lines (texture views, MAMMA lanes,
    hands / face workers of runners that predate the pod events) carry their receive time - the path whose accuracy
    the live validation measured (PIPELINE_PROGRESS.md 12). Read-only; any tracker failure falls back to
    snapshot()."""
    d = Path(session_dir)
    key = str(d.resolve())
    t = time.time()
    with _live_lock():
        ent = _live.get(key)
        if ent is None:
            while len(_live) >= LIVE_MAX:
                _live.pop(min(_live, key=lambda k: _live[k]["req"]))
            ent = _live[key] = {"tracker": Tracker(d), "req": t, "ended": None}
        ent["req"] = t
    if poll_thread:
        _ensure_live_poller()
    try:
        snap = ent["tracker"].snapshot(now=now)
    except Exception:
        with _live_lock():
            _live.pop(key, None)
        return snapshot(d, now=now)
    with _live_lock():
        ent["ended"] = None if snap.get("state") == "running" else (ent["ended"] or t)
    return snap


def live_tracker(session_dir) -> "Tracker | None":
    """The monitor's live Tracker of a session, when one exists (budget() reuses it)."""
    with _live_lock():
        ent = _live.get(str(Path(session_dir).resolve()))
    return ent["tracker"] if ent else None


def _ensure_live_poller() -> None:
    import threading
    with _live_lock():
        th = _live_thread[0]
        if th is not None and th.is_alive():
            return

        def loop():
            while True:
                time.sleep(LIVE_POLL_S)
                now = time.time()
                with _live_lock():
                    for k in [k for k, e in _live.items() if now - e["req"] > LIVE_IDLE_S
                              or (e["ended"] and now - e["ended"] > LIVE_ENDED_S)]:
                        _live.pop(k, None)
                    ents = list(_live.values())
                    if not ents:
                        _live_thread[0] = None
                        return
                for e in ents:
                    try:
                        e["tracker"].poll()
                    except Exception:
                        pass
        th = threading.Thread(target=loop, name="dancecap-progress-live", daemon=True)
        _live_thread[0] = th
        th.start()


# ---------------------------------------------------------------------------------------------------- budget


def _remote_cap() -> float:
    """remote.cap() (DANCECAP_RUNPOD_CAP or remote.DEFAULT_CAP): the one definition of the budget cap."""
    try:
        from . import remote
        return float(remote.cap())
    except Exception:
        return float(os.environ.get("DANCECAP_RUNPOD_CAP", 30.0))


def spent_from_rows(rows: list, now: float | None = None) -> float:
    """Exactly remote.spent(): closed rows' recorded cost, open rows price x lifetime so far."""
    now = time.time() if now is None else now
    total = 0.0
    for r in rows:
        if r.get("deleted_ts") and r.get("cost") is not None:
            total += float(r["cost"])
            continue
        end = r.get("deleted_ts") or now
        total += r["cost_per_hr"] * max(0.0, end - r["created_ts"]) / 3600
    return total


def budget(now: float | None = None, rows: list | None = None, cap: float | None = None, reserve: float | None = None,
           snapshots: list | None = None, queued: list | None = None, pods_listed: list | None = None,
           account: dict | None = None, sessions_dir: Path = SESSIONS) -> dict:
    """Section 8: spent S (= remote.spent()), live cost of open pods, projected p50/p90, hard worst case from the kill
    switches, warnings (green/amber/red). snapshots: progress snapshots of running sessions (computed when None);
    queued: [{p50_cost, p90_cost, label}] sessions planned but not started; pods_listed: `runpodctl pod list` ids
    (optional: phantom / untracked checks); account: remote.status()['account'] (balance/spend/limit only)."""
    now = time.time() if now is None else now
    rows = rows if rows is not None else (P.read_json(LEDGER) or [])
    cap = _remote_cap() if cap is None else cap
    reserve = float(os.environ.get("DANCECAP_RUNPOD_RESERVE", DEFAULT_RESERVE)) if reserve is None else reserve
    S = spent_from_rows(rows, now)
    open_rows = [r for r in rows if not r.get("deleted_ts")]
    if snapshots is None:
        snapshots = []
        for sd in active_sessions(rows=rows, now=now, sessions_dir=sessions_dir):
            try:
                tr = live_tracker(sd)
                snapshots.append(tr.snapshot(now=now) if tr is not None else snapshot(sd, now=now, ledger_rows=rows))
            except Exception as e:  # a broken session file never breaks the budget view
                snapshots.append({"sid": sd.name, "error": str(e)[:200]})
    by_pod = {(s.get("pod") or {}).get("id"): s for s in snapshots if s.get("pod")}
    pods = []
    P50 = P90 = W = 0.0
    warnings = []
    for r in open_rows:
        price = r["cost_per_hr"]
        so_far = price * max(0.0, now - r["created_ts"]) / 3600
        kill_at = r["created_ts"] + (float(r.get("max_minutes") or 0) + KILL_START_TIMEOUT_MIN) * 60
        snap = by_pod.get(r["pod_id"])
        if snap and snap.get("kill"):
            kill_at = snap["kill"]["kill_at"]
        w = price * max(0.0, kill_at - now) / 3600
        if snap and snap.get("state") == "running":
            p50 = snap["cost"]["p50"] - snap["cost"]["so_far"]
            p90 = snap["cost"]["p90"] - snap["cost"]["so_far"]
        else:
            p50 = p90 = w  # unknown remaining work: the kill switch bounds it
        phantom = None
        if now > kill_at + 120:  # past its kill time: the pod must be gone; spent() keeps accruing a ghost
            phantom = price * max(0.0, kill_at - r["created_ts"]) / 3600
            warnings.append({"level": "amber", "msg": f"ledger row {r['pod_id']} is open past its kill time: spent() "
                                                       f"over-counts ${so_far - phantom:.2f} (reconcile with remote "
                                                       "cleanup)"})
            p50 = p90 = w = 0.0
        if pods_listed is not None and r["pod_id"] not in pods_listed and now - r["created_ts"] > 120:
            phantom = phantom if phantom is not None else so_far
            warnings.append({"level": "amber", "msg": f"pod {r['pod_id']} is open in the ledger but not in pod list"})
        P50 += p50
        P90 += p90
        W += w
        pods.append({"pod_id": r["pod_id"], "gpu": r.get("gpu"), "price_per_hr": price, "so_far": round(so_far, 4),
                     "remaining_p50": round(p50, 4), "remaining_p90": round(p90, 4), "worst_remaining": round(w, 4),
                     "kill_at": round(kill_at, 1), "session": (snap or {}).get("sid"), "phantom_cost": phantom})
        if snap and snap.get("kill") and snap["kill"].get("armed") is False:
            warnings.append({"level": "red", "msg": f"pod {r['pod_id']}: kill switch not armed"})
    Q50 = sum(q.get("p50_cost", 0) for q in queued or [])
    Q90 = sum(q.get("p90_cost", 0) for q in queued or [])
    T50, T90, Wc = S + P50 + Q50, S + P90 + Q90, S + W + Q90
    remaining = cap - S
    level = "green"
    if T90 > cap - reserve or remaining < CHEAPEST_80GB_PER_HR:
        level = "amber"
        warnings.append({"level": "amber", "msg": f"projected p90 ${T90:.2f} leaves less than the ${reserve:.2f} "
                                                   f"reserve of the ${cap:.2f} cap" if T90 > cap - reserve else
                         f"remaining ${remaining:.2f} is under one hour of the cheapest 80 GB GPU"})
    if T50 > cap or Wc > cap:
        level = "red"
        warnings.append({"level": "red", "msg": f"projected ${T50:.2f} (p50) / worst ${Wc:.2f} exceeds the "
                                                 f"${cap:.2f} cap"})
    if any(w["level"] == "red" for w in warnings):
        level = "red"
    elif level == "green" and warnings:
        level = "amber"
    if account:
        rate = account.get("currentSpendPerHr")
        if isinstance(rate, (int, float)) and rate > sum(r["cost_per_hr"] for r in open_rows) + 0.05:
            level = "red"
            warnings.append({"level": "red", "msg": f"Runpod bills ${rate:.2f}/h but the ledger knows "
                                                     f"${sum(r['cost_per_hr'] for r in open_rows):.2f}/h: an untracked "
                                                     "pod"})
        bal = account.get("clientBalance")
        if isinstance(bal, (int, float)) and bal < remaining:
            warnings.append({"level": "amber", "msg": f"Runpod balance ${bal:.2f} is below the remaining cap "
                                                       f"${remaining:.2f}"})
    return {"schema": "dancecap.progress.budget/1", "now": round(now, 1), "cap": cap, "reserve": reserve,
            "spent": round(S, 4), "remaining": round(remaining, 4), "open_pods": pods,
            "projected": {"p50": round(T50, 4), "p90": round(T90, 4)}, "worst": round(Wc, 4),
            "queued": queued or [], "level": level, "warnings": warnings,
            "phase_costs": phase_costs(snapshots)}


def phase_costs(snapshots: list) -> list:
    """Where the money goes, per running session: cost of each phase so far (pod-time x price)."""
    out = []
    for s in snapshots or []:
        price = (s.get("pod") or {}).get("price_per_hr") or 0
        if not price or not s.get("phases"):
            continue
        rows = []
        for ph in s["phases"]:
            if ph["kind"] in ("bundle_scan", "bundle_build") or ph.get("elapsed_s") is None:
                continue
            rows.append({"phase": ph["name"], "minutes": round(ph["elapsed_s"] / 60, 2),
                         "usd": round(price * ph["elapsed_s"] / 3600, 4)})
        out.append({"sid": s.get("sid"), "phases": rows})
    return out


def launch_gate(p90_new: float, now: float | None = None, rows: list | None = None, cap: float | None = None,
                snapshots: list | None = None) -> dict:
    """Section 8.3: allowed iff S + sum(worst case of running pods) + P90_new <= cap."""
    b = budget(now=now, rows=rows, cap=cap, snapshots=snapshots or [])
    need = b["spent"] + sum(p["worst_remaining"] for p in b["open_pods"]) + p90_new
    return {"allowed": need <= b["cap"], "need": round(need, 4), "cap": b["cap"], "spent": b["spent"],
            "running_worst": round(sum(p["worst_remaining"] for p in b["open_pods"]), 4), "p90_new": p90_new}


def active_sessions(rows: list | None = None, now: float | None = None, sessions_dir: Path = SESSIONS,
                    recent_s: float = 120.0) -> list[Path]:
    """Sessions that may be running: an open ledger row for state.pod.id, or runner.log changed in the last
    `recent_s` (result.json alone is not proof of an end: a resumed sid keeps the old one)."""
    now = time.time() if now is None else now
    rows = rows if rows is not None else (P.read_json(LEDGER) or [])
    open_ids = {r["pod_id"] for r in rows if not r.get("deleted_ts")}
    out = []
    dirs = sorted(Path(sessions_dir).glob("*"), key=lambda p: p.name)[-40:] if Path(sessions_dir).exists() else []
    for d in dirs:
        if not d.is_dir():
            continue
        try:
            mt = (d / "runner.log").stat().st_mtime
        except OSError:
            mt = 0
        st = P.read_json(d / "state.json") or {}
        pid = (st.get("pod") or {}).get("id")
        if (pid and pid in open_ids) or now - mt < recent_s:
            out.append(d)
    return out


# ---------------------------------------------------------------------------------------------------- planning


def estimate_session(jobs: list[dict], upload_bytes: float, features: list[str], price_per_hr: float,
                     gpu: str | None = None, rates: dict | None = None, wheels_cached: bool = True) -> dict:
    """p50/p90 minutes and $ per phase for a session that has not started (launch dialog). jobs: pod_job specs (dicts
    with name, commands[{name, cmd, timeout_min, optional}], archives, setup). Uses the same history as the live ETA."""
    rates = rates if rates is not None else load_rates()
    fake = {"bundle": {"bytes": upload_bytes}, "jobs": {j["name"]: {"status": "pending", "spec": j} for j in jobs},
            "pod": {"gpu": gpu, "cost_per_hr": price_per_hr}}
    m = SessionModel(Path("planned"), [], fake, time.time(), rates, [])
    m.features = features
    phases = m.build()
    m.features = features
    for ph in phases:
        if ph.kind == "bootstrap":
            ph.remaining = m._bootstrap(ph, 0)
    rows = []
    for ph in phases:
        r = ph.remaining or (0, 0, 0)
        rows.append({"phase": ph.name, "p50_min": round(r[1] / 60, 2), "p90_min": round(r[2] / 60, 2),
                     "billed": ph.kind not in ("bundle_scan", "bundle_build")})
    billed = [ph.remaining for ph in phases if ph.kind not in ("bundle_scan", "bundle_build") and ph.remaining]
    tot = add(*billed)
    p90 = add_p90(*billed)
    return {"phases": rows, "p50_min": round(tot[1] / 60, 1), "p90_min": round(p90 / 60, 1),
            "p50_cost": round(price_per_hr * tot[1] / 3600, 3), "p90_cost": round(price_per_hr * p90 / 3600, 3),
            "price_per_hr": price_per_hr, "gpu": gpu}
