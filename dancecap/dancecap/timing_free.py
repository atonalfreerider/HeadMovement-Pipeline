"""Step timing WITHOUT a beat grid: touchdowns, accents and step intervals of a take that has no soundtrack.

    python -m dancecap.timing_free <take> [--smplx-dir smplx] [--mc 60] [--workers 8] [--mc-corr-noise-mm 10] [--no-plot]

The class recap (larissa_kadu_recap) is two teachers talking and demonstrating, not a dance to a song: there is no beat
grid, no tempo, no rhythm to be on or off. `dancecap timing` needs a grid (its asynchrony, hit rate, surrogate beat tests and
tempo are all measured against it), so this module runs only the parts of it that never looked at a beat:

    kept      the footstep touchdown detector (timing.analyze_dancer: contact model, per-step hinge, per-foot matched
              filter, Monte-Carlo sigma per event), the accents (|acceleration| / |jerk| maxima of the key joints) and
              timing/kinematics.npz (the quintic smoothing spline derivatives the viewer's jerk glow reads)
    added     STEP TIMING in seconds only: per dancer and foot the intervals between successive touchdowns, bouts (runs
              of steps no more than `bout_gap_s` apart: a step pattern, as opposed to standing and talking), the cadence
              (steps per second) of each bout, steps per 10-s window, and the lead / follow touchdown offsets (the
              nearest follow touchdown to each lead touchdown within +-`couple_max_s`: partner timing relative to each
              other, which needs no clock)
    SKIPPED   every beat / rhythm / BPM product: the beat grid (`beats.grid` is null), tempo, asynchrony of an event from
              the nearest beat, hit rate, circular statistics (R, Rayleigh, V), the surrogate and permutation tests against
              the beat, measure effects, absolute timing (`absolute_timing.valid` false: no audio-video sync event). They
              are listed under `skipped` in timing.json so nobody reads an empty field as a result.

Output work/<take>/timing/timing.json (same top-level keys as the beat version where they still mean something, so
export_unity._timing_for_unity and the viewer read it; `beatless` true, `step_timing` new), timing/kinematics.npz (same
arrays as the beat version) and timing/qa_timing.png. Inputs are the raw fit smplx/{lead,follow}.npz, as the demo's timing
was (joints (T,55,3), Y-up metres, floor y = 0); runs headless on a CPU pod (numpy, scipy, matplotlib for the plot).

WINDOWED: timing.py builds dense smoother operators of (4 x samples) x samples, so the memory grows with the square of the take length:
the demo's 188 s fitted in a 16 GB pod, the recap's 360 s did not (SIGKILL after 11 min, twice). The take is therefore cut into cores
of `core_s` = 30 s, each analysed with `margin_s` = 3 s of context on both sides (the detector only ever used a few seconds of context:
a step, its template over the foot's steps in the window), and only the events and derivatives of the CORE are kept. The joint noise is
estimated once over the whole take and shared by every window.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import timing as tm

ROLES = tm.ROLES
VERSION = 2
BOUT_GAP_S = 1.5          # touchdowns of one dancer closer than this belong to one step pattern (a bout)
COUPLE_MAX_S = 0.5        # lead / follow touchdowns this close are one pair (relative timing)
WINDOW_S = 10.0           # steps-per-window series
CORE_S = 30.0             # analysis windows: core length ...
MARGIN_S = 3.0            # ... and context on both sides (dropped from the result)
SKIPPED = [
    "beat grid (beats.grid is null, no tempo / BPM / measure structure)",
    "asynchrony of events from the nearest beat, hit rate, mean asynchrony, circular statistics (R, Rayleigh, V)",
    "surrogate (chance) and permutation tests against the beat, measure effects",
    "absolute timing (no audio-video sync event; the take has no music to be on or off)",
]


# ------------------------------------------------------------------------------------------------ pure statistics


def _stats(x) -> dict | None:
    """n, median, p10, p90, min, max of a 1-D array of seconds (None when empty)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {"n": int(len(x)), "median_s": round(float(np.median(x)), 4), "p10_s": round(float(np.percentile(x, 10)), 4),
            "p90_s": round(float(np.percentile(x, 90)), 4), "min_s": round(float(x.min()), 4), "max_s": round(float(x.max()), 4)}


def intervals(times) -> np.ndarray:
    """Successive differences of sorted touchdown times (seconds)."""
    t = np.sort(np.asarray(times, float))
    return np.diff(t) if len(t) > 1 else np.zeros(0)


def bouts(times, gap_s: float = BOUT_GAP_S, min_steps: int = 2) -> list[dict]:
    """Runs of touchdowns no more than `gap_s` apart: [{t0, t1, n, cadence_hz}] with cadence = (n - 1) / (t1 - t0) steps per
    second (a bout of fewer than `min_steps` touchdowns is a single step, not a pattern, and is dropped)."""
    t = np.sort(np.asarray(times, float))
    out = []
    if not len(t):
        return out
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or t[i] - t[i - 1] > gap_s:
            n = i - start
            if n >= min_steps:
                t0, t1 = float(t[start]), float(t[i - 1])
                out.append({"t0": round(t0, 4), "t1": round(t1, 4), "n": int(n),
                            "cadence_hz": round((n - 1) / (t1 - t0), 4) if t1 > t0 else None})
            start = i
    return out


def steps_per_window(times, t0: float, t1: float, win_s: float = WINDOW_S) -> list[dict]:
    """Touchdown counts per consecutive window of `win_s` seconds over [t0, t1]: [{t0, t1, n}]."""
    t = np.asarray(times, float)
    edges = np.arange(float(t0), float(t1) + win_s, win_s)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append({"t0": round(float(a), 3), "t1": round(float(min(b, t1)), 3), "n": int(((t >= a) & (t < b)).sum())})
    return out


def couple_offsets(lead_t, follow_t, max_s: float = COUPLE_MAX_S) -> dict:
    """Relative timing of the two dancers: for each lead touchdown the signed offset (follow minus lead, seconds) to the NEAREST
    follow touchdown within `max_s`. Positive = the follow lands after the lead. Needs no clock and no beat."""
    lt, ft = np.sort(np.asarray(lead_t, float)), np.sort(np.asarray(follow_t, float))
    off = []
    if len(lt) and len(ft):
        j = np.searchsorted(ft, lt)
        for i, tl in enumerate(lt):
            cands = [ft[k] for k in (j[i] - 1, j[i]) if 0 <= k < len(ft)]
            d = min((c - tl for c in cands), key=abs)
            if abs(d) <= max_s:
                off.append(d)
    off = np.asarray(off, float)
    if not len(off):
        return {"n_pairs": 0, "max_s": max_s, "median_ms": None, "mad_ms": None, "p10_ms": None, "p90_ms": None,
                "share_of_lead_steps_paired": 0.0}
    return {"n_pairs": int(len(off)), "max_s": max_s, "median_ms": round(float(np.median(off)) * 1000, 1),
            "mad_ms": round(float(np.median(np.abs(off - np.median(off)))) * 1000, 1),
            "p10_ms": round(float(np.percentile(off, 10)) * 1000, 1), "p90_ms": round(float(np.percentile(off, 90)) * 1000, 1),
            "share_of_lead_steps_paired": round(len(off) / max(len(lt), 1), 3),
            "note": "follow touchdown minus the nearest lead touchdown, seconds of the video clock; relative timing only"}


def step_summary(events: list[dict], t0: float, t1: float, gap_s: float = BOUT_GAP_S) -> dict:
    """Step timing of one dancer from its touchdown events [{t, foot, ...}]: counts, intervals (all steps and per foot),
    bouts, the share of the take spent inside bouts and the steps per window."""
    times = [e["t"] for e in events]
    b = bouts(times, gap_s)
    in_bout = sum(x["t1"] - x["t0"] for x in b)
    per_foot = {}
    for foot in ("left", "right"):
        ft = [e["t"] for e in events if e["foot"] == foot]
        per_foot[foot] = {"n": len(ft), "interval": _stats(intervals(ft))}
    span = max(float(t1) - float(t0), 1e-9)
    return {"n_touchdowns": len(times), "interval_all": _stats(intervals(times)), "per_foot": per_foot,
            "bout_gap_s": gap_s, "n_bouts": len(b), "bouts": b, "share_in_bouts": round(in_bout / span, 4),
            "longest_still_s": round(float(intervals(times).max()), 3) if len(times) > 1 else None,
            "steps_per_window": steps_per_window(times, t0, t1)}


# ------------------------------------------------------------------------------------------------ run



def split_windows(t, core_s: float = CORE_S, margin_s: float = MARGIN_S) -> list[tuple[int, int, int, int]]:
    """[(i0, i1, c0, c1)]: the sample slice [i0, i1) that is analysed (the core plus `margin_s` of context on both sides, clipped to the
    take) and the core's sample range [c0, c1). The cores tile the take; a tail shorter than half a core joins the last core."""
    t = np.asarray(t, float)
    n = len(t)
    dt = float(np.median(np.diff(t)))
    core = max(8, int(round(core_s / dt)))
    mg = int(round(margin_s / dt))
    out, c0 = [], 0
    while c0 < n:
        c1 = min(n, c0 + core)
        if n - c1 < core // 2:
            c1 = n
        out.append((max(0, c0 - mg), min(n, c1 + mg), c0, c1))
        c0 = c1
    return out


def in_core(events: list[dict], t_lo: float, t_hi: float | None) -> list[dict]:
    """Events with t_lo <= t < t_hi (t_hi None = open end)."""
    return [e for e in events if e["t"] >= t_lo and (t_hi is None or e["t"] < t_hi)]


def _med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def analyse_windowed(t: np.ndarray, joints: np.ndarray, cfg: tm.TimingConfig, rng) -> dict:
    """tm.analyze_dancer over overlapping windows -> one dancer's result (touchdowns, accents, derivative summary, method info),
    events kept from each window's core only."""
    noise = cfg.noise_m if cfg.noise_m is not None else tm.estimate_noise(joints[:, tm._needed(cfg)])
    noise = max(noise, cfg.noise_floor_m)
    cfg_w = tm.TimingConfig(**{**asdict(cfg), "noise_m": noise})
    wins = split_windows(t)
    tds, accents, winfo, deriv, cross = [], {}, [], [], []
    mc_info = None
    for i0, i1, c0, c1 in wins:
        d = tm.analyze_dancer(t[i0:i1], joints[i0:i1], cfg_w, rng)
        lo, hi = float(t[c0]), (None if c1 >= len(t) else float(t[c1]))
        if c0 == 0:
            lo = -np.inf
        tds += in_core(d["touchdowns"], lo, hi)
        for name, kinds in d["accents"].items():
            for kind, evs in kinds.items():
                accents.setdefault(name, {}).setdefault(kind, []).extend(in_core(evs, lo, hi))
        winfo.append({"core_t": [round(float(t[c0]), 3), round(float(t[min(c1, len(t)) - 1]), 3)], "samples": int(i1 - i0),
                      "n_core_touchdowns": len(in_core(d["touchdowns"], lo, hi)),
                      "suspect_feet": d["touchdown_info"].get("suspect_feet"), "n_template": d["touchdown_info"].get("n_template"),
                      "n_fallback": d["touchdown_info"].get("n_fallback")})
        deriv.append(d["derivatives"])
        cross.append(d["cross_check"])
        mc_info = mc_info or d["mc"]
    tds.sort(key=lambda e: e["t"])
    derivatives = {}
    for k in ("acc", "jerk"):
        derivatives[k] = {n: {"cutoff_hz": _med([w[k][n]["cutoff_hz"] for w in deriv]), "snr": _med([w[k][n]["snr"] for w in deriv]),
                              "reliable": float(np.mean([bool(w[k][n]["reliable"]) for w in deriv])) >= 0.5}
                          for n in deriv[0][k]}
    tri = [c["touchdowns_spline_vs_butter"] for c in cross]
    cc = {"butterworth": cross[0]["butterworth"], "touchdowns_spline_vs_butter": [int(sum(x[i] for x in tri)) for i in range(3)],
          "vy0_median_absdiff_ms": _med([c["vy0_median_absdiff_ms"] for c in cross]),
          "vel_rms_diff_m_s": _med([c["vel_rms_diff_m_s"] for c in cross]), "acc_rms_diff_m_s2": _med([c["acc_rms_diff_m_s2"] for c in cross]),
          "note": "medians over the windows"}
    info = {"windowed": {"core_s": CORE_S, "margin_s": MARGIN_S, "n_windows": len(wins), "windows": winfo,
                         "why": "dense smoother operators grow with the square of the take length (16 GB pod, 360 s take)"},
            "n_template": int(sum(w["n_template"] or 0 for w in winfo)), "n_fallback": int(sum(w["n_fallback"] or 0 for w in winfo))}
    return {"noise_m": noise, "noise_estimated": cfg.noise_m is None, "touchdowns": tds, "accents": accents, "derivatives": derivatives,
            "cross_check": cc, "touchdown_info": info, "mc": {**mc_info, "per_window": True}}


def _write_kinematics(ts: dict, js: dict, noise: dict, cfg: tm.TimingConfig, path: Path) -> None:
    """timing/kinematics.npz: the same arrays as timing.run writes (full 55-joint derivatives for the viewer's jerk glow), computed
    window by window (see the module docstring) and joined at the cores. cutoff_hz / snr are the medians over the windows, reliable is
    true when it was in at least half of them, noise_sd is per sample."""
    kin = {}
    for role in ts:
        t, J = ts[role], js[role]
        T, nj = len(t), J.shape[1]
        arr = {n: np.zeros((T, nj, 3), np.float32) for n in ("pos", "vel", "acc", "jerk")}
        nsd = {n: np.zeros((T, nj), np.float32) for n in ("acc", "jerk")}
        per = {n: {"cutoff_hz": [], "snr": [], "reliable": []} for n in ("acc", "jerk")}
        for i0, i1, c0, c1 in split_windows(t):
            k = tm.spline_kinematics(t[i0:i1], J[i0:i1], noise[role], k=cfg.spline_k)
            full = tm.choose_cutoffs(t[i0:i1], J[i0:i1], range(nj), noise[role], cfg)
            tm.apply_cutoffs(t[i0:i1], J[i0:i1], full, (k,))
            sl = slice(c0 - i0, c1 - i0)
            for n in arr:
                arr[n][c0:c1] = getattr(k, n)[sl].astype(np.float32)
            for n in ("acc", "jerk"):
                nsd[n][c0:c1] = np.asarray(full[n]["noise_sd"], np.float32)[sl]
                for key in ("cutoff_hz", "snr", "reliable"):
                    per[n][key].append(full[n][key])
        kin.update({f"{role}_{n}": a for n, a in arr.items()})
        kin[f"{role}_t"] = t
        kin[f"{role}_noise_m"] = np.float32(noise[role])
        for n in ("acc", "jerk"):
            kin[f"{role}_{n}_cutoff_hz"] = np.median(np.asarray(per[n]["cutoff_hz"], float), 0).astype(np.float32)
            kin[f"{role}_{n}_snr"] = np.median(np.asarray(per[n]["snr"], float), 0).astype(np.float32)
            kin[f"{role}_{n}_reliable"] = np.mean(np.asarray(per[n]["reliable"], float), 0) >= 0.5
            kin[f"{role}_{n}_noise_sd"] = nsd[n]
    np.savez_compressed(path, **kin)


def analyse(ts: dict, js: dict, cfg: tm.TimingConfig) -> dict:
    """The beat-free analysis of both dancers -> the (json-able after timing._clean) document + private '_' entries."""
    rng = np.random.default_rng(cfg.seed)
    res = {"version": VERSION, "beatless": True, "config": tm._cfg_json(cfg),
           "beats": {"grid": None, "grid_columns": None, "measure_s": None, "tempo_bpm": None, "skipped": True,
                     "reason": "no soundtrack: the recap is a spoken class, there is no beat grid to measure against",
                     "warnings": ["beat-free timing: touchdowns, accents and step intervals only; every beat / rhythm / BPM "
                                  "product is skipped (see `skipped`)"]},
           "skipped": SKIPPED, "kinematics": {}, "touchdown_method": {}, "footsteps": {}, "accents": {}, "asynchrony": {}}
    steps = {}
    for role in ROLES:
        if role not in js:
            continue
        d = analyse_windowed(ts[role], js[role], cfg, rng)
        res["kinematics"][role] = {"noise_m": d["noise_m"], "noise_estimated": d["noise_estimated"],
                                   "derivatives": d["derivatives"], "cross_check": d["cross_check"]}
        res["touchdown_method"][role] = {**d["touchdown_info"], "mc": d["mc"]}
        res["footsteps"][role] = d["touchdowns"]
        res["accents"][role] = d["accents"]
        t = ts[role]
        steps[role] = step_summary(d["touchdowns"], float(t[0]), float(t[-1]))
    if all(r in steps for r in ROLES):
        steps["couple"] = {"lead_follow_offset": couple_offsets([e["t"] for e in res["footsteps"]["lead"]],
                                                                [e["t"] for e in res["footsteps"]["follow"]])}
    res["step_timing"] = steps
    res["absolute_timing"] = {"valid": False, "av_offset_s": cfg.av_offset_s, "av_sync_event": None,
                              "note": "no beat to be on or off: only step timing in seconds and the lead / follow relative "
                                      "timing mean anything"}
    res["notes"] = [n for n in tm.NOTES if n.startswith(("Positions", "Touchdown", "sigma per event", "Accents"))]
    return res


def plot_qa(res: dict, ts: dict, path: Path) -> None:
    """Touchdown timeline per dancer and foot, steps per 10 s, interval histogram (the review image of this stage)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 2, figsize=(16, 9), gridspec_kw={"height_ratios": [2, 1.2, 1.4]})
    colors = {"lead": "#e34948", "follow": "#2a78d6"}
    for c, role in enumerate(ROLES):
        evs = res["footsteps"].get(role, [])
        a = ax[0, c]
        for y, foot in enumerate(("left", "right")):
            tt = [e["t"] for e in evs if e["foot"] == foot]
            a.vlines(tt, y + 0.1, y + 0.9, color=colors[role], lw=0.8)
        for b in res["step_timing"].get(role, {}).get("bouts", []):
            a.axvspan(b["t0"], b["t1"], color=colors[role], alpha=0.12, lw=0)
        a.set_yticks([0.5, 1.5], ["left foot", "right foot"])
        a.set_title(f"{role}: {len(evs)} touchdowns, {res['step_timing'].get(role, {}).get('n_bouts', 0)} bouts "
                    f"(shaded: steps <= {BOUT_GAP_S:g} s apart)", fontsize=10, loc="left")
        spw = res["step_timing"].get(role, {}).get("steps_per_window", [])
        a = ax[1, c]
        a.bar([(w["t0"] + w["t1"]) / 2 for w in spw], [w["n"] for w in spw], width=WINDOW_S * 0.9, color=colors[role])
        a.set_ylabel(f"steps / {WINDOW_S:g} s")
        a = ax[2, c]
        iv = intervals([e["t"] for e in evs])
        iv = iv[iv < 3.0]
        if len(iv):
            a.hist(iv, bins=np.arange(0, 3.05, 0.05), color=colors[role])
        a.set_xlabel("interval between successive touchdowns (s, < 3 s)")
        a.set_ylabel("count")
        if c == 1:
            for r in (0, 1):
                ax[r, c].set_xlabel("take time (s)")
    fig.suptitle("beat-free step timing (no beat grid: nothing here is measured against a rhythm)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(work: Path, cfg: tm.TimingConfig, smplx_dir: str = "smplx", out_dir: Path | None = None, plot: bool = True) -> dict:
    out_dir = Path(out_dir) if out_dir else work / "timing"
    ts, js = {}, {}
    for role in ROLES:
        p = work / smplx_dir / f"{role}.npz"
        if p.exists():
            ts[role], js[role] = tm.load_smplx(p)
    if not ts:
        raise FileNotFoundError(f"no {smplx_dir}/{{lead,follow}}.npz in {work}")
    t0 = time.time()
    res = analyse(ts, js, cfg)
    res["inputs"] = {"smplx": {r: str(Path(smplx_dir) / f"{r}.npz") for r in ts}, "beats": None}
    for role in ts:
        nz = res["kinematics"][role]["noise_m"]
        if nz < 0.002 and cfg.mc_corr_noise_m <= 0:
            res["beats"]["warnings"].append(f"{role}: estimated white pose noise is only {nz * 1000:.1f} mm (a smooth fused fit): "
                                            "the Monte-Carlo sigma per event is a lower bound; use --mc-corr-noise-mm")
    res["seconds"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timing.json").write_text(json.dumps(tm._clean(res), indent=1), encoding="utf-8")
    _write_kinematics(ts, js, {r: res["kinematics"][r]["noise_m"] for r in ts}, cfg, out_dir / "kinematics.npz")
    if plot:
        plot_qa(res, ts, out_dir / "qa_timing.png")
    res["_out"] = out_dir
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.timing_free", description=__doc__.split("\n\n")[0])
    ap.add_argument("take")
    ap.add_argument("--smplx-dir", default="smplx", help="folder under work/<take> (default: the raw stitched fit, as the demo)")
    ap.add_argument("--mc", type=int, default=60, help="Monte-Carlo draws for sigma per event (default 60)")
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mc-corr-noise-mm", type=float, default=10.0, help="low-frequency pose error SD of the Monte-Carlo (the demo used 10)")
    ap.add_argument("--mc-corr-time-s", type=float, default=0.1)
    ap.add_argument("--noise-mm", type=float, help="joint noise SD (default: estimated)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--work-root", help="folder holding <take>/ (default dancecap/work)")
    a = ap.parse_args(argv)
    from .take import PACKAGE_ROOT
    root = Path(a.work_root) if a.work_root else PACKAGE_ROOT / "work"
    cfg = tm.TimingConfig(mc_draws=a.mc, workers=a.workers, seed=a.seed, mc_corr_noise_m=a.mc_corr_noise_mm / 1000,
                          mc_corr_time_s=a.mc_corr_time_s, noise_m=None if a.noise_mm is None else a.noise_mm / 1000,
                          refine_beats=False)
    try:
        res = run(root / a.take, cfg, a.smplx_dir, plot=not a.no_plot)
    except FileNotFoundError as e:
        raise SystemExit(f"timing_free: {e}")
    for role in ROLES:
        s = res["step_timing"].get(role)
        if s:
            iv = s["interval_all"] or {}
            print(f"[timing_free] {role}: {s['n_touchdowns']} touchdowns, {s['n_bouts']} bouts ({s['share_in_bouts']:.0%} of the take), "
                  f"median interval {iv.get('median_s')} s")
    cp = (res["step_timing"].get("couple") or {}).get("lead_follow_offset")
    if cp:
        print(f"[timing_free] lead/follow: {cp['n_pairs']} pairs, follow minus lead median {cp['median_ms']} ms (MAD {cp['mad_ms']} ms)")
    print(f"[timing_free] SKIPPED: " + "; ".join(SKIPPED))
    print(f"[timing_free] -> {res['_out'] / 'timing.json'} ({res['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
