"""Review images for the real-take timing + physics analysis (what the user looks at).

    python -m dancecap.analysis_review larissa_kadu
        -> work/<take>/review/timing_asynchrony.png, timing_accents.png, physics_overview.png
           + work/<take>/review/analysis_summary.json (numbers quoted in the images)

Reads timing/timing.json (refined beat grid, footsteps, accents), smplx/{lead,follow}.npz (ankle heights) and
physics/physics.{npz,json}. Read-only on every stage output. Besides plotting it computes one extra relative-timing
measure that needs no touchdown detector: the vertical "pulse" (peaks of F_net,y = m (a_y + g) of each dancer's COM,
i.e. the moment of maximum upward push / bottom of the knee bounce), its phase on the eighth-note grid and the
follow-minus-lead lag (paired peaks + cross-correlation). Absolute phases share the unknown A/V offset of the
phones; the follow-minus-lead lag does not.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

COL = {"lead": "#2a78d6", "follow": "#eb6834"}  # dataviz reference palette slots 1-2 (validated pair)
NAME = {"lead": "Kadu (lead)", "follow": "Larissa (follow)"}
GRID_COL = {1: "#3d3d3a", 2: "#8a8984", 3: "#d4d3cd"}
GRID_LW = {1: 1.3, 2: 0.9, 3: 0.6}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
ACC_ROWS = [("head", "head"), ("left_wrist", "L hand"), ("right_wrist", "R hand"), ("pelvis", "hips"),
            ("left_foot", "L foot"), ("right_foot", "R foot")]
ANKLE = {"left": 7, "right": 8}


def _style(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": False, "figure.facecolor": "white", "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})


def _grid(ax, grid, lo, hi, labels=False):
    for t, ty, m, c in grid:
        if lo <= t <= hi:
            ax.axvline(t, color=GRID_COL[int(ty)], lw=GRID_LW[int(ty)], zorder=0)
            if labels:
                ax.text(t, 1.0, f"{int(c)}" if c else f"m{int(m)}", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, color=INK2 if ty != 3 else MUTED)


def _ci_t(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan"), [float("nan"), float("nan")]
    from scipy.stats import t as tdist
    m, se = float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(n))
    h = float(tdist.ppf(0.975, n - 1) * se)
    return m, [m - h, m + h]


def pulse_analysis(ph: dict, grid: np.ndarray, sync_sigma_s: float = 0.005) -> dict:
    """Vertical pulse = peaks of F_net,y / (m g) per dancer; phase on the eighth grid; follow-minus-lead lag."""
    from scipy.signal import find_peaks
    from scipy.interpolate import CubicSpline
    t = ph["t"]
    eighth = grid[:, 0]
    out = {"definition": "peak of F_net,y/(m g) (max upward push of the COM = bottom of the bounce), "
                         "physics.npz <role>_f_net (5 Hz zero-phase spline), cubic-interpolated to 1 ms",
           "roles": {}}
    tf = np.arange(t[0] + 0.15, t[-1] - 0.15, 0.001)  # skip the spline edge
    sig = {}
    for role in ("lead", "follow"):
        g = ph[f"{role}_f_net"][:, 1] / (ph[f"{role}_mass"] * 9.81)
        y = CubicSpline(t, g)(tf)
        sig[role] = y
        pk, pr = find_peaks(y, distance=250, prominence=0.12)
        tp = tf[pk]
        k = np.clip(np.searchsorted(eighth, tp), 1, len(eighth) - 1)
        near = np.where(np.abs(eighth[k] - tp) < np.abs(eighth[k - 1] - tp), k, k - 1)
        ph_ms = (tp - eighth[near]) * 1000
        m, ci = _ci_t(ph_ms)
        out["roles"][role] = {"t": tp.round(4).tolist(), "height": y[pk].round(3).tolist(),
                              "prominence": pr["prominences"].round(3).tolist(),
                              "eighth_index": near.tolist(), "slot": grid[near, 3].astype(int).tolist(),
                              "phase_ms": ph_ms.round(1).tolist(), "mean_phase_ms": m, "ci95_ms": ci,
                              "sd_ms": float(np.std(ph_ms, ddof=1)) if len(ph_ms) > 1 else None}
    # paired peaks follow - lead (same eighth)
    a, b = out["roles"]["lead"], out["roles"]["follow"]
    pairs = [(ta, tb) for ta, ia in zip(a["t"], a["eighth_index"]) for tb, ib in zip(b["t"], b["eighth_index"])
             if ia == ib]
    d = np.array([(tb - ta) * 1000 for ta, tb in pairs])
    m, ci = _ci_t(d) if len(d) > 1 else (float("nan"), [float("nan")] * 2)
    out["follow_minus_lead_paired"] = {"n": int(len(d)), "diff_ms": d.round(1).tolist(), "mean_ms": m, "ci95_ms": ci}
    # steady-bounce section, data-driven: the longest run of consecutive eighths where BOTH dancers have exactly one
    # pulse peak (a regular pulse train on every eighth)
    ia, ib = a["eighth_index"], b["eighth_index"]
    both = sorted(set(i for i in ia if ia.count(i) == 1) & set(i for i in ib if ib.count(i) == 1))
    runs, cur = [], []
    for i in both:
        if cur and i != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    run = max(runs, key=len) if runs else []
    sec = {"eighths": run, "t": [float(grid[run[0], 0] - 0.2), float(grid[run[-1], 0] + 0.2)] if run else None}
    if run:
        for role, p in (("lead", a), ("follow", b)):
            ph_r = [x for x, i in zip(p["phase_ms"], p["eighth_index"]) if i in run]
            m, ci = _ci_t(ph_r)
            sec[role] = {"n": len(ph_r), "mean_phase_ms": m, "ci95_ms": ci, "sd_ms": float(np.std(ph_r, ddof=1))}
        dd = [(tb - ta) * 1000 for ta, i0 in zip(a["t"], ia) for tb, i1 in zip(b["t"], ib) if i0 == i1 and i0 in run]
        m, ci = _ci_t(dd)
        sec["follow_minus_lead"] = {"n": len(dd), "diff_ms": np.round(dd, 1).tolist(), "mean_ms": m, "ci95_ms": ci,
                                    "sd_ms": float(np.std(dd, ddof=1))}
    out["steady_bounce"] = sec
    # cross-correlation over the steady-bounce section and over the whole window
    secs = {"window": (tf[0], tf[-1])}
    if run:
        secs["steady_bounce"] = tuple(sec["t"])
    for name, (lo, hi) in secs.items():
        s = (tf >= lo) & (tf <= hi)
        x, z = sig["lead"][s] - sig["lead"][s].mean(), sig["follow"][s] - sig["follow"][s].mean()
        lags = np.arange(-200, 201)
        cc = np.array([np.sum(x[max(0, -L):len(x) - max(0, L)] * z[max(0, L):len(z) - max(0, -L)])
                       for L in lags]) / (np.std(x) * np.std(z) * len(x))
        j = int(np.argmax(cc))
        out[f"xcorr_{name}"] = {"lag_ms_follow_after_lead": int(lags[j]), "r": float(cc[j]),
                                "r_at_0": float(cc[200])}
    out["note"] = ("pulse phases are on the reference AUDIO clock without an A/V sync event: the common phone "
                   "audio/video offset (unknown, typically tens of ms) shifts both dancers equally. follow-minus-"
                   "lead cancels it; residual camera sync (~%.0f ms per camera, averaged over 8) remains." %
                   (sync_sigma_s * 1000))
    return out


def whole_body_coincident(tm, tol=0.035, min_others=4) -> dict:
    """Per event: True when >= min_others of the OTHER accent joints of the same dancer and kind peak within +-tol.
    Head, both hands, hips and both feet peaking together is a rigid whole-body (root) acceleration - in a fused fit
    usually root-translation jitter - not a body-part accent."""
    out = {}
    for role, joints in tm["accents"].items():
        out[role] = {}
        for kind in ("acc", "jerk"):
            ts = {j: np.array([e["t"] for e in d.get(kind, [])]) for j, d in joints.items()}
            out[role][kind] = {j: [sum(1 for o, to in ts.items() if o != j and len(to) and np.min(np.abs(to - x)) <= tol)
                                   >= min_others for x in tt] for j, tt in ts.items()}
    return out


def root_jitter(sm: dict, lo_hz=4.5, hi_hz=12.0, win_s=0.5) -> dict:
    """Band-passed (4.5-12 Hz) pelvis displacement per dancer, overall and per window RMS (mm): fit jitter well above
    the bounce/step rhythm (2.5 Hz eighths) that the 4th-difference noise estimate does not see."""
    from scipy.signal import butter, filtfilt
    out = {}
    for role, z in sm.items():
        t = z["t"]
        fs = 1.0 / float(np.median(np.diff(t)))
        b, a = butter(4, [lo_hz / (fs / 2), hi_hz / (fs / 2)], "band")
        w = np.linalg.norm(filtfilt(b, a, z["joints"][:, 0], axis=0), axis=1) * 1000
        st = np.arange(t[0], t[-1] - win_s / 2, win_s)
        out[role] = {"rms_mm": float(np.sqrt(np.mean(w ** 2))), "win_t0": st.round(3).tolist(),
                     "win_rms_mm": [float(np.sqrt(np.mean(w[(t >= s) & (t < s + win_s)] ** 2))) for s in st],
                     "t": t, "mag_mm": w}
    return out


def load(take: str):
    work = PACKAGE_ROOT / "work" / take
    tm = json.loads((work / "timing" / "timing.json").read_text())
    pj = json.loads((work / "physics" / "physics.json").read_text())
    ph = dict(np.load(work / "physics" / "physics.npz", allow_pickle=True))
    sm = {r: dict(np.load(work / "smplx" / f"{r}.npz", allow_pickle=True)) for r in ("lead", "follow")}
    grid = np.array(tm["beats"]["grid"], float)
    return work, tm, pj, ph, sm, grid


def fig_asynchrony(tm, sm, grid, out: Path, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1.25, 1.1], width_ratios=[1, 1, 0.9], hspace=0.55, wspace=0.28)
    tol = tm["config"].get("hit_tol_s", 0.05) * 1000
    # A/B: ankle heights + touchdowns
    for row, role in enumerate(("lead", "follow")):
        ax = fig.add_subplot(gs[row, :])
        t, J = sm[role]["t"], sm[role]["joints"]
        _grid(ax, grid, lo, hi, labels=row == 0)
        for foot, ls in (("left", "-"), ("right", "--")):
            ax.plot(t, J[:, ANKLE[foot], 1], ls, color=COL[role], lw=1.6, label=f"{foot} ankle height")
        for e in tm["footsteps"][role]:
            y = np.interp(e["t"], t, J[:, ANKLE[e["foot"]], 1])
            mk = "v" if e["foot"] == "left" else "s"
            ax.errorbar(e["t"], y, xerr=1.96 * e["sigma"], fmt=mk, ms=8, color=COL[role], mec="white", mew=1.2,
                        ecolor=INK2, elinewidth=1, capsize=3, zorder=5)
            ax.annotate("", xy=(e["beat_t"], ax.get_ylim()[0] if False else y + 0.035), xytext=(e["t"], y + 0.035),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
            ax.text(e["t"], y + 0.045, f"{e['async_ms']:+.0f}", ha="center", fontsize=8, color=INK)
        ax.set_xlim(lo, hi)
        ax.set_ylabel("ankle height (m)")
        tdm = tm["touchdown_method"][role]
        ax.set_title(f"{'A' if row == 0 else 'B'}  {NAME[role]}: ankle heights, touchdowns (v left / s right, "
                     f"bar = 95% MC), arrow to the assigned grid slot, label = asynchrony ms   "
                     f"[{tdm['n_template']} template + {tdm['n_fallback']} hinge-only touchdowns]", pad=16)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    # C: asynchrony vs time
    ax = fig.add_subplot(gs[2, :2])
    _grid(ax, grid, lo, hi)
    ax.axhspan(-tol, tol, color="#ecebe6", zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    for k, role in enumerate(("lead", "follow")):
        ev = tm["footsteps"][role]
        tt = np.array([e["t"] for e in ev])
        aa = np.array([e["async_ms"] for e in ev])
        ss = np.array([e["sigma"] * 1000 for e in ev])
        ax.errorbar(tt, aa, yerr=1.96 * ss, fmt="o", ms=7, color=COL[role], mec="white", ecolor=COL[role],
                    elinewidth=1.2, capsize=3, label=f"{NAME[role]} step", zorder=4)
        st = tm["asynchrony"][role]["footsteps"]
        if st.get("n"):
            for pm in st.get("per_measure", []):
                t0 = pm["t0"]
                t1 = t0 + tm["beats"]["measure_s"]
                ax.fill_between([max(t0, lo), min(t1, hi)], *pm["ci95_ms"], color=COL[role], alpha=0.07, lw=0)
                ax.plot([max(t0, lo), min(t1, hi)], [pm["mean_ms"]] * 2, color=COL[role], lw=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-260, 260)
    ax.set_ylabel("step - beat (ms)")
    ax.set_xlabel("reference time (s)")
    ax.set_title(f"C  asynchrony per step (bar = 95% MC); per-measure mean + 95% CI; grey = +-{tol:.0f} ms hit")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    # D: asynchrony vs count position
    ax = fig.add_subplot(gs[2, 2])
    ax.axhspan(-tol, tol, color="#ecebe6", zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    for k, role in enumerate(("lead", "follow")):
        ev = tm["footsteps"][role]
        c = np.array([e["count"] for e in ev]) + (k - 0.5) * 0.18
        ax.errorbar(c, [e["async_ms"] for e in ev], yerr=[1.96 * e["sigma"] * 1000 for e in ev], fmt="o", ms=7,
                    color=COL[role], mec="white", ecolor=COL[role], capsize=2, label=NAME[role])
    ax.set_xticks(range(8))
    ax.set_xticklabels(["0\n1", "1\n&", "2\n2", "3\n&", "4\n3", "5\n&", "6\n4", "7\n&"])
    for x in (0, 4):
        ax.axvline(x, color=GRID_COL[1], lw=GRID_LW[1], zorder=0)
    for x in (3, 6):
        ax.axvline(x, color=GRID_COL[2], lw=GRID_LW[2], zorder=0)
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(-260, 260)
    ax.set_xlabel("zouk eighth slot (top) / beat count (bottom)")
    ax.set_title("D  asynchrony vs position in the measure")
    # E: summary text
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    lines = []
    for role in ("lead", "follow"):
        f = tm["asynchrony"][role]["footsteps"]
        if not f.get("n"):
            lines.append(f"{NAME[role]}: no touchdowns")
            continue
        lines.append(f"{NAME[role]}: {f['n']} touchdowns, mean {f['mean_ms']:+.0f} ms, 95% CI "
                     f"[{f['ci95_ms'][0]:+.0f}, {f['ci95_ms'][1]:+.0f}] ms, sd {np.std([e['async_ms'] for e in tm['footsteps'][role]], ddof=1):.0f} ms, "
                     f"phase-locking R = {f['R']:.2f} (Rayleigh p = {f['rayleigh_p']:.2f}), hit rate {f['hit_rate']:.0%} "
                     f"vs chance {f['surrogate']['hit_rate_mean']:.0%} (p = {f['surrogate']['hit_rate_p']:.2f})")
    fl = tm["asynchrony"]["follow_minus_lead"]
    lines.append(f"Follow - lead (relative, A/V offset cancels): {fl['overall_ms']:+.0f} ms, 95% CI "
                 f"[{fl['overall_ci95_ms'][0]:+.0f}, {fl['overall_ci95_ms'][1]:+.0f}] ms")
    mc = tm["touchdown_method"]["lead"]["mc"]
    lines.append(f"Uncertainty: MC {mc['draws']} draws with white noise (estimated 0.5-0.8 mm) + an ASSUMED low-frequency "
                 f"pose error {mc['corr_noise_m'] * 100:.0f} cm / {mc['corr_time_s']} s, beat jitter "
                 f"{tm['beats']['jitter_used_ms']:.0f} ms, camera sync {tm['absolute_timing']['sync_sigma_s'] * 1000:.0f} ms. "
                 f"Window {sm['lead']['t'][0]:.2f}-{sm['lead']['t'][-1]:.2f} s = {(sm['lead']['t'][-1] - sm['lead']['t'][0]) / tm['beats']['measure_s']:.1f} measures "
                 f"of a {tm['beats']['tempo_bpm']:.1f} BPM grid.")
    lines.append("ABSOLUTE timing is NOT established: no audio-visual sync event (clap/stomp) pins the phone A/V offset, so "
                 "'on/behind the beat' is relative to the audio clock only. Steps are NOT phase-locked to the grid here.")
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=10, color=INK, family="monospace", wrap=True)
    fig.suptitle("larissa_kadu 56-62 s: footstep timing vs the zouk beat grid (dark line = beats 1/3, mid = slots 3/6, "
                 "light = other eighths)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_accents(tm, ph, grid, pulse, jit, out: Path, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 0.45, 1.1, 0.75], width_ratios=[2.3, 1], hspace=0.45, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    axf = fig.add_subplot(gs[0, 1], sharey=ax)
    _grid(ax, grid, lo, hi, labels=True)
    yt, yl = [], []
    y = 0
    rel = tm["kinematics"]
    coinc = whole_body_coincident(tm)
    for role in ("lead", "follow"):
        for key, lab in ACC_ROWS:
            for kind, dy, mk in (("acc", 0.17, "o"), ("jerk", -0.17, "^")):
                evs = tm["accents"][role].get(key, {}).get(kind, [])
                ok = rel[role]["derivatives"][kind][key]["reliable"]
                for i, e in enumerate(evs):
                    good = ok and e.get("reliable", True)
                    if coinc[role][kind][key][i]:
                        ax.plot(e["t"], y + dy, mk, ms=12, mfc="none", mec=INK, mew=1.0)
                    ax.plot(e["t"], y + dy, mk, ms=7, mfc=COL[role] if good else "white",
                            mec=COL[role] if good else MUTED, mew=1.2)
                st = tm["asynchrony"][role]["accents"].get(key, {}).get(kind)
                if st and st.get("n"):
                    axf.errorbar(st["mean_ms"], y + dy, xerr=[[st["mean_ms"] - st["ci95_ms"][0]],
                                                              [st["ci95_ms"][1] - st["mean_ms"]]],
                                 fmt=mk, ms=6, color=COL[role] if ok else MUTED, mfc=COL[role] if ok else "white",
                                 capsize=2, lw=1)
                    axf.text(265, y + dy, f"R {st['R']:.2f} hit {st['hit_rate']:.0%}"
                             + ("" if ok else "  unreliable"), va="center", fontsize=7, color=INK2)
            yt.append(y)
            yl.append(f"{'K' if role == 'lead' else 'L'} {lab}  (snr {rel[role]['derivatives']['acc'][key]['snr']:.1f}/"
                      f"{rel[role]['derivatives']['jerk'][key]['snr']:.1f})")
            y += 1
        y += 0.6
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(lo, hi)
    ax.set_xlabel("reference time (s)")
    ax.set_title("A  accents (K Kadu, L Larissa): o |acc| peak, ^ |jerk| peak; filled = snr >= 2, hollow = unreliable;\n"
                 "black ring = whole-body coincident (>= 4 other joints within 35 ms)", pad=18, fontsize=10)
    axf.axvspan(-50, 50, color="#ecebe6", zorder=0)
    axf.axvline(0, color=MUTED, lw=0.8)
    axf.set_xlim(-200, 400)
    axf.set_xticks([-200, -100, 0, 100, 200])
    axf.tick_params(labelleft=False)
    axf.set_xlabel("mean accent - beat (ms), 95% CI")
    axf.set_title("B  mean accent - beat", pad=18, fontsize=10)
    # jitter strip
    axj = fig.add_subplot(gs[1, 0], sharex=ax)
    for role in ("lead", "follow"):
        axj.plot(jit[role]["t"], jit[role]["mag_mm"], color=COL[role], lw=1.3,
                 label=f"{NAME[role]} (rms {jit[role]['rms_mm']:.1f} mm)")
    axj.set_ylabel("mm")
    axj.set_title("pelvis 4.5-12 Hz displacement = fit jitter (identical on every joint -> rigid root motion)",
                  fontsize=9)
    axj.legend(loc="upper left", fontsize=8, ncol=2)
    axj.set_xlim(lo, hi)
    # C: vertical pulse
    ax = fig.add_subplot(gs[2, :])
    _grid(ax, grid, lo, hi)
    t = ph["t"]
    for role in ("lead", "follow"):
        g = ph[f"{role}_f_net"][:, 1] / (ph[f"{role}_mass"] * 9.81)
        b = ph[f"{role}_f_net_band"][:, 1] / (ph[f"{role}_mass"] * 9.81)
        ax.fill_between(t, g - b, g + b, color=COL[role], alpha=0.18, lw=0)
        ax.plot(t, g, color=COL[role], lw=1.8, label=f"{NAME[role]} F_net,y / mg")
        p = pulse["roles"][role]
        ax.plot(p["t"], p["height"], "o", color=COL[role], mec="white", ms=7)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylabel("vertical push / body weight")
    ax.set_xlabel("reference time (s)")
    sb = pulse["steady_bounce"]
    if sb.get("t"):
        ax.axvspan(*sb["t"], color="#1baf7a", alpha=0.07, lw=0, zorder=0)
        a, b, fl = sb["lead"], sb["follow"], sb["follow_minus_lead"]
        xc = pulse["xcorr_steady_bounce"]
        ax.set_title(f"C  vertical pulse (dots = max upward push = bottom of the bounce). Steady bounce {sb['t'][0]:.1f}-"
                     f"{sb['t'][1]:.1f} s (green, {a['n']} eighths): peak - eighth = Kadu {a['mean_phase_ms']:+.0f} ms "
                     f"[{a['ci95_ms'][0]:+.0f}, {a['ci95_ms'][1]:+.0f}], Larissa {b['mean_phase_ms']:+.0f} ms "
                     f"[{b['ci95_ms'][0]:+.0f}, {b['ci95_ms'][1]:+.0f}];\n"
                     f"Larissa - Kadu {fl['mean_ms']:+.0f} ms "
                     f"[{fl['ci95_ms'][0]:+.0f}, {fl['ci95_ms'][1]:+.0f}] (paired, n={fl['n']}), cross-correlation lag "
                     f"{xc['lag_ms_follow_after_lead']:+d} ms (r={xc['r']:.2f}). After it the pulse breaks up (turn).",
                     fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    nco = {r: (sum(sum(v) for k in ("acc", "jerk") for v in c[k].values()),
               sum(len(v) for k in ("acc", "jerk") for v in c[k].values())) for r, c in coinc.items()}
    ax.text(0, 1.3, f"Whole-body coincident events (>= 4 other joints within 35 ms): Kadu {nco['lead'][0]}/{nco['lead'][1]}, "
                    f"Larissa {nco['follow'][0]}/{nco['follow'][1]}. Larissa's fit carries "
                    f"{jit['follow']['rms_mm'] / jit['lead']['rms_mm']:.1f}x Kadu's 4.5-12 Hz root jitter: many of her "
                    "'accents' are fit jitter, not limb accents.\nKadu's coincident jerk peaks (58.45, 58.72, 59.03, 60.43, 60.59 s) "
                    "sit in the turn; only 59.03 s coincides with Larissa (a coupled move or a shared camera glitch, "
                    "not resolvable here).", va="top", fontsize=10, color=INK, fontweight="bold")
    ax.text(0, 0.45, "Accents are |a| / |jerk| magnitude peaks of single joints (any direction, two per step or swing), so they fall "
                  "at all phases: no joint is phase-locked\n(all Rayleigh p > 0.3, hit rates at chance ~25%). The reliability flag "
                  "compares derivative signal to the ESTIMATED white noise (0.5-0.8 mm) - the real fused-fit error is cm-level\nand "
                  "low-frequency, so 'reliable' is optimistic, and jerk of a 30 Hz fit is mostly smoothing-shaped. The clearest rhythm "
                  "is the whole-body vertical pulse (C): both dancers bounce on\nevery eighth from 56.1 to ~59.1 s, then the pattern "
                  "breaks into a turn. Absolute phase vs the music still carries the unknown phone A/V offset; the follow - lead\n"
                  "lag does not.", va="top", fontsize=10, color=INK)
    fig.suptitle("larissa_kadu 56-62 s: body-part accents vs the zouk beat grid", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_physics(pj, ph, grid, out: Path, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    t = ph["t"]
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.25, 1.75], height_ratios=[1, 1, 1, 0.5], hspace=0.5, wspace=0.16)
    # A: COM paths top-down
    ax = fig.add_subplot(gs[:3, 0])
    beat_t = grid[(grid[:, 0] >= t[0]) & (grid[:, 0] <= t[-1])]
    for role in ("lead", "follow"):
        c = ph[f"{role}_com"]
        xc = ph[f"{role}_xcom"]
        ax.plot(c[:, 0], c[:, 2], color=COL[role], lw=2, label=f"{NAME[role]} COM")
        ax.plot(xc[:, 0], xc[:, 1], ":", color=COL[role], lw=1, label=f"{NAME[role]} XCoM")
        ax.plot(*c[0, [0, 2]], "o", color=COL[role], ms=9, mec="white")
        ax.plot(*c[-1, [0, 2]], "X", color=COL[role], ms=10, mec="white")
        for bt, ty, m, cnt in beat_t:
            if ty == 1:
                p = [np.interp(bt, t, c[:, k]) for k in (0, 2)]
                ax.plot(*p, "s", color=COL[role], ms=6, mec=INK, mew=0.8)
                ax.annotate(f"{bt:.1f}s", p, xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK2)
    for bt, ty, m, cnt in beat_t:
        if ty == 1:
            pa = [np.interp(bt, t, ph["lead_com"][:, k]) for k in (0, 2)]
            pb = [np.interp(bt, t, ph["follow_com"][:, k]) for k in (0, 2)]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=MUTED, lw=0.8, zorder=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world z (m)  (top view, looking down)")
    ax.set_title("A  COM top view: o start, X end,\n    squares + grey link = beats 1 and 3")
    ax.legend(loc="best", fontsize=8)
    # B: couple GRF
    ax = fig.add_subplot(gs[0, 1])
    _grid(ax, grid, lo, hi, labels=True)
    mg = pj["couple"]["mg_n"]
    g = ph["grf_couple"][:, 1] / mg
    b = ph["grf_couple_band"][:, 1] / mg
    ax.fill_between(t, g - b, g + b, color="#3d3d3a", alpha=0.18, lw=0)
    ax.plot(t, g, color=INK, lw=1.6, label="couple vertical GRF / (mA+mB) g")
    for i, (t0, t1) in enumerate(zip(ph["beats_t0"], ph["beats_t1"])):
        mf = ph["beats_mean_force_couple"][i, 1] / mg
        ax.plot([t0, t1], [mf, mf], color="#1baf7a", lw=2.5, label="beat-interval mean (impulse / dt)" if i == 0 else None)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylabel("x body weight")
    ax.set_title(f"B  couple vertical ground reaction (95% band, excl. mass scale +-10%); masses "
                 f"{pj['dancers']['lead']['mass_kg']:.0f} + {pj['dancers']['follow']['mass_kg']:.0f} kg (volume-based)",
                 pad=16)
    ax.legend(loc="upper right", fontsize=8)
    # C: beat impulses per dancer
    ax = fig.add_subplot(gs[1, 1], sharex=ax)
    _grid(ax, grid, lo, hi)
    w = None
    for k, role in enumerate(("lead", "follow")):
        m = ph[f"{role}_mass"] * 9.81
        for i, (t0, t1) in enumerate(zip(ph["beats_t0"], ph["beats_t1"])):
            dt = t1 - t0
            ex = (ph[f"beats_impulse_{role}"][i, 1] - m * dt) / (m * dt) * 100  # excess vertical impulse, % of weight
            band = ph[f"beats_impulse_{role}_band"][i, 1] / (m * dt) * 100
            x0 = t0 + dt * (0.08 + 0.44 * k)
            ax.bar(x0, ex, width=dt * 0.4, align="edge", color=COL[role], label=NAME[role] if i == 0 else None)
            ax.errorbar(x0 + dt * 0.2, ex, yerr=band, color=INK2, capsize=2, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("excess vertical impulse\n(% of weight x dt)")
    ax.set_title("C  beat impulses: per eighth interval, vertical impulse minus body weight x dt (+ = net push up, "
                 "- = net drop); bar = 95%")
    ax.legend(loc="upper right", fontsize=8)
    # D: balance margin + contacts
    ax = fig.add_subplot(gs[2, 1], sharex=ax)
    _grid(ax, grid, lo, hi)
    for k, role in enumerate(("lead", "follow")):
        mrg = ph[f"{role}_margin_xcom"] * 100
        ax.plot(t, mrg, color=COL[role], lw=1.6, label=f"{NAME[role]} XCoM margin")
        cont = ph[f"{role}_contact"]
        for f, foot in enumerate(("L", "R")):
            yb = -62 - 7 * (2 * k + f)
            on = cont[:, f] > 0.5
            ax.fill_between(t, yb, yb + 5, where=on, color=COL[role], alpha=0.85 if f == 0 else 0.45, lw=0, step="mid")
            ax.text(lo + 0.02, yb + 2.5, f"{'K' if role == 'lead' else 'L'} {foot}", fontsize=7, va="center", color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-92, 25)
    ax.set_xlim(lo, hi)
    ax.set_ylabel("cm (+ = inside support)")
    ax.set_xlabel("reference time (s)")
    ax.set_title("D  balance margin: extrapolated COM vs the support polygon (alone; partner support ignored) + "
                 "foot contact rasters")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    # E: notes
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    info = {r: pj["dancers"][r]["info"] for r in ("lead", "follow")}
    st = pj["partner"]
    sts = np.array(ph["partner_status"])
    lines = [
        f"Masses unknown: closed-mesh volume x 1010 kg/m^3 -> Kadu {info['lead']['mass_kg']:.0f} kg, Larissa "
        f"{info['follow']['mass_kg']:.0f} kg (+-10% stated). The fitted meshes are {100 * info['lead']['stature_rel_diff']:+.1f}% / "
        f"{100 * info['follow']['stature_rel_diff']:+.1f}% taller than the given statures (1.727 / 1.651 m): if the world scale is "
        f"that much too large,\nmasses are ~{100 * ((1 + info['lead']['stature_rel_diff']) ** 3 - 1):.0f}% too high "
        f"(~{info['lead']['mass_kg'] / (1 + info['lead']['stature_rel_diff']) ** 3:.0f} / "
        f"{info['follow']['mass_kg'] / (1 + info['follow']['stature_rel_diff']) ** 3:.0f} kg) and accelerations ~6% high. Forces "
        "normalised by body weight (B, C) are nearly unaffected.",
        f"Partner force is not identifiable from kinematics alone: identifiable {100 * st['identifiable_fraction']:.0f}% of frames, "
        f"inconsistent {100 * np.mean(sts == 3):.0f}% (LP), so no partner-force curve is shown. "
        f"Balance margins < 0 are expected while dancing coupled (each leans on the partner) and during steps.",
    ]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=10, color=INK)
    fig.suptitle("larissa_kadu 56-62 s: L0 centroidal physics (world Y-up, metres; MAMMA c8 fit)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(take: str = "larissa_kadu") -> dict:
    work, tm, pj, ph, sm, grid = load(take)
    t = sm["lead"]["t"]
    lo, hi = float(t[0]) - 0.15, float(t[-1]) + 0.15
    rev = work / "review"
    rev.mkdir(parents=True, exist_ok=True)
    pulse = pulse_analysis(ph, grid, tm["absolute_timing"]["sync_sigma_s"])
    fig_asynchrony(tm, sm, grid, rev / "timing_asynchrony.png", lo, hi)
    jit = root_jitter(sm)
    fig_accents(tm, ph, grid, pulse, jit, rev / "timing_accents.png", lo, hi)
    fig_physics(pj, ph, grid, rev / "physics_overview.png", lo, hi)
    summ = {"take": take, "window_s": [float(t[0]), float(t[-1])], "pulse": pulse,
            "footsteps": {r: {k: tm["asynchrony"][r]["footsteps"].get(k) for k in
                              ("n", "mean_ms", "ci95_ms", "R", "rayleigh_p", "hit_rate")} for r in ("lead", "follow")},
            "follow_minus_lead_steps": tm["asynchrony"]["follow_minus_lead"],
            "absolute_timing": tm["absolute_timing"],
            "root_jitter_4p5_12hz": {r: {k: v for k, v in d.items() if k in ("rms_mm", "win_t0", "win_rms_mm")}
                                     for r, d in jit.items()},
            "whole_body_coincident": {r: {k: {j: int(sum(v)) for j, v in c[k].items()} for k in c}
                                      for r, c in whole_body_coincident(tm).items()}}
    (rev / "analysis_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    return summ


if __name__ == "__main__":
    s = main(*(sys.argv[1:2] or ["larissa_kadu"]))
    print(json.dumps(s["pulse"], indent=1, default=float)[:3000])
