"""Review image for the move features and boundary proposals: work/<take>/review/moves_features.png.

One shared time axis (reference seconds). Every panel has one unit (no dual axes); at most three series per panel in
the fixed categorical order (blue, orange, aqua) with a legend. Background on every panel: the beat grid (eighths
faint, half-measures light, measures dark), counterbalance intervals (yellow), masked eighths (grey hatching:
tracking dropout / low fit quality - no proposals there) and the proposed boundaries (dashed; weak ones lighter).
Top strip: the proposal segments filled by their energy band (one-hue ordinal ramp) - with the default energy basis
the band is a RANK WITHIN THIS TAKE, not an absolute level, and the strip says so.
Novelty panel: the boundary strength per eighth boundary; a proposed boundary is drawn at its snapped half-measure
(dashed line) and its novelty peak is marked separately (filled dot, labelled with the peak strength); peaks above
the threshold that were NOT proposed are hollow circles with the reason (edge / gap / spacing / merged).
Proposals are drawn only when proposals.json was computed from the current features.npz (sha check); stale inputs
(features.staleness) are printed at the top.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import counterbalance_path, moves_dir, work_dir  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")          # categorical slots 1-3 (validated all-pairs)
CB_YELLOW = "#eda100"
BAND_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")   # ordinal blue, bands 0..4
MASK_GREY = "#b5b3ab"
STALE_RED = "#c0392b"
REASON_SHORT = {"edge": "edge", "no snap target": "no snap", "merged": "merged", "spacing": "spacing"}

PANELS = [
    ("speed", "m/s", [("couple_speed", "couple centre"), ("lead_com_speed", "lead COM"),
                      ("follow_com_speed", "follow COM")]),
    ("kinetic energy", "J/kg", [("couple_ke", "couple"), ("lead_ke", "lead"), ("follow_ke", "follow")]),
    ("yaw rate", "deg/s", [("couple_yaw_rate", "couple frame"), ("axis_yaw_rate", "lead->follow axis"),
                           ("follow_yaw_rel", "follow rel. couple")]),
    ("distance", "m", [("com_distance", "COM-COM"), ("hand_hand_min", "hand-hand min"),
                       ("lead_hand_on_follow", "lead hand-follow torso")]),
    ("follower angles", "deg", [("follow_head_off_axis", "head off axis"), ("follow_trunk_lean", "trunk lean"),
                                ("follow_torsion", "torsion")]),
    ("height vs standing / foot", "m", [("lead_com_height", "lead COM"), ("follow_com_height", "follow COM"),
                                        ("follow_foot_height", "follow free foot")]),
    ("counterbalance", "0..1", [("cb_score", "counterbalance tier"), ("cb_lean", "lean"),
                                ("cb_connection", "connection")]),
]


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.titlecolor": INK, "text.color": INK, "legend.frameon": False,
                         "hatch.color": MASK_GREY, "hatch.linewidth": 0.6})


def _grid(ax, fs, t0, t1):
    g, ph = np.asarray(fs.grid_t, float), int(fs.phase0)
    for k, x in enumerate(g):
        if not t0 - 1e-6 <= x <= t1 + 1e-6:
            continue
        slot = (k - ph) % 8
        if slot == 0:
            ax.axvline(x, color="#b5b3ab", lw=0.8, zorder=0.5)
        elif slot == 4:
            ax.axvline(x, color=GRID, lw=0.7, zorder=0.5)
        else:
            ax.axvline(x, color="#efeee9", lw=0.5, zorder=0.4)


def _cb(ax, cb_doc, t0, t1):
    for iv in (cb_doc or {}).get("intervals", []):
        a, b = max(float(iv["t0"]), t0), min(float(iv["t1"]), t1)
        if b > a:
            ax.axvspan(a, b, color=CB_YELLOW, alpha=0.16, lw=0, zorder=0.3)


def _masked(ax, props):
    for m in ((props or {}).get("analysis") or {}).get("masked", []):
        ax.axvspan(m["t0"], m["t1"], facecolor="none", edgecolor=MASK_GREY, hatch="////", lw=0, zorder=0.35)


def _bounds(ax, props):
    for b in (props or {}).get("boundaries", []):
        if b.get("kind") == "gap":
            continue
        weak = bool(b.get("weak"))
        ax.axvline(b["t"], color=MUTED if weak else INK, lw=0.9 if weak else 1.0, ls=(0, (4, 3)), zorder=3)


def proposals_current(props: dict | None, npz_path: Path) -> bool:
    from .features import _sha16
    return bool(props) and (props.get("source") or {}).get("features_sha256_16") == _sha16(npz_path)


def draw_features(take: str, out: Path | None = None, fs=None, props: dict | None = None, cb_doc: dict | None = None,
                  npz_path: Path | None = None) -> Path:
    from . import features as F
    _style()
    default_npz = F.features_paths(take)[0]
    npz_path = Path(npz_path) if npz_path else default_npz
    fs = fs or F.load(npz_path)
    if props is None:
        pp = moves_dir(take) / "proposals.json"
        props = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else None
        if props is not None and not proposals_current(props, npz_path):
            props = None
    if cb_doc is None and counterbalance_path(take).exists():
        cb_doc = json.loads(counterbalance_path(take).read_text(encoding="utf-8"))
    stale = F.staleness(take, None, npz_path) if npz_path.resolve() == default_npz.resolve() else []
    t = np.asarray(fs.t, float)
    t0, t1 = float(t[0]), float(t[-1])
    n = len(PANELS) + 2
    heights = [0.55] + [1.25] * len(PANELS) + [1.25]
    fig, axes = plt.subplots(n, 1, figsize=(14, sum(heights) + 1.6), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.32})
    # top strip: proposal segments by energy band
    ax = axes[0]
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    rank_basis = not props or (props.get("method") or {}).get("energy", "rank") == "rank"
    if props:
        _masked(ax, props)
        for s in props["segments"]:
            c = BAND_RAMP[min(int(s["energy_band"]), len(BAND_RAMP) - 1)]
            ax.add_patch(plt.Rectangle((s["t0"], 0.08), s["t1"] - s["t0"], 0.84, color=c, lw=0, zorder=2))
            ax.add_patch(plt.Rectangle((s["t0"], 0.08), s["t1"] - s["t0"], 0.84, fill=False, ec=SURFACE, lw=2,
                                       zorder=3))
            txt = f"{s['energy_label']} | c{s['cluster']}"
            ax.text(0.5 * (s["t0"] + s["t1"]), 0.5, txt, ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if int(s["energy_band"]) >= 2 else INK, zorder=4, clip_on=True)
        ax.set_ylabel("proposals:\nenergy rank\nwithin this take" if rank_basis else "proposals:\nenergy band\n(absolute)",
                      rotation=0, ha="right", va="center", fontsize=7.5)
    else:
        ax.text(0.5, 0.5, "no current proposals.json (run: python -m dancecap.moves proposals " + take + ")",
                transform=ax.transAxes, ha="center", va="center", color=MUTED)
    _bounds(ax, props)
    for ax, (title, unit, series) in zip(axes[1:-1], PANELS):
        _grid(ax, fs, t0, t1)
        _cb(ax, cb_doc, t0, t1)
        _masked(ax, props)
        _bounds(ax, props)
        any_line = False
        for (name, label), col in zip(series, SERIES):
            y = np.asarray(fs.col(name), float)
            if np.isfinite(y).any():
                ax.plot(t, y, color=col, lw=1.6, label=label, zorder=2)
                any_line = True
        if not any_line:
            ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        ax.set_ylabel(unit, fontsize=8)
        ax.set_title(title, loc="left", fontsize=9, pad=2)
        if any_line:
            ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.02), fontsize=7.5, handlelength=1.6)
        ax.grid(axis="y", color=GRID, lw=0.5)
    # novelty
    ax = axes[-1]
    _grid(ax, fs, t0, t1)
    _cb(ax, cb_doc, t0, t1)
    _masked(ax, props)
    _bounds(ax, props)
    notes = []
    if props:
        nt = np.asarray(props["novelty"]["t"], float)
        ns = np.asarray([np.nan if x is None else x for x in props["novelty"]["strength"]], float)
        ax.plot(nt, ns, color=SERIES[0], lw=1.6, marker="o", ms=3, label="boundary strength\n(max over groups)",
                zorder=2)
        thr = props["method"]["threshold"]
        ax.axhline(thr, color=INK2, lw=0.9, ls=":", zorder=1)
        ax.text(t1, thr, f" threshold {thr:g}", va="bottom", ha="right", fontsize=7.5, color=INK2)
        strong = props["method"].get("strong")
        if strong is not None:
            ax.axhline(strong, color=GRID, lw=0.9, ls=":", zorder=1)
            ax.text(t1, strong, f" weak below {strong:g}", va="bottom", ha="right", fontsize=7.5, color=MUTED)
        nf = props["novelty"].get("noise_floor")
        if nf is not None:
            ax.axhline(nf, color=GRID, lw=1.2, zorder=1)
            ax.text(t0, nf, f" noise floor (median) {nf:.2f}", va="bottom", ha="left", fontsize=7, color=MUTED)
        for b in props["boundaries"]:
            if b.get("kind") == "gap":
                continue
            pk = b.get("peak_t", b["t"])
            col = MUTED if b.get("weak") else INK
            ax.plot([pk], [b["strength"]], "o", color=col, ms=5.5, zorder=4)
            if abs(pk - b["t"]) > 1e-6:
                ax.annotate("", xy=(b["t"], b["strength"]), xytext=(pk, b["strength"]),
                            arrowprops={"arrowstyle": "->", "color": col, "lw": 0.8}, zorder=4)
            ax.annotate(f"{b['t']:.2f} s{' (weak)' if b.get('weak') else ''}\npeak {pk:.2f} s: {b['strength']:.2f} "
                        f"{b.get('group') or ''}", (pk, b["strength"]), xytext=(5, 3), textcoords="offset points",
                        fontsize=7.5, color=col)
        rej = props.get("rejected_peaks") or []
        for r in rej:
            ax.plot([r["t"]], [r["strength"]], "o", mfc="none", mec=INK2, mew=1.1, ms=7, zorder=4)
            ax.annotate(f"not proposed: {REASON_SHORT.get(r['reason'], r['reason'])}", (r["t"], r["strength"]),
                        xytext=(5, -10), textcoords="offset points", fontsize=7, color=INK2)
        if rej:
            notes.append(f"hollow = peak >= threshold not proposed (edge = within {props['method']['min_eighths']} "
                         "eighths of the analysed range's ends or a gap; spacing / merged = a stronger boundary "
                         "nearby)")
        top = np.nanmax(ns) if np.isfinite(ns).any() else 0.0
        ax.set_ylim(-0.02, max(1.0, float(top) + 0.15))
        ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.02), fontsize=7.5)
        ev = props.get("evaluation")
        if ev is not None:
            for tr in ev["truth"]:
                ax.axvspan(tr["lo"], tr["hi"], ymin=0, ymax=0.08, color=SERIES[1], lw=0, zorder=2)
            notes.append(f"truth (synthetic): {len(ev['hits'])}/{len(ev['truth'])} transitions found, "
                         f"{len(ev['false_positives'])} false positive(s); orange ticks = true transition spans")
        masked = (props.get("analysis") or {}).get("masked") or []
        if masked:
            notes.append("hatched = masked eighths (" + "; ".join(f"{m['t0']:.2f}-{m['t1']:.2f} s {m['reason']}"
                                                              for m in masked[:3]) + "): no proposals there")
    if notes:
        ax.text(0.0, -0.45, "\n".join(notes), transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax.set_title("boundary proposals (Foote novelty on robust-z per-eighth features) - proposals, not "
                 "classifications; dashed = proposed boundary at its half-measure, dot = its novelty peak",
                 loc="left", fontsize=9, pad=2)
    ax.set_ylabel("0..1", fontsize=8)
    ax.set_xlabel("reference time (s)")
    ax.set_xlim(t0, t1 + float(np.median(np.diff(t))))
    warn = [w for w in fs.meta.get("warnings", [])]
    sub = f"{len(t)} frames, {len(fs.eighths['t0'])} eighths; grid: measures dark, half-measures light, eighths " \
          "faint; yellow = counterbalance intervals; dashed = proposed boundaries (grey = weak)"
    if rank_basis and props:
        sub += "; energy labels are ranks within this take"
    fig.suptitle(f"{take}: motion features and boundary proposals", x=0.06, y=0.995, ha="left", fontsize=12,
                 color=INK)
    top_in = 0.36
    if stale:
        fig.text(0.06, 1 - top_in / fig.get_figheight(), "STALE INPUTS (re-run features + proposals): "
                 + "; ".join(s[:110] for s in stale[:2]) + (" ..." if len(stale) > 2 else ""),
                 fontsize=8, color=STALE_RED, va="top", fontweight="bold")
        top_in += 0.2
    fig.text(0.06, 1 - top_in / fig.get_figheight(), sub + ("\n" + "; ".join(w[:150] for w in warn[:2]) if warn
                                                             else ""),
             fontsize=8, color=INK2, va="top")
    out = Path(out) if out else work_dir(take) / "review" / "moves_features.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.86, top=1 - (top_in + (0.6 if warn else 0.45)) / fig.get_figheight(),
                        bottom=0.075)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
