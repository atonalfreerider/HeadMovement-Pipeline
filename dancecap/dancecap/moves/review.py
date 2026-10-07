"""Review images: the dance graph as the user built it, and a take's fingerprint.

    python -m dancecap.moves review                 work/review/moves_graph.png
    python -m dancecap.moves review <take>          + work/<take>/review/fingerprint.png (from the take's files)

Two orthographic views of the authored Zouk1 layout (Unity axes): FRONT (x right, y up: height = energy, with the
energy bands) and TOP (x right, z up: a Unity top-down camera). Node icons approximate the Zouk1 shapes, fills are
the Zouk1 tones; proposed nodes/links are dashed. Node labels are the display names (Brazilian Portuguese where
established) with the user's Zouk1 name in brackets when it differs ("Corredor (Lateral)").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

from . import PLACEHOLDER_BANNER, REVIEW_DIR, moves_dir, work_dir  # noqa: E402
from .graph import GRAPH_PATH, TONES, Graph, load_graph  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # one-hue blue ramp
BAND_FILL = ["#f3f2ee", "#fcfcfb"]

HOURGLASS = MPath([(-1, 1), (1, 1), (0, 0), (1, -1), (-1, -1), (0, 0), (-1, 1)],
                  [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.LINETO,
                   MPath.CLOSEPOLY])
CYLINDER = MPath([(-0.6, -1), (0.6, -1), (0.6, 1), (-0.6, 1), (-0.6, -1)],
                 [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY])
MARKERS = {"Cylinder": CYLINDER, "Hourglass": HOURGLASS, "Diamond": "D", "Plus": "P", "Star": "*", "Tetra": "^",
           "Ball": "o"}
MSIZE = {"Star": 150, "Plus": 85, "Diamond": 55, "Tetra": 75, "Ball": 70, "Hourglass": 80, "Cylinder": 80}
VIEWS = {"front": (0, 1, "x (authored)", "y = height = energy"), "top": (0, 2, "x (authored)", "z (authored)")}


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.titlecolor": INK, "text.color": INK})


def _bands(ax, graph: Graph, label: bool = True):
    for b in graph.data["energy_bands"]:
        ax.axhspan(b["y_lo"], b["y_hi"], color=BAND_FILL[b["band"] % 2], zorder=0, lw=0)
        ax.axhline(b["y_lo"], color=GRID, lw=0.6, zorder=0)
        if label:
            ax.text(1.004, (b["y_lo"] + b["y_hi"]) / 2, f"{b['band']} {b['label']}\nenergy {b['lo']:.1f}-{b['hi']:.1f}",
                    transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.5, color=INK2, zorder=1,
                    clip_on=False)


def _links(ax, graph: Graph, i: int, j: int, style):
    for l in graph.links:
        st = style(l)
        if st is None:
            continue
        a, b = graph.nodes[l["from"]]["position"], graph.nodes[l["to"]]["position"]
        ax.annotate("", xy=(b[i], b[j]), xytext=(a[i], a[j]), zorder=st.get("z", 2),
                    arrowprops=dict(arrowstyle="-|>", lw=st["lw"], color=st["color"], ls=st.get("ls", "-"),
                                    shrinkA=5, shrinkB=6, mutation_scale=st.get("ms", 7), alpha=st.get("alpha", 1)))


def node_label(n: dict) -> str:
    """Display name, + the authored Zouk1 name in brackets when it is a different name (not just capitalisation)."""
    name, authored = n.get("name") or "", n.get("authored_name") or ""
    if authored and name and authored.lower() != name.lower():
        return f"{name} ({authored})"
    return name


def _nodes(ax, graph: Graph, i: int, j: int, style, labels: bool = True, label_fn=None):
    for n in graph.nodes.values():
        st = style(n)
        if st is None:
            continue
        p = n["position"]
        face = "none" if isinstance(st["face"], str) and st["face"] == "none" else [st["face"]]
        ax.scatter([p[i]], [p[j]], marker=MARKERS.get(n["shape"], "o"), s=st["s"], facecolors=face,
                   edgecolors=st["edge"], linewidths=st.get("lw", 0.8), linestyles=st.get("ls", "-"),
                   zorder=st.get("z", 4), alpha=st.get("alpha", 1.0))
        if labels:
            txt = label_fn(n) if label_fn else node_label(n)
            if txt:
                ax.annotate(txt, (p[i], p[j]), xytext=(5, 2), textcoords="offset points", fontsize=st.get("fs", 6.3),
                            color=st.get("tc", INK), zorder=6, style="italic" if n["status"] != "scaffold" else "normal")


def _legend(fig, graph: Graph, loc=(0.5, 0.005)):
    handles = []
    for shape, s in graph.data["legend"]["shapes"].items():
        handles.append(Line2D([], [], marker=MARKERS[shape], ls="", markersize=8, markerfacecolor="#dcdcd6",
                              markeredgecolor=INK2, label=f"{shape}: {s['label']} ({s['count']})"))
    for name, t in graph.data["legend"]["tones"].items():
        handles.append(Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=TONES[name],
                              markeredgecolor=INK2, label=f"{name}: {t['meaning']} ({t['count']})"))
    handles.append(Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor="none", markeredgecolor=INK2,
                          label="dashed outline = proposed"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=loc, ncol=4, frameon=False, fontsize=8.5)


def draw_graph(graph: Graph, out: Path = REVIEW_DIR / "moves_graph.png") -> Path:
    """The graph as authored (front + top views), shapes as markers, tones as fills, names labelled."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(28, 13.5), gridspec_kw={"width_ratios": [1, 1]})
    c = graph.data["counts"]

    def node_style(n):
        prop = n["status"] != "scaffold"
        return {"s": MSIZE.get(n["shape"], 70) * (1.25 if prop else 1.0), "face": "none" if prop else n["tone"],
                "edge": INK if prop else INK2, "lw": 1.4 if prop else 0.7, "ls": "--" if prop else "-",
                "tc": INK if n["name"] else MUTED}

    def link_style(l):
        if l["status"] == "scaffold":
            return {"lw": 0.55, "color": "#b9b8b1"}
        return {"lw": 1.1, "color": INK2, "ls": "--", "z": 3}

    for ax, (view, (i, j, xl, yl)) in zip(axes, VIEWS.items()):
        if view == "front":
            _bands(ax, graph)
        else:
            ax.grid(True, color=GRID, lw=0.5, zorder=0)
        _links(ax, graph, i, j, link_style)
        _nodes(ax, graph, i, j, node_style)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title({"front": "FRONT view (x, y): height = energy, grounded states at the bottom",
                      "top": "TOP view (x, z): Unity top-down"}[view], fontsize=12, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    lg = graph.data["legend"]
    conf = (f"Legend confirmed by the user {lg['confirmed']['date']}." if lg.get("confirmed_by_user")
            else "Legend not confirmed.")
    fig.suptitle(f"Zouk dance graph - Zouk1.json as authored: {c['scaffold_nodes']} nodes, {c['scaffold_links']} "
                 f"directed links (+ {c['proposed_nodes']} proposed nodes, {c['proposed_links']} proposed links from "
                 f"taxonomy/additions.toml + vocabulary research); {c['moves']} moves, {c['duplicated_moves']} with "
                 f"several instances, {c['blank_nodes']} unnamed ({c.get('via_links', 0)} connections through unnamed "
                 f"junctions).\nNames in Brazilian Portuguese where established (user's name in brackets). {conf}",
                 fontsize=13, x=0.01, ha="left", y=0.99)
    _legend(fig, graph)
    fig.tight_layout(rect=(0, 0.06, 1, 0.955))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _seq(v: float, vmax: float) -> str:
    k = 0 if vmax <= 0 else int(np.clip(np.ceil(v / vmax * (len(SEQ) - 1)), 1, len(SEQ) - 1))
    return SEQ[k]


def draw_fingerprint(graph: Graph, fp: dict, path_doc: dict | None, out: Path) -> Path:
    """Dwell-time heat map on the graph (front + top), band totals, energy over time; PLACEHOLDER banner."""
    _style()
    placeholder = fp.get("provenance") == "placeholder"
    fig = plt.figure(figsize=(26, 14.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 0.6], height_ratios=[3.2, 1.0], hspace=0.18, wspace=0.22)
    ax_f, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    right = gs[0, 2].subgridspec(2, 1, height_ratios=[1.0, 0.75], hspace=0.12)
    ax_b, ax_s, ax_time = fig.add_subplot(right[0]), fig.add_subplot(right[1]), fig.add_subplot(gs[1, :])
    ax_s.axis("off")
    secs = {k: v["seconds"] for k, v in fp["nodes"].items()}
    vmax = max(secs.values()) if secs else 1.0
    counts = fp["links"]
    cmax = max(counts.values()) if counts else 1
    traversed = {}
    for key, n in counts.items():
        a, b = key.split("|")
        traversed[(a, b)] = n
    via = fp.get("via") or {}

    def node_style(n):
        sec = secs.get(n["id"])
        if sec is None:
            return {"s": 14, "face": "#e4e3dd", "edge": "#c9c8c1", "lw": 0.5, "z": 3,
                    "ls": "--" if n["status"] != "scaffold" else "-"}
        return {"s": 70 + 520 * sec / vmax, "face": _seq(sec, vmax), "edge": INK, "lw": 1.0, "z": 5,
                "ls": "--" if n["status"] != "scaffold" else "-", "fs": 8.5}

    def link_style(l):
        return {"lw": 0.4, "color": "#dddcd5", "z": 1}

    def label(n):
        sec = secs.get(n["id"])
        return f"{node_label(n)}  {sec:.1f}s" if sec is not None else ""

    for ax, (view, (i, j, xl, yl)) in zip((ax_f, ax_t), VIEWS.items()):
        if view == "front":
            _bands(ax, graph)
        else:
            ax.grid(True, color=GRID, lw=0.5, zorder=0)
        _links(ax, graph, i, j, link_style)
        for (a, b), n in traversed.items():  # traversed transitions on top, width by count, new links dashed
            new = f"{a}|{b}" in set(fp.get("new_link_keys", []))
            chain = [a, *via.get(f"{a}|{b}", []), b]  # through unnamed junction nodes
            for u, v in zip(chain, chain[1:]):
                pa, pb = graph.nodes[u]["position"], graph.nodes[v]["position"]
                ax.annotate("", xy=(pb[i], pb[j]), xytext=(pa[i], pa[j]), zorder=4,
                            arrowprops=dict(arrowstyle="-|>", lw=1.2 + 2.8 * n / cmax, color=INK2 if new else SEQ[5],
                                            ls="--" if new else "-", shrinkA=7 if u == a else 2,
                                            shrinkB=9 if v == b else 2, mutation_scale=12))
        _nodes(ax, graph, i, j, node_style, label_fn=label)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title({"front": "FRONT (x, y = energy): node size and shade = seconds spent",
                      "top": "TOP (x, z): arrows = transitions taken (width = count, dashed = new link)"}[view],
                     fontsize=11.5, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)

    bands = fp["energy_bands"]
    y = np.arange(len(bands))
    ax_b.barh(y, [b["seconds"] for b in bands], color=SEQ[4], height=0.62, zorder=2)
    for k, b in enumerate(bands):
        ax_b.text(b["seconds"], k, f"  {b['seconds']:.1f} s ({100 * b['share']:.0f}%)", va="center", fontsize=9,
                  color=INK)
    ax_b.set_yticks(y, [f"{b['band']} {b.get('label', '')}" for b in bands])
    ax_b.set_xlim(0, max(1e-6, max(b["seconds"] for b in bands)) * 1.6)
    ax_b.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    ax_b.set_title("time per energy band", fontsize=11.5, loc="left")
    for s in ax_b.spines.values():
        s.set_visible(False)
    lp = fp.get("longest_phrase") or {}
    cb = fp.get("counterbalance_share")
    stats = [f"labelled  {fp['labelled_seconds']:.1f} s",
             f"distinct moves  {fp['distinct_moves']}",
             f"transitions  {fp['transitions']}  (new links {len(fp.get('new_link_keys', []))}, "
             f"walked backwards {len(fp.get('reverse_keys', []))})",
             f"nodes visited  {len(fp['nodes'])}",
             "counterbalance  " + _cb_text(fp),
             f"longest phrase  {lp.get('seconds', 0):.1f} s, {len(lp.get('moves', []))} moves"]
    ax_s.text(0.0, 0.92, "\n".join(stats), transform=ax_s.transAxes, va="top", ha="left", fontsize=10.5,
              color=INK, linespacing=1.7)

    # energy over time (band of each step), with move names
    if path_doc and path_doc.get("steps"):
        for k, st in enumerate(path_doc["steps"]):
            b = graph.nodes[st["node"]]["band"]
            ax_time.plot([st["t0"], st["t1"]], [b, b], color=SEQ[5], lw=3, solid_capstyle="butt", zorder=3)
            if k:
                pb = graph.nodes[path_doc["steps"][k - 1]["node"]]["band"]
                ax_time.plot([st["t0"], st["t0"]], [pb, b], color=INK2 if st.get("new_link") else SEQ[3],
                             lw=1.2, ls="--" if st.get("new_link") else "-", zorder=2)
            ax_time.text((st["t0"] + st["t1"]) / 2, b + 0.18, node_label(graph.nodes[st["node"]]), ha="center",
                         va="bottom", fontsize=8, color=INK, rotation=0)
        ax_time.set_yticks(range(len(bands)), [b.get("label", "") for b in bands])
        ax_time.set_ylim(-0.5, len(bands) - 0.2)
        ax_time.set_xlabel("reference seconds")
        ax_time.grid(True, color=GRID, lw=0.5, zorder=0)
        ax_time.set_title("energy band over time (one bar per labelled segment)", fontsize=11.5, loc="left")
        for s in ax_time.spines.values():
            s.set_visible(False)
    else:
        ax_time.axis("off")

    title = f"Dance fingerprint - {fp['take']}"
    if placeholder:
        title = f"PLACEHOLDER - NOT AN ANALYSIS  |  {title}  (seeded random walk on the graph, beat grid)"
    fig.suptitle(title, fontsize=17 if placeholder else 14, x=0.01, ha="left", y=0.99,
                 color="#9a3412" if placeholder else INK, weight="bold" if placeholder else "normal")
    if placeholder:
        fig.text(0.01, 0.955, f"{PLACEHOLDER_BANNER}: provenance 'placeholder'. The moves shown were generated to "
                 "exercise the viewer prototype and say nothing about what the dancers did.", fontsize=11, color=INK2)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def _cb_text(fp: dict) -> str:
    cb, an = fp.get("counterbalance_share"), fp.get("counterbalance_analysed")
    if fp.get("counterbalance_source") is None:
        return "n/a (no counterbalance.json)"
    if cb is None:
        return "n/a (labels outside the analysed range" + (f" {an[0]:.1f}-{an[1]:.1f} s)" if an else ")")
    rng = (f" of {fp.get('counterbalance_analysed_seconds', 0):.1f} s analysed ({an[0]:.1f}-{an[1]:.1f} s)"
           if an else "")
    return f"{100 * cb:.0f}%{rng}"


def cmd_review(args: argparse.Namespace) -> int:
    graph = load_graph(Path(args.graph))
    print(f"[moves] {draw_graph(graph)}")
    if args.take:
        fpf = moves_dir(args.take) / "fingerprint.json"
        pf = moves_dir(args.take) / "path.json"
        if not fpf.exists():
            print(f"[moves] {fpf} missing (run fingerprint first)")
            return 1
        fp = json.loads(fpf.read_text(encoding="utf-8"))
        pd = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else None
        print(f"[moves] {draw_fingerprint(graph, fp, pd, work_dir(args.take) / 'review' / 'fingerprint.png')}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("review", help="review images: work/review/moves_graph.png (+ <take>/review/fingerprint.png)")
    p.add_argument("take", nargs="?")
    p.add_argument("--graph", default=str(GRAPH_PATH))
    p.set_defaults(func=cmd_review)
