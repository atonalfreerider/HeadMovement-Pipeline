"""Images of the camera-tracking gate (cameras/gate_cameras.py): per camera the focal track with the zoom measured in
the video, the unexplained zoom and the residuals per 10-s block; one heat map of every camera x block; the overlap
consistency of the cameras between neighbouring takes; overlays at the zoom events (static map points + the dancers
projected through the exported camera).

    python -m dancecap.cameras.gate_cameras_plots focal      # data/zoom_compare_* + table.json -> focal_zoom_<cam>.png
    python -m dancecap.cameras.gate_cameras_plots heatmap overlap
    python -m dancecap.cameras.gate_cameras_plots overlays [--cams 04,05,07,09]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import gate_cameras as gc

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEG_COLOR = {"a": "#2a78d6", "38": "#7a6fd0", "b": "#eb6834", "c": "#1baf7a", "d": "#d55181"}
GOOD, WARN, BAD = "#2f9e63", "#e0a31a", "#d64545"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    return plt


def _load_table() -> dict:
    return json.loads((gc.OUT / "data" / "table.json").read_text())


def _motion(seg: str, cam: str) -> dict | None:
    p = gc.OUT / "data" / f"motion_{seg}_{cam}.npz"
    return dict(np.load(p)) if p.exists() else None


def focal_figures(cams=gc.CAMS, segs=("a", "38", "b", "c", "d")) -> list[Path]:
    """One page per camera: (1) the exported focal track with the zoom events seen in the video, (2) the cumulative zoom
    the video shows (black) against the zoom the exported cameras do not explain (colour: 0 = the video is reproduced),
    (3) the exported cameras' angular error against the video's own frame-to-frame motion, (4) the residuals per block."""
    plt = _mpl()
    tab = _load_table()
    out = []
    for cam in cams:
        vid = gc.load_video(cam)
        if vid is None:
            continue
        step = float(vid["step_s"])
        ev = gc.detect_video_events(vid["t"], vid["dlog"], vid["ok"], step)
        fig, axs = plt.subplots(4, 1, figsize=(15, 11.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 2.0, 1.4, 1.5]})
        for ax in axs:
            for e in ev:
                ax.axvspan(e["t0"], e["t1"], color="#f0b6b6", alpha=0.55, lw=0)
        ax_f, ax_c, ax_m, ax_r = axs
        # the video's cumulative zoom, anchored at the exported focal at the start of the recording
        tm = vid["t"]
        zc = np.cumsum(np.where(vid["ok"], vid["dlog"], 0.0))
        for seg in segs:
            d = gc.load_cam(gc.seg_work(seg), cam)
            if d is None:
                continue
            col = SEG_COLOR[seg]
            ax_f.plot(d["t"], d["f"] / d["maxdim"], color=col, lw=1.3, label=f"segment {seg}" if seg != "38" else "larissa_kadu_38")
            mp = _motion(seg, cam)
            if mp is not None:
                cum = np.cumsum(np.nan_to_num(mp["dlog_res"]))
                ax_c.plot(mp["t"], 100 * (np.exp(cum - cum[0]) - 1), color=col, lw=1.4)
                ax_m.plot(mp["t"], mp["motion_deg"], color=col, lw=0.8, alpha=0.9)
        ax_c.plot(tm, 100 * (np.exp(zc) - 1), color=INK, lw=0.8, alpha=0.8)
        ax_f.set_yscale("log")
        ax_f.set_ylabel("exported focal / max dimension")
        ax_f.set_title(f"cam {cam}: exported focal track (colour = segment); red spans = zoom events seen in the VIDEO", loc="left", fontsize=11)
        ax_f.legend(ncol=5, frameon=False, loc="lower left", fontsize=8)
        ax_c.set_ylabel("cumulative zoom (%)")
        ax_c.set_title("black: zoom the video shows since the start of the recording;  colour: what the exported cameras do NOT explain, accumulated "
                       "within each segment (a flat line at 0 = the video is reproduced)", loc="left", fontsize=8.5, color=INK2)
        ax_m.set_yscale("log")
        ax_m.set_ylim(0.005, 30)
        ax_m.axhline(0.1, color=BAD, lw=0.8, ls="--")
        ax_m.set_ylabel("angular error per\n1/6 s pair (deg)")
        for seg in segs:
            if seg == "38":
                continue
            rows = tab["segments"].get(seg, {}).get(cam, {}).get("blocks", [])
            x = [(r["t0"] + r["t1"]) / 2 for r in rows if r.get("dancers_best_px_med") is not None]
            y = [r["dancers_best_px_med"] for r in rows if r.get("dancers_best_px_med") is not None]
            ax_r.plot(x, y, "o-", color=SEG_COLOR[seg], ms=4, lw=1)
            xs = [(r["t0"] + r["t1"]) / 2 for r in rows if r.get("static_px_med") is not None]
            ys = [r["static_px_med"] for r in rows if r.get("static_px_med") is not None]
            ax_r.plot(xs, ys, "s", color=SEG_COLOR[seg], ms=3, alpha=0.6)
        base = tab["baseline"].get(cam, {})
        lim = 2 * (base.get("dancers_best_px_med") or float("nan"))
        if np.isfinite(lim):
            ax_r.axhline(max(lim, 25), color=BAD, lw=0.8, ls="--")
            ax_r.text(0.995, max(lim, 25), " limit: max(2 x larissa_kadu_38, 25 px)", transform=ax_r.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color=BAD)
        ax_r.set_yscale("log")
        ax_r.set_ylabel("px per 10 s block\ncircles: dancers (LOO, synced)\nsquares: static points")
        ax_r.set_xlabel("reference time (s)")
        ax_r.set_xlim(-5, 190)
        fig.tight_layout()
        p = gc.OUT / f"focal_zoom_{cam}.png"
        fig.savefig(p, dpi=105)
        plt.close(fig)
        out.append(p)
        print(f"[gate_cameras_plots] {p.name}", flush=True)
    return out


def heatmap_figure() -> Path:
    """Every camera x 10-s block, worst value over the segments that cover the block; six measures."""
    plt = _mpl()
    tab = _load_table()
    mot = json.loads((gc.OUT / "data" / "motion.json").read_text())
    edges = np.arange(-10, 190, 10.0)
    cams = list(gc.CAMS)
    base = tab["baseline"]
    free38 = lambda c: base.get(c, {}).get("dancers_best_px_med") or 12.0  # noqa: E731

    def collect(source: str, key: str, worst=max, segs=("a", "38", "b", "c", "d")):
        M = np.full((len(cams), len(edges)), np.nan)
        for seg in segs:
            for ci, cam in enumerate(cams):
                if source == "table":
                    rows = tab["segments"].get(seg, {}).get(cam, {}).get("blocks", [])
                    rows = [r for r in rows if r["frames"] >= 15]
                else:
                    rows = (mot.get(seg, {}).get(cam, {}) or {}).get("blocks", [])
                for r in rows:
                    j = int(round((r["t0"] - edges[0]) / 10))
                    v = r.get(key)
                    if v is None or not 0 <= j < len(edges):
                        continue
                    M[ci, j] = v if np.isnan(M[ci, j]) else worst(M[ci, j], v)
        return M

    # (title, matrix, format, warn(cam), fail(cam))
    panels = [
        ("dancers' residual, sync-corrected (px); limit max(2 x larissa_kadu_38, 25)", collect("table", "dancers_best_px_med"), "{:.0f}",
         lambda c: max(1.4 * free38(c), 18.0), lambda c: max(2.0 * free38(c), 25.0)),
        ("static reprojection (px; attached cameras: sub-model consistency, not judged)", collect("table", "static_px_med"), "{:.1f}",
         lambda c: 1e9 if c in gc.ATTACHED else 2.5, lambda c: 1e9 if c in gc.ATTACHED else 4.0),
        ("exported ok-rate (frames kept)", collect("table", "ok_exported", worst=min), "{:.2f}",
         lambda c: 0.95 if c not in gc.ATTACHED else 0.9, lambda c: 0.9 if c not in gc.ATTACHED else 0.8),
        ("camera-centre jitter p95 (mm)", collect("table", "centre_jitter_mm_p95"), "{:.0f}", lambda c: 15.0, lambda c: 30.0),
        ("angular error against the VIDEO's own motion, median per 1/6 s pair (deg)", collect("motion", "motion_deg_med"), "{:.2f}",
         lambda c: 0.08, lambda c: 0.2),
        ("zoom the exported cameras do NOT explain: video minus cameras, max over any 2 s (%)", collect("motion", "unexplained_zoom_max_2s_pct"), "{:.0f}",
         lambda c: 5.0, lambda c: 8.0),
    ]
    fig, axs = plt.subplots(len(panels), 1, figsize=(17, 3.15 * len(panels)))
    for ax, (title, M, fmt, warn, fail) in zip(axs, panels):
        col = np.zeros(M.shape + (3,))
        lower_is_worse = "ok-rate" in title
        for ci, cam in enumerate(cams):
            for j in range(len(edges)):
                v = M[ci, j]
                if np.isnan(v):
                    col[ci, j] = (0.93, 0.93, 0.92)
                    continue
                if lower_is_worse:
                    c = GOOD if v >= warn(cam) else WARN if v >= fail(cam) else BAD
                else:
                    c = GOOD if v <= warn(cam) else WARN if v <= fail(cam) else BAD
                col[ci, j] = matplotlib_rgb(c)
        ax.imshow(col, aspect="auto", extent=(edges[0], edges[-1] + 10, len(cams), 0), interpolation="nearest")
        for ci in range(len(cams)):
            for j in range(len(edges)):
                if not np.isnan(M[ci, j]):
                    ax.text(edges[j] + 5, ci + 0.5, fmt.format(M[ci, j]), ha="center", va="center", fontsize=6.5, color="white")
        ax.set_yticks(np.arange(len(cams)) + 0.5)
        ax.set_yticklabels([f"cam {c}" + (" (att.)" if c in gc.ATTACHED else "") + (" zoom" if c in gc.ZOOM_CAMS else "") for c in cams], fontsize=8)
        ax.set_xticks(np.arange(edges[0], edges[-1] + 11, 20))
        ax.grid(False)
        ax.set_title(title, loc="left", fontsize=10)
    axs[-1].set_xlabel("reference time (s)")
    fig.suptitle("Camera tracking gate, larissa_kadu_full: worst value over the segments that cover each 10-s block "
                 "(green passes, amber warns, red fails; grey: no frames)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p = gc.OUT / "gate_heatmap.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def matplotlib_rgb(hexc: str):
    h = hexc.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def overlap_figure() -> Path:
    plt = _mpl()
    d = json.loads((gc.OUT / "data" / "overlap_consistency.json").read_text())
    seams = ["a|38", "38|b", "b|c", "c|d"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, title) in zip(axs, (("centre_mm_med", "camera centre difference (mm)"), ("rot_deg_med", "rotation difference (deg)"),
                                      ("focal_pct_p95_abs", "focal difference p95 (%)"))):
        for ci, cam in enumerate(gc.CAMS):
            for si, sm in enumerate(seams):
                r = d.get(f"{sm}:{cam}")
                if r is None:
                    continue
                att = r["attached"]
                ax.scatter(ci + (si - 1.5) * 0.18, max(r[key], 1e-3), s=34, marker="D" if att else "o",
                           color=BAD if att and key != "focal_pct_p95_abs" else SEG_COLOR[["a", "38", "b", "c"][si]], zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(range(len(gc.CAMS)))
        ax.set_xticklabels(gc.CAMS)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("camera")
    axs[0].text(0.01, 0.97, "same frames localised in two neighbouring takes (same map). Main-model cameras repeat to mm;\n"
                "attached cameras 05, 06 (red diamonds) change by 0.2-0.8 m and 2-10 deg: their attach is segment-specific",
                transform=axs[0].transAxes, fontsize=7.5, va="top", color=INK2)
    fig.tight_layout()
    p = gc.OUT / "overlap_consistency.png"
    fig.savefig(p, dpi=105)
    plt.close(fig)
    return p


def zoom_events_union(cam: str, max_events: int = 4) -> list[dict]:
    """The largest zoom events of a camera: those seen in the VIDEO and those of the EXPORTED focal track (stitched over
    a, 38, b, c, d), merged when they overlap within 1.5 s; each with its source and factors."""
    vid = gc.load_video(cam)
    ev = []
    if vid is not None:
        ev += [dict(e, src="video") for e in gc.detect_video_events(vid["t"], vid["dlog"], vid["ok"], float(vid["step_s"]))]
    tracks = [(seg, d) for seg in ("a", "38", "b", "c", "d") if (d := gc.load_cam(gc.seg_work(seg), cam)) is not None]
    if tracks:
        ev += [dict(e, src="exported") for e in gc.focal_events_full(tracks)]
    ev.sort(key=lambda e: -abs(np.log(e["factor"])))
    out = []
    for e in ev:
        if all(e["t1"] < o["t0"] - 1.5 or e["t0"] > o["t1"] + 1.5 for o in out):
            out.append(e)
        if len(out) >= max_events:
            break
    return sorted(out, key=lambda e: e["t0"])


def overlay_sheets(cams=("04", "05", "07", "09"), max_events: int = 4, tile_h: int = 560, segs=("a", "b", "c", "d", "38")) -> list[Path]:
    """For each zoom camera, its largest zoom events: before / during / after frames of the segment that holds the instant,
    with the static map points (cyan), the localisation's features (yellow) against their reprojection (red), the detected
    keypoints (green) and the dancers triangulated by the OTHER cameras (magenta), all through the EXPORTED camera."""
    import cv2

    from ..take import Take
    take = Take.load("larissa_kadu_full")
    src = {s.id: Path(s.path) for s in take.sources}
    out = []
    for cam in cams:
        rows = []
        for e in zoom_events_union(cam, max_events):
            seg = next((s for s in segs if (lambda d: d is not None and d["t"][0] + 0.8 < e["t0"] and e["t1"] < d["t"][-1] - 0.8)(
                gc.load_cam(gc.seg_work(s), cam))), None)
            if seg is None:
                continue
            pts = gc.map_points(gc.seg_work(seg))
            tiles = []
            for t in (e["t0"] - 0.6, 0.5 * (e["t0"] + e["t1"]), e["t1"] + 0.6):
                im = gc.overlay_frame(seg, cam, t, src[cam], pts, label=f"{seg} cam{cam}")
                if im is not None:
                    sc = tile_h / im.shape[0]
                    tiles.append(cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))
            if tiles:
                row = np.hstack([np.hstack([t, np.full((t.shape[0], 4, 3), 255, np.uint8)]) for t in tiles])
                cap = np.full((34, row.shape[1], 3), 252, np.uint8)
                cv2.putText(cap, f"cam {cam}, zoom {e['t0']:.1f}-{e['t1']:.1f} s: x{e['factor']:.2f} ({e['src']}), segment {seg}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
                rows.append(np.vstack([cap, row]))
        if not rows:
            continue
        W = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), 252, np.uint8)]) if r.shape[1] < W else r for r in rows]
        p = gc.OUT / f"zoom_events_{cam}.jpg"
        cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 86])
        out.append(p)
        print(f"[gate_cameras_plots] {p.name}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.gate_cameras_plots", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=["focal", "heatmap", "overlap", "overlays"])
    ap.add_argument("--cams")
    a = ap.parse_args(argv)
    try:
        from .video_zoom import lower_priority
        lower_priority()
    except Exception:  # noqa: BLE001
        pass
    cams = tuple(a.cams.split(",")) if a.cams else None
    import psutil
    psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS)   # a below-normal job starves behind Unity (ffmpeg children inherit it)
    if "focal" in a.what:
        focal_figures(cams or gc.CAMS)
    if "heatmap" in a.what:
        print(heatmap_figure())
    if "overlap" in a.what:
        print(overlap_figure())
    if "overlays" in a.what:
        overlay_sheets(cams or ("04", "05", "07", "09"))


if __name__ == "__main__":
    main()
