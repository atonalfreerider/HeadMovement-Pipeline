"""Images of the camera-tracking gate (cameras/gate_cameras.py numbers): focal tracks vs the zoom measured in the video,
per-10-s-block heat-maps, overlays at zoom events.

    python -m dancecap.cameras.gate_images focal [--tag _before]     # review/gate_cameras/focal_tracks<tag>.png
    python -m dancecap.cameras.gate_images blocks                    # review/gate_cameras/blocks.png
    python -m dancecap.cameras.gate_images overlays [--cams 04,05,07,09] [--tag _before]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import gate_cameras as gc
from .gate_cameras import BLOCK_S, CAMS, OUT, SEGS, load_cam, load_video, seg_work


def stitched(cam: str, segs: tuple[str, ...] = SEGS) -> dict | None:
    """Exported track of a camera over the whole dance (segments in time order, the earlier segment wins an overlap;
    larissa_kadu_38 supplies 30-68 s where the camera's segments do not reach)."""
    parts, last = [], -1e9
    for seg in ("a", "38", "b", "c", "d"):
        if seg != "38" and seg not in segs:
            continue
        d = load_cam(seg_work(seg), cam)
        if d is None:
            continue
        m = d["t"] > last + 1e-6
        if not m.any():
            continue
        parts.append((seg, d, m))
        last = d["t"][-1]
    if not parts:
        return None
    out = {"cam": cam, "attached": parts[0][1]["attached"], "maxdim": parts[0][1]["maxdim"]}
    for k in ("t", "f", "ok", "ok_raw", "reproj_px", "recentred"):
        out[k] = np.concatenate([d[k][m] for _, d, m in parts])
    out["seg"] = np.concatenate([np.full(int(m.sum()), seg) for seg, _, m in parts])
    out["C"] = np.concatenate([d["C"][m] for _, d, m in parts])
    return out


def plot_focal_tracks(tag: str = "", note: str = "") -> Path:
    """Per camera the exported focal over the whole dance (f / max dim, log) and the zoom measured in the VIDEO
    (cumulative frame-to-frame scale anchored to the focal over the first 20 s), orange bands = video zoom events."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evp = OUT / "data" / f"zoom_evidence{tag}.json"
    ev = json.loads(evp.read_text()) if evp.exists() else {}
    fig, axs = plt.subplots(len(CAMS), 1, figsize=(17, 2.4 * len(CAMS)), sharex=True)
    for ax, cam in zip(axs, CAMS):
        tr = stitched(cam)
        if tr is None:
            continue
        fm, ok = tr["f"] / tr["maxdim"], tr["ok"]
        ax.semilogy(tr["t"], np.where(ok, fm, np.nan), "-", color="#1f77b4", lw=1.8, label="exported focal (f / max dim)")
        vid = load_video(cam)
        if vid is not None:
            c = np.cumsum(np.where(vid["ok"], vid["dlog"], 0.0))
            tv = vid["t"]
            m = (tr["t"] >= tv[0]) & (tr["t"] <= tv[0] + 20) & ok
            if m.any():
                fa = np.exp(np.median(np.log(fm[m]) - np.interp(tr["t"][m], tv, c)))
                ax.semilogy(tv, fa * np.exp(c), "-", color="#d95f02", lw=0.9, alpha=0.9,
                            label="zoom measured in the video (cumulative from the start; drifts slowly)")
        for e in (ev.get(cam) or {}).get("video_events", []):
            ax.axvspan(e["t0"], e["t1"], color="#d95f02", alpha=0.25, lw=0)
        for edge in (30.15, 68.38):
            ax.axvline(edge, color="0.6", lw=0.6, ls=":")
        ax.set_ylabel(f"cam {cam}\n" + ("attached" if tr["attached"] else "main model"), rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3, which="both")
        if cam == CAMS[0]:
            ax.legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("reference time (s); dotted: larissa_kadu_38 span")
    fig.suptitle("Exported focal track vs the zoom measured directly in the video (orange bands = video zoom events)" + note)
    fig.tight_layout()
    p = OUT / f"focal_tracks{tag}.png"
    fig.savefig(p, dpi=80)
    plt.close(fig)
    return p


def plot_blocks(table: dict, path: Path, zoom: dict | None = None) -> Path:
    """Heat-maps of the per-10-s-block measures: cameras (rows) x blocks (columns)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    measures = [("ok_exported", "exported ok-rate (gate >= 0.95, attached >= 0.90)", 0.8, 1.0, "viridis"),
                ("static_px_med", "static reprojection, px (median; attached cameras: sub-model consistency)", 0.8, 4.0, "magma_r"),
                ("dancers_px_med", "dancers' leave-one-out residual, px (median)", 8, 40, "magma_r"),
                ("centre_jitter_mm_p95", "camera-centre jitter p95, mm", 0, 40, "magma_r")]
    edges = np.arange(-10, 200, BLOCK_S)
    n = len(measures) + (1 if zoom else 0)
    fig, axs = plt.subplots(n, 1, figsize=(17, 3.0 * n))
    axs = np.atleast_1d(axs)

    def fill(get):
        M = np.full((len(CAMS), len(edges)), np.nan)
        for ci, cam in enumerate(CAMS):
            for seg in table["segments"].values():
                for b in (seg.get(cam) or {}).get("blocks", []):
                    j = int(round((b["t0"] - edges[0]) / BLOCK_S))
                    v = get(b)
                    if v is not None and b["frames"] >= 15 and 0 <= j < len(edges):
                        M[ci, j] = v if np.isnan(M[ci, j]) else max(M[ci, j], v)
        return M

    def draw(ax, M, title, lo, hi, cmap, fmt):
        im = ax.imshow(M, aspect="auto", vmin=lo, vmax=hi, cmap=cmap, extent=[edges[0], edges[-1] + BLOCK_S, len(CAMS), 0])
        ax.set_yticks(np.arange(len(CAMS)) + 0.5)
        ax.set_yticklabels(CAMS)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, pad=0.01)
        for ci in range(len(CAMS)):
            for j in range(len(edges)):
                if np.isfinite(M[ci, j]) and edges[j] % 20 == 0:
                    ax.text(edges[j] + BLOCK_S / 2, ci + 0.55, fmt(M[ci, j]), ha="center", va="center", fontsize=6, color="0.5")

    for ax, (key, title, lo, hi, cmap) in zip(axs, measures):
        M = fill(lambda b, k=key: b.get(k))
        if key == "ok_exported":   # the worst value in a block that two segments both cover is the honest one
            M = np.full_like(M, np.nan)
            for ci, cam in enumerate(CAMS):
                for seg in table["segments"].values():
                    for b in (seg.get(cam) or {}).get("blocks", []):
                        j = int(round((b["t0"] - edges[0]) / BLOCK_S))
                        if b["frames"] >= 15 and 0 <= j < len(edges):
                            M[ci, j] = b[key] if np.isnan(M[ci, j]) else min(M[ci, j], b[key])
        draw(ax, M, title, lo, hi, cmap, (lambda v: f"{v:.2f}") if key == "ok_exported" else (lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}"))
    if zoom:
        M = np.full((len(CAMS), len(edges)), np.nan)
        for ci, cam in enumerate(CAMS):
            for seg, m in (zoom.get(cam) or {}).get("segments", {}).items():
                for b in m["blocks"]:
                    j = int(round((b["t0"] - edges[0]) / BLOCK_S))
                    if 0 <= j < len(edges):
                        M[ci, j] = b["unexplained_max_2s_pct"] if np.isnan(M[ci, j]) else max(M[ci, j], b["unexplained_max_2s_pct"])
        draw(axs[-1], M, "zoom the exported cameras do NOT explain: |video zoom - predicted zoom| over any 2 s, % (gate <= 5)",
             0, 10, "magma_r", lambda v: f"{v:.1f}")
    axs[-1].set_xlabel("reference time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def overlay_events(seg: str, cam: str, times: list[float], path: Path, sources: dict, proj_cache: dict | None = None,
                   n_static: int = 250) -> Path | None:
    """Frames of the video at `times` (reference clock) in segment seg with the exported camera projecting (cyan) the room's
    static map points of the frame, (magenta o) the dancers' keypoints triangulated by the OTHER main cameras and (green +)
    the keypoints detected in this frame. One tile per time."""
    import cv2

    from .pipeline import dancer_check
    from .report import project_cv
    work = seg_work(seg)
    d = load_cam(work, cam)
    if d is None:
        return None
    key = (seg, cam)
    if proj_cache is not None and key in proj_cache:
        proj = proj_cache[key]
    else:
        rep = json.loads((work / "cameras" / "report.json").read_text())
        main = sorted(c for c, v in rep["cams"].items() if v.get("in_main_model"))
        tracks = {}
        for c in CAMS:
            x = load_cam(work, c)
            if x is not None:
                tracks[c] = {"t": x["t"], "K": x["K"], "R": x["R"], "tvec": x["tvec"], "k1": x["k1"], "ok": x["ok"],
                             "maxdim": x["maxdim"]}
        res = dancer_check(work, tracks, [cam], qa_dir=None, main_cams=main, attach=rep.get("attach") or {})
        proj = (res.get(cam) or {}).get("_proj", {})
        if proj_cache is not None:
            proj_cache[key] = proj
    q = work / gc.BASE / "cache" / f"loc_{cam}.npz"
    if not q.exists():
        q = work / "cameras" / "cache" / f"loc_{cam}.npz"
    with np.load(q, allow_pickle=True) as z:
        off, xyz = z["corr_off"], z["corr_xyz"]
    tiles = []
    for t in times:
        i = int(np.argmin(np.abs(d["t"] - t)))
        img = gc.read_frame(sources[cam], (d["t"][i] - d["offset"]) / d["slope"], d["w"], d["h"])
        if img is None:
            continue
        a, b = int(off[i]), int(off[i + 1])
        if b > a:
            sel = np.random.default_rng(0).choice(b - a, min(n_static, b - a), replace=False) + a
            p, vis = project_cv(xyz[sel].astype(float), d["K"][i], d["R"][i], d["tvec"][i], d["k1"][i])
            for (px, py), v in zip(p, vis):
                if v and 0 <= px < d["w"] and 0 <= py < d["h"]:
                    cv2.circle(img, (int(px), int(py)), 2, (255, 255, 0), -1)
        if i in proj:
            pp, uv = proj[i]
            for (px, py), (ux, uy) in zip(pp, uv):
                if abs(px) < 1e4 and abs(py) < 1e4:
                    cv2.line(img, (int(ux), int(uy)), (int(px), int(py)), (255, 0, 255), 1)
                    cv2.circle(img, (int(px), int(py)), 5, (255, 0, 255), 2)
                cv2.drawMarker(img, (int(ux), int(uy)), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
        txt = f"cam {cam} t={d['t'][i]:.2f}s f/md={d['f'][i] / d['maxdim']:.3f}"
        sc = max(1.0, d["w"] / 480)
        cv2.putText(img, txt, (6, int(18 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (0, 0, 0), max(2, int(3 * sc)), cv2.LINE_AA)
        cv2.putText(img, txt, (6, int(18 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (0, 255, 255), max(1, int(sc)), cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return None
    Ht = 480
    tiles = [cv2.resize(t, (int(t.shape[1] * Ht / t.shape[0]), Ht), interpolation=cv2.INTER_AREA) for t in tiles]
    row = np.hstack([x for t in tiles for x in (t, np.full((Ht, 4, 3), 255, np.uint8))][:-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), row, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return path


def event_times(ev: dict, max_events: int = 4) -> list[tuple[float, list[float]]]:
    """(event centre, [before, start, middle, end, after]) for the largest video events of a camera."""
    out = []
    for e in sorted(ev["video_events"], key=lambda e: -abs(np.log(e["factor"])))[:max_events]:
        mid = 0.5 * (e["t0"] + e["t1"])
        out.append((mid, [e["t0"] - 0.8, e["t0"], mid, e["t1"], e["t1"] + 0.8]))
    return sorted(out)


def seg_of(t: float) -> str:
    """Segment whose fit span holds t (a, b, c, d); the 38 s span is covered by a (to 34.2) and b (from 64.4)."""
    for seg, lo, hi in (("a", -3.2, 34.2), ("b", 62.4, 107.2), ("c", 107.2, 146.1), ("d", 146.1, 186.0)):
        if lo <= t <= hi:
            return seg
    return "38"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.gate_images", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["focal", "blocks", "overlays"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--cams", default="04,05,07,09")
    ap.add_argument("--max-events", type=int, default=3)
    ap.add_argument("--before", action="store_true")
    a = ap.parse_args(argv)
    from .video_zoom import lower_priority
    lower_priority()
    if a.before:
        gc.BASE = "cameras/s2_gate_backup"
    if a.what == "focal":
        print(plot_focal_tracks(a.tag))
    elif a.what == "blocks":
        tj = json.loads((OUT / "data" / f"table{a.tag}.json").read_text())
        zp = OUT / "data" / f"zoom_evidence{a.tag}.json"
        print(plot_blocks(tj, OUT / f"blocks{a.tag}.png", json.loads(zp.read_text()) if zp.exists() else None))
    else:
        from ..take import Take
        take = Take.load("larissa_kadu_full")
        sources = {s.id: Path(s.path) for s in take.sources}
        ev = json.loads((OUT / "data" / f"zoom_evidence{a.tag}.json").read_text())
        cache: dict = {}
        for cam in a.cams.split(","):
            for mid, ts in event_times(ev[cam], a.max_events):
                seg = seg_of(mid)
                p = OUT / "overlays" / f"{cam}_{mid:06.1f}{a.tag}.jpg"
                r = overlay_events(seg, cam, ts, p, sources, cache)
                print(f"[gate_images] {cam} {mid:.1f}s seg {seg} -> {r}", flush=True)


if __name__ == "__main__":
    main()


def plot_attach_paths(path: Path | None = None) -> Path:
    """Attached cameras (05, 06): the whole-dance centre path of attach_global (line, SfM axes x scale, metres relative to the
    median) against the per-take constant centres of the previous attach (steps, from cameras/s2_gate_backup), and the
    sub-model -> main rotation of every take (before: per take, after: one)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation

    from . import attach_global as ag
    path = path or OUT / "attach_paths.png"
    fig, axs = plt.subplots(2, 3, figsize=(17, 7), sharex=True)
    s_w = gc.sim3_scale(seg_work("a"))
    for r, cam in enumerate(("05", "06")):
        sol = json.loads((ag.SOL_DIR / f"{cam}.json").read_text())
        K, C = np.array(sol["knots"]), np.array(sol["C"])
        med = np.median(C, 0)
        gc_base = gc.BASE
        for ax_i, ax_name in enumerate(("X", "Y", "Z")):
            ax = axs[r, ax_i]
            ax.plot(K, (C[:, ax_i] - med[ax_i]) * s_w, "-o", ms=3, color="#d95f02", label="attach_global: one path for the dance")
            for seg, col in (("38", "#7a6fd0"), ("a", "#2a78d6"), ("b", "#eb6834"), ("c", "#1baf7a"), ("d", "#d55181")):
                try:
                    gc.BASE = "cameras/s2_gate_backup" if seg != "38" else "cameras"
                    d = load_cam(seg_work(seg), cam)
                except Exception:   # noqa: BLE001
                    d = None
                if d is None:
                    continue
                c0 = d["C"][d["ok"]][0] if d["ok"].any() else d["C"][0]
                ax.hlines((c0[ax_i] - med[ax_i]) * s_w, d["t"][0], d["t"][-1], color=col, lw=2.2, alpha=0.8,
                          label=f"previous per-take constant ({seg})")
            gc.BASE = gc_base
            ax.set_title(f"cam {cam}: centre {ax_name} (m, SfM axes)")
            ax.grid(alpha=0.3)
            if r == 0 and ax_i == 0:
                ax.legend(fontsize=7, loc="best")
        # rotation consistency in the title of the row
        rs = []
        for seg in ("38", "a", "b", "c", "d"):
            try:
                import pickle
                work = seg_work(seg)
                poses = pickle.loads((work / "cameras" / "cache" / f"locsub_{cam}.pkl").read_bytes())["value"]
                gc.BASE = "cameras/s2_gate_backup" if seg != "38" else "cameras"
                d = load_cam(work, cam)
                ok = np.array([p.ok for p in poses]) & d["ok"]
                Rsub = np.array([p.R for p in poses])
                M = np.einsum("fji,fjk->fik", Rsub, d["R"])[ok].mean(0)
                U, _, Vt = np.linalg.svd(M)
                rs.append(U @ Vt)
            except Exception:   # noqa: BLE001
                continue
        gc.BASE = gc_base
        if len(rs) >= 2:
            ang = [np.degrees(np.linalg.norm(Rotation.from_matrix(rs[0].T @ x).as_rotvec())) for x in rs[1:]]
            axs[r, 0].set_ylabel(f"cam {cam}\nprevious Rs vs the first take: " + ", ".join(f"{a:.1f} deg" for a in ang), fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("reference time (s)")
    fig.suptitle("Attached cameras: one sub-model->main rotation and a moving centre for the whole dance (line) vs a rotation and a "
                 "constant centre per take (steps)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path
