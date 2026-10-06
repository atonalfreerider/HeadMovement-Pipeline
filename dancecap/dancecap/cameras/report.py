"""QA for the cameras stage: top-down/side plot, focal tracks, per-camera reprojection overlays, gate G2a."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]


def plot_layout(path: Path, world_tracks: dict, points_w: np.ndarray, point_rgb: np.ndarray | None,
                ankles_w: np.ndarray | None, title: str) -> None:
    """Top-down (X-Z) and side (X-Y) views in the world frame: sparse points, camera paths, floor, dancers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(17, 8.5), gridspec_kw={"width_ratios": [1.15, 1]})
    above = points_w[:, 1] > -0.1
    sel = np.nonzero(above)[0]
    if len(sel) > 40000:
        sel = np.random.default_rng(0).choice(sel, 40000, replace=False)
    c = point_rgb[sel] / 255 if point_rgb is not None else "0.6"
    ax = axs[0]
    ax.scatter(points_w[sel, 0], points_w[sel, 2], s=0.4, c=c, alpha=0.5, lw=0)
    for k, (cam, tr) in enumerate(sorted(world_tracks.items())):
        col = COLORS[k % len(COLORS)]
        C, ok = tr["C"], tr["ok"]
        ax.plot(C[:, 0], C[:, 2], "-", color=col, lw=1.2, alpha=0.5)
        ax.plot(C[ok, 0], C[ok, 2], ".", color=col, ms=2)
        i = len(C) // 2
        if not (np.isfinite(C[i]).all() and np.abs(C[i]).max() < 1e3):
            continue
        fwd = tr["R"][i, 2]
        ax.annotate("", xy=(C[i, 0] + 1.2 * fwd[0], C[i, 2] + 1.2 * fwd[2]), xytext=(C[i, 0], C[i, 2]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
        ax.text(C[i, 0], C[i, 2], f" {cam}", color=col, fontsize=11, weight="bold")
    if ankles_w is not None and len(ankles_w):
        ax.scatter(ankles_w[:, 0], ankles_w[:, 2], s=3, c="k", alpha=0.3, label="dancers' ankles")
    ax.plot(0, 0, "k+", ms=14, mew=2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)  (+Z down: true view from above)")
    ax.set_title("top-down (world X-Z), arrows = viewing direction")
    ax.grid(alpha=0.3)
    allC = np.concatenate([tr["C"][tr["ok"]] for tr in world_tracks.values()] + [np.zeros((1, 3))])
    lim = _robust_lim(allC[np.abs(allC).max(1) < 1e3], 3.0)
    ax.set_xlim(lim[0]), ax.set_ylim(lim[2][::-1])  # Y-up right-handed seen from above: +Z points down
    ax = axs[1]
    allsel = np.random.default_rng(1).choice(len(points_w), min(len(points_w), 40000), replace=False)
    ax.scatter(points_w[allsel, 0], points_w[allsel, 1], s=0.4, c=np.where(points_w[allsel, 1] < -0.15, "r", "0.5"),
               alpha=0.4, lw=0)
    for k, (cam, tr) in enumerate(sorted(world_tracks.items())):
        ax.plot(tr["C"][:, 0], tr["C"][:, 1], "-", color=COLORS[k % len(COLORS)], lw=2)
        if np.isfinite(tr["C"][0]).all() and np.abs(tr["C"][0]).max() < 1e3:
            ax.text(tr["C"][0, 0], tr["C"][0, 1], f" {cam}", color=COLORS[k % len(COLORS)], fontsize=10)
    if ankles_w is not None and len(ankles_w):
        ax.scatter(ankles_w[:, 0], ankles_w[:, 1], s=3, c="k", alpha=0.3)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlim(lim[0])
    ax.set_ylim(-3, 5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y up (m)")
    ax.set_title("side view (X-Y): floor y=0, red = below floor (reflections)")
    ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _robust_lim(C: np.ndarray, pad: float) -> list[tuple[float, float]]:
    C = C[np.isfinite(C).all(1)]
    lo, hi = C.min(0) - pad, C.max(0) + pad
    return [(lo[i], hi[i]) for i in range(3)]


def plot_focal(path: Path, tracks: dict) -> None:
    """Raw vs smoothed focal per camera (f / max image dimension) and per-frame reprojection error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cams = sorted(tracks)
    fig, axs = plt.subplots(len(cams), 1, figsize=(12, 1.9 * len(cams)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, cam in zip(axs, cams):
        tr = tracks[cam]
        ok = tr["ok_raw"]
        ax.plot(tr["t"][ok], tr["f_raw"][ok] / tr["maxdim"], ".", ms=3, color="0.5", label="raw")
        ax.plot(tr["t"], tr["f"] / tr["maxdim"], "-", color="C0", lw=1.5, label="smoothed")
        bad = ~tr["ok"]
        if bad.any():
            ax.plot(tr["t"][bad], tr["f"][bad] / tr["maxdim"], "x", color="r", ms=4, label="not ok")
        ax2 = ax.twinx()
        ax2.plot(tr["t"], tr["reproj_px"], "-", color="C1", lw=0.8, alpha=0.7)
        ax2.set_ylabel("reproj px", color="C1", fontsize=8)
        ax.set_ylabel(f"cam {cam}\nf / maxdim", fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("reference time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def project_cv(X: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, k1: float) -> tuple[np.ndarray, np.ndarray]:
    Xc = X @ R.T + t
    z = Xc[:, 2]
    x, y = Xc[:, 0] / np.where(z > 0, z, 1), Xc[:, 1] / np.where(z > 0, z, 1)
    d = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], -1), z > 0.05


def overlay(path: Path, img: np.ndarray, K, R, t, k1, kp_xy: np.ndarray | None, xyz: np.ndarray | None,
            sim3: dict | None, label: str) -> None:
    """Reprojection overlay: inlier keypoints (green) -> projected 3D points (red ticks), 1 m floor grid (cyan)."""
    im = img.copy()
    if sim3 is not None:
        s, Rw, tw = float(sim3["s"]), np.asarray(sim3["R"]), np.asarray(sim3["t"])
        for a in np.arange(-8, 8.01, 1.0):
            for line in (np.stack([np.full(81, a), np.zeros(81), np.linspace(-8, 8, 81)], 1),
                         np.stack([np.linspace(-8, 8, 81), np.zeros(81), np.full(81, a)], 1)):
                Xs = (line - tw) @ Rw / s  # world -> SfM
                uv, vis = project_cv(Xs, K, R, t, k1)
                for p, q, v1, v2 in zip(uv[:-1], uv[1:], vis[:-1], vis[1:]):
                    if v1 and v2 and np.all(np.abs(p) < 1e4) and np.all(np.abs(q) < 1e4):
                        cv2.line(im, tuple(int(x) for x in p), tuple(int(x) for x in q),
                                 (255, 255, 0) if a != 0 else (0, 165, 255), 1, cv2.LINE_AA)
    if kp_xy is not None and xyz is not None and len(xyz):
        uv, vis = project_cv(xyz, K, R, t, k1)
        e = np.linalg.norm(uv - kp_xy, axis=1)
        for p, q, v, ee in zip(kp_xy, uv, vis, e):
            if not v:
                continue
            col = (0, 255, 0) if ee < 2 else ((0, 200, 255) if ee < 5 else (0, 0, 255))
            cv2.circle(im, (int(p[0]), int(p[1])), 2, col, -1)
            q2 = p + (q - p) * 5  # residual x5
            cv2.line(im, (int(p[0]), int(p[1])), (int(q2[0]), int(q2[1])), (0, 0, 255), 1)
        label += f"  median {np.median(e[vis]):.2f}px n={int(vis.sum())}"
    cv2.putText(im, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * im.shape[1] / 480, (0, 0, 0), 4)
    cv2.putText(im, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * im.shape[1] / 480, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY, 90])


def gates(rep: dict, n_cams: int) -> dict:
    """Gate G2a from the per-camera report."""
    cams = rep["cams"]
    reg = [c for c, v in cams.items() if v.get("registered")]
    stat_ok = all(v["map_reproj_px"] < (1.5 if v["hd"] else 1.0) for c, v in cams.items() if v.get("registered"))
    frames = sum(v["frames"] for v in cams.values())
    loc = sum(v.get("frames_ok", 0) for v in cams.values())
    focal_ok = all(v.get("focal", {}).get("plausible", False) for c, v in cams.items() if v.get("registered"))
    return {
        "cameras_registered": {"value": len(reg), "target": n_cams, "accept": 5, "passed": len(reg) >= 5,
                               "cams": reg},
        "static_reproj": {"value": {c: round(cams[c]["map_reproj_px"], 3) for c in reg},
                          "threshold": "< 1 px 480p, < 1.5 px HD", "passed": bool(reg) and stat_ok},
        "frames_localised": {"value": round(loc / max(frames, 1), 4), "target": 0.95,
                             "passed": loc / max(frames, 1) > 0.95,
                             "note": "over ALL window frames of all cameras (unregistered cameras count as 0)"},
        "focal_plausible": {"passed": bool(reg) and focal_ok,
                            "value": {c: cams[c].get("focal", {}) for c in reg}},
    }
