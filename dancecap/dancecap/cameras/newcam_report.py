"""PC side of dancecap.cameras.newcam: merge the 60 s chunks of a camera into its track, export it in the demo's
cameras/<cam>.npz schema (+ a world-frame twin), measure the quality / coverage, and draw the review images.

    track        merge_camera()   chunks/<cam>_<k>.npz  ->  cameras/<cam>.npz   (SfM frame, K at NATIVE display resolution)
                                                            cameras/<cam>_world.npz (metric world frame of the room model)
    quality      camera_quality() per camera: frames ok / localised / interpolated, reprojection, inliers, gaps, chunk overlap
                                  agreement, intrinsics, per-block coverage on the master clock; verdict usable | partial | unusable
    epipolar     epipolar_check() identity-free check between two cameras from the owner's 2D pose JSONs: the people of both
                                  views must satisfy the epipolar constraint of the two tracks (non-circular: the poses come from
                                  static features only)
    review       contact_sheet()  the room model (sparse points + 1 m floor grid + the world axes) projected into 8 frames
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .newcam import CHUNK_S, OVERLAP_S, WINDOW, chunk_windows


# ------------------------------------------------------------------------------------------------ merge / export


def _K_native(f_work: float, scale_to_native: float, native: tuple[int, int]) -> np.ndarray:
    w0, h0 = native
    K = np.eye(3)
    K[0, 0] = K[1, 1] = f_work * scale_to_native
    K[0, 2], K[1, 2] = w0 / 2 - 0.5, h0 / 2 - 0.5
    return K


def merge_chunks(chunks: list[dict]) -> dict:
    """One track from the chunk dicts (each: n, t, pts, ok, R, tvec, C, n_inliers, reproj_px, source, interp, localised, core, window).
    Every frame comes from the chunk whose CORE holds its reference time; frames that two chunks both localise (the 4 s overlaps)
    give the stitching agreement: centre distance (SfM units) and rotation angle (deg) of the two solutions."""
    from scipy.spatial.transform import Rotation as Ro
    chunks = sorted((c for c in chunks if len(c.get("n", [])) > 0), key=lambda c: int(c["chunk"]))
    if not chunks:
        return {"n": np.zeros(0, int)}
    keys = ("pts", "t", "ok", "R", "tvec", "C", "n_inliers", "reproj_px", "source", "interp", "localised")
    pieces = {k: [] for k in keys}
    nn, owner = [], []
    for c in chunks:
        t = np.where(np.isfinite(c["t"]), c["t"], c["pts"])
        a, b = float(c["core"][0]), float(c["core"][1])
        last = c is chunks[-1]
        first = c is chunks[0]
        m = ((t >= a) | first) & ((t < b) | last)
        nn.append(c["n"][m])
        owner.append(np.full(int(m.sum()), int(c["chunk"])))
        for k in keys:
            pieces[k].append(c[k][m])
    n = np.concatenate(nn)
    order = np.argsort(n)
    out = {"n": n[order], "chunk": np.concatenate(owner)[order]}
    for k in keys:
        out[k] = np.concatenate(pieces[k])[order]
    # overlap agreement
    dC, dR = [], []
    for c0, c1 in zip(chunks[:-1], chunks[1:]):
        common, i0, i1 = np.intersect1d(c0["n"], c1["n"], return_indices=True)
        good = c0["ok"][i0] & c1["ok"][i1] & c0["localised"][i0] & c1["localised"][i1]
        if good.any():
            dC.append(np.linalg.norm(c0["C"][i0][good] - c1["C"][i1][good], axis=1))
            rel = Ro.from_matrix(c0["R"][i0][good] @ np.transpose(c1["R"][i1][good], (0, 2, 1)))
            dR.append(np.degrees(np.linalg.norm(rel.as_rotvec(), axis=1)))
    out["overlap_dC"] = np.concatenate(dC) if dC else np.zeros(0)
    out["overlap_dR_deg"] = np.concatenate(dR) if dR else np.zeros(0)
    return out


def load_chunk(p: Path) -> dict:
    z = np.load(p, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["stats"] = json.loads(str(d["stats"])) if "stats" in d else {}
    d["cam"] = str(d["cam"])
    for k in ("chunk",):
        if k in d:
            d[k] = int(d[k])
    return d


def export_track(cam: dict, calib: dict, merged: dict, out_dir: Path, world: dict | None = None, attached: bool = False,
                 frame_name: str = "sfm", stem: str | None = None) -> dict:
    """cameras/<cam>.npz in the demo's schema + additive keys. K is at the camera's native display resolution (the resolution
    of the owner's 2D pose JSONs), k1 is scale-invariant. Returns the arrays written."""
    n = merged["n"]
    K1 = _K_native(float(calib["f_work"]), float(calib["scale_to_native"]), tuple(calib["native_size"]))
    F = len(n)
    K = np.tile(K1, (F, 1, 1))
    ok = merged["ok"].astype(bool)
    reproj_native = merged["reproj_px"] * float(calib["scale_to_native"])
    arrays = dict(t=merged["t"], K=K, R=merged["R"], tvec=merged["tvec"], dist=np.full(F, float(calib["k1"])), ok=ok,
                  reproj_px=reproj_native, n_inliers=merged["n_inliers"].astype(np.int32), attached=np.bool_(attached),
                  axis_sd=np.full(F, np.nan, np.float32), recentred=np.zeros(F, bool),
                  camera_class=np.str_("handheld"), type_model=np.str_("fixed_focal"), type_held=np.zeros(F, bool),
                  # additive keys of the recap
                  pts=merged["pts"], frame_index=n.astype(np.int64), source=merged["source"], interpolated=merged["interp"],
                  localised=merged["localised"], frame=np.str_(frame_name), work_size=np.array(calib["work_size"]),
                  native_size=np.array(calib["native_size"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or cam["id"]
    np.savez(out_dir / f"{stem}.npz", **arrays)
    if world is not None:
        s, Rw, tw = float(world["s"]), np.asarray(world["R"]), np.asarray(world["t"])
        C_w = s * merged["C"] @ Rw.T + tw
        R_wc = merged["R"] @ Rw.T                      # world -> camera rotation (x_c = R_wc X_w + tvec_w)
        tvec_w = -np.einsum("fij,fj->fi", R_wc, C_w)
        np.savez(out_dir / f"{stem}_world.npz", t=merged["t"], K=K, R=R_wc, tvec=tvec_w, C=C_w, dist=arrays["dist"], ok=ok,
                 frame_index=n.astype(np.int64), pts=merged["pts"], interpolated=merged["interp"],
                 note=np.str_("world frame of the room model (Y up, metres): x_cam = R X_world + tvec; C = camera centre"))
    return arrays


# ------------------------------------------------------------------------------------------------ quality / coverage


def runs(mask: np.ndarray, t: np.ndarray, min_s: float = 0.0) -> list[list[float]]:
    """[t0, t1] of the runs where mask is True (reference seconds), runs shorter than min_s dropped."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            if t[j] - t[i] >= min_s:
                out.append([round(float(t[i]), 3), round(float(t[j]), 3)])
            i = j + 1
        else:
            i += 1
    return out


def camera_quality(cam: dict, calib: dict, merged: dict, world: dict | None = None, scale_m: float | None = None,
                   block_s: float = 10.0) -> dict:
    """Quality record of one camera track (all numbers from the merged chunks). `scale_m` = metres per SfM unit (world s)."""
    n = merged["n"]
    F = len(n)
    ok = merged["ok"].astype(bool)
    loc = merged["localised"].astype(bool)
    t = np.where(np.isfinite(merged["t"]), merged["t"], merged["pts"])
    sc = float(calib["scale_to_native"])
    q: dict = {"cam": cam["id"], "file": cam["file"], "frames": int(F), "frames_ok": int(ok.sum()),
               "ok_fraction": round(float(ok.mean()), 4) if F else 0.0,
               "localised_direct": int(loc.sum()), "interpolated": int(merged["interp"].sum()),
               "not_ok": int((~ok).sum()), "display_size": calib["native_size"], "work_size": calib["work_size"]}
    if F == 0:
        return {**q, "verdict": "unusable", "why": "no frames"}
    rp = merged["reproj_px"][loc & ok]
    ni = merged["n_inliers"][loc & ok]
    if len(rp):
        q["reproj_px_native"] = {"median": round(float(np.nanmedian(rp)) * sc, 3), "p90": round(float(np.nanpercentile(rp, 90)) * sc, 3),
                                 "p99": round(float(np.nanpercentile(rp, 99)) * sc, 3)}
        q["inliers"] = {"median": int(np.median(ni)), "p10": int(np.percentile(ni, 10)), "min": int(ni.min())}
    src = merged["source"]
    q["source_counts"] = {str(k): int((src == k).sum()) for k in np.unique(src)}
    q["gaps_over_0.5s"] = runs(~ok, t, 0.5)
    q["longest_gap_s"] = max([b - a for a, b in q["gaps_over_0.5s"]], default=0.0)
    # coverage per 10 s block of the master clock
    b0, b1 = float(np.floor(t[0] / block_s) * block_s), float(t[-1])
    blocks = []
    x = b0
    while x < b1:
        m = (t >= x) & (t < x + block_s)
        if m.any():
            blocks.append({"t0": round(x, 1), "frames": int(m.sum()), "ok": round(float(ok[m].mean()), 3)})
        x += block_s
    q["blocks"] = blocks
    # stitching agreement of the 4 s overlaps
    if len(merged.get("overlap_dC", [])):
        d = merged["overlap_dC"] * (scale_m or 1.0)
        q["chunk_overlap"] = {"n": int(len(d)), "unit": "m" if scale_m else "SfM units",
                              "centre_median": round(float(np.median(d)), 5), "centre_p95": round(float(np.percentile(d, 95)), 5),
                              "rot_median_deg": round(float(np.median(merged["overlap_dR_deg"])), 4),
                              "rot_p95_deg": round(float(np.percentile(merged["overlap_dR_deg"], 95)), 4)}
    # motion (handheld): centre path and rotation rate of the localised frames
    from scipy.spatial.transform import Rotation as Ro
    if loc.sum() > 3:
        Cl, tl, Rl = merged["C"][loc], t[loc], merged["R"][loc]
        dt = np.maximum(np.diff(tl), 1e-3)
        v = np.linalg.norm(np.diff(Cl, axis=0), axis=1) / dt * (scale_m or 1.0)
        w = np.degrees(np.linalg.norm(Ro.from_matrix(Rl[1:] @ np.transpose(Rl[:-1], (0, 2, 1))).as_rotvec(), axis=1)) / dt
        sel = dt < 0.3
        q["motion"] = {"centre_speed_median": round(float(np.median(v[sel])), 4), "centre_speed_p95": round(float(np.percentile(v[sel], 95)), 4),
                       "rot_rate_median_dps": round(float(np.median(w[sel])), 3), "rot_rate_p95_dps": round(float(np.percentile(w[sel], 95)), 3),
                       "centre_spread_m": round(float(np.linalg.norm(np.ptp(Cl, axis=0))) * (scale_m or 1.0), 3),
                       "unit_speed": "m/s" if scale_m else "SfM units/s"}
    q["intrinsics"] = {"f_native_px": round(float(calib["f_work"]) * sc, 2), "f_over_maxdim": round(float(calib["f_over_maxdim"]), 4),
                       "k1": round(float(calib["k1"]), 5), "anchors_used": calib.get("intrinsics", {}).get("n_used"),
                       "f_individual_rel_spread": calib.get("intrinsics", {}).get("f_individual_rel_spread"),
                       "f_individual_rel_range": calib.get("intrinsics", {}).get("f_individual_rel_range"),
                       "calibration_anchor_reproj_med_px_work": calib.get("anchor_reproj_fixed_med_px")}
    q["model"] = calib.get("model")
    return q


def verdict(q: dict, min_ok: float = 0.9, max_gap_s: float = 8.0, max_reproj_native: float = 4.0, in_world: bool = True) -> dict:
    """usable (>= 90 % of the camera's frames ok, no gap over 8 s, reprojection under 4 native px) | partial | unusable."""
    why = []
    if q.get("frames", 0) == 0 or q.get("frames_ok", 0) == 0:
        return {"verdict": "unusable", "why": ["no frame localised"]}
    if q["ok_fraction"] < min_ok:
        why.append(f"only {100 * q['ok_fraction']:.0f} % of the frames are localised")
    if q.get("longest_gap_s", 0) > max_gap_s:
        why.append(f"longest gap {q['longest_gap_s']:.1f} s")
    r = (q.get("reproj_px_native") or {}).get("median")
    if r is not None and r > max_reproj_native:
        why.append(f"reprojection {r:.1f} px")
    if not in_world:
        why.append("pose is in a sub-model frame, not in the room frame (attach pending)")
    if q["ok_fraction"] < 0.3:
        return {"verdict": "unusable", "why": why}
    return {"verdict": "usable" if not why else "partial", "why": why}


# ------------------------------------------------------------------------------------------------ epipolar check (identity-free)


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _undist(x, K, k1):
    """Normalised, undistorted coordinates of pixel points x (N,2) for a SIMPLE_RADIAL camera."""
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    d = (np.asarray(x, float) - [cx, cy]) / f
    u = d.copy()
    for _ in range(6):
        r2 = (u ** 2).sum(-1, keepdims=True)
        u = d / (1 + k1 * r2)
    return u


def epipolar_check(A: dict, B: dict, posesA: list, posesB: list, times: np.ndarray, conf: float = 0.5,
                   perturb_deg: tuple = (0.5, 1.0, 2.0), max_t_gap: float = 0.04, n_min_joints: int = 8) -> dict:
    """Identity-free epipolar consistency of two tracks.

    A, B: track dicts (t, K (F,3,3), R, tvec, dist, ok, frame_index) as exported; posesA / posesB: per video frame the (P, 17, 3)
    people of the owner's JSON. For every reference time in `times` where both cameras have an ok frame within max_t_gap: for each
    person of A the best person of B by the median symmetric epipolar distance (px, over joints with confidence > conf in both) - the
    people seen by both views must lie on each other's epipolar lines when the two poses are right. People that only one view sees have
    no counterpart (their best cost is large whatever the poses), so the figure to read is the SHARE under 3 / 6 px compared with the
    same share when B is turned by `perturb_deg` about its own horizontal / optical axis (the null models: the share must collapse as the
    error grows). Returns {"real": {...}, "pitch_0.5deg": {...}, "roll_0.5deg": {...}, ...}."""
    from scipy.spatial.transform import Rotation as Ro

    def frame_at(T, t):
        i = int(np.argmin(np.abs(T["t"] - t)))
        return i if abs(T["t"][i] - t) <= max_t_gap and T["ok"][i] else None

    def epi_costs(Ra, ta, Ka, ka, Rb, tb, Kb, kb, pa, pb):
        Rab = Rb @ Ra.T
        tab = tb - Rab @ ta
        E = _skew(tab) @ Rab
        costs = np.full((len(pa), len(pb)), np.inf)
        for ia, p in enumerate(pa):
            for ib, q in enumerate(pb):
                sel = (p[:, 2] > conf) & (q[:, 2] > conf)
                if sel.sum() < n_min_joints:
                    continue
                xa = _undist(p[sel, :2], Ka, ka)
                xb = _undist(q[sel, :2], Kb, kb)
                ha = np.c_[xa, np.ones(len(xa))]
                hb = np.c_[xb, np.ones(len(xb))]
                Ex = ha @ E.T                      # epipolar lines in B (normalised units)
                Etx = hb @ E                       # epipolar lines in A
                num = np.abs((hb * Ex).sum(1))
                d_b = num / np.maximum(np.hypot(Ex[:, 0], Ex[:, 1]), 1e-12)
                d_a = num / np.maximum(np.hypot(Etx[:, 0], Etx[:, 1]), 1e-12)
                costs[ia, ib] = float(np.median(0.5 * (d_a * Ka[0, 0] + d_b * Kb[0, 0])))
        return costs

    # Two cameras held at about the same height (both ~1.5 m) have nearly HORIZONTAL epipolar lines: a yaw error slides the people along
    # their lines and costs nothing, so the null models turn B about its own horizontal (pitch) and optical (roll) axes instead.
    levels = {"real": ("x", 0.0), **{f"pitch_{d}deg": ("x", float(d)) for d in perturb_deg},
              **{f"roll_{d}deg": ("z", float(d)) for d in perturb_deg}}
    res = {k: [] for k in levels}
    for t in times:
        ia, ib = frame_at(A, t), frame_at(B, t)
        if ia is None or ib is None:
            continue
        na, nb = int(A["frame_index"][ia]), int(B["frame_index"][ib])
        if na >= len(posesA) or nb >= len(posesB):
            continue
        pa, pb = posesA[na], posesB[nb]
        if len(pa) == 0 or len(pb) == 0:
            continue
        args = (A["R"][ia], A["tvec"][ia], A["K"][ia], float(A["dist"][ia]))
        Cb = -B["R"][ib].T @ B["tvec"][ib]
        for name, (axis, deg) in levels.items():
            ax = np.array([1.0, 0, 0]) if axis == "x" else np.array([0, 0, 1.0])
            Rb = Ro.from_rotvec(np.radians(deg) * ax).as_matrix() @ B["R"][ib]    # B turned about its own camera axis
            tb = -Rb @ Cb                                                           # the camera centre stays put
            c = epi_costs(*args, Rb, tb, B["K"][ib], float(B["dist"][ib]), pa, pb)
            best = np.min(c, axis=1)
            res[name].extend(best[np.isfinite(best)].tolist())
    out = {}
    for name, v in res.items():
        v = np.asarray(v)
        out[name] = {"n_persons": int(len(v)), "median_px": round(float(np.median(v)), 2) if len(v) else None,
                     "under_3px": round(float((v < 3).mean()), 3) if len(v) else None,
                     "under_6px": round(float((v < 6).mean()), 3) if len(v) else None,
                     "under_10px": round(float((v < 10).mean()), 3) if len(v) else None}
    return out


# ------------------------------------------------------------------------------------------------ review images


def project_points(X: np.ndarray, R: np.ndarray, tvec: np.ndarray, K: np.ndarray, k1: float):
    """(uv (N,2), depth (N,)) of SfM points through one pose, SIMPLE_RADIAL, OpenCV pixel centres."""
    Xc = X @ R.T + tvec
    z = Xc[:, 2]
    zz = np.where(z > 1e-6, z, 1e-6)
    x, y = Xc[:, 0] / zz, Xc[:, 1] / zz
    d = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], 1), z


def draw_room(img: np.ndarray, pts: np.ndarray, rgb: np.ndarray, R: np.ndarray, tvec: np.ndarray, K: np.ndarray, k1: float,
              world: dict | None, label: str, floor_half_m: float = 7.0, max_depth: float = 40.0) -> np.ndarray:
    """Draw the sparse room points (dots, tinted by depth), the 1 m floor grid (world y = 0) and the world axes on a frame."""
    out = img.copy()
    h, w = out.shape[:2]
    if world is not None:       # the glossy floor mirrors the room: points below the floor are reflections, not geometry
        yw = (float(world["s"]) * pts @ np.asarray(world["R"]).T + np.asarray(world["t"]))[:, 1]
        pts = pts[yw > -0.05]
    uv, z = project_points(pts, R, tvec, K, k1)
    ok = (z > 0.2) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    for (u, v), zz in zip(uv[ok], z[ok]):
        a = float(np.clip(1 - zz / max_depth, 0.1, 1))
        cv2.circle(out, (int(round(u)), int(round(v))), 2, (int(255 * (1 - a)), int(255 * a), 255), -1, cv2.LINE_AA)
    if world is not None:
        s, Rw, tw = float(world["s"]), np.asarray(world["R"]), np.asarray(world["t"])
        # world -> SfM: X_sfm = Rw^T (X_w - t) / s
        def w2s(Xw):
            return (Xw - tw) @ Rw / s
        g = np.arange(-floor_half_m, floor_half_m + 1e-6, 1.0)
        segs = [(np.array([x, 0, -floor_half_m]), np.array([x, 0, floor_half_m]), (0, 255, 255)) for x in g] + \
               [(np.array([-floor_half_m, 0, zc]), np.array([floor_half_m, 0, zc]), (0, 255, 255)) for zc in g]
        for a, b, col in segs:
            pa = np.linspace(a, b, 40)
            uvg, zg = project_points(w2s(pa), R, tvec, K, k1)
            good = (zg > 0.2) & (np.abs(uvg[:, 0]) < 3 * w) & (np.abs(uvg[:, 1]) < 3 * h)
            for i in range(len(pa) - 1):
                if good[i] and good[i + 1]:
                    cv2.line(out, tuple(np.round(uvg[i]).astype(int)), tuple(np.round(uvg[i + 1]).astype(int)), col, 2, cv2.LINE_AA)
        o = np.zeros(3)
        for ax, col in (((1, 0, 0), (0, 0, 255)), ((0, 1, 0), (0, 255, 0)), ((0, 0, 1), (255, 0, 0))):
            pa = np.array([o, np.array(ax, float)])
            uvg, zg = project_points(w2s(pa), R, tvec, K, k1)
            if (zg > 0.2).all():
                cv2.line(out, tuple(np.round(uvg[0]).astype(int)), tuple(np.round(uvg[1]).astype(int)), col, 3, cv2.LINE_AA)
    sz = max(0.5, w / 700)
    cv2.rectangle(out, (0, 0), (w, int(30 * sz)), (0, 0, 0), -1)
    cv2.putText(out, label, (6, int(21 * sz)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sz, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def read_frame(video: Path, pts: float) -> np.ndarray | None:
    """One frame at a source time (ffmpeg autorotate = display orientation)."""
    import subprocess
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, pts - 0.02):.3f}", "-i", str(video), "-frames:v", "1",
                        "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def contact_sheet(cam: dict, video: Path, track: dict, pts3d: np.ndarray, rgb: np.ndarray, world: dict | None, out: Path,
                  n_frames: int = 8, tile_h: int = 560) -> list[float]:
    """8 frames spread over the camera's ok frames with the room model drawn in; writes `out` (2 rows of 4). Returns the times."""
    ok = np.flatnonzero(track["ok"] & track["localised"])
    if len(ok) == 0:
        return []
    pick = ok[np.round(np.linspace(0, len(ok) - 1, n_frames)).astype(int)]
    tiles, times = [], []
    for i in pick:
        img = read_frame(video, float(track["pts"][i]))
        if img is None:
            continue
        lab = (f"cam {cam['id']}  master {track['t'][i]:.1f} s  frame {int(track['frame_index'][i])}  "
               f"inl {int(track['n_inliers'][i])}  rp {track['reproj_px'][i]:.2f}px")
        d = draw_room(img, pts3d, rgb, track["R"][i], track["tvec"][i], track["K"][i], float(track["dist"][i]), world, lab)
        s = tile_h / d.shape[0]
        tiles.append(cv2.resize(d, (int(d.shape[1] * s), tile_h), interpolation=cv2.INTER_AREA))
        times.append(float(track["t"][i]))
    if not tiles:
        return []
    cols = 4
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    wmax = max(sum(t.shape[1] for t in r) for r in rows)
    sheet = np.concatenate([np.pad(np.concatenate(r, 1), ((0, 0), (0, wmax - sum(t.shape[1] for t in r)), (0, 0))) for r in rows], 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return times
