"""Camera-tracking gate for the whole dance (larissa_kadu_full_{a,b,c,d}): is every camera tracked correctly over its
WHOLE recording, in particular the zooming phones (04, 05, 07, 09)?

Per camera, per segment and per 10-s block of the reference clock (evidence, nothing here changes a camera):

  localisation   raw ok-rate (cameras/cache/loc_<cam>.npz), exported ok-rate (cameras/<cam>.npz), median inliers
  static px      median static reprojection error of the exported track (attached cameras: sub-model consistency)
  dancers px     leave-one-out residual of the dancers' keypoints triangulated by the OTHER main-model cameras
                 (cameras.pipeline.dancer_check): finds a camera that registered consistently at a WRONG place
  focal          f / max image dimension (min, median, max), spikes outside zooms, zoom events (t0, t1, factor)
  depth guard    frames recentred by the zoom-depth guard (cameras/depth_guard.py; 'recentred' in cameras/<cam>.npz)
  smoothness     high-frequency jitter of the camera centre (mm) and rotation (deg) against a 0.5 s running median
  zoom evidence  the zoom measured directly in the VIDEO (cameras/video_zoom.py: similarity scale of successive
                 samples) against the zoom the exported cameras predict for the same samples (static map points of
                 the frame projected through both exported cameras): the unexplained zoom, in % over any 2 s

The gate rules are in RULES (printed with the table). All numbers are written to
work/larissa_kadu_full/review/gate_cameras/data/*.json and the images next to them.

    python -m dancecap.cameras.gate_cameras table [--segs a,b,c,d]      # block tables + focal stats + dancer LOO
    python -m dancecap.cameras.gate_cameras zoom                        # video vs exported zoom (needs video_zoom data)
    python -m dancecap.cameras.gate_cameras plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]            # .../dancecap
WORK = ROOT / "work"
OUT = WORK / "larissa_kadu_full" / "review" / "gate_cameras"
SEGS = ("a", "b", "c", "d")
CAMS = ("01", "02", "03", "04", "05", "06", "07", "09")
ZOOM_CAMS = ("04", "05", "07", "09")                  # user-confirmed (takes/larissa_kadu_full*.toml [cameras])
NO_ZOOM_CAMS = ("01", "02", "03", "06")
ATTACHED = ("05", "06")
BASELINE = "larissa_kadu_38"
BLOCK_S = 10.0
CACHE_TAG = ""          # data cache suffix of dancers_loo_sync*; "_before" together with BASE = "cameras/s2_gate_backup"
BASE = "cameras"        # "cameras/s2_gate_backup" reads the exports as they were BEFORE the attach_zoom fix (those files only)

RULES = {
    "ok_exported": "exported ok-rate >= 0.95 per block (attached cameras >= 0.90)",
    "static_px": "median static reprojection <= 4 px (stage_smooth's max_reproj_px); attached cameras: informative",
    "inliers": "median inliers >= 60 (main-model cameras)",
    "dancers_px": "median dancers' LOO residual <= max(2.0 x the camera's median in larissa_kadu_38, 25 px)",
    "focal_spikes": "no focal spike outside a zoom; no-zoom cameras: focal within 1 % of larissa_kadu_38's median",
    "zoom_tracked": "unexplained zoom (video minus exported-camera prediction) <= 5 % over any 2 s window",
    "centre_jitter": "centre jitter p95 <= max(3 x larissa_kadu_38's p95 for the camera, 30 mm)",
}


def seg_take(seg: str) -> str:
    return BASELINE if seg == "38" else f"larissa_kadu_full_{seg}"


def seg_work(seg: str) -> Path:
    return WORK / seg_take(seg)


# ------------------------------------------------------------------------------------------------ loading

def load_cam(work: Path, cam: str) -> dict | None:
    """Exported track (cameras/<cam>.npz), raw localisation (cache/loc_<cam>.npz without correspondences) and the
    frame geometry. None when the camera has no track in this take."""
    p = work / BASE / f"{cam}.npz"
    if not p.exists() and BASE != "cameras":
        p = work / "cameras" / f"{cam}.npz"
    if not p.exists():
        return None
    e = dict(np.load(p, allow_pickle=True))
    q = work / BASE / "cache" / f"loc_{cam}.npz"
    if not q.exists():
        q = work / "cameras" / "cache" / f"loc_{cam}.npz"
    raw = {}
    if q.exists():
        with np.load(q, allow_pickle=True) as z:
            raw = {k: z[k] for k in z.files if k not in ("corr_xy", "corr_xyz")}
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    w, h = int(meta["width"]), int(meta["height"])
    t = e["t"]
    f = e["K"][:, 0, 0]
    C = -np.einsum("fji,fj->fi", e["R"], e["tvec"])
    return {"cam": cam, "t": t, "K": e["K"], "R": e["R"], "tvec": e["tvec"], "k1": e["dist"], "ok": e["ok"].astype(bool),
            "reproj_px": e["reproj_px"], "n_inliers": e["n_inliers"], "attached": bool(e["attached"]),
            "recentred": e.get("recentred", np.zeros(len(t), bool)).astype(bool),
            "f": f, "C": C, "w": w, "h": h, "maxdim": max(w, h), "raw": raw,
            "ok_raw": raw["ok"].astype(bool) if "ok" in raw else e["ok"].astype(bool),
            "f_raw": raw.get("f", f), "n_inliers_raw": raw.get("n_inliers", e["n_inliers"]),
            "reproj_raw": raw.get("reproj_px", e["reproj_px"]),
            "camera_type": str(e["type_model"]) if "type_model" in e else None,
            "offset": float(meta["offset"]), "slope": float(meta["slope"]),
            "rotation": int(meta.get("rotation") or 0)}


def sim3_scale(work: Path) -> float:
    return float(json.loads((work / "cameras" / "world_from_sfm.json").read_text())["s"])


def running_median(x: np.ndarray, n: int) -> np.ndarray:
    """Centred running median over n samples (edge-padded), per column."""
    from scipy.ndimage import median_filter
    x = np.asarray(x, float)
    size = (n,) + (1,) * (x.ndim - 1)
    return median_filter(x, size=size, mode="nearest")


def jitter(d: dict, s_world: float, win_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(centre jitter mm, rotation jitter deg) per frame: distance of the exported pose to its running median."""
    from scipy.spatial.transform import Rotation
    t = d["t"]
    n = max(3, int(round(win_s / max(np.median(np.diff(t)), 1e-3))) | 1)
    C = d["C"]
    Cm = running_median(C, n)
    jc = np.linalg.norm(C - Cm, axis=1) * s_world * 1000.0
    rv = Rotation.from_matrix(d["R"]).as_rotvec()
    # rotation vectors are continuous for small motions; compare to the running median of the rotation vectors
    rvm = running_median(rv, n)
    jr = np.degrees(np.linalg.norm(rv - rvm, axis=1))
    return jc, jr


def focal_report(d: dict) -> dict:
    """pipeline.focal_stats of the exported track (zoom events, spikes, plausibility)."""
    from .pipeline import focal_stats
    tr = {"f": d["f"], "f_raw": d["f_raw"], "ok_raw": d["ok_raw"], "t": d["t"], "maxdim": d["maxdim"]}
    return focal_stats(tr)


# ------------------------------------------------------------------------------------------------ dancers LOO

def sync_deltas(work: Path) -> dict[str, float]:
    """{cam: delta s} of the take's sync/offsets.json (corrected reference time = t + delta; cameras whose offset is not
    used get 0): the dancers must be compared at the same INSTANT, an unsynced check reads a 200 ms offset as a camera error."""
    p = Path(work) / "sync" / "offsets.json"
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    return {c: float(v["delta"]) for c, v in (doc.get("offsets") or {}).items() if v.get("use", True) and "delta" in v}


def dancer_residuals(work: Path, cams: list[str] | None = None, step: int = 3, kps: tuple[int, ...] = tuple(range(5, 17)),
                     min_conf: float = 0.5, max_dt: float = 0.034, log=print, sync: bool = True,
                     voters: list[str] | None = None) -> dict[str, np.ndarray]:
    """{cam: (F,) per-frame median dancers' residual px (NaN without observations; only every `step`-th frame)} of the
    EXPORTED tracks: COCO keypoints triangulated at that instant by the OTHER main-model cameras that record then
    (per-instant selection - cameras.pipeline.dancer_check intersects the time ranges of all cameras and so only
    sees the span every camera shares), projected into this camera with its exported pose, against the detections.
    Attached cameras never vote. sync=True (default) compares the cameras at their sync-corrected times (t + delta of
    sync/offsets.json); dancer_check applies no sync offset."""
    from .report import project_cv
    from .world import dancer_keypoints, triangulate_robust
    rep = json.loads((work / "cameras" / "report.json").read_text())
    main = sorted(c for c, v in rep["cams"].items() if v.get("in_main_model") and (voters is None or c in voters))
    D = {c: load_cam(work, c) for c in CAMS}
    D = {c: d for c, d in D.items() if d is not None}
    delta = sync_deltas(work) if sync else {}
    TC = {c: d["t"] + delta.get(c, 0.0) for c, d in D.items()}      # sync-corrected frame times
    KP = {c: dancer_keypoints(work, c, len(d["t"])) for c, d in D.items()}
    PEOPLE = {}
    for c in D:
        pz = np.load(work / "people" / f"{c}.npz")
        PEOPLE[c] = (pz["valid"], pz["kpts"])
    out = {}
    for cam in cams or sorted(D):
        d = D[cam]
        others = [c for c in main if c != cam and c in D]
        pf = np.full(len(d["t"]), np.nan)
        pfree = np.full(len(d["t"]), np.nan)     # label-free: the nearest DETECTED PERSON instead of the labelled one
        for i in range(0, len(d["t"]), step):
            if not d["ok"][i]:
                continue
            t = TC[cam][i]
            sel = []
            for c in others:
                o, to = D[c], TC[c]
                j = int(np.searchsorted(to, t))
                j = min(max(j, 0), len(to) - 1)
                if j > 0 and abs(to[j - 1] - t) < abs(to[j] - t):
                    j -= 1
                if abs(to[j] - t) <= max_dt and o["ok"][j]:
                    sel.append((c, j))
            if len(sel) < 2:
                continue
            res = []
            free = []
            valid, kpts = PEOPLE[cam]
            ppl = kpts[i][valid[i]] if i < len(kpts) else np.zeros((0, 17, 3))
            for role in ("lead", "follow"):
                proj = np.full((17, 2), np.nan)
                for k in kps:
                    det = KP[cam][role][i, k]
                    v = [(c, j) for c, j in sel if KP[c][role][j, k, 2] >= min_conf]
                    if len(v) < 2:
                        continue
                    X, _, nv = triangulate_robust(
                        np.array([KP[c][role][j, k, :2] for c, j in v], float), np.array([D[c]["K"][j] for c, j in v]),
                        np.array([D[c]["k1"][j] for c, j in v]), np.array([D[c]["R"][j] for c, j in v]),
                        np.array([D[c]["tvec"][j] for c, j in v]))
                    if X is None or nv < 2:
                        continue
                    p, vis = project_cv(X[None], d["K"][i], d["R"][i], d["tvec"][i], float(d["k1"][i]))
                    if vis[0]:
                        proj[k] = p[0]
                    if det[2] >= min_conf:
                        res.append(float(np.linalg.norm(p[0] - det[:2])) if vis[0] else 1e4)
                have = np.isfinite(proj[:, 0])
                if have.sum() >= 4 and len(ppl):
                    best = np.inf
                    for person in ppl:
                        m = have & (person[:, 2] >= min_conf)
                        if m.sum() >= 4:
                            best = min(best, float(np.median(np.linalg.norm(proj[m] - person[m, :2], axis=1))))
                    if np.isfinite(best):
                        free.append(best)
            if len(res) >= 4:
                pf[i] = float(np.median(res))
            if free:
                pfree[i] = float(np.median(free))
        out[cam] = pf
        out[f"{cam}:free"] = pfree
        log(f"[gate_cameras] dancers LOO {Path(work).name} cam{cam}: {np.isfinite(pf).sum()} frames, median "
            f"{np.nanmedian(pf):.1f} px (label-free {np.nanmedian(pfree):.1f})", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ block table

def block_edges(t: np.ndarray, block_s: float = BLOCK_S) -> np.ndarray:
    a = np.floor(t.min() / block_s) * block_s
    b = np.ceil(t.max() / block_s) * block_s
    return np.arange(a, b + 1e-9, block_s)


def rows_for(seg: str, cam: str, d: dict, dpx: np.ndarray | None, s_world: float, base: dict | None,
             dfree: np.ndarray | None = None) -> list[dict]:
    """One dict per 10-s block of the reference clock that holds frames of this camera in this segment."""
    t = d["t"]
    jc, jr = jitter(d, s_world)
    fm = d["f"] / d["maxdim"]
    rows = []
    for a in block_edges(t):
        b = a + BLOCK_S
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        ok, okr = d["ok"][m], d["ok_raw"][m]
        okm = m & d["ok"]
        r = {"seg": seg, "cam": cam, "t0": float(a), "t1": float(b), "frames": int(m.sum()),
             "span_s": round(float(t[m].max() - t[m].min()), 2),
             "ok_raw": round(float(okr.mean()), 4), "ok_exported": round(float(ok.mean()), 4),
             "inliers_med": float(np.median(d["n_inliers_raw"][m & d["ok_raw"]])) if (m & d["ok_raw"]).any() else None,
             "static_px_med": round(float(np.nanmedian(d["reproj_px"][okm])), 3) if okm.any() else None,
             "static_px_p90": round(float(np.nanpercentile(d["reproj_px"][okm], 90)), 3) if okm.any() else None,
             "f_over_maxdim": [round(float(fm[m].min()), 3), round(float(np.median(fm[m])), 3), round(float(fm[m].max()), 3)],
             "recentred": int(d["recentred"][m].sum()),
             "centre_jitter_mm_med": round(float(np.median(jc[m])), 2), "centre_jitter_mm_p95": round(float(np.percentile(jc[m], 95)), 2),
             "rot_jitter_deg_p95": round(float(np.percentile(jr[m], 95)), 3)}
        if dpx is not None:
            v = dpx[m]
            v = v[np.isfinite(v)]
            r["dancers_px_med"] = round(float(np.median(v)), 2) if len(v) >= 10 else None
            r["dancers_px_n"] = int(len(v))
        if dfree is not None:
            v = dfree[m]
            v = v[np.isfinite(v)]
            r["dancers_free_px_med"] = round(float(np.median(v)), 2) if len(v) >= 10 else None
        # the camera is judged on the labelled residual (the identity gate has vetoed the wrong-person frames: their
        # box is None and the frame is skipped); the label-free one (nearest detected person) stands in where no labelled
        # frame is left (cam07 in segment d)
        r["dancers_best_px_med"] = r.get("dancers_px_med") if r.get("dancers_px_med") is not None else r.get("dancers_free_px_med")
        rows.append(r)
    return rows


def judge(r: dict, cam: str, attached: bool, base: dict | None) -> list[str]:
    """Names of the RULES a block row violates (the per-block ones)."""
    bad = []
    if r["frames"] < 15:
        return bad                      # too few frames to judge (a segment edge)
    if r["ok_exported"] < (0.90 if attached else 0.95):
        bad.append("ok_exported")
    if not attached and r["static_px_med"] is not None and r["static_px_med"] > 4.0:
        bad.append("static_px")
    if not attached and r["inliers_med"] is not None and r["inliers_med"] < 60:
        bad.append("inliers")
    # the camera is judged on the label-free residual (nearest detected person to the projected dancers): a wrong
    # lead / follow label (cam07 in segment d: foreground spectators) is the identity gate's finding, not a camera's
    dm = r.get("dancers_free_px_med") if r.get("dancers_free_px_med") is not None else r.get("dancers_px_med")
    if dm is not None:
        lim = max(2.0 * (base or {}).get("dancers_free_px_med", (base or {}).get("dancers_px_med", 12.0)), 25.0)
        if dm > lim:
            bad.append("dancers_px")
    if r.get("dancers_px_med") is not None and r.get("dancers_free_px_med") is not None             and r["dancers_px_med"] > max(3.0 * r["dancers_free_px_med"], 40.0):
        bad.append("labels")
    lim_j = max(3.0 * (base or {}).get("centre_jitter_p95", 10.0), 30.0)
    if not attached and r["centre_jitter_mm_p95"] > lim_j:
        bad.append("centre_jitter")
    return bad


def cached_loo(seg: str) -> dict[str, np.ndarray]:
    """dancer_residuals (sync-corrected) of a segment, cached in data/dancers_loo_sync<tag>_<seg>.npz (tag: "" after
    the fix, "_before" for the exports before it)."""
    p = OUT / "data" / f"dancers_loo_sync{CACHE_TAG}_{seg}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    r = dancer_residuals(seg_work(seg))
    np.savez(p, **r)
    return r


def excluded_spans(seg: str) -> list[dict]:
    """Camera spans of a segment that were vetoed out of the fit (data/vetoes_gate_cameras.json, the coordinator's
    decision of 2026-10-08 13:00): {cam, t0, t1, why}."""
    p = OUT / "data" / "vetoes_gate_cameras.json"
    if not p.exists():
        return []
    return [e for e in json.loads(p.read_text()).get("entries", []) if e["seg"] == seg]


def baseline_stats(s_world_for: dict | None = None) -> dict:
    """Per camera medians in larissa_kadu_38 (the accepted take): dancers' LOO residual (report.json), centre jitter."""
    work = seg_work("38")
    s = sim3_scale(work)
    dpx = cached_loo("38")
    out = {}
    for c in CAMS:
        d = load_cam(work, c)
        if d is None:
            continue
        jc, jr = jitter(d, s)
        e = {"dancers_px_med": float(np.nanmedian(dpx[c])) if c in dpx else float("nan"),
             "dancers_free_px_med": float(np.nanmedian(dpx[f"{c}:free"])) if f"{c}:free" in dpx else float("nan"),
             "dancers_best_px_med": (float(np.nanmedian(dpx[c])) if c in dpx and np.isfinite(np.nanmedian(dpx[c]))
                                     else float(np.nanmedian(dpx[f"{c}:free"])) if f"{c}:free" in dpx else float("nan")),
             "centre_jitter_p95": float(np.percentile(jc, 95)), "rot_jitter_deg_p95": float(np.percentile(jr, 95)),
             "f_median": float(np.median(d["f"])), "f_over_maxdim_median": float(np.median(d["f"]) / d["maxdim"]),
             "frames": int(len(d["t"])), "t0": float(d["t"][0]), "t1": float(d["t"][-1])}
        out[c] = e
    return out


def run_table(segs: tuple[str, ...] = SEGS, use_dancers: bool = True) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    base = baseline_stats()
    result = {"rules": RULES, "baseline": base, "segments": {}}
    for seg in segs:
        work = seg_work(seg)
        s = sim3_scale(work)
        dpx = cached_loo(seg) if use_dancers else {}
        for ex in excluded_spans(seg):                     # spans vetoed out of the fit: not judged any more
            d_ex = load_cam(work, ex["cam"])
            for key in (ex["cam"], ex["cam"] + ":free"):
                if d_ex is not None and key in dpx:
                    dpx[key] = np.where((d_ex["t"] >= ex["t0"] - 1 / 60) & (d_ex["t"] <= ex["t1"] + 1 / 60), np.nan, dpx[key])
        seg_out = {}
        for cam in CAMS:
            d = load_cam(work, cam)
            if d is None:
                continue
            rows = rows_for(seg, cam, d, dpx.get(cam), s, base.get(cam), dpx.get(f"{cam}:free"))
            for r in rows:
                r["violations"] = judge(r, cam, d["attached"], base.get(cam))
            fr = focal_report(d)
            seg_out[cam] = {"attached": d["attached"], "camera_type": d["camera_type"], "frames": int(len(d["t"])),
                            "t0": float(d["t"][0]), "t1": float(d["t"][-1]), "focal": fr, "blocks": rows,
                            "recentred": int(d["recentred"].sum()), "ok_raw": round(float(d["ok_raw"].mean()), 4),
                            "ok_exported": round(float(d["ok"].mean()), 4)}
        result["segments"][seg] = seg_out
        print(f"[gate_cameras] segment {seg}: {len(seg_out)} cameras", flush=True)
    (OUT / "data" / f"table{CACHE_TAG}.json").write_text(json.dumps(result, indent=1, default=_np))
    return result


# ------------------------------------------------------------------------------------------------ overlays

COCO_EDGES = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def decode_frame(video: Path, pts: float, size: tuple[int, int] | None = None) -> np.ndarray | None:
    """One BGR frame at presentation time `pts`, autorotated exactly like ingest (ffmpeg), at its native size (or
    resized to size = (w, h))."""
    import subprocess

    import cv2
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(pts, 0.0):.4f}", "-i", str(video), "-frames:v", "1", "-f",
           "image2pipe", "-vcodec", "png", "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=120, creationflags=0x08000000
                       if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
    if p.returncode != 0 or not p.stdout:
        return None
    img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is not None and size is not None and (img.shape[1], img.shape[0]) != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return img


def dancers_at(work: Path, cams: list[str], t: float, exclude: str | None = None, min_conf: float = 0.5) -> dict:
    """{role: (17,3) SfM-frame keypoints, NaN unknown} at reference time t, triangulated by the main-model cameras
    `cams` (minus `exclude`) from the frames nearest to t (leave-one-out for the camera under test)."""
    from .world import dancer_keypoints, triangulate_robust
    rep = json.loads((work / "cameras" / "report.json").read_text())
    main = [c for c in cams if c != exclude and rep["cams"].get(c, {}).get("in_main_model")]
    sel = []
    for c in main:
        d = load_cam(work, c)
        i = int(np.argmin(np.abs(d["t"] - t)))
        if abs(d["t"][i] - t) <= 0.04 and d["ok"][i]:
            sel.append((c, i, d))
    out = {r: np.full((17, 3), np.nan) for r in ("lead", "follow")}
    if len(sel) < 2:
        return out
    kps = {c: dancer_keypoints(work, c, len(d["t"])) for c, i, d in sel}
    for role in out:
        for k in range(17):
            v = [(c, i, d) for c, i, d in sel if kps[c][role][i, k, 2] >= min_conf]
            if len(v) < 2:
                continue
            xy = np.array([kps[c][role][i, k, :2] for c, i, d in v], float)
            X, _, nv = triangulate_robust(xy, np.array([d["K"][i] for c, i, d in v]), np.array([d["k1"][i] for c, i, d in v]),
                                          np.array([d["R"][i] for c, i, d in v]), np.array([d["tvec"][i] for c, i, d in v]))
            if X is not None and nv >= 2:
                out[role][k] = X
    return out


def map_points(work: Path, max_pts: int = 20000) -> np.ndarray:
    """Sparse points of the static map (SfM frame), subsampled."""
    import pycolmap
    m = pycolmap.Reconstruction(str(work / "cameras" / "sfm"))
    P = np.array([p.xyz for p in m.points3D.values()]).reshape(-1, 3)
    if len(P) > max_pts:
        P = P[np.random.default_rng(0).choice(len(P), max_pts, replace=False)]
    return P


def overlay_frame(seg: str, cam: str, t: float, src_path: Path, pts: np.ndarray | None = None, label: str = "",
                  loo: bool = True, scale: float = 1.0) -> np.ndarray | None:
    """The video frame nearest to t with the room's static map points (cyan) and the dancers' keypoints triangulated
    by the OTHER main-model cameras (magenta; ankles large) projected through the EXPORTED camera of this segment.
    Detected 2D keypoints (green crosses) and the localisation's inlier features (yellow dots) are drawn for comparison."""
    import cv2

    from .report import project_cv
    from .world import dancer_keypoints
    work = seg_work(seg)
    d = load_cam(work, cam)
    i = int(np.argmin(np.abs(d["t"] - t)))
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    img = decode_frame(src_path, float(meta["frames"][i]["pts"]), (d["w"], d["h"]))
    if img is None:
        return None
    K, R, tv, k1 = d["K"][i], d["R"][i], d["tvec"][i], float(d["k1"][i])
    if pts is None:
        pts = map_points(work)
    uv, vis = project_cv(pts, K, R, tv, k1)
    inb = vis & (uv[:, 0] >= 0) & (uv[:, 0] < d["w"]) & (uv[:, 1] >= 0) & (uv[:, 1] < d["h"])
    lw = max(1, round(img.shape[1] / 480))
    for u, v in uv[inb][::2]:
        cv2.circle(img, (int(u), int(v)), lw, (255, 255, 0), -1, cv2.LINE_AA)
    q = work / "cameras" / "cache" / f"loc_{cam}.npz"
    if q.exists() and not d["attached"]:
        with np.load(q) as z:
            a, b = int(z["corr_off"][i]), int(z["corr_off"][i + 1])
            xy, X = z["corr_xy"][a:b], z["corr_xyz"][a:b]
        sel = np.arange(len(xy))[::max(1, len(xy) // 150)]
        pr, v2 = project_cv(X[sel], K, R, tv, k1)
        for (u, v), (pu, pv), ok in zip(xy[sel], pr, v2):
            if ok:
                cv2.line(img, (int(u), int(v)), (int(pu), int(pv)), (0, 0, 255), lw, cv2.LINE_AA)
                cv2.circle(img, (int(u), int(v)), lw + 1, (0, 255, 255), -1, cv2.LINE_AA)
    D = dancers_at(work, list(CAMS), float(d["t"][i]), exclude=cam if loo else None)
    kp2 = dancer_keypoints(work, cam, len(d["t"]))
    for role, col in (("lead", (255, 0, 255)), ("follow", (255, 128, 255))):
        P = D[role]
        ok = np.isfinite(P).all(1)
        if not ok.any():
            continue
        p2 = np.full((17, 2), np.nan)
        p2[ok], vv = project_cv(P[ok], K, R, tv, k1)
        for a_, b_ in COCO_EDGES:
            if ok[a_] and ok[b_] and np.isfinite(p2[[a_, b_]]).all() and abs(p2[[a_, b_]]).max() < 1e5:
                cv2.line(img, tuple(int(x) for x in p2[a_]), tuple(int(x) for x in p2[b_]), col, lw + 1, cv2.LINE_AA)
        for k in np.flatnonzero(ok):
            if np.isfinite(p2[k]).all() and abs(p2[k]).max() < 1e5:
                cv2.circle(img, (int(p2[k, 0]), int(p2[k, 1])), (5 if k in (15, 16) else 3) * lw, col, lw + 1, cv2.LINE_AA)
        det = kp2[role][i]
        for k in range(17):
            if det[k, 2] >= 0.5:
                cv2.drawMarker(img, (int(det[k, 0]), int(det[k, 1])), (0, 255, 0), cv2.MARKER_CROSS, 8 * lw, lw)
    s = max(1.0, img.shape[1] / 480)
    txt = f"{label or f'{seg} cam{cam}'} t={d['t'][i]:.2f}s f/md={d['f'][i] / d['maxdim']:.3f}"
    cv2.putText(img, txt, (6, int(22 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 0, 0), max(3, int(3 * s)), cv2.LINE_AA)
    cv2.putText(img, txt, (6, int(22 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (255, 255, 255), max(1, int(s)), cv2.LINE_AA)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


# ------------------------------------------------------------------------------------------------ seam consistency

OVERLAPS = (("a", "38"), ("38", "b"), ("b", "c"), ("c", "d"))


def overlap_consistency(pairs=OVERLAPS) -> dict:
    """The same recording localised in two neighbouring takes (same map, same frames): per camera over the overlap the
    centre difference (mm), rotation difference (deg) and focal ratio (%). A camera tracked correctly gives the same
    camera twice; a mis-registration in one of the takes shows here. Attached cameras (05, 06) also report the
    change of their constant centre between takes."""
    from scipy.spatial.transform import Rotation
    out = {}
    for s1, s2 in pairs:
        w1, w2 = seg_work(s1), seg_work(s2)
        sw = sim3_scale(w1)
        for cam in CAMS:
            a, b = load_cam(w1, cam), load_cam(w2, cam)
            if a is None or b is None:
                continue
            lo, hi = max(a["t"][0], b["t"][0]), min(a["t"][-1], b["t"][-1])
            if hi - lo < 1.0:
                continue
            ia = np.flatnonzero((a["t"] >= lo) & (a["t"] <= hi))
            j = np.clip(np.searchsorted(b["t"], a["t"][ia]), 1, len(b["t"]) - 1)
            j = np.where(np.abs(b["t"][j - 1] - a["t"][ia]) < np.abs(b["t"][j] - a["t"][ia]), j - 1, j)
            keep = (np.abs(b["t"][j] - a["t"][ia]) < 0.02) & a["ok"][ia] & b["ok"][j]
            ia, j = ia[keep], j[keep]
            if len(ia) < 10:
                continue
            dC = np.linalg.norm(a["C"][ia] - b["C"][j], axis=1) * sw * 1000.0
            dR = np.degrees(np.linalg.norm(Rotation.from_matrix(a["R"][ia] @ np.transpose(b["R"][j], (0, 2, 1))).as_rotvec(), axis=1))
            df = 100 * (a["f"][ia] / b["f"][j] - 1)
            out[f"{s1}|{s2}:{cam}"] = {
                "t0": round(float(a["t"][ia].min()), 2), "t1": round(float(a["t"][ia].max()), 2), "frames": int(len(ia)),
                "centre_mm_med": round(float(np.median(dC)), 1), "centre_mm_p95": round(float(np.percentile(dC, 95)), 1),
                "rot_deg_med": round(float(np.median(dR)), 3), "rot_deg_p95": round(float(np.percentile(dR, 95)), 3),
                "focal_pct_med": round(float(np.median(df)), 2), "focal_pct_p95_abs": round(float(np.percentile(np.abs(df), 95)), 2),
                "attached": a["attached"] or b["attached"]}
    return out


# ------------------------------------------------------------------------------------------------ zoom evidence

def detect_video_events(t: np.ndarray, dlog: np.ndarray, ok: np.ndarray, step_s: float, min_total: float = 0.05,
                        min_rate: float = 0.03, join_s: float = 0.5) -> list[dict]:
    """Zoom events seen in the VIDEO alone: runs of successive pairs whose (3-pair smoothed) log-scale rate is above
    min_rate per second in one direction (gaps up to join_s joined), total change >= min_total (5 %: a handheld
    no-zoom phone wanders by < 4 % over 2 s with this estimator, camera 06 and 01-03 are the controls)."""
    d = np.where(ok & np.isfinite(dlog), dlog, 0.0)
    sm = np.convolve(d, np.ones(3) / 3, mode="same")
    rate = sm / step_s
    ev, i, n = [], 0, len(d)
    while i < n:
        if abs(rate[i]) < min_rate:
            i += 1
            continue
        sgn = np.sign(rate[i])
        j, last = i, i
        while j + 1 < n and (j + 1 - last) * step_s <= join_s:
            j += 1
            if np.sign(rate[j]) == sgn and abs(rate[j]) >= min_rate:
                last = j
        tot = float(d[i:last + 1].sum())
        if abs(tot) >= min_total:
            ev.append({"t0": round(float(t[i] - step_s / 2), 3), "t1": round(float(t[last] + step_s / 2), 3),
                       "factor": round(float(np.exp(tot)), 4), "change_pct": round(100 * float(np.exp(tot) - 1), 1)})
        i = last + 1
    return ev


def zoom_compare(seg: str, cam: str, vid: dict, step: int = 1) -> dict | None:
    """Per video pair inside the segment's frames: measured log scale (video), the exported cameras' prediction from
    static map points and d log f of the exported focal. None if the segment has no such camera / cache."""
    from .video_zoom import predicted_scale_pairs
    work = seg_work(seg)
    d = load_cam(work, cam)
    q = work / BASE / "cache" / f"loc_{cam}.npz"
    if not q.exists():
        q = work / "cameras" / "cache" / f"loc_{cam}.npz"
    if d is None or not q.exists():
        return None
    t = d["t"]
    t0 = vid["offset"] + vid["slope"] * vid["pts0"]
    t1 = vid["offset"] + vid["slope"] * vid["pts1"]
    dt = float(np.median(np.diff(t)))
    sel = (t0 >= t[0] - 0.5 * dt) & (t1 <= t[-1] + 0.5 * dt)
    if sel.sum() < 5:
        return None
    i0 = np.clip(np.searchsorted(t, t0[sel]), 0, len(t) - 1)
    i0 = np.where(np.abs(t[np.maximum(i0 - 1, 0)] - t0[sel]) < np.abs(t[i0] - t0[sel]), np.maximum(i0 - 1, 0), i0)
    i1 = np.clip(np.searchsorted(t, t1[sel]), 0, len(t) - 1)
    i1 = np.where(np.abs(t[np.maximum(i1 - 1, 0)] - t1[sel]) < np.abs(t[i1] - t1[sel]), np.maximum(i1 - 1, 0), i1)
    with np.load(q, allow_pickle=True) as z:
        corr = {"corr_off": z["corr_off"], "corr_xyz": z["corr_xyz"]}
    tr = {"K": d["K"], "R": d["R"], "tvec": d["tvec"], "k1": d["k1"]}
    pred = predicted_scale_pairs(tr, corr, i0, i1, far=1e5 if d["attached"] else None)   # (moving centre: no fake parallax)
    lf = np.log(d["f"])
    return {"seg": seg, "cam": cam, "t": vid["t"][sel], "measured": vid["dlog"][sel], "ok": vid["ok"][sel],
            "predicted": pred, "dlogf": lf[i1] - lf[i0], "i0": i0, "i1": i1, "frame_ok": d["ok"][i0] & d["ok"][i1],
            "inl": vid["inl"][sel]}


def window_sums(t: np.ndarray, x: np.ndarray, win_s: float) -> np.ndarray:
    """Sum of x over the window [t, t + win_s] for every sample (NaN counted as 0)."""
    x = np.nan_to_num(x)
    c = np.r_[0.0, np.cumsum(x)]
    j = np.searchsorted(t, t + win_s, side="right")
    return c[j] - c[np.arange(len(t))]


def zoom_metrics(zc: dict, win_s: float = 2.0) -> dict:
    """Unexplained zoom = measured - predicted, summed over sliding windows; per 10-s block maxima."""
    ok = zc["ok"] & np.isfinite(zc["predicted"])
    resid = np.where(ok, zc["measured"] - zc["predicted"], 0.0)
    meas = np.where(zc["ok"], zc["measured"], 0.0)
    ws = window_sums(zc["t"], resid, win_s)
    wm = window_sums(zc["t"], meas, win_s)
    wp = window_sums(zc["t"], np.where(ok, zc["predicted"], 0.0), win_s)
    wf = window_sums(zc["t"], zc["dlogf"], win_s)
    blocks = []
    for a in block_edges(zc["t"]):
        m = (zc["t"] >= a) & (zc["t"] < a + BLOCK_S)
        if not m.any():
            continue
        k = int(np.argmax(np.abs(ws[m])))
        blocks.append({"t0": float(a), "t1": float(a + BLOCK_S), "pairs": int(m.sum()),
                       "video_zoom_max_2s_pct": round(100 * float(np.exp(np.abs(wm[m]).max()) - 1), 1),
                       "pred_zoom_max_2s_pct": round(100 * float(np.exp(np.abs(wp[m]).max()) - 1), 1),
                       "focal_zoom_max_2s_pct": round(100 * float(np.exp(np.abs(wf[m]).max()) - 1), 1),
                       "unexplained_max_2s_pct": round(100 * float(np.exp(np.abs(ws[m]).max()) - 1), 1),
                       "unexplained_at_t": round(float(zc["t"][m][k]), 2)})
    hi = zc["ok"] & (np.abs(zc["measured"]) > 0.004)
    return {"blocks": blocks, "unexplained_max_2s_pct": round(100 * float(np.exp(np.abs(ws).max()) - 1), 1),
            "corr_measured_vs_predicted": round(float(np.corrcoef(meas[ok], np.where(ok, zc["predicted"], 0)[ok])[0, 1]), 3)
            if ok.sum() > 10 and np.std(meas[ok]) > 0 and np.std(zc["predicted"][ok]) > 0 else None}


def run_zoom_events(segs: tuple[str, ...] = SEGS, with_38: bool = True) -> dict:
    """Video vs exported zoom for every camera that has video data (data/zoom_video_<cam>.npz): per segment metrics,
    the events seen in the video and what the exported cameras do there."""
    base = OUT / "data"
    res: dict = {"events": {}, "segments": {}}
    allsegs = (("38",) if with_38 else ()) + tuple(segs)
    for cam in CAMS:
        p = base / f"zoom_video_{cam}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        vid = {k: z[k] for k in z.files}
        step = float(vid["step_s"])
        ev = detect_video_events(vid["t"], vid["dlog"], vid["ok"], step)
        res["events"][cam] = {"video": ev, "exported": {}}
        for seg in allsegs:
            zc = zoom_compare(seg, cam, vid)
            if zc is None:
                continue
            m = zoom_metrics(zc)
            np.savez(base / f"zoom_compare_{seg}_{cam}.npz", **{k: v for k, v in zc.items() if isinstance(v, np.ndarray)})
            res["segments"].setdefault(seg, {})[cam] = m
            d = load_cam(seg_work(seg), cam)
            tt = d["t"]
            rows = []
            for e in ev:
                if e["t0"] < tt[0] or e["t1"] > tt[-1]:
                    continue
                lf = np.log(d["f"])
                i0 = int(np.argmin(np.abs(tt - (e["t0"] - 0.3))))
                i1 = int(np.argmin(np.abs(tt - (e["t1"] + 0.3))))
                sel = (zc["t"] >= e["t0"] - 0.3) & (zc["t"] <= e["t1"] + 0.3)
                pred = float(np.nansum(zc["predicted"][sel]))
                meas = float(np.sum(np.where(zc["ok"][sel], zc["measured"][sel], 0.0)))
                rows.append({"t0": e["t0"], "t1": e["t1"], "video_factor": round(float(np.exp(meas)), 3),
                             "exported_focal_factor": round(float(np.exp(lf[i1] - lf[i0])), 3),
                             "exported_predicted_factor": round(float(np.exp(pred)), 3),
                             "tracked": bool(abs(meas - pred) <= max(0.04, 0.25 * abs(meas)))})
            res["events"][cam]["exported"][seg] = rows
        print(f"[gate_cameras] zoom {cam}: {len(ev)} video events", flush=True)
    (base / "zoom.json").write_text(json.dumps(res, indent=1, default=_np))
    return res


def typed_focal_check() -> dict:
    """No-zoom handheld cameras (01, 02, 03, 06): each segment's one focal against larissa_kadu_38's median focal of
    the same camera (rule: within 1 %); k1 reported."""
    base = baseline_stats()
    out = {}
    for seg in SEGS:
        work = seg_work(seg)
        for cam in NO_ZOOM_CAMS:
            d = load_cam(work, cam)
            if d is None or cam not in base:
                continue
            ratio = float(np.median(d["f"]) / base[cam]["f_median"])
            out[f"{seg}:{cam}"] = {"f_over_maxdim": round(float(np.median(d["f"]) / d["maxdim"]), 4),
                                   "f_38_over_maxdim": round(base[cam]["f_over_maxdim_median"], 4),
                                   "diff_pct": round(100 * (ratio - 1), 3), "within_1pct": bool(abs(ratio - 1) <= 0.01),
                                   "f_range_pct": round(100 * float(d["f"].max() / d["f"].min() - 1), 3),
                                   "k1_median": round(float(np.median(d["k1"])), 5), "type_model": d["camera_type"],
                                   "attached": d["attached"]}
    return out


def load_video(cam: str) -> dict | None:
    p = OUT / "data" / f"zoom_video_{cam}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def merge_runs(ev: list[dict], gap_s: float = 1.0) -> list[dict]:
    """Events of one direction closer than gap_s are one zoom (a pinch-zoom is a few strokes)."""
    out: list[dict] = []
    for e in sorted(ev, key=lambda e: e["t0"]):
        if out and e["t0"] - out[-1]["t1"] <= gap_s and (e["factor"] > 1) == (out[-1]["factor"] > 1):
            out[-1] = {"t0": out[-1]["t0"], "t1": e["t1"], "factor": round(out[-1]["factor"] * e["factor"], 4),
                       "change_pct": round(100 * (out[-1]["factor"] * e["factor"] - 1), 1)}
        else:
            out.append(dict(e))
    return out


def focal_events_full(cams_tracks: list[tuple[str, dict]]) -> list[dict]:
    """Zoom events of the EXPORTED focal track over all segments (stitched at the overlaps, the first segment wins):
    t0, t1, factor of every run that changes f by >= 5 % within 2.5 s."""
    ts, fs = [], []
    last = -1e9
    for seg, d in cams_tracks:
        m = d["t"] > last + 1e-6
        ts.append(d["t"][m]), fs.append(d["f"][m])
        last = d["t"][-1]
    t, f = np.concatenate(ts), np.concatenate(fs)
    lf = np.log(f)
    ev, i, n = [], 0, len(t)
    step = float(np.median(np.diff(t)))
    w = max(2, int(round(2.5 / step)))
    # a zoom = |log f(t + 2.5 s) - log f(t)| >= 5 % : take maximal runs of such windows, then the extent of the change
    big = np.zeros(n, bool)
    for i in range(n - w):     # (a window that spans a gap between segments - e.g. 36-62 s without 38 - is no zoom)
        big[i] = abs(lf[i + w] - lf[i]) >= 0.05 and (t[i + w] - t[i]) <= 3.5
    i = 0
    while i < n:
        if not big[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and big[j + 1]:
            j += 1
        a, b = i, min(n - 1, j + w)
        k0 = a + int(np.argmin(np.abs(lf[a:b + 1] - lf[a])[:1])) if False else a
        # tighten: first / last frame where the log focal has moved 10 % of the total from its ends
        tot = lf[b] - lf[a]
        mv = np.abs(lf[a:b + 1] - lf[a])
        k_start = a + int(np.argmax(mv > 0.1 * abs(tot)))
        k_end = a + int(len(mv) - 1 - np.argmax((np.abs(lf[a:b + 1] - lf[b]))[::-1] > 0.1 * abs(tot)))
        ev.append({"t0": round(float(t[k_start]), 2), "t1": round(float(t[max(k_end, k_start)]), 2),
                   "factor": round(float(np.exp(tot)), 3), "change_pct": round(100 * float(np.exp(tot) - 1), 1)})
        i = j + 1
    return merge_runs(ev, 1.0)


def match_events(vid: list[dict], foc: list[dict], tol_s: float = 1.5) -> dict:
    """Pair video events with exported-focal events (overlap within tol_s); factors compared."""
    used = set()
    rows = []
    for v in vid:
        best, bd = None, 1e9
        for k, f in enumerate(foc):
            if (v["factor"] > 1) != (f["factor"] > 1):
                continue
            d = max(0.0, max(v["t0"], f["t0"]) - min(v["t1"], f["t1"]))
            if d <= tol_s and abs(0.5 * (v["t0"] + v["t1"]) - 0.5 * (f["t0"] + f["t1"])) < bd:
                best, bd = k, abs(0.5 * (v["t0"] + v["t1"]) - 0.5 * (f["t0"] + f["t1"]))
        if best is not None:
            used.add(best)
        rows.append({"video": v, "focal": foc[best] if best is not None else None,
                     "factor_ratio": round(v["factor"] / foc[best]["factor"], 3) if best is not None else None})
    return {"matched": rows, "focal_only": [f for k, f in enumerate(foc) if k not in used]}


def run_zoom(segs: tuple[str, ...] = SEGS, cams: tuple[str, ...] = CAMS, tag: str = "") -> dict:
    """Video-vs-export zoom evidence for every camera: per segment unexplained zoom, event pairing over the dance."""
    out: dict = {}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for cam in cams:
        vid = load_video(cam)
        if vid is None:
            continue
        step = float(vid["step_s"])
        vev = merge_runs(detect_video_events(vid["t"], vid["dlog"], vid["ok"], step))
        per_seg, tracks = {}, []
        for seg in segs:
            d = load_cam(seg_work(seg), cam)
            if d is None:
                continue
            tracks.append((seg, d))
            zc = zoom_compare(seg, cam, vid)
            if zc is None:
                continue
            m = zoom_metrics(zc)
            per_seg[seg] = m
            np.savez(OUT / "data" / f"zoomcmp{tag}_{seg}_{cam}.npz", t=zc["t"], measured=zc["measured"], ok=zc["ok"],
                     predicted=zc["predicted"], dlogf=zc["dlogf"])
        fev = focal_events_full(tracks) if tracks else []
        out[cam] = {"video_events": vev, "focal_events": fev, "pairing": match_events(vev, fev), "segments": per_seg,
                    "video_pairs": int(len(vid["t"])), "video_ok": round(float(vid["ok"].mean()), 4)}
        print(f"[gate_cameras] zoom {cam}: {len(vev)} video events, {len(fev)} focal events", flush=True)
    (OUT / "data" / f"zoom_evidence{tag}.json").write_text(json.dumps(out, indent=1, default=_np))
    return out


def _np(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.gate_cameras", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["table", "zoom", "zoom-events", "plots"])
    ap.add_argument("--segs", default=",".join(SEGS))
    ap.add_argument("--tag", default="", help="suffix of the output files (before / after a fix)")
    ap.add_argument("--before", action="store_true", help="read cameras/s2_gate_backup where it exists (state before the fix)")
    ap.add_argument("--cams", default=",".join(CAMS))
    a = ap.parse_args(argv)
    try:
        from .video_zoom import lower_priority
        lower_priority()
    except Exception:  # noqa: BLE001
        pass
    segs = tuple(a.segs.split(","))
    if a.before:
        global BASE, CACHE_TAG
        BASE, CACHE_TAG = "cameras/s2_gate_backup", "_before"
    if a.what == "table":
        run_table(segs)
    elif a.what == "zoom":
        run_zoom(segs, tuple(a.cams.split(",")), tag=a.tag)           # data/zoom_evidence<tag>.json
    elif a.what == "zoom-events":
        run_zoom_events(segs)                                         # data/zoom.json + zoom_compare_<seg>_<cam>.npz


if __name__ == "__main__":
    main()
