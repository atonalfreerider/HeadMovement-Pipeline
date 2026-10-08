"""Does the exported camera reproduce the motion that is in the VIDEO? (camera-tracking gate, independent evidence)

cameras/video_zoom.py measures, for successive samples (dt ~ 1/6 s) of a source video, the static background's motion
(SIFT matches, homography RANSAC) and stores the inlier matches a -> b in native pixels. Here the EXPORTED cameras
of the two frames predict where every matched point must go,

    b_pred = K1 D1( R1 R0^T U0(a) )       U = undistort, D = distort, K = exported intrinsics per frame,

(a pure rotation + zoom about the camera centre; parallax of a handheld phone against a far room is a fraction of a
pixel), and the residual b_pred -> b tells whether rotation AND focal are right:

  motion_deg   median |b_pred - b| / f in degrees of the frame pair (angular error of the relative rotation / zoom)
  dlog_res     log scale of the similarity b_pred -> b: the zoom the exported cameras do NOT explain
                 (an attached camera that zoomed with a constant focal: +-40 % for the pairs of the zoom)

Everything but the exported K, R, k1 comes from the video itself.

    python -m dancecap.cameras.video_motion [--segs a,b,c,d,38] [--cams 04,05,07,09]      # data/motion_<seg>_<cam>.npz, motion.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import gate_cameras as gc
from .video_zoom import similarity


def _undistort(xy: np.ndarray, K: np.ndarray, k1: float) -> np.ndarray:
    x = (xy - K[:2, 2]) / K[0, 0]
    xu = x.copy()
    for _ in range(6):
        r2 = (xu * xu).sum(1, keepdims=True)
        xu = x / (1 + k1 * r2)
    return xu


def _distort(xu: np.ndarray, K: np.ndarray, k1: float) -> np.ndarray:
    r2 = (xu * xu).sum(1, keepdims=True)
    return xu * (1 + k1 * r2) * K[0, 0] + K[:2, 2]


def predict(a: np.ndarray, K0, R0, k0, K1, R1, k1) -> np.ndarray:
    """Pure-rotation prediction of the points a (frame 0, native px) in frame 1."""
    xu = _undistort(a, K0, k0)
    ray = np.c_[xu, np.ones(len(xu))] @ (R1 @ R0.T).T
    z = np.where(ray[:, 2] > 1e-6, ray[:, 2], np.nan)
    return _distort(ray[:, :2] / z[:, None], K1, k1)


def predict3d(a: np.ndarray, depth: np.ndarray, K0, R0, t0, k0, K1, R1, t1, k1) -> np.ndarray:
    """Prediction with the cameras' translation: the points a (frame 0, native px) sit at camera depth `depth` (z, in
    the units of the exported tvec); they are carried to the world through frame 0 and projected by frame 1. For an
    attached camera (one fixed centre) the depth cancels and this equals `predict`."""
    xu = _undistort(a, K0, k0)
    Xc0 = np.c_[xu, np.ones(len(xu))] * depth[:, None]
    Xw = (Xc0 - t0) @ R0                                   # R0^T (Xc0 - t0)
    Xc1 = Xw @ R1.T + t1
    z = np.where(Xc1[:, 2] > 1e-6, Xc1[:, 2], np.nan)
    return _distort(Xc1[:, :2] / z[:, None], K1, k1)


def to_frame_orientation(P: np.ndarray, rotation: int, w: int, h: int) -> np.ndarray:
    """Points measured in the DECODED (un-rotated) video, native pixel centres, -> the autorotated frames the camera
    stage worked on (w x h = frames.json width x height). OpenCV does not apply the display-matrix rotation, ffmpeg does
    (ingest): rotation -90 = a quarter turn clockwise (checked against ffmpeg frames of cam05, 06, 07, 09)."""
    if not rotation:
        return P
    x, y = P[..., 0], P[..., 1]
    if rotation % 360 == 270:                      # -90: clockwise, (x, y) -> (H_l - 1 - y, x), H_l = w
        return np.stack([(w - 1) - y, x], -1)
    if rotation % 360 == 90:                       # +90: counter-clockwise, (x, y) -> (y, W_l - 1 - x), W_l = h
        return np.stack([y, (h - 1) - x], -1)
    if rotation % 360 == 180:
        return np.stack([(w - 1) - x, (h - 1) - y], -1)
    raise ValueError(f"rotation {rotation}")


def pair_nearest(t_frames: np.ndarray, t: float) -> int:
    j = int(np.searchsorted(t_frames, t))
    j = min(max(j, 0), len(t_frames) - 1)
    if j > 0 and abs(t_frames[j - 1] - t) < abs(t_frames[j] - t):
        j -= 1
    return j


def frame_depth(corr: dict | None, d: dict, i: int, pts: np.ndarray, max_px: float = 25.0) -> np.ndarray:
    """Camera depth (z) of the static scene at the image points `pts` of frame i: the depth of the nearest localisation
    inlier (cameras/cache/loc_<cam>.npz corr_xy / corr_xyz of the frame), the frame's median where none is near."""
    if corr is None:
        return np.full(len(pts), 10.0)
    a, b = int(corr["corr_off"][i]), int(corr["corr_off"][i + 1])
    if b - a < 5:
        return np.full(len(pts), 10.0)
    from scipy.spatial import cKDTree
    X = corr["corr_xyz"][a:b].astype(float)
    z = X @ d["R"][i][2] + d["tvec"][i][2]
    z = np.where(z > 0.05, z, np.nan)
    if not np.isfinite(z).any():
        return np.full(len(pts), 10.0)
    dist, j = cKDTree(corr["corr_xy"][a:b].astype(float)).query(pts)
    out = np.where(dist <= max_px, z[j], np.nanmedian(z))
    return np.where(np.isfinite(out), out, np.nanmedian(z))


def motion_pairs(d: dict, vid: dict, max_dt: float = 0.02, corr: dict | None = None) -> dict | None:
    """Per video pair inside the exported track: median angular residual (deg), log-scale residual, inliers used and
    the exported ok flags. None when no pair falls in the track. corr = the camera's localisation cache (corr_off,
    corr_xy, corr_xyz) gives the points' depth so that the cameras' translation counts (a zoom ambiguous with a
    movement along the axis is then judged by what the picture does); without it a pure rotation is assumed."""
    t0 = vid["offset"] + vid["slope"] * vid["pts0"]
    t1 = vid["offset"] + vid["slope"] * vid["pts1"]
    tf = d["t"]
    sel = np.flatnonzero((t0 >= tf[0] - max_dt) & (t1 <= tf[-1] + max_dt) & vid["ok"])
    if not len(sel):
        return None
    out = {k: np.full(len(sel), np.nan) for k in ("t", "motion_deg", "motion_px", "dlog_res", "dlog_video", "n", "centre_dev_px")}
    okf = np.zeros(len(sel), bool)
    for q, k in enumerate(sel):
        i0, i1 = pair_nearest(tf, t0[k]), pair_nearest(tf, t1[k])
        if abs(tf[i0] - t0[k]) > max_dt or abs(tf[i1] - t1[k]) > max_dt or i0 == i1:
            continue
        a, b = vid["ma"][k], vid["mb"][k]
        m = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
        if m.sum() < 15:
            continue
        a, b = a[m].astype(float), b[m].astype(float)
        a, b = (to_frame_orientation(P, d.get("rotation", 0), d["w"], d["h"]) for P in (a, b))
        if corr is None or d["attached"]:
            pred = predict(a, d["K"][i0], d["R"][i0], float(d["k1"][i0]), d["K"][i1], d["R"][i1], float(d["k1"][i1]))
        else:
            pred = predict3d(a, frame_depth(corr, d, i0, a), d["K"][i0], d["R"][i0], d["tvec"][i0], float(d["k1"][i0]),
                             d["K"][i1], d["R"][i1], d["tvec"][i1], float(d["k1"][i1]))
        good = np.isfinite(pred[:, 0])
        if good.sum() < 15:
            continue
        r = np.linalg.norm(pred[good] - b[good], axis=1)
        f = float(d["K"][i1][0, 0])
        out["t"][q] = 0.5 * (t0[k] + t1[k])
        out["motion_px"][q] = float(np.median(r))
        out["motion_deg"][q] = float(np.degrees(np.arctan(np.median(r) / f)))
        out["dlog_res"][q] = similarity(pred[good], b[good], 3.0)[0]
        out["dlog_video"][q] = float(vid["dlog"][k])
        out["n"][q] = int(good.sum())
        okf[q] = bool(d["ok"][i0] and d["ok"][i1])
    out["frame_ok"] = okf
    keep = np.isfinite(out["t"])
    return {k: v[keep] for k, v in out.items()} if keep.any() else None


def block_stats(mp: dict, win_s: float = 2.0, t_all: np.ndarray | None = None) -> list[dict]:
    """Per 10-s block: median / p95 of the pair residuals and the unexplained zoom (max over 2 s windows); with t_all
    (the times of ALL video pairs inside the track) also the share of pairs that had static background to measure
    ('evidence'): a close-up of the dancers hides the room and leaves no evidence there."""
    t = mp["t"]
    res = np.nan_to_num(mp["dlog_res"])
    ws = gc.window_sums(t, res, win_s)
    rows = []
    for a in gc.block_edges(t):
        m = (t >= a) & (t < a + gc.BLOCK_S) & np.isfinite(mp["motion_deg"])
        if m.sum() < 5:
            continue
        w = (t >= a) & (t < a + gc.BLOCK_S)
        ev = None
        if t_all is not None:
            n_all = int(((t_all >= a) & (t_all < a + gc.BLOCK_S)).sum())
            ev = round(float(((t >= a) & (t < a + gc.BLOCK_S)).sum() / max(n_all, 1)), 3)
        rows.append({"t0": float(a), "t1": float(a + gc.BLOCK_S), "pairs": int(m.sum()), "evidence": ev,
                     "motion_deg_med": round(float(np.median(mp["motion_deg"][m])), 4),
                     "motion_deg_p95": round(float(np.percentile(mp["motion_deg"][m], 95)), 4),
                     "motion_px_med": round(float(np.median(mp["motion_px"][m])), 3),
                     "unexplained_zoom_max_2s_pct": round(100 * float(np.exp(np.abs(ws[w]).max()) - 1), 2) if w.any() else None})
    return rows


def run_motion(segs=("a", "38", "b", "c", "d"), cams=gc.CAMS, log=print) -> dict:
    base = gc.OUT / "data"
    res: dict = {}
    for cam in cams:
        vid = gc.load_video(cam)
        if vid is None or "ma" not in vid:
            continue
        for seg in segs:
            d = gc.load_cam(gc.seg_work(seg), cam)
            if d is None:
                continue
            q = gc.seg_work(seg) / "cameras" / "cache" / f"loc_{cam}.npz"
            corr = None
            if q.exists():
                with np.load(q) as z:
                    corr = {k: z[k] for k in ("corr_off", "corr_xy", "corr_xyz")}
            mp = motion_pairs(d, vid, corr=corr)
            if mp is None:
                continue
            np.savez(base / f"motion_{seg}_{cam}.npz", **mp)
            tp = 0.5 * (vid["offset"] + vid["slope"] * (vid["pts0"] + vid["pts1"]))
            rows = block_stats(mp, t_all=tp[(tp >= d["t"][0]) & (tp <= d["t"][-1])])
            res.setdefault(seg, {})[cam] = {
                "pairs": int(len(mp["t"])), "motion_deg_med": round(float(np.median(mp["motion_deg"])), 4),
                "motion_deg_p95": round(float(np.percentile(mp["motion_deg"], 95)), 4),
                "unexplained_zoom_max_2s_pct": round(100 * float(np.exp(np.abs(gc.window_sums(mp["t"], np.nan_to_num(mp["dlog_res"]), 2.0)).max()) - 1), 2),
                "blocks": rows}
            log(f"[video_motion] {seg} cam{cam}: {len(mp['t'])} pairs, median {res[seg][cam]['motion_deg_med']} deg, "
                f"unexplained zoom {res[seg][cam]['unexplained_zoom_max_2s_pct']} %", flush=True)
    (base / "motion.json").write_text(json.dumps(res, indent=1, default=gc._np))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.video_motion", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segs", default="a,38,b,c,d")
    ap.add_argument("--cams", default=",".join(gc.CAMS))
    a = ap.parse_args(argv)
    run_motion(tuple(a.segs.split(",")), tuple(a.cams.split(",")))


if __name__ == "__main__":
    main()
