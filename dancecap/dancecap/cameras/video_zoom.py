"""Zoom measured DIRECTLY from a source video, independent of the camera pipeline (gate_cameras evidence).

For every pair of successive samples (dt ~ 1/6 s) of a video the image motion of the (mostly static) background is
fitted with a similarity transform (RANSAC over SIFT matches); its scale s is the frame-to-frame zoom of the picture
(rotation and a little translation of a handheld phone do not change it to first order). The cumulative sum of log s
is the video's zoom curve; compare it with the exported focal track with `predicted_log_scale` (the same similarity
fit on the static map points projected through the EXPORTED cameras of both frames, so rotation / parallax effects
cancel and only a focal error remains) or simply with d log f.

    python -m dancecap.cameras.video_zoom <take> <cam> [--t0 S --t1 S]     # writes <review>/data/zoom_video_<cam>.npz

Only the video and frames.json (pts -> reference time) are read; nothing of the camera stage is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BASELINE_TAKE = "larissa_kadu_38"


def lower_priority() -> None:
    """The workstation has 4 logical CPUs shared with other work: run below normal, one OpenCV thread."""
    try:
        import cv2
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001 - best effort
        pass


def pts_of(t: np.ndarray, offset: float, slope: float) -> np.ndarray:
    """Reference time -> video presentation time (frames.json: t = offset + slope * pts)."""
    return (np.asarray(t, float) - offset) / slope


def sample_grid(offset: float, slope: float, t0: float, t1: float, step_s: float, pts_max: float | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """(pts, t) of the sample instants: every step_s of VIDEO time whose reference time (t = offset + slope * pts,
    frames.json) lies in [t0, t1] and whose pts lies in [0, pts_max]."""
    a = max(0.0, (t0 - offset) / slope)
    b = (t1 - offset) / slope
    if pts_max is not None:
        b = min(b, pts_max)
    pts = np.arange(a, b + 1e-9, step_s * slope)
    return pts, offset + slope * pts


def similarity(a: np.ndarray, b: np.ndarray, thresh: float = 3.0) -> tuple[float, float, int, int]:
    """(log scale, rotation deg, inliers, n) of the robust similarity a -> b ((n,2) arrays); nan when it fails."""
    import cv2
    if len(a) < 8:
        return float("nan"), float("nan"), 0, len(a)
    M, inl = cv2.estimateAffinePartial2D(a.astype(np.float32), b.astype(np.float32), method=cv2.RANSAC,
                                         ransacReprojThreshold=thresh, maxIters=2000, confidence=0.995)
    if M is None:
        return float("nan"), float("nan"), 0, len(a)
    s = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return float(np.log(max(s, 1e-6))), rot, int(inl.sum()), len(a)


def robust_motion(a: np.ndarray, b: np.ndarray, h_thresh: float = 3.0):
    """Static-background motion between two frames from matches a -> b: RANSAC homography picks the background
    (rotation + zoom of a handheld phone are a homography; people are not), then the similarity of those inliers gives
    (log scale, rotation deg). Returns (log scale, rot deg, n inliers, n matches, inlier mask) (nan / zeros on failure)."""
    import cv2
    n = len(a)
    if n < 15:
        return float("nan"), float("nan"), 0, n, np.zeros(n, bool)
    H, m = cv2.findHomography(a.astype(np.float32), b.astype(np.float32), cv2.RANSAC, h_thresh, maxIters=2000,
                              confidence=0.995)
    if H is None or m is None:
        return float("nan"), float("nan"), 0, n, np.zeros(n, bool)
    inl = m.ravel().astype(bool)
    ls, rot, k, _ = similarity(a[inl], b[inl], 1.5 * h_thresh)
    return ls, rot, int(inl.sum()), n, inl


def _read_samples(video: Path, pts: np.ndarray, maxdim: int, rotation: int | None = None):
    """Yield (i, gray frame) for the sample instants, decoding sequentially (grab skips the unwanted frames).
    OpenCV autorotates (the ingest does too); frames are scaled to maxdim on the long side."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # start near the first wanted frame (the container index is by presentation time)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (pts[0] - 1.0)) * 1000.0)
    j = 0
    while j < len(pts):
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # CAP_PROP_POS_MSEC after grab is the pts of the grabbed frame
        if t + 0.5 / fps < pts[j]:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        sc = maxdim / max(h, w)
        if sc < 1:
            g = cv2.resize(g, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
        else:
            sc = 1.0
        yield j, g, t, sc
        # next wanted sample: skip samples already behind the decoder (a stalled video)
        j += 1
        while j < len(pts) and pts[j] < t:
            j += 1
    cap.release()


class PersonBoxes:
    """Boxes of every detected person (dancers AND spectators; people/<cam>.npz of the take(s) that hold the camera's
    frames, xyxy in the autorotated frame) by reference time: the video estimators drop the features inside them, so that
    only the static room moves the estimate (a close-up of the dancers fills the frame; they move on their own)."""

    def __init__(self, cam: str, takes: tuple[str, ...], work_root: Path, margin: float = 0.10):
        self.parts, self.margin = [], margin
        for tk in takes:
            w = work_root / tk
            pp, fj = w / "people" / f"{cam}.npz", w / "frames" / cam / "frames.json"
            if pp.exists() and fj.exists():
                z = np.load(pp)
                meta = json.loads(fj.read_text())
                t = np.array([f["t"] for f in meta["frames"]])
                n = min(len(t), len(z["boxes"]))
                self.parts.append((t[:n], z["boxes"][:n], z["valid"][:n], int(meta["width"]), int(meta["height"])))

    def at(self, t: float, max_dt: float = 0.05) -> np.ndarray:
        for tt, boxes, valid, w, h in self.parts:
            if tt[0] - max_dt <= t <= tt[-1] + max_dt:
                i = int(np.argmin(np.abs(tt - t)))
                b = boxes[i][valid[i]].astype(float)
                if len(b):
                    c, sz = 0.5 * (b[:, :2] + b[:, 2:]), (b[:, 2:] - b[:, :2]) * (1 + self.margin)
                    b = np.concatenate([c - 0.5 * sz, c + 0.5 * sz], 1)
                return b
        return np.zeros((0, 4))


def outside_boxes(xy: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Mask of the points (n,2) that lie in none of the boxes (m,4) xyxy."""
    if not len(boxes):
        return np.ones(len(xy), bool)
    inside = ((xy[:, None, 0] >= boxes[None, :, 0]) & (xy[:, None, 0] <= boxes[None, :, 2])
              & (xy[:, None, 1] >= boxes[None, :, 1]) & (xy[:, None, 1] <= boxes[None, :, 3])).any(1)
    return ~inside


def measure(video: Path, offset: float, slope: float, t0: float, t1: float, step_s: float = 1 / 6,
            maxdim: int = 360, nfeat: int = 1000, keep: int = 150, log=print, person_boxes: PersonBoxes | None = None,
            geom: tuple[int, int, int] | None = None) -> dict:
    """Frame-to-frame motion of the video over [t0, t1] (reference clock). Returns arrays over the sample pairs:
    t (pair mid-time), pts0, pts1, dlog (log scale of the static background), rot_deg, inl, n, 'ok' (inliers >= 25
    and >= 35 % of the matches), and the inlier matches themselves in NATIVE pixel coordinates of the autorotated
    frames, ma / mb (P, keep, 2) float32 (NaN padded): the evidence to compare with the exported cameras' prediction."""
    import cv2
    pts, tt = sample_grid(offset, slope, t0, t1, step_s, _duration(video))
    sift = cv2.SIFT_create(nfeatures=nfeat)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rng = np.random.default_rng(0)
    prev = None
    T, DL, ROT, INL, N, P0, P1, MA, MB = [], [], [], [], [], [], [], [], []
    for j, g, tv, sc in _read_samples(video, pts, maxdim):
        kp, des = sift.detectAndCompute(g, None)
        xy = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
        if person_boxes is not None and geom is not None and len(xy):
            from .video_motion import to_frame_orientation
            nat = to_frame_orientation(((xy + 0.5) / sc - 0.5).astype(float), geom[0], geom[1], geom[2])
            keep_kp = outside_boxes(nat, person_boxes.at(float(tt[j])))
            xy, des = xy[keep_kp], (des[keep_kp] if des is not None else des)
        cur = (j, tv, xy, des, sc)
        ma = np.full((keep, 2), np.nan, np.float32)
        mb = np.full((keep, 2), np.nan, np.float32)
        if prev is not None and prev[3] is not None and des is not None and len(xy) > 12 and len(prev[2]) > 12:
            m = bf.knnMatch(prev[3], des, k=2)
            good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
            a = prev[2][[g_.queryIdx for g_ in good]]
            b = cur[2][[g_.trainIdx for g_ in good]]
            ls, rot, inl, n, mask = robust_motion(a, b)
            if mask.any():
                sel = np.flatnonzero(mask)
                if len(sel) > keep:
                    sel = np.sort(rng.choice(sel, keep, replace=False))
                ma[:len(sel)] = (a[sel] + 0.5) / prev[4] - 0.5      # small-image pixel centres -> native pixel centres
                mb[:len(sel)] = (b[sel] + 0.5) / sc - 0.5
        else:
            ls = rot = float("nan")
            inl = n = 0
        if prev is not None:
            T.append(0.5 * (tt[prev[0]] + tt[j])), P0.append(prev[1]), P1.append(tv)
            DL.append(ls), ROT.append(rot), INL.append(inl), N.append(n), MA.append(ma), MB.append(mb)
        prev = cur
        if log and j % 300 == 0:
            log(f"[video_zoom] {video.name}: sample {j}/{len(pts)}", flush=True)
    dl, inl, n = np.array(DL), np.array(INL), np.array(N)
    ok = np.isfinite(dl) & (inl >= 25) & (inl >= 0.35 * np.maximum(n, 1))
    return {"t": np.array(T), "pts0": np.array(P0), "pts1": np.array(P1), "dlog": dl, "rot_deg": np.array(ROT),
            "inl": inl, "n": n, "ok": ok, "step_s": np.float64(step_s), "maxdim": np.int64(maxdim),
            "offset": np.float64(offset), "slope": np.float64(slope), "ma": np.array(MA), "mb": np.array(MB),
            "estimator": np.str_("homography-inliers + similarity, matches stored (v2)")}


def _duration(video: Path) -> float | None:
    import cv2
    cap = cv2.VideoCapture(str(video))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(n / fps) if n > 0 and fps > 0 else None


def cumulative(res: dict, gap_s: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """(t, cumulative log scale) of a measure() result; undefined pairs count as no change, and the curve is only
    meaningful over spans of ok pairs (a gap longer than gap_s restarts nothing: the caller compares increments)."""
    d = np.where(res["ok"], res["dlog"], 0.0)
    return res["t"], np.cumsum(d)


def predicted_scale_pairs(tr: dict, corr: dict, i0: np.ndarray, i1: np.ndarray, thresh: float = 3.0,
                          max_pts: int = 600, seed: int = 0, far: float | None = None) -> np.ndarray:
    """Log scale the EXPORTED cameras predict between exported frames i0[k] and i1[k]: the static map points of frame
    i0's correspondences (cameras/cache/loc_<cam>.npz: corr_off, corr_xyz) are projected through both exported
    cameras (tr: K (F,3,3), R, tvec, k1) and fitted with the same similarity as the video. NaN when frame i0 has too
    few points. For an attached camera the points are bearings from its centre (10 units out): with a MOVING centre
    (attach_global's path) frame i0's bearings seen from frame i1's centre carry a fake parallax, so far=1e5 puts every
    point at 1e5 units along its bearing from frame i0's centre: only rotation and focal count."""
    from .report import project_cv
    rng = np.random.default_rng(seed)
    off = corr["corr_off"]
    out = np.full(len(i0), np.nan)
    for k, (a_, b_) in enumerate(zip(i0, i1)):
        a, b = int(off[a_]), int(off[a_ + 1])
        if b - a < 20:
            continue
        X = corr["corr_xyz"][a:b].astype(float)
        if len(X) > max_pts:
            X = X[rng.choice(len(X), max_pts, replace=False)]
        if far:
            C0 = -tr["R"][a_].T @ tr["tvec"][a_]
            u = X - C0
            X = C0 + far * u / np.linalg.norm(u, axis=1, keepdims=True)
        p0, v0 = project_cv(X, tr["K"][a_], tr["R"][a_], tr["tvec"][a_], tr["k1"][a_])
        p1, v1 = project_cv(X, tr["K"][b_], tr["R"][b_], tr["tvec"][b_], tr["k1"][b_])
        v = v0 & v1
        if v.sum() < 20:
            continue
        out[k] = similarity(p0[v], p1[v], thresh)[0]
    return out


def main(argv=None):
    from ..take import Take
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.video_zoom", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", help="a take whose frames/<cam>/frames.json carries the studio clock (offset, slope)")
    ap.add_argument("cams", help="comma-separated camera ids")
    ap.add_argument("--t0", type=float, default=-3.3, help="reference time, first sample")
    ap.add_argument("--t1", type=float, default=187.0, help="reference time, last sample")
    ap.add_argument("--step", type=float, default=1 / 6)
    ap.add_argument("--outdir")
    ap.add_argument("--no-mask", action="store_true", help="do not drop the features inside person boxes")
    ap.add_argument("--normal-priority", action="store_true", help="do not lower the process priority (the machine is "
                    "otherwise saturated and a below-normal job gets no CPU)")
    a = ap.parse_args(argv)
    if not a.normal_priority:
        lower_priority()
    else:
        import cv2
        cv2.setNumThreads(1)
    take = Take.load(a.take)
    work = take.work.resolve()
    outdir = Path(a.outdir) if a.outdir else (work.parent / "larissa_kadu_full" / "review" / "gate_cameras" / "data")
    outdir.mkdir(parents=True, exist_ok=True)
    for cam in a.cams.split(","):
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        src = next(s for s in take.sources if s.id == cam)
        pb = None if a.no_mask else PersonBoxes(cam, ("larissa_kadu_full_a", BASELINE_TAKE, "larissa_kadu_full_b",
                                                        "larissa_kadu_full_c", "larissa_kadu_full_d"), work.parent)
        res = measure(Path(src.path), float(meta["offset"]), float(meta["slope"]), a.t0, a.t1, a.step, person_boxes=pb,
                      geom=(int(meta.get("rotation") or 0), int(meta["width"]), int(meta["height"])))
        out = outdir / f"zoom_video_{cam}.npz"
        np.savez(out, **res)
        print(f"[video_zoom] {cam}: {len(res['t'])} pairs, ok {res['ok'].mean():.2%} -> {out}", flush=True)


if __name__ == "__main__":
    main()
