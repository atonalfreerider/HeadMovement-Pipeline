"""Per-camera residual video offsets from multi-view geometry: sync/offsets.json + review/sync_curves.png.

Every frame already carries a reference time t from the per-phone audio clock fit (frames/<cam>/frames.json). What is
left is a small constant (or slowly drifting) offset delta_c per camera between a phone's video and its audio: the
corrected frame time is t + delta_c. Geometry measures it: for a candidate delta, camera c's frame i was exposed at
t_i + delta, so the dancers seen by the OTHER cameras at that instant (their ma_2d landmarks and per-frame cameras
resampled to t_i + delta, then triangulated by uncertainty-weighted DLT) must project onto c's own landmarks of frame
i. The residual in c is minimal at the true delta. A timing error only moves a landmark ALONG its image motion
(by speed x error), and these dancers move slowly in the images (median ~50 px/s, i.e. 0.5 px per 10 ms), so the
curve minimised is the median |residual . motion direction| over each camera's fastest 30 % of landmarks
("along_px"; the plain median residual over all landmarks is reported too but is far too flat to locate delta).

    python -m dancecap sync larissa_kadu [--cams 01,02,..] [--range-ms 300] [--workers N]

Workers: the residual evaluations of one joint iteration (each camera's step and its finite differences) and the
per-camera curve searches are independent, so they run in a process pool (fork; identical numbers to the serial
path). Default: serial off-pod (the 4-CPU workstation), the container's CPU quota (max 16) on a pod
(larissa_kadu_38 serial on an A100 pod: 1565 s).

Solve: all offsets jointly (solve_joint). For each camera a Gauss-Newton step is the median over its fast landmarks
of -(d . v)/|v|^2 (d = projected - observed, v = image velocity); the dependence of each camera's optimum on the
others' offsets (finite differences) gives a coupling matrix G and the fixed point (I - G) delta = delta* - G offs
is solved by least squares. Independent per-camera searches iterated with the others' latest values oscillated
on larissa_kadu (cam 01 +72 / +29 / +67 ms), because the coupling gains are geometry dependent. Attached cameras
(05, 06: yaw and centre fitted to the dancers) are measured against the static-model cameras but never used to
triangulate. The joint iterates settle into a noisy band; the second half is averaged. Reported delta: the minimum of
each camera's own residual curve with the other cameras at the joint solution (a 20 ms scan over +-range, 4 ms steps
within +-40 ms of its minimum, 1 ms steps +-8 ms, least-squares parabola), i.e. the definition of delta with the
coupling resolved; estimator_gap_ms = |curve minimum - joint value|. Gauge: geometry only sees relative offsets, so the median over cameras
is set to 0 (the reference clock is then the cameras' average A/V alignment). Stability: the search is repeated in 2 s sub-windows;
a camera whose sub-window estimates spread by >= 10 ms is flagged ("stable": false) and a linear drift is reported
(rate, t_ref); mamma_dyn applies rate only where "use_rate" is true.
Clocks: offsets.json records the audio clocks of the frames it was measured on ("clocks"); offsets_check (used by
mamma_dyn.read_sync_offsets) refuses it on any other clocks, and a take with [sync] required = true refuses MAMMA
without its own verified file.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from . import mamma_dyn as md

SCAN_MS = 20.0  # wide pre-scan step (real offsets reach ~100+ ms: beyond the +-120 ms first planned)
COARSE_MS = 4.0
COARSE_HALF_MS = 40.0  # 4 ms steps within +-40 ms of the scan minimum
FINE_MS = 1.0
FINE_HALF_MS = 8.0
POSE_DT = 0.001  # camera poses are pre-resampled on a 1 ms grid of the camera's own clock
SUB_WINDOW_S = 2.0
STABLE_MS = 10.0
SIGMA_REL_MAX = 0.07  # landmarks whose sigma exceeds 7 % of the person's crop are not used (ma_2d median ~5.5 %)
VIS_MIN = 0.5
FAST_PCT = 70.0  # the along-motion residual uses the fastest 30 % of a camera's landmarks ...
FAST_MIN_PX_S = 40.0  # ... and never slower than 40 px/s (0.4 px per 10 ms of timing error)
METRIC = "along_px"  # curve minimised: median |residual . motion direction| over fast landmarks
LMK_STEP = 4  # every 4th of MAMMA's 512 points (128 per person) - plenty for a median, 4x cheaper
TRI_REL_MAX = 0.08  # triangulated point must reproject within 8 % of the crop in the cameras that made it
CLOCK_TOL = 1e-6  # slope (s/s) / offset (s): offsets.json is only valid on the audio clocks it was measured on


# ----------------------------------------------------------------------------------------------- clock provenance
# delta_c is measured ON TOP of the frame times of frames/<cam>/frames.json, i.e. on one set of audio clocks
# (alignment.json or alignment_studio.json). Applied to frames timed by other clocks it corrects the wrong thing (the
# larissa_kadu offsets mix alignment.json's -36..+29 ms audio clock errors into delta). offsets.json therefore records
# the clocks it was measured on ("clocks"), and readers refuse a file whose clocks differ from the frames' or the
# take's ("stale"), or - for a take with [sync] required = true - a file that does not record them ("unrecorded").


def frame_clocks(work: Path) -> dict[str, list[float]]:
    """{cam: [slope, offset]} of the audio clocks the ingested frame times use (frames/<cam>/frames.json)."""
    out = {}
    for p in sorted((work / "frames").glob("*/frames.json")):
        m = json.loads(p.read_text())
        if "slope" in m and "offset" in m:
            out[p.parent.name] = [float(m["slope"]), float(m["offset"])]
    return out


def take_clocks(work: Path) -> dict | None:
    """takes/<work.name>.toml's clocks {alignment, required, sources {cam: [slope, offset]}} when work is that take's
    work dir and the take loads (None for synthetic work dirs, or on a pod without the alignment files)."""
    from .take import PACKAGE_ROOT, Take
    toml = PACKAGE_ROOT / "takes" / f"{work.name}.toml"
    try:
        if not toml.exists() or Path(work).resolve() != (PACKAGE_ROOT / "work" / work.name).resolve():
            return None
        take = Take.load(work.name)
    except (OSError, KeyError, ValueError):
        return None
    return {"alignment": take.cfg["alignment"], "required": sync_required(take),
            "sources": {s.id: [s.slope, s.offset] for s in take.sources}}


def sync_required(take) -> bool:
    """takes/<take>.toml [sync] required = true: MAMMA must not run without this take's own sync/offsets.json."""
    return bool((take.cfg.get("sync") or {}).get("required", False))


def _clock_diff(rec: dict, cur: dict, what: str) -> list[str]:
    bad = []
    for cam in sorted(set(rec) & set(cur)):
        (s1, o1), (s2, o2) = rec[cam], cur[cam]
        if abs(s1 - s2) > CLOCK_TOL or abs(o1 - o2) > CLOCK_TOL:
            bad.append(f"cam{cam}: {what} offset {1000 * (o2 - o1):+.2f} ms, slope {1e6 * (s2 - s1):+.2f} ppm")
    return bad


def offsets_check(work: Path, doc: dict | None = None) -> dict:
    """Can work/sync/offsets.json be applied to work's frame times? -> {status, required, detail}.

    status: "missing"; "ok" (its recorded clocks equal the frames' and the take's); "stale" (they differ: measured on
    other clocks, e.g. copied from another take or made before a re-ingest); "unrecorded" (written before the clocks
    were recorded: cannot be verified - acceptable only for takes that do not require sync)."""
    p = work / "sync" / "offsets.json"
    tk = take_clocks(work)
    required = bool(tk and tk["required"])
    if doc is None and not p.exists():
        return {"status": "missing", "required": required, "path": str(p), "detail": []}
    doc = doc if doc is not None else json.loads(p.read_text())
    rec = (doc.get("clocks") or {}).get("sources") if isinstance(doc, dict) else None
    if not rec:
        return {"status": "unrecorded", "required": required, "path": str(p),
                "detail": ["offsets.json does not record the clocks it was measured on"]}
    bad = _clock_diff(rec, frame_clocks(work), "frames.json") + (_clock_diff(rec, tk["sources"], "take") if tk else [])
    return {"status": "stale" if bad else "ok", "required": required, "path": str(p), "detail": bad}


# ----------------------------------------------------------------------------------------------- geometry


def triangulate(uv_n: np.ndarray, w: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted DLT in normalised image coordinates.

    uv_n (C, M, 2) normalised coordinates K^-1 [u v 1], w (C, M) weights (0 = unused; 1/sigma in normalised units),
    P (C, M, 3, 4) [R|t] per observation. Returns X (M, 3) (NaN where fewer than 2 views) and the number of views.
    """
    rows1 = uv_n[..., 0:1] * P[..., 2, :] - P[..., 0, :]  # (C, M, 4)
    rows2 = uv_n[..., 1:2] * P[..., 2, :] - P[..., 1, :]
    A = np.concatenate([rows1 * w[..., None], rows2 * w[..., None]], 0)  # (2C, M, 4)
    A = np.moveaxis(A, 1, 0)  # (M, 2C, 4)
    AtA = np.matmul(A.transpose(0, 2, 1), A)
    _, vecs = np.linalg.eigh(AtA)
    Xh = vecs[..., 0]  # eigenvector of the smallest eigenvalue
    n = (w > 0).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = Xh[:, :3] / Xh[:, 3:4]
    X[n < 2] = np.nan
    return X, n


def _normalise(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(…, 2) pixel -> normalised coordinates with per-sample K (…, 3, 3) (zero skew)."""
    f = K[..., 0, 0][..., None]
    c = K[..., :2, 2]
    return (uv - c) / f


# ----------------------------------------------------------------------------------------------- data


class CamData:
    """One camera's native landmarks (subset), sigmas and per-frame cameras, on its uncorrected reference clock."""

    def __init__(self, tr: md.CameraTrack, obs: dict, idx: np.ndarray, attached: bool = False,
                 type_model: str | None = None):
        self.name = tr.name
        self.attached = attached  # pose partly fitted to the dancers (cameras stage): never used to triangulate
        # camera-type model the cameras stage applied (cameras/<cam>.npz type_model; None = today's per-frame model).
        # 'constant' (a tripod: ONE pose for the whole video) does not depend on the frame times at all
        self.type_model = type_model
        self.tr = tr
        self.t0 = tr.t.copy()
        self.lm = obs["lm"][:, :, idx]  # (F, P, n, 3)
        self.vis = obs["vis"][:, :, idx]
        self.box = obs["box"]
        sig = md.logvar_to_sigma_px(self.lm[..., 2], self.box[..., None])
        self.sigma = sig
        self.conf = (self.vis > VIS_MIN) & (np.abs(self.lm[..., :2]).sum(-1) > 0) \
            & (sig < SIGMA_REL_MAX * self.box[..., None]) & tr.ok[:, None, None]

        # cameras on a dense grid of the own clock once (slerp is the expensive part of a residual evaluation)
        self.tg = np.arange(self.t0[0] - 2 * tr.frame_dt, self.t0[-1] + 2 * tr.frame_dt, POSE_DT)
        self.cg = md.resample_cameras(self.t0, tr.K, tr.R, tr.tvec, tr.ok, self.tg, tr.frame_dt)
        self._lm0 = np.where(self.conf[..., None], self.lm, 0.0)
        # image velocity (px/s, central differences) of each landmark: timing errors only show along it
        v = np.full(self.lm.shape[:3] + (2,), np.nan)
        dt = (self.t0[2:] - self.t0[:-2])[:, None, None, None]
        both = self.conf[2:] & self.conf[:-2] & self.conf[1:-1]
        v[1:-1] = np.where(both[..., None], (self.lm[2:, ..., :2] - self.lm[:-2, ..., :2]) / dt, np.nan)
        self.vel = v
        speed = np.linalg.norm(v, axis=-1)
        fin = np.isfinite(speed)
        thr = max(np.percentile(speed[fin], FAST_PCT), FAST_MIN_PX_S) if fin.any() else np.inf
        self.fast = fin & (speed >= thr)
        self.fast_thr = float(thr)

    def cameras_at(self, s: np.ndarray) -> dict:
        """Cameras at own-clock times s (nearest 1 ms sample; ok False outside the dense grid)."""
        i = np.rint((s - self.tg[0]) / POSE_DT).astype(int)
        inside = (i >= 0) & (i < len(self.tg))
        i = np.clip(i, 0, len(self.tg) - 1)
        return {"K": self.cg["K"][i], "R": self.cg["R"][i], "t": self.cg["t"][i], "ok": self.cg["ok"][i] & inside}

    def at(self, t_query: np.ndarray, offset: float) -> dict:
        """Landmarks + cameras resampled to reference times t_query, with this camera's clock shifted by offset."""
        t_src = self.t0 + offset
        rs = md.resample_landmarks(t_src, self._lm0, self.conf.astype(float), self.box, t_query, self.tr.frame_dt)
        cg = self.cameras_at(t_query - offset)
        valid = (rs["visibilities"] > VIS_MIN) & (np.abs(rs["landmarks"][..., :2]).sum(-1) > 0) & cg["ok"][:, None, None]
        box_q = np.stack([np.interp(t_query, t_src, self.box[:, p]) for p in range(self.box.shape[1])], 1)
        sigma = md.logvar_to_sigma_px(rs["landmarks"][..., 2], box_q[..., None])
        return {"uv": rs["landmarks"][..., :2], "sigma": sigma, "box": box_q,
                "valid": valid & (sigma < 2 * SIGMA_REL_MAX * box_q[..., None]),
                "K": cg["K"], "R": cg["R"], "t": cg["t"]}


def load(work: Path, cams: list[str] | None = None) -> dict[str, CamData]:
    tracks = md.load_tracks(work, cams, offsets={})  # raw clocks: offsets are what we measure
    native = work / "mamma" / "ma_2d_native"
    idx = np.arange(0, md.N_LMK, LMK_STEP)
    out = {}
    for c, tr in tracks.items():
        z = np.load(work / "cameras" / f"{c}.npz")
        attached = bool(z["attached"]) if "attached" in z.files else False
        model = str(z["type_model"]) if "type_model" in z.files else None
        out[c] = CamData(tr, md.native_observations(native, tr), idx, attached, model)
    return out


def _others(data: dict[str, CamData], c: str) -> list[CamData]:
    """Cameras that triangulate the dancers for camera c: the static-model ones (attached cameras' yaw/centre were
    fitted to the dancers' keypoints at the UNcorrected times, so they would pull every estimate towards 0)."""
    main = [data[o] for o in sorted(data) if o != c and not data[o].attached]
    return main if len(main) >= 2 else [data[o] for o in sorted(data) if o != c]


def camera_models(data: dict[str, CamData]) -> dict[str, str]:
    """{cam: camera-type model the cameras stage applied} ('per_frame' = today's model). A 'constant' camera (tripod:
    one pose for the whole video) resamples to the same pose at any candidate offset, so its residual curve measures
    the landmarks' timing only (cleaner minima); offsets.json records the models next to the deltas."""
    return {c: (v.type_model or "per_frame") for c, v in sorted(data.items())}


# ----------------------------------------------------------------------------------------------- residual


def _proj(X: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point projection: X (M, 3), K/R (M, 3, 3), t (M, 3) -> px (M, 2), depth (M,)."""
    Xc = np.matmul(R, X[:, :, None])[:, :, 0] + t
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = Xc[:, :2] / z[:, None]
    return uv * K[:, [0, 1], [0, 1]] + K[:, :2, 2], z


def residual(c: CamData, others: list[CamData], other_offsets: dict[str, float], delta: float,
             frames: np.ndarray | None = None) -> dict:
    """Median reprojection residual (px) in camera c of the dancers triangulated from the other cameras, with c's
    frames re-timed by delta. frames: subset of c's frame indices (sub-windows)."""
    fi = np.flatnonzero(c.tr.ok) if frames is None else frames[c.tr.ok[frames]]
    tq = c.t0[fi] + delta
    A = [o.at(tq, other_offsets.get(o.name, 0.0)) for o in others]
    V = np.stack([a["valid"] for a in A])  # (C, Q, P, n)
    cand = c.conf[fi] & (V.sum(0) >= 2)
    q, pp, j = np.nonzero(cand)
    if len(q) == 0:
        return {"median_px": np.nan, "n": 0, "tau_s": np.nan, "along_px": np.nan, "n_fast": 0, "mean_trunc_px": np.nan}
    C, M = len(A), len(q)
    uvn = np.empty((C, M, 2))
    w = np.empty((C, M))
    Pm = np.empty((C, M, 3, 4))
    for k, a in enumerate(A):
        K = a["K"][q]
        uv = a["uv"][q, pp, j]
        uvn[k] = (uv - K[:, :2, 2]) / K[:, 0, 0][:, None]
        w[k] = np.where(V[k, q, pp, j], K[:, 0, 0] / np.maximum(a["sigma"][q, pp, j], 0.5), 0.0)
        Pm[k, :, :, :3] = a["R"][q]
        Pm[k, :, :, 3] = a["t"][q]
    X, nv = triangulate(uvn, w, Pm)
    # consistency in the cameras that made the point (rejects mismatched / degenerate triangulations)
    tri_err = np.full((C, M), np.nan)
    for k, a in enumerate(A):
        px, z = _proj(X, a["K"][q], a["R"][q], a["t"][q])
        e = np.linalg.norm(px - a["uv"][q, pp, j], axis=-1) / np.maximum(a["box"][q, pp], 1.0)
        tri_err[k] = np.where(V[k, q, pp, j] & (z > 0), e, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tri_med = np.nanmedian(tri_err, 0)
    g = fi[q]
    px, z = _proj(X, c.tr.K[g], c.tr.R[g], c.tr.tvec[g])
    use = (nv >= 2) & np.isfinite(X).all(-1) & (tri_med < TRI_REL_MAX) & (z > 0)
    d = px - c.lm[g, pp, j, :2]
    ev = np.linalg.norm(d, axis=-1)[use]
    fast = use & c.fast[g, pp, j]
    vel = c.vel[g, pp, j][fast]
    dv = (d[fast] * vel).sum(-1)
    along = np.abs(dv) / np.linalg.norm(vel, axis=-1)
    tau = -dv / (vel ** 2).sum(-1)  # s: how much later c's frame really was than assumed
    return {"median_px": float(np.median(ev)) if ev.size else np.nan, "n": int(ev.size),
            "tau_s": float(np.median(tau)) if tau.size >= 20 else np.nan,
            "along_px": float(np.median(along)) if along.size >= 20 else np.nan, "n_fast": int(along.size),
            "mean_trunc_px": float(np.mean(np.minimum(ev, 50.0))) if ev.size else np.nan}


def _search(c: CamData, others: list[CamData], offs: dict[str, float], range_ms: float,
            frames: np.ndarray | None = None, centre_ms: float = 0.0, coarse: bool = True) -> dict:
    """Coarse + fine search of the residual curve -> best delta (s), curves (ms, px)."""
    curve = []
    if coarse:  # 20 ms scan over +-range, then 4 ms steps around its minimum
        for step, lo, hi in ((SCAN_MS, -range_ms, range_ms), (COARSE_MS, None, None)):
            if lo is None:
                ok = [x for x in curve if np.isfinite(x[1])]
                if not ok:
                    return {"delta_ms": np.nan, "curve": curve, "fine": []}
                m = min(ok, key=lambda x: x[1])[0]
                lo, hi = max(-range_ms, m - COARSE_HALF_MS), min(range_ms, m + COARSE_HALF_MS)
            done = {x[0] for x in curve}
            for d in np.arange(lo, hi + 1e-9, step):
                if round(float(d), 3) in done:
                    continue
                r = residual(c, others, offs, d / 1000.0, frames)
                curve.append((round(float(d), 3), r[METRIC], r["n_fast" if METRIC == "along_px" else "n"]))
        curve.sort()
        ok = [x for x in curve if np.isfinite(x[1])]
        centre_ms = min(ok, key=lambda x: x[1])[0]
    fine = []
    for d in np.arange(centre_ms - FINE_HALF_MS, centre_ms + FINE_HALF_MS + 1e-9, FINE_MS):
        r = residual(c, others, offs, d / 1000.0, frames)
        fine.append((float(d), r[METRIC], r["n_fast" if METRIC == "along_px" else "n"]))
    okf = [x for x in fine if np.isfinite(x[1])]
    if not okf:
        return {"delta_ms": np.nan, "curve": curve, "fine": fine}
    j = int(np.argmin([x[1] for x in okf]))
    best = okf[j][0]
    if len(okf) >= 5:  # least-squares parabola over the fine window: sub-ms and less noisy than a 3-point fit
        xs, ys = np.array([x[0] for x in okf]), np.array([x[1] for x in okf])
        a2, a1, _ = np.polyfit(xs - best, ys, 2)
        if a2 > 0 and abs(-a1 / (2 * a2)) <= FINE_HALF_MS:
            best = best - a1 / (2 * a2)
    at_edge = abs(best) > range_ms - COARSE_MS if coarse else False
    ymin = min(x[1] for x in okf)
    y0 = residual(c, others, offs, 0.0, frames)[METRIC] if coarse else np.nan
    return {"delta_ms": float(best), "curve": curve, "fine": fine, "min_px": float(ymin), "at_zero_px": float(y0),
            "at_edge": bool(at_edge)}


def _subwindows(c: CamData, length_s: float = SUB_WINDOW_S) -> list[tuple[float, np.ndarray]]:
    t = c.t0
    lo, hi = t[c.tr.ok][0], t[c.tr.ok][-1]
    n = max(1, int(round((hi - lo) / length_s)))
    edges = np.linspace(lo, hi + 1e-6, n + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        fr = np.flatnonzero((t >= a) & (t < b))
        if len(fr) >= 15:
            out.append((float((a + b) / 2), fr))
    return out


def estimate(work: Path, cams: list[str] | None = None, range_ms: float = 300.0, passes: int = 8,
             log=print, workers: int | None = None) -> dict:
    data = load(work, cams)
    w = default_workers(len(data) * len(data)) if workers is None else max(1, int(workers))
    log(f"[sync] {len(data)} cameras, {w} worker process(es)")
    res = estimate_data(data, range_ms, passes, log, workers=w)
    res["camera_models"] = camera_models(data)
    return res


def _step(c: CamData, others: list[CamData], offs: dict[str, float], frames: np.ndarray | None = None) -> float:
    """Gauss-Newton time step (s) for camera c at its current offset: median over fast landmarks of
    -(d . v) / |v|^2, d = projected - observed (a frame timed too early sees the triangulated dancer where c saw
    it (true - delta) seconds before). NaN when fewer than 20 fast landmarks."""
    return residual(c, others, offs, offs.get(c.name, 0.0), frames)["tau_s"]


# ----------------------------------------------------------------------------------------------- process pool

_POOL_DATA: dict | None = None  # the cameras, inherited by forked workers (or sent once per spawned worker)


def _pool_init(data):
    global _POOL_DATA
    if data is not None:
        _POOL_DATA = data


def _step_task(args) -> float:
    c, offs = args
    return _step(_POOL_DATA[c], _others(_POOL_DATA, c), offs)


def _curves_task(args):
    c, offs, range_ms = args
    return _camera_curves(_POOL_DATA, c, offs, range_ms)


def default_workers(n_tasks: int) -> int:
    """Serial off-pod; on a pod the container's CPU quota (front_pod.cgroup_cpus), at most 16 and n_tasks."""
    from . import env
    if not env.is_pod():
        return 1
    from .front_pod import cgroup_cpus
    return max(1, min(16, n_tasks, cgroup_cpus()))


class _Mapper:
    """map() over a process pool (workers > 1; fork where available so the cameras are not pickled) or serially."""

    def __init__(self, data: dict, workers: int):
        self.data, self.workers, self.pool = data, workers, None

    def __enter__(self):
        global _POOL_DATA
        _POOL_DATA = self.data
        if self.workers > 1:
            import multiprocessing as mp
            fork = "fork" in mp.get_all_start_methods()
            ctx = mp.get_context("fork" if fork else "spawn")
            self.pool = ctx.Pool(self.workers, initializer=_pool_init, initargs=(None if fork else self.data,))
        return self

    def map(self, fn, items):
        return self.pool.map(fn, items, chunksize=1) if self.pool is not None else [fn(x) for x in items]

    def __exit__(self, *exc):
        global _POOL_DATA
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
        _POOL_DATA = None
        return False


def solve_joint(data: dict[str, CamData], offs: dict[str, float] | None = None, iters: int = 8, h: float = 0.010,
                max_step: float = 0.060, log=print, mapper: "_Mapper | None" = None) -> tuple[dict[str, float], np.ndarray, dict]:
    """All offsets at once. Each camera's best offset delta*_c depends on the other cameras' offsets (their
    resampled landmarks feed the triangulation), with geometry-dependent and sometimes negative or > 1 gains, so
    per-camera searches iterated Jacobi-style oscillate (seen on larissa_kadu: cam 01 +72 / +29 / +67 ms). Here
    delta*_c(offs) = offs_c + step_c is linearised, G[c, o] = d step_c / d offs_o by finite differences (h), and
    the fixed point (I - G) delta = delta* - G offs is solved by least squares with a zero-mean gauge row, then the
    median is set to 0. Newton iterations until the largest change is < 0.5 ms."""
    names = sorted(data)
    n = len(names)
    offs = dict(offs or {c: 0.0 for c in names})
    G = np.zeros((n, n))
    info = {"iterations": []}
    for it in range(iters):
        t0 = time.time()
        star = np.zeros(n)
        G[:] = 0.0
        # every camera's step at offs and with each triangulating camera nudged by h: independent evaluations
        tasks, keys = [], []
        for c in names:
            tasks.append((c, dict(offs)))
            keys.append((c, None))
            for o in _others(data, c):
                o2 = dict(offs)
                o2[o.name] = offs.get(o.name, 0.0) + h
                tasks.append((c, o2))
                keys.append((c, o.name))
        steps = dict(zip(keys, mapper.map(_step_task, tasks) if mapper is not None else
                         [_step(data[c], _others(data, c), o) for c, o in tasks]))
        for i, c in enumerate(names):
            s0 = steps[(c, None)]
            if not np.isfinite(s0):
                star[i] = offs[c]
                continue
            star[i] = offs[c] + s0
            for o in _others(data, c):
                s1 = steps[(c, o.name)]
                if np.isfinite(s1):
                    G[i, names.index(o.name)] = (s1 - s0) / h
        cur = np.array([offs[c] for c in names])
        A = np.vstack([np.eye(n) - G, np.full((1, n), 1.0 / n)])
        b = np.concatenate([star - G @ cur, [0.0]])
        new = np.linalg.lstsq(A, b, rcond=None)[0]
        new = cur + np.clip(new - cur, -max_step, max_step)
        new -= np.median(new)
        change = float(np.abs(new - cur).max())
        offs = {c: float(v) for c, v in zip(names, new)}
        info["iterations"].append({"offsets_ms": {c: round(1000 * v, 2) for c, v in offs.items()},
                                   "max_change_ms": round(1000 * change, 2)})
        log(f"[sync] joint iteration {it + 1}: " + " ".join(f"{c} {1000 * offs[c]:+.1f}" for c in names)
            + f" ms (max change {1000 * change:.1f} ms, {time.time() - t0:.0f} s)")
        if change < 0.0005:
            break
    info["converged"] = bool(info["iterations"][-1]["max_change_ms"] < 0.5)
    # the Newton iterates settle into a noisy band (the median step estimate changes with the landmark set that
    # passes the filters): average the second half of them and report their spread as the solve's noise
    tail = info["iterations"][len(info["iterations"]) // 2:]
    avg = {c: float(np.mean([x["offsets_ms"][c] for x in tail])) / 1000.0 for c in names}
    g = float(np.median(list(avg.values())))
    offs = {c: v - g for c, v in avg.items()}
    info["averaged_over"] = len(tail)
    info["iterate_sd_ms"] = {c: round(float(np.std([x["offsets_ms"][c] for x in tail])), 2) for c in names}
    info["G"] = np.round(G, 3).tolist()
    info["cams"] = names
    return offs, G, info


def _camera_curves(data: dict[str, CamData], c: str, offs: dict[str, float], range_ms: float) -> tuple[dict, list]:
    """Camera c's residual curve with the others at offs, and its 2 s sub-window curves."""
    others = _others(data, c)
    r = _search(data[c], others, offs, range_ms)
    rows = []
    for tm, fr in _subwindows(data[c]):
        rr = _search(data[c], others, offs, range_ms, frames=fr, centre_ms=r["delta_ms"], coarse=False)
        d = rr["delta_ms"]
        if not np.isfinite(d) or abs(d - r["delta_ms"]) >= FINE_HALF_MS - 0.5:  # minimum outside the window
            rr = _search(data[c], others, offs, range_ms, frames=fr)
            d = rr["delta_ms"]
        rows.append({"t_mid": tm, "delta_ms": round(float(d), 2) if np.isfinite(d) else None,
                     "min_px": rr.get("min_px"), "n_frames": int(len(fr)), "fine": rr["fine"]})
    return r, rows


def estimate_data(data: dict[str, CamData], range_ms: float = 300.0, passes: int = 8, log=print,
                  workers: int = 1) -> dict:
    """Offsets of in-memory cameras (see the module docstring). The joint Gauss-Newton solve handles the coupling
    between cameras; the reported delta of each camera is then the minimum of its own residual curve with the other
    cameras at their solved offsets (the definition of delta), re-gauged to median 0. Sub-windows use the same curve
    estimator. Agreement between the two estimators is reported (estimator_gap_ms)."""
    names = sorted(data)
    with _Mapper(data, workers) as mapper:
        offs, G, info = solve_joint(data, iters=passes, log=log, mapper=mapper)
        per_cam = mapper.map(_curves_task, [(c, offs, range_ms) for c in names])
    info["workers"] = workers
    curves, subs, final = {}, {}, {}
    for c, (r, rows) in zip(names, per_cam):
        curves[c] = r
        final[c] = r["delta_ms"] / 1000.0 if np.isfinite(r["delta_ms"]) else offs[c]
        subs[c] = rows
        log(f"[sync] cam {c}: joint {1000 * offs[c]:+.1f} ms, curve minimum {r['delta_ms']:+.1f} ms "
            f"({r.get('at_zero_px', np.nan):.2f} -> {r.get('min_px', np.nan):.2f} px); sub-windows "
            + ", ".join(f"{x['t_mid']:.1f}s {x['delta_ms']:+.1f}" if x["delta_ms"] is not None else
                        f"{x['t_mid']:.1f}s n/a" for x in rows))
    g = float(np.median(list(final.values())))
    out = {}
    for i, c in enumerate(names):
        r = curves[c]
        d_ms = 1000 * (final[c] - g)
        ds = [x["delta_ms"] - 1000 * g for x in subs[c] if x["delta_ms"] is not None]
        spread = float(max(ds) - min(ds)) if len(ds) >= 2 else None
        rate = t_ref = None
        if len(ds) >= 2:
            tm = np.array([x["t_mid"] for x in subs[c] if x["delta_ms"] is not None])
            A = np.stack([tm - tm.mean(), np.ones_like(tm)], 1)
            coef = np.linalg.lstsq(A, np.array(ds), rcond=None)[0]
            rate, t_ref = float(coef[0] / 1000.0), float(tm.mean())
        gap = abs(r["delta_ms"] - 1000 * offs[c]) if np.isfinite(r["delta_ms"]) else np.inf
        stable = spread is not None and spread < STABLE_MS
        out[c] = {"delta": round(d_ms / 1000.0, 5), "delta_ms": round(d_ms, 2), "stable": bool(stable),
                  "use": bool(np.isfinite(r["delta_ms"]) and not r.get("at_edge", False)),
                  "spread_ms": None if spread is None else round(spread, 2),
                  "joint_ms": round(1000 * (offs[c] - g), 2), "estimator_gap_ms": round(float(gap), 2),
                  "joint_iterate_sd_ms": info["iterate_sd_ms"][c], "attached": bool(data[c].attached),
                  "sub_windows": [{"t_mid": x["t_mid"], "delta_ms": None if x["delta_ms"] is None else
                                   round(x["delta_ms"] - 1000 * g, 2), "min_px": x["min_px"],
                                   "n_frames": x["n_frames"]} for x in subs[c]],
                  "rate": rate, "t_ref": t_ref, "use_rate": False,
                  "residual_px": {"metric": METRIC, "at_delta": round(float(r.get("min_px", np.nan)), 3),
                                  "at_zero": round(float(r.get("at_zero_px", np.nan)), 3)},
                  "coupling_row": {o: round(float(G[i, j]), 3) for j, o in enumerate(names) if G[i, j] != 0},
                  "at_search_edge": bool(r.get("at_edge", False))}
    shift = lambda rows: [(x[0] - 1000 * g, *x[1:]) for x in rows]  # noqa: E731  (plot in the final gauge)
    return {"cams": names, "offsets": out, "solve": info, "gauge_shift_ms": round(1000 * g, 2),
            "curves": {c: {"coarse": shift(curves[c]["curve"]), "fine": shift(curves[c]["fine"])} for c in names},
            "sub_curves": {c: [{"t_mid": x["t_mid"], "fine": shift(x["fine"])} for x in subs[c]] for c in names}}


# ----------------------------------------------------------------------------------------------- outputs


def _audio_offsets(work: Path) -> dict:
    p = work / "audio" / "quality.json"
    if not p.exists():
        return {}
    q = json.loads(p.read_text())
    src = q.get("source_cam")
    out = {c: v.get("median_ms") for c, v in (q.get("beat_offsets_vs_source") or {}).items()}
    if src:
        out.setdefault(src, 0.0)
    return out


def plot(res: dict, out_png: Path, audio: dict | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = res["cams"]
    n = len(names)
    cols = 4
    rows = int(np.ceil(n / cols)) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.1 * rows))
    axes = np.atleast_2d(axes)
    for k, c in enumerate(names):
        ax = axes[k // cols, k % cols]
        o = res["offsets"][c]
        cur = np.array([x[:2] for x in res["curves"][c]["coarse"]], float)
        fine = np.array([x[:2] for x in res["curves"][c]["fine"]], float)
        g = 0.0  # curves are computed with the other cameras at their solved offsets: absolute delta axis
        if len(cur):
            ax.plot(cur[:, 0] + g, cur[:, 1], color="#888", lw=1, label="scan 20 ms + coarse 4 ms")
        if len(fine):
            ax.plot(fine[:, 0] + g, fine[:, 1], color="#1f77b4", lw=1.6, label="fine (1 ms)")
        for sw, col in zip(res["sub_curves"][c], ("#d62728", "#2ca02c", "#9467bd", "#8c564b")):
            f = np.array([x[:2] for x in sw["fine"]], float)
            if len(f) and np.isfinite(f[:, 1]).any():
                ax.plot(f[:, 0], f[:, 1], color=col, lw=0.9, ls="--", label=f"2 s @ {sw['t_mid']:.1f}s")
        ax.axvline(o["delta_ms"], color="k", lw=1)
        if audio and c in audio and audio[c] is not None:
            ax.axvline(audio[c], color="orange", lw=1, ls=":", label="audio beat offset")
        flag = "stable" if o["stable"] else "UNSTABLE"
        sp = "n/a" if o["spread_ms"] is None else f"{o['spread_ms']:.1f}"
        ax.set_title(f"cam {c}: {o['delta_ms']:+.1f} ms ({flag}, spread {sp} ms)", fontsize=9,
                     color="k" if o["stable"] else "#d62728")
        ax.set_xlabel("delta (ms)", fontsize=8)
        ax.set_ylabel("median along-motion residual (px)", fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=6)
    for k in range(n, (rows - 1) * cols):
        axes[k // cols, k % cols].axis("off")
    # summary row: offsets vs sub-windows vs audio
    ax = fig.add_subplot(rows, 1, rows)
    for k in range((rows - 1) * cols, rows * cols):
        axes[k // cols, k % cols].axis("off")
    x = np.arange(n)
    d = [res["offsets"][c]["delta_ms"] for c in names]
    ax.bar(x - 0.2, d, 0.4, label="video (geometry) delta", color="#1f77b4")
    if audio:
        a = [audio.get(c) if audio.get(c) is not None else np.nan for c in names]
        ax.bar(x + 0.2, a, 0.4, label="audio beat offset vs source (beats stage)", color="orange")
    for i, c in enumerate(names):
        for sw in res["offsets"][c]["sub_windows"]:
            if sw["delta_ms"] is not None:
                ax.plot(i - 0.2, sw["delta_ms"], "k.", ms=4)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x, [f"cam {c}" for c in names])
    ax.set_ylabel("ms")
    ax.set_title("per-camera offsets (dots = 2 s sub-windows); corrected t = t + delta", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def regauge(res: dict, ref_doc: dict, ref_name: str) -> dict:
    """Pin the gauge to another take of the SAME recordings on the SAME audio clocks (larissa_kadu_full's segment takes
    vs larissa_kadu_38): shift every offset by g so that the median over the shared, used cameras of
    (delta - delta_ref) is 0. Geometry fixes only relative offsets, so this changes no camera's offset relative to
    another; it makes the segments' corrected times agree with the reference take's (stitching). Returns the record."""
    ref = ref_doc.get("offsets", ref_doc)
    diffs = {c: res["offsets"][c]["delta_ms"] - float(ref[c]["delta_ms"]) for c in res["offsets"]
             if c in ref and isinstance(ref[c], dict) and ref[c].get("delta_ms") is not None
             and res["offsets"][c].get("use", True) and ref[c].get("use", True)}
    if not diffs:
        return {"ref": ref_name, "applied": False, "why": "no shared camera"}
    g = float(np.median(list(diffs.values())))
    for c, o in res["offsets"].items():
        o["delta_ms"] = round(o["delta_ms"] - g, 2)
        o["delta"] = round(o["delta_ms"] / 1000.0, 5)
        if o.get("joint_ms") is not None:
            o["joint_ms"] = round(o["joint_ms"] - g, 2)
        for sw in o.get("sub_windows") or []:
            if sw.get("delta_ms") is not None:
                sw["delta_ms"] = round(sw["delta_ms"] - g, 2)
        if c in ref and isinstance(ref[c], dict) and ref[c].get("delta_ms") is not None:
            o["delta_vs_ref_ms"] = round(o["delta_ms"] - float(ref[c]["delta_ms"]), 2)
    shift = lambda rows: [(x[0] - g, *x[1:]) for x in rows]  # noqa: E731
    for c in res.get("curves", {}):
        res["curves"][c] = {k: shift(v) for k, v in res["curves"][c].items()}
    for c in res.get("sub_curves", {}):
        res["sub_curves"][c] = [{**x, "fine": shift(x["fine"])} for x in res["sub_curves"][c]]
    return {"ref": ref_name, "applied": True, "shift_ms": round(-g, 2), "shared_cams": sorted(diffs),
            "diff_before_ms": {c: round(v, 2) for c, v in sorted(diffs.items())},
            "diff_after_ms": {c: round(v - g, 2) for c, v in sorted(diffs.items())}}


def run(take_name: str, cams: list[str] | None = None, range_ms: float = 300.0, workers: int | None = None,
        gauge_ref: str | None = None) -> dict:
    work = md.take_work(take_name)
    if not (work / "mamma" / "ma_2d_native").exists():
        raise SystemExit("[sync] needs mamma/ma_2d_native/<cam>.npz (python -m dancecap mamma prepare first)")
    t0 = time.time()
    res = estimate(work, cams, range_ms, workers=workers)
    gauge = None
    if gauge_ref:  # same recordings, same clocks: the reference take's offsets must have been measured on them too
        ref_p = md.take_work(gauge_ref) / "sync" / "offsets.json"
        ref_doc = json.loads(ref_p.read_text())
        mine = take_clocks(work) or {}
        if (ref_doc.get("clocks") or {}).get("alignment") != mine.get("alignment"):
            raise SystemExit(f"[sync] --gauge-ref {gauge_ref}: its offsets were measured on other audio clocks")
        gauge = regauge(res, ref_doc, gauge_ref)
        print(f"[sync] gauge pinned to {gauge_ref}: shift {gauge.get('shift_ms')} ms over {gauge.get('shared_cams')}")
    audio = _audio_offsets(work)
    doc = {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "what": "per-camera residual video offset: corrected reference time = t + delta (+ rate * (t - t_ref) "
                   "only where use_rate is true)",
           "gauge": ("median over cameras = 0 (geometry fixes relative offsets only)" if not (gauge or {}).get("applied")
                     else f"pinned to {gauge_ref}: median over shared cameras of (delta - delta_{gauge_ref}) = 0"),
           **({"gauge_ref": gauge} if gauge else {}),
           "method": {"landmarks": f"ma_2d native, every {LMK_STEP}th of 512, sigma < {SIGMA_REL_MAX} x crop, "
                                    f"vis > {VIS_MIN}",
                      "triangulation": "uncertainty-weighted DLT from the other cameras resampled to t + delta "
                                       f"(>= 2 views, reprojects within {TRI_REL_MAX} x crop)",
                      "search_ms": {"range": range_ms, "scan": SCAN_MS, "coarse": COARSE_MS, "coarse_half": COARSE_HALF_MS,
                                     "fine": FINE_MS, "fine_half": FINE_HALF_MS},
                      "sub_window_s": SUB_WINDOW_S, "stable_if_spread_below_ms": STABLE_MS},
           "offsets": res["offsets"], "audio_beat_offsets_ms": audio,
           # the audio clocks these deltas sit on top of (offsets_check refuses the file on any other clocks)
           "clocks": {"alignment": (take_clocks(work) or {}).get("alignment"), "sources": frame_clocks(work)},
           "camera_models": res.get("camera_models"),  # camera-type model per camera (cameras/<cam>.npz)
           "seconds": round(time.time() - t0, 1), "workers": res["solve"].get("workers", 1),
           "host": "pod" if os.environ.get("DANCECAP_POD") == "1" else "local"}
    out = work / "sync" / "offsets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    (work / "sync" / "curves.json").write_text(json.dumps({"curves": res["curves"], "sub_curves": res["sub_curves"]}))
    png = plot(res, work / "review" / "sync_curves.png", audio)
    print(f"[sync] -> {out}, {png}")
    return doc
