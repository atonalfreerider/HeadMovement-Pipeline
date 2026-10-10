"""Cameras that are NOT in the room model: per-frame localisation of new handheld phones against the existing SfM map.

The larissa_kadu_full world package holds the room model of the demo shoot (main model: 114 images of six phones, 17.8 k
points; sub-models of two more phones in their own frames). The class recap was filmed in the same room by seven OTHER
handheld phones, so none of their frames is a map image. dancecap.cameras.localize assumes every query camera has registered
map images of its own (neighbour focal, same-camera references); this module drops that assumption.

  calibrate   anchors every ~4 s: ALIKED + LightGlue against the map images (the first anchors against ALL of them, then
              against the camera's best references), P4Pf with a wide focal range, then ONE focal + k1 for the whole video
              (joint robust LM over the anchors, cameras/typed.gn_solve: the phones are handheld and never zoom, owner
              2026-10-09). -> calib/<cam>.json
  track       one 60 s window (+ 4 s overlap each side) in presentation order: every `stride`-th frame is localised with the
              intrinsics FIXED (P3P + pose-only refinement) against the previous frame's inlier 2D-3D correspondences, the last
              map anchor and - every `map_every` frames or when tracking gets weak - the camera's best map images; a final
              batched Gauss-Newton solve over every inlier correspondence of each frame (fixed focal + k1) and a light
              robust smoothing give the pose; the skipped frames are interpolated (slerp + lerp) and flagged.
              -> chunks/<cam>_<k>.npz

Poses are in the SfM frame of the model used (main: the room model; sub_05 / sub_06: a sub-model's own frame, NOT metric and
NOT in the room frame: those cameras need the attach step), exactly like cameras/<cam>.npz of the demo (R, tvec = cam_from_world).
Frames are decoded straight from the video by ffmpeg (autorotate: DISPLAY orientation, the one of the owner's 2D pose JSONs),
down-scaled to `long_side` px, and never written to disk; the people / spectators of the pose JSON are masked out of the
features (convex hull of the keypoints + a margin) exactly like the demo's YOLO masks.

    python -m dancecap.cameras.newcam calib <cam> [--model main|sub_05|sub_06] [--world <dir>] [--src <dir>] ...
    python -m dancecap.cameras.newcam track <cam> <chunk k> ...
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .features import FeatureSettings, Features, keep_outside, prefill
from .localize import LocSettings, solve_pose, reproj, _dedupe
from . import typed as T

from ..take import PACKAGE_ROOT

WORK = PACKAGE_ROOT / "work" / "larissa_kadu_recap"      # on a pod PACKAGE_ROOT/work -> /opt/hm/work
WINDOW = (0.0, 360.48)         # the take's window on the master clock (takes/larissa_kadu_recap.toml)
CHUNK_S, OVERLAP_S = 60.0, 4.0
SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[\d.]+)")


# ------------------------------------------------------------------------------------------------ settings


@dataclass
class NewCamSettings:
    long_side: int = 1296          # frames are decoded at (at most) this long side; smaller ones are up-sampled inside ALIKED
    n_kp_anchor: int = 4096        # keypoints of a map-anchor frame
    n_kp_track: int = 2560         # keypoints of a tracked frame (top by score)
    stride: int = 2                # localise every stride-th grid frame, interpolate the others
    map_every: int = 8             # a map anchor every N localised frames (stride 2 -> every 0.53 s)
    anchor_s: float = 4.0          # calibration anchors
    n_refs_calib_full: int = 3     # calibration anchors matched against ALL map images
    n_refs_keep: int = 24          # best references kept per camera after those
    n_refs_anchor: int = 6         # references per map anchor
    min_inliers: int = 40          # accept a pose
    min_inl_map: int = 60          # accept a map-anchor pose (more of them: they restart the chain)
    max_reproj_px: float = 3.5     # median inlier reprojection of an accepted pose (work px)
    ransac_px: float = 5.0
    mask_margin: float = 0.10      # people-mask growth around the keypoint hull, as a fraction of the person's size
    min_kp_conf: float = 0.25
    drop_dilate_px: int = 6
    border_px: int = 4
    smooth_sigma_s: float = 0.05   # light robust smoothing of the localised poses (as cameras/pipeline.stage_smooth)
    retrieval_every: int = 25      # at most one exhaustive (all map images) retry per N failed frames


# ------------------------------------------------------------------------------------------------ video and frame table


def probe(path: Path) -> dict:
    from ..ingest import probe as _p
    return _p(Path(path))


def video_pts(path: Path, cache: Path | None = None) -> np.ndarray:
    """Presentation timestamps (s) of every video frame, sorted (ffprobe packets: demux only, fast). Cached as JSON."""
    path = Path(path)
    if cache is not None and cache.exists():
        d = json.loads(cache.read_text())
        if d.get("size") == path.stat().st_size:
            return np.array(d["pts"], float)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    pts = np.sort(np.array([float(x.split(",")[0]) for x in out.split() if x.strip() and x.split(",")[0] not in ("N/A", "")]))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"size": path.stat().st_size, "pts": [round(float(x), 6) for x in pts]}))
    return pts


def frame_table(pts: np.ndarray, slope: float | None, offset: float | None, avg_fps: float, window=WINDOW,
                max_fps: float = 30.0, pad: float = 0.0) -> dict:
    """Which video frames form the canonical window frames of a camera (the demo's frames.json rule): every frame whose
    reference time slope * pts + offset lies in [window - pad], decimated by round(fps / max_fps) when fps > 1.5 max_fps
    (the 59.94 fps master keeps every 2nd frame, by presentation index). A camera without a clock (Tony) keeps every frame.

    Returns {n (video frame index), pts, t (reference s or NaN), every}. `i` (position in n) is the canonical frame index."""
    every = max(1, int(round(avg_fps / max_fps))) if avg_fps > 1.5 * max_fps else 1
    n = np.arange(len(pts))
    if slope is None or offset is None:
        t = np.full(len(pts), np.nan)
        keep = np.ones(len(pts), bool)
    else:
        t = slope * pts + offset
        keep = (t >= window[0] - pad) & (t <= window[1] + pad)
    keep &= (n % every == 0)
    return {"n": n[keep], "pts": pts[keep], "t": t[keep], "every": every}


def chunk_windows(t0: float = WINDOW[0], t1: float = WINDOW[1], core_s: float = CHUNK_S, overlap_s: float = OVERLAP_S) -> list[dict]:
    """Equal cores of the master clock nearest to `core_s` (360.48 s -> 6 x 60.08 s), each with `overlap_s` into its neighbours
    (none past the window ends): [{k, core: [a, b], window: [a - o, b + o]}]."""
    n = max(1, int(round((t1 - t0) / core_s)))
    edges = [t0 + j * (t1 - t0) / n for j in range(n + 1)]
    return [{"k": k, "core": [round(a, 3), round(b, 3)], "window": [round(max(t0, a - overlap_s), 3), round(min(t1, b + overlap_s), 3)]}
            for k, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]


def work_size(w: int, h: int, long_side: int) -> tuple[int, int, float]:
    """(w', h', scale) of the working resolution: long side capped at `long_side` (even sizes for ffmpeg), never enlarged."""
    s = min(1.0, long_side / max(w, h))
    return (int(round(w * s / 2) * 2), int(round(h * s / 2) * 2), s) if s < 1.0 else (w, h, 1.0)


class FrameStream:
    """ffmpeg -> BGR frames of one window, in presentation order, with the video frame index and PTS of each.

    A producer thread decodes and prepares (mask) frames ahead of the consumer. Only frames whose video index n satisfies
    n % every == 0 and n in `want` (a set or None) are yielded; the others are decoded and dropped (a 59.94 fps master)."""

    def __init__(self, path: Path, pts_sorted: np.ndarray, start: float, end: float, size: tuple[int, int],
                 want: set | None = None, every: int = 1, prepare=None, depth: int = 6, threads: int = 4):
        self.path, self.pts, self.start, self.end, self.size = Path(path), pts_sorted, start, end, size
        self.want, self.every, self.prepare, self.depth = want, every, prepare, depth
        self.threads = threads

    def _run(self, q: queue.Queue) -> None:
        w, h = self.size
        seek = max(0.0, self.start - 1.0)
        cmd = ["ffmpeg", "-v", "info", "-nostats", "-threads", str(self.threads), "-ss", f"{seek:.6f}",
               "-t", f"{self.end - seek + 0.5:.6f}", "-copyts", "-i", str(self.path), "-map", "0:v:0",
               "-vf", f"scale={w}:{h}:flags=area,showinfo", *_passthrough(), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        log: list[float] = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 24)
        tb = (probe(self.path).get("time_base"))

        def read_err():
            from ..ingest import showinfo_pts
            for line in proc.stderr:
                m = SHOWINFO.search(line.decode("utf-8", "replace"))
                if m:
                    log.append(showinfo_pts(m, tb))
        th = threading.Thread(target=read_err, daemon=True)
        th.start()
        fb = w * h * 3
        i = 0
        try:
            while True:
                buf = proc.stdout.read(fb)
                if len(buf) < fb:
                    break
                for _ in range(200):
                    if len(log) > i:
                        break
                    time.sleep(0.005)
                pts = log[i] if len(log) > i else float("nan")
                i += 1
                if not (self.start - 1e-6 <= pts <= self.end + 1e-6):
                    continue
                n = int(np.searchsorted(self.pts, pts - 1e-5))
                if n % self.every or (self.want is not None and n not in self.want):
                    continue
                img = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
                item = (n, pts, img, self.prepare(n, img) if self.prepare else None)
                q.put(item)
        finally:
            try:
                proc.stdout.close()
            except OSError:
                pass
            proc.wait()
            th.join(timeout=2)
            q.put(None)

    def __iter__(self):
        q: queue.Queue = queue.Queue(self.depth)
        t = threading.Thread(target=self._run, args=(q,), daemon=True)
        t.start()
        while True:
            it = q.get()
            if it is None:
                break
            yield it
        t.join(timeout=5)


_PASS: list[str] = []


def _passthrough() -> list[str]:
    if not _PASS:
        from ..ingest import passthrough_args
        _PASS.extend(passthrough_args())
    return list(_PASS)


# ------------------------------------------------------------------------------------------------ 2D poses -> people masks


def load_pose_frames(path: Path) -> list[np.ndarray]:
    """The owner's per-frame 2D pose JSON (.json or .json.gz): list over frames of (P, 17, 3) float32 arrays [X, Y, conf] in
    DISPLAY pixels of the video."""
    path = Path(path)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rb") as f:
        raw = json.load(f)
    out = []
    for fr in raw:
        if not fr:
            out.append(np.zeros((0, 17, 3), np.float32))
            continue
        out.append(np.array([[[k["X"], k["Y"], k["Z"]] for k in person] for person in fr], np.float32).reshape(-1, 17, 3))
    return out


def people_mask(persons: np.ndarray, scale: float, size: tuple[int, int], margin: float = 0.10, min_conf: float = 0.25) -> np.ndarray:
    """uint8 0/255 mask (h, w) of the people of one frame: convex hull of each person's confident keypoints grown by
    `margin` x the person's size (the keypoints lie inside the body: the hull misses the thickness of the limbs, the top of the
    head and the feet), filled. Degenerate hulls (a side view: nearly collinear keypoints) still give a thick stroke."""
    w, h = size
    m = np.zeros((h, w), np.uint8)
    for p in persons:
        sel = p[:, 2] >= min_conf
        if sel.sum() < 3:
            continue
        pts = (p[sel, :2] * scale).astype(np.float32)
        ext = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 8.0))
        hull = np.round(cv2.convexHull(pts).reshape(-1, 2)).astype(np.int32)
        if len(hull) >= 3:
            cv2.fillConvexPoly(m, hull, 255)
        cv2.polylines(m, [hull.reshape(-1, 1, 2)], True, 255, thickness=max(1, int(round(2 * margin * ext))), lineType=cv2.LINE_8)
    return m


# ------------------------------------------------------------------------------------------------ features and matching


class Feat:
    """Keypoints + descriptors of one image (native work px, OpenCV pixel centres) with lazy GPU copies."""

    __slots__ = ("kpts", "desc", "size", "scores", "_g")

    def __init__(self, kpts, desc, size, scores=None):
        self.kpts, self.desc, self.size, self.scores, self._g = kpts, desc, size, scores, None

    def __len__(self) -> int:
        return len(self.kpts)

    def gpu(self, device: str):
        if self._g is None:
            import torch
            self._g = {"keypoints": torch.from_numpy(self.kpts)[None].to(device),
                       "descriptors": torch.from_numpy(np.asarray(self.desc, np.float32))[None].to(device),
                       "image_size": torch.from_numpy(np.asarray(self.size, np.float32))[None].to(device)}
        return self._g


class Engine:
    """ALIKED extractor + LightGlue matcher (the demo's models and settings, cameras/features.py), arrays in, arrays out."""

    def __init__(self, s: NewCamSettings | None = None, device: str = "cuda"):
        self.s = s or NewCamSettings()
        self.device = device
        self._F = Features(Path("."), Path("."), FeatureSettings(max_keypoints=self.s.n_kp_anchor,
                                                               long_side=self.s.long_side), device)

    def extract(self, img_bgr: np.ndarray, mask: np.ndarray | None, n_keep: int | None = None) -> Feat:
        import torch

        from ..gpu import amp_dtype
        h, w = img_bgr.shape[:2]
        mask = mask if mask is not None else np.zeros((h, w), np.uint8)
        filled = prefill(img_bgr, mask)
        rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        resize = self.s.long_side if max(h, w) < self.s.long_side else None   # 480p frames: up-sampled, as the map's own
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype(), enabled=self.device == "cuda"):
            f = self._F._ext().extract(ten, resize=resize)
        k = f["keypoints"][0].float().cpu().numpy()
        sc = f["keypoint_scores"][0].float().cpu().numpy()
        de = f["descriptors"][0].float().cpu().numpy()
        b = self.s.border_px
        keep = keep_outside(k, mask, self.s.drop_dilate_px) & (k[:, 0] >= b) & (k[:, 1] >= b) & \
            (k[:, 0] <= w - 1 - b) & (k[:, 1] <= h - 1 - b)
        k, sc, de = k[keep], sc[keep], de[keep]
        if n_keep and len(k) > n_keep:
            top = np.argsort(-sc)[:n_keep]
            k, sc, de = k[top], sc[top], de[top]
        return Feat(k.astype(np.float32), de.astype(np.float16), np.array([w, h], np.int32), sc.astype(np.float32))

    def match(self, a: Feat, b: Feat) -> np.ndarray:
        """(M, 2) int64 index pairs into a.kpts, b.kpts."""
        import torch
        if len(a) < 8 or len(b) < 8:
            return np.zeros((0, 2), np.int64)
        with torch.inference_mode():
            out = self._F._mat()({"image0": a.gpu(self.device), "image1": b.gpu(self.device)})
        return out["matches"][0].cpu().numpy().astype(np.int64).reshape(-1, 2)


@dataclass
class QRef:
    """A reference image with 3D points: `kp_idx` indexes feat.kpts, xyz / pid the SfM point of each."""
    key: str
    feat: Feat
    kp_idx: np.ndarray
    xyz: np.ndarray
    pid: np.ndarray
    _lut: np.ndarray | None = field(default=None, repr=False)

    def lut(self) -> np.ndarray:
        if self._lut is None:
            lut = -np.ones(len(self.feat) + 1, np.int64)
            lut[self.kp_idx] = np.arange(len(self.kp_idx))
            self._lut = lut
        return self._lut


def lift(eng: Engine, q: Feat, ref: QRef, cache: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(query keypoint idx, xyz, pid) of the LightGlue matches q <-> ref that land on a keypoint with a 3D point."""
    k = id(ref)
    if cache is not None and k in cache:
        m = cache[k]
    else:
        m = eng.match(q, ref.feat)
        if cache is not None:
            cache[k] = m
    if len(m) == 0 or len(ref.kp_idx) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0, np.int64)
    j = ref.lut()[m[:, 1]]
    ok = j >= 0
    return m[ok, 0], ref.xyz[j[ok]], ref.pid[j[ok]]


@dataclass
class Pose:
    ok: bool = False
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    f: float = float("nan")
    k1: float = 0.0
    n_inliers: int = 0
    n_corr: int = 0
    reproj_px: float = float("nan")
    source: str = ""            # map | track
    ref: QRef | None = None     # inlier 2D-3D correspondences of this frame (kp_idx into its own feature set)


def localize(eng: Engine, q: Feat, refs: list[QRef], w: int, h: int, f0: float, k1: float, ls: LocSettings,
             key: str = "q") -> Pose:
    """Pose of a query from the 2D-3D correspondences lifted through `refs`. ls.fix_focal: P3P with f0 and a pose-only
    refinement (the calibrated camera); else P4Pf (+ P3P with f0) with the focal estimated (calibration anchors)."""
    cache: dict = {}
    qi, xyz, pid = [], [], []
    for r in refs:
        a, b, c = lift(eng, q, r, cache)
        qi.append(a), xyz.append(b), pid.append(c)
    if not qi:
        return Pose()
    qi, xyz, pid = _dedupe(np.concatenate(qi), np.concatenate(xyz), np.concatenate(pid))
    if len(qi) < max(8, ls.min_inliers // 2):
        return Pose(n_corr=len(qi))
    kp = q.kpts[qi].astype(np.float64) + 0.5          # COLMAP pixel convention
    sol = solve_pose(kp, xyz, w, h, f0, k1, ls)
    if sol is None:
        return Pose(n_corr=len(qi))
    inl = sol["inliers"]
    e = reproj(sol["R"], sol["t"], sol["f"], sol["k1"], w, h, xyz[inl], kp[inl])
    return Pose(ok=bool(sol["n"] >= ls.min_inliers), R=sol["R"], t=sol["t"], f=sol["f"], k1=sol["k1"], n_inliers=int(sol["n"]),
                n_corr=len(qi), reproj_px=float(np.median(e)) if len(e) else float("nan"),
                ref=QRef(key, q, qi[inl], xyz[inl], pid[inl]))


# ------------------------------------------------------------------------------------------------ the map


class MapRefs:
    """The registered images of one model of the world package as QRefs (features from the package's cache)."""

    def __init__(self, world_dir: Path, model: str = "main"):
        import pycolmap

        from .features import list_images
        from .localize import map_refs
        self.world_dir, self.model_name = Path(world_dir), model
        ref = self.world_dir / "ref" if (self.world_dir / "ref").is_dir() else self.world_dir
        mdir = ref / "cameras" / ("sfm" if model == "main" else f"sfm_work/{model}/best")
        self.model = pycolmap.Reconstruction(str(mdir))
        info = json.loads((ref / "cameras" / "sfm_work" / "map_images.json").read_text())
        want = {im.name for im in self.model.images.values() if im.has_pose}
        cams = sorted({n.split("/")[1] for n in info["images"]})
        recs = list_images(ref, cams)
        images = [r for c in cams for r in recs[c] if r.name in want]
        F = Features(ref, ref / "cameras" / "cache", FeatureSettings())
        base = map_refs(self.model, images, F)
        self.refs: dict[str, QRef] = {}
        for key, r in base.items():
            d = F.get(r.rec)
            self.refs[key] = QRef(key, Feat(d["kpts"].astype(np.float32), d["desc"], d["size"], d.get("scores")),
                                  r.kp_idx, r.xyz, r.pid)
        self.points = np.array([p.xyz for p in self.model.points3D.values()]).reshape(-1, 3)
        self.rgb = np.array([p.color for p in self.model.points3D.values()]).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.refs)

    def select(self, keys: list[str]) -> list[QRef]:
        return [self.refs[k] for k in keys if k in self.refs]


# ------------------------------------------------------------------------------------------------ calibration


def anchor_frames(n_frames: int, fps: float, anchor_s: float) -> np.ndarray:
    """Indices (into the window's frame list) of the calibration anchors: every anchor_s seconds, centred."""
    step = max(1, int(round(anchor_s * fps)))
    return np.arange(step // 2, n_frames, step)


def fit_intrinsics(poses: list[Pose], w: int, h: int, f_guess: float, k1_guess: float = 0.0, max_pts: int = 600,
                   min_inl: int = 150) -> dict:
    """ONE focal + k1 for a video from its calibration anchors: joint robust LM (typed.gn_solve) with per-anchor pose and a
    shared log focal + k1. Returns {f, k1, n_used, cost, per-anchor reprojection before/after, f_individual spread}."""
    use = [p for p in poses if p.ok and p.ref is not None and p.n_inliers >= min_inl]
    if len(use) < 3:
        use = [p for p in poses if p.ok and p.ref is not None and p.n_inliers >= 40]
    if len(use) < 3:
        return {"ok": False, "why": f"only {len(use)} anchors localised"}
    use = use[:: max(1, len(use) // 40)] if len(use) > 40 else use
    rng = np.random.default_rng(0)
    XX, xx, ff = [], [], []
    for j, p in enumerate(use):
        X, x = p.ref.xyz, p.ref.feat.kpts[p.ref.kp_idx].astype(np.float64)
        if len(X) > max_pts:
            sel = rng.choice(len(X), max_pts, replace=False)
            X, x = X[sel], x[sel]
        XX.append(X), xx.append(x), ff.append(np.full(len(X), j))
    n = np.array([len(a) for a in XX])
    b = T.Batch(np.arange(len(use)), np.concatenate(XX), np.concatenate(xx), np.concatenate(ff), np.r_[0, np.cumsum(n)[:-1]].astype(int))
    R = np.array([p.R for p in use])
    C = np.array([-p.R.T @ p.t for p in use])
    f_ind = np.array([p.f for p in use])
    f0 = float(np.median(f_ind)) if np.isfinite(f_ind).all() else f_guess
    cx, cy = T._cc(w, h)
    lf = np.full(len(use), np.log(f0))
    k1 = np.full(len(use), float(k1_guess))
    r = T.gn_solve(b, R, C, lf, k1, cx, cy, {"rot": "P", "cen": "P", "lf": "S", "k1": "S"}, huber=2.0, iters=30)
    f, k = float(np.exp(r["lf"][0])), float(r["k1"][0])
    # reprojection per anchor under the joint intrinsics
    uv, z = T.project_batch(b.X, r["R"][b.fid], r["C"][b.fid], np.full(len(b.X), f), np.full(len(b.X), k), cx, cy)
    e = np.linalg.norm(uv - b.x, axis=1)
    med = np.array([np.median(e[b.fid == j]) for j in range(len(use))])
    return {"ok": True, "f": f, "k1": k, "n_used": len(use), "cost_before": r["cost_before"], "cost_after": r["cost_after"],
            "anchor_reproj_med_px": round(float(np.median(med)), 3), "anchor_reproj_p90_px": round(float(np.percentile(med, 90)), 3),
            "f_individual_median": round(float(np.median(f_ind)), 2),
            "f_individual_rel_spread": round(float(1.4826 * np.median(np.abs(f_ind - np.median(f_ind))) / np.median(f_ind)), 4),
            "f_individual_rel_range": round(float((f_ind.max() - f_ind.min()) / np.median(f_ind)), 4)}


def calibrate(cam: dict, mp: MapRefs, src: Path, poses_path: Path | None, out_dir: Path, s: NewCamSettings,
              window=WINDOW, log=print, max_anchors: int | None = None) -> dict:
    """Calibration of one camera: anchors -> map poses -> one focal + k1. Writes out_dir/<cam>.json (and returns it)."""
    t_start = time.time()
    cid = cam["id"]
    path = Path(src) / cam["file"]
    pts = video_pts(path, out_dir / f"{cid}_pts.json")
    tab = frame_table(pts, cam.get("slope"), cam.get("offset"), cam["avg_fps"], window)
    w0, h0 = int(cam["width"]), int(cam["height"])
    w, h, sc = work_size(w0, h0, s.long_side)
    fps = float(cam["avg_fps"])
    a_idx = anchor_frames(len(tab["n"]), fps / tab["every"], s.anchor_s)
    if max_anchors:
        a_idx = a_idx[:: max(1, len(a_idx) // max_anchors)]
    want = {int(tab["n"][i]) for i in a_idx}
    pose_frames = load_pose_frames(poses_path) if poses_path and Path(poses_path).exists() else None

    def prep(n, img):
        return people_mask(pose_frames[n], sc, (w, h), s.mask_margin, s.min_kp_conf) if pose_frames and n < len(pose_frames) else None

    eng = Engine(s)
    ls = LocSettings(max_focal_change=2.0, ransac_px=s.ransac_px, min_inliers=s.min_inliers)
    f_guess = 0.85 * max(w, h)
    keys = list(mp.refs)
    counts: dict[str, float] = {k: 0.0 for k in keys}
    poses: list[Pose] = []
    rows = []
    t0, t1 = float(tab["pts"][0]), float(tab["pts"][-1])
    n_seen = 0
    for n, pt, img, mask in FrameStream(path, pts, t0, t1, (w, h), want=want, every=1, prepare=prep):
        q = eng.extract(img, mask, s.n_kp_anchor)
        full = n_seen < s.n_refs_calib_full or len(poses) and not any(p.ok for p in poses[-3:])
        order = keys if full else sorted(keys, key=lambda k: -counts[k])[:s.n_refs_keep]
        # rank the candidate references by their own lifted correspondences, keep the best few for the pose
        cache: dict = {}
        scored = []
        for k in order:
            a, b_, c = lift(eng, q, mp.refs[k], cache)
            scored.append((len(a), k))
        scored.sort(reverse=True)
        top = [k for _, k in scored[:s.n_refs_anchor + 2]]
        for c_, k in scored:
            counts[k] += c_ if n_seen < 2 * s.n_refs_calib_full else 0.7 * c_
        p = localize(eng, q, [mp.refs[k] for k in top], w, h, f_guess, 0.0, ls, key=f"a{n}")
        n_seen += 1
        poses.append(p)
        rows.append({"n": int(n), "t": float(pt), "ok": bool(p.ok), "n_inliers": p.n_inliers, "n_corr": p.n_corr,
                     "f_work": p.f if p.ok else None, "reproj_px": p.reproj_px if p.ok else None, "top": top[:3],
                     "top_counts": [int(c) for c, _ in scored[:3]], "exhaustive": bool(full)})
        if n_seen % 10 == 0:
            log(f"[calib {cid}] {n_seen}/{len(want)} anchors, {sum(x.ok for x in poses)} ok, {time.time() - t_start:.0f} s")
    ok = [p for p in poses if p.ok]
    res = {"cam": cid, "model": mp.model_name, "work_size": [w, h], "scale_to_native": 1.0 / sc, "native_size": [w0, h0],
           "n_anchors": len(poses), "n_ok": len(ok), "window_frames": int(len(tab["n"])), "every": tab["every"],
           "anchors": rows}
    # reference ranking for the tracker
    res["map_refs"] = [k for k, c in sorted(counts.items(), key=lambda kv: -kv[1])[:s.n_refs_keep] if c > 0]
    if ok:
        intr = fit_intrinsics(poses, w, h, f_guess)
        res["intrinsics"] = intr
        if intr.get("ok"):
            res["f_work"], res["k1"] = intr["f"], intr["k1"]
            res["f_native"] = intr["f"] * res["scale_to_native"]
            res["f_over_maxdim"] = intr["f"] / max(w, h)
            # per-anchor pose with the FIXED intrinsics: how many anchors agree with the calibrated model
            rp = []
            for p in ok:
                if p.ref is None:
                    continue
                e = reproj(p.R, p.t, intr["f"], intr["k1"], w, h, p.ref.xyz, p.ref.feat.kpts[p.ref.kp_idx].astype(np.float64) + 0.5)
                rp.append(float(np.median(e)))
            res["anchor_reproj_fixed_med_px"] = round(float(np.median(rp)), 3) if rp else None
    res["seconds"] = round(time.time() - t_start, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cid}.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"[calib {cid}] {len(ok)}/{len(poses)} anchors, f/max {res.get('f_over_maxdim', float('nan')):.3f} "
        f"k1 {res.get('k1', float('nan')):+.4f}, {res['seconds']} s")
    return res


# ------------------------------------------------------------------------------------------------ tracking


def refine_fixed(poses: list[Pose], w: int, h: int, f: float, k1: float, huber: float = 2.0) -> dict:
    """Final per-frame pose: batched robust Gauss-Newton over EVERY inlier correspondence of each frame with the focal and
    k1 held (typed.gn_solve, roles rot/cen per frame, lf/k1 held). Updates the poses in place; returns the cost summary."""
    idx = [i for i, p in enumerate(poses) if p.ok and p.ref is not None and len(p.ref.kp_idx) >= 6]
    if not idx:
        return {"n": 0}
    cx, cy = T._cc(w, h)
    cb = ca = 0.0
    i = 0
    cap = 200_000
    while i < len(idx):
        j, tot = i, 0
        while j < len(idx) and (tot + len(poses[idx[j]].ref.kp_idx) <= cap or j == i):
            tot += len(poses[idx[j]].ref.kp_idx)
            j += 1
        sel = idx[i:j]
        XX = [poses[a].ref.xyz for a in sel]
        xx = [poses[a].ref.feat.kpts[poses[a].ref.kp_idx].astype(np.float64) for a in sel]
        n = np.array([len(a) for a in XX])
        b = T.Batch(np.arange(len(sel)), np.concatenate(XX), np.concatenate(xx),
                    np.concatenate([np.full(len(a), k) for k, a in enumerate(XX)]), np.r_[0, np.cumsum(n)[:-1]].astype(int))
        R = np.array([poses[a].R for a in sel])
        C = np.array([-poses[a].R.T @ poses[a].t for a in sel])
        r = T.gn_solve(b, R, C, np.full(len(sel), np.log(f)), np.full(len(sel), k1), cx, cy,
                       {"rot": "P", "cen": "P", "lf": "F", "k1": "F"}, huber=huber, iters=8)
        cb, ca = cb + r["cost_before"], ca + r["cost_after"]
        uv, z = T.project_batch(b.X, r["R"][b.fid], r["C"][b.fid], np.full(len(b.X), f), np.full(len(b.X), k1), cx, cy)
        e = np.linalg.norm(uv - b.x, axis=1)
        for k, a in enumerate(sel):
            m = b.fid == k
            p = poses[a]
            p.R = r["R"][k]
            p.t = -r["R"][k] @ r["C"][k]
            ek = e[m]
            p.n_inliers = int((ek < 3.0).sum())            # inliers under the refined pose (3 px at work resolution)
            p.reproj_px = float(np.median(ek)) if len(ek) else float("nan")
        i = j
    return {"n": len(idx), "cost_before": round(cb, 2), "cost_after": round(ca, 2)}


def median_reproj(p: Pose, R: np.ndarray, tvec: np.ndarray, f: float, k1: float, w: int, h: int) -> float:
    """Median reprojection (work px) of a frame's inlier correspondences under a given pose."""
    if p.ref is None or len(p.ref.kp_idx) == 0:
        return float("nan")
    x = p.ref.feat.kpts[p.ref.kp_idx].astype(np.float64) + 0.5
    return float(np.median(reproj(R, tvec, f, k1, w, h, p.ref.xyz, x)))


def smooth_poses(t: np.ndarray, R: np.ndarray, tvec: np.ndarray, ok: np.ndarray, n_inl: np.ndarray, sigma_s: float):
    """Light robust local-linear smoothing of centre + rotation of the localised frames (cameras/smooth.smooth_track, the
    demo's stage_smooth setting). Returns (R, tvec, outlier mask)."""
    from .smooth import smooth_track
    wts = np.where(ok, np.clip(n_inl / 100.0, 0.2, 1.0), 0.0)
    sm = smooth_track(t, R, tvec, np.ones(len(t)), ok, wts, sigma_s=sigma_s)
    return sm["R"], sm["tvec"], np.asarray(sm.get("outlier", np.zeros(len(t), bool)), bool)


def interpolate(n_all: np.ndarray, n_loc: np.ndarray, R: np.ndarray, C: np.ndarray, ok: np.ndarray, max_gap: int):
    """Poses at every video frame index of n_all from the localised ones (slerp + lerp between the nearest ok neighbours).
    Returns (R (N,3,3), C (N,3), ok (N,) = localised ok, or between two ok frames not further apart than max_gap, interp (N,))."""
    from scipy.spatial.transform import Rotation, Slerp
    good = np.flatnonzero(ok)
    out_R = np.tile(np.eye(3), (len(n_all), 1, 1))
    out_C = np.zeros((len(n_all), 3))
    out_ok = np.zeros(len(n_all), bool)
    interp = np.zeros(len(n_all), bool)
    if len(good) == 0:
        return out_R, out_C, out_ok, interp
    ng = n_loc[good]
    exact = np.isin(n_all, ng)
    if len(good) >= 2:
        sl = Slerp(ng.astype(float), Rotation.from_matrix(R[good]))
        inside = (n_all >= ng[0]) & (n_all <= ng[-1])
        out_R[inside] = sl(n_all[inside].astype(float)).as_matrix()
        for d in range(3):
            out_C[inside, d] = np.interp(n_all[inside], ng, C[good, d])
        j = np.clip(np.searchsorted(ng, n_all), 1, len(ng) - 1)
        out_ok = inside & ((ng[j] - ng[j - 1]) <= max_gap)
    pos = np.searchsorted(n_loc, n_all[exact])
    out_R[exact], out_C[exact], out_ok[exact] = R[pos], C[pos], True
    interp = out_ok & ~exact
    return out_R, out_C, out_ok, interp


def _release(p: Pose | None) -> None:
    """A frame that is neither the previous frame nor the anchor any more keeps its keypoints (the final solve) but not its
    descriptors / GPU tensors."""
    if p is not None and p.ref is not None:
        p.ref.feat._g = None
        p.ref.feat.desc = None


def track_window(cam: dict, mp: MapRefs, src: Path, poses_path: Path | None, calib: dict, window: tuple[float, float],
                 s: NewCamSettings, log=print, full_window=WINDOW, eng: Engine | None = None) -> dict:
    """Localise one window of one camera with the calibrated intrinsics fixed. Returns the per-frame arrays of the window's
    canonical frames (video frame index n, pts, reference time t, ok, R, tvec (cam_from_world), C, n_inliers, reproj_px,
    source map | track | interp, localised) in the SfM frame of `mp`, at WORK resolution (calib work_size, f_work, k1)."""
    t_start = time.time()
    cid = cam["id"]
    path = Path(src) / cam["file"]
    pts = video_pts(path)
    tab = frame_table(pts, cam.get("slope"), cam.get("offset"), cam["avg_fps"], full_window)
    ref_t = np.where(np.isfinite(tab["t"]), tab["t"], tab["pts"])      # a camera without a clock (Tony): its own video time
    sel = np.flatnonzero((ref_t >= window[0] - 1e-9) & (ref_t <= window[1] + 1e-9))
    n_all = tab["n"][sel]
    if len(n_all) == 0:
        return {"empty": True, "cam": cid}
    w, h = (int(x) for x in calib["work_size"])
    sc = 1.0 / calib["scale_to_native"]
    f0, k1 = float(calib["f_work"]), float(calib["k1"])
    stride = max(1, s.stride)
    loc_set = {int(n) for n in n_all[::stride]}
    pose_frames = load_pose_frames(poses_path) if poses_path and Path(poses_path).exists() else None

    def prep(n, img):
        return people_mask(pose_frames[n], sc, (w, h), s.mask_margin, s.min_kp_conf) if pose_frames and n < len(pose_frames) else None

    eng = eng or Engine(s)
    ls = LocSettings(fix_focal=True, ransac_px=s.ransac_px, min_inliers=s.min_inliers)
    ls_map = LocSettings(fix_focal=True, ransac_px=s.ransac_px, min_inliers=s.min_inl_map)
    mrefs = mp.select(calib.get("map_refs", [])) or list(mp.refs.values())[: s.n_refs_keep]
    ema = {r.key: 1.0 for r in mrefs}
    all_refs = list(mp.refs.values())
    got: dict[int, Pose] = {}
    prev: Pose | None = None
    anchor: Pose | None = None
    since_map = 10 ** 6
    fails_since_retrieval = 10 ** 6
    stats = {"map_anchors": 0, "tracked": 0, "escalated": 0, "retrieval": 0, "failed": 0}

    def good(p: Pose) -> bool:
        return p.ok and p.reproj_px <= s.max_reproj_px

    first, last = int(n_all[0]), int(n_all[-1])
    done = 0
    for n, pt, img, mask in FrameStream(path, pts, float(pts[first]), float(pts[last]), (w, h), want=loc_set, every=1, prepare=prep):
        need_map = prev is None or since_map >= s.map_every or prev.n_inliers < 150
        q = eng.extract(img, mask, s.n_kp_anchor if need_map else s.n_kp_track)
        refs = [p_.ref for p_ in (prev, anchor if anchor is not prev else None) if p_ is not None and p_.ok and p_.ref is not None]
        top = sorted(mrefs, key=lambda r: -ema[r.key])
        use_map = need_map or not refs
        p = localize(eng, q, refs + (top[:s.n_refs_anchor] if use_map else []), w, h, f0, k1, ls_map if use_map else ls, key=f"f{n}")
        p.source = "map" if use_map else "track"
        if not good(p):
            stats["escalated"] += 1
            if not use_map:      # tracking against the previous frame failed: best map images of the camera too (more keypoints)
                q = eng.extract(img, mask, s.n_kp_anchor)
                p = localize(eng, q, refs + top[:s.n_refs_anchor + 2], w, h, f0, k1, ls_map, key=f"f{n}")
                p.source = "map"
            if not good(p) and fails_since_retrieval >= s.retrieval_every:     # last resort: every map image (rate-limited)
                fails_since_retrieval = 0
                stats["retrieval"] += 1
                cache: dict = {}
                ranked = sorted(((len(lift(eng, q, r, cache)[0]), r.key) for r in all_refs), reverse=True)[:s.n_refs_anchor + 2]
                p3 = localize(eng, q, [mp.refs[k] for _, k in ranked], w, h, f0, k1, ls_map, key=f"f{n}")
                p3.source = "map"
                if good(p3):
                    p = p3
                    known = {r.key for r in mrefs}
                    for _, k in ranked:
                        if k not in known:
                            mrefs.append(mp.refs[k])
                            ema[k] = 1.0
            else:
                fails_since_retrieval += 1
        if good(p):
            stats["map_anchors" if p.source == "map" else "tracked"] += 1
            old_prev, old_anchor = prev, anchor
            prev = p
            if p.source == "map":
                anchor = p
                since_map = 0
                for r in mrefs:     # which map images carried this anchor: favour them next time
                    cnt = int(np.isin(p.ref.pid, r.pid).sum())
                    ema[r.key] = 0.6 * ema[r.key] + 0.4 * cnt
            else:
                since_map += 1
            for o in (old_prev, old_anchor):
                if o is not None and o is not prev and o is not anchor:
                    _release(o)
        else:
            stats["failed"] += 1
            p.ok = False
            if prev is not None and prev is not anchor:
                _release(prev)
            prev = None
            since_map = 10 ** 6
            _release(p)
        got[n] = p
        done += 1
        if done % 100 == 0:
            ok_n = sum(x.ok for x in got.values())
            log(f"[track {cid}] {done}/{len(loc_set)} frames, {ok_n} ok, {time.time() - t_start:.0f} s "
                f"({(time.time() - t_start) / done * 1000:.0f} ms/frame)")
    return track_arrays(cam, tab, sel, n_all, got, (w, h), f0, k1, s, stats, t_start, window, log)


def track_arrays(cam: dict, tab: dict, sel: np.ndarray, n_all: np.ndarray, got: dict, wh: tuple[int, int], f0: float, k1: float,
                 s: NewCamSettings, stats: dict, t_start: float, window, log=print) -> dict:
    """Final per-frame solve (fixed intrinsics, every inlier correspondence), light smoothing, interpolation of the skipped
    frames: the arrays of one window (see track_window)."""
    w, h = wh
    order = sorted(got)
    loc_n = np.array(order, dtype=np.int64)
    poses = [got[n] for n in order]
    rf = refine_fixed(poses, w, h, f0, k1)
    ok = np.array([p.ok for p in poses], bool)
    R = np.array([p.R for p in poses])
    tv = np.array([p.t for p in poses])
    nin = np.array([p.n_inliers for p in poses])
    rp = np.array([p.reproj_px for p in poses])
    src_loc = np.array([p.source or "" for p in poses])
    pos_in_tab = {int(n): i for i, n in enumerate(tab["n"])}
    ref_t = np.where(np.isfinite(tab["t"]), tab["t"], tab["pts"])
    t_loc = np.array([ref_t[pos_in_tab[int(n)]] for n in loc_n])
    n_sm_out = 0
    if ok.sum() >= 3:
        Rs, tvs, outl = smooth_poses(t_loc, R, tv, ok, nin, s.smooth_sigma_s)
        n_sm_out = int((outl & ok).sum())
        for i, p in enumerate(poses):       # keep the raw pose where the smoothed one is clearly worse (the demo's rule)
            if not ok[i] or p.ref is None:
                continue
            e_s = median_reproj(p, Rs[i], tvs[i], f0, k1, w, h)
            e_r = median_reproj(p, R[i], tv[i], f0, k1, w, h)
            if np.isfinite(e_s) and e_s <= max(1.5 * e_r, e_r + 0.5):
                R[i], tv[i], rp[i] = Rs[i], tvs[i], e_s
    ok = ok & (np.nan_to_num(rp, nan=1e9) <= s.max_reproj_px)
    C_loc = -np.einsum("fji,fj->fi", R, tv)
    max_gap = int(round(2.0 * float(cam["avg_fps"])))          # never bridge more than 2 s (raw video frames)
    R_all, C_all, ok_all, interp = interpolate(n_all, loc_n, R, C_loc, ok, max_gap)
    is_loc = np.isin(n_all, loc_n)
    pos = np.clip(np.searchsorted(loc_n, n_all), 0, len(loc_n) - 1)
    localised = is_loc & ok[pos] & (loc_n[pos] == n_all)
    out = {"cam": cam["id"], "n": n_all, "pts": tab["pts"][sel], "t": tab["t"][sel], "ok": ok_all, "R": R_all,
           "tvec": -np.einsum("fij,fj->fi", R_all, C_all), "C": C_all,
           "n_inliers": np.where(is_loc, nin[pos], 0), "reproj_px": np.where(is_loc, rp[pos], np.nan),
           "source": np.where(is_loc, src_loc[pos], np.where(interp, "interp", "")), "interp": interp, "localised": localised,
           "f_work": f0, "k1": k1, "work_size": np.array([w, h]),
           "stats": {**stats, "refine": rf, "smoother_outliers": n_sm_out, "seconds": round(time.time() - t_start, 1)}}
    log(f"[track {cam['id']}] window {list(np.round(window, 2))}: {int(localised.sum())}/{len(loc_n)} localised, "
        f"{int(ok_all.sum())}/{len(n_all)} frames ok (interpolated {int(interp.sum())}), {out['stats']}")
    return out


def save_track(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in d.items() if k not in ("stats", "cam")}
    np.savez(path, cam=np.str_(d["cam"]), stats=np.str_(json.dumps(d["stats"], default=float)), **keep)


def run_camera(a, cam: dict, mp: MapRefs, src: Path, poses: Path, s: NewCamSettings) -> int:
    """Pod driver of one camera: calibrate (unless calib/<cam>.json exists), then track its 60 s chunks with `procs` processes at
    a time (they share the GPU: one process leaves a 4090 at ~20 % utilisation). Chunks whose output exists are skipped, so a
    restarted job continues. Returns 0 when every chunk landed."""
    calib_p = Path(a.calib_dir) / f"{a.cam}.json"
    if not calib_p.exists():
        calibrate(cam, mp, src, poses, Path(a.calib_dir), s, max_anchors=a.max_anchors)
    calib = json.loads(calib_p.read_text())
    if "f_work" not in calib:
        print(f"[newcam] {a.cam}: calibration failed (no focal): {calib.get('intrinsics')}", flush=True)
        return 3
    del mp                                              # the chunk processes load their own copy
    windows = chunk_windows()
    if cam.get("offset") is None:
        ks = [0]
    else:
        t0 = cam["offset"]
        t1 = cam["slope"] * cam["duration"] + cam["offset"]
        ks = [w["k"] for w in windows if w["core"][1] > t0 + 1.0 and w["core"][0] < t1 - 1.0]
    if a.chunks:
        ks = [k for k in ks if k in {int(x) for x in a.chunks.split(",")}]
    out_dir = Path(a.out_dir)
    todo = [k for k in ks if not (out_dir / f"{a.cam}_{k}.npz").exists()]
    print(f"[newcam] {a.cam}: chunks {ks}, to do {todo}, {a.procs} at a time", flush=True)
    base = [sys.executable, "-m", "dancecap.cameras.newcam", "track", a.cam]
    common = ["--model", a.model, "--world", a.world, "--stride", str(a.stride), "--cams-json", a.cams_json,
              "--calib-dir", a.calib_dir, "--out-dir", a.out_dir]
    running: list[tuple[int, subprocess.Popen]] = []
    failed = []
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue_ = list(todo)
    while queue_ or running:
        while queue_ and len(running) < max(1, a.procs):
            k = queue_.pop(0)
            lf = open(log_dir / f"{a.cam}_{k}.log", "ab")
            running.append((k, subprocess.Popen(base[:5] + [str(k)] + common, stdout=lf, stderr=subprocess.STDOUT)))
            print(f"[newcam] {a.cam}: chunk {k} started", flush=True)
        time.sleep(2)
        for item in list(running):
            k, pr = item
            if pr.poll() is not None:
                running.remove(item)
                ok = pr.returncode == 0 and (out_dir / f"{a.cam}_{k}.npz").exists()
                print(f"[newcam] {a.cam}: chunk {k} {'done' if ok else 'FAILED rc=' + str(pr.returncode)}", flush=True)
                if not ok:
                    failed.append(k)
                else:   # echo the chunk's summary line into the job log
                    try:
                        lines = (log_dir / f"{a.cam}_{k}.log").read_text(errors="replace").splitlines()
                        print("   " + [x for x in lines if x.startswith(f"[track {a.cam}] window")][-1], flush=True)
                    except (OSError, IndexError):
                        pass
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------ CLI (pod steps)


def _cam_row(cams_json: Path, cid: str) -> dict:
    return json.loads(Path(cams_json).read_text())["cams"][cid]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.newcam", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calib", "track", "run"):
        p = sub.add_parser(name)
        p.add_argument("cam")
        if name == "track":
            p.add_argument("k", type=int)
        if name == "run":
            p.add_argument("--chunks", default=None, help="comma list of chunk indices (default: all the camera records in)")
            p.add_argument("--procs", type=int, default=3, help="chunks tracked at the same time (processes sharing the GPU)")
        p.add_argument("--model", default="main")
        p.add_argument("--world", default=os.environ.get("HM_WORLD_DIR"))
        p.add_argument("--src", default=None, help="folder of the videos (default $HM_ROOT/src/Larissa-Kadu-recap)")
        p.add_argument("--cams-json", default=str(WORK / "cameras" / "recap_cams.json"))
        p.add_argument("--calib-dir", default=str(WORK / "cameras" / "calib"))
        p.add_argument("--out-dir", default=str(WORK / "cameras" / "chunks"))
        p.add_argument("--stride", type=int, default=2)
        p.add_argument("--max-anchors", type=int, default=None)
        p.add_argument("--window", default=None, help="override the chunk window 'a:b' (master s)")
    a = ap.parse_args(argv)
    root = Path(os.environ.get("HM_ROOT", "/workspace/hm"))
    src = Path(a.src) if a.src else root / "src" / "Larissa-Kadu-recap"
    poses = src / "poses" / f"{_cam_row(a.cams_json, a.cam)['file']}.json.gz"
    if not poses.exists():
        poses = src / f"{_cam_row(a.cams_json, a.cam)['file']}.json"
    cam = _cam_row(a.cams_json, a.cam)
    s = NewCamSettings(stride=a.stride)
    mp = MapRefs(Path(a.world), a.model)
    print(f"[newcam] map {a.model}: {len(mp)} images, {len(mp.points)} points", flush=True)
    if a.cmd == "calib":
        calibrate(cam, mp, src, poses, Path(a.calib_dir), s, max_anchors=a.max_anchors)
    elif a.cmd == "run":
        return run_camera(a, cam, mp, src, poses, s)
    else:
        calib = json.loads((Path(a.calib_dir) / f"{a.cam}.json").read_text())
        ch = chunk_windows()[a.k]
        if cam.get("offset") is None:       # a camera without a clock (Tony): one window = its whole video (own pts)
            ch = {"k": 0, "core": [0.0, 1e6], "window": [0.0, 1e6]}
        win = tuple(float(x) for x in a.window.split(":")) if a.window else tuple(ch["window"])
        d = track_window(cam, mp, src, poses, calib, win, s)
        if d.get("empty"):
            print(f"[newcam] {a.cam} chunk {a.k}: the camera records nothing in {win}", flush=True)
            save_track(Path(a.out_dir) / f"{a.cam}_{a.k}.npz", {"cam": a.cam, "stats": {"empty": True}, "n": np.zeros(0, int)})
            return 0
        d["chunk"] = a.k
        d["core"] = np.array(ch["core"])
        d["window"] = np.array(win)
        save_track(Path(a.out_dir) / f"{a.cam}_{a.k}.npz", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
