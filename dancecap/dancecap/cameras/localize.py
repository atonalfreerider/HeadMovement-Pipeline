"""Per-frame localisation of every window frame against the static SfM map.

For each window frame q (in time order, per camera):
  references = registered map images of the same camera nearest in time (+ the previous localised frame,
               whose inlier 2D-3D correspondences are kept), and - if that is not enough - registered images
               of the other cameras near the same reference time and the camera's best keyframes;
  2D-2D LightGlue matches to each reference are lifted to 2D-3D through the reference's 3D point ids;
  pycolmap LO-RANSAC absolute pose with focal estimation (P4Pf) + non-linear refinement of pose and focal
  (k1 is held at the neighbouring map image's value). A fixed-focal P3P run (focal from the nearest map image)
  is tried too and the result with more inliers wins (P4Pf is unstable for short focal lever arms).
Frames that still fail are retried with their localised temporal neighbours as references (frame-to-frame
tracking), and gaps that remain are filled by constant-velocity interpolation with ok=False.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import progress as _progress  # dancecap.progress/1 (stdlib; no-op unless DANCECAP_PROGRESS)
from .features import Features, ImageRec


@dataclass
class LocSettings:
    n_same_cam: int = 4  # nearest registered map images of the same camera
    n_cross: int = 4  # registered images of other cameras (nearest in time), used when same-cam is weak
    n_keyframes: int = 2  # same-camera keyframes with the most 3D points (fallback)
    min_inliers: int = 30  # accept a pose
    good_inliers: int = 120  # skip the cross-camera fallback when same-camera references give this many
    ransac_px: float = 6.0
    max_focal_change: float = 0.35  # |log f / f_neighbour| above this -> reject the P4Pf solution
    track_passes: int = 3
    fix_focal: bool = False  # True: P3P with the reference focal, pose-only refinement (single-camera sub-models)
    kf_by_time: bool = False  # True: references = the nearest map images IN TIME of either kind (keyframes too),
    #   and the cross-camera fallback takes each other camera's nearest image of either kind. For a map reused
    #   from another take (cameras/reuse.py) whose window images cover only part of this take's window.


@dataclass
class Ref:
    """2D keypoint index -> 3D point (map image or a localised window frame)."""
    rec: ImageRec
    kp_idx: np.ndarray  # (M,) keypoint indices of rec that have a 3D point
    xyz: np.ndarray  # (M, 3)
    pid: np.ndarray  # (M,) 3D point ids (-1 for none)


@dataclass
class FramePose:
    ok: bool = False
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    f: float = float("nan")
    k1: float = 0.0
    n_inliers: int = 0
    reproj_px: float = float("nan")
    source: str = ""  # 'map' | 'loc' | 'track' | 'interp' | ''
    corr: Ref | None = None  # inlier correspondences (for tracking + post-smoothing reprojection)
    q_xy: np.ndarray | None = None  # inlier 2D points (native px, OpenCV pixel-centre convention)


def map_refs(rec_model, images: list[ImageRec], F: Features) -> dict[str, Ref]:
    """Ref per registered map image (key -> Ref)."""
    by_name = {im.name: im for im in rec_model.images.values() if im.has_pose}
    out = {}
    for r in images:
        im = by_name.get(r.name)
        if im is None:
            continue
        pts = im.points2D
        ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in pts], np.int64)
        idx = np.nonzero(ids >= 0)[0]
        xyz = np.array([rec_model.points3D[int(i)].xyz for i in ids[idx]]).reshape(-1, 3)
        out[r.key] = Ref(r, idx, xyz, ids[idx])
    return out


def image_camera(rec_model, name: str):
    for im in rec_model.images.values():
        if im.name == name and im.has_pose:
            return im
    return None


def _lift(F: Features, q: ImageRec, ref: Ref, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(query keypoint idx, xyz, pid) from LightGlue matches q <-> ref."""
    k = (q.key, ref.rec.key)
    if k not in cache:
        cache[k] = F.match(q, ref.rec)
    m = cache[k]
    if len(m) == 0 or len(ref.kp_idx) == 0:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros(0, np.int64)
    lut = -np.ones(int(max(m[:, 1].max(), ref.kp_idx.max())) + 1, np.int64)
    lut[ref.kp_idx] = np.arange(len(ref.kp_idx))
    j = lut[np.minimum(m[:, 1].astype(np.int64), len(lut) - 1)]
    ok = j >= 0
    return m[ok, 0].astype(int), ref.xyz[j[ok]], ref.pid[j[ok]]


def _dedupe(qi: np.ndarray, xyz: np.ndarray, pid: np.ndarray):
    """One 3D point per query keypoint (most frequent / first), one keypoint per 3D point id."""
    if len(qi) == 0:
        return qi, xyz, pid
    key = qi.astype(np.int64) * 10_000_000 + np.where(pid >= 0, pid, 9_999_999)
    _, first = np.unique(key, return_index=True)
    qi, xyz, pid = qi[first], xyz[first], pid[first]
    _, first = np.unique(qi, return_index=True)  # several 3D candidates for one keypoint: keep one
    return qi[first], xyz[first], pid[first]


def solve_pose(xy: np.ndarray, xyz: np.ndarray, w: int, h: int, f0: float, k1: float, s: LocSettings):
    """Best of P4Pf (focal estimated) and P3P with focal f0, both refined (pose + focal). xy in COLMAP px.

    Returns dict(R, t, f, inliers (bool), n) or None.
    """
    import pycolmap
    best = None
    for est_f in ((False,) if s.fix_focal else (True, False)):
        cam = pycolmap.Camera.create_from_model_name(0, "SIMPLE_RADIAL", f0, w, h)
        cam.params = [f0, w / 2, h / 2, k1]
        eo = pycolmap.AbsolutePoseEstimationOptions()
        eo.estimate_focal_length = est_f
        eo.ransac.max_error = s.ransac_px
        eo.ransac.min_inlier_ratio = 0.05
        eo.ransac.max_num_trials = 20000
        ro = pycolmap.AbsolutePoseRefinementOptions()
        ro.refine_focal_length = not s.fix_focal
        ro.refine_extra_params = False
        try:
            r = pycolmap.estimate_and_refine_absolute_pose(xy.astype(np.float64), xyz.astype(np.float64), cam, eo, ro)
        except Exception:  # noqa: BLE001 - degenerate input
            r = None
        if r is None:
            continue
        f = float(cam.params[0])
        if not np.isfinite(f) or abs(np.log(f / f0)) > s.max_focal_change:
            continue
        inl = np.asarray(r["inlier_mask"], bool).reshape(-1)
        n = int(inl.sum())
        if best is None or n > best["n"]:
            T = r["cam_from_world"]
            best = {"R": np.asarray(T.rotation.matrix()), "t": np.asarray(T.translation), "f": f, "k1": k1,
                    "inliers": inl, "n": n}
    return best


def reproj(R, t, f, k1, w, h, xyz, xy_colmap) -> np.ndarray:
    """Per-point reprojection error (px) with the SIMPLE_RADIAL model (COLMAP pixel convention)."""
    Xc = xyz @ R.T + t
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    d = 1 + k1 * (x * x + y * y)
    u, v = f * x * d + w / 2, f * y * d + h / 2
    e = np.hypot(u - xy_colmap[:, 0], v - xy_colmap[:, 1])
    return np.where(z > 0, e, 1e6)


class Localizer:
    def __init__(self, F: Features, rec_model, map_images: list[ImageRec], s: LocSettings | None = None):
        self.F, self.model, self.s = F, rec_model, s or LocSettings()
        self.refs = map_refs(rec_model, map_images, F)
        self.reg = [r for r in map_images if r.key in self.refs]
        self.cams_reg: dict[str, list[ImageRec]] = {}
        for r in self.reg:
            self.cams_reg.setdefault(r.cam, []).append(r)
        self.intr = {}
        for im in rec_model.images.values():
            if im.has_pose:
                p = im.camera.params
                self.intr[im.name] = (float(p[0]), float(p[3]) if len(p) > 3 else 0.0)
        self.match_cache: dict = {}

    def neighbour_intrinsics(self, q: ImageRec, prev: FramePose | None) -> tuple[float, float] | None:
        same = self.cams_reg.get(q.cam, [])
        if not same:
            return None
        near = min(same, key=lambda r: abs(r.t - q.t))
        f, k1 = self.intr[near.name]
        if prev is not None and prev.ok and abs(prev.f) > 0:
            f = prev.f
        return f, k1

    def candidates(self, q: ImageRec, cross: bool) -> list[Ref]:
        same = [r for r in self.cams_reg.get(q.cam, []) if r.key != q.key]
        kinds = ("win", "kf") if self.s.kf_by_time else ("win",)
        win = sorted((r for r in same if r.kind in kinds), key=lambda r: abs(r.t - q.t))[:self.s.n_same_cam]
        out = [self.refs[r.key] for r in win]
        if cross:
            kf = sorted((r for r in same if r.kind == "kf" and r not in win), key=lambda r: -len(self.refs[r.key].kp_idx))
            out += [self.refs[r.key] for r in kf[:self.s.n_keyframes]]
            other = [r for r in self.reg if r.cam != q.cam and r.kind in kinds]
            by_cam: dict[str, list[ImageRec]] = {}
            for r in sorted(other, key=lambda r: abs(r.t - q.t)):
                by_cam.setdefault(r.cam, []).append(r)
            best = sorted((v[0] for v in by_cam.values()), key=lambda r: -len(self.refs[r.key].kp_idx))
            out += [self.refs[r.key] for r in best[:self.s.n_cross]]
        return out

    def localize(self, q: ImageRec, refs: list[Ref], w: int, h: int, f0: float, k1: float) -> FramePose:
        qi, xyz, pid = [], [], []
        for ref in refs:
            a, b, c = _lift(self.F, q, ref, self.match_cache)
            qi.append(a), xyz.append(b), pid.append(c)
        qi, xyz, pid = _dedupe(np.concatenate(qi), np.concatenate(xyz), np.concatenate(pid))
        fp = FramePose()
        if len(qi) < max(8, self.s.min_inliers // 2):
            fp.n_inliers = 0
            return fp
        kp = self.F.get(q)["kpts"][qi] + 0.5  # COLMAP convention
        sol = solve_pose(kp, xyz, w, h, f0, k1, self.s)
        if sol is None:
            return fp
        inl = sol["inliers"]
        e = reproj(sol["R"], sol["t"], sol["f"], sol["k1"], w, h, xyz[inl], kp[inl])
        fp = FramePose(ok=sol["n"] >= self.s.min_inliers, R=sol["R"], t=sol["t"], f=sol["f"], k1=sol["k1"],
                       n_inliers=sol["n"], reproj_px=float(np.median(e)) if len(e) else float("nan"), source="loc",
                       corr=Ref(q, qi[inl], xyz[inl], pid[inl]), q_xy=kp[inl] - 0.5)
        return fp


def localize_camera(L: Localizer, frames: list[ImageRec], w: int, h: int, log: bool = True) -> list[FramePose]:
    """Every window frame of one camera (frames in canonical order)."""
    s = L.s
    poses: list[FramePose] = [FramePose() for _ in frames]
    t0 = time.time()
    for i, q in enumerate(frames):
        if q.key in L.refs:  # registered map frame: take the bundle-adjusted pose
            im = image_camera(L.model, q.name)
            T = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
            f, k1 = L.intr[q.name]
            ref = L.refs[q.key]
            kp = L.F.get(q)["kpts"][ref.kp_idx] + 0.5
            R, tt = np.asarray(T.rotation.matrix()), np.asarray(T.translation)
            e = reproj(R, tt, f, k1, w, h, ref.xyz, kp)
            poses[i] = FramePose(True, R, tt, f, k1, len(ref.kp_idx), float(np.median(e)), "map", ref, kp - 0.5)
            continue
        prev = poses[i - 1] if i > 0 else None
        ni = L.neighbour_intrinsics(q, prev)
        if ni is None:
            continue
        refs = L.candidates(q, cross=False)
        if prev is not None and prev.ok and prev.corr is not None:
            refs.append(prev.corr)
        fp = L.localize(q, refs, w, h, *ni)
        if fp.n_inliers < s.good_inliers:
            fp2 = L.localize(q, L.candidates(q, cross=True) + ([prev.corr] if prev is not None and prev.ok
                                                               and prev.corr is not None else []), w, h, *ni)
            if fp2.n_inliers > fp.n_inliers:
                fp = fp2
        poses[i] = fp
        _progress.tick("loc", i + 1, len(frames), "frames", lane=_progress.default_lane() or q.cam)
        if log and (i + 1) % 60 == 0:
            n_ok = sum(p.ok for p in poses[:i + 1])
            print(f"[cameras] {q.cam}: {i + 1}/{len(frames)} frames, {n_ok} ok ({time.time() - t0:.0f}s)", flush=True)
    # frame-to-frame tracking for the failures, using localised neighbours as references. Passes alternate direction
    # (forward, backward, ...): a forward pass carries a pose only FORWARD through a failed stretch (each newly
    # localised frame feeds the next index), so a stretch that fails at the START of a camera (larissa_kadu_full: cam09
    # zoomed out wide at -3..0 s, its single frames do not match the zoomed-in map images) is recovered by the
    # backward pass, frame by frame from the first localised one.
    stale = 0
    for k_pass in range(max(s.track_passes, 2)):
        changed = False
        order = range(len(frames)) if k_pass % 2 == 0 else range(len(frames) - 1, -1, -1)
        for i in order:
            q = frames[i]
            if poses[i].ok:
                continue
            nb = [poses[j] for j in (i - 1, i + 1, i - 2, i + 2) if 0 <= j < len(frames) and poses[j].ok
                  and poses[j].corr is not None]
            if not nb:
                continue
            ni = L.neighbour_intrinsics(q, nb[0])
            fp = L.localize(q, [p.corr for p in nb] + L.candidates(q, cross=False), w, h, *ni)
            if fp.ok:
                fp.source = "track"
                poses[i] = fp
                changed = True
        stale = 0 if changed else stale + 1
        if stale >= 2:  # nothing moved in either direction
            break
    return poses
