"""Virtual cameras of the mono4d route, in the world frame of CONTRACTS.md (Y-up, metres, floor y = 0, OpenCV cameras
x_cam = R @ X_world + t, +X right, +Y down, +Z forward).

    ring(...)              N static cameras on a circle around the couple, looking at chest height (MAMMA's input)
    auto_radius(...)       the smallest ring radius that keeps the couple (a 0.9 m x 2.2 m cylinder) in every ring
                           camera's frame over its whole floor path
    from_track(...)        a real phone's (moving, zooming) pose resampled to the grid: a generated view rendered where a
                           real one exists can be compared with it (holdout.py)
    umeyama(...)           Sim3 dst = s R src + t from point pairs (camera centres, matched room points)
    world_from_scene(...)  a provider's scene frame -> world, from the provider's poses of the input frames and our
                           own track of the same phone (rotation from orientations, scale from the centres); refuses a
                           camera that barely moved (use room points then)
    to_scene(...)          world cameras -> the provider's scene frame (to ask a provider for renders in its frame)
    trajectory(...)        JSON trajectories for a request: c2w in OpenCV AND OpenGL axes + intrinsics per frame

A Sim3 is the dict layout of cameras/world_from_sfm.json: {"s": float, "R": 3x3, "t": 3}, X_world = s R X_scene + t.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import mamma_dyn as md

OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])  # camera axes: +Y down/+Z forward -> +Y up/-Z forward
COUPLE_HEIGHT_M = 2.2   # what must fit vertically: the taller dancer with raised arms and a margin
COUPLE_WIDTH_M = 1.8    # diameter of the floor disc the couple occupies around its centre (open position, turns)


@dataclass
class ViewSpec:
    """One camera over the grid: K (T,3,3), R (T,3,3) world -> camera, t (T,3) metres, image size, ok (T,) bool."""
    name: str
    kind: str  # "ring" | "holdout" | "source"
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    ok: np.ndarray | None = None

    def __post_init__(self):
        if self.ok is None:
            self.ok = np.ones(len(self.K), bool)

    @property
    def T(self) -> int:
        return len(self.K)

    @property
    def centres(self) -> np.ndarray:
        return -np.einsum("fji,fj->fi", self.R, self.t)

    def c2w_opencv(self) -> np.ndarray:
        E = md.extrinsics(self.R, self.t)
        return np.linalg.inv(E)

    def c2w_opengl(self) -> np.ndarray:
        return self.c2w_opencv() @ OPENCV_TO_OPENGL

    def is_static(self, tol: float = 1e-9) -> bool:
        return bool(np.ptp(self.K, 0).max() <= tol and np.ptp(self.R, 0).max() <= tol and np.ptp(self.t, 0).max() <= tol)


def focal_from_hfov(width: int, hfov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def _intrinsics(T: int, f: float, width: int, height: int) -> np.ndarray:
    K = np.zeros((T, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = width / 2.0, height / 2.0, 1.0
    return K


def couple_points(centre_xz, n: int = 16) -> np.ndarray:
    """The couple as a cylinder (radius COUPLE_WIDTH_M / 2, floor to COUPLE_HEIGHT_M) at a floor position -> (2n, 3)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = COUPLE_WIDTH_M / 2
    ring_ = np.stack([centre_xz[0] + r * np.sin(a), np.zeros(n), centre_xz[1] + r * np.cos(a)], 1)
    return np.concatenate([ring_, ring_ + [0.0, COUPLE_HEIGHT_M, 0.0]])


def auto_radius(centres_xz: np.ndarray, hfov_deg: float, size: tuple[int, int], heights: tuple[float, ...] = (1.5,),
                target_height: float = 0.9, edge: float = 0.04, min_radius: float = 2.5, n_az: int = 16) -> float:
    """The smallest ring radius at which every camera of the ring (any azimuth, every height) keeps the couple in frame
    wherever it is on its floor path: the couple_points() cylinder at each sampled floor position must project inside
    the image with `edge` (fraction of the side) to spare. Bisection between min_radius and 25 m.

    centres_xz (T, 2): the couple's floor centre over the window (world x, z); the ring is centred on their mean.
    """
    W, H = size
    c = np.asarray(centres_xz, float).reshape(-1, 2)
    centre = c.mean(0)
    if len(c) > 64:  # the extreme positions matter: keep a spread-out subset
        far = np.argsort(-np.linalg.norm(c - centre, axis=1))[:32]
        c = np.concatenate([c[far], c[:: max(1, len(c) // 32)]])
    pts = np.concatenate([couple_points(p) for p in c])

    def fits(radius: float) -> bool:
        for cam in ring(1, n=n_az, radius=radius, heights=heights, centre=tuple(centre),
                        target_height=target_height, hfov_deg=hfov_deg, size=size):
            px, z = md.project(pts, cam.K[0], cam.R[0], cam.t[0])
            if (z <= 0.1).any():
                return False
            u, v = px[:, 0] / W, px[:, 1] / H
            if u.min() < edge or u.max() > 1 - edge or v.min() < edge or v.max() > 1 - edge:
                return False
        return True

    lo, hi = min_radius, 25.0
    if fits(lo):
        return lo
    if not fits(hi):
        raise ValueError("the couple's path does not fit the ring at 25 m: widen hfov or split the window")
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    return hi


def ring(T: int, n: int = 8, radius: float = 4.0, heights: tuple[float, ...] = (1.5,), centre=(0.0, 0.0),
         target_height: float = 0.9, hfov_deg: float = 55.0, size: tuple[int, int] = (1080, 1920),
         az0_deg: float = 0.0, prefix: str = "v") -> list[ViewSpec]:
    """n static cameras per height, evenly spaced in azimuth, all looking at (centre_x, target_height, centre_z).

    Azimuth 0 is +z, increasing towards +x (the same as dancecap.mamma_dyn.synthetic_rig). Names v00, v01, ... in
    azimuth order, height rings one after the other. Portrait 1080x1920 by default, like the phones.
    """
    W, H = size
    f = focal_from_hfov(W, hfov_deg)
    target = np.array([centre[0], target_height, centre[1]], float)
    out = []
    for h in heights:
        for i in range(n):
            az = math.radians(az0_deg) + 2 * math.pi * i / n
            C = np.array([centre[0] + radius * math.sin(az), h, centre[1] + radius * math.cos(az)])
            R = md._look_at(C[None], target[None], np.zeros(1))[0]
            out.append(ViewSpec(name=f"{prefix}{len(out):02d}", kind="ring", K=_intrinsics(T, f, W, H),
                                R=np.repeat(R[None], T, 0), t=np.repeat((-R @ C)[None], T, 0), width=W, height=H))
    return out


def from_track(track: md.CameraTrack, grid: np.ndarray, kind: str = "holdout") -> ViewSpec:
    """A real phone's cameras resampled to the grid (slerp R, lerp centre, log-focal), as a ViewSpec."""
    g = md.resample_cameras(track.t, track.K, track.R, track.tvec, track.ok, grid, track.frame_dt)
    return ViewSpec(name=track.name, kind=kind, K=g["K"], R=g["R"], t=g["t"], width=track.width,
                    height=track.height, ok=g["ok"])


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> dict:
    """Least-squares similarity dst ~= s R src + t (Umeyama 1991) -> {"s", "R", "t", "rms"}."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("umeyama needs two (N >= 3, 3) arrays of the same shape")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = float((xs ** 2).sum() / len(src))
    if var_s < 1e-12:
        raise ValueError("umeyama: the source points coincide")
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    rms = float(np.sqrt(((dst - (s * src @ R.T + t)) ** 2).sum(1).mean()))
    return {"s": s, "R": R.tolist(), "t": t.tolist(), "rms": rms}


def apply_sim3(sim3: dict, X: np.ndarray) -> np.ndarray:
    return float(sim3["s"]) * np.asarray(X, float) @ np.asarray(sim3["R"], float).T + np.asarray(sim3["t"], float)


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Angle of Ra Rb^T per frame (degrees)."""
    M = np.einsum("...ij,...kj->...ik", Ra, Rb)
    c = np.clip((np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def world_from_scene(scene_c2w_opencv: np.ndarray, source: ViewSpec, min_spread_m: float = 0.15) -> dict:
    """Sim3 world <- scene from the provider's poses of the input frames (T,4,4 c2w, OpenCV axes, scene frame) and our
    track of the same phone on the same frames (source).

    Rotation comes from the camera ORIENTATIONS (chordal mean of R_world_w2c^T R_scene_w2c): camera centres alone leave
    it undetermined when the phone moves along a line. Scale and translation come from the centres given that
    rotation, so a camera that barely moved is refused (align matched room points with umeyama() instead).
    -> {"s", "R", "t", "rms" (m, centres), "rot_err_deg": {median, p95} (orientations after alignment), "n", "spread_m"}
    """
    P = np.asarray(scene_c2w_opencv, float)
    if P.shape != (source.T, 4, 4):
        raise ValueError(f"scene poses {P.shape} do not match the source track ({source.T} frames)")
    ok = source.ok & np.isfinite(P).all((1, 2))
    Cs, Cw = P[ok, :3, 3], source.centres[ok]
    spread = float(np.linalg.norm(Cw - Cw.mean(0), axis=1).max()) if len(Cw) else 0.0
    if len(Cw) < 3 or spread < min_spread_m:
        raise ValueError(f"the input camera moved only {spread:.2f} m (< {min_spread_m} m): camera centres cannot fix "
                         "the scene frame's scale; align matched room points with rig.umeyama instead")
    R_scene_w2c = np.transpose(P[ok, :3, :3], (0, 2, 1))
    # R_world_w2c = R_scene_w2c @ Rws^T  =>  Rws = R_world_w2c^T @ R_scene_w2c (per frame); project the mean onto SO(3)
    U, _, Vt = np.linalg.svd(np.einsum("fji,fjk->ik", source.R[ok], R_scene_w2c))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    Rws = U @ D @ Vt
    xs, xd = Cs - Cs.mean(0), Cw - Cw.mean(0)
    s = float(((xs @ Rws.T) * xd).sum() / max((xs ** 2).sum(), 1e-12))
    t = Cw.mean(0) - s * Rws @ Cs.mean(0)
    rms = float(np.sqrt(((Cw - (s * Cs @ Rws.T + t)) ** 2).sum(1).mean()))
    err = rotation_angle_deg(R_scene_w2c @ Rws.T, source.R[ok])
    return {"s": s, "R": Rws.tolist(), "t": t.tolist(), "rms": rms,
            "rot_err_deg": {"median": float(np.median(err)), "p95": float(np.percentile(err, 95))},
            "n": int(ok.sum()), "spread_m": spread}


def to_scene(spec: ViewSpec, world_from_scene_sim3: dict) -> ViewSpec:
    """World cameras -> the same cameras in the provider's scene frame (scene units)."""
    R, t = md.sfm_cameras(spec.R, spec.t, world_from_scene_sim3)
    return ViewSpec(name=spec.name, kind=spec.kind, K=spec.K.copy(), R=R, t=t, width=spec.width, height=spec.height,
                    ok=spec.ok.copy())


def qvec_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) -> unit quaternion (w, x, y, z) with w >= 0 (COLMAP's qvec)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _r(a: np.ndarray, nd: int = 6) -> list:
    return np.round(np.asarray(a, float), nd).tolist()


def trajectory(specs: list[ViewSpec], t_grid: np.ndarray, frame: str = "world", sim3: dict | None = None) -> dict:
    """Request-side description of every camera: per frame t, fx, fy, cx, cy, c2w_opencv, c2w_opengl (4x4 rows).

    A static camera is written once ("static": true) and holds for every time in time_s. frame names the frame the
    poses are in ("world" = CONTRACTS.md world, "scene" = the provider's); sim3 (world <- scene) is recorded with it.
    """
    out = {"format": "dancecap.mono4d.trajectory/1", "frame": frame, "units": "metres" if frame == "world" else "scene",
           "axes": {"world": "right-handed, Y-up, floor y=0", "c2w_opencv": "+X right, +Y down, +Z forward",
                    "c2w_opengl": "+X right, +Y up, -Z forward"},
           "time_s": _r(t_grid), "world_from_scene": sim3, "cameras": {}}
    for s in specs:
        static = s.is_static()
        idx = [0] if static else range(s.T)
        cv, gl = s.c2w_opencv(), s.c2w_opengl()
        out["cameras"][s.name] = {
            "kind": s.kind, "width": s.width, "height": s.height, "static": static,
            "frames": [{"t": float(round(t_grid[i], 6)), "ok": bool(s.ok[i]), "fx": float(s.K[i, 0, 0]),
                        "fy": float(s.K[i, 1, 1]), "cx": float(s.K[i, 0, 2]), "cy": float(s.K[i, 1, 2]),
                        "c2w_opencv": _r(cv[i]), "c2w_opengl": _r(gl[i])} for i in idx]}
    return out


def specs_from_trajectory(doc: dict) -> list[ViewSpec]:
    """Inverse of trajectory() (a static camera is repeated over time_s)."""
    T = len(doc["time_s"])
    out = []
    for name, c in doc["cameras"].items():
        fr = c["frames"] * T if c.get("static") else c["frames"]
        K = np.array([[[f["fx"], 0, f["cx"]], [0, f["fy"], f["cy"]], [0, 0, 1]] for f in fr], float)
        E = np.linalg.inv(np.array([f["c2w_opencv"] for f in fr], float))
        out.append(ViewSpec(name=name, kind=c.get("kind", "ring"), K=K, R=E[:, :3, :3], t=E[:, :3, 3],
                            width=int(c["width"]), height=int(c["height"]),
                            ok=np.array([f.get("ok", True) for f in fr], bool)))
    return out
