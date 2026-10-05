"""Atlas scene frame <-> stage frame.

p_stage = scale * R @ (p_scene - origin)

* R rotates the scene floor normal to +Z (stage up)
* origin is the dancers' floor position in the scene
* scale converts Atlas units to metres (Atlas does not document metric scale;
  verify with `atlas-bridge check-scale` after the first MAMMA run)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ply_xyz(path: Path, max_points: int = 400_000) -> np.ndarray:
    """Positions from a binary little-endian or ASCII PLY (3DGS exports are binary)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        body = f.read()
    fmt = next(l.split()[1] for l in header if l.startswith("format"))
    props, count, in_vertex = [], 0, False
    for l in header:
        parts = l.split()
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            props.append((parts[2], parts[1]))
    type_map = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
                "int": "<i4", "int32": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}
    if fmt == "ascii":
        data = np.loadtxt(body.decode().splitlines()[:count], ndmin=2)
        names = [p[0] for p in props]
        xyz = data[:, [names.index("x"), names.index("y"), names.index("z")]]
    else:
        dtype = np.dtype([(n, type_map[t]) for n, t in props])
        arr = np.frombuffer(body, dtype=dtype, count=count)
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    if len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(0).choice(len(xyz), max_points, replace=False)]
    return xyz


def ransac_plane(points: np.ndarray, threshold: float, iterations: int = 2000,
                 up_hint: np.ndarray | None = None, max_tilt_deg: float = 35.0,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    """Largest plane (normal n, offset d with n.p + d = 0, inlier mask).

    With up_hint, only planes within max_tilt_deg of it are considered and the
    normal is oriented towards it (a floor, not a wall).
    """
    rng = rng or np.random.default_rng(0)
    best = (None, 0.0, np.zeros(len(points), bool))
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if up_hint is not None:
            if abs(n @ up_hint) < cos_tilt:
                continue
            if n @ up_hint < 0:
                n = -n
        d = -n @ a
        inliers = np.abs(points @ n + d) < threshold
        if inliers.sum() > best[2].sum():
            best = (n, d, inliers)
    if best[0] is None:
        raise RuntimeError("no floor plane found")
    # least-squares refine on inliers
    P = points[best[2]]
    centroid = P.mean(0)
    n = np.linalg.svd(P - centroid)[2][-1]
    if up_hint is not None and n @ up_hint < 0:
        n = -n
    elif up_hint is None and n @ best[0] < 0:
        n = -n
    return n, float(-n @ centroid), best[2]


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def optical_axes_floor_point(c2w_opencv: list[np.ndarray], n: np.ndarray, d: float) -> np.ndarray | None:
    """Median floor intersection of the real cameras' optical axes (where the phones were aimed)."""
    hits = []
    for T in c2w_opencv:
        o, fwd = T[:3, 3], T[:3, 2]
        denom = n @ fwd
        if abs(denom) < 1e-6:
            continue
        t = -(n @ o + d) / denom
        if t > 0:
            hits.append(o + t * fwd)
    return np.median(np.array(hits), axis=0) if hits else None


def scene_to_stage(floor_n: np.ndarray, floor_d: float, dancers_scene: np.ndarray, scale: float,
                   forward_scene: np.ndarray | None = None) -> np.ndarray:
    """4x4 scene->stage. dancers_scene is projected onto the floor to become the origin.

    forward_scene (optional) is rotated to stage +Y, e.g. the reference phone's
    view direction, so "audience front" is consistent across captures.
    """
    origin = dancers_scene - (floor_n @ dancers_scene + floor_d) * floor_n
    R = rotation_between(floor_n, np.array([0.0, 0.0, 1.0]))
    if forward_scene is not None:
        f = R @ forward_scene
        f[2] = 0
        if np.linalg.norm(f) > 1e-6:
            f /= np.linalg.norm(f)
            yaw = np.arctan2(f[0], f[1])  # rotate f onto +Y
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = -scale * R @ origin
    return T


def auto_scene_to_stage(env_ply: Path, source_c2w: dict[str, np.ndarray], scale: float,
                        reference: str | None, floor_threshold: float = 0.03,
                        dancers_scene: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Best-effort alignment from Atlas outputs. Returns (4x4, notes)."""
    notes = []
    pts = read_ply_xyz(env_ply)
    up_hint = None
    if source_c2w:
        # phones are held roughly upright: camera "up" (-Y opencv) averages to world up
        up = -np.mean([T[:3, 1] for T in source_c2w.values()], axis=0)
        up_hint = up / np.linalg.norm(up)
    else:
        notes.append("no source camera poses from Atlas: floor = largest plane (check it isn't a wall)")
    n, d, inliers = ransac_plane(pts, floor_threshold / max(scale, 1e-6), up_hint=up_hint)
    notes.append(f"floor plane: {inliers.mean():.0%} of splats within {floor_threshold * 100:.0f} cm")

    center = optical_axes_floor_point(list(source_c2w.values()), n, d) if source_c2w else None
    if dancers_scene is not None:
        center = np.asarray(dancers_scene, dtype=float)
        notes.append("dancer position from session.toml")
    elif center is None:
        center = pts[inliers].mean(0)
        notes.append("dancer position = floor centroid (no camera rays); set [atlas] dancers_scene to override")
    fwd = None
    if reference and reference in source_c2w:
        fwd = source_c2w[reference][:3, 2]
    return scene_to_stage(n, d, center, scale, fwd), notes
