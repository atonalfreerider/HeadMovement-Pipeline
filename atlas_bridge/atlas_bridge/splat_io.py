"""Write standard 3D Gaussian Splatting PLYs (the format Marble/Atlas-style exports and the Unity preview read)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def write_3dgs_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, sigma: np.ndarray | float,
                   opacity: np.ndarray | float = 0.9) -> None:
    """xyz (N,3), rgb (N,3) in [0,1], sigma = isotropic std-dev in scene units, opacity in (0,1)."""
    n = len(xyz)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float32), (n,))
    opacity = np.clip(np.broadcast_to(np.asarray(opacity, dtype=np.float32), (n,)), 1e-4, 1 - 1e-4)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(n, dtype=[(k, "<f4") for k in names])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    dc = (np.asarray(rgb, dtype=np.float32) - 0.5) / SH_C0
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    data["opacity"] = np.log(opacity / (1 - opacity))  # inverse sigmoid
    for k in ("scale_0", "scale_1", "scale_2"):
        data[k] = np.log(sigma)
    data["rot_0"] = 1.0
    header = "ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n" + \
             "".join(f"property float {k}\n" for k in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def studio(rng: np.random.Generator, radius: float = 4.5, wall_height: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """A simple dance studio in stage frame (z-up): wooden floor disc + a wall ring with a mirror band."""
    nf = 60_000
    r = radius * np.sqrt(rng.uniform(0, 1, nf))
    a = rng.uniform(0, 2 * np.pi, nf)
    floor = np.c_[r * np.cos(a), r * np.sin(a), np.zeros(nf)]
    plank = (np.floor((floor[:, 0] + 10) / 0.18) % 2)[:, None]
    floor_rgb = np.array([0.55, 0.36, 0.2]) * (0.85 + 0.15 * plank) + rng.normal(0, 0.02, (nf, 3))
    nw = 40_000
    a = rng.uniform(0, 2 * np.pi, nw)
    z = rng.uniform(0, wall_height, nw)
    wall = np.c_[radius * np.cos(a), radius * np.sin(a), z]
    mirror = (z > 0.4) & (z < 2.2) & (np.sin(a) > 0.3)
    wall_rgb = np.where(mirror[:, None], [0.55, 0.6, 0.65], [0.82, 0.8, 0.76]) + rng.normal(0, 0.02, (nw, 3))
    return np.r_[floor, wall], np.clip(np.r_[floor_rgb, wall_rgb], 0, 1)
