"""Virtual camera rig + coordinate conventions.

Frames used throughout atlas_bridge:

* **stage**  - right-handed, Z-up, metres, floor at z=0, origin at the dancers'
  mean floor position. This is the canonical frame; MAMMA is run in it.
* **opencv camera** - +X right, +Y down, +Z forward (MAMMA's camera frame).
* **opengl camera** - +X right, +Y up, -Z forward (nerfstudio / many NVS tools).
* **unity** - left-handed, Y-up, metres. unity = (x, z, y) of stage. Swapping two
  axes flips handedness, so this is a proper (non-mirroring) conversion.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

STAGE_UP = np.array([0.0, 0.0, 1.0])

# stage (x, y, z) -> unity (x, z, y)
STAGE_TO_UNITY = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
])

# opencv camera axes -> opengl camera axes (flip Y and Z)
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0])


def stage_to_unity(points: np.ndarray) -> np.ndarray:
    """(..., 3) stage points -> unity points (== points @ STAGE_TO_UNITY.T)."""
    return np.asarray(points)[..., [0, 2, 1]]


@dataclass
class Camera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # camera-to-world rotation (columns = opencv camera axes expressed in stage frame)
    R_wc: np.ndarray = field(repr=False)
    # camera centre in stage frame
    C: np.ndarray = field(repr=False)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def T_wc(self) -> np.ndarray:
        """4x4 camera-to-world, opencv camera axes."""
        T = np.eye(4)
        T[:3, :3] = self.R_wc
        T[:3, 3] = self.C
        return T

    @property
    def T_cw(self) -> np.ndarray:
        """4x4 world-to-camera, opencv camera axes (X_cam = R X_world + t)."""
        return np.linalg.inv(self.T_wc)

    @property
    def T_wc_opengl(self) -> np.ndarray:
        T = self.T_wc.copy()
        T[:3, :3] = self.R_wc @ OPENCV_TO_OPENGL
        return T

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / (2 * self.fy)))

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))

    def project(self, X: np.ndarray) -> np.ndarray:
        """(N,3) stage points -> (N,2) pixels. Points behind the camera give NaN."""
        Xc = (X - self.C) @ self.R_wc  # == R_cw @ (X - C)
        z = Xc[:, 2:3]
        uv = (Xc[:, :2] / np.where(z > 1e-9, z, np.nan)) * [self.fx, self.fy] + [self.cx, self.cy]
        return uv

    def quaternion_wxyz(self) -> list[float]:
        x, y, z, w = Rotation.from_matrix(self.R_wc).as_quat()
        return [float(w), float(x), float(y), float(z)]


def look_at(C: np.ndarray, target: np.ndarray, up: np.ndarray = STAGE_UP) -> np.ndarray:
    """camera-to-world rotation (opencv axes) for a camera at C looking at target."""
    forward = target - C
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight up/down
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return fx, fx, width / 2, height / 2


def ring(
    count: int,
    radius: float,
    height: float,
    target: np.ndarray,
    width: int,
    height_px: int,
    hfov_deg: float,
    phase_deg: float = 0.0,
    prefix: str = "v",
    start_index: int = 0,
) -> list[Camera]:
    """`count` cameras evenly spaced in azimuth around `target`, all looking at it."""
    fx, fy, cx, cy = intrinsics_from_hfov(width, height_px, hfov_deg)
    cams = []
    for i in range(count):
        a = math.radians(phase_deg + 360.0 * i / count)
        C = np.array([target[0] + radius * math.cos(a), target[1] + radius * math.sin(a), height])
        cams.append(Camera(
            name=f"{prefix}{start_index + i:02d}",
            width=width, height=height_px, fx=fx, fy=fy, cx=cx, cy=cy,
            R_wc=look_at(C, target), C=C,
        ))
    return cams


def build_rig(rig_cfg: dict, target: np.ndarray | None = None) -> list[Camera]:
    """Build the virtual rig from the [rig] section of session.toml."""
    if target is None:
        target = np.array([0.0, 0.0, rig_cfg.get("look_at_height", 1.0)])
    width = int(rig_cfg.get("width", 1920))
    height_px = int(rig_cfg.get("height", 1080))
    hfov = float(rig_cfg.get("hfov_deg", 65.0))
    cams: list[Camera] = []
    default_rings = [{"count": 8, "radius": 4.0, "height": 1.3},
                     {"count": 4, "radius": 3.5, "height": 2.8, "phase_deg": 45.0}]
    for r in rig_cfg.get("rings", default_rings):
        cams += ring(
            count=int(r["count"]), radius=float(r["radius"]), height=float(r["height"]),
            target=target, width=width, height_px=height_px, hfov_deg=hfov,
            phase_deg=float(r.get("phase_deg", 0.0)), start_index=len(cams),
        )
    return cams


def check_coverage(cams: list[Camera], stage_radius: float = 1.2, body_height: float = 1.9) -> list[str]:
    """Warn about cameras that can't see a cylinder of the given radius/height at the origin."""
    warnings = []
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts = np.array([[stage_radius * np.cos(a), stage_radius * np.sin(a), z]
                    for a in angles for z in (0.0, body_height)])
    for cam in cams:
        uv = cam.project(pts)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
        frac = float(np.nanmean(inside))
        if frac < 0.9:
            warnings.append(f"{cam.name}: only {frac:.0%} of the {stage_radius} m dance area is in frame")
    return warnings


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def to_mamma_yaml(cams: list[Camera]) -> str:
    """MAMMA calibration YAML (capture/loaders/yaml_loader.py). cam2world, wxyz, metres."""
    lines = ["# generated by atlas_bridge - virtual cameras, stage frame (Z-up, metres)",
             "extrinsics_convention: cam2world",
             "quaternion_order: wxyz",
             "cameras:"]
    for c in cams:
        q = c.quaternion_wxyz()
        lines += [
            f"  {c.name}:",
            "    camera_model: pinhole",
            "    distortion_model: radtan",
            f"    intrinsics: [{c.fx:.6f}, {c.fy:.6f}, {c.cx:.6f}, {c.cy:.6f}]",
            "    distortion_coeffs: [0.0, 0.0, 0.0, 0.0]",
            f"    resolution: [{c.width}, {c.height}]",
            f"    translation: [{c.C[0]:.9f}, {c.C[1]:.9f}, {c.C[2]:.9f}]",
            f"    rotation_quaternion: [{q[0]:.12f}, {q[1]:.12f}, {q[2]:.12f}, {q[3]:.12f}]",
        ]
    return "\n".join(lines) + "\n"


def pose_in_frame(M: np.ndarray, R_wc: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Rigid 4x4 c2w in the frame reached by similarity M (rotation+uniform scale+translation).

    The camera centre is mapped by M; the rotation drops M's scale so the result
    stays orthonormal (renderers expect rigid poses in scene units).
    """
    A = M[:3, :3]
    scale = np.cbrt(np.linalg.det(A))
    T = np.eye(4)
    T[:3, :3] = (A / scale) @ R_wc
    T[:3, 3] = A @ C + M[:3, 3]
    return T


def to_trajectory_json(cams: list[Camera], stage_to_scene: np.ndarray, frame_count: int, fps: float) -> dict:
    """Camera request for a novel-view renderer (Atlas).

    Atlas' camera format is not public yet, so both common conventions are
    written. Poses are expressed in the *scene* (Atlas reconstruction) frame and
    scene units. Every camera is locked off (same pose for every frame), which
    is what MAMMA wants.
    """
    views = []
    for c in cams:
        views.append({
            "name": c.name,
            "width": c.width, "height": c.height,
            "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
            "hfov_deg": c.hfov_deg, "vfov_deg": c.vfov_deg,
            "c2w_opencv": pose_in_frame(stage_to_scene, c.R_wc, c.C).tolist(),
            "c2w_opengl": pose_in_frame(stage_to_scene, c.R_wc @ OPENCV_TO_OPENGL, c.C).tolist(),
            "static": True,
        })
    return {
        "frame_count": frame_count,
        "fps": fps,
        "note": "Every view is a locked-off camera; render the full synced time range for each.",
        "views": views,
    }


def to_unity_json(cams: list[Camera]) -> dict:
    """Camera list for the Unity VirtualCameraRig gizmos (unity frame)."""
    out = []
    for c in cams:
        forward = stage_to_unity(c.R_wc[:, 2])
        up = stage_to_unity(-c.R_wc[:, 1])
        pos = stage_to_unity(c.C)
        out.append({
            "name": c.name,
            "position": dict(zip("xyz", map(float, pos))),
            "forward": dict(zip("xyz", map(float, forward))),
            "up": dict(zip("xyz", map(float, up))),
            "vfov_deg": c.vfov_deg,
            "aspect": c.width / c.height,
        })
    return {"cameras": out}


def save_cameras_npz(path: Path, cams: list[Camera]) -> None:
    np.savez(path,
             names=np.array([c.name for c in cams]),
             K=np.stack([c.K for c in cams]),
             T_wc=np.stack([c.T_wc for c in cams]),
             size=np.array([[c.width, c.height] for c in cams]))


def load_cameras_npz(path: Path) -> list[Camera]:
    d = np.load(path)
    cams = []
    for name, K, T, (w, h) in zip(d["names"], d["K"], d["T_wc"], d["size"]):
        cams.append(Camera(str(name), int(w), int(h), K[0, 0], K[1, 1], K[0, 2], K[1, 2], T[:3, :3], T[:3, 3]))
    return cams


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
