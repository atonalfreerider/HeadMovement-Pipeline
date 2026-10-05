"""Atlas provider interface.

Atlas (World Labs) takes images/video plus explicit camera geometry and returns
camera-controlled video and static 3D Gaussian splats. Its API/file formats are
not public yet, so the pipeline talks to this interface and the concrete
provider is chosen in session.toml ([atlas] provider = "manual" | "http").
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..cameras import Camera


@dataclass
class AtlasJob:
    session: str
    source_videos: dict[str, Path]  # synced, constant-rate, same frame count
    fps: float
    frame_count: int
    virtual_cameras: list[Camera]  # in stage frame
    stage_to_scene: np.ndarray  # 4x4, stage frame -> Atlas scene frame
    want_depth: bool = False
    want_environment_splat: bool = True
    want_splat_sequence: bool = False  # per-frame dynamic splats, if Atlas ever exposes them


@dataclass
class AtlasResult:
    views: dict[str, Path]  # virtual camera name -> rendered video (same frame count as job)
    environment_splat: Path | None = None  # static 3DGS .ply in the Atlas scene frame
    splat_sequence_dir: Path | None = None  # frame_00000.ply ... in the Atlas scene frame
    depth: dict[str, Path] = field(default_factory=dict)
    # poses Atlas estimated for the *real* input cameras (scene frame, opencv c2w), if returned
    source_camera_poses: dict[str, np.ndarray] = field(default_factory=dict)


class AtlasProvider(ABC):
    def __init__(self, work_dir: Path, cfg: dict):
        self.work_dir = work_dir
        self.cfg = cfg

    @abstractmethod
    def submit(self, job: AtlasJob) -> str:
        """Start (or describe) the job. Returns a job id."""

    @abstractmethod
    def fetch(self, job: AtlasJob, job_id: str) -> AtlasResult | None:
        """Return results if ready, None if still pending. Raise on invalid results."""
