"""Farm: chunked, multi-pod, volume-backed 3D-pose pipeline (contract: dancecap/docs/FARM.md).

This package is imported on the workstation AND on a Runpod pod, so the core modules stay light (stdlib + numpy; PIL
only for the coverage PNG; pycolmap only for the world/room consistency check) and import nothing heavy at module level.

Core modules (stage "core"):
    timeline  grid / 1-s cell math, coverage computed from the real outputs, the defrag-map PNG
    chunks    chunk model, gap planner, job-spec writer (step templates live here), S1 segment registration
    world     the WORLD PACKAGE (build / verify / seal / publish / compare), hash helpers, mismatch refusal
    state     atomic JSON state under work/<dance>/farm/ with a file lock
    cli       `python -m dancecap.farm ...` and the plug-in mechanism for the infra / watch modules

Paths (all derived here so every module agrees):
    work/<dance>/farm/{state.json, coverage.json, coverage_frames.npz, plan.json, jobs/, chunks/, review/, logs/}
    dancecap/worlds/<dance>/<world_id>/{manifest.json, files.json}   tracked: hashes + relative paths only
    dancecap/worlds/<dance>/{CURRENT.json, seals.json}               tracked pointers
    work/worlds/<dance>/<world_id>/...                              git-ignored data (same tree on the volume)
"""
from __future__ import annotations

import os
from pathlib import Path

from ..take import PACKAGE_ROOT, WORKSPACE

DEFAULT_DANCE = os.environ.get("DANCECAP_FARM_DANCE", "larissa_kadu_full")
VOLUME_ROOT = "/workspace/hm"  # FARM.md section 4: the network volume mount point + root
CORE_MODULES = ("timeline", "chunks", "analysis", "world", "state", "cli")


def work_dir(dance: str) -> Path:
    return PACKAGE_ROOT / "work" / dance


def farm_dir(dance: str) -> Path:
    return work_dir(dance) / "farm"


def coverage_path(dance: str) -> Path:
    return farm_dir(dance) / "coverage.json"


def state_path(dance: str) -> Path:
    return farm_dir(dance) / "state.json"


def jobs_dir(dance: str) -> Path:
    return farm_dir(dance) / "jobs"


def chunks_dir(dance: str) -> Path:
    """Where pulled chunk outputs land locally: chunks/<chunk_id>/{out/, done.json, ...} (same shape as the volume's)."""
    return farm_dir(dance) / "chunks"


def review_dir(dance: str) -> Path:
    return farm_dir(dance) / "review"


def worlds_tracked(dance: str) -> Path:
    return PACKAGE_ROOT / "worlds" / dance


def worlds_data(dance: str) -> Path:
    return PACKAGE_ROOT / "work" / "worlds" / dance
