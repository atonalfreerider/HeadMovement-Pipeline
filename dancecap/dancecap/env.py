"""Where stage code runs: the local Windows workstation or a Linux Runpod pod.

Stages must not hard-code Windows paths/executables. Use these helpers so the same code runs locally and on a
pod (see dancecap.remote / remote_jobs). On a pod the workspace is laid out exactly like the local one:
<WORKSPACE>/dancecap, <WORKSPACE>/mamma-turing, <WORKSPACE>/mamma/data/..., <WORKSPACE>/smplx.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .take import WORKSPACE

IS_WINDOWS = os.name == "nt"


def is_pod() -> bool:
    """True inside a Runpod pod (Runpod sets RUNPOD_POD_ID; remote_jobs also sets DANCECAP_POD=1)."""
    return bool(os.environ.get("RUNPOD_POD_ID") or os.environ.get("DANCECAP_POD"))


def python_exe() -> str:
    """Interpreter for stage subprocesses (MAMMA scripts etc.): the local venv on Windows, else this interpreter."""
    if IS_WINDOWS:
        venv = WORKSPACE / ".venv-nopo4d" / "Scripts" / "python.exe"
        if venv.exists():
            return str(venv)
    return sys.executable


def mamma_root() -> Path:
    """Patched MAMMA checkout (local-only branch turing-dynamic-cams); DANCECAP_MAMMA_ROOT overrides."""
    return Path(os.environ.get("DANCECAP_MAMMA_ROOT", WORKSPACE / "mamma-turing"))


def mamma_data() -> Path:
    """MAMMA weights/body models (mamma/data in the main checkout); DANCECAP_MAMMA_DATA overrides."""
    return Path(os.environ.get("DANCECAP_MAMMA_DATA", WORKSPACE / "mamma" / "data"))


def cuda_wrapper(argv: list[str]) -> list[str]:
    """Prefix needed to JIT-build CUDA extensions: MSVC + CUDA env via run_cuda.cmd on Windows, nothing on Linux."""
    if IS_WINDOWS:
        return ["cmd", "/c", str(WORKSPACE / "prototype" / "run_cuda.cmd"), *argv]
    return [sys.executable, *argv]
