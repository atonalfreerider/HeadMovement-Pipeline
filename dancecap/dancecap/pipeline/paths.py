"""Where the pipeline backend reads and writes (one place, so tests can point everything at a temp tree).

The package root is `dancecap/` (takes/, work/, taxonomy/ live under it). `DANCECAP_PIPELINE_ROOT` overrides it for
tests and for child processes started by a test (the run wrapper inherits the variable). Nothing here imports heavy
modules: the monitor server imports this at start-up.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..take import PACKAGE_ROOT as _REAL_ROOT

REAL_ROOT = _REAL_ROOT
_override: list[Path | None] = [None]


def set_root(p: str | Path | None) -> None:
    """Point the backend at another package root (tests). None restores the default."""
    _override[0] = Path(p).resolve() if p is not None else None


def root() -> Path:
    if _override[0] is not None:
        return _override[0]
    env = os.environ.get("DANCECAP_PIPELINE_ROOT")
    return Path(env).resolve() if env else REAL_ROOT


def is_real_root() -> bool:
    return root() == REAL_ROOT


def workspace() -> Path:
    return root().parent


def takes_dir() -> Path:
    return root() / "takes"


def work() -> Path:
    return root() / "work"


def take_toml(take: str) -> Path:
    return takes_dir() / f"{take}.toml"


def take_work(take: str) -> Path:
    return work() / take


def pipeline_dir(take: str) -> Path:
    return take_work(take) / "pipeline"


def runs_dir() -> Path:
    return work() / "pipeline" / "runs"


def plans_dir() -> Path:
    return work() / "pipeline" / "plans"


def monitor_dir() -> Path:
    return work() / "monitor"


def runpod_dir() -> Path:
    return work() / "runpod"


def sessions_dir() -> Path:
    return runpod_dir() / "sessions"


def ledger_path() -> Path:
    return runpod_dir() / "ledger.json"


def taxonomy_dir() -> Path:
    return root() / "taxonomy"


def unity_project() -> Path:
    return workspace() / "head-movement"


def rel_display(p: str | Path) -> str:
    """Workspace-relative posix path for display (never an absolute path in an API response). String-based (no
    filesystem calls: the status snapshot formats hundreds of paths per poll)."""
    s = os.path.normpath(os.path.abspath(str(p)))
    ws = os.path.normpath(os.path.abspath(str(workspace())))
    if s.lower() == ws.lower():
        return "."
    if s.lower().startswith(ws.lower() + os.sep):
        return s[len(ws) + 1:].replace("\\", "/")
    return Path(s).name
