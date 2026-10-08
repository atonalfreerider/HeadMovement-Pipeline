"""Cross-process locks of the pipeline (PIPELINE_MONITOR 4.4), shared by EVERY orchestrator (started by the monitor
or from the CLI), so the workstation limits hold globally and not only inside one process:

    work/<take>/pipeline/run.lock       one orchestrator per take (state.json / events.jsonl have one writer)
    work/pipeline/pod_session.lock      one pod session at a time (plus any `remote run` process started elsewhere)
    work/pipeline/slots/local0|1.lock   the 2 local stage slots (a heavy stage takes both: it runs alone)
    work/pipeline/slots/gpu.lock        1 local GPU stage (stage code also takes work/gpu.lock itself)

A lock is a small JSON file created with O_EXCL: {pid, create_time, at, ...owner}. A lock whose holder process is
gone (pid + create time: pids get reused) is stale and is taken over. Locks are not re-entrant (two stages of one
orchestrator need two slots). Release removes the file only when this process holds it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import paths
from .util import pid_alive, read_json

LOCAL_SLOTS = 2
_ME: list = []


def me() -> tuple[int, float | None]:
    if not _ME:
        try:
            import psutil
            ct = psutil.Process(os.getpid()).create_time()
        except Exception:  # noqa: BLE001
            ct = None
        _ME.extend([os.getpid(), ct])
    return _ME[0], _ME[1]


def take_lock(take: str) -> Path:
    return paths.pipeline_dir(take) / "run.lock"


def pod_lock() -> Path:
    return paths.work() / "pipeline" / "pod_session.lock"


def slots_dir() -> Path:
    return paths.work() / "pipeline" / "slots"


def holder(path: Path) -> dict | None:
    """The live holder's record, or None (no lock, or a stale one)."""
    path = Path(path)
    if not path.exists():
        return None
    d = read_json(path, None)
    if not isinstance(d, dict):
        try:  # being written right now (or unreadable): held for a few seconds, then stale
            return {"pid": None, "unreadable": True} if time.time() - path.stat().st_mtime < 10 else None
        except OSError:
            return None
    if not pid_alive(d.get("pid"), d.get("create_time")):
        return None
    return d


def _mine(d: dict | None) -> bool:
    pid, ct = me()
    return bool(d) and d.get("pid") == pid and (ct is None or d.get("create_time") is None
                                                 or abs(float(d["create_time"]) - ct) < 1.0)


def acquire(path: Path, **owner) -> bool:
    """Create the lock (O_EXCL); take over a stale one. False when a live process holds it (this one included)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pid, ct = me()
    rec = {"pid": pid, "create_time": ct, "at": time.time(), **owner}
    for _ in range(4):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if holder(path) is not None:
                return False
            # stale: move it aside (atomic), then retry the exclusive create
            aside = path.with_name(f"{path.name}.stale.{pid}.{int(time.time() * 1000)}")
            try:
                os.replace(path, aside)
                os.unlink(aside)
            except OSError:
                time.sleep(0.05)
            continue
        except OSError:
            time.sleep(0.05)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec))
        return True
    return False


def release(path: Path) -> None:
    path = Path(path)
    try:
        d = read_json(path, None)
        if _mine(d):
            path.unlink()
    except OSError:
        pass


def acquire_local(owner: dict, heavy: bool = False, gpu: bool = False) -> tuple[list[Path] | None, str | None]:
    """The local slots a stage needs (1, or all of them for a heavy stage, plus the GPU slot). All or nothing."""
    got: list[Path] = []
    if gpu:
        g = slots_dir() / "gpu.lock"
        if not acquire(g, **owner):
            h = holder(g) or {}
            return None, f"a local GPU stage runs in another pipeline run ({h.get('take')}/{h.get('stage')})"
        got.append(g)
    need = LOCAL_SLOTS if heavy else 1
    n = 0
    for i in range(LOCAL_SLOTS):
        if n >= need:
            break
        p = slots_dir() / f"local{i}.lock"
        if acquire(p, **owner):
            got.append(p)
            n += 1
    if n < need:
        for p in got:
            release(p)
        busy = [holder(slots_dir() / f"local{i}.lock") for i in range(LOCAL_SLOTS)]
        who = ", ".join(f"{h.get('take')}/{h.get('stage')}" for h in busy if h)
        return None, (f"heavy: waits for every local slot ({who} running)" if heavy else
                      f"{LOCAL_SLOTS} local stages at a time across pipeline runs ({who})")
    return got, None


def release_all(locks: list[Path] | None) -> None:
    for p in locks or []:
        release(p)


def external_session_runner() -> bool:
    """A `dancecap remote run` / face_pod run process (or a hand-started hands_pod run: retired from the pipeline on
    2026-10-07, but it still creates a pod) exists (a pod session started elsewhere, or a session this process left
    running). The caller only asks when it runs no session itself."""
    try:
        from .runpod_view import _runner_processes
        return _runner_processes() > 0
    except Exception:  # noqa: BLE001
        return False
