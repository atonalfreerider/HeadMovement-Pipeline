"""Atomic JSON state of one dance's farm (FARM.md sections 6-7): pods, chunks, jobs, events, spend.

`work/<dance>/farm/state.json` is the single source of truth for the Pipeline Monitor next to `coverage.json`. Several
processes write it (the CLI, the detached watcher, launch/pull helpers), so every change is a locked read-modify-write
of the whole file, written to a temp file and renamed over the target (a reader never sees a half-written file):

    st = State("larissa_kadu_full")
    st.update(lambda s: s["chunks"]["g2146"].update(status="running"))      # one locked transaction
    with st.locked() as s:                                                   # several edits, one write
        s["pods"][pod_id] = {...}
        add_event(s, "pod", "pod up", pod=pod_id)
    st.set_chunk("g2146", status="done", progress=1.0)                       # helpers: one transaction each
    st.event("pull", "chunk g2146 pulled", chunk="g2146")

The lock is a lock FILE created with O_EXCL (works on Windows and on the pod's Linux volume); a lock older than `stale`
seconds belongs to a dead process and is broken. Nothing here is daemon-specific: the watcher only calls these.
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Callable

from . import state_path

SCHEMA = "dancecap.farm.state/1"
EVENT_CAP = 500
# chunk lifecycle: planned (spec only) -> queued (spec + inputs on the volume) -> running -> done (done.json on the
# volume) -> pulled (verified local copy) -> stitched (in the final result); failed / cancelled end a try
CHUNK_STATUSES = ("planned", "queued", "running", "done", "pulled", "stitched", "failed", "cancelled")
CLAIMING = ("queued", "running", "done", "pulled", "stitched")   # statuses whose frames the planner treats as taken


class LockTimeout(RuntimeError):
    pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def atomic_write_json(path: Path, obj, indent: int | None = 1, retries: int = 40) -> None:
    """Write JSON to path atomically (temp file in the same directory + os.replace). Windows refuses the rename while a
    reader has the target open: retry for a couple of seconds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{int(time.time() * 1000) % 10**9}.tmp")
    data = json.dumps(obj, indent=indent, ensure_ascii=False, default=_default)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(data)
        fh.flush()
        with contextlib.suppress(OSError):
            os.fsync(fh.fileno())
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == retries - 1:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
            time.sleep(0.05)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def read_json(path: Path, default=None, retries: int = 20):
    """JSON file -> object, `default` when missing; retries a transient sharing violation / half-seen file."""
    path = Path(path)
    for i in range(retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (PermissionError, json.JSONDecodeError):
            if i == retries - 1:
                raise
            time.sleep(0.05)
    return default


class FileLock:
    """Cross-process lock file (O_EXCL). `stale`: seconds after which a lock is considered abandoned and broken."""

    def __init__(self, path: Path, timeout: float = 30.0, stale: float = 120.0, poll: float = 0.05):
        self.path, self.timeout, self.stale, self.poll = Path(path), timeout, stale, poll
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        t_end = time.time() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "t": time.time()}).encode())
                self._fd = fd
                return
            except FileExistsError:
                self._maybe_break()
            except PermissionError:  # Windows: the file is being deleted by its holder
                pass
            if time.time() > t_end:
                raise LockTimeout(f"{self.path} is held ({self._holder()}) for more than {self.timeout:.0f} s")
            time.sleep(self.poll)

    def _holder(self) -> str:
        try:
            return self.path.read_text()[:200]
        except OSError:
            return "?"

    def _maybe_break(self) -> None:
        try:
            m0 = self.path.stat().st_mtime
        except OSError:
            return
        if time.time() - m0 > self.stale:
            with contextlib.suppress(OSError):
                if self.path.stat().st_mtime == m0:  # nobody renewed it meanwhile
                    self.path.unlink()

    def release(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None
            with contextlib.suppress(OSError):
                self.path.unlink()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def empty_state(dance: str) -> dict:
    return {"schema": SCHEMA, "dance": dance, "created": now_iso(), "updated": now_iso(), "world": {}, "pods": {},
            "chunks": {}, "jobs": {}, "events": [], "spend": {}, "watcher": {}}


def add_event(s: dict, kind: str, msg: str, **data) -> dict:
    """Append an event to an open state dict (use inside State.locked()/update()); keeps the last EVENT_CAP."""
    ev = {"ts": now_iso(), "t": round(time.time(), 3), "kind": kind, "msg": msg, **data}
    s.setdefault("events", []).append(ev)
    if len(s["events"]) > EVENT_CAP:
        del s["events"][:len(s["events"]) - EVENT_CAP]
    return ev


class State:
    def __init__(self, dance: str, path: Path | None = None, lock_timeout: float = 30.0):
        self.dance = dance
        self.path = Path(path) if path else state_path(dance)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.lock_timeout = lock_timeout

    # reading never takes the lock: writes are atomic renames
    def load(self) -> dict:
        s = read_json(self.path, None)
        if not isinstance(s, dict):
            return empty_state(self.dance)
        for k, v in empty_state(self.dance).items():
            s.setdefault(k, v)
        return s

    @contextlib.contextmanager
    def locked(self):
        """Yield the state dict under the lock; it is written back (atomically) when the block ends without an error."""
        with FileLock(self.lock_path, timeout=self.lock_timeout):
            s = self.load()
            yield s
            s["updated"] = now_iso()
            atomic_write_json(self.path, s)

    def update(self, fn: Callable[[dict], object]):
        """One locked transaction: fn(state) mutates the dict; returns fn's result."""
        with self.locked() as s:
            return fn(s)

    # ---- helpers: one transaction each -----------------------------------------------------------------------------
    def set_chunk(self, chunk_id: str, **fields) -> dict:
        def f(s):
            c = s["chunks"].setdefault(chunk_id, {"id": chunk_id})
            if "status" in fields and fields["status"] not in CHUNK_STATUSES:
                raise ValueError(f"unknown chunk status {fields['status']!r}")
            if "status" in fields and fields["status"] != c.get("status"):
                c["status_ts"] = now_iso()
                add_event(s, "chunk", f"{chunk_id}: {c.get('status', 'new')} -> {fields['status']}", chunk=chunk_id,
                          status=fields["status"])
            c.update(fields)
            return dict(c)
        return self.update(f)

    def put_chunks(self, chunks: list[dict], replace_planned: bool = True) -> list[str]:
        """Store chunk dicts by id. Chunks whose stored status is 'planned' (or absent) are replaced by the new plan,
        every other stored chunk is kept as it is (its status, pod, progress are the watcher's). Returns the ids that were
        added or replaced. With replace_planned, planned chunks that are not in `chunks` any more are dropped."""
        def f(s):
            keep, changed = {}, []
            new_ids = {c["id"] for c in chunks}
            for cid, c in s["chunks"].items():
                if c.get("status") not in (None, "planned") or cid in new_ids or not replace_planned:
                    keep[cid] = c
            for c in chunks:
                old = keep.get(c["id"])
                if old and old.get("status") not in (None, "planned"):
                    continue
                keep[c["id"]] = {**(old or {}), **c, "status": c.get("status") or "planned"}
                changed.append(c["id"])
            s["chunks"] = dict(sorted(keep.items(), key=lambda kv: (kv[1].get("k0", 0), kv[0])))
            if changed:
                add_event(s, "plan", f"{len(changed)} chunk(s) planned: {', '.join(changed)}")
            return changed
        return self.update(f)

    def chunk(self, chunk_id: str) -> dict | None:
        return self.load()["chunks"].get(chunk_id)

    def upsert_pod(self, pod_id: str, **fields) -> dict:
        def f(s):
            p = s["pods"].setdefault(pod_id, {"id": pod_id, "first_seen": now_iso()})
            if "state" in fields and fields["state"] != p.get("state"):
                add_event(s, "pod", f"pod {pod_id}: {p.get('state', 'new')} -> {fields['state']}", pod=pod_id)
            p.update(fields)
            return dict(p)
        return self.update(f)

    def remove_pod(self, pod_id: str, keep_history: bool = True) -> None:
        def f(s):
            p = s["pods"].get(pod_id)
            if p is None:
                return
            if keep_history:
                p["state"] = "deleted"
                p["deleted"] = now_iso()
            else:
                del s["pods"][pod_id]
        self.update(f)

    def set_spend(self, **fields) -> dict:
        def f(s):
            s["spend"].update(fields)
            s["spend"]["updated"] = now_iso()
            return dict(s["spend"])
        return self.update(f)

    def set_world(self, **fields) -> dict:
        def f(s):
            s["world"].update(fields)
            return dict(s["world"])
        return self.update(f)

    def set_watcher(self, **fields) -> dict:
        def f(s):
            s["watcher"].update(fields)
            s["watcher"]["heartbeat"] = now_iso()
            return dict(s["watcher"])
        return self.update(f)

    def event(self, kind: str, msg: str, **data) -> dict:
        return self.update(lambda s: add_event(s, kind, msg, **data))

    def set_job(self, job_id: str, **fields) -> dict:
        def f(s):
            j = s["jobs"].setdefault(job_id, {"id": job_id})
            j.update(fields)
            return dict(j)
        return self.update(f)

    # ---- queries (lock-free snapshots) -----------------------------------------------------------------------------
    def claimed_ranges(self) -> list[tuple[int, int, str]]:
        """(core_k0, core_k1, chunk_id) of the chunks whose frames are taken (queued .. stitched)."""
        out = []
        for cid, c in self.load()["chunks"].items():
            if c.get("status") in CLAIMING and "core_k0" in c:
                out.append((int(c["core_k0"]), int(c["core_k1"]), cid))
        return sorted(out)

    def summary(self) -> dict:
        s = self.load()
        by: dict[str, int] = {}
        for c in s["chunks"].values():
            by[c.get("status", "?")] = by.get(c.get("status", "?"), 0) + 1
        pods = [p for p in s["pods"].values() if p.get("state") not in ("deleted", "gone")]
        return {"chunks": by, "pods_open": len(pods), "pods": [p.get("id") for p in pods],
                "spend": s.get("spend", {}), "events": len(s["events"]), "updated": s.get("updated")}
