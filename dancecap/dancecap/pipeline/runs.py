"""Run registry (PIPELINE_MONITOR 4): every process the monitor or the orchestrator starts is a run on disk.

work/pipeline/runs/<run_id>/
    run.json     written by the starter: {id, kind, take, stages, plan_hash, origin, argv_display, cwd, created,
                 session_sid, env_keys, parent, url}
    pid.json     written by the wrapper itself: {pid, create_time, started} (the venv python.exe is a redirector:
                 the Popen pid is not the interpreter's)
    rc.json      written by the wrapper on exit: {rc, ended, error}
    output.log   stdout + stderr of the run
    stop.json    written by the monitor: {mode, at}

State is derived from these files, so a server restart re-discovers every run. Processes start with
CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP (never DETACHED_PROCESS: console grandchildren would pop windows),
stdin = DEVNULL; local heavy runs get BELOW_NORMAL priority so the UI and Unity stay responsive on 4 CPUs.

A run with rc.json has ENDED, whatever its rc: it is never active (a pipeline that ends at a human gate, rc 3, shows
`waiting_human` but blocks nothing and offers no stop). Finished runs are immutable, so their description is cached in
memory; run.json (take, kind, stages) is cached too, so listing a take's runs describes only that take's runs.
Old finished runs are moved to work/pipeline/runs_archive/<YYYYMM>/ by archive() (the newest run of every
take/stage pair stays, so a stage's last failure keeps showing).
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import paths
from .util import RUN_RE, pid_alive, read_json, redact, write_json_atomic

KINDS = ("pipeline", "stage", "session", "dry_run", "tool", "probe", "export", "media", "external")
ACTIVE = ("starting", "running", "stopping")
ENDED = ("done", "failed", "stopped", "waiting_human")
MAX_SCAN = 3000
_info_cache: dict[str, dict] = {}
_desc_cache: dict[str, dict] = {}
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
BELOW_NORMAL = 0x00004000
ENV_PASS = ("PYTHONIOENCODING", "PYTHONUTF8", "PYTHONUNBUFFERED", "DANCECAP_PROGRESS", "DANCECAP_PROGRESS_JOB",
            "DANCECAP_PIPELINE_ROOT", "DANCECAP_RUN_ID")


def new_id() -> str:
    return f"r-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def run_dir(run_id: str) -> Path:
    if not RUN_RE.match(run_id or ""):
        raise ValueError("invalid run id")
    return paths.runs_dir() / run_id


def python_exe() -> str:
    """The interpreter for runs: this one (the server and the orchestrator run in .venv-nopo4d)."""
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        cand = exe[:-len("pythonw.exe")] + "python.exe"
        if Path(cand).exists():
            exe = cand
    return exe


def start(kind: str, module_argv: list[str], take: str | None = None, stages: list[str] | None = None,
          plan_hash: str | None = None, origin: str = "monitor", parent: str | None = None,
          heavy: bool = False, env: dict | None = None, session_sid: str | None = None,
          run_id: str | None = None) -> dict:
    """Start `python -m dancecap.pipeline.runwrap --run-dir D -- <module_argv...>` and register it.

    module_argv starts with "-m", "<module>" (allowlisted by the caller: registry templates only)."""
    if kind not in KINDS:
        raise ValueError(f"unknown run kind {kind}")
    if not module_argv or module_argv[0] != "-m":
        raise ValueError("runs execute modules only (-m <module> ...)")
    rid = run_id or new_id()
    d = run_dir(rid)
    d.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(8):
        try:
            d.mkdir(exist_ok=False)
            break
        except FileExistsError:
            if run_id:
                raise
            rid = new_id()  # two runs started in the same second with the same 16-bit suffix
            d = run_dir(rid)
    cenv = dict(os.environ)
    extra = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "DANCECAP_RUN_ID": rid}
    if not paths.is_real_root():
        extra["DANCECAP_PIPELINE_ROOT"] = str(paths.root())
    if take:
        # dancecap.progress (PIPELINE_PROGRESS 3.1): instrumented stages write their events here (on Windows the
        # helper adds .<pid> to the name: one file per process)
        prog = paths.take_work(take) / "progress" / f"{(stages or [kind])[0]}.{rid}.jsonl"
        extra.update({"DANCECAP_PROGRESS": str(prog), "DANCECAP_PROGRESS_JOB": (stages or [kind])[0],
                      "DANCECAP_TAKE": take})
    extra.update(env or {})
    cenv.update(extra)
    pp = [str(paths.REAL_ROOT)] + ([cenv["PYTHONPATH"]] if cenv.get("PYTHONPATH") else [])
    cenv["PYTHONPATH"] = os.pathsep.join(pp)
    argv = [python_exe(), "-m", "dancecap.pipeline.runwrap", "--run-dir", str(d), "--", *module_argv[1:]]
    info = {"id": rid, "kind": kind, "take": take, "stages": stages or [], "plan_hash": plan_hash, "origin": origin,
            "argv_display": ["python", *module_argv], "cwd": paths.rel_display(paths.root()) or "dancecap",
            "created": time.time(), "session_sid": session_sid, "parent": parent, "heavy": heavy,
            "env_keys": sorted(k for k in extra if k in ENV_PASS or k.startswith("DANCECAP_")), "url": None}
    write_json_atomic(d / "run.json", info)
    flags = 0
    if os.name == "nt":
        flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | (BELOW_NORMAL if heavy else 0)
    with open(d / "output.log", "ab") as log:
        p = subprocess.Popen(argv, cwd=str(paths.root()), env=cenv, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, creationflags=flags, close_fds=True, start_new_session=os.name != "nt")
    write_json_atomic(d / "launcher.json", {"launcher_pid": p.pid, "at": time.time()})
    return describe(rid)


def _state(d: Path, info: dict) -> tuple[str, dict, dict]:
    pid = read_json(d / "pid.json", {}) or {}
    rc = read_json(d / "rc.json", None)
    stop = read_json(d / "stop.json", None)
    if rc is not None:
        code = rc.get("rc")
        stopped_before_end = bool(stop) and float(stop.get("at") or 0) <= float(rc.get("ended") or 0) + 1
        if stop and code != 0:
            state = "stopped"
        elif code == 0:
            # a tool that swallows the interrupt (the narration server) exits 0: it was still stopped by the user
            state = "stopped" if stopped_before_end and info.get("kind") in ("tool", "media", "probe") else "done"
        elif code == 3 and info.get("kind") in ("pipeline", "dry_run"):
            state = "waiting_human"  # ended at a human gate: final, never active
        else:
            state = "failed"
        return state, pid, rc
    if not pid:
        launcher = read_json(d / "launcher.json", {}) or {}
        if launcher and time.time() - float(launcher.get("at", 0)) > 60 and not pid_alive(launcher.get("launcher_pid")):
            return "lost", pid, {}
        return ("stopping" if stop else "starting"), pid, {}
    if pid_alive(pid.get("pid"), pid.get("create_time")):
        return ("stopping" if stop else "running"), pid, {}
    return ("stopped" if stop else "lost"), pid, {}


def _ck(run_id: str) -> str:
    return f"{paths.runs_dir()}|{run_id}"  # tests use several roots in one process


def _info(run_id: str) -> dict | None:
    """run.json (written once by start(); cached)."""
    c = _info_cache.get(_ck(run_id))
    if c is not None:
        return c
    info = read_json(run_dir(run_id) / "run.json", None)
    if info:
        _info_cache[_ck(run_id)] = info
    return info


def describe(run_id: str, fresh: bool = False) -> dict | None:
    if not fresh:
        c = _desc_cache.get(_ck(run_id))
        if c is not None:
            return dict(c)
    d = run_dir(run_id)
    info = _info(run_id) if not fresh else read_json(d / "run.json", None)
    if not info:
        return None
    state, pid, rc = _state(d, info)
    cur = read_json(d / "current.json", {}) or {}
    stop = (read_json(d / "stop.json", None) or {}) if state == "stopping" else {}
    out = dict(info)
    out.update({"state": state, "pid": pid.get("pid"), "started": pid.get("started"), "ended": rc.get("ended"),
                "rc": rc.get("rc"), "error": redact(rc.get("error") or "") or None,
                "current": cur or None, "log": f"/api/runs/{run_id}/log",
                "stop_modes": stop_modes(info, state, cur, stop.get("mode"))})
    if cur.get("url"):
        out["url"] = cur["url"]
    if cur.get("progress"):
        out["progress"] = cur["progress"]
    if state in ("starting", "running") and info.get("kind") == "pipeline" and cur.get("session_active"):
        out["stop_note"] = ("a pod session is running: 'now' is not offered (it would leave the pod without its "
                            "download); stop after the step, or delete the pod in the Runpod panel")
    if state in ENDED and rc:
        _desc_cache[_ck(run_id)] = dict(out)  # finished runs never change
    return out


def stop_modes(info: dict, state: str, cur: dict | None = None, stopping_mode: str | None = None) -> list[str]:
    if state == "stopping" and info.get("kind") == "pipeline" and stopping_mode == "after_step":
        # already finishing its step: may still be escalated to 'now' (never while a pod session runs)
        return [] if (cur or {}).get("session_active") else ["now"]
    if state not in ("starting", "running"):
        return []  # ended (rc.json) or already stopping: nothing to stop
    if info.get("kind") == "pipeline":
        # never 'now' while a pod session command runs: its pod would be left without download / ledger close
        return ["after_step"] if (cur or {}).get("session_active") else ["after_step", "now"]
    if info.get("kind") == "session":
        return []  # pod sessions: only delete_pod with a typed confirmation (PIPELINE_MONITOR 4.3 / 9)
    return ["now"]


def list_runs(take: str | None = None, active: bool = False, limit: int = 50, kind: str | None = None,
              max_scan: int = MAX_SCAN) -> list[dict]:
    """Newest first. Filters on the cached run.json BEFORE describing, so only matching runs are read; ended runs
    come from the cache."""
    d = paths.runs_dir()
    if not d.exists():
        return []
    try:
        names = sorted((e.name for e in os.scandir(d) if RUN_RE.match(e.name)), reverse=True)[:max_scan]
    except OSError:
        return []
    out = []
    for name in names:
        info = _info(name)
        if not info:
            continue
        if take and info.get("take") != take:
            continue
        if kind and info.get("kind") != kind:
            continue
        if active and _ck(name) in _desc_cache:
            continue  # ended
        r = describe(name)
        if not r:
            continue
        if active and r["state"] not in ACTIVE:
            continue
        out.append(r)
        if len(out) >= limit:
            break
    return out


def archive(max_age_days: float = 14.0, keep: int = 300, now: float | None = None) -> int:
    """Move ended runs older than max_age_days (beyond the newest `keep`) to work/pipeline/runs_archive/<YYYYMM>/.
    The newest run of each take/stage pair always stays. Returns the number of runs moved."""
    import shutil
    now = now or time.time()
    d = paths.runs_dir()
    if not d.exists():
        return 0
    names = sorted((e.name for e in os.scandir(d) if RUN_RE.match(e.name)), reverse=True)
    seen_pairs: set[tuple] = set()
    moved = 0
    for i, name in enumerate(names):
        info = _info(name) or {}
        pairs = {(info.get("take"), st) for st in (info.get("stages") or [info.get("kind")])}
        newest_of_a_pair = bool(pairs - seen_pairs)
        seen_pairs |= pairs
        if i < keep or newest_of_a_pair:
            continue
        r = describe(name)
        if not r or r["state"] not in ENDED:
            continue
        if now - float(r.get("ended") or r.get("created") or now) < max_age_days * 86400:
            continue
        dst = paths.work() / "pipeline" / "runs_archive" / name[2:8] / name
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d / name), str(dst))
            moved += 1
            _desc_cache.pop(_ck(name), None)
            _info_cache.pop(_ck(name), None)
        except OSError:
            continue
    return moved


def request_stop(run_id: str, mode: str) -> dict:
    r = describe(run_id, fresh=True)
    if r is None:
        raise KeyError(run_id)
    if mode not in r["stop_modes"]:
        raise PermissionError(f"stop mode {mode!r} is not offered for this run ({r['stop_modes']})"
                              + (f": {r['stop_note']}" if r.get("stop_note") else ""))
    write_json_atomic(run_dir(run_id) / "stop.json", {"mode": mode, "at": time.time()})
    return describe(run_id, fresh=True)


def set_current(run_id: str, **kw) -> None:
    """Written by the run itself (orchestrator / tool) through the env DANCECAP_RUN_ID."""
    try:
        d = run_dir(run_id)
        cur = read_json(d / "current.json", {}) or {}
        cur.update(kw)
        write_json_atomic(d / "current.json", cur)
    except (ValueError, OSError):
        pass


def log_chunk(path: Path, offset: int = 0, max_bytes: int = 65536, tail: bool = False) -> dict:
    """LogChunk: lines cut at '\\n' only, redacted, plain text."""
    try:
        size = path.stat().st_size
    except OSError:
        return {"offset": offset, "next": offset, "size": 0, "eof": True, "lines": [], "truncated_head": False}
    max_bytes = max(1024, min(int(max_bytes), 1 << 20))
    truncated = False
    if tail and offset == 0 and size > max_bytes:
        offset, truncated = size - max_bytes, True
    offset = max(0, min(int(offset), size))
    with open(path, "rb") as fh:
        fh.seek(offset)
        data = fh.read(max_bytes)
    if truncated:
        nl = data.find(b"\n")
        if nl >= 0:
            offset += nl + 1
            data = data[nl + 1:]
    end = data.rfind(b"\n")
    if end < 0 and offset + len(data) < size:
        data = b""
    elif end >= 0 and offset + len(data) < size:
        data = data[:end + 1]
    elif end >= 0 and offset + len(data) >= size and not data.endswith(b"\n"):
        data = data[:end + 1]  # hold back an incomplete last line
    txt = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = [redact(x) for x in txt.split("\n")[:-1]] if txt else []
    nxt = offset + len(data)
    return {"offset": offset, "next": nxt, "size": size, "eof": nxt >= size, "lines": lines, "truncated_head": truncated}
