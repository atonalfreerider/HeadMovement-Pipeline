"""Run a long pod command DETACHED from the ssh session that started it, and re-attach to it after a connection loss.

Why: remote_jobs runs every job command as one `ssh ... bash -lc` that streams its output. When the home connection
drops, ssh exits (rc 255) and the command dies with it (SIGPIPE/SIGHUP on its next write), and remote_jobs then packs
and DELETES the pod. A 55 min MAMMA fit must survive that. With this module the pod command is a child of its own
session (setsid), the ssh command only tails its log, and running the same ssh command again attaches to the SAME
child instead of starting another one.

Pod side (stdlib only):

    python -m dancecap.pod_detach run --name fit_c8 --log <child log> --timeout-min 95 -- '<shell command>'

  state in /workspace/.dancecap_detach/<name>.{pid,rc,cmd}; the child's combined output is --log (appended).
  - rc file present            -> print the log tail, exit with the recorded rc (a finished command is never re-run)
  - child alive                -> attach (tail from ~2 KB before the end)
  - neither                    -> start the child (own session, `timeout --kill-after=30`), tail from the start
  Exit code: the child's rc (255 is reported as 254: 255 is reserved for "the ssh connection was lost"); 250 when the
  child vanished without an rc (killed with its shell); 124 on the --timeout-min watchdog.

Local side: `make_resilient(pod)` swaps the pod object's class so that `pod.run(cmd)` of a command that contains
"dancecap.pod_detach" (every command of fit_full's jobs) re-runs the SAME ssh command (= re-attach) while the
connection is down, up to `max_lost_min` of continuous loss and while the pod still exists. Other commands are
untouched. Nothing here edits remote.py / remote_jobs.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("DANCECAP_DETACH_DIR", "/workspace/.dancecap_detach"))
RC_SSH_LOST = 255
RC_VANISHED = 250
RC_TIMEOUT = 124
MARK = "dancecap.pod_detach"


# ------------------------------------------------------------------------------------------------ pod side


def _alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would TERMINATE the process on Windows; only the tests run this branch
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie is not alive
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _read_int(p: Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def _map_rc(rc: int) -> int:
    return 254 if rc == RC_SSH_LOST else rc


def start(name: str, cmd: str, log: Path, timeout_min: float, state: Path = STATE) -> int:
    """Start the child in its own session; returns the pid of its shell. rc is written by the shell itself."""
    state.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    rc_f = state / f"{name}.rc"
    secs = max(1, int(timeout_min * 60))
    rcq = shlex.quote(rc_f.as_posix())
    script = (f"timeout --kill-after=30 {secs} bash -c {shlex.quote(cmd)}; rc=$?; "
              f"echo $rc > {rcq}.tmp && mv {rcq}.tmp {rcq}; exit $rc")
    (state / f"{name}.cmd").write_text(json.dumps({"cmd": cmd, "log": str(log), "timeout_min": timeout_min,
                                                   "started": time.time()}))
    with open(log, "ab") as fh:
        fh.write(f"[pod_detach] {time.strftime('%FT%TZ', time.gmtime())} start {name}\n".encode())
        fh.flush()
        proc = subprocess.Popen(["bash", "-c", script], stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    (state / f"{name}.pid").write_text(str(proc.pid))
    return proc.pid


def attach(name: str, log: Path, state: Path = STATE, poll_s: float = 2.0, from_start: bool = False,
           out=None) -> int:
    """Tail `log` until the child has written its rc; returns that rc (mapped). BrokenPipe -> 255 (ssh is gone)."""
    out = out or sys.stdout
    rc_f, pid_f = state / f"{name}.rc", state / f"{name}.pid"
    pos = 0 if from_start else max(0, (log.stat().st_size if log.exists() else 0) - 2048)
    gone_since = None

    def drain() -> bool:
        nonlocal pos
        if not log.exists():
            return False
        with open(log, "rb") as fh:
            fh.seek(pos)
            data = fh.read()
        if data:
            pos += len(data)
            out.write(data.decode("utf-8", "replace"))
            out.flush()
        return bool(data)

    try:
        while True:
            drain()
            rc = _read_int(rc_f)
            if rc is not None:
                while drain():
                    pass
                out.write(f"[pod_detach] {name} finished rc={rc}\n")
                out.flush()
                return _map_rc(rc)
            pid = _read_int(pid_f)
            if pid is None or not _alive(pid):
                gone_since = gone_since or time.time()
                if time.time() - gone_since > 6:  # the shell is gone and no rc was written: killed from outside
                    drain()
                    rc_f.write_text(str(RC_VANISHED))
                    out.write(f"[pod_detach] {name} vanished without an rc (killed?)\n")
                    out.flush()
                    return RC_VANISHED
            else:
                gone_since = None
            time.sleep(poll_s)
    except (BrokenPipeError, OSError):
        return RC_SSH_LOST


def run(name: str, cmd: str, log: Path, timeout_min: float, state: Path = STATE, out=None) -> int:
    rc_f, pid_f = state / f"{name}.rc", state / f"{name}.pid"
    out = out or sys.stdout
    if _read_int(rc_f) is not None:
        return attach(name, log, state, from_start=False, out=out)
    pid = _read_int(pid_f)
    if pid is not None and _alive(pid):
        out.write(f"[pod_detach] {name}: already running (pid {pid}), attaching\n")
        out.flush()
        return attach(name, log, state, from_start=False, out=out)
    start(name, cmd, log, timeout_min, state)
    out.write(f"[pod_detach] {name}: started detached, log {log}\n")
    out.flush()
    return attach(name, log, state, from_start=True, out=out)


def status(name: str, state: Path = STATE) -> dict:
    pid = _read_int(state / f"{name}.pid")
    return {"name": name, "pid": pid, "alive": bool(pid and _alive(pid)), "rc": _read_int(state / f"{name}.rc")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pod_detach", description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["run", "status", "kill"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--log")
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument("--timeout-min", type=float, default=120.0)
    ap.add_argument("cmd", nargs="*", help="after --: the shell command (one string)")
    # Python 3.11's argparse gives a nargs='*' positional zero arguments when the first positional ('run') is consumed and
    # then rejects everything after '--' as unrecognized (the pod preflight died on this, 2026-10-08): split at '--' here.
    argv = list(sys.argv[1:] if argv is None else argv)
    tail: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    a.cmd = list(a.cmd or []) + tail
    state = Path(a.state)
    if a.action == "status":
        print(json.dumps(status(a.name, state)))
        return 0
    if a.action == "kill":
        pid = _read_int(state / f"{a.name}.pid")
        if pid and _alive(pid):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        return 0
    if not a.cmd or not a.log:
        ap.error("run needs --log and a command after --")
    return run(a.name, " ".join(a.cmd), Path(a.log), a.timeout_min, state)


# ------------------------------------------------------------------------------------------------ local side


def _resilient_run(self, cmd: str, check: bool = True, timeout: float | None = None, log: Path | None = None,
                   echo: bool = True) -> int:
    from . import remote
    base = remote.Pod.run
    if MARK not in cmd:
        return base(self, cmd, check=check, timeout=timeout, log=log, echo=echo)
    t_end = time.time() + (timeout or 8 * 3600)
    lost_since = None
    gone = 0
    n_lost = 0
    while True:
        left = t_end - time.time()
        if left <= 0:
            raise TimeoutError(f"pod command exceeded {timeout:.0f}s: {cmd[:120]}")
        rc = base(self, cmd, check=False, timeout=left, log=log, echo=echo)
        if rc != RC_SSH_LOST:
            if check and rc != 0:
                raise RuntimeError(f"pod command failed ({rc}): {cmd[:200]}")
            if n_lost:
                _note(log, f"[resilient] re-attached after {n_lost} connection loss(es); rc={rc}")
            return rc
        n_lost += 1
        lost_since = lost_since or time.time()
        down_min = (time.time() - lost_since) / 60
        _note(log, f"[resilient] {time.strftime('%H:%M:%S')} ssh to pod {self.id} lost (loss #{n_lost}, "
                   f"{down_min:.1f} min); the pod command keeps running, re-attaching in 20 s")
        if down_min > getattr(self, "max_lost_min", 45):
            _note(log, "[resilient] giving up the re-attach: connection down too long")
            return RC_SSH_LOST
        time.sleep(20)
        if n_lost % 3 == 0:  # is the pod still there at all?
            try:
                info = remote._ctl("pod", "get", self.id, check=False)
                st = str((info or {}).get("desiredStatus") or (info or {}).get("status") or "") if isinstance(info, dict) else ""
                gone = gone + 1 if (not isinstance(info, dict) or not info.get("id") or st.upper() in
                                    ("EXITED", "TERMINATED", "STOPPED")) else 0
            except Exception:  # noqa: BLE001  (API unreachable too: the home link is down, not the pod)
                gone = 0
            if gone >= 3:
                _note(log, f"[resilient] pod {self.id} no longer exists; giving up")
                return RC_SSH_LOST
        # connected again? the next loop iteration re-runs the command = attaches
        if base(self, "true", check=False, timeout=40, log=None, echo=False) == 0:
            lost_since = None


def _note(log: Path | None, msg: str) -> None:
    print(msg, flush=True)
    if log:
        try:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass


def make_resilient(pod, max_lost_min: float = 45.0):
    """Swap `pod`'s class (a remote.Pod) for one whose run() re-attaches detached commands; returns the pod."""
    cls = type(pod)
    if getattr(cls, "_dancecap_resilient", False):
        return pod
    pod.__class__ = type("ResilientPod", (cls,), {"run": _resilient_run, "_dancecap_resilient": True})
    pod.max_lost_min = max_lost_min
    return pod


def detached(name: str, cmd: str, job: str, timeout_min: float) -> str:
    """The job-command string that runs `cmd` detached. The child's log goes next to the job's command logs
    (remote_jobs exports DANCECAP_SID to every command, so it lands in the session's pod_logs/<job>/ and is packed)."""
    log = f"/workspace/HeadMovement/dancecap/work/runpod/sessions/$DANCECAP_SID/pod_logs/{job}/{name}.child.log"
    return (f"python -m dancecap.pod_detach run --name {shlex.quote(name)} --log \"{log}\" "
            f"--timeout-min {timeout_min:g} -- {shlex.quote(cmd)}")


if __name__ == "__main__":
    sys.exit(main())
