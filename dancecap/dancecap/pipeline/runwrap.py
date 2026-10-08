"""Wrapper of every process the monitor or the orchestrator starts (PIPELINE_MONITOR 4.2).

    python -m dancecap.pipeline.runwrap --run-dir work/pipeline/runs/<id> -- <module> [args...]

- On start: pid.json {pid: os.getpid(), create_time, started} (the venv python.exe is a redirector, so only the
  interpreter knows its own pid).
- Runs the module with runpy (run_name="__main__", sys.argv set).
- On exit: rc.json {rc, ended, error} from a finally block (and atexit as a backstop).
- Stop watcher: a daemon thread polls stop.json every second. Mode `now`:
  1. KeyboardInterrupt in the main thread FIRST, so the module's own cleanup runs (finally blocks; the orchestrator
     writes stop.json to each of its child runs and waits for them, so every child writes its rc.json);
  2. stage / tool runs only: after 3 s, child processes are terminated (a main thread blocked waiting on a child -
     COLMAP, ffmpeg - then gets its interrupt);
  3. after the grace (DANCECAP_STOP_GRACE_S, default 30 s; a pipeline run gets 2 x grace + 15 s, longer than its
     children's) the remaining children are killed and the run exits with 130.
  A pod-session runner (`dancecap remote run`, hands_pod / face_pod run, or any child run of kind `session`) and its
  descendants are NEVER terminated or killed here: killing it would leave its pod running without the download or
  the ledger close. A run of kind `session` ignores stop.json (stop it by deleting its pod, typed confirmation).
  Mode `after_step` is left to the module (the orchestrator polls it between steps).
"""
from __future__ import annotations

import _thread
import atexit
import json
import os
import runpy
import sys
import threading
import time
import traceback
from pathlib import Path

GRACE_S = float(os.environ.get("DANCECAP_STOP_GRACE_S", 30))
_written = [False]


def _write(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    for i in range(4):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1 * (i + 1))


def _create_time() -> float | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).create_time()
    except Exception:
        return None


# hands_pod is retired from the pipeline (2026-10-07); it stays listed because a hand-started run still creates a pod
SESSION_MARKERS = (" remote run", "hands_pod run", "face_pod run")


def _run_kind(run_dir: Path) -> str | None:
    try:
        return (json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8")) or {}).get("kind")
    except (OSError, ValueError):
        return None


def _is_session_proc(p) -> bool:
    """A pod-session runner: its run.json says kind session, or its command line is a pod runner's."""
    try:
        cl = p.cmdline()
    except Exception:  # noqa: BLE001
        return False
    if "--run-dir" in cl:
        i = cl.index("--run-dir")
        if i + 1 < len(cl) and _run_kind(Path(cl[i + 1])) == "session":
            return True
    line = " ".join(cl)
    return "dancecap" in line and any(m in line for m in SESSION_MARKERS)


def _children(protect_sessions: bool = True) -> list:
    """This process's descendants, minus every pod-session runner and its own descendants."""
    try:
        import psutil
        kids = psutil.Process(os.getpid()).children(recursive=True)
    except Exception:  # noqa: BLE001
        return []
    if not protect_sessions:
        return kids
    keep: set[int] = set()
    for k in kids:
        if k.pid in keep or not _is_session_proc(k):
            continue
        keep.add(k.pid)
        try:
            keep.update(c.pid for c in k.children(recursive=True))
        except Exception:  # noqa: BLE001
            pass
    return [k for k in kids if k.pid not in keep]


def _terminate(kids: list, timeout: float = 5.0) -> int:
    for k in kids:
        try:
            k.terminate()
        except Exception:  # noqa: BLE001
            pass
    try:
        import psutil
        _, alive = psutil.wait_procs(kids, timeout=timeout)
        for k in alive:
            try:
                k.kill()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return len(kids)


def _children_kill() -> int:
    """Terminate (then kill) every non-session descendant."""
    return _terminate(_children())


def _finish(run_dir: Path, rc: int, error: str | None) -> None:
    if _written[0]:
        return
    _written[0] = True
    _write(run_dir / "rc.json", {"rc": rc, "ended": time.time(), "error": (error or None) and error[:500]})


def _stop_mode(stop: Path) -> str | None:
    try:
        return (json.loads(stop.read_text(encoding="utf-8")) or {}).get("mode", "now")
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return "now"


def _watch(run_dir: Path, stop_evt: threading.Event, kind: str | None = None) -> None:
    """mode 'now' stops this process (see the module docstring); 'after_step' is left to the module."""
    stop = run_dir / "stop.json"
    told = False
    while not stop_evt.wait(1.0):
        mode = _stop_mode(stop)
        if mode is None or mode == "after_step":
            continue
        if kind == "session":
            if not told:
                print("[runwrap] stop ignored: a pod session is stopped by deleting its pod (Runpod panel, typed "
                      "confirmation)", flush=True)
                told = True
            continue
        print(f"[runwrap] stop requested ({mode}): interrupting the run (its cleanup runs first)", flush=True)
        _thread.interrupt_main()
        if kind != "pipeline":
            if stop_evt.wait(3.0):
                return
            n = _children_kill()
            if n:
                print(f"[runwrap] {n} child process(es) terminated", flush=True)
            grace = GRACE_S
        else:
            grace = 2 * GRACE_S + 15  # the orchestrator stops its child runs and waits for them (GRACE_S + 5)
        if stop_evt.wait(grace):
            return
        n = _children_kill()
        print(f"[runwrap] grace period over: {n} child process(es) killed (pod sessions never)", flush=True)
        _finish(run_dir, 130, "stopped (killed after the grace period)")
        os._exit(130)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 4 or argv[0] != "--run-dir" or "--" not in argv:
        print("usage: python -m dancecap.pipeline.runwrap --run-dir <dir> -- <module> [args...]", file=sys.stderr)
        return 2
    run_dir = Path(argv[1])
    rest = argv[argv.index("--") + 1:]
    module, margs = rest[0], rest[1:]
    run_dir.mkdir(parents=True, exist_ok=True)
    _write(run_dir / "pid.json", {"pid": os.getpid(), "create_time": _create_time(), "started": time.time()})
    stop_evt = threading.Event()
    threading.Thread(target=_watch, args=(run_dir, stop_evt, _run_kind(run_dir)), daemon=True,
                     name="runwrap-stop").start()
    atexit.register(lambda: _finish(run_dir, 1, "exited without a result"))
    rc, err = 0, None
    try:
        sys.argv = [module, *margs]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        code = e.code
        rc = code if isinstance(code, int) else (0 if code is None else 1)
        if not isinstance(code, (int, type(None))):
            err = str(code)
    except KeyboardInterrupt:
        rc, err = 130, "stopped"
    except BaseException as e:  # noqa: BLE001
        rc, err = 1, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        stop_evt.set()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        _finish(run_dir, rc, err)
    return rc


if __name__ == "__main__":
    sys.exit(main())
