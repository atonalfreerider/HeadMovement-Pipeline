"""Pipeline Monitor: `serve()` (the waitress server) and `launch()` (double-click entry: reuse or start the server,
then open a window). docs/PIPELINE_MONITOR.md 3.

    python -m dancecap.pipeline monitor --launch [--take T --view videos|overview|runpod|new]
    python -m dancecap.pipeline monitor --serve [--port 8790] [--replace]

- The server binds 127.0.0.1 only (first free port of 8790..8799), runs as python.exe (never pythonw: console
  grandchildren such as runpodctl / ssh / ffmpeg would pop windows), writes its output to
  work/monitor/logs/monitor.log, holds work/monitor/monitor.lock for its lifetime (single instance; the OS releases
  the lock when the process dies), writes work/monitor/server.json {pid, port, instance, started, version} (never
  the token) and exits after `idle_exit_min` (20) minutes without a request. Runs are separate processes: closing
  the window or the server never stops a stage or a pod session.
- The window is a Microsoft Edge --app window with its own profile (%LOCALAPPDATA%\\dancecap\\monitor-edge, which keeps
  the microphone grant of the narration tool); the default browser is the fallback.
This module imports only the standard library at import time (fast `--launch`).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
PORTS = range(8790, 8800)
VIEWS = {"videos": "#/take/{take}/videos", "overview": "#/take/{take}", "runpod": "#/runpod", "new": "#/new", "farm": "#/"}


def _paths():
    from .. import paths
    return paths


def _log(name: str, msg: str) -> None:
    try:
        d = _paths().monitor_dir() / "logs"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{name}.log", "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def edge_path() -> str | None:
    try:
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe") as k:
                    v, _ = winreg.QueryValueEx(k, "")
                    if v and Path(v).exists():
                        return v
            except OSError:
                continue
    except ImportError:
        pass
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            p = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if p.exists():
                return str(p)
    return None


def open_window(url: str, shell: str = "edge-app") -> None:
    """An Edge --app window (own profile) or, as a fallback, the default browser. Never an arbitrary URL: callers
    pass our own origin."""
    if not url.startswith("http://127.0.0.1:"):
        raise ValueError("only local monitor URLs")
    edge = edge_path() if shell == "edge-app" else None
    if edge:
        prof = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "dancecap" / "monitor-edge"
        # the app talks to 127.0.0.1 only; these keep the browser's own background traffic (sync, component and
        # field-trial updates, pings, reliability reports) from leaving the PC on behalf of this private profile
        subprocess.Popen([edge, f"--app={url}", f"--user-data-dir={prof}", "--no-first-run",
                          "--no-default-browser-check", "--window-size=1600,1000",
                          "--disable-background-networking", "--disable-sync", "--disable-component-update",
                          "--no-pings", "--disable-domain-reliability"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0, close_fds=True)
        return
    import webbrowser
    webbrowser.open(url)


def _health(port: int, timeout: float = 1.0) -> dict | None:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/health")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _running_server() -> dict | None:
    from ..util import pid_alive, read_json
    sj = read_json(_paths().monitor_dir() / "server.json", None)
    if not sj or not pid_alive(sj.get("pid")):
        return None
    h = _health(int(sj.get("port", 0)))
    if h and h.get("instance") == sj.get("instance"):
        return sj
    return None


def route_for(take: str | None = None, view: str | None = None, dance: str | None = None) -> str:
    from ..util import TAKE_RE
    if dance and (view in (None, "farm")) and TAKE_RE.match(dance):
        return f"#/farm/{dance}"
    if view in VIEWS and (take or "{take}" not in VIEWS[view]):
        if take and not TAKE_RE.match(take):
            return "#/"
        return VIEWS[view].replace("{take}", take or "")
    if take and TAKE_RE.match(take):
        return f"#/take/{take}"
    return "#/"


def launch(take: str | None = None, view: str | None = None, replace: bool = False, no_window: bool = False,
           dance: str | None = None) -> int:
    route = route_for(take, view, dance)
    if not replace:
        sj = _running_server()
        if sj:
            if not no_window:
                open_window(f"http://127.0.0.1:{sj['port']}/{route}")
            _log("launcher", f"reuse server pid {sj['pid']} port {sj['port']} -> {route}")
            return 0
    args = [sys.executable, "-m", "dancecap.pipeline", "monitor", "--serve"] + (["--replace"] if replace else [])
    flags = (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    from ..paths import REAL_ROOT
    subprocess.Popen(args, cwd=str(REAL_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    t0 = time.time()
    while time.time() - t0 < 25:
        sj = _running_server()
        if sj and (not replace or sj.get("started", 0) >= t0 - 1):
            if not no_window:
                open_window(f"http://127.0.0.1:{sj['port']}/{route}")
            _log("launcher", f"started server pid {sj['pid']} port {sj['port']} in {time.time() - t0:.1f}s -> {route}")
            return 0
        time.sleep(0.25)
    _log("launcher", "the server did not answer within 25 s (see work/monitor/logs/monitor.log)")
    return 1


class _Lock:
    """work/monitor/monitor.lock held for the process lifetime (msvcrt / fcntl); released by the OS on exit."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            self.fh.close()
            self.fh = None
            return False


def _free_port(ports=PORTS) -> int | None:
    for p in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
        finally:
            s.close()
    return None


def _redirect_output() -> None:
    d = _paths().monitor_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    log = d / "monitor.log"
    try:
        if log.exists() and log.stat().st_size > 2 << 20:
            for i in range(4, 0, -1):
                src = d / f"monitor.log.{i}"
                if src.exists():
                    os.replace(src, d / f"monitor.log.{i + 1}")
            os.replace(log, d / "monitor.log.1")
    except OSError:
        pass
    fh = open(log, "a", encoding="utf-8", buffering=1, errors="replace")
    sys.stdout = fh
    sys.stderr = fh


def serve(port: int | None = None, replace: bool = False, idle_exit_min: float | None = None,
          redirect: bool = True) -> int:
    if redirect:
        _redirect_output()
    paths = _paths()
    lock = _Lock(paths.monitor_dir() / "monitor.lock")
    t0 = time.time()
    while not lock.acquire():
        if not replace or time.time() - t0 > 20:
            print("[monitor] another monitor holds work/monitor/monitor.lock", flush=True)
            return 3
        time.sleep(0.5)
    port = port or _free_port()
    if port is None:
        print("[monitor] no free port in 8790-8799", flush=True)
        return 1
    from .server import VERSION, Monitor, _static_rev, create_app
    from ..util import write_json_atomic
    m = Monitor(port=port)
    app = create_app(m)
    idle = float(idle_exit_min if idle_exit_min is not None else m.settings.get("idle_exit_min", 20))
    write_json_atomic(paths.monitor_dir() / "server.json",
                      {"pid": os.getpid(), "port": port, "instance": m.instance, "started": m.started,
                       "version": VERSION, "static_rev": _static_rev()})
    print(f"[monitor] serving http://127.0.0.1:{port}/ (pid {os.getpid()}, idle exit {idle:g} min)", flush=True)

    def housekeeping():
        """Old finished runs go to work/pipeline/runs_archive/ (the status reads every run of a take: keep it fast)."""
        try:
            from .. import runs as _runs
            n = _runs.archive()
            if n:
                print(f"[monitor] archived {n} finished run(s) older than 14 days", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[monitor] run archive: {e}", flush=True)

    def watchdog():
        last_pods = 0.0
        last_house = 0.0
        last_tick = time.time()
        while True:
            time.sleep(1.0)
            now = time.time()
            if now - last_tick > 60:
                # the loop sleeps 1 s: a gap this long means the PC slept (or the process was suspended). Idle time is
                # wall time, so without this the server would exit the moment the PC wakes, under an open window.
                print(f"[monitor] resumed after {now - last_tick:.0f} s asleep: idle timer restarted", flush=True)
                m.last_request = now
            last_tick = now
            if now - last_house > 6 * 3600:
                last_house = now
                threading.Thread(target=housekeeping, daemon=True, name="monitor-archive").start()
            if m.quit.is_set():
                if m.restart.is_set():
                    subprocess.Popen([sys.executable, "-m", "dancecap.pipeline", "monitor", "--launch", "--replace"],
                                     cwd=str(paths.REAL_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     creationflags=(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0)
                print("[monitor] quit", flush=True)
                os._exit(0)
            if idle > 0 and now - m.last_request > idle * 60:
                print(f"[monitor] idle for {idle:g} min: exit (runs keep going)", flush=True)
                os._exit(0)
            busy = bool(m.poller.pods)  # pods listed: poll every 30 s; else every 5 min
            if now - last_pods > (30 if busy else 300):
                last_pods = now
                try:
                    m.poller.refresh(force=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[monitor] pod poll: {e}", flush=True)

    threading.Thread(target=watchdog, daemon=True, name="monitor-watchdog").start()
    import waitress
    waitress.serve(app, host="127.0.0.1", port=port, threads=16, channel_timeout=120, ident="dancecap-monitor",
                   _quiet=True)
    return 0
