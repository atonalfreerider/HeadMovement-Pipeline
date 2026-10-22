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


def _post_quit(port: int, timeout: float = 5.0) -> bool:
    """Ask the monitor on this port to quit, as its own page would: GET / (the session cookie and the per-instance token
    in the page's meta tag), then POST /api/quit with X-Monitor-Token. Local only; the token is never written anywhere."""
    import http.cookiejar
    import re
    try:
        op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        page = op.open(f"http://127.0.0.1:{port}/", timeout=timeout).read().decode("utf-8", "replace")
        m = re.search(r'name="dancecap-token" content="([^"]+)"', page)
        if not m:
            return False
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/quit", data=b"{}", method="POST",
                                     headers={"X-Monitor-Token": m.group(1), "Content-Type": "application/json"})
        with op.open(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def stop_running(timeout: float = 20.0) -> dict | None:
    """Stop OUR running monitor server (`monitor --launch --replace` could not: the new server waited 20 s for the lock
    and gave up). Only the process named by work/monitor/server.json whose /api/health answers with that file's instance
    id: first its own POST /api/quit; when it has not exited after `timeout`, that one pid (checked again by instance)
    is terminated. Never another process. None when no server runs."""
    from ..util import pid_alive
    sj = _running_server()
    if not sj:
        return None
    pid, port = int(sj["pid"]), int(sj["port"])
    how = "quit" if _post_quit(port) else "quit refused"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not pid_alive(pid):
            _log("launcher", f"stopped server pid {pid} ({how}) in {time.time() - t0:.1f}s")
            return {"pid": pid, "how": how}
        time.sleep(0.25)
    h = _health(port)
    if h and h.get("instance") == sj.get("instance") and int(h.get("pid") or 0) == pid:
        try:
            import psutil
            psutil.Process(pid).terminate()
            psutil.Process(pid).wait(10)
            how = "terminated"
        except Exception as e:  # noqa: BLE001
            how = f"terminate failed: {e}"
    _log("launcher", f"stop server pid {pid}: {how}")
    return {"pid": pid, "how": how}


# ------------------------------------------------------------------------------------------------ one window (2026-10-22)
# The owner saw 7-10 monitor windows open: every `monitor --launch` opened another Edge --app window. There is now one
# window. A launch, or a new job (ensure_running(route=...)), writes work/monitor/focus.json; the open window polls
# /api/focus and follows it (app.js). A launch brings that window forward instead of opening one, and extra monitor
# windows are closed (they are only ever the --app windows of the monitor's own Edge profile).

FOCUS_FILE = "focus.json"
TITLE_SUFFIX = "Pipeline Monitor"   # app.js: document.title = "<view> · Pipeline Monitor"


def set_focus(route: str, reason: str) -> None:
    """Point the open monitor window at `route` (a "#/..." view of this app). Never raises."""
    try:
        if not route.startswith("#/"):
            return
        d = _paths().monitor_dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{FOCUS_FILE}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"route": route, "reason": str(reason)[:200], "ts": time.time()}))
        os.replace(tmp, d / FOCUS_FILE)
    except Exception:  # noqa: BLE001 - the monitor never breaks the work it watches
        pass


def read_focus() -> dict:
    try:
        d = json.loads((_paths().monitor_dir() / FOCUS_FILE).read_text())
        if isinstance(d, dict) and str(d.get("route", "")).startswith("#/"):
            return {"route": d["route"], "reason": d.get("reason"), "ts": float(d.get("ts") or 0)}
    except (OSError, ValueError, TypeError):
        pass
    return {}


def monitor_windows() -> list[int]:
    """Handles of the visible monitor windows (Edge --app windows titled "... Pipeline Monitor"), front-most first. A
    normal browser tab never matches: its window title ends with the browser's name."""
    if os.name != "nt":
        return []
    try:
        import ctypes
        from ctypes import wintypes
        u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
        k32.OpenProcess.restype = wintypes.HANDLE
        found: list[int] = []

        def exe_of(hwnd) -> str:
            pid = wintypes.DWORD()
            u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            hp = k32.OpenProcess(0x1000, False, pid.value)   # PROCESS_QUERY_LIMITED_INFORMATION
            if not hp:
                return ""
            try:
                buf, n = ctypes.create_unicode_buffer(520), wintypes.DWORD(520)
                return buf.value if k32.QueryFullProcessImageNameW(hp, 0, buf, ctypes.byref(n)) else ""
            finally:
                k32.CloseHandle(hp)

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def each(hwnd, _):
            if u32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(512)
                u32.GetWindowTextW(hwnd, buf, 512)
                if buf.value.endswith(TITLE_SUFFIX) and exe_of(hwnd).lower().endswith("msedge.exe"):
                    found.append(int(hwnd))
            return True

        u32.EnumWindows(each, 0)   # z-order: the front-most window first
        return found
    except Exception:  # noqa: BLE001
        return []


def dedupe_windows() -> list[int]:
    """Close every monitor window but the front-most one; returns what is left ([] or [hwnd])."""
    wins = monitor_windows()
    if len(wins) > 1:
        import ctypes
        for hwnd in wins[1:]:
            ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)   # WM_CLOSE: the window closes as if by its X
        _log("launcher", f"closed {len(wins) - 1} extra monitor window(s)")
    return wins[:1]


def _raise_window(hwnd: int) -> None:
    try:
        import ctypes
        u32 = ctypes.windll.user32
        if u32.IsIconic(hwnd):
            u32.ShowWindow(hwnd, 9)   # SW_RESTORE
        u32.SetForegroundWindow(hwnd)   # Windows may only flash the taskbar button when another app has the focus
    except Exception:  # noqa: BLE001
        pass


def show(port: int, route: str, reason: str) -> str:
    """The one-window rule: point the open monitor window at `route` and bring it forward, else open a window."""
    set_focus(route, reason)
    wins = dedupe_windows()
    if wins:
        _raise_window(wins[0])
        return "reused"
    open_window(f"http://127.0.0.1:{port}/{route}")
    return "opened"


def launch(take: str | None = None, view: str | None = None, replace: bool = False, no_window: bool = False,
           dance: str | None = None) -> int:
    route = route_for(take, view, dance)
    if not replace:
        sj = _running_server()
        if sj:
            if not no_window:
                show(sj["port"], route, "launch")
            _log("launcher", f"reuse server pid {sj['pid']} port {sj['port']} -> {route}")
            return 0
    else:
        stop_running()   # --replace: the running server lets go of monitor.lock first
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
                show(sj["port"], route, "launch")   # after --replace the open window reloads itself onto the new server
            _log("launcher", f"started server pid {sj['pid']} port {sj['port']} in {time.time() - t0:.1f}s -> {route}")
            return 0
        time.sleep(0.25)
    _log("launcher", "the server did not answer within 25 s (see work/monitor/logs/monitor.log)")
    return 1


# ------------------------------------------------------------------------------------------------ auto-start (2026-10-21)
# The owner: the dashboard must be "maintained and running every time we analyze anything". Every entry point that
# starts work (`dancecap remote run`, `dancecap.pipeline run`, `dancecap.pipeline analyze`, the farm launcher) calls
# ensure_running(): a live server is reused; else one is started headless (no window, no console). The caller leaves a
# LEASE (work/monitor/leases/<pid>.json): the server does not idle-exit while a lease's process lives, while a pod is
# open in the ledger, or while an analysis runs (watchdog in serve()).

LEASE_DIR = "leases"


def _monitor_python() -> str:
    """The interpreter for a headless server: DANCECAP_MONITOR_PYTHON, else the workspace venv the launcher uses
    (it has flask + waitress), else this one."""
    env = os.environ.get("DANCECAP_MONITOR_PYTHON")
    if env and Path(env).exists():
        return env
    from ..paths import REAL_ROOT
    for cand in (REAL_ROOT.parent / ".venv-nopo4d" / "Scripts" / "python.exe", REAL_ROOT.parent / ".venv-nopo4d" / "bin" / "python"):
        if cand.exists():
            return str(cand)
    return sys.executable


def write_lease(reason: str, pid: int | None = None) -> Path | None:
    """work/monitor/leases/<pid>.json {pid, create_time, reason, ts}: keeps the server up while that process lives."""
    try:
        from ..util import write_json_atomic
        pid = int(pid or os.getpid())
        ct = None
        try:
            import psutil
            ct = psutil.Process(pid).create_time()
        except Exception:  # noqa: BLE001
            pass
        f = _paths().monitor_dir() / LEASE_DIR / f"{pid}.json"
        write_json_atomic(f, {"pid": pid, "create_time": ct, "reason": str(reason)[:120], "ts": time.time()})
        return f
    except Exception:  # noqa: BLE001
        return None


def live_leases(prune: bool = True) -> list[dict]:
    """Leases whose process is alive (dead ones are removed when prune)."""
    from ..util import pid_alive, read_json
    d = _paths().monitor_dir() / LEASE_DIR
    out = []
    if not d.is_dir():
        return out
    for f in d.glob("*.json"):
        info = read_json(f, None)
        ok = isinstance(info, dict) and pid_alive(info.get("pid"), info.get("create_time"))
        if ok:
            out.append(info)
        elif prune:
            try:
                f.unlink()
            except OSError:
                pass
    return out


def activity_hold(now: float | None = None) -> str | None:
    """Why the server must not idle-exit now (None = it may): a live lease, an open ledger row inside its kill time, or a
    running analysis. Cheap: a few small JSON reads."""
    now = now or time.time()
    leases = live_leases()
    if leases:
        return f"lease: {leases[0].get('reason')}"
    try:
        from ..util import read_json
        rows = read_json(_paths().ledger_path(), []) or []
        for r in rows if isinstance(rows, list) else []:
            if not r.get("deleted_ts") and now < float(r.get("created_ts") or 0) + (float(r.get("max_minutes") or 0) + 16) * 60:
                return f"pod {r.get('pod_id')} open"
    except Exception:  # noqa: BLE001
        pass
    try:
        from .. import analyze
        for d in analyze.active_runs():
            if analyze.driver_alive(d) or any(s.get("status") == "running" for s in d.get("steps") or []):
                return f"analysis {d.get('run')} is running"
    except Exception:  # noqa: BLE001
        pass
    return None


def ensure_running(reason: str, lease: bool = True, wait_s: float = 0.0, route: str | None = None) -> dict:
    """Make sure a monitor server runs (headless; the window is the user's choice) and lease it for this process.
    `route` ("#/analysis/<run>", ...) points the open monitor window at this job (the one-window rule) and closes extra
    monitor windows. Never raises, never opens a window or a console, returns in milliseconds when a server is up.
    Skipped in tests (another package root), on pods and with DANCECAP_NO_MONITOR=1."""
    try:
        from .. import paths as P
        if (os.environ.get("DANCECAP_NO_MONITOR") == "1" or os.environ.get("DANCECAP_POD") == "1" or not P.is_real_root()
                or os.environ.get("PYTEST_CURRENT_TEST")):
            return {"state": "skipped"}
        if lease:
            write_lease(reason)
        if route:
            set_focus(route, reason)
            dedupe_windows()
        sj = _running_server()
        if sj:
            return {"state": "running", "port": sj.get("port"), "pid": sj.get("pid")}
        flags = (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
        subprocess.Popen([_monitor_python(), "-m", "dancecap.pipeline", "monitor", "--serve"], cwd=str(P.REAL_ROOT),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags, close_fds=True)
        _log("launcher", f"auto-start for {reason!r} (pid {os.getpid()})")
        t0 = time.time()
        while time.time() - t0 < wait_s:
            sj = _running_server()
            if sj:
                return {"state": "started", "port": sj.get("port"), "pid": sj.get("pid")}
            time.sleep(0.25)
        return {"state": "starting"}
    except Exception as e:  # noqa: BLE001 - the monitor never breaks the work it watches
        return {"state": "error", "error": str(e)[:200]}


# ------------------------------------------------------------------------------------------------ server housekeeping
# (fix round 2, 2026-10-22) Leases were pruned only when the server was idle, which it never was under a polling window:
# 37 dead lease files piled up. Stale ledger rows (closes lost on 2026-10-11) made the budget say "6 open pods" while none
# ran. Volume-farm runs took no snapshots at all. The watchdog below calls these.

def prune_leases() -> int:
    """Remove the lease files of processes that are gone; returns how many live ones are left."""
    return len(live_leases(prune=True))


STALE_GRACE_MIN = 12.0   # remote.OPEN_ROW_GRACE_MIN (10) + 2: a row this far past its kill time has no pod behind it


def stale_ledger_rows(rows: list, now: float) -> list[dict]:
    """Open ledger rows past their pod's kill switch (max_minutes) plus the grace: their close was lost."""
    out = []
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict) and not r.get("deleted_ts") and r.get("max_minutes") and \
                now > float(r.get("created_ts") or now) + (float(r["max_minutes"]) + STALE_GRACE_MIN) * 60:
            out.append(r)
    return out


def reconcile_stale(now: float | None = None, look_gap_s: float = 20.0) -> dict | None:
    """When the ledger holds rows past their kill time, run the repo's own reconcile (remote.reconcile_ledger, as
    `python -m dancecap remote reconcile` does: two looks at Runpod's pod list `look_gap_s` apart; it closes a row only
    when Runpod has not listed its pod both times, and never deletes a pod). Rows younger than the youngest stale one are
    left out (min_age_s), so a fresh pod's row is never judged by this. None when nothing is stale."""
    from ..util import read_json
    now = now or time.time()
    stale = stale_ledger_rows(read_json(_paths().ledger_path(), []) or [], now)
    if not stale:
        return None
    from ... import remote
    youngest = min(now - float(r["created_ts"]) for r in stale)
    r = remote.reconcile_ledger(confirm_s=15, min_age_s=max(300.0, youngest - 1.0))
    if r.get("flagged"):
        time.sleep(look_gap_s)
        r = remote.reconcile_ledger(confirm_s=15, min_age_s=max(300.0, youngest - 1.0))
    return {"stale": [x.get("pod_id") for x in stale], **{k: r.get(k) for k in ("closed", "flagged", "error")}}


FARM_SNAP_STATE = "farm_snaps.json"      # work/monitor/: {dance: {t, run, finished, hb, last}} (survives restarts)
FARM_SNAP_GAP_S = 60.0                   # at most one farm snapshot a minute per dance, except the end
FARM_HEARTBEAT_S = 900.0                 # owner: snapshots at progress points; one every 15 min while chunks run
FARM_LIVE_S = 1800.0                     # a farm whose state.json is older than this is not running: no heartbeat
# which farm milestone a batch's snapshot shows when several land in one tick (lower wins), by classify()'s event name
FARM_EVENT_RANK = {"end": 0, "start": 0, "failed": 1, "error": 1, "chunk_failed": 1, "launch_failed": 1, "pull_failed": 2,
                   "plan": 2, "launch": 3, "chunk_pulled": 4, "chunk_done": 5}


class FarmSnapper:
    """Snapshot hooks for volume-farm runs, on the monitor side (the farm code has no hook point the dashboard owns):
    every `tick` reads the farm/state.json of dances whose state changed, and asks snapshots.request() for one image when
    a run starts (plan / first launch after a finish), a pod is launched, a chunk is pulled or ends, the run fails or
    finishes, and every 15 min while chunks run. A run's snapshots go to work/pipeline/snapshots/<start>-farm_<dance>/.
    The first sight of a dance (a server start) only records where its events are: history is never re-snapshotted."""

    def __init__(self):
        from ..util import read_json
        self.path = _paths().monitor_dir() / FARM_SNAP_STATE
        d = read_json(self.path, {}) or {}
        self.seen: dict[str, dict] = d if isinstance(d, dict) else {}
        self.mt: dict[str, float] = {}

    def _save(self) -> None:
        try:
            from ..util import write_json_atomic
            write_json_atomic(self.path, self.seen)
        except OSError:
            pass

    @staticmethod
    def classify(e: dict) -> str | None:
        """Snapshot event name of one farm event, None when it is not a progress point. The farm writes a chunk's status
        changes and failures as kind "chunk" (watch.py add_event(s, "chunk" if c is not None else "job", ...)) and the
        jobs that are not chunks as "job": 'jlg0000: running -> done', 'jlsynth failed on <pod> (...); re-queued 1/2',
        'cag0000 FAILED for good: ...'. A pull or launch can report its own failure ('pull attempt 1 failed', 'launch
        failed'): never a 'pulled' milestone, and a failed launch never starts a run."""
        k, msg = str(e.get("kind") or ""), str(e.get("msg") or "")
        if k in ("chunk", "job"):
            if "-> done" in msg or "-> pulled" in msg:
                return "chunk_done"
            if "-> failed" in msg or " failed on " in msg or "FAILED" in msg:
                return "chunk_failed"
            return None
        if k == "pull":
            return "pull_failed" if "failed" in msg.lower() else "chunk_pulled"
        if k == "launch":
            return "launch_failed" if "failed" in msg.lower() else "launch"
        return {"finish": "end", "plan": "plan", "fail": "failed", "error": "error"}.get(k)

    def tick(self, now: float | None = None, request=None) -> list[tuple[str, str, str]]:
        """-> [(run, event, label)] requested this tick (`request` = snapshots.request; injectable for tests)."""
        from ..util import TAKE_RE, mtime, read_json
        from .. import snapshots
        now = now or time.time()
        request = request or snapshots.request
        out = []
        try:
            files = list(_paths().work().glob("*/farm/state.json"))
        except OSError:
            return out
        changed = False
        for sp in files:
            dance = sp.parents[1].name
            mt = mtime(sp)
            if mt is None or not TAKE_RE.match(dance) or self.mt.get(dance) == mt and not self._heartbeat_due(dance, now):
                continue
            self.mt[dance] = mt
            st = read_json(sp, None)
            if not isinstance(st, dict) or st.get("demo"):
                continue
            evs = [e for e in st.get("events") or [] if isinstance(e, dict) and isinstance(e.get("t"), (int, float))]
            last_t = max((float(e["t"]) for e in evs), default=0.0)
            rec = self.seen.get(dance)
            if rec is None:   # first sight: a baseline; a run in flight gets its id from its last plan / launch
                starts = [e for e in evs if e.get("kind") in ("plan", "launch")]
                ends = [e for e in evs if e.get("kind") == "finish"]
                # an unfinished run whose state has not changed for 6 h is an abandoned one (larissa_kadu_full, 2026-10-08)
                running = starts and (not ends or float(starts[-1]["t"]) > float(ends[-1]["t"])) and now - mt < 6 * 3600
                first = next((e for e in starts if not ends or float(e["t"]) > float(ends[-1]["t"])), None)
                self.seen[dance] = {"t": last_t, "run": snapshots.farm_run_id(dance, first["t"]) if running and first
                                    else None, "finished": not running, "hb": now, "last": 0.0}
                changed = True
                continue
            new = sorted((e for e in evs if float(e["t"]) > float(rec.get("t") or 0)), key=lambda e: float(e["t"]))
            pick = tuple(rec["pending"]) if rec.get("pending") else None   # a milestone the 60-s gap held back
            if new and float(new[0]["t"]) - float(rec.get("t") or 0) > 6 * 3600:
                rec["finished"] = True     # 6 h without an event: that run was abandoned, the next plan / launch is new
            if not new and not rec.get("finished") and now - mt > 6 * 3600:
                # a record an earlier server left unfinished for a farm silent for 6 h (larissa_kadu_full, state.json of
                # 2026-10-08): abandoned, so no heartbeat is due and its state is not re-read every tick
                rec.update(finished=True, run=None)
                rec.pop("pending", None)
                changed = True
                continue
            for e in new:
                ev = self.classify(e)
                if ev is None:
                    continue
                if ev in ("plan", "launch") and (rec.get("run") is None or rec.get("finished")):
                    rec["run"], rec["finished"] = snapshots.farm_run_id(dance, e["t"]), False
                    ev = "start"
                if ev == "end":
                    rec["finished"] = True
                rank = FARM_EVENT_RANK.get(ev, 9)
                if pick is None or rank <= pick[0]:
                    pick = (rank, ev, str(e.get("msg") or "")[:150])
            if new:
                rec["t"] = max(float(e["t"]) for e in new)
                changed = True
            run = rec.get("run")
            if run and pick and (pick[1] in ("end", "start", "failed") or now - float(rec.get("last") or 0) >= FARM_SNAP_GAP_S):
                if request(run, f"farm_{pick[1]}", f"farm {dance}: {pick[2]}", final=pick[1] == "end"):
                    out.append((run, pick[1], pick[2]))
                rec["last"] = rec["hb"] = now
                rec.pop("pending", None)
                changed = True
            elif run and pick:
                rec["pending"] = list(pick)    # inside the gap: taken on a later tick, never dropped
                self.mt.pop(dance, None)       # look again next tick even if the file does not change
                changed = True
            elif run and not rec.get("finished") and self._heartbeat_due(dance, now) and self._chunks_running(st) and \
                    now - mt < FARM_LIVE_S:   # the watcher writes state.json every minute while a farm runs
                if request(run, "farm_heartbeat", f"farm {dance}: 15-min heartbeat"):
                    out.append((run, "heartbeat", ""))
                rec["last"] = rec["hb"] = now
                changed = True
        if changed or out:
            self._save()
        return out

    def _heartbeat_due(self, dance: str, now: float) -> bool:
        rec = self.seen.get(dance) or {}
        return bool(rec.get("run")) and not rec.get("finished") and now - float(rec.get("hb") or 0) >= FARM_HEARTBEAT_S

    @staticmethod
    def _chunks_running(st: dict) -> bool:
        return any(isinstance(c, dict) and c.get("status") in ("running", "starting", "queued")
                   for c in (st.get("chunks") or {}).values())


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

    def reconcile_job():
        try:
            r = reconcile_stale()
            if r:
                print(f"[monitor] ledger reconcile: {len(r['stale'])} row(s) past their kill time; closed {r.get('closed')}"
                      f"{' (' + str(r['error']) + ')' if r.get('error') else ''}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[monitor] ledger reconcile: {type(e).__name__}: {e}", flush=True)

    snapper = None
    try:
        snapper = FarmSnapper()
    except Exception as e:  # noqa: BLE001
        print(f"[monitor] farm snapshots off: {e}", flush=True)

    def watchdog():
        last_pods = 0.0
        last_house = 0.0
        last_tick = time.time()
        last_hold_check, last_hold = 0.0, None
        last_leases = 0.0
        last_farm = 0.0
        last_reconcile = time.time() - 30 * 60 + 120   # the first look 2 min after the start, then every 30 min
        while True:
            time.sleep(1.0)
            now = time.time()
            if now - last_leases > 600:
                last_leases = now
                try:
                    prune_leases()
                except Exception as e:  # noqa: BLE001
                    print(f"[monitor] lease prune: {e}", flush=True)
            if snapper is not None and now - last_farm > 30:
                last_farm = now
                try:
                    for run, ev, _ in snapper.tick(now):
                        print(f"[monitor] farm snapshot {run}: {ev}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[monitor] farm snapshots: {type(e).__name__}: {e}", flush=True)
            if now - last_reconcile > 30 * 60 and m.poller.connection in ("ok", "unknown"):
                last_reconcile = now
                threading.Thread(target=reconcile_job, daemon=True, name="monitor-reconcile").start()
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
                    # no new window: the open windows notice the new instance and reload themselves (app.js), and an
                    # agent's /api/restart must not pop a window on the owner's desktop
                    subprocess.Popen([sys.executable, "-m", "dancecap.pipeline", "monitor", "--launch", "--replace",
                                      "--no-window"],
                                     cwd=str(paths.REAL_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     creationflags=(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0)
                print("[monitor] quit", flush=True)
                os._exit(0)
            if idle > 0 and now - m.last_request > idle * 60:
                if now - last_hold_check > 30:
                    last_hold_check = now
                    try:
                        hold = activity_hold(now)
                    except Exception as e:  # noqa: BLE001
                        hold = None
                        print(f"[monitor] activity check: {e}", flush=True)
                    if hold and hold != last_hold:
                        print(f"[monitor] idle, but staying up: {hold}", flush=True)
                    last_hold = hold
                if not last_hold:
                    print(f"[monitor] idle for {idle:g} min and nothing is running: exit (runs keep going)", flush=True)
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
