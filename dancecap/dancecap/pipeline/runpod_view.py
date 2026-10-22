"""Runpod panel data, read-only: pods (runpodctl through remote._ctl), the ledger budget, and the remote_jobs
sessions under work/runpod/sessions (progress level L0: runner.log + state.json, PIPELINE_PROGRESS 4.2).

Never creates, stops or deletes anything. The API key stays inside remote._ctl's child environment; pod JSON is
reduced to allowlisted fields; runpodctl error text is redacted. Files are opened, read and closed at once (a held
handle would break the runner's writes on Windows).
"""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime
from pathlib import Path

from . import estimate, paths, registry
from .util import SID_RE, mtime, read_json, redact

POD_FIELDS = ("id", "name", "desiredStatus", "costPerHr", "lastStartedAt")
_STEP_START = re.compile(r"^\[pod-runner (\d\d:\d\d:\d\d)\] (\S+) \.\.\.\s*$")
_STEP_END = re.compile(r"^\[pod-runner (\d\d:\d\d:\d\d)\] (\S+) (done|FAILED) in ([\d.]+)s")
_JOB_END = re.compile(r"^\[pod-runner (\d\d:\d\d:\d\d)\] (\S+): (ok|failed|FAILED)\b")
_TEXTURE = re.compile(r"^\[texture\] (\d+)/(\d+) views")
_TEXPASS = re.compile(r"^\[texture\] pass (\d+)")
_LOC = re.compile(r"^\[cameras\] (\w+): (\d+)/(\d+) frames")
_FRONT = re.compile(r"^\[front\] (?:\d\d:\d\d:\d\d )?start (\S+)|^\[front\] \S+ (\S+): ok")
_ROOM = re.compile(r"^\[room\] step\s+(\d+)")
_SPLATS = re.compile(r"^\[dancer_splats\] step (\d+)/(\d+) stage (S\d) psnr ([\d.]+)")
_SPLATS_EVAL = re.compile(r"^\[dancer_splats\] eval (\d+)/(\d+) views")
_CHUNKS = re.compile(r"^\[chunks\] (\S+): rc=(\d+)")
_TERMINAL = re.compile(r"\] (dry run: no pod|nothing to do)\s*$")
_TASK_FAILED = re.compile(r"^\[front\] (?:\d\d:\d\d:\d\d )?(\S+): failed\b")
_TASK_SKIPPED = re.compile(r"^\[front\] (?:\d\d:\d\d:\d\d )?(\S+): skipped \(dependency failed")


def task_failures(text: str) -> dict:
    """Failed and skipped tasks of a pod job (front_pod's task runner lines in runner.log): {failed, skipped, first}.
    A job whose early tasks fail (e.g. ingest) skips everything after them and ends without useful outputs, while
    its pod keeps billing until then: the monitor shows it as a warning and an alert."""
    failed, skipped, first = [], 0, None
    for line in text.splitlines():
        line = line.rstrip("\r")
        m = _TASK_FAILED.match(line)
        if m:
            failed.append(m.group(1).rstrip(":"))
            first = first or line.split("] ", 1)[-1][:160]
            continue
        if _TASK_SKIPPED.match(line):
            skipped += 1
    return {"failed": len(failed), "skipped": skipped, "first": first, "tasks": failed[:12]}


def _with_failures(out: dict, text: str) -> dict:
    sstate = (out.get("summary") or {}).get("state")
    if sstate not in ("running", "preparing", None):
        out["progress"] = None  # ended: the last progress line of the log is not 'now' any more
        out["current"] = None
    if sstate == "orphaned":
        out["eta"] = None  # no runner: an ETA from its last phase ("pack ETA 5 h 59") is meaningless
        out["warnings"] = ["the runner process of this session is gone while its pod is still open in the ledger: "
                           "nothing will download its outputs; the pod bills until its kill switch (or delete it)"] + \
            list(out.get("warnings") or [])
    bad = (out.get("summary") or {}).get("jobs_bad") or {}
    if sstate == "failed" and bad and not out.get("error_summary"):
        out["error_summary"] = "; ".join(f"{k}: {v}" for k, v in list(bad.items())[:4])[:300]
    tf = task_failures(text)
    out["task_failures"] = tf
    if tf["failed"]:
        msg = (f"{tf['failed']} pod task(s) failed and {tf['skipped']} were skipped after them (first: {tf['first']}): "
               "the job may end without useful outputs while the pod bills - read the runner log")
        out["warnings"] = [msg] + list(out.get("warnings") or [])
    return out


# ------------------------------------------------------------------------------------------------ sessions

def _sid_date(sid: str) -> datetime | None:
    try:
        return datetime.strptime(sid[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def parse_runner_log(text: str) -> dict:
    """L0: runner phases, the open step, and the last progress line we recognise."""
    steps, open_steps, prog, jobs_done = [], {}, None, []
    tex_pass = 0
    for line in text.splitlines():
        line = line.rstrip("\r")
        m = _STEP_START.match(line)
        if m:
            open_steps[m.group(2)] = m.group(1)
            continue
        m = _STEP_END.match(line)
        if m:
            steps.append({"step": m.group(2), "ok": m.group(3) == "done", "seconds": float(m.group(4)),
                          "end_hms": m.group(1), "start_hms": open_steps.pop(m.group(2), None)})
            continue
        m = _JOB_END.match(line)
        if m:
            jobs_done.append({"job": m.group(2), "status": m.group(3).lower()})
            continue
        m = _TEXPASS.match(line)
        if m:
            tex_pass = int(m.group(1)) + 1
            continue
        m = _TEXTURE.match(line)
        if m:
            prog = {"step": f"bake/pass{tex_pass}", "done": int(m.group(1)), "total": int(m.group(2)), "unit": "views"}
            continue
        m = _LOC.match(line)
        if m:
            prog = {"step": f"front/loc:{m.group(1)}", "done": int(m.group(2)), "total": int(m.group(3)), "unit": "frames"}
            continue
        m = _ROOM.match(line)
        if m:
            prog = {"step": "room/train", "done": int(m.group(1)), "total": 30000, "unit": "steps"}
            continue
        m = _SPLATS.match(line)
        if m:
            prog = {"step": f"dancer_splats/train:{m.group(3)}", "done": int(m.group(1)), "total": int(m.group(2)),
                    "unit": "steps", "msg": f"PSNR {m.group(4)}"}
            continue
        m = _SPLATS_EVAL.match(line)
        if m:
            prog = {"step": "dancer_splats/eval", "done": int(m.group(1)), "total": int(m.group(2)), "unit": "views"}
            continue
        m = _FRONT.match(line)
        if m:
            prog = {"step": f"front/{m.group(1) or m.group(2)}", "done": None, "total": None, "unit": "tasks"}
    current = list(open_steps.items())[-1] if open_steps else None
    terminal = bool(_TERMINAL.search(text.rstrip().splitlines()[-1])) if text.strip() else False
    return {"steps": steps, "current": {"step": current[0], "start_hms": current[1]} if current else None,
            "progress": prog, "jobs_done": jobs_done, "terminal": terminal}


def _farm_managed(pod_id: str) -> bool:
    """A pod recorded (not deleted / lost) in a farm dance's state.json whose watcher is alive belongs to the farm (the Farm view shows it
    with its job and kill switch): it is not an orphan of a fit_full session. (Found by the farm drill: every farm pod was flagged.)"""
    try:
        import json
        for sp in (paths.REAL_ROOT / "work").glob("*/farm/state.json"):
            rec = ((json.loads(sp.read_text(encoding="utf-8")).get("pods") or {}).get(pod_id)) or {}
            if rec and rec.get("state") not in ("deleted", "gone", "lost"):
                from ..farm import watch
                if watch.health(sp.parents[1].name).get("running"):
                    return True
    except Exception:  # noqa: BLE001  a broken state file never hides a real orphan
        pass
    return False


def _runner_processes() -> int:
    """Local `python -m dancecap remote run` / *_pod run processes (argv only; never environments). hands_pod is
    retired from the pipeline (2026-10-07) but still creates a pod when run by hand, so it still counts here."""
    try:
        import psutil
    except ImportError:
        return 0
    n = 0
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
        except Exception:
            continue
        if "dancecap" in cl and ((" remote run" in cl) or re.search(r"(hands_pod|face_pod) run", cl)):
            n += 1
    return n


_sum_cache: dict[str, tuple[tuple, dict]] = {}


def session_summary(sid: str, ledger: list[dict] | None = None) -> dict | None:
    if not SID_RE.match(sid or ""):
        return None
    d = paths.sessions_dir() / sid
    key = (mtime(d / "state.json"), mtime(d / "runner.log"), mtime(d / "result.json"), mtime(paths.ledger_path()),
           int(time.time() // 60))
    c = _sum_cache.get(str(d))
    if c and c[0] == key:
        return c[1]
    out = _session_summary(sid, d, ledger)
    if out is not None:
        _sum_cache[str(d)] = (key, out)
    return out


def _session_summary(sid: str, d: Path, ledger: list[dict] | None) -> dict | None:
    st = read_json(d / "state.json", None)
    if st is None:
        return None
    ledger = estimate.ledger_rows() if ledger is None else ledger
    pod = st.get("pod") or {}
    row = next((r for r in ledger if pod.get("id") and r.get("pod_id") == pod.get("id")), None)
    res_m, log_m, st_m = mtime(d / "result.json"), mtime(d / "runner.log"), mtime(d / "state.json")
    newest = max(x for x in (log_m, st_m, 0) if x is not None)
    finished = res_m is not None and res_m >= newest - 1
    jobs = st.get("jobs") or {}
    take = next((j.get("spec", {}).get("take") for j in jobs.values() if j.get("spec", {}).get("take")), None)
    now = time.time()
    bad_jobs = {k: str(v.get("status") or "pending") for k, v in jobs.items() if str(v.get("status") or "") != "ok"}
    runner_alive = runner_alive_of(d)
    if row is not None and not row.get("deleted_ts"):
        # an open pod whose runner process is gone: nobody downloads its outputs or deletes it (its kill switch will)
        state = "orphaned" if runner_alive is False and not finished else "running"
    elif finished:
        if not pod and not st.get("error"):
            state = "dry_run"
        else:
            # a job "skipped: bootstrap feature failed" or still "pending" in an ended session is NOT ok (2026-10-21:
            # the dashboard probe's skipped job was shown as a green ok session)
            state = "failed" if st.get("error") or bad_jobs else "ok"
    elif not pod and not st.get("error") and (runner_alive or newest and now - newest < 300 and _is_newest(d)):
        # bundling / starting its pod: also a session whose runner still lives though it is not the newest dir (every
        # -p1..p(n-1) session of `remote run --parallel`, shown as "unknown" and then "no pod" before, review 2026-10-21)
        state = "preparing"
    elif row is not None and row.get("deleted_ts"):
        state = "ended"
    else:
        state = "unknown"
    cost = None
    if row is not None:
        cost = row.get("cost") if row.get("deleted_ts") else round(
            float(row.get("cost_per_hr", 0)) * (now - float(row.get("created_ts", now))) / 3600, 4)
    started = (st.get("timings") or [{}])[0].get("start") if st.get("timings") else None
    return {"sid": sid, "take": take, "label": sid[16:] if len(sid) > 16 else sid, "origin": "external",
            "state": state, "started": started or (_sid_date(sid).timestamp() if _sid_date(sid) else None),
            # a session whose runner died (no result.json) ended when its pod was deleted: the Older-tools table kept
            # counting its duration (2 h 27 for a 2-min session)
            "ended": res_m if finished else (float(row["deleted_ts"]) if state == "ended" and row is not None else None),
            "pod_id": pod.get("id"), "gpu": pod.get("gpu"),
            "cost_per_hr": pod.get("cost_per_hr"), "cost_usd": cost, "jobs": sorted(jobs),
            "jobs_status": {k: v.get("status") for k, v in jobs.items()}, "progress_level": "L0",
            "jobs_bad": bad_jobs if finished or state not in ("running", "preparing") else {},
            "runner_alive": runner_alive, "analysis": st.get("analysis"),
            "max_minutes": st.get("max_minutes"), "error": redact(st.get("error") or "") or None}


def runner_alive_of(d: Path) -> bool | None:
    """Whether the `remote run` process of a session dir is alive (runner.json {pid, started}); None when unknown."""
    info = read_json(d / "runner.json", None)
    if not isinstance(info, dict) or not info.get("pid"):
        return None
    try:
        import psutil
        pr = psutil.Process(int(info["pid"]))
        # pids are reused: the runner wrote runner.json after it started, so its process must be older than that
        return bool(pr.is_running() and pr.create_time() <= float(info.get("started") or 0) + 5.0)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001 - NoSuchProcess / AccessDenied
        return False


def _is_newest(d: Path) -> bool:
    try:
        names = sorted(p.name for p in d.parent.iterdir() if SID_RE.match(p.name))
    except OSError:
        return False
    return bool(names) and names[-1] == d.name


def sessions(take: str | None = None, limit: int = 50) -> list[dict]:
    d = paths.sessions_dir()
    if not d.exists():
        return []
    led = estimate.ledger_rows()
    out = []
    for p in sorted(d.iterdir(), key=lambda x: x.name, reverse=True):
        s = session_summary(p.name, led)
        if not s or (take and s.get("take") != take):
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def active_sessions(take: str | None = None, recent: int = 8) -> list[dict]:
    """Sessions that may be running (open ledger row, or still bundling): only the newest dirs are looked at."""
    d = paths.sessions_dir()
    if not d.exists():
        return []
    led = estimate.ledger_rows()
    open_ids = {r.get("pod_id") for r in estimate.open_rows(led)}
    names = sorted((p.name for p in d.iterdir() if SID_RE.match(p.name)), reverse=True)[:recent]
    out = []
    for n in names:
        s = session_summary(n, led)
        if s and s["state"] in ("running", "preparing", "orphaned") and (not take or s.get("take") == take):
            out.append(s)
    if open_ids - {s.get("pod_id") for s in out}:  # an open pod in an older session dir: look further
        for n in sorted((p.name for p in d.iterdir() if SID_RE.match(p.name)), reverse=True)[recent:]:
            s = session_summary(n, led)
            if s and s.get("pod_id") in open_ids and (not take or s.get("take") == take):
                out.append(s)
    return out


_snap_cache: dict[str, tuple[tuple, float, dict]] = {}


def _progress_eta():
    """dancecap.progress_eta (PIPELINE_PROGRESS 10.1) when present and the backend reads the real tree."""
    if not paths.is_real_root():
        return None
    try:
        from .. import progress_eta  # type: ignore[attr-defined]
        return progress_eta
    except Exception:
        return None


def session_snapshot(sid: str) -> dict | None:
    """SessionSnapshot: progress_eta.snapshot() when available (phases with fractions, p10-p90 ETA, kill margin,
    cost), else the L0 parse below. Cached for a few seconds per (runner.log, state.json, ledger) mtime."""
    s = session_summary(sid)
    if s is None:
        return None
    d = paths.sessions_dir() / sid
    key = (mtime(d / "runner.log"), mtime(d / "state.json"), mtime(paths.ledger_path()))
    c = _snap_cache.get(sid)
    if c and c[0] == key and time.time() - c[1] < 5:
        return c[2]
    out = _session_snapshot(sid, s, d)
    _snap_cache[sid] = (key, time.time(), out)
    return out


def _session_snapshot(sid: str, s: dict, d: Path) -> dict:
    st = read_json(d / "state.json", {}) or {}
    text = estimate.read_log_tail(d / "runner.log", 1 << 20)
    parsed = parse_runner_log(text)
    pe = _progress_eta()
    if pe is not None:
        try:
            # the live Tracker (receive-time stamps for timestamp-less L0 lines, background-polled) when running
            snap = pe.live_snapshot(d) if s.get("state") in ("running", "preparing") and \
                hasattr(pe, "live_snapshot") else pe.snapshot(d)
            run = [p for p in snap.get("phases") or [] if p.get("status") == "running"]
            prog = parsed["progress"]
            if run:
                ph = run[0]
                prog = dict(prog or {}, step=ph.get("name"), frac=ph.get("fraction"), msg=ph.get("detail"))
            eta = snap.get("eta")
            out = dict(snap)
            out["dir"] = paths.rel_display(d)  # never an absolute path in an API response
            out.update({"sid": sid, "take": s["take"], "origin": "external", "summary": s, "progress": prog,
                        "current": {"step": snap.get("current")} if isinstance(snap.get("current"), str) else
                        snap.get("current"), "eta": ({"p10": eta.get("p10"), "p50": eta.get("p50"),
                                                      "p90": eta.get("p90"), "basis": f"progress_eta {snap.get('level')}",
                                                      "branches": eta.get("branches")} if eta else None),
                        "kill_at": (snap.get("kill") or {}).get("kill_at"),
                        "kill_margin": (snap.get("kill") or {}).get("margin_p90_s"), "progress_level": snap.get("level"),
                        "runner_alive": None, "source": "progress_eta",
                        "warnings": [w.get("msg") if isinstance(w, dict) else w for w in snap.get("warnings") or []],
                        "stop_modes": ["delete_pod"] if s["state"] == "running" and s.get("pod_id") else []})
            return _with_failures(out, text)
        except Exception:
            pass
    pod = st.get("pod") or {}
    kill_at = None
    if pod.get("started") and (pod.get("killswitch") or {}).get("after_min"):
        kill_at = float(pod["started"]) + float(pod["killswitch"]["after_min"]) * 60
    phases = [{"step": t.get("step"), "start": t.get("start"), "seconds": t.get("seconds"), "ok": t.get("ok")}
              for t in st.get("timings") or []]
    est_min = st.get("estimate_min") or {}
    elapsed = time.time() - float(pod["started"]) if pod.get("started") and s["state"] == "running" else None
    eta = None
    if elapsed is not None and est_min.get("total"):
        rem = max(0.0, float(est_min["total"]) * 60 - elapsed)
        eta = {"p10": round(rem * 0.6), "p50": round(rem), "p90": round(rem * 1.6), "basis": "session estimate - elapsed"}
    price = float(pod.get("cost_per_hr") or 0)
    cost = {"so_far": s.get("cost_usd"), "p50": None, "p90": None,
            "worst": round(price * max(0.0, (kill_at or 0) - time.time()) / 3600 + (s.get("cost_usd") or 0), 2)
            if kill_at and s["state"] == "running" else s.get("cost_usd")}
    if eta:
        cost["p50"] = round((s.get("cost_usd") or 0) + price * eta["p50"] / 3600, 2)
        cost["p90"] = round((s.get("cost_usd") or 0) + price * eta["p90"] / 3600, 2)
    warnings = []
    if pod and not (pod.get("killswitch") or {}).get("armed", True):
        warnings.append("kill switch not armed: the local delete is the only guard")
    margin = (kill_at - time.time() - (eta or {}).get("p90", 0)) if kill_at and eta else None
    if margin is not None and margin < 0:
        warnings.append("the kill switch may fire before the last download (outputs lost)")
    return _with_failures({"sid": sid, "take": s["take"], "origin": "external", "runner_alive": None, "progress_level": "L0",
            "summary": s, "phases": phases, "current": parsed["current"], "steps": parsed["steps"][-40:],
            "progress": parsed["progress"], "eta": eta, "kill_at": kill_at,
            "kill_margin": round(margin) if margin is not None else None, "cost": cost, "warnings": warnings,
            "estimate_min": est_min, "source": "l0",
            "stop_modes": ["delete_pod"] if s["state"] == "running" and s.get("pod_id") else []}, text)


def stage_activity(take: str) -> dict[str, dict]:
    """{stage: {sid, progress, eta_s}} for stages advanced by an active session of this take (any origin)."""
    out: dict[str, dict] = {}
    for s in active_sessions(take):
        snap = session_snapshot(s["sid"]) or {}
        prog = snap.get("progress")
        cur = snap.get("current")
        phase = cur.get("step") if isinstance(cur, dict) else cur
        for job, status in (s.get("jobs_status") or {}).items():
            if status not in ("pending", "running", None):
                continue
            for st in registry.stages_for_remote_job(job):
                out[st] = {"sid": s["sid"], "job": job, "progress": prog, "eta": snap.get("eta"), "phase": phase}
    return out


# ------------------------------------------------------------------------------------------------ pods

def key_present() -> bool:
    """Whether a Runpod API key is configured - WITHOUT reading it (the same sources as remote._api_key: the
    RUNPOD_API_KEY variable or the key file). Only remote._ctl ever reads the key, into its child's environment."""
    import os
    if os.environ.get("RUNPOD_API_KEY"):
        return True
    f = Path(os.environ.get("DANCECAP_RUNPOD_KEY_FILE", Path.home() / "Desktop" / "runpod-key.txt"))
    try:
        return f.is_file() and f.stat().st_size > 0
    except OSError:
        return False


class PodPoller:
    """Cached runpodctl pod list (30 s) and account (5 min). Only ever reads."""

    def __init__(self, pods_ttl: float = 30.0, account_ttl: float = 300.0):
        self.pods_ttl, self.account_ttl = pods_ttl, account_ttl
        self._lock = threading.Lock()
        self.pods: list[dict] | None = None
        self.pods_at = 0.0
        self.account: dict | None = None
        self.account_at = 0.0
        self.connection = "unknown"
        self.message: str | None = None
        self.last_forced = 0.0
        self.last_seen: dict[str, float] = {}

    def _ctl_pods(self) -> list[dict]:
        from .. import remote
        data = remote._ctl("pod", "list", timeout=60)
        return data if isinstance(data, list) else (data or {}).get("pods", []) if isinstance(data, dict) else []

    def _key_present(self) -> bool:
        return key_present()

    def refresh(self, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if not force and self.pods is not None and now - self.pods_at < self.pods_ttl:
                return
            if not self._key_present():
                self.connection, self.message, self.pods, self.pods_at = "no_key", "set RUNPOD_API_KEY or create the key file", [], now
                return
            try:
                raw = self._ctl_pods()
                self.pods = [self._reduce(p) for p in raw if isinstance(p, dict)]
                for p in self.pods:
                    self.last_seen[p["id"]] = now
                self.connection, self.message = "ok", None
            except Exception as e:  # noqa: BLE001
                self.connection, self.message = "api_error", redact(str(e))[:300]
                if self.pods is None:
                    self.pods = []
            self.pods_at = now
            if now - self.account_at > self.account_ttl and self.connection == "ok":
                try:
                    from .. import remote
                    acc = remote.status().get("account")
                    self.account = acc if isinstance(acc, dict) else None
                except Exception:
                    self.account = None
                self.account_at = now

    @staticmethod
    def _reduce(p: dict) -> dict:
        out = {k: p.get(k) for k in POD_FIELDS}
        gpu = None
        m, g = p.get("machine"), p.get("gpu")
        if isinstance(m, dict):
            gpu = m.get("gpuDisplayName") or m.get("gpuTypeId")
        if not gpu and isinstance(g, dict):
            gpu = g.get("displayName") or g.get("id")
        out["gpu"] = gpu or p.get("gpuTypeId") or p.get("gpuDisplayName")
        out["gpu_count"] = p.get("gpuCount")
        out["cloud"] = p.get("cloudType") or ("SECURE" if p.get("secureCloud") else None)
        rt = p.get("runtime")
        out["runtime_status"] = (p.get("runtimeStatus") or p.get("status")
                                 or ("running" if isinstance(rt, dict) and (rt.get("uptimeInSeconds") or rt.get("uptime"))
                                     else None))
        return out

    def summary(self, force: bool = False) -> dict:
        self.refresh(force=force)
        now = time.time()
        led = estimate.ledger_rows()
        by_id = {r.get("pod_id"): r for r in led}
        act = {s["pod_id"]: s for s in active_sessions() if s.get("pod_id")}
        pods = []
        for p in self.pods or []:
            row = by_id.get(p["id"])
            managed = str(p.get("name") or "").startswith("dancecap-") or row is not None
            created = float(row["created_ts"]) if row else None
            kill_at = created + float(row.get("max_minutes") or 0) * 60 if row else None
            sess = act.get(p["id"])
            flags = []
            if row is not None and sess is None and not row.get("deleted_ts") and _runner_processes() == 0 and not _farm_managed(p["id"]):
                flags.append("orphan")
            if row is None and managed is False:
                flags.append("untracked")
            pods.append({"id": p["id"], "name": p.get("name"), "managed": managed, "gpu": p.get("gpu"),
                         "cloud": p.get("cloud"), "cost_per_hr": p.get("costPerHr"), "status": p.get("desiredStatus"),
                         "runtime_status": p.get("runtime_status"),
                         "connection": "connected" if sess else ("ended" if row and row.get("deleted_ts") else "starting"),
                         "created": created, "age_s": round(now - created) if created else None,
                         "cost_so_far_usd": round(float(row["cost_per_hr"]) * (now - created) / 3600, 2) if row else None,
                         "kill": {"at": kill_at, "source": "estimate", "armed": None,
                                  "margin_s": round(kill_at - now) if kill_at else None} if kill_at else None,
                         "gpu_util": None, "session": {"sid": sess["sid"], "take": sess.get("take")} if sess else None,
                         "ledger": ("closed" if row.get("deleted_ts") else "open") if row else "missing", "flags": flags,
                         "actions": ["delete"] if managed else []})
        # phantom accrual: open ledger rows whose pod is gone from the list
        phantom = []
        if self.connection == "ok":
            listed = {p["id"] for p in self.pods or []}
            for r in estimate.open_rows(led):
                if r.get("pod_id") not in listed and now - self.last_seen.get(r.get("pod_id"), 0) > 120:
                    phantom.append(r.get("pod_id"))
        open_rate = sum(float(r.get("cost_per_hr") or 0) for r in estimate.open_rows(led))
        acct = self.account or {}
        spend_hr = next((v for k, v in acct.items() if "spend" in k.lower() and isinstance(v, (int, float))), None)
        untracked = max(0.0, float(spend_hr) - open_rate) if spend_hr is not None else 0.0
        if not paths.is_real_root():
            untracked = 0.0  # a test tree has its own (empty) ledger: the account's real pods are not "untracked" spend
        bud = estimate.budget(untracked_per_hr=untracked if untracked > 0.05 else 0.0)
        act = active_sessions()
        level = "L0"  # the best progress level among the running sessions (what the session views show)
        failing = []  # running sessions whose pod job has failed tasks (an alert: the pod bills meanwhile)
        for s in act:
            try:
                sn = session_snapshot(s["sid"]) or {}
            except Exception:  # noqa: BLE001 - a session being written: L0 for now
                sn = {}
            lv = str(sn.get("progress_level") or "L0")
            if re.fullmatch(r"L\d", lv) and lv > level:
                level = lv
            tf = sn.get("task_failures") or {}
            if tf.get("failed"):
                failing.append({"sid": s["sid"], "take": s.get("take"), "failed": tf["failed"],
                                "skipped": tf.get("skipped", 0), "first": tf.get("first")})
        return {"connection": self.connection, "checked": self.pods_at or None, "message": self.message,
                "account": {"balance_usd": next((v for k, v in acct.items() if "balance" in k.lower()), None),
                            "spend_per_hr_usd": spend_hr,
                            "spend_limit": next((v for k, v in acct.items() if "limit" in k.lower()), None)},
                "pods": pods, "pods_running": len(pods), "phantom_accrual": phantom,
                "sessions_active": [s["sid"] for s in act], "progress_level": level, "budget": bud,
                "sessions_failing": failing}


def delete_pod(pod_id: str, confirm: str, poller: PodPoller) -> dict:
    """User-confirmed delete of one dancecap pod (remote._delete closes its ledger row)."""
    from .util import POD_ID_RE
    if not POD_ID_RE.match(pod_id or ""):
        raise ValueError("invalid pod id")
    if not confirm or confirm != pod_id[-4:]:
        raise PermissionError("type the last 4 characters of the pod id to confirm")
    poller.refresh(force=True)
    pod = next((p for p in poller.pods or [] if p["id"] == pod_id), None)
    led = {r.get("pod_id") for r in estimate.ledger_rows()}
    if pod is None and pod_id not in led:
        raise KeyError(pod_id)
    if pod is not None and not (str(pod.get("name") or "").startswith("dancecap-") or pod_id in led):
        raise PermissionError("not a dancecap pod: the monitor does not touch it")
    from .. import remote
    remote._delete(pod_id, reason="monitor: user delete")
    poller.refresh(force=True)
    return {"deleted": pod_id}
