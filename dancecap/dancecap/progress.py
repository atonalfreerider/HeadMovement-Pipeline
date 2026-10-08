"""dancecap.progress/1: progress events for long stages and Runpod sessions (stdlib only, never fatal).

Contract: docs/PIPELINE_PROGRESS.md. One JSON object per line:
    {"v": 1, "ts": <epoch s>, "job": .., "step": .., "kind": plan|start|tick|end|fail|note|metric|heartbeat,
     "done": n, "total": n|null, "unit": frames|views|iters|steps|bytes|tasks|chunks|cams|features, "msg": ..,
     "lane": .., "cmd": .., "eta_s": .., "extra": {..}, "rc": ..}

Producers (a no-op unless DANCECAP_PROGRESS is set; the runner sets it for every pod command, the pipeline launcher for
local stages):

    from dancecap import progress as P
    P.plan([("bake/pass0", n, "views"), ("bake/pass1", n, "views")])
    with P.step("bake/pass1", total=n, unit="views") as s:     # start .. end, or fail (exception re-raised)
        for i, v in enumerate(views):
            s.update(i + 1)                                       # throttled: >= 2 s AND >= 0.5 % apart
    P.tick("train", step, total=iters, unit="steps")              # one-line hook (auto start, throttled)
    P.note("gate FAILED: ...", step="gate", level="warn", branch="fallback", eta_s=3960)
    P.metric("camera_motion", lane="05", source="probe_b", centre_p95_m=0.004, focal_ratio=1.0, mount="tripod")
    stop = P.watch_log(path, "mamma_iter", step="fit_chunks", lane=tag)   # thread: 3rd-party log -> events

Readers / L0 (sessions that predate the protocol): LineParser turns existing log lines ('[texture] 900/2292 views',
'[pod-runner 10:26:57] avatar/bake done in 1080.8s', 'Iter. 187 of 500', tqdm bars, ...) into the same events; one
adapter table is shared by the L0 runner.log parser, watch_log on the pod and the puller's log tails.

    python -m dancecap.progress tail <sid>               # terminal view of a session (read-only)
    python -m dancecap.progress eta <sid> [--at HH:MM]   # monitor snapshot as JSON (progress_eta.snapshot)
    python -m dancecap.progress watch <sid>              # live ETA / budget in the terminal, every few seconds
    python -m dancecap.progress budget                   # spent / projected / worst case / warnings
    python -m dancecap.progress rates --rebuild          # work/progress/rates.json from every past session

Environment: DANCECAP_PROGRESS (file), DANCECAP_PROGRESS_JOB, DANCECAP_PROGRESS_CMD, DANCECAP_SID, DANCECAP_TAKE,
DANCECAP_PROGRESS_LANE (default lane, e.g. the front_pod task a worker runs),
DANCECAP_PROGRESS_ECHO=1 (also print a short line). On Windows each process writes its own '<stem>.<pid>.jsonl' (concurrent
appends are not atomic there); on Linux all processes of a pod session append to one file (single write() < 4 KB).
"""
from __future__ import annotations

import calendar
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

V = 1
SCHEMA = "dancecap.progress/1"
MAX_LINE = 1024          # events; plan events may use up to MAX_PLAN_LINE (still one atomic write < PIPE_BUF)
MAX_PLAN_LINE = 4000
MSG_MAX = 200
TICK_MIN_S = 2.0
TICK_MIN_FRAC = 0.005
HEARTBEAT_S = 30.0
KINDS = ("plan", "start", "tick", "end", "fail", "note", "metric", "heartbeat")
UNITS = ("frames", "views", "iters", "steps", "bytes", "tasks", "chunks", "cams", "features")

# texture passes per preset (1 + avatar_texture.PRESETS[p]["align_passes"]; TextureConfig default align_passes = 1)
TEXTURE_PASSES = {"pod": 3, "local": 2, "smoke": 2}
ROOM_ITERS = {"pod": 30000, "smoke": 300}
# MAMMA ma_3d optimisation runs (iterations per run, before early stopping): ma3d_config.yaml as used by dancecap
MAMMA_RUN_TOTALS = (50, 500, 100, 25)


# ---------------------------------------------------------------------------------------------------- safety


_TOKENISH = re.compile(r"rpa_[A-Za-z0-9_-]+|[A-Za-z0-9_\-]{32,}")
_HEX = re.compile(r"[0-9a-f]{32,}")


def redact(s: str) -> str:
    """Mask credential-looking runs (Runpod 'rpa_' keys, >= 32 chars of [A-Za-z0-9_-]); lowercase hex (sha256) stays."""
    def sub(m):
        t = m.group(0)
        return t if (not t.startswith("rpa_") and _HEX.fullmatch(t)) else "[redacted]"
    try:
        return _TOKENISH.sub(sub, str(s))
    except Exception:
        return ""


def _dumps(ev: dict) -> str:
    return json.dumps(ev, ensure_ascii=True, separators=(",", ":"), default=str, allow_nan=False)


def _clean(v):
    """JSON-safe scalars (nan/inf -> None), small containers."""
    if isinstance(v, float):
        return None if v != v or v in (float("inf"), float("-inf")) else round(v, 6)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item") and callable(v.item):  # numpy scalars
        try:
            return _clean(v.item())
        except Exception:
            return str(v)
    return v


# ---------------------------------------------------------------------------------------------------- writer


class Writer:
    """Appends events to one JSONL file: one os.write per event (atomic for < 4 KB appends on Linux). Thread-safe
    within a process. Every failure is swallowed: a lost event never fails a stage."""

    def __init__(self, path: str | os.PathLike, job: str | None = None, cmd: str | None = None,
                 sid: str | None = None, take: str | None = None, echo: bool = False):
        self.path = Path(path)
        self.job = job or "local"
        self.cmd, self.sid, self.take, self.echo = cmd, sid, take, echo
        self.lane = os.environ.get("DANCECAP_PROGRESS_LANE") or None
        self._lock = threading.RLock()
        self.last_write = 0.0
        self.open_steps: dict[tuple, "Step"] = {}
        self._hb: threading.Thread | None = None
        self._dir_ok = False

    def emit(self, kind: str, step: str, **kw) -> dict | None:
        try:
            ev = {"v": V, "ts": round(kw.pop("ts", None) or time.time(), 3), "job": kw.pop("job", None) or self.job,
                  "step": str(step), "kind": kind}
            if self.cmd and "cmd" not in kw:
                ev["cmd"] = self.cmd
            if self.lane and kw.get("lane") is None:  # e.g. a front_pod worker: its task name
                kw["lane"] = self.lane
            for k, x in kw.items():
                if x is None:
                    continue
                ev[k] = _clean(x)
            if "msg" in ev:
                ev["msg"] = redact(str(ev["msg"]))[:MSG_MAX]
            if self.sid:
                ev.setdefault("sid", self.sid)
            if self.take:
                ev.setdefault("take", self.take)
            ev.setdefault("pid", os.getpid())
            line = _dumps(ev)
            limit = MAX_PLAN_LINE if kind == "plan" else MAX_LINE
            if len(line) > limit:  # drop the bulky optional parts first
                for k in ("extra", "steps", "msg"):
                    if k in ev and len(line) > limit:
                        ev[k] = "(truncated)" if k == "msg" else None
                        ev = {a: b for a, b in ev.items() if b is not None}
                        line = _dumps(ev)
                if len(line) > limit:
                    return None
            data = (line + "\n").encode("utf-8")
            with self._lock:
                if not self._dir_ok:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._dir_ok = True
                fd = os.open(str(self.path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
                self.last_write = time.time()
            if self.echo:
                _echo_line(ev)
            return ev
        except Exception:
            return None

    # heartbeat: while a step is open and nothing was written for HEARTBEAT_S, say "alive"
    def _ensure_heartbeat(self):
        if self._hb is not None and self._hb.is_alive():
            return
        def loop():
            while True:
                time.sleep(min(10.0, HEARTBEAT_S / 3))
                with self._lock:
                    steps = list(self.open_steps.values())
                    idle = time.time() - self.last_write
                if not steps:
                    return
                if idle >= HEARTBEAT_S:
                    s = steps[-1]
                    self.emit("heartbeat", s.name, lane=s.lane, done=s.done, total=s.total, unit=s.unit)
        try:
            self._hb = threading.Thread(target=loop, name="dancecap-progress-heartbeat", daemon=True)
            self._hb.start()
        except Exception:
            self._hb = None


def _echo_line(ev: dict) -> None:
    try:
        d, t = ev.get("done"), ev.get("total")
        frac = f" {d}/{t}" if d is not None else ""
        print(f"[progress] {ev['kind']} {ev['job']}/{ev['step']}{'@' + str(ev['lane']) if ev.get('lane') else ''}"
              f"{frac}{' ' + ev['msg'] if ev.get('msg') else ''}", flush=True)
    except Exception:
        pass


class Step:
    """One step of a job: start .. tick* .. end|fail. A context manager (an exception -> fail, re-raised)."""

    def __init__(self, w: Writer | None, name: str, total=None, unit=None, lane=None, job=None):
        self.w, self.name, self.total, self.unit, self.lane, self.job = w, name, total, unit, lane, job
        self.done = 0
        self.t0 = None
        self._last_ts = 0.0
        self._last_done = 0
        self.closed = False

    @property
    def key(self):
        return (self.job, self.name, self.lane)

    def start(self, msg: str | None = None, **extra):
        if self.w is None or self.t0 is not None:
            return self
        self.t0 = self._last_ts = time.time()
        self.w.emit("start", self.name, job=self.job, lane=self.lane, total=self.total, unit=self.unit, msg=msg,
                    extra=extra or None)
        with self.w._lock:
            self.w.open_steps[self.key] = self
        self.w._ensure_heartbeat()
        return self

    def update(self, done, total=None, msg: str | None = None, eta_s: float | None = None, force: bool = False,
               **extra) -> bool:
        """Throttled tick; returns True when an event was written. A changed total is always written."""
        if self.w is None:
            return False
        try:
            if self.t0 is None:
                self.start()
            revised = total is not None and total != self.total
            if total is not None:
                self.total = total
            self.done = done
            now = time.time()
            tot = self.total or 0
            frac_ok = (not tot) or abs(done - self._last_done) >= TICK_MIN_FRAC * tot
            if not (force or revised or (now - self._last_ts >= TICK_MIN_S and frac_ok)):
                return False
            self._last_ts, self._last_done = now, done
            self.w.emit("tick", self.name, job=self.job, lane=self.lane, done=done, total=self.total, unit=self.unit,
                        msg=msg, eta_s=eta_s, extra=extra or None)
            return True
        except Exception:
            return False

    def end(self, done=None, msg: str | None = None, **extra):
        if self.w is None or self.closed:
            return
        self.closed = True
        if self.t0 is None:
            self.start()
        if done is not None:
            self.done = done
        self.w.emit("end", self.name, job=self.job, lane=self.lane, done=self.done if self.done else done,
                    total=self.total, unit=self.unit, msg=msg,
                    extra={"seconds": round(time.time() - (self.t0 or time.time()), 2), **extra})
        with self.w._lock:
            self.w.open_steps.pop(self.key, None)

    def fail(self, rc=None, msg: str | None = None, **extra):
        if self.w is None or self.closed:
            return
        self.closed = True
        self.w.emit("fail", self.name, job=self.job, lane=self.lane, done=self.done, total=self.total,
                    unit=self.unit, rc=rc, msg=msg, extra=extra or None)
        with self.w._lock:
            self.w.open_steps.pop(self.key, None)

    def __enter__(self):
        return self.start()

    def __exit__(self, et, e, tb):
        if et is None:
            self.end()
        else:
            self.fail(rc=getattr(et, "__name__", "error"), msg=str(e)[:MSG_MAX] if e else None)
        return False


_writers: dict[str, Writer] = {}
_wlock = threading.Lock()


def _resolve_path(p: str, nt: bool | None = None) -> Path:
    path = Path(p)
    if (os.name == "nt") if nt is None else nt:  # concurrent appends are not atomic on Windows: one file per process
        pid = str(os.getpid())
        stem = path.name[:-6] if path.name.endswith(".jsonl") else path.name
        if not stem.endswith("." + pid):
            path = path.with_name(f"{stem}.{pid}.jsonl")
    return path


def writer() -> Writer | None:
    """The process' writer from DANCECAP_PROGRESS, or None (then every helper call is a no-op)."""
    p = os.environ.get("DANCECAP_PROGRESS")
    if not p:
        return None
    try:
        with _wlock:
            w = _writers.get(p)
            if w is None:
                w = Writer(_resolve_path(p), job=os.environ.get("DANCECAP_PROGRESS_JOB") or None,
                           cmd=os.environ.get("DANCECAP_PROGRESS_CMD") or None,
                           sid=os.environ.get("DANCECAP_SID") or None, take=os.environ.get("DANCECAP_TAKE") or None,
                           echo=os.environ.get("DANCECAP_PROGRESS_ECHO") == "1")
                _writers[p] = w
            return w
    except Exception:
        return None


def enabled() -> bool:
    return writer() is not None


def default_lane() -> str | None:
    """DANCECAP_PROGRESS_LANE (set per worker by front_pod): the task this process runs, e.g. 'loc:02:1151'."""
    return os.environ.get("DANCECAP_PROGRESS_LANE") or None


def plan(steps, job: str | None = None, step: str = "plan", msg: str | None = None, **extra) -> None:
    """steps: [(step, total, unit) | (step, total, unit, weight) | {step, total, unit, weight?, after?, lane?}]."""
    w = writer()
    if w is None:
        return
    try:
        out = []
        for s in steps:
            if isinstance(s, dict):
                out.append({k: v for k, v in s.items() if v is not None})
            else:
                d = {"step": s[0], "total": s[1] if len(s) > 1 else None, "unit": s[2] if len(s) > 2 else None}
                if len(s) > 3 and s[3] is not None:
                    d["weight"] = s[3]
                out.append(d)
        w.emit("plan", step, job=job, steps=out, msg=msg, extra=extra or None)
    except Exception:
        pass


def step(name: str, total=None, unit: str | None = None, lane: str | None = None, job: str | None = None) -> Step:
    return Step(writer(), name, total, unit, lane, job)


def _registered(name, lane, job, total=None, unit=None) -> Step | None:
    w = writer()
    if w is None:
        return None
    with w._lock:
        s = w.open_steps.get((job, name, lane))
    if s is None:
        s = Step(w, name, total, unit, lane, job).start()
    return s


def tick(name: str, done, total=None, unit: str | None = None, lane: str | None = None, job: str | None = None,
         **extra) -> None:
    """One-line hook: auto-starts the step on the first call, then throttled ticks."""
    try:
        s = _registered(name, lane, job, total, unit)
        if s is not None:
            if unit and not s.unit:
                s.unit = unit
            s.update(done, total=total, **extra)
    except Exception:
        pass


def start(name: str, total=None, unit: str | None = None, lane: str | None = None, job: str | None = None,
          msg: str | None = None) -> None:
    try:
        _registered(name, lane, job, total, unit)
    except Exception:
        pass


def end(name: str, done=None, lane: str | None = None, job: str | None = None, msg: str | None = None, **extra):
    try:
        s = _registered(name, lane, job)
        if s is not None:
            s.end(done=done, msg=msg, **extra)
    except Exception:
        pass


def fail(name: str, rc=None, lane: str | None = None, job: str | None = None, msg: str | None = None):
    try:
        s = _registered(name, lane, job)
        if s is not None:
            s.fail(rc=rc, msg=msg)
    except Exception:
        pass


def note(msg: str, step: str = "note", level: str = "info", job: str | None = None, lane: str | None = None,
         eta_s: float | None = None, **extra) -> None:
    w = writer()
    if w is not None:
        w.emit("note", step, job=job, lane=lane, msg=msg, eta_s=eta_s, extra={"level": level, **extra})


def metric(name: str, step: str | None = None, job: str | None = None, lane: str | None = None, **values) -> None:
    w = writer()
    if w is not None:
        w.emit("metric", step or name, job=job, lane=lane, extra={"metric": name, **values})


# ---------------------------------------------------------------------------------------------------- adapters


def _hms(s: str) -> int:
    h, m, x = (int(v) for v in s.split(":"))
    return h * 3600 + m * 60 + x


def _mmss(s: str) -> float:
    parts = [float(p) for p in s.split(":")]
    out = 0.0
    for p in parts:
        out = out * 60 + p
    return out


R = {  # the shared adapter table (docs/PIPELINE_PROGRESS.md 3.2)
    "runner": re.compile(r"^\[pod-runner (\d\d:\d\d:\d\d)\] (.*)$"),
    "runner_start": re.compile(r"^(\S+) \.\.\.$"),
    "runner_end": re.compile(r"^(\S+) (done|FAILED) in ([\d.]+)s$"),
    "runner_bundle": re.compile(r"^bundle (\S+): ([\d.]+) MB \((\d+) files.*?; upload ~([\d.]+) min; features \[(.*?)\]; "
                                r"pod max (\d+) min"),
    "runner_job": re.compile(r"^([\w.-]+): (ok|failed: .*|skipped: .*) \((\d+)s\)$|^([\w.-]+): (skipped: .*)$"),
    "runner_err": re.compile(r"^session error: (.*)$"),
    "pod_runner": re.compile(r"^\[pod-runner\] (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (start|end) (\S+?)/(\S+)(?: rc=(\S+))?$"),
    "pod_runner_done": re.compile(r"^\[pod-runner\] (\S+?)/(\S+) already done$"),
    "bootstrap": re.compile(r"^\[bootstrap\] (\w+): (start|done in (\d+)s|FAILED \(rc=(\d+)\) after (\d+)s|already done)$"),
    "bootstrap_fin": re.compile(r"^\[bootstrap\] finished rc=(\d+)"),
    "texture_preset": re.compile(r"^\[texture\] preset (\S+): (\d+) views from (\d+) cameras"),
    "texture_views": re.compile(r"^\[texture\] (\d+)/(\d+) views"),
    "texture_pass": re.compile(r"^\[texture\] pass (\d+)( \(aligned\))?: (\d+) s"),
    "texture_done": re.compile(r"^\[texture\] -> .* \((\d+) s\)\s*$"),
    "room_plan": re.compile(r"^\[room\] (\d+) views \((\d+) train, (\d+) held out\).*\(preset (\w+)\)"),
    "room_step": re.compile(r"^\[room\] step\s+(\d+)\s+L1 ([\d.]+)\s+masked PSNR ([\d.]+)\s+N ([\d,]+)\s+(\d+)s"),
    "room_trained": re.compile(r"^\[room\] splat_full\.ply"),
    "hands": re.compile(r"^\[(hand_worker|mano_worker)\] (\w+) (\d+)/(\d+) (\d+)s$"),
    "face_worker": re.compile(r"^\[face_worker\] (\d+)/(\d+) (\d+)s$"),
    "hands_done": re.compile(r"^\[hand_worker\] done (\d+) hand crops.* in (\d+)s|^\[mano_worker\] (\w+): \{"),
    "front_plan": re.compile(r"^\[front\] (\S+): (\d+) tasks;"),
    "front_start": re.compile(r"^\[front\] (\d\d:\d\d:\d\d) start (\S+)$"),
    "front_end": re.compile(r"^\[front\] (\d\d:\d\d:\d\d) (\S+): (\w+) rc=(\S+) in (\d+)s"),
    "front_earlier": re.compile(r"^\[front\] (\S+): done earlier"),
    "front_done": re.compile(r"^\[front\] done; not ok: (.*?);"),
    "loc": re.compile(r"^\[cameras\] (\w+): (\d+)/(\d+) frames, (\d+) ok \((\d+)s\)"),
    "chunks_plan": re.compile(r"^\[chunks\] (\d+) chunks of (\S+?)(?: \+ held-out (\S+))?, (\d+) in parallel"),
    "chunks_lane": re.compile(r"^\[chunks\] ([\w-]+_f\d+-\d+): rc=(\S+) in (\d+)s$"),
    "chunks_gate": re.compile(r"^\[chunks\] gate (passed|PASSED|FAILED)(?::? ?(.*))?$"),
    "chunks_fallback": re.compile(r"^\[chunks\] fallback: one full fit of (\S+) \((\d+) frames, ~(\d+) min\)"),
    "chunks_notime": re.compile(r"^\[chunks\] no time for the full-fit fallback"),
    "mamma_iter": re.compile(r"^Iter\. (\d+) of (\d+)"),
    "mamma_stop": re.compile(r"^Stopping early at iteration (\d+)"),
    "mamma_took": re.compile(r"^Optimization took ([\d.]+) seconds"),
    "tqdm": re.compile(r"(\d+)/(\d+) \[(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?)"),
    "detect": re.compile(r"^\[detect\] (\w+): (\d+) frames .*?(\d+) ms/frame"),
    # dancecap.dancer_splats (docs/DANCER_SPLATS_RESEARCH.md 5.4): global step over the S1-S3 schedule
    "splats_step": re.compile(r"^\[dancer_splats\] step (\d+)/(\d+) stage (S\d) psnr ([\d.]+) iou ([\d.]+) "
                              r"N_gauss (\d+) (\d+)s"),
    "splats_eval": re.compile(r"^\[dancer_splats\] eval (\d+)/(\d+) views"),
    "splats_trained": re.compile(r"^\[dancer_splats\] trained (\d+) steps in ([\d.]+) min"),
}
ADAPTERS = ("runner", "bootstrap", "texture", "room", "hands", "front", "loc", "chunks", "mamma_iter", "tqdm",
            "detect", "splats")
RUNNER_LOG_ADAPTERS = ("runner", "bootstrap", "texture", "room", "hands", "front", "loc", "chunks", "detect",
                       "splats")


class LineParser:
    """Stateful log-line -> event translator for one stream (a runner.log, a pod log tail, a watched file).

    Events are dicts like the protocol's, plus 'src' (adapter), 'line' (index in this stream), 'rx' (receive time when
    given) and 'ts' only when the line itself carries a clock: '[pod-runner HH:MM:SS]' (the runner's local clock, date
    from date0 with midnight roll-over), '[pod-runner] <ISO>Z' and '[front] HH:MM:SS' (pod clock, UTC)."""

    def __init__(self, adapters=RUNNER_LOG_ADAPTERS, date0: tuple[int, int, int] | None = None,
                 job: str | None = None, step: str | None = None, lane: str | None = None,
                 texture_passes: int | None = None, room_iters: int | None = None, run_totals=MAMMA_RUN_TOTALS):
        self.adapters = set(adapters)
        self.date0 = date0
        self._day = 0
        self._last_sod = None
        self.utc_date: tuple[int, int, int] | None = None
        self._utc_day, self._utc_last = 0, None  # '[front] HH:MM:SS' midnight-UTC roll-over
        self.job, self.cmd = job, None   # context: the pod command now running
        self.base_step, self.lane = step, lane
        self.n = 0
        self.tex_pass = 0
        self.tex_passes = texture_passes
        self.tex_views = None
        self.room_iters = room_iters
        self.mamma_run = 0
        self.run_totals = run_totals
        self.attempt = 0

    # clocks --------------------------------------------------------------------------------------------------
    def _local_ts(self, hms: str) -> float | None:
        if not self.date0:
            return None
        sod = _hms(hms)
        if self._last_sod is not None and sod < self._last_sod - 6 * 3600:
            self._day += 1
        self._last_sod = sod
        y, m, d = self.date0
        return time.mktime((y, m, d + self._day, 0, 0, 0, 0, 0, -1)) + sod

    def _utc_hms(self, hms: str) -> float | None:
        """'[front] HH:MM:SS' (pod clock, UTC): the date of the last '[pod-runner] <ISO>Z' line (else the session's
        local date), rolling over when the clock wraps past midnight UTC (17:00 PDT); never before the session's
        local midnight (a front job started after 00:00 UTC without an ISO line before it)."""
        sod = _hms(hms)
        if self.utc_date is None:
            if not self.date0:
                return None
            g = time.gmtime(time.mktime((*self.date0, 12, 0, 0, 0, 0, -1)))
            self.utc_date = (g.tm_year, g.tm_mon, g.tm_mday)
            self._utc_day, self._utc_last = 0, None
            floor = time.mktime((*self.date0, 0, 0, 0, 0, 0, -1))
            if calendar.timegm((*self.utc_date, 0, 0, 0)) + sod < floor:
                self._utc_day = 1
        if self._utc_last is not None and sod < self._utc_last - 6 * 3600:
            self._utc_day += 1
        self._utc_last = sod
        y, m, d = self.utc_date
        return calendar.timegm((y, m, d, 0, 0, 0)) + 86400 * self._utc_day + sod

    def _ctx(self, ev: dict) -> dict:
        if self.job and "job" not in ev:
            ev["job"] = self.job
        if self.cmd and "cmd" not in ev:
            ev["cmd"] = self.cmd
        if self.lane and "lane" not in ev:
            ev["lane"] = self.lane
        return ev

    def _step(self, s: str) -> str:
        return f"{self.base_step}/{s}" if self.base_step else s

    def feed(self, line: str, rx: float | None = None) -> list[dict]:
        out: list[dict] = []
        self.n += 1
        try:
            line = line.rstrip("\r\n")
            # tqdm writes \r updates: every segment is a line of its own
            segs = [s for s in line.split("\r") if s.strip()] if "\r" in line else [line]
            for seg in segs:
                out += self._feed(seg.strip() if seg.startswith(" ") and "tqdm" in self.adapters else seg)
        except Exception:
            return out
        for ev in out:
            ev.setdefault("v", V)
            ev["line"] = self.n
            if rx is not None:
                ev["rx"] = rx
        return out

    def _feed(self, line: str) -> list[dict]:
        a = self.adapters
        if "runner" in a and line.startswith("[pod-runner"):
            return self._runner(line)
        if "bootstrap" in a and line.startswith("[bootstrap]"):
            return self._bootstrap(line)
        if "texture" in a and line.startswith("[texture]"):
            return self._texture(line)
        if "room" in a and line.startswith("[room]"):
            return self._room(line)
        if "splats" in a and line.startswith("[dancer_splats]"):
            return self._splats(line)
        if "hands" in a and line.startswith(("[hand_worker]", "[mano_worker]", "[face_worker]")):
            return self._hands(line)
        if "front" in a and line.startswith("[front]"):
            return self._front(line)
        if "loc" in a and line.startswith("[cameras]"):
            m = R["loc"].match(line)
            if m:
                return [self._ctx({"kind": "tick", "step": self._step("loc"), "lane": m.group(1),
                                   "done": int(m.group(2)), "total": int(m.group(3)), "unit": "frames", "src": "loc",
                                   "extra": {"ok": int(m.group(4)), "elapsed_s": int(m.group(5))}})]
            return []
        if "chunks" in a and line.startswith("[chunks]"):
            return self._chunks(line)
        if "detect" in a and line.startswith("[detect]"):
            m = R["detect"].match(line)
            if m:
                return [self._ctx({"kind": "end", "step": self._step("detect"), "lane": m.group(1),
                                   "done": int(m.group(2)), "total": int(m.group(2)), "unit": "frames",
                                   "src": "detect", "extra": {"ms_per_frame": int(m.group(3))}})]
            return []
        if "mamma_iter" in a and line[:1] in ("I", "S", "O"):
            r = self._mamma(line)
            if r:
                return r
        if "tqdm" in a:
            ms = list(R["tqdm"].finditer(line))
            if ms:
                m = ms[-1]
                d, t = int(m.group(1)), int(m.group(2))
                return [self._ctx({"kind": "tick" if d < t else "end", "step": self.base_step or "tqdm",
                                   "done": d, "total": t, "unit": "frames", "src": "tqdm",
                                   "eta_s": _mmss(m.group(4)), "extra": {"elapsed_s": _mmss(m.group(3))}})]
        return []

    # runner ------------------------------------------------------------------------------------------------
    def _runner(self, line: str) -> list[dict]:
        m = R["pod_runner"].match(line)
        if m:
            iso, what, job, cmd, rc = m.groups()
            ts = calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))
            g = time.gmtime(ts)
            self.utc_date = (g.tm_year, g.tm_mon, g.tm_mday)
            self._utc_day, self._utc_last = 0, g.tm_hour * 3600 + g.tm_min * 60 + g.tm_sec
            if what == "start":
                self.job, self.cmd = job, cmd
                self.tex_pass, self.tex_views, self.mamma_run, self._tex_started = 0, None, 0, -1
                return [{"kind": "start", "job": job, "step": cmd, "ts": ts, "clock": "pod", "src": "pod-runner"}]
            ok = rc == "0"
            ev = {"kind": "end" if ok else "fail", "job": job, "step": cmd, "ts": ts, "clock": "pod",
                  "src": "pod-runner", "rc": int(rc) if rc and rc.lstrip("-").isdigit() else rc}
            self.cmd = None
            return [ev]
        m = R["pod_runner_done"].match(line)
        if m:
            return [{"kind": "note", "job": m.group(1), "step": m.group(2), "src": "pod-runner",
                     "msg": "already done (marker)", "extra": {"cached": True}}]
        m = R["runner"].match(line)
        if not m:
            return []
        ts = self._local_ts(m.group(1))
        body = m.group(2)
        base = {"ts": ts, "clock": "local", "src": "runner"}
        x = R["runner_start"].match(body)
        if x:
            name = x.group(1)
            job, st = (name.split("/", 1) if "/" in name else ("_session", name))
            if name == "bundle_scan":
                self.attempt += 1
            return [{**base, "kind": "start", "job": job, "step": st, "extra": {"runner_step": name,
                                                                               "attempt": self.attempt}}]
        x = R["runner_end"].match(body)
        if x:
            name, what, secs = x.groups()
            job, st = (name.split("/", 1) if "/" in name else ("_session", name))
            return [{**base, "kind": "end" if what == "done" else "fail", "job": job, "step": st,
                     "extra": {"runner_step": name, "seconds": float(secs), "attempt": self.attempt}}]
        x = R["runner_bundle"].match(body)
        if x:
            feats = [f.strip().strip("'\"") for f in x.group(5).split(",") if f.strip()]
            return [{**base, "kind": "note", "job": "_session", "step": "bundle", "msg": body[:MSG_MAX],
                     "extra": {"bundle": x.group(1), "mb": float(x.group(2)), "files": int(x.group(3)),
                               "upload_min_est": float(x.group(4)), "features": feats,
                               "pod_max_min": int(x.group(6)), "attempt": self.attempt}}]
        x = R["runner_job"].match(body)
        if x:
            if x.group(1):
                job, status, secs = x.group(1), x.group(2), int(x.group(3))
            else:
                job, status, secs = x.group(4), x.group(5), None
            kind = "end" if status == "ok" else ("note" if status.startswith("skipped") else "fail")
            return [{**base, "kind": kind, "job": job, "step": "_job", "msg": status[:MSG_MAX],
                     "extra": {"seconds": secs, "status": status[:MSG_MAX]}}]
        x = R["runner_err"].match(body)
        if x:
            return [{**base, "kind": "fail", "job": "_session", "step": "session", "msg": x.group(1)[:MSG_MAX]}]
        if body.startswith(("dry run", "nothing to do")):
            return [{**base, "kind": "note", "job": "_session", "step": "session", "msg": body[:MSG_MAX]}]
        return [{**base, "kind": "note", "job": "_session", "step": "runner", "msg": body[:MSG_MAX],
                 "extra": {"anchor": True}}]

    def _bootstrap(self, line: str) -> list[dict]:
        m = R["bootstrap"].match(line)
        if m:
            f, what = m.group(1), m.group(2)
            base = {"job": "_session", "step": f"bootstrap/{f}", "unit": "features", "src": "bootstrap"}
            if what == "start":
                return [{**base, "kind": "start"}]
            if what.startswith("done"):
                return [{**base, "kind": "end", "extra": {"seconds": int(m.group(3))}}]
            if what == "already done":
                return [{**base, "kind": "end", "extra": {"seconds": 0, "cached": True}}]
            return [{**base, "kind": "fail", "rc": int(m.group(4)), "extra": {"seconds": int(m.group(5))}}]
        m = R["bootstrap_fin"].match(line)
        if m:
            return [{"kind": "note", "job": "_session", "step": "bootstrap", "src": "bootstrap",
                     "msg": f"bootstrap finished rc={m.group(1)}", "rc": int(m.group(1))}]
        return []

    # stage adapters -----------------------------------------------------------------------------------------
    def _texture(self, line: str) -> list[dict]:
        m = R["texture_views"].match(line)
        if m:
            d, t = int(m.group(1)), int(m.group(2))
            out = []
            if self.tex_views is None:
                self.tex_views = t
            if d <= 50 and self._tex_started != self.tex_pass:
                out.append(self._ctx({"kind": "start", "step": f"bake/pass{self.tex_pass}", "total": t,
                                      "unit": "views", "src": "texture"}))
                self._tex_started = self.tex_pass
            out.append(self._ctx({"kind": "tick", "step": f"bake/pass{self.tex_pass}", "done": d, "total": t,
                                  "unit": "views", "src": "texture"}))
            return out
        m = R["texture_pass"].match(line)
        if m:
            k, secs = int(m.group(1)), int(m.group(3))
            self.tex_pass = k + 1
            return [self._ctx({"kind": "end", "step": f"bake/pass{k}", "done": self.tex_views,
                               "total": self.tex_views, "unit": "views", "src": "texture",
                               "extra": {"seconds": secs, "aligned": bool(m.group(2))}})]
        m = R["texture_preset"].match(line)
        if m:
            preset, n = m.group(1), int(m.group(2))
            self.tex_views = n
            k = self.tex_passes or TEXTURE_PASSES.get(preset, 2)
            self.tex_passes = k
            steps = [{"step": "bake/load", "total": None, "unit": None}] + [
                {"step": f"bake/pass{i}", "total": n, "unit": "views"} for i in range(k)] + [
                {"step": "bake/finish", "total": None, "unit": None}]
            return [self._ctx({"kind": "plan", "step": "bake", "steps": steps, "src": "texture",
                               "extra": {"preset": preset, "views": n, "cams": int(m.group(3)), "passes": k}})]
        m = R["texture_done"].match(line)
        if m:
            return [self._ctx({"kind": "end", "step": "bake/finish", "src": "texture",
                               "extra": {"seconds_total": int(m.group(1))}})]
        return []

    _tex_started = -1

    def _room(self, line: str) -> list[dict]:
        m = R["room_step"].match(line)
        if m:
            st = int(m.group(1))
            total = self.room_iters or ROOM_ITERS["pod"]
            done = st + 1 if st > 0 else 0
            ev = self._ctx({"kind": "tick", "step": self._step("train"), "done": done, "total": total,
                            "unit": "steps", "src": "room",
                            "extra": {"l1": float(m.group(2)), "psnr": float(m.group(3)),
                                      "gaussians": int(m.group(4).replace(",", "")), "elapsed_s": int(m.group(5))}})
            out = []
            if st == 0:
                out.append(self._ctx({"kind": "start", "step": self._step("train"), "total": total, "unit": "steps",
                                      "src": "room"}))
            if done >= total:
                ev["kind"] = "end"
            return out + [ev]
        m = R["room_plan"].match(line)
        if m:
            self.room_iters = self.room_iters or ROOM_ITERS.get(m.group(4), ROOM_ITERS["pod"])
            return [self._ctx({"kind": "plan", "step": "room", "src": "room",
                               "steps": [{"step": "train", "total": self.room_iters, "unit": "steps"},
                                         {"step": "quest", "total": None, "unit": "steps"},
                                         {"step": "post", "total": None, "unit": None}],
                               "extra": {"views": int(m.group(1)), "preset": m.group(4)}})]
        return []

    _splats_started = False

    def _splats(self, line: str) -> list[dict]:
        m = R["splats_step"].match(line)
        if m:
            d, t = int(m.group(1)), int(m.group(2))
            out = []
            if not self._splats_started or d <= 1:
                self._splats_started = True
                out.append(self._ctx({"kind": "start", "step": self._step("splats/train"), "total": t, "unit": "steps",
                                      "src": "splats"}))
            out.append(self._ctx({"kind": "tick" if d < t else "end", "step": self._step("splats/train"), "done": d,
                                  "total": t, "unit": "steps", "src": "splats",
                                  "extra": {"stage": m.group(3), "psnr": float(m.group(4)), "iou": float(m.group(5)),
                                            "gaussians": int(m.group(6)), "elapsed_s": int(m.group(7))}}))
            return out
        m = R["splats_eval"].match(line)
        if m:
            d, t = int(m.group(1)), int(m.group(2))
            return [self._ctx({"kind": "tick" if d < t else "end", "step": self._step("splats/eval"), "done": d,
                               "total": t, "unit": "views", "src": "splats"})]
        m = R["splats_trained"].match(line)
        if m:
            return [self._ctx({"kind": "end", "step": self._step("splats/train"), "src": "splats",
                               "extra": {"steps": int(m.group(1)), "seconds": float(m.group(2)) * 60}})]
        return []

    def _hands(self, line: str) -> list[dict]:
        m = R["face_worker"].match(line)
        if m:
            d, t, el = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return [self._ctx({"kind": "tick", "step": self._step("face_worker"), "done": d, "total": t,
                               "unit": "frames", "src": "hands", "extra": {"elapsed_s": el}})]
        m = R["hands"].match(line)
        if m:
            worker, stage, d, t, el = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
            out = []
            if d == 0 or el <= 15 and d < t * 0.02:
                out.append(self._ctx({"kind": "start", "step": self._step(f"{worker}/{stage}"), "total": t,
                                      "unit": "frames", "src": "hands"}))
            out.append(self._ctx({"kind": "tick", "step": self._step(f"{worker}/{stage}"), "done": d, "total": t,
                                  "unit": "frames", "src": "hands", "extra": {"elapsed_s": el}}))
            return out
        m = R["hands_done"].match(line)
        if m:
            if m.group(1):
                return [self._ctx({"kind": "end", "step": self._step("hand_worker"), "src": "hands",
                                   "extra": {"seconds": int(m.group(2))}})]
            return [self._ctx({"kind": "end", "step": self._step(f"mano_worker/{m.group(3)}"), "src": "hands"})]
        return []

    def _front(self, line: str) -> list[dict]:
        m = R["front_start"].match(line)
        if m:
            return [self._ctx({"kind": "start", "step": self._step("front"), "lane": m.group(2),
                               "ts": self._utc_hms(m.group(1)), "clock": "pod", "src": "front", "unit": "tasks"})]
        m = R["front_end"].match(line)
        if m:
            ok = m.group(3) == "ok"
            return [self._ctx({"kind": "end" if ok else "fail", "step": self._step("front"), "lane": m.group(2),
                               "ts": self._utc_hms(m.group(1)), "clock": "pod", "src": "front",
                               "rc": m.group(4), "extra": {"seconds": int(m.group(5)), "state": m.group(3)}})]
        m = R["front_earlier"].match(line)
        if m:
            return [self._ctx({"kind": "end", "step": self._step("front"), "lane": m.group(1), "src": "front",
                               "extra": {"seconds": 0, "earlier": True}})]
        m = R["front_plan"].match(line)
        if m:
            return [self._ctx({"kind": "plan", "step": self._step("front"), "src": "front",
                               "steps": [], "total": int(m.group(2)), "unit": "tasks",
                               "extra": {"tasks": int(m.group(2)), "resumed": "RESUMED" in line}})]
        m = R["front_done"].match(line)
        if m:
            return [self._ctx({"kind": "note", "step": self._step("front"), "src": "front",
                               "msg": f"front done; not ok: {m.group(1)}"[:MSG_MAX]})]
        return []

    def _chunks(self, line: str) -> list[dict]:
        m = R["chunks_lane"].match(line)
        if m:
            tag, rc, secs = m.groups()
            fr = re.search(r"_f(\d+)-(\d+)$", tag)
            return [self._ctx({"kind": "end" if rc == "0" else "fail", "step": self._step("fit_chunks"),
                               "lane": tag, "src": "chunks", "rc": int(rc) if rc.isdigit() else rc,
                               "extra": {"seconds": int(secs),
                                         "frames": int(fr.group(2)) - int(fr.group(1)) if fr else None}})]
        m = R["chunks_plan"].match(line)
        if m:
            n, tag, held, par = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
            return [self._ctx({"kind": "plan", "step": self._step("fit_chunks"), "src": "chunks",
                               "steps": [{"step": "fit_chunks", "lane": "*", "total": None, "unit": "iters"},
                                         {"step": "merge"}, {"step": "gate"}, {"step": "select"}],
                               "extra": {"chunks": n, "tag": tag, "holdout": held, "parallel": par,
                                         "lanes": n + (1 if held else 0), "mps": "MPS on" in line}})]
        m = R["chunks_gate"].match(line)
        if m:
            passed = m.group(1).lower() == "passed"
            return [self._ctx({"kind": "note", "step": self._step("gate"), "src": "chunks",
                               "msg": line[len("[chunks] "):][:MSG_MAX],
                               "extra": {"level": "info" if passed else "warn",
                                         "branch": "pass" if passed else "fallback?", "gate": passed}})]
        m = R["chunks_fallback"].match(line)
        if m:
            return [self._ctx({"kind": "plan", "step": self._step("gate"), "src": "chunks",
                               "steps": [{"step": "gate/fallback_fit", "total": None, "unit": "iters"}],
                               "eta_s": int(m.group(3)) * 60,
                               "msg": line[len("[chunks] "):][:MSG_MAX],
                               "extra": {"branch": "fallback", "frames": int(m.group(2)), "tag": m.group(1)}})]
        if R["chunks_notime"].match(line):
            return [self._ctx({"kind": "note", "step": self._step("gate"), "src": "chunks",
                               "msg": line[len("[chunks] "):][:MSG_MAX],
                               "extra": {"level": "warn", "branch": "no_fallback"}})]
        return []

    def _mamma(self, line: str) -> list[dict]:
        m = R["mamma_iter"].match(line)
        if m:
            d, t = int(m.group(1)), int(m.group(2))
            out = []
            # the four runs have distinct totals (50/500/100/25): a tail that starts mid-log still finds its run
            k = self.run_totals.index(t) + 1 if t in self.run_totals else None
            if k is not None and k != self.mamma_run and d != 1:
                self.mamma_run = k
                out.append(self._ctx({"kind": "start", "step": self._step(f"run{k}"), "total": t, "unit": "iters",
                                      "src": "mamma_iter", "extra": {"joined": True}}))
            elif d == 1 or self.mamma_run == 0:
                self.mamma_run = k if (k is not None and k > self.mamma_run) else self.mamma_run + 1
                out.append(self._ctx({"kind": "start", "step": self._step(f"run{self.mamma_run}"), "total": t,
                                      "unit": "iters", "src": "mamma_iter"}))
            out.append(self._ctx({"kind": "tick" if d < t else "end", "step": self._step(f"run{self.mamma_run}"),
                                  "done": d, "total": t, "unit": "iters", "src": "mamma_iter"}))
            return out
        m = R["mamma_stop"].match(line)
        if m:
            d = int(m.group(1))
            return [self._ctx({"kind": "end", "step": self._step(f"run{max(self.mamma_run, 1)}"), "done": d,
                               "total": d, "unit": "iters", "src": "mamma_iter",
                               "msg": f"Stopping early at iteration {d}", "extra": {"early_stop": True}})]
        m = R["mamma_took"].match(line)
        if m:
            return [self._ctx({"kind": "end", "step": self._step("fit") if self.base_step else "fit",
                               "src": "mamma_iter", "extra": {"seconds": float(m.group(1)), "runs": self.mamma_run}})]
        return []


# ---------------------------------------------------------------------------------------------------- watch_log


def watch_log(path: str | os.PathLike, adapter: str | tuple, step: str | None = None, lane: str | None = None,
              job: str | None = None, interval: float = 2.0, from_start: bool = True):
    """Tail a third-party log in a daemon thread and write its progress (adapters above) as events. Returns stop();
    stop() reads what is left, then returns. A no-op (returns a no-op stop) when progress is disabled."""
    w = writer()
    if w is None:
        return lambda: None
    adapters = (adapter,) if isinstance(adapter, str) else tuple(adapter)
    parser = LineParser(adapters, step=step, lane=lane)
    stop_ev = threading.Event()
    p = Path(path)
    state = {"off": 0, "buf": b""}
    if not from_start:
        try:
            state["off"] = p.stat().st_size
        except OSError:
            pass
    tick_steps: dict[tuple, Step] = {}

    def pump():
        try:
            size = p.stat().st_size
        except OSError:
            return
        if size < state["off"]:  # truncated / rewritten
            state["off"], state["buf"] = 0, b""
        if size == state["off"]:
            return
        with open(p, "rb") as fh:
            fh.seek(state["off"])
            data = fh.read(min(size - state["off"], 4 << 20))
        state["off"] += len(data)
        buf = state["buf"] + data
        parts = re.split(rb"[\r\n]", buf)
        state["buf"] = parts.pop()
        for raw in parts:
            if not raw.strip():
                continue
            for ev in parser.feed(raw.decode("utf-8", "replace")):
                _relay(w, ev, job, tick_steps)

    def loop():
        while not stop_ev.wait(interval):
            try:
                pump()
            except Exception:
                pass
        try:
            pump()
        except Exception:
            pass

    t = threading.Thread(target=loop, name=f"dancecap-watch-{p.name}", daemon=True)
    t.start()

    def stop():
        stop_ev.set()
        t.join(timeout=interval + 5)
    return stop


def _relay(w: Writer, ev: dict, job: str | None, steps: dict) -> None:
    """Parser event -> writer event, with the writer's throttling for ticks."""
    kind, name, lane = ev.get("kind"), ev.get("step"), ev.get("lane")
    key = (job, name, lane)
    if kind == "tick":
        s = steps.get(key)
        if s is None:
            s = steps[key] = Step(w, name, ev.get("total"), ev.get("unit"), lane, job).start()
        s.update(ev.get("done"), total=ev.get("total"), eta_s=ev.get("eta_s"))
        return
    if kind == "start":
        if key not in steps:
            steps[key] = Step(w, name, ev.get("total"), ev.get("unit"), lane, job).start()
        return
    if kind in ("end", "fail"):
        s = steps.pop(key, None) or Step(w, name, ev.get("total"), ev.get("unit"), lane, job)
        if kind == "end":
            if ev.get("total") is not None:
                s.total = ev["total"]
            s.end(done=ev.get("done"), msg=ev.get("msg"), **(ev.get("extra") or {}))
        else:
            s.fail(rc=ev.get("rc"), msg=ev.get("msg"))
        return
    w.emit(kind, name, job=job, lane=lane, msg=ev.get("msg"), eta_s=ev.get("eta_s"), steps=ev.get("steps"),
           extra=ev.get("extra"))


# ---------------------------------------------------------------------------------------------------- readers


def read_jsonl(path: str | os.PathLike, offset: int = 0, max_bytes: int | None = None) -> tuple[list[dict], int]:
    """Events from byte `offset`; an incomplete last line is left for the next read (offset stops before it).
    Opens, reads and closes at once (never holds a handle a writer's os.replace could trip over)."""
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            fh.seek(offset)
            data = fh.read() if max_bytes is None else fh.read(max_bytes)
    except OSError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    out = []
    for raw in data[:end].split(b"\n"):
        if not raw.strip():
            continue
        try:
            ev = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(ev, dict) and ev.get("v") == V and "kind" in ev:
            out.append(ev)
    return out, offset + end + 1


def read_json(path: str | os.PathLike, retries: int = 3, wait_s: float = 0.1):
    """JSON written with truncate-in-place (state.json, ledger.json): retry a partial read, None when absent/bad."""
    p = Path(path)
    for i in range(retries):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (ValueError, OSError):
            if i < retries - 1:
                time.sleep(wait_s)
    return None


def write_json_atomic(path: str | os.PathLike, obj, retries: int = 10) -> bool:
    """tmp + os.replace, retried on PermissionError (a Windows reader holding the file)."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
        for i in range(retries):
            try:
                os.replace(tmp, p)
                return True
            except PermissionError:
                time.sleep(0.05 * (i + 1))
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return False


def sid_date(sid: str) -> tuple[int, int, int] | None:
    m = re.match(r"^(\d{4})(\d\d)(\d\d)-", sid or "")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def parse_runner_log(path: str | os.PathLike, offset: int = 0, parser: LineParser | None = None,
                     date0: tuple[int, int, int] | None = None, rx: float | None = None,
                     max_bytes: int | None = None) -> tuple[list[dict], int, LineParser]:
    """L0: events from a session's runner.log from byte `offset` (complete lines only). Keep the returned parser and
    offset to continue incrementally; `rx` stamps the new lines with their receive time (live monitoring)."""
    p = Path(path)
    if parser is None:
        parser = LineParser(RUNNER_LOG_ADAPTERS, date0=date0 or sid_date(p.parent.name))
    try:
        with open(p, "rb") as fh:
            fh.seek(offset)
            data = fh.read() if max_bytes is None else fh.read(max_bytes)
    except OSError:
        return [], offset, parser
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset, parser
    out = []
    for raw in data[:end].split(b"\n"):
        out += parser.feed(raw.decode("utf-8", "replace"), rx=rx)
    return out, offset + end + 1, parser


# ---------------------------------------------------------------------------------------------------- CLI


def _session_dir(sid: str) -> Path:
    from .remote_jobs import SESSIONS
    p = Path(sid)
    return p if p.is_dir() else SESSIONS / sid


def _fmt_s(s) -> str:
    if s is None:
        return "?"
    s = max(0.0, float(s))
    return f"{s / 3600:.1f} h" if s >= 5400 else (f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f} s")


def _print_snapshot(snap: dict) -> None:
    eta, cost = snap.get("eta") or {}, snap.get("cost") or {}
    pod = snap.get("pod") or {}
    print(f"session {snap['sid']}  [{snap.get('level')}]  state {snap.get('state')}  "
          f"pod {pod.get('id') or '-'} {pod.get('gpu') or ''} {pod.get('connection') or ''}")
    print(f"  now: {snap.get('current') or '-'}")
    if eta:
        print(f"  ETA: {_fmt_s(eta.get('p50'))} (p10 {_fmt_s(eta.get('p10'))} .. p90 {_fmt_s(eta.get('p90'))})"
              + (f"  [{eta['branches']}]" if eta.get("branches") else ""))
    if snap.get("kill"):
        k = snap["kill"]
        print(f"  kill switch in {_fmt_s(k.get('left_s'))}; margin at p90 {_fmt_s(k.get('margin_p90_s'))} "
              f"({k.get('level')})")
    if cost:
        print(f"  cost: so far ${cost.get('so_far', 0):.2f}  p50 ${cost.get('p50', 0):.2f}  "
              f"p90 ${cost.get('p90', 0):.2f}  worst ${cost.get('worst', 0):.2f}")
    for ph in snap.get("phases", []):
        bar = ""
        if ph.get("fraction") is not None:
            n = int(round(20 * ph["fraction"]))
            bar = "[" + "#" * n + "." * (20 - n) + "]"
        rem = ph.get("remaining") or {}
        print(f"   {ph['status'][:7]:<7} {ph['name']:<28} {bar:<22} "
              f"{_fmt_s(ph.get('elapsed_s')) if ph.get('elapsed_s') is not None else '':>7} "
              f"{('left ' + _fmt_s(rem.get('p50'))) if rem and ph['status'] != 'done' else ''}"
              f"{('  ' + ph['detail']) if ph.get('detail') else ''}")
    for w in snap.get("warnings", []):
        print(f"  [{w['level']}] {w['msg']}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m dancecap.progress", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tail", help="events of a session (L0 parse of runner.log + L1/L2 files)")
    t.add_argument("sid")
    t.add_argument("--kinds", default="plan,start,end,fail,note,metric")
    e = sub.add_parser("eta", help="snapshot as JSON (what the monitor shows)")
    e.add_argument("sid")
    e.add_argument("--at", help="replay: the snapshot as of this local time (HH:MM[:SS]) or epoch")
    e.add_argument("--text", action="store_true")
    wa = sub.add_parser("watch", help="live terminal view (read-only)")
    wa.add_argument("sid")
    wa.add_argument("--every", type=float, default=5.0)
    wa.add_argument("--count", type=int, default=0)
    sub.add_parser("budget", help="spent / projected / worst case / warnings")
    r = sub.add_parser("rates", help="history of rates from every past session")
    r.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    from . import progress_eta as E
    if a.cmd == "tail":
        d = _session_dir(a.sid)
        kinds = set(a.kinds.split(","))
        for ev in E.session_events(d):
            if ev.get("kind") in kinds:
                ts = ev.get("ts")
                print(f"{time.strftime('%H:%M:%S', time.localtime(ts)) if ts else '  ?     '} {ev['kind']:<6} "
                      f"{ev.get('job', '')}/{ev.get('step', '')}{'@' + str(ev['lane']) if ev.get('lane') else ''} "
                      f"{'' if ev.get('done') is None else str(ev.get('done')) + '/' + str(ev.get('total'))} "
                      f"{ev.get('msg') or ''}")
    elif a.cmd == "eta":
        at = E.parse_at(a.at, _session_dir(a.sid).name) if a.at else None
        snap = E.snapshot(_session_dir(a.sid), at=at)
        if a.text:
            _print_snapshot(snap)
        else:
            print(json.dumps(snap, indent=1, default=str))
    elif a.cmd == "watch":
        tr = E.Tracker(_session_dir(a.sid))
        n = 0
        while True:
            snap = tr.snapshot()
            print(time.strftime("%H:%M:%S"), "-" * 60)
            _print_snapshot(snap)
            n += 1
            if (a.count and n >= a.count) or snap.get("state") in ("ended", "failed", "dry_run"):
                break
            time.sleep(a.every)
    elif a.cmd == "budget":
        print(json.dumps(E.budget(), indent=1, default=str))
    elif a.cmd == "rates":
        if a.rebuild:
            doc = E.rates_rebuild()
        else:
            doc = E.load_rates()
        print(json.dumps({k: v for k, v in doc.items() if k != "keys"}, indent=1))
        for k, v in sorted(doc.get("keys", {}).items()):
            print(f"  {k}: " + json.dumps({x: y for x, y in v.items() if x not in ("samples", "profile")}))


if __name__ == "__main__":
    main()
