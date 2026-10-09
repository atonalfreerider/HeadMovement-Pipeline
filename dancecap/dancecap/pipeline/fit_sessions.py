"""Read-only adapter: the live `dancecap.fit_full` MAMMA fit sessions as farm pods + chunk jobs.

The four fits of a full dance (segments a-d) run on ordinary pods through `python -m dancecap.fit_full`, NOT as farm
jobs. The Pipeline Monitor still has to show them next to the farm: this module reads the same files that
`fit_full.status()` reads (work/<dance>/fit/runner_<seg>.json + commitments.json, the newest
work/runpod/sessions/*-fit-<seg>/state.json, its progress/puller.json heartbeat, the progress_eta snapshot for the ETA) and
turns them into plain dicts. It never writes, never starts or stops anything, never touches a pod, and it opens every file,
reads it and closes it at once (the runners rewrite these files all the time; a held handle would break their os.replace).

    live_fits(dance)  ->  {"dance", "segments": {seg: Segment}, "pods": [Pod], "found": n}

Segment = {seg, dance, take, sid, session_state, status, step, jobs: [{name, status, step, detail}], runner: {pid, alive},
           pod: {...} | None, progress: {frac, basis}, eta: {p10, p50, p90} | None, heartbeat_ts, heartbeat_age_s,
           started, worst_usd, error, estimate_min}
`status` is a farm chunk status (queued | running | done | failed) so the monitor can treat it like a chunk job.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from . import paths
from .util import SID_RE, mtime, pid_alive, read_json, redact

FIT_JOBS = ("fit", "refine", "quality")          # fit_full.JOBS, in order
SESSION_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-fit-([a-z])$")
HEARTBEAT_STALE_S = 90.0


def _sessions_by_seg(limit_days: float = 14.0) -> dict[str, list[Path]]:
    """{seg: [session dirs, oldest first]} for the sessions named <stamp>-fit-<seg>."""
    d = paths.sessions_dir()
    out: dict[str, list[Path]] = {}
    try:
        names = sorted(p.name for p in d.iterdir() if p.is_dir())
    except OSError:
        return out
    cutoff = time.time() - limit_days * 86400
    for n in names:
        m = SESSION_RE.match(n)
        if not m or not SID_RE.match(n):
            continue
        p = d / n
        mt = mtime(p / "state.json")
        if mt is None or mt < cutoff:
            continue
        out.setdefault(m.group(1), []).append(p)
    return out


def _take_of(state: dict) -> str | None:
    for j in (state.get("jobs") or {}).values():
        t = (j.get("spec") or {}).get("take")
        if t:
            return str(t)
    return None


def _job_step(job: dict) -> str | None:
    cur = job.get("current")
    if isinstance(cur, dict):
        cur = cur.get("step") or cur.get("name")
    return str(cur) if cur else None


def _job_status(raw) -> str:
    s = str(raw or "pending")
    if s in ("ok", "done"):
        return "done"
    if s.startswith("failed"):
        return "failed"
    if s.startswith("skipped"):
        return "skipped"
    return s if s in ("pending", "running") else "pending"


def _snapshot(sid: str) -> dict:
    """progress_eta snapshot (phases, ETA, kill margin) through the monitor's existing reader; {} when unavailable."""
    try:
        from . import runpod_view
        return runpod_view.session_snapshot(sid) or {}
    except Exception:  # noqa: BLE001 - a session being written: no ETA this time
        return {}


def _started(state: dict, sid: str) -> float | None:
    tm = state.get("timings") or []
    if tm and isinstance(tm[0], dict) and tm[0].get("start"):
        return float(tm[0]["start"])
    try:
        from datetime import datetime
        return datetime.strptime(sid[:15], "%Y%m%d-%H%M%S").timestamp()
    except ValueError:
        return None


def _segment(seg: str, sdir: Path, fit_dir: Path | None, now: float, commit: dict) -> dict | None:
    sid = sdir.name
    state = read_json(sdir / "state.json", None)
    if not isinstance(state, dict):
        return None
    take = _take_of(state)
    dance = take.rsplit("_", 1)[0] if take and take.endswith(f"_{seg}") else None
    jobs_raw = state.get("jobs") or {}
    jobs = []
    for name in FIT_JOBS:
        j = jobs_raw.get(name) or {}
        jobs.append({"name": name, "status": _job_status(j.get("status")), "step": _job_step(j),
                     "detail": redact(str(j.get("status") or ""))[:120] if _job_status(j.get("status")) in ("failed", "skipped") else None})
    pod = state.get("pod") or {}
    runner = None
    if fit_dir is not None:
        ri = read_json(fit_dir / f"runner_{seg}.json", None)
        if isinstance(ri, dict):
            # a resumed session runs in another process: fit_full records it in commitments.json
            cpid = (commit.get(seg) or {}).get("pid")
            runner = {"pid": ri.get("pid"), "alive": pid_alive(ri.get("pid")) or pid_alive(cpid),
                      "started": ri.get("started"), "order": ri.get("order")}
    res_m, state_m = mtime(sdir / "result.json"), mtime(sdir / "state.json")
    finished = res_m is not None and state_m is not None and res_m >= state_m - 1
    statuses = [j["status"] for j in jobs]
    running = "running" in statuses
    failed = "failed" in statuses or bool(state.get("error"))
    if all(s == "done" for s in statuses):
        status, sstate = "done", "ok"
    elif failed and not running:
        status, sstate = "failed", "failed"
    elif running or (pod.get("id") and not finished):
        status, sstate = "running", "running"
    elif (runner and runner["alive"]) or not finished:
        status, sstate = "queued", "preparing"      # bundling / waiting for a pod: no step is running yet
    else:
        status, sstate = "failed", "ended"
    # the step: the running job's current step, else the last timing step (upload / extract / bootstrap ...)
    step = next((f"{j['name']}/{j['step']}" if j["step"] else j["name"] for j in jobs if j["status"] == "running"), None)
    if step is None:
        tm = state.get("timings") or []
        last = tm[-1].get("step") if tm and isinstance(tm[-1], dict) else None
        step = last or ("waiting for a pod" if not pod.get("id") else None)
    snap = _snapshot(sid) if status in ("running", "queued") else {}
    cur = snap.get("current")
    cur = cur.get("step") if isinstance(cur, dict) else cur
    if status == "running" and cur:
        step = re.sub(r"\s*\(.*$", "", str(cur)).strip() or step     # 'fit/fit_c8 ( (no history...))' -> 'fit/fit_c8'
    eta = snap.get("eta") if isinstance(snap.get("eta"), dict) else None
    started = _started(state, sid)
    prog = None
    if status == "done":
        prog = {"frac": 1.0, "basis": "all jobs done"}
    elif eta and eta.get("p50") is not None and started:
        el = max(0.0, now - started)
        rem = max(0.0, float(eta["p50"]))
        prog = {"frac": round(el / (el + rem), 3) if el + rem > 0 else None,
                "basis": "elapsed / (elapsed + ETA p50): an estimate, MAMMA's fit has no step counter"}
    pul = read_json(sdir / "progress" / "puller.json", None) if (sdir / "progress").exists() else None
    hb = float(pul["last_ok"]) if isinstance(pul, dict) and pul.get("last_ok") else None
    gpu = (pul or {}).get("gpu") if isinstance(pul, dict) else None
    gpu_util = gpu[0].get("util") if isinstance(gpu, list) and gpu and isinstance(gpu[0], dict) else None
    price = pod.get("cost_per_hr")
    cost = None
    summ = {}
    try:
        from . import runpod_view
        summ = runpod_view.session_summary(sid) or {}
        cost = round(summ["cost_usd"], 2) if summ.get("cost_usd") is not None else None
    except Exception:  # noqa: BLE001
        pass
    podrec = None
    if pod.get("id"):
        created = pod.get("created_ts")
        pod_state = "running" if summ.get("state") == "running" else ("ended" if summ.get("state") in ("ok", "ended", "failed") else "starting")
        podrec = {"id": pod["id"], "name": pod.get("name") or f"fit-{seg}", "gpu": pod.get("gpu"),
                  "price_hr": float(price) if price is not None else None, "created_ts": created,
                  "age_s": round(now - float(created)) if created else None, "cost_usd": cost, "state": pod_state,
                  "kill_at": snap.get("kill_at"), "gpu_util": gpu_util}
    return {"seg": seg, "dance": dance, "take": take, "sid": sid, "session_state": sstate, "status": status, "step": step,
            "jobs": jobs, "runner": runner, "pod": podrec, "progress": prog,
            "eta": ({k: eta.get(k) for k in ("p10", "p50", "p90")} if eta else None),
            "eta_basis": (eta or {}).get("basis"), "heartbeat_ts": hb,
            "heartbeat_age_s": round(now - hb) if hb else None,
            "heartbeat_stale": bool(hb and now - hb > HEARTBEAT_STALE_S and status == "running"),
            "started": started, "worst_usd": (commit.get(seg) or {}).get("worst"),
            "error": redact(str(state.get("error") or ""))[:200] or None,
            "estimate_min": state.get("estimate_min"), "max_minutes": state.get("max_minutes"),
            "warnings": [str(w)[:200] for w in (snap.get("warnings") or [])][:4]}


def live_fits(dance: str | None = None, now: float | None = None) -> dict:
    """The newest fit_full session of every segment (optionally only those of `dance`). Safe to call every few seconds."""
    now = now or time.time()
    by_seg = _sessions_by_seg()
    fit_dirs: dict[str, Path] = {}
    try:
        for p in paths.work().glob("*/fit/runner_*.json"):
            fit_dirs[p.parent.parent.name] = p.parent
    except OSError:
        pass
    segs: dict[str, dict] = {}
    for seg, dirs in by_seg.items():
        sdir = dirs[-1]                       # fit_full._latest_session: the newest name
        rec_raw = read_json(sdir / "state.json", None)
        take = _take_of(rec_raw) if isinstance(rec_raw, dict) else None
        d = take.rsplit("_", 1)[0] if take and take.endswith(f"_{seg}") else None
        if dance and d != dance:
            continue
        fdir = fit_dirs.get(d) if d else None
        commit = read_json(fdir / "commitments.json", {}) if fdir else {}
        rec = _segment(seg, sdir, fdir, now, commit if isinstance(commit, dict) else {})
        if rec:
            segs[seg] = rec
    pods = [s["pod"] | {"seg": seg, "dance": s["dance"], "source": "fit_full", "sid": s["sid"]}
            for seg, s in sorted(segs.items()) if s.get("pod")]
    return {"dance": dance, "segments": segs, "pods": pods, "found": len(segs), "at": now}
