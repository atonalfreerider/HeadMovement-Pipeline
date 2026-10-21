"""What is running now, in one bundle: analysis runs, live pod sessions (uploads, phases, cost, ETA), recent sessions and
the budget. Read-only (files only, plus the runpod summary the monitor already holds). Used by GET /api/activity, the
analysis view (GET /api/analyses/<run>) and the snapshot renderer (snapshots.py).

Owner, 2026-10-21: "a dashboard that shows the status of uploads, analyses, cost and time ETAs ... maintained and running
every time we analyze anything". Before this bundle the Farm page (the defrag map) showed only the volume farm's chunk
jobs: the 95 remote-runner sessions of 2026-10-19..21 appeared nowhere on it.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import analyze, estimate, paths, runpod_view
from .util import SID_RE, mtime, read_json

RECENT_S = 24 * 3600
SCHEMA = "dancecap.monitor.activity/1"


# ------------------------------------------------------------------------------------------------ sessions

def upload_info(sid: str, st: dict | None = None) -> dict | None:
    """{total, sent, mbps, streams, done, eta_s, history[[t, MB]], source} of a session's bundle upload: the striped
    uploader's own counter (progress/upload.json, exact) or the puller's probe of the .part file (every 15 s)."""
    d = paths.sessions_dir() / sid
    st = st if st is not None else (read_json(d / "state.json", {}) or {})
    total = ((st.get("bundle") or {}).get("bytes"))
    up = read_json(d / "progress" / "upload.json", None)
    timing = next((t for t in st.get("timings") or [] if t.get("step") == "upload"), None)
    if isinstance(up, dict) and up.get("total"):
        total = up["total"]
        sent = int(up.get("sent") or 0)
        done = bool(up.get("done")) or bool(timing and timing.get("ok"))
        el = max(1e-3, float(up.get("ts") or time.time()) - float(up.get("t0") or time.time()))
        rate = sent / 1e6 / el
        hist = [[h[0], round(h[1] / 1e6, 2)] for h in up.get("history") or [] if isinstance(h, list) and len(h) == 2]
        return {"total": total, "sent": total if done else sent, "mbps": round(rate, 3), "streams": up.get("streams"),
                "done": done, "eta_s": None if done or rate <= 0 else round((total - sent) / 1e6 / rate),
                "history": hist[-120:], "source": "uploader", "seconds": (timing or {}).get("seconds")}
    if not total:
        return None
    pl = read_json(d / "progress" / "puller.json", None) or {}
    part = pl.get("upload_part_bytes")
    if timing:   # finished (ok or failed)
        secs = float(timing.get("seconds") or 0)
        return {"total": total, "sent": total if timing.get("ok") else part, "mbps": round(total / 1e6 / secs, 3)
                if secs and timing.get("ok") else None, "streams": 1, "done": bool(timing.get("ok")),
                "failed": not timing.get("ok"), "eta_s": None, "history": [], "source": "runner timings", "seconds": secs}
    if part is None:
        return {"total": total, "sent": None, "mbps": None, "streams": 1, "done": False, "eta_s": None, "history": [],
                "source": "no byte count yet"}
    start = next((t.get("start") for t in st.get("timings") or [] if t.get("step") == "upload"), None)
    pod_started = (st.get("pod") or {}).get("started")
    el = max(1.0, time.time() - float(start or pod_started or time.time()))
    rate = part / 1e6 / el
    return {"total": total, "sent": part, "mbps": round(rate, 3), "streams": 1, "done": bool(pl.get("upload_done")),
            "eta_s": round((total - part) / 1e6 / rate) if rate > 0 else None, "history": [], "source": "puller (15 s)"}


def session_card(sid: str, live: bool = True) -> dict | None:
    """One session as the dashboard draws it: state, the running phase, upload, ETA, cost (so far, expected, worst), pod,
    analysis tag, warnings."""
    s = runpod_view.session_summary(sid)
    if s is None:
        return None
    d = paths.sessions_dir() / sid
    st = read_json(d / "state.json", {}) or {}
    card = {"sid": sid, "state": s["state"], "take": s.get("take"), "jobs": s.get("jobs"), "jobs_status": s.get("jobs_status"),
            "jobs_bad": s.get("jobs_bad"), "gpu": s.get("gpu"), "price_hr": s.get("cost_per_hr"), "cost_usd": s.get("cost_usd"),
            "started": s.get("started"), "ended": s.get("ended"), "analysis": st.get("analysis"),
            "runner_alive": s.get("runner_alive"), "error": s.get("error"), "upload": upload_info(sid, st),
            "bundle_mb": round(((st.get("bundle") or {}).get("bytes") or 0) / 1e6, 1) or None}
    if live and s["state"] in ("running", "preparing", "orphaned"):
        try:
            sn = runpod_view.session_snapshot(sid) or {}
        except Exception:  # noqa: BLE001
            sn = {}
        run = [p for p in sn.get("phases") or [] if p.get("status") == "running"]
        card.update({"phase": run[0].get("name") if run else (sn.get("current") or {}).get("step")
                     if isinstance(sn.get("current"), dict) else sn.get("current"),
                     "phase_detail": run[0].get("detail") if run else None,
                     "phase_frac": run[0].get("fraction") if run else None,
                     "eta": sn.get("eta"), "cost": sn.get("cost"), "kill_at": sn.get("kill_at"),
                     "warnings": (sn.get("warnings") or [])[:4],
                     "phases": [{"name": p.get("name"), "status": p.get("status"), "elapsed_s": p.get("elapsed_s"),
                                 "fraction": p.get("fraction")} for p in sn.get("phases") or []][:40]})
    else:
        card["phases"] = ended_phases(st)
    return card


def ended_phases(st: dict) -> list[dict]:
    """The runner's timings with their REAL outcome: a step 'ok' in the timings only means it did not raise; a bootstrap
    with rc != 0 or a command with rc != 0 ran with check=False (2026-10-21: an 860 s failed bootstrap was drawn green)."""
    jobs = st.get("jobs") or {}
    tm = st.get("timings") or []
    boot_bad = st.get("bootstrap_rc") not in (None, 0)
    retried = any(x.get("step") == "bootstrap_retry" for x in tm)
    out = []
    for t in tm:
        name = str(t.get("step") or "")
        ok = bool(t.get("ok"))
        if name == "bootstrap" and (boot_bad or retried):     # a retry runs only after a failed first bootstrap
            ok = False
        elif name == "bootstrap_retry" and boot_bad:
            ok = False
        elif "/" in name:
            job, cmd = name.split("/", 1)
            rc = ((jobs.get(job) or {}).get("commands") or {}).get(cmd)
            if rc not in (None, 0):
                ok = False
        out.append({"name": name, "status": "done" if ok else "failed", "elapsed_s": t.get("seconds")})
    return out[:40]


def _session_dirs(limit: int = 60) -> list[str]:
    d = paths.sessions_dir()
    if not d.is_dir():
        return []
    return sorted((p.name for p in d.iterdir() if SID_RE.match(p.name)), reverse=True)[:limit]


def recent_sessions(now: float, limit: int = 12) -> list[dict]:
    out = []
    for sid in _session_dirs(40):
        s = runpod_view.session_summary(sid)
        if not s or s["state"] == "dry_run":
            continue
        t = s.get("started") or 0
        if now - float(t) > RECENT_S and out:
            break
        out.append(session_card(sid, live=False))
        if len(out) >= limit:
            break
    return [c for c in out if c]


# ------------------------------------------------------------------------------------------------ analyses

_cost_cache: dict[str, tuple[float, float]] = {}


def _run_cost(run: str) -> float:
    c = _cost_cache.get(run)
    if c and time.time() - c[0] < 20:
        return c[1]
    v = analyze.refresh_cost(run)
    _cost_cache[run] = (time.time(), v)
    return v


def analysis_card(doc: dict, now: float, sessions: bool = True) -> dict:
    run = doc["run"]
    doc = dict(doc)
    try:
        doc["cost_so_far_usd"] = _run_cost(run)
    except Exception:  # noqa: BLE001
        pass
    tagged = analyze.sessions_of(run) if sessions else []
    cards = [c for c in (session_card(t["sid"]) for t in tagged[:20]) if c]
    by_step: dict[str, list[str]] = {}
    for t in tagged:
        if t.get("step"):
            by_step.setdefault(t["step"], []).append(t["sid"])
    steps = []
    for s in doc.get("steps") or []:
        sids = sorted(set(by_step.get(s["id"], [])) | set(s.get("sessions") or []), reverse=True)
        live_sess = [c for c in cards if c["sid"] in sids and c["state"] in ("running", "preparing", "orphaned")]
        row = {k: s.get(k) for k in ("id", "label", "where", "lanes", "deps", "p50_min", "p90_min", "executor", "optional",
                                     "status", "started", "ended", "message", "progress", "basis", "note", "price_hr")}
        if live_sess and row["status"] in ("pending", "waiting_agent"):
            # an agent drives this step: its tagged pod sessions are running now, so the step is running (for the map,
            # the ETA and the counts), from its first session's start
            row["status"] = "running"
            row["started"] = min(float(c.get("started") or now) for c in live_sess)
            row["message"] = f"driven by an agent: {len(live_sess)} pod session(s) running"
        eta_s = None
        if row["status"] == "running" and live_sess:
            etas = [c["eta"]["p50"] for c in live_sess if (c.get("eta") or {}).get("p50") is not None]
            eta_s = max(etas) if etas else None
        steps.append({**row, "sessions": sids, "eta_s": eta_s, "cost_p50_usd": analyze.step_cost(s)})
    lv = analyze.live({**doc, "steps": steps}, now)
    return {"run": run, "take": doc.get("take"), "status": doc.get("status"), "created": doc.get("created"),
            "agent_hint": doc.get("agent_hint"),
            "updated": doc.get("updated"), "route": (doc.get("input") or {}).get("route"),
            "kind": (doc.get("input") or {}).get("kind"), "input": (doc.get("input") or {}).get("path"),
            "facts": doc.get("facts"), "plan": doc.get("plan"), "live": lv, "steps": steps, "sessions": cards,
            "driver_alive": analyze.driver_alive(doc), "has_farm": (paths.work() / doc.get("take", "") / "farm").is_dir()}


# ------------------------------------------------------------------------------------------------ snapshots index

def snapshot_runs(limit: int = 20) -> list[dict]:
    from . import snapshots
    return snapshots.list_runs(limit)


# ------------------------------------------------------------------------------------------------ the bundle

def bundle(rp: dict | None = None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    runs = analyze.list_runs(8)
    cards = []
    for d in runs:
        active = d.get("status") not in ("done", "failed", "cancelled") or now - float(d.get("updated") or 0) < 3600
        if active or len(cards) < 3:
            try:
                cards.append(analysis_card(d, now, sessions=active))
            except Exception as e:  # noqa: BLE001 - one broken run never hides the others
                cards.append({"run": d.get("run"), "take": d.get("take"), "status": "error", "error": str(e)[:200]})
    focus = next((c for c in cards if c.get("status") in ("running", "waiting", "planned")), None)
    act = runpod_view.active_sessions()
    tagged = {s["sid"] for c in cards for s in (c.get("sessions") or [])}
    live = [c for c in (session_card(s["sid"]) for s in act) if c]
    rp = rp or {}
    bud = rp.get("budget") or estimate.budget()
    acct = rp.get("account") or {}
    return {"schema": SCHEMA, "now": now, "analyses": cards, "focus": focus["run"] if focus else None,
            "focus_take": focus["take"] if focus else None,
            "focus_dance": focus["take"] if focus and focus.get("has_farm") else None,
            "sessions_live": live, "sessions_live_untagged": [c["sid"] for c in live if c["sid"] not in tagged],
            "sessions_recent": recent_sessions(now),
            "budget": {k: bud.get(k) for k in ("cap", "spent", "remaining", "level", "messages", "open_pods",
                                                "projected", "running", "source")},
            "cap_source": _cap_source(), "balance": acct.get("balance_usd"), "spend_per_hr": acct.get("spend_per_hr_usd"),
            "snapshots": snapshot_runs(8)}


def _cap_source() -> str | None:
    try:
        if paths.is_real_root():
            from .. import remote
            return remote.effective_cap().get("source")
    except Exception:  # noqa: BLE001
        pass
    return None


def analysis_detail(run: str, now: float | None = None) -> dict | None:
    now = time.time() if now is None else now
    doc = analyze.load(run)
    if doc is None:
        return None
    card = analysis_card(doc, now)
    from .util import read_jsonl
    card["events"] = read_jsonl(analyze.run_dir(run) / "events.jsonl", limit=60)[::-1][:40]
    from . import snapshots
    card["snapshots"] = snapshots.index(run).get("shots", [])[-60:]
    card["coverage"] = coverage_mini(doc.get("take"))
    return card


def coverage_mini(take: str | None) -> dict | None:
    """The take's farm coverage cells (the defrag map), compact: [state letter] per cell + counts."""
    if not take:
        return None
    from . import farm_view
    cov = farm_view.read_coverage(take)
    if not cov:
        return None
    cells = farm_view.compact_cells(cov)
    return {"cells": "".join({"complete": "c", "partial": "p", "none": "n"}[c["s"]] for c in cells),
            "stages": "".join({"front": "f", "fit": "t", "refined": "r", "analysed": "a"}.get(c["g"], "-") for c in cells),
            "cell_s": cov.get("cell_s") or 1.0, "updated": cov.get("updated"), "percent": (farm_view.counts_of(cells)
                                                                                          if cells else {}).get("percent")}
