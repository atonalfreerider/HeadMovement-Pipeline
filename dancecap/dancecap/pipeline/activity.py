"""What is running now, in one bundle: analysis runs, live pod sessions (uploads, phases, cost, ETA), recent sessions and
the budget. Read-only (files only, plus the runpod summary the monitor already holds). Used by GET /api/activity, the
analysis view (GET /api/analyses/<run>) and the snapshot renderer (snapshots.py).

Owner, 2026-10-21: "a dashboard that shows the status of uploads, analyses, cost and time ETAs ... maintained and running
every time we analyze anything". Before this bundle the Farm page (the defrag map) showed only the volume farm's chunk
jobs: the 95 remote-runner sessions of 2026-10-19..21 appeared nowhere on it.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from . import analyze, estimate, paths, runpod_view
from .util import SID_RE, mtime, read_json, redact

RECENT_S = 24 * 3600
RECENT_MAX = 120          # "Recent pod sessions (24 h)" lists every session of the last 24 h (17 on 2026-10-21), at most this
SCHEMA = "dancecap.monitor.activity/1"
LIVE = ("running", "preparing", "orphaned")
SLOW_X = 2.0              # a pod command at >= 2x the reference rate (frames.json slow_x) is flagged (runner: slow_gpu)


# ------------------------------------------------------------------------------------------------ sessions

def upload_info(sid: str, st: dict | None = None, session_state: str | None = None) -> dict | None:
    """{total, sent, mbps, streams, done, eta_s, history[[t, MB]], source} of a session's bundle upload: the striped
    uploader's own counter (progress/upload.json, exact) or the puller's probe of the .part file (every 15 s).
    A session that has ENDED (any state but running / preparing / orphaned) never shows an upload in progress: its counter
    stopped where it was (`stopped`, no ETA: 2026-10-21 review, "65/146 MB, 31 s left" on a session gone for an hour),
    and one that never got a pod has nothing to show here (`no_pod`: its error says why)."""
    d = paths.sessions_dir() / sid
    st = st if st is not None else (read_json(d / "state.json", {}) or {})
    ended = session_state is not None and session_state not in LIVE and session_over(sid, st, session_state)
    total = ((st.get("bundle") or {}).get("bytes"))
    up = read_json(d / "progress" / "upload.json", None)
    timing = next((t for t in st.get("timings") or [] if t.get("step") == "upload"), None)
    if session_state == "dry_run":
        return None   # a dry run uploads nothing by design
    if ended and not (st.get("pod") or {}).get("id") and not timing and not (isinstance(up, dict) and up.get("sent")):
        return {"total": total, "sent": None, "mbps": None, "streams": None, "done": False, "eta_s": None, "history": [],
                "no_pod": True, "source": "no pod started"} if total else None
    if isinstance(up, dict) and up.get("total"):
        total = up["total"]
        sent = int(up.get("sent") or 0)
        done = bool(up.get("done")) or bool(timing and timing.get("ok"))
        el = max(1e-3, float(up.get("ts") or time.time()) - float(up.get("t0") or time.time()))
        rate = sent / 1e6 / el
        hist = [[h[0], round(h[1] / 1e6, 2)] for h in up.get("history") or [] if isinstance(h, list) and len(h) == 2]
        stopped = ended and not done
        return {"total": total, "sent": total if done else sent, "mbps": round(rate, 3), "streams": up.get("streams"),
                "done": done, "stopped": stopped or None, "failed": bool(timing and not timing.get("ok")) or None,
                "eta_s": None if done or stopped or rate <= 0 else round((total - sent) / 1e6 / rate),
                "history": hist[-120:], "source": "uploader", "seconds": (timing or {}).get("seconds")}
    if not total:
        return None
    pl = read_json(d / "progress" / "puller.json", None) or {}
    part = pl.get("upload_part_bytes")
    if timing:   # finished (ok or failed)
        secs = float(timing.get("seconds") or 0)
        return {"total": total, "sent": total if timing.get("ok") else part, "mbps": round(total / 1e6 / secs, 3)
                if secs and timing.get("ok") else None, "streams": timing.get("streams") or 1, "done": bool(timing.get("ok")),
                "failed": not timing.get("ok"), "eta_s": None, "history": [], "source": "runner timings", "seconds": secs}
    if ended:
        return {"total": total, "sent": part, "mbps": None, "streams": 1, "done": False, "stopped": True, "eta_s": None,
                "history": [], "source": "puller (15 s)"}
    if part is None:
        return {"total": total, "sent": None, "mbps": None, "streams": 1, "done": False, "eta_s": None, "history": [],
                "source": "no byte count yet"}
    start = next((t.get("start") for t in st.get("timings") or [] if t.get("step") == "upload"), None)
    pod_started = (st.get("pod") or {}).get("started")
    el = max(1.0, time.time() - float(start or pod_started or time.time()))
    rate = part / 1e6 / el
    return {"total": total, "sent": part, "mbps": round(rate, 3), "streams": 1, "done": bool(pl.get("upload_done")),
            "eta_s": round((total - part) / 1e6 / rate) if rate > 0 else None, "history": [], "source": "puller (15 s)"}


def frames_info(sid: str, ended: bool = False) -> list[dict]:
    """The frame counters of a session's pod commands (progress/frames.json, written by the runner: remote_jobs.FrameProgress,
    fix round 2): [{key, job, cmd, step, stage, unit, done, total, rate_s, rate_avg_s, rate_unit, eta_s, state, gpu,
    slow_x, ref, slow}]. `slow`: the measured rate is >= SLOW_X times the reference GPU's (SAM 3D Body on an A100 at 2.0 s
    per person-frame against 0.355 s on a 4090 looked normal in the 2026-10-21 trial). An ended session's counters are
    history: no ETA."""
    fj = read_json(paths.sessions_dir() / sid / "progress" / "frames.json", None)
    if not isinstance(fj, dict) or not isinstance(fj.get("jobs"), dict):
        return []
    out = []
    for key, j in fj["jobs"].items():
        if not isinstance(j, dict):
            continue
        sx = j.get("slow_x")
        sx = float(sx) if isinstance(sx, (int, float)) and not isinstance(sx, bool) else None
        ref = j.get("ref") if isinstance(j.get("ref"), dict) else None
        state = str(j.get("state") or "running")
        out.append({"key": str(key), "job": j.get("job"), "cmd": j.get("cmd"), "step": j.get("step"),
                    "stage": j.get("stage"), "stages_total": j.get("stages_total"), "unit": j.get("unit") or "frames",
                    "done": j.get("done"), "total": j.get("total"), "rate_s": j.get("rate_s"),
                    "rate_avg_s": j.get("rate_avg_s"), "rate_unit": j.get("rate_unit") or j.get("unit") or "frames",
                    "eta_s": None if ended or state != "running" else j.get("eta_s"),
                    "state": "ended" if ended and state == "running" else state, "gpu": j.get("gpu") or fj.get("gpu"),
                    "slow_x": round(sx, 2) if sx is not None else None,
                    "ref": {k: ref.get(k) for k in ("rate_s", "rate_unit", "gpu", "source")} if ref else None,
                    "slow": bool(sx is not None and sx >= SLOW_X), "ts": j.get("ts")})
    out.sort(key=lambda x: -(float(x.get("ts") or 0)))
    return out[:12]


def _state_warnings(st: dict) -> list[dict]:
    """The runner's own warnings of a session (state.json `warnings`: gpu_choice when the session max or the budget ruled
    out a faster GPU, slow_gpu when the measured rate is >= 2x the reference): [{kind, level, msg}]."""
    out = []
    for w in st.get("warnings") or []:
        if isinstance(w, dict) and w.get("msg"):
            out.append({"kind": w.get("kind"), "level": w.get("level") or "warn", "msg": redact(str(w["msg"]))[:400]})
        elif isinstance(w, str):
            out.append({"kind": None, "level": "warn", "msg": redact(w)[:400]})
    return out[:6]


def _choice_info(st: dict) -> dict | None:
    """Why a session runs on the GPU it does, or why it got none: state pod.choice (a pod started) or gpu_choice (none
    did), reduced to {chosen, tried[{gpu, cloud, price, need, result}], warning, no_pod}."""
    pod = st.get("pod") if isinstance(st.get("pod"), dict) else {}
    ch = pod.get("choice") if isinstance(pod.get("choice"), dict) else None
    no_pod = False
    if ch is None and isinstance(st.get("gpu_choice"), dict):
        ch, no_pod = st["gpu_choice"], True
    if not ch:
        return None
    tried = [{k: t.get(k) for k in ("gpu", "cloud", "price", "need", "result")} for t in ch.get("tried") or []
             if isinstance(t, dict)]
    return {"chosen": ch.get("chosen"), "tried": tried[:24], "warning": ch.get("warning"), "no_pod": no_pod,
            "max_cost": ch.get("max_cost"), "max_minutes": ch.get("max_minutes")}


def session_over(sid: str, st: dict, state: str | None) -> bool:
    """Whether a session that runpod_view does not call live has really finished. 'unknown' is also what a session gets
    while it bundles and starts its pod when it is not the newest session dir (every -p1..p(n-1) session of `remote run
    --parallel`) or writes nothing for 5 min (gzip of a large bundle): the 2026-10-21 review saw those drawn as crit "no
    pod" with an alert. Over = result.json, or the runner recorded an error / that no GPU could be started, or a final
    state (ok / failed / ended / dry_run)."""
    if state in LIVE:
        return False
    if state != "unknown":
        return True
    return bool((paths.sessions_dir() / sid / "result.json").exists() or st.get("error") or st.get("gpu_choice"))


# runner warnings (state.json) that the session card carries as state_warnings: progress_eta copies them into its
# snapshot warnings too, and both copies became alerts (the review's synthetic run: 3 alerts for one slow GPU)
_STATE_WARNING_KINDS = ("gpu_choice", "slow_gpu")


def session_card(sid: str, live: bool = True) -> dict | None:
    """One session as the dashboard draws it: state, the running phase, upload, ETA, cost (so far, expected, worst), pod,
    analysis tag, warnings, frame counters, the GPU choice."""
    s = runpod_view.session_summary(sid)
    if s is None:
        return None
    d = paths.sessions_dir() / sid
    st = read_json(d / "state.json", {}) or {}
    over = session_over(sid, st, s["state"])
    ended = s["state"] not in LIVE and over
    choice = _choice_info(st)
    card = {"sid": sid, "state": s["state"], "take": s.get("take"), "jobs": s.get("jobs"), "jobs_status": s.get("jobs_status"),
            "jobs_bad": s.get("jobs_bad"), "gpu": s.get("gpu"), "price_hr": s.get("cost_per_hr"), "cost_usd": s.get("cost_usd"),
            "started": s.get("started"), "ended": s.get("ended"), "analysis": st.get("analysis"),
            "runner_alive": s.get("runner_alive"), "error": s.get("error"), "upload": upload_info(sid, st, s["state"]),
            "bundle_mb": round(((st.get("bundle") or {}).get("bytes") or 0) / 1e6, 1) or None,
            # no pod = the session FINISHED without one (not a dry run, not one still bundling / starting its pod)
            "no_pod": bool(ended and s["state"] != "dry_run" and not (st.get("pod") or {}).get("id")),
            "preparing": bool(s["state"] == "unknown" and not over) or None, "choice": choice,
            "frames": frames_info(sid, ended), "state_warnings": _state_warnings(st)}
    if live and s["state"] in LIVE:
        try:
            sn = runpod_view.session_snapshot(sid) or {}
        except Exception:  # noqa: BLE001
            sn = {}
        state_msgs = {str(w.get("msg"))[:400] for w in st.get("warnings") or [] if isinstance(w, dict) and w.get("msg")}
        run = [p for p in sn.get("phases") or [] if p.get("status") == "running"]
        card.update({"phase": run[0].get("name") if run else (sn.get("current") or {}).get("step")
                     if isinstance(sn.get("current"), dict) else sn.get("current"),
                     "phase_detail": run[0].get("detail") if run else None,
                     "phase_frac": run[0].get("fraction") if run else None,
                     "eta": sn.get("eta"), "cost": sn.get("cost"), "kill_at": sn.get("kill_at"),
                     # runpod_view flattens them to their text: match the runner's own messages by text too
                     "warnings": [({**w, "msg": redact(str(w.get("msg") or ""))[:400]} if isinstance(w, dict) else
                                   redact(str(w))[:400]) for w in sn.get("warnings") or []
                                  if not (isinstance(w, dict) and w.get("kind") in _STATE_WARNING_KINDS)
                                  and _wmsg(w)[:400] not in state_msgs][:6],
                     "phases": [{"name": p.get("name"), "status": p.get("status"), "elapsed_s": p.get("elapsed_s"),
                                 "fraction": p.get("fraction"), "optional": p.get("optional") or None,
                                 "rem": (p.get("remaining") or {}).get("p50") if isinstance(p.get("remaining"), dict) else None}
                                for p in sn.get("phases") or []][:60]})
        card["progress"] = session_progress(card)
    else:
        card["phases"] = ended_phases(st)
    return card


def session_progress(card: dict) -> dict | None:
    """How far a LIVE session is, from its phases: the time of the finished phases plus the running one's elapsed time,
    against that plus the model's remaining time of the open phases (progress_eta: frame counters, byte counts, history).
    {frac, detail}. None when the session is not running or has no phases yet. This is what the analysis map fills a
    running step with: real phases, never the step's elapsed time against its plan."""
    if card.get("state") not in LIVE or not card.get("phases"):
        return None
    done = left = 0.0
    n_done = n = 0
    for p in card["phases"]:
        stt = p.get("status")
        if stt == "skipped" or (p.get("optional") and stt == "pending"):
            continue
        n += 1
        el = float(p.get("elapsed_s") or 0.0)
        rem = float(p.get("rem") or 0.0)
        if stt in ("done", "failed"):
            done += el
            n_done += 1
        elif stt == "running":
            done += el
            left += rem
        else:
            left += rem
    if done + left <= 0:
        return None
    fr = next((f for f in card.get("frames") or [] if f.get("state") == "running" and f.get("total")), None)
    detail = f"{n_done} of {n} phases done"
    if fr:
        detail += f"; {fr['key']} {fr.get('done')}/{fr['total']} {fr.get('unit') or 'frames'}"
    return {"frac": round(done / (done + left), 4), "detail": detail}


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


def recent_sessions(now: float, limit: int = RECENT_MAX) -> list[dict]:
    """Every session of the last 24 h (newest first; at least the newest one when none is that recent). It stopped at 12
    before (the 2026-10-21 review counted 17)."""
    out = []
    for sid in _session_dirs(limit + 20):
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

_cost_cache: dict[str, tuple[float, dict]] = {}


def _run_cost(run: str) -> dict:
    """analyze.cost_detail(run) (every pod the run paid for, including those its sessions deleted themselves), cached 20 s.
    Falls back to refresh_cost's total on an older analyze."""
    c = _cost_cache.get(run)
    if c and time.time() - c[0] < 20:
        return c[1]
    if hasattr(analyze, "cost_detail"):
        v = analyze.cost_detail(run)
    else:
        v = {"total": analyze.refresh_cost(run), "by_step": {}, "pods": []}
    _cost_cache[run] = (time.time(), v)
    return v


def is_reused(s: dict) -> bool:
    """A step whose results an agent reused instead of computing them (analysis step ... done --reused). Runs from before
    that flag carry it only in their note: "REUSED, not computed: ..." (the 2026-10-21 trial's camera, people, shape)."""
    if s.get("reused") is not None:
        return bool(s["reused"])
    return s.get("status") == "done" and "REUSED" in f"{s.get('message') or ''} {s.get('note') or ''}"


STALL_RE = re.compile(r"no progress for|stall", re.I)


def session_eta_basis(live_sess: list[dict]) -> str:
    """How far the live sessions' ETA can be trusted (analyze.schedule): "timeout" = built from a job's timeout (it said 21
    min for a 3-min job: not used), "measured" = every session's ETA comes from progress_eta (measured phases), none from a
    generic prior and none stalled (then it may pull the step's time left in), else "session" (it can only push out)."""
    details = " ".join(str(c.get("phase_detail") or "") for c in live_sess)
    if "timeout" in details:
        return "timeout"
    measured = all(str((c.get("eta") or {}).get("basis") or "").startswith("progress_eta") for c in live_sess)
    stalled = any(STALL_RE.search(_wmsg(w)) for c in live_sess for w in c.get("warnings") or [])
    return "measured" if measured and "generic prior" not in details and not stalled else "session"


def step_progress(row: dict, live_sess: list[dict]) -> dict | None:
    """The REAL progress of a running step, when there is a source: its own byte / frame counter (the volume upload), else
    its live pod sessions' phases (session_progress: frame counters and phases, the slowest session counts). None = no
    progress figure: the map then shows elapsed time against the plan, drawn differently."""
    pr = row.get("progress") or {}
    if isinstance(pr, dict) and pr.get("total"):
        return {"frac": round(min(1.0, float(pr.get("done") or 0) / float(pr["total"])), 4),
                "basis": f"{pr.get('unit') or 'units'} counted", "source": "step"}
    ps = [c["progress"] for c in live_sess if c.get("progress") and c["progress"].get("frac") is not None]
    if ps:
        p = min(ps, key=lambda x: x["frac"])
        return {"frac": p["frac"], "basis": f"pod session phases ({p.get('detail')})", "source": "sessions"}
    return None


def analysis_card(doc: dict, now: float, sessions: bool = True) -> dict:
    run = doc["run"]
    doc = dict(doc)
    det = None
    try:
        det = _run_cost(run)
        doc["cost_so_far_usd"] = det["total"]
    except Exception:  # noqa: BLE001
        pass
    tagged = analyze.sessions_of(run) if sessions else []
    cards = [c for c in (session_card(t["sid"]) for t in tagged[:30]) if c]
    by_step: dict[str, list[str]] = {}
    for t in tagged:
        if t.get("step"):
            by_step.setdefault(t["step"], []).append(t["sid"])
    steps = []
    for s in doc.get("steps") or []:
        sids = sorted(set(by_step.get(s["id"], [])) | set(s.get("sessions") or []), reverse=True)
        live_sess = [c for c in cards if c["sid"] in sids and c["state"] in LIVE]
        row = {k: s.get(k) for k in ("id", "label", "where", "lanes", "deps", "p50_min", "p90_min", "executor", "optional",
                                     "status", "started", "ended", "message", "progress", "basis", "note", "price_hr",
                                     "reused", "compute_min", "overhead_min")}
        row["reused"] = is_reused(s)
        if doc.get("status") in ("done", "failed", "cancelled") and row["status"] == "running" and not live_sess:
            row["status"] = "stopped"   # the run ended under it: no time keeps counting, no ETA (cancelled verification run)
        if live_sess and row["status"] in ("pending", "waiting_agent"):
            # an agent drives this step: its tagged pod sessions are running now, so the step is running (for the map,
            # the ETA and the counts), from its first session's start
            row["status"] = "running"
            row["started"] = min(float(c.get("started") or now) for c in live_sess)
            row["message"] = f"driven by an agent: {len(live_sess)} pod session(s) running"
        eta_s = eta_p90 = basis = None
        if row["status"] == "running" and live_sess:
            etas = [c["eta"] for c in live_sess if (c.get("eta") or {}).get("p50") is not None]
            eta_s = max(e["p50"] for e in etas) if etas else None
            eta_p90 = max((e.get("p90") or e["p50"]) for e in etas) if etas else None
            basis = session_eta_basis(live_sess) if etas else None
        row["progress_frac"] = step_progress(row, live_sess) if row["status"] == "running" else None
        row["whole_step"] = s.get("whole_step", True)
        row["live_lanes"] = len(live_sess)
        steps.append({**row, "sessions": sids, "eta_s": eta_s, "eta_s_p90": eta_p90, "eta_basis": basis,
                      "cost_p50_usd": analyze.step_cost({**s, "reused": row["reused"]}),
                      "cost_usd": round(float((det or {}).get("by_step", {}).get(s["id"]) or 0.0), 4) if det else None})
    lv = analyze.live({**doc, "steps": steps}, now)
    for row in steps:   # the step row shows the time left the run's ETA uses (they contradicted each other: 92 s vs 396 s)
        if row["id"] in (lv.get("left") or {}):
            left = lv["left"][row["id"]]
            row.update(eta_session_s=row.get("eta_s"), eta_s=left["p50"], eta_s_p90=left["p90"])
    status = doc.get("status")
    if status in ("waiting", "planned") and any(s["status"] == "running" for s in steps):
        # the badge said "waiting" while an agent's step ran (its sessions live, or a `start` mark): the run is running
        status = "running"
    card = {"run": run, "take": doc.get("take"), "status": status, "created": doc.get("created"),
            "agent_hint": doc.get("agent_hint"),
            "updated": doc.get("updated"), "ended": analyze.run_end(doc) if hasattr(analyze, "run_end") else None,
            "route": (doc.get("input") or {}).get("route"),
            "kind": (doc.get("input") or {}).get("kind"), "input": (doc.get("input") or {}).get("path"),
            "facts": doc.get("facts"), "plan": doc.get("plan"), "live": lv, "steps": steps, "sessions": cards,
            "cost_by_step": (det or {}).get("by_step"), "pods": (det or {}).get("pods"),
            "max_cost_usd": doc.get("max_cost_usd"),
            "heartbeat_alive": _heartbeat_alive(run),
            "driver_alive": analyze.driver_alive(doc), "has_farm": (paths.work() / doc.get("take", "") / "farm").is_dir()}
    card["alerts"] = analysis_alerts(card, now)
    return card


def _heartbeat_alive(run: str) -> bool | None:
    try:
        return bool(analyze.heartbeat_alive(run)) if hasattr(analyze, "heartbeat_alive") else None
    except Exception:  # noqa: BLE001
        return None


def _wmsg(w) -> str:
    return str(w.get("msg") or w.get("message") or "") if isinstance(w, dict) else str(w or "")


def analysis_alerts(card: dict, now: float) -> list[dict]:
    """What needs the owner's eyes at the top of the analysis page (and the Farm strip), not only inside a session row:
    stalls and kill-switch margins of live sessions, a pod command far slower than its reference, a runner that is gone,
    a step past its plan's p90, failed steps, a session that got no pod while its step is still open.
    [{level: red|amber|info, id, message, link}]"""
    out: list[dict] = []
    run = card["run"]
    open_steps = {s["id"] for s in card.get("steps") or [] if s.get("status") not in ("done", "skipped")}
    for c in card.get("sessions") or []:
        sid = c["sid"]
        link = f"#/runpod/session/{sid}"
        step = (c.get("analysis") or {}).get("step") or "?"
        if c.get("state") in LIVE:
            if c["state"] == "orphaned":
                out.append({"level": "red", "id": f"orphan:{sid}", "link": link, "message":
                            f"{step}: session {sid} - its runner is gone while the pod bills until its kill switch"})
            for w in c.get("warnings") or []:
                msg = _wmsg(w)
                lvl = (w.get("level") if isinstance(w, dict) else None) or "amber"
                if not msg:
                    continue
                low = msg.lower()
                if "no progress" in low or "no new log line" in low or "kill switch" in low or "gpu idle" in low or \
                        "slower" in low or "orphan" in low or lvl == "red":
                    out.append({"level": "red" if lvl == "red" else "amber", "id": f"warn:{sid}:{msg[:40]}", "link": link,
                                "message": f"{step}: session {sid}: {msg}"})
            for f in c.get("frames") or []:
                if f.get("slow") and f.get("state") == "running":
                    ref = f.get("ref") or {}
                    out.append({"level": "amber", "id": f"slow:{sid}:{f['key']}", "link": link, "message":
                                f"{step}: {f['key']} runs {f['slow_x']:.1f}x slower than measured on "
                                f"{(ref.get('gpu') or 'the reference GPU').replace('NVIDIA ', '')} "
                                f"({f.get('rate_s')} s per {f.get('rate_unit')} against {ref.get('rate_s')} s)"})
        slow_frames = any(a["id"].startswith(f"slow:{sid}:") for a in out)   # one alert per slow GPU, not two
        for w in c.get("state_warnings") or []:
            if w.get("kind") == "slow_gpu" and c.get("state") in LIVE and not slow_frames:
                out.append({"level": "amber", "id": f"slowgpu:{sid}", "link": link, "message": f"{step}: {w['msg']}"})
        if c.get("no_pod") and step in open_steps and now - float(c.get("ended") or c.get("started") or 0) < 6 * 3600:
            out.append({"level": "amber", "id": f"nopod:{sid}", "link": link, "message":
                        f"{step}: session {sid} got no pod: {c.get('error') or 'no GPU could be started'}"})
    for s in card.get("steps") or []:
        if s.get("status") == "failed":
            out.append({"level": "red", "id": f"failed:{s['id']}", "link": f"#/analysis/{run}",
                        "message": f"step {s['id']} failed: {(s.get('message') or '')[:200]}"})
        elif s.get("status") == "running" and s.get("started") and s.get("p90_min"):
            el = (now - float(s["started"])) / 60.0
            if el > float(s["p90_min"]):
                out.append({"level": "amber", "id": f"overrun:{s['id']}", "link": f"#/analysis/{run}",
                            "message": f"step {s['id']} has run {el:.0f} min, past its plan's p90 of "
                                       f"{float(s['p90_min']):.0f} min (p50 {float(s.get('p50_min') or 0):.0f})"})
    seen, uniq = set(), []
    for a in sorted(out, key=lambda a: {"red": 0, "amber": 1, "info": 2}.get(a["level"], 3)):
        if a["id"] not in seen:
            seen.add(a["id"])
            uniq.append(a)
    return uniq[:12]


# ------------------------------------------------------------------------------------------------ snapshots index

def snapshot_runs(limit: int = 20) -> list[dict]:
    from . import snapshots
    return snapshots.list_runs(limit)


# ------------------------------------------------------------------------------------------------ the bundle

ACTIVE_RUN = ("running", "waiting", "planned")
FINISHED_SHOWN = 3        # finished runs listed when nothing runs (most recently finished first)


def _last_activity(d: dict) -> float:
    """When a run last did something: its end (analyze.run_end) for a finished run, else its last write."""
    try:
        end = analyze.run_end(d) if hasattr(analyze, "run_end") else None
    except Exception:  # noqa: BLE001
        end = None
    return float(end or d.get("updated") or d.get("created") or 0)


def pick_runs(runs: list[dict], now: float) -> list[tuple[dict, bool]]:
    """[(doc, active)] for the strip: every active run (and one that changed in the last hour), then the most recently
    FINISHED ones by their end time. Sorting by creation (before) let a later-created, earlier-cancelled run hide the mono
    trial that finished after it (2026-10-21 review)."""
    act, fin = [], []
    for d in runs:
        active = d.get("status") in ACTIVE_RUN or (d.get("status") not in ("done", "failed", "cancelled")) or \
            now - float(d.get("updated") or 0) < 3600
        (act if active else fin).append(d)
    act.sort(key=_last_activity, reverse=True)
    fin.sort(key=_last_activity, reverse=True)
    out = [(d, True) for d in act]
    out += [(d, False) for d in fin[:max(0, FINISHED_SHOWN - len(out))]]
    return out


def ledger_vs_runpod(bud: dict, rp: dict) -> dict:
    """The budget's pod counts kept apart (2026-10-21: "6 open pods" while none ran): rows OPEN in the ledger, of which the
    ones past their kill time (their pod is gone; `remote reconcile` closes them), against the pods Runpod LISTS now
    (dancecap's and others; None when the pod list was not read)."""
    pods = bud.get("pods") if isinstance(bud.get("pods"), list) else []
    n_open = bud.get("open_pods") if isinstance(bud.get("open_pods"), int) else len(pods)
    stale = [p.get("pod_id") for p in pods if p.get("phantom_cost") is not None]
    listed = rp.get("pods") if rp.get("connection") == "ok" else None
    return {"ledger_open": n_open, "ledger_stale": len(stale), "ledger_stale_ids": stale[:12],
            "runpod_listed": len(listed) if listed is not None else None,
            "runpod_ours": sum(1 for p in listed if p.get("managed")) if listed is not None else None,
            "runpod_checked": rp.get("checked")}


def bundle(rp: dict | None = None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    runs = analyze.list_runs(12)
    cards = []
    for d, active in pick_runs(runs, now):
        try:
            cards.append(analysis_card(d, now, sessions=active))
        except Exception as e:  # noqa: BLE001 - one broken run never hides the others
            cards.append({"run": d.get("run"), "take": d.get("take"), "status": "error", "error": str(e)[:200]})
    focus = next((c for c in cards if c.get("status") in ACTIVE_RUN), None)
    act = runpod_view.active_sessions()
    tagged = {s["sid"] for c in cards for s in (c.get("sessions") or [])}
    live = [c for c in (session_card(s["sid"]) for s in act) if c]
    rp = rp or {}
    bud = rp.get("budget") or estimate.budget()
    acct = rp.get("account") or {}
    alerts = [dict(a, run=c["run"]) for c in cards if c.get("status") in ACTIVE_RUN for a in c.get("alerts") or []]
    for c in live:   # untagged live sessions: their stalls are alerts too
        if c["sid"] in tagged:
            continue
        for w in c.get("warnings") or []:
            msg = _wmsg(w)
            if msg and any(k in msg.lower() for k in ("no progress", "no new log line", "kill switch", "gpu idle", "orphan")):
                alerts.append({"level": "amber", "id": f"warn:{c['sid']}:{msg[:40]}", "link": f"#/runpod/session/{c['sid']}",
                               "message": f"session {c['sid']}: {msg}"})
    return {"schema": SCHEMA, "now": now, "analyses": cards, "focus": focus["run"] if focus else None,
            "focus_take": focus["take"] if focus else None,
            # the Farm page opens on the take being analysed: its defrag map, or (a runner analysis) its step completion
            "focus_dance": focus["take"] if focus and focus.get("take") and (paths.work() / focus["take"]).is_dir()
            else None,
            "sessions_live": live, "sessions_live_untagged": [c["sid"] for c in live if c["sid"] not in tagged],
            "sessions_recent": recent_sessions(now), "alerts": alerts[:12],
            "budget": {**{k: bud.get(k) for k in ("cap", "spent", "remaining", "level", "messages", "open_pods",
                                                  "projected", "running", "source")}, **ledger_vs_runpod(bud, rp)},
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
    shots = snapshots.index(run).get("shots", [])
    card["snapshots"] = shots[-60:]
    card["snapshots_n"] = len(shots)   # the button said "Snapshots (60)" with 80 on disk: the list is cut, the count is not
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
