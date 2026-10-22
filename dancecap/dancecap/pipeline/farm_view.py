"""Pipeline Monitor, farm view data (docs/FARM.md section 7): ONE bundle per dance for the defrag map and its panels.

Reads only files, never starts or stops anything:
    work/<dance>/farm/coverage.json   per 1-s cell: complete | partial | none (+ stage, cameras, reasons, owning chunk)
    work/<dance>/farm/state.json      chunks, jobs, pods, spend, watcher, events (core's dancecap.farm.state schema)
    farm_volume.json                  the persistent Runpod volume (tracked, no secrets)
    the live `dancecap.fit_full` sessions through fit_sessions.py (read-only adapter): the four MAMMA fits of segments
    a-d appear as pods + chunk jobs and their chunks are drawn as "being worked on"
and the runpod summary the server already holds (budget, account balance, pods on the account).

No numpy / farm import: this module is imported by the monitor server at start-up, and by tests with a temp package root
(`paths.set_root`), so every path goes through `paths.work()`.

Bundle (`build()`), schema dancecap.monitor.farm/1:
    dance, dances[], now, coverage{exists, updated, counts, seconds, percent, stages, cells[]}, chunks[], jobs[], queue[],
    pods[], budget{}, volume{}, world{}, watcher{}, events[], alerts[], live{fit_full: n}, outputs{newest_mtime, stale}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import fit_sessions, paths
from .util import TAKE_RE, mtime, pid_alive, read_json, redact

SCHEMA = "dancecap.monitor.farm/1"
STAGES = ("none", "front", "fit", "refined", "analysed")
RUNNING = ("running", "starting")
DONE_STATES = ("done", "pulled", "stitched")
# where a downloaded fit / refine / quality result lands (work/<take>/...): their mtimes tell the monitor that
# coverage.json is out of date and `farm coverage` should run again
OUTPUT_DIRS = ("smplx", "smplx_rf2_noslip", "refine", "quality", "physics", "timing", "stitch.json")
FIT_DIRS_REL = ("mamma/fits",)
WATCHER_STALE_S = 180.0
DEMO_FLAG = "demo"
VOLUME_USAGE = "farm_volume_usage.json"       # work/monitor/: {used_gb, measured, source}; state.json["volume"] wins when newer


# ------------------------------------------------------------------------------------------------ paths and dances

def farm_dir(dance: str) -> Path:
    return paths.work() / dance / "farm"


def volume_file() -> Path:
    return paths.root() / "farm_volume.json"


def valid_dance(dance: str | None) -> bool:
    return bool(dance) and bool(TAKE_RE.match(dance))


def _is_demo(dance: str) -> bool:
    st = read_json(farm_dir(dance) / "state.json", None)
    return bool(isinstance(st, dict) and st.get(DEMO_FLAG))


ANALYSIS_OUTPUTS = ("smplx_rf2_noslip", "smplx", "timing", "physics", "counterbalance", "floorcraft", "export")


def analyses_by_take(limit: int = 30) -> dict[str, dict]:
    """{take: its newest pipeline analysis run (pipeline/analyze.py)}, reduced to {run, status, done, total, reused,
    ended, created, outputs[]}. The farm only knows its own volume-farm outputs; a take analysed by the remote runner
    (`analyze` + agents) has no coverage.json, and the Farm page called it "0 % complete, no coverage map" (2026-10-21
    review). `outputs`: which result folders exist under work/<take>/."""
    try:
        from . import analyze
        docs = analyze.list_runs(limit)
    except Exception:  # noqa: BLE001 - the analysis records never break the farm view
        return {}
    # per take: a run in flight, else its newest FINISHED (done) run, else its newest run. A later cancelled run (the
    # plan group's 4/12 verification run of trial_mono_jl20) must not hide the 12/12 analysis of the same take.
    rank = {"running": 0, "waiting": 0, "planned": 0, "done": 1}
    best: dict[str, dict] = {}
    for d in docs:   # newest run first
        t = d.get("take")
        if t and (t not in best or rank.get(d.get("status"), 2) < rank.get(best[t].get("status"), 2)):
            best[t] = d
    out: dict[str, dict] = {}
    for t, d in best.items():
        if not TAKE_RE.match(t):
            continue
        steps = [s for s in d.get("steps") or [] if not s.get("optional")]
        end = None
        try:
            end = analyze.run_end(d) if hasattr(analyze, "run_end") else None
        except Exception:  # noqa: BLE001
            pass
        base = paths.work() / t
        out[t] = {"run": d.get("run"), "status": d.get("status"), "created": d.get("created"), "ended": end,
                  "done": sum(1 for s in steps if s.get("status") in ("done", "skipped")), "total": len(steps),
                  "reused": [s.get("id") for s in d.get("steps") or [] if s.get("reused") or
                             (s.get("reused") is None and "REUSED" in str(s.get("message") or ""))],
                  "route": (d.get("input") or {}).get("route"),
                  "outputs": [o for o in ANALYSIS_OUTPUTS if (base / o).exists()]}
    return out


def list_dances(show_demo: bool = False) -> list[dict]:
    """Dances that have a farm directory with a coverage.json or a state.json, newest coverage first, and takes with a
    pipeline analysis run (runner analyses: their completion is the run's steps, not a farm map)."""
    out = []
    try:
        names = [p.name for p in paths.work().iterdir() if p.is_dir() and TAKE_RE.match(p.name)]
    except OSError:
        return out
    runs = analyses_by_take()
    for n in names:
        fd = farm_dir(n)
        cm, sm = mtime(fd / "coverage.json"), mtime(fd / "state.json")
        an = runs.get(n)
        if cm is None and sm is None and an is None:
            continue
        demo = _is_demo(n) if (cm is not None or sm is not None) else False
        if demo and not show_demo:
            continue
        cov = read_json(fd / "coverage.json", None) or {}
        pct = (cov.get("percent") or {}).get("complete")
        out.append({"id": n, "label": n + (" (demo)" if demo else ""), "demo": demo,
                    "updated_ts": max(cm or 0, sm or 0, float((an or {}).get("ended") or (an or {}).get("created") or 0)),
                    "complete_pct": pct if cm is not None else None, "has_coverage": cm is not None,
                    "farm": cm is not None or sm is not None,
                    "analysis": {k: an[k] for k in ("run", "status", "done", "total")} if an else None})
    out.sort(key=lambda d: d["updated_ts"], reverse=True)
    return out


def default_dance(show_demo: bool = False) -> str | None:
    """The dance with live fit sessions, else the one whose farm files changed last."""
    ds = list_dances(show_demo)
    if not ds:
        return None
    try:
        live = fit_sessions.live_fits()
        for s in live["segments"].values():
            if s.get("dance") and any(d["id"] == s["dance"] for d in ds) and s.get("status") in ("running", "queued"):
                return s["dance"]
    except Exception:  # noqa: BLE001
        pass
    # the defrag map is this page's subject: a dance with farm files first; a runner-only take (no map) only when none has
    return next((d["id"] for d in ds if d.get("farm", True)), ds[0]["id"])


# ------------------------------------------------------------------------------------------------ coverage

_cov_cache: dict[str, tuple[tuple, dict]] = {}


def read_coverage(dance: str) -> dict | None:
    """coverage.json (core's schema dancecap.farm.coverage/1), else another workflow's coverage_cells.json (same cell
    format). The parse is cached by (path, mtime, size)."""
    for p in (farm_dir(dance) / "coverage.json", paths.work() / dance / "coverage_cells.json"):
        try:
            st = p.stat()
        except OSError:
            continue
        key = (str(p), st.st_mtime_ns, st.st_size)
        c = _cov_cache.get(str(p))
        if c and c[0] == key:
            return c[1]
        d = read_json(p, None)
        if isinstance(d, dict) and isinstance(d.get("cells"), list):
            d = dict(d)
            d["_path"], d["_mtime"] = str(p), st.st_mtime
            _cov_cache[str(p)] = (key, d)
            return d
    return None


def _num(x, default=None):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x else default


def _norm_state(s) -> str:
    s = str(s or "none").lower()
    return s if s in ("complete", "partial", "none") else ("complete" if s in ("done", "ok") else "none")


def _norm_stage(s) -> str:
    s = str(s or "none").lower()
    return s if s in STAGES else ("refined" if s.startswith("refine") else "none")


def compact_cells(cov: dict) -> list[dict]:
    """The wire shape of the map: {i, t0, t1, k0, k1, s, g, n, m, f, r[], c}. Missing fields get safe defaults."""
    out = []
    for i, c in enumerate(cov.get("cells") or []):
        if not isinstance(c, dict):
            continue
        out.append({"i": int(c.get("i", i)), "t0": _num(c.get("t0")), "t1": _num(c.get("t1")),
                    "k0": _num(c.get("k0")), "k1": _num(c.get("k1")), "s": _norm_state(c.get("state")),
                    "g": _norm_stage(c.get("stage")), "n": _num(c.get("n_cams")), "m": _num(c.get("max_cams")),
                    "f": c.get("flag"), "r": [str(x)[:120] for x in (c.get("reasons") or [])][:6],
                    "c": c.get("chunk") or None})
    return out


def counts_of(cells: list[dict], cell_s: float = 1.0) -> dict:
    cnt = {"complete": 0, "partial": 0, "none": 0}
    sec = {"complete": 0.0, "partial": 0.0, "none": 0.0}
    for c in cells:
        cnt[c["s"]] += 1
        dur = (c["t1"] - c["t0"]) if c["t0"] is not None and c["t1"] is not None else cell_s
        sec[c["s"]] += dur
    n = max(1, len(cells))
    tot = sum(sec.values()) or 1.0
    return {"counts": cnt, "seconds": {k: round(v, 1) for k, v in sec.items()},
            "percent": {k: round(100.0 * v / tot, 1) for k, v in sec.items()}, "n": n}


# ------------------------------------------------------------------------------------------------ chunks

def _time_of_k(cov: dict, k) -> float | None:
    if k is None or cov.get("anchor") is None or not cov.get("hz"):
        return None
    return float(cov["anchor"]) + float(k) / float(cov["hz"])


def _chunk_status(state_status, live_status) -> str:
    if live_status:
        return live_status
    return str(state_status or "planned")


def _merge_chunks(cov: dict, state: dict, live: dict, now: float) -> list[dict]:
    """Chunks from state.json (richest), else coverage.json's list; the live fit_full segments override the status of the
    chunk they fit. Sorted by their first frame."""
    by_id: dict[str, dict] = {}
    for c in cov.get("chunks") or []:
        if isinstance(c, dict) and c.get("id"):
            by_id[str(c["id"])] = dict(c)
    for cid, c in (state.get("chunks") or {}).items():
        if isinstance(c, dict):
            by_id[str(cid)] = {**by_id.get(str(cid), {}), **c, "id": str(cid)}
    for seg, ls in (live.get("segments") or {}).items():
        if seg in by_id:
            by_id[seg]["live"] = ls
    out = []
    for cid, c in by_id.items():
        ls = c.pop("live", None)
        status = _chunk_status(c.get("status"), ls["status"] if ls else None)
        k0, k1 = c.get("core_k0", c.get("k0")), c.get("core_k1", c.get("k1"))
        row = {"id": cid, "kind": c.get("kind"), "origin": c.get("origin"), "status": status,
               "state_status": c.get("status"), "k0": c.get("k0"), "k1": c.get("k1"), "core_k0": k0, "core_k1": k1,
               "t0": _time_of_k(cov, k0), "t1": _time_of_k(cov, (k1 + 1) if k1 is not None else None),
               "core_s": c.get("core_s"), "source": "fit_full" if ls else "farm",
               "pod": ls["pod"]["id"] if ls and ls.get("pod") else c.get("pod"),
               "stages_done": c.get("stages_done") or [], "estimate": c.get("estimate"), "take": c.get("take")}
        if row["t0"] is not None and row["t1"] is not None and row["core_s"] is None:
            row["core_s"] = round(row["t1"] - row["t0"], 1)
        out.append(row)
    out.sort(key=lambda r: (r["core_k0"] if isinstance(r["core_k0"], (int, float)) else 1 << 40, r["id"]))
    return out


def _annotate_cells(cells: list[dict], chunks: list[dict], cov: dict) -> None:
    """Owner chunk, overlap chunks and the 'being worked on' flag of every cell, from the chunks' frame ranges."""
    hz = float(cov.get("hz") or 30.0)
    for c in cells:
        k0, k1 = c.get("k0"), c.get("k1")
        if k0 is None or k1 is None:
            k0 = k1 = None
        owner, best, over, live = c.get("c"), 0, [], False
        for ch in chunks:
            ck0, ck1 = ch.get("core_k0"), ch.get("core_k1")
            wk0, wk1 = ch.get("k0"), ch.get("k1")
            if k0 is not None and isinstance(ck0, (int, float)) and isinstance(ck1, (int, float)):
                ov = min(k1, ck1) - max(k0, ck0) + 1
                if ov > best:
                    best, owner = ov, ch["id"]
                if ov > 0 and ch["status"] in RUNNING:
                    live = True
                elif isinstance(wk0, (int, float)) and isinstance(wk1, (int, float)) and min(k1, wk1) - max(k0, wk0) + 1 > 0 and ov <= 0:
                    over.append(ch["id"])
            elif ch["id"] == owner and ch["status"] in RUNNING:
                live = True
        c["c"], c["o"], c["live"] = owner, over, live


# ------------------------------------------------------------------------------------------------ jobs, queue, pods

def _frac(p) -> float | None:
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        return float(p) if p <= 1.0 else float(p) / 100.0
    if isinstance(p, dict):
        if p.get("frac") is not None:
            return _frac(p["frac"])
        if p.get("total"):
            return float(p.get("done") or 0) / float(p["total"])
    return None


def _ts(x) -> float | None:
    """epoch seconds from an epoch or an ISO local time string (state.json writes both)."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str) and x:
        try:
            return time.mktime(time.strptime(x[:19], "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            return None
    return None


_spec_cache: dict[str, tuple[float, list[str]]] = {}


def _spec_step_names(dance: str, jid: str) -> list[str]:
    p = farm_dir(dance) / "jobs" / f"{jid}.json"
    m = mtime(p)
    if m is None:
        return []
    c = _spec_cache.get(str(p))
    if c and c[0] == m:
        return c[1]
    spec = read_json(p, {}) or {}
    names = [str(x.get("name")) for x in (spec.get("steps") or []) if isinstance(x, dict) and x.get("name")]
    _spec_cache[str(p)] = (m, names)
    return names


def _farm_job_steps(ch: dict, kind: str | None, names_override: list[str] | None = None) -> list[dict]:
    si, sn = ch.get("step_i"), ch.get("steps_n")
    if names_override and isinstance(si, int) and sn == len(names_override) and len(names_override) <= 7:
        return [{"name": n, "status": "done" if i < si else "running" if i == si and ch.get("status") == "running" else "pending",
                 "step": None} for i, n in enumerate(names_override)]
    names = ["front", "fit", "refine"] if (kind or "full") in ("full",) else [kind or "fit"]
    done = set(ch.get("stages_done") or [])
    step = str(ch.get("step") or "")
    run = next((n for n in names if n in step), None)
    out = []
    for n in names:
        alias = {"refine": "refined"}.get(n, n)
        st = "done" if (n in done or alias in done) else ("running" if n == run else "pending")
        out.append({"name": n, "status": st, "step": None})
    if not run and ch.get("status") == "running":
        nxt = next((s for s in out if s["status"] == "pending"), None)
        if nxt:
            nxt["status"] = "running"
    return out


def _jobs_and_queue(dance: str, chunks: list[dict], state: dict, live: dict, now: float) -> tuple[list[dict], list[dict]]:
    jobs, queue = [], []
    sjobs = state.get("jobs") or {}
    for ch in chunks:
        cid = ch["id"]
        ls = (live.get("segments") or {}).get(cid)
        sc = (state.get("chunks") or {}).get(cid) or {}
        sj = sjobs.get(cid) or {}
        if ls and ch["source"] == "fit_full":
            eta = ls.get("eta")
            jobs.append({"id": cid, "chunk": cid, "source": "fit_full", "kind": "fit",
                         "label": f"fit_full session {ls['sid'][-12:]}", "status": ls["status"], "step": ls["step"],
                         "steps": [{"name": j["name"], "status": j["status"], "step": j["step"], "detail": j["detail"]}
                                   for j in ls["jobs"]],
                         "progress": ls["progress"], "eta": eta, "heartbeat_ts": ls.get("heartbeat_ts"),
                         "heartbeat_age_s": ls.get("heartbeat_age_s"), "heartbeat_stale": ls.get("heartbeat_stale"),
                         "pod": ls["pod"]["id"] if ls.get("pod") else None, "gpu": ls["pod"]["gpu"] if ls.get("pod") else None,
                         "cost_usd": ls["pod"]["cost_usd"] if ls.get("pod") else None, "started": ls.get("started"),
                         "kill_at": ls["pod"].get("kill_at") if ls.get("pod") else None, "error": ls.get("error"),
                         "span": [ch["t0"], ch["t1"]], "core_s": ch["core_s"]})
            continue
        st = ch["status"]
        if st in ("running", "starting"):
            hb = _ts(sc.get("heartbeat") or sc.get("heartbeat_ts") or sj.get("heartbeat"))
            upd = _ts(state.get("updated"))
            if hb is None and sc.get("hb_age_s") is not None and upd:
                hb = upd - float(sc["hb_age_s"])          # the watcher writes the age at its last cycle
            eta_s = sc.get("eta_s")
            if eta_s is None and sc.get("eta_min") is not None:
                eta_s = float(sc["eta_min"]) * 60
            jobs.append({"id": cid, "chunk": cid, "source": "farm", "kind": ch.get("kind") or sj.get("kind"),
                         "label": "farm worker", "status": st, "step": sc.get("step") or sj.get("step"),
                         "steps": _farm_job_steps({**sc, "status": st}, ch.get("kind"), _spec_step_names(dance, cid)),
                         "progress": {"frac": _frac(sc.get("progress")), "basis": "worker heartbeat"}
                         if _frac(sc.get("progress")) is not None else None,
                         "eta": {"p50": eta_s, "p10": None, "p90": None} if eta_s is not None else None,
                         "heartbeat_ts": hb, "heartbeat_age_s": round(now - hb) if hb else None,
                         "heartbeat_stale": bool(sc.get("stalled") or (hb and now - hb > 120)), "pod": sc.get("pod"),
                         "gpu": sc.get("gpu") or (state.get("pods") or {}).get(sc.get("pod") or "", {}).get("gpu"),
                         "gpu_util": sc.get("gpu_util"), "step_i": sc.get("step_i"), "steps_n": sc.get("steps_n"),
                         "attempt": sc.get("attempt"),
                         "cost_usd": sc.get("cost_usd"), "started": _ts(sc.get("started") or sc.get("status_ts")),
                         "span": [ch["t0"], ch["t1"]], "core_s": ch["core_s"], "error": sc.get("error")})
        elif st in ("planned", "queued"):
            est = (sc.get("estimate") or ch.get("estimate") or {}).get("total_min")
            queue.append({"id": cid, "chunk": cid, "kind": ch.get("kind") or sj.get("kind"), "status": st,
                          "after": sj.get("after") or [], "needs_gpu": sj.get("needs_gpu"), "est_min": est,
                          "span": [ch["t0"], ch["t1"]], "core_s": ch["core_s"], "origin": ch.get("origin")})
        elif st == "failed":
            jobs.append({"id": cid, "chunk": cid, "source": ch["source"], "kind": ch.get("kind"), "label": "failed",
                         "status": "failed", "step": sc.get("step"), "steps": _farm_job_steps(sc, ch.get("kind")),
                         "progress": None, "eta": None, "error": redact(str(sc.get("error") or sc.get("reason") or ""))[:200],
                         "span": [ch["t0"], ch["t1"]], "core_s": ch["core_s"]})
    chunk_ids = {c["id"] for c in chunks}
    for jid, j in sjobs.items():                        # jobs that are not chunks: the dance's stitch / analyse job
        if jid in chunk_ids or not isinstance(j, dict):
            continue
        if j.get("status") in ("planned", "queued", "held", None):
            queue.append({"id": jid, "chunk": None, "kind": j.get("kind"), "status": j.get("status") or "planned",
                          "after": j.get("after") or j.get("needs") or [], "needs_gpu": j.get("needs_gpu"),
                          "est_min": None, "span": [None, None], "core_s": None, "origin": None})
        elif j.get("status") == "running":
            jobs.append({"id": jid, "chunk": None, "source": "farm", "kind": j.get("kind"), "label": "farm worker",
                         "status": "running", "step": j.get("step"), "steps": [], "progress": None, "eta": None,
                         "span": [None, None], "core_s": None})
    return jobs, queue


STALE_POD_S = 6 * 3600     # a farm pod of state.json older than this that Runpod no longer lists is history, not a pod


def _pods(state: dict, live: dict, rp: dict | None, now: float) -> list[dict]:
    out, seen = [], set()
    if state.get(DEMO_FLAG):
        rp = None          # a demo dance never lists the account's real pods
    listed = {p.get("id") for p in (rp or {}).get("pods") or []} if (rp or {}).get("connection") == "ok" else None
    for pid, p in (state.get("pods") or {}).items():
        if not isinstance(p, dict) or p.get("state") in ("deleted", "gone"):
            continue
        created = _ts(p.get("created_ts") or p.get("created") or p.get("first_seen"))
        pstate = p.get("state") or "running"
        if listed is not None and str(pid) not in listed and created and now - created > STALE_POD_S:
            # 2026-10-21 review: "lost" 4090s at 247 h and a cpu pod "running" at 272 h ($141.64, a made-up figure) were
            # shown as live pods; Runpod lists none of them. They are history: left out (the farm log keeps them).
            continue
        price = _num(p.get("price_hr"), _num(p.get("cost_per_hr"), _num(p.get("price"))))
        cost = _num(p.get("cost_usd"), _num(p.get("spend")))
        cost = round(cost, 2) if cost is not None else None
        if cost is None and price is not None and created and pstate in ("running", "starting"):
            cost = round(price * max(0.0, now - created) / 3600, 2)
        out.append({"id": str(pid), "name": p.get("name"), "gpu": p.get("gpu"), "price_hr": price,
                    "state": pstate, "age_s": round(now - created) if created else None,
                    "created_ts": created, "cost_usd": cost,
                    "chunks": p.get("chunks") or p.get("jobs") or ([p["chunk"]] if p.get("chunk") else []),
                    "source": "farm", "kind": p.get("role") or p.get("kind") or "gpu", "dc": p.get("dc"), "kill_at": _ts(p.get("kill_at")),
                    "stoppable": not p.get("borrowed"), "flags": ["stalled"] if p.get("stalled") else []})
        seen.add(str(pid))
    for p in live.get("pods") or []:
        out.append({"id": p["id"], "name": p.get("name"), "gpu": p.get("gpu"), "price_hr": p.get("price_hr"),
                    "state": p.get("state") or "running", "age_s": p.get("age_s"), "created_ts": p.get("created_ts"),
                    "cost_usd": p.get("cost_usd"), "chunks": [p["seg"]], "source": "fit_full", "kind": "gpu",
                    "dc": None, "kill_at": p.get("kill_at"), "stoppable": False, "flags": [],
                    "note": f"fit_full session {p['sid'][-12:]}: stop it with python -m dancecap.fit_full, not here"})
        seen.add(p["id"])
    for p in (rp or {}).get("pods") or []:        # other pods billing on the account: shown so nothing bills unseen
        if p.get("id") in seen:
            if p.get("flags"):
                for o in out:
                    if o["id"] == p["id"]:
                        o["flags"] = list(p["flags"])
            continue
        if p.get("ledger") == "closed":
            continue
        sess = p.get("session") or {}
        runner = bool(sess.get("sid")) or str(p.get("name") or "").startswith("dancecap-batch")
        out.append({"id": p.get("id"), "name": p.get("name"), "gpu": p.get("gpu"), "price_hr": p.get("cost_per_hr"),
                    "state": str(p.get("runtime_status") or p.get("status") or "running").lower(), "age_s": p.get("age_s"),
                    "created_ts": p.get("created"), "cost_usd": p.get("cost_so_far_usd"),
                    "chunks": [sess.get("take")] if sess.get("take") else [], "source": "runner" if runner else "other",
                    "kind": "gpu", "dc": None, "kill_at": (p.get("kill") or {}).get("at"), "stoppable": False,
                    "flags": list(p.get("flags") or []), "session": sess.get("sid"),
                    "note": (f"remote-runner session {sess['sid']} (see Analyses above / its session page)" if sess.get("sid")
                             else "not started by this farm or by fit_full (read-only here)")})
    order = {"farm": 0, "fit_full": 1, "runner": 2, "other": 3}
    out.sort(key=lambda p: (order.get(p["source"], 3), p.get("created_ts") or 0))
    return out


# ------------------------------------------------------------------------------------------------ budget, volume, world, watcher

def _budget(rp: dict | None, state: dict) -> dict:
    rp = rp or {}
    b = rp.get("budget") or {}
    acct = rp.get("account") or {}
    bal = acct.get("balance_usd")
    cap, spent = b.get("cap"), b.get("spent")
    remaining = b.get("remaining")
    reserve = 2.0
    worst_open = ((b.get("running") or {}).get("worst_usd")) if b else None
    avail = None
    if remaining is not None:
        avail = remaining if bal is None else min(remaining, float(bal) - reserve)
        avail = round(avail - (worst_open or 0.0), 2)
    # pods Runpod lists now vs rows the ledger still holds open (2026-10-21: "6 open pods" while none ran: their closes
    # were lost; `remote reconcile` closes such rows once Runpod has not listed the pod twice)
    pods_b = b.get("pods") if isinstance(b.get("pods"), list) else []
    listed = rp.get("pods") if rp.get("connection") == "ok" and isinstance(rp.get("pods"), list) else None
    return {"cap": cap, "spent": spent, "remaining": remaining, "balance": bal, "reserve_usd": reserve,
            "level": b.get("level") or "green", "messages": b.get("messages") or [], "open_pods": b.get("open_pods"),
            "ledger_open": b.get("open_pods") if isinstance(b.get("open_pods"), int) else len(pods_b),
            "ledger_stale": sum(1 for p in pods_b if isinstance(p, dict) and p.get("phantom_cost") is not None),
            "runpod_listed": len(listed) if listed is not None else None,
            "open_rate_hr": acct.get("spend_per_hr_usd"), "worst_open_usd": worst_open, "available_usd": avail,
            "farm_spend": (state.get("spend") or {}), "connection": rp.get("connection"), "checked": rp.get("checked")}


def volume_info(state: dict | None = None) -> dict | None:
    v = read_json(volume_file(), None)
    if not isinstance(v, dict):
        return None
    size = _num(v.get("size_gb"))
    price = _num(v.get("price_usd_per_gb_month"), 0.07)
    used, measured, source = None, None, None
    u = read_json(paths.monitor_dir() / VOLUME_USAGE, None)
    if isinstance(u, dict) and _num(u.get("used_gb")) is not None:
        used, measured, source = float(u["used_gb"]), u.get("measured"), u.get("source")
    sv = (state or {}).get("volume") if isinstance(state, dict) else None
    if isinstance(sv, dict) and (_num(sv.get("used_gb")) is not None or _num(sv.get("used_bytes")) is not None):
        used = float(sv["used_gb"]) if _num(sv.get("used_gb")) is not None else float(sv["used_bytes"]) / 1e9
        measured, source = sv.get("measured") or sv.get("updated"), "farm state"
    return {"id": v.get("id"), "name": v.get("name"), "dc": v.get("dc"), "tier": v.get("tier"), "size_gb": size,
            "used_gb": used, "used_measured": measured, "used_source": source, "price_gb_month": price,
            "price_month": round(size * price, 2) if size is not None else None, "mount": v.get("mount"),
            "created": v.get("created")}


def _world(cov: dict | None, state: dict) -> dict:
    w = dict((cov or {}).get("world") or {})
    for k, v in (state.get("world") or {}).items():
        w.setdefault(k, v)
    if not w and (cov or {}).get("world_id"):
        w = {"id": cov["world_id"]}
    return {"id": w.get("id"), "pose_id": w.get("pose_id"), "sealed": w.get("sealed"), "avatar_id": w.get("avatar_id"),
            "sha256": (w.get("sha256") or "")[:12] or None}


def _watcher(dance: str, state: dict, now: float) -> dict:
    """The detached watcher (dancecap.farm.watch): farm/watcher.pid {pid, heartbeat (epoch), state, cycle, poll_s}, farm/watcher.log,
    state.json["watcher"] {pid, state, heartbeat, heartbeat_epoch}. Running = its process is alive and its heartbeat is recent."""
    fd = farm_dir(dance)
    w = dict(state.get("watcher") or {})
    info = read_json(fd / "watcher.pid", None)
    info = info if isinstance(info, dict) else {}
    pid = info.get("pid") or w.get("pid")
    hb = _ts(info.get("heartbeat")) or _ts(w.get("heartbeat_epoch")) or _ts(w.get("heartbeat"))
    poll = float(info.get("poll_s") or w.get("poll_s") or 30.0)
    alive = bool(pid and pid_alive(pid)) if pid else None
    stale_after = max(WATCHER_STALE_S, 6 * poll)
    fresh = bool(hb and now - hb < stale_after)
    pstate = info.get("state") or w.get("state")
    running = bool((alive or state.get(DEMO_FLAG)) and fresh and pstate in (None, "running", "starting"))   # a demo has no process
    log = next((n for n in ("watcher.log", "watch.log") if (fd / n).exists()), None)
    return {"heartbeat_ts": hb, "heartbeat_age_s": round(now - hb) if hb else None, "pid": pid, "pid_alive": alive,
            "state": pstate, "cycle": info.get("cycle") or w.get("cycle"), "running": running, "fresh": fresh,
            "hung": bool(alive and not fresh), "log": log, "known": bool(w or info)}


def _events(state: dict, n: int = 14) -> list[dict]:
    ev = [e for e in (state.get("events") or []) if isinstance(e, dict)]
    return [{"ts": e.get("ts"), "t": e.get("t"), "kind": e.get("kind"), "msg": redact(str(e.get("msg") or ""))[:200]}
            for e in ev[-n:]][::-1]


# ------------------------------------------------------------------------------------------------ outputs / landing detection

def _dir_mtime_deep1(p: Path) -> float:
    """mtime of a dir and of its immediate sub-dirs (cheap: stat only, no recursion)."""
    best = mtime(p) or 0.0
    try:
        with os.scandir(p) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    try:
                        best = max(best, e.stat().st_mtime)
                    except OSError:
                        pass
    except OSError:
        pass
    return best


def outputs_mtime(dance: str, chunks: list[dict]) -> float:
    """Newest mtime among the places a fit / refine / quality result lands for this dance's chunk takes."""
    takes = {c.get("take") or f"{dance}_{c['id']}" for c in chunks}
    best = 0.0
    for t in takes:
        base = paths.work() / t
        if not base.is_dir():
            continue
        for rel in OUTPUT_DIRS:
            best = max(best, mtime(base / rel) or 0.0)
        for rel in FIT_DIRS_REL:
            best = max(best, _dir_mtime_deep1(base / rel))
    return best


# ------------------------------------------------------------------------------------------------ alerts

def _alerts(b: dict, cov_ok: bool, chunks: list[dict], pods: list[dict], watcher: dict, outputs: dict, bud: dict,
            counts: dict | None, analysis: dict | None = None) -> list[dict]:
    out = []
    if not cov_ok and not analysis:
        out.append({"level": "amber", "id": "no-coverage", "message":
                    "No coverage map yet for this dance: run  python -m dancecap.farm coverage  (the map is computed from the real outputs)."})
    if bud.get("level") in ("amber", "red"):
        out.append({"level": bud["level"], "id": "budget", "message": "; ".join(bud.get("messages") or ["budget"])})
    for p in pods:
        for f in p.get("flags") or []:
            out.append({"level": "red", "id": f"pod:{p['id']}:{f}",
                        "message": f"pod {p['id']} ({p.get('name') or '?'}): {f} - it keeps billing; check it in Runpod"})
    farm_active = [c for c in chunks if c["source"] == "farm" and c["status"] in ("queued", "running", "done")]
    if farm_active and not watcher.get("running"):
        out.append({"level": "amber", "id": "watcher", "message":
                    "Farm chunks are queued or running but no watcher is alive: finished chunks are not pulled and pods are not "
                    "deleted. Use Restart watcher."})
    if outputs.get("stale"):
        out.append({"level": "info", "id": "stale", "message":
                    "New fit / refine output has landed since the map was computed: the map refreshes itself."})
    for c in chunks:
        if c["status"] == "failed":
            out.append({"level": "red", "id": f"chunk:{c['id']}", "message": f"chunk {c['id']} failed"})
    return out


# ------------------------------------------------------------------------------------------------ the bundle

def build(dance: str, rp: dict | None = None, now: float | None = None, show_demo: bool = False,
          actions: list[dict] | None = None) -> dict:
    now = now or time.time()
    fd = farm_dir(dance)
    state = read_json(fd / "state.json", None)
    state = state if isinstance(state, dict) else {}
    cov = read_coverage(dance)
    live = fit_sessions.live_fits(dance, now=now) if not state.get(DEMO_FLAG) else {"segments": {}, "pods": [], "found": 0}
    chunks = _merge_chunks(cov or {}, state, live, now)
    cells = compact_cells(cov) if cov else []
    _annotate_cells(cells, chunks, cov or {})
    tot = counts_of(cells, float((cov or {}).get("cell_s") or 1.0)) if cells else None
    jobs, queue = _jobs_and_queue(dance, chunks, state, live, now)
    pods = _pods(state, live, rp, now)
    bud = _budget(rp, state)
    if state.get(DEMO_FLAG) and isinstance(state.get("budget"), dict):      # demo dances carry their own made-up budget
        bud = {**bud, **state["budget"]}
    watcher = _watcher(dance, state, now)
    out_m = outputs_mtime(dance, chunks)
    cov_m = (cov or {}).get("_mtime") or 0.0
    outputs = {"newest_mtime": out_m or None, "coverage_mtime": cov_m or None, "stale": bool(cov and out_m > cov_m + 5)}
    coverage = {"exists": bool(cov), "updated": (cov or {}).get("updated"), "updated_ts": cov_m or None,
                "source": Path((cov or {}).get("_path", "")).name or None, "t0": (cov or {}).get("t0"),
                "t1": (cov or {}).get("t1"), "cell_s": (cov or {}).get("cell_s") or 1.0, "anchor": (cov or {}).get("anchor"),
                "hz": (cov or {}).get("hz"), "k0": (cov or {}).get("k0"), "k1": (cov or {}).get("k1"),
                "n_cells": len(cells), "cells": cells,
                "counts": (tot or {}).get("counts") or (cov or {}).get("counts") or {"complete": 0, "partial": 0, "none": 0},
                "seconds": (tot or {}).get("seconds") or (cov or {}).get("seconds") or {"complete": 0, "partial": 0, "none": 0},
                "percent": (tot or {}).get("percent") or (cov or {}).get("percent") or {"complete": 0, "partial": 0, "none": 0},
                "stages": (cov or {}).get("stages") or {}, "total_s": round(sum(((tot or {}).get("seconds") or {}).values()), 1)}
    analysis = analyses_by_take().get(dance)
    alerts = _alerts({}, bool(cov), chunks, pods, watcher, outputs, bud, coverage["counts"], analysis)
    live_n = sum(1 for c in cells if c.get("live"))
    return {"schema": SCHEMA, "dance": dance, "demo": bool(state.get(DEMO_FLAG)), "now": now, "coverage": coverage,
            "analysis": analysis,
            "chunks": chunks, "jobs": jobs, "queue": queue, "pods": pods, "budget": bud, "volume": volume_info(state),
            "world": _world(cov, state), "watcher": watcher, "events": _events(state), "alerts": alerts,
            "outputs": outputs, "live": {"fit_full": live.get("found", 0), "cells": live_n},
            "actions": actions or [], "state_updated": state.get("updated"),
            "state_mtime": mtime(fd / "state.json")}


def rev_of(bundle: dict) -> str:
    """Stable fingerprint of a bundle without its clock-dependent fields (ETag: a poll with nothing new answers 304)."""
    import hashlib

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("now", "age_s", "heartbeat_age_s", "at", "checked")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    return hashlib.sha1(json.dumps(strip(bundle), sort_keys=True, default=str).encode()).hexdigest()[:16]
