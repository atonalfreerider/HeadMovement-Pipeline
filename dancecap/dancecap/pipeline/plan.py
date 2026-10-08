"""Plans: what a run would do, with time, pod cost, budget and consent - computed with no side effects.

A plan (PIPELINE_MONITOR 7.3 `Plan`) lists every stage of the requested scope with an action:
  run | rerun_stale | skip_done | blocked (a human gate, a stage not built yet, a lock, already running) | n/a
pod stages grouped into their sessions (S1_front, S2_fit, S_room) with p50/p90 minutes and dollars
(upload is billed GPU time), the budget launch gate (spent + worst case of running pods + p90 of the new sessions
<= cap; over cap - reserve needs an explicit flag), the cloud-processing consent of every dancer, and the camera-type
band (unconfirmed cameras are priced as handheld + zoom at p90).

The plan hash covers the steps, the sessions, the prices and the mtimes of the take TOML and the steps' outputs, so a
plan approved in the UI is refused (412) when anything changed. Only a plan can lead to a pod: the server starts the
orchestrator with the plan file, and the orchestrator runs a session only when the plan carries the user's approval,
bound to that run id and used once (orchestrator.load_plan_for_run).

Honesty rules:
- One session entry per POD: a later wave of a group (S2_fit#2: texture waits for fit_qa, which waits for mamma_fit)
  creates its own pod, so it is its own entry with its own upload, bootstrap, kill timer and cost ceiling; the
  approval counts entries.
- A pod JOB that cannot be narrowed to the selected stages (S1_front: front_pod has no per-stage selection) is priced,
  kill-timed and shown with everything it will really (re-)run: its done stages without resume markers become
  `rerun_stale` ("the S1_front pod job re-runs it"), per-camera stages scaled to the cameras without markers.
- keep_existing stages (cam_map: the static SfM map is the world frame) are never rebuilt without
  options.rebuild_map = true, whatever their status or the mode.
- A review gate the run would pass (fit_qa, geoveto) halts it: every DAG descendant of a gated stage that re-runs
  is `blocked` ("after the review gate at ..."), even through stages that are fresh now (their inputs change once the
  review is confirmed), so its sessions are neither priced nor approved in this plan.
- What-if plans (options.assume_reviews, DRY-RUN ONLY: revalidate refuses a real run): the human steps become `pause`
  ("a real run pauses here for you") and the review gates do not block their descendants, so a new capture's whole
  order, wall time and pod sessions can be previewed before any review exists. Not-built and manual stages stay
  blocked.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

from . import camtypes, estimate, paths, registry, status, takes
from .util import STAGE_RE, mtime, read_json, write_json_atomic

VALID_S = 600
SCOPES = ("only", "upto", "all")
MODES = ("resume", "force")


class PlanError(ValueError):
    pass


def _number(v, field: str) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise PlanError(f"{field} must be a number")
    try:
        x = float(v)
    except ValueError:
        raise PlanError(f"{field} must be a number") from None
    if not math.isfinite(x) or x <= 0:
        raise PlanError(f"{field} must be a positive number")
    return x


def normalise_request(req: dict) -> dict:
    if req is not None and not isinstance(req, dict):
        raise PlanError("the plan request must be a JSON object")
    req = dict(req or {})
    target = req.get("target")
    if target is not None and not STAGE_RE.match(str(target)):
        raise PlanError("invalid target stage")
    scope = req.get("scope") or ("upto" if target else "all")
    mode = req.get("mode") or "resume"
    if scope not in SCOPES or mode not in MODES:
        raise PlanError("scope must be only|upto|all and mode resume|force")
    if scope != "all" and not target:
        raise PlanError("scope only/upto needs a target stage")
    inc_in = req.get("include_optional") or []
    if not isinstance(inc_in, list):
        raise PlanError("include_optional must be a list of stage ids")
    inc = [s for s in inc_in if isinstance(s, str) and STAGE_RE.match(s)]
    opts = req.get("options") or {}
    if not isinstance(opts, dict):
        raise PlanError("options must be a JSON object")
    for k in ("gpus", "clouds"):
        if opts.get(k) is not None and not isinstance(opts.get(k), list):
            raise PlanError(f"options.{k} must be a list")
    gpus = [g for g in opts.get("gpus") or [] if isinstance(g, str) and len(g) < 64]
    clouds = [c for c in opts.get("clouds") or [] if c in ("COMMUNITY", "SECURE")]
    cap_usd = _number(opts.get("max_cost_usd"), "options.max_cost_usd")
    rebuild = opts.get("rebuild_map", False)
    if not isinstance(rebuild, bool):
        raise PlanError("options.rebuild_map must be true or false")
    assume = opts.get("assume_reviews", False)
    if not isinstance(assume, bool):
        raise PlanError("options.assume_reviews must be true or false")
    capture = opts.get("capture")
    from .util import CAPTURE_RE
    if capture is not None and not CAPTURE_RE.match(str(capture)):
        raise PlanError("invalid capture name")
    return {"target": target, "scope": scope, "mode": mode, "include_optional": sorted(inc),
            "options": {"gpus": gpus, "clouds": clouds, "max_cost_usd": cap_usd, "capture": capture,
                        "rebuild_map": rebuild, "assume_reviews": assume}}


def _price(session: str, opts: dict) -> float:
    spec = registry.POD_SESSIONS[session]
    if spec["cloud"] == "SECURE":
        return estimate.A40_SECURE if "A40" in spec.get("gpu_hint", "") else estimate.DEFAULT_PRICE["SECURE"]
    clouds = opts.get("clouds") or ["COMMUNITY"]
    if spec.get("price_per_hr") and clouds[0] == spec["cloud"]:  # a session priced at the GPU the runner picks
        return float(spec["price_per_hr"])
    return estimate.DEFAULT_PRICE.get(clouds[0], estimate.DEFAULT_PRICE["COMMUNITY"])


def _session_waves(steps: list[dict], vstages: list[dict], vby: dict, run_ids: set) -> list[tuple[str, list[dict]]]:
    """Pod steps grouped into pods. A member whose input (outside the session) is downstream of another member
    (texture <- fit_qa <- mamma_fit) cannot share that pod: it goes into a later wave, "<group>#2" (one more upload
    and bootstrap, priced as such). The orchestrator applies the same rules."""
    out = []
    for sess in registry.POD_SESSIONS:
        mem = [st for st in steps if st.get("session") == sess and st["stage"] in run_ids]
        remaining = [st["stage"] for st in mem]
        w = 1
        while remaining:
            cur = []
            for m in remaining:
                outside = [d for d in vby[m]["deps"] if d in run_ids and vby[d].get("session") != sess]
                if not any(x in registry.ancestors(vstages, d) for d in outside for x in remaining if x != m):
                    cur.append(m)
            cur = cur or list(remaining)
            out.append((sess if w == 1 else f"{sess}#{w}", [st for st in mem if st["stage"] in cur]))
            remaining = [m for m in remaining if m not in cur]
            w += 1
    return out


GATE_REASON = "after the review gate at {g} (the run halts there for your review; confirm it, then plan again)"


def _steps(take: str, req: dict, by: dict, vstages: list[dict], vby: dict, scope: list[str], capture, tag,
           forced: dict[str, float], hold: dict[str, str] | None = None) -> tuple[list[dict], list[str]]:
    """The plan's steps in DAG order (scope + the stages a pod job will run anyway), and the review gates passed.
    `hold`: {stage: reason} pod steps held back because their session's job cannot start yet (make_plan)."""
    hold = hold or {}
    assume = bool(req["options"].get("assume_reviews"))  # what-if (dry-run only): your reviews are assumed done
    steps, blocked_ids, rerun_ids, gate_ids = [], set(), set(), []
    behind_gate: dict[str, str] = {}  # every DAG descendant of a gate stage that re-runs -> that gate
    satisfied = ("done", "skipped", "n/a")
    in_plan = set(scope) | set(forced)
    for sid in [s["id"] for s in vstages if s["id"] in in_plan]:
        x, s = by[sid], vby[sid]
        st = x["status"]
        extra = sid not in scope  # only here because the pod job runs it
        assumed_note = None
        if assume and st == "blocked" and s.get("runnable") in ("local", "session"):
            assumed_note, st = x.get("status_reason") or "needs you", "ready"  # e.g. a consent or camera-type gate
        if extra and not all(by[d]["status"] in satisfied or d in rerun_ids for d in s["deps"]):
            continue  # its inputs are not ready: the job's task fails fast on the pod (warning below)
        after = [d for d in s["deps"] if d in rerun_ids]
        keep = s.get("keep_existing") and st in ("done", "stale") and not req["options"].get("rebuild_map")
        if st == "done" and after and s.get("runnable") in ("local", "session") and not keep:
            st = "stale"  # its inputs are about to change: re-run it too (honest time and cost)
            x = dict(x, outputs=dict(x["outputs"], stale_because=[f"after {d} re-runs" for d in after]))
        frac = forced.get(sid)
        step = {"stage": sid, "runs_on": s.get("runs_on"), "session": s.get("session"), "status": st,
                "est": {"p50_s": x["estimate"]["p50_s"], "p90_s": x["estimate"]["p90_s"]},
                "cost": {"p50_usd": x["estimate"]["cost_p50_usd"], "p90_usd": x["estimate"]["cost_p90_usd"]}}
        ct_split = {k: v for k, v in (x.get("camera_types") or {}).items() if v}
        if ct_split:  # the camera stages: which cameras this step solves with which model ({class: [cams]})
            step["camera_types"] = ct_split
        if frac is not None and frac < 1.0:  # per-camera job work: only the cameras without resume markers
            step["est"] = {k: round(v * frac) for k, v in step["est"].items()}
            step["cost"] = {k: round(v * frac, 2) for k, v in step["cost"].items()}
            step["job_fraction"] = frac
        dep_block = [d for d in s["deps"] if d in blocked_ids]
        gate_dep = [behind_gate[sid]] if sid in behind_gate else []
        optional_skip = st == "skipped" and sid not in req["include_optional"]
        if keep:
            step.update(action="skip_done", reason=f"kept: {s['keep_existing']}" if st == "stale" else "outputs fresh")
            if st == "stale":
                step["warning"] = ("kept although stale (" + "; ".join(x["outputs"]["stale_because"][:2])
                                   + "): set options.rebuild_map to rebuild it")
        elif st in ("n/a",) or optional_skip:
            step.update(action="n/a", reason=x.get("status_reason") or st)
        elif sid in hold:  # only stages that would run are held (make_plan)
            step.update(action="blocked", reason=hold[sid])
        elif st == "skipped":
            step.update(action="run", reason="optional, included")
        elif st == "done" and req["mode"] == "resume" and frac is None:
            step.update(action="skip_done", reason="outputs fresh")
        elif st == "planned":
            step.update(action="blocked", reason=x.get("status_reason") or "not built yet")
        elif st == "running":
            step.update(action="blocked", reason=f"running now ({x.get('status_reason')})")
        elif dep_block:
            step.update(action="blocked", reason="after " + ", ".join(dep_block))
        elif gate_dep:
            step.update(action="blocked", reason=GATE_REASON.format(g=", ".join(gate_dep)))
        elif assume and s.get("runnable") == "human":
            step.update(action="pause", reason=f"a real run pauses here for you ({x.get('status_reason') or 'needs you'}); "
                                               "what-if: assumed done")
        elif st == "blocked" or s.get("runnable") in ("human", None):
            step.update(action="blocked", reason=x.get("status_reason") or "needs you")
        elif s.get("runnable") == "manual":
            step.update(action="blocked", reason=f"manual: {s.get('manual_hint') or 'run by hand'}")
        else:
            if frac is not None and st == "done":
                st = step["status"] = "stale"
                step["action"] = "rerun_stale"
                step["reason"] = (f"the {s.get('session')} pod job re-runs it (no per-stage selection"
                                  + (f"; {frac:.0%} of the cameras have no resume markers" if frac < 1 else "")
                                  + "): its outputs are overwritten")
            else:
                step["action"] = "rerun_stale" if st == "stale" else "run"
                step["reason"] = "; ".join(x["outputs"]["stale_because"][:2]) if st == "stale" else (
                    "forced" if st == "done" else x.get("status_reason") or st)
            if extra:
                step["reason"] = (f"outside the requested scope, but the {s.get('session')} pod job runs it: "
                                  + step["reason"])
            if assumed_note:
                step["reason"] = f"what-if: runs once you act ({assumed_note})"
                step["status"] = x["status"]
            ctx_cmd = {"take": take, "capture": capture, "tag": tag}
            try:
                if s.get("runnable") == "local":
                    step["argv_display"] = [["python", *a] for a in registry.commands(sid, ctx_cmd)]
                elif s.get("runnable") == "session":
                    step["argv_display"] = [["python", *a] for a in registry.session_command(
                        s["session"], [sid], {**ctx_cmd, "max_cost": None}, dry_run=False)]
            except registry.NotRunnable as e:
                step.update(action="blocked", reason=str(e))
            if s.get("unity_lock") and (paths.work() / "unity.lock").exists():
                step["warning"] = "work/unity.lock is held: the export waits for it (or override)"
        if step["action"] == "blocked":
            blocked_ids.add(sid)
        if step["action"] == "pause":
            rerun_ids.add(sid)
        if step["action"] in ("run", "rerun_stale"):
            rerun_ids.add(sid)
            if s.get("gate_human"):
                gate_ids.append(sid)
                if not assume:
                    for dsc in registry.descendants(vstages, [sid]):
                        behind_gate.setdefault(dsc, sid)  # its inputs change after the review: never in this run
        steps.append(step)
    return steps, gate_ids


def make_plan(take: str, request: dict, now: float | None = None) -> dict:
    now = now or time.time()
    req = normalise_request(request)
    ctx = status.context(take)
    ctx["include_optional"] = set(req["include_optional"])
    snap = status.snapshot(take, ctx)
    by = {x["id"]: x for x in snap["stages"]}
    vstages = registry.variant_stages(ctx["variant"])
    vby = {s["id"]: s for s in vstages}
    tgt = req["target"]
    if tgt is not None and tgt not in vby:
        raise PlanError(f"{tgt} is not a stage of the {ctx['variant']}-view pipeline")
    if req["scope"] == "only":
        scope = [tgt]
    elif req["scope"] == "upto":
        scope = registry.ancestors(vstages, tgt) + [tgt]
    else:
        scope = [s["id"] for s in vstages]
    capture = req["options"].get("capture") or takes.last_capture(take)
    tag = f"c{len(ctx['used'])}"
    forced: dict[str, float] = {}
    job_warnings: list[str] = []
    job_cache: dict[str, dict] = {}
    hold: dict[str, str] = {}
    for _ in range(5):  # a pod job that cannot be narrowed adds its stages (fixpoint in a few passes)
        steps, gate_ids = _steps(take, req, by, vstages, vby, scope, capture, tag, forced, hold)
        running_groups = {st["session"] for st in steps if st.get("session") and st["action"] in ("run", "rerun_stale")}
        want: dict[str, float] = {}
        for g in sorted(running_groups):
            if g not in job_cache:
                job_cache[g] = registry.session_job_reruns(g, take, ctx["used"])
            for sid, frac in job_cache[g].items():
                if sid in vby and vby[sid].get("implemented") and vby[sid].get("session") == g:
                    want[sid] = frac
        # One pod per job: a session whose job runs ALL its stages (S1_front) cannot start while one of those stages
        # still waits (a human gate such as labels_review, a running stage, ...): the job would attempt it anyway
        # (failing on the pod), and the stage would need a second pod (a second upload and bootstrap) later.
        new_hold = dict(hold)
        for g in sorted(running_groups):
            waiting = [st for st in steps if st["stage"] in job_cache.get(g, {}) and st["action"] == "blocked"
                       and st["stage"] not in hold and vby.get(st["stage"], {}).get("session") == g]
            if not waiting:
                continue
            why = waiting[0].get("reason") or "blocked"
            for st in steps:
                if st.get("session") == g and st["action"] in ("run", "rerun_stale"):
                    new_hold[st["stage"]] = (f"waits with the {g} pod job for {waiting[0]['stage']} ({why}): the job "
                                             "runs all its stages in one pod, so it starts once they can all run")
        if want == forced and new_hold == hold:
            break
        forced, hold = want, new_hold
    planned_ids = {st["stage"] for st in steps}
    skipped_job = [sid for sid in forced if sid not in planned_ids]
    if skipped_job:
        job_warnings.append("the S1_front pod job also attempts " + ", ".join(skipped_job)
                            + " (their inputs are not ready: those tasks fail on the pod)")

    # pod sessions of the steps that will run
    sessions = []
    wl = ctx["wl"]
    run_ids = {st["stage"] for st in steps if st["action"] in ("run", "rerun_stale")}
    stage_sess: dict[str, str] = {}
    for sess, mem in _session_waves(steps, vstages, vby, run_ids):
        spec = registry.POD_SESSIONS[sess.split("#")[0]]
        for st in mem:
            stage_sess[st["stage"]] = sess
        over = estimate.session_overhead(spec.get("upload_mb") or spec.get("upload_mb_per_cm", 500) * wl["CM"])
        w50 = sum(st["est"]["p50_s"] for st in mem) / 60 + over["p50_min"]
        w90 = sum(st["est"]["p90_s"] for st in mem) / 60 + over["p90_min"]
        price = _price(sess.split("#")[0], req["options"])
        max_min = round(w90 * 1.1 + 13)
        if spec.get("fixed_budget"):  # a job that bounds itself: the approved kill switch and ceiling bound the
            max_min = int(spec["max_minutes"])  # session (its training ends 15 min before the kill switch)
            w50, w90 = min(w50, max_min), min(w90, max_min)
        sessions.append({"id": sess, "group": sess.split("#")[0], "stages": [st["stage"] for st in mem],
                         "jobs": registry.session_jobs(sess.split("#")[0], [st["stage"] for st in mem], take),
                         "gpu": (req["options"]["gpus"] or [spec.get("gpu_hint")])[0], "cloud": spec["cloud"],
                         "price_per_hr": price, "upload_bytes": int(over["upload_mb"] * 1e6),
                         "est_min": {"p50": round(w50), "p90": round(w90)},
                         "overhead_min": over,
                         "cost_usd": {"p50": round(price * w50 / 60, 2), "p90": round(price * w90 / 60, 2)},
                         "max_minutes": max_min,
                         "max_cost_usd": (round(float(spec["max_cost"]), 2) if spec.get("fixed_budget") else
                                          round(price * (max_min + 6) / 60, 2))})
    run_steps = [st for st in steps if st["action"] in ("run", "rerun_stale")]
    local_s = sum(st["est"]["p50_s"] for st in run_steps if st["runs_on"] in ("local", "local_gpu"))
    # wall: longest dependency chain over the steps that run (sessions count once, with their overhead)
    finish: dict[str, tuple[float, float]] = {}
    sess_over = {s["id"]: (s["overhead_min"]["p50_min"] * 60, s["overhead_min"]["p90_min"] * 60) for s in sessions}
    for s in vstages:
        sid = s["id"]
        st = next((x for x in steps if x["stage"] == sid), None)
        d50 = d90 = 0.0
        if st and st["action"] in ("run", "rerun_stale"):
            d50, d90 = st["est"]["p50_s"], st["est"]["p90_s"]
            o = sess_over.pop(stage_sess.get(sid), None) if stage_sess.get(sid) else None
            if o:
                d50, d90 = d50 + o[0], d90 + o[1]
        # soft deps that run in this plan are waited for by the orchestrator: they are on the path too
        waits = list(s["deps"]) + [d for d in s.get("soft_deps") or [] if d in run_ids]
        prev = max((finish[d] for d in waits if d in finish), key=lambda t: t[0], default=(0.0, 0.0))
        finish[sid] = (prev[0] + d50, prev[1] + d90)
    wall = max(finish.values(), key=lambda t: t[0], default=(0.0, 0.0))
    p50_cost = sum(s["cost_usd"]["p50"] for s in sessions)
    p90_cost = sum(s["cost_usd"]["p90"] for s in sessions)
    gate = estimate.launch_gate(p90_cost) if sessions else None
    bud = {"spent": round(estimate.spent(), 2), "after_p50": None, "after_p90": None, "cap": estimate.cap(),
           "reserve": estimate.reserve(), "gate": "ok"}
    if gate:
        bud.update(after_p90=gate["after_p90"], gate=gate["gate"], running_worst=gate["running_worst"])
        bud["after_p50"] = round(bud["spent"] + p50_cost, 2)
    cfg = ctx["cfg"]
    consent = takes.cloud_consent(cfg) if sessions else {"cloud_processing": {}, "ok": True}
    unconf = [t["cam"] for t in ctx["types"] if t["state"] != "confirmed"]
    band = {"unconfirmed": unconf, "p90_extra_pod_min": 0, "p90_extra_usd": 0.0}
    loc = next((st for st in run_steps if st["stage"] == "cam_loc"), None)
    if loc and unconf:
        e_conf = estimate.stage_estimate(registry.get("cam_loc"), wl,
                                         [dict(t, state="confirmed") for t in ctx["types"]])
        extra = max(0.0, (loc["est"]["p90_s"] - e_conf["p90_s"]) / 60)
        band.update(p90_extra_pod_min=round(extra), p90_extra_usd=round(extra * estimate.DEFAULT_PRICE["COMMUNITY"] / 60, 2))
    # per camera: the model the camera stages of this plan apply (the effective class: the user's answer; an
    # unconfirmed suggestion is solved as handheld + zoom) and what that changes (PIPELINE.md 4.1)
    camera_types = [{"cam": t["cam"], "state": t["state"], "source": t.get("source"), "mount": t.get("mount"),
                     "zoom": t.get("zoom"), "class": t["effective"]["class"],
                     "suggested": ({"mount": (t.get("suggestion") or {}).get("mount"),
                                    "zoom": (t.get("suggestion") or {}).get("zoom"),
                                    "confidence": (t.get("suggestion") or {}).get("confidence")}
                                   if t.get("suggestion") else None),
                     "effects": camtypes.CLASS_EFFECTS.get(t["effective"]["class"], [])} for t in ctx["types"]]
    warnings = list(job_warnings)
    for st in run_steps:
        b = (by[st["stage"]]["estimate"].get("branches") or [])
        if b:
            warnings.append(f"{st['stage']}: p90 includes the '{b[-1]['name']}' branch")
        if st.get("warning"):
            warnings.append(f"{st['stage']}: {st['warning']}")
    if any(st["stage"] == "demo_export" and st["action"] in ("run", "rerun_stale") for st in steps) and not capture:
        warnings.append("demo_export needs a capture name (options.capture)")
    if bud["gate"] == "over_cap":
        warnings.append(f"budget: p90 ${bud['after_p90']:.2f} would exceed the cap ${bud['cap']:.2f}")
    elif bud["gate"] == "over_reserve":
        warnings.append(f"budget: p90 ${bud['after_p90']:.2f} is over cap - reserve")
    if sessions and not consent["ok"]:
        warnings.append("consent: cloud_processing is not recorded as true for every dancer ([consent] in the take TOML)")
    # root causes: blocked steps not merely waiting "after" another blocked step, that hold back something in scope
    in_scope = {st["stage"] for st in steps if st["action"] not in ("n/a", "skip_done")}
    blocking = [st for st in steps if st["action"] == "blocked" and not str(st.get("reason", "")).startswith("after ")
                and (set(registry.descendants(vstages, [st["stage"]])) & in_scope or st["stage"] == tgt)]
    pauses = [st["stage"] for st in steps if st["action"] == "pause"]
    if req["options"].get("assume_reviews"):
        warnings.insert(0, "what-if plan (dry-run only): your reviews are assumed done - a real run stops at the first "
                           "of " + ", ".join(dict.fromkeys(pauses + gate_ids) or ["(none)"]))
    elif gate_ids:
        warnings.append("the run halts for your review after " + ", ".join(gate_ids)
                        + ": the stages after it are not part of this plan")
    if any(st.get("warning", "").startswith("kept although stale") for st in steps):
        warnings += [f"{st['stage']}: {st['warning']}" for st in steps
                     if st.get("warning", "").startswith("kept although stale") and st["action"] == "skip_done"]
    plan = {"take": take, "variant": ctx["variant"], "request": req, "created": now, "valid_for_s": VALID_S,
            "capture": capture, "tag": tag, "steps": steps, "sessions": sessions,
            "totals": {"wall_s": {"p10": round(wall[0] * 0.7), "p50": round(wall[0]), "p90": round(wall[1])},
                       "local_s": round(local_s), "pod_cost_usd": {"p50": round(p50_cost, 2), "p90": round(p90_cost, 2)},
                       "n_run": len(run_steps)},
            "budget": bud, "consent": consent, "camera_type_band": band, "camera_types": camera_types,
            "warnings": warnings,
            "halts_at": list(dict.fromkeys([st["stage"] for st in blocking] + pauses + gate_ids)),
            "what_if": bool(req["options"].get("assume_reviews")),
            "review_gates": gate_ids, "pods": len(sessions)}
    plan["plan_hash"] = plan_hash(plan, take)
    return plan


def _input_mtimes(plan: dict, take: str) -> list:
    out = [mtime(paths.take_toml(take))]
    for st in plan["steps"]:
        s = registry.get(st["stage"]) or {}
        for pat in registry.done_patterns(s, take)[:3]:
            fs = registry.pattern_files(pat)
            out.append(max((mtime(f) or 0 for f in fs), default=0))
    return [round(x or 0, 1) for x in out]


def plan_hash(plan: dict, take: str) -> str:
    core = {"take": take, "request": plan["request"],
            "steps": [(s["stage"], s["action"]) for s in plan["steps"]],
            "sessions": [(s["id"], s["jobs"], s["price_per_hr"], s["max_minutes"]) for s in plan["sessions"]],
            "inputs": _input_mtimes(plan, take)}
    return "sha1:" + hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


def plan_path(h: str):
    hexpart = h.split(":", 1)[-1]
    if not hexpart.isalnum() or len(hexpart) != 40:
        raise PlanError("invalid plan hash")
    return paths.plans_dir() / f"{hexpart}.json"


def save(plan: dict) -> None:
    write_json_atomic(plan_path(plan["plan_hash"]), plan)


def load(h: str) -> dict | None:
    return read_json(plan_path(h), None)


class StalePlan(RuntimeError):
    def __init__(self, fresh: dict):
        super().__init__("the plan changed")
        self.fresh = fresh


class ApprovalError(PermissionError):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def revalidate(take: str, h: str, approve: dict | None, dry_run: bool, run_id: str | None = None) -> dict:
    """POST runs: recompute the plan; refuse when stale, unapproved, over budget or without consent. The approval is
    bound to `run_id` (the run the server is about to start): the orchestrator honours it for that run only, once."""
    old = load(h)
    if old is None or old.get("take") != take:
        raise PlanError("unknown plan (make a new one)")
    if time.time() - float(old.get("created", 0)) > VALID_S:
        fresh = make_plan(take, old["request"])
        save(fresh)
        raise StalePlan(fresh)
    fresh = make_plan(take, old["request"])
    if fresh["plan_hash"] != h:
        save(fresh)
        raise StalePlan(fresh)
    if fresh.get("what_if") and not dry_run:
        raise PlanError("a what-if plan (your reviews assumed done) can only be dry-run: untick 'what-if' to plan a "
                        "real run")
    if fresh["sessions"] and not dry_run:
        approve = approve if isinstance(approve, dict) else {}
        n = len(fresh["sessions"])
        need = round(sum(s["max_cost_usd"] for s in fresh["sessions"]), 2)
        try:
            a_pods = int(approve.get("pods") or 0) if not isinstance(approve.get("pods"), bool) else 0
            a_cost = float(approve.get("max_cost_usd") or 0) if not isinstance(approve.get("max_cost_usd"), bool) else 0
        except (TypeError, ValueError):
            a_pods, a_cost = 0, 0.0
        if not math.isfinite(a_cost):
            a_cost = 0.0
        if a_pods != n or a_cost + 1e-6 < need:
            raise ApprovalError("conflict", f"this plan creates {n} paid pod session(s) costing at most ${need:.2f}: "
                                            "approve exactly that", {"pods": n, "max_cost_usd": need})
        if not fresh["consent"]["ok"]:
            raise ApprovalError("consent_gate", "cloud_processing consent is missing for a dancer",
                                fresh["consent"])
        g = fresh["budget"]["gate"]
        if g == "over_cap":
            raise ApprovalError("budget_gate", "the p90 cost would exceed the Runpod cap", fresh["budget"])
        if g == "over_reserve" and not approve.get("over_reserve"):
            raise ApprovalError("budget_gate", "the p90 cost is over cap - reserve: approve over_reserve explicitly",
                                fresh["budget"])
        fresh["approved"] = {"pods": n, "max_cost_usd": a_cost, "at": time.time(),
                             "over_reserve": bool(approve.get("over_reserve")), "run_id": run_id}
    fresh["dry_run"] = bool(dry_run)
    save(fresh)
    return fresh
