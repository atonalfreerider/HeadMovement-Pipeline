"""Stage status from the files on disk, so existing takes show their real state (PipelineSnapshot / TakeSummary).

Rules (stages_draft conventions.status_rule + PIPELINE.md 7.4):
- done: every `done_when` path exists (globs allowed) and is not older than a hard dependency's outputs (2 s
  tolerance). A few stages use a rule instead (`registry.OVERLAY[...]["status_rule"]`): the take TOML, the clock
  file named by `alignment`, the window, the labels review, the narration (placeholder labels do not count), the
  Unity export.
- implied: outputs missing but a hard dependent is done -> the take was processed another way (e.g. MAMMA run
  directly on a 6 s take, without chunks): done ("implied"), or skipped for a stage that is not built yet.
- stale: older than a dependency, or a dependency is stale (it propagates), or the camera type of a camera changed
  since the stages that READ camera types ran (TYPE_STAGES: cam_loc, cam_world, mono_camera, pair_calib - never the
  static SfM map cam_map, which does not read them). What a stage applied comes from cameras/report.json
  `camera_types.applied` (written by every `dancecap cameras` run: CLI, pod or orchestrator), else the orchestrator's
  work/<take>/pipeline/camera_types_applied.json; legacy takes count as handheld+zoom. The comparison uses the POLICY
  that run used (report.json camera_types.policy / the side file's `policy`; default 'user' = confirmed answers only),
  so a run with `--camera-types auto` is not stale forever. A type the cameras stage REFUSED (report.json
  cams.<cam>.camera_type.check.passed false: the camera moves / zooms / the track would get shakier; it kept a safer
  model) is a warning on the camera_types step (status_reason, summary `refused`), not a block.
  Config-like outputs (the take TOML) never make anything stale by mtime: editing metadata is not a reason to re-run.
  A review the user rejected (gates.py) makes its stage stale.
- running: an active run of the monitor/orchestrator, or an active Runpod session whose job maps to the stage.
- failed: the latest run of the stage ended with rc != 0 after its outputs were written.
- blocked: a human gate is open (camera types uncertain or in conflict, labels review, the window), or a review
  gate the orchestrator opened after running fit_qa / geoveto (gates.py) waits for the user's confirmation.
- ready / waiting: deps satisfied (done, skipped, n/a) or not.
- planned: proposed in PIPELINE.md, not built yet (never faked).
- skipped / n/a: optional and not requested, or not applicable (no soundtrack, manual Unity check).
Precedence: running > failed > blocked > stale > done > ready > waiting > skipped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import camtypes, estimate, gates, paths, registry, runpod_view, runs, takes
from .util import mtime, read_json

TOL_S = 2.0
SATISFIED = ("done", "skipped", "n/a")
CAMERA_STAGES = ("cam_map", "cam_loc", "cam_world", "mono_camera", "pair_calib")  # display (per-type split)
TYPE_STAGES = ("cam_loc", "cam_world", "mono_camera", "pair_calib")  # read the camera types: stale on a change
GATED_BY_TYPES = ("cam_map", "cam_loc", "cam_world", "mono_camera", "pair_calib")
REVIEW_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4")


def _files(stage: dict, take: str) -> tuple[bool, float | None, list[str], list[dict]]:
    """(all patterns matched, completion time = min over patterns of the newest file, missing, detail)."""
    pats = registry.done_patterns(stage, take)
    if not pats:
        return False, None, [], []
    ok, t_done, missing, detail = True, None, [], []
    for rel, pat in zip(stage.get("done_when") or [], pats):
        fs = registry.pattern_files(pat)
        times = [m for m in (mtime(f) for f in fs) if m is not None]
        newest = max(times) if times else None
        detail.append({"path": paths.rel_display(pat) if "*" not in pat else rel, "exists": bool(fs),
                       "count": len(fs), "mtime": newest})
        if not fs:
            ok = False
            missing.append(rel)
        elif newest is not None:
            t_done = newest if t_done is None else min(t_done, newest)
    return ok, t_done, missing, detail


def _labels_kind(take: str) -> tuple[bool, bool]:
    """(human/narration labels exist, auto labels exist) in moves/labels.json (placeholders never count)."""
    d = read_json(paths.take_work(take) / "moves" / "labels.json", None)
    if not isinstance(d, dict) or d.get("placeholder"):
        return False, False
    segs = d.get("segments") or []
    prov = {str(s.get("provenance", "")) for s in segs if isinstance(s, dict)}
    return bool(prov & {"human", "narration"}), "auto" in prov


def _raw(stage: dict, take: str, cfg: dict, ctx: dict) -> dict:
    """Pass 1: existence, completion time and rule-based statuses."""
    rule = stage.get("status_rule")
    r = {"exists": False, "t": None, "missing": [], "detail": [], "rule_status": None, "reason": None, "config": False}
    if rule == "intake":
        r.update(exists=paths.take_toml(take).exists(), config=True)
        probs = ctx["intake_problems"]
        if probs:
            r["reason"] = "; ".join(probs)
        return r
    if rule == "camera_types":
        r["config"] = True
        types = ctx["types"]
        if not types:
            r.update(rule_status="n/a", reason="no videos registered")
            return r
        conf = [t["cam"] for t in types if t["state"] == "conflict"]
        unsure = [t["cam"] for t in types if camtypes.needs_user(t)]
        unconf = [t["cam"] for t in types if t["state"] != "confirmed"]
        refused = [t for t in types if (t.get("model_check") or {}).get("refused")]
        cams_done = any(ctx["raw_exists"].get(s) for s in ("cam_loc", "mono_camera", "pair_calib"))
        if conf:
            r.update(rule_status="blocked", reason=f"conflict: cameras {', '.join(conf)} (the data contradicts your type)")
        elif unsure and not cams_done:
            r.update(rule_status="blocked", reason=f"camera types needed: {', '.join(unsure)}")
        else:
            r["exists"] = True
            notes = []
            if unconf:
                notes.append(f"{len(unconf)} not confirmed ({', '.join(unconf)}): the camera stages "
                             + ("used" if cams_done else "will use")
                             + " the general model (handheld + zoom) for them until you confirm")
            if refused:
                notes.append("the cameras stage could not use your type for " + "; ".join(
                    f"{t['cam']} ({t['model_check']['declared']}: {t['model_check'].get('why') or 'refused'}; kept "
                    f"{t['model_check'].get('model')})" for t in refused) + " - change the answer or keep it")
            if notes:
                r["reason"] = "; ".join(notes)
        r["refused"] = [t["cam"] for t in refused]
        return r
    if rule == "alignment":
        # config-like: a take may point `alignment` at the studio-derived clocks (written AFTER song_align), so its
        # mtime says nothing about the order of the clock stages
        p = takes.alignment_path(cfg)
        r["config"] = True
        if p and p.exists():
            r.update(exists=True, t=mtime(p), detail=[{"path": paths.rel_display(p), "exists": True, "count": 1,
                                                       "mtime": mtime(p)}])
        else:
            r["missing"] = ["alignment (clock/alignment.json)"]
        return r
    if rule == "window":
        r["config"] = True
        al = takes.alignment_path(cfg)
        if not (al and al.exists()):
            # every dancecap stage loads the take with its clocks: nothing after the window can run without them
            r["reason"] = "waits on the clocks: the alignment file does not exist yet (audio_sync)"
            return r
        r["exists"] = takes.window(cfg) is not None
        if not r["exists"]:
            sug = suggested_window(cfg)
            r.update(rule_status="blocked", reason="set the analysis window (reference seconds) in the take TOML"
                     + (f"; suggested: the maximum overlap {sug[0]:.2f}-{sug[1]:.2f} s ({sug[2]} cameras)" if sug else ""))
        return r
    if rule == "labels_review":
        r["config"] = True
        ij = takes.intake_json(take)
        r["exists"] = bool(ij.get("labels_reviewed_at")) or (paths.take_work(take) / "dancers" / "labels_override.json").exists()
        return r
    if rule == "narrate":
        human, _ = _labels_kind(take)
        r["exists"] = human
        p = paths.take_work(take) / "moves" / "labels.json"
        r["t"] = mtime(p) if human else None
        if not human and p.exists():
            r["reason"] = "only placeholder / automatic labels so far"
        return r
    if rule == "classifier":
        _, auto = _labels_kind(take)
        r["exists"] = auto
        return r
    if rule == "manual":
        r.update(rule_status="n/a", reason=stage.get("manual_hint") or "manual step")
        return r
    if rule == "optional_human":
        dancers = (cfg.get("dancers") or {})
        has = any(isinstance(d, dict) and (d.get("shape_edits") or d.get("shoe_geometry")) for d in dancers.values())
        r.update(exists=has, config=True)
        if not has:
            r.update(rule_status="skipped", reason="optional: no shape edits or shoe geometry")
        return r
    ok, t, missing, detail = _files(stage, take)
    r.update(exists=ok, t=t, missing=missing, detail=detail,
             t_max=max((x["mtime"] for x in detail if x.get("mtime")), default=None))
    if not stage.get("done_when"):
        r.update(rule_status="n/a" if stage.get("implemented") else None, reason="no output contract")
    orule = stage.get("optional_rule")
    if not ok and orule == "soundtrack" and not ctx["soundtrack"]:
        r.update(rule_status="skipped", reason="no soundtrack: the phone clocks and grid are used")
    if not ok and orule == "portraits" and not ctx["portraits"]:
        r.update(rule_status="skipped", reason="no portraits")
    return r


def _tmax(r: dict) -> float | None:
    """Newest output of a stage: what its dependents compare against (any rewrite of an input counts)."""
    return r.get("t_max") if r.get("t_max") is not None else r.get("t")


def suggested_window(cfg: dict) -> tuple[float, float, int] | None:
    """The first maximum-overlap interval of the take's clock file (audio_sync / the prototype write it)."""
    al = takes.alignment(cfg) or {}
    mo = al.get("maximum_overlap") or []
    if mo and isinstance(mo[0], dict) and mo[0].get("end") is not None:
        try:
            return float(mo[0]["start"]), float(mo[0]["end"]), int(mo[0].get("count") or 0)
        except (TypeError, ValueError):
            return None
    return None


def _applied_types(take: str) -> tuple[dict, dict]:
    """({stage: {cam: class}}, {stage: policy}) the type-reading stages applied. cameras/report.json
    `camera_types` {policy, applied} is written by every `dancecap cameras` run (CLI, pod session or orchestrator), so
    a re-run from anywhere clears the staleness; the orchestrator's side file ({stage: {cam: class}}, `policy`:
    {stage: policy}) covers the stages without a report (mono_camera, pair_calib) and older reports. The newer record
    wins. A camera missing from a record had the general class (report.json lists the typed cameras only)."""
    side_p = paths.pipeline_dir(take) / "camera_types_applied.json"
    side = read_json(side_p, {}) or {}
    side = side if isinstance(side, dict) else {}
    pol = side.get("policy") if isinstance(side.get("policy"), dict) else {}
    out = {k: v for k, v in side.items() if k != "policy" and isinstance(v, dict)}
    pols = {k: v for k, v in pol.items() if isinstance(v, str)}
    rp = paths.take_work(take) / "cameras" / "report.json"
    rep = read_json(rp, {}) or {}
    ctr = rep.get("camera_types") if isinstance(rep, dict) else None
    ap = ctr.get("applied") if isinstance(ctr, dict) else None
    if isinstance(ap, dict) and (mtime(rp) or 0) >= (mtime(side_p) or 0) - 1:
        cls = {}
        for cam, v in ap.items():
            c = v.get("class") if isinstance(v, dict) else v
            if isinstance(c, str):
                cls[str(cam)] = c
        for sid in ("cam_loc", "cam_world"):
            out[sid] = dict(cls)
            pols[sid] = str(ctr.get("policy") or "user")
    return out, pols


def _last_runs(take: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in runs.list_runs(take=take, limit=300):
        if r.get("kind") not in ("stage", "probe", "export", "session"):
            continue  # a dry run starts nothing: its rc (3 at any gate) says nothing about a stage
        for s in r.get("stages") or []:
            if s not in out and r.get("kind") != "pipeline":
                out[s] = r
    return out


def progress_events(take: str, stage_id: str, run_id: str) -> tuple[dict | None, list[dict]]:
    """(latest progress, per-step summary) from the dancecap.progress events of one run (L2 instrumented stages):
    work/<take>/progress/<stage>.<run>[.<pid>].jsonl."""
    import glob as _glob
    files = _glob.glob(str(paths.take_work(take) / "progress" / f"{stage_id}.{run_id}*.jsonl"))
    steps: dict[str, dict] = {}
    last = None
    for f in files:
        for line in estimate.read_log_tail(Path(f), 65536).splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict) or "step" not in ev:
                continue
            st = steps.setdefault(ev["step"], {"step": ev["step"], "state": "running"})
            k = ev.get("kind")
            if k in ("tick", "end", "start") and ev.get("total"):
                st.update(done=ev.get("done"), total=ev.get("total"), unit=ev.get("unit"))
                if ev.get("done") is not None:
                    st["frac"] = round(min(1.0, float(ev["done"]) / float(ev["total"])), 4)
                last = st
            if k == "end":
                st["state"] = "done"
            elif k == "fail":
                st["state"] = "failed"
            if ev.get("msg"):
                st["msg"] = str(ev["msg"])[:200]
    prog = None
    if last:
        prog = {k: last.get(k) for k in ("step", "done", "total", "unit", "frac", "msg")}
    return prog, list(steps.values())


def _progress_from_log(stage: dict, run: dict) -> dict | None:
    try:
        prog, _ = progress_events(run.get("take") or "", stage["id"], run["id"])
        if prog:
            return prog
    except Exception:  # noqa: BLE001 - progress never breaks the status
        pass
    pat = (stage.get("progress") or {}).get("pattern")
    if not pat:
        return None
    import re
    try:
        rx = re.compile(pat)
    except re.error:
        return None
    from .runs import run_dir
    tail = estimate.read_log_tail(run_dir(run["id"]) / "output.log", 32768)
    last = None
    for line in tail.splitlines():
        m = rx.search(line.strip())
        if m:
            last = m
    if not last:
        return None
    g = last.groups()
    return {"step": f"{stage['id']}:{g[0]}" if g else stage["id"], "done": None, "total": None, "unit": "cams",
            "msg": last.group(0)[:120]}


def context(take: str) -> dict:
    cfg, etag = takes.load_cfg(take)
    var = takes.variant(take, cfg)
    vids = takes.videos(take, cfg)
    used = [v["cam"] for v in vids if not v["excluded"]]
    types, _ = camtypes.load(take, used)
    snd = takes.soundtrack(cfg)
    portraits = any(d.get("portrait") for d in takes.dancers(cfg).values())
    if not portraits:
        sdir = takes.source_dir(cfg)
        names = {(d.get("name") or "").lower() for d in takes.dancers(cfg).values()}
        portraits = bool(sdir and sdir.exists() and any((sdir / f"{n}.jpg").exists() for n in names if n))
    probs = []
    ds = takes.dancers(cfg)
    for role in takes.ROLES:
        if not (ds.get(role) or {}).get("stature_m"):
            probs.append(f"{role}: stature missing")
    dance = cfg.get("dance") or {}
    if dance.get("level") not in takes.LEVELS:
        probs.append("dance level not set")
    if dance.get("category") not in takes.CATEGORIES:
        probs.append("dance category not set")
    if cfg.get("_error"):
        probs.append(cfg["_error"])
    return {"cfg": cfg, "etag": etag, "variant": var, "videos": vids, "used": used, "types": types,
            "soundtrack": bool(snd and snd["present"]), "portraits": portraits, "intake_problems": probs,
            "wl": estimate.workload(take, cfg), "raw_exists": {}}


def snapshot(take: str, ctx: dict | None = None, with_activity: bool = True, with_runs: bool = True) -> dict:
    """PipelineSnapshot. with_runs=False / with_activity=False judge the files only (the orchestrator uses that to
    check a stage it just ran)."""
    now = time.time()
    ctx = ctx or context(take)
    cfg, var, types = ctx["cfg"], ctx["variant"], ctx["types"]
    stages = registry.variant_stages(var)
    order = [s["id"] for s in stages]
    by = {s["id"]: s for s in stages}
    children: dict[str, list[str]] = {sid: [] for sid in order}
    for s in stages:
        for d in s["deps"]:
            children.setdefault(d, []).append(s["id"])

    # pass 1: raw (camera_types needs to know whether the camera stages already ran)
    raw: dict[str, dict] = {}
    for sid in ("cam_loc", "mono_camera", "pair_calib"):
        if sid in by:
            ctx["raw_exists"][sid] = _files(by[sid], take)[0]
    for s in stages:
        raw[s["id"]] = _raw(s, take, cfg, ctx)
    # pass 2: implied by done dependents (reverse order)
    implied: dict[str, str] = {}
    for sid in reversed(order):
        r = raw[sid]
        if r["exists"] or r["rule_status"] in ("skipped", "n/a", "blocked") or by[sid].get("no_implied"):
            continue
        dep_done = [c for c in children.get(sid, []) if (raw[c]["exists"] and not raw[c]["config"]) or c in implied]
        if dep_done:
            implied[sid] = dep_done[0]

    active_runs = runs.list_runs(take=take, active=True) if with_runs else []
    running_stage: dict[str, dict] = {}
    for r in active_runs:
        if r.get("kind") == "dry_run":
            continue  # nothing runs in a dry run
        cur = (r.get("current") or {}).get("stage")
        for s in ([cur] if cur else []) + ([] if r.get("kind") == "pipeline" else r.get("stages") or []):
            if s in by:
                running_stage.setdefault(s, {"run": r})
    if with_activity:
        for s, act in runpod_view.stage_activity(take).items():
            if s in by:
                running_stage.setdefault(s, {"session": act})
    last = _last_runs(take) if with_runs else {}
    applied, applied_policy = _applied_types(take)
    accepted = read_json(paths.pipeline_dir(take) / "accepted_inputs.json", {}) or {}
    gate_doc = gates.load(take)
    tmap = {t["cam"]: t for t in types}
    prices = None

    out_stages, status = [], {}
    semantic_stale: dict[str, str] = {}
    for s in stages:
        sid = s["id"]
        r = raw[sid]
        est = estimate.stage_estimate(s, ctx["wl"], types if sid == "cam_loc" else None, prices)
        st_reason = r.get("reason")
        stale_because: list[str] = []
        soft_newer: list[str] = []
        upstream_stale: list[str] = []
        planned = not s.get("implemented")
        if r["rule_status"] in ("skipped", "n/a"):
            st = r["rule_status"]
        elif r["exists"] or sid in implied:
            if planned and not r["exists"]:
                st, st_reason = "skipped", f"not built yet; the take was processed without it ({implied[sid]} is done)"
            else:
                st = "done"
                if not r["exists"]:
                    st_reason = f"implied: {implied[sid]} is done (outputs not found: {', '.join(r['missing'][:3])})"
                t_me = r["t"]
                for d in s["deps"]:
                    rd = raw.get(d, {})
                    if d in semantic_stale:
                        stale_because.append(f"{d}: {semantic_stale[d]}")
                    elif t_me is not None and _tmax(rd) is not None and not rd.get("config") \
                            and _tmax(rd) > t_me + TOL_S and not accepted.get(sid, {}).get(d, 0) >= _tmax(rd) - 0.5:
                        stale_because.append(f"{d} is newer")
                    elif status.get(d) == "stale":
                        upstream_stale.append(d)
                for d in s.get("soft_deps", []):
                    rd = raw.get(d, {})
                    if t_me is not None and _tmax(rd) is not None and _tmax(rd) > t_me + TOL_S:
                        soft_newer.append(f"{d} is newer (soft input)")
                if sid in TYPE_STAGES and r["exists"]:
                    rec = applied.get(sid) or {}
                    pol = applied_policy.get(sid) or "user"  # compare under the policy that run used
                    for cam, t in tmap.items():
                        now_cls = camtypes.effective_class(t, pol)
                        before = rec.get(cam) or "handheld_zoom"
                        if before != now_cls:
                            stale_because.append(f"camera type {cam} changed ({before} -> {now_cls})")
                            semantic_stale[sid] = f"camera type {cam} changed"
                if any(x.split(":")[0] in semantic_stale for x in stale_because) and sid not in semantic_stale:
                    semantic_stale[sid] = "upstream camera-type change"
                g = gate_doc.get(sid) if s.get("gate_human") else None
                if isinstance(g, dict) and g.get("decision") == "reject" and g.get("confirmed_at") and \
                        (r["t"] or 0) <= float(g["confirmed_at"]) + TOL_S:
                    stale_because.append("you rejected its review: re-run it (or its inputs)")
                if stale_because:
                    st = "stale"
        elif r["rule_status"] == "blocked":
            st = "blocked"
        elif planned:
            st = "planned"
            st_reason = s.get("planned_reason") or "proposed in docs/PIPELINE.md; not built yet"
        else:
            deps_ok = all(status.get(d) in SATISFIED for d in s["deps"])
            bad = [d for d in s["deps"] if status.get(d) not in SATISFIED]
            opt = s.get("optional") and not s.get("optional_rule")
            if opt and not ctx.get("include_optional", set()) & {sid}:
                st = "skipped"
                st_reason = f"optional: {s['optional']}" if isinstance(s.get("optional"), str) else "optional"
            elif not deps_ok:
                st = "waiting"
                st_reason = "waits on " + ", ".join(f"{d} ({status.get(d)})" for d in bad[:4])
            elif s.get("runs_on") == "human":
                st = "blocked"
                st_reason = st_reason or {"labels_review": "look at dancers/labels_review.jpg and confirm (or write "
                                                           "dancers/labels_override.json)",
                                          "narrate": "narrate the moves (Narrate moves)"}.get(sid, "needs you")
            elif sid in GATED_BY_TYPES and any(camtypes.needs_user(t) for t in types):
                st = "blocked"
                st_reason = "camera types needed for " + ", ".join(t["cam"] for t in types if camtypes.needs_user(t))
            else:
                st = "ready"
        open_gate = gates.is_open(take, sid, gate_doc) if s.get("gate_human") else None
        if open_gate and st in ("done", "stale"):
            st, st_reason = "blocked", (f"review gate: {open_gate.get('message') or s['gate_human']} - confirm it "
                                        "(stage drawer) before anything after it runs")
        # failed / running override
        lr = last.get(sid)
        if lr and lr.get("state") == "failed" and st in ("done", "stale", "ready", "waiting") and \
                (r["t"] is None or (lr.get("ended") or 0) > r["t"] + TOL_S):
            st, st_reason = "failed", f"last run {lr['id']} failed (rc {lr.get('rc')})"
        progress, eta = None, None
        act = running_stage.get(sid)
        if act:
            st = "running"
            if "run" in act:
                rr = act["run"]
                st_reason = f"run {rr['id']}"
                progress = (rr.get("current") or {}).get("progress") or _progress_from_log(s, rr)
                elapsed = now - (rr.get("started") or rr.get("created") or now)
                frac = progress.get("frac") if progress else None
                eta = estimate.live_blend(est["p50_s"], est["p90_s"], frac, elapsed)
            else:
                ss = act["session"]
                st_reason = f"session {ss['sid']}: {ss.get('phase') or ss.get('job')}"
                p = ss.get("progress")
                if p and p.get("total"):
                    progress = dict(p, frac=round(p["done"] / p["total"], 3) if p.get("done") is not None else None)
                e = ss.get("eta") or {}
                eta = {"p10_s": e.get("p10"), "p50_s": e.get("p50"), "p90_s": e.get("p90"),
                       "basis": e.get("basis", "session"), "confidence": "session"} if e else None
        status[sid] = st
        cam_split = None
        if sid in CAMERA_STAGES or sid in ("mamma_prepare", "cam_refine_body"):
            cam_split = {k: [] for k in camtypes.CLASS_EFFECTS}
            for t in types:
                cam_split[t["effective"]["class"]].append(t["cam"])
        gates_ = []
        if sid in GATED_BY_TYPES:
            open_ = any(camtypes.needs_user(t) for t in types)
            gates_.append({"id": "camera_types", "open": open_ and st not in ("done",),
                          "message": "camera types uncertain or in conflict" if open_ else None})
        if s.get("runs_on") in ("pod", "pod_secure"):
            cc = takes.cloud_consent(cfg)
            gates_.append({"id": "consent", "open": not cc["ok"], "message": None if cc["ok"] else
                          "cloud_processing consent not recorded for every dancer ([consent] in the take TOML)"})
        if s.get("gate_human"):
            gates_.append({"id": "review", "open": bool(open_gate), "message": s["gate_human"],
                           "decisions": list(gates.DECISIONS.get(sid, ("ok",))) + ["reject"]})
        actions = []
        if s.get("runnable") == "local" and st not in ("running",):
            actions += ["run", "run_upto", "dry_run"]
        if s.get("runnable") == "session" and st not in ("running",):
            actions += ["plan", "dry_run"]
        if st == "running":
            actions.append("stop")
        if st == "stale" and not any("camera type" in x or "review" in x for x in stale_because):
            actions.append("accept_inputs")
        if open_gate:
            actions.append("confirm_gate")
        actions.append("reveal")
        out_stages.append({
            "id": sid, "name": s.get("name"), "column": s["column"], "runs_on": s.get("runs_on"),
            "session": s.get("session"), "optional": bool(s.get("optional")), "in_variant": True,
            "implemented": bool(s.get("implemented")), "runnable": s.get("runnable"),
            "status": st, "status_reason": st_reason, "progress": progress,
            "eta": eta or ({"p50_s": est["p50_s"], "p90_s": est["p90_s"], "basis": est["basis"],
                            "confidence": est["confidence"]} if st not in ("done", "skipped", "n/a", "planned") else None),
            "estimate": est, "cost": {"p50_usd": est["cost_p50_usd"], "p90_usd": est["cost_p90_usd"]},
            "camera_types": cam_split,
            "last_run": {"run": lr["id"], "rc": lr.get("rc"), "ended": lr.get("ended"), "state": lr.get("state")} if lr else None,
            "outputs": {"fresh": st == "done" and not upstream_stale, "stale_because": stale_because,
                        "upstream_stale": upstream_stale, "soft_newer": soft_newer,
                        "missing": r["missing"], "detail": r["detail"]},
            "gates": gates_, "actions": actions, "deps": s["deps"], "soft_deps": s.get("soft_deps", []),
        })

    edges = registry.edges(stages)
    groups = []
    for sess in registry.POD_SESSIONS:
        members = [x["id"] for x in out_stages if x.get("session") == sess]
        if members:
            act_sid = next((running_stage[m]["session"]["sid"] for m in members
                            if m in running_stage and "session" in running_stage[m]), None)
            groups.append({"id": sess, "stages": members, "sid": act_sid})
    remaining, cp = _critical_path(out_stages)
    pod_cost50 = sum(x["cost"]["p50_usd"] for x in out_stages if x["status"] not in ("done", "skipped", "n/a", "planned"))
    pod_cost90 = sum(x["cost"]["p90_usd"] for x in out_stages if x["status"] not in ("done", "skipped", "n/a", "planned"))
    branches = [{"stage": x["id"], "label": "risk", "options": x["estimate"]["branches"]} for x in out_stages
                if x["estimate"].get("branches") and x["status"] not in ("done", "skipped", "n/a")]
    gates_open = []
    for x in out_stages:
        if x["status"] == "blocked":
            gates_open.append({"id": x["id"], "stage": x["id"], "message": x["status_reason"],
                               "link": (registry.OVERLAY.get(x["id"], {}).get("ui") or f"#/take/{take}/stage/{x['id']}")
                               .replace("{take}", take)})
    nxt = []
    if any(g["id"] == "camera_types" for g in gates_open):
        nxt.append({"label": "Confirm camera types", "route": f"#/take/{take}/videos"})
    if any(x["status"] == "ready" and x["runnable"] == "local" for x in out_stages):
        tgt = "demo_export" if "demo_export" in by else out_stages[-1]["id"]
        nxt.append({"label": f"Plan & run up to {tgt}", "plan": {"target": tgt, "scope": "upto", "mode": "resume"}})
    return {"take": take, "variant": var, "columns": registry.COLUMNS, "stages": out_stages, "edges": edges,
            "groups": groups, "critical_path": cp, "branches": branches,
            "remaining": remaining, "pod_cost_remaining_usd": {"p50": round(pod_cost50, 2), "p90": round(pod_cost90, 2)},
            "gates_open": gates_open, "next_actions": nxt, "workload": ctx["wl"], "toml_etag": ctx["etag"],
            "computed_at": now}


def _critical_path(stages: list[dict]) -> tuple[dict, list[str]]:
    """Longest p50 path through the stages still to do (done/skipped/planned stages cost 0)."""
    todo = {x["id"]: x for x in stages if x["status"] not in ("done", "skipped", "n/a", "planned")}
    best: dict[str, tuple[float, float, list[str]]] = {}
    for x in stages:
        sid = x["id"]
        w50 = w90 = 0.0
        if sid in todo:
            e = x.get("eta") or {}
            w50, w90 = float(e.get("p50_s") or 0), float(e.get("p90_s") or 0)
        prev = max((best[d] for d in x["deps"] if d in best), key=lambda b: b[0], default=(0.0, 0.0, []))
        best[sid] = (prev[0] + w50, prev[1] + w90, prev[2] + ([sid] if sid in todo else []))
    if not best:
        return {"p10_s": 0, "p50_s": 0, "p90_s": 0}, []
    end = max(best.values(), key=lambda b: b[0])
    return {"p10_s": round(end[0] * 0.7), "p50_s": round(end[0]), "p90_s": round(end[1])}, end[2]


def summary(take: str, snap: dict | None = None, ctx: dict | None = None) -> dict:
    """TakeSummary (PIPELINE_MONITOR 7.3)."""
    ctx = ctx or context(take)
    snap = snap or snapshot(take, ctx)
    cfg = ctx["cfg"]
    counts = {k: 0 for k in ("done", "stale", "running", "failed", "blocked", "waiting", "ready", "queued", "skipped",
                             "planned", "n/a")}
    for x in snap["stages"]:
        counts[x["status"]] = counts.get(x["status"], 0) + 1
    counts["total"] = len(snap["stages"])
    ct = {"confirmed": 0, "suggested": 0, "unset": 0, "conflict": 0, "refused": 0}
    for t in ctx["types"]:
        ct[t["state"]] = ct.get(t["state"], 0) + 1
        ct["refused"] += bool((t.get("model_check") or {}).get("refused"))
    dance = cfg.get("dance") or {}
    act = runs.list_runs(take=take, active=True)
    mt = [m for m in (mtime(paths.take_toml(take)), mtime(paths.pipeline_dir(take) / "events.jsonl")) if m]
    return {"take": take, "title": takes.title(cfg), "level": dance.get("level"), "category": dance.get("category"),
            "variant": snap["variant"], "n_videos": len(ctx["videos"]), "n_used": len(ctx["used"]),
            "camera_types": ct, "counts": counts, "strip": [[x["id"], x["status"]] for x in snap["stages"]],
            "active_runs": [r["id"] for r in act], "remaining": snap["remaining"],
            "pod_cost_remaining_usd": snap["pod_cost_remaining_usd"], "gates_open": len(snap["gates_open"]),
            "last_activity": max(mt) if mt else None, "problems": ctx["intake_problems"]}


def accept_inputs(take: str, stage: str) -> dict:
    """The user accepts a stage's outputs although an input is newer (e.g. a QA-only rewrite upstream): record the
    inputs' current times in work/<take>/pipeline/accepted_inputs.json; a later change makes it stale again."""
    from .util import write_json_atomic
    ctx = context(take)
    by = {s["id"]: s for s in registry.variant_stages(ctx["variant"])}
    if stage not in by:
        raise KeyError(stage)
    rec = {}
    for d in by[stage]["deps"]:
        r = _raw(by[d], take, ctx["cfg"], ctx)
        if _tmax(r) is not None and not r.get("config"):
            rec[d] = _tmax(r)
    p = paths.pipeline_dir(take) / "accepted_inputs.json"
    allr = read_json(p, {}) or {}
    allr[stage] = rec
    write_json_atomic(p, allr)
    return rec
