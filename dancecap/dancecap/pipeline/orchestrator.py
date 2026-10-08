"""The orchestrator: runs a plan's stages in DAG order with the workstation's limits, resumable.

    python -m dancecap.pipeline run <take> --plan <work/pipeline/plans/<hash>.json> [--dry-run]
    python -m dancecap.pipeline run <take> --target demo_export [--scope upto] [--mode resume] [--dry-run]
    python -m dancecap.pipeline run <take> --what-if        (dry run of the whole pipeline, reviews assumed done)

- Order: the plan's steps (topological). A step starts when all its deps inside the plan finished; deps outside the
  plan were checked by the plan (skip_done) and are re-checked here from the files. A SOFT dep ("uses it when
  present": demo_export <- hair_groom / triggers, triggers <- floorcraft / moves_run, narrate <- moves_proposals) that
  runs in the same run is waited for too, whatever its result, so the step gets the better input (never a deadlock:
  a soft dep that is also a descendant is ignored).
- Limits (PIPELINE_MONITOR 4.4): at most 2 local CPU stages at a time, a `heavy` stage alone; 1 local GPU stage
  (stage code also takes work/gpu.lock); exports wait while work/unity.lock/ exists; 1 pod session at a time, and a
  pod session only when the plan carries the user's approval (`approved`, written by the server after the plan
  dialog, bound to this run id, one-shot). Without it the session steps are reported as blocked: this process never
  creates a pod on its own. The limits hold ACROSS processes (locks.py): one orchestrator per take
  (work/<take>/pipeline/run.lock, CLI or monitor), the local slots and the pod-session lock are shared by every
  orchestrator, and a `remote run` started elsewhere also counts as the one pod session.
- Human gates and stages that are not built yet halt their descendants; a gated stage the run itself ran (fit_qa,
  geoveto: gates.py) opens its review gate and halts everything after it. The run ends with rc 3 (waiting_human)
  when nothing else can run, rc 1 when a stage failed, rc 130 when stopped, else 0.
- Every stage command is its own `stage` run (work/pipeline/runs/<id>/, parent = this run), so each has its log, rc
  and stop button. Multi-command stages run their commands in sequence.
- State: work/<take>/pipeline/state.json (per-stage state, rewritten atomically) + events.jsonl (append-only).
  Stop: stop.json of this run, mode after_step (finish the running steps, start nothing new) or now (runwrap
  interrupts this process; running LOCAL children get their own stop.json and are waited for; a running pod session
  is never stopped here - `now` is not offered while one runs, and if the run is interrupted anyway the session run
  keeps going on its own and is reported as detached).
- --dry-run: no process at all - logs what would run, in order, with the same limits. A what-if plan
  (options.assume_reviews) is dry-run only: its `pause` steps (labels_review, narrate, ...) are logged as "a real run
  pauses here for you" and the run goes on as if they were done, through the review gates, to the end of the plan.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import camtypes, gates, locks, paths, plan as planmod, registry, runs, status, takes
from .util import append_jsonl, read_json, write_json_atomic

MAX_LOCAL = 2
POLL_S = 1.0


class Orchestrator:
    def __init__(self, take: str, plan: dict, dry_run: bool = False, run_id: str | None = None,
                 poll_s: float = POLL_S):
        self.take, self.plan, self.dry = take, plan, dry_run
        self.run_id = run_id or os.environ.get("DANCECAP_RUN_ID")
        self.poll_s = poll_s
        self.pdir = paths.pipeline_dir(take)
        self.pdir.mkdir(parents=True, exist_ok=True)
        self.events = self.pdir / "events.jsonl"
        self.state_p = self.pdir / "state.json"
        vstages = registry.variant_stages(plan["variant"])
        self.vby = {s["id"]: s for s in vstages}
        self.order = [s["id"] for s in vstages]
        self.steps = {st["stage"]: st for st in plan["steps"]}
        acts = ("run", "rerun_stale", "pause") if dry_run else ("run", "rerun_stale")
        self.todo = [sid for sid in self.order if self.steps.get(sid, {}).get("action") in acts]
        self.blocked = {sid for sid, st in self.steps.items() if st["action"] == "blocked"}
        self.state = {"take": take, "run": self.run_id, "plan_hash": plan.get("plan_hash"), "dry_run": dry_run,
                      "started": time.time(), "updated": time.time(), "status": "running",
                      "stages": {sid: {"state": "pending"} for sid in self.todo}}
        for sid in self.blocked:
            self.state["stages"][sid] = {"state": "blocked", "reason": self.steps[sid].get("reason")}
        self.active: dict[str, dict] = {}  # sid -> {run, cmds, i, kind, session}
        self.finished: dict[str, str] = {}  # sid -> ok | failed | blocked | skipped
        self.stopping: str | None = None
        self.ctx_cmd = {"take": take, "capture": plan.get("capture"), "tag": plan.get("tag")}
        self.held: dict[str, list[Path]] = {}  # sid -> slot / pod-session locks held for it
        self.owner = {"take": take, "run": self.run_id}
        self._soft: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ bookkeeping
    def event(self, kind: str, stage: str | None = None, **kw) -> None:
        ev = {"ts": time.time(), "run": self.run_id, "kind": kind, "stage": stage, **kw}
        append_jsonl(self.events, ev)
        print(f"[pipeline] {kind} {stage or ''} {kw.get('msg') or ''}".rstrip(), flush=True)

    def save(self) -> None:
        self.state["updated"] = time.time()
        self.state["current"] = sorted(self.active)
        write_json_atomic(self.state_p, self.state)
        if self.run_id:
            done = sum(1 for v in self.finished.values() if v == "ok")
            runs.set_current(self.run_id, stage=next(iter(sorted(self.active)), None),
                             progress={"frac": round(done / max(1, len(self.todo)), 3), "done": done,
                                       "total": len(self.todo), "unit": "stages"},
                             session_active=any(a.get("run_kind") == "session" for a in self.active.values()))

    def set_stage(self, sid: str, **kw) -> None:
        self.state["stages"].setdefault(sid, {}).update(kw)

    def stop_requested(self) -> str | None:
        if not self.run_id:
            return None
        try:
            d = read_json(runs.run_dir(self.run_id) / "stop.json", None)
        except ValueError:
            return None
        return (d or {}).get("mode") if d else None

    # ------------------------------------------------------------------ scheduling
    def deps_ok(self, sid: str) -> bool | None:
        """True: can start; False: not yet; None: never (a dep failed or is blocked)."""
        for d in self.vby[sid]["deps"]:
            if d in self.todo:
                f = self.finished.get(d)
                if f is None:
                    return False
                if f != "ok":
                    return None  # failed, blocked, skipped, or a review gate this run opened
            elif d in self.blocked:
                return None
        for d in self.soft_wait(sid):
            if self.finished.get(d) is None:
                return False  # a soft dep that runs in this run: wait for it (its result does not matter)
        return True

    def soft_wait(self, sid: str) -> list[str]:
        """Soft deps of `sid` that run in this run (never one that is also a descendant of `sid`)."""
        c = self._soft.get(sid)
        if c is None:
            soft = [d for d in self.vby[sid].get("soft_deps") or [] if d in self.todo]
            if soft:
                desc = set(registry.descendants(list(self.vby.values()), [sid]))
                soft = [d for d in soft if d not in desc]
            c = self._soft[sid] = soft
        return c

    def limits_allow(self, sid: str) -> tuple[bool, str | None]:
        s = self.vby[sid]
        if self.steps.get(sid, {}).get("action") == "pause":
            return True, None  # what-if dry run: a human step takes no slot
        on = s.get("runs_on")
        act = [self.vby[a] for a in self.active]
        if s.get("runnable") == "session":
            if any(a.get("runnable") == "session" for a in act):
                return False, "one pod session at a time"
            if not self.dry:
                h = locks.holder(locks.pod_lock())
                if h is not None:
                    return False, f"one pod session at a time: {h.get('take')} ({h.get('session')}) runs one"
                if locks.external_session_runner():
                    return False, "one pod session at a time: a session started elsewhere is running"
            return True, None
        if on == "local_gpu":
            if any(a.get("runs_on") == "local_gpu" for a in act):
                return False, "one local GPU stage at a time"
        locals_ = [a for a in act if a.get("runs_on") in ("local", "local_gpu") and a.get("runnable") == "local"]
        if any(a.get("heavy") for a in locals_):
            return False, "a heavy stage runs alone"
        if s.get("heavy") and locals_:
            return False, "heavy: waits for the other local stages"
        if len(locals_) >= MAX_LOCAL:
            return False, f"{MAX_LOCAL} local stages at a time"
        if s.get("unity_lock") and (paths.work() / "unity.lock").exists() and \
                not self.plan["request"]["options"].get("override_lock"):
            if self.dry:
                return True, "work/unity.lock is held now: a real run would wait for it"
            if not self.active:  # nothing else running: do not spin on a lock that may be held for hours
                return None, "work/unity.lock is held (Unity work in progress): export later or override"
            return False, "work/unity.lock is held (Unity work in progress)"
        return True, None

    def session_members(self, sid: str) -> list[str]:
        sess = self.vby[sid].get("session")
        return [x for x in self.todo if self.vby[x].get("session") == sess and x not in self.finished]

    def start(self, sid: str) -> None:
        s = self.vby[sid]
        if self.steps.get(sid, {}).get("action") == "pause":  # what-if dry run only (todo has no pause otherwise)
            self.active[sid] = {"cmds": [], "i": 0, "kind": "pause", "members": [sid], "run": None,
                                "dry_until": time.time() + 0.05}
            self.set_stage(sid, state="running", started=time.time())
            self.event("would_pause", sid, msg=self.steps[sid].get("reason") or "a real run pauses here for you")
            return
        if s.get("runnable") == "session":
            members = [m for m in self.session_members(sid) if self.deps_ok_outside(m, sid)]
            self.start_session(s["session"], members)
            return
        cmds = registry.commands(sid, self.ctx_cmd)
        self.active[sid] = {"cmds": cmds, "i": 0, "kind": "stage", "members": [sid]}
        self.set_stage(sid, state="running", started=time.time())
        self.event("start", sid, msg=" ".join(cmds[0]))
        self.launch(sid)

    def session_ready(self, leader: str) -> bool | None:
        """A pod session waits until the members it can batch are ready (one upload, one pod). A member whose
        pending input itself needs this session's results (texture <- fit_qa <- mamma_fit) does not hold the session
        back: it runs in a later session of the same group (plan.py prices that second wave)."""
        sess = self.vby[leader].get("session")
        members = self.session_members(leader)
        vst = list(self.vby.values())
        for m in members:
            pend = [d for d in self.vby[m]["deps"] if d in self.todo and self.vby[d].get("session") != sess
                    and self.finished.get(d) is None]
            if pend and not any(sm in registry.ancestors(vst, d) for d in pend for sm in members):
                return False
        return True

    def deps_ok_outside(self, member: str, leader: str) -> bool:
        """A session member may join the session when its deps are done or inside the same session."""
        sess = self.vby[leader].get("session")
        for d in self.vby[member]["deps"]:
            if d in self.todo and self.vby[d].get("session") == sess:
                continue
            if d in self.todo and self.finished.get(d) != "ok":
                return False
            if d in self.blocked:
                return False
        return True

    def start_session(self, sess: str, members: list[str]) -> None:
        approved = self.plan.get("approved")
        if not approved and not self.dry:
            for m in members:
                self.finished[m] = "blocked"
                self.set_stage(m, state="blocked", reason="pod session not approved in the plan")
            self.event("blocked", members[0], session=sess, msg=f"{sess}: no approval in the plan (no pod created)")
            return
        if not self.dry:
            if not locks.acquire(locks.pod_lock(), session=sess, stages=members, **self.owner):
                raise Busy("one pod session at a time (another pipeline run holds the pod-session lock)")
            self.held[members[0]] = [locks.pod_lock()]
        sp = next((x for x in self.plan.get("sessions", []) if members[0] in x.get("stages", [])),
                  next((x for x in self.plan.get("sessions", []) if x["id"] == sess), {}))
        ctx = dict(self.ctx_cmd, max_cost=sp.get("max_cost_usd"), max_minutes=sp.get("max_minutes"),
                   gpus=self.plan["request"]["options"].get("gpus") or None)
        try:
            prep = registry.session_prep(sess, members, ctx)
            cmds = prep + registry.session_command(sess, members, ctx, dry_run=False)
        except registry.NotRunnable:
            locks.release_all(self.held.pop(members[0], None))
            raise
        lead = members[0]
        self.active[lead] = {"cmds": cmds, "i": 0, "kind": "session", "members": members, "session": sess,
                             "n_prep": len(prep)}
        for m in members:
            self.set_stage(m, state="running", started=time.time(), session=sess)
        self.event("start", lead, session=sess, members=members, msg=f"{sess}: {len(cmds)} command(s)")
        self.launch(lead)

    def launch(self, sid: str) -> None:
        a = self.active[sid]
        argv = a["cmds"][a["i"]]
        if self.dry:
            self.event("would_run", sid, argv=["python", *argv], msg=" ".join(argv))
            a["run"] = None
            a["dry_until"] = time.time() + 0.05
            return
        is_session_cmd = a["kind"] == "session" and a["i"] >= a.get("n_prep", len(a["cmds"]) - 1)
        kind = "session" if is_session_cmd else "stage"
        a["run_kind"] = kind
        r = runs.start(kind, argv, take=self.take, stages=a["members"], plan_hash=self.plan.get("plan_hash"),
                       origin="pipeline", parent=self.run_id, heavy=bool(self.vby[sid].get("heavy")))
        a["run"] = r["id"]
        for m in a["members"]:
            self.set_stage(m, run=r["id"])

    def poll(self, sid: str) -> str | None:
        """None while running; 'ok' or 'failed' when the stage (all its commands) ended."""
        a = self.active[sid]
        if self.dry:
            if time.time() < a.get("dry_until", 0):
                return None
            rc = 0
        else:
            r = runs.describe(a["run"])
            if r is None or r["state"] in ("starting", "running", "stopping"):
                if r is not None and r["state"] == "starting" and time.time() - r["created"] > 120:
                    return "failed"
                return None
            rc = r.get("rc")
            if r["state"] == "lost":
                rc = -1
        if rc != 0:
            self.event("fail", sid, rc=rc, run=a.get("run"), msg=f"command {a['i'] + 1}/{len(a['cmds'])} rc={rc}")
            return "failed"
        a["i"] += 1
        if a["i"] < len(a["cmds"]):
            self.launch(sid)
            return None
        return "ok"

    def complete(self, sid: str, result: str) -> None:
        a = self.active.pop(sid)
        locks.release_all(self.held.pop(sid, None))
        for m in a["members"]:
            res = result
            gate_msg = self.vby[m].get("gate_human")
            if result == "ok" and not self.dry:
                # trust the files: a session may end ok with a job failed inside
                st = status.snapshot(self.take, with_activity=False, with_runs=False)
                got = next((x for x in st["stages"] if x["id"] == m), {})
                if got.get("status") == "stale":
                    self.event("note", m, msg="ran, but its outputs are older than an input: "
                                              + "; ".join((got.get("outputs") or {}).get("stale_because") or []))
                elif got.get("status") == "blocked" and gates.is_open(self.take, m):
                    pass  # outputs written; an earlier review gate of this stage is still open (re-opened below)
                elif got.get("status") not in ("done",):
                    res = "failed"
                    self.event("fail", m, msg=f"outputs not found after the run ({got.get('status')}: "
                                              f"{(got.get('outputs') or {}).get('missing')})")
            if res == "ok" and gate_msg:
                if self.dry:
                    self.event("note", m, msg=f"review gate: a real run halts here until you confirm {m} "
                                              f"({gate_msg})")
                else:
                    gates.open_gate(self.take, m, gate_msg, run=self.run_id)
                    self.finished[m] = "gate"
                    self.set_stage(m, state="done", gate="open", reason=f"review gate: {gate_msg}", ended=time.time())
                    self.event("gate", m, msg=f"waits for your review: {gate_msg} (confirm it in the monitor)")
                    if m in status.TYPE_STAGES:
                        self.record_types(m)
                    continue
            self.finished[m] = res
            self.set_stage(m, state="done" if res == "ok" else "failed", ended=time.time())
            if res == "ok":
                self.event("end", m, msg="dry run" if self.dry else "done")
                if m in status.TYPE_STAGES and not self.dry:
                    self.record_types(m)

    def record_types(self, sid: str) -> None:
        """What a type-reading stage applied ({cam: class} + the policy) -> pipeline/camera_types_applied.json. The
        cameras stage's own record (cameras/report.json camera_types: written where it ran, pod or local) is copied
        when it exists; otherwise the classes under the policy this process passes on (DANCECAP_CAMERA_TYPES, default
        'user' = confirmed answers only - the CLI's and the pod jobs' default too)."""
        cfg, _ = takes.load_cfg(self.take)
        used = takes.used_cams(self.take, cfg)
        types, _ = camtypes.load(self.take, used)
        p = self.pdir / "camera_types_applied.json"
        rec = read_json(p, {}) or {}
        rep = read_json(paths.take_work(self.take) / "cameras" / "report.json", {}) or {}
        ctr = rep.get("camera_types") if isinstance(rep, dict) and sid in ("cam_loc", "cam_world") else None
        if isinstance(ctr, dict) and isinstance(ctr.get("applied"), dict):
            pol = str(ctr.get("policy") or "user")
            ap = ctr["applied"]
            rec[sid] = {t["cam"]: (ap.get(t["cam"]) or {}).get("class", "handheld_zoom")
                        if isinstance(ap.get(t["cam"]), dict) else "handheld_zoom" for t in types}
        else:
            pol = os.environ.get("DANCECAP_CAMERA_TYPES") or "user"
            rec[sid] = {t["cam"]: camtypes.effective_class(t, pol) for t in types}
        rec.setdefault("policy", {})[sid] = pol
        write_json_atomic(p, rec)

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        lk = locks.take_lock(self.take)
        if not locks.acquire(lk, **self.owner, dry_run=self.dry):
            h = locks.holder(lk) or {}
            print(f"[pipeline] another pipeline run of {self.take} is active (pid {h.get('pid')}, run "
                  f"{h.get('run') or 'started from the CLI'}): not starting", file=sys.stderr, flush=True)
            return 2
        try:
            return self._run()
        finally:
            for v in list(self.held.values()):
                locks.release_all(v)
            self.held.clear()
            locks.release(lk)

    def _run(self) -> int:
        self.event("plan", None, msg=f"{len(self.todo)} stage(s) to run, {len(self.blocked)} blocked",
                   todo=self.todo, blocked=sorted(self.blocked), dry_run=self.dry)
        self.save()
        try:
            while True:
                mode = self.stop_requested()
                if mode and (not self.stopping or (mode == "now" and self.stopping == "after_step")):
                    self.stopping = mode
                    self.event("stop", None, msg=f"stop requested ({mode})")
                    if mode == "now":
                        self.stop_children()
                for sid in list(self.active):
                    res = self.poll(sid)
                    if res is not None:
                        self.complete(sid, res)
                        self.save()
                if not self.stopping:
                    for sid in self.todo:
                        if sid in self.finished or sid in self.active:
                            continue
                        if any(sid in a["members"] for a in self.active.values()):
                            continue
                        ok = self.deps_ok(sid)
                        if ok and self.vby[sid].get("runnable") == "session":
                            ok = self.session_ready(sid)
                        if ok is None:
                            gate = [d for d in self.vby[sid]["deps"] if self.finished.get(d) == "gate"]
                            why = (f"waits for your review of {', '.join(gate)}" if gate else
                                   "a dependency failed or is blocked")
                            self.finished[sid] = "skipped"
                            self.set_stage(sid, state="skipped", reason=why)
                            self.event("skip", sid, msg=why)
                            continue
                        if not ok:
                            continue
                        allow, why = self.limits_allow(sid)
                        if allow is None:
                            self.finished[sid] = "blocked"
                            self.set_stage(sid, state="blocked", reason=why)
                            self.event("blocked", sid, msg=why)
                            continue
                        if not allow:
                            self.set_stage(sid, state="queued", reason=why)
                            continue
                        if not self.dry and self.vby[sid].get("runnable") == "local":
                            got, busy = locks.acquire_local(dict(self.owner, stage=sid),
                                                            heavy=bool(self.vby[sid].get("heavy")),
                                                            gpu=self.vby[sid].get("runs_on") == "local_gpu")
                            if got is None:
                                self.set_stage(sid, state="queued", reason=busy)
                                continue
                            self.held[sid] = got
                        if why:
                            self.event("note", sid, msg=why)
                        try:
                            self.start(sid)
                        except Busy as e:
                            locks.release_all(self.held.pop(sid, None))
                            self.set_stage(sid, state="queued", reason=str(e))
                            continue
                        except registry.NotRunnable as e:
                            locks.release_all(self.held.pop(sid, None))
                            self.finished[sid] = "blocked"
                            self.set_stage(sid, state="blocked", reason=str(e))
                            self.event("blocked", sid, msg=str(e))
                        self.save()
                if not self.active and (self.stopping or all(s in self.finished for s in self.todo)):
                    break
                time.sleep(self.poll_s)
        except KeyboardInterrupt:
            self.stopping = "now"
            self.event("stop", None, msg="interrupted")
            self.stop_children()
        return self.finish()

    def stop_children(self, wait_s: float | None = None) -> None:
        """Stop the LOCAL child runs (stop.json now, then wait for their rc.json). A pod-session run is never stopped
        here: it keeps running on its own (download, pod delete, ledger close) and its stages are reported as
        detached; delete its pod in the Runpod panel to stop it."""
        if wait_s is None:
            try:
                wait_s = float(os.environ.get("DANCECAP_STOP_GRACE_S", 30)) + 5
            except ValueError:
                wait_s = 35.0
        local = [a["run"] for a in self.active.values() if a.get("run") and a.get("run_kind") != "session"]
        for rid in local:
            try:
                if not (runs.run_dir(rid) / "rc.json").exists():
                    write_json_atomic(runs.run_dir(rid) / "stop.json", {"mode": "now", "at": time.time(),
                                                                        "by": self.run_id})
            except (ValueError, OSError):
                pass
        t0 = time.time()
        ids = list(local)
        while ids and time.time() - t0 < wait_s:
            ids = [rid for rid in ids
                   if (runs.describe(rid, fresh=True) or {}).get("state") in ("starting", "running", "stopping")]
            if ids:
                time.sleep(0.5)
        for sid in list(self.active):
            a = self.active.pop(sid)
            locks.release_all(self.held.pop(sid, None))
            if a.get("run_kind") == "session" and a.get("run"):
                for m in a["members"]:
                    self.finished[m] = "detached"
                    self.set_stage(m, state="running", detached=True, run=a["run"],
                                   reason=f"pod session run {a['run']} keeps running on its own (stop it by deleting "
                                          "its pod in the Runpod panel)")
                self.event("note", sid, msg=f"{a.get('session')}: the pod session run {a['run']} keeps running "
                                            "(a stop never kills it: its pod needs the download and the delete)")
                continue  # the `remote run` process itself still counts as the one pod session (locks)
            for m in a["members"]:
                self.finished[m] = "stopped"
                self.set_stage(m, state="stopped", ended=time.time())

    def finish(self) -> int:
        fails = [s for s, v in self.finished.items() if v == "failed"]
        opened = sorted(s for s, v in self.finished.items() if v == "gate")
        human = [s for s in self.plan.get("halts_at") or [] if s in self.blocked] or \
            [s for s, v in self.finished.items() if v == "blocked"]
        human = sorted(set(human) | set(opened))
        if self.stopping:
            self.state["status"], rc = "stopped", 130
        elif fails:
            self.state["status"], rc = "failed", 1
        elif human or any(v in ("blocked", "skipped") for v in self.finished.values()):
            self.state["status"], rc = "waiting_human", 3
            gate = sorted(set(human) | {s for s, v in self.finished.items() if v == "blocked"})
            self.state["gate"] = [{"stage": g, "reason": self.state["stages"].get(g, {}).get("reason"),
                                   "review": g in opened} for g in gate]
        else:
            self.state["status"], rc = "done", 0
        self.state["ended"] = time.time()
        self.save()
        n_ok = sum(v in ("ok", "gate") for v in self.finished.values())
        pauses = [sid for sid in self.todo if self.steps.get(sid, {}).get("action") == "pause"]
        gated = [sid for sid in self.todo if self.vby[sid].get("gate_human")]
        self.event("finish", None, rc=rc, msg=f"{self.state['status']}: {n_ok}/{len(self.todo)} stage(s) done"
                                              + (f", failed {fails}" if fails else "")
                                              + (f", waiting for your review of {opened}" if opened else "")
                                              + (f" (what-if: a real run pauses for you at "
                                                 f"{', '.join(dict.fromkeys(pauses + gated))})" if pauses else ""))
        return rc


class Busy(RuntimeError):
    """A cross-process limit is taken right now: the stage stays queued."""


def load_plan_for_run(take: str, plan_file: str, run_id: str | None, now: float | None = None) -> dict:
    """A plan file from work/pipeline/plans/, checked before anything runs.

    - Its approval (pod sessions) counts only when the monitor bound it to THIS run (approved.run_id ==
      DANCECAP_RUN_ID), it is younger than the plan validity and it was not used before; it is then consumed (one
      shot: a replay of the same file has no approval). Any other approval is stripped: pod steps stay blocked.
    - A plan without a bound approval must still be current: not older than plan.VALID_S and identical (hash) to a
      plan recomputed now from its request. Otherwise PlanError: make a new plan."""
    now = now or time.time()
    p = Path(plan_file)
    if not p.is_absolute():
        p = paths.root() / p
    if p.resolve().parent != paths.plans_dir().resolve():
        raise planmod.PlanError("plans are read from work/pipeline/plans/ only")
    plan = read_json(p, None)
    if not plan or plan.get("take") != take:
        raise planmod.PlanError("plan not found or for another take")
    appr = plan.get("approved") if isinstance(plan.get("approved"), dict) else None
    bound = bool(appr and run_id and appr.get("run_id") == run_id and not appr.get("consumed_by")
                 and now - float(appr.get("at") or 0) <= planmod.VALID_S)
    if bound:
        disk = read_json(p, {}) or {}
        if isinstance(disk.get("approved"), dict):
            disk["approved"].update(consumed_by=run_id, consumed_at=now)
            write_json_atomic(p, disk)
        return plan
    plan.pop("approved", None)
    if appr:
        print("[pipeline] the plan's approval is not bound to this run (or was used, or is too old): pod sessions "
              "stay blocked", flush=True)
    if now - float(plan.get("created") or 0) > planmod.VALID_S:
        raise planmod.PlanError(f"the plan is older than {planmod.VALID_S // 60} min: make a new one")
    fresh = planmod.make_plan(take, plan.get("request") or {})
    if fresh["plan_hash"] != plan.get("plan_hash"):
        raise planmod.PlanError("the plan changed since it was made (stage outputs or the take TOML): make a new one")
    return plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline run")
    ap.add_argument("take")
    ap.add_argument("--plan", help="plan JSON written by the monitor (work/pipeline/plans/<hash>.json)")
    ap.add_argument("--target")
    ap.add_argument("--scope", choices=planmod.SCOPES)
    ap.add_argument("--mode", choices=planmod.MODES, default="resume")
    ap.add_argument("--capture")
    ap.add_argument("--dry-run", action="store_true", help="log what would run; start no process")
    ap.add_argument("--what-if", action="store_true", help="dry run of the whole plan with your reviews assumed done "
                                                           "(implies --dry-run)")
    a = ap.parse_args(argv)
    if a.plan:
        try:
            plan = load_plan_for_run(a.take, a.plan, os.environ.get("DANCECAP_RUN_ID"))
        except planmod.PlanError as e:
            print(f"[pipeline] {e}", file=sys.stderr)
            return 2
    else:
        req = {"target": a.target, "scope": a.scope or ("upto" if a.target else "all"), "mode": a.mode,
               "options": {"capture": a.capture, "assume_reviews": bool(a.what_if)}}
        if a.what_if:
            a.dry_run = True
        plan = planmod.make_plan(a.take, req)
        plan.pop("approved", None)  # a CLI plan never carries an approval: pod sessions stay blocked
    dry = a.dry_run or bool(plan.get("dry_run"))
    if plan.get("what_if") and not dry:
        print("[pipeline] a what-if plan (your reviews assumed done) can only be dry-run", file=sys.stderr)
        return 2
    for st in plan["steps"]:
        print(f"[pipeline] plan {st['stage']:16s} {st['action']:12s} {st.get('reason') or ''}", flush=True)
    return Orchestrator(a.take, plan, dry_run=dry).run()


if __name__ == "__main__":
    sys.exit(main())
