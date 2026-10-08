"""`python -m dancecap.farm ...`: one front for the farm, plus the plug-in mechanism for the other stages.

Core commands (this module):
    status [--refresh] [--balance]      one screen / one JSON line: world, coverage, chunks, pods, spend, watcher, events
    coverage [--png [PATH]] [--ascii]   recompute coverage.json from the real outputs, optionally the defrag-map PNG
    plan [--dry-run] [--ignore-s1]      coverage -> gaps -> 30-s chunks -> POD job specs (skips what already has a job): chunk jobs
                                        (front/fit/refine on GPU pods) + ONE `stitch_analyse` job over all chunks (CPU pod)
    queue [--local-only] [--hold]       hand the planned pod jobs to the volume uploader (a hook); the worker enforces `after`
    analyse [--analyses a,b] [--take T] registered analyses on one take as a CPU-pod job (`--list`): nothing analyses locally
                                        (FARM.md 0.10); coverage is read from the small evidence files the pods wrote
    chunks [list|register-s1|retry ID|cancel ID]
    world [publish|build|verify|seal|show|list|compare TAKE|pin]
    events [-n N]   plugins   chunk-take|chunk-prep|chunk-fit ID   (pod-side helpers the job steps call)

PLUG-INS. Every other module of this package (volume, pods, worker, watch, ...) may define

    def register(sub):                       # sub = the argparse sub-parsers action of this CLI
        p = sub.add_parser("launch", help="...")
        p.add_argument(...)
        p.set_defaults(func=cmd_launch)      # cmd_launch(args) -> int; args.dance and args.json always exist
        from .cli import register_hook
        register_hook("queue_upload", upload_fn)   # optional: hooks other commands call

and is picked up automatically (no edit of this file): a plug-in that fails to import or register is skipped and listed by
`plugins` with its error. Hooks used by the core: `queue_upload(dance, jobs, args) -> {chunk_id: {"ok": bool, ...}}`
(called by `queue`), `coverage_updated(dance, coverage)` (after `coverage` / `plan` recompute it). Plug-ins must stay light
at import time (the CLI imports them on every call that is not a core command, and for --help).
"""
from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import sys
from pathlib import Path

from . import CORE_MODULES, DEFAULT_DANCE, coverage_path, farm_dir, review_dir
from .state import State, now_iso

HOOKS: dict[str, list] = {}
PLUGINS: dict[str, dict] = {}
CORE_COMMANDS = ("status", "coverage", "plan", "queue", "chunks", "world", "events", "plugins", "analyse", "chunk-take",
                 "chunk-prep", "chunk-fit", "chunk-prepare", "stitch-prep", "chunk-evidence")


def register_hook(name: str, fn) -> None:
    HOOKS.setdefault(name, []).append(fn)


def call_hook(name: str, *a, **k) -> list:
    out = []
    for fn in HOOKS.get(name, []):
        try:
            out.append(fn(*a, **k))
        except Exception as e:  # noqa: BLE001  a broken plug-in never breaks the core command
            out.append({"error": f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}: {type(e).__name__}: {e}"})
    return out


def load_plugins(sub) -> dict:
    """Import every non-core module of the package and let it register its sub-commands / hooks."""
    from . import __path__ as pkg_path
    for mi in sorted(pkgutil.iter_modules(pkg_path), key=lambda m: m.name):
        name = mi.name
        if name in CORE_MODULES or name.startswith("_") or name in PLUGINS:
            continue
        rec = {"module": name, "registered": False, "error": None}
        PLUGINS[name] = rec
        try:
            mod = importlib.import_module(f"{__package__}.{name}")
            reg = getattr(mod, "register", None)
            if callable(reg):
                reg(sub)
                rec["registered"] = True
        except BaseException as e:  # noqa: BLE001  incl. SystemExit from a bad add_argument
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
    return PLUGINS


# ------------------------------------------------------------------------------------------------------ helpers


def _out(a, obj, text: str | None = None) -> None:
    if getattr(a, "json", False) or text is None:
        print(json.dumps(obj, default=str, ensure_ascii=False))
    else:
        print(text)


def _world_header(dance: str) -> dict:
    from . import world as W
    cur = W.current(dance)
    if not cur:
        return {}
    return {"id": cur["world_id"], "pose_id": cur.get("pose_id"), "sha256": cur.get("world_sha256"),
            "sealed": cur.get("pose_id") in W.seals(dance), "avatar_id": cur.get("avatar_id")}


def _state_chunks(dance: str) -> list[dict]:
    return list(State(dance).load()["chunks"].values())


def refresh_coverage(dance: str, external=True, write: bool = True):
    from .timeline import compute_coverage
    chunks = [c for c in _state_chunks(dance) if "core_k0" in c and c.get("status") not in ("failed", "cancelled")]
    cov = compute_coverage(dance, chunks=chunks, external=external, world=_world_header(dance), write=write)
    if write:   # a dry run leaves state.json alone
        State(dance).set_world(**{k: v for k, v in _world_header(dance).items()})
        call_hook("coverage_updated", dance, cov)
    return cov


def _ledger_spend() -> dict:
    try:
        from .. import remote
        rows = remote.ledger()
        open_rows = [r for r in rows if not r.get("deleted_ts")]
        return {"cap": remote.cap(), "spent": round(remote.spent(), 2), "remaining": round(remote.remaining(), 2),
                "pods_open_in_ledger": len(open_rows), "rate_hr": round(sum(r["cost_per_hr"] for r in open_rows), 3)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}


# ----------------------------------------------------------------------------------------------------- status


def cmd_status(a) -> int:
    from .timeline import Coverage, DanceSpec
    spec = DanceSpec.load(a.dance)
    cov = refresh_coverage(a.dance) if a.refresh else Coverage.load(a.dance, with_frames=False)
    st = State(a.dance).load()
    spend = _ledger_spend()
    if a.balance:
        try:
            from .. import remote
            spend["account"] = remote.status().get("account")
        except Exception as e:  # noqa: BLE001
            spend["account_error"] = str(e)[:200]
    chunks = st["chunks"]
    by: dict[str, int] = {}
    for c in chunks.values():
        by[c.get("status", "?")] = by.get(c.get("status", "?"), 0) + 1
    obj = {"dance": a.dance, "time": now_iso(), "world": _world_header(a.dance),
           "grid": {**spec.grid.to_dict(), "T": spec.grid.T, "seconds": round(spec.grid.seconds, 1), "cells": spec.grid.n_cells},
           "coverage": None if cov is None else {"counts": cov.counts(), "percent": cov.percent(), "seconds": cov.seconds(),
                                                 "stages": cov.stages(), "updated": cov.updated},
           "chunks": {"by_status": by, "list": [{"id": c["id"], "status": c.get("status"), "kind": c.get("kind"),
                                                 "core_s": c.get("core_s"), "pod": c.get("pod"),
                                                 "progress": c.get("progress")} for c in chunks.values()]},
           "pods": [p for p in st["pods"].values() if p.get("state") not in ("deleted", "gone")],
           "spend": {**spend, **{k: v for k, v in (st.get("spend") or {}).items() if k not in spend}},
           "watcher": st.get("watcher", {}), "last_events": st["events"][-5:], "state_updated": st.get("updated")}
    if a.json:
        print(json.dumps(obj, default=str, ensure_ascii=False))
        return 0
    w = obj["world"]
    print(f"{a.dance}: {spec.grid.seconds:.1f} s on a 30 Hz lattice (k {spec.grid.k0}..{spec.grid.k1}, {spec.grid.n_cells} cells)")
    print(f"world   {w.get('id', '-')}  pose {w.get('pose_id', '-')}  {'SEALED' if w.get('sealed') else 'not sealed'}")
    if cov:
        c, p, s = cov.counts(), cov.percent(), cov.seconds()
        print(f"coverage  complete {c['complete']} cells {p['complete']}% ({s['complete']} s)   partial {c['partial']} {p['partial']}% "
              f"({s['partial']} s)   none {c['none']} {p['none']}% ({s['none']} s)   [{cov.updated}]")
    else:
        print("coverage  (not computed yet: python -m dancecap.farm coverage)")
    print("chunks  " + (", ".join(f"{v} {k}" for k, v in sorted(by.items())) or "none"))
    for c in obj["chunks"]["list"]:
        print(f"   {c['id']:<8} {c['status']:<10} {c['kind']:<6} {c['core_s']} s  pod {c['pod'] or '-'}")
    print(f"pods    {len(obj['pods'])} open" + "".join(f"\n   {p.get('id')} {p.get('state')} {p.get('gpu', '')}" for p in obj["pods"]))
    sp = obj["spend"]
    print(f"spend   ${sp.get('spent', '?')} of ${sp.get('cap', '?')} cap, ${sp.get('remaining', '?')} left, ${sp.get('rate_hr', 0)}/h open")
    return 0


# --------------------------------------------------------------------------------------------------- coverage


def cmd_coverage(a) -> int:
    from .timeline import ascii_map, render_png
    ext = False if a.no_external else (a.external or True)
    cov = refresh_coverage(a.dance, external=ext, write=not a.no_write)
    res = {"path": str(coverage_path(a.dance)), "counts": cov.counts(), "percent": cov.percent(), "seconds": cov.seconds(),
           "stages": cov.stages(), "gaps_missing": [list(g) for g in cov.gaps("missing")]}
    if a.png is not None:
        p = Path(a.png) if a.png else review_dir(a.dance) / "coverage_map.png"
        render_png(cov, p)
        res["png"] = str(p)
    if a.json:
        print(json.dumps(res))
    else:
        if a.ascii or a.png is None:
            print(ascii_map(cov))
        print(f"-> {res['path']}" + (f"   png {res['png']}" if "png" in res else ""))
    return 0


# ------------------------------------------------------------------------------------------------------- plan


def _table(chs, grid) -> str:
    """Chunks as a table; window / core in seconds on the reference clock."""
    rows = [f"{'id':<6} {'core (ref clock s)':>20} {'core s':>6} {'window s':>8} {'cams':>4} {'kind':<5} {'origin':<7} "
            f"{'still to do':<17} {'est min':>7}"]
    for c in chs:
        rows.append(f"{c.id:<6} {grid.t(c.core_k0):>9.1f} .. {grid.t(c.core_k1 + 1):<7.1f} {c.core_T / grid.hz:>6.1f} "
                    f"{c.T / grid.hz:>8.1f} {len(c.cams):>4} {c.kind:<5} {c.origin:<7} "
                    f"{','.join(x for x in ('front', 'fit', 'refined') if x not in c.stages_done):<17} "
                    f"{c.estimate.get('total_min', 0):>7.0f}")
    return "\n".join(rows)


def _final_chunks(st: State, extra: list) -> list[dict]:
    """The chunk dicts the dance's stitch job covers: everything in state that is not failed / cancelled, plus `extra` Chunks."""
    have = {cid: c for cid, c in st.load()["chunks"].items() if c.get("status") not in ("failed", "cancelled") and "core_k0" in c}
    for c in extra:
        have.setdefault(c.id, c.to_dict())
    return sorted(have.values(), key=lambda c: c["k0"])


def cmd_plan(a) -> int:
    """Coverage -> gaps -> 30-s chunks -> POD job specs (FARM.md 0.10): a chunk job per chunk (front / fit / refine on a GPU pod) and
    ONE final `stitch_analyse` job over the whole chunk list (stitch + analyses + coverage evidence on a CPU pod, no
    transfers). Nothing runs on this machine."""
    from . import analysis as A
    from . import chunks as C
    from . import world as W
    from .timeline import DanceSpec
    spec = DanceSpec.load(a.dance)
    try:
        w = W.load(a.dance)
    except W.WorldError as e:
        if not a.no_world:
            print(f"plan: {e}", file=sys.stderr)
            return 2
        w = None
    st = State(a.dance)
    s1 = [] if a.ignore_s1 else C.register_s1(spec, w.id if w else None)
    if s1 and not a.dry_run:
        cur = st.load()["chunks"]
        st.put_chunks([c.to_dict() for c in s1 if c.id not in cur], replace_planned=False)
    cov = refresh_coverage(a.dance, external=not a.no_external, write=not a.dry_run)
    stc = st.load()["chunks"]
    claimed = [(int(c["core_k0"]), int(c["core_k1"]), cid) for cid, c in stc.items()
               if c.get("status") in ("queued", "running", "done", "pulled", "stitched") and "core_k0" in c]
    claimed += [(c.core_k0, c.core_k1, c.id) for c in s1 if stc.get(c.id, {}).get("status", "planned") not in ("failed", "cancelled")]
    planned, rep = C.plan_gaps(cov, spec, claimed=claimed, mode=a.mode, overlap_s=a.overlap_s, world_id=w.id if w else None,
                               kind=a.kind)
    jobs = []
    plan_doc = {"created": now_iso(), "dance": a.dance, "world": _world_header(a.dance), "mode": a.mode, "ignore_s1": a.ignore_s1,
                "report": rep, "coverage": {"counts": cov.counts(), "percent": cov.percent(), "seconds": cov.seconds()},
                "chunks": [c.to_dict() for c in planned], "registered_s1": [c.to_dict() for c in s1]}
    analyses = [x for x in (a.analyses or "").split(",") if x]
    final = None
    code = C.code_info()
    todo = planned + [c for c in s1 if stc.get(c.id, {}).get("status", "planned") == "planned"]
    if not a.dry_run:
        for c in todo:
            job = C.build_job(c, spec, w, code=code)
            if not job["steps"]:   # nothing left to run: its results are already here
                st.set_chunk(c.id, status="pulled")
                continue
            C.write_job(job, a.dance)
            jobs.append({"id": c.id, "kind": job["kind"], "stages": job["stages"], "inputs": len(job["inputs"]),
                         "needs_gpu": job["needs_gpu"], "max_minutes": job["max_minutes"]})
        st.put_chunks([c.to_dict() for c in planned + s1], replace_planned=True)   # the S1 chunks stay (they are planned too)
        for j in jobs:
            st.set_job(j["id"], chunk=j["id"], kind=j["kind"], status="planned", written=now_iso(),
                       spec=str(farm_dir(a.dance) / "jobs" / f"{j['id']}.json"))
    if not a.no_final:
        cl = _final_chunks(st, planned + s1) if not a.dry_run else [c.to_dict() for c in planned + s1]
        final = A.build_final_job(spec, cl, w, analyses=analyses or A.DEFAULT_ANALYSES, hash_inputs=not a.dry_run, code=code)
        entry = {"id": final["id"], "kind": final["what"], "stages": final["stages"], "after": final["after"],
                 "local_chunks": final["local_chunks"], "analyses": final["analyses"], "inputs": len(final["inputs"]),
                 "needs_gpu": final["needs_gpu"], "max_minutes": final["max_minutes"]}
        if not a.dry_run:
            A.write_job(final)
            st.set_job(final["id"], kind=final["what"], status="planned", after=final["after"], written=now_iso(),
                       spec=str(farm_dir(a.dance) / "jobs" / f"{final['id']}.json"))
        jobs.append(entry)
    plan_doc["jobs"] = jobs
    if not a.dry_run:
        from .state import atomic_write_json
        atomic_write_json(farm_dir(a.dance) / "plan.json", plan_doc)
    if a.out:
        from .state import atomic_write_json
        atomic_write_json(Path(a.out), plan_doc)
    out = {"report": rep, "chunks": [c.to_dict() for c in planned], "s1": [c.to_dict() for c in s1], "jobs": jobs}
    if a.json:
        print(json.dumps(out, default=str))
        return 0
    print(f"gaps ({a.mode}): " + (", ".join(f"k {g[0]}..{g[1]} ({s} s)" for g, s in zip(rep["gaps"], rep["gap_s"])) or "none"))
    if rep["claimed_skipped"]:
        print("already claimed: " + ", ".join(f"{c[2]} (k {c[0]}..{c[1]})" for c in rep["claimed_skipped"]))
    if planned:
        print(_table(planned, spec.grid))
    else:
        print("no new chunks needed")
    if s1:
        print("registered S1 segments (front done):")
        print(_table(s1, spec.grid))
    for j in jobs:
        print(f"job {j['id']:<7} {j['kind']:<15} {'GPU' if j['needs_gpu'] else 'CPU'} pod  max {j['max_minutes']:.0f} min"
              + (f"  after {','.join(j['after'])}" if j.get("after") else "") + (f"  analyses {','.join(j['analyses'])}" if j.get("analyses") else ""))
    if jobs and not a.dry_run:
        print(f"{len(jobs)} job spec(s) -> {farm_dir(a.dance) / 'jobs'}   next: python -m dancecap.farm queue")
    return 0


def queue_jobs(dance: str, *, local_only: bool = False, hold: bool = False, args=None) -> dict:
    """Hand every planned job spec (chunk jobs and the dance's analysis / stitch jobs) to the volume uploader. The worker only
    claims a job once every id in its `after` has a done.json, so everything can be queued at once; hold=True keeps a job
    with unmet `after` here until its chunks are done (the watcher then calls release_ready every cycle). Returns
    {queued: [...], held: [...], missing: [...], uploader: bool}."""
    from . import analysis as A
    st = State(dance)
    cur = st.load()
    ids = [cid for cid, c in cur["chunks"].items() if c.get("status") in (None, "planned")]
    ids += [jid for jid, j in cur["jobs"].items() if j.get("status") == "planned" and jid not in cur["chunks"]
            and not (j.get("chunk") and j["chunk"] in cur["chunks"])]
    specs, missing = [], []
    for jid in dict.fromkeys(ids):
        p = farm_dir(dance) / "jobs" / f"{jid}.json"
        (specs if p.exists() else missing).append(json.loads(p.read_text(encoding="utf-8")) if p.exists() else jid)
    status = {cid: c.get("status") for cid, c in cur["chunks"].items()}
    ready, held = A.ready_jobs(specs, status) if hold else (specs, [])
    res = {"queued": [], "held": [{"id": j["id"], "after": [n for n in A.deps(j) if status.get(n) not in A.DONE_STATES]}
                      for j in held], "missing": missing, "uploader": bool(HOOKS.get("queue_upload")), "results": {}}
    if local_only or not ready or not HOOKS.get("queue_upload"):
        res["ready"] = [j["id"] for j in ready]
        return res
    for r in call_hook("queue_upload", dance, ready, args):
        if isinstance(r, dict):
            if "error" in r:
                res.setdefault("errors", []).append(r["error"])
            else:
                res["results"].update(r)
    for jid, r in res["results"].items():
        if isinstance(r, dict) and r.get("ok"):
            res["queued"].append(jid)
            if jid in cur["chunks"]:
                st.set_chunk(jid, status="queued", queued=now_iso())
            st.set_job(jid, status="queued", queued=now_iso())
    return res


def release_ready(dance: str, args=None) -> dict:
    """Watcher hook for a held queue: queue the jobs whose `after` chunks have just finished."""
    return queue_jobs(dance, hold=True, args=args)


def cmd_queue(a) -> int:
    res = queue_jobs(a.dance, local_only=a.local_only, hold=a.hold, args=a)
    if a.json:
        print(json.dumps(res, default=str))
        return 0
    if res.get("missing"):
        print("queue: no spec for " + ", ".join(res["missing"]) + " (run plan first)", file=sys.stderr)
    for e in res.get("errors", []):
        print("queue: uploader error:", e, file=sys.stderr)
    if "ready" in res:
        print(f"queue: {len(res['ready'])} spec(s) ready in {farm_dir(a.dance) / 'jobs'}"
              + ("" if a.local_only else "" if res["uploader"] else "  (no volume uploader installed: farm.volume registers the 'queue_upload' hook)"))
    else:
        print(f"queue: {len(res['queued'])} job(s) queued: {', '.join(res['queued']) or '-'}")
    for h in res["held"]:
        print(f"queue: {h['id']} held until {', '.join(h['after'])} finish(es)")
    return 0


def cmd_analyse(a) -> int:
    """`farm analyse`: registered analyses on one take as a POD job (never run here). --list shows what is registered."""
    from . import analysis as A
    from . import chunks as C
    from . import world as W
    from .timeline import DanceSpec
    if a.list:
        for n, x in sorted(A.ANALYSES.items()):
            print(f"{n:<16} {'GPU' if x.gpu else 'CPU'} {x.scope:<6} {'optional' if x.optional else 'required':<8} {x.desc}")
        return 0
    spec = DanceSpec.load(a.dance)
    try:
        w = W.load(a.dance)
    except W.WorldError:
        w = None
    names = [x for x in a.analyses.split(",") if x]
    take = a.take or a.dance
    job = A.build_analysis_job(spec, take, names, w, hash_inputs=not a.dry_run, code=C.code_info())
    if not a.dry_run:
        A.write_job(job)
        State(a.dance).set_job(job["id"], kind="analyse", status="planned", after=job["after"], written=now_iso(),
                               spec=str(farm_dir(a.dance) / "jobs" / f"{job['id']}.json"))
    _out(a, {k: job[k] for k in ("id", "kind", "what", "take", "analyses", "after", "needs_gpu", "max_minutes")},
         f"job {job['id']}: {', '.join(job['analyses'])} on {take}, {'GPU' if job['needs_gpu'] else 'CPU'} pod"
         + ("" if a.dry_run else "   next: python -m dancecap.farm queue"))
    return 0


def cmd_chunks(a) -> int:
    from . import chunks as C
    from .timeline import DanceSpec
    st = State(a.dance)
    if a.action == "register-s1":
        spec = DanceSpec.load(a.dance)
        cs = C.register_s1(spec, _world_header(a.dance).get("id"))
        cur = st.load()["chunks"]
        st.put_chunks([c.to_dict() for c in cs if c.id not in cur], replace_planned=False)
        print(_table(cs, spec.grid))
        return 0
    if a.action in ("retry", "cancel"):
        if not a.id:
            print("chunks retry|cancel needs a chunk id", file=sys.stderr)
            return 2
        st.set_chunk(a.id, status="planned" if a.action == "retry" else "cancelled")
        print(f"{a.id}: {a.action}")
        return 0
    chs = st.load()["chunks"]
    if a.json:
        print(json.dumps(chs, default=str))
    else:
        for c in chs.values():
            print(f"{c['id']:<8} {c.get('status', '?'):<10} k {c.get('core_k0')}..{c.get('core_k1')} ({c.get('core_s')} s) "
                  f"{c.get('kind', '')} {c.get('origin', '')} stages_done={c.get('stages_done')}")
    return 0


# ---------------------------------------------------------------------------------------------------- world


def cmd_world(a) -> int:
    from . import world as W
    act = a.action
    if act == "list":
        _out(a, W.list_worlds(a.dance), "\n".join(f"{w['world_id']} pose {w['pose_id']} avatar {w['avatar_id']} "
                                                  f"{'SEALED ' if w['sealed'] else ''}{'data ok' if w['data'] else 'NO DATA'} {w['created']}"
                                                  for w in W.list_worlds(a.dance)) or "no worlds")
        return 0
    if act == "show":
        w = W.load(a.dance, a.id)
        m = w.manifest
        print(f"world {w.id}  pose {w.pose_id}  avatar {w.avatar_id}  {'SEALED' if w.sealed else 'not sealed'}  "
              f"{m['n_files']} files {m['bytes'] / 1e6:.0f} MB  created {m['created']}") if not a.json else print(json.dumps(m))
        if not a.json:
            for k, v in m["members"].items():
                print(f"  {k:<16} {v['scope']:<6} {v['sha256'][:12]}  {v['n_files']:>4} files {v['bytes'] / 1e6:>8.2f} MB  {v.get('root', v.get('path', ''))}")
            print(f"  data: {w.root}   volume: {w.volume_path()}")
        return 0
    if act in ("build", "publish"):
        if act == "build":
            spec, members, hm, files, wid = W.plan(a.dance)
            _out(a, {**wid, "files": len(files), "members": {k: v["sha256"][:12] for k, v in hm.items()}},
                 f"would publish world {wid['world_id']} (pose {wid['pose_id']}, avatar {wid['avatar_id']}), {len(files)} files")
            return 0
        r = W.build(a.dance, link=a.link, make_current=not a.no_current, allow_new_pose=a.new_pose_world)
        _out(a, r.to_dict(), f"world {r.world_id} (pose {r.pose_id}) {'built' if r.created else 'unchanged'} in {r.seconds} s, "
                             f"verify {'ok' if r.verify and r.verify['ok'] else r.verify}  -> {r.root}")
        State(a.dance).set_world(**_world_header(a.dance))
        State(a.dance).event("world", f"world {r.world_id} {'built' if r.created else 'confirmed'}", world_id=r.world_id)
        return 0
    if act == "verify":
        w = W.load(a.dance, a.id, Path(a.root) if a.root else None)
        rep = w.verify("quick" if a.quick else "full")
        _out(a, rep.to_dict(), f"world {w.id} {'verifies' if rep.ok else 'FAILS'} ({rep.level}, {rep.files} files, {rep.seconds} s)"
                               + "".join(f"\n  {e}" for e in rep.errors[:10]))
        return 0 if rep.ok else 1
    if act == "check":
        r = W.check_current(a.dance)
        _out(a, r, f"world {r['current']}: {r['advice']}" + (f"  (changed: {', '.join(r['changed_members'])})" if r["changed_members"] else ""))
        return 0 if not r["changed_pose_members"] else 1
    if act == "seal":
        r = W.seal(a.dance, a.id, a.note or "")
        _out(a, r, f"sealed: {r}")
        State(a.dance).set_world(**_world_header(a.dance))
        return 0
    if act == "compare":
        bad = 0
        for t in a.takes:
            r = W.compare_take(t, a.dance, a.id, deep=not a.no_deep)
            bad += 0 if r["ok"] or not r["member"] else 1
            if a.json:
                print(json.dumps(r))
                continue
            print(f"{t}: {'consistent' if r['ok'] else 'INCONSISTENT'} with world {r['world_id']}"
                  f"{'' if r['member'] else ' (declared not a member)'}{'' if r['avatar_ok'] else '  [avatar config differs]'}")
            for c in r["checks"]:
                print(f"   {'ok ' if c['ok'] else ('?  ' if c['ok'] is None else 'BAD')} {c['name']}: {c['detail'][:200]}")
        return 1 if bad else 0
    if act == "pin":
        r = W.pin_tomls(a.dance, a.takes or None, a.id)
        _out(a, r, "\n".join(str(x) for x in r))
        return 0
    return 2


# ---------------------------------------------------------------------------------------------------- the rest


def cmd_events(a) -> int:
    ev = State(a.dance).load()["events"][-a.n:]
    for e in ev:
        print(f"{e['ts']}  {e['kind']:<8} {e['msg']}") if not a.json else print(json.dumps(e))
    return 0


def cmd_plugins(a) -> int:
    _out(a, PLUGINS, "\n".join(f"{p['module']:<10} {'registered' if p['registered'] else 'no register()'}"
                               + (f"  ERROR {p['error']}" if p["error"] else "") for p in PLUGINS.values()) or "no plug-ins")
    return 0


def cmd_pod_helper(a) -> int:
    from . import chunks as C
    from . import analysis as A
    if a.cmd == "chunk-take":
        r = C.chunk_take(a.id, a.dance, a.job)
    elif a.cmd == "chunk-prep":
        r = C.chunk_prep(a.id, a.dance, a.world, a.job)
    elif a.cmd == "chunk-prepare":
        r = A.chunk_prepare(a.id, a.dance, a.job)
    elif a.cmd == "stitch-prep":
        r = A.stitch_prep(a.id, a.dance, a.job)
    elif a.cmd == "chunk-evidence":
        r = A.chunk_evidence(a.id, a.stage, a.dance, a.job)
    else:
        r = C.chunk_fit(a.id, a.dance, a.world, a.job)
    print(json.dumps(r, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m dancecap.farm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dance", default=DEFAULT_DANCE)
    ap.add_argument("--json", action="store_true", help="one JSON document / line instead of text")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("status", help="world, coverage, chunks, pods, spend, watcher, events")
    p.add_argument("--refresh", action="store_true", help="recompute coverage first")
    p.add_argument("--balance", action="store_true", help="also read the Runpod account balance (API call)")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("coverage", help="coverage.json from the real outputs (+ defrag-map PNG)")
    p.add_argument("--png", nargs="?", const="", default=None, metavar="PATH", help="render the map (default farm/review/coverage_map.png)")
    p.add_argument("--ascii", action="store_true")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--no-external", action="store_true", help="ignore work/<dance>/coverage_cells.json")
    p.add_argument("--external", metavar="PATH", help="another workflow's {cells: [...]} file to merge (raises cells only)")
    p.set_defaults(func=cmd_coverage)
    p = sub.add_parser("plan", help="coverage -> gaps -> 30-s chunks -> job specs")
    p.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    p.add_argument("--mode", choices=("missing", "all"), default="missing")
    p.add_argument("--overlap-s", type=float, default=4.0)
    p.add_argument("--kind", choices=("full", "front", "fit", "refine", "analyse"), default="full")
    p.add_argument("--ignore-s1", action="store_true", help="plan from scratch: do not let the S1 segments a-d claim their frames")
    p.add_argument("--no-external", action="store_true")
    p.add_argument("--no-world", action="store_true", help="plan without a published world (specs lack world ids)")
    p.add_argument("--out", metavar="PATH", help="also write the plan document here (with --dry-run: no state or specs change)")
    p.add_argument("--no-final", action="store_true", help="do not emit the dance's stitch+analyse pod job")
    p.add_argument("--analyses", default="", help="analyses of the final job (default timing,physics,counterbalance,floorcraft)")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("queue", help="hand the planned POD jobs to the volume uploader (the worker runs a job only after its `after` chunks)")
    p.add_argument("--local-only", action="store_true", help="only report what is ready")
    p.add_argument("--hold", action="store_true", help="keep jobs whose `after` chunks are not done here until they are (release_ready)")
    p.set_defaults(func=cmd_queue)
    p = sub.add_parser("chunks", help="list / register-s1 / retry ID / cancel ID")
    p.add_argument("action", nargs="?", default="list", choices=("list", "register-s1", "retry", "cancel"))
    p.add_argument("id", nargs="?")
    p.set_defaults(func=cmd_chunks)
    p = sub.add_parser("world", help="the world package: publish | build (dry) | verify | seal | show | list | compare | pin")
    p.add_argument("action", choices=("publish", "build", "check", "verify", "seal", "show", "list", "compare", "pin"))
    p.add_argument("takes", nargs="*", help="compare / pin: take names")
    p.add_argument("--id", help="a world id (default CURRENT)")
    p.add_argument("--root", help="verify: the data directory to check (e.g. a volume copy)")
    p.add_argument("--quick", action="store_true", help="verify: file list + sizes only (default: re-hash every file)")
    p.add_argument("--link", action="store_true", help="publish: hard-link instead of copying the big files")
    p.add_argument("--no-current", action="store_true")
    p.add_argument("--new-pose-world", action="store_true", help="publish a changed room model although CURRENT is sealed")
    p.add_argument("--note")
    p.add_argument("--no-deep", action="store_true", help="compare: skip the COLMAP model comparison")
    p.set_defaults(func=cmd_world)
    p = sub.add_parser("events", help="the last events of state.json")
    p.add_argument("-n", type=int, default=20)
    p.set_defaults(func=cmd_events)
    p = sub.add_parser("plugins", help="which plug-in modules registered commands")
    p.set_defaults(func=cmd_plugins)
    for name in ("chunk-take", "chunk-prep", "chunk-fit", "chunk-prepare", "stitch-prep", "chunk-evidence"):
        p = sub.add_parser(name, help="pod-side step helper (see chunks.STEP_TEMPLATES / analysis.STEP_TEMPLATES)")
        p.add_argument("id")
        p.add_argument("--job", help="job spec path (default $FARM_JOB, farm/jobs/<id>.json, volume queue/<id>.json)")
        p.add_argument("--world", help="the verified world directory (default $HM_WORLD_DIR)")
        p.add_argument("--stage", default="fit", help="chunk-evidence: the last stage the job ran (front|fit|refined|analysed|final)")
        p.set_defaults(func=cmd_pod_helper)
    p = sub.add_parser("analyse", help="registered analyses on one take as a POD job (CPU pod; never runs here)")
    p.add_argument("--analyses", default=",".join(("timing", "physics", "counterbalance", "floorcraft")),
                   help="comma list (see --list); default timing,physics,counterbalance,floorcraft")
    p.add_argument("--take", help="the take to analyse (default: the dance)")
    p.add_argument("--list", action="store_true", help="list the registered analyses")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_analyse)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)   # --dance / --json are accepted anywhere on the line
    pre.add_argument("--dance")
    pre.add_argument("--json", action="store_true")
    ns, rest = pre.parse_known_args(argv)
    ap = build_parser()
    first = next((x for x in rest if not x.startswith("-")), None)
    sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    # plug-ins are imported unless the command is a core one (keeps `status` fast and independent of their imports)
    if first not in CORE_COMMANDS or "-h" in rest or "--help" in rest:
        load_plugins(sub)
    a = ap.parse_args(rest if first else ["status", *rest])
    a.dance = ns.dance or DEFAULT_DANCE
    a.json = bool(ns.json)
    try:
        return int(a.func(a) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
