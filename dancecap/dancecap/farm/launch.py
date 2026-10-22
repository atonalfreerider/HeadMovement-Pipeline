"""farm launch | up | pull | down | the volume uploader behind `farm queue` (FARM.md section 6).

    python -m dancecap.farm launch [--pods N] [--gpus 4090,l4] [--dry-run]    start GPU pods from the volume for the queued jobs
    python -m dancecap.farm up [--only a,b] [--dry-run]                      plan -> sync code + world -> queue -> launch -> watch
    python -m dancecap.farm pull [IDS] [--all]                               verified pull of finished jobs through any pod
    python -m dancecap.farm down [--abandon]                                 verified pull, then delete this farm's pods (keeps the volume)

Money rules (user, 2026-10-08): every pod creation passes pods.gate (`max_cost <= balance - 2` and `<= remaining cap`, counting the
worst case of every pod already open in the ledger), the kill switch is generous (default 100 min) and extended by the watcher only
while the job is healthy and under the hard ceiling (default 240 min); `--dry-run` prints prices, worst case and the plan and
creates nothing. `down` deletes ONLY the pods recorded in this dance's state.json (never by name prefix: other agents' pods carry the
same prefix), and only after every job they finished is pulled and verified.
"""
from __future__ import annotations

import contextlib
import json
import shlex
import sys
import time
from pathlib import Path

from . import pull as P
from . import watch as Wt
from .state import State

_BACKEND = None   # tests / embedding callers: replaces watch.RunpodBackend
GPU_SLOWDOWN = 0.45   # RTX 4090 fit time relative to the A100 model of chunks.fit_seconds: measured 0.41 (966 s vs 2362 s for 900 frames, drill 2026-10-08), rounded up


def backend_for(dance: str):
    return _BACKEND or Wt.RunpodBackend(dance)


def _csv(x) -> list[str]:
    if not x:
        return []
    if isinstance(x, str):
        return [i.strip() for i in x.split(",") if i.strip()]
    return [str(i) for i in x]


def say(msg: str) -> None:
    """Progress lines go to stderr so `--json` output on stdout stays one parseable document."""
    print(f"[farm] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ a link to the volume


@contextlib.contextmanager
def volume_link(dance: str, backend=None, keep: bool = False):
    """An ssh Link to the volume: a pod of this farm that answers, else a running farm pod (borrowed, never deleted), else a small
    CPU gateway pod (budget-gated; recorded in state.json; deleted on exit unless keep=True)."""
    backend = backend or backend_for(dance)
    st = State(dance)
    link, made = None, None
    for p in st.load()["pods"].values():
        if p.get("state") in ("deleted", "gone", "lost", "stuck") or not (p.get("host") and p.get("port")):
            continue
        try:
            lk = backend.make_link(p)
            lk.sh("true", 30)
            link = lk
            break
        except Exception:  # noqa: BLE001
            continue
    if link is None:
        info, created = backend.gateway(minutes=45, vcpu=2, reuse=True, name=f"hm-farm-{Wt.dance_tag(dance)}-gateway")
        link = backend.make_link({"id": info["pod_id"], "name": info.get("name"), "gpu": info.get("gpu"),
                                  "price_hr": info.get("cost_per_hr"), "host": info["host"], "port": info["port"]})
        if created:
            Wt.record_pod(st, info, "gateway", by="cli")
            made = info["pod_id"]
            say(f"gateway pod {made} started (${float(info.get('cost_per_hr') or 0):.2f}/h) to reach the volume")
    try:
        yield link
    finally:
        if made and not keep:
            with contextlib.suppress(Exception):
                backend.delete(made, "cli finished with the gateway")
                Wt.mark_pod_deleted(st, made, "cli finished with the gateway", None)
                say(f"gateway pod {made} deleted")


def ensure_synced(link, dance: str, specs: list[dict]) -> tuple[dict, dict]:
    """Code bundle and world packages are on the volume before any job that names them is queued (hash-diffed: unchanged = no
    transfer). Returns (code info, {world_id: result or error})."""
    from . import volume
    code = volume.sync_code(link)
    say(f"code {code['key']} (git {str(code.get('git_sha'))[:8]}{' + uncommitted changes' if code.get('dirty') else ''}) on the volume")
    worlds: dict = {}
    last: dict = {}

    def progress(rel, done, size):   # the first world upload moves hundreds of MB over the home uplink: say so now and then
        if time.time() - last.get("t", 0) > 15 and size >= (4 << 20):
            last["t"] = time.time()
            say(f"  world upload {rel.rsplit('/', 1)[-1]}: {done / 1e6:.0f}/{size / 1e6:.0f} MB")
    for wid in sorted({s.get("world_id") for s in specs if s.get("world_id")}):
        t0 = time.time()
        try:
            worlds[wid] = volume.sync_world(link, dance, wid, progress=progress)
            say(f"world {wid} on the volume ({'already there' if worlds[wid].get('skipped') is True else f'uploaded in {time.time() - t0:.0f} s'})")
        except Exception as e:  # noqa: BLE001
            worlds[wid] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            say(f"world {wid}: {worlds[wid]['error']}")
    return code, worlds


def prepare_spec(spec: dict, code: dict | None) -> dict:
    """A copy of the planned spec ready for volume.submit_job: input `local` paths absolute (plan writes them workspace-relative),
    the code bundle pinned (every pod runs the code that was uploaded with the job)."""
    spec = json.loads(json.dumps(spec))
    ws = Wt.workspace_root()
    for inp in spec.get("inputs", []):
        loc = inp.get("local")
        if loc and not Path(loc).is_absolute():
            inp["local"] = str(ws / loc)
    if code:
        spec["code"] = {**(spec.get("code") or {}), "bundle_sha256": code.get("bundle_sha256"), "key": code.get("key"),
                        "git_sha": code.get("git_sha"), "dirty": code.get("dirty")}
    return spec


def archive_stale(link, jid: str) -> str:
    """A finished (done.json / failed.json) chunks/<jid>/ on the volume belongs to an EARLIER plan of the same id; left in place the
    worker would skip the new job and the watcher would pull the old result. It is moved aside (chunks/<jid>.old.<ts>, never deleted).
    Returns 'clean' | 'archived' | 'claimed' (a worker holds the job: nothing is touched)."""
    q = shlex.quote
    out = link.sh(f"cd {q(link.abs('chunks'))} 2>/dev/null || {{ echo clean; exit 0; }}; "
                  f"if [ -e {q(jid)}/claim.lock ]; then echo claimed; "
                  f"elif [ -e {q(jid)}/done.json ] || [ -e {q(jid)}/failed.json ]; then mv {q(jid)} {q(jid)}.old.$(date +%s) && echo archived; "
                  f"else echo clean; fi", 60)
    return out.split()[-1] if out.split() else "clean"


def archive_replanned(link, spec: dict) -> str:
    """A half-finished chunks/<id>/ (step markers, a last_failure.json) from an EARLIER plan of the same id: if the plan changed (command,
    inputs, world or code) its step results must not be resumed. The worker ignores markers of another plan by their fingerprint; this also
    moves the directory aside (chunks/<id>.old.<ts>) so nothing of the old plan is mixed into the new result. 'same' | 'archived' | 'claimed' | 'clean'."""
    from . import worker
    jid = spec["id"]
    q = shlex.quote
    old = None
    with contextlib.suppress(Exception):
        old = link.read_json(f"queue/{jid}.json")
    if not isinstance(old, dict) or worker.job_fingerprint(old) == worker.job_fingerprint(spec):
        return "same"
    out = link.sh(f"cd {q(link.abs('chunks'))} 2>/dev/null || {{ echo clean; exit 0; }}; "
                  f"if [ -e {q(jid)}/claim.lock ]; then echo claimed; "
                  f"elif [ -d {q(jid)}/steps ] || [ -e {q(jid)}/last_failure.json ]; then mv {q(jid)} {q(jid)}.old.$(date +%s) && echo archived; "
                  f"else echo clean; fi", 60)
    return out.split()[-1] if out.split() else "clean"


def prefer_volume_results(link, spec: dict) -> list[str]:
    """The planner lists a chunk whose motion is on THIS machine as a motion.tar input to upload. When the farm itself produced that
    chunk (chunks/<id>/done.json + out/result.tar on the volume) the pod takes it from there instead: nothing crosses the home
    uplink, the job runs `after` the chunk. Returns the ids swapped."""
    swapped = []
    for c in spec.get("stitch_chunks") or []:
        cid = c.get("id")
        if cid not in (spec.get("local_chunks") or []):
            continue
        inp = next((i for i in spec.get("inputs", []) if i.get("path") == f"dancecap/work/{c['take']}/motion.tar"), None)
        if inp is None or not (link.exists(f"chunks/{cid}/done.json") and link.exists(f"chunks/{cid}/out/result.tar")):
            continue
        spec["inputs"].remove(inp)
        spec["inputs"].append({"path": f"chunks/{cid}/out/result.tar", "archive": True})
        spec["after"] = sorted(set(spec.get("after") or []) | {cid})
        spec["local_chunks"] = [x for x in spec["local_chunks"] if x != cid]
        swapped.append(cid)
    return swapped


def foreign_fits(jobs: list[dict]) -> dict[str, str]:
    """Chunks another session is fitting right now (fit_full names its pods `dancecap-fit-<seg>`): queueing the same chunk here would
    pay for the same fit twice, so the uploader leaves it alone unless args.force_foreign is set."""
    try:
        from .. import remote
        open_rows = {r.get("name"): r.get("pod_id") for r in remote.ledger() if not r.get("deleted_ts")}
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for j in jobs:
        for nm in (f"dancecap-fit-{j['id']}", f"fit-{j['id']}"):
            if nm in open_rows:
                out[j["id"]] = f"pod {open_rows[nm]} ({nm}) of another session is open for this chunk"
    return out


def queue_upload(dance: str, jobs: list[dict], args=None) -> dict:
    """The `queue_upload` hook of cli.queue_jobs: sync code + world, then volume.submit_job for each spec (inputs hash-diffed, resumable,
    queue/<id>.json written last). Honors args.only / args.exclude. Returns {job id: {ok, ...}}."""
    from . import volume
    only, exclude = set(_csv(getattr(args, "only", None))), set(_csv(getattr(args, "exclude", None)))
    jobs = [j for j in jobs if (not only or j["id"] in only) and j["id"] not in exclude]
    if not jobs:
        return {}
    out: dict = {}
    if not getattr(args, "force_foreign", False):
        for jid, why in foreign_fits(jobs).items():
            out[jid] = {"ok": False, "error": f"not queued: {why} (--force-foreign to override)"}
            say(f"{jid}: {out[jid]['error']}")
        jobs = [j for j in jobs if j["id"] not in out]
        if not jobs:
            return out
    last = {}

    def progress(rel, done, size):
        if time.time() - last.get(rel, 0) > 5 or done >= size:
            last[rel] = time.time()
            say(f"  upload {rel}: {done / 1e6:.0f}/{size / 1e6:.0f} MB")
    try:
        with volume_link(dance, keep=bool(getattr(args, "keep_gateway", False))) as link:
            code, worlds = ensure_synced(link, dance, jobs)
            for spec in jobs:
                wid = spec.get("world_id")
                if wid and "error" in worlds.get(wid, {}):
                    out[spec["id"]] = {"ok": False, "error": f"world {wid}: {worlds[wid]['error']}"}
                    continue
                try:
                    stale = archive_stale(link, spec["id"])
                    if stale == "claimed":
                        out[spec["id"]] = {"ok": False, "error": "a worker holds a claim on this job id on the volume; not touched"}
                        say(f"{spec['id']}: {out[spec['id']]['error']}")
                        continue
                    if stale == "archived":
                        say(f"{spec['id']}: a finished run of this id was on the volume; moved to chunks/{spec['id']}.old.<time>")
                    ready = prepare_spec(spec, code)
                    if archive_replanned(link, ready) == "archived":
                        say(f"{spec['id']}: unfinished work of an earlier plan with other steps / inputs was on the volume; moved to chunks/{spec['id']}.old.<time>")
                    swapped = prefer_volume_results(link, ready)
                    if swapped:
                        say(f"{spec['id']}: chunk result(s) {', '.join(swapped)} come from the volume (not uploaded from here)")
                    r = volume.submit_job(link, ready, progress=progress)
                    out[spec["id"]] = {"ok": True, **r}
                    say(f"queued {spec['id']}: {r['inputs']} input file(s) pushed")
                except Exception as e:  # noqa: BLE001
                    out[spec["id"]] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
                    say(f"queue {spec['id']} failed: {out[spec['id']]['error']}")
    except Exception as e:  # noqa: BLE001
        for j in jobs:
            out.setdefault(j["id"], {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return out


# ------------------------------------------------------------------------------------------------ what is waiting to run


def job_rows(dance: str) -> list[dict]:
    """Every job in state.json with its spec facts: [{id, status, needs_gpu, after, est_min, max_minutes, what}]."""
    st = State(dance).load()
    rows = []
    for jid, j in st["jobs"].items():
        spec = Wt.load_spec(dance, jid)
        rec = st["chunks"].get(jid) or j
        rows.append({"id": jid, "status": rec.get("status"), "kind": spec.get("kind") or j.get("kind"), "what": spec.get("what"),
                     "needs_gpu": Wt.needs_gpu(spec) if spec else None, "after": spec.get("after") or [],
                     "est_min": (spec.get("estimate") or {}).get("total_min"), "max_minutes": spec.get("max_minutes"),
                     "core_s": rec.get("core_s")})
    return rows


def estimate_run(est_min: list[float], n: int, price: float, hard_minutes: float, slowdown: float = 1.0) -> dict:
    """Wall time and cost of `est_min` minutes-per-job (A100 basis) on `n` pods of `price` $/h: longest-first onto the least loaded
    pod, + pod start (about 1.5 min). `slowdown` scales the GPU (RTX 4090, measured by the drill 2026-10-08: 0.41 for a 900-frame fit, see GPU_SLOWDOWN). The worst case
    is every pod running to its hard ceiling."""
    n = max(1, n)
    loads = [0.0] * n
    for m in sorted((x * slowdown for x in est_min), reverse=True):
        i = loads.index(min(loads))
        loads[i] += m
    wall = max(loads) + 1.5 if est_min else 0.0
    return {"pods": n, "jobs": len(est_min), "wall_min": round(wall, 0), "pod_minutes": round(sum(loads) + 1.5 * n, 0),
            "cost_usd": round(price * (sum(loads) + 1.5 * n) / 60, 2),
            "worst_usd": round(price * n * (hard_minutes + 8.0) / 60, 2), "slowdown": slowdown}


# ------------------------------------------------------------------------------------------------ launch


def launch_gpu_pods(dance: str, n: int, pol: Wt.Policy, *, backend=None, dry_run: bool = False) -> list[dict]:
    """Start `n` GPU pods (sequentially: each create waits for ssh, ~40 s) and record every placed pod in state.json. The budget
    gates are pods.launch's; the farm's own budget (policy.budget_usd) caps the SUM of the worst cases."""
    backend = backend or backend_for(dance)
    st = State(dance)
    room = None
    if pol.budget_usd is not None:
        spent, committed = Wt.farm_cost(st.load(), time.time())
        room = max(0.0, pol.budget_usd - spent - committed)
    res = backend.launch_gpu(n, gpus=pol.gpus or None, minutes=pol.minutes, hard_minutes=pol.hard_minutes, max_cost=room,
                             kinds=Wt.GPU_KINDS, prefix=f"hm-farm-{Wt.dance_tag(dance)}", dry_run=dry_run)
    if not dry_run:
        for r in res:
            if r.get("pod_id"):
                Wt.record_pod(st, r, "gpu", by="launch")
    return res


def launch_cpu_pods(dance: str, n: int, pol: Wt.Policy, *, backend=None, dry_run: bool = False) -> list[dict]:
    backend = backend or backend_for(dance)
    st = State(dance)
    res = backend.launch_cpu(n, vcpu=pol.cpu_vcpu, minutes=pol.cpu_minutes, hard_minutes=pol.cpu_hard_minutes, max_cost=None,
                             kinds=Wt.CPU_KINDS, prefix=f"hm-farm-{Wt.dance_tag(dance)}-cpu", dry_run=dry_run)
    if not dry_run:
        for r in res:
            if r.get("pod_id"):
                Wt.record_pod(st, r, "cpu", by="launch")
    return res


def plan_launch(dance: str, a, pol: Wt.Policy, backend) -> dict:
    """What `launch` would do now, from state.json (queued jobs) and the live offers. Raises SystemExit-style errors as dicts."""
    rows = job_rows(dance)
    queued = [r for r in rows if r["status"] == "queued"]
    gpu_jobs = [r for r in queued if r["needs_gpu"]]
    cpu_jobs = [r for r in queued if r["needs_gpu"] is False]
    want = a.pods if getattr(a, "pods", None) is not None else pol.target_gpu_pods
    alive = [p for p in State(dance).load()["pods"].values()
             if p.get("role") == "gpu" and p.get("state") not in ("deleted", "gone", "lost", "stuck")]
    # pods this farm already runs count: a second `launch` / `up` tops up to the target and to the number of waiting jobs, it never doubles
    n = want if getattr(a, "force_n", False) else max(0, min(want - len(alive), len(gpu_jobs) - len(alive)))
    offers = []
    with contextlib.suppress(Exception):
        offers = backend.offers(pol.gpus or None)
    price = offers[0]["price"] if offers else None
    est = None
    if gpu_jobs and price:
        est = estimate_run([r["est_min"] or 60.0 for r in gpu_jobs], max(1, n), price, pol.hard_minutes, float(getattr(a, "slowdown", GPU_SLOWDOWN)))
    return {"queued_gpu_jobs": [r["id"] for r in gpu_jobs], "queued_cpu_jobs": [r["id"] for r in cpu_jobs], "pods_wanted": want,
            "pods_running": [p["id"] for p in alive], "pods_to_start": n, "offers": offers[:6], "estimate": est,
            "note": "estimates use A100 minutes x slowdown (RTX 4090 measured 2026-10-08: 900 frames fit in 16.1 min vs 39 min modelled on an A100)"}


def _monitor_up(a, what: str) -> None:
    """The dashboard (the defrag view) runs whenever the farm works (owner 2026-10-21). Never fatal."""
    if getattr(a, "dry_run", False):
        return
    try:
        from ..pipeline import monitor
        monitor.ensure_running(f"farm {what} {a.dance}", route=f"#/farm/{a.dance}")
    except Exception:  # noqa: BLE001
        pass


def cmd_launch(a) -> int:
    _monitor_up(a, "launch")
    dance = a.dance
    backend = backend_for(dance)
    base = Wt.Policy.load(dance)
    pol = Wt.policy_from_args(a, base)
    plan = plan_launch(dance, a, pol, backend)
    cpu_n = int(getattr(a, "cpu", 0) or 0)
    out = {"dance": dance, "dry_run": bool(a.dry_run), "policy": {k: getattr(pol, k) for k in ("target_gpu_pods", "gpus", "minutes", "hard_minutes", "budget_usd")},
           **plan}
    if a.dry_run:
        res = []
        if plan["pods_to_start"] > 0:
            try:
                res = launch_gpu_pods(dance, plan["pods_to_start"], pol, backend=backend, dry_run=True)
            except Exception as e:  # noqa: BLE001
                res = [{"error": f"{type(e).__name__}: {str(e)[:300]}"}]
        out["would_start"] = res
        _print(a, out, _launch_text(out))
        return 0
    if plan["pods_to_start"] <= 0 and not cpu_n:
        out["error"] = "nothing to launch: no queued GPU job (run `farm queue` first, or --force-n)"
        _print(a, out, out["error"])
        return 2
    if pol != base:
        pol.save(dance)
    res = []
    if plan["pods_to_start"] > 0:
        say(f"starting {plan['pods_to_start']} GPU pod(s) for {len(plan['queued_gpu_jobs'])} job(s): {', '.join(plan['queued_gpu_jobs'])}")
        try:
            res = launch_gpu_pods(dance, plan["pods_to_start"], pol, backend=backend)
        except Exception as e:  # noqa: BLE001
            res = [{"error": f"{type(e).__name__}: {str(e)[:400]}"}]
    if cpu_n:
        res += launch_cpu_pods(dance, cpu_n, pol, backend=backend)
    out["started"] = res
    placed = [r for r in res if r.get("pod_id")]
    if not a.no_watch and (placed or Wt.health(dance)["running"] is False and plan["queued_gpu_jobs"]):
        out["watcher"] = Wt.start_detached(dance)
    _print(a, out, _launch_text(out))
    return 0 if placed else 1


def _launch_text(o: dict) -> str:
    lines = [f"{'DRY RUN: ' if o['dry_run'] else ''}jobs waiting: GPU {', '.join(o['queued_gpu_jobs']) or '-'}; CPU {', '.join(o['queued_cpu_jobs']) or '-'}"]
    for k, x in enumerate(o.get("offers", [])[:4]):
        lines.append(f"  offer {x['gpu']} ${x['price']:.2f}/h stock {x['stock']}" + ("  <- first choice" if k == 0 else ""))
    if o.get("estimate"):
        e = o["estimate"]
        lines.append(f"  estimate: {e['pods']} pod(s), about {e['wall_min']:.0f} min wall, ${e['cost_usd']} expected, ${e['worst_usd']} worst case "
                     f"(kill switch {o['policy']['minutes']:.0f} min, ceiling {o['policy']['hard_minutes']:.0f} min; GPU slowdown x{e['slowdown']})")
    for r in o.get("would_start", []) + o.get("started", []):
        if r.get("error"):
            lines.append(f"  ERROR {r['error']}")
        elif r.get("dry_run"):
            lines.append(f"  would start {r.get('gpu') or r.get('flavor')} ${r['price']:.2f}/h worst case ${r['worst']}")
        else:
            lines.append(f"  started {r['pod_id']} {r.get('gpu')} ${r.get('cost_per_hr'):.2f}/h ssh {r.get('host')}:{r.get('port')}")
    if "watcher" in o:
        w = o["watcher"]
        lines.append(f"  watcher: {'already running' if w.get('already_running') else 'started' if w.get('started') else 'NOT started'} (pid {w.get('pid')}), log {w.get('log')}")
    if o.get("error"):
        lines.append(o["error"])
    return "\n".join(lines)


def _print(a, obj: dict, text: str) -> None:
    print(json.dumps(obj, default=str) if getattr(a, "json", False) else text)


# ------------------------------------------------------------------------------------------------ up


def cmd_up(a) -> int:
    """plan -> queue (code + world + inputs uploaded, hash-diffed) -> launch -> watcher. --dry-run changes nothing and prints the
    plan, the queue and the launch estimate."""
    from . import cli
    _monitor_up(a, "up")
    dance = a.dance
    backend = backend_for(dance)
    base = Wt.Policy.load(dance)
    pol = Wt.policy_from_args(a, base)
    steps: dict = {}
    parser = cli.build_parser()
    if not a.no_plan:
        pa = parser.parse_args(["plan", *(["--dry-run"] if a.dry_run else []), *(["--no-final"] if a.no_final else [])])
        pa.dance, pa.json = dance, False
        cli.cmd_plan(pa)
        steps["plan"] = "dry run" if a.dry_run else "done"
    if a.dry_run:
        qa = parser.parse_args(["queue", "--local-only"])
        qa.dance, qa.json = dance, True
        qa.only, qa.exclude = _csv(a.only), _csv(a.exclude)
        import io
        buf, old = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            cli.cmd_queue(qa)
        finally:
            sys.stdout = old
        steps["queue"] = json.loads(buf.getvalue() or "{}")
        only, exclude = set(_csv(a.only)), set(_csv(a.exclude))
        if "ready" in steps["queue"]:   # what the uploader would actually take
            steps["queue"]["ready"] = [j for j in steps["queue"]["ready"] if (not only or j in only) and j not in exclude]
        plan = plan_launch(dance, a, pol, backend)
        steps["launch"] = plan
        _print(a, {"dance": dance, "dry_run": True, "steps": steps}, _up_text(steps, True))
        return 0
    if pol != base:
        pol.save(dance)
    qa = parser.parse_args(["queue", *(["--hold"] if pol.hold else [])])
    qa.dance, qa.json = dance, False
    qa.only, qa.exclude = _csv(a.only), _csv(a.exclude)
    qa.keep_gateway = True
    say("queue: uploading code, world and job inputs to the volume ...")
    qres = cli.queue_jobs(dance, hold=pol.hold, args=qa)
    steps["queue"] = {k: qres.get(k) for k in ("queued", "held", "missing", "errors")}
    if not qres.get("queued") and not qres.get("held"):
        steps["error"] = "nothing was queued: " + json.dumps({k: qres.get(k) for k in ("missing", "errors", "ready")}, default=str)[:400]
        _print(a, {"dance": dance, "steps": steps}, _up_text(steps, False))
        return 2
    plan = plan_launch(dance, a, pol, backend)
    steps["launch"] = plan
    placed = []
    if plan["pods_to_start"] > 0:
        say(f"launch: {plan['pods_to_start']} GPU pod(s) for {', '.join(plan['queued_gpu_jobs'])}")
        try:
            res = launch_gpu_pods(dance, plan["pods_to_start"], pol, backend=backend)
        except Exception as e:  # noqa: BLE001
            res = [{"error": f"{type(e).__name__}: {str(e)[:400]}"}]
        steps["started"] = res
        placed = [r for r in res if r.get("pod_id")]
    steps["watcher"] = Wt.start_detached(dance)   # even with no pod yet: the watcher launches replacements and CPU pods itself
    _print(a, {"dance": dance, "steps": steps}, _up_text(steps, False))
    return 0 if (placed or not plan["queued_gpu_jobs"]) and steps["watcher"].get("started", steps["watcher"].get("already_running")) else 1


def _up_text(steps: dict, dry: bool) -> str:
    lines = [f"{'DRY RUN: ' if dry else ''}up"]
    q = steps.get("queue") or {}
    if dry:
        lines.append(f"  would queue: {', '.join(q.get('ready') or []) or '-'}   held: {', '.join(h['id'] for h in q.get('held', [])) or '-'}")
    else:
        lines.append(f"  queued: {', '.join(q.get('queued') or []) or '-'}   held: {', '.join(h['id'] for h in (q.get('held') or [])) or '-'}")
    if steps.get("launch"):
        lines.append(_launch_text({"dry_run": dry, "policy": {"minutes": 0, "hard_minutes": 0}, **steps["launch"],
                                   "started": steps.get("started", []), "would_start": []}).replace("(kill switch 0 min, ceiling 0 min; ", "("))
    if "watcher" in steps:
        w = steps["watcher"]
        lines.append(f"  watcher {'already running' if w.get('already_running') else 'started' if w.get('started') else 'NOT started'}, log {w.get('log')}")
    if steps.get("error"):
        lines.append("  " + steps["error"])
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ pull


def pull_ready(dance: str, link, ids: list[str] | None = None, *, redo: bool = False, backend=None) -> dict:
    """Verified pull of finished jobs through `link`: `ids`, else every job state.json calls queued / running / done that the volume
    shows done. Updates state.json like the watcher. Returns {pulled: {id: result}, failed: {id: error}, pending: [...]}."""
    st = State(dance)
    cur = st.load()
    cand = ids or [jid for jid, j in {**cur["jobs"], **cur["chunks"]}.items()
                   if (cur["chunks"].get(jid) or j).get("status") in Wt.ACTIVE and Wt.is_mine(cur["chunks"].get(jid) or j)]
    snap = P.take_snapshot(link, cand, [])
    out: dict = {"pulled": {}, "failed": {}, "pending": []}
    w = Wt.Recorder(dance, log=say)
    for jid in cand:
        E = snap["jobs"].get(jid) or {}
        if not E.get("done"):
            out["pending"].append(jid)
            continue
        if not redo and (cur["chunks"].get(jid) or cur["jobs"].get(jid) or {}).get("status") in ("pulled", "stitched"):
            local = Wt.chunks_dir(dance) / jid
            prev = Wt.read_json(local / "pull.json", None) or {}
            if prev.get("verified") and prev.get("tar_sha256") == ((E.get("manifest") or {}).get("tar") or {}).get("sha256"):
                continue
        try:
            take = Wt.load_spec(dance, jid).get("take")
            res = P.pull_job(link, jid, Wt.chunks_dir(dance) / jid, Wt.workspace_root(), take=take,
                             progress=lambda rel, pos, size, j=jid: None)
            w.pull_done(jid, res)
            out["pulled"][jid] = {k: res.get(k) for k in ("files", "tar_bytes", "seconds", "world_id", "already")}
        except Exception as e:  # noqa: BLE001
            out["failed"][jid] = f"{type(e).__name__}: {str(e)[:300]}"
            say(f"pull {jid} failed: {out['failed'][jid]}")
    if out["pulled"]:
        with contextlib.suppress(Exception):
            from . import cli
            cli.refresh_coverage(dance)
    return out


def cmd_pull(a) -> int:
    dance = a.dance
    with volume_link(dance, keep=False) as link:
        r = pull_ready(dance, link, _csv(a.ids) or None, redo=a.redo)
    _print(a, r, "pulled: " + (", ".join(r["pulled"]) or "-") + "\nstill running / not finished: " + (", ".join(r["pending"]) or "-")
           + ("\nFAILED: " + json.dumps(r["failed"]) if r["failed"] else ""))
    return 1 if r["failed"] else 0


# ------------------------------------------------------------------------------------------------ down


def verified_local(dance: str, jid: str, E: dict) -> bool:
    """Is the finished job's CURRENT result (the tar sha256 on the volume) home with a verified pull.json?"""
    tar = (E.get("manifest") or {}).get("tar") or {}
    prev = Wt.read_json(Wt.chunks_dir(dance) / jid / "pull.json", None) or {}
    return bool(prev.get("verified") and tar and prev.get("tar_sha256") == tar.get("sha256"))


def cmd_down(a) -> int:
    """Verified pull of everything finished, then delete this farm's pods (only those recorded in state.json). Jobs still running on a
    pod keep it alive unless --abandon. The volume is never touched."""
    dance = a.dance
    backend = backend_for(dance)
    st = State(dance)
    cur = st.load()
    pods = [p for p in cur["pods"].values() if p.get("state") not in ("deleted", "gone", "lost") and not p.get("borrowed")]
    if a.pod:
        pods = [p for p in pods if p["id"] in set(_csv(a.pod))]
    h = Wt.health(dance)
    if h["running"] and not a.keep_watcher:
        say("stopping the watcher (it would re-launch pods) ...")
        Wt.stop(dance, force=True)
    out: dict = {"dance": dance, "pods": [p["id"] for p in pods], "deleted": [], "kept": {}, "pull": None}
    if not pods:
        _print(a, out, "no pods of this farm are open")
        return 0
    pull_res = None
    try:
        with volume_link(dance, backend=backend, keep=True) as link:
            pull_res = pull_ready(dance, link, backend=backend)
            snap = P.take_snapshot(link, [j for j in {*cur["jobs"], *cur["chunks"]}], [])
    except Exception as e:  # noqa: BLE001
        out["pull_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        snap = None
    out["pull"] = pull_res
    for p in pods:
        why = None
        if snap is not None:   # the volume's truth, not state.json's: a result counts as home only with a verified pull.json of THAT tar
            here = [j for j, E in snap["jobs"].items() if E.get("done") and (E["done"].get("pod_id") == p["id"])]
            unpulled = [j for j in here if not verified_local(dance, j, snap["jobs"][j])]
            run = [j for j, E in snap["jobs"].items() if (E.get("claim") or {}).get("pod_id") == p["id"] and not E.get("done")]
            if unpulled:
                why = f"finished but not pulled/verified: {', '.join(unpulled)}"
            elif run and not a.abandon:
                why = f"still running {', '.join(run)} (use --abandon to delete anyway)"
        else:
            unpulled = []
            if not a.abandon:
                why = "volume unreachable: cannot verify what is running or pulled (use --abandon)"
        if why:
            out["kept"][p["id"]] = why
            continue
        try:
            backend.delete(p["id"], "farm down")
            Wt.mark_pod_deleted(st, p["id"], "farm down (verified pull)", None)
            out["deleted"].append(p["id"])
        except Exception as e:  # noqa: BLE001
            out["kept"][p["id"]] = f"delete failed: {type(e).__name__}: {str(e)[:200]}"
    _print(a, out, f"deleted: {', '.join(out['deleted']) or '-'}\n" + "".join(f"kept {k}: {v}\n" for k, v in out["kept"].items())
           + "volume untouched.")
    return 0 if not out["kept"] else 1


# ------------------------------------------------------------------------------------------------ CLI registration


def register_commands(sub) -> None:
    p = sub.add_parser("launch", help="start GPU pods from the volume for the queued jobs (gates, kill switches), then the watcher")
    Wt.add_policy_flags(p)
    p.add_argument("--cpu", type=int, default=0, help="also start N CPU analysis pods now (the watcher starts one when the stitch job is ready)")
    p.add_argument("--dry-run", action="store_true", help="prices, worst case and the plan; creates nothing")
    p.add_argument("--no-watch", action="store_true", help="do not start the watcher")
    p.add_argument("--force-n", action="store_true", help="start --pods pods even when fewer jobs are queued")
    p.add_argument("--slowdown", type=float, default=GPU_SLOWDOWN, help="estimate: GPU time relative to the A100 the estimates are in (default %s: the RTX 4090, measured)" % GPU_SLOWDOWN)
    p.set_defaults(func=cmd_launch)
    p = sub.add_parser("up", help="plan -> queue (code, world, inputs) -> launch -> watch, in one go")
    Wt.add_policy_flags(p)
    p.add_argument("--only", help="comma list of job ids to queue / launch for (e.g. g0000,g0900)")
    p.add_argument("--exclude", help="comma list of job ids to leave alone (e.g. a,b,c,d: another session's fits)")
    p.add_argument("--no-plan", action="store_true", help="use the job specs already planned")
    p.add_argument("--no-final", action="store_true", help="plan without the final stitch+analyse job")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--slowdown", type=float, default=GPU_SLOWDOWN)
    p.add_argument("--force-n", action="store_true")
    p.set_defaults(func=cmd_up)
    p = sub.add_parser("pull", help="verified pull of finished jobs (through any pod that mounts the volume)")
    p.add_argument("ids", nargs="?", help="comma list of job ids (default: every active job the volume shows done)")
    p.add_argument("--redo", action="store_true", help="pull again even when a verified pull exists")
    p.set_defaults(func=cmd_pull)
    p = sub.add_parser("down", help="verified pull, then delete this farm's pods (only those in state.json; the volume stays)")
    p.add_argument("--pod", help="comma list of pod ids (default: all of this farm's)")
    p.add_argument("--abandon", action="store_true", help="delete pods even if a job is still running on them")
    p.add_argument("--keep-watcher", action="store_true")
    p.set_defaults(func=cmd_down)
