"""Recap people (agent R3): the pod job that finds the two teachers, and its PC-side orchestration.

    python -m dancecap.recap_people_run plan  [--no-detcompare]             job spec + cost estimate, nothing is started
    python -m dancecap.recap_people_run run   [--max-cost 2.5] [--minutes 60 --hard-minutes 100]
    python -m dancecap.recap_people_run pull                               pull a finished job again (resumable)

ONE farm job `rc-people` (kind front, a GPU pod of the hm-farm volume because of the detector comparison; the steps before it need only CPU):
tables -> world -> teachers -> assoc -> appearance -> finalize -> detcompare (optional). Everything it reads is already on the volume
(`src/Larissa-Kadu-recap/poses/*.json.gz`, the videos) or is a small input (R2's camera tracks, 20 MB). The pod is deleted when the job is pulled
(`finally`), and it has the farm's own kill switch. The spend is booked in the farm ledger under the owner R3-recap-people.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .take import PACKAGE_ROOT

OWNER = "R3-recap-people"
JOB = "rc-people"
WORK = PACKAGE_ROOT / "work" / "larissa_kadu_recap"
PEOPLE = WORK / "people"
STATE = PEOPLE / "run_state.json"
REL = "dancecap/work/larissa_kadu_recap"
CAMS = ("00", "01", "02", "03", "05", "06")
PRICE_4090 = 0.89


def log(msg: str) -> None:
    print(f"[recap_people_run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cameras_archive() -> dict:
    """R2's camera tracks (12 npz) + recap_cams.json as ONE deterministic tar input (the volume uploader's parallel tar batches race on the
    parent directories of many small inputs: 'Cannot open: No such file or directory'). The worker extracts it at workspace paths."""
    import hashlib

    from .farm import chunks as C
    names = [f"{c}.npz" for c in CAMS] + [f"{c}_world.npz" for c in CAMS] + ["recap_cams.json"]
    dest = PEOPLE / "in" / "cameras_in.tar"
    C.pack_tar(dest, [(WORK / "cameras" / n, f"{REL}/cameras/{n}") for n in names])
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {"path": f"{REL}/people/cameras_in.tar", "archive": True, "sha256": h, "bytes": dest.stat().st_size, "local": str(dest)}


def job_spec(detcompare: bool = True, step: int = 1, procs: int = 4, n_det: int = 300, code: dict | None = None, job_id: str = JOB) -> dict:
    """The dancecap.farm.job/1 spec of the people job (see the module doc)."""
    from .farm import worker
    cams = ",".join(CAMS)
    inputs = [cameras_archive()]
    steps = [
        {"name": "tables", "cmd": f"python -m dancecap.recap_people tables --cams {cams}", "timeout_min": 20},
        {"name": "world", "cmd": f"python -m dancecap.recap_people world --step {step}", "timeout_min": 40},
        {"name": "teachers", "cmd": f"python -m dancecap.recap_people teachers --step {step}", "timeout_min": 15},
        {"name": "assoc", "cmd": f"python -m dancecap.recap_people assoc --cams {cams} --procs {procs}", "timeout_min": 25},
        {"name": "appearance", "cmd": f"python -m dancecap.recap_people appearance --cams {cams} --procs {procs}", "timeout_min": 40},
        {"name": "finalize", "cmd": f"python -m dancecap.recap_people finalize --cams {cams}", "timeout_min": 15},
    ]
    if detcompare:
        steps += [{"name": "weights", "cmd": "python -m dancecap.front_full fetch-weights", "timeout_min": 12, "optional": True},
                  {"name": "detcompare", "cmd": f"python -m dancecap.recap_people detcompare --cams {cams} --n {n_det}", "timeout_min": 45,
                   "optional": True},
                  {"name": "detcompare_miss", "cmd": f"python -m dancecap.recap_people detcompare --cams {cams} --n {n_det} --mode miss", "timeout_min": 45,
                   "optional": True}]
    outs = [f"world_people.npz", "track3d.npz", "roles.npz", "teachers.json", "coverage.json", "detcompare.json"]
    outs += [f"assoc_{c}.npz" for c in CAMS] + [f"appearance_{c}.npz" for c in CAMS] + [f"teachers_{c}.npz" for c in CAMS]
    outs += [f"detcompare_{c}.json" for c in CAMS] + ["detcompare_miss.json"] + [f"detcompare_miss_{c}.json" for c in CAMS]
    outputs = [{"path": f"{REL}/people/{o}", **({"optional": True} if o.startswith("detcompare") else {})} for o in outs]
    spec = worker.analyse_spec(job_id, "larissa_kadu_recap", steps, outputs, inputs=inputs, max_minutes=120, priority=20, gpu=True, kind="front")
    spec.update({"code": code or {}, "retries": 1, "min_vram_gb": 10})
    return spec


def estimate(detcompare: bool = True, price: float = PRICE_4090) -> dict:
    """Minutes per step (measured on the PC for a single process, scaled for the pod's parallelism) and the dollars of one pod."""
    mins = {"pod start + env + code": 3.0, "tables (6 JSON.gz parses)": 2.0, "world (10815 instants, 17 ms each)": 3.5, "teachers (2 Viterbi passes)": 0.7,
            "assoc (6 cameras, 4 processes)": 2.0, "appearance (decode 6 videos, 4 processes; the master is the longest)": 7.0, "finalize": 1.0}
    if detcompare:
        mins["detcompare + detcompare miss (2 x 1800 frames, yolo12x + yolo11x-pose at two scales)"] = 20.0
    tot = sum(mins.values())
    return {"minutes": mins, "total_min": round(tot, 1), "gpu_hours": round(tot / 60, 2), "price_hr": price, "usd": round(tot / 60 * price, 2),
            "usd_with_50pct_margin": round(1.5 * tot / 60 * price, 2)}


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {"pods": {}, "jobs": {}, "pulled": []}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str))
    tmp.replace(STATE)


def run_job(spec: dict, minutes: float, hard_minutes: float, max_cost: float, gpus: list[str] | None, poll_s: float = 30.0, log_fn=log) -> dict:
    """Queue the job on the volume, start ONE GPU pod, wait, pull the result, delete every pod this call created."""
    from . import recap_cameras as rc
    from .farm import pods, volume
    st = load_state()
    mine: list[str] = []
    gw = None
    try:
        gw, _ = pods.gateway(minutes=45, vcpu=2, reuse=False, owner=OWNER)
        mine.append(gw.id)
        st["pods"][gw.id] = {"role": "gateway", "started": time.time(), "price_hr": gw.cost_per_hr}
        save_state(st)
        link = volume.Link(gw)
        info = volume.sync_code(link, make_current=False)
        spec["code"] = {"git_sha": info.get("git_sha"), "dirty": info.get("dirty"), "bundle_sha256": info.get("bundle_sha256")}
        volume.submit_job(link, spec)
        st["jobs"][spec["id"]] = {"status": "queued", "queued": time.time(), "code": info.get("key")}
        save_state(st)
        log_fn(f"job {spec['id']} queued (code {info['key']}); starting one GPU pod")
        launched = pods.launch(1, gpus=gpus, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds="front", idle_min=6,
                               prefix="hm-farm-rpeople", owner=OWNER)
        for l in launched:
            if l.get("pod_id"):
                mine.append(l["pod_id"])
                st["pods"][l["pod_id"]] = {"role": "gpu", "gpu": l.get("gpu"), "price_hr": l.get("cost_per_hr"), "started": time.time(), "kill_at": l.get("kill_at")}
            else:
                log_fn(f"launch problem: {l}")
        save_state(st)
        if not any(l.get("pod_id") for l in launched):
            raise RuntimeError(f"no GPU pod could be started: {launched}")
        t_last = 0.0
        while True:
            status = rc.poll(link, [spec["id"]]).get(spec["id"], "queued")
            st["jobs"][spec["id"]]["status"] = status
            save_state(st)
            if time.time() - t_last > 120:
                t_last = time.time()
                log_fn(f"job {spec['id']}: {status}")
            if status == "done":
                r = volume.pull_result(link, spec["id"], PACKAGE_ROOT.parent)
                log_fn(f"pulled {spec['id']}: {r['files']} files {r['bytes'] / 1e6:.1f} MB")
                st["pulled"].append(spec["id"])
                save_state(st)
                return {"status": "done", **r}
            if status == "failed":
                try:
                    log_fn("failed.json: " + json.dumps(link.read_json(f"chunks/{spec['id']}/failed.json"))[:1500])
                except Exception as e:  # noqa: BLE001
                    log_fn(f"cannot read failed.json: {e}")
                return {"status": "failed"}
            live = [p for p in mine if p != gw.id]
            if live and not any(rc._pod_alive(p) for p in live):
                raise RuntimeError("the GPU pod is gone while the job is still open")
            time.sleep(poll_s)
    finally:
        for pid in reversed(mine):
            try:
                pods.delete(pid, reason="recap people run finished")
                log_fn(f"pod {pid} deleted")
                st["pods"].get(pid, {})["deleted"] = time.time()
            except Exception as e:  # noqa: BLE001
                log_fn(f"WARNING pod {pid} NOT confirmed deleted: {e}")
        save_state(st)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.recap_people_run", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("plan", "run", "pull"))
    ap.add_argument("--no-detcompare", action="store_true")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--n-det", type=int, default=300)
    ap.add_argument("--job-id", default=JOB, help="farm job id (a finished id on the volume is never claimed again: use a new one per run)")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--hard-minutes", type=float, default=100)
    ap.add_argument("--max-cost", type=float, default=2.5)
    ap.add_argument("--gpus", default="NVIDIA GeForce RTX 4090,NVIDIA L4")
    a = ap.parse_args(argv)
    spec = job_spec(not a.no_detcompare, a.step, a.procs, a.n_det, job_id=a.job_id)
    est = estimate(not a.no_detcompare)
    if a.cmd == "plan":
        print(json.dumps({"job": {k: spec[k] for k in ("id", "kind", "needs_gpu", "max_minutes")}, "steps": [s["name"] for s in spec["steps"]],
                          "inputs": [i["path"] for i in spec["inputs"]], "outputs": len(spec["outputs"]), "estimate": est}, indent=1))
        return 0
    if a.cmd == "pull":
        from .farm import pods, volume
        gw, created = pods.gateway(minutes=20, vcpu=2, reuse=True, owner=OWNER)
        try:
            print(volume.pull_result(volume.Link(gw), a.job_id, PACKAGE_ROOT.parent))
        finally:
            if created:
                pods.delete(gw.id, reason="recap people pull done")
        return 0
    log(f"estimate: {est['total_min']} min = {est['usd']} USD ({est['usd_with_50pct_margin']} with 50 % margin); ceiling {a.max_cost}")
    res = run_job(spec, a.minutes, a.hard_minutes, a.max_cost, [g for g in a.gpus.split(",") if g])
    print(json.dumps(res, default=str))
    return 0 if res.get("status") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
