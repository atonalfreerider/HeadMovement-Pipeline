"""john_lula analyses on CPU pods (agent J4, 2026-10-11): farm/ca_analysis.py for a take whose stitched motion is already on this machine.

    python -m dancecap.farm --dance john_lula jl-analysis [--dry-run] [--only jlan-timing,jlan-physics] [--mc 60]
    HM_OWNER=J4 HM_QUEUE_OWNER=J4 HM_POD_OWNER=J4 python -m dancecap.farm --dance john_lula queue --only jlan-timing,jlan-physics
    python -c "from dancecap.farm import jl_analysis as R; R.launch_mem_pod()"        # the 64 GB pod (kind analyse_mem)

The inputs are the PROMOTED arrays on this machine (smplx/, smplx_rf2_noslip/: the depth-spike repair of J3b was applied after the pod's stitch, so the pod's
result.tar is not the motion in use) + the beat grid (song/beats_heat.json, segments.json) in ONE small archive uploaded by `queue`; no `after`.
Jobs (spec field "owner": "J4": only a worker whose HM_OWNER is J4 claims them):
    jlan-timing   timing (refined fit, the heat's beat grid, steps inside the no-beat intervals dropped, ONE camera) -> counterbalance -> floorcraft
    jlan-physics  physics (raw stitched fit as the demo / Carlos+Aline; the beat grid)
Both are kind analyse_mem (a 64 GB pod: the timing step's dense smoother operators grow with the square of the take; 88 s needs far less than the 194 s of
Carlos+Aline, but a SIGKILL costs a launch, a 64 GB pod costs cents).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from . import analysis as A
from . import chunks as C
from . import jobs_dir
from .state import State, now_iso
from ..take import PACKAGE_ROOT

DANCE = "john_lula"
CAPTURE = "John-Lula"
OWNER = "J4"
MEM_KIND = "analyse_mem"
IN_REL = f"src/{CAPTURE}/jl_an_in.tar"
REFINED = "smplx_rf2_noslip"
SINGLE = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
THREADS = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "MPLBACKEND": "Agg"}
BEATS = f"work/{DANCE}/song/beats_heat.json"
SYNC_MS, N_CAMS = 15.0, 1       # one camera: the audio is the phone's own; 15 ms = the Carlos+Aline per-camera residual (frame quantisation + clock), no averaging over cameras
WORK = PACKAGE_ROOT / "work" / DANCE


def no_beat_arg(seg: dict) -> str:
    iv = sorted([float(a), float(b)] for a, b in seg["no_beat"])
    return ",".join(f"{a:g}:{b:g}" for a, b in iv)


def job_defs(no_beat: str, mc: int = 60, workers: int = 8) -> list[dict]:
    t_args = (f"--beats {BEATS} --smplx-dir {REFINED} --no-refine --mc {mc} --workers {workers} --mc-corr-noise-mm 10 --sync-ms {SYNC_MS:g} --n-cams {N_CAMS} "
              f"--no-beat {no_beat}")
    return [
        {"id": "jlan-timing", "est_min": 20, "steps": [("timing", "timing", t_args, 60, SINGLE),
                                                         ("counterbalance", "counterbalance", f"--smplx-dir {REFINED} --no-review", 30, THREADS),
                                                         ("floorcraft", "floorcraft", f"--smplx-dir {REFINED}", 30, THREADS)],
         "outputs": ["timing/", "counterbalance/", "floorcraft/"], "kind": MEM_KIND, "min_mem_gb": 32},
        {"id": "jlan-physics", "est_min": 15, "steps": [("physics", "physics", f"--beats {BEATS}", 40, THREADS)], "outputs": ["physics/"], "kind": MEM_KIND, "min_mem_gb": 32},
    ]


def input_archive(out: Path | None = None) -> dict:
    out = out or WORK / "farm" / "jl_an_in.tar"
    rel = f"dancecap/work/{DANCE}"
    items = [(WORK / "song" / "beats_heat.json", f"{rel}/song/beats_heat.json"), (WORK / "song" / "segments.json", f"{rel}/song/segments.json"),
             (WORK / "clock" / "alignment.json", f"{rel}/clock/alignment.json")]
    for d in ("smplx", REFINED):
        for r in ("lead", "follow"):
            items.append((WORK / d / f"{r}.npz", f"{rel}/{d}/{r}.npz"))
    C.pack_tar(out, items)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    return {"path": IN_REL, "sha256": h, "bytes": out.stat().st_size, "local": C._ws(out), "archive": True}


def build_jobs(spec, seg: dict, in_tar: dict, *, mc: int = 60, workers: int = 8, code: dict | None = None) -> list[dict]:
    from . import worker
    jobs = []
    for d in job_defs(no_beat_arg(seg), mc, workers):
        steps = []
        for name, base, args, minutes, env in d["steps"]:
            st = worker.analysis_steps(base, DANCE, args, minutes=minutes, name=name)[0]
            st["env"] = dict(env)
            st["stage"] = "analyse"
            steps.append(st)
        outputs = [{"path": f"dancecap/work/{DANCE}/{p}"} for p in d["outputs"]]
        chunk = {"id": d["id"], "take": DANCE, "k0": spec.grid.k0, "k1": spec.grid.k1, "origin": "analysis", "kind": "analyse"}
        job = A._job(spec, d["id"], "analyse", DANCE, steps=steps, outputs=outputs, inputs=[dict(in_tar)], after=[], analyses=[], world=None, code=code, gpu=False,
                     max_minutes=2.0 * sum(s[3] for s in d["steps"]), estimate_min=d["est_min"], chunk=chunk)
        job["analyses"] = [s[0] for s in d["steps"]]
        job["resources"] = {"cpu_only": True, "min_vcpu": workers, "est_minutes": d["est_min"], "min_mem_gb": d["min_mem_gb"]}
        job["kind"] = d.get("kind", job["kind"])
        job["env"] = {**(job.get("env") or {}), "HM_FRONT_DANCE": DANCE, "HM_FRONT_CAPTURE": CAPTURE}
        job["profile"] = DANCE
        job["owner"] = OWNER
        jobs.append(job)
    return jobs


def launch_mem_pod(vcpu: int = 8, minutes: float = 40, hard_minutes: float = 70) -> list[dict]:
    """One 64 GB CPU pod (cpu5m / cpu3m) that claims ONLY analyse_mem jobs of owner J4 (HM_QUEUE_OWNER=J4 in the environment), recorded in the farm state."""
    from . import pods
    from . import watch as Wt
    os.environ.setdefault("HM_QUEUE_OWNER", OWNER)
    os.environ.setdefault("HM_POD_OWNER", OWNER)
    res = pods.launch_cpu(1, vcpu=vcpu, flavors=["cpu5m", "cpu3m"], minutes=minutes, hard_minutes=hard_minutes, kinds=MEM_KIND,
                          prefix=f"hm-farm-{Wt.dance_tag(DANCE)}-mem", owner=OWNER)
    st = State(DANCE)
    for r in res:
        if r.get("pod_id"):
            Wt.record_pod(st, r, "cpu", by="launch")
    return res


def cmd_jl_analysis(a) -> int:
    from .timeline import DanceSpec
    spec = DanceSpec.load(a.dance)
    seg = json.loads((WORK / "song" / "segments.json").read_text(encoding="utf-8"))
    in_tar = {"path": IN_REL} if a.dry_run else input_archive()
    jobs = build_jobs(spec, seg, in_tar, mc=a.mc, workers=a.workers, code=None if a.dry_run else C.code_info())
    only = {x for x in (a.only or "").split(",") if x}
    jobs = [j for j in jobs if not only or j["id"] in only]
    for j in jobs:
        print(f"job {j['id']} (kind {j['kind']}, owner {j['owner']}): " + " ; ".join(s["cmd"] for s in j["steps"]))
        if not a.dry_run:
            A.write_job(j)
            State(a.dance).set_job(j["id"], kind=j["kind"], status="planned", after=j["after"], written=now_iso(), spec=str(jobs_dir(a.dance) / f"{j['id']}.json"))
    return 0


def register(sub) -> None:
    p = sub.add_parser("jl-analysis", help="J4: john_lula timing (beat grid, refined fit, no-beat steps dropped) + counterbalance + floorcraft + physics as CPU-pod jobs (owner J4)")
    p.add_argument("--only", default="", help="comma list of job ids (jlan-timing, jlan-physics)")
    p.add_argument("--mc", type=int, default=60)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_jl_analysis)


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
