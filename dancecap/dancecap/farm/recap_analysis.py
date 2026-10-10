"""Analyses of the class recap on CPU pods: the stages of the demo that need no music (2026-10-09, agent F2).

The recap (larissa_kadu_recap) has no soundtrack, so the demo's beat / rhythm / BPM products are NOT produced here. What runs, each as its
own CPU-pod job behind the stitch job `final` (the stitched arrays arrive as the archive input `chunks/final/out/result.tar`, nothing is
uploaded from the PC), in parallel:

    an-timing-w  dancecap.timing_free   touchdowns, accents, step intervals / bouts / cadence, lead-follow offsets (raw fit), analysed in 30-s
                                        windows (the dense smoother operators of timing.py ran a 16 GB pod out of memory on 360 s)
    an-physics-m dancecap physics       COM, F_net + band, couple GRF, XCoM (raw fit; `physics` skips its beat impulses by itself). Job kind
                                        `analyse_mem`: only a pod started with kinds=analyse_mem (cpu5m, 64 GB) claims it, because physics on
                                        the 10815-frame take also exceeded 16 GB (SIGKILL, twice) and a small pod must never take it
    an-floor     counterbalance, floorcraft   intervals / pivots / T axes / dials / path (refined fit); both fall back to a nominal beat
                                              LENGTH (60 / 76.4 s) as a TIME SCALE when there is no grid, so their `*_beats` fields are
                                              meaningless for this take (the seconds fields are real)

    python -m dancecap.farm --dance larissa_kadu_recap recap-analysis [--dry-run] [--only an-timing,...] [--mc 60]
    python -m dancecap.farm --dance larissa_kadu_recap queue --only an-timing-w,an-physics-m,an-floor
    python -m dancecap.farm --dance larissa_kadu_recap launch --cpu 1              # the analyse pod(s)
    python -c "from dancecap.farm import recap_analysis as R; R.launch_mem_pod('larissa_kadu_recap')"   # the 64 GB pod of an-physics-m

Registered with the farm's analysis registry too (`farm analyse --list`): timing_free, physics_free.
"""
from __future__ import annotations

import json
import sys

from . import analysis as A
from .state import State, now_iso

DANCE = "larissa_kadu_recap"
AFTER = "final"
MEM_KIND = "analyse_mem"      # claimed only by the pod launch_mem_pod starts (64 GB); the 16 GB `analyse` pods never take it
THREADS = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "MPLBACKEND": "Agg"}
SINGLE = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"}

for _a in (
    A.Analysis("timing_free", "timing_free", "--mc 60", 40, optional=False, outputs=("timing/",),
               desc="footstep touchdowns, accents, step intervals / bouts / cadence, lead-follow offsets: NO beat grid (raw fit)"),
    A.Analysis("physics_free", "physics", "", 30, optional=False, outputs=("physics/",),
               desc="L0 centroidal dynamics without beat impulses (raw fit)"),
):
    A.register_analysis(_a)


def job_defs(refined_dir: str = "smplx_rf2_noslip", mc: int = 60, workers: int = 8) -> list[dict]:
    """The jobs as plain dicts: id, what, steps [(name, base, args, minutes, env)], outputs (paths under work/<take>/), estimate."""
    return [
        {"id": "an-timing-w", "steps": [("timing_free", "timing_free", f"--mc {mc} --workers {workers} --mc-corr-noise-mm 10", 60, SINGLE)],
         "outputs": ["timing/"], "est_min": 15},
        {"id": "an-physics-m", "steps": [("physics", "physics", "", 60, THREADS)], "outputs": ["physics/"], "est_min": 20,
         "kind": MEM_KIND, "min_mem_gb": 32},
        {"id": "an-floor", "steps": [("counterbalance", "counterbalance", f"--smplx-dir {refined_dir} --no-review", 40, THREADS),
                                     ("floorcraft", "floorcraft", f"--smplx-dir {refined_dir}", 40, THREADS)],
         "outputs": ["counterbalance/", "floorcraft/"], "est_min": 25},
    ]


def build_jobs(spec, *, only=None, mc: int = 60, workers: int = 8, code: dict | None = None) -> list[dict]:
    """Farm job specs for the recap analyses: kind analyse (CPU pod), after the stitch job, inputs = its result archive on the volume."""
    from . import worker
    take = spec.name
    jobs = []
    for d in job_defs(spec.refined_dir, mc, workers):
        if only and d["id"] not in only:
            continue
        steps = []
        for name, base, args, minutes, env in d["steps"]:
            st = worker.analysis_steps(base, take, args, minutes=minutes, name=name)[0]
            st["env"] = dict(env)
            st["stage"] = "analyse"
            steps.append(st)
        outputs = [{"path": f"dancecap/work/{take}/{p}"} for p in d["outputs"]]
        chunk = {"id": d["id"], "take": take, "k0": spec.grid.k0, "k1": spec.grid.k1, "origin": "analysis", "kind": "analyse"}
        job = A._job(spec, d["id"], "analyse", take, steps=steps, outputs=outputs,
                     inputs=[{"path": f"chunks/{AFTER}/out/result.tar", "archive": True}], after=[AFTER], analyses=[], world=None,
                     code=code, gpu=False, max_minutes=2.0 * sum(s[3] for s in d["steps"]), estimate_min=d["est_min"], chunk=chunk)
        job["analyses"] = [s[0] for s in d["steps"]]
        job["resources"] = {"cpu_only": True, "min_vcpu": workers, "est_minutes": d["est_min"], **({"min_mem_gb": d["min_mem_gb"]} if "min_mem_gb" in d else {})}
        job["kind"] = d.get("kind", job["kind"])
        jobs.append(job)
    return jobs


def launch_mem_pod(dance: str = DANCE, vcpu: int = 8, minutes: float = 75, hard_minutes: float = 120) -> list[dict]:
    """One memory-optimised CPU pod (cpu5m / cpu3m, 8 GB per vCPU = 64 GB at 8 vCPUs) that claims ONLY jobs of kind analyse_mem, recorded
    in the farm state so the watcher pulls the result and deletes the pod like any farm pod. Calls pods.launch_cpu directly: the farm's
    RunpodBackend.launch_cpu does not pass `flavors` on and would start a 16 GB cpu5c pod that takes the job and runs out of memory."""
    from . import pods
    from . import watch as Wt
    from .state import State
    res = pods.launch_cpu(1, vcpu=vcpu, flavors=["cpu5m", "cpu3m"], minutes=minutes, hard_minutes=hard_minutes, kinds=MEM_KIND,
                          prefix=f"hm-farm-{Wt.dance_tag(dance)}-mem", owner="F2-recap-analysis")
    st = State(dance)
    for r in res:
        if r.get("pod_id"):
            Wt.record_pod(st, r, "cpu", by="launch")
    return res


def cmd_recap_analysis(a) -> int:
    from . import chunks as C
    from .timeline import DanceSpec
    from . import farm_dir
    spec = DanceSpec.load(a.dance)
    only = {x for x in (a.only or "").split(",") if x}
    jobs = build_jobs(spec, only=only or None, mc=a.mc, workers=a.workers, code=None if a.dry_run else C.code_info())
    for j in jobs:
        if not a.dry_run:
            A.write_job(j)
            State(a.dance).set_job(j["id"], kind="analyse", status="planned", after=j["after"], written=now_iso(),
                                   spec=str(farm_dir(a.dance) / "jobs" / f"{j['id']}.json"))
        print(f"job {j['id']}: " + " ; ".join(s["cmd"] for s in j["steps"]) + f"  (after {','.join(j['after'])}, CPU pod)")
    if not a.dry_run:
        print("next: python -m dancecap.farm --dance " + a.dance + " queue --only " + ",".join(j["id"] for j in jobs))
    if a.json:
        print(json.dumps([{k: j[k] for k in ("id", "after", "outputs")} for j in jobs]))
    return 0


def register(sub) -> None:
    p = sub.add_parser("recap-analysis", help="plan the beat-free analyses of the class recap (timing_free, physics, counterbalance + floorcraft) "
                                               "as CPU-pod jobs behind the stitch job `final`")
    p.add_argument("--only", help="comma list of job ids (an-timing, an-physics, an-floor)")
    p.add_argument("--mc", type=int, default=60, help="Monte-Carlo draws of the step timing")
    p.add_argument("--workers", type=int, default=8, help="processes of the step timing Monte-Carlo (= vCPUs of the pod)")
    p.add_argument("--dry-run", action="store_true", help="print the jobs, write nothing")
    p.set_defaults(func=cmd_recap_analysis)


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
