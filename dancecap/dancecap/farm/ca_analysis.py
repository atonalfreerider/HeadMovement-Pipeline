"""carlos_aline analyses on CPU pods (agent C6, 2026-10-10): the recap's analysis jobs (farm/recap_analysis.py, recap_fix.analysis2_jobs) WITH the beat grid.

    python -m dancecap.farm --dance carlos_aline ca-analysis [--dry-run] [--after cafinal2] [--out-take carlos_aline_t6]
    python -m dancecap.farm --dance carlos_aline queue --only ca-an-timing,ca-an-physics
    python -m dancecap.farm --dance carlos_aline launch --cpu 1 --pods 0
    python -c "from dancecap.farm import ca_analysis as R; R.launch_mem_pod()"          # the 64 GB pod of ca-an-physics (kind analyse_mem)

Jobs (each behind the stitch job, whose result archive is its input; a first step copies the stitched motion from work/<out_take>/ to work/carlos_aline/):
    ca-an-timing   timing (refined fit smplx_rf2_noslip, the studio beat grid song/beats_studio.json, steps inside the no-beat intervals dropped, 2 cameras) ->
                   counterbalance -> floorcraft (both read the grid from timing/timing.json; refined fit)           CPU pod, 8 vCPU
    ca-an-physics  physics (raw stitched fit, as the recap and the demo; the beat grid)                             64 GB pod (analyse_mem)
The small input archive (beats_studio.json, segments.json) is packed here and uploaded by `queue` like any job input.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from . import analysis as A
from . import farm_dir, jobs_dir
from . import chunks as C
from .state import State, now_iso
from ..take import PACKAGE_ROOT

DANCE = "carlos_aline"
CAPTURE = "Carlos-Aline"
AFTER = "cafinal2"
OUT_TAKE = "carlos_aline_t6"
MEM_KIND = "analyse_mem"
IN_REL = f"src/{CAPTURE}/ca_an_in.tar"
REFINED = "smplx_rf2_noslip"
SINGLE = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
THREADS = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "MPLBACKEND": "Agg"}
BEATS = f"work/{DANCE}/song/beats_studio.json"
PRELUDE = "mkdir -p work/{dance} && cp -r work/{out}/. work/{dance}/"
SYNC_MS, N_CAMS = 15.0, 2              # per-camera residual video-sync SD (C3: flat +-20 ms) and the number of cameras averaged


def no_beat_arg(seg: dict) -> str:
    """`A:B,A:B` for `dancecap timing --no-beat`: the no-beat intervals of song/segments.json plus the seam between the two governing grids (no whole measure of either)."""
    iv = [list(map(float, x)) for x in seg["no_beat"]]
    g = seg.get("governing", {})
    if g.get("kojo_until_s") is not None and g.get("guetta_from_s") is not None:
        iv.append([float(g["kojo_until_s"]), float(g["guetta_from_s"])])
    iv.sort()
    return ",".join(f"{a:g}:{b:g}" for a, b in iv)


def job_defs(no_beat: str, mc: int = 60, workers: int = 8) -> list[dict]:
    t_args = (f"--beats {BEATS} --smplx-dir {REFINED} --no-refine --mc {mc} --workers {workers} --mc-corr-noise-mm 10 --sync-ms {SYNC_MS:g} --n-cams {N_CAMS} "
              f"--no-beat {no_beat}")
    return [
        {"id": "ca-an-timing", "est_min": 40, "steps": [("timing", "timing", t_args, 80, SINGLE),
                                                          ("counterbalance", "counterbalance", f"--smplx-dir {REFINED} --no-review", 40, THREADS),
                                                          ("floorcraft", "floorcraft", f"--smplx-dir {REFINED}", 40, THREADS)],
         "outputs": ["timing/", "counterbalance/", "floorcraft/"], "kind": MEM_KIND, "min_mem_gb": 32},   # the timing step's 8 Monte-Carlo workers were SIGKILLed (rc -9) on a 16 GB pod after 19 min
        {"id": "ca-an-physics", "est_min": 25, "steps": [("physics", "physics", f"--beats {BEATS}", 60, THREADS)], "outputs": ["physics/"], "kind": MEM_KIND, "min_mem_gb": 32},
    ]


def input_archive(out: Path | None = None) -> dict:
    out = out or PACKAGE_ROOT / "work" / DANCE / "farm" / "ca_an_in.tar"
    w = PACKAGE_ROOT / "work" / DANCE / "song"
    rel = f"dancecap/work/{DANCE}/song"
    C.pack_tar(out, [(w / "beats_studio.json", f"{rel}/beats_studio.json"), (w / "segments.json", f"{rel}/segments.json")])
    import hashlib
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    return {"path": IN_REL, "sha256": h, "bytes": out.stat().st_size, "local": C._ws(out), "archive": True}


def build_jobs(spec, seg: dict, in_tar: dict, *, after: str = AFTER, out_take: str = OUT_TAKE, mc: int = 60, workers: int = 8, code: dict | None = None) -> list[dict]:
    from . import worker
    jobs = []
    for d in job_defs(no_beat_arg(seg), mc, workers):
        steps = [{"name": "stage_motion", "cmd": PRELUDE.format(dance=DANCE, out=out_take), "timeout_min": 10, "optional": False, "stage": "analyse"}]
        for name, base, args, minutes, env in d["steps"]:
            st = worker.analysis_steps(base, DANCE, args, minutes=minutes, name=name)[0]
            st["env"] = dict(env)
            st["stage"] = "analyse"
            steps.append(st)
        outputs = [{"path": f"dancecap/work/{DANCE}/{p}"} for p in d["outputs"]]
        chunk = {"id": d["id"], "take": DANCE, "k0": spec.grid.k0, "k1": spec.grid.k1, "origin": "analysis", "kind": "analyse"}
        job = A._job(spec, d["id"], "analyse", DANCE, steps=steps, outputs=outputs, inputs=[{"path": f"chunks/{after}/out/result.tar", "archive": True}, dict(in_tar)],
                     after=[after], analyses=[], world=None, code=code, gpu=False, max_minutes=2.0 * sum(s[3] for s in d["steps"]), estimate_min=d["est_min"], chunk=chunk)
        job["analyses"] = [s[0] for s in d["steps"]]
        job["resources"] = {"cpu_only": True, "min_vcpu": workers, "est_minutes": d["est_min"], **({"min_mem_gb": d["min_mem_gb"]} if "min_mem_gb" in d else {})}
        job["kind"] = d.get("kind", job["kind"])
        job["env"] = {**(job.get("env") or {}), "HM_FRONT_DANCE": DANCE, "HM_FRONT_CAPTURE": CAPTURE}
        job["profile"] = "carlos_aline"
        jobs.append(job)
    return jobs


def launch_mem_pod(vcpu: int = 8, minutes: float = 75, hard_minutes: float = 120) -> list[dict]:
    """One 64 GB CPU pod (cpu5m / cpu3m) that claims ONLY analyse_mem jobs, recorded in the farm state (recap_analysis.launch_mem_pod, this dance, owner from HM_POD_OWNER)."""
    import os
    from . import pods
    from . import watch as Wt
    res = pods.launch_cpu(1, vcpu=vcpu, flavors=["cpu5m", "cpu3m"], minutes=minutes, hard_minutes=hard_minutes, kinds=MEM_KIND,
                          prefix=f"hm-farm-{Wt.dance_tag(DANCE)}-mem", owner=os.environ.get("HM_POD_OWNER") or "C6")
    st = State(DANCE)
    for r in res:
        if r.get("pod_id"):
            Wt.record_pod(st, r, "cpu", by="launch")
    return res


def cmd_ca_analysis(a) -> int:
    from .timeline import DanceSpec
    spec = DanceSpec.load(a.dance)
    seg = json.loads((PACKAGE_ROOT / "work" / DANCE / "song" / "segments.json").read_text(encoding="utf-8"))
    in_tar = {"path": IN_REL} if a.dry_run else input_archive()
    jobs = build_jobs(spec, seg, in_tar, after=a.after, out_take=a.out_take, mc=a.mc, workers=a.workers, code=None if a.dry_run else C.code_info())
    only = {x for x in (a.only or "").split(",") if x}
    jobs = [j for j in jobs if not only or j["id"] in only]
    for j in jobs:
        print(f"job {j['id']} (kind {j['kind']}, after {','.join(j['after'])}): " + " ; ".join(s["cmd"] for s in j["steps"]))
        if not a.dry_run:
            A.write_job(j)
            State(a.dance).set_job(j["id"], kind=j["kind"], status="planned", after=j["after"], written=now_iso(), spec=str(jobs_dir(a.dance) / f"{j['id']}.json"))
    return 0


def register(sub) -> None:
    p = sub.add_parser("ca-analysis", help="C6: carlos_aline timing (beat grid, refined fit, no-beat steps dropped) + counterbalance + floorcraft + physics as CPU-pod jobs behind the stitch")
    p.add_argument("--only", default="", help="comma list of job ids (ca-an-timing, ca-an-physics)")
    p.add_argument("--after", default=AFTER)
    p.add_argument("--out-take", dest="out_take", default=OUT_TAKE)
    p.add_argument("--mc", type=int, default=60)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_ca_analysis)


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
