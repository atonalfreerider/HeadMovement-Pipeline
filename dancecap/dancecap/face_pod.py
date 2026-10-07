"""One Runpod job for the heavy face steps of a take (Runpod-first, user 2026-10-06: the PC has 4 logical CPUs and a
shared 2080 Ti; MediaPipe over thousands of face crops belongs on a pod, locally only orchestration + review).

    python -m dancecap.face_pod spec <take> [--write]          print the job (dancecap.pod_job/1); --write puts it at
                                                               work/<take>/face/pod_job.json (DISCOVERABLE by
                                                               `dancecap remote run --take <take>`: do not write it
                                                               while another batch of that take is in flight)
    python -m dancecap.face_pod run  <take> [--max-cost 2] [--dry-run] [--session SID]
                                                               own pod session (the spec is passed directly, nothing
                                                               discoverable is written)
    # or batch it with the take's other jobs in ONE session (frames uploaded once):
    python -m dancecap.face_pod spec <take> --write && python -m dancecap remote run --take <take> --jobs ...,face

The job runs, in order (remote_jobs: one bundle upload, bootstrap base, resumable commands with .done markers):
  setup       bash remote/face_setup.sh mediapipe (mediapipe 1.1.0 + face_landmarker.task from storage.googleapis.com)
  survey      python -m dancecap.face_survey <take>        face sizes / angles / occlusion per camera frame + jobs.json
  mediapipe   N sharded `face_worker jobs <take> --mp-only --shard i/N` processes (N = container CPUs - 1, <= 24),
              then `face_worker merge <take> N`             -> face/detections.npz (RTMW is skipped: not used)
  eval        python -m dancecap.face_eval <take>          (optional) detector rates / jitter / cross-camera checks
  expr        python -m dancecap.face_expr <take> --no-review   (optional) smplx_face/ + hold-out checks (CPU)
Locally afterwards (seconds to minutes, one process):
  python -m dancecap.face_expr <take> --review-only   adds the FLAME eyelid blendshapes (gated: never uploaded) and
                                                      draws review/face_expr.jpg + face_expr_confidence.png
  python -m dancecap.face_look <take>                 portrait vs video look check
  python -m dancecap.avatar_face <take>               portrait face textures (needs the bake + portrait landmarks)

The bundle holds the take's frames, dancer masks, cameras and SMPL-X fit (likeness + SMPL-X-licensed data on a private
pod that is deleted at the end) but no MANO / FLAME file, so community cloud is allowed. Estimate for larissa_kadu_38
(8 cameras x ~1150 frames, ~9000 face crops): MediaPipe ~0.06 s per crop per CPU -> ~1 min on 24 CPUs; survey ~5 min;
expr ~5 min; dominated by the frames upload (~2.6 GB at 2.5 MB/s ~ 18 min) unless batched with the texture bake.
"""
from __future__ import annotations

import json
import sys

from . import env
from .take import PACKAGE_ROOT

GPUS = ["NVIDIA RTX A5000", "NVIDIA RTX A4000", "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S",
        "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]       # CPU work: the cheapest pod with many vCPUs
CLOUDS = ["SECURE", "COMMUNITY"]
MAX_MINUTES = 90


def spec(take: str, expr: bool = True, survey: bool = True) -> dict:
    from .face_common import take_cams
    work = PACKAGE_ROOT / "work" / take
    w = f"dancecap/work/{take}"
    cams = take_cams(work)
    if not cams:
        raise SystemExit(f"{work}/cameras has no <cam>.npz: run the cameras stage first")
    missing = [r for r in ("lead", "follow") if not (work / "smplx" / f"{r}.npz").exists()]
    if missing:
        raise SystemExit(f"{work}/smplx/{missing[0]}.npz is missing: the face job needs the take's SMPL-X fit "
                         f"(MAMMA) first")
    md = env.mamma_data() / "body_models"
    inputs = [f"{w}/smplx/lead.npz", f"{w}/smplx/follow.npz", f"{w}/cameras/world_from_sfm.json",
              *[f"{w}/cameras/{c}.npz" for c in cams], f"{w}/frames/",
              *[f"{w}/dancers/{c}/masks/" for c in cams],
              {"local": str(md / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"),
               "pod": "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz"}]
    inputs += [f"{w}/dancers/{c}/hair/" for c in cams if (work / "dancers" / c / "hair").exists()]
    if (work / "avatar" / "head_fix.npz").exists():
        inputs.append(f"{w}/avatar/head_fix.npz")
    pf = [f"{w}/avatar/face/{r}_portrait_fit.npz" for r in ("lead", "follow")
          if (work / "avatar" / "face" / f"{r}_portrait_fit.npz").exists()]
    inputs += pf                                  # the expression fit reads the portrait fit for its report only
    if not survey:
        inputs += [f"{w}/face/survey.npz", f"{w}/face/survey.json", f"{w}/face/jobs.json"]
    shards = ("N=$(bash remote/face_setup.sh ncpu); S=$(( N > 25 ? 24 : (N > 2 ? N - 1 : 1) )); echo shards $S; "
              "for i in $(seq 0 $((S - 1))); do OMP_NUM_THREADS=1 python -m dancecap.face_worker jobs " + take +
              " --mp-only --shard $i/$S > work/" + take + "/face/logs_mp_shard_$i.log 2>&1 & done; wait; "
              "python -m dancecap.face_worker merge " + take + " $S")
    cmds = [{"name": "setup_mediapipe", "cmd": "bash remote/face_setup.sh mediapipe", "timeout_min": 10}]
    if survey:
        cmds.append({"name": "survey", "cmd": f"OMP_NUM_THREADS=4 python -m dancecap.face_survey {take}",
                     "timeout_min": 40})
    cmds += [{"name": "mediapipe", "cmd": f"mkdir -p work/{take}/face && " + shards, "timeout_min": 30},
             {"name": "eval", "cmd": f"python -m dancecap.face_eval {take}", "timeout_min": 10, "optional": True}]
    if expr:
        cmds.append({"name": "expr", "cmd": f"python -m dancecap.face_expr {take} --no-review", "timeout_min": 30,
                     "optional": True})
    return {
        "schema": "dancecap.pod_job/1", "name": "face", "take": take,
        "max_minutes": sum(c["timeout_min"] for c in cmds), "needs": [], "inputs": inputs, "commands": cmds,
        "exclude": [f"{w}/face/logs/*", f"{w}/face/pod_job.json", f"{w}/face/detections_shard*"],
        "env": {"PYTHONUNBUFFERED": "1"},
        "outputs": [f"{w}/face/", f"{w}/smplx_face/", f"{w}/review/"],
        "verify": [f"{w}/face/survey.npz", f"{w}/face/detections.npz"],
        "after_download": [f"python -m dancecap.face_expr {take} --review-only",
                           f"python -m dancecap.face_look {take}", f"python -m dancecap.avatar_face {take}"],
        "notes": ["no gated model in the bundle (FLAME eyelids are added locally): community cloud allowed",
                  "weights downloaded on the pod: MediaPipe face_landmarker.task (storage.googleapis.com, Apache-2.0)",
                  "RTMW is not run (--mp-only): its face points are unused and it answers even under hair"],
    }


def write_spec(take: str, **kw):
    p = PACKAGE_ROOT / "work" / take / "face" / "pod_job.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(spec(take, **kw), indent=1))
    return p


def run(take: str, max_cost: float = 2.0, dry_run: bool = False, sid: str | None = None, force: bool = False,
        expr: bool = True) -> dict:
    from . import remote_jobs as rj
    job = rj.parse_job(spec(take, expr=expr), f"face_pod:{take}")
    files = rj.bundle_files([job])
    up_min = sum(f.size for f in files) / 1e6 / rj.UPLOAD_MBPS / 60
    mm = min(MAX_MINUTES, max(30.0, rj.estimate_minutes([job], 0)["bootstrap"] + 30 + up_min))
    return rj.run_session([job], sid=sid, label=f"face-{take}", max_cost=max_cost, gpus=GPUS, clouds=CLOUDS,
                          dry_run=dry_run, force=force, max_minutes=mm)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["spec", "run"])
    ap.add_argument("take")
    ap.add_argument("--write", action="store_true", help="spec: write work/<take>/face/pod_job.json (discoverable)")
    ap.add_argument("--no-expr", action="store_true")
    ap.add_argument("--max-cost", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--session")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "spec":
        d = spec(a.take, expr=not a.no_expr)
        from . import remote_jobs as rj
        job = rj.parse_job(d, f"face_pod:{a.take}")
        files = rj.bundle_files([job])
        mb = sum(f.size for f in files) / 1e6
        print(json.dumps(d, indent=1))
        print(f"{len(job.commands)} commands, {len(files)} bundle files, {mb:.0f} MB raw "
              f"(upload ~{mb / rj.UPLOAD_MBPS / 60:.0f} min alone; nothing extra when batched with the frames' job)")
        if a.write:
            print("written:", write_spec(a.take, expr=not a.no_expr))
        return 0
    res = run(a.take, a.max_cost, a.dry_run, a.session, a.force, expr=not a.no_expr)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
