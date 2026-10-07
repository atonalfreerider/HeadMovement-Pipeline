"""Dancers stage (SAM 2.1 masks, identity repair, body/hair split, QA) for many cameras IN PARALLEL on one big GPU.

    python -m dancecap.dancers_pod run larissa_kadu_38 [--cams 01,02] [--workers 5]   # on an 80 GB pod
    python -m dancecap.dancers_pod write-job larissa_kadu_38                          # local: dancers/pod_job.json

Same per-camera work as `python -m dancecap dancers <take>` (dancers.run_camera, unchanged), but each camera runs in
its own process (DANCECAP_GPU_SHARED=1: no work/gpu.lock; SAM 2.1 large needs ~3-5 GB of an 80 GB GPU per camera)
under CUDA MPS when available, largest cameras first. Each worker writes dancers/pod_cam_<cam>.json (its roles.json
frames + report entry); the parent then merges them into dancers/roles.json and dancers/report.json exactly as
dancers.run does (labels_source, swaps_repaired, identity_flags) and renders review/dancers_body_hair.jpg.

Inputs (made locally first): frames/<cam>/ (+ frames.json), people/<cam>.npz, dancers/labels.json (+ overrides) from
`python -m dancecap dancers-labels <take>` (review dancers/labels_review.jpg BEFORE the pod run). Host RAM: a worker
holds all its frames, masks and probabilities (cam02 at 2302 x 1080p: ~32 GB), so --workers bounds the peak.
Pod job (schema dancecap.pod_job/1, bootstrap feature "dancers" = SAM 2.1 at the local commit): `write-job`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .take import PACKAGE_ROOT, WORKSPACE, Take


def _take(name: str):
    try:
        return Take.load(name)
    except FileNotFoundError:
        raise SystemExit(f"take {name}: takes/{name}.toml or its alignment file is missing")


def one(take_name: str, cam_id: str) -> None:
    """Worker: one camera, no GPU lock (DANCECAP_GPU_SHARED=1 is set by the parent)."""
    from .dancers import Cam, DancersConfig, load_labels, run_camera
    from .dancers_sam import build_predictor
    take = _take(take_name)
    work = take.work
    cfg = DancersConfig()
    labels = load_labels(work)
    held: dict = {}

    def predictor():
        if "p" not in held:
            held["p"] = build_predictor()
        return held["p"]
    t0 = time.time()
    r = run_camera(Cam.load(work, cam_id), labels[cam_id], predictor, cfg)
    r["report"]["wall_seconds_pod"] = round(time.time() - t0, 1)
    out = {"frames": r["frames"], "report": r["report"],
           "sam_overrides": getattr(held.get("p"), "dancecap_overrides", None)}
    p = work / "dancers" / f"pod_cam_{cam_id}.json"
    p.write_text(json.dumps(out))
    rp = r["report"]
    print(f"[dancers] {cam_id}: {rp['frames']} frames, SAM {rp['sam_s_per_frame']:.2f} s/frame "
          f"(total {rp['total_s_per_frame']:.2f}), anchors {rp['anchors']}, embrace {rp['embrace_frames']}, "
          f"swaps repaired {rp['swaps_repaired']}, tight {100 * rp['tight_rate']:.1f}%"
          + (f", IDENTITY FLAGS {rp['identity_flags']}" if rp["identity_flags"] else ""), flush=True)


def merge(take: Take, cams: list[str]) -> dict:
    """pod_cam_<cam>.json -> roles.json + report.json (dancers.run's bookkeeping) + review/dancers_body_hair.jpg."""
    from .dancers import DancersConfig, body_hair_review, load_labels
    work = take.work
    out = work / "dancers"
    labels = load_labels(work)
    roles_path, report_path = out / "roles.json", out / "report.json"
    roles = json.loads(roles_path.read_text()) if roles_path.exists() else {"cams": {}}
    report = json.loads(report_path.read_text()) if report_path.exists() else {"cams": {}}
    for c in cams:
        p = out / f"pod_cam_{c}.json"
        if not p.exists():
            print(f"[dancers] {c}: no result (worker failed) - not merged", flush=True)
            continue
        r = json.loads(p.read_text())
        roles["cams"][c] = {"frames": r["frames"]}
        report["cams"][c] = r["report"]
        if r.get("sam_overrides") is not None:
            report["sam_overrides"] = r["sam_overrides"]
    roles["labels_source"] = {k: labels[k]["by"] for k in labels}
    roles["swaps_repaired"] = [{"cam": k, "start": s0, "end": s1}
                               for k, v in sorted(report["cams"].items()) for s0, s1 in v["swaps_repaired"]]
    roles["identity_flags"] = {k: v.get("identity_flags", []) for k, v in sorted(report["cams"].items())
                               if v.get("identity_flags")}
    report["parallel_run"] = {"host": "pod" if os.environ.get("DANCECAP_POD") else "local",
                              "module": "dancecap.dancers_pod", "cams": cams}
    roles_path.write_text(json.dumps(roles))
    report_path.write_text(json.dumps(report, indent=1))
    body_hair_review(take, report, DancersConfig())
    return report


def _mps_start(env: dict) -> dict | None:
    if not shutil.which("nvidia-cuda-mps-control"):
        return None
    pipe, logd = Path("/tmp/dancecap_mps"), Path("/tmp/dancecap_mps_log")
    pipe.mkdir(exist_ok=True)
    logd.mkdir(exist_ok=True)
    menv = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(logd))
    if subprocess.run(["nvidia-cuda-mps-control", "-d"], env=menv).returncode != 0:
        return None
    env.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(logd))
    return menv


def run_parallel(take_name: str, cams: list[str] | None, workers: int, timeout_min: float) -> dict:
    take = _take(take_name)
    work = take.work
    cams = cams or [s.id for s in take.sources]
    labels = json.loads((work / "dancers" / "labels.json").read_text())
    missing = [c for c in cams if c not in labels]
    if missing:
        raise SystemExit(f"dancers/labels.json has no label for {missing}: run dancers-labels first")
    for c in cams:
        (work / "dancers" / f"pod_cam_{c}.json").unlink(missing_ok=True)
    nfr = {c: len(json.loads((work / "frames" / c / "frames.json").read_text())["frames"])
           * (4 if json.loads((work / "frames" / c / "frames.json").read_text())["width"] >= 1000 else 1) for c in cams}
    order = sorted(cams, key=lambda c: -nfr[c])  # largest first: the long poles start immediately
    env = dict(os.environ, DANCECAP_GPU_SHARED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4")
    mps = _mps_start(env)
    logs = work / "dancers" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"[dancers] {len(cams)} cameras, {workers} workers, CUDA MPS {'on' if mps else 'off'}: {order}", flush=True)

    def go(c):
        t0 = time.time()
        with open(logs / f"dancers_{c}.log", "w") as fh:
            pr = subprocess.Popen([sys.executable, "-m", "dancecap.dancers_pod", "one", take_name, c], stdout=fh,
                                  stderr=subprocess.STDOUT, env=env, cwd=str(PACKAGE_ROOT))
            try:
                rc = pr.wait(timeout=timeout_min * 60)
            except subprocess.TimeoutExpired:
                pr.kill()
                pr.wait()
                rc = "timeout"
        s = round(time.time() - t0, 1)
        print(f"[dancers] {c}: rc={rc} in {s:.0f}s", flush=True)
        return c, {"rc": rc, "seconds": s}
    try:
        with ThreadPoolExecutor(workers) as ex:
            res = dict(ex.map(go, order))
    finally:
        if mps:
            subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", env=mps)
    rep = merge(take, cams)
    (work / "dancers" / "pod_run.json").write_text(json.dumps({"workers": workers, "mps": bool(mps), "cams": res},
                                                              indent=1))
    bad = [c for c, r in res.items() if r["rc"] != 0]
    if bad:
        raise SystemExit(f"[dancers] cameras failed: {bad} (logs in {logs})")
    return rep


def write_job(take_name: str, workers: int = 5, max_minutes: float = 75) -> Path:
    """dancers/pod_job.json (dancecap.pod_job/1): uploads frames + people + labels + the SAM 2.1 checkpoint."""
    from .dancers_sam import CHECKPOINT
    take = _take(take_name)
    work_rel = f"dancecap/work/{take_name}"
    align = (PACKAGE_ROOT / "takes" / take.cfg["alignment"]).resolve().relative_to(WORKSPACE).as_posix()
    d = take.work / "dancers"
    extra = [f"{work_rel}/dancers/labels_override.json"] if (d / "labels_override.json").exists() else []
    job = {
        "schema": "dancecap.pod_job/1", "name": "dancers", "take": take_name, "max_minutes": max_minutes,
        "needs": ["dancers"],
        "inputs": [f"{work_rel}/frames/", f"{work_rel}/people/", f"{work_rel}/dancers/labels.json", *extra, align,
                   CHECKPOINT.relative_to(WORKSPACE).as_posix()],
        "exclude": [f"{work_rel}/people/qa", "*.mp4"],
        "commands": [{"name": "dancers", "timeout_min": max_minutes - 5,
                      "cmd": f"python -m dancecap.dancers_pod run {take_name} --workers {workers} "
                             f"--timeout-min {max_minutes - 10}"}],
        "outputs": [f"{work_rel}/dancers/", f"{work_rel}/review/dancers_body_hair.jpg"],
        "verify": [f"{work_rel}/dancers/roles.json", f"{work_rel}/dancers/report.json"]
                  + [f"{work_rel}/dancers/{s.id}/masks/" for s in take.sources],
        "after_download": [f"python -m dancecap.cameras.reuse ... / cameras {take_name} (attach, world, qa need "
                           f"dancers/roles.json)", f"python -m dancecap review {take_name} --instants 8"],
        "notes": ["front stage 2026-10-06: SAM 2.1 propagation of ~10k frames x 8 cameras moved off the 11 GB "
                  "workstation; per-camera processes under CUDA MPS (dancecap.dancers_pod)",
                  "outputs = the whole dancers/ dir (masks/body/hair PNGs, roles.json, report.json, qa/, cache/ SAM "
                  "probabilities for --reuse-sam, logs/)"],
    }
    p = d / "pod_job.json"
    p.write_text(json.dumps(job, indent=1))
    return p


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m dancecap.dancers_pod", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "one", "merge", "write-job"])
    p.add_argument("take")
    p.add_argument("cam", nargs="?")
    p.add_argument("--cams")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--timeout-min", type=float, default=65)
    p.add_argument("--max-minutes", type=float, default=75)
    a = p.parse_args(argv)
    if a.action == "one":
        one(a.take, a.cam)
    elif a.action == "run":
        run_parallel(a.take, a.cams.split(",") if a.cams else None, a.workers, a.timeout_min)
    elif a.action == "merge":
        take = _take(a.take)
        merge(take, a.cams.split(",") if a.cams else [s.id for s in take.sources])
    else:
        print(write_job(a.take, a.workers, a.max_minutes))


if __name__ == "__main__":
    main()
