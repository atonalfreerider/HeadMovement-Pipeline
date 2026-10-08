"""python -m dancecap.dancer_splats <take> {prepare,write-pod-job,train,render,eval,export,review,run-pod} [options]

See dancecap/dancer_splats/__init__.py and docs/DANCER_SPLATS_RESEARCH.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import SplatConfig, default_window, preset


def _cfg(a) -> SplatConfig:
    cfg = preset(a.preset)
    cfg.window = (float(a.window[0]), float(a.window[1])) if a.window else default_window(a.take)
    if a.cams:
        cfg.cams = tuple(c.strip() for c in a.cams.split(",") if c.strip())
    if a.holdout is not None:
        cfg.holdout_cams = () if a.holdout.lower() in ("none", "") else tuple(a.holdout.split(","))
    if a.max_side is not None:
        cfg.max_side = int(a.max_side)
    if a.time_budget_min is not None:
        cfg.time_budget_min = float(a.time_budget_min)
    if a.iters:
        cfg.iters = tuple(int(x) for x in a.iters.split(","))
    if a.init_samples is not None:
        cfg.init_samples = int(a.init_samples)
    if a.device:
        cfg.device = a.device
    return cfg


def main(argv=None) -> int:
    from ..take import PACKAGE_ROOT
    ap = argparse.ArgumentParser(prog="python -m dancecap.dancer_splats", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("action", choices=["prepare", "write-pod-job", "train", "render", "eval", "export", "review", "run-pod"])
    ap.add_argument("--preset", default="pod", choices=["pod", "smoke", "local"])
    ap.add_argument("--out", help="output dir (default work/<take>/dancer_splats); the bundle is <out>/bundle")
    ap.add_argument("--window", nargs=2, type=float, help="prepare: sync-corrected reference seconds")
    ap.add_argument("--cams", help="prepare: comma-separated cameras (default all)")
    ap.add_argument("--holdout", help="prepare: held-out cameras (default 03; 'none')")
    ap.add_argument("--max-side", type=int, help="prepare: downsize crops to this max side (smoke)")
    ap.add_argument("--init-samples", type=int)
    ap.add_argument("--force", action="store_true", help="prepare: rewrite a fresh bundle; run-pod: re-run done jobs")
    ap.add_argument("--allow-stale", action="store_true", help="prepare: accept smplx/ older than world_from_sfm.json")
    ap.add_argument("--time-budget-min", type=float, help="train: shrink the schedule to end inside this")
    ap.add_argument("--iters", help="train: S1,S2,S3 steps")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--review", help="eval: review dir (default work/<take>/review/splats)")
    ap.add_argument("--cam", default="03", help="render: bundle camera or 'novel'")
    ap.add_argument("--t", type=float, help="render: reference time")
    ap.add_argument("--sh", type=int, default=1, help="export: SH degree of the PLYs")
    ap.add_argument("--max-gaussians", type=int, help="export: cap per frame (opacity x volume)")
    ap.add_argument("--max-cost", type=float, help="run-pod: session ceiling ($)")
    ap.add_argument("--retry", action="store_true", help="write-pod-job / run-pod: a retry session (uploads the "
                    "checkpoint / model the failed session brought back; budget = what is left of the approved $2)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    work = PACKAGE_ROOT / "work" / a.take
    out = Path(a.out) if a.out else work / "dancer_splats"
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    bundle = out / "bundle"
    default_out = out.resolve() == (work / "dancer_splats").resolve()
    cfg = _cfg(a)
    if a.action == "prepare":
        from .data import prepare
        man = prepare(a.take, cfg, out, force=a.force, allow_stale=a.allow_stale)
        print(json.dumps({k: man[k] for k in ("n_views", "n_train", "n_test_cam", "n_test_time", "window_frames",
                                              "bytes", "prep_s")}, indent=1))
    elif a.action == "write-pod-job":
        from .pod import spec, write_pod_job
        p = write_pod_job(a.take, retry=a.retry)
        from .. import remote_jobs as rj
        job = rj.parse_job(spec(a.take, retry=a.retry), str(p))
        files = rj.bundle_files([job])
        mb = sum(f.size for f in files if not f.link) / 1e6
        print(f"written {p}: {len(job.commands)} commands, needs {rj.needed_features([job])}, {len(files)} bundle "
              f"files, {mb:.0f} MB raw (upload ~{mb / rj.UPLOAD_MBPS / 60:.1f} min at {rj.UPLOAD_MBPS} MB/s)")
        print(json.loads(p.read_text())["run"])
    elif a.action == "train":
        from .train import train
        train(bundle, out, cfg, resume=not a.no_resume)
    elif a.action == "eval":
        from .evaluate import evaluate
        review = Path(a.review) if a.review else work / "review" / "splats"
        evaluate(bundle, out, cfg, review, prefix="dancer_splats" if default_out else out.name)
    elif a.action == "export":
        from .export import export
        export(bundle, out, sh=a.sh, max_gaussians=a.max_gaussians, device=a.device or "cuda")
    elif a.action == "review":
        from .review import extra_review
        review = Path(a.review) if a.review else work / "review" / "splats"
        extra_review(bundle, out, review, a.take, device=a.device or "cuda")
    elif a.action == "render":
        from .export import render_at
        if a.t is None:
            ap.error("render needs --t")
        render_at(bundle, out, a.cam, a.t)
    elif a.action == "run-pod":
        from .pod import MAX_COST, run
        res = run(a.take, max_cost=a.max_cost or MAX_COST, dry_run=a.dry_run, force=a.force, retry=a.retry)
        print(json.dumps({k: res.get(k) for k in ("sid", "jobs", "error", "spent_total", "remaining")}, indent=1,
                         default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
