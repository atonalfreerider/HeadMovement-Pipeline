"""python -m dancecap.mono4d <command> ...   (one phone -> 4D provider -> virtual cameras -> MAMMA; see __init__.py)

    vendors                                 TypeSafe readiness watch of Gracia / Atlas / the World API (live pages)
    request  <take> --cam 06 --t0 --t1      cut the input, design the ring (+ held-out phones), write the request and
             [--provider manual|gracia|worldlabs] [--holdout all|none|01,02]  hand it to the provider
    fetch    <take> --run R                 validate the delivery in drop/ (or poll the API) -> result.json
    render-job <take> --run R               render_job.json for render_gracia.py (a .mint delivery)
    package  <take> --run R                 rendered views + the real phone -> work/<take>__m4d_<run>/ (CONTRACTS layout)
    judge    <take> --run R --allow-upload  Jev visual judgments of the generated views -> judge.json
    views2d  <take> --run R                 generated h<cam> views vs the real phones (needs detect/dancers on both)
    holdout  <take> --run R --test DIR [--baseline DIR] [--reference DIR]   3D errors vs the multi-camera fit
    gate     <take> --run R                 may the views enter MAMMA? -> quality_gate.json
    judge-calibrate [--times 3]             live check of the judge on labelled mannequin frames (no real people)
    oracle   <take> --run R                 stand-in delivery: the take's own fit drawn as mannequins at every camera
    viggle-test <take> --cam 06 --t0 58 --t1 62 --allow-upload
                                            Viggle JST-2 Motion 3D per dancer (masked crops), scored vs the take's fit
    atlas-test <take> --cam 06 --t 60 --targets 01,03,05,07 --allow-upload [--clean-plate]
                                            Atlas beta (Marble 2 API): pose one frame, render the other phones' poses,
                                            score them against those phones (YOLO PCK, Jev) -> mono4d/atlas/<cam>_<ms>/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT


def judge_presets() -> dict:
    from .judge import PRESETS
    return PRESETS


def _work(take: str) -> Path:
    return PACKAGE_ROOT / "work" / take


def _couple_xz(work: Path, t0: float, t1: float) -> np.ndarray:
    """The couple's floor centre over the window: the take's fit if there is one, else the one-camera start
    (mamma/mono_init.npz), else the world origin (CONTRACTS: the origin is the dancers' mean floor position)."""
    pts = []
    for role in ("lead", "follow"):
        p = work / "smplx" / f"{role}.npz"
        if p.exists():
            z = np.load(p)
            sel = (z["t"] >= t0) & (z["t"] <= t1)
            pts.append(z["joints"][sel, 0][:, [0, 2]])
    if not pts and (work / "mamma" / "mono_init.npz").exists():
        z = np.load(work / "mamma" / "mono_init.npz")
        pos = z["pos"].reshape(-1, 3)
        pts.append(pos[np.isfinite(pos).all(1)][:, [0, 2]])
    if not pts:
        return np.zeros((1, 2))
    return np.concatenate(pts).reshape(-1, 2)


def cmd_request(a) -> int:
    from .. import mamma_dyn as md
    from . import providers as pv
    from . import rig
    work = _work(a.take)
    run = a.run or f"{a.provider}_{a.cam}_{int(a.t0 * 1000)}_{int(a.t1 * 1000)}"
    rd = pv.run_dir(work, run)
    if (rd / "request" / "request.json").exists() and not a.force:
        raise SystemExit(f"{rd} has a request already (--force to rewrite it)")
    if rd.exists():  # a new request invalidates everything that was delivered or judged against the old one
        import shutil
        for d in ("request", "drop"):
            shutil.rmtree(rd / d, ignore_errors=True)
        for f in ("provider_job.json", "result.json", "render_job.json", "judge.json", "views2d.json", "holdout.json",
                  "quality_gate.json"):
            (rd / f).unlink(missing_ok=True)
    grid = md.master_grid(a.t0, a.t1)
    tracks = md.load_tracks(work)
    if a.cam not in tracks:
        raise SystemExit(f"no camera track for {a.cam} in {work / 'cameras'}")
    tr = tracks[a.cam]
    sel = (tr.t >= a.t0) & (tr.t <= a.t1)
    src = rig.ViewSpec(name=a.cam, kind="source", K=tr.K[sel], R=tr.R[sel], t=tr.tvec[sel], width=tr.width,
                       height=tr.height, ok=tr.ok[sel])
    W, H = (int(x) for x in a.size.lower().split("x"))
    xz = _couple_xz(work, a.t0, a.t1)
    centre = xz.mean(0)
    heights = tuple(float(h) for h in a.heights.split(","))
    radius = rig.auto_radius(xz, a.hfov, (W, H), heights) if a.radius == "auto" else float(a.radius)
    targets = rig.ring(len(grid), n=a.n, radius=radius, heights=heights, centre=tuple(centre), hfov_deg=a.hfov,
                       size=(W, H))
    held = [] if a.holdout == "none" else sorted(c for c in tracks if c != a.cam) if a.holdout == "all" \
        else [c.strip() for c in a.holdout.split(",") if c.strip()]
    for c in held:
        s = rig.from_track(tracks[c], grid)
        s.name = f"h{c}"
        targets.append(s)
    (rd / "request").mkdir(parents=True, exist_ok=True)
    info = pv.cut_input_video(work / "frames" / a.cam, a.t0, a.t1, rd / "request" / "inputs" / f"{a.cam}.mp4")
    if info["frames"] != src.T:
        raise SystemExit(f"input video has {info['frames']} frames, the camera track {src.T} in the window")
    (rd / "request" / "source_trajectory.json").write_text(json.dumps(
        rig.trajectory([src], tr.t[sel]), indent=None))
    (rd / "request" / "target_trajectory.json").write_text(json.dumps(rig.trajectory(targets, grid), indent=None))
    req = pv.Mono4DRequest(take=a.take, input_cam=a.cam, provider=a.provider, t0=a.t0, t1=a.t1, grid=grid.tolist(),
                           video=str(rd / "request" / "inputs" / f"{a.cam}.mp4"), video_fps=info["fps"],
                           video_frames=info["frames"], targets=[s.name for s in targets],
                           allow_upload=a.allow_upload,
                           notes=[f"ring: {a.n} x heights {a.heights} m, radius {radius:.2f} m around "
                                  f"({centre[0]:.2f}, {centre[1]:.2f}), hfov {a.hfov} deg, {W}x{H}",
                                  f"held-out phone poses: {held or 'none'}"])
    pv.save_request(rd, req)
    prov = pv.make_provider(a.provider, rd)
    job = prov.submit(req, spend_ok=a.yes_spend) if a.provider == "worldlabs" else prov.submit(req)
    (rd / "provider_job.json").write_text(json.dumps(job, indent=1))
    print(json.dumps({"run": run, "dir": str(rd), "grid_frames": len(grid), "targets": req.targets, "job": job},
                     indent=1))
    return 0


def cmd_fetch(a) -> int:
    from . import providers as pv
    rd = pv.run_dir(_work(a.take), a.run)
    req = pv.load_request(rd)
    job = json.loads((rd / "provider_job.json").read_text())
    res = pv.make_provider(req.provider, rd).fetch(req, job)
    if res is None:
        print(f"[mono4d] {a.run}: nothing delivered yet ({job.get('status')})")
        return 2
    pv.save_result(rd, res)
    print(json.dumps({"kind": res.kind, "dynamic": res.dynamic, "views": len(res.views), "asset": res.asset,
                      "world_from_scene": bool(res.world_from_scene), "notes": res.notes}, indent=1))
    return 0


def cmd_render_job(a) -> int:
    from . import providers as pv
    from . import rig
    rd = pv.run_dir(_work(a.take), a.run)
    req, res = pv.load_request(rd), pv.load_result(rd)
    if res.kind != "mint":
        raise SystemExit(f"render-job is for a .mint delivery; this run has {res.kind!r}")
    if not res.world_from_scene:
        raise SystemExit("no world_from_scene (drop/world_from_scene.json or drop/scene_cameras.json): our cameras "
                         "cannot be placed in the asset's frame")
    specs = rig.specs_from_trajectory(json.loads((rd / "request" / "target_trajectory.json").read_text()))
    sizes = {(s.width, s.height) for s in specs}
    if len(sizes) != 1:
        raise SystemExit(f"graciasdk renders one size per run; the targets have {sorted(sizes)} (split the job)")
    t0 = req.t0 if a.scene_t0 is None else a.scene_t0
    cams = {}
    for s in specs:
        sc = rig.to_scene(s, res.world_from_scene)
        idx = [0] if s.is_static() else range(sc.T)
        cams[s.name] = {"frames": [{"qvec": rig.qvec_wxyz(sc.R[i]).tolist(), "tvec": sc.t[i].tolist(),
                                    "fx": float(sc.K[i, 0, 0]), "fy": float(sc.K[i, 1, 1]),
                                    "cx": float(sc.K[i, 0, 2]), "cy": float(sc.K[i, 1, 2])} for i in idx]}
    W, H = sizes.pop()
    job = {"asset": res.asset, "out": str(rd / "drop" / "views"), "width": W, "height": H,
           "times": [float(t - t0) for t in req.grid], "cameras": cams}
    p = rd / "render_job.json"
    p.write_text(json.dumps(job))
    print(f"[mono4d] {p}: {len(cams)} cameras x {len(req.grid)} times; run render_gracia.py with Python >= 3.12 "
          "(see its docstring), then fetch again")
    return 0


def cmd_package(a) -> int:
    from . import providers as pv
    from .virtual_take import package
    rd = pv.run_dir(_work(a.take), a.run)
    out = package(a.take, a.run, pv.load_request(rd), pv.load_result(rd), vtake=a.vtake, force=a.force)
    print(f"[mono4d] virtual take -> {out}")
    return 0


def cmd_judge(a) -> int:
    from . import judge
    from . import providers as pv
    work = _work(a.take)
    rd = pv.run_dir(work, a.run)
    req = pv.load_request(rd)
    vwork = _work(a.vtake or f"{a.take}__m4d_{a.run}")
    man = json.loads((vwork / "mono4d.json").read_text())
    frames = {}
    for v in man["synthetic_cams"]:
        ref_work, ref_cam = (work, v[1:]) if v.startswith("h") and (work / "frames" / v[1:]).exists() else (None, None)
        if a.no_context:  # only the generated frame goes out (e.g. oracle mannequins: no real person in the request)
            ref_work = ref_cam = None
        rows = judge.sample_frames(vwork, v, a.every, ref_work, ref_cam)
        src = judge.sample_frames(vwork, v, a.every, vwork, req.input_cam)  # the input phone at the same time
        for r, s in zip(rows, src):
            r["source"] = None if a.no_context else s["reference"]
        frames[v] = rows
    rep = judge.judge_views(frames, allow_upload=a.allow_upload, workers=a.workers, th=judge.PRESETS[a.preset])
    (rd / "judge.json").write_text(json.dumps(rep, indent=1))
    for v, r in rep["views"].items():
        print(f"[mono4d] judge {v}: {r['bad']} of {r['judged']} frames flagged ({r['errors']} errors) -> "
              f"{'ok' if r['ok'] else 'DROP'}")
    print(f"[mono4d] {rep['input_tokens']} input tokens")
    return 0


def cmd_views2d(a) -> int:
    from . import holdout
    from . import providers as pv
    work = _work(a.take)
    rd = pv.run_dir(work, a.run)
    vwork = _work(a.vtake or f"{a.take}__m4d_{a.run}")
    man = json.loads((vwork / "mono4d.json").read_text())
    out = {}
    for v in man["synthetic_cams"]:
        if v.startswith("h") and (work / "people" / f"{v[1:]}.npz").exists() and (vwork / "people" / f"{v}.npz").exists():
            out[v] = holdout.compare_views(work, v[1:], vwork, v)
    (rd / "views2d.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


def cmd_holdout(a) -> int:
    from . import holdout
    from . import providers as pv
    work = _work(a.take)
    rd = pv.run_dir(work, a.run)
    req = pv.load_request(rd)
    ref = Path(a.reference) if a.reference else work / "smplx"
    doc = {"reference": str(ref), "test": a.test, "baseline": a.baseline, "window": [req.t0, req.t1],
           "route": holdout.compare_fits(ref, Path(a.test), (req.t0, req.t1))}
    if a.baseline:
        doc["baseline_fit"] = holdout.compare_fits(ref, Path(a.baseline), (req.t0, req.t1))
        doc["verdict"] = holdout.verdict(doc["route"], doc["baseline_fit"])
    (rd / "holdout.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in doc if k in ("route", "verdict")}, indent=1))
    return 0


def cmd_gate(a) -> int:
    from . import gate
    from . import providers as pv
    out = gate.decide(pv.run_dir(_work(a.take), a.run))
    print(json.dumps({k: out[k] for k in ("admit", "views_admitted", "views_dropped", "why")}, indent=1))
    return 0 if out["admit"] else 1


def cmd_vendors(a) -> int:
    from . import vendor_watch as vw
    doc = vw.run(a.vendor or None, save=not a.no_save)
    print(vw.table(doc))
    for v, d in doc["vendors"].items():
        for p in d["pages"]:
            if "error" in p:
                print(f"  {v}: {p['url']}: {p['error']}")
    print("changes since the last snapshot: " + ("; ".join(doc["changes"]) or "none"))
    return 0


def cmd_oracle(a) -> int:
    from .oracle import deliver
    print(json.dumps(deliver(a.take, a.run, a.reference), indent=1))
    return 0


def cmd_atlas_test(a) -> int:
    from .atlas_eval import run
    rep = run(a.take, a.cam, a.t, [c.strip() for c in a.targets.split(",") if c.strip()], clean_plate=a.clean_plate,
              use_jev=not a.no_jev, allow_upload=a.allow_upload)
    print(json.dumps({"focal_ratio": rep["atlas_focal_vs_ours"], "scale": rep["world_from_atlas"]["s"],
                      "pck_median": rep["pck_median"],
                      "targets": {k: {"dancers": v["dancers"], "jev": v.get("jev")} for k, v in rep["targets"].items()},
                      "clean_plate": rep.get("clean_plate"), "calls": rep["calls"]}, indent=1))
    return 0


def cmd_viggle_test(a) -> int:
    from .viggle import run
    rep = run(a.take, a.cam, a.t0, a.t1, allow_upload=a.allow_upload, fps=a.fps, reference=a.reference)
    print(json.dumps({r: {k: d[k] for k in ("frames", "joints", "pa_mpjpe_mm", "limb_err_deg_all")}
                      for r, d in rep["dancers"].items()}, indent=1))
    return 0


def cmd_calibrate(a) -> int:
    from . import calibrate
    doc = calibrate.run(times=a.times, with_source=not a.no_source)
    print(json.dumps(doc["per_question"], indent=1))
    print(f"[mono4d] {doc['images']} images, {doc['input_tokens']} input tokens ({doc['model']})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("vendors")
    sp.add_argument("--vendor", action="append", help="gracia | worldlabs_atlas | worldlabs_world_api (repeatable)")
    sp.add_argument("--no-save", action="store_true")
    sp.set_defaults(fn=cmd_vendors)
    sp = sub.add_parser("request")
    sp.add_argument("take")
    sp.add_argument("--cam", required=True, help="the one input phone")
    sp.add_argument("--t0", type=float, required=True)
    sp.add_argument("--t1", type=float, required=True)
    sp.add_argument("--run")
    sp.add_argument("--provider", default="manual", choices=["manual", "gracia", "worldlabs"])
    sp.add_argument("--n", type=int, default=8, help="ring cameras per height")
    sp.add_argument("--heights", default="1.5", help="ring heights in metres, comma-separated")
    sp.add_argument("--radius", default="auto")
    sp.add_argument("--hfov", type=float, default=55.0)
    sp.add_argument("--size", default="1080x1920", help="WxH of every generated view (renders stay 1080p)")
    sp.add_argument("--holdout", default="none", help="also render at these real phones' poses: all | none | 01,02")
    sp.add_argument("--allow-upload", action="store_true", help="the owner's go-ahead to send this footage out")
    sp.add_argument("--yes-spend", action="store_true", help="worldlabs: accept the credit cost of a generation")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_request)
    for name, fn in (("fetch", cmd_fetch), ("render-job", cmd_render_job), ("package", cmd_package),
                     ("judge", cmd_judge), ("views2d", cmd_views2d), ("holdout", cmd_holdout), ("gate", cmd_gate),
                     ("oracle", cmd_oracle)):
        sp = sub.add_parser(name)
        sp.add_argument("take")
        sp.add_argument("--run", required=True)
        sp.set_defaults(fn=fn)
        if name == "render-job":
            sp.add_argument("--scene-t0", type=float, help="reference seconds at the asset's t = 0 (default t0)")
        if name in ("package", "judge", "views2d"):
            sp.add_argument("--vtake", help="virtual take name (default <take>__m4d_<run>)")
        if name == "package":
            sp.add_argument("--force", action="store_true")
        if name == "judge":
            sp.add_argument("--allow-upload", action="store_true",
                            help="the owner's go-ahead to send these frames (likeness) to TypeSafe")
            sp.add_argument("--every", type=float, default=1.0, help="seconds between judged frames")
            sp.add_argument("--workers", type=int, default=4)
            sp.add_argument("--preset", default="real", choices=sorted(judge_presets()),
                            help="thresholds: real renders (default) or mannequin (oracle / calibration runs)")
            sp.add_argument("--no-context", action="store_true",
                            help="send only the generated frames (no real reference / source frames)")
        if name == "oracle":
            sp.add_argument("--reference", help="smplx dir to draw (default work/<take>/smplx)")
        if name == "holdout":
            sp.add_argument("--test", required=True, help="smplx dir of the route's fit (fit A)")
            sp.add_argument("--baseline", help="smplx dir of the one-camera fit of the same window (fit B)")
            sp.add_argument("--reference", help="smplx dir of the multi-camera fit (default work/<take>/smplx)")
    sp = sub.add_parser("atlas-test")
    sp.add_argument("take")
    sp.add_argument("--cam", required=True)
    sp.add_argument("--t", type=float, required=True, help="reference seconds of the input frame")
    sp.add_argument("--targets", required=True, help="other phones whose poses Atlas renders, comma-separated")
    sp.add_argument("--clean-plate", action="store_true", help="also try atlasMasked people removal")
    sp.add_argument("--no-jev", action="store_true")
    sp.add_argument("--allow-upload", action="store_true", help="the owner's go-ahead to send the frames to World Labs")
    sp.set_defaults(fn=cmd_atlas_test)
    sp = sub.add_parser("viggle-test")
    sp.add_argument("take")
    sp.add_argument("--cam", required=True)
    sp.add_argument("--t0", type=float, required=True)
    sp.add_argument("--t1", type=float, required=True)
    sp.add_argument("--fps", type=float, default=30.0)
    sp.add_argument("--reference", help="smplx dir to score against (default work/<take>/smplx)")
    sp.add_argument("--allow-upload", action="store_true", help="the owner's go-ahead to send the clips to Viggle")
    sp.set_defaults(fn=cmd_viggle_test)
    sp = sub.add_parser("judge-calibrate")
    sp.add_argument("--times", type=int, default=3)
    sp.add_argument("--no-source", action="store_true")
    sp.set_defaults(fn=cmd_calibrate)
    a = ap.parse_args(argv)
    t = time.time()
    rc = a.fn(a)
    print(f"[mono4d] {a.cmd} done in {time.time() - t:.1f} s", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
