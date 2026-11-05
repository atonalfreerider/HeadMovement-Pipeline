#!/usr/bin/env python
"""Pod side of the 4DAnyone test (dancecap.fda_test; job work/lucas_luiza/jobs/pod/lucas_luiza_fda.json). Runs in the
4DAnyone venv that install.sh builds, never in the image python.

    fda_pod.py patch                      # GVHMR tracker -> OUR dancer boxes when FDA_BOXES is set (install.sh runs it)
    fda_pod.py place-models               # the bundle's SMPLX_NEUTRAL.npz -> 4DAnyone's models/body_models/smplx/
    fda_pod.py run --inputs <dir> --out <dir> [--only <run,...>] [--views 12] [--reserve-min 6]
    fda_pod.py extract --inputs <dir> --out <dir>      # per run: fda.npz (GVHMR joints in camera + canonical frame, cameras)
    fda_pod.py detect --out <dir> [--every 4]          # per run: det.npz (YOLO11x-pose people in every generated view)

4DAnyone (ant-research/4DAnyone @ 6dcf626, Apache-2.0 code; Turbo LoRA CC BY-NC-SA, GVHMR research licence) turns ONE
video of ONE person into N generated target views on a ring around that person, conditioned on GVHMR's skeleton of that
person. Our dances are couples, so every input clip is run once PER DANCER: the GVHMR tracker (YOLOv8 + its own track
choice) is replaced by that dancer's boxes from our people stage (FDA_BOXES = boxes_<role>.npy, (121, 4) xyxy in the
input video's pixels). Everything else (BiRefNet foreground of the whole frame, framing, the camera ring, generation) is
4DAnyone's own, at its defaults except the view count. Integrity per run (a failure = "not run correctly", not "worse"):
the tracker patch was hit (the log line), 121 frames, every generated video 121 frames, GVHMR joints finite.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("FDA_ROOT", "/workspace/fda"))
REPO = ROOT / "4DAnyone"
WORKSPACE = Path(__file__).resolve().parents[3]
NJ = 22
PATCH_OLD = "            bbx_xyxy = tracker.get_one_track(video_path).float()\n"
PATCH_NEW = ("            _fda = os.environ.get(\"FDA_BOXES\")\n"
             "            if _fda:\n"
             "                import numpy as _np\n"
             "                bbx_xyxy = torch.from_numpy(_np.load(_fda).astype(\"float32\"))\n"
             "                print(f\"[fda] GVHMR boxes from {_fda} ({len(bbx_xyxy)} frames)\", flush=True)\n"
             "            else:\n"
             "                bbx_xyxy = tracker.get_one_track(video_path).float()\n")
HIT = "[fda] GVHMR boxes from"
# 4DAnyone renders each view's skeleton video one after another in numpy on one core (~45 s per view at 1080x1920,
# 18 views per 12-view run: 14 min before generation starts, session 20261105-124614). The same function, the same
# arguments, run in forked worker processes (CPU only: numpy + PyAV), serial again if the pool fails.
PAR_OLD_T = ("    target_paths = tuple(\n"
             "        render_skeleton(camera, skeleton_root / f\"{camera.camera_id:02d}.mp4\") for camera in raw_target_cameras\n"
             "    )\n")
PAR_NEW_T = ("    target_paths = _fda_parallel(render_skeleton, [(camera, skeleton_root / f\"{camera.camera_id:02d}.mp4\")\n"
             "                                                   for camera in raw_target_cameras])\n")
PAR_OLD_R = ("        rcp_paths = tuple(\n"
             "            render_skeleton(camera, rcp_root / f\"{camera.camera_id:02d}.mp4\") for camera in raw_rcp_cameras\n"
             "        )\n")
PAR_NEW_R = ("        rcp_paths = _fda_parallel(render_skeleton, [(camera, rcp_root / f\"{camera.camera_id:02d}.mp4\")\n"
             "                                                    for camera in raw_rcp_cameras])\n")
PAR_HELPER = '''

_FDA_RENDER = None


def _fda_call(job):
    return _FDA_RENDER(*job)


def _fda_parallel(fn, jobs):
    """[fda] the skeleton videos in forked processes (dancecap remote/cand_fda/fda_pod.py patch)."""
    global _FDA_RENDER
    import concurrent.futures
    import multiprocessing
    import os
    import time as _t
    t0 = _t.time()
    _FDA_RENDER = fn
    try:
        n = max(1, min(len(jobs), int(os.environ.get("FDA_RENDER_WORKERS", "16"))))
        with concurrent.futures.ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("fork")) as ex:
            out = tuple(ex.map(_fda_call, jobs))
        print(f"[fda] {len(jobs)} skeleton videos in {n} processes, {_t.time() - t0:.0f} s", flush=True)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[fda] parallel skeleton render failed ({e!r}); serial", flush=True)
        return tuple(fn(*j) for j in jobs)
'''


def log(msg: str) -> None:
    print(f"[fda_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_patch(args) -> None:
    p = REPO / "fdanyone" / "motion" / "gvhmr.py"
    s = p.read_text()
    if PATCH_NEW in s:
        log("tracker patch already applied")
        return
    if s.count(PATCH_OLD) != 1:
        raise SystemExit(f"patch target found {s.count(PATCH_OLD)} times in {p} (want 1)")
    p.write_text(s.replace(PATCH_OLD, PATCH_NEW))
    log(f"tracker patch applied to {p}")


def cmd_patch_render(args) -> None:
    p = REPO / "fdanyone" / "skeleton" / "pipeline.py"
    s = p.read_text()
    if "_fda_parallel(render_skeleton" in s:
        log("render patch already applied")
        return
    for old in (PAR_OLD_T, PAR_OLD_R):
        if s.count(old) != 1:
            raise SystemExit(f"render patch target found {s.count(old)} times in {p} (want 1)")
    s = s.replace(PAR_OLD_T, PAR_NEW_T).replace(PAR_OLD_R, PAR_NEW_R).rstrip("\n") + "\n" + PAR_HELPER
    p.write_text(s)
    log(f"render patch applied to {p}")


def cmd_place_models(args) -> None:
    src = WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"
    dst = REPO / "models" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not src.is_file():
        raise SystemExit(f"missing {src} (job models: neutral)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    z = np.load(src, allow_pickle=True)
    log(f"SMPL-X neutral linked: {dst} -> {src} (v_template {z['v_template'].shape}, shapedirs {z['shapedirs'].shape})")


def runs_of(inputs: Path, only: str | None) -> list[Path]:
    """Run input folders; with --only, in the order given (the job lists clip by clip, both cameras of a dancer together)."""
    rs = {p.name: p for p in sorted(inputs.iterdir()) if (p / "meta.json").exists()}
    if only:
        return [rs[x.strip()] for x in only.split(",") if x.strip() in rs]
    return list(rs.values())


def video_frames(path: Path) -> int:
    import av
    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(video=0))


def cmd_run(args) -> None:
    inputs, out = Path(args.inputs).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep_p = out / "run.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {"runs": {}}
    deadline = time.time() + 60 * args.budget_min if args.budget_min else None
    durations = [v["seconds"] for v in rep["runs"].values() if v.get("ok")]
    for d in runs_of(inputs, args.only):
        name = d.name
        if rep["runs"].get(name, {}).get("ok"):
            log(f"{name}: done before, skipped")
            continue
        if deadline and durations and time.time() + max(durations) * 1.2 + 60 * args.reserve_min > deadline:
            log(f"{name}: not started (budget: {len(durations)} runs done, {max(durations):.0f} s per run)")
            rep["runs"][name] = {"ok": False, "failed": "not started: time budget"}
            continue
        meta = json.loads((d / "meta.json").read_text())
        dst = out / name
        if dst.exists():
            subprocess.run(["rm", "-rf", str(dst)], check=True)
        env = dict(os.environ, FDA_BOXES=str(d / f"boxes_{meta['role']}.npy"))
        cmd = [sys.executable, "inference.py", "--video_path", str(d / "clip.mp4"), "--output_dir", str(dst),
               "--views_per_layer", str(args.views), "--seed", "42"]
        log(f"{name}: {' '.join(cmd[1:])}")
        t0 = time.time()
        logf = out / f"{name}.log"
        with open(logf, "w") as fh:
            p = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=60 * args.run_timeout_min)
        sec = time.time() - t0
        text = logf.read_text(errors="replace")
        r = {"seconds": round(sec, 1), "rc": p.returncode, "tracker_patch_hit": HIT in text}
        fails = []
        if p.returncode != 0:
            fails.append(f"rc {p.returncode}: " + " | ".join(text.strip().splitlines()[-3:])[-400:])
        if not r["tracker_patch_hit"]:
            fails.append("tracker patch not hit (GVHMR tracked its own person)")
        vids = sorted((dst / "videos").glob("*.mp4")) if dst.exists() else []
        if p.returncode == 0:
            if len(vids) != args.views:
                fails.append(f"{len(vids)} generated videos, want {args.views}")
            else:
                n = video_frames(vids[0])
                r["frames"] = n
                if n != 121:
                    fails.append(f"generated video has {n} frames, want 121")
            try:
                r["metadata"] = {k: v for k, v in json.loads((dst / "metadata.json").read_text()).items()
                                 if k in ("timings", "resources", "seconds", "timing")}
            except Exception as e:   # noqa: BLE001 - informational only
                r["metadata_error"] = str(e)[:200]
        r["ok"] = not fails
        if fails:
            r["failed"] = "; ".join(fails)
        else:
            durations.append(sec)
        rep["runs"][name] = r
        rep_p.write_text(json.dumps(rep, indent=1))
        log(f"{name}: {'ok' if r['ok'] else 'FAILED ' + r['failed']} in {sec:.0f} s")
    rep_p.write_text(json.dumps(rep, indent=1))
    bad = [k for k, v in rep["runs"].items() if not v.get("ok")]
    log(f"runs: {sum(v.get('ok', False) for v in rep['runs'].values())} ok, {len(bad)} not ({', '.join(bad)})")
    if args.require_all and bad:
        raise SystemExit(1)


def smplx_joints(params: dict, model) -> np.ndarray:
    import torch
    with torch.inference_mode():
        out = model(**{k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in params.items()})
    return out.joints[:, :NJ].numpy().astype(np.float64)


def cmd_extract(args) -> None:
    sys.path.insert(0, str(REPO))
    import smplx
    import torch
    from fdanyone.assets import MHR70_REGRESSOR
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry

    inputs, out = Path(args.inputs).resolve(), Path(args.out).resolve()
    rep = json.loads((out / "run.json").read_text())
    gv = REPO / "third_party" / "GVHMR"
    model = None
    for d in runs_of(inputs, None):
        name = d.name
        if not rep["runs"].get(name, {}).get("ok"):
            continue
        rd = out / name
        motion = MotionResult.load(rd / "gvhmr")
        T = motion.num_frames
        if model is None:
            model = smplx.create(str(WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"), model_type="smplx", gender="neutral",
                                 num_betas=10, use_pca=False, batch_size=T)
        Pg = {k: v.detach().cpu().numpy() for k, v in motion.smpl_params_global.items()}
        Pc = {k: v.detach().cpu().numpy() for k, v in motion.smpl_params_incam.items()}
        J_glob, J_cam = smplx_joints(Pg, model), smplx_joints(Pc, model)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        geo = _body_geometry(motion, REPO / "models" / MHR70_REGRESSOR, gv, dev)
        W = np.asarray(geo.motion_world_to_canonical_world, float)
        J_can = J_glob @ W[:3, :3].T + W[:3, 3]
        # check: the SMPL 24-joint skeleton 4DAnyone builds in the same canonical frame (body joints 0-21 ~ ours)
        chk = float(np.median(np.linalg.norm(J_can[:, [0, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19]]
                                             - np.asarray(geo.joints_world)[:, [0, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19]],
                                             axis=-1)) * 1000)
        cams = json.loads((rd / "cameras.json").read_text())
        cs = cams["cameras"]
        np.savez_compressed(
            rd / "fda.npz", J_cam=J_cam, J_canon=J_can, K_in=motion.K_fullimg.detach().cpu().numpy().astype(float),
            cam_K=np.array([c["K"] for c in cs], float), cam_c2w=np.array([c["camera_to_world"] for c in cs], float),
            cam_wh=np.array([[c["image_width"], c["image_height"]] for c in cs]), cam_yaw=np.array([c["yaw"] for c in cs]),
            cam_pitch=np.array([c["pitch"] for c in cs]), front_ids=np.array(cams.get("front_camera_ids", []), int),
            kp2d_vitpose=motion.observed_keypoints_2d.detach().cpu().numpy())
        ok = bool(np.isfinite(J_cam).all() and np.isfinite(J_can).all())
        rep["runs"][name]["extract"] = {"ok": ok, "canon_vs_fda_smpl_mm": round(chk, 1), "views": len(cs),
                                        "framing": cams.get("framing")}
        if not ok:
            rep["runs"][name].update(ok=False, failed="non-finite GVHMR joints")
        log(f"{name}: joints extracted (canonical vs 4DAnyone's SMPL skeleton {chk:.1f} mm median)")
    (out / "run.json").write_text(json.dumps(rep, indent=1, default=float))


def cmd_detect(args) -> None:
    import av
    from ultralytics import YOLO
    out = Path(args.out).resolve()
    rep = json.loads((out / "run.json").read_text())
    yolo = YOLO(args.weights)
    P = 6
    for name, r in sorted(rep["runs"].items()):
        if not r.get("ok"):
            continue
        rd = out / name
        vids = sorted((rd / "videos").glob("*.mp4"))
        fr = np.arange(0, 121, args.every)
        kp = np.full((len(vids), len(fr), P, 17, 3), np.nan, np.float32)
        box = np.full((len(vids), len(fr), P, 5), np.nan, np.float32)
        for v, vp in enumerate(vids):
            with av.open(str(vp)) as c:
                imgs = [f.to_ndarray(format="bgr24") for i, f in enumerate(c.decode(video=0)) if i in set(fr.tolist())]
            res = yolo.predict(imgs, conf=0.25, verbose=False, half=True)
            for j, rr in enumerate(res):
                if rr.keypoints is None or len(rr.boxes) == 0:
                    continue
                n = min(P, len(rr.boxes))
                o = np.argsort(-rr.boxes.conf.cpu().numpy())[:n]
                kp[v, j, :n] = rr.keypoints.data.cpu().numpy()[o]
                box[v, j, :n, :4] = rr.boxes.xyxy.cpu().numpy()[o]
                box[v, j, :n, 4] = rr.boxes.conf.cpu().numpy()[o]
        np.savez_compressed(rd / "det.npz", kp=kp, box=box, frames=fr, videos=np.array([p.name for p in vids]))
        n05 = np.nansum(box[..., 4] >= 0.5, axis=-1)
        r["detect"] = {"weights": args.weights, "frames": int(len(fr)), "people_ge05_median": float(np.median(n05))}
        log(f"{name}: detected {len(vids)} views x {len(fr)} frames, median people (conf >= 0.5) {np.median(n05):.0f}")
    (out / "run.json").write_text(json.dumps(rep, indent=1, default=float))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("patch")
    sub.add_parser("patch-render")
    sub.add_parser("place-models")
    p = sub.add_parser("run")
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only")
    p.add_argument("--views", type=int, default=12)
    p.add_argument("--run-timeout-min", type=float, default=20)
    p.add_argument("--budget-min", type=float, default=0, help="stop starting new runs this many minutes after start")
    p.add_argument("--reserve-min", type=float, default=6)
    p.add_argument("--require-all", action="store_true")
    p = sub.add_parser("extract")
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("detect")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--weights", default="yolo11x-pose.pt")
    a = ap.parse_args(argv)
    {"patch": cmd_patch, "patch-render": cmd_patch_render, "place-models": cmd_place_models, "run": cmd_run, "extract": cmd_extract,
     "detect": cmd_detect}[a.cmd](a)


if __name__ == "__main__":
    main()
