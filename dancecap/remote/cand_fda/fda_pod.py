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


def run_one(d: Path, out: Path, views: int, timeout_min: float) -> dict:
    """One 4DAnyone run of input folder d into out/<d.name>. Variant "ours" (round 2 B): out/<name>/gvhmr already holds
    the injected motion (4DAnyone reuses it: no GVHMR, so no tracker patch), and view.json sets the ring's start yaw and
    pitch so view 0 faces the held-out camera."""
    name = d.name
    meta = json.loads((d / "meta.json").read_text())
    ours = meta.get("variant") == "ours"
    dst = out / name
    if dst.exists() and not ours:
        subprocess.run(["rm", "-rf", str(dst)], check=True)
    env = dict(os.environ, FDA_BOXES=str(d / f"boxes_{meta['role']}.npy"))
    cmd = [sys.executable, "inference.py", "--video_path", str(d / "clip.mp4"), "--output_dir", str(dst),
           "--views_per_layer", str(views), "--seed", "42"]
    if (d / "view.json").exists():
        v = json.loads((d / "view.json").read_text())
        cmd += ["--start_yaw", str(int(v["start_yaw"])), "--layer_pitches", f"[{int(v['pitch'])}]"]
    log(f"{name}: {' '.join(cmd[1:])}")
    t0 = time.time()
    logf = out / f"{name}.log"
    with open(logf, "w") as fh:
        p = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=60 * timeout_min)
    sec = time.time() - t0
    text = logf.read_text(errors="replace")
    r = {"seconds": round(sec, 1), "rc": p.returncode, "tracker_patch_hit": HIT in text, "variant": meta.get("variant")}
    fails = []
    if p.returncode != 0:
        fails.append(f"rc {p.returncode}: " + " | ".join(text.strip().splitlines()[-3:])[-400:])
    if ours:
        if "Reusing GVHMR motion" not in text:
            fails.append("the injected motion was not reused")
    elif not r["tracker_patch_hit"]:
        fails.append("tracker patch not hit (GVHMR tracked its own person)")
    vids = sorted((dst / "videos").glob("*.mp4")) if dst.exists() else []
    if p.returncode == 0:
        if len(vids) != views:
            fails.append(f"{len(vids)} generated videos, want {views}")
        else:
            n = video_frames(vids[0])
            r["frames"] = n
            if n != 121:
                fails.append(f"generated video has {n} frames, want 121")
    r["ok"] = not fails
    if fails:
        r["failed"] = "; ".join(fails)
    log(f"{name}: {'ok' if r['ok'] else 'FAILED ' + r['failed']} in {sec:.0f} s")
    return r


def _report(out: Path) -> tuple[Path, dict]:
    rep_p = out / "run.json"
    return rep_p, (json.loads(rep_p.read_text()) if rep_p.exists() else {"runs": {}})


def cmd_run(args) -> None:
    inputs, out = Path(args.inputs).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep_p, rep = _report(out)
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
        r = run_one(d, out, args.views, args.run_timeout_min)
        if r["ok"]:
            durations.append(r["seconds"])
        rep["runs"][name] = r
        rep_p.write_text(json.dumps(rep, indent=1))
    rep_p.write_text(json.dumps(rep, indent=1))
    bad = [k for k, v in rep["runs"].items() if not v.get("ok")]
    log(f"runs: {sum(v.get('ok', False) for v in rep['runs'].values())} ok, {len(bad)} not ({', '.join(bad)})")
    if args.require_all and bad:
        raise SystemExit(1)


def cmd_pipeline(args) -> None:
    """Round 2: per input, A (masked clip, its own GVHMR) -> inject our fit as B's motion -> B. Pairs finish together, so
    the time budget never leaves an A without its B."""
    inputs, out = Path(args.inputs).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep_p, rep = _report(out)
    deadline = time.time() + 60 * args.budget_min if args.budget_min else None
    durs = []
    for d in runs_of(inputs, args.only):
        name = d.name
        if deadline and durs and time.time() + max(durs) * 1.15 + 60 * args.reserve_min > deadline:
            log(f"{name}: pair not started (budget: {max(durs):.0f} s per pair)")
            rep["runs"][name] = rep["runs"][name + "_ours"] = {"ok": False, "failed": "not started: time budget"}
            continue
        t0 = time.time()
        rA = rep["runs"].get(name) if rep["runs"].get(name, {}).get("ok") else run_one(d, out, args.views, args.run_timeout_min)
        rep["runs"][name] = rA
        rep_p.write_text(json.dumps(rep, indent=1))
        if not rA["ok"]:
            rep["runs"][name + "_ours"] = {"ok": False, "failed": "A failed"}
            continue
        try:
            inj = inject(d, out / name, inputs, out)
        except Exception as e:   # noqa: BLE001 - reported, the next pair still runs
            import traceback
            traceback.print_exc()
            rep["runs"][name + "_ours"] = {"ok": False, "failed": f"inject: {e!r}"[:400]}
            rep_p.write_text(json.dumps(rep, indent=1))
            continue
        rB = run_one(inputs / (name + "_ours"), out, args.views, args.run_timeout_min)
        rB["inject"] = inj
        rep["runs"][name + "_ours"] = rB
        durs.append(time.time() - t0)
        rep_p.write_text(json.dumps(rep, indent=1))
    bad = [k for k, v in rep["runs"].items() if not v.get("ok")]
    log(f"runs: {sum(v.get('ok', False) for v in rep['runs'].values())} ok, {len(bad)} not ({', '.join(bad)})")


def _aa_to_mat(aa: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(aa).as_matrix()


def _mat_to_aa(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def inject(d: Path, a_dir: Path, inputs: Path, out: Path) -> dict:
    """B's motion = A's motion (same clip identity, A's ViTPose 2D) with OUR two-view fit: 10 betas + per-frame translation
    fitted to the fit's 22 body joints (our fit has 16 betas on MAMMA's locked-head SMPL-X; GVHMR's model takes 10), the
    fit's pose; incam through our camera (x_c = R X + t), K = ours in the input clip's pixels. Then view 0's yaw and the
    ring pitch toward the held-out camera, by 4DAnyone's own camera_ring formula in its canonical frame."""
    sys.path.insert(0, str(REPO))
    import dataclasses
    import shutil
    import smplx
    import torch
    from fdanyone.assets import MHR70_REGRESSOR
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry, _front_direction
    name = d.name
    op = dict(np.load(d / "ourpose.npz"))
    mot = MotionResult.load(a_dir / "gvhmr")
    T = mot.num_frames
    model = smplx.create(str(WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"), model_type="smplx", gender="neutral",
                         num_betas=10, use_pca=False, batch_size=T)
    tt = lambda x: torch.as_tensor(np.asarray(x, np.float32))  # noqa: E731
    go, bp, Jt = tt(op["global_orient"]), tt(op["body_pose"]), tt(op["joints"][:, :NJ])
    betas = torch.nn.Parameter(tt(op["betas"][:10])[None].clone())
    transl = torch.nn.Parameter(tt(op["transl"]).clone())
    opt = torch.optim.Adam([betas, transl], lr=0.02)
    for it in range(400):
        opt.zero_grad()
        J = model(global_orient=go, body_pose=bp, betas=betas.expand(T, -1), transl=transl).joints[:, :NJ]
        loss = ((J - Jt) ** 2).sum(-1).mean() + 1e-4 * (betas ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        J = model(global_orient=go, body_pose=bp, betas=betas.expand(T, -1), transl=transl).joints[:, :NJ]
        resid = float((J - Jt).norm(dim=-1).mean() * 1000)
        zero = model(global_orient=torch.zeros(T, 3), body_pose=torch.zeros(T, 63), betas=betas.expand(T, -1),
                     transl=torch.zeros(T, 3)).joints[0, 0].numpy().astype(float)
    b10 = betas.detach().numpy()[0].astype(float)
    tr = transl.detach().numpy().astype(float)
    R, t = op["R"].astype(float), op["t"].astype(float)
    Rg = _aa_to_mat(op["global_orient"].astype(float))
    orient_c = _mat_to_aa(np.einsum("fij,fjk->fik", R, Rg))
    transl_c = np.einsum("fij,fj->fi", R, tr + zero) + t - zero
    s, ox, oy = op["crop"]
    K = op["K"].astype(float).copy()
    K[:, 0, 0] *= s
    K[:, 1, 1] *= s
    K[:, 0, 2] = (K[:, 0, 2] - ox) * s
    K[:, 1, 2] = (K[:, 1, 2] - oy) * s
    f32 = lambda x: torch.as_tensor(np.asarray(x, np.float32))  # noqa: E731
    bt = np.repeat(b10[None], T, 0)
    glob = {"body_pose": f32(op["body_pose"]), "betas": f32(bt), "global_orient": f32(op["global_orient"]), "transl": f32(tr)}
    inc = {"body_pose": f32(op["body_pose"]), "betas": f32(bt), "global_orient": f32(orient_c), "transl": f32(transl_c)}
    new = dataclasses.replace(mot, smpl_params_global=glob, smpl_params_incam=inc, K_fullimg=f32(K))
    new.validate()
    # check: our incam joints project onto A's ViTPose 2D (the dancer the clip shows), median px
    with torch.no_grad():
        Jc = model(**inc).joints[:, :NJ].numpy().astype(float)
    uv = np.einsum("fij,fnj->fni", K, Jc)
    uv = uv[..., :2] / uv[..., 2:]
    kp = mot.observed_keypoints_2d.numpy()
    coco = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
    ci, si = list(coco), list(coco.values())
    good = kp[:, ci, 2] >= 0.5
    reproj = float(np.median(np.linalg.norm(uv[:, si] - kp[:, ci, :2], axis=-1)[good])) if good.any() else None
    # B's output and input folders
    b_out, b_in = out / (name + "_ours"), inputs / (name + "_ours")
    for p in (b_out, b_in):
        if p.exists():
            shutil.rmtree(p)
    b_out.mkdir(parents=True)
    new.save(b_out / "gvhmr")
    req = [x for x in a_dir.iterdir() if x.name.startswith(".") and x.suffix == ".json" and x.is_file()]
    for x in req:
        shutil.copy2(x, b_out / x.name)
    b_in.mkdir(parents=True)
    (b_in / "clip.mp4").symlink_to((d / "clip.mp4").resolve())
    shutil.copy2(d / f"boxes_{json.loads((d / 'meta.json').read_text())['role']}.npy", b_in)
    meta = json.loads((d / "meta.json").read_text())
    meta["variant"] = "ours"
    (b_in / "meta.json").write_text(json.dumps(meta, indent=1))
    # aim view 0 at the held-out camera: camera_ring puts camera k at azimuth front_az + pi + start_yaw (+ k * span / n),
    # position = centre + r (cos az, ., sin az), in the canonical frame of _body_geometry
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geo = _body_geometry(new, REPO / "models" / MHR70_REGRESSOR, REPO / "third_party" / "GVHMR", dev)
    Wc = np.asarray(geo.motion_world_to_canonical_world, float)
    targets = np.asarray(geo.vertices_world).mean(axis=1)
    centre = targets.mean(axis=0)
    front = _front_direction(np.asarray(geo.joints_world))
    front_az = float(np.arctan2(front[2], front[0]))
    hc = op["held_centre"][T // 2] @ Wc[:3, :3].T + Wc[:3, 3]
    az = float(np.arctan2(hc[2] - centre[2], hc[0] - centre[0]))
    start_yaw = int(round(np.degrees(az - front_az - np.pi))) % 360
    dist = float(np.hypot(hc[0] - centre[0], hc[2] - centre[2]))
    pitch = int(np.clip(round(np.degrees(np.arctan2(hc[1] - 1.0, dist))), -15, 45))
    view = {"start_yaw": start_yaw, "pitch": pitch, "held_centre_canon": hc.tolist(), "centre": centre.tolist(),
            "held_dist_m": dist, "world_to_canon": Wc.tolist()}
    (b_in / "view.json").write_text(json.dumps(view, indent=1))
    info = {"betas10_joint_resid_mm": round(resid, 1), "reproj_vs_vitpose_px_median": reproj, "start_yaw": start_yaw,
            "pitch": pitch, "held_dist_m": round(dist, 2), "fit_dt_ms_max": float(np.abs(op["dt_ms"]).max())}
    log(f"{name}: injected our fit (betas10 joint residual {resid:.1f} mm, reprojection vs ViTPose {reproj} px); "
        f"view 0 -> held-out camera: start_yaw {start_yaw}, pitch {pitch}")
    return info


def cmd_appearance(args) -> None:
    """B runs: view 0 (aimed at the held-out camera) vs that camera's real frames of the dancer. Per detected frame: the
    generated person's box -> the same 256 x 512 letterboxed canvas as the real crop; LPIPS (alex) and colour-matched PSNR
    inside the REAL mask; the angle between view 0's direction and the held-out camera's, both from the body centre."""
    import av
    import cv2
    import lpips
    import torch
    inputs, out, held = Path(args.inputs).resolve(), Path(args.out).resolve(), Path(args.heldout).resolve()
    rep_p, rep = _report(out)
    net = lpips.LPIPS(net="alex", verbose=False).eval()
    W, H = 256, 512
    for name, r in sorted(rep["runs"].items()):
        if not name.endswith("_ours") or not r.get("ok") or not (out / name / "det.npz").exists():
            continue
        a = name[:-5]
        hz = np.load(held / f"{a}.npz")
        det = np.load(out / name / "det.npz")
        fda = np.load(out / name / "fda.npz")
        view = json.loads((inputs / name / "view.json").read_text())
        Wc = np.array(view["world_to_canon"])
        hc_w = np.load(inputs / a / "ourpose.npz")["held_centre"]
        hc = hc_w @ Wc[:3, :3].T + Wc[:3, 3]
        c0 = fda["cam_c2w"][0][:3, 3]
        body = fda["J_canon"][:, 0]
        with av.open(str(sorted((out / name / "videos").glob("*.mp4"))[0])) as c:
            gen = {i: f.to_ndarray(format="bgr24") for i, f in enumerate(c.decode(video=0)) if i in set(det["frames"].tolist())}
        rows = []
        for j, f in enumerate(det["frames"]):
            f = int(f)
            ang = float(np.degrees(np.arccos(np.clip(np.dot(
                (c0 - body[f])[[0, 2]] / np.linalg.norm((c0 - body[f])[[0, 2]]),
                (hc[f] - body[f])[[0, 2]] / np.linalg.norm((hc[f] - body[f])[[0, 2]])), -1, 1))))
            hi = int(hz["j"][f])
            if not hz["ok"][hi]:
                continue
            bx = det["box"][0, j]
            k = int(np.nanargmax(np.nan_to_num(bx[:, 4], nan=-1)))
            if not np.isfinite(bx[k, 4]) or bx[k, 4] < 0.25:
                rows.append({"frame": f, "angle": ang, "found": False})
                continue
            x0, y0, x1, y1 = bx[k, :4]
            mx, my = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
            x0, y0, x1, y1 = x0 - mx, y0 - my, x1 + mx, y1 + my
            s = min(W / (x1 - x0), H / (y1 - y0))
            A = np.array([[s, 0, W / 2 - s * (x0 + x1) / 2], [0, s, H / 2 - s * (y0 + y1) / 2]], np.float32)
            g = cv2.warpAffine(gen[f], A, (W, H), flags=cv2.INTER_AREA)
            real, m = hz["crops"][hi], hz["masks"][hi]
            g[~m] = 0
            gm, rm = g[m].astype(float), real[m].astype(float)
            gcc = (gm - gm.mean(0)) / (gm.std(0) + 1e-6) * rm.std(0) + rm.mean(0)
            psnr = float(10 * np.log10(255 ** 2 / max(((gcc - rm) ** 2).mean(), 1e-6)))
            tg = lambda im: torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 127.5 - 1  # noqa: E731
            with torch.no_grad():
                lp = float(net(tg(g), tg(real)).item())
            rows.append({"frame": f, "angle": ang, "found": True, "lpips": lp, "psnr_cc": psnr})
            if j in (0, len(det["frames"]) // 2):
                cv2.imwrite(str(out / name / f"held_pair_{f:03d}.jpg"), np.concatenate([real, g], 1))
        sel = [x for x in rows if x.get("found") and x["angle"] < args.max_deg]
        summ = {"n": len(sel), "n_all": len(rows), "found_share": float(np.mean([x.get("found", False) for x in rows])) if rows else None,
                "angle_deg": float(np.median([x["angle"] for x in rows])) if rows else None,
                "lpips": float(np.median([x["lpips"] for x in sel])) if sel else None,
                "psnr_cc": float(np.median([x["psnr_cc"] for x in sel])) if sel else None}
        (out / name / "appearance.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
        r["appearance"] = summ
        log(f"{name}: view 0 vs held-out camera: {summ}")
    rep_p.write_text(json.dumps(rep, indent=1, default=float))


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
    p = sub.add_parser("pipeline")
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only")
    p.add_argument("--views", type=int, default=6)
    p.add_argument("--run-timeout-min", type=float, default=20)
    p.add_argument("--budget-min", type=float, default=0)
    p.add_argument("--reserve-min", type=float, default=6)
    p = sub.add_parser("appearance")
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--max-deg", type=float, default=20.0)
    p = sub.add_parser("extract")
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("detect")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--weights", default="yolo11x-pose.pt")
    a = ap.parse_args(argv)
    {"patch": cmd_patch, "patch-render": cmd_patch_render, "place-models": cmd_place_models, "run": cmd_run, "extract": cmd_extract,
     "detect": cmd_detect, "pipeline": cmd_pipeline, "appearance": cmd_appearance}[a.cmd](a)


if __name__ == "__main__":
    main()
