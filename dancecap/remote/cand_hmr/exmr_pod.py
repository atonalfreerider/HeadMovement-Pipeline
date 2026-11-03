#!/usr/bin/env python
"""Pod side of the ExMR test (dancecap.hmr_test; job work/lucas_luiza/jobs/pod/lucas_luiza_exmr.json). Runs in the ExMR
venv that exmr_install.sh builds (python 3.12, torch 2.8.0+cu126), never in the image python.

    exmr_pod.py place-models                       # SMPL-X + MANO from the bundle into checkpoints/ (sha256-checked)
    exmr_pod.py run --smoke --clips <dir> --only E1 --roles lead --frames 60 --out <dir>       # the smoke step
    exmr_pod.py run --clips <dir> --roles lead,follow --mode balanced --smoothing none --no_render --out <dir>

ExMR (twehrbein/ExMR @ 91191a1, demo code) reconstructs ONE person per video. Per clip x role (one output directory
each: the cache keys hold no boxes, so a shared directory would hand the follow the lead's keypoints and features) this
script pre-writes ExMR's own caches (exmr/pipeline.py _cached; check error 7) instead of patching the pipeline:
  - cache/tracks.pt            {settings {video, frames, detector "yolo11m.pt"}, value: OUR teacher boxes (the people
                               stage's, lucas_luiza_h/people/teachers_01.npz) with ExMR's own 5-frame moving average twice}
  - cache/camera_rotations.pt  {settings {video, frames, odometry "simplevo", f_mm}, value: OUR world-to-camera R of
                               camera 02 per frame (relative to the first frame, as SimpleVO's; only R[t+1] R[t]^T
                               reaches the network)}
and passes the clip's camera-02 focal as ExMR's --f_mm (f_mm = median f02 * 43.267 / 1468.6: K02 varies by <= 0.4 %
inside a clip; ExMR builds K = estimate_K(w, h, f_mm) for every frame). The tracker (YOLO11, AGPL; an empty placeholder
satisfies Assets.check) and SimpleVO are replaced by stubs that RAISE, so a silent fallback fails loudly.
Integrity per run (a failure = "not run correctly", not "worse"): tracks.pt and camera_rotations.pt read from OUR cache
exactly once and never computed, keypoints / body / hand features computed once; K_fullimg within 0.5 % of K02 (focal)
and 1 px (principal point); median IoU of ExMR's boxes vs our raw boxes >= 0.9; every output frame finite and L long;
--smoke also: joints_2d vs our tripod teacher kp <= 10 % of the box height, and the rotation convention against SimpleVO
when the camera pans more than 1 degree. Writes <out>/<clip>_<role>/results.npz (ExMR's) and <out>/run.json.
Never prints a token: the gated SAM 3D Body checkpoint was downloaded by exmr_install.sh with HF_TOKEN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("EXMR_ROOT", "/workspace/exmr"))
REPO = ROOT / "ExMR"
CKPT = Path(os.environ.get("EXMR_CHECKPOINT_DIR", REPO / "checkpoints"))
WORKSPACE = Path(__file__).resolve().parents[3]
ROLES = ("lead", "follow")
SMPLX_SHA256 = "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"   # = ExMR's pin
DIAG_MM = float(np.hypot(36.0, 24.0))
DETECTOR = "yolo11m"
K_F_TOL, K_PP_TOL_PX, IOU_MIN, KP2D_MAX_PCT, PAN_MIN_DEG = 0.005, 1.0, 0.9, 10.0, 1.0
# COCO-17 index -> SMPL-X joint (the first 55 of ExMR's 76 joints), as dancecap.carlos_aline_fitcheck.COCO_TO_SMPLX
COCO_TO_SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
CACHES_OURS = ("tracks.pt", "camera_rotations.pt")
CACHES_EXMR = ("keypoints.pt", "body_features.pt", "hand_features.pt")


def log(msg: str) -> None:
    print(f"[exmr_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ pure helpers


def f_mm_of(f_px: float, width: int, height: int) -> float:
    """ExMR's --f_mm whose estimate_K(width, height, f_mm) has focal f_px."""
    return float(f_px) * DIAG_MM / float(np.hypot(width, height))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise IoU of (N, 4) xyxy boxes."""
    x0, y0 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x1, y1 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda c: np.clip(c[:, 2] - c[:, 0], 0, None) * np.clip(c[:, 3] - c[:, 1], 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        return inter / (area(a) + area(b) - inter)


def relative_rotations(R: np.ndarray) -> np.ndarray:
    """World-to-camera rotations (L, 3, 3) -> relative to the first frame (R[t] R[0]^T), SimpleVO's convention (the
    first one identity). The network only sees R[t+1] R[t]^T, which this leaves unchanged."""
    return np.einsum("lij,kj->lik", R, R[0])


def rot_angle_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Angle (degrees) of A B^T per frame."""
    tr = np.einsum("lij,lij->l", A, B)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def k_check(K_exmr: np.ndarray, K02: np.ndarray) -> dict:
    """ExMR's K_fullimg against our K02, per frame: focal within K_F_TOL, principal point within K_PP_TOL_PX."""
    df = np.abs(K_exmr[:, 0, 0] / K02[:, 0, 0] - 1)
    dpp = np.abs(K_exmr[:, :2, 2] - K02[:, :2, 2]).max(1)
    return {"focal_rel_max": float(df.max()), "pp_px_max": float(dpp.max()),
            "ok": bool(df.max() <= K_F_TOL and dpp.max() <= K_PP_TOL_PX)}


def kp2d_check(joints_2d: np.ndarray, kp: np.ndarray, box: np.ndarray) -> dict:
    """ExMR's joints_2d (L, >=55, 2) vs our tripod teacher kp (L, 17, 3): per frame the median px error over the COCO
    joints with conf >= 0.5 (at least 6) / the box height; the median over frames in %."""
    coco = sorted(COCO_TO_SMPLX)
    smx = [COCO_TO_SMPLX[c] for c in coco]
    v = []
    for f in range(len(kp)):
        c = kp[f, coco, 2] >= 0.5
        if c.sum() < 6:
            continue
        e = np.linalg.norm(joints_2d[f, smx][c] - kp[f, coco][c, :2], axis=1)
        v.append(100 * np.median(e) / max(float(box[f, 3] - box[f, 1]), 50.0))
    med = float(np.median(v)) if v else float("nan")
    return {"median_pct": med, "frames": len(v), "ok": bool(np.isfinite(med) and med <= KP2D_MAX_PCT)}


def cache_check(calls: dict) -> dict:
    """calls: {cache file: ["hit" | "computed", ...]} recorded around exmr.pipeline._cached during one run."""
    bad = [f"{c}: {calls.get(c)}" for c in CACHES_OURS if calls.get(c) != ["hit"]]
    bad += [f"{c}: {calls.get(c)}" for c in CACHES_EXMR if len(calls.get(c) or []) != 1]
    return {"calls": calls, "ok": not bad, "bad": bad}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ place-models


def cmd_place_models(args) -> None:
    """SMPL-X (bundle: smplx/SMPLX_NEUTRAL.npz) and MANO (bundle: models/mano/MANO_RIGHT.pkl) into checkpoints/. The
    MANO pickle holds chumpy objects that ExMR's numpy-2 env cannot load: ExMR's own downloader converts it from a zip
    (models/MANO_RIGHT.pkl) and checks its arrays against MANO v1.2 (check error 6). Bundle files arrive after the early
    install, hence a command of its own (check error 4)."""
    src = WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"
    dst = CKPT / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    got = sha256_file(dst)
    if got != SMPLX_SHA256:
        raise SystemExit(f"{dst}: sha256 {got[:12]} is not ExMR's SMPL-X ({SMPLX_SHA256[:12]})")
    mano = WORKSPACE / "models" / "mano" / "MANO_RIGHT.pkl"
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        z = Path(tmp) / "mano_v1_2_right.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.write(mano, "models/MANO_RIGHT.pkl")
        r = subprocess.run([sys.executable, "tools/download_models.py", "--checkpoint_dir", str(CKPT), "--only", "smplx",
                            "mano", "--mano_zip", str(z), "--verify"], cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        raise SystemExit(f"download_models.py --only smplx mano failed (rc {r.returncode})")
    yolo = CKPT / "yolo" / f"{DETECTOR}.pt"
    if not yolo.exists():
        yolo.parent.mkdir(parents=True, exist_ok=True)
        yolo.write_bytes(b"")   # placeholder for Assets.check: the tracker is a stub that raises
    log(f"models in place: SMPL-X {got[:12]}, MANO converted and checked, {yolo.name} placeholder")


# ------------------------------------------------------------------------------------------------ run


def install_guards(calls: dict) -> None:
    """Stub the tracker module (never imports ultralytics), make SimpleVO's background start raise, and record every
    cache read / compute of exmr.pipeline._cached."""
    import exmr.pipeline as P

    class Tracker:
        def __init__(self, *a, **k):
            raise RuntimeError("exmr_pod: ExMR's own tracker must not run (our boxes are in cache/tracks.pt)")

    stub = types.ModuleType("exmr.preprocessing.tracker")
    stub.Tracker = Tracker
    sys.modules["exmr.preprocessing.tracker"] = stub

    def no_background(fn, *a, **k):
        raise RuntimeError(f"exmr_pod: {getattr(fn, '__name__', fn)} must not run (our rotations are in "
                           "cache/camera_rotations.pt)")
    P._in_background = no_background
    orig = getattr(P, "_exmr_pod_orig_cached", P._cached)
    P._exmr_pod_orig_cached = orig

    def cached(path, settings, compute):
        calls.setdefault(Path(path).name, []).append("hit" if P._is_cached(Path(path), settings) else "computed")
        return orig(path, settings, compute)
    P._cached = cached


def smooth_boxes(box: np.ndarray):
    """ExMR's own track smoothing (tracker.get_main_track): moving average of 5 frames, applied twice."""
    import torch
    from exmr.geometry.smoothing import moving_average_smooth
    b = torch.from_numpy(np.asarray(box, np.float32))
    return moving_average_smooth(moving_average_smooth(b, window_size=5, dim=0), window_size=5, dim=0).float()


def write_caches(out: Path, video: Path, L: int, boxes_raw: np.ndarray, R_w2c: np.ndarray, f_mm: float, vo: str) -> None:
    import torch
    from exmr.pipeline import _atomic_save
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tracker_settings = {"video": str(video), "frames": int(L), "detector": f"{DETECTOR}.pt"}
    _atomic_save({"settings": tracker_settings, "value": smooth_boxes(boxes_raw)}, cache / "tracks.pt")
    vo_settings = {"video": str(video), "frames": int(L), "odometry": vo, "f_mm": f_mm}
    _atomic_save({"settings": vo_settings, "value": torch.from_numpy(relative_rotations(R_w2c).astype(np.float64))},
                 cache / "camera_rotations.pt")


def tidy_outputs(out: Path, keep: Path) -> list[str]:
    """After every run, success or failure (review finding 4): ExMR's extracted JPEG frames (out/frames, ~60-100 MB a
    run; ExMR removes them only at the end of a successful run), a CFR rebuild of the video (out/*_cfr.mp4) and
    results.pt (the same values as results.npz) are deleted; the caches (tracks, rotations, keypoints, features) move to
    keep, on the pod, out of the download. Never raises (it runs in a finally): a step that fails is logged and skipped.
    -> what it did."""
    done = []
    for p in [out / "frames", *sorted(out.glob("*_cfr.mp4")), out / "results.pt"]:
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            elif p.exists() or p.is_symlink():
                p.unlink()
            else:
                continue
            done.append(f"removed {p.name}")
        except OSError as e:
            log(f"tidy: could not remove {p}: {e}")
    if (out / "cache").exists():
        try:
            if keep.exists():
                shutil.rmtree(keep)
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out / "cache"), str(keep))
            done.append(f"cache -> {keep}")
        except OSError as e:
            log(f"tidy: could not move {out / 'cache'} to {keep}: {e}")
    return done


def ffmpeg(args: list, what: str) -> None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *map(str, args)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what}: ffmpeg rc {r.returncode}: {r.stderr[-600:]}")


def convention_verdict(direct_deg: float, transposed_deg: float) -> bool:
    """False only when SimpleVO clearly agrees with the TRANSPOSE of our rotations (the convention read from the pinned
    visual_odometry.py - world-to-camera, R[t+1] R[t]^T - would then be wrong); SimpleVO's own noise never fails it."""
    return not (transposed_deg < 0.5 * direct_deg and direct_deg > 1.0)


def convention_check(video: Path, R_rel: np.ndarray, f_mm: float) -> dict:
    """SimpleVO (ExMR's own, CPU) on the clip vs our relative rotations: which of R or R^T it agrees with."""
    from exmr.preprocessing.visual_odometry import SimpleVO
    pan = float(rot_angle_deg(R_rel, np.repeat(np.eye(3)[None], len(R_rel), 0)).max())
    if pan < PAN_MIN_DEG:
        return {"pan_deg": pan, "ok": True, "note": f"pans < {PAN_MIN_DEG} deg: nothing to compare"}
    vo = SimpleVO(f_mm=f_mm, progress=False)(video).numpy()
    direct = float(np.median(rot_angle_deg(vo, R_rel)))
    transposed = float(np.median(rot_angle_deg(vo, np.transpose(R_rel, (0, 2, 1)))))
    return {"pan_deg": pan, "simplevo_vs_ours_deg": direct, "simplevo_vs_ours_transposed_deg": transposed,
            "ok": convention_verdict(direct, transposed)}


def run_one(args, clip_dir: Path, role: str, out_root: Path, preset: dict) -> dict:
    from exmr.assets import Assets
    from exmr.pipeline import Options, run_video
    from exmr.preprocessing import video as V
    meta = json.loads((clip_dir / "meta.json").read_text())
    clip = meta["clip"]
    bz, cz = np.load(clip_dir / "boxes.npz"), np.load(clip_dir / "cam02.npz")
    L = int(meta["L"]) if not args.frames else min(int(args.frames), int(meta["L"]))
    name = f"{clip}_{role}"
    out = out_root / name
    runs = ROOT / "runs" / out_root.name
    runs.mkdir(parents=True, exist_ok=True)
    video = runs / f"{name}.mp4"            # outside the outputs (no link in the packed results); a unique stem per run
    if L < int(meta["L"]):                  # smoke: the first L frames, re-encoded losslessly
        if not video.exists():
            ffmpeg(["-i", clip_dir / "clip.mp4", "-frames:v", L, "-an", "-c:v", "libx264", "-preset", "ultrafast",
                    "-qp", 0, "-pix_fmt", "yuv420p", video], "smoke cut")
    else:
        if video.is_symlink() or video.exists():
            video.unlink()
        video.symlink_to((clip_dir / "clip.mp4").resolve())
    n_container = V.video_lwh(video)[0]
    if n_container != L:
        raise RuntimeError(f"{name}: the video has {n_container} frames, expected {L}")
    box_raw = np.asarray(bz[f"{role}_box"][:L], np.float64)
    K02, R02 = np.asarray(cz["K"][:L], np.float64), np.asarray(cz["R"][:L], np.float64)
    f_mm = float(meta["f_mm"])
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    calls: dict = {}
    try:
        write_caches(out, video, L, box_raw, R02, f_mm, preset["vo"])
        install_guards(calls)
        options = Options(pose_backend=preset["pose_backend"], sapiens2_resolution=preset["sapiens2_resolution"],
                          detector=DETECTOR, static_cam=False, vo=preset["vo"], f_mm=f_mm, smoothing=args.smoothing,
                          render=False, keep_frames=False)
        res = run_video(video, out, options, Assets(CKPT))
    finally:   # success or failure: frames and caches never reach the packed outputs (review finding 4)
        tidy_outputs(out, ROOT / "caches" / out_root.name / name)
    sec = time.time() - t0
    J = res["joints_incam"].numpy()
    Kx = res["K_fullimg"].numpy().astype(np.float64)
    bx = res["bbx_xyxy"].numpy().astype(np.float64)
    checks = {"cache": cache_check(calls), "K": k_check(Kx, K02),
              "boxes": {"iou_median": float(np.nanmedian(box_iou(bx, box_raw)))},
              "frames": {"L": L, "joints": int(len(J)), "finite": bool(np.isfinite(J).all()),
                         "ok": bool(len(J) == L and np.isfinite(J).all())},
              "kp2d": kp2d_check(res["joints_2d"].numpy(), np.asarray(bz[f"{role}_kp"][:L]), box_raw)}
    checks["boxes"]["ok"] = checks["boxes"]["iou_median"] >= IOU_MIN
    gates = ["cache", "K", "boxes", "frames"] + (["kp2d"] if args.smoke else [])
    if args.smoke:
        conv = convention_check(video, relative_rotations(R02), f_mm)
        if conv["pan_deg"] < PAN_MIN_DEG and L < int(meta["L"]):
            # the smoke window barely pans: compare on the whole clip instead (SimpleVO only, CPU, no ExMR run)
            conv["whole_clip"] = convention_check(clip_dir / "clip.mp4", relative_rotations(np.asarray(cz["R"], float)),
                                                  f_mm)
            conv["ok"] = conv["whole_clip"]["ok"]
        checks["convention"] = conv
        gates.append("convention")
    failed = [g for g in gates if not checks[g]["ok"]]
    log(f"{name}: {L} frames in {sec:.0f} s = {sec / L:.3f} s per person-frame; K focal {checks['K']['focal_rel_max'] * 100:.2f} %,"
        f" box IoU {checks['boxes']['iou_median']:.3f}, 2D {checks['kp2d']['median_pct']:.1f} % of box height; "
        + ("OK" if not failed else f"FAILED {failed}"))
    return {"ok": not failed, "failed": failed, "checks": checks, "seconds": round(sec, 1), "L": L, "f_mm": f_mm,
            "n0": meta["n0"], "video": str(video), "out": str(out)}


PRESETS = {"balanced": dict(pose_backend="sapiens2", sapiens2_resolution="448x336", vo="simplevo"),
           "fast": dict(pose_backend="rtmw", sapiens2_resolution="448x336", vo="simplevo")}


def cmd_run(args) -> None:
    clips_root, out_root = Path(args.clips), Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    clips = sorted(p for p in clips_root.iterdir() if (p / "meta.json").exists())
    if args.only:
        want = {c.strip() for c in args.only.split(",")}
        clips = [p for p in clips if p.name in want]
    if not clips or any(r not in ROLES for r in roles):
        raise SystemExit(f"nothing to run: clips {[p.name for p in clips]} roles {roles}")
    if not args.no_render:
        raise SystemExit("--no_render is required (no nvdiffrast on the pod)")
    rj = out_root / "run.json"
    report = json.loads(rj.read_text()) if rj.exists() else {}
    report.update({"schema": "dancecap.hmr_exmr/1", "exmr_commit": os.environ.get("EXMR_COMMIT"),
                   "exmr_hf_rev": os.environ.get("EXMR_HF_REV"), "mode": args.mode, "smoothing": args.smoothing,
                   "smoke": bool(args.smoke), "frames": args.frames, "runs": report.get("runs", {})})
    import torch
    report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    total = 0.0
    for cd in clips:
        for role in roles:
            name = f"{cd.name}_{role}"
            prev = report["runs"].get(name)
            if prev and prev.get("ok") and (out_root / name / "results.npz").exists() and not args.frames:
                log(f"{name}: done already")
                continue
            try:
                report["runs"][name] = run_one(args, cd, role, out_root, PRESETS[args.mode])
            except Exception as e:  # noqa: BLE001 - recorded, the other runs still go
                report["runs"][name] = {"ok": False, "failed": [f"{type(e).__name__}: {str(e)[:400]}"]}
                log(f"{name}: FAILED {type(e).__name__}: {str(e)[:400]}")
            total += report["runs"][name].get("seconds", 0.0)
            report["ok"] = all(v.get("ok") for v in report["runs"].values())
            rj.write_text(json.dumps(report, indent=1, default=float))
            if args.smoke and not report["runs"][name].get("ok"):
                break
    report["ok"] = all(v.get("ok") for v in report["runs"].values())
    rj.write_text(json.dumps(report, indent=1, default=float))
    log(f"{len(report['runs'])} run(s), {total:.0f} s; " + ("all OK" if report["ok"] else "NOT all OK: " + ", ".join(
        k for k, v in report["runs"].items() if not v.get("ok"))))
    if not report["ok"]:
        raise SystemExit(3)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("place-models")
    p = sub.add_parser("run")
    p.add_argument("--clips", required=True, help="work/<take>/hmr_test/clips (dancecap.hmr_test clips)")
    p.add_argument("--out", required=True)
    p.add_argument("--roles", default="lead,follow")
    p.add_argument("--only", help="comma-separated clip names")
    p.add_argument("--frames", type=int, help="the first N frames only (the smoke step)")
    p.add_argument("--mode", choices=list(PRESETS), default="balanced")
    p.add_argument("--smoothing", choices=["global", "both", "none"], default="none")
    p.add_argument("--no_render", action="store_true")
    p.add_argument("--smoke", action="store_true", help="also gate on the 2D check and the rotation convention")
    args = ap.parse_args(argv)
    if args.cmd == "place-models":
        cmd_place_models(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
