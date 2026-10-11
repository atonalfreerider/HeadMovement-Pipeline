r"""Ground-truth check of the ONE-camera ma_3d path on the synthetic couple (agent J3b, 2026-10-11).

    python -m dancecap.mono_synth run [--frames 300] [--noise 2.0] [--scales 1.0,0.9,1.1] [--init-noise 0.25]

The synthetic couple (dancecap.synthetic, two SMPL-X bodies with real FK in the Y-up world, floor y = 0, ground truth at 120 Hz) is seen by ONE camera that looks like the john_lula phone:
538x960, f = 808 px, 4.2 m from the couple, 1.43 m high, panning to follow them (tilt / pan changes every frame). Its 512 dense landmarks per dancer carry Gaussian pixel noise. The fit gets the
camera with its focal times s (s = 1: the truth; 0.9 / 1.1: the focal hypothesis is wrong by 10 %, exactly what J2 could not exclude), the pelvis start of dancecap.jl_mono replaced by the true
pelvis plus noise (init-noise m: the real start from the ankle rays is good to a few decimetres), the betas pinned to the truth (the real chunks pin the shape pass's betas) and the same
losses as the real fits (MA3D_MONO_OVERRIDES). Reported against the ground truth: the pelvis error split along and across the camera ray, the pose relative to the pelvis, the joint error.
This says what the single-camera optimiser can and cannot recover when its inputs are right (the 2D landmarks here are perfect apart from the noise, there is no clothing, no occlusion, no
blur, no wrong identity): an upper bound on the real accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import mamma_dyn as md
from .take import PACKAGE_ROOT

GT_DIR = PACKAGE_ROOT / "work" / "synthetic_mamma" / "gt120_anat"
OUT = PACKAGE_ROOT / "work" / "synthetic_mono"
W, H, F0 = 538, 960, 808.0
CAM_CENTRE = np.array([1.7, 1.43, 3.85])


def rig(t: np.ndarray, target: np.ndarray, f: float = F0) -> dict:
    """The phone: fixed centre, looks at the (smoothed) couple centre, a little roll; K (F, 3, 3), R (F, 3, 3) world -> camera, t (F, 3)."""
    C = np.repeat(CAM_CENTRE[None], len(t), 0)
    roll = np.deg2rad(1.5) * np.sin(2 * np.pi * t / 7.0)
    R = md._look_at(C, target, roll)
    K = np.zeros((len(t), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = W / 2, H / 2, 1.0
    return {"K": K, "R": R, "t": -np.einsum("fij,fj->fi", R, C)}


def smooth(x: np.ndarray, n: int = 31) -> np.ndarray:
    k = np.ones(n) / n
    pad = np.pad(x, ((n // 2, n // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


def ray_split(err: np.ndarray, gt_pt: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split error vectors (N, 3) into the part along the camera ray through the true point and the rest. Returns (along (N,), across (N,))."""
    u = gt_pt - centre
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    along = (err * u).sum(-1)
    across = np.linalg.norm(err - along[:, None] * u, axis=-1)
    return along, across


def metrics(pred: dict, gt_joints: dict, centre: np.ndarray) -> dict:
    out = {}
    for role in ("lead", "follow"):
        P, G = np.asarray(pred[role], float), np.asarray(gt_joints[role], float)
        e_pel = P[:, 0] - G[:, 0]
        al, ac = ray_split(e_pel, G[:, 0], centre)
        rel = np.linalg.norm((P[:, :22] - P[:, :1]) - (G[:, :22] - G[:, :1]), axis=-1).mean(1)
        j = np.linalg.norm(P[:, :22] - G[:, :22], axis=-1)
        rng_p = np.hypot(P[:, 0, 0] - centre[0], P[:, 0, 2] - centre[2])
        rng_g = np.hypot(G[:, 0, 0] - centre[0], G[:, 0, 2] - centre[2])
        out[role] = {"pelvis_along_ray_m": {"median": round(float(np.median(al)), 3), "mean_abs": round(float(np.abs(al).mean()), 3), "p95_abs": round(float(np.percentile(np.abs(al), 95)), 3)},
                     "pelvis_across_ray_m": {"median": round(float(np.median(ac)), 3), "p95": round(float(np.percentile(ac, 95)), 3)},
                     "range_ratio_median": round(float(np.median(rng_p / rng_g)), 4),
                     "pelvis_height_error_m": {"median": round(float(np.median(e_pel[:, 1])), 3), "mean_abs": round(float(np.abs(e_pel[:, 1]).mean()), 3)},
                     "pose_relative_to_pelvis_mm": {"median": round(float(np.median(rel)) * 1000, 1), "p95": round(float(np.percentile(rel, 95)) * 1000, 1)},
                     "joints22_mpjpe_mm": {"median": round(float(np.median(j)) * 1000, 1), "mean": round(float(j.mean()) * 1000, 1), "p95": round(float(np.percentile(j, 95)) * 1000, 1)},
                     "floor_path_ratio": round(float(np.linalg.norm(np.diff(P[:, 0, [0, 2]], axis=0), axis=-1).sum() / max(np.linalg.norm(np.diff(G[:, 0, [0, 2]], axis=0), axis=-1).sum(), 1e-9)), 4)}
    return out


def build_scene(frames: int, noise_px: float, seed: int) -> dict:
    """GT couple at the 30 Hz grid, the true camera, the noisy landmarks."""
    gt = {r: np.load(GT_DIR / "smplx" / f"{r}.npz") for r in ("lead", "follow")}
    t120 = gt["lead"]["t"]
    verts = {r: md._synthetic_verts(gt[r]) for r in gt}
    D = md._load_verts512()
    pts = np.stack([np.matmul(D, verts[r]) for r in ("lead", "follow")], 1)                 # (F120, 2, 512, 3)
    frames = min(frames, int((float(t120[-1]) - 0.55) * md.GRID_HZ))                       # the ground truth is 7 s long
    grid = md.master_grid(0.5, 0.5 + (frames - 1) / md.GRID_HZ)
    gi = np.searchsorted(t120, grid - 1e-6)
    assert np.allclose(t120[gi], grid, atol=1e-6)
    pel = np.asarray(gt["lead"]["joints"])[gi][:, 0], np.asarray(gt["follow"]["joints"])[gi][:, 0]
    target = smooth(0.5 * (pel[0] + pel[1]) * np.array([1, 0.0, 1]) + np.array([0, 0.9, 0]))
    rg = rig(grid, target)
    rng = np.random.default_rng(seed)
    X = pts[gi]
    px, z = md.project(X, rg["K"][:, None], rg["R"][:, None], rg["t"][:, None])
    px = px + rng.normal(0, noise_px, px.shape)
    inside = (z > 0.1) & (px[..., 0] >= 0) & (px[..., 0] < W) & (px[..., 1] >= 0) & (px[..., 1] < H)
    box = np.zeros(px.shape[:2])
    for f in range(len(grid)):
        for p in range(2):
            q = px[f, p][inside[f, p]]
            box[f, p] = 1.2 * (q.max(0) - q.min(0)).max() if len(q) > 2 else 512.0
    lv = md.sigma_px_to_logvar(np.full(px.shape[:3], max(noise_px, 1.0)), box[..., None])
    lm = np.concatenate([px, lv[..., None]], -1) * inside[..., None]
    gj = {r: np.asarray(gt[r]["joints"])[gi] for r in gt}
    return {"grid": grid, "rig": rg, "lm": lm, "inside": inside, "box": box, "gt_joints": gj, "betas": {r: np.asarray(gt[r]["betas"], float) for r in gt}, "pelvis": pel}


def fit_one(scene: dict, scale: float, work: Path, init_noise: float, seed: int, config_name: str = "ma3d_config.yaml") -> dict:
    grid, rg = scene["grid"], scene["rig"]
    if work.exists():
        shutil.rmtree(work)
    (work / "cameras").mkdir(parents=True)
    (work / "cameras" / "world_from_sfm.json").write_text(json.dumps({"s": 1.0, "R": np.eye(3).tolist(), "t": [0, 0, 0]}))
    native = work / "mamma" / "ma_2d_native"
    native.mkdir(parents=True)
    cam = "01"
    np.savez(native / f"{cam}.npz", landmarks=scene["lm"].astype(np.float32), visibilities=scene["inside"].astype(np.float32),
             contacts=np.zeros(scene["inside"].shape, np.float32), floor_contacts=np.zeros(scene["inside"].shape, np.float32))
    np.save(native / f"{cam}_boxes.npy", scene["box"].astype(np.float32))
    K = rg["K"].copy()
    K[:, 0, 0] *= scale
    K[:, 1, 1] *= scale                                                                        # the focal hypothesis of the FIT
    np.savez(work / "cameras" / f"{cam}.npz", t=grid, K=K, R=rg["R"], tvec=rg["t"], dist=np.zeros(len(grid)), ok=np.ones(len(grid), bool), reproj_px=np.zeros(len(grid)),
             n_inliers=np.full(len(grid), 1000))
    (work / "frames" / cam).mkdir(parents=True)
    (work / "frames" / cam / "frames.json").write_text(json.dumps({"camera": cam, "width": W, "height": H, "frame_count": len(grid),
                                                                    "frames": [{"file": f"{i:06d}.jpg", "t": float(tt)} for i, tt in enumerate(grid)]}))
    tracks = md.load_tracks(work)
    for tr in tracks.values():
        tr.files = []
    seq = "synthetic_mono"
    md.build_inputs(work, tracks, native, grid, seq)
    rng = np.random.default_rng(seed + 7)
    pos = np.stack(scene["pelvis"], 1) + rng.normal(0, init_noise, (len(grid), 2, 3))
    np.savez(work / "mamma" / "mono_init.npz", pos=pos.astype(np.float32))
    betas = {str(i): np.concatenate([scene["betas"][r], np.zeros(6)]).tolist() for i, r in enumerate(("lead", "follow"))}
    (work / "mamma" / "pin.json").write_text(json.dumps(betas))
    env = {"MAMMA_MONO": "1", "MAMMA_MONO_INIT": str(work / "mamma" / "mono_init.npz"), "MAMMA_INIT_BETAS": str(work / "mamma" / "pin.json"), "MAMMA_FREEZE_BETAS": "1"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        cfg = md.write_ma3d_config(work / "mamma" / config_name)
        t0 = time.time()
        run_dir, stats = md.run_ma3d(work, seq, [f"cam{cam}"], cfg, tag="run")
        secs = time.time() - t0
        md.export_smplx(run_dir, grid, work / "mamma" / "ma_cap" / seq / "gt", work / "mamma" / "ma_2d" / seq, work / "smplx")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    pred = {r: np.load(work / "smplx" / f"{r}.npz")["joints"] for r in ("lead", "follow")}
    m = metrics(pred, scene["gt_joints"], CAM_CENTRE)
    m["fit_seconds"] = round(secs, 1)
    m["reproj_px"] = {r: float(np.nanmedian(np.load(work / "smplx" / f"{r}.npz")["reproj_px"])) for r in ("lead", "follow")}
    return m


def run(frames: int, noise_px: float, scales: list[float], init_noise: float, seed: int = 0) -> dict:
    scene = build_scene(frames, noise_px, seed)
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"frames": frames, "noise_px": noise_px, "init_noise_m": init_noise, "camera": {"f_px": F0, "centre": CAM_CENTRE.tolist(), "size": [W, H]}, "fits": {}}
    for s in scales:
        rep["fits"][f"focal_x{s:g}"] = fit_one(scene, s, OUT / f"f{int(round(s * 100)):03d}", init_noise, seed)
        (OUT / "report.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps({f"focal_x{s:g}": rep["fits"][f"focal_x{s:g}"]}, indent=1), flush=True)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono_synth", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--noise", type=float, default=2.0)
    ap.add_argument("--scales", default="1.0,0.9,1.1")
    ap.add_argument("--init-noise", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(a.frames, a.noise, [float(x) for x in a.scales.split(",")], a.init_noise, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
