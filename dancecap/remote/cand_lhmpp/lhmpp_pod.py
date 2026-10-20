"""Pod side of the LHM++ appearance candidate (dancecap.mono4d.cand_lhmpp; environment from install.sh).

Runs in LHM++'s own python 3.10 venv with the repo root as cwd (its code reads ./pretrained_models/...). Mirrors
scripts/inference/to_gs_ply.py (the upstream GS export path) for building the avatar, then animates it with OUR poses:

    model     LHMPP-700M-PixelShuffle (SMPLX-FREE shape head + GS output), parse_app_configs / build_app_model exactly
              as to_gs_ply.py; the app disables the Multi-HMR shape estimator for this model (use_pred_shape_for_render)
    avatar    infer_single_view on the reference crops of one dancer (input/refs/<role>/<set>/*.png, already in
              LHM++'s 504x840 white-background format; image 00 = most frontal), the synthetic one-frame camera of
              to_gs_ply; shape modes:
                ref   betas = reference betas[:10], use_pred_shape_for_render=False
                pred  betas 0 + LHM++'s predicted shape merged for animation (= to_gs_ply / the app)
    animate   renderer.animate_gs_model (the same warp as forward_animate_gs / to_gs_ply's posed export) with the reference
              SMPL-X parameters of each held-out view; the joint rotations are used as they are (same SMPL-X tree,
              flat_hand_mean=True on both sides); trans = pelvis_ref - J0_rest(betas used) so the pelvis lands on the
              reference pelvis; expression 0 (the reference fit has none)
    render    gsplat.rasterization with the renderer's own settings (GSPlatRenderer.forward_single_view: RGB colours,
              eps2d 0.3, pinhole, classic) of both dancers together, colour channels + a role one-hot, at each phone's
              native size; variant <set>_<shape>; plus t16_ref_aa = the same with rasterize_mode="antialiased"
    checks    standard SMPL-X FK (smplx package, the prior's SMPLX_NEUTRAL.npz) of the transferred parameters vs the
              reference joints; distance of the posed Gaussian centres to that posed mesh and to the reference
              (16-betas) mesh; rest stature of the betas used
Outputs (--out): renders/<variant>/<key>.png (RGB) + <key>_role.png (R lead, G follow, B alpha), gs/<role>_<variant>_cano.ply
    (canonical A-pose 3DGS PLY), gs/scene_<variant>_<key>.ply (both dancers posed, world frame, one view time),
    turntable/<role>_<variant>.png, lhmpp_pod.json (versions, timings, per variant info, transfer checks, errors).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROLES = ("lead", "follow")


def log(*a):
    print("[lhmpp-pod]", time.strftime("%H:%M:%S"), *a, flush=True)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def look_at(C, P, up=(0.0, 1.0, 0.0)):
    z = np.asarray(P, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(-np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ np.asarray(C, float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=os.environ.get("LHMPP_ROOT", "/workspace/lhmpp_cand/LHM-plusplus"))
    ap.add_argument("--model-name", default=os.environ.get("LHMPP_MODEL", "LHMPP-700M-PixelShuffle"))
    ap.add_argument("--sets", default="w8,t16")
    ap.add_argument("--shapes", default="ref,pred")
    a = ap.parse_args()
    inp, out = Path(a.input).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": a.model_name, "root": a.root, "timings": {}, "variants": {}, "transfer": {}, "errors": []}
    t_start = time.time()
    os.chdir(a.root)
    sys.path.insert(0, a.root)
    os.environ.update({"APP_ENABLED": "1", "APP_MODEL_NAME": a.model_name, "APP_TYPE": "infer.human_lrm_a4o",
                       "NUMBA_THREADING_LAYER": "omp"})
    import torch
    torch._dynamo.config.disable = True
    from PIL import Image
    try:
        report["commit"] = subprocess.check_output(["git", "-C", a.root, "log", "-1", "--format=%H %cI"], text=True).strip()
    except Exception:  # noqa: BLE001
        report["commit"] = None
    import gsplat
    from gsplat.rendering import rasterization
    report["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "gsplat": gsplat.__version__,
                          "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0]}
    log(report["versions"])

    tg = _load_module("_lhmpp_to_gs_ply", os.path.join(a.root, "scripts/inference/to_gs_ply.py"))
    from accelerate import Accelerator

    from core.utils.model_card import MODEL_CONFIG
    model_path = os.path.join(a.root, "pretrained_models", "Damo_XR_Lab", a.model_name)
    cards = {a.model_name: {"model_path": model_path, "model_config": MODEL_CONFIG[a.model_name]}}
    t0 = time.time()
    _ = Accelerator()
    cfg, _cfg_train = tg.parse_app_configs(cards)
    model = tg.build_app_model(cfg)
    model.to("cuda")
    model.eval()
    report["timings"]["model_load_s"] = round(time.time() - t0, 1)
    report["use_smplx_shape_estimator"] = bool(cfg.get("use_smplx_shape_estimator", True))
    report["peak_gpu_gb_after_load"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    log(f"model loaded in {report['timings']['model_load_s']} s; shape estimator {report['use_smplx_shape_estimator']}")
    renderer = model.renderer

    import smplx
    hm = os.path.join(a.root, "pretrained_models", "human_model_files")
    views = json.loads((inp / "views.json").read_text())["views"]
    poses = json.loads((inp / "poses.json").read_text())
    F = len(views)
    keys = [v["key"] for v in views]
    for r in ROLES:
        assert [fr["key"] for fr in poses[r]["frames"]] == keys, "poses.json / views.json order"

    def smplx_fk(nb: int, betas: np.ndarray, P: dict, trans: np.ndarray):
        m = smplx.create(hm, model_type="smplx", gender="neutral", ext="npz", use_pca=False, flat_hand_mean=True,
                         num_betas=nb, batch_size=F).cuda()
        f = lambda k: torch.tensor(np.array([fr[k] for fr in P["frames"]], np.float32), device="cuda")  # noqa: E731
        b = torch.tensor(np.asarray(betas[:nb], np.float32), device="cuda")[None].expand(F, -1)
        with torch.no_grad():
            o = m(betas=b, global_orient=f("root_pose"), body_pose=f("body_pose"), jaw_pose=f("jaw_pose"),
                  leye_pose=f("leye_pose"), reye_pose=f("reye_pose"), left_hand_pose=f("lhand_pose"),
                  right_hand_pose=f("rhand_pose"), transl=torch.tensor(trans, dtype=torch.float32, device="cuda"))
        return o.vertices, o.joints[:, :55]

    def rest_j0_and_height(nb: int, betas: np.ndarray):
        m = smplx.create(hm, model_type="smplx", gender="neutral", ext="npz", use_pca=False, flat_hand_mean=True,
                         num_betas=nb, batch_size=1)
        with torch.no_grad():
            o = m(betas=torch.tensor(np.asarray(betas[:nb], np.float32))[None])
        v = o.vertices[0].numpy()
        return o.joints[0, 0].numpy().astype(float), float(v[:, 1].max() - v[:, 1].min())

    sets = [s for s in a.sets.split(",") if s]
    shapes = [s for s in a.shapes.split(",") if s]
    posed: dict = {}   # variant -> role -> list of (xyz, rot) per view, shared (opacity, scale, rgb)
    for role in ROLES:
        P = poses[role]
        betas16 = np.asarray(P["betas"], float)
        pel = np.array([fr["pelvis"] for fr in P["frames"]], float)
        J22 = np.array([fr["joints22"] for fr in P["frames"]], float)
        v_ref, j_ref = smplx_fk(16, betas16, P, np.array([fr["transl"] for fr in P["frames"]], float))
        e16 = np.linalg.norm(j_ref[:, :22].cpu().numpy() - J22, axis=-1)
        report["transfer"][role] = {"ref16_fk_vs_ref_joints_mm_median": round(float(np.median(e16) * 1e3), 2),
                                    "ref16_fk_vs_ref_joints_mm_max": round(float(e16.max() * 1e3), 2)}
        for sname in sets:
            files = sorted((inp / "refs" / role / sname).glob("*.png"))
            if not files:
                report["errors"].append(f"{role} {sname}: no reference crops")
                continue
            imgs = [np.asarray(Image.open(p).convert("RGB")) for p in files]
            ref = torch.from_numpy(np.stack(imgs) / 255.0).permute(0, 3, 1, 2).float().cuda()
            for shape in shapes:
                var = f"{sname}_{shape}"
                info = {"role": role, "set": sname, "shape": shape, "n_refs": len(files)}
                try:
                    t0 = time.time()
                    motion = tg._build_synthetic_motion_seq(cfg)
                    one = tg.slice_motion_seq_to_single_frame(motion, frame_idx=0)
                    sp_in = {k: v.cuda() for k, v in one["smplx_params"].items()}
                    if shape == "ref":
                        sp_in["betas"] = torch.tensor(betas16[:10], dtype=torch.float32, device="cuda")[None]
                    outs = None
                    for attempt in (1, 2):
                        try:
                            with torch.no_grad():
                                outs = model.infer_single_view(
                                    ref[None], None, None, render_c2ws=one["render_c2ws"].cuda(),
                                    render_intrs=one["render_intrs"].cuda(),
                                    render_bg_colors=one["render_bg_colors"].cuda(), smplx_params=dict(sp_in),
                                    ref_imgs_bool=torch.ones(1, len(files), dtype=torch.bool, device="cuda"),
                                    return_pred_shape=True, use_pred_shape_for_render=(shape == "pred"))
                            break
                        except torch.cuda.OutOfMemoryError:
                            if attempt == 2:
                                raise
                            log(f"{role} {var}: CUDA OOM, retrying once after empty_cache")
                            outs = None
                            torch.cuda.empty_cache()
                    info["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    gs_list, qp, tmat, _hid, _if, _me, _pe, pred_shape = outs
                    torch.cuda.synchronize()
                    info["infer_s"] = round(time.time() - t0, 2)
                    ps = None if pred_shape is None else pred_shape.detach().float().cpu().numpy().reshape(-1)[:10]
                    info["pred_shape"] = None if ps is None else [round(float(x), 4) for x in ps]
                    if shape == "pred":
                        if ps is None:
                            raise RuntimeError("model returned no predicted shape")
                        b10 = ps.astype(float)
                    else:
                        b10 = betas16[:10]
                    info["betas_used"] = [round(float(x), 4) for x in b10]
                    j0, height = rest_j0_and_height(10, b10)
                    info["rest_height_m"] = round(height, 4)
                    if ps is not None:
                        info["pred_shape_rest_height_m"] = round(rest_j0_and_height(10, ps)[1], 4)
                    trans = pel - j0[None]
                    # animate in chunks of 10 views
                    t1 = time.time()
                    xyz_all, rot_all, cano = [], [], None
                    for c0 in range(0, F, 10):
                        fr = P["frames"][c0:c0 + 10]
                        n = len(fr)
                        tt = lambda k, shp: torch.tensor(np.array([x[k] for x in fr], np.float32),  # noqa: E731
                                                         device="cuda").reshape(n, *shp)
                        sp = {"root_pose": tt("root_pose", (3,)), "body_pose": tt("body_pose", (21, 3)),
                              "jaw_pose": tt("jaw_pose", (3,)), "leye_pose": tt("leye_pose", (3,)),
                              "reye_pose": tt("reye_pose", (3,)), "lhand_pose": tt("lhand_pose", (15, 3)),
                              "rhand_pose": tt("rhand_pose", (15, 3)),
                              "trans": torch.tensor(trans[c0:c0 + n], dtype=torch.float32, device="cuda"),
                              "expr": torch.zeros(n, int(cfg.get("expr_param_dim", 100) or 100), device="cuda"),
                              "betas": torch.tensor(np.asarray(b10, np.float32), device="cuda")[None],
                              "transform_mat_neutral_pose": tmat[0]}
                        if sp["expr"].shape[1] != 100:
                            sp["expr"] = torch.zeros(n, 100, device="cuda")
                        with torch.no_grad():
                            anim, canos = renderer.animate_gs_model(gs_list[0], qp["neutral_coords"][0], sp, debug=False,
                                                                    mesh_meta=qp["mesh_meta"])
                        xyz_all += [g.xyz.detach() for g in anim]
                        rot_all += [g.rotation.detach() for g in anim]
                        cano = canos[0]
                    g0 = cano
                    torch.cuda.synchronize()
                    info["animate_s"] = round(time.time() - t1, 2)
                    info["gaussians"] = int(g0.xyz.shape[0])
                    info["use_rgb"] = bool(g0.use_rgb)
                    shs = g0.shs.detach()
                    rgb = shs.reshape(shs.shape[0], -1, 3)[:, 0] if g0.use_rgb else (shs.reshape(shs.shape[0], -1, 3)[:, 0] * 0.28209479177387814 + 0.5)
                    posed.setdefault(var, {})[role] = {"xyz": xyz_all, "rot": rot_all, "scale": g0.scaling.detach(),
                                                      "opacity": g0.opacity.detach().reshape(-1), "rgb": rgb.clamp(0, 1),
                                                      "cano": g0}
                    (out / "gs").mkdir(exist_ok=True)
                    g0.save_ply(str(out / "gs" / f"{role}_{var}_cano.ply"))
                    info["opacity_median"] = round(float(g0.opacity.median()), 4)
                    info["scale_median_mm"] = round(float(g0.scaling.max(-1).values.median()) * 1e3, 2)
                    # transfer checks
                    v_own, j_own = smplx_fk(10, b10, P, trans)
                    J = j_own[:, :22].cpu().numpy()
                    e = np.linalg.norm(J - J22, axis=-1)
                    info["fk_vs_ref_joints_mm"] = {"median": round(float(np.median(e) * 1e3), 1),
                                                   "p95": round(float(np.percentile(e, 95) * 1e3), 1),
                                                   "pelvis_max": round(float(e[:, 0].max() * 1e3), 2)}
                    from pytorch3d.ops import knn_points
                    d_own, d_ref = [], []
                    for f_ in range(F):
                        x = xyz_all[f_][None].float()
                        d_own.append(knn_points(x, v_own[f_][None].float(), K=1).dists[0, :, 0].sqrt())
                        d_ref.append(knn_points(x, v_ref[f_][None].float(), K=1).dists[0, :, 0].sqrt())
                    d_own, d_ref = torch.cat(d_own).cpu().numpy(), torch.cat(d_ref).cpu().numpy()
                    info["gauss_to_own_mesh_mm"] = {"median": round(float(np.median(d_own) * 1e3), 1),
                                                    "p95": round(float(np.percentile(d_own, 95) * 1e3), 1)}
                    info["gauss_to_ref_mesh_mm"] = {"median": round(float(np.median(d_ref) * 1e3), 1),
                                                    "p95": round(float(np.percentile(d_ref, 95) * 1e3), 1)}
                    log(f"{role} {var}: {info}")
                except Exception as ex:  # noqa: BLE001
                    info["error"] = f"{type(ex).__name__}: {ex}"
                    report["errors"].append(f"{role} {var}: {info['error']}")
                    log(traceback.format_exc())
                report["variants"].setdefault(var, {})[role] = info
                outs = gs_list = qp = tmat = _hid = _if = _me = _pe = pred_shape = None  # noqa: F841 - free the GPU
                torch.cuda.empty_cache()
    report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # ------------------------------------------------------------------ render the held-out views
    t0 = time.time()
    import cv2
    render_vars = [(v, "classic") for v in posed if all(r in posed[v] for r in ROLES)]
    if "t16_ref" in posed and all(r in posed["t16_ref"] for r in ROLES):
        render_vars.append(("t16_ref", "antialiased"))
    for var, mode in render_vars:
        name = var if mode == "classic" else f"{var}_aa"
        d = out / "renders" / name
        d.mkdir(parents=True, exist_ok=True)
        G = posed[var]
        n_l = G["lead"]["opacity"].shape[0]
        n_f = G["follow"]["opacity"].shape[0]
        onehot = torch.zeros(n_l + n_f, 2, device="cuda")
        onehot[:n_l, 0] = 1
        onehot[n_l:, 1] = 1
        cols = torch.cat([torch.cat([G["lead"]["rgb"], G["follow"]["rgb"]]), onehot], 1)
        scl = torch.cat([G["lead"]["scale"], G["follow"]["scale"]])
        opa = torch.cat([G["lead"]["opacity"], G["follow"]["opacity"]])
        for k, v in enumerate(views):
            means = torch.cat([G["lead"]["xyz"][k], G["follow"]["xyz"][k]]).float()
            quats = torch.cat([G["lead"]["rot"][k], G["follow"]["rot"][k]]).float()
            vm = torch.eye(4, device="cuda")
            vm[:3, :3] = torch.tensor(v["R"], device="cuda")
            vm[:3, 3] = torch.tensor(v["t_w"], device="cuda")
            K = torch.tensor(v["K"], dtype=torch.float32, device="cuda")
            with torch.no_grad():
                img, alpha, _ = rasterization(means=means, quats=quats, scales=scl.float(), opacities=opa.float(),
                                              colors=cols.float(), viewmats=vm[None], Ks=K[None], width=v["width"],
                                              height=v["height"], near_plane=0.01, far_plane=100.0, eps2d=0.3,
                                              render_mode="RGB", backgrounds=torch.zeros(1, 5, device="cuda"),
                                              camera_model="pinhole", rasterize_mode=mode)
            im = img[0].clamp(0, 1).cpu().numpy()
            al = alpha[0, :, :, 0].clamp(0, 1).cpu().numpy()
            cv2.imwrite(str(d / f"{v['key']}.png"), (im[:, :, :3] * 255 + 0.5).astype(np.uint8)[:, :, ::-1])
            role_img = np.stack([im[:, :, 3], im[:, :, 4], al], -1)
            cv2.imwrite(str(d / f"{v['key']}_role.png"), (role_img * 255 + 0.5).astype(np.uint8)[:, :, ::-1])
        if var == "t16_ref" and mode == "classic":  # one posed scene PLY (world frame) for viewers
            from core.structures.gaussian_model import GaussianModel
            k = len(views) // 2
            gm = GaussianModel(xyz=torch.cat([G["lead"]["xyz"][k], G["follow"]["xyz"][k]]),
                               opacity=opa[:, None], rotation=torch.cat([G["lead"]["rot"][k], G["follow"]["rot"][k]]),
                               scaling=scl, shs=cols[:, :3][:, None], use_rgb=True)
            gm.save_ply(str(out / "gs" / f"scene_{var}_{views[k]['key']}.ply"))
    report["timings"]["render_s"] = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ canonical turntables (front, left, back, right)
    try:
        td = out / "turntable"
        td.mkdir(exist_ok=True)
        H, W = 840, 504
        Kc = torch.tensor([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]], device="cuda")
        for var, G in posed.items():
            for role, g in G.items():
                c = g["cano"]
                ctr = ((c.xyz.max(0).values + c.xyz.min(0).values) / 2).cpu().numpy()
                hgt = float((c.xyz[:, 1].max() - c.xyz[:, 1].min()).item())
                dist = 1000.0 * hgt * 1.15 / H
                tiles = []
                for az in (0, 90, 180, 270):
                    C = ctr + dist * np.array([math.sin(math.radians(az)), 0.0, math.cos(math.radians(az))])
                    R, t = look_at(C, ctr)
                    vm = torch.eye(4, device="cuda")
                    vm[:3, :3] = torch.tensor(R, dtype=torch.float32, device="cuda")
                    vm[:3, 3] = torch.tensor(t, dtype=torch.float32, device="cuda")
                    with torch.no_grad():
                        img, _, _ = rasterization(means=c.xyz.float(), quats=c.rotation.float(), scales=g["scale"].float(),
                                                  opacities=g["opacity"].float(), colors=g["rgb"].float(),
                                                  viewmats=vm[None], Ks=Kc[None], width=W, height=H, near_plane=0.01,
                                                  far_plane=100.0, eps2d=0.3, backgrounds=torch.ones(1, 3, device="cuda"))
                    tiles.append((img[0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))
                cv2.imwrite(str(td / f"{role}_{var}.png"), np.concatenate(tiles, 1)[:, :, ::-1])
    except Exception as ex:  # noqa: BLE001
        report["errors"].append(f"turntable: {type(ex).__name__}: {ex}")
        log(traceback.format_exc())

    report["timings"]["total_s"] = round(time.time() - t_start, 1)
    report["summary"] = {"variants_rendered": [v if m == "classic" else f"{v}_aa" for v, m in render_vars],
                         "errors": len(report["errors"]), "gpu": report["versions"]["gpu"],
                         "timings": report["timings"], "peak_gpu_gb": report["peak_gpu_gb"]}
    (out / "lhmpp_pod.json").write_text(json.dumps(report, indent=1, default=str))
    log("done", json.dumps(report["summary"]))
    return 0 if render_vars else 3


if __name__ == "__main__":
    raise SystemExit(main())
