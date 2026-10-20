"""Pod side of the LHM++ COMPARE mode (dancecap.mono4d.cand_lhmpp compare-*; environment from install.sh).

Runs in LHM++'s own python 3.10 venv with the repo root as cwd, like lhmpp_pod.py, and uses the same calls for the
avatar (to_gs_ply's parse_app_configs / build_app_model, infer_single_view, renderer.animate_gs_model) and the same
pose transfer (shape mode ref: betas = reference betas[:10], use_pred_shape_for_render False; joint rotations as they
are; trans = pelvis_ref - J0_rest(betas[:10]); expression 0). What differs is the camera list: every view of the
appearance-comparison spec (input/cameras.json, written by cand_lhmpp compare-prep from mono4d/compare/views.json).

    per variant (input/refs/<role>/<variant>/NN.png) and dancer: one LHM++ forward -> canonical 3DGS (PLY), animated at
    every pose key of input/poses.json (real-view photo times, the turntable instant, the 105 clip grid times, phone 03's
    105 clip frame times)
    render   gsplat.rasterization of BOTH dancers' Gaussians together (mutual occlusion), the renderer's own settings
             (RGB colours, eps2d 0.3, pinhole, classic) + a role one-hot; K's principal point + 0.5 (gsplat pixel
             centres at +0.5; the spec's K has them at integers, = dancer_splats.model); background 0 -> RGBA with
             straight alpha (rgb / alpha), transparent background
Outputs: <renders>/<variant>/<out>.png for every camera (out from cameras.json; RGBA), <renders>/<variant>/role/<id>.png
    (R lead, G follow, B alpha; real views), <out>/gs/<role>_<variant>_cano.ply (canonical A-pose 3DGS),
    <out>/gs/scene_<variant>_<key>.ply (both dancers posed at the turntable instant, world frame),
    <out>/turntable/<role>_<variant>.png (canonical front/left/back/right), <out>/lhmpp_compare_pod.json.
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
    print("[lhmpp-cmp]", time.strftime("%H:%M:%S"), *a, flush=True)


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


def write_rgba(cv2, path: Path, rgb_premul: np.ndarray, alpha: np.ndarray) -> None:
    """Premultiplied colour over black + alpha -> 8-bit RGBA PNG with straight alpha."""
    a = alpha[..., None]
    rgb = np.where(a > 1.0 / 1024, rgb_premul / np.maximum(a, 1e-6), 0.0).clip(0, 1)
    out = np.concatenate([rgb[..., ::-1], a], -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (out * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 4])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--root", default=os.environ.get("LHMPP_ROOT", "/workspace/lhmpp_cand/LHM-plusplus"))
    ap.add_argument("--model-name", default=os.environ.get("LHMPP_MODEL", "LHMPP-700M-PixelShuffle"))
    ap.add_argument("--variants", default="lhmpp_1phone,lhmpp_8phone")
    a = ap.parse_args()
    inp, out, ren = Path(a.input).resolve(), Path(a.out).resolve(), Path(a.renders).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": a.model_name, "root": a.root, "timings": {}, "variants": {}, "transfer": {}, "errors": [],
              "render_settings": {"rasterize_mode": "classic", "eps2d": 0.3, "render_mode": "RGB (5 ch: rgb + role)",
                                  "camera_model": "pinhole", "near": 0.01, "far": 100.0, "principal_point": "+0.5",
                                  "background": "0 -> RGBA straight alpha", "png_compression": 4}}
    t_start = time.time()
    os.chdir(a.root)
    sys.path.insert(0, a.root)
    os.environ.update({"APP_ENABLED": "1", "APP_MODEL_NAME": a.model_name, "APP_TYPE": "infer.human_lrm_a4o",
                       "NUMBA_THREADING_LAYER": "omp"})
    import cv2
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
    log(f"model loaded in {report['timings']['model_load_s']} s")
    renderer = model.renderer

    import smplx
    from pytorch3d.ops import knn_points
    hm = os.path.join(a.root, "pretrained_models", "human_model_files")
    poses = json.loads((inp / "poses.json").read_text())
    cams = json.loads((inp / "cameras.json").read_text())["renders"]
    keys = [fr["key"] for fr in poses["lead"]["frames"]]
    assert [fr["key"] for fr in poses["follow"]["frames"]] == keys, "poses.json: lead / follow key order"
    kidx = {k: n for n, k in enumerate(keys)}
    F = len(keys)
    missing = sorted({c["pose"] for c in cams} - set(keys))
    assert not missing, f"cameras.json poses without a pose key: {missing[:5]}"
    report["pose_keys"] = F
    report["cameras"] = len(cams)

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

    v_ref = {}
    for role in ROLES:
        P = poses[role]
        J22 = np.array([fr["joints22"] for fr in P["frames"]], float)
        vr, jr = smplx_fk(16, np.asarray(P["betas"], float), P, np.array([fr["transl"] for fr in P["frames"]], float))
        v_ref[role] = vr
        e16 = np.linalg.norm(jr[:, :22].cpu().numpy() - J22, axis=-1)
        report["transfer"][role] = {"ref16_fk_vs_ref_joints_mm_median": round(float(np.median(e16) * 1e3), 3),
                                    "ref16_fk_vs_ref_joints_mm_max": round(float(e16.max() * 1e3), 3)}
    log("transfer (16-betas FK vs reference joints):", report["transfer"])

    variants = [v for v in a.variants.split(",") if v]
    tt_key = next((c["pose"] for c in cams if c["group"] == "turntable"), keys[0])
    for var in variants:
        G = {}
        for role in ROLES:
            P = poses[role]
            betas16 = np.asarray(P["betas"], float)
            pel = np.array([fr["pelvis"] for fr in P["frames"]], float)
            J22 = np.array([fr["joints22"] for fr in P["frames"]], float)
            files = sorted((inp / "refs" / role / var).glob("*.png"))
            info = {"role": role, "variant": var, "shape": "ref", "n_refs": len(files)}
            outs = gs_list = qp = tmat = None
            try:
                if not files:
                    raise RuntimeError("no reference crops")
                imgs = [np.asarray(Image.open(p).convert("RGB")) for p in files]
                ref = torch.from_numpy(np.stack(imgs) / 255.0).permute(0, 3, 1, 2).float().cuda()
                t0 = time.time()
                motion = tg._build_synthetic_motion_seq(cfg)
                one = tg.slice_motion_seq_to_single_frame(motion, frame_idx=0)
                sp_in = {k: v.cuda() for k, v in one["smplx_params"].items()}
                sp_in["betas"] = torch.tensor(betas16[:10], dtype=torch.float32, device="cuda")[None]
                for attempt in (1, 2):
                    try:
                        with torch.no_grad():
                            outs = model.infer_single_view(
                                ref[None], None, None, render_c2ws=one["render_c2ws"].cuda(),
                                render_intrs=one["render_intrs"].cuda(),
                                render_bg_colors=one["render_bg_colors"].cuda(), smplx_params=dict(sp_in),
                                ref_imgs_bool=torch.ones(1, len(files), dtype=torch.bool, device="cuda"),
                                return_pred_shape=True, use_pred_shape_for_render=False)
                        break
                    except torch.cuda.OutOfMemoryError:
                        if attempt == 2:
                            raise
                        log(f"{role} {var}: CUDA OOM, retrying once after empty_cache")
                        outs = None
                        torch.cuda.empty_cache()
                info["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                gs_list, qp, tmat = outs[0], outs[1], outs[2]  # (+ hidden, img feats, motion emb, pos emb, pred shape)
                torch.cuda.synchronize()
                info["infer_s"] = round(time.time() - t0, 2)
                b10 = betas16[:10]
                info["betas_used"] = [round(float(x), 4) for x in b10]
                j0, height = rest_j0_and_height(10, b10)
                info["rest_height_m"] = round(height, 4)
                trans = pel - j0[None]
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
                          "expr": torch.zeros(n, 100, device="cuda"),
                          "betas": torch.tensor(np.asarray(b10, np.float32), device="cuda")[None],
                          "transform_mat_neutral_pose": tmat[0]}
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
                rgb = shs.reshape(shs.shape[0], -1, 3)[:, 0] if g0.use_rgb else \
                    (shs.reshape(shs.shape[0], -1, 3)[:, 0] * 0.28209479177387814 + 0.5)
                G[role] = {"xyz": xyz_all, "rot": rot_all, "scale": g0.scaling.detach(),
                           "opacity": g0.opacity.detach().reshape(-1), "rgb": rgb.clamp(0, 1), "cano": g0}
                (out / "gs").mkdir(exist_ok=True)
                ply = out / "gs" / f"{role}_{var}_cano.ply"
                g0.save_ply(str(ply))
                info["cano_ply"] = ply.name
                info["cano_ply_bytes"] = ply.stat().st_size
                info["opacity_median"] = round(float(g0.opacity.median()), 4)
                info["scale_median_mm"] = round(float(g0.scaling.max(-1).values.median()) * 1e3, 2)
                v_own, j_own = smplx_fk(10, b10, P, trans)
                e = np.linalg.norm(j_own[:, :22].cpu().numpy() - J22, axis=-1)
                info["fk_vs_ref_joints_mm"] = {"median": round(float(np.median(e) * 1e3), 2),
                                               "p95": round(float(np.percentile(e, 95) * 1e3), 2),
                                               "pelvis_max": round(float(e[:, 0].max() * 1e3), 3)}
                d_own, d_ref = [], []
                for f_ in range(0, F, 3):
                    x = xyz_all[f_][None].float()
                    d_own.append(knn_points(x, v_own[f_][None].float(), K=1).dists[0, :, 0].sqrt())
                    d_ref.append(knn_points(x, v_ref[role][f_][None].float(), K=1).dists[0, :, 0].sqrt())
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
            outs = gs_list = qp = tmat = ref = None  # free the GPU
            torch.cuda.empty_cache()
        if not all(r in G for r in ROLES):
            report["errors"].append(f"{var}: not rendered (a dancer is missing)")
            continue

        # ------------------------------------------------------------------ every camera of the spec
        t0 = time.time()
        vdir = ren / var
        n_l = G["lead"]["opacity"].shape[0]
        n_f = G["follow"]["opacity"].shape[0]
        onehot = torch.zeros(n_l + n_f, 2, device="cuda")
        onehot[:n_l, 0] = 1
        onehot[n_l:, 1] = 1
        cols = torch.cat([torch.cat([G["lead"]["rgb"], G["follow"]["rgb"]]), onehot], 1).float()
        scl = torch.cat([G["lead"]["scale"], G["follow"]["scale"]]).float()
        opa = torch.cat([G["lead"]["opacity"], G["follow"]["opacity"]]).float()
        counts, failed = {}, []
        for c in cams:
            try:
                k = kidx[c["pose"]]
                means = torch.cat([G["lead"]["xyz"][k], G["follow"]["xyz"][k]]).float()
                quats = torch.cat([G["lead"]["rot"][k], G["follow"]["rot"][k]]).float()
                vm = torch.eye(4, device="cuda")
                vm[:3, :3] = torch.tensor(c["R"], device="cuda")
                vm[:3, 3] = torch.tensor(c["tvec"], device="cuda")
                K = torch.tensor(c["K"], dtype=torch.float32, device="cuda")
                K[0, 2] += 0.5
                K[1, 2] += 0.5
                with torch.no_grad():
                    img, alpha, _ = rasterization(means=means, quats=quats, scales=scl, opacities=opa, colors=cols,
                                                  viewmats=vm[None], Ks=K[None], width=int(c["W"]), height=int(c["H"]),
                                                  near_plane=0.01, far_plane=100.0, eps2d=0.3, render_mode="RGB",
                                                  backgrounds=torch.zeros(1, 5, device="cuda"),
                                                  camera_model="pinhole", rasterize_mode="classic")
                im = img[0].clamp(0, 1).cpu().numpy()
                al = alpha[0, :, :, 0].clamp(0, 1).cpu().numpy()
                write_rgba(cv2, vdir / c["out"], im[:, :, :3], al)
                if c.get("role_map"):
                    (vdir / "role").mkdir(parents=True, exist_ok=True)
                    role_img = np.stack([im[:, :, 3], im[:, :, 4], al], -1)
                    cv2.imwrite(str(vdir / "role" / f"{c['id']}.png"),
                                (role_img * 255 + 0.5).astype(np.uint8)[:, :, ::-1])
                counts[c["group"]] = counts.get(c["group"], 0) + 1
            except Exception as ex:  # noqa: BLE001
                failed.append(f"{c['id']}: {type(ex).__name__}: {ex}")
        report["variants"][var]["renders"] = counts
        report["variants"][var]["render_failed"] = failed[:20]
        report["variants"][var]["render_s"] = round(time.time() - t0, 1)
        if failed:
            report["errors"].append(f"{var}: {len(failed)} renders failed (first: {failed[0]})")
        log(f"{var}: rendered {counts} in {report['variants'][var]['render_s']} s; failed {len(failed)}")

        # ------------------------------------------------------------------ scene PLY (world frame) at the turntable time
        try:
            from core.structures.gaussian_model import GaussianModel
            k = kidx[tt_key]
            gm = GaussianModel(xyz=torch.cat([G["lead"]["xyz"][k], G["follow"]["xyz"][k]]), opacity=opa[:, None],
                               rotation=torch.cat([G["lead"]["rot"][k], G["follow"]["rot"][k]]), scaling=scl,
                               shs=cols[:, :3][:, None], use_rgb=True)
            gm.save_ply(str(out / "gs" / f"scene_{var}_{tt_key}.ply"))
            report["variants"][var]["scene_ply"] = f"scene_{var}_{tt_key}.ply"
        except Exception as ex:  # noqa: BLE001
            report["errors"].append(f"{var} scene ply: {type(ex).__name__}: {ex}")

        # ------------------------------------------------------------------ canonical turntables (front, left, back, right)
        try:
            td = out / "turntable"
            td.mkdir(exist_ok=True)
            H, W = 840, 504
            Kc = torch.tensor([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]], device="cuda")
            for role, g in G.items():
                cg = g["cano"]
                ctr = ((cg.xyz.max(0).values + cg.xyz.min(0).values) / 2).cpu().numpy()
                hgt = float((cg.xyz[:, 1].max() - cg.xyz[:, 1].min()).item())
                dist = 1000.0 * hgt * 1.15 / H
                tiles = []
                for az in (0, 90, 180, 270):
                    C = ctr + dist * np.array([math.sin(math.radians(az)), 0.0, math.cos(math.radians(az))])
                    R, t = look_at(C, ctr)
                    vm = torch.eye(4, device="cuda")
                    vm[:3, :3] = torch.tensor(R, dtype=torch.float32, device="cuda")
                    vm[:3, 3] = torch.tensor(t, dtype=torch.float32, device="cuda")
                    with torch.no_grad():
                        img, _, _ = rasterization(means=cg.xyz.float(), quats=cg.rotation.float(),
                                                  scales=g["scale"].float(), opacities=g["opacity"].float(),
                                                  colors=g["rgb"].float(), viewmats=vm[None], Ks=Kc[None], width=W,
                                                  height=H, near_plane=0.01, far_plane=100.0, eps2d=0.3,
                                                  backgrounds=torch.ones(1, 3, device="cuda"))
                    tiles.append((img[0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))
                cv2.imwrite(str(td / f"{role}_{var}.png"), np.concatenate(tiles, 1)[:, :, ::-1])
        except Exception as ex:  # noqa: BLE001
            report["errors"].append(f"{var} turntable: {type(ex).__name__}: {ex}")
            log(traceback.format_exc())
        G = cols = scl = opa = onehot = None
        torch.cuda.empty_cache()

    report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    report["timings"]["total_s"] = round(time.time() - t_start, 1)
    done = [v for v in variants if report["variants"].get(v, {}).get("renders")]
    report["summary"] = {"variants_rendered": done, "errors": len(report["errors"]), "gpu": report["versions"]["gpu"],
                         "timings": report["timings"], "peak_gpu_gb": report["peak_gpu_gb"],
                         "gaussians": {v: {r: report["variants"][v].get(r, {}).get("gaussians") for r in ROLES}
                                       for v in variants}}
    (out / "lhmpp_compare_pod.json").write_text(json.dumps(report, indent=1, default=str))
    log("done", json.dumps(report["summary"]))
    return 0 if len(done) == len(variants) else 3


if __name__ == "__main__":
    raise SystemExit(main())
