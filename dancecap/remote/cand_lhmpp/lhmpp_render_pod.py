"""Pod side of the john_lula FINAL LHM++ renders (dancecap.mono4d.jl_realism_render; environment from install.sh).

Same LHM++ calls as lhmpp_compare_pod.py (to_gs_ply's parse_app_configs / build_app_model, infer_single_view with shape
mode ref: betas = reference betas[:10], use_pred_shape_for_render False; renderer.animate_gs_model; trans = pelvis_ref -
J0_rest(betas[:10]); expression 0) and the same rasterisation (gsplat, both dancers' Gaussians in ONE call = mutual
occlusion, eps2d 0.3, pinhole, classic, K principal point + 0.5, background 0 -> RGBA straight alpha). What differs: the
whole dance is far too long to keep every animated frame on the GPU (2 x 160k Gaussians x ~6000 pose keys), so poses
are animated in chunks of CHUNK keys and every camera that uses those keys is rendered at once (streaming), and the PNG
encoding runs in a thread pool.

    input/refs/<role>/*.png   LHM++ reference crops (504 x 840, white outside the mask; slot 00 = frontal reference)
    input/poses.npz           per role: root/body/jaw/leye/reye/lhand/rhand axis-angle, pelvis, joints22, betas, t
    input/views.npz/.json     per group: K, R, tvec, pose key, file index; W, H; passes (priority order)
Outputs: <renders>/<group>/<index:05d>.png (RGBA), <out>/gs/<role>_cano.ply (canonical 3DGS), <out>/gs/scene_k<k>.ply
(both dancers posed, world frame), <out>/turntable/<role>.png, <out>/lhmpp_render_pod.json (timings, counts, errors).
The pod's kill deadline (/workspace/.dancecap_kill_at) is read before every chunk: when the time left would not cover
packing + downloading what exists, the remaining renders are skipped (reported), so finished passes always come home.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROLES = ("lead", "follow")
CHUNK = 10
KILL_FILE = Path("/workspace/.dancecap_kill_at")


def log(*a):
    print("[lhmpp-render]", time.strftime("%H:%M:%S"), *a, flush=True)


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


def kill_at() -> float | None:
    try:
        return float(KILL_FILE.read_text().split()[0])
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--root", default=os.environ.get("LHMPP_ROOT", "/workspace/lhmpp_cand/LHM-plusplus"))
    ap.add_argument("--model-name", default=os.environ.get("LHMPP_MODEL", "LHMPP-700M-PixelShuffle"))
    ap.add_argument("--png-level", type=int, default=3)
    ap.add_argument("--writers", type=int, default=12)
    ap.add_argument("--reserve-min", type=float, default=5.0, help="minutes kept for pack + download overhead")
    ap.add_argument("--download-mbps", type=float, default=8.0, help="assumed download MB/s for the reserve")
    ap.add_argument("--only", default="", help="comma list of groups (debug)")
    a = ap.parse_args()
    inp, out, ren = Path(a.input).resolve(), Path(a.out).resolve(), Path(a.renders).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": a.model_name, "root": a.root, "timings": {}, "dancers": {}, "groups": {}, "errors": [],
              "stopped_early": None,
              "render_settings": {"rasterize_mode": "classic", "eps2d": 0.3, "render_mode": "RGB",
                                  "camera_model": "pinhole", "near": 0.01, "far": 100.0, "principal_point": "+0.5",
                                  "background": "0 -> RGBA straight alpha", "png_compression": a.png_level,
                                  "both_dancers_one_rasterisation": True, "chunk_keys": CHUNK}}
    t_start = time.time()
    rep_path = out / "lhmpp_render_pod.json"

    def save_report():
        report["timings"]["total_s"] = round(time.time() - t_start, 1)
        rep_path.write_text(json.dumps(report, indent=1, default=str))

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
                          "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
                          "cpus": os.cpu_count()}
    log(report["versions"])

    # ------------------------------------------------------------------ inputs
    with np.load(inp / "poses.npz") as z:  # into memory once (NpzFile re-reads an array on every access)
        Z = {k: np.asarray(z[k]) for k in z.files}
    with np.load(inp / "views.npz") as z:
        V = {k: np.asarray(z[k]) for k in z.files}
    vj = json.loads((inp / "views.json").read_text())
    T_keys = np.asarray(Z["t"], float)
    NK = len(T_keys)
    report["pose_keys"] = NK
    only = {g for g in a.only.split(",") if g}
    passes = [[g for g in p if not only or g in only] for p in vj["passes"]]
    passes = [p for p in passes if p]
    log(f"{NK} pose keys; passes {passes}; groups " +
        ", ".join(f"{g}: {len(V[g + '_key'])} x {vj['groups'][g]['W']}x{vj['groups'][g]['H']}" for p in passes for g in p))

    # ------------------------------------------------------------------ model
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
    hm = os.path.join(a.root, "pretrained_models", "human_model_files")

    def rest_j0_and_height(nb: int, betas: np.ndarray):
        m = smplx.create(hm, model_type="smplx", gender="neutral", ext="npz", use_pca=False, flat_hand_mean=True,
                         num_betas=nb, batch_size=1)
        with torch.no_grad():
            o = m(betas=torch.tensor(np.asarray(betas[:nb], np.float32))[None])
        v = o.vertices[0].numpy()
        return o.joints[0, 0].numpy().astype(float), float(v[:, 1].max() - v[:, 1].min())

    fk_models = {}

    def fk_joints(role: str, b10: np.ndarray, idx: np.ndarray, trans: np.ndarray) -> np.ndarray:
        """10-betas SMPL-X FK (LHM++'s shape) of the given keys: joints 0-21 (transfer check)."""
        n = len(idx)
        m = smplx.create(hm, model_type="smplx", gender="neutral", ext="npz", use_pca=False, flat_hand_mean=True,
                         num_betas=10, batch_size=n).cuda()
        f = lambda k, w: torch.tensor(np.asarray(Z[f"{role}_{k}"][idx], np.float32).reshape(n, w),  # noqa: E731
                                      device="cuda")
        with torch.no_grad():
            o = m(betas=torch.tensor(np.asarray(b10, np.float32), device="cuda")[None].expand(n, -1),
                  global_orient=f("root", 3), body_pose=f("body", 63), jaw_pose=f("jaw", 3), leye_pose=f("leye", 3),
                  reye_pose=f("reye", 3), left_hand_pose=f("lhand", 45), right_hand_pose=f("rhand", 45),
                  transl=torch.tensor(trans[idx], dtype=torch.float32, device="cuda"))
        return o.joints[:, :22].cpu().numpy()

    # ------------------------------------------------------------------ the two avatars (one LHM++ forward each)
    A = {}
    for role in ROLES:
        files = sorted((inp / "refs" / role).glob("*.png"))
        info = {"role": role, "shape": "ref", "n_refs": len(files), "refs": [p.name for p in files]}
        try:
            if not files:
                raise RuntimeError("no reference crops")
            imgs = [np.asarray(Image.open(p).convert("RGB")) for p in files]
            info["ref_size"] = list(imgs[0].shape[:2])
            ref = torch.from_numpy(np.stack(imgs) / 255.0).permute(0, 3, 1, 2).float().cuda()
            betas16 = np.asarray(Z[f"{role}_betas"], float)
            t0 = time.time()
            motion = tg._build_synthetic_motion_seq(cfg)
            one = tg.slice_motion_seq_to_single_frame(motion, frame_idx=0)
            sp_in = {k: v.cuda() for k, v in one["smplx_params"].items()}
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
                            return_pred_shape=True, use_pred_shape_for_render=False)
                    break
                except torch.cuda.OutOfMemoryError:
                    if attempt == 2:
                        raise
                    log(f"{role}: CUDA OOM, retrying once after empty_cache")
                    torch.cuda.empty_cache()
            torch.cuda.synchronize()
            info["infer_s"] = round(time.time() - t0, 2)
            info["peak_gpu_gb_after_infer"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            gs_list, qp, tmat = outs[0], outs[1], outs[2]
            b10 = betas16[:10]
            info["betas_used"] = [round(float(x), 4) for x in b10]
            j0, height = rest_j0_and_height(10, b10)
            info["rest_height_m"] = round(height, 4)
            trans = np.asarray(Z[f"{role}_pelvis"], float) - j0[None]
            A[role] = {"gs": gs_list[0], "nc": qp["neutral_coords"][0], "mesh_meta": qp["mesh_meta"], "tmat": tmat[0],
                       "trans": trans, "b10": b10}
            # canonical model: one animate call at key 0 gives the canonical Gaussians
            g0 = animate(renderer, torch, A[role], Z, role, np.array([0]))[1]
            A[role]["cano"] = g0
            info["gaussians"] = int(g0.xyz.shape[0])
            info["use_rgb"] = bool(g0.use_rgb)
            shs = g0.shs.detach()
            rgb = shs.reshape(shs.shape[0], -1, 3)[:, 0] if g0.use_rgb else \
                (shs.reshape(shs.shape[0], -1, 3)[:, 0] * 0.28209479177387814 + 0.5)
            A[role].update({"scale": g0.scaling.detach().float(), "opacity": g0.opacity.detach().reshape(-1).float(),
                            "rgb": rgb.clamp(0, 1).float()})
            (out / "gs").mkdir(exist_ok=True)
            ply = out / "gs" / f"{role}_cano.ply"
            g0.save_ply(str(ply))
            info["cano_ply"] = ply.name
            info["cano_ply_bytes"] = ply.stat().st_size
            info["opacity_median"] = round(float(g0.opacity.median()), 4)
            info["scale_median_mm"] = round(float(g0.scaling.max(-1).values.median()) * 1e3, 2)
            idx = np.linspace(0, NK - 1, 60).round().astype(int)
            J = fk_joints(role, b10, idx, trans)
            e = np.linalg.norm(J - np.asarray(Z[f"{role}_joints22"], float)[idx], axis=-1)
            info["fk10_vs_ref_joints_mm"] = {"median": round(float(np.median(e) * 1e3), 2),
                                             "p95": round(float(np.percentile(e, 95) * 1e3), 2),
                                             "pelvis_max": round(float(e[:, 0].max() * 1e3), 3)}
            log(f"{role}: {info}")
        except Exception as ex:  # noqa: BLE001
            info["error"] = f"{type(ex).__name__}: {ex}"
            report["errors"].append(f"{role}: {info['error']}")
            log(traceback.format_exc())
        report["dancers"][role] = info
        outs = gs_list = qp = tmat = ref = None
        torch.cuda.empty_cache()
    save_report()
    if not all(r in A for r in ROLES):
        report["errors"].append("a dancer is missing: nothing rendered")
        save_report()
        return 3

    n_l = A["lead"]["opacity"].shape[0]
    cols = torch.cat([A["lead"]["rgb"], A["follow"]["rgb"]]).contiguous()
    scl = torch.cat([A["lead"]["scale"], A["follow"]["scale"]]).contiguous()
    opa = torch.cat([A["lead"]["opacity"], A["follow"]["opacity"]]).contiguous()
    report["gaussians_total"] = int(opa.shape[0])

    # ------------------------------------------------------------------ canonical turntables (front, left, back, right)
    try:
        td = out / "turntable"
        td.mkdir(exist_ok=True)
        H, W = 840, 504
        Kc = torch.tensor([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]], device="cuda")
        for role in ROLES:
            cg = A[role]["cano"]
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
                    img, _, _ = rasterization(means=cg.xyz.float(), quats=cg.rotation.float(), scales=A[role]["scale"],
                                              opacities=A[role]["opacity"], colors=A[role]["rgb"], viewmats=vm[None],
                                              Ks=Kc[None], width=W, height=H, near_plane=0.01, far_plane=100.0,
                                              eps2d=0.3, backgrounds=torch.ones(1, 3, device="cuda"))
                tiles.append((img[0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))
            cv2.imwrite(str(td / f"{role}.png"), np.concatenate(tiles, 1)[:, :, ::-1])
    except Exception as ex:  # noqa: BLE001
        report["errors"].append(f"turntable: {type(ex).__name__}: {ex}")
        log(traceback.format_exc())

    # ------------------------------------------------------------------ streaming passes
    pool = ThreadPoolExecutor(max_workers=max(2, min(a.writers, (os.cpu_count() or 4))))
    pending = threading.Semaphore(96)
    written = {"bytes": 0, "files": 0}
    wlock = threading.Lock()
    png_args = [cv2.IMWRITE_PNG_COMPRESSION, int(a.png_level)]

    def write_png(path: Path, arr: np.ndarray):
        try:
            ok = cv2.imwrite(str(path), arr, png_args)
            if not ok:
                raise RuntimeError("cv2.imwrite returned False")
            with wlock:
                written["bytes"] += path.stat().st_size
                written["files"] += 1
        except Exception as ex:  # noqa: BLE001
            with wlock:
                report["errors"].append(f"write {path.name}: {type(ex).__name__}: {ex}")
        finally:
            pending.release()

    def reserve_ok() -> bool:
        ka = kill_at()
        if ka is None:
            return True
        need = a.reserve_min * 60 + written["bytes"] / 1e6 / a.download_mbps + written["bytes"] / 1e6 / 150.0
        return time.time() + need < ka

    scene_done = False
    for pi, groups in enumerate(passes):
        t_pass = time.time()
        cams = []  # (key, group, j)
        for g in groups:
            (ren / g).mkdir(parents=True, exist_ok=True)
            keys = np.asarray(V[g + "_key"], int)
            cams += [(int(k), g, j) for j, k in enumerate(keys)]
            report["groups"][g] = {"planned": int(len(keys)), "rendered": 0, "W": vj["groups"][g]["W"],
                                   "H": vj["groups"][g]["H"], "dir": f"{g}/", "pass": pi}
        cams.sort()
        need_keys = sorted({c[0] for c in cams})
        by_key: dict = {}
        for c in cams:
            by_key.setdefault(c[0], []).append(c)
        mats = {g: (torch.tensor(np.asarray(V[g + "_R"], np.float32), device="cuda"),
                    torch.tensor(np.asarray(V[g + "_tvec"], np.float32), device="cuda"),
                    torch.tensor(np.asarray(V[g + "_K"], np.float32), device="cuda"),
                    np.asarray(V[g + "_index"], int)) for g in groups}
        t_anim = t_rast = 0.0
        stop = False
        for c0 in range(0, len(need_keys), CHUNK):
            if not reserve_ok():
                report["stopped_early"] = {"pass": pi, "groups": groups, "at_key_chunk": c0,
                                           "of": len(need_keys), "written_mb": round(written["bytes"] / 1e6, 1)}
                log(f"STOP: kill deadline too close ({report['stopped_early']})")
                stop = True
                break
            kk = np.array(need_keys[c0:c0 + CHUNK])
            ta = time.time()
            xyz, rot = {}, {}
            for role in ROLES:
                anim, _ = animate(renderer, torch, A[role], Z, role, kk)
                xyz[role] = [g_.xyz.detach().float() for g_ in anim]
                rot[role] = [g_.rotation.detach().float() for g_ in anim]
            torch.cuda.synchronize()
            t_anim += time.time() - ta
            tr_ = time.time()
            for n, k in enumerate(kk):
                means = torch.cat([xyz["lead"][n], xyz["follow"][n]])
                quats = torch.cat([rot["lead"][n], rot["follow"][n]])
                if not scene_done and k == int(vj.get("scene_key", -1)):
                    try:
                        from core.structures.gaussian_model import GaussianModel
                        gm = GaussianModel(xyz=means, opacity=opa[:, None], rotation=quats, scaling=scl,
                                           shs=cols[:, None], use_rgb=True)
                        gm.save_ply(str(out / "gs" / f"scene_key{k}.ply"))
                        report["scene_ply"] = f"scene_key{k}.ply"
                        scene_done = True
                    except Exception as ex:  # noqa: BLE001
                        report["errors"].append(f"scene ply: {type(ex).__name__}: {ex}")
                for (_, g, j) in by_key[int(k)]:
                    Rm, tv, Km, idx = mats[g]
                    Wd, Hd = int(vj["groups"][g]["W"]), int(vj["groups"][g]["H"])
                    try:
                        vm = torch.eye(4, device="cuda")
                        vm[:3, :3] = Rm[j]
                        vm[:3, 3] = tv[j]
                        K = Km[j].clone()
                        K[0, 2] += 0.5
                        K[1, 2] += 0.5
                        with torch.no_grad():
                            img, alpha, _ = rasterization(means=means, quats=quats, scales=scl, opacities=opa,
                                                          colors=cols, viewmats=vm[None], Ks=K[None], width=Wd,
                                                          height=Hd, near_plane=0.01, far_plane=100.0, eps2d=0.3,
                                                          render_mode="RGB", backgrounds=torch.zeros(1, 3, device="cuda"),
                                                          camera_model="pinhole", rasterize_mode="classic")
                            al = alpha[0].clamp(0, 1)
                            rgb = torch.where(al > 1.0 / 1024, img[0] / al.clamp_min(1e-6), torch.zeros_like(img[0]))
                            u8 = (torch.cat([rgb.clamp(0, 1).flip(-1), al], -1) * 255 + 0.5).to(torch.uint8)
                            arr = u8.cpu().numpy()
                        pending.acquire()
                        pool.submit(write_png, ren / g / f"{int(idx[j]):05d}.png", arr)
                        report["groups"][g]["rendered"] += 1
                    except Exception as ex:  # noqa: BLE001
                        report["errors"].append(f"{g} {j}: {type(ex).__name__}: {ex}")
            t_rast += time.time() - tr_
            xyz = rot = None
            if c0 % (CHUNK * 30) == 0:
                log(f"pass {pi} {groups}: keys {c0 + len(kk)}/{len(need_keys)} anim {t_anim:.0f} s rast {t_rast:.0f} s; "
                    f"written {written['files']} files {written['bytes'] / 1e6:.0f} MB")
        for g in groups:
            report["groups"][g].update({"anim_s": round(t_anim, 1), "raster_s": round(t_rast, 1),
                                        "pass_wall_s": round(time.time() - t_pass, 1)})
        save_report()
        if stop:
            break
    pool.shutdown(wait=True)
    report["written"] = {"files": written["files"], "mb": round(written["bytes"] / 1e6, 1)}
    report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    for g, gi in report["groups"].items():
        gi["files_on_disk"] = sum(1 for _ in (ren / g).glob("*.png"))
    report["summary"] = {"groups": {g: f"{gi['files_on_disk']}/{gi['planned']}" for g, gi in report["groups"].items()},
                         "errors": len(report["errors"]), "gpu": report["versions"]["gpu"],
                         "gaussians": {r: report["dancers"][r].get("gaussians") for r in ROLES},
                         "stopped_early": report["stopped_early"]}
    save_report()
    log("done", json.dumps(report["summary"]), f"total {report['timings']['total_s']} s")
    complete = all(gi["files_on_disk"] == gi["planned"] for gi in report["groups"].values())
    return 0 if complete and not report["errors"] else 3


def animate(renderer, torch, Ar: dict, Z, role: str, kk: np.ndarray):
    """LHM++'s own LBS animation of one avatar at pose keys kk -> (list of posed GaussianModel, canonical model)."""
    n = len(kk)
    f = lambda k, shp: torch.tensor(np.asarray(Z[f"{role}_{k}"][kk], np.float32), device="cuda").reshape(n, *shp)  # noqa: E731
    sp = {"root_pose": f("root", (3,)), "body_pose": f("body", (21, 3)), "jaw_pose": f("jaw", (3,)),
          "leye_pose": f("leye", (3,)), "reye_pose": f("reye", (3,)), "lhand_pose": f("lhand", (15, 3)),
          "rhand_pose": f("rhand", (15, 3)),
          "trans": torch.tensor(Ar["trans"][kk], dtype=torch.float32, device="cuda"),
          "expr": torch.zeros(n, 100, device="cuda"),
          "betas": torch.tensor(np.asarray(Ar["b10"], np.float32), device="cuda")[None],
          "transform_mat_neutral_pose": Ar["tmat"]}
    with torch.no_grad():
        anim, canos = renderer.animate_gs_model(Ar["gs"], Ar["nc"], sp, debug=False, mesh_meta=Ar["mesh_meta"])
    return anim, canos[0]


if __name__ == "__main__":
    raise SystemExit(main())
