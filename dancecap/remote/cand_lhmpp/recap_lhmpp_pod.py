"""Pod side of the recap LHM++ splat avatars (dancecap.mono4d.recap_splatav job; environment from install.sh).

Same LHM++ calls as lhmpp_render_pod.py (to_gs_ply's parse_app_configs / build_app_model, infer_single_view with shape
mode ref: betas = the recap fit's betas[:10], use_pred_shape_for_render False; renderer.animate_gs_model with
trans = pelvis_ref - J0_rest(betas[:10]), expression 0) and the same gsplat rasterisation (both dancers in ONE call,
eps2d 0.3, pinhole, classic, principal point + 0.5, background 0 -> RGBA straight alpha). New here:

EXPORT (docs/SPLAT_AVATAR_FORMAT.md section 7). Per dancer, from LHM++'s own tensors:
    mean0, rot0   animate_gs_model at the ZERO pose (all 55 rotations 0, trans 0): x0 = M_null x + shape blendshape and
                  normalize(quat(M_null)) * rotation_raw, i.e. the canonical zero pose of the shaped body
    weights       the 55 LBS weights LHM++ applies (get_transform_mat_vertex: the diffused voxel field sampled at
                  x = query point + offset, the template SMPL-X weights for hand and face query points), verified by
                  rebuilding LHM++'s posed centres from them
    rest joints   smplx_model.get_zero_pose_human(betas) (55), parents, the model's neutral pose (provenance)
    scale, opacity, rgb   activated values of the Gaussian attributes (GaussianModel: exp / sigmoid / RGB)
    flags         per Gaussian the query point's part flags (face, hands, upper / lower body) and |offset| (hair cue)
A2 CHECK. At the a2_* view keys: LHM++'s own animation vs our plain LBS from the exported data (numpy float64 FK about
the rest joints, pelvis at the reference pelvis, top-4 renormalised weights; also all 55): centre difference (mm),
rotation difference (deg; rotation of the blended 3x3 = LHM++'s rule, and the normalised quaternion blend), and the
gsplat renders of both dancers into the real phone (PSNR of ours vs LHM++, whole frame and inside the alpha).
PREVIEW. Groups prev_* rendered with LHM++'s own animation (streaming in chunks of CHUNK keys, as lhmpp_render_pod.py).

    input/refs/<role>/*.png, input/poses.npz, input/views.npz, input/views.json  (as lhmpp_render_pod.py)
Outputs: <out>/export/<role>_export.npz, <out>/a2/<view>_{lhmpp,ours4,ours4q,diff}.png, <out>/a2.json,
<out>/turntable/<role>_{zero,apose}.png, <renders>/<group>/<index:05d>.png, <out>/recap_lhmpp_pod.json.
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
    print("[recap-lhmpp]", time.strftime("%H:%M:%S"), *a, flush=True)


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


# ------------------------------------------------------------------ plain LBS (numpy float64), the format's posing rule
def rotmats(aa: np.ndarray) -> np.ndarray:
    """(J, 3) axis-angle -> (J, 3, 3) (Rodrigues)."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def skin_mats(aa55: np.ndarray, J: np.ndarray, parents: np.ndarray, pelvis: np.ndarray) -> np.ndarray:
    """S_j = G_j T(-J_j) with G_0 = [R_0 | pelvis], G_j = G_p [R_j | J_j - J_p] (SPLAT_AVATAR_FORMAT section 4)."""
    R = rotmats(aa55)
    G = np.zeros((55, 4, 4))
    for j in range(55):
        L = np.eye(4)
        L[:3, :3] = R[j]
        if parents[j] < 0:
            L[:3, 3] = pelvis
            G[j] = L
        else:
            L[:3, 3] = J[j] - J[parents[j]]
            G[j] = G[parents[j]] @ L
    S = G.copy()
    S[:, :3, 3] -= np.einsum("jab,jb->ja", G[:, :3, :3], J)
    return S


def _np_matrix_to_quaternion(M: np.ndarray) -> np.ndarray:
    """numpy port of pytorch3d.transforms.matrix_to_quaternion (0.7.x): best-conditioned of the four candidates."""
    m00, m01, m02 = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    m10, m11, m12 = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    m20, m21, m22 = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
    x = np.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], -1)
    q_abs = np.sqrt(np.maximum(x, 0.0))
    cand = np.stack([
        np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
        np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
        np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
        np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1)], -2)
    cand = cand / (2.0 * np.maximum(q_abs[..., None], 0.1))
    k = np.argmax(q_abs, -1)
    q = np.take_along_axis(cand, k[..., None, None].repeat(4, -1), -2)[..., 0, :]
    return np.where(q[..., :1] < 0, -q, q)


def quat_wxyz_from_mats(M: np.ndarray) -> np.ndarray:
    """pytorch3d's matrix_to_quaternion for (N,3,3) (LHM++'s rule, also for non-orthonormal blends), then normalised.
    Falls back to the numpy port where pytorch3d is not installed (local checks)."""
    try:
        import torch
        from pytorch3d.transforms import matrix_to_quaternion
        q = matrix_to_quaternion(torch.tensor(M, dtype=torch.float64))
        return torch.nn.functional.normalize(q, dim=-1).numpy()
    except ImportError:
        q = _np_matrix_to_quaternion(np.asarray(M, np.float64))
        return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qmul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def ang_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    d = np.abs(np.sum(q1 * q2, -1) / (np.linalg.norm(q1, axis=-1) * np.linalg.norm(q2, axis=-1)))
    return np.degrees(2 * np.arccos(np.clip(d, 0, 1)))


def lbs_pose(ex: dict, S: np.ndarray, top4: bool, qblend: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Posed centres and wxyz rotations of one avatar from the exported data and skin matrices S (55,4,4)."""
    x0 = ex["mean0"].astype(np.float64)
    if top4:
        idx, w = ex["top4_idx"], ex["top4_w"].astype(np.float64)
        M = np.einsum("nk,nkab->nab", w, S[idx])
    else:
        M = np.einsum("nj,jab->nab", ex["weights"].astype(np.float64), S)
    p = np.einsum("nab,nb->na", M[:, :3, :3], x0) + M[:, :3, 3]
    if qblend:
        qj = quat_wxyz_from_mats(S[:, :3, :3])  # (55,4)
        idx = ex["top4_idx"]
        w = ex["top4_w"].astype(np.float64)
        qk = qj[idx]  # (N,4,4)
        sgn = np.sign(np.sum(qk * qk[:, :1], -1, keepdims=True))
        sgn[sgn == 0] = 1
        qb = np.sum(w[..., None] * sgn * qk, 1)
        qb /= np.linalg.norm(qb, axis=-1, keepdims=True)
    else:
        qb = quat_wxyz_from_mats(M[:, :3, :3])
    return p, qmul_wxyz(qb, ex["rot0"].astype(np.float64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--root", default=os.environ.get("LHMPP_ROOT", "/workspace/lhmpp_cand/LHM-plusplus"))
    ap.add_argument("--model-name", default=os.environ.get("LHMPP_MODEL", "LHMPP-700M-PixelShuffle"))
    ap.add_argument("--png-level", type=int, default=3)
    ap.add_argument("--writers", type=int, default=12)
    ap.add_argument("--reserve-min", type=float, default=4.0)
    ap.add_argument("--download-mbps", type=float, default=8.0)
    a = ap.parse_args()
    inp, out, ren = Path(a.input).resolve(), Path(a.out).resolve(), Path(a.renders).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": a.model_name, "root": a.root, "timings": {}, "dancers": {}, "groups": {}, "errors": [],
              "stopped_early": None,
              "render_settings": {"rasterize_mode": "classic", "eps2d": 0.3, "render_mode": "RGB",
                                  "camera_model": "pinhole", "near": 0.01, "far": 100.0, "principal_point": "+0.5",
                                  "background": "0 -> RGBA straight alpha", "both_dancers_one_rasterisation": True}}
    t_start = time.time()
    rep_path = out / "recap_lhmpp_pod.json"

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
                          "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0], "cpus": os.cpu_count()}
    log(report["versions"])

    with np.load(inp / "poses.npz") as z:
        Z = {k: np.asarray(z[k]) for k in z.files}
    with np.load(inp / "views.npz") as z:
        V = {k: np.asarray(z[k]) for k in z.files}
    vj = json.loads((inp / "views.json").read_text())
    NK = len(Z["t"])
    report["pose_keys"] = NK

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
    renderer = model.renderer
    sm = renderer.smplx_model
    report["renderer"] = {"class": type(renderer).__name__, "skinning_class": type(sm).__name__,
                          "smpl_type": getattr(renderer, "smpl_type", None),
                          "apply_pose_blendshape": getattr(sm, "apply_pose_blendshape", None),
                          "cano_pose_type": getattr(getattr(sm, "base_skinning", None), "cano_pose_type", None)}
    log(f"model loaded in {report['timings']['model_load_s']} s; {report['renderer']}")

    import smplx
    hm = os.path.join(a.root, "pretrained_models", "human_model_files")

    def rest_j0(b10: np.ndarray) -> np.ndarray:
        m = smplx.create(hm, model_type="smplx", gender="neutral", ext="npz", use_pca=False, flat_hand_mean=True,
                         num_betas=10, batch_size=1)
        with torch.no_grad():
            o = m(betas=torch.tensor(np.asarray(b10, np.float32))[None])
        return o.joints[0, :55].numpy().astype(float)

    (out / "export").mkdir(exist_ok=True)
    A = {}
    for role in ROLES:
        files = sorted((inp / "refs" / role).glob("*.png"))
        info = {"role": role, "shape": "ref", "n_refs": len(files), "refs": [p.name for p in files]}
        try:
            imgs = [np.asarray(Image.open(p).convert("RGB")) for p in files]
            ref = torch.from_numpy(np.stack(imgs) / 255.0).permute(0, 3, 1, 2).float().cuda()
            betas16 = np.asarray(Z[f"{role}_betas"], float)
            b10 = betas16[:10]
            t0 = time.time()
            motion = tg._build_synthetic_motion_seq(cfg)
            one = tg.slice_motion_seq_to_single_frame(motion, frame_idx=0)
            sp_in = {k: v.cuda() for k, v in one["smplx_params"].items()}
            sp_in["betas"] = torch.tensor(b10, dtype=torch.float32, device="cuda")[None]
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
                    torch.cuda.empty_cache()
            torch.cuda.synchronize()
            info["infer_s"] = round(time.time() - t0, 2)
            info["peak_gpu_gb_after_infer"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            gs_list, qp, tmat = outs[0], outs[1], outs[2]
            gs = gs_list[0]
            info["betas_used"] = [round(float(x), 5) for x in b10]
            j0_std = rest_j0(b10)
            trans = np.asarray(Z[f"{role}_pelvis"], float) - j0_std[0][None]
            A[role] = {"gs": gs, "nc": qp["neutral_coords"][0], "mesh_meta": qp["mesh_meta"], "tmat": tmat[0],
                       "trans": trans, "b10": b10}
            # ---- export: zero-pose canonical through LHM++'s own animation
            zero = {"root_pose": torch.zeros(1, 3, device="cuda"), "body_pose": torch.zeros(1, 21, 3, device="cuda"),
                    "jaw_pose": torch.zeros(1, 3, device="cuda"), "leye_pose": torch.zeros(1, 3, device="cuda"),
                    "reye_pose": torch.zeros(1, 3, device="cuda"), "lhand_pose": torch.zeros(1, 15, 3, device="cuda"),
                    "rhand_pose": torch.zeros(1, 15, 3, device="cuda"), "trans": torch.zeros(1, 3, device="cuda"),
                    "expr": torch.zeros(1, 100, device="cuda"),
                    "betas": torch.tensor(b10, dtype=torch.float32, device="cuda")[None],
                    "transform_mat_neutral_pose": A[role]["tmat"]}
            with torch.no_grad():
                anim0, canos0 = renderer.animate_gs_model(gs, A[role]["nc"], zero, debug=False,
                                                          mesh_meta=A[role]["mesh_meta"])
            g0, gA = anim0[0], canos0[0]
            mean0 = g0.xyz.detach().double().cpu().numpy()
            rot0 = torch.nn.functional.normalize(g0.rotation.detach().double(), dim=-1).cpu().numpy()
            # ---- the weights LHM++ applies (get_transform_mat_vertex)
            with torch.no_grad():
                x = (A[role]["nc"] + gs.offset_xyz).float()
                W = sm.query_voxel_skinning_weights(x.view(1, -1, 3)).view(-1, 55)
                fix = ((sm.is_rhand + sm.is_lhand + sm.is_face) > 0)
                W[fix] = sm.skinning_weight[fix].to(W)
                Jz = sm.get_zero_pose_human(shape_param=torch.tensor(b10, dtype=torch.float32, device="cuda")[None],
                                            device="cuda", face_offset=None, joint_offset=None)[0]
            W = W.double().cpu().numpy()
            J = Jz.double().cpu().numpy()
            parents = np.asarray(sm.smplx_layer.parents.cpu().numpy(), int)
            parents[0] = -1
            order = np.argsort(-W, 1)[:, :4]
            w4 = np.take_along_axis(W, order, 1)
            wsum = W.sum(1)
            w4n = w4 / np.maximum(w4.sum(1, keepdims=True), 1e-12)
            scale = gs.scaling.detach().double().cpu().numpy()
            opac = gs.opacity.detach().double().reshape(-1).cpu().numpy()
            shs = gs.shs.detach().double()
            rgb = (shs.reshape(shs.shape[0], -1, 3)[:, 0] if gs.use_rgb else
                   shs.reshape(shs.shape[0], -1, 3)[:, 0] * 0.28209479177387814 + 0.5).clamp(0, 1).cpu().numpy()
            mm = A[role]["mesh_meta"]
            flags = {k: np.asarray(mm[k].detach().cpu().numpy()).astype(bool).reshape(-1)
                     for k in ("is_face", "is_lhand", "is_rhand", "is_upper_body", "is_lower_body", "is_constrain_body")
                     if k in mm and hasattr(mm[k], "detach")}
            nb = getattr(sm.base_skinning, "neutral_body_pose", None)
            ex = {"mean0": mean0.astype(np.float32), "rot0": rot0.astype(np.float32), "scale": scale.astype(np.float32),
                  "opacity": opac.astype(np.float32), "rgb": rgb.astype(np.float32),
                  "weights": W.astype(np.float32), "top4_idx": order.astype(np.int16), "top4_w": w4n.astype(np.float32),
                  "rest_joints": J, "rest_joints_std_smplx": j0_std, "parents": parents, "betas10": b10,
                  "betas16": betas16, "offset_norm": gs.offset_xyz.detach().norm(dim=-1).cpu().numpy().astype(np.float32),
                  "neutral_x": x.detach().cpu().numpy().astype(np.float32),
                  "apose_xyz": gA.xyz.detach().cpu().numpy().astype(np.float32),
                  "neutral_body_pose": (nb.detach().cpu().numpy() if hasattr(nb, "detach") else np.zeros((21, 3))),
                  "rot_raw": gs.rotation.detach().cpu().numpy().astype(np.float32),
                  **{f"flag_{k}": v for k, v in flags.items()}}
            np.savez_compressed(out / "export" / f"{role}_export.npz", **ex)
            A[role]["ex"] = ex
            A[role]["cano"] = gA
            A[role].update({"scale": gs.scaling.detach().float(), "opacity": gs.opacity.detach().reshape(-1).float(),
                            "rgb": torch.tensor(rgb, dtype=torch.float32, device="cuda")})
            info.update({"gaussians": int(mean0.shape[0]), "use_rgb": bool(gs.use_rgb),
                         "weight_sum": {"min": float(wsum.min()), "max": float(wsum.max())},
                         "top4_mass": {"mean": float(w4.sum(1).mean() / wsum.mean()),
                                       "min": float((w4.sum(1) / wsum).min()),
                                       "p01": float(np.percentile(w4.sum(1) / wsum, 1))},
                         "weights_nonzero_per_gaussian": {"median": float(np.median((W > 1e-4).sum(1))),
                                                          "max": int((W > 1e-4).sum(1).max())},
                         "rest_joints_vs_std_smplx_mm": float(np.abs(J - j0_std).max() * 1e3),
                         "fix_mask_gaussians": int(fix.sum().item()), "flags": {k: int(v.sum()) for k, v in flags.items()},
                         "opacity_median": float(np.median(opac)), "scale_max_median_mm": float(np.median(scale.max(1)) * 1e3)})
            log(f"{role}: {info}")
        except Exception as ex_:  # noqa: BLE001
            info["error"] = f"{type(ex_).__name__}: {ex_}"
            report["errors"].append(f"{role}: {info['error']}")
            log(traceback.format_exc())
        report["dancers"][role] = info
        torch.cuda.empty_cache()
        save_report()
    if not all(r in A and "ex" in A[r] for r in ROLES):
        report["errors"].append("a dancer is missing")
        save_report()
        return 3

    def animate(role: str, kk: np.ndarray):
        Ar = A[role]
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
            anim, _ = renderer.animate_gs_model(Ar["gs"], Ar["nc"], sp, debug=False, mesh_meta=Ar["mesh_meta"])
        return anim

    def aa55(role: str, k: int) -> np.ndarray:
        parts = [Z[f"{role}_root"][k].reshape(1, 3), Z[f"{role}_body"][k].reshape(21, 3),
                 Z[f"{role}_jaw"][k].reshape(1, 3), Z[f"{role}_leye"][k].reshape(1, 3),
                 Z[f"{role}_reye"][k].reshape(1, 3), Z[f"{role}_lhand"][k].reshape(15, 3),
                 Z[f"{role}_rhand"][k].reshape(15, 3)]
        return np.concatenate(parts, 0).astype(float)

    n_l = A["lead"]["opacity"].shape[0]
    cols = torch.cat([A["lead"]["rgb"], A["follow"]["rgb"]]).contiguous()
    scl = torch.cat([A["lead"]["scale"], A["follow"]["scale"]]).contiguous()
    opa = torch.cat([A["lead"]["opacity"], A["follow"]["opacity"]]).contiguous()

    def raster(means, quats_wxyz, R, tvec, K, Wd, Hd):
        vm = torch.eye(4, device="cuda")
        vm[:3, :3] = torch.as_tensor(R, dtype=torch.float32, device="cuda")
        vm[:3, 3] = torch.as_tensor(tvec, dtype=torch.float32, device="cuda")
        Kt = torch.as_tensor(K, dtype=torch.float32, device="cuda").clone()
        Kt[0, 2] += 0.5
        Kt[1, 2] += 0.5
        with torch.no_grad():
            img, alpha, _ = rasterization(means=means.float(), quats=quats_wxyz.float(), scales=scl, opacities=opa,
                                          colors=cols, viewmats=vm[None], Ks=Kt[None], width=int(Wd), height=int(Hd),
                                          near_plane=0.01, far_plane=100.0, eps2d=0.3, render_mode="RGB",
                                          backgrounds=torch.zeros(1, 3, device="cuda"), camera_model="pinhole",
                                          rasterize_mode="classic")
        return img[0].clamp(0, 1), alpha[0, ..., 0].clamp(0, 1)

    def to_rgba_u8(img, al):
        rgb = torch.where(al[..., None] > 1.0 / 1024, img / al[..., None].clamp_min(1e-6), torch.zeros_like(img))
        return (torch.cat([rgb.clamp(0, 1).flip(-1), al[..., None]], -1) * 255 + 0.5).to(torch.uint8).cpu().numpy()

    # ------------------------------------------------------------------ A2 check
    a2 = {"views": [], "summary": {}}
    try:
        (out / "a2").mkdir(exist_ok=True)
        groups = [g for g in vj["groups"] if g.startswith("a2_")]
        for g in groups:
            keys = np.asarray(V[g + "_key"], int)
            for j, k in enumerate(keys):
                rec = {"group": g, "j": int(j), "key": int(k), "index": int(V[g + "_index"][j])}
                posed = {}
                for role in ROLES:
                    an = animate(role, np.array([k]))[0]
                    p_l = an.xyz.detach().double().cpu().numpy()
                    q_l = torch.nn.functional.normalize(an.rotation.detach().double(), dim=-1).cpu().numpy()
                    ex = A[role]["ex"]
                    pel = np.asarray(Z[f"{role}_pelvis"][k], float)
                    # LHM++ puts the pelvis at J0_std + trans = pelvis_ref; our rule puts G_0 at the pelvis directly.
                    S = skin_mats(aa55(role, k), ex["rest_joints"], ex["parents"],
                                  ex["rest_joints"][0] + A[role]["trans"][k])
                    p55, q55 = lbs_pose(ex, S, top4=False)
                    p4, q4 = lbs_pose(ex, S, top4=True)
                    _, q4q = lbs_pose(ex, S, top4=True, qblend=True)
                    S_p = skin_mats(aa55(role, k), ex["rest_joints"], ex["parents"], pel)
                    p4p, _ = lbs_pose(ex, S_p, top4=True)
                    d = lambda P: np.linalg.norm(P - p_l, axis=1) * 1e3  # noqa: E731
                    rec[role] = {
                        "pos_mm_all55": {"mean": float(d(p55).mean()), "max": float(d(p55).max())},
                        "pos_mm_top4": {"mean": float(d(p4).mean()), "p99": float(np.percentile(d(p4), 99)),
                                        "max": float(d(p4).max())},
                        "pos_mm_top4_pelvis_rule": {"mean": float(d(p4p).mean()), "max": float(d(p4p).max())},
                        "rot_deg_all55": {"mean": float(ang_deg(q55, q_l).mean()),
                                          "p99": float(np.percentile(ang_deg(q55, q_l), 99))},
                        "rot_deg_top4": {"mean": float(ang_deg(q4, q_l).mean()),
                                         "p99": float(np.percentile(ang_deg(q4, q_l), 99))},
                        "rot_deg_top4_qblend": {"mean": float(ang_deg(q4q, q_l).mean()),
                                                "p99": float(np.percentile(ang_deg(q4q, q_l), 99))},
                        "pelvis_rule_offset_mm": float(np.linalg.norm(ex["rest_joints"][0] + A[role]["trans"][k] - pel) * 1e3)}
                    posed[role] = {"lhmpp": (p_l, q_l), "ours4": (p4, q4), "ours4q": (p4, q4q)}
                R_, tv_, K_ = V[g + "_R"][j], V[g + "_tvec"][j], V[g + "_K"][j]
                Wd, Hd = vj["groups"][g]["W"], vj["groups"][g]["H"]
                imgs = {}
                for meth in ("lhmpp", "ours4", "ours4q"):
                    means = torch.tensor(np.concatenate([posed[r][meth][0] for r in ROLES]), device="cuda")
                    quats = torch.tensor(np.concatenate([posed[r][meth][1] for r in ROLES]), device="cuda")
                    imgs[meth] = raster(means, quats, R_, tv_, K_, Wd, Hd)
                ref_img, ref_al = imgs["lhmpp"]
                inside = (ref_al > 0.02) | (imgs["ours4"][1] > 0.02)
                for meth in ("ours4", "ours4q"):
                    im, al = imgs[meth]
                    mse = float(((im - ref_img) ** 2).mean())
                    mse_in = float(((im - ref_img) ** 2)[inside].mean()) if inside.any() else float("nan")
                    rec[f"psnr_{meth}"] = round(10 * math.log10(1.0 / max(mse, 1e-12)), 2)
                    rec[f"psnr_{meth}_inside"] = round(10 * math.log10(1.0 / max(mse_in, 1e-12)), 2)
                    rec[f"alpha_absdiff_{meth}"] = round(float((al - ref_al).abs().mean()), 6)
                name = f"{g}_{j}"
                for meth in ("lhmpp", "ours4"):
                    cv2.imwrite(str(out / "a2" / f"{name}_{meth}.png"), to_rgba_u8(*imgs[meth]))
                diff = ((imgs["ours4"][0] - ref_img).abs().clamp(0, 0.25) * 4 * 255 + 0.5).to(torch.uint8)
                cv2.imwrite(str(out / "a2" / f"{name}_diff.png"), diff.flip(-1).cpu().numpy())
                a2["views"].append(rec)
                log(f"A2 {name}: psnr {rec['psnr_ours4']} / inside {rec['psnr_ours4_inside']} dB; "
                    f"lead {rec['lead']['pos_mm_top4']}, follow {rec['follow']['pos_mm_top4']}")
        vv = a2["views"]
        if vv:
            a2["summary"] = {
                "psnr_ours4_min": min(v["psnr_ours4"] for v in vv), "psnr_ours4_median": float(np.median([v["psnr_ours4"] for v in vv])),
                "psnr_ours4_inside_min": min(v["psnr_ours4_inside"] for v in vv),
                "psnr_ours4_inside_median": float(np.median([v["psnr_ours4_inside"] for v in vv])),
                "psnr_ours4q_inside_median": float(np.median([v["psnr_ours4q_inside"] for v in vv])),
                **{f"{r}_pos_mm_top4_mean": float(np.mean([v[r]["pos_mm_top4"]["mean"] for v in vv])) for r in ROLES},
                **{f"{r}_pos_mm_top4_max": float(np.max([v[r]["pos_mm_top4"]["max"] for v in vv])) for r in ROLES},
                **{f"{r}_pos_mm_all55_max": float(np.max([v[r]["pos_mm_all55"]["max"] for v in vv])) for r in ROLES},
                **{f"{r}_rot_deg_top4_mean": float(np.mean([v[r]["rot_deg_top4"]["mean"] for v in vv])) for r in ROLES},
                **{f"{r}_rot_deg_top4_qblend_mean": float(np.mean([v[r]["rot_deg_top4_qblend"]["mean"] for v in vv])) for r in ROLES}}
        (out / "a2.json").write_text(json.dumps(a2, indent=1))
        report["a2"] = a2["summary"]
    except Exception as ex_:  # noqa: BLE001
        report["errors"].append(f"a2: {type(ex_).__name__}: {ex_}")
        log(traceback.format_exc())
    save_report()

    # ------------------------------------------------------------------ turntables (zero pose, A-pose)
    try:
        td = out / "turntable"
        td.mkdir(exist_ok=True)
        H, W = 840, 504
        Kc = np.array([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]])
        for role in ROLES:
            ex = A[role]["ex"]
            for name, xyz, rot in (("zero", ex["mean0"], ex["rot0"]),
                                   ("apose", A[role]["cano"].xyz.detach().cpu().numpy(),
                                    A[role]["cano"].rotation.detach().cpu().numpy())):
                ctr = (xyz.max(0) + xyz.min(0)) / 2
                hgt = float(xyz[:, 1].max() - xyz[:, 1].min())
                dist = 1000.0 * hgt * 1.15 / H
                tiles = []
                for az in (0, 90, 180, 270):
                    C = ctr + dist * np.array([math.sin(math.radians(az)), 0.0, math.cos(math.radians(az))])
                    R, t = look_at(C, ctr)
                    vm = torch.eye(4, device="cuda")
                    vm[:3, :3] = torch.tensor(R, dtype=torch.float32, device="cuda")
                    vm[:3, 3] = torch.tensor(t, dtype=torch.float32, device="cuda")
                    with torch.no_grad():
                        img, _, _ = rasterization(means=torch.tensor(xyz, device="cuda").float(),
                                                  quats=torch.tensor(rot, device="cuda").float(),
                                                  scales=A[role]["scale"], opacities=A[role]["opacity"],
                                                  colors=A[role]["rgb"], viewmats=vm[None],
                                                  Ks=torch.tensor(Kc, dtype=torch.float32, device="cuda")[None],
                                                  width=W, height=H, near_plane=0.01, far_plane=100.0, eps2d=0.3,
                                                  backgrounds=torch.ones(1, 3, device="cuda"))
                    tiles.append((img[0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))
                cv2.imwrite(str(td / f"{role}_{name}.png"), np.concatenate(tiles, 1)[:, :, ::-1])
    except Exception as ex_:  # noqa: BLE001
        report["errors"].append(f"turntable: {type(ex_).__name__}: {ex_}")
        log(traceback.format_exc())
    save_report()

    # ------------------------------------------------------------------ preview passes (LHM++'s own animation)
    pool = ThreadPoolExecutor(max_workers=max(2, min(a.writers, (os.cpu_count() or 4))))
    pending = threading.Semaphore(96)
    written = {"bytes": 0, "files": 0}
    wlock = threading.Lock()
    png_args = [cv2.IMWRITE_PNG_COMPRESSION, int(a.png_level)]

    def write_png(path: Path, arr: np.ndarray):
        try:
            if not cv2.imwrite(str(path), arr, png_args):
                raise RuntimeError("cv2.imwrite returned False")
            with wlock:
                written["bytes"] += path.stat().st_size
                written["files"] += 1
        except Exception as ex2:  # noqa: BLE001
            with wlock:
                report["errors"].append(f"write {path.name}: {type(ex2).__name__}: {ex2}")
        finally:
            pending.release()

    def reserve_ok() -> bool:
        ka = kill_at()
        if ka is None:
            return True
        need = a.reserve_min * 60 + written["bytes"] / 1e6 / a.download_mbps
        return time.time() + need < ka

    groups = [g for g in vj.get("passes", [[]])[0]] if vj.get("passes") else [g for g in vj["groups"] if g.startswith("prev_")]
    cams = []
    for g in groups:
        (ren / g).mkdir(parents=True, exist_ok=True)
        keys = np.asarray(V[g + "_key"], int)
        cams += [(int(k), g, j) for j, k in enumerate(keys)]
        report["groups"][g] = {"planned": int(len(keys)), "rendered": 0, "W": vj["groups"][g]["W"],
                               "H": vj["groups"][g]["H"], "dir": f"{g}/"}
    cams.sort()
    need_keys = sorted({c[0] for c in cams})
    by_key: dict = {}
    for c in cams:
        by_key.setdefault(c[0], []).append(c)
    t_p = time.time()
    for c0 in range(0, len(need_keys), CHUNK):
        if not reserve_ok():
            report["stopped_early"] = {"at_key_chunk": c0, "of": len(need_keys)}
            break
        kk = np.array(need_keys[c0:c0 + CHUNK])
        an = {r: animate(r, kk) for r in ROLES}
        for n, k in enumerate(kk):
            means = torch.cat([an["lead"][n].xyz.detach(), an["follow"][n].xyz.detach()])
            quats = torch.cat([an["lead"][n].rotation.detach(), an["follow"][n].rotation.detach()])
            for (_, g, j) in by_key[int(k)]:
                try:
                    img, al = raster(means, quats, V[g + "_R"][j], V[g + "_tvec"][j], V[g + "_K"][j],
                                     vj["groups"][g]["W"], vj["groups"][g]["H"])
                    arr = to_rgba_u8(img, al)
                    pending.acquire()
                    pool.submit(write_png, ren / g / f"{int(V[g + '_index'][j]):05d}.png", arr)
                    report["groups"][g]["rendered"] += 1
                except Exception as ex2:  # noqa: BLE001
                    report["errors"].append(f"{g} {j}: {type(ex2).__name__}: {ex2}")
        an = None
    pool.shutdown(wait=True)
    report["timings"]["preview_s"] = round(time.time() - t_p, 1)
    report["written"] = {"files": written["files"], "mb": round(written["bytes"] / 1e6, 1)}
    report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    for g, gi in report["groups"].items():
        gi["files_on_disk"] = sum(1 for _ in (ren / g).glob("*.png"))
    save_report()
    log("done", json.dumps({"groups": {g: f"{gi['files_on_disk']}/{gi['planned']}" for g, gi in report["groups"].items()},
                            "errors": report["errors"][:5], "a2": report.get("a2")}, default=str))
    complete = all(gi["files_on_disk"] == gi["planned"] for gi in report["groups"].values())
    return 0 if complete and not report["errors"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
