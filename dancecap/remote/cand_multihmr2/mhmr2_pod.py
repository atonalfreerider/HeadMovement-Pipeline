"""Pod side of the Multi-HMR 2 candidate (dancecap.mono4d.cand_multihmr2; installed by install.sh).

Input  (prep): input/meta.json {frames: [{file, i, t, K (3x3, full-res OpenCV)}], width, height} + input/frames/*.jpg
       (the canonical 1080x1920 frames of the window, unchanged), input/roles.json {i: {lead: box, follow: box}}
       (only used to decide which detections keep their full SMPL-X-topology mesh) and input/smplx_jreg.npz
       (J_regressor (55, 10475) of the neutral SMPL-X model).
Run: per frame the official preprocessing (multihmr2.datasets.itw_image.preprocess_image: longest side 768, zero pad to
     /16) and ONE encoder pass; the HPH decoder then runs twice on the same features:
       predK  the encoder's own predicted camera (z.K) - exactly what `multihmr2 --video` does
       ourK   our intrinsics (cameras/06.npz K, scaled to the 768-px input) passed as the decoder's K argument. In the
              decoder K only enters the person translation (dist = f / exp(d), transl = K^-1 [loc, 1] * dist), so pose
              and shape are unchanged and depth scales with f.
     Each variant has its own FeatPelvisTracker, fed exactly as multihmr2.api.infer_video feeds it.
Conversion (official Anny -> SMPL-X path): the decoder's Anny inputs (163-bone pose parameters, phenotypes) are captured
     with forward hooks on its body model and re-posed through anny.create_fullbody_model(..., topology="smplx") - the
     same MakeHuman rig and parameterization with NAVER's non-commercial anny2smplx retopology - which gives every
     detection a 10475-vertex SMPL-X-topology mesh, placed with the decoder's own translation; SMPL-X joints = the
     SMPL-X J_regressor applied to that posed mesh.
Output (--out): multihmr2_<variant>.npz (per detection: frame, track, conf, transl, pelvis (Anny root), j55 (SMPL-X joints,
     camera metres, OpenCV axes), anny_j3d (163 bone heads), shape, rotmat (local pose), offset, iou_lead / iou_follow;
     verts + verts_det for detections overlapping a dancer box; per frame K768, f_pred) + mhmr2_pod.json (versions,
     timings, checks, errors).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
SCALE_NOTE = "uv_full_opencv = uv_768 / s - 0.5 with s = 768 / max(W, H) (continuous-pixel convention of the resize)"


def log(*a):
    print("[mhmr2-pod]", time.strftime("%H:%M:%S"), *a, flush=True)


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


def git_head(path: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", path, "log", "-1", "--format=%H %cI"], text=True).strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input dir (meta.json, frames/, roles.json, smplx_jreg.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="/workspace/mhmr2_cand/checkpoints/multihmr2.pt")
    ap.add_argument("--conf-thresh", type=float, default=0.4)
    ap.add_argument("--dist-thresh-nms", type=float, default=0.25)
    ap.add_argument("--verts-iou", type=float, default=0.1, help="keep the full mesh of detections whose projected-joint "
                    "box overlaps a dancer box by more than this IoU")
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (smoke test)")
    a = ap.parse_args()
    inp, out = Path(a.input), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"args": vars(a), "errors": [], "timings": {}, "checks": {}, "scale_note": SCALE_NOTE}
    T0 = time.time()
    try:
        import anny
        import importlib.metadata as md
        from multihmr2 import init_hmr_session
        from multihmr2.datasets.itw_image import preprocess_image
        from multihmr2.tracker import FeatPelvisTracker

        rep["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                           "anny": md.version("anny"), "numpy": np.__version__,
                           "multihmr2": git_head("/workspace/mhmr2_cand/multi-hmr2"),
                           "dinov3": git_head(os.path.join(os.environ.get("TORCH_HOME", ""), "hub",
                                                           "facebookresearch_dinov3_main"))}
        log("versions", rep["versions"])
        meta = json.loads((inp / "meta.json").read_text())
        roles = json.loads((inp / "roles.json").read_text())
        frames = meta["frames"][: a.limit] if a.limit else meta["frames"]
        Jr = torch.tensor(np.load(inp / "smplx_jreg.npz")["J_regressor"], dtype=torch.float32, device="cuda")

        t0 = time.time()
        sess = init_hmr_session(a.ckpt, compile_model=False)
        dec = sess.model.full_body_decoder
        rep["timings"]["session_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        sx = anny.create_fullbody_model(local_changes=True, pose_parameterization="root_relative", topology="smplx",
                                        all_phenotypes=False).to(dtype=torch.float32, device="cuda")
        sx.set_skinning_method("lbs")
        rep["timings"]["smplx_model_s"] = round(time.time() - t0, 1)
        rep["checks"]["rig_equal"] = list(sx.bone_labels) == list(dec.body_model.bone_labels)
        rep["checks"]["phenotypes"] = list(dec.body_model.phenotype_labels)
        rep["checks"]["smplx_verts"] = int(sx.template_vertices.shape[0])
        log("session", rep["timings"], "rig_equal", rep["checks"]["rig_equal"], "smplx verts", rep["checks"]["smplx_verts"])
        if not rep["checks"]["rig_equal"]:
            raise RuntimeError("smplx-topology Anny rig differs from the decoder's rig")

        cap: dict = {}

        def pre(mod, args, kwargs):
            cap["pose"] = kwargs["pose_parameters"].detach().float()
            cap["pheno"] = {k: v.detach().float() for k, v in kwargs["phenotype_kwargs"].items()}

        def post(mod, args, kwargs, output):
            cap["bone_poses"] = output["bone_poses"].detach().float()

        dec.body_model.register_forward_pre_hook(pre, with_kwargs=True)
        dec.body_model.register_forward_hook(post, with_kwargs=True)

        variants = ("ourK", "predK")
        trackers = {v: FeatPelvisTracker() for v in variants}
        for tr in trackers.values():
            tr.reset()
        rows = {v: [] for v in variants}
        verts = {v: [] for v in variants}
        perframe = {v: {"K768": [], "n": []} for v in variants}
        f_pred, times = [], []
        max_bone_diff = {v: 0.0 for v in variants}
        max_match_diff = {v: 0.0 for v in variants}
        lc0 = None
        for k, fr in enumerate(frames):
            t1 = time.time()
            path = str(inp / "frames" / fr["file"])
            x = preprocess_image(path).to(sess.device)
            hh, ww = x.shape[-2:]
            s = max(hh, ww) / max(meta["width"], meta["height"])
            Kf = np.asarray(fr["K"], float)
            K768 = torch.tensor([[Kf[0, 0] * s, 0.0, (Kf[0, 2] + 0.5) * s],
                                 [0.0, Kf[1, 1] * s, (Kf[1, 2] + 0.5) * s],
                                 [0.0, 0.0, 1.0]], dtype=torch.float32, device=sess.device)[None]
            with torch.inference_mode():
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    z = sess.model.encoder(x)
                f_pred.append(float(z.K[0, 0, 0]) / s)
                for v in variants:
                    Kv = K768 if v == "ourK" else z.K
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        pred = dec(z, K=Kv, conf_thresh=a.conf_thresh, dist_thresh_nms=a.dist_thresh_nms, lowres=False)[0]
                    P = pred.persons
                    n = int(P.num_person)
                    perframe[v]["K768"].append(Kv[0].float().cpu().numpy())
                    perframe[v]["n"].append(n)
                    if n == 0:  # as infer_video: the tracker is not called on a frame without detections
                        continue
                    trackfeat = P.trackfeat.clone().float()
                    pelvis_xyz = P.transl_pelvis.clone().float()
                    pelvis_ijn = torch.cat([P.j2d[:, 0] / sess.img_size,
                                            torch.log(1 / pelvis_xyz[:, 2:3].clamp_min(1e-5))], dim=-1).float()
                    pelvis_ori = P.bone_poses[:, 0, :3, :3].clone().float()
                    ids = trackers[v].track_next_frame(k, trackfeat, pelvis_xyz, pelvis_ijn, pelvis_ori)
                    # post-NMS persons -> the body-model rows the hook captured (pre-NMS)
                    bp = P.bone_poses.float()
                    d = (bp[:, None, :, :3, 3] - cap["bone_poses"][None, :, :, :3, 3]).abs().amax(dim=(-1, -2))
                    mrow = d.argmin(dim=1)
                    max_match_diff[v] = max(max_match_diff[v], float(d.min(dim=1).values.max()))
                    pose = cap["pose"][mrow]
                    pheno = {kk: vv[mrow] for kk, vv in cap["pheno"].items()}
                    if lc0 is None or lc0[next(iter(lc0))].shape[0] != n:
                        lc0 = {kk: torch.zeros(n, device=sess.device) for kk in sx.local_change_labels}
                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        o = sx(pose_parameters=pose, phenotype_kwargs=pheno, local_changes_kwargs=lc0)
                    j3d = P.j3d.float()
                    offset = j3d[:, 0] - bp[:, 0, :3, 3]
                    max_bone_diff[v] = max(max_bone_diff[v], float((o["bone_poses"][:, :, :3, 3] - bp[:, :, :3, 3]).abs().max()))
                    V = o["vertices"].float() + offset[:, None]
                    J55 = torch.einsum("jv,nvc->njc", Jr, V)
                    # projected-joint boxes (full-res OpenCV pixels) against the dancer boxes of this frame
                    uv = torch.einsum("ij,nkj->nki", Kv[0].float(), J55 / J55[..., 2:3].clamp_min(1e-3))[..., :2] / s - 0.5
                    rb = roles.get(str(fr["i"])) or {}
                    for p in range(n):
                        bb = [float(uv[p, :, 0].min()), float(uv[p, :, 1].min()), float(uv[p, :, 0].max()), float(uv[p, :, 1].max())]
                        il = box_iou(bb, rb["lead"]) if rb.get("lead") else 0.0
                        ifo = box_iou(bb, rb["follow"]) if rb.get("follow") else 0.0
                        rows[v].append({"frame": k, "i": fr["i"], "t": fr["t"], "track": int(ids[p]), "conf": float(P.conf[p, 0]),
                                        "transl": P.transl_pelvis[p, 0].float().cpu().numpy() if P.transl_pelvis.dim() == 3
                                        else P.transl_pelvis[p].float().cpu().numpy(),
                                        "pelvis": j3d[p, 0].cpu().numpy(), "j55": J55[p].cpu().numpy(),
                                        "anny_j3d": j3d[p].cpu().numpy(), "shape": P.shape[p].float().cpu().numpy(),
                                        "rotmat": pose[p, :, :3, :3].cpu().numpy(), "offset": offset[p].cpu().numpy(),
                                        "j2d_root": P.j2d[p, 0].float().cpu().numpy(), "iou_lead": il, "iou_follow": ifo,
                                        "box": np.array(bb, np.float32)})
                        if max(il, ifo) > a.verts_iou:
                            verts[v].append((len(rows[v]) - 1, V[p].cpu().numpy().astype(np.float32)))
            times.append(time.time() - t1)
            if k % 10 == 0 or k == len(frames) - 1:
                log(f"frame {k + 1}/{len(frames)} i={fr['i']} {times[-1]:.2f} s; persons ourK {perframe['ourK']['n'][-1]} "
                    f"predK {perframe['predK']['n'][-1]}; f_pred {f_pred[-1]:.0f} px (ours {Kf[0, 0]:.0f})")
        rep["timings"]["frames_s"] = round(float(np.sum(times)), 1)
        rep["timings"]["per_frame_s_median"] = round(float(np.median(times)), 3)
        rep["checks"]["max_bone_head_diff_m"] = max_bone_diff
        rep["checks"]["max_hook_match_diff_m"] = max_match_diff
        rep["f_pred_px"] = {"median": float(np.median(f_pred)), "min": float(np.min(f_pred)), "max": float(np.max(f_pred))}
        rep["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        for v in variants:
            R = rows[v]
            keys = ["frame", "i", "t", "track", "conf", "transl", "pelvis", "j55", "anny_j3d", "shape", "rotmat", "offset",
                    "j2d_root", "iou_lead", "iou_follow", "box"]
            arr = {kk: np.stack([np.asarray(r[kk]) for r in R]) if R else np.zeros((0,)) for kk in keys}
            for kk in ("j55", "anny_j3d", "rotmat", "transl", "pelvis", "offset", "shape", "box"):
                arr[kk] = arr[kk].astype(np.float32)
            arr["verts_det"] = np.array([i for i, _ in verts[v]], np.int64)
            arr["verts"] = np.stack([vv for _, vv in verts[v]]) if verts[v] else np.zeros((0, 10475, 3), np.float32)
            arr["K768"] = np.stack(perframe[v]["K768"]).astype(np.float32)
            arr["n_per_frame"] = np.array(perframe[v]["n"])
            arr["f_pred_full"] = np.array(f_pred, np.float32)
            arr["frame_i"] = np.array([f["i"] for f in frames])
            arr["frame_t"] = np.array([f["t"] for f in frames])
            arr["scale"] = np.float32(s)
            np.savez_compressed(out / f"multihmr2_{v}.npz", **arr)
            ntr = len(set(int(r["track"]) for r in R))
            rep[v] = {"detections": len(R), "tracks": ntr, "verts_kept": len(verts[v]),
                      "persons_per_frame": {"min": int(min(perframe[v]["n"])), "median": float(np.median(perframe[v]["n"])),
                                            "max": int(max(perframe[v]["n"]))}}
            log(v, rep[v])
        rc = 0
    except Exception as e:  # noqa: BLE001
        rep["errors"].append(f"{type(e).__name__}: {e}")
        rep["traceback"] = traceback.format_exc()
        log("FAILED", rep["errors"][-1])
        print(rep["traceback"], flush=True)
        rc = 1
    rep["timings"]["total_s"] = round(time.time() - T0, 1)
    (out / "mhmr2_pod.json").write_text(json.dumps(rep, indent=1, default=str))
    log("done rc", rc, rep["timings"])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
