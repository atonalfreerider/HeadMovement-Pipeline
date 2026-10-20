#!/usr/bin/env python
"""Pod side of mono4d track "sam3d_body" (dancecap.mono4d.cand_sam3d_body).

    sam3db_pod.py infer   --input <dir> --out <dir> --variants mask255,mask01,nomask [--limit N]
    sam3db_pod.py convert --input <dir> --out <dir> --variants mask255,mask01,nomask [--single-identity mask255]

infer    SAM 3D Body (facebookresearch/sam-3d-body, SAM License) on the window's canonical frames. Each dancer is
         prompted separately with OUR SAM 2.1 mask (input/masks/mask_{i:04d}_{01|02}.png) and the tight box of that
         mask, and our intrinsics (cam_int = K of cameras/06.npz). No detector / segmentor / FOV estimator. Variants:
         mask255 (mask 0/255, the official notebook's external-mask recipe), mask01 (mask 0/1, the demo's SAM 2 path),
         nomask (same box, no mask). Writes mhr_<variant>.npz (MHR outputs per role per frame, camera frame, metres)
         and mhr_faces.npy (the MHR LOD1 faces of the 3DB head).
convert  The official MHR -> SMPL-X converter (facebookresearch/MHR tools/mhr_smpl_conversion, Apache-2.0):
         Conversion(method="pytorch").convert_sam3d_output_to_smpl on pred_vertices + pred_cam_t (per-frame identity),
         and optionally convert_mhr2smpl(single_identity=True) per dancer. SMPL-X joints = smplx forward of the fitted
         parameters (joints[:, :55], camera frame). Writes smplx_<variant>.npz (+ smplx_<variant>_1id.npz).
         The converter only needs the MHR model for its LOD1 faces: MHR.from_files (pymomentum) is tried first, else
         stubs carry the faces of the tool's own assets/mhr_face_mask.ply (the full LOD1 mesh); both are checked
         against the 3DB head's faces. sam3db_pod.json records which path ran. The pymomentum path SEGFAULTS in the
         pod venv (rc 139, session 20261019-175225-batch): the jobs pass --force-stub.
Never prints a token: the gated checkpoint was downloaded by install.sh with HF_TOKEN from the environment.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("S3DB_ROOT", "/workspace/s3db_cand"))
ROLES = ("lead", "follow")
TRAIN_ONLY_KEYS = ("backbone.encoder.mask_token",)
MHR_KEYS = ("pred_vertices", "pred_cam_t", "focal_length", "pred_keypoints_3d", "pred_keypoints_2d", "pred_joint_coords",
            "global_rot", "body_pose_params", "hand_pose_params", "scale_params", "shape_params", "expr_params",
            "mhr_model_params", "bbox", "lhand_bbox", "rhand_bbox")


def log(msg: str) -> None:
    print(f"[sam3db_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def update_report(out: Path, key: str, val) -> None:
    p = out / "sam3db_pod.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = val
    p.write_text(json.dumps(d, indent=1, default=str))


def _cfg_get(cfg, path: str, default=None):
    node = cfg
    for k in path.split("."):
        try:
            node = node[k] if k in node else default
        except Exception:  # noqa: BLE001
            return default
        if node is default:
            return default
    return node


# ---------------------------------------------------------------------------------------------------- infer


def infer(a) -> None:
    os.environ["MOMENTUM_ENABLED"] = "0"  # any value: the 3DB MHR head uses the TorchScript assets/mhr_model.pt
    import cv2
    import torch
    sys.path.insert(0, str(ROOT / "sam-3d-body"))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    inp, out = Path(a.input).resolve(), Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((inp / "meta.json").read_text())
    rows = meta["frames"][: a.limit or None]
    ck = ROOT / "ckpt" / "sam-3d-body-dinov3"
    t0 = time.time()
    model, cfg = load_sam_3d_body(str(ck / "model.ckpt"), device="cuda", mhr_path=str(ck / "assets" / "mhr_model.pt"))
    load_s = time.time() - t0
    # load_state_dict(strict=False) would leave a renamed backbone silently random: check the keys ourselves
    sd = torch.load(str(ck / "model.ckpt"), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    mk, sk = set(model.state_dict()), set(sd)
    missing, unexpected = sorted(mk - sk), sorted(sk - mk)
    group = lambda ks: {g: sum(1 for k in ks if k.startswith(g + ".") or k == g)  # noqa: E731
                        for g in sorted({".".join(k.split(".")[:2]) for k in ks})}
    keys = {"model_keys": len(mk), "ckpt_keys": len(sk), "matched": len(mk & sk), "missing": len(missing),
            "unexpected": len(unexpected), "missing_groups": group(missing), "unexpected_groups": group(unexpected)}
    del sd
    # fatal only for the backbone (a torch.hub code/ckpt name mismatch would leave it random); the rest is reported.
    # Expected and harmless (run 20261019-172933: 1127/1127 checkpoint keys matched, 0 unexpected): the MHR head's
    # buffers (head_pose*.mhr.*, hand_pose_comps_ori - built from assets/mhr_model.pt) and DINOv3's mask_token, a
    # masked-image-modelling TRAINING parameter that inference never reads (only used when masks are passed).
    bad = [k for k in missing if k.startswith("backbone.") and k not in TRAIN_ONLY_KEYS]
    cfg_info = {k: str(_cfg_get(cfg, k)) for k in (
        "MODEL.BACKBONE.TYPE", "MODEL.BACKBONE.NAME", "MODEL.IMAGE_SIZE", "MODEL.DECODER.CONDITION_TYPE",
        "MODEL.DECODER.USE_INTRIN_CENTER", "MODEL.PROMPT_ENCODER.MASK_EMBED_TYPE", "MODEL.PROMPT_ENCODER.MASK_PROMPT",
        "MODEL.MHR_HEAD.MHR_MODEL_PATH")}
    log(f"model loaded in {load_s:.0f} s; keys {keys}; cfg {cfg_info}")
    update_report(out, "model", {"load_s": round(load_s, 1), "keys": keys, "cfg": cfg_info,
                                 "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                 "cuda": torch.version.cuda, "momentum_enabled_env": os.environ["MOMENTUM_ENABLED"]})
    if bad:
        raise SystemExit(f"checkpoint does not cover the network: {len(bad)} missing keys, e.g. {bad[:5]}")
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    np.save(out / "mhr_faces.npy", np.asarray(est.faces, np.int64))
    masks_cache = {}

    def mask(i: int, role: str):
        key = (i, role)
        if key not in masks_cache:
            m = cv2.imread(str(inp / "masks" / f"mask_{i:04d}_{meta['role_pid'][role]}.png"), cv2.IMREAD_GRAYSCALE)
            masks_cache[key] = (m > 127).astype(np.uint8)
        return masks_cache[key]

    report = {}
    for var in a.variants.split(","):
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        F = len(rows)
        store = {r: {k: [None] * F for k in MHR_KEYS} for r in ROLES}
        valid = {r: np.zeros(F, bool) for r in ROLES}
        first_log = ""
        for k, row in enumerate(rows):
            img = cv2.imread(str(inp / "frames" / row["file"]))[:, :, ::-1].copy()  # RGB, as the estimator expects
            use = [r for r in ROLES if row["mask_boxes"].get(r)]
            if not use:
                continue
            boxes = np.array([row["mask_boxes"][r] for r in use], np.float32)
            masks = None
            if var != "nomask":
                scale = 255 if var == "mask255" else 1
                masks = np.stack([mask(row["i"], r) * scale for r in use]).astype(np.uint8)
            cam_int = torch.tensor(np.array(row["K"], np.float32)[None])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = est.process_one_image(img, bboxes=boxes, masks=masks, cam_int=cam_int, inference_type="full")
            if k == 0:
                first_log = buf.getvalue()[-2000:]
            if len(res) != len(use):
                log(f"{var} frame {row['i']}: {len(res)} outputs for {len(use)} prompts")
                continue
            for r, o in zip(use, res):
                for key in MHR_KEYS:
                    store[r][key][k] = np.asarray(o[key], np.float32)
                valid[r][k] = True
            if k % 30 == 0:
                log(f"{var}: frame {k + 1}/{F}")
        arrays = {"frame_i": np.array([r["i"] for r in rows]), "t": np.array([r["t"] for r in rows]),
                  "K": np.array([r["K"] for r in rows], np.float32)}
        for r in ROLES:
            arrays[f"{r}_valid"] = valid[r]
            for key in MHR_KEYS:
                ref = next((x for x in store[r][key] if x is not None), None)
                if ref is None:
                    continue
                arrays[f"{r}_{key}"] = np.stack([x if x is not None else np.full_like(ref, np.nan)
                                                 for x in store[r][key]])
        np.savez_compressed(out / f"mhr_{var}.npz", **arrays)
        dt = time.time() - t1
        report[var] = {"seconds": round(dt, 1), "frames": F, "valid": {r: int(valid[r].sum()) for r in ROLES},
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "first_frame_log": first_log}
        log(f"{var}: {F} frames in {dt:.0f} s, valid {report[var]['valid']}, peak {report[var]['peak_gb']} GB")
        update_report(out, "variants", report)


# ---------------------------------------------------------------------------------------------------- convert


class _Anything:
    """Stand-in for pymomentum objects the converter only touches on the pymomentum path (never called here)."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        raise RuntimeError("pymomentum stub called: this conversion path needs the real pymomentum")

    def __getattr__(self, name):
        return _Anything()


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything


def _install_stubs(faces: np.ndarray) -> None:
    import torch
    for name in ("pymomentum", "pymomentum.geometry", "pymomentum.solver", "pymomentum.torch",
                 "pymomentum.torch.character"):
        sys.modules[name] = _StubModule(name)
    sys.modules["pymomentum"].geometry = sys.modules["pymomentum.geometry"]
    sys.modules["pymomentum"].solver = sys.modules["pymomentum.solver"]
    sys.modules["pymomentum"].torch = sys.modules["pymomentum.torch"]
    sys.modules["pymomentum.torch"].character = sys.modules["pymomentum.torch.character"]

    class _Mesh:
        pass

    class _Character:
        pass

    class MHR(torch.nn.Module):  # only .character.mesh.faces and .to() are used by convert_mhr2smpl (pytorch)
        def __init__(self):
            super().__init__()
            self.character = _Character()
            self.character.mesh = _Mesh()
            self.character.mesh.faces = faces

    import mhr  # the real package (face expression names only): mhr.mhr is replaced
    mod = types.ModuleType("mhr.mhr")
    mod.MHR = MHR
    sys.modules["mhr.mhr"] = mod
    mhr.mhr = mod


def _mhr_model(mhr_root: Path, ply_faces: np.ndarray, force_stub: bool):
    import torch
    if not force_stub:
        try:
            from mhr.mhr import MHR
            m = MHR.from_files(folder=mhr_root / "assets", device=torch.device("cpu"), lod=1)
            return m, "MHR.from_files (pymomentum, LOD1)", np.asarray(m.character.mesh.faces)
        except Exception as e:  # noqa: BLE001
            reason = f"{type(e).__name__}: {str(e)[:300]}"
            for k in [k for k in sys.modules if k == "mhr.mhr" or k.startswith("pymomentum")]:
                sys.modules.pop(k, None)
    else:
        reason = "forced"
    _install_stubs(ply_faces)
    from mhr.mhr import MHR
    return MHR(), f"stub MHR with the faces of assets/mhr_face_mask.ply ({reason})", ply_faces


def convert(a) -> None:
    import torch
    inp, out = Path(a.input).resolve(), Path(a.output).resolve()
    mhr_root = Path(a.mhr_root or (ROOT / "MHR")).resolve()
    tools = mhr_root / "tools" / "mhr_smpl_conversion"
    sys.path[:0] = [str(tools), str(mhr_root)]
    os.chdir(tools)  # the tool's asset paths are relative ("./assets/...")
    import smplx
    import trimesh
    t0 = time.time()
    ply_faces = np.asarray(trimesh.load("assets/mhr_face_mask.ply", process=False).faces, np.int64)
    mhr_model, mode, faces = _mhr_model(mhr_root, ply_faces, a.force_stub)
    from conversion import Conversion
    from utils import get_batched_parameters
    head_faces = np.load(out / "mhr_faces.npy") if (out / "mhr_faces.npy").exists() else None
    checks = {"faces_converter_eq_ply": bool(np.array_equal(np.asarray(faces, np.int64), ply_faces)),
              "faces_converter_eq_3db_head": None if head_faces is None else
              bool(np.array_equal(np.asarray(faces, np.int64), head_faces)), "n_faces": int(len(faces))}
    log(f"MHR model: {mode}; {checks}")
    if head_faces is not None and not checks["faces_converter_eq_3db_head"]:
        raise SystemExit("the converter's MHR faces differ from the 3DB head's: the barycentric map would be wrong")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    smplx_model = smplx.SMPLX(model_path=str(inp / "smplx" / "SMPLX_NEUTRAL.npz"), gender="neutral", use_pca=False,
                              flat_hand_mean=True).to(dev)
    conv = Conversion(mhr_model=mhr_model, smpl_model=smplx_model, method="pytorch")
    # a later convert run (the SAM-Body4D job) adds its variants instead of replacing the report
    rp = out / "sam3db_pod.json"
    prev = json.loads(rp.read_text()).get("convert", {}) if rp.exists() else {}
    rep = {**prev, "mhr_model": mode, "checks": checks, "device": dev, "smplx": "SMPL-X 2020 neutral, 10 betas, "
           "10 expr, use_pca False, flat_hand_mean True (example.py settings)"}

    def joints_of(params: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
        J, V = [], []
        for s in range(0, n, 128):
            e = min(n, s + 128)
            b = get_batched_parameters(params, s, e, dev, "smpl")
            with torch.no_grad():
                o = smplx_model(**b)
            J.append(o.joints[:, :55].cpu().numpy())
            V.append(o.vertices.cpu().numpy())
        return np.concatenate(J), np.concatenate(V)

    def target_of(lst: list[dict]) -> np.ndarray:
        mv = np.stack([100.0 * (d["pred_vertices"] + d["pred_cam_t"][None]) for d in lst])  # cm, as the tool does
        with torch.no_grad():
            return conv._compute_target_vertices(torch.from_numpy(mv).float().to(dev), "mhr2smpl").cpu().numpy()

    jobs = [(v, False) for v in a.variants.split(",")]
    if a.single_identity:
        jobs += [(v, True) for v in a.single_identity.split(",")]
    def one(var: str, one_id: bool) -> None:
        tag = f"{var}_1id" if one_id else var
        p = out / f"mhr_{var}.npz"
        if not p.exists():
            log(f"{p.name} missing: {tag} skipped")
            return
        t1 = time.time()
        z = np.load(p)
        res_arrays = {"frame_i": z["frame_i"], "t": z["t"]}
        info = {}
        if not one_id:  # the documented SAM3D path: all persons and frames in one call, own identity per frame
            lst, where = [], []
            for r in ROLES:
                for k in np.nonzero(z[f"{r}_valid"])[0]:
                    lst.append({key: z[f"{r}_{key}"][k] for key in
                                ("pred_vertices", "pred_cam_t", "mhr_model_params", "shape_params", "expr_params")})
                    where.append((r, int(k)))
            res = conv.convert_sam3d_output_to_smpl(sam3d_outputs=lst, return_smpl_parameters=True,
                                                    return_smpl_vertices=False, return_fitting_errors=True,
                                                    batch_size=256)
            groups = {r: [n for n, (rr, _) in enumerate(where) if rr == r] for r in ROLES}
            params_all, err_all, tgt_all = res.result_parameters, np.asarray(res.result_errors), target_of(lst)
        for r in ROLES:
            F = len(z["frame_i"])
            val = np.asarray(z[f"{r}_valid"])
            if one_id:  # per dancer: one SMPL-X shape over the window (convert_mhr2smpl, single_identity=True)
                ks = np.nonzero(val)[0]
                lst = [{key: z[f"{r}_{key}"][k] for key in ("pred_vertices", "pred_cam_t")} for k in ks]
                mv = np.stack([100.0 * (d["pred_vertices"] + d["pred_cam_t"][None]) for d in lst])
                res = conv.convert_mhr2smpl(mhr_vertices=torch.from_numpy(mv).float(), single_identity=True,
                                            is_tracking=False, return_smpl_parameters=True, return_fitting_errors=True,
                                            batch_size=256)
                params, err, tgt = res.result_parameters, np.asarray(res.result_errors), target_of(lst)
            else:
                idx = torch.tensor(groups[r], device=params_all["betas"].device)
                params = {k: (v[idx] if v.shape[0] == len(lst) else v) for k, v in params_all.items()}
                ks = np.array([where[n][1] for n in groups[r]], int)
                err, tgt = err_all[groups[r]], tgt_all[groups[r]]
            J, V = joints_of(params, len(ks))
            vdist = np.linalg.norm(V - tgt, axis=-1).mean(1) * 100.0  # cm, fitted SMPL-X vs mapped target vertices
            jf = np.full((F, 55, 3), np.nan, np.float32)
            jf[ks] = J
            ef = np.full(F, np.nan, np.float32)
            ef[ks] = vdist
            res_arrays[f"{r}_joints"] = jf
            res_arrays[f"{r}_fit_err_cm"] = ef
            res_arrays[f"{r}_tool_err"] = np.full(F, np.nan, np.float32)
            res_arrays[f"{r}_tool_err"][ks] = err
            for k, v in params.items():
                v = v.detach().cpu().numpy()
                if v.shape[0] == len(ks):
                    a_ = np.full((F,) + v.shape[1:], np.nan, np.float32)
                    a_[ks] = v.reshape((len(ks),) + v.shape[1:])
                    res_arrays[f"{r}_smplx_{k}"] = a_
                else:
                    res_arrays[f"{r}_smplx_{k}"] = v
            info[r] = {"frames": int(len(ks)), "fit_err_cm_median": float(np.median(vdist)),
                       "fit_err_cm_p90": float(np.percentile(vdist, 90)), "tool_err_median": float(np.median(err))}
        np.savez_compressed(out / f"smplx_{tag}.npz", **res_arrays)
        info["seconds"] = round(time.time() - t1, 1)
        rep[tag] = info
        log(f"{tag}: {info}")
        update_report(out, "convert", {**rep, "total_s": round(time.time() - t0, 1)})

    failed = {}
    for var, one_id in jobs:  # one failing variant never costs the others
        tag = f"{var}_1id" if one_id else var
        try:
            one(var, one_id)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failed[tag] = f"{type(e).__name__}: {str(e)[:400]}"
            log(f"{tag} FAILED: {failed[tag]}")
            update_report(out, "convert_failed", failed)
    if failed and len(failed) == len(jobs):
        raise SystemExit(f"every conversion failed: {failed}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("infer", "convert"):
        s = sub.add_parser(name)
        s.add_argument("--input", required=True)
        s.add_argument("--out", dest="output", required=True)
        s.add_argument("--variants", default="mask255,mask01,nomask")
        if name == "infer":
            s.add_argument("--limit", type=int, default=0)
        else:
            s.add_argument("--single-identity", default="")
            s.add_argument("--mhr-root", default="")
            s.add_argument("--force-stub", action="store_true")
    a = ap.parse_args(argv)
    {"infer": infer, "convert": convert}[a.cmd](a)


if __name__ == "__main__":
    main()
