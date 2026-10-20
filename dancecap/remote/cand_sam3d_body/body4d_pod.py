#!/usr/bin/env python
"""Pod side of mono4d track "sam3d_body", SAM-Body4D variants (dancecap.mono4d.cand_sam3d_body, job cand_sam3d_body4d).

    body4d_pod.py run --input <dir> --out <dir> --body4d-root <dir> --ckpt-root <dir> --tag body4d [--no-completion]
                      [--batch-size 0=auto] [--limit N]

SAM-Body4D (github.com/gaomingqi/sam-body4d @ 21af102, MIT; arXiv 2512.08406) = SAM 3 masklets + Diffusion-VAS
occlusion completion (amodal mask + RGB content completion of occluded frames, kaihuac/diffusion-vas-*, MIT; depth
from Depth-Anything-V2-Large, CC-BY-NC-4.0) + its own fork of SAM 3D Body that runs a batch of frames per identity:
Kalman-smoothed body / hand pose, EMA-smoothed global rotation, MHR shape + scale frozen at the identity's first frame.

SAM 3 (facebook/sam3: gated, access not requested) is BYPASSED: the masklets are OUR SAM 2.1 lead / follow masks,
written in the app's own layout (OUTPUT_DIR/images/%08d.jpg + OUTPUT_DIR/masks/%08d.png, P-mode, 1 = lead (01),
2 = follow (02); the rare overlap pixels go to the lead), and the app's OWN OfflineApp.on_4d_generation runs unchanged,
except that
  - no SAM 3 / ViTDet detector / MoGe-2 FOV model is built (OfflineApp.__init__ is skipped; the attributes it would
    set are set here), so 3DB gets OUR intrinsics: process_image_with_mask is wrapped to pass cam_int = K of
    cameras/06.npz (the app itself would call MoGe-2 per frame);
  - the mesh rendering / .ply export / mp4 are no-ops; the per-frame MHR outputs are captured instead;
  - each identity keeps ITS OWN frozen shape / scale (upstream shares the first identity's with every other one: see
    pim() below).
--no-completion sets completion.enable = false (the app's documented switch: temporal smoothing + frozen shape only).
Diffusion-VAS resolutions are transposed to (512, 256) / (1024, 512) for our PORTRAIT video (the app's landscape
defaults would stretch it 3.5x sideways); --app-resolutions keeps the defaults.
Writes mhr_<tag>.npz in the layout of sam3db_pod.py infer (+ <role>_occluded: frames the app completed), which
sam3db_pod.py convert turns into SMPL-X; body4d_pod.json records keys, timing, memory, occlusion counts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np

ROLES = ("lead", "follow")
OBJ = {"lead": 1, "follow": 2}
KEYS = ("pred_vertices", "pred_cam_t", "focal_length", "pred_keypoints_3d", "pred_keypoints_2d", "pred_joint_coords",
        "global_rot", "body_pose_params", "hand_pose_params", "scale_params", "shape_params", "expr_params",
        "mhr_model_params", "bbox", "lhand_bbox", "rhand_bbox")


def log(msg: str) -> None:
    print(f"[body4d_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def update_report(out: Path, key: str, val) -> None:
    p = out / "body4d_pod.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = val
    p.write_text(json.dumps(d, indent=1, default=str))


def _write_app_inputs(inp: Path, rows: list[dict], role_pid: dict, work: Path, palette) -> dict:
    import cv2
    from PIL import Image
    if work.exists():
        shutil.rmtree(work)
    (work / "images").mkdir(parents=True)
    (work / "masks").mkdir()
    overlap, px = 0, {r: [] for r in ROLES}
    for k, row in enumerate(rows):
        shutil.copy2(inp / "frames" / row["file"], work / "images" / f"{k:08d}.jpg")
        lab = None
        for role in ("follow", "lead"):  # lead written last: it keeps the (rare) pixels both masks claim
            m = cv2.imread(str(inp / "masks" / f"mask_{row['i']:04d}_{role_pid[role]}.png"), cv2.IMREAD_GRAYSCALE) > 127
            if lab is None:
                lab = np.zeros(m.shape, np.uint8)
            if role == "lead":
                overlap += int((m & (lab == OBJ["follow"])).sum())
            lab[m] = OBJ[role]
        for role in ROLES:
            px[role].append(int((lab == OBJ[role]).sum()))
        im = Image.fromarray(lab, mode="P")
        im.putpalette(palette)
        im.save(work / "masks" / f"{k:08d}.png")
    return {"frames": len(rows), "overlap_px_total": overlap,
            "label_px_min": {r: int(min(v)) for r, v in px.items()}}


def import_app(root: Path):
    """scripts/offline_app.py as a module, in the app's own import layout (it appends these itself; here the top dir
    goes first, for its `utils` / `models` packages). Mesh rendering (pyrender / OpenGL, not installed) is never used:
    the two modules that import it are stubbed."""
    sys.path[:0] = [str(root), str(root / "models" / "sam_3d_body"), str(root / "models" / "diffusion_vas"),
                    str(root / "scripts")]
    vis = types.ModuleType("models.sam_3d_body.tools.vis_utils")
    vis.visualize_sample_together = lambda img, *x, **k: img
    vis.visualize_sample = lambda img, *x, **k: []
    sys.modules["models.sam_3d_body.tools.vis_utils"] = vis
    rnd = types.ModuleType("sam_3d_body.visualization.renderer")

    class Renderer:  # imported by the fork's notebook/utils.py (save_mesh_results etc.), never instantiated here
        def __init__(self, *x, **k):
            raise RuntimeError("rendering is stubbed in body4d_pod.py")

    rnd.Renderer = Renderer
    sys.modules["sam_3d_body.visualization.renderer"] = rnd
    os.chdir(root)
    import offline_app  # noqa: E402  (module code: imports + device / TF32 setup only; SAM 3 is imported lazily)
    return offline_app


def check(a) -> None:
    import diffusers
    import transformers
    oa = import_app(Path(a.body4d_root).resolve())
    print(f"[body4d_pod] offline_app import ok ({oa.__file__}); diffusers {diffusers.__version__}, transformers "
          f"{transformers.__version__}; process_image_with_mask from {oa.process_image_with_mask.__module__}", flush=True)


def run(a) -> None:
    import torch
    root = Path(a.body4d_root).resolve()
    ck = Path(a.ckpt_root).resolve()
    inp, out = Path(a.input).resolve(), Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((inp / "meta.json").read_text())
    rows = meta["frames"][: a.limit or None]
    K = np.array(rows[0]["K"], np.float32)
    if any(np.abs(np.array(r["K"], np.float32) - K).max() > 1e-3 for r in rows):
        raise SystemExit("K varies over the window: the app takes one cam_int per call")
    t0 = time.time()
    oa = import_app(root)
    from omegaconf import OmegaConf

    work = Path(a.work).resolve()
    prep = _write_app_inputs(inp, rows, meta["role_pid"], work, oa.DAVIS_PALETTE)
    log(f"app inputs: {prep}")
    comp = not a.no_completion
    if not a.batch_size:  # the app's default 64 (its 80 GB profile: 2 ids fit); smaller on smaller cards
        gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        a.batch_size = 64 if gb >= 70 else 32 if gb >= 40 else 16
    # Diffusion-VAS works on resized frames: the app's (H, W) = (256, 512) / (512, 1024) are landscape. Our phone video
    # is PORTRAIT (1080 x 1920): squashed into 256 x 512 it would be stretched 3.5x sideways, and the completed RGB
    # would come back with 512 of 1920 rows. Same pixel count, transposed to the frame's orientation, unless asked not to.
    h, w = (int(meta.get("height") or 0), int(meta.get("width") or 0))
    portrait = h > w and not a.app_resolutions
    det_res, comp_res = ([512, 256], [1024, 512]) if portrait else ([256, 512], [512, 1024])
    prep["diffusion_vas_resolutions"] = {"detection": det_res, "completion": comp_res,
                                         "why": "transposed for the portrait frame" if portrait else "app defaults"}
    cfg = OmegaConf.create({
        "paths": {"ckpt_root": str(ck)},
        "sam_3d_body": {"ckpt_path": str(ck / "sam-3d-body-dinov3" / "model.ckpt"),
                        "mhr_path": str(ck / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"),
                        "fov_path": "", "batch_size": a.batch_size, "detector_path": "", "segmentor_path": ""},
        "runtime": {"output_dir": str(work.parent)},
        "completion": {"enable": comp, "detection_resolution": det_res, "completion_resolution": comp_res,
                       "model_path_mask": str(ck / "diffusion-vas-amodal-segmentation"),
                       "model_path_rgb": str(ck / "diffusion-vas-content-completion"),
                       "model_path_depth": str(ck / "depth_anything_v2_vitl.pth"), "depth_encoder": "vitl",
                       "max_occ_len": 25}})
    dev = torch.device("cuda")
    model, mcfg = oa.load_sam_3d_body(cfg.sam_3d_body["ckpt_path"], device=dev, mhr_path=cfg.sam_3d_body["mhr_path"])
    # the fork loads with strict=False: a renamed backbone would stay random - check the keys ourselves
    sd = torch.load(cfg.sam_3d_body["ckpt_path"], map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    mk, sk = set(model.state_dict()), set(sd)
    missing = sorted(mk - sk)
    keys = {"model_keys": len(mk), "ckpt_keys": len(sk), "matched": len(mk & sk), "missing": len(missing),
            "unexpected": len(sk - mk), "missing_examples": missing[:8]}
    del sd
    update_report(out, f"{a.tag}_model", {"keys": keys, "load_s": round(time.time() - t0, 1),
                                          "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
    if any(k.startswith("backbone.") and k != "backbone.encoder.mask_token" for k in missing):  # mask_token: training only
        raise SystemExit(f"checkpoint does not cover the fork's backbone: {keys}")
    est = oa.SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=mcfg, human_detector=None, human_segmentor=None,
                                fov_estimator=None)

    app = oa.OfflineApp.__new__(oa.OfflineApp)  # __init__ would build SAM 3 (gated), ViTDet and MoGe-2: not used
    app.CONFIG = cfg
    app.sam3_3d_body_model = est
    if comp:
        t1 = time.time()
        app.pipeline_mask, app.pipeline_rgb, app.depth_model, app.max_occ_len, app.generator = \
            oa.build_diffusion_vas_config(cfg)
        log(f"Diffusion-VAS + Depth-Anything-V2 loaded in {time.time() - t1:.0f} s")
    else:
        app.pipeline_mask = app.pipeline_rgb = app.depth_model = app.max_occ_len = app.generator = None
    app.OUTPUT_DIR = str(work)
    app.RUNTIME = {"batch_size": a.batch_size, "detection_resolution": det_res,
                   "completion_resolution": comp_res, "smpl_export": False, "bboxes": None,
                   "out_obj_ids": [OBJ["lead"], OBJ["follow"]]}

    cam_int = torch.from_numpy(K[None].copy())
    captured = []
    orig = oa.process_image_with_mask

    shape_dicts: dict = {}

    def pim(estimator, image_path, mask_path, idx_path, idx_dict, mhr_shape_scale_dict, occ_dict, **kw):
        """The app's process_image_with_mask, once PER IDENTITY with that identity's own frozen shape / scale.
        Upstream (21af102) passes ONE mhr_shape_scale_dict for all identities, and the fork keys it by the person index
        inside the call (run_inference_batch: `for hid in range(num_human)`), which is always 0 because the app calls
        3DB per identity - so every identity after the first inherits the FIRST identity's MHR shape and scale (here
        the follow would get the lead's). Calling it per identity with a per-identity dict keeps the intent ("keep
        shape & scale same as the first frame", per human) and changes nothing else; our K replaces MoGe-2's."""
        kw.setdefault("cam_int", cam_int)  # OUR intrinsics instead of MoGe-2's per-frame estimate
        occ = {int(k): list(v) for k, v in occ_dict.items()}
        n = len(image_path)
        per = {}
        for oid in sorted(occ_dict, key=int):
            sub_occ = {oid: occ_dict[oid]}
            sub_idx = {k: v for k, v in idx_dict.items() if int(k) == int(oid)}
            sub_path = {k: v for k, v in idx_path.items() if int(k) == int(oid)}
            res = orig(estimator, image_path, mask_path, sub_path, sub_idx, shape_dicts.setdefault(int(oid), {}),
                       sub_occ, **kw)
            final, ids, empty = res
            got, j = {}, 0
            for fi in range(n):
                if fi in empty:
                    continue
                for o_id, o in zip(ids[j], final[j]):
                    got[fi] = (int(o_id), o)
                j += 1
            per[int(oid)] = got
        final, ids, empty = [], [], []
        for fi in range(n):
            fo = [per[oid][fi] for oid in sorted(per) if fi in per[oid]]
            if not fo:
                empty.append(fi)
                continue
            final.append([o for _, o in fo])
            ids.append([oid for oid, _ in fo])
        res = (final, ids, empty)
        captured.append({"images": list(image_path), "occ": occ, "completed": {int(k): list(v) for k, v in
                                                                               idx_dict.items()}, "res": res})
        return res

    oa.process_image_with_mask = pim
    oa.save_mesh_results = lambda *x, **k: None
    oa.jpg_folder_to_mp4 = lambda *x, **k: None
    torch.cuda.reset_peak_memory_stats()
    t2 = time.time()
    app.on_4d_generation()
    run_s = time.time() - t2

    F = len(rows)
    index = {f"{k:08d}.jpg": k for k in range(F)}
    store = {r: {key: [None] * F for key in KEYS} for r in ROLES}
    valid = {r: np.zeros(F, bool) for r in ROLES}
    occluded = {r: np.zeros(F, bool) for r in ROLES}
    for c in captured:
        final, id_batch, empty = c["res"]
        n_empty = 0
        for fi, img_path in enumerate(c["images"]):
            if fi in empty:
                n_empty += 1
                continue
            k = index[os.path.basename(img_path)]
            for oid, o in zip(id_batch[fi - n_empty], final[fi - n_empty]):
                role = "lead" if oid == OBJ["lead"] else "follow"
                for key in KEYS:
                    if key in o and o[key] is not None:
                        store[role][key][k] = np.asarray(o[key], np.float32)
                valid[role][k] = True
        for oid, flags in c["occ"].items():  # 0 = occluded (the app completed this frame for that id)
            role = "lead" if oid == OBJ["lead"] else "follow"
            ks = [index[os.path.basename(p)] for p in c["images"]]
            for k, f in zip(ks, flags):
                occluded[role][k] = f == 0
    arrays = {"frame_i": np.array([r["i"] for r in rows]), "t": np.array([r["t"] for r in rows]),
              "K": np.array([r["K"] for r in rows], np.float32)}
    for r in ROLES:
        arrays[f"{r}_valid"] = valid[r]
        arrays[f"{r}_occluded"] = occluded[r]
        for key in KEYS:
            ref = next((x for x in store[r][key] if x is not None), None)
            if ref is None:
                continue
            arrays[f"{r}_{key}"] = np.stack([x if x is not None else np.full_like(ref, np.nan) for x in store[r][key]])
    np.savez_compressed(out / f"mhr_{a.tag}.npz", **arrays)
    rep = {"completion": comp, "batch_size": a.batch_size, "seconds_4d": round(run_s, 1),
           "seconds_total": round(time.time() - t0, 1), "frames": F,
           "valid": {r: int(valid[r].sum()) for r in ROLES}, "occluded_frames": {r: int(occluded[r].sum()) for r in ROLES},
           "completed_segments": [c["completed"] for c in captured],
           "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "app_inputs": prep,
           "cam_int": K.tolist(), "sam3": "bypassed (our SAM 2.1 masks as masklets)"}
    update_report(out, a.tag, rep)
    log(f"{a.tag}: {rep}")
    if not all(valid[r].any() for r in ROLES):
        raise SystemExit(f"{a.tag}: a dancer has no output")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--input", required=True)
    s.add_argument("--out", dest="output", required=True)
    s.add_argument("--body4d-root", required=True)
    s.add_argument("--ckpt-root", required=True)
    s.add_argument("--work", required=True, help="the app's OUTPUT_DIR (images / masks / completion)")
    s.add_argument("--tag", default="body4d")
    s.add_argument("--no-completion", action="store_true")
    s.add_argument("--batch-size", type=int, default=0, help="0 = 64 on >= 70 GB cards, 32 on >= 40 GB, else 16")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--app-resolutions", action="store_true",
                   help="keep the app's landscape Diffusion-VAS resolutions even for a portrait video")
    c = sub.add_parser("check")
    c.add_argument("--body4d-root", required=True)
    a = ap.parse_args(argv)
    {"run": run, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
