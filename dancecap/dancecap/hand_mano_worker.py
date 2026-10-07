"""MANO hand-mesh regressors on the hand jobs of a take: WiLoR (CVPR 2025) and HaMeR (CVPR 2024). Pod-only for `run`.

    python -m dancecap.hand_mano_worker convert-mano
        local, once: models/mano/MANO_{RIGHT,LEFT}.pkl (chumpy pickles from the MANO site) -> models/mano/nochumpy/
        (plain numpy / scipy arrays, same keys) so the pod needs no chumpy (broken on numpy >= 1.24). Both folders are
        git-ignored and licence-restricted (MANO: non-commercial, no redistribution) - never copy them anywhere public.
    python -m dancecap.hand_mano_worker run <take> --model wilor|hamer --repo DIR [--batch 48] [--limit N] [--no-det]
        hands/jobs.json (refine_hands survey) + frames/ -> hands/mano_<model>.npz (+ .json)

Both models regress MANO (right hand; left hands are run on the mirrored crop and mirrored back) from a 256 px crop
around a hand box. The box ("prompt") comes from the fitted MAMMA hand projected into the frame (tight box of its 21
points); for WiLoR its own YOLO hand detector also runs on the job crop and, when it finds a hand of the right side
near the fitted one, its box is used instead (box_src 1) and its confidence is recorded (the detection rate is the
honest "can the detector see this hand" number). HaMeR has no hand detector of its own (it uses ViTPose whole-body
keypoints + a body detector in its demo), so it always uses the fitted box.

Outputs per job (same order as jobs.json "jobs"):
    pose_aa   (n,15,3)  finger rotations, axis-angle, SMPL-X/MANO joint order (index, middle, pinky, ring, thumb x 3),
                        flat-hand-mean convention (0 = flat hand), already mirrored to the job's side. SMPL-X hand pose
                        is MANO-compatible: same 15 joints, same PCA, template frames within 2.1 deg (checked 2026-10-06).
    orient    (n,3,3)   hand (wrist) rotation in the crop camera frame (~ the real camera's axes), job's side
    betas     (n,10)    MANO shape
    kp2d      (n,21,2)  the regressed hand's 21 joints (OpenPose / COCO-WholeBody hand order) in full-frame pixels of
                        the canonical frame (distorted, like the other detectors; refine_hands undistorts)
    kp3d      (n,21,3)  joints in the crop camera frame (metres, root-relative up to the weak-perspective depth)
    box (n,4), box_src (n,) 0 fitted / 1 detector, det_conf (n,), det_side_ok (n,), det_dist_len (n,), ok (n,)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
WORKSPACE = PKG.parent
MANO_DIR = WORKSPACE / "models" / "mano"
NOCHUMPY = MANO_DIR / "nochumpy"
MIRROR = np.diag([-1.0, 1.0, 1.0])
RESCALE = 2.0          # demo default of both repos: crop side = 2 x the tight hand box (then aspect-expanded)
DET_CONF = 0.15        # WiLoR YOLO floor (demo uses 0.3; we also keep weak detections, with their confidence)
DET_MAX_DIST_LEN = 0.6  # detector box centre within this x hand length of the fitted hand's centre


# ------------------------------------------------------------------------------------------------ MANO files

class _Stub:
    """Stand-in for chumpy objects while unpickling the MANO files."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        self.__dict__.update(st if isinstance(st, dict) else {"_state": st})


class _NoChumpy(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        return super().find_class(module, name)


def _unchumpy(v):
    """chumpy Ch -> its value; chumpy reordering.Select (MANO shapedirs) -> a[idxs] reshaped; anything else as is."""
    if not isinstance(v, _Stub):
        return v
    d = v.__dict__
    if "x" in d and d["x"] is not None:
        return np.asarray(d["x"])
    if "a" in d and "idxs" in d:
        a = np.asarray(_unchumpy(d["a"]))
        out = a.ravel()[np.asarray(d["idxs"])]
        shp = d.get("preferred_shape")
        return out.reshape(shp) if shp is not None else out
    raise ValueError(f"unsupported chumpy object with state {sorted(d)}")


def load_mano_plain(path: Path) -> dict:
    with open(path, "rb") as fh:
        d = _NoChumpy(fh, encoding="latin1").load()
    return {k: _unchumpy(v) for k, v in d.items()}


def convert_mano(src_dir: Path = MANO_DIR, out_dir: Path = NOCHUMPY) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for side in ("RIGHT", "LEFT"):
        src = src_dir / f"MANO_{side}.pkl"
        if not src.exists():
            continue
        d = load_mano_plain(src)
        bad = [k for k, v in d.items() if isinstance(v, _Stub)]
        if bad:
            raise ValueError(f"{src}: unconverted keys {bad}")
        dst = out_dir / src.name
        with open(dst, "wb") as fh:
            pickle.dump(d, fh, protocol=2)
        done.append(dst)
    (out_dir / "README.txt").write_text(
        "MANO model files converted to plain numpy (no chumpy) by dancecap.hand_mano_worker convert-mano.\n"
        "Derived from the user's licensed MANO download (mano.is.tue.mpg.de): non-commercial research only, do not\n"
        "redistribute, never copy into a repository or public location. Pod bundles holding them use SECURE cloud.\n")
    return done


# ------------------------------------------------------------------------------------------------ geometry helpers

def mirror_rotmats(R: np.ndarray) -> np.ndarray:
    """Right-hand <-> left-hand rotation (x-mirror conjugation): M R M, M = diag(-1, 1, 1). Axis-angle (x,y,z) ->
    (x,-y,-z) - the SMPL-X / MANO left-right hand-pose convention."""
    return MIRROR @ R @ MIRROR


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    shp = R.shape[:-2]
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*shp, 3)


def tight_box(px: np.ndarray, pad: float = 0.1) -> np.ndarray:
    """Tight square-ish box around projected hand points (+pad x size on each side)."""
    lo, hi = np.nanmin(px, 0), np.nanmax(px, 0)
    c, s = (lo + hi) / 2, max(float((hi - lo).max()), 4.0) * (1 + 2 * pad)
    return np.array([c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2], np.float32)


def crop_kp_to_image(kp_norm: np.ndarray, center: np.ndarray, size: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The repos' pred_keypoints_2d (normalised crop coords, crop centre = 0, crop side = 1; computed on the mirrored
    crop for left hands) -> full-image pixels."""
    kp = np.array(kp_norm, np.float64)
    sgn = np.where(np.asarray(right) > 0.5, 1.0, -1.0)
    kp[..., 0] *= sgn[:, None]
    return center[:, None, :] + kp * np.asarray(size, np.float64).reshape(-1, 1, 1)


# ------------------------------------------------------------------------------------------------ models (pod)

def _load_model(name: str, repo: Path):
    """chdir into the repo (both use relative data paths) and load the checkpoint. -> (model, cfg, ViTDetDataset)"""
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if name == "wilor":
        from wilor.models import load_wilor
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                                cfg_path="./pretrained_models/model_config.yaml")
    elif name == "hamer":
        from hamer.models import HAMER, load_hamer
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        # HAMER.__init__ builds pyrender renderers by default (init_renderer=True), which needs an EGL context the pod
        # does not grant (GLError, 2026-10-06); inference never renders, so load without them
        orig = HAMER.load_from_checkpoint
        HAMER.load_from_checkpoint = lambda *a, **k: orig(*a, init_renderer=False, **k)
        model, cfg = load_hamer("./_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
    else:
        raise ValueError(name)
    return model, cfg, ViTDetDataset


def _detect(det, img, jobs_here, size=320):
    """WiLoR YOLO hand detector on each job crop (upscaled to `size`). -> per job (box full px | None, conf, side_ok,
    dist in hand lengths)."""
    import cv2
    crops, Ts = [], []
    for j in jobs_here:
        x0, y0, x1, y1 = j["box"]
        s = size / max(x1 - x0, 1e-6)
        A = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)
        crops.append(cv2.warpAffine(img, A, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT))
        Ts.append((s, x0, y0))
    res = det(crops, conf=DET_CONF, verbose=False, imgsz=size)
    out = []
    for j, r, (s, x0, y0) in zip(jobs_here, res, Ts):
        mpx = np.asarray(j["model_px"], float)
        mc = (np.nanmin(mpx, 0) + np.nanmax(mpx, 0)) / 2
        L = max(float(j["len_px"]), 4.0)
        best = None
        for b in r.boxes:
            xyxy = b.xyxy.cpu().numpy().reshape(-1)[:4] / s + np.array([x0, y0, x0, y0])
            c = (xyxy[:2] + xyxy[2:]) / 2
            d = float(np.linalg.norm(c - mc)) / L
            side_ok = int(b.cls.item()) == (1 if j["side"] == "right" else 0)
            key = (not side_ok, d)
            if best is None or key < best[0]:
                best = (key, xyxy.astype(np.float32), float(b.conf.item()), side_ok, d)
        out.append((None, 0.0, False, np.inf) if best is None else best[1:])
    return out


def run(take: str, model_name: str, repo: Path, batch: int = 48, limit: int | None = None, use_det: bool = True,
        device: str = "cuda") -> Path:
    import cv2
    import torch
    from torch.utils.data import default_collate
    work = PKG / "work" / take
    out_p = work / "hands" / f"mano_{model_name}.npz"
    J = json.loads((work / "hands" / "jobs.json").read_text())
    jobs = J["jobs"][:limit] if limit else J["jobs"]
    n = len(jobs)
    t0 = time.time()
    model, cfg, Dataset = _load_model(model_name, repo)
    model = model.to(device).eval()
    det = None
    if use_det and model_name == "wilor":
        from ultralytics import YOLO
        det = YOLO("./pretrained_models/detector.pt")
    print(f"[mano_worker] {model_name} loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}; "
          f"{n} hand jobs", flush=True)
    R = {"job_cam": np.array([j["cam"] for j in jobs]), "job_i": np.array([j["i"] for j in jobs], np.int32),
         "job_role": np.array([j["role"] for j in jobs]), "job_side": np.array([j["side"] for j in jobs]),
         "len_px": np.array([j["len_px"] for j in jobs], np.float32),
         "ok": np.zeros(n, bool), "pose_aa": np.zeros((n, 15, 3), np.float32), "orient": np.zeros((n, 3, 3), np.float32),
         "betas": np.zeros((n, 10), np.float32), "kp2d": np.full((n, 21, 2), np.nan, np.float32),
         "kp3d": np.full((n, 21, 3), np.nan, np.float32), "cam": np.zeros((n, 3), np.float32),
         "box": np.zeros((n, 4), np.float32), "box_src": np.zeros(n, np.int8), "det_conf": np.zeros(n, np.float32),
         "det_side_ok": np.zeros(n, bool), "det_dist_len": np.full(n, np.inf, np.float32)}
    by_frame: dict = {}
    for k, j in enumerate(jobs):
        by_frame.setdefault((j["cam"], int(j["i"])), []).append(k)
    pending: list = []   # (job index, dataset item)

    def flush():
        if not pending:
            return
        ks = [k for k, _ in pending]
        b = default_collate([it for _, it in pending])
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            o = model(b)
        mp = o["pred_mano_params"]
        go = mp["global_orient"].reshape(-1, 3, 3).float().cpu().numpy()
        hp = mp["hand_pose"].reshape(-1, 15, 3, 3).float().cpu().numpy()
        right = b["right"].float().cpu().numpy()
        kp2 = crop_kp_to_image(o["pred_keypoints_2d"].float().cpu().numpy(), b["box_center"].float().cpu().numpy(),
                               b["box_size"].float().cpu().numpy(), right)
        kp3 = o["pred_keypoints_3d"].float().cpu().numpy()
        for m, k in enumerate(ks):
            if right[m] < 0.5:   # mirrored crop -> back to the left hand
                go[m], hp[m] = mirror_rotmats(go[m]), mirror_rotmats(hp[m])
                kp3[m, :, 0] *= -1
            R["orient"][k], R["pose_aa"][k] = go[m], rotmat_to_aa(hp[m])
            R["betas"][k] = mp["betas"][m].float().cpu().numpy()
            R["kp2d"][k], R["kp3d"][k] = kp2[m], kp3[m]
            R["cam"][k] = o["pred_cam"][m].float().cpu().numpy()
            R["ok"][k] = True
        pending.clear()

    done = 0
    for c, (cam, i) in enumerate(sorted(by_frame)):
        ks = by_frame[(cam, i)]
        img = cv2.imread(str(work / "frames" / cam / f"{i:06d}.jpg"))
        if img is None:
            continue
        dets = _detect(det, img, [jobs[k] for k in ks]) if det is not None else [(None, 0.0, False, np.inf)] * len(ks)
        boxes, rights = [], []
        for k, (dbox, conf, side_ok, dist) in zip(ks, dets):
            j = jobs[k]
            R["det_conf"][k], R["det_side_ok"][k], R["det_dist_len"][k] = conf, side_ok, dist
            use = dbox is not None and side_ok and dist <= DET_MAX_DIST_LEN
            box = dbox if use else tight_box(np.asarray(j["model_px"], float))
            R["box"][k], R["box_src"][k] = box, int(use)
            boxes.append(box)
            rights.append(1.0 if j["side"] == "right" else 0.0)
        ds = Dataset(cfg, img, np.asarray(boxes, np.float32), np.asarray(rights, np.float32), rescale_factor=RESCALE)
        for m, k in enumerate(ks):
            pending.append((k, ds[m]))
        if len(pending) >= batch:
            flush()
        done += len(ks)
        if c % 200 == 0:
            print(f"[mano_worker] {model_name} {done}/{n} {time.time() - t0:.0f}s", flush=True)
    flush()
    np.savez_compressed(out_p, **R)
    det_rate = {}
    for cam in sorted(set(R["job_cam"].tolist())):
        sel = R["job_cam"] == cam
        det_rate[cam] = round(float((R["box_src"][sel] == 1).mean()), 3)
    info = {"take": take, "model": model_name, "jobs": n, "ok": int(R["ok"].sum()), "seconds": round(time.time() - t0, 1),
            "gpu": torch.cuda.get_device_name(0), "rescale": RESCALE, "detector_used": det is not None,
            "detector_box_rate_per_cam": det_rate if det is not None else None}
    out_p.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(f"[mano_worker] {model_name}: {info}", flush=True)
    return out_p


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("convert-mano")
    r = sub.add_parser("run")
    r.add_argument("take")
    r.add_argument("--model", choices=["wilor", "hamer"], required=True)
    r.add_argument("--repo", required=True)
    r.add_argument("--batch", type=int, default=48)
    r.add_argument("--limit", type=int)
    r.add_argument("--no-det", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "convert-mano":
        for p in convert_mano():
            print("wrote", p)
        return 0
    run(a.take, a.model, Path(a.repo), a.batch, a.limit, not a.no_det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
