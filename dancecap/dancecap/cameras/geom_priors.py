"""Monocular metric-geometry priors on a few frames of a video (runs on a GPU pod; pip-installs what it needs).

    python -m dancecap.cameras.geom_priors <video> --frames 15,90,... --out <dir> [--models moge,dav2]

For every requested video frame and model it stores (npz per model) a downsampled depth / point map and the focal the
model believes in. These are PRIORS for a camera that cannot triangulate (a single phone video): a metric scale and a
field of view to cross-check the focal that the architecture (vanishing points) and the dancers' statures give. They are
never the calibration itself.

  moge   MoGe-2 (Ruicheng/moge-2-vitl-normal): metric point map in camera coordinates, normals, mask, intrinsics normalised
         by the image size (fx = f_px / W). Stored: points (h, w, 3) float16 at <= 384 px on the long side, normal, mask,
         depth, fov_x_deg.
  dav2   Depth-Anything-V2 metric indoor (transformers): metric depth only, no focal.

Output <dir>/<model>.npz: frame (n,), size (2,), then per-frame arrays stacked along axis 0 (frames are all the same size).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[geom {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_frames(video: str, frames: list[int]) -> dict[int, np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(video)
    out = {}
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, im = cap.read()
        if ok:
            out[fi] = im
    cap.release()
    return out


def pip_install(*pkgs: str, deps: bool = True) -> None:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *(() if deps else ("--no-deps",)), *pkgs], capture_output=True, text=True)
    if r.returncode:
        log(f"pip install {' '.join(pkgs)} failed: {r.stderr[-400:]}")
        raise RuntimeError("pip install failed")


def ensure(import_stmt: str, spec: str, extra: tuple[str, ...] = ()) -> None:
    """Make `import_stmt` work. A package that would drag in another torch is installed with --no-deps (the farm environment's torch
    must stay as it is: its other modules and the running jobs share it); the modules it still misses are installed one by one."""
    import importlib
    for _ in range(12):
        try:
            exec(import_stmt, {})
            return
        except ModuleNotFoundError as e:
            miss = (e.name or "").split(".")[0]
            if miss in ("moge", "vggt") or not extra and miss == "":
                pip_install(spec, deps=False)
                importlib.invalidate_caches()
                continue
            pip_install({"cv2": "opencv-python-headless", "PIL": "pillow", "utils3d": "utils3d_moge"}.get(miss, miss), deps=False)
            importlib.invalidate_caches()
        except ImportError:
            pip_install(spec, deps=False)
            importlib.invalidate_caches()
    exec(import_stmt, {})


def small(a: np.ndarray, long_side: int) -> np.ndarray:
    import cv2
    h, w = a.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return a
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    if a.dtype == bool:
        return cv2.resize(a.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return cv2.resize(a, size, interpolation=cv2.INTER_AREA)


def run_moge(imgs: dict[int, np.ndarray], out: Path, name: str = "Ruicheng/moge-2-vitl-normal") -> dict:
    import torch
    from moge.model.v2 import MoGeModel      # installed by the orchestrator (single_cam_pod: --no-deps, pinned commits)
    dev = torch.device("cuda")
    model = MoGeModel.from_pretrained(name).to(dev).eval()
    rows = {"frame": [], "points": [], "normal": [], "mask": [], "depth": [], "fov_x_deg": [], "intrinsics": [], "metric_scale": []}
    for fi, im in imgs.items():
        import cv2
        h, w = im.shape[:2]
        t = torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.inference_mode():
            o = model.infer(t)
        K = o["intrinsics"].cpu().numpy()
        fx = float(K[0, 0]) * w
        rows["frame"].append(fi)
        rows["intrinsics"].append(K)
        rows["fov_x_deg"].append(float(np.degrees(2 * np.arctan(0.5 / K[0, 0]))))
        rows["points"].append(small(o["points"].cpu().numpy(), 384).astype(np.float16))
        rows["depth"].append(small(o["depth"].cpu().numpy(), 384).astype(np.float16))
        rows["mask"].append(small(o["mask"].cpu().numpy().astype(bool), 384))
        if "normal" in o:
            rows["normal"].append(small(o["normal"].cpu().numpy(), 384).astype(np.float16))
        rows["metric_scale"].append(float(o["metric_scale"]) if "metric_scale" in o else 1.0)
        log(f"moge frame {fi}: fx {fx:.0f} px of {w}x{h}, fov_x {rows['fov_x_deg'][-1]:.1f} deg")
        del o
    np.savez_compressed(out / "moge.npz", size=np.array([w, h]), **{k: np.array(v) for k, v in rows.items() if v})
    return {"frames": len(rows["frame"])}


def run_dav2(imgs: dict[int, np.ndarray], out: Path, name: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf") -> dict:
    import cv2
    import torch
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], capture_output=True)     # its binary does not match the env's torch
    pip_install("transformers")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from PIL import Image
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForDepthEstimation.from_pretrained(name).cuda().eval()
    rows = {"frame": [], "depth": []}
    for fi, im in imgs.items():
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        inp = {k: v.cuda() for k, v in proc(images=pil, return_tensors="pt").items()}
        with torch.inference_mode():
            d = model(**inp).predicted_depth
        d = torch.nn.functional.interpolate(d[None], size=im.shape[:2], mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
        rows["frame"].append(fi)
        rows["depth"].append(small(d, 384).astype(np.float16))
        log(f"dav2 frame {fi}: depth median {np.median(d):.2f} m")
    np.savez_compressed(out / "dav2.npz", size=np.array([im.shape[1], im.shape[0]]), **{k: np.array(v) for k, v in rows.items()})
    return {"frames": len(rows["frame"])}


VGGT_GROUPS: list[list[int]] = []


def run_vggt(imgs: dict[int, np.ndarray], out: Path, name: str = "facebook/VGGT-1B", group: int = 14) -> dict:
    """VGGT on groups of frames of ONE zoom state (a group = frames whose size of the field of view is the same): per frame the focal it
    predicts (px of the ORIGINAL frame; VGGT pads the frame to a square), a depth map and its confidence. Frames are passed in sorted
    order in groups of `group`; the caller picks frames so that a group shares one zoom (see --vggt-groups)."""
    import cv2
    import torch
    ensure("from vggt.models.vggt import VGGT", "git+https://github.com/facebookresearch/vggt.git", extra=("einops", "safetensors"))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dev = "cuda"
    model = VGGT.from_pretrained(name).to(dev).eval()
    tmp = out / "vggt_in"
    tmp.mkdir(exist_ok=True)
    keys = sorted(imgs)
    rows = {"frame": [], "focal_px": [], "depth": [], "conf": [], "group": []}
    h0, w0 = imgs[keys[0]].shape[:2]
    groups = [[k for k in g if k in imgs] for g in VGGT_GROUPS] if VGGT_GROUPS else [keys[i:i + group] for i in range(0, len(keys), group)]
    for g, ks in enumerate(groups):
        if not ks:
            continue
        paths = []
        for k in ks:
            p = tmp / f"{k:06d}.png"
            cv2.imwrite(str(p), imgs[k])
            paths.append(str(p))
        x = load_and_preprocess_images(paths, mode="pad").to(dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)
        ex, K = pose_encoding_to_extri_intri(pred["pose_enc"], x.shape[-2:])
        K = K[0].float().cpu().numpy()                      # (n,3,3) px of the padded square image
        scale = max(h0, w0) / x.shape[-1]
        for i, k in enumerate(ks):
            rows["frame"].append(k)
            rows["focal_px"].append(float(0.5 * (K[i, 0, 0] + K[i, 1, 1]) * scale))
            rows["depth"].append(pred["depth"][0, i, ..., 0].float().cpu().numpy().astype(np.float16))
            rows["conf"].append(pred["depth_conf"][0, i].float().cpu().numpy().astype(np.float16))
            rows["group"].append(g)
        log(f"vggt group {g}: frames {ks[0]}..{ks[-1]} focal px " + " ".join(f"{v:.0f}" for v in rows["focal_px"][-len(ks):]))
        del pred
    np.savez_compressed(out / "vggt.npz", size=np.array([w0, h0]), **{k: np.array(v) for k, v in rows.items()})
    return {"frames": len(rows["frame"])}


MODELS = {"moge": run_moge, "dav2": run_dav2, "vggt": run_vggt}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", required=True, help="comma list of video frame indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="moge,dav2,vggt")
    ap.add_argument("--vggt-groups", default="", help="';'-separated groups of comma-separated frame indices (one zoom state each)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    VGGT_GROUPS[:] = [[int(v) for v in g.split(",") if v] for g in a.vggt_groups.split(";") if g]
    imgs = read_frames(a.video, [int(v) for v in a.frames.split(",")])
    res = {}
    for m in a.models.split(","):
        t0 = time.time()
        try:
            res[m] = MODELS[m](imgs, out)
            res[m]["seconds"] = time.time() - t0
        except Exception as e:      # one model failing must not lose the others
            res[m] = {"error": f"{type(e).__name__}: {e}"[:300]}
            log(f"{m} FAILED: {res[m]['error']}")
    (out / "geom_report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
