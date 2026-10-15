r"""Render a Gracia 4DGS asset (.mint, or a static .ply/.sog) at our cameras with graciasdk. Standalone on purpose.

graciasdk (github.com/gracia-labs/python-sdk) needs CPython >= 3.12 and a Vulkan GPU (Windows/Linux) or Metal (Apple
silicon); dancecap's .venv-nopo4d is 3.11, so this file imports nothing from dancecap and runs in its own environment:

    py -3.13 -m venv .venv-gracia
    .venv-gracia\Scripts\python -m pip install numpy pillow <graciasdk wheel from the SDK repo's wheels/>
    .venv-gracia\Scripts\python dancecap\dancecap\mono4d\render_gracia.py work\<take>\mono4d\<run>\render_job.json

The job (written by `python -m dancecap.mono4d render-job`):
    {"asset": ".../drop/asset.mint", "out": ".../drop/views", "width": W, "height": H, "times": [scene seconds ...],
     "cameras": {name: {"frames": [{"qvec": [w,x,y,z], "tvec": [3], "fx", "fy", "cx", "cy"} ... one per time]}}}
Cameras are COLMAP-style world->camera poses (OpenCV axes) ALREADY IN THE ASSET'S SCENE FRAME (rig.to_scene); the SDK's
camera_from_colmap handles its own Vulkan conventions. Every view must have the job's width x height (the SDK renders at
the size it was created with).

Output per camera: views/<name>/000000.jpg ... (RGB composited on black), alpha/<name>/000000.png (8-bit coverage:
a free foreground mask for the dancers stage) and depth/<name>.npz {depth (T,H,W) float16 view-space |z| metres in scene
units, 0 = empty}. Colour is the SDK's float16 RGBA in [0, 1]; whether it is premultiplied is not documented, so the
composite treats it as straight alpha (check one frame against the SDK viewer before trusting edges).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(job_path: str) -> int:
    from graciasdk import GraciaSDK, camera_from_colmap
    from PIL import Image

    job = json.loads(Path(job_path).read_text())
    W, H = int(job["width"]), int(job["height"])
    out = Path(job["out"])
    sdk = GraciaSDK(W, H)
    scene = sdk.load(job["asset"])
    if scene.is_video:
        scene.wait_ready(timeout=120.0)
    times = job["times"]
    for name, cam in job["cameras"].items():
        frames = cam["frames"]
        if len(frames) not in (1, len(times)):
            raise ValueError(f"{name}: {len(frames)} poses for {len(times)} times")
        (out / name).mkdir(parents=True, exist_ok=True)
        (out.parent / "alpha" / name).mkdir(parents=True, exist_ok=True)
        depth = np.zeros((len(times), H, W), np.float16)
        for i, t in enumerate(times):
            f = frames[i if len(frames) > 1 else 0]
            if scene.is_video:
                scene.set_time(float(t))
            c = camera_from_colmap(f["qvec"], f["tvec"], f["fx"], f["fy"], f["cx"], f["cy"], W, H)
            r = sdk.render(scene, c)
            rgba = np.asarray(r.color, np.float32)
            a = np.clip(rgba[..., 3:4], 0.0, 1.0)
            rgb = np.clip(rgba[..., :3] * a, 0.0, 1.0)
            Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).save(out / name / f"{i:06d}.jpg", quality=95)
            Image.fromarray((a[..., 0] * 255 + 0.5).astype(np.uint8)).save(out.parent / "alpha" / name / f"{i:06d}.png")
            depth[i] = np.abs(np.asarray(r.depth, np.float32))
        (out.parent / "depth").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out.parent / "depth" / f"{name}.npz", depth=depth)
        print(f"[render_gracia] {name}: {len(times)} frames -> {out / name}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
