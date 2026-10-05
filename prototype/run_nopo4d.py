"""Local NoPo4D feasibility run. Original synchronized frames, no resynchronization.

Run with .venv-nopo4d/Scripts/python.exe prototype/run_nopo4d.py
Camera poses and geometry have arbitrary scale until independently calibrated.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from src.model.nopo4d import NoPo4D

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, nargs='+', default=[0, 1])
    p.add_argument('--width', type=int, default=252)
    p.add_argument('--height', type=int, default=140)
    p.add_argument('--out', type=Path, default=ROOT / 'prototype/nopo4d_smoke')
    p.add_argument('--render', action='store_true')
    p.add_argument('--cameras', type=int, nargs='+', default=[1, 2])
    a = p.parse_args()
    if a.width % 14 or a.height % 14:
        raise ValueError('Resolution must be divisible by the patch size, 14')
    if len(a.frames) < 2 or sorted(set(a.frames)) != a.frames or min(a.frames) < 0 or max(a.frames) >= 134:
        raise ValueError('Supply at least two increasing unique frame indices in 0..133')
    a.out.mkdir(parents=True, exist_ok=True)
    images = []
    for cam in a.cameras:
        cap = cv2.VideoCapture(str(ROOT / f'carlos-aline-spin/carlos-aline-spin-cam{cam}.mp4'))
        for idx in a.frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode cam{cam} frame {idx}')
            rgb = cv2.cvtColor(cv2.resize(bgr, (a.width, a.height)), cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(a.out / f'input_cam{cam}_{idx:05}.png')
            images.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255)
        cap.release()
    x = torch.stack(images).unsqueeze(0).cuda()
    ts = torch.tensor(a.frames, dtype=torch.float32, device='cuda')
    ts = ((ts-ts[0])/(ts[-1]-ts[0])).repeat(len(a.cameras)).unsqueeze(0)
    print('Loading pretrained NoPo4D', flush=True)
    model = NoPo4D.from_pretrained(str(ROOT / 'external/checkpoints/nopo4d')).cuda().eval()
    model.decoder.chunk_size = 1
    started = time.time()
    with torch.inference_mode():
        print(f'Encoding {tuple(x.shape)}', flush=True)
        result = model(x, timestamps=ts, num_cameras=len(a.cameras))
        gs = result.gaussians
        arrays = {f.name: getattr(gs, f.name).float().cpu().numpy()
                  for f in dataclasses.fields(gs) if getattr(gs, f.name) is not None}
        for name, arr in arrays.items():
            if not np.isfinite(arr).all():
                raise ValueError(f'Nonfinite Gaussian attribute: {name}')
        np.savez_compressed(a.out / 'dynamic_gaussians.npz', **arrays)
        poses = {k: v.float().cpu().numpy() for k, v in result.camera_pose.items() if isinstance(v, torch.Tensor)}
        np.savez_compressed(a.out / 'cameras.npz', **poses)
        np.savez_compressed(a.out / 'depth.npz', **{k:v.cpu().numpy() for k,v in result.depth.items()})
        stats = dict(frame_indices=a.frames, cameras=a.cameras, fps=24000/1001, width=a.width, height=a.height,
                     gaussian_count=int(gs.means.shape[1]), seconds=time.time()-started,
                     peak_vram_gb=torch.cuda.max_memory_allocated()/1e9,
                     metric_scale_validated=False, pose_validated=False,
                     note='Experimental learned reconstruction. Not a validated MAMMA input.')
        (a.out/'report.json').write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats), flush=True)
        if a.render:
            for cam in range(len(a.cameras)):
                for j, idx in enumerate(a.frames):
                    index = cam * len(a.frames) + j
                    output = model.render(gs, result.camera_pose['extrinsic_c2w'][:,index:index+1],
                                          result.camera_pose['intrinsic'][:,index:index+1],
                                          (a.height,a.width), ts[:,index:index+1])
                    im = output.color[0,0].clamp(0,1).permute(1,2,0).cpu().numpy()
                    Image.fromarray((im*255).astype(np.uint8)).save(a.out/f'render_cam{cam+1}_{idx:05}.png')
    print('Done', flush=True)


if __name__ == '__main__':
    main()
