"""Export sampled dynamic splats for the existing simplified Unity viewer."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from src.model.types import Gaussians
from reconstruct_clip import save_ply
from render_saved import between

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'prototype/larissa_kadu/nopo4d_four_second'
OUT=ROOT/'prototype/larissa_kadu/unity_preview'
OUT.mkdir(exist_ok=True)
(OUT/'splats').mkdir(exist_ok=True)
arrays=np.load(RUN/'dynamic_gaussians.npz')
gs=Gaussians(**{k:torch.from_numpy(arrays[k]).cuda() for k in arrays})
cams=np.load(RUN/'cameras.npz')
c=between(cams['extrinsic_c2w'][0,8],cams['extrinsic_c2w'][0,12],.1)
k=.9*cams['intrinsic'][0,8]+.1*cams['intrinsic'][0,12]
flip=np.diag([1.,-1.,1.])
with torch.inference_mode():
    for i in range(48):
        frame=gs.transform_to_target_time(torch.tensor([[i*2/95]],device='cuda'),0)
        save_ply(OUT/f'splats/frame_{i:05}.ply',gs,frame,1.,np.eye(3),np.zeros(3),max_points=60000)
info=dict(frame_count=48,fps=12,camera_position=(flip@c[:3,3]).tolist(),
          camera_rotation=Rotation.from_matrix(flip@c[:3,:3]@flip).as_quat().tolist(),
          pivot=(flip@np.median(arrays['means'][0],axis=0)).tolist(),
          vertical_fov=float(np.degrees(2*np.arctan(224/(2*k[1,1])))),
          note='Subsampled DC-only splats, 12 fps; four observed timestamps. Unvalidated geometry.')
(OUT/'preview.json').write_text(json.dumps(info,indent=2))
print('Exported 48 frames, 60000 splats/frame maximum',flush=True)
