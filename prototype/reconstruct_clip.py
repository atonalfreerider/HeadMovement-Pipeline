"""Chunked 134-frame NoPo4D experiment with explicit approximate alignment.

Saves learned 4D parameters per chunk, fixed-camera renders, and a DC-only PLY
sequence for Unity's existing preview renderer. Does not claim metric accuracy.
"""
import argparse
import dataclasses
import gc
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from src.model.nopo4d import NoPo4D
from render_saved import between

ROOT=Path(__file__).resolve().parents[1]
FPS=24000/1001


def align_cameras(current,reference):
    """Similarity from first camera rotation/centre and two-camera baseline."""
    baseline=np.linalg.norm(current[1,:3,3]-current[0,:3,3])
    if baseline<1e-6:
        raise ValueError('Predicted cameras have a degenerate baseline')
    s=np.linalg.norm(reference[1,:3,3]-reference[0,:3,3])/baseline
    r=reference[0,:3,:3]@current[0,:3,:3].T
    t=reference[0,:3,3]-s*r@current[0,:3,3]
    return float(s),r,t


def to_local_camera(global_c,s,r,t):
    c=np.eye(4,dtype=np.float32)
    c[:3,:3]=r.T@global_c[:3,:3]
    c[:3,3]=r.T@(global_c[:3,3]-t)/s
    return c


def save_ply(path,gs,transformed,s,r,t,max_points=120000):
    xyz=transformed['means'][0].cpu().numpy()
    op=transformed['opacities'][0].cpu().numpy()
    keep=np.flatnonzero((op>.04)&np.isfinite(xyz).all(axis=1))
    if len(keep)>max_points:
        keep=keep[np.linspace(0,len(keep)-1,max_points,dtype=int)]
    xyz=(xyz[keep]@r.T)*s+t
    scale=gs.scales[0,keep].clamp(1e-8,.05).cpu().numpy()*s
    quat=transformed['rotations'][0,keep].cpu().numpy()
    # NoPo4D and standard 3DGS store wxyz; scipy expects xyzw.
    qr=Rotation.from_matrix(r)*Rotation.from_quat(quat[:,[1,2,3,0]])
    quat=qr.as_quat()[:,[3,0,1,2]]
    dc=gs.harmonics[0,keep,:,0].cpu().numpy()
    names=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity',
           'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    data=np.zeros(len(keep),dtype=[(k,'<f4') for k in names])
    for j,k in enumerate(['x','y','z']): data[k]=xyz[:,j]
    for j in range(3):
        data[f'f_dc_{j}']=dc[:,j]
        data[f'scale_{j}']=np.log(scale[:,j])
    for j in range(4): data[f'rot_{j}']=quat[:,j]
    alpha=np.clip(op[keep],1e-6,1-1e-6)
    data['opacity']=np.log(alpha/(1-alpha))
    header='ply\nformat binary_little_endian 1.0\n'+f'element vertex {len(data)}\n'+''.join(f'property float {k}\n' for k in names)+'end_header\n'
    with path.open('wb') as f:
        f.write(header.encode()); f.write(data.tobytes())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'prototype/nopo4d_full')
    p.add_argument('--width',type=int,default=448)
    p.add_argument('--height',type=int,default=252)
    p.add_argument('--chunk',type=int,default=4)
    p.add_argument('--limit',type=int,default=134)
    p.add_argument('--no-render',action='store_true')
    a=p.parse_args()
    if a.width%14 or a.height%14 or a.chunk<2 or not 2<=a.limit<=134:
        raise ValueError('Invalid resolution, chunk size, or frame limit')
    for d in ['splats','chunks','renders/view0','renders/view1','renders/view2']:
        (a.out/d).mkdir(parents=True,exist_ok=True)
    videos=[]
    for cam in (1,2):
        cap=cv2.VideoCapture(str(ROOT/f'carlos-aline-spin/carlos-aline-spin-cam{cam}.mp4'))
        frames=[]
        for i in range(a.limit):
            ok,im=cap.read()
            if not ok: raise RuntimeError(f'cam{cam} ended at {i}')
            frames.append(cv2.cvtColor(cv2.resize(im,(a.width,a.height)),cv2.COLOR_BGR2RGB))
        cap.release(); videos.append(frames)
    print('Loading model',flush=True)
    model=NoPo4D.from_pretrained(str(ROOT/'external/checkpoints/nopo4d')).cuda().eval()
    model.decoder.chunk_size=1
    ref=None; stats=[]; start_time=time.time()
    with torch.inference_mode():
        for start in range(0,a.limit,a.chunk):
            stop=min(start+a.chunk,a.limit)
            indices=list(range(start,stop))
            # Last single frame uses its preceding frame as context.
            if len(indices)==1: indices.insert(0,start-1)
            n=len(indices)
            x=torch.from_numpy(np.stack([videos[c][i] for c in range(2) for i in indices])).permute(0,3,1,2).float()[None].cuda()/255
            ts=torch.linspace(0,1,n,device='cuda').repeat(2)[None]
            tick=time.time()
            result=model(x,timestamps=ts,num_cameras=2)
            gs=result.gaussians
            arrays={f.name:getattr(gs,f.name).cpu().numpy() for f in dataclasses.fields(gs) if getattr(gs,f.name) is not None}
            if any(not np.isfinite(v).all() for v in arrays.values()): raise ValueError('Nonfinite model output')
            np.savez_compressed(a.out/f'chunks/chunk_{start:05}.npz',**arrays)
            c=result.camera_pose['extrinsic_c2w'][0,[0,n]].cpu().numpy()
            k=result.camera_pose['intrinsic'][0,[0,n]].cpu().numpy()
            if ref is None:
                ref=c.copy()
                targets=np.stack([c[0],between(*c),c[1]])
                target_k=np.stack([k[0],(k[0]+k[1])/2,k[1]])
                np.savez(a.out/'fixed_cameras.npz',c2w=targets,intrinsics=target_k)
            s,r,t=align_cameras(c,ref)
            local=np.stack([to_local_camera(v,s,r,t) for v in targets])
            rotation_error=float(Rotation.from_matrix(ref[1,:3,:3].T@r@c[1,:3,:3]).magnitude()*180/np.pi)
            baseline=np.linalg.norm(ref[1,:3,3]-ref[0,:3,3])
            centre_error=float(np.linalg.norm(s*r@c[1,:3,3]+t-ref[1,:3,3])/baseline)
            np.savez(a.out/f'chunks/cameras_{start:05}.npz',c2w=c,intrinsics=k,scale=s,rotation=r,translation=t,frame_indices=indices)
            for j,idx in enumerate(indices):
                if idx<start: continue
                timestamp=ts[:,:n][:,j:j+1]
                transformed=gs.transform_to_target_time(timestamp,0)
                save_ply(a.out/f'splats/frame_{idx:05}.ply',gs,transformed,s,r,t)
                if not a.no_render:
                    for v in range(3):
                        out=model.render(gs,torch.tensor(local[v:v+1][None],device='cuda'),
                                         torch.tensor(target_k[v:v+1][None],device='cuda'),
                                         (a.height,a.width),timestamp)
                        im=out.color[0,0].clamp(0,1).permute(1,2,0).cpu().numpy()
                        Image.fromarray((im*255).astype('uint8')).save(a.out/f'renders/view{v}/frame_{idx:05}.png')
                        del out
                del transformed
            item=dict(start=start,stop=stop,seconds=time.time()-tick,scale_to_first=s,
                      camera1_rotation_drift_degrees=rotation_error,camera1_centre_drift_fraction_baseline=centre_error)
            stats.append(item)
            metadata=dict(frame_count=stop,fps=FPS,width=a.width,height=a.height,chunks=stats,
                          elapsed_seconds=time.time()-start_time,metric_scale_validated=False,
                          pose_validated=False,alignment='Approximate chunk similarity from camera0 and baseline; no global bundle adjustment',
                          ply_appearance='Subsampled DC-only preview; original higher-order color and view-dependent opacity retained in chunk NPZs')
            (a.out/'report.json').write_text(json.dumps(metadata,indent=2))
            print(json.dumps(item),flush=True)
            del result,gs,x,arrays
            gc.collect(); torch.cuda.empty_cache()
    print('Sequence complete',flush=True)


if __name__=='__main__': main()
