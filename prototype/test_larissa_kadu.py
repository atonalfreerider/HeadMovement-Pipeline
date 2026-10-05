"""Experimental NoPo4D test with individual poses/intrinsics for handheld views."""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from src.model.nopo4d import NoPo4D
from render_saved import between

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'prototype/larissa_kadu'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frames',type=int,nargs='+',default=[0,1])
    p.add_argument('--width',type=int,default=168)
    p.add_argument('--height',type=int,default=294)
    p.add_argument('--out',type=Path,default=BASE/'nopo4d_test')
    a=p.parse_args()
    manifest=json.loads((BASE/'test_manifest.json').read_text())
    if a.width%14 or a.height%14 or len(a.frames)<2 or sorted(set(a.frames))!=a.frames or not 0<=min(a.frames)<=max(a.frames)<96:
        raise ValueError('Use patch-aligned dimensions and increasing frame indices inside 0..95')
    a.out.mkdir(exist_ok=True,parents=True)
    inputs=[]
    for row in manifest['cameras']:
        cap=cv2.VideoCapture(str(ROOT/row['clip']))
        for idx in a.frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ok,bgr=cap.read()
            if not ok: raise RuntimeError(f'Cannot decode {row["id"]}: {idx}')
            rgb=cv2.cvtColor(cv2.resize(bgr,(a.width,a.height)),cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(a.out/f'input_{row["id"]}_{idx:03}.png')
            inputs.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255)
        cap.release()
    x=torch.stack(inputs)[None].cuda()
    n=len(a.frames); count=len(manifest['cameras'])
    ts=torch.tensor(a.frames,device='cuda',dtype=torch.float32)
    ts=((ts-ts[0])/(ts[-1]-ts[0])).repeat(count)[None]
    print(f'Loading NoPo4D: {count} cameras x {n} timestamps; {a.width}x{a.height}; per-frame poses',flush=True)
    model=NoPo4D.from_pretrained(str(ROOT/'external/checkpoints/nopo4d')).cuda().eval()
    model.decoder.chunk_size=1
    start=time.time()
    with torch.inference_mode():
        result=model(x,timestamps=ts,num_cameras=count,average_poses=False)
        gs=result.gaussians
        arrays={f.name:getattr(gs,f.name).float().cpu().numpy() for f in dataclasses.fields(gs) if getattr(gs,f.name) is not None}
        if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError('Nonfinite Gaussian attributes')
        np.savez_compressed(a.out/'dynamic_gaussians.npz',**arrays)
        poses={k:v.float().cpu().numpy() for k,v in result.camera_pose.items() if isinstance(v,torch.Tensor)}
        np.savez_compressed(a.out/'cameras.npz',**poses)
        errors=[]
        for c,row in enumerate(manifest['cameras']):
            for j,idx in enumerate(a.frames):
                k=c*n+j
                output=model.render(gs,result.camera_pose['extrinsic_c2w'][:,k:k+1],result.camera_pose['intrinsic'][:,k:k+1],(a.height,a.width),ts[:,k:k+1])
                rgb=output.color[0,0].clamp(0,1)
                mse=float(((rgb-x[0,k])**2).mean())
                errors.append(dict(camera=row['id'],frame=idx,source_psnr=-10*np.log10(max(mse,1e-12))))
                Image.fromarray((rgb.permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(a.out/f'render_{row["id"]}_{idx:03}.png')
            print(f'Rendered source camera {row["id"]}',flush=True)
        # Small displacement from camera 03 toward neighboring numbered camera 04.
        # A full midpoint can leave a thin learned surface; render both for inspection.
        for alpha in (.1,.5):
            for j,idx in enumerate(a.frames):
                c2w=between(poses['extrinsic_c2w'][0,2*n+j],poses['extrinsic_c2w'][0,3*n+j],alpha)
                intr=(1-alpha)*poses['intrinsic'][0,2*n+j]+alpha*poses['intrinsic'][0,3*n+j]
                output=model.render(gs,torch.tensor(c2w[None,None],device='cuda'),torch.tensor(intr[None,None],device='cuda'),(a.height,a.width),ts[:,j:j+1])
                rgb=output.color[0,0].clamp(0,1).permute(1,2,0).cpu().numpy()
                Image.fromarray((rgb*255).astype('uint8')).save(a.out/f'novel_{alpha:.1f}_{idx:03}.png')
        report=dict(frame_indices=a.frames,reference_times=[58+i/24 for i in a.frames],cameras=[r['id'] for r in manifest['cameras']],
                    fps=24,width=a.width,height=a.height,average_poses=False,gaussian_count=int(gs.means.shape[1]),
                    seconds=time.time()-start,peak_vram_gb=torch.cuda.max_memory_allocated()/1e9,source_metrics=errors,
                    metric_scale_validated=False,pose_validated=False,mamma_ready=False,
                    note='Experimental handheld/zoom input to a static-rig model; mirror reflections retained. Source PSNR is not novel-view validation.')
        (a.out/'report.json').write_text(json.dumps(report,indent=2))
    # Source/render pairs are placed side by side, ordered 01..07,09.
    sheet=Image.new('RGB',(a.width*8,(a.height+26)*2),(25,25,25)); draw=ImageDraw.Draw(sheet)
    for c,row in enumerate(manifest['cameras']):
        ox=(c%4)*a.width*2; oy=(c//4)*(a.height+26)
        draw.text((ox+4,oy+5),f'{row["id"]} original | NoPo4D',fill='white')
        for k,kind in enumerate(('input','render')):
            sheet.paste(Image.open(a.out/f'{kind}_{row["id"]}_{a.frames[0]:03}.png'),(ox+k*a.width,oy+26))
    sheet.save(a.out/'source_comparison.jpg')
    print(json.dumps({k:v for k,v in report.items() if k!='source_metrics'},indent=2),flush=True)


if __name__=='__main__': main()
