"""Experimental photometric refinement, not an official paper reproduction.

Keeps predicted cameras fixed. Uses MSE + local SSIM (no LPIPS), all source
frames, and constrained Gaussian activations. Records before/after source-view
metrics; these are fitting metrics, not held-out novel-view validation.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.model.types import Gaussians
from src.model.decoder.config import Decoder4DGSCfg
from src.model.decoder.decoder_4dgs import Decoder4DGS
from render_saved import between


def ssim(x,y):
    mx=F.avg_pool2d(x,7,1,3); my=F.avg_pool2d(y,7,1,3)
    vx=F.avg_pool2d(x*x,7,1,3)-mx*mx
    vy=F.avg_pool2d(y*y,7,1,3)-my*my
    cov=F.avg_pool2d(x*y,7,1,3)-mx*my
    return (((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=100)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    report=json.loads((a.source/'report.json').read_text())
    frames=report['frame_indices']; n=len(frames); h,w=report['height'],report['width']
    data=np.load(a.source/'dynamic_gaussians.npz')
    keep=data['opacities'][0]>.01
    tensors={k:torch.tensor(data[k][:,keep],device='cuda') for k in data}
    params={}; groups=[]
    rates=dict(means=.001,harmonics=.005,opacities=.025,scales=.005,rotations=.001,
               cov_t=.002,ms3_fwd=.001,ms3_bwd=.001,omega_fwd=.001,omega_bwd=.001,opacity_sh=.005)
    for k,v in tensors.items():
        if k=='t': continue
        if k in ('scales','cov_t'): v=v.clamp_min(1e-8).log()
        if k=='opacities': v=torch.logit(v.clamp(1e-5,1-1e-5))
        params[k]=torch.nn.Parameter(v.clone())
        groups.append(dict(params=[params[k]],lr=rates[k]))
    def gaussians():
        values={'t':tensors['t']}
        for k,v in params.items():
            if k in ('scales','cov_t'): v=v.exp()
            if k=='opacities': v=v.sigmoid()
            if k=='rotations': v=F.normalize(v,dim=-1)
            values[k]=v
        return Gaussians(**values)
    cams=np.load(a.source/'cameras.npz')
    c=torch.tensor(cams['extrinsic_c2w'],device='cuda')
    k=torch.tensor(cams['intrinsic'],device='cuda')
    targets=torch.stack([torch.tensor(np.asarray(Image.open(a.source/f'input_cam{cam}_{idx:05}.png')).copy(),device='cuda').permute(2,0,1).float()/255
                         for cam in (1,2) for idx in frames])
    times=torch.tensor([(idx-frames[0])/(frames[-1]-frames[0]) for cam in (1,2) for idx in frames],device='cuda')
    decoder=Decoder4DGS(Decoder4DGSCfg('4dgs',[1,1,1],1)).cuda()
    optimizer=torch.optim.Adam(groups)
    def render(gs,i):
        return decoder(gs,c[:,i:i+1],k[:,i:i+1],(h,w),times[i:i+1][None]).color[0]
    def evaluate(tag):
        metrics=[]
        with torch.no_grad():
            gs=gaussians()
            for i in range(len(targets)):
                pred=render(gs,i)
                mse=F.mse_loss(pred,targets[i:i+1]).item()
                metrics.append(dict(view=i//n,frame=frames[i%n],psnr=float(-10*np.log10(max(mse,1e-12))),ssim=float(ssim(pred,targets[i:i+1]))))
                Image.fromarray((pred[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(a.out/f'{tag}_source{i//n}_{frames[i%n]:05}.png')
            mid=torch.tensor(between(c[0,0].cpu().numpy(),c[0,n].cpu().numpy()),device='cuda')[None,None]
            midk=(k[:,:1]+k[:,n:n+1])/2
            for j,idx in enumerate(frames):
                im=decoder(gs,mid,midk,(h,w),times[j:j+1][None]).color[0,0]
                Image.fromarray((im.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(a.out/f'{tag}_novel_{idx:05}.png')
        return metrics
    before=evaluate('before'); history=[]; tick=time.time()
    print(f'Refining {keep.sum()} Gaussians for {a.steps} steps',flush=True)
    generator=torch.Generator().manual_seed(2026)
    order=[]
    for step in range(a.steps):
        if not order: order=torch.randperm(len(targets),generator=generator).tolist()
        i=order.pop()
        optimizer.zero_grad(set_to_none=True)
        pred=render(gaussians(),i)
        loss=F.mse_loss(pred,targets[i:i+1])+.05*(1-ssim(pred,targets[i:i+1]))
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
        optimizer.step()
        history.append(float(loss.detach()))
        if step%10==0: print(f'{step}: {history[-1]:.6f}',flush=True)
    after=evaluate('after')
    with torch.no_grad():
        gs=gaussians()
        np.savez_compressed(a.out/'dynamic_gaussians.npz',**{f.name:getattr(gs,f.name).detach().cpu().numpy() for f in dataclasses.fields(gs) if getattr(gs,f.name) is not None})
    np.savez(a.out/'cameras.npz',**{name:cams[name] for name in cams})
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    result=dict(steps=a.steps,seconds=time.time()-tick,gaussians=int(keep.sum()),before=before,after=after,loss=history,
                official_reproduction=False,held_out_novel_view_ground_truth=False,accepted_for_mamma=False,
                method='Fixed-camera MSE + 0.05 local SSIM; no LPIPS. Gaussian attributes optimized with Adam.')
    (a.out/'refinement.json').write_text(json.dumps(result,indent=2))
    print('Mean source PSNR:',np.mean([x['psnr'] for x in before]),'->',np.mean([x['psnr'] for x in after]),flush=True)


if __name__=='__main__': main()
