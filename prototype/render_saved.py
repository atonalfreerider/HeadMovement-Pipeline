"""Render saved NoPo4D Gaussians, including an interpolated novel camera."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from src.model.types import Gaussians
from src.model.decoder.config import Decoder4DGSCfg
from src.model.decoder.decoder_4dgs import Decoder4DGS


def between(a,b,alpha=.5):
    c=np.eye(4,dtype=np.float32)
    c[:3,:3]=Slerp([0,1],Rotation.from_matrix(np.stack([a[:3,:3],b[:3,:3]])))(alpha).as_matrix()
    c[:3,3]=(1-alpha)*a[:3,3]+alpha*b[:3,3]
    return c


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text())
    n=len(report['frame_indices'])
    arrays=np.load(a.run/'dynamic_gaussians.npz')
    gs=Gaussians(**{k:torch.from_numpy(arrays[k]).cuda() for k in arrays})
    cams=np.load(a.run/'cameras.npz')
    if cams['extrinsic_c2w'].shape[1] == n:
        c=cams['extrinsic_c2w'][0,0]
        k=cams['intrinsic'][0,0]
        poses=np.repeat(c[None],3,axis=0)
        distance=float(torch.median(gs.means[0,:,2]).cpu())*.03
        poses[1,:3,3]+=c[:3,0]*distance
        poses[2,:3,3]-=c[:3,0]*distance
        intrinsics=np.repeat(k[None],3,axis=0)
    else:
        c=cams['extrinsic_c2w'][0,[0,n]]
        k=cams['intrinsic'][0,[0,n]]
        poses=np.stack([c[0],between(*c),c[1]])
        intrinsics=np.stack([k[0],(k[0]+k[1])/2,k[1]])
    decoder=Decoder4DGS(Decoder4DGSCfg(name='4dgs',background_color=[1,1,1],chunk_size=1)).cuda()
    with torch.inference_mode():
        for i,idx in enumerate(report['frame_indices']):
            t=(idx-report['frame_indices'][0])/(report['frame_indices'][-1]-report['frame_indices'][0])
            for v in range(3):
                out=decoder(gs,torch.tensor(poses[v:v+1][None],device='cuda'),
                            torch.tensor(intrinsics[v:v+1][None],device='cuda'),
                            (report['height'],report['width']),torch.tensor([[t]],device='cuda'))
                im=out.color[0,0].clamp(0,1).permute(1,2,0).cpu().numpy()
                Image.fromarray((im*255).astype('uint8')).save(a.run/f'view{v}_{idx:05}.png')
                print(f'rendered view{v} frame{idx}',flush=True)


if __name__=='__main__':
    main()
