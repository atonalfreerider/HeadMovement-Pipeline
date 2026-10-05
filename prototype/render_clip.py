"""Render the saved chunk sequence without re-running the neural encoder."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.model.types import Gaussians
from src.model.decoder.config import Decoder4DGSCfg
from src.model.decoder.decoder_4dgs import Decoder4DGS
from reconstruct_clip import to_local_camera


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('--stride',type=int,default=1)
    a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text())
    fixed=np.load(a.run/'fixed_cameras.npz')
    decoder=Decoder4DGS(Decoder4DGSCfg('4dgs',[1,1,1],1)).cuda()
    with torch.inference_mode():
        for chunk in report['chunks']:
            start=chunk['start']
            with np.load(a.run/f'chunks/chunk_{start:05}.npz') as data:
                gs=Gaussians(**{k:torch.from_numpy(data[k]).cuda() for k in data})
            cams=np.load(a.run/f'chunks/cameras_{start:05}.npz')
            indices=cams['frame_indices']
            local=np.stack([to_local_camera(c,float(cams['scale']),cams['rotation'],cams['translation']) for c in fixed['c2w']])
            for idx in indices:
                if idx<start or idx%a.stride: continue
                t=float((idx-indices[0])/(indices[-1]-indices[0]))
                for v in range(3):
                    out=decoder(gs,torch.tensor(local[v:v+1][None],device='cuda'),
                                torch.tensor(fixed['intrinsics'][v:v+1][None],device='cuda'),
                                (report['height'],report['width']),torch.tensor([[t]],device='cuda'))
                    im=out.color[0,0].clamp(0,1).permute(1,2,0).cpu().numpy()
                    Image.fromarray((im*255).astype('uint8')).save(a.run/f'renders/view{v}/frame_{idx:05}.png')
                    del out
            print(f'Rendered chunk {start}',flush=True)
            del gs
            torch.cuda.empty_cache()


if __name__=='__main__': main()
