"""Render a fixed novel camera through the saved four-second 4D model.

Only the timestamps recorded in report.json were observed by the model. Frames
between them are inferred, not additional measurements or validated motion.
"""
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.model.types import Gaussians
from src.model.decoder.config import Decoder4DGSCfg
from src.model.decoder.decoder_4dgs import Decoder4DGS
from render_saved import between

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'prototype/larissa_kadu/nopo4d_four_second'


def main():
    report=json.loads((RUN/'report.json').read_text())
    arrays=np.load(RUN/'dynamic_gaussians.npz')
    gs=Gaussians(**{k:torch.from_numpy(arrays[k]).cuda() for k in arrays})
    cams=np.load(RUN/'cameras.npz'); n=len(report['frame_indices'])
    decoder=Decoder4DGS(Decoder4DGSCfg(name='4dgs',background_color=[1,1,1],chunk_size=1)).cuda()
    with torch.inference_mode():
        for alpha in (.1,.5):
            c2w=between(cams['extrinsic_c2w'][0,2*n],cams['extrinsic_c2w'][0,3*n],alpha)
            intr=(1-alpha)*cams['intrinsic'][0,2*n]+alpha*cams['intrinsic'][0,3*n]
            c2w=torch.tensor(c2w[None,None],device='cuda'); intr=torch.tensor(intr[None,None],device='cuda')
            frames=RUN/f'novel_{alpha:.1f}_frames'; frames.mkdir(exist_ok=True)
            for idx in range(96):
                output=decoder(gs,c2w,intr,(report['height'],report['width']),torch.tensor([[idx/95]],device='cuda'))
                rgb=output.color[0,0].clamp(0,1).permute(1,2,0).cpu().numpy()
                Image.fromarray((rgb*255).astype('uint8')).save(frames/f'{idx:03}.png')
            subprocess.run(['ffmpeg','-v','error','-y','-framerate','24','-i',str(frames/'%03d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(RUN/f'novel_{alpha:.1f}.mp4')],check=True)
            print(f'Rendered 96 frames at camera interpolation {alpha}',flush=True)
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(ROOT/'prototype/larissa_kadu/clips/cam03.mp4'),'-i',str(RUN/'novel_0.1.mp4'),
                    '-filter_complex','[0:v]scale=336:588[a];[1:v]scale=336:588[b];[a][b]hstack[out]',
                    '-map','[out]','-an','-c:v','libx264','-crf','18',str(RUN/'original_vs_novel.mp4')],check=True)


if __name__=='__main__': main()
