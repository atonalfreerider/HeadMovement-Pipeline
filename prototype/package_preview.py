"""Validate sequence completeness and package inspectable preview artifacts."""
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'prototype/nopo4d_full'


def main():
    report=json.loads((RUN/'report.json').read_text())
    n=report['frame_count']; assert n==134
    for i in range(n):
        assert (RUN/f'splats/frame_{i:05}.ply').is_file()
        for v in range(3): assert (RUN/f'renders/view{v}/frame_{i:05}.png').is_file()
    fixed=np.load(RUN/'fixed_cameras.npz')
    c=fixed['c2w'][0]; flip=np.diag([1.,-1.,1.])
    with np.load(RUN/'chunks/chunk_00000.npz') as data:
        pivot=flip@np.median(data['means'][0],axis=0)
    preview=dict(frame_count=n,fps=report['fps'],camera_position=(flip@c[:3,3]).tolist(),
                 camera_rotation=Rotation.from_matrix(flip@c[:3,:3]@flip).as_quat().tolist(),
                 pivot=pivot.tolist(),vertical_fov=float(np.degrees(2*np.arctan(report['height']/(2*fixed['intrinsics'][0,1,1])))))
    (RUN/'preview.json').write_text(json.dumps(preview,indent=2))
    for v in range(3):
        path=RUN/f'view{v}.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-framerate','24000/1001','-i',
                        str(RUN/f'renders/view{v}/frame_%05d.png'),'-c:v','libx264','-crf','18',
                        '-pix_fmt','yuv420p',str(path)],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                       '-show_entries','stream=nb_frames,r_frame_rate','-of','json',str(path)]))
        assert int(probe['streams'][0]['nb_frames'])==n
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(ROOT/'carlos-aline-spin/carlos-aline-spin-cam1.mp4'),
                    '-vn','-c:a','pcm_s16le',str(RUN/'audio.wav')],check=True)
    # This is a validation montage of program outputs, not synthesized footage.
    w,h=504,280
    montage=Image.new('RGB',(w*3,h*2+90),(24,27,33))
    draw=ImageDraw.Draw(montage)
    files=[ROOT/'prototype/nopo4d_quality/input_cam1_00000.png',
           ROOT/'prototype/nopo4d_quality/view1_00000.png',
           ROOT/'prototype/nopo4d_quality/input_cam2_00000.png',
           ROOT/'prototype/nopo4d_quality/view0_00000.png',
           ROOT/'prototype/nopo4d_mono/view1_00000.png',
           ROOT/'prototype/nopo4d_quality/view2_00000.png']
    labels=['Original camera 1','Two-camera NOVEL view: duplicated couple','Original camera 2',
            'Reconstructed camera 1: ghost copy','Monocular sideways view: holes','Reconstructed camera 2: ghost copy']
    for i,(file,label) in enumerate(zip(files,labels)):
        x=(i%3)*w; y=(i//3)*(h+35)
        draw.text((x+10,y+10),label,fill='white')
        montage.paste(Image.open(file).resize((w,h)),(x,y+35))
    draw.text((10,h*2+78),'RESEARCH BASELINE — geometry not accepted for MAMMA / contact / dance physics',fill=(255,170,80))
    montage.save(RUN/'quality_comparison.jpg')
    gate=dict(accepted_for_mamma=False,
              reason='Two-camera fusion duplicates dancers; monocular novel views reveal holes. No metric scale or reliable cross-view identity.',
              frames_exported=n,novel_camera_video='view1.mp4',
              max_chunk_rotation_drift_deg=max(x['camera1_rotation_drift_degrees'] for x in report['chunks']),
              max_chunk_centre_drift_baseline_fraction=max(x['camera1_centre_drift_fraction_baseline'] for x in report['chunks']),
              unity_preview='DC-only splat discs; does not reproduce full Gaussian covariance or view-dependent appearance',
              mamma_run=False,roles_verified=False)
    (RUN/'quality_gate.json').write_text(json.dumps(gate,indent=2))
    print(json.dumps(gate,indent=2))


if __name__=='__main__': main()
