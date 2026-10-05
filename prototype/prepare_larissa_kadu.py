"""Prepare the maximum-coverage dance test using measured affine audio clocks."""
import concurrent.futures
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'prototype/larissa_kadu'
SOURCE = ROOT/'dances/Larissa-Kadu-demo'


def main():
    alignment = json.loads((OUT/'alignment.json').read_text())
    clips = OUT/'clips'
    clips.mkdir(exist_ok=True)
    rows = [dict(r) for r in alignment['sources'] if r['id'] != '08']
    def prepare(row):
        start = (58-row['offset'])/row['slope']
        dst = clips/f"cam{row['id']}.mp4"
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(start),'-i',str(SOURCE/row['name']),
                        '-map','0:v:0','-an','-vf',f"setpts=(PTS-STARTPTS)*{row['slope']},fps=24,scale=448:784:force_original_aspect_ratio=decrease,pad=448:784:(ow-iw)/2:(oh-ih)/2,setsar=1",
                        '-frames:v','96','-c:v','libx264','-preset','fast','-crf','18',str(dst)],check=True)
        return dict(id=row['id'],source=row['name'],source_start_seconds=start,
                    source_end_seconds=(62-row['offset'])/row['slope'],slope=row['slope'],offset=row['offset'],
                    clip=str(dst.relative_to(ROOT)),numbered_counterclockwise=row['numbered'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(prepare,rows))
    manifest=dict(reference=alignment['reference'],reference_start=58,reference_end=62,
                  maximum_overlap_reference=[29.97133470205688,68.508973804989],
                  fps=24,frame_count=96,width=448,height=784,cameras=records,
                  excluded={'08':'Appears to duplicate camera 09, with overlaid text and upscaling.'},
                  caveats=['Audio alignment has residual uncertainty of roughly one or two video frames.',
                           'Handheld camera motion, optical/digital zoom, mirror reflections retained.',
                           'Camera 09 is unnumbered; its room ordering is unspecified.',
                           'Letterboxing preserves full field of view; no dancers have been masked.'])
    (OUT/'test_manifest.json').write_text(json.dumps(manifest,indent=2))
    args=['ffmpeg','-v','error','-y']
    for r in records: args+=['-i',str(ROOT/r['clip'])]
    filters=[f'[{i}:v]scale=168:294[v{i}]' for i in range(8)]
    filters+=[''.join(f'[v{i}]' for i in range(8))+'xstack=inputs=8:layout=0_0|168_0|336_0|504_0|0_294|168_294|336_294|504_294[out]']
    subprocess.run(args+['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-crf','18',str(OUT/'synchronized_mosaic.mp4')],check=True)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
