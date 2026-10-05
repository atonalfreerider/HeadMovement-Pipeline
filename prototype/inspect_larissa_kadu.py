"""Generate synchronized contact sheets with ffmpeg autorotation."""
import concurrent.futures
import json
import subprocess
from pathlib import Path
from PIL import Image,ImageDraw
from sync_larissa_kadu import SOURCE,OUT


def frame(row,t):
    source_t=(t-row['offset'])/row['slope']
    dst=OUT/f'inspect/t{t:06.2f}_{row["id"]}.jpg'
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(source_t),'-i',str(SOURCE/row['name']),
                    '-frames:v','1','-vf','scale=270:480:force_original_aspect_ratio=decrease,pad=270:480:(ow-iw)/2:(oh-ih)/2',str(dst)],check=True)
    return dst


def main():
    (OUT/'inspect').mkdir(exist_ok=True)
    rows=json.loads((OUT/'alignment.json').read_text())['sources']
    for t in (38.,48.,58.):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            paths=list(pool.map(lambda r:frame(r,t),rows))
        sheet=Image.new('RGB',(810,1530),(22,25,30)); draw=ImageDraw.Draw(sheet)
        for i,(row,path) in enumerate(zip(rows,paths)):
            x=(i%3)*270; y=(i//3)*510
            draw.text((x+5,y+5),f'{row["id"]} {row["name"][:21]} | ref {t:.1f}s',fill='white')
            sheet.paste(Image.open(path),(x,y+30))
        sheet.save(OUT/f'contact_{int(t)}.jpg')
    print('Contact sheets ready',flush=True)


if __name__=='__main__': main()
