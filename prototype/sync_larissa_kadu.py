"""Multi-anchor audio synchronization and maximum simultaneous video coverage.

offset convention: reference_time = slope * source_time + offset.
Uses independent ten-second audio windows, rejects disagreeing anchors, and
preserves all evidence. Audio alignment does not prove frame-exact visual sync.
"""
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy import fft,signal

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'dances/Larissa-Kadu-demo'
OUT=ROOT/'prototype/larissa_kadu'
SR=8000


def match(sig,ref):
    n=fft.next_fast_len(len(sig)+len(ref)-1)
    cross=fft.rfft(ref,n)*fft.rfft(sig,n).conj()
    # Fractional PHAT, restricted to the actual passband. Full whitening of
    # float32 stop-band noise can create false peaks at signal boundaries.
    frequencies=fft.rfftfreq(n,1/SR)
    cross/=np.maximum(np.abs(cross),1e-5)**.7
    cross[(frequencies<300)|(frequencies>2800)]=0
    cc=fft.irfft(cross,n)
    # Only locations where the whole anchor fits inside the reference.
    cc=np.abs(cc[:len(ref)-len(sig)+1])
    peak=int(cc.argmax())
    side=cc.copy(); side[max(0,peak-int(.2*SR)):peak+int(.2*SR)]=0
    return peak/SR,float(cc[peak]/(np.percentile(side,99.9)+1e-12))


def main():
    OUT.mkdir(exist_ok=True)
    probes=json.loads((OUT/'probes.json').read_text())
    audios={}; entries=[]
    filt=signal.butter(4,[250,3000],btype='bandpass',fs=SR,output='sos')
    for i,record in enumerate(probes):
        name=record['name']; cam=f'{i+1:02}'
        cache=OUT/f'audio_{cam}.npy'
        if not cache.exists():
            raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(SOURCE/name),'-vn','-ac','1','-ar',str(SR),'-f','f32le','-'])
            np.save(cache,np.frombuffer(raw,dtype=np.float32))
        audios[cam]=signal.sosfiltfilt(filt,np.load(cache)).astype('float32')
        v=next(s for s in record['probe']['streams'] if s['codec_type']=='video')
        entries.append(dict(id=cam,name=name,duration=float(v['duration']),numbered=i<7))
    ref=audios['06']
    for row in entries:
        if row['id']=='06':
            row.update(slope=1.,offset=0.,anchors=[],accepted=True,max_residual_seconds=0.)
            continue
        audio=audios[row['id']]; duration=len(audio)/SR
        anchors=[]
        for start in np.linspace(2,max(2,duration-12),9):
            sig=audio[int(start*SR):int((start+10)*SR)]
            location,confidence=match(sig,ref)
            anchors.append(dict(source_start=float(start),reference_start=location,offset=location-start,confidence=confidence))
        good=np.array([a['confidence']>2 for a in anchors])
        offsets=np.array([a['offset'] for a in anchors])
        if not good.any():
            row.update(accepted=False,anchors=anchors); print(row['id'],'NO ALIGNMENT',flush=True); continue
        # Dominant agreement cluster is resistant to edits/repeated musical phrases.
        best=max(np.flatnonzero(good),key=lambda i: int(np.sum(good&(np.abs(offsets-offsets[i])<.15))))
        keep=good&(np.abs(offsets-offsets[best])<.15)
        x=np.array([a['source_start'] for a in anchors]); y=np.array([a['reference_start'] for a in anchors])
        slope,offset=np.polyfit(x[keep],y[keep],1) if keep.sum()>=3 else (1.,float(np.median(offsets[keep])))
        residual=y-(slope*x+offset)
        for a,k,r in zip(anchors,keep,residual): a.update(inlier=bool(k),residual_seconds=float(r))
        row.update(slope=float(slope),offset=float(offset),anchors=anchors,
                   accepted=bool(keep.sum()>=5 and abs(slope-1)<.005),
                   max_residual_seconds=float(np.max(np.abs(residual[keep]))),anchor_inliers=int(keep.sum()))
        print(row['id'],row['name'],row['offset'],row['slope'],row['anchor_inliers'],row['accepted'],flush=True)
    events=[]
    for row in entries:
        if row['accepted']:
            events.extend([row['offset'],row['offset']+row['slope']*row['duration']])
    boundaries=sorted(set(events)); intervals=[]
    for start,end in zip(boundaries,boundaries[1:]):
        mid=(start+end)/2
        active=[r['id'] for r in entries if r['accepted'] and r['offset']<=mid<r['offset']+r['slope']*r['duration']]
        intervals.append(dict(start=start,end=end,cameras=active,count=len(active),numbered_count=sum(int(c)<=7 for c in active)))
    maximum=max(x['count'] for x in intervals)
    result=dict(reference='06-Demo-John.mp4',convention='reference_time = slope * source_time + offset',
                sources=entries,intervals=intervals,max_count=maximum,
                maximum_overlap=[x for x in intervals if x['count']==maximum],
                warning='Unnumbered files may duplicate cameras. Audio propagation and A/V offset still require visual verification.')
    (OUT/'alignment.json').write_text(json.dumps(result,indent=2))
    print('MAXIMUM OVERLAP',result['maximum_overlap'],flush=True)


if __name__=='__main__': main()
