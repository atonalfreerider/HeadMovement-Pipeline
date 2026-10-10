"""Stage `vad`: Silero VAD speech regions + frame probabilities for every source.

Output raw/<cam>/vad.json:
  {stage, cam, duration, regions: [{start, end}], speech_s, speech_frac, frame_step, probs: [...]  (rounded to 2 decimals)}
The regions use min_silence 300 ms and 120 ms padding (a teacher's pauses between words are shorter than that, a breath between
two sentences is not). All later stages read this file instead of running the model again.
"""
from __future__ import annotations

import time

from . import common as C


def run(cams: dict, force: bool = False) -> dict:
    done = {}
    for i, (cam, wav) in enumerate(cams.items()):
        out = C.cam_dir(cam) / "vad.json"
        if out.exists() and not force:
            C.log(f"vad {cam}: exists, skipped")
            continue
        t0 = time.time()
        audio = C.load_audio(wav)
        dur = len(audio) / C.SR
        regs = C.vad_regions(audio)
        probs, step = C.vad_probabilities(audio)
        regions = [{"start": round(s / C.SR, 3), "end": round(e / C.SR, 3)} for s, e in regs]
        speech = sum(r["end"] - r["start"] for r in regions)
        C.write_json(out, {"stage": "vad", "version": C.STAGE_VERSION, "cam": cam, "file": wav.name, "duration": round(dur, 3),
                           "regions": regions, "speech_s": round(speech, 2), "speech_frac": round(speech / dur, 4) if dur else 0.0,
                           "frame_step": step, "probs": [round(float(p), 2) for p in probs],
                           "params": {"min_silence_ms": 300, "pad_ms": 120, "threshold": 0.5}, "seconds": round(time.time() - t0, 2)})
        C.log(f"vad {cam}: {len(regions)} regions, {speech:.1f} s of {dur:.1f} s speech ({time.time() - t0:.1f} s)")
        C.progress_tick("speech/vad", i + 1, len(cams))
        done[cam] = len(regions)
    return done


def load(cam: str) -> dict | None:
    return C.read_json(C.cam_dir(cam) / "vad.json")
