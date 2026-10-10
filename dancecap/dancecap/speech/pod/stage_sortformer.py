"""Stage `sortformer`: end-to-end overlap-aware diarization (NVIDIA Sortformer 4-speaker) on every source.

    python -m dancecap.speech.pod.run sortformer [--cams 00,01] [--win 60] [--step 50] [--force]

Sortformer (nvidia/diar_sortformer_4spk-v1, CC-BY-NC-4.0: this project is non-commercial research) labels up to four speakers per
audio window with frame-level activity at 80 ms, so two people talking at once are both marked. The model handles about 90 s at a
time, so each source is cut into `win`-second windows every `step` seconds (10 s overlap); speaker numbers are only meaningful
INSIDE one window. The PC links the numbers across windows and sources (dancecap.speech.cluster) through the speaker embeddings;
what this stage is used for is (a) overlap flags, (b) sharper turn boundaries than the 0.75 s embedding hop.

Output raw/<cam>/sortformer.json: {stage, model, versions, cam, win, step, windows: [{t0, t1, segments: [{start, end, spk}]}], seconds}
Times are source seconds (window start added back).
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from . import common as C

MODEL = "nvidia/diar_sortformer_4spk-v1"


def parse_segment(line, offset: float) -> dict | None:
    """'12.48 15.92 speaker_1' (or a (start, end, spk) tuple) -> {start, end, spk} shifted by `offset`."""
    try:
        if isinstance(line, str):
            a, b, spk = line.split()[:3]
        else:
            a, b, spk = line[:3]
        n = int("".join(ch for ch in str(spk) if ch.isdigit()) or 0)
        return {"start": round(offset + float(a), 3), "end": round(offset + float(b), 3), "spk": n}
    except (ValueError, TypeError, IndexError):
        return None


def window_starts(duration: float, win: float, step: float) -> list[float]:
    out, t = [], 0.0
    while t < duration - 1.0:
        out.append(round(t, 3))
        if t + win >= duration:
            break
        t += step
    return out


def load_model(device: str):
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained(MODEL)
    return m.eval().to(device)


def run(cams: dict, force: bool = False, win: float = 60.0, step: float = 50.0, device: str = "cuda") -> dict:
    todo = {c: w for c, w in cams.items() if force or not (C.cam_dir(c) / "sortformer.json").exists()}
    if not todo:
        C.log("sortformer: nothing to do")
        return {}
    import soundfile as sf
    t0 = time.time()
    model = load_model(device)
    C.log(f"sortformer loaded in {time.time() - t0:.1f} s")
    done = {}
    for i, (cam, wav) in enumerate(todo.items()):
        t1 = time.time()
        audio = C.load_audio(wav)
        dur = len(audio) / C.SR
        windows = []
        with tempfile.TemporaryDirectory(prefix="sortformer_") as td:
            paths, starts = [], window_starts(dur, win, step)
            for k, s in enumerate(starts):
                p = Path(td) / f"w{k:03d}.wav"
                sf.write(str(p), audio[int(s * C.SR):int(min(dur, s + win) * C.SR)], C.SR, subtype="PCM_16")
                paths.append(str(p))
            for k, p in enumerate(paths):
                res = model.diarize(audio=[p], batch_size=1)
                if isinstance(res, tuple):
                    res = res[0]
                segs = [parse_segment(x, starts[k]) for x in (res[0] if res else [])]
                windows.append({"t0": starts[k], "t1": round(min(dur, starts[k] + win), 3), "segments": [s for s in segs if s]})
        C.write_json(C.cam_dir(cam) / "sortformer.json",
                     {"stage": "sortformer", "version": C.STAGE_VERSION, "model": MODEL, "versions": C.versions("nemo", "torch"), "cam": cam,
                      "duration": round(dur, 3), "win": win, "step": step, "windows": windows, "seconds": round(time.time() - t1, 2)})
        C.log(f"sortformer {cam}: {len(windows)} windows, {sum(len(w['segments']) for w in windows)} segments ({time.time() - t1:.1f} s)")
        C.progress_tick("speech/sortformer", i + 1, len(todo))
        done[cam] = len(windows)
    return done
