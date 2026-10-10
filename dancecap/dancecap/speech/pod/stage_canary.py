"""Stage `canary`: NVIDIA canary-1b-v2 (NeMo FastConformer encoder + Transformer decoder, 25 languages) as the THIRD ASR family:
Whisper is an encoder-decoder trained on 680k h of web audio, parakeet a transducer, canary an attention encoder-decoder trained
on a different curated mix. It needs the source language told per utterance, taken from Whisper's language id of the VAD regions
(stage whisper) when that file exists, else English; Portuguese segments are decoded with source_lang='pt'.

    python -m dancecap.speech.pod.run canary [--cams 00,01] [--force]

Word times come from the model's own 80 ms-resolution timestamps (coarser than parakeet's; use them to align, not to cut). No word
confidences (null).

Output raw/<cam>/canary-1b-v2.json: same schema as the parakeet file, plus per-segment `lang`.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import stage_vad
from .stage_parakeet import parse_hypothesis, _hyp_text

MODEL = "nvidia/canary-1b-v2"
NAME = "canary-1b-v2"


def segment_language(a: float, b: float, regions: list[dict]) -> str:
    """'pt' when the Whisper region ids overlapping [a, b] vote mostly Portuguese (weighted by overlap), else 'en'."""
    votes = {"en": 0.0, "pt": 0.0}
    for r in regions or []:
        ov = min(b, r["end"]) - max(a, r["start"])
        if ov <= 0 or not r.get("lid"):
            continue
        p = r["lid"]
        votes["pt"] += ov * float(p.get("pt", 0.0))
        votes["en"] += ov * float(p.get("en", 0.0))
    return "pt" if votes["pt"] > votes["en"] else "en"


def run(cams: dict, force: bool = False, device: str = "cuda") -> dict:
    todo = {c: w for c, w in cams.items() if force or not (C.cam_dir(c) / f"{NAME}.json").exists()}
    if not todo:
        C.log("canary: nothing to do")
        return {}
    import soundfile as sf
    import nemo.collections.asr as nemo_asr
    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL).eval().to(device)
    load_s = time.time() - t0
    C.log(f"canary loaded in {load_s:.1f} s")
    done = {}
    for i, (cam, wav) in enumerate(todo.items()):
        t1 = time.time()
        audio = C.load_audio(wav)
        dur = len(audio) / C.SR
        vad = stage_vad.load(cam)
        regions = [(r["start"], r["end"]) for r in vad["regions"]] if vad else [(s / C.SR, e / C.SR) for s, e in C.vad_regions(audio)]
        segs = C.split_for_asr(regions, 28.0)
        lid = (C.read_json(C.cam_dir(cam) / "whisper-large-v3.json", {}) or {}).get("regions", [])
        langs = [segment_language(a, b, lid) for a, b in segs]
        words: list[dict] = []
        rows: list[dict] = [None] * len(segs)    # type: ignore[list-item]
        with tempfile.TemporaryDirectory(prefix="canary_") as td:
            paths = []
            for k, (a, b) in enumerate(segs):
                p = Path(td) / f"s{k:04d}.wav"
                sf.write(str(p), audio[int(a * C.SR):int(b * C.SR)], C.SR, subtype="PCM_16")
                paths.append(str(p))
            for lang in sorted(set(langs)):
                idx = [k for k, lg in enumerate(langs) if lg == lang]
                for j in range(0, len(idx), 8):
                    part = idx[j:j + 8]
                    hyps = model.transcribe([paths[k] for k in part], source_lang=lang, target_lang=lang, timestamps=True, batch_size=len(part))
                    if isinstance(hyps, tuple):
                        hyps = hyps[0]
                    for k, h in zip(part, hyps):
                        ws = parse_hypothesis(h, segs[k][0], 0.08)
                        for w in ws:
                            w["seg"] = k
                            w["prob"] = None
                        words += ws
                        rows[k] = {"start": round(segs[k][0], 3), "end": round(segs[k][1], 3), "text": _hyp_text(h), "n_words": len(ws), "lang": lang}
        words.sort(key=lambda w: (w["start"], w["end"]))
        C.write_json(C.cam_dir(cam) / f"{NAME}.json",
                     {"stage": "canary", "version": C.STAGE_VERSION, "engine": "nemo-canary", "model": MODEL, "versions": C.versions("nemo", "torch"),
                      "cam": cam, "file": wav.name, "duration": round(dur, 3), "confidence": False, "device": device, "load_s": round(load_s, 2),
                      "seconds": round(time.time() - t1, 2), "params": {"max_segment_s": 28.0, "language": "from whisper region id, en/pt"},
                      "segments": rows, "words": words})
        C.log(f"canary {cam}: {len(words)} words in {len(segs)} segments ({time.time() - t1:.1f} s), pt segments {langs.count('pt')}")
        C.progress_tick("speech/canary", i + 1, len(todo))
        done[cam] = len(words)
    return done
