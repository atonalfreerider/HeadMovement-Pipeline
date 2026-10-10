"""Stage `regions`: a second, targeted look at hard passages (the jokes, laughter, places where the engines disagree).

    python -m dancecap.speech.pod.run regions --regions regions.json [--cams 00,01,02]

`regions.json` (written by the PC next to glossary.json, shipped in the code archive) is
  {"clocks": {"00": {"slope": 1.0000145, "offset": -8.9023}, ...},          source -> master clock (reference = slope * t + offset)
   "regions": [{"id": "r01", "label": "opening", "m0": -9.0, "m1": 16.0, "cams": ["00", "01", "02"]}, ...]}
For every region and phone the audio [m0, m1] (plus 1 s each side) is cut in the phone's own time and decoded by several settings:

  whisper-large-v3 / whisper-large-v3-turbo   en and auto, WITHOUT the glossary prompt (the prompt names "Larissa and Kadu", which can pull a
                                              doubtful sound towards those names), en WITH it, pt; plain and +12 dB louder
  parakeet-tdt-0.6b-v3                        as is
  canary-1b-v2                                en and pt

Whisper's decoder is deterministic for a given audio and prompt, so the variants are the only way to see its alternatives; the other two
engines are decoded once per language. Nothing is chosen here: every hypothesis is written with its words (times in SOURCE seconds) so the
fusion can vote and a human can read them side by side.

Output raw_regions/<region id>/<cam>.json: {region, cam, source_window, hypotheses: [{model, variant, language, text, words: [...], avg_logprob?}]}
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from . import common as C

PAD = 1.0


def load_spec(path: str | Path | None = None) -> dict:
    p = Path(path) if path else C.speech_dir() / "regions.json"
    return json.loads(p.read_text(encoding="utf-8"))


def source_window(clock: dict, m0: float, m1: float, duration: float) -> tuple[float, float]:
    """Master [m0, m1] -> the phone's own seconds, clipped to the recording; (0, 0) when the phone has nothing there."""
    a = (m0 - PAD - clock["offset"]) / clock["slope"]
    b = (m1 + PAD - clock["offset"]) / clock["slope"]
    a, b = max(0.0, a), min(duration, b)
    return (a, b) if b - a >= 1.0 else (0.0, 0.0)


def _whisper_words(segments, t_off: float) -> tuple[list[dict], str, float | None]:
    words, texts, lps = [], [], []
    for s in segments:
        texts.append(s.text.strip())
        lps.append(float(s.avg_logprob))
        for w in s.words or []:
            words.append({"word": w.word.strip(), "start": round(t_off + float(w.start), 3), "end": round(t_off + float(w.end), 3),
                          "prob": round(float(w.probability), 4)})
    return words, " ".join(texts), (round(float(np.mean(lps)), 4) if lps else None)


def run(cams: dict, regions_path: str | None = None, force: bool = False, device: str = "cuda") -> dict:
    spec = load_spec(regions_path)
    clocks = spec["clocks"]
    out_root = C.raw_dir().parent / "raw_regions"
    todo = []
    for r in spec["regions"]:
        for cam in r.get("cams") or list(cams):
            if cam in cams and cam in clocks and (force or not (out_root / r["id"] / f"{cam}.json").exists()):
                todo.append((r, cam))
    if not todo:
        C.log("regions: nothing to do")
        return {}
    import soundfile as sf
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    import nemo.collections.asr as nemo_asr
    from .stage_parakeet import _hyp_text, parse_hypothesis, word_confidences
    from .stage_parakeet import MODEL as PK_MODEL
    from .stage_canary import MODEL as CA_MODEL
    t0 = time.time()
    whisper = {name: WhisperModel(download_model(name), device=device, compute_type="float16") for name in ("large-v3", "large-v3-turbo")}
    parakeet = nemo_asr.models.ASRModel.from_pretrained(model_name=PK_MODEL).eval().to(device)
    canary = nemo_asr.models.ASRModel.from_pretrained(model_name=CA_MODEL).eval().to(device)
    C.log(f"regions: models loaded in {time.time() - t0:.1f} s")
    g = C.glossary().get("prompts", {})
    prompt_en = g.get("whisper_initial_prompt", "")
    audio_cache: dict[str, np.ndarray] = {}
    done = {}
    for r, cam in todo:
        wav = cams[cam]
        audio = audio_cache.setdefault(cam, C.load_audio(wav))
        a, b = source_window(clocks[cam], r["m0"], r["m1"], len(audio) / C.SR)
        if b <= a:
            continue
        seg = audio[int(a * C.SR):int(b * C.SR)]
        loud, gain = C.normalise_gain(seg)
        boosted = (np.clip(seg * 10 ** (12 / 20), -1, 1)).astype(np.float32)
        hyps = []
        for name, model in whisper.items():
            for lang, prompt, label, x in (("en", None, "en no-prompt", seg), ("auto", None, "auto no-prompt", seg), ("en", prompt_en, "en glossary-prompt", seg),
                                           ("pt", None, "pt no-prompt", seg), ("en", None, "en no-prompt +12dB", boosted)):
                kw = dict(word_timestamps=True, beam_size=5, condition_on_previous_text=False, vad_filter=False)
                if lang == "auto":
                    kw.update(language=None, multilingual=True)
                else:
                    kw["language"] = lang
                if prompt:
                    kw["initial_prompt"] = prompt
                segs, info = model.transcribe(x, **kw)
                words, text, lp = _whisper_words(segs, a)
                hyps.append({"model": f"whisper-{name}", "variant": label, "language": getattr(info, "language", lang), "text": text, "avg_logprob": lp,
                             "words": words})
        with tempfile.TemporaryDirectory(prefix="regions_") as td:
            p = str(Path(td) / "seg.wav")
            sf.write(p, seg, C.SR, subtype="PCM_16")
            h = parakeet.transcribe([p], batch_size=1, timestamps=True, verbose=False)
            h = h[0][0] if isinstance(h, tuple) else h[0]
            nw = len((getattr(h, "timestamp", None) or {}).get("word") or [])
            hyps.append({"model": "parakeet-tdt-0.6b-v3", "variant": "default", "language": "auto", "text": _hyp_text(h),
                         "words": parse_hypothesis(h, a, 0.08, word_confidences(parakeet, h, nw))})
            for lang in ("en", "pt"):
                h = canary.transcribe([p], source_lang=lang, target_lang=lang, timestamps=True, batch_size=1)
                h = h[0][0] if isinstance(h, tuple) else h[0]
                hyps.append({"model": "canary-1b-v2", "variant": lang, "language": lang, "text": _hyp_text(h), "words": parse_hypothesis(h, a, 0.08)})
        C.write_json(out_root / r["id"] / f"{cam}.json", {"region": r, "cam": cam, "source_window": [round(a, 3), round(b, 3)], "gain_db_loudnorm": gain,
                                                          "hypotheses": hyps})
        C.log(f"regions {r['id']} cam {cam}: {len(hyps)} hypotheses on [{a:.1f}, {b:.1f}] s")
        done[f"{r['id']}/{cam}"] = len(hyps)
    return done
