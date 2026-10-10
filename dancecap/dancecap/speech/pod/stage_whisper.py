"""Stage `whisper`: faster-whisper (CTranslate2) on every source, word timestamps + word probabilities, glossary prompt.

    python -m dancecap.speech.pod.run whisper --model large-v3 [--cams 00,01] [--passes auto,en,pt] [--force]

Per source: Silero VAD regions (stage vad) joined with 1 s of silence and decoded together (the neighbouring speech is context; each
word is mapped back to the region where it ends, common.remap_pass), then ONE decoding pass per entry of --passes:
  auto  language undetected, `multilingual=True` (language re-detected for every 30 s window: the class mixes English and Portuguese)
  en    forced English, glossary prompt (the Portuguese move names spelled correctly inside an English sentence)
  pt    forced Portuguese, Portuguese glossary prompt
plus Whisper's own language probabilities for every VAD region (`lid`, one encoder pass per region). The fusion decides per
utterance which pass to trust (by language probability, word probability and glossary hits); nothing is chosen here.

Output raw/<cam>/whisper-<model>.json:
  {stage, engine, model, version, cam, duration, gain_db, load_s, seconds, vram_mib, params, prompts,
   passes: [{requested, language, language_probability, seconds, words: [{word, start, end, prob, seg, region}], segments: [{start, end,
   text, avg_logprob, no_speech_prob, compression_ratio, temperature}]}], regions: [{start, end, lid: {en, pt, ...}, top}]}
Times are source seconds. Word texts keep Whisper's punctuation and case.
"""
from __future__ import annotations

import time

from . import common as C
from . import stage_vad

LID_LANGS = ("en", "pt", "es", "it", "fr", "gl", "de")


def prompts() -> dict:
    g = C.glossary().get("prompts", {})
    return {"en": g.get("whisper_initial_prompt", ""), "pt": g.get("whisper_initial_prompt_pt", ""), "hotwords": g.get("hotwords", "")}


def region_lid(model, audio, regions: list[dict], pad: float = 0.1) -> list[dict]:
    out = []
    for r in regions:
        a = audio[max(0, int((r["start"] - pad) * C.SR)):int((r["end"] + pad) * C.SR)]
        if a.size < int(0.3 * C.SR):
            out.append({"start": r["start"], "end": r["end"], "lid": None, "top": None})
            continue
        try:
            _, _, probs = model.detect_language(audio=a)
        except Exception as e:  # noqa: BLE001
            out.append({"start": r["start"], "end": r["end"], "lid": None, "top": None, "error": str(e)[:100]})
            continue
        d = dict(probs)
        out.append({"start": r["start"], "end": r["end"], "lid": {k: round(float(d.get(k, 0.0)), 4) for k in LID_LANGS},
                    "top": max(d.items(), key=lambda kv: kv[1])[0], "top_p": round(float(max(d.values())), 4)})
    return out


def decode_pass(model, src, table, requested: str, prompt: str, beam: int = 5) -> dict:
    t0 = time.time()
    kw = dict(word_timestamps=True, vad_filter=False, beam_size=beam, condition_on_previous_text=False)
    if requested == "auto":
        kw["language"] = None
        kw["multilingual"] = True
    else:
        kw["language"] = requested
    if prompt:
        kw["initial_prompt"] = prompt
    if src.size == 0:
        return {"requested": requested, "language": None, "language_probability": 0.0, "seconds": 0.0, "words": [], "segments": []}
    segments, info = model.transcribe(src, **kw)
    words, segs = [], []
    for s in segments:
        segs.append({"start": round(float(s.start), 3), "end": round(float(s.end), 3), "text": s.text.strip(),
                     "avg_logprob": round(float(s.avg_logprob), 4), "no_speech_prob": round(float(s.no_speech_prob), 4),
                     "compression_ratio": round(float(s.compression_ratio), 3),
                     "temperature": float(s.temperature) if s.temperature is not None else None})
        for w in s.words or []:
            words.append({"word": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                          "prob": round(float(w.probability), 4), "seg": len(segs) - 1})
    if table is not None:
        C.remap_pass(words, segs, table)
    return {"requested": requested, "language": getattr(info, "language", requested),
            "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
            "seconds": round(time.time() - t0, 2), "words": words, "segments": segs}


def run(cams: dict, model_name: str = "large-v3", passes: str = "auto,en,pt", force: bool = False, device: str = "cuda",
        compute_type: str | None = None, lid: bool = True) -> dict:
    todo = {c: w for c, w in cams.items() if force or not (C.cam_dir(c) / f"whisper-{model_name}.json").exists()}
    if not todo:
        C.log(f"whisper {model_name}: nothing to do")
        return {}
    import faster_whisper
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    ct = compute_type or ("float16" if device == "cuda" else "int8")
    t0 = time.time()
    path = download_model(model_name)
    vram0 = C.vram_used_mib() if device == "cuda" else None
    model = WhisperModel(path, device=device, compute_type=ct)
    load_s = time.time() - t0
    C.log(f"whisper {model_name} loaded in {load_s:.1f} s on {device}/{ct}")
    pr = prompts()
    wanted = [p.strip() for p in passes.split(",") if p.strip()]
    done = {}
    for i, (cam, wav) in enumerate(todo.items()):
        t1 = time.time()
        audio = C.load_audio(wav)
        dur = len(audio) / C.SR
        audio, gain = C.normalise_gain(audio)
        vad = stage_vad.load(cam)
        if vad is None:
            regs = C.vad_regions(audio)
            regions = [{"start": round(s / C.SR, 3), "end": round(e / C.SR, 3)} for s, e in regs]
        else:
            regions = vad["regions"]
            regs = [(int(r["start"] * C.SR), int(r["end"] * C.SR)) for r in regions]
        src, table = C.concat_with_gaps(audio, regs, 1.0)
        passes_out = []
        for p in wanted:
            prompt = pr["pt"] if p == "pt" else pr["en"]
            res = decode_pass(model, src, table if regs else None, p, prompt)
            C.log(f"whisper {model_name} {cam} pass {p}: {len(res['words'])} words, lang {res['language']} "
                  f"({res['language_probability']:.2f}) in {res['seconds']:.1f} s")
            passes_out.append(res)
        regions_lid = region_lid(model, audio, regions) if lid else regions
        out = {"stage": "whisper", "version": C.STAGE_VERSION, "engine": "faster-whisper", "model": model_name,
               "versions": C.versions("faster_whisper", "ctranslate2"), "cam": cam, "file": wav.name, "duration": round(dur, 3),
               "gain_db": gain, "device": device, "compute_type": ct, "load_s": round(load_s, 2),
               "seconds": round(time.time() - t1, 2), "vram_mib": {"before": vram0, "after": C.vram_used_mib() if device == "cuda" else None},
               "params": {"beam_size": 5, "condition_on_previous_text": False, "segmentation": "vad regions + 1 s gaps",
                          "vad": "silero 300 ms / 120 ms pad"},
               "prompts": {k: v for k, v in pr.items() if k != "hotwords"}, "passes": passes_out, "regions": regions_lid}
        C.write_json(C.cam_dir(cam) / f"whisper-{model_name}.json", out)
        C.progress_tick(f"speech/whisper-{model_name}", i + 1, len(todo))
        done[cam] = {"words": [len(p["words"]) for p in passes_out], "seconds": out["seconds"]}
    return done
