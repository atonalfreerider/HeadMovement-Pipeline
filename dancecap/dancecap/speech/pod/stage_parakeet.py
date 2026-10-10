"""Stage `parakeet`: NVIDIA parakeet-tdt-0.6b-v3 (NeMo FastConformer-TDT, 25 European languages incl. English and Portuguese, language
detected by the model itself) on every source. A different architecture and training set than Whisper, so its errors are mostly
different ones: that is what the fusion needs. No prompt / hotwords (the model has none); the glossary is applied afterwards.

    python -m dancecap.speech.pod.run parakeet [--cams 00,01] [--force]

Per source the VAD speech regions are joined into segments of at most 28 s (common.split_for_asr), each decoded as its own utterance
with word timestamps; word times are offset back to source seconds. Word confidences come from NeMo's confidence estimator when it
can be switched on (max softmax probability per token, mean over the word's tokens, 0..1; the entropy variants gave values near 0 for
correct words in a test on camera 04, so they were not used); `confidence` in the output says whether they exist.

Output raw/<cam>/parakeet-tdt-0.6b-v3.json:
  {stage, engine, model, versions, cam, duration, confidence: bool, segments: [{start, end, text, n_words}],
   words: [{word, start, end, prob (null without confidence), seg}], seconds, ...}
"""
from __future__ import annotations

import copy
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import stage_vad

MODEL = "nvidia/parakeet-tdt-0.6b-v3"
NAME = "parakeet-tdt-0.6b-v3"


def load_model(device: str = "cuda"):
    import torch
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL)
    m.eval()
    m = m.to(device)
    conf = False
    try:
        from omegaconf import open_dict
        dec = copy.deepcopy(m.cfg.decoding)
        with open_dict(dec):
            dec.preserve_alignments = True
            dec.confidence_cfg = {"preserve_frame_confidence": True, "preserve_token_confidence": True, "preserve_word_confidence": False,
                                  "exclude_blank": True, "aggregation": "mean",
                                  "method_cfg": {"name": "max_prob"}}      # softmax max-probability per token, mean over the word's tokens
            if "compute_timestamps" in dec:
                dec.compute_timestamps = True
        m.change_decoding_strategy(dec)
        conf = True
    except Exception as e:  # noqa: BLE001
        C.log(f"parakeet: word confidence not available ({type(e).__name__}: {str(e)[:120]})")
    return m, conf


def word_confidences(model, h, n_words: int) -> list[float | None]:
    """Mean token confidence of every word (a word starts at a sentencepiece token that begins with U+2581). NeMo's own word-level
    aggregation raised RuntimeError on a hypothesis of camera 01 ("no'cause": 76 words, 77 confidences), so the words are rebuilt here; a
    hypothesis whose tokens do not add up to `n_words` words gets None instead of failing the stage."""
    tc = getattr(h, "token_confidence", None)
    ids = getattr(h, "y_sequence", None)
    none = [None] * n_words
    if tc is None or ids is None:
        return none
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    if len(tc) != len(ids):
        return none
    try:
        toks = model.tokenizer.ids_to_tokens(ids)
    except Exception:  # noqa: BLE001
        return none
    groups: list[list[float]] = []
    for tok, c in zip(toks, tc):
        if (tok.startswith("▁") or not groups) and tok != "▁":
            groups.append([])
        if groups:
            groups[-1].append(float(c))
    if len(groups) != n_words:
        return none
    return [round(sum(g) / len(g), 4) for g in groups]


def _hyp_text(h) -> str:
    return (getattr(h, "text", None) or (h if isinstance(h, str) else "")).strip()


def parse_hypothesis(h, offset: float, stride: float = 0.08, conf: list | None = None) -> list[dict]:
    """Words {word, start, end, prob} of one NeMo Hypothesis, times shifted by `offset` seconds. Handles timestamp dicts that carry
    seconds ('start' / 'end') or only frame offsets ('start_offset' / 'end_offset' x stride)."""
    ts = getattr(h, "timestamp", None) or {}
    words = ts.get("word") or []
    if conf is None:
        conf = getattr(h, "word_confidence", None)
    out = []
    for i, w in enumerate(words):
        s = w.get("start")
        e = w.get("end")
        if s is None and w.get("start_offset") is not None:
            s = float(w["start_offset"]) * stride
        if e is None and w.get("end_offset") is not None:
            e = float(w["end_offset"]) * stride
        if s is None or e is None:
            continue
        p = None
        if conf is not None and i < len(conf):
            try:
                p = round(float(conf[i]), 4)
            except (TypeError, ValueError):
                p = None
        out.append({"word": str(w.get("word", w.get("char", ""))).strip(), "start": round(offset + float(s), 3),
                    "end": round(offset + float(e), 3), "prob": p})
    return out


def transcribe_segments(model, audio: np.ndarray, segs: list[tuple[float, float]], batch_size: int = 8) -> tuple[list[dict], list[dict]]:
    import soundfile as sf
    words: list[dict] = []
    seg_rows: list[dict] = []
    stride = 0.08
    try:
        stride = float(model.cfg.preprocessor.window_stride) * 8
    except Exception:  # noqa: BLE001
        pass
    with tempfile.TemporaryDirectory(prefix="parakeet_") as td:
        paths = []
        for i, (a, b) in enumerate(segs):
            p = Path(td) / f"s{i:04d}.wav"
            sf.write(str(p), audio[int(a * C.SR):int(b * C.SR)], C.SR, subtype="PCM_16")
            paths.append(str(p))
        for k in range(0, len(paths), batch_size):
            chunk = paths[k:k + batch_size]
            try:
                hyps = model.transcribe(chunk, batch_size=batch_size, timestamps=True, verbose=False)
            except TypeError:
                hyps = model.transcribe(chunk, batch_size=batch_size, timestamps=True)
            if isinstance(hyps, tuple):          # older NeMo: (best_hyps, all_hyps)
                hyps = hyps[0]
            for j, h in enumerate(hyps):
                a, b = segs[k + j]
                nw = len((getattr(h, "timestamp", None) or {}).get("word") or [])
                ws = parse_hypothesis(h, a, stride, word_confidences(model, h, nw))
                for w in ws:
                    w["seg"] = k + j
                words += ws
                seg_rows.append({"start": round(a, 3), "end": round(b, 3), "text": _hyp_text(h), "n_words": len(ws)})
    return words, seg_rows


def run(cams: dict, force: bool = False, device: str = "cuda") -> dict:
    todo = {c: w for c, w in cams.items() if force or not (C.cam_dir(c) / f"{NAME}.json").exists()}
    if not todo:
        C.log("parakeet: nothing to do")
        return {}
    t0 = time.time()
    model, conf = load_model(device)
    load_s = time.time() - t0
    C.log(f"parakeet loaded in {load_s:.1f} s (word confidence: {conf})")
    done = {}
    for i, (cam, wav) in enumerate(todo.items()):
        t1 = time.time()
        audio = C.load_audio(wav)
        dur = len(audio) / C.SR
        vad = stage_vad.load(cam)
        if vad is None:
            regions = [(s / C.SR, e / C.SR) for s, e in C.vad_regions(audio)]
        else:
            regions = [(r["start"], r["end"]) for r in vad["regions"]]
        segs = C.split_for_asr(regions, 28.0)
        words, seg_rows = transcribe_segments(model, audio, segs)
        out = {"stage": "parakeet", "version": C.STAGE_VERSION, "engine": "nemo-parakeet", "model": MODEL,
               "versions": C.versions("nemo", "torch"), "cam": cam, "file": wav.name, "duration": round(dur, 3),
               "confidence": conf and any(w["prob"] is not None for w in words), "device": device, "load_s": round(load_s, 2),
               "seconds": round(time.time() - t1, 2), "params": {"max_segment_s": 28.0, "segmentation": "silero vad regions"},
               "segments": seg_rows, "words": words}
        C.write_json(C.cam_dir(cam) / f"{NAME}.json", out)
        C.log(f"parakeet {cam}: {len(words)} words in {len(segs)} segments ({time.time() - t1:.1f} s)")
        C.progress_tick("speech/parakeet", i + 1, len(todo))
        done[cam] = len(words)
    return done
