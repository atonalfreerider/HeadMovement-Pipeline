"""Pod-side quality gate of the voice-to-voice pieces (agent V1): an independent ASR and a speaker embedding of every candidate audio.

    python -m dancecap.speech.pod.stage_v2v asr --manifest <dir>/manifest.json --out <dir>/out [--model large-v3] [--device cuda]

manifest.json {"items": [{"key": "p004__v1", "file": "wav/p004__v1.wav", ...}, ...]}: 16 kHz mono wavs (the pieces of the original mix, the audio sent to
the voice changer, and every converted candidate). For each item:

  * faster-whisper large-v3 (the engine of the pipeline, but NO prompt / glossary: it must not be helped towards the transcript), English forced,
    word timestamps, a light Silero VAD, no conditioning on previous text -> text, words (start, end, probability), avg log-probability
  * a speechbrain ECAPA-TDNN embedding of the whole file (192-d, L2-normalised) for the voice check (the PC compares it with the voice's reference)

Output: <out>/asr.json {model, versions, seconds, items: {key: {text, words, avg_logprob, no_speech_prob, seconds}}}, <out>/embeddings.npz {keys, ecapa}.
The wavs are read as they are; nothing here talks to the network except the model download from the volume's HF cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C


def transcribe_all(items: list[dict], root: Path, model_name: str, device: str) -> dict:
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    ct = "float16" if device == "cuda" else "int8"
    model = WhisperModel(download_model(model_name), device=device, compute_type=ct)
    out = {}
    for i, it in enumerate(items):
        t0 = time.time()
        audio = C.load_audio(root / it["file"])
        segs, info = model.transcribe(audio, language="en", beam_size=5, word_timestamps=True, vad_filter=True,
                                      vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
                                      condition_on_previous_text=False, temperature=[0.0, 0.2, 0.4])
        words, texts, lp, ns = [], [], [], []
        for s in segs:
            texts.append(s.text.strip())
            lp.append(float(s.avg_logprob))
            ns.append(float(s.no_speech_prob))
            for w in (s.words or []):
                words.append({"word": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3), "prob": round(float(w.probability), 4)})
        out[it["key"]] = {"text": " ".join(texts), "words": words, "avg_logprob": round(float(np.mean(lp)), 4) if lp else None,
                          "no_speech_prob": round(float(np.max(ns)), 4) if ns else None, "seconds": round(time.time() - t0, 2),
                          "duration": round(len(audio) / C.SR, 3)}
        C.log(f"asr {i + 1}/{len(items)} {it['key']}: {len(words)} words in {time.time() - t0:.1f} s")
    return out


def enhance_all(items: list[dict], root: Path, out: Path, device: str) -> list[dict]:
    """speech enhancement (speechbrain SepFormer trained on DNS4, 16 kHz: noise, music and reverb removed) of the items flagged "enhance": true; writes
    <out>/enh/<key>__enh.wav and returns the new items (kind "<kind>_enh") so the ASR and the voice check see them too"""
    import soundfile as sf
    import torch
    try:
        from speechbrain.inference.separation import SepformerSeparation as Sep
    except ImportError:
        from speechbrain.pretrained import SepformerSeparation as Sep
    root_w = C._hm_root() / "weights" / "speech" / "sepformer-dns4"
    model = Sep.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement", savedir=str(root_w), run_opts={"device": device})
    new = []
    (out / "enh").mkdir(parents=True, exist_ok=True)
    for it in items:
        if not it.get("enhance"):
            continue
        a = C.load_audio(root / it["file"])
        t0 = time.time()
        with torch.no_grad():
            est = model.separate_batch(torch.from_numpy(a)[None, :].to(device))
        y = est[0, :, 0].float().cpu().numpy()
        peak = float(np.abs(y).max()) + 1e-9
        y = (y / peak * 0.5).astype(np.float32)
        key = f"{it['key']}__enh"
        sf.write(str(out / "enh" / f"{key}.wav"), y, C.SR, subtype="PCM_16")
        new.append({"key": key, "file": os.path.relpath(out / "enh" / f"{key}.wav", root).replace(os.sep, "/"), "piece": it.get("piece"), "kind": f"{it.get('kind')}_enh"})
        C.log(f"enhance {it['key']}: {len(a) / C.SR:.1f} s in {time.time() - t0:.1f} s")
    return new


def embed_all(items: list[dict], root: Path, device: str) -> tuple[list[str], np.ndarray]:
    import torch
    from .stage_embed import load_ecapa
    clf = load_ecapa(device)
    keys, vecs = [], []
    with torch.no_grad():
        for it in items:
            a = C.load_audio(root / it["file"])
            if len(a) < C.SR // 2:
                a = np.pad(a, (0, C.SR // 2 - len(a)))
            e = clf.encode_batch(torch.from_numpy(a)[None, :].to(device)).squeeze().float().cpu().numpy()
            vecs.append(e / (np.linalg.norm(e) + 1e-9))
            keys.append(it["key"])
    return keys, np.stack(vecs) if vecs else np.zeros((0, 192), np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.speech.pod.stage_v2v")
    ap.add_argument("step", choices=["asr"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enhance", action="store_true", help="also enhance the items flagged enhance:true (SepFormer DNS4) and transcribe the result")
    a = ap.parse_args(argv)
    man = Path(a.manifest)
    items = json.loads(man.read_text(encoding="utf-8"))["items"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.enhance:
        try:
            items = items + enhance_all(items, man.parent, out, a.device)
        except Exception as ex:  # noqa: BLE001  the plain ASR still runs
            C.log(f"enhancement unavailable: {type(ex).__name__}: {str(ex)[:300]}")
    asr = transcribe_all(items, man.parent, a.model, a.device)
    C.write_json(out / "asr.json", {"model": f"faster-whisper {a.model} (no prompt, en, light VAD)", "versions": C.versions("faster_whisper", "ctranslate2"),
                                    "seconds": round(time.time() - t0, 1), "items": asr})
    status = "ok"
    try:
        keys, e = embed_all(items, man.parent, a.device)
        np.savez(out / "embeddings.npz", keys=np.array(keys), ecapa=e)
    except Exception as ex:  # noqa: BLE001  the ASR result stays usable without the voice check
        status = f"unavailable: {type(ex).__name__}: {str(ex)[:200]}"
        C.log(f"ECAPA {status}")
    C.write_json(out / "done.json", {"items": len(items), "embeddings": status, "seconds": round(time.time() - t0, 1)})
    C.log(f"v2v gate done: {len(items)} items in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
