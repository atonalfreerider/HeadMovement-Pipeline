"""Standalone faster-whisper worker for the narration tool (no dancecap imports: it runs in the interpreter that has
faster-whisper - the dedicated .venv-asr per docs/MOVES_RESEARCH.md A2).

    python asr_worker.py --wav narration.wav --out transcript.json [--model large-v3] [--device cuda]
                         [--compute-type float16] [--passes en,pt] [--hotwords "Corredor, Viradinha, ..."]
                         [--lid] [--no-normalize] [--allow-download] [--check]

One model load, then one decoding pass per language of --passes (default "en"; "auto" = Whisper's own language
detection; the legacy --language X means --passes X). The narrator speaks English with Portuguese move names and
sometimes Portuguese phrases, so the narration tool decodes twice (en + pt) and fuses the two per utterance by
vocabulary match and probability (dancecap.moves.annotate.asr.fuse_passes).
Decoding (MOVES_RESEARCH A1-A3): word_timestamps, beam 5, condition_on_previous_text off, hotwords = the move
vocabulary (faster-whisper puts them in the prompt of every 30 s window; trimmed here to whole terms within ~220
tokens). Segmentation (--segmentation, measured on the user's 6-minute narration): 'gaps' (default) runs Silero VAD
first (min silence 400 ms; a recording without speech never loads the model), joins the speech regions with 1 s of
silence and maps every word back to the region where it ENDS - faster-whisper's own vad_filter joins them without a
gap and moved words up to 7 s into the wrong region; 'clips' (each region alone) lost confidence and hallucinated
"Thanks for watching!". Quiet recordings are gained up first (peak -> -3 dBFS, at most +30 dB; one browser narration
peaked at -29 dBFS). --lid adds Whisper's language probabilities for each VAD speech region (en / pt / es ...; one
extra encoder pass per region; measured: all regions of the user's narration are English, so it is off by default).
Turing GPU: float16, never bfloat16. The model is loaded with local_files_only unless --allow-download. On Windows the
pip CUDA wheels (nvidia-cublas-cu12) are put first: CUDA_PATH -> site-packages/nvidia/cublas (CTranslate2 calls
SetDllDirectory on %CUDA_PATH%\\bin before PATH, and the machine's CUDA_PATH is the older 12.2) and their bin folders
are prepended to PATH. The process exits when done, so the GPU memory is always returned.

Output JSON: {engine, model, model_path, device, compute_type, duration, gain_db, load_s, seconds, vram_used_mib,
  passes: [{language, requested, language_probability, seconds, words:[{word, start, end, prob}],
            segments:[{start, end, text, avg_logprob, no_speech_prob, compression_ratio, temperature, language}]}],
  regions: [{start, end, lid: {en: p, pt: p, ...}}]   (--lid only)
  language, words, segments (= the first pass, for older readers)}
--check prints {faster_whisper, ctranslate2, cuda_devices, model_cached, model_path, models_cached}.
"""
from __future__ import annotations

import argparse
import json
import os
import site
import subprocess
import sys
import time
from pathlib import Path

SR = 16000
KNOWN_MODELS = ("large-v3", "large-v3-turbo")
LID_LANGS = ("en", "pt", "es", "it", "fr", "gl")


def setup_cuda_dlls() -> list[str]:
    added = []
    roots = []
    try:
        roots += site.getsitepackages()
    except Exception:  # noqa: BLE001
        pass
    roots.append(str(Path(sys.prefix) / "Lib" / "site-packages"))
    for sp in dict.fromkeys(roots):
        nv = Path(sp) / "nvidia"
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            b = nv / sub / "bin"
            if b.is_dir():
                os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(b))
                    except OSError:
                        pass
                if sub == "cublas":
                    os.environ["CUDA_PATH"] = str(nv / sub)
                added.append(str(b))
    return added


def vram_used_mib() -> int | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10,
                           creationflags=0x08000000 if os.name == "nt" else 0)
        return int(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def log(msg: str) -> None:
    print(f"[asr_worker] {msg}", file=sys.stderr, flush=True)


def check(model: str) -> dict:
    info = {"python": sys.executable}
    try:
        import ctranslate2
        import faster_whisper
        info["faster_whisper"] = getattr(faster_whisper, "__version__", "?")
        info["ctranslate2"] = getattr(ctranslate2, "__version__", "?")
        try:
            info["cuda_devices"] = int(ctranslate2.get_cuda_device_count())
        except Exception as e:  # noqa: BLE001
            info["cuda_devices"] = 0
            info["cuda_error"] = str(e)[:200]
        from faster_whisper.utils import download_model
        cached = {}
        for m in dict.fromkeys((model, *KNOWN_MODELS)):
            try:
                cached[m] = download_model(m, local_files_only=True)
            except Exception as e:  # noqa: BLE001
                if m == model:
                    info["model_error"] = f"{type(e).__name__}: {e}"[:200]
        info["models_cached"] = sorted(cached)
        info["model_cached"] = model in cached
        if model in cached:
            info["model_path"] = cached[model]
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"[:300]
    return info


def normalise_gain(audio, target_peak: float = 0.708, quiet_peak: float = 0.25, max_gain_db: float = 30.0):
    """(audio, gain dB): a quiet recording (peak below -12 dBFS) is scaled so its peak reaches -3 dBFS (at most
    +30 dB; never clips). Whisper's log-mel is fairly level-invariant, Silero VAD is not."""
    import numpy as np
    if audio.size == 0:
        return audio, 0.0
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-6 or peak >= quiet_peak:
        return audio, 0.0
    g = min(target_peak / peak, 10 ** (max_gain_db / 20))
    return (audio * g).astype(np.float32), round(20 * float(np.log10(g)), 2)


def trim_hotwords(model, hotwords: str, max_tokens: int = 220) -> str:
    """Whole comma-separated terms that fit faster-whisper's prompt budget (it cuts at 223 tokens mid-word)."""
    terms = [t.strip() for t in hotwords.split(",") if t.strip()]
    try:
        enc = model.hf_tokenizer.encode
        n = lambda s: len(enc(" " + s, add_special_tokens=False).ids)  # noqa: E731
    except Exception:  # noqa: BLE001
        n = lambda s: len(s) // 3 + 1  # noqa: E731
    out = []
    for t in terms:
        if n(", ".join(out + [t])) > max_tokens:
            break
        out.append(t)
    return ", ".join(out)


def speech_regions(audio, min_silence_ms: int, pad_ms: int = 200) -> list[tuple[int, int]]:
    """Silero VAD speech regions (sample indices)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms))
    return [(int(r["start"]), int(r["end"])) for r in ts]


def concat_with_gaps(audio, regions: list[tuple[int, int]], gap_s: float):
    """The speech regions back to back with gap_s of silence between them -> (audio, table [(c0, c1, o0)]):
    concatenated samples c0..c1 come from original sample o0. faster-whisper's own vad_filter joins the regions
    WITHOUT a gap; with sparse one- or two-word narration its word alignment then moves words across the joins
    (measured: 'Open' 1.5 s before its region, 'Body' 7 s early). A short silence keeps every word inside its own
    region."""
    import numpy as np
    gap = np.zeros(int(gap_s * SR), dtype=np.float32)
    parts, table, pos = [], [], 0
    for s, e in regions:
        if parts:
            parts.append(gap)
            pos += gap.size
        parts.append(audio[s:e])
        table.append((pos, pos + (e - s), s))
        pos += e - s
    return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), table


def region_of(t: float, table: list[tuple[int, int, int]], eps: float = 0.03) -> int:
    """Index of the region holding concatenated time t (seconds); a time in a gap belongs to the region before."""
    x = t * SR
    k = 0
    for i, (c0, c1, _) in enumerate(table):
        if x >= c0 - eps * SR:
            k = i
    return k


def remap_pass(words: list[dict], segs: list[dict], table) -> None:
    """In place: concatenated-audio times -> recording seconds. A word belongs to the region where it ENDS:
    Whisper's word starts absorb the silence before a word (after a gap the start lies in the gap or even in the
    previous region), its ends are reliable. The start is clipped to that region."""
    def conv(t: float, k: int) -> float:
        c0, c1, o0 = table[k]
        x = min(max(t * SR, c0), c1)
        return (o0 + x - c0) / SR

    for w in words:
        k = region_of(max(w["start"], w["end"] - 0.05), table)
        w["start"], w["end"] = round(conv(w["start"], k), 3), round(conv(w["end"], k), 3)
        if w["end"] < w["start"]:
            w["start"] = w["end"]
    for sg in segs:
        s0, s1 = sg["start"], sg["end"]
        sg["start"] = round(conv(s0, region_of(s0 + 0.05, table)), 3)
        sg["end"] = round(conv(s1, region_of(max(s0, s1 - 0.05), table)), 3)


def region_lid(model, audio, regions: list[dict], pad: float = 0.1) -> list[dict]:
    out = []
    for r in regions:
        a = audio[max(0, int((r["start"] - pad) * SR)):int((r["end"] + pad) * SR)]
        if a.size < int(0.2 * SR):
            continue
        try:
            _, _, probs = model.detect_language(audio=a)
        except Exception as e:  # noqa: BLE001
            out.append({**r, "error": str(e)[:120]})
            continue
        d = dict(probs)
        out.append({"start": r["start"], "end": r["end"],
                    "lid": {k: round(float(d.get(k, 0.0)), 4) for k in LID_LANGS},
                    "top": max(d.items(), key=lambda kv: kv[1])[0]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--out")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--passes", default=None, help="comma list of languages, e.g. en,pt (auto = detect)")
    ap.add_argument("--language", default=None, help="legacy: one pass in this language")
    ap.add_argument("--hotwords", default="")
    ap.add_argument("--initial-prompt", default="")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--vad-min-silence", type=int, default=400)
    ap.add_argument("--segmentation", choices=["gaps", "clips", "vad", "none"], default="gaps",
                    help="gaps (default): the VAD speech regions joined with --gap s of silence, decoded together "
                         "(neighbouring narration is context) and mapped back per region; clips: every region "
                         "decoded alone (measured on a real narration: word probabilities fall, 'Thanks for "
                         "watching!' hallucinated, 3x slower); vad: faster-whisper's own vad_filter (no gaps: words "
                         "drift across the joins, up to 7 s); none: the whole recording")
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--lid", action="store_true")
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    dlls = setup_cuda_dlls() if os.name == "nt" else []
    if a.check:
        info = check(a.model)
        info["cuda_dll_dirs"] = dlls
        print(json.dumps(info))
        return 0 if "error" not in info else 2

    t_start = time.time()
    from faster_whisper import WhisperModel, decode_audio
    from faster_whisper.utils import download_model

    audio = decode_audio(a.wav, sampling_rate=SR)
    duration = audio.size / SR
    gain_db = 0.0
    if not a.no_normalize:
        audio, gain_db = normalise_gain(audio)
        if gain_db:
            log(f"quiet recording: +{gain_db:.1f} dB")
    vad = {"min_silence_duration_ms": a.vad_min_silence}
    passes_req = [p.strip() for p in (a.passes or a.language or "en").split(",") if p.strip()]

    table, regions_s = None, []
    src = audio
    clips = None
    if a.segmentation in ("gaps", "clips"):  # VAD first: a recording without speech never loads the model
        regs = speech_regions(audio, a.vad_min_silence)
        regions_s = [{"start": round(s / SR, 3), "end": round(e / SR, 3)} for s, e in regs]
        if a.segmentation == "gaps":
            src, table = concat_with_gaps(audio, regs, a.gap)
        else:
            clips = [x for r in regions_s for x in (r["start"], r["end"])]
            if not clips:
                src = audio[:0]
        log(f"{len(regs)} speech regions ({sum(e - s for s, e in regs) / SR:.1f} s of {duration:.1f} s), "
            f"segmentation {a.segmentation}")

    model_path = download_model(a.model, local_files_only=not a.allow_download)
    model, vram0, vram1, load_s = None, None, None, 0.0
    if src.size:
        t0 = time.time()
        vram0 = vram_used_mib() if a.device == "cuda" else None
        model = WhisperModel(model_path, device=a.device, compute_type=a.compute_type, cpu_threads=a.cpu_threads)
        load_s = time.time() - t0
        vram1 = vram_used_mib() if a.device == "cuda" else None
        log(f"{a.model} loaded on {a.device}/{a.compute_type} in {load_s:.1f} s")
    hotwords = trim_hotwords(model, a.hotwords) if a.hotwords and model is not None else ""

    passes = []
    vram_peak = vram1
    for lang in passes_req:
        t0 = time.time()
        kw = dict(language=None if lang == "auto" else lang, word_timestamps=True,
                  vad_filter=a.segmentation == "vad", vad_parameters=vad, beam_size=a.beam_size,
                  condition_on_previous_text=False)
        if hotwords:
            kw["hotwords"] = hotwords
        if a.initial_prompt:
            kw["initial_prompt"] = a.initial_prompt
        if lang == "auto":
            kw["multilingual"] = True
        if clips:
            kw["clip_timestamps"] = clips
        if src.size == 0:  # no speech at all: nothing to decode (Whisper would only hallucinate)
            passes.append({"language": None if lang == "auto" else lang, "requested": lang,
                           "language_probability": 0.0, "seconds": 0.0, "words": [], "segments": []})
            continue
        segments, info = model.transcribe(src, **kw)
        words, segs = [], []
        for s in segments:
            segs.append({"start": round(float(s.start), 3), "end": round(float(s.end), 3), "text": s.text.strip(),
                         "avg_logprob": round(float(s.avg_logprob), 4),
                         "no_speech_prob": round(float(s.no_speech_prob), 4),
                         "compression_ratio": round(float(s.compression_ratio), 3),
                         "temperature": float(s.temperature) if s.temperature is not None else None})
            for w in s.words or []:
                words.append({"word": w.word.strip(), "start": round(float(w.start), 3),
                              "end": round(float(w.end), 3), "prob": round(float(w.probability), 4),
                              "seg": len(segs) - 1})
        if table is not None:
            remap_pass(words, segs, table)
        if a.device == "cuda":
            v = vram_used_mib()
            vram_peak = max(x for x in (vram_peak, v) if x is not None) if v is not None else vram_peak
        passes.append({"language": getattr(info, "language", lang), "requested": lang,
                       "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
                       "seconds": round(time.time() - t0, 2), "words": words, "segments": segs})
        log(f"pass {lang}: {len(words)} words in {time.time() - t0:.1f} s")

    regions = regions_s
    if a.lid and model is not None:
        t0 = time.time()
        if not regions:
            regions = [{"start": round(s / SR, 3), "end": round(e / SR, 3)}
                       for s, e in speech_regions(audio, a.vad_min_silence)]
        regions = region_lid(model, audio, regions)
        log(f"language id on {len(regions)} speech regions in {time.time() - t0:.1f} s")

    first = passes[0] if passes else {"words": [], "segments": [], "language": None}
    out = {"engine": "faster-whisper", "model": a.model, "model_path": str(model_path), "device": a.device,
           "compute_type": a.compute_type, "duration": round(duration, 3), "gain_db": gain_db,
           "load_s": round(load_s, 2), "seconds": round(time.time() - t_start, 2),
           "vram_used_mib": {"before_load": vram0, "after_load": vram1, "peak_sampled": vram_peak},
           "beam_size": a.beam_size, "vad": vad, "segmentation": a.segmentation, "gap_s": a.gap,
           "hotwords": hotwords or None, "passes": passes, "regions": regions,
           "language": first["language"], "words": first["words"], "segments": first["segments"]}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
