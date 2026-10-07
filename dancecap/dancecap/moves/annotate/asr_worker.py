"""Standalone faster-whisper worker for the narration tool (no dancecap imports: it runs in whichever interpreter has
faster-whisper - a dedicated .venv-asr per docs/MOVES_RESEARCH.md A2, or the current one).

    python asr_worker.py --wav narration.wav --out transcript.json [--model large-v3-turbo] [--device cuda]
                         [--compute-type float16] [--language en] [--hotwords "Corredor Viradinha ..."]
                         [--allow-download] [--check]

Decisions (MOVES_RESEARCH A1-A3): word_timestamps, VAD (min silence 400 ms), beam 5, condition_on_previous_text off,
hotwords = the move vocabulary (faster-whisper prepends them to every 30 s window, max ~223 tokens). Turing GPU:
float16, never bfloat16. The model is loaded with local_files_only unless --allow-download (the ~1.6 GB
large-v3-turbo download needs the user's OK). On Windows the pip CUDA wheels (nvidia-cublas-cu12, optional
nvidia-cudnn-cu12) are put first: CUDA_PATH -> site-packages/nvidia/cublas (CTranslate2 calls SetDllDirectory on
%CUDA_PATH%\\bin before PATH) and their bin folders are prepended to PATH.
Output JSON: {engine, model, device, compute_type, language, duration, words:[{word, start, end, prob}],
segments:[{start, end, text}]}; --check prints {faster_whisper, ctranslate2, cuda_devices, model_cached}.
"""
from __future__ import annotations

import argparse
import json
import os
import site
import sys
from pathlib import Path


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
        for sub in ("cublas", "cudnn", "cuda_runtime"):
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--out")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default="en")
    ap.add_argument("--hotwords", default="")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    dlls = setup_cuda_dlls() if os.name == "nt" else []
    if a.check:
        info = {"python": sys.executable, "cuda_dll_dirs": dlls}
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
            try:
                from faster_whisper.utils import download_model
                info["model_path"] = download_model(a.model, local_files_only=True)
                info["model_cached"] = True
            except Exception as e:  # noqa: BLE001
                info["model_cached"] = False
                info["model_error"] = f"{type(e).__name__}: {e}"[:200]
        except Exception as e:  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"[:300]
        print(json.dumps(info))
        return 0 if "error" not in info else 2
    from faster_whisper import WhisperModel
    model = WhisperModel(a.model, device=a.device, compute_type=a.compute_type,
                         local_files_only=not a.allow_download)
    kw = dict(language=a.language or None, word_timestamps=True, vad_filter=True,
              vad_parameters={"min_silence_duration_ms": 400}, beam_size=a.beam_size,
              condition_on_previous_text=False)
    if a.hotwords:
        kw["hotwords"] = a.hotwords
    try:
        segments, info = model.transcribe(a.wav, **kw)
    except TypeError:  # older faster-whisper without hotwords
        kw.pop("hotwords", None)
        kw["initial_prompt"] = a.hotwords or None
        segments, info = model.transcribe(a.wav, **kw)
    words, segs = [], []
    for s in segments:
        segs.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
        for w in s.words or []:
            words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end),
                          "prob": float(w.probability)})
    out = {"engine": "faster-whisper", "model": a.model, "device": a.device, "compute_type": a.compute_type,
           "language": getattr(info, "language", a.language), "duration": float(getattr(info, "duration", 0.0)),
           "words": words, "segments": segs}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
