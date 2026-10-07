"""Local speech-to-text for narrations: engine selection, audio conversion, transcript format. Nothing leaves the machine.

Engines (MOVES_RESEARCH A1-A4), tried in this order by engine='auto':
  faster-whisper  primary: large-v3-turbo, word timestamps, VAD, move vocabulary as hotwords; runs asr_worker.py in
                  a subprocess with the first interpreter that has faster_whisper: $DANCECAP_ASR_PYTHON,
                  <workspace>/.venv-asr, then this interpreter. GPU (float16) under dancecap.gpu.exclusive('asr')
                  with a short wait (gpu_wait, default 15 s); GPU busy or failing -> CPU (int8, same model, or
                  $DANCECAP_ASR_CPU_MODEL). Model files are never downloaded unless DANCECAP_ASR_ALLOW_DOWNLOAD=1.
                  NOT INSTALLED on this workstation yet (needs the user's OK for ~0.55 GB CUDA wheels + 1.6 GB model):
                  `python -m dancecap.moves annotate asr-check` prints the setup commands.
  sapi            fallback, built into Windows: System.Speech desktop recogniser (en-US), offline. Two passes
                  (sapi_transcribe.ps1): a grammar of the move vocabulary between wildcards (recognised + rejected
                  best guesses) and free dictation. Merge: a recognised grammar result (confidence >= 0.5) wins its
                  time span; elsewhere the dictation words are used and overlapping grammar guesses become
                  `alternatives` (tried by narration.map_utterances when the dictation words name no move).
                  Weak for real Portuguese pronunciations - expect more 'unknown' phrases than with whisper.
  none            no recogniser: typed labels, 'M' taps, and transcript import still work.
Transcript import: any engine's output as JSON [{word, start, end, prob}] or {words: [...]} (seconds of the
narration audio).

Transcript (narration/<id>/transcript.json): {version:1, engine, model, device, language, duration,
  words:[{word, start, end, prob, source}], utterances:[{start, end, text, words:[...], alternatives:[{text, prob,
  start, end, words}]}], warnings:[...], created}
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

from ...take import WORKSPACE
from ..narration import CONTROL, Lexicon, normalise

HERE = Path(__file__).resolve().parent
WORKER = HERE / "asr_worker.py"
SAPI_SCRIPT = HERE / "sapi_transcribe.ps1"
SR = 16000
UTTERANCE_GAP = 0.6
SAPI_ACCEPT = 0.5
DEFAULT_MODEL = "large-v3-turbo"
CPU_MODEL = os.environ.get("DANCECAP_ASR_CPU_MODEL")  # e.g. small.en; default: the same model in int8
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW

_engine_cache: dict | None = None


class ASRError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------------------------------------------

def ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def wav_info(path: Path) -> tuple[int, int, int, float] | None:
    """(rate, channels, sample width, seconds) of a PCM wav; None when it is not one."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError):
        return None


def to_wav16k(src: Path, dst: Path) -> Path:
    """16 kHz mono 16-bit PCM wav (copied when it already is one; else decoded with ffmpeg: webm/ogg/mp4/...)."""
    src, dst = Path(src), Path(dst)
    info = wav_info(src)
    if info and info[:3] == (SR, 1, 2):
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return dst
    ff = ffmpeg()
    if not ff:
        raise ASRError(f"ffmpeg not found: cannot convert {src.name} to 16 kHz wav")
    r = subprocess.run([ff, "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-sample_fmt",
                        "s16", str(dst)], capture_output=True, text=True, creationflags=NO_WINDOW)
    if r.returncode != 0 or not dst.exists():
        raise ASRError(f"ffmpeg failed on {src.name}: {r.stderr.strip()[:300]}")
    return dst


# ---------------------------------------------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------------------------------------------

def respell(term: str) -> str | None:
    """An English-recogniser spelling of a Portuguese move name ('viradinha' -> 'virajinya'), None if unchanged."""
    t = normalise(term)
    r = t
    for a, b in (("nh", "ny"), ("lh", "ly"), ("di", "ji"), ("ao", "ow"), ("qu", "k")):
        r = r.replace(a, b)
    return r if r != t else None


def grammar_phrases(lex: Lexicon) -> tuple[list[str], dict[str, str]]:
    """(phrases for a recogniser grammar, respelling -> original term)."""
    phrases, back = [], {}
    for t in lex.terms:
        if t.kind == "number":
            continue
        phrases.append(t.text)
        r = respell(t.text)
        if r and r not in lex.exact:
            phrases.append(r)
            back[r] = t.text
    phrases += list(CONTROL)
    return list(dict.fromkeys(p for p in phrases if p)), back


def hotwords(lex: Lexicon, max_chars: int = 600) -> str:
    """Display names first (Portuguese where established), then aliases with non-English spelling; ~223 tokens."""
    names = []
    for m in lex.moves.values():
        n = m.get("name") or m.get("display_name")
        if n:
            names.append(n)
    extra = [a for m in lex.moves.values() for a in (m.get("aliases") or []) if normalise(a) != a.lower()]
    out, n = [], 0
    for w in dict.fromkeys(names + extra):
        if n + len(w) + 1 > max_chars:
            break
        out.append(w)
        n += len(w) + 1
    return " ".join(out)


# ---------------------------------------------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------------------------------------------

def _candidate_pythons() -> list[str]:
    c = []
    if os.environ.get("DANCECAP_ASR_PYTHON"):
        c.append(os.environ["DANCECAP_ASR_PYTHON"])
    for v in (WORKSPACE / ".venv-asr" / "Scripts" / "python.exe", WORKSPACE / ".venv-asr" / "bin" / "python"):
        if v.exists():
            c.append(str(v))
    c.append(sys.executable)
    return list(dict.fromkeys(c))


def _check_whisper(py: str, model: str) -> dict | None:
    try:
        r = subprocess.run([py, str(WORKER), "--check", "--model", model], capture_output=True, text=True,
                           timeout=120, creationflags=NO_WINDOW)
        info = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None
    return info if info.get("faster_whisper") else None


def _check_sapi() -> dict | None:
    ps = shutil.which("powershell")
    if os.name != "nt" or not ps:
        return None
    with tempfile.TemporaryDirectory() as d:
        w = Path(d) / "x.wav"
        with wave.open(str(w), "wb") as f:
            f.setnchannels(1), f.setsampwidth(2), f.setframerate(SR)
            f.writeframes(b"\0\0" * 1600)
        try:
            r = subprocess.run([ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SAPI_SCRIPT), "-Wav",
                                str(w), "-Check"], capture_output=True, text=True, timeout=60,
                               creationflags=NO_WINDOW, encoding="utf-8", errors="replace")
            return json.loads(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None


def engines(refresh: bool = False, model: str = DEFAULT_MODEL) -> dict:
    """{faster-whisper: {available, python, model_cached, cuda_devices, ...}, sapi: {available, recognizer}}."""
    global _engine_cache
    if _engine_cache is not None and not refresh:
        return _engine_cache
    out = {"faster-whisper": {"available": False, "model": model}, "sapi": {"available": False}}
    for py in _candidate_pythons():
        info = _check_whisper(py, model)
        if info:
            out["faster-whisper"] = {"available": bool(info.get("model_cached")) or
                                     os.environ.get("DANCECAP_ASR_ALLOW_DOWNLOAD") == "1",
                                     "python": py, "model": model, **info}
            break
    s = _check_sapi()
    if s:
        out["sapi"] = {"available": True, **s}
    _engine_cache = out
    return out


def setup_help() -> str:
    return "\n".join([
        "faster-whisper is not installed (it needs the user's OK: ~0.55 GB CUDA wheel + ~1.6 GB model).",
        "To install it into its own environment (MOVES_RESEARCH A2; not into .venv-nopo4d), from "
        "<workspace>:",
        "  py -3.13 -m venv .venv-asr",
        '  .venv-asr/Scripts/python -m pip install "faster-whisper>=1.1" nvidia-cublas-cu12',
        '  .venv-asr/Scripts/python -c "from faster_whisper.utils import download_model as d; '
        "print(d('large-v3-turbo'))\"   (one-time model download)",
        "then `python -m dancecap.moves annotate asr-check` should report faster-whisper available "
        "(cuda_devices > 0); it runs offline from then on.",
        "Without it the tool uses the Windows recogniser (sapi), typed labels, or an imported transcript."])


# ---------------------------------------------------------------------------------------------------------------
# transcription
# ---------------------------------------------------------------------------------------------------------------

def utterances_from_words(words: list[dict], gap: float = UTTERANCE_GAP, max_len: float = 10.0) -> list[dict]:
    """Group words into utterances at pauses > gap seconds or sentence punctuation."""
    out: list[dict] = []
    cur: list[dict] = []
    for w in sorted(words, key=lambda w: (w["start"], w["end"])):
        if cur and (w["start"] - cur[-1]["end"] > gap or w["start"] - cur[0]["start"] > max_len
                    or cur[-1]["word"].rstrip().endswith((".", "?", "!"))):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return [{"start": u[0]["start"], "end": u[-1]["end"], "text": " ".join(w["word"] for w in u).strip(),
             "words": u} for u in out]


def _clean_words(words: list[dict], source: str | None = None) -> list[dict]:
    out = []
    for w in words:
        try:
            s, e = float(w["start"]), float(w.get("end", w["start"]))
        except (KeyError, TypeError, ValueError):
            continue
        text = str(w.get("word", w.get("text", ""))).strip()
        if not text or s < 0:
            continue
        p = w.get("prob", w.get("probability", w.get("confidence")))
        d = {"word": text, "start": round(s, 3), "end": round(max(s, e), 3),
             "prob": None if p is None else round(float(p), 3)}
        if source or w.get("source"):
            d["source"] = source or w.get("source")
        out.append(d)
    return out


def from_words(words: list[dict], engine: str, model: str | None = None, **extra) -> dict:
    ws = _clean_words(words)
    return {"version": 1, "engine": engine, "model": model, "words": ws, "utterances": utterances_from_words(ws),
            "warnings": [], "created": now_iso(), **extra}


def load_transcript_json(obj, engine: str = "import") -> dict:
    """A transcript from any engine: [{word, start, end, prob}] or {words: [...]} (+ optional engine/model)."""
    if isinstance(obj, (str, Path)):
        obj = json.loads(Path(obj).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        words = obj.get("words") or [w for s in obj.get("segments") or [] for w in s.get("words") or []]
        return from_words(words, obj.get("engine") or engine, obj.get("model"))
    if isinstance(obj, list):
        return from_words(obj, engine)
    raise ASRError("transcript JSON must be a list of {word, start, end, prob} or {words: [...]}")


def _sapi_abs(results: list[dict], back: dict[str, str]) -> list[dict]:
    out = []
    for r in results or []:
        st = float(r.get("start") or 0.0)
        words = []
        for w in r.get("words") or []:
            if w.get("word") in (None, "", "...") or float(w.get("start", -1)) < 0:
                continue
            text = back.get(normalise(w["word"]), w["word"])
            words.append({"word": text, "heard": w["word"] if text != w["word"] else None,
                          "start": round(st + float(w["start"]), 3), "end": round(st + float(w["end"]), 3),
                          "prob": round(float(w.get("prob") or 0.0), 3)})
        for w in words:
            if w["heard"] is None:
                w.pop("heard")
        if words:
            out.append({"kind": r.get("kind", "ok"), "text": " ".join(w["word"] for w in words),
                        "confidence": float(r.get("confidence") or 0.0), "start": words[0]["start"],
                        "end": words[-1]["end"], "words": words})
    return out


def merge_sapi(grammar: list[dict], dictation: list[dict], gap: float = UTTERANCE_GAP) -> list[dict]:
    """Utterances from the two SAPI passes (absolute times): accepted grammar results win their span, dictation
    elsewhere, other grammar guesses become alternatives."""
    def overlap(a, b, pad=0.15):
        return a["start"] - pad < b["end"] and b["start"] - pad < a["end"]

    acc = [g for g in grammar if g["kind"] == "ok" and g["confidence"] >= SAPI_ACCEPT]
    weak = [g for g in grammar if g not in acc]
    utts = []
    for g in acc:
        for w in g["words"]:
            w["source"] = "grammar"
        utts.append({"start": g["start"], "end": g["end"], "text": g["text"], "words": g["words"],
                     "alternatives": []})
    for d in dictation:
        if any(overlap(d, g) for g in acc):
            continue
        for w in d["words"]:
            w["source"] = "dictation"
        alts = [{"text": g["text"], "prob": round(g["confidence"], 3), "start": g["start"], "end": g["end"],
                 "kind": g["kind"], "words": g["words"]} for g in weak if overlap(d, g)]
        utts.append({"start": d["start"], "end": d["end"], "text": d["text"], "words": d["words"],
                     "alternatives": alts})
    for g in weak:  # a guess where dictation heard nothing
        if any(overlap(g, u) for u in utts):
            continue
        utts.append({"start": g["start"], "end": g["end"], "text": "", "words": [],
                     "alternatives": [{"text": g["text"], "prob": round(g["confidence"], 3), "start": g["start"],
                                       "end": g["end"], "kind": g["kind"], "words": g["words"]}]})
    return sorted(utts, key=lambda u: u["start"])


def transcribe_sapi(wav: Path, lex: Lexicon | None, timeout: float = 600) -> dict:
    ps = shutil.which("powershell")
    if not ps:
        raise ASRError("powershell not found")
    phrases, back = grammar_phrases(lex) if lex is not None else ([], {})
    with tempfile.TemporaryDirectory() as d:
        pf = Path(d) / "phrases.txt"
        pf.write_text("\n".join(phrases) + "\n", encoding="utf-8")
        t0 = time.time()
        r = subprocess.run([ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SAPI_SCRIPT), "-Wav",
                            str(wav), "-PhraseFile", str(pf)], capture_output=True, text=True, timeout=timeout,
                           creationflags=NO_WINDOW, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise ASRError(f"sapi failed: {(r.stderr or r.stdout).strip()[:400]}")
    raw = json.loads(r.stdout.strip())
    g, dct = _sapi_abs(raw.get("grammar"), back), _sapi_abs(raw.get("dictation"), back)
    utts = merge_sapi(g, dct)
    words = [w for u in utts for w in u["words"]]
    info = wav_info(wav)
    return {"version": 1, "engine": "sapi", "model": raw.get("recognizer"), "device": "cpu", "language": "en-US",
            "duration": info[3] if info else None, "words": words, "utterances": utts,
            "passes": {"grammar": g, "dictation": dct}, "seconds": round(time.time() - t0, 2),
            "warnings": [], "created": now_iso()}


def transcribe_whisper(wav: Path, lex: Lexicon | None, info: dict, gpu_wait: float = 15.0,
                       language: str = "en", log=print) -> dict:
    from ...gpu import exclusive
    py = info["python"]
    model = info.get("model") or DEFAULT_MODEL
    hw = hotwords(lex) if lex is not None else ""
    allow = os.environ.get("DANCECAP_ASR_ALLOW_DOWNLOAD") == "1"
    warnings = []

    def run(device: str, compute: str, mdl: str) -> dict:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.json"
            cmd = [py, str(WORKER), "--wav", str(wav), "--out", str(out), "--model", mdl, "--device", device,
                   "--compute-type", compute, "--language", language, "--hotwords", hw]
            if allow:
                cmd.append("--allow-download")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, creationflags=NO_WINDOW,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0 or not out.exists():
                raise ASRError(f"faster-whisper ({device}) failed: {(r.stderr or r.stdout).strip()[-400:]}")
            return json.loads(out.read_text(encoding="utf-8"))

    res = None
    if int(info.get("cuda_devices") or 0) > 0:
        try:
            with exclusive("asr", timeout_s=gpu_wait, poll_s=1.0):
                res = run("cuda", "float16", model)
        except TimeoutError:
            warnings.append(f"GPU busy (work/gpu.lock) for {gpu_wait:.0f} s - transcribed on the CPU")
        except ASRError as e:
            warnings.append(f"GPU transcription failed, CPU fallback: {e}")
    if res is None:
        log("[annotate] faster-whisper on CPU (int8)")
        res = run("cpu", "int8", CPU_MODEL or model)
    t = from_words(res.get("words") or [], "faster-whisper", res.get("model"), device=res.get("device"),
                   language=res.get("language"), duration=res.get("duration"), segments=res.get("segments"))
    t["warnings"] = warnings
    return t


def transcribe(wav: Path, engine: str = "auto", lex: Lexicon | None = None, gpu_wait: float = 15.0,
               log=print) -> dict:
    """Transcript dict (see module doc) of a 16 kHz mono wav. engine: auto | faster-whisper | sapi."""
    av = engines()
    order = ["faster-whisper", "sapi"] if engine == "auto" else [engine]
    errors = []
    for name in order:
        info = av.get(name) or {}
        if not info.get("available"):
            errors.append(f"{name}: not available")
            continue
        try:
            if name == "faster-whisper":
                return transcribe_whisper(Path(wav), lex, info, gpu_wait, log=log)
            if name == "sapi":
                return transcribe_sapi(Path(wav), lex)
        except (ASRError, OSError, subprocess.TimeoutExpired, ValueError) as e:
            errors.append(f"{name}: {e}")
            log(f"[annotate] {name} failed: {e}")
    raise ASRError("no speech recogniser worked: " + "; ".join(errors))
