"""Local speech-to-text for narrations: engine selection, audio conversion, transcript format. Nothing leaves the machine.

Engines (MOVES_RESEARCH A1-A4):
  faster-whisper  THE recogniser (engine 'auto'): Whisper large-v3 (multilingual; 'faster-whisper:large-v3-turbo' is
                  the faster option, ~2.5x quicker but clearly weaker on Portuguese move names: 15-17 of 24
                  synthesised Brazilian names right vs 21-24 for large-v3 - Sahada, Shikachi, Solchinho, Torcel),
                  run by asr_worker.py in
                  <workspace>/.venv-asr (Python 3.11: faster-whisper 1.2.1, ctranslate2 4.8.2, nvidia-cublas-cu12;
                  $DANCECAP_ASR_PYTHON overrides). Models in the Hugging Face cache (Systran/faster-whisper-large-v3,
                  mobiuslabsgmbh/faster-whisper-large-v3-turbo; MIT), loaded offline.
                  Decoding: Silero VAD speech regions joined with 1 s silences (words stay in their own region), the
                  move vocabulary as hotwords (Portuguese names first, then distinctive English words; trimmed to
                  Whisper's 223-token prompt), word timestamps, beam 5, two passes - language en and language pt -
                  fused per utterance (fuse_passes: the hypothesis whose words name moves with the higher score x word
                  probability wins, +0.02 prior for English; the other one becomes an `alternative`).
                  GPU: float16 under dancecap.gpu.exclusive('asr') (wait gpu_wait, 15 s); when a pipeline stage holds
                  the lock and nvidia-smi shows enough free VRAM (model + 1.5 GB margin, the lowest of 3 readings) the
                  short job runs beside it ('guest'; DANCECAP_ASR_GPU_GUEST=0 turns that off); otherwise the
                  narration is QUEUED for the GPU (re-checked every 20 s, up to DANCECAP_ASR_GPU_QUEUE_H = 12 h; log
                  lines '[annotate] GPU queue: ...'). skip_gpu_queue(wav) (the UI's 'Transcribe on CPU now') or
                  DANCECAP_ASR_GPU_BUSY=cpu transcribe at once on the CPU instead: int8 large-v3-turbo, English pass
                  only (~1.5x the speech time), weaker on Portuguese names - such a transcript carries
                  retranscribe_reason() and the UI marks it 're-transcribe recommended'. The worker is a subprocess:
                  its exit returns all GPU memory.
  sapi            Windows System.Speech (en-US only, grammar + dictation; sapi_transcribe.ps1). Only when chosen
                  explicitly (engine 'sapi'): 'auto' NEVER falls back to it - it turned real narrations into
                  "one role" / "at a" / "leaner". When Whisper is unavailable, transcribe() raises ASRError with the
                  reason and the fix (setup_help), and the UI shows it.
  none            no recogniser: typed labels, 'M' taps, and transcript import still work.
Transcript import: any engine's output as JSON [{word, start, end, prob}] or {words: [...]} (seconds of the
narration audio).

Transcript (narration/<id>/transcript.json):
  Whisper (version 2): {version, engine, model, device, compute_type, gpu (lock | guest | cpu), language ('en+pt'),
    languages, duration, gain_db, words:[{word, start, end, prob, lang}], utterances:[{start, end, text, words, lang,
    scores:{en, pt}, alternatives:[{text, prob, start, end, words, kind: whisper-<lang>, conf_cap}]}],
    raw: {passes:[{language, text, words, segments}], regions, hotwords} (the unfused recogniser output, kept),
    fusion, timing, vram_mib, seconds, warnings, created}
  SAPI / import (version 1): {version, engine, model, device, language, duration, words:[{word, start, end, prob,
    source}], utterances:[{start, end, text, words, alternatives}], warnings, created}
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

from ...take import WORKSPACE
from ..narration import (CONTROL, FILLERS, Lexicon, breaks_of, decide, looks_portuguese, match_tokens, normalise,
                         tokens_of)

HERE = Path(__file__).resolve().parent
WORKER = HERE / "asr_worker.py"
SAPI_SCRIPT = HERE / "sapi_transcribe.ps1"
SR = 16000
UTTERANCE_GAP = 0.6
SAPI_ACCEPT = 0.5
DEFAULT_MODEL = os.environ.get("DANCECAP_ASR_MODEL", "large-v3")
FAST_MODEL = "large-v3-turbo"
BEST_MODEL = "large-v3"  # the model a transcript is judged against (retranscribe_reason)
MODELS = {  # VRAM measured on the 2080 Ti (fp16, beam 5): large-v3 +3.9 GB, turbo +2.2 GB over the idle desktop
    "large-v3": {"repo": "Systran/faster-whisper-large-v3", "vram_mib": 4100, "label": "Whisper large-v3"},
    "large-v3-turbo": {"repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "vram_mib": 2300,
                       "label": "Whisper large-v3-turbo (faster; weaker on Portuguese names)"},
}
PASSES = ("en", "pt")
CPU_MODEL = os.environ.get("DANCECAP_ASR_CPU_MODEL", FAST_MODEL)  # CPU int8: turbo decodes ~2x faster than large-v3
CPU_PASSES = tuple(x for x in os.environ.get("DANCECAP_ASR_CPU_PASSES", "en").split(",") if x)
GUEST_MARGIN_MIB = 1500
QUEUE_POLL_S = 20.0      # GPU queue: how long each wait for the lock lasts before guest / skip are checked again
QUEUE_LOG_S = 120.0      # ... and how often the job log says it is still waiting
_skip_queue: set[str] = set()  # wav paths whose queued job should go to the CPU now (skip_gpu_queue)
EN_PRIOR = 0.02          # fusion: the narrator speaks English
ALT_CONF_CAP = 0.8       # fusion: an alternative from the other language pass is a full Whisper hypothesis
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


_EN_COMMON = set("""left right hand hands turn back front open close closed hold behind walk side step forward basic arm
arms across shoulder shoulders raise body cross hip in out spin double simple reverse lead fall fast inside lower leg one
two up down over overhead push break wrap hug stand split stop tight drag dip dive chair snake wings wave pop launch lean
project reposition replace rotate throw untwist unwind unwrap with from the do little twist switch tilt bent intimate
submissive extend lunge and to airplane bonus forth roll""".split())
_PT_STOP = {"para", "pro", "e", "de", "do", "da", "a", "o", "com", "no", "na", "dos", "das", "os", "as", "em"}


def _is_pt_name(m: dict) -> bool:
    n, ne = m.get("name") or m.get("display_name"), m.get("name_en")
    return bool(n) and (not ne or normalise(n) != normalise(ne) or looks_portuguese(n))


def hotwords(lex: Lexicon) -> str:
    """The move vocabulary for Whisper's prompt, most useful first (the worker keeps whole terms up to its 223-token
    budget): the words of the Portuguese display names (one-word names first: Corredor, Viradinha, Bumerangue,
    Cambré, Ioiô, Sarrada ...; then Giro, Simples, Duplo, Cavalheiro ...), then the uncommon words of the English
    names (Howl, Pirouette, Boomerang, Counterbalance, Springload ...). Aliases are left out on purpose: several
    are learned misspellings (veraginha, cicada) that must not be suggested to the recogniser."""
    one, multi, en = [], [], []
    for m in lex.moves.values():
        n = m.get("name") or m.get("display_name") or ""
        if _is_pt_name(m):
            ws = [w for w in re.split(r"[\s\-]+", n) if w and normalise(w) not in _PT_STOP and not w.isdigit()]
            ws = [w if w[:1].isupper() else w.capitalize() for w in ws]
            if len(ws) == 1:
                one += ws
            else:
                multi += [w for w in ws if normalise(w) not in _EN_COMMON]
        for w in re.split(r"[\s\-]+", m.get("name_en") or ("" if _is_pt_name(m) else n)):
            if w and not w.isdigit() and normalise(w) not in _EN_COMMON and normalise(w) not in FILLERS:
                en.append(w.capitalize())
    seen, out = set(), []
    for w in one + multi + en:
        k = normalise(w)
        if k and k not in seen:
            seen.add(k)
            out.append(w)
    return ", ".join(out)


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


def engines(refresh: bool = False, model: str | None = None) -> dict:
    """{faster-whisper: {available, python, model, models_cached, cuda_devices, reason?, fix?}, sapi: {available,
    english_only}, auto: <what engine 'auto' runs, or null>, summary: one line for the UI}."""
    global _engine_cache
    model = model or DEFAULT_MODEL
    if _engine_cache is not None and not refresh and _engine_cache.get("faster-whisper", {}).get("model") == model:
        return _engine_cache
    fw: dict = {"available": False, "model": model}
    for py in _candidate_pythons():
        info = _check_whisper(py, model)
        if info:
            fw = {"python": py, "model": model, **info}
            fw["available"] = bool(info.get("model_cached")) or os.environ.get("DANCECAP_ASR_ALLOW_DOWNLOAD") == "1"
            if not fw["available"]:
                fw["reason"] = f"faster-whisper is installed ({py}) but the {model} model is not downloaded"
            break
    else:
        fw["reason"] = "faster-whisper is not installed (no .venv-asr with faster_whisper found)"
    if not fw["available"]:
        fw["fix"] = setup_help(model)
    out = {"faster-whisper": fw, "sapi": {"available": False, "english_only": True}}
    s = _check_sapi()
    if s:
        out["sapi"] = {"available": True, "english_only": True, **s}
    out["auto"] = "faster-whisper" if fw["available"] else None
    out["summary"] = engine_summary(out)
    _engine_cache = out
    return out


def engine_summary(av: dict) -> str:
    fw = av.get("faster-whisper") or {}
    if fw.get("available"):
        dev = "GPU" if int(fw.get("cuda_devices") or 0) > 0 else "CPU only"
        cached = ", ".join(fw.get("models_cached") or [fw.get("model")])
        return (f"speech: Whisper {fw.get('model')} ({dev}; English + Portuguese passes, move vocabulary as hotwords; "
                f"models here: {cached}), local")
    return ("speech: Whisper UNAVAILABLE - " + (fw.get("reason") or "unknown reason") + ". Narrations are not "
            "transcribed until it is fixed (the English-only Windows recogniser is never used automatically). "
            "Fix: python -m dancecap.moves annotate asr-check")


def parse_engine(engine: str | None) -> tuple[str, str | None]:
    """'auto' | 'faster-whisper' | 'faster-whisper:<model>' | 'whisper' | 'sapi' -> (engine, model or None)."""
    e = (engine or "auto").strip()
    name, _, model = e.partition(":")
    if name in ("whisper", "faster_whisper"):
        name = "faster-whisper"
    return name, (model or None)


def resolve(engine: str | None, av: dict) -> tuple[str, str | None]:
    """The engine to run for a setting: ('faster-whisper', model) or ('sapi', None). 'auto' is Whisper or an
    ASRError with the fix - never SAPI. SAPI only when asked for by name."""
    name, model = parse_engine(engine)
    if name in ("auto", "faster-whisper"):
        fw = av.get("faster-whisper") or {}
        if not fw.get("python"):
            raise ASRError("Whisper is unavailable: " + (fw.get("reason") or "faster-whisper not found") + "\n"
                           + (fw.get("fix") or setup_help()))
        want = model or fw.get("model") or DEFAULT_MODEL
        cached = set(fw.get("models_cached") or ([fw.get("model")] if fw.get("model_cached") else []))
        if want not in cached and os.environ.get("DANCECAP_ASR_ALLOW_DOWNLOAD") != "1":
            raise ASRError(f"Whisper model {want} is not downloaded (have: {', '.join(sorted(cached)) or 'none'})\n"
                           + setup_help(want))
        return "faster-whisper", want
    if name == "sapi":
        if not (av.get("sapi") or {}).get("available"):
            raise ASRError("the Windows recogniser (SAPI) is not available")
        return "sapi", None
    raise ASRError(f"unknown speech engine {engine!r} (auto | faster-whisper | faster-whisper:large-v3-turbo | sapi)")


def setup_help(model: str | None = None) -> str:
    model = model or DEFAULT_MODEL
    repo = MODELS.get(model, {}).get("repo", model)
    return "\n".join([
        "Whisper lives in its own environment <workspace>/.venv-asr (MOVES_RESEARCH A2; never "
        "in .venv-nopo4d). To (re)create it, from <workspace>:",
        "  <miniconda>/envs/mamma/python.exe -m venv .venv-asr",
        '  .venv-asr/Scripts/python -m pip install "faster-whisper>=1.2,<2" "nvidia-cublas-cu12==12.*" rapidfuzz',
        f'  .venv-asr/Scripts/python -c "from faster_whisper.utils import download_model as d; print(d(\'{model}\'))"'
        f"   (one-time download of {repo}, MIT)",
        "then `python -m dancecap.moves annotate asr-check` must report faster-whisper available (cuda_devices 1), and "
        "restart the narration tool. It runs offline from then on."])


def gpu_free_mib() -> int | None:
    """Free VRAM of GPU 0 by nvidia-smi (None when unknown)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW)
        return int(r.stdout.strip().splitlines()[0].split(",")[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def gpu_free_min_mib(samples: int = 3, gap_s: float = 1.0) -> int | None:
    """The lowest of a few free-VRAM readings (a lock-holding stage may be loading more onto the GPU)."""
    vals = []
    for i in range(samples):
        if i:
            time.sleep(gap_s)
        v = gpu_free_mib()
        if v is None:
            return None
        vals.append(v)
    return min(vals) if vals else None


def gpu_used_mib() -> int | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW)
        return int(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


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


def run_worker(py: str, wav: Path, model: str, device: str, compute: str, passes, hw: str, log=print,
               timeout: float = 3600) -> dict:
    """One asr_worker.py subprocess (its exit frees all of its GPU memory)."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "t.json"
        cmd = [py, str(WORKER), "--wav", str(wav), "--out", str(out), "--model", model, "--device", device,
               "--compute-type", compute, "--passes", ",".join(passes), "--hotwords", hw]
        if os.environ.get("DANCECAP_ASR_ALLOW_DOWNLOAD") == "1":
            cmd.append("--allow-download")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW,
                           encoding="utf-8", errors="replace")
        for line in (r.stderr or "").splitlines():
            if line.startswith("[asr_worker]"):
                log(line)
        if r.returncode != 0 or not out.exists():
            raise ASRError(f"faster-whisper ({device}) failed: {(r.stderr or r.stdout).strip()[-400:]}")
        return json.loads(out.read_text(encoding="utf-8"))


def _lock_holder() -> str:
    from ...gpu import LOCK
    try:
        return " ".join(LOCK.read_text().split()[1:]) or "another job"
    except OSError:
        return "another job"


def _wav_key(wav) -> str:
    return str(Path(wav).resolve()).lower()


def skip_gpu_queue(wav) -> None:
    """A narration queued for the GPU (transcribe_whisper) goes to the CPU at its next check (<= QUEUE_POLL_S)."""
    _skip_queue.add(_wav_key(wav))


def _queue_policy() -> str:
    return "cpu" if os.environ.get("DANCECAP_ASR_GPU_BUSY", "queue").strip().lower() == "cpu" else "queue"


def transcribe_whisper(wav: Path, lex: Lexicon | None, info: dict, model: str | None = None, gpu_wait: float = 15.0,
                       log=print, passes=PASSES) -> dict:
    """GPU under the lock; when a pipeline stage holds it: beside it if VRAM allows, else QUEUED until the lock is
    free (skip_gpu_queue / DANCECAP_ASR_GPU_BUSY=cpu / the queue limit -> CPU, flagged for re-transcription); then
    fuse_passes."""
    from ... import gpu
    py = info["python"]
    model = model or info.get("model") or DEFAULT_MODEL
    hw = hotwords(lex) if lex is not None else ""
    warnings: list[str] = []
    t0 = time.time()
    res, mode = None, "cpu"
    key = _wav_key(wav)
    _skip_queue.discard(key)  # a click meant for an earlier run of this recording
    if int(info.get("cuda_devices") or 0) > 0:
        need = MODELS.get(model, {}).get("vram_mib", 4100)
        guest_ok = os.environ.get("DANCECAP_ASR_GPU_GUEST", "1") != "0"
        policy = _queue_policy()
        queue_max = float(os.environ.get("DANCECAP_ASR_GPU_QUEUE_H", "12")) * 3600.0
        wait, queued_at, last_log = gpu_wait, None, 0.0
        while res is None:
            try:
                with gpu.exclusive("asr", timeout_s=wait, poll_s=1.0):
                    res, mode = run_worker(py, wav, model, "cuda", "float16", passes, hw, log), "lock"
                if queued_at is not None:
                    warnings.append(f"queued {time.time() - queued_at:.0f} s for the GPU before transcribing")
                break
            except TimeoutError:
                pass
            except ASRError as e:
                warnings.append(f"GPU transcription failed, CPU fallback: {e}")
                break
            holder = _lock_holder()
            free = gpu_free_min_mib() if guest_ok else None
            if guest_ok and free is not None and free >= need + GUEST_MARGIN_MIB:
                log(f"[annotate] GPU lock held by '{holder}'; {free} MiB free >= {need} + {GUEST_MARGIN_MIB}: "
                    f"Whisper runs beside it")
                try:
                    res, mode = run_worker(py, wav, model, "cuda", "float16", passes, hw, log), "guest"
                    warnings.append(f"the GPU lock was held by '{holder}': Whisper ran beside it for "
                                    f"{time.time() - t0:.0f} s ({free} MiB were free, it needs ~{need} MiB)")
                    break
                except ASRError as e:
                    guest_ok = False  # not again beside this job: wait for the lock instead
                    warnings.append(f"GPU beside '{holder}' failed ({e})")
            busy = (f"the GPU is held by '{holder}'" + (f" ({free} MiB free; Whisper {model} needs ~{need} + "
                                                        f"{GUEST_MARGIN_MIB} MiB margin)" if free is not None else ""))
            if policy == "cpu":
                warnings.append(f"{busy}: transcribed on the CPU at once (DANCECAP_ASR_GPU_BUSY=cpu)")
                break
            if key in _skip_queue:
                _skip_queue.discard(key)
                warnings.append(f"{busy}: you chose to transcribe on the CPU now instead of waiting "
                                f"(after {time.time() - (queued_at or t0):.0f} s in the GPU queue)")
                break
            if queued_at is not None and time.time() - queued_at > queue_max:
                warnings.append(f"{busy}: gave up waiting after {queue_max / 3600:.0f} h in the GPU queue; CPU")
                break
            if queued_at is None:
                queued_at = time.time()
            if time.time() - last_log >= QUEUE_LOG_S or last_log == 0.0:
                waited = time.time() - queued_at
                log(f"[annotate] GPU queue: {busy}. Waiting for it ({waited / 60:.0f} min so far); 'Transcribe on "
                    f"CPU now' skips the wait (turbo, English only: weaker on Portuguese names)")
                last_log = time.time()
            wait = QUEUE_POLL_S
    else:
        warnings.append("no CUDA device: transcribed on the CPU")
    if res is None:
        cm = CPU_MODEL if CPU_MODEL in (info.get("models_cached") or [CPU_MODEL]) else model
        log(f"[annotate] faster-whisper {cm} on the CPU (int8, passes {','.join(CPU_PASSES)}): ~1.5x the speech time")
        res, mode = run_worker(py, wav, cm, "cpu", "int8", CPU_PASSES, hw, log), "cpu"
        warnings.append(f"CPU fallback ran {cm} int8 with the {'+'.join(CPU_PASSES)} pass only "
                        f"({time.time() - t0:.0f} s); re-transcribe when the GPU is free for large-v3 en+pt")
    _skip_queue.discard(key)
    return build_transcript(res, lex, mode, warnings, time.time() - t0, gpu_used_mib() if mode != "cpu" else None)


def retranscribe_reason(tr: dict | None) -> str | None:
    """Why a transcript should be made again with the best setup (Whisper large-v3 on the GPU, English + Portuguese
    passes), or None. Imported transcripts are the user's own choice: None."""
    if not tr:
        return None
    eng = tr.get("engine")
    if eng == "sapi":
        return ("transcribed by the English-only Windows recogniser (SAPI), which mishears Portuguese move names - "
                "re-transcribe with auto (Whisper large-v3)")
    if eng != "faster-whisper":
        return None
    why = []
    m = tr.get("model")
    if m and m != BEST_MODEL:
        why.append(f"Whisper {m} (weaker on Portuguese move names than {BEST_MODEL})")
    if tr.get("gpu") == "cpu" or tr.get("device") == "cpu":
        why.append("CPU fallback")
    langs = [x for x in (tr.get("languages") or str(tr.get("language") or "").split("+")) if x]
    if langs and "pt" not in langs:
        why.append("English pass only")
    if not why:
        return None
    return ", ".join(why) + f" - re-transcribe with auto (Whisper {BEST_MODEL}, GPU, English + Portuguese)"


# ---------------------------------------------------------------------------------------------------------------
# fusion of the language passes
# ---------------------------------------------------------------------------------------------------------------

MISPLACED_WINDOW = 8.0   # s: the same move heard by both passes but put in neighbouring speech regions


def _hallucinated(seg: dict) -> bool:
    """Whisper's own no-speech rule (no_speech_prob > 0.6 and avg_logprob < -1)."""
    return float(seg.get("no_speech_prob") or 0.0) > 0.6 and float(seg.get("avg_logprob") or 0.0) < -1.0


def _decoder_repeat(a: dict, b: dict, lex: Lexicon | None) -> bool:
    """Is word b a decoder repetition of word a ("Bônus, Bônus" within half a second), not part of one name?
    Never for a hyphen-split word ("Yo -Yo," is Ioiô / yo-yo), a doubled word that is itself a term ("yo yo",
    "yoyo"), or a word that is not a whole term on its own."""
    k = normalise(a["word"])
    if not k or k != normalise(b["word"]) or b["start"] - a["end"] >= 0.5:
        return False
    if a["word"].rstrip().endswith("-") or b["word"].lstrip().startswith("-"):
        return False
    if lex is not None and (f"{k} {k}" in lex.exact or k + k in lex.exact or k not in lex.exact):
        return False
    return True


def _pass_words(p: dict, lex: Lexicon | None = None) -> list[dict]:
    segs = p.get("segments") or []
    lang = p.get("requested") or p.get("language")
    out = []
    for w in p.get("words") or []:
        k = w.get("seg")
        if isinstance(k, int) and 0 <= k < len(segs) and _hallucinated(segs[k]):
            continue
        try:
            out.append({"word": str(w["word"]).strip(), "start": round(float(w["start"]), 3),
                        "end": round(float(w["end"]), 3),
                        "prob": None if w.get("prob") is None else round(float(w["prob"]), 3), "lang": lang})
        except (KeyError, TypeError, ValueError):
            continue
    ded: list[dict] = []
    for w in (w for w in out if w["word"]):  # "Bônus, Bônus" within half a second: a decoder repetition, not two moves
        if ded and _decoder_repeat(ded[-1], w, lex):
            keep = ded[-1] if (ded[-1]["prob"] or 0) >= (w["prob"] or 0) else w
            ded[-1] = {**keep, "start": ded[-1]["start"], "end": max(ded[-1]["end"], w["end"]), "repeated": True}
            continue
        ded.append(w)
    return ded


def _groups(regions: list[dict], words: list[dict], gap: float = UTTERANCE_GAP) -> list[list[float]]:
    """Utterance spans: the VAD speech regions (or the words' own pauses), merged across pauses < gap."""
    spans = sorted((float(r["start"]), float(r["end"])) for r in regions or [] if "start" in r)
    if not spans:
        spans = [(u["start"], u["end"]) for u in utterances_from_words(words)]
    out: list[list[float]] = []
    for a, b in spans:
        if out and a - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _group_of(w: dict, groups: list[list[float]]) -> int:
    """The group holding the word's end (Whisper's word ends are reliable, starts absorb silence); else the nearest."""
    t = float(w["end"])
    best, bd = 0, None
    for i, (a, b) in enumerate(groups):
        d = 0.0 if a - 0.05 <= t <= b + 0.05 else min(abs(t - a), abs(t - b))
        if bd is None or d < bd:
            best, bd = i, d
    return best


def _hyp_moves(ws: list[dict], lex: Lexicon, prev: str | None) -> dict[str, dict]:
    """{move: {score (0-1 match score x mean word prob, halved when ambiguous), idx: word indices}} - one per move."""
    toks = tokens_of(ws)
    out: dict[str, dict] = {}
    for f in match_tokens(toks, lex, prev, breaks_of(ws, toks)):
        top = f["cands"][0]
        if not top.move:
            continue
        idx = sorted({toks[j][1] for j in range(f["i0"], f["i1"])})
        pr = [float(ws[i]["prob"]) for i in idx if ws[i].get("prob") is not None]
        p = sum(pr) / len(pr) if pr else 0.5
        w = 1.0 if decide(f["cands"])[0] != "ambiguous" else 0.5
        sc = w * top.score / 100.0 * p
        if top.move not in out or sc > out[top.move]["score"]:
            out[top.move] = {"score": sc, "idx": idx, "prob": p}
    return out


def _hyp_score(ws: list[dict], moves: dict[str, dict], lang: str | None) -> float:
    if not ws:
        return -1.0
    pr = [float(w["prob"]) for w in ws if w.get("prob") is not None]
    return sum(m["score"] for m in moves.values()) + 0.25 * (sum(pr) / len(pr) if pr else 0.0) + \
        (EN_PRIOR if lang == "en" else 0.0)


def fuse_passes(passes: list[dict], regions: list[dict], lex: Lexicon | None) -> list[dict]:
    """Per utterance (VAD regions merged across < 0.6 s pauses) the language pass whose words name moves best:
    score = sum over distinct moves of match score x mean word probability (ambiguous matches count half)
    + 0.25 x mean word probability + 0.02 for English. A move that the two passes put in neighbouring utterances
    (the same spoken word, aligned differently) is kept only where its words are more probable. The losing
    hypothesis stays as an alternative (conf_cap 0.8: narration.map_utterances tries it when the winner names no
    move)."""
    plist = [(p.get("requested") or p.get("language") or f"p{i}", _pass_words(p, lex)) for i, p in enumerate(passes)]
    allw = [w for _, ws in plist for w in ws]
    if not allw:
        return []
    groups = _groups(regions, plist[0][1] or allw)
    hyp: dict[tuple[str, int], list[dict]] = {}
    for lang, ws in plist:
        for w in ws:
            hyp.setdefault((lang, _group_of(w, groups)), []).append(w)
    langs = [lang for lang, _ in plist]
    moves: dict[tuple[str, int], dict] = {}
    for key, ws in hyp.items():
        moves[key] = _hyp_moves(ws, lex, None) if lex is not None else {}
    # the same move placed in neighbouring groups by two passes: keep the more probable placement
    for i in range(len(groups)):
        for la in langs:
            for lb in langs:
                if la == lb:
                    continue
                for m, info in list(moves.get((la, i), {}).items()):
                    if m in moves.get((lb, i), {}):
                        continue
                    for j in range(len(groups)):
                        if j == i or abs(groups[j][0] - groups[i][0]) > MISPLACED_WINDOW:
                            continue
                        other = moves.get((lb, j), {}).get(m)
                        if other and m not in moves.get((la, j), {}) and other["prob"] >= info["prob"]:
                            drop = set(info["idx"])
                            hyp[(la, i)] = [w for k, w in enumerate(hyp[(la, i)]) if k not in drop]
                            moves[(la, i)] = _hyp_moves(hyp[(la, i)], lex, None)
                            break
    utts = []
    for i, (g0, g1) in enumerate(groups):
        cands = [(lang, hyp.get((lang, i)) or []) for lang in langs]
        scored = [(_hyp_score(ws, moves.get((lang, i), {}), lang), lang, ws) for lang, ws in cands]
        scored = [x for x in scored if x[2]]
        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])
        best, lang, ws = scored[0]
        ws = sorted(ws, key=lambda w: (w["start"], w["end"]))
        text = " ".join(w["word"] for w in ws).strip()
        alts = []
        for sc, la, aw in scored[1:]:
            aw = sorted(aw, key=lambda w: (w["start"], w["end"]))
            at = " ".join(w["word"] for w in aw).strip()
            if normalise(at) == normalise(text):
                continue
            pr = [float(w["prob"]) for w in aw if w.get("prob") is not None]
            alts.append({"text": at, "prob": round(sum(pr) / len(pr), 3) if pr else None,
                         "start": aw[0]["start"], "end": aw[-1]["end"], "words": aw, "kind": f"whisper-{la}",
                         "conf_cap": ALT_CONF_CAP})
        utts.append({"start": ws[0]["start"], "end": ws[-1]["end"], "text": text, "words": ws, "lang": lang,
                     "scores": {la: round(sc, 3) for sc, la, _ in scored}, "alternatives": alts})
    return utts


def build_transcript(res: dict, lex: Lexicon | None, mode: str, warnings: list[str], wall: float,
                     vram_after: int | None) -> dict:
    """Worker output -> transcript.json (version 2; see the module doc). The raw passes are kept under 'raw'."""
    passes = res.get("passes") or [{"language": res.get("language"), "requested": res.get("language"),
                                     "words": res.get("words") or [], "segments": res.get("segments") or []}]
    utts = fuse_passes(passes, res.get("regions") or [], lex)
    langs = [p.get("requested") or p.get("language") for p in passes]
    chosen: dict[str, int] = {}
    for u in utts:
        chosen[u["lang"]] = chosen.get(u["lang"], 0) + 1
    warnings = list(warnings)
    if not res.get("regions") and not any(p.get("words") for p in passes):
        warnings.append("no speech detected in the recording (voice activity detector): nothing to transcribe - "
                        "check the microphone level")
    return {"version": 2, "engine": "faster-whisper", "model": res.get("model"), "device": res.get("device"),
            "compute_type": res.get("compute_type"), "gpu": mode, "language": "+".join(x for x in langs if x),
            "languages": langs, "duration": res.get("duration"), "gain_db": res.get("gain_db"),
            "words": [w for u in utts for w in u["words"]], "utterances": utts,
            "raw": {"passes": [{"language": p.get("requested") or p.get("language"), "detected": p.get("language"),
                                "text": " ".join(w["word"] for w in p.get("words") or []),
                                "words": p.get("words") or [], "segments": p.get("segments") or [],
                                "seconds": p.get("seconds")} for p in passes],
                    "regions": res.get("regions") or [], "hotwords": res.get("hotwords"),
                    "segmentation": res.get("segmentation"), "beam_size": res.get("beam_size")},
            "fusion": {"method": "per utterance: distinct moves x match score x word probability (+0.02 en); "
                                 "the other pass kept as an alternative", "chosen": chosen},
            "timing": {"load_s": res.get("load_s"), "passes_s": {p.get("requested") or p.get("language"):
                                                                 p.get("seconds") for p in passes},
                       "worker_s": res.get("seconds"), "wall_s": round(wall, 2)},
            "vram_mib": {**(res.get("vram_used_mib") or {}), "after_exit": vram_after},
            "seconds": round(wall, 2), "warnings": warnings, "created": now_iso()}


def transcribe(wav: Path, engine: str = "auto", lex: Lexicon | None = None, gpu_wait: float = 15.0,
               log=print) -> dict:
    """Transcript dict (see module doc) of a 16 kHz mono wav. engine: auto | faster-whisper[:model] | sapi.
    'auto' runs Whisper or raises ASRError with the fix: it never falls back to SAPI."""
    av = engines()
    name, model = resolve(engine, av)
    try:
        if name == "faster-whisper":
            return transcribe_whisper(Path(wav), lex, av["faster-whisper"], model, gpu_wait, log=log)
        return transcribe_sapi(Path(wav), lex)
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        raise ASRError(f"{name}: {e}") from e
