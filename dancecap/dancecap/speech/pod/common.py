"""Shared helpers of the pod-side speech scripts (they also run on the PC for a quick check).

Everything here uses numpy / soundfile; torch, faster-whisper, NeMo ... are imported by the stage that needs them. Layout:

  SPEECH_AUDIO   directory of the 16 kHz mono wavs  `NN-Recap-X.16k.wav`      (pod: $HM_ROOT/src/larissa_kadu_recap/speech_audio)
  SPEECH_RAW     output directory, one folder per source                      (pod: <ws>/dancecap/work/larissa_kadu_recap/speech/raw)
  SPEECH_DIR     the speech work directory (glossary.json lives here)

All times in the raw files are SOURCE seconds (the source's own clock); the PC rebases them with speech.sources.Clock.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
STAGE_VERSION = 1


def log(msg: str) -> None:
    print(f"[speech {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _hm_root() -> Path:
    return Path(os.environ.get("HM_ROOT", "/workspace/hm"))


def _ws() -> Path:
    return Path(os.environ.get("HM_WS", "/opt/hm/ws"))


def speech_dir() -> Path:
    if os.environ.get("SPEECH_DIR"):
        return Path(os.environ["SPEECH_DIR"])
    here = Path(__file__).resolve()
    # .../dancecap/dancecap/speech/pod/common.py -> .../dancecap/work/larissa_kadu_recap/speech
    return here.parents[3] / "work" / "larissa_kadu_recap" / "speech"


def audio_dir() -> Path:
    if os.environ.get("SPEECH_AUDIO"):
        return Path(os.environ["SPEECH_AUDIO"])
    pod = _hm_root() / "src" / "larissa_kadu_recap" / "speech_audio"
    return pod if pod.is_dir() else speech_dir() / "audio"


def raw_dir() -> Path:
    return Path(os.environ["SPEECH_RAW"]) if os.environ.get("SPEECH_RAW") else speech_dir() / "raw"


def cams(only: str | None = None) -> dict[str, Path]:
    """{cam id: 16 kHz wav} found in audio_dir(), optionally limited to a comma list of ids."""
    want = {c.strip() for c in only.split(",") if c.strip()} if only else None
    out = {}
    for p in sorted(audio_dir().glob("[0-9][0-9]-Recap-*.16k.wav")):
        if want is None or p.name[:2] in want:
            out[p.name[:2]] = p
    return out


def cam_dir(cam: str) -> Path:
    d = raw_dir() / cam
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj) -> None:
    """Atomic JSON write (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_audio(path: Path) -> np.ndarray:
    """float32 mono samples in [-1, 1] at 16 kHz."""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path.name}: {sr} Hz, expected {SR}")
    return np.ascontiguousarray(x)


def glossary() -> dict:
    return read_json(speech_dir() / "glossary.json", {}) or {}


def progress_note(msg: str, **kw) -> None:
    try:
        from dancecap import progress as P
        P.note(msg, **kw)
    except Exception:  # noqa: BLE001 - progress is never fatal
        pass


def progress_tick(name: str, done: int, total: int | None = None, unit: str = "cams") -> None:
    try:
        from dancecap import progress as P
        P.tick(name, done, total=total, unit=unit)
    except Exception:  # noqa: BLE001
        pass


def versions(*mods: str) -> dict:
    out = {}
    for m in mods:
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            out[m] = f"unavailable: {type(e).__name__}"
    return out


def gpu_name() -> str | None:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def vram_used_mib() -> int | None:
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


# ------------------------------------------------------------------------------------------------ VAD


def vad_regions(audio: np.ndarray, min_silence_ms: int = 300, pad_ms: int = 120, threshold: float = 0.5) -> list[tuple[int, int]]:
    """Silero VAD speech regions (sample indices) through faster-whisper's bundled ONNX model."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(audio, VadOptions(threshold=threshold, min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms))
    return [(int(r["start"]), int(r["end"])) for r in ts]


def vad_probabilities(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """(per-frame speech probability, frame step in seconds) from the Silero model (512-sample frames at 16 kHz = 32 ms)."""
    from faster_whisper.vad import get_vad_model
    model = get_vad_model()
    x = audio
    pad = (-len(x)) % 512
    if pad:
        x = np.concatenate([x, np.zeros(pad, np.float32)])
    probs = np.asarray(model(x), dtype=np.float32).reshape(-1)
    return probs, 512 / SR


def merge_regions(regions: list[tuple[float, float]], max_gap: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(regions):
        if out and a - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def split_for_asr(regions_s: list[tuple[float, float]], max_len: float = 28.0, min_break_gap: float = 0.25) -> list[tuple[float, float]]:
    """Speech regions (seconds) -> segments of at most `max_len` s for an engine that wants short utterances: neighbouring regions are
    joined while the joined span stays under max_len, a region longer than max_len is cut at its middle (recursively)."""
    segs: list[tuple[float, float]] = []

    def cut(a: float, b: float) -> list[tuple[float, float]]:
        if b - a <= max_len:
            return [(a, b)]
        m = (a + b) / 2
        return cut(a, m) + cut(m, b)

    cur: list[float] | None = None
    for a, b in sorted(regions_s):
        if cur is not None and b - cur[0] <= max_len and a - cur[1] <= 4.0:
            cur[1] = b
            continue
        if cur is not None:
            segs += cut(cur[0], cur[1])
        cur = [a, b]
    if cur is not None:
        segs += cut(cur[0], cur[1])
    return segs


# ------------------------------------------------------------------------------------------------ whisper helpers (from dancecap.moves.annotate.asr_worker)


def normalise_gain(audio: np.ndarray, target_peak: float = 0.708, quiet_peak: float = 0.25, max_gain_db: float = 30.0):
    """(audio, gain dB): a quiet recording (peak below -12 dBFS) is scaled so its peak reaches -3 dBFS (at most +30 dB; never clips)."""
    if audio.size == 0:
        return audio, 0.0
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-6 or peak >= quiet_peak:
        return audio, 0.0
    g = min(target_peak / peak, 10 ** (max_gain_db / 20))
    return (audio * g).astype(np.float32), round(20 * float(np.log10(g)), 2)


def concat_with_gaps(audio: np.ndarray, regions: list[tuple[int, int]], gap_s: float):
    """Speech regions back to back with `gap_s` of silence between them -> (audio, table [(c0, c1, o0)]): concatenated samples c0..c1
    come from original sample o0. (faster-whisper's own vad_filter joins regions WITHOUT a gap and its word alignment then moves
    words across the joins: measured up to 7 s on the narration recordings.)"""
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
    x = t * SR
    k = 0
    for i, (c0, c1, _) in enumerate(table):
        if x >= c0 - eps * SR:
            k = i
    return k


def remap_pass(words: list[dict], segs: list[dict], table) -> None:
    """In place: concatenated-audio seconds -> recording seconds. A word belongs to the region where it ENDS (Whisper's word starts
    absorb the silence before a word); its start is clipped to that region."""
    def conv(t: float, k: int) -> float:
        c0, c1, o0 = table[k]
        x = min(max(t * SR, c0), c1)
        return (o0 + x - c0) / SR

    for w in words:
        k = region_of(max(w["start"], w["end"] - 0.05), table)
        w["region"] = k
        w["start"], w["end"] = round(conv(w["start"], k), 3), round(conv(w["end"], k), 3)
        if w["end"] < w["start"]:
            w["start"] = w["end"]
    for sg in segs:
        s0, s1 = sg["start"], sg["end"]
        sg["start"] = round(conv(s0, region_of(s0 + 0.05, table)), 3)
        sg["end"] = round(conv(s1, region_of(max(s0, s1 - 0.05), table)), 3)


def trim_terms(model, text: str, max_tokens: int = 200) -> str:
    """Whole comma-separated terms that fit Whisper's prompt budget."""
    terms = [t.strip() for t in text.split(",") if t.strip()]
    try:
        enc = model.hf_tokenizer.encode
        n = lambda s: len(enc(" " + s, add_special_tokens=False).ids)  # noqa: E731
    except Exception:  # noqa: BLE001
        n = lambda s: len(s) // 3 + 1  # noqa: E731
    out: list[str] = []
    for t in terms:
        if n(", ".join(out + [t])) > max_tokens:
            break
        out.append(t)
    return ", ".join(out)


# ------------------------------------------------------------------------------------------------ small signal helpers


def rms_db(x: np.ndarray, floor: float = 1e-6) -> float:
    return float(20 * np.log10(max(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0, floor)))


def frame_rms_db(x: np.ndarray, frame: int = 320, hop: int = 160) -> np.ndarray:
    """RMS in dBFS per frame (20 ms / 10 ms at 16 kHz)."""
    if x.size < frame:
        return np.array([rms_db(x)], dtype=np.float32)
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx].astype(np.float64)
    r = np.sqrt(np.mean(fr * fr, axis=1))
    return (20 * np.log10(np.maximum(r, 1e-6))).astype(np.float32)


def f0_track(x: np.ndarray, sr: int = SR, fmin: float = 70.0, fmax: float = 400.0, frame_s: float = 0.04, hop_s: float = 0.01,
             min_clarity: float = 0.5, min_db: float = -50.0) -> tuple[np.ndarray, np.ndarray]:
    """(f0 Hz or 0 where unvoiced, clarity 0..1) per 10 ms hop: normalised autocorrelation through the FFT. A frame is voiced when
    its clarity is >= min_clarity and its level is above min_db; the lag picked is the smallest one whose peak is >= 0.9 of the
    best (guards against octave-down errors). Crude on purpose: it feeds a gender-level prior (about 110 vs 200 Hz), not a pitch
    contour."""
    frame, hop = int(frame_s * sr), int(hop_s * sr)
    if x.size < frame + hop:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx].astype(np.float32)
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr *= np.hanning(frame).astype(np.float32)[None, :]
    nfft = 1 << int(np.ceil(np.log2(2 * frame)))
    spec = np.fft.rfft(fr, nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), nfft, axis=1)[:, :frame]
    ac0 = ac[:, :1] + 1e-9
    nc = ac / ac0
    lo, hi = int(sr / fmax), min(int(sr / fmin), frame - 1)
    seg = nc[:, lo:hi + 1]
    best = seg.max(axis=1)
    thresh = 0.9 * best
    first = np.argmax(seg >= thresh[:, None], axis=1)
    lag = lo + first
    # parabolic refinement
    lag_f = lag.astype(np.float64)
    ok = (lag > lo) & (lag < hi)
    ii = np.where(ok)[0]
    a, b, c = nc[ii, lag[ii] - 1], nc[ii, lag[ii]], nc[ii, lag[ii] + 1]
    den = a - 2 * b + c
    shift = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    lag_f[ii] = lag[ii] + np.clip(shift, -1, 1)
    f0 = sr / lag_f
    level = 20 * np.log10(np.maximum(np.sqrt(np.mean(np.square(x[idx].astype(np.float64)), axis=1)), 1e-6))
    voiced = (best >= min_clarity) & (level >= min_db) & (f0 > fmin * 1.03) & (f0 < fmax * 0.985)   # a pick at the search edge is no pitch
    return np.where(voiced, f0, 0.0).astype(np.float32), best.astype(np.float32)


def parse_cams_arg(s: str | None) -> str | None:
    return s if s and re.fullmatch(r"[0-9, ]+", s) else None
