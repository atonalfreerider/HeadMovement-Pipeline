"""Voice-to-voice class audio (agent V1): the teachers' recorded speech, converted turn by turn into their own clean cloned voices.

The recap class (take larissa_kadu_recap) was recorded by seven phones in a noisy room; the speech is hard to understand. With the
OWNER'S explicit permission (2026-10-10, his own ElevenLabs key, the voices "Larissa" and "Kadu" of his account) the teachers' speech
is sent to ElevenLabs speech-to-speech (the "voice changer"): the same words, timing and prosody come back in the clean voice of that
person. Nothing else leaves the machine (no student audio exists in the class; no other service). The key is read at run time from the
owner's Desktop file and is never printed, logged or written (`ElevenKey`); the request ledger holds credits and request ids only.

Stages (each idempotent, files under work/<take>/speech/v2v/):

    plan      pieces = whole stretches of one teacher from transcript_final.json (words on the master clock), cut at pauses, <= 40 s;
              the opening (owner's jokes, words pending) and every stretch where the two voices overlap stay ORIGINAL (logged)
    sources   per piece the cleanest phone (speech-band SNR of the phone on the master clock, `rank_phones`)
    prepare   cut from the phone on the master clock, high-pass + spectral-gate denoise + level (no heavy processing)
    convert   speech-to-speech request per piece with the teacher's voice id (cached by a hash of audio + settings + model)
    align     envelope cross-correlation of the converted piece with the source piece: shift to the master clock, drift check per piece,
              time-stretch only if the drift exceeds 80 ms

The pod-side quality gate (ASR WER per piece, voice check) is `v2v_gate`; the assembly of the soundtrack is `v2v_mix`.

    python -m dancecap.speech v2v plan | sources | prepare | probe | convert | align | status
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import SPEECH, TAKE, WORK

SR = 48000
V2V = SPEECH / "v2v"
KEY_FILE = Path.home() / "Desktop" / "eleven-key.txt"
API = "https://api.elevenlabs.io"
VOICE_NAMES = {"Larissa": "Larissa", "Kadu": "Kadu"}
MODEL_DEFAULT = "eleven_multilingual_sts_v2"
LEDGER = V2V / "spend_ledger.jsonl"

MAX_PIECE_S = 40.0          # whole turns, well under the 60 s the task allows
SPLIT_GAP_S = 1.2           # a pause this long ends a piece
MIN_PIECE_S = 0.45          # shorter stretches of speech stay original
PAD_S = 0.35                # context kept around the first / last word (never into a neighbour's word)
MIN_OVERLAP_S = 0.2         # shorter overlaps are word-boundary double labels, not two voices speaking together
OVERLAP_MARGIN_S = 0.08     # two words of different voices closer than this count as overlapped
OPENING_S = 9.0             # the owner's two joke moments: original until the first pause after this (unless owner_opening.json exists)
DRIFT_LIMIT_MS = 80.0       # time-stretch a converted piece only when its drift against the source exceeds this
LOG = print


class V2VError(RuntimeError):
    """a problem the message explains"""


# ------------------------------------------------------------------------------------------------ the plan


@dataclass
class Piece:
    id: str
    speaker: str
    t0: float                       # cut start (master seconds), words start at w0
    t1: float
    w0: float
    w1: float
    n_words: int
    text: str
    keep: list[list[float]] = field(default_factory=list)   # [[a, b], ...] inside [t0, t1] where the ORIGINAL is kept (voices overlap)
    cam: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


def read_transcript(path: Path | None = None) -> dict:
    return json.loads((path or SPEECH / "transcript_final.json").read_text(encoding="utf-8"))


def transcript_words(tr: dict) -> list[dict]:
    """every spoken word/filler of the two teachers {t0, t1, text, speaker, kind, utt}, sorted by start (events such as laughter tokens are not words)"""
    out = []
    for u in tr["utterances"]:
        if u.get("speaker") not in VOICE_NAMES:
            continue
        for w in u["words"]:
            if w.get("kind") in ("word", "filler"):
                out.append({"t0": float(w["start"]), "t1": float(w["end"]), "text": w["text"], "speaker": u["speaker"], "kind": w["kind"], "utt": u["id"]})
    out.sort(key=lambda w: (w["t0"], w["t1"]))
    return out


def opening_end(words: list[dict], owner_opening: Path | None = None, open_s: float = OPENING_S) -> float:
    """Master second up to which the ORIGINAL audio is kept: 0 when the owner has settled the opening (speech/owner_opening.json exists), else the
    middle of the first pause of at least 0.15 s that starts after `open_s` (so the splice sits between two words)."""
    p = owner_opening or (SPEECH / "owner_opening.json")
    if p.exists():
        return 0.0
    ws = [w for w in words if w["t1"] > 0]
    for a, b in zip(ws, ws[1:]):
        if a["t1"] >= open_s and b["t0"] - a["t1"] >= 0.15:
            return round((a["t1"] + b["t0"]) / 2, 3)
    return open_s


def overlap_intervals(words: list[dict], margin: float = OVERLAP_MARGIN_S) -> list[tuple[float, float, str, str]]:
    """[(a, b, speaker_a, speaker_b)] where words of DIFFERENT speakers overlap in time (grown by `margin` each side)"""
    out = []
    for i, a in enumerate(words):
        for b in words[i + 1:]:
            if b["t0"] > a["t1"] + 2 * margin:
                break
            if a["speaker"] != b["speaker"] and b["t0"] < a["t1"] + margin and a["t0"] < b["t1"] + margin:
                out.append((max(a["t0"], b["t0"]) - margin, min(a["t1"], b["t1"]) + margin, a["speaker"], b["speaker"]))
    return out


def _runs(words: list[dict]) -> list[list[dict]]:
    runs: list[list[dict]] = []
    for w in words:
        if runs and runs[-1][-1]["speaker"] == w["speaker"]:
            runs[-1].append(w)
        else:
            runs.append([w])
    return runs


def _split_run(run: list[dict], split_gap: float, max_len: float) -> list[list[dict]]:
    """cut a run at every pause >= split_gap, then (recursively) at the longest pause of any piece still longer than max_len"""
    parts: list[list[dict]] = [[run[0]]]
    for w in run[1:]:
        if w["t0"] - parts[-1][-1]["t1"] >= split_gap:
            parts.append([w])
        else:
            parts[-1].append(w)

    def cut(p: list[dict]) -> list[list[dict]]:
        if p[-1]["t1"] - p[0]["t0"] <= max_len or len(p) < 2:
            return [p]
        gaps = [(p[i + 1]["t0"] - p[i]["t1"], i) for i in range(len(p) - 1)
                if p[i]["t1"] - p[0]["t0"] > 4.0 and p[-1]["t1"] - p[i + 1]["t0"] > 4.0]
        if not gaps:
            gaps = [(p[i + 1]["t0"] - p[i]["t1"], i) for i in range(len(p) - 1)]
        _, i = max(gaps)
        return cut(p[:i + 1]) + cut(p[i + 1:])

    out: list[list[dict]] = []
    for p in parts:
        out += cut(p)
    return out


def plan_pieces(words: list[dict], *, open_end: float, duration: float, split_gap: float = SPLIT_GAP_S, max_len: float = MAX_PIECE_S,
                min_len: float = MIN_PIECE_S, pad: float = PAD_S) -> tuple[list[Piece], list[dict]]:
    """(pieces to convert, kept-original stretches [{t0, t1, speaker, why}]). Words before `open_end` or before 0 are original; a word of a teacher
    that overlaps a word of the other teacher keeps the original in the overlap (the piece around it is still converted, the overlap window is
    marked `keep`); a stretch of speech shorter than `min_len` stays original."""
    ws = [w for w in words if w["t1"] > 0]
    kept: list[dict] = []
    body = []
    for w in ws:
        if w["t0"] < open_end:
            continue
        body.append(w)
    if ws and open_end > 0:
        kept.append({"t0": 0.0, "t1": open_end, "speaker": "both", "why": "opening: the owner's two joke moments, words not yet settled by the owner"})
    ov = overlap_intervals(body)
    pieces: list[Piece] = []
    for run in _runs(body):
        for part in _split_run(run, split_gap, max_len):
            w0, w1 = part[0]["t0"], part[-1]["t1"]
            if w1 - w0 < min_len:
                kept.append({"t0": round(w0, 3), "t1": round(w1, 3), "speaker": part[0]["speaker"],
                             "why": f"stretch of {w1 - w0:.2f} s is shorter than {min_len} s: too short for a clean conversion", "text": " ".join(w["text"] for w in part)})
                continue
            idx = body.index(part[0])
            prev_w = body[idx - 1] if idx > 0 else None
            nxt = body[body.index(part[-1]) + 1] if body.index(part[-1]) + 1 < len(body) else None
            lp = pad if prev_w is None else max(0.03, min(pad, 0.5 * (w0 - prev_w["t1"])))
            rp = pad if nxt is None else max(0.03, min(pad, 0.5 * (nxt["t0"] - w1)))
            if prev_w is not None and prev_w["speaker"] != part[0]["speaker"]:
                lp = min(lp, max(0.03, 0.5 * (w0 - prev_w["t1"])))
            t0 = max(open_end, w0 - lp)
            t1 = min(duration, w1 + rp)
            keep = [[round(max(a, t0), 3), round(min(b, t1), 3)] for a, b, sa, sb in ov
                    if part[0]["speaker"] in (sa, sb) and b > t0 and a < t1 and any(x["t0"] < b and x["t1"] > a for x in part)]
            keep = [k for k in _merge_intervals(keep) if k[1] - k[0] >= MIN_OVERLAP_S]    # a few ms of double-labelled boundary are not an overlap
            p = Piece(id=f"p{len(pieces) + 1:03d}", speaker=part[0]["speaker"], t0=round(t0, 3), t1=round(t1, 3), w0=round(w0, 3), w1=round(w1, 3),
                      n_words=len(part), text=" ".join(w["text"] for w in part), keep=keep)
            if keep:
                p.notes.append("overlap with the other voice at " + ", ".join(f"{a:.2f}-{b:.2f}" for a, b in keep) + ": original kept there")
            pieces.append(p)
    for a, b in zip(pieces, pieces[1:]):                  # neighbouring cuts never overlap (a few tens of ms of pad-on-pad)
        if b.t0 < a.t1:
            m = round((a.w1 + b.w0) / 2, 3) if a.w1 < b.w0 else round((a.t1 + b.t0) / 2, 3)
            a.t1 = min(a.t1, m)
            b.t0 = max(b.t0, m)
    for p in pieces:                                      # the keep windows follow the final cut
        p.keep = [[max(a, p.t0), min(b, p.t1)] for a, b in p.keep if min(b, p.t1) - max(a, p.t0) >= MIN_OVERLAP_S]
    return pieces, kept


def stabilise_ids(pieces: list[Piece], old: list[Piece]) -> list[Piece]:
    """After a re-plan (e.g. the owner settled the opening) keep the id of every piece that is still the same stretch (same speaker, words within 50 ms),
    so the conversions, gate results and decisions that exist for it stay valid; new pieces get ids after the highest old one."""
    nxt = max([int(o.id[1:]) for o in old if re.fullmatch(r"p\d+", o.id)] + [0]) + 1
    used: set[str] = set()
    matched: set[int] = set()
    for k, p in enumerate(pieces):
        m = next((o for o in old if o.speaker == p.speaker and abs(o.w0 - p.w0) < 0.05 and abs(o.w1 - p.w1) < 0.05 and o.id not in used), None)
        if m:
            p.id = m.id
            used.add(m.id)
            matched.add(k)
    for k, p in enumerate(pieces):
        if k not in matched:
            while f"p{nxt:03d}" in used:
                nxt += 1
            p.id = f"p{nxt:03d}"
            used.add(p.id)
            nxt += 1
    return pieces


def _merge_intervals(iv: list[list[float]], gap: float = 0.0) -> list[list[float]]:
    iv = sorted(([a, b] for a, b in iv if b > a), key=lambda x: x[0])
    out: list[list[float]] = []
    for a, b in iv:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def cost_estimate(pieces: list[Piece], credits_per_min: float = 605.0) -> dict:
    """Speech-to-speech is billed by the duration of the audio sent (measured on this account: 10.1 credits per second = 605 per minute, the first request,
    `character-cost` header, see NOTES_V1.md). Returns the credits of one pass over all pieces."""
    sec = sum(p.dur for p in pieces)
    return {"audio_s": round(sec, 1), "credits_one_pass": int(math.ceil(sec / 60.0 * credits_per_min)), "credits_per_min": credits_per_min}


def write_plan(pieces: list[Piece], kept: list[dict], path: Path | None = None) -> Path:
    path = path or V2V / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "dancecap.speech/1/v2v_plan", "take": TAKE, "clock": "master seconds", "pieces": [asdict(p) for p in pieces],
                                "kept_original": kept, "estimate": cost_estimate(pieces)}, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def read_plan(path: Path | None = None) -> tuple[list[Piece], list[dict]]:
    d = json.loads((path or V2V / "plan.json").read_text(encoding="utf-8"))
    return [Piece(**p) for p in d["pieces"]], d["kept_original"]


# ------------------------------------------------------------------------------------------------ phones on the master clock


def phone_wav(cam: str) -> Path:
    fs = sorted((SPEECH / "audio").glob(f"{cam}-Recap-*.48k.wav"))
    if not fs:
        raise V2VError(f"no 48 kHz wav of phone {cam} in {SPEECH / 'audio'}")
    return fs[0]


def phone_on_master(cam: str, duration: float, *, cache: bool = True) -> np.ndarray:
    """mono float32 48 kHz of phone `cam` on the master clock (sample 0 = master 0), cached as a 16-bit wav"""
    from .. import film_recap_audio as fa
    from .sources import load_clocks
    out = V2V / "master" / f"{cam}.wav"
    n = int(round(duration * SR))
    if cache and out.exists() and out.stat().st_mtime >= phone_wav(cam).stat().st_mtime:
        x, _ = sf.read(str(out), dtype="float32")
        return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
    clk = load_clocks(WORK / "sync" / "sync.json", None).get(cam)
    if clk is None:
        raise V2VError(f"phone {cam} has no clock")
    x, sr = sf.read(str(phone_wav(cam)), dtype="float32")
    assert sr == SR, sr
    y, _ = fa.warp_to_master(x, clk.offset, clk.slope, n, SR)
    y = y[:, 0]
    if cache:
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.clip(y, -1, 1), SR, subtype="PCM_16")
    return y


def band_levels(x: np.ndarray, sr: int = SR, frame_s: float = 0.05) -> np.ndarray:
    from .. import film_recap_audio as fa
    return fa.band_db(x, sr, frame_s=frame_s)


def rank_phones(levels: dict[str, np.ndarray], covers: dict[str, tuple[float, float]], piece: Piece, words: list[dict], *,
                frame_s: float = 0.05, context_s: float = 12.0) -> list[dict]:
    """Phones that cover the whole piece, best first. Score = the phone's speech-band level on the piece's own WORD frames (median) minus its noise
    floor (10th percentile of the +-context_s neighbourhood) = the speech-to-noise ratio of THIS teacher's words on that phone; the level of the
    other teacher's words inside the piece is not counted (we only look at frames of the target's words)."""
    i = lambda t: int(round(t / frame_s))
    mine = [w for w in words if w["speaker"] == piece.speaker and w["t0"] >= piece.w0 - 1e-3 and w["t1"] <= piece.w1 + 1e-3]
    out = []
    for cam, e in levels.items():
        a, b = covers[cam]
        if a > piece.t0 + 0.05 or b < piece.t1 - 0.05:
            continue
        frames = np.concatenate([e[i(w["t0"]):max(i(w["t0"]) + 1, i(w["t1"]))] for w in mine]) if mine else e[i(piece.t0):i(piece.t1)]
        ctx = e[max(0, i(piece.t0 - context_s), i(a) + 1):min(len(e), i(piece.t1 + context_s))]
        speech, noise = float(np.median(frames)), float(np.percentile(ctx, 10))
        out.append({"cam": cam, "speech_db": round(speech, 2), "noise_db": round(noise, 2), "snr_db": round(speech - noise, 2)})
    out.sort(key=lambda r: -r["snr_db"])
    return out


# ------------------------------------------------------------------------------------------------ light pre-processing


def highpass(x: np.ndarray, sr: int, f: float = 80.0) -> np.ndarray:
    sos = signal.butter(2, f, btype="high", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)


def spectral_denoise(x: np.ndarray, sr: int, reduction_db: float = 9.0, nfft: int = 1024, noise_pct: float = 12.0, noise_scale: float = 8.0) -> np.ndarray:
    """Mild stationary-noise reduction with no latency: STFT, per-bin noise power = the `noise_pct` percentile over the piece's frames, a Wiener-like
    gain 1 - noise/power floored at -reduction_db, smoothed over frequency and time (little musical noise), inverse STFT. The floor is the whole
    point: speech is never gated, only the steady room tone is lowered by at most `reduction_db`."""
    f, t, Z = signal.stft(x.astype(np.float64), sr, nperseg=nfft, noverlap=nfft * 3 // 4, window="hann")
    P = np.abs(Z) ** 2
    noise = np.percentile(P, noise_pct, axis=1, keepdims=True) * noise_scale   # the 12th percentile of an exponential power is 0.128 of its mean: x8 = the mean
    floor = 10 ** (-reduction_db / 20.0)
    gain = np.clip(1.0 - noise / np.maximum(P, 1e-20), 0.0, 1.0) ** 0.5
    gain = np.maximum(gain, floor)
    from scipy.ndimage import uniform_filter
    gain = uniform_filter(gain, size=(3, 3), mode="nearest")
    gain = np.maximum(gain, floor)
    _, y = signal.istft(Z * gain, sr, nperseg=nfft, noverlap=nfft * 3 // 4, window="hann")
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def level_to(x: np.ndarray, target_db: float = -20.0, peak_db: float = -1.5) -> tuple[np.ndarray, float]:
    """gain so the 95th percentile of the 50 ms frame RMS (the speech level) is target_db, peak kept under peak_db; returns (y, gain_db)"""
    n = int(0.05 * SR)
    m = len(x) // n
    if m < 2:
        return x, 0.0
    rms = np.sqrt((x[:m * n].reshape(m, n).astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    cur = 20 * math.log10(float(np.percentile(rms, 95)) + 1e-9)
    g = target_db - cur
    pk = 20 * math.log10(float(np.abs(x).max()) + 1e-9)
    g = min(g, peak_db - pk)
    return (x * 10 ** (g / 20.0)).astype(np.float32), float(g)


def prepare_piece(x: np.ndarray, *, denoise: bool = True, reduction_db: float = 9.0) -> tuple[np.ndarray, dict]:
    """the audio sent to the voice changer: high-pass 80 Hz, mild spectral-gate denoise (<= reduction_db), level -20 dBFS (95th pct of 50 ms RMS)"""
    y = highpass(x, SR)
    if denoise:
        y = spectral_denoise(y, SR, reduction_db)
    y, g = level_to(y)
    return y, {"highpass_hz": 80, "denoise_db": reduction_db if denoise else 0, "gain_db": round(g, 2)}


def to_wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    b = io.BytesIO()
    sf.write(b, np.clip(x, -1, 1), sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


# ------------------------------------------------------------------------------------------------ the ElevenLabs client


class ElevenKey:
    """the API key, read at run time from the owner's Desktop file; its repr/str never show the value"""

    def __init__(self, path: Path | None = None):
        p = path or KEY_FILE
        try:
            v = p.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise V2VError(f"cannot read the key file {p.name}: {type(e).__name__}") from None
        if not v:
            raise V2VError("the key file is empty")
        self._v = v

    def header(self) -> dict:
        return {"xi-api-key": self._v}

    def __repr__(self) -> str:
        return "ElevenKey(<hidden>)"

    __str__ = __repr__


def http_session(key: ElevenKey | None = None):
    import requests
    s = requests.Session()
    s.headers.update((key or ElevenKey()).header())
    return s


def list_voices(session) -> dict[str, dict]:
    """{name: {voice_id, category, labels}} for the two teachers' voices of the owner's account (exact name match; others are not listed)"""
    r = session.get(API + "/v1/voices", timeout=60)
    if r.status_code != 200:
        raise V2VError(f"GET /v1/voices -> {r.status_code}: {_clean(r.text)}")
    found = {}
    for v in r.json()["voices"]:
        if v["name"] in VOICE_NAMES:
            found[v["name"]] = {"voice_id": v["voice_id"], "category": v.get("category"), "labels": v.get("labels")}
    missing = [n for n in VOICE_NAMES if n not in found]
    if missing:
        raise V2VError(f"voices not on the account: {missing}")
    return found


def _clean(text: str, n: int = 400) -> str:
    return re.sub(r"xi-api-key\S*", "<hidden>", text)[:n].replace("\n", " ")


def request_hash(audio: bytes, voice_id: str, model: str, settings: dict) -> str:
    h = hashlib.sha256()
    h.update(audio)
    h.update(json.dumps({"v": voice_id, "m": model, "s": settings}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def ledger_add(row: dict, path: Path | None = None) -> None:
    path = path or LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **row}) + "\n")


def ledger_credits(path: Path | None = None) -> int:
    path = path or LEDGER
    if not path.exists():
        return 0
    return sum(int(json.loads(l).get("credits") or 0) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


class QuotaError(V2VError):
    """the account has not enough credits (the run stops; what was converted so far stays usable)"""


def sts_request(session, voice_id: str, wav: bytes, *, model: str = MODEL_DEFAULT, settings: dict | None = None,
                remove_background_noise: bool = False, output_format: str = "pcm_44100", seed: int | None = None,
                purpose: str = "", timeout: float = 300.0) -> tuple[bytes, dict]:
    """POST /v1/speech-to-speech/{voice_id}: returns (raw audio bytes in `output_format`, {credits, request_id, seconds}). Errors raise V2VError
    (QuotaError for a quota problem); the key never appears in a message."""
    data = {"model_id": model, "remove_background_noise": "true" if remove_background_noise else "false"}
    if settings:
        data["voice_settings"] = json.dumps(settings)
    if seed is not None:
        data["seed"] = str(seed)
    t0 = time.time()
    r = session.post(f"{API}/v1/speech-to-speech/{voice_id}", params={"output_format": output_format}, data=data,
                     files={"audio": ("piece.wav", wav, "audio/wav")}, timeout=timeout)
    if r.status_code != 200:
        body = _clean(r.text, 600)
        if r.status_code in (401, 402, 429) and re.search(r"quota|credit", body, re.I):
            raise QuotaError(f"speech-to-speech -> {r.status_code}: {body}")
        raise V2VError(f"speech-to-speech -> {r.status_code}: {body}")
    info = {"credits": int(r.headers.get("character-cost") or 0), "request_id": r.headers.get("request-id") or r.headers.get("x-request-id"),
            "seconds": round(time.time() - t0, 1), "bytes": len(r.content)}
    ledger_add({"voice_id": voice_id[:6] + "..", "model": model, "settings": settings, "rbn": remove_background_noise, "purpose": purpose, **info})
    return r.content, info


def pcm_to_float(raw: bytes, output_format: str) -> tuple[np.ndarray, int]:
    """decode the response: pcm_<rate> = 16-bit little-endian mono, mp3_* through soundfile/ffmpeg"""
    m = re.match(r"pcm_(\d+)$", output_format)
    if m:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, int(m.group(1))
    x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return x.mean(axis=1), int(sr)


# ------------------------------------------------------------------------------------------------ text: normalisation and WER


_NORM = re.compile(r"[^a-z0-9' ]+")


def norm_words(text: str) -> list[str]:
    """lower-case words without punctuation (apostrophes kept: "it's"); the emoji/event tokens and the dash disappear"""
    t = text.lower().replace("\u2019", "'").replace("-", " ")
    return [w.strip("'") for w in _NORM.sub(" ", t).split() if w.strip("'")]


def edit_counts(ref: list[str], hyp: list[str]) -> dict:
    """Levenshtein alignment of two word lists: {sub, ins, del, n_ref, wer}"""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, sub, ins, dele = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            sub += int(ref[i - 1] != hyp[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {"sub": sub, "ins": ins, "del": dele, "n_ref": n, "wer": (sub + ins + dele) / max(n, 1)}


def wer(ref_text: str, hyp_text: str) -> float:
    return edit_counts(norm_words(ref_text), norm_words(hyp_text))["wer"]


def phone_asr_text(cam: str, piece: Piece, engines=("parakeet-tdt-0.6b-v3", "whisper-large-v3", "whisper-large-v3-turbo")) -> dict[str, str]:
    """{engine: the words that engine heard on phone `cam` inside [w0, w1] of the piece (master clock)} from the S1 raw outputs"""
    from .sources import load_clocks
    clk = load_clocks(WORK / "sync" / "sync.json", None)[cam]
    out = {}
    for eng in engines:
        p = SPEECH / "raw" / cam / f"{eng}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "words" in d:
            words = d["words"]
        else:                                          # whisper: the English pass
            ps = d["passes"]
            ps = [x for x in ps if x.get("requested") == "en"] or ps
            words = ps[0]["words"]
        sel = [w["word"] for w in words if piece.w0 - 0.15 <= clk.to_master((w["start"] + w["end"]) / 2) <= piece.w1 + 0.15]
        out[eng] = " ".join(sel)
    return out


def rank_by_asr(piece: Piece, cams: list[str]) -> dict[str, dict]:
    """{cam: {engine: wer vs the final text of the piece, mean}} : how well the S1 engines understood this teacher's words on that phone"""
    res = {}
    for cam in cams:
        t = phone_asr_text(cam, piece)
        w = {e: round(wer(piece.text, s), 3) for e, s in t.items()}
        res[cam] = {**w, "mean": round(float(np.mean(list(w.values()))), 3) if w else None}
    return res


# ------------------------------------------------------------------------------------------------ the phone per piece and the cut


def combine_rank(snr_rows: list[dict], asr: dict[str, dict], snr_weight: float = 0.01) -> list[dict]:
    """Final phone order for one piece: score = mean WER of the S1 engines on that phone (lower is better; None -> 0.5) - snr_weight * SNR dB.
    5 dB of speech-to-noise ratio are worth 0.05 WER. Returns [{cam, score, snr_db, asr_wer}] best first."""
    rows = []
    for r in snr_rows:
        a = (asr.get(r["cam"]) or {}).get("mean")
        a = 0.5 if a is None else a
        rows.append({"cam": r["cam"], "score": round(a - snr_weight * r["snr_db"], 4), "snr_db": r["snr_db"], "asr_wer": a})
    rows.sort(key=lambda x: x["score"])
    return rows


def choose_sources(pieces: list[Piece], words: list[dict], duration: float) -> dict[str, list[dict]]:
    """sets piece.cam (best) and notes the second best; returns {piece id: ranking}. Needs the phones on the master clock (cached)."""
    levels, covers = {}, {}
    for cam in ("00", "01", "02", "03", "05", "06"):
        x = phone_on_master(cam, duration)
        levels[cam] = band_levels(x)
        nz = np.flatnonzero(np.abs(x) > 1e-6)
        covers[cam] = (nz[0] / SR, nz[-1] / SR)
    table = {}
    for p in pieces:
        snr = rank_phones(levels, covers, p, words)
        asr = rank_by_asr(p, [r["cam"] for r in snr])
        rows = combine_rank(snr, asr)
        table[p.id] = rows
        if rows:
            p.cam = rows[0]["cam"]
    return table


def cut_master(x: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return x[max(0, int(round(t0 * SR))):int(round(t1 * SR))].astype(np.float32)


def prepare_all(pieces: list[Piece], duration: float, *, cams: dict[str, str] | None = None, tag: str = "pre", denoise: bool = True,
                reduction_db: float = 9.0) -> dict[str, dict]:
    """writes v2v/src/<id>.<cam>.wav (the phone's own cut) and v2v/<tag>/<id>.<cam>.wav (the audio sent); returns {id: info}"""
    cache: dict[str, np.ndarray] = {}
    out = {}
    for p in pieces:
        cam = (cams or {}).get(p.id) or p.cam
        if cam not in cache:
            cache[cam] = phone_on_master(cam, duration)
        raw = cut_master(cache[cam], p.t0, p.t1)
        pre, info = prepare_piece(raw, denoise=denoise, reduction_db=reduction_db)
        for d, a in (("src", raw), (tag, pre)):
            f = V2V / d / f"{p.id}.{cam}.wav"
            f.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(f), np.clip(a, -1, 1), SR, subtype="PCM_16")
        out[p.id] = {"cam": cam, **info}
    return out


# ------------------------------------------------------------------------------------------------ alignment of a converted piece to its source


def speech_envelope(x: np.ndarray, sr: int, frame_s: float = 0.01, band: tuple[float, float] = (250.0, 3500.0)) -> np.ndarray:
    """log energy (dB) of the speech band in `frame_s` frames (the quantity that survives a change of voice: syllable rhythm)"""
    sos = signal.butter(4, [band[0], min(band[1], sr / 2 - 100)], btype="band", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, x.astype(np.float64))
    n = int(round(frame_s * sr))
    m = len(y) // n
    e = (y[:m * n].reshape(m, n) ** 2).mean(axis=1)
    return 10.0 * np.log10(e + 1e-10)


def _znorm(e: np.ndarray) -> np.ndarray:
    e = np.clip(e - np.percentile(e, 10), 0, 40)          # the noise floor is zero, speech is up to 40 dB above it
    return (e - e.mean()) / (e.std() + 1e-9)


def envelope_lag(ref: np.ndarray, sig: np.ndarray, frame_s: float = 0.01, max_lag_s: float = 0.6) -> tuple[float, float]:
    """(lag in seconds, normalised peak) of `sig` against `ref` (both envelopes on the same frame grid): positive = `sig` is LATER; sub-frame by
    parabolic interpolation of the correlation peak"""
    a, b = _znorm(ref), _znorm(sig)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    k = int(max_lag_s / frame_s)
    full = signal.correlate(b, a, mode="full", method="fft")
    mid = n - 1
    lo, hi = max(0, mid - k), min(len(full), mid + k + 1)
    seg = full[lo:hi]
    j = int(np.argmax(seg))
    frac = 0.0
    if 0 < j < len(seg) - 1:
        y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return float((lo + j - mid + frac) * frame_s), float(seg[j] / n)


def window_lags(ref_x: np.ndarray, sig_x: np.ndarray, sr: int, centre_lag_s: float, win_s: float = 5.0, hop_s: float = 2.5,
                search_s: float = 0.25, min_peak: float = 0.35) -> list[dict]:
    """lag of `sig_x` against `ref_x` in sliding windows, each searched +-search_s around the global lag: [{t, lag_s, peak}] (t = window centre in the
    piece; the normalised correlation of the window's envelope against the shifted other envelope, parabolic peak)"""
    from numpy.lib.stride_tricks import sliding_window_view
    fr = 0.01
    er, es = _znorm_keep(speech_envelope(ref_x, sr, fr)), _znorm_keep(speech_envelope(sig_x, sr, fr))
    w, h = int(win_s / fr), int(hop_s / fr)
    c, S = int(round(centre_lag_s / fr)), int(round(search_s / fr))
    out = []
    i = 0
    while i + w <= len(er):
        seg = er[i:i + w]
        lo, hi = i + c - S, i + c + S + w
        if lo >= 0 and hi <= len(es):
            win = sliding_window_view(es[lo:hi], w)                       # [2S + 1, w]
            a = seg - seg.mean()
            b = win - win.mean(axis=1, keepdims=True)
            den = np.sqrt((a * a).sum() * (b * b).sum(axis=1)) + 1e-9
            cc = (b @ a) / den
            j = int(np.argmax(cc))
            frac = 0.0
            if 0 < j < len(cc) - 1:
                y0, y1, y2 = cc[j - 1], cc[j], cc[j + 1]
                d = y0 - 2 * y1 + y2
                frac = 0.5 * (y0 - y2) / d if abs(d) > 1e-12 else 0.0
            out.append({"t": round((i + w / 2) * fr, 3), "lag_s": round(float((c - S + j + frac) * fr), 4), "peak": round(float(cc[j]), 3)})
        i += h
    return [o for o in out if o["peak"] >= min_peak]


def _znorm_keep(e: np.ndarray) -> np.ndarray:
    return np.clip(e - np.percentile(e, 10), 0, 40)


def align_converted(src: np.ndarray, conv: np.ndarray, sr: int = SR, *, limit_ms: float = DRIFT_LIMIT_MS) -> tuple[np.ndarray, dict]:
    """Put the converted piece back on the source piece's time base (same length as `src`). 1) global lag = envelope cross-correlation; 2) windowed
    lags give the drift: a straight-line fit lag(t) = lag0 + rate * t; 3) the converted audio is shifted by lag0 (fractional); only when the fitted line
    leaves the global lag by more than `limit_ms` anywhere in the piece it is also time-stretched by 1/(1 + rate) (and the report says so)."""
    from .. import film_recap_audio as fa
    n = len(src)
    e_s, e_c = speech_envelope(src, sr), speech_envelope(conv, sr)
    lag0, peak = envelope_lag(e_s, e_c)
    wl = window_lags(src, conv, sr, lag0)
    rep: dict = {"lag_global_ms": round(lag0 * 1000, 1), "peak": round(peak, 3), "windows": len(wl), "stretched": False, "len_src_s": round(n / sr, 3),
                 "len_conv_s": round(len(conv) / sr, 3)}
    rate = 0.0
    lag_line = lag0
    if len(wl) >= 3:
        t = np.array([w["t"] for w in wl])
        l = np.array([w["lag_s"] for w in wl])
        # robust line: Theil-Sen
        slopes = [(l[j] - l[i]) / (t[j] - t[i]) for i in range(len(t)) for j in range(i + 1, len(t)) if t[j] - t[i] > 1.0]
        rate = float(np.median(slopes)) if slopes else 0.0
        icpt = float(np.median(l - rate * t))
        resid = l - (icpt + rate * t)
        rep["window_lag_ms"] = [round(float(x) * 1000, 1) for x in l]
        rep["window_resid_ms_max"] = round(float(np.abs(resid).max()) * 1000, 1)
        span = (t.max() - t.min())
        rep["drift_ms"] = round(abs(rate) * span * 1000, 1)
        rep["lag_spread_ms"] = round(float(l.max() - l.min()) * 1000, 1)
        if abs(rate) * (n / sr) * 1000 > limit_ms:
            rep["stretched"] = True
            rep["rate"] = round(rate, 6)
            lag_line = icpt
    # convert: y(t_src) = conv(t_src + lag(t)) ; lag(t) = lag_line + rate*t  (rate only when stretched)
    r = rate if rep["stretched"] else 0.0
    # warp_to_master maps out[m] = x(src sample (m - offset*sr)/slope): choose slope = 1 / (1 + r) and offset = -lag_line
    slope = 1.0 / (1.0 + r)
    y, _ = fa.warp_to_master(conv.astype(np.float32), -lag_line * slope if r else -lag_line, slope, n, sr)
    y = y[:, 0]
    rep["lag_applied_ms"] = round(lag_line * 1000, 1)
    return y.astype(np.float32), rep


# ------------------------------------------------------------------------------------------------ converting the pieces


@dataclass
class Variant:
    """one set of conversion parameters; `name` is the folder under v2v/conv/ and v2v/aligned/"""
    name: str
    model: str = MODEL_DEFAULT
    settings: dict | None = None
    remove_background_noise: bool = False
    pre: str = "pre"                     # which prepared audio is sent (folder under v2v/)
    output_format: str = "pcm_24000"


def voice_ids(path: Path | None = None) -> dict[str, str]:
    """{teacher: voice_id} from v2v/voices.json (written by `probe`; ids are not secrets) or, failing that, the account's voice list"""
    p = path or V2V / "voices.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
    else:
        d = list_voices(http_session())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return {k: v["voice_id"] for k, v in d.items()}


def conv_paths(variant: Variant, piece: Piece, cam: str) -> tuple[Path, Path]:
    d = V2V / "conv" / variant.name
    return d / f"{piece.id}.{cam}.wav", d / f"{piece.id}.{cam}.json"


def convert_piece(session, voice_id: str, piece: Piece, cam: str, variant: Variant, *, force: bool = False, request=sts_request) -> dict:
    """Send the prepared audio of `piece` (phone `cam`) to the voice changer unless the cache (a hash of audio + voice + model + settings) already has it.
    `request` is the HTTP layer (replaced in the tests). Returns the meta dict (credits, request id, hash, cached)."""
    wav_f, meta_f = conv_paths(variant, piece, cam)
    src_f = V2V / variant.pre / f"{piece.id}.{cam}.wav"
    audio = src_f.read_bytes()
    h = request_hash(audio, voice_id, variant.model, {"s": variant.settings, "rbn": variant.remove_background_noise, "f": variant.output_format})
    if not force and wav_f.exists() and meta_f.exists():
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        if meta.get("hash") == h:
            return {**meta, "cached": True}
    raw, info = request(session, voice_id, audio, model=variant.model, settings=variant.settings, remove_background_noise=variant.remove_background_noise,
                        output_format=variant.output_format, purpose=f"{variant.name} {piece.id}")
    x, sr = pcm_to_float(raw, variant.output_format)
    wav_f.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_f), x, sr, subtype="PCM_16")
    meta = {"hash": h, "piece": piece.id, "cam": cam, "voice": piece.speaker, "variant": asdict(variant), "rate": sr, "seconds_audio": round(len(x) / sr, 3),
            "sent_s": round(len(audio) / 2 / SR, 3), **info}
    meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {**meta, "cached": False}


def convert_all(pieces: list[Piece], variant: Variant, *, only: set[str] | None = None, force: bool = False, session=None, request=sts_request,
                budget_credits: int | None = None) -> dict:
    """Convert every piece (or `only`) with `variant`. Stops at the first QuotaError or when `budget_credits` would be exceeded (the converted ones
    stay usable). Returns {done, cached, credits, stopped}."""
    session = session or http_session()
    ids = voice_ids()
    res: dict = {"done": [], "cached": [], "credits": 0, "stopped": None, "failed": {}}
    for p in pieces:
        if only and p.id not in only:
            continue
        if budget_credits is not None and res["credits"] + math.ceil(p.dur * 10.2) > budget_credits:
            res["stopped"] = f"credit budget {budget_credits} reached before {p.id}"
            break
        try:
            m = convert_piece(session, ids[p.speaker], p, p.cam, variant, force=force, request=request)
        except QuotaError as e:
            res["stopped"] = f"quota: {e}"
            break
        except V2VError as e:
            res["failed"][p.id] = str(e)[:300]
            continue
        (res["cached"] if m["cached"] else res["done"]).append(p.id)
        res["credits"] += 0 if m["cached"] else int(m.get("credits") or 0)
        LOG(f"  {variant.name} {p.id} {p.speaker} {p.dur:.1f} s -> {'cached' if m['cached'] else str(m.get('credits')) + ' credits'}")
    return res


def align_all(pieces: list[Piece], variant: Variant, *, cams: dict[str, str] | None = None) -> dict[str, dict]:
    """aligned/<variant>/<id>.wav (48 kHz, the length of the piece's source cut, master time base) + one report per piece"""
    out: dict[str, dict] = {}
    for p in pieces:
        cam = (cams or {}).get(p.id) or p.cam
        wav_f, meta_f = conv_paths(variant, p, cam)
        if not wav_f.exists():
            continue
        conv, sr = sf.read(str(wav_f), dtype="float32")
        if sr != SR:
            conv = signal.resample_poly(conv, SR // math.gcd(SR, sr), sr // math.gcd(SR, sr)).astype(np.float32)
        src, _ = sf.read(str(V2V / "src" / f"{p.id}.{cam}.wav"), dtype="float32")
        y, rep = align_converted(src, conv)
        rep["clipped_samples"] = int((np.abs(y) >= 0.999).sum())
        dst = V2V / "aligned" / variant.name / f"{p.id}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), np.clip(y, -1, 1), SR, subtype="PCM_16")
        (V2V / "aligned" / variant.name / f"{p.id}.json").write_text(json.dumps({"piece": p.id, "cam": cam, **rep}, indent=1), encoding="utf-8")
        out[p.id] = rep
    return out


# ------------------------------------------------------------------------------------------------ optional: ElevenLabs audio isolation before the conversion


def isolate_piece(session, piece: Piece, cam: str, *, src_dir: str = "pre", out_dir: str = "iso", force: bool = False) -> dict:
    """POST /v1/audio-isolation (the same service and key as the conversion; ~10 credits per second of audio): the voice without the room.
    Writes v2v/<out_dir>/<id>.<cam>.wav (48 kHz) and returns {credits, cached}."""
    import subprocess
    out = V2V / out_dir / f"{piece.id}.{cam}.wav"
    meta = out.with_suffix(".json")
    src = V2V / src_dir / f"{piece.id}.{cam}.wav"
    audio = src.read_bytes()
    h = hashlib.sha256(audio).hexdigest()[:16]
    if not force and out.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("hash") == h:
        return {"credits": 0, "cached": True}
    r = session.post(API + "/v1/audio-isolation", files={"audio": ("piece.wav", audio, "audio/wav")}, timeout=300)
    if r.status_code != 200:
        body = _clean(r.text, 500)
        raise (QuotaError if re.search(r"quota|credit", body, re.I) else V2VError)(f"audio-isolation -> {r.status_code}: {body}")
    credits = int(r.headers.get("character-cost") or 0)
    ledger_add({"purpose": f"isolation {piece.id}", "credits": credits, "bytes": len(r.content)})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".mp3")
    tmp.write_bytes(r.content)
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(tmp), "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(out)], capture_output=True)
    tmp.unlink(missing_ok=True)
    if p.returncode != 0:
        raise V2VError("ffmpeg could not decode the isolated audio")
    meta.write_text(json.dumps({"hash": h, "credits": credits}), encoding="utf-8")
    return {"credits": credits, "cached": False}
