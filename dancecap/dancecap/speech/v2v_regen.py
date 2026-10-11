"""The class soundtrack made ONLY of regenerated ElevenLabs lines (agent V3), building on V2's pieces, pod helpers and mix code.

V2 left 27 of 81 sentence pieces as enhanced originals (speech-to-speech garbled or desynced them) and several cloned pieces were poor (ASR WER up to 1.0).
The owner wants every teacher line regenerated. Per piece, in this order:

  1. a speech-to-speech try (V2's, or one of up to 32 new ones: other input phone / enhancer, settings, model, seed) whose isolated ASR WER against the
     transcript sentence is <= WER_OK and whose speech-band envelope is in step with the transcript clock (no 1 s window more than LAG_OK_MS off; a try that
     lags more is re-timed with `v2v_warp.retime` and checked again);
  2. otherwise TEXT-TO-SPEECH in the same cloned voice from the transcript sentence (ElevenLabs text-to-speech with timestamps, context from the neighbouring
     sentences of the same teacher, up to 3 takes, the best by ASR WER), fitted to the sentence's time window: the word onsets of the take are moved onto the
     transcript word times (runs of continuous words are time-warped with WSOLA, pauses are silence), so the line sits on the transcript clock.

Everything is checked with the independent ASR (faster-whisper large-v3 on a pod, via V2's `v2v_full.pod_asr`). Audio of the tries lives next to V2's in
work/<take>/speech/v2v2/aligned/<tag>/<id>.wav (tags u<k>, tts<k>, o<k>), the candidate table in v2v2/regen.json, credits in v2v2/spend_ledger_v3.jsonl
(request ids and credits only, never the key). Voices of real people: never commit work/.
"""
from __future__ import annotations

import base64
import json
import math
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

from . import v2v as V
from . import v2v_full as F
from . import v2v_warp as W

SR = V.SR
V2 = F.V2
LEDGER = V2 / "spend_ledger_v3.jsonl"
REGEN = V2 / "regen.json"
CAP_CREDITS = 90000
WER_OK = 0.15                  # a try is accepted at this word error rate (isolated ASR, transcript sentence = reference)
LAG_OK_MS = 80.0               # and when no 1 s window of it is further than this from the transcript clock (speech-band envelope vs the best phone)
LAG_WARP_MAX_MS = 300.0        # tries lagging more than LAG_OK_MS and up to this are re-timed
TTS_MODEL = "eleven_multilingual_v2"
FILLERS = {"uh", "um", "mm", "hm", "hmm", "uhm", "er", "erm", "ah"}

# ------------------------------------------------------------------------------------------------ credits


def use_ledger() -> None:
    V.LEDGER = LEDGER


def credits_used() -> int:
    return V.ledger_credits(LEDGER)


_LOCK = threading.Lock()


# ------------------------------------------------------------------------------------------------ timing of a piece against the transcript clock


def lag_windows(env_ref: np.ndarray, env_y: np.ndarray, win_s: float = 1.0, hop_s: float = 0.5, frame_s: float = 0.01) -> list[float]:
    """local envelope lag (ms, positive = `y` later) of y against the reference over 1 s windows; windows without speech structure are skipped"""
    from .v2v_check import local_lag_ms
    n = min(len(env_ref), len(env_y))
    out, t = [], 0.0
    while t + win_s <= n * frame_s:
        lag = local_lag_ms(env_y[:n], env_ref[:n], t, t + win_s)
        if lag is not None:
            out.append(lag)
        t += hop_s
    return out


def lag_max(env_ref: np.ndarray, env_y: np.ndarray) -> float:
    return max((abs(x) for x in lag_windows(env_ref, env_y)), default=0.0)


def accept(wer: float | None, lag_ms: float | None, *, wer_ok: float = WER_OK, lag_ok: float = LAG_OK_MS) -> bool:
    return wer is not None and wer <= wer_ok and (lag_ms is None or lag_ms <= lag_ok)


def rank_key(c: dict) -> tuple:
    """order of candidates of one piece: accepted first, then lower WER, then smaller lag"""
    return (0 if accept(c.get("wer"), c.get("lag_ms")) else 1, c.get("wer") if c.get("wer") is not None else 9.0, c.get("lag_ms") or 0.0)


# ------------------------------------------------------------------------------------------------ the candidate table (regen.json)


def load_table() -> dict:
    return json.loads(REGEN.read_text(encoding="utf-8")) if REGEN.exists() else {}


_DIRTY: set = set()


def save_table(t: dict) -> None:
    """write the table, merging with what other processes saved meanwhile: only the (piece, tag) entries this process added or changed are applied on top of the file"""
    import os
    lock = REGEN.with_suffix(".lock")
    for _ in range(600):
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            time.sleep(0.1)
    try:
        disk = json.loads(REGEN.read_text(encoding="utf-8")) if REGEN.exists() else {}
        for pid, tag in list(_DIRTY):
            c = next((x for x in t.get(pid, []) if x["tag"] == tag), None)
            if c is not None:
                add_cand(disk, pid, c, mark=False)
        REGEN.write_text(json.dumps(disk, indent=1), encoding="utf-8")
        t.clear()
        t.update(disk)
        _DIRTY.clear()
    finally:
        try:
            os.rmdir(lock)
        except OSError:
            pass


def add_cand(table: dict, pid: str, cand: dict, mark: bool = True) -> None:
    lst = table.setdefault(pid, [])
    lst[:] = [c for c in lst if c["tag"] != cand["tag"]]
    lst.append(cand)
    if mark:
        _DIRTY.add((pid, cand["tag"]))


def best_cand(table: dict, pid: str, kinds=("sts", "tts")) -> dict | None:
    cs = [c for c in table.get(pid, []) if c.get("kind") in kinds and c.get("wer") is not None]
    return min(cs, key=rank_key) if cs else None


def best_accepted(table: dict, pid: str, kind: str) -> dict | None:
    cs = [c for c in table.get(pid, []) if c.get("kind") == kind and accept(c.get("wer"), c.get("lag_ms"))]
    return min(cs, key=rank_key) if cs else None


# ------------------------------------------------------------------------------------------------ the new speech-to-speech recipes (up to 32 per piece)

INPUTS = ["mossformer48__bf", "bf", "r1", "r2", "mossformer48__r1", "drrof_dn__bf", "pre1"]
SET_KEYS = ["S", "B", "L0", "L2", "M5s3", "T", "default", "L2nb"]
MODEL_EN = "eleven_english_sts_v2"


def sts_recipes(n: int = 32) -> list[tuple[str, str, str, int]]:
    """recipe k (1-based) = (input, settings key, model, seed); seeds start at 101 so no draw repeats a V2 one; the english model every fifth try"""
    out = []
    for k in range(1, n + 1):
        out.append((INPUTS[(k * 2) % len(INPUTS)], SET_KEYS[(k * 3 + k // 8) % len(SET_KEYS)], MODEL_EN if k % 5 == 0 else V.MODEL_DEFAULT, 100 + k))
    return out


# ------------------------------------------------------------------------------------------------ text-to-speech: request


def tts_request(session, voice_id: str, text: str, *, model: str = TTS_MODEL, settings: dict | None = None, previous_text: str | None = None,
                next_text: str | None = None, seed: int | None = None, purpose: str = "", timeout: float = 180.0) -> tuple[np.ndarray, int, dict, dict]:
    """POST /v1/text-to-speech/{voice}/with-timestamps (pcm_24000): returns (audio float32, rate, alignment{characters, starts, ends}, info{credits, request_id}).
    Errors raise V2VError (QuotaError for a quota problem); the key never appears in a message."""
    body: dict = {"text": text, "model_id": model}
    if settings:
        body["voice_settings"] = settings
    if previous_text:
        body["previous_text"] = previous_text
    if next_text:
        body["next_text"] = next_text
    if seed is not None:
        body["seed"] = int(seed)
    t0 = time.time()
    r = session.post(f"{V.API}/v1/text-to-speech/{voice_id}/with-timestamps", params={"output_format": "pcm_24000"}, json=body, timeout=timeout)
    if r.status_code != 200:
        msg = V._clean(r.text, 600)
        if r.status_code in (401, 402, 429) and re.search(r"quota|credit", msg, re.I):
            raise V.QuotaError(f"text-to-speech -> {r.status_code}: {msg}")
        raise V.V2VError(f"text-to-speech -> {r.status_code}: {msg}")
    j = r.json()
    raw = base64.b64decode(j["audio_base64"])
    x, sr = V.pcm_to_float(raw, "pcm_24000")
    al = j.get("alignment") or {}
    align = {"characters": al.get("characters") or [], "starts": al.get("character_start_times_seconds") or [], "ends": al.get("character_end_times_seconds") or []}
    info = {"credits": int(r.headers.get("character-cost") or len(text)), "request_id": r.headers.get("request-id") or r.headers.get("x-request-id"),
            "seconds": round(time.time() - t0, 1), "chars": len(text)}
    with _LOCK:
        V.ledger_add({"voice_id": voice_id[:6] + "..", "model": model, "kind": "tts", "settings": settings, "purpose": purpose, **info}, LEDGER)
    return x, sr, align, info


# ------------------------------------------------------------------------------------------------ text-to-speech: words, alignment, fitting to the transcript clock


def piece_tts_words(words: list[dict]) -> list[dict]:
    """the words of a piece to speak: the transcript words (kind word, not fillers) with their punctuation; [{text, punct, t0, t1}]"""
    out = []
    for w in words:
        if w.get("kind") != "word":
            continue
        t = (w.get("text") or "").strip()
        if not t or t.lower().strip(".,!?") in FILLERS:
            continue
        out.append({"text": t, "punct": w.get("punct") or "", "t0": float(w["start"]), "t1": float(w["end"])})
    return out


def tts_text_spans(words: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """the text to speak (words with their punctuation, one space between) and, per word, the character span of its letters [c0, c1) (punctuation excluded)"""
    text, spans = "", []
    for i, w in enumerate(words):
        if i:
            text += " "
        c0 = len(text)
        text += w["text"]
        c1 = len(text)
        text += w["punct"]
        spans.append((c0, c1))
    return text, spans


def word_times_from_alignment(align: dict, spans: list[tuple[int, int]]) -> list[tuple[float, float]]:
    """(start, end) seconds of every word from the character alignment of the text built by tts_text_spans"""
    st, en = align["starts"], align["ends"]
    out = []
    for c0, c1 in spans:
        c1 = min(c1, len(st))
        c0 = min(c0, c1 - 1)
        a, b = float(st[c0]), float(en[c1 - 1])
        out.append((a, max(b, a + 0.02)))
    # starts must increase (a take can report equal times for a very short word)
    for i in range(1, len(out)):
        if out[i][0] <= out[i - 1][0]:
            out[i] = (out[i - 1][0] + 0.005, max(out[i][1], out[i - 1][0] + 0.025))
    return out


def plan_runs(tw: list[tuple[float, float]], aw: list[tuple[float, float]], split_gap: float = 0.25) -> list[tuple[int, int]]:
    """runs [i0, i1) of words that stay one stretch of audio: a new run starts where the transcript has a pause of >= split_gap, or the take has one"""
    runs, s = [], 0
    for i in range(1, len(tw) + 1):
        if i == len(tw) or tw[i][0] - tw[i - 1][1] >= split_gap or aw[i][0] - aw[i - 1][1] >= split_gap:
            runs.append((s, i))
            s = i
    return runs


def run_time_map(t_on: list[float], a_on: list[float], hop_s: float = 0.008, lead_s: float = 0.05, tail_s: float = 0.12, rate_min: float = 0.75, rate_max: float = 1.4,
                 smooth_s: float = 0.25, tail_rate: tuple[float, float] = (0.8, 1.25)) -> tuple[float, np.ndarray]:
    """analysis time (take seconds) for every output hop of ONE run of continuous words. The take's word onsets a_on[i] should sound at the transcript onsets
    t_on[i] (both increasing). A word-by-word fit would need local rates of 0.4-2.5 (the transcript's word times are not a speech rhythm), which WSOLA cannot
    render well; instead the local rate wanted between anchors is SMOOTHED (Gaussian smooth_s), clipped to [rate_min, rate_max] and scaled so that the run
    starts and ends on its anchors (the first and last onset sound exactly on the transcript). Before the first anchor the rate is 1, after the last it is the
    run's mean rate clipped to tail_rate. Returns (u0, tmap): the run's output starts at transcript time u0 and tmap[k] is the take time to play at u0 + k * hop_s."""
    u0 = t_on[0] - lead_s
    span_t = max(t_on[-1] - t_on[0], 1e-3)
    mean_r = (a_on[-1] - a_on[0]) / span_t if len(t_on) > 1 else 1.0
    r_tail = float(np.clip(mean_r, *tail_rate))
    u1 = t_on[-1] + tail_s / r_tail
    n = int(math.ceil((u1 - u0) / hop_s)) + 1
    u = u0 + np.arange(n) * hop_s
    want = np.full(n, mean_r)
    for i in range(len(t_on) - 1):
        r = (a_on[i + 1] - a_on[i]) / max(t_on[i + 1] - t_on[i], 1e-3)
        want[(u >= t_on[i]) & (u < t_on[i + 1])] = float(np.clip(r, 0.4, 2.5))
    inside = (u >= t_on[0]) & (u <= t_on[-1])
    rate = np.ones(n)
    if smooth_s > 0 and len(t_on) > 1:
        sm = gaussian_filter1d(np.where(inside, want, mean_r), smooth_s / hop_s, mode="nearest")
    else:
        sm = np.where(inside, want, mean_r)
    rate[inside] = np.clip(sm[inside], rate_min, rate_max)
    if len(t_on) > 1 and inside.sum() > 2:                        # scale so that the integral over the run's span equals the take's span
        c = (a_on[-1] - a_on[0]) / max(float(rate[inside].sum() * hop_s), 1e-6)
        rate[inside] = rate[inside] * float(np.clip(c, 0.8, 1.25))
    rate[u > t_on[-1]] = r_tail
    tmap = np.empty(n)
    tmap[0] = a_on[0] - lead_s
    for k in range(1, n):
        tmap[k] = tmap[k - 1] + rate[k - 1] * hop_s
    k0 = int(round((t_on[0] - u0) / hop_s))
    tmap += a_on[0] - tmap[min(k0, n - 1)]
    return u0, tmap


def fit_take(x: np.ndarray, sr: int, tw: list[tuple[float, float]], aw: list[tuple[float, float]], n_out: int, t_origin: float = 0.0, *,
             split_gap: float = 0.25, tail_s: float = 0.12, fade_s: float = 0.008, refine: bool = True) -> tuple[np.ndarray, dict]:
    """Render a text-to-speech take `x` (mono, rate sr, word times aw in take seconds) onto the transcript clock: the output has n_out samples at SR (the
    piece's time base; piece time = transcript time - t_origin), word i starts at tw[i][0] - t_origin. Runs of continuous words are time-warped by WSOLA
    (`v2v_warp.wsola`, pitch kept), pauses between runs are silence. Returns (audio, report{runs, rate_min, rate_max, onset_pred_ms [per word]})."""
    if sr != SR:
        from scipy import signal as sg
        g = math.gcd(SR, sr)
        x = sg.resample_poly(x, SR // g, sr // g).astype(np.float32)
        sr = SR
    y = np.zeros(n_out + 4 * SR // 10, dtype=np.float64)
    rep_runs, rmin, rmax = [], 9.0, 0.0
    tw0 = [(a - t_origin, b - t_origin) for a, b in tw]
    for i in range(1, len(tw0)):                                  # transcript onsets strictly increasing
        if tw0[i][0] <= tw0[i - 1][0]:
            tw0[i] = (tw0[i - 1][0] + 0.01, max(tw0[i][1], tw0[i - 1][0] + 0.02))
    pred = [0.0] * len(tw)
    from .v2v_check import onset_near, speech_env_db
    env_take = speech_env_db(x, sr)
    env_take = np.maximum(env_take, np.percentile(env_take, 95) - 40.0)
    for (i0, i1) in plan_runs(tw0, aw, split_gap):
        t_on = [tw0[i][0] for i in range(i0, i1)]
        a_on = [aw[i][0] for i in range(i0, i1)]
        o_first = onset_near(env_take, aw[i0][0], before=0.12, after=0.25) if refine else None
        if o_first is not None and abs(o_first - aw[i0][0]) <= 0.25:     # the word is AUDIBLE here, not where the character alignment says it starts
            d_on = o_first - aw[i0][0]
            a_on = [a + d_on for a in a_on]
        else:
            d_on = 0.0
        if i1 - i0 == 1:                                          # a lone word: its own start and end are the two anchors (its duration is fitted too)
            t_on, a_on = [tw0[i0][0], max(tw0[i0][1], tw0[i0][0] + 0.12)], [aw[i0][0] + d_on, aw[i0][1] + d_on]
        u0, tmap = run_time_map(t_on, a_on, tail_s=tail_s + (aw[i1 - 1][1] - aw[i1 - 1][0]))
        seg = W.wsola(x, sr, np.clip(tmap, 0, len(x) / sr))
        # the take ends where the last word ends (plus a short decay): cut the output there
        end_take = min(aw[i1 - 1][1] + d_on + tail_s, len(x) / sr)
        n_keep = int(np.searchsorted(tmap, end_take)) * int(0.008 * sr)
        seg = seg[:max(n_keep, int(0.1 * sr))]
        if len(seg) > 0:
            from .v2v_mix import cos_fade
            seg = seg * cos_fade(len(seg), int(fade_s * sr))
        s0 = int(round(u0 * sr))
        a, b = max(s0, 0), min(s0 + len(seg), len(y))
        if b > a:
            y[a:b] += seg[a - s0:b - s0]
        rates = np.diff(tmap) / 0.008
        rmin, rmax = min(rmin, float(rates.min())), max(rmax, float(rates.max()))
        for i in range(i0, i1):                                   # predicted onset: where tmap passes the take's onset
            k = int(np.searchsorted(tmap, aw[i][0] + d_on))
            pred[i] = round((u0 + min(k, len(tmap) - 1) * 0.008 - tw0[i][0]) * 1000, 1)
        rep_runs.append({"words": [i0, i1], "t0": round(u0, 3)})
    return y[:n_out].astype(np.float32), {"runs": rep_runs, "rate_min": round(rmin, 2), "rate_max": round(rmax, 2), "onset_pred_ms": pred,
                                          "onset_pred_max_ms": round(max((abs(p) for p in pred), default=0.0), 1)}


def speed_for(take_s: float, target_s: float, lo: float = 0.85, hi: float = 1.2, dead_band: float = 0.08) -> float:
    """the API speed (voice setting) that brings a take of take_s seconds of speech to about target_s: 1.0 inside the dead band"""
    if target_s <= 0 or take_s <= 0:
        return 1.0
    r = take_s / target_s
    return 1.0 if abs(r - 1.0) <= dead_band else float(np.clip(r, lo, hi))


# text-to-speech voice settings of the three takes (stability, similarity, style): natural, a little different each time
TTS_TAKES = [
    {"stability": 0.45, "similarity_boost": 0.8, "style": 0.1, "use_speaker_boost": True},
    {"stability": 0.55, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True},
    {"stability": 0.35, "similarity_boost": 0.85, "style": 0.2, "use_speaker_boost": True},
]


# ------------------------------------------------------------------------------------------------ final choice


def tts_key(c: dict) -> tuple:
    """order of text-to-speech takes: clear words first (WER <= 0.10), then the least predicted onset drift (extreme time-warp rates cost 60 ms), then WER"""
    w = c.get("wer") if c.get("wer") is not None else 9.0
    extreme = c.get("rate_min", 1.0) < 0.8 or c.get("rate_max", 1.0) > 1.3
    return (0 if w <= 0.10 else 1, w if w > 0.10 else 0.0, c.get("onset_pred_max_ms", 0.0) + (60.0 if extreme else 0.0), w)


def choose_final(table: dict, pids: list[str], opening: dict | None = None) -> dict:
    """{piece id: {kind: sts | tts | original, tag, wer, flag}}: an accepted speech-to-speech try first, else the best text-to-speech take, else the best
    speech-to-speech try (flagged), else original (flagged: nothing regenerated)"""
    out = {}
    for pid in pids:
        s = best_accepted(table, pid, "sts")
        if s:
            out[pid] = {"kind": "sts", "tag": s["tag"], "wer": s["wer"], "lag_ms": s.get("lag_ms"), "flag": None}
            continue
        tt = [c for c in table.get(pid, []) if c.get("kind") == "tts" and c.get("wer") is not None]
        t = min(tt, key=tts_key) if tt else None
        if t:
            out[pid] = {"kind": "tts", "tag": t["tag"], "wer": t["wer"], "lag_ms": t.get("lag_ms"), "flag": "tts"}
            continue
        b = best_cand(table, pid, ("sts",))
        if b:
            out[pid] = {"kind": "sts", "tag": b["tag"], "wer": b["wer"], "lag_ms": b.get("lag_ms"), "flag": f"sts WER {b['wer']:.2f} (no better take)"}
        else:
            out[pid] = {"kind": "original", "tag": None, "wer": None, "lag_ms": None, "flag": "no regenerated line"}
    return out


# ------------------------------------------------------------------------------------------------ run-level re-alignment of a speech-to-speech piece


def realign_runs(y: np.ndarray, ref: np.ndarray, t0: list[float], t1: list[float], sr: int = SR, *, min_gap: float = 0.25, tol_ref: float = 0.08, act_s: float = 0.09,
                 max_shift_s: float = 0.15, fade_s: float = 0.01) -> tuple[np.ndarray, list[dict]]:
    """Put the stretches of a converted piece `y` (piece time base) that follow a pause onto the transcript: for every word that starts after a pause of >= min_gap
    (and the first word) the audible onset in `y` (speech-band envelope threshold) is compared with the transcript time t0[i]; when the best phone `ref` shows the
    word within tol_ref of the transcript (so the comparison is trustworthy) and `y` is off by more than act_s (at most max_shift_s), everything from the quiet
    point of the pause before that word up to the next corrected word is moved by the error (silence is inserted or removed in the pause; the first stretch moves
    with the start of the piece). Returns (audio of the same length, [{i, err_ms, shift_ms}])."""
    from .v2v_check import onset_near, speech_env_db
    n = len(y)
    ey, er = speech_env_db(y, sr), speech_env_db(ref, sr)
    ey = np.maximum(ey, np.percentile(ey, 95) - 40.0)               # a converted piece is digitally silent between words: its faint gating blips must not count as onsets (the mix has a room-tone bed)
    er = np.maximum(er, np.percentile(er, 95) - 40.0)
    adj = []
    for i in range(len(t0)):
        if i and t0[i] - t1[i - 1] < min_gap:
            continue
        oy, orf = onset_near(ey, t0[i]), onset_near(er, t0[i])
        if oy is None or orf is None or abs(orf - t0[i]) > tol_ref:
            continue
        err = oy - t0[i]
        if act_s < abs(err) <= max_shift_s:
            adj.append((i, err))
    if not adj:
        return y, []
    # segment boundaries: the quietest 10 ms frame in the middle half of the pause before each corrected word
    cuts, shifts, rows = [0], [0.0], []
    for i, err in adj:
        if i == 0:
            shifts[0] = -err
        else:
            a, b = t1[i - 1], t0[i]
            lo, hi = int((a + 0.25 * (b - a)) / 0.01), max(int((b - 0.25 * (b - a)) / 0.01), int((a + 0.25 * (b - a)) / 0.01) + 1)
            k = lo + int(np.argmin(ey[lo:hi])) if hi <= len(ey) and hi > lo else int(0.5 * (a + b) / 0.01)
            cuts.append(int(k * 0.01 * sr))
            shifts.append(-err)
        rows.append({"i": i, "err_ms": round(err * 1000, 1), "shift_ms": round(-err * 1000, 1)})
    cuts.append(n)
    out = np.zeros(n + int(max_shift_s * sr) * 2, dtype=np.float64)
    pad = int(max_shift_s * sr)
    f = int(fade_s * sr)
    for j in range(len(cuts) - 1):
        a, b = cuts[j], cuts[j + 1]
        seg = y[a:b].astype(np.float64)
        w = np.ones(len(seg))
        if j > 0 and len(seg) > 2 * f:
            w[:f] = 0.5 - 0.5 * np.cos(np.pi * np.arange(f) / f)
        if j < len(cuts) - 2 and len(seg) > 2 * f:
            w[-f:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(f) / f)
        s0 = pad + a + int(round(shifts[j] * sr))
        out[s0:s0 + len(seg)] += seg * w
    return out[pad:pad + n].astype(np.float32), rows
