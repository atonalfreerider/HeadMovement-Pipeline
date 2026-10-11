"""Orchestration of the regenerated-lines soundtrack (agent V3): speech-to-speech batches, text-to-speech takes, ASR on the pod, the final choice.

Needs the owner's key file (read at run time, never printed), the V3 pod (`v2v_pod.use_label("V3", "pod_v3.json")`) and the work files of V2. The pure parts
(timing, fitting, choosing) are in `v2v_regen`.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import v2v as V
from . import v2v_full as F
from . import v2v_warp as W
from . import v2v_regen as R
from .v2v_regen import SR, V2

_ALLW: list = []
_ENV: dict = {}


def raw_words() -> list[dict]:
    """every word/filler of the two teachers with punctuation: {text, punct, start, end, kind, speaker}"""
    out = []
    for u in V.read_transcript()["utterances"]:
        if u.get("speaker") not in V.VOICE_NAMES:
            continue
        out += [{**w, "speaker": u["speaker"]} for w in u["words"] if w.get("kind") in ("word", "filler")]
    out.sort(key=lambda w: (w["start"], w["end"]))
    return out


_ALLW_LOCK = threading.Lock()


def allw() -> list[dict]:
    with _ALLW_LOCK:                          # several threads ask at once: fill the list exactly once
        if not _ALLW:
            _ALLW.extend(raw_words())
    return _ALLW


def words_of(p: V.Piece) -> list[dict]:
    return [w for w in allw() if w["speaker"] == p.speaker and w["start"] >= p.w0 - 1e-3 and w["end"] <= p.w1 + 1e-3]


def aligned_path(tag: str, pid: str) -> Path:
    return V2 / "aligned" / tag / f"{pid}.wav"


def ref_env(pid: str) -> np.ndarray:
    """speech-band envelope of the best phone's cut of the piece (the transcript-clock reference for the lag checks)"""
    from .v2v_check import speech_env_db
    if pid not in _ENV:
        x, _ = sf.read(str(F.IN_DIR / "r1" / f"{pid}.wav"), dtype="float32")
        _ENV[pid] = speech_env_db(x, SR)
    return _ENV[pid]


def measure_lag(tag: str, pid: str) -> float:
    from .v2v_check import speech_env_db
    y, _ = sf.read(str(aligned_path(tag, pid)), dtype="float32")
    return round(R.lag_max(ref_env(pid), speech_env_db(y, SR)), 1)


def tts_ref(p: V.Piece) -> str:
    """reference of a text-to-speech take: the piece's words without fillers"""
    return " ".join(w["text"] for w in R.piece_tts_words(words_of(p)))


# ------------------------------------------------------------------------------------------------ V2's tries and re-timing


def scan_existing(pieces: list[V.Piece], table: dict, top: int = 4) -> dict:
    """V2's tries: per piece the `top` lowest-WER sets (isolated ASR) with their lag against the best phone -> table (kind sts)"""
    sets = {f.stem.replace("aligned__", ""): json.loads(f.read_text(encoding="utf-8")) for f in sorted(F.ASR_DIR.glob("aligned__*.json"))
            if f.stem.replace("aligned__", "") not in ("final", "full")}
    for p in pieces:
        cs = []
        for tag, a in sets.items():
            it = a.get(p.id)
            if it and aligned_path(tag, p.id).exists():
                cs.append((V.edit_counts(V.norm_words(p.text), V.norm_words(it["text"]))["wer"], tag))
        for w, tag in sorted(cs)[:top]:
            if not any(c["tag"] == tag for c in table.get(p.id, [])):
                R.add_cand(table, p.id, {"tag": tag, "kind": "sts", "src": "v2", "wer": round(w, 4), "lag_ms": measure_lag(tag, p.id)})
    return table


def retime_cands(pieces: list[V.Piece], table: dict, ids: set[str] | None = None) -> list[str]:
    """tries with WER <= WER_OK whose lag is above LAG_OK_MS (and <= LAG_WARP_MAX_MS) are re-timed onto the best phone's rhythm (v2v_warp) into the new tag
    <tag>_rt (WER unknown until the ASR on the pod). Returns the new tags."""
    new = set()
    for p in pieces:
        if ids and p.id not in ids:
            continue
        if R.best_accepted(table, p.id, "sts"):
            continue
        for c in list(table.get(p.id, [])):
            if c.get("kind") != "sts" or c.get("wer") is None or c["wer"] > R.WER_OK or c["tag"].endswith("_rt"):
                continue
            if not (R.LAG_OK_MS < (c.get("lag_ms") or 0) <= R.LAG_WARP_MAX_MS):
                continue
            tg = c["tag"] + "_rt"
            if any(x["tag"] == tg for x in table[p.id]):
                continue
            src, _ = sf.read(str(F.IN_DIR / "r1" / f"{p.id}.wav"), dtype="float32")
            y, _ = sf.read(str(aligned_path(c["tag"], p.id)), dtype="float32")
            z, info = W.retime(src, y, SR)
            f = aligned_path(tg, p.id)
            f.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(f), np.clip(z, -1, 1), SR, subtype="PCM_16")
            R.add_cand(table, p.id, {"tag": tg, "kind": "sts", "src": "retime of " + c["tag"], "wer": None, "lag_ms": measure_lag(tg, p.id), "retime": info})
            new.add(tg)
    return sorted(new)


def fill_wers(pieces: list[V.Piece], table: dict) -> int:
    """fill 'wer' / 'text' of every candidate with an ASR result file (v2v2/asr/aligned__<tag>.json); returns the number filled"""
    n = 0
    cache: dict = {}
    for p in pieces:
        for cand in table.get(p.id, []):
            t = cand["tag"]
            if cand.get("wer") is not None:
                continue
            if t not in cache:
                f = F.ASR_DIR / f"aligned__{t}{'x' if t.startswith('tts') else ''}.json"
                cache[t] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            it = cache[t].get(p.id)
            if it is None:
                continue
            ref = tts_ref(p) if t.startswith("tts") else p.text
            c = V.edit_counts(V.norm_words(ref), V.norm_words(it["text"]))
            cand["wer"] = round(c["wer"], 4)
            cand["text"] = " ".join(it["text"].split())[:300]
            R._DIRTY.add((p.id, t))
            n += 1
    return n


def trim_for_asr(tag: str, pad_s: float = 0.3, floor_db: float = 45.0) -> str:
    """aligned/<tag>x/: each file of a text-to-speech tag cut to its speech +- pad_s (long digital silence makes the ASR repeat sentences); returns the new tag"""
    out = V2 / "aligned" / f"{tag}x"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((V2 / "aligned" / tag).glob("*.wav")):
        x, sr = sf.read(str(f), dtype="float32")
        e = np.abs(x)
        idx = np.nonzero(e > e.max() * 10 ** (-floor_db / 20))[0]
        a, b = (max(int(idx[0]) - int(pad_s * sr), 0), min(int(idx[-1]) + int(pad_s * sr), len(x))) if len(idx) else (0, len(x))
        sf.write(str(out / f.name), x[a:b], sr, subtype="PCM_16")
    return f"{tag}x"


def asr_tags(pieces: list[V.Piece], tags: list[str], embed: bool = False) -> dict:
    """ASR on the pod of aligned/<tag>/ for each tag (text-to-speech tags: the trimmed copies <tag>x); reloads the table, fills the WERs, saves it, returns it"""
    tags = [trim_for_asr(t) if t.startswith("tts") and not t.endswith("x") else t for t in tags]
    if tags:
        F.push_aligned(tags)
        F.pod_asr([f"aligned/{t}" for t in tags], embed=embed, force=True)
        F.pull_asr([f"aligned__{t}" for t in tags], npz=embed)
    with R._LOCK:
        table = R.load_table()
        fill_wers(pieces, table)
        R.save_table(table)
    return table


# ------------------------------------------------------------------------------------------------ speech-to-speech batches


def convert_sts(session, voice_id: str, src_wav: Path, tag: str, pid: str, *, model: str, settings_key: str, seed: int, budget: int) -> dict:
    """one speech-to-speech request into v2v2/conv/<tag>/<pid>.wav (cached by a hash); refuses beyond the V3 credit budget"""
    out_dir = V2 / "conv" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_f, meta_f = out_dir / f"{pid}.wav", out_dir / f"{pid}.json"
    audio = src_wav.read_bytes()
    st = F.SETTINGS[settings_key]
    h = V.request_hash(audio, voice_id, model, {"s": st, "rbn": True, "seed": seed, "f": "pcm_24000"})
    if wav_f.exists() and meta_f.exists() and json.loads(meta_f.read_text(encoding="utf-8")).get("hash") == h:
        return {"cached": True}
    sec = len(audio) / 2 / SR
    if R.credits_used() + math.ceil(sec * 10.3) > budget:
        raise V.V2VError(f"V3 credit budget {budget} would be exceeded (used {R.credits_used()})")
    for attempt in range(6):
        try:
            raw, info = V.sts_request(session, voice_id, audio, model=model, settings=st, remove_background_noise=True, output_format="pcm_24000", seed=seed,
                                      purpose=f"{tag} {pid}")
            break
        except V.QuotaError:
            raise
        except V.V2VError as e:
            if "429" in str(e) or "concurrent" in str(e).lower():
                time.sleep(3 + 4 * attempt)
                continue
            raise
    else:
        raise V.V2VError("rate limited: gave up")
    x, sr = V.pcm_to_float(raw, "pcm_24000")
    sf.write(str(wav_f), x, sr, subtype="PCM_16")
    meta_f.write_text(json.dumps({"hash": h, "model": model, "settings": settings_key, "seed": seed, "sent_s": round(sec, 3), **info}), encoding="utf-8")
    return {"cached": False, **info}


def sts_batch(pieces: list[V.Piece], table: dict, ids: set[str], ks: list[int], *, threads: int = 3, budget: int = R.CAP_CREDITS, log=print) -> list[str]:
    """new speech-to-speech tries u<k> for recipes ks (1-based) on the pieces `ids`: convert, align to the input, measure the lag, re-time a try that lags
    (80-300 ms). Returns the tags (ASR still to run). Stops at a quota error or the credit budget."""
    R.use_ledger()
    recipes = R.sts_recipes(max(ks))
    vids = V.voice_ids()
    stop = {"why": None}
    lk = threading.Lock()
    sel = [p for p in pieces if p.id in ids]
    for k in ks:
        cand = recipes[k - 1][0]
        if not (F.IN_DIR / cand).exists() or len(list((F.IN_DIR / cand).glob("*.wav"))) < len(pieces):
            F.pull_out(cand)
        F.make_sts_inputs(sel, cand)
    local = threading.local()

    def work(job):
        p, k = job
        if stop["why"]:
            return None
        cand, skey, model, seed = recipes[k - 1]
        tag = f"u{k}"
        if not hasattr(local, "s"):
            local.s = V.http_session()
        src = F.sts_input_path(cand, p.id)
        if not src.exists():
            return None
        try:
            convert_sts(local.s, vids[p.speaker], src, tag, p.id, model=model, settings_key=skey, seed=seed, budget=budget)
        except V.QuotaError as e:
            stop["why"] = f"quota: {e}"
            return None
        except V.V2VError as e:
            if "budget" in str(e):
                stop["why"] = str(e)
            log(f"{tag} {p.id}: {str(e)[:120]}")
            return None
        F.align_to_source(V2 / "conv" / tag / f"{p.id}.wav", src, aligned_path(tag, p.id))
        lag = measure_lag(tag, p.id)
        entry = {"tag": tag, "kind": "sts", "src": "v3", "k": k, "input": cand, "settings": skey, "model": model, "seed": seed, "wer": None, "lag_ms": lag}
        if R.LAG_OK_MS < lag <= R.LAG_WARP_MAX_MS:
            r1, _ = sf.read(str(F.IN_DIR / "r1" / f"{p.id}.wav"), dtype="float32")
            y, _ = sf.read(str(aligned_path(tag, p.id)), dtype="float32")
            z, info = W.retime(r1, y, SR)
            sf.write(str(aligned_path(tag, p.id)), np.clip(z, -1, 1), SR, subtype="PCM_16")
            entry.update(retime=info, lag_before_ms=lag, lag_ms=measure_lag(tag, p.id))
        with lk:
            R.add_cand(table, p.id, entry)
        return tag

    jobs = [(p, k) for k in ks for p in sel]
    with cf.ThreadPoolExecutor(threads) as ex:
        done = [r for r in ex.map(work, jobs) if r]
    R.save_table(table)
    if stop["why"]:
        log("STOPPED: " + stop["why"])
    return sorted(set(done), key=lambda s: int(s[1:]))


# ------------------------------------------------------------------------------------------------ text-to-speech takes


def piece_context(pieces: list[V.Piece], p: V.Piece) -> tuple[str | None, str | None]:
    """text of the previous / next sentence of the same teacher when it follows within 2 s (prosody context for the text-to-speech request)"""
    i = [q.id for q in pieces].index(p.id)

    def txt(q):
        text, _ = R.tts_text_spans(R.piece_tts_words(words_of(q)))
        return text

    prv = pieces[i - 1] if i > 0 and pieces[i - 1].speaker == p.speaker and p.t0 - pieces[i - 1].t1 < 2.0 else None
    nxt = pieces[i + 1] if i + 1 < len(pieces) and pieces[i + 1].speaker == p.speaker and pieces[i + 1].t0 - p.t1 < 2.0 else None
    return (txt(prv) if prv else None), (txt(nxt) if nxt else None)


def tts_available_end(pieces: list[V.Piece], p: V.Piece) -> float:
    """the latest transcript second the piece's audio may reach: 0.4 s past the sentence, never into the next sentence of anybody (kept 40 ms)"""
    later = [q.w0 for q in pieces if q.w0 > p.w1 + 1e-3]
    nxt = min(later) if later else 1e9
    return min(p.t1 + 0.4, nxt - 0.04)


def tts_take(session, pieces: list[V.Piece], p: V.Piece, take: int, speed: float, table: dict, *, budget: int = R.CAP_CREDITS) -> dict | None:
    """one text-to-speech take of the piece in its teacher's voice, fitted to the transcript clock -> aligned/tts<take>/<id>.wav; adds the candidate to the table"""
    R.use_ledger()
    ws = R.piece_tts_words(words_of(p))
    if not ws:
        return None
    text, spans = R.tts_text_spans(ws)
    if R.credits_used() + len(text) > budget:
        raise V.V2VError(f"V3 credit budget {budget} would be exceeded")
    prev, nxt = piece_context(pieces, p)
    st = dict(R.TTS_TAKES[(take - 1) % len(R.TTS_TAKES)])
    if abs(speed - 1.0) > 1e-6:
        st["speed"] = round(speed, 3)
    seed = 200 + take * 17 + int(p.id[1:])
    vid = V.voice_ids()[p.speaker]
    raw = V2 / "tts_raw" / f"tts{take}"
    raw.mkdir(parents=True, exist_ok=True)
    key = V.request_hash(text.encode(), vid, R.TTS_MODEL, {"s": st, "seed": seed, "prev": prev, "next": nxt})
    cf_, cj = raw / f"{p.id}.wav", raw / f"{p.id}.json"
    if cf_.exists() and cj.exists() and json.loads(cj.read_text(encoding="utf-8")).get("key") == key:      # a take already paid for
        meta = json.loads(cj.read_text(encoding="utf-8"))
        x, sr = sf.read(str(cf_), dtype="float32")
        al, info = meta["align"], meta["info"]
    else:
        x, sr, al, info = R.tts_request(session, vid, text, settings=st, previous_text=prev, next_text=nxt, seed=seed, purpose=f"tts{take} {p.id}")
        sf.write(str(cf_), x, sr, subtype="PCM_16")
        cj.write_text(json.dumps({"key": key, "align": al, "info": info}), encoding="utf-8")
    aw = R.word_times_from_alignment(al, spans)
    tw = [(w["t0"], w["t1"]) for w in ws]
    n_out = int(round((tts_available_end(pieces, p) - p.t0) * SR))
    y, rep = R.fit_take(x, sr, tw, aw, n_out, t_origin=p.t0)
    tag = f"tts{take}"
    f = aligned_path(tag, p.id)
    f.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(f), np.clip(y, -1, 1), SR, subtype="PCM_16")
    # how long the take is against the window: speech seconds of the take over those of the transcript, run by run
    runs = R.plan_runs([(a - p.t0, b - p.t0) for a, b in tw], aw)
    s_take = sum(aw[i1 - 1][1] - aw[i0][0] for i0, i1 in runs)
    s_tgt = sum(tw[i1 - 1][1] - tw[i0][0] for i0, i1 in runs)
    entry = {"tag": tag, "kind": "tts", "take": take, "speed": speed, "settings": st, "seed": seed, "wer": None, "lag_ms": None, "chars": len(text), "ratio": round(s_take / max(s_tgt, 1e-3), 3),
             "rate_min": rep["rate_min"], "rate_max": rep["rate_max"], "onset_pred_max_ms": rep["onset_pred_max_ms"], "credits": info["credits"], "text": text}
    with R._LOCK:
        R.add_cand(table, p.id, entry)
    return entry


def tts_batch(pieces: list[V.Piece], table: dict, ids: set[str], take: int, speeds: dict[str, float] | None = None, *, threads: int = 3,
              budget: int = R.CAP_CREDITS, log=print) -> list[str]:
    """text-to-speech take number `take` for every piece in ids (speed per piece from an earlier take's length ratio); returns [tag] when any was made"""
    sel = [p for p in pieces if p.id in ids]
    stop = {"why": None}
    lk = threading.Lock()
    local = threading.local()

    def work(p):
        if stop["why"]:
            return None
        if not hasattr(local, "s"):
            local.s = V.http_session()
        for attempt in range(5):
            try:
                with lk:
                    pass
                e = tts_take(local.s, pieces, p, take, (speeds or {}).get(p.id, 1.0), table, budget=budget)
                return e
            except V.QuotaError as ex:
                stop["why"] = f"quota: {ex}"
                return None
            except V.V2VError as ex:
                if "429" in str(ex) or "concurrent" in str(ex).lower():
                    time.sleep(3 + 4 * attempt)
                    continue
                if "budget" in str(ex):
                    stop["why"] = str(ex)
                log(f"tts{take} {p.id}: {str(ex)[:150]}")
                return None
        return None

    with cf.ThreadPoolExecutor(threads) as ex:
        res = [r for r in ex.map(work, sel) if r]
    R.save_table(table)
    if stop["why"]:
        log("STOPPED: " + stop["why"])
    return [f"tts{take}"] if res else []


# ------------------------------------------------------------------------------------------------ the opening (owner-pending words): speech-to-speech only, never text-to-speech

OPENING_PIECES = [   # (id, teacher, cut t0, cut t1, first word, last word end, words) -- transcript_final.json, the words the audit still flags `needs_owner`
    ("o0", "Larissa", 3.50, 4.80, 3.60, 4.64, "this is Kadu"),
    ("o1", "Kadu", 5.35, 6.85, 5.45, 6.716, "yeah look at my red hair"),
    ("o2", "Larissa", 6.75, 8.25, 6.806, 8.146, "and yes it's very red"),
    ("o3", "Larissa", 8.43, 9.62, 8.533, 9.513, "we talked to"),
]
OPENING_CAMS = ["00", "01", "02", "05"]
OPENING_SETS = ["S", "B", "L0", "L2"]


def opening_pieces() -> list[V.Piece]:
    return [V.Piece(id=i, speaker=s, t0=a, t1=b, w0=w0, w1=w1, n_words=len(t.split()), text=t) for i, s, a, b, w0, w1, t in OPENING_PIECES]


def opening_batch(table: dict, ks: list[int], *, only: set[str] | None = None, threads: int = 3, budget: int = R.CAP_CREDITS, log=print) -> list[str]:
    """tries o<k> (k = 1..): phone OPENING_CAMS[(k-1) % 4] (light clean-up, as V1), settings OPENING_SETS[(k-1) // 4 % 4], seed 300 + k, for each opening piece"""
    R.use_ledger()
    vids = V.voice_ids()
    ps = [p for p in opening_pieces() if not only or p.id in only]
    cache: dict = {}
    srcs: dict = {}
    for p in ps:
        for k in ks:
            cam = OPENING_CAMS[(k - 1) % 4]
            if cam not in cache:
                cache[cam] = V.phone_on_master(cam, F.DURATION)
            cut = V.cut_master(cache[cam], p.t0, p.t1)
            pre, _ = V.prepare_piece(cut)
            y = F.prepare_for_sts(pre)
            f = V2 / "sts_in" / f"o{k}" / f"{p.id}.wav"
            f.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(f), np.clip(y, -1, 1), SR, subtype="PCM_16")
            srcs[(p.id, k)] = (f, cut)
    local = threading.local()
    stop = {"why": None}

    def work(job):
        p, k = job
        if stop["why"]:
            return None
        if not hasattr(local, "s"):
            local.s = V.http_session()
        f, cut = srcs[(p.id, k)]
        tag = f"o{k}"
        try:
            convert_sts(local.s, vids[p.speaker], f, tag, p.id, model=V.MODEL_DEFAULT, settings_key=OPENING_SETS[(k - 1) // 4 % 4], seed=300 + k, budget=budget)
        except V.V2VError as e:
            stop["why"] = str(e) if ("budget" in str(e) or isinstance(e, V.QuotaError)) else None
            log(f"{tag} {p.id}: {str(e)[:120]}")
            return None
        F.align_to_source(V2 / "conv" / tag / f"{p.id}.wav", f, aligned_path(tag, p.id))
        np.save(V2 / "conv" / tag / f"{p.id}.cut.npy", cut)
        R.add_cand(table, p.id, {"tag": tag, "kind": "sts", "src": "opening", "k": k, "cam": OPENING_CAMS[(k - 1) % 4], "settings": OPENING_SETS[(k - 1) // 4 % 4],
                                 "wer": None, "lag_ms": None})
        return tag

    with cf.ThreadPoolExecutor(threads) as ex:
        done = [r for r in ex.map(work, [(p, k) for p in ps for k in ks]) if r]
    R.save_table(table)
    return sorted(set(done), key=lambda s: int(s[1:]))


# ------------------------------------------------------------------------------------------------ the final choice and the assembly of class_mix_regen.wav

OUT_WAV_NAME = "class_mix_regen.wav"
BED_BELOW_SPEECH_DB = 38.0
GAP_DILATE_S = 0.3


def bed_speech_spans(open_end: float, dilate: float = GAP_DILATE_S) -> list[tuple[float, float]]:
    """spans (seconds) where a teacher speaks or the original plays, widened by `dilate`: the room-tone bed is built from the frames outside them"""
    sp = [(w["start"] - dilate, w["end"] + dilate) for w in allw()]
    sp.append((-1.0, open_end + dilate))
    return sp


def make_loader(final: dict):
    """loader for v2v_mix.assemble: sts / tts -> v2v2/aligned/<tag>/<id>.wav, original -> V2's enhanced original (flagged)"""
    def load(p: V.Piece, d: dict):
        f = final[p.id]
        if f["kind"] in ("sts", "tts"):
            x, _ = sf.read(str(aligned_path(f["tag"], p.id)), dtype="float32")
            return x, f"converted ({f['kind']} {f['tag']})", True
        x, _ = sf.read(str(F.IN_DIR / F.ORIG_CAND / f"{p.id}.wav"), dtype="float32")
        return V.highpass(x, SR), "original (enhanced: beamformed 3 phones + MossFormer2_SE_48K)", False
    return load


def assemble_regen(pieces: list[V.Piece], kept: list[dict], final: dict, out_wav: Path, stems_dir: Path, report_path: Path, *, open_end: float,
                   bed_below_db: float = BED_BELOW_SPEECH_DB, events: list[dict] | None = None, event_gain_db: float = 0.0) -> dict:
    from .. import film_recap_audio as fa
    from . import v2v_mix as M
    dec = {pid: {"variant": f["tag"], "use": f["kind"], "why": f["flag"] or f"regenerated ({f['kind']})"} for pid, f in final.items()}
    return M.assemble(pieces, kept, dec, orig_path=fa.AUDIO / "class_mix.wav", out_wav=out_wav, stems_dir=stems_dir, report_path=report_path, loader=make_loader(final),
                      bed_below_db=bed_below_db, bed_speech_spans=bed_speech_spans(open_end), events=events, event_gain_db=event_gain_db)


# ------------------------------------------------------------------------------------------------ verification of a finished mix (the ASR and the embeddings run on the pod)


def identity_check(pieces: list[V.Piece], final: dict, stems_dir: Path) -> dict:
    """ECAPA embedding of every piece cut (its words) from the final teacher stems, against the voices' own training samples: {id: {own, other, ok}}"""
    import tarfile
    import math as _m
    from scipy import signal as sg
    from . import v2v_gate as G
    from . import v2v_pod as Pd
    refs = F.reference_embeddings()
    import shutil
    d = V2 / "ident"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(exist_ok=True)
    stems = {s: sf.read(str(stems_dir / f"{s}.wav"), dtype="float32", always_2d=True)[0].mean(axis=1) for s in ("larissa", "kadu")}
    for p in pieces:
        if final[p.id]["kind"] == "original" or p.w1 - p.w0 < 0.6:      # a piece under 0.6 s of speech gives no usable embedding
            continue
        x = stems[p.speaker.lower()][int(p.w0 * SR):int(p.w1 * SR)]
        g = _m.gcd(16000, SR)
        y = sg.resample_poly(x, 16000 // g, SR // g).astype(np.float32)
        (d / "wav").mkdir(exist_ok=True)
        sf.write(str(d / "wav" / f"{p.id}.wav"), y, 16000, subtype="PCM_16")
    tar = V2 / "ident_push.tar"
    with tarfile.open(tar, "w") as t:
        for f in sorted((d / "wav").glob("*.wav")):
            t.add(f, arcname=f"ident/{f.name}")
    h = Pd.handle()
    h.put(tar, "/root/ident_push.tar", timeout=1800)
    h.run("cd /root/data && rm -rf ident && tar -xf /root/ident_push.tar", timeout=600, echo=False)
    h.run("cd /root/data && PYTHONPATH=/root/ws/dancecap HM_ROOT=/workspace/hm /opt/hm/env_speech/bin/python -m dancecap.speech.pod.asr_v2v --root /root/data --dirs ident --embed-only 2>&1 | tail -1", timeout=1800)
    F.pull_asr(["ident"], npz=True)
    emb = F.read_embeddings_set("ident")
    out = {}
    for p in pieces:
        if p.id in emb:
            out[p.id] = G.voice_check(p.speaker, emb[p.id], refs)
    return out


def verify_mix(pieces: list[V.Piece], mix_path: Path, final: dict, stems_dir: Path, report_path: Path, *, compare: dict[str, Path] | None = None) -> dict:
    """whole-class ASR of the mix (and of the `compare` mixes, same pipeline), word onsets of 20 words, join steps, clipping, voice identity; writes report_path"""
    from . import v2v_check as C
    mixes = {"regen": mix_path, **(compare or {})}
    wer = F.full_mix_wer(mixes)
    mix, _ = sf.read(str(mix_path), dtype="float32", always_2d=True)
    ws = [w for w in V.transcript_words(V.read_transcript()) if w["t1"] > 0]
    ref = V.phone_on_master("00", F.DURATION)
    n = min(len(ref), len(mix))
    env_ref = C.speech_env_db(ref[:n], SR)
    words = C.pick_clean_words(ws, n=20, ref_env=env_ref, tol_s=0.08)
    on = C.onset_report(mix[:n].mean(axis=1), ref[:n], words)
    lag = C.word_lag_report(mix[:n].mean(axis=1), ref[:n], words)
    vs_t = [abs(r["d_mix_vs_transcript_ms"]) for r in on["rows"] if r["d_mix_vs_transcript_ms"] is not None]
    # the same words in each teacher's own stem (no neighbour's tail in the window; the faint gating blips of a converted piece are clamped 40 dB under its speech like the bed
    # would mask them); words that follow the previous word by at least 0.35 s so the 0.25 s look-back sees no tail
    stem_env = {}
    for sp in ("larissa", "kadu"):
        x, _ = sf.read(str(stems_dir / f"{sp}.wav"), dtype="float32", always_2d=True)
        e = C.speech_env_db(x.mean(axis=1), SR)
        stem_env[sp] = np.maximum(e, np.percentile(e[e > -100], 95) - 40.0)
    words40 = C.pick_clean_words(ws, n=20, min_gap=0.35, ref_env=env_ref, tol_s=0.08)
    srows = []
    for w in words40:
        o = C.onset_near(stem_env[w["speaker"].lower()], w["t0"])
        srows.append({"t0": round(w["t0"], 3), "text": w["text"], "speaker": w["speaker"], "err_ms": None if o is None else round((o - w["t0"]) * 1000, 1)})
    se = [abs(r["err_ms"]) for r in srows if r["err_ms"] is not None]
    steps = C.join_steps([{"id": p.id, "t0": p.t0, "t1": p.t1, "w0": p.w0, "w1": p.w1} for p in pieces if final[p.id]["kind"] != "original"], mix)
    ident = identity_check(pieces, final, stems_dir)
    bad_id = {k: v for k, v in ident.items() if not v["ok"]}
    rep = {"wer": wer, "onsets": {"words": len(words), "measured": on["measured"], "within_80ms_vs_phone": on["within_80ms"], "max_ms_vs_phone": on["max_abs_ms"],
                                  "median_ms_vs_phone": on["median_abs_ms"], "vs_transcript_within_80": sum(1 for x in vs_t if x <= 80), "vs_transcript_n": len(vs_t),
                                  "vs_transcript_median_ms": float(np.median(vs_t)) if vs_t else None, "rows": on["rows"]},
           "stem_onsets": {"words": len(words40), "measured": len(se), "within_80ms": sum(1 for x in se if x <= 80), "median_ms": float(np.median(se)) if se else None,
                           "max_ms": max(se, default=None), "rows": srows},
           "word_lag": {k: lag[k] for k in ("measured", "max_abs_ms", "median_abs_ms", "within_80ms")}, "join_steps": {k: steps[k] for k in ("joins", "max_abs_speech_step_db", "over_3db", "max_abs_floor_step_db")},
           "clipping": C.clipping(mix), "identity": {"checked": len(ident), "not_ok": bad_id, "min_margin": min((v["own"] - v["other"] for v in ident.values()), default=None)}}
    report_path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


# ------------------------------------------------------------------------------------------------ the extra pieces of this version

OPENING_CUT_S = 8.45            # the original keeps 0 - 8.45 s (the owner's jokes, the laugh); "we talked to" after it is regenerated (piece o3)
OPENING_REGEN = {"o3": "o16"}   # opening piece -> accepted speech-to-speech try tag


def extra_pieces() -> list[V.Piece]:
    """o3 (the opening's last words, speech-to-speech accepted) and g1 (Kadu's 0.26 s 'good' at 358.1 s that V2 left original: text-to-speech)"""
    o3 = [p for p in opening_pieces() if p.id == "o3"][0]
    g1 = V.Piece(id="g1", speaker="Kadu", t0=358.0, t1=358.6, w0=358.136, w1=358.396, n_words=1, text="good")
    return [o3, g1]


def all_pieces() -> tuple[list[V.Piece], list[dict]]:
    """the 81 sentence pieces + o3 + g1, by time; and the stretches that stay the original recording (the opening up to OPENING_CUT_S)"""
    pieces, kept, _ = F.read_segments()
    allp = sorted(pieces + extra_pieces(), key=lambda p: p.t0)
    kept = [{"t0": 0.0, "t1": OPENING_CUT_S, "speaker": "both", "why": "opening: the owner's two joke moments and the laugh; the words are still owner-pending (OWNER_QUESTIONS.md), so no regenerated line"}]
    return allp, kept


def realign_final(pieces: list[V.Piece], final: dict) -> dict:
    """re-align the stretches after a pause of every chosen speech-to-speech piece onto the transcript word onsets (`v2v_regen.realign_runs`); a changed piece is
    written to aligned/<tag>_ra/<id>.wav and the final choice points to it. Returns {id: rows}"""
    out = {}
    for p in pieces:
        f = final[p.id]
        if f["kind"] != "sts" or f["tag"].startswith("o"):
            continue
        base = f["tag"].removesuffix("_ra")
        y, _ = sf.read(str(aligned_path(base, p.id)), dtype="float32")
        ref, _ = sf.read(str(F.IN_DIR / "r1" / f"{p.id}.wav"), dtype="float32")
        ws = words_of(p)
        z, rows = R.realign_runs(y, ref[:len(y)], [w["start"] - p.t0 for w in ws], [w["end"] - p.t0 for w in ws])
        if rows:
            o = aligned_path(base + "_ra", p.id)
            o.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(o), np.clip(z, -1, 1), SR, subtype="PCM_16")
            f["tag"] = base + "_ra"
            f["realigned"] = rows
            out[p.id] = rows
        else:
            f["tag"] = base
    return out
