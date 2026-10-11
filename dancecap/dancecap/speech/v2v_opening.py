"""The class recap's opening (0 - 8.5 s) regenerated in the two cloned voices from the OWNER'S words (agent O1, 2026-10-11).

V3's soundtrack (`class_mix_regen.wav`) regenerated every line except the opening: its words were unknown and speech-to-speech on the laughing, overlapping phones
came back garbled. The owner has now given the words (speech/owner_opening.json -> transcript_final.json: "Hello! I'm Kadu and this is Larissa!" (Larissa),
"Yeah, you see my hair?" (Kadu), "Yeah, it's very red." (Larissa)), so every line can be made by text-to-speech in the teacher's cloned voice, fitted to the word times
of the transcript, or by speech-to-speech when the ASR gate passes. The original room's laughter (no speech under it) stays as an EVENT.

    pieces()            the opening's lines as `v2v.Piece`s (from the owner's words of transcript_final.json, split at pauses of >= 0.6 s)
    tts_batch / sts_batch   ElevenLabs requests (key read at run time from the owner's Desktop file, never printed or stored; credits in v2v2/spend_ledger_o1.jsonl)
    asr_local()         faster-whisper large-v3 on the local GPU (asr_local.py in the .venv-asr interpreter) for the tries
    choose()            the best candidate per piece by ASR WER, onset error, fit
    laugh events        `LAUGH_EVENTS` = stretches of the ORIGINAL class mix without speech (cut as events, level of the old opening)

Audio of the tries: work/<take>/speech/v2v2/aligned/<tag>/<id>.wav (tags n<k> speech-to-speech, nt<k> text-to-speech), table v2v2/opening_o1.json. Never commit work/.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import SPEECH
from . import v2v as V
from . import v2v_full as F
from . import v2v_regen as R
from . import v2v_regen_run as RR

SR = V.SR
V2 = F.V2
TABLE = V2 / "opening_o1.json"
LEDGER = V2 / "spend_ledger_o1.jsonl"
CAP_CREDITS = 15000
GAP_SPLIT_S = 0.6                # a pause this long between two words of one teacher starts a new piece (the laugh)
PAD_BEFORE_S = 0.30
TAIL_S = 0.40
PY_ASR = Path(__file__).resolve().parents[3] / ".venv-asr" / "Scripts" / "python.exe"
SCRIPT_ASR = Path(__file__).resolve().parent / "asr_local.py"

# stretches of the ORIGINAL class mix that are laughter without speech (master seconds): cut as events (fades 0.08 s)
LAUGH_EVENTS = [{"id": "laugh1", "t0": 2.20, "t1": 3.30, "why": "the room laughs after \"I'm Kadu\" (burst at 2.15 s, speech again at 3.28 s); no word is spoken in it"}]

_LOCK = threading.Lock()


def use_ledger() -> None:
    R.LEDGER = LEDGER
    V.LEDGER = LEDGER


def credits_used() -> int:
    return V.ledger_credits(LEDGER)


# ------------------------------------------------------------------------------------------------ the pieces


def owner_words() -> list[dict]:
    """[{text, punct, t0, t1, speaker}]: the owner's words of the opening (flag owner_opening, kind word) in time order, from transcript_final.json"""
    out = []
    for u in V.read_transcript()["utterances"]:
        for w in u["words"]:
            if w.get("kind") == "word" and "owner_opening" in (w.get("flags") or []):
                out.append({"text": w["text"], "punct": w.get("punct") or "", "t0": float(w["start"]), "t1": float(w["end"]), "speaker": u["speaker"]})
    out.sort(key=lambda w: w["t0"])
    return out


def display(ws: list[dict]) -> list[dict]:
    """the words with display case (a capital at the start, after . ? !, and for I'm / names): the text sent to text-to-speech"""
    out, cap = [], True
    for w in ws:
        t = w["text"]
        if cap or t.lower() in ("kadu", "larissa") or t.lower().startswith("i'"):
            t = t[:1].upper() + t[1:]
        out.append(dict(w, text=t))
        cap = w["punct"] in (".", "?", "!")
    return out


def group(ws: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for w in ws:
        if groups and groups[-1][-1]["speaker"] == w["speaker"] and w["t0"] - groups[-1][-1]["t1"] < GAP_SPLIT_S:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def pieces() -> tuple[list[V.Piece], dict[str, list[dict]]]:
    """(pieces n0.., {id: its display words}); t0 = first word - 0.3 s, t1 = the end of the audio the piece may use (last word + 0.4 s, never into the next piece, 40 ms kept)"""
    gs = group(display(owner_words()))
    ps, wd = [], {}
    for i, g in enumerate(gs):
        w0, w1 = g[0]["t0"], g[-1]["t1"]
        prev_end = gs[i - 1][-1]["t1"] if i else 0.0
        nxt = gs[i + 1][0]["t0"] if i + 1 < len(gs) else 8.43
        t0 = max(w0 - PAD_BEFORE_S, (prev_end + w0) / 2 if i else 0.0)
        t1 = min(w1 + TAIL_S, nxt - 0.04)
        text = " ".join(w["text"] + w["punct"] for w in g)
        ps.append(V.Piece(id=f"n{i}", speaker=g[0]["speaker"], t0=round(t0, 3), t1=round(t1, 3), w0=w0, w1=w1, n_words=len(g), text=text))
        wd[f"n{i}"] = g
    return ps, wd


def aligned_path(tag: str, pid: str) -> Path:
    return V2 / "aligned" / tag / f"{pid}.wav"


def load_table() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8")) if TABLE.exists() else {}


def save_table(t: dict) -> None:
    TABLE.write_text(json.dumps(t, indent=1, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ text-to-speech takes

# voice settings of the takes: (stability, similarity, style): more expressive than the class lines (a joke: playful, a little amused); speed from the length ratio
OPEN_TAKES = [
    {"stability": 0.35, "similarity_boost": 0.80, "style": 0.35, "use_speaker_boost": True},
    {"stability": 0.30, "similarity_boost": 0.75, "style": 0.55, "use_speaker_boost": True},
    {"stability": 0.45, "similarity_boost": 0.80, "style": 0.20, "use_speaker_boost": True},
    {"stability": 0.25, "similarity_boost": 0.85, "style": 0.45, "use_speaker_boost": True},
    {"stability": 0.55, "similarity_boost": 0.75, "style": 0.00, "use_speaker_boost": True},
    {"stability": 0.40, "similarity_boost": 0.70, "style": 0.65, "use_speaker_boost": True},
]


def text_variants(p: V.Piece, g: list[dict]) -> dict[str, list[dict]]:
    """text variants of a piece (the words are the owner's; only the punctuation / ellipsis differs): {name: words}"""
    v = {"plain": [dict(w) for w in g]}
    last = g[-1]
    if last["punct"] == "":
        alt = [dict(w) for w in g]
        alt[-1]["punct"] = "..."
        v["trail"] = alt
    return v


def tts_take(session, p: V.Piece, g: list[dict], take: int, *, variant: str = "plain", speed: float = 1.0, ctx: tuple[str | None, str | None] = (None, None),
             budget: int = CAP_CREDITS, avail_end: float | None = None) -> dict:
    """one text-to-speech take of piece p fitted onto the transcript clock -> aligned/nt<take>/<id>.wav; returns the table entry (the candidate)"""
    use_ledger()
    vs = text_variants(p, g)
    variant = variant if variant in vs else "plain"
    ws = vs[variant]
    text, spans = R.tts_text_spans(ws)
    if credits_used() + len(text) > budget:
        raise V.V2VError(f"O1 credit budget {budget} would be exceeded (used {credits_used()})")
    st = dict(OPEN_TAKES[(take - 1) % len(OPEN_TAKES)])
    if abs(speed - 1.0) > 1e-6:
        st["speed"] = round(speed, 3)
    seed = 500 + take * 13 + int(p.id[1:])
    vid = V.voice_ids()[p.speaker]
    raw = V2 / "tts_raw" / f"nt{take}"
    raw.mkdir(parents=True, exist_ok=True)
    key = V.request_hash(text.encode(), vid, R.TTS_MODEL, {"s": st, "seed": seed, "prev": ctx[0], "next": ctx[1]})
    cf_, cj = raw / f"{p.id}.wav", raw / f"{p.id}.json"
    if cf_.exists() and cj.exists() and json.loads(cj.read_text(encoding="utf-8")).get("key") == key:
        meta = json.loads(cj.read_text(encoding="utf-8"))
        x, sr = sf.read(str(cf_), dtype="float32")
        al, info = meta["align"], meta["info"]
    else:
        x, sr, al, info = R.tts_request(session, vid, text, settings=st, previous_text=ctx[0], next_text=ctx[1], seed=seed, purpose=f"o1 nt{take} {p.id}")
        sf.write(str(cf_), x, sr, subtype="PCM_16")
        cj.write_text(json.dumps({"key": key, "align": al, "info": info}), encoding="utf-8")
    aw = R.word_times_from_alignment(al, spans)
    tw = [(w["t0"], w["t1"]) for w in g]
    end = avail_end if avail_end is not None else p.t1
    n_out = int(round((end - p.t0) * SR))
    y, rep = R.fit_take(x, sr, tw, aw, n_out, t_origin=p.t0)
    tag = f"nt{take}"
    f = aligned_path(tag, p.id)
    f.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(f), np.clip(y, -1, 1), SR, subtype="PCM_16")
    runs = R.plan_runs([(a - p.t0, b - p.t0) for a, b in tw], aw)
    s_take = sum(aw[i1 - 1][1] - aw[i0][0] for i0, i1 in runs)
    s_tgt = sum(tw[i1 - 1][1] - tw[i0][0] for i0, i1 in runs)
    return {"tag": tag, "kind": "tts", "take": take, "variant": variant, "speed": speed, "settings": st, "seed": seed, "wer": None, "chars": len(text), "text": text,
            "ratio": round(s_take / max(s_tgt, 1e-3), 3), "rate_min": rep["rate_min"], "rate_max": rep["rate_max"], "onset_pred_max_ms": rep["onset_pred_max_ms"],
            "credits": info["credits"], "take_s": round(len(x) / sr, 3)}


def tts_batch(ps: list[V.Piece], wd: dict, table: dict, takes: list[int], *, speeds: dict[str, float] | None = None, variants: dict[int, str] | None = None,
              threads: int = 3, budget: int = CAP_CREDITS, log=print) -> int:
    """text-to-speech takes `takes` for every piece (the previous / next line of the same teacher as context); fills table[pid] with the candidates"""
    local = threading.local()
    stop = {"why": None}

    def ctx_of(p: V.Piece):
        i = [q.id for q in ps].index(p.id)
        prv = ps[i - 1] if i and ps[i - 1].speaker == p.speaker else None
        nxt = ps[i + 1] if i + 1 < len(ps) and ps[i + 1].speaker == p.speaker else None
        t = lambda q: " ".join(w["text"] + w["punct"] for w in wd[q.id])      # noqa: E731
        return (t(prv) if prv else None), (t(nxt) if nxt else None)

    def work(job):
        p, k = job
        if stop["why"]:
            return None
        if not hasattr(local, "s"):
            local.s = V.http_session()
        for attempt in range(5):
            try:
                return p.id, tts_take(local.s, p, wd[p.id], k, variant=(variants or {}).get(k, "plain"), speed=(speeds or {}).get((p.id, k), (speeds or {}).get(p.id, 1.0)),
                                      ctx=ctx_of(p), budget=budget)
            except V.QuotaError as ex:
                stop["why"] = f"quota: {ex}"
                return None
            except V.V2VError as ex:
                if "429" in str(ex) or "concurrent" in str(ex).lower():
                    time.sleep(3 + 4 * attempt)
                    continue
                if "budget" in str(ex):
                    stop["why"] = str(ex)
                log(f"nt{k} {p.id}: {str(ex)[:150]}")
                return None
        return None

    n = 0
    with cf.ThreadPoolExecutor(threads) as ex:
        for r in ex.map(work, [(p, k) for p in ps for k in takes]):
            if r:
                pid, entry = r
                lst = table.setdefault(pid, [])
                lst[:] = [c for c in lst if c["tag"] != entry["tag"]]
                lst.append(entry)
                n += 1
    if stop["why"]:
        log("STOPPED: " + stop["why"])
    return n


# ------------------------------------------------------------------------------------------------ speech-to-speech tries (the known words let the ASR gate decide)

STS_CAMS = ["00", "01", "02"]
STS_SETS = ["S", "L2", "B"]


def sts_cut(p: V.Piece, cam: str, cache: dict) -> tuple[Path, np.ndarray, float]:
    """the piece's cut of phone `cam` (master clock, only its own speech: 60 ms before the first word to 120 ms after the last), cleaned and levelled as V1/V3 did;
    returns (file, cut, cut start in master seconds)"""
    if cam not in cache:
        cache[cam] = V.phone_on_master(cam, F.DURATION)
    c0, c1 = max(p.t0, p.w0 - 0.06), min(p.w1 + 0.12, p.t1)
    cut = V.cut_master(cache[cam], c0, c1)
    pre, _ = V.prepare_piece(cut)
    y = F.prepare_for_sts(pre)
    f = V2 / "sts_in" / "n" / f"{p.id}.{cam}.wav"
    f.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(f), np.clip(y, -1, 1), SR, subtype="PCM_16")
    return f, cut, c0


def sts_batch(ps: list[V.Piece], table: dict, ks: list[int], *, only: set[str] | None = None, threads: int = 3, budget: int = CAP_CREDITS, log=print) -> int:
    """tries n<k>: phone STS_CAMS[(k-1) % 3], settings STS_SETS[(k-1) // 3 % 3], seed 700 + k"""
    use_ledger()
    vids = V.voice_ids()
    cache: dict = {}
    jobs = []
    for p in ps:
        if only and p.id not in only:
            continue
        for k in ks:
            cam = STS_CAMS[(k - 1) % len(STS_CAMS)]
            f, cut, c0 = sts_cut(p, cam, cache)
            jobs.append((p, k, cam, f, c0))
    local = threading.local()
    stop = {"why": None}

    def work(job):
        p, k, cam, f, c0 = job
        if stop["why"]:
            return None
        if not hasattr(local, "s"):
            local.s = V.http_session()
        tag = f"n{k}"
        skey = STS_SETS[(k - 1) // len(STS_CAMS) % len(STS_SETS)]
        try:
            RR.convert_sts(local.s, vids[p.speaker], f, tag, p.id, model=V.MODEL_DEFAULT, settings_key=skey, seed=700 + k, budget=budget)
        except V.V2VError as e:
            stop["why"] = str(e) if ("budget" in str(e) or isinstance(e, V.QuotaError)) else None
            log(f"{tag} {p.id}: {str(e)[:120]}")
            return None
        rep = F.align_to_source(V2 / "conv" / tag / f"{p.id}.wav", f, aligned_path(tag, p.id))
        y, _ = sf.read(str(aligned_path(tag, p.id)), dtype="float32")        # the file starts at the piece's t0 (the cut started c0 - t0 later)
        sf.write(str(aligned_path(tag, p.id)), np.concatenate([np.zeros(int(round((c0 - p.t0) * SR)), np.float32), y]), SR, subtype="PCM_16")
        return p.id, {"tag": tag, "kind": "sts", "k": k, "cam": cam, "settings": skey, "seed": 700 + k, "wer": None, "lag_ms": rep.get("lag_ms")}

    n = 0
    with cf.ThreadPoolExecutor(threads) as ex:
        for r in ex.map(work, jobs):
            if r:
                pid, entry = r
                lst = table.setdefault(pid, [])
                lst[:] = [c for c in lst if c["tag"] != entry["tag"]]
                lst.append(entry)
                n += 1
    return n


# ------------------------------------------------------------------------------------------------ local ASR of the candidates


def trimmed(f: Path, out: Path, pad_s: float = 0.3, floor_db: float = 45.0) -> Path:
    """a candidate cut to its speech +- pad_s (long digital silence makes the ASR repeat sentences)"""
    x, sr = sf.read(str(f), dtype="float32")
    e = np.abs(x)
    idx = np.nonzero(e > e.max() * 10 ** (-floor_db / 20))[0]
    a, b = (max(int(idx[0]) - int(pad_s * sr), 0), min(int(idx[-1]) + int(pad_s * sr), len(x))) if len(idx) else (0, len(x))
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), x[a:b], sr, subtype="PCM_16")
    return out


def run_asr(items: list[dict], work: Path) -> dict:
    """items [{key, file, t0?, t1?}] -> {key: {text, words, avg_logprob}} through asr_local.py in the .venv-asr interpreter (local GPU)"""
    work.mkdir(parents=True, exist_ok=True)
    inp, outp = work / "asr_in.json", work / "asr_out.json"
    inp.write_text(json.dumps(items), encoding="utf-8")
    r = subprocess.run([str(PY_ASR), str(SCRIPT_ASR), str(inp), str(outp)], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError("asr_local failed: " + (r.stderr or "")[-600:])
    return json.loads(outp.read_text(encoding="utf-8"))


def score(ps: list[V.Piece], wd: dict, table: dict, tags: list[str] | None = None, work: Path | None = None) -> None:
    """ASR of every candidate (all, or those with `tags`) of every piece; fills wer, the heard text, insertions and the pitch statistics"""
    work = work or (V2 / "o1_asr")
    items, meta = [], {}
    for p in ps:
        for c in table.get(p.id, []):
            if tags and c["tag"] not in tags:
                continue
            f = aligned_path(c["tag"], p.id)
            if not f.exists():
                continue
            key = f"{c['tag']}::{p.id}"
            t = trimmed(f, work / "trim" / c["tag"] / f"{p.id}.wav")
            # trimmed start (to map word times back to the piece clock)
            x, sr = sf.read(str(f), dtype="float32")
            idx = np.nonzero(np.abs(x) > np.abs(x).max() * 10 ** (-45 / 20))[0]
            a = max(int(idx[0]) - int(0.3 * sr), 0) if len(idx) else 0
            items.append({"key": key, "file": str(t)})
            meta[key] = (p, c, a / sr)
    res = run_asr(items, work)
    for key, (p, c, a0) in meta.items():
        r = res[key]
        ref = V.norm_words(" ".join(w["text"] for w in wd[p.id]))
        hyp = V.norm_words(r["text"])
        e = V.edit_counts(ref, hyp)
        c["wer"], c["asr_text"], c["ins"] = round(e["wer"], 4), r["text"], e["ins"]
        c["asr_logprob"] = r["avg_logprob"]
        c["f0"] = f0_stats(aligned_path(c["tag"], p.id))
    save_table(table)


def f0_stats(f: Path) -> dict:
    """pitch range of a candidate (librosa pyin on the speech part): median Hz and the 10-90 percentile spread in semitones (a flat reading has a small spread)"""
    import librosa
    x, sr = sf.read(str(f), dtype="float32")
    idx = np.nonzero(np.abs(x) > np.abs(x).max() * 10 ** (-40 / 20))[0]
    if len(idx) < 1000:
        return {"median_hz": None, "spread_st": None}
    x = x[idx[0]:idx[-1]]
    y = librosa.resample(x, orig_sr=sr, target_sr=16000)
    f0, vf, vp = librosa.pyin(y, fmin=70, fmax=500, sr=16000, frame_length=1024, hop_length=160)
    v = f0[vf & (vp > 0.6)]
    if len(v) < 10:
        return {"median_hz": None, "spread_st": None}
    st = 12 * np.log2(v / np.median(v))
    return {"median_hz": round(float(np.median(v)), 1), "spread_st": round(float(np.percentile(st, 90) - np.percentile(st, 10)), 2)}


# ------------------------------------------------------------------------------------------------ choice


def rank(c: dict) -> tuple:
    """clear words first (WER 0, then <= 0.15), then the smallest predicted onset drift of the fit (text-to-speech) / the envelope lag (speech-to-speech), then fewer
    insertions, then the pitch spread (a livelier reading is better for a joke). The ASR word times of a 1-2 s clip are not used: whisper starts the first word at 0."""
    w = c.get("wer") if c.get("wer") is not None else 9.0
    drift = c.get("onset_pred_max_ms") if c.get("kind") == "tts" else abs(c.get("lag_ms") or 0.0)
    spread = (c.get("f0") or {}).get("spread_st") or 0.0
    return (0 if w <= 0.0 else (1 if w <= 0.15 else 2), w, (drift if drift is not None else 999.0) // 30, c.get("ins") or 0, -spread)


def choose(table: dict, ps: list[V.Piece], prefer: dict[str, str] | None = None) -> dict:
    """{piece id: {kind, tag, wer, ...}}: the best candidate; `prefer` = {id: tag} overrides the ranking (listening or ASR judgment recorded in the notes)"""
    out = {}
    for p in ps:
        cs = [c for c in table.get(p.id, []) if c.get("wer") is not None]
        if prefer and p.id in prefer:
            c = next(x for x in cs if x["tag"] == prefer[p.id])
        else:
            c = min(cs, key=rank)
        out[p.id] = {"kind": c["kind"], "tag": c["tag"], "wer": c["wer"], "lag_ms": c.get("lag_ms"), "onset_pred_max_ms": c.get("onset_pred_max_ms"), "flag": f"owner opening ({c['kind']})"}
    return out


# ------------------------------------------------------------------------------------------------ the mix

OPEN_END_S = 8.45
EVENT_GAIN_DB = -8.0            # the gain the original opening had in class_mix_regen.wav (v2v2/regen_mix.json: original_gain_db): the laugh keeps its balance to the speech


def assemble(final_new: dict, out_wav: Path, stems_dir: Path, report_path: Path, *, events: list[dict] | None = None) -> dict:
    """class_mix_regen.wav again, with the opening regenerated too: V3's 83 pieces (v2v2/regen_final.json) + the opening's pieces `final_new`
    ({id: {kind: tts | sts, tag, ...}}), no original stretch, the laugh events (LAUGH_EVENTS) cut from the original class mix, the room-tone bed under everything"""
    ps_old, _kept = RR.all_pieces()
    ps_new, _wd = pieces()
    allp = sorted(ps_old + ps_new, key=lambda p: p.t0)
    final = json.loads((V2 / "regen_final.json").read_text(encoding="utf-8"))
    final.update(final_new)
    return RR.assemble_regen(allp, [], final, out_wav, stems_dir, report_path, open_end=OPEN_END_S, events=LAUGH_EVENTS if events is None else events,
                             event_gain_db=EVENT_GAIN_DB)


# ------------------------------------------------------------------------------------------------ verification (local: ASR on the local GPU; the ECAPA identity check runs on a pod)


def mix_asr(paths: dict[str, Path], work: Path | None = None) -> dict:
    """whole-mix ASR (faster-whisper large-v3, local GPU) of every mix {name: path}: WER against the transcript's teacher words (t1 > 0), overall, in the opening (< 9.67 s) and after it"""
    work = work or (V2 / "o1_asr")
    res = run_asr([{"key": k, "file": str(v)} for k, v in paths.items()], work / "mix")
    ws = [w for w in V.transcript_words(V.read_transcript()) if w["t1"] > 0]
    out = {}
    for nm, r in res.items():
        hyp_all = V.norm_words(" ".join(w for w, s, e in r["words"]))
        c = V.edit_counts(V.norm_words(" ".join(w["text"] for w in ws)), hyp_all)
        ref_b = V.norm_words(" ".join(w["text"] for w in ws if w["t0"] >= 9.67))
        cb = V.edit_counts(ref_b, V.norm_words(" ".join(w for w, s, e in r["words"] if s >= 9.67)))
        ref_o = V.norm_words(" ".join(w["text"] for w in ws if w["t0"] < 9.67))
        co = V.edit_counts(ref_o, V.norm_words(" ".join(w for w, s, e in r["words"] if s < 9.67)))
        out[nm] = {"wer": round(c["wer"], 4), "sub": c["sub"], "ins": c["ins"], "del": c["del"], "n_ref": c["n_ref"], "wer_after_opening": round(cb["wer"], 4),
                   "n_ref_after_opening": cb["n_ref"], "wer_opening": round(co["wer"], 4), "n_ref_opening": co["n_ref"], "opening_text": " ".join(w for w, s, e in r["words"] if s < 9.67)}
    return out


def opening_onsets(stems_dir: Path) -> list[dict]:
    """envelope onset (own teacher's stem, faint gating blips clamped 40 dB under the speech) of the first word of every stretch of the opening, against the transcript"""
    from . import v2v_check as C
    ws = [w for w in V.transcript_words(V.read_transcript()) if 0.0 <= w["t0"] < 9.0]
    stem_env = {}
    for sp in ("larissa", "kadu"):
        x, _ = sf.read(str(stems_dir / f"{sp}.wav"), dtype="float32", always_2d=True)
        e = C.speech_env_db(x.mean(axis=1), SR)
        stem_env[sp] = np.maximum(e, np.percentile(e[e > -100], 95) - 40.0)
    rows = []
    for i, w in enumerate(ws):
        prev = ws[i - 1] if i else None
        if w["kind"] != "word" or (prev is not None and w["t0"] - prev["t1"] < 0.25 and prev["speaker"] == w["speaker"]):
            continue
        o = C.onset_near(stem_env[w["speaker"].lower()], w["t0"])
        rows.append({"t0": round(w["t0"], 3), "text": w["text"], "speaker": w["speaker"], "err_ms": None if o is None else round((o - w["t0"]) * 1000, 1)})
    return rows
