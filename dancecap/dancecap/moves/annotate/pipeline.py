"""Narration pass -> move timeline: transcript + player log -> move events -> labels.json segments; merging.

    process(ndir, ctx, lex, settings)       narration/<id>/{audio.wav, log.json} -> transcript.json + result.json
    remap(ndir, ctx, lex, settings)         new lag / snap / judgment settings without transcribing again
    merge_segments(existing, new)           apply a narration to the working timeline

Segments (labels.json contract, provenance 'narration' until the user confirms them -> 'human'):
  t0       snapped start (timemap.snap_start) of the event time = label time shown `lag` seconds (wall clock) before
           the move was named; t1 = the next narrated segment's start, or the end of the stretch of video the
           narrator watched (a seek ends it), whichever is first. A start colliding with the previous one moves to
           the next grid eighth before the event, else the event time itself; if none is free the later phrase
           replaces the earlier (a correction), same-move repeats are merged.
  confidence = match confidence (score x recogniser probability) x timing score (tap 1.0, proposal 0.9, grid 0.85,
           eighth 0.75, unsnapped 0.6); null for unknown phrases.
  extra keys (MOVES_RESEARCH L3): anchor_t (label time the word was spoken at), anchor_word, lag_s, b0 / b1 (grid
           eighth indices of t0 / t1), asr {engine, model, word_prob}, narration (id), match {status, method, score,
           phrase, candidates}, snap; unknown phrases: move null + unresolved '<phrase>' + candidates - they are
           saved under labels.json 'unresolved' (not 'segments') until the user picks or proposes a move.
Merge policy (merge_segments): placeholder segments are dropped; human segments are kept and the narration is
  clipped around them; other existing segments (narration / auto / unresolved) are trimmed where the new segments lie.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..narration import Lexicon, map_utterances, summary
from . import asr
from .timemap import TIMING_SCORE, Grid, TimeMap, snap_start

VERSION = 1
MIN_LEN = 0.25


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class NarrationSettings:
    lag: float = 0.4          # seconds (wall clock) between seeing a move start and naming it
    snap: str = "half"        # measure | half | beat | eighth | none
    tap_lag: float = 0.15     # reaction time of an 'M' tap
    min_len: float = MIN_LEN
    typesafe: bool = False    # ask TypeSafe about ambiguous phrases (sends only the words)
    engine: str = "auto"      # auto (Whisper large-v3; never SAPI) | faster-whisper[:large-v3-turbo] | sapi
    annotator: str | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "NarrationSettings":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k) and v is not None:
                cur = getattr(s, k)
                try:
                    setattr(s, k, type(cur)(v) if isinstance(cur, (int, float)) and not isinstance(cur, bool)
                            else (bool(v) if isinstance(cur, bool) else v))
                except (TypeError, ValueError):
                    pass
        s.lag = min(3.0, max(0.0, s.lag))
        return s


@dataclass
class TimelineContext:
    """What the pipeline needs from the session: clocks, beat grid, proposal boundaries."""
    to_label: Callable[[str, float], float] = lambda cam, v: float(v)
    slope: Callable[[str], float] = lambda cam: 1.0
    grid: Grid | None = None
    boundaries: list[float] = field(default_factory=list)
    take: str | None = None


# ---------------------------------------------------------------------------------------------------------------

def build_segments(events: list[dict], tm: TimeMap, ctx: TimelineContext, settings: NarrationSettings,
                   transcript: dict | None = None, narration_id: str | None = None) -> list[dict]:
    spans = tm.spans()
    taps = [tm.label_time(a - settings.tap_lag) for a in tm.taps()]
    grid = ctx.grid
    eighth = grid.unit("eighth") if grid is not None and len(grid) > 1 else 0.25
    rows = []
    for e in events:
        if e["kind"] not in ("move", "unknown"):
            continue
        t_ev, a_used = tm.event_time(e["a0"], settings.lag)
        anchor = tm.label_time(e["a0"])
        t0, how = snap_start(t_ev, grid, settings.snap, taps, ctx.boundaries)
        e["t_event"], e["anchor_t"], e["t0"], e["snap"] = round(t_ev, 4), round(anchor, 4), round(t0, 4), how
        rows.append({"ev": e, "t0": t0, "t_event": t_ev, "anchor": anchor, "how": how,
                     "span": tm.span_of(anchor, spans) or tm.span_of(t_ev, spans)})
    rows.sort(key=lambda r: (r["t0"], r["ev"]["a0"]))
    kept: list[dict] = []
    for r in rows:
        if kept and r["t0"] < kept[-1]["t0"] + settings.min_len:
            p = kept[-1]
            if r["ev"].get("move") and r["ev"].get("move") == p["ev"].get("move"):
                r["ev"]["merged_into_previous"] = True
                continue
            lo = p["t0"] + settings.min_len
            alt = None
            if grid is not None and len(grid):
                c = grid.t[(grid.t >= lo) & (grid.t <= r["t_event"] + 0.2)]
                alt = float(c[-1]) if len(c) else None
            if alt is not None:
                r["t0"], r["how"] = alt, "grid-shifted"
            elif r["t_event"] >= lo:
                r["t0"], r["how"] = r["t_event"], "none"
            else:
                p["ev"]["superseded"] = True
                kept[-1] = r
                continue
        kept.append(r)
    segs = []
    for i, r in enumerate(kept):
        e = r["ev"]
        nxt = kept[i + 1]["t0"] if i + 1 < len(kept) else None
        span_end = r["span"][1] if r["span"] else None
        ends = [x for x in (nxt, span_end) if x is not None and x > r["t0"]]
        t1 = min(ends) if ends else r["t0"] + 4 * eighth
        if t1 - r["t0"] < settings.min_len:
            e["dropped"] = "too short"
            continue
        timing = TIMING_SCORE.get(r["how"], 0.6)
        conf = None if e.get("confidence") is None else round(float(e["confidence"]) * timing, 3)
        seg = {"id": f"nar-{uuid.uuid4().hex[:10]}", "t0": round(r["t0"], 6), "t1": round(t1, 6),
               "move": e.get("move"), "node": None, "tier": "couple", "provenance": "narration",
               "confidence": conf, "annotator": settings.annotator, "transcript": e.get("transcript"),
               "notes": None, "created": now_iso(),
               "anchor_t": round(r["anchor"], 4), "anchor_word": e.get("phrase"), "lag_s": settings.lag,
               "asr": {"engine": (transcript or {}).get("engine"), "model": (transcript or {}).get("model"),
                       "word_prob": e.get("word_prob")},
               "narration": narration_id, "snap": r["how"],
               "match": {"status": e.get("status"), "method": e.get("method"), "score": e.get("score"),
                         "phrase": e.get("phrase"), "candidates": e.get("candidates") or []}}
        if e.get("alternative"):
            seg["match"]["alternative"] = e["alternative"]
        if e.get("judgment"):
            seg["match"]["judgment"] = e["judgment"]
        if grid is not None and len(grid):
            seg["b0"], seg["b1"] = grid.eighth_index(seg["t0"]), grid.eighth_index(seg["t1"] - 1e-6)
        if seg["move"] is None:
            seg["unresolved"] = e.get("unresolved_alias") or e.get("phrase")
            seg["candidates"] = e.get("candidates") or []
        e["segment"] = seg["id"]
        segs.append(seg)
    return segs


def map_and_build(transcript: dict, log_doc: dict, ctx: TimelineContext, lex: Lexicon,
                  settings: NarrationSettings, narration_id: str | None = None, client=None) -> dict:
    entries = log_doc.get("entries") or []
    tm = TimeMap(entries, ctx.to_label, ctx.slope)
    events = map_utterances(transcript.get("utterances") or [], lex, judge=settings.typesafe, client=client)
    segs = build_segments(events, tm, ctx, settings, transcript, narration_id)
    words = []
    for w in transcript.get("words") or []:
        words.append({"word": w["word"], "a": w["start"], "t": round(tm.label_time(float(w["start"])), 4),
                      "prob": w.get("prob")})
    for e in events:
        e.setdefault("anchor_t", round(tm.label_time(e["a0"]), 4))
    warnings = list(transcript.get("warnings") or [])
    if tm.empty:
        warnings.append("the player log is empty: times are recording seconds, not video times")
    asr_info = {k: transcript.get(k) for k in ("engine", "model", "device", "gpu", "language", "seconds")
                if transcript.get(k) is not None}
    why = asr.retranscribe_reason(transcript)
    if why:  # the UI marks the narration 're-transcribe recommended' (turbo / CPU fallback / English only / SAPI)
        asr_info["retranscribe"] = why
    return {"version": VERSION, "kind": "narration", "id": narration_id, "take": ctx.take,
            "engine": transcript.get("engine"), "model": transcript.get("model"), "asr": asr_info,
            "settings": asdict(settings), "spans": tm.spans(), "taps": [round(tm.label_time(a), 4) for a in tm.taps()],
            "events": events, "segments": segs, "words": words, "summary": summary(events),
            "warnings": warnings, "created": now_iso()}


def transcript_tag(tr: dict | None) -> str:
    """File tag of a transcript's recogniser setup: 'sapi', 'import', 'faster-whisper-large-v3' (GPU, en+pt),
    'faster-whisper-large-v3-turbo', 'faster-whisper-large-v3-turbo-cpu-en' (CPU fallback, English pass only)."""
    tr = tr or {}
    eng = str(tr.get("engine") or "previous")
    parts = [eng]
    if eng == "faster-whisper":
        parts.append(str(tr.get("model") or "unknown"))
        if tr.get("gpu") == "cpu" or tr.get("device") == "cpu":
            parts.append("cpu")
        lang = str(tr.get("language") or "")
        if lang and lang != "en+pt":
            parts.append(lang.replace("+", "-"))
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(parts)).strip("-.") or "previous"


def _stamp(tr: dict, path: Path) -> str:
    c = str(tr.get("created") or "")
    m = re.match(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)", c)
    if m:
        return "{}{}{}-{}{}{}".format(*m.groups())
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y%m%d-%H%M%S")
    except OSError:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def keep_previous(ndir: Path, new_engine: str | None = None) -> list[Path]:
    """Before transcript.json is replaced (re-transcription, transcript import): copy transcript.json + result.json
    to transcript.<tag>.json / result.<tag>.json, tag = engine + model + CPU / single-language mode (transcript_tag),
    e.g. transcript.faster-whisper-large-v3.json. A backup of that name that holds something else is never
    overwritten: the pair then goes to transcript.<tag>.<created>.json (UTC). Returns the new files ([] when an
    identical copy is already kept). new_engine is unused (kept for callers): every overwrite keeps a copy."""
    ndir = Path(ndir)
    tp = ndir / "transcript.json"
    if not tp.exists():
        return []
    try:
        old = json.loads(tp.read_text(encoding="utf-8"))
        old = old if isinstance(old, dict) else {}
    except (OSError, ValueError):
        old = {}
    tag, stamp = transcript_tag(old), _stamp(old, tp)
    srcs = [(name, ndir / f"{name}.json") for name in ("transcript", "result") if (ndir / f"{name}.json").exists()]
    data = {name: src.read_bytes() for name, src in srcs}
    for suffix in [tag, f"{tag}.{stamp}"] + [f"{tag}.{stamp}-{i}" for i in range(2, 1000)]:
        dsts = [(name, ndir / f"{name}.{suffix}.json") for name, _ in srcs]
        if all(not d.exists() or d.read_bytes() == data[name] for name, d in dsts):
            kept = []
            for name, d in dsts:
                if not d.exists():
                    d.write_bytes(data[name])
                    kept.append(d)
            return kept
    raise OSError(f"no free backup name for {tp} ({tag})")


def process(ndir: Path, ctx: TimelineContext, lex: Lexicon, settings: NarrationSettings,
            retranscribe: bool = False, log=print, client=None) -> dict:
    """narration/<id>/: audio.(wav|webm|ogg) + log.json -> audio16k.wav, transcript.json, result.json. Re-transcribing
    keeps the previous files as transcript.<tag>.json / result.<tag>.json (keep_previous: tag = engine + model +
    mode, e.g. faster-whisper-large-v3; a timestamped name when that one is taken)."""
    ndir = Path(ndir)
    log_doc = json.loads((ndir / "log.json").read_text(encoding="utf-8"))
    tp = ndir / "transcript.json"
    if retranscribe or not tp.exists():
        src = next((p for p in sorted(ndir.glob("audio.*")) if p.suffix.lower() in
                    (".wav", ".webm", ".ogg", ".oga", ".mp4", ".m4a", ".mp3", ".flac")), None)
        if src is None:
            raise asr.ASRError(f"no audio in {ndir}")
        wav = asr.to_wav16k(src, ndir / "audio16k.wav")
        tr = asr.transcribe(wav, settings.engine, lex, log=log)
        for k in keep_previous(ndir, tr.get("engine")):
            log(f"[annotate] previous {k.name} kept")
        tp.write_text(json.dumps(tr, indent=1, ensure_ascii=False), encoding="utf-8")
    else:
        tr = json.loads(tp.read_text(encoding="utf-8"))
    res = map_and_build(tr, log_doc, ctx, lex, settings, ndir.name, client)
    (ndir / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


def remap(ndir: Path, ctx: TimelineContext, lex: Lexicon, settings: NarrationSettings, client=None) -> dict:
    return process(ndir, ctx, lex, settings, retranscribe=False, client=client)


# ---------------------------------------------------------------------------------------------------------------
# merging into the working timeline
# ---------------------------------------------------------------------------------------------------------------

def _subtract(seg: dict, cuts: list[tuple[float, float]], min_len: float) -> list[dict]:
    pieces = [(float(seg["t0"]), float(seg["t1"]))]
    for c0, c1 in cuts:
        nxt = []
        for a, b in pieces:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        pieces = nxt
    out = []
    for k, (a, b) in enumerate(p for p in pieces if p[1] - p[0] >= min_len):
        d = dict(seg)
        d["t0"], d["t1"] = round(a, 6), round(b, 6)
        if k > 0:
            d["id"] = f"{seg.get('id', 'seg')}-{k + 1}"
        out.append(d)
    return out


def merge_segments(existing: list[dict], new: list[dict], min_len: float = MIN_LEN) -> list[dict]:
    """Existing working segments + a narration's segments -> one non-overlapping list (see module doc)."""
    keep = [s for s in existing if s.get("provenance") != "placeholder"]
    human = [(float(s["t0"]), float(s["t1"])) for s in keep if s.get("provenance") == "human"]
    added = []
    for s in new:
        added += _subtract(s, sorted(human), min_len)
    cuts = sorted((float(s["t0"]), float(s["t1"])) for s in added)
    out = []
    for s in keep:
        out += [s] if s.get("provenance") == "human" else _subtract(s, cuts, min_len)
    out += added
    return sorted(out, key=lambda s: (float(s["t0"]), float(s["t1"])))


def overlaps(segs: list[dict], eps: float = 1e-6) -> list[tuple[str, str]]:
    s = sorted(segs, key=lambda x: (float(x["t0"]), float(x["t1"])))
    return [(a.get("id"), b.get("id")) for a, b in zip(s, s[1:]) if float(b["t0"]) < float(a["t1"]) - eps]
