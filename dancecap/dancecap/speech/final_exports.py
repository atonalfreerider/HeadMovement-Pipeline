"""S4 exports of the final transcript: SRT / VTT (speaker-prefixed cues), a readable text file, and captions.json for the film's word-highlight captions.

    transcript_final.json -> transcript_final.srt / .vtt / .txt, captions.json

Caption rules (the film's captions, a later workflow, read captions.json):
  * at most 2 lines of at most 42 characters per caption (an emoji counts 2), at most 6.5 s and no pause of 1 s or more inside a caption;
  * words before the master video starts (master time < 0: only phone 00 was recording) are not captions: the film starts at master 0;
  * fillers (um, uh) stay in the transcript but are not shown; a laugh token is its own caption word (kind "emoji") with its own time;
  * every word keeps its start / end so the film can highlight it while it is spoken; `check` marks a caption that contains a word the owner should
    proofread first (confidence below 0.5 or a decision flagged needs_owner) so that the film can dim or skip it.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import SCHEMA, SPEECH, TAKE
from .final_merge import SENTENCE_END, mmss
from .fuse_post import json_safe

MAX_CHARS = 42
MAX_LINES = 2
MAX_CUE_S = 6.5
MAX_GAP_S = 1.0
LINGER_S = 0.3
MIN_SHOW_S = 1.0
CHECK_CONF = 0.5
PALETTE = {"speaker_larissa": "#F28C6B", "speaker_kadu": "#5BA8E8", "speaker_other": "#B8B8B8"}   # a suggestion; the film chooses its own
KEYS = {"Larissa": "speaker_larissa", "Kadu": "speaker_kadu"}


def clen(s: str) -> int:
    """Display width: an emoji (outside the BMP) counts 2."""
    return sum(2 if ord(c) > 0xFFFF else 1 for c in s)


def display_words(words: list[dict], cap_first: bool = True) -> list[dict]:
    """The words of one utterance as caption tokens: display text (capital at the start of the sentence, punctuation), times, kind; fillers dropped.
    cap_first False = the utterance continues a sentence the previous one left open (no capital on its first word)."""
    out: list[dict] = []
    cap = cap_first
    for w in words:
        if w["kind"] == "event":
            out.append({"text": w["text"], "start": w["start"], "end": w["end"], "kind": "emoji", "conf": w["conf"], "flags": w.get("flags", []), "event": w.get("event"), "joke": w.get("joke")})
            continue
        if w["kind"] == "filler":
            continue
        t = (w["text"][:1].upper() + w["text"][1:] if cap else w["text"]) + w["punct"]
        cap = w["punct"] in SENTENCE_END
        out.append({"text": t, "start": w["start"], "end": w["end"], "kind": "word", "conf": w["conf"], "flags": w.get("flags", [])})
    return out


def _join(ws: list[dict]) -> str:
    return " ".join(w["text"] for w in ws)


def split_lines(ws: list[dict]) -> list[list[dict]] | None:
    """The words of one caption as 1-2 lines of at most MAX_CHARS, most even split; None when it does not fit."""
    if clen(_join(ws)) <= MAX_CHARS:
        return [ws]
    best, score = None, None
    for k in range(1, len(ws)):
        a, b = ws[:k], ws[k:]
        la, lb = clen(_join(a)), clen(_join(b))
        if la <= MAX_CHARS and lb <= MAX_CHARS:
            s = abs(la - lb) - (4 if a[-1]["text"][-1:] in ",.;:?!" else 0)
            if score is None or s < score:
                best, score = [a, b], s
    return best


def split_cues(toks: list[dict]) -> list[list[dict]]:
    """Greedy captions: as many words as fit in MAX_LINES x MAX_CHARS, MAX_CUE_S and without a pause of MAX_GAP_S; when the sentence goes on, the
    caption ends at the last punctuation or the widest pause of its second half, and a one-word orphan is avoided."""
    cues: list[list[dict]] = []
    i, n = 0, len(toks)
    while i < n:
        j = i + 1
        while j < n:
            cand = toks[i: j + 1]
            gap = cand[-1]["start"] - cand[-2]["end"] if cand[-1]["kind"] == "word" and cand[-2]["kind"] == "word" else 0.0
            if split_lines(cand) is None or cand[-1]["end"] - cand[0]["start"] > MAX_CUE_S or gap >= MAX_GAP_S:
                break
            j += 1
        if j < n:
            lo = i + max(2, (j - i) // 2)
            marks = [k for k in range(lo, j) if toks[k - 1]["text"][-1:] in ",.;:?!"]
            if marks:
                j = marks[-1]
            else:
                gaps = [(toks[k]["start"] - toks[k - 1]["end"], k) for k in range(lo, j) if toks[k]["kind"] == "word"]
                if gaps and max(gaps)[0] >= 0.25:
                    j = max(gaps)[1]
            if n - j == 1 and j - i > 3 and toks[j]["kind"] == "word":      # do not leave a one-word orphan
                j -= 1
        while j < n and toks[j]["kind"] == "emoji":                         # a laugh token stays with the words it follows
            j += 1
        cues.append(toks[i:j])
        i = j
    return cues


def build_captions(final: dict) -> dict:
    caps: list[dict] = []
    for u in final["utterances"]:
        toks = [t for t in display_words(u["words"], u.get("cap_first", True)) if t["start"] >= 0.0]
        if not toks or all(t["kind"] == "emoji" for t in toks):
            continue
        for cue in split_cues(toks):
            lines = split_lines(cue) or [cue]
            first, last = cue[0], cue[-1]
            flags = sorted({f for t in cue for f in t["flags"]})
            confs = [t["conf"] for t in cue if t["kind"] == "word"]
            caps.append({
                "id": f"c{len(caps):03d}", "utterance": u["id"], "turn": u["turn"], "speaker": u["speaker"], "speaker_key": KEYS.get(u["speaker"], "speaker_other"),
                "start": round(first["start"], 3), "end": round(max(t["end"] for t in cue), 3), "lines": [_join(l) for l in lines], "text": "\n".join(_join(l) for l in lines),
                "words": [{"text": t["text"], "start": round(t["start"], 3), "end": round(t["end"], 3), "kind": t["kind"], "line": next(k for k, l in enumerate(lines) if t in l),
                           "conf": round(t["conf"], 3)} for t in cue],
                "conf_min": round(min(confs), 3) if confs else None,
                "check": bool(any((t["conf"] < CHECK_CONF) or ("needs_owner" in t["flags"]) for t in cue)),
                "flags": [f for f in flags if f in ("joke", "needs_owner", "unclear", "owner_fact", "emoji", "name", "speaker_override", "speaker_unresolved")],
            })
    caps.sort(key=lambda c: (c["start"], c["end"]))
    for k, c in enumerate(caps):
        c["id"] = f"c{k:03d}"
    # on-screen interval: linger after the last word, at least MIN_SHOW_S, never into the next caption of the same speaker
    for k, c in enumerate(caps):
        end = max(c["end"] + LINGER_S, c["start"] + MIN_SHOW_S)
        nxt = next((d for d in caps[k + 1:] if d["start"] > c["start"] + 0.2), None)
        if nxt is not None and nxt["speaker"] == c["speaker"]:
            end = min(end, max(nxt["start"] - 0.03, c["end"]))
        c["show_start"], c["show_end"] = c["start"], round(max(end, c["end"]), 3)
    doc = {
        "schema": SCHEMA + "/captions", "take": TAKE, "clock": "master = 01-Recap-Tina video time, seconds (the film starts at 0)",
        "rules": {"max_lines": MAX_LINES, "max_chars_per_line": MAX_CHARS, "emoji_width": 2, "max_caption_s": MAX_CUE_S, "fillers": "not shown", "before_master_start": "not shown",
                  "check": "the caption holds a word with confidence < %.1f or a span the owner should proofread: dim or skip it" % CHECK_CONF},
        "speakers": [{"key": k, "name": n, "color_hex_suggestion": PALETTE[k]} for n, k in KEYS.items()] + [{"key": "speaker_other", "name": "other", "color_hex_suggestion": PALETTE["speaker_other"]}],
        "naming_note": "speaker names rest on an acoustic prior (naming confidence %s), not on addressing" % ((final.get("naming") or {}).get("naming_confidence")),
        "captions": caps,
        "stats": {"captions": len(caps), "check": sum(1 for c in caps if c["check"]), "emoji": sum(1 for c in caps for w in c["words"] if w["kind"] == "emoji"),
                  "words": sum(1 for c in caps for w in c["words"] if w["kind"] == "word"), "max_lines_used": max((len(c["lines"]) for c in caps), default=0),
                  "max_line_chars": max((clen(l) for c in caps for l in c["lines"]), default=0), "overlaps": sum(1 for a, b in zip(caps, caps[1:]) if b["start"] < a["show_end"])},
    }
    return doc


# ------------------------------------------------------------------------------------------------ subtitles


def _ts(t: float, sep: str) -> str:
    t = max(t, 0.0)
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def write_srt(caps: list[dict], path: Path) -> Path:
    out = []
    for k, c in enumerate(caps, 1):
        lines = list(c["lines"])
        lines[0] = f"{c['speaker']}: {lines[0]}"
        out.append(f"{k}\n{_ts(c['show_start'], ',')} --> {_ts(c['show_end'], ',')}\n" + "\n".join(lines) + "\n")
    Path(path).write_text("\n".join(out), encoding="utf-8", newline="\n")
    return Path(path)


def write_vtt(caps: list[dict], path: Path) -> Path:
    out = ["WEBVTT", "", "NOTE master clock = 01-Recap-Tina video time; captions of the zouk class recap (final transcript, not proofread by a human)", ""]
    for c in caps:
        lines = list(c["lines"])
        lines[0] = f"{c['speaker']}: {lines[0]}"
        out.append(f"{c['id']}\n{_ts(c['show_start'], '.')} --> {_ts(c['show_end'], '.')}\n" + "\n".join(lines) + "\n")
    Path(path).write_text("\n".join(out), encoding="utf-8", newline="\n")
    return Path(path)


def write_txt(final: dict, path: Path, jokes: dict | None = None) -> Path:
    """The readable transcript: one line per sentence with its start time and speaker; *word* = confidence below 0.6; the proofread-first list at the end."""
    lines = [
        "Zouk class recap: final transcript (adjudicated; NOT proofread by a human)",
        "Master clock = 01-Recap-Tina video time [m:ss.s]; a negative time is before the master video starts (only phone 00 was recording).",
        "*word* = confidence below 0.6 (the vote or the editor was unsure). The laughing-face emoji marks a joke / laugh moment: see 'Jokes' at the end.",
        "Speaker names (Larissa, Kadu) rest on an acoustic prior (naming confidence %s), not on how the teachers address each other." % ((final.get("naming") or {}).get("naming_confidence")),
        "",
    ]
    from .final_merge import render
    for u in final["utterances"]:
        ws = u["words"]
        who = u["speaker"] if not final_turn_interjection(final, u["turn"]) else f"{u['speaker']} (second voice)"
        doubt = " (speaker not settled)" if any("speaker_unresolved" in (w.get("flags") or []) for w in ws) else ""
        lines.append(f"[{mmss(u['start'])}] {who}{doubt}: " + render([_as_full(w) for w in ws], mark_below=0.6, cap_first=u.get("cap_first", True)))
    lines += ["", "Jokes and laughter tokens:"]
    for e in final["events"]:
        lines.append(f"  {e['text']} at {mmss(e['start'])}-{mmss(e['end'])} ({e.get('joke') or e.get('event')}): {e.get('evidence') or ''}".rstrip())
    if jokes:
        for c in jokes.get("laugh_candidates", []):
            lines.append(f"  candidate (no emoji in the text) {mmss(c['t0'])}-{mmss(c['t1'])}: {c['why']}")
    lines += ["", "Proofread first (spans the editors could not settle; listen at the master time):"]
    adj = final["adjudication"]["spans"]
    for s in adj:
        if s["needs_owner"]:
            lines.append(f"  {s['id']} {mmss(s['start'])}-{mmss(s['end'])} -> {s['text']!r} (confidence {s['confidence']}): {s.get('reason') or ''}")
    lines += ["", "Other unclear spans (confidence below 0.6 or flagged unclear):"]
    for s in adj:
        if not s["needs_owner"] and ("unclear" in s["flags"] or s["decision"] == "uncertain"):
            lines.append(f"  {s['id']} {mmss(s['start'])}-{mmss(s['end'])} -> {s['text']!r} (confidence {s['confidence']})")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return Path(path)


def final_turn_interjection(final: dict, turn_id: str) -> bool:
    t = next((x for x in final["turns"] if x["id"] == turn_id), None)
    return bool(t and t.get("interjection"))


def _as_full(w: dict) -> dict:
    d = dict(w)
    d.setdefault("conf", 1.0)
    return d


def write_all(final: dict, out_dir: Path = SPEECH, jokes: dict | None = None) -> dict:
    cap_doc = build_captions(final)
    (Path(out_dir) / "captions.json").write_text(json.dumps(json_safe(cap_doc), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8", newline="\n")
    write_srt(cap_doc["captions"], Path(out_dir) / "transcript_final.srt")
    write_vtt(cap_doc["captions"], Path(out_dir) / "transcript_final.vtt")
    write_txt(final, Path(out_dir) / "transcript_final.txt", jokes)
    return cap_doc
