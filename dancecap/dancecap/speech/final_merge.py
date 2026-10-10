"""S4: the FINAL transcript = consensus_v0.json + the adjudicators' decisions + the owner facts (local, pure python, seconds).

    consensus_v0.json (S2: 944 words, vote shares)        disagreements/batch_NN.json + batch_NN.decisions.json (177 disputed spans, one decision each)
      -> every span's decision text replaces the words of its slots (equal words keep their consensus time and share, new words get a time from the raw
         word times of the engines that wrote them, else a share of the span's window), punctuation and case from the decision
      -> speaker per word (the diarization voice of the replaced words; small explicit corrections in final_overrides.json), mini-turns that sit
         between two stretches of the same voice are merged
      -> laughter tokens (U+1F602) at the times in final_overrides.json, after the words of the joke
      -> turns, sentences ("utterances"), per-word and per-turn confidence, flags
      -> transcript_final.json

Nothing here knows the class: everything that belongs to THIS recording (speaker corrections, laugh times, curated chapters and teaching events) is read from
work/<take>/speech/final_overrides.json, which is git-ignored with the rest of the real-person data. The words of the opening jokes are the OWNER's words (no
engine writes 'Kadu'); they carry flags `joke`, `owner_fact` so that no later step mistakes them for something the audio proved.

After an independent audit the file also corrects what the audit found, always with a one-line reason: `span_overrides` (a corrected decision text / confidence / flags
for a span), `speaker_windows` (a label, or only a lower confidence and the flag `speaker_unresolved`), `utterance_breaks` (a sentence split at a time, structure only:
no word changes, the second half gets the id of the first plus a letter). `work/<take>/speech/owner_opening.json`, when it exists, replaces the opening with what
the owner heard (segments with a speaker, the emoji times): see apply_owner_opening.

    python -m dancecap.speech final            # transcript_final.json + exports + chapters + teaching events + jokes + review page + evaluation
"""
from __future__ import annotations

import copy
import datetime as dt
import difflib
import json
import re
from collections import Counter
from pathlib import Path

from . import SCHEMA, SPEECH, TAKE, fuse
from .fuse import ENGINE_TIME_W, FILLERS
from .fuse_post import SENTENCE_END, json_safe

LAUGH = "\U0001F602"
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]+")
OUT_FINAL = SPEECH / "transcript_final.json"
BATCH_DIR = SPEECH / "disagreements"
OVERRIDES = SPEECH / "final_overrides.json"
OWNER_OPENING = SPEECH / "owner_opening.json"
VOICE_OF = {"larissa": 1, "kadu": 0}       # the two teachers (takes/larissa_kadu_recap.toml); any other speaker name is "other" (no voice)
OWNER_CONF = 0.95                           # confidence of a word the owner typed after listening
PAUSE_BREAK = 1.2          # s: a pause this long ends a sentence
MIN_DUR = 0.04
MIN_WORD_S = 0.10          # s a new word gets when the window is tight
SPAN_WORD_FLAGS = {"unclear", "needs_owner", "joke", "owner_fact", "name", "overlap"}   # decision flags that mark every word of the span (the others describe the span as a whole)
I_FORMS = ("i", "i'm", "i'll", "i've", "i'd", "i’m", "i’ll", "i’ve", "i’d")


# ------------------------------------------------------------------------------------------------ inputs


def load_batches(batch_dir: Path = BATCH_DIR) -> tuple[dict, dict, list[dict]]:
    """({span id: span}, {span id: decision}, [batch headers]) of every batch_NN.json / batch_NN.decisions.json. A missing decisions file is an error: a span without
    a decision would silently keep the unadjudicated vote."""
    spans: dict[str, dict] = {}
    decisions: dict[str, dict] = {}
    heads: list[dict] = []
    for f in sorted(Path(batch_dir).glob("batch_[0-9][0-9].json")):
        b = json.loads(f.read_text(encoding="utf-8"))
        heads.append({k: b.get(k) for k in ("batch", "of", "time_range", "n_spans")})
        for s in b["spans"]:
            spans[s["id"]] = s
        df = f.with_name(f.stem + ".decisions.json")
        if not df.exists():
            raise FileNotFoundError(f"no decisions for {f.name}: {df.name}")
        d = json.loads(df.read_text(encoding="utf-8"))
        d = d["decisions"] if isinstance(d, dict) and "decisions" in d else d
        for x in d:
            if x["id"] in decisions:
                raise ValueError("duplicate decision " + x["id"])
            decisions[x["id"]] = x
    missing = set(spans) - set(decisions)
    extra = set(decisions) - set(spans)
    if missing or extra:
        raise ValueError(f"spans without decision {sorted(missing)}; decisions without span {sorted(extra)}")
    return spans, decisions, heads


def load_overrides(path: Path = OVERRIDES, opening_path: Path | None = OWNER_OPENING) -> dict:
    """final_overrides.json (this recording's corrections) plus, when the file exists, owner_opening.json under the key `owner_opening`."""
    base = {"event_times": {}, "speaker_windows": [], "word_edits": [], "chapters": [], "teaching_events": {}, "joke_candidates": [], "notes": [], "span_overrides": {},
            "utterance_breaks": [], "owner_clips": []}
    if Path(path).exists():
        base.update(json.loads(Path(path).read_text(encoding="utf-8")))
    if opening_path is not None and Path(opening_path).exists():
        base["owner_opening"] = json.loads(Path(opening_path).read_text(encoding="utf-8"))
    return base


SPAN_OVERRIDE_FIELDS = {"why", "text", "confidence", "decision", "flags", "add_flags", "drop_flags", "needs_owner", "alternatives", "note"}


def apply_span_overrides(spans: dict, decisions: dict, rules: dict) -> tuple[dict, list[dict]]:
    """final_overrides.json["span_overrides"] = {span id: {why, text?, confidence?, decision?, flags?, add_flags?, drop_flags?, needs_owner?, alternatives?, note?}}: a
    corrected decision for a span (an audit found the editor's text unsupported). `why` is mandatory (one line); a changed text without a `decision` becomes "rewrite".
    The editor's own reason stays in `editor_reason`, the old values in `override.was`. Returns (a corrected copy of the decisions, the log). Unknown ids and unknown
    fields are errors: a silent typo would leave the wrong text in the film."""
    out = copy.deepcopy(decisions)
    log: list[dict] = []
    for sid, ov in (rules or {}).items():
        if sid not in out:
            raise ValueError(f"span override for an unknown span {sid}")
        why = str(ov.get("why") or "").strip()
        if not why:
            raise ValueError(f"span override {sid} has no `why`: every correction needs its one-line reason")
        bad = sorted(set(ov) - SPAN_OVERRIDE_FIELDS)
        if bad:
            raise ValueError(f"span override {sid}: unknown fields {bad}")
        d = out[sid]
        keys = ("text", "confidence", "decision", "needs_owner", "flags", "alternatives")
        was = {k: copy.deepcopy(d.get(k)) for k in keys}
        if "text" in ov:
            d["text"] = ov["text"]
            if "decision" not in ov and ov["text"] != was["text"]:
                d["decision"] = "rewrite"
        for k in ("confidence", "decision", "needs_owner", "alternatives", "note"):
            if k in ov:
                d[k] = ov[k]
        flags = list(ov["flags"]) if "flags" in ov else list(d.get("flags", []))
        flags = [f for f in flags if f not in ov.get("drop_flags", [])]
        flags += [f for f in ov.get("add_flags", []) if f not in flags]
        d["flags"] = flags
        d["editor_reason"] = d.get("reason")
        d["reason"] = why
        d["override"] = {"why": why, "was": was}
        log.append({"span": sid, "why": why, "was": was, "now": {k: copy.deepcopy(d.get(k)) for k in keys}})
    return out, log


def flatten_consensus(c: dict) -> list[dict]:
    """The consensus words in order, each with the voice / name of its turn."""
    words: list[dict] = []
    for t in c["turns"]:
        for w in t["words"]:
            d = copy.deepcopy(w)
            d["turn_v0"] = t["id"]
            d["voice"] = t["voice"]
            d["speaker"] = t["speaker"]
            d["kind"] = "filler" if d.get("filler") else "word"
            words.append(d)
    return words


# ------------------------------------------------------------------------------------------------ decision text -> tokens


def is_event(tok: str) -> bool:
    return bool(EMOJI_RE.fullmatch(tok))


def parse_text(text: str, names: set[str]) -> list[dict]:
    """Tokens of a decision text: {kind word|filler|event, text (display, lower case except names / 'I' / a capital inside a sentence), norm (canonical), punct}.
    Canonical forms come from fuse.pieces so that the tokens compare with the consensus words ('gonna' = going to, 'OK' = okay)."""
    out: list[dict] = []
    prev_end = True
    for raw in text.split():
        if is_event(raw):
            out.append({"kind": "event", "text": raw, "norm": "", "punct": ""})
            continue
        ps = fuse.pieces(raw)
        if not ps:
            continue
        p = fuse.trailing_punct(raw)
        for k, (n, disp) in enumerate(ps):
            d = disp
            low = d.lower()
            if low in I_FORMS:
                d = "I" + low[1:]
            elif low in names:
                d = low[:1].upper() + low[1:]
            elif d[:1].isupper() and not prev_end and len(d) > 1 and not d.isupper():
                pass                                  # a capital inside a sentence = a proper noun (Gostosinho): keep
            elif d.isupper() and len(d) > 1:
                pass                                  # acronym
            else:
                d = low
            out.append({"kind": "filler" if n in FILLERS else "word", "text": d, "norm": n, "punct": p if k == len(ps) - 1 else ""})
        prev_end = bool(p and p in SENTENCE_END)
    return out


# ------------------------------------------------------------------------------------------------ timing of new words


class RawTimes:
    """Word times of the voting hypotheses (master clock): the evidence for where a word the vote lost sits in time."""

    def __init__(self, hyps: list):
        self.by_norm: dict[str, list[tuple[float, float, float, str]]] = {}
        for h in hyps:
            w = ENGINE_TIME_W.get(h.engine, 1.0)
            for t in h.toks:
                self.by_norm.setdefault(t.norm, []).append((t.t0, t.t1, w, h.engine))

    def find(self, norm: str, lo: float, hi: float) -> tuple[float, float] | None:
        rows = [(a, b, w, e) for a, b, w, e in self.by_norm.get(norm, []) if lo <= a <= hi]
        if not rows:
            return None
        t0 = fuse.wmedian([r[0] for r in rows], [r[2] for r in rows])
        ends = [(r[1], r[2]) for r in rows if r[3] != "canary-1b-v2"]
        t1 = fuse.wmedian([e for e, _ in ends], [w for _, w in ends]) if ends else t0 + 0.25
        return float(t0), float(max(t1, t0 + MIN_DUR))


MIN_NEW_S = 0.08           # shortest duration given to a new word


def place_run(toks: list[dict], lo: float, hi: float, raw: RawTimes | None, left: dict | None = None, right: dict | None = None) -> None:
    """Give start / end to the tokens of one untimed run inside the window [lo, hi]: the evidence of the engines when it lies inside the window (+-0.4 s),
    else the token's share of the window by length. Times are made monotone here; the whole sequence is made monotone again by fix_times."""
    k = len(toks)
    if hi - lo < MIN_WORD_S * k:
        hi = lo + MIN_WORD_S * k
    lens = [max(len(t["norm"]), 2) for t in toks]
    tot = float(sum(lens))
    a = lo
    exp = []
    for L in lens:
        b = a + (hi - lo) * L / tot
        exp.append((a, b))
        a = b
    prev_start = lo - 1e-6
    for t, (e0, e1) in zip(toks, exp):
        got = raw.find(t["norm"], max(lo, e0) - 0.25, min(hi, e1) + 0.25) if raw and t["norm"] else None
        if got and any(nb is not None and nb["norm"] == t["norm"] and abs(got[0] - nb["start"]) < 0.15 for nb in (left, right)):
            got = None                      # that is the neighbour's own word (a repeat: "we, we"), not the new one
        if got and got[0] >= prev_start:
            t["start"], t["end"], t["time_src"] = got[0], max(got[1], got[0] + MIN_NEW_S), "engines"
        else:
            t["start"], t["end"], t["time_src"] = e0, max(e1, e0 + MIN_NEW_S), "window"
        prev_start = t["start"]


def fix_times(words: list[dict]) -> int:
    """Monotone starts, end >= start + MIN_DUR, no overlap between consecutive spoken words (the earlier one is shortened). Events are left alone (a laugh can
    overlap speech). Returns how many words had to move."""
    moved = 0
    prev = None
    for w in words:
        if w["kind"] == "event":
            continue
        if prev is not None and w["start"] < prev["start"] + MIN_DUR:
            w["end"] = max(w["end"], prev["start"] + 2 * MIN_DUR)
            w["start"] = prev["start"] + MIN_DUR
            w.setdefault("flags", []).append("time_clamped")
            moved += 1
        w["end"] = max(w["end"], w["start"] + MIN_DUR)
        if prev is not None and prev["end"] > w["start"]:
            prev["end"] = max(prev["start"] + MIN_DUR, w["start"])
        prev = w
    return moved


# ------------------------------------------------------------------------------------------------ the merge


def _new_word(tok: dict, span_id: str, dec: dict, edit: str, proto: dict | None) -> dict:
    w = {"text": tok["text"], "norm": tok["norm"], "start": None, "end": None, "punct": tok["punct"], "kind": tok["kind"], "share": None,
         "conf": float(dec["confidence"]), "conf_src": "adjudicator", "flags": [], "span": span_id, "edit": edit}
    if proto:
        for k in ("voice", "speaker", "speaker_conf", "voice_conf", "interjection", "speaker_edge"):
            w[k] = proto.get(k)
    return w


def merge_span(words: list[dict], idxs: list[int], span: dict, dec: dict, names: set[str]) -> tuple[list[dict], dict]:
    """The words that replace the consensus words `idxs` under decision `dec`: old words that the decision keeps (same canonical word, in order) stay
    (time, share, speaker); the others are new and untimed (start None). Returns (new list, change record)."""
    old = [words[i] for i in idxs]
    toks = parse_text(dec["text"], names)
    old_n = [w["norm"] for w in old]
    new_n = [t["norm"] for t in toks if t["kind"] != "event"]
    flag_set = set(dec.get("flags", [])) & SPAN_WORD_FLAGS
    if dec.get("needs_owner"):
        flag_set.add("needs_owner")
    doubtful = "unclear" in flag_set or dec["decision"] == "uncertain"
    sid = span["id"]
    out: list[dict] = []
    ti = [k for k, t in enumerate(toks) if t["kind"] != "event"]       # token positions of the non-event tokens
    sm = difflib.SequenceMatcher(None, old_n, new_n, autojunk=False)
    new_pos_of: dict[int, dict] = {}
    kept_old: set[int] = set()
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            for da, db in zip(range(a0, a1), range(b0, b1)):
                w = copy.deepcopy(old[da])
                t = toks[ti[db]]
                if t["punct"]:
                    w["punct"] = t["punct"]
                share = min(0.99, float(w["share"]))
                w["conf"] = min(float(dec["confidence"]), share) if doubtful else max(float(dec["confidence"]), share)
                w["conf_src"], w["span"], w["edit"] = "adjudicator", sid, "kept"
                kept_old.add(da)
                new_pos_of[ti[db]] = w
        elif op in ("replace", "insert"):
            proto = old[min(a0, len(old) - 1)] if old else None
            for db in range(b0, b1):
                nw = _new_word(toks[ti[db]], sid, dec, "changed" if op == "replace" else "inserted", proto)
                if op == "replace":
                    nw["win"] = (old[a0]["start"], old[a1 - 1]["end"])
                new_pos_of[ti[db]] = nw
        # delete: the old words disappear (recorded below)
    for k, t in enumerate(toks):
        if t["kind"] == "event":
            ev = {"text": t["text"], "norm": "", "start": None, "end": None, "punct": "", "kind": "event", "share": None, "conf": float(dec["confidence"]), "conf_src": "adjudicator",
                  "flags": [], "span": sid, "edit": "event", "event": "laugh" if t["text"] == LAUGH else "token"}
            new_pos_of[k] = ev
    seq = [new_pos_of[k] for k in range(len(toks)) if k in new_pos_of]
    # the last spoken word keeps the span-final punctuation of the old last word when the decision gives none (the editor saw a fragment)
    spoken = [w for w in seq if w["kind"] != "event"]
    if spoken and old and not spoken[-1]["punct"] and old[-1]["punct"]:
        if (len(old) - 1) not in kept_old:
            spoken[-1]["punct"] = old[-1]["punct"]
        else:           # the last old word stays but words now follow it: its sentence-final mark moves to the new end (the editor extended the sentence)
            kw = next((w for w in spoken if w["edit"] == "kept" and w.get("i") == old[-1].get("i")), None)
            if kw is not None and kw is not spoken[-1] and kw["punct"] in SENTENCE_END and not any(t["punct"] for t in toks if t["norm"] == kw["norm"]):
                spoken[-1]["punct"], kw["punct"] = kw["punct"], ""
    for w in seq:
        for f in flag_set:
            if f not in w["flags"]:
                w["flags"].append(f)
        if w["kind"] == "event" and "emoji" not in w["flags"]:
            w["flags"].append("emoji")
    for dflag in ("term", "portuguese", "number", "filler", "disfluency", "grammar", "dropped_word", "speaker", "merged"):
        if dflag in dec.get("flags", []):
            for w in seq:
                if w["edit"] in ("changed", "inserted") and w["kind"] != "event":
                    w["flags"].append("span_" + dflag)
    removed = [old[a]["text"] for op, a0, a1, b0, b1 in sm.get_opcodes() if op in ("replace", "delete") for a in range(a0, a1)]
    rec = {"span": sid, "decision": dec["decision"], "old": " ".join(w["text"] for w in old), "new": " ".join(w["text"] for w in seq),
           "removed": removed, "added": [w["text"] for w in seq if w["edit"] in ("changed", "inserted")]}
    return seq, rec


def apply_decisions(words: list[dict], spans: dict, decisions: dict, raw: RawTimes | None, names: set[str], overrides: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    """Replace the words of every span by its decision. Returns (words, change records, stats)."""
    overrides = overrides or {}
    owner: dict[int, str] = {}
    blocks: list[dict] = []
    clashes = []
    for sid in sorted(spans, key=lambda s: (spans[s]["start"], spans[s]["end"], s)):
        sp = spans[sid]
        lo, hi = sp["slots"]
        idxs = [k for k, w in enumerate(words) if lo <= w["slot"] <= hi and sp["start"] - 0.03 <= w["start"] <= sp["end"] + 0.03]
        mine = []
        for k in idxs:
            if k in owner:
                clashes.append((sid, owner[k], words[k]["text"]))
            else:
                owner[k] = sid
                mine.append(k)
        pos = mine[0] if mine else next((k for k, w in enumerate(words) if w["start"] >= sp["start"] - 0.03), len(words))
        seq, rec = merge_span(words, mine, sp, decisions[sid], names)
        blocks.append({"span": sid, "old": mine, "pos": pos, "seq": seq, "rec": rec, "start": sp["start"], "end": sp["end"]})
    blocks.sort(key=lambda b: (b["pos"], b["start"]))
    by_first: dict[int, list[dict]] = {}
    for b in blocks:
        by_first.setdefault(b["pos"], []).append(b)
    removed = set(owner)
    out: list[dict] = []
    for k in range(len(words) + 1):
        for b in by_first.get(k, []):
            out += b["seq"]
        if k < len(words) and k not in removed:
            w = words[k]
            w["edit"] = "vote"
            w["conf_src"] = "vote"
            w["conf"] = min(0.99, float(w["share"]))
            out.append(w)
    # time the untimed words, run by run (a run never crosses from one span into another: every span has its own window)
    k = 0
    n = len(out)
    while k < n:
        if out[k]["start"] is not None or out[k]["kind"] == "event":
            k += 1
            continue
        j = k
        while j < n and out[j]["start"] is None and out[j]["kind"] != "event" and out[j]["span"] == out[k]["span"]:
            j += 1
        run = out[k:j]
        sid = run[0]["span"]
        left = next((w for w in reversed(out[:k]) if w["start"] is not None and w["kind"] != "event"), None)
        right = next((w for w in out[j:] if w["start"] is not None and w["kind"] != "event"), None)
        lo = left["end"] if left is not None else spans[sid]["start"]
        hi = right["start"] if right is not None else spans[sid]["end"]
        lo_s, hi_s = max(lo, spans[sid]["start"]), min(hi, spans[sid]["end"])
        if hi_s - lo_s >= MIN_WORD_S * len(run):         # the span's own window, when the neighbours leave room inside it
            lo, hi = lo_s, hi_s
        wins = [w["win"] for w in run if w.get("win")]
        if wins:
            lo_w, hi_w = min(a for a, _ in wins), max(b for _, b in wins)
            if len(wins) == len(run):
                lo, hi = lo_w, hi_w
            else:
                lo, hi = min(lo, lo_w), max(hi, hi_w)
        place_run(run, lo, hi, raw, left, right)
        for w in run:
            w["flags"].append("time_" + w.pop("time_src"))
            w.pop("win", None)
        k = j
    moved = fix_times(out)
    # speakers of new words: inherit from the replaced / neighbouring words
    for k, w in enumerate(out):
        if w.get("voice") is None and "voice" not in w or (w["kind"] == "event"):
            ref = next((x for x in reversed(out[:k]) if x.get("voice") is not None and x["kind"] != "event"), None) or next((x for x in out[k:] if x.get("voice") is not None), None)
            if ref is not None:
                for f in ("voice", "speaker", "speaker_conf", "voice_conf", "interjection", "speaker_edge"):
                    w[f] = ref.get(f)
    stats = {"clashes": clashes, "time_moved": moved, "blocks": len(blocks)}
    return out, [b["rec"] for b in sorted(blocks, key=lambda b: b["start"])], stats


# ------------------------------------------------------------------------------------------------ events, speakers, overrides


def place_events(words: list[dict], event_times: dict) -> list[dict]:
    """Time of every laugh token: the override `event_times[span id][n]` = {t0, t1} (n = which token of that span), else right after the previous word."""
    seen: Counter = Counter()
    for k, w in enumerate(words):
        if w["kind"] != "event":
            continue
        n = seen[w["span"]]
        seen[w["span"]] += 1
        ov = (event_times.get(w["span"]) or [])
        ov = ov[n] if n < len(ov) else None
        prev = next((x for x in reversed(words[:k]) if x["kind"] != "event" and x["start"] is not None), None)
        if ov:
            w["start"], w["end"] = float(ov["t0"]), float(ov["t1"])
            w["time_src"] = ov.get("why", "override")
            for key in ("joke", "evidence", "alt_t"):
                if key in ov:
                    w[key] = ov[key]
        else:
            t = (prev["end"] if prev else 0.0)
            w["start"], w["end"] = t, t + 0.4
            w["time_src"] = "after previous word"
    # an event keeps its place in the reading order (right after the words of its span); its time never precedes the word before it
    for k, w in enumerate(words):
        if w["kind"] == "event":
            prev = next((x for x in reversed(words[:k]) if x["kind"] != "event" and x["start"] is not None), None)
            if prev is not None and w["start"] < prev["start"]:
                w["start"] = prev["start"]
                w["end"] = max(w["end"], w["start"] + 0.1)
    return words


def apply_speaker_windows(words: list[dict], windows: list[dict]) -> int:
    """final_overrides.json["speaker_windows"]: [{t0, t1 | span, voice?, name?, interjection?, conf?, flags?, why}] -> every word whose centre lies inside the window (or
    that belongs to the decision span) moves to that voice; a laugh token follows the word before it. A window WITHOUT `voice` keeps the label and only lowers the
    confidence (`conf`: voice_conf and speaker_conf become the smaller of the old value and this one) and adds `flags` (the audit uses the flag `speaker_unresolved`
    for a label the independent evidence does not settle)."""
    n = 0
    for win in windows:
        for w in words:
            if w["kind"] == "event" or w["start"] is None:
                continue
            c = 0.5 * (w["start"] + w["end"])
            hit = (w.get("span") == win["span"]) if "span" in win else (win["t0"] <= c <= win["t1"])
            if hit:
                relabel = "voice" in win
                if relabel:
                    w["voice"], w["speaker"] = win["voice"], win["name"]
                if "interjection" in win:
                    w["interjection"] = bool(win["interjection"])
                if "conf" in win:
                    cf = float(win["conf"])
                    naming = float(win.get("naming_conf", 0.85))
                    if relabel:
                        w["voice_conf"], w["speaker_conf"] = cf, round(cf * naming, 3)
                    else:
                        w["voice_conf"] = min(cf, w["voice_conf"]) if w.get("voice_conf") is not None else cf
                        w["speaker_conf"] = min(round(cf * naming, 3), w["speaker_conf"]) if w.get("speaker_conf") is not None else round(cf * naming, 3)
                if relabel:
                    w.setdefault("flags", []).append("speaker_override")
                for f in win.get("flags", []):
                    if f not in w.setdefault("flags", []):
                        w["flags"].append(f)
                n += 1
    return n


def apply_word_edits(words: list[dict], edits: list[dict]) -> list[dict]:
    """final_overrides.json["word_edits"]: [{t, from, to, why}] a systematic correction of a word at a time (+-0.3 s), flagged `corrected`. Returns the log."""
    log = []
    for e in edits:
        hit = next((w for w in words if w["kind"] == "word" and abs(w["start"] - e["t"]) <= 0.3 and w["norm"] == e["from"]), None)
        if hit is None:
            log.append({"edit": e, "applied": False})
            continue
        if "to" in e:
            hit["text"] = e["to"]
            hit["norm"] = fuse.pieces(e["to"])[0][0] if fuse.pieces(e["to"]) else e["to"]
        if "punct" in e:
            hit["punct"] = e["punct"]
        hit["flags"].append("corrected")
        log.append({"edit": e, "applied": True})
    return log


def normalise_interjections(words: list[dict], max_gap: float = 1.5) -> list[dict]:
    """A mini-turn of a voice that sits between two stretches of the SAME voice (no other voice around it) is not an interjection: it is a word of that
    turn. Returns the list of words changed (for the log)."""
    runs: list[list[int]] = []
    for k, w in enumerate(words):
        if w["kind"] == "event":
            continue
        key = (w.get("voice"), bool(w.get("interjection")))
        if runs and (words[runs[-1][0]].get("voice"), bool(words[runs[-1][0]].get("interjection"))) == key:
            runs[-1].append(k)
        else:
            runs.append([k])
    log = []
    for r in range(1, len(runs) - 1):
        first = words[runs[r][0]]
        if not first.get("interjection"):
            continue
        prev_w, next_w = words[runs[r - 1][-1]], words[runs[r + 1][0]]
        if prev_w.get("voice") == first.get("voice") == next_w.get("voice"):
            gap_a = first["start"] - prev_w["end"]
            gap_b = next_w["start"] - words[runs[r][-1]]["end"]
            if gap_a <= max_gap and gap_b <= max_gap:
                for k in runs[r]:
                    words[k]["interjection"] = False
                    words[k]["flags"].append("interjection_merged")
                log.append({"t": first["start"], "voice": first.get("voice"), "text": " ".join(words[k]["text"] for k in runs[r])})
    return log


# ------------------------------------------------------------------------------------------------ the owner's opening


def apply_owner_opening(words: list[dict], opening: dict, names: set[str], raw: RawTimes | None = None) -> tuple[list[dict], dict | None]:
    """work/<take>/speech/owner_opening.json = what the owner heard in the opening after listening to the clips:

        {"segments": [{"start": 1.2, "end": 3.4, "speaker": "Larissa", "text": "Hello!"}, ...],     # in time order; text may hold inline emoji tokens
         "emoji": [{"t": 3.2, "speaker": "Larissa", "t1": 3.6, "joke": "name"}],                  # the laugh tokens the owner wants (none listed = none in the text)
         "from": 1.2, "until": 8.5,                                                              # optional: the stretch that is replaced (default: the segments + emoji)
         "conf": 0.95, "interjection": ..., "chapter_summary": "...", "note": "..."}              # all optional

    Every word and token of the final transcript that starts inside [from, until) is replaced: the segments become words (case, punctuation and names as in a decision
    text; each word is timed from the engines that wrote it inside the segment, else spread over the segment by length), the emoji become laugh tokens at the given times.
    The owner's words are the highest-confidence words of the transcript (`conf_src` owner), never flagged unclear / needs_owner, and the spans of the opening that the
    stretch covers are marked resolved. A segment is a second voice (interjection) when the speakers before and after it are the same other voice.
    Returns (the new word list, the record: from / until / counts / removed words, or None when the file holds nothing)."""
    segs = sorted(opening.get("segments") or [], key=lambda x: float(x["start"]))
    emo = sorted(opening.get("emoji") or [], key=lambda x: float(x["t"]))
    if not segs and not emo:
        return words, None
    for sg in segs:
        if not (float(sg["end"]) > float(sg["start"])) or not str(sg.get("text", "")).strip():
            raise ValueError(f"owner_opening segment {sg!r}: needs start < end and a text")
    lo = float(opening["from"]) if "from" in opening else min([float(x["start"]) for x in segs] + [float(e["t"]) for e in emo])
    hi = float(opening["until"]) if "until" in opening else max([float(x["end"]) for x in segs] + [float(e.get("t1", float(e["t"]) + 0.4)) for e in emo])
    conf = float(opening.get("conf", OWNER_CONF))
    before = [w for w in words if w["start"] < lo - 1e-6]
    after = [w for w in words if w["start"] >= hi - 1e-6]
    gone = [w for w in words if lo - 1e-6 <= w["start"] < hi - 1e-6]

    def voice_name(sp):
        if sp is None:
            return None, "other"
        v = VOICE_OF.get(str(sp).strip().lower())
        return v, (str(sp).strip().capitalize() if v is not None else "other")

    prev_sp = next((w.get("speaker") for w in reversed(before) if w["kind"] != "event"), None)
    next_sp = next((w.get("speaker") for w in after if w["kind"] != "event"), None)
    new: list[dict] = []
    for si, sg in enumerate(segs):
        v, nm = voice_name(sg.get("speaker"))
        toks = parse_text(str(sg["text"]), names)
        run = [t for t in toks if t["kind"] != "event"]
        place_run(run, float(sg["start"]), float(sg["end"]), raw)
        left = voice_name(segs[si - 1].get("speaker"))[1] if si > 0 else prev_sp
        right = voice_name(segs[si + 1].get("speaker"))[1] if si + 1 < len(segs) else next_sp
        inter = bool(sg["interjection"]) if "interjection" in sg else bool(left is not None and left == right and left != nm)
        last_end = float(sg["start"])
        for t in toks:
            base = {"voice": v, "speaker": nm, "speaker_conf": conf, "voice_conf": conf, "interjection": inter, "speaker_edge": False}
            if t["kind"] == "event":
                new.append(dict(base, text=t["text"], norm="", start=last_end, end=last_end + 0.4, punct="", kind="event", share=None, conf=conf, conf_src="owner",
                                flags=["emoji", "owner_fact", "owner_opening"], span="owner_opening", edit="event", event="laugh" if t["text"] == LAUGH else "token",
                                time_src="owner (inline in the segment text)"))
                continue
            last_end = t["end"]
            new.append(dict(base, text=t["text"], norm=t["norm"], start=t["start"], end=t["end"], punct=t["punct"], kind=t["kind"], share=None, conf=conf, conf_src="owner",
                            flags=["owner_opening", "owner_fact", "time_" + t["time_src"]], span="owner_opening", edit="owner"))
    for e in emo:
        t0 = float(e["t"])
        v, nm = voice_name(e.get("speaker"))
        ev = {"text": LAUGH, "norm": "", "start": t0, "end": float(e.get("t1", t0 + 0.4)), "punct": "", "kind": "event", "share": None, "conf": conf, "conf_src": "owner",
              "flags": ["emoji", "owner_fact", "owner_opening"], "span": "owner_opening", "edit": "event", "event": "laugh", "time_src": "owner (owner_opening.json)",
              "voice": v, "speaker": nm, "speaker_conf": conf, "voice_conf": conf, "interjection": False, "speaker_edge": False, "laugher": e.get("speaker")}
        if e.get("joke"):
            ev["joke"] = e["joke"]
        ev["evidence"] = e.get("evidence") or "the owner's answer (owner_opening.json)"
        idx = max([k for k, w in enumerate(new) if w["kind"] != "event" and w["start"] <= t0], default=-1)
        new.insert(idx + 1, ev)                 # reading order: right after the last word spoken before the laugh
    v0_removed = [w["text"] for w in gone if w.get("i") is not None and w.get("edit") in ("vote", "kept")]
    rec = {"from": lo, "until": hi, "segments": len(segs), "emoji": len(emo), "conf": conf, "removed": [w["text"] for w in gone if w["kind"] != "event"],
           "removed_events": len([w for w in gone if w["kind"] == "event"]), "removed_v0": len(v0_removed), "added": [w["text"] for w in new if w["kind"] != "event"],
           "chapter_summary": opening.get("chapter_summary"), "note": opening.get("note")}
    return before + new + after, rec


# ------------------------------------------------------------------------------------------------ turns, utterances


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:] if t else t


def render(words: list[dict], clean: bool = False, mark_below: float | None = None) -> str:
    """Display text of a run of words: a capital after . ? ! (and at the start), punctuation after each word, laugh tokens inline; clean = no events and no fillers."""
    parts: list[str] = []
    cap = True
    for w in words:
        if w["kind"] == "event":
            if not clean:
                parts.append(w["text"])
            continue
        if clean and w["kind"] == "filler":
            continue
        t = _cap(w["text"]) if cap else w["text"]
        t += w["punct"]
        if mark_below is not None and w["conf"] < mark_below:
            t = f"*{t}*"
        parts.append(t)
        cap = w["punct"] in SENTENCE_END
    return " ".join(parts)


def build_structure(words: list[dict], naming_conf: float, breaks: tuple | list = ()) -> tuple[list[dict], list[dict]]:
    """(turns, utterances). A turn = consecutive words of one (voice, interjection); an utterance = a sentence of a turn (end punctuation, or a pause of 1.2 s).
    `breaks` = times (master s) where a sentence is split although nothing in the words ends it (final_overrides.json["utterance_breaks"]): structure only, no word
    changes; the second half keeps the sentence's id plus a letter (u003 -> u003 + u003b) so that every other utterance id stays what it was."""
    groups: list[list[dict]] = []
    key = None
    for w in words:
        k = (w.get("voice"), bool(w.get("interjection")))
        if w["kind"] == "event" and groups:
            groups[-1].append(w)       # an event belongs to the turn of the word before it
            continue
        if k != key or not groups:
            groups.append([])
            key = k
        groups[-1].append(w)
    turns, utts = [], []
    uid = 0
    for ti, g in enumerate(groups):
        spoken = [w for w in g if w["kind"] != "event"]
        if not spoken:
            continue
        sp = spoken[0]
        voice = sp.get("voice")
        name = sp.get("speaker") or ("other" if voice is None else None)
        sconf = [w["speaker_conf"] for w in spoken if w.get("speaker_conf") is not None]
        vconf = [w["voice_conf"] for w in spoken if w.get("voice_conf") is not None]
        tid = f"t{len(turns):03d}"
        # sentences
        cur: list[dict] = []
        ulist = []
        forced_after: set[int] = set()
        for k, w in enumerate(g):
            cur.append(w)
            if w["kind"] == "event":
                continue
            nxt = next((x for x in g[k + 1:] if x["kind"] != "event"), None)
            hard = nxt is None or w["punct"] in SENTENCE_END or (nxt["start"] - w["end"] >= PAUSE_BREAK)
            forced = (not hard) and any(w["start"] < b <= nxt["start"] for b in breaks)
            if hard or forced:
                if forced:
                    forced_after.add(len(ulist))
                ulist.append(cur)
                cur = []
        if cur:
            if ulist:
                ulist[-1] += cur
            else:
                ulist.append(cur)
        uids = []
        base_id, letter = None, 0
        for ui, u in enumerate(ulist):
            us = [w for w in u if w["kind"] != "event"]
            if not us:
                continue
            ev = [w for w in u if w["kind"] == "event"]
            confs = [w["conf"] for w in us]
            flags = sorted({f for w in u for f in w.get("flags", [])})
            continues = None
            if ui > 0 and (ui - 1) in forced_after and base_id is not None:
                letter += 1
                this_id, continues = f"{base_id}{chr(ord('a') + letter)}", uids[-1]
            else:
                letter = 0
                this_id = base_id = f"u{uid:03d}"
                uid += 1
            row = {
                "id": this_id, "turn": tid, "speaker": name, "voice": voice, "start": round(min(w["start"] for w in u), 3), "end": round(max(w["end"] for w in u), 3),
                "text": render(u), "text_clean": render(u, clean=True), "conf_mean": round(sum(confs) / len(confs), 3), "conf_min": round(min(confs), 3),
                "n_words": len([w for w in us if w["kind"] == "word"]), "flags": flags, "events": [e["text"] for e in ev], "pre_roll": bool(us[-1]["end"] <= 0.0),
                "words": [slim_word(w) for w in u],
            }
            if continues:
                row["continues"] = continues           # the same sentence as `continues`, split by an utterance break (structure only)
            utts.append(row)
            uids.append(this_id)
        allc = [w["conf"] for w in spoken]
        turns.append({
            "id": tid, "speaker": name, "voice": voice, "interjection": bool(sp.get("interjection")), "start": round(spoken[0]["start"], 3), "end": round(spoken[-1]["end"], 3),
            "speaker_conf": round(sum(sconf) / len(sconf), 3) if sconf else None, "voice_conf": round(sum(vconf) / len(vconf), 3) if vconf else None,
            "text_conf": round(sum(allc) / len(allc), 3), "n_words": len([w for w in spoken if w["kind"] == "word"]), "utterances": uids,
            "naming_conf": naming_conf,
        })
    return turns, utts


def slim_word(w: dict) -> dict:
    d = {"text": w["text"], "norm": w["norm"], "start": round(w["start"], 3), "end": round(w["end"], 3), "punct": w["punct"], "kind": w["kind"],
         "conf": round(float(w["conf"]), 3), "conf_src": w["conf_src"], "edit": w["edit"]}
    if w.get("share") is not None:
        d["share"] = w["share"]
    if w.get("span"):
        d["span"] = w["span"]
    if w.get("i") is not None and w.get("edit") in ("vote", "kept"):
        d["v0"] = w["i"]
    if w.get("flags"):
        d["flags"] = sorted(set(w["flags"]))
    for k in ("event", "joke", "evidence", "alt_t", "time_src"):
        if w.get(k) is not None:
            d[k] = w[k]
    return d


# ------------------------------------------------------------------------------------------------ the whole thing


def mmss(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60)}:{t % 60:04.1f}"


def names_from_glossary(glossary_path: Path) -> set[str]:
    names = {"larissa", "kadu"}
    try:
        doc = json.loads(Path(glossary_path).read_text(encoding="utf-8"))
        for t in doc.get("terms", []):
            if t.get("kind") in ("person", "person_other"):
                names.add(t["canonical"].lower())
    except (OSError, ValueError):
        pass
    return names


def build_final(consensus: dict, spans: dict, decisions: dict, raw: RawTimes | None, names: set[str], overrides: dict, diar: dict | None = None) -> dict:
    """The final transcript document from the consensus, the decisions and the overrides (no file access: testable with synthetic data)."""
    decisions, span_log = apply_span_overrides(spans, decisions, overrides.get("span_overrides") or {})
    words = flatten_consensus(consensus)
    n_v0 = len(words)
    words, changes, stats = apply_decisions(words, spans, decisions, raw, names, overrides)
    words = place_events(words, overrides.get("event_times", {}))
    fix_times(words)
    edits_log = apply_word_edits(words, overrides.get("word_edits", []))
    n_sw = apply_speaker_windows(words, overrides.get("speaker_windows", []))
    opening = overrides.get("owner_opening")
    opening_rec = None
    if opening:
        words, opening_rec = apply_owner_opening(words, opening, names, raw)        # last of the word edits: the owner's ear beats every window above
        if opening_rec:
            fix_times(words)
    merged = normalise_interjections(words)
    naming_conf = float((consensus.get("naming") or {}).get("naming_confidence", 0.85))
    breaks = [float(b["t"]) for b in overrides.get("utterance_breaks", [])]
    if opening_rec:           # a break was made for the words the owner has now replaced: the owner's segments are the structure there
        breaks = [t for t in breaks if not (opening_rec["from"] <= t < opening_rec["until"])]
    turns, utts = build_structure(words, naming_conf, breaks)
    spoken = [w for w in words if w["kind"] != "event"]
    real = [w for w in spoken if w["kind"] == "word"]
    dcount = Counter(d["decision"] for d in decisions.values())
    resolved: set[str] = set()
    if opening_rec:           # the spans the owner's stretch covers are settled by the owner
        resolved = {sid for sid, sp in spans.items() if sp["start"] >= opening_rec["from"] - 0.05 and sp["end"] <= opening_rec["until"] + 0.05}
    unclear = sorted(sid for sid, d in decisions.items() if sid not in resolved and ("unclear" in d.get("flags", []) or d["decision"] == "uncertain"))
    owner = sorted(sid for sid, d in decisions.items() if sid not in resolved and d.get("needs_owner"))
    changed_words = [w for w in words if w.get("edit") in ("changed", "inserted")]

    def span_row(sid: str, d: dict) -> dict:
        row = {"id": sid, "start": spans[sid]["start"], "end": spans[sid]["end"], "decision": d["decision"], "text": d["text"], "confidence": d["confidence"], "flags": d.get("flags", []),
               "needs_owner": bool(d.get("needs_owner")), "reason": d.get("reason"), "note": d.get("note"), "alternatives": d.get("alternatives") or [],
               "consensus_text": spans[sid].get("consensus_text"), "listen_to": spans[sid].get("listen_to")}
        if d.get("override"):
            row["override"], row["editor_reason"] = d["override"], d.get("editor_reason")
        if sid in resolved:
            row.update({"decision": "owner", "needs_owner": False, "owner_resolved": True, "editor_text": d["text"],
                        "flags": [f for f in row["flags"] if f not in ("unclear", "needs_owner")] + ["owner_resolved"],
                        "reason": "settled by the owner (owner_opening.json); the editors' text was: " + str(d["text"])})
        return row

    doc = {
        "schema": SCHEMA + "/transcript_final", "take": TAKE, "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "final = consensus_v0 + adjudicated disputed spans + owner facts (jokes); NOT proofread by a human" + ("; the opening is the owner's own transcription" if opening_rec else ""),
        "clock": "master = 01-Recap-Tina video time, seconds (negative = before the master starts: only phone 00 was recording)",
        "speakers": [
            {"id": "larissa", "name": "Larissa", "voice": 1, "color_key": "speaker_larissa"},
            {"id": "kadu", "name": "Kadu", "voice": 0, "color_key": "speaker_kadu"},
            {"id": "other", "name": "other", "voice": None, "color_key": "speaker_other", "note": "no third voice was found in the class: nobody else speaks"},
        ],
        "naming": consensus.get("naming"),
        "turns": turns, "utterances": utts,
        "events": [{"text": w["text"], "event": w.get("event"), "start": round(w["start"], 3), "end": round(w["end"], 3), "span": w.get("span"), "flags": sorted(set(w["flags"])),
                    "joke": w.get("joke"), "evidence": w.get("evidence"), "time_src": w.get("time_src"), "laugher": w.get("laugher")} for w in words if w["kind"] == "event"],
        "adjudication": {
            "n_spans": len(spans), "decisions": dict(dcount), "unclear_spans": unclear, "needs_owner_spans": owner, "owner_resolved_spans": sorted(resolved),
            "changes": changes, "clashes": stats["clashes"], "span_overrides": span_log,
            "spans": [span_row(sid, d) for sid, d in sorted(decisions.items())],
        },
        "stats": {
            "words_v0": n_v0, "words": len(real), "fillers": len(spoken) - len(real), "events": len(words) - len(spoken), "turns": len(turns), "utterances": len(utts),
            "words_changed_by_adjudication": len([w for w in changed_words if w["kind"] == "word"]), "words_removed_by_adjudication": sum(len(c["removed"]) for c in changes),
            "spans_total": len(spans), "spans_changed": sum(1 for c in changes if c["removed"] or c["added"]), "spans_unclear": len(unclear), "spans_needs_owner": len(owner),
            "spans_overridden": len(span_log), "spans_owner_resolved": len(resolved),
            "time_moved": stats["time_moved"], "speaker_overrides": n_sw, "interjections_merged": len(merged),
            "speaker_unresolved_words": len([w for w in real if "speaker_unresolved" in w.get("flags", [])]),
            "conf_mean": round(sum(w["conf"] for w in real) / len(real), 4), "conf_lt_0.6": sum(1 for w in real if w["conf"] < 0.6),
            "conf_lt_0.8": sum(1 for w in real if w["conf"] < 0.8), "conf_ge_0.9": sum(1 for w in real if w["conf"] >= 0.9),
        },
        "word_edits": edits_log, "interjections_merged": merged, "utterance_breaks": overrides.get("utterance_breaks", []), "owner_opening": opening_rec,
    }
    doc["proofread_first"] = proofread_order(doc["adjudication"]["spans"])
    doc["checks"] = verify_final(doc, consensus)
    doc["systematic"] = systematic_report(doc, overrides.get("systematic", []))
    doc["unplaced"] = {k: {"clock": v.get("clock"), "note": v.get("note"), "voices": v.get("voices"), "words": v.get("words"),
                           "turns": [{"start": t["start"], "end": t["end"], "speaker": t.get("speaker"), "text": t.get("text"), "speaker_conf": t.get("speaker_conf")} for t in v.get("turns", [])]}
                       for k, v in (consensus.get("unplaced_sources") or {}).items()}
    return doc


def run_final_merge(log=print) -> dict:
    from .sources import load_clocks
    cfg = SPEECH / "config.json"
    recap = json.loads(cfg.read_text(encoding="utf-8")).get("source_dir") if cfg.exists() else None
    consensus = json.loads((SPEECH / "consensus_v0.json").read_text(encoding="utf-8"))
    spans, decisions, heads = load_batches()
    clocks = load_clocks(recap_dir=recap)
    hyps = fuse.load_hyps(SPEECH / "raw", clocks)
    doc = build_final(consensus, spans, decisions, RawTimes(hyps), names_from_glossary(SPEECH / "glossary.json"), load_overrides())
    doc["inputs"] = {"consensus": "consensus_v0.json", "batches": heads, "overrides": "final_overrides.json" if OVERRIDES.exists() else None,
                     "owner_opening": "owner_opening.json" if OWNER_OPENING.exists() else None}
    OUT_FINAL.write_text(json.dumps(json_safe(doc), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    log(f"{OUT_FINAL.name}: {doc['stats']['words']} words, {doc['stats']['turns']} turns, {doc['stats']['utterances']} utterances")
    return doc


# ------------------------------------------------------------------------------------------------ checks


def verify_final(final: dict, consensus: dict) -> dict:
    """Integrity checks of the final transcript against the consensus it came from. Every number must be 0 (or explained) for the file to be trusted:
      * non_monotone       spoken words whose start is earlier than the previous word's start
      * overlap            consecutive spoken words that overlap by more than 20 ms
      * lost_words         consensus words outside every decided span that are not in the final (by v0 index), and words inside a span that changed without being recorded
      * duplicated_words   consensus words that appear twice in the final
      * adjacent_repeats   a word repeated at once within 0.5 s (listed; a disfluency kept by an editor is flagged and not counted)
      * turn_edge_repeats  the last word of a turn equals the first word of the next turn within 0.3 s
      * turn_order         turns that start before the previous turn started
      * text_roundtrip     utterance words that do not add up to the turn's words
    """
    words = [(u["id"], w) for u in final["utterances"] for w in u["words"] if w["kind"] != "event"]
    nonmono = [(w["text"], w["start"]) for (_, a), (_, w) in zip(words, words[1:]) if w["start"] < a["start"] - 1e-6]
    overlap = [(w["text"], w["start"]) for (_, a), (_, w) in zip(words, words[1:]) if a["end"] - w["start"] > 0.02]
    v0 = {}
    for t in consensus["turns"]:
        for w in t["words"]:
            v0[w["i"]] = w
    seen = Counter(w["v0"] for _, w in words if w.get("v0") is not None)
    dup = sorted(i for i, n in seen.items() if n > 1)
    in_final = set(seen)
    kept_vote = [i for i, w in v0.items() if i not in in_final]
    # a consensus word may disappear only when a decided span replaced it: those carry a removal record
    removed_total = sum(len(c["removed"]) for c in final["adjudication"]["changes"]) + int((final.get("owner_opening") or {}).get("removed_v0", 0))
    lost = max(0, len(kept_vote) - removed_total)
    repeats = []
    for (_, a), (_, w) in zip(words, words[1:]):
        if a["norm"] == w["norm"] and w["start"] - a["end"] < 0.5 and w["kind"] == "word":
            flagged = "span_disfluency" in (w.get("flags") or []) or "span_disfluency" in (a.get("flags") or []) or a["norm"] in ("very",)
            repeats.append({"text": w["text"], "t": w["start"], "flagged_disfluency": flagged})
    unflagged = [r for r in repeats if not r["flagged_disfluency"]]
    edge = []
    turn_words: dict[str, list] = {}
    for u in final["utterances"]:
        turn_words.setdefault(u["turn"], []).extend(x for x in u["words"] if x["kind"] != "event")
    order = [t for t in final["turns"]]
    for a, b in zip(order, order[1:]):
        wa, wb = turn_words.get(a["id"]), turn_words.get(b["id"])
        if wa and wb and wa[-1]["norm"] == wb[0]["norm"] and wb[0]["start"] - wa[-1]["end"] < 0.3:
            edge.append({"turn": a["id"], "word": wa[-1]["text"], "t": wa[-1]["end"]})
    bad_order = [b["id"] for a, b in zip(order, order[1:]) if b["start"] < a["start"] - 1e-6]
    rt = [t["id"] for t in order if sum(1 for u in final["utterances"] if u["turn"] == t["id"] for x in u["words"] if x["kind"] == "word") != t["n_words"]]
    return {"non_monotone": len(nonmono), "overlap_gt_20ms": len(overlap), "lost_words": lost, "duplicated_words": len(dup), "adjacent_repeats": len(repeats), "adjacent_repeats_unflagged": unflagged,
            "turn_edge_repeats": edge, "turn_order": bad_order, "text_roundtrip": rt, "removed_by_decisions": removed_total, "v0_words": len(v0), "v0_words_in_final": len(in_final),
            "examples": {"non_monotone": nonmono[:5], "overlap": overlap[:5]}}


def systematic_report(final: dict, rules: list[dict]) -> list[dict]:
    """The recurring terms the editors reported (final_overrides.json["systematic"]: [{term, mishearings, keep_verbatim, do_not_use, why}]): how often the right term
    stands in the final text and which mishearings are left (must be none; a leftover is listed with its time, never replaced silently)."""
    words = [w for u in final["utterances"] for w in u["words"] if w["kind"] == "word"]
    norms = [w["norm"] for w in words]
    text = " ".join(norms)
    starts, pos = [], 0
    for n in norms:
        starts.append(pos)
        pos += len(n) + 1

    def find(phrase: str) -> list[float]:
        key = " ".join(fuse.pieces(x)[0][0] for x in phrase.split() if fuse.pieces(x))
        out = []
        for m in re.finditer(r"(?<![a-z0-9'])" + re.escape(key) + r"(?![a-z0-9'])", text):
            i = max(k for k, p in enumerate(starts) if p <= m.start())
            out.append(round(words[i]["start"], 2))
        return out

    rep = []
    for r in rules:
        term_hits = find(r["term"])
        left = {m: find(m) for m in (r.get("mishearings", []) + r.get("do_not_use", []))}
        left = {m: t for m, t in left.items() if t}
        rep.append({"term": r["term"], "occurrences": len(term_hits), "times": term_hits, "mishearings_left": left, "ok": not left, "why": r.get("why"), "keep_verbatim": bool(r.get("keep_verbatim"))})
    return rep


def proofread_order(spans: list[dict], limit: int = 30) -> list[dict]:
    """What a human should listen to first: the joke spans, then every span the editors could not settle (needs_owner), then the other unclear spans, lowest confidence first."""
    def rank(s: dict) -> tuple:
        joke = "joke" in s["flags"]
        return (0 if joke else 1 if s["needs_owner"] else 2 if ("unclear" in s["flags"] or s["decision"] == "uncertain") else 3, s["confidence"], s["start"])
    pick = [s for s in sorted(spans, key=rank) if rank(s)[0] < 3 and not s.get("owner_resolved")][:limit]
    return [{"rank": k + 1, "span": s["id"], "start": s["start"], "end": s["end"], "text": s["text"], "consensus_text": s.get("consensus_text"), "confidence": s["confidence"],
             "why": ("joke" if "joke" in s["flags"] else "needs the owner" if s["needs_owner"] else "unclear") + ": " + (s.get("reason") or ""), "alternatives": s.get("alternatives") or []} for k, s in enumerate(pick)]
