"""From the confusion network to a transcript: words, speakers, turns, sentences, events, joke scan (local, numpy, seconds).

    fuse.Consensus  ->  words (text, times, vote, alternatives)  ->  speaker per word (diarization_global.json turns, snapped to the
    silences between consensus words)  ->  turns / sentences  ->  events (laughter ...)  ->  consensus_v0.json

Decisions that are NOT in the vote (each is flagged in the output):
  * punctuation and capitalisation are voted among the hypotheses that wrote the winning word (the engines punctuate differently, so they are
    a draft, not a measurement); sentences also break at a pause of 1.2 s or more and at every speaker change;
  * a diarization turn boundary is only +-0.4 s accurate (0.75 s grid), so each boundary between two different voices is moved to the
    inter-word gap nearest to it (`speaker_edge` when there is no real gap);
  * a quiet second voice inside the other voice's turn is dropped by the vote (most engines never write it). When the interjection window has
    no consensus word at all, the reading of the window that agrees best with the other readings (the medoid) is inserted as a flagged
    mini-turn (`from_interjection_reading`) and the disagreement batches carry every reading;
  * laughter and other reactions are timed event tokens placed between words, never words; U+1F602 marks laughter.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import fuse
from .fuse import FILLERS, Consensus, ENGINE_SHORT

LAUGH = "\U0001F602"
EVENT_TOKENS = {"laughter": LAUGH, "applause": "\U0001F44F", "cheering": "\U0001F389"}
SENTENCE_END = {".", "?", "!", "..."}
KNOWN_CONFUSIONS = [
    # (regex over the canonical words of the consensus, what it probably is, source). S1's list of misheard class words, plus what the first
    # fusion shows; they are only SUSPECTS for the language-model editor, nothing is replaced here.
    (r"\bcounter ?(bounds?|bounce|bound|balls)\b", "counterbalance", "S1: counterbounds / counter bounce / Count your balls"),
    (r"\b(lights|light|alexa|lens|lex|lecs) (engage|engaged)\b", "legs engaged", "S1: 'Alexa engage' for 'legs engaged'"),
    (r"\b(fallers?|the photos|ballers|follows)\b", "followers", "S1: 'fallers' / 'the photos' for 'followers'"),
    (r"\b(restock|reinstall|re start)\b", "restart", "S1: restock / reinstall for restart"),
    (r"\b(saccada|sakada|secada|cicada|sicada)\b", "sacada", "S1: the class says 'sacada'; ZOUK_TERMINOLOGY.md leaves 'cicada' open"),
]


def json_safe(o):
    """NaN / inf -> None and numpy scalars -> python, recursively, so that json.dumps(allow_nan=False) accepts the document."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.floating):
        return json_safe(float(o))
    if isinstance(o, np.integer):
        return int(o)
    return o


# ------------------------------------------------------------------------------------------------ words


def load_glossary(path: Path) -> dict:
    """{canonical norm tuple: term dict} for all terms and variants; names keep their casing."""
    from .texts import norm_token
    out: dict[tuple[str, ...], dict] = {}
    if not Path(path).exists():
        return out
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for t in doc.get("terms", []):
        forms = [t["canonical"]] + list(t.get("variants", []))
        if t.get("en"):
            forms.append(t["en"])
        for k, f in enumerate(forms):
            key = tuple(x for x in (norm_token(w) for w in re.split(r"[\s\-]+", f)) if x)
            if key and key not in out:
                out[key] = {"canonical": t["canonical"], "kind": t["kind"], "weight": t.get("weight", 0), "variant": k > 0, "form": f, "key": t.get("key")}
    return out


def _surface(slot: fuse.Slot, winner: str, glossary: dict) -> str:
    sur: Counter = Counter()
    for tok in slot.mem.values():
        if tok.norm == winner:
            sur[tok.text] += tok.w
    if not sur:
        return winner
    # lower-cased forms count together; capitalised only wins when it is a name
    low: Counter = Counter()
    for s, v in sur.items():
        low[s.lower()] += v
    best = low.most_common(1)[0][0]
    g = glossary.get((winner,))
    if g and g["kind"] in ("person", "person_other") and not g["variant"]:
        return g["canonical"]
    if best in ("i", "i'm", "i'll", "i've", "i'd", "i’m", "i’ll", "i’ve", "i’d"):
        return "I" + best[1:]
    return best


def _punct(slot: fuse.Slot, winner: str) -> str:
    c: Counter = Counter()
    tot = 0.0
    for tok in slot.mem.values():
        if tok.norm == winner:
            c[tok.punct] += tok.w
            tot += tok.w
    if not c:
        return ""
    p, v = c.most_common(1)[0]
    return p if (p == "" or v / tot >= 0.5) else ""


def build_words(C: Consensus, glossary: dict, nick: dict[int, str] | None = None) -> list[dict]:
    """One dict per consensus word, in network order."""
    engines = {h.hid: h.engine for h in C.hyps}
    short = {h.hid: h.short for h in C.hyps}
    cam_of = {h.hid: h.cam for h in C.hyps}
    words: list[dict] = []
    for r, s in zip(C.results, C.slots):
        if r.winner is None:
            continue
        t0, t1 = fuse.best_word_time(s, r.winner, engines)
        win_ids = [h for n, _, h in r.tally if n == r.winner]
        win_ids = win_ids[0] if win_ids else []
        starts = [s.mem[h].t0 for h in win_ids if h in s.mem]
        spread = float(np.percentile(starts, 75) - np.percentile(starts, 25)) if len(starts) >= 2 else 0.0
        tally = []
        for n, share, ids in r.tally[:5]:
            e = {"w": n, "share": round(share, 3), "n": len(ids)}
            if n != r.winner and ids:
                e["by"] = [short[h] for h in ids][:8]
                e["text"] = _surface(s, n, glossary) if n else None
            tally.append(e)
        flags = []
        if r.share < 0.5:
            flags.append("low_share")
        if r.null_share >= 0.25 and r.winner is not None:
            flags.append("null_contest")
        if r.kept_by_overlap:
            flags.append("kept_in_overlap")
        if r.overlap:
            flags.append("overlap")
        words.append({
            "text": _surface(s, r.winner, glossary), "norm": r.winner, "start": round(t0, 3), "end": round(t1, 3), "punct": _punct(s, r.winner),
            "share": round(r.share, 3), "margin": round(r.margin, 3), "null_share": round(r.null_share, 3), "tally": tally,
            "hyps": len(win_ids), "phones": len({cam_of[h] for h in win_ids}), "engines": len({engines[h] for h in win_ids}),
            "eligible_phones": r.eligible_cams, "t_spread": round(spread, 3), "filler": r.winner in FILLERS, "flags": flags, "slot": r.idx,
        })
    # monotone times
    prev_start = -1e9
    for w in words:
        if w["start"] < prev_start:
            w["start"] = round(prev_start, 3)
            w["flags"].append("time_clamped")
        w["end"] = round(max(w["end"], w["start"] + 0.04), 3)
        prev_start = w["start"]
    for a, b in zip(words, words[1:]):
        if b["start"] > a["start"] + 0.04 and a["end"] > b["start"]:
            a["end"] = round(max(a["start"] + 0.04, b["start"]), 3)
    for i, w in enumerate(words):
        w["i"] = i
    return words


# ------------------------------------------------------------------------------------------------ speakers


def speaker_runs(turns: list[dict], merge_gap: float = 3.0) -> list[dict]:
    runs: list[dict] = []
    for t in sorted(turns, key=lambda x: x["start"]):
        if runs and runs[-1]["voice"] == t["voice"] and t["start"] - runs[-1]["end"] <= merge_gap:
            runs[-1]["end"] = max(runs[-1]["end"], t["end"])
            runs[-1]["conf"] = min(runs[-1]["conf"], t.get("confidence", 1.0))
            runs[-1]["n"] += 1
        else:
            runs.append({"start": t["start"], "end": t["end"], "voice": t["voice"], "name": t.get("name"), "conf": t.get("confidence", 1.0), "n": 1})
    return runs


def _turn_conf(turns: list[dict], t: float, voice) -> float:
    """Confidence of the diarization turn of `voice` nearest to time t (0.4 s of tolerance for the +-0.4 s moment grid)."""
    best, bd = None, 1e9
    for x in turns:
        if x["voice"] != voice:
            continue
        d = 0.0 if x["start"] - 0.4 <= t <= x["end"] + 0.4 else min(abs(t - x["start"]), abs(t - x["end"]))
        if d < bd:
            best, bd = x, d
    return float(best.get("confidence", 1.0)) if best else 0.0


def assign_speakers(words: list[dict], diar: dict, naming_conf: float, snap_s: float = 1.0) -> list[dict]:
    """Voice, name and confidences per word; boundaries between different voices snapped to the nearest real silence between words."""
    runs = speaker_runs(diar["turns"])
    if not runs or not words:
        return []
    n = len(words)
    centre = [0.5 * (w["start"] + w["end"]) for w in words]
    cut_info: list[dict] = []
    owner = [0] * n                 # run index of every word
    lo = 0
    for k in range(len(runs) - 1):
        a, b = runs[k], runs[k + 1]
        if a["voice"] == b["voice"]:
            continue
        B = 0.5 * (a["end"] + b["start"])
        cands = []
        for i in range(max(lo, 0), n - 1):
            mid = 0.5 * (words[i]["end"] + words[i + 1]["start"])
            if mid < B - snap_s - 0.5:
                continue
            if mid > B + snap_s + 0.5:
                break
            gap = words[i + 1]["start"] - words[i]["end"]
            cands.append((gap - 0.3 * abs(mid - B), i, gap))
        if cands:
            _, cut, gap = max(cands)
        else:       # no word near the boundary: split where the centres cross it
            cut = max(lo, sum(1 for c in centre if c < B) - 1)
            gap = 0.0
        cut_info.append({"cut_after": cut, "gap": gap, "nominal": B, "from": k, "to": k + 1})
        for i in range(lo, cut + 1):
            owner[i] = k
        lo = cut + 1
        for i in range(lo, n):
            owner[i] = k + 1
    # same-voice neighbours: words after the last cut already carry the later run; fix words that sit between same-voice runs
    out: list[dict] = []
    for i, w in enumerate(words):
        r = runs[owner[i]]
        vc = _turn_conf(diar["turns"], centre[i], r["voice"])
        out.append({"voice": r["voice"], "name": r["name"], "voice_conf": vc, "speaker_conf": round(vc * naming_conf, 3),
                    "speaker_edge": False, "interjection": False})
    for c in cut_info:
        for i in range(max(0, c["cut_after"] - 1), min(n, c["cut_after"] + 3)):
            near = abs(words[i]["start"] - words[c["cut_after"] + 1]["start"]) < 0.4 if c["cut_after"] + 1 < n else True
            if c["gap"] < 0.08 and (near or abs(i - c["cut_after"]) <= 0):
                out[i]["speaker_edge"] = True
    return out


def interjection_windows(diar: dict, base_voice_at) -> list[dict]:
    """Second voices inside another voice's turn or in a pause (S1 route B), without the 'same voice continues across a pause' kind."""
    out = []
    for j in diar.get("interjections", []):
        mid = 0.5 * (j["start"] + j["end"])
        if j.get("gap_fill"):
            continue
        bv = base_voice_at(mid)
        if bv is None or bv != j["voice"]:
            out.append(dict(j, base_voice_here=bv))
    return out


BACKCHANNEL = {"yes", "yeah", "yep", "no", "okay", "right", "oh", "ah", "um", "uh", "sorry", "good", "great", "thanks", "thank", "hello", "hi", "wow", "nice", "exactly",
               "sure", "true", "really", "bonus", "perfect", "hmm", "mhm", "ha", "haha", "hey", "please", "well"}


def apply_interjections(words: list[dict], spk: list[dict], wins: list[dict], naming_conf: float, names: dict, max_words: int = 2) -> None:
    """Words inside a second-voice window move to that voice only when the window holds at most `max_words` words or the word is a back-channel
    ("yes", "right", "okay"): a teacher's sentence that merely overlaps a short interjection stays with the base voice, flagged `maybe_voice` so the
    editor sees it."""
    for j in wins:
        inside = [i for i, w in enumerate(words) if j["start"] - 0.05 <= 0.5 * (w["start"] + w["end"]) <= j["end"] + 0.05]
        for i in inside:
            w, s = words[i], spk[i]
            vc = round(min(1.0, 0.4 + float(j.get("margin", 0.0))), 3)
            if len(inside) <= max_words or w["norm"] in BACKCHANNEL:
                s["voice"], s["name"] = j["voice"], names.get(j["voice"], j.get("name"))
                s["interjection"] = True
                s["voice_conf"] = vc
                s["speaker_conf"] = round(vc * naming_conf, 3)
            else:
                w.setdefault("flags", []).append("overlaps_other_voice")
                w["maybe_voice"] = {"voice": j["voice"], "name": names.get(j["voice"], j.get("name")), "window": [j["start"], j["end"]], "confidence": vc}


def medoid_reading(C: Consensus, t0: float, t1: float, pad: float = 0.2, min_support: float = 0.4) -> dict | None:
    """The reading of [t0, t1] that the other readings support best: the tokens of every hypothesis whose time key is inside are readings; the one with the
    largest summed word support (the share of readings that contain the word) wins, and only its words with support >= min_support are kept."""
    readings = []
    for h in C.hyps:
        # a word counts when at least 70 % of it lies inside the window (a neighbour's "and" that merely touches the edge does not)
        toks = [t for t in h.toks if t.norm not in FILLERS and (min(t1, t.t1) - max(t0, t.t0)) >= 0.7 * max(0.04, t.t1 - t.t0)]
        if toks:
            readings.append((h, toks))
    if len(readings) < 3:
        return None
    sup = Counter()
    for _, toks2 in readings:
        for n in {t.norm for t in toks2}:
            sup[n] += 1
    support = {n: sup[n] / len(readings) for n in sup}
    best, best_score = None, -1.0
    for h, toks in readings:
        score = sum(support[t.norm] for t in toks if support[t.norm] >= min_support)
        if score > best_score:
            best, best_score = (h, toks), score
    if best is None or best_score <= 0:
        return None
    h, toks = best
    toks = [t for t in toks if support[t.norm] >= min_support]
    return {"hyp": h.short, "toks": toks, "n_readings": len(readings), "support": support, "score": best_score}


def fuse_lev_words(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def fill_interjections(words: list[dict], spk: list[dict], wins: list[dict], C: Consensus, naming_conf: float, names: dict, glossary: dict,
                       laughter: list[tuple[float, float]] | None = None) -> list[dict]:
    """Insert the medoid reading of every interjection window that has no consensus word (the vote drops a quiet second voice). A window inside a detected
    laugh is skipped (the engines hallucinate words on laughter). Returns the filled windows."""
    filled = []
    for j in wins:
        if laughter and any(min(j["end"], b) - max(j["start"], a) >= 0.5 * (j["end"] - j["start"]) for a, b in laughter):
            continue
        inside = [w for w, s in zip(words, spk) if j["start"] - 0.05 <= 0.5 * (w["start"] + w["end"]) <= j["end"] + 0.05]
        dur = j["end"] - j["start"]
        if len(inside) >= max(1, int(0.8 * dur)):
            continue
        m = medoid_reading(C, j["start"], j["end"])
        if not m or not m["toks"]:
            continue
        new_w, new_s = [], []
        for t in m["toks"]:
            new_w.append({"text": t.text, "norm": t.norm, "start": round(t.t0, 3), "end": round(t.t1, 3), "punct": t.punct, "share": round(m["support"].get(t.norm, 0.0), 3),
                          "margin": 0.0, "null_share": 0.0, "tally": [{"w": t.norm, "share": round(m["support"].get(t.norm, 0.0), 3), "n": 1}], "hyps": 1, "phones": 1, "engines": 1,
                          "eligible_phones": 0, "t_spread": 0.0, "filler": False, "flags": ["from_interjection_reading", "low_share"], "slot": -1,
                          "reading_hyp": m["hyp"]})
            new_s.append({"voice": j["voice"], "name": names.get(j["voice"], j.get("name")), "voice_conf": round(min(1.0, 0.4 + float(j.get("margin", 0.0))), 3),
                          "speaker_conf": round(min(1.0, 0.4 + float(j.get("margin", 0.0))) * naming_conf, 3), "speaker_edge": False, "interjection": True})
        # splice by time
        pos = sum(1 for w in words if w["start"] < new_w[0]["start"])
        words[pos:pos] = new_w
        spk[pos:pos] = new_s
        filled.append({"start": j["start"], "end": j["end"], "voice": j["voice"], "n_words": len(new_w), "reading_hyp": m["hyp"], "n_readings": m["n_readings"]})
    for i, w in enumerate(words):
        w["i"] = i
    return filled


# ------------------------------------------------------------------------------------------------ turns, sentences


def _capitalise(txt: str) -> str:
    return txt[:1].upper() + txt[1:] if txt else txt


def render(words: list[dict], events: list[dict] | None = None, clean: bool = False, mark_below: float | None = None) -> str:
    """Text of a run of words: voted punctuation, a capital after a sentence end, events (emoji) between words at their time; with mark_below, words whose vote share
    is lower are wrapped in asterisks (human-readable transcript)."""
    parts: list[str] = []
    cap = True
    evs = sorted(events or [], key=lambda e: e["t"])
    ei = 0
    for w in words:
        while ei < len(evs) and evs[ei]["t"] <= w["start"]:
            if not clean and evs[ei].get("token"):
                parts.append(evs[ei]["token"])
            ei += 1
        if clean and w["filler"]:
            continue
        t = w["text"]
        if cap:
            t = _capitalise(t)
        t = t + (w["punct"] if w["punct"] != "..." else "...")
        parts.append(f"*{t}*" if mark_below is not None and w["share"] < mark_below else t)
        cap = w["punct"] in SENTENCE_END
    while ei < len(evs):
        if not clean and evs[ei].get("token"):
            parts.append(evs[ei]["token"])
        ei += 1
    return " ".join(parts)


def sentence_breaks(words: list[dict], pause: float = 1.2) -> list[int]:
    """Indices (into `words`) of the last word of each sentence."""
    out = []
    for i, w in enumerate(words):
        last = i == len(words) - 1
        if last or w["punct"] in SENTENCE_END or (words[i + 1]["start"] - w["end"] >= pause):
            out.append(i)
    return out


def build_turns(words: list[dict], spk: list[dict], events: list[dict], diar: dict) -> list[dict]:
    """Consecutive words of one voice are one turn; a second voice inside a turn is its own mini-turn."""
    turns: list[dict] = []
    cur = None
    for w, s in zip(words, spk):
        key = (s["voice"], s["interjection"])
        if cur is None or cur["key"] != key:
            cur = {"key": key, "voice": s["voice"], "name": s["name"], "words": [], "spk": [], "interjection": s["interjection"]}
            turns.append(cur)
        cur["words"].append(w)
        cur["spk"].append(s)
    out = []
    sid = 0
    for k, t in enumerate(turns):
        ws = t["words"]
        sc = [s["speaker_conf"] for s in t["spk"]]
        a, b = ws[0]["start"], ws[-1]["end"]
        # events before the first turn's first word belong to that turn, events between two turns to the earlier turn
        lo_t = a if k else -1e9
        hi_t = turns[k + 1]["words"][0]["start"] if k + 1 < len(turns) else 1e9
        evs = [e for e in events if lo_t <= e["t"] < hi_t]
        sents = []
        last = 0
        for bidx in sentence_breaks(ws):
            seg = ws[last: bidx + 1]
            s_evs = [e for e in evs if seg[0]["start"] <= e["t"] <= (ws[bidx + 1]["start"] if bidx + 1 < len(ws) else 1e9)]
            sents.append({"id": f"s{sid:04d}", "start": seg[0]["start"], "end": seg[-1]["end"], "text": render(seg, s_evs), "text_clean": render(seg, None, clean=True),
                          "words": [last, bidx]})
            sid += 1
            last = bidx + 1
        out.append({
            "id": f"t{k:03d}", "voice": t["voice"], "speaker": t["name"], "start": a, "end": b, "interjection": t["interjection"],
            "speaker_conf": round(float(np.mean(sc)), 3), "voice_conf": round(float(np.mean([s["voice_conf"] for s in t["spk"]])), 3),
            "n_words": len(ws), "n_fillers": sum(1 for w in ws if w["filler"]),
            "agreement_mean": round(float(np.mean([w["share"] for w in ws])), 3), "agreement_min": round(float(min(w["share"] for w in ws)), 3),
            "n_low_share": sum(1 for w in ws if "low_share" in w["flags"]),
            "text": render(ws, evs), "text_clean": render(ws, None, clean=True), "sentences": sents,
            "events": [e["id"] for e in evs],
            "words": [dict(w, speaker_edge=s["speaker_edge"], speaker_conf=s["speaker_conf"], voice_conf=s["voice_conf"], interjection=s["interjection"]) for w, s in zip(ws, t["spk"])],
        })
    return out


# ------------------------------------------------------------------------------------------------ glossary hits, suspects


def glossary_hits(words: list[dict], glossary: dict, max_len: int = 4) -> list[dict]:
    """Glossary terms (and variants) found in the consensus words, plus known confusions."""
    norms = [w["norm"] for w in words]
    hits = []
    for i in range(len(norms)):
        for L in range(min(max_len, len(norms) - i), 0, -1):
            key = tuple(norms[i: i + L])
            g = glossary.get(key)
            if g and (L > 1 or len(key[0]) >= 4 or g["kind"] in ("person", "person_other")):
                hits.append({"i0": i, "i1": i + L - 1, "start": words[i]["start"], "end": words[i + L - 1]["end"], "term": g["canonical"], "kind": g["kind"], "via_variant": g["variant"],
                             "text": " ".join(norms[i: i + L])})
                break
    return hits


def suspects(words: list[dict]) -> list[dict]:
    """Known mishearings (KNOWN_CONFUSIONS) in the consensus text."""
    norms = [w["norm"] for w in words]
    text = " ".join(norms)
    starts = []
    pos = 0
    for n in norms:
        starts.append(pos)
        pos += len(n) + 1
    out = []
    for rx, what, why in KNOWN_CONFUSIONS:
        for m in re.finditer(rx, text):
            i0 = max(i for i, p in enumerate(starts) if p <= m.start())
            i1 = max(i for i, p in enumerate(starts) if p < m.end())
            out.append({"i0": i0, "i1": i1, "start": words[i0]["start"], "end": words[i1]["end"], "text": m.group(0), "probably": what, "why": why})
    return out


# ------------------------------------------------------------------------------------------------ events


def gather_events(events_master: dict | None, diar: dict) -> list[dict]:
    """Timed reaction tokens: the AST laughter / applause events (events_master.json) and the joke candidates of the diarization file. The laugh that belongs to
    a joke candidate is ONE event (the token sits at the time of the laugh); candidates without a detected laugh are kept as 'joke_reaction_candidate' tokens."""
    evs: list[dict] = []
    for e in (events_master or {}).get("events", []):
        evs.append({"id": e["id"], "type": e["kind"], "token": EVENT_TOKENS.get(e["kind"]), "t": float(e["t0"]), "t0": float(e["t0"]), "t1": float(e["t1"]),
                    "source": "AST AudioSet tagging, " + str(len([c for c, v in e.get("cams", {}).items() if v >= 0.04])) + " phones above 0.04", "peak": e.get("peak"),
                    "confidence": "weak" if e.get("peak", 1) < 0.2 else "medium", "needs_owner_confirmation": False})
    for k, tkn in enumerate(diar.get("emoji_tokens", [])):
        t = float(tkn["t"])
        match = next((e for e in evs if e["t0"] - 0.6 <= t <= e["t1"] + 0.6 and e["type"] == "laughter"), None)
        joke = tkn.get("joke")
        if match is not None:
            match["joke"] = joke
            match["needs_owner_confirmation"] = True
            match["alt_t"] = tkn.get("alt_t")
            match["evidence"] = tkn.get("evidence")
        else:
            evs.append({"id": f"jk{k:02d}", "type": "joke_reaction_candidate", "token": LAUGH, "t": t, "t0": t, "t1": t, "joke": joke, "alt_t": tkn.get("alt_t"),
                        "source": "S1 joke candidate (owner fact); no laughter detected at this time", "confidence": "owner_fact", "needs_owner_confirmation": True,
                        "evidence": tkn.get("evidence")})
    evs.sort(key=lambda e: e["t"])
    return evs


def wordless_events(levels, words: list[dict], events_ast: dict[str, list] | None = None, snr_min: float = 7.0, min_phones: int = 3, min_s: float = 0.4,
                    word_pad: float = 0.25, step: float = 0.1) -> list[dict]:
    """Stretches that are clearly louder than the room on at least `min_phones` phones but where no consensus word is: laughter, applause, footsteps, a
    cough, a "mm" nobody wrote. CANDIDATES for reaction tokens (ASR engines drop them); AST top classes of the same moment are attached when available."""
    m_lo, m_hi = levels.m_lo, levels.m_hi
    ts = np.arange(m_lo + step / 2, m_hi, step)
    idx = np.clip(((ts - m_lo) / levels.grid).astype(int), 0, levels.lvl_s.shape[1] - 1)
    loud = np.zeros(len(ts), dtype=int)
    exc = np.zeros(len(ts))
    for i in range(levels.lvl_s.shape[0]):
        lv = levels.lvl_s[i, idx]
        fl = levels.flo[i, idx]
        d = lv - fl
        ok = ~np.isnan(d) & (d >= snr_min)
        loud += ok
        exc = np.maximum(exc, np.where(np.isnan(d), 0.0, d))
    covered = np.array([(~np.isnan(levels.lvl_s[:, k])).sum() for k in idx])
    need = np.minimum(min_phones, np.maximum(covered, 1))
    cand = loud >= need
    wstart = np.array([w["start"] for w in words])
    wend = np.array([w["end"] for w in words])
    near_word = np.zeros(len(ts), dtype=bool)
    for a, b in zip(wstart, wend):
        near_word |= (ts >= a - word_pad) & (ts <= b + word_pad)
    cand &= ~near_word
    out = []
    i = 0
    while i < len(ts):
        if not cand[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ts) and (cand[j + 1] or (j + 3 < len(ts) and cand[j + 2])):
            j += 1
        if (ts[j] - ts[i] + step) >= min_s:
            out.append({"t0": round(float(ts[i] - step / 2), 2), "t1": round(float(ts[j] + step / 2), 2), "peak_excess_db": round(float(exc[i: j + 1].max()), 1),
                        "mean_excess_db": round(float(exc[i: j + 1].mean()), 1), "phones": int(loud[i: j + 1].max())})
        i = j + 1
    return out


JOKE_PATTERNS = {
    "self_reference": r"\b(this is|i am|i'm|im|my name is|call me|i'm called)\b",
    "name_mention": r"\b(larissa|kadu|cadu|clarissa|lari)\b",
    "appearance": r"\b(hair|bald|beard|tall|short|red|blonde|skinny|fat|young|old|handsome|pretty|beautiful|shoes|shirt|dress|wearing|look at my)\b",
    "irony_marker": r"\b(just kidding|kidding|joke|jokes|funny|laugh|laughing|haha|ha ha|hehe|sarcastic)\b",
    "other_person": r"\b(my (wife|husband|girlfriend|boyfriend|partner|mother|father)|he|she)\b",
}


NAME_FORMS = {"larissa": {"larissa", "clarissa", "lari", "larisa"}, "kadu": {"kadu", "cadu"}}


def joke_scan(turns: list[dict], diar: dict, names: dict, other_names: list[str]) -> list[dict]:
    """Candidate jokes / contradictions in the fused text (S2 pass): sentences where a voice talks about itself or its looks, mentions a name or flags irony,
    with whether what it says fits who the voice is. Nothing is asserted: the owner confirms. `name_contradicts_voice` = a self-introduction in which the voice
    named X says the OTHER teacher's name (the owner's first joke has exactly that shape, but no engine writes the name, so the scan cannot see it there)."""
    out = []
    for t in turns:
        me = (t["speaker"] or "").lower()
        for s in t["sentences"]:
            low = s["text_clean"].lower()
            hits = {k: re.findall(rx, low) for k, rx in JOKE_PATTERNS.items()}
            hits = {k: v for k, v in hits.items() if v}
            if not any(k in hits for k in ("self_reference", "appearance", "irony_marker", "name_mention")):
                continue
            kinds = []
            said = {x.lower() for x in hits.get("name_mention", [])}
            if "name_mention" in hits and "self_reference" in hits:
                kinds.append("self_introduction")
                other = [n for n, forms in NAME_FORMS.items() if n != me and said & forms]
                if me in NAME_FORMS and other:
                    kinds.append("name_contradicts_voice")
            elif "name_mention" in hits:
                kinds.append("name_mention")
            if "appearance" in hits and ("self_reference" in hits or re.search(r"\bmy\b", low)):
                kinds.append("self_appearance")
            elif "appearance" in hits and re.search(r"\b(hair|bald|beard)\b", low):
                kinds.append("appearance")
            if "irony_marker" in hits:
                kinds.append("irony_marker")
            if not kinds:
                continue
            out.append({"turn": t["id"], "start": s["start"], "end": s["end"], "speaker": t["speaker"], "voice": t["voice"], "kinds": kinds, "text": s["text"], "matches": hits})
    return out
