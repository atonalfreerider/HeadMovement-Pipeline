"""Disagreement batches for the language-model editor: the places where the consensus is not unanimous with high confidence (local, seconds).

A SLOT of the confusion network is disputed when the vote did not settle it:

  low_share        the winner holds less than `share_hi` of the voting weight (threshold chosen from the held-out reliability curve, fuse_eval.py:
                   the share above which a held-out transcript agrees with the consensus word almost always; see NOTES_speech2.md)
  null_contest     the winner is a word but "nothing" holds `null_contest` or more (is the word there at all?)
  dropped_word     "nothing" won, but a single word still holds `null_word` or more (a word may have been voted out)
  overlap          the slot lies in an overlapping-speech window or an interjection window (a quiet second voice; the vote cannot be trusted)
  suspect          a known mishearing of the class (fuse_post.KNOWN_CONFUSIONS) or a near miss of a glossary term (sound-alike, not equal)
  owner_fact       the slot lies in a window the owner described (the two jokes of the opening): nothing there is final without the owner

Flagged slots that are adjacent (or separated by one agreed word) form one SPAN; each span carries: the speaker, the master time, the consensus text,
the sentences around it (2 before, 2 after), every candidate reading with the (phone, engine) sources and their word confidences, the readings of the
non-voting targeted decodes (raw_regions) in the window, the glossary hits and the reasons. Spans are written to speech/disagreements/batch_NN.json in time
order, 15-25 per batch.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import SPEECH, fuse
from .fuse import FILLERS, Consensus
from .fuse_post import KNOWN_CONFUSIONS, json_safe

OWNER_FACTS_FILE = SPEECH / "owner_facts.json"


def load_owner_facts(path: Path | None = None) -> tuple[list[dict], list[tuple[float, float, str]]]:
    """(facts, windows) from work/<take>/speech/owner_facts.json: what the owner told us about the class (the two jokes of the opening) and the master-time windows
    they belong to. The file is git-ignored (it describes real people); without it the batches simply have no owner windows."""
    p = Path(path) if path else OWNER_FACTS_FILE
    if not p.exists():
        return [], []
    doc = json.loads(p.read_text(encoding="utf-8"))
    facts = doc.get("facts", [])
    wins = [(float(w["start"]), float(w["end"]), w["name"]) for f in facts for w in f.get("windows", [])]
    return facts, sorted(wins)


@dataclass
class BatchParams:
    share_hi: float = 0.90
    margin_min: float = 0.60
    null_contest: float = 0.25
    null_word: float = 0.15
    merge_words: int = 0             # agreed words allowed between two flagged slots inside one span (0 = only contiguous flagged slots)
    max_slots: int = 14              # a longer run is split at its most agreed slot
    per_batch: tuple[int, int] = (15, 25)
    target: int = 20
    owner_windows: tuple[tuple[float, float, str], ...] = ()      # (master start, end, name) from owner_facts.json: the engines disagree on every word there
    near_miss_sim: float = 0.80
    near_miss_max_count: int = 3     # a word the class says more often than this is ordinary vocabulary, not a mis-heard glossary term
    near_miss_min_len: int = 5
    max_readings: int = 10


def owner_key(t: float, wins) -> str | None:
    for a, b, name in wins:
        if a <= t < b:
            return name
    return None


def flag_slots(C: Consensus, words: list[dict], suspects: list[dict], near_miss: dict[int, str], P: BatchParams, windows: list[tuple[float, float, str]]) -> dict[int, list[str]]:
    """{slot index: reasons}."""
    flags: dict[int, list[str]] = {}
    for r in C.results:
        why = []
        if r.winner is not None and r.share < P.share_hi:
            why.append(f"low_share {r.share:.2f}")
        elif r.winner is not None and r.margin < P.margin_min:
            why.append(f"low_margin {r.margin:.2f}")
        if r.winner is not None and r.null_share >= P.null_contest and not r.kept_by_overlap:
            why.append(f"null_contest {r.null_share:.2f}")
        if r.winner is None and r.word_share >= P.null_word:
            why.append(f"dropped_word {r.best_word}:{r.word_share:.2f}")
        if r.kept_by_overlap:
            why.append("overlap_kept")
        elif r.overlap and (r.winner is None or r.share < 0.95):
            why.append("overlap")
        if owner_key(r.t, windows) and (r.winner is not None or r.word_share > 0.05):
            why.append("owner_fact")
        if why:
            flags[r.idx] = why
    for s in suspects:
        for i in range(s["i0"], s["i1"] + 1):
            slot = words[i]["slot"]
            if slot >= 0:
                flags.setdefault(slot, []).append(f"suspect '{s['text']}' -> {s['probably']}")
    for slot, term in near_miss.items():
        flags.setdefault(slot, []).append(f"glossary_near_miss {term}")
    return flags


def near_misses(words: list[dict], glossary: dict, P: BatchParams) -> dict[int, str]:
    """{slot: glossary term} for consensus words that sound like a glossary word without being it (sacada / saccada, counterbalance / counter balls ...)."""
    singles = {k[0]: v for k, v in glossary.items() if len(k) == 1 and not v["variant"] and len(k[0]) >= P.near_miss_min_len and v["weight"] >= 60}
    variants = {k[0] for k, v in glossary.items() if len(k) == 1}
    freq = Counter(w["norm"] for w in words)
    out = {}
    for w in words:
        n = w["norm"]
        if n in variants or len(n) < P.near_miss_min_len or w["slot"] < 0 or freq[n] > P.near_miss_max_count:
            continue
        for g, v in singles.items():
            if abs(len(g) - len(n)) <= 2 and fuse.sim(n, g) >= P.near_miss_sim:
                out[w["slot"]] = v["canonical"]
                break
    return out


def group_spans(flags: dict[int, list[str]], C: Consensus, merge_words: int, max_slots: int = 14, windows=()) -> list[list[int]]:
    """Runs of flagged slots (an agreed word between two flagged slots joins them only when merge_words allows); a run longer than max_slots is
    split at its most agreed slot, so a span stays a phrase an editor can judge at a glance."""
    idxs = sorted(flags)
    runs: list[list[int]] = []
    for i in idxs:
        if runs and owner_key(C.results[runs[-1][-1]].t, windows) != owner_key(C.results[i].t, windows):
            runs.append([i])
            continue
        if runs:
            gap = [k for k in range(runs[-1][-1] + 1, i) if C.results[k].winner is not None]
            if i == runs[-1][-1] + 1 or (merge_words and len(gap) <= merge_words and i - runs[-1][-1] <= merge_words + 4):
                runs[-1] = list(range(runs[-1][0], i + 1))
                continue
        runs.append([i])
    out: list[list[int]] = []
    todo = list(runs)
    while todo:
        r = todo.pop(0)
        if len(r) <= max_slots:
            out.append(r)
            continue
        lo, hi = len(r) // 3, 2 * len(r) // 3
        cut = max(range(lo, hi + 1), key=lambda k: (C.results[r[k]].share if C.results[r[k]].winner is not None else 0.0, -abs(k - len(r) // 2)))
        todo.insert(0, r[cut + 1:])
        todo.insert(0, r[: cut + 1])
    return out


def _surface_of(tok: fuse.Tok) -> str:
    return tok.text


def readings_for(C: Consensus, slot_ids: list[int], t_window: tuple[float, float] | None = None) -> list[dict]:
    """Candidate readings of a set of slots: one per distinct normalised text, with the (phone, engine) sources. With `t_window` (owner windows, where the engines'
    timestamps are unreliable and the alignment may put a word in a far slot) each hypothesis' reading is also completed with its tokens whose own time lies in the window."""
    per_hyp: dict[int, dict[int, fuse.Tok]] = defaultdict(dict)
    for k in slot_ids:
        for hid, tok in C.slots[k].mem.items():
            if t_window is None or t_window[0] - 1.5 <= tok.tk <= t_window[1] + 1.5:
                per_hyp[hid][tok.idx] = tok
    if t_window is not None:
        for h in C.hyps:
            for tok in h.toks:
                if t_window[0] - 0.2 <= tok.tk <= t_window[1] + 0.2:
                    per_hyp[h.hid][tok.idx] = tok
    elig = [h for h in C.hyps if any(fuse.eligible(h, C.slots[k].t, C.params) for k in slot_ids)]
    groups: dict[str, dict] = {}
    elig = [h for h in C.hyps if any(fuse.eligible(h, C.slots[k].t, C.params) for k in slot_ids)]
    wtot = 0.0
    for h in elig:
        toks = [per_hyp[h.hid][i] for i in sorted(per_hyp.get(h.hid, {}))]
        key = " ".join(t.norm for t in toks)
        w = float(np.mean([t.w for t in toks])) if toks else float(np.mean([t.w for t in h.toks]) if h.toks else 0.5)
        wtot += w
        g = groups.setdefault(key, {"norm": key, "texts": Counter(), "sources": [], "weight": 0.0})
        g["texts"][" ".join(t.text for t in toks)] += 1
        confs = [t.conf for t in toks if t.conf is not None]
        g["sources"].append({"phone": h.cam, "engine": h.engine, "conf": round(float(np.mean(confs)), 2) if confs else None, "conf_min": round(float(min(confs)), 2) if confs else None})
        g["weight"] += w
    out = []
    for key, g in groups.items():
        src = [f"{x['phone']}/{fuse.ENGINE_SHORT.get(x['engine'], x['engine'])}" + ("" if x["conf"] is None else f" {x['conf']}" + ("" if x["conf_min"] == x["conf"] else f" (min {x['conf_min']})")) for x in g["sources"]]
        out.append({"text": g["texts"].most_common(1)[0][0], "norm": key, "n_sources": len(g["sources"]), "weight_share": round(g["weight"] / max(wtot, 1e-9), 3),
                    "engines": sorted({fuse.ENGINE_SHORT.get(x["engine"], x["engine"]) for x in g["sources"]}), "phones": sorted({x["phone"] for x in g["sources"]}), "sources": src,
                    "kind": "nothing" if key == "" else "full_pass"})
    out.sort(key=lambda g: -g["weight_share"])
    return out


def region_readings(region_hyps: list[dict], t0: float, t1: float, pad: float = 0.3) -> list[dict]:
    """The non-voting targeted decodes (raw_regions): tokens whose centre lies in the window, one reading per (phone, model, variant)."""
    out = []
    for h in region_hyps:
        toks = [t for t in h["toks"] if t0 - pad <= 0.5 * (t.t0 + t.t1) <= t1 + pad]
        if toks:
            out.append(f"{h['cam']}/{fuse.ENGINE_SHORT.get(h['model'], h['model'])} {h['variant']}: " + " ".join(t.text for t in toks))
    return out


def load_region_hyps(raw_regions: Path, clocks: dict) -> list[dict]:
    out = []
    for f in sorted(Path(raw_regions).glob("*/[0-9][0-9].json")):
        cam = f.stem
        clk = clocks.get(cam)
        if clk is None:
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        for h in doc.get("hypotheses", []):
            if h["variant"].startswith(("pt", "auto")):
                continue
            words = [{"word": w["word"], "start": clk.to_master(float(w["start"])), "end": clk.to_master(float(w["end"])), "prob": w.get("prob")} for w in h.get("words", []) if w.get("word")]
            if words:
                out.append({"region": f.parent.name, "cam": cam, "model": h["model"], "variant": h["variant"], "toks": fuse.make_toks(words)})
    return out


def glossary_in_readings(readings: list[dict], glossary: dict, P: BatchParams) -> list[dict]:
    """Glossary terms (or sound-alikes) that appear in any candidate reading: [{term, kind, reading indices, exact}]."""
    singles = {k[0]: v for k, v in glossary.items() if len(k) == 1}
    multi = {k: v for k, v in glossary.items() if len(k) > 1}
    found: dict[str, dict] = {}
    for ri, r in enumerate(readings):
        ws = r["norm"].split()
        for i, w in enumerate(ws):
            g = singles.get(w)
            if g:
                e = found.setdefault(g["canonical"], {"term": g["canonical"], "kind": g["kind"], "readings": set(), "exact": True, "note": g.get("form")})
                e["readings"].add(ri)
            elif len(w) >= P.near_miss_min_len:
                for k, v in singles.items():
                    if not v["variant"] and len(k) >= P.near_miss_min_len and v["weight"] >= 60 and abs(len(k) - len(w)) <= 2 and fuse.sim(w, k) >= P.near_miss_sim:
                        e = found.setdefault(v["canonical"], {"term": v["canonical"], "kind": v["kind"], "readings": set(), "exact": False, "note": f"sounds like '{w}'"})
                        e["readings"].add(ri)
        for key, g in multi.items():
            L = len(key)
            for i in range(len(ws) - L + 1):
                if tuple(ws[i: i + L]) == key:
                    e = found.setdefault(g["canonical"], {"term": g["canonical"], "kind": g["kind"], "readings": set(), "exact": True, "note": g.get("form")})
                    e["readings"].add(ri)
    return [dict(e, readings=sorted(e["readings"])) for e in found.values()]


def make_spans(C: Consensus, words: list[dict], turns: list[dict], sentences: list[dict], suspects: list[dict], glossary: dict, region_hyps: list[dict], levels,
               clocks: dict, windows: list[tuple[float, float, str]], P: BatchParams, owner_facts: list[dict] | None = None,
               diar_interjections: list[dict] | None = None) -> list[dict]:
    """All disagreement spans in time order: one per owner window (its slots chosen by TIME, because mis-timed words of the laugh sit out of network order) and one
    per run of flagged slots elsewhere."""
    owner_facts = owner_facts or []
    word_of_slot = {w["slot"]: w for w in words if w["slot"] >= 0}
    nm = near_misses(words, glossary, P)
    flags = flag_slots(C, words, suspects, nm, P, windows)
    owner_slots: dict[str, list[int]] = {}
    # owner windows keep every slot that carries any word evidence, flagged or not
    for a_, b_, name in windows:
        sel = [r.idx for r in C.results if a_ <= r.t < b_ and (r.winner is not None or r.word_share > 0.03)]
        owner_slots[name] = sorted(set(sel))
        for k in owner_slots[name]:
            flags.setdefault(k, [])
            if "owner_fact" not in flags[k]:
                flags[k].append("owner_fact")
    general = {k: v for k, v in flags.items() if not owner_key(C.results[k].t, windows)}
    runs = [(r, None) for r in group_spans(general, C, P.merge_words, P.max_slots)]
    runs += [(v, name) for name, v in owner_slots.items() if v]
    turn_words = [(w, t) for t in turns for w in t["words"]]
    spans = []
    for slots, ow in runs:
        reasons: dict[str, int] = {}
        details: list[str] = []
        for k in slots:
            for r in flags.get(k, []):
                reasons[r.split(" ")[0]] = reasons.get(r.split(" ")[0], 0) + 1
                if r not in details:
                    details.append(r)
        # time span: winners' times, else the slot time
        ts0, ts1 = [], []
        for k in slots:
            w = word_of_slot.get(k)
            if w is not None:
                ts0.append(w["start"])
                ts1.append(w["end"])
            else:
                tt = C.slots[k].t
                ts0.append(tt - 0.1)
                ts1.append(tt + 0.1)
        t0, t1 = min(ts0), max(ts1)
        if ow:
            wa, wb = next((a_, b_) for a_, b_, n_ in windows if n_ == ow)
            t0, t1 = max(t0, wa - 0.3), min(t1, wb + 0.3)
        readings = readings_for(C, slots, (wa, wb) if ow else None)
        cwords = [word_of_slot[k] for k in slots if k in word_of_slot]
        extra = [w for w in words if w["slot"] < 0 and t0 - 0.05 <= w["start"] <= t1 + 0.05]
        cwords = sorted(cwords + extra, key=lambda w: w["start"])
        cons_text = " ".join(w["text"] for w in cwords)
        spk = Counter()
        edge = False
        interj = False
        vconf = []
        for w, t in turn_words:
            if t0 - 0.05 <= 0.5 * (w["start"] + w["end"]) <= t1 + 0.05:
                spk[(t["voice"], t["speaker"])] += 1
                edge = edge or w.get("speaker_edge", False)
                interj = interj or w.get("interjection", False)
                vconf.append(w.get("speaker_conf", 0.0))
        if not spk:      # no consensus word: the turn around the time
            for t in turns:
                if t["start"] - 0.5 <= 0.5 * (t0 + t1) <= t["end"] + 0.5:
                    spk[(t["voice"], t["speaker"])] += 1
        speaker = [{"voice": v, "name": n, "words": c} for (v, n), c in spk.most_common()]
        sent_idx = [i for i, s in enumerate(sentences) if s["end"] >= t0 - 0.05 and s["start"] <= t1 + 0.05]
        if sent_idx:
            s_a, s_b = sent_idx[0], sent_idx[-1]
        else:
            s_a = s_b = max(0, sum(1 for s in sentences if s["end"] < t0) - 1)
        before = [s["text"] for s in sentences[max(0, s_a - 2): s_a]]
        after = [s["text"] for s in sentences[s_b + 1: s_b + 3]]
        here = " ".join(s["text"] for s in sentences[s_a: s_b + 1]) if sentences else ""
        rr = region_readings(region_hyps, t0, t1)
        listen = None
        if levels is not None:
            best, bs = None, -1e9
            for cam in levels.cams:
                if clocks.get(cam) is None:
                    continue
                snr, rel = levels.feats(cam, t0, t1)
                if snr + rel > bs:
                    best, bs = cam, snr + rel
            if best:
                clk = clocks[best]
                listen = {"phone": best, "master": [round(t0 - 0.3, 2), round(t1 + 0.3, 2)], "phone_seconds": [round(clk.to_source(t0 - 0.3), 2), round(clk.to_source(t1 + 0.3), 2)]}
        shares = [C.results[k].share for k in slots if C.results[k].winner is not None]
        nulls = [C.results[k].null_share for k in slots]
        prio = (1.0 - (min(shares) if shares else 0.0)) + 0.5 * max(nulls) + (0.5 if "suspect" in reasons else 0) + (1.0 if "owner_fact" in reasons else 0) \
            + (0.3 if any(r in reasons for r in ("overlap", "overlap_kept")) else 0)
        readings_out = readings[: P.max_readings]
        facts = [f["text"] for f in owner_facts if any(w["name"] == ow for w in f.get("windows", []))] if ow else []
        spans.append({
            "start": round(t0, 2), "end": round(t1, 2), "speaker": speaker, "speaker_edge": edge, "interjection": interj,
            "speaker_conf": round(float(np.mean(vconf)), 3) if vconf else None,
            "consensus_text": cons_text, "consensus_words": [{"text": w["text"], "share": w["share"], "start": w["start"]} for w in cwords],
            "sentence": here, "context_before": before, "context_after": after,
            "reasons": reasons, "reason_details": details, "priority": round(prio, 2),
            "candidates": readings_out, "n_candidates_total": len(readings), "targeted_decodes": rr,
            "glossary_hits": glossary_in_readings(readings_out + [{"norm": " ".join(w["norm"] for w in cwords)}], glossary, P),
            "slots": [min(slots), max(slots)], "n_slots": len(slots), "listen_to": listen, "owner_window": ow, "owner_facts": facts,
            "diarization_second_voice": [{"voice": j.get("voice"), "name": j.get("name"), "start": j["start"], "end": j["end"], "phones": j.get("cams"), "support": j.get("support"),
                                          "margin": j.get("margin"), "kind": j.get("kind")}
                                         for j in (diar_interjections or []) if j["start"] <= t1 + 0.1 and j["end"] >= t0 - 0.1],
        })
    spans.sort(key=lambda sp: (sp["start"], sp["end"]))
    return spans


def split_batches(spans: list[dict], P: BatchParams) -> list[list[dict]]:
    """Time-ordered spans in batches of 15-25 (as equal as possible)."""
    n = len(spans)
    if n == 0:
        return []
    k = max(1, round(n / P.target))
    while n / k > P.per_batch[1]:
        k += 1
    base, extra = divmod(n, k)
    out, pos = [], 0
    for i in range(k):
        m = base + (1 if i < extra else 0)
        out.append(spans[pos: pos + m])
        pos += m
    return out


EDITOR_BRIEF = (
    "You are the editor of a transcript of a 6-minute zouk class by two teachers, Larissa and Kadu (English with Brazilian-Portuguese move names). "
    "24 automatic transcripts (4 engines x 6 phones) were fused by word-level voting; every span below is a place where the vote did not settle. "
    "The consensus_text is stitched word by word and can be a reading that no engine wrote: prefer a whole-phrase candidate when they differ. "
    "For each span choose the reading that is most plausible FROM THE AUDIO EVIDENCE GIVEN (which engines/phones wrote it, their confidences, the glossary, the context), "
    "or write a better one only when the candidates together point at it. Never invent words that no candidate supports, except for glossary terms that sound like a candidate. "
    "Keep disfluencies and fillers that are in the audio. Do not 'correct' names or jokes (see owner_facts). Return one object per span with the span id, "
    "decision ('keep_consensus' | 'choose_candidate' | 'rewrite' | 'uncertain'), the final text, the candidate index if chosen, a confidence 0-1, a one-line reason, and "
    "needs_owner (true when only the owner can settle it)."
)


def write_batches(spans: list[dict], out_dir: Path, P: BatchParams, meta: dict) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("batch_*.json"):
        f.unlink()
    for i, s in enumerate(spans):
        s["id"] = f"d{i + 1:04d}"
    by_sentence: dict[str, list[str]] = defaultdict(list)
    for s in spans:
        if s.get("sentence"):
            by_sentence[s["sentence"]].append(s["id"])
    for s in spans:       # spans inside one sentence are one problem for the editor: decide them together
        s["same_sentence_spans"] = [x for x in by_sentence.get(s.get("sentence"), []) if x != s["id"]]
    batches = split_batches(spans, P)
    paths = []
    for bi, bs in enumerate(batches, 1):
        doc = {
            "schema": "dancecap.speech/1/disagreement_batch", "batch": bi, "of": len(batches), "take": meta.get("take"), "created": meta.get("created"),
            "clock": "master = 01-Recap-Tina video time (seconds)", "time_range": [bs[0]["start"], bs[-1]["end"]], "n_spans": len(bs),
            "brief": EDITOR_BRIEF,
            "legend": {"phones": "01 = master (Tina) ... 06; clock = master seconds", "engines": {v: k for k, v in fuse.ENGINE_SHORT.items()},
                       "sources": "phone/engine word-confidence (whisper probability, parakeet max-softmax; canary has none)", "weight_share": "share of the voting weight of the hypotheses that wrote this reading",
                       "targeted_decodes": "non-voting extra decodes of 4 hard windows (other prompts / gain), phone/engine variant: text"},
            "response_schema": {"id": "d0001", "decision": "keep_consensus | choose_candidate | rewrite | uncertain", "text": "final text of the span", "candidate": "index into candidates or null",
                                "confidence": "0-1", "reason": "one line", "needs_owner": "true | false"},
            "fusion": meta.get("fusion"), "spans": bs,
        }
        p = out_dir / f"batch_{bi:02d}.json"
        p.write_text(json.dumps(json_safe(doc), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        paths.append(p)
    return paths
