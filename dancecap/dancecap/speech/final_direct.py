"""S4: from the final transcript to what directs the film: chapters, the teaching outline (moments a 3D film could show), jokes and laughter candidates.

    transcript_final.json (+ final_overrides.json: curated chapters / teaching events / joke candidates)  ->  chapters.json, teaching_events.json, jokes.json

The teaching outline is the bridge from the transcript to the film: every moment where the teachers explicitly name something a 3D film can show (a
body part, the axis, a weight change, a step, a direction, a count, a move name, "again", "slow") with a PROPOSED visualisation tag. Rules (RULES) propose
the events from the words; final_overrides.json["teaching_events"] drops, retags or adds by phrase and time. Nothing here is a decision of the film: the
tags are proposals for the director step.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import SCHEMA, SPEECH, TAKE
from .final_merge import mmss
from .fuse_post import json_safe

TAGS = ["camera_focus", "overlay_T_axis", "overlay_weight_change", "overlay_footsteps/physics", "overlay_timing", "overlay_path/floorcraft", "slow_motion", "replay",
        "graph_moves", "caption_only"]

# (kind, regex over the canonical words joined by single spaces, proposed tags in order of preference, focus part)
RULES: list[tuple[str, str, list[str], str | None]] = [
    ("body_part", r"\b(?:arms?)\b", ["camera_focus"], "arm"),
    ("body_part", r"\b(?:shoulders?|shoulder blades|blades)\b", ["camera_focus"], "shoulders"),
    ("body_part", r"\bhips?\b", ["camera_focus", "overlay_weight_change"], "hip"),
    ("body_part", r"\b(?:open )?chest\b", ["camera_focus"], "chest"),
    ("body_part", r"\bupper body\b", ["camera_focus"], "upper body"),
    ("body_part", r"\b(?:legs?|thighs?|knees?)\b", ["camera_focus", "overlay_footsteps/physics"], "legs"),
    ("body_part", r"\b(?:foot|feet)\b", ["overlay_footsteps/physics", "camera_focus"], "feet"),
    ("body_part", r"\bhands?\b", ["camera_focus"], "hands"),
    ("body_part", r"\bframe\b", ["camera_focus"], "frame"),
    ("body_part", r"\bconnect(?:ion|ed|ing)?\b", ["camera_focus"], "connection"),
    ("axis", r"\baxis\b", ["overlay_T_axis"], None),
    ("axis", r"\boff cent(?:er|ered|re|red)\b", ["overlay_T_axis", "overlay_path/floorcraft"], None),
    ("axis", r"\bexits?\b", ["overlay_T_axis", "graph_moves"], None),
    ("axis", r"\brotat(?:e|es|ed|ing|ion|ions)\b", ["overlay_T_axis", "graph_moves"], None),
    ("axis", r"\bpivot(?:s|ing|ed)?\b", ["overlay_T_axis", "overlay_footsteps/physics"], None),
    ("weight", r"\b(?:weights?|heavy|suspension|balance|counterbalance|shift(?:s|ing)?)\b", ["overlay_weight_change"], None),
    ("footwork", r"\b(?:footwork|steps?|stepping|lunge|walk(?:ing)?|plant(?:ed)?)\b", ["overlay_footsteps/physics"], "feet"),
    ("timing", r"\b(?:momentum|speed|energy|at the same time|pause|stop)\b", ["overlay_timing"], None),
    ("timing", r"\b(?:slow|slowing|slower)\b", ["slow_motion", "overlay_timing"], None),
    ("path", r"\b(?:space|side to side|walk(?:ing)? around|around|forward|backward|too far)\b", ["overlay_path/floorcraft"], None),
    ("direction", r"\b(?:right side|left side|to the right|to the left|on the left|on the right|right foot|left foot)\b", ["overlay_path/floorcraft", "overlay_footsteps/physics"], None),
    ("replay", r"\b(?:again|one more time|once again|once more|restart(?:ed)?)\b", ["replay"], None),
    ("enumeration", r"\b(?:first|second|third|last) (?:option|exit|part|one)\b", ["graph_moves", "caption_only"], None),
    ("move_name", r"\b(?:lateral|sacada|bonus|gostosinho|simple turn|lunge|basic|prep)\b", ["graph_moves", "caption_only"], None),
    ("demonstration", r"\b(?:like this|this way|that way|do this|from here|starting here|do here|here you can|look at it|watch)\b", ["camera_focus", "replay"], None),
]
SUBJECT_RX = [("leader", r"\b(?:leaders?|lead|i)\b"), ("follower", r"\b(?:followers?|follow|her)\b")]


# ------------------------------------------------------------------------------------------------ helpers


def spoken_words(final: dict) -> list[dict]:
    """Every spoken (non-event, non-filler) word of the final transcript in order, with its utterance and speaker."""
    out = []
    for u in final["utterances"]:
        for w in u["words"]:
            if w["kind"] == "word":
                out.append(dict(w, utt=u["id"], speaker=u["speaker"], turn=u["turn"]))
    return out


def phrase_times(words: list[dict], norm_phrase: str, after: float = -1e9, nth: int = 0) -> tuple[int, int] | None:
    """(first word index, last word index) of the nth occurrence of a phrase (canonical words) at or after `after` seconds."""
    target = norm_phrase.split()
    k = len(target)
    hits = [i for i in range(len(words) - k + 1) if words[i]["start"] >= after and [w["norm"] for w in words[i: i + k]] == target]
    if len(hits) <= nth:
        return None
    return hits[nth], hits[nth] + k - 1


def snap_start(utts: list[dict], t: float, tol: float = 1.0) -> float:
    """The start of the utterance nearest to t (within tol), else t."""
    best = min(utts, key=lambda u: abs(u["start"] - t))
    return best["start"] if abs(best["start"] - t) <= tol else t


# ------------------------------------------------------------------------------------------------ chapters


def build_chapters(final: dict, overrides: dict, master_duration: float) -> dict:
    """chapters.json: curated chapters ([{start, title, summary, terms}] in final_overrides.json) snapped to utterance starts, contiguous from 0 to the end of the master."""
    utts = [u for u in final["utterances"] if u["end"] > 0]
    spec = sorted(overrides.get("chapters", []), key=lambda c: c["start"])
    out = []
    opening = overrides.get("owner_opening") or {}
    for k, c in enumerate(spec):
        start = 0.0 if k == 0 else snap_start(utts, c["start"])
        row = dict(c, start=round(start, 3))
        if k == 0 and opening.get("chapter_summary"):          # the owner heard the opening: his summary replaces the curated one (owner_opening.json)
            row["summary"], row["summary_source"] = str(opening["chapter_summary"]), "owner_opening"
            if opening.get("chapter_title"):
                row["title"] = str(opening["chapter_title"])
        out.append(row)
    for k, c in enumerate(out):
        c["end"] = round(out[k + 1]["start"] if k + 1 < len(out) else master_duration, 3)
        c["id"] = f"ch{k:02d}"
    checks = []
    allw = spoken_words(final)
    for k, c in enumerate(out):
        us = [u for u in utts if (c["start"] - 1e-6 <= u["start"] < c["end"] - 1e-6) or (k == 0 and u["start"] < c["end"] - 1e-6)]
        ws = [w for w in allw if (c["start"] - 1e-6 <= w["start"] < c["end"]) or (k == 0 and w["start"] < c["end"])]
        secs: Counter = Counter()
        for u in us:
            secs[u["speaker"]] += max(0.0, min(u["end"], c["end"]) - max(u["start"], c["start"]))
        text = " ".join(w["norm"] for w in ws)
        terms = c.get("terms", [])
        missing = [t for t in terms if t.lower().replace("-", " ") not in text and t.lower() not in " ".join(w["text"].lower() for w in ws)]
        c["title_words"] = len(c["title"].split())
        c["utterances"] = [u["id"] for u in us]
        c["speakers"] = {s: round(v, 1) for s, v in secs.most_common()}
        c["n_words"] = len(ws)
        c["conf_mean"] = round(sum(w["conf"] for w in ws) / len(ws), 3) if ws else None
        c["needs_owner_utterances"] = [u["id"] for u in us if "needs_owner" in u["flags"]]
        c["terms_not_in_text"] = missing
        if c["title_words"] > 5:
            checks.append(f"{c['id']}: title has {c['title_words']} words")
        if missing:
            checks.append(f"{c['id']}: key terms not found in the chapter text: {missing}")
    if not 8 <= len(out) <= 15:
        checks.append(f"{len(out)} chapters (expected 8-15)")
    doc = {
        "schema": SCHEMA + "/chapters", "take": TAKE, "clock": "master = 01-Recap-Tina video time, seconds; chapter 0 starts at 0 (the film start), the last ends at the master's end",
        "master_duration_s": master_duration, "n_chapters": len(out), "chapters": out, "checks": checks,
        "note": "titles are plain English with pt-BR terms where the teachers use them; starts are snapped to sentence starts of the final transcript; summaries are written from the transcript and "
                "inherit its unclear spans (see needs_owner_utterances)",
    }
    return doc


def chapter_of(chapters: list[dict], t: float) -> str | None:
    for c in chapters:
        if c["start"] - 1e-6 <= t < c["end"]:
            return c["id"]
    return chapters[-1]["id"] if chapters and t >= chapters[-1]["end"] else None


# ------------------------------------------------------------------------------------------------ teaching events


def match_rules(words: list[dict]) -> list[dict]:
    """One match per rule hit: trigger words, times, kind, proposed tags, focus part."""
    norms = [w["norm"] for w in words]
    text = " ".join(norms)
    starts, pos = [], 0
    for n in norms:
        starts.append(pos)
        pos += len(n) + 1
    out = []
    for kind, rx, tags, part in RULES:
        for m in re.finditer(rx, text):
            i0 = max(i for i, p in enumerate(starts) if p <= m.start())
            i1 = max(i for i, p in enumerate(starts) if p < m.end())
            seg = words[i0: i1 + 1]
            if len({w["utt"] for w in seg}) != 1:
                continue
            out.append({"kind": kind, "trigger": " ".join(norms[i0: i1 + 1]), "i0": i0, "i1": i1, "start": seg[0]["start"], "end": seg[-1]["end"], "tags": list(tags), "part": part,
                        "utt": seg[0]["utt"], "speaker": seg[0]["speaker"], "conf": min(w["conf"] for w in seg)})
    out.sort(key=lambda e: (e["start"], e["i1"]))
    return out


ROLE_OF = {"Kadu": "leader", "Larissa": "follower"}      # takes/larissa_kadu_recap.toml: Kadu = lead, Larissa = follow, as in the demo


def _subject(sentence_norms: list[str], speaker: str | None) -> str:
    """Whose body the film should look at: the roles named in the sentence, plus the speaker's own role when the sentence is first person ("my arm", "I bring her"); both = the couple."""
    txt = " ".join(sentence_norms)
    lead = bool(re.search(r"\b(?:leaders?|lead|his)\b", txt))
    foll = bool(re.search(r"\b(?:followers?|follow|her)\b", txt))
    if re.search(r"\b(?:i|my|me|im|ive|ill)\b", txt) and speaker in ROLE_OF:
        lead, foll = lead or ROLE_OF[speaker] == "leader", foll or ROLE_OF[speaker] == "follower"
    if lead and foll:
        return "couple"
    return "follower" if foll else ("leader" if lead else "couple")


def _merge(a: dict, b: dict) -> None:
    """Fold event b into a: union of triggers, kinds, parts and tags (order kept), the longer time span, the lower confidence."""
    a["start"], a["end"] = min(a["start"], b["start"]), max(a["end"], b["end"])
    a["i1"] = max(a["i1"], b["i1"])
    for t in b["triggers"]:
        if t not in a["triggers"]:
            a["triggers"].append(t)
    for k in b["kinds"]:
        if k not in a["kinds"]:
            a["kinds"].append(k)
    for t in b["tags"]:
        if t not in a["tags"]:
            a["tags"].append(t)
    for p_ in b["parts"]:
        if p_ not in a["parts"]:
            a["parts"].append(p_)
    a["conf"] = min(a["conf"], b["conf"])
    a["curated"] = bool(a.get("curated") or b.get("curated"))
    a["note"] = a.get("note") or b.get("note")


def build_teaching_events(final: dict, overrides: dict, chapters: list[dict]) -> dict:
    words = spoken_words(final)
    utt_words: dict[str, list[dict]] = defaultdict(list)
    utt_text: dict[str, str] = {u["id"]: u["text"] for u in final["utterances"]}
    for w in words:
        utt_words[w["utt"]].append(w)
    matches = [m for m in match_rules(words) if m["end"] > 0.0]          # nothing before the film starts (master 0)
    cur = overrides.get("teaching_events") or {}

    def near(e: dict, spec: dict) -> bool:
        return abs(e["start"] - float(spec["near"])) <= float(spec.get("tol", 0.8)) and ("trigger" not in spec or spec["trigger"] == e["trigger"] or spec["trigger"] in e["trigger"])             and ("kind" not in spec or spec["kind"] == e["kind"])

    dropped = 0
    keep = []
    for e in matches:
        if any(near(e, d) for d in cur.get("drop", [])):
            dropped += 1
            continue
        for r in cur.get("retag", []):
            if near(e, r):
                e["tags"] = list(r["tags"])
                e["curated"] = True
                if "note" in r:
                    e["note"] = r["note"]
        e["triggers"], e["kinds"], e["parts"] = [e["trigger"]], [e["kind"]], ([e["part"]] if e.get("part") else [])
        keep.append(e)
    # events of one utterance that overlap in time (or are the same kind less than 1 s apart) are one moment: "second exit" (enumeration) + "exit" (axis)
    merged: list[dict] = []
    for e in keep:
        prev = next((m for m in reversed(merged) if m["utt"] == e["utt"] and (e["start"] < m["end"] + 0.15 and e["end"] > m["start"] - 0.15
                                                                              or (m["kinds"] == e["kinds"] and e["start"] - m["end"] < 1.0 and m["tags"] == e["tags"]))), None)
        if prev is not None:
            _merge(prev, e)
        else:
            merged.append(e)
    added = 0
    for a in cur.get("add", []):
        hit = phrase_times(words, a["phrase"], float(a.get("after", -1e9)), int(a.get("nth", 0)))
        if hit is None:
            continue
        seg = words[hit[0]: hit[1] + 1]
        merged.append({"kind": a.get("kind", "demonstration"), "kinds": [a.get("kind", "demonstration")], "trigger": a["phrase"], "triggers": [a["phrase"]], "i0": hit[0], "i1": hit[1],
                       "start": seg[0]["start"], "end": seg[-1]["end"], "tags": list(a["tags"]), "part": a.get("part"), "parts": [a["part"]] if a.get("part") else [], "utt": seg[0]["utt"],
                       "speaker": seg[0]["speaker"], "conf": min(w["conf"] for w in seg), "note": a.get("note"), "curated": True})
        added += 1
    merged.sort(key=lambda e: (e["start"], e["i1"]))
    events = []
    for k, e in enumerate(merged):
        tags = [t for t in e["tags"] if t in TAGS]
        if len(tags) > 1 and "caption_only" in tags:
            tags.remove("caption_only")                 # a caption alone is the fallback: drop it when something can be drawn
        subj = _subject([w["norm"] for w in utt_words[e["utt"]]], e["speaker"])
        events.append({
            "id": f"te{k:03d}", "start": round(e["start"], 3), "end": round(e["end"], 3), "speaker": e["speaker"], "utterance": e["utt"], "chapter": chapter_of(chapters, e["start"]),
            "kinds": e["kinds"], "triggers": e["triggers"], "text": utt_text[e["utt"]], "tags": tags, "primary_tag": tags[0] if tags else "caption_only",
            "focus": {"who": subj, "where": (e["parts"] or [None])[0], "speaker": e["speaker"]},
            "conf": round(e["conf"], 3), "needs_check": e["conf"] < 0.6, "note": e.get("note"), "curated": bool(e.get("curated")),
        })
    by_tag = Counter(t for e in events for t in e["tags"])
    by_kind = Counter(k for e in events for k in e["kinds"])
    return {
        "schema": SCHEMA + "/teaching_events", "take": TAKE, "clock": "master = 01-Recap-Tina video time, seconds (the film starts at 0)",
        "what": "Moments where a teacher explicitly refers to something a 3D film could show, with a PROPOSED visualisation tag; proposals for the director step, not decisions.",
        "tags": TAGS, "n_events": len(events), "by_tag": dict(by_tag), "by_kind": dict(by_kind), "dropped_by_curation": dropped, "added_by_curation": added,
        "roles": {"Kadu": "leader", "Larissa": "follower", "note": "takes/larissa_kadu_recap.toml: Kadu = lead, Larissa = follow"}, "events": events,
        "tag_meaning": {
            "camera_focus": "put the camera on the person / body part named (focus.who = leader | follower | couple, focus.where = the body part)",
            "overlay_T_axis": "show the leader's T axis / the exit and rotation record (floorcraft t_axes, axis_pivots)",
            "overlay_weight_change": "show the weight / counterbalance dial or the weight transfer",
            "overlay_footsteps/physics": "show footsteps, plants and the physics of the step",
            "overlay_timing": "show the timing / momentum (speed, pause, simultaneity)",
            "overlay_path/floorcraft": "show the floor path, space, direction of travel",
            "slow_motion": "slow the playback while the teacher talks about slow / slowing down",
            "replay": "replay the passage ('again', 'one more time', 'like this')",
            "graph_moves": "show the move graph / name the move",
            "caption_only": "nothing to draw: a caption (a name or an enumeration) is enough",
        },
    }


# ------------------------------------------------------------------------------------------------ jokes


def build_jokes(final: dict, consensus: dict, overrides: dict, ast_peaks: list[dict]) -> dict:
    """jokes.json: the two owner-stated jokes (with the evidence found and what only the hint provided), the emoji tokens placed, and other laughter / humour candidates."""
    tokens = []
    flat = [w for u in final["utterances"] for w in u["words"]]
    for e in final["events"]:
        pos = next((k for k, w in enumerate(flat) if w["kind"] == "event" and abs(w["start"] - e["start"]) < 1e-6 and w.get("span") == e.get("span")), 0)
        before = [w for w in flat[:pos] if w["kind"] == "word"][-6:]                       # reading order: the words the token follows
        tokens.append({"token": e["text"], "t0": e["start"], "t1": e["end"], "after_words": " ".join(w["text"] for w in before), "flags": sorted(set(e["flags"]) | {"joke"}), "joke": e.get("joke"),
                       "evidence": e.get("evidence"), "span": e.get("span"), "needs_owner": "needs_owner" in e["flags"], "owner_given": "owner_opening" in e["flags"]})
    cands = []
    for c in overrides.get("joke_candidates", []):
        cands.append(dict(c, flags=sorted(set(c.get("flags", [])) | {"laugh_candidate"})))
    spoken = [w for u in final["utterances"] for w in u["words"] if w["kind"] == "word"]
    explained = []
    seen = {(round(c["t0"], 1), round(c["t1"], 1)) for c in cands}
    for s in consensus.get("wordless_sounds", []):
        key = (round(s["t0"], 1), round(s["t1"], 1))
        if key in seen or any(c["t0"] - 0.05 <= s["t0"] and s["t1"] <= c["t1"] + 0.05 for c in cands):       # already a candidate (or inside one)
            continue
        inside = [w["text"] for w in spoken if w["start"] < s["t1"] + 0.25 and w["end"] > s["t0"] - 0.25]
        if inside:
            tok = [e for e in final.get("events", []) if s["t0"] - 0.5 <= e["start"] <= s["t1"] + 0.5]
            explained.append({"t0": s["t0"], "t1": s["t1"], "wordless_db": s["peak_excess_db"], "phones": s["phones"], "words_in_final": inside,
                              "near_laugh_token": [e.get("joke") for e in tok] or None,
                              "why": "a loud stretch the vote left empty; the adjudicated transcript has speech there, so it is not a laugh candidate"
                                     + ("; a laugh token sits next to it (the reaction and the second voice overlap here)" if tok else "")})
            continue
        cands.append({"id": f"auto_w{len(cands):02d}", "t0": s["t0"], "t1": s["t1"], "speaker": None, "text": None, "flags": ["laugh_candidate", "auto"],
                      "why": f"loud stretch without a word in the final transcript: +{s['peak_excess_db']} dB over the room on {s['phones']} phone(s)",
                      "evidence": {"wordless_db": s["peak_excess_db"], "phones": s["phones"]}})
    for p in ast_peaks:
        key = (round(p["t0"], 1), round(p["t1"], 1))
        if not any(abs(c["t0"] - p["t0"]) < 1.5 for c in cands):
            cands.append({"id": f"auto_a{len(cands):02d}", "t0": p["t0"], "t1": p["t1"], "speaker": None, "text": None, "flags": ["laugh_candidate", "auto"],
                          "why": f"AudioSet laughter / applause tags {p['peak']} on {p['phones']} phone(s) (weak)", "evidence": p})
    cands.sort(key=lambda c: c["t0"])
    owner_tokens = [t for t in tokens if t["owner_given"]]
    for k, c in enumerate(cands):
        c["id"] = c.get("id") if c.get("id") and not str(c["id"]).startswith("auto_") else f"jc{k:02d}"
        c.setdefault("approve_emoji", None)
        if c["approve_emoji"] is None and any(c["t0"] - 0.5 <= t["t0"] <= c["t1"] + 0.5 for t in owner_tokens):
            c["approve_emoji"] = True                       # the owner asked for a laugh token inside this candidate (owner_opening.json)
    return {
        "schema": SCHEMA + "/jokes", "take": TAKE, "clock": "master = 01-Recap-Tina video time, seconds",
        "owner_facts": overrides.get("owner_jokes", []), "found_by_pipeline": overrides.get("joke_findings", {}),
        "tokens_in_transcript": tokens, "laugh_candidates": cands, "explained_wordless_sounds": explained,
        "note": "tokens_in_transcript hold the U+1F602 tokens placed after the two jokes the owner stated; laugh_candidates are NOT in the text: approve one by setting approve_emoji to true.",
    }


def ast_peaks(raw_dir: Path, clocks: dict, thr: float = 0.05, min_phones: int = 2) -> list[dict]:
    """Windows of the AudioSet tagging where a laughter-family or applause-family class peaks above `thr`, on the master clock (candidates for the owner)."""
    fam_l = ["Laughter", "Snicker", "Giggle", "Chuckle, chortle", "Belly laugh", "Baby laughter"]
    fam_a = ["Applause", "Clapping", "Cheering", "Whoop"]
    rows: dict[int, list] = {}
    for cam, clk in clocks.items():
        p = Path(raw_dir) / cam / "events.json"
        if clk is None or not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        for w in doc.get("windows", []):
            pr = w.get("p", {})
            lv = max([pr.get(k, 0.0) for k in fam_l] or [0.0])
            ap = max([pr.get(k, 0.0) for k in fam_a] or [0.0])
            t = clk.to_master(0.5 * (w["t0"] + w["t1"]))
            rows.setdefault(int(round(t)), []).append((cam, lv, ap))
    out = []
    for t, v in sorted(rows.items()):
        lv = max(x[1] for x in v)
        ap = max(x[2] for x in v)
        n = sum(1 for x in v if x[1] >= thr or x[2] >= thr)
        if max(lv, ap) >= thr and n >= min_phones:
            out.append({"t0": t - 1.0, "t1": t + 1.0, "peak": round(max(lv, ap), 3), "phones": n, "family": "laughter" if lv >= ap else "applause / cheering"})
    return out
