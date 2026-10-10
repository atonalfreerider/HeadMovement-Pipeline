"""S5: consistency checks of the exports of the final transcript (SRT, VTT, captions.json, chapters.json, teaching_events.json) against
transcript_final.json. Pure local reading; no pod. python -m dancecap.speech.audit_struct -> audit_s5/struct_checks.json
"""
from __future__ import annotations

import json
import re
import sys

from . import SPEECH
from .audit import AUDIT

EMOJI = "\U0001F602"


def load(name: str):
    return json.loads((SPEECH / name).read_text(encoding="utf-8"))


def parse_srt(text: str) -> list[dict]:
    out = []
    for blk in re.split(r"\n\s*\n", text.strip()):
        lines = blk.strip().split("\n")
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+) --> (\d+):(\d+):(\d+)[,.](\d+)", lines[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        out.append({"i": lines[0], "start": g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000, "end": g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                    "text": "\n".join(lines[2:])})
    return out


def parse_vtt(text: str) -> list[dict]:
    out = []
    for blk in re.split(r"\n\s*\n", text.strip()):
        lines = blk.strip().split("\n")
        k = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if k is None:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+) --> (\d+):(\d+):(\d+)[,.](\d+)", lines[k])
        g = [int(x) for x in m.groups()]
        out.append({"start": g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000, "end": g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000, "text": "\n".join(lines[k + 1:])})
    return out


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9' \U0001F602]", "", s.lower().replace("\n", " ")).strip()


def run() -> dict:
    final = load("transcript_final.json")
    caps_doc = load("captions.json")
    caps = caps_doc["captions"]
    chap = load("chapters.json")
    ev = load("teaching_events.json")
    srt = parse_srt((SPEECH / "transcript_final.srt").read_text(encoding="utf-8"))
    vtt = parse_vtt((SPEECH / "transcript_final.vtt").read_text(encoding="utf-8"))
    dur = float(chap["master_duration_s"])
    problems: list[str] = []
    info: dict = {"master_duration_s": dur}

    # --- SRT / VTT
    info["srt_cues"], info["vtt_cues"], info["captions"] = len(srt), len(vtt), len(caps)
    if not (len(srt) == len(vtt) == len(caps)):
        problems.append(f"cue counts differ: srt {len(srt)} vtt {len(vtt)} captions {len(caps)}")
    bad_t = [c["i"] for c in srt if c["start"] < -1e-6 or c["end"] > dur + 1e-6 or c["end"] <= c["start"]]
    info["srt_cues_outside_0_to_duration_or_empty"] = bad_t
    if bad_t:
        problems.append(f"SRT cues with times outside 0..{dur:.3f} or non-positive length: {bad_t}")
    mism_time = [i for i, (a, b) in enumerate(zip(srt, vtt)) if abs(a["start"] - b["start"]) > 0.002 or abs(a["end"] - b["end"]) > 0.002]
    mism_text = []
    for i, (a, c) in enumerate(zip(srt, caps)):
        t_srt = norm(re.sub(r"^[^:]+:\s*", "", a["text"]))
        t_cap = norm(c["text"])
        if t_srt != t_cap:
            mism_text.append((i, t_srt, t_cap))
    info["srt_vtt_time_mismatch"] = mism_time
    info["srt_caption_text_mismatch"] = mism_text[:10]
    if mism_time:
        problems.append(f"SRT and VTT times differ in cues {mism_time[:10]}")
    if mism_text:
        problems.append(f"SRT and captions.json text differ in {len(mism_text)} cues, first {mism_text[:2]}")
    starts = [c["start"] for c in srt]
    if starts != sorted(starts):
        problems.append("SRT cue starts are not monotone")
    # line rule
    too_long = []
    for c in caps:
        if len(c["lines"]) > 2 or any(len(l) - l.count(EMOJI) * 0 > 42 + 2 * l.count(EMOJI) for l in c["lines"]):
            too_long.append(c["id"])
    info["captions_violating_2x42"] = too_long
    if too_long:
        problems.append(f"captions over 2 lines / 42 characters: {too_long}")
    long_show = [(c["id"], round(c["show_end"] - c["show_start"], 2)) for c in caps if c["show_end"] - c["show_start"] > 6.5 + 1e-6]
    short_show = [(c["id"], round(c["show_end"] - c["show_start"], 2)) for c in caps if c["show_end"] - c["show_start"] < 0.3]
    info["captions_longer_than_6.5s"] = long_show
    info["captions_shorter_than_0.3s"] = short_show

    # --- overlaps
    same, cross = [], []
    for a, b in zip(caps, caps[1:]):
        ov = a["show_end"] - b["show_start"]
        if ov > 0.02:
            (same if a["speaker"] == b["speaker"] else cross).append((a["id"], b["id"], round(ov, 2), a["speaker"], b["speaker"]))
    # captions are sorted by start; a long caption can overlap a non-neighbour, check pairwise too
    pair_same = []
    for i, a in enumerate(caps):
        for b in caps[i + 1:]:
            if b["show_start"] >= a["show_end"]:
                break
            if a["speaker"] == b["speaker"] and a["show_end"] - b["show_start"] > 0.02:
                pair_same.append((a["id"], b["id"], round(a["show_end"] - b["show_start"], 2)))
    info["overlaps_same_speaker"] = pair_same
    info["overlaps_cross_speaker"] = cross
    if pair_same:
        problems.append(f"captions of the SAME speaker overlap in show time: {pair_same[:5]}")
    # show_end of a caption vs its last word end
    lag = [(c["id"], round(c["show_end"] - c["end"], 2)) for c in caps if c["show_end"] - c["end"] > 1.0]
    info["captions_shown_more_than_1s_after_last_word"] = lag

    # --- every transcript word (except fillers and pre-roll) in exactly one caption, in order
    exp = []
    for u in final["utterances"]:
        for w in u["words"]:
            if w["kind"] == "filler" or w["start"] < 0.0:       # captions rule: fillers and words that start before master 0 (the film's start) are not captions
                continue
            exp.append((norm(w["text"] + w["punct"]) if w["kind"] != "event" else EMOJI, round(w["start"], 3)))
    got = []
    for c in caps:
        for w in c["words"]:
            got.append((norm(w["text"]) if w.get("kind") != "emoji" else EMOJI, round(w["start"], 3)))
    n_exp, n_got = len(exp), len(got)
    info["words_expected_in_captions"], info["words_in_captions"] = n_exp, n_got
    miss = [e for e in exp if e not in got]
    extra = [g for g in got if g not in exp]
    info["words_missing_from_captions"] = miss[:10]
    info["words_not_in_transcript"] = extra[:10]
    if miss or extra:
        problems.append(f"captions vs transcript words: missing {len(miss)}, extra {len(extra)} (first {miss[:3]} / {extra[:3]})")
    t_words = [w for c in caps for w in c["words"] if w["start"] < -1e-6 or w["end"] > dur + 1e-6]
    if t_words:
        problems.append(f"caption words outside 0..{dur:.2f}: {t_words[:3]}")
    emoji_caps = [(c["id"], w["start"]) for c in caps for w in c["words"] if w.get("kind") == "emoji"]
    info["emoji_in_captions"] = emoji_caps

    # --- chapters
    chs = chap["chapters"]
    gaps = []
    if abs(chs[0]["start"]) > 1e-6:
        gaps.append(f"first chapter starts at {chs[0]['start']}")
    for a, b in zip(chs, chs[1:]):
        if abs(a["end"] - b["start"]) > 1e-6:
            gaps.append(f"{a['id']}.end {a['end']} != {b['id']}.start {b['start']}")
    if abs(chs[-1]["end"] - dur) > 1e-3:
        gaps.append(f"last chapter ends at {chs[-1]['end']} not {dur}")
    info["chapter_gaps"] = gaps
    problems += [f"chapters: {g}" for g in gaps]
    utt = {u["id"]: u for u in final["utterances"]}
    used = [x for c in chs for x in c["utterances"]]
    if sorted(used) != sorted(utt) or len(used) != len(set(used)):
        problems.append(f"chapters do not hold every utterance exactly once: {len(used)} listed vs {len(utt)}")
    off_start = []
    for c in chs:
        first = utt[c["utterances"][0]]
        if abs(first["start"] - c["start"]) > 0.02 and c["start"] != 0.0:
            off_start.append((c["id"], c["start"], first["start"]))
        if len(c["title"].split()) > 5:
            problems.append(f"chapter {c['id']} title longer than 5 words")
        lastu = utt[c["utterances"][-1]]
        if lastu["end"] > c["end"] + 0.5:
            problems.append(f"chapter {c['id']} ends before its last utterance ends ({lastu['end']} > {c['end']})")
    info["chapter_start_not_on_utterance_start"] = off_start
    wsum = sum(c["n_words"] for c in chs)
    info["chapter_n_words_sum"] = wsum
    info["transcript_n_words"] = sum(u["n_words"] for u in final["utterances"])

    # --- teaching events
    bad_ev = []
    for e in ev["events"]:
        u = utt.get(e["utterance"])
        if u is None or not (u["start"] - 0.05 <= e["start"] <= u["end"] + 0.05) or e["end"] > dur + 1e-6 or e["start"] < 0:
            bad_ev.append(e["id"])
        elif e["speaker"] != u["speaker"]:
            bad_ev.append(e["id"] + ":speaker")
    info["events_with_bad_time_or_speaker"] = bad_ev
    if bad_ev:
        problems.append(f"teaching events outside their utterance / master range: {bad_ev[:10]}")
    out = {"problems": problems, "info": info, "ok": not problems}
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "struct_checks.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=1, ensure_ascii=False)[:6000])
    sys.exit(0 if r["ok"] else 1)
