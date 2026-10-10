"""The 30-clip review list: where the owner can confirm "who is who" (and the two jokes) in about three minutes.

    python -m dancecap.speech review      # writes work/<take>/speech/review/{review_list.json, review_list.md, review.html, clips/*.wav}

Every clip is a few seconds of the phone that hears the voice best, cut from the 16 kHz audio (ffmpeg, local, a decode), with the master
timestamp, the system's answer (voice, name, confidence) and ONE sentence on why this clip is on the list. The clips hold the voices of real
people: they stay in the git-ignored work tree and are never published.

Selection (30 in total, filled in this order; a category that runs short is topped up from the biggest remaining ones):
  2  the opening, where the engines disagree about "Hello! This is ..." and "very red" (the owner's two jokes);
  5  + 5  the clearest solo stretches of each of the two voices, spread over the class (anchors: if these are right the naming is right);
  4  the addressing events that moved the naming most (a name said to / about someone, or who answered);
  4  interjections / overlaps found by Sortformer (a second voice inside a turn);
  4  the least certain turns and 3 the hardest passages by engine disagreement;
  3  voice changes where the boundary margin is smallest (is the boundary where we say it is?).
"""
from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

from . import SPEECH
from .audio import AUDIO, NO_WINDOW

REVIEW = SPEECH / "review"
MAX_CLIP = 7.0


def pick_spaced(items: list[dict], key, n: int, min_gap: float, t_key: str = "start", taken: list[float] | None = None) -> list[dict]:
    """The n best items by `key` (higher first) whose times are at least min_gap apart from each other and from `taken`."""
    taken = list(taken or [])
    out = []
    for it in sorted(items, key=key, reverse=True):
        if len(out) >= n:
            break
        if all(abs(it[t_key] - t) >= min_gap for t in taken):
            out.append(it)
            taken.append(it[t_key])
    return out


def clip_window(a: float, b: float, centre: float | None = None, max_len: float = MAX_CLIP, pad: float = 0.4) -> tuple[float, float]:
    """A clip span inside [a - pad, b + pad] of at most max_len seconds, centred on `centre` (default the middle)."""
    lo, hi = a - pad, b + pad
    if hi - lo <= max_len:
        return round(lo, 2), round(hi, 2)
    c = centre if centre is not None else 0.5 * (a + b)
    s = min(max(lo, c - max_len / 2), hi - max_len)
    return round(s, 2), round(s + max_len, 2)


def best_cam(doc: dict, a: float, b: float, prefer: str | None) -> str | None:
    """The phone to cut the clip from: `prefer` when it covers [a, b], else the master, else any covering phone with a clock."""
    clocks = doc["sources"]["clocks"]
    dur = doc.get("_durations", {})
    def covers(cam):
        c = clocks.get(cam)
        if not c or cam not in dur:
            return False
        return c["slope"] * 0 + c["offset"] <= a and c["slope"] * dur[cam] + c["offset"] >= b
    for cam in [prefer, "01", *sorted(clocks)]:
        if cam and covers(cam):
            return cam
    return None


def build(doc: dict, hard_windows: list[dict] | None = None, n_total: int = 30) -> list[dict]:
    turns = [t for t in doc["turns"] if t["voice"] is not None and t["voice"] >= 0]
    names = {str(v["voice"]): v.get("name") for v in doc["voices"]}
    teacher_voices = [int(k) for k, v in names.items() if v in ("Larissa", "Kadu")]
    clips: list[dict] = []

    def add(cat, a, b, reason, voice=None, conf=None, text=None, prefer=None, centre=None, extra=None, max_len=MAX_CLIP):
        t0, t1 = clip_window(a, b, centre, max_len)
        clips.append({"category": cat, "master_t0": t0, "master_t1": t1, "reason": reason, "voice": voice,
                      "name": names.get(str(voice)) if voice is not None else None, "confidence": conf, "text": text, "prefer_cam": prefer, **(extra or {})})

    # 1. the opening
    intro = max((i for i in doc.get("introduction_readings", []) if 0 <= i["start"] < 20), key=lambda i: i["n_readings"], default=None)
    laugh = next((e for e in doc["reactions"] if e["kind"] == "laughter"), None)
    add("opening", -0.5, 9.5, "The owner's two jokes ('introduces herself as Kadu', 'look at my red hair') should be here: the engines hear a different "
        "name / words every time, so please say what is said and by whom.", centre=4.5, max_len=10.0,
        extra={"readings": (intro or {}).get("readings")})
    if laugh:
        add("opening", laugh["t0"] - 2.5, laugh["t1"] + 1.5, "The only laughter the detector finds in the whole class (several phones agree): does the laugh follow the "
            "introduction, and who laughs?", centre=laugh["t0"] + 1.0)
    # 2. anchors per voice
    solo = [t for t in turns if t["end"] - t["start"] >= 5.0 and t["confidence"] >= 0.9 and t.get("overlap_s", 0) < 0.2]
    inter_times = [(i["start"], i["end"]) for i in doc.get("interjections", []) if i.get("kind") == "interjection_or_overlap"]
    for vid in teacher_voices:
        cands = []
        for t in solo:
            if t["voice"] != vid:
                continue
            length = t["end"] - t["start"]
            centres = [0.5 * (t["start"] + t["end"])] if length < 14 else [t["start"] + 0.25 * length, 0.5 * (t["start"] + t["end"]), t["start"] + 0.75 * length]
            for c in centres:
                a, b = max(t["start"], c - 3.5), min(t["end"], c + 3.5)
                if any(i0 < b and i1 > a for i0, i1 in inter_times):
                    continue          # a clean anchor has no second voice in it
                cands.append(dict(t, start=a, end=b, score=t["margin"] * (b - a), centre=c))
        for t in pick_spaced(cands, lambda x: x["score"], 5, 30.0, t_key="centre"):
            add("anchor", t["start"], t["end"], f"A clear solo stretch that the system gives to {names.get(str(vid)) or 'voice ' + str(vid)}: "
                f"if this is NOT {names.get(str(vid))}, the naming is wrong.", voice=vid, conf=t["confidence"], prefer=t["loudest_cam"])
    # 3. addressing events
    ev = []
    for e in doc["naming"].get("addressing", {}).get("events", []):
        ev.append(e)
    for e in sorted(ev, key=lambda x: -x["weight"])[:4]:
        add("addressing", e["t"] - 3.0, e["t"] + 3.5, f"A name is said here ({e['kind']} of {e['name'].capitalize()}, said by voice {e['speaker']}); it argues "
            f"'{e['supports']}'. Is the person addressed the one who answers?", voice=e["speaker"], centre=e["t"])
    # 4. interjections / overlap
    real = [i for i in doc.get("interjections", []) if i.get("kind") in ("interjection_or_overlap", "extra_turn") and not i.get("gap_fill")]
    for it in sorted(real, key=lambda x: -(x["support"] * (x["end"] - x["start"]) + (3.0 if x.get("kind") == "interjection_or_overlap" else 0)))[:4]:
        where = f"inside a turn of {it.get('base_name') or it['base_voice']}" if it.get("base_voice") is not None else "in a pause between the turns"
        add("interjection", it["start"] - 1.5, it["end"] + 1.5, f"A second voice ({it.get('name') or it['voice']}, {it['support']} phones) {where}: "
            f"is it really a different person, and who?", voice=it["voice"], centre=0.5 * (it["start"] + it["end"]))
    # 5. least certain turns
    unsure = [dict(t, score=-t["confidence"] * 10 - t["margin"]) for t in turns if t["end"] - t["start"] >= 2.0]
    for t in pick_spaced(unsure, lambda x: x["score"], 4, 30.0):
        add("uncertain", t["start"], t["end"], f"One of the least certain turns (margin {t['margin']:.2f}, confidence {t['confidence']:.2f}): who speaks?",
            voice=t["voice"], conf=t["confidence"], prefer=t["loudest_cam"])
    for w in (hard_windows or [])[:3]:
        add("hard", w["t0"] + 5, w["t0"] + 12, f"A passage where the ASR engines agree least ({w['agreement']:.2f} on {w['n']} words): helps to judge the fused transcript.")
    # 6. boundaries
    changes = []
    for a, b in zip(turns, turns[1:]):
        if a["voice"] != b["voice"] and b["start"] - a["end"] <= 1.0 and a["voice"] in teacher_voices and b["voice"] in teacher_voices:
            changes.append({"start": b["start"], "score": -min(a["margin"], b["margin"]), "a": a, "b": b})
    for c in pick_spaced(changes, lambda x: x["score"], 3, 30.0):
        add("boundary", c["start"] - 3.0, c["start"] + 3.0, f"A change of speaker at {c['start']:.1f} s with the smallest margin: does the voice change exactly here?",
            voice=c["b"]["voice"], centre=c["start"])
    # fill up to n_total from the remaining long turns
    have = {(c["category"], c["master_t0"]) for c in clips}
    if len(clips) < n_total:
        rest = [t for t in turns if t["end"] - t["start"] >= 2.0 and all(abs(0.5 * (t["start"] + t["end"]) - 0.5 * (c["master_t0"] + c["master_t1"])) > 8 for c in clips)]
        for t in pick_spaced([dict(t, score=t["n_moments"]) for t in rest], lambda x: x["score"], n_total - len(clips), 8.0):
            add("fill", t["start"], t["end"], f"An ordinary turn of {names.get(str(t['voice'])) or t['voice']} to complete the sample.", voice=t["voice"],
                conf=t["confidence"], prefer=t["loudest_cam"])
    return clips[:n_total]


def attach_text(clips: list[dict], utterances: list[dict]) -> None:
    for c in clips:
        us = [u for u in utterances if u["end"] >= c["master_t0"] and u["start"] <= c["master_t1"]]
        c["text"] = c.get("text") or " / ".join(f"[{u['name'] or u['voice']}] {u['text']}" for u in us)[:400]


def export(doc: dict, clips: list[dict], clocks: dict, out_dir: Path = REVIEW) -> None:
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for old in clip_dir.glob("c[0-9][0-9]_*.wav"):      # a rebuilt list must not leave the clips of the previous one behind
        old.unlink()
    for i, c in enumerate(clips, 1):
        c["id"] = f"c{i:02d}"
        cam = best_cam(doc, c["master_t0"], c["master_t1"], c.get("prefer_cam"))
        c["cam"] = cam
        if cam is None:
            c["file"] = None
            continue
        clk = clocks[cam]
        s0 = clk.to_source(c["master_t0"])
        src = next(AUDIO.glob(f"{cam}-Recap-*.16k.wav"), None)
        if src is None:
            c["file"] = None
            continue
        dst = clip_dir / f"{c['id']}_{c['category']}_{c['master_t0']:.1f}s_cam{cam}.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, s0):.3f}", "-t", f"{c['master_t1'] - c['master_t0']:.3f}", "-i", str(src),
                        "-c:a", "pcm_s16le", str(dst)], capture_output=True, creationflags=NO_WINDOW)
        c["file"] = dst.name


def write(doc: dict, clips: list[dict], out_dir: Path = REVIEW) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in c.items() if k != "prefer_cam"} for c in clips]
    (out_dir / "review_list.json").write_text(json.dumps({"note": "voices of real people: keep this folder out of every repository", "clips": slim}, indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    lines = ["# Who is who: 30 clips to confirm (master clock = 01-Recap-Tina)", "",
             "System answer (voice, name, confidence) is NOT certain; the clips decide. Play `clips/<file>` from the phone listed.", "",
             "| # | master time | phone | category | system says | why this clip | words (preview) |", "|---|---|---|---|---|---|---|"]
    for c in slim:
        who = f"{c['name'] or '-'} (voice {c['voice']}, conf {c['confidence']})" if c.get("voice") is not None else "-"
        lines.append(f"| {c['id']} | {c['master_t0']:.1f}-{c['master_t1']:.1f} s | {c['cam']} | {c['category']} | {who} | {c['reason']} | {(c.get('text') or '')[:160]} |")
    (out_dir / "review_list.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    for c in slim:
        audio = f'<audio controls preload="none" src="clips/{html.escape(c["file"])}"></audio>' if c.get("file") else "(no phone covers this time)"
        who = f"{c['name'] or '-'} / voice {c['voice']} / conf {c['confidence']}" if c.get("voice") is not None else "-"
        rows.append(f"<tr><td>{c['id']}</td><td>{c['master_t0']:.1f}-{c['master_t1']:.1f}</td><td>{c['cam']}</td><td>{html.escape(c['category'])}</td><td>{audio}</td>"
                    f"<td>{html.escape(who)}</td><td>{html.escape(c['reason'])}<br><small>{html.escape((c.get('text') or '')[:300])}</small></td>"
                    f"<td><select><option>?</option><option>Larissa</option><option>Kadu</option><option>both</option><option>someone else</option></select></td></tr>")
    page = ("<!doctype html><meta charset=utf-8><title>Who is who</title><style>body{font:14px system-ui;margin:16px}td,th{border-bottom:1px solid #8884;padding:6px;"
            "vertical-align:top}table{border-collapse:collapse}small{color:#888}</style><h1>Who is who: 30 clips</h1><p>Local file, voices of real people: never publish. "
            "Master clock = 01-Recap-Tina. The last column is only for you (it is not saved).</p><table><tr><th>#<th>master s<th>phone<th>category<th>play<th>system says"
            "<th>why<th>who is it?</tr>" + "".join(rows) + "</table>")
    (out_dir / "review.html").write_text(page, encoding="utf-8")


def run(doc: dict | None = None, n_total: int = 30) -> list[dict]:
    """Build, cut and write the review list from work/<take>/speech/diarization_global.json (+ report.json for the hardest passages)."""
    from . import report as report_mod
    from .sources import load_clocks
    doc = doc or json.loads((SPEECH / "diarization_global.json").read_text(encoding="utf-8"))
    raw = SPEECH / "raw"
    doc["_durations"] = {c: json.loads((raw / c / "vad.json").read_text(encoding="utf-8"))["duration"] for c in doc["sources"]["placed"]}
    rep = json.loads((SPEECH / "report.json").read_text(encoding="utf-8")) if (SPEECH / "report.json").exists() else {}
    hard = report_mod.hard_windows(rep.get("agreement", {})) if rep else []
    clips = build(doc, hard, n_total)
    attach_text(clips, doc["utterances"])
    export(doc, clips, load_clocks())
    write(doc, clips)
    return clips
