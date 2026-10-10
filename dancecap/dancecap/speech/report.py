"""Per-source and per-engine statistics that need no reference transcript, language findings, and the readable side-by-side of engines.

    python -m dancecap.speech report       # writes work/<take>/speech/report.json

* engine table: words, words per speech minute, mean word probability, share of low-probability words (< 0.5), per phone and engine;
* agreement: for two engines on the same phone (or one engine on two phones) the share of words that match after alignment, in 30-s
  master-clock windows (difflib on normalised tokens). It is a WER-free proxy: two independent engines (or phones) that agree on a
  passage are very probably right, so the windows where they do NOT agree are the hard passages, and the average over a phone says
  how much that phone's audio costs. 1 - agreement is NOT a word error rate;
* language: Whisper's language id per VAD region (the share of speech time per language and per phone) and how the three decoding
  passes differ;
* quality: the pod's quality.json per phone (SNR, level, clipping, codec band) next to the above.
"""
from __future__ import annotations

import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .texts import Word, engine_keys, load_engine, norm_token


def tokens(ws: list[Word]) -> list[str]:
    return [t for t in (norm_token(w.text) for w in ws) if t]


def similarity(a: list[str], b: list[str]) -> tuple[float, int]:
    """(matched / max(len), max(len)) between two token lists: 1.0 identical, 0.0 nothing in common."""
    n = max(len(a), len(b))
    if n == 0:
        return 1.0, 0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(blk.size for blk in sm.get_matching_blocks()) / n, n


def window_agreement(a: list[Word], b: list[Word], t_lo: float, t_hi: float, win: float = 30.0) -> list[dict]:
    """Agreement of two word streams in master-clock windows of `win` seconds (words by centre time)."""
    out = []
    t = t_lo
    while t < t_hi:
        wa = [w for w in a if t <= 0.5 * (w.m0 + w.m1) < t + win]
        wb = [w for w in b if t <= 0.5 * (w.m0 + w.m1) < t + win]
        ta, tb = tokens(wa), tokens(wb)
        if len(ta) + len(tb) >= 8:
            s, n = similarity(ta, tb)
            out.append({"t0": round(t, 1), "t1": round(t + win, 1), "agreement": round(s, 3), "n": n})
        t += win
    return out


def engine_table(raw_dir: Path, clocks: dict) -> dict:
    rows: dict[str, dict] = {}
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        cam = d.name
        vad = json.loads((d / "vad.json").read_text(encoding="utf-8")) if (d / "vad.json").exists() else {}
        speech_s = vad.get("speech_s") or 0.0
        for key in engine_keys(raw_dir, cam):
            ws = load_engine(raw_dir, cam, key, clocks.get(cam))
            probs = [w.prob for w in ws if w.prob is not None]
            rows.setdefault(cam, {})[key] = {
                "words": len(ws), "words_per_speech_min": round(60 * len(ws) / speech_s, 1) if speech_s else None,
                "mean_prob": round(float(np.mean(probs)), 3) if probs else None,
                "low_prob_share": round(float(np.mean([p < 0.5 for p in probs])), 3) if probs else None,
                "with_prob": len(probs)}
    return rows


PAIRS = [("whisper-large-v3:en", "parakeet-tdt-0.6b-v3"), ("whisper-large-v3:en", "canary-1b-v2"), ("parakeet-tdt-0.6b-v3", "canary-1b-v2"),
         ("whisper-large-v3:en", "whisper-large-v3-turbo:en"), ("whisper-large-v3:en", "whisper-large-v3:auto")]


def engine_agreement(raw_dir: Path, clocks: dict, win: float = 30.0) -> dict:
    """{cam: {"a|b": {mean, windows: [...]}}} for the engine pairs on each phone."""
    out: dict[str, dict] = {}
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        cam = d.name
        clk = clocks.get(cam)
        if clk is None:
            continue
        have = set(engine_keys(raw_dir, cam))
        cache: dict[str, list[Word]] = {}
        for a, b in PAIRS:
            if a not in have or b not in have:
                continue
            wa = cache.setdefault(a, load_engine(raw_dir, cam, a, clk))
            wb = cache.setdefault(b, load_engine(raw_dir, cam, b, clk))
            lo = min(w.m0 for w in wa + wb)
            hi = max(w.m1 for w in wa + wb)
            wins = window_agreement(wa, wb, lo, hi, win)
            if wins:
                tot = sum(w["n"] for w in wins)
                out.setdefault(cam, {})[f"{a}|{b}"] = {"mean": round(sum(w["agreement"] * w["n"] for w in wins) / tot, 3), "windows": wins}
    return out


def phone_agreement(raw_dir: Path, clocks: dict, engine: str = "parakeet-tdt-0.6b-v3", win: float = 30.0) -> dict:
    """The same engine on two phones in the master-clock windows both cover: how much the phone changes the transcript."""
    cams = [d.name for d in sorted(Path(raw_dir).glob("[0-9][0-9]")) if clocks.get(d.name) is not None and (d / (engine.split(":")[0] + ".json")).exists()]
    words = {c: load_engine(raw_dir, c, engine, clocks[c]) for c in cams}
    out = {}
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            if not words[a] or not words[b]:
                continue
            lo = max(min(w.m0 for w in words[a]), min(w.m0 for w in words[b]))
            hi = min(max(w.m1 for w in words[a]), max(w.m1 for w in words[b]))
            if hi - lo < win:
                continue
            wins = window_agreement(words[a], words[b], lo, hi, win)
            if wins:
                tot = sum(w["n"] for w in wins)
                out[f"{a}|{b}"] = round(sum(w["agreement"] * w["n"] for w in wins) / tot, 3)
    return out


def language_findings(raw_dir: Path) -> dict:
    """Share of speech time per language (Whisper's region id), per phone; regions whose top language is not English."""
    out: dict[str, dict] = {}
    total: Counter = Counter()
    pt_regions = []
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        f = d / "whisper-large-v3.json"
        if not f.exists():
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        c: Counter = Counter()
        for r in doc.get("regions", []):
            if not r.get("top"):
                continue
            dur = r["end"] - r["start"]
            c[r["top"]] += dur
            if r["top"] != "en" and dur >= 1.0:
                pt_regions.append({"cam": d.name, "start": r["start"], "end": r["end"], "lang": r["top"], "p": (r.get("lid") or {}).get(r["top"], r.get("top_p"))})
        tot = sum(c.values()) or 1.0
        out[d.name] = {k: round(v / tot, 3) for k, v in c.most_common()}
        total.update(c)
        passes = {p["requested"]: {"language": p["language"], "p": p["language_probability"], "words": len(p["words"])} for p in doc["passes"]}
        out[d.name]["passes"] = passes
    tt = sum(total.values()) or 1.0
    return {"per_cam": out, "overall": {k: round(v / tt, 3) for k, v in total.most_common()}, "non_english_regions": len(pt_regions),
            "non_english_examples": pt_regions[:20]}


def quality_table(raw_dir: Path) -> dict:
    out = {}
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        f = d / "quality.json"
        if f.exists():
            out[d.name] = json.loads(f.read_text(encoding="utf-8"))["overall"]
    return out


def build(raw_dir: Path, clocks: dict) -> dict:
    doc = {"engines": engine_table(raw_dir, clocks), "agreement": engine_agreement(raw_dir, clocks),
           "phone_agreement_parakeet": phone_agreement(raw_dir, clocks), "language": language_findings(raw_dir), "quality": quality_table(raw_dir)}
    return doc


def hard_windows(agreement: dict, n: int = 8, pair: str = "whisper-large-v3:en|parakeet-tdt-0.6b-v3", cam: str = "01") -> list[dict]:
    """The n windows (30 s) of one phone where two engines agree least (with enough words to mean something)."""
    wins = agreement.get(cam, {}).get(pair, {}).get("windows", [])
    return sorted([w for w in wins if w["n"] >= 25], key=lambda w: w["agreement"])[:n]
