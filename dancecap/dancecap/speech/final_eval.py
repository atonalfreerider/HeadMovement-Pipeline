"""S4 evaluation of the final transcript (local, numpy-free pure python on top of fuse_eval's banded edit distance; seconds).

There is no reference transcript, so every number is AGREEMENT between transcripts (pseudo-WER: word edits / words of the other transcript, canonical words, fillers
dropped, 61-359 s where all six phones recorded):

  rows                 for each of the 24 voting sources (6 phones x 4 engines): pseudo-WER of the unadjudicated vote (v0) and of the final transcript against that source.
                       NOT held out: the vote and the editors saw every source, so this table shows how close the text is to what the engines wrote, not how right it is.
  held_out_reference   the held-out numbers that exist are S2's (fusion_metrics.json, leave-one-source-out): the vote is 25 % closer to a transcript it did not see than the best
                       single engine. The adjudication has no held-out measurement (no sources were left out of it).
  expected_accuracy    a model-based range, with its assumptions in the statement: the optimistic end uses S2's three-cornered-hat error of the vote, the pessimistic end uses the
                       held-out disagreement rate by vote share (which includes the held-out transcripts' own errors); adjudicated words count as wrong with probability 1 - the editor's
                       confidence. Treat it as an order of magnitude, not as a WER.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import fuse_eval as FE
from .fuse import FILLERS

WINDOW = FE.EVAL_WINDOW


def _tk(start: float, end: float) -> float:
    return start + min(end - start, 1.0) / 2.0


def seq_final(final: dict, window=WINDOW) -> tuple[list[str], list[float]]:
    xs = [(w["norm"], _tk(w["start"], w["end"])) for u in final["utterances"] for w in u["words"]
          if w["kind"] == "word" and window[0] <= _tk(w["start"], w["end"]) <= window[1]]
    return [a for a, _ in xs], [b for _, b in xs]


def seq_v0(consensus: dict, window=WINDOW) -> tuple[list[str], list[float]]:
    xs = [(w["norm"], _tk(w["start"], w["end"])) for t in consensus["turns"] for w in t["words"]
          if w["norm"] not in FILLERS and window[0] <= _tk(w["start"], w["end"]) <= window[1]]
    return [a for a, _ in xs], [b for _, b in xs]


def _bin_p_wrong(share: float, bins: list[dict]) -> float:
    for b in bins:
        if b["share"][0] <= share < b["share"][1]:
            return float(b["p_wrong_word"]) + float(b["missing"]) / max(1, float(b["slots_x_heldout"]))
    return 0.02


def _lev(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def span_effects(final: dict, consensus: dict, hyps: list, pad: float = 0.3) -> dict:
    """For every span the editors changed: how many word edits the vote's words and the final words have against each source transcript that was recording, in the span's window.
    delta > 0 = the decision moved the text TOWARD what the sources wrote, delta < 0 = AWAY (the editor restored a word most sources lack, or overruled the majority)."""
    allw_f = seq_final(final, (-1e9, 1e9))
    allw_v = seq_v0(consensus, (-1e9, 1e9))
    rows = []
    for c in final["adjudication"]["changes"]:
        if not (c["removed"] or c["added"]):
            continue
        sp = next(x for x in final["adjudication"]["spans"] if x["id"] == c["span"])
        lo, hi = sp["start"] - pad, sp["end"] + pad
        a = [n for n, t in zip(*allw_f) if lo <= t <= hi]
        b = [n for n, t in zip(*allw_v) if lo <= t <= hi]
        used = [h for h in hyps if h.cover[0] <= lo and h.cover[1] >= hi]
        if not used:
            continue
        tf = tv = 0
        for h in used:
            ref = [t.norm for t in h.toks if lo - 0.2 <= t.tk <= hi + 0.2 and t.norm not in FILLERS]
            tf += _lev(a, ref)
            tv += _lev(b, ref)
        rows.append({"span": c["span"], "decision": c["decision"], "confidence": sp["confidence"], "needs_owner": sp["needs_owner"], "start": sp["start"], "old": c["old"], "new": c["new"],
                     "sources": len(used), "delta_edits_per_source": round((tv - tf) / len(used), 3)})
    rows.sort(key=lambda r: r["delta_edits_per_source"])
    return {"what": "per changed span: mean word edits against the recording sources, vote minus final; negative = the decision moved away from what most sources wrote", "spans": rows,
            "toward": sum(1 for r in rows if r["delta_edits_per_source"] > 0.01), "neutral": sum(1 for r in rows if abs(r["delta_edits_per_source"]) <= 0.01),
            "away": sum(1 for r in rows if r["delta_edits_per_source"] < -0.01)}


def bands(words: list[dict]) -> list[dict]:
    out = []
    for name, lo, hi in (("conf >= 0.9", 0.9, 1.01), ("0.8 - 0.9", 0.8, 0.9), ("0.6 - 0.8", 0.6, 0.8), ("< 0.6", -1.0, 0.6)):
        ws = [w for w in words if lo <= w["conf"] < hi]
        out.append({"band": name, "n": len(ws), "share_of_words": round(len(ws) / max(1, len(words)), 3), "mean_conf": round(sum(w["conf"] for w in ws) / len(ws), 3) if ws else None})
    return out


def expected_accuracy(final: dict, consensus: dict, metrics: dict | None) -> dict:
    """Expected word accuracy by confidence band. The top band (conf >= 0.9) is taken at 3 % error (2-4 %: S2's three-cornered-hat error of the vote is 4 %, an underestimate, and the
    held-out disagreement rate above 0.9 vote share is 2-4 %); the other bands use 1 - mean confidence (the vote share, or the editor's own confidence on a decided span), and the
    range scales those by 0.8 .. 1.2."""
    words = [w for u in final["utterances"] for w in u["words"] if w["kind"] == "word"]
    n = len(words)
    table = bands(words)
    top = table[0]
    rest_err = sum(b["n"] * (1.0 - b["mean_conf"]) for b in table[1:] if b["n"])
    central = (top["n"] * 0.03 + rest_err) / n
    low = (top["n"] * 0.02 + 0.8 * rest_err) / n
    high = (top["n"] * 0.04 + 1.2 * rest_err) / n
    adj = [w for w in words if w["conf_src"] != "vote"]
    changed = [w for w in words if w.get("edit") in ("changed", "inserted")]
    unclear_words = sum(1 for w in words if "unclear" in (w.get("flags") or []))
    stmt = (f"Expected word accuracy about {100 * (1 - central):.0f} % (range {100 * (1 - high):.0f} to {100 * (1 - low):.0f} %), i.e. a word error rate of about {100 * central:.0f} % "
            f"({100 * low:.0f} to {100 * high:.0f} %). It is very uneven: {top['n']} of {n} words ({100 * top['share_of_words']:.0f} %) have confidence 0.9 or more and are expected to be right about 97 % of the time; "
            f"the other {n - top['n']} words ({100 * (1 - top['share_of_words']):.0f} %) carry nearly all the errors. {len(adj)} words lie inside the {len(final['adjudication']['spans'])} disputed spans "
            f"and {len(changed)} of them were changed by an editor; {table[3]['n']} words are below 0.6 and {unclear_words} words sit in spans the editors flagged unclear. The opening jokes "
            "(first 8 s: overlapping voices and laughter) are the weakest part. Method: no ground truth exists; the estimate multiplies each confidence band by its error (top band 3 %, the others "
            "1 - mean confidence) and the range scales the bands by 0.8 to 1.2: an order of magnitude, not a measured word error rate.")
    return {"n_words": n, "vote_words": n - len(adj), "words_in_decided_spans": len(adj), "words_changed_by_editors": len(changed), "conf_mean": round(sum(w["conf"] for w in words) / n, 4),
            "bands": table, "error_central": round(central, 4), "error_low": round(low, 4), "error_high": round(high, 4), "statement": stmt}


def evaluate(final: dict, consensus: dict, hyps: list, metrics: dict | None = None) -> dict:
    cf, cv = seq_final(final), seq_v0(consensus)
    rows = []
    for h in sorted(hyps, key=lambda h: (h.cam, h.engine)):
        ref = FE.seq_of(h, WINDOW)
        if len(ref[0]) < 50:
            continue
        pf, ef, n = FE.pwer(cf, ref)
        pv, ev, _ = FE.pwer(cv, ref)
        rows.append({"cam": h.cam, "engine": h.engine, "n_ref": n, "pwer_v0": round(pv, 4), "pwer_final": round(pf, 4), "delta": round(pf - pv, 4)})
    by_engine: dict[str, list] = defaultdict(list)
    by_cam: dict[str, list] = defaultdict(list)
    for r in rows:
        by_engine[r["engine"]].append(r)
        by_cam[r["cam"]].append(r)

    def mean(xs, k):
        return round(sum(x[k] for x in xs) / len(xs), 4)

    loso = (metrics or {}).get("loso") or {}
    st = final["stats"]
    return {
        "schema": "dancecap.speech/1/evaluation", "window_s": list(WINDOW),
        "what": "pseudo-WER of the unadjudicated vote (v0) and of the final transcript against each of the 24 voting sources; not held out (see the module doc)",
        "rows": rows,
        "mean": {"pwer_v0": mean(rows, "pwer_v0"), "pwer_final": mean(rows, "pwer_final")},
        "by_engine": {k: {"pwer_v0": mean(v, "pwer_v0"), "pwer_final": mean(v, "pwer_final")} for k, v in sorted(by_engine.items())},
        "by_phone": {k: {"pwer_v0": mean(v, "pwer_v0"), "pwer_final": mean(v, "pwer_final")} for k, v in sorted(by_cam.items())},
        "single_vs_single_for_scale": {"note": "S2 leave-one-source-out: a single engine's transcript scored against a held-out reference", "single_engine_pwer_mean": loso.get("single_engine_pwer_mean"),
                                       "best_single_engine": loso.get("best_single_engine"), "fused_pwer_held_out": loso.get("fused_pwer_mean")},
        "held_out_reference": {"fused_vote_vs_best_single_engine": {"fused": loso.get("fused_pwer_mean"), "best_single_engine_mean": (loso.get("single_engine_pwer_mean") or {}).get(loso.get("best_single_engine")),
                                                                      "fused_beats_best_engine_in": loso.get("fused_beats_best_engine_mean")},
                               "note": "the adjudication step has no held-out measurement"},
        "counts": {k: st[k] for k in ("words_v0", "words", "fillers", "events", "words_changed_by_adjudication", "words_removed_by_adjudication", "spans_total", "spans_changed", "spans_unclear",
                                       "spans_needs_owner", "conf_mean")},
        "adjudication_vs_sources": span_effects(final, consensus, hyps),
        "reading_the_rows": "The final is on average FARTHER from the single source transcripts than the vote is (rows), because the editors restored words the vote had dropped and overruled the "
                            "majority reading on some spans: each such change moves the text away from most single sources by design. This metric cannot show the adjudication is right; it shows "
                            "where the editors went against the majority (adjudication_vs_sources, most negative first): the best places for a human to check.",
        "decisions": final["adjudication"]["decisions"], "checks": final.get("checks"),
        "expected_accuracy": expected_accuracy(final, consensus, metrics),
    }
