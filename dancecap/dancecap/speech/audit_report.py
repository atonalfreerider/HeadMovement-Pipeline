"""S5: the numbers of the audit that need the pod outputs (audit_s5/pod_out/). Local, seconds. python -m dancecap.speech.audit_report <part>

    heldout     pseudo-WER of the final transcript and of the unadjudicated vote against whisper large-v2 (a model nobody used before: no prompt, own VAD), per phone, and
                for every span the editors changed: did the decision move TOWARD or AWAY from that engine (the held-out test S4 said does not exist). parakeet-tdt-1.1b
                drops many words on the far phones (recall 35-85 % of the other engines), so it is used as a reading, not in the pseudo-WER.
    timing      CTC forced alignment (wav2vec2) of the final words against the master audio: offsets of the 15 audit words and of every word of the sampled utterances
    speakers    independent speaker evidence: WavLM x-vector leave-one-out test on every utterance, the opening in 1 s windows, pyin F0
    sample      error counts of the 25 sampled utterances from audit_s5/sample_judgments.json (my manual judgments), by stratum
    laughter    CLAP laughing / applause scores: where the laughs are in the first 12 s and over the whole class, against the emoji tokens
    scores      teacher-forced readings (score.json) as a table of margins
    all         everything above plus the structure checks -> audit_report.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import SPEECH
from . import final_eval as FV
from . import fuse_eval as FE
from .audit import AUDIT, PHONES, final_doc
from .sources import load_clocks
from .texts import norm_token

POD = AUDIT / "pod_out"
FILLERS = {"um", "uh"}


def clocks():
    return load_clocks()


def indep_words(kind: str, cam: str) -> list[dict]:
    d = json.loads((POD / f"{kind}_full_{cam}.json").read_text(encoding="utf-8"))
    return d["words"]


def tk(a: float, b: float) -> float:
    return a + min(b - a, 1.0) / 2.0


def seq_indep(kind: str, cam: str, clk, window=FE.EVAL_WINDOW):
    xs = []
    for w in indep_words(kind, cam):
        n = norm_token(w["word"])
        if not n or n in FILLERS:
            continue
        t = tk(clk.to_master(w["start"]), clk.to_master(w["end"]))
        if window[0] <= t <= window[1]:
            xs.append((n, t))
    return [a for a, _ in xs], [b for _, b in xs]


def s1_hyps_seq(cam: str, key: str, clk, window=FE.EVAL_WINDOW):
    base, _, req = key.partition(":")
    d = json.loads((SPEECH / "raw" / cam / f"{base}.json").read_text(encoding="utf-8"))
    ws = d["words"] if "words" in d else next(p for p in d["passes"] if p["requested"] == req)["words"]
    xs = []
    for w in ws:
        n = norm_token(w["word"])
        if not n or n in FILLERS:
            continue
        t = tk(clk.to_master(w["start"]), clk.to_master(w["end"]))
        if window[0] <= t <= window[1]:
            xs.append((n, t))
    return [a for a, _ in xs], [b for _, b in xs]


def heldout() -> dict:
    final = final_doc()
    cons = json.loads((SPEECH / "consensus_v0.json").read_text(encoding="utf-8"))
    clk = clocks()
    cf, cv = FV.seq_final(final), FV.seq_v0(cons)
    rows = []
    for kind in ("wv2",):
        for cam in PHONES:
            ref = seq_indep(kind, cam, clk[cam])
            if len(ref[0]) < 50:
                continue
            pf, ef, n = FE.pwer(cf, ref)
            pv, ev, _ = FE.pwer(cv, ref)
            rows.append({"engine": kind, "cam": cam, "n_ref": n, "pwer_v0": round(pv, 4), "pwer_final": round(pf, 4), "delta": round(pf - pv, 4)})
    pk_recall = {cam: round(len(seq_indep("pk11", cam, clk[cam])[0]) / max(1, len(seq_indep("wv2", cam, clk[cam])[0])), 2) for cam in PHONES}
    # the same score for every S1 single engine (seen by the vote) as a scale, against the held-out engines
    scale = []
    for key in ("whisper-large-v3:en", "whisper-large-v3-turbo:en", "parakeet-tdt-0.6b-v3", "canary-1b-v2"):
        for cam in ("01", "02", "05"):
            hyp = s1_hyps_seq(cam, key, clk[cam])
            for kind in ("wv2",):
                ref = seq_indep(kind, cam, clk[cam])
                if len(ref[0]) < 50 or len(hyp[0]) < 50:
                    continue
                p, _, _ = FE.pwer(hyp, ref)
                scale.append({"engine_seen_by_vote": key, "cam": cam, "held_out": kind, "pwer": round(p, 4)})

    def mean(xs, k):
        return round(float(np.mean([x[k] for x in xs])), 4) if xs else None

    by = {k: {"n": len([r for r in rows if r["engine"] == k]), "pwer_v0": mean([r for r in rows if r["engine"] == k], "pwer_v0"),
              "pwer_final": mean([r for r in rows if r["engine"] == k], "pwer_final")} for k in ("wv2",)}
    # span effects against the held-out engines only
    allw_f, allw_v = FV.seq_final(final, (-1e9, 1e9)), FV.seq_v0(cons, (-1e9, 1e9))
    span_rows = []
    for c in final["adjudication"]["changes"]:
        if not (c["removed"] or c["added"]):
            continue
        sp = next(x for x in final["adjudication"]["spans"] if x["id"] == c["span"])
        lo, hi = sp["start"] - 0.3, sp["end"] + 0.3
        a = [n for n, t in zip(*allw_f) if lo <= t <= hi]
        b = [n for n, t in zip(*allw_v) if lo <= t <= hi]
        tf = tv = 0
        used = 0
        for kind in ("wv2",):
            for cam in PHONES:
                cl = clk[cam]
                if cl.to_master(0.0) > lo - 0.5:
                    continue
                dur_ok = True
                ref = [norm_token(w["word"]) for w in indep_words(kind, cam)
                       if lo - 0.2 <= tk(cl.to_master(w["start"]), cl.to_master(w["end"])) <= hi + 0.2 and norm_token(w["word"]) not in FILLERS and norm_token(w["word"])]
                if not dur_ok:
                    continue
                tf += FV._lev(a, ref)
                tv += FV._lev(b, ref)
                used += 1
        if used:
            span_rows.append({"span": c["span"], "start": sp["start"], "decision": c["decision"], "confidence": sp["confidence"], "old": c["old"], "new": c["new"],
                              "n_indep_sources": used, "delta_edits_per_source": round((tv - tf) / used, 3)})
    span_rows.sort(key=lambda r: r["delta_edits_per_source"])
    toward = [r for r in span_rows if r["delta_edits_per_source"] > 0.05]
    away = [r for r in span_rows if r["delta_edits_per_source"] < -0.05]
    out = {"what": "pseudo-WER (edits / words of the held-out transcript, 61-359 s) of the vote (v0) and of the final against engines that neither the vote nor the editors saw",
           "pk11_word_recall_vs_wv2": pk_recall, "rows": rows, "mean_by_engine": by, "mean_all": {"pwer_v0": mean(rows, "pwer_v0"), "pwer_final": mean(rows, "pwer_final")},
           "scale_seen_engines_vs_heldout": scale,
           "scale_mean": mean([{"p": s["pwer"]} for s in scale], "p"),
           "span_effects_vs_heldout": {"spans": span_rows, "toward": len(toward), "away": len(away), "neutral": len(span_rows) - len(toward) - len(away)}}
    (AUDIT / "heldout.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------------------ timing (CTC forced alignment)


def _targets(words: list[str], labels: list[str]) -> tuple[list[int], list[tuple[int, int]]]:
    """token ids of the words joined by '|' and, per word, the (first, last) index into the token list. Characters the CTC vocabulary lacks are dropped."""
    idx = {c: i for i, c in enumerate(labels)}
    toks: list[int] = []
    spans: list[tuple[int, int]] = []
    for k, w in enumerate(words):
        cs = [idx[c] for c in w.upper() if c in idx and c not in ("-", "|")]
        if k > 0:
            toks.append(idx["|"])
        a = len(toks)
        toks += cs
        spans.append((a, len(toks) - 1))
    return toks, spans


def ctc_align(lp: np.ndarray, words: list[str], labels: list[str]):
    """Forced alignment of `words` to the log-probabilities lp [T, C]; returns per word (start_frame, end_frame_exclusive, mean token probability) or None per word that
    could not be placed (a word without usable characters). Frame k covers [0.02 k, 0.02 (k + 1)) s of the window."""
    import torch
    import torchaudio.functional as AF
    toks, spans = _targets(words, labels)
    if not toks or lp.shape[0] < len(toks):
        return None
    ali, sc = AF.forced_align(torch.from_numpy(lp)[None], torch.tensor([toks], dtype=torch.int32), blank=0)
    ts = AF.merge_tokens(ali[0], sc[0].exp())
    ts = [t for t in ts if t.token != 0]
    # one span per target token (a repeated letter is one span per occurrence in merge_tokens when separated by blank frames; guard anyway)
    if len(ts) != len(toks):
        return None
    out = []
    for a, b in spans:
        if b < a:
            out.append(None)
            continue
        seg = ts[a:b + 1]
        out.append((seg[0].start, seg[-1].end, float(np.mean([t.score for t in seg]))))
    return out


def timing() -> dict:
    final = final_doc()
    clk = clocks()
    inp = json.loads((AUDIT / "audit_inputs.json").read_text(encoding="utf-8"))
    z = np.load(POD / "w2v.npz")
    meta = json.loads((POD / "w2v_meta.json").read_text(encoding="utf-8"))
    labels = meta["labels"]
    utt = {u["id"]: u for u in final["utterances"]}
    c01 = clk["01"]
    win = {w["id"]: w for w in inp["w2v"]}
    # (a) the 15 audit words: the word alone in a ~1 s window, and the same word inside the alignment of its whole utterance
    rows = []
    for tw in inp["timing_words"]:
        wid = f"w_{tw['utt']}_{tw['i']}"
        t0 = win[wid]["t0"]
        lp = z[wid]
        a1 = ctc_align(lp, [tw["norm"]], labels)
        row = {k: tw[k] for k in ("utt", "i", "text", "start", "end", "timing_class", "edit", "conf")}
        claimed_c = (tw["start"] + tw["end"]) / 2
        if a1 and a1[0]:
            f0, f1, sc = a1[0]
            s = c01.to_master(t0 + 0.02 * f0)
            e = c01.to_master(t0 + 0.02 * f1)
            row["alone"] = {"start": round(s, 3), "end": round(e, 3), "d_start_ms": round(1000 * (s - tw["start"])), "d_center_ms": round(1000 * ((s + e) / 2 - claimed_c)), "score": round(sc, 3)}
        uid = f"u_{tw['utt']}"
        u = utt[tw["utt"]]
        wds = [w for w in u["words"] if w["kind"] == "word"]
        lpu = z[uid]
        au = ctc_align(lpu, [w["norm"] for w in wds], labels)
        if au:
            k = next(j for j, w in enumerate(wds) if abs(w["start"] - tw["start"]) < 1e-6 and w["norm"] == tw["norm"])
            if au[k]:
                f0, f1, sc = au[k]
                s = c01.to_master(win[uid]["t0"] + 0.02 * f0)
                e = c01.to_master(win[uid]["t0"] + 0.02 * f1)
                row["in_utterance"] = {"start": round(s, 3), "end": round(e, 3), "d_start_ms": round(1000 * (s - tw["start"])), "d_center_ms": round(1000 * ((s + e) / 2 - claimed_c)), "score": round(sc, 3)}
        rows.append(row)
    # (b) every word of the sampled utterances + the utterances of the timing words, aligned in context on the master phone
    allw = []
    done = set()
    for wid, w in win.items():
        if w["kind"] != "utt" or w["utt"] in done:
            continue
        done.add(w["utt"])
        u = utt[w["utt"]]
        wds = [x for x in u["words"] if x["kind"] == "word"]
        au = ctc_align(z[wid], [x["norm"] for x in wds], labels)
        if not au:
            continue
        for x, a in zip(wds, au):
            if not a:
                continue
            s = c01.to_master(w["t0"] + 0.02 * a[0])
            e = c01.to_master(w["t0"] + 0.02 * a[1])
            fl = set(x.get("flags", []))
            cls = "editor-estimated" if ("time_window" in fl and "time_engines" not in fl) else "engine-anchored"
            allw.append({"utt": u["id"], "text": x["text"], "start": x["start"], "end": x["end"], "class": cls, "conf": x["conf"], "ali_start": round(s, 3), "ali_end": round(e, 3),
                         "d_start_ms": round(1000 * (s - x["start"])), "d_center_ms": round(1000 * ((s + e) / 2 - (x["start"] + x["end"]) / 2)), "score": round(a[2], 3)})

    def summ(rs, key):
        v = np.abs(np.array([r[key] for r in rs], dtype=float))
        if not len(v):
            return None
        return {"n": int(len(v)), "median_abs_ms": round(float(np.median(v))), "p90_abs_ms": round(float(np.percentile(v, 90))), "share_within_150ms": round(float((v <= 150).mean()), 3),
                "share_within_100ms": round(float((v <= 100).mean()), 3)}

    out = {"what": "CTC forced alignment (torchaudio wav2vec2 large lv60k 960h, 20 ms frames, master phone 01) of the final words; offsets = aligned minus claimed (ms)",
           "audit_words_15": rows,
           "audit_words_alone_center": summ([r["alone"] for r in rows if "alone" in r], "d_center_ms"),
           "audit_words_alone_start": summ([r["alone"] for r in rows if "alone" in r], "d_start_ms"),
           "audit_words_in_utterance_center": summ([r["in_utterance"] for r in rows if "in_utterance" in r], "d_center_ms"),
           "all_words_in_context": {"n": len(allw), "center": summ(allw, "d_center_ms"), "start": summ(allw, "d_start_ms"),
                                    "engine_anchored_center": summ([a for a in allw if a["class"] == "engine-anchored"], "d_center_ms"),
                                    "editor_estimated_center": summ([a for a in allw if a["class"] == "editor-estimated"], "d_center_ms"),
                                    "worst": sorted(allw, key=lambda r: -abs(r["d_center_ms"]))[:15]}}
    (AUDIT / "timing.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------------------ speakers


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def speakers() -> dict:
    """Independent speaker evidence, from models the pipeline did not use.
    (1) WavLM x-vectors (microsoft/wavlm-base-plus-sv; the pipeline used ECAPA, TitaNet and Sortformer) of every utterance on its best phone and on the next one, centred
        on the mean of the reference utterances (a raw x-vector carries a common component that makes plain clustering degenerate). Reference = the utterances after the
        opening that last at least 1.5 s. Leave-one-out nearest-centroid test: each utterance is classified by the centroids of the OTHER reference utterances of the two
        labels; the result is compared with the label of the final transcript.
    (2) pyin F0 of each utterance (the F0 of the male voice is not tracked reliably: the tracker returns its floor; only the female range is a usable cue).
    (3) the opening in 1 s windows (hop 0.25 s) on three phones: margin = similarity to the Larissa centroid minus similarity to the Kadu centroid."""
    final = final_doc()
    z = np.load(POD / "speaker.npz")
    zo = np.load(POD / "speaker_open.npz")
    meta = {m["key"]: m for m in json.loads((POD / "speaker_meta.json").read_text(encoding="utf-8"))["items"]}
    mo = json.loads((POD / "speaker_open_meta.json").read_text(encoding="utf-8"))["items"]
    turns = {t["id"]: t for t in final["turns"]}
    ref = [u for u in final["utterances"] if u["start"] >= 9.0 and u["end"] - u["start"] >= 1.5]
    gm = np.stack([z[f"{u['id']}_best"] for u in ref]).mean(0)

    def centroids(exclude: str | None):
        o = [x for x in ref if x["id"] != exclude]
        cl = _unit(np.mean([z[f"{x['id']}_best"] - gm for x in o if x["speaker"] == "Larissa"], axis=0))
        ck = _unit(np.mean([z[f"{x['id']}_best"] - gm for x in o if x["speaker"] == "Kadu"], axis=0))
        return cl, ck

    rows = []
    for u in final["utterances"]:
        cl, ck = centroids(u["id"])
        d = {"id": u["id"], "label": u["speaker"], "start": round(u["start"], 2), "seconds": round(u["end"] - u["start"], 2), "interjection": bool(turns[u["turn"]].get("interjection")),
             "text": u["text"][:40]}
        for rank in ("best", "second"):
            k = f"{u['id']}_{rank}"
            if k not in z.files:
                continue
            v = _unit(z[k] - gm)
            m = float(v @ cl - v @ ck)
            d[rank] = {"pred": "Larissa" if m > 0 else "Kadu", "margin": round(m, 2), "cam": meta[k]["cam"], "padded": meta[k]["padded"], "f0": meta[k]["f0_median"]}
        rows.append(d)
    body = [r for r in rows if r["start"] >= 9.0 and "best" in r]
    long_body = [r for r in body if r["seconds"] >= 1.5]
    short_body = [r for r in body if r["seconds"] < 1.5]

    def ok(rs):
        return sum(1 for r in rs if r["best"]["pred"] == r["label"])

    bad = [r for r in body if r["best"]["pred"] != r["label"] or ("second" in r and r["second"]["pred"] != r["label"])]
    f0 = defaultdict(list)
    for u in ref:
        m = meta[f"{u['id']}_best"]
        if m["f0_median"] and m["voiced_frames"] >= 15:
            f0[u["speaker"]].append(m["f0_median"])
    f0s = {k: {"n": len(v), "median": round(float(np.median(v)), 1), "p10": round(float(np.percentile(v, 10)), 1), "p90": round(float(np.percentile(v, 90)), 1)} for k, v in f0.items()}
    cl, ck = centroids(None)
    opening: dict[str, list] = defaultdict(list)
    for m in mo:
        v = _unit(zo[m["key"]] - gm)
        opening[m["cam"]].append({"m0": m["m0"], "margin": round(float(v @ cl - v @ ck), 2), "f0": m["f0_median"], "rms_db": m["rms_db"]})
    out = {"what": "independent speaker evidence for the labels of the final transcript (WavLM x-vector, centred, leave-one-out)",
           "reference_utterances": len(ref), "loo_reference_long": {"n": len(long_body), "ok": ok(long_body)},
           "loo_body_short_under_1_5s": {"n": len(short_body), "ok": ok(short_body)},
           "loo_body_all": {"n": len(body), "ok": ok(body)},
           "disagreements_best_or_second_phone": bad, "f0_by_label": f0s, "opening_windows": opening, "rows": rows}
    (AUDIT / "speakers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------------------ sample, laughter, scores


def sample_stats() -> dict:
    """Error counts of the 25 sampled utterances (my judgments in sample_judgments.json) and the same by stratum."""
    sample = json.loads((AUDIT / "sample.json").read_text(encoding="utf-8"))["utterances"]
    J = json.loads((AUDIT / "sample_judgments.json").read_text(encoding="utf-8"))["utterances"]
    if {r["id"] for r in sample} != set(J):
        raise ValueError("sample.json and sample_judgments.json hold different utterances")
    final = {u["id"]: u for u in final_doc()["utterances"]}
    rows = []
    for r in sample:
        j = J[r["id"]]
        n = sum(1 for w in final[r["id"]]["words"] if w["kind"] == "word")
        conf = sum(x["words"] for x in j.get("errors", []))
        likely = sum(x["words"] for x in j.get("likely", []))
        unc = sum(x["words"] for x in j.get("uncertain", []))
        rows.append({"id": r["id"], "group": r["group"], "third": r["third"], "loud": r["loud"], "snr_best": r["snr_best"], "words": n, "confirmed": conf, "likely": likely,
                     "uncertain": unc, "verdict": j["verdict"]})

    def agg(rs):
        n = sum(x["words"] for x in rs)
        c, l, u = (sum(x[k] for x in rs) for k in ("confirmed", "likely", "uncertain"))
        return {"utterances": len(rs), "words": n, "confirmed": c, "likely": l, "uncertain": u,
                "error_rate_confirmed": round(c / n, 4) if n else None, "error_rate_confirmed_plus_likely": round((c + l) / n, 4) if n else None,
                "error_rate_upper_all_uncertain_wrong": round((c + l + u) / n, 4) if n else None,
                "error_rate_central_half_of_uncertain": round((c + l + 0.5 * u) / n, 4) if n else None}

    out = {"what": "word error counts of the 25 randomly drawn utterances (outside the opening), by my judgment from independent engines", "all": agg(rows),
           "utterances_with_a_confirmed_or_likely_error": sum(1 for x in rows if x["confirmed"] or x["likely"]),
           "by_group": {g: agg([x for x in rows if x["group"] == g]) for g in ("Larissa", "Kadu", "other")},
           "by_third": {g: agg([x for x in rows if x["third"] == g]) for g in ("start", "middle", "end")},
           "by_loudness": {g: agg([x for x in rows if x["loud"] == g]) for g in ("loud", "quiet")}, "rows": rows}
    (AUDIT / "sample_stats.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def laughter() -> dict:
    """CLAP (laion/clap-htsat-unfused, zero-shot) laughing / applause: the first 12 s on phones 00, 01, 02 (hop 0.25 s) and the whole class on phones 01, 02, 05 (hop 0.5 s)."""
    d = json.loads((POD / "laughter.json").read_text(encoding="utf-8"))
    labels = d["labels"]
    opening = {}
    for cam, rows in d["cams"].items():
        opening[cam] = [{"m0": r["m0"], "laugh": max(r["a person laughing"], r["people laughing and giggling"]), "people": r["people laughing and giggling"], "talk": r["a person talking"],
                         "applause": r["applause"]} for r in rows]
    bins: dict[int, list[float]] = defaultdict(list)
    for cam, rows in opening.items():
        for r in rows:
            bins[int(round(r["m0"] / 0.25))].append(r["laugh"])
    hit = sorted(k * 0.25 for k, v in bins.items() if len(v) >= 2 and sorted(v)[-2] >= 0.4)
    interval = [min(hit), max(hit) + 1.0] if hit else None
    full = json.loads((POD / "laughter_all.json").read_text(encoding="utf-8"))
    grid: dict[int, dict[str, list]] = defaultdict(dict)
    for cam, rows in full["cams"].items():
        for r in rows:
            grid[int(round(r[0] / 0.5))][cam] = r[1:]
    cands = []
    for k in sorted(grid):
        t = k * 0.5
        if t < 12.0:
            continue
        sc = sorted((max(v[0], v[1]) for v in grid[k].values()), reverse=True)
        if sc and sc[0] >= 0.35 and len(sc) >= 2 and sc[1] >= 0.15:
            cands.append({"m0": t, "scores": {c: round(max(v[0], v[1]), 2) for c, v in grid[k].items()}})
    after = [max(v[0], v[1]) for k, g in grid.items() if k * 0.5 > 12 for v in g.values()]
    out = {"what": "independent laughter evidence (CLAP zero-shot, not the AST tagger of the pipeline)", "labels": labels,
           "opening_laugh_interval_two_phones_ge_0_4": interval, "opening": opening,
           "tokens_in_final": [{"t": 5.02, "to": 5.80, "name": "emoji 1 (after the name)"}, {"t": 6.75, "to": 7.10, "name": "emoji 2 (after the hair line)"}],
           "other_candidates_after_12s": cands, "after_12s_mean_laugh": round(float(np.mean(after)), 3), "after_12s_p99": round(float(np.percentile(after, 99)), 3),
           "applause_two_phones_ge_0_5_after_12s": [k * 0.5 for k, g in grid.items() if k * 0.5 > 12 and sum(1 for v in g.values() if v[4] >= 0.5) >= 2]}
    (AUDIT / "laughter_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def scores_table() -> dict:
    """Margins (sum of log-probabilities over the phones of the entry, best candidate = 0) for both whisper models."""
    d = json.loads((POD / "score.json").read_text(encoding="utf-8"))
    inp = json.loads((AUDIT / "audit_inputs.json").read_text(encoding="utf-8"))
    what = {s["id"]: s["what"] for s in inp["scores"]}
    agg: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for r in d["rows"]:
        agg[r["entry"]][r["model"].rsplit("-", 1)[-1]][r["cand"]] += r["logprob"]
    out = {}
    for e, models in agg.items():
        out[e] = {"what": what.get(e), "models": {m: {c: round(v - max(cands.values()), 1) for c, v in sorted(cands.items(), key=lambda kv: -kv[1])} for m, cands in models.items()}}
    (AUDIT / "scores_table.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def run_all() -> dict:
    from . import audit_struct
    res = {"struct": audit_struct.run(), "heldout": heldout(), "timing": timing(), "speakers": speakers(), "sample": sample_stats(), "laughter": laughter(), "scores": scores_table()}
    brief = {"struct_ok": res["struct"]["ok"], "struct_problems": res["struct"]["problems"],
             "heldout_mean": res["heldout"]["mean_all"], "heldout_span_effects": {k: v for k, v in res["heldout"]["span_effects_vs_heldout"].items() if k != "spans"},
             "timing_15_alone_center": res["timing"]["audit_words_alone_center"], "timing_15_alone_start": res["timing"]["audit_words_alone_start"],
             "timing_all": res["timing"]["all_words_in_context"]["center"], "speakers_loo": {k: res["speakers"][k] for k in ("loo_reference_long", "loo_body_short_under_1_5s", "loo_body_all")},
             "sample_all": res["sample"]["all"], "opening_laugh_interval": res["laughter"]["opening_laugh_interval_two_phones_ge_0_4"]}
    (AUDIT / "audit_report.json").write_text(json.dumps({"brief": brief, "parts": {k: v for k, v in res.items() if k != "struct"}, "struct": res["struct"]}, indent=1, ensure_ascii=False), encoding="utf-8")
    return brief


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    part = argv[0] if argv else "all"
    fn = {"heldout": heldout, "timing": timing, "speakers": speakers, "sample": sample_stats, "laughter": laughter, "scores": scores_table, "all": run_all}.get(part)
    if fn is None:
        print(__doc__)
        return 2
    r = fn()
    skip = ("rows", "opening", "opening_windows", "audit_words_15", "scale_seen_engines_vs_heldout")
    print(json.dumps(r if part == "all" else {k: v for k, v in r.items() if k not in skip}, indent=1, ensure_ascii=False)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
