"""Objective checks of the fused transcript without any reference transcript (local, numpy, minutes).

There is no ground truth for the class speech, so the measure is AGREEMENT with a transcript the consensus did not see ("pseudo-WER": the
word error rate of one transcript scored against another, on canonical words, fillers dropped, only the 61-359 s stretch where all six
phones recorded). A pseudo-WER contains the errors of both sides; the comparison that matters is between candidates scored against the SAME
held-out reference, so the reference's own errors cancel as long as they are independent of the candidates' errors.

  LOSO (leave one source out)   the consensus is built WITHOUT phone k and scored against each engine's transcript of phone k; every single
                                (phone j != k, engine e') transcript is scored against the same reference. The fused result is better than a
                                single engine when its pseudo-WER is lower.
  LOSEO (leave source+engine)   the consensus is built without phone k AND without the reference's engine e, and scored against (k, e) with the
                                single transcripts of the other engines as rivals. This removes the advantage of "same engine, similar
                                mistakes" that the plain LOSO gives to both the consensus and the same-engine rivals.
  three-cornered hat            for three systems A, B, C with independent errors, err(A) = (d(A,B) + d(A,C) - d(B,C)) / 2 (d = edits per word);
                                reported for the fused result vs a held-out reference vs a rival engine. Errors of engines on the same audio are positively
                                correlated, so the hat underestimates every system's error: it is a ranking aid, not an absolute WER.
  reliability curve             after building the consensus without phone k, the held-out transcripts of k are aligned into the network; the
                                share of held-out tokens that equal the consensus word, by the winner's vote share, is the empirical P(correct) that
                                the disagreement threshold of fuse_batches.py is chosen from.
"""
from __future__ import annotations

import copy
import time
from dataclasses import replace

import numpy as np

from . import fuse
from .fuse import FILLERS, Hyp, Params, Slot

EVAL_WINDOW = (61.0, 359.0)


def clone_hyps(hyps: list[Hyp], cams: set[str] | None = None, drop_cams: set[str] | None = None, drop_engines: set[str] | None = None,
               only_engines: set[str] | None = None) -> list[Hyp]:
    out = []
    for h in hyps:
        if cams is not None and h.cam not in cams:
            continue
        if drop_cams and h.cam in drop_cams:
            continue
        if drop_engines and h.engine in drop_engines:
            continue
        if only_engines is not None and h.engine not in only_engines:
            continue
        out.append(Hyp(h.hid, h.cam, h.engine, [replace(t) for t in h.toks], h.cover, h.key, h.w0))
    return out


def seq_of(h: Hyp, window: tuple[float, float] = EVAL_WINDOW) -> tuple[list[str], list[float]]:
    xs = [(t.norm, t.tk) for t in h.toks if window[0] <= t.tk <= window[1] and t.norm not in FILLERS]
    return [a for a, _ in xs], [b for _, b in xs]


def seq_of_consensus(C: "fuse.Consensus", window: tuple[float, float] = EVAL_WINDOW) -> tuple[list[str], list[float]]:
    xs = [(r.winner, r.t) for r in C.results if r.winner is not None and r.winner not in FILLERS and window[0] <= r.t <= window[1]]
    return [a for a, _ in xs], [b for _, b in xs]


def edit_distance(a: list[str], ta: list[float], b: list[str], tb: list[float], band_s: float = 8.0) -> int:
    """Word-level Levenshtein distance (unit costs) restricted to pairs whose times are within band_s (so it is O(N * band))."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return max(n, m)
    INF = 10 ** 9
    ta_arr, tb_arr = np.asarray(ta), np.asarray(tb)
    lo = np.maximum.accumulate(np.searchsorted(tb_arr, ta_arr - band_s, "left"))
    hi = np.maximum.accumulate(np.searchsorted(tb_arr, ta_arr + band_s, "right"))
    prev = [INF] * (m + 1)
    for j in range(0, min(m, int(hi[0])) + 1):
        prev[j] = j
    last_h = min(m, int(hi[0]))
    for i in range(1, n + 1):
        h = m if i == n else min(m, int(hi[i - 1]))
        l = min(int(lo[i - 1]), last_h)
        cur = [INF] * (m + 1)
        if l == 0 and prev[0] < INF:
            cur[0] = prev[0] + 1
        ai = a[i - 1]
        for j in range(max(l, 1), h + 1):
            c = prev[j - 1] + (ai != b[j - 1])
            v = prev[j] + 1
            if v < c:
                c = v
            v = cur[j - 1] + 1
            if v < c:
                c = v
            cur[j] = c
        prev = cur
        last_h = h
    return int(prev[m]) if prev[m] < INF else max(n, m)


def pwer(c: tuple[list[str], list[float]], ref: tuple[list[str], list[float]]) -> tuple[float, int, int]:
    """(edits / len(ref), edits, len(ref)) of candidate `c` against `ref`."""
    d = edit_distance(c[0], c[1], ref[0], ref[1])
    return d / max(1, len(ref[0])), d, len(ref[0])


def build(hyps: list[Hyp], P: Params) -> "fuse.Consensus":
    return fuse.consensus(hyps, P)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def loso(hyps_prod: list[Hyp], hyps_raw: list[Hyp], P: Params, cross_engine: bool = False, log=print) -> dict:
    """Pseudo-WER table. `hyps_prod` are the production-normalised hypotheses (compounds merged with all the others), used as references and as
    single rivals; `hyps_raw` are the un-merged ones the held-out consensus is built from. cross_engine=True is the LOSEO variant."""
    cams = sorted({h.cam for h in hyps_raw})
    engines = sorted({h.engine for h in hyps_raw})
    seqs = {(h.cam, h.engine): seq_of(h) for h in hyps_prod}
    rows = []
    t0 = time.time()
    for k in cams:
        for e in (engines if cross_engine else [None]):
            sub = clone_hyps(hyps_raw, drop_cams={k}, drop_engines={e} if e else None)
            C = fuse.consensus(sub, P)
            cs = seq_of_consensus(C)
            for ref_e in ([e] if cross_engine else engines):
                ref = seqs[(k, ref_e)]
                wc, dc, n = pwer(cs, ref)
                singles = {}
                for j in cams:
                    if j == k:
                        continue
                    for e2 in engines:
                        if cross_engine and e2 == ref_e:
                            continue
                        singles[(j, e2)] = pwer(seqs[(j, e2)], ref)[0]
                rows.append({"held_out": k, "ref_engine": ref_e, "n_ref": n, "fused": wc, "fused_edits": dc, "singles": {f"{j}|{e2}": v for (j, e2), v in singles.items()}})
            log(f"  held out {k}{' without ' + e if e else ''}: {time.time() - t0:.0f} s")
    return summarize(rows, engines, cams, cross_engine)


def summarize(rows: list[dict], engines: list[str], cams: list[str], cross_engine: bool) -> dict:
    fused = [r["fused"] for r in rows]
    # mean pseudo-WER of each single engine (all rival phones) and of each single (phone, engine) over the same references
    by_engine: dict[str, list[float]] = {e: [] for e in engines}
    by_pair: dict[str, list[float]] = {}
    for r in rows:
        for key, v in r["singles"].items():
            j, e2 = key.split("|")
            by_engine[e2].append(v)
            by_pair.setdefault(key, []).append(v)
    eng_mean = {e: _mean(v) for e, v in by_engine.items() if v}
    best_engine = min(eng_mean, key=eng_mean.get)
    # per reference: fused vs the best single transcript, vs the best engine (mean over its phones), vs the median single
    beats_best_single = 0
    beats_best_engine = 0
    beats_median = 0
    diffs = []
    for r in rows:
        sv = list(r["singles"].values())
        beats_best_single += r["fused"] < min(sv)
        beats_median += r["fused"] < float(np.median(sv))
        per_e = {}
        for key, v in r["singles"].items():
            per_e.setdefault(key.split("|")[1], []).append(v)
        be = min(float(np.mean(v)) for v in per_e.values())
        beats_best_engine += r["fused"] < be
        diffs.append(be - r["fused"])
    n = len(rows)
    pair_mean = {k: _mean(v) for k, v in by_pair.items()}
    best_pair = min(pair_mean, key=pair_mean.get)
    return {
        "variant": "leave source and engine out" if cross_engine else "leave source out",
        "references": n,
        "fused_pwer_mean": _mean(fused),
        "single_engine_pwer_mean": eng_mean,
        "best_single_engine": best_engine,
        "best_single_pair": {"pair": best_pair, "pwer_mean": pair_mean[best_pair]},
        "single_pair_pwer_mean": pair_mean,
        "median_single_pwer_mean": _mean([float(np.median(list(r["singles"].values()))) for r in rows]),
        "fused_beats_best_single_transcript": f"{beats_best_single}/{n}",
        "fused_beats_best_engine_mean": f"{beats_best_engine}/{n}",
        "fused_beats_median_single": f"{beats_median}/{n}",
        "mean_gain_over_best_engine": _mean(diffs),
        "relative_gain_over_best_engine": (_mean(diffs) / eng_mean[best_engine]) if eng_mean[best_engine] else None,
        "by_held_out_phone": {k: _mean([r["fused"] for r in rows if r["held_out"] == k]) for k in cams},
        "rows": rows,
    }


# ------------------------------------------------------------------------------------------------ reliability of the vote share


def copy_slots(slots: list[Slot]) -> list[Slot]:
    out = []
    for s in slots:
        n = Slot()
        n.mem = dict(s.mem)
        n.t = s.t
        n.cand = dict(s.cand)
        n.wsum = s.wsum
        out.append(n)
    return out


def reliability_curve(hyps_prod: list[Hyp], hyps_raw: list[Hyp], P: Params, bins=(0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01), log=print) -> dict:
    """P(a held-out transcript has the consensus word | the winner's vote share), the consensus built without the held-out phone."""
    cams = sorted({h.cam for h in hyps_raw})
    engines = sorted({h.engine for h in hyps_raw})
    stats = {b: [0, 0, 0] for b in bins[:-1]}      # per bin: [held-out tokens equal to winner, held-out tokens different, held-out nothing]
    null_stats = {b: [0, 0] for b in bins[:-1]}    # winner "nothing": [held-out nothing, held-out has a word]
    for k in cams:
        sub = clone_hyps(hyps_raw, drop_cams={k})
        C = fuse.consensus(sub, P)
        inside = list(C.hyps)
        held = clone_hyps(hyps_prod, cams={k})
        for h in held:
            for t in h.toks:
                t.hid = h.hid
        fuse.hyp_engine[0].update({h.hid: h.engine for h in held})
        for h in held:
            slots2 = copy_slots(C.slots)
            fuse.align_hyp(slots2, h, inside, P)
            for r, s2 in zip(C.results, slots2):
                if not (EVAL_WINDOW[0] <= r.t <= EVAL_WINDOW[1]):
                    continue
                tok = s2.mem.get(h.hid)
                b = max(x for x in bins[:-1] if x <= r.share + 1e-9)
                if r.winner is None:
                    null_stats[b][0 if tok is None else 1] += 1
                    continue
                if tok is None:
                    stats[b][2] += 1
                elif tok.norm == r.winner:
                    stats[b][0] += 1
                else:
                    stats[b][1] += 1
        log(f"  reliability: held out {k}")
    out = {"bins": [], "note": "agree = held-out token equals the consensus word; differ = another word; missing = the held-out transcript has no word in that slot"}
    for b, nb in zip(bins[:-1], bins[1:]):
        a, d, m = stats[b]
        tot = a + d + m
        out["bins"].append({"share": [b, min(nb, 1.0)], "slots_x_heldout": tot, "agree": a, "differ": d, "missing": m,
                            "p_agree": a / tot if tot else None, "p_wrong_word": d / tot if tot else None,
                            "null_winner_heldout_also_empty": null_stats[b][0], "null_winner_heldout_has_word": null_stats[b][1]})
    return out


def three_cornered_hat(hyps_prod: list[Hyp], hyps_raw: list[Hyp], P: Params, log=print) -> dict:
    """err(fused), err(rival engine), err(reference engine) from pairwise distances of the held-out consensus, a rival and the reference."""
    cams = sorted({h.cam for h in hyps_raw})
    engines = sorted({h.engine for h in hyps_raw})
    seqs = {(h.cam, h.engine): seq_of(h) for h in hyps_prod}
    acc = {"fused": [], "rival": [], "ref": []}
    per_engine: dict[str, list[float]] = {e: [] for e in engines}
    for k in cams:
        sub = clone_hyps(hyps_raw, drop_cams={k})
        C = fuse.consensus(sub, P)
        cs = seq_of_consensus(C)
        for ref_e in engines:
            ref = seqs[(k, ref_e)]
            for j in cams:
                if j == k:
                    continue
                for e2 in engines:
                    if e2 == ref_e:
                        continue
                    riv = seqs[(j, e2)]
                    dcr = edit_distance(cs[0], cs[1], ref[0], ref[1])
                    dcs = edit_distance(cs[0], cs[1], riv[0], riv[1])
                    drs = edit_distance(ref[0], ref[1], riv[0], riv[1])
                    n = float(np.mean([len(cs[0]), len(ref[0]), len(riv[0])]))
                    ec = (dcr + dcs - drs) / 2.0 / n
                    er = (dcr + drs - dcs) / 2.0 / n
                    es = (dcs + drs - dcr) / 2.0 / n
                    acc["fused"].append(ec)
                    acc["ref"].append(er)
                    acc["rival"].append(es)
                    per_engine[e2].append(es)
        log(f"  hat: held out {k}")
    return {"estimate_edits_per_word": {k: _mean(v) for k, v in acc.items()}, "rival_by_engine": {e: _mean(v) for e, v in per_engine.items()},
            "triples": len(acc["fused"]),
            "caveat": "assumes the errors of the three transcripts are independent; engines on the same audio share hard words, so every estimate is too low"}


# ------------------------------------------------------------------------------------------------ clocks seen from the speech


def clock_residuals(C: "fuse.Consensus", min_voters: int = 8, block_s: float = 60.0) -> dict:
    """Is the audio clock of every phone still right when judged by the WORDS? For every consensus word that at least `min_voters` hypotheses wrote, the start time
    each hypothesis gives minus the word's consensus start (weighted median over the hypotheses); per phone and engine the median (ms), the spread (MAD) and the slope
    over the class (ppm), and the per-minute medians of the tightest engine (parakeet). The speech timestamps of the engines are only good to about +-50-100 ms, so
    a phone is only re-timed when its offset against the other phones (same engine) is beyond 40 ms on at least three engines or it drifts."""
    engines = {h.hid: h.engine for h in C.hyps}
    cams = {h.hid: h.cam for h in C.hyps}
    rows: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r, s in zip(C.results, C.slots):
        if r.winner is None or len(s.mem) < min_voters:
            continue
        t0, _ = fuse.best_word_time(s, r.winner, engines)
        for hid, tok in s.mem.items():
            if tok.norm == r.winner:
                rows.setdefault((cams[hid], engines[hid]), []).append((t0, tok.t0 - t0))
    out: dict = {"per_phone_engine": {}, "per_phone": {}, "parakeet_blocks_ms": {}}
    for (cam, eng), v in sorted(rows.items()):
        a = np.array(v)
        med = float(np.median(a[:, 1]))
        mad = float(np.median(np.abs(a[:, 1] - med)))
        slope = float(np.polyfit(a[:, 0], a[:, 1], 1)[0] * 1e6) if len(a) > 20 else None
        out["per_phone_engine"][f"{cam}|{eng}"] = {"n": len(a), "median_ms": round(med * 1000, 1), "mad_ms": round(mad * 1000, 1), "slope_ppm": None if slope is None else round(slope, 1)}
    # each engine has its own timing bias (whisper starts words early, canary late); the clock of a PHONE is what differs between phones for the same engine
    eng_mean = {}
    for e in {k.split("|")[1] for k in out["per_phone_engine"]}:
        eng_mean[e] = float(np.median([v["median_ms"] for k, v in out["per_phone_engine"].items() if k.endswith("|" + e)]))
    out["engine_bias_ms"] = {e: round(m, 1) for e, m in sorted(eng_mean.items())}
    for cam in sorted({c for c, _ in rows}):
        rel = [out["per_phone_engine"][k]["median_ms"] - eng_mean[k.split("|")[1]] for k in out["per_phone_engine"] if k.startswith(cam + "|")]
        out["per_phone"][cam] = {"offset_vs_other_phones_ms": round(float(np.median(rel)), 1), "per_engine_ms": [round(r, 1) for r in rel], "engines_beyond_40ms": int(sum(abs(m) > 40 for m in rel))}
        pk = rows.get((cam, "parakeet-tdt-0.6b-v3"))
        if pk:
            a = np.array(pk)
            blocks = {}
            for b0 in np.arange(np.floor(a[:, 0].min() / block_s) * block_s, a[:, 0].max(), block_s):
                sel = a[(a[:, 0] >= b0) & (a[:, 0] < b0 + block_s)]
                if len(sel) >= 15:
                    blocks[f"{b0:.0f}"] = round(float(np.median(sel[:, 1])) * 1000, 1)
            out["parakeet_blocks_ms"][cam] = blocks
    return out
