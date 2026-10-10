"""S5: independent audit of the final class transcript (local side). Everything it writes goes to work/<take>/speech/audit_s5/ (git-ignored).

The audit does NOT trust the pipeline: it draws its own stratified sample of utterances, cuts the audio of the span from the phone that heard it best,
and compares the final text with engines the vote and the editors never saw (faster-whisper large-v2 without any prompt, NVIDIA parakeet-tdt-1.1b)
that run once on a pod (pod/stage_audit.py). Steps (python -m dancecap.speech.audit <step>):

    sample     25 utterances stratified by speaker group, third of the class and loudness of the best phone  -> sample.json
    inputs     the windows the pod stage decodes / aligns / embeds / scores                                  -> audit_inputs.json
    (pod)      pod/stage_audit.py                                                                             -> pod_out/
    report     all comparisons (WER of final and vote against the held-out engines, timing, speakers, ...)  -> audit_report.json

Times: `master` = the 01-Recap-Tina clock; a phone's own time is (master - offset) / slope (sources.Clock).
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np

from . import SOURCES, SPEECH
from .sources import load_clocks

AUDIT = SPEECH / "audit_s5"
PHONES = ["00", "01", "02", "03", "05", "06"]          # 04 (Tony) has no clock and is not in the 6-minute transcript
SEED = 5
PAD = 1.0                                              # seconds of context around a sampled utterance
OPENING_END = 9.0                                      # master seconds: the opening (pre-roll, the two jokes) is audited separately


def wav_path(cam: str) -> Path:
    return SPEECH / "audio" / f"{Path(SOURCES[cam]).stem}.16k.wav"


def final_doc() -> dict:
    return json.loads((SPEECH / "transcript_final.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ levels, best phone


def phone_levels(force: bool = False) -> dict[str, np.ndarray]:
    """50 ms frame levels (dB, 300-3400 Hz band) of every phone's 16 kHz audio, cached in audit_s5/levels.npz."""
    p = AUDIT / "levels.npz"
    if p.exists() and not force:
        z = np.load(p)
        return {k: z[k] for k in z.files}
    import soundfile as sf
    from scipy.signal import butter, sosfilt
    sos = butter(4, [300, 3400], btype="band", fs=16000, output="sos")
    out = {}
    for cam in PHONES:
        x, _ = sf.read(wav_path(cam), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        y = sosfilt(sos, x)
        n = len(y) // 800
        fr = y[:n * 800].reshape(n, 800)
        out[cam] = (10 * np.log10((fr ** 2).mean(1) + 1e-12)).astype(np.float32)
    AUDIT.mkdir(parents=True, exist_ok=True)
    np.savez(p, **out)
    return out


def span_snr(lev: dict[str, np.ndarray], clocks: dict, a: float, b: float) -> dict[str, float]:
    """{phone: SNR dB} of the master span [a, b]: the 75th percentile frame level in the span minus the phone's 10th percentile level
    (its noise floor), for the phones that were recording the whole span."""
    out = {}
    for cam, db in lev.items():
        c = clocks.get(cam)
        if c is None:
            continue
        ta, tb = c.to_source(a), c.to_source(b)
        if ta < 0.0 or int(tb * 20) + 1 > len(db):
            continue
        seg = db[int(ta * 20): int(tb * 20) + 1]
        if len(seg):
            out[cam] = float(np.percentile(seg, 75) - np.percentile(db, 10))
    return out


# ------------------------------------------------------------------------------------------------ sample


def group_of(u: dict, turn: dict | None) -> str:
    """Larissa / Kadu = a normal turn of that teacher; other = the second voice or an interjection (short answers inside the other's turn)."""
    if turn is not None and turn.get("interjection"):
        return "other"
    return u["speaker"]


def third_of(t: float) -> str:
    return "start" if t < 120 else "middle" if t < 240 else "end"


def build_sample(n: int = 25, seed: int = SEED, write: bool = True) -> dict:
    """Seeded randomised greedy stratified draw: quotas Larissa 9, Kadu 9, other 7; thirds 8/9/8; loud (best-phone SNR above the median of all
    utterances) 13 / quiet 12. An utterance must have at least two phones recording it and at least 0.2 s of duration."""
    doc = final_doc()
    clocks = load_clocks()
    lev = phone_levels()
    turns = {t["id"]: t for t in doc["turns"]}
    rows = []
    for u in doc["utterances"]:
        a, b = u["start"], u["end"]
        snr = span_snr(lev, clocks, a - PAD, b + PAD)
        order = sorted(snr, key=snr.get, reverse=True)
        if len(order) < 2 or b - a < 0.2 or a < OPENING_END:
            continue
        rows.append({"id": u["id"], "turn": u["turn"], "speaker": u["speaker"], "group": group_of(u, turns.get(u["turn"])), "start": a, "end": b,
                     "third": third_of((a + b) / 2), "best": order[0], "second": order[1], "snr_best": round(snr[order[0]], 1),
                     "snr": {k: round(v, 1) for k, v in snr.items()}, "n_words": u["n_words"], "text": u["text"], "conf_mean": u["conf_mean"]})
    med = float(np.median([r["snr_best"] for r in rows]))
    for r in rows:
        r["loud"] = "loud" if r["snr_best"] >= med else "quiet"
    quota = {"group": {"Larissa": 9, "Kadu": 9, "other": 7}, "third": {"start": 8, "middle": 9, "end": 8}, "loud": {"loud": 13, "quiet": 12}}
    rng = random.Random(seed)
    pool = rows[:]
    rng.shuffle(pool)
    chosen: list[dict] = []
    count = {k: {x: 0 for x in v} for k, v in quota.items()}

    def deficit(r) -> float:
        return sum((quota[k][r[k]] - count[k][r[k]]) / quota[k][r[k]] for k in quota)

    for _ in range(n):
        cand = [r for r in pool if all(count[k][r[k]] < quota[k][r[k]] for k in quota)] or pool
        best = max(cand, key=lambda r: (deficit(r), rng.random()))
        chosen.append(best)
        pool.remove(best)
        for k in quota:
            count[k][best[k]] += 1
    chosen.sort(key=lambda r: r["start"])
    out = {"what": "S5 audit sample: seeded stratified random draw of final-transcript utterances", "seed": seed, "median_snr_best": round(med, 1),
           "quota": quota, "count": count, "n_pool": len(rows), "utterances": chosen}
    if write:
        AUDIT.mkdir(parents=True, exist_ok=True)
        (AUDIT / "sample.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------------------ inputs of the pod stage


def to_phone(clocks: dict, cam: str, m0: float, m1: float, dur: float) -> tuple[float, float]:
    c = clocks[cam]
    return max(0.0, c.to_source(m0)), min(dur, c.to_source(m1))


def wav_dur(cam: str) -> float:
    import soundfile as sf
    return sf.info(str(wav_path(cam))).duration


# Candidate readings the pod scores by teacher forcing (log-probability of each text given the audio of a window of the master clock). They quote the class and the
# owner's jokes, so they live in the git-ignored audit_s5/score_candidates.json: {"scores": [{"id", "m0", "m1", "what", "candidates": {key: text}, "all_cams"?: bool,
# "cams_override"?: [phone ids]}]}. Every entry compares sentence variants that differ only in the disputed words (a rare word costs more than a common one under the
# language model of the scorer: read the margins with that bias in mind). The entries are the places the editors flagged (proofread_first), the opening, and the places
# where the audit found a problem by reading.
def load_scores() -> list[dict]:
    p = AUDIT / "score_candidates.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing (it is git-ignored: it quotes the class)")
    return json.loads(p.read_text(encoding="utf-8"))["scores"]


def scoring_entries(clocks: dict, lev: dict) -> list[dict]:
    out = []
    for s in load_scores():
        snr = span_snr(lev, clocks, s["m0"], s["m1"])
        order = sorted(snr, key=snr.get, reverse=True)
        cams = order if s.get("all_cams") else order[:2]
        # the opening is on the master and the phones that heard it best in the other steps (00, 02); keep the best two by SNR plus the master
        if s["id"] in ("opening_name", "red_hair", "very_red", "name_only"):
            cams = [c for c in ("01", "00", "02") if c in snr]
        out.append({**s, "cams": cams, "windows": {c: list(to_phone(clocks, c, s["m0"], s["m1"], wav_dur(c))) for c in cams}})
    return out


def words_for_timing(n: int = 15, seed: int = SEED) -> list[dict]:
    """15 words for the timing check: 8 whose time was anchored on an engine's own word (edit 'kept' or 'changed' with time_engines) and 7 whose time is an
    editor-word estimate (flag time_window only); outside the opening 8.5 s, at least 0.12 s long, text of 3+ letters, one per utterance."""
    doc = final_doc()
    rng = random.Random(seed)
    anch, est = [], []
    for u in doc["utterances"]:
        if u["start"] < 9.0:
            continue
        for i, w in enumerate(u["words"]):
            if w.get("kind") != "word" or len(w["norm"]) < 3 or w["end"] - w["start"] < 0.12:
                continue
            fl = set(w.get("flags", []))
            rec = {"utt": u["id"], "i": i, "text": w["text"], "norm": w["norm"], "start": w["start"], "end": w["end"], "edit": w.get("edit"),
                   "conf": w.get("conf"), "flags": sorted(fl)}
            if "time_window" in fl and "time_engines" not in fl:
                est.append(rec)
            elif w.get("edit") == "kept":
                anch.append(rec)
    rng.shuffle(anch)
    rng.shuffle(est)
    pick, seen = [], set()
    for src, k in ((anch, 8), (est, n - 8)):
        for r in src:
            if r["utt"] in seen:
                continue
            seen.add(r["utt"])
            r["timing_class"] = "engine-anchored" if src is anch else "editor-estimated"
            pick.append(r)
            if sum(1 for p in pick if p["timing_class"] == r["timing_class"]) >= k:
                break
    pick.sort(key=lambda r: r["start"])
    return pick


def build_inputs() -> dict:
    """audit_inputs.json for the pod stage: sample clips (best + second phone), scoring windows, w2v windows (master phone, timing words and
    sample utterances), speaker windows (every utterance on its best phone)."""
    doc = final_doc()
    clocks = load_clocks()
    lev = phone_levels()
    sample = json.loads((AUDIT / "sample.json").read_text(encoding="utf-8")) if (AUDIT / "sample.json").exists() else build_sample()
    durs = {c: wav_dur(c) for c in PHONES}
    clips = []
    for r in sample["utterances"]:
        for rank, cam in (("best", r["best"]), ("second", r["second"])):
            t0, t1 = to_phone(clocks, cam, r["start"] - PAD, r["end"] + PAD, durs[cam])
            clips.append({"id": f"{r['id']}_{cam}", "utt": r["id"], "cam": cam, "rank": rank, "t0": round(t0, 3), "t1": round(t1, 3),
                          "m0": r["start"] - PAD, "m1": r["end"] + PAD})
    timing = words_for_timing()
    w2v = []
    for w in timing:                              # 1 s windows on the master phone, centred on the claimed word
        mid = (w["start"] + w["end"]) / 2
        t0, t1 = to_phone(clocks, "01", mid - 0.5, mid + 0.5, durs["01"])
        w2v.append({"id": f"w_{w['utt']}_{w['i']}", "kind": "word", "cam": "01", "t0": round(t0, 3), "t1": round(t1, 3), "utt": w["utt"], "i": w["i"]})
    for r in sample["utterances"]:                # whole sampled utterances on the master phone (context alignment)
        t0, t1 = to_phone(clocks, "01", r["start"] - 0.5, r["end"] + 0.5, durs["01"])
        w2v.append({"id": f"u_{r['id']}", "kind": "utt", "cam": "01", "t0": round(t0, 3), "t1": round(t1, 3), "utt": r["id"]})
    for w in timing:                              # the utterance of every timing word, too (context alignment)
        if not any(x["id"] == f"u_{w['utt']}" for x in w2v):
            u = next(u for u in doc["utterances"] if u["id"] == w["utt"])
            t0, t1 = to_phone(clocks, "01", u["start"] - 0.5, u["end"] + 0.5, durs["01"])
            w2v.append({"id": f"u_{u['id']}", "kind": "utt", "cam": "01", "t0": round(t0, 3), "t1": round(t1, 3), "utt": u["id"]})
    speaker = []
    for u in doc["utterances"]:
        snr = span_snr(lev, clocks, u["start"], u["end"])
        if not snr:
            continue
        order = sorted(snr, key=snr.get, reverse=True)
        item = {"id": u["id"], "label": u["speaker"], "voice": u["voice"], "best": order[0], "win": {}}
        item["win"]["best"] = [round(x, 3) for x in to_phone(clocks, order[0], u["start"], u["end"], durs[order[0]])]
        if len(order) > 1:
            item["second"] = order[1]
            item["win"]["second"] = [round(x, 3) for x in to_phone(clocks, order[1], u["start"], u["end"], durs[order[1]])]
        speaker.append(item)
    laughter_cams = {c: round(clocks[c].offset, 4) for c in ("00", "01", "02")}
    all_cams_offsets = {c: round(clocks[c].offset, 4) for c in ("01", "02", "05")}
    inp = {"what": "S5 audit inputs of pod/stage_audit.py", "clips": clips, "timing_words": timing, "w2v": w2v, "scores": scoring_entries(clocks, lev),
           "speaker": speaker, "laughter_cams": laughter_cams, "all_cams_offsets": all_cams_offsets, "cams": PHONES}
    (AUDIT / "audit_inputs.json").write_text(json.dumps(inp, indent=1, ensure_ascii=False), encoding="utf-8")
    return inp


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    step = argv[0] if argv else "sample"
    if step == "sample":
        s = build_sample()
        print(json.dumps({"count": s["count"], "median_snr_best": s["median_snr_best"], "ids": [r["id"] for r in s["utterances"]]}))
    elif step == "inputs":
        i = build_inputs()
        print(json.dumps({k: len(v) for k, v in i.items() if isinstance(v, list)}))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
