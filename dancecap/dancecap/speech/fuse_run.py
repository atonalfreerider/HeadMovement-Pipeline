"""The S2 pipeline end to end: consensus_v0.json, disagreement batches, fusion_metrics.json (local, minutes; no pod).

    python -m dancecap.speech fuse         # consensus_v0.json + disagreements/batch_NN.json
    python -m dancecap.speech fuse-eval    # fusion_metrics.json (leave-one-source-out, ablations, reliability curve)
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from . import SCHEMA, SPEECH, TAKE, fuse, fuse_batches as FB, fuse_eval as FE, fuse_levels, fuse_post as FP
from .sources import Clock, load_clocks

OUT_CONSENSUS = SPEECH / "consensus_v0.json"
OUT_METRICS = SPEECH / "fusion_metrics.json"
OUT_TEXT = SPEECH / "transcript_v0.txt"
NL = chr(10)
OUT_BATCHES = SPEECH / "disagreements"

FILL_INTERJECTIONS = False
PRODUCTION = dict(weight_mode="cal", cal_gamma=1.0, refine_rounds=2, compounds=True)


def production_params(**kw) -> fuse.Params:
    d = dict(PRODUCTION)
    d.update(kw)
    return fuse.Params(**d)


def _recap_dir() -> str | None:
    cfg = SPEECH / "config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8")).get("source_dir")
    return None


def load_inputs(with_levels: bool = True):
    clocks = load_clocks(recap_dir=_recap_dir())
    levels = fuse_levels.load_levels(clocks) if with_levels else None
    raw = fuse.load_hyps(SPEECH / "raw", clocks)
    if levels is not None:
        fuse.attach_levels(raw, levels)
    return clocks, levels, raw


def overlap_windows(diar: dict) -> list[tuple[float, float]]:
    w = [(o["start"], o["end"]) for o in diar.get("overlaps_sortformer", [])]
    return sorted(w)


def run_unplaced(diar: dict, cam: str, glossary: dict, naming_conf: float) -> dict | None:
    """Fuse the engines of a phone that has no clock (04-Recap-Tony), in the phone's own seconds."""
    un = (diar.get("unplaced_sources") or {}).get(cam)
    own = Clock(cam, 1.0, 0.0, "own seconds")
    hyps = fuse.load_hyps(SPEECH / "raw", {cam: own}, cams=[cam])
    if not hyps:
        return None
    P = production_params(weight_mode="prior", refine_rounds=2)
    C = fuse.consensus(hyps, P)
    words = FP.build_words(C, glossary)
    turns_in = (un or {}).get("turns", [])
    spk = []
    if turns_in:
        d = {"turns": [{"start": t["start"], "end": t["end"], "voice": t["voice"], "name": t.get("name"), "confidence": min(1.0, 0.4 + float(t.get("margin", 0.0)))} for t in turns_in]}
        spk = FP.assign_speakers(words, d, naming_conf)
    if not spk:
        spk = [{"voice": None, "name": None, "voice_conf": 0.0, "speaker_conf": 0.0, "speaker_edge": False, "interjection": False} for _ in words]
    turns = FP.build_turns(words, spk, [], diar)
    return {"clock": "the phone's own seconds (no clock: not part of any cross-phone step)", "hypotheses": len(hyps), "words": len(words), "turns": turns,
            "voices": "provisional: ECAPA/TitaNet windows matched to the global centroids (S1), not a diarization of this clip",
            "note": "04-Recap-Tony is a question-and-answer clip that R1 could not place in the 6 minutes; its transcript is here for completeness and has no disagreement batches."}


def _mmss(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60)}:{t % 60:04.1f}"


def write_readable(doc: dict, path: Path = OUT_TEXT) -> Path:
    """The fused transcript as plain text for a human: one line per turn, uncertain words (vote share < 0.6) between asterisks, events inline."""
    lines = ["Fused transcript v0 (NOT adjudicated). Master clock = 01-Recap-Tina video time. *word* = the vote gave it less than 60 % of the weight;",
             "speaker names rest on a prior (naming confidence %s), see consensus_v0.json. The emoji is the laughing-face token at the time of a laugh / joke candidate." %
             doc["naming"].get("naming_confidence"), ""]
    evs = {e["id"]: e for e in doc["events"]}
    for t in doc["turns"]:
        evt = [evs[i] for i in t.get("events", []) if i in evs]
        lines.append(f"[{_mmss(t['start'])} .. {_mmss(t['end'])}] {t['speaker'] or '?'}{' (second voice)' if t['interjection'] else ''}: " + FP.render(t["words"], evt, mark_below=0.6))
    lines += ["", "Events:"] + [f"  {e['id']} {e['type']} at {_mmss(e['t'])} ({e['t0']}-{e['t1']} s) token {e.get('token')} {e.get('confidence', '')}{' - needs the owner' if e.get('needs_owner_confirmation') else ''}"
                                for e in doc["events"]]
    lines += ["", "Loud stretches without a word (candidates for reactions): " + "; ".join(f"{w['t0']}-{w['t1']} s ({w['phones']} phones, +{w['peak_excess_db']} dB)" for w in doc["wordless_sounds"])]
    path.write_text(NL.join(lines) + NL, encoding="utf-8")
    return path


def run_fuse(args=None) -> dict:
    t_start = time.time()
    clocks, levels, raw = load_inputs()
    diar = json.loads((SPEECH / "diarization_global.json").read_text(encoding="utf-8"))
    em = json.loads((SPEECH / "events_master.json").read_text(encoding="utf-8")) if (SPEECH / "events_master.json").exists() else {}
    glossary = FP.load_glossary(SPEECH / "glossary.json")
    names = {int(k): v for k, v in {str(v["voice"]): v.get("name") for v in diar["voices"]}.items()}
    naming_conf = float(diar["naming"].get("naming_confidence", 0.85))
    ov = overlap_windows(diar)
    P = production_params()
    C = fuse.consensus(raw, P, overlaps=ov)
    words = FP.build_words(C, glossary)
    spk = FP.assign_speakers(words, diar, naming_conf)
    runs = FP.speaker_runs(diar["turns"])

    def base_voice_at(t: float):
        for r in runs:
            if r["start"] - 0.2 <= t <= r["end"] + 0.2:
                return r["voice"]
        return None

    wins = FP.interjection_windows(diar, base_voice_at)
    FP.apply_interjections(words, spk, wins, naming_conf, names)
    events = FP.gather_events(em, diar)
    laughs = [(e["t0"], e["t1"]) for e in events if e["type"] == "laughter"]
    # filling a quiet second voice from the best-agreeing reading was tried and produced junk on this audio (whisper's smeared "and" for the hair line, hallucinated
    # words on the laugh); it stays available (FP.fill_interjections) but is off: the disagreement spans carry those readings instead
    filled = FP.fill_interjections(words, spk, wins, C, naming_conf, names, glossary, laughs) if FILL_INTERJECTIONS else []
    turns = FP.build_turns(words, spk, events, diar)
    flat_words = [w for t in turns for w in t["words"]]
    sentences = [s for t in turns for s in t["sentences"]]
    gl_hits = FP.glossary_hits(flat_words, glossary)
    sus = FP.suspects(flat_words)
    wordless = FP.wordless_events(levels, flat_words)
    for e in events:
        if e.get("needs_owner_confirmation"):
            e["nearby_wordless_sounds"] = [w for w in wordless if w["t0"] <= e["t1"] + 3.0 and w["t1"] >= e["t0"] - 1.0]
    jokes = FP.joke_scan(turns, diar, names, [g["canonical"] for g in glossary.values() if g["kind"] == "person_other"])
    owner_facts, windows = FB.load_owner_facts()
    BP = FB.BatchParams(owner_windows=tuple(windows))
    region_hyps = FB.load_region_hyps(SPEECH / "raw_regions", clocks)
    spans = FB.make_spans(C, flat_words, turns, sentences, sus, glossary, region_hyps, levels, clocks, windows, BP, owner_facts, diar.get("interjections", []))
    created = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {"take": TAKE, "created": created, "fusion": {"params": P.__dict__, "hypotheses": len(raw), "slots": len(C.slots)}}
    paths = FB.write_batches(spans, OUT_BATCHES, BP, meta)
    unplaced = run_unplaced(diar, "04", glossary, naming_conf)
    ws = [w for w in flat_words if not w["filler"]]
    doc = {
        "schema": SCHEMA + "/consensus_v0", "take": TAKE, "created": created, "stage": "v0 = fused, NOT adjudicated (no language-model edit, no owner confirmation)",
        "clock": "master = 01-Recap-Tina video time (seconds); every phone moved with sync/sync.json (slope * t + offset)",
        "method": {
            "hypotheses": [h.key for h in raw], "voting_engines": fuse.VOTING, "excluded": ["whisper :auto (language-id noise, duplicates :en)", "whisper :pt (translates)", "04-Recap-Tony (no clock)"],
            "alignment": "ROVER-style confusion network: progressive dynamic-programming alignment with sound/spelling substitution cost, gap cost, time penalty, leave-one-out refinement",
            "weights": "logistic P(token = vote of all the others) from engine, own word confidence, local SNR, distance proxy, length / duration" if P.weight_mode == "cal" else P.weight_mode,
            "calibration": C.calibration, "params": P.__dict__,
            "speakers": "diarization_global.json turns (naming_confidence %.2f is a prior from AudioSet gender tags + pitch, not addressing), boundaries snapped to the nearest silence between words; "
                        "interjection windows from the same file" % naming_conf,
            "punctuation": "voted among the hypotheses that wrote the winning word; sentence breaks also at pauses >= 1.2 s",
        },
        "stats": {
            "words": len(ws), "fillers": len(flat_words) - len(ws), "turns": len(turns), "slots": len(C.slots), "slots_with_word": sum(1 for r in C.results if r.winner is not None),
            "share_mean": round(float(np.mean([w["share"] for w in flat_words])), 4),
            "share_ge_0.95": round(float(np.mean([w["share"] >= 0.95 for w in flat_words])), 4),
            "share_ge_0.80": round(float(np.mean([w["share"] >= 0.80 for w in flat_words])), 4),
            "share_lt_0.50": round(float(np.mean([w["share"] < 0.50 for w in flat_words])), 4),
            "disagreement_spans": len(spans), "batches": len(paths), "interjection_windows": len(wins), "interjection_windows_filled": filled,
            "seconds": round(time.time() - t_start, 1),
        },
        "voices": diar["voices"], "naming": {k: diar["naming"].get(k) for k in ("assignment", "naming_confidence", "evidence", "note")},
        "turns": turns,
        "events": events, "wordless_sounds": wordless,
        "glossary_hits": gl_hits, "suspects": sus,
        "jokes": {"diarization_candidates": diar.get("jokes", []), "emoji_tokens": diar.get("emoji_tokens", []), "owner_facts": [f["text"] for f in owner_facts], "scan": jokes,
                  "note": "The words are the vote of the engines. None of them writes 'Kadu' in the opening; the owner says Larissa says it as a joke. The disagreement spans of the opening "
                          "carry every reading and the owner facts for the editor. The emoji tokens are at S1's times (2.0 s = the only detected laugh, alternative 5.08 s; 6.8 s = after the hair line)."},
        "unplaced_sources": {"04": unplaced} if unplaced else {},
        "limitations": [
            "no ground truth: all scores are agreement with held-out transcripts (fusion_metrics.json)",
            "errors that every engine shares (a name nobody knows, 'lights engage' for 'legs engaged') survive the vote: the suspects list and the disagreement spans catch the known ones",
            "a quiet second voice (Kadu at 5.2-6.8 s) is dropped by the vote; its best-agreeing reading is inserted as a flagged mini-turn",
            "speaker per word is the diarization turn (+-0.4 s) snapped to silences; naming rests on a prior",
            "punctuation and capitals are a draft",
        ],
    }
    OUT_CONSENSUS.write_text(json.dumps(FP.json_safe(doc), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_readable(doc)
    return {"consensus": str(OUT_CONSENSUS), "words": len(ws), "turns": len(turns), "spans": len(spans), "batches": [str(p) for p in paths], "stats": doc["stats"]}


def clone_for_check(raw):
    return FE.clone_hyps(raw)


def run_eval(configs: list[str] | None = None, log=print) -> dict:
    """fusion_metrics.json: the production configuration (LOSO, LOSEO, hat, reliability curve) plus ablations (LOSO)."""
    t0 = time.time()
    clocks, levels, raw = load_inputs()
    prod = FE.clone_hyps(raw)
    fuse.merge_compounds(prod)
    P = production_params()
    if configs and set(configs) == {"clock"}:        # refresh only the clock check of an existing metrics file (fast)
        out = json.loads(OUT_METRICS.read_text(encoding="utf-8"))
        out["clock_check"] = FE.clock_residuals(fuse.consensus(clone_for_check(raw), P))
        OUT_METRICS.write_text(json.dumps(FP.json_safe(out), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return out
    out: dict = {"schema": SCHEMA + "/fusion_metrics", "take": TAKE, "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "what": "pseudo-WER = word errors of a transcript scored against another transcript, canonical words, fillers dropped, 61-359 s (all six phones recording)",
                 "params": P.__dict__}
    log("LOSO (production)")
    out["loso"] = FE.loso(prod, raw, P, log=log)
    log("LOSEO (production)")
    out["loseo"] = FE.loso(prod, raw, P, cross_engine=True, log=log)
    log("hat")
    out["three_cornered_hat"] = FE.three_cornered_hat(prod, raw, P, log=log)
    log("reliability")
    out["reliability_curve"] = FE.reliability_curve(prod, raw, P, log=log)
    log("clock check")
    out["clock_check"] = FE.clock_residuals(fuse.consensus(clone_for_check(raw), P))
    abl = {}
    for name, kw in [("uniform", dict(weight_mode="uniform")), ("prior", dict(weight_mode="prior")), ("cal_no_compounds", dict(compounds=False)),
                     ("cal_refine1", dict(refine_rounds=1))]:
        if configs and name not in configs:
            continue
        log(f"ablation {name}")
        r = FE.loso(prod, raw, production_params(**kw), log=lambda *a: None)
        abl[name] = {k: r[k] for k in ("fused_pwer_mean", "fused_beats_best_single_transcript", "fused_beats_best_engine_mean", "by_held_out_phone")}
    out["ablations_loso"] = abl
    sens = {}
    for name, kw in [("cal_gamma2", dict(cal_gamma=2.0)), ("family_balance1", dict(family_balance=1.0)), ("null_gamma0.7", dict(null_gamma=0.7)), ("null_gamma1.3", dict(null_gamma=1.3)),
                     ("gap0.45", dict(gap=0.45)), ("gap0.8", dict(gap=0.8))]:
        if configs and name not in configs:
            continue
        log(f"sensitivity {name}")
        r = FE.loso(prod, raw, production_params(refine_rounds=1, **kw), log=lambda *a: None)
        sens[name] = {k: r[k] for k in ("fused_pwer_mean", "fused_beats_best_single_transcript")}
    out["sensitivity_loso_refine1"] = sens
    out["seconds"] = round(time.time() - t0, 1)
    OUT_METRICS.write_text(json.dumps(FP.json_safe(out), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return out
