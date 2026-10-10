"""The global diarization: joint voices on the master clock, names, interjections and overlap, utterances, jokes -> diarization_global.json.

    python -m dancecap.speech diarize            # reads work/<take>/speech/raw/, writes diarization_global.json

Two routes, combined:

  A. moments (cluster.py): the windows of all phones fused per 0.75 s of master time, clustered into global voices, Viterbi-smoothed:
     the BASE TURNS. Stable, but a 0.7 s interjection inside a long turn is smoothed away.
  B. Sortformer segments (pod stage sortformer + segembed): every phone's overlap-aware segments, each embedded on its own and given the
     global voice whose centroid is nearest; the segments of several phones that agree in time and voice are merged. Where route B
     finds a DIFFERENT voice than the base turn for >= 0.4 s, supported by >= 2 phones (or by the only phone that covers the time), that
     is written as an INTERJECTION (voice, start, end, support) on top of the base turn; where it coincides with the base voice speaking
     it is an OVERLAP.

Everything on the master clock = 01-Recap-Tina video time. A phone without a clock (04-Recap-Tony until the audio alignment places it) is
left out of every cross-phone step and listed in `sources.not_placed`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import SPEECH, SOURCES, MASTER, SCHEMA, TAKE
from . import cluster, events as events_mod, identity, report, texts
from .sources import load_clocks

PRIMARY = ["parakeet-tdt-0.6b-v3", "whisper-large-v3:en", "canary-1b-v2"]


# ------------------------------------------------------------------------------------------------ route B: Sortformer segments


def cam_means(raw_dir: Path, clocks: dict, models: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Per phone and model the mean of the L2-normalised window embeddings (the channel offset cluster.load_table subtracts)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        f = d / "embed.npz"
        if clocks.get(d.name) is None or not f.exists():
            continue
        z = np.load(f)
        out[d.name] = {m: cluster.l2n(z[m].astype(np.float64)).mean(axis=0) for m in models if m in z.files}
    return out


def segment_labels(raw_dir: Path, clocks: dict, cent: np.ndarray, models: list[str], means: dict) -> list[dict]:
    """Every Sortformer segment of every placed phone with its nearest global voice: [{cam, t0, t1 (source), m0, m1 (master), spk, label,
    margin, sim: [..]}]."""
    out: list[dict] = []
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        cam = d.name
        clk = clocks.get(cam)
        f = d / "segembed.npz"
        if clk is None or not f.exists() or cam not in means:
            continue
        z = np.load(f)
        blocks = []
        for m in models:
            e = cluster.l2n(z[m].astype(np.float64))
            blocks.append(cluster.l2n(e - means[cam][m][None, :]))
        emb = np.concatenate(blocks, axis=1) / np.sqrt(len(blocks))
        sim = emb @ cent.T
        order = np.argsort(-sim, axis=1)
        for i in range(len(emb)):
            best = int(order[i, 0])
            second = float(sim[i, order[i, 1]]) if sim.shape[1] > 1 else -1.0
            out.append({"cam": cam, "t0": float(z["t0"][i]), "t1": float(z["t1"][i]), "m0": clk.to_master(float(z["t0"][i])),
                        "m1": clk.to_master(float(z["t1"][i])), "spk": int(z["spk"][i]), "label": best, "margin": float(sim[i, best] - second),
                        "rms": float(z["rms_db"][i]), "f0": float(z["f0_med"][i])})
    return out


def merge_segments(segs: list[dict], min_overlap: float = 0.5) -> list[dict]:
    """Segments of several phones that cover the same time with the same voice -> one event {label, m0, m1, support, cams, margin}.
    Greedy by start time; two segments merge when their overlap is >= min_overlap of the shorter one."""
    segs = sorted(segs, key=lambda s: (s["label"], s["m0"]))
    groups: list[list[dict]] = []
    for s in segs:
        placed = False
        for g in reversed(groups[-6:]):
            if g[0]["label"] != s["label"]:
                continue
            a0, a1 = min(x["m0"] for x in g), max(x["m1"] for x in g)
            ov = min(a1, s["m1"]) - max(a0, s["m0"])
            if ov >= min_overlap * min(a1 - a0, s["m1"] - s["m0"]) and ov > 0 and all(x["cam"] != s["cam"] for x in g):
                g.append(s)
                placed = True
                break
        if not placed:
            groups.append([s])
    out = []
    for g in groups:
        out.append({"label": g[0]["label"], "m0": round(float(np.median([x["m0"] for x in g])), 3), "m1": round(float(np.median([x["m1"] for x in g])), 3),
                    "support": len({x["cam"] for x in g}), "cams": sorted({x["cam"] for x in g}), "margin": round(float(np.mean([x["margin"] for x in g])), 3)})
    out.sort(key=lambda e: e["m0"])
    return out


def covering_cams(clocks: dict, durations: dict, t0: float, t1: float) -> int:
    n = 0
    for cam, clk in clocks.items():
        if clk is None or cam not in durations:
            continue
        a, b = clk.to_master(0.0), clk.to_master(durations[cam])
        if a <= t0 and b >= t1:
            n += 1
    return n


def interjections(events: list[dict], mom_t0: np.ndarray, mom_t1: np.ndarray, path: np.ndarray, clocks: dict, durations: dict,
                  min_len: float = 0.4) -> list[dict]:
    """Route-B events whose voice differs from the base turn at their midpoint, or that overlap it (same voice active twice is not an event)."""
    out = []
    for e in events:
        if e["m1"] - e["m0"] < min_len:
            continue
        need = min(2, max(1, covering_cams(clocks, durations, e["m0"], e["m1"])))
        if e["support"] < need:
            continue
        mid = 0.5 * (e["m0"] + e["m1"])
        k = int(np.searchsorted(mom_t1, mid, side="right"))
        base = int(path[k]) if k < len(path) and mom_t0[k] <= mid else None
        if base == e["label"]:
            continue
        # base None: the moment grid has no speech there (a pause the base turns skip) but several phones' Sortformer hears a voice
        out.append({"start": e["m0"], "end": e["m1"], "voice": e["label"], "base_voice": base, "support": e["support"], "cams": e["cams"],
                    "margin": e["margin"], "kind": "interjection_or_overlap" if base is not None else "extra_turn"})
    return out


# ------------------------------------------------------------------------------------------------ overlap from Sortformer alone


def sortformer_overlaps(raw_dir: Path, clocks: dict, min_len: float = 0.25) -> list[dict]:
    """Intervals (master clock) where one phone's Sortformer has >= 2 speakers active, merged over phones: {start, end, cams}."""
    from .pod.stage_embed import core_segments  # noqa: F401  (same core rule as segembed)
    per_cam = []
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        f = d / "sortformer.json"
        clk = clocks.get(d.name)
        if clk is None or not f.exists():
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        overlap = max(0.0, doc["windows"][0]["t1"] - doc["windows"][1]["t0"]) if len(doc["windows"]) > 1 else 0.0
        for k, w in enumerate(doc["windows"]):
            lo = w["t0"] + (overlap / 2 if k > 0 else 0.0)
            hi = w["t1"] - (overlap / 2 if k < len(doc["windows"]) - 1 else 0.0)
            edges = []
            for s in w["segments"]:
                a, b = max(s["start"], lo), min(s["end"], hi)
                if b > a:
                    edges += [(a, 1), (b, -1)]
            edges.sort()
            n, t_prev = 0, None
            for t, dlt in edges:
                if n >= 2 and t_prev is not None and t - t_prev >= min_len:
                    per_cam.append((d.name, clk.to_master(t_prev), clk.to_master(t)))
                n += dlt
                t_prev = t
    per_cam.sort(key=lambda r: r[1])
    merged: list[dict] = []
    for cam, a, b in per_cam:
        if merged and a <= merged[-1]["end"] + 0.1:
            merged[-1]["end"] = max(merged[-1]["end"], b)
            merged[-1]["cams"] = sorted(set(merged[-1]["cams"]) | {cam})
        else:
            merged.append({"start": round(a, 3), "end": round(b, 3), "cams": [cam]})
    for m in merged:
        m["end"] = round(m["end"], 3)
    return merged


# ------------------------------------------------------------------------------------------------ gender tags


def gender_scores(raw_dir: Path, clocks: dict, mom: cluster.Moments, path: np.ndarray) -> dict[int, dict]:
    """Per voice: mean(P(female speech) - P(male speech)) of the AudioSet tags of the windows that cover its moments (all phones)."""
    male, fem = "Male speech, man speaking", "Female speech, woman speaking"
    rows = []
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        f = d / "events.json"
        clk = clocks.get(d.name)
        if clk is None or not f.exists():
            continue
        for w in json.loads(f.read_text(encoding="utf-8"))["windows"]:
            rows.append((clk.to_master(w["t0"]), clk.to_master(w["t1"]), w["p"].get(male, 0.0), w["p"].get(fem, 0.0)))
    arr = np.array(rows) if rows else np.zeros((0, 4))
    g = np.full((len(mom.t0), 2), np.nan)
    for i in range(len(mom.t0)):
        tc = 0.5 * (mom.t0[i] + mom.t1[i])
        sel = (arr[:, 0] + 0.5 <= tc) & (tc <= arr[:, 1] - 0.5) if len(arr) else np.zeros(0, bool)
        if sel.any():
            g[i] = [arr[sel, 2].mean(), arr[sel, 3].mean()]
    out = {}
    for v in sorted(set(path.tolist())):
        sel = (path == v) & ~np.isnan(g[:, 0])
        if sel.any():
            out[int(v)] = {"p_male": round(float(g[sel, 0].mean()), 3), "p_female": round(float(g[sel, 1].mean()), 3),
                           "f_minus_m": round(float((g[sel, 1] - g[sel, 0]).mean()), 3), "share_female_gt_male": round(float((g[sel, 1] > g[sel, 0]).mean()), 3)}
    return out


# ------------------------------------------------------------------------------------------------ utterances


def base_stream(words: list[texts.Word]) -> list[texts.Word]:
    """One stream that covers the whole timeline: parakeet of the master phone from 0 s, parakeet of the phone that started earlier before
    that (phone 00 starts 8.9 s before the master)."""
    master = [w for w in words if w.engine == PRIMARY[0] and w.cam == MASTER]
    early = [w for w in words if w.engine == PRIMARY[0] and w.cam != MASTER and w.m0 < (master[0].m0 if master else 0.0) - 0.2]
    return sorted(early + master, key=lambda w: w.m0)


def route_b_voice(t0: float, t1: float, inter: list[dict]) -> int | None:
    """The voice of a route-B interjection / extra turn that covers most of [t0, t1], else None."""
    best, best_ov = None, 0.0
    for it in inter:
        ov = min(t1, it["end"]) - max(t0, it["start"])
        if ov > best_ov and ov >= 0.5 * (t1 - t0):
            best, best_ov = it["voice"], ov
    return best


def build_utterances(stream: list[texts.Word], mom, path, marg, names: dict, gap: float = 0.8, inter: list[dict] | None = None) -> list[dict]:
    assign = names.get("assignment", {})
    inter = inter or []
    conf_name = names.get("naming_confidence", 0.0)
    utts: list[dict] = []
    cur: list[texts.Word] = []
    cur_voice = None

    def flush():
        if not cur:
            return
        a, b = cur[0].m0, cur[-1].m1
        v, c = identity.voice_at(a, b, mom.t0, mom.t1, path, marg)
        vb = route_b_voice(a, b, inter)
        if vb is not None and vb != v:          # Sortformer + embeddings on several phones say a different voice speaks these words
            v, c = vb, 0.5
        probs = [w.prob for w in cur if w.prob is not None]
        utts.append({"start": round(a, 3), "end": round(b, 3), "voice": v, "name": assign.get(str(v)) if v is not None else None,
                     "turn_confidence": c, "confidence": round(c * (conf_name if assign.get(str(v), "").startswith(("Larissa", "Kadu")) else 0.5), 3),
                     "text": " ".join(w.text for w in cur), "n_words": len(cur), "engine": PRIMARY[0],
                     "word_conf_mean": round(float(np.mean(probs)), 3) if probs else None})

    for w in stream:
        v, _ = identity.voice_at(w.m0, w.m1, mom.t0, mom.t1, path, marg)
        vb = route_b_voice(w.m0, w.m1, inter)
        if vb is not None:
            v = vb
        if cur and (w.m0 - cur[-1].m1 > gap or v != cur_voice):
            flush()
            cur = []
        cur.append(w)
        cur_voice = v
    flush()
    return utts


def alt_texts(words: list[texts.Word], a: float, b: float, cams: tuple[str, ...] = ("01", "00", "02")) -> dict:
    """The words of every engine on a few phones inside [a, b] (for joke search and for the owner to read side by side)."""
    out: dict[str, list[str]] = {}
    for w in words:
        if w.cam in cams and a - 0.1 <= 0.5 * (w.m0 + w.m1) <= b + 0.1:
            out.setdefault(f"{w.engine}@{w.cam}", []).append(w.text)
    return {k: " ".join(v) for k, v in out.items()}


SELF_INTRO = __import__("re").compile(r"\b(this is|i'?m|i am|my name'?s?(?: is)?|call me)\s+([\w' -]{1,24})", __import__("re").I)


def unresolved_introductions(utts: list[dict], words: list[texts.Word], reactions: list[dict], horizon: float = 20.0) -> list[dict]:
    """Self-introduction phrases ("this is X", "I'm X") in the first `horizon` seconds or within 8 s of a laughter event, with the
    competing readings of every engine. ASR cannot decide a name it does not know; this lists the readings for a human."""
    out = []
    laugh = [e for e in reactions if e["kind"] == "laughter"]
    for u in utts:
        near = u["start"] < horizon or any(abs(u["start"] - e["t0"]) <= 8 for e in laugh)
        if not near:
            continue
        alts = alt_texts(words, u["start"] - 0.5, u["end"] + 0.5)
        found = {}
        for key, text in alts.items():
            mt = SELF_INTRO.search(text)
            if mt:
                found[key] = " ".join(mt.group(0).split()[:6])
        if found:
            out.append({"start": u["start"], "end": u["end"], "voice": u["voice"], "name": u["name"], "readings": found,
                        "n_readings": len(set(found.values())), "kadu_heard_by_any_engine": any("kadu" in v.lower() or "cadu" in v.lower() for v in found.values()),
                        "laughter_nearby": [e["id"] for e in laugh if abs(u["start"] - e["t0"]) <= 8]})
    return out


# ------------------------------------------------------------------------------------------------ sources without a clock


def unplaced_turns(raw_dir: Path, cam: str, cent: np.ndarray, models: list[str], penalty: float = 0.15) -> dict:
    """Voice turns of a phone that has NO clock, in its OWN seconds: its windows (channel mean removed from its own windows) matched to the
    global centroids and smoothed. Provisional: with 77 s of two voices the phone's mean is a weak estimate of its channel."""
    f = raw_dir / cam / "embed.npz"
    if not f.exists() or cent.shape[0] == 0:
        return {}
    z = np.load(f)
    blocks = []
    for m in models:
        e = cluster.l2n(z[m].astype(np.float64))
        blocks.append(cluster.l2n(e - e.mean(axis=0, keepdims=True)))
    emb = np.concatenate(blocks, axis=1) / np.sqrt(len(blocks))
    sim = emb @ cent.T
    t0, t1 = z["t0"].astype(float), z["t1"].astype(float)
    gaps = np.zeros(len(t0), bool)
    if len(t0) > 1:
        gaps[:-1] = (t0[1:] - t1[:-1]) > 1.5
    path = cluster.viterbi_smooth(sim, gaps[:-1] if len(gaps) else gaps, penalty=penalty)
    marg = cluster.margins(sim, path)
    turns, cur = [], None
    for i in range(len(path)):
        if cur is not None and cur["voice"] == int(path[i]) and t0[i] - cur["end"] <= 0.8:
            cur["end"] = float(t1[i]); cur["_m"].append(float(marg[i]))
            continue
        if cur is not None:
            turns.append(cur)
        cur = {"start": float(t0[i]), "end": float(t1[i]), "voice": int(path[i]), "_m": [float(marg[i])]}
    if cur is not None:
        turns.append(cur)
    for t in turns:
        t["margin"] = round(float(np.mean(t.pop("_m"))), 3)
    return {"clock": "the phone's own seconds (no clock)", "windows": int(len(t0)), "voice_share": {str(v): round(float(np.mean(path == v)), 3) for v in sorted(set(path.tolist()))},
            "turns": turns}


# ------------------------------------------------------------------------------------------------ the whole thing


def run(raw_dir: Path | None = None, out: Path | None = None, recap_dir: str | None = None) -> dict:
    raw_dir = raw_dir or SPEECH / "raw"
    out = out or SPEECH / "diarization_global.json"
    t_start = time.time()
    clocks = load_clocks(recap_dir=recap_dir)
    placed = sorted(c for c, k in clocks.items() if k is not None and (raw_dir / c / "embed.npz").exists())
    not_placed = sorted(c for c in SOURCES if c not in placed)
    cl = cluster.run(raw_dir, clocks)
    tab, mom, path, sim = cl["table"], cl["moments"], cl["labels"], cl["sim"]
    marg = cluster.margins(sim, path)
    gender = gender_scores(raw_dir, clocks, mom, path)
    voices = cl["voices"]
    for v in voices:
        g = gender.get(v["voice"])
        if g:
            v["gender_f_minus_m"] = g["f_minus_m"]
            v["gender_tags"] = g
    words = texts.load_words(raw_dir, clocks)
    mentions = identity.find_mentions(words)
    for m in mentions:
        m.speaker, m.speaker_conf = identity.voice_at(m.t0 - 0.2, m.t1 + 0.3, mom.t0, mom.t1, path, marg)
    for tr in cl["turns"]:
        tr["confidence"] = tr.get("confidence", 0.0)
    names = identity.name_voices(voices, mentions, cl["turns"])
    assign = names.get("assignment", {})
    for v in voices:
        v["name"] = assign.get(str(v["voice"]))
    # route B
    cent = cluster.centroids(mom.emb, path)
    models = cl["models"]
    means = cam_means(raw_dir, clocks, models)
    segs = segment_labels(raw_dir, clocks, cent, models, means) if cent.shape[0] else []
    merged = merge_segments(segs)
    durations = {}
    for c in placed:
        vj = json.loads((raw_dir / c / "vad.json").read_text(encoding="utf-8"))
        durations[c] = vj["duration"]
    inter = interjections(merged, mom.t0, mom.t1, path, clocks, durations)
    for it in inter:
        it["name"] = assign.get(str(it["voice"]))
        it["base_name"] = assign.get(str(it["base_voice"])) if it["base_voice"] is not None else None
        if it["base_voice"] is None:
            # a pause of the base turns: if the SAME voice speaks right before and after, this is just that voice continuing (not an interjection)
            before = identity.voice_at(it["start"] - 0.8, it["start"] - 0.1, mom.t0, mom.t1, path, marg)[0]
            after = identity.voice_at(it["end"] + 0.1, it["end"] + 0.8, mom.t0, mom.t1, path, marg)[0]
            it["neighbour_voices"] = [before, after]
            it["gap_fill"] = bool(before == it["voice"] and after == it["voice"])
    overlaps = sortformer_overlaps(raw_dir, clocks)
    unplaced = {c: unplaced_turns(raw_dir, c, cent, models) for c in not_placed if (raw_dir / c / "embed.npz").exists()}
    for c, u in unplaced.items():
        for t in u.get("turns", []):
            t["name"] = assign.get(str(t["voice"]))
    # reactions
    ev = events_mod.run(raw_dir, clocks)
    reactions = ev["events"]
    # texts
    stream = base_stream(words)
    utts = build_utterances(stream, mom, path, marg, names, inter=inter)
    rwords = texts.load_regions(raw_dir.parent / "raw_regions", clocks) if (raw_dir.parent / "raw_regions").exists() else []
    all_words = words + rwords
    for u in utts:
        u["alt"] = alt_texts(all_words, u["start"], u["end"], ("01",)) if u["start"] >= 0 else alt_texts(all_words, u["start"], u["end"], ("00",))
        u["text_all"] = " ".join(u["alt"].values())
    inter_utts = []
    for it in inter:
        alt = alt_texts(all_words, it["start"] - 0.2, it["end"] + 0.2, tuple(placed))
        inter_utts.append({"start": it["start"], "end": it["end"], "voice": it["voice"], "text": " ".join(alt.values()), "alt": alt, "kind": it["kind"]})
        it["alt"] = alt
    jokes = identity.find_jokes(mentions, [dict(u, text=u["text_all"]) for u in utts] + inter_utts, names,
                                [e for e in reactions if e["kind"] in ("laughter", "cheering", "crowd")])
    intros = unresolved_introductions(utts, all_words, reactions)
    for i in intros:
        if i["laughter_nearby"] and i["n_readings"] >= 3 and not i["kadu_heard_by_any_engine"]:
            ev0 = next(e for e in reactions if e["id"] == i["laughter_nearby"][0])
            jokes.append({"kind": "self_introduction_unresolved", "t": i["start"], "t_end": i["end"], "t_token": ev0["t0"], "t_token_alt": i["end"],
                          "voice": i["voice"], "voice_name": i["name"], "contradiction": None, "emoji": identity.LAUGH_EMOJI, "readings": i["readings"],
                          "reaction": [{"id": ev0["id"], "kind": ev0["kind"], "t0": ev0["t0"], "peak": ev0["peak"]}],
                          "score": round(0.5 * names.get("naming_confidence", 0.0) + 0.15, 3),
                          "why": f"{i['n_readings']} different readings of 'this is ...' from the engines and phones, none of them 'Kadu', with laughter detected at "
                                 f"{ev0['t0']:.1f}-{ev0['t1']:.1f} s: an introduction the ASR cannot spell, said by the voice we name {i['name']}. The owner says this is "
                                 f"Larissa introducing herself as 'Kadu'; the audio must confirm the word (review clip c01)."})
    jokes.sort(key=lambda j: -j["score"])
    emoji_tokens = []
    for j in jokes:
        if j.get("emoji") and not any(abs(t["t"] - j["t_token"]) < 1.5 for t in emoji_tokens):
            emoji_tokens.append({"t": j["t_token"], "emoji": j["emoji"], "joke": j["kind"], "voice_name": j.get("voice_name"), "alt_t": j.get("t_token_alt", j.get("t_end")),
                                 "evidence": j["why"], "needs_owner_confirmation": True})
    emoji_tokens.sort(key=lambda t: t["t"])
    for t in cl["turns"]:
        t["name"] = assign.get(str(t["voice"]), None)
        ov = [o for o in overlaps if o["start"] < t["end"] and o["end"] > t["start"]]
        t["overlap_s"] = round(sum(min(o["end"], t["end"]) - max(o["start"], t["start"]) for o in ov), 2)
    doc = {
        "schema": f"{SCHEMA}/diarization_global", "take": TAKE, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t_start, 1),
        "clock": "master = 01-Recap-Tina video time; reference_time = slope * source_time + offset (work/<take>/sync/sync.json)",
        "sources": {"placed": placed, "not_placed": not_placed, "clocks": {c: {"slope": k.slope, "offset": round(k.offset, 4), "origin": k.origin}
                                                                       for c, k in clocks.items() if k is not None}},
        "method": {"route_a": "moments (0.75 s master bins, phones fused) -> average-linkage clustering on mean-normalised ECAPA+TitaNet embeddings -> Viterbi",
                   "route_b": "Sortformer segments of every phone embedded and matched to the global centroids; merged over phones",
                   "models": models, "threshold": cl["threshold"], "silhouette": cl["silhouette"], "windows": len(tab), "moments": len(mom.t0)},
        "note": identity.JOKE_NOTE,
        "voices": voices, "naming": names,
        "turns": cl["turns"], "unplaced_sources": unplaced, "interjections": inter, "overlaps_sortformer": overlaps, "utterances": utts,
        "mentions": [m.__dict__ for m in mentions], "region_mentions": [m.__dict__ for m in identity.find_mentions(rwords)], "reactions": reactions, "jokes": jokes, "emoji_tokens": emoji_tokens, "introduction_readings": intros,
        "limitations": [
            "voice names come from addressing + acoustic priors, not from a self-introduction (see note); naming_confidence is not a calibrated probability",
            "turn boundaries are +-0.4 s (0.75 s moment grid); interjections shorter than 0.4 s and voices that never speak alone are not found",
            "utterance text is parakeet of the master phone (a preview, not the fused transcript)",
            "04-Recap-Tony has no clock yet" if "04" in not_placed else "all seven sources placed"],
    }
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding="utf-8")
    doc["_arrays"] = {"moments": mom, "path": path, "marg": marg, "table": tab, "sim": sim, "words": words, "merged": merged}
    return doc
