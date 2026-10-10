"""dancecap.speech: pure-python parts (clocks, glossary, window / segment helpers, F0, joint clustering, event merging, name mentions,
voice naming, joke candidates, job spec). Offline, no GPU, no models, no audio files.

    ..\\.venv-nopo4d\\Scripts\\python -m pytest tests/test_speech.py -q
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.speech import cluster, events, glossary, identity, jobs, sources, texts  # noqa: E402
from dancecap.speech.pod import common as C  # noqa: E402
from dancecap.speech.pod import stage_canary, stage_embed, stage_events, stage_parakeet, stage_sortformer  # noqa: E402

PKG = Path(__file__).resolve().parents[1] / "dancecap" / "speech"


# ------------------------------------------------------------------------------------------------ clocks


def test_clock_roundtrip_and_sync_json(tmp_path):
    sync = {"cameras": [{"id": "00", "offset_s": -8.9, "slope": 1.00001, "confidence": 0.5},
                        {"id": "04", "offset_s": None, "slope": 1.0},
                        {"id": "03", "offset_s": 60.93, "slope": 1.0}]}
    p = tmp_path / "sync.json"
    p.write_text(json.dumps(sync))
    (tmp_path / "camera-time-offsets.json").write_text(json.dumps({"D:\\clips\\x\\02-Recap-Alyssa.mp4": 1.3469,
                                                                    "D:\\clips\\x\\01-Recap-Tina.mp4": 0.0}))
    clocks = sources.load_clocks(p, tmp_path)
    assert clocks["01"].offset == 0.0 and clocks["01"].origin == "master"
    assert clocks["00"].origin == "sync.json" and clocks["00"].slope == pytest.approx(1.00001)
    assert clocks["02"].origin == "camera-time-offsets.json" and clocks["02"].provisional      # matched by FILE NAME, not by path
    assert clocks["04"] is None                                                               # Tony: no clock until the audio alignment finds one
    c = clocks["00"]
    assert c.to_source(c.to_master(123.456)) == pytest.approx(123.456)


# ------------------------------------------------------------------------------------------------ glossary


def test_glossary_has_teachers_terms_and_rescue():
    g = glossary.build()
    keys = {t["key"] for t in g["terms"]}
    assert {"larissa", "kadu", "corredor", "counterbalance", "t axis", "weight change", "viradinha"} <= keys
    assert "Larissa" in g["prompts"]["whisper_initial_prompt"] and "Kadu" in g["prompts"]["whisper_initial_prompt"]
    assert len(g["prompts"]["whisper_initial_prompt"]) < 900            # stays inside Whisper's 224-token prompt
    by_key = {t["key"]: t for t in g["terms"]}
    assert by_key["alyssa"]["kind"] == "person_other"                    # a student, never a teacher
    assert glossary.rescue_candidates("corridor", g["terms"])[0]["term"] == "Corredor"
    assert glossary.rescue_candidates("Cadu", g["terms"])[0]["term"] == "Kadu"
    assert glossary.rescue_candidates("viradinha", g["terms"])[0]["how"] == "exact"
    assert glossary.edit_distance("kitten", "sitting") == 3


# ------------------------------------------------------------------------------------------------ pod helpers


def test_split_for_asr_and_merge():
    regs = [(0.0, 5.0), (5.5, 12.0), (30.0, 31.0), (31.5, 80.0)]
    segs = C.split_for_asr(regs, 28.0)
    assert all(b - a <= 28.0 + 1e-9 for a, b in segs)
    assert segs[0] == (0.0, 12.0) and segs[1] == (30.0, 31.0) or segs[1][0] == 30.0
    assert abs(sum(b - a for a, b in segs) - sum(b - a for a, b in regs) - 0) < 5.0                # nothing lost, only joined across short gaps
    assert C.merge_regions([(0, 1), (1.2, 2), (5, 6)], 0.5) == [(0, 2), (5, 6)]


def test_gap_concat_and_remap_keep_words_in_their_region():
    audio = np.zeros(C.SR * 10, np.float32)
    regs = [(C.SR * 1, C.SR * 3), (C.SR * 6, C.SR * 8)]
    src, table = C.concat_with_gaps(audio, regs, 1.0)
    assert len(src) == (2 + 1 + 2) * C.SR
    # a word at concatenated 3.4-3.8 s lies in the second region (which starts at 3.0 s of the concatenation) = original 6.4-6.8 s
    words = [{"start": 3.4, "end": 3.8, "word": "x"}]
    segs = [{"start": 3.2, "end": 3.9}]
    C.remap_pass(words, segs, table)
    assert words[0]["start"] == pytest.approx(6.4, abs=0.01) and words[0]["end"] == pytest.approx(6.8, abs=0.01) and words[0]["region"] == 1


def _harmonic(f0: float, secs: float = 1.0) -> np.ndarray:
    t = np.arange(int(secs * C.SR)) / C.SR
    return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)).astype(np.float32) * 0.2


def test_f0_track_finds_pitch_and_ignores_noise():
    for f in (110.0, 210.0):
        f0, cl = C.f0_track(_harmonic(f))
        v = f0[f0 > 0]
        assert len(v) > 0.8 * len(f0) and abs(float(np.median(v)) - f) < 4.0
    noise = np.random.default_rng(0).normal(0, 0.1, C.SR).astype(np.float32)
    f0, _ = C.f0_track(noise)
    assert np.mean(f0 > 0) < 0.2


def test_windows_and_stats():
    probs = np.concatenate([np.ones(100), np.zeros(100)]).astype(np.float32)    # 3.2 s speech then 3.2 s silence at 32 ms frames
    wins = stage_embed.make_windows(6.4, probs, 0.032, win=1.5, hop=0.75)
    assert wins and all(t1 <= 3.76 for t0, t1, _ in wins) and wins[-1][2] < 0.7 and wins[0][0] == 0.0
    x = np.concatenate([_harmonic(200.0, 3.2), np.zeros(int(3.2 * C.SR), np.float32)])
    st = stage_embed.window_stats(x, wins)
    assert abs(float(np.median(st["f0_med"][st["f0_med"] > 0])) - 200.0) < 5.0


def test_sortformer_and_parakeet_parsers():
    assert stage_sortformer.parse_segment("12.48 15.92 speaker_1", 50.0) == {"start": 62.48, "end": 65.92, "spk": 1}
    assert stage_sortformer.parse_segment("junk", 0.0) is None
    assert stage_sortformer.window_starts(77.3, 60.0, 50.0) == [0.0, 50.0]
    hyp = SimpleNamespace(timestamp={"word": [{"word": "Hello", "start": 0.2, "end": 0.5}, {"word": "there", "start_offset": 8, "end_offset": 10}]},
                          word_confidence=[0.9, 0.5])
    ws = stage_parakeet.parse_hypothesis(hyp, 10.0, 0.08)
    assert ws[0] == {"word": "Hello", "start": 10.2, "end": 10.5, "prob": 0.9}
    assert ws[1]["start"] == pytest.approx(10.64) and ws[1]["prob"] == 0.5
    lid = [{"start": 0, "end": 5, "lid": {"en": 0.1, "pt": 0.9}}, {"start": 5, "end": 10, "lid": {"en": 0.95, "pt": 0.01}}]
    assert stage_canary.segment_language(1, 4, lid) == "pt" and stage_canary.segment_language(6, 9, lid) == "en"
    assert stage_canary.segment_language(20, 25, lid) == "en"


def test_events_helpers():
    labels = stage_events.keep_labels({0: "Speech", 1: "Laughter", 2: "Dog", 3: "Applause"})
    assert set(labels) == {"Speech", "Laughter", "Applause"}
    t = np.arange(3 * C.SR) / C.SR
    rng = np.random.default_rng(1)
    laugh = (rng.normal(0, 0.3, t.size) * (0.55 + 0.45 * np.sin(2 * np.pi * 5.0 * t))).astype(np.float32)
    steady = rng.normal(0, 0.3, t.size).astype(np.float32)
    assert stage_events.am_band_ratio(laugh) > 2 * stage_events.am_band_ratio(steady)


# ------------------------------------------------------------------------------------------------ joint clustering


def _synthetic_table(seed=0, n_per=60, cams=("01", "02", "05")):
    """Two speakers alternating in blocks of 6 moments, heard by three phones, each with its own channel offset in embedding space."""
    rng = np.random.default_rng(seed)
    d = 24
    spk = [l2 for l2 in cluster.l2n(rng.normal(size=(3, d)))]
    chan = {c: 0.5 * cluster.l2n(rng.normal(size=d)) for c in cams}
    rows = {k: [] for k in ("cam", "m0", "m1", "f0", "f0v", "rms", "vad", "emb", "truth")}
    n = 0
    for blk in range(n_per // 6):
        who = blk % 2
        for k in range(6):
            t0 = n * 0.75
            n += 1
            for c in cams:
                e = spk[who] + chan[c] + rng.normal(0, 0.08, d)
                rows["cam"].append(c); rows["m0"].append(t0 + 0.0); rows["m1"].append(t0 + 1.5); rows["f0"].append(210.0 if who == 0 else 115.0)
                rows["f0v"].append(0.8); rows["rms"].append(-30.0 + (3 if c == "01" else 0)); rows["vad"].append(1.0); rows["emb"].append(e); rows["truth"].append(who)
    t0 = np.array(rows["m0"]); t1 = np.array(rows["m1"])
    tab = cluster.Table(np.array(rows["cam"]), t0, t1, t0, t1, np.array(rows["f0"]), np.array(rows["f0v"]), np.array(rows["rms"]), np.array(rows["vad"]),
                        np.array(rows["emb"]), ["synthetic"])
    return tab, np.array(rows["truth"])


def test_camera_mean_normalisation_separates_speakers_not_phones():
    tab, truth = _synthetic_table()
    # per-phone mean subtraction as load_table does it
    emb = tab.emb.copy()
    for c in set(tab.cam):
        m = tab.cam == c
        e = cluster.l2n(emb[m])
        emb[m] = cluster.l2n(e - e.mean(axis=0, keepdims=True))
    tab.emb = emb
    m = cluster.build_moments(tab)
    assert len(m.t0) == 60 and all(len(set(c)) == 3 for c in m.cams)              # three phones fused into one moment
    pick = cluster.pick_threshold(m.emb)
    lab = pick["labels"]
    truth_m = np.array([truth[idx[0]] for idx in m.members])
    # clusters agree with the truth up to renaming
    ct = np.array([[np.sum((lab == a) & (truth_m == b)) for b in (0, 1)] for a in sorted(set(lab.tolist()))])
    assert ct.max(axis=1).sum() / len(lab) > 0.95 and len(set(lab.tolist())) == 2
    desc = cluster.describe(m, lab, tab)
    assert sorted(round(v["f0_median"]) for v in desc) == [115, 210]


def test_viterbi_removes_single_flips_but_keeps_real_turns():
    sim = np.zeros((20, 2))
    sim[:10, 0], sim[:10, 1] = 0.8, 0.2
    sim[10:, 0], sim[10:, 1] = 0.2, 0.8
    sim[4] = [0.35, 0.45]                                    # one confusing moment inside voice 0
    path = cluster.viterbi_smooth(sim, np.zeros(19, bool), penalty=0.15)
    assert path.tolist() == [0] * 10 + [1] * 10
    free = cluster.viterbi_smooth(sim, np.ones(19, bool))     # breaks everywhere: no smoothing
    assert free[4] == 1


def test_relabel_turns_and_confidence():
    lab = np.array([2, 2, 2, 0, 0, 1, 2, 2, 0, 0, 0])
    new, sizes = cluster.relabel_by_size(lab, min_size=2)
    assert sizes == [5, 5] and new[5] == -1 and set(new[[0, 1, 2, 6, 7]].tolist()) == {1} and set(new[[3, 4, 8, 9, 10]].tolist()) == {0}
    m = SimpleNamespace(t0=np.arange(5) * 0.75, t1=np.arange(5) * 0.75 + 0.75, f0=np.array([200, 200, 0, 110, 110.0]),
                        loud_cam=["01"] * 5, cams=[["01", "02"]] * 5)
    turns = cluster.to_turns(m, np.array([0, 0, 0, 1, 1]), np.array([0.2, 0.2, 0.2, 0.1, 0.1]))
    assert [(t["voice"], t["n_moments"]) for t in turns] == [(0, 3), (1, 2)] and turns[0]["f0_median"] == 200.0
    assert cluster.margin_to_confidence(0.2, 5) > cluster.margin_to_confidence(0.0, 5) > 0
    assert cluster.margin_to_confidence(0.2, 1) < cluster.margin_to_confidence(0.2, 5)


# ------------------------------------------------------------------------------------------------ master-clock events


def test_event_needs_support_from_several_phones():
    n = 40
    cov = {c: np.ones(n, bool) for c in ("01", "02", "05")}
    grids = {c: np.zeros(n) for c in cov}
    grids["01"][10:16] = 0.8                                      # only phone 01 hears it: handling noise
    assert events.detect(grids, cov) == []
    grids["02"][11:15] = 0.5                                      # a second phone tags it too
    ev = events.detect(grids, cov)
    assert len(ev) == 1 and ev[0]["support_max"] >= 2 and ev[0]["peak"] == pytest.approx(0.8)
    short = {c: np.zeros(n) for c in cov}
    short["01"][5] = short["02"][5] = 0.9
    assert events.detect(short, cov) == []                        # 0.5 s < min_s


def test_spread_windows_peaks_at_the_centre():
    g = events.spread_windows([(10.0, 13.0, 0.9)], 10.0, 8)
    assert g.argmax() in (2, 3) and g[2] > g[0]


# ------------------------------------------------------------------------------------------------ names, mentions, naming, jokes


@pytest.mark.parametrize("before,name,after,kind", [
    ("Hello I'm", "Kadu", "and this is", "self_strong"),
    ("my name is", "Larissa", "", "self_strong"),
    ("Hi this is", "Larissa", "and yes", "self_weak"),
    ("Thank you", "Kadu.", "", "vocative"),
    ("", "Kadu,", "can you show us", "vocative"),
    ("Okay. Larissa", "", "", None),
    ("We did it with", "Larissa", "yesterday", "third"),
    ("", "Kadu", "is going to lead", "third"),
])
def test_classify_context(before, name, after, kind):
    if kind is None:
        return
    assert identity.classify_context(before.split(), after.split(), name) == kind


def test_name_of():
    assert identity.name_of("Larisa,") == "larissa" and identity.name_of("Clarissa") == "larissa"
    assert identity.name_of("Cadu") == "kadu" and identity.name_of("Kadu!") == "kadu"
    assert identity.name_of("Alyssa") == "student:Alyssa"           # a student's name is not Larissa
    assert identity.name_of("Alissa") == "student:Alyssa"
    assert identity.name_of("lady") is None and identity.name_of("cad") is None


def _w(cam, eng, text, m0, prob=None):
    return texts.Word(cam, eng, text, m0, m0 + 0.3, m0, m0 + 0.3, prob)


def test_find_mentions_merges_streams():
    words = []
    for cam, eng in (("01", "parakeet-tdt-0.6b-v3"), ("02", "whisper-large-v3:en")):
        words += [_w(cam, eng, "Okay.", 9.0), _w(cam, eng, "Kadu,", 10.0 + (0.2 if cam == "02" else 0)), _w(cam, eng, "show", 10.5), _w(cam, eng, "us", 10.8)]
    ms = identity.find_mentions(words)
    assert len(ms) == 1 and ms[0].name == "kadu" and ms[0].kind == "vocative" and ms[0].support == 2


def _turns(spec):
    return [{"start": s, "end": e, "voice": v, "confidence": 0.9} for s, e, v in spec]


def _mention(i, t, name, kind, speaker, support=2, conf=0.9):
    return identity.Mention(f"m{i}", name, kind, t, t, t + 0.3, support, ["x"], ["01"], f"... [{name}] ...", speaker=speaker, speaker_conf=conf)


def test_addressing_names_the_voices_and_a_self_intro_does_not():
    # voice 0 says "Kadu, ..." three times and answers nothing; voice 1 says "Larissa" twice. Voice 0 also INTRODUCES itself as Kadu (a joke).
    ms = [_mention(0, 10, "kadu", "vocative", 0), _mention(1, 30, "kadu", "vocative", 0), _mention(2, 50, "larissa", "vocative", 1),
          _mention(3, 70, "kadu", "third", 0), _mention(4, 90, "larissa", "third", 1), _mention(5, 5, "kadu", "self_strong", 0)]
    turns = _turns([(0, 8, 0), (8.5, 15, 1), (28, 29, 0), (29.5, 40, 1), (48, 49, 1), (49.5, 55, 0)])
    ev = identity.addressing_evidence(ms, turns, 0, 1)
    assert ev["llr"] > 2 and all(e["id"] != "m5" for e in ev["events"])            # voice 0 is Larissa; the self-introduction was ignored
    voices = [{"voice": 0, "seconds": 120.0, "f0_median": 215.0}, {"voice": 1, "seconds": 110.0, "f0_median": 118.0}, {"voice": 2, "seconds": 20.0, "f0_median": 160.0}]
    names = identity.name_voices(voices, ms, turns)
    assert names["assignment"] == {"0": "Larissa", "1": "Kadu", "2": "other_1"} and names["agreement"] == "addressing and pitch agree"
    # the pitch prior alone must be overturned by strong, opposite addressing evidence
    voices_inv = [{"voice": 0, "seconds": 120.0, "f0_median": 110.0}, {"voice": 1, "seconds": 110.0, "f0_median": 205.0}]
    names2 = identity.name_voices(voices_inv, ms, turns)
    assert names2["assignment"]["0"] == "Larissa" and names2["agreement"] == "addressing and pitch DISAGREE"


def test_pitch_prior():
    assert identity.pitch_prior(215, 118)[0] > 0 > identity.pitch_prior(118, 215)[0]
    assert identity.pitch_prior(150, 140)[0] == 0 and identity.pitch_prior(None, 140)[0] == 0


def test_joke_candidates_need_a_contradiction_and_attach_the_laugh():
    names = {"assignment": {"0": "Larissa", "1": "Kadu"}, "naming_confidence": 0.9}
    ms = [_mention(0, 20, "kadu", "self_strong", 0), _mention(1, 40, "larissa", "self_strong", 0)]
    utts = [{"start": 100.0, "end": 103.0, "voice": 1, "text": "Look at my red hair"}, {"start": 150.0, "end": 152.0, "voice": 0, "text": "my red hair is long"}]
    laugh = [{"id": "ev001", "kind": "laughter", "t0": 22.0, "t1": 24.0, "peak": 0.7}, {"id": "ev002", "kind": "laughter", "t0": 104.0, "t1": 106.0, "peak": 0.6}]
    js = identity.find_jokes(ms, utts, names, laugh)
    by = {(j["kind"], round(j["t"])): j for j in js}
    intro = by[("self_introduction", 20)]
    assert intro["contradiction"] and intro["emoji"] == identity.LAUGH_EMOJI == "\U0001F602" and intro["t_token"] == 22.0
    honest = by[("self_introduction", 40)]
    assert not honest["contradiction"] and honest["emoji"] is None                       # Larissa saying "Larissa" is not a joke
    hair = by[("appearance", 100)]
    assert hair["contradiction"] and hair["voice_name"] == "Kadu" and hair["t_token"] == 104.0 and hair["emoji"]
    assert not by[("appearance", 150)]["contradiction"]                                  # Larissa's own red hair is just a remark


def test_voice_at_uses_overlap_and_margin():
    t0 = np.arange(6) * 0.75
    t1 = t0 + 0.75
    path = np.array([0, 0, 0, 1, 1, 1])
    marg = np.full(6, 0.2)
    assert identity.voice_at(0.0, 2.0, t0, t1, path, marg)[0] == 0
    v, c = identity.voice_at(2.3, 4.4, t0, t1, path, marg)
    assert v == 1 and c > 0.5
    assert identity.voice_at(10.0, 11.0, t0, t1, path, marg) == (None, 0.0)


# ------------------------------------------------------------------------------------------------ texts, job spec, repo rules


def test_norm_token():
    assert texts.norm_token("Corredor,") == "corredor" and texts.norm_token("it's") == "its" and texts.norm_token("Bônus!") == "bonus"


def test_job_spec_shape_and_hygiene(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "audio_files", lambda: [tmp_path / f"{i:02d}-Recap-X.16k.wav" for i in range(7)])
    monkeypatch.setattr(jobs, "code_archive", lambda out=None: tmp_path / "speech_code.tar")
    spec = jobs.job_spec("t1")
    names = [s["name"] for s in spec["steps"]]
    assert names[:4] == ["env", "check", "vad", "whisper_large-v3"] and {"parakeet", "canary", "embed", "events", "sortformer", "quality"} <= set(names)
    assert spec["kind"] == "asr" and spec["needs_gpu"] and spec["outputs"][0]["path"].endswith("speech/raw/")
    assert all(i["path"].startswith("src/larissa_kadu_recap/speech_audio/") or i["path"] == "speech_code.tar" for i in spec["inputs"])
    assert all(s.get("save") for s in spec["steps"] if s["name"] != "env")
    assert "env" in names and not [s for s in spec["steps"] if s["name"] in ("vad", "whisper_large-v3") and s.get("optional")]


def test_no_absolute_user_paths_or_secrets_in_the_package():
    bad = []
    for f in list(PKG.rglob("*.py")) + list(PKG.rglob("*.sh")):
        txt = f.read_text(encoding="utf-8")
        if re.search(r"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z]", txt) or re.search(r"/home/[a-z]+/", txt):
            bad.append(f.name)
        if re.search(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{20,})", txt):
            bad.append(f.name + " (secret)")
    assert not bad, bad


# ------------------------------------------------------------------------------------------------ route B, report, review, segments


from dancecap.speech import diarize, report, review  # noqa: E402
from dancecap.speech.pod import stage_embed as _se  # noqa: E402


def test_core_segments_drop_window_duplicates_and_split_long_ones():
    doc = {"windows": [{"t0": 0.0, "t1": 60.0, "segments": [{"start": 0.0, "end": 8.0, "spk": 0}, {"start": 52.0, "end": 57.0, "spk": 1}]},
                       {"t0": 50.0, "t1": 77.0, "segments": [{"start": 50.0, "end": 53.0, "spk": 1}, {"start": 60.0, "end": 61.0, "spk": 0}]}]}
    segs = _se.core_segments(doc)
    # window overlap 10 s: window 0 owns [0, 55), window 1 owns [55, end); 52-57 (middle 54.5) stays with window 0, 50-53 (51.5) is window 1's
    # duplicate of it and is dropped, 60-61 (a 1 s segment after the overlap) belongs to window 1
    assert [(s["t0"], s["win"]) for s in segs if s["t0"] >= 50] == [(52.0, 0), (54.5, 0), (60.0, 1)]       # the 5 s segment is cut into two pieces of 2.5 s
    long = [s for s in segs if s["win"] == 0 and s["t1"] <= 8.0 + 1e-6 and s["t0"] < 8.0]
    assert len(long) == 3 and all(s["t1"] - s["t0"] <= 3.0 + 1e-6 for s in long)          # 8 s -> 3 pieces of 2.67 s
    assert _se.widen(2.16, 2.88, 360.0) == pytest.approx((2.12, 2.92))


def test_merge_segments_needs_distinct_phones_and_interjection_needs_support():
    segs = [{"cam": "00", "label": 0, "m0": 2.1, "m1": 3.0, "margin": 0.3}, {"cam": "01", "label": 0, "m0": 2.16, "m1": 2.9, "margin": 0.2},
            {"cam": "02", "label": 0, "m0": 2.2, "m1": 2.7, "margin": 0.1}, {"cam": "01", "label": 1, "m0": 0.0, "m1": 2.2, "margin": 0.5}]
    ev = diarize.merge_segments(segs)
    male = [e for e in ev if e["label"] == 0][0]
    assert male["support"] == 3 and male["cams"] == ["00", "01", "02"]
    mom_t0, mom_t1, path = np.array([0.0, 0.75, 1.5, 2.25, 3.0]), np.array([0.75, 1.5, 2.25, 3.0, 3.75]), np.array([1, 1, 1, 1, 1])
    clocks = {"00": sources.Clock("00", 1.0, 0.0, "x"), "01": sources.Clock("01", 1.0, 0.0, "x"), "02": sources.Clock("02", 1.0, 0.0, "x")}
    dur = {"00": 100.0, "01": 100.0, "02": 100.0}
    inter = diarize.interjections(ev, mom_t0, mom_t1, path, clocks, dur)
    assert len(inter) == 1 and inter[0]["voice"] == 0 and inter[0]["base_voice"] == 1 and inter[0]["support"] == 3
    solo = [{"cam": "00", "label": 0, "m0": 2.1, "m1": 3.0, "margin": 0.3}]
    assert diarize.interjections(diarize.merge_segments(solo), mom_t0, mom_t1, path, clocks, dur) == []          # one phone of three: rejected


def test_report_similarity_and_windows():
    assert report.similarity("a b c d".split(), "a b c d".split())[0] == 1.0
    assert report.similarity("a b c d".split(), "a x c d".split())[0] == 0.75
    assert report.similarity([], [])[0] == 1.0
    mk = lambda eng, toks: [texts.Word("01", eng, t, i * 0.5, i * 0.5 + 0.3, i * 0.5, i * 0.5 + 0.3, None) for i, t in enumerate(toks)]
    a = mk("a", [f"w{i}" for i in range(40)])
    b = mk("b", [f"w{i}" if i % 4 else "zzz" for i in range(40)])
    w = report.window_agreement(a, b, 0.0, 20.0, win=20.0)
    assert len(w) == 1 and w[0]["agreement"] == pytest.approx(0.75)


def test_review_selection_helpers():
    items = [{"start": 10.0, "s": 5}, {"start": 20.0, "s": 9}, {"start": 70.0, "s": 7}, {"start": 71.0, "s": 8}]
    got = review.pick_spaced(items, lambda x: x["s"], 3, 45.0)
    assert [g["start"] for g in got] == [20.0, 71.0]                       # 10 is too close to 20, 70 too close to 71
    assert review.clip_window(10.0, 12.0) == (9.6, 12.4)
    a, b = review.clip_window(10.0, 40.0, centre=30.0)
    assert b - a == pytest.approx(review.MAX_CLIP) and a <= 30.0 <= b


def test_unplaced_source_gets_voice_turns_in_its_own_time(tmp_path):
    rng = np.random.default_rng(3)
    d = 16
    a, b = cluster.l2n(rng.normal(size=d)), cluster.l2n(rng.normal(size=d))
    cent = np.stack([a, b])
    cam = tmp_path / "04"
    cam.mkdir()
    t0 = np.arange(0, 20, 0.75)
    emb = np.stack([(a if (t // 6) % 2 == 0 else b) + rng.normal(0, 0.05, d) for t in t0])
    np.savez(cam / "embed.npz", t0=t0, t1=t0 + 1.5, ecapa=emb)
    out = diarize.unplaced_turns(tmp_path, "04", cent, ["ecapa"])
    assert out["windows"] == len(t0) and len(out["turns"]) >= 3
    assert {t["voice"] for t in out["turns"]} == {0, 1} and out["clock"].startswith("the phone's own seconds")


def test_regions_words_move_to_the_master_clock(tmp_path):
    reg = tmp_path / "raw_regions" / "r01"
    reg.mkdir(parents=True)
    (reg / "00.json").write_text(json.dumps({"hypotheses": [{"model": "canary-1b-v2", "variant": "en", "words": [
        {"word": "hair", "start": 15.4, "end": 15.9, "prob": None}]}]}))
    ws = texts.load_regions(tmp_path / "raw_regions", {"00": sources.Clock("00", 1.0, -8.9, "x")})
    assert len(ws) == 1 and ws[0].m0 == pytest.approx(6.5) and ws[0].engine == "region:canary-1b-v2:en"


def test_alt_texts_and_unresolved_introductions():
    mk = lambda eng, cam, toks, t0: [texts.Word(cam, eng, t, t0 + i * 0.3, t0 + i * 0.3 + 0.2, t0 + i * 0.3, t0 + i * 0.3 + 0.2, None) for i, t in enumerate(toks)]
    words = mk("parakeet-tdt-0.6b-v3", "01", "this is a little stuck".split(), 3.6) + mk("whisper-large-v3:en", "01", "this is Larissa".split(), 3.6)         + mk("canary-1b-v2", "01", "this is Lila Marisa".split(), 3.6)
    utts = [{"start": 3.5, "end": 5.0, "voice": 1, "name": "Larissa"}]
    out = diarize.unresolved_introductions(utts, words, [{"id": "ev000", "kind": "laughter", "t0": 2.0, "t1": 4.0}])
    assert len(out) == 1 and out[0]["n_readings"] == 3 and not out[0]["kadu_heard_by_any_engine"] and out[0]["laughter_nearby"] == ["ev000"]
