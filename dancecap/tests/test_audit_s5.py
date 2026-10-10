"""Tests of the S5 audit helpers (dancecap.speech.audit, audit_struct, audit_report): pure functions on synthetic data; the sampler test needs the speech work files
and is skipped when they are absent (git-ignored data)."""
from __future__ import annotations

import numpy as np
import pytest

from dancecap.speech import SPEECH, audit, audit_report, audit_struct


class _Clock:
    def __init__(self, offset: float = 0.0, slope: float = 1.0):
        self.offset, self.slope = offset, slope

    def to_source(self, m: float) -> float:
        return (m - self.offset) / self.slope


def test_third_and_group():
    assert audit.third_of(0.0) == "start" and audit.third_of(119.9) == "start"
    assert audit.third_of(120.0) == "middle" and audit.third_of(239.9) == "middle"
    assert audit.third_of(240.0) == "end"
    u = {"speaker": "Kadu"}
    assert audit.group_of(u, {"interjection": True}) == "other"
    assert audit.group_of(u, {"interjection": False}) == "Kadu"
    assert audit.group_of(u, None) == "Kadu"


def test_span_snr_uses_only_phones_recording_the_whole_span():
    n = 20 * 60                                    # 60 s of 50 ms frames
    quiet = np.full(n, -50.0, dtype=np.float32)
    loud = quiet.copy()
    loud[20 * 10:20 * 20] = -20.0                  # speech between 10 and 20 s of the phone's own clock
    lev = {"01": loud, "02": quiet.copy()}
    clocks = {"01": _Clock(0.0), "02": _Clock(30.0)}   # phone 02 starts at master 30 s
    snr = audit.span_snr(lev, clocks, 11.0, 18.0)
    assert set(snr) == {"01"}                       # phone 02 had not started
    assert snr["01"] > 25.0
    snr2 = audit.span_snr(lev, clocks, 31.0, 38.0)
    assert set(snr2) == {"01", "02"} and snr2["01"] < 1.0


def test_ctc_targets_and_alignment_on_synthetic_emissions():
    pytest.importorskip("torchaudio")
    labels = ["-", "|", "H", "I", "T", "E"]
    toks, spans = audit_report._targets(["hi", "the"], labels)
    assert [labels[t] for t in toks] == ["H", "I", "|", "T", "H", "E"]
    assert spans == [(0, 1), (3, 5)]
    # 30 frames: H at 5, I at 8, | at 12, T at 15, H at 18, E at 22, blank elsewhere
    T = 30
    lp = np.full((T, len(labels)), -9.0, dtype=np.float32)
    lp[:, 0] = -0.01
    for frame, ch in [(5, "H"), (8, "I"), (12, "|"), (15, "T"), (18, "H"), (22, "E")]:
        lp[frame, :] = -9.0
        lp[frame, labels.index(ch)] = -0.01
    out = audit_report.ctc_align(lp, ["hi", "the"], labels)
    assert out is not None and len(out) == 2
    (a0, a1, _), (b0, b1, _) = out
    assert a0 == 5 and a1 == 9           # 'hi' spans the frames of H..I (end exclusive)
    assert b0 == 15 and b1 == 23


def test_parse_srt_and_norm():
    text = "1\n00:00:01,240 --> 00:00:01,720\nLarissa: Hello!\n\n2\n00:01:02,000 --> 00:01:03,500\nKadu: Okay, so\nthe first thing\n"
    cues = audit_struct.parse_srt(text)
    assert len(cues) == 2
    assert cues[0]["start"] == pytest.approx(1.24) and cues[0]["end"] == pytest.approx(1.72)
    assert cues[1]["start"] == pytest.approx(62.0) and cues[1]["text"].count("\n") == 1
    assert audit_struct.norm("Hello!\nWorld, it's") == "hello world it's"


@pytest.mark.skipif(not (SPEECH / "transcript_final.json").exists() or not (SPEECH / "audio").exists(), reason="needs the git-ignored speech work files")
def test_sample_is_seeded_stratified_and_skips_the_opening():
    a = audit.build_sample(25, seed=5, write=False)
    b = audit.build_sample(25, seed=5, write=False)
    ids = [r["id"] for r in a["utterances"]]
    assert ids == [r["id"] for r in b["utterances"]]
    assert len(ids) == len(set(ids)) == 25
    assert all(r["start"] >= audit.OPENING_END for r in a["utterances"])
    assert set(r["group"] for r in a["utterances"]) <= {"Larissa", "Kadu", "other"}
    assert {r["third"] for r in a["utterances"]} == {"start", "middle", "end"}
    c = audit.build_sample(25, seed=6, write=False)
    assert [r["id"] for r in c["utterances"]] != ids
