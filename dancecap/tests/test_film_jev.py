"""film_jev (Jev judgments for the recap film) with a fake client: requests are well formed, answers are mapped, segments
aggregate their utterances, and the overlay gate flags at the threshold. No network."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the workspace root too
from dancecap import film_jev as J  # noqa: E402


class FakeClient:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def system_one(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(answers=self.answers, usage=SimpleNamespace(input_tokens=900), model="jev-1.13.0")


def test_utterance_request_names_the_target_and_maps_the_answers():
    us = [{"id": f"u{i:03d}", "speaker": "Kadu", "text": f"line {i}", "start": i, "end": i + 1} for i in range(8)]
    c = FakeClient({"need_3d": SimpleNamespace(score=2.6, confidence=0.7, probabilities={0: 0.0, 1: 0.1, 2: 0.2, 3: 0.7}),
                    "overhead": SimpleNamespace(noul=0.12)})
    r = J.judge_utterance(us, 4, client=c)
    kw = c.calls[0]
    assert kw["state"]["target"] == "u004" and set(kw["state"]["utterances"]) == {"u002", "u003", "u004", "u005", "u006"}
    assert "`utterances.u004.text`" in kw["questions"]["need_3d"].instructions and len(kw["questions"]["need_3d"].criteria) == 4
    assert r["need_3d"] == 2.6 and r["need_3d_p"][3] == 0.7 and r["overhead"] == 0.12 and r["tokens"] == 900


def test_segments_take_their_say_ids():
    by_id = {"u001": {"need_3d": 0.1, "overhead": 0.0, "text": "hi"},
             "u009": {"need_3d": 2.9, "overhead": 0.1, "text": "the counterbalance is the leader placing the weight back"}}
    d = {"segments": [{"id": "a", "film": [0, 5], "say": "Larissa: u001; Kadu: u009"}, {"id": "b", "film": [5, 7], "say": ""}]}
    a, b = J.per_segment(d, by_id)
    assert a["need_3d_max"] == 2.9 and 2.0 < a["need_3d_mean"] < 2.9 and a["utterances"] == ["u001", "u009"]
    assert b["need_3d_max"] is None


def test_overlay_gate_flags_at_the_threshold():
    assert J.flagged({"legend_over_dancer": 0.46, "overlay_over_head": 0.1}, 0.35)
    assert not J.flagged({"legend_over_dancer": 0.07, "overlay_over_head": 0.14}, 0.35)
