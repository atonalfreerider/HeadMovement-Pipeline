"""dancecap.film_pure: the pure-dance film (one 1x segment, film 0 = audio 0, the capture's song as the mix, chapter marks, no narration or
call-outs) and the demo preset that points the library at it.

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_film_pure.py -q
"""
from __future__ import annotations

import json
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import film_library as fl  # noqa: E402
from dancecap import film_pure as fp  # noqa: E402

TTA = 3.1312644
CHAPTERS = [{"title": "Opening", "segment": "dance", "offset_s": 0.0}, {"title": "Finale", "segment": "dance", "offset_s": 160.731},
            {"title": "The exit step", "segment": "dance", "offset_s": 28.131}]


def _direction(**kw):
    args = dict(name="pure", title="Pure", capture="99_Capture", take="t", dance_range=[-3.11549, 184.98451], time_to_audio=TTA,
                film_end=188.116, chapters=CHAPTERS)
    args.update(kw)
    return fp.build_direction(**args)


def test_one_segment_at_1x_covers_the_whole_dance_with_film_0_at_audio_0():
    d = _direction()
    assert len(d["segments"]) == 1
    s = d["segments"][0]
    assert s["id"] == "dance" and s["speed"] == 1.0 and s["audio"] == "locked"
    assert s["film"] == [0.0, 188.116]
    assert s["dance"][0] == pytest.approx(-TTA) and s["dance"][1] == pytest.approx(188.116 - TTA)
    assert d["timeline"]["film_duration_s"] == 188.116 and d["timeline"]["narration_lines"] == 0


def test_no_narration_annotations_cards_flow_or_skeleton_mode():
    d = _direction()
    s = d["segments"][0]
    assert s["annotations"] == [] and "narration" not in s and "title" not in s and "hold_s" not in s and "replay" not in s
    assert "narration" not in d and "flow" not in d
    lay = s["layers"]
    assert "skeleton_mode" not in lay and "physics" not in lay                 # the viewer's own toggle (VR wrist) decides physics / rhythm
    assert lay["floor"] is True                                                # the footprints of the physics view need the floor
    for k in ("moves", "graph_inset", "counterbalance", "traces", "neck", "hud", "beat_counter", "rhythm_score"):
        assert lay[k] is False, k
    assert lay["floorcraft"] == "off"


def test_chapters_are_film_seconds_sorted_and_the_library_reads_them():
    d = _direction()
    assert [c["title"] for c in d["chapters"]] == ["Opening", "The exit step", "Finale"]
    assert [c["film_s"] for c in d["chapters"]] == [0.0, 28.131, 160.731] and all(c["segment"] == "dance" for c in d["chapters"])
    via_direction = fl.derive_chapters(d)                                      # the direction's own chapters
    via_plan = fl.derive_chapters(d, CHAPTERS)                                 # the preset's plan (segment + offset_s)
    assert via_direction == via_plan
    assert via_plan[-1] == {"title": "Finale", "startS": 160.731, "endS": 188.116}


def test_bad_chapters_say_what_to_change():
    with pytest.raises(fp.PureError, match="one segment"):
        _direction(chapters=[{"title": "X", "segment": "open_title"}])
    with pytest.raises(fp.PureError, match="outside the film"):
        _direction(chapters=[{"title": "Late", "startS": 190.0}])


def _capture(root: Path, seconds: float = 190.0) -> Path:
    cap = root / "99_Capture"
    cap.mkdir(parents=True)
    (cap / "capture.json").write_text(json.dumps({"audio": "audio.wav", "times": "times.json", "time_to_audio": TTA}), encoding="utf-8")
    (cap / "times.json").write_text(json.dumps([-3.11549, 0.0, 184.98451]), encoding="utf-8")
    with wave.open(str(cap / "audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(1000)
        w.writeframes(struct.pack("<h", 0) * int(seconds * 1000))
    return cap


def test_build_writes_the_direction_and_the_mix_beside_it(tmp_path, monkeypatch):
    cap = _capture(tmp_path / "streaming")
    monkeypatch.setattr(fp, "STREAMING", tmp_path / "streaming")
    preset = {"id": "x", "title": "X", "capture": "99_Capture", "direction": str(tmp_path / "film" / "direction_dance.json"),
              "pure": {"name": "x_dance", "take": "t"}, "chapters": CHAPTERS}
    path, d, _notes = fp.build_from_preset(preset)
    assert path.exists() and json.loads(path.read_text(encoding="utf-8"))["name"] == "x_dance"
    mix = path.parent / "audio" / "x_dance_mix.wav"
    assert mix.exists() and mix.stat().st_size == (cap / "audio.wav").stat().st_size
    assert d["timeline"]["film_duration_s"] == 188.116
    assert not (path.parent / "narration").exists()


def test_a_film_past_its_soundtrack_is_refused(tmp_path, monkeypatch):
    _capture(tmp_path / "streaming", seconds=100.0)
    monkeypatch.setattr(fp, "STREAMING", tmp_path / "streaming")
    preset = {"id": "x", "capture": "99_Capture", "direction": str(tmp_path / "d.json"), "pure": {"name": "x_dance"}}
    with pytest.raises(fp.PureError, match="past the soundtrack"):
        fp.build_from_preset(preset, dry_run=True)


def test_the_demo_preset_is_the_pure_dance_film():
    demo = fl.read_presets()["demo"]
    assert demo["direction"].endswith("/direction_dance.json")
    assert demo["pure"]["name"] == "kadu_larissa_dance" and demo.get("slowmo_audio", "pitch") == "pitch"
    assert [c["title"] for c in demo["chapters"]] == ["Opening", "The exit step", "Floor craft", "The counterbalance", "A path through the moves",
                                                    "Head movement", "The last counterbalance", "Finale"]
    assert [c["offset_s"] for c in demo["chapters"]] == [0.0, 28.131, 35.431, 59.631, 71.031, 121.531, 149.131, 160.731]
    path = fl._abs(demo["direction"])
    if not path.exists():
        pytest.skip("the demo film's work folder is not on this machine")
    d = fl.read_direction(path)
    assert d["name"] == "kadu_larissa_dance" and len(d["segments"]) == 1 and fl.film_duration(d) == pytest.approx(188.116)
    assert (path.parent / "audio" / "kadu_larissa_dance_mix.wav").exists()
