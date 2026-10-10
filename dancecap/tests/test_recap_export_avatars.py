"""recap_export --avatars: which validation failures are expected by design (pure part: classify)."""
from __future__ import annotations

from dancecap import recap_export as re_


def _checks():
    return [{"name": "capture.json", "ok": True, "detail": ""},
            {"name": "lead albedo", "ok": False, "detail": "no smplx_albedo entry (untextured avatar)"},
            {"name": "hair groom", "ok": False, "detail": "x"},
            {"name": "audio", "ok": False, "detail": "no audio (silent clock)"},
            {"name": "lead skin", "ok": False, "detail": "broken"}]


def test_a_plain_capture_expects_no_avatar_layers():
    c = re_.classify(_checks())
    assert {e["name"] for e in c["expected"]} == {"lead albedo", "hair groom", "audio"}
    assert [f["name"] for f in c["failed"]] == ["lead skin"]


def test_an_avatar_capture_only_expects_the_missing_audio():
    c = re_.classify(_checks(), re_.EXPECTED_AVATARS)
    assert [e["name"] for e in c["expected"]] == ["audio"]
    assert {f["name"] for f in c["failed"]} == {"lead albedo", "hair groom", "lead skin"}


def test_a_capture_with_the_class_audio_expects_nothing_to_fail():
    """T3: --audio puts the class mix into the capture; then a missing audio is a real failure like any other."""
    c = re_.classify(_checks(), re_.EXPECTED_FULL)
    assert c["expected"] == [] and {f["name"] for f in c["failed"]} == {"lead albedo", "hair groom", "audio", "lead skin"}


def test_class_audio_is_the_film_mix_when_it_exists(tmp_path):
    class T:
        work = tmp_path
    assert re_.class_audio(T) is None
    (tmp_path / "film" / "audio").mkdir(parents=True)
    (tmp_path / "film" / "audio" / "class_mix.wav").write_bytes(b"x")
    assert re_.class_audio(T) == tmp_path / "film" / "audio" / "class_mix.wav"


def test_class_audio_follows_the_soundtrack_switch(tmp_path):
    """V2: `recap_export --audio` without a path reads the same mix the film soundtrack uses (film/audio/soundtrack.json), not always class_mix.wav."""
    import json

    class T:
        work = tmp_path
    d = tmp_path / "film" / "audio"
    d.mkdir(parents=True)
    for n in ("class_mix.wav", "class_mix_v2v.wav", "class_mix_v2v_full.wav"):
        (d / n).write_bytes(b"x")
    assert re_.class_audio(T) == d / "class_mix_v2v_full.wav"
    (d / "soundtrack.json").write_text(json.dumps({"class_mix": "v2v"}))
    assert re_.class_audio(T) == d / "class_mix_v2v.wav"
    (d / "soundtrack.json").write_text(json.dumps({"class_mix": "original"}))
    assert re_.class_audio(T) == d / "class_mix.wav"
