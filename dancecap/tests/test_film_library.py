"""dancecap.film_library: the viewer's library.json (cards, thumbnails, chapters) from a film direction.

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_film_library.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import film_library as fl  # noqa: E402


def _direction(tmp_path: Path, segs, **extra) -> Path:
    """a minimal direction.json: segments [(id, f0, f1)] (+ optional per-segment extras via a dict in the tuple's 4th place)"""
    segments = []
    for s in segs:
        seg = {"id": s[0], "film": [s[1], s[2]]}
        if len(s) > 3:
            seg.update(s[3])
        segments.append(seg)
    d = {"name": "film", "capture": "99_Capture", "segments": segments, "timeline": {"film_duration_s": segs[-1][2]}}
    d.update(extra)
    p = tmp_path / "work" / "film" / "direction.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


SEGS = [("open_title", 0, 11), ("cams", 11, 20), ("lp1_a", 20, 30), ("lp1_b", 30, 44), ("floor", 44, 70), ("end", 70, 74)]


def test_a_plan_anchors_chapters_at_segments_and_they_are_contiguous(tmp_path):
    p = _direction(tmp_path, SEGS)
    plan = [{"title": "Intro", "segment": "open_title"}, {"title": "  The   exit ", "segment": "lp1_a"}, {"title": "Floor", "segment": "floor"}]
    ch = fl.derive_chapters(fl.read_direction(p), plan)
    assert ch == [{"title": "Intro", "startS": 0.0, "endS": 20.0},
                  {"title": "The exit", "startS": 20.0, "endS": 44.0},
                  {"title": "Floor", "startS": 44.0, "endS": 74.0}]


def test_a_plan_naming_a_missing_segment_is_an_error_that_says_which(tmp_path):
    p = _direction(tmp_path, SEGS)
    with pytest.raises(fl.LibraryError, match="nope"):
        fl.derive_chapters(fl.read_direction(p), [{"title": "X", "segment": "nope"}])


def test_a_plan_may_use_times_and_is_sorted(tmp_path):
    p = _direction(tmp_path, SEGS)
    ch = fl.derive_chapters(fl.read_direction(p), [{"title": "B", "startS": 30.5}, {"title": "A", "startS": 0}])
    assert [c["title"] for c in ch] == ["A", "B"] and ch[0]["endS"] == 30.5 and ch[1]["endS"] == 74.0


def test_a_chapter_outside_the_film_is_refused(tmp_path):
    p = _direction(tmp_path, SEGS)
    with pytest.raises(fl.LibraryError, match="outside the film"):
        fl.derive_chapters(fl.read_direction(p), [{"title": "Late", "startS": 80}])


def test_chapters_written_by_the_direction_win_over_the_automatic_split(tmp_path):
    segs = [("a_1", 0, 10), ("a_2", 10, 20, {"chapter": "Second part"}), ("b_1", 20, 40, {"chapter": "Third"})]
    p = _direction(tmp_path, segs)
    ch = fl.derive_chapters(fl.read_direction(p))
    assert [(c["title"], c["startS"]) for c in ch] == [("Second part", 10.0), ("Third", 20.0)]
    p2 = _direction(tmp_path / "t2", SEGS, chapters=[{"title": "All", "film_s": 0}, {"title": "Half", "segment": "floor"}])
    assert [(c["title"], c["startS"]) for c in fl.derive_chapters(fl.read_direction(p2))] == [("All", 0.0), ("Half", 44.0)]


def test_automatic_split_by_id_prefix_merges_short_runs(tmp_path):
    p = _direction(tmp_path, SEGS)
    ch = fl.derive_chapters(fl.read_direction(p))
    # open (11 s), cams (9 s), lp1 (24 s), floor (26 s), end (4 s: joins floor)
    assert [(c["title"], c["startS"], c["endS"]) for c in ch] == [
        ("Open", 0.0, 11.0), ("Cams", 11.0, 20.0), ("Lp1", 20.0, 44.0), ("Floor", 44.0, 74.0)]


def test_the_entry_has_the_documented_fields_and_pending_films_are_marked(tmp_path):
    lib = tmp_path / "library"
    p = _direction(tmp_path, SEGS)
    thumb = tmp_path / "poster.JPG"
    thumb.write_bytes(b"\xff\xd8not really a jpeg")
    spec = {"id": "demo", "title": "Title × é", "subtitle": "Sub", "direction": str(p), "thumbnail": str(thumb),
            "chapters": [{"title": "Intro", "segment": "open_title"}]}
    entry, notes = fl.build_entry(spec, library_dir=lib)
    assert set(entry) == {"id", "title", "subtitle", "capture", "thumbnail", "direction", "durationS", "chapters", "available", "dialogue", "slowmoAudio"}
    assert entry["dialogue"] is False and entry["slowmoAudio"] == "pitch"            # the defaults: music slows with the picture
    assert entry["capture"] == "99_Capture" and entry["durationS"] == 74.0 and entry["available"] is True
    assert entry["thumbnail"] == "thumbs/demo.jpg" and (lib / "thumbs" / "demo.jpg").read_bytes() == thumb.read_bytes()
    assert entry["chapters"][0] == {"title": "Intro", "startS": 0.0, "endS": 74.0}
    assert any("chapters" in n for n in notes)  # one chapter: outside 8-14

    pending, pnotes = fl.build_entry({"id": "recap", "title": "R", "capture": "06_X", "direction": str(tmp_path / "work" / "later" / "direction.json")},
                                     library_dir=lib)
    assert pending["durationS"] is None and pending["chapters"] == [] and pending["capture"] == "06_X" and pending["thumbnail"] is None
    assert any("not built yet" in n for n in pnotes)

    held, _ = fl.build_entry({**spec, "id": "held", "hold": True}, library_dir=lib)
    assert held["available"] is False


def test_bad_specs_say_what_to_change(tmp_path):
    lib = tmp_path / "library"
    with pytest.raises(fl.LibraryError, match="id"):
        fl.build_entry({"id": "a b", "title": "T", "direction": "x"}, library_dir=lib)
    with pytest.raises(fl.LibraryError, match="title"):
        fl.build_entry({"id": "a", "direction": "x"}, library_dir=lib)
    with pytest.raises(fl.LibraryError, match="capture"):
        fl.build_entry({"id": "a", "title": "T", "direction": str(tmp_path / "missing.json")}, library_dir=lib)


def test_the_slow_motion_sound_flags_are_data(tmp_path):
    """dialogue (speech: the viewer fades it out in slow motion) and slowmo_audio (pitch | mute, for a film without dialogue)"""
    lib = tmp_path / "library"
    p = _direction(tmp_path, SEGS)
    base = {"id": "x", "title": "T", "direction": str(p)}
    talk, _ = fl.build_entry({**base, "dialogue": True}, library_dir=lib)
    assert talk["dialogue"] is True and talk["slowmoAudio"] == "pitch"
    mute, _ = fl.build_entry({**base, "slowmo_audio": "Mute"}, library_dir=lib)
    assert mute["dialogue"] is False and mute["slowmoAudio"] == "mute"
    with pytest.raises(fl.LibraryError, match="slowmo_audio"):
        fl.build_entry({**base, "slowmo_audio": "loud"}, library_dir=lib)
    with pytest.raises(fl.LibraryError, match="dialogue"):
        fl.build_entry({**base, "dialogue": "yes"}, library_dir=lib)
    bad = {"films": [{"id": "a", "title": "T", "capture": "C", "direction": "d", "dialogue": "yes", "slowmoAudio": "loud"}]}
    text = " | ".join(fl.check_library(bad))
    assert "dialogue must be" in text and "slowmoAudio must be" in text
    old = {"films": [{"id": "a", "title": "T", "capture": "C", "direction": "d"}]}       # a library file written before the flags existed
    assert fl.check_library(old) == []


def test_the_presets_say_which_film_carries_speech():
    presets = fl.read_presets()
    assert presets["recap"]["dialogue"] is True                                          # the class recap: the teachers speak
    for name in ("demo", "carlos_aline", "john_lula"):                                   # the music films keep the pitch behaviour
        assert not presets[name].get("dialogue") and presets[name].get("slowmo_audio", "pitch") == "pitch"


def test_upsert_keeps_the_other_films_and_their_order():
    lib = fl.empty_library()
    for i in "abc":
        fl.upsert(lib, {"id": i, "title": i})
    fl.upsert(lib, {"id": "b", "title": "B2"})
    assert [(f["id"], f["title"]) for f in lib["films"]] == [("a", "a"), ("b", "B2"), ("c", "c")]
    assert lib["version"] == fl.LIBRARY_VERSION


def test_check_finds_the_problems_the_viewer_would_trip_over():
    ok = {"films": [{"id": "a", "title": "T", "capture": "C", "direction": "d", "durationS": 10, "chapters": [{"startS": 0}, {"startS": 5}]}]}
    assert fl.check_library(ok) == []
    bad = {"films": [{"id": "a", "title": "T", "capture": "C", "direction": "d", "durationS": 10, "chapters": [{"startS": 5}, {"startS": 0}]},
                     {"id": "a", "title": "", "capture": "C", "direction": "d", "durationS": 10, "chapters": [{"startS": 12}]}]}
    text = " | ".join(fl.check_library(bad))
    assert "time order" in text and "repeated" in text and "no title" in text and "after the film's end" in text


def test_cli_build_writes_utf8_json_atomically_and_a_second_film_is_appended(tmp_path, capsys):
    lib = tmp_path / "library"
    p = _direction(tmp_path, SEGS)
    plan = tmp_path / "plan.toml"
    plan.write_text('[[chapters]]\ntitle = "Début"\nsegment = "open_title"\n\n[[chapters]]\ntitle = "Floor"\nsegment = "floor"\n', encoding="utf-8")
    assert fl.main(["build", "--id", "one", "--title", "Un × Deux", "--capture", "05_X", "--direction", str(p), "--chapters-file", str(plan),
                    "--library-dir", str(lib)]) == 0
    assert fl.main(["build", "--id", "two", "--title", "Two", "--capture", "06_X", "--direction", str(tmp_path / "nope.json"),
                    "--library-dir", str(lib)]) == 0
    raw = (lib / "library.json").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw
    data = json.loads(raw.decode("utf-8"))
    assert [f["id"] for f in data["films"]] == ["one", "two"] and data["films"][0]["title"] == "Un × Deux"
    assert data["films"][0]["chapters"][0]["title"] == "Début" and not list(lib.glob(".library-*"))
    # the same id again replaces it in place
    assert fl.main(["build", "--id", "one", "--title", "Uno", "--capture", "05_X", "--direction", str(p), "--library-dir", str(lib)]) == 0
    again = json.loads((lib / "library.json").read_text(encoding="utf-8"))
    assert [(f["id"], f["title"]) for f in again["films"]] == [("one", "Uno"), ("two", "Two")]
    assert fl.main(["check", "--library-dir", str(lib)]) == 0
    assert fl.main(["list", "--library-dir", str(lib)]) == 0
    assert "Uno" in capsys.readouterr().out


def test_dry_run_writes_nothing(tmp_path):
    lib = tmp_path / "library"
    p = _direction(tmp_path, SEGS)
    assert fl.main(["build", "--id", "x", "--title", "X", "--direction", str(p), "--library-dir", str(lib), "--dry-run"]) == 0
    assert not (lib / "library.json").exists()


def test_copy_film_brings_the_direction_narration_and_mix_along(tmp_path):
    lib = tmp_path / "library"
    p = _direction(tmp_path, SEGS)
    (p.parent / "narration").mkdir()
    (p.parent / "narration" / "timeline.json").write_text("{}", encoding="utf-8")
    (p.parent / "narration" / "other.json").write_text("{}", encoding="utf-8")
    (p.parent / "audio").mkdir()
    (p.parent / "audio" / "film_mix.wav").write_bytes(b"RIFF")
    entry, _ = fl.build_entry({"id": "c", "title": "C", "direction": str(p)}, library_dir=lib, copy=True)
    assert entry["direction"] == "films/c/film/direction.json"
    base = lib / "films" / "c" / "film"
    assert (base / "direction.json").exists() and (base / "narration" / "timeline.json").exists() and (base / "audio" / "film_mix.wav").exists()
    assert not (base / "narration" / "other.json").exists()


def test_the_preset_file_parses_and_its_chapter_anchors_exist_in_the_demo_direction():
    presets = fl.read_presets()
    assert {"demo", "recap"} <= set(presets)
    demo = presets["demo"]
    assert 8 <= len(demo["chapters"]) <= 14 and all(c.get("title") and c.get("segment") for c in demo["chapters"])
    direction = fl._abs(demo["direction"])
    if not direction.exists():
        pytest.skip("the demo film's work folder is not on this machine")
    ids = {s["id"] for s in fl.segments(fl.read_direction(direction))}
    assert all(c["segment"] in ids for c in demo["chapters"])
    chapters = fl.derive_chapters(fl.read_direction(direction), demo["chapters"])
    assert chapters[0]["startS"] == 0.0 and all(a["endS"] == b["startS"] for a, b in zip(chapters, chapters[1:]))
