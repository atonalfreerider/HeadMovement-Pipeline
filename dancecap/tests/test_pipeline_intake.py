"""dancecap.pipeline.intake: copy-free registration of a new capture (videos + soundtrack + metadata + camera types)
into takes/<take>.toml, with validation. Uses a fake probe (fast) and, when ffmpeg exists, tiny real videos."""
import json
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_fixtures import fake_probe, have_ffmpeg, make_audio, make_root, make_video  # noqa: E402
from dancecap.pipeline import camtypes, intake, paths, probe, status, takes  # noqa: E402


@pytest.fixture()
def root(tmp_path):
    r = make_root(tmp_path)
    yield r
    paths.set_root(None)


def files(tmp_path, n=3, ext=".mp4"):
    d = tmp_path / "src" / "Couple-demo"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(1, n + 1):
        p = d / f"{i:02d}-cam{ext}"
        p.write_bytes(b"x")
        out.append(p)
    return out


def request(vids, **kw):
    r = {"take": "couple_demo", "source_mode": "reference",
         "videos": [{"cam": f"{i + 1:02d}", "path": str(p)} for i, p in enumerate(vids)],
         "dancers": {"lead": {"name": "Kai", "stature_m": "5'8\"", "description": "black shirt"},
                     "follow": {"name": "Lia", "stature_m": "165 cm", "shoe_sole_m": 0.0}},
         "dance": {"level": "intermediate", "category": "lesson", "song": "Tune", "artist": "Band"},
         "consent": {"lead": {"capture": True, "cloud_processing": True}, "follow": {"capture": True}}}
    r.update(kw)
    return r


@pytest.mark.parametrize("raw,metres", [("5'8\"", 1.727), ("5' 8\"", 1.727), ("5ft 8in", 1.727), ("173 cm", 1.73),
                                        ("173", 1.73), ("1.727", 1.727), ("1.727 m", 1.727), (165, 1.65),
                                        (1.62, 1.62), ("tall", None), ("", None)])
def test_parse_stature(raw, metres):
    assert intake.parse_stature(raw) == metres


def test_create_writes_a_take_the_backend_reads(root, tmp_path):
    vids = files(tmp_path)
    req = request(vids)
    req["videos"][0]["type"] = {"mount": "tripod", "zoom": False}
    req["videos"][1]["type"] = {"mount": "handheld", "zoom": True}
    res = intake.create(req, probe_fn=fake_probe())
    assert res["variant"] == "multi"
    text = paths.take_toml("couple_demo").read_text(encoding="utf-8")
    cfg = tomllib.loads(text)
    assert cfg["name"] == "couple_demo" and cfg["alignment"] == "../work/couple_demo/clock/alignment.json"
    assert cfg["dancers"]["lead"]["stature_m"] == 1.727 and cfg["dancers"]["follow"]["stature_m"] == 1.65
    assert cfg["dancers"]["follow"]["shoe_sole_m"] == 0.0 and cfg["dance"]["level"] == "intermediate"
    assert cfg["consent"]["lead"]["cloud_processing"] is True and cfg["consent"]["follow"]["cloud_processing"] == "?"
    assert cfg["cameras"]["01"] == {**cfg["cameras"]["01"], "file": "01-cam.mp4", "mount": "tripod", "zoom": False,
                                    "source": "user"}
    assert "mount" not in cfg["cameras"]["03"]  # no type given: the probe suggestion / review fills it in
    assert cfg["sync"]["required"] is True
    # the files are referenced in place (never copied)
    assert (paths.takes_dir() / cfg["source_dir"] / "01-cam.mp4").resolve() == vids[0].resolve()
    ij = json.loads((paths.take_work("couple_demo") / "intake.json").read_text(encoding="utf-8"))
    assert [v["cam"] for v in ij["videos"]] == ["01", "02", "03"]
    assert (paths.take_work("couple_demo") / "probe" / "probe.json").exists()
    # the rest of the backend reads it
    v = takes.videos("couple_demo", cfg)
    assert [x["cam"] for x in v] == ["01", "02", "03"] and v[0]["path"].resolve() == vids[0].resolve()
    types, _ = camtypes.load("couple_demo", ["01", "02", "03"])
    assert [t["state"] for t in types] == ["confirmed", "confirmed", "unset"]
    snap = status.snapshot("couple_demo", with_activity=False)
    s = {x["id"]: x for x in snap["stages"]}
    assert s["intake"]["status"] == "done" and s["probe"]["status"] == "done"
    assert s["audio_sync"]["status"] == "ready"     # the clock port runs next (python -m dancecap.pipeline audio-sync)
    assert s["window"]["status"] == "waiting"         # waits on the clocks
    assert s["ingest"]["status"] == "waiting"         # nothing that loads the take runs before its clock file exists
    assert takes.title(cfg) == "Kai + Lia x Tune"
    with pytest.raises(intake.IntakeError) as e:
        intake.create(req, probe_fn=fake_probe())
    assert e.value.problems[0]["field"] == "take"


def test_validation_problems(root, tmp_path):
    vids = files(tmp_path)
    req = request(vids, take="Bad Name")
    req["videos"][1]["cam"] = "01"
    req["videos"][2]["path"] = str(tmp_path / "missing.mp4")
    req["dancers"]["follow"]["stature_m"] = "3 m"
    req["dance"]["level"] = "pro"
    with pytest.raises(intake.IntakeError) as e:
        intake.create(req, probe_fn=fake_probe())
    fields = {p["field"] for p in e.value.problems}
    assert {"take", "videos[1].cam", "videos[2].path", "dancers.follow.stature_m", "dance.level"} <= fields
    with pytest.raises(intake.IntakeError) as e:
        intake.create(request(vids), probe_fn=fake_probe(seconds=2.0))
    assert any("too short" in p["message"] for p in e.value.problems)
    with pytest.raises(intake.IntakeError) as e:
        intake.create(request(vids[:2], capture_type="multi"), probe_fn=fake_probe())
    assert any(p["field"] == "capture_type" for p in e.value.problems)


def test_no_audio_camera_is_excluded_and_variant_can_be_forced_down(root, tmp_path):
    vids = files(tmp_path)

    def probe_fn(p):
        return fake_probe(audio=not str(p).endswith("03-cam.mp4"))(p)

    res = intake.create(request(vids, capture_type="auto"), dry_run=True, probe_fn=probe_fn)
    cfg = tomllib.loads(res["toml_preview"])
    assert cfg["exclude"] == ["03"] and res["variant"] == "two"
    assert any("no audio" in w for w in res["warnings"])
    assert not paths.take_toml("couple_demo").exists()  # dry run writes nothing
    res = intake.create(request(vids, capture_type="two"), dry_run=True, probe_fn=fake_probe())
    cfg = tomllib.loads(res["toml_preview"])
    assert cfg["pipeline"]["variant"] == "two" and "sync" not in cfg


@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")
def test_real_media_probe_and_create(root, tmp_path):
    d = tmp_path / "media"
    vids = [make_video(d / "01-a.mp4", 6.0), make_video(d / "02-b.mp4", 6.0, rate=60)]
    song = make_audio(d / "song.mp3")
    pr = probe.probe_file(vids[1])
    assert pr["kind"] == "video" and pr["audio"] and abs(pr["fps"] - 60) < 0.5 and pr["duration_s"] > 5
    assert probe.probe_file(song)["kind"] == "audio"
    req = request(vids, soundtrack={"path": str(song)})
    res = intake.create(req)
    cfg = tomllib.loads(paths.take_toml("couple_demo").read_text(encoding="utf-8"))
    assert res["variant"] == "two" and cfg["dance"]["song_file"].endswith("song.mp3")
    out = probe.run("couple_demo", suggest=False)
    assert set(out) == {"01", "02"} and out["02"]["fps"] > 59


def _link_dir(link: Path, target: Path):
    if sys.platform == "win32":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def test_source_dir_does_not_follow_a_junction(root, tmp_path):
    """A worktree whose work/ is a junction to the main checkout: source_dir names the path as given, not the link's target
    (resolve() made it climb five levels out of the checkout)."""
    real = tmp_path / "elsewhere"
    vids_real = files(real)
    link = tmp_path / "linked"
    try:
        _link_dir(link, real)
    except OSError as e:
        pytest.skip(f"cannot make a directory link here: {e}")
    vids = [link / p.relative_to(real) for p in vids_real]
    intake.create(request(vids), probe_fn=fake_probe())
    cfg = tomllib.loads(paths.take_toml("couple_demo").read_text(encoding="utf-8"))
    assert "linked" in cfg["source_dir"] and "elsewhere" not in cfg["source_dir"]
    assert (paths.takes_dir() / cfg["source_dir"] / "01-cam.mp4").is_file()
