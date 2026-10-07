"""Unit tests for dancecap.clock_studio (studio-derived phone clocks) and the sync-offsets clock guard (dancecap.sync
offsets_check, mamma_dyn.read_sync_offsets / prepare). Synthetic: no audio, no video, no GPU.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_clock_studio.py -q      (from the workspace root)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import clock_studio as cs  # noqa: E402
from dancecap import mamma_dyn as md  # noqa: E402
from dancecap import sync  # noqa: E402
from dancecap.take import PACKAGE_ROOT, Source, Take  # noqa: E402


class _Take(Take):
    """A Take whose work dir is a temp dir (Take.work is always PACKAGE_ROOT/work/<name>)."""
    _work: Path | None = None

    @property
    def work(self) -> Path:
        return self._work


def _take(name: str, work: Path, sources: list[Source], cfg: dict | None = None,
          window=(30.0, 31.0)) -> _Take:
    t = _Take(name=name, sources=sources, window=tuple(window), keyframe_span=tuple(window), keyframe_step=4.0,
              cfg=cfg or {})
    t._work = work
    return t


# ------------------------------------------------------------------------------------------------ clock math
def test_compose_inverts_song_to_source_and_maps_the_same_instant():
    rng = np.random.default_rng(3)
    for _ in range(20):
        s, o = 0.98 + 0.04 * rng.random(), rng.uniform(-10, 10)  # studio -> reference
        clk = {"slope": 1 + rng.uniform(-3e-4, 3e-4), "offset": rng.uniform(-5, 5)}  # phone source -> reference
        a, b = cs.song_to_source(s, o, clk)
        sl, of = cs.compose_clock(s, o, a, b)
        assert sl == pytest.approx(clk["slope"], rel=1e-12) and of == pytest.approx(clk["offset"], abs=1e-9)
        x = rng.uniform(0, 200, 5)  # song seconds: the phone instant it predicts lands on the studio line
        u = a * x + b
        assert np.allclose(clk["slope"] * u + clk["offset"], s * x + o, atol=1e-9)
    # same formula as song.camera_check's studio_derived_clock (sc = slope / a, oc = offset - slope * b / a)
    s, o, a, b = 0.98165042, -6.96837, 0.981, 1.234
    assert cs.compose_clock(s, o, a, b) == (s / a, o - s * b / a)


def test_clock_change_sign_and_drift():
    old = {"slope": 1.0, "offset": 0.0}
    later = {"slope": 1.0, "offset": 0.010}
    assert cs.clock_change_ms(old, later, 50.0) == pytest.approx(10.0)  # new clock puts the frame 10 ms later
    drift = {"slope": 1.0 + 1e-4, "offset": 0.0}  # +6 ms/min against the old clock
    assert cs.clock_change_ms(old, drift, 60.0) == pytest.approx(6.0, abs=1e-9)
    # Source round trip (what reuse-keyframes and ingest use)
    src = Source("01", Path("x.mp4"), 1.0002, 1.58)
    assert src.to_source(src.to_reference(12.345)) == pytest.approx(12.345, abs=1e-12)


def test_video_offsets_expected_moves_sync_deltas_by_the_clock_change(tmp_path):
    base = _take("base", tmp_path, [], window=(56.0, 62.0))
    (tmp_path / "sync").mkdir()
    (tmp_path / "sync" / "offsets.json").write_text(json.dumps({"offsets": {
        "01": {"delta_ms": 10.0, "stable": True, "spread_ms": 3.0},
        "02": {"delta_ms": -10.0, "stable": False, "spread_ms": 20.0},
        "03": {"delta_ms": 0.0, "stable": False, "spread_ms": 12.0}}}))
    old = {c: {"slope": 1.0, "offset": 0.0} for c in ("01", "02", "03")}
    new = {"01": {"slope": 1.0, "offset": 0.020}, "02": {"slope": 1.0, "offset": 0.0},
           "03": {"slope": 1.0, "offset": -0.005}}
    D = cs.video_offsets_expected(base, old, new)
    pc = D["per_camera"]
    # raw = delta_old - change: 01 -10, 02 -10, 03 +5 -> median -10 -> 0, 0, +15
    assert pc["01"]["clock_change_ms"] == pytest.approx(20.0)
    assert [pc[c]["expected_new_ms"] for c in ("01", "02", "03")] == pytest.approx([0.0, 0.0, 15.0])
    assert pc["01"]["stable"] and not pc["02"]["stable"]
    assert cs.video_offsets_expected(_take("none", tmp_path / "nope", []), old, new) is None


# ------------------------------------------------------------------------------------------------ window
def test_inward_window_rounding():
    assert cs.inward_window(30.007134973, 68.446816466) == (901 / 30, 2053 / 30)
    assert cs.inward_window(901 / 30, 2053 / 30) == (901 / 30, 2053 / 30)  # already on the grid: unchanged
    t0, t1 = cs.inward_window(30.0334, 30.0999)
    assert t0 == pytest.approx(30 + 2 / 30) and t1 == pytest.approx(30 + 2 / 30)


def _alignment(tmp_path: Path, clocks: dict[str, tuple[float, float]]) -> Path:
    p = tmp_path / "al.json"
    p.write_text(json.dumps({"reference": "a.mp4", "sources": [
        {"id": c, "name": f"{c}.mp4", "slope": sl, "offset": of, "accepted": True} for c, (sl, of) in clocks.items()]}))
    return p


def test_window_margins_and_sync_corrected_check(tmp_path):
    al = _alignment(tmp_path, {"07": (1.0, 30.0071), "04": (1.0, 0.0), "06": (1.0, 0.0)})
    spans = {"07.mp4": (0.0, 50.0, 1500), "04.mp4": (20.0, 68.4468, 1500), "06.mp4": (0.0, 80.0, 2400)}
    r = cs.window(al, set(), (30.0, 68.5), tmp_path, spans=spans, current=(901 / 30, 2053 / 30))
    assert r["window"] == [901 / 30, 2053 / 30] and r["limited_by"] == {"start": ["07"], "end": ["04"]}
    assert r["margins_ms"]["07"]["start"] == pytest.approx(1000 * (901 / 30 - 30.0071), abs=0.01)  # ~26.2 ms
    assert r["margins_ms"]["04"]["end"] == pytest.approx(1000 * (68.4468 - 2053 / 30), abs=0.01)  # ~13.5 ms
    assert r["current"]["inside_every_camera"]
    # sync: cam07's video is 40 ms later than its audio clock says -> its first frame moves past the window start
    offs = {"07": {"delta": 0.040, "rate": 0.0, "t_ref": 0.0}, "04": {"delta": -0.005, "rate": 0.0, "t_ref": 0.0}}
    r2 = cs.window(al, set(), (30.0, 68.5), tmp_path, sync_offsets=offs, spans=spans, current=(901 / 30, 2053 / 30))
    assert r2["frame_times"] == "sync_corrected"
    assert r2["cameras"]["07"]["sync_corrected"][0] == pytest.approx(30.0471)
    cur = r2["current"]
    assert not cur["inside_every_camera"] and set(cur["short_cameras"]) == {"07"}
    assert cur["short_cameras"]["07"]["start"] == pytest.approx(1000 * (901 / 30 - 30.0471), abs=0.01)
    assert cur["trimmed"] == [902 / 30, 2053 / 30]  # one frame off the start; cam04 still covers the end (8.5 ms)
    assert cur["margins_ms"]["04"]["end"] == pytest.approx(1000 * (68.4418 - 2053 / 30), abs=0.01)


def test_cli_window_take_and_validate_flags(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(cs, "take_window", lambda take: seen.setdefault("take", take) and {
        "take": take, "frame_times": "reference", "note": "", "window": [0, 1], "exact": [0, 1], "limited_by": {},
        "current": {}})
    monkeypatch.setattr(cs, "validate", lambda *a, **k: seen.update(validate=(a, k)))
    cs.main(["window", "--take", "t38"])
    assert seen["take"] == "t38" and '"take": "t38"' in capsys.readouterr().out
    cs.main(["validate", "--alignment", "x.json", "--review-take", "t38"])
    assert seen["validate"][1] == {"plot": True, "write_back": False}  # write-back only when asked
    cs.main(["validate", "--alignment", "x.json", "--review-take", "t38", "--write-back"])
    assert seen["validate"][1]["write_back"] is True


# ------------------------------------------------------------------------------------------------ write guards
def _clock_studio_file(p: Path):
    p.write_text(json.dumps({"sources": [], "provenance": {"module": "dancecap.clock_studio build"}}))


def test_build_and_validate_refuse_the_base_alignment_and_foreign_files(tmp_path, monkeypatch):
    base_al = tmp_path / "alignment.json"
    base_al.write_text(json.dumps({"reference": "a.mp4", "sources": []}))
    before = base_al.read_bytes()
    base = _take("base", tmp_path / "work_base", [], cfg={"alignment": str(base_al)})
    monkeypatch.setattr(Take, "load", classmethod(lambda cls, name: base))
    assert cs._alignment_path(base) == base_al.resolve()
    with pytest.raises(SystemExit, match="alignment file"):
        cs._check_target(base_al, base, "build")
    with pytest.raises(SystemExit, match="alignment file"):
        cs.build("base", base_al)  # refused before reading song.json
    with pytest.raises(SystemExit, match="alignment file"):
        cs.validate("base", base_al, "review", write_back=True)  # refused before decoding any audio
    other = tmp_path / "hand_made.json"
    other.write_text(json.dumps({"sources": []}))
    with pytest.raises(SystemExit, match="not written by dancecap.clock_studio"):
        cs._check_target(other, base, "validate --write-back")
    ours = tmp_path / "alignment_studio.json"
    _clock_studio_file(ours)
    cs._check_target(ours, base, "build")  # clock_studio's own file: allowed
    cs._check_target(tmp_path / "new.json", base, "build")  # new file: allowed
    assert base_al.read_bytes() == before


# ------------------------------------------------------------------------------------------------ keyframes
def test_reuse_keyframes_copies_and_remaps_t(tmp_path, monkeypatch):
    old = Source("07", tmp_path / "07.mp4", 1.0, 30.0)
    new = Source("07", tmp_path / "07.mp4", 1.00001, 30.0071)
    src = _take("src", tmp_path / "src", [old])
    dst = _take("dst", tmp_path / "dst", [new])
    takes = {"src": src, "dst": dst}
    monkeypatch.setattr(Take, "load", classmethod(lambda cls, name: takes[name]))
    kf = src.work / "keyframes" / "07"
    kf.mkdir(parents=True)
    recs = [{"file": f"kf_{int(1000 * old.to_reference(p))}.jpg", "pts": p, "t": old.to_reference(p)}
            for p in (0.0, 4.0, 8.0)]
    (kf / "keyframes.json").write_text(json.dumps(recs))
    masks = src.work / "people_masks_kf" / "07"
    masks.mkdir(parents=True)
    for r in recs:
        (kf / r["file"]).write_bytes(b"jpeg" + r["file"].encode())
        (masks / (Path(r["file"]).stem + ".png")).write_bytes(b"png")
    meta = cs.reuse_keyframes("src", "dst")
    assert meta["cameras"]["07"]["n"] == 3
    out = json.loads((dst.work / "keyframes" / "07" / "keyframes.json").read_text())
    for r_old, r_new in zip(recs, out):
        assert r_new["file"] == r_old["file"] and r_new["pts"] == r_old["pts"]
        assert r_new["t"] == pytest.approx(new.slope * r_new["pts"] + new.offset, abs=1e-12)
        a, b = kf / r_old["file"], dst.work / "keyframes" / "07" / r_new["file"]
        assert b.read_bytes() == a.read_bytes()
        assert os.stat(b).st_nlink == 1 and not os.path.samefile(a, b)  # a real copy, never a link
        m = dst.work / "people_masks_kf" / "07" / (Path(r_new["file"]).stem + ".png")
        assert m.exists() and os.stat(m).st_nlink == 1
    (dst.work / "keyframes" / "07" / recs[0]["file"]).write_bytes(b"re-ingested")  # writing the copy ...
    assert (kf / recs[0]["file"]).read_bytes().startswith(b"jpeg")  # ... never touches the base take


# ------------------------------------------------------------------------------------------------ sync guard
def _frames(work: Path, clocks: dict[str, tuple[float, float]], n: int = 3):
    for c, (sl, of) in clocks.items():
        d = work / "frames" / c
        d.mkdir(parents=True, exist_ok=True)
        d.joinpath("frames.json").write_text(json.dumps({
            "camera": c, "width": 64, "height": 48, "slope": sl, "offset": of, "frame_count": n,
            "frames": [{"file": f"{i:06d}.jpg", "pts": i / 30, "t": sl * i / 30 + of} for i in range(n)]}))


def _offsets(work: Path, clocks: dict | None, deltas: dict[str, float]):
    (work / "sync").mkdir(parents=True, exist_ok=True)
    doc = {"offsets": {c: {"delta": d, "delta_ms": 1000 * d, "stable": True, "use": True, "rate": 0.0,
                           "use_rate": False} for c, d in deltas.items()}}
    if clocks is not None:
        doc["clocks"] = {"alignment": "x.json", "sources": {c: list(v) for c, v in clocks.items()}}
    (work / "sync" / "offsets.json").write_text(json.dumps(doc))


def test_offsets_check_ok_stale_unrecorded(tmp_path, monkeypatch):
    clocks = {"01": (1.0002, 1.5843), "06": (1.0, 0.0)}
    _frames(tmp_path, clocks)
    assert sync.take_clocks(tmp_path) is None  # not a takes/<name>.toml work dir
    assert sync.offsets_check(tmp_path)["status"] == "missing"
    assert md.read_sync_offsets(tmp_path) == {}
    _offsets(tmp_path, clocks, {"01": 0.012, "06": -0.004})
    assert sync.offsets_check(tmp_path)["status"] == "ok"
    assert md.read_sync_offsets(tmp_path)["01"]["delta"] == pytest.approx(0.012)
    # measured on other clocks (e.g. copied from the take with alignment.json's clocks): refused everywhere
    _offsets(tmp_path, {"01": (1.0002, 1.5843 - 0.0253), "06": (1.0, 0.0)}, {"01": 0.012, "06": -0.004})
    chk = sync.offsets_check(tmp_path)
    assert chk["status"] == "stale" and "cam01" in chk["detail"][0] and "+25.30 ms" in chk["detail"][0]
    with pytest.raises(SystemExit, match="stale"):
        md.read_sync_offsets(tmp_path)
    with pytest.raises(SystemExit, match="stale"):
        md.load_tracks(tmp_path, allow_sfm_frame=True)
    # a legacy file (no clocks block): accepted unless the take requires sync
    _offsets(tmp_path, None, {"01": 0.012, "06": -0.004})
    assert sync.offsets_check(tmp_path)["status"] == "unrecorded"
    assert md.read_sync_offsets(tmp_path)["01"]["delta"] == pytest.approx(0.012)
    monkeypatch.setattr(sync, "take_clocks", lambda work: {"alignment": "x.json", "required": True,
                                                           "sources": {c: list(v) for c, v in clocks.items()}})
    with pytest.raises(SystemExit, match="unrecorded"):
        md.read_sync_offsets(tmp_path)
    # recorded clocks equal to the frames' but not to the take's current alignment (frames not re-ingested)
    _offsets(tmp_path, clocks, {"01": 0.012, "06": -0.004})
    monkeypatch.setattr(sync, "take_clocks", lambda work: {"alignment": "y.json", "required": True,
                                                           "sources": {"01": [1.0002, 1.6], "06": [1.0, 0.0]}})
    assert sync.offsets_check(tmp_path)["status"] == "stale"


def _camera_take(tmp_path: Path, monkeypatch, required: bool) -> Path:
    work = tmp_path / "w"
    clocks = {"01": (1.0, 0.0), "02": (1.0, 0.0)}
    _frames(work, clocks)
    (work / "cameras").mkdir(parents=True)
    for c in clocks:
        np.savez(work / "cameras" / f"{c}.npz", t=np.arange(3) / 30 + 30.0, K=np.tile(np.eye(3), (3, 1, 1)),
                 R=np.tile(np.eye(3), (3, 1, 1)), tvec=np.zeros((3, 3)), ok=np.ones(3, bool))
    tk = _take("req", work, [], cfg={"sync": {"required": required}}, window=(30.0, 30.1))
    monkeypatch.setattr(Take, "load", classmethod(lambda cls, name: tk))
    monkeypatch.setattr(sync, "take_clocks", lambda w: {"alignment": "a.json", "required": required,
                                                        "sources": {c: list(v) for c, v in clocks.items()}})
    return work


def test_prepare_refuses_without_this_takes_sync(tmp_path, monkeypatch):
    work = _camera_take(tmp_path, monkeypatch, required=True)
    with pytest.raises(SystemExit, match="--no-sync refused"):
        md.prepare("req", use_sync=False, skip_2d=True, allow_sfm_frame=True)
    with pytest.raises(SystemExit, match="sync/offsets.json is missing"):
        md.prepare("req", skip_2d=True, allow_sfm_frame=True)
    assert not (work / "mamma" / "prepare.json").exists() and not (work / "mamma" / "pod_job.json").exists()
    _offsets(work, None, {"01": 0.01, "02": -0.01})  # e.g. copied from another take: cannot be verified
    with pytest.raises(SystemExit, match="unrecorded"):
        md.prepare("req", skip_2d=True, allow_sfm_frame=True)
    assert not (work / "mamma" / "prepare.json").exists()


def test_sync_required_flag_of_the_38s_take():
    p = PACKAGE_ROOT / "takes" / "larissa_kadu_38.toml"
    if not p.exists():
        pytest.skip("takes/larissa_kadu_38.toml not present")
    assert sync.sync_required(Take.load("larissa_kadu_38"))
    assert not sync.sync_required(Take.load("larissa_kadu"))  # the 6 s take keeps its legacy sync file


# ------------------------------------------------------------------------------------------------ decoded grid
def _decoded(work: Path, cam: str, t: np.ndarray):
    d = work / "frames" / cam
    d.mkdir(parents=True, exist_ok=True)
    d.joinpath("frames.json").write_text(json.dumps({"frames": [{"file": f"{i:06d}.jpg", "pts": float(x), "t": float(x)}
                                                                for i, x in enumerate(t)]}))


def test_decoded_coverage_matches_mamma_prepare_grid(tmp_path):
    win = (901 / 30, 2053 / 30)  # 1153 grid points
    # what ingest decodes: frames strictly inside the window, ~33 ms apart, each phone on its own phase
    _decoded(tmp_path, "09", 30.0636 + np.arange(1152) / 30)  # 30.0636 .. 68.4303: starts 30.3 ms after the window
    _decoded(tmp_path, "07", 68.4066 - np.arange(1152)[::-1] / 30)  # 30.0400 .. 68.4066: ends 26.7 ms before it
    _decoded(tmp_path, "06", np.arange(30.0400, 68.4300, 1 / 30))
    d = cs.decoded_coverage(tmp_path, ["06", "07", "09", "04"], win)
    assert d["window_grid_T"] == 1153 and d["missing_cameras"] == ["04"]
    assert d["limited_by"] == {"start": ["09"], "end": ["07"]}
    g = d["grid"]
    ref = md.master_grid(30.0636, 68.4066)  # mamma_dyn.prepare's rule
    assert g["T"] == len(ref) == 1151 and g["t0"] == pytest.approx(30.0636) and not g["on_lattice"]
    assert d["frames_lost_vs_window"] == 2
    # sync deltas move the decoded edges: cam07 40 ms later -> its last frame no longer limits the end
    offs = {"07": {"delta": 0.040, "rate": 0.0, "t_ref": 0.0}, "09": {"delta": -0.010, "rate": 0.0, "t_ref": 0.0}}
    d2 = cs.decoded_coverage(tmp_path, ["06", "07", "09"], win, offs)
    assert d2["frame_times"] == "sync_corrected" and d2["per_camera"]["09"]["first"] == pytest.approx(30.0536)
    assert d2["limited_by"]["end"] == ["06"] and d2["limited_by"]["start"] == ["07"]
    assert cs.decoded_coverage(tmp_path / "none", ["01"], win) is None


def test_time_base_reports_the_reference_cameras_delta(tmp_path):
    al = tmp_path / "al.json"
    al.write_text(json.dumps({"reference": "06.mp4", "sources": [{"id": "06", "name": "06.mp4"},
                                                                 {"id": "07", "name": "07.mp4"}]}))
    tk = _take("t", tmp_path / "w", [], cfg={"alignment": str(al)})
    (tk.work / "clock_studio").mkdir(parents=True)
    (tk.work / "clock_studio" / "validation.json").write_text(json.dumps({"summary": {"D_video_offsets_expected": {
        "per_camera": {"06": {"expected_new_ms": -44.46}}}}}))
    tb = cs.time_base(tk, None)
    assert tb["reference_camera"] == "06" and tb["pose_minus_beat_ms_predicted"] == -44.46
    assert tb["pose_minus_beat_ms_measured"] is None
    tb = cs.time_base(tk, {"06": {"delta": -0.0312, "rate": 0.0, "t_ref": 0.0}})
    assert tb["pose_minus_beat_ms_measured"] == pytest.approx(-31.2)
