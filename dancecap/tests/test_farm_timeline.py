"""dancecap.farm.timeline: grid / 1-s cell math and coverage computed from synthetic outputs."""
import json

import numpy as np
import pytest

from _farm_fixtures import ANCHOR, make_spec, patch_root, write_motion, write_prepare
from dancecap.farm import timeline as tl


def test_grid_cell_math_of_the_real_dance():
    g = tl.Grid(30.151176844227862, -998, 4645)
    assert g.T == 5644 and g.n_cells == 189 and g.cell_frames == 30
    assert abs(g.seconds - 188.133) < 1e-3
    assert g.cell_k_range(0) == (-998, -969)
    assert g.cell_k_range(188) == (4642, 4645)           # the last cell holds the 4 left-over frames
    assert g.cell_of_k(-998) == 0 and g.cell_of_k(-969) == 0 and g.cell_of_k(-968) == 1 and g.cell_of_k(4645) == 188
    for k in (-998, -1, 0, 1147, 4645):                  # t <-> k round trip on the lattice
        assert g.k_of(float(g.t(k))) == k
    assert g.k_floor(float(g.t(5)) + 1e-9) == 5 and g.k_ceil(float(g.t(5)) - 1e-9) == 5
    assert g.k_floor(float(g.t(5)) - 0.004) == 4         # a hair before frame 5 floors to 4
    t0, t1 = g.cell_t_range(188)
    assert abs((t1 - t0) - 4 / 30) < 1e-9
    assert tl.Grid.from_dict(g.to_dict()) == g


def test_runs_inclusive_k_indices():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert tl.runs(m, 100) == [(101, 102), (105, 105), (107, 109)]
    assert tl.runs(np.zeros(5, bool), 0) == []
    assert tl.runs(np.ones(4, bool), -2) == [(-2, 1)]


def test_dancespec_of_the_real_take_toml():
    s = tl.DanceSpec.load("larissa_kadu_full")
    assert s.grid.anchor == pytest.approx(30.151176844227862) and (s.grid.k0, s.grid.k1) == (-998, 4645)
    assert s.reference == "larissa_kadu_38" and s.ref_k == (0, 1147)
    assert set(s.segments) == {"a", "b", "c", "d"} and s.segments["a"]["fit_k"] == [-998, 120]
    assert s.refined_dir == "smplx_rf2_noslip"


def _dance(tmp_path, monkeypatch, k1=299, ref_end=89):
    patch_root(monkeypatch, tmp_path)
    return make_spec(k1=k1, reference="ref", ref_k=(0, ref_end), segments={"a": {"fit_k": [90, 179]}, "b": {"fit_k": [180, 269]}})


def test_coverage_none_partial_complete_from_outputs(tmp_path, monkeypatch):
    spec = _dance(tmp_path, monkeypatch)                           # 300 frames = 10 cells
    w = tmp_path / "work"
    write_motion(w / "ref" / "smplx", 0, 89, n_cams=8, flag=3)     # the reused reference: complete, cells 0-2
    write_motion(w / "ref" / "rf", 0, 89, n_cams=8, flag=3)
    write_prepare(w / "dance_a", 90, 179, ["01", "02", "03"])      # segment a: prepared (front) + raw fit (fit), cells 3-5
    write_motion(w / "dance_a" / "smplx", 90, 179, n_cams=3, flag=3)
    write_prepare(w / "dance_b", 180, 239, ["01", "02", "03"])     # segment b: front only on cells 6-7
    cov = tl.compute_coverage("dance", spec=spec, external=False, write=True)
    st = [c["state"] for c in cov.cells]
    assert st[:3] == ["complete"] * 3
    assert st[3:8] == ["partial"] * 5 and cov.cells[3]["stage"] == "fit" and "raw fit, not refined" in cov.cells[3]["reasons"]
    assert cov.cells[6]["stage"] == "front" and "front only (no 3D yet)" in cov.cells[6]["reasons"]
    assert st[8:] == ["none", "none"] and cov.cells[9]["reasons"] == ["no data"]
    assert cov.counts() == {"none": 2, "partial": 5, "complete": 3}
    assert cov.percent()["complete"] == pytest.approx(30.0, abs=0.1)
    # a cell cut by a span boundary is partial too (10 of its 30 frames refined)
    write_motion(w / "ref" / "smplx", 0, 99, n_cams=8, flag=3)
    write_motion(w / "ref" / "rf", 0, 99, n_cams=8, flag=3)
    cov2 = tl.compute_coverage("dance", spec=_dance(tmp_path, monkeypatch, ref_end=99), external=False, write=False)
    assert cov2.cells[3]["state"] == "partial" and "10/30 frames refined" in cov2.cells[3]["reasons"]
    # the file on disk has the FARM.md shape and reloads
    d = json.loads((w / "dance" / "farm" / "coverage.json").read_text())
    assert d["dance"] == "dance" and d["cell_s"] == 1 and len(d["cells"]) == 10 and set(d["counts"]) == {"complete", "partial", "none"}
    again = tl.Coverage.load("dance")
    assert again is not None and again.counts() == cov.counts() and again.frames is not None


def test_quality_flags_seams_reprojection_and_repaired_frames(tmp_path, monkeypatch):
    spec = _dance(tmp_path, monkeypatch, ref_end=119)
    w = tmp_path / "work"
    flags = np.full(120, 3, np.int8)
    flags[:30] = 1                                                 # cell 0 mono
    flags[30:60] = 2                                               # cell 1 two_view
    flags[60:90] = 0                                               # cell 2 absent
    rp = np.full(120, 6.0)
    rp[90:95] = 80.0                                               # 5 bad frames in cell 3
    write_motion(w / "ref" / "smplx", 0, 119, flag=flags, n_cams=8, reproj=rp)
    write_motion(w / "ref" / "rf", 0, 119, flag=flags, n_cams=8, reproj=rp)
    cov = tl.compute_coverage("dance", spec=spec, external=False, write=False)
    c = cov.cells
    assert [c[i]["state"] for i in range(4)] == ["partial"] * 4
    assert "mono" in c[0]["reasons"] and "two_view" in c[1]["reasons"] and "absent" in c[2]["reasons"]
    assert any("reprojection over the gate on 5 frames" in r for r in c[3]["reasons"])
    # frames the fit's own repair step interpolated are handled, not failures
    rep = np.zeros(120, bool)
    rep[90:95] = True
    write_motion(w / "ref" / "smplx", 0, 119, flag=3, n_cams=8, reproj=rp, repaired=rep)
    write_motion(w / "ref" / "rf", 0, 119, flag=3, n_cams=8, reproj=rp, repaired=rep)
    assert tl.compute_coverage("dance", spec=spec, external=False, write=False).cells[3]["state"] == "complete"


def test_stitch_json_failed_seam_makes_cells_partial(tmp_path, monkeypatch):
    spec = _dance(tmp_path, monkeypatch)
    w = tmp_path / "work"
    for d in ("smplx", "rf"):
        write_motion(w / "dance" / d, 0, 299, flag=3, n_cams=6, reproj=6.0)
    (w / "dance" / "stitch.json").write_text(json.dumps({"seams": [
        {"name": "a|b", "window_k": [120, 180], "gate": {"passed": False, "why": ["elbow p95"]}, "status": "ok"},
        {"name": "b|c", "window_k": [240, 270], "gate": {"passed": True, "why": []}, "status": "ok"}]}))
    cov = tl.compute_coverage("dance", spec=spec, external=False, write=False)
    assert [c["state"] for c in cov.cells[:4]] == ["complete"] * 4
    assert cov.cells[4]["state"] == "partial" and any("seam gate failed: a|b" in r for r in cov.cells[4]["reasons"])
    assert cov.cells[5]["state"] == "partial" and cov.cells[6]["state"] == "complete"   # window_k [120, 180) = cells 4 and 5
    assert cov.cells[8]["state"] == "complete"                     # the passing seam changes nothing
    assert tl.runs(cov.gap_mask("missing"), 0) == [(120, 179)]     # the failed window is a gap to refit (window_k end exclusive)


def test_parse_stitch_tolerates_layouts():
    g = tl.Grid(ANCHOR, 0, 999)
    ok = tl.parse_stitch({"seams": {"x": {"ok": False, "blend_s": [31.0, 32.0]}, "y": {"status": "passed", "blend_k": [10, 20]}}}, g)
    by = {s["name"]: s for s in ok}
    assert by["x"]["ok"] is False and (by["x"]["k0"], by["x"]["k1"]) == (30, 60)
    assert by["y"]["ok"] is True and (by["y"]["k0"], by["y"]["k1"]) == (10, 20)
    assert tl.parse_stitch({}, g) == [] and tl.parse_stitch({"seams": [{"name": "no window"}]}, g) == []


def test_unfixable_frames_are_not_gaps(tmp_path, monkeypatch):
    """Refined frames that stay mono / two_view because every recording phone is already used cannot be improved by a refit."""
    spec = _dance(tmp_path, monkeypatch)
    w = tmp_path / "work"
    write_motion(w / "dance" / "smplx", 0, 299, flag=1, n_cams=1)
    write_motion(w / "dance" / "rf", 0, 299, flag=1, n_cams=1)
    cams = {"01": (ANCHOR - 5, ANCHOR + 20.0)}                      # only one phone records, ever: nothing to add
    (w / "dance" / "farm").mkdir(parents=True)
    (w / "dance" / "farm" / "cameras.json").write_text(json.dumps({"spans": cams}))
    cov = tl.compute_coverage("dance", spec=spec, external=False, write=False)
    assert cov.counts()["complete"] == 0 and all(c["state"] == "partial" for c in cov.cells)
    assert cov.gaps("missing") == [] and cov.gaps("all") == []
    cams["02"] = (ANCHOR - 5, ANCHOR + 20.0)                        # a second phone records: the frames could be refit with it
    (w / "dance" / "farm" / "cameras.json").write_text(json.dumps({"spans": cams}))
    cov = tl.compute_coverage("dance", spec=spec, external=False, write=False)
    assert cov.gaps("all") == [(0, 299)] and cov.gaps("missing") == []


def test_external_cells_only_raise_and_accept_both_formats(tmp_path, monkeypatch):
    spec = _dance(tmp_path, monkeypatch)
    w = tmp_path / "work"
    write_motion(w / "ref" / "smplx", 0, 29, flag=3, n_cams=8)
    write_motion(w / "ref" / "rf", 0, 29, flag=3, n_cams=8)           # cell 0 complete
    t = lambda k: ANCHOR + k / 30                                       # noqa: E731
    ext = {"cells": [{"t0": t(0), "t1": t(30), "state": "none", "reasons": ["x"]},          # must not lower cell 0
                     {"t0": t(30), "t1": t(60), "state": "partial", "reasons": ["mamma fit done"]},
                     {"t0": t(60), "t1": t(90), "state": "done"}]}      # 'done' is the other workflow's word for complete
    p = w / "dance" / "coverage_cells.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(ext))
    cov = tl.compute_coverage("dance", spec=spec, external=True, write=False)
    assert [cov.cells[i]["state"] for i in range(4)] == ["complete", "partial", "complete", "none"]
    assert cov.cells[1]["reasons"] == ["coverage_cells.json: mamma fit done"] and cov.cells[1]["src"].startswith("external:")
    assert tl.compute_coverage("dance", spec=spec, external=False, write=False).cells[1]["state"] == "none"
    # farm's own coverage.json is read back as an external file too
    cov.save()
    assert len(tl.load_cells(w / "dance" / "farm" / "coverage.json")) == 10


def test_evidence_files_roundtrip_and_make_the_motion_files_unnecessary(tmp_path, monkeypatch):
    """A pod writes work/<take>/farm/evidence.npz next to its outputs; the workstation computes coverage from it alone."""
    spec = _dance(tmp_path, monkeypatch)
    w = tmp_path / "work"
    write_prepare(w / "dance_a", 90, 179, ["01", "02", "03"])
    write_motion(w / "dance_a" / "smplx", 90, 179, flag=3, n_cams=3)
    write_motion(w / "dance_a" / "rf", 90, 179, flag=3, n_cams=3)
    fr = tl.build_evidence(spec, w / "dance_a", "a")
    p = fr.to_npz(w / "dance_a" / tl.EVIDENCE)
    assert p.stat().st_size < 20000 and tl.Frames.from_npz(p, spec.grid).level.max() == tl.L_REFINED
    # only the evidence file is on this machine: the motion is gone
    for d in ("smplx", "rf"):
        for f in (w / "dance_a" / d).glob("*"):
            f.unlink()
    (w / "dance_a" / "mamma" / "prepare.json").unlink()
    cov = tl.compute_coverage("dance", spec=spec, external=False, write=False)
    assert all(c["stage"] == "refined" for c in cov.cells[3:6]) and any(s["kind"] == "evidence" for s in cov.sources)
    # but a chunk's own evidence is not the stitched result: cells stay 'awaiting stitch' until a final evidence file says so
    assert cov.cells[3]["state"] == "partial" and "refined, awaiting stitch" in cov.cells[3]["reasons"]
    final = tl.Frames(spec.grid)
    final.apply(np.arange(90, 180), tl.L_REFINED, n_cams=3, flag=3, final=True)
    final.to_npz(w / "dance" / tl.EVIDENCE)
    cov = tl.compute_coverage("dance", spec=spec, external=False, write=False)
    assert [c["state"] for c in cov.cells[3:6]] == ["complete"] * 3
    # a foreign grid is refused, not merged
    other = tl.Frames(tl.Grid(ANCHOR + 1.0, 0, 299))
    other.to_npz(w / "dance_b_bad.npz")
    with pytest.raises(ValueError):
        tl.Frames.from_npz(w / "dance_b_bad.npz", spec.grid)


def test_png_renders_with_chunks_and_states(tmp_path, monkeypatch):
    spec = _dance(tmp_path, monkeypatch)
    w = tmp_path / "work"
    write_motion(w / "ref" / "smplx", 0, 89, flag=3, n_cams=8)
    write_motion(w / "ref" / "rf", 0, 89, flag=3, n_cams=8)
    write_prepare(w / "dance_a", 90, 179, ["01", "02"])
    chunks = [{"id": "a", "k0": 80, "k1": 190, "core_k0": 90, "core_k1": 179, "status": "running", "kind": "full"}]
    cov = tl.compute_coverage("dance", spec=spec, chunks=chunks, external=False, write=False)
    assert cov.cells[4]["chunk"] == "a"
    out = tl.render_png(cov, tmp_path / "map.png")
    from PIL import Image
    im = Image.open(out)
    assert im.size[0] == 1080 and im.size[1] > 400
    txt = tl.ascii_map(cov)
    assert "#" in txt and "+" in txt and "." in txt
