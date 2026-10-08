"""dancecap.farm.timeline (core stage): grid / cell math, coverage computed from synthetic outputs, the PNG."""
import json

import numpy as np
import pytest

from _farm_core_fixtures import ANCHOR, K0, K1, make_spec, patch_root, write_cameras, write_motion, write_prepare
from dancecap.farm import timeline as tl


def ci(k):
    """Cell index of grid frame k in the synthetic dance."""
    return (k - K0) // 30


def test_grid_and_cell_math_of_the_full_dance():
    g = tl.Grid(30.151176844227862, -998, 4645)
    assert g.T == 5644 and g.n_cells == 189 and g.cell_frames == 30
    assert abs(g.seconds - 188.133) < 1e-2
    assert abs(g.t_start - (30.151176844227862 - 998 / 30)) < 1e-12
    assert g.k_of(g.t(1234)) == 1234 and g.k_of(30.151176844227862) == 0
    assert g.k_floor(g.t(10) + 0.01) == 10 and g.k_ceil(g.t(10) + 0.01) == 11
    assert g.k_ceil(g.t(10)) == 10 and g.k_floor(g.t(10)) == 10          # lattice points are exact, never off by one
    assert g.cell_k_range(0) == (-998, -969)
    assert g.cell_k_range(188) == (4642, 4645)                            # the last cell is short: 4 frames
    a, b = g.cell_t_range(5)
    assert abs((b - a) - 1.0) < 1e-9 and abs(a - (g.t_start + 5)) < 1e-9
    assert g.cell_of_k(-998) == 0 and g.cell_of_k(-969) == 0 and g.cell_of_k(-968) == 1 and g.cell_of_k(4645) == 188
    # the reused span k 0..1147 starts mid-cell 33 and ends mid-cell 71
    assert g.cell_of_k(0) == 33 and g.cell_k_range(33) == (-8, 21) and g.cell_of_k(1147) == 71


def test_dance_spec_of_the_real_take():
    spec = tl.DanceSpec.load("larissa_kadu_full")
    g = spec.grid
    assert (g.k0, g.k1, g.T) == (-998, 4645, 5644) and abs(g.anchor - 30.151176844227862) < 1e-12
    assert spec.reference == "larissa_kadu_38" and spec.ref_k == (0, 1147)
    assert list(spec.segments) == ["a", "b", "c", "d"] and spec.refined_dir == "smplx_rf2_noslip"


def test_flag_decoding():
    assert list(tl.decode_flag(["mono", "two_view", "multi", "absent", "???"])) == [1, 2, 3, 0, -1]
    assert list(tl.decode_flag(np.array([0, 1, 2, 3, 9]))) == [0, 1, 2, 3, -1]
    assert list(tl.flag_from_ncams([0, 1, 2, 3, 8])) == [-1, 1, 2, 3, 3]


def test_reference_span_is_complete_everything_else_is_not_done(tmp_path, monkeypatch):
    root = patch_root(monkeypatch, tmp_path)
    spec = make_spec()
    write_motion(root / "work" / "ref" / "smplx", range(0, 300), n_cams=4)
    write_motion(root / "work" / "ref" / "smplx_rf2_noslip", range(0, 300), n_cams=4)
    write_prepare(root / "work" / "ref", 0, 300, ["01", "02", "03", "04"], tag="c4")
    write_cameras(root, "dance", {"01": [0.0, 100.0], "02": [0.0, 100.0], "03": [0.0, 100.0], "04": [0.0, 100.0]})
    cov = tl.compute_coverage("dance", spec=spec, write=False, external=False)
    st = [c["state"] for c in cov.cells]
    assert st[:3] == ["none"] * 3 and st[3:13] == ["complete"] * 10 and st[13:] == ["none"] * 4
    assert cov.counts() == {"none": 7, "partial": 0, "complete": 10}
    assert cov.cells[0]["reasons"] == ["no data"] and cov.cells[5]["reasons"] == []
    assert cov.cells[5]["n_cams"] == 4 and cov.cells[5]["stage"] == "refined"
    assert cov.percent()["complete"] == pytest.approx(100 * 300 / 510, abs=0.1)
    assert cov.gaps() == [(K0, -1), (300, K1)]                      # the two not-done spans, as grid indices


def test_front_only_raw_fit_and_awaiting_stitch_are_partial(tmp_path, monkeypatch):
    root = patch_root(monkeypatch, tmp_path)
    spec = make_spec(segments={"a": {}, "b": {}})
    write_cameras(root, "dance", {c: [0.0, 100.0] for c in ("01", "02", "03", "04")})
    # a: prepared only (k -90..29); b: raw fit + refined but not stitched (k 300..419)
    write_prepare(root / "work" / "dance_a", -90, 120, ["01", "02", "03", "04"])
    write_prepare(root / "work" / "dance_b", 300, 120, ["01", "02", "03"])
    write_motion(root / "work" / "dance_b" / "smplx", range(300, 330), n_cams=3)
    cov = tl.compute_coverage("dance", spec=spec, write=False, external=False)
    assert cov.cells[0]["state"] == "partial" and cov.cells[0]["stage"] == "front"
    assert cov.cells[0]["reasons"] == ["front only (no 3D yet)"]
    assert cov.cells[ci(300)]["state"] == "partial" and cov.cells[ci(300)]["reasons"][0] == "raw fit, not refined"
    assert cov.cells[ci(330)]["reasons"] == ["front only (no 3D yet)"]
    write_motion(root / "work" / "dance_b" / "smplx_rf2_noslip", range(300, 420), n_cams=3)
    write_motion(root / "work" / "dance_b" / "smplx", range(300, 420), n_cams=3)
    cov = tl.compute_coverage("dance", spec=spec, write=False, external=False)
    assert all(c["state"] == "partial" for c in cov.cells[ci(300):])
    assert all("refined, awaiting stitch" in c["reasons"] for c in cov.cells[ci(300):])   # not in the stitched result yet
    assert not any(b >= 300 for a, b in cov.gaps())                                   # but nothing left to plan there
    # the stitched result takes them over: complete
    write_motion(root / "work" / "dance" / "smplx_rf2_noslip", range(300, 420), n_cams=3, flag="multi", owner="b")
    cov = tl.compute_coverage("dance", spec=spec, write=False, external=False)
    assert all(c["state"] == "complete" for c in cov.cells[ci(300):])


def test_mono_two_view_and_absent_are_partial_but_not_gaps(tmp_path, monkeypatch):
    root = patch_root(monkeypatch, tmp_path)
    spec = make_spec(reference=None)
    # one phone for the first 30 frames, two for the next 30, then three
    write_cameras(root, "dance", {"01": [ANCHOR - 3.0, ANCHOR + 40.0], "02": [ANCHOR - 2.0 + 1.0, ANCHOR + 40.0],
                                  "03": [ANCHOR + 0.0, ANCHOR + 40.0]})
    ks = np.arange(K0, K1 + 1)
    flag = np.where(ks < -60, "absent", np.where(ks < -30, "mono", np.where(ks < 0, "two_view", "multi")))
    ncam = np.where(ks < -30, 1, np.where(ks < 0, 2, 3))
    write_motion(root / "work" / "dance" / "smplx", ks, n_cams=ncam, flag=flag, owner="s")
    write_motion(root / "work" / "dance" / "smplx_rf2_noslip", ks, n_cams=ncam, flag=flag, owner="s")
    cov = tl.compute_coverage("dance", spec=spec, write=False, external=False)
    r0, r1, r2, r3 = (cov.cells[i]["reasons"] for i in range(4))
    assert "absent" in r0 and "mono" in r1 and "two_view" in r2 and r3 == []
    assert cov.cells[3]["state"] == "complete"
    assert [c["state"] for c in cov.cells[:3]] == ["partial"] * 3
    # nothing the cameras could improve is planned: absent / mono (every recording phone used) are unfixable
    mask = cov.gap_mask("missing")
    assert not mask[: 60].any()           # k -90..-31: absent / mono
    assert cov.frames.unfixable()[:60].all()
    assert not cov.gap_mask("all")[:60].any()


def test_failed_seam_and_reprojection_make_cells_partial(tmp_path, monkeypatch):
    root = patch_root(monkeypatch, tmp_path)
    spec = make_spec(reference=None)
    write_cameras(root, "dance", {c: [0.0, 100.0] for c in ("01", "02", "03")})
    ks = np.arange(K0, K1 + 1)
    rp = np.full(len(ks), 6.0)
    rp[(ks >= 150) & (ks < 165)] = 60.0           # 15 outliers in one cell: over the gate
    rp[(ks >= 210) & (ks < 213)] = 60.0           # 3 outliers: tolerated by the cell rule
    rep = np.zeros(len(ks), bool)
    rep[(ks >= 240) & (ks < 270)] = True          # a whole cell of repaired frames: handled, not a failure
    rp[(ks >= 240) & (ks < 270)] = 60.0
    write_motion(root / "work" / "dance" / "smplx_rf2_noslip", ks, n_cams=3, flag="multi", reproj=rp, repaired=rep)
    write_motion(root / "work" / "dance" / "smplx", ks, n_cams=3, flag="multi", reproj=rp, repaired=rep)
    t0, t1 = ANCHOR + 90 / 30, ANCHOR + 120 / 30
    (root / "work" / "dance" / "stitch.json").write_text(json.dumps({"seams": [
        {"name": "a|b", "blend_s": [t0, t1], "ok": False},
        {"name": "b|c", "blend_s": [ANCHOR + 12.0, ANCHOR + 13.0], "ok": False, "repaired": True}]}))
    cov = tl.compute_coverage("dance", spec=spec, write=False, external=False)
    assert cov.cells[ci(60)]["state"] == "complete"                # k 60..89; the failed seam window is cell 6 (k 90..119)
    cell = cov.cells[ci(90)]
    assert cell["state"] == "partial" and any(r.startswith("seam gate failed: a|b") for r in cell["reasons"])
    c_bad = cov.cells[ci(150)]
    assert c_bad["state"] == "partial" and any("reprojection over the gate on 15" in r for r in c_bad["reasons"])
    assert cov.cells[ci(210)]["state"] == "complete"               # 3 of 30 outliers
    assert cov.cells[ci(240)]["state"] == "complete"               # repaired frames are not failures
    assert cov.cells[ci(360)]["state"] == "complete"               # the seam b|c was repaired
    assert cov.gaps("missing") == [(90, 119)]                      # only the failed seam is re-planned, not the reprojection
    assert any(a <= 150 and b >= 164 for a, b in cov.gaps("all"))


def test_parse_stitch_tolerates_layouts():
    g = tl.Grid(ANCHOR, K0, K1)
    d = {"seams": {"a|b": {"blend_k": [10, 70], "gates": {"body": {"ok": True}, "elbow": {"ok": False}}},
                    "b|c": {"window_s": [ANCHOR + 5, ANCHOR + 6], "status": "passed"}}}
    got = {s["name"]: s for s in tl.parse_stitch(d, g)}
    assert got["a|b"]["k0"] == 10 and got["a|b"]["ok"] is False
    assert got["b|c"]["ok"] is True and (got["b|c"]["k0"], got["b|c"]["k1"]) == (150, 180)
    assert tl.parse_stitch({"seams": [{"name": "x"}]}, g) == []     # no window: ignored


def test_external_cells_only_raise(tmp_path, monkeypatch):
    root = patch_root(monkeypatch, tmp_path)
    spec = make_spec()
    write_motion(root / "work" / "ref" / "smplx_rf2_noslip", range(0, 300))
    write_motion(root / "work" / "ref" / "smplx", range(0, 300))
    g = spec.grid
    ext = {"cells": [{"t0": float(g.t(K0 + 30 * i)), "t1": float(g.t(K0 + 30 * i + 30)),
                      "state": "complete" if i in (0, 5) else "partial", "reasons": ["fits done"]} for i in range(17)]}
    (root / "work" / "dance").mkdir(parents=True, exist_ok=True)
    (root / "work" / "dance" / "coverage_cells.json").write_text(json.dumps(ext))
    cov = tl.compute_coverage("dance", spec=spec, write=False, external=True)
    assert cov.cells[0]["state"] == "complete" and cov.cells[0]["src"] == "external:coverage_cells.json"
    assert cov.cells[1]["state"] == "partial" and cov.cells[1]["reasons"] == ["coverage_cells.json: fits done"]
    assert cov.cells[5]["state"] == "complete" and cov.cells[5]["src"] == "computed"   # never lowered
    assert tl.load_cells(root / "work" / "dance" / "coverage_cells.json")[0]["state"] == "complete"
    cov2 = tl.compute_coverage("dance", spec=spec, write=False, external=False)
    assert cov2.cells[0]["state"] == "none"


def test_coverage_json_roundtrip_and_png(tmp_path, monkeypatch):
    root = patch_root(monkeypatch, tmp_path)
    spec = make_spec()
    write_motion(root / "work" / "ref" / "smplx_rf2_noslip", range(0, 300))
    write_motion(root / "work" / "ref" / "smplx", range(0, 300))
    write_prepare(root / "work" / "dance_a", -90, 120, ["01", "02"])
    spec.segments = {"a": {}}
    cov = tl.compute_coverage("dance", spec=spec, write=True, external=False,
                              chunks=[{"id": "a", "k0": -90, "k1": 29, "core_k0": -90, "core_k1": -1, "status": "running", "kind": "full"}])
    doc = json.loads((root / "work" / "dance" / "farm" / "coverage.json").read_text())
    assert doc["dance"] == "dance" and doc["cell_s"] == 1 and len(doc["cells"]) == 17
    assert set(doc["counts"]) == {"complete", "partial", "none"} and doc["cells"][0]["chunk"] == "a"
    for c in doc["cells"]:
        assert {"i", "t0", "t1", "state", "stage", "n_cams", "reasons", "chunk"} <= set(c)
    back = tl.Coverage.load("dance")
    assert back.counts() == cov.counts() and back.frames is not None
    assert back.gaps() == cov.gaps() and back.grid == cov.grid
    png = tl.render_png(back, tmp_path / "map.png")
    from PIL import Image
    im = Image.open(png)
    assert im.size[0] == 1080 and im.size[1] > 500
    assert "#" in tl.ascii_map(cov) and "complete" in tl.ascii_map(cov)


def test_on_lattice_check_rejects_foreign_grids(tmp_path):
    g = tl.Grid(ANCHOR, K0, K1)
    write_motion(tmp_path / "ok", range(0, 10))
    write_motion(tmp_path / "off", range(0, 10), anchor=ANCHOR + 0.011)     # 11 ms off the lattice
    assert tl.load_motion(tmp_path / "ok", g) is not None
    assert tl.load_motion(tmp_path / "off", g) is None
    assert tl.load_motion(tmp_path / "missing", g) is None
