"""Pure-function checks of dancecap.hair_groom (no SMPL-X model, no take data needed)."""
import json

import numpy as np
import pytest

from dancecap import hair_groom as hg


def test_fit_sphere_recovers_centre_and_radius():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    c0 = np.array([0.1, 1.6, -0.05])
    c, r = hg.fit_sphere(c0 + 0.09 * d)
    assert np.allclose(c, c0, atol=1e-9) and r == pytest.approx(0.09, abs=1e-9)


def test_az_el_conventions():
    az, el = hg.az_el(np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]))
    assert az[0] == pytest.approx(0) and az[1] == pytest.approx(90) and abs(az[2]) == pytest.approx(180)
    assert el[3] == pytest.approx(90) and el[0] == pytest.approx(0)


def test_hairline_is_lowest_at_the_nape():
    assert hg.hairline_el(0) > hg.hairline_el(90) > hg.hairline_el(180)
    assert hg.hairline_el(-180) == hg.hairline_el(180)


def test_strand_length_over_the_skull():
    r = 0.09
    # a crown root falls over a quarter circle (drop r, arc r pi/2) then hangs to the tip level
    L = hg.strand_length(1.75, 90.0, 1.20, r)
    assert L == pytest.approx(0.55 + r * (np.pi / 2 - 1))
    # roots below the head's equator hang straight
    assert hg.strand_length(1.55, -20.0, 1.20, r) == pytest.approx(0.35)
    # never shorter than 5 cm
    assert hg.strand_length(1.3, 10.0, 1.4, r) >= 0.05


def test_tip_level_blends_front_to_back():
    assert hg.tip_level(0.0, 1.34, 1.20) == pytest.approx(1.34)
    assert hg.tip_level(180.0, 1.34, 1.20) == pytest.approx(1.20)
    mid = hg.tip_level(95.0, 1.34, 1.20)
    assert 1.20 < mid < 1.34


def test_farthest_points_spread():
    rng = np.random.default_rng(1)
    P = rng.random((2000, 3))
    idx = hg.farthest_points(P, 50)
    assert len(set(idx.tolist())) == 50
    Q = P[idx]
    d = np.linalg.norm(Q[:, None] - Q[None], axis=-1) + np.eye(50)
    # far more even than random picks: minimum spacing well above that of a random subset
    R = P[rng.choice(2000, 50, replace=False)]
    dr = np.linalg.norm(R[:, None] - R[None], axis=-1) + np.eye(50)
    assert d.min() > 2 * dr.min()


def test_kabsch_angle():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(40, 3))
    th = np.radians(30)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert hg.kabsch_angle_deg(A, A @ Rz.T + 0.3) == pytest.approx(30, abs=1e-6)


def test_peaks_are_separated():
    t = np.arange(100) / 30
    w = np.zeros(100)
    w[[10, 11, 50, 80]] = [300, 290, 200, 100]
    p = hg.peaks(t, w, 3, sep_s=0.5)
    assert [x["k"] for x in p] == [10, 50, 80]


def test_to_unity_flips_z():
    assert hg.to_unity([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, -3.0]


def test_export_copies_into_capture_folders(tmp_path, monkeypatch):
    class FakeTake:
        name = "faketake"
        work = tmp_path / "work"
    (FakeTake.work / "avatar").mkdir(parents=True)
    (FakeTake.work / "avatar" / "hair_groom.json").write_text(json.dumps({"version": 1}))
    streaming = tmp_path / "StreamingAssets"
    for name, take in (("01_A", "faketake"), ("02_B", "other")):
        (streaming / name).mkdir(parents=True)
        (streaming / name / "capture.json").write_text(json.dumps({"provenance": {"take": take}}))
    (FakeTake.work / "export" / "01_A").mkdir(parents=True)
    out = hg.export(FakeTake, streaming=streaming)
    assert sorted(p.parent.name for p in out) == ["01_A", "01_A"]
    assert (streaming / "01_A" / "hair_groom.json").exists() and not (streaming / "02_B" / "hair_groom.json").exists()
    with pytest.raises(FileNotFoundError):
        hg.export(FakeTake, capture="03_missing", streaming=streaming)
