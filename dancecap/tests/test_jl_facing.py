"""Pure parts of the john_lula facing check (dancecap.mono4d.jl_facing): torso frame + rel yaw, classes, 2D left/right
order, the bib detector, the YuNet decoder, statistics, the HMM, disagreement windows, targets, the owner-label store and
the 127.0.0.1 review server. Synthetic data only (no frames, no masks, no fits)."""
from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pytest

import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # run from the workspace root too
from dancecap.mono4d import jl_facing as F


def _skeleton(fwd_yaw_deg: float = 0.0, at=(0.0, 0.0, 0.0)) -> np.ndarray:
    """(22, 3) joints of an upright body whose torso faces world yaw fwd_yaw_deg (0 = +Z) at `at`."""
    J = np.zeros((22, 3))
    J[F.L_HIP] = (0.1, 0.9, 0.0)
    J[F.R_HIP] = (-0.1, 0.9, 0.0)
    J[F.L_SHO] = (0.18, 1.45, 0.0)
    J[F.R_SHO] = (-0.18, 1.45, 0.0)
    J[F.PELVIS] = (0.0, 0.95, 0.0)
    a = np.radians(fwd_yaw_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return J @ Ry.T + np.asarray(at)


def test_torso_axes_canonical_faces_plus_z():
    fwd, up, left = F.torso_axes(_skeleton())
    assert np.allclose(fwd, [0, 0, 1], atol=1e-9)
    assert np.allclose(up, [0, 1, 0], atol=1e-9)
    assert np.allclose(left, [1, 0, 0], atol=1e-9)
    assert np.isclose(F.world_yaw_deg(fwd), 0.0)
    fwd90, _, _ = F.torso_axes(_skeleton(90.0))
    assert np.isclose(F.world_yaw_deg(fwd90), 90.0)


@pytest.mark.parametrize("cam, rel", [((0, 1.5, 5), 0.0), ((0, 1.5, -5), 180.0), ((5, 1.5, 0), 90.0), ((-5, 1.5, 0), -90.0)])
def test_rel_yaw_signs(cam, rel):
    J = _skeleton()
    fwd, up, _ = F.torso_axes(J)
    r = float(F.rel_yaw_deg(fwd, up, J[F.PELVIS], np.asarray(cam, float)))
    assert abs(F.wrap_deg(r - rel)) < 1e-6


def test_rel_yaw_vectorised_and_classes():
    Js = np.stack([_skeleton(y) for y in (0, 45, 90, 135, 180, -100)])
    fwd, up, _ = F.torso_axes(Js)
    rel = F.rel_yaw_deg(fwd, up, Js[:, F.PELVIS], np.array([0, 1.5, 6.0]))
    assert np.allclose(np.abs(rel), [0, 45, 90, 135, 180, 100], atol=1.0)
    assert F.facing_class(rel).tolist() == [0, 0, 1, 2, 2, 1]
    assert F.facing_class(np.array([np.nan]))[0] == -1


def test_lr_rho_front_and_back():
    # facing the camera: the body's left shoulder / hip on the image's right
    ls, rs, lh, rh = np.array([130, 80]), np.array([70, 80]), np.array([120, 140]), np.array([80, 140])
    assert F.lr_rho(ls, rs, lh, rh) > 0.5
    assert F.lr_rho(rs, ls, rh, lh) < -0.5


def test_project_pinhole():
    K = np.array([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
    uv, z = F.project(np.array([[0.0, 0.0, 2.0], [1.0, -1.0, 4.0]]), K, np.eye(3), np.zeros(3))
    assert np.allclose(uv, [[50, 40], [75, 15]]) and np.allclose(z, [2, 4])


def _bib_scene(digits: bool = True, colour: bool = False):
    img = np.full((300, 200, 3), 40, np.uint8)       # dark t-shirt
    own = np.zeros((300, 200), bool)
    own[20:290, 40:160] = True
    img[60:88, 80:120] = (40, 220, 230) if colour else 235    # bib paper 40 x 28 px (0.15 of the 270 px body)
    if digits:
        img[66:82, 86:89] = 30                       # "1"
        img[66:82, 96:99] = 30
        img[66:69, 96:110] = 30                      # "7"-ish strokes
        img[66:82, 106:109] = 30
    return img, own


def test_bib_detector_finds_a_white_rectangle_with_digits():
    img, own = _bib_scene()
    c = F.bib_candidates(img, own, None, (30, 150), 270.0)
    assert c and c[0]["score"] >= 0.6
    x0, y0, x1, y1 = c[0]["box"]
    assert 75 <= x0 <= 85 and 55 <= y0 <= 65 and 115 <= x1 <= 125


def test_bib_detector_rejects_plain_white_and_colour_and_out_of_band():
    img, own = _bib_scene(digits=False)
    c = F.bib_candidates(img, own, None, (30, 150), 270.0)
    assert not c or c[0]["score"] < 0.3
    img, own = _bib_scene(colour=True)
    c = F.bib_candidates(img, own, None, (30, 150), 270.0)
    assert not c or c[0]["score"] < 0.3
    img, own = _bib_scene()
    assert F.bib_candidates(img, own, None, (150, 290), 270.0) == []      # the band excludes it
    partner = np.zeros_like(own)
    partner[50:100, 70:130] = True                                         # covered by the partner's mask
    assert F.bib_candidates(img, own, partner, (30, 150), 270.0) == []


def test_point_segment_and_arm_distance():
    assert np.isclose(F.point_segment_distance((0, 1), (0, 0), (2, 0)), 1.0)
    assert np.isclose(F.point_segment_distance((3, 0), (0, 0), (2, 0)), 1.0)
    kp = np.zeros((133, 3))
    kp[5] = (0, 0, 0.9)
    kp[7] = (0, 10, 0.9)
    assert np.isclose(F.arm_distance(kp, 3, 5), 3.0)
    kp[:, 2] = 0.0
    assert F.arm_distance(kp, 3, 5) == np.inf


def test_yunet_decode_single_face():
    S = 64
    outs = []
    for st in (8, 16, 32):
        n = (S // st) ** 2
        outs.append(np.zeros((1, n, 1)))
    outs += [np.zeros((1, (S // st) ** 2, 1)) for st in (8, 16, 32)]
    outs += [np.zeros((1, (S // st) ** 2, 4)) for st in (8, 16, 32)]
    outs += [np.zeros((1, (S // st) ** 2, 10)) for st in (8, 16, 32)]
    # one face on the stride-16 grid at row 1, col 2: centre (2.5, 1.5) * 16, size exp(log 2) * 16 = 32
    n16 = S // 16
    i = 1 * n16 + 2
    outs[1][0, i, 0] = 0.81
    outs[4][0, i, 0] = 1.0
    outs[7][0, i] = (0.5, 0.5, np.log(2.0), np.log(2.0))
    d = F.yunet_decode(outs, S, thr=0.5)
    assert len(d) == 1
    x, y, w, h = d[0]["box"]
    assert np.isclose(d[0]["score"], 0.9) and np.isclose(w, 32) and np.isclose(x + w / 2, 40) and np.isclose(y + h / 2, 24)


def test_precision_recall():
    pr = F.precision_recall(np.array([1, 1, 0, 0, 1], bool), np.array([1, 0, 1, 0, 1], bool))
    assert pr["tp"] == 2 and pr["fp"] == 1 and pr["fn"] == 1
    assert pr["precision"] == round(2 / 3, 3) and pr["recall"] == round(2 / 3, 3)
    assert F.precision_recall(np.zeros(3, bool), np.zeros(3, bool))["precision"] is None


def test_hmm_forward_backward():
    lik = np.full((20, 3), 1 / 3)
    lik[10] = (0.01, 0.01, 0.98)
    P = F.hmm_forward_backward(lik, F.TRANS)
    assert np.allclose(P.sum(1), 1)
    assert P[10].argmax() == 2 and P[11, 2] > P[15, 2] > 1 / 3      # evidence spreads to the neighbours, decaying
    iid = F.hmm_forward_backward(lik, np.full((3, 3), 1 / 3))
    assert np.allclose(iid[10], lik[10] / lik[10].sum())


def test_runs():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert F.runs(m) == [(1, 2), (5, 5), (7, 9)]
    assert F.runs(m, max_gap=1) == [(1, 2), (5, 9)]
    assert F.runs(m, min_len=2) == [(1, 2), (7, 9)]
    assert F.runs(np.zeros(4, bool)) == []


def test_disagreements_flip_window():
    post = np.tile([0.1, 0.1, 0.8], (30, 1))
    post[:10] = (0.9, 0.05, 0.05)
    post[10:20] = (0.02, 0.03, 0.95)                    # the image says back in 10..19
    fit = np.zeros(30, int)                             # the fit stays front all along
    fit[20:] = 2
    w = F.disagreements(post, fit)
    assert len(w) == 1 and w[0]["start"] == 10 and w[0]["end"] == 19 and w[0]["kind"] == "flip180"
    assert w[0]["image_class"] == "back"


def test_target_yaw_and_delta():
    ty, half = F.target_yaw(2, 30.0, 40.0)              # back: away from the camera direction
    assert np.isclose(ty, -150.0) and np.isclose(half, 60.0)
    ty, half = F.target_yaw(0, 30.0, 40.0)
    assert np.isclose(ty, 30.0)
    ty, _ = F.target_yaw(1, 0.0, 80.0)                  # side: the profile nearer the fit
    assert np.isclose(ty, 90.0)
    ty, _ = F.target_yaw(1, 0.0, -100.0)
    assert np.isclose(ty, -90.0)
    assert F.delta_into_band(10.0, 0.0, 60.0) == 0.0
    assert np.isclose(F.delta_into_band(0.0, 180.0, 60.0), 120.0) or np.isclose(F.delta_into_band(0.0, 180.0, 60.0), -120.0)
    assert np.isclose(F.delta_into_band(100.0, 0.0, 60.0), -40.0)


def test_merge_owner_labels(tmp_path):
    p = tmp_path / "owner_labels.json"
    F.merge_owner_labels(p, {"labels": {"follow:120": {"label": "back", "window": "w00", "t": 4.0}}})
    d = F.merge_owner_labels(p, {"labels": {"lead:7": "front", "follow:120": {"label": "side"}}})
    assert d["labels"]["follow:120"]["label"] == "side" and d["labels"]["lead:7"]["label"] == "front"
    assert json.loads(p.read_text())["labels"]["follow:120"]["label"] == "side"
    for bad in ({"labels": {"bob:1": "front"}}, {"labels": {"lead:x": "front"}}, {"labels": {"lead:1": "upside"}}, {"x": 1}):
        with pytest.raises(ValueError):
            F.merge_owner_labels(p, bad)


def test_review_server_serves_and_saves(tmp_path):
    root = tmp_path / "review"
    root.mkdir()
    (root / "index.html").write_text("<!doctype html><title>t</title>ok")
    labels = tmp_path / "owner_labels.json"
    srv = F.make_server(0, root, labels)
    assert srv.server_address[0] == "127.0.0.1"
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert b"ok" in urllib.request.urlopen(base + "/index.html", timeout=5).read()
        req = urllib.request.Request(base + "/api/labels", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"labels": {"follow:300": {"label": "back"}}}).encode())
        r = json.loads(urllib.request.urlopen(req, timeout=5).read())
        assert r["ok"] and r["n"] == 1
        got = json.loads(urllib.request.urlopen(base + "/api/labels", timeout=5).read())
        assert got["labels"]["follow:300"]["label"] == "back"
        bad = urllib.request.Request(base + "/api/labels", method="POST", data=b'{"labels": {"x:1": "front"}}',
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad, timeout=5)
        assert e.value.code == 400
    finally:
        srv.shutdown()
        srv.server_close()


def test_disagreements_skip_side_and_boundary():
    post = np.tile([0.05, 0.9, 0.05], (12, 1))          # the image says side (unreliable): no window by default
    fit = np.zeros(12, int)
    assert F.disagreements(post, fit) == []
    assert len(F.disagreements(post, fit, include_side=True)) == 1
    post = np.tile([0.92, 0.04, 0.04], (12, 1))         # the image says front, the fit says side at 70 deg: a boundary case
    fit = np.ones(12, int)
    assert F.disagreements(post, fit, fit_rel=np.full(12, 70.0)) == []
    assert len(F.disagreements(post, fit, fit_rel=np.full(12, 95.0))) == 1


@pytest.mark.parametrize("yaw, cam", [(98.4, (3.0, 1.5, 9.0)), (-140.0, (-4.0, 1.4, 2.0)), (10.0, (0.5, 1.6, -6.0))])
def test_rel_equals_camera_heading_minus_body_heading(yaw, cam):
    """For an upright body the rel yaw is the world heading of pelvis->camera minus the torso's heading."""
    J = _skeleton(yaw, at=(1.0, 0.0, 2.0))
    fwd, up, _ = F.torso_axes(J)
    C = np.asarray(cam, float)
    rel = float(F.rel_yaw_deg(fwd, up, J[F.PELVIS], C))
    d = C - J[F.PELVIS]
    assert abs(F.wrap_deg(rel - (F.world_yaw_deg(d) - F.world_yaw_deg(fwd)))) < 1e-6
    ty, half = F.target_yaw(0, float(F.world_yaw_deg(d)), float(F.world_yaw_deg(fwd)))
    assert abs(F.wrap_deg(ty - F.world_yaw_deg(d))) < 1e-9
