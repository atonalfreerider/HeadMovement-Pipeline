"""Unit tests for dancecap.cameras.twoview on synthetic two-camera data (no GPU, no take data, no frames).

    ../.venv/bin/python -m pytest tests/test_cameras_twoview.py -q

The synthetic take: two dancers (COCO-17, rigid legs, swinging arms, translation with stops) seen by a panning tripod
'02' (30 fps, the master) and an orbiting portrait camera '01' (29 fps) whose frame labelled L shows the scene at
L + delta(L); camera tracks are written in an SfM frame that a known Sim3 maps to the world.
"""
from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap import mamma_dyn as md  # noqa: E402
from dancecap import sync as S  # noqa: E402
from dancecap.cameras import twoview as tv  # noqa: E402
from dancecap.cameras import world as W  # noqa: E402
from dancecap.mono4d import atlas as A  # noqa: E402

STAT = {"lead": (1.75, 0.025), "follow": (1.63, 0.06)}
CFG = {"dancers": {"lead": {"stature_m": 1.75, "shoe_sole_m": 0.025}, "follow": {"stature_m": 1.63, "shoe_sole_m": 0.06}},
       "cameras": {"01": {"mount": "handheld"}, "02": {"mount": "tripod"}, "x_cam": "02"}}
SIM3 = {"s": 2.5, "R": Rot.from_euler("xyz", [10, 35, -5], degrees=True).as_matrix().tolist(), "t": [1.0, -2.0, 0.5]}
QUIET = lambda *a, **k: None  # noqa: E731


def _look_at(C, target, up=np.array([0, 1.0, 0])):
    """OpenCV world->cam rotation for a camera at C looking at target (world Y up -> image y down)."""
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def _template(H, sole):
    """COCO-17 rest pose whose unfolded chain is exactly world.py's CHAIN_FRAC x chain_stature and whose ankles stand at
    the expected ankle-joint height (dancer frame: x right, y up)."""
    Lc = W.CHAIN_FRAC * W.chain_stature(H, sole)
    ank = W.ANKLE_FRAC * H + sole
    knee, hip = ank + 0.24 * Lc, ank + 0.48 * Lc
    sh = hip + 0.30 * Lc
    eye = sh + 0.22 * Lc
    P = np.zeros((17, 3))
    P[0] = [0, eye - 0.03, 0.08]
    P[1], P[2] = [-0.03, eye, 0], [0.03, eye, 0]
    P[3], P[4] = [-0.07, eye, -0.02], [0.07, eye, -0.02]
    P[5], P[6] = [-0.18, sh, 0], [0.18, sh, 0]
    P[7], P[8] = [-0.20, sh - 0.28, 0], [0.20, sh - 0.28, 0]
    P[9], P[10] = [-0.22, sh - 0.54, 0], [0.22, sh - 0.54, 0]
    P[11], P[12] = [-0.10, hip, 0], [0.10, hip, 0]
    P[13], P[14] = [-0.10, knee, 0], [0.10, knee, 0]
    P[15], P[16] = [-0.10, ank, 0], [0.10, ank, 0]
    return P


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0], R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = 1, c, -s, s, c
    return R


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 1, 1], R[:, 0, 0], R[:, 0, 2], R[:, 2, 0], R[:, 2, 2] = 1, c, s, -s, c
    return R


def _scene(t):
    """World joints (T, 2, 17, 3) of lead and follow at true times t."""
    t = np.atleast_1d(np.asarray(t, float))
    th = 0.35 * (t - np.sin(2 * np.pi * t / 4) * 4 / (2 * np.pi))  # the couple circles with a stop every 4 s
    cen = np.stack([0.6 * np.cos(th), 0 * t, 0.6 * np.sin(th)], 1)
    out = []
    for r, (H, sole) in enumerate(STAT.values()):
        P = np.broadcast_to(_template(H, sole), (len(t), 17, 3)).copy()
        al = 0.7 * np.sin(2 * np.pi * 1.1 * t + r)
        for sh_, arm, sgn in ((5, (7, 9), 1), (6, (8, 10), -1)):
            Rx = _rx(sgn * al)
            for k in arm:
                P[:, k] = np.einsum("tij,tj->ti", Rx, P[:, k] - P[:, sh_]) + P[:, sh_]
        Ry = _ry(0.6 * np.sin(2 * np.pi * t / 5) + r * np.pi)
        P = np.einsum("tij,tkj->tki", Ry, P) + cen[:, None] + np.array([-0.3 + 0.6 * r, 0, 0])
        out.append(P)
    return np.stack(out, 1)


def _project(X, R, C, f, w, h, k1):
    """X (T, ..., 3) world, per-frame R (T,3,3), C (T,3) -> pixels (T, ..., 2)."""
    sh = X.shape
    Xc = np.einsum("tij,t...j->t...i", R, X - C.reshape((len(C),) + (1,) * (len(sh) - 2) + (3,)))
    x = Xc[..., :2] / Xc[..., 2:3]
    d = 1 + k1 * (x ** 2).sum(-1, keepdims=True)
    return x * d * f + np.array([w / 2 - 0.5, h / 2 - 0.5])


def _cam02(t):
    C = np.array([0.3, 1.6, 6.0])
    R = np.array([_look_at(C, np.array([0.5 * np.sin(0.3 * x), 1.0, 0.0])) for x in t])
    return R, np.broadcast_to(C, (len(t), 3)).copy()


def _cam01(L):
    ph = 0.15 * np.asarray(L, float)
    C = np.stack([3 * np.sin(ph), np.full_like(ph, 1.5), 3 * np.cos(ph)], 1)
    return np.array([_look_at(c, np.array([0, 1.0, 0])) for c in C]), C


def _write_cam(work, cam, t, Rw, Cw, f, w, h, k1, slope=1.0, offset=0.0):
    F = len(t)
    tw = -np.einsum("fij,fj->fi", Rw, Cw)
    Rs, ts = md.sfm_cameras(Rw, tw, SIM3)
    K = np.zeros((F, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2 - 0.5, h / 2 - 0.5, 1.0
    (work / "cameras").mkdir(parents=True, exist_ok=True)
    np.savez(work / "cameras" / f"{cam}.npz", t=t, K=K, R=Rs, tvec=ts, dist=np.full(F, k1), ok=np.ones(F, bool),
             reproj_px=np.full(F, 0.8))
    fd = work / "frames" / cam
    fd.mkdir(parents=True, exist_ok=True)
    (fd / "frames.json").write_text(json.dumps({"camera": cam, "width": w, "height": h, "slope": slope, "offset": offset,
                                                "frames": [{"file": f"{i:06d}.jpg", "pts": float(x), "t": float(x)}
                                                           for i, x in enumerate(t)]}))


def _write_people(work, cam, t, px, rng, roles):
    """people/<cam>.npz with lead, follow and a static judge in shuffled slots; roles.json boxes of the dancers."""
    F = len(t)
    judge = np.stack([np.linspace(20, 60, 17), np.linspace(30, 200, 17)], 1)
    kp = np.zeros((F, 3, 17, 3), np.float32)
    boxes = np.zeros((F, 3, 4), np.float32)
    frames = []
    for i in range(F):
        order = (0, 1, 2) if i % 2 == 0 else (2, 1, 0)  # slots: lead, follow, judge / judge, follow, lead
        persons = [px[i, 0] + rng.normal(0, 0.4, (17, 2)), px[i, 1] + rng.normal(0, 0.4, (17, 2)), judge]
        fr = {"i": i, "t": float(t[i])}
        for slot, who in enumerate(order):
            xy = persons[who]
            kp[i, slot, :, :2], kp[i, slot, :, 2] = xy, 0.9
            boxes[i, slot] = [xy[:, 0].min() - 10, xy[:, 1].min() - 10, xy[:, 0].max() + 10, xy[:, 1].max() + 10]
            if who < 2:
                fr[tv.ROLES[who]] = {"box": boxes[i, slot].tolist(), "conf": 0.9}
        frames.append(fr)
    (work / "people").mkdir(parents=True, exist_ok=True)
    np.savez(work / "people" / f"{cam}.npz", t=t, boxes=boxes, scores=np.full((F, 3), 0.9, np.float32),
             valid=np.ones((F, 3), bool), kpts=kp)
    roles[cam] = {"frames": frames}


def _delta_lin(L):
    return 0.037 + 1e-4 * (np.asarray(L, float) - 40.0)  # 37 ms at 40 s, drift 10 ms / 100 s


def make_work(root: Path, delta=_delta_lin, seed=0) -> Path:
    """The synthetic two-camera take in root (cameras in the SfM frame, world_from_sfm.json = SIM3)."""
    rng = np.random.default_rng(seed)
    work = root
    tm = 10 + np.arange(1798) / 29.97
    R2, C2 = _cam02(tm)
    _write_cam(work, "02", tm, R2, C2, 1500.0, 1280, 720, -0.05)
    L = 9.0 + np.arange(int(62.5 * 29)) / 29.0
    R1, C1 = _cam01(L)
    _write_cam(work, "01", L, R1, C1, 1000.0, 720, 1280, 0.02, slope=0.9999163, offset=12.1229)
    roles: dict = {}
    _write_people(work, "02", tm, _project(_scene(tm), R2, C2, 1500.0, 1280, 720, -0.05), rng, roles)
    _write_people(work, "01", L, _project(_scene(L + delta(L)), R1, C1, 1000.0, 720, 1280, 0.02), rng, roles)
    (work / "dancers").mkdir(parents=True, exist_ok=True)
    (work / "dancers" / "roles.json").write_text(json.dumps({"cams": roles, "labels_source": {}, "swaps_repaired": [],
                                                             "identity_flags": {}}))
    (work / "cameras" / "world_from_sfm.json").write_text(json.dumps(SIM3))
    return work


@pytest.fixture(scope="module")
def synced(tmp_path_factory):
    work = make_work(tmp_path_factory.mktemp("ll"))
    doc = tv.cmd_sync(work, "02", "01", range_ms=300.0, sub_s=15.0, log=QUIET)
    return work, doc


# ------------------------------------------------------------------------------------------------ helpers


def test_label_at_inverts_apply_offset():
    off = {"delta": 0.031, "rate": 2e-5, "t_ref": 80.0}
    L = np.linspace(10, 200, 7)
    assert np.allclose(tv.label_at(md.apply_offset(L, off), off), L, atol=1e-12)
    assert np.allclose(tv.label_at(L, None), L)


def test_interp_kp_never_bridges_a_gap_and_keeps_frames():
    kt = np.arange(6) / 30.0
    kp = np.stack([np.arange(6) * 3.0, np.zeros(6), np.full(6, 0.9)], 1)[:, None]  # (6, 1, 3): x moves 90 px/s
    kp[3] = np.nan
    out, spd = tv.interp_kp(kt, kp, np.array([0.5, 2.5, 3.5, 2.0, 4.0]) / 30.0)
    assert out[0, 0, 0] == pytest.approx(1.5) and spd[0, 0] == pytest.approx(90.0)
    assert np.isnan(out[1, 0, 0]) and np.isnan(out[2, 0, 0])  # either side of the missing frame
    assert out[3, 0, 0] == pytest.approx(6.0) and out[4, 0, 0] == pytest.approx(12.0)  # on a frame next to the gap
    out, _ = tv.interp_kp(kt, kp, np.array([-0.1, 0.3]))
    assert np.isnan(out[:, 0, 0]).all()  # outside the track


def test_chordal_mean_is_robust():
    rng = np.random.default_rng(1)
    Q = Rot.from_rotvec([0.3, -1.1, 0.4]).as_matrix()
    M = np.array([Rot.from_rotvec(rng.normal(0, np.radians(0.02), 3)).as_matrix() @ Q for _ in range(200)])
    M[:40] = Rot.random(40, random_state=2).as_matrix()  # 20 % outliers
    Qh, th = tv.chordal_mean(M)
    assert tv.rot_deg(Qh.T @ Q) < 0.01
    assert np.median(th[40:]) < 0.05 and np.min(th[:40]) > 1.0


def _fps_filter_times(n, shift, cut_fps=30):
    """An independent model of `ffmpeg -ss <shift> -i <59.94 fps original> -vf fps=30` (libavfilter vf_fps, round=near),
    in exact rationals: the seek drops the source frames before shift, source frame k gets the output pts
    round((k / 59.94 - shift) x 30) (halves away from zero), and output frame n holds the last source frame whose pts is
    <= n (the filter drops its buffered frame once the next one's pts is <= the output pts). -> this take's time of n."""
    src, S = Fraction(60000, 1001), Fraction(shift).limit_denominator(10000)
    n = np.asarray(n, int)
    keep, pts = [], []
    for k in range(int(S * src), int((S + Fraction(int(n.max()) + 2, cut_fps)) * src) + 2):
        tk = k / src - S
        if tk >= 0:
            keep.append(k)
            pts.append(math.floor(tk * cut_fps + Fraction(1, 2)))
    keep = np.array(keep)
    return keep[np.searchsorted(np.array(pts), n, side="right") - 1] / float(src)


def test_ptz_sample_times_follow_the_fps_filter():
    n = np.arange(6000)
    for shift in (14.0, 3.7):
        assert np.allclose(tv.ptz_sample_times(n, shift), _fps_filter_times(n, shift), atol=1e-12, rtol=0)
    t = tv.ptz_sample_times(n, 14.0)
    assert round(t[0] * 60000 / 1001) == 840  # the first source frame after the seek point (the nearest, 839, is before it)
    assert np.all(np.diff(t) > 0)
    lag = t - tv.ptz_sample_times(n, 14.0, rule="nominal")
    # 0-16.7 ms after S + n / 30 (down to -1/1001 of a source frame where (n + 0.5) / 30 nearly hits a source frame)
    assert lag.min() >= -1001 / 60000 / 1000 - 1e-12 and lag.max() <= 1001 / 60000 + 1e-9 and lag.mean() > 0.008
    d = np.round((t - tv.ptz_sample_times(n, 14.0, rule="nearest")) * 60000 / 1001)
    ph = tv.cut_phase(n, 14.0)
    assert set(np.unique(d)) == {0.0, 1.0}  # the nearest rule: one source frame early ...
    assert np.all(d[(ph > 0.01) & (ph < 0.49)] == 1) and np.all(d[ph > 0.51] == 0)  # ... in the lower half of the phase
    with pytest.raises(SystemExit, match="cut rule"):
        tv.ptz_sample_times(n, 14.0, rule="round")


def test_sym_epi_zero_on_true_correspondences_and_grows_off_line():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 0.5, (50, 3)) + [0, 1, 0]
    Rm, Ro = _look_at(np.array([0, 1.6, 6.0]), np.array([0, 1.0, 0])), _look_at(np.array([3, 1.5, 1.0]), np.array([0, 1.0, 0]))
    tm, to = -Rm @ [0, 1.6, 6.0], -Ro @ [3, 1.5, 1.0]
    xm, xo = X @ Rm.T + tm, X @ Ro.T + to
    xm, xo = xm / xm[:, 2:], xo / xo[:, 2:]
    R_om = (Ro @ Rm.T)[None]
    t_om = (to - Ro @ Rm.T @ tm)[None]
    d = tv.sym_epi(xo[None], xm[None], R_om, t_om, np.array([1000.0]), np.array([1500.0]))
    assert d.max() < 1e-9
    xo2 = xo.copy()
    xo2[:, 1] += 5 / 1000.0  # 5 px off in the other view
    assert np.median(tv.sym_epi(xo2[None], xm[None], R_om, t_om, np.array([1000.0]), np.array([1500.0]))) > 0.5


# ------------------------------------------------------------------------------------------------ sync


def test_sync_recovers_offset_and_drift(synced):
    work, doc = synced
    o = doc["offsets"]["01"]
    assert doc["method"] == "two_view_epipolar" and doc["gauge"] == "camera 02 = 0"
    assert o["delta_ms"] == pytest.approx(37.0, abs=1.0)
    assert o["fit"]["b_ms_per_100s"] == pytest.approx(10.0, abs=2.0)
    assert o["t_ref"] == pytest.approx(40.0, abs=2.0) and o["use_rate"] and o["rate"] == pytest.approx(1e-4, abs=2e-5)
    assert o["stable"] and o["use"] is True and not o["at_search_edge"] and o["n_sub_valid"] == 4
    assert o["delta_ms"] == pytest.approx(o["fit"]["a_ms"], abs=0.01) and o["delta_global_ms"] == pytest.approx(37.0, abs=1.0)
    assert all({"t_mid", "delta_ms", "min_px", "n_frames", "valid"} <= set(s) and "fine" not in s for s in o["sub_windows"])
    assert o["residual_px"]["ratio_at_-40ms"] > 1.15 and o["residual_px"]["ratio_at_+40ms"] > 1.15
    assert o["swap"]["share"] == 0.0 and o["swap"]["frames"] > 1000
    assert doc["offsets"]["02"]["delta"] == 0.0 and doc["offsets"]["02"]["master"]
    # the clocks block exactly as recap_front.fixed_sync_doc builds it: offsets_check / mamma accept the file
    assert doc["clocks"] == {"alignment": (S.take_clocks(work) or {}).get("alignment"), "sources": S.frame_clocks(work)}
    assert S.offsets_check(work)["status"] == "ok"
    offs = md.read_sync_offsets(work)
    assert offs["01"]["delta"] == pytest.approx(0.037, abs=0.001) and offs["01"]["rate"] == pytest.approx(1e-4, abs=2e-5)
    assert (work / "sync" / "curves_2v.json").exists() and (work / "review" / "sync_2v.png").exists()
    assert json.loads((work / "sync" / "curves_2v.json").read_text())["accepted"]


def test_sync_never_writes_use_false(tmp_path):
    """An unstable estimate (a 30 ms jump halfway) exits non-zero and writes no offsets.json."""
    work = make_work(tmp_path / "jump", delta=lambda L: np.where(np.asarray(L) < 40.0, 0.0, 0.030))
    with pytest.raises(SystemExit) as e:
        tv.cmd_sync(work, "02", "01", range_ms=300.0, sub_s=15.0, log=QUIET)
    assert e.value.code == 2
    assert not (work / "sync" / "offsets.json").exists()
    cur = json.loads((work / "sync" / "curves_2v.json").read_text())
    assert not cur["accepted"] and any("unstable" in w for w in cur["why"])


def test_sync_accepts_a_measured_drift_and_applies_it_from_t_ref(tmp_path, monkeypatch):
    """25 ms / 100 s of drift over 6 sub-windows of 10 s (raw spread ~12.5 ms: dancecap.sync's rule alone would call it
    unstable) and 60 % of camera 01's frames unusable after 45 s (the whole-window minimum then sits off t_ref): written
    with use_rate, stable about the line, delta = the line at t_ref; the applied offset follows the truth, the farm
    profile accepts the file, and the jump of test_sync_never_writes_use_false is still refused."""
    from dancecap.farm import ca_profile as cap
    truth = lambda L: 0.037 + 2.5e-4 * (np.asarray(L, float) - 40.0)  # noqa: E731
    work = make_work(tmp_path / "lucas_luiza", delta=truth)
    p = work / "people" / "01.npz"
    z = dict(np.load(p))
    z["kpts"][(z["t"] > 45.0) & (np.random.default_rng(1).random(len(z["t"])) < 0.6), :, :, 2] = 0.3
    np.savez(p, **z)
    doc = tv.cmd_sync(work, "02", "01", range_ms=300.0, sub_s=10.0, log=QUIET)
    o = doc["offsets"]["01"]
    assert o["use_rate"] and o["stable"] and o["stable_on"].startswith("spread about") and o["n_sub_valid"] == 6
    assert o["spread_ms"] > S.STABLE_MS and o["spread_detrended_ms"] < 3.0
    assert o["fit"]["b_ms_per_100s"] == pytest.approx(25.0, abs=1.5) and o["delta_ms"] == pytest.approx(o["fit"]["a_ms"], abs=0.01)
    assert abs(o["delta_global_ms"] - o["delta_ms"]) > 0.3  # the whole-window minimum: at the joints' weighted time
    L = np.linspace(12.0, 68.0, 15)
    err = 1000 * (md.apply_offset(L, md.read_sync_offsets(work)["01"]) - L - truth(L))
    assert np.abs(err).max() < 1.0, err
    monkeypatch.setattr(cap, "WORK_ROOT", tmp_path)
    assert cap.LL.work == work and cap.LL.offsets_doc()["offsets"]["01"]["use_rate"]
    assert cap.LL.offset_ms(cap.LL.offsets_doc(), "01", 60.0) == pytest.approx(1000 * truth(60.0), abs=1.0)


def test_sync_at_search_edge_exits(tmp_path):
    work = make_work(tmp_path / "edge", delta=lambda L: np.full(np.shape(L), 0.120))
    (work / "sync").mkdir(parents=True)
    (work / "sync" / "offsets.json").write_text('{"keep": true}')
    with pytest.raises(SystemExit) as e:
        tv.cmd_sync(work, "02", "01", range_ms=100.0, sub_s=30.0, log=QUIET)
    assert e.value.code == 2
    assert json.loads((work / "sync" / "offsets.json").read_text()) == {"keep": True}  # left as it was
    assert any("search edge" in w for w in json.loads((work / "sync" / "curves_2v.json").read_text())["why"])


# ------------------------------------------------------------------------------------------------ check


def test_check_triangulation_numbers(synced):
    work, _ = synced
    doc = tv.cmd_check(work, CFG, log=QUIET)
    assert doc["master"] == "02" and doc["other"] == "01"
    for c in ("01", "02"):
        assert doc["reproj_px"][c]["median"] < 0.8 and doc["reproj_px"][c]["p90"] < 2.0
    for r, (H, sole) in STAT.items():
        assert doc["coverage"][r]["share"] > 0.99 and not doc["coverage"][r]["windows_below_50pct"]
        P = _template(H, sole)
        assert doc["bones"][r]["thigh_l"]["median_m"] == pytest.approx(np.linalg.norm(P[11] - P[13]), abs=0.005)
        assert doc["bones"][r]["forearm_r"]["median_m"] == pytest.approx(np.linalg.norm(P[8] - P[10]), abs=0.005)
        assert doc["bones"][r]["shoulder_width"]["median_m"] == pytest.approx(0.36, abs=0.005)
        assert doc["bone_cv_max"][r] < 0.03 and doc["bone_lr_max"][r] < 0.01
        assert doc["stature"][r]["ratio"] == pytest.approx(1.0, abs=0.005)
        assert doc["stature"][r]["implied_stature_m"] == pytest.approx(H, abs=0.01)
        a = doc["stance_ankles"][r]
        assert a["left"]["n"] > 50 and abs(a["diff_m"]) < 0.005
    assert doc["stature_lead_follow_ratio"] == pytest.approx(1.0, abs=0.006)
    assert doc["swap"]["share"] == 0.0
    assert doc["cam_to_couple_m"]["02"] == pytest.approx(6.0, abs=0.3)
    assert doc["cam_to_couple_m"]["01"] == pytest.approx(3.0, abs=0.7)
    assert (work / "cameras" / "qa" / "twoview_check.json").exists() and (work / "review" / "twoview_check.png").exists()


def test_check_coverage_counts_every_master_frame(synced):
    """A 12 s loc gap of the master (02) counts as not covered (its frames are not in the pair's rows): the share drops
    by the gap's length, its 10 s window is listed, and so is a window of the TOML window without master frames."""
    work, _ = synced
    z = dict(np.load(work / "cameras" / "02.npz"))
    z["ok"] = z["ok"].copy()
    gap = (z["t"] >= 40.0) & (z["t"] < 52.0)
    z["ok"][gap] = False
    (work / "gapped").mkdir(exist_ok=True)
    np.savez(work / "gapped" / "02.npz", **z)
    doc = tv.cmd_check(work, {**CFG, "window": [10.0, 75.0]}, tracks={"02": "gapped/02.npz"}, tag="gapped", log=QUIET)
    for r in tv.ROLES:
        c = doc["coverage"][r]
        assert c["frames"] == len(z["t"]) and c["frames_used"] == len(z["t"]) - int(gap.sum())
        assert c["share"] == pytest.approx(1 - gap.mean(), abs=0.01)
        low = {w[0]: w[2] for w in c["windows_below_50pct"]}
        assert low == {40.0: 0.0, 70.0: 0.0}  # [50, 60) is 80 % covered; [70, 75] has no master frame
    assert json.loads((work / "cameras" / "qa" / "twoview_check_gapped.json").read_text())["tracks"]["02"].endswith("gapped/02.npz")


def test_role_kp_refuses_roles_out_of_step_with_the_people_frames(tmp_path):
    """dancer_keypoints reads roles.json's frames by position: a shorter or shifted list is refused, not silently used."""
    t = 10 + np.arange(5) / 30.0
    (tmp_path / "people").mkdir()
    np.savez(tmp_path / "people" / "02.npz", t=t, boxes=np.zeros((5, 1, 4)), valid=np.zeros((5, 1), bool),
             kpts=np.zeros((5, 1, 17, 3)))
    (tmp_path / "dancers").mkdir()

    def roles(frames):
        (tmp_path / "dancers" / "roles.json").write_text(json.dumps({"cams": {"02": {"frames": frames}}}))

    roles([{"i": i, "t": float(x)} for i, x in enumerate(t[:4])])
    with pytest.raises(SystemExit, match="4 frames, people/02.npz 5"):
        tv.role_kp(tmp_path, "02")
    roles([{"i": i, "t": float(x) + (0.0334 if i >= 2 else 0.0)} for i, x in enumerate(t)])
    with pytest.raises(SystemExit, match="frame 2 is at t"):
        tv.role_kp(tmp_path, "02")
    roles([{"i": i, "t": float(x) + 4e-5} for i, x in enumerate(t[:3])] + [None, {"i": 4}])  # rounding, gaps: fine
    assert tv.role_kp(tmp_path, "02")[1].shape == (5, 2, 17, 3)


def test_check_with_a_swapped_camera_counts_the_swap(tmp_path):
    work = make_work(tmp_path / "swap")
    tv.cmd_sync(work, "02", "01", sub_s=15.0, log=QUIET)  # on the right labels (a swapped camera does not sync)
    rp = work / "dancers" / "roles.json"
    roles = json.loads(rp.read_text())
    for fr in roles["cams"]["01"]["frames"]:  # 01's roles exchanged in every frame
        fr["lead"], fr["follow"] = fr["follow"], fr["lead"]
    rp.write_text(json.dumps(roles))
    doc = tv.cmd_check(work, CFG, tag="swapped", log=QUIET)
    assert doc["swap"]["share"] > 0.9
    assert (work / "cameras" / "qa" / "twoview_check_swapped.json").exists()


# ------------------------------------------------------------------------------------------------ PTZ


SHIFT = 14.0
Q_TRUE = Rot.from_rotvec([0.4, 1.2, -0.3]).as_matrix()
C_TRUE = np.array([1.0, 2.0, 3.0])
SF_TRUE = 0.93


def _pan(t):
    u = np.asarray(t, float) - SHIFT
    return np.radians(20) * np.sin(2 * np.pi * u / 15), np.radians(3) * np.sin(2 * np.pi * u / 10)


def _R_ptz(t):
    pan, tilt = _pan(t)
    return Rot.from_euler("yx", np.stack([pan, tilt], 1)).as_matrix()


def _f_ptz(t):
    u = np.asarray(t, float) - SHIFT
    return 1800.0 * np.exp(np.log(2070 / 1800) * np.clip((u - 20.0) / 0.5, 0, 1))


def _ptz_files(root: Path, n_frames=900) -> Path:
    n = np.arange(n_frames)
    ts = _fps_filter_times(n, SHIFT)  # the source frame each cut frame holds (the independent model, not the module's)
    rms, inl = np.ones(n_frames), np.full(n_frames, 300)
    rms[100] = 5.0
    final = root / "ptzwork" / "camera_work" / "final"
    final.mkdir(parents=True)
    np.savez(final / "track_s1.0.npz", frame=n, rv=Rot.from_matrix(_R_ptz(ts)).as_rotvec(), f=_f_ptz(ts), rms_px=rms,
             n_inl=inl)
    np.savez(final / "skel_s1.0.npz", frame=n[::15], rv=np.zeros((60, 3)), f=np.full(60, 1800.0),
             lens=np.array([-0.05, 0, 0, 0]), size=np.array([1280, 720]))
    return root / "ptzwork"


def _ptz_take(root: Path) -> tuple[Path, Path, np.ndarray]:
    """work with a loc track of camera 02 = the PTZ truth x Q (SfM frame), its raw loc cache and frames.json."""
    work = root / "work"
    k = np.arange(int(np.ceil((SHIFT + 1) * 60000 / 1001)), int((SHIFT + 28) * 60000 / 1001), 2)
    t = k * 1001 / 60000  # every 2nd frame of the 59.94 fps original
    R = np.einsum("nij,jk->nik", _R_ptz(t), Q_TRUE)
    F = len(t)
    K = np.zeros((F, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = SF_TRUE * _f_ptz(t)
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = 639.5, 359.5, 1.0
    (work / "cameras" / "cache").mkdir(parents=True)
    np.savez(work / "cameras" / "02.npz", t=t, K=K, R=R, tvec=-R @ C_TRUE, dist=np.full(F, -0.04), ok=np.ones(F, bool),
             reproj_px=np.full(F, 0.7))
    rng = np.random.default_rng(5)
    Craw = C_TRUE + rng.normal(0, 0.002, (F, 3))
    np.savez(work / "cameras" / "cache" / "loc_02.npz", t=t, ok=np.ones(F, bool), R=R, tvec=-np.einsum("nij,nj->ni", R, Craw),
             f=K[:, 0, 0], corr_xy=np.zeros((10, 2)))
    (work / "cameras" / "world_from_sfm.json").write_text(json.dumps({"s": 2.5, "R": np.eye(3).tolist(), "t": [0, 0, 0]}))
    (work / "frames" / "02").mkdir(parents=True)
    (work / "frames" / "02" / "frames.json").write_text(json.dumps(
        {"width": 1280, "height": 720, "slope": 1.0, "offset": 0.0,
         "frames": [{"file": f"{i:06d}.jpg", "pts": float(x), "t": float(x)} for i, x in enumerate(t)]}))
    return work, _ptz_files(root), t


def test_ptz_check_uses_the_fps_filter_source_times(tmp_path, monkeypatch):
    """The PTZ truth is timed by the independent fps-filter model: the module's default rule leaves no residual and no
    shift in either half of the cut phase; the nearest rule shows a one-source-frame shift in the lower half only."""
    work, ptzw, t = _ptz_take(tmp_path)
    gap = (SHIFT + 19.0, SHIFT + 22.0)
    doc = tv.cmd_ptz(work, "02", ptzw, SHIFT, "check", zoom_gap=gap, log=QUIET)["check"]
    assert doc["residual_deg"]["median"] < 0.003 and doc["residual_deg"]["p95"] < 0.003
    assert doc["relative"]["median_deg"] < 0.003 and abs(doc["relative"]["best_time_shift_ms"]) < 1.0
    cp = doc["cut_phase"]
    assert cp["rule"] == "fps" and min(cp["phase_below_half"]["frames"], cp["phase_above_half"]["frames"]) > 200
    assert abs(cp["phase_below_half"]["best_time_shift_ms"]) < 1.0 and abs(cp["phase_above_half"]["best_time_shift_ms"]) < 1.0
    assert tv.rot_deg(np.asarray(doc["Q"]).T @ Q_TRUE) < 0.003
    assert doc["focal_ratio"]["median"] == pytest.approx(SF_TRUE, abs=1e-3)
    assert doc["focal_ratio"]["mad_rel_outside_gap"] < 1e-3
    assert doc["focal_levels"]["loc"]["ratio"] == pytest.approx(2070 / 1800, abs=2e-3)
    assert doc["centre"]["robust_sd_m"] == pytest.approx(0.002 * np.sqrt(3) * 2.5, rel=0.3)
    near = tv.cmd_ptz(work, "02", ptzw, SHIFT, "check", rule="nearest", zoom_gap=gap, log=QUIET)["check"]
    cn = near["cut_phase"]
    assert cn["phase_below_half"]["best_time_shift_ms"] == pytest.approx(-1000 * 1001 / 60000, abs=1.5)  # dated 16.7 ms early
    assert abs(cn["phase_above_half"]["best_time_shift_ms"]) < 1.0 and abs(cn["shift_difference_ms"]) > 15.0
    assert near["residual_deg"]["p95"] > 5 * doc["residual_deg"]["p95"]
    # the CLI: --cut-rule nominal (S + n / 30): larger residuals still, recorded in ptz_check.json
    monkeypatch.setattr(tv.md, "take_work", lambda name: {"synthetic": work, "ptzsrc": ptzw}[name])
    monkeypatch.setattr(tv, "take_cfg", lambda name: CFG)
    tv.main(["ptz", "synthetic", "--cam", "02", "--ptz-take", "ptzsrc", "--clock-shift", str(SHIFT), "--mode", "check",
             "--cut-rule", "nominal", "--zoom-gap", f"{gap[0]},{gap[1]}"])
    q = json.loads((work / "cameras" / "qa" / "ptz_check.json").read_text())
    assert q["ptz"]["rule"] == "nominal" and q["check"]["residual_deg"]["p95"] > 5 * doc["residual_deg"]["p95"]


def _map_images(t_all, idx):
    """Synthetic map images of camera 02 at frame times t_all[idx] plus two keyframe times between frames."""
    rng = np.random.default_rng(7)
    t = np.sort(np.concatenate([t_all[idx], t_all[idx[:2]] + 0.0167]))
    R = np.array([Rot.from_rotvec(rng.normal(0, np.radians(0.01), 3)).as_matrix() @ r
                  for r in np.einsum("nij,jk->nik", _R_ptz(t), Q_TRUE)])
    C = C_TRUE + rng.normal(0, 1e-4, (len(t), 3))
    f = SF_TRUE * _f_ptz(t)
    k1 = -0.04 + rng.normal(0, 1e-3, len(t))
    pp = np.tile([639.5, 359.5], (len(t), 1))
    xyz, xy = [], []
    for j in range(len(t)):
        X = C[j] + (R[j].T @ np.array([0, 0, 1.0])) * 8 + rng.normal(0, 1.5, (30, 3))
        uv, _ = tv.reproj(X, np.broadcast_to(R[j], (30, 3, 3)), np.broadcast_to(-R[j] @ C[j], (30, 3)), np.full(30, f[j]),
                          np.broadcast_to(pp[j], (30, 2)), np.full(30, k1[j]))
        xyz.append(X)
        xy.append(uv)
    return {"name": [f"frames/02/{i}.jpg" for i in range(len(t))], "t": t, "R": R, "tvec": -np.einsum("nij,nj->ni", R, C),
            "C": C, "f": f, "k1": k1, "pp": pp, "xy": xy, "xyz": xyz}


def test_ptz_attach_recovers_Q_centre_focal_and_adopts(tmp_path, monkeypatch):
    work, ptzw, t = _ptz_take(tmp_path)
    idx = np.linspace(10, len(t) - 10, 12).astype(int)
    monkeypatch.setattr(tv, "map_images_of", lambda w, c: _map_images(t, idx))
    loc_sha = tv._sha1(work / "cameras" / "02.npz")
    doc = tv.cmd_ptz(work, "02", ptzw, SHIFT, "attach", adopt=True, log=QUIET)
    a = doc["attach"]
    assert a["ok"] and a["images_used"] == 14
    assert tv.rot_deg(np.asarray(a["Q"]).T @ Q_TRUE) < 0.01
    assert np.allclose(a["C"], C_TRUE, atol=2e-4) and a["s_f"] == pytest.approx(SF_TRUE, abs=1e-3)
    assert a["k1"] == pytest.approx(-0.04, abs=2e-3)
    assert a["map_points"]["attached_median_px"] < 0.5 and a["map_points"]["map_median_px"] < 1e-6
    ptz_p = work / "cameras" / "ptz" / "02.npz"  # outside cameras/*.npz: every camera reader globs that
    assert tv.ptz_track_path(work, "02") == ptz_p and Path(a["out"]) == ptz_p
    z = np.load(ptz_p)
    assert str(z["camera_class"]) == "tripod_zoom" and str(z["type_model"]) == "fixed_centre" and not bool(z["attached"])
    good = z["ok"]
    assert good.mean() > 0.98
    assert np.median(tv.rot_deg(np.einsum("nji,njk->nik", z["R"][good], np.einsum("nij,jk->nik", _R_ptz(t[good]), Q_TRUE)))) < 0.01
    assert np.allclose(-np.einsum("nji,nj->ni", z["R"], z["tvec"]), C_TRUE, atol=2e-4)
    assert np.allclose(z["K"][good, 0, 0], SF_TRUE * _f_ptz(t[good]), rtol=2e-3)
    # adoption: the loc track -> orig/02_loc.npz (the marker cameras.pipeline.adopted_track honours), PTZ installed
    from dancecap.cameras.pipeline import adopted_track
    orig = work / "cameras" / "orig" / "02_loc.npz"
    assert adopted_track(work, "02") == orig and tv._sha1(orig) == loc_sha
    assert tv._sha1(work / "cameras" / "02.npz") == tv._sha1(ptz_p)
    assert sorted(p.stem for p in (work / "cameras").glob("*.npz")) == ["02"]
    rec = json.loads((work / "cameras" / "adopted.json").read_text())["02"]
    assert rec["first_adoption"] and rec["loc_track"] == "cameras/orig/02_loc.npz" and rec["from"] == "cameras/ptz/02.npz"
    assert (work / "cameras" / "cache" / "loc_02.npz").exists()  # the loc cache stays for the world / qa stages
    # a second adoption keeps the original loc track; the check then compares against it, not against the PTZ copy
    tv.cmd_ptz(work, "02", ptzw, SHIFT, "attach", adopt=True, log=QUIET)
    assert tv._sha1(orig) == loc_sha
    assert not json.loads((work / "cameras" / "adopted.json").read_text())["02"]["first_adoption"]
    chk = tv.cmd_ptz(work, "02", ptzw, SHIFT, "check", zoom_gap=(SHIFT + 19, SHIFT + 22), log=QUIET)
    assert chk["check"]["loc_track"] == str(orig) and "attach" in chk and "adopted" in chk
    # check --track 02=cameras/02_ptz.npz (the runbook's earlier name) reads the attached track
    assert tv._track_arg(work, "02", "cameras/02_ptz.npz", QUIET) == ptz_p
    assert tv._track_arg(work, "02", "cameras/other.npz", QUIET) == Path("cameras/other.npz")


def test_ptz_attach_refuses_too_few_or_unspread_images(tmp_path, monkeypatch):
    work, ptzw, t = _ptz_take(tmp_path)
    monkeypatch.setattr(tv, "map_images_of", lambda w, c: _map_images(t, np.array([20, 21])))
    with pytest.raises(SystemExit, match="only 4 map images"):
        tv.cmd_ptz(work, "02", ptzw, SHIFT, "attach", log=QUIET)
    monkeypatch.setattr(tv, "map_images_of", lambda w, c: _map_images(t, np.arange(20, 26)))
    with pytest.raises(SystemExit, match="pan range"):
        tv.cmd_ptz(work, "02", ptzw, SHIFT, "attach", adopt=True, log=QUIET)
    assert not tv.ptz_track_path(work, "02").exists() and not (work / "cameras" / "orig").exists()
    assert json.loads((work / "cameras" / "qa" / "ptz_check.json").read_text())["attach"]["ok"] is False


# ------------------------------------------------------------------------------------------------ Atlas


def test_atlas_sim3_rotation_and_non_metric_focal():
    rng = np.random.default_rng(11)
    n = 40
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    Cs = np.stack([2 * np.cos(ang), 0.3 * np.sin(3 * ang), 2 * np.sin(ang)], 1)
    Rs = np.array([_look_at(c, np.zeros(3)) for c in Cs])
    s, R, t = 0.4, Rot.from_rotvec([0.2, -0.8, 0.1]).as_matrix(), np.array([0.5, -1.0, 2.0])
    Ca = (Cs - t) @ R / s  # X_sfm = s R X_atlas + t
    Ca[:4] += rng.normal(0, 1.5, (4, 3))  # outliers
    Ra, Ca2, fa = [], [], []
    for c, r in zip(Ca, Rs @ R):
        cam = A.to_atlas_camera(np.array([[2650.0, 0, 360], [0, 2650.0, 640], [0, 0, 1]]), r, -r @ c, 720, 1280)
        Ka, Rk, tk, _, _ = A.from_atlas_camera(cam)  # the RUB round trip the command reads
        Ra.append(Rk), Ca2.append(-Rk.T @ tk), fa.append(Ka[0, 0])
    res = tv.atlas_compare(np.array(Ra), np.array(Ca2), np.array(fa), Rs, Cs, np.full(n, 1000.0))
    assert res["sim3"]["inliers"] == n - 4 and res["sim3"]["s"] == pytest.approx(s, rel=1e-6)
    assert res["centre_residual_rel"]["median"] < 1e-6 and res["rotation_residual_deg"]["median"] < 1e-4
    assert res["fx_ratio"]["median"] == pytest.approx(2.65) and not res["fx_ratio"]["prior_usable"]


# ------------------------------------------------------------------------------------------------ map gate


def _map_json(work, link_pairs=True, reg02=180):
    mi = [f"01/kf_{i}" for i in range(99)] + [f"01/win_{i}" for i in range(99)] + \
         [f"02/kf_{i}" for i in range(99)] + [f"02/win_{i}" for i in range(99)]
    e = {"pairs": 410, "max_inliers": 820, "sum_inliers": 50000, **({"link_pairs": 37} if link_pairs else {})}
    doc = {"map_images": mi, "models": [{"cams": {"01": 190, **({"02": reg02} if reg02 else {})}}],
           "verification": {"per_campair": {"01-02": e}, "link_min_inliers": 150},
           "strong_link_groups": [["01", "02"]] if reg02 else [["01"], ["02"]], "main_cams": ["01", "02"],
           "refine": {"mean_reproj_after": 0.91}, "intrinsics_mode": {"01": "shared", "02": "per_image"}}
    (work / "cameras" / "sfm_work").mkdir(parents=True, exist_ok=True)
    (work / "cameras" / "sfm_work" / "map.json").write_text(json.dumps(doc))


def test_map_gate(tmp_path):
    _map_json(tmp_path)
    doc = tv.cmd_map_gate(tmp_path, ["01", "02"], log=QUIET)
    assert doc["exit"] == 0 and doc["cams"]["02"]["share"] == pytest.approx(180 / 198, abs=1e-4)
    lk = doc["links"]["01-02"]
    assert lk["link_pairs"] == 37 and lk["strong_link"] and "link_pairs" in lk["source"]
    assert doc["mean_reproj_px"] == pytest.approx(0.91) and doc["cams"]["01"]["intrinsics_mode"] == "shared"
    _map_json(tmp_path, link_pairs=False, reg02=0)
    doc = tv.cmd_map_gate(tmp_path, ["01", "02"], log=QUIET)
    assert doc["exit"] == 3 and doc["missing_from_main_model"] == ["02"]
    assert doc["links"]["01-02"]["link_pairs"] is None and not doc["links"]["01-02"]["strong_link"]
    assert "strong_link_groups" in doc["links"]["01-02"]["source"]
    assert json.loads((tmp_path / "cameras" / "qa" / "map_gate.json").read_text())["exit"] == 3


# ------------------------------------------------------------------------------------------------ gates


SYNC_AT = "2026-10-24T10:00:00"


def _gate_inputs(work: Path) -> None:
    def wj(rel, d):
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d))
    wj("cameras/qa/map_gate.json", {"cams": {"01": {"registered": 190, "map_images": 198, "share": 0.96,
                                                    "focal_px": [1000, 1010, 1020], "intrinsics_mode": "shared"},
                                             "02": {"registered": 180, "map_images": 198, "share": 0.91}},
                                    "links": {"01-02": {"link_pairs": 12, "max_inliers": 600, "link_min_inliers": 150,
                                                        "strong_link": True}}, "mean_reproj_px": 0.9,
                                    "doppelgangers": {"flagged": 0, "per_cam": {}, "kept_majority": {}}})
    wj("cameras/report.json", {"cams": {
        "01": {"camera_type": {"model": "fixed_focal", "check": {"passed": True}}, "height_m": [1.2, 1.4, 1.6],
               "focal": {"zoom_events": []}},
        "02": {"camera_type": {"model": "fixed_centre", "check": {"passed": True}}, "height_m": [1.5, 1.6, 1.7],
               "dist_to_origin_m": 6.2, "focal": {"zoom_events": [{"t0": 125.0, "t1": 126.5, "change_pct": 15.2}]}}},
        "map_mean_reproj_px": 0.9, "gates_G2a": {"world_scale": {"passed": True}, "world_scale_chain": {"passed": True}}})
    for cam, n in (("01", 5705), ("02", 5896)):
        t = np.linspace(16.8, 213.5, n)
        ok = np.ones(n, bool)
        ok[1000:1020] = False
        np.savez(work / "cameras" / f"{cam}.npz", t=t, ok=ok, reproj_px=np.full(n, 0.8), K=np.zeros((n, 3, 3)))
    wj("cameras/qa/ptz_check.json", {"check": {"residual_deg": {"median": 0.05, "p95": 0.2},
                                               "relative": {"median_deg": 0.03, "best_time_shift_ms": 0.1},
                                               "focal_ratio": {"median": 1.02, "mad_rel_outside_gap": 0.01},
                                               "focal_levels": {"loc": {"ratio": 1.15}}, "centre": {"robust_sd_m": 0.01}}})
    wj("cameras/qa/atlas_check.json", {"n_atlas": 99, "keyframes_matched": 99, "n": 99, "sim3": {"inliers": 90},
                                       "centre_residual_rel": {"median": 0.02}, "rotation_residual_deg": {"median": 1.0, "p90": 3.0},
                                       "depth_scale": {"iqr_rel": 0.05}, "fx_ratio": {"median": 2.6, "prior_usable": False}})
    wj("cameras/world_from_sfm.json", {
        "s": 2.0, "R": np.eye(3).tolist(), "t": [0, 0, 0], "x_axis": "camera 02 mean right vector (horizontal)",
        "gravity_source": "architecture vertical", "diagnostics": {
            "vertical": {"adopted": True, "n_cams": 2, "max_cam_deg": 0.8},
            "dancer_roles": {"01": {"source": "roles.json"}, "02": {"source": "roles.json"}},
            "low_ankle_height_m": {"median": 0.10, "expected_ankle_joint": 0.105},
            "room_metric_check": {"horizontal_edges": [{"elevation_vs_world_y_deg": 0.3}],
                                  "walls": [{"tilt_vs_world_y_deg": 0.8}]}}})
    wj("cameras/qa/twoview_check.json", {
        "reproj_px": {"02": {"median": 3.0, "p90": 9.0}, "01": {"median": 4.0, "p90": 12.0}},
        "coverage": {"lead": {"share": 0.85, "windows_below_50pct": []}, "follow": {"share": 0.8, "windows_below_50pct": []}},
        "bone_cv_max": {"lead": 0.05, "follow": 0.06}, "bone_cv_robust_max": {"lead": 0.03, "follow": 0.04},
        "bone_lr_max": {"lead": 0.03, "follow": 0.02},
        "stature": {"lead": {"ratio": 1.01}, "follow": {"ratio": 0.995}}, "stature_lead_follow_ratio": 1.015,
        "swap": {"share": 0.002, "share_any": 0.004}, "cam_to_couple_m": {"02": 6.1, "01": 2.0},
        "stance_ankles": {"lead": {"diff_m": 0.004}, "follow": {"diff_m": -0.006}},
        "offsets_file": {"method": "two_view_epipolar", "created": SYNC_AT}})
    wj("sync/offsets.json", {"method": "two_view_epipolar", "created": SYNC_AT, "offsets": {"01": {
        "delta_ms": -30.0, "at_search_edge": False, "n_sub_valid": 6, "spread_ms": 4.0, "fit": {"b_ms_per_100s": 1.0}}}})
    wj("sync/curves_2v.json", {"accepted": True, "why": [], "doc": {"created": SYNC_AT}, "depth_ratio": {"-40": 2.0, "+40": 1.9}})
    wj("scale/chain_sync.json", {"s_chain": 1.000})
    wj("scale/chain_audio_clocks.json", {"s_chain": 1.004})


def test_gates_table(tmp_path):
    cfg = {**CFG, "window": [16.8, 213.5]}
    _gate_inputs(tmp_path)
    assert tv.cmd_gates(tmp_path, cfg, log=QUIET) == 0
    g = json.loads((tmp_path / "cameras" / "qa" / "twoview_gates.json").read_text())
    for k in ("G1_map", "G2_loc", "G3_types_tripod", "G4_tripod_vs_ptz", "G5_atlas", "G6_world", "G7_dancers", "G8_sync"):
        assert g[k]["pass"], (k, {c: v for c, v in g[k]["checks"].items() if v["status"] != "pass"})
        assert {"value", "threshold", "pass"} <= set(g[k])
    assert g["G3_types_tripod"]["checks"]["focal_steps_115_130"]["value"][0]["change_pct"] == 15.2
    # failures do not change the exit status; soft fails are reported as such
    p = tmp_path / "sync" / "offsets.json"
    d = json.loads(p.read_text())
    d["offsets"]["01"]["fit"]["b_ms_per_100s"] = 5.0
    p.write_text(json.dumps(d))
    p = tmp_path / "cameras" / "qa" / "map_gate.json"
    d = json.loads(p.read_text())
    d["cams"]["02"]["share"] = 0.6
    p.write_text(json.dumps(d))
    assert tv.cmd_gates(tmp_path, cfg, log=QUIET) == 0
    g = json.loads((tmp_path / "cameras" / "qa" / "twoview_gates.json").read_text())
    assert g["G8_sync"]["status"] == "fail" and g["G8_sync"]["checks"]["drift_ms_per_100s"]["status"] == "fail"
    assert g["G1_map"]["status"] == "soft" and not g["G1_map"]["pass"]
    # a map that still holds a doppelganger copy of the room fails G1 (cameras.plausible)
    d["doppelgangers"] = {"flagged": 27, "per_cam": {"01": 27}, "kept_majority": {}}
    p.write_text(json.dumps(d))
    tv.cmd_gates(tmp_path, cfg, log=QUIET)
    g = json.loads((tmp_path / "cameras" / "qa" / "twoview_gates.json").read_text())
    assert g["G1_map"]["status"] == "fail" and g["G1_map"]["checks"]["no_doppelgangers"]["status"] == "fail"
    # a missing input: exit 1, the file is still written and names it
    (tmp_path / "cameras" / "qa" / "atlas_check.json").unlink()
    assert tv.cmd_gates(tmp_path, cfg, log=QUIET) == 1
    g = json.loads((tmp_path / "cameras" / "qa" / "twoview_gates.json").read_text())
    assert g["G5_atlas"]["status"] == "missing" and any("atlas_check" in m for m in g["_meta"]["missing_inputs"])


def _edit_json(work: Path, rel: str, fn) -> None:
    p = work / rel
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d))


def test_gates_edge_cases(tmp_path):
    """Stale sync files, the Atlas inlier count of the 99 cameras, an exact 0 deg gravity, an empty room check, the swap
    share of the runbook (cheaper at all), the plain bone CV, role frames vs people frames, and G3 without report.json's
    camera_type (the npz's type_model)."""
    cfg = {**CFG, "window": [16.8, 213.5]}
    _gate_inputs(tmp_path)
    ev = lambda: tv.gates_eval(tmp_path, cfg, "02", "01")["gates"]  # noqa: E731
    ch = lambda g, gate: g[gate]["checks"]  # noqa: E731
    g = ev()
    assert ch(g, "G8_sync")["offsets_of_last_sync_run"]["status"] == "pass"
    assert ch(g, "G7_dancers")["offsets_of_this_sync"]["status"] == "pass" and ch(g, "G7_dancers")["swap_share"]["value"] == 0.004
    # a rejected re-run: curves_2v.json says so while the previous offsets.json stays
    _edit_json(tmp_path, "sync/curves_2v.json", lambda d: d.update(accepted=False, why=["unstable: ..."],
                                                                    doc={"created": "2026-10-24T12:00:00"}))
    g = ev()
    c = ch(g, "G8_sync")["offsets_of_last_sync_run"]
    assert c["status"] == "fail" and c["value"]["why"] == ["unstable: ..."] and g["G8_sync"]["status"] == "fail"
    # an accepted re-run the check has not seen
    _edit_json(tmp_path, "sync/curves_2v.json", lambda d: d.update(accepted=True, why=[]))
    _edit_json(tmp_path, "sync/offsets.json", lambda d: d.update(created="2026-10-24T12:00:00"))
    g = ev()
    assert ch(g, "G8_sync")["offsets_of_last_sync_run"]["status"] == "pass"
    assert ch(g, "G7_dancers")["offsets_of_this_sync"]["status"] == "fail" and g["G7_dancers"]["status"] == "fail"
    _edit_json(tmp_path, "cameras/qa/twoview_check.json", lambda d: d["offsets_file"].update(created="2026-10-24T12:00:00"))
    # G7: the runbook's swap share (cheaper at all) and the plain CV gate; the margin share / robust CV are shown only
    _edit_json(tmp_path, "cameras/qa/twoview_check.json", lambda d: (d["swap"].update(share=0.0, share_any=0.02),
                                                                      d["bone_cv_max"].update(lead=0.09)))
    g = ev()
    assert ch(g, "G7_dancers")["swap_share"]["status"] == "fail" and ch(g, "G7_dancers")["swap_share_margin"]["status"] == "info"
    assert ch(g, "G7_dancers")["bone_cv_max_lead"]["status"] == "fail"
    rb = ch(g, "G7_dancers")["bone_cv_robust_max_lead"]
    assert rb["status"] == "info" and rb["value"] == 0.03
    # G5: 70 of the 99 Atlas cameras, an unregistered keyframe counting as an outlier (45 / 60 registered would pass 70 / 99)
    _edit_json(tmp_path, "cameras/qa/atlas_check.json", lambda d: (d.update(keyframes_matched=60, n=60,
                                                                            keyframes_unregistered=list(range(60, 99))),
                                                                   d["sim3"].update(inliers=45)))
    g = ev()
    assert ch(g, "G5_atlas")["sim3_inliers"]["status"] == "fail" and ch(g, "G5_atlas")["sim3_inliers"]["value"] == "45/99"
    assert ch(g, "G5_atlas")["keyframes_registered"]["value"] == "60/99"
    # G6: max_cam_deg exactly 0 passes, an empty room check is not a pass, role frames must match the people frames
    _edit_json(tmp_path, "cameras/world_from_sfm.json", lambda d: d["diagnostics"].update(
        vertical={"adopted": True, "n_cams": 2, "max_cam_deg": 0.0}, room_metric_check={"horizontal_edges": [], "walls": []},
        dancer_roles={"01": {"source": "roles.json", "frames": 5705}, "02": {"source": "roles.json", "frames": 5000}}))
    (tmp_path / "people").mkdir()
    for cam, n in (("01", 5705), ("02", 5896)):
        np.savez(tmp_path / "people" / f"{cam}.npz", t=np.linspace(16.8, 213.5, n))
    g = ev()
    assert ch(g, "G6_world")["gravity"]["status"] == "pass" and ch(g, "G6_world")["room_check"]["status"] == "missing"
    assert ch(g, "G6_world")["dancer_roles"]["status"] == "fail"
    assert ch(g, "G6_world")["dancer_roles"]["value"]["02"] == {"source": "roles.json", "frames": 5000, "people_frames": 5896}
    # G3: report.json without camera_type (probe B failed): the tracks' type_model, the loc track's once adopted
    _edit_json(tmp_path, "cameras/report.json", lambda d: [d["cams"][c].pop("camera_type") for c in ("01", "02")])
    for cam, model in (("01", "fixed_focal"), ("02", "per_frame")):
        z = dict(np.load(tmp_path / "cameras" / f"{cam}.npz"))
        np.savez(tmp_path / "cameras" / f"{cam}.npz", **z, type_model=np.str_(model))
    g = ev()
    assert ch(g, "G3_types_tripod")["camera_type_01"]["status"] == "pass"
    assert ch(g, "G3_types_tripod")["camera_type_01"]["value"]["source"] == "cameras/01.npz type_model"
    assert ch(g, "G3_types_tripod")["camera_type_02"]["status"] == "fail"
    (tmp_path / "cameras" / "orig").mkdir()
    z = dict(np.load(tmp_path / "cameras" / "02.npz"))
    np.savez(tmp_path / "cameras" / "orig" / "02_loc.npz", **{**z, "type_model": np.str_("fixed_centre")})
    g = ev()
    assert ch(g, "G3_types_tripod")["camera_type_02"]["status"] == "pass"
    assert ch(g, "G3_types_tripod")["camera_type_02"]["value"]["source"] == "cameras/orig/02_loc.npz type_model"


def test_gates_read_the_files_this_module_writes(synced, tmp_path):
    """Key names: G7 / G8 on the synthetic sync + check outputs, G3 / G4 on a ptz check, G5 on atlas_compare's record."""
    work, _ = synced
    tv.cmd_check(work, CFG, log=QUIET)
    g = tv.gates_eval(work, CFG, "02", "01")["gates"]
    for k in ("G7_dancers", "G8_sync"):
        bad = {c: v["status"] for c, v in g[k]["checks"].items() if v["status"] == "missing"}
        assert set(bad) <= {"chain_sync_vs_audio"}, (k, bad)  # the scale files come from world_scale chain
    assert g["G7_dancers"]["checks"]["swap_share"]["status"] == "pass"
    assert g["G7_dancers"]["checks"]["offsets_of_this_sync"]["status"] == "pass"
    assert g["G7_dancers"]["checks"]["coverage_lead"]["status"] == "pass"
    assert g["G8_sync"]["checks"]["delta_ms_01"]["status"] == "pass" and g["G8_sync"]["checks"]["curve_depth"]["status"] == "pass"
    assert g["G8_sync"]["checks"]["offsets_of_last_sync_run"]["status"] == "pass"
    pw, ptzw, _ = _ptz_take(tmp_path)
    tv.cmd_ptz(pw, "02", ptzw, SHIFT, "check", zoom_gap=(SHIFT + 19, SHIFT + 22), log=QUIET)
    g = tv.gates_eval(pw, CFG, "02", "01")["gates"]
    assert all(v["status"] == "pass" for v in g["G4_tripod_vs_ptz"]["checks"].values() if v["status"] != "info")
    cp = g["G4_tripod_vs_ptz"]["checks"]["cut_phase_shift_ms"]["value"]
    assert cp["rule"] == "fps" and abs(cp["phase_below_half"]) < 1.0 and abs(cp["phase_above_half"]) < 1.0
    assert g["G3_types_tripod"]["checks"]["tripod_centre_sd_m"]["status"] == "pass"


def test_cli_exit_codes(tmp_path, monkeypatch):
    """main(): map-gate exits 3 for a camera outside the main model, gates exits 1 on missing inputs (file written)."""
    monkeypatch.setattr(tv.md, "take_work", lambda name: tmp_path)
    monkeypatch.setattr(tv, "take_cfg", lambda name: CFG)
    _map_json(tmp_path, reg02=0)
    with pytest.raises(SystemExit) as e:
        tv.main(["map-gate", "synthetic", "--need", "01,02"])
    assert e.value.code == 3
    _map_json(tmp_path)
    tv.main(["map-gate", "synthetic", "--need", "01,02"])  # all there: returns normally (exit 0)
    with pytest.raises(SystemExit) as e:
        tv.main(["gates", "synthetic"])
    assert e.value.code == 1 and (tmp_path / "cameras" / "qa" / "twoview_gates.json").exists()
    with pytest.raises(SystemExit, match="missing input"):
        tv.main(["sync", "synthetic", "--master", "02", "--other", "01"])


# ------------------------------------------------------------------------------------------------ pycolmap readers (fake model)


class _FakeRec:
    """The parts of a pycolmap.Reconstruction the module reads: images (name, has_pose, camera_id, cam_from_world(),
    points2D), cameras (params, model.name), points3D (xyz), compute_mean_reprojection_error()."""

    def __init__(self, cams: list[tuple[str, np.ndarray, np.ndarray, list]], params: dict, points: np.ndarray):
        from types import SimpleNamespace as N
        self.points3D = {i: N(xyz=p) for i, p in enumerate(points)}
        self.cameras = {k: N(params=np.asarray(v, float), model=N(name="SIMPLE_RADIAL")) for k, v in params.items()}
        self.images = {}
        for i, (name, R, tvec, obs) in enumerate(cams):
            T = N(rotation=N(matrix=lambda R=R: R), translation=tvec)
            p2 = [N(xy=np.asarray(xy, float), point3D_id=pid, has_point3D=lambda: True) for xy, pid in obs]
            self.images[i] = N(name=name, has_pose=True, camera_id=name.split("/")[1], cam_from_world=lambda T=T: T,
                               points2D=p2)

    def compute_mean_reprojection_error(self):
        return 0.91


def test_map_images_of_reads_poses_times_and_pixel_convention(tmp_path, monkeypatch):
    import pycolmap
    t = 20.0 + np.arange(5) / 29.97
    (tmp_path / "frames" / "02").mkdir(parents=True)
    (tmp_path / "frames" / "02" / "frames.json").write_text(json.dumps(
        {"frames": [{"file": f"{i:06d}.jpg", "t": float(x)} for i, x in enumerate(t)]}))
    (tmp_path / "cameras" / "sfm").mkdir(parents=True)
    R = _look_at(np.array([0, 1.6, 6.0]), np.zeros(3))
    tvec = -R @ np.array([0, 1.6, 6.0])
    X = np.array([[0.2, 1.0, 0.0], [-0.5, 0.3, 0.4]])
    uv, _ = tv.reproj(X, np.broadcast_to(R, (2, 3, 3)), np.broadcast_to(tvec, (2, 3)), np.full(2, 1500.0),
                      np.full((2, 2), [639.5, 359.5]), np.full(2, -0.05))
    rec = _FakeRec([("frames/02/000003.jpg", R, tvec, [(uv[0] + 0.5, 0), (uv[1] + 0.5, 1)]),
                    ("frames/01/000003.jpg", R, tvec, [])], {"02": [1500.0, 640.0, 360.0, -0.05], "01": [900, 360, 640, 0]}, X)
    monkeypatch.setattr(pycolmap, "Reconstruction", lambda p: rec)
    img = tv.map_images_of(tmp_path, "02")
    assert img["name"] == ["frames/02/000003.jpg"] and img["t"][0] == pytest.approx(t[3])
    assert np.allclose(img["C"][0], [0, 1.6, 6.0]) and img["f"][0] == 1500.0 and img["k1"][0] == -0.05
    assert np.allclose(img["pp"][0], [639.5, 359.5]) and np.allclose(img["xy"][0], uv)  # COLMAP corner -> OpenCV centre
    arr = {"ok": np.ones(5, bool), "R": np.broadcast_to(R, (5, 3, 3)), "tvec": np.broadcast_to(tvec, (5, 3)),
           "K": np.broadcast_to(np.array([[1500.0, 0, 639.5], [0, 1500.0, 359.5], [0, 0, 1]]), (5, 3, 3)),
           "dist": np.full(5, -0.05)}
    mp = tv.map_points_px(img, arr, t)
    assert mp["images"] == 1 and mp["attached_median_px"] < 1e-6 and mp["map_median_px"] < 1e-6
    # map-gate reads registrations and focal from the model when cameras/sfm is there
    _map_json(tmp_path)
    (tmp_path / "cameras" / "sfm" / "images.bin").write_bytes(b"")
    doc = tv.cmd_map_gate(tmp_path, ["01", "02"], log=QUIET)
    assert doc["registered_from"] == "cameras/sfm" and doc["cams"]["02"]["registered"] == 1
    assert doc["cams"]["02"]["focal_px"][1] == 1500.0 and doc["cams"]["01"]["colmap_cameras"] == 1
    assert doc["mean_reproj_px"] == 0.91  # the written model's, not map.json's pre-clean refine value
    assert doc["doppelgangers"]["flagged"] == 0


def test_cmd_atlas_end_to_end_non_metric_scale(tmp_path, monkeypatch):
    """Atlas frame = the world scaled by 0.8 and turned about Y: the check recovers the Sim3, reports 0.8 m per Atlas unit
    (a drift-free but non-metric scale), heights that agree and Atlas's up = world Y; people pixels are not used."""
    import cv2
    import pycolmap
    work = tmp_path
    n, rng = 24, np.random.default_rng(4)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    Cw = np.stack([2.2 * np.cos(ang), 1.4 + 0.1 * np.sin(2 * ang), 2.2 * np.sin(ang)], 1)
    Rw = np.array([_look_at(c, np.array([0, 1.0, 0])) for c in Cw])
    Rs, ts = md.sfm_cameras(Rw, -np.einsum("nij,nj->ni", Rw, Cw), SIM3)
    a, Ry, taw = 0.8, Rot.from_euler("y", 40, degrees=True).as_matrix(), np.array([0.3, -0.2, 1.0])  # X_w = a Ry X_a + taw
    room_w = np.concatenate([rng.uniform([-5, 0, -5], [5, 3, 5], (1500, 3))])
    room_s = (room_w - np.asarray(SIM3["t"])) @ np.asarray(SIM3["R"]) / SIM3["s"]
    kfs, cams, rows = [], [], []
    K = np.array([[1000.0, 0, 359.5], [0, 1000.0, 639.5], [0, 0, 1]])
    (work / "atlas").mkdir()
    (work / "people_masks_kf" / "01").mkdir(parents=True)
    for i in range(n):
        Ra, Ca = Rw[i] @ Ry, Ry.T @ (Cw[i] - taw) / a
        cam = A.to_atlas_camera(np.array([[2650.0, 0, 360], [0, 2650.0, 641.25], [0, 0, 1]]), Ra, -Ra @ Ca, 720, 1280)
        f = f"{i:07d}.jpg"
        kfs.append({"file": f, "pts": float(i), "t": 16.8 + 2 * i})
        cams.append(cam)
        rows.append({"kf": i, "t": 16.8 + 2 * i, "f": 2650.0, "W": 720, "H": 1280, "C": Ca.tolist(), "zmed": float("nan")})
        Xc = room_s @ Rs[i].T + ts[i]
        u, v = K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]
        ok = (Xc[:, 2] > 0.3) & (u >= 0) & (u < 719) & (v >= 0) & (v < 1279)
        z = np.full((1280, 720), np.nan, np.float32)
        z[v[ok].astype(int), u[ok].astype(int)] = Xc[ok, 2] * SIM3["s"] / a  # Atlas depth = world depth / 0.8
        z[600:700, 300:400] = 1.0  # a wrong depth on a "person": masked out below
        np.savez(work / "atlas" / f"view_01_{i:02d}.npz", z=z, conf=np.ones_like(z))
        m = np.zeros((1280, 720), np.uint8)
        m[600:700, 300:400] = 255
        cv2.imwrite(str(work / "people_masks_kf" / "01" / f"{Path(f).stem}.png"), m)
    (work / "atlas" / "pose_01_24.json").write_text(json.dumps({"rows": rows, "cameras": cams, "meta": {}}))
    (work / "keyframes" / "01").mkdir(parents=True)
    (work / "keyframes" / "01" / "keyframes.json").write_text(json.dumps(kfs))
    (work / "frames" / "01").mkdir(parents=True)
    (work / "frames" / "01" / "frames.json").write_text(json.dumps({"width": 720, "height": 1280, "frames": []}))
    (work / "cameras" / "sfm").mkdir(parents=True)
    (work / "cameras" / "world_from_sfm.json").write_text(json.dumps(SIM3))
    rec = _FakeRec([(f"keyframes/01/{k['file']}", Rs[i], ts[i], []) for i, k in enumerate(kfs)],
                   {"01": [1000.0, 360.0, 640.0, 0.0]}, room_s)
    monkeypatch.setattr(pycolmap, "Reconstruction", lambda p: rec)
    doc = tv.cmd_atlas(work, "01", work / "atlas" / "pose_01_24.json", log=QUIET)
    assert doc["keyframes_matched"] == n and doc["sim3"]["inliers"] == n
    assert doc["centre_residual_rel"]["median"] < 1e-6 and doc["rotation_residual_deg"]["median"] < 1e-3
    assert doc["fx_ratio"]["median"] == pytest.approx(2.65) and not doc["fx_ratio"]["prior_usable"]
    ds = doc["depth_scale"]
    assert ds["keyframes"] == n and ds["m_per_atlas_unit_median"] == pytest.approx(a, rel=0.01)
    assert ds["iqr_rel"] < 0.01 and "NOT metric" in ds["note"]
    h = doc["heights_world_m"]
    assert abs(h["atlas_minus_sfm_median"]) < 1e-6 and h["atlas_up_vs_world_y_deg"] < 1e-4
    assert (work / "cameras" / "qa" / "atlas_check.json").exists() and (work / "review" / "atlas_check.png").exists()


def test_undistort_matches_world_undistort_pts():
    K = np.array([[1500.0, 0, 639.5], [0, 1500.0, 359.5], [0, 0, 1]])
    xy = np.array([[10.0, 20.0], [1200.0, 700.0], [640.0, 360.0]])
    ref = W.undistort_pts(xy, K, -0.05)
    got = tv.undistort(xy[None], K[None], np.array([-0.05]))[0]
    assert np.allclose(got, ref, atol=1e-12)
