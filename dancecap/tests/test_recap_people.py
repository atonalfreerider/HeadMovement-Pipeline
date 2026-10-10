"""Unit tests for dancecap.recap_people (synthetic skeletons and cameras; no footage, no pod, no GPU).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_recap_people.py -q      (from the workspace root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import recap_people as rp  # noqa: E402


# ------------------------------------------------------------------------------------------------ synthetic world


def skeleton(x: float, z: float, scale: float = 1.0, y0: float = 0.0) -> np.ndarray:
    """A standing COCO-17 person at (x, z) in the Y-up world (metres)."""
    J = {0: (0, 1.62, 0.05), 1: (0.03, 1.66, 0.05), 2: (-0.03, 1.66, 0.05), 3: (0.08, 1.64, 0), 4: (-0.08, 1.64, 0),
         5: (0.19, 1.45, 0), 6: (-0.19, 1.45, 0), 7: (0.22, 1.15, 0), 8: (-0.22, 1.15, 0), 9: (0.22, 0.90, 0.05), 10: (-0.22, 0.90, 0.05),
         11: (0.10, 0.95, 0), 12: (-0.10, 0.95, 0), 13: (0.10, 0.50, 0), 14: (-0.10, 0.50, 0), 15: (0.10, 0.08, 0), 16: (-0.10, 0.08, 0)}
    X = np.array([J[i] for i in range(17)], float) * scale
    X[:, 1] += y0
    X[:, 0] += x
    X[:, 2] += z
    return X


def look_at(C, target=(0.0, 1.0, 0.0)):
    """R (world -> camera), tvec of a camera at C looking at target (OpenCV: +Z forward, +Y down)."""
    C = np.asarray(C, float)
    f = np.asarray(target, float) - C
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return R, -R @ C


def make_cam(cid, C, persons_world, f=900.0, w=720, h=1280, k1=0.0, order=None, attached=False, conf=0.9, t=(0.0,)):
    """A camera with F frames (all the same people) and the pose table of its projections (person order optionally permuted)."""
    R, tv = look_at(C)
    K = np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1.0]])
    F = len(t)
    P = len(persons_world)
    order = list(order) if order is not None else list(range(P))
    kp = np.full((F, P, 17, 3), np.nan, np.float32)
    for i in range(F):
        for j, src in enumerate(order):
            uv, z = rp.project(persons_world[src], R, tv, K, k1)
            kp[i, j, :, :2] = uv
            kp[i, j, :, 2] = conf
    n = np.full(F, P, np.int16)
    return rp.Cam(cid, np.array(t, float), np.ones(F, bool), np.arange(F), np.repeat(K[None], F, 0), np.repeat(R[None], F, 0), np.repeat(tv[None], F, 0),
                  np.repeat(np.asarray(C, float)[None], F, 0), np.full(F, k1), (w, h), attached, np.array(t, float), kp, n)


PEOPLE = [skeleton(0.0, 0.0), skeleton(1.0, -0.4), skeleton(-3.0, 4.0)]
CAM_POS = {"00": (0.0, 1.5, 6.0), "01": (5.0, 1.6, 4.0), "02": (-5.0, 1.4, 3.0), "03": (3.0, 1.7, -5.0)}


def four_cams(order_by_cam=None, **kw):
    order_by_cam = order_by_cam or {}
    return {c: make_cam(c, p, PEOPLE, order=order_by_cam.get(c), **kw) for c, p in CAM_POS.items()}


# ------------------------------------------------------------------------------------------------ geometry


def test_triangulate_views_recovers_joints_and_drops_an_outlier_view():
    X = PEOPLE[0]
    U, ok, Pm, f = [], [], [], []
    for c, p in CAM_POS.items():
        R, tv = look_at(p)
        K = np.array([[900.0, 0, 359.5], [0, 900.0, 639.5], [0, 0, 1.0]])
        uv, _ = rp.project(X, R, tv, K, 0.0)
        U.append(rp.undistort_norm(uv, K, 0.0))
        ok.append(np.ones(17, bool))
        Pm.append(np.hstack([R, tv[:, None]]))
        f.append(900.0)
    U, ok, Pm, f = np.stack(U), np.stack(ok), np.stack(Pm), np.array(f)
    Xe, nv, res = rp.triangulate_views(U, ok, Pm, f)
    assert np.nanmax(np.abs(Xe - X)) < 1e-6 and (nv == 4).all() and res < 1e-3
    U2 = U.copy()
    U2[2, 9] += 0.08                                         # one joint of one view 72 px off: that view is dropped for that joint only
    Xe2, nv2, _ = rp.triangulate_views(U2, ok, Pm, f)
    assert np.linalg.norm(Xe2[9] - X[9]) < 0.01 and nv2[9] == 3 and nv2[8] == 4


def test_instant_candidates_match_people_across_cameras_whatever_their_order():
    cams = four_cams(order_by_cam={"00": [2, 0, 1], "01": [1, 2, 0], "02": [0, 2, 1], "03": [2, 1, 0]})
    near = {c: np.array([0]) for c in cams}
    cands = rp.instant_candidates(cams, near, 0)
    assert len(cands) == 3
    truth = np.array([rp.centre(p) for p in PEOPLE])
    for c in cands:
        d = np.linalg.norm(truth - rp.centre(c.X), axis=1)
        k = int(np.argmin(d))
        assert d[k] < 0.02 and set(c.views) == set(cams)
        for cam_id, (i, row) in c.views.items():          # the row of each camera really is that person: its joints are the truth's projection
            R, tv = cams[cam_id].R[i], cams[cam_id].tvec[i]
            uv, _ = rp.project(PEOPLE[k], R, tv, cams[cam_id].K[i], 0.0)
            assert np.abs(cams[cam_id].persons(i)[row][:, :2] - uv).max() < 1e-3


def test_cluster_never_puts_two_people_of_one_camera_together():
    cams = four_cams()
    views = {c: rp.make_view(cam, 0) for c, cam in cams.items()}
    for cl in rp.cluster_views(views):
        assert len(cl) == len(set(cl))                        # dict: one row per camera by construction
    sizes = sorted(len(cl) for cl in rp.cluster_views(views))
    assert sizes == [4, 4, 4]


def test_bone_score_flags_a_person_assembled_from_two():
    good = skeleton(0, 0)
    assert rp.bone_score(good) == 1.0
    bad = good.copy()
    bad[[9, 10, 15, 16]] += np.array([[1.5, 0, 0]] * 4)      # wrists and ankles of somebody else
    assert rp.bone_score(bad) < 0.7


# ------------------------------------------------------------------------------------------------ the pair Viterbi


def synthetic_world(J=60, miss=(30, 36)):
    """Two teachers walking in the dance zone (A left, B right), a seated spectator 3 m away, a standing spectator 4 m away; B is not
    triangulated at instants miss[0]..miss[1]-1."""
    X, nv, res, fi, row, off = [], [], [], [], [], [0]
    for j in range(J):
        a = skeleton(-0.4 + 0.01 * j, 0.0)
        b = skeleton(0.5 + 0.01 * j, 0.2)
        seat = skeleton(-3.0, -1.0, scale=0.55)                # shoulders at 0.8 m, hips at 0.52: seated-like
        far = skeleton(3.8, -2.5)
        for who, p in (("a", a), ("seat", seat), ("far", far), ("b", b)):
            if who == "b" and miss[0] <= j < miss[1]:
                continue
            X.append(p)
            nv.append(np.full(17, 3 if who in "ab" else 2))
            res.append(1.5)
            fi.append([0, 0])
            row.append([0, 0])
        off.append(len(X))
    return rp.World(np.arange(J), np.array(off), np.array(X, np.float32), np.array(nv, np.int8), np.array(res, np.float32),
                    np.array(fi, np.int32), np.array(row, np.int16), ["00", "01"])


def test_pair_viterbi_follows_the_two_teachers_and_ignores_spectators():
    W = synthetic_world()
    f = rp.cand_features(W)
    r = rp.pair_viterbi(W, rp.TeacherCfg(centre0=(0.0, 0.0)), dt=1 / 30)
    idx = r["idx"]
    cen = f["cen"]
    sel_x = np.where(idx >= 0, cen[np.maximum(idx, 0)][..., 0], np.nan)           # (J,2)
    # a slot keeps ONE teacher the whole time: its x never jumps between the teachers (0.9 m apart)
    for s in range(2):
        v = sel_x[:, s][np.isfinite(sel_x[:, s])]
        assert np.abs(np.diff(v)).max() < 0.2
    assert set(np.round(np.nanmedian(sel_x, axis=0), 1)) == {-0.1, 0.8} or abs(np.nanmedian(sel_x[:, 0]) - np.nanmedian(sel_x[:, 1])) > 0.8
    chosen_z = cen[idx[idx >= 0]][:, 2]
    assert (np.abs(cen[idx[idx >= 0]][:, 0]) < 1.5).all() and (chosen_z > -1.0).all()        # never the seated (-3, -1) or the far (3.8, -2.5) person
    gap = (idx[30:36] < 0).sum(1)
    assert (gap >= 1).all() and ((idx[:30] >= 0).all(1)).all()                         # B is missing in the gap, both present before it


def test_track3d_at_interpolates_only_across_short_gaps():
    X = np.full((10, 2, 17, 3), np.nan, np.float32)
    X[0, 0] = skeleton(0, 0)
    X[5, 0] = skeleton(1, 0)
    tr = rp.Track3D(np.arange(10) * 30, np.zeros((10, 2), np.int32), X, np.zeros((10, 2), np.int8), np.zeros((10, 2), np.float32), (0, 0))   # instants 1 s apart
    got, kind = tr.at(0, 0.0)
    assert kind == "observed" and np.allclose(got, X[0, 0])
    got, kind = tr.at(0, 2.5)
    assert kind == "none"                                       # the neighbours are 5 s apart: more than 2 x max_gap
    X[1, 0] = skeleton(0.5, 0)
    X[3, 0] = skeleton(0.7, 0)
    tr.X = X
    got, kind = tr.at(0, 2.0)                                       # neighbours at 1 s and 3 s, each within max_gap_s (1 s) of the query
    assert kind == "interp" and abs(rp.centre(got)[0] - 0.6) < 1e-4
    assert tr.at(1, 0.0)[1] == "none"


# ------------------------------------------------------------------------------------------------ association


def test_associate_matches_the_two_projected_tracks_and_rejects_the_rest():
    cam = make_cam("00", CAM_POS["00"], PEOPLE, order=[2, 1, 0], t=(0.0, 0.1))
    X = np.full((2, 2, 17, 3), np.nan, np.float32)
    X[:, 0] = PEOPLE[0]
    X[:, 1] = PEOPLE[1]
    tr = rp.Track3D(np.array([0, 3]), np.zeros((2, 2), np.int32), X, np.zeros((2, 2), np.int8), np.zeros((2, 2), np.float32), (0, 0))
    A = rp.associate(cam, tr)
    # order [2,1,0]: row 0 = far spectator, row 1 = teacher B (PEOPLE[1]), row 2 = teacher A (PEOPLE[0])
    assert (A.row[:, 0] == 2).all() and (A.row[:, 1] == 1).all()
    assert (A.cost[A.row >= 0] < 0.02).all() and (A.kind[:, 0] > 0).all()


def test_associate_does_not_give_one_person_to_both_slots():
    cam = make_cam("00", CAM_POS["00"], PEOPLE[:1], t=(0.0,))
    X = np.full((1, 2, 17, 3), np.nan, np.float32)
    X[:, 0] = PEOPLE[0]
    X[:, 1] = PEOPLE[0] + np.array([0.05, 0, 0])
    tr = rp.Track3D(np.array([0]), np.zeros((1, 2), np.int32), X, np.zeros((1, 2), np.int8), np.zeros((1, 2), np.float32), (0, 0))
    A = rp.associate(cam, tr)
    assert (A.row[0] >= 0).sum() == 1


# ------------------------------------------------------------------------------------------------ roles


def desc_vec(torso_v, leg_v, red, n=500):
    d = np.full(len(rp.DESC_KEYS), np.nan, np.float32)
    for k, v in (("torso_v", torso_v), ("leg_v", leg_v), ("red", red), ("torso_n", n), ("legs_n", n), ("head_n", n)):
        d[rp.DESC_KEYS.index(k)] = v
    return d


def test_colour_evidence_prefers_the_black_top_and_no_red_for_the_lead():
    desc = np.stack([np.stack([desc_vec(30, 90, 0.0), desc_vec(180, 150, 0.5)]), np.stack([desc_vec(170, 150, 0.45), desc_vec(40, 85, 0.0)])])
    feats = rp._feat_matrix(desc)
    llr = rp.llr_obs(feats, rp.CLASS0)
    assert llr[0, 0] > 1 and llr[0, 1] < -1 and llr[1, 0] < -1 and llr[1, 1] > 1
    few = desc.copy()
    few[:, :, rp.DESC_KEYS.index("torso_n")] = 10                   # too few pixels: the torso cue abstains, the others still speak
    assert abs(rp.llr_obs(rp._feat_matrix(few), rp.CLASS0)[0, 0]) > 0.5


def test_role_hmm_switches_once_on_lasting_evidence_not_on_a_blip_and_not_across_a_gap_between_the_dancers():
    e = np.r_[np.full(50, 4.0), np.full(50, -4.0)]
    st, p = rp.role_hmm(e, np.full(100, 6.0))
    assert list(np.flatnonzero(np.diff(st))) == [49] and p[10] > 0.99 and p[90] < 0.01
    blip = np.full(100, 4.0)
    blip[40] = -4.0
    st, _ = rp.role_hmm(blip, np.full(100, 6.0))
    assert st.sum() == 0
    st, _ = rp.role_hmm(e, np.full(100, 25.0))                         # far apart: a swap would be a teleport, 4 x 50 bins of evidence still wins
    assert np.flatnonzero(np.diff(st)).size == 1
    weak = np.r_[np.full(50, 4.0), np.full(2, -4.0), np.full(48, 4.0)]
    st, _ = rp.role_hmm(weak, np.full(100, 25.0))
    assert st.sum() == 0


def test_teacher_conf_orders_the_ways_of_finding_a_person():
    cost = np.array([[0.02, 0.02, 0.02], [0.02, 0.2, np.nan]])
    kind = np.array([[1, 2, 3], [1, 1, 0]])
    nj = np.full((2, 3), 14)
    c = rp.teacher_conf(cost, kind, nj, np.ones((2, 3)), attached=False)
    assert c[0, 0] > c[0, 1] > c[0, 2] > 0.5 and c[1, 1] < 0.3 and c[1, 2] == 0.0
    ca = rp.teacher_conf(cost, kind, nj, np.ones((2, 3)), attached=True)
    assert (ca[0] > 0.5).all() and ca[1, 2] == 0.0 and (ca[0] > 0.6 * c[0]).all() and ca[1, 1] < 0.4


def test_coverage_counts_cameras_per_second_and_role():
    t = np.arange(0, 3, 1 / 30.0)
    mk = lambda lead, follow: {"t": t, "ok": np.ones(len(t), bool), "conf": np.stack([np.full(len(t), lead), np.full(len(t), follow)], 1), "attached": np.array(False)}   # noqa: E731
    cov = rp.coverage_from({"01": mk(0.9, 0.9), "02": mk(0.9, 0.1)}, {}, seconds=3)
    assert cov["lead"][0] == ["01", "02"] and cov["follow"][0] == ["01"] and cov["both"][0] == ["01"]
    assert cov["per_camera"]["02"]["follow"] == 0.0 and cov["both_cameras_per_second"]["1"] == 3


def test_kde_density_peaks_where_the_points_are():
    rng = np.random.default_rng(0)
    pts = rng.normal([0.3, -0.5], 0.3, (2000, 2))
    f = rp.kde_density(pts)
    assert f(0.3, -0.5) > 0.9 and f(4.0, 4.0) < 0.01 and f(20.0, 0.0) == 0.0


def test_person_box_and_height_use_only_confident_joints():
    p = skeleton(0, 0)[:, :2] * 100 + 500
    P = np.c_[p, np.full(17, 0.9)]
    P[16, 2] = 0.05
    b = rp.person_box(P)
    assert b[0] < P[:, 0].min() and b[1] < P[:, 1].min() and np.isfinite(b).all()
    assert rp.person_height_px(P) == pytest.approx(max(np.ptp(P[:16, 0]), np.ptp(P[:16, 1])))
