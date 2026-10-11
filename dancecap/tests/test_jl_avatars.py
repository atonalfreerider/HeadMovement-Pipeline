"""Pure parts of the john_lula avatar graph (dancecap.jl_avatars, dancecap.jl_avatars_pod): zoom-step gates, facing from the shoulders, the face-candidate
gate, the mask check, foot boxes, window choice, the trust shrinkage, the pod job's dedicated kind. Synthetic data only (no portraits, no videos, no pods)."""
from __future__ import annotations

import json

import numpy as np

from dancecap import ca_avatars as ca
from dancecap import jl_avatars as jl
from dancecap import jl_avatars_pod as jp


def _body(n, facing=1.0, iod=20.0, off=0.0, score=0.9):
    """kp (n,17,3): a dancer facing the camera (facing = +1) or away (-1): shoulders 60 px apart, torso 60 px."""
    kp = np.zeros((n, 17, 3))
    kp[:, :, 2] = score
    kp[:, 1, 0], kp[:, 2, 0] = 100 - iod / 2, 100 + iod / 2
    kp[:, 1, 1] = kp[:, 2, 1] = 50
    kp[:, 0, 0], kp[:, 0, 1] = 100 + off * iod, 60
    kp[:, 5, 0], kp[:, 6, 0] = 100 + 30 * facing, 100 - 30 * facing          # left shoulder on the image's right when facing the camera
    kp[:, 5, 1] = kp[:, 6, 1] = 80
    kp[:, 11, 0], kp[:, 12, 0] = 100 + 20 * facing, 100 - 20 * facing
    kp[:, 11, 1] = kp[:, 12, 1] = 140
    return kp


def _people(n=40, **over):
    t = np.arange(n) / 30.0
    kp = np.stack([_body(n), _body(n, facing=-1)], 1)                          # dancer 0 faces the camera, dancer 1 is turned away
    wb = np.zeros((n, 2, 133, 3))
    wb[:, :, 23:91, 2] = 0.9
    d = {"t": t, "kp": kp, "kp_wb": wb, "conf": np.ones((n, 2)), "role_conf": np.ones((n, 2)), "merged": np.zeros((n, 2), bool),
         "skel_in_partner": np.zeros((n, 2)), "box_iou": np.zeros(n)}
    d.update(over)
    return d


def test_zoom_step_gate_has_a_margin():
    t = np.arange(0, 20, 1.0)
    z = jl.in_zoom_step(t, [(10.0, 10.7)], 0.4)
    assert list(np.nonzero(z)[0]) == [10, 11]                 # 9.6 .. 11.1
    assert jl.in_zoom_step(t, [(10.0, 10.7)], 1.2)[[9, 10, 11]].all()


def test_shoulder_order_says_front_or_back():
    ff = jl.facing_flags(np.concatenate([_body(3, 1.0), _body(3, -1.0)]), rho_ref=1.0)
    assert ff["front"][:3].all() and not ff["front"][3:].any()
    assert ff["back"][3:].all() and not ff["back"][:3].any()
    assert np.allclose(ff["cos"][:3], 1.0, atol=0.01)


def test_rho_reference_is_the_square_on_value_not_the_mean():
    rho = np.r_[np.full(10, 1.0), np.full(10, 0.2), np.nan]
    assert abs(jl.rho_reference(rho) - 1.0) < 1e-9


def test_facing_angle_and_side_groups():
    kp = np.concatenate([_body(1, 1.0), _body(1, -1.0)])
    kp[0, 0, 0] = 100.0                                         # nose in the middle
    wb = np.zeros((2, 133, 3))
    psi = jl.facing_angle(kp, wb, rho_ref=1.0)
    g = jl.side_group(psi)
    assert g[0] == 0 and g[1] == 2                              # front, back
    # a dancer facing the image's right: shoulders foreshortened to 0, the nose points right
    k = _body(1, 0.0)
    k[0, 5, 0], k[0, 6, 0] = 100.0, 100.0
    k[0, 0, 0] = 120.0
    p = jl.facing_angle(k, np.zeros((1, 133, 3)), rho_ref=1.0)
    assert 60 < np.degrees(p[0]) < 120 and jl.side_group(p)[0] == 1
    k[0, 0, 0] = 80.0
    assert jl.side_group(jl.facing_angle(k, np.zeros((1, 133, 3)), rho_ref=1.0))[0] == 3
    assert jl.side_group(np.array([np.nan]))[0] == -1


def test_candidate_gate_demands_front_size_sharpness_and_no_zoom_step():
    p = _people(40)
    g0 = jl.candidate_gate(p, 0, [])
    g1 = jl.candidate_gate(p, 1, [])
    assert g0["passing"]["all"] > 30 and g1["passing"]["all"] == 0       # the dancer turned away never scores
    assert (g0["score"][1:-1] > 0).all()
    z = jl.candidate_gate(p, 0, [(0.5, 0.6)], zoom_margin=0.2)
    assert (z["score"][14:20] == 0).all() and z["score"][5] > 0
    small = _people(40)
    small["kp"] = small["kp"].copy()
    small["kp"][:, 0] = _body(40, iod=4.0)
    assert jl.candidate_gate(small, 0, [])["passing"]["all"] == 0          # 4 px between the eyes: too small
    low = _people(40, role_conf=np.full((40, 2), 0.6))
    assert jl.candidate_gate(low, 0, [])["passing"]["all"] == 0            # identity not confident
    merged = _people(40, merged=np.ones((40, 2), bool))
    assert jl.candidate_gate(merged, 0, [])["passing"]["all"] == 0


def test_candidate_gate_drops_fast_heads():
    p = _people(40)
    kp = p["kp"].copy()
    kp[:, 0, 0:3, 0] += np.arange(40)[:, None] * 12.0                       # nose and eyes run 12 px / frame = 0.6 eye distances
    p["kp"] = kp
    assert jl.candidate_gate(p, 0, [], max_speed=0.3)["passing"]["all"] == 0
    assert jl.candidate_gate(p, 0, [], max_speed=0.9)["passing"]["all"] > 0


def test_mask_face_check_own_and_partner():
    own = np.zeros((100, 100), bool)
    own[20:80, 20:80] = True
    par = np.zeros((100, 100), bool)
    par[:, 70:] = True
    a, b = jl.mask_face_check(own, par, [50, 50, 55.0])
    assert a == 1.0 and b == 0.0
    a, b = jl.mask_face_check(own, par, [90, 50, 40.0])
    assert a == 0.0 and b > 0.9
    assert jl.mask_face_check(None, par, [50, 50, 55.0]) == (0.0, 0.0)
    assert jl.mask_face_check(own, par, [500, 500, 20.0]) == (0.0, 0.0)      # outside the image


def test_foot_box_and_strata():
    foot = np.array([[100.0, 200, 0.9], [96, 203, 0.9], [80, 201, 0.9]])      # big toe, small toe, heel
    x0, y0, x1, y1, L = jl.foot_box(foot)
    assert abs(L - 20.02) < 0.1 and x0 < 80 and x1 > 100 and y0 < 200 - 0.35 * L and y1 > 203
    bad = foot.copy()
    bad[1, 2] = 0.1
    assert jl.foot_box(bad) is None
    tiny = np.array([[10.0, 10, 0.9], [11, 10, 0.9], [9, 10, 0.9]])
    assert jl.foot_box(tiny) is None
    rows = np.arange(100.0)
    s = jl.split_strata(rows, 0.3)
    assert s.sum() == 30 and s[-1] and not s[0]


def test_not_floor_distance():
    px = np.array([[200.0, 190, 180], [60, 60, 70], [215, 160, 170]])
    keep = jl.not_floor(px, [205, 195, 185], 22.0)
    assert list(keep) == [False, True, True]


def test_chromatic_keeps_the_pink_and_drops_grey():
    px = np.array([[200.0, 150, 145], [120, 118, 119], [30, 30, 33], [190, 170, 120]])
    assert list(jl.chromatic(px)) == [True, False, False, True]


def test_intervals_and_windows():
    t = np.arange(0, 20, 0.5)
    m = np.zeros(len(t), bool)
    m[2:6] = True
    m[8:10] = True          # 0.5 s run: gap to the previous run is 1.0 s -> merged
    m[30] = True            # a lone frame
    iv = jl.intervals(m, t, gap_s=1.5, min_s=0.5)
    assert iv == [[1.0, 4.5]]
    sc = np.array([1.0, 0.9, 0.95, 0.2, 0.8])
    st = np.array([0.0, 1.0, 20.0, 30.0, 40.0])
    assert jl.choose_windows_overlap(sc, st, 3, 14.0) == [0, 2, 4]


def test_complementary_windows_add_the_missing_side():
    T = 300
    sight = np.zeros((T, 2, 8))
    sight[0:100, :, 0] = 0.1      # front seen in the first third
    sight[100:200, :, 4] = 0.1    # back in the second
    sight[200:300, :, 0] = 0.1    # front again (a duplicate of the first)
    contact = np.zeros(T)
    starts = np.array([0, 100, 200])
    ch = jl.choose_complementary(sight, contact, starts, 100, 2, 50, need=5.0)
    assert set(ch) == {0, 1} or set(ch) == {1, 2}                  # one front window + the back window, never two fronts
    assert jl.coverage_objective(sight, np.zeros(T, bool), contact, 5.0) == 0.0
    both = np.zeros(T, bool)
    both[:200] = True
    one = np.zeros(T, bool)
    one[:100] = True
    assert jl.coverage_objective(sight, both, contact, 5.0) > jl.coverage_objective(sight, one, contact, 5.0)


def test_side_seconds_counts_frames_and_weights():
    g = np.array([0, 0, 1, 2, 2, 2, 3, -1])
    w = np.array([1.0, 0.5, 0, 1, 1, 1, 0.2, 1.0])
    r = jl.side_seconds(g, w, 0.5)
    assert r["front"] == {"seconds": 1.0, "weighted": 0.8} and r["right_side"]["seconds"] == 0.0 and r["back"]["seconds"] == 1.5
    assert r["left_side"]["weighted"] == 0.1


def test_shrink_weight_bounds_and_monotone():
    assert jp.shrink_weight(10.0, 0.0) == 1.0
    assert jp.shrink_weight(10.0, 10.0) == 0.5
    assert jp.shrink_weight(10.0, 100.0) == 0.25                          # the floor
    assert jp.shrink_weight(10.0, 5.0) > jp.shrink_weight(10.0, 8.0)


def test_combine_betas_matches_the_face_identity_rule_and_split_half_is_small_for_one_face():
    from dancecap import face_identity as fi
    rng = np.random.default_rng(1)
    truth = rng.normal(size=300)
    B = truth + 0.1 * rng.normal(size=(12, 300))
    B[3] += 5.0                                                          # an outlier image
    err = np.full(12, 0.05)
    per = [{"name": f"v{i}", "beta": B[i], "err_mp_occ": 0.05 / 1.0, "err_contour_rtmw_occ": 0.0} for i in range(12)]
    ref = fi.combine(per, 1.0)["beta"]
    got = jp.combine_betas(B, np.ones(12), err)
    assert np.allclose(ref, got) and np.linalg.norm(got - truth) < 0.6
    sh = jp.split_half(B, err, np.random.default_rng(0))
    assert sh["n"] == 12 and sh["half_distance_mean"] < 3.0
    assert sh["implied_full_set_error"] == round(sh["half_distance_mean"] / 2, 3) or abs(sh["implied_full_set_error"] - sh["half_distance_mean"] / 2) < 1e-3
    assert jp.split_half(B[:3], err[:3], np.random.default_rng(0))["half_distance_mean"] is None


def test_projection_scale_and_nearest_rank():
    ref = np.array([1.0, 0.0, 0.0])
    assert abs(jp.projection_scale(np.array([2.0, 0.0, 0.0]), ref) - 0.5) < 1e-9
    assert abs(jp.projection_scale(np.array([1.0, 0.0, 0.0]), ref) - 1.0) < 1e-9
    est = np.array([0.8, 0.3, 0.0])
    assert jp.nearest_rank(est, ref, [np.array([0.0, 5.0, 0.0]), np.array([0.8, 0.2, 0.0])]) == 1


def test_pod_job_has_its_own_kind_that_no_farm_worker_claims():
    assert jl.KIND == "avatarjl" and jl.OWNER == "J5a"
    farm_kinds = {"front", "fit", "refine", "full", "asr", "analyse", "avatar", "avatarca", "default", "agents"}
    assert jl.KIND not in farm_kinds
    steps = jp.identity_steps()
    assert [s["name"] for s in steps] == ["links", "identity"] and "jl_avatars_pod pod-identity" in steps[1]["cmd"]
    outs = [o["path"] for o in jp.identity_outputs()]
    assert all(o.startswith("dancecap/work/john_lula/") for o in outs)
    assert {"dancecap/work/john_lula/face_shape/lead_identity.npz", "dancecap/work/john_lula/face_shape/follow_identity.npz"} <= set(outs)
    est = jp.estimate()
    assert 0 < est["usd"] < 0.2


def test_stage_table_lists_every_stage_once_with_its_class():
    names = [s.name for s in jl.STAGES]
    assert len(names) == len(set(names))
    free = {s.name for s in jl.STAGES if not s.poses}
    assert {"face-frames", "face-landmarks", "face-identity", "shoes-video", "texture-preview"} <= free
    assert {"texture-fuse", "face-texture", "hair-groom", "shoes", "export"} <= {s.name for s in jl.STAGES if s.poses}
    json.dumps([ca.asdict(s) for s in jl.STAGES])


def test_no_name_or_absolute_path_in_the_module_sources():
    import re
    from pathlib import Path
    for mod in (jl, jp):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:[\\/]Users", src)
        assert not re.search(r"\b(John|Lula|Carlos|Aline)\b", src), mod.__name__
