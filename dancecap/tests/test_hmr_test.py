"""dancecap.hmr_test (and the pure helpers of remote/cand_hmr/exmr_pod.py) on small synthetic arrays: clip / frame
mapping, the pelvis re-ray placements A / B (check error 1), COCO-joint projection and the per-frame score, the owner's
pass bar, the scorer self-check tolerances (check error 10), the sheet times and the ExMR integrity checks."""
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

from dancecap import hmr_test as H
from dancecap.carlos_aline_fitcheck import project

POD = Path(__file__).resolve().parents[1] / "remote" / "cand_hmr" / "exmr_pod.py"


def _pod():
    spec = importlib.util.spec_from_file_location("exmr_pod_under_test", POD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _K(f, cx=640.0, cy=360.0, n=1):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return np.repeat(K[None], n, 0)


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _body(n=4, seed=0, z=5.0):
    rng = np.random.default_rng(seed)
    J = rng.normal(0, 0.3, (n, H.NJ, 3))
    J[:, 0] = 0
    return J + np.array([0.4, -0.2, z]) + rng.normal(0, 0.2, (n, 1, 3))


# ----------------------------------------------------------------------------------------------- clips and frame mapping


def test_clip_cut_ranges_match_the_plan_table():
    got = {c: H.clip_cut_range(*H.CLIPS[c], 30) for c in H.CLIPS}
    assert got == {"E1": (2823, 3117, 295), "E2": (4050, 4428, 379), "H1": (2070, 2490, 421), "H3": (5040, 5559, 520)}
    assert sum(L for _, _, L in got.values()) == 1615


def test_grid_and_cut_indices_are_inverse():
    i = np.arange(10)
    n = H.cut_of_grid_index(i, 507)
    assert (n == i + 87).all()
    assert (H.grid_index_of_cut(n, 507) == i).all()


def test_h_index_is_grid_plus_three_and_flags_missing_rows():
    th = np.round(2.8 + np.arange(50) / 30, 6)
    tg = 16.9 + np.arange(60) / 30
    ih, ok = H.h_index(tg, th)
    assert ok[:47].all() and not ok[47:].any()
    assert (ih[:47] == np.arange(47) + 3).all()


def test_frame_map_rows_within_20_ms():
    tc = np.arange(0, 2, 1 / 29.0)
    t = np.array([0.0, 0.51, 3.0])
    rows, ok = H.frame_map(t, tc, max_dt=H.MAX_DT)
    assert list(rows[:2]) == [0, 15] and ok[:2].all() and not ok[2]
    sparse = np.array([0.0, 0.1, 0.2])                   # 0.019 s and 0.021 s from the nearest row
    rows, ok = H.frame_map(np.array([0.119, 0.179]), sparse, max_dt=H.MAX_DT)
    assert list(rows) == [1, 2] and ok[0] and not ok[1]


def test_interp_grid_is_linear_and_nan_outside():
    tg = np.arange(5) / 30.0
    J = np.arange(5, dtype=float)[:, None, None] * np.ones((5, 2, 3))
    out = H.interp_grid(J, tg, np.array([0.5 / 30, 2.25 / 30, 10.0, -1.0]))
    assert np.allclose(out[0], 0.5) and np.allclose(out[1], 2.25)
    assert np.isnan(out[2]).all() and np.isnan(out[3]).all()


# ----------------------------------------------------------------------------------------------- placements (check error 1)


def test_reray_keeps_the_pelvis_pixel_the_depth_target_and_the_body_shape():
    J = _body()
    Ku, Kn = _K(1000.0, n=4), _K(1183.0, 639.5, 359.5, n=4)
    z_new = np.array([4.0, 5.5, 6.0, 7.2])
    out = H.reray(J, Ku, Kn, z_new)
    for f in range(4):
        u_before = H.pinhole(J[f, :1], Ku[f])[0][0]
        u_after = H.pinhole(out[f, :1], Kn[f])[0][0]
        assert np.allclose(u_before, u_after, atol=1e-9)
    assert np.allclose(out[:, 0, 2], z_new)
    assert np.allclose(out - out[:, :1], J - J[:, :1])


def test_placement_a_scales_the_depth_by_the_focal_ratio_and_b_takes_the_reference_depth():
    J = _body()
    Ku, K2 = _K(1000.0, n=4), _K(1180.0, n=4)
    A = H.placement_a(J, Ku, K2)
    assert np.allclose(A[:, 0, 2], J[:, 0, 2] * 1.18)
    zref = np.full(4, 6.25)
    B = H.placement_b(J, Ku, K2, zref)
    assert np.allclose(B[:, 0, 2], 6.25)
    assert np.allclose(H.placement_a(J, K2, K2), J)      # the right focal: nothing moves


def test_old_slide_moves_the_image_off_centre_by_the_focal_ratio_the_reray_does_not():
    J = _body(n=1)
    Ku, K2 = _K(1000.0, n=1), _K(1180.0, n=1)
    z = np.array([J[0, 0, 2] * 1.3])
    naive, fixed = H.slide_naive(J, z), H.reray(J, Ku, K2, z)
    u0 = H.pinhole(J[0, :1], Ku[0])[0][0]
    un = H.pinhole(naive[0, :1], K2[0])[0][0]
    uf = H.pinhole(fixed[0, :1], K2[0])[0][0]
    c = np.array([640.0, 360.0])
    assert np.allclose(un - c, 1.18 * (u0 - c))         # the B1 / B2 penalty of the old formula
    assert np.allclose(uf, u0)
    assert np.allclose(H.slide_naive(J, z)[0, 0, 2], z[0])
    same = H.reray(J, Ku, Ku, z)
    assert np.allclose(same, naive)                      # with the method's own K both agree


def test_to_world_inverts_the_world_to_camera_transform():
    rng = np.random.default_rng(1)
    Xw = rng.normal(0, 1, (3, H.NJ, 3))
    R = np.stack([_rot("y", 30), _rot("x", -10) @ _rot("z", 5), np.eye(3)])
    t = rng.normal(0, 2, (3, 3))
    assert np.allclose(H.to_world(H.to_camera(Xw, R, t), R, t), Xw)


# ----------------------------------------------------------------------------------------------- COCO projection and score


def _scene(seed=2):
    """World body 3 m in front of a camera turned 20 deg; the teacher kp = the exact projection of the COCO joints."""
    J = _body(n=1, seed=seed, z=0.0)[0]
    R, t = _rot("y", 20), np.array([0.1, 0.2, 3.5])
    K = _K(1500.0, 359.5, 639.5)[0]
    uv, z = project(J[H.SMX], R, t, K, 0.0)
    kp = np.zeros((17, 3))
    kp[H.COCO, :2], kp[H.COCO, 2] = uv, 0.9
    return J, R, t, K, uv, z, kp


def test_coco_mapping_is_the_smplx_body_joints():
    assert H.COCO == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert H.SMX == [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]


def test_frame_score_zero_on_the_exact_projection_and_percent_of_box_height():
    J, R, t, K, uv, z, kp = _scene()
    assert H.frame_score(uv, z, kp, 400.0) == (0.0, 0.0)
    pct, px = H.frame_score(uv + np.array([6.0, 8.0]), z, kp, 200.0)
    assert px == pytest.approx(10.0) and pct == pytest.approx(5.0)
    pct, _ = H.frame_score(uv + np.array([6.0, 8.0]), z, kp, 20.0)     # box height floored at 50 px
    assert pct == pytest.approx(20.0)


def test_frame_score_needs_six_confident_joints_in_front_of_the_camera():
    J, R, t, K, uv, z, kp = _scene()
    low = kp.copy()
    low[H.COCO[:7], 2] = 0.3                              # 5 confident joints left
    assert np.isnan(H.frame_score(uv, z, low, 300.0)[0])
    zb = z.copy()
    zb[:7] = -1.0                                         # 7 joints behind the camera: 5 left
    assert np.isnan(H.frame_score(uv, zb, kp, 300.0)[0])
    nan_uv = uv.copy()
    nan_uv[0] = np.nan
    assert np.isnan(H.frame_score(nan_uv, z, kp, 300.0)[0])
    unconf_nan = kp.copy()
    unconf_nan[H.COCO[0], 2] = 0.1                        # a NaN on an unused joint does not matter
    assert H.frame_score(nan_uv, z, unconf_nan, 300.0)[0] == 0.0


def test_pinhole_own_image_and_the_wrong_dancer_rule():
    Jc = np.array([[0.0, 0.0, 4.0], [0.4, -0.2, 2.0], [0.0, 0.0, -1.0]])
    uv, z = H.pinhole(Jc, _K(1000.0)[0])
    assert np.allclose(uv[0], [640, 360]) and np.allclose(uv[1], [840, 260]) and np.isnan(uv[2]).all()
    assert H.wrong_dancer(10.0, 4.0) == 1.0 and H.wrong_dancer(4.0, 10.0) == 0.0
    assert np.isnan(H.wrong_dancer(np.nan, 3.0))


def test_pa_mpjpe_is_zero_for_a_similar_copy():
    rng = np.random.default_rng(3)
    Y = rng.normal(0, 0.4, (H.NJ, 3))
    X = 1.3 * Y @ _rot("z", 40).T + np.array([2.0, -1.0, 0.5])
    assert H.pa_mpjpe(X, Y) == pytest.approx(0.0, abs=1e-6)
    Xn = X + rng.normal(0, 0.01, X.shape)
    assert 2.0 < H.pa_mpjpe(Xn, Y) < 20.0


def test_summary_median_p90_and_gross_share():
    s = H.summary([1.0, 2.0, 3.0, 20.0, np.nan])
    assert s["n"] == 4 and s["median"] == 2.5 and s["gross"] == 0.25
    assert H.summary([np.nan])["median"] is None


# ----------------------------------------------------------------------------------------------- pass bar and self-check


def test_pass_bar_needs_both_metrics_no_worse_than_b1_and_b2():
    m1b = {"B1": 4.56, "B2": 3.74, "cand": 3.70}
    pa = {"B1": 80.4, "B2": 50.0, "cand": 49.0}
    assert H.verdict(m1b, pa)["improves"] is True
    assert H.verdict({**m1b, "cand": 3.80}, pa)["improves"] is False          # better than B1, worse than B2
    assert H.verdict(m1b, {**pa, "cand": 50.4})["improves"] is False
    assert H.verdict({**m1b, "cand": 3.744}, pa)["improves"] is True           # a tie at the sheet's 0.01 % passes
    assert H.verdict({**m1b, "cand": None}, pa)["improves"] is None
    assert H.verdict(m1b, pa)["text"] == "improves accuracy"


def _measured(**over):
    m = {"b0_m1_good": (2.60, 2.60), "b1_m1_naive_good": (5.47, 5.97), "b1_pa_good": (80.4, 64.1),
         "n_good_mask": (676, 589), "n_good": (660, 578), "n_held": (901, 1003), "b2_m1b_e1": (1.9, 1.7)}
    m.update(over)
    return m


def test_self_check_passes_on_the_independent_checks_numbers():
    res = H.self_check(_measured())
    assert res["ok"] and all(r["ok"] for r in res["rows"])


@pytest.mark.parametrize("over", [
    {"b0_m1_good": (2.50, 2.60)},          # 0.16 below 2.66
    {"b1_m1_naive_good": (5.47, 5.75)},    # the corrected placement would not reproduce the basis
    {"b1_pa_good": (86.0, 73.0)},          # the MEAN, not the per-frame median
    {"n_good_mask": (679, 589)},
    {"n_good": (676, 589)},                # the mask count where the scored count is checked
    {"n_held": (901, 1006)},
    {"b0_m1_good": (None, 2.6)},
    {"b2_m1b_e1": (1.9, 6.0)},
])
def test_self_check_fails_outside_the_tolerances(over):
    assert not H.self_check(_measured(**over))["ok"]


def test_self_check_tolerance_edges():
    assert H.self_check(_measured(b0_m1_good=(2.81, 2.52)))["ok"]          # +-0.15
    assert H.self_check(_measured(n_held=(903, 1001)))["ok"]               # +-2


def test_common_verdicts_use_the_frames_every_method_scored():
    n = 8
    good = np.ones(n, bool)
    per = {f"good__{r}": good for r in H.ROLES}
    for r in H.ROLES:
        per[f"B1__{r}__m1B"] = np.full(n, 5.0)
        per[f"B2__{r}__m1B"] = np.full(n, 4.0)
        per[f"exmr__{r}__m1B"] = np.r_[np.full(n - 2, 3.0), 99.0, 99.0]
        per[f"B1__{r}__pa"] = np.full(n, 80.0)
        per[f"B2__{r}__pa"] = np.full(n, 50.0)
        per[f"exmr__{r}__pa"] = np.full(n, 45.0)
        per[f"B0__{r}__m1B"] = np.r_[np.full(n - 2, 2.5), 9.0, 9.0]
    per["B2__lead__m1B"] = np.r_[np.full(n - 2, 4.0), np.nan, np.nan]     # B2 lacks the two bad frames
    v = H.common_verdicts(per, {"exmr": "ok"}, ["B0", "B1", "B2", "exmr"])
    assert v["exmr"]["lead"]["n_common"] == n - 2 and v["exmr"]["lead"]["improves"] is True
    assert v["exmr"]["follow"]["n_common"] == n and v["exmr"]["follow"]["m1B"]["cand"] == 3.0
    assert v["exmr"]["lead"]["b0_m1B"] == 2.5                             # B0 on the same (common) frames
    assert v["exmr"]["follow"]["b0_m1B"] == pytest.approx(2.5)            # median of 6 x 2.5 + 2 x 9.0
    v = H.common_verdicts(per, {"exmr": "not run correctly (x)"}, ["B1", "B2", "exmr"])
    assert v["exmr"]["lead"]["improves"] is None and "not run correctly" in v["exmr"]["lead"]["text"]


# ----------------------------------------------------------------------------------------------- sheet times and clip checks


def test_sheet_times_are_fixed_fractions_and_skip_unlocalised_rows():
    c1_t = np.arange(0, 220, 1 / 29.0)
    ok = np.ones(len(c1_t), bool)
    clips = {"E1": (3273, 3507)}
    out = H.sheet_times(c1_t, ok, lambda t: t - 0.0645, clips)
    assert [d["n"] for d in out] == [2912, 2970, 3028] and all(d["moved"] == 0 for d in out)
    row = out[0]["row01"]
    ok[row] = False
    out2 = H.sheet_times(c1_t, ok, lambda t: t - 0.0645, clips)
    assert out2[0]["n"] != 2912 and abs(out2[0]["moved"]) <= 2 and out2[0]["row01"] != row
    assert out2[1:] == out[1:]


def test_endpoint_check_catches_an_off_by_one():
    rng = np.random.default_rng(4)
    src = {n: rng.integers(0, 255, (8, 8, 3), dtype=np.uint8) for n in range(9, 22)}
    good = {0: src[10], 10: src[20]}
    assert H.endpoint_check(src, good, 10, 20, 11)["ok"]
    shifted = {0: src[11], 10: src[21]}
    assert not H.endpoint_check(src, shifted, 10, 20, 11)["ok"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="no ffmpeg / ffprobe")
def test_cut_clip_runs_on_this_ffmpeg_and_passes_the_end_point_check(tmp_path):
    """Review finding 1: ffmpeg 7+ refused -fps_mode passthrough with -r 30. A 4 s 128x72 B-frame video: the cut has
    exactly the frames n0..n1, 30/1 fps, and its end points match the source frames n0 / n1 better than their neighbours."""
    import subprocess
    pytest.importorskip("cv2")
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=128x72:rate=30:duration=4", "-c:v", "libx264", "-bf", "3", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    n0, n1 = 37, 71
    L = n1 - n0 + 1
    dst = tmp_path / "clip.mp4"
    H.cut_clip(src, dst, n0, n1)
    assert dst.exists() and not (tmp_path / "clip.part.mp4").exists()
    assert H.count_frames(dst) == L
    assert H.video_size(dst) == (128, 72, "30/1")
    sf = H.read_frames(src, {n0 - 1, n0, n0 + 1, n1 - 1, n1, n1 + 1})
    chk = H.endpoint_check(sf, H.read_frames(dst, {0, L - 1}), n0, n1, L)
    assert chk["ok"], chk


def test_f_mm_reproduces_the_focal_through_exmrs_estimate_k():
    f_mm = H.f_mm_of(2121.0, 1280, 720)
    focal = (1280 ** 2 + 720 ** 2) ** 0.5 / (24 ** 2 + 36 ** 2) ** 0.5 * f_mm      # exmr.geometry.camera.estimate_K
    assert focal == pytest.approx(2121.0, rel=1e-12)
    assert f_mm == pytest.approx(2121.0 * 43.267 / 1468.6, rel=1e-4)


# ----------------------------------------------------------------------------------------------- exmr_pod.py integrity helpers


def test_pod_relative_rotations_keep_the_frame_to_frame_motion():
    P = _pod()
    R = np.stack([_rot("y", a) @ _rot("x", 3) for a in (0, 2, 5, 9)])
    Rr = P.relative_rotations(R)
    assert np.allclose(Rr[0], np.eye(3))
    for t in range(3):
        assert np.allclose(Rr[t + 1] @ Rr[t].T, R[t + 1] @ R[t].T)
    assert np.allclose(P.rot_angle_deg(Rr, np.repeat(np.eye(3)[None], 4, 0)), [0, 2, 5, 9], atol=1e-6)


def test_pod_k_check_and_box_iou():
    P = _pod()
    K02 = _K(2121.0, 639.5, 359.5, n=3)
    Kx = _K(2121.0 * 1.004, 640.0, 360.0, n=3)
    assert P.k_check(Kx, K02)["ok"]
    assert not P.k_check(_K(2121.0 * 1.006, n=3), K02)["ok"]
    assert not P.k_check(_K(2121.0, 642.0, 360.0, n=3), K02)["ok"]
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10.0]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10.0]])
    assert np.allclose(P.box_iou(a, b), [1.0, 1 / 3])
    assert P.f_mm_of(2121.0, 1280, 720) == pytest.approx(H.f_mm_of(2121.0, 1280, 720))


def test_pod_convention_gate_fails_only_on_clear_evidence_for_the_transpose():
    P = _pod()
    assert P.convention_verdict(0.3, 4.1)            # SimpleVO agrees with our world-to-camera R
    assert P.convention_verdict(1.4, 1.1)            # noise: not decisive
    assert not P.convention_verdict(3.8, 0.4)        # it agrees with R^T: the convention would be wrong


def test_pod_cache_check_and_2d_check():
    P = _pod()
    good = {"tracks.pt": ["hit"], "camera_rotations.pt": ["hit"], "keypoints.pt": ["computed"],
            "body_features.pt": ["computed"], "hand_features.pt": ["hit"]}
    assert P.cache_check(good)["ok"]
    assert not P.cache_check({**good, "tracks.pt": ["computed"]})["ok"]          # ExMR's tracker would have run
    assert not P.cache_check({**good, "camera_rotations.pt": ["hit", "hit"]})["ok"]
    assert not P.cache_check({k: v for k, v in good.items() if k != "keypoints.pt"})["ok"]
    L = 5
    j2d = np.zeros((L, 76, 2))
    kp = np.zeros((L, 17, 3))
    coco = sorted(P.COCO_TO_SMPLX)
    for c in coco:
        j2d[:, P.COCO_TO_SMPLX[c]] = [100 + 3 * c, 200 + 5 * c]
        kp[:, c, :2] = [100 + 3 * c, 200 + 5 * c + 20]               # 20 px off
        kp[:, c, 2] = 0.9
    box = np.tile([0, 0, 100, 400.0], (L, 1))
    r = P.kp2d_check(j2d, kp, box)
    assert r["median_pct"] == pytest.approx(5.0) and r["ok"]
    r = P.kp2d_check(j2d, kp, np.tile([0, 0, 100, 150.0], (L, 1)))
    assert r["median_pct"] == pytest.approx(100 * 20 / 150) and not r["ok"]


def test_pod_tidy_outputs_after_a_failed_run_leaves_only_small_files(tmp_path):
    """Review finding 4: a run_video that raised leaves ExMR's frames and the caches in out/; the finally moves the caches
    to the pod-side keep directory and deletes the frames, so neither is packed. Idempotent, never raises."""
    P = _pod()
    out, keep = tmp_path / "exmr" / "E1_lead", tmp_path / "caches" / "exmr" / "E1_lead"
    (out / "frames").mkdir(parents=True)
    for i in range(3):
        (out / "frames" / f"{i:06d}.jpg").write_bytes(b"\xff\xd8" + bytes(64))
    (out / "frames" / ".extracted").write_text("v\n3")
    (out / "cache").mkdir()
    (out / "cache" / "tracks.pt").write_bytes(b"t")
    (out / "cache" / "keypoints.pt").write_bytes(b"k")
    (out / "E1_lead_cfr.mp4").write_bytes(b"v")
    (out / "run.json").write_text("{}")                    # ExMR's own small run record stays
    keep.mkdir(parents=True)
    (keep / "stale.pt").write_bytes(b"s")                  # an older run's caches are replaced
    done = P.tidy_outputs(out, keep)
    assert sorted(p.name for p in out.iterdir()) == ["run.json"]
    assert sorted(p.name for p in keep.iterdir()) == ["keypoints.pt", "tracks.pt"]
    assert any("frames" in d for d in done) and any("cache" in d for d in done)
    assert P.tidy_outputs(out, keep) == []                 # nothing left to do
    assert (keep / "tracks.pt").exists()


# ----------------------------------------------------------------------------------------------- synthetic end to end


def _look(C, yaw_deg):
    """World (y up) -> OpenCV camera (y down) at C, looking along the world direction yaw_deg about +y from +z."""
    base = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])   # looks along world +z
    R = base @ _rot("y", -yaw_deg)
    return R, -R @ C


def _synthetic_take(tmp_path, monkeypatch):
    """A 120-frame ll grid (k 507..626) with two exactly known dancers, camera 02 (tripod, panning 0.05 deg / frame)
    and camera 01 (29 fps, sync delta -64.5 ms), teachers = the exact projections of the truth, B1 / B2 = the truth
    seen with an 18 % too short focal (pixel-consistent pelvis, as a real mono fit), ExMR = the truth in camera 02."""
    W = tmp_path / "work"
    monkeypatch.setattr(H, "WORK", W)
    monkeypatch.setattr(H, "CLIPS", {"E1": (530, 560), "E2": (575, 605)})
    monkeypatch.setattr(H, "CLIP_NOTE", {"E1": "a", "E2": "b"})
    ll, h = W / "lucas_luiza", W / "lucas_luiza_h"
    for d in ("cameras", "sync", "people", "smplx_rf2_noslip"):
        (ll / d).mkdir(parents=True, exist_ok=True)
    NG = 120
    tg = np.round(16.9 + np.arange(NG) / 30, 6)
    rng = np.random.default_rng(7)
    tmpl = rng.normal(0, 0.18, (H.NJ, 3))
    tmpl[0] = 0
    truth = {}
    for r, x0 in (("lead", -0.45), ("follow", 0.45)):
        J = np.zeros((NG, 55, 3))
        for i in range(NG):
            J[i, :H.NJ] = tmpl @ _rot("y", 2.0 * i + (0 if r == "lead" else 90)).T + [x0 + 0.004 * i, 1.0, 0.1 * (r == "follow")]
        J[:, H.NJ:] = J[:, :1]
        truth[r] = J
        np.savez(ll / "smplx_rf2_noslip" / f"{r}.npz", t=tg, joints=J, quality_flag=np.full(NG, 2, np.int8),
                 repaired=np.zeros(NG, bool), suspect=np.zeros(NG, bool), blend_seam=np.full(NG, -1, np.int16),
                 stance=rng.random((NG, 2)) < 0.4)
    (ll / "cameras" / "world_from_sfm.json").write_text('{"s": 1.0, "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0]}')
    (ll / "sync" / "offsets.json").write_text('{"offsets": {"01": {"delta": -0.0645}, "02": {"delta": 0.0}}}')
    # camera 02: the grid's own times, panning
    R2, t2 = zip(*[_look(np.array([0.0, 1.5, -6.0]), 0.05 * i) for i in range(NG)])
    c2 = dict(t=tg.copy(), K=_K(2000.0, 639.5, 359.5, NG), R=np.array(R2), tvec=np.array(t2), dist=np.full(NG, -0.01),
              ok=np.ones(NG, bool))
    np.savez(ll / "cameras" / "02.npz", **c2)
    t1 = np.round(16.8 + np.arange(118) / 29.0, 6)
    R1, tv1 = _look(np.array([5.0, 1.6, 0.0]), -90.0)
    c1 = dict(t=t1, K=_K(1700.0, 359.5, 639.5, len(t1)), R=np.repeat(R1[None], len(t1), 0),
              tvec=np.repeat(tv1[None], len(t1), 0), dist=np.full(len(t1), 0.01), ok=np.ones(len(t1), bool))
    np.savez(ll / "cameras" / "01.npz", **c1)

    def teachers(c, tq, name):
        n = len(c["t"])
        kp = np.zeros((n, 2, 17, 3), np.float32)
        box = np.zeros((n, 2, 4), np.float32)
        for s, r in enumerate(H.ROLES):
            Jt = H.interp_grid(truth[r][:, :H.NJ], tg, tq)
            for k in range(n):
                if not np.isfinite(Jt[k]).all():
                    continue
                uv, _ = project(Jt[k][H.SMX], c["R"][k], c["tvec"][k], c["K"][k], float(c["dist"][k]))
                kp[k, s, H.COCO, :2], kp[k, s, H.COCO, 2] = uv, 0.9
                box[k, s] = [uv[:, 0].min() - 10, uv[:, 1].min() - 60, uv[:, 0].max() + 10, uv[:, 1].max() + 30]
        np.savez(ll / "people" / name, t=c["t"], row=np.tile([[0, 1]], (n, 1)).astype(np.int16),
                 conf=np.full((n, 2), 0.9, np.float32), kp=kp, box=box, lead_slot=np.zeros(n, np.int8))
    teachers(c1, t1 - 0.0645, "teachers_01.npz")
    teachers(c2, tg, "teachers_02.npz")
    (ll / "stitch.json").write_text('{"chunks": [{"name": "syn", "owned_k": [507, 626]}]}')
    q = W / "lucas_luiza_syn" / "quality"
    q.mkdir(parents=True)
    np.savez(q / "quality_rf2_noslip_frames.npz", t=tg, **{f"px_{c}_{r}": np.full(NG, 5.0) for c in ("01", "02")
                                                          for r in H.ROLES})
    # B1 / B2: the truth in camera 02, its pelvis at depth / 1.18 (an 18 % short focal), seen through an h world
    fh = 2000.0 / 1.18
    Ncut = NG + 87 + 10
    n_of_i = np.arange(NG) + 87
    Rh, th_ = _rot("y", 25) @ _rot("x", 5), np.array([0.3, -0.1, 0.2])
    (h / "smplx_rf2_noslip").mkdir(parents=True)
    (h / "cameras").mkdir(parents=True)
    (h / "realism" / "sam3db" / "pod_out").mkdir(parents=True)
    (h / "realism" / "sam3db" / "input").mkdir(parents=True)
    b2 = {"frame_i": np.arange(Ncut)}
    Jcam = {}
    for r in H.ROLES:
        Jc = H.to_camera(truth[r][:, :H.NJ], c2["R"], c2["tvec"])
        Jcam[r] = Jc
        P = Jc[:, :1]
        Jh = Jc - P + P * np.array([1.0, 1.0, 1 / 1.18])
        hw = np.full((NG + 3, 55, 3), np.nan)
        hw[3:, :H.NJ] = H.to_world(Jh, np.repeat(Rh[None], NG, 0), np.repeat(th_[None], NG, 0))
        hw[:3, :H.NJ] = hw[3, :H.NJ]
        hw[:, H.NJ:] = hw[:, :1]
        np.savez(h / "smplx_rf2_noslip" / f"{r}.npz", t=np.round(2.8 + np.arange(NG + 3) / 30, 6), joints=hw)
        jb = np.full((Ncut, 55, 3), np.nan, np.float32)
        jb[n_of_i, :H.NJ] = Jh
        jb[n_of_i[30:32]] = np.nan                              # two frames SAM 3D Body skipped (in E1)
        b2[f"{r}_joints"] = jb
    np.savez(h / "realism" / "sam3db" / "pod_out" / "smplx_mask255.npz", **b2)
    import json as _json
    (h / "realism" / "sam3db" / "input" / "meta.json").write_text(_json.dumps(
        {"frames": [{"K": [[fh, 0, 640.0], [0, fh, 360.0], [0, 0, 1]]} for _ in range(Ncut)]}))
    np.savez(h / "cameras" / "01.npz", t=np.round(np.arange(Ncut) / 30, 6), K=_K(fh, n=Ncut), R=np.repeat(Rh[None], Ncut, 0),
             tvec=np.repeat(th_[None], Ncut, 0))
    # the clips (meta only) and ExMR's outputs: the truth in camera 02 with K02
    od = H.out_dir()
    runs = {}
    for c, (k0, k1) in H.CLIPS.items():
        n0, n1, L = H.clip_cut_range(k0, k1, 5)
        (od / "clips" / c).mkdir(parents=True)
        (od / "clips" / c / "meta.json").write_text(_json.dumps({"clip": c, "n0": n0, "n1": n1, "L": L}))
        for r in H.ROLES:
            i = H.grid_index_of_cut(np.arange(n0, n1 + 1), 507)
            d = od / "exmr" / f"{c}_{r}"
            d.mkdir(parents=True)
            J76 = np.zeros((L, 76, 3), np.float32)
            J76[:, :H.NJ] = Jcam[r][i]
            np.savez(d / "results.npz", joints_incam=J76, K_fullimg=c2["K"][i].astype(np.float32))
            runs[f"{c}_{r}"] = {"ok": True}
    (od / "exmr" / "run.json").write_text(_json.dumps({"runs": runs}))
    return od


def _args(**kw):
    import argparse
    return argparse.Namespace(take="lucas_luiza", methods="B0,B1,B2,exmr,flowhmr", **kw)


def test_score_and_sheet_end_to_end_on_a_synthetic_take(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    od = _synthetic_take(tmp_path, monkeypatch)
    H.cmd_score(_args())
    import json as _json
    res = _json.loads((od / "score" / "score.json").read_text())
    M = res["methods"]
    for r in H.ROLES:
        p = lambda m, k: M[m]["roles"][r]["pooled"][k]
        assert p("B0", "m1B_good")["median"] == pytest.approx(0.0, abs=1e-3)            # teachers = truth (float32 kp)
        assert p("exmr", "m1B_good")["median"] == pytest.approx(0.0, abs=1e-3)
        assert p("exmr", "pa")["median"] == pytest.approx(0.0, abs=1e-3)
        assert p("exmr", "m1b")["median"] < 0.3 and M["exmr"]["roles"][r]["pooled"]["m1c"]["share"] == 0.0
        assert p("B1", "m1B_good")["median"] < 0.3                                       # the re-ray undoes the focal
        assert p("B1", "m1B_naive")["median"] > 1.0 + p("B1", "m1B_good")["median"]     # the old slide does not
        assert p("B1", "pa")["median"] == pytest.approx(0.0, abs=1e-3)
        assert p("B1", "m1A")["median"] < 0.3 and p("B1", "pelA_err")["median"] < 5e-3   # placement A: f02 / f_h (0.5 px pp)
        assert p("B2", "pa")["n"] == p("B1", "pa")["n"] - 2 and p("B2", "m1B_good")["n"] < p("B1", "m1B_good")["n"]
        assert res["frames"][r]["good"] == res["frames"][r]["held"] == 62
    assert M["exmr"]["status"] == "ok" and M["flowhmr"]["status"].startswith("not run")
    assert not res["self_check"]["ok"]                     # synthetic numbers are not the basis numbers
    H.cmd_sheet(_args())
    md = (od / "score" / "sheet.md").read_text()
    assert "INVALID: scorer" in md and "improves accuracy" not in md
    for f in ("contact_cam01.jpg", "contact_cam02.jpg", "sheet.json", "per_frame.npz", "sheet_points.npz"):
        assert (od / "score" / f).exists()
    # with a passing self-check the pass bar decides: ExMR = the truth, so it is no worse than B1 and B2
    monkeypatch.setattr(H, "self_check", lambda m: {"ok": True, "rows": []})
    H.cmd_score(_args())
    H.cmd_sheet(_args())
    sj = _json.loads((od / "score" / "sheet.json").read_text())
    assert sj["valid"] and all(sj["verdicts"]["exmr"][r]["improves"] for r in H.ROLES)
    assert all(sj["verdicts"]["exmr"][r]["b0_m1B"] == pytest.approx(0.0, abs=1e-3) for r in H.ROLES)
    md = (od / "score" / "sheet.md").read_text()
    assert md.count("(B0 reference 0.00)") == 2 and "improves accuracy" in md
    assert 56 <= sj["verdicts"]["exmr"]["lead"]["n_common"] < 62 and "flowhmr" in sj["verdicts"]   # B2 lacks 2 (+ interp)
    assert sj["verdicts"]["flowhmr"]["lead"]["improves"] is None
