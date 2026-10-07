"""Unit tests for dancecap.mamma_dyn (no GPU, no MAMMA run) and the turing-dynamic-cams MAMMA patches.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_mamma_dyn.py -q      (from the workspace root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import mamma_dyn as md  # noqa: E402

MAMMA = md.MAMMA_ROOT


def _rand_rot(rng):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rng.normal(0, 1, 3)).as_matrix()


def test_world_from_sfm_roundtrip_and_projection():
    rng = np.random.default_rng(0)
    F = 5
    R_w = np.stack([_rand_rot(rng) for _ in range(F)])
    t_w = rng.normal(0, 3, (F, 3))
    sim3 = {"s": 0.37, "R": _rand_rot(rng).tolist(), "t": rng.normal(0, 2, 3).tolist()}
    R_s, t_s = md.sfm_cameras(R_w, t_w, sim3)
    R_b, t_b = md.world_cameras(R_s, t_s, sim3)
    assert np.allclose(R_b, R_w) and np.allclose(t_b, t_w)
    # same pixel whether the point is projected in the world or the SfM frame
    Xw = rng.normal(0, 1, (F, 7, 3)) + np.array([0, 1, 0])
    s, Rs, ts = sim3["s"], np.array(sim3["R"]), np.array(sim3["t"])
    Xs = np.einsum("ji,fnj->fni", Rs, Xw - ts) / s
    K = np.tile(np.array([[1000, 0, 500], [0, 1000, 400], [0, 0, 1.0]]), (F, 1, 1))
    pw, zw = md.project(Xw, K, R_w, t_w)
    ps, zs = md.project(Xs, K, R_s, t_s)
    assert np.allclose(pw, ps, atol=1e-6)
    assert np.allclose(zw, zs * s)  # world camera coordinates are metric


def test_undistort_inverts_simple_radial():
    K = np.array([[1200, 0, 540], [0, 1200, 960], [0, 0, 1.0]])
    px = np.random.default_rng(1).uniform([0, 0], [1080, 1920], (100, 2))
    for k1 in (-0.05, 0.03):
        back = md.undistort_simple_radial(md.distort_simple_radial(px, K, k1), K, k1)
        assert np.abs(back - px).max() < 1e-3


def test_logvar_sigma_matches_mamma_convention():
    box = np.array([300.0, 600.0])
    lv = md.sigma_px_to_logvar(np.array([2.0, 2.0]), box)
    assert np.allclose(md.logvar_to_sigma_px(lv, box), 2.0)
    # ma_3d: std = sqrt(exp(lv)) (fitting.process_2d_pred), sigma_crop = std / 2 * 512 (losses.proj_pts_loss)
    sigma_crop = np.sqrt(np.exp(lv)) / 2 * 512
    assert np.allclose(sigma_crop, 2.0 * 512 / box)


def _linear_track(t, P=2, N=4):
    """Landmarks moving linearly in time: x = 100 + 300 t + 10 n, y = 200 + 50 p."""
    lm = np.zeros((len(t), P, N, 3))
    for p in range(P):
        for n in range(N):
            lm[:, p, n, 0] = 100 + 300 * t + 10 * n
            lm[:, p, n, 1] = 200 + 50 * p
    box = np.full((len(t), P), 400.0)
    lm[..., 2] = md.sigma_px_to_logvar(np.full(lm.shape[:3], 1.5), box[..., None])
    return lm, np.ones(lm.shape[:3]), box


def test_resample_landmarks_interpolates_inflates_and_drops_gaps():
    t = 0.01 + np.arange(30) / 30.0  # 10 ms phase offset from the grid
    lm, vis, box = _linear_track(t)
    vis[10:14, 0] = 0.0  # person 0 missing for 4 frames
    grid = md.master_grid(0.05, 0.9)
    out = md.resample_landmarks(t, lm, vis, box, grid, 1 / 30.0)
    x_true = 100 + 300 * grid[:, None] + 10 * np.arange(4)[None]
    p1 = out["landmarks"][:, 1]
    assert np.allclose(p1[..., 0], x_true, atol=1e-6)  # exact for linear motion
    sig = md.logvar_to_sigma_px(p1[..., 2], np.full(p1.shape[:2], 400.0))
    w = (grid - 0.01) * 30 % 1
    expect = np.sqrt(1.5 ** 2 + (10.0 * np.minimum(w, 1 - w)) ** 2)  # |dx| per frame = 300/30 = 10 px
    assert np.allclose(sig, expect[:, None], atol=1e-4)
    # person 0: grid times whose nearest valid frame is > 1 frame away are invisible and zeroed
    t_valid = np.delete(t, np.arange(10, 14))
    far = np.min(np.abs(grid[:, None] - t_valid[None]), 1) * 30 > 1.0 + 1e-6
    assert far.any()
    assert np.all(out["visibilities"][far, 0] == 0) and np.all(out["landmarks"][far, 0] == 0)
    near_only = ~far & (out["visibilities"][:, 0, 0] > 0)
    assert np.all(out["visibilities"][~far, 0] > 0)
    assert near_only.sum() > 0
    # outside the camera's coverage -> invisible
    out2 = md.resample_landmarks(t, lm, vis, box, np.array([1.5]), 1 / 30.0)
    assert np.all(out2["visibilities"] == 0)


def test_resample_cameras_slerp_logfocal_and_coverage():
    from scipy.spatial.transform import Rotation
    t = np.arange(20) / 30.0 + 0.012
    ang = 0.8 * t  # constant angular velocity about a fixed axis -> slerp is exact
    axis = np.array([0.3, 0.9, 0.1]) / np.linalg.norm([0.3, 0.9, 0.1])
    R = Rotation.from_rotvec(ang[:, None] * axis).as_matrix()
    C = np.stack([np.sin(t), 1.5 + 0 * t, 4 + t], 1)  # linear-ish centre
    C[:, 0] = 2 * t
    tv = -np.einsum("fij,fj->fi", R, C)
    f = 1000 * np.exp(0.5 * t)  # exponential zoom -> log-focal interpolation is exact
    K = np.zeros((20, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = 540, 960, 1
    ok = np.ones(20, bool)
    ok[8] = False  # an unregistered frame inside: bracket spans 2 frames -> nearest-within-1-frame snap
    grid = md.master_grid(0.05, 0.6)
    g = md.resample_cameras(t, K, R, tv, ok, grid, 1 / 30.0)
    Rt = Rotation.from_rotvec((0.8 * grid)[:, None] * axis).as_matrix()
    bracket_ok = ~((grid > t[7]) & (grid < t[9]))
    assert np.allclose(g["R"][bracket_ok], Rt[bracket_ok], atol=1e-9)
    assert np.allclose(g["K"][bracket_ok, 0, 0], 1000 * np.exp(0.5 * grid[bracket_ok]), rtol=1e-9)
    Cg = -np.einsum("fji,fj->fi", g["R"], g["t"])
    assert np.allclose(Cg[bracket_ok, 0], 2 * grid[bracket_ok], atol=1e-9)
    assert g["ok"].all()  # every grid time is within one frame of a registered frame
    far = md.resample_cameras(t, K, R, tv, ok, np.array([2.0]), 1 / 30.0)
    assert not far["ok"][0] and np.isfinite(far["K"]).all()


def test_synthetic_rig_sees_the_couple_and_moves():
    t = np.arange(180) / 30.0
    rig = md.synthetic_rig(8, t)
    centre = np.array([[0.0, 0.9, 0.0]])
    for cam in rig:
        px, z = md.project(np.repeat(centre[None], len(t), 0), cam["K"], cam["R"], cam["t"])
        assert (z > 2.5).all()
        assert (px[:, 0, 0] > 0.2 * cam["w"]).all() and (px[:, 0, 0] < 0.8 * cam["w"]).all()
        assert (px[:, 0, 1] > 0.2 * cam["h"]).all() and (px[:, 0, 1] < 0.8 * cam["h"]).all()
        C = -np.einsum("fji,fj->fi", cam["R"], cam["t"])
        assert np.ptp(C, 0).max() > 0.2  # handheld: camera centre moves
        assert cam["K"][:, 0, 0].max() / cam["K"][:, 0, 0].min() > 1.3  # zoom
        assert np.allclose(np.linalg.det(cam["R"]), 1)


@pytest.mark.skipif(not (MAMMA / "optimization" / "utils" / "fitting.py").exists(), reason="no mamma-turing worktree")
def test_mamma_fitting_per_frame_cameras(monkeypatch):
    monkeypatch.syspath_prepend(str(MAMMA / "optimization"))
    import importlib.util
    src = (MAMMA / "optimization" / "utils" / "fitting.py").read_text()
    # exercise per_frame_cameras without importing fitting's heavy dependency chain
    ns: dict = {"np": np}
    start = src.index("def per_frame_cameras")
    end = src.index("def load_frames_in_parallel")
    exec(compile(src[start:end], "fitting.per_frame_cameras", "exec"), ns)
    pfc = ns["per_frame_cameras"]
    K = np.eye(3)
    E = np.eye(4)
    k, e = pfc(K, E, 7)
    assert k.shape == (7, 3, 3) and e.shape == (7, 4, 4)
    Ks = np.repeat(K[None], 7, 0) * np.arange(1, 8)[:, None, None]
    Es = np.repeat(E[None], 7, 0)
    k, e = pfc(Ks, Es[:, :3], 7)  # (T,3,4) is padded
    assert np.allclose(k, Ks) and e.shape == (7, 4, 4) and np.allclose(e[:, 3], [0, 0, 0, 1])
    with pytest.raises(ValueError):
        pfc(Ks, Es, 6)
    assert importlib.util is not None


@pytest.mark.skipif(not (MAMMA / "visualization" / "cameras.py").exists(), reason="no mamma-turing worktree")
def test_ma_vis_per_frame_camera_fold(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MAMMA))
    cams = pytest.importorskip("visualization.cameras")
    try:
        overlay = __import__("visualization.overlay", fromlist=["_cam_points"])
    except Exception as e:  # renderer deps (pyrender/EGL) may be missing in this env
        pytest.skip(f"visualization.overlay not importable: {e}")
    rng = np.random.default_rng(3)
    T = 6
    K = np.zeros((T, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = 900 + 100 * np.arange(T)
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = 500 + np.arange(T), 300, 1
    R = np.stack([_rand_rot(rng) @ np.eye(3) for _ in range(T)])
    E = md.extrinsics(R, rng.normal(0, 0.1, (T, 3)) + [0, 0, 5])
    np.savez(tmp_path / "cam01.npz", cam_name=np.array("cam01"), cam_int=K, cam_ext=E, cam_img_w=1000, cam_img_h=600)
    cam = cams._load_one(str(tmp_path / "cam01.npz"))
    assert cam.intrinsics_seq.shape == (T, 3, 3) and np.allclose(cam.intrinsics, K[0])
    X = rng.normal(0, 0.3, (20, 3))
    for f in range(T):
        pc = overlay._cam_points(cam, X, f)
        via_k0 = (pc @ K[0].T)[:, :2] / pc[:, 2:]
        truth, _ = md.project(X[None], K[f][None], R[f][None], E[f, :3, 3][None])
        assert np.allclose(via_k0, truth[0], atol=1e-6)
    half = cam.scaled(0.5)
    assert np.allclose(half.intrinsics_seq[:, 0, 0], K[:, 0, 0] / 2)


@pytest.mark.skipif(not (MAMMA / "optimization" / "utils" / "optimization.py").exists(), reason="no mamma-turing worktree")
def test_mamma_sdf_check_is_loud_and_dt_follows_fps(monkeypatch, capsys):
    import types
    monkeypatch.syspath_prepend(str(MAMMA / "optimization"))
    try:
        from utils.optimization import OptimizeSMPLX, frame_dt
    except Exception as e:  # pragma: no cover - env without MAMMA deps
        pytest.skip(f"MAMMA optimization not importable: {e}")
    cfg = {"optim": {"fourth_run": {"losses": {"intersection_loss": {"weight": 1.0}}}}}
    missing = types.SimpleNamespace(sdf_loss=types.SimpleNamespace(sdf=None))
    assert OptimizeSMPLX._check_sdf(missing, cfg) is False
    assert "NOT importable" in capsys.readouterr().out
    with pytest.raises(RuntimeError):
        OptimizeSMPLX._check_sdf(missing, dict(cfg, require_sdf=True))
    live = types.SimpleNamespace(sdf_loss=types.SimpleNamespace(sdf=object()))
    assert OptimizeSMPLX._check_sdf(live, cfg) is True
    assert frame_dt({"fps": 30}) == pytest.approx(1 / 30) and frame_dt({}) == pytest.approx(1 / 20)


def test_pytorch_sdf_signed_distance_on_a_cube():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    sdf = pytest.importorskip("pytorch_sdf.sdf")
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=(1, 1, 1))
    tri = torch.tensor(box.vertices[box.faces], dtype=torch.float32, device="cuda")[None]
    q = torch.tensor([[[0, 0, 0], [0.4, 0, 0], [0.7, 0, 0], [0, 0, 1.5]]], dtype=torch.float32, device="cuda")
    d, inside = sdf.SDF(distance_method="bvh").sdf_with_winding_numbers(q, tri, None)
    assert torch.allclose(d[0].cpu(), torch.tensor([-0.5, -0.1, 0.2, 1.0]), atol=1e-4)
    assert inside[0].cpu().tolist() == [True, True, False, False]


def test_outlier_frames_are_detected_and_refilled_by_interpolation():
    from scipy.spatial.transform import Rotation
    T = 20
    reproj = np.full(T, 2.0)
    reproj[[5, 12]] = [300.0, np.nan]
    bad = md.outlier_frames(reproj)
    assert np.flatnonzero(bad).tolist() == [5, 12]
    ang = np.linspace(0, 1.0, T)
    pose = np.zeros((T, 165))
    pose[:, 3:6] = ang[:, None] * np.array([0, 0, 1.0])  # one joint rotating at constant speed
    transl = np.stack([np.linspace(0, 1, T)] * 3, 1)
    noisy_pose, noisy_t = pose.copy(), transl.copy()
    noisy_pose[bad] = 2.5
    noisy_t[bad] = 9.0
    p2, t2 = md._interpolate_frames(noisy_pose, noisy_t, bad)
    assert np.allclose(t2, transl) and np.allclose(p2[:, 3:6], pose[:, 3:6], atol=1e-9)
    assert np.allclose(Rotation.from_rotvec(p2[:, :3]).magnitude(), 0)


def test_outlier_frames_catch_partial_blowups_and_their_neighbours():
    reproj = np.full(40, 1.9)
    reproj[10] = 12.4  # partial blow-up (seed-5 frame 53 in the review): 6.5x the median, missed by 25 px / 8x
    reproj[[20, 21]] = [300.0, 6.5]  # catastrophic frame + a neighbour at 3.4x the median
    reproj[30] = 6.0  # isolated 3.2x frame: suspect, reported, not repaired
    bad, suspect = md.outlier_frames(reproj, return_suspect=True)
    assert np.flatnonzero(bad).tolist() == [10, 20, 21]
    assert np.flatnonzero(suspect & ~bad).tolist() == [30]


def test_resample_halving_is_per_person():
    t = 0.01 + np.arange(30) / 30.0
    lm, vis, box = _linear_track(t)
    grid = md.master_grid(0.05, 0.9)
    both = md.resample_landmarks(t, lm, vis, box, grid, 1 / 30.0, extra_scale=np.array([2.0, 1.0]))
    one = md.resample_landmarks(t, lm, vis, box, grid, 1 / 30.0, extra_scale=1.0)
    two = md.resample_landmarks(t, lm, vis, box, grid, 1 / 30.0, extra_scale=2.0)
    assert np.allclose(both["landmarks"][:, 0], two["landmarks"][:, 0])
    assert np.allclose(both["landmarks"][:, 1], one["landmarks"][:, 1])
    # ma_3d's rule on what it loads: smallest crop sigma > 15 px -> halved
    lm2 = lm.copy()
    lm2[:, 1, :, 2] = md.sigma_px_to_logvar(np.full(lm.shape[0:1] + lm.shape[2:3], 30.0), box[:, 1:2] * 0 + 512)
    assert md._ma3d_halves(lm2, vis, 1) and not md._ma3d_halves(lm, vis, 0)


def test_nearest_index_picks_the_closer_frame():
    t_src = np.array([0.0, 1 / 30, 2 / 30, 3 / 30])
    q = np.array([-0.01, 0.010, 0.020, 0.048, 0.09, 0.5])
    assert md.nearest_index(t_src, q).tolist() == [0, 0, 1, 1, 3, 3]


def test_link_dir_is_recreated_when_it_points_elsewhere(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "mask_0000_01.png").write_bytes(d.name.encode())
    link = tmp_path / "stage" / "masks"
    md.link_dir(link, a)
    assert (link / "mask_0000_01.png").read_bytes() == b"a"
    assert md.link_dir(link, a).endswith("(current)")
    md.link_dir(link, b)  # e.g. masks/ from an earlier run, now body/
    assert (link / "mask_0000_01.png").read_bytes() == b"b"
    assert (a / "mask_0000_01.png").exists()  # removing a link never touches its old target


def _fake_take(tmp_path, cams=("01", "02"), F=6, world=True):
    work = tmp_path / "work"
    for cam in cams:
        (work / "frames" / cam).mkdir(parents=True)
        (work / "frames" / cam / "frames.json").write_text(json.dumps(
            {"width": 100, "height": 200, "frames": [{"file": f"{i:06d}.jpg", "t": i / 30} for i in range(F)]}))
        body = work / "dancers" / cam / "body"
        body.mkdir(parents=True)
        import cv2
        m = np.zeros((200, 100), np.uint8)
        m[50:150, 20:60] = 255
        for i in range(F):
            for pid in (1, 2):
                cv2.imwrite(str(body / f"mask_{i:04d}_{pid:02d}.png"), m)
        (work / "cameras").mkdir(exist_ok=True)
        np.savez(work / "cameras" / f"{cam}.npz", t=np.arange(F) / 30, K=np.tile(np.eye(3), (F, 1, 1)),
                 R=np.tile(np.eye(3), (F, 1, 1)), tvec=np.zeros((F, 3)), ok=np.ones(F, bool))
    if world:
        (work / "cameras" / "world_from_sfm.json").write_text(json.dumps({"s": 1.0, "R": np.eye(3).tolist(),
                                                                         "t": [0, 0, 0]}))
    return work


def test_world_frame_is_required_unless_explicitly_allowed(tmp_path):
    work = _fake_take(tmp_path, world=False)
    with pytest.raises(SystemExit):
        md.load_tracks(work)
    assert set(md.load_tracks(work, allow_sfm_frame=True)) == {"01", "02"}


def test_ma2d_cache_is_invalidated_when_masks_change(tmp_path, monkeypatch):
    import os
    import time as _time
    work = _fake_take(tmp_path)
    calls = []

    def fake_run(cmd, cwd, log, label):
        cams = cmd[cmd.index("--cam_names") + 1:cmd.index("--mask_path")]
        out = Path(cmd[cmd.index("--out_folder") + 1])
        for c in cams:
            np.savez(out / f"{c}.npz", landmarks=np.zeros((6, 2, 512, 3), np.float32))
        calls.append(list(cams))
        return {"seconds": 0, "gpu_used_peak_mib": 0}

    monkeypatch.setattr(md, "_run_logged", fake_run)
    out = work / "mamma" / "ma_2d_native"
    assert set(md.run_ma2d(work, ["01", "02"], out).values()) == {"computed"}
    assert set(md.run_ma2d(work, ["01", "02"], out).values()) == {"cached"}
    m = work / "dancers" / "02" / "body" / "mask_0003_02.png"
    st = m.stat()
    os.utime(m, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # dancers stage re-ran
    st_ = md.run_ma2d(work, ["01", "02"], out)
    assert st_ == {"01": "cached", "02": "computed"} and calls[-1] == ["02"]
    assert md.run_ma2d(work, ["01"], out, force=True) == {"01": "computed"}
    assert _time is not None


def test_dancecap_config_adds_a_light_temporal_term_to_the_second_run(tmp_path):
    yaml = pytest.importorskip("yaml")
    if not (MAMMA / "optimization" / md.MA3D_CONFIG).exists():
        pytest.skip("no mamma-turing worktree")
    p = md.write_ma3d_config(tmp_path / "cfg.yaml")
    cfg = yaml.safe_load(p.read_text())
    base = yaml.safe_load((MAMMA / "optimization" / md.MA3D_CONFIG).read_text())
    l2 = cfg["optim"]["second_run"]["losses"]
    assert l2["angular_acc_loss"]["weight"] == 0.0005 and l2["acceleration_loss"]["weight"] == 0.0002
    assert l2["reproj_loss"] == base["optim"]["second_run"]["losses"]["reproj_loss"]
    assert cfg["optim"]["third_run"] == base["optim"]["third_run"]
    assert l2["angular_acc_loss"]["weight"] < base["optim"]["third_run"]["losses"]["angular_acc_loss"]["weight"]


def _fit(cams, shared_lead, shared_follow, held=None):
    res = {}
    for c in ("01", "02", "03", "05"):
        if c in cams:
            res[c] = {"lead": {"median_px": shared_lead}, "follow": {"median_px": shared_follow}}
        else:
            res[c] = {"lead": {"median_px": held}, "follow": {"median_px": held}}
    return {"cams": cams, "residuals": res}


def test_choose_fit_prefers_more_cameras_unless_they_conflict():
    c8, c7 = ["01", "02", "03", "05"], ["01", "02", "03"]
    tag, why = md.choose_fit({"c8": _fit(c8, 10.0, 11.0), "c7": _fit(c7, 9.9, 10.8, held=14.0)})
    assert tag == "c8", why
    tag, why = md.choose_fit({"c8": _fit(c8, 13.0, 11.0), "c7": _fit(c7, 10.0, 10.8, held=14.0)})
    assert tag == "c7" and "worsens" in why[0]
    tag, why = md.choose_fit({"c8": _fit(c8, 10.0, 11.0), "c7": _fit(c7, 9.9, 10.8, held=45.0)})
    assert tag == "c7" and "held-out" in why[0]


@pytest.mark.skipif(not (MAMMA / "optimization" / "run_ma_3d.py").exists(), reason="no mamma-turing worktree")
def test_ma3d_fps_from_global_npz_is_opt_in():
    src = (MAMMA / "optimization" / "run_ma_3d.py").read_text(encoding="utf-8")
    start = src.index("def _apply_turing_options")
    end = src.index("\ndef ", start + 10)
    ns: dict = {"np": np}
    exec(compile(src[start:end], "run_ma_3d._apply_turing_options", "exec"), ns)
    f = ns["_apply_turing_options"]

    class Meta:
        files = ["fps"]

        def __getitem__(self, k):
            return np.array(30)

    assert f({}, Meta()).get("fps") is None  # upstream behaviour: no silent dt change
    assert f({"fps_from_metadata": True}, Meta())["fps"] == 30.0
    import types
    assert f({}, Meta(), types.SimpleNamespace(fps=25.0))["fps"] == 25.0
