"""Tiny synthetic checks for the MANO-regressor hand route (hand_mano_worker, refine_hands mano/fusion, hands_pod)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dancecap import hand_mano_worker as hmw
from dancecap import refine_hands as rh


def test_mirror_is_the_smplx_left_right_convention():
    aa = np.array([[0.3, -0.2, 0.5], [0.1, 0.4, -0.7]])
    R = Rotation.from_rotvec(aa).as_matrix()
    got = hmw.rotmat_to_aa(hmw.mirror_rotmats(R))
    assert np.allclose(got, aa * [1, -1, -1], atol=1e-9)
    assert np.allclose(hmw.mirror_rotmats(hmw.mirror_rotmats(R)), R)


def test_crop_keypoints_to_image_right_and_left():
    kp = np.array([[[0.25, -0.1], [0.0, 0.0]], [[0.25, -0.1], [0.0, 0.0]]])
    center = np.array([[100.0, 50.0], [100.0, 50.0]])
    size = np.array([40.0, 40.0])
    out = hmw.crop_kp_to_image(kp, center, size, np.array([1.0, 0.0]))
    assert np.allclose(out[0, 0], [110.0, 46.0])       # right: x kept
    assert np.allclose(out[1, 0], [90.0, 46.0])        # left (mirrored crop): x flipped back
    assert np.allclose(out[:, 1], [[100.0, 50.0]] * 2)


def test_tight_box_is_square_and_padded():
    px = np.array([[10.0, 20.0], [30.0, 25.0]])
    b = hmw.tight_box(px, pad=0.1)
    assert np.isclose(b[2] - b[0], b[3] - b[1]) and np.isclose(b[2] - b[0], 24.0)
    assert np.allclose([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], [20.0, 22.5])


def test_unchumpy_select_and_ch():
    ch = hmw._Stub()
    ch.__setstate__({"x": np.arange(6.0)})
    sel = hmw._Stub()
    sel.__setstate__({"a": ch, "idxs": np.array([5, 4, 3, 2, 1, 0]), "preferred_shape": (3, 2)})
    assert np.allclose(hmw._unchumpy(ch), np.arange(6.0))
    assert np.allclose(hmw._unchumpy(sel), np.arange(6.0)[::-1].reshape(3, 2))
    assert hmw._unchumpy(7) == 7


def test_robust_rotation_mean_ignores_an_outlier_view():
    rng = np.random.default_rng(1)
    truth = Rotation.from_rotvec(rng.normal(0, 0.4, (15, 3))).as_matrix()
    views = []
    for _ in range(5):
        noise = Rotation.from_rotvec(rng.normal(0, np.radians(3), (15, 3))).as_matrix()
        views.append(noise @ truth)
    views.append(Rotation.from_rotvec(rng.normal(0, 1.5, (15, 3))).as_matrix())   # one wild view
    M, spread = rh.robust_rotation_mean(np.stack(views), np.ones(6))
    err = rh._geo_deg(M, truth)
    assert err.max() < 6.0, err
    assert np.all(spread < 15.0)
    M1, s1 = rh.robust_rotation_mean(np.stack(views[:1]), np.ones(1))
    assert np.allclose(M1, views[0], atol=1e-6) and np.isnan(s1).all()


def test_smooth_rotvecs_keeps_constant_and_marks_gaps():
    T = 40
    th = np.tile(np.array([0.2, -0.1, 0.3])[None, None], (T, 15, 1))
    conf = np.ones(T)
    th[10:30] = np.nan
    conf[10:30] = 0
    out, cs = rh.smooth_rotvecs(th, conf, 1.5)
    ok = np.isfinite(out).all((1, 2))
    assert np.allclose(out[ok], th[0])
    assert not ok[15:25].any() and ok[:10].all() and ok[30:].all()
    assert cs[0] > 0.5 and cs[20] == 0


def test_mano_target_holdout_fallback():
    T = 5
    mz = {"lead_left_theta": np.zeros((T, 15, 3)), "lead_left_conf": np.full(T, 2.0),
          "lead_left_theta_wo_01": np.ones((T, 15, 3)), "lead_left_conf_wo_01": np.full(T, 0.5)}
    th, w = rh.mano_target(mz, "lead", "left")
    assert np.all(th == 0) and np.all(w == 1.0)          # confidence clipped to 1
    th, w = rh.mano_target(mz, "lead", "left", holdout="01")
    assert np.all(th == 1) and np.all(w == 0.5)
    th, w = rh.mano_target(mz, "lead", "left", holdout="09")   # no per-camera fusion stored -> full fusion
    assert np.all(th == 0)
    assert rh.mano_target(None, "lead", "left") is None


def test_hands_pod_spec_is_secure_and_private():
    from dancecap import hands_pod, remote_jobs as rj
    from dancecap.take import PACKAGE_ROOT
    if not (PACKAGE_ROOT / "work" / "larissa_kadu" / "smplx" / "lead.npz").exists():
        pytest.skip("needs the larissa_kadu fit")
    d = hands_pod.spec("larissa_kadu")
    job = rj.parse_job(d, "test")
    assert hands_pod.CLOUDS == ["SECURE"]
    pods = [p for _, p in job.inputs]
    assert "models/mano/nochumpy/MANO_RIGHT.pkl" in pods
    assert not any("MANO_LEFT" in p or p.startswith("models/flame") for p in pods)
    assert all(o.startswith("dancecap/work/larissa_kadu/") for o in job.outputs)
    names = [c.name for c in job.commands]
    assert names.index("kp2d") < names.index("detectors") < names.index("mano") < names.index("fingers_hybrid") \
        < names.index("contact")
    assert not any(c.optional for c in job.commands if c.name in ("setup_detectors", "kp2d"))


def test_load_take_falls_back_without_alignment_files(monkeypatch):
    def boom(name):
        raise FileNotFoundError("prototype/<take>/alignment.json (not on a pod)")
    monkeypatch.setattr(rh.Take, "load", staticmethod(boom))
    t = rh.load_take("some_take")
    assert t.name == "some_take" and t.work.name == "some_take" and t.work.parent.name == "work"


def test_npz_loads_everything_into_memory(tmp_path):
    p = tmp_path / "a.npz"
    np.savez_compressed(p, x=np.arange(3), y=np.ones((2, 2)))
    d = rh._npz(p)
    assert isinstance(d, dict) and set(d) == {"x", "y"} and d["x"].sum() == 3
