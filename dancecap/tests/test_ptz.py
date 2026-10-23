"""dancecap.cameras.ptz: a pan / tilt / zoom camera recovered from background matches alone (synthetic)."""
import numpy as np

from dancecap.cameras import ptz as P


def _scene(n=40, seed=3, k1=-0.04, noise=0.4):
    rng = np.random.default_rng(seed)
    lens = P.Lens(1920, 1080, k1=k1)
    t = np.arange(n)
    yaw = np.radians(14 * np.sin(t / 7.0))
    pitch = np.radians(4 * np.sin(t / 5.0 + 1))
    roll = np.radians(0.6 * np.sin(t / 9.0))
    from scipy.spatial.transform import Rotation as Rot
    R = Rot.from_euler("YXZ", np.stack([yaw, pitch, roll], 1)).as_matrix()
    R = np.einsum("nab,bc->nac", R, np.eye(3))
    rv0 = Rot.from_matrix(R @ R[0].T).as_rotvec()          # frame 0 = reference
    f = 1700 * np.exp(0.18 * np.sin(t / 6.0 + 0.5))
    # background directions in the reference frame (frame 0 camera frame): a wide wall of rays
    d = np.stack([rng.uniform(-0.9, 0.9, 6000), rng.uniform(-0.45, 0.45, 6000), np.ones(6000)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Rm = P.rotations(rv0)
    i, j, pp, qq = [], [], [], []
    for a in range(n):
        for b in (a + 1, a + 2, a + 5, a + 11):
            if b >= n:
                continue
            da = d @ Rm[a].T
            db = d @ Rm[b].T
            va = (da[:, 2] > 0.3) & (db[:, 2] > 0.3)
            pa = P.ray_to_pix(da[va], np.full(va.sum(), f[a]), lens)
            pb = P.ray_to_pix(db[va], np.full(va.sum(), f[b]), lens)
            ok = (pa[:, 0] > 0) & (pa[:, 0] < 1920) & (pa[:, 1] > 0) & (pa[:, 1] < 1080) \
                & (pb[:, 0] > 0) & (pb[:, 0] < 1920) & (pb[:, 1] > 0) & (pb[:, 1] < 1080)
            pa, pb = pa[ok][:150], pb[ok][:150]
            if len(pa) < 30:
                continue
            i += [a] * len(pa); j += [b] * len(pa)
            pp.append(pa + rng.normal(0, noise, pa.shape)); qq.append(pb + rng.normal(0, noise, pb.shape))
    pairs = P.Pairs(np.array(i), np.array(j), np.concatenate(pp), np.concatenate(qq))
    # 5 % gross outliers
    bad = rng.random(len(pairs)) < 0.05
    pairs.q[bad] += rng.uniform(-80, 80, (bad.sum(), 2))
    return lens, rv0, f, pairs


def test_ray_roundtrip_and_distortion():
    lens = P.Lens(1920, 1080, k1=-0.08, k2=0.01, ox=5, oy=-3)
    p = np.array([[100.0, 80.0], [960, 540], [1800, 1000]])
    d = P.pix_to_ray(p, np.full(3, 1500.0), lens)
    assert np.allclose(P.ray_to_pix(d, np.full(3, 1500.0), lens), p, atol=1e-6)


def test_ptz_recovers_pan_tilt_zoom_and_lens():
    lens, rv0, f, pairs = _scene()
    edges = {}
    for a, b in {(int(x), int(y)) for x, y in zip(pairs.i, pairs.j) if y == x + 1}:
        s = (pairs.i == a) & (pairs.j == b)
        H, m = P.homography_edge(pairs.p[s], pairs.q[s], 4.0)
        edges[(a, b)] = H
    n = 40
    guess = P.Lens(1920, 1080)
    rv_i, f_i, order = P.init_chain(n, edges, guess, 1400.0)       # 18 % wrong start focal
    assert len(order) == n
    sol = P.solve_ptz(n, pairs, rv_i, f_i, guess, free=("k1",))
    keep = P.reject_outliers(sol, pairs, 6.0)
    pairs2 = P.Pairs(pairs.i[keep], pairs.j[keep], pairs.p[keep], pairs.q[keep])
    sol = P.solve_ptz(n, pairs2, sol.rv, sol.f, sol.lens, free=("k1",))
    assert sol.rms_px < 0.8
    assert abs(sol.lens.k1 - lens.k1) < 0.01
    # the focal scale is a global gauge only through the matches: relative error small
    assert np.max(np.abs(sol.f / f - 1)) < 0.03
    R = sol.R
    Rt = P.rotations(rv0)
    ang = [np.degrees(np.arccos(np.clip((np.trace(R[k] @ Rt[k].T) - 1) / 2, -1, 1))) for k in range(n)]
    assert max(ang) < 0.3


def test_refine_frame_matches_truth():
    lens, rv0, f, pairs = _scene(n=12)
    sol = P.PtzSolution(rv0.copy(), f.copy(), lens)
    # take frame 6's matches to frame 7 (frame 6 as 'new'), reference frame 7 and 8
    rng = np.random.default_rng(1)
    d = np.stack([rng.uniform(-0.6, 0.6, 400), rng.uniform(-0.3, 0.3, 400), np.ones(400)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Rm = P.rotations(rv0)
    p_new, q_ref = [], []
    for b in (7, 8):
        pa = P.ray_to_pix(d @ Rm[6].T, np.full(400, f[6]), lens)
        pb = P.ray_to_pix(d @ Rm[b].T, np.full(400, f[b]), lens)
        p_new.append(pa + rng.normal(0, 0.3, pa.shape)); q_ref.append(pb + rng.normal(0, 0.3, pb.shape))
    rv, fn, rms = P.refine_frame(sol, [7, 8], p_new, q_ref, rv0[6] + 0.01, f[6] * 1.05)
    assert abs(fn / f[6] - 1) < 0.005
    assert np.linalg.norm(rv - rv0[6]) < 0.002
    assert rms < 1.0


def test_tied_focal_and_refine_tracks_roundtrip():
    lens, rv0, f, pairs = _scene(n=24, seed=5)
    f_const = np.full(24, 1700.0)
    # same scene but with a constant focal: regenerate matches with f = 1700
    rng = np.random.default_rng(2)
    Rm = P.rotations(rv0)
    d = np.stack([rng.uniform(-0.8, 0.8, 4000), rng.uniform(-0.4, 0.4, 4000), np.ones(4000)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    i, j, pp, qq = [], [], [], []
    for a in range(24):
        for b in (a + 1, a + 3, a + 7):
            if b >= 24:
                continue
            pa = P.ray_to_pix(d @ Rm[a].T, f_const, lens) if False else P.ray_to_pix(d @ Rm[a].T, np.full(4000, 1700.0), lens)
            pb = P.ray_to_pix(d @ Rm[b].T, np.full(4000, 1700.0), lens)
            ok = (pa[:, 0] > 0) & (pa[:, 0] < 1920) & (pa[:, 1] > 0) & (pa[:, 1] < 1080) & (pb[:, 0] > 0) & (pb[:, 0] < 1920) & (pb[:, 1] > 0) & (pb[:, 1] < 1080)
            pa, pb = pa[ok][:120], pb[ok][:120]
            i += [a] * len(pa); j += [b] * len(pa); pp.append(pa + rng.normal(0, .3, pa.shape)); qq.append(pb + rng.normal(0, .3, pb.shape))
    pr = P.Pairs(np.array(i), np.array(j), np.concatenate(pp), np.concatenate(qq))
    sol = P.solve_ptz(24, pr, rv0 * 1.1, np.full(24, 1700.0 * 1.1), lens, free=(), tie_f=True, fix_f=False, max_nfev=80)
    assert np.ptp(sol.f) < 1e-6                      # one focal for all frames
    assert abs(sol.f[0] / 1700 - 1) < 0.02
    solf = P.solve_ptz(24, pr, rv0 * 1.0, np.full(24, 1700.0), lens, free=(), tie_f=True, fix_f=True, max_nfev=30)
    assert abs(solf.f[0] - 1700) < 1e-9 and solf.median_px < 0.6


def test_refine_frame_robust_drops_a_bad_reference():
    lens, rv0, f, pairs = _scene(n=12)
    rng = np.random.default_rng(3)
    Rm = P.rotations(rv0)
    d = np.stack([rng.uniform(-0.6, 0.6, 400), rng.uniform(-0.3, 0.3, 400), np.ones(400)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rv_bad = rv0.copy()
    rv_bad[7] = rv0[7] + np.array([0.0, np.radians(0.25), 0.0])         # one reference frame is 0.25 deg off (~13 px at f = 3000)
    sol = P.PtzSolution(rv_bad, f.copy(), lens)
    p_new, q_ref = [], []
    for b in (7, 8, 9):
        pa = P.ray_to_pix(d @ Rm[6].T, np.full(400, f[6]), lens)
        pb = P.ray_to_pix(d @ Rm[b].T, np.full(400, f[b]), lens)
        p_new.append(pa + rng.normal(0, .3, pa.shape)); q_ref.append(pb + rng.normal(0, .3, pb.shape))
    rv_plain, f_plain, rms_plain = P.refine_frame(sol, [7, 8, 9], p_new, q_ref, rv0[6], f[6])
    rv, fn, rms, used = P.refine_frame_robust(sol, [7, 8, 9], p_new, q_ref, rv0[6], f[6], fix_f=f[6], bad_px=1.0)
    assert 7 not in used and len(used) == 2
    assert np.linalg.norm(rv - rv0[6]) < np.linalg.norm(rv_plain - rv0[6])
    assert np.linalg.norm(rv - rv0[6]) < 0.0008


def test_best_root_skips_a_badly_linked_first_frame():
    from dancecap.cameras import ptz as P
    # frame 0 links only to 1 (a fade from black); 2..9 form the well-linked chain
    edges = {(0, 1): None}
    for i in range(2, 9):
        edges[(i, i + 1)] = None
        if i + 2 <= 9:
            edges[(i, i + 2)] = None
    r = P.best_root(10, edges)
    assert 2 <= r <= 9
    full = {(i, i + 1): None for i in range(9)}
    assert P.best_root(10, full) in (1, 2, 3, 4, 5, 6, 7, 8)   # interior frames have the most edges
    assert P.best_root(1, {}) == 0
