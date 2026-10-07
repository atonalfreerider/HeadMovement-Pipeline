"""Unit tests for dancecap.sync (synthetic moving cameras, no GPU, no MAMMA).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_sync.py -q      (from the workspace root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import mamma_dyn as md  # noqa: E402
from dancecap import sync  # noqa: E402


def test_weighted_dlt_is_exact_and_ignores_zero_weight_views():
    rng = np.random.default_rng(0)
    t = np.zeros(1)
    rig = md.synthetic_rig(4, t, seed=1)
    X = rng.normal(0, 0.4, (50, 3)) + [0, 1, 0]
    uvn, w, P = [], [], []
    for k, cam in enumerate(rig):
        px, _ = md.project(X[None], cam["K"], cam["R"], cam["t"])
        if k == 3:
            px = px + 300.0  # garbage, but weight 0
        uvn.append(sync._normalise(px[0], np.repeat(cam["K"], 50, 0)))
        w.append(np.full(50, 0.0 if k == 3 else 1.0))
        P.append(np.repeat(np.concatenate([cam["R"], cam["t"][..., None]], -1), 50, 0))
    Xh, n = sync.triangulate(np.stack(uvn), np.stack(w), np.stack(P))
    assert np.allclose(Xh, X, atol=1e-6) and (n == 3).all()
    w2 = np.stack(w)
    w2[1:] = 0
    Xn, n1 = sync.triangulate(np.stack(uvn), w2, np.stack(P))
    assert np.isnan(Xn).all() and (n1 == 1).all()


def _scene(true_offsets_ms: dict[str, float], frames: int = 75, n_pts: int = 24, noise_px: float = 0.3, seed: int = 0):
    """Two 'dancers' (point clouds translating + spinning + waving) seen by handheld zooming cameras whose
    reported frame times are late/early by true_offsets_ms (reported t = true t - delta)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, [0.2, 0.45, 0.15], (2, n_pts, 3)) + [0, 1.0, 0]
    base[1, :, 0] += 0.5
    amp = rng.uniform(0.05, 0.35, (2, n_pts))

    def pts(t):  # (T, 2, N, 3) world points at true times t
        tt = np.asarray(t, float)[:, None, None]  # (T, 1, 1)
        ang = 1.6 * np.sin(2 * np.pi * 0.6 * tt)
        c, s = np.cos(ang), np.sin(ang)
        x, y, z = base[..., 0], base[..., 1], base[..., 2]  # (2, N)
        wx = amp * np.sin(2 * np.pi * 2.1 * tt + 3 * x)
        wy = amp * np.cos(2 * np.pi * 1.7 * tt + 5 * y)
        X = np.stack([c * x + s * z + wx + 0.4 * np.sin(2 * np.pi * 0.4 * tt), y + wy + 0 * tt,
                      -s * x + c * z], -1)
        return X

    data = {}
    idx = np.arange(n_pts)
    names = sorted(true_offsets_ms)
    for k, cam in enumerate(names):
        d = true_offsets_ms[cam] / 1000.0
        t_true = 0.3 + k * 0.009 + np.arange(frames) / 30.0
        rig = md.synthetic_rig(len(names), t_true, seed=3)[k]
        X = pts(t_true)
        px, z = md.project(X, rig["K"][:, None], rig["R"][:, None], rig["t"][:, None])
        px = px + rng.normal(0, noise_px, px.shape)
        box = np.full((frames, 2), 600.0)
        lv = md.sigma_px_to_logvar(np.full(px.shape[:3], 2.0), box[..., None])
        lm = np.concatenate([px, lv[..., None]], -1)
        tr = md.CameraTrack(name=cam, t=t_true - d, K=rig["K"], R=rig["R"], tvec=rig["t"], ok=np.ones(frames, bool),
                            k1=np.zeros(frames), width=rig["w"], height=rig["h"], files=[])
        obs = {"lm": lm, "vis": np.ones(px.shape[:3]), "box": box}
        data[cam] = sync.CamData(tr, obs, idx)
    return data


def test_offsets_are_recovered_up_to_the_gauge():
    true = {"01": 0.0, "02": 17.0, "03": -11.0, "04": 6.0, "05": -3.0}
    data = _scene(true, frames=130)  # 4.3 s -> two 2 s sub-windows for the stability check
    res = sync.estimate_data(data, range_ms=32.0, log=lambda *a: None)
    est = np.array([res["offsets"][c]["delta_ms"] for c in sorted(true)])
    tru = np.array([true[c] for c in sorted(true)])
    assert abs(np.median(est)) < 1e-6  # gauge
    assert np.abs((est - np.median(est)) - (tru - np.median(tru))).max() < 1.5, (est, tru)
    for c in true:
        o = res["offsets"][c]
        assert o["stable"] and o["spread_ms"] < sync.STABLE_MS
        assert o["residual_px"]["at_delta"] <= o["residual_px"]["at_zero"] + 1e-6


def test_residual_grows_away_from_the_true_offset():
    data = _scene({"01": 0.0, "02": 0.0, "03": 25.0, "04": 0.0})
    others = [data[c] for c in ("01", "02", "04")]
    r = {d: sync.residual(data["03"], others, {}, d / 1000.0)["median_px"] for d in (-25.0, 0.0, 25.0, 50.0)}
    assert r[25.0] < r[0.0] < r[-25.0] and r[25.0] < r[50.0]
    assert r[25.0] < 2.0 and r[25.0] < 0.5 * r[0.0]  # noise + linear-interpolation error of the others


def test_mamma_reads_sync_offsets_with_optional_drift(tmp_path):
    (tmp_path / "sync").mkdir()
    (tmp_path / "sync" / "offsets.json").write_text(
        '{"offsets": {"01": {"delta": 0.01}, "02": {"delta": -0.02, "rate": 0.004, "t_ref": 58.0, "use_rate": true},'
        ' "03": {"delta": 0.05, "use": false}, "04": 0.003}}')
    off = md.read_sync_offsets(tmp_path)
    t = np.array([56.0, 58.0, 60.0])
    assert np.allclose(md.apply_offset(t, off["01"]), t + 0.01)
    assert np.allclose(md.apply_offset(t, off["02"]), t - 0.02 + 0.004 * (t - 58.0))
    assert np.allclose(md.apply_offset(t, off["03"]), t)  # not trusted -> no correction
    assert np.allclose(md.apply_offset(t, off["04"]), t + 0.003)
