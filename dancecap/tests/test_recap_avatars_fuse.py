"""Pure parts of the texture fusion / paint-out (recap_avatars_fuse) and of the per-window texture job additions (recap_avatars_pod): no pod, no GPU,
no model files."""
from __future__ import annotations

import json

import numpy as np

from dancecap import recap_avatars_fuse as fz
from dancecap import recap_avatars_pod as pod


def _flat(h, w, rgb):
    return np.broadcast_to(np.asarray(rgb, np.float32), (h, w, 3)).copy()


def test_srgb_round_trip():
    x = np.linspace(0, 1, 11, dtype=np.float32)
    assert np.allclose(fz.lin_to_srgb(fz.srgb_to_lin(x)), x, atol=1e-4)


def test_exposure_gains_recover_a_brightness_offset_and_keep_the_mean():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.1, 0.6, (5000, 3)).astype(np.float32)
    lin = [base, base * np.array([1.2, 1.0, 0.9], np.float32)]
    meas = [np.ones(5000, bool)] * 2
    g = fz.exposure_gains(lin, meas)
    # the two sources end up equal after their gains, and the geometric mean gain is 1
    assert np.allclose(lin[0] * g[0], lin[1] * g[1], atol=2e-3)
    assert np.allclose(np.exp(np.log(g).mean(0)), 1.0, atol=1e-5)


def test_exposure_gains_one_source_is_identity_and_few_texels_are_ignored():
    x = np.full((10, 3), 0.4, np.float32)
    assert np.allclose(fz.exposure_gains([x], [np.ones(10, bool)]), 1.0)
    g = fz.exposure_gains([x, x * 1.5], [np.ones(10, bool)] * 2)       # < min_texels: no estimate, neutral
    assert np.allclose(g, 1.0)


def test_source_quality_orders_measured_over_filled_and_penalises_hair_cover():
    a = np.array([1.0, 1.0, 0.0, 1.0], np.float32)
    conf = np.array([1.0, 1.0, 1.0, 0.1], np.float32)
    ncam = np.array([3, 3, 3, 1])
    cover = np.array([0.0, 1.0, 0.0, 0.0], np.float32)
    q = fz.source_quality(a, conf, ncam, 1.0, cover)
    assert q[0] > q[1] > 0 and q[2] == 0 and 0 < q[3] < q[0]


def test_blur_norm_does_not_bleed_uncovered_texels():
    img = np.zeros((9, 9, 3), np.float32)
    img[:, :4] = 0.8
    w = np.zeros((9, 9), np.float32)
    w[:, :4] = 1.0                      # only the left half is measured; the right half is 0 and must not darken the left edge
    out, ww = fz.blur_norm(img, w, 1.5)
    assert np.allclose(out[:, 3], 0.8, atol=1e-3) and ww[:, 0].max() > 0


def test_fuse_images_drops_the_streak_that_only_one_window_shows():
    h = w = 24
    good = _flat(h, w, [0.5, 0.5, 0.5])
    bad = good.copy()
    bad[8:16, 8:16] = 0.02              # a dark sleeve / hair residue in ONE window
    covered = np.ones((h, w), bool)
    q = [np.ones((h, w), np.float32)] * 3
    fused, st = fz.fuse_images([good, good.copy(), bad], q, covered, low_sigma=1.0, wsigma=1.0, tau=0.10)
    assert abs(float(fused[12, 12, 0]) - 0.5) < 0.05                 # the outlier did not get through
    assert st["outlier_mean_factor"][2] < st["outlier_mean_factor"][0]
    assert np.isfinite(fused).all()


def test_fuse_images_unmeasured_texels_keep_the_filled_colour_not_black():
    h = w = 16
    a = _flat(h, w, [0.3, 0.2, 0.1])
    q = [np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)]  # nothing measured anywhere
    fused, _ = fz.fuse_images([a, a.copy()], q, np.ones((h, w), bool), low_sigma=1.0, wsigma=1.0)
    assert np.allclose(fused, a, atol=1e-3)


def test_fuse_images_prefers_the_lighter_window_on_light_texels():
    h = w = 20
    white = _flat(h, w, [0.8, 0.8, 0.8])
    grey = _flat(h, w, [0.45, 0.45, 0.45])          # darkened by a contaminant, consistently
    q = [np.ones((h, w), np.float32)] * 2
    cov = np.ones((h, w), bool)
    plain, _ = fz.fuse_images([white, grey], q, cov, low_sigma=1.0, wsigma=1.0, tau=0.5)
    light, _ = fz.fuse_images([white, grey], q, cov, low_sigma=1.0, wsigma=1.0, tau=0.5, light_texels=np.ones((h, w), bool))
    assert light.mean() > plain.mean() + 0.02


def test_region_mask_and_grow():
    p = np.array([[0, 0, 0], [0.04, 0, 0], [0.2, 0, 0]], float)
    assert fz.region_mask(p, [0, 0, 0], 0.05).tolist() == [True, True, False]
    m = np.zeros((11, 11), bool)
    m[5, 5] = True
    assert fz.grow_texels(m, 2).sum() > 5 and fz.grow_texels(m, 0).sum() == 1


def test_color_match_gain_scales_the_donor_to_the_target_ring():
    tgt = _flat(10, 10, [0.4, 0.4, 0.4])
    don = _flat(10, 10, [0.8, 0.4, 0.2])
    ring = np.ones((10, 10), bool)
    assert np.allclose(fz.color_match_gain(tgt, don, ring), [0.5, 1.0, 2.0], atol=1e-4)
    assert np.allclose(fz.color_match_gain(tgt, don, np.zeros((10, 10), bool)), 1.0)     # too few ring texels: neutral


def test_paint_one_replaces_only_the_dark_grey_blob_with_the_donor():
    H = W = 64
    img = _flat(H, W, [0.75, 0.72, 0.70])           # light jeans
    img[28:36, 28:36] = 0.05                         # the black pack
    donor = _flat(H, W, [0.74, 0.71, 0.69])
    index = np.arange(H * W, dtype=np.int32).reshape(H, W)   # every pixel is a texel
    tm = np.ones(H * W, bool)                        # the region covers the whole image
    spec = {"name": "pack", "kind": "donor", "luma_max": 0.3, "sat_max": 0.5, "grow_texels": 3, "feather_texels": 3}
    before_far = img[2, 2].copy()
    rep = fz.paint_one(img, spec, tm, index, donor)
    assert rep["texels"] >= 64 and rep["gain"] is not None
    assert img[32, 32].mean() > 0.6                  # the pack is gone
    assert np.allclose(img[2, 2], before_far, atol=1e-3)    # the far jeans are untouched


def test_paint_one_without_a_donor_inpaints_and_nothing_to_paint_is_reported():
    H = W = 48
    img = _flat(H, W, [0.6, 0.6, 0.6])
    img[20:26, 20:26] = 0.0
    index = np.arange(H * W, dtype=np.int32).reshape(H, W)
    tm = np.ones(H * W, bool)
    rep = fz.paint_one(img, {"kind": "inpaint", "luma_max": 0.2, "grow_texels": 2}, tm, index, None)
    assert rep["gain"] is None and img[22, 22].mean() > 0.4
    clean = _flat(H, W, [0.6, 0.6, 0.6])
    assert fz.paint_one(clean, {"luma_max": 0.2}, tm, index, None)["texels"] == 0


def test_agree_fraction():
    x = np.full((100, 3), 0.3, np.float32)
    y = x.copy()
    y[:10] = 0.9
    f = fz.agree_fraction([x, y], [np.ones(100, bool)] * 2, tol=0.1)
    assert 0.0 <= f <= 1.0


# ------------------------------------------------------------------------------------------------ pod additions

def test_mask_camera_frames_switches_off_the_stretch_only(tmp_path):
    (tmp_path / "cameras").mkdir()
    t = np.arange(0, 20, 0.5)
    np.savez(tmp_path / "cameras" / "02.npz", t=t, ok=np.ones(len(t), bool), K=np.zeros((len(t), 3, 3)))
    res = pod.mask_camera_frames(tmp_path, [("02", 5.0, 8.0), ("09", 0.0, 1.0)])
    assert res["02"] == 6 and res["09"] == 0            # frames 5.0 .. 7.5, a camera that is not there is skipped
    with np.load(tmp_path / "cameras" / "02.npz") as z:
        ok = z["ok"]
        assert not ok[(t >= 5.0) & (t < 8.0)].any() and ok[(t < 5.0) | (t >= 8.0)].all()
        assert z["K"].shape == (len(t), 3, 3)           # the other arrays are kept


def test_texture_steps_mask_step_sits_between_stage_and_bake():
    names = [s["name"] for s in pod.texture_steps("tx9", mask="02:242:278")]
    assert names.index("stage") < names.index("maskcams") < names.index("bake")
    assert "maskcams" not in [s["name"] for s in pod.texture_steps("tx9")]
    cmd = next(s for s in pod.texture_steps("tx9", mask="01:327.5:334,02:350:355.5") if s["name"] == "maskcams")["cmd"]
    assert "--mask 01:327.5:334,02:350:355.5" in cmd


def test_save_state_merges_parallel_writers(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, "STATE", tmp_path / "pod_state.json")
    pod.save_state({"pods": {"a": {"x": 1}}, "jobs": {"j1": {"status": "queued"}}, "pulled": []})
    pod.save_state({"pods": {"b": {"x": 2}}, "jobs": {"j2": {"status": "queued"}}, "pulled": ["j2"]})     # a second process that never saw the first
    d = json.loads((tmp_path / "pod_state.json").read_text())
    assert set(d["pods"]) == {"a", "b"} and set(d["jobs"]) == {"j1", "j2"} and d["pulled"] == ["j2"]


# ------------------------------------------------------------------------------------------------ cloth / scalp corrections

class _Atlas:
    """A 8 x 8 atlas of one image row of texels per (y, x): every pixel is a texel; two triangles over four vertices give each texel a face + bary."""
    size = 8

    def __init__(self):
        yy, xx = np.mgrid[0:8, 0:8]
        self.texel_yx = np.stack([yy.ravel(), xx.ravel()], 1)
        self.faces = np.array([[0, 1, 2], [1, 2, 3]])
        self.face = np.zeros(64, int)
        self.bary = np.tile(np.array([[1 / 3, 1 / 3, 1 / 3]]), (64, 1))

    def to_image(self, values, fill=0):
        v = np.asarray(values)
        img = np.full((8, 8) + v.shape[1:], fill, v.dtype)
        img[self.texel_yx[:, 0], self.texel_yx[:, 1]] = v
        return img

    def index_image(self):
        return np.arange(64, dtype=np.int32).reshape(8, 8)


def test_lift_dark_lifts_only_the_darkening_and_only_inside_the_mask():
    img = _flat(40, 40, [0.8, 0.8, 0.8])
    img[18:22, 18:22] = 0.3                        # a dark stain on light cloth
    img[5:8, 5:8] = 0.95                           # a bright highlight
    mask = np.zeros((40, 40), bool)
    mask[:, :30] = True                            # the right part is outside the region
    img[18:22, 34:38] = 0.3                        # a stain outside: untouched
    before = img.copy()
    rep = fz.lift_dark(img, mask, strength=0.6, sigma=8.0, deadband=0.02)
    assert img[20, 20].mean() > before[20, 20].mean() + 0.05
    assert np.allclose(img[6, 6], before[6, 6], atol=1e-3)               # brighter than its surroundings: untouched
    assert np.allclose(img[20, 36], before[20, 36])                      # outside the mask: untouched
    assert rep["mean_lift_pct"] > 0
    assert fz.lift_dark(img, np.zeros((40, 40), bool))["note"].startswith("region too small")


def test_scalp_weight_fades_along_the_mesh_inside_the_boundary():
    at = _Atlas()
    d = np.array([0.0, 0.03, 0.06, -0.05], np.float32)      # per-vertex signed distance, > 0 inside the scalp
    w = fz.scalp_weight(at, {"scalp_dist_v": d}, feather_m=0.03)
    assert w.shape == (64,) and np.allclose(w, 1.0 / 1.0 * np.clip(np.mean([0.0, 0.03, 0.06]) / 0.03, 0, 1))
    assert (w >= 0).all() and (w <= 1).all()


def test_smooth_scalp_evens_out_a_blotchy_scalp_and_reports():
    n = 48

    class Big(_Atlas):
        size = n

        def __init__(self):
            yy, xx = np.mgrid[0:n, 0:n]
            self.texel_yx = np.stack([yy.ravel(), xx.ravel()], 1)
            self.faces = np.array([[0, 1, 2], [1, 2, 3]])
            self.face = np.zeros(n * n, int)
            self.bary = np.tile(np.array([[1 / 3, 1 / 3, 1 / 3]]), (n * n, 1))

        def to_image(self, values, fill=0):
            v = np.asarray(values)
            out = np.full((n, n) + v.shape[1:], fill, v.dtype)
            out[self.texel_yx[:, 0], self.texel_yx[:, 1]] = v
            return out

    rng = np.random.default_rng(1)
    blot = np.kron(0.45 + 0.12 * rng.standard_normal((8, 8, 1)), np.ones((6, 6, 1))) + np.zeros((1, 1, 3))
    img = np.clip(blot, 0, 1).astype(np.float32)
    regions = {"scalp_dist_v": np.full(4, 0.2, np.float32), "prior": np.ones(n * n, bool), "face_core": np.zeros(n * n, bool)}
    std0 = float(img.std())
    rep = fz.smooth_scalp(img, Big(), regions, strength=0.8, sigma=6.0, detail=0.2)
    assert rep["texels"] == n * n and float(img.std()) < 0.6 * std0


def test_apply_paint_runs_a_lift_dark_region_of_the_stage_only():
    at = _Atlas()
    H = W = 8
    img = _flat(H, W, [0.8, 0.8, 0.8])
    regions = {"dom": np.full(64, 3), "head": np.zeros(64, bool)}
    spec = {"regions": [{"role": "follow", "name": "lift", "stage": "final", "kind": "lift_dark", "region": "joints", "joints": [3]}]}
    assert fz.apply_paint(img, "follow", spec, regions, at.index_image(), None, stage=None, atlas=at) == []       # not this stage
    out = fz.apply_paint(img, "follow", spec, regions, at.index_image(), None, stage="final", atlas=at)
    assert out and out[0]["kind"] == "lift_dark"
    assert fz.apply_paint(img, "lead", spec, regions, at.index_image(), None, stage="final", atlas=at) == []       # another role


def test_scalp_weight_survives_unreachable_vertices():
    at = _Atlas()
    d = np.array([-np.inf, 0.05, 0.05, 0.05], np.float32)      # a vertex far outside the scalp: the geodesic field is -inf there
    w = fz.scalp_weight(at, {"scalp_dist_v": d}, feather_m=0.03)
    assert np.isfinite(w).all()
