r"""dancecap.mono4d.recap_splat_hair take profiles (2026-10-23): the recap's grey fringe rule kept, the black-hair rule for
carlos_aline (Aline's near-black hair, red top), profile paths. Synthetic data only.

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_recap_splat_hair_profile.py -q -p no:cacheprovider
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.mono4d import recap_splat_hair as RH  # noqa: E402
from dancecap.mono4d import splat_hair as SH  # noqa: E402

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _recap_after():
    RH.use(None)
    yield
    RH.use(None)


def test_recap_profile_paths_are_the_old_constants():
    hp = RH.use(None)
    assert (hp.take, hp.capture, hp.fringe) == ("larissa_kadu_recap", "06_LarissaKaduRecap", "grey")
    assert hp.src == RH.SRC and hp.out == RH.OUT and hp.unity_dir == RH.UNITY_DIR and hp.groom == RH.GROOM
    assert RH.use("recap") == RH.HAIR_PROFILES["larissa_kadu_recap"]


def test_carlos_aline_profile_paths():
    hp = RH.use("carlos_aline")
    assert hp.fringe == "black"
    assert hp.src == PKG / "work" / "carlos_aline" / "lhmpp"
    assert hp.out == PKG / "work" / "carlos_aline" / "splat_hair_unity"
    assert hp.unity_dir.parts[-3:] == ("StreamingAssets", "07_CarlosAline", "splats")
    assert hp.groom.parts[-2:] == ("07_CarlosAline", "hair_groom.json")
    with pytest.raises(SystemExit):
        RH.use("nope")
    assert RH.use("carlos_aline", out_dir="X").out == Path("X")


LAB = np.array([[20.0, 10.0, 5.0],     # 0 near-black hair, warm cast
                [32.0, 3.0, 2.0],      # 1 lit black hair (grey by the recap rule: a* < 15, L* > 15)
                [70.0, 2.0, 1.0],      # 2 white fringe
                [45.0, 55.0, 30.0],    # 3 red top under the hair
                [60.0, 12.0, 18.0],    # 4 skin
                [33.0, 40.0, 25.0]])   # 5 red hair (the recap)


def test_grey_rule_is_the_recap_rule():
    free = np.ones(6, bool)
    m, c = RH.fringe_mask(LAB, free, None, "grey")
    light = LAB[:, 0] > RH.BRIGHT_L
    grey = (LAB[:, 1] < RH.GREY_A) & (LAB[:, 0] > RH.GREY_DARK_L)
    assert np.array_equal(m, free & (light | grey))
    assert list(m) == [True, True, True, False, True, False]    # black hair (0, 1) kept off the strands; red (3, 5) bound
    assert c == {"free_light_fringe_kept_lbs": 2, "free_grey_kept_lbs": 2}       # the skin (4) is lighter than BRIGHT_L


def test_black_rule_uses_the_learned_hair_colour():
    free = np.array([True, True, True, True, True, False])
    hairc = np.array([True, True, False, False, False, True])   # select_hair's model: 0, 1 are hair-coloured
    m, c = RH.fringe_mask(LAB, free, hairc, "black")
    assert list(m) == [False, False, True, True, True, False]   # lit black hair (1) is bound now; top / skin / fringe stay
    assert c == {"free_light_fringe_kept_lbs": 2, "free_not_hair_coloured_kept_lbs": 1}
    with pytest.raises(ValueError):
        RH.fringe_mask(LAB, free, None, "black")
    with pytest.raises(ValueError):
        RH.fringe_mask(LAB, free, hairc, "pink")


def test_cli_select_and_bind_accept_take(monkeypatch):
    seen = {}
    monkeypatch.setattr(RH, "select_report", lambda: seen.setdefault("select", RH.HP.take))
    monkeypatch.setattr(RH, "run_bind", lambda dump, install=False: seen.setdefault("bind", (RH.HP.take, Path(dump).name, install)))
    RH.main(["select", "--take", "carlos_aline"])
    RH.main(["bind", "--take", "carlos_aline", "--dump", "d.json", "--install"])
    assert seen == {"select": "carlos_aline", "bind": ("carlos_aline", "d.json", True)}


def test_finish_is_one_command(tmp_path, monkeypatch):
    RH.use("carlos_aline", out_dir=str(tmp_path / "out"))
    dump = tmp_path / "hair_dump.json"
    dump.write_text("{}")
    calls = []
    monkeypatch.setattr(RH, "run_bind", lambda d, install=False: calls.append(("bind", Path(d).name, install)) or
                        {"bound": 5, "out": ["x"], "installed": ["y"], "binding_info": {"guides": 160}})
    monkeypatch.setattr(RH, "verify", lambda: calls.append(("verify",)) or
                        {"sha256_ok": True, "bound": 5, "in_range": True, "unbound_zero": True})
    monkeypatch.setattr(RH, "sheet", lambda dumps=None, device="cpu": calls.append(("sheet", device)) or [])
    out = RH.finish(dump)
    assert calls == [("bind", "unity_hair_dump_frame0.json", True), ("verify",), ("sheet", "cpu")]
    assert (tmp_path / "out" / "unity_hair_dump_frame0.json").read_text() == "{}"
    assert out["ok"] and (tmp_path / "out" / "finish.json").exists()
    assert out["next_quest_redistill"][0] == ("python -m dancecap.mono4d.quest_distill prepare --lessons carlos_aline "
                                              "--roles follow --work work/quest_distill_ca_hair")


# ---------------------------------------------------------------- the black-hair look fixes (fix round 1, 2026-10-23)
def test_profiles_carry_the_look_only_for_black_hair():
    assert RH.use(None).look is None                       # the recap: no look fix (its outputs stay byte-identical)
    assert RH.use("carlos_aline").look == RH.HAIR_LOOK_BLACK


def _rq(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_needle_shape_aligns_with_the_strand_and_keeps_the_face_area():
    rng = np.random.default_rng(3)
    n = 400
    q = _rq(rng, n)
    S = np.exp(rng.uniform(np.log(0.001), np.log(0.03), (n, 3)))
    t = rng.normal(size=(n, 3))
    q2, ls2 = RH.needle_shape(q, np.log(S), t, 0.003, 0.05)
    R2 = RH.q_to_mat(q2)
    tu = t / np.linalg.norm(t, axis=1, keepdims=True)
    assert np.allclose(np.abs((R2[:, :, 0] * tu).sum(1)), 1, atol=1e-6)            # local x = the strand tangent
    assert np.allclose(np.linalg.det(R2), 1, atol=1e-6)
    S2 = np.exp(ls2)
    Ss = np.sort(S, 1)
    assert np.all(S2[:, 1] <= 0.003 + 1e-9) and np.all(S2[:, 2] <= 0.003 + 1e-9)    # thin across
    assert np.all(S2[:, 1] <= Ss[:, 1] + 1e-12) and np.all(S2[:, 2] <= Ss[:, 0] + 1e-12)  # never thicker than it was
    assert np.all(S2[:, 0] >= Ss[:, 2] - 1e-12)                                      # never shorter than it was
    assert np.all(S2[:, 0] <= np.maximum(0.05, Ss[:, 2]) + 1e-12)                   # along clamped (unless it was longer)
    keep = Ss[:, 2] * Ss[:, 1] / np.minimum(Ss[:, 1], 0.003) <= 0.05
    assert np.allclose((S2[:, 0] * S2[:, 1])[keep], (Ss[:, 2] * Ss[:, 1])[keep], rtol=1e-9)   # largest face area kept
    thin = Ss[:, 1] <= 0.003                                                           # already thin: the same sigmas
    assert np.allclose(np.sort(S2[thin], 1), Ss[thin])


def test_needle_shape_keeps_the_old_middle_axis_across():
    q = np.array([[0.0, 0.0, 0.0, 1.0]])                    # identity: axes x, y, z
    S = np.array([[0.02, 0.002, 0.001]])                    # longest along x, middle along y
    q2, ls2 = RH.needle_shape(q, np.log(S), np.array([[0.0, 0.0, 1.0]]), 0.003, 0.05)   # strand along z
    R2 = RH.q_to_mat(q2)[0]
    assert np.allclose(R2[:, 0], [0, 0, 1]) and np.allclose(np.abs(R2[:, 1]), [0, 1, 0], atol=1e-9)
    assert np.allclose(np.exp(ls2[0]), [0.02, 0.002, 0.001])


def test_stack_cap_limits_the_composite_of_near_copies():
    c = np.array([[0, 0, 0], [0.0005, 0, 0], [0.001, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.2005, 0, 0]], float)
    a = np.array([0.69, 0.69, 0.30, 0.9, 0.1, 0.1])
    a2, info = RH.stack_cap(c, a, 0.25, 0.002)
    assert abs((1 - np.prod(1 - a2[:3])) - 0.25) < 1e-9     # the stack of three: composite = the cap
    r = np.log1p(-a2[:3]) / np.log1p(-a[:3])
    assert np.allclose(r, r[0])                             # members keep their ratios
    assert abs(a2[3] - 0.25) < 1e-9                         # a single opaque Gaussian is a group of one
    assert np.allclose(a2[4:], a[4:])                       # under the cap: unchanged
    assert info["groups"] == 3 and info["groups_capped"] == 2 and info["gaussians_capped"] == 4


def test_paint_mask_takes_dark_achromatic_back_paint_only():
    J = np.zeros((55, 3))
    J[3, 1], J[9], J[12, 1] = 0.0, [0.0, 0.2, 0.0], 0.4
    mean = np.array([[0, 0.2, 0.1]] * 7 + [[0, -0.1, 0.1], [0.3, 0.2, 0.1], [0, 0.5, 0.1]], float)
    nz = np.full(10, -0.9)
    nz[1] = 0.9                                             # 1 faces front
    lab = np.array([[6.0, 1.0, 1.0]] * 10)
    lab[2] = [18, 30, 17]                                   # 2 the top's shaded red (C* > 12)
    lab[3] = [30, 2, 2]                                     # 3 lit grey: not dark, not hair-coloured
    unbound = np.ones(10, bool)
    unbound[4] = False                                      # 4 bound hair
    rigid = np.zeros(10, bool)
    rigid[5] = True                                         # 5 rigid scalp
    hairc = np.zeros(10, bool)
    hairc[6] = True
    lab[6] = [30, 2, 2]                                     # 6 lit but hair-coloured: paint
    m = RH.paint_mask(mean, J, nz, lab, unbound, rigid, hairc, 0.03)
    assert list(m) == [True, False, False, False, False, False, True, False, False, False]   # 7 low, 8 side, 9 high


def test_crown_tone_darkens_the_crown_outside_the_face_box_and_keeps_hue():
    n = 8
    lab = np.array([[30.0, 4.0, 3.0]] * n)
    lab[7] = [5.0, 1.0, 1.0]
    op = np.ones(n)
    el = np.array([60.0, 60.0, 20.0, 5.0, 60.0, 60.0, 60.0, -10.0])
    az = np.array([180.0, 0.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0])
    hairline = np.full(n, 70.0)                             # 1: in front, under the hairline + 6 = the face box
    lab[5] = [30.0, 20.0, 10.0]                             # 5: chroma 22 > 15 (skin / top): kept
    cand = np.array([True, True, True, True, False, True, True, False])   # 4: not a candidate
    bound = np.zeros(n, bool)
    bound[7] = True                                         # the hanging hair: L* 5
    out, w, info = RH.crown_tone(lab, op, el, az, hairline, cand, bound, RH.HAIR_LOOK_BLACK["crown"])
    assert w[0] == 1 and w[6] == 1 and w[1] == 0 and w[3] == 0 and w[4] == 0 and w[5] == 0 and 0 < w[2] < 1
    assert out[0, 0] <= 5.0 + 1e-6 and out[0, 0] < lab[0, 0]
    assert np.allclose(out[[1, 3, 4, 5, 7]], lab[[1, 3, 4, 5, 7]])
    assert np.allclose(np.arctan2(out[[0, 2], 2], out[[0, 2], 1]), np.arctan2(lab[[0, 2], 2], lab[[0, 2], 1]))  # hue kept
    assert lab[2, 0] > out[2, 0] > out[0, 0]                # the ramp: partly darkened at 20 deg
    assert info["crown"] == 3                               # 0, 2, 6


def test_apply_look_keeps_the_binding():
    rng = np.random.default_rng(5)
    G, P = 3, 6
    rest = np.zeros((G, P, 3))
    for g in range(G):
        for i in range(P):
            rest[g, i] = [0.03 * (g - 1), 0.0 - 0.04 * i, 0.04 + 0.01 * i]
    n = 60
    out = np.zeros(n, RH.RECORD)
    c = rest[rng.integers(0, G, n), rng.integers(1, P - 1, n)] + rng.uniform(-0.005, 0.005, (n, 3))
    out["mean"] = c
    out["rotation"] = _rq(rng, n)
    out["log_scale"] = np.log(rng.uniform(0.004, 0.02, (n, 3)))
    out["opacity"] = rng.normal(1.0, 1.0, n)
    out["sh0"] = rng.normal(-1.5, 0.1, (n, 3))
    out["hair"] = 1
    out["hair"][50:] = 0                                    # 50..59: head Gaussians not flagged as hair
    idx = np.arange(40)
    q = out["rotation"][idx].astype(float)
    b = RH.bind(c[idx], q / np.linalg.norm(q, axis=1, keepdims=True), rest)
    out["strand"] = -1
    out["segment"] = -1
    out["strand"][idx], out["segment"][idx] = b["strand"], b["segment"]
    out["local_offset"][idx] = b["local_offset"]
    before = out.copy()
    f = {"normal": np.tile([0.0, 0.0, -1.0], (n, 1)), "sd": np.zeros(n), "dom": np.full(n, 15), "el": np.full(n, 60.0),
         "az": np.full(n, 180.0), "dh": np.zeros(n)}
    f["dh"][55:] = 0.1                                      # 55..59 off the scalp shell
    sel = {"features": f, "hair_colour": np.ones(n, bool)}
    J = np.zeros((55, 3))
    J[3, 1], J[12, 1] = -1.0, -0.9                          # paint zone far below: nothing painted here
    lab = np.tile([25.0, 1.0, 1.0], (n, 1))
    q2, info = RH.apply_look(out, idx, b, rest, J, lab, np.zeros(n, bool), sel, RH.HAIR_LOOK_BLACK)
    for k in ("mean", "strand", "segment", "local_offset", "hair"):
        assert np.array_equal(out[k], before[k]), k
    F = RH.segment_frames(rest)[b["strand"], b["segment"]]
    qb = RH.q_mul(RH.q_from_mat(F), out["local_rotation"][idx].astype(float))
    assert np.allclose(np.abs((qb * q2).sum(1)), 1, atol=1e-5)                    # local_rotation encodes the new rotation
    assert info["large_opaque_bound"]["after"] == 0
    assert np.all(1 / (1 + np.exp(-out["opacity"][idx].astype(float))) <= 0.25 + 1e-6)
    assert np.array_equal(out["opacity"][40:], before["opacity"][40:])            # unbound, not paint: untouched opacity
    assert info["crown"]["crown"] == 15                    # unbound hair 40..49 + the scalp shell's non-hair 50..54
    assert info["crown"]["not_hair_flagged"] == 5
    assert np.array_equal(out["sh0"][55:], before["sh0"][55:])


# ---------------------------------------------------------------- fix round 2: the layer the faded paint exposes
def _back_scene():
    """A strip of her upper back seen from behind (Unity +z = her back): paint at z 0.08, the top's layer at z 0.074."""
    J = np.zeros((55, 3))
    J[3, 1], J[6, 1], J[9], J[12, 1] = 0.0, 0.1, [0.0, 0.2, 0.0], 0.4
    rows = []
    for x in np.linspace(-0.12, 0.12, 25):
        for y in np.linspace(0.12, 0.34, 23):
            rows.append([x, y, 0.074])
    mean = np.array(rows)
    n = len(mean)
    lab = np.tile([24.0, 31.0, 17.0], (n, 1))                  # the top's red
    grey = (np.abs(mean[:, 0]) < 0.03) & (np.abs(mean[:, 1] - 0.24) < 0.03)
    lab[grey] = [23.0, 12.0, 4.0]                               # LHM++'s grey guess under the hair
    dark = (np.abs(mean[:, 0] - 0.07) < 0.012) & (np.abs(mean[:, 1] - 0.2) < 0.012)
    lab[dark] = [9.0, 14.0, 6.0]                                # a brown-black smudge
    paint_xyz = mean[grey | dark] + [0.0, 0.0, 0.006]
    mean = np.concatenate([mean, paint_xyz])
    lab = np.concatenate([lab, np.tile([7.0, 2.0, 1.0], (len(paint_xyz), 1))])
    paint = np.zeros(len(mean), bool)
    paint[n:] = True
    grey = np.concatenate([grey, np.zeros(len(paint_xyz), bool)])
    dark = np.concatenate([dark, np.zeros(len(paint_xyz), bool)])
    return J, mean, lab, paint, grey, dark


def test_under_mask_finds_the_grey_and_dark_layer_under_the_paint_only():
    J, mean, lab, paint, grey, dark = _back_scene()
    n = len(mean)
    nz = np.full(n, -0.9)
    unbound, rigid, hair = np.ones(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    hair[np.nonzero(dark)[0][:3]] = True                        # dark smudges count whether hair-flagged or not
    g, d = RH.under_mask(mean, J, nz, lab, unbound, rigid, hair, paint, 0.03, RH.HAIR_LOOK_BLACK["under_tint"])
    assert np.array_equal(g, grey) and np.array_equal(d, dark)
    far = paint.copy()
    far[paint] = False                                          # no paint: nothing exposed, nothing to tint
    g2, d2 = RH.under_mask(mean, J, nz, lab, unbound, rigid, hair, far, 0.03, RH.HAIR_LOOK_BLACK["under_tint"])
    assert not g2.any() and not d2.any()
    nz2 = nz.copy()
    nz2[grey] = 0.9                                             # facing front: not the back
    g3, _ = RH.under_mask(mean, J, nz2, lab, unbound, rigid, hair, paint, 0.03, RH.HAIR_LOOK_BLACK["under_tint"])
    assert not g3.any()
    hair2 = hair.copy()
    hair2[grey] = True                                          # grey but hair-flagged: hair, not the top
    g4, _ = RH.under_mask(mean, J, nz, lab, unbound, rigid, hair2, paint, 0.03, RH.HAIR_LOOK_BLACK["under_tint"])
    assert not g4.any()


def test_under_tint_gives_the_targets_the_local_red_and_keeps_shading():
    J, mean, lab, paint, grey, dark = _back_scene()
    lab[np.nonzero(grey)[0][0]] = [27.0, 12.0, 4.0]             # one lighter grey: stays lighter (shading kept)
    op = np.full(len(mean), 0.5)
    cfg = RH.HAIR_LOOK_BLACK["under_tint"]
    C = np.hypot(lab[:, 1], lab[:, 2])
    red = (C >= cfg["red_chroma"]) & ~paint
    out, info = RH.under_tint(mean, lab, op, grey, red, cfg)
    assert info["tinted"] == int(grey.sum()) and info["kept_no_red"] == 0
    t = out[grey]
    assert np.all(np.hypot(t[:, 1], t[:, 2]) > 25)               # red now
    hue = np.degrees(np.arctan2(t[:, 2], t[:, 1]))
    assert np.allclose(hue, np.degrees(np.arctan2(17.0, 31.0)), atol=1e-6)    # the top's hue
    i0 = np.nonzero(grey)[0][0]
    assert out[i0, 0] > np.median(t[:, 0])                        # the lighter one stays lighter
    assert np.array_equal(out[~grey], lab[~grey])                 # nothing else changes

    out2, info2 = RH.under_tint(mean, lab, op, grey, np.zeros(len(mean), bool), cfg)
    assert info2["kept_no_red"] == int(grey.sum()) and np.array_equal(out2, lab)   # no red reference: unchanged



def test_apply_look_tints_the_exposed_layer_and_keeps_opacity():
    J, mean, lab, paint, grey, dark = _back_scene()
    n = len(mean)
    out = np.zeros(n, RH.RECORD)
    out["mean"] = mean
    out["rotation"] = [0, 0, 0, 1]
    out["log_scale"] = np.log(0.006)
    out["opacity"] = 0.0
    out["sh0"] = RH.lab_to_sh0(lab)
    out["strand"], out["segment"] = -1, -1
    out["hair"] = 0
    out["hair"][paint] = 1
    before = out.copy()
    G, P = 2, 4
    rest = np.zeros((G, P, 3))
    rest[:, :, 1] = -0.05 * np.arange(P)
    rest[1, :, 0] = 0.05
    f = {"normal": np.tile([0.0, 0.0, -1.0], (n, 1)), "sd": np.zeros(n), "dom": np.full(n, 9), "el": np.full(n, -60.0),
         "az": np.full(n, 180.0), "dh": np.full(n, 0.3)}
    sel = {"features": f, "hair_colour": paint.copy()}
    idx = np.zeros(0, int)
    b = {"strand": np.zeros(0, int), "segment": np.zeros(0, int)}
    q2, info = RH.apply_look(out, idx, b, rest, J, lab, np.zeros(n, bool), sel, RH.HAIR_LOOK_BLACK)
    ut = info["under_tint"]
    assert ut["grey"]["tinted"] == int(grey.sum()) and ut["dark"]["tinted"] == int(dark.sum())
    assert ut["paint"]["tinted"] == int(paint.sum()) and info["paint_fade"]["n"] == int(paint.sum())
    newlab = SH._lab(np.clip(out["sh0"].astype(float) * SH.SH_C0 + 0.5, 0, 1))
    assert np.all(np.hypot(newlab[grey, 1], newlab[grey, 2]) > 20)
    assert np.array_equal(out["opacity"][~paint], before["opacity"][~paint])      # tint only: opacity of the layer kept
    assert np.array_equal(out["sh0"][~(grey | dark | paint)], before["sh0"][~(grey | dark | paint)])
