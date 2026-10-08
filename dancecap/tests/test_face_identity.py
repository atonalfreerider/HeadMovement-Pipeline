"""Face identity transplant (dancecap.face_identity + the `identity*` keys of dancecap.shape_edits' face block) on plain
SMPL-X bodies with a synthetic FLAME identity (needs the git-ignored FLAME files; skipped without them).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_face_identity.py -q
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import face_identity as fi  # noqa: E402
from dancecap import shape_edits as se  # noqa: E402
from dancecap.export_unity import DEFAULT_MODEL, shaped_skin, textured_skin  # noqa: E402

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists() or not fi.flame_available(),
                                reason="SMPL-X / FLAME model files not installed")


@pytest.fixture(scope="module")
def body_a():
    return shaped_skin(DEFAULT_MODEL, np.zeros(10))


@pytest.fixture(scope="module")
def body_b():
    return shaped_skin(DEFAULT_MODEL, np.random.default_rng(3).standard_normal(10) * 1.3)     # another head


def _ratios(v):
    m = se.face_measures(v)
    return {"lower": m["lower_face_sn_me_mm"] / m["outer_canthal_mm"], "jaw": m["jaw_angle_c4_12_mm"] / m["cheek_c1_15_mm"],
            "chin": m["chin_c6_10_mm"] / m["cheek_c1_15_mm"]}


@pytest.fixture(scope="module")
def ident(tmp_path_factory, body_a, body_b):
    """A synthetic identity (FLAME shape code) whose lower face / jaw / chin proportions differ from BOTH test heads."""
    F = fi.flame()
    ids = F["sx_ids"]
    heads = [_ratios(b.vertices) for b in (body_a, body_b)]
    for seed in range(60):
        beta = np.random.default_rng(100 + seed).standard_normal(300) * 0.9
        V = np.zeros((fi.N_SX, 3))
        V[ids] = fi.flame_shape(beta, F)
        r = _ratios(V)
        if all(abs(r["lower"] - h["lower"]) / h["lower"] > 0.07 and abs(r["jaw"] - h["jaw"]) / h["jaw"] > 0.02 for h in heads):
            p = tmp_path_factory.mktemp("ident") / "identity.npz"
            np.savez(p, beta=beta)
            return {"path": p, "beta": beta, "target": r}
    pytest.skip("no synthetic identity with distinct proportions found")


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return tmp_path_factory.mktemp("work")


def _edit(body, work, ident, face=None, role="follow"):
    cfg = {"face": {"identity": str(ident["path"]), **(face or {})}}
    return se.apply_vertices(cfg, body.vertices, body.joints, body.weights, body.faces, role, work)


@pytest.fixture(scope="module")
def edit_a(body_a, work, ident):
    return _edit(body_a, work, ident)


def test_identity_moves_the_lower_face_and_keeps_the_protected_regions(body_a, edit_a):
    out, rep = edit_a
    F = fi.flame()
    mv = np.linalg.norm(out - body_a.vertices, axis=1)
    assert 0.001 < mv.max() < 0.03                                                 # a real, but not wild, change
    assert rep["face"]["protected_flame_max_move_mm"] == {k: 0.0 for k in fi.PROTECTED_REGIONS}   # eyes, eyeballs, brows,
    ids = F["sx_ids"]                                                                # forehead, scalp, ears, neck seam ring
    assert mv[ids[F["masks"]["nose"]]].max() == 0.0                                # the nose
    assert mv[ids[F["masks"]["eye_region"]]].max() == 0.0
    fr = se.face_frame(body_a.vertices)
    assert mv[body_a.vertices[:, 1] < fr["y_me"] - 0.075].max() == 0.0             # the neck-shoulder seam and below
    assert mv[se.arm_weight(body_a.weights) > 0.999].max() == 0.0
    assert rep["face"]["identity"]["max_move_mm"] == pytest.approx(1000 * mv.max(), rel=0.02)
    assert rep["face"]["protected_max_move_mm"]["nose"] == 0.0
    lower = se.face_measures(out)
    assert lower["lower_face_sn_me_mm"] != se.face_measures(body_a.vertices)["lower_face_sn_me_mm"]


def test_identity_keeps_uvs_weights_joints_and_topology(body_a, work, ident):
    edited, rep = se.apply(body_a, {"face": {"identity": str(ident["path"])}}, "follow", work)
    assert edited.joints is body_a.joints and edited.weights is body_a.weights and edited.faces is body_a.faces
    a, b = textured_skin(body_a, DEFAULT_MODEL), textured_skin(replace(body_a, vertices=edited.vertices), DEFAULT_MODEL)
    if a.uv is None:
        pytest.skip("model without UVs")
    assert np.array_equal(a.uv, b.uv) and np.array_equal(a.faces, b.faces) and np.array_equal(a.vertex_ids, b.vertex_ids)
    assert np.array_equal(a.weights, b.weights)
    s = se.summary(rep)["face"]
    assert s["identity"]["file"] == str(ident["path"]) and s["identity"]["weight"] == 1.0


def test_identity_absent_or_zero_weight_is_a_no_op(body_a, work, ident):
    out, rep = se.apply_vertices({"face": {"identity_weight": 1.0, "identity_mouth": 0.3}}, body_a.vertices, body_a.joints,
                                 body_a.weights, body_a.faces, "follow", work)
    assert np.abs(out - body_a.vertices).max() == 0.0 and "identity" not in rep["face"]
    out0, rep0 = _edit(body_a, work, ident, {"identity_weight": 0.0})
    assert np.abs(out0 - body_a.vertices).max() == 0.0 and rep0["face"]["moved_vertices"] == 0


def test_identity_weight_is_linear_and_extrapolates(body_a, work, ident, edit_a):
    d1 = edit_a[0] - body_a.vertices
    d5 = _edit(body_a, work, ident, {"identity_weight": 0.5})[0] - body_a.vertices
    d13 = _edit(body_a, work, ident, {"identity_weight": 1.3})[0] - body_a.vertices
    assert np.abs(d5 - 0.5 * d1).max() < 1e-6 and np.abs(d13 - 1.3 * d1).max() < 1e-6


def test_identity_mouth_share(body_a, work, ident, edit_a):
    F = fi.flame()
    lips = F["sx_ids"][F["masks"]["lips"]]
    d1 = (edit_a[0] - body_a.vertices)[lips]
    d0 = (_edit(body_a, work, ident, {"identity_mouth": 0.0})[0] - body_a.vertices)[lips]
    d5 = (_edit(body_a, work, ident, {"identity_mouth": 0.5})[0] - body_a.vertices)[lips]
    assert np.abs(d0).max() == 0.0 and np.linalg.norm(d1, axis=1).max() > 0.5e-3   # kept in place / moves with the identity
    assert np.abs(d5 - 0.5 * d1).max() < 1e-9
    fr = se.face_frame(body_a.vertices)
    far = (np.abs(fr["y_go"] - body_a.vertices[:, 1]) < 0.004) & (np.abs(body_a.vertices[:, 0] - fr["x_mid"]) > 0.04)   # jaw angles
    full = edit_a[0] - body_a.vertices
    part = _edit(body_a, work, ident, {"identity_mouth": 0.0})[0] - body_a.vertices
    assert far.sum() > 4 and np.abs(part[far] - full[far]).max() < 1e-6            # (the blend is 14 mm: the jaw angles are far)


def test_identity_neck_falls_off_smoothly_to_the_seam(body_a, edit_a):
    out, _ = edit_a
    v0 = body_a.vertices
    fr = se.face_frame(v0)
    d = out - v0
    sel = np.nonzero((np.abs(v0[:, 0] - fr["x_mid"]) < 0.006) & (v0[:, 2] > fr["z_ear"] + 0.02) & (v0[:, 1] < fr["y_me"] - 0.012)
                     & (v0[:, 1] > fr["y_me"] - 0.12))[0]
    sel = sel[np.argsort(v0[sel, 1])]
    mag = np.linalg.norm(d[sel], axis=1)
    y = v0[sel, 1]
    assert mag[y < fr["y_me"] - 0.075].max() == 0.0                                # exactly nothing at the seam
    slope = np.abs(np.diff(mag) / np.maximum(np.diff(y), 1e-4))[np.diff(y) > 0.002]
    assert slope.max() < 0.5                                                       # no step, no ledge: bounded strain
    # biharmonic extension: the profile has no interior kink: the slope changes smoothly between neighbouring samples
    up = y > fr["y_me"] - 0.07
    ds = np.diff(np.diff(mag[up]))
    assert np.abs(ds).max() < 0.0035


def test_identity_smooth_no_folds(body_a, work, ident):
    out, _ = _edit(body_a, work, ident, {"identity_mouth": 0.5})                   # the share the candidates use
    v0, F = body_a.vertices, body_a.faces

    def nrm(v):
        n = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]])
        return n, np.linalg.norm(n, axis=1)
    moved = (np.linalg.norm(out - v0, axis=1) > 1e-9)[F].any(1)
    n0, a0 = nrm(v0)
    n1, a1 = nrm(out)
    dots = ((n0 * n1).sum(1) / np.maximum(a0 * a1, 1e-18))[moved]
    ratio = (a1 / np.maximum(a0, 1e-18))[moved]
    assert dots.min() > 0.5                                                        # no flipped / folded triangle
    assert 0.25 < ratio.min() and ratio.max() < 4.0
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    d = out - v0
    strain = np.linalg.norm(d[e[:, 0]] - d[e[:, 1]], axis=1) / np.maximum(np.linalg.norm(v0[e[:, 0]] - v0[e[:, 1]], axis=1), 1e-9)
    assert strain.max() < 1.0


def test_identity_retargets_to_a_second_head(body_a, body_b, work, ident, edit_a):
    """The displacement is computed per head: other betas -> another displacement, and each head's lower-face proportions
    end up closer to the identity's own."""
    out_a, _ = edit_a
    out_b, rep_b = _edit(body_b, work, ident)
    d_a, d_b = out_a - body_a.vertices, out_b - body_b.vertices
    assert np.abs(d_a - d_b).max() > 3e-4                                          # not the same displacement
    assert rep_b["face"]["protected_flame_max_move_mm"] == {k: 0.0 for k in fi.PROTECTED_REGIONS}
    tgt = ident["target"]
    for body, out in ((body_a, out_a), (body_b, out_b)):
        before, after = _ratios(body.vertices), _ratios(out)
        closer = [abs(after[k] - tgt[k]) < abs(before[k] - tgt[k]) for k in ("lower", "jaw", "chin")]
        assert closer[0] and sum(closer) >= 2, (before, after, tgt)
    # both heads keep their own identity-independent parts: the eyes' distance is untouched
    assert se.face_measures(out_b)["outer_canthal_mm"] == pytest.approx(se.face_measures(body_b.vertices)["outer_canthal_mm"],
                                                                        abs=1e-6)


def test_identity_cache_reuse_and_invalidation(body_a, body_b, tmp_path, ident, monkeypatch):
    cache = tmp_path / "face_shape" / "follow_identity_disp.npz"
    out1, rep1 = _edit(body_a, tmp_path, ident)
    assert rep1["face"]["identity"]["cache_hit"] is False and cache.exists()
    out2, rep2 = _edit(body_a, tmp_path, ident)
    assert rep2["face"]["identity"]["cache_hit"] is True and np.abs(out2 - out1).max() < 1e-6
    # FLAME is not needed on a hit: make any recomputation fail
    monkeypatch.setattr(fi, "displacement", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("recomputed")))
    out3, rep3 = _edit(body_a, tmp_path, ident)
    assert rep3["face"]["identity"]["cache_hit"] is True and np.abs(out3 - out1).max() < 1e-6
    shift = np.array([0.2, 1.1, -0.4])                                             # the same head elsewhere (hair_groom.RestModel
    out4, rep4 = _edit(replace(body_a, vertices=body_a.vertices + shift), tmp_path, ident)   # puts the sole at y = 0) is a hit too
    assert rep4["face"]["identity"]["cache_hit"] is True and np.abs((out4 - shift) - out1).max() < 1e-6
    with pytest.raises(RuntimeError, match="recomputed"):                          # another head -> a miss
        _edit(body_b, tmp_path, ident)
    other = tmp_path / "other_identity.npz"
    np.savez(other, beta=ident["beta"] * 0.5)
    with pytest.raises(RuntimeError, match="recomputed"):                          # another identity file -> a miss
        se.apply_vertices({"face": {"identity": str(other)}}, body_a.vertices, body_a.joints, body_a.weights,
                          body_a.faces, "follow", tmp_path)
    monkeypatch.setitem(fi.MASK_DEFAULT, "clear_m", 0.05)
    with pytest.raises(RuntimeError, match="recomputed"):                          # other mask parameters -> a miss
        _edit(body_a, tmp_path, ident)
    monkeypatch.undo()
    monkeypatch.setattr(fi, "flame_available", lambda: False)
    (tmp_path / "face_shape" / "follow_identity_disp.npz").unlink()
    with pytest.raises(FileNotFoundError, match="no valid cache"):                 # no cache and no FLAME
        _edit(body_a, tmp_path, ident)
    with pytest.raises(FileNotFoundError, match="missing"):
        se.apply_vertices({"face": {"identity": str(tmp_path / "nope.npz")}}, body_a.vertices, body_a.joints, body_a.weights,
                          body_a.faces, "follow", tmp_path)


def test_identity_cache_is_per_role(body_a, tmp_path, ident):
    _edit(body_a, tmp_path, ident, role="lead")
    assert (tmp_path / "face_shape" / "lead_identity_disp.npz").exists() and not (tmp_path / "face_shape" / "follow_identity_disp.npz").exists()


def test_identity_combines_with_the_geometric_factors(body_a, work, ident, edit_a):
    """The factors act on the head the identity produced (identity first), and C1 / protection still hold."""
    face = {"jaw_taper": 0.9, "chin_point": 0.4, "neck_under_jaw": 0.96}
    out, rep = _edit(body_a, work, ident, face)
    only_id = edit_a[0]
    assert se.face_measures(out)["jaw_angle_c4_12_mm"] < se.face_measures(only_id)["jaw_angle_c4_12_mm"]
    assert rep["face"]["protected_flame_max_move_mm"]["eye_region"] == 0.0 and rep["face"]["protected_flame_max_move_mm"]["forehead"] == 0.0
    assert rep["face"]["protected_flame_max_move_mm"]["left_ear"] < 0.1 and rep["face"]["protected_flame_max_move_mm"]["boundary"] == 0.0
    mv = np.linalg.norm(out - body_a.vertices, axis=1)
    assert mv[body_a.vertices[:, 1] < se.face_frame(body_a.vertices)["y_me"] - 0.075].max() == 0.0


# ------------------------------------------------------------------------------------------------ the FLAME-side pieces

def test_transplant_mask_is_zero_where_protected_and_smooth_elsewhere():
    F = fi.flame()
    m = fi.transplant_mask()
    assert m.min() >= 0.0 and m.max() <= 1.0 and m.max() > 0.999
    for k in ("eye_region", "forehead", "scalp", "left_eyeball", "right_eyeball", "left_ear", "right_ear", "boundary", "nose"):
        assert m[F["masks"][k]].max() == 0.0, k
    # the transition from the protected set to the full set is at least `clear_m` of surface long: bounded slope
    f, vt = F["f"], F["v_template"]
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    slope = np.abs(m[e[:, 0]] - m[e[:, 1]]) / np.linalg.norm(vt[e[:, 0]] - vt[e[:, 1]], axis=1)
    assert slope.max() < 3.0 * 1.5 / fi.MASK_DEFAULT["nose_clear_m"] / 3                  # < 125 / m (the nose gap is the shortest)
    near = fi._geodesic(vt, f, np.isin(np.arange(len(vt)), np.concatenate([F["masks"]["left_ear"], F["masks"]["right_ear"]])))
    assert m[near < 0.01].max() < 0.2                                              # skin next to the ear barely moves


def test_neck_extension_keeps_the_rest_and_is_bounded():
    F = fi.flame()
    rng = np.random.default_rng(5)
    D = rng.standard_normal((len(F["v_template"]), 3)) * 1e-3
    m = fi.transplant_mask()
    D = D * m[:, None]
    out = fi.neck_extension(D)
    st = fi._sets(fi.MASK_DEFAULT["throat_drop_m"], fi.MASK_DEFAULT["nape_back_m"])
    free = st["neck"] & ~st["zero"]
    assert free.any() and np.abs(out[~free] - D[~free]).max() == 0.0               # only the free neck vertices change
    assert np.abs(out[st["zero"]]).max() == 0.0 if np.abs(D[st["zero"]]).max() == 0.0 else True
    assert np.isfinite(out).all() and np.abs(out[free]).max() < 3 * np.abs(D).max()
    const = np.zeros_like(D)
    const[~st["zero"] & ~free] = np.array([0.0, 0.004, -0.004])                      # the face moves, the ring does not
    ext = fi.neck_extension(const)
    assert np.abs(ext[free]).max() <= 0.0057 * 1.25                               # thin-plate: a gentle overshoot at most


def test_mouth_weight():
    F = fi.flame()
    w = fi.mouth_weight()
    assert w[F["masks"]["lips"]].min() == 1.0 and w.min() == 0.0 and w.max() == 1.0
    assert w[F["masks"]["eye_region"]].max() == 0.0 and w[F["masks"]["forehead"]].max() == 0.0
    d = fi._geodesic(F["v_template"], F["f"], np.isin(np.arange(len(w)), F["masks"]["lips"]))
    o = np.argsort(d)
    assert np.all(np.diff(w[o]) <= 1e-12)                                          # fades monotonically with the distance
    assert w[d >= fi.MASK_DEFAULT["mouth_blend_m"]].max() == 0.0


def test_fit_equivalent_reproduces_a_flame_head(body_a):
    """The FLAME equivalent of an SMPL-X head is close to it (the transplant's basis): rms below 2.5 mm."""
    F = fi.flame()
    eq = fi.fit_equivalent(body_a.vertices[F["sx_ids"]], F=F)
    assert eq["rms_mm"] < 2.5 and eq["max_mm"] < 10.0
    # an exactly-FLAME head (a known similarity of a known shape) is recovered
    beta = np.random.default_rng(1).standard_normal(300) * 0.7
    V = 0.9 * fi.flame_shape(beta, F) @ np.array([[0.995, 0, 0.1], [0, 1, 0], [-0.1, 0, 0.995]]).T + np.array([0.01, 0.2, -0.03])
    eq2 = fi.fit_equivalent(V, F=F)
    assert eq2["rms_mm"] < 0.5 and eq2["s"] == pytest.approx(0.9, rel=0.01)
