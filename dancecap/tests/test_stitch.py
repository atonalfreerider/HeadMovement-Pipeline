"""dancecap.stitch: stitching the segment fits of a long take (synthetic segments built from larissa_kadu_38's REAL arrays).

The 38 s take (smplx/ raw, smplx_rf2_noslip/ refined) is the reused middle chunk. The segments are its motion at
reflected frame indices, moved by smooth per-segment offsets (translation 2 cm, arm rotations 0.04 rad) with joints from
the real SMPL-X FK, on the same lattice offsets as the full dance (a ends 120 frames into 38, b starts 120 frames before
its end, c / d overlap their neighbour). They need local data (the 38 s take + the SMPL-X model): skipped without.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from dancecap import stitch as st
from dancecap.take import PACKAGE_ROOT

REF = PACKAGE_ROOT / "work" / "larissa_kadu_38"
MODEL = PACKAGE_ROOT.parent / "mamma" / "data" / "body_models" / "smplx_locked_head"
needs_data = pytest.mark.skipif(not ((REF / "smplx" / "lead.npz").exists() and (REF / "smplx_rf2_noslip" / "lead.npz").exists()
                                     and MODEL.exists()), reason="needs the larissa_kadu_38 fit + SMPL-X model (local data)")
A = st.ANCHOR
# compact layout on the real lattice: 38 = k 0..1147; a ends 120 frames into it, b starts 120 frames before its end
LAYOUT = {"a": (-200, 120), "b": (1027, 1300), "c": (1200, 1500), "d": (1400, 1700)}
N38 = 1148


def _fold(k):
    m = np.asarray(k) % (2 * (N38 - 1))
    return np.where(m <= N38 - 1, m, 2 * (N38 - 1) - m)


def _smooth(ks, seed, periods=(11.0, 17.0, 23.0)):
    rng = np.random.default_rng(seed)
    t = np.asarray(ks) / 30.0
    out = np.zeros((len(ks), 3))
    for p in periods:
        out += np.sin(2 * np.pi * t[:, None] / p + rng.uniform(0, 2 * np.pi, 3))
    return out / len(periods)


def _make(src: dict, k0: int, k1: int, seed: int, role_seed: int) -> dict:
    """A segment on k0..k1: the source arrays at reflected frames, perturbed, with FK joints."""
    ks = np.arange(k0, k1 + 1)
    idx = _fold(ks)
    z = {}
    n = len(ks)
    for key, v in src.items():
        if hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == N38 and key not in st.STATIC:
            z[key] = np.array(v[idx])
        else:
            z[key] = np.array(v)
    z["t"] = A + ks / 30.0
    z["transl"] = z["transl"] + 0.02 * _smooth(ks, seed * 10 + role_seed)
    bp = z["body_pose"].copy()
    for j in (16, 17, 18, 19, 20, 21):
        bp[:, (j - 1) * 3:(j - 1) * 3 + 3] += 0.04 * _smooth(ks, seed * 100 + role_seed * 7 + j)
    z["body_pose"] = bp
    z["global_orient"] = z["global_orient"] + 0.01 * _smooth(ks, seed * 1000 + role_seed)
    z["joints"] = st.fk_joints(st.pose165(z), z["transl"], z["betas"])
    return z


def _save(d: Path, roles: dict):
    d.mkdir(parents=True, exist_ok=True)
    for r, z in roles.items():
        np.savez(d / f"{r}.npz", **z)


def _prep(d: Path, spans: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / "prepare.json").write_text(json.dumps({"grid": {"trim": {"cams_per_frame": spans}}}))
    return d / "prepare.json"


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    """Segments a, b, c, d (raw + refined) + the real 38 as the reference chunk; returns the helper dict."""
    root = tmp_path_factory.mktemp("stitch_layout")
    raw38 = st.read_dir(REF / "smplx")
    ref38 = st.read_dir(REF / "smplx_rf2_noslip")
    dirs = {}
    for seed, (name, (k0, k1)) in enumerate(LAYOUT.items(), 1):
        _save(root / name / "raw", {r: _make(raw38[r], k0, k1, seed, i) for i, r in enumerate(st.ROLES)})
        _save(root / name / "ref", {r: _make(ref38[r], k0, k1, seed, i) for i, r in enumerate(st.ROLES)})
        t0, t1 = A + k0 / 30.0, A + k1 / 30.0
        # camera spans (reference s): a has cam 09 from its start, 06 from +3 s, 01 from +6 s, the rest from +9 s
        spans = {"09": [t0, t1], "06": [t0 + 3, t1], "01": [t0 + 6, t1], "02": [t0 + 9, t1], "03": [t0 + 9, t1]} \
            if name == "a" else {c: [t0, t1] for c in ("01", "02", "03", "05", "06")}
        dirs[name] = {"raw": root / name / "raw", "ref": root / name / "ref", "prepare": _prep(root / name / "p", spans)}
    return {"root": root, "dirs": dirs, "raw38": raw38, "ref38": ref38}


def make_spec(layout, tmp, **over) -> st.Spec:
    chunks = []
    for name, (k0, k1) in LAYOUT.items():
        d = layout["dirs"][name]
        chunks.append(st.Chunk(name, d["raw"], d["ref"], k0, k1, 0.2, False, "segment", d["prepare"]))
    chunks.append(st.Chunk("38", REF / "smplx", REF / "smplx_rf2_noslip", 0, N38 - 1, 0.5, True, "reuse"))
    chunks.sort(key=lambda c: (c.k0, c.k1))
    kw = dict(take="syn", chunks=chunks, work=Path(tmp) / "work_syn", beats=REF / "beats.json")
    kw.update(over)
    return st.Spec(**kw)


@pytest.fixture(scope="module")
def good(layout, tmp_path_factory):
    """One full stitch of the good layout, written to disk (shared by the tests that only read it)."""
    tmp = tmp_path_factory.mktemp("stitch_good")
    spec = make_spec(layout, tmp)
    res = st.stitch(spec, sheets=False, verbose=False)
    return {"spec": spec, "res": res, "work": spec.work}


# ------------------------------------------------------------------------------------------------ pure functions


def test_constants_match_mamma_chunks():
    from dancecap import mamma_chunks as mc
    assert st.JOINT_NAMES == tuple(mc.BODY_JOINTS)
    assert st.ARM == tuple(mc.ARM_JOINTS)
    assert tuple((w, tuple(f)) for w, f in st.FINGERS) == tuple((w, tuple(f)) for w, f in mc.FINGERS)
    for mine, theirs in (("body_p95", "seam_p95_mm"), ("elbow_wrist_p95", "seam_arm_p95_mm"),
                         ("fingers_rel_wrist_p95", "seam_fingers_rel_p95_mm"), ("pelvis_median", "seam_pelvis_median_mm"),
                         ("betas_swap_p95", "betas_swap_p95_mm")):
        assert st.GATE_DEFAULT[mine] == mc.GATE[theirs]


def test_window_stats_equal_mamma_chunks_seam_stats():
    from dancecap import mamma_chunks as mc
    rng = np.random.default_rng(3)
    n = 90
    jp = rng.normal(size=(n, 55, 3)) * 0.2
    jn = jp + rng.normal(size=(n, 55, 3)) * 0.02
    w = np.zeros(n)
    w[20:80] = st.cosine_weights(60)
    w[80:] = 1.0
    ss = mc.seam_stats(jp, jn, w)["blend"]
    d, rel = st.disagreement(jp, jn)
    mine = st.window_stats(d, rel, slice(20, 80))
    for k in ("joint_p95_max", "arm_p95_max", "fingers_rel_wrist_p95_max"):
        assert abs(mine[k] - ss[k]) < 0.01, k
    assert mine["joint"] == ss["joint"] and mine["arm_joint"] == ss["arm_joint"]


def test_cosine_weights_strictly_inside_and_symmetric():
    w = st.cosine_weights(60)
    assert 0 < w[0] < 0.001 and 0.999 < w[-1] < 1 and np.all(np.diff(w) > 0)
    assert np.allclose(w + w[::-1], 1.0)


def test_lattice_range_checks():
    t = A + np.arange(-5, 40) / 30.0
    assert st.lattice_range(t, A, 30.0, "x") == (-5, 39)
    with pytest.raises(ValueError, match="off the fused lattice"):
        st.lattice_range(t + 0.01, A, 30.0, "x")
    with pytest.raises(ValueError, match="contiguous"):
        st.lattice_range(np.r_[t[:5], t[7:]], A, 30.0, "x")


def test_blend_rotvec_shortest_arc_and_branches():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(40, 4, 3))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    ang = rng.uniform(0.1, 3.0, size=(40, 4, 1))
    a = u * ang
    b = Rot.from_rotvec((a + 0.05 * rng.normal(size=a.shape)).reshape(-1, 3)).as_rotvec().reshape(a.shape)
    # the same rotations on the other axis-angle branch: (theta - 2 pi) about the same axis
    th = np.linalg.norm(b, axis=-1, keepdims=True)
    b_flip = b / th * (th - 2 * np.pi)
    w = np.linspace(0.0, 1.0, 40)
    r1 = st.blend_rotvec(a, b, w)
    r2 = st.blend_rotvec(a, b_flip, w)
    R1 = Rot.from_rotvec(r1.reshape(-1, 3))
    R2 = Rot.from_rotvec(r2.reshape(-1, 3))
    assert (R1 * R2.inv()).magnitude().max() < 1e-6              # same rotation whatever the input branch
    Ra, Rb = Rot.from_rotvec(a.reshape(-1, 3)), Rot.from_rotvec(b.reshape(-1, 3))
    for i, wi in ((0, 0.0), (39, 1.0)):                          # ends reproduce the inputs
        sl = slice(i * 4, i * 4 + 4)
        assert (R1[sl] * (Ra if i == 0 else Rb)[sl].inv()).magnitude().max() < 1e-6
    # the blend lies on the shortest arc: half way is half the angle between them
    mid = st.blend_rotvec(a[:1], b[:1], np.array([0.5]))
    full = (Ra[:4].inv() * Rb[:4]).magnitude()
    half = (Ra[:4].inv() * Rot.from_rotvec(mid.reshape(-1, 3))).magnitude()
    assert np.allclose(half, full / 2, atol=1e-6)


def test_blend_rotvec_keeps_the_branch_nearest_the_inputs():
    """axis-angle near pi with flipped axis: a = +3.1 u, b = -3.1 u are 0.08 rad apart; the blend must stay near both."""
    u = np.array([0.0, 1.0, 0.0])
    a = (3.1 * u)[None, None]
    b = (-3.1 * u)[None, None]
    for w in (0.1, 0.5, 0.9):
        v = st.blend_rotvec(a, b, np.array([w]))
        ang = (Rot.from_rotvec(v[0]) * Rot.from_rotvec(a[0]).inv()).magnitude()
        assert ang < 0.1
        assert np.abs(v).max() < 3.3                              # no jump to the 2 pi - theta vector of the other branch


def test_deterministic_npz(tmp_path):
    arrays = {"x": np.arange(12.0).reshape(3, 4), "s": np.array("hello"), "b": np.array([True, False])}
    st.save_npz(tmp_path / "a.npz", arrays)
    st.save_npz(tmp_path / "b.npz", arrays)
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    with np.load(tmp_path / "a.npz") as z:
        assert np.array_equal(z["x"], arrays["x"]) and str(z["s"]) == "hello" and z["b"].tolist() == [True, False]


def test_quality_flags_rules():
    spec = st.Spec(take="x", chunks=[], mono=[[0.0, 1.0]], low_view=[[1.0, 2.0]], absent={"lead": [[0.0, 0.5]]})
    t = np.arange(0, 120) / 30.0
    n = np.full(120, 5, np.int16)
    n[40:45] = 0                      # 5-frame dropout: weak, not absent
    n[60:100] = 0                     # 1.3 s with no camera: absent
    n[100:110] = 2
    cov = np.ones(120, bool)
    fl, info = st.quality_flags(spec, "lead", n, t, cov)
    assert (fl[:16] == 0).all()                        # lead declared absent until 0.5 s
    assert (fl[16:31] == 1).all()                      # mono span (<= 1.0 s, inclusive)
    assert (fl[31:40] == 2).all()                      # low_view span (<= 2 s)
    assert (fl[40:45] == 1).all()                      # short dropout -> mono, not absent
    assert (fl[60:100] == 0).all() and (fl[100:110] == 2).all() and (fl[110:] == 3).all()
    fl2, _ = st.quality_flags(spec, "follow", n, t, cov)
    assert (fl2[:30] == 1).all() and fl2[5] == 1 and (fl2[60:100] == 0).all()   # follow is not declared absent
    cov[110:] = False
    assert (st.quality_flags(spec, "follow", n, t, cov)[0][110:] == 0).all()    # uncovered = absent


def _fake_chain(spans, margins=None, quiet=None, seed=0):
    """Chunks on spans [(name, k0, k1)], joints = one smooth trajectory + a disagreement of 10 mm inside `quiet` (a k range)
    and 120 mm elsewhere between a chunk and its predecessor (no FK, no files)."""
    rng = np.random.default_rng(seed)
    kmin = min(k0 for _, k0, _ in spans)
    kmax = max(k1 for _, _, k1 in spans)
    t = np.arange(kmin, kmax + 1) / 30.0
    base = np.zeros((len(t), 55, 3))
    base[:, :, 0] = 0.3 * np.sin(2 * np.pi * 0.4 * t)[:, None] + 0.02 * np.arange(55)[None]
    base[:, :, 1] = 1.0 + 0.1 * np.sin(2 * np.pi * 0.9 * t)[:, None]
    L, J = [], {r: [] for r in st.ROLES}
    for i, (name, k0, k1) in enumerate(spans):
        m = (margins or {}).get(name, 0)
        L.append(st.Loaded(st.Chunk(name, Path("x"), None, k0, k1, 0.0, False, "segment"), k0, k1, m, m, {}))
        ks = np.arange(k0, k1 + 1)
        for r in st.ROLES:
            j = base[ks - kmin].copy()
            if i > 0:                                  # later chunks are displaced from the earlier ones
                inq = np.ones(len(ks), bool) if quiet is None else (ks >= quiet[0]) & (ks < quiet[1])
                j[:, :, 2] += np.where(inq, 0.010, 0.120)[:, None] * (1 if i % 2 else -1)
            j += rng.normal(scale=0.0005, size=j.shape)
            J[r].append(j)
    return L, J


def test_window_lands_in_the_quiet_part_of_the_overlap():
    L, J = _fake_chain([("p", 0, 300), ("n", 150, 450)], quiet=(205, 290))
    spec = st.Spec(take="x", chunks=[])
    seams = st.plan_seams(spec, L)
    st._choose_windows(spec, L, seams, J, spec.thresholds(), None, None)
    s = seams[0]
    assert s["status"] == "ok" and s["hi"] - s["lo"] == 60 and s["tie_break"]["method"] == "motion"
    assert 205 <= s["lo"] and s["hi"] <= 290                       # only windows entirely inside the 10 mm zone pass
    assert s["utilisation"] < 0.3 and s["best_utilisation"] <= s["utilisation"]


def test_windows_respect_margins_order_and_short_overlaps():
    L, J = _fake_chain([("a", 0, 200), ("m", 100, 300), ("c", 250, 450)], margins={"m": 15})
    spec = st.Spec(take="x", chunks=[])
    seams = st.plan_seams(spec, L)
    st._choose_windows(spec, L, seams, J, spec.thresholds(), None, None)
    s0, s1 = seams
    assert s0["status"] == s1["status"] == "ok"
    assert s0["allowed"] == [115, 201]                            # m's first 15 frames are never used
    assert s1["allowed"] == [250, 286]                            # ... nor its last 15
    assert s1["n"] == 36 and s1["short"] and not s0["short"]       # the second overlap leaves 36 frames
    assert s0["hi"] <= s1["lo"] and 115 <= s0["lo"] and s0["hi"] <= 201
    assert 250 <= s1["lo"] and s1["hi"] <= 286
    own = st.ownership(spec, L, seams, 0, 450)
    assert (own["owner"] >= 0).all()
    assert own["owned"][1][0] == s0["hi"] and own["owned"][1][1] == s1["lo"]
    # the first / last chunk keep their own edge frames (nothing replaces them)
    assert own["owned"][0][0] == 0 and own["owned"][2][1] == 451


def test_degenerate_chunk_lists_are_reported():
    spec = st.Spec(take="x", chunks=[])
    L, _ = _fake_chain([("a", 0, 300), ("inner", 100, 200)])
    with pytest.raises(ValueError, match="ends inside"):
        st.plan_seams(spec, L)
    L, J = _fake_chain([("a", 0, 100), ("b", 96, 300)])           # 5-frame overlap: no room for a window
    seams = st.plan_seams(spec, L)
    assert seams[0]["status"] == "no_room" and "usable overlap" in seams[0]["why"]
    L, J = _fake_chain([("a", 0, 100), ("b", 130, 300)])
    seams = st.plan_seams(spec, L)
    assert seams[0]["status"] == "gap" and "29 frames uncovered" in seams[0]["why"]


def test_n_cams_from_ma2d_visibilities_with_a_fallback(tmp_path):
    n = 10
    d = tmp_path / "ma_2d"
    d.mkdir()
    vis = np.zeros((n, 2, 512), np.float32)
    vis[:, 0, :100] = 1                                  # lead: 100 landmarks in every frame
    vis[:5, 1, :100] = 1                                 # follow: first 5 frames only
    np.savez(d / "cam01.npz", visibilities=vis)
    vis2 = np.zeros((n, 2, 512), np.float32)
    vis2[5:, 0, :60] = 1                                 # lead again in the second half
    vis2[:, 1, :10] = 1                                  # follow: 10 landmarks = below the 50 minimum
    np.savez(d / "cam02.npz", visibilities=vis2)
    spec = st.Spec(take="x", chunks=[])
    c = st.Loaded(st.Chunk("a", tmp_path, None, 0, n - 1, 0.0, False, "segment", None, d), 0, n - 1, 0, 0, {})
    lead, src = st.chunk_ncams(c, spec, "lead")
    assert src == "ma_2d visibilities" and lead.tolist() == [1] * 5 + [2] * 5
    follow, _ = st.chunk_ncams(c, spec, "follow")
    assert follow.tolist() == [1] * 5 + [0] * 5
    # an ma_2d from another grid (re-prepared after the fit) is not this fit's observations: spans, with a warning
    prep = _prep(tmp_path / "p", {"09": [A - 1, A + 1], "06": [A + 0.09, A + 5]})
    c2 = st.Loaded(st.Chunk("b", tmp_path, None, 0, 6, 0.0, False, "segment", prep, d), 0, 6, 0, 0, {})
    n2, src2 = st.chunk_ncams(c2, spec, "lead")
    assert src2 == "prepare.json camera spans" and c2.warnings and "n_cams from the camera spans" in c2.warnings[0]
    assert n2.tolist() == [1, 1, 1, 2, 2, 2, 2]


# ------------------------------------------------------------------------------------------------ the real take's grid


def test_real_full_take_offsets_from_prepare():
    """The frame offsets of the real larissa_kadu_full segments from their prepare.json (fit_k) and the reused 38 s take."""
    if not (PACKAGE_ROOT / "takes" / "larissa_kadu_full.toml").exists():
        pytest.skip("no larissa_kadu_full take")
    spec = st.spec_from_take("larissa_kadu_full")
    by = {c.name: c for c in spec.chunks}
    assert (by["38"].k_lo, by["38"].k_hi) == (0, 1147)
    assert by["38"].reference and by["38"].margin_s == 0.5 and by["38"].kind == "reuse"
    names = [c.name for c in spec.chunks]
    assert names == sorted(names, key=lambda n: by[n].k_lo)
    for c in spec.chunks:
        prep = c.prepare
        if c.kind == "segment" and prep is not None and prep.exists():
            t = np.asarray(json.loads(prep.read_text())["grid"]["t"], float)
            k = (t - spec.anchor) * 30.0
            assert np.abs(k - np.rint(k)).max() < 1e-6               # every segment grid sits on 38's lattice
            assert (c.k_lo, c.k_hi) == (int(round(k[0])), int(round(k[-1])))
            assert c.k_hi - c.k_lo + 1 == len(t)
    if "a" in by:
        assert by["a"].k_lo == -998 and by["a"].k_hi == 120
        assert by["38"].k_lo - by["a"].k_lo == 998                  # 38's frame i = full-grid frame i + 998
        assert by["a"].k_hi - by["38"].k_lo + 1 == 121               # 4.03 s overlap a|38 (30.15-34.15 s)
    if "b" in by:
        assert by["b"].k_lo == 1027 and by["b"].k_hi == 2312
        assert by["38"].k_hi - by["b"].k_lo + 1 == 121               # 38|b overlap 64.38-68.38 s
    chain = sorted(by.values(), key=lambda c: c.k_lo)
    for p, q in zip(chain, chain[1:]):
        assert q.k_lo <= p.k_hi - 89                                 # every seam overlaps by >= 3 s
    assert spec.thresholds()["body_p95"] == 60 and spec.thresholds()["betas_swap_p95"] == 40
    assert spec.absent["lead"] and spec.mono and spec.low_view
    assert spec.out_refined == "smplx_rf2_noslip"


# ------------------------------------------------------------------------------------------------ stitching


@needs_data
def test_windows_stay_inside_the_trusted_part_of_38_and_do_not_overlap(good):
    res = good["res"]
    seams = {s["name"]: s for s in res["seams"]}
    assert set(seams) == {"a|38", "38|b", "b|c", "c|d"}
    n = 60
    for s in seams.values():
        assert s["status"] == "ok" and s["hi"] - s["lo"] == n and s["gate"]["passed"]
    # 38's first / last 0.5 s (15 frames) are never used: a|38 inside k 15..120, 38|b inside 1027..1132
    assert 15 <= seams["a|38"]["lo"] and seams["a|38"]["hi"] <= 121
    assert seams["38|b"]["lo"] >= 1027 and seams["38|b"]["hi"] <= 1133
    chain = [seams[k] for k in ("a|38", "38|b", "b|c", "c|d")]
    for p, q in zip(chain, chain[1:]):
        assert p["hi"] <= q["lo"]
    for s in chain:                                    # inside both chunks' overlap
        assert s["overlap"][0] <= s["lo"] and s["hi"] <= s["overlap"][1]
    # a 2 s window: tie-break and scan are recorded
    assert seams["b|c"]["tie_break"]["method"] in ("beat_phase", "motion")
    assert len(seams["b|c"]["scan"]["worst"]) > 5
    assert abs(seams["b|c"]["utilisation"] - min(seams["b|c"]["scan"]["worst"])) < 0.2


@needs_data
def test_best_window_is_chosen_and_ties_are_within_tolerance(good):
    for s in good["res"]["seams"]:
        scan = np.array(s["scan"]["worst"])
        i = int(np.argmin(np.abs(np.array(s["scan"]["centre_t"]) - s["centre_t"])))
        assert scan[i] <= scan.min() * 1.05 + 0.02 + 1e-3     # the pick is within the tie tolerance of the best


@needs_data
def test_bit_identity_outside_the_windows(good):
    """38's frames outside the blend windows (and its never-used edge margins) are copied bit for bit."""
    res, work = good["res"], good["work"]
    rep = res["bit_identity"]["raw"]["38"]
    assert rep["identical"] and rep["frames"] > 800
    for r in st.ROLES:
        assert rep["roles"][r]["max_abs_diff_pose"] == 0.0 and rep["roles"][r]["max_abs_diff_transl"] == 0.0
        assert rep["roles"][r]["max_abs_diff_joints"] == 0.0
    # independent check against the real file on 38's k range outside the windows
    seams = {s["name"]: s for s in res["seams"]}
    lo1, hi1 = seams["a|38"]["lo"], seams["a|38"]["hi"]
    lo2, hi2 = seams["38|b"]["lo"], seams["38|b"]["hi"]
    kmin = res["grid"]["k"][0]
    for r in st.ROLES:
        out = st.read_dir(work / "smplx")[r]
        src = st.read_dir(REF / "smplx")[r]
        sel = slice(hi1 - kmin, lo2 - kmin)
        sl38 = slice(hi1, lo2)
        for key in (*st.POSE_KEYS, "transl", "joints", "contact", "floor_contact", "quality", "reproj_px"):
            assert np.array_equal(out[key][sel], src[key][sl38]), (r, key)
        assert out["betas"].tobytes() == src["betas"].tobytes()
        # where 38 does NOT own the output (its first 15 frames, a's) it is not 38's
        assert not np.array_equal(out["transl"][0 - kmin + 0:15 - kmin], src["transl"][0:15])
    # on the written files too
    ver = res["verified_on_disk"]["raw"]
    assert all(v["identical"] for v in ver.values())


@needs_data
def test_reference_edges_are_documented(good):
    ref = good["res"]["bit_identity"]["reference"]
    assert ref["chunk"] == "38" and ref["k"] == [0, N38 - 1] and len(ref["windows"]) == 2
    runs = ref["not_owned"]
    assert sum(r["frames"] for r in runs) + (ref["owned_k"][1] - ref["owned_k"][0]) == N38   # every frame of 38 is accounted for
    by = {r["what"]: r for r in runs}
    assert by["owned by a"]["frames"] >= 15 and by["owned by b"]["frames"] >= 15      # 38's unused edge frames: a's and b's
    assert by["owned by a"]["k"][0] == 0 and by["owned by b"]["k"][1] == N38
    assert by["blend a|38"]["frames"] + by["blend 38|b"]["frames"] == 120


@needs_data
def test_every_chunk_owns_its_frames_bit_identically(good):
    for which in ("raw", "refined"):
        for name, v in good["res"]["bit_identity"][which].items():
            assert v["identical"], (which, name)
            for r in st.ROLES:
                assert v["roles"][r]["t_max_abs_diff_s"] < 1e-9


@needs_data
def test_continuity_at_the_seams(good):
    res = good["res"]
    kmin = res["grid"]["k"][0]
    for r in st.ROLES:
        J = st.read_dir(good["work"] / "smplx")[r]["joints"]
        step = np.linalg.norm(np.diff(J[:, :55], axis=0), axis=-1) * 1000
        for s in res["seams"]:
            a, b = s["lo"] - kmin - 3, s["hi"] - kmin + 3
            c = s["roles"][r]["continuity"]
            assert c["smooth"], (s["name"], r, c)
            assert c["max_joint_step_mm"] <= 1.3 * c["max_joint_step_chunks_mm"] + 1.0
            # the handover frames (window edges) show no jump either
            for edge in (s["lo"] - kmin, s["hi"] - kmin):
                assert step[edge - 1:edge + 1].max() <= 1.5 * step[a:b].max() + 1e-6
        # no seam pop anywhere in the take: the stitched step distribution is the chunks' own
        src_steps = np.concatenate([np.linalg.norm(np.diff(j[:, :55], axis=0), axis=-1).ravel() * 1000
                                    for j in res["_"]["J"][r]])
        assert np.percentile(step, 99.9) <= 1.1 * np.percentile(src_steps, 99.9) + 1.0
        assert step.max() <= 1.1 * src_steps.max() + 1.0


@needs_data
def test_output_contract_and_provenance(good):
    res, work = good["res"], good["work"]
    kmin, kmax = res["grid"]["k"]
    T = kmax - kmin + 1
    assert T == 1901 and res["grid"]["frame_offset"]["38"] == 200
    for r in st.ROLES:
        z = st.read_dir(work / "smplx")[r]
        ref = st.read_dir(REF / "smplx")[r]
        assert len(z["t"]) == T and np.abs(z["t"] - (A + np.arange(kmin, kmax + 1) / 30.0)).max() == 0.0
        assert z["betas"].tobytes() == ref["betas"].tobytes() and z["betas"].shape == (16,)
        assert z["joints"].shape == (T, 55, 3) and z["contact"].shape == (T, 512) and z["contact"].dtype == np.float32
        for key in st.REQUIRED:
            assert key in z
        assert z["segment_owner"].shape == (T,) and list(z["chunk_names"]) == ["a", "38", "b", "c", "d"]
        assert set(np.unique(z["segment_owner"])) == {"a", "38", "b", "c", "d"}                # the owning chunk's name
        assert set(np.unique(z["segment_index"])) == {0, 1, 2, 3, 4}                           # ... and its index
        assert all(z["chunk_names"][i] == n for i, n in zip(z["segment_index"][::97], z["segment_owner"][::97]))
        assert z["quality_flag"].dtype == np.int8 and z["n_cams"].shape == (T,) and list(z["quality_flag_names"]) == list(st.FLAG_NAMES)
        assert z["covered"].all()
        # owners follow the chunk order in time
        assert np.all(np.diff(z["segment_index"].astype(int)) >= 0)
        # the stitched joints are the FK of the stitched parameters (also inside the blended windows)
        pose = st.pose165(z)
        for s in res["seams"]:
            sl = slice(s["lo"] - kmin, s["hi"] - kmin, 7)
            J = st.fk_joints(pose[sl], z["transl"][sl], z["betas"])
            assert np.abs(J - z["joints"][sl][:, :55]).max() < 1e-4
        assert (z["blend_weight"] > 0).sum() == 4 * 60
        assert np.all(z["blend_seam"][z["blend_weight"] > 0] >= 0)
    assert (work / "stitch.json").exists()
    doc = json.loads((work / "stitch.json").read_text())
    assert doc["schema"] == st.SCHEMA and doc["gate"]["passed"] and doc["outputs"]["promoted"]
    assert doc["reference"] == "38" and doc["chunk_names"] == ["a", "38", "b", "c", "d"]
    assert doc["chunks"][1]["betas"]["lead"]["reference"] and doc["chunks"][0]["betas"]["lead"]["pinned"]


@needs_data
def test_refined_stitch_uses_the_same_windows(good):
    res, work = good["res"], good["work"]
    assert res["refined"]["dir"] == "smplx_rf2_noslip" and res["refined"]["seams"]
    rep = res["bit_identity"]["refined"]["38"]
    assert rep["identical"] and rep["frames"] > 800
    kmin = res["grid"]["k"][0]
    for r in st.ROLES:
        raw = st.read_dir(work / "smplx")[r]
        ref = st.read_dir(work / "smplx_rf2_noslip")[r]
        src = st.read_dir(REF / "smplx_rf2_noslip")[r]
        assert np.array_equal(raw["blend_weight"], ref["blend_weight"]) and np.array_equal(raw["segment_owner"], ref["segment_owner"])
        assert np.array_equal(raw["segment_index"], ref["segment_index"])
        sel, sl38 = slice(150 - kmin, 1000 - kmin), slice(150, 1000)
        for key in (*st.POSE_KEYS, "transl", "joints", "stance"):
            assert np.array_equal(ref[key][sel], src[key][sl38]), key
        sm = json.loads(str(ref["smoothing"]))
        assert sm["tool"] == "dancecap.stitch" and set(sm["chunks"]) == {"a", "38", "b", "c", "d"}
        assert sm["chunks"]["38"]["mode"] == "refine"
        # refined joints in a window = FK of the refined stitched parameters
        s = res["seams"][1]
        sl = slice(s["lo"] - kmin, s["hi"] - kmin, 9)
        J = st.fk_joints(st.pose165(ref)[sl], ref["transl"][sl], ref["betas"])
        assert np.abs(J - ref["joints"][sl][:, :55]).max() < 1e-4
        # quality / absent flags are the raw stitch's
        assert np.array_equal(ref["quality_flag"], raw["quality_flag"])


@needs_data
def test_numpy_fk_equals_mamma_dyn_forward(layout):
    """stitch's numpy-only FK (no torch / smplx: runs on a CPU pod) = mamma_dyn._forward's joints to float32 round-off."""
    from dancecap import mamma_dyn as md
    for role, b_extra in (("follow", 0.0), ("lead", 0.7)):
        z = layout["raw38"][role]
        sl = slice(0, 40)
        pose = st.pose165(z)[sl]
        betas = np.asarray(z["betas"], float) + b_extra * (np.arange(16) == 1)       # another body shape too
        ref = md._forward(pose, z["transl"][sl], betas, True)[1]
        mine = st.fk_joints(pose, z["transl"][sl], betas)
        assert np.abs(mine - ref).max() < 2e-5
        assert np.array_equal(mine, st.fk_joints(pose, z["transl"][sl], betas))
    z = layout["raw38"]["follow"]
    assert np.abs(st.fk_joints(st.pose165(z)[:40], z["transl"][:40], z["betas"]) - z["joints"][:40, :55]).max() * 1000 < 0.05
    j0 = st.rest_joints(z["betas"])
    assert j0.shape == (55, 3) and abs(j0[0, 1] - 0.0) < 1.0


@needs_data
def test_fk_of_38_refined_reproduces_its_stored_joints(layout):
    """The smplx FK used for blended frames matches pose_refine's stored joints (refined files) to sub-millimetre."""
    z = layout["ref38"]["lead"]
    sl = slice(100, 1100, 100)
    J = st.fk_joints(st.pose165(z)[sl], z["transl"][sl], z["betas"])
    assert np.abs(J - z["joints"][sl][:, :55]).max() * 1000 < 1.0


@needs_data
def test_flags_from_camera_counts(good):
    """a has 1 camera for 3 s, 2 for 3 s, 3 for 3 s, then 5 (camera spans in prepare.json): mono / two_view / multi."""
    kmin = good["res"]["grid"]["k"][0]
    z = st.read_dir(good["work"] / "smplx")["lead"]
    q, n = z["quality_flag"], z["n_cams"]
    assert (n[:90] == 1).all() and (q[:90] == 1).all()                # first 3 s: cam 09 only = mono
    assert (n[90:180] == 2).all() and (q[90:180] == 2).all()          # + 06 = two_view
    assert (n[180:270] == 3).all() and (q[180:270] == 3).all()
    assert (q[-200:] == 3).all() and (n[-200:] >= 3).all()
    doc = json.loads((good["work"] / "stitch.json").read_text())
    assert doc["flags"]["lead"]["counts"]["mono"] == 90 and doc["flags"]["lead"]["absent_runs_t"] == []
    cov = doc["coverage"]
    assert cov["frames"]["none"] == 0 and cov["frames"]["partial"] == 180            # a's first 6 s are mono / two_view
    assert cov["tiles"][:6] == "PPPPPP" and set(cov["tiles"][7:]) == {"C"}


@needs_data
def test_absent_lead_is_flagged_and_zero_quality(layout, tmp_path):
    t_end = A + (LAYOUT["a"][0] + 60) / 30.0
    spec = make_spec(layout, tmp_path, absent={"lead": [[A + LAYOUT["a"][0] / 30.0, t_end]]})
    res = st.stitch(spec, check=True, sheets=False, verbose=False)
    lead, follow = res["_"]["out"]["lead"], res["_"]["out"]["follow"]
    assert (lead["quality_flag"][:61] == 0).all() and lead["quality_flag"][61] == 1
    assert (lead["quality"][:61] == 0).all() and (lead["quality_parts"][:61] == 0).all() and lead["absent"][:61].all()
    assert follow["quality_flag"][0] == 1 and follow["quality"][0] > 0                 # the follow is tracked there
    assert res["flags"]["lead"]["first_tracked_frame"] == 61
    # quality is zeroed on absent frames by design; everything else (and quality elsewhere) is still a bit-identical copy
    assert res["bit_identity"]["raw"]["a"]["identical"] and res["bit_identity"]["refined"]["a"]["identical"]
    assert res["flags"]["lead"]["absent_runs_t"][0][0] == pytest.approx(A + LAYOUT["a"][0] / 30.0, abs=1e-3)


@needs_data
def test_idempotent_byte_for_byte(layout, tmp_path):
    spec = make_spec(layout, tmp_path)
    r1 = st.stitch(spec, sheets=False, verbose=False)
    sha1 = {p.name + "/" + f.name: hashlib.sha256(f.read_bytes()).hexdigest() for p in (spec.work / "smplx", spec.work / "smplx_rf2_noslip")
            for f in p.glob("*.npz")}
    j1 = json.loads((spec.work / "stitch.json").read_text())
    r2 = st.stitch(spec, sheets=False, verbose=False)
    sha2 = {p.name + "/" + f.name: hashlib.sha256(f.read_bytes()).hexdigest() for p in (spec.work / "smplx", spec.work / "smplx_rf2_noslip")
            for f in p.glob("*.npz")}
    j2 = json.loads((spec.work / "stitch.json").read_text())
    assert sha1 == sha2 and len(sha1) == 4
    for j in (j1, j2):
        j.pop("created"), j.pop("seconds")
    assert j1 == j2
    assert j1["outputs"]["sha256"] == {k: v for k, v in sha1.items()}
    assert r1["gate"]["passed"] and r2["gate"]["passed"]


@needs_data
def test_check_writes_nothing_and_seam_filter(layout, tmp_path):
    spec = make_spec(layout, tmp_path)
    res = st.stitch(spec, check=True, sheets=False, verbose=False)
    assert res["gate"]["passed"] and not spec.work.exists()
    one = st.stitch(spec, only_seam="b|c", sheets=False, verbose=False)
    assert [s["name"] for s in one["seams"]] == ["b|c"] and not spec.work.exists()
    with pytest.raises(SystemExit, match="no seam"):
        st.stitch(spec, only_seam="x|y", sheets=False, verbose=False)


def _variant(layout, tmp_path, name, mutate, **spec_over):
    """Spec where chunk `name` (raw) is replaced by a mutated copy."""
    d = tmp_path / f"var_{name}"
    roles = {r: dict(z) for r, z in st.read_dir(layout["dirs"][name]["raw"]).items()}
    for r in st.ROLES:
        mutate(roles[r], r)
    _save(d, roles)
    spec = make_spec(layout, tmp_path, **spec_over)
    for c in spec.chunks:
        if c.name == name:
            c.src = d
    return spec


@needs_data
def test_a_bad_segment_fails_its_gate_and_recommends_a_refit(layout, tmp_path):
    """Segment c's left elbow bent 0.9 rad away from what b and d see: the b|c and c|d gates fail on the arm, the
    outputs go to stitch_rejected/ (smplx/ untouched) and the refit window is 12 s centred on the blend."""
    def bad(z, role):
        z["body_pose"] = z["body_pose"].copy()
        z["body_pose"][:, (18 - 1) * 3 + 2] += 0.9
        z["joints"] = st.fk_joints(st.pose165(z), z["transl"], z["betas"])
    spec = _variant(layout, tmp_path, "c", bad)
    res = st.stitch(spec, sheets=False, verbose=False)
    assert not res["gate"]["passed"]
    failed = {s["name"]: s for s in res["seams"] if not s["gate"]["passed"]}
    assert {"b|c", "c|d"} <= set(failed)
    assert any("l_wrist" in w or "l_elbow" in w for w in failed["b|c"]["gate"]["why"])
    assert any("elbows / wrists" in w or "body joints" in w for w in failed["b|c"]["gate"]["why"])
    rec = failed["b|c"]["refit"]
    assert rec["frames"] == 360 and rec["window_k"][1] - rec["window_k"][0] == 360
    centre = failed["b|c"]["centre_k"]
    assert abs((rec["window_k"][0] + rec["window_k"][1] - 1) / 2 - centre) <= 1.0
    assert not (spec.work / "smplx" / "lead.npz").exists() and (spec.work / "stitch_rejected" / "smplx" / "lead.npz").exists()
    doc = json.loads((spec.work / "stitch.json").read_text())
    assert doc["outputs"]["promoted"] is False and not doc["gate"]["passed"]
    assert not failed["a|38"] if "a|38" in failed else True
    # --force promotes
    res2 = st.stitch(spec, sheets=False, verbose=False, force=True)
    assert (spec.work / "smplx" / "lead.npz").exists() and res2["outputs"]["forced"]


@needs_data
def test_main_exit_codes(layout, tmp_path):
    spec = make_spec(layout, tmp_path)
    cj = tmp_path / "chunks.json"
    cj.write_text(json.dumps({"take": "syn", "work": str(tmp_path / "w"), "beats": str(REF / "beats.json"), "chunks": [
        {"name": c.name, "src": str(c.src), "refined": str(c.refined), "k": [c.k0, c.k1], "margin_s": c.margin_s,
         "reference": c.reference, "prepare": str(c.prepare) if c.prepare else None} for c in spec.chunks]}))
    assert st.main(["syn", "--chunks", str(cj), "--check", "--no-sheets"]) == 0
    w = tmp_path / "w"
    assert not (w / "smplx").exists() and not (w / "smplx_rf2_noslip").exists() and not (w / "stitch.json").exists()
    assert not any(p.suffix == ".npz" for p in w.rglob("*")) if w.exists() else True    # --check writes no motion
    assert st.main(["syn", "--chunks", str(cj), "--no-sheets"]) == 0
    assert (tmp_path / "w" / "smplx" / "follow.npz").exists() and (tmp_path / "w" / "stitch.json").exists()


@needs_data
def test_quaternion_sign_flips_do_not_change_the_stitch(layout, tmp_path):
    """Segment b stored with every rotation on the OTHER axis-angle branch ((theta - 2 pi) about the same axis, which
    also flips the quaternion sign): the stitched rotations are the same ones."""
    def flip(z, role):
        for key in ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose"):
            v = z[key].reshape(len(z[key]), -1, 3)
            th = np.linalg.norm(v, axis=-1, keepdims=True)
            z[key] = np.where(th > 0.05, v / np.maximum(th, 1e-12) * (th - 2 * np.pi), v).reshape(z[key].shape)
    spec0 = make_spec(layout, tmp_path / "o")
    spec1 = _variant(layout, tmp_path, "b", flip)
    spec1.work = tmp_path / "flipped"
    r0 = st.stitch(spec0, check=True, sheets=False, verbose=False)
    r1 = st.stitch(spec1, check=True, sheets=False, verbose=False)
    s0 = {s["name"]: (s["lo"], s["hi"]) for s in r0["seams"]}
    s1 = {s["name"]: (s["lo"], s["hi"]) for s in r1["seams"]}
    assert s0 == s1
    kmin = r0["grid"]["k"][0]
    for r in st.ROLES:
        for key in ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose"):
            a, b = r0["_"]["out"][r][key], r1["_"]["out"][r][key]
            ra = Rot.from_rotvec(a.reshape(-1, 3))
            rb = Rot.from_rotvec(b.reshape(-1, 3))
            assert (ra * rb.inv()).magnitude().max() < 1e-5, key
        # the output stays continuous as rotations through the 38|b window (b is the flipped chunk)
        o = r1["_"]["out"][r]["body_pose"]
        s = r1["seams"][1]
        a, bb = s["lo"] - kmin - 2, s["hi"] - kmin + 2
        q = Rot.from_rotvec(o[a:bb].reshape(-1, 3)).as_matrix().reshape(bb - a, -1, 3, 3)
        step = np.arccos(np.clip((np.einsum("tjab,tjab->tj", q[1:], q[:-1]) - 1) / 2, -1, 1))
        assert step.max() < 0.35
        # inside the window the parameters stay on the earlier chunk's branch (no 2 pi jump of the vector)
        w = o[s["lo"] - kmin:s["hi"] - kmin]
        assert np.abs(np.diff(w, axis=0)).max() < 0.3


@needs_data
def test_unpinned_betas_are_reexpressed_and_gated(layout, tmp_path):
    """Segment b fitted with other betas: re-expressed with 38's keeping its pelvis trajectory; a big shape change
    fails the betas-swap gate."""
    ref_betas = layout["raw38"]["lead"]["betas"]

    def other(delta):
        def f(z, role):
            z["betas"] = np.asarray(ref_betas, float) + delta
            z["joints"] = st.fk_joints(st.pose165(z), z["transl"], z["betas"])
        return f
    spec = _variant(layout, tmp_path, "b", other(np.r_[0.4, np.zeros(15)]))
    res = st.stitch(spec, check=True, sheets=False, verbose=False)
    b = next(c for c in res["chunks"] if c["name"] == "b")["betas"]["lead"]
    assert not b["pinned"] and b["re_expressed"] and b["swap_mm"]["p95"] > 1.0
    out = res["_"]["out"]["lead"]
    assert out["betas"].tobytes() == np.asarray(ref_betas).tobytes()
    # b's pelvis trajectory is unchanged by the re-expression (its own file's pelvis joint)
    orig = st.read_dir(spec.chunks[[c.name for c in spec.chunks].index("b")].src)["lead"]["joints"][:, 0]
    k0 = LAYOUT["b"][0]
    own_b = next(c for c in res["chunks"] if c["name"] == "b")["owned_k"]
    ks = np.arange(own_b[0], own_b[1])
    kmin = res["grid"]["k"][0]
    assert np.abs(out["joints"][ks - kmin, 0] - orig[ks - k0]).max() < 2e-4
    bad = _variant(layout, tmp_path, "b", other(np.r_[3.0, np.zeros(15)]))
    res2 = st.stitch(bad, check=True, sheets=False, verbose=False)
    assert not res2["gate"]["passed"] and any("betas swap" in w for w in res2["gate"]["why"])


@needs_data
def test_partial_stitch_flags_the_gap(layout, tmp_path):
    spec = make_spec(layout, tmp_path)
    for c in spec.chunks:
        if c.name == "d":
            c.src = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError, match="no fit yet"):
        st.stitch(spec, check=True, sheets=False, verbose=False)
    res = st.stitch(spec, check=True, partial=True, sheets=False, verbose=False)
    assert res["missing_chunks"] == ["d"] and res["grid"]["k"][1] == LAYOUT["d"][1]
    out = res["_"]["out"]["follow"]
    assert not out["covered"][-5:].any() and out["covered"][:500].all()
    assert (out["quality_flag"][-5:] == 0).all() and (out["segment_index"][-5:] == -1).all()
    assert (out["segment_owner"][-5:] == "").all() and out["segment_owner"][0] == "a"
    assert res["coverage"]["frames"]["none"] > 100 and set(res["coverage"]["tiles"]) >= {"N", "C"}


@needs_data
def test_chunks_need_overlap(layout, tmp_path):
    """Two chunks that only touch cannot be blended: the seam is reported and gates fail."""
    spec = make_spec(layout, tmp_path)
    spec.chunks = [c for c in spec.chunks if c.name in ("38", "d")]      # 38 ends k 1147, d starts 1400: a gap
    res = st.stitch(spec, check=True, sheets=False, verbose=False)
    s = res["seams"][0]
    assert s["status"] == "gap" and not res["gate"]["passed"] and "uncovered" in s["why"]
    assert res["_"]["out"]["lead"]["covered"].sum() < res["grid"]["T"]


@needs_data
def test_sheets_are_written(layout, tmp_path):
    spec = make_spec(layout, tmp_path)
    res = st.stitch(spec, verbose=False)
    for k in ("pose_seams_full", "stitch_overview"):
        p = Path(res["_"] and json.loads((spec.work / "stitch.json").read_text())["review"][k])
        assert p.exists() and p.stat().st_size > 20_000


def test_arg_errors_name_the_problem(tmp_path):
    with pytest.raises(FileNotFoundError):
        st.stitch(st.Spec(take="x", chunks=[st.Chunk("a", tmp_path / "none")], work=tmp_path / "w"), verbose=False)


# ------------------------------------------------------------------------------------------------ pod-ready: headless, work root, job


def test_stitch_is_numpy_scipy_only_and_headless():
    """Nothing heavy at import: no torch / smplx / mamma_dyn / mamma_chunks / matplotlib (a CPU pod runs it)."""
    import subprocess
    import sys
    code = ("import sys, dancecap.stitch as st, dancecap.stitch_job; "
            "bad = [m for m in ('torch', 'smplx', 'dancecap.mamma_dyn', 'dancecap.mamma_chunks', 'matplotlib', 'cv2') if m in sys.modules]; "
            "assert not bad, bad; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(PACKAGE_ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


def test_work_root_env_and_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DANCECAP_WORK_ROOT", str(tmp_path / "wr"))
    assert st.work_root() == (tmp_path / "wr").resolve()
    assert st._wp("a/b") == (tmp_path / "wr").resolve() / "a" / "b"
    assert st._wp(str(tmp_path / "abs")) == tmp_path / "abs"
    monkeypatch.delenv("DANCECAP_WORK_ROOT")
    assert st.work_root() == PACKAGE_ROOT / "work"


@needs_data
def test_chunk_list_round_trip_relative_to_the_work_root(layout, tmp_path, monkeypatch):
    """spec -> chunk list JSON (paths under the work root become relative) -> spec: the same chunks; the JSON is portable."""
    monkeypatch.setenv("DANCECAP_WORK_ROOT", str(layout["root"]))
    spec = make_spec(layout, tmp_path, mono=[[1.0, 2.0]], absent={"lead": [[0.0, 0.5]]}, gate={"body_p95": 55.0})
    spec.world_id, spec.pose_id = "w1", "p1"
    cl = st.chunk_list(spec)
    by = {c["name"]: c for c in cl["chunks"]}
    assert by["a"]["src"] == "a/raw" and by["a"]["refined"] == "a/ref" and by["a"]["prepare"] == "a/p/prepare.json"
    assert by["38"]["reference"] is True and by["38"]["k"] == [0, N38 - 1] and Path(by["38"]["src"]).is_absolute()
    assert cl["gate_mm"] == {"body_p95": 55.0} and cl["lead_absent"] == [[0.0, 0.5]] and cl["world_id"] == "w1"
    f = tmp_path / "cl.json"
    f.write_text(json.dumps(cl))
    back = st.spec_from_json(f)
    assert [c.name for c in back.chunks] == [c.name for c in spec.chunks]
    for c0, c1 in zip(spec.chunks, back.chunks):
        assert (c0.src, c0.refined, c0.k0, c0.k1, c0.margin_s, c0.reference) == (c1.src, c1.refined, c1.k0, c1.k1, c1.margin_s, c1.reference)
    assert back.thresholds() == spec.thresholds() and back.mono == spec.mono and back.absent == spec.absent
    assert back.world_id == "w1" and back.pose_id == "p1" and back.out_refined == spec.out_refined


def _stub(name, world=None, pose=None):
    return st.Loaded(st.Chunk(name, Path("x"), world_id=world, pose_id=pose), 0, 9, 0, 0, {})


def test_world_guard():
    spec = st.Spec(take="x", chunks=[])
    same = st.check_worlds([_stub("a", "w1", "p1"), _stub("b", "w1", "p1")], spec)
    assert same["pose_ids"] == ["p1"] and not same["warnings"]
    avatar = st.check_worlds([_stub("a", "w1", "p1"), _stub("b", "w2", "p1")], spec)       # avatar-only difference
    assert avatar["pose_ids"] == ["p1"] and "avatar" in avatar["warnings"][0]
    with pytest.raises(st.WorldMismatch, match="different worlds"):
        st.check_worlds([_stub("a", "w1", "p1"), _stub("b", "w2", "p2")], spec)
    spec.allow_mixed_world = True
    assert "MIXED" in st.check_worlds([_stub("a", "w1", "p1"), _stub("b", "w2", "p2")], spec)["warnings"][0]
    spec.allow_mixed_world = False
    unk = st.check_worlds([_stub("a", "w1", "p1"), _stub("b")], spec)                        # unknown: reported, not refused
    assert unk["unknown"] == ["b"] and "no world id" in unk["warnings"][0]
    spec.require_world = True
    with pytest.raises(st.WorldMismatch, match="carry no world_id"):
        st.check_worlds([_stub("a", "w1", "p1"), _stub("b")], spec)
    spec2 = st.Spec(take="x", chunks=[], world_id="wd", pose_id="pd")                         # the list's default world
    assert st.check_worlds([_stub("a"), _stub("b", "wd", "pd")], spec2)["world_ids"] == ["wd"]


def test_world_ids_come_from_a_chunk_header(tmp_path):
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / "chunk.json").write_text(json.dumps({"world": {"id": "w9", "pose_id": "p9"}}))
    f = tmp_path / "cl.json"
    f.write_text(json.dumps({"take": "x", "chunks": [
        {"name": "a", "src": str(tmp_path / "s"), "header": str(tmp_path / "h" / "chunk.json")},
        {"name": "b", "src": str(tmp_path / "t"), "world_id": "w1"}]}))
    sp = st.spec_from_json(f)
    assert (sp.chunks[0].world_id, sp.chunks[0].pose_id) == ("w9", "p9") and sp.chunks[1].world_id == "w1"


@needs_data
def test_main_relative_work_root_out_and_dance_label(layout, tmp_path, monkeypatch):
    """python -m dancecap.stitch <dance> --chunks <json> --work-root <root> --out <rel>: every relative path under the root."""
    monkeypatch.setenv("DANCECAP_WORK_ROOT", str(layout["root"]))     # restored at teardown whatever main sets
    spec = make_spec(layout, tmp_path)
    cl = st.chunk_list(spec)
    cl["take"] = "label_in_json"            # the dance named on the command line wins over the list's label
    (layout["root"] / "relative_chunks.json").write_text(json.dumps(cl))
    rc = st.main(["mydance", "--chunks", str(layout["root"] / "relative_chunks.json"), "--work-root", str(layout["root"]),
                  "--out", "rel_out", "--no-sheets"])
    assert rc == 0
    out = layout["root"] / "rel_out"
    doc = json.loads((out / "stitch.json").read_text())
    assert doc["take"] == "mydance" and doc["gate"]["passed"] and (out / "smplx" / "lead.npz").exists()
    assert (out / "smplx_rf2_noslip" / "follow.npz").exists() and not (layout["root"] / "mydance").exists()
    # check without --out writes nothing (the default folder would be <work root>/<dance>)
    assert st.main(["mydance", "--chunks", str(layout["root"] / "relative_chunks.json"), "--work-root", str(layout["root"]),
                    "--check", "--no-sheets"]) == 0
    assert not (layout["root"] / "mydance").exists()


@needs_data
def test_check_with_out_writes_the_report_and_sheets_but_no_motion(layout, tmp_path, monkeypatch):
    monkeypatch.setenv("DANCECAP_WORK_ROOT", str(tmp_path))
    spec = make_spec(layout, tmp_path)
    cj = tmp_path / "cl.json"
    cj.write_text(json.dumps(st.chunk_list(spec)))
    out = tmp_path / "chk"
    assert st.main(["syn", "--chunks", str(cj), "--out", str(out), "--check"]) == 0
    doc = json.loads((out / "stitch_check.json").read_text())
    assert doc["check"] and doc["gate"]["passed"] and len(doc["seams"]) == 4
    assert all(s["ok"] and len(s["window_k"]) == 2 for s in doc["seams"])
    for k in ("pose_seams_full", "stitch_overview"):
        assert Path(doc["review"][k]).exists() and Path(doc["review"][k]).stat().st_size > 20_000
    assert not (out / "smplx").exists() and not (out / "stitch.json").exists()


@needs_data
def test_gate_exit_report_vs_strict_and_world_refusal(layout, tmp_path, monkeypatch):
    monkeypatch.setenv("DANCECAP_WORK_ROOT", str(tmp_path))

    def bad(z, role):
        z["body_pose"] = z["body_pose"].copy()
        z["body_pose"][:, (18 - 1) * 3 + 2] += 0.9
        z["joints"] = st.fk_joints(st.pose165(z), z["transl"], z["betas"])
    spec = _variant(layout, tmp_path, "c", bad)
    cj = tmp_path / "cl.json"
    cj.write_text(json.dumps(st.chunk_list(spec)))
    base = ["syn", "--chunks", str(cj), "--no-sheets", "--check"]
    assert st.main(base) == 2                                          # strict: a failing gate is exit 2
    assert st.main(base + ["--gate-exit", "report"]) == 0              # a pod job pulls the verdict instead of failing
    # a pod stitch that fails its gate keeps the motion in stitch_rejected/ and still exits 0 with --gate-exit report
    out = tmp_path / "rej"
    assert st.main(["syn", "--chunks", str(cj), "--out", str(out), "--no-sheets", "--gate-exit", "report"]) == 0
    assert (out / "stitch_rejected" / "smplx" / "lead.npz").exists() and not (out / "smplx").exists()
    names = [s["name"] for s in json.loads((out / "stitch.json").read_text())["seams"]]
    bc = json.loads((out / "stitch.json").read_text())["seams"][names.index("b|c")]
    assert bc["ok"] is False and bc["refit"]["frames"] == 360
    # different pose worlds: refused (exit 3) unless --allow-mixed-world
    cl = json.loads(cj.read_text())
    cl["chunks"][0]["world_id"], cl["chunks"][0]["pose_id"] = "w1", "p1"
    cl["chunks"][1]["world_id"], cl["chunks"][1]["pose_id"] = "w2", "p2"
    cj.write_text(json.dumps(cl))
    assert st.main(["syn", "--chunks", str(cj), "--no-sheets", "--check"]) == 3
    assert st.main(["syn", "--chunks", str(cj), "--no-sheets", "--check", "--allow-mixed-world", "--gate-exit", "report"]) == 0


def test_stitch_job_spec_shape_and_templates(tmp_path):
    from dancecap import stitch_job as sj
    cl = {"take": "x", "out_refined": "smplx_rf2_noslip", "chunks": [
        {"name": "a", "src": "x_a/mamma/fits/c8/smplx", "k": [-998, 120]},
        {"name": "38", "src": "ref/smplx", "k": [0, 1147], "reference": True}]}
    job = sj.build_job("dance1", cl, world_id="w1", pose_id="p1", world_sha256="abc", code={"git_sha": "g"},
                       job_id="stitch-dance1-t")
    assert job["schema"] == "dancecap.farm.job/1" and job["kind"] == "analyse" and job["needs_gpu"] is False
    assert job["cwd"] == "dancecap" and job["env"]["DANCECAP_WORK_ROOT"] == "{WS}/dancecap/work"
    assert (job["world_id"], job["world_sha256"], job["pose_id"]) == ("w1", "abc", "p1") and job["code"] == {"git_sha": "g"}
    assert job["chunk"]["k0"] == -998 and job["chunk"]["k1"] == 1147
    assert [s["name"] for s in job["steps"]] == ["stitch"]               # nothing to stage: read in place
    s = job["steps"][0]
    assert s["cmd"] == ("python -m dancecap.stitch dance1 --chunks work/dance1/farm/stitch/stitch-dance1-t.chunks.json "
                        "--work-root work --out dance1 --gate-exit report")
    assert s["timeout_min"] == 35.0 and s["optional"] is False and s["env"]["DANCECAP_POD"] == "1"
    assert job["max_minutes"] == 60.0 and job["priority"] == 1000.0
    out_paths = [o["path"] for o in job["outputs"]]
    assert out_paths[0] == "dancecap/work/dance1/stitch.json" and "dancecap/work/dance1/smplx/" in out_paths
    assert "dancecap/work/dance1/smplx_rf2_noslip/" in out_paths and "dancecap/work/dance1/review/" in out_paths
    assert job["outputs"][0].get("optional") is None and job["outputs"][1]["optional"] is True
    inp = job["inputs"][0]
    assert inp["path"] == "dancecap/work/dance1/farm/stitch/stitch-dance1-t.chunks.json"
    text = job["_chunk_list_text"].encode()
    assert inp["sha256"] == hashlib.sha256(text).hexdigest() and inp["bytes"] == len(text)
    emb = json.loads(job["_chunk_list_text"])
    assert emb["world_id"] == "w1" and emb["pose_id"] == "p1" and emb["take"] == "dance1"
    for k in ("id", "dance", "kind", "take", "chunk", "world_id", "world_sha256", "code", "cams", "sources", "inputs", "steps",
              "outputs", "max_minutes", "priority", "estimate", "created"):       # the farm's job-spec keys
        assert k in job
    # tars add the staging step (first); partial and check modes
    jt = sj.build_job("dance1", cl, tars={"a": {"path": "chunks/a/out/result.tar", "sha256": "ff"}}, partial=True, code={},
                      job_id="j")
    assert [x["name"] for x in jt["steps"]] == ["stage", "stitch"] and jt["steps"][1]["cmd"].endswith("--gate-exit report --partial")
    assert jt["steps"][0]["cmd"] == "python -m dancecap.stitch_job stage --chunks work/dance1/farm/stitch/j.chunks.json --work-root work"
    assert json.loads(jt["_chunk_list_text"])["chunks"][0]["tar"] == {"path": "chunks/a/out/result.tar", "sha256": "ff"}
    jc = sj.build_job("dance1", cl, mode="check", code={}, job_id="jc")
    assert jc["steps"][-1]["name"] == "stitch_check" and " --check " in jc["steps"][-1]["cmd"]
    assert "--out dance1_check" in jc["steps"][-1]["cmd"]
    assert [o["path"] for o in jc["outputs"]] == ["dancecap/work/dance1_check/stitch_check.json", "dancecap/work/dance1_check/review/"]
    with pytest.raises(ValueError):
        sj.build_job("dance1", cl, mode="nope")
    # register() adds the templates to the farm's dict without replacing anything
    farm = {"fit": [{"name": "fit"}], "stitch": [{"name": "mine"}]}
    sj.register(farm)
    assert farm["stitch"] == [{"name": "mine"}] and [s["name"] for s in farm["stitch_check"]] == ["stage", "stitch_check"]
    # write_job: the spec (without the embedded text) + the input file at the workspace-relative path
    sp, cp = sj.write_job(job, local_root=tmp_path)
    assert json.loads(sp.read_text())["id"] == "stitch-dance1-t" and "_chunk_list_text" not in json.loads(sp.read_text())
    assert cp == tmp_path / inp["path"] and hashlib.sha256(cp.read_bytes()).hexdigest() == inp["sha256"]


def test_stage_extracts_result_tars_safely(tmp_path):
    import io
    import tarfile
    from dancecap import stitch_job as sj
    vol, ws = tmp_path / "vol", tmp_path / "ws"
    src = tmp_path / "src" / "dancecap" / "work" / "t_a" / "mamma" / "fits" / "c8" / "smplx"
    src.mkdir(parents=True)
    (src / "lead.npz").write_bytes(b"lead-bytes")
    (vol / "chunks" / "a" / "out").mkdir(parents=True)
    tp = vol / "chunks" / "a" / "out" / "result.tar"
    with tarfile.open(tp, "w") as t:
        t.add(src / "lead.npz", arcname="dancecap/work/t_a/mamma/fits/c8/smplx/lead.npz")
    sha = hashlib.sha256(tp.read_bytes()).hexdigest()
    cl = tmp_path / "cl.json"
    cl.write_text(json.dumps({"chunks": [{"name": "a", "tar": {"path": "chunks/a/out/result.tar", "sha256": sha}}, {"name": "b"}]}))
    rep = sj.stage(cl, volume=vol, dest=ws)
    assert rep["tars"][0]["files"] == 1
    assert (ws / "dancecap/work/t_a/mamma/fits/c8/smplx/lead.npz").read_bytes() == b"lead-bytes"
    sj.stage(cl, volume=vol, dest=ws)                                        # idempotent
    cl.write_text(json.dumps({"chunks": [{"name": "a", "tar": {"path": "chunks/a/out/result.tar", "sha256": "0" * 64}}]}))
    with pytest.raises(ValueError, match="sha256"):
        sj.stage(cl, volume=vol, dest=ws)
    cl.write_text(json.dumps({"chunks": [{"name": "a", "tar": "chunks/missing.tar"}]}))
    with pytest.raises(FileNotFoundError, match="not on the volume"):
        sj.stage(cl, volume=vol, dest=ws)
    with tarfile.open(vol / "evil.tar", "w") as t:
        ti = tarfile.TarInfo("../outside.txt")
        ti.size = 1
        t.addfile(ti, io.BytesIO(b"x"))
    cl.write_text(json.dumps({"chunks": [{"name": "a", "tar": "evil.tar"}]}))
    with pytest.raises(ValueError, match="unsafe path"):
        sj.stage(cl, volume=vol, dest=ws)
    assert not (tmp_path / "outside.txt").exists()


@needs_data
def test_the_pod_job_runs_end_to_end_like_the_worker(layout, tmp_path, monkeypatch):
    """Run the built job's steps the way the worker does (cwd = <workspace>/dancecap, DANCECAP_WORK_ROOT, a shell command):
    the stitch step writes every non-optional declared output at its workspace-relative path."""
    import os
    import subprocess
    import sys
    from dancecap import stitch_job as sj
    spec = make_spec(layout, tmp_path)
    spec.chunks = [c for c in spec.chunks if c.name in ("a", "38")]
    monkeypatch.setenv("DANCECAP_WORK_ROOT", str(tmp_path / "unrelated"))     # no chunk lies under it: absolute paths
    cl = st.chunk_list(spec)
    ws = tmp_path / "ws"
    job = sj.build_job("podance", cl, py=f'"{sys.executable}"', code={}, job_id="stitch-podance-e2e", extra_flags="--no-sheets")
    sj.write_job(job, local_root=ws)
    env = dict(os.environ, PYTHONPATH=str(PACKAGE_ROOT), DANCECAP_WORK_ROOT=str(ws / "dancecap" / "work"), HM_WS=str(ws),
               PYTHONUNBUFFERED="1")
    env.update(job["steps"][0]["env"])
    for step in job["steps"]:
        r = subprocess.run(step["cmd"], shell=True, cwd=ws / "dancecap", env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, (step["name"], r.stdout[-1500:], r.stderr[-1500:])
    for o in job["outputs"]:
        if not o.get("optional"):
            assert (ws / o["path"]).exists(), o
    doc = json.loads((ws / "dancecap" / "work" / "podance" / "stitch.json").read_text())
    assert doc["take"] == "podance" and doc["gate"]["passed"] and [s["name"] for s in doc["seams"]] == ["a|38"]
    assert (ws / "dancecap/work/podance/smplx/lead.npz").exists()
    assert (ws / "dancecap/work/podance/smplx_rf2_noslip/lead.npz").exists()


def test_a_fit_up_to_two_frames_inside_its_spec_window_is_accepted_with_a_warning(tmp_path):
    """lucas_luiza llg0000: mamma prepare's grid began at k 507 (the take's first instant had no camera frame before it),
    the spec at 506: the stitch failed on two pods. A fit <= EDGE_TOL_FRAMES inside its window is accepted (the fit wins);
    further off, or outside the window, it is still refused."""
    import numpy as np
    import pytest
    from dancecap import stitch as ST

    def fit_dir(d, k0, k1):
        d.mkdir(parents=True, exist_ok=True)
        n = k1 - k0 + 1
        t = 0.0 + np.arange(k0, k1 + 1) / 30.0
        for r in ST.ROLES:
            arr = {"t": t, "transl": np.zeros((n, 3)), "betas": np.zeros(16), "joints": np.zeros((n, 55, 3))}
            arr.update({k: np.zeros((n, 63 if k == "body_pose" else 45 if "hand" in k else 3)) for k in ST.POSE_KEYS})
            np.savez(d / f"{r}.npz", **arr)
        return d

    spec = ST.Spec(take="t", chunks=[], anchor=0.0)
    src = fit_dir(tmp_path / "a", 507, 600)
    L = ST.load_chunk(ST.Chunk(name="a", src=src, k0=506, k1=600), spec, want_refined=False)
    assert (L.k0, L.k1) == (507, 600) and any("the fit wins" in w for w in L.warnings)
    with pytest.raises(ValueError):
        ST.load_chunk(ST.Chunk(name="a", src=src, k0=503, k1=600), spec, want_refined=False)   # 4 frames inside
    with pytest.raises(ValueError):
        ST.load_chunk(ST.Chunk(name="a", src=src, k0=508, k1=600), spec, want_refined=False)   # the fit starts before the spec
