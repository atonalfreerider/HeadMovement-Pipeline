"""dancecap.moves features / proposals / dataset / model: data plumbing only (no classifier is trained or run).

    ..\\.venv-nopo4d\\Scripts\\python -m pytest tests/test_moves_features.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import synthetic as syn  # noqa: E402
from dancecap.moves import features as F  # noqa: E402
from dancecap.moves import proposals as P  # noqa: E402
from dancecap.moves.dataset import (X_FEATURES, DatasetConfig, DatasetError, build_dataset,  # noqa: E402
                                    load_windows, window_starts, windows_for)
from dancecap.moves.labels import Labels, Segment, save_labels, validate_labels  # noqa: E402
from dancecap.moves.model import (MoveClassifier, merge_into_gaps, readiness, segments_from_votes,  # noqa: E402
                                  window_votes)
from dancecap.take import PACKAGE_ROOT, WORKSPACE  # noqa: E402

SYN = PACKAGE_ROOT / "work" / "synthetic"
SYN_CB = PACKAGE_ROOT / "work" / "synthetic_cb"
needs_syn = pytest.mark.skipif(not (SYN / "smplx" / "lead.npz").exists(), reason="work/synthetic not generated")


def _grid(rows):
    return np.array([float(r[0]) for r in rows]), np.array([int(r[1]) for r in rows])


def _window_npzs(t0=0.0, t1=12.0):
    w = syn.window(SYN, t0, t1)
    return w["npz"], _grid(w["beats"])


@pytest.fixture(scope="module")
def syn_fs():
    npzs, grid = _window_npzs()
    td = json.loads((SYN / "truth.json").read_text())["touchdowns"]
    return F.compute_features(npzs, F.FeatureConfig(com="joints"), grid, None,
                              {r: [e["t"] for e in td[r]] for r in F.ROLES}, take="synthetic")


# ----------------------------------------------------------------------------------------------- features

def test_feature_list_contract():
    assert len(F.NAMES) == len(set(F.NAMES))
    assert F.INDEX == {n: i for i, n in enumerate(F.NAMES)}
    groups = {f.group for f in F.FEATURES}
    assert groups == {"couple", "dancer", "counterbalance", "beat"}
    for f in F.FEATURES:
        assert f.unit and f.description and f.floor > 0
        assert (f.role is None) == (f.group != "dancer")
        if f.group == "beat":
            assert not f.novelty
    # both dancers carry the same per-dancer descriptors (fixed role order lead, follow), + follow_yaw_rel
    lead = {n[5:] for n in F.NAMES if n.startswith("lead_")} - {"hand_on_follow"}
    follow = {n[7:] for n in F.NAMES if n.startswith("follow_")} - {"yaw_rel", "hand_on_lead", "lateral", "forward"}
    assert lead == follow
    for name in ("couple_speed", "couple_yaw_rate", "follow_yaw_rate", "follow_head_off_axis", "com_distance",
                 "hand_hand_min", "lead_hand_on_follow", "lead_arm_ext_l", "follow_contact_r", "couple_ke",
                 "lead_com_height", "cb_score", "beat_phase", "measure_phase"):
        assert name in F.INDEX, name


@needs_syn
def test_shapes_units_and_roundtrip(syn_fs, tmp_path):
    fs = syn_fs
    T, Fn = fs.X.shape
    assert Fn == len(F.FEATURES) and T == len(fs.t) and fs.X.dtype == np.float32
    N = len(fs.eighths["t0"])
    assert fs.pooled.shape == (N, Fn, 4) == fs.pooled_z.shape and fs.z_scale.shape == (Fn, 4)
    assert N >= 30                                            # 12 s at 88 BPM = 35 eighths
    assert np.all(np.diff(fs.eighths["t0"]) > 0) and np.all(fs.eighths["t1"] > fs.eighths["t0"])
    assert set(fs.eighths["slot"]) <= set(range(8))
    # no counterbalance.json -> cb_* NaN, everything else finite
    cb = [n for n in F.NAMES if n.startswith("cb_")]
    assert np.isnan(fs.X[:, [F.INDEX[n] for n in cb]]).all()
    rest = [F.INDEX[n] for n in F.NAMES if n not in cb]
    assert np.isfinite(fs.X[:, rest]).all()
    for n in ("beat_phase", "measure_phase"):
        x = fs.col(n)
        assert x.min() >= 0 and x.max() < 1
    for n in ("lead_contact_l", "follow_contact_r"):
        assert 0 <= fs.col(n).min() and fs.col(n).max() <= 1
    assert set(np.unique(fs.col("n_hand_holds"))) <= {0.0, 1.0, 2.0}
    assert 0.5 < np.median(fs.col("follow_arm_ext_l")) <= 1.0
    # robust z uses the floors: a near-constant feature is not blown up
    assert np.all(fs.z_scale >= np.array([f.floor for f in F.FEATURES])[:, None] - 1e-12)
    src = {"smplx_dir": "test", "com": "joints"}
    npz_p, sch_p = F.save(fs, src, tmp_path)
    back = F.load(npz_p)
    assert np.array_equal(back.X, fs.X, equal_nan=True)
    assert np.array_equal(back.pooled_z, fs.pooled_z, equal_nan=True)
    assert np.array_equal(back.eighths["slot"], fs.eighths["slot"]) and back.phase0 == fs.phase0
    sch = json.loads(sch_p.read_text())
    assert sch["version"] == 1 and sch["time_base"] == "reference" and sch["stats"] == list(F.STATS)
    assert [f["name"] for f in sch["features"]] == F.NAMES
    assert all(f["unit"] for f in sch["features"])
    assert sch["files"]["npz_sha256_16"] == F._sha16(npz_p)


@needs_syn
def test_couple_frame_on_synthetic(syn_fs):
    """The synthetic couple faces each other (lead at +z facing -z at t = 0), the couple rotates once every 4
    measures counter-clockwise about +Y, its centre stays at the origin, every beat is a step."""
    fs = syn_fs
    lng, fwd = fs.frame["frame_long"], fs.frame["frame_forward"]
    assert np.allclose(np.linalg.norm(lng, axis=1), 1) and np.allclose(np.linalg.norm(fwd, axis=1), 1)
    assert np.allclose((lng * fwd).sum(1), 0, atol=1e-9)
    assert np.allclose(fwd, np.stack([-lng[:, 1], lng[:, 0]], 1))      # forward = long x up
    # t = 0: the lead faces -z, so his left (the long axis) points to -x and forward is -z
    assert np.allclose(fwd[0], [0, -1], atol=0.05) and np.allclose(lng[0], [-1, 0], atol=0.05)
    npzs, _ = _window_npzs()
    spine3 = np.asarray(npzs["lead"]["joints"])[:, 9][:, [0, 2]]
    assert np.abs(fs.frame["frame_origin"] - spine3).max() < 0.01
    # the follower is straight in front of the leader's chest, face to face
    assert np.median(fs.col("follow_forward")) > 0.4
    assert np.abs(fs.col("follow_forward") - fs.col("com_distance")).max() < 0.03
    assert np.abs(fs.col("follow_lateral")).max() < 0.06
    assert fs.col("facing_cos").max() < -0.99
    rot = 360.0 / (4 * syn.MEASURE)                                     # deg/s, + = counter-clockwise from above
    assert abs(np.median(fs.col("couple_yaw_rate")) - rot) < 1.0
    assert abs(np.median(fs.col("follow_yaw_rate")) - rot) < 1.0
    assert np.abs(fs.col("follow_yaw_rel")).max() < 2.0
    assert fs.col("couple_speed").max() < 0.02
    assert abs(np.median(fs.col("lead_step_rate")) - 1 / syn.BEAT) < 0.1 / syn.BEAT
    # the follow's head circles at a constant tilt: off-axis angle steady, its direction turning
    assert np.median(fs.col("follow_head_ang_speed")) > 2 * np.median(fs.col("lead_head_ang_speed"))
    # beat phase ~0 at the frame nearest each beat (beats = even slots of the eighth grid)
    beats = [x for k, x in enumerate(fs.grid_t) if (k - fs.phase0) % 2 == 0 and fs.t[2] < x < fs.t[-2]]
    ph = fs.col("beat_phase")[[int(np.argmin(np.abs(fs.t - x))) for x in beats]]
    assert len(beats) > 10 and np.all(np.minimum(ph, 1 - ph) < 0.06)


@needs_syn
def test_couple_frame_is_independent_of_the_room():
    """Rotating + translating the whole capture changes no couple-frame feature (joints mode, exact data)."""
    npzs, grid = _window_npzs(0.0, 6.0)
    a = math.radians(70.0)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    moved = {}
    for r, d in npzs.items():
        d2 = dict(d)
        d2["joints"] = np.asarray(d["joints"]) @ R.T + np.array([2.0, 0.0, -1.0])
        moved[r] = d2
    cfg = F.FeatureConfig(com="joints")
    f0 = F.compute_features(npzs, cfg, grid)
    f1 = F.compute_features(moved, cfg, grid)
    for n in ("follow_lateral", "follow_forward", "facing_cos", "facing_sin", "com_distance", "couple_yaw_rate",
              "axis_yaw_rate", "couple_speed", "lead_torsion", "follow_trunk_lean", "follow_head_off_axis",
              "hand_hand_min", "lead_lean_away", "follow_com_height", "couple_ke", "lead_step_rate"):
        assert np.allclose(f0.col(n), f1.col(n), atol=1e-4), n
    # the frame itself turns with the room
    th0 = np.arctan2(-f0.frame["frame_long"][:, 1], f0.frame["frame_long"][:, 0])
    th1 = np.arctan2(-f1.frame["frame_long"][:, 1], f1.frame["frame_long"][:, 0])
    assert np.allclose(np.angle(np.exp(1j * (th1 - th0))), a, atol=1e-6)   # rotation by +70 deg about +Y


@needs_syn
def test_physics_mode_short_window():
    pytest.importorskip("open3d")
    if not (WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz").exists():
        pytest.skip("SMPL-X model files not available")
    npzs, grid = _window_npzs(0.0, 3.0)
    fs = F.compute_features(npzs, F.FeatureConfig(com="physics"), grid)
    assert np.isfinite(fs.col("couple_ke")).all() and fs.col("couple_ke").min() > 0
    assert abs(np.median(fs.col("lead_com_height"))) < 0.05             # upright stepping ~ standing height
    assert set(np.unique(fs.col("lead_contact_l"))) <= {0.0, 1.0}       # physics contacts are booleans
    assert np.median(fs.col("follow_forward")) > 0.4 and fs.col("facing_cos").max() < -0.99
    # rotation-based head axis: the lead's head is not moved relative to the chest in the synthetic
    assert np.median(fs.col("lead_head_off_axis")) < 3.0
    assert fs.meta["dancers"]["lead"]["com_source"].startswith("physics")


def test_pool_eighths_and_robust_z():
    t = np.arange(0, 4.0, 0.1)
    grid_t = np.arange(0, 4.01, 0.5)
    types = np.array([F.ZOUK_PATTERN[i % 8] for i in range(len(grid_t))])
    e = F.eighth_table(t, grid_t, 0, types)
    assert list(e["slot"]) == list(range(8)) and np.allclose(e["coverage"], 1.0)
    X = np.stack([t, np.full_like(t, 3.0)], 1)
    Pp = F.pool(X, e)
    assert np.allclose(Pp[:, 0, 0], [0.2 + 0.5 * k for k in range(8)])     # mean of 5 frames per eighth
    assert np.allclose(Pp[:, 0, 3], 0.4) and np.allclose(Pp[:, 1, 3], 0.0)  # delta = last - first frame
    Z, c, s = F.robust_z(Pp, np.array([0.01, 0.5]))
    assert np.allclose(Z[:, 1], 0.0) and np.allclose(s[1], 0.5)           # constant feature: scale = floor, z = 0
    assert np.isclose(np.median(Z[:, 0, 0]), 0.0)


# ----------------------------------------------------------------------------------------------- proposals

def test_snap_targets():
    slots = np.arange(24) % 8
    assert list(P.snap_targets(slots, "half")) == [0, 4, 8, 12, 16, 20]
    assert list(P.snap_targets(slots, "measure")) == [0, 8, 16]
    assert len(P.snap_targets(slots, "eighth")) == 24


def test_pick_boundaries_snapping_and_spacing():
    N = 32
    slots = np.arange(N) % 8
    cfg = P.ProposalConfig(threshold=0.3, min_eighths=4)

    def strength(peaks):
        s = np.zeros(N + 1)
        s[0] = s[N] = np.nan
        for b, v in peaks.items():
            s[b] = v
        return s

    # a peak on slot 2 is half-way between two half-measures: the stronger side wins
    assert [b["eighth"] for b in P.pick_boundaries(strength({10: 0.7, 12: 0.5, 8: 0.2}), slots, cfg)] == [12]
    assert [b["eighth"] for b in P.pick_boundaries(strength({10: 0.7, 8: 0.5, 12: 0.2}), slots, cfg)] == [8]
    # nearest half-measure otherwise; the reported strength is the peak's
    out = P.pick_boundaries(strength({17: 0.6}), slots, cfg)
    assert [(b["eighth"], b["peak_eighth"], b["strength"]) for b in out] == [(16, 17, 0.6)]
    # below the threshold, or too close to the analysed range's edges: nothing
    assert P.pick_boundaries(strength({16: 0.29}), slots, cfg) == []
    assert P.pick_boundaries(strength({1: 0.9}), slots, cfg) == []        # would snap to 0 (< min_eighths)
    assert P.pick_boundaries(strength({31: 0.9}), slots, cfg) == []       # no half-measure within reach <= N - 4
    assert [b["eighth"] for b in P.pick_boundaries(strength({2: 0.9}), slots, cfg)] == [4]   # 4 eighths = allowed
    # two peaks snapping within min_eighths: the stronger one survives
    out = P.pick_boundaries(strength({16: 0.5, 19: 0.8}), slots, P.ProposalConfig(threshold=0.3, min_eighths=8))
    assert [b["eighth"] for b in out] == [20]
    # measure snapping: slot 0 only
    out = P.pick_boundaries(strength({13: 0.6}), slots, P.ProposalConfig(threshold=0.3, snap="measure"))
    assert [b["eighth"] for b in out] == [16]


CHANGE = {"couple_ke": 0.4, "couple_speed": 0.6, "couple_yaw_rate": 90.0, "axis_yaw_rate": 90.0,
          "follow_yaw_rel": 150.0, "follow_yaw_rate": 240.0, "follow_com_speed": 0.9, "follow_ke": 0.6}


def _toy_fs(n_eighths: int = 32, change_at: int | None = 16, fps: float = 30.0, eighth_s: float = 0.4,
            invalid: tuple = ()) -> F.FeatureSet:
    """A FeatureSet with constant features except a step change in a few couple features at `change_at`; the frames
    of the eighths in `invalid` are not pooled (a tracking dropout)."""
    grid_t = np.arange(n_eighths + 1) * eighth_s
    types = np.array([F.ZOUK_PATTERN[i % 8] for i in range(len(grid_t))])
    t = np.arange(0, grid_t[-1] - 1e-9, 1.0 / fps)
    X = np.zeros((len(t), len(F.FEATURES)))
    rng = np.random.default_rng(0)
    X += rng.normal(0, 0.002, X.shape)
    X[:, F.INDEX["couple_ke"]] = 0.05
    if change_at is not None:
        # the couple starts travelling and the follower spins: several couple + follower descriptors change
        late = t >= grid_t[change_at]
        for name, v in CHANGE.items():
            X[late, F.INDEX[name]] = v
    X[:, [F.INDEX[n] for n in F.NAMES if n.startswith("cb_")]] = np.nan
    valid = np.ones(len(t), bool)
    for i in invalid:
        valid[(t >= grid_t[i]) & (t < grid_t[i + 1])] = False
    e = F.eighth_table(t, grid_t, 0, types)
    Pp = F.pool(np.where(valid[:, None], X, np.nan), e)
    Z, c, s = F.robust_z(Pp, np.array([f.floor for f in F.FEATURES]))
    return F.FeatureSet(take="toy", t=t, X=X.astype(np.float32), valid=valid, quality=np.ones(len(t)),
                        run=np.zeros(len(t), int), frame={}, grid_t=grid_t, grid_type=types, phase0=0, eighths=e,
                        pooled=Pp.astype(np.float32), pooled_z=Z.astype(np.float32), z_center=c, z_scale=s,
                        meta={"warnings": []})


def test_propose_step_change_contract():
    fs = _toy_fs(32, change_at=24)                     # the last quarter of the take travels and spins
    doc = P.propose(fs, P.ProposalConfig())
    assert P.validate_proposals(doc) == []
    nov = [b for b in doc["boundaries"] if b["kind"] == "novelty"]
    assert [b["eighth"] for b in nov] == [24] and nov[0]["strength"] > 0.5
    assert nov[0]["group"] in ("couple", "follow") and set(nov[0]["strength_by_group"]) >= {"all", "couple", "follow"}
    assert {c["feature"] for c in nov[0]["top_changes"]} <= set(CHANGE)
    assert set(doc["novelty"]["groups"]) >= {"all", "couple", "lead", "follow"}
    assert "counterbalance" not in doc["novelty"]["groups"]                # cb_* NaN -> no counterbalance group
    segs = doc["segments"]
    assert len(segs) == 2 and segs[0]["t0"] == pytest.approx(fs.t[0]) and segs[1]["t0"] == pytest.approx(9.6)
    assert segs[1]["t1"] == pytest.approx(doc["coverage"][1]) and segs[1]["n_eighths"] == 8
    assert segs[0]["t1"] == segs[1]["t0"]
    assert all(s["move"] is None and s["node"] is None and s["provenance"] == "auto" and s["tier"] == "couple"
               for s in segs)
    assert segs[1]["energy_rank"] > segs[0]["energy_rank"] and segs[1]["energy_band"] > segs[0]["energy_band"]
    bands = P.graph_energy_bands()
    assert all(s["energy_label"] == bands[s["energy_band"]]["label"] for s in segs)
    assert segs[0]["cluster"] != segs[1]["cluster"]
    assert "cb_score" in doc["method"]["features_dropped"] and "beat_phase" not in doc["method"]["features"]
    assert "NOT CLASSIFICATIONS" in doc["note"]
    assert nov[0]["weak"] is False and nov[0]["margin"] > 0.4 and nov[0]["peak_t"] is not None
    assert doc["energy_basis"].startswith("rank within this take")
    # homogeneous: no boundary, one segment covering everything
    flat = P.propose(_toy_fs(32, change_at=None), P.ProposalConfig())
    assert flat["boundaries"] == [] and len(flat["segments"]) == 1
    assert max(x for x in flat["novelty"]["strength"] if x is not None) < 0.1
    assert flat["analysis"]["masked"] == [] and flat["rejected_peaks"] == []


def test_propose_masks_dropout_eighths_not_features():
    """Review issue (high): eighths without valid frames used to make every feature 'NaN somewhere', so all features
    were dropped and no boundary was proposed, silently. Now the eighths are masked, the features kept, the hole is
    a 'gap' boundary with its reason, and the warning says so."""
    fs = _toy_fs(48, change_at=32, invalid=(9, 10, 11))
    assert np.isnan(fs.pooled[9:12, :, 0]).all()                         # the dropout eighths are all-NaN
    doc = P.propose(fs, P.ProposalConfig())
    assert P.validate_proposals(doc) == []
    assert doc["method"]["n_features"] == sum(f.novelty for f in F.FEATURES) - 6       # only cb_* dropped
    assert doc["analysis"]["analysed_eighths"] == 45
    assert [m["eighths"] for m in doc["analysis"]["masked"]] == [[9, 12]]
    assert doc["analysis"]["masked"][0]["reason"] == "tracking dropout"
    kinds = [(b["kind"], b["eighth"]) for b in doc["boundaries"]]
    assert kinds == [("gap", 12), ("novelty", 32)]
    gap = doc["boundaries"][0]
    assert gap["gap"] == [pytest.approx(3.6), pytest.approx(4.8)] and "dropout" in gap["reason"]
    assert any("masked" in w and "tracking dropout" in w for w in doc["warnings"])
    segs = doc["segments"]
    assert [(s["start"], s["end"]) for s in segs] == [("edge", "gap"), ("gap", "novelty"), ("novelty", "edge")]
    assert segs[0]["t1"] == pytest.approx(3.6) and segs[1]["t0"] == pytest.approx(4.8)   # the hole is not covered
    # a novelty peak next to the gap is not snapped into a stub: it is listed as rejected ('edge')
    fs = _toy_fs(48, change_at=13, invalid=(9, 10, 11))
    doc = P.propose(fs, P.ProposalConfig())
    assert [b["kind"] for b in doc["boundaries"]] == ["gap"]
    assert [r["reason"] for r in doc["rejected_peaks"]] == ["edge"] and doc["rejected_peaks"][0]["strength"] > 0.5
    # a feature NaN on a few analysed eighths is kept (NaN-aware similarity), not dropped
    fs = _toy_fs(48, change_at=32)
    fs.pooled_z[5:8, F.INDEX["couple_speed"], :] = np.nan
    doc = P.propose(fs, P.ProposalConfig())
    assert "couple_speed" in doc["method"]["features"] and [b["eighth"] for b in doc["boundaries"]] == [32]


def test_pick_boundaries_reports_rejected_peaks():
    N = 32
    slots = np.arange(N) % 8
    s = np.zeros(N + 1)
    s[0] = s[N] = np.nan
    s[[1, 16, 19, 31]] = [0.9, 0.5, 0.8, 0.95]
    rej = []
    out = P.pick_boundaries(s, slots, P.ProposalConfig(threshold=0.3, min_eighths=8), rejected=rej)
    assert [b["eighth"] for b in out] == [20]
    assert sorted((r["peak_eighth"], r["reason"]) for r in rej) == [(1, "edge"), (16, "spacing"),
                                                                    (31, "no snap target")]
    # a gap (blocks) is an edge too: a peak 1 eighth after the gap cannot be proposed
    blocks = np.where(np.arange(N) < 12, 0, 1)
    rej = []
    s2 = np.zeros(N + 1)
    s2[[0, N]] = np.nan
    s2[12] = np.nan
    s2[13] = 0.9
    assert P.pick_boundaries(s2, slots, P.ProposalConfig(threshold=0.3), blocks, rej) == []
    assert rej == [{"peak_eighth": 13, "strength": 0.9, "reason": "edge"}]


@needs_syn
@pytest.mark.skipif(not (SYN_CB / "smplx" / "lead.npz").exists(), reason="work/synthetic_cb not generated")
def test_proposals_on_synthetic_truth():
    """Tuning check (joints-mode features): the homogeneous synthetic gives no boundary; synthetic_cb's two
    transitions (ordinary -> counterbalance pivot -> ordinary) are found within half a measure, no false positive."""
    out = {}
    for take, work in (("synthetic", SYN), ("synthetic_cb", SYN_CB)):
        npzs = F.load_npzs(work / "smplx")
        cbp = work / "counterbalance" / "counterbalance.json"
        fs = F.compute_features(npzs, F.FeatureConfig(com="joints"), F.load_grid(work),
                                json.loads(cbp.read_text()) if cbp.exists() else None, take=take)
        doc = P.propose(fs, P.ProposalConfig())
        ev = P.evaluate(doc, P.truth_transitions(work), 4 * float(np.median(np.diff(fs.grid_t))))
        out[take] = (doc, ev)
        assert P.validate_proposals(doc) == []
    doc, ev = out["synthetic"]
    assert doc["boundaries"] == [] and ev["false_positives"] == []
    doc, ev = out["synthetic_cb"]
    assert len(ev["hits"]) == 2 and ev["false_positives"] == [] and ev["recall"] == 1.0
    segs = doc["segments"]
    assert len(segs) == 3 and segs[1]["energy_band"] > segs[0]["energy_band"]
    assert segs[2]["cluster"] == segs[0]["cluster"] and segs[2]["repeat_of"] == segs[0]["id"]
    assert all(b["slot"] in (0, 4) for b in doc["boundaries"])          # half-measure snapping
    assert all(not b["weak"] for b in doc["boundaries"])                 # the true transitions are strong


def _syn_cb():
    npzs = F.load_npzs(SYN_CB / "smplx")
    cbp = SYN_CB / "counterbalance" / "counterbalance.json"
    return npzs, F.load_grid(SYN_CB), (json.loads(cbp.read_text()) if cbp.exists() else None)


@needs_syn
@pytest.mark.skipif(not (SYN_CB / "smplx" / "lead.npz").exists(), reason="work/synthetic_cb not generated")
def test_proposals_survive_a_tracking_dropout():
    """Review issue (high), reproduced: the lead's joints blanked for 1.2 s at t = 4 s used to drop every feature and
    every boundary without a warning. Now: both true transitions are still found, the dropout is a gap."""
    import copy
    npzs, grid, cb = _syn_cb()
    drop = copy.deepcopy(npzs)
    t = np.asarray(drop["lead"]["t"])
    j = np.asarray(drop["lead"]["joints"], float).copy()
    j[(t >= 4.0) & (t < 5.2)] = np.nan
    drop["lead"]["joints"] = j
    fs = F.compute_features(drop, F.FeatureConfig(com="joints"), grid, cb, take="synthetic_cb")
    assert np.isnan(fs.pooled[:, :, 0]).all(1).sum() >= 2                # whole eighths without a valid frame
    doc = P.propose(fs, P.ProposalConfig())
    assert P.validate_proposals(doc) == []
    assert doc["method"]["n_features"] == sum(f.novelty for f in F.FEATURES)   # nothing dropped
    ev = P.evaluate(doc, P.truth_transitions(SYN_CB), 4 * float(np.median(np.diff(fs.grid_t))))
    assert len(ev["hits"]) == 2 and ev["false_positives"] == []
    gaps = [b for b in doc["boundaries"] if b["kind"] == "gap"]
    eighth = float(np.median(np.diff(fs.grid_t)))                        # the gap is eighth-granular
    assert len(gaps) == 1 and gaps[0]["gap"][0] <= 4.0 + eighth and gaps[0]["gap"][1] >= 5.2 - eighth
    assert "tracking dropout" in gaps[0]["reason"]
    assert any("masked" in w for w in doc["warnings"])


@needs_syn
@pytest.mark.skipif(not (SYN_CB / "smplx" / "lead.npz").exists(), reason="work/synthetic_cb not generated")
def test_noise_calibration_weak_vs_strong():
    """The noise calibration case (review): ~1 cm of slow per-joint wobble + jitter on the homogeneous synthetic can
    push a false boundary over the threshold (0.25-0.37 over seeds 0-5); such boundaries must be flagged weak, while
    synthetic_cb's true transitions under the same noise stay strong (~0.73) with no false positive."""
    import copy

    def wobble(npzs, sigma, seed):
        rng = np.random.default_rng(seed)
        nz = copy.deepcopy(npzs)
        for r in nz:
            j = np.asarray(nz[r]["joints"], float)
            drift = np.cumsum(rng.normal(0, sigma / 6, (len(j), 1, 3)), 0)
            nz[r]["joints"] = j + rng.normal(0, sigma, j.shape) + drift - drift.mean(0)
        return nz

    fs = F.compute_features(wobble(F.load_npzs(SYN / "smplx"), 0.01, 0), F.FeatureConfig(com="joints"),
                            F.load_grid(SYN), None, take="synthetic")
    doc = P.propose(fs, P.ProposalConfig())
    nov = [b for b in doc["boundaries"] if b["kind"] == "novelty"]
    assert nov and all(b["weak"] and b["strength"] < P.ProposalConfig().strong for b in nov)
    assert any("weak" in w for w in doc["warnings"])
    npzs, grid, cb = _syn_cb()
    fs = F.compute_features(wobble(npzs, 0.01, 0), F.FeatureConfig(com="joints"), grid, cb, take="synthetic_cb")
    doc = P.propose(fs, P.ProposalConfig())
    ev = P.evaluate(doc, P.truth_transitions(SYN_CB), 4 * float(np.median(np.diff(fs.grid_t))))
    assert len(ev["hits"]) == 2 and ev["false_positives"] == []
    assert all(not b["weak"] and b["margin"] > 0.4 for b in doc["boundaries"] if b["kind"] == "novelty")


@needs_syn
def test_low_fit_quality_frames_are_not_pooled():
    npzs, grid = _window_npzs(0.0, 6.0)
    t = np.asarray(npzs["follow"]["t"])
    q = np.ones(len(t))
    bad = (t >= 2.0) & (t < 3.0)
    q[bad] = 0.05                                                        # interpolated / >= 23 px
    npzs["follow"] = {**npzs["follow"], "quality": q}
    fs = F.compute_features(npzs, F.FeatureConfig(com="joints"), grid, take="synthetic")
    assert fs.meta["low_quality_frames"] == int(bad.sum()) and not fs.valid[bad].any()
    assert any("fit quality" in w for w in fs.meta["warnings"])
    vf = F.eighth_valid_fraction(fs)
    assert (vf < 0.5).sum() >= 2 and vf.max() == 1.0
    ok, reasons = P.eighth_mask(fs, P.ProposalConfig())
    assert {r for r in reasons if r} == {"low fit quality"}
    # 0 disables the quality mask
    fs0 = F.compute_features(npzs, F.FeatureConfig(com="joints", min_quality=0.0), grid, take="synthetic")
    assert fs0.meta["low_quality_frames"] == 0 and fs0.valid.all()


@needs_syn
def test_staleness_detects_changed_inputs(tmp_path):
    """Review issue (low): nothing noticed when timing.json / the fits / counterbalance.json changed after the
    features were computed. features.staleness compares the recorded hashes with the files on disk."""
    import shutil
    work = tmp_path / "synthetic"
    shutil.copytree(SYN / "smplx", work / "smplx")
    shutil.copy(SYN / "beats.json", work / "beats.json")
    out = work / "moves"
    fs, npz_p, sch_p = F.run("synthetic", None, F.FeatureConfig(com="joints"), out, work)
    assert F.staleness("synthetic", work, npz_p) == []
    src = json.loads(sch_p.read_text())["source"]
    assert src["beats_sha256_16"] and src["timing_sha256_16"] is None and src["counterbalance_sha256_16"] is None
    rows = json.loads((work / "beats.json").read_text())
    rows[3][0] += 0.01
    (work / "beats.json").write_text(json.dumps(rows))
    st = F.staleness("synthetic", work, npz_p)
    assert len(st) == 1 and "beats.json" in st[0]
    (work / "timing").mkdir()
    (work / "timing" / "timing.json").write_text(json.dumps({"footsteps": {}}))
    (work / "counterbalance").mkdir()
    (work / "counterbalance" / "counterbalance.json").write_text(json.dumps({"source": {"lead_sha1": "0" * 40}}))
    st = F.staleness("synthetic", work, npz_p)
    assert any("timing/timing.json changed" in m for m in st)
    assert any("counterbalance.json appeared" in m for m in st)
    lead = work / "smplx" / "lead.npz"
    lead.write_bytes(lead.read_bytes() + b"\0")
    assert any("lead.npz changed" in m for m in F.staleness("synthetic", work, npz_p))
    # counterbalance.json computed from other fits (sha1 mismatch) is caught even when the file itself is unchanged
    assert "other fits" in F.counterbalance_fits_mismatch({"source": {"lead_sha1": "0" * 40}}, work / "smplx")


# ----------------------------------------------------------------------------------------------- dataset

def _write_take(root: Path, take: str, fs: F.FeatureSet, segments: list[Segment]) -> None:
    fs.take = take
    F.save(fs, {"smplx_dir": "toy"}, root / take / "moves")
    save_labels(Labels(take=take, segments=segments), root / take / "moves" / "labels.json")


def test_dataset_refuses_placeholder_only(tmp_path):
    fs = _toy_fs(48, change_at=24)
    ph = [Segment(0.0, 9.6, "lateral", provenance="placeholder"), Segment(9.6, 19.2, "soltinho", provenance="placeholder")]
    _write_take(tmp_path, "toy_a", fs, ph)
    with pytest.raises(DatasetError, match="no human or narration labels"):
        build_dataset(["toy_a"], DatasetConfig(), tmp_path / "out", root=tmp_path)
    # auto segments (proposals / classifier output) are not training targets either
    _write_take(tmp_path, "toy_a", fs, [Segment(0.0, 9.6, "lateral", provenance="auto", confidence=0.9)])
    with pytest.raises(DatasetError):
        build_dataset(["toy_a"], DatasetConfig(), tmp_path / "out", root=tmp_path)
    assert not (tmp_path / "out" / "windows.npz").exists()


def _write_raw_labels(root: Path, take: str, fs: F.FeatureSet, doc: dict) -> None:
    fs.take = take
    F.save(fs, {"smplx_dir": "toy"}, root / take / "moves")
    (root / take / "moves" / "labels.json").write_text(json.dumps(doc), encoding="utf-8")


def test_dataset_rejects_unspecified_provenance_and_invalid_labels(tmp_path):
    """Review issue: a segment without a 'provenance' key is not human training data, a placeholder-marked file is
    excluded (or refused when it mixes in human segments), and labels are validated against the graph."""
    fs = _toy_fs(48, change_at=24)
    base = {"version": 1, "take": "toy", "time_base": "reference", "video": None}
    # (C) no provenance key + placeholder marker: never training data -> refused, nothing written
    _write_raw_labels(tmp_path, "toy", fs, {**base, "placeholder": True, "generator": "some older placeholder tool",
                                            "segments": [{"id": "s1", "t0": 0.0, "t1": 9.6, "move": "lateral"},
                                                         {"id": "s2", "t0": 9.6, "t1": 19.2, "move": "soltinho"}]})
    with pytest.raises(DatasetError, match="no human or narration"):
        build_dataset(["toy"], DatasetConfig(window=8), tmp_path / "out", root=tmp_path)
    # no provenance key, no marker: still not training data (Segment would default it to 'human')
    _write_raw_labels(tmp_path, "toy", fs, {**base, "segments": [{"id": "s1", "t0": 0.0, "t1": 9.6,
                                                                  "move": "lateral"}]})
    with pytest.raises(DatasetError, match="unspecified"):
        build_dataset(["toy"], DatasetConfig(window=8), tmp_path / "out", root=tmp_path)
    # a placeholder generator file that also holds human segments: ambiguous -> refused
    _write_raw_labels(tmp_path, "toy", fs, {**base, "placeholder": True,
                                            "generator": {"module": "dancecap.moves.placeholder", "seed": 1},
                                            "segments": [{"id": "s1", "t0": 0.0, "t1": 9.6, "move": "lateral",
                                                          "provenance": "human"}]})
    with pytest.raises(DatasetError, match="placeholder"):
        build_dataset(["toy"], DatasetConfig(window=8), tmp_path / "out", root=tmp_path)
    # an alias instead of the move slug, an unknown move, overlapping couple segments: refused with the reasons
    for segs, why in (([{"t0": 0.0, "t1": 9.6, "move": "corredor", "provenance": "human"}], "alias"),
                      ([{"t0": 0.0, "t1": 9.6, "move": "not-a-move", "provenance": "human"}], "unknown move"),
                      ([{"t0": 0.0, "t1": 9.6, "move": "lateral", "provenance": "human"},
                        {"t0": 8.0, "t1": 12.0, "move": "soltinho", "provenance": "narration"}], "overlap")):
        _write_raw_labels(tmp_path, "toy", fs, {**base, "segments": [{"id": f"s{i}", **s} for i, s in
                                                                     enumerate(segs)]})
        with pytest.raises(DatasetError, match=why):
            build_dataset(["toy"], DatasetConfig(window=8), tmp_path / "out", root=tmp_path)
    assert not (tmp_path / "out" / "windows.npz").exists()
    # explicit provenance -> accepted
    _write_raw_labels(tmp_path, "toy", fs, {**base, "segments": [{"id": "s1", "t0": 0.0, "t1": 9.6, "move": "lateral",
                                                                  "provenance": "human"},
                                                                 {"id": "s2", "t0": 9.6, "t1": 12.0,
                                                                  "move": "soltinho"}]})
    man = build_dataset(["toy"], DatasetConfig(window=8), tmp_path / "out", root=tmp_path)
    assert man["takes"][0]["segments_by_provenance"] == {"human": 1, "unspecified": 1}
    assert man["takes"][0]["excluded"] == {"unspecified": 1} and [c["slug"] for c in man["classes"]] == ["lateral"]


def test_dataset_windows_split_by_dance(tmp_path):
    fs = _toy_fs(48, change_at=24)
    # take a: human + narration labels, plus a placeholder segment that must be ignored
    _write_take(tmp_path, "toy_a", fs, [Segment(0.0, 9.6, "lateral", provenance="human"),
                                        Segment(9.6, 16.0, "soltinho", provenance="narration", confidence=0.8),
                                        Segment(16.0, 19.2, "viradinha", provenance="placeholder")])
    # take b: only placeholder -> contributes no windows, listed with its exclusions
    _write_take(tmp_path, "toy_b", _toy_fs(48, change_at=None),
                [Segment(0.0, 19.2, "lateral", provenance="placeholder")])
    cfg = DatasetConfig(window=8, hop=4)
    man = build_dataset(None, cfg, tmp_path / "out", root=tmp_path)
    w = load_windows(tmp_path / "out" / "windows.npz")
    assert list(w["classes"]) == ["lateral", "soltinho"]                 # placeholder 'viradinha' never a class
    assert w["X"].shape[1:] == (8, len(X_FEATURES), 4) and len(w["y"]) == man["n_windows"] > 0
    # beat bookkeeping (eighth_pos = position in the dance) is not classifier input; it is kept apart
    assert "eighth_pos" not in list(w["feature_names"]) and list(w["feature_names"]) == X_FEATURES
    assert list(w["beat_feature_names"]) == ["eighth_pos", "beat_phase", "measure_phase"]
    assert w["X_beat"].shape == (len(w["y"]), 8, 3, 4)
    assert man["inputs"]["excluded_from_X"] == list(w["beat_feature_names"])
    # X_raw = the same windows in feature units; X = its per-take robust z
    assert w["X_raw"].shape == w["X"].shape
    xi = [F.INDEX[n] for n in X_FEATURES]
    z = (w["X_raw"] - fs.z_center[xi][None, None]) / fs.z_scale[xi][None, None]
    fin = np.isfinite(w["X"])
    assert fin.any() and np.allclose(z[fin], w["X"][fin], atol=1e-3)
    k = int(np.argmax(w["start_eighth"]))
    assert np.allclose(w["X_raw"][k, :, :, :][np.isfinite(w["X_raw"][k])],
                       fs.pooled[w["start_eighth"][k]:w["start_eighth"][k] + 8][:, xi][np.isfinite(w["X_raw"][k])])
    assert set(w["takes"]) == {"toy_a"} and set(w["split"]) == {"loto"} and set(w["fold"]) == {0}
    starts = w["start_eighth"]
    assert np.all(starts % 4 == 0)                                       # half-measure aligned (phase 0)
    assert np.all(w["labelled"] >= cfg.min_labelled) and np.all((w["purity"] > 0) & (w["purity"] <= 1))
    # windows fully inside [0, 9.6) are 'lateral', fully inside [9.6, 16) 'soltinho'
    lat = list(w["classes"]).index("lateral")
    assert all(y == lat for y, t1 in zip(w["y"], w["t1"]) if t1 <= 9.6 + 1e-6)
    tb = {t["take"]: t for t in man["takes"]}
    assert tb["toy_a"]["excluded"] == {"placeholder": 1} and tb["toy_a"]["training_segments"] == 2
    assert tb["toy_b"]["training_segments"] == 0 and tb["toy_b"]["windows"] == 0
    assert man["readiness"]["stages"]["B1"]["met"] is False
    assert (tmp_path / "out" / man["files"]["eighths"]).exists()
    m2 = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m2["split"]["takes"] == {"toy_a": "loto"} and m2["n_windows"] == man["n_windows"]


def test_window_starts_are_contiguous_half_measures():
    fs = _toy_fs(40, change_at=None)
    st = window_starts(fs, 16, 4)
    assert st and all(fs.eighths["slot"][i] % 4 == 0 for i in st) and st[-1] + 16 <= 40
    X, st2 = windows_for(fs, 16, 4)
    assert st2 == st and X.shape == (len(st), 16, len(X_FEATURES), 4)
    Xr, _ = windows_for(fs, 16, 4, kind="raw")
    assert Xr.shape == X.shape and np.allclose(Xr[0], fs.pooled[st[0]:st[0] + 16][:, [F.INDEX[n] for n in X_FEATURES]],
                                               equal_nan=True)


def _drop_eighths(fs: F.FeatureSet, drop: list[int]) -> F.FeatureSet:
    keep = np.ones(len(fs.eighths["t0"]), bool)
    keep[drop] = False
    for k in list(fs.eighths):
        fs.eighths[k] = np.asarray(fs.eighths[k])[keep]
    fs.pooled, fs.pooled_z = fs.pooled[keep], fs.pooled_z[keep]
    return fs


def test_window_starts_after_a_gap():
    """A 2-eighth gap (not a multiple of 4) must not shift every later window off the half-measures (review issue);
    the hop phase restarts after the gap; a hop that is not a multiple of 4 is refused."""
    fs = _drop_eighths(_toy_fs(64, change_at=None), [20, 21])            # grid eighths 20, 21 missing
    st = window_starts(fs, 16, 4)
    assert st == [0, 4, 22, 26, 30, 34, 38, 42, 46]                      # array positions (grid 0, 4, 24, ... 48)
    assert all(fs.eighths["slot"][i] % 4 == 0 for i in st)
    assert all(fs.eighths["index"][i + 15] - fs.eighths["index"][i] == 15 for i in st)
    # hop 8: every measure, phase restarting at the first start after the gap
    assert window_starts(fs, 16, 8) == [0, 22, 30, 38, 46]
    with pytest.raises(DatasetError, match="multiple of 4"):
        window_starts(fs, 16, 6)
    with pytest.raises(DatasetError):
        build_dataset(["x"], DatasetConfig(hop=2), Path("unused"), root=Path("unused"))
    # eighths with < 50 % valid frames (a dropout) are gaps for the windows too
    fs2 = _toy_fs(48, change_at=None, invalid=(18,))
    st2 = window_starts(fs2, 8, 4)
    assert st2 and not any(i <= 18 < i + 8 for i in st2)
    assert st2 == [0, 4, 8, 20, 24, 28, 32, 36, 40]


# ----------------------------------------------------------------------------------------------- model interface

class FixedHalves(MoveClassifier):
    """NOT a model: fixed probabilities (first half of the take -> class 0, second half -> class 1) to exercise the
    propose_labels plumbing without training anything."""
    name, version = "fixed-halves", "test"

    def __init__(self, n_eighths: int, **kw):
        super().__init__(["lateral", "soltinho"], **kw)
        self.n = n_eighths

    def fit(self, windows):
        raise AssertionError("never called")

    def predict_proba(self, X):
        from dancecap.moves.dataset import window_starts as ws
        starts = ws(self._fs, self.config.window, self.config.hop)
        return np.array([[0.9, 0.1] if s + self.config.window / 2 <= self.n / 2 else [0.2, 0.8] for s in starts])


def test_classifier_interface_plumbing():
    with pytest.raises(TypeError):
        MoveClassifier()                                                   # abstract
    fs = _toy_fs(48, change_at=None)
    clf = FixedHalves(48)
    clf._fs = fs
    lab = clf.propose_labels(fs, take="toy")
    assert validate_labels(lab) == []
    segs = lab.tier("couple")
    assert [s.move for s in segs] == ["lateral", "soltinho"]
    assert all(s.provenance == "auto" and 0 <= s.confidence <= 1 and s.annotator == "model:fixed-halves@test"
               for s in segs)
    # VIEWER_SPEC 10: alternatives [{move, p}] (chosen move first), calibrated flag + calibration record id
    for s in segs:
        alts = s.extra["alternatives"]
        assert alts[0]["move"] == s.move and alts[0]["p"] == pytest.approx(s.confidence, abs=1e-4)
        assert len(alts) == 2 and {a["move"] for a in alts} == {"lateral", "soltinho"}
        assert sum(a["p"] for a in alts) == pytest.approx(1.0, abs=1e-3)
        assert s.extra["calibrated"] is False and s.extra["calibration"] is None
        assert s.to_dict()["alternatives"] == alts                       # written to labels.json
    assert lab.extra["classifier"]["calibrated"] is False
    with pytest.raises(ValueError, match="calibration"):
        clf.propose_labels(fs, take="toy", require_calibrated=True)
    cal = FixedHalves(48, calibration={"id": "cal-test-1", "method": "temperature", "fitted_on": "loto folds"})
    cal._fs = fs
    lab_c = cal.propose_labels(fs, take="toy", require_calibrated=True)
    assert all(s.extra["calibrated"] is True and s.extra["calibration"] == "cal-test-1" for s in lab_c.segments)
    edge = segs[0].t1
    assert segs[1].t0 == edge
    k = int(np.argmin(np.abs(fs.eighths["t0"] - edge)))
    assert fs.eighths["slot"][k] % 4 == 0                                  # inner edge on a half-measure
    # classifier output never overwrites the user's labels
    human = Labels(take="toy", segments=[Segment(6.0, 10.0, "viradinha", provenance="human")])
    merged = merge_into_gaps(human, lab)
    assert validate_labels(merged) == []
    h = [s for s in merged.segments if s.provenance == "human"]
    assert len(h) == 1 and (h[0].t0, h[0].t1, h[0].move) == (6.0, 10.0, "viradinha")
    for s in merged.segments:
        if s.provenance == "auto":
            assert s.t1 <= 6.0 + 1e-9 or s.t0 >= 10.0 - 1e-9


def test_votes_and_low_confidence_gaps():
    fs = _toy_fs(16, change_at=None)
    V = window_votes(16, [0, 4, 8], np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]), 8)
    assert V.shape == (16, 2) and np.allclose(V.sum(1), 1.0)
    lab = segments_from_votes(fs, np.tile([0.4, 0.35], (16, 1)), ["a", "b"], "toy", "model:x@0",
                              min_confidence=0.5)
    assert lab.segments == []                                              # below min_confidence -> a gap
    r = readiness(per_take=[{"training_segments": 3, "segment_moves": {"lateral": 2, "soltinho": 1}}])
    assert r["stages"]["B1"]["met"] is False and r["instances"] == {"lateral": 2, "soltinho": 1}
    # B1 needs 10 dances AND 5 moves with >= 5 instances (one well-covered move is not enough)
    one = [{"training_segments": 1, "segment_moves": {"lateral": 1}} for _ in range(10)]
    r = readiness(per_take=one)
    assert r["stages"]["B1"]["met"] is False and r["stages"]["B1"]["have"]["moves_with_min_instances"] == 1
    five = [{"training_segments": 5, "segment_moves": {m: 1 for m in ("lateral", "soltinho", "viradinha", "dip",
                                                                      "pop")}} for _ in range(10)]
    r = readiness(per_take=five + [{"training_segments": 1, "segment_moves": {"bonus": 1}}])
    assert r["stages"]["B1"]["met"] is True and r["stages"]["B1"]["have"]["moves_below_min"] == ["bonus"]


def test_cli_registration_and_model_refusal(capsys):
    from dancecap.moves.__main__ import build_parser, main
    ap = build_parser()
    sub = next(a for a in ap._actions if a.dest == "command")
    for name in ("features", "proposals", "dataset", "model"):
        assert name in sub.choices, name
    assert main(["model", "fit"]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert main(["model", "plan", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [s["id"] for s in plan["plan"]["stages"]] == ["A", "B1", "B2", "C"]
