"""sides.evidence / sides.adapters (DESIGN 8): the vectorised triangulation equals hand_contact.triangulate_joint (its fit gates opened),
per-chunk video offsets, no read of the audio-clock 3D files, the identity gate, the covariance shrinking with views, the
visibility-consistent re-triangulation dropping a planted hallucination, the role-swap check, the held-out plan.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_sides_evidence.py -q -p no:cacheprovider     (from the workspace root)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sides_rig as RIG  # noqa: E402

from dancecap import hand_contact as hc
from dancecap.sides import evidence as E
from dancecap.sides import geom
from dancecap.sides.adapters import MultiView
from dancecap.sides.config import SidesCfg

CFG = SidesCfg()
SIDES_DIR = Path(__file__).resolve().parents[1] / "dancecap" / "sides"


def _rig_detections(T=40, V=5, seed=0, outliers=0.15, missing=0.2):
    rng = np.random.default_rng(seed)
    rig = RIG.ring(V)
    X = np.array([0.0, 1.2, 0.0]) + rng.normal(0, 0.25, (T, 3))
    uv = np.zeros((V, T, 1, 2))
    valid = rng.random((V, T, 1)) > missing
    for v, (C, R, t) in enumerate(rig):
        for k in range(T):
            u = RIG.project(R, t, X[k]) + rng.normal(0, 1.5, 2)
            if rng.random() < outliers:
                u = u + rng.normal(0, 80, 2)
            uv[v, k, 0] = u
    return rig, X, uv, valid


class _Obs:
    def __init__(self, rig, T):
        self.cams = [f"{i:02d}" for i in range(len(rig))]
        self.z = {c: {"ok": np.ones(T, bool), "K": np.repeat(RIG.K0[None], T, 0), "R": np.repeat(R[None], T, 0), "t": np.repeat(t[None], T, 0)}
                  for c, (C, R, t) in zip(self.cams, rig)}


class _Judge:
    def __init__(self, cams, uv, valid):
        self.cams, self.uv, self.valid = cams, uv, valid

    def joint(self, c, role, side, k, which="wrist"):
        v = self.cams.index(c)
        return (self.uv[v, k, 0].copy(), True) if self.valid[v, k, 0] else (None, False)


def test_triangulate_all_equals_hand_contact_triangulate_joint():
    T, V = 40, 5
    rig, X, uv, valid = _rig_detections(T, V)
    P, Cc, f = RIG.stack_rig(rig, T)
    got = E.triangulate_all(P, Cc, f, uv, valid, CFG)
    obs = _Obs(rig, T)
    judge = _Judge(obs.cams, uv, valid)
    J = np.zeros((T, 55, 3))
    J[:, 21] = X
    J[:, 17] = X + [0.0, 0.3, 0.0]
    ref = hc.triangulate_joint({"joints": J}, "lead", "right", np.arange(T), judge, obs, obs.cams,
                               hc.TriCfg(max_reach_m=100.0, max_from_fit_m=100.0), "wrist")
    a, b = got["X"][:, 0], ref["X"]
    assert (np.isfinite(a).all(1) == np.isfinite(b).all(1)).all()
    ok = np.isfinite(b).all(1)
    assert ok.sum() > 30
    assert np.abs(a[ok] - b[ok]).max() < 1e-6
    assert (got["nv"][:, 0] == ref["n_views"]).all()
    assert np.allclose(got["err"][ok, 0], ref["err_px"][ok], atol=1e-6)
    three = ok & (ref["n_views"] >= 3)
    assert np.linalg.norm(a[three] - X[three], axis=1).max() < 0.03      # with >= 3 views the outliers were rejected


def test_parallel_workers_give_the_same_result():
    T, V = 30, 4
    rig, X, uv, valid = _rig_detections(T, V, seed=2)
    P, Cc, f = RIG.stack_rig(rig, T)
    a = E.triangulate_all(P, Cc, f, uv, valid, CFG, block=7)
    b = E.triangulate_all(P, Cc, f, uv, valid, CFG, block=7, workers=2)
    assert np.array_equal(np.isnan(a["X"]), np.isnan(b["X"])) and np.nanmax(np.abs(a["X"] - b["X"])) == 0


def test_identity_gate_and_covariance_shrinks_with_views():
    T, V = 10, 5
    rig, X, uv, valid = _rig_detections(T, V, seed=1, outliers=0.0, missing=0.0)
    P, Cc, f = RIG.stack_rig(rig, T)
    far = (X + [0.7, 0, 0])[:, None]                 # the base joint 0.7 m away: a role swap
    g = E.triangulate_all(P, Cc, f, uv, valid, CFG, ref=far)
    assert np.isnan(g["X"]).all() and g["gated"].all()
    near = (X + [0.3, 0, 0])[:, None]                # 0.3 m: kept (the 0.30 m TriCfg gate is not used)
    assert np.isfinite(E.triangulate_all(P, Cc, f, uv, valid, CFG, ref=near)["X"]).all()
    two = valid.copy()
    two[2:] = False
    from dataclasses import replace
    c0 = replace(CFG, sigma_floor_m=1e-4)
    S2 = E.triangulate_all(P, Cc, f, uv, two, c0)["S"][:, 0]
    S5 = E.triangulate_all(P, Cc, f, uv, valid, c0)["S"][:, 0]
    tr2, tr5 = np.trace(S2, axis1=1, axis2=2), np.trace(S5, axis1=1, axis2=2)
    assert (tr5 < tr2).all()
    S5f = E.triangulate_all(P, Cc, f, uv, valid, CFG)["S"][:, 0]
    assert (np.linalg.eigvalsh(S5f)[:, 0] >= CFG.sigma_floor_m ** 2 - 1e-12).all()


def test_visibility_consistent_retriangulation_drops_a_planted_hallucination():
    T, V = 6, 5
    rig = RIG.ring(V)
    P, Cc, f = RIG.stack_rig(rig, T)
    X = np.tile(np.array([0.0, 1.2, 0.0]), (T, 1))
    W = np.array([0.25, 1.0, 0.1])                   # where three views hallucinate the hidden joint (consistently)
    uv = np.zeros((V, T, 1, 2))
    for v, (C, R, t) in enumerate(rig):
        for k in range(T):
            uv[v, k, 0] = RIG.project(R, t, X[k] if v < 2 else W)
    valid = np.ones((V, T, 1), bool)
    allv = E.triangulate_all(P, Cc, f, uv, valid, CFG)
    assert np.linalg.norm(allv["X"][0, 0] - W) < 0.01          # RANSAC alone takes the 3-view hallucination
    # an occluder in front of views 2-4 hides the true joint from them
    obs = {f"{v}": {"C": Cc[v], "R": np.repeat(rig[v][1][None], T, 0), "t": np.repeat(rig[v][2][None], T, 0), "ok": np.ones(T, bool)}
           for v in range(V)}
    OA, OB, r = [], [], []
    for v in range(2, V):
        C = rig[v][0]
        m = X[0] + 0.3 * (C - X[0]) / np.linalg.norm(C - X[0])
        OA.append(m - [0, 0.1, 0])
        OB.append(m + [0, 0.1, 0])
        r.append(0.08)
    OA = np.repeat(np.array(OA)[None], T, 0)
    OB = np.repeat(np.array(OB)[None], T, 0)
    vis = E.visible(obs, list(obs), X[:, None], OA, OB, np.array(r), None, 0.035)
    assert vis[:, 0, :2].all() and not vis[:, 0, 2:].any()
    cons = E.triangulate_all(P, Cc, f, uv, valid & np.transpose(vis, (2, 0, 1)), CFG)
    assert np.linalg.norm(cons["X"][:, 0] - X, axis=1).max() < 1e-3
    assert (cons["nv"][:, 0] == 2).all()


def test_per_chunk_offsets_and_role_swap_check(tmp_path):
    work = RIG.write_take(tmp_path, "tk", offsets=(0.1, -0.05))
    ad = MultiView("tk", CFG, work)
    co = ad.coco_tracks()
    T = ad.T
    owner = ad.owner
    XL = ad.base_coco("lead")
    o = ad.obs()
    for c in ad.cams:
        for k in (5, T // 2 + 5):                     # one frame of each chunk
            want = RIG.project(o[c]["R"][k], o[c]["t"][k], XL[k, 5])
            got = co[c]["uv"][k, 0, 5]
            assert np.linalg.norm(got - want) < 0.5, (c, k, owner[k])
    # the same with one chunk's offsets used for both (the trap): pixels off by a frame of motion
    import json
    cb = tmp_path / "tk_cb" / "sync" / "offsets.json"
    cb.write_text(json.dumps({"offsets": {c: {"delta": 0.1} for c in ad.cams}}))
    co2 = MultiView("tk", CFG, work).coco_tracks()
    k = T // 2 + 5
    c = ad.cams[0]
    want = RIG.project(o[c]["R"][k], o[c]["t"][k], XL[k, 5])
    assert not np.isfinite(co2[c]["uv"][k, 0, 5]).all() or np.linalg.norm(co2[c]["uv"][k, 0, 5] - want) > 3.0
    # role check: as stored passes, swapped teachers fail loudly
    rc = MultiView("tk", CFG, work).role_swap_check()
    assert all(v["ok"] for v in rc.values())
    work2 = RIG.write_take(tmp_path, "sw", swap=True)
    with pytest.raises(RuntimeError):
        MultiView("sw", CFG, work2).role_swap_check()


def test_heldout_plan():
    t = np.arange(0, 40.0, 1 / 30)
    seen = np.ones((len(t), 6), bool)
    seen[t >= 30, 3:] = False                       # the last block: only 3 views see the dancer
    ho = E.heldout_plan(t, seen, CFG)
    for b in range(3):
        m = (t >= 10 * b) & (t < 10 * (b + 1))
        assert (ho[m] == b % 6).all()
    assert (ho[t >= 30] == -1).all()


def test_no_audio_clock_3d_files_are_read():
    for p in SIDES_DIR.glob("*.py"):
        src = p.read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", src, flags=re.S)        # docstrings may name the trap
        code = "\n".join(line.split("#")[0] for line in code.splitlines())
        assert "track3d" not in code and "world_people" not in code, p.name


def test_crossing_vote_sign_matches_3d():
    rng = np.random.default_rng(5)
    K = RIG.K0
    agree = n = 0
    for _ in range(300):
        C = rng.normal(0, 1, 3) * 0.5 + [0, 1.5, 4.0]
        R, t = RIG.look_at(C, (0, 1.2, 0))
        A0, B0 = np.array([0, 1.2, 0.0]) + rng.normal(0, 0.03, 3), np.array([0, 1.2, 0.0]) + rng.normal(0, 0.03, 3)
        u, v = rng.normal(size=3), rng.normal(size=3)
        u *= 0.3 / np.linalg.norm(u)
        v *= 0.3 / np.linalg.norm(v)
        a0, a1, b0, b1 = A0 - u / 2, A0 + u / 2, B0 - v / 2, B0 + v / 2
        pa = np.array([[RIG.project(R, t, a0), RIG.project(R, t, a1)]])
        pb = np.array([[RIG.project(R, t, b0), RIG.project(R, t, b1)]])
        hit, _ = E.seg_cross_2d(pa[:, 0], pa[:, 1], pb[:, 0], pb[:, 1])
        if not hit[0]:
            continue
        # who is in front at the crossing pixel: the depth of each segment's point that projects there (dense search)
        def at_pixel(p0, p1, px):
            ss = np.linspace(0, 1, 2001)
            pts = p0[None] + ss[:, None] * (p1 - p0)[None]
            pr = np.array([RIG.project(R, t, q) for q in pts[::20]])
            i = int(np.argmin(np.linalg.norm(pr - px, axis=1))) * 20
            lo, hi = max(0, i - 20), min(2000, i + 20)
            pr2 = np.array([RIG.project(R, t, q) for q in pts[lo:hi + 1]])
            q = pts[lo + int(np.argmin(np.linalg.norm(pr2 - px, axis=1)))]
            return (R @ q + t)[2]
        da, db = pa[0, 1] - pa[0, 0], pb[0, 1] - pb[0, 0]
        cr = lambda x, y: x[0] * y[1] - x[1] * y[0]
        s2 = cr(pb[0, 0] - pa[0, 0], db) / cr(da, db)
        px = pa[0, 0] + s2 * da
        dA, dB = at_pixel(a0, a1, px), at_pixel(b0, b1, px)
        cfA, cfB = (np.array([[0.9, 0.9]]), np.array([[0.5, 0.5]])) if dA < dB else (np.array([[0.5, 0.5]]), np.array([[0.9, 0.9]]))
        vote = E.crossing_votes(pa, cfA, pb, cfB)[0]
        s3 = np.sign(np.cross(a1 - a0, b1 - b0) @ (A0 - B0))
        n += 1
        agree += int(vote == s3)
    assert n > 50 and agree / n > 0.99


def test_debias_keeps_the_heading_of_a_triangulated_line_where_the_base_is_turned():
    T = 4
    J = RIG.body_joints(T, 0.0, 0.0, 1.0)              # the base faces +z
    X = np.full((T, 17, 3), np.nan)
    ang = np.radians(70.0)                               # the real body is turned 70 deg (a spin the base missed)
    left = np.array([np.cos(ang), 0, -np.sin(ang)])
    for j, (side, y) in {11: (1, 0.95), 12: (-1, 0.95), 5: (1, 1.45), 6: (-1, 1.45)}.items():
        X[:, j] = np.array([0, y, 0]) + side * (0.15 if j in (11, 12) else 0.22) * left
    nv = np.full((T, 17), 4, np.int8)
    bias = np.zeros((17, 3))
    bias[11] = [0.05, 0.05, 0.0]                         # COCO hips 5 cm outward and up (left = +x in the dancer frame)
    bias[12] = [-0.05, 0.05, 0.0]
    Xd = E.debias(X, J, bias, nv)
    h_raw = geom.line_heading(X[:, 11], X[:, 12])
    assert np.allclose(geom.line_heading(Xd[:, 11], Xd[:, 12]), h_raw, atol=1e-6)
    assert np.allclose(np.linalg.norm(Xd[:, 11] - Xd[:, 12], axis=1), 0.2, atol=1e-6)
    old = E.debias(X, J, bias)                           # in the base's frame the line turns
    assert np.abs(geom.wrap_deg(geom.line_heading(old[:, 11], old[:, 12]) - h_raw)).min() > 10
