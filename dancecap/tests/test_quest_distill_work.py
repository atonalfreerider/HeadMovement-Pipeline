r"""dancecap.mono4d.quest_distill run folders (2026-10-23: carlos_aline's Quest sets): the default job is unchanged, a --work run
folder gets its own bundle / job / pod_out paths (the other lessons' bundle is not uploaded), --roles limits the dancers, and
the lesson table lists carlos_aline. Synthetic splat / motion files only.

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_quest_distill_work.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.mono4d import quest_distill as QD  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402


def test_lessons():
    assert QD.LESSONS["carlos_aline"] == "07_CarlosAline"
    assert QD.LESSONS["recap"] == "06_LarissaKaduRecap" and QD.LESSONS["john_lula"] == "08_JohnLula"


def test_default_job_is_unchanged():
    j = QD.job_spec(None, 4000)
    assert j["name"] == "quest_distill" and j["inputs"] == ["dancecap/work/quest_distill/bundle/"]
    assert j["commands"][0]["cmd"] == ("python -m dancecap.mono4d.quest_distill distill --bundle work/quest_distill/bundle "
                                       "--out work/quest_distill/pod_out --iters 4000 --time-budget-min 40")
    assert j["after_download"] == ["python -m dancecap.mono4d.quest_distill import"]
    saved = PACKAGE_ROOT / "work" / "quest_distill" / "pod_job.json"
    if saved.exists() and json.loads(saved.read_text()).get("commands", [{}])[0].get("cmd", "").endswith("--iters 4000 --time-budget-min 40"):
        assert json.dumps(j, indent=1) == saved.read_text()


def test_work_folder_job():
    j = QD.job_spec("work/quest_distill_ca", 4000)
    assert j["name"] == "quest_distill_quest_distill_ca"
    assert j["inputs"] == ["dancecap/work/quest_distill_ca/bundle/"]
    assert j["outputs"] == ["dancecap/work/quest_distill_ca/pod_out/"]
    assert "--bundle work/quest_distill_ca/bundle --out work/quest_distill_ca/pod_out" in j["commands"][0]["cmd"]
    assert j["after_download"] == ["python -m dancecap.mono4d.quest_distill import --work work/quest_distill_ca"]


def _motion(path: Path, T: int = 50) -> None:
    h = {"frames": T, "frame_floats": 3 + 55 * 4 + 55 * 3}
    fr = np.zeros((T, h["frame_floats"]), np.float32)
    fr[:, 3:3 + 220].reshape(T, 55, 4)[..., 3] = 1.0
    hb = json.dumps(h).encode()
    path.write_bytes(b"DCXM" + struct.pack("<I", len(hb)) + hb + fr.tobytes())


@pytest.fixture
def fake_sa(tmp_path, monkeypatch):
    sa = tmp_path / "SA"
    sp = sa / "07_CarlosAline" / "splats"
    sp.mkdir(parents=True)
    for role in ("lead", "follow"):
        rec = np.zeros(4, QD.RECORD)
        rec["rotation"][:, 3] = 1.0
        QD.write_splatav(sp / f"{role}.splatav", rec)
        (sp / f"{role}.json").write_text(json.dumps({"role": role}))
        _motion(sa / "07_CarlosAline" / f"smplx_{role}.bin")
    monkeypatch.setattr(QD, "SA", sa)
    work = PACKAGE_ROOT / "work" / "_pytest_quest_distill_work"
    yield work
    shutil.rmtree(work, ignore_errors=True)


def test_prepare_into_its_own_folder_with_roles(fake_sa):
    p = QD.prepare(["carlos_aline"], [40000, 15000], n_poses=8, iters=100, work=fake_sa, roles=("follow",))
    assert p == fake_sa / "pod_job.json"
    man = json.loads((fake_sa / "bundle" / "manifest.json").read_text())
    assert [(d["capture"], d["role"]) for d in man["dancers"]] == [("07_CarlosAline", "follow")]
    assert (fake_sa / "bundle" / "07_CarlosAline" / "follow.splatav").exists()
    assert not (fake_sa / "bundle" / "07_CarlosAline" / "lead.splatav").exists()
    job = json.loads(p.read_text())
    assert job["inputs"] == ["dancecap/work/_pytest_quest_distill_work/bundle/"]
    assert "--iters 100" in job["commands"][0]["cmd"]


# ---------------------------------------------------------------- fix round 1 (2026-10-23): verifier F6
def _set_records(n, hair_n, bound_n):
    rec = np.zeros(n, QD.RECORD)
    rec["rotation"] = [0, 0, 0, 1]
    rec["strand"], rec["segment"] = -1, -1
    rec["hair"][:hair_n] = 1
    rec["strand"][:bound_n] = 3
    rec["segment"][:bound_n] = 2
    return rec


def _teacher_header():
    return {"role": "follow", "count": 160000, "hair_count": 21518,
            "hair": {"binding": "unity_hair_strands", "hair_count": 21518, "bound": 11968,
                     "binding_info": {"guides": 160, "bound": 11968}}}


def test_recount_hair_takes_the_sets_own_counts():
    h = QD.recount_hair(_teacher_header(), _set_records(100, 40, 7))
    assert h["hair_count"] == 40 and h["hair"]["hair_count"] == 40
    assert h["hair"]["bound"] == 7 and h["hair"]["binding_info"]["bound"] == 7
    assert h["hair"]["teacher_bound"] == 11968 and h["hair"]["teacher_hair_count"] == 21518
    h2 = QD.recount_hair(json.loads(json.dumps(h)), _set_records(100, 40, 7))
    assert h2 == h                                           # idempotent: the teacher's numbers are not overwritten
    lead = QD.recount_hair({"role": "lead", "hair_count": 0, "hair": {"bound": 0}}, _set_records(10, 0, 0))
    assert lead["hair_count"] == 0 and lead["hair"]["bound"] == 0 and "binding_info" not in lead["hair"]


def test_import_rewrites_the_copied_set_headers(tmp_path, monkeypatch):
    """import_run on a pod_out whose json still carries the teacher's hair.bound (sets made before this fix)."""
    src = tmp_path / "pod_out"
    (src / "sets" / "quest_15k").mkdir(parents=True)
    rec = _set_records(50, 20, 9)
    QD.write_splatav(src / "sets" / "quest_15k" / "follow.splatav", rec)
    h = _teacher_header()
    h["data_sha256"] = QD.sha256(src / "sets" / "quest_15k" / "follow.splatav")
    (src / "sets" / "quest_15k" / "follow.json").write_text(json.dumps(h))
    (src / "distill_report.json").write_text(json.dumps({"dancers": [
        {"capture": "07_X", "role": "follow", "sizes": {"15000": {"files": "sets/quest_15k"}}, "eval": {}}]}))
    monkeypatch.setattr(QD, "SA", tmp_path / "SA")
    (tmp_path / "work").mkdir()
    QD.import_run(None, src, tmp_path / "work")
    out = json.loads((tmp_path / "SA" / "07_X" / "splats" / "quest_15k" / "follow.json").read_text())
    assert out["hair"]["bound"] == 9 and out["hair"]["binding_info"]["bound"] == 9 and out["hair_count"] == 20
    assert out["data_sha256"] == h["data_sha256"]           # the records are untouched


# ---------------------------------------------------------------- fix round 2: the teacher's strand look on the students
def _look_header(look=True):
    from dancecap.mono4d import recap_splat_hair as RH
    G, P = 3, 6
    rest = np.zeros((G, P, 3))
    for g in range(G):
        for i in range(P):
            rest[g, i] = [0.03 * (g - 1), -0.04 * i, 0.04 + 0.01 * i]
    h = {"role": "follow", "rest_joints": np.zeros((55, 3)).tolist(),
         "hair": {"binding": "unity_hair_strands",
                  "binding_info": {"guides": G, "points": P, "rest_points": rest.reshape(-1, 3).tolist()}}}
    if look:
        h["hair"]["look"] = {"params": dict(RH.HAIR_LOOK_BLACK)}
    return h, rest


def _bound_records(rest, n=40, seed=2):
    rng = np.random.default_rng(seed)
    rec = np.zeros(n, QD.RECORD)
    G, P = rest.shape[:2]
    s_, k_ = rng.integers(0, G, n), rng.integers(0, P - 1, n)
    rec["mean"] = rest[s_, k_] + rng.uniform(-0.004, 0.004, (n, 3))
    q = rng.normal(size=(n, 4))
    rec["rotation"] = q / np.linalg.norm(q, axis=1, keepdims=True)
    rec["log_scale"] = np.log(rng.uniform(0.008, 0.012, (n, 3)))        # the round-1 students: 8-12 mm, opaque
    rec["opacity_logit"] = 4.0                                          # 0.98
    rec["strand"], rec["segment"] = s_, k_
    rec["strand"][30:], rec["segment"][30:] = -1, -1                    # 30..39 unbound body
    rec["local_rotation"] = [0, 0, 0, 1]
    return rec


def test_bound_look_makes_the_distilled_bound_hair_strand_shaped():
    from dancecap.mono4d import recap_splat_hair as RH
    h, rest = _look_header()
    rec = _bound_records(rest)
    before = rec.copy()
    rec, info = QD.bound_look(rec, h)
    assert info["applied"] and info["bound"] == 30
    assert info["large_opaque_bound"]["before"] == 30 and info["large_opaque_bound"]["after"] == 0
    b = rec["strand"] >= 0
    op = 1 / (1 + np.exp(-rec["opacity_logit"].astype(float)))
    assert np.all(op[b] <= 0.25 + 1e-5)
    S = np.exp(rec["log_scale"].astype(float))
    assert np.all(np.sort(S[b], 1)[:, :2] <= 0.003 + 1e-6)
    F = RH.segment_frames(rest)[rec["strand"][b], rec["segment"][b]]
    qb = RH.q_mul(RH.q_from_mat(F), rec["local_rotation"][b].astype(float))
    assert np.allclose(np.abs((qb * rec["rotation"][b].astype(float)).sum(1)), 1, atol=1e-5)   # binding encodes the rotation
    t = rest[rec["strand"][b], rec["segment"][b] + 1] - rest[rec["strand"][b], rec["segment"][b]]
    ax = RH.q_to_mat(rec["rotation"][b].astype(float))[:, :, 0]
    assert np.allclose(np.abs((ax * t / np.linalg.norm(t, axis=1, keepdims=True)).sum(1)), 1, atol=1e-5)   # along the strand
    for k in ("mean", "strand", "segment", "joints", "weights", "sh0"):
        assert np.array_equal(rec[k], before[k]), k
    assert np.array_equal(rec[~b], before[~b])                              # unbound records untouched
    rec2, info2 = QD.bound_look(rec.copy(), h)                              # idempotent
    assert np.allclose(rec2["log_scale"], rec["log_scale"], atol=1e-6)
    assert np.allclose(1 / (1 + np.exp(-rec2["opacity_logit"].astype(float))), op, atol=1e-5)


def test_bound_look_leaves_teachers_without_a_look_alone():
    h, rest = _look_header(look=False)
    rec = _bound_records(rest)
    before = rec.copy()
    rec, info = QD.bound_look(rec, h)
    assert info == {"applied": False} and np.array_equal(rec, before)
    assert QD.look_params({"role": "lead"}) is None and QD.rest_guides({"role": "lead"}) is None


def test_look_project_clamps_the_students_on_bound_hair_only():
    torch = pytest.importorskip("torch")
    h, rest = _look_header()
    lk = QD.look_params(h)
    n = 6
    log_s = torch.log(torch.tensor([[0.02, 0.01, 0.006]] * n, dtype=torch.float32)).requires_grad_(True)
    logit = torch.full((n,), 4.0).requires_grad_(True)
    q = torch.tensor([[1.0, 0, 0, 0]] * n).requires_grad_(True)
    sb = torch.tensor([True, True, True, False, False, False])
    QD.look_project(lk, sb, log_s, logit, q, np.tile([0.0, -1.0, 0.0], (3, 1)))
    s = log_s.detach().exp().numpy()
    assert np.allclose(np.sort(s[:3], 1), [[0.003, 0.003, 0.02]] * 3, atol=1e-6)    # thin across, along kept (<= 5 cm)
    assert np.allclose(s[3:], [[0.02, 0.01, 0.006]] * 3, atol=1e-7)
    o = torch.sigmoid(logit.detach()).numpy()
    assert np.all(o[:3] <= 0.25 + 1e-6) and np.allclose(o[3:], 1 / (1 + np.exp(-4.0)))
    from dancecap.mono4d import recap_splat_hair as RH
    qx = np.concatenate([q.detach().numpy()[:, 1:], q.detach().numpy()[:, :1]], 1)
    ax = RH.q_to_mat(qx[:3].astype(float))[:, :, 0]
    assert np.allclose(np.abs(ax[:, 1]), 1, atol=1e-5)                               # long axis along the tangent
