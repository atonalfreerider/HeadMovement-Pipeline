"""Unit tests for dancecap.recap_avatars (the pure parts: face-frame choice, stage table, the identity job spec; no footage, no pod).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_recap_avatars.py -q      (from the workspace root)
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import recap_avatars as ra  # noqa: E402


def kp_frames(F=40, iod=30.0, nose_dx=0.0, seen=0.9):
    """(F, 17, 3) keypoints of a frontal face of eye distance `iod` px centred at (500, 400)."""
    kp = np.zeros((F, 17, 3))
    kp[:, 0] = [500 + nose_dx, 420, seen]       # nose
    kp[:, 1] = [500 - iod / 2, 400, seen]       # left eye
    kp[:, 2] = [500 + iod / 2, 400, seen]       # right eye
    return kp


def test_face_scores_rejects_profiles_unseen_and_merged_frames():
    kp = kp_frames()
    kp[3, 0, 0] = 500 + 20.0                    # nose far off the eye midpoint: a profile
    kp[4, 1, 2] = 0.1                           # an eye not seen
    conf = np.full(40, 0.9)
    conf[5] = 0.2                               # the person track doubts the identity
    merged = np.zeros(40, bool)
    merged[6] = True                            # two bodies in one detection
    iod, ok = ra.face_scores(kp, conf, merged)
    assert not ok[[3, 4, 5, 6]].any() and ok[[0, 1, 2, 7]].all()
    assert iod[0] == 30.0 and (iod[[3, 4, 5, 6]] == 0).all()


def test_pick_frames_largest_first_with_time_gap():
    t = np.arange(100) / 30.0
    iod = np.zeros(100)
    iod[[10, 12, 60, 90]] = [40, 50, 45, 20]
    got = ra.pick_frames(iod, t, n=3, min_gap_s=1.0, iod_min=25.0)
    assert got == [12, 60]                      # 10 is within 1 s of 12, 90 is under iod_min


def test_face_box_is_a_square_around_the_eyes_and_nose():
    kp = kp_frames(1, iod=30.0)[0]
    cx, cy, side = ra.face_box(kp)
    assert abs(cx - 500) < 1e-9 and abs(cy - (400 + 400 + 420) / 3) < 1e-9 and abs(side - 5.5 * 30) < 1e-9


def rec(name, yaw=0.0, occ=44.0, sharp=200.0, ok=True):
    return {"name": name, "ok": ok, "yaw": yaw, "occ_px": occ, "sharpness": sharp}


def test_select_faces_filters_excludes_and_spreads_in_time():
    cand = {n: {"cam": c, "t": t, "role": "follow"} for n, c, t in
            [("a", "01", 10.0), ("b", "01", 11.0), ("c", "01", 40.0), ("d", "06", 41.0), ("e", "01", 70.0), ("f", "01", 90.0), ("g", "01", 120.0)]}
    recs = [rec("a", sharp=500), rec("b", sharp=400), rec("c", yaw=60), rec("d", sharp=900), rec("e", ok=False), rec("f", occ=10), rec("g", sharp=100)]
    got = ra.select_faces(recs, cand, n=5, cams=("01",), exclude=("g",))
    assert [r["name"] for r in got] == ["a"]    # b is 1 s from a, c is turned away, d is the wrong camera, e failed, f is small, g excluded
    got = ra.select_faces(recs, cand, n=5)
    assert [r["name"] for r in got] == ["d", "a", "g"]   # all cameras, best sharpness x size first, one per 3 s per camera


def test_stage_table_is_consistent():
    names = [s.name for s in ra.STAGES]
    assert len(names) == len(set(names))
    for s in ra.STAGES:
        assert s.where in ("local", "pod")
        assert s.reads and s.writes and s.cost and s.what
    free = [s.name for s in ra.STAGES if not s.poses]
    assert free[:3] == ["face-frames", "face-landmarks", "face-identity"]
    assert all(s.poses for s in ra.STAGES if s.name.startswith(("texture", "hair-groom", "face-texture", "export")))
    assert not any(s.poses for s in ra.STAGES if s.name in ("shoes-seed", "hair-portrait", "body-shape", "head-review"))


def test_every_stage_command_is_a_cli_subcommand():
    parser_cmds = set(re.findall(r'sub\.add_parser\("([a-z-]+)"', Path(ra.__file__).read_text(encoding="utf-8")))
    for s in ra.STAGES:
        if s.cmd.startswith("("):
            continue                            # the export is another agent's command
        if not s.poses:
            assert s.cmd.split()[0] in parser_cmds, s.name


def test_pod_job_is_a_dedicated_kind_no_other_worker_claims(tmp_path, monkeypatch):
    from dancecap import recap_avatars_pod as pod
    from dancecap.farm import worker as W
    monkeypatch.setattr(pod, "identity_archive", lambda: {"path": "dancecap/work/x/in.tar", "archive": True, "sha256": "0" * 64, "bytes": 1,
                                                          "local": "x.tar", "files": 1})
    spec = pod.identity_spec("ra-identity-test")
    assert spec["kind"] == ra.KIND == "avatar" and spec["needs_gpu"] is False and spec["schema"] == W.SCHEMA
    assert spec["id"].startswith("ra-") and spec["after"] == []
    assert [s["name"] for s in spec["steps"]] == ["links", "identity"]
    # the farm's fit / analysis pods run these kinds: none of them may claim the job
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / "ra-identity-test.json").write_text(json.dumps(spec))
    for kinds in ({"front", "fit", "refine", "full", "asr"}, {"analyse"}):
        w = W.Worker(tmp_path, pod_id="farm", kinds=kinds, ws=tmp_path / "ws", work=tmp_path / "work", state=tmp_path / "st")
        assert not w.claimable(spec), kinds
    mine = W.Worker(tmp_path, pod_id="mine", kinds={"avatar"}, ws=tmp_path / "ws", work=tmp_path / "work", state=tmp_path / "st")
    assert mine.claimable(spec)


def test_job_outputs_are_workspace_relative_under_the_take():
    from dancecap import recap_avatars_pod as pod
    for o in pod.identity_outputs():
        assert o["path"].startswith(f"dancecap/work/{ra.TAKE}/") and ".." not in o["path"]
    assert any(o["path"].endswith("lead_identity_disp.npz") for o in pod.identity_outputs())


def test_pod_commands_name_importable_modules():
    from dancecap import recap_avatars_pod as pod
    for st in pod.identity_steps():
        for mod in re.findall(r"python -m (dancecap[\w.]*)", st["cmd"]):
            assert importlib.util.find_spec(mod) is not None, mod


def test_estimate_is_cheap_and_priced_per_hour():
    from dancecap import recap_avatars_pod as pod
    e = pod.estimate()
    assert 0 < e["usd"] < 0.5 and abs(e["usd"] - e["total_min"] / 60 * ra.PRICE_CPU_8) < 1e-3


# ------------------------------------------------------------------------------------------------ texture-select (pure parts)

def test_heading_and_sectors():
    from dancecap import recap_avatars_select as sel
    rv = np.array([[0, 0, 0], [0, np.pi / 2, 0], [0, np.pi, 0]], float)      # facing +z, +x, -z
    f = sel.rotvec_forward(rv)
    assert np.allclose(f, [[0, 0, 1], [1, 0, 0], [0, 0, -1]], atol=1e-9)
    az = sel.heading_azimuth(rv)
    assert np.allclose(az, [0, np.pi / 2, np.pi]) or np.allclose(np.abs(az[2]), np.pi)
    # a phone straight in front is sector 0, straight behind is sector n / 2, to her left / right the neighbours of the quarter
    rel = np.array([0.0, np.pi, np.pi / 2, -np.pi / 2, 0.2])
    assert list(sel.sector_of(rel)) == [0, 4, 2, 6, 0]


def synthetic_poses(T=600, hz=30.0):
    """She turns slowly through 360 degrees in 20 s (every side shown to a phone that stands still in front of the start pose)."""
    t = np.arange(T) / hz
    yaw = 2 * np.pi * t / 20.0
    return t, yaw


def test_sector_seconds_and_window_score():
    from dancecap import recap_avatars_select as sel
    t, yaw = synthetic_poses()
    xz = np.zeros((len(t), 2))
    cams = {"01": np.tile([0.0, 1.5, 3.0], (len(t), 1))}              # the phone stands at +z, 3 m away
    seen = {"01": np.ones(len(t), bool)}
    sec = sel.sector_seconds(xz, yaw, cams, seen, 1 / 30.0)
    # one full turn: every sector gets the same sight, 20 s x quality / 8 each
    assert np.allclose(sec.sum(0), 20.0 * sel.QUALITY["01"] / 8, rtol=0.05)
    full = sel.window_score(sec, sec, need_s=1.0)
    assert full["worst_sector"] == 1.0 and full["mean_coverage"] == 1.0
    half = sel.window_score(sec[:len(t) // 2], sec[:len(t) // 2], need_s=1.0)
    assert half["score"] < full["score"] or half["worst_sector"] < 1.0


def test_choose_prefers_the_window_that_shows_every_side():
    from dancecap import recap_avatars_select as sel
    t = np.arange(900) / 30.0                                          # 30 s
    n = sel.N_SECTORS
    a = np.zeros((900, n))
    b = np.zeros((900, n))
    a[:450, 0] = 0.05                                                  # first 15 s: both teachers only ever seen from the front
    b[:450, 0] = 0.05
    for k in range(450, 900):                                          # last 15 s: every sector in turn
        a[k, (k // 10) % n] = 0.05
        b[k, (k // 10) % n] = 0.05
    r = sel.choose(a, b, t, length_s=15.0, step_s=5.0, need_s=2.0, max_windows=1)
    assert r["windows"][0]["t0"] >= 15.0 - 1e-9
    r2 = sel.choose(a, b, t, length_s=15.0, step_s=5.0, need_s=2.0, max_windows=2)
    assert len(r2["windows"]) == 2 and r2["combined"]["score"] >= r["combined"]["score"]
    assert r["second_best_adds"] is not None


def test_seen_from_coverage_maps_seconds_to_frames():
    from dancecap import recap_avatars_select as sel
    rows = [["01"], ["01", "06"], []]
    seen = sel.seen_from_coverage(rows, np.array([0.2, 1.5, 2.9, 7.0]), ["01", "06"])
    assert list(seen["01"]) == [True, True, False, False] and list(seen["06"]) == [False, True, False, False]


def test_every_pose_stage_command_is_a_cli_subcommand_too():
    parser_cmds = set(re.findall(r'sub\.add_parser\("([a-z-]+)"', Path(ra.__file__).read_text(encoding="utf-8")))
    for s in ra.STAGES:
        if s.cmd.startswith("("):
            continue
        assert s.cmd.split()[0] in parser_cmds, s.name


def test_window_job_and_slice_smplx(tmp_path):
    from dancecap import recap_avatars_pod as pod
    base = ra.WORK / "farm" / "jobs" / "shape.json"
    if not base.exists():
        import pytest
        pytest.skip("the farm's shape job spec is not on this machine")
    job = pod.window_job("tx9", 120.02, 150.0, ["01", "06"])
    c = job["chunk"]
    assert job["id"] == c["id"] == "tx9" and job["take"] == c["take"] == "larissa_kadu_recap_tx9" and job["cams"] == ["01", "06"] == c["cams"]
    assert c["k0"] == 3601 and c["k1"] == 4499 and c["T"] == 899 and c["core_k0"] == c["k0"]
    # the base spec is not touched
    assert json.loads(base.read_text())["id"] == "shape"
    t = 100 + np.arange(300) / 30.0
    d = tmp_path / "smplx"
    d.mkdir()
    for r in ra.ROLES:
        np.savez(d / f"{r}.npz", t=t, joints=np.zeros((300, 55, 3)), betas=np.arange(16.0), model=np.array("m"))
    out = tmp_path / "out"
    info = pod.slice_smplx(d, out, 102.0, 104.0)
    z = np.load(out / "lead.npz")
    assert info["lead"] == len(z["t"]) and z["joints"].shape[0] == len(z["t"]) and z["betas"].shape == (16,) and z["t"][0] >= 101.99


def test_contact_penalty_prefers_the_window_where_the_couple_is_apart():
    from dancecap import recap_avatars_select as sel
    t = np.arange(900) / 30.0                                          # 30 s, both halves show every side equally
    n = sel.N_SECTORS
    a = np.zeros((900, n))
    for k in range(900):
        a[k, (k // 10) % n] = 0.05
    contact = np.zeros(900, bool)
    contact[:450] = True                                               # embraced for the first 15 s
    plain = sel.choose(a, a.copy(), t, length_s=15.0, step_s=5.0, need_s=0.5)
    pen = sel.choose(a, a.copy(), t, length_s=15.0, step_s=5.0, need_s=0.5, contact=contact)
    assert pen["windows"][0]["t0"] >= 15.0 - 1e-9 and pen["windows"][0]["contact"] == 0.0
    assert all("contact" in w for w in pen["ranking"]) and "contact" not in plain["windows"][0]
