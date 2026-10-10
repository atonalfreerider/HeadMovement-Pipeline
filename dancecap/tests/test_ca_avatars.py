"""Pure parts of the carlos_aline avatar graph (dancecap.ca_avatars, dancecap.ca_avatars_pod): face candidates, identity continuity,
colour clustering, provisional texture windows, the pod job's dedicated kind. Synthetic data only (no portraits, no videos, no pods)."""
from __future__ import annotations

import importlib.util
import json
import re

import numpy as np

from dancecap import ca_avatars as ca


def _kp(n, iod=30.0, off=0.0, score=0.9):
    kp = np.zeros((n, 17, 3))
    kp[:, :, 2] = score
    kp[:, 1, 0], kp[:, 2, 0] = 100 - iod / 2, 100 + iod / 2
    kp[:, 1, 1] = kp[:, 2, 1] = 50
    kp[:, 0, 0], kp[:, 0, 1] = 100 + off * iod, 60
    return kp


def test_face_metrics_and_scores_prefer_big_frontal_slow_faces():
    kp = np.concatenate([_kp(1, 30), _kp(1, 20), _kp(1, 40, off=0.8), _kp(1, 30, score=0.1)])
    m = ca.face_metrics(kp)
    assert list(m["seen"]) == [True, True, True, False]
    sp = np.array([0.0, 0.0, 0.0, 0.0])
    sc = ca.face_scores(m, sp)
    assert sc[0] > sc[1] > 0 and sc[2] == 0 and sc[3] == 0          # turned and unseen faces score 0
    assert ca.face_scores(m, np.array([0.5, 0, 0, 0]))[0] == 0      # blurred (fast head)


def test_head_speed_central_difference_and_nan_edges():
    kp = _kp(5)
    kp[:, 1, 0] += np.arange(5) * 6.0
    kp[:, 2, 0] += np.arange(5) * 6.0
    v = ca.head_speed(kp, np.full(5, 30.0))
    assert np.isnan(v[0]) and np.isnan(v[-1]) and np.allclose(v[1:-1], 6.0 / 30.0)


def test_pick_frames_gap_and_order():
    t = np.arange(10, dtype=float)
    sc = np.array([0, 5, 9, 8, 0, 7, 1, 6, 0, 0], float)
    assert ca.pick_frames(sc, t, 3, 2.0) == [2, 5, 7]               # 3 (8) is within 2 s of 2 (9): skipped
    assert ca.pick_frames(sc, t, 2, 0.5) == [2, 3]


def test_appearance_evidence_white_top_is_carlos():
    white, red = np.array([180.0, 175, 170]), np.array([170.0, 30, 35])
    assert ca.appearance_evidence(np.stack([white, red])) > 1       # detection 0 is Carlos
    assert ca.appearance_evidence(np.stack([red, white])) < -1
    assert ca.appearance_evidence(np.full((2, 3), np.nan)) == 0.0


def test_identity_viterbi_follows_continuity_against_weak_noisy_evidence():
    # truth: detection 0 is Carlos for the whole run; frames 8..10 have misleading appearance (his sleeve over her top)
    T = 20
    ev = np.full(T, 1.5)
    ev[8:11] = -3.0
    flip = np.zeros(T - 1)
    marg = np.ones(T - 1)
    cont = np.ones(T - 1, bool)
    s = ca.identity_viterbi(ev, flip, marg, cont)
    assert (s == 0).all()
    # a real swap in the ordering (the sizes crossed): matching says the persons exchange slots at i = 9
    flip2 = flip.copy()
    flip2[9] = 1
    ev2 = np.where(np.arange(T) <= 9, 1.5, -1.5)
    assert list(ca.identity_viterbi(ev2, flip2, marg, cont)) == [0] * 10 + [1] * 10
    # the same appearance without continuity (gaps) follows the evidence frame by frame
    s3 = ca.identity_viterbi(ev, flip, marg, np.zeros(T - 1, bool))
    assert (s3[8:11] == 1).all() and (s3[:8] == 0).all()


def test_identity_table_keeps_the_pair_assignment_through_a_bright_sleeve_frame():
    n = 12
    kp, box, rgb, conf, fr = [], [], [], [], []
    for f in range(n):
        for k, (x, col) in enumerate(((300.0, (175.0, 172, 170)), (500.0, (170.0, 30, 35)))):
            p = np.zeros((17, 3))
            p[:, 2] = 0.9
            p[:, 0] = x + f
            p[:, 1] = np.linspace(100, 500, 17)
            kp.append(p)
            box.append([x - 50, 100, x + 50, 500 - 5 * k])        # Carlos (slot 0) is the larger box
            c = col
            if f == 6 and k == 1:
                c = (176.0, 170, 168)                              # his sleeve covers her torso: she looks white too
            rgb.append(c)
            conf.append(0.9)
            fr.append(f)
    det = {"frame": np.array(fr), "box": np.array(box, float), "kp": np.array(kp, float), "rgb": np.array(rgb, float), "conf": np.array(conf)}
    tab = ca.identity_table(det, n, 100.0)
    assert tab["frames"] == n
    assert np.allclose(tab["kp"][:, 0, 0, 0], 300.0 + np.arange(n))     # Carlos stays the left one in every frame, also frame 6
    assert np.allclose(tab["kp"][:, 1, 0, 0], 500.0 + np.arange(n))


def test_apply_relabel_swaps_roles_and_keeps_the_tracker_label():
    cand = {"lead_01_000001": {"role": "lead", "cam": "01"}, "follow_01_000002": {"role": "follow", "cam": "01"}}
    out = ca.apply_relabel(cand, {"follow_01_000002": "lead", "nope": "lead", "lead_01_000001": "bogus"})
    assert out["follow_01_000002"]["role"] == "lead" and out["follow_01_000002"]["tracker_role"] == "follow"
    assert out["lead_01_000001"]["role"] == "lead" and cand["follow_01_000002"]["role"] == "follow"


def test_select_faces_rules():
    mk = lambda n, yaw=0.0, occ=40.0, sharp=100.0, ok=True: {"name": n, "ok": ok, "yaw": yaw, "occ_px": occ, "sharpness": sharp}   # noqa: E731
    cand = {n: {"cam": "01", "t": t} for n, t in (("a", 1.0), ("b", 1.5), ("c", 10.0), ("d", 20.0), ("e", 30.0))}
    recs = [mk("a", sharp=200), mk("b", sharp=300), mk("c", yaw=60), mk("d", occ=5), mk("e")]
    got = [r["name"] for r in ca.select_faces(recs, cand, 5, 42.0, 2.0, 12.0, ("01",), ("e",))]
    assert got == ["b"]                                             # a is within 2 s of b; c too turned; d too small; e excluded


def test_dominant_colours_two_clusters():
    rng = np.random.default_rng(0)
    black = rng.normal(20, 3, (300, 3))
    white = rng.normal(230, 3, (100, 3))
    cl = ca.dominant_colours(np.concatenate([black, white]), 2)
    assert len(cl) == 2 and cl[0]["share"] > 0.7 and cl[0]["rgb"][0] < 40 and cl[1]["rgb"][0] > 200
    assert cl[0]["hex"].startswith("#") and len(cl[0]["hex"]) == 7
    assert ca.dominant_colours(np.zeros((0, 3))) == []


def test_sectors_and_camera_quality():
    assert ca.sector_of(np.array([0.0]))[0] == 0 and ca.sector_of(np.array([np.pi]))[0] == ca.N_SECTORS // 2
    assert ca.sector_of(np.array([2 * np.pi / ca.N_SECTORS]))[0] == 1
    q = ca.camera_quality(np.array([10.0, 20.0]), np.array([4800.0, 4800.0]))
    assert q[0] > q[1] and q[0] == 1.2


def test_forward_axis_of_a_dancer_facing_plus_z():
    T = 3
    J = np.zeros((17, T, 3))
    J[5] = [1.0, 1.5, 0]          # left shoulder at +x (she faces +z, y up)
    J[6] = [-1.0, 1.5, 0]
    J[11] = [0.5, 1.0, 0]
    J[12] = [-0.5, 1.0, 0]
    f = ca._fwd(J)
    assert np.allclose(np.arctan2(f[:, 0], f[:, 2]), 0.0)


def test_window_scores_prefer_covered_windows_without_contact():
    T, N = 200, ca.N_SECTORS
    sight = np.zeros((T, 2, N))
    sight[:100, :, :] = 0.1                                    # first half: every sector, both dancers
    sight[100:, :, 0] = 0.1                                    # second half: only the front
    contact = np.zeros(T)
    sc = ca.window_scores(sight, contact, np.array([0, 100]), 100, 2.0, 0.1)
    assert sc[0] > sc[1]
    contact[:100] = 1.0
    sc2 = ca.window_scores(sight, contact, np.array([0, 100]), 100, 2.0, 0.1)
    assert sc2[0] < sc[0]
    assert ca.choose_windows(np.array([3.0, 2.9, 1.0]), np.array([0, 50, 400]), 100, 2, 20) == [0, 2]


def test_stage_table_is_consistent():
    names = [s.name for s in ca.STAGES]
    assert len(names) == len(set(names))
    parser_cmds = {"plan", "face-frames", "face-landmarks", "face-identity", "face-retarget", "hair-ref", "shoes-video", "texture-preview",
                   "body-shape", "head-review"}
    for s in ca.STAGES:
        if not s.poses:
            assert s.cmd.split()[0] in parser_cmds, s.name
    assert {s.name for s in ca.STAGES if s.poses} >= {"texture-select", "texture-run", "texture-fuse", "face-texture", "hair-groom", "export"}


def test_pod_job_is_a_dedicated_kind_no_other_worker_claims(tmp_path, monkeypatch):
    from dancecap import ca_avatars_pod as pod
    from dancecap.farm import worker as W
    monkeypatch.setattr(pod, "identity_archive", lambda b, r: {"path": "dancecap/work/x/in.tar", "archive": True, "sha256": "0" * 64, "bytes": 1,
                                                                 "local": "x.tar", "files": 1})
    spec = pod.identity_spec("ca-identity-test", None)
    assert spec["kind"] == ca.KIND == "avatarca" and spec["needs_gpu"] is False and spec["id"].startswith("ca-")
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / "ca-identity-test.json").write_text(json.dumps(spec))
    for kinds in ({"front", "fit", "refine", "full", "asr"}, {"analyse"}, {"avatar"}):       # the farm, the analysis pods, the recap's avatar pods
        w = W.Worker(tmp_path, pod_id="x", kinds=kinds, ws=tmp_path / "ws", work=tmp_path / "work", state=tmp_path / "st")
        assert not w.claimable(spec), kinds
    mine = W.Worker(tmp_path, pod_id="mine", kinds={ca.KIND}, ws=tmp_path / "ws", work=tmp_path / "work", state=tmp_path / "st")
    assert mine.claimable(spec)


def test_job_outputs_and_commands():
    from dancecap import ca_avatars_pod as pod
    for betas in (True, False):
        for ret in (False, True):
            for o in pod.identity_outputs(betas, ret):
                assert o["path"].startswith(f"dancecap/work/{ca.TAKE}/") and ".." not in o["path"]
    assert not any("_disp" in o["path"] for o in pod.identity_outputs(False, False))
    assert any(o["path"].endswith("lead_identity_disp.npz") for o in pod.identity_outputs(True, True))
    assert not any(o["path"].endswith("lead_identity.npz") for o in pod.identity_outputs(True, True))     # a retarget job does not write the identity
    for ret in (False, True):
        for st in pod.identity_steps(ret):
            for mod in re.findall(r"python -m (dancecap[\w.]*)", st["cmd"]):
                assert importlib.util.find_spec(mod) is not None
    assert "--retarget-only" in pod.identity_steps(True)[1]["cmd"]


def test_estimate_is_cheap():
    from dancecap import ca_avatars_pod as pod
    e = pod.estimate()
    assert 0 < e["usd"] < 0.5 and pod.estimate(True)["usd"] < e["usd"]


# ------------------------------------------------------------------------------------------------ pose-dependent parts (synthetic poses)

def _synthetic_work(tmp_path, T=900, spin_first=True):
    """30 Hz poses over 30 s: both dancers spin once per 10 s on the spot (Carlos at x = -0.5, Aline at x = +0.5); two fixed cameras 10 m away
    (+z) that record the whole time. -> (work, smplx_dir)"""
    work = tmp_path / "work"
    (work / "cameras").mkdir(parents=True)
    sm = work / "smplx"
    sm.mkdir()
    t = 15.0 + np.arange(T) / 30.0
    for r, x in (("lead", -0.5), ("follow", 0.5)):
        yaw = 2 * np.pi * (t - t[0]) / 10.0
        go = np.zeros((T, 3))
        go[:, 1] = yaw                                  # rotation about y
        joints = np.zeros((T, 3, 3))
        joints[:, 0] = [x, 1.0, 0.0]
        np.savez(sm / f"{r}.npz", t=t, joints=joints, global_orient=go)
    for c, f, cx in (("01", 4800.0, -6.4), ("02", 3350.0, 0.4)):
        n = 1200
        np.savez(work / "cameras" / f"{c}.npz", t=np.linspace(10.0, 50.0, n), ok=np.ones(n, bool), focal_px=np.full(n, f), centre_world=np.array([cx, 1.5, 10.0]))
    return work, sm


def test_texture_select_on_synthetic_spinning_dancers(tmp_path):
    work, sm = _synthetic_work(tmp_path)
    res = ca.texture_select(work, sm, length_s=20.0, step_s=5.0, need_s=1.5, max_windows=1, t_range=(15.0, 44.0), out=work / "w.json")
    w = res["windows"][0]
    assert w["id"] == "tx1" and w["t1"] - w["t0"] == 20.0 and w["mean_coverage"] > 0.9       # a spin shows every side
    assert (work / "w.json").exists() and res["t_range"][0] >= 15.0


def test_texture_select_a_camera_that_is_off_gives_less_coverage(tmp_path):
    work, sm = _synthetic_work(tmp_path)
    for c in ("01", "02"):
        z = dict(np.load(work / "cameras" / f"{c}.npz"))
        z["ok"] = np.zeros_like(z["ok"])
        np.savez(work / "cameras" / f"{c}.npz", **z)
    res = ca.texture_select(work, sm, length_s=20.0, step_s=5.0, need_s=1.5, max_windows=1, t_range=(15.0, 44.0))
    assert res["windows"][0]["mean_coverage"] == 0.0


def test_recolour_groom_changes_colour_only():
    g = {"colour": {"mid": {"linear_srgb": [0.2, 0.03, 0.02], "hex": "#7A2F28"}, "source": "x"},
         "render": {"albedo_gain": 1.85, "albedo_linear": [0.36, 0.05, 0.04], "copper_lift": [1.12, 1.18, 1.0], "copper_lift_m": 0.1,
                    "kajiya_kay": {"primary_tint": [1.0, 0.92, 0.88], "highlight_linear": [0.66, 0.1, 0.1]}},
         "guides": [[1, 2, 3]], "style": {"texture": "sleek"}}
    out = ca.recolour_groom(g)
    assert out["guides"] == g["guides"] and out["style"] == g["style"]                       # geometry untouched
    assert g["render"]["copper_lift_m"] == 0.1                                                # the input is not modified
    assert out["render"]["copper_lift_m"] == 0.0 and out["render"]["copper_lift"] == [1.0, 1.0, 1.0]
    assert max(out["render"]["albedo_linear"]) < 0.05 and out["colour"]["mid"]["hex"] == ca.srgb_hex(np.array(ca.BLACK_HAIR["mid"]))
    assert out["render"]["kajiya_kay"]["highlight_linear"] == out["colour"]["highlight"]["linear_srgb"]


import pytest  # noqa: E402

_CASHAPE = ca.WORK / "farm" / "jobs" / "cashape.json"


@pytest.mark.skipif(not _CASHAPE.exists(), reason="needs the take's git-ignored farm job specs")
def test_texture_spec_is_a_dedicated_gpu_job_with_the_takes_front_env(tmp_path, monkeypatch):
    from dancecap import ca_avatars_pod as pod
    monkeypatch.setattr(pod, "texture_archive", lambda w, j, s, y: {"path": f"dancecap/work/{pod.texture_take(w)}/texture_in.tar", "archive": True,
                                                                     "sha256": "0" * 64, "bytes": 1, "local": "x.tar", "files": 1})
    job = pod.window_job("tx1", 60.0, 105.0)
    assert job["chunk"]["k0"] == 1800 and job["chunk"]["k1"] == 3149 and job["take"] == "carlos_aline_tx1" and job["cams"] == ["01", "02"]
    spec = pod.texture_spec("ca-texture-test", "tx1", 100.0, tmp_path / "j.json", tmp_path, tmp_path)
    assert spec["kind"] == ca.KIND and spec["needs_gpu"] is True and spec["id"].startswith("ca-")
    assert spec["env"].get("HM_FRONT_DANCE") == ca.TAKE
    names = [s["name"] for s in spec["steps"]]
    assert names == ["chunk_take", "weights", "front", "stage", "bake"]
    assert any(i["path"].endswith("ca_in.tar") and "local" not in i for i in spec["inputs"])
    assert "dancecap texture carlos_aline_tx1 --preset pod" in spec["steps"][-1]["cmd"]
    own = pod.texture_spec("ca-texture-test", "tx1", 100.0, tmp_path / "j.json", tmp_path, tmp_path, mask="02:10:12", kind="avatarcatx1")
    assert own["kind"] == "avatarcatx1" and "maskcams" in [s["name"] for s in own["steps"]]


def test_fusion_roles_of_this_pair_and_the_recap_defaults_are_kept():
    import inspect

    from dancecap import recap_avatars_fuse as fz
    assert set(ca.FUSE_LIGHT_ROLES) == {"lead", "follow"}                 # hair and a dark sleeve only darken his white top AND her skin / red cloth
    assert set(ca.FUSE_POWERS) == {"lead", "follow"}
    sig = inspect.signature(fz.run).parameters
    assert sig["light_roles"].default == ("follow",) and sig["powers"].default is None   # the recap's own behaviour is the default


def test_texture_job_carries_the_teacher_evidence_switch(tmp_path, monkeypatch):
    if not _CASHAPE.exists():
        import pytest
        pytest.skip("needs the take's git-ignored farm job specs")
    from dancecap import ca_avatars_pod as pod
    monkeypatch.setattr(pod, "texture_archive", lambda w, j, s, y: {"path": "x.tar", "archive": True, "sha256": "0" * 64, "bytes": 1, "local": "x.tar", "files": 1})
    spec = pod.texture_spec("ca-texture-test", "tx1", 100.0, tmp_path / "j.json", tmp_path, tmp_path)
    assert spec["env"]["HM_TEACHER_EVIDENCE"] == "1"


def test_red_top_switch_turns_the_red_hair_rule_off_only_when_set(monkeypatch):
    from dancecap import avatar_hair as ah
    lab = np.array([[40.0, 40.0, 20.0], [80.0, 5.0, 6.0]])            # a red-cast texel (cloth / hair) and a white one
    monkeypatch.delenv("HM_RED_TOP", raising=False)
    assert list(ah.red_cast(lab)) == [True, False]                    # the recap's / demo's rule is unchanged
    monkeypatch.setenv("HM_RED_TOP", "1")
    assert list(ah.red_cast(lab)) == [False, False]                   # her red top is cloth, not leaked hair


def test_recolour_groom_scales_or_drops_the_tip_glow_and_darkens_the_hair():
    g = {"colour": {}, "render": {"albedo_gain": 1.85, "tip_emission": {"e": [0.0, 0.012, 0.024, 0.219, 1.398, 3.1]}, "kajiya_kay": {}}}
    off = ca.recolour_groom(g, tip_glow=0.0)
    assert all(x == 0 for x in off["render"]["tip_emission"]["e"]) and off["render"]["tip_emission"]["scaled_by"] == 0.0
    on = ca.recolour_groom(g, tip_glow=1.0)
    assert on["render"]["tip_emission"]["e"] == g["render"]["tip_emission"]["e"]          # 1 = the original curve, untouched
    dark = ca.recolour_groom(g, albedo_gain=ca.HAIR_ALBEDO_GAIN)
    assert max(dark["render"]["albedo_linear"]) < 0.02                                   # near-black, not the red groom's gain
    assert g["render"]["albedo_gain"] == 1.85                                            # the input groom is not modified


def test_fusion_window_weights_default_to_one():
    import inspect

    from dancecap import recap_avatars_fuse as fz
    assert inspect.signature(fz.run).parameters["window_weights"].default is None
