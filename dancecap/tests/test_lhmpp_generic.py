"""dancecap.mono4d.lhmpp_generic (2026-10-22): the generic LHM++ avatars step of a one-phone take. Synthetic data only: the frame
choice rules, the facing estimates, the mask checks, LHM++'s crop format, the pod inputs and the pod job document. No video,
no pod.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_lhmpp_generic.py -q        (from dancecap/ or the workspace)
"""
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.mono4d import cand_lhmpp as CL  # noqa: E402
from dancecap.mono4d import lhmpp_generic as G  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402

W, H, FPS = 480, 848, 30.0


def _person(cx: float, top: float, h: float, front: bool = True) -> np.ndarray:
    """COCO-17 keypoints (17, 3) of an upright person h px tall, head at top; front = facing the camera (the person's left
    shoulder on the image right)."""
    s = 1 if front else -1
    kp = np.zeros((17, 3))
    sh, hip, kn, an = top + 0.18 * h, top + 0.5 * h, top + 0.72 * h, top + 0.95 * h
    pts = {0: (cx, top + 0.07 * h), 1: (cx + s * 4, top + 0.05 * h), 2: (cx - s * 4, top + 0.05 * h),
           3: (cx + s * 8, top + 0.06 * h), 4: (cx - s * 8, top + 0.06 * h), 5: (cx + s * 0.13 * h, sh),
           6: (cx - s * 0.13 * h, sh), 7: (cx + s * 0.16 * h, top + 0.33 * h), 8: (cx - s * 0.16 * h, top + 0.33 * h),
           9: (cx + s * 0.17 * h, top + 0.47 * h), 10: (cx - s * 0.17 * h, top + 0.47 * h), 11: (cx + s * 0.08 * h, hip),
           12: (cx - s * 0.08 * h, hip), 13: (cx + s * 0.08 * h, kn), 14: (cx - s * 0.08 * h, kn),
           15: (cx + s * 0.08 * h, an), 16: (cx - s * 0.08 * h, an)}
    for j, (x, y) in pts.items():
        kp[j] = (x, y, 0.9)
    return kp


def _take(F: int = 300):
    """Two upright dancers far apart, everything clean; facing turns 36 deg / s."""
    t = np.arange(F) / FPS
    kp = np.zeros((F, 2, 17, 3))
    kp[:, 0] = _person(140, 250, 300)
    kp[:, 1] = _person(340, 260, 280)
    box = np.zeros((F, 2, 4))
    box[:, 0] = (90, 250, 190, 550)
    box[:, 1] = (290, 260, 390, 540)
    T = {"t": t, "frame_index": np.arange(F), "ok": np.ones(F, bool), "kp": kp, "conf": np.full((F, 2), 0.8),
         "role_conf": np.full((F, 2), 0.95), "visible": np.full((F, 2), 0.9), "skel_in_partner": np.zeros((F, 2)),
         "edge": np.zeros((F, 2), bool), "size": np.array([W, H])}
    M = {"area": np.full((F, 2), 15000.0), "box": box, "size": (W, H), "partner_occ": np.zeros((F, 2)),
         "main_frac": np.ones((F, 2)), "head_cov": np.ones((F, 2)), "sharp": np.full((F, 2), 100.0)}
    yaw = ((36.0 * t + 180.0) % 360.0) - 180.0
    facing = np.stack([yaw, yaw], 1)
    return T, M, facing


# ------------------------------------------------------------------------------------------------ facing


def test_yaw_2d_front_back_and_face():
    front, back = _person(200, 100, 300, True), _person(200, 100, 300, False)
    assert abs(G.yaw_2d(front)) < 1.0
    assert abs(G.yaw_2d(back)) > 179.0
    # the detector's shoulders flipped on a back view, but the face landmarks say "not seen": back
    assert abs(G.yaw_2d(front, np.array(0.4))) > 179.0
    assert abs(G.yaw_2d(back, np.array(0.95))) < 1.0
    unseen = front.copy()
    unseen[[5, 6, 11, 12], 2] = 0
    assert np.isnan(G.yaw_2d(unseen))


def test_body_yaw_matches_cand_lhmpp():
    from scipy.spatial.transform import Rotation as Rot
    rng = np.random.default_rng(3)
    N = 20
    go = rng.normal(size=(N, 3))
    pel = rng.normal(size=(N, 3))
    R = Rot.from_rotvec(rng.normal(size=(N, 3)) * 0.5).as_matrix()
    tv = rng.normal(size=(N, 3)) + [0, 0, 4]
    ours = G.body_yaw_deg(go, pel, R, tv)
    ref = [CL.body_yaw_deg(go[i], pel[i], R[i], tv[i]) for i in range(N)]
    assert np.allclose(ours, ref, atol=1e-9)


def test_pose_speed_ignores_travel():
    F = 10
    t = np.arange(F) / FPS
    kp = np.repeat(_person(200, 100, 300)[None, None], F, 0).repeat(2, 1)
    kp[:, 1, :, 0] += np.arange(F)[:, None] * 20.0          # dancer 1 travels 600 px / s: no pose change
    sp = G.pose_speed(kp, t)
    assert np.array_equal(sp[0], sp[1]) and np.array_equal(sp[-1], sp[-2])
    assert np.nanmax(sp) < 1e-9
    kp[:, 0, [7, 8, 9, 10, 13, 14, 15, 16], 1] += np.arange(F)[:, None] * 15.0   # dancer 0 swings arms and legs
    sp = G.pose_speed(kp, t)
    assert np.nanmin(sp[:, 0]) > 3.0 and np.nanmax(sp[:, 1]) < 1e-9


# ------------------------------------------------------------------------------------------------ frame choice


def test_choose_frames_gates_spread_and_frontal_first():
    T, M, facing = _take()
    F = len(T["t"])
    bad = {"cut": range(0, 30), "box_iou": range(30, 60), "skel_in_partner": range(60, 90), "role_conf": range(90, 120),
           "partner_occ": range(120, 150), "head_cov": range(150, 180)}
    r = np.array(list(bad["cut"]))
    M["box"][r, 0] = (0, 250, 100, 550)
    r = np.array(list(bad["box_iou"]))
    M["box"][r, 1] = (100, 250, 200, 550)                     # the partner's box on top of the lead's
    T["skel_in_partner"][list(bad["skel_in_partner"]), 0] = 0.5
    T["role_conf"][list(bad["role_conf"]), 0] = 0.5
    M["partner_occ"][list(bad["partner_occ"]), 0] = 0.1
    M["head_cov"][list(bad["head_cov"]), 0] = 0.1
    sel = G.choose_frames(T, M, n=8, min_gap_s=1.0, facing=facing)
    lead = sel["lead"]
    rows = [p["row"] for p in lead["picks"]]
    assert len(rows) == 8 and len(set(rows)) == 8
    assert all(f >= 180 for f in rows), rows                 # every bad stretch avoided
    assert lead["n_relaxed_picks"] == 0
    assert lead["min_gap_used_s"] < 1.0                       # 4 clean seconds cannot hold 8 picks 1 s apart
    ts = sorted(T["t"][rows])
    assert min(np.diff(ts)) >= lead["min_gap_used_s"] - 1e-9
    for k in bad:
        assert lead["drop_counts"].get(k, 0) >= 30, (k, lead["drop_counts"])
    clean_rows = np.arange(180, F)
    assert abs(lead["picks"][0]["yaw_deg"]) == pytest.approx(np.abs(facing[clean_rows, 0]).min(), abs=0.06)
    yaws = np.abs([p["yaw_deg"] for p in lead["picks"]])
    assert yaws.min() < 45 and yaws.max() > 135               # front AND back views
    assert all(p["yaw_src"] == "fit" for p in lead["picks"])
    # the follow only shares the box_iou stretch: it gets 8 clean picks 1 s apart
    fol = sel["follow"]
    assert len(fol["picks"]) == 8 and fol["min_gap_used_s"] == 1.0
    assert not any(30 <= p["row"] < 60 for p in fol["picks"])


def test_choose_frames_relaxes_when_clean_frames_run_out():
    T, M, facing = _take()
    T["role_conf"][:, 0] = 0.75                               # between RELAXED (0.7) and CLEAN (0.85) everywhere ...
    T["role_conf"][[20, 140, 260], 0] = 0.95                  # ... except three frames
    sel = G.choose_frames(T, M, n=8, min_gap_s=1.0, facing=facing)["lead"]
    assert len(sel["picks"]) == 8
    assert sel["clean"] == 3 and sel["n_relaxed_picks"] == 5
    assert not sel["picks"][0]["relaxed"]
    assert {p["row"] for p in sel["picks"] if not p["relaxed"]} == {20, 140, 260}
    assert sel["min_gap_used_s"] == 0.25


def test_unsurveyed_frames_and_hard_gate():
    T, M, facing = _take()
    for k in ("partner_occ", "main_frac", "head_cov", "sharp"):
        M[k][:150] = np.nan                                   # the survey never reached the first half
    M["area"][200:, 0] = 500                                  # no mask in the last part
    sel = G.choose_frames(T, M, n=4, min_gap_s=1.0, facing=facing)["lead"]
    assert all(150 <= p["row"] < 200 for p in sel["picks"])
    assert sel["drop_counts"]["not surveyed"] == 150 and sel["drop_counts"]["mask"] == 100
    assert sel["n_relaxed_picks"] == 0


def test_frontal_pick_prefers_face_and_arms_down():
    T, M, facing = _take(60)
    facing[:] = 20.0
    T["kp"][:30, 0, [9, 10], 1] = 200                        # wrists above the head in the first second
    T["conf"][:30, 0] = 0.9                                   # ... even though those frames score higher
    sel = G.choose_frames(T, M, n=2, min_gap_s=0.5, facing=facing)["lead"]
    assert sel["picks"][0]["row"] >= 30


# ------------------------------------------------------------------------------------------------ masks and crops


def test_head_cov_and_mask_stats():
    kp = _person(200, 100, 300)
    own = np.zeros((H, W), bool)
    own[140:400, 150:250] = True                              # neck to hips ... (no head)
    assert G.head_cov(own, kp) < 0.05
    cv2.circle(own.view(np.uint8), (200, 119), 26, 1, -1)    # ... + the head above the shoulders
    assert G.head_cov(own, kp) > 0.9
    hand = np.zeros_like(own)
    hand[250:275, 170:230] = True                             # the partner's hand in front of the dancer's waist
    st = G.mask_stats(own & ~hand, hand, kp)
    assert st["partner_occ"] > 0.04 and st["main_frac"] == 1.0 and st["head_cov"] > 0.9
    arm = np.zeros_like(own)
    arm[250:280, 120:280] = True                              # the partner's arm right across: the mask in two parts
    assert G.mask_stats(own & ~arm, arm, kp)["main_frac"] < G.CLEAN["main_frac"]
    behind = np.zeros_like(own)
    behind[150:400, 240:330] = True                           # the partner BEHIND (own mask complete)
    assert G.mask_stats(own, behind & ~own, kp)["partner_occ"] < 0.01


def test_crop_format_white_outside_mask():
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (200, 30, 30)
    m = np.zeros((H, W), bool)
    cv2.ellipse(m.view(np.uint8), (240, 420), (50, 150), 0, 0, 360, 1, -1)   # 100 x 300 px dancer
    c = G.crop_ref(img, m)
    assert c.shape == (840, 504, 3) and c.dtype == np.uint8
    assert (c[:5, :5] == 255).all() and (c[-5:, -5:] == 255).all()
    assert (c[:, :100] == 255).all()                          # 100 x 300 -> 280 x 840 centred: columns 0..111 white
    assert tuple(c[420, 252]) == (200, 30, 30)
    assert ((c == 255).all(-1) | (np.abs(c.astype(int) - [200, 30, 30]).max(-1) < 3)).mean() > 0.98
    assert (c[30, 252] == 255).all() or (c[30, 160] == 255).all()   # bbox corners outside the ellipse: white
    # a stray blob does not move the crop (cleaned mask), and the crop is exactly cand_lhmpp's
    m2 = m.copy()
    m2[5:8, 5:8] = True
    assert np.array_equal(G.crop_ref(img, m2), c)
    ref = (np.clip(CL.lhm_crop(img, m), 0, 1) * 255 + 0.5).astype(np.uint8)
    assert np.array_equal(c, ref)


def test_photo_ref_rgba(tmp_path):
    rgba = np.zeros((400, 200, 4), np.uint8)
    rgba[..., :3] = (10, 120, 240)                            # BGR on disk
    rgba[50:350, 50:150, 3] = 255
    p = tmp_path / "cut.png"
    cv2.imwrite(str(p), rgba)
    c = G.photo_ref(p)
    assert c.shape == (840, 504, 3) and (c[:3, :3] == 255).all()
    assert tuple(c[420, 252]) == (240, 120, 10)               # RGB out


# ------------------------------------------------------------------------------------------------ pod inputs, job, import


def test_write_pod_inputs_match_the_pod_scripts(tmp_path):
    betas = {"lead": np.arange(16) * 0.1, "follow": np.zeros(16)}
    fit = {"t": np.arange(5) / 30.0, "absent": np.array([0, 0, 1, 0, 0], bool), "global_orient": np.ones((5, 3)),
           "body_pose": np.zeros((5, 63)), "jaw_pose": np.zeros((5, 3)), "leye_pose": np.zeros((5, 3)),
           "reye_pose": np.zeros((5, 3)), "left_hand_pose": np.zeros((5, 45)), "right_hand_pose": np.zeros((5, 45)),
           "joints": np.arange(5 * 55 * 3, dtype=float).reshape(5, 55, 3)}
    k_fit = G.pose_key(fit, 2 / 30.0 + 0.001, betas["lead"])  # the nearest instant is absent: its neighbour
    assert k_fit["src"] == "fit" and k_fit["t"] in (1 / 30.0, 3 / 30.0)
    k_zero = G.pose_key(None, 1.0, betas["follow"])
    assert k_zero["src"].startswith("zero pose") and np.abs(k_zero["body"]).max() == 0
    G.write_pod_inputs(tmp_path, {"lead": k_fit, "follow": k_zero}, betas)
    with np.load(tmp_path / "poses.npz") as z:
        Z = {k: z[k] for k in z.files}
    with np.load(tmp_path / "views.npz") as z:
        assert z.files == []
    vj = json.loads((tmp_path / "views.json").read_text())
    assert vj["groups"] == {} and vj["passes"] == []
    NK = len(Z["t"])
    assert NK == 1
    for role in G.ROLES:                                      # lhmpp_render_pod.animate / fk_joints / trans
        for k, shp in (("root", (3,)), ("body", (21, 3)), ("jaw", (3,)), ("leye", (3,)), ("reye", (3,)),
                       ("lhand", (15, 3)), ("rhand", (15, 3))):
            assert np.asarray(Z[f"{role}_{k}"][[0]], np.float32).reshape(1, *shp).shape == (1, *shp)
        assert Z[f"{role}_pelvis"].shape == (1, 3) and Z[f"{role}_joints22"].shape == (1, 22, 3)
        assert Z[f"{role}_betas"].shape == (16,)
        idx = np.linspace(0, NK - 1, 60).round().astype(int)
        assert Z[f"{role}_joints22"][idx].shape == (60, 22, 3)
    assert np.allclose(Z["lead_pelvis"][0], fit["joints"][k_fit["t"] * 30 > 2 and 3 or 1, 0])


def test_job_doc_fields():
    d = G.job_doc("synth_take", "dancecap/work/synth_take/realism/lhmpp_generic/input",
                  "dancecap/work/synth_take/realism/lhmpp_generic/pod_out")
    assert d["schema"] == "dancecap.pod_job/1" and d["name"] == "synth_take_lhmpp_avatars" and d["take"] == "synth_take"
    assert d["needs"] == [] and d["min_vram_gb"] == 24 and d["cwd"] == "dancecap"
    assert set(G.EXCLUDE) <= set(d["exclude"]) and "mamma-turing" in d["exclude"]
    assert any("pywheels" in e for e in d["exclude"]) and any("wheelhouse" in e for e in d["exclude"])
    assert d["inputs"] == ["dancecap/work/synth_take/realism/lhmpp_generic/input/"]
    assert d["outputs"] == ["dancecap/work/synth_take/realism/lhmpp_generic/pod_out/"]
    assert d["env"]["LHMPP_COMMIT"] == CL.LHMPP_COMMIT and d["env"]["LHMPP_MODEL"] == CL.MODEL_NAME
    names = [c["name"] for c in d["commands"]]
    assert names == ["install", "build", "export"]
    assert d["commands"][0]["cmd"] == "bash remote/cand_lhmpp/install.sh"
    build = d["commands"][1]["cmd"]
    assert "remote/cand_lhmpp/lhmpp_render_pod.py" in build and build.startswith(G.POD_PY)
    assert "--input work/synth_take/realism/lhmpp_generic/input " in build      # relative to the commands' cwd
    assert d["commands"][2].get("optional") is True and "recap_lhmpp_pod.py" in d["commands"][2]["cmd"]
    assert all(f"gs/{r}_cano.ply" in " ".join(d["verify"]) for r in G.ROLES)
    assert all(f"turntable/{r}.png" in " ".join(d["verify"]) for r in G.ROLES)
    assert sum(c["timeout_min"] for c in d["commands"][:2]) <= d["max_minutes"]
    assert [c["name"] for c in G.job_doc("x", "dancecap/a", "dancecap/b", export=False)["commands"]] == ["install", "build"]
    from dancecap import remote_jobs as rj
    js = rj._parse_v1(d, "test")
    assert js.cwd == "dancecap" and js.min_vram_gb == 24 and [c.optional for c in js.commands] == [False, False, True]


def test_ws_rel_keeps_the_junction_path():
    p = PACKAGE_ROOT / "work" / "synth_take" / "realism" / "lhmpp_generic" / "input"
    assert G.ws_rel(p) == "dancecap/work/synth_take/realism/lhmpp_generic/input"
    assert G.default_dir("synth_take") == PACKAGE_ROOT / "work" / "synth_take" / "realism" / "lhmpp_generic"


def test_ply_info(tmp_path):
    n = 5
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\n"
            "property float z\nend_header\n").encode()
    p = tmp_path / "a.ply"
    p.write_bytes(head + struct.pack(f"<{3 * n}f", *range(3 * n)))
    info = G.ply_info(p)
    assert info["ok"] and info["gaussians"] == n and info["properties"] == 3
    p.write_bytes(head + b"\0" * 8)                           # truncated
    assert not G.ply_info(p)["ok"]
