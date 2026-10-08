"""mamma_chunks merge gate: per-joint seam test + the rejected merge's re-tag (2026-10-07 larissa_kadu_38 pod session).

The pooled body-joint p95 of the first version could not see one wrong limb: on follow seam 2 the pooled body p95
was 44 mm while her left wrist was 107 mm apart through the blend window. The gate now tests every joint's own p95.
"""
import json

import numpy as np

from dancecap import mamma_chunks as mc


def _report(seam):
    rr = {"seam_mm": [seam], "betas_swap_mm": [{"chunk": "a", "median": 5.0, "p95": 18.0}]}
    return {"roles": {"follow": rr}, "chunk_frames": {"a": 232, "b": 232},
            "chunk_median_reproj_px": {"a": {"follow": 6.9}, "b": {"follow": 7.0}},
            "smplx": {"follow": {"median_reproj_px": 6.92}}}


def _joints(n=48, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.3, (1, 55, 3)) + np.zeros((n, 1, 1))
    return base + rng.normal(0, 0.002, (n, 55, 3))  # ~3 mm of fit noise per coordinate


def test_one_wrong_wrist_fails_although_pooled_body_p95_is_small():
    jp = _joints()
    jn = jp + np.random.default_rng(1).normal(0, 0.002, jp.shape)
    jn[:, 20] += [0.10, 0.0, 0.0]  # left wrist 10 cm off in the later chunk, every frame
    jn[:, 25:40] += [0.10, 0.0, 0.0]  # its fingers ride along with it (unchanged relative to the wrist)
    s = mc.seam_stats(jp, jn)
    assert s["p95_body"] < 60  # 1 of 22 joints: the pooled p95 does not see it
    assert s["blend"]["arm_joint"] == "l_wrist" and s["blend"]["arm_p95_max"] > 95
    assert s["blend"]["fingers_rel_wrist_p95_max"] < 20  # fingers relative to the wrist agree
    g = mc.gate(_report({"between": ["a", "b"], **s}))
    assert not g["passed"] and any("l_wrist" in w for w in g["why"])
    assert g["checks"]["follow"]["seam_arm_worst"].startswith("l_wrist")


def test_noise_only_passes_and_finger_curl_has_its_own_limit():
    jp = _joints()
    jn = jp + np.random.default_rng(2).normal(0, 0.003, jp.shape)
    assert mc.gate(_report({"between": ["a", "b"], **mc.seam_stats(jp, jn)}))["passed"]
    jc = jn.copy()
    jc[:, 37:40] += [0.0, -0.05, 0.0]  # a different finger curl (5 cm at the tips): tolerated
    assert mc.gate(_report({"between": ["a", "b"], **mc.seam_stats(jp, jc)}))["passed"]
    jf = jn.copy()
    jf[:, 25:40] += [0.0, -0.15, 0.0]  # a flipped hand (15 cm) fails
    g = mc.gate(_report({"between": ["a", "b"], **mc.seam_stats(jp, jf)}))
    assert not g["passed"] and any("fingers" in w for w in g["why"])


def test_blend_window_only_is_gated():
    jp = _joints()
    jn = jp + np.random.default_rng(3).normal(0, 0.002, jp.shape)
    jn[:4, 7] += [0.0, 0.0, 0.12]  # ankle off in the first quarter only (not cross-faded)
    s = mc.seam_stats(jp, jn)
    assert s["overlap"]["joint_p95_max"] > 60 and s["blend"]["joint_p95_max"] < 30
    assert mc.gate(_report({"between": ["a", "b"], **s}))["passed"]


def test_old_reports_use_the_pooled_all_joint_p95():
    old = {"between": ["a", "b"], "median": 10.0, "p95": 82.66, "p95_body": 44.0, "pelvis_median": 9.0}
    g = mc.gate(_report(old))
    assert not g["passed"] and "seam p95 83" in g["why"][0]
    assert mc.gate(_report({**old, "p95": 50.0}))["passed"]


def test_rejected_merge_is_retagged(tmp_path):
    d = tmp_path / "c8_merged_rejected"
    d.mkdir()
    (d / "fit.json").write_text(json.dumps({"tag": "c8", "run_tag": "c8", "partial": False, "chunked": True}))
    mc._retag_rejected(d)
    r = json.loads((d / "fit.json").read_text())
    assert r["tag"] == "c8_merged_rejected" and r["partial"] is True and r["rejected"] is True
