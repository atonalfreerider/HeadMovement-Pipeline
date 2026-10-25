"""The recap's facts the sides step must keep (DESIGN 8; skipped without the local work data): R6 undecided and R7 "inside hers" at
53.8-62.3 s, the input's arm pass-throughs at 30.27-34.87 / 74.30 / 74.53 s found, the gate calibration of DESIGN 7.2, the 199-s splice
exit at 200.10 +- 0.1 s, the Carlos & Aline unwraps (joints within 1 cm, wraps 0)."""
import json

import numpy as np
import pytest

from dancecap.take import PACKAGE_ROOT

R = PACKAGE_ROOT / "work" / "larissa_kadu_recap" / "sides"
CA = PACKAGE_ROOT / "work" / "carlos_aline" / "smplx_rf2_noslip" / "follow.npz"
need = pytest.mark.skipif(not (R / "relations.npz").exists(), reason="no local recap work data")


@need
def test_r6_undecided_and_r7_inside_hers_in_the_first_counterbalance():
    rel = np.load(R / "relations.npz")
    t = rel["t"]
    m = (t >= 53.8) & (t < 62.3)
    assert (rel["R6__s"][m] == 0).all()
    s7 = rel["R7__s"][m]
    assert (s7 < 0).sum() >= 30 and (s7 > 0).sum() == 0


@need
def test_the_inputs_arm_passthroughs_are_found():
    v = json.loads((R / "violations.json").read_text(encoding="utf-8"))["items"]
    ts = sorted(it["t1"] for it in v if it["reason"] == "thru" and it.get("rel") == "R9_RL" and it["t1"] < 100)
    for want in (30.27, 32.47, 34.87, 74.30, 74.53):
        assert min(abs(x - want) for x in ts) < 0.05, (want, ts)
    assert sum(30.0 <= x <= 35.0 for x in ts) == 7


@need
def test_gate_calibration_passes():
    g = json.loads((R / "gate_calibration.json").read_text(encoding="utf-8"))
    c = g["cases"]
    assert c["v43_runs"]["n"] == 16 and c["v43_runs"]["rejected"] >= 15
    assert c["splice"]["ok"] and c["noop"]["ok"] and c["synthetic_swing"]["ok"] and c["synthetic_passthrough_removal"]["ok"]
    assert g["ok"]


@need
def test_splice_exit_at_200p1():
    p = json.loads((R / "cpu" / "prepare.json").read_text(encoding="utf-8"))
    s = [x for x in p["splices"] if 198.0 < x["t0"] < 200.0]
    assert len(s) == 1
    assert abs(s[0]["t_exit"] - 200.10) <= 0.1
    assert s[0]["pelvis_only_exit_t"] < s[0]["t_exit"] - 0.2      # the pelvis alone agrees earlier, with a foot still wound


@pytest.mark.skipif(not CA.exists(), reason="no local carlos_aline fit")
def test_carlos_aline_unwraps():
    from dancecap import stitch
    from dancecap.sides import unwrap as uw
    with np.load(CA, allow_pickle=True) as z:
        fit = {k: z[k] for k in z.files}
    rest = stitch.rest_joints(fit["betas"])
    new, rep = uw.unwrap_fit(fit, rest=rest, split=False)
    after = uw.chain_wraps(new, rest=rest)
    t = fit["t"]
    for a, b in ((126.13, 127.87), (154.47, 156.50)):
        m = (t >= a - 1e-6) & (t <= b + 1e-6)
        assert np.linalg.norm(new["joints"][m] - fit["joints"][m], axis=-1).max() < 0.01
        assert sum(int(after[c]["wrapped"][m].sum()) for c in after) == 0


@pytest.mark.skipif(not (R / "gate.json").exists() or not (R / "unity_motion" / "unity_motion.json").exists(),
                    reason="no merged sides fit / side export")
def test_the_merged_fit_and_its_export_change_only_adopted_windows():
    """the whole-take gate passes (outside the windows byte-identical to the input), the merged files are the ones gate.json
    recorded, and the side export changes no motion row outside the adopted windows / the prepare splices"""
    from dancecap.sides.evidence import sha256
    g = json.loads((R / "gate.json").read_text(encoding="utf-8"))
    assert g["take_gate"]["ok"] and all(g["take_gate"]["outside_identical"].values())
    for r, o in g["outputs"].items():
        assert sha256(PACKAGE_ROOT / "work" / "larissa_kadu_recap" / "smplx_sides" / f"{r}.npz") == o["sha256"]
    u = json.loads((R / "unity_motion" / "unity_motion.json").read_text(encoding="utf-8"))
    p = u["proof_vs_input_export"]
    assert p["ok"] and p["lead"]["changed_rows_outside_allowed"] == 0 and p["follow"]["changed_rows_outside_allowed"] == 0


@pytest.mark.skipif(not (R / "gate.json").exists() or not (R / "gate_calibration.json").exists(), reason="no merged sides fit")
def test_the_window_whose_arm_left_the_phones_stays_reverted():
    """verify_r0 major 2: the window at dance 228.37-233.63 s moved his right arm away from every phone that sees it, hidden by the
    pooled G1 median (spine3 moved, so the torso and both arms were pooled). The per-chain G1 rejects every attempt: the merged fit keeps
    the input there. The gate calibration ran with the per-chain gate on and still accepts the 199-s splice."""
    g = json.loads((R / "gate.json").read_text(encoding="utf-8"))
    w = [x for x in g["windows"].values() if abs(x["t0"] - 228.37) < 0.05]
    assert w and all(x["status"] == "revert" for x in w)
    c = json.loads((R / "gate_calibration.json").read_text(encoding="utf-8"))
    assert c["cfg"]["gate_per_chain"] and c["cases"]["splice"]["ok"] and c["ok"]
