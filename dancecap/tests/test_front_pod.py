"""Offline tests for dancecap.front_pod (scheduler) and dancecap.mamma_chunks (chunk plan, seam weights, slerp)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import front_pod as fp  # noqa: E402
from dancecap import mamma_chunks as mc  # noqa: E402


def _runner(tmp_path, caps=None, budget=100.0):
    return fp.Runner(tmp_path, dict(os.environ), budget, caps or {"cpu": 2}, tmp_path)


def test_runner_deps_after_and_dynamic_cmd(tmp_path):
    R = _runner(tmp_path)
    R.add(fp.Task("a", [sys.executable, "-c", "print(1)"]))
    R.add(fp.Task("b", lambda: [sys.executable, "-c", "import sys; sys.exit(3)"], deps=["a"]))
    R.add(fp.Task("c", [sys.executable, "-c", "print(2)"], after=["b"]))   # soft: runs after b failed
    R.add(fp.Task("d", [sys.executable, "-c", "print(3)"], deps=["b"]))    # hard: skipped
    R.add(fp.Task("e", None, fn=lambda: None, after=["c", "d"]))
    R.loop(0.1)
    st = {k: t.state for k, t in R.tasks.items()}
    assert st == {"a": "ok", "b": "failed", "c": "ok", "d": "skipped", "e": "ok"}
    assert R.tasks["b"].rc == 3
    assert (tmp_path / "front" / "front_pod.json").exists()


def test_runner_callable_error_fails_task(tmp_path):
    R = _runner(tmp_path)

    def bad():
        raise RuntimeError("no inputs")
    R.add(fp.Task("x", bad))
    R.add(fp.Task("y", [sys.executable, "-c", "print(1)"], deps=["x"]))
    R.loop(0.1)
    assert R.tasks["x"].state == "failed" and "no inputs" in str(R.tasks["x"].rc)
    assert R.tasks["y"].state == "skipped"


def test_runner_respects_memory_budget(tmp_path):
    """Two 6 GB tasks under an 8 GB budget never run at the same time."""
    R = _runner(tmp_path, caps={"cpu": 5}, budget=8.0)
    for k in ("p", "q"):
        R.add(fp.Task(k, [sys.executable, "-c", "import time; time.sleep(0.6)"], mem_gb=6.0))
    R.loop(0.05)
    a, b = R.tasks["p"], R.tasks["q"]
    assert a.state == b.state == "ok"
    assert b.t0 >= a.t1 - 0.05 or a.t0 >= b.t1 - 0.05


def test_runner_reserves_memory_for_critical(tmp_path):
    """A small task may not take the memory a not-yet-ready critical task needs."""
    R = _runner(tmp_path, caps={"cpu": 5}, budget=10.0)
    R.add(fp.Task("gate", [sys.executable, "-c", "import time; time.sleep(0.4)"], mem_gb=1.0))
    R.add(fp.Task("big", [sys.executable, "-c", "print(1)"], deps=["gate"], mem_gb=8.0, critical=True))
    R.add(fp.Task("small", [sys.executable, "-c", "import time; time.sleep(0.2)"], mem_gb=3.0))
    R.loop(0.05)
    assert all(t.state == "ok" for t in R.tasks.values())
    assert R.tasks["small"].t0 >= R.tasks["big"].t0  # waited for the reservation to be released


def test_plan_chunks_cover_and_overlap():
    for T, n, ov in ((1151, 6, 36), (176, 3, 36), (100, 1, 30), (1000, 4, 30)):
        ch = mc.plan_chunks(T, n, ov)
        assert len(ch) == n
        assert ch[0][0] == 0 and ch[-1][1] == T
        lens = {b - a for a, b in ch}
        assert len(lens) == 1  # equal lengths
        for (a0, b0), (a1, b1) in zip(ch, ch[1:]):
            assert b0 - a1 >= ov  # neighbours share at least the overlap
        covered = np.zeros(T, bool)
        for a, b in ch:
            covered[a:b] = True
        assert covered.all()


def test_seam_weights_and_slerp():
    w = mc._seam_weights(36)
    assert w[0] == 0 and w[-1] == 1 and np.all(np.diff(w) >= 0)
    assert (w[:9] == 0).all() and (w[-9:] == 1).all()
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(55, 3)) * 0.5, rng.normal(size=(55, 3)) * 0.5
    np.testing.assert_allclose(mc._slerp_aa(a, b, np.zeros(55)), a, atol=1e-6)
    np.testing.assert_allclose(mc._slerp_aa(a, b, np.ones(55)), b, atol=1e-6)
    from scipy.spatial.transform import Rotation
    mid = Rotation.from_rotvec(mc._slerp_aa(a, b, np.full(55, 0.5)))
    ra, rb = Rotation.from_rotvec(a), Rotation.from_rotvec(b)
    np.testing.assert_allclose((ra.inv() * mid).magnitude(), (mid.inv() * rb).magnitude(), atol=1e-6)


@pytest.mark.parametrize("frames,expect", [(176, (0.40 + 0.0048 * 176) * 675 / 60)])
def test_solo_minutes(frames, expect):
    assert abs(mc.solo_minutes(frames) - expect) < 1e-9


def test_runner_resume_rc_files_done_if_and_requeue(tmp_path):
    """A new driver reads the rc files of workers that finished after the old driver died, pre-completes tasks whose
    outputs exist, re-queues workers that died with the driver, and keeps dependency order."""
    import json
    R = _runner(tmp_path)
    for k in ("fin", "outs", "lost", "dep"):
        R.add(fp.Task(k, [sys.executable, "-c", "print(1)"], deps=["lost"] if k == "dep" else [],
                      done_if=(lambda: True) if k == "outs" else None))
    (tmp_path / "front").mkdir(exist_ok=True)
    prev = {"tasks": {"fin": {"state": "running", "pid": 999999, "start_ts": 1.0},
                      "lost": {"state": "running", "pid": 999998, "start_ts": 1.0}}}
    (tmp_path / "front" / "front_pod.json").write_text(json.dumps(prev))
    R.rc_file(R.tasks["fin"]).write_text("0")
    R.resume()
    assert R.tasks["fin"].state == "ok" and R.tasks["outs"].state == "ok"
    assert R.tasks["lost"].state == "waiting" and R.tasks["lost"].restarts == 1
    R.loop(0.05)
    assert all(t.state == "ok" for t in R.tasks.values())
    assert R.tasks["dep"].t0 >= R.tasks["lost"].t1 - 0.05


def test_join_raw_reassembles_split_localisation():
    rng = np.random.default_rng(1)
    n = 9
    k = rng.integers(0, 5, n)
    off = np.r_[0, np.cumsum(k)]
    raw = {"t": np.arange(n, dtype=float), "ok": k > 0, "R": rng.normal(size=(n, 3, 3)), "tvec": rng.normal(size=(n, 3)),
           "f": rng.normal(size=n), "k1": rng.normal(size=n), "n_inliers": k, "reproj_px": rng.normal(size=n),
           "source": np.array(["loc"] * n), "corr_off": off, "corr_xy": rng.normal(size=(off[-1], 2)),
           "corr_xyz": rng.normal(size=(off[-1], 3))}

    def part(a, b):
        d = {key: raw[key][a:b] for key in ("t", "ok", "R", "tvec", "f", "k1", "n_inliers", "reproj_px", "source")}
        d["corr_off"] = off[a:b + 1] - off[a]
        d["corr_xy"], d["corr_xyz"] = raw["corr_xy"][off[a]:off[b]], raw["corr_xyz"][off[a]:off[b]]
        return d
    out = fp.join_raw([part(0, 4), part(4, n)])
    for key in raw:
        np.testing.assert_array_equal(out[key], raw[key])
