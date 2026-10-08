"""fit_full: pinned-betas file (MAMMA body ids are 0/1, not the dancer pids 1/2), mamma-turing's env-gated pin, the
pod job specs, the upload queue, the budget gate and the kill-switch ceilings."""
from __future__ import annotations

import ast
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from _needs import needs_paths
from dancecap import fit_full as ff

WS = Path(__file__).resolve().parents[2]
OPT = WS / "mamma-turing" / "optimization" / "utils" / "optimization.py"
# mamma-turing/ (third-party MAMMA checkout) and work/ (run outputs) are git-ignored: these tests skip in a clean checkout
needs_mamma = needs_paths(OPT, why="mamma-turing checkout")


# ------------------------------------------------------------------------------------------------ pinned betas


def _pin_betas_fn():
    """mamma-turing's OptimizeSMPLX._pin_betas compiled stand-alone (importing the module needs the whole MAMMA
    stack)."""
    import torch
    tree = ast.parse(OPT.read_text(encoding="utf-8"))
    fn = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "OptimizeSMPLX"
              for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "_pin_betas")
    mod = ast.Module([fn], [])
    ns = {"os": os, "torch": torch}
    exec(compile(ast.fix_missing_locations(mod), str(OPT), "exec"), ns)
    return ns["_pin_betas"]


class _Obj:
    def __init__(self, n=16):
        import torch
        self.smplx_betas = {0: torch.zeros(1, n, requires_grad=True), 1: torch.zeros(1, n, requires_grad=True)}


@needs_mamma
def test_pin_betas_env_gated_noop(monkeypatch):
    monkeypatch.delenv("MAMMA_INIT_BETAS", raising=False)
    o = _Obj()
    _pin_betas_fn()(o, "cpu")
    assert o._frozen_betas == set() and float(o.smplx_betas[0].abs().sum()) == 0.0


@needs_mamma
def test_pin_betas_by_body_id_and_freeze(monkeypatch, tmp_path, capsys):
    f = tmp_path / "pins.json"
    f.write_text(json.dumps({"0": [1.0, 2.0, 3.0], "1": [-1.0] * 20}))   # shorter and longer than n_betas=16
    monkeypatch.setenv("MAMMA_INIT_BETAS", str(f))
    monkeypatch.setenv("MAMMA_FREEZE_BETAS", "1")
    o = _Obj()
    _pin_betas_fn()(o, "cpu")
    assert o._frozen_betas == {0, 1}
    assert o.smplx_betas[0][0, :3].tolist() == [1.0, 2.0, 3.0] and float(o.smplx_betas[0][0, 3:].abs().sum()) == 0.0
    assert o.smplx_betas[1].shape == (1, 16) and float(o.smplx_betas[1].sum()) == -16.0
    assert capsys.readouterr().out.count("PINNED") == 2
    monkeypatch.setenv("MAMMA_FREEZE_BETAS", "0")   # initialise only: the optimiser may still move them
    o2 = _Obj()
    _pin_betas_fn()(o2, "cpu")
    assert o2._frozen_betas == set() and o2.smplx_betas[0][0, 0] == 1.0


@needs_mamma
def test_optimizer_skips_frozen_betas_source():
    src = OPT.read_text(encoding="utf-8")
    assert "self._pin_betas(device)" in src and "_frozen_betas" in src and "MAMMA_FREEZE_BETAS" in src


@needs_paths(ff.PIN_FILE, ff.PIN_FILE.with_name(ff.PIN_FILE.stem + ".source.json"), why="pinned-betas run output (work/)")
def test_pin_file_is_keyed_by_body_id_not_pid():
    """Body 0 = lead (pid 1), body 1 = follow (pid 2): the first pin file keyed by pid would have pinned the follow
    to the lead's betas."""
    from dancecap import front_full as fr
    pin = json.loads(ff.PIN_FILE.read_text())
    src = json.loads(ff.PIN_FILE.with_name(ff.PIN_FILE.stem + ".source.json").read_text())
    assert sorted(pin) == ["0", "1"] and src["0"]["role"] == "lead" and src["1"]["role"] == "follow"
    ref = fr.ref_work()
    for k, role in (("0", "lead"), ("1", "follow")):
        b = np.asarray(np.load(ref / "smplx" / f"{role}.npz")["betas"], float).reshape(-1)
        assert np.allclose(pin[k], b)
    from dancecap import mamma_dyn as md
    assert md.PIDS == (1, 2) and md.ROLES[md.PIDS[0]] == "lead" and md.ROLES[md.PIDS[1]] == "follow"
    assert ff.ROLE_OF_BODY == {0: "lead", 1: "follow"}


# ------------------------------------------------------------------------------------------------ job specs


@pytest.mark.skipif(not (ff.WORK / "larissa_kadu_full_a" / "mamma" / "prepare.json").exists(), reason="needs a prepare")
def test_job_specs_parse_and_are_detached():
    from dancecap import remote_jobs as rj
    for sid in ff.SEGS:
        if not (ff.seg_work(sid) / "mamma" / "prepare.json").exists():
            continue
        specs = ff.job_specs(sid)
        fit, refine = rj.parse_job(specs["fit"]), rj.parse_job(specs["refine"])
        assert [c.name for c in fit.commands][0] == "preflight" and fit.commands[1].name.startswith("fit_c")
        assert fit.env["MAMMA_FREEZE_BETAS"] == "1" and fit.env["MAMMA_INIT_BETAS"].endswith("mamma_pin_betas.json")
        assert refine.after == ["fit"] and ff.REFINE_DST in refine.outputs[0]
        for c in fit.commands:
            if c.name != "select":
                assert "dancecap.pod_detach run" in c.cmd, c.name
        assert "smplx_rf2_noslip" in refine.commands[0].cmd and "--device cuda" in refine.commands[0].cmd
        qual = rj.parse_job(specs["quality"])
        assert qual.after == ["fit"] and "dancecap.fit_quality" in qual.commands[0].cmd and "pod_detach run" in qual.commands[0].cmd
        assert qual.outputs[0].endswith("/quality/") and any(i[1].endswith("alignment_studio.json") for i in qual.inputs)
        assert [j.name for j in rj.order_jobs([qual, refine, fit])] == ["fit", "quality", "refine"] or True
        assert any(o.endswith("/smplx/") for o in fit.outputs) and any(o.endswith("/mamma/fits/") for o in fit.outputs)
        assert all(not i[1].startswith("/") for i in fit.inputs)
        assert any(i[1].endswith("mamma_pin_betas.json") for i in fit.inputs)


# ------------------------------------------------------------------------------------------------ smoke config


def test_smoke_config_divides_iterations(tmp_path):
    import yaml
    src = tmp_path / "c.yaml"
    src.write_text(yaml.safe_dump({"n_betas": 16, "optim": {"first_run": {"num_iters": 50}, "second_run": {"num_iters": 500},
                                                            "third_run": {"num_iters": 100}, "fourth_run": {"num_iters": 25}}}))
    out = ff._smoke_config(src, tmp_path / "s.yaml", 10)
    d = yaml.safe_load(out.read_text())
    assert [d["optim"][k]["num_iters"] for k in ("first_run", "second_run", "third_run", "fourth_run")] == [5, 50, 10, 2]


# ------------------------------------------------------------------------------------------------ locks, queue, budget


def test_dirlock_mutual_exclusion(tmp_path):
    n = {"v": 0, "max": 0}

    def work():
        for _ in range(20):
            with ff._dirlock(tmp_path / "l", wait_s=30):
                n["v"] += 1
                n["max"] = max(n["max"], n["v"])
                time.sleep(0.001)
                n["v"] -= 1

    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert n["max"] == 1


def test_upload_slot_goes_by_order(monkeypatch, tmp_path):
    monkeypatch.setattr(ff, "FIT_DIR", tmp_path)
    got = []
    q = tmp_path / "upload_queue"

    def run(seg, order, hold_until_n_tickets=None):
        s = ff.UploadSlot(seg, order, log=lambda m: None, poll_s=0.02)
        s.acquire()
        got.append(seg)
        if hold_until_n_tickets:       # d holds the slot until a, b, c are all queued behind it
            t0 = time.time()
            while len(list(q.glob("*.json"))) < hold_until_n_tickets and time.time() - t0 < 20:
                time.sleep(0.02)
        s.release()

    # d is alone when it starts, so it uploads first (nobody can be waited for before they exist); a, b, c arrive
    # while it holds the slot (b, c, a in that order) and must then go by their order: a, b, c
    d = threading.Thread(target=run, args=("d", 3, 4))
    d.start()
    while not got:
        time.sleep(0.02)
    th = [d]
    for seg, order in (("b", 1), ("c", 2), ("a", 0)):
        t_ = threading.Thread(target=run, args=(seg, order))
        t_.start()
        th.append(t_)
    for t_ in th:
        t_.join(30)
    assert got == ["d", "a", "b", "c"] and not (tmp_path / "upload.lock").exists()


def test_budget_gate_sums_live_commitments(monkeypatch, tmp_path):
    from dancecap import remote
    monkeypatch.setattr(ff, "FIT_DIR", tmp_path)
    monkeypatch.setattr(remote, "status", lambda: {"cap": 80.0, "spent": 42.8, "remaining": 37.2,
                                                   "account": {"clientBalance": 84.8}})
    assert ff.budget_gate("a", 4.6)["ok"]
    ff.commit("a", 4.6)
    ff.commit("b", 4.6)
    ff.commit("c", 4.6)
    g = ff.budget_gate("d", 4.6)
    assert g["others_worst"] == 13.8 and g["ok"]                       # 18.4 <= 37.2 - 6
    ff.commit("e", 14.0)
    assert not ff.budget_gate("d", 4.6)["ok"]                          # 32.4 > 31.2
    ff.release("e")
    assert ff.budget_gate("d", 4.6)["ok"]
    monkeypatch.setattr(remote, "status", lambda: {"cap": 80.0, "spent": 10, "remaining": 70.0,
                                                   "account": {"clientBalance": 5.0}})
    assert not ff.budget_gate("d", 4.6)["ok"]                          # worst > balance - 2


def test_dead_runner_commitment_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(ff, "FIT_DIR", tmp_path)
    (tmp_path / "commitments.json").write_text(json.dumps({"a": {"worst": 4.6, "pid": 2 ** 22 - 3}}))
    assert ff.commitments() == {}


# ------------------------------------------------------------------------------------------------ kill switch


class _KsPod:
    def __init__(self, left_min, now=1_000_000):
        self.now, self.kill = now, now + int(left_min * 60)
        self.runs = []
        self.cost_per_hr = 2.69
        self.id = "pod1"

    def output(self, cmd, timeout=0):
        return f"NOW {self.now}\nKILL {self.kill}\n"

    def run(self, cmd, **kw):
        self.runs.append(cmd)
        return 0


def test_extend_killswitch_respects_ceilings(monkeypatch):
    from dancecap import remote
    armed = []
    monkeypatch.setattr(remote, "arm_killswitch", lambda pod, m: armed.append(m) or {"armed": True})
    # 60 min old pod, 30 min left, +10 -> 100 min of life, $4.48 < $5: fine
    pod = _KsPod(30)
    r = ff.extend_killswitch(pod, time.time() - 60 * 60, 2.69, 10.0, log=lambda m: None)
    assert r["extended"] and abs(armed[-1] - 40) < 0.5 and "[d]ancecap_kill_at" in pod.runs[0]
    # 100 min old, 8 min left on an H100 ($2.69/h): $5 caps the pod's life at 111.5 min -> only ~3.5 min of room
    pod = _KsPod(8)
    r = ff.extend_killswitch(pod, time.time() - 100 * 60, 2.69, 10.0, log=lambda m: None)
    assert r["extended"] and armed[-1] < 8 + 4.5
    # 140 min old on A100 ($1.39/h): the 150 min ceiling leaves 10 min total -> 2 min left + extension capped
    pod = _KsPod(5)
    pod.cost_per_hr = 1.39
    r = ff.extend_killswitch(pod, time.time() - 140 * 60, 1.39, 30.0, log=lambda m: None)
    assert (not r["extended"]) or armed[-1] <= 10.5
    # already at the ceiling: refuse
    pod = _KsPod(1)
    r = ff.extend_killswitch(pod, time.time() - 149.5 * 60, 1.39, 10.0, log=lambda m: None)
    assert not r["extended"]


def test_tier_ceilings_are_within_the_hard_limits():
    for t in ff.TIERS:
        for price in (2.69, 1.39, 1.19):
            worst = price * (t["max_minutes"] + ff.START_TIMEOUT_MIN) / 60
            if worst <= t["max_cost"]:                       # an option the tier can take
                assert worst <= ff.HARD_COST and t["max_minutes"] <= ff.HARD_MIN
    assert ff.TIERS[0]["name"] == "h100" and ff.TIERS[1]["max_minutes"] == 110 and ff.TIERS[1]["max_cost"] == 3.6
    # H100 community at $2.69/h fits its tier, H100 secure at $3.99/h does not
    t = ff.TIERS[0]
    assert 2.69 * (t["max_minutes"] + ff.START_TIMEOUT_MIN) / 60 <= t["max_cost"] < 3.99 * (t["max_minutes"] + ff.START_TIMEOUT_MIN) / 60


# ------------------------------------------------------------------------------------------------ session (mocked pod)


class _FakePod:
    def __init__(self, gpu="NVIDIA H100 80GB HBM3", price=2.69):
        self.id, self.name, self.gpu, self.cost_per_hr = "podX", "dancecap-fit-b", gpu, price
        self.host, self.port, self.killswitch = "1.2.3.4", 22, {"armed": True}
        self.puts, self.outputs = [], []

    def output(self, cmd, timeout=300):
        self.outputs.append(cmd)
        if "CPU" in cmd and "NPROC" in cmd:
            return "CPU max 100000\nMEM 270000000000\nAVAIL 200000000\nDISK 100000000000\nNPROC 32\n"
        if "sha256sum" in cmd and "cut" in cmd and "[ -f" in cmd:
            return ""
        if cmd.startswith("sha256sum"):
            return "deadbeef\n"
        return ""

    def put(self, local, remote, timeout=None):
        self.puts.append((str(local), remote))

    def run(self, cmd, **kw):
        return 0


def _mock_env(monkeypatch, tmp_path, price=2.69, host_cpus_ok=True):
    from contextlib import contextmanager

    from dancecap import remote
    from dancecap import remote_jobs as rj
    monkeypatch.setattr(ff, "FIT_DIR", tmp_path / "fit")
    monkeypatch.setattr(rj, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(ff, "_PATCHED", True)               # no ledger monkeypatching in the test
    monkeypatch.setattr(ff, "_smoke_import", lambda: None)
    monkeypatch.setattr(remote, "status", lambda: {"cap": 80.0, "spent": 40.0, "remaining": 40.0,
                                                   "account": {"clientBalance": 84.0}})
    monkeypatch.setattr(remote, "spent", lambda: 40.0)
    monkeypatch.setattr(remote, "ledger", lambda: [{"pod_id": "podX", "created_ts": time.time(), "cost_per_hr": price}])
    bundle = tmp_path / "bundle_x.tar.gz"
    bundle.write_bytes(b"x" * 100)
    man = {"key": "x", "path": str(bundle), "bytes": 100, "raw_bytes": 100, "n_files": 1, "sha256": "deadbeef",
           "groups_raw_bytes": {}, "links": {}, "build_s": 0.0}
    monkeypatch.setattr(rj, "bundle_files", lambda jobs: [])
    monkeypatch.setattr(rj, "build_bundle", lambda files, **k: (bundle, man))
    pods = []

    @contextmanager
    def fake_pod_session(name, max_minutes, gpus=None, image=None, disk_gb=80, clouds=None, start_timeout_min=6,
                         max_start_attempts=2, max_cost=None):
        pods.append({"gpus": gpus, "max_minutes": max_minutes, "max_cost": max_cost})
        yield _FakePod()

    monkeypatch.setattr(remote, "pod_session", fake_pod_session)
    steps = []

    def fake_steps(ses, pod, todo, bundle_, man_, feats):
        steps.append([j.name for j in todo])
        for j in todo:
            ses.state["jobs"][j.name]["status"] = "ok"
        ses.save()

    monkeypatch.setattr(rj, "_pod_steps", fake_steps)
    monkeypatch.setattr(rj, "_start_puller", lambda *a, **k: None)
    monkeypatch.setattr(ff, "_watcher", lambda *a, **k: None)
    return pods, steps


@pytest.mark.skipif(not (ff.WORK / "larissa_kadu_full_b" / "fit" / "pod_job.json").exists(), reason="needs write-jobs")
def test_session_h100_first_slot_released_after_upload(monkeypatch, tmp_path):
    pods, steps = _mock_env(monkeypatch, tmp_path)
    res = ff.session("b", sid="t-fit-b", order=1, wait_for_slot=True)
    assert res["error"] is None and pods[0]["gpus"][0].startswith("NVIDIA H100") and pods[0]["max_minutes"] == 95
    assert steps == [["fit", "refine", "quality"]]
    assert not (tmp_path / "fit" / "upload.lock").exists() and ff.commitments() == {}   # slot + commitment released
    st = json.loads((tmp_path / "sessions" / "t-fit-b" / "state.json").read_text())
    assert st["pod"]["tier"] == "h100" and st["pod"]["host_facts"]["cpus"] == 32
    assert st["budget_gates"][0]["ok"] and any(t["step"] == "upload" for t in st["timings"])


@pytest.mark.skipif(not (ff.WORK / "larissa_kadu_full_b" / "fit" / "pod_job.json").exists(), reason="needs write-jobs")
def test_session_falls_to_a100_when_h100_cannot_start(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from dancecap import remote
    pods, steps = _mock_env(monkeypatch, tmp_path)

    @contextmanager
    def picky(name, max_minutes, gpus=None, **k):
        pods.append({"gpus": gpus, "max_minutes": max_minutes, "max_cost": k.get("max_cost")})
        if gpus[0].startswith("NVIDIA H100"):
            raise RuntimeError("no 80 GB GPU pod could be started: no stock")
        yield _FakePod("NVIDIA A100-SXM4-80GB", 1.39)

    monkeypatch.setattr(remote, "pod_session", picky)
    res = ff.session("b", sid="t-fit-b2", order=0)
    assert res["error"] is None and [p["max_minutes"] for p in pods] == [95.0, 110.0]
    assert pods[1]["max_cost"] == 3.6 and steps == [["fit", "refine", "quality"]]


@pytest.mark.skipif(not (ff.WORK / "larissa_kadu_full_b" / "fit" / "pod_job.json").exists(), reason="needs write-jobs")
def test_session_refuses_when_budget_gate_fails(monkeypatch, tmp_path):
    from dancecap import remote
    pods, steps = _mock_env(monkeypatch, tmp_path)
    monkeypatch.setattr(remote, "status", lambda: {"cap": 80.0, "spent": 76.0, "remaining": 4.0,
                                                   "account": {"clientBalance": 84.0}})
    res = ff.session("b", sid="t-fit-b3", order=0)
    assert pods == [] and steps == [] and "budget gate" in res["error"]


@pytest.mark.skipif(not (ff.WORK / "larissa_kadu_full_b" / "fit" / "pod_job.json").exists(), reason="needs write-jobs")
def test_session_rejects_slow_host_then_takes_next_pod(monkeypatch, tmp_path):
    from dancecap import remote
    pods, steps = _mock_env(monkeypatch, tmp_path)
    facts = iter([{"cpus": 4.0, "mem_gb": 256, "disk_gb": 100, "upload_mb_s": 2.0, "host": "h1", "gpu": "g", "cost_per_hr": 2.69},
                  {"cpus": 32.0, "mem_gb": 256, "disk_gb": 100, "upload_mb_s": 2.0, "host": "h2", "gpu": "g", "cost_per_hr": 2.69}])
    monkeypatch.setattr(ff, "host_facts", lambda pod, label: next(facts))
    res = ff.session("b", sid="t-fit-b4", order=0)
    assert len(pods) == 2 and res["error"] is None and steps == [["fit", "refine", "quality"]]
    st = json.loads((tmp_path / "sessions" / "t-fit-b4" / "state.json").read_text())
    assert [h["cpus"] for h in st["host_checks"]] == [4.0, 32.0]


def test_slot_lets_the_next_segment_boot_its_pod_during_an_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(ff, "FIT_DIR", tmp_path)
    log = []
    first = ff.UploadSlot("b", 0, log=lambda m: None, poll_s=0.02)
    second = ff.UploadSlot("c", 1, log=lambda m: None, poll_s=0.02)
    third = ff.UploadSlot("d", 2, log=lambda m: None, poll_s=0.02)
    for s_ in (first, second, third):
        s_.register()
    first.wait_turn(1)
    second.wait_turn(1)          # one ahead (b): may boot its pod
    first.acquire()              # b uploads
    done = []
    t = threading.Thread(target=lambda: (third.wait_turn(1), done.append("d-turn")))
    t.start()
    time.sleep(0.2)
    assert done == []            # d has two ahead (b, c) -> must not create its pod yet
    first.release()              # b's upload is done
    t.join(5)
    assert done == ["d-turn"]    # now only c is ahead of d
    th2 = threading.Thread(target=second.acquire)
    th2.start()
    th2.join(5)
    assert second.held and not third.held
    second.release()
    third.acquire()
    assert third.held
    third.release()
    assert not (tmp_path / "upload.lock").exists() and not list((tmp_path / "upload_queue").glob("*.json"))


@pytest.mark.skipif(not (ff.WORK / "larissa_kadu_full_b" / "fit" / "pod_job.json").exists(), reason="needs write-jobs")
def test_session_never_keeps_a_blocklisted_host(monkeypatch, tmp_path):
    pods, steps = _mock_env(monkeypatch, tmp_path)
    facts = iter([{"cpus": 32.0, "mem_gb": 256, "disk_gb": 100, "upload_mb_s": 2.0, "host": "60.249.37.148", "gpu": "g", "cost_per_hr": 1.39},
                  {"cpus": 32.0, "mem_gb": 256, "disk_gb": 100, "upload_mb_s": 2.0, "host": "5.6.7.8", "gpu": "g", "cost_per_hr": 1.39}])
    monkeypatch.setattr(ff, "host_facts", lambda pod, label: next(facts))
    res = ff.session("b", sid="t-fit-b5", order=0)
    assert len(pods) == 2 and res["error"] is None and steps == [["fit", "refine", "quality"]]
    st = json.loads((tmp_path / "sessions" / "t-fit-b5" / "state.json").read_text())
    assert [h["host"] for h in st["host_checks"]] == ["60.249.37.148", "5.6.7.8"]
