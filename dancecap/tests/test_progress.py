"""dancecap.progress/1: helper, adapters, L0 parser, ETA, budget, puller, runner hooks (no pods, no media)."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dancecap import progress as P
from dancecap import progress_eta as E
from dancecap import progress_pull as PP

SESS = Path(__file__).resolve().parents[1] / "work" / "runpod" / "sessions"


def _events(path: Path) -> list[dict]:
    out = []
    for f in sorted(path.parent.glob(path.stem + "*.jsonl")) if not path.exists() else [path]:
        evs, _ = P.read_jsonl(f)
        out += evs
    return out


@pytest.fixture
def prog(tmp_path, monkeypatch):
    f = tmp_path / "progress.jsonl"
    monkeypatch.setenv("DANCECAP_PROGRESS", str(f))
    monkeypatch.setenv("DANCECAP_PROGRESS_JOB", "avatar")
    monkeypatch.setenv("DANCECAP_PROGRESS_CMD", "bake")
    P._writers.clear()
    yield f
    P._writers.clear()


# ---------------------------------------------------------------------------------------------------- helper


def test_disabled_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("DANCECAP_PROGRESS", raising=False)
    P._writers.clear()
    assert not P.enabled()
    with P.step("x", total=10) as s:
        s.update(5)
    P.tick("y", 1, 2)
    P.note("hello")
    P.plan([("a", 1, "views")])
    assert P.watch_log(tmp_path / "nope.log", "tqdm")() is None
    assert list(tmp_path.iterdir()) == []


def test_step_events_throttle_and_fail(prog, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(P.time, "time", lambda: clock[0])
    P.plan([("bake/pass0", 100, "views"), ("bake/pass1", 100, "views", 0.6)])
    with P.step("bake/pass0", total=100, unit="views") as s:
        for i in range(100):
            clock[0] += 0.1  # 10 s for the pass: ticks at most every 2 s
            s.update(i + 1)
    with pytest.raises(ValueError):
        with P.step("bake/pass1", total=100, unit="views") as s:
            s.update(3)
            raise ValueError("boom")
    evs = _events(prog)
    kinds = [e["kind"] for e in evs]
    assert kinds[0] == "plan" and evs[0]["steps"][1]["weight"] == 0.6
    ticks = [e for e in evs if e["kind"] == "tick"]
    assert 3 <= len(ticks) <= 6  # 10 s / 2 s
    assert all(b["ts"] - a["ts"] >= 2.0 - 1e-9 for a, b in zip(ticks, ticks[1:]))
    end = next(e for e in evs if e["kind"] == "end")
    assert end["done"] == 100 and end["job"] == "avatar" and end["cmd"] == "bake" and end["extra"]["seconds"] > 0
    fail = next(e for e in evs if e["kind"] == "fail")
    assert fail["step"] == "bake/pass1" and fail["rc"] == "ValueError" and "boom" in fail["msg"]
    assert all(e["v"] == 1 and len(json.dumps(e)) <= P.MAX_PLAN_LINE for e in evs)


def test_one_line_hooks_and_total_revision(prog, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(P.time, "time", lambda: clock[0])
    for i in range(50):
        clock[0] += 1
        P.tick("run2", i + 1, 500, "iters", lane="c8_f0-232")
    P.tick("run2", 51, 270, lane="c8_f0-232")  # early stop: revised total is always written
    P.end("run2", done=270, lane="c8_f0-232", msg="Stopping early at iteration 270")
    evs = _events(prog)
    assert evs[0]["kind"] == "start" and evs[0]["lane"] == "c8_f0-232"
    rev = [e for e in evs if e["kind"] == "tick" and e["total"] == 270]
    assert rev and rev[0]["done"] == 51
    assert evs[-1]["kind"] == "end" and evs[-1]["done"] == 270


def test_redaction_and_size_cap(prog):
    key = "rpa_" + "A1b2C3d4" * 6
    P.note(f"token {key} and sha 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 ok")
    P.note("x" * 5000, step="big", blob="y" * 5000)
    evs = _events(prog)
    assert key not in evs[0]["msg"] and "[redacted]" in evs[0]["msg"]
    assert "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08" in evs[0]["msg"]
    assert len(evs[0]["msg"]) <= P.MSG_MAX
    big = [e for e in evs if e["step"] == "big"]
    assert not big or len(json.dumps(big[0])) <= P.MAX_LINE


def test_windows_writes_one_file_per_process(tmp_path):
    p = P._resolve_path(str(tmp_path / "ingest.jsonl"), nt=True)
    assert p.name == f"ingest.{os.getpid()}.jsonl"
    assert P._resolve_path(str(p), nt=True) == p
    assert P._resolve_path(str(tmp_path / "x.jsonl"), nt=False).name == "x.jsonl"


def test_read_jsonl_offsets_hold_back_a_partial_line(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_bytes(b'{"v":1,"kind":"tick","done":1}\n{"v":1,"kind":"ti')
    evs, off = P.read_jsonl(f)
    assert [e["done"] for e in evs] == [1] and off == len(b'{"v":1,"kind":"tick","done":1}\n')
    with open(f, "ab") as fh:
        fh.write(b'ck","done":2}\nnot json\n')
    evs, off2 = P.read_jsonl(f, off)
    assert [e["done"] for e in evs] == [2] and off2 == f.stat().st_size


# ---------------------------------------------------------------------------------------------------- adapters

RUNNER_FIXTURE = """\
[pod-runner 06:57:44] bundle_scan ...
[pod-runner 06:57:47] bundle_scan done in 3.3s
[pod-runner 07:00:05] bundle bundle_265a8da5f063.tar.gz: 1505.6 MB (417 files, raw 1574.3 MB, 8 hard links); upload ~10.0 min; features ['base', 'mamma', 'texture', 'check']; pod max 245 min
[pod-runner 07:00:40] upload ...
[pod-runner 07:09:38] upload done in 537.5s
[pod-runner 07:09:58] bootstrap ...
[bootstrap] base: start
[bootstrap] base: done in 174s
[bootstrap] room: FAILED (rc=1) after 26s
[bootstrap] mamma: already done
[bootstrap] finished rc=0 (MAX_JOBS=28); wheels in /x
[pod-runner 07:15:16] bootstrap done in 318.3s
[pod-runner 10:08:57] avatar/bake ...
[pod-runner] 2026-10-07T17:08:55Z start avatar/bake
[texture] preset pod: 2292 views from 8 cameras over t=30.15..68.38 s, atlas 2048^2 with 3077723 texels
[texture] 50/2292 views
[texture] 2250/2292 views
[texture] pass 0: 145 s
[texture] 50/2292 views
[texture] 1450/2292 views
[texture] pass 1 (aligned): 315 s, flow median 3.5 px
[texture] -> /workspace/HeadMovement/dancecap/work/larissa_kadu_38/avatar (1076 s)
[pod-runner] 2026-10-07T17:26:54Z end avatar/bake rc=0
[pod-runner 10:26:57] avatar/bake done in 1080.8s
[pod-runner 10:27:04] avatar: ok (1097s)
[pod-runner 10:27:05] mamma: skipped: bootstrap feature failed ['mamma']
"""


def test_runner_log_adapter_fixture():
    p = P.LineParser(date0=(2026, 10, 7))
    evs = [e for ln in RUNNER_FIXTURE.splitlines() for e in p.feed(ln)]
    starts = [e for e in evs if e["src"] == "runner" and e["kind"] == "start"]
    assert [e["step"] for e in starts] == ["bundle_scan", "upload", "bootstrap", "bake"]
    assert starts[0]["ts"] == time.mktime((2026, 10, 7, 6, 57, 44, 0, 0, -1))
    up = next(e for e in evs if e["src"] == "runner" and e["kind"] == "end" and e["step"] == "upload")
    assert up["extra"]["seconds"] == 537.5
    note = next(e for e in evs if e.get("step") == "bundle")
    assert note["extra"]["features"] == ["base", "mamma", "texture", "check"] and note["extra"]["pod_max_min"] == 245
    boot = [(e["kind"], e["step"]) for e in evs if e["src"] == "bootstrap" and e["kind"] != "note"]
    assert boot == [("start", "bootstrap/base"), ("end", "bootstrap/base"), ("fail", "bootstrap/room"),
                    ("end", "bootstrap/mamma")]
    pod = [e for e in evs if e["src"] == "pod-runner"]
    assert pod[0]["ts"] == 1791392935 and pod[1]["kind"] == "end" and pod[1]["rc"] == 0
    tex = [e for e in evs if e["src"] == "texture"]
    assert tex[0]["kind"] == "plan" and tex[0]["extra"]["passes"] == 3 and tex[0]["job"] == "avatar"
    assert [e["step"] for e in tex if e["kind"] == "tick"] == ["bake/pass0", "bake/pass0", "bake/pass1", "bake/pass1"]
    assert all(e["cmd"] == "bake" for e in tex)
    ends = [e for e in tex if e["kind"] == "end"]
    assert [e["extra"].get("seconds") for e in ends[:2]] == [145, 315] and ends[-1]["extra"]["seconds_total"] == 1076
    jobs = [e for e in evs if e.get("step") == "_job"]
    assert jobs[0]["kind"] == "end" and jobs[1]["kind"] == "note"


def test_midnight_rollover():
    p = P.LineParser(date0=(2026, 10, 6))
    a = p.feed("[pod-runner 23:59:50] upload ...")[0]["ts"]
    b = p.feed("[pod-runner 00:00:10] upload done in 20.0s")[0]["ts"]
    assert b - a == pytest.approx(20, abs=1)


def test_stage_adapters():
    lines = {
        "room": ["[room] 456 views (414 train, 42 held out) from cams 01,02 at <= 1920 px (preset pod); floor",
                 "[room] step     0  L1 0.1460  masked PSNR 14.23  N 14,787      1s  refl floor 0.15/0.12",
                 "[room] step 29999  L1 0.0322  masked PSNR 26.20  N 1,000,000   1939s  refl floor 0.14/0.10"],
        "hands": ["[hand_worker] hands 500/5779 40s", "[mano_worker] wilor 725/5779 34s"],
        "front": ["[front] larissa_kadu_38: 32 tasks; cpus 28, RAM limit 116 GB",
                  "[front] 08:53:52 start sam:02", "[front] 09:08:14 sam:02: ok rc=0 in 862s (rss peak 34.6 GB)",
                  "[front] sam:01: done earlier (outputs present)"],
        "loc": ["[cameras] 02: 60/1151 frames, 60 ok (41s)"],
        "chunks": ["[chunks] 6 chunks of c8 + held-out h03_f736-968, 7 in parallel, CUDA MPS on",
                   "[chunks] c8_f368-600: rc=0 in 2261s", "[chunks] follow: repaired 11 outlier frame(s) [20]",
                   "[chunks] gate FAILED: follow: seam p95 83 mm > 60",
                   "[chunks] fallback: one full fit of c8 (1148 frames, ~66 min); the rejected merge is kept in /x"],
        "detect": ["[detect] 09: 1151 frames @640+960, 12.2 people/frame, pair 86.2%, 85 ms/frame, kf 0 in 0s"],
    }
    p = P.LineParser(P.ADAPTERS, date0=(2026, 10, 7))
    got = {k: [e for ln in v for e in p.feed(ln)] for k, v in lines.items()}
    room = got["room"]
    assert room[0]["kind"] == "plan" and room[0]["steps"][0]["total"] == 30000
    assert room[-1]["kind"] == "end" and room[-1]["extra"]["gaussians"] == 1_000_000
    assert room[-1]["extra"]["elapsed_s"] == 1939
    h = got["hands"]
    assert [(e["step"], e["done"], e["total"]) for e in h if e["kind"] == "tick"] == [
        ("hand_worker/hands", 500, 5779), ("mano_worker/wilor", 725, 5779)]
    f = got["front"]
    assert f[0]["kind"] == "plan" and f[0]["extra"]["tasks"] == 32
    assert f[1]["kind"] == "start" and f[1]["lane"] == "sam:02" and f[1]["clock"] == "pod"
    assert f[2]["extra"]["seconds"] == 862 and f[2]["ts"] - f[1]["ts"] == 862
    assert f[3]["extra"]["earlier"] is True
    assert got["loc"][0]["lane"] == "02" and got["loc"][0]["done"] == 60
    ch = got["chunks"]
    assert ch[0]["extra"]["lanes"] == 7 and ch[1]["lane"] == "c8_f368-600" and ch[1]["extra"]["frames"] == 232
    assert len(ch) == 4 and ch[2]["extra"]["gate"] is False and ch[3]["eta_s"] == 3960
    assert got["detect"][0]["extra"]["ms_per_frame"] == 85


def test_mamma_iter_and_tqdm():
    p = P.LineParser(("mamma_iter",), step="fit_chunks", lane="c8_f0-232")
    seq = ["Iter. 1 of 50 | Loss", "Iter. 16 of 50 | Loss", "Stopping early at iteration 16 due to lack",
           "Iter. 1 of 500 | x", "Iter. 270 of 500", "Stopping early at iteration 270", "Iter. 1 of 100",
           "Iter. 100 of 100", "Iter. 1 of 25", "Stopping early at iteration 17", "Optimization took 2191.15 seconds."]
    evs = [e for ln in seq for e in p.feed(ln)]
    ends = [(e["step"], e.get("done"), e.get("total")) for e in evs if e["kind"] == "end"]
    assert ends == [("fit_chunks/run1", 16, 16), ("fit_chunks/run2", 270, 270), ("fit_chunks/run3", 100, 100),
                    ("fit_chunks/run4", 17, 17), ("fit_chunks/fit", None, None)]
    assert all(e["lane"] == "c8_f0-232" for e in evs)
    # a tail that starts mid-log finds its run from the distinct totals
    q = P.LineParser(("mamma_iter",))
    e = q.feed("Iter. 77 of 100 | Loss")
    assert e[0]["step"] == "run3" and e[-1]["done"] == 77
    t = P.LineParser(("tqdm",), step="front/ma2d", lane="02")
    evs = t.feed(" 50%|#####     | 1151/2302 [04:36<04:36,  4.16it/s]\r 50%|#####     | 1152/2302 [04:37<04:35,  4.1it/s]")
    assert [e["done"] for e in evs] == [1151, 1152] and evs[-1]["eta_s"] == 275 and evs[-1]["lane"] == "02"


def test_watch_log_relays_a_growing_log(prog, tmp_path):
    log = tmp_path / "ma_3d_c8_f0-232.log"
    log.write_text("old run\n")
    stop = P.watch_log(log, "mamma_iter", step="fit_chunks", lane="c8_f0-232", interval=0.05, from_start=False)
    time.sleep(0.2)
    with open(log, "w") as fh:  # mamma_dyn re-opens its log with "w"
        fh.write("Iter. 1 of 50 | x\n")
    time.sleep(0.2)
    with open(log, "a") as fh:
        fh.write("Iter. 2 of 50 | x\nStopping early at iteration 16 due to\n")
    time.sleep(0.2)
    stop()
    evs = _events(prog)
    assert any(e["kind"] == "start" and e["step"] == "fit_chunks/run1" for e in evs)
    end = next(e for e in evs if e["kind"] == "end")
    assert end["done"] == 16 and end["lane"] == "c8_f0-232"


@pytest.mark.skipif(not (SESS / "20261007-065744-batch" / "state.json").exists(), reason="local session data")
def test_l0_replay_matches_state_timings():
    d = SESS / "20261007-065744-batch"
    st = json.loads((d / "state.json").read_text())
    evs, _, _ = P.parse_runner_log(d / "runner.log")
    starts = [e for e in evs if e["src"] == "runner" and e["kind"] == "start"]
    ends = [e for e in evs if e["src"] == "runner" and e["kind"] in ("end", "fail")
            and (e.get("extra") or {}).get("runner_step")]
    assert len(ends) == len(st["timings"])
    for e, t in zip(ends, st["timings"]):
        assert e["extra"]["runner_step"] == t["step"]
        assert e["extra"]["seconds"] == pytest.approx(t["seconds"], abs=0.11)
    for e, t in zip(starts, st["timings"]):
        assert abs(e["ts"] - t["start"]) <= 2.0, (t["step"], e["ts"], t["start"])


# ---------------------------------------------------------------------------------------------------- ETA


def test_linear_eta_converges():
    rng = __import__("random").Random(1)
    N, s_true = 2292, 0.132
    obs, t, d = [], 0.0, 0
    while d < 0.2 * N:
        d += 50
        t += 50 * s_true * (1 + 0.1 * (rng.random() - 0.5))
        obs.append((t, d))
    r = E.remaining_linear(N, d, obs, hist_s=0.114, n_hist=3, sigma_hist=0.15, now=t)
    truth = (N - d) * s_true
    assert abs(r[1] - truth) / truth < 0.10
    assert r[0] < r[1] < r[2]


def test_texture_example_section_6_8():
    obs = [(i * 50 * 0.132, i * 50) for i in range(1, 30)]  # pass 2 at 1450/2292 views, live 0.132 s/view
    r = E.remaining_linear(2292, 1450, obs, hist_s=0.114, n_hist=3, sigma_hist=0.15, now=obs[-1][0])
    assert r[1] == pytest.approx(111, rel=0.06)


def test_profile_eta_room():
    T = 1944.0
    # a run 20 % slower than history, at 40 % of the steps
    x = 0.4
    el = 1.2 * T * E._interp_profile(E.ROOM_PROFILE, x)
    r = E.remaining_profile(T, E.ROOM_PROFILE, x, el, n_hist=1)
    truth = 1.2 * T * (1 - E._interp_profile(E.ROOM_PROFILE, x))
    assert abs(r[1] - truth) / truth < 0.05


def test_early_stop_lanes_branches_and_add():
    assert E.expected_total(500, 10, 0.54) == 270
    assert E.expected_total(500, 300, 0.54) == 301
    assert E.lanes_max([(1, 2, 3), (2, 5, 9), (0, 1, 2)]) == (2, 5, 9)
    s = E.add((8, 10, 12), (8, 10, 12))
    assert s[1] == 20 and s[2] == pytest.approx(20 + math.sqrt(8))
    assert E.add_p90((8, 10, 12), (8, 10, 12)) == 24
    r = E.remaining_after_elapsed((60, 100, 200), 150)
    assert r[1] >= 15 and r[2] > r[1]


def test_dag_simulation_caps_and_deps():
    tasks = {"sam:a": {"kind": "sam", "state": "pending", "duration": 100},
             "sam:b": {"kind": "sam", "state": "pending", "duration": 100},
             "ma2d:a": {"kind": "ma2d", "deps": ["sam:a"], "state": "pending", "duration": 50},
             "sync": {"kind": "cpu", "deps": [], "after": ["ma2d:a"], "state": "pending", "duration": 10}}
    assert E.simulate_dag(tasks, {"sam": 2}) == pytest.approx(160)
    assert E.simulate_dag(tasks, {"sam": 1}) == pytest.approx(200)  # critical path first: sam:a, then ma2d:a || sam:b
    tasks["sam:a"]["state"] = "running"
    tasks["sam:a"]["remaining"] = 20
    assert E.simulate_dag(tasks, {"sam": 1}) == pytest.approx(120)


FRONT_POD = Path(__file__).resolve().parents[1] / "work" / "larissa_kadu_38" / "front" / "front_pod.json"


@pytest.mark.skipif(not FRONT_POD.exists(), reason="local front data")
def test_front_dag_replay_within_15_percent():
    doc = json.loads(FRONT_POD.read_text())
    tasks = {k: {"kind": v["kind"], "deps": v["deps"], "after": v["after"], "state": "pending",
                 "duration": v["seconds"] or 0.0} for k, v in doc["tasks"].items()
             if v.get("seconds")}  # tasks done in an earlier session are not part of this run
    for v in tasks.values():
        v["deps"] = [d for d in v["deps"] if d in tasks]
        v["after"] = [d for d in v["after"] if d in tasks]
    caps = {"sam": 5, "loc": 5, "ma2d": 2, "cpu": 2}
    span = E.simulate_dag(tasks, caps)
    measured = max(v["start_ts"] + v["seconds"] for v in doc["tasks"].values() if v.get("seconds")) - \
        min(v["start_ts"] for v in doc["tasks"].values() if v.get("seconds"))
    assert abs(span - measured) / measured < 0.15, (span, measured)


def test_mamma_model_constants_match_mamma_chunks():
    mc = pytest.importorskip("dancecap.mamma_chunks")
    assert (E.MAMMA_ITER_FIXED_S, E.MAMMA_ITER_PER_FRAME_S, E.MAMMA_N_ITERS) == (
        mc.ITER_FIXED_S, mc.ITER_PER_FRAME_S, mc.N_ITERS)
    assert E.MAMMA_PARALLEL_SLOWDOWN == mc.PARALLEL_SLOWDOWN
    assert E.mamma_solo_s(1148) / 60 == pytest.approx(mc.solo_minutes(1148))


def _mk_session(root: Path, sid: str, lines: list[str], state: dict) -> Path:
    d = root / sid
    d.mkdir(parents=True)
    (d / "runner.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "state.json").write_text(json.dumps(state))
    return d


def _texture_job(views=2292):
    return {"name": "avatar", "take": "t", "max_minutes": 39, "commands": [
        {"name": "bake", "cmd": "python -m dancecap texture t --preset pod", "timeout_min": 39, "optional": False}],
        "archives": [["a.tar.gz", ""]], "setup": []}


def test_snapshot_of_a_running_bake(tmp_path):
    t0 = time.mktime((2026, 10, 7, 10, 0, 0, 0, 0, -1))
    hh = lambda s: time.strftime("%H:%M:%S", time.localtime(t0 + s))
    iso = lambda s: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 + s))
    lines = [f"[pod-runner {hh(0)}] bundle_scan ...", f"[pod-runner {hh(1)}] bundle_scan done in 1.0s",
             f"[pod-runner {hh(1)}] bundle_build ...", f"[pod-runner {hh(1)}] bundle_build done in 0.0s",
             f"[pod-runner {hh(1)}] bundle b.tar.gz: 1521.7 MB (450 files, raw 1556.8 MB, 1 hard links); upload "
             f"~10.1 min; features ['base', 'texture', 'check']; pod max 72 min",
             f"[pod-runner {hh(40)}] upload ...", f"[pod-runner {hh(590)}] upload done in 550.0s",
             f"[pod-runner {hh(590)}] extract ...", f"[pod-runner {hh(600)}] extract done in 10.0s",
             f"[pod-runner {hh(600)}] bootstrap ...", "[bootstrap] base: start", "[bootstrap] base: done in 80s",
             "[bootstrap] texture: start", "[bootstrap] texture: done in 0s", "[bootstrap] check: start",
             "[bootstrap] check: done in 3s", f"[pod-runner {hh(690)}] bootstrap done in 90.0s",
             f"[pod-runner {hh(690)}] wheels_back ...", f"[pod-runner {hh(691)}] wheels_back done in 1.0s",
             f"[pod-runner {hh(691)}] avatar/archive ...", f"[pod-runner {hh(700)}] avatar/archive done in 9.0s",
             f"[pod-runner {hh(700)}] avatar/bake ...", f"[pod-runner] {iso(700)} start avatar/bake",
             "[texture] preset pod: 2292 views from 8 cameras over t=30.15..68.38 s"]
    lines += [f"[texture] {k}/2292 views" for k in range(50, 2292, 50)] + ["[texture] pass 0: 145 s"]
    lines += [f"[texture] {k}/2292 views" for k in range(50, 1451, 50)]
    state = {"sid": "20261007-100000-batch", "timings": [], "bundle": {"bytes": 1521690351},
             "jobs": {"avatar": {"status": "running", "spec": _texture_job()}}, "max_minutes": 72,
             "pod": {"id": "podX", "gpu": "NVIDIA A100-SXM4-80GB", "cost_per_hr": 1.39, "started": t0 + 38,
                     "killswitch": {"armed": True, "after_min": 72}}}
    d = _mk_session(tmp_path, "20261007-100000-batch", lines, state)
    rows = [{"pod_id": "podX", "cost_per_hr": 1.39, "created_ts": t0 + 5, "max_minutes": 72}]
    # replay as a live tracker would: lines stamped with receive times (pass 1 at 0.132 s/view from t0 + 700 + 300)
    evs, _, _ = P.parse_runner_log(d / "runner.log")
    t_pass1 = t0 + 700 + 160 + 145
    for e in evs:
        if e.get("step") == "bake/pass1" and e["kind"] == "tick":
            e["ts"] = t_pass1 + e["done"] * 0.132
        elif e.get("ts") is None:
            e["ts"] = t_pass1
    now = t_pass1 + 1450 * 0.132 + 1
    snap = E.snapshot(d, now=now, rates={"keys": {}}, ledger_rows=rows, events=evs)
    assert snap["state"] == "running" and snap["level"] == "L0"
    ph = {p["name"]: p for p in snap["phases"]}
    assert ph["upload"]["status"] == "done" and ph["avatar/bake"]["status"] == "running"
    assert "pass 2/3: 1450/2292" in ph["avatar/bake"]["detail"]
    rem = ph["avatar/bake"]["remaining"]["p50"]
    # rest of pass 1 (~111 s) + pass 2 at the live rate (~300 s) + post (~30 s, measured 2026-10-07)
    assert 380 < rem < 600, rem
    assert snap["kill"]["level"] == "green" and snap["kill"]["left_s"] > 0
    c = snap["cost"]
    assert c["so_far"] == pytest.approx(1.39 * (now - rows[0]["created_ts"]) / 3600, abs=1e-3)
    assert c["so_far"] < c["p50"] <= c["p90"] <= c["worst"] + 1e-6
    # the same session replayed after the end, from the file alone, is not 'running' once the ledger row closes
    rows[0].update(deleted_ts=now + 900, cost=1.0)
    snap2 = E.snapshot(d, now=now + 1000, rates={"keys": {}}, ledger_rows=rows)
    assert snap2["cost"]["so_far"] == 1.0


# ---------------------------------------------------------------------------------------------------- budget


def test_budget_spent_matches_remote_spent(tmp_path, monkeypatch):
    from dancecap import remote
    now = time.time()
    rows = [{"pod_id": "a", "cost_per_hr": 1.2, "created_ts": now - 7200, "deleted_ts": now - 3600, "cost": 1.25,
             "max_minutes": 60},
            {"pod_id": "b", "cost_per_hr": 1.39, "created_ts": now - 1800, "max_minutes": 72}]
    lp = tmp_path / "ledger.json"
    lp.write_text(json.dumps(rows))
    monkeypatch.setattr(remote, "LEDGER", lp)
    assert E.spent_from_rows(rows) == pytest.approx(remote.spent(), abs=1e-3)
    b = E.budget(now=now, rows=rows, cap=30.0, snapshots=[])
    assert b["spent"] == pytest.approx(1.25 + 1.39 * 0.5, abs=1e-3)
    pod = b["open_pods"][0]
    assert pod["worst_remaining"] == pytest.approx(1.39 * (72 + 6 - 30) / 60, abs=1e-3)
    assert b["level"] == "green" and b["worst"] >= b["projected"]["p90"]


def test_budget_warnings_phantom_and_launch_gate():
    now = time.time()
    rows = [{"pod_id": "ghost", "cost_per_hr": 1.0, "created_ts": now - 5 * 3600, "max_minutes": 60},
            {"pod_id": "p1", "cost_per_hr": 1.5, "created_ts": now - 600, "max_minutes": 120},
            {"pod_id": "p2", "cost_per_hr": 1.5, "created_ts": now - 600, "max_minutes": 120}]
    b = E.budget(now=now, rows=rows, cap=12.0, reserve=2.0, snapshots=[])
    assert any("past its kill time" in w["msg"] for w in b["warnings"])
    ghost = next(p for p in b["open_pods"] if p["pod_id"] == "ghost")
    assert ghost["phantom_cost"] == pytest.approx(1.0 * 66 / 60, abs=1e-3)
    assert b["level"] in ("amber", "red")
    g = E.launch_gate(1.0, now=now, rows=rows, cap=12.0, snapshots=[])
    # spent ~5.0 + 0.25 + 0.25, running worst 2 x 1.5 x (126 - 10) / 60 = 5.8 -> 12.3 > 12
    assert not g["allowed"] and g["running_worst"] == pytest.approx(5.8, abs=0.01)
    assert E.launch_gate(0.1, now=now, rows=rows[1:], cap=12.0, snapshots=[])["allowed"]
    red = E.budget(now=now, rows=rows[1:], cap=3.0, snapshots=[])
    assert red["level"] == "red"
    acct = E.budget(now=now, rows=rows[1:], cap=30.0, snapshots=[], account={"currentSpendPerHr": 4.5,
                                                                            "clientBalance": 100.0})
    assert any("untracked" in w["msg"] for w in acct["warnings"]) and acct["level"] == "red"


# ---------------------------------------------------------------------------------------------------- camera types


def test_camera_types_follow_the_canonical_stage_types(tmp_path, monkeypatch):
    """The estimator prices what the camera stages apply (dancecap.camera_types.stage_type): under the default `user`
    policy only confirmed answers count; an 'auto' suggestion, a table without `source`, or a missing zoom leave the
    camera on today's general model (review 2026-10-07)."""
    monkeypatch.delenv("DANCECAP_CAMERA_TYPES", raising=False)
    (tmp_path / "t.toml").write_text(
        'name = "t"\n[cameras]\nloc = { kf_by_time = true }\n'
        '[cameras."01"]\nmount = "tripod"\nzoom = false\nsource = "user"\n'
        '[cameras."02"]\nmount = "tripod"\nzoom = false\nsource = "auto"\n'
        '[cameras."03"]\nmount = "tripod"\nzoom = false\n'
        '[cameras."04"]\nmount = "tripod"\nsource = "user"\n'
        '[cameras."05"]\nmount = "tripod"\nzoom = true\nsource = "user"\n', encoding="utf-8")
    ct = E.camera_types("t", tmp_path)
    assert set(ct) == {"01", "02", "03", "04", "05"}
    assert ct["01"]["class"] == "tripod" and ct["01"]["confirmed"]
    for c in ("02", "03", "04"):  # unconfirmed suggestion / no source / no zoom: handheld + zoom, not confirmed
        assert ct[c]["class"] == "handheld_zoom" and not ct[c]["confirmed"], c
    assert ct["02"]["suggested"]["mount"] == "tripod"
    assert ct["05"]["class"] == "tripod_zoom" and ct["05"]["confirmed"]
    auto = E.camera_types("t", tmp_path, policy="auto")  # auto: only CONFIDENT suggestions (none here)
    assert auto["02"]["class"] == "handheld_zoom"
    off = E.camera_types("t", tmp_path, policy="off")
    assert off["01"]["class"] == "handheld_zoom" and off["01"]["answer"] == {"mount": "tripod", "zoom": False}

    # localisation units match stage_loc (front_pod never splits it): keyframes + their next frame only for a fixed,
    # non-zooming tripod
    assert E.loc_units(1151) == 1151
    tri = E.loc_units(1151, "tripod", False)
    assert 0.02 * 1151 < tri < 0.1 * 1151
    assert E.loc_units(1151, "tripod", True) == 1151           # zooming tripod: every frame (both code paths)
    assert E.loc_units(1151, "tripod", False, pans=True) == 1151  # a panning head: every frame
    est = E.loc_estimate({c: 1151 for c in ("01", "02", "03", "04", "05", "06")}, ct)
    assert est["units_p50"] == tri + 5 * 1151                   # only the confirmed fixed tripod saves
    assert est["units_p90"] == (tri + 1151) + 5 * 1151          # + its fallback: keyframes, then every frame
    assert est["unconfirmed"] == ["02", "03", "04", "06"] and est["p90"] > est["p50"]


def test_front_estimate_prices_effective_types_and_the_fallback():
    frames = {"01": 1152, "02": 2302, "03": 1152, "04": 1152, "05": 1152, "06": 1152, "07": 1151, "09": 1151}
    main, att = ["01", "02", "03", "04", "07", "09"], ["05", "06"]
    base = E.estimate_front(frames, {}, main, att, rates={"keys": {}})
    assert base["saving_s"] == 0 and 3600 < base["p50"] < 5400  # larissa_kadu_38 front job took 4374 s
    user = {c: {"mount": "tripod", "zoom": False, "source": "user", "confirmed": True} for c in frames}
    u = E.estimate_front(frames, user, main, att, rates={"keys": {}})
    assert u["p50"] < base["p50"] - 600 and u["loc_units_p50"] < 0.1 * base["loc_units_p50"]
    assert u["p90"] >= base["p90"]  # p90 keeps the keyframe fallback (constant pose rejected -> every frame)
    # unconfirmed suggestions change nothing in p50 (the stage ignores them) but say what confirming would save
    sugg = {c: {**E.DEFAULT_TYPE, "suggested": {"mount": "tripod", "zoom": False, "pans": False, "confidence": 0.8}}
            for c in frames}
    a = E.estimate_front(frames, sugg, main, att, rates={"keys": {}})
    assert a["p50"] == base["p50"] and a["p90"] == base["p90"]
    assert a["if_confirmed_p50"] == pytest.approx(u["p50"]) and a["potential_saving_s"] > 600
    zt = {c: {"mount": "tripod", "zoom": True, "source": "user", "confirmed": True} for c in frames}
    assert E.estimate_front(frames, zt, main, att, rates={"keys": {}})["p50"] == base["p50"]  # tripod + zoom


def test_front_dag_mirrors_front_pods_long_pole_split():
    frames = {"01": 1152, "02": 2302, "03": 1152, "05": 1153}
    T = E.front_dag(frames, {}, ["01", "02", "03"], ["05"])
    assert {"loc:02:0", "loc:02:1151", "loc:02"} <= set(T) and "loc:01:0" not in T
    assert T["loc:02"]["deps"] == ["loc:02:0", "loc:02:1151"]
    tri = {"02": {"mount": "tripod", "zoom": False, "source": "user", "confirmed": True}}
    T2 = E.front_dag(frames, tri, ["01", "02", "03"], ["05"])  # a locked-tripod long pole is never split (one
    assert "loc:02:0" not in T2 and "loc:02" in T2                 # keyframe decision per camera, as front_pod)
    assert T2["loc:02"]["duration"] < 0.2 * T["loc:02:0"]["duration"]
    # the long pole by SAM size, among ALL cameras: an attached long pole means no split at all
    T3 = E.front_dag(frames, {}, ["01", "02", "03"], ["05"], sizes={"01": 9, "02": 20, "03": 9, "05": 30})
    assert not any(k.count(":") == 2 for k in T3)
    odd = E.front_dag({"01": 1151, "02": 1151}, {}, ["01", "02"], [])
    assert "loc:01:576" in odd  # round(1151 / 2) = 576, as front_pod.build_tasks names it


def test_camera_motion_flags_use_probe_b_and_camera_types_thresholds():
    from dancecap import camera_types as CT
    still = {"centre_p95_m": 0.004, "axis_p95_deg": 0.1, "angvel_p50_dps": 0.01, "hf_rot_deg": 0.01,
             "hf_centre_mm": 0.2, "focal_ratio": 1.003, "focal_step_pct": 0.2, "frames": 1150, "metric": True}
    walk = {"centre_p95_m": 0.18, "axis_p95_deg": 6.0, "angvel_p50_dps": 4.0, "hf_rot_deg": 0.3,
            "hf_centre_mm": 9.0, "focal_ratio": 1.4, "focal_step_pct": 12.0, "frames": 1150, "metric": True}
    s_still, s_walk = CT.suggest_from_metrics(still), CT.suggest_from_metrics(walk)
    ev = lambda m, sug: {"metric": "camera_motion", "source": "probe_b", "constrained": False, **m,  # noqa: E731
                         **{k: sug[k] for k in ("mount", "zoom", "confidence", "mount_confidence", "zoom_confidence")}}
    tripod_user = {"mount": "tripod", "zoom": False, "source": "user", "confirmed": True, "set_at": 1000.0,
                   "answer": {"mount": "tripod", "zoom": False}}
    assert E.camera_type_flags("05", tripod_user, ev(still, s_still), at=2000.0) == []
    flags = E.camera_type_flags("03", tripod_user, ev(walk, s_walk), at=2000.0)
    assert [f["field"] for f in flags] == ["mount", "zoom"] and all(f["level"] == "amber" for f in flags)
    assert "18.0 cm" in flags[0]["msg"] and "x1.400" in flags[1]["msg"]
    # measured before the user's (re)answer: acknowledged, like camera_types' conflict state
    assert E.camera_type_flags("03", tripod_user, ev(walk, s_walk), at=500.0) == []
    # unconfirmed + a static camera: suggestions (info), never amber
    info = E.camera_type_flags("05", None, ev(still, s_still))
    assert {f["field"] for f in info} == {"mount", "zoom"} and all(f["level"] == "info" for f in info)
    # metrics are recomputed from the evidence when the event carries no suggestion
    assert [f["field"] for f in E.camera_type_flags("03", tripod_user, {"source": "probe_b", **walk})] == \
        ["mount", "zoom"]
    # the old metric on the type-constrained export (0 cm / x1.00 by construction) is ignored
    assert E.camera_type_flags("03", tripod_user, {"centre_spread_m": 0.0, "focal_ratio": 1.0}) == []
    # under policy `off` the answer is still what is checked
    off = {**E.DEFAULT_TYPE, "confirmed": True, "answer": {"mount": "tripod", "zoom": False}}
    assert [f["field"] for f in E.camera_type_flags("03", off, ev(walk, s_walk))] == ["mount", "zoom"]


def test_cameras_stage_emits_probe_b_metric(prog):
    from dancecap.cameras import pipeline as CP
    from dancecap import camera_types as CT
    m = {"centre_p95_m": 0.18, "axis_p95_deg": 6.0, "angvel_p50_dps": 4.0, "hf_rot_deg": 0.3, "hf_centre_mm": 9.0,
         "focal_ratio": 1.4, "focal_step_pct": 12.0, "frames": 1150, "metric": True}
    s = CT.suggest_from_metrics(m)
    s["evidence"]["attached"] = False
    CP._progress_camera_motion({"03": s}, {"s": 1.0})
    evs = [e for e in _events(prog) if e["kind"] == "metric"]
    assert len(evs) == 1
    x = evs[0]["extra"]
    assert x["metric"] == "camera_motion" and x["source"] == "probe_b" and x["constrained"] is False
    assert x["mount"] == "handheld" and x["centre_p95_m"] == 0.18 and evs[0]["lane"] == "03"
    flags = E.camera_type_flags("03", {"mount": "tripod", "zoom": False, "confirmed": True}, x, at=evs[0]["ts"])
    assert [f["field"] for f in flags] == ["mount", "zoom"]


# ---------------------------------------------------------------------------------------------------- review fixes


def test_live_rate_drops_a_resumed_pause():
    obs = [(i * 6.6, i * 50) for i in range(1, 15)]           # 0.132 s/view
    t = obs[-1][0] + 300                                        # a 300 s pause, then normal ticks again
    obs += [(t + i * 6.6, obs[-1][1] + i * 50) for i in range(1, 5)]
    i = E.live_rate_info(obs)
    assert i["dropped"] == 1 and i["s"] == pytest.approx(0.132, rel=0.05) and i["sigma"] >= 0.1
    r = E.remaining_linear(2292, obs[-1][1], obs, hist_s=0.13, n_hist=3, sigma_hist=0.15, now=obs[-1][0])
    truth = (2292 - obs[-1][1]) * 0.132
    assert r[0] < truth < r[2] and abs(r[1] - truth) / truth < 0.15
    # same-time ticks (one read batch) are one observation: no units lost, no rate inflation
    batch = [(0, 0), (10, 100), (10, 150), (20, 250), (20, 300), (30, 400)]
    assert E.live_rate_info(batch)["s"] == pytest.approx(30 / 400, rel=0.01)


def test_stalled_step_stops_counting_down_and_widens():
    obs = [(i * 10.0, i * 50) for i in range(1, 11)]  # 5 s/unit... 0.2 s/view, ticks every 10 s
    t_last = obs[-1][0]
    r_ok = E.remaining_linear(1000, 500, obs, hist_s=0.2, n_hist=3, now=t_last + 20)
    r_400 = E.remaining_linear(1000, 500, obs, hist_s=0.2, n_hist=3, now=t_last + 400)
    r_1000 = E.remaining_linear(1000, 500, obs, hist_s=0.2, n_hist=3, now=t_last + 1000, diag=(d := {}))
    assert r_ok[1] == pytest.approx(80, rel=0.1)                      # 100 s at the tick, 20 s spent since
    assert r_400[1] == r_1000[1] == pytest.approx(70, rel=0.1)        # stalled: only 3 tick gaps credited
    assert r_1000[2] > r_400[2] > r_ok[2] and d["stalled"] and d["since_advance_s"] == 1000
    # heartbeats / repeated values are no progress
    hb = obs + [(t_last + 30 * k, 500) for k in range(1, 30)]
    assert E.remaining_linear(1000, 500, hb, hist_s=0.2, n_hist=3, now=t_last + 900)[1] == pytest.approx(70, rel=0.1)


def _bake_session(tmp_path, t0, pod_events, now_s, sid="20261007-100000-batch"):
    hh = lambda s: time.strftime("%H:%M:%S", time.localtime(t0 + s))  # noqa: E731
    iso = lambda s: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 + s))  # noqa: E731
    lines = [f"[pod-runner {hh(0)}] bundle_scan ...", f"[pod-runner {hh(1)}] bundle_scan done in 1.0s",
             f"[pod-runner {hh(1)}] bundle_build ...", f"[pod-runner {hh(1)}] bundle_build done in 0.0s",
             f"[pod-runner {hh(40)}] upload ...", f"[pod-runner {hh(590)}] upload done in 550.0s",
             f"[pod-runner {hh(590)}] extract ...", f"[pod-runner {hh(600)}] extract done in 10.0s",
             f"[pod-runner {hh(600)}] bootstrap ...", f"[pod-runner {hh(690)}] bootstrap done in 90.0s",
             f"[pod-runner {hh(690)}] wheels_back ...", f"[pod-runner {hh(691)}] wheels_back done in 1.0s",
             f"[pod-runner {hh(691)}] avatar/archive ...", f"[pod-runner {hh(700)}] avatar/archive done in 9.0s",
             f"[pod-runner {hh(700)}] avatar/bake ...", f"[pod-runner] {iso(700)} start avatar/bake"]
    state = {"timings": [], "bundle": {"bytes": 1.5e9}, "max_minutes": 40,
             "jobs": {"avatar": {"status": "running", "spec": _texture_job()}},
             "pod": {"id": "podX", "gpu": "NVIDIA A100-SXM4-80GB", "cost_per_hr": 1.39, "started": t0 + 38,
                     "killswitch": {"armed": True, "after_min": 40}}}
    d = _mk_session(tmp_path, sid, lines, state)
    (d / "progress").mkdir()
    (d / "progress" / "pod.jsonl").write_text("\n".join(json.dumps(e) for e in pod_events) + "\n")
    rows = [{"pod_id": "podX", "cost_per_hr": 1.39, "created_ts": t0 + 5, "max_minutes": 40}]
    return E.snapshot(d, now=t0 + now_s, rates={"keys": {}}, ledger_rows=rows)


def test_stuck_step_warns_despite_heartbeats_and_resume_is_not_red(tmp_path):
    t0 = time.mktime((2026, 10, 7, 10, 0, 0, 0, 0, -1))
    base = {"v": 1, "job": "avatar", "cmd": "bake", "src": "pod"}
    pod = [{**base, "ts": t0 + 900, "step": "bake", "kind": "plan", "extra": {"views": 2292, "passes": 3}},
           {**base, "ts": t0 + 900, "step": "bake/load", "kind": "end", "extra": {"seconds": 200}},
           {**base, "ts": t0 + 900, "step": "bake/pass0", "kind": "start", "total": 2292, "unit": "views"}]
    pod += [{**base, "ts": t0 + 900 + k * 0.07, "step": "bake/pass0", "kind": "tick", "done": k, "total": 2292}
            for k in range(100, 1201, 100)]
    t_hang = t0 + 900 + 1200 * 0.07
    hbs = [{**base, "ts": t_hang + 30 * j, "step": "bake/pass0", "kind": "heartbeat", "done": 1200, "total": 2292}
           for j in range(1, 30)]
    snap = _bake_session(tmp_path / "a", t0, pod + hbs, (t_hang - t0) + 700)
    w = [x for x in snap["warnings"] if "no progress in avatar/bake" in x["msg"]]
    assert w and "alive (heartbeats)" in w[0]["msg"] and w[0]["stalled"]["done"] == 1200
    bake = next(p for p in snap["phases"] if p["name"] == "avatar/bake")
    assert bake["stalled"] and bake["stalled"]["since_s"] >= 690
    # a long pause that RESUMED: the rate stays ~0.07 s/view, the kill switch is not red
    t_res = t_hang + 600
    more = [{**base, "ts": t_res + j * 7, "step": "bake/pass0", "kind": "tick", "done": 1200 + 100 * j,
             "total": 2292} for j in range(1, 5)]
    snap2 = _bake_session(tmp_path / "b", t0, pod + more, (t_res - t0) + 30)
    bake2 = next(p for p in snap2["phases"] if p["name"] == "avatar/bake")
    assert "0.07" in bake2["detail"] and snap2["kill"]["level"] != "red"
    assert not any("no progress in" in x["msg"] for x in snap2["warnings"])


def test_cold_l0_texture_pass_is_timed_from_the_command_start(tmp_path):
    """An L0 log read cold (no receive times): an unfinished pass's ticks start after the load phase, not at the
    command start (review: 0.561 s/view read for a 0.07 s/view pass)."""
    t0 = time.mktime((2026, 10, 7, 10, 0, 0, 0, 0, -1))
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))
    lines = [f"[pod-runner] {iso} start avatar/bake", "[texture] preset pod: 2292 views from 8 cameras"]
    lines += [f"[texture] {k}/2292 views" for k in range(50, 501, 50)]
    evs, _, parser = P.parse_runner_log(_mk_session(tmp_path, "20261007-100000-batch", lines, {}) / "runner.log")
    end = t0 + 0.128 * 2292 + 500 * 0.07  # load phase (history) + 500 views at 0.07 s/view
    E._fill_times(evs, end, None, parser.n)
    ticks = [e for e in evs if e.get("kind") == "tick"]
    assert ticks[-1]["ts"] == pytest.approx(end) and ticks[-1]["ts_est"] == "interp_pass"
    s = (ticks[-1]["ts"] - ticks[0]["ts"]) / (ticks[-1]["done"] - ticks[0]["done"])
    assert s == pytest.approx(0.07, rel=0.05)


def test_tracker_stamps_batches_and_live_snapshot(tmp_path, monkeypatch):
    t0 = time.mktime((2026, 10, 7, 10, 0, 0, 0, 0, -1))
    hh = lambda s: time.strftime("%H:%M:%S", time.localtime(t0 + s))  # noqa: E731
    d = _mk_session(tmp_path, "20261007-100000-batch", [f"[pod-runner {hh(0)}] bundle_scan ..."], {})
    log = d / "runner.log"
    tr = E.Tracker(d, rates={"keys": {}})
    tr.poll(now=t0 + 1)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("[texture] 50/2292 views\n")
    os.utime(log, (t0 + 3, t0 + 3))
    tr.poll(now=t0 + 4)                       # within LIVE_BATCH_S: the file's mtime
    assert tr.events[-1]["rx"] == pytest.approx(t0 + 3)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("".join(f"[texture] {k}/2292 views\n" for k in (100, 150, 200, 250)))
    os.utime(log, (t0 + 100, t0 + 100))
    tr.poll(now=t0 + 104)                     # a 100 s gap: spread over (last poll, mtime] by line
    rx = [e["rx"] for e in tr.events if e.get("kind") == "tick"][-4:]
    assert rx == sorted(rx) and rx[0] < t0 + 40 and rx[-1] == pytest.approx(t0 + 100)
    log.write_text("[pod-runner 10:00:00] bundle_scan ...\n")  # rewritten (shorter): start over
    tr.poll(now=t0 + 110)
    assert len(tr.events) == 1
    # the monitor's entry point: one tracker per session, reused, background poller optional
    monkeypatch.setattr(E, "_live", {})
    a = E.live_snapshot(d, poll_thread=False)
    assert a["sid"] == d.name and E.live_tracker(d) is not None
    E.live_snapshot(d, poll_thread=False)
    assert len(E._live) == 1


def test_front_utc_clock_rolls_over_midnight():
    p = P.LineParser(date0=(2026, 10, 7))
    a = p.feed("[front] 00:10:00 sam:02: ok rc=0 in 10s")[0]["ts"]   # no ISO line yet: after 00:00 UTC -> 10-08
    assert a >= time.mktime((2026, 10, 7, 0, 0, 0, 0, 0, -1))
    q = P.LineParser(date0=(2026, 10, 7))
    q.feed("[pod-runner] 2026-10-07T23:50:00Z start front/run")
    b = q.feed("[front] 23:55:00 start loc:02")[0]["ts"]
    c = q.feed("[front] 00:10:00 loc:02: ok rc=0 in 900s")[0]["ts"]
    assert c - b == pytest.approx(900)


def test_small_review_fixes():
    assert E._cmd_key("prep", {"cmd": "python -m dancecap.front_pod fetch-sam2", "name": "f"}) == \
        "front_pod.fetch-sam2"
    assert E._cmd_key("t", {"cmd": "python -m dancecap texture larissa_kadu_38 --preset pod"}) == "texture"
    assert E._cmd_key("m", {"cmd": "python -m dancecap.mamma_chunks fit larissa_kadu_38 --lanes 7"}) == \
        "mamma_chunks.fit"
    assert E.gpu_command("texture") and E.gpu_command("mamma_chunks.fit") and not E.gpu_command("timing")
    assert not E.gpu_command("physics") and not E.gpu_command("pose_quality") and not E.gpu_command(
        "hand_mano_worker.run")
    from dancecap import remote
    assert E._remote_cap() == remote.cap()
    sig = E.SessionModel(Path("x"), [], {}, 0.0, {"keys": {"k": {"n": 3, "p50": 1.0, "sigma": 0.05}}}, [])._hist(
        "k", 1.0)[2]
    assert sig >= E.HIST_SIGMA_FLOOR


def test_runner_timed_tolerates_reserved_extra_keys(tmp_path):
    from dancecap import remote_jobs as RJ
    ses = RJ.Session.__new__(RJ.Session)
    ses.dir, ses.sid, ses.state = tmp_path, "s", {"timings": []}
    ses.save = lambda: None
    ses.log = lambda msg: None
    assert ses.timed("x", lambda: 5, seconds=3, msg="m", mb=1.0) == 5
    assert ses.state["timings"][0]["mb"] == 1.0


# ---------------------------------------------------------------------------------------------------- puller


def test_probe_is_read_only():
    src = PP.PROBE
    for bad in ("environ", "kill", ".write(", "remove(", "unlink", "rmtree", "rename", "replace(", "chmod",
                "system(", "Popen", '"w"', "'w'", '"a"'):
        assert bad not in src, bad
    compile(src, "probe", "exec")
    assert len(src.encode()) < 8192


def test_puller_mirrors_offsets_exactly_and_never_raises(tmp_path):
    pod_file = tmp_path / "pod" / "progress.jsonl"
    pod_file.parent.mkdir()
    ev = lambda d: json.dumps({"v": 1, "ts": 1.0, "job": "avatar", "step": "bake/pass1", "kind": "tick", "done": d,
                               "total": 10, "unit": "views"})
    pod_file.write_bytes((ev(1) + "\n" + ev(2)[:20]).encode())  # a split last line (bytes: no CRLF on Windows)
    tail = tmp_path / "pod" / "ma_3d_c8_f0-232.log"
    tail.write_bytes(b"Iter. 1 of 50 | x\nIter. 2 of 50 | x\n")
    kill = tmp_path / "pod" / "kill_at"
    kill.write_text("1791400000\n")
    part = tmp_path / "pod" / "bundle.tar.gz.part"
    part.write_bytes(b"x" * 1234)
    sess = tmp_path / "sess"
    pu = PP.Puller(sess, "podX", PP.local_transport(), str(pod_file), interval=0.05, upload_part=str(part),
                   upload_bytes=5000, deadline_file=str(kill),
                   tails=[{"glob": str(tmp_path / "pod" / "ma_3d_*.log"), "adapter": "mamma_iter",
                           "lane_from": r"ma_3d_(.+)\.log$", "step": "fit_chunks", "job": "mamma"}])
    assert pu.tick()
    st = json.loads((sess / "progress" / "puller.json").read_text())
    assert st["offset"] == len(ev(1)) + 1 and st["kill_at"] == 1791400000 and st["upload_part_bytes"] == 1234
    with open(pod_file, "ab") as fh:
        fh.write((ev(2)[20:] + "\n" + ev(3) + "\n").encode())
    with open(tail, "ab") as fh:
        fh.write(b"Stopping early at iteration 16 due\n")
    assert pu.tick()
    st = json.loads((sess / "progress" / "puller.json").read_text())
    assert st["offset"] == pod_file.stat().st_size
    evs, _ = P.read_jsonl(sess / "progress" / "pod.jsonl")
    mirrored = [e for e in evs if e.get("src") != "tail"]
    assert [e["done"] for e in mirrored] == [1, 2, 3] and all(e["pod"] == "podX" and "skew" in e for e in mirrored)
    tails = [e for e in evs if e.get("src") == "tail"]
    assert [e["kind"] for e in tails if e["kind"] != "start"] == ["tick", "tick", "end"]
    assert all(e["lane"] == "c8_f0-232" and e["job"] == "mamma" for e in tails)
    assert tails[-1]["step"] == "fit_chunks/run1" and tails[-1]["done"] == 16

    def broken(_):
        raise RuntimeError("ssh: connect timed out rpa_" + "X" * 40)
    bad = PP.Puller(sess, "podX", broken, str(pod_file), interval=0.01, slow_interval=0.02)
    for _ in range(4):
        assert bad.tick() is False
    st = json.loads((sess / "progress" / "puller.json").read_text())
    assert st["fails"] == 4 and "rpa_" not in st["last_err"] and "[redacted]" in st["last_err"]
    bad.start()
    time.sleep(0.15)
    bad.stop()  # thread ends cleanly, nothing escaped
    assert not bad.thread.is_alive()


# ---------------------------------------------------------------------------------------------------- runner hooks


def test_command_script_sets_progress_env_and_stays_valid_bash():
    from dancecap import remote_jobs as rj
    job = rj.parse_job({"schema": rj.SCHEMA, "name": "avatar", "take": "lk", "max_minutes": 10,
                        "commands": [{"name": "bake", "cmd": "echo hi"}], "outputs": ["dancecap/work/lk/a.json"],
                        "env": {"DANCECAP_PROGRESS_JOB": "override"}})
    c = rj.command_script(job, job.commands[0], "dancecap/work/runpod/sessions/S1/pod_logs/avatar")
    assert "DANCECAP_PROGRESS=/workspace/HeadMovement/dancecap/work/runpod/sessions/S1/progress.jsonl" in c
    assert "DANCECAP_PROGRESS_CMD=bake" in c and "DANCECAP_SID=S1" in c
    assert "DANCECAP_PROGRESS_JOB=override" in c  # a job's own env still wins
    assert rj._progress_env(job, job.commands[0], "weird/path") == {}
    bash = shutil.which("bash")
    if bash:
        assert subprocess.run([bash, "-n", "-c", c]).returncode == 0


def test_session_events_and_runner_info(tmp_path, monkeypatch):
    from dancecap import remote_jobs as rj
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    ses = rj.Session("20261007-120000-test")
    ses.write_runner_info("test", ["a"])
    ses.write_runner_info("test", ["a"])
    info = json.loads((ses.dir / "runner.json").read_text())
    assert info["pid"] == os.getpid() and info["attempt"] == 2 and "argv" not in info
    assert ses.timed("upload", lambda: 7, mb=12.5) == 7
    with pytest.raises(RuntimeError):
        ses.timed("avatar/bake", lambda: (_ for _ in ()).throw(RuntimeError("x")), job="avatar")
    evs, _ = P.read_jsonl(ses.dir / "progress" / "runner.jsonl")
    assert [(e["kind"], e["job"], e["step"]) for e in evs] == [
        ("start", "_session", "upload"), ("end", "_session", "upload"), ("start", "avatar", "bake"),
        ("fail", "avatar", "bake")]
    assert evs[1]["extra"]["mb"] == 12.5 and evs[1]["extra"]["runner_step"] == "upload"
    # runner.log is unchanged in format (the L0 parser still reads it)
    parsed, _, _ = P.parse_runner_log(ses.dir / "runner.log")
    assert [e["kind"] for e in parsed if e["src"] == "runner"] == ["start", "end", "start", "fail"]


def test_estimate_minutes_unchanged_without_rates_and_history_with():
    from dancecap import remote_jobs as rj
    job = rj.parse_job({"schema": rj.SCHEMA, "name": "avatar", "take": "lk", "max_minutes": 39, "needs": ["texture"],
                        "commands": [{"name": "bake", "cmd": "python -m dancecap texture lk --preset pod",
                                      "timeout_min": 39}], "outputs": ["dancecap/work/lk/a.json"]})
    a = rj.estimate_minutes([job], 1_500_000_000, cached_wheels=True)
    b = rj.estimate_minutes([job], 1_500_000_000, cached_wheels=True, rates={"keys": {}})
    assert {k: b[k] for k in a} == a and "history" not in a
    assert b["history"]["p50_min"] > 10 and b["p90_total"] > b["history"]["p50_min"]


def test_rates_rebuild_on_fake_sessions(tmp_path):
    t0 = 1791390000.0
    st = {"timings": [{"step": "upload", "seconds": 600.0, "ok": True, "start": t0, "mb": 1500.0},
                      {"step": "avatar/bake", "seconds": 1080.0, "ok": True, "start": t0 + 700, "job": "avatar"}],
          "jobs": {"avatar": {"status": "ok", "commands": {"bake": 0}, "spec": _texture_job()}},
          "pod": {"id": "p", "gpu": "NVIDIA A100-SXM4-80GB", "started": t0 - 30}}
    lines = ["[pod-runner] 2026-10-07T17:08:55Z start avatar/bake",
             "[texture] preset pod: 2292 views from 8 cameras", "[texture] pass 0: 145 s",
             "[texture] pass 1 (aligned): 315 s", "[texture] pass 2 (aligned): 303 s",
             "[texture] -> /x (1076 s)", "[pod-runner] 2026-10-07T17:26:54Z end avatar/bake rc=0"]
    _mk_session(tmp_path / "s", "20261007-100000-batch", lines, st)
    out = tmp_path / "rates.json"
    doc = E.rates_rebuild(tmp_path / "s", out, ledger_rows=[{"pod_id": "p", "created_ts": t0 - 64}])
    k = doc["keys"]
    assert k["session/upload/MBps"]["p50"] == pytest.approx(2.5)
    assert k["session/pod_start/s"]["p50"] == pytest.approx(34)
    assert k["texture/pass0/s_per_view@A100-SXM4"]["p50"] == pytest.approx(145 / 2292, rel=1e-3)
    assert k["texture/aligned/s_per_view@A100-SXM4"]["n"] == 2
    assert k["texture/fixed/s_per_view@A100-SXM4"]["p50"] == pytest.approx((1076 - 763) / 2292, rel=1e-3)
    assert k["cmd/texture/s@A100-SXM4"]["p50"] == 1080.0
    assert json.loads(out.read_text())["keys"]
    v, n, sig, src = E.lookup(doc, "texture/aligned/s_per_view", "NVIDIA A100 80GB PCIe")
    assert src.startswith("history (A100-SXM4)")
    assert v == pytest.approx(k["texture/aligned/s_per_view@A100-SXM4"]["p50"] * 1.05, rel=1e-6)


def test_local_stage_snapshot_and_pid_alive(tmp_path, monkeypatch):
    f = tmp_path / "t" / "progress" / "ingest.jsonl"
    monkeypatch.setenv("DANCECAP_PROGRESS", str(f))
    P._writers.clear()
    clock = [100.0]
    monkeypatch.setattr(P.time, "time", lambda: clock[0])
    for i in range(0, 600, 10):
        clock[0] += 5
        P.tick("decode", i + 10, 1200, "frames", lane="02")
    P._writers.clear()
    snap = E.local_stage_snapshot("t", "ingest", now=clock[0], work=tmp_path)
    st = snap["steps"][0]
    assert st["lane"] == "02" and st["status"] == "running" and 0.45 < st["fraction"] <= 0.5
    assert st["remaining"]["p50"] == pytest.approx(300, rel=0.1)  # 600 frames left at 0.5 s/frame
    assert E.pid_alive(os.getpid()) is True and E.pid_alive("x") is None


@pytest.mark.skipif(not (SESS / "20261007-002359-batch" / "state.json").exists() or not FRONT_POD.exists(),
                    reason="local session data")
def test_front_replay_eta_within_15_percent():
    d = SESS / "20261007-002359-batch"
    evs = E.session_events(d)
    starts = [e for e in evs if e.get("src") == "front" and e["kind"] == "start"]
    t_first = starts[0]["ts"]
    t_end = max(e["ts"] for e in evs if e.get("src") == "front" and e["kind"] == "end" and e.get("lane") == "prepare")
    for dt in (900, 1800):
        snap = E.snapshot(d, at=t_first + dt, rates={"keys": {}})
        ph = next(p for p in snap["phases"] if p["name"] == "front/front" and p["status"] == "running")
        actual = t_end - (t_first + dt)
        assert abs(ph["remaining"]["p50"] - actual) / actual < 0.15, (dt, ph["remaining"], actual)


def test_l2_pod_events_win_over_l0_for_the_same_steps(tmp_path):
    t0 = time.mktime((2026, 10, 7, 10, 0, 0, 0, 0, -1))
    hh = lambda s: time.strftime("%H:%M:%S", time.localtime(t0 + s))
    iso = lambda s: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 + s))
    lines = [f"[pod-runner {hh(0)}] bundle_scan ...", f"[pod-runner {hh(1)}] bundle_scan done in 1.0s",
             f"[pod-runner {hh(1)}] bundle_build ...", f"[pod-runner {hh(1)}] bundle_build done in 0.0s",
             f"[pod-runner {hh(40)}] upload ...", f"[pod-runner {hh(590)}] upload done in 550.0s",
             f"[pod-runner {hh(590)}] extract ...", f"[pod-runner {hh(600)}] extract done in 10.0s",
             f"[pod-runner {hh(600)}] bootstrap ...", f"[pod-runner {hh(690)}] bootstrap done in 90.0s",
             f"[pod-runner {hh(690)}] wheels_back ...", f"[pod-runner {hh(691)}] wheels_back done in 1.0s",
             f"[pod-runner {hh(691)}] avatar/archive ...", f"[pod-runner {hh(700)}] avatar/archive done in 9.0s",
             f"[pod-runner {hh(700)}] avatar/bake ...", f"[pod-runner] {iso(700)} start avatar/bake",
             "[texture] preset pod: 2292 views from 8 cameras", "[texture] 50/2292 views"]
    state = {"timings": [], "bundle": {"bytes": 1.5e9}, "max_minutes": 72,
             "jobs": {"avatar": {"status": "running", "spec": _texture_job()}},
             "pod": {"id": "podX", "gpu": "NVIDIA A100-SXM4-80GB", "cost_per_hr": 1.39, "started": t0 + 38,
                     "killswitch": {"armed": True, "after_min": 72}}}
    d = _mk_session(tmp_path, "20261007-100000-batch", lines, state)
    (d / "progress").mkdir()
    base = {"v": 1, "job": "avatar", "cmd": "bake", "src": "pod"}
    pod = [{**base, "ts": t0 + 700 + 200, "step": "bake", "kind": "plan", "extra": {"views": 2292, "passes": 3}},
           {**base, "ts": t0 + 900, "step": "bake/load", "kind": "end", "extra": {"seconds": 200}},
           {**base, "ts": t0 + 900, "step": "bake/pass0", "kind": "start", "total": 2292, "unit": "views"}]
    pod += [{**base, "ts": t0 + 900 + k * 0.07, "step": "bake/pass0", "kind": "tick", "done": k, "total": 2292,
             "unit": "views"} for k in range(100, 1201, 100)]
    (d / "progress" / "pod.jsonl").write_text("\n".join(json.dumps(e) for e in pod) + "\n")
    now = t0 + 900 + 1200 * 0.07 + 1
    snap = E.snapshot(d, now=now, rates={"keys": {}}, ledger_rows=[{"pod_id": "podX", "cost_per_hr": 1.39,
                                                                    "created_ts": t0 + 5, "max_minutes": 72}])
    bake = next(p for p in snap["phases"] if p["name"] == "avatar/bake")
    assert snap["level"] == "L2" and "pass 1/3: 1200/2292" in bake["detail"] and "0.070 s/view" in bake["detail"]


def test_worker_lane_from_environment(prog, monkeypatch):
    monkeypatch.setenv("DANCECAP_PROGRESS_LANE", "loc:02:1151")
    P._writers.clear()
    P.tick("loc", 1, 1151, "frames")
    P.tick("loc", 1, 1151, "frames", lane="explicit")
    evs = _events(prog)
    assert {e["lane"] for e in evs} == {"loc:02:1151", "explicit"} and P.default_lane() == "loc:02:1151"


def test_ssh_transport_uses_the_pods_own_options(monkeypatch):
    from dancecap import remote
    pod = remote.Pod("podX", "batch", "A100", 1.39, "1.2.3.4", 23073)
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["input"] = argv, kw.get("input")
        return subprocess.CompletedProcess(argv, 0, stdout='{"now": 1, "v": 1}\n', stderr="")
    monkeypatch.setattr(PP.subprocess, "run", fake_run)
    out = PP.ssh_transport(pod)("QUJD")
    a = seen["argv"]
    assert out.startswith("{") and a[0] == "ssh" and "23073" in a and a[-2] == "root@1.2.3.4"
    assert a[-1] == "python3 - QUJD" and "ConnectTimeout=10" in a and "BatchMode=yes" in a
    assert seen["input"] == PP.PROBE
    assert PP.pod_progress_path("S1").endswith("/dancecap/work/runpod/sessions/S1/progress.jsonl")
    tails = PP.default_tails([{"take": "lk", "commands": [{"cmd": "python -m dancecap.mamma_chunks fit lk"}]}])
    assert tails[0]["adapter"] == "mamma_iter" and tails[0]["glob"].endswith("/lk/mamma/logs/ma_3d_*.log")
