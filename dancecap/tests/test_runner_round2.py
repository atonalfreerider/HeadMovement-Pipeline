"""Runner round 2 (2026-10-21 trial gaps): bootstrap features for video / physics, the FrameStream hang, GPU order (fastest
first, GPU before cloud, broken hosts last), max-cost honesty, and the pod-job frame counter. Fakes only: no pod is created."""
import contextlib
import hashlib
import io
import json
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import progress as P  # noqa: E402
from dancecap import remote  # noqa: E402
from dancecap import remote_jobs as rj  # noqa: E402
from dancecap.cameras import newcam  # noqa: E402

BOOT = Path(rj.PACKAGE_ROOT) / "remote" / "bootstrap.sh"
BASH = shutil.which("bash")
G4090, L40S, A40, H100, A100P = ("NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA A40", "NVIDIA H100 80GB HBM3",
                                 "NVIDIA A100 80GB PCIe")


def _bash_path(p: Path) -> str:
    """A path as Git Bash on Windows wants it inside PATH / globs (/c/Users/...); the posix path elsewhere."""
    if shutil.which("cygpath"):
        return subprocess.run(["cygpath", "-u", str(p)], capture_output=True, text=True).stdout.strip()
    return p.as_posix()


def _job(name="j", **kw):
    d = {"schema": rj.SCHEMA, "name": name, "take": "t", "max_minutes": 10, "commands": ["true"],
         "outputs": [f"dancecap/work/t/{name}/"]}
    d.update(kw)
    return rj.parse_job(d, "test")


# ------------------------------------------------------------------------------------------------ 1. bootstrap features


def test_video_and_physics_jobs_get_ffmpeg_and_open3d_without_declaring_them():
    front = _job("front", needs=["ma2d", "mamma"], commands=["python work/t/agent/trial_front.py all"])
    s3db = _job("s3db", inputs=["dancecap/work/trial_src/t/t.mp4"], commands=["bash remote/cand_sam3d_body/install.sh"])
    analyses = _job("analyses", needs=["mamma"], commands=["python -m dancecap timing t", "python -m dancecap physics t",
                                                           "python -m dancecap.floorcraft t"])
    probe = _job("probe", commands=["ffprobe -v error x"])
    fit = _job("fit", needs=["mamma"], commands=["python -m dancecap mamma fit t_tj0000 --tag c1"])
    assert rj.implied_features(front) == ["ffmpeg"] and rj.implied_features(s3db) == ["ffmpeg"]
    assert rj.implied_features(analyses) == ["open3d"] and rj.implied_features(probe) == ["ffmpeg"]
    assert rj.implied_features(fit) == [] and front.needs == ["ma2d", "mamma"]          # the spec itself is untouched
    assert rj.needed_features([front, analyses]) == ["base", "mamma", "ma2d", "ffmpeg", "open3d", "check"]
    assert rj.implied_features(_job("x", needs=["ffmpeg"], inputs=["a.mp4"])) == []      # declared: not twice
    # light features never pull the 330 MB pywheels / the CUDA wheelhouse into a slim bundle
    os.environ.pop("DANCECAP_BUNDLE_PYWHEELS", None)
    pol = rj.wheel_policy([_job("y", needs=["ffmpeg", "open3d"]), s3db])
    assert pol["pywheels"] is False and pol["wheelhouse"] is False
    assert all(f in rj.BOOTSTRAP_MIN and f in rj.BOOTSTRAP_MIN_CACHED for f in rj.FEATURES)


def test_failed_implied_feature_skips_the_job(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    ses = rj.Session("20261021-200000-t")
    job = _job("analyses", needs=["mamma"], commands=["python -m dancecap physics t"])
    ses.state["jobs"][job.name] = {"status": "pending"}
    (ses.dir / "bootstrap_timings.tsv").write_text("base\t90\t0\nmamma\t30\t0\nopen3d\t20\t1\n")
    monkeypatch.setattr(rj, "_failed_features", lambda s: {"open3d"})
    ran = []
    monkeypatch.setattr(rj, "_run_job", lambda *a, **k: ran.append(1))

    class Pod:
        def output(self, cmd, timeout=0):
            return ""

        def run(self, cmd, **kw):
            return 0

        def get(self, *a, **k):
            raise RuntimeError("no file")
    monkeypatch.setattr(rj, "_retry", lambda fn, **k: None)
    monkeypatch.setattr(rj, "snapshot", lambda *a, **k: None)
    monkeypatch.setattr(rj, "_sync_wheels", lambda pod, s: [])
    rj._pod_steps(ses, Pod(), [job], tmp_path / "bundle_x.tar.gz", {"key": "k", "n_files": 1, "bytes": 1, "sha256": "x"},
                  rj.needed_features([job]))
    assert not ran and ses.state["jobs"]["analyses"]["status"] == "skipped: bootstrap feature failed ['open3d']"


def test_bootstrap_defines_every_runner_feature_and_parses():
    text = BOOT.read_text()
    for f in rj.FEATURES:
        assert f"feature_{f}()" in text, f
    assert '--ignore-installed "blinker>=1.6"' in text.split("feature_open3d()")[1].split("\n}")[0]
    bg = text.split("ffmpeg_fetch_bg() {")[1].split("\n}")[0]
    assert "ffmpeg_fetch; rc=$?" in bg and '> "$STATE/ffmpeg_fetch.log" 2>&1 &' in bg    # fetched next to base, not after it
    if BASH:
        assert subprocess.run([BASH, "-n", str(BOOT)]).returncode == 0


def _fake_ffmpeg_tarball(path: Path) -> None:
    script = b"#!/bin/sh\necho \"ffmpeg version 8.1-fake Copyright\"\n"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as t:
        for name in ("ffmpeg-n8.1-latest-linux64-gpl-8.1/bin/ffmpeg", "ffmpeg-n8.1-latest-linux64-gpl-8.1/bin/ffprobe",
                     "ffmpeg-n8.1-latest-linux64-gpl-8.1/doc/ffmpeg.html"):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(script), 0o755
            t.addfile(ti, io.BytesIO(script))
    path.write_bytes(lzma.compress(raw.getvalue()))


@pytest.mark.skipif(not BASH or not shutil.which("curl"), reason="needs bash + curl")
def test_bootstrap_ffmpeg_feature_installs_a_static_build(tmp_path):
    """The real bootstrap.sh, feature 'ffmpeg' only (base marked done): the static tarball is fetched (here from a file://
    URL, after one dead URL), ffmpeg + ffprobe land in DANCECAP_FFMPEG_BIN, the feature leaves its .ok marker."""
    tb = tmp_path / "ffmpeg.tar.xz"
    _fake_ffmpeg_tarball(tb)
    state, binp = tmp_path / "state", tmp_path / "bin"
    state.mkdir()
    (state / "base.ok").write_text("v4\n")
    env = dict(os.environ, DANCECAP_BOOTSTRAP_STATE=state.as_posix(), DANCECAP_FFMPEG_BIN=binp.as_posix(),
               DANCECAP_WS=tmp_path.as_posix(), DANCECAP_WHEELHOUSE=(tmp_path / "wh").as_posix(),
               PY=Path(sys.executable).as_posix(),
               DANCECAP_FFMPEG_URLS=f"{(tmp_path / 'missing.tar.xz').as_uri()} {tb.as_uri()}")
    # a PATH without any ffmpeg (the local one lives elsewhere); the install dir first, as /usr/local/bin is on a pod
    script = f'export PATH="{_bash_path(binp)}:/usr/bin:/bin:/mingw64/bin"; bash "{BOOT.as_posix()}" ffmpeg'
    p = subprocess.run([BASH, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out
    assert "ffmpeg: static build fetched" in out and "ffmpeg version 8.1-fake" in out, out
    assert (binp / "ffmpeg").exists() and (binp / "ffprobe").exists()
    assert (state / "ffmpeg.ok").read_text().strip() == "v4"
    assert "ffmpeg\t" in (state / "timings.tsv").read_text()
    p2 = subprocess.run([BASH, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert p2.returncode == 0 and "ffmpeg: already done" in p2.stdout     # idempotent


# ------------------------------------------------------------------------------------------------ 2. FrameStream hang


def _stream_in_thread(fs, timeout=15.0):
    """Iterate a FrameStream in a thread: (items, exception) or ('hung', None) when it does not finish in time."""
    box = {}

    def go():
        try:
            box["items"] = list(fs)
        except Exception as e:  # noqa: BLE001
            box["err"] = e
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return "hung", None
    return box.get("items"), box.get("err")


def test_framestream_probe_failure_ends_the_stream_with_the_error(monkeypatch):
    def no_ffprobe(path):
        raise FileNotFoundError(2, "No such file or directory", "ffprobe")
    monkeypatch.setattr(newcam, "probe", no_ffprobe)
    monkeypatch.setattr(newcam, "_passthrough", lambda: [])
    t0 = time.time()
    items, err = _stream_in_thread(newcam.FrameStream(Path("x.mp4"), np.arange(10) / 30, 0.0, 0.3, (64, 48)))
    assert items != "hung" and isinstance(err, RuntimeError), (items, err)   # used to wait forever
    assert "ffprobe" in str(err) and time.time() - t0 < 10


def test_framestream_missing_ffmpeg_ends_the_stream_with_the_error(monkeypatch):
    monkeypatch.setattr(newcam, "probe", lambda p: {"time_base": "1/30"})
    monkeypatch.setattr(newcam, "_passthrough", lambda: [])

    def popen(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")
    monkeypatch.setattr(newcam.subprocess, "Popen", popen)
    items, err = _stream_in_thread(newcam.FrameStream(Path("x.mp4"), np.arange(10) / 30, 0.0, 0.3, (64, 48)))
    assert items != "hung" and isinstance(err, RuntimeError) and "ffmpeg" in str(err)


def test_framestream_prepare_error_reaches_the_consumer(monkeypatch, tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("no local ffmpeg")
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-frames:v", "12",
                    "-pix_fmt", "yuv420p", str(clip)], check=True, timeout=60)
    pts = newcam.video_pts(clip)
    good = newcam.FrameStream(clip, pts, float(pts[0]), float(pts[-1]), (64, 48))
    items, err = _stream_in_thread(good)
    assert err is None and len(items) == 12 and items[0][2].shape == (48, 64, 3)
    first = next(iter(newcam.FrameStream(clip, pts, float(pts[0]), float(pts[-1]), (64, 48))))   # early stop: no hang
    assert first[0] == 0

    def bad(n, img):
        raise ValueError("mask failed")
    items, err = _stream_in_thread(newcam.FrameStream(clip, pts, float(pts[0]), float(pts[-1]), (64, 48), prepare=bad))
    assert items != "hung" and isinstance(err, RuntimeError) and "mask failed" in str(err)
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    monkeypatch.setattr(newcam, "probe", lambda p: {"time_base": "1/30"})
    items, err = _stream_in_thread(newcam.FrameStream(junk, pts, 0.0, 0.3, (64, 48)))
    assert items != "hung" and isinstance(err, RuntimeError) and "decoded no frame" in str(err)


# ------------------------------------------------------------------------------------------------ 3. GPU choice


def test_default_gpu_order_is_fastest_first_with_measured_memory_needs():
    assert rj.job_vram_gb(_job("s3db")) == 24.0 and rj.job_vram_gb(_job("x", min_vram_gb=8)) == 8.0
    assert rj.job_vram_gb(_job("r", needs=["room"])) == 80.0
    assert rj.job_vram_gb(_job("ct3", commands=["python ct3_full.py --cotracker3"])) == 80.0
    order = remote.gpus_for([rj.job_vram_gb(_job("s3db"))])
    assert order[0] == G4090 and order[-1] == "NVIDIA GeForce RTX 3090"
    assert order.index(H100) < order.index("NVIDIA A100-SXM4-80GB") < order.index(A40)
    assert not any("5090" in g or "PRO" in g or "B200" in g for g, _ in remote.GPU_LADDER)   # no sm_120 kernels
    big = remote.gpus_for([80.0])
    assert big[:3] == [H100, "NVIDIA H100 NVL", "NVIDIA H100 PCIe"] and G4090 not in big
    assert remote.GPU_PREFERENCE[0].startswith("NVIDIA H100")


def test_option_order_tries_a_gpu_in_every_cloud_before_the_next_gpu():
    order, demoted = remote.option_order([G4090, L40S], ["COMMUNITY", "SECURE"], {(G4090, "COMMUNITY"): "203.0.113.9"})
    assert order == [(G4090, "SECURE"), (L40S, "COMMUNITY"), (L40S, "SECURE"), (G4090, "COMMUNITY")]
    assert demoted == [(G4090, "COMMUNITY")]
    assert remote.option_order([G4090], ["SECURE"])[0] == [(G4090, "SECURE")]


@pytest.fixture
def fake_runpod(tmp_path, monkeypatch):
    """remote.pod_session against fakes. env.prices {(gpu, cloud): $/h | None}, env.hosts {(gpu, cloud): ip},
    env.cuda_bad {ip}; env.created / env.deleted record the calls."""
    class Env:
        prices: dict = {}
        hosts: dict = {}
        cuda_bad: set = set()
        created: list = []
        deleted: list = []
        budget = 100.0
    env = Env()
    env.prices, env.hosts, env.cuda_bad, env.created, env.deleted = {}, {}, set(), [], []
    monkeypatch.setattr(remote, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(remote, "BAD_HOSTS", tmp_path / "bad_hosts.json")
    monkeypatch.setattr(remote, "_BROKEN_PAIRS", {})
    monkeypatch.setattr(remote, "KNOWN_BAD_SINCE", time.time() - 3600)   # the seed host's demotion is fresh in these tests
    monkeypatch.setattr(remote, "ensure_ssh_key", lambda: None)
    monkeypatch.setattr(remote, "gpu_price", lambda g, c: env.prices.get((g, c)))
    monkeypatch.setattr(remote, "remaining", lambda: env.budget)

    def ctl(*args, **kw):
        assert args[:2] == ("pod", "create"), args
        g, c = args[args.index("--gpu-id") + 1], args[args.index("--cloud-type") + 1]
        pid = f"pod{len(env.created)}"
        env.created.append((g, c, pid))
        return {"id": pid, "costPerHr": env.prices[(g, c)]}
    monkeypatch.setattr(remote, "_ctl", ctl)
    monkeypatch.setattr(remote, "_wait_running", lambda pid, t: {"id": pid})

    def pod_ssh(pj):
        g, c, _ = next(x for x in env.created if x[2] == pj["id"])
        return env.hosts.get((g, c), "198.51.100.7"), 22
    monkeypatch.setattr(remote, "_pod_ssh", pod_ssh)
    monkeypatch.setattr(remote.Pod, "run", lambda self, *a, **k: 0)
    monkeypatch.setattr(remote, "preflight", lambda p: ((False, "CUDA failed: unknown error") if p.host in env.cuda_bad
                                                        else (True, "PREFLIGHT_CUDA_OK fake")))
    monkeypatch.setattr(remote, "arm_killswitch", lambda p, mm: {"armed": True})
    monkeypatch.setattr(remote, "_record", lambda *a, **k: None)
    monkeypatch.setattr(remote, "_delete", lambda pid, reason: env.deleted.append((pid, reason)))
    return env


def _session(**kw):
    with remote.pod_session("t", **{"max_minutes": 100, "gpus": [G4090, L40S, A40], "clouds": ["COMMUNITY", "SECURE"],
                                    **kw}) as p:
        return p


def test_fast_gpu_on_secure_before_a_slower_gpu_on_community(fake_runpod):
    fake_runpod.prices = {(G4090, "COMMUNITY"): None, (G4090, "SECURE"): 0.89, (L40S, "COMMUNITY"): 0.79}
    p = _session()
    assert [c[:2] for c in fake_runpod.created] == [(G4090, "SECURE")] and p.gpu == G4090
    assert p.choice["chosen"] == {"gpu": G4090, "cloud": "SECURE", "price": 0.89, "rank": 1}
    assert "warning" not in p.choice and [r["result"] for r in p.choice["tried"]] == ["no stock", "ok"]
    assert fake_runpod.deleted == [("pod0", "session end")]


def test_max_cost_that_excludes_the_fast_gpu_is_reported(fake_runpod, capsys):
    fake_runpod.prices = {(G4090, "COMMUNITY"): 0.69, (G4090, "SECURE"): 0.89, (L40S, "COMMUNITY"): 0.79,
                          (A40, "SECURE"): 0.40}
    p = _session(max_cost=1.0)   # 106 min: 4090 $1.22, L40S $1.40, A40 $0.71
    assert p.gpu == A40
    w = p.choice["warning"]
    assert "session max $1.00 ruled out the faster RTX 4090 (COMMUNITY" in w and "--max-cost 1.22" in w and "A40" in w
    # (L40S community is the seeded broken host's pair: tried last, never reached)
    assert [(x["gpu"], x["cloud"]) for x in p.choice["excluded_faster"]] == [(G4090, "COMMUNITY"), (G4090, "SECURE")]
    assert "NOTE: the session max $1.00 ruled out" in capsys.readouterr().out
    # the same card in another cloud is not slower: no warning
    fake_runpod.prices = {(G4090, "COMMUNITY"): 0.34, (G4090, "SECURE"): 0.89}
    p = _session(max_cost=1.0, clouds=["SECURE", "COMMUNITY"])
    assert p.choice["chosen"]["cloud"] == "COMMUNITY" and p.choice["tried"][0]["result"] == "over the session max"
    assert "warning" not in p.choice


def test_no_option_under_the_max_names_what_the_max_would_need(fake_runpod):
    fake_runpod.prices = {(G4090, "COMMUNITY"): 0.69, (A40, "SECURE"): 0.59}
    with pytest.raises(RuntimeError) as ei:
        _session(max_cost=0.5)
    msg = str(ei.value)
    assert "every option exceeds the session max $0.50" in msg and "the cheapest, A40 (SECURE), needs $1.04" in msg
    assert "the fastest, RTX 4090 (COMMUNITY), $1.22" in msg and not fake_runpod.created
    assert [r["result"] for r in ei.value.choice["tried"] if r.get("price")] == ["over the session max"] * 2


def test_remaining_budget_skips_to_a_cheaper_gpu_and_says_so(fake_runpod):
    fake_runpod.prices = {(H100, "COMMUNITY"): 2.69, (A100P, "COMMUNITY"): 1.19}
    fake_runpod.budget = 1.0
    with remote.pod_session("t", 30, gpus=[H100, A100P], clouds=["COMMUNITY"]) as p:   # H100 $1.61 > $1.00 left
        assert p.gpu == A100P and "remaining Runpod budget ruled out the faster H100" in p.choice["warning"]
    fake_runpod.budget = 0.5
    with pytest.raises(remote.BudgetError, match="remaining"):                           # nothing fits: as before
        with remote.pod_session("t", 30, gpus=[H100, A100P], clouds=["COMMUNITY"]):
            pass


def test_broken_host_is_remembered_for_the_rest_of_the_session(fake_runpod, tmp_path):
    fake_runpod.prices = {(G4090, "COMMUNITY"): 0.34, (G4090, "SECURE"): 0.89, (L40S, "COMMUNITY"): 0.79}
    fake_runpod.hosts = {(G4090, "COMMUNITY"): "203.0.113.9"}
    fake_runpod.cuda_bad = {"203.0.113.9"}
    p = _session()
    assert [c[:2] for c in fake_runpod.created] == [(G4090, "COMMUNITY"), (G4090, "SECURE")] and p.gpu == G4090
    assert p.choice["tried"][0]["result"] == "broken host" and p.choice["tried"][0]["host"] == "203.0.113.9"
    saved = json.loads((tmp_path / "bad_hosts.json").read_text())["hosts"]["203.0.113.9"]
    assert saved["gpu"] == G4090 and saved["cloud"] == "COMMUNITY" and saved["fails"] == 1
    assert remote.broken_pairs()[(G4090, "COMMUNITY")] == "203.0.113.9"
    fake_runpod.created.clear()
    p2 = _session()                       # the next session of this process: that pair is tried last
    assert fake_runpod.created[0][:2] == (G4090, "SECURE") and p2.choice["demoted"][0]["host"] == "203.0.113.9"
    # a later process: the persisted deny entry demotes the pair too (and the seeded L40S community host)
    remote._BROKEN_PAIRS.clear()
    assert (G4090, "COMMUNITY") in remote.broken_pairs() and (L40S, "COMMUNITY") in remote.broken_pairs()


def test_host_already_on_the_deny_list_is_remembered_with_its_pair(fake_runpod, tmp_path):
    fake_runpod.prices = {(L40S, "COMMUNITY"): 0.79, (L40S, "SECURE"): 1.09}
    fake_runpod.hosts = {(L40S, "COMMUNITY"): "60.249.37.148"}
    remote._BROKEN_PAIRS.clear()
    (tmp_path / "bad_hosts.json").write_text(json.dumps({"hosts": {}}))
    order, demoted = remote.option_order([L40S], ["COMMUNITY", "SECURE"], remote.broken_pairs())
    assert demoted == [(L40S, "COMMUNITY")]                           # the seed carries its GPU / cloud
    with remote.pod_session("t", 30, gpus=[L40S], clouds=["COMMUNITY", "SECURE"]) as p:
        assert p.choice["chosen"]["cloud"] == "SECURE"
    assert [c[:2] for c in fake_runpod.created] == [(L40S, "SECURE")]   # never created on the broken host again


def test_run_session_records_the_gpu_choice_and_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(remote, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(rj, "_monitor_up", lambda ses: None)
    monkeypatch.setattr(remote, "record_cap_in_use", lambda sid=None: None)
    monkeypatch.setattr(rj, "snapshot", lambda *a, **k: None)
    monkeypatch.setattr(rj, "_start_puller", lambda *a, **k: None)
    monkeypatch.setattr(rj, "bundle_files", lambda jobs: [])
    bundle = tmp_path / "bundle_x.tar.gz"
    bundle.write_bytes(b"x")
    man = {"key": "x", "path": str(bundle), "bytes": 1, "raw_bytes": 1, "n_files": 1, "sha256": "s",
           "groups_raw_bytes": {}, "links": {}, "build_s": 0.0}
    monkeypatch.setattr(rj, "build_bundle", lambda files, **k: (bundle, man))
    from dancecap import progress_eta
    monkeypatch.setattr(progress_eta, "rates_rebuild", lambda *a, **k: None)
    seen = {}
    warn = "the session max $1.00 ruled out the faster RTX 4090 (COMMUNITY, $0.69/h, worst case $1.22 for 100 min)"

    @contextlib.contextmanager
    def fake_session(name, max_minutes, gpus=None, max_cost=None, clouds=None, **k):
        seen["gpus"] = gpus
        yield remote.Pod("podZ", name, A40, 0.4, "198.51.100.7", 22,
                         choice={"chosen": {"gpu": A40}, "warning": warn, "excluded_faster": [{"gpu": G4090}]})
    monkeypatch.setattr(remote, "pod_session", fake_session)

    def steps(ses, pod, todo, b, m, feats, served=None):
        for j in todo:
            ses.state["jobs"][j.name]["status"] = "ok"
    monkeypatch.setattr(rj, "_pod_steps", steps)
    res = rj.run_session([_job("s3db")], sid="20261021-210000-t", max_cost=1.0)
    st = json.loads((tmp_path / "sessions" / "20261021-210000-t" / "state.json").read_text())
    assert seen["gpus"][0] == G4090 and st["gpu_need_gb"] == 24.0 and res["error"] is None
    assert st["pod"]["choice"]["warning"] == warn
    assert st["warnings"][0]["kind"] == "gpu_choice" and st["warnings"][0]["msg"] == warn
    log = (tmp_path / "sessions" / "20261021-210000-t" / "runner.log").read_text()
    assert "WARNING (gpu_choice): the session max $1.00" in log and "GPUs for 24 GB, fastest first: RTX 4090, L40S" in log

    def no_pod(*a, **k):
        e = RuntimeError("no GPU pod could be started: every option exceeds the session max $0.50")
        e.choice = {"tried": [{"gpu": G4090, "result": "over the session max", "need": 1.22}]}
        raise e
    monkeypatch.setattr(remote, "pod_session", no_pod)
    rj.run_session([_job("s3db")], sid="20261021-210500-t", max_cost=0.5)
    st = json.loads((tmp_path / "sessions" / "20261021-210500-t" / "state.json").read_text())
    assert "session max $0.50" in st["error"] and st["gpu_choice"]["tried"][0]["need"] == 1.22


# ------------------------------------------------------------------------------------------------ 4. frame counter


S3DB_LINES = [
    "[sam3db_pod 22:25:02] model loaded in 18 s; keys {...}",
    "[sam3db_pod 22:27:24] mask255: chunk 1/3 (200 frames, 400 prompts) in 800 s = 2.000 s per person-frame; valid [200, 200]",
    "[sam3db_pod 22:29:46] mask255: chunk 2/3 (200 frames, 400 prompts) in 800 s = 2.000 s per person-frame; valid [200, 200]",
]


def test_frame_counter_sam3d_body_and_slow_gpu_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    ses = rj.Session("20261021-220000-t")
    job = _job("trial_sam3d_body", commands=[{"name": "infer", "cmd": "python sam3db_pod.py infer"}])
    fp = rj.FrameProgress(ses, job, job.commands[0], "NVIDIA A100-SXM4-80GB")
    t = 1000.0
    for i, line in enumerate(S3DB_LINES):
        fp.feed(line + "\n", now=t + 800 * i)
    doc = json.loads((ses.dir / "progress" / "frames.json").read_text())
    e = doc["jobs"]["trial_sam3d_body/infer"]
    assert (e["done"], e["total"], e["unit"], e["stage"]) == (400, 600, "frames", "chunk 2/3")
    assert e["rate_s"] == 2.0 and e["rate_unit"] == "person-frame" and e["slow_x"] == 5.63 and e["eta_s"] == 800.0
    assert e["ref"]["gpu"] == G4090 and e["gpu"] == "NVIDIA A100-SXM4-80GB" and doc["sid"] == ses.sid
    w = ses.state["warnings"]
    assert len(w) == 1 and w[0]["kind"] == "slow_gpu" and "5.6x the 0.355 s measured on RTX 4090" in w[0]["msg"]
    fp.feed("[sam3db_pod 22:32:06] mask255: chunk 3/3 (199 frames, 398 prompts) in 796 s = 2.000 s per person-frame\n",
            now=t + 2400)
    fp.close(0)
    e = json.loads((ses.dir / "progress" / "frames.json").read_text())["jobs"]["trial_sam3d_body/infer"]
    assert (e["done"], e["total"], e["state"], e["rc"]) == (599, 599, "done", 0) and len(ses.state["warnings"]) == 1
    evs, _ = P.read_jsonl(ses.dir / "progress" / "runner.jsonl")
    mine = [x for x in evs if x.get("src") == "frames"]
    assert mine[0]["kind"] == "start" and mine[-1]["kind"] == "end" and mine[-1]["cmd"] == "infer"
    assert mine[-1]["job"] == "trial_sam3d_body" and mine[-1]["done"] == 599 and mine[-1]["unit"] == "frames"
    assert any(x["kind"] == "note" and x["step"] == "slow_gpu" for x in evs)


def test_frame_counter_mamma_iterations_from_the_relayed_log(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    ses = rj.Session("20261021-220500-t")
    job = _job("trial_fit", needs=["mamma"], commands=[{"name": "fit", "cmd": "python -m dancecap mamma fit t_tj --tag c1"}])
    fp = rj.FrameProgress(ses, job, job.commands[0], G4090)
    w = "[watch ma_3d_c1.log] "
    seq = [(0, "[mamma] mamma ma_3d: python run_ma_3d.py ..."), (1, w + "sequence length:  599"), (2, w + "RUN:  first_run"),
           (3, w + "Iter. 1 of 50"), (20, w + "Iter. 10 of 50"), (40, w + "Iter. 20 of 50"),
           (44, w + "Stopping early at iteration 22 due to lack of improvement."), (45, w + "RUN:  second_run"),
           (50, w + "Iter. 1 of 500"), (64.4, w + "Iter. 10 of 500"), (80.4, w + "Iter. 20 of 500")]
    for dt, line in seq:
        fp.feed(line, now=5000.0 + dt)
    e = json.loads((ses.dir / "progress" / "frames.json").read_text())["jobs"]["trial_fit/fit"]
    assert (e["src"], e["unit"], e["stage"], e["stages_total"], e["frames"]) == ("mamma_iter", "iters", "run2", 4, 599)
    assert (e["done"], e["total"], e["state"]) == (20, 500, "running")
    assert e["rate_s"] == 1.6 and e["rate_avg_s"] == 1.6 and e["eta_s"] == pytest.approx(480 * 1.6)
    assert e["slow_x"] is None and "warnings" not in ses.state    # no reference rate for MAMMA iterations
    fp.close(1)
    e = json.loads((ses.dir / "progress" / "frames.json").read_text())["jobs"]["trial_fit/fit"]
    assert e["state"] == "failed" and e["eta_s"] is None


def test_frame_counter_tqdm_and_garbage_never_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    ses = rj.Session("20261021-221000-t")
    job = _job("front")
    fp = rj.FrameProgress(ses, job, job.commands[0], G4090)
    fp.feed(" 40%|####      | 240/599 [00:12<00:18, 19.8it/s]\n", now=10.0)
    fp.feed("\x00\xff garbage [sam3db_pod x] chunk", now=11.0)
    fp.feed(None)   # a broken caller: swallowed
    e = json.loads((ses.dir / "progress" / "frames.json").read_text())["jobs"]["front/cmd1"]
    assert (e["src"], e["done"], e["total"]) == ("tqdm", 240, 599)


def test_run_job_wires_the_counter_into_pod_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    monkeypatch.setattr(rj, "snapshot", lambda *a, **k: None)
    ses = rj.Session("20261021-223000-t")
    job = _job("trial_sam3d_body", extract="session", outputs=["dancecap/work/t/pod_out/"],
               commands=[{"name": "infer", "cmd": "python sam3db_pod.py infer"}])
    ses.state["jobs"][job.name] = {"status": "pending"}
    arch = tmp_path / "out.tar.gz"
    with tarfile.open(arch, "w:gz") as t:
        ti = tarfile.TarInfo("dancecap/work/t/pod_out/a.json")
        ti.size = 2
        t.addfile(ti, io.BytesIO(b"{}"))
    sha = hashlib.sha256(arch.read_bytes()).hexdigest()

    class Pod:
        gpu = "NVIDIA A100-SXM4-80GB"

        def run(self, cmd, check=True, timeout=None, log=None, echo=True, on_line=None):
            for line in S3DB_LINES:
                on_line(line + "\n")
            return 0

        def output(self, cmd, timeout=0):
            return f"SIZE {arch.stat().st_size}\nSHA {sha}\n"

        def get(self, remote_path, local, timeout=None):
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(arch, local)
    rj._run_job(ses, Pod(), job, ses.state["jobs"][job.name])
    assert ses.state["jobs"][job.name]["status"] == "ok"
    e = json.loads((ses.dir / "progress" / "frames.json").read_text())["jobs"]["trial_sam3d_body/infer"]
    assert e["done"] == 400 and e["state"] == "ended" and e["rc"] == 0 and e["gpu"] == Pod.gpu
    assert ses.state["warnings"][0]["kind"] == "slow_gpu"


def test_pod_run_hands_every_line_to_on_line(monkeypatch):
    if not BASH:
        pytest.skip("needs bash")
    p = remote.Pod("p", "n", "g", 0.0, "localhost", 22)
    monkeypatch.setattr(remote.Pod, "_ssh_base", lambda self: [BASH, "--noprofile", "--norc", "-c"])
    got = []

    def cb(line):
        got.append(line.strip())
        raise ValueError("a broken counter")
    rc = p.run("echo one; echo two", check=False, echo=False, on_line=cb)
    assert rc == 0 and [x for x in got if x] == ["one", "two"]      # an on_line error never ends the command


def test_mamma_fit_commands_relay_their_ma3d_log():
    job = _job("fit", needs=["mamma"], commands=[{"name": "fit", "cmd": "python -m dancecap mamma fit t_tj0000 --tag c1"},
                                                 {"name": "select", "cmd": "python -m dancecap mamma select t_tj0000"}],
               watch=["dancecap/work/t/logs/extra_*.log"])
    assert rj.watch_globs(job, job.commands[0]) == ["dancecap/work/t/logs/extra_*.log",
                                                    "dancecap/work/t_tj0000/mamma/logs/ma_3d_*.log"]
    assert rj.watch_globs(job, job.commands[1]) == ["dancecap/work/t/logs/extra_*.log"]
    c = rj.command_script(job, job.commands[0], "dancecap/work/runpod/sessions/s/pod_logs/fit")
    assert "tail -n +1 -F --pid=$$" in c and "/workspace/HeadMovement/dancecap/work/t_tj0000/mamma/logs/ma_3d_*.log" in c
    assert c.index("_wm=$(mktemp)") < c.index("timeout --kill-after")     # relay starts before the command
    assert "grep -qi mawk && A='awk -W interactive'" in c   # Ubuntu's mawk reads pipes in blocks (live probe 1: one burst)
    assert "_wm=" not in rj.command_script(_job("plain"), _job("plain").commands[0], "dancecap/work/runpod/sessions/s/pod_logs/p")
    plain = _job("p2")
    d = rj.asdict(plain)
    d.pop("source")
    d.pop("watch")   # an unset `watch` leaves the fingerprint as it was before the field existed (session resume)
    for k in ("base", "cpu", "min_ram_gb", "min_vcpu", "models"):   # the same for the fields of 2026-10-22
        d.pop(k)
    for cd in d["commands"]:
        cd.pop("early")
    assert plain.fingerprint() == hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    assert _job("p2", watch=["dancecap/work/t/x.log"]).fingerprint() != plain.fingerprint()
    if BASH:
        assert subprocess.run([BASH, "-n", "-c", c]).returncode == 0


@pytest.mark.skipif(not BASH, reason="needs bash")
def test_watch_relay_follows_only_new_logs_and_ends_with_the_command(tmp_path):
    logs = tmp_path / "dancecap" / "work" / "t" / "mamma" / "logs"
    logs.mkdir(parents=True)
    (logs / "ma_3d_old.log").write_text("Iter. 5 of 50 | old\n")
    time.sleep(1.1)
    relay = rj.watch_script(["dancecap/work/t/mamma/logs/ma_3d_*.log"], every_s=1, root=_bash_path(tmp_path))
    new = _bash_path(logs) + "/ma_3d_c1.log"
    script = (relay + "sleep 1.5; printf 'sequence length:  599\\nRUN:  first_run\\nIter. 1 of 50 | L\\nIter. 2 of 50 | L\\n' "
              f"> {new}; sleep 1.5; printf 'Iter. 10 of 50 | L\\nIter. 22 of 50 | L\\nStopping early at iteration 22\\n' "
              f">> {new}; sleep 2.5; echo main-done")
    t0 = time.time()
    p = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60)
    lines = [x for x in p.stdout.splitlines() if x.strip()]
    assert lines == ["[watch ma_3d_c1.log] sequence length:  599", "[watch ma_3d_c1.log] RUN:  first_run",
                     "[watch ma_3d_c1.log] Iter. 1 of 50", "[watch ma_3d_c1.log] Iter. 10 of 50",
                     "[watch ma_3d_c1.log] Stopping early at iteration 22", "main-done"], p.stdout
    assert time.time() - t0 < 20


# ------------------------------------------------------------------------------------------------ integration (round 2 review)


def _tqdm(d: int, t: int) -> str:
    return f"{100 * d // t:3d}%|####| {d}/{t} [00:01<00:09, 1.00it/s]\n"


def _counting_writes(monkeypatch) -> list:
    n = []
    real = P.write_json_atomic

    def w(p, doc, *a, **k):
        n.append(p)
        return real(p, doc, *a, **k)
    monkeypatch.setattr(P, "write_json_atomic", w)
    return n


def test_second_tqdm_bar_is_a_new_count_not_a_stream_of_ends(tmp_path, monkeypatch):
    """Review: after a command's first bar reached its total, every later line was an unthrottled 'end' event plus a forced
    frames.json write, and the dashboard read 'done' while the command ran on."""
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    writes = _counting_writes(monkeypatch)
    ses = rj.Session("20261021-230000-t")
    job = _job("front")
    fp = rj.FrameProgress(ses, job, job.commands[0], G4090)
    t = 100.0
    for d in range(1, 11):
        fp.feed(_tqdm(d, 10), now=t + d)
    fp.feed(_tqdm(10, 10), now=t + 10.5)                      # tqdm prints its last line again when the bar closes
    for d in range(1, 501):
        fp.feed(_tqdm(d, 1000), now=t + 20 + 0.1 * d)
    e = fp.entry                                               # the file lags by the 2-s write throttle
    assert (e["state"], e["done"], e["total"]) == ("running", 500, 1000)
    assert e["rate_avg_s"] == pytest.approx(0.1, abs=1e-3) and e["eta_s"] == pytest.approx(50.0, abs=0.5)
    f = json.loads((ses.dir / "progress" / "frames.json").read_text())["jobs"]["front/cmd1"]
    assert f["state"] == "running" and f["total"] == 1000 and f["done"] >= 480
    evs, _ = P.read_jsonl(ses.dir / "progress" / "runner.jsonl")
    kinds = [x["kind"] for x in evs if x.get("src") == "frames"]
    assert kinds.count("start") == 2 and kinds.count("end") == 1 and len(kinds) < 12
    assert len(writes) < 40                                    # throttled (2 s), plus one forced write per finished count


def test_nested_tqdm_bars_resume_the_outer_count(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    ses = rj.Session("20261021-230100-t")
    job = _job("front")
    fp = rj.FrameProgress(ses, job, job.commands[0], G4090)
    t = 0.0
    fp.feed(_tqdm(0, 5), now=t)
    for o in range(1, 6):
        for i in range(1, 101):
            t += 0.05
            fp.feed(_tqdm(i, 100), now=t)
        t += 0.05
        fp.feed(_tqdm(o, 5), now=t)
    evs, _ = P.read_jsonl(ses.dir / "progress" / "runner.jsonl")
    starts = [x for x in evs if x.get("src") == "frames" and x["kind"] == "start"]
    assert len(starts) == 6 and len([x for x in evs if x.get("src") == "frames"]) < 40   # 1 outer + 5 inner bars
    e = json.loads((ses.dir / "progress" / "frames.json").read_text())["jobs"]["front/cmd1"]
    assert (e["done"], e["total"], e["state"], e["t0"]) == (5, 5, "done", 0.0)   # the outer bar resumed, never restarted


def test_a_count_a_dead_runner_left_running_is_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    sid = "20261021-230200-t"
    job = _job("trial_sam3d_body", commands=[{"name": "infer", "cmd": "python sam3db_pod.py infer"}])
    fp = rj.FrameProgress(rj.Session(sid), job, job.commands[0], "NVIDIA A100-SXM4-80GB")
    fp.feed(S3DB_LINES[1], now=1000.0)                         # the runner dies here: never closed
    e = json.loads((tmp_path / sid / "progress" / "frames.json").read_text())["jobs"]["trial_sam3d_body/infer"]
    assert e["state"] == "running" and e["eta_s"]
    rj.FrameProgress(rj.Session(sid), job, job.commands[0], G4090).close(1)   # the resumed command printed no counter line
    e = json.loads((tmp_path / sid / "progress" / "frames.json").read_text())["jobs"]["trial_sam3d_body/infer"]
    assert (e["state"], e["rc"], e["eta_s"]) == ("failed", 1, None)
    fp2 = rj.FrameProgress(rj.Session(sid), _job("other"), _job("other").commands[0], G4090)
    fp2.feed(_tqdm(3, 10), now=2000.0)
    assert rj.end_stale_frames(rj.Session(sid)) == 1             # run_session does this when it resumes a session
    e = json.loads((tmp_path / sid / "progress" / "frames.json").read_text())["jobs"]["other/cmd1"]
    assert e["state"] == "ended" and e["eta_s"] is None


def test_sam3d_body_chunks_reused_from_an_earlier_run_count_as_done(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    ses = rj.Session("20261021-230300-t")
    job = _job("s3db", commands=[{"name": "infer", "cmd": "python sam3db_pod.py infer"}])
    fp = rj.FrameProgress(ses, job, job.commands[0], G4090)
    fp.feed(S3DB_LINES[2], now=10.0)                          # chunk 1/3 was reused: no line for it
    e = json.loads((ses.dir / "progress" / "frames.json").read_text())["jobs"]["s3db/infer"]
    assert (e["done"], e["total"], e["state"]) == (400, 600, "running")
    fp.feed("[sam3db_pod 22:32:06] mask255: chunk 3/3 (199 frames, 398 prompts) in 70 s = 0.352 s per person-frame\n",
            now=80.0)
    e = json.loads((ses.dir / "progress" / "frames.json").read_text())["jobs"]["s3db/infer"]
    assert (e["done"], e["total"], e["state"]) == (599, 599, "done")


def test_physics_reached_through_counterbalance_or_quality_gets_open3d():
    for cmd in ("python -m dancecap.counterbalance t", "python -m dancecap counterbalance t",
                "python -m dancecap.pose_quality t --physics", "python -m dancecap.pose_quality t --physics --x 1"):
        assert rj.implied_features(_job("a", commands=[cmd])) == ["open3d"], cmd
    for cmd in ("python -m dancecap.pose_quality t", "python -m dancecap.synthetic_counterbalance t",
                "python -m dancecap.physics_spline x"):
        assert rj.implied_features(_job("a", commands=[cmd])) == [], cmd


def test_jobs_sized_for_one_80gb_pod_keep_the_80gb_list(tmp_path, monkeypatch):
    for cmd in ("python -m dancecap.mamma_chunks fit larissa_kadu_38 --par 6 --timeout-min 80",
                "python -m dancecap.dancers_pod run larissa_kadu_38 --workers 5", "python -m dancecap.front_pod run lk"):
        assert rj.job_vram_gb(_job("j", commands=[cmd])) == 80.0, cmd
    assert remote.gpus_for([80.0])[0] == H100
    # a single MAMMA fit over more than ~3000 frames: 48 GB (6.4 GiB per 900 frames); a chunk (--fit-frames) or a short take: 24
    (tmp_path / "takes").mkdir()
    (tmp_path / "takes" / "longtake.toml").write_text('window = [0.0, 120.0]\n[timeline]\nhz = 30.0\n')
    (tmp_path / "takes" / "short.toml").write_text('window = [0.0, 20.0]\n')
    monkeypatch.setattr(rj, "PACKAGE_ROOT", tmp_path)
    assert rj.job_vram_gb(_job("f", commands=["python -m dancecap mamma fit longtake --tag c1"])) == 48.0
    assert rj.job_vram_gb(_job("f", commands=["python -m dancecap mamma fit longtake --fit-frames 0:900"])) == 24.0
    assert rj.job_vram_gb(_job("f", commands=["python -m dancecap mamma fit short"])) == 24.0
    assert rj.job_vram_gb(_job("f", commands=["python -m dancecap mamma fit nosuchtake"])) == 24.0
    assert remote.gpus_for([48.0])[0] == L40S


def test_a_broken_pair_waits_behind_the_next_gpu_only_and_lapses(tmp_path, monkeypatch):
    ada = "NVIDIA RTX 6000 Ada Generation"
    order, demoted = remote.option_order([G4090, L40S, ada, A40], ["COMMUNITY", "SECURE"], {(L40S, "COMMUNITY"): "h"})
    assert order == [(G4090, "COMMUNITY"), (G4090, "SECURE"), (L40S, "SECURE"), (ada, "COMMUNITY"), (ada, "SECURE"),
                     (L40S, "COMMUNITY"), (A40, "COMMUNITY"), (A40, "SECURE")]
    assert demoted == [(L40S, "COMMUNITY")]
    order, _ = remote.option_order([G4090, L40S], ["COMMUNITY"], {(L40S, "COMMUNITY"): "h"})
    assert order == [(G4090, "COMMUNITY"), (L40S, "COMMUNITY")]          # the last GPU: still tried
    monkeypatch.setattr(remote, "BAD_HOSTS", tmp_path / "bad_hosts.json")
    monkeypatch.setattr(remote, "_BROKEN_PAIRS", {})
    now = time.time()
    (tmp_path / "bad_hosts.json").write_text(json.dumps({"hosts": {
        "203.0.113.5": {"fails": 3, "ok": 0, "gpu": G4090, "cloud": "SECURE", "last": now - 4 * 86400},
        "203.0.113.6": {"fails": 3, "ok": 0, "gpu": A40, "cloud": "SECURE", "last": now - 3600}}}))
    monkeypatch.setattr(remote, "KNOWN_BAD_SINCE", now - 4 * 86400)
    pairs = remote.broken_pairs(now)
    assert (A40, "SECURE") in pairs and (G4090, "SECURE") not in pairs and (L40S, "COMMUNITY") not in pairs
    assert remote.host_denied("203.0.113.5", now)             # the host itself stays denied: a pod on it is deleted


def test_every_pod_a_session_creates_is_in_its_choice(fake_runpod):
    fake_runpod.prices = {(G4090, "COMMUNITY"): 0.34, (G4090, "SECURE"): 0.89}
    fake_runpod.hosts = {(G4090, "COMMUNITY"): "203.0.113.9"}
    fake_runpod.cuda_bad = {"203.0.113.9"}
    p = _session()
    rows = [r for r in p.choice["tried"] if r.get("pod_id")]
    assert [(r["pod_id"], r["result"]) for r in rows] == [("pod0", "broken host"), ("pod1", "ok")]


def test_a_tagged_session_is_capped_at_the_runs_pod_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(remote, "LEDGER", tmp_path / "ledger.json")
    for k, v in {"_monitor_up": lambda ses: None, "snapshot": lambda *a, **k: None, "_start_puller": lambda *a, **k: None,
                 "bundle_files": lambda jobs: []}.items():
        monkeypatch.setattr(rj, k, v)
    monkeypatch.setattr(remote, "record_cap_in_use", lambda sid=None: None)
    bundle = tmp_path / "bundle_x.tar.gz"
    bundle.write_bytes(b"x")
    man = {"key": "x", "path": str(bundle), "bytes": 1, "raw_bytes": 1, "n_files": 1, "sha256": "s",
           "groups_raw_bytes": {}, "links": {}, "build_s": 0.0}
    monkeypatch.setattr(rj, "build_bundle", lambda files, **k: (bundle, man))
    from dancecap import progress_eta
    from dancecap.pipeline import analyze
    monkeypatch.setattr(progress_eta, "rates_rebuild", lambda *a, **k: None)
    monkeypatch.setenv("DANCECAP_ANALYSIS", "20261021-120000-synth")
    monkeypatch.setenv("DANCECAP_ANALYSIS_STEP", "fits")
    seen = []

    @contextlib.contextmanager
    def fake_session(name, max_minutes, gpus=None, max_cost=None, clouds=None, **k):
        seen.append(max_cost)
        yield remote.Pod("podZ", name, G4090, 0.89, "198.51.100.7", 22)
    monkeypatch.setattr(remote, "pod_session", fake_session)
    monkeypatch.setattr(rj, "_pod_steps", lambda ses, pod, todo, b, m, f: None)
    monkeypatch.setattr(analyze, "budget_left", lambda run: 0.4)
    rj.run_session([_job("fit")], sid="20261021-231000-t", max_cost=1.0)
    assert seen == [0.4]
    rj.run_session([_job("fit")], sid="20261021-231100-t")
    assert seen == [0.4, 0.4]
    monkeypatch.setattr(analyze, "budget_left", lambda run: 0.0)
    res = rj.run_session([_job("fit")], sid="20261021-231200-t", max_cost=1.0)
    assert len(seen) == 2 and "no pod limit left" in res["error"]       # no pod at all past the run's limit
    monkeypatch.setattr(analyze, "budget_left", lambda run: None)          # no limit: the session max as given
    rj.run_session([_job("fit")], sid="20261021-231300-t", max_cost=1.0)
    assert seen[-1] == 1.0


@pytest.mark.skipif(not BASH or not shutil.which("curl"), reason="needs bash + curl")
def test_bootstrap_ffmpeg_never_fetches_twice(tmp_path):
    """Review: a retried bootstrap started a second background fetch into the same directory while the first ran, and a
    failed background fetch was repeated in the foreground. Now a live fetch is waited for, a failed one goes to apt."""
    tb = tmp_path / "ffmpeg.tar.xz"
    _fake_ffmpeg_tarball(tb)
    state, binp = tmp_path / "state", tmp_path / "bin"
    state.mkdir()
    (state / "base.ok").write_text("v4\n")
    env = dict(os.environ, DANCECAP_BOOTSTRAP_STATE=state.as_posix(), DANCECAP_FFMPEG_BIN=binp.as_posix(),
               DANCECAP_WS=tmp_path.as_posix(), DANCECAP_WHEELHOUSE=(tmp_path / "wh").as_posix(),
               PY=Path(sys.executable).as_posix(), DANCECAP_FFMPEG_URLS=tb.as_uri())
    path = f'export PATH="{_bash_path(binp)}:/usr/bin:/bin:/mingw64/bin"; '
    # 1. the fetch of an earlier run is still going (its pid file): waited for, not doubled
    script = path + f'sleep 3 & echo $! > "{_bash_path(state)}/ffmpeg_fetch.pid"; bash "{BOOT.as_posix()}" ffmpeg'
    t0 = time.time()
    p = subprocess.run([BASH, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "still going" in out and "static build fetched" in out, out
    assert time.time() - t0 >= 2.0 and (binp / "ffprobe").exists()
    # 2. a failed static fetch is not repeated: straight to apt (which Git Bash does not have: the feature fails)
    for f in ("ffmpeg.ok", "ffmpeg_fetch.pid"):
        (state / f).unlink(missing_ok=True)
    shutil.rmtree(state / "ffmpeg_static", ignore_errors=True)
    shutil.rmtree(binp, ignore_errors=True)
    (state / "ffmpeg_fetch.failed").write_text("")
    p = subprocess.run([BASH, "-c", path + f'bash "{BOOT.as_posix()}" ffmpeg'], env=env, capture_output=True, text=True,
                       timeout=120)
    out = p.stdout + p.stderr
    assert "static build from" not in out and "static build fetched" not in out and "Ubuntu's apt ffmpeg" in out, out


def test_gpu_table_lists_the_ladder_in_the_order_it_is_tried(monkeypatch):
    monkeypatch.setattr(remote, "gpu_list", lambda: [
        {"gpuId": "NVIDIA A100 80GB PCIe", "memoryInGb": 80, "communityCloud": True, "secureCloud": True,
         "communityPricePerHr": 1.19, "securePricePerHr": 1.39, "stockStatus": "High"},
        {"gpuId": G4090, "memoryInGb": 24, "communityCloud": False, "secureCloud": True, "communityPricePerHr": 0.34,
         "securePricePerHr": 0.89, "stockStatus": "Low", "dataCenterAvailability": [{"dataCenterId": "EU-RO-1",
                                                                                    "stockStatus": "Low"}]}])
    rows = remote.gpu_table()
    assert [r["gpu"] for r in rows] == [g for g, _ in remote.GPU_LADDER] and rows[0]["rank"] == 1
    assert (rows[0]["gpu"], rows[0]["community"], rows[0]["secure"], rows[0]["datacenters"]) == (G4090, None, 0.89,
                                                                                                ["EU-RO-1"])
    assert next(r for r in rows if r["gpu"] == L40S)["stock"] == "not offered"
    assert [r["gpu"] for r in remote.gpu_table(min_gb=80)] == ["NVIDIA A100 80GB PCIe"]   # the old table on request


def test_framestream_says_when_a_window_ends_short(tmp_path, capsys):
    if not shutil.which("ffmpeg"):
        pytest.skip("no local ffmpeg")
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-frames:v", "12",
                    "-pix_fmt", "yuv420p", str(clip)], check=True, timeout=60)
    pts = newcam.video_pts(clip)
    more = np.concatenate([pts, pts[-1] + (np.arange(1, 13) / 30.0)])   # the table says 24 frames, the file holds 12
    fs = newcam.FrameStream(clip, more, float(more[0]), float(more[-1]), (64, 48))
    items, err = _stream_in_thread(fs)
    assert err is None and len(items) == 12 and fs.short == (12, 24)    # no ffmpeg error: a warning, not a failure
    assert "decoded 12 of 24 frames" in capsys.readouterr().err
    full = newcam.FrameStream(clip, pts, float(pts[0]), float(pts[-1]), (64, 48))
    items, err = _stream_in_thread(full)
    assert err is None and len(items) == 12 and full.short is None


def test_an_early_stop_ends_the_mamma_run_it_belongs_to(tmp_path, monkeypatch):
    """Live probe 2026-10-21 19:51 (session 20261021-194737-batch): 'Stopping early at iteration 30' (total = 30 of 50)
    was taken for a second bar: a second 'start run1', and the run's real end only when the command closed."""
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    ses = rj.Session("20261021-235000-t")
    job = _job("runner_probe", commands=[{"name": "progress", "cmd": "python probe_progress.py"}])
    fp = rj.FrameProgress(ses, job, job.commands[0], G4090)
    w = "[watch ma_3d_c1.log] "
    lines = [w + "sequence length:  599", w + "RUN:  first_run"] + [w + f"Iter. {i} of 50" for i in (1, 10, 20, 30)] + \
        [w + "Stopping early at iteration 30 due to lack of improvement."] + \
        [f"[sam3db_pod 19:51:{10 + 5 * c}] mask255: chunk {c}/3 (200 frames, 400 prompts) in 360 s = 0.900 s per "
         f"person-frame; valid [200, 200]" for c in (1, 2, 3)]
    for i, line in enumerate(lines):
        fp.feed(line, now=100.0 + 5 * i)
    fp.close(0)
    evs, _ = P.read_jsonl(ses.dir / "progress" / "runner.jsonl")
    seq = [(e["kind"], e["step"]) for e in evs if e.get("src") == "frames" and e["kind"] in ("start", "end")]
    assert seq == [("start", "run1"), ("end", "run1"), ("start", "sam3db/mask255"), ("end", "sam3db/mask255")]
    run1_end = next(e for e in evs if e.get("src") == "frames" and e["kind"] == "end" and e["step"] == "run1")
    assert (run1_end["done"], run1_end["total"]) == (30, 30)
