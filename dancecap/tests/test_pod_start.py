"""Pod start (2026-10-22): the parallel boot (bootstrap + setup commands next to the upload), uv in the bootstrap, jobs that
bring their own environment, CPU pods, host RAM / vCPU minimums, the analyses' slim body models, the ready -> compute
measure, and the round-2 hand-off leftovers. Fakes only: no pod is created."""
import contextlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tarfile
import io
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import remote  # noqa: E402
from dancecap import remote_jobs as rj  # noqa: E402

BOOT = Path(rj.PACKAGE_ROOT) / "remote" / "bootstrap.sh"
BASH = shutil.which("bash")
G4090, A100S = "NVIDIA GeForce RTX 4090", "NVIDIA A100-SXM4-80GB"


def _job(name="j", **kw):
    d = {"schema": rj.SCHEMA, "name": name, "take": "t", "max_minutes": 10, "commands": ["true"],
         "outputs": [f"dancecap/work/t/{name}/"]}
    d.update(kw)
    return rj.parse_job(d, "test")


# ------------------------------------------------------------------------------------------------ 1. features / bootstrap


def test_jobs_with_their_own_env_skip_base_and_check():
    s3 = _job("s3", base=False, inputs=["dancecap/work/t/t.mp4"],
              commands=[{"name": "install", "cmd": "bash remote/cand_sam3d_body/install.sh"}])
    assert rj.needed_features([s3]) == []                               # its venv's cv2 decodes the video
    assert rj.needed_features([_job("v", base=False, commands=["ffprobe x.mp4"])]) == ["ffmpeg"]
    assert rj.needed_features([_job("tool", base=False)]) == []
    assert rj.needed_features([s3, _job("a")]) == ["base", "check"]
    assert rj.needed_features([_job("m", base=False, needs=["mamma"])]) == ["base", "mamma", "check"]   # needs torch
    assert rj.boot_args(["ffmpeg"]) == "--no-base ffmpeg" and rj.boot_args([]) == "--no-base"
    assert rj.boot_args(["base", "mamma", "check"]) == "mamma check"
    # the new fields leave every earlier fingerprint as it was (session resume)
    assert _job("p").fingerprint() == _job("p", base=True).fingerprint() != _job("p", base=False).fingerprint()
    assert _job("p", cpu=True).fingerprint() != _job("p").fingerprint()
    assert _job("p", commands=[{"name": "x", "cmd": "true", "early": True}]).fingerprint() != \
        _job("p", commands=[{"name": "x", "cmd": "true"}]).fingerprint()


def test_bootstrap_installs_with_uv_first_and_keeps_pip_as_the_fallback():
    text = BOOT.read_text()
    pipq = text.split("pipq() {")[1].split("\n}")[0]
    assert pipq.index("uvq") < pipq.rindex('"$PY" -m pip install -q "$@"')
    assert "--index-strategy unsafe-best-match" in text.split("prefetch() {")[1].split("\n}")[0]
    for f in ("base", "mamma", "open3d", "room", "gsplat", "dancers", "cameras", "ma2d"):
        assert f"PINS_{f}=(" in text, f
    assert 'if [ "${1:-}" = --no-base ]; then shift; features=("$@")' in text
    assert "apt_bg" in text.split("prefetch \"${features[@]}\"")[0].split("features=(")[-1]   # apt next to the installs
    assert "wait_bundle" in text.split("install_wheel_or_build() {")[1].split("\n}")[0]
    if BASH:
        assert subprocess.run([BASH, "-n", str(BOOT)]).returncode == 0


def _bash_fn(name: str) -> str:
    text = BOOT.read_text()
    return f"{name}() {{" + text.split(f"{name}() {{")[1].split("\n}\n")[0] + "\n}\n"


@pytest.mark.skipif(not BASH, reason="needs bash")
def test_bootstrap_waits_for_the_bundle_marker(tmp_path):
    m = tmp_path / "bundle.ready"
    script = 'log() { echo "[bootstrap] $*"; }\n' + _bash_fn("wait_bundle")
    env = dict(os.environ, DANCECAP_WAIT_BUNDLE=m.as_posix(), DANCECAP_WAIT_BUNDLE_S="20")
    p = subprocess.Popen([BASH, "-c", script + "wait_bundle"], env=env, stdout=subprocess.PIPE, text=True)
    time.sleep(1.5)
    m.write_text("")
    out = p.communicate(timeout=30)[0]
    assert "waiting for the bundle" in out and "bundle there after" in out
    env["DANCECAP_WAIT_BUNDLE"] = (tmp_path / "never").as_posix()
    env["DANCECAP_WAIT_BUNDLE_S"] = "1"
    out = subprocess.run([BASH, "-c", script + "wait_bundle"], env=env, capture_output=True, text=True, timeout=30).stdout
    assert "no bundle after 1s" in out


@pytest.mark.skipif(not BASH, reason="needs bash")
def test_bootstrap_without_base_runs_nothing_heavy(tmp_path):
    env = dict(os.environ, DANCECAP_BOOTSTRAP_STATE=(tmp_path / "state").as_posix(), DANCECAP_WS=tmp_path.as_posix(),
               DANCECAP_WHEELHOUSE=(tmp_path / "wh").as_posix(), PY=Path(sys.executable).as_posix(), DANCECAP_NO_UV="1")
    p = subprocess.run([BASH, BOOT.as_posix(), "--no-base"], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "finished rc=0" in p.stdout and "base: start" not in p.stdout
    assert not (tmp_path / "state" / "base.ok").exists()


# ------------------------------------------------------------------------------------------------ 2. the parallel boot


class FakePod:
    """Records every pod call in order; answers like a pod whose bundle is already uploaded and whose bootstrap ends rc 0."""

    def __init__(self, cpu=False, boot_rc=0, feed_fails=False):
        self.id, self.gpu, self.host, self.port, self.cpu = "podX", G4090, "198.51.100.7", 22, cpu
        self.calls: list = []
        self.boot_rc, self.feed_fails = boot_rc, feed_fails
        self.secrets: dict = {}
        self.sha = None

    def feed(self, cmd, data, timeout=120):
        self.calls.append(("feed", cmd, data))
        if self.feed_fails:
            raise RuntimeError("pod command failed (255): ssh")
        return "BOOT_PID=4242\n"

    def output(self, cmd, timeout=300):
        self.calls.append(("output", cmd))
        if "sha256sum" in cmd:
            return self.sha or ""
        if "@@DANCECAP_SPLIT@@" in cmd:
            return '{"t": 1}\n@@DANCECAP_SPLIT@@\nbase\t5\t0\n'
        return ""

    def run(self, cmd, check=True, timeout=None, log=None, echo=True, on_line=None):
        self.calls.append(("run", cmd))
        if "BOOT_RC=" in cmd and on_line:
            on_line(f"[bootstrap] base: done in 5s\n")
            on_line(f"BOOT_RC={self.boot_rc}\n")
        return 0

    def put(self, local, remote_path, timeout=None):
        self.calls.append(("put", remote_path))

    def get(self, remote_path, local, timeout=None):
        self.calls.append(("get", remote_path))
        raise RuntimeError("no file")

    def put_secret(self, name, value, timeout=60):
        self.calls.append(("secret", name))
        self.secrets[name] = value


def _session(tmp_path, monkeypatch, sid="20261022-010000-t"):
    monkeypatch.setattr(rj, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(rj, "snapshot", lambda *a, **k: None)
    ses = rj.Session(sid)
    ses.state["pod"] = {"started": time.time() - 5}
    return ses


def _man(tmp_path):
    b = tmp_path / "bundle_k1.tar.gz"
    b.write_bytes(b"x")
    return b, {"key": "k1", "bytes": 1, "sha256": "abc", "n_files": 3}


def test_bootstrap_starts_before_the_upload_and_is_waited_for_after_the_extract(tmp_path, monkeypatch):
    ses = _session(tmp_path, monkeypatch)
    pod = FakePod()
    pod.sha = "abc"                          # the pod already has the bundle: the upload is one sha256 check
    bundle, man = _man(tmp_path)
    rj._pod_steps(ses, pod, [], bundle, man, ["base", "mamma", "check"])
    kinds = [c[0] for c in pod.calls]
    feed = next(c for c in pod.calls if c[0] == "feed")
    assert kinds.index("feed") < kinds.index("output")             # boot before the upload check
    assert "DANCECAP_WAIT_BUNDLE=/workspace/.dancecap_bundle_k1.ready" in feed[1] and "nohup" in feed[1]
    assert "bootstrap.sh mamma check" in feed[1] and "tar -xzf - -C /workspace/HeadMovement" in feed[1]
    with tarfile.open(fileobj=io.BytesIO(feed[2]), mode="r:gz") as t:   # the boot kit: dancecap/remote/, LF scripts
        names = t.getnames()
        assert "dancecap/remote/bootstrap.sh" in names
        assert b"\r\n" not in t.extractfile("dancecap/remote/bootstrap.sh").read()
    runs = [c[1] for c in pod.calls if c[0] == "run"]
    ext = next(r for r in runs if "tar -xzf /workspace/upload/bundle_k1.tar.gz" in r)
    assert "touch /workspace/.dancecap_bundle_k1.ready" in ext and "--exclude=dancecap/remote" in ext
    assert runs.index(ext) < next(i for i, r in enumerate(runs) if "BOOT_RC=" in r)
    assert ses.state["boot"]["mode"] == "parallel" and ses.state["bootstrap_rc"] == 0
    assert any(t["step"] == "bootstrap" and t.get("mode") == "parallel" for t in ses.state["timings"])
    assert (ses.dir / "bootstrap_timings.tsv").read_text().startswith("base\t5\t0")
    assert any(t["step"] == "wheels_back" for t in ses.state["timings"])          # mamma brings back compiled wheels
    assert not any("bootstrap.sh mamma check 2>&1 | tee" in r for r in runs)    # no second, foreground bootstrap


def test_bootstrap_runs_after_the_upload_when_it_cannot_start_early(tmp_path, monkeypatch):
    ses = _session(tmp_path, monkeypatch)
    pod = FakePod(feed_fails=True, cpu=True)
    pod.sha = "abc"
    bundle, man = _man(tmp_path)
    rj._pod_steps(ses, pod, [], bundle, man, ["base", "open3d", "check"])
    runs = [c[1] for c in pod.calls if c[0] == "run"]
    fg = [r for r in runs if "dancecap/remote/bootstrap.sh open3d check 2>&1 | tee" in r]
    assert fg and fg[0].startswith("DANCECAP_CPU_POD=1 ") and ses.state["boot"]["mode"] == "after_upload"
    assert not any("--exclude=dancecap/remote" in r for r in runs)              # no kit: the bundle brings remote/
    assert not any(t["step"] == "wheels_back" for t in ses.state["timings"])     # no CUDA wheel feature


def test_cpu_pod_bootstrap_gets_the_cpu_env(tmp_path, monkeypatch):
    ses = _session(tmp_path, monkeypatch)
    pod = FakePod(cpu=True)
    assert rj.start_boot(ses, pod, ["base", "check"], {"key": "k"})["pid"] == "4242"
    assert "DANCECAP_CPU_POD=1 DANCECAP_WAIT_BUNDLE=" in pod.calls[0][1]


def test_setup_commands_of_own_env_jobs_start_at_pod_ready(tmp_path, monkeypatch):
    ses = _session(tmp_path, monkeypatch)
    key = tmp_path / "hf.txt"
    key.write_text("hf_secret_value\n")
    s3 = _job("s3", base=False, secrets={"HF_TOKEN": str(key)},
              commands=[{"name": "install", "cmd": "bash remote/cand_sam3d_body/install.sh"},
                        {"name": "infer", "cmd": "python infer.py"}])
    other = _job("fit", commands=[{"name": "install", "cmd": "bash x.sh"}])     # uses base: waits for the bootstrap
    assert [(j.name, c.name) for j, c in rj.early_commands([s3, other])] == [("s3", "install")]
    pod = FakePod()
    started = rj.start_early(ses, pod, [s3, other])
    pidf = started[("s3", "install")]
    assert pod.secrets == {"HF_TOKEN": "hf_secret_value"} and ses._secrets_pushed == {"s3": ["HF_TOKEN"]}
    run = next(c[1] for c in pod.calls if c[0] == "run")
    assert run.startswith("nohup bash -c ") and f"echo $! > {pidf}" in run and "remote/cand_sam3d_body/install.sh" in run
    assert "hf_secret_value" not in run                                        # the value only through ssh stdin
    # in the job: the early command is followed to its end, then command_script (its .done marker skips it); no 2nd push
    ses._early = started
    ses.state["jobs"]["s3"] = {"status": "pending"}
    pod.calls.clear()
    monkeypatch.setattr(rj, "_retry", lambda fn, **k: fn())
    rj._run_job(ses, pod, s3, ses.state["jobs"]["s3"])
    runs = [c[1] for c in pod.calls if c[0] == "run"]
    assert f"cat {pidf}" in runs[0] and "tail -n +1 -F --pid=" in runs[0] and "install.log" in runs[0]
    assert "already done" in runs[1] and not any(c[0] == "secret" for c in pod.calls)
    assert ses.state["jobs"]["s3"]["secrets"] == ["HF_TOKEN"]
    assert ses.state["ready_to_compute_s"] >= 0 and ses.state["compute_start_ts"]          # infer = the compute


def test_ready_to_compute_is_measured_once(tmp_path, monkeypatch):
    ses = _session(tmp_path, monkeypatch)
    ses.state["pod"]["started"] = time.time() - 75
    rj.mark_compute_start(ses, "j/run")
    first = ses.state["ready_to_compute_s"]
    assert 74 <= first <= 80
    rj.mark_compute_start(ses, "j/later")
    assert ses.state["ready_to_compute_s"] == first
    assert "pod ready -> compute (j/run)" in (ses.dir / "runner.log").read_text()


def test_small_bundles_are_striped_too():
    assert rj.upload_streams(1 << 20) == 1 and rj.upload_streams(7 << 20) == 4


# ------------------------------------------------------------------------------------------------ 3. CPU pods, host minimums


@pytest.fixture
def fake_rest(tmp_path, monkeypatch):
    class Env:
        bodies: list = []
        machines: list = []
        deleted: list = []
        recorded: list = []
        ctl_creates: list = []
    env = Env()
    env.bodies, env.machines, env.deleted, env.recorded, env.ctl_creates = [], [], [], [], []
    monkeypatch.setattr(remote, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(remote, "BAD_HOSTS", tmp_path / "bad_hosts.json")
    monkeypatch.setattr(remote, "_BROKEN_PAIRS", {})
    monkeypatch.setattr(remote, "ensure_ssh_key", lambda: None)
    monkeypatch.setattr(remote, "gpu_price", lambda g, c: {(G4090, "SECURE"): 0.89, (A100S, "SECURE"): 1.79}.get((g, c)))
    monkeypatch.setattr(remote, "remaining", lambda: 50.0)

    def rest(method, path, body=None, timeout=60, base=None):
        assert (method, path) == ("POST", "/pods")
        env.bodies.append(body)
        return {"id": f"pod{len(env.bodies)}", "costPerHr": 0.56 if body.get("computeType") == "CPU" else 1.79}
    monkeypatch.setattr(remote, "rest", rest)
    # the v2 catalog: at 16 vCPUs cpu5c in EUR-IS-1 only; at 8 the cpu3 flavors in EU-RO-1 too
    offers = {16: {"cpu5c": ["EUR-IS-1"], "cpu3c": ["EUR-IS-1"]}, 8: {"cpu3c": ["EU-RO-1", "US-CA-2"], "cpu5c": ["EUR-IS-1"]}}
    monkeypatch.setattr(remote, "cpu_offers", lambda n: offers.get(n, {}))

    def ctl(*args, **kw):
        env.ctl_creates.append(args)
        return {"id": f"ctl{len(env.ctl_creates)}", "costPerHr": 0.89}
    monkeypatch.setattr(remote, "_ctl", ctl)

    def wait_running(pid, t):
        m = env.machines.pop(0) if env.machines else {}
        return {"id": pid, **m}
    monkeypatch.setattr(remote, "_wait_running", wait_running)
    monkeypatch.setattr(remote, "_pod_ssh", lambda pj: ("198.51.100.9", 22))
    monkeypatch.setattr(remote.Pod, "run", lambda self, *a, **k: 0)

    def no_cuda_check(p):
        raise AssertionError("a CPU pod gets no CUDA check")
    monkeypatch.setattr(remote, "preflight", lambda p: no_cuda_check(p) if p.cpu else (True, "PREFLIGHT_CUDA_OK fake"))
    monkeypatch.setattr(remote, "arm_killswitch", lambda p, mm: {"armed": True})
    monkeypatch.setattr(remote, "_record", lambda *a, **k: env.recorded.append(a))
    monkeypatch.setattr(remote, "_delete", lambda pid, reason: env.deleted.append((pid, reason)))
    return env


def test_cpu_session_creates_a_secure_cpu_pod_through_rest(fake_rest):
    fake_rest.machines = [{"vcpuCount": 16, "memoryInGb": 32}]
    with remote.pod_session("an", 20, cpu={"vcpu": 16}) as p:
        assert p.cpu and p.gpu == "cpu:cpu5cx16" and p.machine == {"vcpu": 16.0, "ram_gb": 32.0}
    b = fake_rest.bodies[0]
    assert b["computeType"] == "CPU" and b["cpuFlavorIds"] == ["cpu5c"] and b["vcpuCount"] == 16
    assert b["dataCenterIds"] == ["EUR-IS-1"]                       # where the catalog has that flavor and size
    assert b["cloudType"] == "SECURE" and b["ports"] == ["22/tcp"] and b["volumeInGb"] == 0 and b["name"].startswith("dancecap-an-")
    assert p.choice["order"] == ["cpu:cpu5cx16 (SECURE)", "cpu:cpu3cx16 (SECURE)", "cpu:cpu5cx8 (SECURE)",
                                 "cpu:cpu3cx8 (SECURE)"]                       # the requested size first, then half
    assert p.choice["tried"][0]["price"] == pytest.approx(0.56)
    assert fake_rest.recorded[0][2] == "cpu:cpu5cx16" and fake_rest.deleted == [("pod1", "session end")]
    assert not fake_rest.ctl_creates
    assert remote.cpu_options(16, min_ram_gb=48) == ["cpu:cpu5gx16", "cpu:cpu3gx16"]   # 2 GB / vCPU flavors are too small


def test_host_minimums_go_through_rest_and_a_small_host_is_replaced(fake_rest):
    fake_rest.machines = [{"vcpuCount": 9, "memoryInGb": 155}, {"vcpuCount": 28, "memoryInGb": 117}]
    with remote.pod_session("front", 60, gpus=[A100S, G4090], clouds=["SECURE"], min_vcpu=24, min_ram_gb=96) as p:
        assert p.machine == {"vcpu": 28.0, "ram_gb": 117.0} and p.gpu == G4090
    assert all(b["minVCPUPerGPU"] == 24 and b["minRAMPerGPU"] == 96 for b in fake_rest.bodies)
    assert [b["gpuTypeIds"] for b in fake_rest.bodies] == [[A100S], [G4090]]
    assert fake_rest.deleted[0][0] == "pod1" and "too small" in fake_rest.deleted[0][1] and "9 vCPUs < 24" in fake_rest.deleted[0][1]
    assert [r["result"] for r in p.choice["tried"]] == ["host too small", "ok"]
    assert p.choice["tried"][0]["machine"] == {"vcpu": 9.0, "ram_gb": 155.0} and p.choice["needs"]["min_vcpu"] == 24
    # without minimums a GPU pod is still created through runpodctl
    with remote.pod_session("plain", 20, gpus=[G4090], clouds=["SECURE"]):
        pass
    assert fake_rest.ctl_creates and fake_rest.ctl_creates[-1][:2] == ("pod", "create")


def test_host_needs_by_kind_and_declared():
    front = _job("front", commands=[{"name": "front", "cmd": "python -m dancecap.front_full supervise"}])
    assert rj.job_host_needs(front) == {"min_vcpu": 24.0, "min_ram_gb": 96.0}
    assert rj.job_host_needs(_job("x", min_ram_gb=64)) == {"min_ram_gb": 64.0}
    assert rj.batch_host_needs([front, _job("y", min_vcpu=32)]) == {"min_vcpu": 32.0, "min_ram_gb": 96.0}
    assert rj.job_host_needs(_job("plain")) == {}


def test_cpu_jobs_get_no_cuda_wheels_and_refuse_gpu_features(tmp_path, monkeypatch):
    an = _job("an", cpu=True, needs=["mamma"])
    assert rj.wheel_policy([an])["wheelhouse"] is False and rj.wheel_policy([_job("f", needs=["mamma"])])["wheelhouse"]
    # unset: the analyses go to a CPU pod by themselves, anything else (or "cpu": false) to a GPU
    cmds = [{"name": "timing", "cmd": "python -m dancecap timing t"}, {"name": "cb", "cmd": "python -m dancecap.counterbalance detect --take t"}]
    assert rj.job_on_cpu(_job("a1", needs=["mamma"], commands=cmds))
    assert not rj.job_on_cpu(_job("a2", needs=["mamma"], commands=cmds, cpu=False))
    assert not rj.job_on_cpu(_job("a3", needs=["mamma"], commands=cmds + [{"name": "x", "cmd": "python train.py"}]))
    assert not rj.job_on_cpu(_job("a4", needs=["mamma", "gsplat"], commands=cmds))
    with pytest.raises(ValueError, match="CPU job cannot need"):
        _job("d", cpu=True, needs=["dancers"])
    monkeypatch.setattr(rj, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(rj, "_monitor_up", lambda ses: None)
    monkeypatch.setattr(remote, "record_cap_in_use", lambda sid=None: None)
    monkeypatch.setattr(remote, "pod_session", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no pod")))
    res = rj.run_session([_job("g", needs=["gsplat"])], sid="20261022-020000-t", cpu=True)
    assert "a CPU pod cannot run ['gsplat']" in res["error"]


# ------------------------------------------------------------------------------------------------ 4. body models


def test_analysis_jobs_bundle_only_the_neutral_model(tmp_path):
    cmds = [{"name": "timing", "cmd": "python -m dancecap timing t --mc 60"},
            {"name": "physics", "cmd": "python -m dancecap physics t"},
            {"name": "cb", "cmd": "python -m dancecap.counterbalance detect --take t"},
            {"name": "fc", "cmd": "python -m dancecap.floorcraft t --no-review"}]
    assert rj.job_models(_job("an", needs=["mamma"], commands=cmds)) == "neutral"
    assert rj.job_models(_job("fit", needs=["mamma"], commands=["python -m dancecap mamma fit t --tag a"])) == "all"
    assert rj.job_models(_job("an2", needs=["mamma"], commands=cmds, models="all")) == "all"
    assert rj.job_models(_job("tool", commands=cmds)) == "none"
    d = tmp_path / "smplx"
    d.mkdir()
    np.savez(d / "lead.npz", gender=np.array("male"))
    male = [{"local": str(d), "pod": "dancecap/work/t/smplx/"}]
    assert rj.job_models(_job("an3", needs=["mamma"], commands=cmds, inputs=male)) == "all"   # a male fit: all three
    arcs = [a for _, a in rj.model_items("neutral")]
    assert arcs == ["mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz", "smplx/SMPLX_NEUTRAL.npz"]
    assert rj.model_items("none") == [] and len(rj.model_items("all")) == 6


# ------------------------------------------------------------------------------------------------ 5. hand-off leftovers


def test_puller_tails_the_mamma_log_of_the_take_each_fit_names():
    from dancecap import progress_pull as PP
    tails = PP.default_tails([{"name": "trial_fit", "take": "trial", "commands": [
        {"cmd": "python -m dancecap mamma fit trial_tj0000 --tag a"}, {"cmd": "python -m dancecap mamma select trial_tj0000"}]}])
    assert [t["glob"] for t in tails] == ["/workspace/HeadMovement/dancecap/work/trial_tj0000/mamma/logs/ma_3d_*.log"]
    assert tails[0]["job"] == "trial_fit"


def test_face_pod_is_secure_only_and_front_full_uses_the_bootstrap_ffmpeg():
    from dancecap import face_pod, front_full
    assert face_pod.CLOUDS == ["SECURE"]
    src = inspect.getsource(front_full.job) if hasattr(front_full, "job") else inspect.getsource(front_full)
    assert "apt-get install -y -qq --no-install-recommends ffmpeg" not in src
    assert '"needs": ["dancers", "cameras", "ma2d", "ffmpeg"]' in inspect.getsource(front_full)


def test_gpus_help_names_the_ladder():
    from dancecap import cli
    assert "GPU_PREFERENCE" not in inspect.getsource(cli).split('"--gpus"')[1].split(")")[0]


def test_heartbeat_is_started_once_for_marks_seconds_apart(tmp_path):
    from dancecap.pipeline import analyze as A, paths
    sys.path.insert(0, str(Path(__file__).parent))
    from _pipeline_fixtures import make_root
    make_root(tmp_path)
    try:
        run = "20261022-010000-synth"
        A.run_dir(run).mkdir(parents=True, exist_ok=True)
        spawned = []
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        try:
            def spawn(argv):
                spawned.append(argv)
                return child.pid
            assert A.ensure_heartbeat(run, spawn=spawn) is True
            assert A.ensure_heartbeat(run, spawn=spawn) is False          # the first child has not registered yet
            hb = json.loads((A.run_dir(run) / "heartbeat.json").read_text())
            assert hb["spawn"]["pid"] == child.pid and len(spawned) == 1 and spawned[0][-2:] == ["heartbeat", run]
        finally:
            child.kill()
            child.wait()
        assert A.ensure_heartbeat(run, spawn=spawn) is True               # the spawned one died: start another
        assert A.ensure_heartbeat(run) is False                           # never a real process from a test
    finally:
        paths.set_root(None)


def test_detached_children_append_to_one_log_without_overwriting(tmp_path):
    from dancecap.pipeline.util import append_fd
    log = tmp_path / "logs" / "heartbeat.log"
    log.parent.mkdir()
    log.write_text("first line\n")
    fds = [append_fd(log), append_fd(log)]          # both opened before either child writes (the 2026-10-21 race)
    try:
        for i, fd in enumerate(fds):
            subprocess.run([sys.executable, "-c", f"print('child {i} ' + 'x' * {40 - 10 * i})"], stdout=fd,
                           stderr=subprocess.STDOUT, check=True)
    finally:
        for fd in fds:
            os.close(fd)
    lines = log.read_text().splitlines()
    assert lines[0] == "first line" and lines[1].startswith("child 0 ") and lines[2].startswith("child 1 ")


# ------------------------------------------------------------------------------------------------ 6. the volume route


def test_volume_options_go_first_within_a_gpu_type_and_for_every_cpu_flavor(fake_rest, monkeypatch):
    order = [(G4090, "COMMUNITY"), (G4090, "SECURE"), (A100S, "SECURE")]
    assert remote.with_volume(order) == [(G4090, remote.VOL_CLOUD), (G4090, "COMMUNITY"), (G4090, "SECURE"),
                                         (A100S, remote.VOL_CLOUD), (A100S, "SECURE")]
    monkeypatch.setattr(remote, "gpu_list", lambda: [{"gpuId": G4090, "dataCenterAvailability": [
        {"dataCenterId": "EU-RO-1", "stockStatus": "Medium"}]}, {"gpuId": A100S, "dataCenterAvailability": [
        {"dataCenterId": "US-KS-2", "stockStatus": "Low"}]}])
    assert remote.stock_in(G4090, "EU-RO-1") and not remote.stock_in(A100S, "EU-RO-1")
    vol = {"id": "vol1", "dc": "EU-RO-1", "mount": "/hmvol", "root": "hm"}
    with remote.pod_session("an", 20, cpu={"vcpu": 16}, volume=vol) as p:
        assert p.volume == vol and p.gpu == "cpu:cpu3cx8"
    b = fake_rest.bodies[0]
    assert b["networkVolumeId"] == "vol1" and b["dataCenterIds"] == ["EU-RO-1"] and b["volumeMountPath"] == "/hmvol"
    # the volume's data center first (only cpu3c at 8 vCPUs is offered there), then the plain options
    assert p.choice["order"][0] == f"cpu:cpu3cx8 ({remote.VOL_CLOUD})" and p.choice["order"][1] == "cpu:cpu5cx16 (SECURE)"
    # a GPU without stock in the volume's data center skips its volume option without a create call
    with remote.pod_session("fit", 20, gpus=[A100S], clouds=["SECURE"], volume=vol) as p:
        assert p.volume is None
    assert p.choice["tried"][0]["result"] == "no stock" and p.choice["tried"][1]["result"] == "ok"
    assert fake_rest.ctl_creates[-1][:2] == ("pod", "create") and "--network-volume-id" not in fake_rest.ctl_creates[-1]


def test_volume_split_moves_the_body_models_out_of_the_bundle(tmp_path, monkeypatch):
    from dancecap.farm import volume as fv
    monkeypatch.setattr(fv, "SHA_CACHE", tmp_path / "sha_cache.json")
    big = tmp_path / "SMPLX_NEUTRAL.npz"
    big.write_bytes(os.urandom(9 << 20))
    small = tmp_path / "a.json"
    small.write_text("{}")
    files = [rj.BundleFile("mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz", str(big), 9 << 20, 1),
             rj.BundleFile("smplx/SMPLX_NEUTRAL.npz", str(big), 9 << 20, 1,
                           link="mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz"),
             rj.BundleFile("dancecap/work/t/big.npz", str(big), 9 << 20, 1),          # no volume path: stays
             rj.BundleFile("mamma/data/body_models/downsampled_verts/x.npz", str(small), 2, 1)]   # small: stays
    main, served = rj.volume_split(files)
    assert [f.arc for f in main] == ["dancecap/work/t/big.npz", "mamma/data/body_models/downsampled_verts/x.npz"]
    assert [(d["arc"], d["vol"]) for d in served] == [
        ("mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz",
         "models/mamma_data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz"),
        ("smplx/SMPLX_NEUTRAL.npz", "models/smplx/SMPLX_NEUTRAL.npz")]
    assert served[0]["sha256"] == served[1]["sha256"] == fv.sha256_file(big)
    # the slim ma_2d copy maps to the farm's weights/mamma/ma_2d/
    assert rj.volume_path("mamma/data/weights/ma_2d/mamma_mask_full_cvpr.ckpt") == "weights/mamma/ma_2d/mamma_mask_full_cvpr.ckpt"
    script = rj.link_script(served, {"mount": "/hmvol", "root": "hm"})
    assert "sha256sum /hmvol/hm/models/smplx/SMPLX_NEUTRAL.npz" in script and "ln -sfn" in script
    assert script.count("echo MISS") == 2 and served[0]["sha256"] in script
    vol = rj.runner_volume()
    assert vol is None or (vol["mount"] == "/hmvol" and vol["root"] == "hm")
    monkeypatch.setenv("DANCECAP_RUNNER_VOLUME", "0")
    assert rj.runner_volume() is None


def test_files_missing_on_the_volume_come_in_a_side_bundle(tmp_path, monkeypatch):
    ses = _session(tmp_path, monkeypatch)
    monkeypatch.setattr(rj, "BUNDLES", tmp_path / "bundles")
    f1, f2 = tmp_path / "m1.npz", tmp_path / "m2.npz"
    f1.write_bytes(b"1" * 100)
    f2.write_bytes(b"2" * 100)
    served = [{"arc": "smplx/SMPLX_NEUTRAL.npz", "vol": "models/smplx/SMPLX_NEUTRAL.npz", "local": str(f1), "size": 100,
               "sha256": "a" * 64},
              {"arc": "mamma/data/weights/ma_2d/x.ckpt", "vol": "weights/mamma/ma_2d/x.ckpt", "local": str(f2), "size": 100,
               "sha256": "b" * 64}]
    pod = FakePod()
    pod.volume = {"mount": "/hmvol", "root": "hm"}
    pod.output = lambda cmd, timeout=300: (pod.calls.append(("output", cmd)),
                                           "LINKED smplx/SMPLX_NEUTRAL.npz\nMISS mamma/data/weights/ma_2d/x.ckpt\n"
                                           if "sha256sum /hmvol" in cmd else "")[1]
    ups = []
    monkeypatch.setattr(rj, "upload_bundle", lambda ses, pod, b, man, step: ups.append((b.name, step)))
    res = rj.place_served(ses, pod, served)
    assert res["linked"] == 1 and res["uploaded"] == 1
    ext = [c[1] for c in pod.calls if c[0] == "run" and "tar -xzf /workspace/upload/bundle_" in c[1]]
    assert ups and ups[0][1] == "side_upload" and ext                 # the side bundle went up and was extracted
    with tarfile.open(next((tmp_path / "bundles").glob("bundle_*.tar.gz"))) as t:
        assert t.getnames() == ["mamma/data/weights/ma_2d/x.ckpt"]
    # a pod without the volume uploads them all
    pod2 = FakePod()
    pod2.volume = None
    assert rj.place_served(ses, pod2, served) == {"linked": 0, "uploaded": 2, "side_mb": pytest.approx(0.0, abs=0.01)}
