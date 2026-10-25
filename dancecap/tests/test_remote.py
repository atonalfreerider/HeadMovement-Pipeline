"""Offline tests for dancecap.remote: pod-record parsing and budget accounting (no Runpod calls)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import remote  # noqa: E402

# Real shapes observed on 2026-10-06 (ids/IPs replaced by documentation values; no secrets).
MCP_SHAPE = {  # REST v2 via the Runpod MCP server, pod created WITH --ports 22/tcp
    "id": "podidexample01", "status": "RUNNING",
    "runtime": {"ports": [{"ip": "100.64.0.10", "private": 19123, "public": 60758, "type": "http"},
                          {"ip": "203.0.113.10", "private": 22, "public": 11793, "type": "tcp"}], "uptime": 226},
    "ssh": {"direct": {"command": "ssh root@203.0.113.10 -p 11793", "host": "203.0.113.10", "port": 11793,
                       "username": "root"},
            "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "podidexample01-0a1b2c3d"}},
}
RUNPODCTL_SHAPE = {  # runpodctl 2.14 `pod get`: ssh block flattened, runtime ports null
    "id": "podidexample02", "desiredStatus": "RUNNING", "runtime": {"ports": None, "uptime": None},
    "ssh": {"id": "podidexample02", "ip": "198.51.100.20", "port": 23017,
            "ssh_command": "ssh root@198.51.100.20 -p 23017"},
}
NO_TCP_SHAPE = {  # created WITHOUT --ports 22/tcp: only an internal http port + the proxy
    "id": "podidexample03", "status": "RUNNING",
    "runtime": {"ports": [{"ip": "100.64.0.10", "private": 19123, "public": 60757, "type": "http"}], "uptime": 223},
    "ssh": {"direct": None, "proxy": {"host": "ssh.runpod.io", "port": 22}},
}
LEGACY_SHAPE = {"id": "x", "runtime": {"ports": [{"ip": "1.2.3.4", "privatePort": 22, "publicPort": 40022,
                                                   "isIpPublic": True, "type": "tcp"}]}}


def test_direct_ssh_all_shapes():
    assert remote._direct_ssh(MCP_SHAPE) == ("203.0.113.10", 11793)
    assert remote._direct_ssh(RUNPODCTL_SHAPE) == ("198.51.100.20", 23017)
    assert remote._direct_ssh(LEGACY_SHAPE) == ("1.2.3.4", 40022)
    assert remote._direct_ssh(NO_TCP_SHAPE) is None


def test_internal_ip_is_not_direct_ssh():
    shape = {"ssh": {"ip": "100.64.0.10", "port": 60757}}
    assert remote._direct_ssh(shape) is None


def test_spent_counts_open_and_closed_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "LEDGER", tmp_path / "ledger.json")
    now = time.time()
    rows = [{"pod_id": "a", "cost_per_hr": 3.6, "created_ts": now - 3600, "deleted_ts": now - 3000},  # 10 min
            {"pod_id": "b", "cost_per_hr": 1.2, "created_ts": now - 600}]  # still open: 10 min so far
    (tmp_path / "ledger.json").write_text(json.dumps(rows))
    assert abs(remote.spent() - (0.6 + 0.2)) < 0.01
    monkeypatch.setenv("DANCECAP_RUNPOD_CAP", "1.0")
    assert abs(remote.remaining() - 0.2) < 0.01


def test_default_cap_is_106_7(monkeypatch):
    monkeypatch.delenv("DANCECAP_RUNPOD_CAP", raising=False)
    assert remote.DEFAULT_CAP == 106.7  # J3b 2026-10-11 (101.7 on 2026-10-09, 80 on 2026-10-07)
    assert remote.cap() == 106.7         # only when the balance was never read (conftest turns the live read off)


def test_cap_follows_the_runpod_balance(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(remote, "BALANCE_CAP", tmp_path / "balance_cap.json")
    now = time.time()
    (tmp_path / "ledger.json").write_text(json.dumps(
        [{"pod_id": "a", "cost_per_hr": 3.6, "created_ts": now - 3600, "deleted_ts": now - 3000, "cost": 0.6}]))
    calls = []

    def fake_ctl(*args, **kw):
        calls.append(args)
        return {"clientBalance": 62.7, "spendLimit": 80}

    monkeypatch.setattr(remote, "_ctl", fake_ctl)
    monkeypatch.delenv("DANCECAP_RUNPOD_CAP", raising=False)
    monkeypatch.delenv("DANCECAP_RUNPOD_BALANCE")
    assert remote.cap() == round(0.6 + 62.7 - 2.0, 2)                  # spent + balance - reserve
    assert abs(remote.remaining() - 60.7) < 0.01
    assert len(calls) == 1                                              # memoised: one runpodctl call per minute
    assert json.loads((tmp_path / "balance_cap.json").read_text())["cap"] == 61.3
    monkeypatch.setenv("DANCECAP_RUNPOD_CAP", "balance")
    assert remote.cap() == 61.3
    monkeypatch.setenv("DANCECAP_RUNPOD_CAP", "20")                      # an explicit cap is honoured as given
    assert remote.cap() == 20.0
    monkeypatch.delenv("DANCECAP_RUNPOD_CAP")

    def down(*args, **kw):
        raise RuntimeError("runpodctl user failed (1)")

    monkeypatch.setattr(remote, "_ctl", down)
    monkeypatch.setattr(remote, "_balance_memo", {})
    assert remote.cap() == 61.3                                         # Runpod unreachable: the last good read, not DEFAULT_CAP
    (tmp_path / "balance_cap.json").unlink()
    monkeypatch.setattr(remote, "_balance_memo", {})
    assert remote.cap() == remote.DEFAULT_CAP


def test_create_args_request_direct_ssh(monkeypatch):
    calls, bodies = [], []

    def fake_ctl(*args, **kw):
        calls.append(args)
        raise RuntimeError("stop after create args")

    def fake_rest(body):
        bodies.append(body)
        raise RuntimeError("stop after create body")

    monkeypatch.setattr(remote, "_ctl", fake_ctl)
    monkeypatch.setattr(remote, "_rest_create", fake_rest)
    monkeypatch.setattr(remote, "ensure_ssh_key", lambda: None)
    monkeypatch.setattr(remote, "gpu_price", lambda g, c: 1.0)
    monkeypatch.setattr(remote, "remaining", lambda: 100.0)
    try:   # community: REST with a public IP (runpodctl 3.x has no --public-ip flag)
        with remote.pod_session("t", 5, gpus=["G"], clouds=["COMMUNITY"]):
            pass
    except RuntimeError:
        pass
    assert bodies and bodies[0]["cloudType"] == "COMMUNITY" and bodies[0]["supportPublicIp"] is True and bodies[0]["ports"] == ["22/tcp"]
    assert not [a for a in calls if a[:2] == ("pod", "create")]
    try:   # secure: runpodctl with the ssh port
        with remote.pod_session("t", 5, gpus=["G"], clouds=["SECURE"]):
            pass
    except RuntimeError:
        pass
    create = next(a for a in calls if a[:2] == ("pod", "create"))
    assert "--ports" in create and create[create.index("--ports") + 1] == "22/tcp" and "--public-ip" not in create


def test_max_cost_skips_expensive_options(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "_ctl", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(remote, "_rest_create", lambda b: calls.append(("pod", "create", "--gpu-id", b["gpuTypeIds"][0]))
                        or (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(remote, "ensure_ssh_key", lambda: None)
    monkeypatch.setattr(remote, "gpu_price", lambda g, c: {"CHEAP": 1.0, "DEAR": 3.0}[g])
    monkeypatch.setattr(remote, "remaining", lambda: 100.0)
    try:  # 60 + 6 min: CHEAP worst case $1.10 <= 2.5, DEAR $3.30 > 2.5 -> never created
        with remote.pod_session("t", 60, gpus=["DEAR", "CHEAP"], clouds=["COMMUNITY"], max_cost=2.5):
            pass
    except RuntimeError:
        pass
    created = [a[a.index("--gpu-id") + 1] for a in calls if a[:2] == ("pod", "create")]
    assert created == ["CHEAP"]


# ---------------------------------------------------------------------------------------------------------------
# remote_jobs: job-spec parsing + bundle contents/layout (offline)
# ---------------------------------------------------------------------------------------------------------------
import io  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402

import pytest  # noqa: E402

from dancecap import remote_jobs as rj  # noqa: E402
from dancecap.take import WORKSPACE  # noqa: E402


def _v1(**kw):
    d = {"schema": rj.SCHEMA, "name": "j", "take": "t", "max_minutes": 10, "commands": ["echo hi"],
         "outputs": ["dancecap/work/t/out/"]}
    d.update(kw)
    return d


def test_ws_rel_normalises_and_rejects():
    assert rj.ws_rel("/workspace/HeadMovement/dancecap/work/x/") == "dancecap/work/x/"
    assert rj.ws_rel("dancecap\\work\\a.json") == "dancecap/work/a.json"
    assert rj.ws_rel("./a/./b") == "a/b"
    for bad in ("/etc/passwd", "../x", "a/../../b", "C:/x", ""):
        with pytest.raises(ValueError):
            rj.ws_rel(bad)


def test_parse_v1_defaults_and_validation():
    j = rj.parse_job(_v1(inputs=["dancecap/work/t/in.json", {"local": "C:/abs/model.npz", "pod": "smplx/m.npz"}],
                         commands=["echo a", {"name": "fit", "cmd": "python x", "timeout_min": 5, "optional": True}]))
    assert j.cwd == "dancecap" and j.extract == "workspace" and j.needs == []
    assert j.verify == ["dancecap/work/t/out/"]  # default: every output must come back
    assert j.inputs == [("dancecap/work/t/in.json", "dancecap/work/t/in.json"), ("C:/abs/model.npz", "smplx/m.npz")]
    assert [(c.name, c.timeout_min, c.optional) for c in j.commands] == [("cmd1", 10.0, False), ("fit", 5.0, True)]
    with pytest.raises(ValueError, match="features"):
        rj.parse_job(_v1(needs=["nvdiffrastx"]))
    with pytest.raises(ValueError, match="unique"):
        rj.parse_job(_v1(commands=[{"name": "a", "cmd": "x"}, {"name": "a", "cmd": "y"}]))
    with pytest.raises(ValueError, match="missing 'outputs'"):
        rj.parse_job({k: v for k, v in _v1().items() if k != "outputs"})
    with pytest.raises(ValueError, match="relative"):
        rj.parse_job(_v1(outputs=["/etc/x"]))
    with pytest.raises(ValueError, match="not a pod job"):
        rj.parse_job({"take": "t"})
    a = rj.parse_job(_v1(archives=[{"local": "x/in.tar.gz", "extract_to": "."}], after=["mamma"],
                         inputs_if_alone=["dancecap/work/t/smplx"]))
    assert a.archives == [("x/in.tar.gz", "")] and a.after == ["mamma"]
    assert a.inputs_if_alone == [("dancecap/work/t/smplx", "dancecap/work/t/smplx")]


LEGACY_MAMMA = {  # shape written by mamma_dyn.write_pod_job (2026-10-06)
    "take": "lk", "estimate_minutes": {"setup": 12, "fit_each": 10, "total_max": 50},
    "upload": {"groups": {"code": {"items": [{"local": "C:\\ws\\dancecap\\dancecap", "pod": "dancecap/dancecap"}]},
                          "inputs": {"items": [{"local": "C:\\ws\\x\\prepare.json",
                                                "pod": "dancecap/work/lk/mamma/prepare.json"}]}}},
    "setup": ["pip install smplx"], "env": {"DANCECAP_POD": "1", "PYOPENGL_PLATFORM": "egl"},
    "dry_check": "cd /workspace/HeadMovement/dancecap && python -m dancecap mamma fit lk --tag c7 --fit-frames 90:93",
    "run": ["cd /workspace/HeadMovement/dancecap && python -m dancecap mamma fit lk --tag c7",
            "cd /workspace/HeadMovement/dancecap && python -m dancecap mamma select lk --no-qa"],
    "cwd": "/workspace/HeadMovement/dancecap",
    "download": ["dancecap/work/lk/mamma/fits/", "dancecap/work/lk/mamma/logs/", "dancecap/work/lk/mamma/select.json"],
    "after_download_locally": "python -m dancecap mamma select lk",
}


def test_parse_legacy_mamma():
    j = rj.parse_job(LEGACY_MAMMA)
    assert j.legacy == "mamma-v0" and j.name == "mamma" and j.needs == ["mamma"] and j.max_minutes == 50
    assert [c.name for c in j.commands] == ["dry_check", "fit_c7", "select"]
    assert ("C:\\ws\\x\\prepare.json", "dancecap/work/lk/mamma/prepare.json") in j.inputs
    assert j.cwd == "dancecap" and j.env["PYOPENGL_PLATFORM"] == "egl"
    assert "dancecap/work/lk/mamma/logs/" not in j.verify and "dancecap/work/lk/mamma/select.json" in j.verify
    assert j.setup == []  # legacy pip lines are replaced by bootstrap feature 'mamma'


def test_parse_legacy_room_rewrites_paths():
    d = {"take": "lk", "max_minutes": 110,
         "inputs": {"code": {"local": "c.tar.gz", "pod": "/workspace/code.tar.gz"},
                    "bundle": {"local": "b.tar", "pod": "/workspace/bundle.tar",
                               "from": str(WORKSPACE / "dancecap" / "work" / "lk" / "room" / "bundle")}},
         "pod_commands": {"setup": ["pip install gsplat"],
                          "train": "cd /workspace/dancecap && python -m dancecap room lk --stages train,quest 2>&1 | "
                                   "tee -a /workspace/dancecap/work/lk/room/pod_train.log",
                          "post": "cd /workspace/dancecap && python -m dancecap room lk --stages mesh,eval",
                          "pack": "tar czf /workspace/room_out.tar.gz room"},
         "outputs": ["room/splat_full.ply", "room/room_mesh.obj", "room/pod_train.log", "review/room_renders.jpg"],
         "fallback": "run mesh,eval locally"}
    j = rj.parse_job(d)
    assert j.legacy == "room-v0" and j.needs == ["room"]
    assert j.inputs == [(d["inputs"]["bundle"]["from"], "dancecap/work/lk/room/bundle")]
    tr = j.commands[0].cmd
    assert "/workspace/HeadMovement/dancecap/work/lk/room/pod_train.log" in tr
    assert "cd /workspace/HeadMovement/dancecap " in tr
    assert "/workspace/dancecap" not in tr.replace("/workspace/HeadMovement/dancecap", "")
    assert j.commands[1].name == "post" and j.commands[1].optional
    assert j.outputs[0] == "dancecap/work/lk/room/splat_full.ply"
    assert j.verify == ["dancecap/work/lk/room/splat_full.ply"]


LEGACY_AVATAR = {
    "take": "lk", "max_minutes": 35,
    "inputs": {"texture_inputs": {"local": "C:\\ws\\avatar\\pod\\texture_inputs.tar.gz", "pod": "/workspace/t.tar.gz"},
               "smplx (standalone only)": {"local": "s.tar.gz", "pod": "/workspace/smplx.tar.gz",
                                           "from": str(WORKSPACE / "dancecap" / "work" / "lk" / "smplx")},
               "body_model (standalone only)": {"local": "m.npz", "pod": "/workspace/SMPLX_NEUTRAL.npz"}},
    "pod_commands": {"extract": "tar xzf ...", "setup": "pip install smplx",
                     "run": "cd /workspace/HeadMovement/dancecap && python -m dancecap texture lk --preset pod",
                     "pack": "tar czf ..."},
    "outputs": ["avatar/lead_albedo.png", "avatar/report.json", "avatar/pod_texture.log", "review/avatar_uv.jpg"],
}


def test_parse_legacy_avatar_and_ordering():
    av = rj.parse_job(LEGACY_AVATAR)
    assert av.legacy == "avatar-v0" and av.needs == ["texture"] and av.after == ["mamma"]
    assert av.archives == [("C:\\ws\\avatar\\pod\\texture_inputs.tar.gz", "")]
    assert av.verify == ["dancecap/work/lk/avatar/lead_albedo.png", "dancecap/work/lk/avatar/report.json"]
    ma = rj.parse_job(LEGACY_MAMMA)
    assert [j.name for j in rj.order_jobs([av, ma])] == ["mamma", "avatar"]
    # without MAMMA in the batch the local smplx/ goes up; with it, the pod's fresh fit is used
    alone = rj.effective_inputs(av, [av])
    assert "dancecap/work/lk/smplx" in [p for _, p in alone]
    chained = rj.effective_inputs(av, [ma, av])
    assert "dancecap/work/lk/smplx" not in [p for _, p in chained]
    assert ("C:\\ws\\avatar\\pod\\texture_inputs.tar.gz",
            "dancecap/work/runpod/archives/avatar/texture_inputs.tar.gz") in chained


def test_order_jobs_detects_cycles():
    a = rj.parse_job(_v1(name="a", after=["b"]))
    b = rj.parse_job(_v1(name="b", after=["a"]))
    with pytest.raises(ValueError, match="cyclic"):
        rj.order_jobs([a, b])


def _tree(root: Path):
    (root / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("print(1)\n")
    (root / "pkg" / "a.pyc").write_bytes(b"x")
    (root / "pkg" / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"x")
    (root / "pkg" / ".git").write_text("gitdir: elsewhere")
    (root / "run.sh").write_bytes(b"#!/bin/bash\r\necho ok\r\n")
    big = os.urandom(1 << 20)
    (root / "m1.npz").write_bytes(big)
    (root / "m2.npz").write_bytes(big)


def test_bundle_layout_excludes_dedupe_and_extract(tmp_path):
    s = tmp_path / "src"
    _tree(s)
    items = [(s / "pkg", "dancecap/dancecap"), (s / "run.sh", "dancecap/remote/run.sh"),
             (s / "m1.npz", "mamma/data/m.npz"), (s / "m2.npz", "smplx/m.npz"),
             (s / "pkg" / "a.py", "dancecap/dancecap/a.py")]  # same file twice: fine
    files = rj.expand(items)
    assert [f.arc for f in files] == ["dancecap/dancecap/a.py", "dancecap/remote/run.sh", "mamma/data/m.npz",
                                      "smplx/m.npz"]
    assert {f.arc: f.link for f in files}["smplx/m.npz"] == "mamma/data/m.npz"  # identical content stored once
    out, man = rj.build_bundle(files, tmp_path / "b")
    assert man["n_files"] == 4 and man["links"] == {"smplx/m.npz": "mamma/data/m.npz"}
    assert man["raw_bytes"] < 2 * (1 << 20) and out.stat().st_size < 1.2 * (1 << 20)
    # content-keyed reuse: same files -> same archive, not rebuilt
    mt = out.stat().st_mtime_ns
    out2, _ = rj.build_bundle(files, tmp_path / "b")
    assert out2 == out and out2.stat().st_mtime_ns == mt
    # extraction reproduces the workspace layout; scripts are LF + executable; the hard link resolves
    dest = tmp_path / "ws"
    with tarfile.open(out) as tar:
        assert tar.getmember("dancecap/remote/run.sh").mode == 0o755
        assert tar.getmember("smplx/m.npz").islnk()
        tar.extractall(dest)
    assert (dest / "dancecap/remote/run.sh").read_bytes() == b"#!/bin/bash\necho ok\n"
    assert (dest / "smplx/m.npz").read_bytes() == (s / "m1.npz").read_bytes()
    assert (dest / "dancecap/dancecap/a.py").exists() and not (dest / "dancecap/dancecap/a.pyc").exists()


def test_bundle_conflict_is_an_error(tmp_path):
    (tmp_path / "a").write_text("1")
    (tmp_path / "b").write_text("2")
    with pytest.raises(ValueError, match="conflict"):
        rj.expand([(tmp_path / "a", "x/f"), (tmp_path / "b", "x/f")])
    with pytest.raises(FileNotFoundError):
        rj.expand([(tmp_path / "missing", "x/g")])


@pytest.mark.skipif(not (WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz").exists() or
                    not (WORKSPACE / "mamma-turing" / "optimization").exists(), reason="local models/MAMMA absent")
def test_validation_bundle_contents():
    files = rj.bundle_files([rj.validation_job(30)])
    arcs = {f.arc for f in files}
    for need in ("dancecap/dancecap/cli.py", "dancecap/dancecap/remote_jobs.py", "dancecap/remote/bootstrap.sh",
                 "dancecap/takes/larissa_kadu.toml", "mamma-turing/optimization/run_ma_3d.py",
                 "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz",
                 "mamma/data/body_models/downsampled_verts/verts_512.pkl", "smplx/SMPLX_NEUTRAL.npz",
                 "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_MALE.npz",
                 "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_FEMALE.npz",
                 "dancecap/work/synthetic_mamma/gt120/smplx/lead.npz"):
        assert need in arcs, need
    for a in arcs:
        parts = a.split("/")
        assert "__pycache__" not in parts and ".git" not in parts and not a.endswith(".pyc")
        assert not a.startswith(("mamma-turing/docs/", "mamma-turing/gui/", "mamma-turing/data/", "dancecap/weights/"))
        assert not a.startswith("dancecap/work/") or a.startswith(("dancecap/work/synthetic_mamma/gt120/",
                                                                     "dancecap/work/runpod/wheelhouse/",
                                                                     "dancecap/work/runpod/pywheels/"))
    links = {f.arc: f.link for f in files if f.link}
    assert links.get("smplx/SMPLX_NEUTRAL.npz") == "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz"
    # a room-only batch carries no body models
    room = rj.parse_job(_v1(name="r", needs=["room"], outputs=["dancecap/work/t/room/"]))
    assert not any(f.arc.startswith(("smplx/", "mamma/data/")) for f in rj.bundle_files([room]))


def test_safe_extract_whitelist(tmp_path):
    arch = tmp_path / "o.tar.gz"
    with tarfile.open(arch, "w:gz") as tar:
        for name in ("dancecap/work/t/out/a.json", "dancecap/work/t/secret.txt", "../evil.txt",
                     "dancecap/work/runpod/sessions/s/pod_logs/j/cmd1.log"):
            ti = tarfile.TarInfo(name)
            ti.size = 1
            tar.addfile(ti, io.BytesIO(b"x"))
        ln = tarfile.TarInfo("dancecap/work/t/out/link")
        ln.type, ln.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(ln)
    got = rj.safe_extract(arch, tmp_path / "outs", tmp_path / "ws", ["dancecap/work/t/out/"],
                          "dancecap/work/runpod/sessions/s/pod_logs/j")
    assert sorted(got) == ["dancecap/work/runpod/sessions/s/pod_logs/j/cmd1.log", "dancecap/work/t/out/a.json"]
    assert (tmp_path / "outs/dancecap/work/t/out/a.json").exists()
    assert (tmp_path / "ws/dancecap/work/runpod/sessions/s/pod_logs/j/cmd1.log").exists()
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "outs/dancecap/work/t/out/link").exists()


def test_pod_shell_snippets_are_valid_bash():
    job = rj.parse_job(_v1(env={"A": "b c"}, outputs=["dancecap/work/t/out/", "dancecap/work/t/r.json"]))
    c = rj.command_script(job, job.commands[0], "dancecap/work/runpod/sessions/s/pod_logs/j")
    assert "cmd1.done" in c and "timeout --kill-after=30 600" in c and "A='b c'" in c and "PIPESTATUS[0]" in c
    p = rj.pack_command(job, "dancecap/work/runpod/sessions/s/pod_logs/j", "/workspace/out/x.tar.gz")
    assert "MISSING" in p and "SHA" in p and "dancecap/work/t/out " in p
    bash = shutil.which("bash")
    if bash:
        for script in (c, p):
            assert subprocess.run([bash, "-n", "-c", script]).returncode == 0
        boot = Path(rj.PACKAGE_ROOT) / "remote" / "bootstrap.sh"
        assert subprocess.run([bash, "-n", str(boot)]).returncode == 0
        assert rj.WHEEL_KEY in boot.read_text()


def test_estimate_and_features():
    ma = rj.parse_job(LEGACY_MAMMA)
    room = rj.parse_job(_v1(name="r", needs=["room"], max_minutes=100))
    assert rj.needed_features([ma, room]) == ["base", "mamma", "room", "check"]
    cold = rj.estimate_minutes([ma, room], 250_000_000, cached_wheels=False)
    warm = rj.estimate_minutes([ma, room], 250_000_000, cached_wheels=True)
    assert cold["bootstrap"] > warm["bootstrap"] and abs(cold["upload"] - 250 / 2.5 / 60) < 0.1
    assert cold["total"] == pytest.approx(cold["bootstrap"] + cold["upload"] + 150 + 3 + 10, abs=0.2)


def test_echo_survives_a_console_that_cannot_encode_pod_output(monkeypatch):
    import io
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="\n")  # Windows stdout to a file
    monkeypatch.setattr(sys, "stdout", out)
    remote._echo("[pod] Creating new Ultralytics Settings v0.0.6 file \u2705\n")  # used to raise and kill the session
    out.flush()
    assert raw.getvalue().decode("cp1252") == "[pod] Creating new Ultralytics Settings v0.0.6 file ?\n"


def test_job_secrets_are_pushed_and_exported_never_written(tmp_path):
    """secrets {ENV: local key file}: validated names, the value read only at push time, exported from the pod's 0600
    file in setup and command shells; the spec, its fingerprint and the scripts carry the name, never the value."""
    key = tmp_path / "hf-token.txt"
    key.write_text("  hf_exampleSECRET123 \n")
    d = {"name": "s", "take": "x", "max_minutes": 5, "commands": ["echo ok"], "outputs": ["dancecap/work/x/o/"],
         "secrets": {"HF_TOKEN": str(key)}}
    job = rj._parse_v1(d, None)
    script = rj.command_script(job, job.commands[0], "dancecap/work/runpod/sessions/s1/pod_logs/s")
    assert 'export HF_TOKEN="$(cat /workspace/.secrets/HF_TOKEN)"' in script and "hf_example" not in script
    assert "hf_example" not in json.dumps(rj.asdict(job)) and "hf_example" not in job.fingerprint()

    class FakePod:
        got = {}

        def put_secret(self, k, v):
            self.got[k] = v

    pod = FakePod()
    assert rj.push_secrets(pod, job) == ["HF_TOKEN"] and pod.got == {"HF_TOKEN": "hf_exampleSECRET123"}
    for bad in ({"bad-name": str(key)}, {"lower": str(key)}):
        try:
            rj._parse_v1({**d, "secrets": bad}, None)
            raise AssertionError("bad secret name accepted")
        except ValueError:
            pass
    missing = rj._parse_v1({**d, "secrets": {"HF_TOKEN": str(tmp_path / "nope.txt")}}, None)
    try:
        rj.push_secrets(pod, missing)
        raise AssertionError("missing key file accepted")
    except FileNotFoundError as e:
        assert "nope.txt" in str(e)
