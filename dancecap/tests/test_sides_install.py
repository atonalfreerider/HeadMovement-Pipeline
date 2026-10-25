"""dancecap.sides.install (DESIGN 8): a motion-only install into a temporary capture - the sha256 manifest is checked, the backup is made
once and never overwritten, --check / --restore, refused while work/unity.lock is held by someone else, splats/ and capture.json
untouched; the export proof fails when a frame outside the windows changed; the install script is LF-only bash and does the same."""
import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest

from dancecap.sides import install as ins

FILES = ins.MOTION_FILES


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _capture(tmp_path, tag: bytes):
    cap = tmp_path / "StreamingAssets" / "06_Test"
    (cap / "splats").mkdir(parents=True)
    for f in FILES:
        (cap / f).write_bytes(tag + f.encode() * 50)
    (cap / "capture.json").write_text('{"name": "06_Test"}', encoding="utf-8")
    (cap / "splats" / "lead.ply").write_bytes(b"splat" * 100)
    return cap


def _udir(tmp_path, tag: bytes, size_like: dict | None = None):
    u = tmp_path / "unity_motion"
    u.mkdir(exist_ok=True)
    for f in FILES:
        b = tag + f.encode() * 50
        if size_like and f in size_like:
            b = (b * 10)[:size_like[f]]
        (u / f).write_bytes(b)
    (u / "unity_motion.json").write_text(json.dumps({"sha256": {f: _sha(u / f) for f in FILES}}), encoding="utf-8")
    return u


def _same_size(cap):
    return {f: (cap / f).stat().st_size for f in FILES}


def test_install_backup_once_check_restore(tmp_path):
    cap = _capture(tmp_path, b"OLD")
    orig = {f: (cap / f).read_bytes() for f in FILES}
    cj, sp = (cap / "capture.json").read_bytes(), (cap / "splats" / "lead.ply").read_bytes()
    u = _udir(tmp_path, b"NEW", _same_size(cap))
    lock = tmp_path / "unity.lock"
    r = ins.install(u, cap, check=True, lock=lock)
    assert r["check"] == "ok" and all(not v["identical"] for v in r["files"].values())
    assert {f: (cap / f).read_bytes() for f in FILES} == orig                      # --check changes nothing
    r = ins.install(u, cap, lock=lock)
    assert r["installed"] and r["backed_up_now"]
    assert all((cap / f).read_bytes() == (u / f).read_bytes() for f in FILES)
    assert all((u / ins.BACKUP / f).read_bytes() == orig[f] for f in FILES)
    assert (cap / "capture.json").read_bytes() == cj and (cap / "splats" / "lead.ply").read_bytes() == sp
    assert not lock.exists()                                                       # released
    # a second install (another export) never overwrites the first backup
    u2 = _udir(tmp_path, b"NWR", _same_size(cap))
    r = ins.install(u2, cap, lock=lock)
    assert not r["backed_up_now"]
    assert all((u2 / ins.BACKUP / f).read_bytes() == orig[f] for f in FILES)
    r = ins.install(u2, cap, restore=True, lock=lock)
    assert r["restored"] and all((cap / f).read_bytes() == orig[f] for f in FILES)


def test_install_refuses_a_stale_source_and_a_held_lock(tmp_path):
    cap = _capture(tmp_path, b"OLD")
    u = _udir(tmp_path, b"NEW", _same_size(cap))
    (u / "smplx_lead.bin").write_bytes(b"changed after the export")
    with pytest.raises(RuntimeError):
        ins.install(u, cap, lock=tmp_path / "unity.lock")
    u = _udir(tmp_path, b"NEW", _same_size(cap))
    lock = tmp_path / "unity.lock"
    lock.mkdir()
    (lock / "owner.txt").write_text("someone recording a film\n", encoding="utf-8")
    with pytest.raises(ins.LockHeld):
        ins.install(u, cap, lock=lock)
    assert ins.install(u, cap, check=True, lock=lock)["check"] == "ok"            # a check never needs the lock
    assert (lock / "owner.txt").exists()                                           # someone else's lock stays


def _motion(path, frames):
    from dancecap import export_unity as eu
    T = len(frames)
    eu.write_motion(path, "lead", np.arange(T) / 30.0, frames[:, :3], frames[:, 3:3 + 220], frames[:, 223:])


def test_prove_export_flags_a_frame_outside_the_windows(tmp_path):
    from dancecap import export_unity as eu
    T = 90
    rng = np.random.default_rng(0)
    base = rng.normal(size=(T, eu.FRAME_FLOATS)).astype(np.float32)
    ref, new = tmp_path / "ref", tmp_path / "new"
    ref.mkdir()
    new.mkdir()
    for f in ("smplx_lead.bin", "smplx_follow.bin"):
        _motion(ref / f, base)
    for f in ("figure1.json", "figure2.json"):
        (ref / f).write_text("{}")
        (new / f).write_text("{}")
    gate = {"windows": {"0": {"status": "adopt", "r0": 30, "r1": 50, "dancers": {"lead": [19]}}}}
    m = base.copy()
    m[35:45, 10] += 0.1
    _motion(new / "smplx_lead.bin", m)
    _motion(new / "smplx_follow.bin", base)
    p = ins.prove_export(new, ref, gate, {})
    assert p["ok"] and p["lead"]["changed_rows"] == 10
    m[70, 3] += 0.1                                                                # outside every window
    _motion(new / "smplx_lead.bin", m)
    p = ins.prove_export(new, ref, gate, {})
    assert not p["ok"] and p["lead"]["changed_rows_outside_allowed"] == 1
    # the follower changes only on spin / limit / splice frames
    _motion(new / "smplx_lead.bin", base)
    mf = base.copy()
    mf[40, 5] += 0.1
    _motion(new / "smplx_follow.bin", mf)
    p = ins.prove_export(new, ref, gate, {}, spin_frames=np.zeros(T, bool))
    assert not p["ok"]


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sha256sum") is None, reason="no bash / sha256sum")
def test_install_script_is_lf_bash_and_installs_into_a_test_capture(tmp_path):
    cap = _capture(tmp_path, b"OLD")
    orig = {f: (cap / f).read_bytes() for f in FILES}
    u = _udir(tmp_path, b"NEW", _same_size(cap))
    sh = ins.write_install_script(u, "06_Test")
    assert b"\r\n" not in sh.read_bytes()
    env = {"HM_CAPTURE_DIR": str(cap).replace("\\", "/"), "HM_UNITY_LOCK": str(tmp_path / "nolock").replace("\\", "/"),
           "PATH": __import__("os").environ["PATH"]}
    run = lambda *a: subprocess.run(["bash", str(sh).replace("\\", "/"), *a], capture_output=True, text=True, env=env)
    r = run("--check")
    assert r.returncode == 0 and "check ok" in r.stdout, r.stdout + r.stderr
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert all((cap / f).read_bytes() == (u / f).read_bytes() for f in FILES)
    assert all((u / ins.BACKUP / f).read_bytes() == orig[f] for f in FILES)
    r = run("--restore")
    assert r.returncode == 0 and all((cap / f).read_bytes() == orig[f] for f in FILES)
