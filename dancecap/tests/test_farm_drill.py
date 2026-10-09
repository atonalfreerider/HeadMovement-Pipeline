"""The farm drill (farm/drill.py): window slicing of prepared MAMMA inputs and the pod-vs-reference comparison. Synthetic data only."""
import json

import numpy as np
import pytest

from dancecap.farm import drill


def _prepared(root, T=60):
    m = root / "larissa_kadu_38" / "mamma"
    seq = "larissa_kadu_38_c8"
    t = [30.0 + i / 30 for i in range(T)]
    prep = {"take": "larissa_kadu_38", "world_frame": True, "grid": {"t0": t[0], "T": T, "hz": 30.0, "t": t, "trim": {}},
            "sync": {"x": 1}, "ma_2d": {}, "config": "ma3d_config.yaml", "config_overrides": {}, "base_config": "b.yaml",
            "sets": {"c8": {"seq": seq, "cams": ["01", "02"], "inputs": {"01": {}, "02": {}}},
                     "h03": {"seq": "larissa_kadu_38_h03", "cams": ["01"], "inputs": {}}},
            "eval": {"seq": seq, "cams": ["01", "02"]}}
    m.mkdir(parents=True)
    (m / "prepare.json").write_text(json.dumps(prep))
    (m / "ma3d_config.yaml").write_text("n_betas: 16\n")
    for c in ("01", "02"):
        d2 = m / "ma_2d" / seq
        d2.mkdir(parents=True, exist_ok=True)
        np.savez(d2 / f"cam{c}.npz", landmarks=np.arange(T * 2 * 4 * 3, dtype=np.float32).reshape(T, 2, 4, 3),
                 visibilities=np.ones((T, 2, 4), np.float32))
        dg = m / "ma_cap" / seq / "gt"
        dg.mkdir(parents=True, exist_ok=True)
        np.savez(dg / f"cam{c}.npz", cam_int=np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)), t=np.asarray(t),
                 frame_start=np.asarray(0, np.int32), frame_end=np.asarray(T, np.int32), cam_name=np.asarray(f"cam{c}"),
                 distortion_coeffs=np.zeros(4))
    np.savez(m / "ma_cap" / seq / "gt" / "global.npz", frames_len=np.asarray(T, np.int32), fps=np.asarray(30, np.int32),
             frame_start=np.asarray(0, np.int32), frame_end=np.asarray(T, np.int32), people_len=np.asarray(2, np.int32),
             seq_name=np.asarray(seq), t=np.asarray(t))
    return t


def test_slice_take_is_an_exact_window_of_every_array(tmp_path, monkeypatch):
    t = _prepared(tmp_path)
    monkeypatch.setattr(drill, "work_dir", lambda take: tmp_path / take)
    info = drill.slice_take("dra", 10, 20)
    assert info["take"] == "farmdrill_dra" and info["frames"] == [10, 30] and info["cams"] == ["01", "02"]
    root = tmp_path / "farmdrill_dra" / "mamma"
    prep = json.loads((root / "prepare.json").read_text())
    assert prep["grid"]["T"] == 20 and prep["grid"]["t0"] == t[10] and prep["grid"]["t"] == t[10:30]
    assert set(prep["sets"]) == {"c8"} and prep["sets"]["c8"]["seq"] == "farmdrill_dra_c8" and prep["eval"]["seq"] == "farmdrill_dra_c8"
    full = np.load(tmp_path / "larissa_kadu_38" / "mamma" / "ma_2d" / "larissa_kadu_38_c8" / "cam02.npz")["landmarks"]
    got = np.load(root / "ma_2d" / "farmdrill_dra_c8" / "cam02.npz")["landmarks"]
    assert got.shape[0] == 20 and np.array_equal(got, full[10:30])
    gt = np.load(root / "ma_cap" / "farmdrill_dra_c8" / "gt" / "cam01.npz")
    assert gt["cam_int"].shape[0] == 20 and int(gt["frame_start"]) == 0 and int(gt["frame_end"]) == 20 and gt["distortion_coeffs"].shape == (4,)
    g = np.load(root / "ma_cap" / "farmdrill_dra_c8" / "gt" / "global.npz")
    assert int(g["frames_len"]) == 20 and int(g["frame_end"]) == 20 and str(g["seq_name"]) == "farmdrill_dra_c8" and len(g["t"]) == 20
    assert (root / "ma3d_config.yaml").read_text() == "n_betas: 16\n"


def test_slice_take_refuses_a_window_outside_the_grid(tmp_path, monkeypatch):
    _prepared(tmp_path)
    monkeypatch.setattr(drill, "work_dir", lambda take: tmp_path / take)
    with pytest.raises(SystemExit):
        drill.slice_take("x", 50, 20)


def _motion(path, t, joints, reproj, betas):
    path.mkdir(parents=True, exist_ok=True)
    for role in ("lead", "follow"):
        np.savez(path / f"{role}.npz", t=t, joints=joints, reproj_px=reproj, betas=betas)


def test_compare_measures_millimetres_and_reprojection(tmp_path):
    T = 50
    t = np.arange(T) / 30.0
    rng = np.random.default_rng(1)
    ref_j = rng.normal(size=(T, 55, 3))
    betas = np.linspace(-1, 1, 16)
    _motion(tmp_path / "ref", t, ref_j, np.full(T, 5.0), betas)
    got_j = ref_j[10:30].copy()
    got_j[..., 0] += 0.010            # a 10 mm shift on every joint
    _motion(tmp_path / "got", t[10:30], got_j, np.full(20, 5.5), betas)
    r = drill._compare(tmp_path / "got", tmp_path / "ref", 10, 20)
    for role in ("lead", "follow"):
        assert r[role]["joint_mm_median"] == pytest.approx(10.0, abs=1e-6) and r[role]["joint_mm_p95"] == pytest.approx(10.0, abs=1e-6)
        assert r[role]["reproj_px_median_got"] == 5.5 and r[role]["reproj_px_median_ref"] == 5.0 and r[role]["betas_maxabs_diff"] == 0.0
    # a window whose time axis is not the reference's is reported, not compared
    _motion(tmp_path / "shifted", t[11:31], got_j, np.full(20, 5.5), betas)
    assert "error" in drill._compare(tmp_path / "shifted", tmp_path / "ref", 10, 20)["lead"]


# ------------------------------------------------------------------------------------------------ two pod-side bugs the drill found


def test_safe_extract_writes_through_a_symlinked_work_dir(tmp_path):
    """On a pod <ws>/dancecap/work is a symlink to the local scratch disk; resolve()-based containment skipped every member there."""
    import io
    import tarfile

    from dancecap.farm import worker
    ws, scratch = tmp_path / "ws", tmp_path / "scratch"
    (ws / "dancecap").mkdir(parents=True)
    scratch.mkdir()
    try:
        (ws / "dancecap" / "work").symlink_to(scratch, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot create symlinks")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in (("dancecap/work/farmdrill_dra/mamma/prepare.json", b"{}"), ("dancecap/other/a.txt", b"a"),
                           ("../escape.txt", b"x"), ("/abs.txt", b"x")):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r") as tar:
        worker._safe_extract(tar, ws)
    assert (scratch / "farmdrill_dra" / "mamma" / "prepare.json").read_bytes() == b"{}"
    assert (ws / "dancecap" / "other" / "a.txt").read_bytes() == b"a"
    assert not (tmp_path / "escape.txt").exists() and not (ws / "abs.txt").exists()


def test_world_verify_ignores_the_workers_installed_marker(tmp_path):
    """The worker drops `.installed` into the world copy on a pod; `world.verify` must not call it a stray file."""
    from dancecap.farm import world as W
    f = tmp_path / "shared" / "x.json"
    f.parent.mkdir(parents=True)
    f.write_text("{}")
    sha, n = W.cache().sha(f, use_cache=False)
    members = {"m": {"sha256": W.member_sha([("shared/x.json", sha, n)]), "scope": "pose"}}
    man = {**W.ids(members), "members": members, "files": [{"path": "shared/x.json", "sha256": sha, "bytes": n, "member": "m"}]}
    w = W.World("t", man, tmp_path)
    assert w.verify("quick").ok
    (tmp_path / ".installed").write_text("abc")
    assert w.verify("quick").ok
    (tmp_path / "stray.txt").write_text("x")
    assert not w.verify("quick").ok
