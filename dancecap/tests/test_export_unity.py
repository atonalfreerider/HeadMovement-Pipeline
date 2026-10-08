"""Exporter checks on the synthetic take (run `python -m dancecap synthetic` first; timing/physics optional).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_export_unity.py -q
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import export_unity as eu  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402
from _needs import needs_paths  # noqa: E402

WORK = PACKAGE_ROOT / "work" / "synthetic"
# the synthetic take (git-ignored run output) and the gated SMPL-X model (never in git) are absent from a clean checkout: skip, do not fail
needs_export_inputs = needs_paths(WORK / "smplx" / "lead.npz", WORK / "smplx" / "follow.npz", eu.DEFAULT_MODEL,
                                  why="synthetic take (python -m dancecap synthetic) / gated SMPL-X model")
CAPTURE = "_pytest_export_unity"
S = np.diag([1.0, 1.0, -1.0])


def _export():
    rep = eu.export("synthetic", CAPTURE, install=False, audio_t0=-0.5)
    return rep, Path(rep["out"])


def test_quaternion_mirror_matches_matrix_conjugation():
    rng = np.random.default_rng(0)
    rv = rng.normal(size=(200, 3)) * 1.5
    q = eu.rotvec_to_quat(rv)
    from scipy.spatial.transform import Rotation
    assert np.allclose(eu.quat_to_matrix(q), Rotation.from_rotvec(rv).as_matrix(), atol=1e-9)
    Ru = eu.quat_to_matrix(eu.to_unity_quats(q))
    assert np.allclose(Ru, S @ Rotation.from_rotvec(rv).as_matrix() @ S, atol=1e-9)
    assert np.allclose(eu.rotvec_to_quat(np.zeros((1, 3))), [[0, 0, 0, 1]])


@needs_export_inputs
def test_export_files_manifest_and_times():
    rep, out = _export()
    try:
        m = json.loads((out / "capture.json").read_text())
        assert m["version"] == 3 and m["times"] == "times.json"
        assert m["smplx"] == {"lead": "smplx_lead.bin", "follow": "smplx_follow.bin"}
        assert m["smplx_skin"] == {"lead": "smplx_lead_skin.bin", "follow": "smplx_follow_skin.bin"}
        for k in ("figure1.json", "figure2.json", "video_meta.json", "audio.wav"):
            assert (out / k).exists(), k
        z = np.load(WORK / "smplx" / "lead.npz")
        times = np.array(json.loads((out / "times.json").read_text()))
        assert np.allclose(times, z["t"], atol=1e-6)
        assert m["frame_count"] == len(times) and abs(m["fps"] - 30.0) < 1e-6
        # audio sample 0 is reference t = -0.5 s -> audio time = t + 0.5
        assert abs(m["time_to_audio"] - 0.5) < 1e-9 and abs(m["audio_offset"] - (times[0] + 0.5)) < 1e-9
        beats = json.loads((out / "zouk-time-analysis.json").read_text())
        ref = json.loads((WORK / "beats.json").read_text())
        assert abs(beats[1][0] - (ref[1][0] + 0.5)) < 2e-3, (beats[:2], ref[:2])  # refined grid ~ analytic grid
        fig = json.loads((out / "figure2.json").read_text())
        jf = np.asarray(np.load(WORK / "smplx" / "follow.npz")["joints"])
        assert len(fig) == len(times) and len(fig[0]) == 24
        p = fig[100][15]
        assert np.allclose([p["x"], p["y"], p["z"]], jf[100, 15] * [1, 1, -1], atol=1e-4)
        if (WORK / "timing" / "timing.json").exists():
            tj = json.loads((out / "timing.json").read_text())
            assert tj["footsteps"]["lead"] and "async_ms" in tj["footsteps"]["follow"][0]
            if "jerk" in tj:
                assert np.asarray(tj["jerk"]["lead"]).shape == (len(times), 24)
    finally:
        shutil.rmtree(out, ignore_errors=True)


@needs_export_inputs
def test_motion_fk_and_skin_lbs_match_smplx():
    """Unity-side FK of the exported rotations reproduces the npz joints, and LBS of the exported skin reproduces
    the smplx library's posed vertices (pose blendshapes off), after undoing the Unity mirror."""
    import smplx
    import torch
    rep, out = _export()
    try:
        for role in eu.ROLES:
            h, frames = eu.read_motion(out / f"smplx_{role}.bin")
            sh, skin = eu.read_skin(out / f"smplx_{role}_skin.bin")
            assert frames.shape == (h["frames"], eu.FRAME_FLOATS) and sh["joints"] == 55
            # seam-split (textured) skin: one vertex per unique (v, vt) pair, weights/rest copied from vertex_ids
            assert sh["version"] == 2 and skin.uv is not None and skin.vertex_ids is not None
            V = len(skin.vertices)
            assert V == 11313 and skin.weights.shape == (V, 55) and skin.uv.shape == (V, 2)
            assert skin.vertex_ids.max() == 10474 and len(np.unique(skin.vertex_ids)) == 10475
            assert np.allclose(np.linalg.norm(skin.normals, axis=1), 1, atol=1e-4)
            assert np.allclose(skin.weights.sum(1), 1, atol=1e-4) and skin.parents[0] == -1
            transl = frames[:, :3].astype(float)
            quats = frames[:, 3:3 + 220].reshape(-1, 55, 4).astype(float)
            joints = frames[:, 223:].reshape(-1, 55, 3).astype(float)
            assert np.allclose(np.linalg.norm(quats, axis=-1), 1, atol=1e-5)
            P, R = eu.fk_unity(skin.joints.astype(float), skin.parents, transl, quats)
            assert np.abs(P - joints).max() < 2e-4, np.abs(P - joints).max()
            # winding: reversed for the mirror, so the signed volume of the Unity mesh stays positive
            v, f = skin.vertices.astype(float), skin.faces
            vol = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6
            assert 0.04 < vol < 0.15, vol  # m^3 of a human body
            # LBS vs the smplx library for a few frames
            z = np.load(WORK / "smplx" / f"{role}.npz")
            model = smplx.create(str(eu.DEFAULT_MODEL), model_type="smplx", gender="neutral", use_pca=False,
                                 flat_hand_mean=True, num_betas=len(z["betas"]), batch_size=1)
            model.posedirs.data.zero_()
            for i in (0, 123, 400):
                ft = lambda k: torch.tensor(np.asarray(z[k][i:i + 1], np.float32))
                with torch.no_grad():
                    o = model(betas=torch.tensor(z["betas"][None], dtype=torch.float32), global_orient=ft("global_orient"),
                              body_pose=ft("body_pose"), left_hand_pose=ft("left_hand_pose"),
                              right_hand_pose=ft("right_hand_pose"), jaw_pose=ft("jaw_pose"), transl=ft("transl"))
                ours = eu.lbs_unity(eu.Skin(v, f, skin.weights.astype(float), skin.joints.astype(float),
                                            skin.parents, "", np.zeros(0)), P, R, i) * [1, 1, -1]
                err = np.abs(ours - o.vertices[0].numpy()[skin.vertex_ids]).max()
                assert err < 1e-3, (role, i, err)
    finally:
        shutil.rmtree(out, ignore_errors=True)


@needs_export_inputs
def test_uv_split_matches_texture_stage_and_model_uvs():
    """Split mesh == avatar_uv.write_uv_mesh's (vertex_ids[ft] == f) and every split face keeps the model's vt."""
    m = np.load(eu.DEFAULT_MODEL, allow_pickle=True)
    vid, uv, faces = eu.uv_split(m["f"], m["ft"], m["vt"])
    assert np.array_equal(vid[faces], np.asarray(m["f"]))
    assert np.allclose(uv[faces], np.asarray(m["vt"])[np.asarray(m["ft"])])
    assert len(vid) == len(np.unique(np.stack([m["f"].ravel(), m["ft"].ravel()], 1), axis=0))


def test_capture_name_and_audio_rules(tmp_path):
    import pytest
    for bad in ("..", ".", "/abs", "a/b", "a\b", "", "C:x"):
        with pytest.raises(ValueError):
            eu.validate_capture_name(bad)
    assert eu.validate_capture_name("02_LarissaKadu") == "02_LarissaKadu"
    with pytest.raises(ValueError):  # --audio without --audio-t0
        eu.resolve_audio(tmp_path, tmp_path / "x.wav", None)
    with pytest.raises(ValueError):  # nothing to play and no --no-audio
        eu.resolve_audio(tmp_path, None, None)
    assert eu.resolve_audio(tmp_path, None, None, allow_no_audio=True) == (None, 0.0, "none")
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "audio_overlap.wav").write_bytes(b"RIFF")
    (tmp_path / "audio" / "audio_overlap.json").write_text(json.dumps({"reference_start": 29.97}))
    wav, t0, src = eu.resolve_audio(tmp_path, None, None)
    assert wav.name == "audio_overlap.wav" and t0 == 29.97
    (tmp_path / "audio.wav").write_bytes(b"RIFF")
    import shutil as sh
    sh.rmtree(tmp_path / "audio")
    with pytest.raises(ValueError):  # real take audio.wav without a start time
        eu.resolve_audio(tmp_path, None, None)


def test_stale_physics_and_kinematics_are_skipped():
    """A physics/timing run on another grid must not be exported (np.interp / Nearest would clamp silently)."""
    times = np.arange(30) / 30.0
    w: list = []
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ph = Path(d) / "physics.json"
        ph.write_text(json.dumps({"t": (times + 0.5).tolist()}))
        assert not eu._physics_on_grid(ph, times, w) and w
        ph.write_text(json.dumps({"t": times.tolist()}))
        assert eu._physics_on_grid(ph, times, w)
        kin = Path(d) / "kinematics.npz"
        np.savez(kin, lead_t=times[5:], lead_jerk=np.ones((25, 55, 3)), follow_t=times,
                 follow_jerk=np.ones((30, 55, 3)), follow_jerk_reliable=np.r_[np.zeros(1, bool), np.ones(54, bool)])
        w2: list = []
        out = eu._timing_for_unity({"beats": {}}, kin, times, w2)
        assert "lead" not in out["jerk"] and len(w2) == 1
        fj = np.asarray(out["jerk"]["follow"])
        assert fj.shape == (30, 24) and (fj[:, 0] == 0).all() and (fj[:, 1] > 0).all()


@needs_export_inputs
def test_synthetic_time_to_audio_is_never_negative_zero():
    rep, out = None, None
    try:
        rep = eu.export("synthetic", CAPTURE, install=False, room=False)
        out = Path(rep["out"])
        txt = (out / "capture.json").read_text()
        assert '"time_to_audio": -0.0' not in txt and json.loads(txt)["time_to_audio"] == 0.0
    finally:
        if out is not None:
            shutil.rmtree(out, ignore_errors=True)


def test_mamma_model_selection():
    assert eu.model_for({"model": np.array("smplx_locked_head (MAMMA)")}) in (eu.MAMMA_MODEL, eu.DEFAULT_MODEL)
    assert eu.model_for({}) == eu.DEFAULT_MODEL
    assert eu.model_for({}, Path("x.npz")) == Path("x.npz")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
