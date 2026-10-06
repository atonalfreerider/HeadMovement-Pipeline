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

WORK = PACKAGE_ROOT / "work" / "synthetic"
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
            assert skin.vertices.shape == (10475, 3) and skin.weights.shape == (10475, 55)
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
                err = np.abs(ours - o.vertices[0].numpy()).max()
                assert err < 1e-3, (role, i, err)
    finally:
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
