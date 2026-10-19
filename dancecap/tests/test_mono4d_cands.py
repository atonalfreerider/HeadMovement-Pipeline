r"""Unit tests for dancecap.mono4d.viggle (GLB animation reader, scoring) and dancecap.mono4d.candidates (the one-camera
candidate harness). Synthetic data only.

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_mono4d_cands.py -q      (from the workspace root)
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import mamma_dyn as md  # noqa: E402
from dancecap.mono4d import candidates as C  # noqa: E402
from dancecap.mono4d import viggle as V  # noqa: E402


def make_glb(times: np.ndarray, rotations: np.ndarray) -> bytes:
    """mixamorig:Hips (root, translated) -> mixamorig:LeftUpLeg (child 0.1 m along x) -> LeftLeg (0.4 m down);
    one animation rotating the Hips about z with the given XYZW quaternions."""
    tbytes = times.astype(np.float32).tobytes()
    rbytes = rotations.astype(np.float32).tobytes()
    binc = tbytes + rbytes
    binc += b"\0" * (-len(binc) % 4)
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"name": "mixamorig:Hips", "translation": [0, 1, 0], "children": [1]},
                     {"name": "mixamorig:LeftUpLeg", "translation": [0.1, 0, 0], "children": [2]},
                     {"name": "mixamorig:LeftLeg", "translation": [0, -0.4, 0]}],
           "buffers": [{"byteLength": len(binc)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(tbytes)},
                           {"buffer": 0, "byteOffset": len(tbytes), "byteLength": len(rbytes)}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(times), "type": "SCALAR",
                          "min": [float(times.min())], "max": [float(times.max())]},
                         {"bufferView": 1, "componentType": 5126, "count": len(times), "type": "VEC4"}],
           "animations": [{"channels": [{"sampler": 0, "target": {"node": 0, "path": "rotation"}}],
                           "samplers": [{"input": 0, "output": 1, "interpolation": "LINEAR"}]}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binc)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binc), 0x004E4942) + binc)


def test_glb_reader_animates_the_chain():
    times = np.array([0.0, 1.0])
    half = np.sin(np.pi / 4)  # 90 deg about z at t = 1
    glb = make_glb(times, np.array([[0, 0, 0, 1], [0, 0, half, np.cos(np.pi / 4)]]))
    names, t, P = V.glb_joints(glb, fps=2.0)
    assert names == ["mixamorig:Hips", "mixamorig:LeftUpLeg", "mixamorig:LeftLeg"] and np.allclose(t, [0, 0.5, 1.0])
    assert np.allclose(P[0], [[0, 1, 0], [0.1, 1, 0], [0.1, 0.6, 0]], atol=1e-6)
    assert np.allclose(P[2], [[0, 1, 0], [0, 1.1, 0], [0.4, 1.1, 0]], atol=1e-6)  # rotated 90 deg about z
    mid = P[1, 1] - P[1, 0]
    assert np.isclose(np.degrees(np.arctan2(mid[1], mid[0])), 45.0, atol=1e-4)  # slerp half way
    assert V.mixamo_index(names) == {0: 0, 1: 1, 4: 2}


def _ref_pose(T=20, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.3, size=(22, 3)) + [0, 1, 0]
    J = np.repeat(base[None], T, 0) + rng.normal(scale=0.01, size=(T, 22, 3))
    J[:, :, 0] += np.linspace(0, 0.5, T)[:, None]  # the dancer walks along x
    return J


def test_score_arrays_metrics():
    T = 20
    t = 10 + np.arange(T) / 30
    ref = _ref_pose(T)
    same = C.score_arrays(t, ref, "world", t, ref, (10, 11))
    assert same["frames"] == T and same["pa22_mm"]["median"] < 1e-6 and same["pelvis_mm"]["median"] < 1e-6
    assert same["rr_mm"]["median"] < 1e-6 and same["limb_deg"]["median"] < 1e-2  # arccos near 1
    # rotated + scaled + shifted copy in a free frame: pose errors vanish, world metrics are not computed
    th = np.radians(40)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    free = 1.3 * ref @ Rz.T + [5, 0, 2]
    s = C.score_arrays(t, free, "free", t, ref, (10, 11))
    assert s["pa22_mm"]["median"] < 1e-6 and s["rr_mm"]["n"] == 0 and s["pelvis_mm"]["n"] == 0
    # a world-frame copy 10 cm off: pose fine, pelvis 100 mm
    off = C.score_arrays(t, ref + [0.1, 0, 0], "world", t, ref, (10, 11))
    assert abs(off["pelvis_mm"]["median"] - 100) < 1e-6 and off["rr_mm"]["median"] < 1e-6
    # 14-joint rigs: NaN elsewhere, still scored on the shared joints
    J14 = np.full_like(ref, np.nan)
    J14[:, C.J14] = ref[:, C.J14]
    s14 = C.score_arrays(t, J14, "world", t, ref, (10, 11))
    assert s14["pa14_mm"]["median"] < 1e-6 and s14["pa22_mm"]["median"] < 1e-6


def test_to_world_inverts_the_camera():
    T = 5
    C0 = np.array([[2.0, 1.5, 3.0]] * T)
    R = md._look_at(C0, np.repeat([[0.0, 0.9, 0.0]], T, 0), np.zeros(T))
    t = -np.einsum("fij,fj->fi", R, C0)
    Xw = np.random.default_rng(1).normal(size=(T, 22, 3))
    Xc = np.einsum("fij,fnj->fni", R, Xw) + t[:, None]
    assert np.allclose(C.to_world(Xc, R, t), Xw)


def test_assign_roles_by_2d_boxes(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PACKAGE_ROOT", tmp_path)
    K = np.array([[1000.0, 0, 540], [0, 1000.0, 960], [0, 0, 1]])
    T = 6
    ft = 5 + np.arange(T) / 30

    def body(x):  # a 0.5 m wide, 1.7 m tall person 4 m away, centred at x
        P = np.zeros((T, 22, 3))
        P[:, :11] = [x - 0.25, -0.85, 4.0]
        P[:, 11:] = [x + 0.25, 0.85, 4.0]
        return P

    def box(x):
        return [540 + 1000 * (x - 0.25) / 4, 960 - 1000 * 0.85 / 4, 540 + 1000 * (x + 0.25) / 4, 960 + 1000 * 0.85 / 4]

    frames = [{"i": k, "t": float(ft[k]), "lead": {"box": box(0.6)}, "follow": {"box": box(-0.6)}} for k in range(T)]
    (tmp_path / "work" / "tk" / "dancers").mkdir(parents=True)
    (tmp_path / "work" / "tk" / "dancers" / "roles.json").write_text(json.dumps({"cams": {"06": {"frames": frames}}}))
    a = C.assign_roles({"p0": body(-0.6), "p1": body(0.6), "p2": body(3.0)}, "tk", "06", K, ft)
    assert a["lead"] == "p1" and a["follow"] == "p0"


def test_save_and_score_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PACKAGE_ROOT", tmp_path)
    T = 12
    t = 20 + np.arange(T) / 30
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    for role in C.ROLES:
        J = np.zeros((T, 55, 3))
        J[:, :22] = _ref_pose(T, seed=3 if role == "lead" else 4)
        np.savez(ref_dir / f"{role}.npz", t=t, joints=J, quality=np.ones(T))
        C.save("tk", "perfect", role, t, J[:, :22], "world", {"method": "copy"})
    out = C.score("tk", "perfect", (20, 21), reference=ref_dir)
    assert all(out["roles"][r]["pa22_mm"]["median"] < 1e-6 for r in C.ROLES)
    assert json.loads((C.cand_dir("tk", "perfect") / "meta.json").read_text())["method"] == "copy"
    with pytest.raises(ValueError):
        C.save("tk", "bad", "lead", t, np.zeros((T, 10, 3)), "world")
