"""Export a take to a head-movement capture folder (capture.json version 3, see CONTRACTS.md "Unity capture").

Inputs (work/<take>/): smplx/{lead,follow}.npz (contract: SMPL-X params on the master grid, Y-up right-handed
world, metres, floor y=0), optional timing/timing.json + timing/kinematics.npz, physics/physics.json, beats.json,
audio.wav. Output: work/<take>/export/<Capture>/, mirrored to head-movement/Assets/StreamingAssets/<Capture>/
(git-ignored - the shaped SMPL-X skin is model-derived data and must never be committed to the MIT repo).

    python -m dancecap export synthetic --capture 01_SyntheticSMPLX

Files written (all little-endian):
  capture.json            version 3 manifest (v2 keys kept: fps, frame_count, duration, audio, audio_offset, beats,
                          dancers -> figure1/2.json)
  times.json              per-frame reference seconds (T,); audio time of frame i = times[i] + time_to_audio
  figure1.json/figure2.json  v2 SMPL-24 joint positions per frame (Unity coords) for the glowing skeletons
  video_meta.json         v2 {duration, frame_count}
  zouk-time-analysis.json beats [[audio_t, type]] (refined timing grid when available)
  smplx_<role>.bin        motion: "DCXM" + u32 header_len + header json (padded to 4) + T x FRAME_FLOATS float32:
                          transl(3) + 55 local joint rotations (x,y,z,w Unity quaternions) + 55 joint positions (Unity)
  smplx_<role>_skin.bin   shaped skin: "DCXS" + u32 header_len + header json + blocks rest_vertices (V,3) f32,
                          faces (F,3) i32 (Unity winding), weights (V,55) f32, rest_joints (55,3) f32, parents (55,) i32
  timing.json             Unity subset of timing/timing.json (+ per-frame |jerk| of the SMPL-24 joints from the
                          kinematics spline) in reference seconds
  physics.json            copy of physics/physics.json (world frame; Unity converts (x,y,z) -> (x,y,-z))

Unity conversion of the Y-up right-handed world: p -> (x, y, -z); quaternion (qx,qy,qz,qw) -> (-qx,-qy,qz,qw);
triangle winding reversed (the mirror flips handedness). The skin omits SMPL-X pose-corrective blendshapes
(plain LBS), so elbows/knees lose a little volume at large bends.
"""
from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE

ROLES = ("lead", "follow")
N_JOINTS = 55
FRAME_FLOATS = 3 + N_JOINTS * 4 + N_JOINTS * 3
MOTION_MAGIC = b"DCXM"
SKIN_MAGIC = b"DCXS"
UNITY_STREAMING = WORKSPACE / "head-movement" / "Assets" / "StreamingAssets"
DEFAULT_MODEL = WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"
MAMMA_MODEL = WORKSPACE / "mamma" / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"
# SMPL-24 order used by Dancer.cs: SMPL-X 0..21, then left/right_middle1 stand in for SMPL's L_Hand/R_Hand
SMPL24_FROM_SMPLX = list(range(22)) + [28, 43]
MIRROR = np.array([1.0, 1.0, -1.0])


def to_unity_points(p: np.ndarray) -> np.ndarray:
    """(..., 3) world points/vectors -> Unity (x, y, -z)."""
    return np.asarray(p, float) * MIRROR


def to_unity_quats(q_xyzw: np.ndarray) -> np.ndarray:
    """(..., 4) world quaternions (x,y,z,w) -> Unity (-x, -y, z, w): the z-mirror conjugation S R S."""
    return np.asarray(q_xyzw, float) * np.array([-1.0, -1.0, 1.0, 1.0])


def full_pose(d: dict) -> np.ndarray:
    """(T, 55, 3) local axis-angle rotations in SMPL-X joint order (hand poses absolute, flat_hand_mean=True)."""
    T = len(d["t"])
    z3 = np.zeros((T, 3))
    parts = [d["global_orient"], d["body_pose"], d.get("jaw_pose", z3), d.get("leye_pose", z3),
             d.get("reye_pose", z3), d.get("left_hand_pose", np.zeros((T, 45))),
             d.get("right_hand_pose", np.zeros((T, 45)))]
    pose = np.concatenate([np.asarray(p, float).reshape(T, -1) for p in parts], 1)
    if pose.shape[1] != N_JOINTS * 3:
        raise ValueError(f"pose has {pose.shape[1] // 3} joints, expected {N_JOINTS}")
    return pose.reshape(T, N_JOINTS, 3)


def rotvec_to_quat(rv: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 4) quaternion (x, y, z, w), w >= 0."""
    rv = np.asarray(rv, float)
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * angle
    # sin(half)/angle with its small-angle limit 0.5
    k = np.where(angle > 1e-8, np.sin(half) / np.maximum(angle, 1e-12), 0.5 - angle ** 2 / 48.0)
    return np.concatenate([rv * k, np.cos(half)], -1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


@dataclass
class Skin:
    """Shaped SMPL-X rest mesh for one dancer (world frame, before the Unity mirror)."""
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3) int, right-handed counter-clockwise
    weights: np.ndarray  # (V, 55)
    joints: np.ndarray  # (55, 3)
    parents: np.ndarray  # (55,) int, -1 for the root
    model: str
    betas: np.ndarray


def model_for(d: dict, override: Path | None = None) -> Path:
    """The SMPL-X model file a smplx npz was fitted with (MAMMA's locked-head model, else the plain neutral)."""
    if override is not None:
        return Path(override)
    tag = str(d.get("model", ""))
    return MAMMA_MODEL if "locked_head" in tag and MAMMA_MODEL.exists() else DEFAULT_MODEL


def shaped_skin(model_path: Path, betas: np.ndarray) -> Skin:
    """v_shaped = v_template + shapedirs[..., :B] betas; joints = J_regressor v_shaped (expression = 0)."""
    m = np.load(model_path, allow_pickle=True)
    betas = np.asarray(betas, float).reshape(-1)
    sd = np.asarray(m["shapedirs"], float)[:, :, :len(betas)]
    v = np.asarray(m["v_template"], float) + sd @ betas
    J = np.asarray(m["J_regressor"], float) @ v
    parents = np.asarray(m["kintree_table"], np.int64)[0].copy()
    parents[0] = -1
    return Skin(vertices=v, faces=np.asarray(m["f"], np.int64), weights=np.asarray(m["weights"], float),
                joints=J[:N_JOINTS], parents=parents[:N_JOINTS], model=str(model_path), betas=betas)


def fk_unity(rest_joints_u: np.ndarray, parents: np.ndarray, transl_u: np.ndarray, quats_u: np.ndarray):
    """Forward kinematics exactly as SmplxAvatar.cs does it (Unity coords): root at rest_joint0 + transl, every
    other bone at its rest offset from the parent, local rotations from the motion file.
    Returns world joint positions (T,55,3) and world rotation matrices (T,55,3,3)."""
    T = len(transl_u)
    R_loc = quat_to_matrix(quats_u)
    R = np.zeros((T, N_JOINTS, 3, 3))
    P = np.zeros((T, N_JOINTS, 3))
    for j in range(N_JOINTS):
        p = parents[j]
        if p < 0:
            R[:, j] = R_loc[:, j]
            P[:, j] = rest_joints_u[j] + transl_u
        else:
            R[:, j] = R[:, p] @ R_loc[:, j]
            P[:, j] = P[:, p] + np.einsum("tab,b->ta", R[:, p], rest_joints_u[j] - rest_joints_u[p])
    return P, R


def lbs_unity(skin_u: Skin, P: np.ndarray, R: np.ndarray, frame: int) -> np.ndarray:
    """Posed vertices of one frame by linear blend skinning (bindpose = translation to the rest joint)."""
    rel =skin_u.vertices[:, None, :] - skin_u.joints[None]  # (V,55,3)
    posed = np.einsum("jab,vjb->vja", R[frame], rel) + P[frame][None]
    return np.einsum("vj,vja->va", skin_u.weights, posed)


def unity_skin(skin: Skin) -> Skin:
    """Mirror a world-frame skin into Unity coordinates (z flip, reversed winding)."""
    return Skin(vertices=to_unity_points(skin.vertices), faces=skin.faces[:, [0, 2, 1]], weights=skin.weights,
                joints=to_unity_points(skin.joints), parents=skin.parents, model=skin.model, betas=skin.betas)


def _write_blob(path: Path, magic: bytes, header: dict, blocks: list[np.ndarray]) -> None:
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 4)
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", len(h)))
        f.write(h)
        for b in blocks:
            f.write(np.ascontiguousarray(b).astype(b.dtype.newbyteorder("<")).tobytes())


def read_blob(path: Path) -> tuple[bytes, dict, bytes]:
    raw = Path(path).read_bytes()
    n = struct.unpack("<I", raw[4:8])[0]
    return raw[:4], json.loads(raw[8:8 + n]), raw[8 + n:]


def write_motion(path: Path, role: str, t: np.ndarray, transl_u: np.ndarray, quats_u: np.ndarray,
                 joints_u: np.ndarray) -> dict:
    T = len(t)
    frames = np.concatenate([transl_u.reshape(T, 3), quats_u.reshape(T, -1), joints_u.reshape(T, -1)], 1)
    assert frames.shape[1] == FRAME_FLOATS
    header = {"format": "dancecap smplx motion", "version": 1, "role": role, "frames": T, "joints": N_JOINTS,
              "frame_floats": FRAME_FLOATS, "layout": [["transl", 3], ["rotations_xyzw", N_JOINTS * 4],
                                                       ["joint_positions", N_JOINTS * 3]],
              "coords": "unity (x, y, -z) of the Y-up right-handed world", "t0": float(t[0]), "t1": float(t[-1])}
    _write_blob(path, MOTION_MAGIC, header, [frames.astype(np.float32)])
    return header


def read_motion(path: Path) -> tuple[dict, np.ndarray]:
    magic, header, body = read_blob(path)
    assert magic == MOTION_MAGIC, magic
    return header, np.frombuffer(body, "<f4").reshape(header["frames"], header["frame_floats"])


def write_skin(path: Path, role: str, skin_u: Skin) -> dict:
    V, F = len(skin_u.vertices), len(skin_u.faces)
    header = {"format": "dancecap smplx skin", "version": 1, "role": role, "vertices": V, "faces": F,
              "joints": N_JOINTS, "model": Path(skin_u.model).name, "betas": [round(float(b), 6) for b in skin_u.betas],
              "blocks": [["rest_vertices", "f4", [V, 3]], ["faces", "i4", [F, 3]], ["weights", "f4", [V, N_JOINTS]],
                         ["rest_joints", "f4", [N_JOINTS, 3]], ["parents", "i4", [N_JOINTS]]],
              "coords": "unity (x, y, -z); winding reversed for the left-handed frame",
              "licence": "SMPL-X derived (non-commercial). Generated locally; never commit."}
    _write_blob(path, SKIN_MAGIC, header, [skin_u.vertices.astype(np.float32), skin_u.faces.astype(np.int32),
                                          skin_u.weights.astype(np.float32), skin_u.joints.astype(np.float32),
                                          skin_u.parents.astype(np.int32)])
    return header


def read_skin(path: Path) -> tuple[dict, Skin]:
    magic, h, body = read_blob(path)
    assert magic == SKIN_MAGIC, magic
    out, off = [], 0
    for _, dt, shape in h["blocks"]:
        n = int(np.prod(shape))
        out.append(np.frombuffer(body, "<" + dt, n, off).reshape(shape))
        off += n * 4
    v, f, w, j, p = out
    return h, Skin(v, f, w, j, p, h["model"], np.asarray(h["betas"]))


def _beats_audio(work: Path, beats: Path | None, timing: dict | None, time_to_audio: float) -> list | None:
    """Beat grid as [[audio_t, type]] - the refined timing grid when available, else beats.json."""
    if timing is not None and timing.get("beats", {}).get("grid"):
        grid = [(float(r[0]), int(r[1])) for r in timing["beats"]["grid"]]
    else:
        src = beats or next((p for p in (work / "timing" / "beats.json", work / "beats.json") if p.exists()), None)
        if src is None:
            return None
        grid = [(float(r[0]), int(r[1])) for r in json.loads(Path(src).read_text())]
    return [[round(t + time_to_audio, 6), k] for t, k in grid]


def _timing_for_unity(timing: dict, kin: Path, times: np.ndarray) -> dict:
    """The parts of timing.json the overlays use, plus per-frame |jerk| (SMPL-24 joints) on the capture grid."""
    keep = ("t", "foot", "async_ms", "beat_t", "beat_type", "sigma", "count", "measure", "detect_rate", "method")
    out = {"version": 1, "time": "reference seconds; audio time = t + capture.time_to_audio",
           "beats": {k: timing["beats"].get(k) for k in ("grid", "grid_columns", "measure_s", "tempo_bpm", "warnings")},
           "footsteps": {r: [{k: s.get(k) for k in keep} for s in timing.get("footsteps", {}).get(r, [])]
                         for r in ROLES},
           "asynchrony": {}}
    for r in ROLES:
        fs = timing.get("asynchrony", {}).get(r, {}).get("footsteps", {})
        out["asynchrony"][r] = {k: fs.get(k) for k in ("n", "mean_ms", "median_ms", "ci95_ms", "hit_rate", "R")}
    fml = timing.get("asynchrony", {}).get("follow_minus_lead", {})
    out["asynchrony"]["follow_minus_lead_ms"] = fml.get("overall_ms")
    if kin.exists():
        k = np.load(kin)
        jerk = {}
        for r in ROLES:
            if f"{r}_jerk" not in k.files:
                continue
            kt, kj = np.asarray(k[f"{r}_t"], float), np.asarray(k[f"{r}_jerk"], float)[:, SMPL24_FROM_SMPLX]
            mag = np.linalg.norm(kj, axis=-1)  # (Tk, 24)
            mag = np.stack([np.interp(times, kt, np.nan_to_num(mag[:, j])) for j in range(mag.shape[1])], 1)
            jerk[r] = np.round(mag, 2).tolist()
        if jerk:
            out["jerk"] = {"source": "timing/kinematics.npz quintic smoothing spline |d3x/dt3| (m/s^3), SMPL-24",
                           **jerk}
    return out


def export(take: str, capture: str | None = None, model: Path | None = None, audio: Path | None = None,
           audio_t0: float = 0.0, beats: Path | None = None, install: bool = True,
           streaming_assets: Path = UNITY_STREAMING) -> dict:
    """Write work/<take>/export/<capture>/ and (install=True) mirror it into head-movement StreamingAssets."""
    work = PACKAGE_ROOT / "work" / take
    capture = capture or take
    data = {}
    for r in ROLES:
        p = work / "smplx" / f"{r}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing (run the mamma stage, or `python -m dancecap synthetic`)")
        with np.load(p, allow_pickle=True) as z:
            data[r] = {k: z[k] for k in z.files}
    t = np.asarray(data["lead"]["t"], float)
    if len(data["follow"]["t"]) != len(t) or np.abs(np.asarray(data["follow"]["t"]) - t).max() > 1e-6:
        raise ValueError("lead and follow npz are not on the same time grid")
    if np.any(np.diff(t) <= 0):
        raise ValueError("smplx t must be strictly increasing")
    T = len(t)
    fps = float(1.0 / np.median(np.diff(t)))
    time_to_audio = -float(audio_t0)  # audio time = reference t - (reference time of audio sample 0)
    out = work / "export" / capture
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    (out / "times.json").write_text(json.dumps([round(float(x), 6) for x in t]))
    manifest: dict = {
        "version": 3, "fps": round(fps, 6), "frame_count": T, "duration": float(t[-1] - t[0] + 1.0 / fps),
        "audio": None, "audio_offset": float(t[0] + time_to_audio), "time_to_audio": time_to_audio,
        "times": "times.json", "beats": None,
        "dancers": [{"role": "lead", "joints": "figure1.json"}, {"role": "follow", "joints": "figure2.json"}],
        "coordinate_system": "unity: left-handed, y-up, metres, floor y=0",
        "world": "right-handed Y-up metres, floor y=0; Unity = (x, y, -z)",
        "smplx": {}, "smplx_skin": {}, "provenance": {"take": take, "source": "dancecap export_unity"},
    }
    report: dict = {"capture": capture, "frames": T, "fps": fps, "dancers": {}}
    for r, fig in zip(ROLES, ("figure1.json", "figure2.json")):
        d = data[r]
        joints_u = to_unity_points(np.asarray(d["joints"], float)[:, :N_JOINTS])
        j24 = joints_u[:, SMPL24_FROM_SMPLX]
        (out / fig).write_text(json.dumps(
            [[{"x": round(float(p[0]), 5), "y": round(float(p[1]), 5), "z": round(float(p[2]), 5)} for p in f]
             for f in j24], separators=(",", ":")))
        quats_u = to_unity_quats(rotvec_to_quat(full_pose(d)))
        transl_u = to_unity_points(np.asarray(d["transl"], float))
        write_motion(out / f"smplx_{r}.bin", r, t, transl_u, quats_u, joints_u)
        mp = model_for(d, model)
        skin_u = unity_skin(shaped_skin(mp, d["betas"]))
        write_skin(out / f"smplx_{r}_skin.bin", r, skin_u)
        P, _ = fk_unity(skin_u.joints, skin_u.parents, transl_u, quats_u)
        fk_err = float(np.abs(P - joints_u).max())
        manifest["smplx"][r] = f"smplx_{r}.bin"
        manifest["smplx_skin"][r] = f"smplx_{r}_skin.bin"
        report["dancers"][r] = {"model": str(mp), "betas": len(d["betas"]), "fk_max_err_mm": round(fk_err * 1000, 3)}
        if fk_err > 0.01:
            print(f"[export] WARNING {r}: FK of the exported rotations misses the npz joints by {fk_err * 1000:.1f} mm "
                  f"(model mismatch? pass --model)")
    (out / "video_meta.json").write_text(json.dumps({"duration": manifest["duration"], "frame_count": T}))

    timing_path = work / "timing" / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    grid = _beats_audio(work, beats, timing, time_to_audio)
    if grid:
        (out / "zouk-time-analysis.json").write_text(json.dumps(grid))
        manifest["beats"] = "zouk-time-analysis.json"
    if timing is not None:
        (out / "timing.json").write_text(json.dumps(_timing_for_unity(timing, work / "timing" / "kinematics.npz", t),
                                                    separators=(",", ":")))
        manifest["timing"] = "timing.json"
    phys = work / "physics" / "physics.json"
    if phys.exists():
        shutil.copy2(phys, out / "physics.json")
        manifest["physics"] = "physics.json"
    audio = audio or (work / "audio.wav" if (work / "audio.wav").exists() else None)
    if audio is not None:
        shutil.copy2(audio, out / "audio.wav")
        manifest["audio"] = "audio.wav"
    (out / "capture.json").write_text(json.dumps(manifest, indent=2))
    report["out"] = str(out)
    report["files"] = sorted(p.name for p in out.iterdir())

    if install:
        dst = Path(streaming_assets) / capture
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(out, dst)
        report["installed"] = str(dst)
    return report
