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

  <role>_albedo.png       photoreal albedo (copy of work/<take>/avatar/<role>_albedo.png, SMPL-X UV layout) when the
                          texture stage ran; the skin is split at the UV seams so it can carry it (skin version 2)
  room_mesh.bin           static room for Quest ("DCXR"): textured triangle mesh already in Unity coordinates (world =
                          world_from_sfm @ SfM, then (x, y, -z)), submeshes "room"/"floor" -> room_tex.png/floor_tex.png
  splat_quest.ply         Quest Gaussian splat of the room (SfM frame) + capture.json environment_splat.scene_to_unity

Unity conversion of the Y-up right-handed world: p -> (x, y, -z); quaternion (qx,qy,qz,qw) -> (-qx,-qy,qz,qw);
triangle winding reversed (the mirror flips handedness). The skin omits SMPL-X pose-corrective blendshapes
(plain LBS), so elbows/knees lose a little volume at large bends.

Audio: export needs the music. Resolution order: --audio + --audio-t0 (both required together), else
audio/audio_overlap.wav with reference_start from audio/audio_overlap.json (the beats stage), else work/<take>/audio.wav
of a synthetic take (truth.json present: the generator starts the wav at reference 0) - otherwise an error unless
allow_no_audio (--no-audio), in which case Unity plays a silent clock.
"""
from __future__ import annotations

import json
import re
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
ROOM_MAGIC = b"DCXR"
CAPTURE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


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
    uv: np.ndarray | None = None  # (V, 2) Unity UV (v up, PNG row 0 = v 1) - seam-split skins only
    normals: np.ndarray | None = None  # (V, 3) rest normals of the UNSPLIT mesh (no shading seams at UV cuts)
    vertex_ids: np.ndarray | None = None  # (V,) SMPL-X vertex of each split vertex


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


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals of a right-handed CCW mesh (outward)."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(verts, dtype=float)
    for k in range(3):
        np.add.at(out, faces[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def uv_split(f: np.ndarray, ft: np.ndarray, vt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a mesh at its UV seams: one output vertex per unique (v, vt) pair used by the faces.
    Returns vertex_ids (V',) = source vertex, uv (V', 2), faces (F, 3) into the split vertices."""
    f, ft = np.asarray(f, np.int64), np.asarray(ft, np.int64)
    if f.shape != ft.shape:
        raise ValueError(f"f {f.shape} and ft {ft.shape} differ")
    pairs = np.stack([f.ravel(), ft.ravel()], 1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    return uniq[:, 0], np.asarray(vt, float)[uniq[:, 1]], inv.reshape(f.shape)


def textured_skin(skin: Skin, model_path: Path) -> Skin:
    """The shaped skin split at the model's UV seams (vt/ft of the SMPL-X npz): rest positions and LBS weights are
    duplicated per split vertex, normals come from the unsplit mesh. Unchanged when the model has no UVs."""
    m = np.load(model_path, allow_pickle=True)
    if "vt" not in m.files or "ft" not in m.files:
        return skin
    vid, uv, faces = uv_split(skin.faces, m["ft"], m["vt"])
    normals = vertex_normals(skin.vertices, skin.faces)
    return Skin(vertices=skin.vertices[vid], faces=faces, weights=skin.weights[vid], joints=skin.joints,
                parents=skin.parents, model=skin.model, betas=skin.betas, uv=uv, normals=normals[vid],
                vertex_ids=vid)


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
                joints=to_unity_points(skin.joints), parents=skin.parents, model=skin.model, betas=skin.betas,
                uv=skin.uv, normals=None if skin.normals is None else to_unity_points(skin.normals),
                vertex_ids=skin.vertex_ids)


def _write_blob(path: Path, magic: bytes, header: dict, blocks: list[np.ndarray]) -> None:
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 4)
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", len(h)))
        f.write(h)
        for b in blocks:
            raw = np.ascontiguousarray(b).astype(b.dtype.newbyteorder("<")).tobytes()
            f.write(raw + b"\0" * ((-len(raw)) % 4))


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
    """Version 1 = the five SMPL-X blocks; version 2 appends uv (V,2) f4, normals (V,3) f4 and vertex_ids (V,) i4
    for a seam-split (textured) skin. Readers locate blocks by the header's "blocks" list, in order."""
    V, F = len(skin_u.vertices), len(skin_u.faces)
    blocks = [["rest_vertices", "f4", [V, 3]], ["faces", "i4", [F, 3]], ["weights", "f4", [V, N_JOINTS]],
              ["rest_joints", "f4", [N_JOINTS, 3]], ["parents", "i4", [N_JOINTS]]]
    data = [skin_u.vertices.astype(np.float32), skin_u.faces.astype(np.int32), skin_u.weights.astype(np.float32),
            skin_u.joints.astype(np.float32), skin_u.parents.astype(np.int32)]
    split = skin_u.uv is not None
    if split:
        blocks += [["uv", "f4", [V, 2]], ["normals", "f4", [V, 3]], ["vertex_ids", "i4", [V]]]
        data += [skin_u.uv.astype(np.float32), skin_u.normals.astype(np.float32), skin_u.vertex_ids.astype(np.int32)]
    header = {"format": "dancecap smplx skin", "version": 2 if split else 1, "role": role, "vertices": V, "faces": F,
              "joints": N_JOINTS, "model": Path(skin_u.model).name, "betas": [round(float(b), 6) for b in skin_u.betas],
              "blocks": blocks,
              "coords": "unity (x, y, -z); winding reversed for the left-handed frame",
              "uv": ("u right, v up; PNG row 0 = v 1 (Unity Texture2D.LoadImage)" if split else None),
              "licence": "SMPL-X derived (non-commercial). Generated locally; never commit."}
    _write_blob(path, SKIN_MAGIC, header, data)
    return header


def _read_blocks(h: dict, body: bytes) -> dict:
    out, off = {}, 0
    for name, dt, shape in h["blocks"]:
        n = int(np.prod(shape))
        size = np.dtype(dt).itemsize
        out[name] = np.frombuffer(body, "<" + dt, n, off).reshape(shape)
        off += n * size
        off += (-off) % 4
    return out


def read_skin(path: Path) -> tuple[dict, Skin]:
    magic, h, body = read_blob(path)
    assert magic == SKIN_MAGIC, magic
    b = _read_blocks(h, body)
    return h, Skin(b["rest_vertices"], b["faces"], b["weights"], b["rest_joints"], b["parents"], h["model"],
                   np.asarray(h["betas"]), uv=b.get("uv"), normals=b.get("normals"), vertex_ids=b.get("vertex_ids"))


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


def _timing_for_unity(timing: dict, kin: Path, times: np.ndarray, warnings: list[str] | None = None) -> dict:
    """The parts of timing.json the overlays use, plus per-frame |jerk| (SMPL-24 joints) on the capture grid.
    A role's jerk is used only when its kinematics grid covers the capture's [t0, t1] within half a frame (else a
    stale timing run would be np.interp-clamped onto the wrong times); joints the spline marks unreliable
    (<role>_jerk_reliable False: mostly noise) are written as 0 so the glow ignores them."""
    warnings = warnings if warnings is not None else []
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
        half = 0.5 * float(np.median(np.diff(times))) if len(times) > 1 else 0.0
        jerk, reliable = {}, {}
        for r in ROLES:
            if f"{r}_jerk" not in k.files or f"{r}_t" not in k.files:
                continue
            kt = np.asarray(k[f"{r}_t"], float)
            if len(kt) < 2 or kt[0] > times[0] + half + 1e-6 or kt[-1] < times[-1] - half - 1e-6:
                warnings.append(f"timing/kinematics.npz {r}_t [{kt[0]:.3f}, {kt[-1]:.3f}] does not cover the capture "
                                f"[{times[0]:.3f}, {times[-1]:.3f}] - stale timing run? {r} jerk skipped "
                                f"(Unity falls back to finite differences)")
                continue
            kj = np.asarray(k[f"{r}_jerk"], float)[:, SMPL24_FROM_SMPLX]
            mag = np.linalg.norm(kj, axis=-1)  # (Tk, 24)
            mag = np.stack([np.interp(times, kt, np.nan_to_num(mag[:, j])) for j in range(mag.shape[1])], 1)
            if f"{r}_jerk_reliable" in k.files:
                ok = np.asarray(k[f"{r}_jerk_reliable"], bool)[SMPL24_FROM_SMPLX]
                mag[:, ~ok] = 0.0
                reliable[r] = ok.tolist()
            jerk[r] = np.round(mag, 2).tolist()
        if jerk:
            out["jerk"] = {"source": "timing/kinematics.npz quintic smoothing spline |d3x/dt3| (m/s^3), SMPL-24; "
                                     "joints with reliable=false are written as 0", **jerk}
            if reliable:
                out["jerk_reliable"] = reliable
    return out


def validate_capture_name(capture: str) -> str:
    """Capture folder names are used in rmtree(work/<take>/export/<capture>) and rmtree(StreamingAssets/<capture>):
    plain names only (no separators, no '.'/'..', no absolute paths)."""
    if not isinstance(capture, str) or capture in (".", "..") or not CAPTURE_NAME.match(capture):
        raise ValueError(f"invalid capture name {capture!r}: use letters, digits, '_', '.', '-' (e.g. 02_LarissaKadu)")
    return capture


def resolve_audio(work: Path, audio: Path | None, audio_t0: float | None,
                  allow_no_audio: bool = False) -> tuple[Path | None, float, str]:
    """(wav, reference time of its first sample, source). See the module docstring for the order."""
    if audio is not None:
        if audio_t0 is None:
            raise ValueError("--audio needs --audio-t0 (reference time of the wav's first sample); a missing t0 "
                             "silently shifts A/V sync. For the beats stage's wav use audio/audio_overlap.json "
                             "reference_start")
        if not Path(audio).exists():
            raise FileNotFoundError(f"audio {audio} missing")
        return Path(audio), float(audio_t0), "--audio"
    overlap, meta = work / "audio" / "audio_overlap.wav", work / "audio" / "audio_overlap.json"
    if overlap.exists() and meta.exists():
        t0 = float(json.loads(meta.read_text())["reference_start"])
        return overlap, (t0 if audio_t0 is None else float(audio_t0)), "audio/audio_overlap.wav"
    if (work / "audio.wav").exists():
        if audio_t0 is not None:
            return work / "audio.wav", float(audio_t0), "audio.wav"
        if (work / "truth.json").exists():  # synthetic generators start the wav at reference 0
            return work / "audio.wav", 0.0, "audio.wav (synthetic, t0 = 0)"
        raise ValueError(f"{work / 'audio.wav'} has no known start time: pass --audio-t0")
    if allow_no_audio:
        return None, 0.0, "none"
    raise ValueError(f"no audio for {work.name}: run the beats stage (audio/audio_overlap.wav), pass --audio and "
                     f"--audio-t0, or --no-audio (Unity then plays a silent clock)")


def _physics_on_grid(phys: Path, t: np.ndarray, warnings: list[str]) -> bool:
    """physics.json must be on the capture's smplx grid (else Unity would show wrong-time COM/forces)."""
    try:
        pt = np.asarray(json.loads(phys.read_text())["t"], float)
    except (KeyError, ValueError) as e:
        warnings.append(f"physics/physics.json unreadable ({e}) - physics layer skipped")
        return False
    if len(pt) != len(t) or np.abs(pt - t).max() > 1e-3:
        warnings.append(f"physics/physics.json t ({len(pt)} frames, {pt[0] if len(pt) else 'n/a'}..) is not the smplx "
                        f"grid ({len(t)} frames, {t[0]:.3f}..) - stale physics run? physics layer skipped")
        return False
    return True


# ------------------------------------------------------------------------------------------------ room (Quest)

def world_from_sfm(work: Path) -> np.ndarray:
    """4x4 Sim3 SfM -> world from cameras/world_from_sfm.json."""
    w = json.loads((work / "cameras" / "world_from_sfm.json").read_text())
    M = np.eye(4)
    M[:3, :3] = float(w["s"]) * np.asarray(w["R"], float)
    M[:3, 3] = np.asarray(w["t"], float)
    return M


def scene_to_unity(work: Path) -> np.ndarray:
    """4x4 SfM -> Unity (world_from_sfm, then the (x, y, -z) mirror)."""
    return np.diag([1.0, 1.0, -1.0, 1.0]) @ world_from_sfm(work)


def read_obj(path: Path) -> dict:
    """Minimal OBJ reader (v, vt, f with v/vt[/vn], usemtl groups, triangles or fans)."""
    v, vt, groups, cur = [], [], {}, "default"
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            v.append(line.split()[1:4])
        elif line.startswith("vt "):
            vt.append(line.split()[1:3])
        elif line.startswith("usemtl "):
            cur = line.split(None, 1)[1].strip()
        elif line.startswith("f "):
            idx = [w.split("/") for w in line.split()[1:]]
            vi = [int(a[0]) for a in idx]
            ti = [int(a[1]) if len(a) > 1 and a[1] else 0 for a in idx]
            for k in range(1, len(vi) - 1):  # fan
                groups.setdefault(cur, []).append((vi[0], vi[k], vi[k + 1], ti[0], ti[k], ti[k + 1]))
    V = np.asarray(v, float)
    VT = np.asarray(vt, float) if vt else np.zeros((1, 2))
    out = {"v": V, "vt": VT, "groups": {}}
    for name, rows in groups.items():
        a = np.asarray(rows, np.int64)
        f, ft = a[:, :3], a[:, 3:]
        f = np.where(f > 0, f - 1, len(V) + f)  # 1-based or negative
        ft = np.where(ft > 0, ft - 1, np.where(ft < 0, len(VT) + ft, 0))
        out["groups"][name] = (f, ft)
    return out


def floor_clearance_mask(floor_P: np.ndarray, floor_F: np.ndarray, room_P: np.ndarray, room_F: np.ndarray,
                         erode_m: float = 0.4, max_height_m: float = 2.3, cell_m: float = 0.05) -> np.ndarray:
    """True for "room" triangles to DROP: lying over the reconstructed dance floor (floor submesh footprint eroded by
    erode_m, so the walls' feet survive) and entirely below max_height_m. In a TSDF fused over the whole take these
    are ghosts of the dancers (and spectators who walked through), not furniture: in VR they hang as ragged shards
    around the avatars. Coordinates: Unity (y up), floor y ~ 0."""
    import cv2
    xz = floor_P[:, [0, 2]]
    lo = xz.min(0) - 1.0
    shape = np.ceil((xz.max(0) + 1.0 - lo) / cell_m).astype(int)
    occ = np.zeros((shape[1], shape[0]), np.uint8)
    pts = np.round((xz - lo) / cell_m).astype(np.int32)
    for tri in floor_F:
        cv2.fillConvexPoly(occ, pts[tri], 1)
    k = max(1, int(round(erode_m / cell_m)))
    occ = cv2.erode(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    tri = room_P[room_F]  # (T, 3, 3)
    c = tri.mean(1)
    ij = np.round((c[:, [0, 2]] - lo) / cell_m).astype(int)
    inside = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    over = np.zeros(len(c), bool)
    over[inside] = occ[ij[inside, 1], ij[inside, 0]] > 0
    return over & (tri[:, :, 1].max(1) < max_height_m)


def read_mtl_textures(path: Path) -> dict:
    tex, cur = {}, None
    if not Path(path).exists():
        return tex
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] == "newmtl":
            cur = parts[1].strip()
        elif len(parts) == 2 and parts[0] == "map_Kd" and cur:
            tex[cur] = parts[1].strip()
    return tex


def _pot_texture(src: Path, dst: Path, max_size: int = 2048) -> list[int]:
    """Copy a texture as PNG resized to power-of-two sides <= max_size (mipmaps + ETC2/ASTC/DXT on Quest)."""
    import cv2
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read {src}")
    h, w = img.shape[:2]
    pot = lambda n: int(min(max_size, 2 ** int(round(np.log2(max(n, 4))))))  # noqa: E731
    W, H = pot(w), pot(h)
    if (W, H) != (w, h):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA if W * H < w * h else cv2.INTER_CUBIC)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return [W, H]


def write_room_mesh(path: Path, positions_u: np.ndarray, uv: np.ndarray | None, colors: np.ndarray | None,
                    submeshes: list[dict], indices: np.ndarray, extra: dict) -> dict:
    N, M = len(positions_u), len(indices)
    blocks = [["positions", "f4", [N, 3]]]
    data = [positions_u.astype(np.float32)]
    if uv is not None:
        blocks.append(["uv", "f4", [N, 2]])
        data.append(uv.astype(np.float32))
    if colors is not None:
        blocks.append(["colors", "u1", [N, 4]])
        data.append(colors.astype(np.uint8))
    blocks.append(["indices", "i4", [M]])
    data.append(indices.astype(np.int32))
    header = {"format": "dancecap room mesh", "version": 1, "vertices": N, "indices": M, "submeshes": submeshes,
              "blocks": blocks, "coords": "unity (x, y, -z) of the Y-up world; winding = Unity (render double-sided)",
              "uv": "u right, v up; PNG row 0 = v 1", **extra}
    _write_blob(path, ROOM_MAGIC, header, data)
    return header


def read_room_mesh(path: Path) -> tuple[dict, dict]:
    magic, h, body = read_blob(path)
    assert magic == ROOM_MAGIC, magic
    return h, _read_blocks(h, body)


def export_room(work: Path, out: Path, warnings: list[str], clear_dance_floor: bool = True) -> dict:
    """room/room_mesh.obj (+ textures) -> room_mesh.bin + room_tex.png/floor_tex.png in Unity coordinates, and the
    Quest splat room/splat_quest.ply with its SfM -> Unity matrix. Returns the manifest entries.
    clear_dance_floor drops "room" triangles standing on the dance floor below 2.3 m (dancer ghosts of the TSDF,
    floor_clearance_mask); the flat textured floor submesh stays."""
    room = work / "room"
    entries: dict = {}
    if not (work / "cameras" / "world_from_sfm.json").exists():
        if room.exists():
            warnings.append("room/ exists but cameras/world_from_sfm.json is missing - room skipped")
        return entries
    S2U = scene_to_unity(work)
    tr = room / "transform.json"
    if tr.exists():
        w = json.loads(tr.read_text()).get("world_from_sfm")
        if w:
            Mr = np.eye(4)
            Mr[:3, :3] = float(w["s"]) * np.asarray(w["R"], float)
            Mr[:3, 3] = np.asarray(w["t"], float)
            if np.abs(Mr - world_from_sfm(work)).max() > 1e-6:
                warnings.append("room/transform.json world_from_sfm differs from cameras/world_from_sfm.json: the room "
                                "was built before the last world-frame change (it is placed with the current one)")
    flip = np.linalg.det(S2U[:3, :3]) < 0  # mirror of the handedness -> reverse the winding
    obj, ply = room / "room_mesh.obj", room / "room_mesh.ply"
    if obj.exists():
        o = read_obj(obj)
        tex = read_mtl_textures(room / "room_mesh.mtl")
        P_all, UV_all, idx_all, subs = [], [], [], []
        base = 0
        tex_out = {}
        groups = dict(o["groups"])
        cleared = 0
        if clear_dance_floor and "floor" in groups and "room" in groups:
            to_u = lambda X: (S2U[:3, :3] @ X.T).T + S2U[:3, 3]  # noqa: E731
            drop = floor_clearance_mask(to_u(o["v"]), groups["floor"][0], to_u(o["v"]), groups["room"][0])
            cleared = int(drop.sum())
            groups["room"] = (groups["room"][0][~drop], groups["room"][1][~drop])
        for name, (f, ft) in groups.items():
            vid, uv, faces = uv_split(f, ft, o["vt"])
            if flip:
                faces = faces[:, [0, 2, 1]]
            P = (S2U[:3, :3] @ o["v"][vid].T).T + S2U[:3, 3]
            texture = None
            if name in tex and (room / tex[name]).exists():
                texture = {"room": "room_tex.png", "floor": "floor_tex.png"}.get(name, f"{name}_tex.png")
                tex_out[texture] = _pot_texture(room / tex[name], out / texture)
            subs.append({"name": name, "start": int(sum(len(i) for i in idx_all)), "count": int(faces.size),
                         "texture": texture, "vertex_start": base, "vertices": int(len(P))})
            P_all.append(P)
            UV_all.append(uv)
            idx_all.append((faces + base).ravel())
            base += len(P)
        P, UV, I = np.concatenate(P_all), np.concatenate(UV_all), np.concatenate(idx_all)
        h = write_room_mesh(out / "room_mesh.bin", P, UV, None, subs, I,
                            {"source": "room/room_mesh.obj", "textures": tex_out, "cleared_dance_floor_triangles": cleared,
                             "bounds_min": P.min(0).round(3).tolist(), "bounds_max": P.max(0).round(3).tolist()})
        entries["room"] = {"mesh": "room_mesh.bin", "shading": "unlit textured (lighting baked in the photos)",
                           "triangles": int(len(I) // 3), "vertices": int(len(P)), "cleared_triangles": cleared,
                           "textures": sorted(tex_out), "bounds_min": h["bounds_min"], "bounds_max": h["bounds_max"]}
    elif ply.exists():
        from plyfile import PlyData  # optional: only for the vertex-colour fallback
        d = PlyData.read(str(ply))
        vx = d["vertex"]
        V = np.stack([vx["x"], vx["y"], vx["z"]], 1).astype(float)
        C = np.stack([vx["red"], vx["green"], vx["blue"], np.full(len(V), 255)], 1)
        F = np.stack(list(d["face"]["vertex_indices"]), 0).astype(np.int64)
        if flip:
            F = F[:, [0, 2, 1]]
        P = (S2U[:3, :3] @ V.T).T + S2U[:3, 3]
        h = write_room_mesh(out / "room_mesh.bin", P, None, C, [{"name": "room", "start": 0, "count": int(F.size),
                                                                  "texture": None}], F.ravel(),
                            {"source": "room/room_mesh.ply", "bounds_min": P.min(0).round(3).tolist(),
                             "bounds_max": P.max(0).round(3).tolist()})
        entries["room"] = {"mesh": "room_mesh.bin", "shading": "unlit vertex colour", "triangles": int(len(F)),
                           "vertices": int(len(P)), "textures": [], "bounds_min": h["bounds_min"],
                           "bounds_max": h["bounds_max"]}
    splat = room / "splat_quest.ply"
    if splat.exists():
        shutil.copy2(splat, out / "splat_quest.ply")
        entries["environment_splat"] = {"path": "splat_quest.ply",
                                        "scene_to_unity": [round(float(x), 9) for x in S2U.ravel()],
                                        "frame": "SfM (room/transform.json) -> Unity"}
    return entries


# ------------------------------------------------------------------------------------------------ export

def export(take: str, capture: str | None = None, model: Path | None = None, audio: Path | None = None,
           audio_t0: float | None = None, beats: Path | None = None, install: bool = True,
           streaming_assets: Path = UNITY_STREAMING, allow_no_audio: bool = False, textures: bool = True,
           room: bool = True, smplx_dir: str | None = None, faces: bool = True, face_albedo: str = "face") -> dict:
    """Write work/<take>/export/<capture>/ and (install=True) mirror it into head-movement StreamingAssets.
    smplx_dir: a contract folder under work/<take> other than smplx/ (e.g. smplx_detail: refined hands + face).
    faces: when work/<take>/smplx_face/ exists, add the face block (dancecap.face_export: smplx_<role>_face.bin
    blendshapes + weights, jaw / eye rotations from smplx_face, the face_albedo texture variant: face = portrait look,
    videolook, bake)."""
    work = PACKAGE_ROOT / "work" / take
    sdir = work / (smplx_dir or "smplx")
    capture = validate_capture_name(capture or take)
    warnings: list[str] = []
    data = {}
    for r in ROLES:
        p = sdir / f"{r}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing (run the mamma stage, or `python -m dancecap synthetic`)")
        with np.load(p, allow_pickle=True) as z:
            data[r] = {k: z[k] for k in z.files}
    wfs = work / "cameras" / "world_from_sfm.json"
    if wfs.exists() and min((sdir / f"{r}.npz").stat().st_mtime for r in ROLES) < wfs.stat().st_mtime:
        warnings.append("smplx/ is older than cameras/world_from_sfm.json: the fit may be in a previous world frame")
    t = np.asarray(data["lead"]["t"], float)
    if len(data["follow"]["t"]) != len(t) or np.abs(np.asarray(data["follow"]["t"]) - t).max() > 1e-6:
        raise ValueError("lead and follow npz are not on the same time grid")
    if np.any(np.diff(t) <= 0):
        raise ValueError("smplx t must be strictly increasing")
    audio_path, t0, audio_source = resolve_audio(work, Path(audio) if audio else None, audio_t0, allow_no_audio)
    T = len(t)
    fps = float(1.0 / np.median(np.diff(t)))
    time_to_audio = -float(t0) + 0.0  # audio time = reference t - (reference time of audio sample 0); never -0.0
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
        "smplx": {}, "smplx_skin": {}, "provenance": {"take": take, "source": "dancecap export_unity",
                                                     "audio": audio_source, "smplx_dir": sdir.name},
    }
    report: dict = {"capture": capture, "frames": T, "fps": fps, "dancers": {}, "audio": audio_source,
                    "warnings": warnings}
    albedo = {}
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
        skin = shaped_skin(mp, d["betas"])
        if textures:
            skin = textured_skin(skin, mp)
        skin_u = unity_skin(skin)
        write_skin(out / f"smplx_{r}_skin.bin", r, skin_u)
        P, _ = fk_unity(skin_u.joints, skin_u.parents, transl_u, quats_u)
        fk_err = float(np.abs(P - joints_u).max())
        manifest["smplx"][r] = f"smplx_{r}.bin"
        manifest["smplx_skin"][r] = f"smplx_{r}_skin.bin"
        report["dancers"][r] = {"model": str(mp), "betas": len(d["betas"]), "fk_max_err_mm": round(fk_err * 1000, 3),
                                "skin_vertices": int(len(skin_u.vertices)), "uv_split": skin_u.uv is not None}
        tex = work / "avatar" / f"{r}_albedo.png"
        if textures and skin_u.uv is not None and tex.exists():
            shutil.copy2(tex, out / f"{r}_albedo.png")
            albedo[r] = f"{r}_albedo.png"
        if fk_err > 0.01:
            print(f"[export] WARNING {r}: FK of the exported rotations misses the npz joints by {fk_err * 1000:.1f} mm "
                  f"(model mismatch? pass --model)")
    if albedo:
        manifest["smplx_albedo"] = albedo
        rep_path = work / "avatar" / "report.json"
        if rep_path.exists():
            fr = json.loads(rep_path.read_text()).get("smplx_freshness", {})
            if fr.get("stale"):
                warnings.append("avatar albedo was baked on a stale smplx/ (avatar/report.json smplx_freshness)")
    if faces and (work / "smplx_face").exists():
        from .face_export import add_faces
        report["faces"] = add_faces(work, out, manifest, face_albedo, warnings)
    (out / "video_meta.json").write_text(json.dumps({"duration": manifest["duration"], "frame_count": T}))

    timing_path = work / "timing" / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    grid = _beats_audio(work, beats, timing, time_to_audio)
    if grid:
        (out / "zouk-time-analysis.json").write_text(json.dumps(grid))
        manifest["beats"] = "zouk-time-analysis.json"
    if timing is not None:
        (out / "timing.json").write_text(json.dumps(
            _timing_for_unity(timing, work / "timing" / "kinematics.npz", t, warnings), separators=(",", ":")))
        manifest["timing"] = "timing.json"
    phys = work / "physics" / "physics.json"
    if phys.exists() and _physics_on_grid(phys, t, warnings):
        shutil.copy2(phys, out / "physics.json")
        manifest["physics"] = "physics.json"
    if audio_path is not None:
        shutil.copy2(audio_path, out / "audio.wav")
        manifest["audio"] = "audio.wav"
    if room:
        manifest.update(export_room(work, out, warnings))
        if "room" in manifest:
            report["room"] = manifest["room"]
    (out / "capture.json").write_text(json.dumps(manifest, indent=2))
    report["out"] = str(out)
    report["files"] = sorted(p.name for p in out.iterdir())
    report["bytes"] = int(sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
    for w in warnings:
        print(f"[export] WARNING {w}")

    if install:
        dst = Path(streaming_assets) / capture
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(out, dst)
        report["installed"] = str(dst)
    return report
