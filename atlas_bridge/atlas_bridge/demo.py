"""Synthetic capture so the Unity side can be exercised before Atlas/MAMMA data exists.

Two SMPL-X bodies (real body model, procedural choreography) doing a zouk-ish
basic while the couple slowly rotates, with an elastic partner distance (so the
connection alternates tension/compression) and a follow who steps ~90 ms late in
measures 3-4 of every 8 (so the floor timing check has something to flag).
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from . import cameras
from .beats import write_beats, zouk_grid

BPM = 88.0


_FK_SCRIPT = """
import sys, numpy as np, smplx, torch
model_file, pose_file, out_file = sys.argv[1:4]
pose = np.load(pose_file)
model = smplx.create(model_file, model_type="smplx", gender="neutral", use_pca=False,
                     flat_hand_mean=True, num_betas=10, batch_size=len(pose))
with torch.no_grad():
    out = model(body_pose=torch.tensor(pose, dtype=torch.float32), global_orient=torch.zeros(len(pose), 3))
np.save(out_file, out.joints.numpy())
"""


def _smplx_joints(model_file: Path, body_pose: np.ndarray, body_model_python: str) -> np.ndarray:
    """SMPL-X forward kinematics in the interpreter that has smplx+torch (usually the mamma env)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        pose_file, out_file = Path(d) / "pose.npy", Path(d) / "joints.npy"
        np.save(pose_file, body_pose.astype(np.float32))
        subprocess.run([body_model_python, "-c", _FK_SCRIPT, str(model_file), str(pose_file), str(out_file)],
                       check=True)
        return np.load(out_file).astype(np.float64)  # (F,J,3) SMPL-X canonical: y-up, facing +z


def _set(pose: np.ndarray, joint: int, axis: int, angle: np.ndarray) -> None:
    pose[:, (joint - 1) * 3 + axis] += angle


def choreography(frames: int, fps: float, role: str) -> np.ndarray:
    t = np.arange(frames) / fps
    beat = 60.0 / BPM
    measure = 4 * beat
    pose = np.zeros((frames, 63))
    late = 0.09 if role == "follow" else 0.0
    m_idx = np.floor(t / measure).astype(int) % 8
    tt = t - np.where((m_idx == 2) | (m_idx == 3), late, 0.0)
    # alternating steps, one per beat: swing leg flexes hip + knee
    phase = (tt / beat) % 2.0
    lift_l = np.clip(np.sin(np.pi * phase), 0, None) * (phase < 1)
    lift_r = np.clip(np.sin(np.pi * (phase - 1)), 0, None) * (phase >= 1)
    for hip, knee, lift in ((1, 4, lift_l), (2, 5, lift_r)):
        _set(pose, hip, 0, -0.45 * lift)
        _set(pose, knee, 0, 0.9 * lift)
    # weight-shift sway through the spine
    _set(pose, 3, 2, 0.06 * np.sin(np.pi * tt / beat))
    if role == "follow":
        # zouk head movement: circular torso/head roll over two measures
        a = 2 * np.pi * t / (2 * measure)
        for j, g in ((6, 0.12), (9, 0.18), (12, 0.2), (15, 0.25)):
            _set(pose, j, 0, g * np.sin(a))
            _set(pose, j, 2, g * np.cos(a))
    # frame: left arms forward and down (hands meet), right arms around partner
    side = 1.0 if role == "lead" else -1.0
    _set(pose, 16, 1, -0.9)
    _set(pose, 16, 2, -0.9)
    _set(pose, 18, 1, -0.9 * side)
    _set(pose, 17, 1, 1.1)
    _set(pose, 17, 2, 0.5)
    _set(pose, 19, 1, 1.2)
    return pose


def make_demo(out_dir: Path, smplx_model: Path, body_model_python: str, seconds: float = 24.0,
              fps: float = 30.0) -> Path:
    from .unity_export import export_capture, smplx_to_smpl24

    frames = int(seconds * fps)
    t = np.arange(frames) / fps
    measure = 4 * 60.0 / BPM
    people = {}
    for role in ("lead", "follow"):
        j = _smplx_joints(smplx_model, choreography(frames, fps, role), body_model_python)
        j = j[..., [0, 2, 1]] * [1.0, -1.0, 1.0]  # SMPL-X y-up (x,y,z) -> stage z-up (x,-z,y)
        j = smplx_to_smpl24(j)
        j[..., 2] -= j[0, [10, 11], 2].min()  # rest feet on the floor
        people[role] = j

    # couple rotates once per 4 measures; partner distance breathes each measure
    yaw = 2 * np.pi * t / (4 * measure)
    half_gap = 0.5 * (0.62 + 0.12 * np.sin(2 * np.pi * t / measure))
    for role, sign in (("lead", -1.0), ("follow", 1.0)):
        facing = yaw + (0.0 if role == "lead" else np.pi)  # canonical faces -Y in stage
        c, s = np.cos(facing), np.sin(facing)
        R = np.stack([np.stack([c, -s, np.zeros_like(c)], -1),
                      np.stack([s, c, np.zeros_like(c)], -1),
                      np.stack([np.zeros_like(c), np.zeros_like(c), np.ones_like(c)], -1)], 1)
        j = people[role]
        pelvis_xy = j[:, :1, :].copy()
        pelvis_xy[..., 2] = 0
        j = np.einsum("fij,fnj->fni", R, j - pelvis_xy)
        offset = np.stack([sign * half_gap * np.sin(yaw), -sign * half_gap * np.cos(yaw), np.zeros_like(t)], -1)
        people[role] = j + offset[:, None, :]

    out_dir.mkdir(parents=True, exist_ok=True)
    audio = out_dir / "_demo_audio.wav"
    _write_groove(audio, seconds + 1.0)
    beats = out_dir / "_demo_beats.json"
    write_beats(beats, zouk_grid(BPM, seconds + 1.0))

    # Environment splat in a made-up "Atlas scene frame" (y-up, 0.5 units per metre, shifted, yawed) so the
    # scene->stage->unity path gets exercised exactly as it will be with a real Atlas export.
    from .align import rotation_between
    from .splat_io import studio, write_3dgs_ply
    rng = np.random.default_rng(3)
    xyz_stage, rgb = studio(rng)
    scene_to_stage = np.eye(4)
    yaw = np.radians(30)
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    scene_to_stage[:3, :3] = 2.0 * R_yaw @ rotation_between(np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    scene_to_stage[:3, 3] = [0.3, -0.2, 0.0]
    stage_to_scene = np.linalg.inv(scene_to_stage)
    xyz_scene = np.einsum("ij,nj->ni", stage_to_scene[:3, :3], xyz_stage) + stage_to_scene[:3, 3]
    env = out_dir / "_demo_environment.ply"
    write_3dgs_ply(env, xyz_scene, rgb, sigma=0.012, opacity=0.85)

    cams = cameras.build_rig({})
    export_capture(out_dir, people["lead"], people["follow"], fps, audio, 0.0, beats,
                   cameras_unity=cameras.to_unity_json(cams), environment_splat=env,
                   scene_to_stage=scene_to_stage, floor_z=0.0,
                   extra_meta={"source": "atlas_bridge demo (synthetic)", "bpm": BPM})
    for tmp in (audio, beats, env):
        tmp.unlink()
    return out_dir


def _write_groove(path: Path, seconds: float, sr: int = 44100) -> None:
    n = int(seconds * sr)
    y = np.zeros(n)
    rng = np.random.default_rng(1)
    for time, kind in zouk_grid(BPM, seconds):
        i = int(time * sr)
        if kind == 1:  # kick
            L = int(0.25 * sr)
            tt = np.arange(L) / sr
            hit = np.sin(2 * np.pi * (55 + 60 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12) * 0.9
        elif kind == 2:  # snare-ish
            L = int(0.18 * sr)
            tt = np.arange(L) / sr
            hit = (rng.standard_normal(L) * 0.35 + np.sin(2 * np.pi * 190 * tt) * 0.3) * np.exp(-tt * 22)
        else:  # shaker
            L = int(0.05 * sr)
            hit = rng.standard_normal(L) * 0.08 * np.exp(-np.arange(L) / sr * 80)
        y[i:i + L] += hit[: max(0, min(L, n - i))]
    # a soft pad so it isn't just percussion
    tt = np.arange(n) / sr
    pad = sum(np.sin(2 * np.pi * f * tt) for f in (220.0, 277.18, 329.63)) * 0.04
    y = np.clip(y + pad * (0.6 + 0.4 * np.sin(2 * np.pi * tt / (4 * 4 * 60 / BPM))), -1, 1)
    pcm = (np.stack([y, y], 1) * 32000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
