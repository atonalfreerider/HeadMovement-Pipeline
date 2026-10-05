"""Write a head-movement capture folder (Assets/StreamingAssets/<Capture>/).

Backwards compatible with the existing loader (figure1.json, figure2.json,
video_meta.json, audio.wav, zouk-time-analysis.json) and adds capture.json,
which the new loader reads for fps, audio offset, splats and virtual cameras.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .cameras import STAGE_TO_UNITY, stage_to_unity

# SMPL-X joint indices (smplx.joint_names.JOINT_NAMES). 0..21 are identical to SMPL.
SMPLX_LEFT_MIDDLE1 = 28
SMPLX_RIGHT_MIDDLE1 = 43
SMPL_L_FOOT, SMPL_R_FOOT, SMPL_L_ANKLE, SMPL_R_ANKLE, SMPL_HEAD = 10, 11, 7, 8, 15


def smplx_to_smpl24(joints: np.ndarray) -> np.ndarray:
    """(F,J>=55,3) SMPL-X joints -> (F,24,3) SMPL order used by Dancer.cs / Pose.cs.

    SMPL's L_Hand/R_Hand (22/23) sit at the knuckles; SMPL-X has no such joint,
    so the middle-finger MCP (left/right_middle1) stands in.
    """
    if joints.shape[1] == 24:
        return joints
    if joints.shape[1] < 55:
        raise ValueError(f"expected SMPL-X joints (>=55), got {joints.shape[1]}")
    return np.concatenate([joints[:, :22], joints[:, [SMPLX_LEFT_MIDDLE1]], joints[:, [SMPLX_RIGHT_MIDDLE1]]], axis=1)


def standing_height(j24: np.ndarray) -> float:
    """Median head-to-floor height (stage frame, z-up); a rough size proxy."""
    feet = np.minimum(j24[:, SMPL_L_FOOT, 2], j24[:, SMPL_R_FOOT, 2])
    return float(np.median(j24[:, SMPL_HEAD, 2] - feet))


def assign_roles(people: dict[int, np.ndarray], lead: str | int, follow: str | int) -> tuple[int, int, str]:
    """Pick lead/follow ids. 'auto' = taller dancer leads (override in session.toml when wrong)."""
    ids = list(people)
    if len(ids) < 2:
        raise ValueError(f"need two dancers, MAMMA found {len(ids)}: {ids}")
    if lead != "auto" and follow != "auto":
        return int(lead), int(follow), "roles from session.toml"
    heights = {i: standing_height(people[i]) for i in ids}
    by_height = sorted(ids, key=lambda i: -heights[i])
    extra = f"; ignoring extra people {by_height[2:]}" if len(ids) > 2 else ""
    note = "auto roles by height " + ", ".join(f"id {i}: {heights[i]:.2f} m" for i in by_height) + extra
    return by_height[0], by_height[1], note


def floor_offset(dancers: list[np.ndarray], percentile: float = 2.0) -> float:
    """Height of the floor in stage z, from the lowest foot joints over the take."""
    feet = np.concatenate([d[:, [SMPL_L_FOOT, SMPL_R_FOOT], 2].ravel() for d in dancers])
    return float(np.percentile(feet, percentile))


def to_figure_json(j24_unity: np.ndarray) -> list:
    return [[{"x": round(float(p[0]), 5), "y": round(float(p[1]), 5), "z": round(float(p[2]), 5)} for p in frame]
            for frame in j24_unity]


def stage_to_unity_matrix(stage_from_scene: np.ndarray | None) -> list[float]:
    """Row-major 4x4 mapping scene (Atlas) points -> unity points."""
    M = np.eye(4)
    M[:3, :3] = STAGE_TO_UNITY
    if stage_from_scene is not None:
        M = M @ stage_from_scene
    return [float(v) for v in M.ravel()]


def export_capture(out_dir: Path, lead: np.ndarray, follow: np.ndarray, fps: float,
                   audio: Path | None, audio_offset: float, beats: Path | None,
                   cameras_unity: dict | None = None,
                   environment_splat: Path | None = None, splat_sequence_dir: Path | None = None,
                   scene_to_stage: np.ndarray | None = None, extra_meta: dict | None = None,
                   floor_z: float | None = None, smplx: dict | None = None) -> Path:
    """lead/follow: (F,24,3) stage-frame joints. Returns the capture folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = min(len(lead), len(follow))
    lead, follow = lead[:frames].copy(), follow[:frames].copy()

    fz = floor_offset([lead, follow]) if floor_z is None else floor_z
    lead[..., 2] -= fz
    follow[..., 2] -= fz

    (out_dir / "figure1.json").write_text(json.dumps(to_figure_json(stage_to_unity(lead)), separators=(",", ":")))
    (out_dir / "figure2.json").write_text(json.dumps(to_figure_json(stage_to_unity(follow)), separators=(",", ":")))
    duration = frames / fps
    (out_dir / "video_meta.json").write_text(json.dumps({"duration": duration, "frame_count": frames}))

    manifest: dict = {
        "version": 2,
        "fps": fps,
        "frame_count": frames,
        "duration": duration,
        "audio": None,
        "audio_offset": audio_offset,
        "beats": None,
        "dancers": [{"role": "lead", "joints": "figure1.json"}, {"role": "follow", "joints": "figure2.json"}],
        "coordinate_system": "unity: left-handed, y-up, metres, floor y=0",
    }
    if audio is not None:
        shutil.copy2(audio, out_dir / "audio.wav")
        manifest["audio"] = "audio.wav"
    if beats is not None:
        shutil.copy2(beats, out_dir / "zouk-time-analysis.json")
        manifest["beats"] = "zouk-time-analysis.json"
    if cameras_unity is not None:
        cams = {"cameras": [dict(c, position=dict(c["position"], y=c["position"]["y"] - fz))
                            for c in cameras_unity["cameras"]]}
        (out_dir / "cameras.json").write_text(json.dumps(cams, indent=1))
        manifest["virtual_cameras"] = "cameras.json"

    # Splats stay in the Atlas scene frame; Unity applies scene->unity at load time.
    to_unity = None
    if scene_to_stage is not None:
        S = scene_to_stage.copy()
        S[2, 3] -= fz  # floor shift is along stage z
        to_unity = stage_to_unity_matrix(S)
    if environment_splat is not None:
        shutil.copy2(environment_splat, out_dir / "environment.ply")
        manifest["environment_splat"] = {"path": "environment.ply", "scene_to_unity": to_unity}
    if splat_sequence_dir is not None:
        dst = out_dir / "splats"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(splat_sequence_dir, dst)
        manifest["splat_sequence"] = {"dir": "splats", "pattern": "frame_{0:D5}.ply", "fps": fps,
                                      "scene_to_unity": to_unity}
    if smplx is not None:
        np.savez_compressed(out_dir / "smplx_params.npz", **smplx)
        manifest["smplx_params"] = "smplx_params.npz"
    if extra_meta:
        manifest["provenance"] = extra_meta
    (out_dir / "capture.json").write_text(json.dumps(manifest, indent=2))
    return out_dir
