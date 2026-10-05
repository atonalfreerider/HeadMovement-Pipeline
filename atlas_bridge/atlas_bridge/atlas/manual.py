"""File-drop Atlas provider for early access (web UI, no API).

submit() writes work/atlas/request/: the synced input videos, the exact virtual
camera trajectories in both OpenCV and OpenGL conventions, and a checklist.
You render each view in Atlas and drop the outputs into work/atlas/renders/:

    renders/v00.mp4 ... vNN.mp4     one locked-off video per virtual camera
    renders/environment.ply         optional static splat export
    renders/splats/frame_00000.ply  optional per-frame splats (if Atlas offers them)

fetch() validates frame count / fps / resolution so MAMMA gets frame-locked views.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..cameras import to_trajectory_json
from ..sync import probe
from .base import AtlasJob, AtlasProvider, AtlasResult


class ManualAtlasProvider(AtlasProvider):
    @property
    def request_dir(self) -> Path:
        return self.work_dir / "atlas" / "request"

    @property
    def renders_dir(self) -> Path:
        return self.work_dir / "atlas" / "renders"

    def submit(self, job: AtlasJob) -> str:
        req = self.request_dir
        req.mkdir(parents=True, exist_ok=True)
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        for name, path in job.source_videos.items():
            dst = req / "inputs" / f"{name}.mp4"
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists() or dst.stat().st_mtime < path.stat().st_mtime:
                shutil.copy2(path, dst)
        traj = to_trajectory_json(job.virtual_cameras, job.stage_to_scene, job.frame_count, job.fps)
        (req / "virtual_cameras.json").write_text(json.dumps(traj, indent=2))

        v0 = job.virtual_cameras[0]
        lines = [
            f"# Atlas render request - {job.session}",
            "",
            f"Inputs (synced, {job.fps:g} fps constant, {job.frame_count} frames each): `inputs/`",
            "",
            f"Render **{len(job.virtual_cameras)} locked-off views**, full duration, "
            f"{v0.width}x{v0.height} @ {job.fps:g} fps, output frame i == input frame i.",
            "Camera poses are in `virtual_cameras.json` (both `c2w_opencv` and `c2w_opengl`, scene frame).",
            "",
            "Drop results into `../renders/` named exactly:",
            "",
        ]
        for c in job.virtual_cameras:
            lines.append(f"- [ ] `{c.name}.mp4`  hfov {c.hfov_deg:.1f} deg, centre "
                         f"({c.C[0]:.2f}, {c.C[1]:.2f}, {c.C[2]:.2f}) m stage frame")
        lines += ["", "Optional:", "- [ ] `environment.ply` static splat export (scene frame)",
                  "- [ ] `splats/frame_00000.ply ...` per-frame splats, if available",
                  "- [ ] `source_cameras.json` Atlas' estimated poses for the input videos "
                  "(`{name: 4x4 c2w opencv}`) - enables automatic stage alignment"]
        (req / "REQUEST.md").write_text("\n".join(lines) + "\n")
        return "manual"

    def fetch(self, job: AtlasJob, job_id: str) -> AtlasResult | None:
        rd = self.renders_dir
        missing = [c.name for c in job.virtual_cameras if not (rd / f"{c.name}.mp4").exists()]
        if missing:
            print(f"[atlas] waiting for {len(missing)}/{len(job.virtual_cameras)} renders in {rd}: "
                  f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
            return None

        problems = []
        for c in job.virtual_cameras:
            info = probe(rd / f"{c.name}.mp4")
            frames = round(info["duration"] * info["fps"])
            if abs(info["fps"] - job.fps) > 0.01:
                problems.append(f"{c.name}: {info['fps']:.3f} fps, expected {job.fps:g}")
            if abs(frames - job.frame_count) > 1:
                problems.append(f"{c.name}: ~{frames} frames, expected {job.frame_count}")
            if (info["width"], info["height"]) != (c.width, c.height):
                problems.append(f"{c.name}: {info['width']}x{info['height']}, expected {c.width}x{c.height} "
                                "(intrinsics would be wrong)")
        if problems:
            raise RuntimeError("Atlas renders don't match the request:\n  " + "\n  ".join(problems))

        env = rd / "environment.ply"
        seq = rd / "splats"
        src_poses = {}
        if (rd / "source_cameras.json").exists():
            import numpy as np
            src_poses = {k: np.array(v) for k, v in json.loads((rd / "source_cameras.json").read_text()).items()}
        return AtlasResult(
            views={c.name: rd / f"{c.name}.mp4" for c in job.virtual_cameras},
            environment_splat=env if env.exists() else None,
            splat_sequence_dir=seq if seq.is_dir() and any(seq.glob("*.ply")) else None,
            source_camera_poses=src_poses,
        )
