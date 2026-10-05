"""MAMMA adapter: synthetic multi-view capture in, per-person SMPL-X out.

Input layout written into the MAMMA repo (see mamma/docs/YOUR-DATA.md):

    mamma/data/atlas/<session>.json               capture descriptor (stem = dataset name)
    mamma/data/atlas/<session>/calib.yaml         MAMMA YAML calibration, cam2world, wxyz
    mamma/data/atlas/<session>/take/videos/vNN.mp4

MAMMA has no time sync - it assumes frame-locked cameras, which Atlas renders
are by construction (one reconstruction, many viewpoints, same timestamps).

Output: output/ma_3d/<out_tag>/<session>/.../verts_joints_body_id-NN.npz
(pred_joints (F,J,3), stage frame, metres) and smplx_params_body_id-NN.npz.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .cameras import Camera, to_mamma_yaml

SEQ_NAME = "take"


def capture_paths(mamma_root: Path, session: str) -> tuple[Path, Path]:
    base = mamma_root / "data" / "atlas"
    return base / f"{session}.json", base / session


def write_capture(mamma_root: Path, session: str, views: dict[str, Path], cams: list[Camera], fps: float,
                  link: bool = True) -> Path:
    capture_json, root = capture_paths(mamma_root, session)
    vids = root / SEQ_NAME / "videos"
    vids.mkdir(parents=True, exist_ok=True)
    names = [c.name for c in cams]
    if set(names) != set(views):
        raise ValueError(f"views {sorted(views)} != cameras {sorted(names)}")
    for name, src in views.items():
        dst = vids / f"{name}.mp4"
        if dst.exists():
            dst.unlink()
        if link:
            try:
                os.link(src, dst)  # hardlink: no copy of large renders on the same volume
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)
    (root / "calib.yaml").write_text(to_mamma_yaml(cams))
    capture_json.write_text(json.dumps({
        "capture_root": session,
        "calib": f"{session}/calib.yaml",
        "cam_fps": fps,
        "sequences": {"000": {"name": SEQ_NAME}},
        "cams": names,
        "videos_subdir": "videos",
        "up_axis": "z",
    }, indent=2))
    return capture_json


def run_command(mamma_root: Path, capture_json: Path, preset: str, out_tag: str, conda_env: str,
                conda: str = "conda") -> list[str]:
    return [conda, "run", "--no-capture-output", "-n", conda_env, "python", "-m", "inference", "run",
            "--cfg", preset, "--capture", str(capture_json.relative_to(mamma_root)).replace("\\", "/"),
            "--out-tag", out_tag, "-v"]


def run(mamma_root: Path, capture_json: Path, preset: str, out_tag: str, conda_env: str, conda: str = "conda",
        docker_hint: bool = True) -> None:
    cmd = run_command(mamma_root, capture_json, preset, out_tag, conda_env, conda)
    print("[mamma] " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=mamma_root)
    if proc.returncode != 0:
        msg = f"MAMMA exited with {proc.returncode}."
        if docker_hint:
            msg += (" On Windows, detectron2/pytorch_sdf CUDA builds often fail - the Docker path "
                    "(mamma/docker/README.md, engine: docker per step) is the reliable route.")
        raise RuntimeError(msg)


def find_outputs(mamma_root: Path, session: str, out_tag: str) -> dict[int, Path]:
    base = mamma_root / "output" / "ma_3d" / out_tag / session
    found = {}
    for p in base.rglob("verts_joints_body_id-*.npz"):
        m = re.search(r"body_id-(\d+)", p.name)
        if m:
            found[int(m.group(1))] = p
    return dict(sorted(found.items()))


def load_people(mamma_root: Path, session: str, out_tag: str) -> dict[int, dict]:
    """person id -> {"joints": (F,J,3) stage metres, "smplx": dict of params or None}."""
    people = {}
    for pid, jpath in find_outputs(mamma_root, session, out_tag).items():
        with np.load(jpath) as d:
            joints = np.asarray(d["pred_joints"], dtype=np.float64)
        if joints.ndim == 4:  # (F,1,J,3)
            joints = joints[:, 0]
        ppath = jpath.with_name(jpath.name.replace("verts_joints", "smplx_params"))
        params = None
        if ppath.exists():
            with np.load(ppath, allow_pickle=True) as d:
                params = {k: d[k] for k in d.files}
        people[pid] = {"joints": joints, "smplx": params}
    if not people:
        raise FileNotFoundError(f"no MAMMA ma_3d outputs for {session!r} tag {out_tag!r} under {mamma_root / 'output'}")
    return people
