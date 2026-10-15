"""A stand-in delivery for plumbing tests: the take's own multi-camera fit drawn as mannequins at every requested camera.

    python -m dancecap.mono4d oracle <take> --run R      -> drop/views/<camera>/000000.jpg ... (one per grid time)

It plays a perfect provider: geometry exactly right, no hallucination, no likeness (capsule mannequins from the
reference joints, calibrate.draw). It exercises request -> fetch -> package -> judge -> gate on real cameras and real
motion before any vendor delivers. It is not MAMMA input: MAMMA's 2D network is trained on photos, not mannequins.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT
from . import calibrate, rig
from .providers import load_request, run_dir


def deliver(take: str, run: str, reference: Path | None = None) -> dict:
    work = PACKAGE_ROOT / "work" / take
    rd = run_dir(work, run)
    req = load_request(rd)
    ref = Path(reference) if reference else work / "smplx"
    J = {r: np.load(ref / f"{r}.npz") for r in ("lead", "follow")}
    grid = np.asarray(req.grid)
    idx = {r: np.array([int(np.argmin(np.abs(z["t"] - g))) for g in grid]) for r, z in J.items()}
    specs = rig.specs_from_trajectory(json.loads((rd / "request" / "target_trajectory.json").read_text()))
    for s in specs:
        d = rd / "drop" / "views" / s.name
        d.mkdir(parents=True, exist_ok=True)
        for k in range(len(grid)):
            one = rig.ViewSpec(s.name, s.kind, s.K[k:k + 1], s.R[k:k + 1], s.t[k:k + 1], s.width, s.height)
            bodies = [(J[r]["joints"][idx[r][k], :22], calibrate.COLOURS[r], 1.0, set(), True) for r in J]
            calibrate.draw(bodies, one).save(d / f"{k:06d}.jpg", "JPEG", quality=90)
    note = {"provider": "oracle", "reference": str(ref), "views": [s.name for s in specs], "frames": len(grid)}
    (rd / "drop" / "ORACLE.json").write_text(json.dumps(note, indent=1))
    return note
