"""Shared helpers of the tests/test_farm_*.py files: a throw-away dancecap work tree and synthetic smplx-contract motion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dancecap.farm import timeline as tl

ANCHOR = 30.0
MODULES = ("dancecap.farm", "dancecap.farm.timeline", "dancecap.farm.chunks", "dancecap.farm.analysis", "dancecap.farm.world",
           "dancecap.farm.state", "dancecap.farm.cli")


def patch_root(monkeypatch, root: Path) -> Path:
    """Point every farm module's PACKAGE_ROOT at `root` (work/, takes/, worlds/ live under it)."""
    import importlib
    for name in MODULES:
        mod = importlib.import_module(name)
        if hasattr(mod, "PACKAGE_ROOT"):
            monkeypatch.setattr(mod, "PACKAGE_ROOT", root, raising=True)
    (root / "work").mkdir(parents=True, exist_ok=True)
    return root


def make_spec(name="dance", k0=0, k1=299, reference=None, ref_k=None, segments=None, refined="rf") -> tl.DanceSpec:
    return tl.DanceSpec(name=name, grid=tl.Grid(ANCHOR, k0, k1), reference=reference, ref_k=ref_k, cams=["01", "02", "03"],
                        segments=segments or {}, refined_dir=refined,
                        cfg={"name": name, "dancers": {"lead": {}, "follow": {}}, "dance": {}, "source_dir": "x", "alignment": "y"})


def write_motion(d: Path, k0: int, k1: int, *, reproj=6.0, flag=None, n_cams=None, repaired=None, owner=None, hz=30.0):
    """lead/follow npz with t on the lattice ANCHOR + k/hz for k0..k1 (inclusive) and the per-frame arrays timeline reads."""
    d.mkdir(parents=True, exist_ok=True)
    ks = np.arange(k0, k1 + 1)
    T = len(ks)
    for role in ("lead", "follow"):
        a = {"t": ANCHOR + ks / hz, "reproj_px": np.full(T, reproj, float) if np.isscalar(reproj) else np.asarray(reproj, float),
             "transl": np.zeros((T, 3)), "betas": np.zeros(16)}
        if flag is not None:
            a["quality_flag"] = np.full(T, flag, np.int8) if np.isscalar(flag) else np.asarray(flag, np.int8)
        if n_cams is not None:
            a["n_cams"] = np.full(T, n_cams, np.int16) if np.isscalar(n_cams) else np.asarray(n_cams, np.int16)
        if repaired is not None:
            a["repaired"] = np.asarray(repaired, bool)
        if owner is not None:
            a["segment_owner"] = np.full(T, owner, np.int16)
        np.savez(d / f"{role}.npz", **a)


def write_prepare(take_work: Path, k0: int, k1: int, cams, hz=30.0, per_cam=None):
    """mamma/prepare.json of a segment take (grid + camera spans) as timeline.prepared_ncams reads it."""
    t0 = ANCHOR + k0 / hz
    T = k1 - k0 + 1
    cpf = per_cam or {c: [t0 - 1.0, ANCHOR + (k1 + 1) / hz + 1.0] for c in cams}
    (take_work / "mamma").mkdir(parents=True, exist_ok=True)
    doc = {"take": take_work.name, "grid": {"t0": t0, "T": T, "hz": hz, "trim": {"cams_per_frame": cpf}},
           "sets": {"all": {"seq": f"{take_work.name}_all", "cams": list(cams)}}}
    (take_work / "mamma" / "prepare.json").write_text(json.dumps(doc))
    return doc
