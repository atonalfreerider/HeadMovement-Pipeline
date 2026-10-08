"""Shared fixtures of the core farm tests (test_farm_timeline / _chunks): a synthetic dance (grid, work dirs, motion npz
files) in a temp package root. No pods, no Runpod calls, no real take touched (farm.PACKAGE_ROOT is pointed at the temp tree).
(tests/_farm_fixtures.py is the analysis stage's helper module; this one is the core stage's.)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dancecap.farm as farm  # noqa: E402
from dancecap.farm import timeline as tl  # noqa: E402

ANCHOR = 10.0
K0, K1 = -90, 419          # T = 510 frames = 17 cells of 30
REF_K = (0, 299)           # the reused reference span = cells 3..12


def make_spec(name: str = "dance", reference: str | None = "ref", k0: int = K0, k1: int = K1, **kw) -> tl.DanceSpec:
    return tl.DanceSpec(name=name, grid=tl.Grid(ANCHOR, k0, k1), reference=reference,
                        ref_k=REF_K if reference else None, cams=["01", "02", "03", "04"],
                        segments=kw.pop("segments", {}), cfg=kw.pop("cfg", {}), **kw)


def patch_root(monkeypatch, tmp_path: Path) -> Path:
    """Point farm.work_dir / farm_dir / ... at tmp_path/dancecap (timeline / chunks resolve them through farm)."""
    root = tmp_path / "dancecap"
    (root / "work").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(farm, "PACKAGE_ROOT", root)
    return root


def write_motion(d: Path, ks, *, anchor: float = ANCHOR, n_cams=8, reproj=6.0, flag=None, repaired=None,
                 owner=None, T_extra: dict | None = None) -> None:
    """lead.npz + follow.npz of the smplx contract's per-frame fields on the lattice frames `ks`."""
    d.mkdir(parents=True, exist_ok=True)
    ks = np.asarray(list(ks), int)
    t = anchor + ks / 30.0
    for role in ("lead", "follow"):
        a = {"t": t, "reproj_px": np.full(len(ks), reproj, float) if np.isscalar(reproj) else np.asarray(reproj, float),
             "repaired": np.zeros(len(ks), bool) if repaired is None else np.asarray(repaired, bool)}
        if n_cams is not None:
            a["n_cams"] = np.full(len(ks), n_cams, np.int16) if np.isscalar(n_cams) else np.asarray(n_cams, np.int16)
        if flag is not None:
            a["quality_flag"] = np.array(flag) if not np.isscalar(flag) else np.array([flag] * len(ks))
        if owner is not None:
            a["segment_owner"] = np.array(owner) if not np.isscalar(owner) else np.array([owner] * len(ks))
        a.update(T_extra or {})
        np.savez(d / f"{role}.npz", **a)


def write_prepare(work: Path, k_lo: int, T: int, cams, *, anchor: float = ANCHOR, tag: str = "c4",
                  cams_per_frame: dict | None = None) -> None:
    g = {"t0": anchor + k_lo / 30.0, "T": T, "hz": 30.0, "trim": {}}
    if cams_per_frame:
        g["trim"]["cams_per_frame"] = cams_per_frame
    prep = {"take": work.name, "grid": g, "config": "ma3d_config.yaml", "sets": {tag: {"seq": f"{work.name}_{tag}", "cams": list(cams)}},
            "eval": {"seq": f"{work.name}_{tag}", "cams": list(cams)}}
    (work / "mamma").mkdir(parents=True, exist_ok=True)
    (work / "mamma" / "prepare.json").write_text(json.dumps(prep))
    (work / "mamma" / "ma3d_config.yaml").write_text("optim: {}\n")


def write_cameras(root: Path, dance: str, spans: dict) -> None:
    p = root / "work" / dance / "farm" / "cameras.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"spans": spans}))
