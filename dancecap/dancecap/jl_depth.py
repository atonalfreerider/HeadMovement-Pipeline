r"""john_lula: how much does the 3D change when the camera's focal length is a different guess? (agent J3b, 2026-10-11). Pure numpy on two fits.

One camera cannot measure its own focal length from a dancer: distances scale with it (J2: 3.6 m at 706 px, 4.2 at 808, 5.35 at 1010; the camera height does not). The fit of a window is
re-solved with every frame's focal times s (jl_front.step_cameras, JL_FOCAL_SCALE; rotation and camera centre unchanged, betas pinned to the same statures); this module compares that
re-solve with the adopted fit of the same frames:

    python -m dancecap.jl_depth compare --base john_lula --src smplx_rf2_noslip --var john_lula_jlfp110 --var-src smplx --k0 K0 --k1 K1

per dancer: the horizontal range from the camera centre (ratio variant / base), the change of the travelled floor path (path length of the pelvis on the floor), the pose seen from the pelvis
(MPJPE of the joints relative to the pelvis, i.e. does the dance still look the same), the change of speeds, the heights (does he still stand 1.8 m tall).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

ROLES = ("lead", "follow")


def _load(take: str, src: str) -> dict:
    d = PACKAGE_ROOT / "work" / take / src
    return {r: dict(np.load(d / f"{r}.npz", allow_pickle=True)) for r in ROLES}


def align(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i in a, j in b) of the instants both fits have (matching times to 1 ms)."""
    ta, tb = np.asarray(a["lead"]["t"], float), np.asarray(b["lead"]["t"], float)
    j = np.clip(np.searchsorted(tb, ta), 1, len(tb) - 1)
    j = np.where(np.abs(tb[j - 1] - ta) < np.abs(tb[j] - ta), j - 1, j)
    m = np.abs(tb[j] - ta) < 1e-3
    return np.flatnonzero(m), j[m]


def compare_arrays(Ja: np.ndarray, Jb: np.ndarray, cam_centre: np.ndarray, dt: float) -> dict:
    """Ja, Jb (N, 55, 3) joints of the same frames in two solutions; cam_centre (3,). Returns the numbers described in the module docstring."""
    ra = np.hypot(Ja[:, 0, 0] - cam_centre[0], Ja[:, 0, 2] - cam_centre[2])
    rb = np.hypot(Jb[:, 0, 0] - cam_centre[0], Jb[:, 0, 2] - cam_centre[2])
    rel_a, rel_b = Ja[:, :22] - Ja[:, :1], Jb[:, :22] - Jb[:, :1]
    mpjpe = np.linalg.norm(rel_a - rel_b, axis=-1).mean(1)
    path = lambda J: float(np.linalg.norm(np.diff(J[:, 0, [0, 2]], axis=0), axis=-1).sum())     # noqa: E731
    spd = lambda J: np.linalg.norm(np.diff(J[:, 0], axis=0), axis=-1) / dt                      # noqa: E731
    height = lambda J: J[:, 15, 1] - np.minimum(J[:, 10, 1], J[:, 11, 1])                       # head joint above the lower toe joint
    return {"range_ratio_median": round(float(np.median(rb / ra)), 4), "range_ratio_p5_p95": [round(float(np.percentile(rb / ra, 5)), 4), round(float(np.percentile(rb / ra, 95)), 4)],
            "range_m_base_median": round(float(np.median(ra)), 3), "range_m_variant_median": round(float(np.median(rb)), 3),
            "floor_path_m": {"base": round(path(Ja), 2), "variant": round(path(Jb), 2), "ratio": round(path(Jb) / max(path(Ja), 1e-9), 4)},
            "pose_relative_to_pelvis_mpjpe_mm": {"median": round(float(np.median(mpjpe)) * 1000, 1), "p95": round(float(np.percentile(mpjpe, 95)) * 1000, 1)},
            "pelvis_speed_ratio_median": round(float(np.median(spd(Jb)) / max(np.median(spd(Ja)), 1e-9)), 4),
            "pelvis_height_m": {"base": round(float(np.median(Ja[:, 0, 1])), 3), "variant": round(float(np.median(Jb[:, 0, 1])), 3)},
            "head_above_toe_m": {"base": round(float(np.median(height(Ja))), 3), "variant": round(float(np.median(height(Jb))), 3)},
            "pelvis_offset_m": {"median": round(float(np.median(np.linalg.norm(Ja[:, 0] - Jb[:, 0], axis=-1))), 3), "p95": round(float(np.percentile(np.linalg.norm(Ja[:, 0] - Jb[:, 0], axis=-1), 95)), 3)}}


def compare(base: str, src: str, var: str, var_src: str, cam_centre: np.ndarray, k0: int | None = None, k1: int | None = None) -> dict:
    A, B = _load(base, src), _load(var, var_src)
    ia, ib = align(A, B)
    if k0 is not None:
        t = np.asarray(A["lead"]["t"], float)[ia]
        keep = (t >= k0 / 30.0 - 1e-6) & (t <= k1 / 30.0 + 1e-6)
        ia, ib = ia[keep], ib[keep]
    dt = float(np.median(np.diff(np.asarray(A["lead"]["t"], float))))
    out = {"base": f"{base}/{src}", "variant": f"{var}/{var_src}", "frames": int(len(ia)), "per_role": {}}
    for r in ROLES:
        out["per_role"][r] = compare_arrays(np.asarray(A[r]["joints"], float)[ia], np.asarray(B[r]["joints"], float)[ib], cam_centre, dt)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_depth", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("compare",))
    ap.add_argument("--base", default="john_lula")
    ap.add_argument("--src", default="smplx_rf2_noslip")
    ap.add_argument("--var", required=True)
    ap.add_argument("--var-src", default="smplx")
    ap.add_argument("--k0", type=int)
    ap.add_argument("--k1", type=int)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cam = np.load(PACKAGE_ROOT / "work" / "john_lula" / "cameras" / "01.npz", allow_pickle=True)
    rep = compare(a.base, a.src, a.var, a.var_src, np.asarray(cam["centre_world"], float), a.k0, a.k1)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
