r"""john_lula: repair of the monocular DEPTH SPIKES of a fit (agent J3b, 2026-10-11). Pure numpy on contract folders (smplx/, smplx_rf2_noslip/).

    python -m dancecap.jl_repair <folder> [<folder> ...] [--dev 0.25] [--out-suffix ""]

With ONE camera a frame can match every 2D landmark and still sit a metre or two further along the optical axis: the picture does not forbid it. The first fit of john_lula had one such
run (John, 37.60-37.67 s: the pelvis jumps 1.5 m in depth for three frames and comes back; the 2D reprojection of those frames is 13-18 px against 8-12 px around them). The motion
filter of the refinement spreads it over six frames (up to 17 m/s at the ankles). This module finds the frames whose pelvis deviates from the robust (9-frame median) path of the
dancer by more than `dev` metres, widens the run by one frame on each side, and replaces the pelvis by a straight line between the last good frames. The whole body is moved by the
same offset (transl and every joint: the pose relative to the pelvis is untouched), the frames are flagged (`repaired`, and the new mask `depth_spike_repaired`). Frames that were not
detected are bit-identical. The motion in the run is therefore an interpolation, not a measurement: flagged, never hidden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

ROLES = ("lead", "follow")


def find_spikes(P: np.ndarray, dev: float = 0.25, win: int = 9, widen: int = 1) -> np.ndarray:
    """P (T, 3) pelvis positions. Frames whose distance to the 9-frame median path exceeds dev (m), widened by `widen` frames on both sides. Returns (T,) bool."""
    m = np.stack([median_filter(P[:, k], size=win, mode="nearest") for k in range(3)], 1)
    bad = np.linalg.norm(P - m, axis=1) > dev
    if widen and bad.any():
        k = np.ones(2 * widen + 1)
        bad = np.convolve(bad.astype(float), k, mode="same") > 0
    return bad


def repair_pelvis(P: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Offsets (T, 3) that move the bad frames' pelvis onto the straight line between the nearest good frames (zero elsewhere)."""
    off = np.zeros_like(P)
    if not bad.any() or bad.all():
        return off
    idx = np.arange(len(P))
    good = ~bad
    for k in range(3):
        off[bad, k] = np.interp(idx[bad], idx[good], P[good, k]) - P[bad, k]
    return off


def repair_folder(folder: Path, dev: float = 0.25) -> dict:
    rep = {}
    for role in ROLES:
        p = folder / f"{role}.npz"
        z = dict(np.load(p, allow_pickle=True))
        P = np.asarray(z["joints"], float)[:, 0]
        bad = find_spikes(P, dev)
        if not bad.any():
            rep[role] = {"frames": 0}
            continue
        off = repair_pelvis(P, bad)
        z["joints"] = np.asarray(z["joints"], float) + off[:, None, :]
        z["transl"] = np.asarray(z["transl"], float) + off
        z["repaired"] = np.asarray(z["repaired"], bool) | bad
        z["depth_spike_repaired"] = bad
        np.savez_compressed(p, **z)
        t = np.asarray(z["t"], float)
        runs, a = [], None
        for i in range(len(bad) + 1):
            on = bool(bad[i]) if i < len(bad) else False
            if on and a is None:
                a = i
            elif not on and a is not None:
                runs.append([round(float(t[a]), 3), round(float(t[i - 1]), 3), round(float(np.abs(off[a:i]).max()), 2)])
                a = None
        rep[role] = {"frames": int(bad.sum()), "runs_t0_t1_maxoffset_m": runs}
    (folder / "depth_spike_repair.json").write_text(json.dumps({"dev_m": dev, "what": __doc__.split("\n")[2], "roles": rep}, indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_repair", description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--dev", type=float, default=0.25)
    a = ap.parse_args(argv)
    for f in a.folders:
        print(f, json.dumps(repair_folder(Path(f), a.dev)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
