"""How much of the 9:16 frame the couple fills in the recap film's overhead shots (the owner's rule 2026-10-10: close, 55-70 % of the width at the
tightest, never below 45 %, a pivot or an exit never cropped).

    python -m dancecap.recap_overhead_check [--direction PATH] [--step 0.1]

The viewer's `FilmCamera.OverheadFillDistance` is replayed in Python on the teachers' triangulated joints (people/track3d.npz, 17 COCO joints per
teacher in the world frame): for every overhead segment, the height the shot asks for at each time (the weighted envelope over -0.8 .. +0.8 s, never
less than what the instant and 0.3 s ahead need, never further than `min_fill` asks), and from it the share of the frame width the dancers' joint box plus 0.22 m (a body's radius) fills
and the box's widest extent against the frame (1.0 = touching the edge). Written to film/overhead_check.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

WORK = PACKAGE_ROOT / "work" / "larissa_kadu_recap"
FOV_DEG = 24.0
ASPECT = 9.0 / 16.0
BAND_Y = 0.5
NEVER = 0.9
MARGIN_M = 0.22


def extent_series(track_npz: Path, hz: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """(lo (T, 2), hi (T, 2)): the floor-plan box (x, z) of both teachers' joints per 30 Hz frame"""
    x = np.load(track_npz, allow_pickle=True)["X"].astype(float)          # (T, 2, 17, 3)
    flat = x[..., [0, 2]].reshape(x.shape[0], -1, 2)
    ok = np.isfinite(flat).all(axis=2)
    lo = np.where(ok[..., None], flat, np.inf).min(axis=1)
    hi = np.where(ok[..., None], flat, -np.inf).max(axis=1)
    bad = ~ok.any(axis=1)
    lo[bad] = np.nan
    hi[bad] = np.nan
    return lo, hi


def need(lo: np.ndarray, hi: np.ndarray, frac_w: float, frac_h: float, tx: float, ty: float) -> float:
    w, h = hi[0] - lo[0], hi[1] - lo[1]
    return max((w + 2 * MARGIN_M) / (2 * tx * frac_w), (h + 2 * MARGIN_M) / (2 * ty * BAND_Y * frac_h))


def distance_at(lo_s: np.ndarray, hi_s: np.ndarray, t: float, fill: float, hz: float = 30.0, tx: float | None = None, ty: float | None = None,
                min_fill: float = 0.46) -> float:
    """OverheadFillDistance (FilmCamera.cs) at dance time t"""
    ty = ty if ty is not None else float(np.tan(np.radians(FOV_DEG / 2)))
    tx = tx if tx is not None else ty * ASPECT
    n = len(lo_s)

    def at(tt: float):
        i = int(round(min(max(tt, 0.0), (n - 1) / hz) * hz))
        return lo_s[i], hi_s[i]

    num = den = 0.0
    for k in range(-4, 5):
        l, h = at(t + k * 0.2)
        if np.isnan(l).any():
            continue
        wgt = 1.0 - abs(k) / 5.0
        num += wgt * need(l, h, fill, NEVER, tx, ty)
        den += wgt
    env = num / den if den else 8.0
    inst = 0.0
    for dt in (0.0, 0.3):
        l, h = at(t + dt)
        if not np.isnan(l).any():
            inst = max(inst, need(l, h, NEVER, NEVER, tx, ty))
    l0, h0 = at(t)
    far = need(l0, h0, min_fill, min_fill, tx, ty) if not np.isnan(l0).any() else env
    return max(min(env, far), inst)


def check(direction: dict, track_npz: Path, step: float = 0.1) -> dict:
    lo, hi = extent_series(track_npz)
    ty = float(np.tan(np.radians(FOV_DEG / 2)))
    tx = ty * ASPECT
    rows = []
    allfill: list[float] = []
    cropped = 0
    total = 0
    for s in direction["segments"]:
        cam = s["camera"]
        if cam.get("mode") != "overhead_floorcraft" or s.get("replay") and False:
            continue
        fill = float(cam.get("fill", 0.0))
        if fill <= 0:
            continue
        d0, d1 = s["dance"]
        speed = s.get("speed", 1.0)
        fills, widest = [], []
        for t in np.arange(d0, d1, step * (speed if isinstance(speed, (int, float)) else 1.0)):
            i = int(round(t * 30.0))
            if i >= len(lo) or np.isnan(lo[i]).any():
                continue
            dist = max(3.0, distance_at(lo, hi, float(t), fill))
            width = dist * 2 * tx
            fw = (hi[i][0] - lo[i][0] + 2 * MARGIN_M) / width                  # the joints' box plus a body's radius against the frame width
            fh = (hi[i][1] - lo[i][1] + 2 * MARGIN_M) / (dist * 2 * ty * BAND_Y)
            fills.append(fw)
            widest.append(max(fw, fh))
            total += 1
            cropped += int(max(fw, fh) > 1.0)
        if fills:
            rows.append({"segment": s["id"], "dance": [round(d0, 1), round(d1, 1)], "fill_width_median": round(float(np.median(fills)), 3),
                         "fill_width_p90": round(float(np.percentile(fills, 90)), 3), "fill_width_min": round(float(np.min(fills)), 3),
                         "widest_extent_max": round(float(np.max(widest)), 3)})
            allfill += fills
    return {"fov_deg": FOV_DEG, "aspect": ASPECT, "target_fill_incl_margin": [c.get("camera", {}).get("fill") for c in []] or None,
            "segments": rows, "frames": total, "frames_cropped": cropped,
            "fill_width": {"median": round(float(np.median(allfill)), 3) if allfill else None, "p10": round(float(np.percentile(allfill, 10)), 3) if allfill else None,
                           "p90": round(float(np.percentile(allfill, 90)), 3) if allfill else None, "min": round(float(np.min(allfill)), 3) if allfill else None}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--direction", default=str(WORK / "film" / "direction.json"))
    ap.add_argument("--step", type=float, default=0.1)
    a = ap.parse_args(argv)
    doc = json.loads(Path(a.direction).read_text(encoding="utf-8"))
    rep = check(doc, WORK / "people" / "track3d.npz", a.step)
    (WORK / "film" / "overhead_check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "segments"}))
    for r in rep["segments"]:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
