"""Review sheets of the "sides" step (DESIGN 4.1 `look.py`): raw phone crops (undistorted, around the couple) with the skeletons drawn on.

    white = the input (detected) fit, magenta = v4.3 (the installed motion, when given), azure = v44a (an earlier sides candidate, when
    given), green = sides, orange = the partner's arm chain (sides), cyan = the triangulated joints (>= 2 views), yellow = the phone's own
    COCO detections

    window_sheet(take, ids)         the frame of each window's largest change, the two phones that see the moved joints best
    worst12 / random6               the 12 adopted windows that changed the most / 6 random adopted windows
    owner_times_sheet(take, times)  the owner's moments (film times in the label)

-> work/<take>/review/sides/*.jpg (local only: the phone videos never leave the PC).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geom
from .config import load_cfg

BONES = ((0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (9, 13), (13, 16),
         (16, 18), (18, 20), (9, 14), (14, 17), (17, 19), (19, 21))
ARM_BONES = {"R": ((14, 17), (17, 19), (19, 21)), "L": ((13, 16), (16, 18), (18, 20))}
COCO_BONES = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16))
COL = {"input": (255, 255, 255), "v43": (255, 0, 255), "v44a": (255, 150, 0), "sides": (60, 220, 60), "partner": (0, 140, 255), "tri": (255, 255, 0),
       "coco": (0, 230, 255)}


def _proj(Kc, R, t, X):
    from .crops import project
    return project(Kc, R, t, np.asarray(X, float))


def draw_skeleton(img, Kc, R, t, J, color, bones=BONES, th=2, pts: bool = False):
    import cv2
    uv, d = _proj(Kc, R, t, J)
    for a, b in bones:
        if d[a] > 0.1 and d[b] > 0.1 and np.isfinite(uv[[a, b]]).all():
            cv2.line(img, tuple(int(x) for x in uv[a]), tuple(int(x) for x in uv[b]), color, th, cv2.LINE_AA)
    if pts:
        for i in range(len(J)):
            if d[i] > 0.1 and np.isfinite(uv[i]).all():
                cv2.circle(img, tuple(int(x) for x in uv[i]), 4, color, -1, cv2.LINE_AA)


def tile(scene, cam: str, k: int, fits: dict, tri: dict | None, views: dict | None, lines: list, moved: dict | None = None,
         cfg=None):
    """one labelled crop of phone `cam` at grid frame k with the skeletons -> BGR tile (None when the phone cannot show it)"""
    import cv2
    from .crops import CropCfg, crop, crop_K, label_tile
    cfg = cfg or CropCfg()
    img, meta = crop(scene, cam, k / 30.0, "couple", cfg)
    if img is None:
        return None
    cx, cy, side = meta["box"]
    c = scene.camera(cam, k)
    Kc = crop_K(c["K"], cx, cy, side, cfg.tile)
    R, t = c["R"], c["t"]
    img = img.copy()
    # COCO detections (yellow) and the triangulated joints (cyan) of both dancers
    if views is not None and cam in views:
        z = views[cam]
        sc = cfg.tile / side
        for p in (0, 1):
            uv = z["uv"][k, p].astype(float)
            cf = z["conf"][k, p].astype(float)
            q = (uv - np.array([cx - side / 2, cy - side / 2])) * sc        # undistorted px -> crop px (the crop is a pinhole resample)
            for a, b in COCO_BONES:
                if cf[a] >= 0.5 and cf[b] >= 0.5 and np.isfinite(q[[a, b]]).all():
                    cv2.line(img, tuple(int(x) for x in q[a]), tuple(int(x) for x in q[b]), COL["coco"], 1, cv2.LINE_AA)
    if tri is not None:
        for r, tr in tri.items():
            X = np.where((tr["nv"][k] >= 2)[:, None], tr["Xd"][k].astype(float), np.nan)
            uv, d = _proj(Kc, R, t, np.nan_to_num(X))
            for i in range(5, 17):
                if np.isfinite(X[i]).all() and d[i] > 0.1:
                    cv2.circle(img, tuple(int(x) for x in uv[i]), 4, COL["tri"], 1, cv2.LINE_AA)
    for name in ("v43", "v44a", "input", "sides"):
        f = fits.get(name)
        if f is None:
            continue
        for r in ("lead", "follow"):
            J = np.asarray(f[r]["joints"], float)[k]
            draw_skeleton(img, Kc, R, t, J, COL[name], th=1 if name != "sides" else 2)
    if "sides" in fits:
        # the partner's arm chain of the moved arm (orange): her left arm against his right
        J = np.asarray(fits["sides"]["follow"]["joints"], float)[k]
        draw_skeleton(img, Kc, R, t, J, COL["partner"], ARM_BONES["L"], th=2)
    return label_tile(img, lines + [f"phone {cam}"])


def best_cams(scene, views: dict | None, k: int, role: str = "lead", joints=(19, 21), n: int = 2) -> list:
    """the phones that see the couple at frame k and predict the moved joints visible, the most visible first"""
    from .crops import CropCfg
    cfg = CropCfg()
    scores = []
    for c in scene.cams:
        ok, share = scene.sees(c, k, "couple", cfg)
        if not ok:
            continue
        v = 0.0
        if views is not None and c in views:
            p = 0 if role == "lead" else 1
            coco = [geom.C2S.index(j) for j in joints if j in geom.C2S]
            v = float(np.asarray(views[c]["vis_pred"][k, p][coco], float).mean()) if coco else 0.0
        scores.append((v + 0.1 * share, c))
    return [c for _, c in sorted(scores, reverse=True)[:n]]


def grid(tiles: list, cols: int):
    from .crops import grid as g
    tiles = [t for t in tiles if t is not None]
    return g(tiles, cols) if tiles else None


def _load(take: str, compare: dict | None):
    from .adapters import adapter_for
    from .evidence import load_tri, load_views
    from .prepare import load_fit
    from .crops import Scene
    cfg = load_cfg(take)
    ad = adapter_for(take, cfg)
    scene = Scene(take, ad.work)
    fits = {"input": {r: load_fit(ad.base_dir / f"{r}.npz") for r in ("lead", "follow")}}
    sdir = ad.work / "smplx_sides"
    if (sdir / "lead.npz").exists():
        fits["sides"] = {r: load_fit(sdir / f"{r}.npz") for r in ("lead", "follow")}
    for nm, d in (compare or {}).items():
        fits[nm] = {r: load_fit(Path(d) / f"{r}.npz") for r in ("lead", "follow")}
    tri = {r: load_tri(ad.work, r) for r in ("lead", "follow")}
    views = load_views(ad.work, ad.cams)
    return cfg, ad, scene, fits, tri, views


def change_frame(fits: dict, w: dict) -> tuple[int, str, float]:
    """the window frame (and dancer) with the largest joint displacement of sides against the input"""
    best = (w["k0"], "lead", 0.0)
    for r in ("lead", "follow"):
        a = np.asarray(fits["input"][r]["joints"], float)[w["r0"]:w["r1"] + 1, :22]
        b = np.asarray(fits["sides"][r]["joints"], float)[w["r0"]:w["r1"] + 1, :22]
        d = np.linalg.norm(a - b, axis=-1).max(1)
        i = int(np.argmax(d))
        if d[i] > best[2]:
            best = (w["r0"] + i, r, float(d[i]))
    return best


def sheets(take: str, compare: dict | None = None, out_dir: Path | None = None, owner_times: list | None = None, seed: int = 0) -> dict:
    import cv2
    from .filmtime import FilmMap, mmss
    cfg, ad, scene, fits, tri, views = _load(take, compare)
    od = Path(out_dir or ad.work / "review" / "sides")
    od.mkdir(parents=True, exist_ok=True)
    fm = FilmMap.for_work(ad.work)
    gj = json.loads((ad.work / "sides" / "gate.json").read_text(encoding="utf-8")) if (ad.work / "sides" / "gate.json").exists() else {"windows": {}}
    plan = json.loads((ad.work / "sides" / "plan.json").read_text(encoding="utf-8"))
    wins = {w["id"]: w for w in plan["windows"]}
    adopted = [int(i) for i, s in gj["windows"].items() if s.get("status") == "adopt"]
    out = {}
    if "sides" in fits and adopted:
        ch = sorted(((change_frame(fits, wins[i]), i) for i in adopted), key=lambda x: -x[0][2])
        rng = np.random.default_rng(seed)
        picks = {"worst12": [i for _, i in ch[:12]], "random6": sorted(rng.choice(adopted, min(6, len(adopted)), replace=False).tolist())}
        for name, ids in picks.items():
            tiles = []
            for i in ids:
                (k, role, dmax) = change_frame(fits, wins[i])
                w = wins[i]
                js = w["dancers"].get(role, {}).get("joints", [19, 21])
                for c in best_cams(scene, views, k, role, [j for j in (18, 19, 20, 21, 4, 5, 7, 8) if j in set(js) or j - 2 in set(js)] or [19, 21]):
                    lines = [f"win {i} {w['film0']}-{w['film1']} ({', '.join(w['reasons'])})",
                             f"film {mmss(fm.main(k / 30.0))}  dance {k / 30.0:.2f} s  {role} moved {dmax * 100:.1f} cm"]
                    tiles.append(tile(scene, c, k, fits, tri, views, lines))
            p = od / f"{name}.jpg"
            g = grid(tiles, 4)
            if g is not None:
                cv2.imwrite(str(p), g, [cv2.IMWRITE_JPEG_QUALITY, 88])
                out[name] = str(p)
    times = list(owner_times or [])
    for part in range(0, len(times), 6):
        tiles = []
        for t in times[part:part + 6]:
            k = int(round(t * 30))
            for c in best_cams(scene, views, k, "lead", (19, 21)):
                gi = [i for i, w in wins.items() if w["r0"] <= k <= w["r1"]]
                st = ",".join(f"{i}:{gj['windows'].get(str(i), {}).get('status', '?')}" for i in gi) or "no window"
                tiles.append(tile(scene, c, k, fits, tri, views, [f"film {mmss(fm.main(t))}  dance {t:.2f} s", f"windows {st}"]))
        p = od / f"owner_times_{part // 6 + 1}.jpg"
        g = grid(tiles, 4)
        if g is not None:
            cv2.imwrite(str(p), g, [cv2.IMWRITE_JPEG_QUALITY, 88])
            out[f"owner_times_{part // 6 + 1}"] = str(p)
    return out
