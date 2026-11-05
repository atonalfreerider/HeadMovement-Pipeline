#!/usr/bin/env python
"""LHM++ baseline score for 4DAnyone round 2 B (dancecap.fda_lhmpp; runs in the LHM++ venv with lpips).

Per B run: the LHM++ renders (RGBA) at 4DAnyone's view-0 camera (<run>_v0, the dancer alone) and at the real held-out
camera (<run>_real, both dancers) vs heldout/<run>.npz (the held-out camera's real crops of the dancer). The canvas, LPIPS
(alex) and colour-matched PSNR inside the real mask are fda_pod.py appearance's: v0 renders are cropped by their alpha box
(4DAnyone B: the YOLO person box), real-camera renders by the REAL mask's box (same camera, same pixels). The v0 frames are
those 4DAnyone B scored (its appearance.json rows, same < 20 deg angle filter), so the two rows compare like for like.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch

W, H = 256, 512


def letterbox(img: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    mx, my = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    x0, y0, x1, y1 = x0 - mx, y0 - my, x1 + mx, y1 + my
    s = min(W / (x1 - x0), H / (y1 - y0))
    A = np.array([[s, 0, W / 2 - s * (x0 + x1) / 2], [0, s, H / 2 - s * (y0 + y1) / 2]], np.float32)
    return cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))


def score(net, g: np.ndarray, real: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    g = g.copy()
    g[~m] = 0
    gm, rm = g[m].astype(float), real[m].astype(float)
    gcc = (gm - gm.mean(0)) / (gm.std(0) + 1e-6) * rm.std(0) + rm.mean(0)
    psnr = float(10 * np.log10(255 ** 2 / max(((gcc - rm) ** 2).mean(), 1e-6)))
    tg = lambda im: torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 127.5 - 1  # noqa: E731
    with torch.no_grad():
        return float(net(tg(g), tg(real)).item()), psnr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True)
    ap.add_argument("--max-deg", type=float, default=20.0)
    a = ap.parse_args()
    ft = Path(a.ft)
    ren = ft / "lhmpp" / "renders"
    vj = json.loads((ft / "lhmpp" / "input" / "views.json").read_text())
    net = lpips.LPIPS(net="alex", verbose=False).eval()
    (ft / "lhmpp" / "pairs").mkdir(parents=True, exist_ok=True)
    out = {}
    for g in vj["groups"]:
        if not g.endswith("_v0"):
            continue
        name = g[:-3]
        hz = np.load(ft / "heldout" / f"{name}.npz")
        apj = ft / "runs_m" / f"{name}_ours" / "appearance.json"
        b_rows = {r["frame"]: r for r in json.loads(apj.read_text())["rows"]} if apj.exists() else {}
        res = {"v0": [], "real": []}
        for f, br in sorted(b_rows.items()):
            hi = int(hz["j"][f])
            if not hz["ok"][hi]:
                continue
            real, m = hz["crops"][hi], hz["masks"][hi]
            p = ren / f"{name}_v0" / f"{f:05d}.png"
            if p.exists():
                rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                al = rgba[..., 3] > 127
                if al.sum() > 500:
                    ys, xs = np.nonzero(al)
                    lp, ps = score(net, letterbox(rgba[..., :3], (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)), real, m)
                    res["v0"].append({"frame": f, "angle": br["angle"], "b_found": br.get("found"), "lpips": lp, "psnr_cc": ps,
                                      "b_lpips": br.get("lpips"), "b_psnr_cc": br.get("psnr_cc")})
            p = ren / f"{name}_real" / f"{hi:05d}.png"
            if p.exists():
                rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                lb = letterbox(rgba[..., :3], hz["boxes"][hi])
                lp, ps = score(net, lb, real, m)
                res["real"].append({"frame": f, "lpips": lp, "psnr_cc": ps})
                if f in (0, 60):
                    cv2.imwrite(str(ft / "lhmpp" / "pairs" / f"{name}_{f:03d}.jpg"), np.concatenate([real, lb], 1))

        def med(rows, key, sel=lambda r: True):
            v = [r[key] for r in rows if sel(r) and r.get(key) is not None]
            return (float(np.median(v)) if v else None), len(v)
        both = lambda r: r["angle"] < a.max_deg and r.get("b_found")  # noqa: E731 - frames 4DAnyone B scored
        out[name] = {
            "lhmpp_v0": {"lpips": med(res["v0"], "lpips", both)[0], "psnr_cc": med(res["v0"], "psnr_cc", both)[0],
                         "n": med(res["v0"], "lpips", both)[1]},
            "fda_b_same_frames": {"lpips": med(res["v0"], "b_lpips", both)[0], "psnr_cc": med(res["v0"], "b_psnr_cc", both)[0]},
            "lhmpp_real_camera": {"lpips": med(res["real"], "lpips")[0], "psnr_cc": med(res["real"], "psnr_cc")[0],
                                  "n": len(res["real"])},
            "rows": res}
        print(name, {k: v for k, v in out[name].items() if k != "rows"}, flush=True)
    (ft / "lhmpp" / "score.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
