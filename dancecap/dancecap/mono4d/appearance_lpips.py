r"""LPIPS for every method of the appearance comparison (2026-10-19 review fix).

metrics.json had LPIPS only for the two LHM++ variants (scored on their pod). This scores all six methods - the current
avatar, AGS, both LHM++ variants and both hybrids - on the same 18 real views, inside the same real masks and crops, with
the same function (dancer_splats.evaluate._crop_scores, lpips 0.1.4, AlexNet), so the rows are comparable. lpips and its
AlexNet weights are not installed locally, so the scoring runs on a pod through the farm runner.

    python -m dancecap.mono4d.appearance_lpips pack      # local: small input folder (renders cropped to the dancers)
    python -m dancecap remote run --jobs remote/cand_lhmpp/pod_job_lpips.json ...   # pod: step "score"
    python -m dancecap.mono4d.appearance_lpips import    # local: lpips_all.json + lpips_all.md, LHM++ re-check

Inputs (pack): compare/lhmpp/input/real/<view>.jpg and masks/<view>_<01|02>.png (the files the LHM++ pod scored with;
metrics.json's merge check found the same PSNR to 0.0 dB locally) and compare/renders/<method>/<view>.png cropped to the
union of the two dancers' crop boxes, zero outside the two real masks (lossless PNG; LPIPS and PSNR never read those
pixels, SSIM does, so the pod's SSIM is not comparable). Likeness data: work/ only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT

TAKE = "larissa_kadu_38"
CMP = PACKAGE_ROOT / "work" / TAKE / "mono4d" / "compare"
SRC = CMP / "lhmpp" / "input"
OUT = CMP / "lpips_all"
INPUT = OUT / "input"
POD_OUT = OUT / "pod_out"
METHODS = ("avatar", "ags_8phone", "lhmpp_1phone", "lhmpp_8phone", "hybrid_mesh", "hybrid_mesh_shell")
ROLES = ("lead", "follow")
PID = {"lead": "01", "follow": "02"}
UNSEEN_CAM03 = ("i0_c03", "i2_c03", "i4_c03", "i5_c03")


def log(msg: str) -> None:
    print(f"[appearance_lpips] {msg}", flush=True)


def _masks(root: Path, vid: str) -> dict:
    out = {}
    for r in ROLES:
        m = cv2.imread(str(root / "masks" / f"{vid}_{PID[r]}.png"), cv2.IMREAD_GRAYSCALE)
        out[r] = None if m is None or (m > 127).sum() < 50 else (m > 127)
    return out


def pack() -> dict:
    from .cand_lhmpp import _crop_box
    views = [v["id"] for v in json.loads((CMP / "views.json").read_text())["real_views"]]
    if INPUT.exists():
        shutil.rmtree(INPUT)
    (INPUT / "real").mkdir(parents=True)
    (INPUT / "masks").mkdir(parents=True)
    man = {"views": {}, "methods": list(METHODS), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    for vid in views:
        shutil.copy2(SRC / "real" / f"{vid}.jpg", INPUT / "real" / f"{vid}.jpg")
        for r in ROLES:
            shutil.copy2(SRC / "masks" / f"{vid}_{PID[r]}.png", INPUT / "masks" / f"{vid}_{PID[r]}.png")
        gt = cv2.imread(str(SRC / "real" / f"{vid}.jpg"), cv2.IMREAD_COLOR)
        H, W = gt.shape[:2]
        ms = _masks(SRC, vid)
        boxes = {r: [int(x) for x in _crop_box(m)] for r, m in ms.items() if m is not None}
        x0 = min(b[0] for b in boxes.values()); y0 = min(b[1] for b in boxes.values())  # noqa: E702
        x1 = max(b[2] for b in boxes.values()); y1 = max(b[3] for b in boxes.values())  # noqa: E702
        man["views"][vid] = {"H": H, "W": W, "union_box": [x0, y0, x1, y1], "role_boxes": boxes, "renders": {}}
        # LPIPS / PSNR (raw and cc) read the render only inside a real mask (_lpips_score: where(sel, pred, gt);
        # masked MSE; the colour fit is fitted on sel): zero the rest to keep the upload small. SSIM is not exact.
        keep = np.zeros((H, W), bool)
        for m_ in ms.values():
            if m_ is not None:
                keep |= m_
        for m in METHODS:
            im = cv2.imread(str(CMP / "renders" / m / f"{vid}.png"), cv2.IMREAD_UNCHANGED)
            if im is None or im.ndim != 3 or im.shape[2] != 4 or im.shape[:2] != (H, W):
                man["views"][vid]["renders"][m] = None
                log(f"missing / wrong size: {m} {vid}")
                continue
            d = INPUT / "renders" / m
            d.mkdir(parents=True, exist_ok=True)
            im = np.where(keep[..., None], im, 0).astype(np.uint8)
            cv2.imwrite(str(d / f"{vid}.png"), im[y0:y1, x0:x1], [cv2.IMWRITE_PNG_COMPRESSION, 9])
            man["views"][vid]["renders"][m] = f"renders/{m}/{vid}.png"
    (INPUT / "pack.json").write_text(json.dumps(man, indent=1))
    size = sum(f.stat().st_size for f in INPUT.rglob("*") if f.is_file())
    log(f"packed {len(views)} views x {len(METHODS)} methods into {INPUT} ({size / 1e6:.1f} MB)")
    return man


def score() -> dict:
    """On the pod: _crop_scores with the lpips net for every (view, dancer, method)."""
    from ..dancer_splats.evaluate import _crop_scores, _lpips
    from .cand_lhmpp import _crop_box, _read_rgb
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _lpips(device)
    if net is None:
        raise RuntimeError("lpips unavailable")
    man = json.loads((INPUT / "pack.json").read_text())
    rows = []
    t0 = time.time()
    for vid, v in man["views"].items():
        gt = _read_rgb(INPUT / "real" / f"{vid}.jpg")
        ms = _masks(INPUT, vid)
        x0, y0, x1, y1 = v["union_box"]
        for m in METHODS:
            rel = v["renders"].get(m)
            if rel is None:
                continue
            crop = cv2.imread(str(INPUT / rel), cv2.IMREAD_UNCHANGED)
            im = np.zeros((v["H"], v["W"], 4), np.uint8)
            im[y0:y1, x0:x1] = crop
            rgb = im[:, :, 2::-1].astype(np.float32) / 255.0
            a = im[:, :, 3].astype(np.float32) / 255.0
            pred = rgb * a[..., None]
            for r in ROLES:
                mk = ms[r]
                if mk is None:
                    continue
                bx0, by0, bx1, by1 = _crop_box(mk)
                s = _crop_scores(pred[by0:by1, bx0:bx1], gt[by0:by1, bx0:bx1], mk[by0:by1, bx0:bx1], net, device)
                rows.append({"view": vid, "cam": vid.split("_c")[1], "role": r, "method": m,
                             "crop": [int(bx0), int(by0), int(bx1), int(by1)],
                             **{k: (None if s.get(k) is None else float(s[k])) for k in s}})
        log(f"{vid} done")
    from importlib.metadata import version
    out = {"how": "dancer_splats.evaluate._crop_scores(pred crop, real crop, real mask crop, lpips alex net); crop = "
                  "cand_lhmpp._crop_box(real mask) (bbox + 10 %); pred = render RGB x alpha (over black)",
           "lpips_version": version("lpips"), "torch": torch.__version__, "device": device,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "seconds": round(time.time() - t0, 1),
           "rows": rows}
    POD_OUT.mkdir(parents=True, exist_ok=True)
    (POD_OUT / "lpips_all.json").write_text(json.dumps(out, indent=1))
    log(f"{len(rows)} rows in {time.time() - t0:.0f} s")
    return out


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 4) if xs else None


def import_results() -> dict:
    """Local: summarise the pod rows (medians over dancer-views, like metrics.md) and re-check them against metrics.json
    (PSNR per row must match; LHM++ LPIPS must match the first pod's rows)."""
    pod = json.loads((POD_OUT / "lpips_all.json").read_text())
    mt = json.loads((CMP / "metrics.json").read_text())
    loc = {(r["view"], r["role"], r["method"]): r for r in mt["rows"] if not r.get("missing")}
    status = {(r["view"], r["role"], r["method"]): r.get("status") for r in mt["rows"] if not r.get("missing")}
    d_psnr, d_lp, d_lpcc, crop_mismatch = [], [], [], 0
    for r in pod["rows"]:
        q = loc.get((r["view"], r["role"], r["method"]))
        if q is None:
            continue
        if q["crop"] != r["crop"]:
            crop_mismatch += 1
        d_psnr.append(abs(q["psnr"] - r["psnr"]))
        if q.get("lpips") is not None:
            d_lp.append(abs(q["lpips"] - r["lpips"]))
            d_lpcc.append(abs(q["lpips_cc"] - r["lpips_cc"]))
    check = {"rows": len(pod["rows"]), "matched_rows": len(d_psnr), "crop_mismatch": crop_mismatch,
             "psnr_abs_diff_max": round(max(d_psnr), 5) if d_psnr else None,
             "lhmpp_lpips_rows_rechecked": len(d_lp),
             "lhmpp_lpips_abs_diff_max": round(max(d_lp), 5) if d_lp else None,
             "lhmpp_lpips_cc_abs_diff_max": round(max(d_lpcc), 5) if d_lpcc else None}
    sets = {"all_18_views": lambda r: True,
            "cam03_unseen_frames": lambda r: r["view"] in UNSEEN_CAM03,
            "cam03": lambda r: r["cam"] == "03",
            "cam02_06": lambda r: r["cam"] in ("02", "06"),
            "heldout_for_this_method": lambda r: status.get((r["view"], r["role"], r["method"])) == 0}
    summ = {}
    for m in METHODS:
        summ[m] = {}
        for sname, f in sets.items():
            rr = [r for r in pod["rows"] if r["method"] == m and f(r)]
            summ[m][sname] = {"n": len(rr), **{k: _med([r.get(k) for r in rr])
                                               for k in ("lpips", "lpips_cc", "psnr", "psnr_cc")}}
    # per-view wins: lowest LPIPS cc per (view, dancer)
    wins = {m: 0 for m in METHODS}
    keys = sorted({(r["view"], r["role"]) for r in pod["rows"]})
    for k in keys:
        rr = [r for r in pod["rows"] if (r["view"], r["role"]) == k and r.get("lpips_cc") is not None]
        if rr:
            wins[min(rr, key=lambda r: r["lpips_cc"])["method"]] += 1
    out = {"schema": "dancecap.mono4d.appearance_lpips/1", "source": "lpips_all/pod_out/lpips_all.json",
           "pod": {k: pod.get(k) for k in ("lpips_version", "torch", "device", "gpu", "seconds")},
           "how": pod["how"] + "; summary = median over dancer-views (lower LPIPS = closer to the photo)",
           "check_vs_metrics_json": check, "summary_median": summ, "lowest_lpips_cc_per_dancer_view": wins,
           "n_dancer_views": len(keys)}
    (CMP / "lpips_all.json").write_text(json.dumps(out, indent=1))
    lines = ["| view set | method | dancer-views | LPIPS | LPIPS cc | PSNR | PSNR cc |", "|---|---|---|---|---|---|---|"]
    f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    f2 = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    for sname in sets:
        for m in METHODS:
            s = summ[m][sname]
            lines.append(f"| {sname} | {m} | {s['n']} | {f3(s['lpips'])} | {f3(s['lpips_cc'])} | {f2(s['psnr'])} | "
                         f"{f2(s['psnr_cc'])} |")
    lines.append("")
    lines.append(f"Lowest LPIPS cc per dancer-view ({len(keys)}): " + ", ".join(f"{m} {n}" for m, n in wins.items()))
    lines.append(f"Check against metrics.json: {json.dumps(check)}")
    (CMP / "lpips_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(json.dumps(check))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="appearance_lpips")
    ap.add_argument("step", choices=("pack", "score", "import"))
    a = ap.parse_args(argv)
    {"pack": pack, "score": score, "import": import_results}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
