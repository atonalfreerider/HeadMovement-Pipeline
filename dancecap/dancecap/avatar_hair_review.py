"""Before/after review of the hair-aware texture (avatar_hair.py): held-out photometric error, hair-coloured texel
fractions and review sheets. Read-only on the bakes it compares; writes images only into the review dir.

    python -m dancecap.avatar_hair compare larissa_kadu --before work/larissa_kadu/avatar_hairdev/before \
        --after work/larissa_kadu/avatar_hairdev/after --review work/larissa_kadu_38/review

Held-out photos: frames the bakes never sampled (between their step slots, or every frame of --heldout-cam when both
bakes left that camera out). Error = mean |render - photo| (sRGB 8-bit) over the follow's visible BODY pixels (her
full mask minus her hair mask dilated 4 px, eroded 3 px, not the lead, rendered as the follow); "gain" columns refit
one linear RGB gain per view first (the albedo is exposure-normalised, each photo is not). false_hair = share of those
pixels where the render is hair-coloured (avatar_hair.red_cast) and the photo is not. Reported raw AND gain-fitted,
over all held-out views and per camera, for the body, the upper body (chest / shoulders / back / neck) and her face
(face pixels not under her hair mask: a face metric the bake does not optimise directly).

The primary test is leave-one-camera-out (--heldout-cam): the frames-between-step-slots set sits one frame (~33 ms)
from baked frames, so it is a weak test. Gate (summary.gate, leave-one-camera-out only): no upper-body regression on
the held-out camera, raw and gain-fitted (after <= before + 0.1 sRGB levels).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .avatar_hair import lab_of, red_cast, texel_regions, texture_metrics
from .avatar_texture import (PIDS, ROLES, _Geo, _read_mask, camera_views, linear_to_srgb, load_bodies, render,
                             srgb_to_linear)
from .avatar_uv import build_atlas
from .mamma_dyn import SMPLX_MODELS
from .review import _label
from .take import PACKAGE_ROOT

LABEL = {"lead": "LEAD (Kadu)", "follow": "FOLLOW (Larissa)"}


def _load_dir(d: Path) -> dict:
    rep = json.loads((d / "report.json").read_text())
    tex = {}
    for r in ROLES:
        z = np.load(d / f"{r}_texels.npz")
        tex[r] = {"srgb": np.asarray(z["rgb_srgb"], np.float32),
                  "hair": (np.asarray(z["hair_region"]) > 127) if "hair_region" in z.files else None}
    return {"dir": d, "report": rep, "tex": tex, "size": int(rep["config"]["size"])}


def heldout_views(work: Path, bake: dict, per_cam: int = 3, heldout_cam: str | None = None) -> list[dict]:
    """Frames inside the bake window that the bake did not sample, per camera `per_cam` spread over the window,
    each the held-out frame with the most visible follow body (largest body mask) in its time slice."""
    rep = bake["report"]
    t_lo, t_hi = rep["t_window"]
    cams = list(rep["cams"]) + ([heldout_cam] if heldout_cam and heldout_cam not in rep["cams"] else [])
    used, _ = camera_views(work, rep["cams"], t_lo, t_hi, float(rep["config"]["step_s"]))
    used_set = {(v["cam"], v["i"]) for v in used}
    allv, _ = camera_views(work, cams, t_lo, t_hi, 1e-4)  # every frame
    out = []
    for c in cams:
        cand = [v for v in allv if v["cam"] == c and (v["cam"], v["i"]) not in used_set and v["ok"]]
        if not cand:
            continue
        edges = np.linspace(t_lo, t_hi, per_cam + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            sl = [v for v in cand if a <= v["t"] < b and v["swap"] < 0.2]
            best, area = None, 0
            for v in sl[::2]:
                dd = work / "dancers" / c
                m = cv2.imread(str(dd / "body" / f"mask_{v['i']:04d}_02.png"), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    m = cv2.imread(str(dd / "masks" / f"mask_{v['i']:04d}_02.png"), cv2.IMREAD_GRAYSCALE)
                ar = int((m > 127).sum()) if m is not None else 0
                if ar > area:
                    best, area = v, ar
            if best is not None:
                out.append(best)
    return out


def _eval_masks(work: Path, v: dict, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    dd = work / "dancers" / v["cam"]
    full = _read_mask(dd / "masks" / f"mask_{v['i']:04d}_{PIDS['follow']:02d}.png", (H, W)) > 0
    lead = _read_mask(dd / "masks" / f"mask_{v['i']:04d}_{PIDS['lead']:02d}.png", (H, W)) > 0
    hp = dd / "hair" / f"mask_{v['i']:04d}_{PIDS['follow']:02d}.png"
    hair = _read_mask(hp, (H, W)) > 0 if hp.exists() else np.zeros((H, W), bool)
    hair_d = cv2.dilate(hair.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    body = full & ~hair_d & ~lead
    body = cv2.erode(body.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return body, hair


def _gain_fit(ren: np.ndarray, img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Per-channel linear gain photo ~ g x render over m (linear light), clamped to [0.5, 2]."""
    a = srgb_to_linear(ren[m])
    b = srgb_to_linear(img[m])
    g = (a * b).sum(0) / np.maximum((a * a).sum(0), 1e-9)
    return np.clip(g, 0.5, 2.0)


def compare(take: str, before: Path, after: Path, review: Path, extra: dict | None = None,
            heldout_cam: str | None = None, per_cam: int = 3, device: str | None = None, tag: str = "",
            lock: bool = True, label: str | None = None, sheets: bool = True) -> dict:
    import contextlib

    import torch

    from . import gpu
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with gpu.exclusive("texture-hair-review") if (lock and device == "cuda") else contextlib.nullcontext():
        return _compare(take, before, after, review, extra, heldout_cam, per_cam, device, tag, label or take, sheets)


def _compare(take: str, before: Path, after: Path, review: Path, extra: dict | None, heldout_cam: str | None,
             per_cam: int, device: str, tag: str, label: str, sheets: bool) -> dict:
    import torch
    work = PACKAGE_ROOT / "work" / take
    review.mkdir(parents=True, exist_ok=True)
    B, A = _load_dir(before), _load_dir(after)
    if B["size"] != A["size"]:
        raise ValueError("before/after atlas sizes differ")
    atlas = build_atlas(SMPLX_MODELS, A["size"])
    regions = texel_regions(atlas, SMPLX_MODELS)
    geo = _Geo(atlas, device)
    bodies, _ = load_bodies(work, out=after)
    cols = {k: {r: torch.as_tensor(d["tex"][r]["srgb"], device=device) for r in ROLES} for k, d in (("before", B), ("after", A))}
    region_col = torch.as_tensor(np.stack([regions["upper"], regions["face"], regions["head"]], 1).astype(np.float32),
                                 device=device)
    views = heldout_views(work, A, per_cam, heldout_cam)
    rows, per_view = [], []
    for v in views:
        img = cv2.imread(str(work / "frames" / v["cam"] / f"{v['i']:06d}.jpg"))[:, :, ::-1]
        H, W = img.shape[:2]
        posed = {r: b.at(v["t"]) for r, b in bodies.items()}
        if any(p is None for p in posed.values()):
            continue
        posed = {r: p[0] for r, p in posed.items()}
        body, hair = _eval_masks(work, v, H, W)
        rr, rid = render(torch, geo, posed, {"lead": cols["after"]["lead"], "follow": region_col}, v["K"], v["R"],
                         v["t_w"], v["k1"], H, W, device)
        own = rid == ROLES.index("follow")
        m = body & own
        if m.sum() < 300:
            continue
        upper = m & (rr[..., 0] > 0.5)
        facem = m & (rr[..., 1] > 0.5)
        rec = {"cam": v["cam"], "i": v["i"], "t": round(v["t"], 3), "px": int(m.sum()), "upper_px": int(upper.sum()),
               "face_px": int(facem.sum())}
        imgf = img.astype(np.float32) / 255.0
        photo_rc = red_cast(lab_of(imgf[m]))
        tiles = []
        for k in ("before", "after"):
            ren, rid2 = render(torch, geo, posed, cols[k], v["K"], v["R"], v["t_w"], v["k1"], H, W, device)
            ren = np.clip(ren, 0, 1)
            g = _gain_fit(ren, imgf, m)
            reng = linear_to_srgb(srgb_to_linear(ren) * g)
            err = np.abs(ren - imgf).mean(-1) * 255
            errg = np.abs(reng - imgf).mean(-1) * 255
            rc = red_cast(lab_of(ren[m]))
            rec[k] = {"mae": round(float(err[m].mean()), 2), "mae_gain": round(float(errg[m].mean()), 2),
                      "mae_upper": round(float(err[upper].mean()), 2) if upper.sum() > 50 else None,
                      "mae_gain_upper": round(float(errg[upper].mean()), 2) if upper.sum() > 50 else None,
                      "mae_face": round(float(err[facem].mean()), 2) if facem.sum() > 50 else None,
                      "mae_gain_face": round(float(errg[facem].mean()), 2) if facem.sum() > 50 else None,
                      "false_hair_pct": round(100 * float((rc & ~photo_rc).mean()), 2)}
            bg = np.full_like(img, 40)
            anyb = rid2 >= 0
            bg[anyb] = (ren[anyb] * 255).astype(np.uint8)
            tiles.append((k, bg))
        per_view.append(rec)
        rows.append((v, img, tiles, own | (cv2.dilate(hair.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0), rec))
    summary = _summarise(per_view)
    summary["per_cam"] = {c: _summarise([p for p in per_view if p["cam"] == c]) for c in sorted({p["cam"] for p in per_view})}
    summary["worse_views"] = _worse_counts(per_view)
    if heldout_cam:
        summary["gate"] = _gate(summary["per_cam"].get(heldout_cam), heldout_cam)
    summary["registration"] = _flow_compare(B["report"], A["report"])
    summary["texture_metrics"] = {"before": _tm(B, regions), "after": _tm(A, regions)}
    for lab, d in (extra or {}).items():
        summary["texture_metrics"][lab] = _tm(_load_dir(Path(d)), regions)
    summary["hair_region"] = A["report"].get("hair_region")
    summary["hair_sample"] = A["report"]["roles"]["follow"].get("hair_sample")
    summary["hair_masks"] = A["report"].get("hair_masks")
    paths = []
    if sheets:
        paths.append(_before_after_sheet(rows, review / f"avatar_texture_follow_before_after{tag}.jpg", label,
                                         summary, heldout_cam))
    if sheets and not heldout_cam:
        paths.append(_uv_sheet(B, A, atlas, review / "avatar_uv_before_after.jpg", label))
        paths.append(_turntables(torch, geo, bodies, cols, review / "avatar_turntable_before_after.jpg", label, device))
        paths.append(_faces(torch, geo, work, bodies, cols, A, review / "avatar_face_before_after.jpg", label, device))
        paths.append(_head_region(torch, geo, bodies, A, B, review / "avatar_hair_region_head.jpg", label, device))
    return {"summary": summary, "views": per_view, "review": [str(p) for p in paths],
            "before": str(before), "after": str(after), "heldout_cam": heldout_cam}


def _worse_counts(per_view: list[dict], tol: float = 0.0) -> dict:
    """Views where after > before (+ tol) per metric, of the views where both are defined."""
    out = {}
    for m in ("mae", "mae_gain", "mae_upper", "mae_gain_upper", "mae_face", "mae_gain_face"):
        pairs = [(p["before"][m], p["after"][m]) for p in per_view
                 if p["before"].get(m) is not None and p["after"].get(m) is not None]
        out[m] = {"worse": int(sum(a > b + tol for b, a in pairs)), "better": int(sum(a < b - tol for b, a in pairs)),
                  "n": len(pairs)}
    return out


def _gate(cam_summary: dict | None, cam: str, tol: float = 0.1) -> dict:
    """Acceptance gate for the full bake: no upper-body regression on the held-out camera, raw and gain-fitted."""
    if not cam_summary or cam_summary.get("n_views", 0) == 0:
        return {"cam": cam, "passed": None, "why": "no held-out views of this camera"}
    chk = {}
    for m in ("mae_upper", "mae_gain_upper"):
        b, a = cam_summary.get(f"before_{m}"), cam_summary.get(f"after_{m}")
        chk[m] = {"before": b, "after": a, "ok": None if b is None or a is None else bool(a <= b + tol)}
    oks = [c["ok"] for c in chk.values() if c["ok"] is not None]
    return {"cam": cam, "tolerance_srgb_levels": tol, "checks": chk, "passed": bool(oks) and all(oks)}


def _flow_compare(rb: dict, ra: dict, thr: float = 2.0) -> dict:
    """Per-view registration flow (90th percentile px over the dancers) of the last aligned pass, before vs after:
    views worse / better by more than thr px (the views lists hold the LAST pass's per-view stats)."""
    fb = {(v["cam"], v["i"]): v["flow"].get("p90_px") for v in rb.get("views", []) if v.get("flow")}
    fa = {(v["cam"], v["i"]): v["flow"].get("p90_px") for v in ra.get("views", []) if v.get("flow")}
    common = [k for k in fb if k in fa and fb[k] is not None and fa[k] is not None]
    d = np.array([fa[k] - fb[k] for k in common], float)
    med = lambda r: [p.get("flow_median_px") for p in r.get("passes", []) if p.get("aligned")]
    return {"views": len(common), "p90_worse_gt_2px": int((d > thr).sum()), "p90_better_gt_2px": int((d < -thr).sum()),
            "median_p90_delta_px": round(float(np.median(d)), 2) if len(d) else None,
            "flow_median_px_before": med(rb), "flow_median_px_after": med(ra)}


def _tm(d: dict, regions: dict) -> dict:
    return {r: texture_metrics(d["tex"][r]["srgb"], regions, d["tex"][r]["hair"]) for r in ROLES}


def _summarise(per_view: list[dict]) -> dict:
    """Pixel-weighted means over the views (body: px, upper: upper_px, face: face_px)."""
    out = {"n_views": len(per_view), "cams": sorted({p["cam"] for p in per_view})}
    for k in ("before", "after"):
        for m in ("mae", "mae_gain", "mae_upper", "mae_gain_upper", "mae_face", "mae_gain_face", "false_hair_pct"):
            wkey = "upper_px" if "upper" in m else ("face_px" if "face" in m else "px")
            sel = [p for p in per_view if p[k].get(m) is not None]
            vals = np.array([p[k][m] for p in sel], float)
            wts = np.array([p[wkey] for p in sel], float)
            out[f"{k}_{m}"] = round(float((vals * wts).sum() / max(wts.sum(), 1)), 2) if len(vals) else None
    return out


def _crop_box(mask: np.ndarray, aspect: float = 0.62, pad: float = 0.1):
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h = (y1 - y0) * (1 + 2 * pad)
    w = max((x1 - x0) * (1 + 2 * pad), h * aspect)
    h = max(h, w / aspect)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def _crop(img, box, out_h):
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    pad = [(max(0, -y0), max(0, y1 - H)), (max(0, -x0), max(0, x1 - W))] + [(0, 0)] * (img.ndim - 2)
    c = np.pad(img, pad)[y0 + pad[0][0]:y1 + pad[0][0], x0 + pad[1][0]:x1 + pad[1][0]]
    s = out_h / c.shape[0]
    return cv2.resize(c, (max(1, int(round(c.shape[1] * s))), out_h),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def _before_after_sheet(rows, path: Path, take: str, summ: dict, heldout_cam, panel_h: int = 520,
                        per_line: int = 2, max_rows: int = 8) -> Path:
    """Photo | before | after crops of her head-to-waist for one held-out view per camera (the one with the most
    visible chest / shoulders / back); the numbers in the header are over every held-out view."""
    best = {}
    for row in rows:
        c = row[0]["cam"]
        if c not in best or row[4]["upper_px"] > best[c][4]["upper_px"]:
            best[c] = row
    sel = [best[c] for c in sorted(best)][:max_rows]
    tiles = []
    for v, img, ren, focus, rec in sel:
        ys = np.nonzero(focus.any(1))[0]
        upper = focus.copy()
        upper[int(ys[0] + 0.6 * (ys[-1] - ys[0])):] = False  # head to waist
        box = _crop_box(upper, aspect=0.8, pad=0.06)
        a = _crop(np.ascontiguousarray(img[:, :, ::-1]), box, panel_h)
        _label(a, f"cam {v['cam']} #{v['i']} t={v['t']:.2f}s", (6, 24), (255, 255, 255), 0.6)
        _label(a, "held-out photo", (6, panel_h - 10), (255, 255, 255), 0.6)
        parts = [a]
        for k, r in ren:
            t = _crop(np.ascontiguousarray(r[:, :, ::-1]), box, panel_h)
            _label(t, f"{k.upper()}  MAE {rec[k]['mae_gain']:.1f}", (6, 24), (255, 255, 255), 0.6)
            _label(t, f"false hair px {rec[k]['false_hair_pct']:.1f}%", (6, 48), (220, 220, 220), 0.55)
            parts.append(t)
        tiles.append(np.concatenate(parts + [np.zeros((panel_h, 10, 3), np.uint8)], 1))
    if not tiles:
        tiles = [np.zeros((panel_h, 600, 3), np.uint8)]
    lines = []
    for a in range(0, len(tiles), per_line):
        grp = tiles[a:a + per_line]
        lines.append(np.concatenate(grp, 1))
    w = max(l.shape[1] for l in lines)
    sheet = np.concatenate([np.pad(l, ((0, 8), (0, w - l.shape[1]), (0, 0))) for l in lines], 0)
    what = f"leave-one-camera-out: cam {heldout_cam} was never baked" if heldout_cam else \
        "frames the bakes never sampled (1 frame from baked ones: weak test)"
    head = np.full((170, w, 3), 12, np.uint8)
    _label(head, f"{take} FOLLOW (Larissa) texture, held-out photos ({what})", (10, 32), (255, 255, 255), 0.8)
    _label(head, "BEFORE = old bake (soft hair down-weighting, one weighted mean); AFTER = hair excluded, anatomical "
                 "scalp, guarded class-aware fill, two-band fusion; same views, same settings", (10, 66), (230, 230, 230), 0.62)
    _label(head, f"all {summ['n_views']} held-out views, body-pixel MAE (sRGB 8-bit) raw {summ['before_mae']} -> "
                 f"{summ['after_mae']}, gain-fitted {summ['before_mae_gain']} -> {summ['after_mae_gain']}; chest/shoulders/"
                 f"back raw {summ['before_mae_upper']} -> {summ['after_mae_upper']}, gain-fitted "
                 f"{summ['before_mae_gain_upper']} -> {summ['after_mae_gain_upper']}", (10, 102), (230, 230, 120), 0.62)
    g = summ.get("gate")
    gtxt = (f"GATE cam {g['cam']} upper body raw {g['checks']['mae_upper']['before']} -> {g['checks']['mae_upper']['after']}, "
            f"gain-fitted {g['checks']['mae_gain_upper']['before']} -> {g['checks']['mae_gain_upper']['after']}: "
            f"{'PASS' if g['passed'] else 'FAIL'}") if g and g.get("checks") else ""
    _label(head, f"face raw {summ['before_mae_face']} -> {summ['after_mae_face']}; rendered hair-coloured where the photo "
                 f"is not {summ['before_false_hair_pct']}% -> {summ['after_false_hair_pct']}%   {gtxt}", (10, 138),
           (230, 230, 120), 0.62)
    sheet = np.concatenate([head, sheet], 0)
    if sheet.shape[1] > 3600:
        s = 3600 / sheet.shape[1]
        sheet = cv2.resize(sheet, (3600, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def _atlas_img(atlas, srgb):
    from .avatar_texture import gutter
    img = gutter(atlas.to_image(srgb), atlas.to_image(np.ones(atlas.n, np.uint8)) > 0, 4)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]


def _uv_sheet(B, A, atlas, path: Path, take: str, tile: int = 1100) -> Path:
    """Atlas crops before / after: follow head + torso (hair), lead head + torso (view seams)."""
    S = atlas.size
    crops = {"head": (0.0, 0.0, 0.5, 0.36), "torso": (0.0, 0.36, 0.5, 1.0)}  # (x0, y0, x1, y1) fractions of the atlas
    rows = []
    for role in ("follow", "lead"):
        for name, (x0, y0, x1, y1) in crops.items():
            parts = []
            for lab, d in (("BEFORE", B), ("AFTER", A)):
                img = _atlas_img(atlas, d["tex"][role]["srgb"])[int(y0 * S):int(y1 * S), int(x0 * S):int(x1 * S)]
                s = tile / img.shape[1]
                img = cv2.resize(img, (tile, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
                _label(img, f"{lab}: {role} albedo, {name} islands", (10, 34), (255, 255, 255), 0.9)
                parts.append(img)
            hair = A["tex"][role]["hair"]
            if hair is not None and name == "head":
                hm = atlas.to_image((hair * 255).astype(np.uint8))[int(y0 * S):int(y1 * S), int(x0 * S):int(x1 * S)]
                hm = cv2.resize(hm, (parts[1].shape[1], parts[1].shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                edge = cv2.morphologyEx(hm.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
                ov = parts[1].copy()
                ov[edge] = (0, 255, 255)
                ov[:50] = 30
                _label(ov, "AFTER + scalp-under-hair border (yellow): anatomical scalp + learned hairline",
                       (10, 34), (255, 255, 255), 0.9)
                parts.append(ov)
            rows.append(np.concatenate(parts, 1))
    w = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0))) for r in rows], 0)
    head = np.full((50, w, 3), 12, np.uint8)
    _label(head, f"{take} albedo atlases before / after (SMPL-X UV; head island, torso island): follow = hair removed "
                 "from the body, lead = view seams", (10, 34), (255, 255, 255), 0.8)
    sheet = np.concatenate([head, sheet], 0)
    if sheet.shape[1] > 3400:
        s = 3400 / sheet.shape[1]
        sheet = cv2.resize(sheet, (3400, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def _turntables(torch, geo, bodies, cols, path: Path, take: str, device) -> Path:
    from .avatar_review import _look_at
    H, W = 900, 560
    k = len(bodies["follow"].t) // 2
    lines = []
    for lab, role, key in (("BEFORE", "follow", "before"), ("AFTER", "follow", "after"), ("AFTER", "lead", "after")):
        b = bodies[role]
        v = b.verts[k]
        center = v.mean(0)
        hgt = v[:, 1].max() - v[:, 1].min()
        f = H * 3.2 / (hgt * 1.25)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        J = b.joints[k]
        fwd = np.cross(J[2] - J[1], np.array([0.0, 1.0, 0.0]))
        fwd = -fwd / max(np.linalg.norm(fwd), 1e-9)
        tiles = []
        for ang, name in ((0, "front"), (90, "side"), (180, "back"), (270, "side"), (35, "3/4 front"), (215, "3/4 back")):
            a = np.deg2rad(ang)
            Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            C = center + Ry @ fwd * 3.2
            R, t = _look_at(C, center)
            ren, rid = render(torch, geo, {role: v}, {role: cols[key][role]}, K, R, t, 0.0, H, W, device)
            img = np.where((rid >= 0)[..., None], ren, 0.16)
            tile = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
            _label(tile, f"{lab} {LABEL[role]} {name}", (8, 28), (255, 255, 255), 0.65)
            tiles.append(tile)
        lines.append(np.concatenate(tiles, 1))
    sheet = np.concatenate(lines, 0)
    head = np.full((46, sheet.shape[1], 3), 12, np.uint8)
    _label(head, f"{take}  turntable at t={bodies['follow'].t[k]:.2f}s: follow before / after, lead after (no hair "
                 "geometry: the Unity hair cards go on the learned scalp)", (10, 32), (255, 255, 255), 0.75)
    sheet = np.concatenate([head, sheet], 0)
    if sheet.shape[1] > 3600:
        s = 3600 / sheet.shape[1]
        sheet = cv2.resize(sheet, (3600, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def _faces(torch, geo, work, bodies, cols, A, path: Path, take: str, device, tile: int = 300) -> Path:
    from .avatar_review import _face_views
    rep = A["report"]
    stats = rep["views"]
    views, _ = camera_views(work, rep["cams"], rep["t_window"][0], rep["t_window"][1], float(rep["config"]["step_s"]))
    rows = []
    for role in ROLES:
        tiles = []
        for k, sc, uv in _face_views(bodies, views, stats, role, n=3):
            v = views[k]
            img = cv2.imread(str(work / "frames" / v["cam"] / f"{v['i']:06d}.jpg"))
            H, W = img.shape[:2]
            posed = {r: b.at(v["t"])[0] for r, b in bodies.items()}
            (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
            c, h = (uv.min(0) + uv.max(0)) / 2, max(x1 - x0, y1 - y0) * 1.6
            box = (int(c[0] - h / 2), int(c[1] - h / 2), int(c[0] + h / 2), int(c[1] + h / 2))
            a = _crop(img, box, tile)
            _label(a, f"{LABEL[role]} cam {v['cam']} #{v['i']}", (6, 22), (255, 255, 255), 0.5)
            parts = [a]
            for key in ("before", "after"):
                ren, rid = render(torch, geo, posed, cols[key], v["K"], v["R"], v["t_w"], v["k1"], H, W, device)
                ren8 = (np.clip(ren, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
                bg = np.where((rid >= 0)[..., None], ren8, 40).astype(np.uint8)
                b = _crop(bg, box, tile)
                _label(b, key, (6, 22), (255, 255, 255), 0.5)
                parts.append(b)
            tiles += parts + [np.zeros((tile, 6, 3), np.uint8)]
        if tiles:
            rows.append(np.concatenate(tiles, 1))
    w = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0))) for r in rows], 0)
    head = np.full((40, w, 3), 12, np.uint8)
    _label(head, f"{take}  face close-ups where each face is seen best: photo | before | after (upscaled crops)",
           (8, 28), (255, 255, 255), 0.6)
    cv2.imwrite(str(path), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def _head_region(torch, geo, bodies, A, B, path: Path, take: str, device, size: int = 420) -> Path:
    """Her head from 5 angles: before / after albedo, and the after scalp-under-hair region (green; where the Unity
    hair roots) and hair cover (blue; where her hanging hair covered the body in the photos, NOT a root region)."""
    from .avatar_review import _look_at
    b = bodies["follow"]
    k = len(b.t) // 2
    v, J = b.verts[k], b.joints[k]
    head = J[15]
    f = size * 4.8
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    fwd = np.cross(J[2] - J[1], np.array([0.0, 1.0, 0.0]))
    fwd = -fwd / max(np.linalg.norm(fwd), 1e-9)
    srgb_a = A["tex"]["follow"]["srgb"]
    hr = A["tex"]["follow"]["hair"]
    hr = hr.astype(np.float32) if hr is not None else np.zeros(len(srgb_a), np.float32)
    z = np.load(A["dir"] / "follow_texels.npz")
    cover = (np.asarray(z["hair_cover"], np.float32) / 255.0) if "hair_cover" in z.files else np.zeros(len(srgb_a))
    ov = srgb_a * (1 - 0.65 * hr[:, None]) + 0.65 * hr[:, None] * np.array([0.1, 1.0, 0.2])
    cv_ = np.clip(cover, 0, 1)[:, None] * (1 - hr[:, None])
    ov = ov * (1 - 0.6 * cv_) + 0.6 * cv_ * np.array([0.15, 0.35, 1.0])
    rows = []
    for lab, col in (("BEFORE", B["tex"]["follow"]["srgb"]), ("AFTER", srgb_a),
                     ("AFTER scalp region (green) / hair cover (blue)", ov.astype(np.float32))):
        tiles = []
        for ang in (0, 35, 90, 270, 180):
            a = np.deg2rad(ang)
            Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            C = head + Ry @ fwd * 1.6
            R, t = _look_at(C, head)
            ren, rid = render(torch, geo, {"follow": v}, {"follow": torch.as_tensor(col, device=device)}, K, R, t, 0.0,
                              size, size, device)
            img = np.where((rid >= 0)[..., None], ren, 0.16)
            tile = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
            _label(tile, f"{lab if ang == 0 else ''} {ang} deg".strip(), (8, 24), (255, 255, 255), 0.55)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, 1))
    sheet = np.concatenate(rows, 0)
    headb = np.full((44, sheet.shape[1], 3), 12, np.uint8)
    _label(headb, f"{take} FOLLOW head at t={b.t[k]:.2f}s: green = scalp-under-hair region (hair cards root here: "
                  "anatomical scalp + learned hairline, never cheek / ear / jaw)", (10, 30), (255, 255, 255), 0.6)
    cv2.imwrite(str(path), np.concatenate([headb, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path
