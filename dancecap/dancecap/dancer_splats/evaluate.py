"""Held-out evaluation of the dancer splats against the textured avatar, gates (DANCER_SPLATS_RESEARCH 7.4) and the
review sheets.

Scored views: the held-out camera (03) over the whole window, the held-out instants of the other cameras (every 10th
grid frame) and a sample of training views (fitting metric). Inside the union of the dancers' masks (minus ignore
pixels): PSNR / SSIM / LPIPS (alex, when the lpips package is installed) and MAE (sRGB 0-255), raw and after a
least-squares affine colour fit (as room.evaluate); per-role and hair IoU; ghost fraction (rendered alpha > 0.5 outside
the union mask dilated by 2 % of the crop diagonal, as a fraction of the crop). The avatar (its albedo on the same posed
SMPL-X, rendered by the prep) is scored on the same pixels. Temporal stability: a fixed novel camera halfway between
cameras 03 and 04, mean frame-to-frame change of the body (hair excluded) vs the avatar's.

Held-out colour transforms: a held-out camera has none (identity = the mean transform, held there by the regulariser);
held-out instants of a training camera use that camera's mean transform.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .. import progress as _progress
from . import ROLES, SplatConfig
from .data import decode_labels, load_bundle, read_avatar, read_view
from .model import AGSModel, composite
from .train import _view_tensors, write_json_atomic

HAIR_TURNS = (59.18, 60.88)


def _lpips(device):
    try:
        import lpips
        net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        return net
    except Exception:  # noqa: BLE001 - optional (not installed locally; the pod job installs it)
        return None


def _lpips_score(net, pred: np.ndarray, gt: np.ndarray, sel: np.ndarray, device) -> float | None:
    if net is None:
        return None
    import torch
    p = np.where(sel[..., None], pred, gt)
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device).permute(2, 0, 1)[None] * 2 - 1  # noqa: E731
    with torch.no_grad():
        return float(net(t(p), t(gt)).mean())


def masked_metrics_dev(render: np.ndarray, gt: np.ndarray, valid: np.ndarray, device="cpu") -> dict:
    """room.masked_metrics with the SSIM convolutions on `device` (the GPU on the pod: ~200 views x 12 SSIMs)."""
    import torch

    from ..room import ssim_map
    mse = float(((render - gt) ** 2)[valid].mean())
    a = torch.as_tensor(render, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    b = torch.as_tensor(np.where(valid[..., None], gt, render), dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    with torch.no_grad():
        s = ssim_map(a, b)[0].mean(0)
        ss = float(s[torch.as_tensor(valid, device=device)].mean())
    return {"psnr": float(-10 * math.log10(max(mse, 1e-10))), "ssim": ss}


def image_metrics(pred: np.ndarray, gt: np.ndarray, sel: np.ndarray, net=None, device="cpu") -> dict:
    """pred, gt (H, W, 3) float [0, 1]; sel (H, W) bool: raw + colour-corrected PSNR / SSIM / MAE (+ LPIPS)."""
    from ..room import fit_affine
    if sel.sum() < 50:
        return {}
    m = masked_metrics_dev(pred, gt, sel, device)
    cc = fit_affine(pred, gt, sel)
    mc = masked_metrics_dev(cc, gt, sel, device)
    out = {"psnr": m["psnr"], "ssim": m["ssim"], "psnr_cc": mc["psnr"], "ssim_cc": mc["ssim"],
           "mae": float(np.abs(pred - gt)[sel].mean() * 255), "mae_cc": float(np.abs(cc - gt)[sel].mean() * 255)}
    if net is not None:
        out["lpips"] = _lpips_score(net, pred, gt, sel, device)
        out["lpips_cc"] = _lpips_score(net, cc, gt, sel, device)
    return out


def iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float | None:
    u = ((a | b) & valid).sum()
    return float((a & b & valid).sum() / u) if u else None


def ghost_fraction(alpha: np.ndarray, fg: np.ndarray, valid: np.ndarray, frac: float = 0.02) -> float:
    H, W = alpha.shape
    r = max(1, int(round(frac * math.hypot(H, W))))
    dil = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    return float(((alpha > 0.5) & valid & ~dil).sum() / (H * W))


def render_view(model: AGSModel, v: dict, by_cam: dict, sh_degree: int, groups=None) -> dict:
    """numpy renders of one view: rgb (colour transform of the view, black background), alpha, role alphas."""
    import torch
    _view_tensors(model, v)
    with torch.no_grad():
        Gs = model.gaussians(v["t"], groups=groups or tuple(model.groups))
        r = model.render(Gs, v, sh_degree)
        A, b = model.appearance_for(v, by_cam)
        rgb = composite(model, r, A, b, torch.zeros(3, device=model.dev))
        return {"rgb": rgb.clamp(0, 1).cpu().numpy(), "alpha": r["alpha"].cpu().numpy(),
                "roles": r["roles"].cpu().numpy()}


def score_view(model: AGSModel, B: dict, v: dict, by_cam: dict, net, sh: int) -> tuple[dict, dict]:
    img, L = read_view(B["dir"], v)
    gt = img.astype(np.float32) / 255
    m = decode_labels(L)
    sel = m["fg"] & m["valid"]
    r = render_view(model, v, by_cam, sh)
    row = {"i": v["i"], "cam": v["cam"], "split": v["split"], "t": v["t"], "frame": v["frame"],
           "fg_px": int(sel.sum())}
    s = image_metrics(r["rgb"], gt, sel, net, model.dev)
    row.update({f"splat_{k}": vv for k, vv in s.items()})
    lead_t = m["lead"] | m["both"]
    fol_t = m["follow"] | m["hair"] | m["both"]
    ro = r["roles"]
    row["splat_iou_union"] = iou(r["alpha"] > 0.5, m["fg"], m["valid"])
    row["splat_iou_lead"] = iou(ro[..., 0] > 0.5, lead_t, m["valid"])
    row["splat_iou_follow"] = iou((ro[..., 1] + ro[..., 2]) > 0.5, fol_t, m["valid"])
    row["splat_iou_hair"] = iou(ro[..., 2] > 0.5, m["hair"], m["valid"]) if m["hair"].any() else None
    row["ghost_frac"] = ghost_fraction(r["alpha"], m["fg"], m["valid"])
    av = read_avatar(B["dir"], v)
    if av is not None:
        a_img, idm = av
        s = image_metrics(a_img, gt, sel, net, model.dev)
        row.update({f"avatar_{k}": vv for k, vv in s.items()})
        row["avatar_iou_union"] = iou(idm >= 0, m["fg"], m["valid"])
        row["avatar_iou_lead"] = iou(idm == 0, lead_t, m["valid"])
        row["avatar_iou_follow"] = iou(idm == 1, fol_t, m["valid"])
    # per dancer: each one's own pixels (the overlap 'both' is excluded: owner unknown); same pixels for the avatar
    for role, msk in (("lead", m["lead"]), ("follow", m["follow"] | m["hair"])):
        sr = msk & m["valid"]
        for who, img_ in (("splat", r["rgb"]), ("avatar", av[0] if av is not None else None)):
            if img_ is None:
                continue
            s = image_metrics(img_, gt, sr, net, model.dev)
            row.update({f"{who}_{role}_{k}": s[k] for k in PER_DANCER_KEYS if k in s})
    if v["cam"] in CROP_CAMS:
        try:
            row.update(crop_metrics(B, v, gt, m, r["rgb"], av[0] if av is not None else None, net, model.dev))
        except Exception as e:  # noqa: BLE001 - a crop never fails the eval
            row["crop_error"] = f"{type(e).__name__}: {e}"[:200]
    return row, {"gt": gt, "splat": r["rgb"], "avatar": av[0] if av is not None else None, "alpha": r["alpha"],
                 "labels": L}


PER_DANCER_KEYS = ("psnr", "ssim", "psnr_cc", "ssim_cc", "mae_cc", "lpips", "lpips_cc")
CROP_CAMS = ("02", "06", "09")  # the cameras with face detail (02, 06, zoomed 09)
FACE_HALF_M = 0.16   # face crop: a 32 cm square around the head (skull centre)
HAIR_HALF_M = 0.30   # hair crop: a 60 cm square from the crown down past the shoulder blades
LPIPS_MIN_SIDE = 64  # crops smaller than this are upsampled for LPIPS (alexnet's receptive field)


def head_points(B: dict, role: str, t: float) -> tuple[np.ndarray, np.ndarray]:
    """(skull centre, head-joint) world positions of one dancer at time t (the fit, no pose corrections)."""
    from .data import at_time
    bd = B["bodies"]
    tg = np.asarray(bd["t_grid"], float)
    J15 = at_time(bd[f"{role}_Jp"][:, 15], tg, t)
    Rh = at_time(bd[f"{role}_Rg"][:, 15], tg, t)
    return J15 + Rh @ np.array([0.0, 0.07, 0.01]), J15


def crop_box(v: dict, centre: np.ndarray, half_m: float, dy_m: float = 0.0) -> tuple[int, int, int, int] | None:
    """Square pixel box (x0, y0, x1, y1) of half-size half_m metres around a world point (shifted dy_m down), clipped
    to the crop; None when behind the camera or outside."""
    from .data import project
    c = np.asarray(centre, float) - np.array([0.0, dy_m, 0.0])
    px, z = project(c[None], np.asarray(v["K"]), np.asarray(v["R"]), np.asarray(v["t_w"]))
    if z[0] <= 0.1:
        return None
    h = float(v["K"][0][0]) * half_m / float(z[0])
    x0, y0 = int(math.floor(px[0, 0] - h)), int(math.floor(px[0, 1] - h))
    x1, y1 = int(math.ceil(px[0, 0] + h)), int(math.ceil(px[0, 1] + h))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(v["W"], x1), min(v["H"], y1)
    return (x0, y0, x1, y1) if x1 - x0 >= 6 and y1 - y0 >= 6 else None


def _crop_scores(pred, gt, sel, net, device) -> dict:
    from ..room import fit_affine
    if sel.sum() < 30:
        return {}
    m = masked_metrics_dev(pred, gt, sel, device)
    cc = fit_affine(pred, gt, sel)
    mc = masked_metrics_dev(cc, gt, sel, device)
    out = {"psnr": m["psnr"], "ssim": m["ssim"], "psnr_cc": mc["psnr"], "ssim_cc": mc["ssim"], "px": int(sel.sum())}
    if net is not None:
        f = max(1.0, LPIPS_MIN_SIDE / min(gt.shape[:2]))
        if f > 1:
            sz = (int(round(gt.shape[1] * f)), int(round(gt.shape[0] * f)))
            up = lambda a: cv2.resize(a.astype(np.float32), sz, interpolation=cv2.INTER_CUBIC).clip(0, 1)  # noqa: E731
            sel_u = cv2.resize(sel.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
            out["lpips"] = _lpips_score(net, up(pred), up(gt), sel_u, device)
            out["lpips_cc"] = _lpips_score(net, up(cc), up(gt), sel_u, device)
        else:
            out["lpips"] = _lpips_score(net, pred, gt, sel, device)
            out["lpips_cc"] = _lpips_score(net, cc, gt, sel, device)
    return out


def crop_metrics(B: dict, v: dict, gt, m: dict, splat, avatar, net, device) -> dict:
    """Face crops of both dancers and the follow's hair crop (cameras 02 / 06 / 09): splat vs avatar on the dancer's own
    pixels inside the crop. Keys crop_<face_lead|face_follow|hair>_<splat|avatar>_<metric> + crop_<kind>_px_side."""
    out = {}
    hair_role = str(B["hair"]["role"]) if B.get("hair") is not None else "follow"
    kinds = [("face_lead", "lead", FACE_HALF_M, 0.0), ("face_follow", "follow", FACE_HALF_M, 0.0),
             ("hair", hair_role, HAIR_HALF_M, 0.18)]
    for kind, role, half, dy in kinds:
        sc, _ = head_points(B, role, v["t"])
        box = crop_box(v, sc, half, dy)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        own = (m["lead"] if role == "lead" else (m["follow"] | m["hair"])) & m["valid"]
        if kind == "hair":
            own = (m["hair"] | m["follow"]) & m["valid"]
        sel = own[y0:y1, x0:x1]
        out[f"crop_{kind}_px_side"] = int(max(x1 - x0, y1 - y0))
        for who, img_ in (("splat", splat), ("avatar", avatar)):
            if img_ is None:
                continue
            s = _crop_scores(img_[y0:y1, x0:x1], gt[y0:y1, x0:x1], sel, net, device)
            out.update({f"crop_{kind}_{who}_{k}": vv for k, vv in s.items()})
    return out


def _med(rows, key):
    xs = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.median(xs)), 4) if xs else None


def summarise(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r if k.startswith(("splat_", "avatar_", "crop_")) or k == "ghost_frac"})
    out = {}
    groups = {"heldout_cam": [r for r in rows if r["split"] == "test_cam"],
              "heldout_time": [r for r in rows if r["split"] == "test_time"],
              "train": [r for r in rows if r["split"] == "train"]}
    for c in sorted({r["cam"] for r in rows}):  # per camera: unseen instants / in-sample views
        groups[f"cam_{c}"] = [r for r in rows if r["cam"] == c and r["split"] != "train"]
        groups[f"cam_{c}_insample"] = [r for r in rows if r["cam"] == c and r["split"] == "train"]
    for name, rs in groups.items():
        if rs:
            out[name] = {"n": len(rs), **{k: _med(rs, k) for k in keys}}
            out[name]["ghost_frac_max"] = round(max(r["ghost_frac"] for r in rs), 5)
    return out


def compact_tables(summ: dict) -> dict:
    """The asked-for comparison in one place: splat vs avatar PSNR / SSIM / LPIPS (colour-corrected and raw) per group
    (held-out camera 03, held-out instants of the training cameras, each camera, in-sample) and per dancer; face /
    hair crops of cameras 02 / 06 / 09."""
    mets = ("psnr", "psnr_cc", "ssim", "ssim_cc", "lpips", "lpips_cc")
    per = {}
    for g, s in summ.items():
        row = {"n": s["n"]}
        for who in ("splat", "avatar"):
            row[who] = {m: s.get(f"{who}_{m}") for m in mets}
            for role in ("lead", "follow"):
                row[f"{who}_{role}"] = {m: s.get(f"{who}_{role}_{m}") for m in mets}
        per[g] = row
    crops = {}
    for g, s in summ.items():
        if not g.startswith("cam_"):
            continue
        for kind in ("face_lead", "face_follow", "hair"):
            vals = {who: {m: s.get(f"crop_{kind}_{who}_{m}") for m in mets} for who in ("splat", "avatar")}
            if any(x is not None for d in vals.values() for x in d.values()):
                crops.setdefault(g, {})[kind] = {**vals, "px_side": s.get(f"crop_{kind}_px_side")}
    return {"splat_vs_avatar": per, "crops": crops}


def temporal(model: AGSModel, B: dict, sh: int) -> dict | None:
    """Body-only frame-to-frame change seen from the fixed novel camera vs the avatar's (prep)."""
    import torch
    nov = B.get("novel")
    if not nov:
        return None
    v = {"i": -1, "cam": "novel", "split": "novel", "K": nov["K"], "R": nov["R"], "t_w": nov["t_w"], "W": nov["W"],
         "H": nov["H"]}
    prev, ch, frames = None, [], {}
    for k in nov["grid"]:
        v["t"] = float(model.t_grid[k])
        r = render_view(model, v, {}, sh, groups=("lead", "follow"))
        if prev is not None:
            sel = (r["alpha"] > 0.5) | (prev["alpha"] > 0.5)
            if sel.any():
                ch.append(float(np.abs(r["rgb"] - prev["rgb"])[sel].mean()))
        frames[k] = r
        prev = r
    torch.cuda.empty_cache()
    av = nov.get("avatar_change")
    sp = float(np.mean(ch)) if ch else None
    return {"splat_change": round(sp, 5) if sp is not None else None, "avatar_change": av,
            "ratio": round(sp / av, 3) if sp and av else None, "frames": len(nov["grid"]),
            "_frames": frames}


def gates(summ: dict, rows: list[dict], temp: dict | None, train_s: float | None) -> dict:
    g = {}
    hc = [r for r in rows if r["split"] == "test_cam"]
    held = [r for r in rows if r["split"] in ("test_cam", "test_time")]
    if held:
        mx = max(r["ghost_frac"] for r in held)
        g["1_no_ghosts"] = {"passed": mx <= 0.01, "value": round(mx, 5),
                            "rule": "rendered alpha outside the 2 % dilated union mask <= 1 % of the crop on every "
                                    "held-out frame"}
    s = summ.get("heldout_cam")
    if s:
        fi, li = s.get("splat_iou_follow"), s.get("splat_iou_lead")
        g["2_silhouette"] = {"passed": bool(fi is not None and li is not None and fi >= 0.85 and li >= 0.88),
                             "value": {"follow": fi, "lead": li, "avatar_follow": s.get("avatar_iou_follow"),
                                       "avatar_lead": s.get("avatar_iou_lead")},
                             "rule": "held-out camera median IoU vs SAM: follow >= 0.85, lead >= 0.88"}
        tr = summ.get("train", {})
        sm, am = s.get("splat_mae_cc"), s.get("avatar_mae_cc")
        ok = bool(sm is not None and am is not None and sm <= 0.6 * am and (s.get("splat_psnr_cc") or 0) >= 24
                  and (tr.get("splat_psnr_cc") or 0) >= 27)
        g["3_colour"] = {"passed": ok, "value": {"mae_cc": sm, "avatar_mae_cc": am, "psnr_cc": s.get("splat_psnr_cc"),
                                                 "train_psnr_cc": tr.get("splat_psnr_cc")},
                         "rule": "held-out masked MAE (colour-corrected) <= 0.6 x the avatar's; PSNR >= 24 dB held "
                                 "out, >= 27 dB in-sample"}
        better = [s.get("splat_psnr_cc") is not None and s.get("avatar_psnr_cc") is not None
                  and s["splat_psnr_cc"] > s["avatar_psnr_cc"],
                  s.get("splat_ssim_cc") is not None and s.get("avatar_ssim_cc") is not None
                  and s["splat_ssim_cc"] >= s["avatar_ssim_cc"]]
        if s.get("splat_lpips_cc") is not None and s.get("avatar_lpips_cc") is not None:
            better.append(s["splat_lpips_cc"] <= s["avatar_lpips_cc"])
        g["heldout_vs_avatar"] = {"passed": all(better), "value": {
            k: {"splat": s.get(f"splat_{k}"), "avatar": s.get(f"avatar_{k}")}
            for k in ("psnr", "psnr_cc", "ssim", "ssim_cc", "lpips", "lpips_cc", "mae_cc")},
            "rule": "held-out camera: splat better than the textured avatar on PSNR, SSIM (and LPIPS when "
                    "available), colour-corrected"}
    if hc:
        vals = {}
        for tt in HAIR_TURNS:
            r = min(hc, key=lambda x: abs(x["t"] - tt))
            if abs(r["t"] - tt) < 0.05:
                vals[str(tt)] = r.get("splat_iou_hair")
        if vals:
            g["4_hair_turns"] = {"passed": all(x is not None and x >= 0.5 for x in vals.values()), "value": vals,
                                 "rule": "hair-mask IoU >= 0.5 at 59.18 s and 60.88 s on the held-out camera (and "
                                         "trailing hair visible on the review sheet)"}
    if temp and temp.get("ratio") is not None:
        g["5_temporal"] = {"passed": temp["ratio"] <= 1.1, "value": {k: temp[k] for k in ("splat_change",
                                                                                         "avatar_change", "ratio")},
                           "rule": "novel camera between 03 and 04: body frame-to-frame change <= avatar's + 10 %"}
    g["6_face"] = {"passed": None, "rule": "no worse than avatar_face at camera 02: the user's eye (review sheets)"}
    g["7_unity"] = {"passed": None, "rule": "plays the 105 frames in layer G at >= 30 fps (after export; Unity)"}
    if train_s is not None:
        g["8_cost"] = {"passed": train_s <= 30 * 60, "value": {"train_min": round(train_s / 60, 1)},
                       "rule": "training <= 30 min (session <= $1.50: the runner's ledger)"}
    return g


# ------------------------------------------------------------------------------------------------ review

def _u8(x):
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)[:, :, ::-1]


def _tile(img, h: int, label: str | None = None):
    im = img if img.dtype == np.uint8 else _u8(img)
    s = h / im.shape[0]
    im = cv2.resize(im, (max(1, int(im.shape[1] * s)), h), interpolation=cv2.INTER_AREA)
    if label:
        cv2.putText(im, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def review_sheets(review: Path, picks: list[tuple[dict, dict]], temp: dict | None, B: dict, prefix: str = "dancer_splats",
                  h: int = 360) -> list[str]:
    review.mkdir(parents=True, exist_ok=True)
    written, rows = [], []
    for row, im in picks:
        gt, sp, av = im["gt"], im["splat"], im["avatar"]
        fg = decode_labels(im["labels"])["fg"]
        e = np.where(fg, np.abs(sp - gt).mean(-1), 0.0)[..., None]
        err = cv2.applyColorMap(_u8(np.repeat(e * 4, 3, -1))[:, :, 0], cv2.COLORMAP_INFERNO)
        tiles = [_tile(gt, h, f"real cam {row['cam']} t {row['t']:.2f}"),
                 _tile(sp, h, f"splat psnr_cc {row.get('splat_psnr_cc') or 0:.1f} iou {row.get('splat_iou_union') or 0:.2f}")]
        if av is not None:
            tiles.append(_tile(av, h, f"avatar psnr_cc {row.get('avatar_psnr_cc') or 0:.1f} "
                                      f"iou {row.get('avatar_iou_union') or 0:.2f}"))
        tiles.append(_tile(err, h, "|splat - real| x4 (dancers)"))
        strip = np.hstack(tiles)
        p = review / f"{prefix}_t{row['t']:.2f}_cam{row['cam']}.jpg"
        cv2.imwrite(str(p), strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written.append(p.name)
        rows.append(strip)
    if rows:
        W = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.zeros((r.shape[0], W - r.shape[1], 3), np.uint8)]) for r in rows]
        p = review / f"{prefix}_heldout.jpg"
        cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(p.name)
    if temp and temp.get("_frames"):
        fr = temp["_frames"]
        ks = sorted(fr)
        pick = [ks[len(ks) // 4], ks[len(ks) // 2], ks[3 * len(ks) // 4]]
        tiles = []
        for k in pick:
            tiles.append(_tile(fr[k]["rgb"], h, f"splat novel t {B['novel']['t'][ks.index(k)]:.2f}"))
            ap = B["dir"] / "eval" / f"novel_avatar_{int(k):04d}.jpg"
            if ap.exists():
                tiles.append(_tile(cv2.imread(str(ap)), h, "avatar"))
        p = review / f"{prefix}_novel.jpg"
        cv2.imwrite(str(p), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(p.name)
    return written


def pose_delta_plot(out: Path, review: Path, prefix: str = "dancer_splats") -> str | None:
    p = out / "pose_delta.npz"
    if not p.exists():
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    z = np.load(p)
    t = z["t_grid"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for r in ROLES:
        ax[0].plot(t, z[f"{r}_max_deg"], label=f"{r} max joint correction (deg)")
        ax[1].plot(t, z[f"{r}_dT_cm"], label=f"{r} root shift (cm)")
    if "hair_node_offset_m_mean" in z.files:
        ax[1].plot(t, z["hair_node_offset_m_mean"] * 100, label="hair node offset mean (cm)", ls="--")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
        for tt in HAIR_TURNS:
            if t[0] <= tt <= t[-1]:
                a.axvline(tt, color="k", alpha=0.2)
    ax[1].set_xlabel("reference time (s)")
    fig.suptitle("dancer_splats: per-frame corrections of the MAMMA fit (QA signal; smplx/ untouched)", fontsize=10)
    fig.tight_layout()
    review.mkdir(parents=True, exist_ok=True)
    q = review / f"{prefix}_pose_delta.png"
    fig.savefig(q, dpi=90)
    plt.close(fig)
    return q.name


def evaluate(bundle: Path, out: Path, cfg: SplatConfig, review: Path, log=print, prefix: str = "dancer_splats") -> dict:
    import torch

    from .. import gpu
    t0 = time.time()
    B = load_bundle(bundle)
    with gpu.exclusive("dancer_splats_eval"):
        model = AGSModel.load(B, out / "model.pt", cfg.device)
        sh = model.cfg.sh_degree
        by_cam = model.cam_app_index()
        net = _lpips(model.dev)
        sel = [v for v in B["views"] if v["split"] in ("test_cam", "test_time") or v.get("eval_train")]
        rows, picks = [], []
        for n, v in enumerate(sel):
            row, im = score_view(model, B, v, by_cam, net, sh)
            rows.append(row)
            if v.get("review"):
                picks.append((row, im))
            _progress.tick("splats/eval", n + 1, len(sel), "views")
            if (n + 1) % 25 == 0 or n + 1 == len(sel):
                log(f"[dancer_splats] eval {n + 1}/{len(sel)} views")
        _progress.end("splats/eval", done=len(sel))
        temp = temporal(model, B, sh)
        summ = summarise(rows)
        train_s = (model.extra or {}).get("train_s")
        gt = gates(summ, rows, temp, train_s)
        written = review_sheets(review, sorted(picks, key=lambda p: p[0]["t"]), temp, B, prefix)
        pd = pose_delta_plot(out, review, prefix)
        if pd:
            written.append(pd)
        del model
        torch.cuda.empty_cache()
    met = {"take": B["manifest"]["take"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "window": B["manifest"]["prep_config"]["window"], "heldout_cams": B["manifest"]["prep_config"]["holdout_cams"],
           "lpips": net is not None, "summary": summ, "tables": compact_tables(summ),
           "temporal": {k: v for k, v in (temp or {}).items() if not k.startswith("_")} or None,
           "gates": gt, "review": written, "views": rows, "eval_s": round(time.time() - t0, 1),
           "notes": ["avatar = its albedo (avatar/<role>_albedo.png, the video bake) on the same posed SMPL-X, no "
                     "hair geometry, rendered by the prep; its metrics are in-sample for the avatar (every camera "
                     "was used in the bake)",
                     "held-out camera: identity colour transform; held-out instants: their camera's mean transform"]}
    write_json_atomic(out / "metrics.json", met)
    rp = out / "report.json"
    rep = json.loads(rp.read_text()) if rp.exists() else {}
    rep["eval"] = {"summary": summ, "gates": gt, "temporal": met["temporal"], "eval_s": met["eval_s"],
                   "review": written}
    write_json_atomic(rp, rep)
    which = next((k for k in ("heldout_cam", "heldout_time", "train") if k in summ), None)
    s = summ.get(which) or {}
    log(f"[dancer_splats] {which}: splat PSNR_cc {s.get('splat_psnr_cc')} SSIM_cc {s.get('splat_ssim_cc')} "
        f"LPIPS {s.get('splat_lpips_cc')} vs avatar PSNR_cc {s.get('avatar_psnr_cc')} SSIM_cc "
        f"{s.get('avatar_ssim_cc')} LPIPS {s.get('avatar_lpips_cc')}; IoU follow {s.get('splat_iou_follow')} "
        f"lead {s.get('splat_iou_lead')}; ghosts max {s.get('ghost_frac_max')}")
    log("[dancer_splats] gates: " + ", ".join(f"{k} {'PASS' if v['passed'] else 'n/a' if v['passed'] is None else 'FAIL'}"
                                              for k, v in gt.items()))
    log(f"[dancer_splats] review -> {review} ({len(written)} files)")
    return met
