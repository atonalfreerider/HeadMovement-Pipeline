r"""john_lula LHM++ preview - the owner's side-by-sides and an INDEPENDENT check of the build (2026-10-19).

Reads only what dancecap.mono4d.jl_lhmpp_preview (the build) wrote under work/john_lula/lhmpp_preview/ (renders, pod
scene PLYs, cameras.json / prep.json / refs.json, pod_out/metrics.json) plus the take's own frames, masks and fit;
writes new files only (sheets/sbs_*, sheets/closeups_face_hair.jpg, sheets/turntable_12az.jpg, sheets/inputs_crops.jpg,
videos/jl_*.mp4, check/*). Nothing of the build or of larissa_kadu_38 is modified. Local, no pod.

    python -m dancecap.mono4d.jl_lhmpp_owner_sheets metrics     # _crop_scores on the held-out frames (CPU) + checks
    python -m dancecap.mono4d.jl_lhmpp_owner_sheets heldout     # sheets/sbs_heldout_<k>_of_4.jpg
    python -m dancecap.mono4d.jl_lhmpp_owner_sheets turntable   # sheets/turntable_12az.jpg
    python -m dancecap.mono4d.jl_lhmpp_owner_sheets closeups    # gsplat (GPU lock, MSVC/CUDA env) + avatar close-ups
    python -m dancecap.mono4d.jl_lhmpp_owner_sheets videos      # videos/jl_phone_path.mp4, jl_vr_seat.mp4 + decode check
    python -m dancecap.mono4d.jl_lhmpp_owner_sheets inputs      # sheets/inputs_crops.jpg (the build's inputs.jpg)
    python -m dancecap.mono4d.jl_lhmpp_owner_sheets extra       # check/extra_checks.json (clip IoU, fit jumps, bibs)

METRICS: dancer_splats.evaluate._crop_scores (the larissa_kadu_38 code) per held-out view, dancer and method on the
dancer's real-mask bbox + 10 % (cand_lhmpp._crop_box), inside the real mask (people/masks_01.npz, read directly), the
render composited over black (straight RGBA x alpha) against the frame read directly from frames/01/. LPIPS needs the
lpips package + AlexNet weights, which are not on this PC (downloading them was not approved), so LPIPS / LPIPS_cc are
the POD's values for the same rows (pod_out/metrics.json, the same _crop_scores); the local PSNR / SSIM of every row are
compared with the pod's to show the inputs are identical. IoU / coverage from the method's role map (R lead, G follow).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import jl_lhmpp_preview as P

OUT = P.OUT
INPUT, RENDERS, POD_OUT = P.INPUT, P.RENDERS, P.POD_OUT
SHEETS = OUT / "sheets"
VIDEOS = OUT / "videos"
CHECK = OUT / "check"
ROLES = P.ROLES
NAME = P.NAME
METHODS = ("avatar", "lhmpp_video8", "lhmpp_video8_photo", "lhmpp_video8_photofull")
SBS = ("avatar", "lhmpp_video8", "lhmpp_video8_photo")             # the sheet / video columns after the real frame
TITLE = {"real": "REAL phone frame (held out)", "avatar": "current avatar *",
         "lhmpp_video8": "LHM++ video8 (8 phone crops)", "lhmpp_video8_photo": "LHM++ video8 + photo",
         "lhmpp_video8_photofull": "diag.: + John's full occluded photo"}
SHORT = {"real": "real phone", "avatar": "current avatar *", "lhmpp_video8": "LHM++ video8",
         "lhmpp_video8_photo": "LHM++ video8+photo", "lhmpp_video8_photofull": "diag. +photo full"}
FOOT = ("* current avatar = offline render of the shipped texture on the SMPL-X body (export 08_JohnLula); the Unity "
        "build adds hair cards (Lula's ponytail), shoes and the portrait face shading. All methods posed by the same "
        "john_lula MAMMA mono fit.")
GREY = 128
MAXW = 1920
FONT = {"r": "C:/Windows/Fonts/arial.ttf", "b": "C:/Windows/Fonts/arialbd.ttf"}


def log(msg: str) -> None:
    print(f"[jl_owner] {msg}", flush=True)


# ================================================================================================ drawing helpers (RGB)
def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT["b" if bold else "r"], size)
    except OSError:
        return ImageFont.load_default()


def put(img: np.ndarray, text: str, xy, size: int = 18, bold: bool = True, anchor: str = "la") -> np.ndarray:
    """White text with a black outline drawn in place on an RGB uint8 image."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.text(xy, text, font=_font(size, bold), fill=(255, 255, 255), stroke_width=max(2, size // 9),
           stroke_fill=(0, 0, 0), anchor=anchor)
    img[:] = np.asarray(im)
    return img


def banner(lines, W: int, size: int = 20, bold_first: bool = True, pad: int = 6) -> np.ndarray:
    from PIL import Image, ImageDraw
    if isinstance(lines, str):
        lines = [lines]
    lh = int(size * 1.3)
    im = Image.new("RGB", (W, pad * 2 + lh * len(lines)), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for k, ln in enumerate(lines):
        d.text((8, pad + k * lh), ln, font=_font(size if k == 0 else size - 3, bold_first and k == 0), fill=(0, 0, 0))
    return np.asarray(im).copy()


def hcat(tiles, gap: int = 4, fill: int = 255) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    out = []
    for k, t in enumerate(tiles):
        t = np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0)), constant_values=fill)
        out.append(t)
        if k < len(tiles) - 1:
            out.append(np.full((h, gap, 3), fill, np.uint8))
    return np.concatenate(out, 1)


def vcat(rows, gap: int = 4, fill: int = 255) -> np.ndarray:
    W = max(r.shape[1] for r in rows)
    out = []
    for k, r in enumerate(rows):
        out.append(np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=fill))
        if k < len(rows) - 1:
            out.append(np.full((gap, W, 3), fill, np.uint8))
    return np.concatenate(out, 0)


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Lanczos when enlarging (the phone is only 538 x 960), area when shrinking."""
    up = w > img.shape[1] or h > img.shape[0]
    return cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LANCZOS4 if up else cv2.INTER_AREA)


def read_frame(i: int) -> np.ndarray:
    return cv2.imread(str(P.frame_path(i)), cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def read_rgba(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    return im[:, :, 2::-1].astype(np.float32) / 255.0, im[:, :, 3].astype(np.float32) / 255.0


def on_grey(p: Path, grey: int = GREY) -> np.ndarray | None:
    ra = read_rgba(p)
    if ra is None:
        return None
    rgb, a = ra
    return (np.clip(rgb * a[..., None] + grey / 255.0 * (1 - a[..., None]), 0, 1) * 255 + 0.5).astype(np.uint8)


def save_jpg(img: np.ndarray, p: Path, q: int = 92) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return p


def load_json(p: Path):
    return json.loads(Path(p).read_text())


def cams_all() -> list[dict]:
    return load_json(INPUT / "cameras.json")["renders"]


def project(X, K, R, tvec):
    X = np.asarray(X, float).reshape(-1, 3)
    Xc = X @ np.asarray(R, float).T + np.asarray(tvec, float).reshape(1, 3)
    uv = Xc[:, :2] / Xc[:, 2:3]
    K = np.asarray(K, float)
    return np.stack([K[0, 0] * uv[:, 0] + K[0, 2], K[1, 1] * uv[:, 1] + K[1, 2]], 1), Xc[:, 2]


def fits():
    from . import appearance_compare as AC
    return {r: AC.Fit(P.WORK / "smplx" / f"{r}.npz") for r in ROLES}


# ================================================================================================ metrics + checks
def metrics() -> dict:
    """Independent re-score of the held-out views (see the module doc) + identity / white-patch / data checks."""
    from ..dancer_splats.evaluate import _crop_scores
    from . import cand_lhmpp as CL
    t0 = time.time()
    cams = [c for c in cams_all() if c["group"] == "heldout"]
    M = P.Masks()
    pod = load_json(POD_OUT / "metrics.json")
    podrows = {(r["view"], r["role"], r["method"]): r for r in pod["rows"]}
    rows = []
    for c in cams:
        gt = read_frame(c["frame"]).astype(np.float32) / 255.0
        for meth in METHODS:
            ra = read_rgba(RENDERS / meth / c["out"])
            rid = cv2.imread(str(RENDERS / meth / "role" / f"{c['id']}.png"), cv2.IMREAD_COLOR)
            if ra is None or rid is None:
                log(f"missing render / role map: {meth} {c['id']}")
                continue
            rgb = ra[0] * ra[1][..., None]
            rw = rid[:, :, ::-1].astype(np.float32) / 255.0
            for role in ROLES:
                m = M.get(c["frame"], role)
                if m is None or m.sum() < 50:
                    continue
                x0, y0, x1, y1 = CL._crop_box(m)
                s = _crop_scores(rgb[y0:y1, x0:x1], gt[y0:y1, x0:x1], m[y0:y1, x0:x1], None, "cpu")
                k_own, k_oth = ROLES.index(role), 1 - ROLES.index(role)
                own, oth = rw[:, :, k_own] > 0.5, rw[:, :, k_oth] > 0.5
                iou = float((own & m).sum() / max(1, (own | m).sum()))
                iou_sw = float((oth & m).sum() / max(1, (oth | m).sum()))
                both = own & m
                lum = lambda im, sel: float(np.median(im[sel] @ np.array([0.2126, 0.7152, 0.0722]))) if sel.sum() else None  # noqa: E731
                wh = lambda im, sel: float((((im.min(-1) > 0.80) & (im.max(-1) - im.min(-1) < 0.12))[sel]).mean()) if sel.sum() else None  # noqa: E731
                ra_ = ra[0]
                row = {"view": c["id"], "frame": c["frame"], "t": round(c["t"], 4), "role": role, "method": meth,
                       "crop": [int(x0), int(y0), int(x1), int(y1)], **{k: float(v) for k, v in s.items()},
                       "iou": iou, "iou_if_swapped": iou_sw, "coverage": float(both.sum() / m.sum()),
                       "lum_render": lum(ra_, both), "lum_real": lum(gt, both),
                       "white_render": wh(ra_, both), "white_real": wh(gt, both)}
                pr = podrows.get((c["id"], role, meth))
                if pr is not None:
                    row["lpips_pod"] = pr.get("lpips")
                    row["lpips_cc_pod"] = pr.get("lpips_cc")
                    row["d_psnr_vs_pod"] = abs(row["psnr"] - pr["psnr"])
                    row["d_ssim_vs_pod"] = abs(row["ssim"] - pr["ssim"])
                    row["d_psnr_cc_vs_pod"] = abs(row["psnr_cc"] - pr["psnr_cc"])
                    row["d_iou_vs_pod"] = abs(row["iou"] - pr["iou"]) if pr.get("iou") is not None else None
                rows.append(row)
    keys = ("psnr", "ssim", "lpips_pod", "psnr_cc", "ssim_cc", "lpips_cc_pod", "iou", "coverage", "iou_if_swapped",
            "white_render", "white_real", "lum_render", "lum_real")
    summ = {}
    for meth in METHODS:
        summ[meth] = {}
        for g in ("all",) + ROLES:
            rr = [r for r in rows if r["method"] == meth and (g == "all" or r["role"] == g)]
            summ[meth][g] = {"n": len(rr), **{k: (round(float(np.median([r[k] for r in rr if r.get(k) is not None])), 4)
                                                  if any(r.get(k) is not None for r in rr) else None) for k in keys}}
    agree = {k: round(float(max((r[k] for r in rows if r.get(k) is not None), default=float("nan"))), 6)
             for k in ("d_psnr_vs_pod", "d_ssim_vs_pod", "d_psnr_cc_vs_pod", "d_iou_vs_pod")}
    swaps = [(r["view"], r["role"], r["method"]) for r in rows if r["iou_if_swapped"] > r["iou"]]
    # per-view win counts (LHM++ vs the current avatar), on PSNR and the pod's LPIPS
    wins = {}
    for meth in METHODS[1:]:
        w = {"psnr": 0, "lpips": 0, "n": 0}
        for r in rows:
            if r["method"] != meth:
                continue
            a = next((x for x in rows if x["method"] == "avatar" and x["view"] == r["view"] and x["role"] == r["role"]), None)
            if a is None:
                continue
            w["n"] += 1
            w["psnr"] += int(r["psnr"] > a["psnr"])
            if r.get("lpips_pod") is not None and a.get("lpips_pod") is not None:
                w["lpips"] += int(r["lpips_pod"] < a["lpips_pod"])
        wins[meth] = w
    out = {"what": __doc__.split("METRICS:")[1].strip(), "rows": rows, "summary": summ, "agreement_with_pod_max_abs": agree,
           "identity_swaps": swaps, "wins_vs_avatar": wins, "data_checks": data_checks(),
           "photo_effect": photo_effect(), "seconds": round(time.time() - t0, 1)}
    CHECK.mkdir(parents=True, exist_ok=True)
    (CHECK / "metrics_check.json").write_text(json.dumps(out, indent=1, default=float))
    (CHECK / "metrics_table.md").write_text(md_table(summ))
    log(md_table(summ))
    log(f"agreement with pod: {agree}; swaps: {swaps}; wins: {wins}")
    log(f"data checks: {json.dumps(out['data_checks'])}")
    log(f"photo effect: {json.dumps(out['photo_effect'])}")
    return out


def md_table(summ: dict) -> str:
    hdr = ("| method | dancer | n | PSNR | PSNR cc | SSIM | SSIM cc | LPIPS | LPIPS cc | IoU | coverage |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|\n")
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = []
    for meth, g in summ.items():
        for grp in ("all",) + ROLES:
            s = g[grp]
            who = {"all": "both", "lead": "John", "follow": "Lula"}[grp]
            lines.append(f"| {SHORT[meth]} | {who} | {s['n']} | {f(s['psnr'], 2)} | {f(s['psnr_cc'], 2)} | "
                         f"{f(s['ssim'])} | {f(s['ssim_cc'])} | {f(s['lpips_pod'])} | {f(s['lpips_cc_pod'])} | "
                         f"{f(s['iou'])} | {f(s['coverage'])} |")
    return hdr + "\n".join(lines) + "\n"


def data_checks() -> dict:
    """Frames vs cameras/01.npz, held-out / clip frames vs every input crop frame, refs counts."""
    with np.load(P.WORK / "cameras" / f"{P.CAM}.npz") as z:
        pts_cam = np.asarray(z["pts"], float)
    fj = load_json(P.WORK / "frames" / P.CAM / "frames.json")
    pts = np.array([f["pts"] for f in fj["frames"]], float)
    n_jpg = sum(1 for _ in (P.WORK / "frames" / P.CAM).glob("*.jpg"))
    refs = load_json(INPUT / "refs.json")
    prep = load_json(INPUT / "prep.json")
    crop_t = {r: [f["t"] for f in refs[r]["frames"]] for r in ROLES}
    all_t = crop_t["lead"] + crop_t["follow"]
    held = [{"id": h["id"], "t": h["t"], "min_dt_to_any_crop_s": round(min(abs(h["t"] - c) for c in all_t), 3)}
            for h in prep["heldout"]]
    clip = prep["clip"]
    clip_gap = min(min(abs(c - clip["t0"]), abs(c - clip["t1"])) if not (clip["t0"] <= c <= clip["t1"]) else 0.0
                   for c in all_t)
    nref = {r: {v: len(list((INPUT / "refs" / r / v).glob("*.png"))) for v in P.VARIANTS} for r in ROLES}
    cams = cams_all()
    ph = [c for c in cams if c["group"] == "clip_phone"]
    vr = [c for c in cams if c["group"] == "clip_vr_seat"]
    return {"frames": {"jpg_on_disk": n_jpg, "frames_json": len(pts), "camera_pts": len(pts_cam),
                       "max_abs_pts_diff_s": float(np.abs(pts - pts_cam).max()) if len(pts) == len(pts_cam) else None},
            "heldout_vs_crops": held, "clip_window_s": [clip["t0"], clip["t1"]],
            "clip_min_gap_to_crop_s": round(float(clip_gap), 3),
            "crop_frames": {r: [f["i"] for f in refs[r]["frames"]] for r in ROLES}, "refs_png_counts": nref,
            "clip_phone_frames_contiguous": [c["frame"] for c in ph] == list(range(ph[0]["frame"], ph[0]["frame"] + len(ph))),
            "clip_vr_grid_contiguous": [c["grid_k"] for c in vr] == list(range(vr[0]["grid_k"], vr[0]["grid_k"] + len(vr))),
            "clip_vr_dt_s": [round(float(np.min(np.diff([c["t"] for c in vr]))), 5),
                             round(float(np.max(np.diff([c["t"] for c in vr]))), 5)]}


def photo_effect() -> dict:
    """Per-Gaussian colour change video8 -> video8_photo / photofull in the canonical PLYs (LHM++ Gaussians are
    anchored to the same template points, index i = the same body point in every variant), by body region (A-pose
    height above the feet): how much the photo changed the head vs the body."""
    from .appearance_hybrid import read_gs
    out = {}
    for role in ROLES:
        g0 = read_gs(POD_OUT / "gs" / f"{role}_lhmpp_video8_cano.ply")
        y = g0["xyz"][:, 1]
        h = (y - y.min()) / (y.max() - y.min())
        reg = {"head (top 13 %)": h > 0.87, "torso / arms (55-87 %)": (h > 0.55) & (h <= 0.87),
               "legs (< 55 %)": h <= 0.55}
        out[role] = {}
        for v in ("lhmpp_video8_photo", "lhmpp_video8_photofull"):
            g1 = read_gs(POD_OUT / "gs" / f"{role}_{v}_cano.ply")
            if len(g1["rgb"]) != len(g0["rgb"]):
                out[role][v] = "different Gaussian counts"
                continue
            d = np.abs(g1["rgb"] - g0["rgb"]).mean(1) * 255
            dx = np.linalg.norm(g1["xyz"] - g0["xyz"], axis=1) * 1000
            out[role][v] = {k: {"mean_abs_rgb_change_0_255": round(float(d[m].mean()), 2),
                                "mean_xyz_change_mm": round(float(dx[m].mean()), 2), "n": int(m.sum())}
                            for k, m in reg.items()}
    return out


# ================================================================================================ held-out sheets
def _metric_lines(mrows, view: str, meth: str) -> list[str]:
    out = []
    for role in ROLES:
        r = next((x for x in mrows if x["view"] == view and x["method"] == meth and x["role"] == role), None)
        if r is None:
            continue
        s = f"{NAME[role]} {r['psnr']:.1f} dB"
        if r.get("lpips_pod") is not None:
            s += f"  LPIPS {r['lpips_pod']:.2f}"
        out.append(s)
    return out


def head_box(F, role: str, t: float, K, R, tvec, half_m: float = 0.19, W=None, H=None):
    """Square box (px) around the skull centre of a dancer, half_m metres half-size at its depth."""
    c = F[role].skull(t)
    uv, z = project(c, K, R, tvec)
    if z[0] <= 0.05:
        return None
    s = float(np.asarray(K)[1, 1] * half_m / z[0])
    u, v = uv[0]
    return [u - s, v - s * 1.05, u + s, v + s * 0.95]


def real_head_shift(real_mask: np.ndarray, own_render: np.ndarray, b) -> tuple[int, int]:
    """Native-pixel shift (dx, dy) that puts the real head box on the REAL head: the MAMMA fit's head can sit a few
    pixels off in the phone frame. Best overlap (IoU) of the dancer's real mask with the method's rendered own-role mask
    in a window around the fitted head (head + neck rows), shifts up to 0.45 x the box half-size, used only when it
    raises the IoU by >= 0.06 (else no shift)."""
    H, W = real_mask.shape
    s = (b[2] - b[0]) / 2
    u, v = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x0, x1 = int(max(0, u - 1.6 * s)), int(min(W, u + 1.6 * s))
    y0, y1 = int(max(0, v - 1.6 * s)), int(min(H, v + 1.2 * s))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0, 0
    R = own_render[y0:y1, x0:x1]
    best, lim = (-1.0, 0, 0), int(0.45 * s)
    step = max(1, lim // 10)
    M0 = real_mask[y0:y1, x0:x1]
    iou0 = float((M0 & R).sum() / max(1, (M0 | R).sum()))
    for dy in range(-lim, lim + 1, step):
        for dx in range(-lim, lim + 1, step):
            ya, yb, xa, xb = y0 + dy, y1 + dy, x0 + dx, x1 + dx
            if ya < 0 or xa < 0 or yb > H or xb > W:
                continue
            Mm = real_mask[ya:yb, xa:xb]
            iou = float((Mm & R).sum() / max(1, (Mm | R).sum()))
            if iou > best[0] + 1e-9 or (abs(iou - best[0]) < 1e-9 and abs(dx) + abs(dy) < abs(best[1]) + abs(best[2])):
                best = (iou, dx, dy)
    if best[0] - iou0 < 0.06:          # no clear gain (blur, occluded head, a mask that misses the head): keep the fit
        return 0, 0
    if max(abs(best[1]), abs(best[2])) > lim - step:   # optimum on the search border: not a real head match
        return 0, 0
    return best[1], best[2]


def crop_box_px(img: np.ndarray, b, out: int) -> np.ndarray:
    """Crop a (possibly out-of-image) float box with grey padding and resize it to out x out."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), constant_values=GREY)
        x0, x1, y0, y1 = x0 + pad, x1 + pad, y0 + pad, y1 + pad
    return resize(img[y0:y1, x0:x1], out, out)


def heldout_sheets() -> list[Path]:
    cams = {c["id"]: c for c in cams_all()}
    prep = load_json(INPUT / "prep.json")
    mrows = load_json(CHECK / "metrics_check.json")["rows"] if (CHECK / "metrics_check.json").exists() else []
    F = fits()
    Mk = P.Masks()
    tw = (MAXW - 3 * 4) // 4           # 4 columns
    hw = (MAXW - 7 * 4) // 8           # 8 head tiles
    blocks = []
    for h in prep["heldout"]:
        vid, i, t = h["id"], h["i"], h["t"]
        cz, cn = cams[f"{vid}_zoom"], cams[vid]
        x0, y0, x1, y1 = cz["box"]
        th = int(round(cz["H"] * tw / cz["W"]))
        real = read_frame(i)
        up = tw / (x1 - x0)
        tiles = [put(resize(real[y0:y1, x0:x1], tw, th), TITLE["real"], (8, 6), 19)]
        put(tiles[0], f"phone 538x960, crop {'upscaled' if up > 1 else 'scaled'} x{up:.2f}"
                      f"{' (Lanczos)' if up > 1 else ''}", (8, 30), 15, False)
        for m in SBS:
            im = on_grey(RENDERS / m / cz["out"])
            t_ = put(resize(im, tw, th), TITLE[m], (8, 6), 19)
            for n_, ln in enumerate(reversed(_metric_lines(mrows, vid, m))):
                put(t_, ln, (8, th - 26 - 24 * n_), 17)
            tiles.append(t_)
        row = hcat(tiles)
        # heads: John then Lula, real + 3 methods each (box from the fit's skull centre, 0.38 m square)
        heads = []
        rid = cv2.imread(str(RENDERS / "lhmpp_video8" / "role" / f"{vid}.png"), cv2.IMREAD_COLOR)[:, :, ::-1]
        for role in ROLES:
            bn = head_box(F, role, cn["t"], cn["K"], cn["R"], cn["tvec"])
            bz = head_box(F, role, cz["t"], cz["K"], cz["R"], cz["tvec"])
            if bn is None or bz is None:
                continue
            dx, dy = real_head_shift(Mk.get(i, role), rid[:, :, ROLES.index(role)] > 127, bn)
            h.setdefault("real_head_shift_px", {})[role] = [dx, dy]
            ht = crop_box_px(real, [bn[0] + dx, bn[1] + dy, bn[2] + dx, bn[3] + dy], hw)
            put(ht, f"{NAME[role]}: real", (6, 4), 16)
            put(ht, f"{bn[2] - bn[0]:.0f} px in the phone, upscaled", (6, hw - 22), 13, False)
            heads.append(ht)
            for m in SBS:
                im = on_grey(RENDERS / m / cz["out"])
                ht = crop_box_px(im, bz, hw)
                put(ht, f"{NAME[role]}: {SHORT[m]}", (6, 4), 15)
                heads.append(ht)
        hrow = hcat(heads)
        near = h.get("nearest_crop_s")
        hdr = banner([f"Held-out phone frame {i} (t = {t:.2f} s): the nearest LHM++ input crop is {near:.1f} s away",
                      "Top: the same crop around both dancers in every column; renders re-rendered at 1080 px for this "
                      "crop, real = the phone's pixels resampled with Lanczos.",
                      "Bottom: heads in 0.38 m boxes around each dancer's fitted head (the real box nudged onto the real "
                      "head when the mask says so), upscaled with Lanczos."], row.shape[1], 21)
        blocks.append(vcat([hdr, row, hrow], 3))
        log(f"{vid}: real head box shift (native px, fit -> real head) {h.get('real_head_shift_px')}")
    out = []
    n_sheets = math.ceil(len(blocks) / 2)
    for k in range(n_sheets):
        body = vcat(blocks[2 * k:2 * k + 2] + [banner(FOOT, MAXW, 15, False)], 10)
        out.append(save_jpg(body, SHEETS / f"sbs_heldout_{k + 1}_of_{n_sheets}.jpg"))
    log(f"held-out sheets: {[p.name for p in out]} {[cv2.imread(str(p)).shape for p in out]}")
    return out


# ================================================================================================ turntable sheet
def turntable_sheet() -> Path:
    cams = [c for c in cams_all() if c["group"] == "turntable"]
    prep = load_json(INPUT / "prep.json")
    union = None
    imgs = {}
    for m in METHODS:
        for c in cams:
            ra = read_rgba(RENDERS / m / c["out"])
            imgs[(m, c["id"])] = ra
            ys, xs = np.nonzero(ra[1] > 0.02)
            b = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            union = b if union is None else [min(union[0], b[0]), min(union[1], b[1]), max(union[2], b[2]),
                                             max(union[3], b[3])]
    x0, y0, x1, y1 = union
    p = 10
    x0, y0, x1, y1 = max(0, x0 - p), max(0, y0 - p), min(1080, x1 + p), min(1080, y1 + p)
    tw = (MAXW - 11 * 3) // 12
    th = int(round((y1 - y0) * tw / (x1 - x0)))
    rows = []
    for m in METHODS:
        tiles = []
        for c in cams:
            rgb, a = imgs[(m, c["id"])]
            im = (np.clip(rgb * a[..., None] + GREY / 255 * (1 - a[..., None]), 0, 1) * 255 + 0.5).astype(np.uint8)
            t_ = resize(im[y0:y1, x0:x1], tw, th)
            put(t_, f"{c['azimuth_from_phone_deg']:.0f}\u00b0", (4, 3), 14)
            tiles.append(t_)
        r = hcat(tiles, 3)
        rows += [banner(TITLE[m] + ("   (diagnostic variant)" if m.endswith("photofull") else ""), r.shape[1], 18), r]
    tt = prep["novel"]["turntable"]
    body = vcat([banner([f"Turntable at t = {tt['t']:.2f} s (phone frame {tt['frame']}, inside the spin clip): 12 "
                         f"cameras 30\u00b0 apart, 3 m from the couple, 1.6 m high; 0\u00b0 = the phone's side",
                         "Only the 0\u00b0 column is near a real camera; every other angle is unseen at this instant "
                         "(LHM++ fills it from its 8 / 9 input crops)."], tw * 12 + 33, 20)] + rows +
                [banner(FOOT, tw * 12 + 33, 14, False)], 3)
    p_ = save_jpg(body, SHEETS / "turntable_12az.jpg")
    log(f"turntable -> {p_} {body.shape} (crop {x0, y0, x1, y1}, tile {tw}x{th})")
    return p_


# ================================================================================================ close-ups (GPU)
def closeup_cams(F, t: float) -> list[dict]:
    """Face / back close-up cameras at the turntable instant from the fit's skull centre and head rotation."""
    from . import appearance_compare as AC
    out = []

    def add(kind, role, C, target, field_m, W, H):
        d = float(np.linalg.norm(np.asarray(target) - np.asarray(C)))
        f = H * d / field_m
        c = AC.look_at(C, target, f, W, H)
        out.append({"kind": kind, "role": role, "K": np.asarray(c["K"], float), "R": np.asarray(c["R"], float),
                    "tvec": np.asarray(c["tvec"], float).reshape(3), "W": W, "H": H, "C": np.asarray(C, float),
                    "target": np.asarray(target, float), "field_m": field_m})

    for role in ROLES:
        sk = F[role].skull(t)
        Rh = F[role].head_rot(t)
        fwd = Rh @ np.array([0.0, 0.0, 1.0])
        fh = np.array([fwd[0], 0.0, fwd[2]])
        fh /= np.linalg.norm(fh)
        face = sk + Rh @ np.array([0.0, -0.05, 0.03])
        if role == "lead":
            add("John - face, front", role, face + 0.85 * fh, face, 0.36, 600, 600)
            a = math.radians(40)
            rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            add("John - face, 3/4", role, face + 0.85 * (rot @ fh), face, 0.36, 600, 600)
        else:
            add("Lula - face, front", role, face + 0.85 * fh, face, 0.36, 600, 600)
            tgt = sk + np.array([0.0, -0.22, 0.0]) - 0.04 * fh
            add("Lula - ponytail + back", role, tgt - 1.3 * fh + np.array([0.0, 0.12, 0.0]), tgt, 0.75, 600, 750)
    return out


def _gs_render(G: dict, c: dict, sl: slice, torch, rasterization):
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device="cuda")  # noqa: E731
    vm = torch.eye(4, device="cuda")
    vm[:3, :3] = T(c["R"])
    vm[:3, 3] = T(np.asarray(c["tvec"]).reshape(3))
    K = T(c["K"]).clone()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    with torch.no_grad():
        img, alpha, _ = rasterization(means=T(G["xyz"][sl]), quats=T(G["quats"][sl]), scales=T(G["scales"][sl]),
                                      opacities=T(G["opacity"][sl]), colors=T(G["rgb"][sl]), viewmats=vm[None],
                                      Ks=K[None], width=int(c["W"]), height=int(c["H"]), near_plane=0.01,
                                      far_plane=100.0, eps2d=0.3, render_mode="RGB",
                                      camera_model="pinhole",
                                      rasterize_mode="classic")
    return img[0].clamp(0, 1).cpu().numpy(), alpha[0, :, :, 0].clamp(0, 1).cpu().numpy()


def closeups() -> Path:
    """gsplat renders of the pod's posed scene PLYs (both dancers at the turntable instant, world frame; lead = first
    half) with the pod's settings, validated against the pod's own turntable renders; the current avatar with the same
    cameras (AvatarRenderer, the other dancer moved out of view)."""
    import torch
    from gsplat import rasterization

    from .. import gpu
    from .appearance_avatar_ags import AvatarRenderer
    from .appearance_hybrid import read_gs
    prep = load_json(INPUT / "prep.json")
    tt = prep["novel"]["turntable"]
    t = float(tt["t"])
    key = f"f{tt['frame']}"
    F = fits()
    ccs = closeup_cams(F, t)
    tt_cams = [c for c in cams_all() if c["group"] == "turntable"]
    res, valid = {}, {}
    CHECK.mkdir(parents=True, exist_ok=True)
    with gpu.exclusive("jl_owner_closeups", timeout_s=1800):
        for v in P.VARIANTS:
            G = read_gs(POD_OUT / "gs" / f"scene_{v}_{key}.ply")
            n = len(G["xyz"])
            nl = n // 2
            # validation: the pod's tt00 / tt03 / tt06 from the same scene with the pod's settings
            vv = []
            for c in [tt_cams[0], tt_cams[3], tt_cams[6]]:
                rgb, a = _gs_render(G, {**c, "K": np.array(c["K"])}, slice(None), torch, rasterization)
                ref = read_rgba(RENDERS / v / c["out"])
                pre_ref = ref[0] * ref[1][..., None]
                mse = float(((rgb - pre_ref) ** 2).mean())
                vv.append({"view": c["id"], "psnr_vs_pod_render_db": round(10 * math.log10(1 / max(mse, 1e-12)), 2),
                           "alpha_max_abs_diff": round(float(np.abs(a - ref[1]).max()), 4)})
            valid[v] = {"gaussians": n, "views": vv}
            log(f"{v}: local gsplat vs pod turntable renders: {vv}")
            for c in ccs:
                sl = slice(0, nl) if c["role"] == "lead" else slice(nl, n)
                rgb, a = _gs_render(G, c, sl, torch, rasterization)
                res[(v, c["kind"])] = (rgb / np.maximum(a, 1e-6)[..., None], a)
            del G
            torch.cuda.empty_cache()
        ar = AvatarRenderer(P.TAKE, device="cuda", ss=2, capture="08_JohnLula")
        base_verts = ar.verts
        for c in ccs:
            hide = "follow" if c["role"] == "lead" else "lead"
            ar.verts = (lambda r, tt_, h=hide: base_verts(r, tt_) + (np.array([0.0, 500.0, 0.0]) if r == h else 0.0))
            rgb, a, _, _ = ar.render({"cam": {"K": c["K"], "R": c["R"], "tvec": c["tvec"], "W": c["W"], "H": c["H"]},
                                      "t": t})
            res[("avatar", c["kind"])] = (rgb, a)
        ar.verts = base_verts
    # save the raw close-ups (RGBA) for the record
    for (m, kind), (rgb, a) in res.items():
        p = CHECK / "closeups" / f"{m}__{kind.replace(' ', '').replace(',', '_').replace('/', '-').replace('+', '_')}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        rgba = np.concatenate([np.clip(rgb, 0, 1)[..., ::-1], a[..., None]], -1)
        cv2.imwrite(str(p), (rgba * 255 + 0.5).astype(np.uint8))
    (CHECK / "closeups_validation.json").write_text(json.dumps(
        {"t": t, "frame": tt["frame"], "validation_vs_pod": valid,
         "cams": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in ccs]}, indent=1))
    return closeup_sheet(res, ccs, t, tt["frame"], valid)


def _ref_tiles(ccs) -> dict:
    """Reference images (NOT renders): the photo (an INPUT of the +photo variants) for John's face and Lula's back; a
    phone frame for the other two rows (the most frontal large sharp view of that dancer in the survey)."""
    photo = cv2.imread(str(P.PHOTO), cv2.IMREAD_COLOR)[:, :, ::-1]
    seg = load_json(OUT / "photo" / "seg.json")
    out = {}
    lb = seg["lead"]["box"]          # John's mask box in the photo
    lm = cv2.imread(str(OUT / "photo" / "lead_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    ys, xs = np.nonzero(lm[: P.photo_head_cut(lm)])
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    s = (ys.max() - ys.min()) * 0.62
    out["John - face, front"] = (crop_box_px(photo, [cx - s, cy - s, cx + s, cy + s], 400),
                                 "photo GM6B6189 (INPUT of +photo)")
    fb = seg["follow"]["box"]
    w = fb[2] - fb[0]
    out["Lula - ponytail + back"] = (photo[fb[1]:fb[1] + int(w * 1.25), fb[0]:fb[2]], "photo GM6B6189 (INPUT of +photo)")
    del lb
    # phone references, picked by eye from the sheets (John's face is rarely seen from the front by the phone):
    # John 3/4 = held-out frame 2438 (beard visible), Lula's face = frame 578 (0.13 s from her input crop f582)
    F = fits()
    from .. import mamma_dyn as md
    tr = md.load_tracks(P.WORK, [P.CAM])[P.CAM]
    for kind, role, i, note in (("John - face, 3/4", "lead", 2438, "held out, his profile"),
                                ("Lula - face, front", "follow", 578, "near her input crop f582")):
        b_ = head_box(F, role, float(tr.t[i]), tr.K[i], tr.R[i], tr.tvec[i])
        if b_ is None:
            continue
        out[kind] = (crop_box_px(read_frame(i), b_, 400), f"phone frame {i}, {note} ({b_[2] - b_[0]:.0f} px)")
    return out


def closeup_sheet(res: dict, ccs, t: float, frame: int, valid: dict) -> Path:
    meths = METHODS
    refs = _ref_tiles(ccs)
    tw = (MAXW - 4 * 4) // 5
    rows = []
    for c in ccs:
        th = int(round(c["H"] * tw / c["W"]))
        tiles = []
        r_ = refs.get(c["kind"])
        if r_ is not None:
            im, lab = r_
            sc = min(tw / im.shape[1], th / im.shape[0])
            im2 = resize(im, int(im.shape[1] * sc), int(im.shape[0] * sc))
            canvas = np.full((th, tw, 3), 255, np.uint8)
            oy, ox = (th - im2.shape[0]) // 2, (tw - im2.shape[1]) // 2
            canvas[oy:oy + im2.shape[0], ox:ox + im2.shape[1]] = im2
            put(canvas, "REFERENCE (not a render)", (6, 4), 15)
            put(canvas, lab, (6, th - 22), 13, False)
            tiles.append(canvas)
        else:
            tiles.append(np.full((th, tw, 3), 255, np.uint8))
        for m in meths:
            rgb, a = res[(m, c["kind"])]
            im = (np.clip(rgb * a[..., None] + GREY / 255 * (1 - a[..., None]), 0, 1) * 255 + 0.5).astype(np.uint8)
            t_ = resize(im, tw, th)
            put(t_, SHORT[m], (6, 4), 16)
            tiles.append(t_)
        r = hcat(tiles)
        rows += [banner(f"{c['kind']}  ({c['field_m']:.2f} m tall view, camera {np.linalg.norm(c['target'] - c['C']):.2f} m "
                        "away; the partner is hidden)", r.shape[1], 18), r]
    vmin = min(x["psnr_vs_pod_render_db"] for v in valid.values() for x in v["views"])
    body = vcat([banner([f"Face / hair close-ups at t = {t:.2f} s (phone frame {frame}), posed by the MAMMA fit; "
                         "novel close cameras (no real camera this close)",
                         f"LHM++: the pod's posed scene PLYs re-rendered locally with gsplat and the pod's settings "
                         f"(matches the pod's own turntable renders to >= {vmin:.0f} dB); avatar: the same cameras."],
                        tw * 5 + 16, 20)] + rows + [banner(FOOT, tw * 5 + 16, 14, False)], 3)
    p = save_jpg(body, SHEETS / "closeups_face_hair.jpg")
    log(f"close-ups -> {p} {body.shape}")
    return p


# ================================================================================================ videos
def _ffmpeg_writer(out: Path, W: int, H: int, fps: float):
    out.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", f"{fps}", str(out)],
                            stdin=subprocess.PIPE)


def _even(x: float) -> int:
    return int(round(x / 2)) * 2


def videos() -> dict:
    cams = cams_all()
    prep = load_json(INPUT / "prep.json")
    ph = [c for c in cams if c["group"] == "clip_phone"]
    vr = [c for c in cams if c["group"] == "clip_vr_seat"]
    info = {}
    # ---- (a) the phone's own path: real | avatar | video8 | video8+photo
    cb = P.clip_box(ph)
    x0, y0, x1, y1 = cb
    pw = MAXW // 4
    phh = _even((y1 - y0) * pw / (x1 - x0))
    hdr_h = 0
    W, H = pw * 4, phh + hdr_h
    out_a = VIDEOS / "jl_phone_path.mp4"
    keep = {}
    wr = _ffmpeg_writer(out_a, W, H, 30.0)
    stats = {m: [] for m in SBS}
    for j, c in enumerate(ph):
        tiles = []
        real = read_frame(c["frame"])[y0:y1, x0:x1]
        t_ = resize(real, pw, phh)
        put(t_, "REAL phone (held out)", (8, 6), 20)
        put(t_, f"t = {c['t']:.2f} s   frame {c['frame']}", (8, phh - 30), 17, False)
        tiles.append(t_)
        for m in SBS:
            ra = read_rgba(RENDERS / m / c["out"])
            stats[m].append(_alpha_stats(ra[1]))
            im = (np.clip(ra[0] * ra[1][..., None] + GREY / 255 * (1 - ra[1][..., None]), 0, 1) * 255 + .5).astype(np.uint8)
            t_ = resize(im[y0:y1, x0:x1], pw, phh)
            put(t_, SHORT[m], (8, 6), 20)
            tiles.append(t_)
        fr = np.concatenate(tiles, 1)
        if j in (0, 60, 119):
            keep[j] = fr.copy()
        wr.stdin.write(np.ascontiguousarray(fr).tobytes())
    wr.stdin.close()
    wr.wait()
    info["jl_phone_path"] = {"file": str(out_a), "frames_written": len(ph), "size": [W, H], "fps": 30.0,
                             "crop_native_px": cb, "panel_upscale": round(pw / (x1 - x0), 3),
                             "decode": decode_check(out_a, keep), "smoothness": smooth(stats),
                             "real_frames": [ph[0]["frame"], ph[-1]["frame"]], "t": [ph[0]["t"], ph[-1]["t"]]}
    # ---- (b) the VR seat: avatar | video8 | video8+photo
    union = None
    for m in SBS:
        for c in vr[::5] + [vr[-1]]:
            a = read_rgba(RENDERS / m / c["out"])[1]
            ys, xs = np.nonzero(a > 0.02)
            b = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            union = b if union is None else [min(union[0], b[0]), min(union[1], b[1]), max(union[2], b[2]), max(union[3], b[3])]
    pad = 24
    bx0, by0, bx1, by1 = max(0, union[0] - pad), max(0, union[1] - pad), min(1080, union[2] + pad), min(1080, union[3] + pad)
    pw2 = MAXW // 3
    ph2 = _even((by1 - by0) * pw2 / (bx1 - bx0))
    if ph2 > 1080:
        ph2 = 1080
        pw2 = _even((bx1 - bx0) * ph2 / (by1 - by0))
    W2, H2 = pw2 * 3, ph2
    out_b = VIDEOS / "jl_vr_seat.mp4"
    keep = {}
    wr = _ffmpeg_writer(out_b, W2, H2, 30.0)
    stats = {m: [] for m in SBS}
    seat = prep["novel"]["vr_seat"]
    for j, c in enumerate(vr):
        tiles = []
        for m in SBS:
            ra = read_rgba(RENDERS / m / c["out"])
            stats[m].append(_alpha_stats(ra[1]))
            im = (np.clip(ra[0] * ra[1][..., None] + GREY / 255 * (1 - ra[1][..., None]), 0, 1) * 255 + .5).astype(np.uint8)
            t_ = resize(im[by0:by1, bx0:bx1], pw2, ph2)
            put(t_, SHORT[m], (8, 6), 22)
            if m == SBS[0]:
                put(t_, f"VR seat {seat['dist_m']:.1f} m away, {seat['height_m']:.1f} m high, opposite the phone "
                        f"(no real camera here)", (8, 34), 15, False)
            put(t_, f"t = {c['t']:.2f} s", (8, ph2 - 30), 17, False)
            tiles.append(t_)
        fr = np.concatenate(tiles, 1)
        if j in (0, 60, 119):
            keep[j] = fr.copy()
        wr.stdin.write(np.ascontiguousarray(fr).tobytes())
    wr.stdin.close()
    wr.wait()
    info["jl_vr_seat"] = {"file": str(out_b), "frames_written": len(vr), "size": [W2, H2], "fps": 30.0,
                          "crop_px_of_1080": [bx0, by0, bx1, by1], "decode": decode_check(out_b, keep),
                          "smoothness": smooth(stats), "grid_k": [vr[0]["grid_k"], vr[-1]["grid_k"]],
                          "t": [vr[0]["t"], vr[-1]["t"]]}
    CHECK.mkdir(parents=True, exist_ok=True)
    (CHECK / "videos_check.json").write_text(json.dumps(info, indent=1, default=float))
    log(json.dumps(info, indent=1, default=float)[:4000])
    return info


def _alpha_stats(a: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(a > 0.5)
    if len(xs) == 0:
        return [float("nan")] * 3
    return [float(xs.mean()), float(ys.mean()), float(len(xs))]


def smooth(stats: dict) -> dict:
    """Frame-to-frame motion of the rendered couple's silhouette centroid (px) and area change: a frame jump shows as a
    step far above the clip's typical step."""
    out = {}
    for m, s in stats.items():
        s = np.asarray(s, float)
        d = np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1)
        da = np.abs(np.diff(s[:, 2])) / np.maximum(s[:-1, 2], 1)
        med = float(np.median(d))
        out[m] = {"centroid_step_px_median": round(med, 2), "centroid_step_px_max": round(float(d.max()), 2),
                  "max_step_at": int(np.argmax(d)), "area_change_max": round(float(da.max()), 3),
                  "steps_over_4x_median_and_8px": int(((d > 4 * med) & (d > 8)).sum())}
    return out


def decode_check(p: Path, keep: dict) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt,duration", "-of", "json",
                        str(p)], capture_output=True, text=True)
    st = json.loads(r.stdout)["streams"][0] if r.returncode == 0 else {"error": r.stderr[-300:]}
    cap = cv2.VideoCapture(str(p))
    n, match = 0, {}
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n in keep:
            ref = keep[n].astype(np.float32)
            got = fr[:, :, ::-1].astype(np.float32)
            mse = float(((ref - got) ** 2).mean())
            match[n] = round(10 * math.log10(255 ** 2 / max(mse, 1e-6)), 2)
        n += 1
    cap.release()
    return {"ffprobe": st, "cv2_frames_decoded": n, "psnr_decoded_vs_composed_db": match,
            "mb": round(p.stat().st_size / 1e6, 2)}


# ================================================================================================ inputs sheet
def inputs_sheet() -> Path:
    im = cv2.imread(str(OUT / "inputs.jpg"))[:, :, ::-1]
    W = im.shape[1]
    s = min(1.0, MAXW / W)
    if s < 1:
        im = resize(im, int(W * s), int(im.shape[0] * s))
    body = vcat([banner(["LHM++ inputs (the build's inputs.jpg): 8 crops per dancer from the ONE phone video (whole take, "
                         "occluded / white-patch frames dropped) + the photo crop",
                         "John's photo crop = head + shoulders (first; video8+photo); 'PHOTO full' = his whole visible "
                         "mask (diagnostic variant). Lula's photo crop = her back (last). Labels: frame, time, yaw to "
                         "the phone; w = white fraction, q = quality."], im.shape[1], 18), im], 3)
    p = save_jpg(body, SHEETS / "inputs_crops.jpg")
    log(f"inputs -> {p} {body.shape}")
    return p


def extra_checks() -> dict:
    """Clip silhouettes vs the real masks, the fit's frame-to-frame jumps in the clip, white Gaussians by body region
    (bibs vs occluder patches), LHM++'s run-to-run noise (Lula's photo and photofull inputs are identical)."""
    from .appearance_hybrid import read_gs
    cams = cams_all()
    M = P.Masks()
    out = {"clip_phone_union_iou": {}}
    for m in SBS:
        ious = []
        for c in [c for c in cams if c["group"] == "clip_phone"]:
            a = read_rgba(RENDERS / m / c["out"])[1] > 0.5
            u = np.zeros_like(a)
            for r in ROLES:
                mk = M.get(c["frame"], r)
                if mk is not None:
                    u |= mk
            ious.append(float((a & u).sum() / max(1, (a | u).sum())))
        ious = np.array(ious)
        out["clip_phone_union_iou"][m] = {"median": round(float(np.median(ious)), 3), "min": round(float(ious.min()), 3),
                                          "min_at_clip_frame": int(ious.argmin())}
    prep = load_json(INPUT / "prep.json")
    k0, k1 = int(math.ceil(prep["clip"]["t0"] * 30)), int(math.ceil(prep["clip"]["t0"] * 30)) + 119
    out["fit_clip_jumps"] = {}
    for r in ROLES:
        with np.load(P.WORK / "smplx" / f"{r}.npz", allow_pickle=True) as z:
            J, t, rp = np.asarray(z["joints"], float), np.asarray(z["t"], float), np.asarray(z["reproj_px"], float)
        pel = 1e3 * np.linalg.norm(np.diff(J[:, 0], axis=0), axis=1)
        w = pel[k0:k1]
        k = k0 + int(np.argmax(w))
        out["fit_clip_jumps"][r] = {"pelvis_step_mm_median_clip": round(float(np.median(w)), 1),
                                    "pelvis_step_mm_max_clip": round(float(w.max()), 1), "at_grid_k": k,
                                    "t": [round(float(t[k]), 3), round(float(t[k + 1]), 3)],
                                    "reproj_px_around": np.round(rp[k - 3:k + 4], 1).tolist(),
                                    "take_pelvis_step_mm_median_p95": [round(float(np.median(pel)), 1),
                                                                       round(float(np.percentile(pel, 95)), 1)]}
    out["white_gaussians"] = {}
    for role in ROLES:
        for v in P.VARIANTS:
            g = read_gs(POD_OUT / "gs" / f"{role}_{v}_cano.ply")
            y, z = g["xyz"][:, 1], g["xyz"][:, 2]
            c = g["rgb"]
            w = (c.min(1) > 0.75) & ((c.max(1) - c.min(1)) < 0.15) & (g["opacity"] > 0.3)
            h = (y - y.min()) / (y.max() - y.min())
            front = z > np.median(z[(h > 0.55) & (h < 0.85)])
            out["white_gaussians"][f"{role}/{v}"] = {
                "total": int(w.sum()), "torso_back": int((w & (h > 0.55) & (h <= 0.87) & ~front).sum()),
                "torso_front": int((w & (h > 0.55) & (h <= 0.87) & front).sum()), "head": int((w & (h > 0.87)).sum()),
                "hips_legs_back": int((w & (h <= 0.55) & ~front).sum()), "hips_legs_front": int((w & (h <= 0.55) & front).sum())}
    a = read_gs(POD_OUT / "gs" / "follow_lhmpp_video8_photo_cano.ply")
    b = read_gs(POD_OUT / "gs" / "follow_lhmpp_video8_photofull_cano.ply")
    out["lhmpp_noise_floor_identical_inputs"] = {
        "what": "Lula lhmpp_video8_photo vs lhmpp_video8_photofull (byte-identical refs)",
        "mean_abs_rgb_0_255": round(float(np.abs(a["rgb"] - b["rgb"]).mean() * 255), 3),
        "max_abs_rgb_0_255": round(float(np.abs(a["rgb"] - b["rgb"]).max() * 255), 2),
        "mean_xyz_mm": round(float(np.linalg.norm(a["xyz"] - b["xyz"], axis=1).mean() * 1000), 3)}
    CHECK.mkdir(parents=True, exist_ok=True)
    (CHECK / "extra_checks.json").write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps(out, indent=1, default=float))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.jl_lhmpp_owner_sheets")
    ap.add_argument("cmd", choices=["metrics", "heldout", "turntable", "closeups", "videos", "inputs", "extra"])
    a = ap.parse_args(argv)
    if a.cmd == "closeups" and P_needs_cuda_env():
        import os
        import sys

        from ..env import cuda_wrapper
        r = subprocess.run(cuda_wrapper(["-m", "dancecap.mono4d.jl_lhmpp_owner_sheets", "closeups"]),
                           cwd=P.PACKAGE_ROOT, env={**os.environ, "DANCECAP_CUDA_ENV": "1", "PYTHONUNBUFFERED": "1"})
        del sys
        return r.returncode
    {"metrics": metrics, "heldout": heldout_sheets, "turntable": turntable_sheet, "closeups": closeups,
     "videos": videos, "inputs": inputs_sheet, "extra": extra_checks}[a.cmd]()
    return 0


def P_needs_cuda_env() -> bool:  # noqa: N802
    from .appearance_hybrid import _needs_cuda_env
    return _needs_cuda_env()


if __name__ == "__main__":
    raise SystemExit(main())
