"""Local review sheets of a trained AGS model (after the pod run; the shared 2080 Ti or the CPU, a few dozen renders):

  review/splats/sheet_cam03_heldout.jpg   video | textured avatar | splats at the 6 review moments (incl. her three
  review/splats/sheet_cam02_train.jpg     fastest head turns 59.18 / 60.88 / 61.41 s), held-out camera 03 and training
                                          camera 02 (59.94 fps, face detail)
  review/splats/turntable_t<t>.jpg        12 novel views around the couple at one moment (splats; avatar row below)
  review/splats/hair_closeups.jpg         the follow's head + hair from cameras 02 / 06 / 09 in the three turns
  review/splats/face_closeups.jpg         both faces from cameras 02 / 06 / 09 in the three turns
  review/splats/temporal_strip_cam03.jpg  12 consecutive frames through the 59.18 s turn on the held-out camera
  review/splats/temporal_strip_novel.jpg  the same frames from the fixed novel camera + the frame-to-frame change x4

    python -m dancecap.dancer_splats <take> review
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from . import ROLES
from .data import REVIEW_TIMES, at_time, decode_labels, load_bundle, read_albedo, read_view, render_avatar
from .evaluate import FACE_HALF_M, HAIR_HALF_M, _tile, _u8, crop_box, head_points, image_metrics, render_view
from .model import AGSModel

TURNS = (59.18, 60.88, 61.41)


def _font(img, text, org=(4, 16), s=0.45):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _pad_w(rows: list[np.ndarray]) -> list[np.ndarray]:
    W = max(r.shape[1] for r in rows)
    return [np.hstack([r, np.zeros((r.shape[0], W - r.shape[1], 3), np.uint8)]) if r.shape[1] < W else r for r in rows]


def look_at(C, target, f: float, W: int, H: int) -> dict:
    """A pinhole camera at C looking at target (world Y up; image x right, y down), principal point centred."""
    C, target = np.asarray(C, float), np.asarray(target, float)
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    return {"K": K.tolist(), "R": R.tolist(), "t_w": (-R @ C).tolist(), "W": int(W), "H": int(H)}


class Reviewer:
    def __init__(self, bundle: Path, out: Path, take: str, device: str = "cuda"):
        import torch
        self.torch = torch
        self.B = load_bundle(bundle)
        dev = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.model = AGSModel.load(self.B, out / "model.pt", dev)
        self.by_cam = self.model.cam_app_index()
        self.sh = self.model.cfg.sh_degree
        bd = self.B["bodies"]
        self.tg = np.asarray(bd["t_grid"], float)
        d = self.model.dev
        self.faces = torch.as_tensor(bd["faces"], dtype=torch.long, device=d)
        self.ft = torch.as_tensor(bd["ft"], dtype=torch.long, device=d)
        self.vt = torch.as_tensor(bd["vt"], dtype=torch.float32, device=d)
        from ..take import PACKAGE_ROOT
        av = PACKAGE_ROOT / "work" / take / "avatar"
        alb = {r: read_albedo(av / f"{r}_albedo.png") for r in ROLES}
        self.alb = None if any(a is None for a in alb.values()) else {
            r: torch.as_tensor(a.transpose(2, 0, 1).copy(), device=d) for r, a in alb.items()}

    def splat(self, v: dict) -> np.ndarray:
        return render_view(self.model, dict(v), self.by_cam, self.sh)["rgb"]

    def avatar(self, v: dict) -> np.ndarray | None:
        if self.alb is None:
            return None
        pv = {r: at_time(self.B["bodies"][f"{r}_verts"], self.tg, v["t"]) for r in ROLES}
        img, _ = render_avatar(self.torch, pv, self.faces, self.ft, self.vt, self.alb, v["K"], v["R"], v["t_w"],
                               v["H"], v["W"])
        return img

    def view_at(self, cam: str, t: float, tol: float = 0.02) -> dict | None:
        c = [v for v in self.B["views"] if v["cam"] == cam]
        if not c:
            return None
        v = min(c, key=lambda x: abs(x["t"] - t))
        return v if abs(v["t"] - t) <= tol else None

    # ------------------------------------------------------------------------------------------------ sheets
    def side_by_side(self, cam: str, review: Path, tag: str, h: int = 380) -> tuple[str | None, list[dict]]:
        rows, scores = [], []
        for t in REVIEW_TIMES:
            v = self.view_at(cam, t)
            if v is None:
                continue
            img, L = read_view(self.B["dir"], v)
            gt = img.astype(np.float32) / 255
            m = decode_labels(L)
            sel = m["fg"] & m["valid"]
            sp = self.splat(v)
            av = self.avatar(v)
            s_sp = image_metrics(sp, gt, sel)
            s_av = image_metrics(av, gt, sel) if av is not None else {}
            scores.append({"cam": cam, "t": v["t"], "split": v["split"], "splat_psnr_cc": s_sp.get("psnr_cc"),
                           "splat_ssim_cc": s_sp.get("ssim_cc"), "avatar_psnr_cc": s_av.get("psnr_cc"),
                           "avatar_ssim_cc": s_av.get("ssim_cc")})
            turn = " (head turn)" if any(abs(v["t"] - x) < 0.04 for x in TURNS) else ""
            tiles = [_tile(gt, h, f"video cam {cam} t {v['t']:.2f}{turn} [{v['split']}]"),
                     _tile(av if av is not None else np.zeros_like(gt), h,
                           f"textured avatar psnr_cc {s_av.get('psnr_cc', 0):.1f}"),
                     _tile(sp, h, f"splats psnr_cc {s_sp.get('psnr_cc', 0):.1f} ssim_cc {s_sp.get('ssim_cc', 0):.3f}")]
            rows.append(np.hstack(tiles))
        if not rows:
            return None, scores
        p = review / f"sheet_cam{cam}_{tag}.jpg"
        cv2.imwrite(str(p), np.vstack(_pad_w(rows)), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return p.name, scores

    def turntable(self, t: float, review: Path, n: int = 12, size: tuple[int, int] = (360, 480)) -> str:
        bd = self.B["bodies"]
        pel = np.mean([at_time(bd[f"{r}_Jp"][:, 0], self.tg, t) for r in ROLES], 0)
        target = np.array([pel[0], 0.95, pel[2]])
        v03 = (self.view_at("03", t, 0.05) or next((v for v in self.B["views"] if v["cam"] == "03"), None)
               or min(self.B["views"], key=lambda x: abs(x["t"] - t)))  # (no camera 03: any view sets the radius)
        C03 = -np.asarray(v03["R"]).T @ np.asarray(v03["t_w"])
        dist = float(np.clip(np.linalg.norm((C03 - target)[[0, 2]]), 2.5, 5.0))
        a0 = math.atan2(C03[0] - target[0], C03[2] - target[2])
        W, H = size
        f = 0.92 * H * dist / 2.3
        top, bot = [], []
        for k in range(n):
            a = a0 + 2 * math.pi * k / n
            C = target + np.array([dist * math.sin(a), 0.55, dist * math.cos(a)])
            v = {"i": -1, "cam": "novel", "split": "novel", "t": float(t), **look_at(C, target, f, W, H)}
            lab = f"{int(round(math.degrees(a - a0))) % 360} deg from cam 03"
            top.append(_font(np.ascontiguousarray(_u8(self.splat(v))), lab))
            av = self.avatar(v)
            if av is not None:
                bot.append(_font(np.ascontiguousarray(_u8(av)), "avatar"))
        half = (n + 1) // 2
        rows = [np.hstack(top[:half]), np.hstack(top[half:])]
        if bot:
            rows += [np.hstack(bot[:half]), np.hstack(bot[half:])]
        sheet = np.vstack(_pad_w(rows))
        _font(sheet, f"novel-view turntable t {t:.2f} s (rows 1-2 splats, 3-4 textured avatar)", (6, H - 8), 0.55)
        p = review / f"turntable_t{t:.2f}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return p.name

    def closeups(self, review: Path, kind: str, cams=("02", "06", "09"), times=TURNS, h: int = 200) -> str | None:
        """kind 'hair' (the follow's head + hair) or 'face' (both dancers): video | avatar | splats per camera x time."""
        hair_role = str(self.B["hair"]["role"]) if self.B.get("hair") is not None else "follow"
        targets = [(hair_role, HAIR_HALF_M, 0.18)] if kind == "hair" else [(r, FACE_HALF_M, 0.0) for r in ROLES]
        rows = []
        for t in times:
            cells = []
            for cam in cams:
                v = self.view_at(cam, t, 0.04)
                if v is None:
                    continue
                img, _ = read_view(self.B["dir"], v)
                gt = img.astype(np.float32) / 255
                sp = self.splat(v)
                av = self.avatar(v)
                for role, half, dy in targets:
                    sc, _ = head_points(self.B, role, v["t"])
                    box = crop_box(v, sc, half, dy)
                    if box is None:
                        continue
                    x0, y0, x1, y1 = box
                    side = max(x1 - x0, y1 - y0)
                    trip = [_tile(a[y0:y1, x0:x1], h) for a in (gt, av if av is not None else np.zeros_like(gt), sp)]
                    cell = np.hstack(trip)
                    _font(cell, f"cam {cam} t {v['t']:.2f} {role if kind == 'face' else 'hair'} ({side}px)")
                    _font(cell, "video | avatar | splats", (4, h - 6), 0.4)
                    cells.append(np.vstack([cell, np.zeros((4, cell.shape[1], 3), np.uint8)]))
            if cells:
                rows.append(np.hstack([np.hstack([c, np.zeros((c.shape[0], 6, 3), np.uint8)]) for c in cells]))
        if not rows:
            return None
        p = review / f"{kind}_closeups.jpg"
        cv2.imwrite(str(p), np.vstack(_pad_w(rows)), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return p.name

    def temporal_strips(self, review: Path, t0: float = 59.0, n: int = 12, h: int = 260) -> list[str]:
        out = []
        k0 = int(np.searchsorted(self.tg, t0 - 1e-6))
        ks = [k for k in range(k0, min(k0 + n, len(self.tg)))]
        vid, avs, sps = [], [], []
        for k in ks:
            v = self.view_at("03", float(self.tg[k]), 0.02)
            if v is None:
                continue
            img, _ = read_view(self.B["dir"], v)
            vid.append(_tile(img.astype(np.float32) / 255, h, f"video {v['t']:.3f}"))
            av = self.avatar(v)
            if av is not None:
                avs.append(_tile(av, h, "avatar"))
            sps.append(_tile(self.splat(v), h, "splats"))
        if vid:
            rows = [np.hstack(vid), np.hstack(sps)] + ([np.hstack(avs)] if avs else [])
            p = review / "temporal_strip_cam03.jpg"
            cv2.imwrite(str(p), np.vstack(_pad_w(rows)), [cv2.IMWRITE_JPEG_QUALITY, 86])
            out.append(p.name)
        nov = self.B.get("novel")
        if nov:
            base = {"i": -1, "cam": "novel", "split": "novel", "K": nov["K"], "R": nov["R"], "t_w": nov["t_w"],
                    "W": nov["W"], "H": nov["H"]}
            frames, diffs, prev = [], [], None
            for k in ks:
                v = {**base, "t": float(self.tg[k])}
                r = render_view(self.model, v, {}, self.sh)["rgb"]
                frames.append(_tile(r, h, f"novel {self.tg[k]:.3f}"))
                if prev is not None:
                    diffs.append(_tile(np.clip(np.abs(r - prev) * 4, 0, 1), h, "|change| x4"))
                else:
                    diffs.append(_tile(np.zeros_like(r), h, "|change| x4"))
                prev = r
            p = review / "temporal_strip_novel.jpg"
            cv2.imwrite(str(p), np.vstack(_pad_w([np.hstack(frames), np.hstack(diffs)])), [cv2.IMWRITE_JPEG_QUALITY, 86])
            out.append(p.name)
        return out


def extra_review(bundle: Path, out: Path, review: Path, take: str, device: str = "cuda", log=print) -> dict:
    t0 = time.time()
    review.mkdir(parents=True, exist_ok=True)
    rv = Reviewer(bundle, out, take, device)
    res = {"written": [], "side_by_side": []}
    held = rv.B["manifest"]["prep_config"]["holdout_cams"]
    for cam, tag in ((held[0] if held else "03", "heldout"), ("02", "train")):
        name, sc = rv.side_by_side(cam, review, tag)
        if name:
            res["written"].append(name)
            res["side_by_side"] += sc
    res["written"].append(rv.turntable(TURNS[0], review))
    for kind in ("hair", "face"):
        n = rv.closeups(review, kind)
        if n:
            res["written"].append(n)
    res["written"] += rv.temporal_strips(review)
    rv.torch.cuda.empty_cache()
    res["s"] = round(time.time() - t0, 1)
    (out / "review_extra.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"[dancer_splats] review: {len(res['written'])} sheets in {res['s']:.0f}s -> {review}")
    return res
