"""QA renders for the avatar textures (avatar_texture.py).

  review/avatar_texture_<role>.jpg   the textured avatar re-rendered into real frames: PHOTO | AVATAR | AVATAR OVER PHOTO
                                     (one row per camera, the frame where that dancer is seen best), with the
                                     render-vs-photo error inside the dancer's mask
  review/avatar_uv.jpg               both albedo atlases + coverage (measured cameras / mirrored / diffused)
  review/avatar_turntable.jpg        both avatars at one frame from 4 virtual cameras around them (front/side/back)
  review/avatar_face.jpg             head close-ups where each face is seen best (largest, most frontal): PHOTO | AVATAR

Renders use avatar_raster (pure-torch triangle rasteriser + bilinear atlas lookup). The smoke preset writes the same
sheets as review/avatar_smoke_*.jpg from avatar/smoke/.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import cv2
import numpy as np

from .avatar_texture import ROLES, TextureConfig, _Geo, camera_views, load_bodies, render
from .avatar_uv import build_atlas, model_npz, vertex_normals, vertex_regions
from .mamma_dyn import SMPLX_MODELS
from .review import _label
from .take import Take

LABEL = {"lead": "LEAD (Kadu)", "follow": "FOLLOW (Larissa)"}


def _crop_box(mask: np.ndarray, W: int, H: int, aspect: float = 0.62, pad: float = 0.12):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, W, H
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


def _pick_views(stats: list[dict], role: str, per_cam: int = 1) -> list[int]:
    by_cam: dict = {}
    for k, s in enumerate(stats):
        d = s.get("roles", {}).get(role)
        if not d or not d.get("n_samples"):
            continue
        score = d["iou"] * np.sqrt(d["n_samples"])
        by_cam.setdefault(s["cam"], []).append((score, k))
    out = []
    for c in sorted(by_cam):
        best = sorted(by_cam[c], reverse=True)
        chosen = []
        for sc, k in best:  # spread in time when more than one per camera
            if all(abs(stats[k]["t"] - stats[j]["t"]) > 1.0 for j in chosen):
                chosen.append(k)
            if len(chosen) == per_cam:
                break
        out += chosen
    return out


def _texel_colors(torch, out_dir: Path, device) -> dict:
    out = {}
    for r in ROLES:
        z = np.load(out_dir / f"{r}_texels.npz")
        out[r] = torch.as_tensor(np.asarray(z["rgb_srgb"], np.float32), device=device)
    return out


def render_review(take: Take, cfg: TextureConfig | None = None, bodies=None, views=None, stats=None,
                  device: str | None = None, lock: bool = True, panel_h: int = 460, out: Path | None = None,
                  prefix: str = "avatar") -> list[Path]:
    import torch

    from . import gpu
    cfg = cfg or TextureConfig()
    work = take.work
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rev = work / "review"
    rev.mkdir(parents=True, exist_ok=True)
    out = out or work / "avatar"
    rep_p = out / "report.json"
    report = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    stale = (report.get("smplx_freshness") or {}).get("stale")
    note = "  [STALE smplx: previous world frame]" if stale else ""
    if bodies is None:
        bodies, _ = load_bodies(work, out=out)
    if stats is None:
        stats = report["views"]
    if views is None:
        cams = report.get("cams") or sorted({s["cam"] for s in stats})
        if report.get("t_window"):  # the bake's window and step (stats are indexed like its view list)
            t_lo, t_hi = report["t_window"]
            step = float(report.get("config", {}).get("step_s", cfg.step_s))
        else:
            t_lo = max(b.t[0] for b in bodies.values())
            t_hi = min(b.t[-1] for b in bodies.values())
            step = cfg.step_s
        views, _ = camera_views(work, cams, t_lo, t_hi, step)
        if len(views) != len(stats):
            raise RuntimeError(f"review: {len(views)} views rebuilt but the report has {len(stats)}; re-run the bake")
    atlas = build_atlas(SMPLX_MODELS, cfg.size)
    written = []
    ctx = gpu.exclusive("texture-review") if (lock and device == "cuda") else contextlib.nullcontext()
    metrics: dict = {}
    with ctx:
        geo = _Geo(atlas, device)
        colors = _texel_colors(torch, out, device)
        for role in ROLES:
            rows, metrics[role] = [], {}
            for k in _pick_views(stats, role, per_cam=1):
                v = views[k]
                img = cv2.imread(str(work / "frames" / v["cam"] / f"{v['i']:06d}.jpg"))[:, :, ::-1]
                H, W = img.shape[:2]
                posed = {r: b.at(v["t"])[0] for r, b in bodies.items()}
                ren, rid = render(torch, geo, posed, colors, v["K"], v["R"], v["t_w"], v["k1"], H, W, device)
                ren8 = (np.clip(ren, 0, 1) * 255).astype(np.uint8)
                own = rid == ROLES.index(role)
                m = cv2.imread(str(work / "dancers" / v["cam"] / "masks" / f"mask_{v['i']:04d}_{ROLES.index(role) + 1:02d}.png"),
                               cv2.IMREAD_GRAYSCALE)
                m = (m > 127) if m is not None else np.zeros((H, W), bool)
                both = own & cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                err = float(np.abs(ren8[both].astype(float) - img[both].astype(float)).mean()) if both.any() else np.nan
                metrics[role][v["cam"]] = {"i": v["i"], "t": round(v["t"], 3), "mae_srgb8": round(err, 1),
                                           "iou": stats[k]["roles"][role]["iou"]}
                box = _crop_box(own | m, W, H)
                over = img.copy()
                anyb = rid >= 0
                over[anyb] = ren8[anyb]
                bg = np.full_like(ren8, 40)
                bg[anyb] = ren8[anyb]
                tiles = [_crop(np.ascontiguousarray(x[:, :, ::-1]), box, panel_h) for x in (img, bg, over)]
                for tile, name in zip(tiles, ("photo", "avatar", "avatar over photo")):
                    _label(tile, name, (8, panel_h - 12), (255, 255, 255), 0.6)
                _label(tiles[0], f"cam {v['cam']} #{v['i']}", (8, 28), (255, 255, 255), 0.75)
                _label(tiles[0], f"t={v['t']:.2f}s IoU {stats[k]['roles'][role]['iou']:.2f}", (8, 54), (220, 220, 220), 0.55)
                _label(tiles[1], f"MAE {err:.0f}/255", (8, 28), (220, 220, 220), 0.6)
                rows.append(np.concatenate(tiles + [np.zeros((panel_h, 6, 3), np.uint8)], 1))
            written.append(_sheet(rows, rev / f"{prefix}_texture_{role}.jpg",
                                  f"{take.name} {LABEL[role]} [{cfg.preset} {cfg.size}^2]{note}: texture re-rendered "
                                  f"(best frame per camera), naked SMPL-X, no hair"))
        written.append(_face_sheet(torch, geo, work, bodies, colors, views, stats, rev / f"{prefix}_face.jpg",
                                   take.name + note, device))
        written.append(_uv_sheet(out, atlas, rev / f"{prefix}_uv.jpg", take.name + note))
        written.append(_turntable(torch, geo, bodies, colors, rev / f"{prefix}_turntable.jpg", take.name + note, device))
    if rep_p.exists():
        report["review_metrics"] = metrics
        rep_p.write_text(json.dumps(report, indent=1, default=float))
    return written


def _face_views(bodies, views, stats, role, n: int = 4, stride: int = 2) -> list[tuple[int, float, np.ndarray]]:
    """Views where the role's face is large, frontal and unoccluded: score = head height (px) x mean over face
    vertices of facing cosine x visible (coarse vertex z-buffer of both bodies)."""
    m = np.load(model_npz(SMPLX_MODELS), allow_pickle=True)
    reg = vertex_regions(m)
    faces = np.asarray(m["f"], np.int64)
    head_idx, face_idx = np.flatnonzero(reg["head"]), np.flatnonzero(reg["face"])
    cand = []
    for k in range(0, len(views), stride):
        v, s = views[k], stats[k]
        d = s.get("roles", {}).get(role)
        if not d or d.get("iou", 0) < 0.45 or not d.get("n_samples"):
            continue
        V = bodies[role].at(v["t"])[0].astype(np.float64)
        C = -v["R"].T @ v["t_w"]
        N = vertex_normals(V, faces)
        dirc = C - V[face_idx]
        cos = (N[face_idx] * dirc / np.linalg.norm(dirc, axis=1, keepdims=True)).sum(1)
        Xc = V[head_idx] @ v["R"].T + v["t_w"]
        uv = Xc[:, :2] / Xc[:, 2:3] * v["K"][0, 0] + v["K"][:2, 2]
        hpx = uv[:, 1].max() - uv[:, 1].min()
        # occlusion: coarse z-buffer (8 px cells) of both bodies' vertices, 3x3 min filter
        cell = 8.0
        allv = np.concatenate([b.at(v["t"])[0] for b in bodies.values()]).astype(np.float64)
        Xa = allv @ v["R"].T + v["t_w"]
        ua = (Xa[:, :2] / Xa[:, 2:3] * v["K"][0, 0] + v["K"][:2, 2]) / cell
        Wc, Hc = int(2 * v["K"][0, 2] / cell) + 2, int(2 * v["K"][1, 2] / cell) + 2
        ia = np.round(ua).astype(int)
        ok = (ia[:, 0] >= 0) & (ia[:, 0] < Wc) & (ia[:, 1] >= 0) & (ia[:, 1] < Hc) & (Xa[:, 2] > 0.1)
        zb = np.full((Hc, Wc), 1e9)
        np.minimum.at(zb, (ia[ok, 1], ia[ok, 0]), Xa[ok, 2])
        zb = -cv2.dilate(-zb.astype(np.float32), np.ones((3, 3), np.uint8))
        Xf = V[face_idx] @ v["R"].T + v["t_w"]
        uf = np.round((Xf[:, :2] / Xf[:, 2:3] * v["K"][0, 0] + v["K"][:2, 2]) / cell).astype(int)
        inb = (uf[:, 0] >= 0) & (uf[:, 0] < Wc) & (uf[:, 1] >= 0) & (uf[:, 1] < Hc)
        vis = np.zeros(len(face_idx), bool)
        vis[inb] = Xf[inb, 2] <= zb[uf[inb, 1], uf[inb, 0]] + 0.04
        cand.append((float((np.clip(cos, 0, 1) * vis).mean()) * hpx, k, uv))
    cand.sort(key=lambda c: -c[0])
    out = []
    for sc, k, uv in cand:
        if all(views[k]["cam"] != views[j]["cam"] or abs(views[k]["t"] - views[j]["t"]) > 1.5 for _, j, _ in out):
            out.append((sc, k, uv))
        if len(out) == n:
            break
    return [(k, sc, uv) for sc, k, uv in out]


def _face_sheet(torch, geo, work, bodies, colors, views, stats, path: Path, take: str, device, tile: int = 300) -> Path:
    rows = []
    for role in ROLES:
        tiles = []
        for k, sc, uv in _face_views(bodies, views, stats, role):
            v = views[k]
            img = cv2.imread(str(work / "frames" / v["cam"] / f"{v['i']:06d}.jpg"))
            H, W = img.shape[:2]
            posed = {r: b.at(v["t"])[0] for r, b in bodies.items()}
            ren, rid = render(torch, geo, posed, colors, v["K"], v["R"], v["t_w"], v["k1"], H, W, device)
            ren8 = (np.clip(ren, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
            bg = np.where((rid >= 0)[..., None], ren8, 40).astype(np.uint8)
            (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
            c, h = (uv.min(0) + uv.max(0)) / 2, max(x1 - x0, y1 - y0) * 1.6
            box = (int(c[0] - h / 2), int(c[1] - h / 2), int(c[0] + h / 2), int(c[1] + h / 2))
            a, b = _crop(img, box, tile), _crop(bg, box, tile)
            _label(a, f"{LABEL[role]} cam {v['cam']} #{v['i']}", (6, 22), (255, 255, 255), 0.5)
            _label(a, f"head {y1 - y0:.0f} px tall", (6, 44), (220, 220, 220), 0.5)
            _label(b, "avatar", (6, 22), (255, 255, 255), 0.5)
            tiles += [a, b, np.zeros((tile, 6, 3), np.uint8)]
        if tiles:
            rows.append(np.concatenate(tiles, 1))
    if not rows:
        rows = [np.zeros((tile, tile, 3), np.uint8)]
    w = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0))) for r in rows], 0)
    head = np.full((40, w, 3), 12, np.uint8)
    _label(head, f"{take}  faces where seen best (largest x most frontal): photo | textured avatar (upscaled crops)",
           (8, 28), (255, 255, 255), 0.6)
    cv2.imwrite(str(path), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def _sheet(rows: list[np.ndarray], path: Path, title: str, per_row: int = 2, max_w: int = 3600) -> Path:
    if not rows:
        img = np.zeros((200, 1200, 3), np.uint8)
        _label(img, "no views", (20, 100), (255, 255, 255), 1.0)
        cv2.imwrite(str(path), img)
        return path
    lines = []
    for a in range(0, len(rows), per_row):
        grp = rows[a:a + per_row]
        h = max(r.shape[0] for r in grp)
        lines.append(np.concatenate([np.pad(r, ((0, h - r.shape[0]), (0, 0), (0, 0))) for r in grp], 1))
    w = max(l.shape[1] for l in lines)
    lines = [np.pad(l, ((0, 6), (0, w - l.shape[1]), (0, 0))) for l in lines]
    sheet = np.concatenate(lines, 0)
    head = np.full((46, w, 3), 12, np.uint8)
    _label(head, title, (10, 32), (255, 255, 255), 0.75)
    sheet = np.concatenate([head, sheet], 0)
    if sheet.shape[1] > max_w:
        s = max_w / sheet.shape[1]
        sheet = cv2.resize(sheet, (max_w, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def coverage_vis(atlas, albedo_bgr: np.ndarray, cov_bgr: np.ndarray) -> np.ndarray:
    """Albedo dimmed, tinted by source: measured = cameras count (dark->green->yellow), mirror cyan, diffused magenta."""
    alpha, ncam, src = cov_bgr[:, :, 2] / 255.0, cov_bgr[:, :, 1] // 32, cov_bgr[:, :, 0]
    base = albedo_bgr.astype(np.float32) * 0.35
    lut = cv2.applyColorMap(np.linspace(0, 255, 9).astype(np.uint8)[:, None], cv2.COLORMAP_VIRIDIS)[:, 0].astype(np.float32)
    tint = lut[np.clip(ncam, 0, 8)]
    tint[src == 128] = (255, 255, 0)
    tint[src == 255] = (255, 0, 255)
    covered = atlas.to_image(np.ones(atlas.n, np.uint8)) > 0
    out = np.where(covered[..., None], base + 0.65 * tint, 20).astype(np.uint8)
    del alpha
    return out


def _uv_sheet(out_dir: Path, atlas, path: Path, take: str, tile: int = 1200) -> Path:
    tiles = []
    for r in ROLES:
        alb = cv2.imread(str(out_dir / f"{r}_albedo.png"))
        cov = cv2.imread(str(out_dir / f"{r}_coverage.png"))
        a = cv2.resize(alb, (tile, tile), interpolation=cv2.INTER_AREA)
        c = cv2.resize(coverage_vis(atlas, alb, cov), (tile, tile), interpolation=cv2.INTER_AREA)
        _label(a, f"{LABEL[r]} albedo {alb.shape[1]}x{alb.shape[0]}", (10, 34), (255, 255, 255), 0.9)
        _label(c, f"{LABEL[r]} coverage", (10, 34), (255, 255, 255), 0.9)
        tiles.append(np.concatenate([a, c], 1))
    sheet = np.concatenate(tiles, 0)
    leg = np.full((60, sheet.shape[1], 3), 12, np.uint8)
    _label(leg, f"{take}  coverage: measured (colour = inlier cameras 1 dark .. 8 yellow), CYAN = mirrored from the "
                "other side, MAGENTA = diffused over the mesh (no camera saw it)", (10, 38), (255, 255, 255), 0.75)
    sheet = np.concatenate([leg, sheet], 0)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def _look_at(C, target, up=np.array([0.0, 1.0, 0.0])):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def _turntable(torch, geo, bodies, colors, path: Path, take: str, device, H: int = 1100, W: int = 700) -> Path:
    tiles = []
    k = len(next(iter(bodies.values())).t) // 2
    for r in ROLES:
        b = bodies[r]
        v = b.verts[k]
        center = v.mean(0)
        hgt = v[:, 1].max() - v[:, 1].min()
        f = H * 3.2 / (hgt * 1.25)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        # front = the direction the pelvis faces (global orient z axis, horizontal)
        J = b.joints[k]
        fwd = np.cross(J[2] - J[1], np.array([0.0, 1.0, 0.0]))  # right hip -> left hip x up
        fwd = -fwd / max(np.linalg.norm(fwd), 1e-9)
        for ang, name in ((0, "front"), (90, "side"), (180, "back"), (270, "side")):
            a = np.deg2rad(ang)
            Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            C = center + Ry @ fwd * 3.2
            R, t = _look_at(C, center)
            ren, rid = render(torch, geo, {r: v}, {r: colors[r]}, K, R, t, 0.0, H, W, device)
            img = np.where((rid >= 0)[..., None], ren, 0.16)
            tile = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
            _label(tile, f"{LABEL[r]} {name}", (10, 32), (255, 255, 255), 0.8)
            tiles.append(tile)
    row1, row2 = np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)
    sheet = np.concatenate([row1, row2], 0)
    head = np.full((46, sheet.shape[1], 3), 12, np.uint8)
    _label(head, f"{take}  textured SMPL-X avatars at t={bodies['lead'].t[k]:.2f}s from virtual cameras (no hair, "
                 f"naked body surface, baked studio lighting)", (10, 32), (255, 255, 255), 0.75)
    sheet = np.concatenate([head, sheet], 0)
    if sheet.shape[1] > 3600:
        s = 3600 / sheet.shape[1]
        sheet = cv2.resize(sheet, (3600, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path
