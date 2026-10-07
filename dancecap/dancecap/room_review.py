"""Review sheets for the room stage (work/<take>/review/room_*.jpg).

room_renders.jpg   held-out views per camera: real frame (people tinted) | full splat | Quest splat | Quest mesh
room_overview.jpg  top-down orthographic splat (ceiling cropped, cameras + 1 m world grid), oblique views, and a
                   side section of the Gaussian centres (floor at y = 0, nothing underneath = no mirrored floor)
room_mesh.jpg      Quest mesh: textured and shaded geometry from oblique world viewpoints + held-out camera views
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img: np.ndarray, text: str, sub: str = "", scale: float = 0.55) -> np.ndarray:
    img = np.ascontiguousarray(img)
    h = int(32 * scale / 0.55) + (int(20 * scale / 0.55) if sub else 0)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, text, (6, int(22 * scale / 0.55)), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (6, int(42 * scale / 0.55)), FONT, scale * 0.8, (200, 220, 255), 1, cv2.LINE_AA)
    return img


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 20, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def to8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def world_cam(C_w: np.ndarray, target_w: np.ndarray, up=np.array([0, 1.0, 0])):
    """OpenCV world->camera rotation/translation (world frame, Y up) for a camera at C looking at target."""
    z = target_w - C_w
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C_w


def sfm_viewmat(Rw: np.ndarray, tw: np.ndarray, world: dict) -> np.ndarray:
    """Camera given in WORLD coordinates -> viewmat on SfM points (camera coordinates in SfM units = world / s)."""
    s, R, t = world["s"], world["R"], world["t"]
    M = np.eye(4)
    M[:3, :3] = Rw @ R
    M[:3, 3] = (Rw @ t + tw) / s
    return M


def to_world(X: np.ndarray, world: dict) -> np.ndarray:
    return (world["s"] * (world["R"] @ X.T)).T + world["t"]


def _cut(splat, world: dict, y_max: float):
    import torch
    Xw = to_world(splat.params["means"].detach().cpu().numpy(), world)
    return torch.as_tensor(Xw[:, 1] < y_max, device=splat.params["means"].device)


def render_virtual(splat, Rw, tw, world, K, W, H, y_max=None, ortho=False):
    import torch
    vm = sfm_viewmat(Rw, tw, world)
    K = K.copy()
    if ortho:
        K[0, 0] *= world["s"]
        K[1, 1] *= world["s"]
    mask = _cut(splat, world, y_max) if y_max is not None else None
    with torch.no_grad():
        out, alpha, _ = splat.render(vm, K, W, H, mask=mask, camera_model="ortho" if ortho else "pinhole",
                                     backgrounds=torch.full((1, 3), 0.12, device="cuda"))
    return to8(out[0].cpu().numpy())


def top_down(splat, views, world, extent=(-9.0, 6.0, -7.0, 6.0), ppm: int = 90, y_max: float = 2.6,
             mirror: dict | None = None) -> np.ndarray:
    x0, x1, z0, z1 = extent
    W, H = int((x1 - x0) * ppm), int((z1 - z0) * ppm)
    Rw = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # image right = +X, image down = +Z, looking down -Y
    C = np.array([(x0 + x1) / 2, 30.0, (z0 + z1) / 2])
    tw = -Rw @ C
    K = np.array([[ppm, 0, W / 2], [0, ppm, H / 2], [0, 0, 1.0]])
    img = render_virtual(splat, Rw, tw, world, K, W, H, y_max=y_max, ortho=True)
    px = lambda X: (int((X[0] - x0) * ppm), int((X[2] - z0) * ppm))
    for gx in range(int(np.ceil(x0)), int(x1) + 1):
        cv2.line(img, px((gx, 0, z0)), px((gx, 0, z1)), (90, 90, 90) if gx else (255, 255, 0), 1)
    for gz in range(int(np.ceil(z0)), int(z1) + 1):
        cv2.line(img, px((x0, 0, gz)), px((x1, 0, gz)), (90, 90, 90) if gz else (255, 255, 0), 1)
    if mirror is not None and mirror.get("is_mirror"):
        n, al = np.asarray(mirror["n_w"]), np.asarray(mirror["along_w"])
        base = -mirror["d_w"] * n
        p0, p1 = base + al * mirror["along_lo"], base + al * mirror["along_hi"]
        cv2.line(img, px(p0), px(p1), (0, 230, 255), 3)
        cv2.putText(img, f"mirror wall (0-{mirror['top_m']:.1f} m)", px(p0 + 0.3 * n), FONT, 0.7, (0, 230, 255), 2,
                    cv2.LINE_AA)
    seen = {}
    for v in views:
        if v.cam in seen:
            continue
        seen[v.cam] = True
        Cw = to_world(v.center[None], world)[0]
        fwd = to_world((v.center + v.R[2] * 1.0)[None], world)[0] - Cw
        fwd = fwd / max(np.linalg.norm(fwd[[0, 2]]), 1e-6) * 0.8
        p = px(Cw)
        cv2.arrowedLine(img, p, px(Cw + fwd), (255, 80, 80), 2, tipLength=0.3)
        cv2.circle(img, p, 6, (255, 80, 80), -1)
        cv2.putText(img, v.cam, (p[0] + 8, p[1] + 5), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return label(img, f"top-down orthographic (world X right, +Z down), Gaussians below {y_max} m, 1 m grid, "
                      f"yellow = world axes through the dancers' origin, cyan = mirror", scale=0.6)


def section_plot(splat, world, geo) -> np.ndarray:
    """Side section of opaque Gaussian centres (world X-Y and Z-Y): the floor must be a single layer at y=0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = splat.numpy()
    op = 1 / (1 + np.exp(-d["opacities"]))
    Xw = to_world(d["means"], world)
    C = np.clip(0.5 + 0.28209479177387814 * d["sh0"][:, 0], 0, 1)
    sel = op > 0.3
    rng = np.random.default_rng(0)
    idx = np.nonzero(sel)[0]
    idx = rng.choice(idx, min(len(idx), 200_000), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(24, 6), gridspec_kw={"width_ratios": [2, 2, 1]})
    for a, (i, name) in zip(ax[:2], [(0, "X"), (2, "Z")]):
        a.scatter(Xw[idx, i], Xw[idx, 1], s=0.2, c=C[idx])
        a.axhline(0, color="k", lw=0.8)
        a.set_xlim(-12, 8)
        a.set_ylim(-2, 6)
        a.set_xlabel(f"world {name} (m)")
        a.set_ylabel("world Y up (m)")
        a.set_title(f"Gaussian centres (opacity > 0.3), side view {name}-Y")
        a.set_facecolor("#dddddd")
    near = sel & (np.abs(Xw[:, 0]) < 6) & (np.abs(Xw[:, 2]) < 6)
    ax[2].hist(Xw[near, 1], bins=np.arange(-1.0, 1.0, 0.01), color="#446")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("world Y (m), |X|,|Z| < 6 m")
    ax[2].set_title(f"heights near the floor: {np.mean(Xw[near, 1] < -0.1):.2%} below -10 cm")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def overview(splat, views, geo, world) -> np.ndarray:
    td = top_down(splat, views, world, mirror=geo.get("mirror"))
    obl = []
    W, H = 900, 560
    K = np.array([[560.0, 0, W / 2], [0, 560.0, H / 2], [0, 0, 1]])
    for C, tgt, txt in [((0.0, 6.5, 9.0), (0.0, 0.5, -1.5), "oblique from behind the main cameras"),
                        ((7.0, 6.0, -1.0), (-2.0, 0.5, -1.0), "oblique from +X"),
                        ((-8.0, 6.0, 3.0), (1.0, 0.5, -1.5), "oblique from -X"),
                        ((0.0, 1.6, 3.5), (0.0, 1.2, -3.0), "eye level at the camera ring (1.6 m)")]:
        Rw, tw = world_cam(np.array(C), np.array(tgt))
        obl.append(label(render_virtual(splat, Rw, tw, world, K, W, H, y_max=4.3), txt, f"camera at {C} m, ceiling "
                                                                                          f"> 4.3 m cropped"))
    grid = np.vstack([np.hstack(obl[:2]), np.hstack(obl[2:])])
    td = fit(td, grid.shape[1], int(grid.shape[1] * td.shape[0] / td.shape[1]))
    sec = section_plot(splat, world, geo)
    sec = fit(sec, grid.shape[1], int(grid.shape[1] * sec.shape[0] / sec.shape[1]))
    return np.vstack([td, grid, sec])


def renders_sheet(work: Path, splat, quest, app, refl, views, cfg, ev: dict, mesh=None, tile=(270, 480)) -> np.ndarray:
    from .room import load_view, render_view
    from .room_mesh import render_mesh
    tr = [v for v in views if v.split == "train"]
    by_cam: dict[str, list[int]] = {}
    for i, v in enumerate(tr):
        by_cam.setdefault(v.cam, []).append(i)
    metr = {r["view"]: r for r in ev["views"]}
    rows = []
    tw, th = tile
    for cam in sorted(by_cam):
        tests = [v for v in views if v.split == "test" and v.cam == cam]
        picks = [v for v in tests if v.kind == "kf"][:1] + [v for v in tests if v.kind == "win"][2:3]
        for v in picks:
            A, b = app.mean_for(by_cam[cam])
            img, valid = load_view(work, v, cfg)
            real = img.copy()
            real[~valid] = (real[~valid] * 0.45 + np.array([200, 30, 30]) * 0.55).astype(np.uint8)
            m = metr[v.label]
            r = render_view(splat, v, A, b, refl)
            Am, bm = A.cpu().numpy(), b.cpu().numpy()
            tiles = [label(fit(real, tw, th), f"cam {cam} REAL ({'keyframe' if v.kind == 'kf' else 'window'})",
                           f"t={v.t:.2f}s held out; red = masked"),
                     label(fit(to8(r["rgb"]), tw, th), "model: splat + reflections",
                           f"PSNR {m['psnr']:.1f} (cc {m['psnr_cc']:.1f}) SSIM {m['ssim']:.2f}"),
                     label(fit(to8(r["direct"] @ Am.T + bm), tw, th), "diffuse room = splat_full",
                           "(no floor/mirror reflections)")]
            if quest is not None:
                rq = render_view(quest, v, A, b, refl)
                tiles.append(label(fit(to8(rq["direct"] @ Am.T + bm), tw, th), f"Quest splat {len(quest):,}",
                                   "diffuse"))
            if mesh is not None:
                im = render_mesh(mesh["V"], mesh["F"], v.viewmat(), v.K, v.W, v.H, uv=mesh.get("uv"),
                                 tex=mesh.get("tex"), vcol=mesh.get("vcol"), ftex=mesh.get("ftex"))
                im = to8((im.astype(np.float32) / 255) @ Am.T + bm)
                tiles.append(label(fit(im, tw, th), f"Quest mesh {len(mesh['F']):,} tris"))
            rows.append(np.hstack(tiles))
    # two camera blocks per sheet row
    pairs = [np.hstack(rows[i:i + 2]) if i + 1 < len(rows) else np.hstack([rows[i], np.zeros_like(rows[i])])
             for i in range(0, len(rows), 2)]
    return np.vstack(pairs)


def mesh_sheet(mesh, views, world, work: Path, cfg) -> np.ndarray:
    from .room import load_view
    from .room_mesh import render_mesh
    W, H = 900, 560
    K = np.array([[560.0, 0, W / 2], [0, 560.0, H / 2], [0, 0, 1]])
    tiles = []
    for C, tgt, txt in [((0.0, 6.5, 9.0), (0.0, 0.5, -1.5), "from behind the main cameras"),
                        ((-8.0, 6.0, 3.0), (1.0, 0.5, -1.5), "from -X")]:
        Rw, tw = world_cam(np.array(C), np.array(tgt))
        vm = sfm_viewmat(Rw, tw, world)
        tex = render_mesh(mesh["V"], mesh["F"], vm, K, W, H, uv=mesh.get("uv"), tex=mesh.get("tex"),
                          vcol=mesh.get("vcol"), ftex=mesh.get("ftex"))
        geo = render_mesh(mesh["V"], mesh["F"], vm, K, W, H, shade=True)
        tiles += [label(tex, f"Quest mesh textured, {txt}", f"{len(mesh['F']):,} triangles"),
                  label(geo, f"shaded geometry, {txt}")]
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cams = []
    for cam in ("01", "03", "07", "09"):
        vs = [v for v in views if v.cam == cam and v.split == "test" and v.kind == "kf"]
        if not vs:
            continue
        v = vs[0]
        img, _ = load_view(work, v, cfg)
        geo_img = render_mesh(mesh["V"], mesh["F"], v.viewmat(), v.K, v.W, v.H, shade=True)
        tex_img = render_mesh(mesh["V"], mesh["F"], v.viewmat(), v.K, v.W, v.H, uv=mesh.get("uv"),
                              tex=mesh.get("tex"), vcol=mesh.get("vcol"), ftex=mesh.get("ftex"))
        th = 480
        tw = int(th * v.W / v.H)
        cams.append(np.hstack([label(fit(img, tw, th), f"cam {cam} real"), label(fit(tex_img, tw, th), "mesh"),
                               label(fit(geo_img, tw, th), "geometry")]))
    if cams:
        strip = np.hstack(cams)
        strip = fit(strip, grid.shape[1], int(grid.shape[1] * strip.shape[0] / strip.shape[1]))
        grid = np.vstack([grid, strip])
    return grid


def load_mesh(out: Path) -> dict | None:
    """Quest mesh from a room output dir (textured cache if present, else the vertex-coloured PLY)."""
    uvp = out / "cache" / "room_mesh_uv.npz"
    texp = out / "room_mesh_tex.png"
    if uvp.exists() and texp.exists():
        d = np.load(uvp)
        texs = [cv2.imread(str(texp))[:, :, ::-1].copy()]
        if (out / "room_floor_tex.png").exists():
            texs.append(cv2.imread(str(out / "room_floor_tex.png"))[:, :, ::-1].copy())
        return {"V": d["V"], "F": d["F"], "uv": d["uv"], "vcol": d["vcol"], "tex": texs,
                "ftex": d["ftex"] if "ftex" in d.files else None}
    if (out / "room_mesh.ply").exists():
        import open3d as o3d
        m = o3d.io.read_triangle_mesh(str(out / "room_mesh.ply"))
        return {"V": np.asarray(m.vertices), "F": np.asarray(m.triangles),
                "vcol": np.asarray(m.vertex_colors).astype(np.float32)}
    return None


def write(path: Path, img_rgb: np.ndarray, max_w: int = 3600) -> Path:
    if img_rgb.shape[1] > max_w:
        s = max_w / img_rgb.shape[1]
        img_rgb = cv2.resize(img_rgb, (max_w, int(img_rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def render_all(work: Path, splat, app, refl, views, geo, cfg, ev: dict, log=print, out: Path | None = None,
               review: Path | None = None) -> list[str]:
    """work = where the views' images live (bundle dir); out = the room outputs (default work/room); review = the
    sheet folder (default work/review)."""
    from .room import Splat
    out = Path(out) if out is not None else work / "room"
    rev = Path(review) if review is not None else work / "review"
    world = geo["world"]
    quest = Splat.from_ply(out / "splat_quest.ply") if (out / "splat_quest.ply").exists() else None
    mesh = load_mesh(out)
    paths = [write(rev / "room_renders.jpg", renders_sheet(work, splat, quest, app, refl, views, cfg, ev, mesh)),
             write(rev / "room_overview.jpg", overview(splat, views, geo, world))]
    if quest is not None:
        paths.append(write(rev / "room_overview_quest.jpg", top_down(quest, views, world, mirror=geo.get("mirror"))))
    if mesh is not None:
        paths.append(write(rev / "room_mesh.jpg", mesh_sheet(mesh, views, world, work, cfg)))
    for p in paths:
        log(f"[room] review -> {p}")
    return [str(p) for p in paths]
