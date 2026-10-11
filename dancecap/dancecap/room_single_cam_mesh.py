"""The room of the single-camera take (john_lula): planes from the MoGe-2 point maps on the people-based metric, textured floor + wall mesh.

    python -m dancecap.room_single_cam room        (also reachable as `python -m dancecap.room_single_cam room`)

The monocular geometry model gives the SHAPE of the hall (a floor, a far wall, a left wall, a ceiling) in its own metre; the people give the scale
(camera height from the dancers' head / feet rays and statures). The cloud of 22 frames is placed in the world frame through the per-frame camera
of cameras/01.npz and rescaled so that the model's floor lies at the camera height below the camera. Planes are the peaks of the offset among the
points whose normal fits; the per-frame medians show how stable they are. The mesh textures come from 35 frames of the video (people masked).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import room_single_cam as RS
from .cameras import ptz_feats as F

SIZE, FPS = RS.SIZE, RS.FPS
MAX_FLOOR_TEXEL_DISTANCE_M = 11.0


def moge_world_cloud(arr: dict | None = None, moge_path: Path | None = None) -> dict:
    """The MoGe-2 point maps in the WORLD frame. The model's depth z is kept, the pixel -> ray map is OURS (focal and k1 of the track), then
    everything is scaled so that the model's floor (mode of the up-facing points below the camera) lies at the track's camera height.
    Returns P (n,3) world metres, N (n,3) world normals, F (n,) video frame, k (the scale), floor_mode_y_model."""
    arr = arr or dict(np.load(RS.CAM_DIR / "01.npz", allow_pickle=True))
    rig = json.loads((RS.CAM_DIR / "rig.json").read_text())
    C = np.array(rig["centre_world_m"])
    h = float(rig["camera_height_m"])
    m = np.load(moge_path or RS.POD / "moge.npz")
    pts, nor, mask, fr = m["points"].astype(np.float32), m["normal"].astype(np.float32), m["mask"], m["frame"]
    H, Wd = pts.shape[1:3]
    v, u = np.mgrid[0:H, 0:Wd]
    px, py = (u + 0.5) * SIZE[0] / Wd, (v + 0.5) * SIZE[1] / H
    Ps, Ns, Fs = [], [], []
    for i, f0 in enumerate(fr):
        j = int(np.argmin(np.abs(arr["frame_index"] - f0)))
        R, f, k1 = arr["R"][j], float(arr["K"][j, 0, 0]), float(arr["dist"][j])
        xd = np.stack([(px - SIZE[0] / 2) / f, (py - SIZE[1] / 2) / f], -1)
        xu = xd.copy()
        for _ in range(6):
            xu = xd / (1 + k1 * (xu ** 2).sum(-1, keepdims=True))
        z = pts[i][..., 2]
        Xc = np.concatenate([xu * z[..., None], z[..., None]], -1)
        ok = mask[i] & np.isfinite(Xc).all(-1) & (z > 0.2)
        # camera frame -> world axes (row-vector form of R^T x); the camera is the origin of this cloud, up = +Y
        Ps.append(Xc[ok] @ R)
        Ns.append(nor[i][ok] @ R)
        Fs.append(np.full(int(ok.sum()), int(f0)))
    P_, N_, F_ = np.concatenate(Ps), np.concatenate(Ns), np.concatenate(Fs)
    fl = (N_[:, 1] > 0.9) & (P_[:, 1] < 0)
    hist, edges = np.histogram(P_[fl, 1], bins=np.arange(-3, 0, 0.02))
    yf = float(edges[np.argmax(hist)] + 0.01)
    k = h / abs(yf)
    Pw = P_ * k
    Pw[:, 1] += h
    Pw[:, 0] += C[0]
    Pw[:, 2] += C[2]
    return {"P": Pw.astype(np.float32), "N": N_.astype(np.float32), "F": F_, "k": float(k), "floor_mode_y_model": yf}


def peak(vals: np.ndarray, lo: float, hi: float, bw: float = 0.25, min_count: int = 3000) -> float | None:
    hist, edges = np.histogram(vals, bins=np.arange(lo, hi, bw))
    i = int(np.argmax(hist))
    if hist[i] < min_count:
        return None
    return float(np.median(vals[np.abs(vals - (edges[i] + bw / 2)) < 0.4]))


def fit_planes(cl: dict) -> dict:
    """Far wall (normal +Z), left wall (normal +X) and ceiling (normal down): the peak of the plane offset among the points whose normal fits, the
    per-frame medians as the evidence of the scatter, the measured azimuth of the normals (the walls are then taken parallel to the world axes)."""
    P, N, F_ = cl["P"], cl["N"], cl["F"]
    mid = (P[:, 1] > 0.4) & (P[:, 1] < 4.0) & (np.abs(N[:, 1]) < 0.3)
    far = mid & (N[:, 2] > 0.9) & (P[:, 2] < -9.0)
    left = mid & (N[:, 0] > 0.9) & (P[:, 0] < -6.0)
    ceil = (N[:, 1] < -0.9) & (P[:, 1] > 4.0)
    out = {}
    for name, sel, axis, lo, hi in (("far_wall", far, 2, -24, -9), ("left_wall", left, 0, -16, -6), ("ceiling", ceil, 1, 4.0, 8.0)):
        z0 = peak(P[sel, axis], lo, hi, 0.25, 2500)
        per = []
        for f0 in np.unique(F_):
            m = sel & (F_ == f0) & (np.abs(P[:, axis] - (z0 if z0 is not None else 0)) < 1.8)
            if m.sum() > 300:
                per.append(float(np.median(P[m, axis])))
        near = sel & (np.abs(P[:, axis] - (z0 or 0)) < 0.5)
        az = np.degrees(np.arctan2(N[near, 2], N[near, 0])) if z0 is not None and near.any() else np.array([0.0])
        out[name] = {"offset_m": z0, "per_frame_median_m": per,
                     "per_frame_mad_m": float(1.4826 * np.median(np.abs(np.array(per) - np.median(per)))) if per else None,
                     "n_points": int(sel.sum()), "n_frames": len(per), "normal_azimuth_deg_median": float(np.median(az))}
    return out


def write_obj(path: Path, groups: dict, textures: dict) -> None:
    """groups: {name: (V (n,3), UV (n,2), F (m,3))}; one OBJ + MTL with map_Kd per group (CONTRACTS.md room/room_mesh.obj: usemtl floor / room)."""
    lines = [f"mtllib {path.stem}.mtl"]
    base = 0
    for name, (V, UV, Fc) in groups.items():
        lines += [f"o {name}"] + [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in V] + [f"vt {u:.6f} {v:.6f}" for u, v in UV]
        lines.append(f"usemtl {name}")
        lines += [f"f {a + 1 + base}/{a + 1 + base} {b + 1 + base}/{b + 1 + base} {c + 1 + base}/{c + 1 + base}" for a, b, c in Fc]
        base += len(V)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".mtl").write_text("".join(f"newmtl {n}\nKd 1 1 1\nmap_Kd {tex}\n" for n, tex in textures.items()))


def grid_mesh(origin, du, dv, nu: int, nv: int):
    """Regular grid P = origin + i du + j dv with UV = (i / nu, 1 - j / nv) (image row 0 = j 0) and two triangles per cell."""
    ii, jj = np.meshgrid(np.arange(nu + 1), np.arange(nv + 1))
    V = origin[None] + ii.ravel()[:, None] * du[None] + jj.ravel()[:, None] * dv[None]
    UV = np.stack([ii.ravel() / nu, 1.0 - jj.ravel() / nv], 1)
    idx = lambda i, j: j * (nu + 1) + i          # noqa: E731
    Fc = []
    for j in range(nv):
        for i in range(nu):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            Fc += [(a, b, c), (a, c, d)]
    return V, UV, np.array(Fc)


def texture_sources(times: list[float]) -> list[dict]:
    arr = dict(np.load(RS.CAM_DIR / "01.npz", allow_pickle=True))
    fr = [int(round(t * FPS)) for t in times]
    got = RS.read_frames(fr)
    imgs = [got[k] for k in fr]
    masks = []
    for b in range(0, len(imgs), 8):
        pm = F.person_masks_yolo(imgs[b:b + 8], str(RS.YOLO_SEG))
        masks += [F.static_ignore(im.shape[:2], m, []) for im, m in zip(imgs[b:b + 8], pm)]
    return [{"arr": arr, "i": int(np.argmin(np.abs(arr["frame_index"] - k))), "img": im, "ignore": cv2.dilate(mk, np.ones((9, 9), np.uint8))}
            for k, im, mk in zip(fr, imgs, masks)]


def stage_room(a=None) -> dict:
    """room/room_model.json + room_mesh.obj / .mtl + room_floor_tex.png + room_mesh_tex.png + mesh_report.json."""
    from . import room_two_cams_walls as WL
    RS.ROOM_DIR.mkdir(parents=True, exist_ok=True)
    rig = json.loads((RS.CAM_DIR / "rig.json").read_text())
    cl = moge_world_cloud()
    pl = fit_planes(cl)
    zf, xl, yc = pl["far_wall"]["offset_m"], pl["left_wall"]["offset_m"], pl["ceiling"]["offset_m"]
    x_right, z_near = 8.0, 4.0
    quads = [([[xl, 0, zf], [x_right, 0, zf], [x_right, yc, zf], [xl, yc, zf]], [255, 160, 0]),
             ([[xl, 0, zf], [xl, 0, z_near], [xl, yc, z_near], [xl, yc, zf]], [255, 0, 255])]
    model = {"schema": "john_lula.room_model/1",
             "frame": "world: right-handed, Y up, metres, floor y = 0, origin = the dancers' mean floor position, X along the floor boards, Z towards the camera",
             "scale": {"source": "camera height from the dancers' head / feet rays and their statures; the MoGe-2 point maps are rescaled to it",
                       "moge_scale_to_people_metric": cl["k"], "floor_mode_y_model_m": cl["floor_mode_y_model"],
                       "uncertainty": "+-5 % from the height, +-10 % more through the focal (distances), far-field model error ~8 %: wall offsets +-2.5 m"},
             "axes": rig["x_axis"],
             "floor": {"y": 0.0, "x": [xl, x_right], "z": [zf, z_near], "source": "feet on the floor + the floor plane of the monocular model",
                       "textured_from": "35 frames of the video, people masked, per-texel median; what nobody sees is inpainted"},
             "far_wall": {"plane": {"normal": [0, 0, 1], "z": zf}, "x": [xl, x_right], "y": [0, yc], **pl["far_wall"],
                          "note": "grey-green hall wall behind the dancers (banner wall); the extent beyond what the camera sees (right edge) is a guess"},
             "left_wall": {"plane": {"normal": [1, 0, 0], "x": xl}, "z": [zf, z_near], "y": [0, yc], **pl["left_wall"]},
             "ceiling": {"y": yc, **pl["ceiling"], "note": "acoustic baffles under the roof (the lighting truss hangs lower); NOT in the mesh"},
             "unobserved": ["right wall and the floor to the right of the pictures (the camera never turns there)", "the wall behind the camera",
                            "the floor nearer than ~1.8 m in front of the camera"],
             "camera_world_m": rig["centre_world_m"], "quads": quads}
    (RS.ROOM_DIR / "room_model.json").write_text(json.dumps(RS._jsonable(model), indent=1))
    srcs = texture_sources(list(np.arange(1.0, 87.5, 2.5)))
    RS.log(f"{len(srcs)} frames decoded + masked")
    res = 0.05
    nx, nz = int(round((x_right - xl) / res)), int(round((z_near - zf) / res))
    X_, Z_ = np.meshgrid(xl + (np.arange(nx) + 0.5) * res, zf + (np.arange(nz) + 0.5) * res)
    tex, _ = WL.ortho_texture(np.stack([X_.ravel(), np.zeros(X_.size), Z_.ravel()], 1), (nz, nx), srcs, min_obs=2)
    # a texel far from the camera is a few pixels of a grazing view (stretched streaks, audience legs): keep the well-resolved floor, inpaint the rest
    C = np.array(rig["centre_world_m"])
    tex[np.hypot(X_ - C[0], Z_ - C[2]) > MAX_FLOOR_TEXEL_DISTANCE_M] = np.nan
    lum = np.nanmean(tex, axis=2)
    med = np.nanpercentile(tex.reshape(-1, 3), 60, axis=0)
    tex[lum < 0.6 * np.nanmedian(lum)] = np.nan             # dark texels are legs / shadows leaking through the people masks, not floor
    cover_f = np.isfinite(tex[:, :, 0])
    floor_img = np.where(cover_f[:, :, None], np.nan_to_num(tex), med[None, None]).astype(np.float32)     # the floor is one even light wood: plain median where unseen
    soft = cv2.GaussianBlur(cover_f.astype(np.float32), (0, 0), 3.0)[:, :, None]
    floor_img = np.clip(soft * cv2.GaussianBlur(np.nan_to_num(tex).astype(np.float32), (0, 0), 1.0) + (1 - soft) * med[None, None], 0, 255).astype(np.uint8)
    nwx, nwy = int(round((x_right - xl) / res)), int(round(yc / res))
    XW, YW = np.meshgrid(xl + (np.arange(nwx) + 0.5) * res, yc - (np.arange(nwy) + 0.5) * res)
    wtex, _ = WL.ortho_texture(np.stack([XW.ravel(), YW.ravel(), np.full(XW.size, zf)], 1), (nwy, nwx), srcs, min_obs=2)
    far_img = WL.fill_holes(wtex, (120, 135, 130))
    cover_w = np.isfinite(wtex[:, :, 0])
    nlz = int(round((z_near - zf) / res))
    ZW, YL = np.meshgrid(zf + (np.arange(nlz) + 0.5) * res, yc - (np.arange(nwy) + 0.5) * res)
    ltex, _ = WL.ortho_texture(np.stack([np.full(ZW.size, xl), YL.ravel(), ZW.ravel()], 1), (nwy, nlz), srcs, min_obs=2)
    left_img = WL.fill_holes(ltex, (120, 135, 130))
    cover_l = np.isfinite(ltex[:, :, 0])
    atlas = np.zeros((nwy, nwx + nlz, 3), np.uint8)               # the "room" submesh: far wall left, left wall right
    atlas[:, :nwx], atlas[:, nwx:] = far_img, left_img
    cv2.imwrite(str(RS.ROOM_DIR / "room_floor_tex.png"), floor_img)
    cv2.imwrite(str(RS.ROOM_DIR / "room_mesh_tex.png"), atlas)
    Vf, UVf, Ff = grid_mesh(np.array([xl, 0.0, zf]), np.array([(x_right - xl) / 20, 0, 0]), np.array([0, 0, (z_near - zf) / 18]), 20, 18)
    Vw, UVw, Fw = grid_mesh(np.array([xl, yc, zf]), np.array([(x_right - xl) / 8, 0, 0]), np.array([0, -yc / 2, 0]), 8, 2)
    UVw = UVw * [nwx / atlas.shape[1], 1.0]
    Vl, UVl, Fl = grid_mesh(np.array([xl, yc, zf]), np.array([0, 0, (z_near - zf) / 6]), np.array([0, -yc / 2, 0]), 6, 2)
    UVl = UVl * [nlz / atlas.shape[1], 1.0] + [nwx / atlas.shape[1], 0.0]
    V_room, UV_room, F_room = np.concatenate([Vw, Vl]), np.concatenate([UVw, UVl]), np.concatenate([Fw, Fl + len(Vw)])
    write_obj(RS.ROOM_DIR / "room_mesh.obj", {"floor": (Vf, UVf, Ff), "room": (V_room, UV_room, F_room)},
              {"floor": "room_floor_tex.png", "room": "room_mesh_tex.png"})
    rep = {"floor_texture": {"px": [nx, nz], "m_per_px": res, "coverage": float(cover_f.mean())},
           "far_wall_texture": {"px": [nwx, nwy], "m_per_px": res, "coverage": float(cover_w.mean())},
           "left_wall_texture": {"px": [nlz, nwy], "m_per_px": res, "coverage": float(cover_l.mean())},
           "mesh": {"floor_triangles": int(len(Ff)), "room_triangles": int(len(F_room)),
                    "files": ["room_mesh.obj", "room_mesh.mtl", "room_floor_tex.png", "room_mesh_tex.png"]}}
    (RS.ROOM_DIR / "mesh_report.json").write_text(json.dumps(rep, indent=1))
    np.savez_compressed(RS.RW / "tex_cover.npz", floor=cover_f, far=cover_w, left=cover_l)
    RS.log(json.dumps(rep))
    return {"model": model, "report": rep}
