r"""Render track 'avatar_ags' of the appearance comparison (2026-10-19): the CURRENT shipped avatar and the AGS 8-phone
splats rendered at every camera / time of compare/views.json (appearance_compare.py spec), for the owner's picture
side-by-sides ("we need to decide between avatar and 3d gauss splat moving ... the rendering needs to be improved").

    python -m dancecap.mono4d.appearance_avatar_ags render --method avatar     [--groups real,turntable,...] [--ss 2]
    python -m dancecap.mono4d.appearance_avatar_ags render --method ags_8phone [--groups ...]
    python -m dancecap.mono4d.appearance_avatar_ags check     # placement sheets + IoU numbers -> renders/_check/

Outputs (work/<take>/mono4d/compare/renders/<method>/, RGBA PNG with a transparent background, pinhole cameras: the
real phones' k1 is ignored, <= ~7 px at the corners of phone 02):
    <view_id>.png               real views i<k>_c<cam> (native size), turntable tt00..tt11 (1080 x 1080),
                                vr_seat_i<k> (the VR seat at each instant, 1080 x 1080)
    cu_<instant>_c<cam>_<box>.png  the spec's close-up squares (face_lead / face_follow / hair_follow on phones 02, 06)
                                rendered through a zoomed camera, longer side CU_SIZE px (the render's own detail; the
                                photo crop at native size is the same box of <view_id>.png)
    clip_vr_seat/<n:04d>.png    the 105 clip frames n = 0..104 (grid k 836..940) from the VR seat
    clip_cam03/<n:04d>.png      the same 105 instants along phone 03's own track (real frames, native 480 x 848)
    roles/<id>.png              per-pixel owner of the still renders (0 none, 1 lead, 2 follow incl. hair)
    manifest.json               what was rendered, with which inputs, per-view times / cameras / colour transform /
                                placement IoU vs the SAM masks (real views)

Methods
    avatar      the assets the Unity export ships for this take (export/04_LarissaKaduDemo: smplx_<role>_skin.bin = the
                shaped + shape-edited rest skin split at the UV seams, <role>_albedo.png = avatar/<role>_albedo_face.png,
                the 'face' portrait albedo variant), skinned exactly like SmplxAvatar.cs (export_unity.fk_unity /
                lbs_unity: plain LBS, no pose correctives) with the reference fit smplx/<role>.npz (MAMMA) at the view's
                time (vertices lerped between the bracketing 30 Hz grid frames), rasterised with
                avatar_texture.render (both dancers, mutual occlusion, unlit albedo), ss x ss supersampled (Unity
                renders with MSAA + mip-maps). Unity adds hair cards with tip bloom, procedural shoes, eye blinks and
                35 % scene shading of the albedo (HM_AvatarLit _LightInfluence 0.35: main light + SH ambient).
    ags_8phone  dancer_splats (take larissa_kadu, the same source videos) model.pt (+ its pose corrections, the same
                values as pose_delta.npz) through evaluate.render_view (gsplat, antialiased) in the other take's world
                and clock: real views and phone 03's clip use the other take's OWN camera and time of the same photo
                (views.json real_views[*].lk, clip.cam03.frames[*].lk_*; AGS's bundle cameras equal those tracks); novel
                cameras use the body-fitted Sim3 world_38 -> world_lk (views.json lk blocks: R2 = R Rs^T,
                t2 = s tvec - R2 ts) at t_lk = t - 6 ms. Colour: a training photo gets its own trained colour transform,
                a held-out photo of a training phone its phone's mean, phone 03 and novel cameras the identity (AGS's
                own evaluation convention). AGS poses the bodies with ITS take's fit + its pose corrections, not with
                larissa_kadu_38's reference fit (mean joint difference 12.5 mm after the Sim3).
Likeness data: everything stays under work/ (git-ignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT
from .appearance_compare import AVATAR_CAPTION, ROLES, Fit, _rel, compare_dir, project, read_mask

SCHEMA = "dancecap.mono4d.appearance_compare.renders/1"
METHODS = ("avatar", "ags_8phone")
GROUPS = ("real", "turntable", "vr_seat", "closeup", "clip_vr_seat", "clip_cam03")
CAPTURE = "04_LarissaKaduDemo"          # the Unity capture of larissa_kadu_38 (the shipped avatar assets)
OTHER = "larissa_kadu"                  # AGS's take
CU_SIZE = 512                           # close-up renders: longer side (px)
COL = {"lead": (60, 220, 60), "follow": (60, 60, 240)}  # BGR


# ------------------------------------------------------------------------------------------------ spec -> render jobs
def load_spec(take: str) -> dict:
    return json.loads((compare_dir(take) / "views.json").read_text())


def _cam(K, R, tvec, W, H) -> dict:
    return {"K": np.asarray(K, float), "R": np.asarray(R, float), "tvec": np.asarray(tvec, float),
            "W": int(W), "H": int(H)}


def zoom_K(K, box, out_w: int, out_h: int) -> np.ndarray:
    """K of the pixel box [x0, y0, x1, y1) resampled to out_w x out_h (pixel centres at integers: u' = (u - x0 + 0.5)
    s - 0.5). Also the supersampling K (box = the whole image)."""
    K = np.asarray(K, float)
    x0, y0, x1, y1 = box
    sx, sy = out_w / (x1 - x0), out_h / (y1 - y0)
    return np.array([[sx * K[0, 0], 0.0, sx * (K[0, 2] - x0 + 0.5) - 0.5],
                     [0.0, sy * K[1, 1], sy * (K[1, 2] - y0 + 0.5) - 0.5], [0.0, 0.0, 1.0]])


def build_jobs(spec: dict, groups=GROUPS) -> list[dict]:
    """Every render the spec asks for. Each job: id, group, file (relative to the method dir), t (this take's
    reference seconds), cam (this take's world), lk (the other take: t, K, R, tvec, phone, frame; K None = look it up
    in AGS's bundle), meta."""
    jobs = []
    real = {v["id"]: v for v in spec["real_views"]}
    if "real" in groups:
        for v in spec["real_views"]:
            lk = v["lk"]
            jobs.append({"id": v["id"], "group": "real", "file": f"{v['id']}.png", "t": float(v["t"]),
                         "cam": _cam(v["K"], v["R"], v["tvec"], v["W"], v["H"]),
                         "lk": {"t": float(lk["t"]), "K": np.asarray(lk["K"], float), "R": np.asarray(lk["R"], float),
                                "tvec": np.asarray(lk["tvec"], float), "phone": v["cam"], "frame": int(lk["frame"])},
                         "meta": {"instant": v["instant"], "phone": v["cam"], "frame": v["frame"], "photo": v["file"],
                                  "both_well": v["both_well"], "seen_by": v["seen_by"], "lk_frame": lk["frame"],
                                  "lk_photo": lk["file"]}})
    tt = spec["novel"]["turntable"]
    if "turntable" in groups:
        for v in tt["views"]:
            jobs.append({"id": v["id"], "group": "turntable", "file": f"{v['id']}.png", "t": float(tt["t"]),
                         "cam": _cam(v["K"], v["R"], v["tvec"], v["W"], v["H"]),
                         "lk": {"t": float(v["lk"]["t"]), "K": np.asarray(v["K"], float),
                                "R": np.asarray(v["lk"]["R"], float), "tvec": np.asarray(v["lk"]["tvec"], float),
                                "phone": None, "frame": None},
                         "meta": {"instant": tt["instant"], "azimuth_from_cam03_deg": v["azimuth_from_cam03_deg"]}})
    vr = spec["novel"]["vr_seat"]
    vr_cam = _cam(vr["K"], vr["R"], vr["tvec"], vr["W"], vr["H"])
    vr_lk = {"K": np.asarray(vr["K"], float), "R": np.asarray(vr["lk"]["R"], float),
             "tvec": np.asarray(vr["lk"]["tvec"], float), "phone": None, "frame": None}
    if "vr_seat" in groups:
        for x in spec["instants"]:
            jobs.append({"id": f"vr_seat_{x['id']}", "group": "vr_seat", "file": f"vr_seat_{x['id']}.png",
                         "t": float(x["t"]), "cam": vr_cam, "lk": {**vr_lk, "t": float(x["t_other"])},
                         "meta": {"instant": x["id"]}})
    if "closeup" in groups:
        for c in spec["closeups"]:
            for phone, cc in c["cams"].items():
                v = real[cc["view_id"]]
                for name, b in cc["boxes"].items():
                    if not b.get("box"):
                        continue
                    x0, y0, x1, y1 = b["box"]
                    s = CU_SIZE / max(x1 - x0, y1 - y0)
                    ow, oh = int(round((x1 - x0) * s)), int(round((y1 - y0) * s))
                    cid = f"cu_{c['instant']}_c{phone}_{name}"
                    jobs.append({"id": cid, "group": "closeup", "file": f"{cid}.png", "t": float(v["t"]),
                                 "cam": _cam(zoom_K(v["K"], b["box"], ow, oh), v["R"], v["tvec"], ow, oh),
                                 "lk": {"t": float(v["lk"]["t"]), "K": zoom_K(v["lk"]["K"], b["box"], ow, oh),
                                        "R": np.asarray(v["lk"]["R"], float), "tvec": np.asarray(v["lk"]["tvec"], float),
                                        "phone": phone, "frame": int(v["lk"]["frame"])},
                                 "meta": {"instant": c["instant"], "phone": phone, "view_id": v["id"], "box_name": name,
                                          "box": b["box"], "box_side_px": b["side_px"], "clipped": b["clipped"],
                                          "best_cam": c["best_cam"], "best_cam_hair": c["best_cam_hair"],
                                          "zoom": round(s, 3)}})
    clip = spec["clip"]
    if "clip_vr_seat" in groups:
        for n, (t, to) in enumerate(zip(clip["t"], clip["t_other"])):
            jobs.append({"id": f"clip_vr_seat/{n:04d}", "group": "clip_vr_seat", "file": f"clip_vr_seat/{n:04d}.png",
                         "t": float(t), "cam": vr_cam, "lk": {**vr_lk, "t": float(to)},
                         "meta": {"n": n, "k": clip["k"][0] + n}})
    if "clip_cam03" in groups:
        c3 = clip["cam03"]
        for n, fr in enumerate(c3["frames"]):
            jobs.append({"id": f"clip_cam03/{n:04d}", "group": "clip_cam03", "file": f"clip_cam03/{n:04d}.png",
                         "t": float(fr["t"]), "cam": _cam(fr["K"], fr["R"], fr["tvec"], c3["W"], c3["H"]),
                         "lk": {"t": fr["lk_t"], "K": None, "R": None if fr["lk_R"] is None else np.asarray(fr["lk_R"]),
                                "tvec": None if fr["lk_tvec"] is None else np.asarray(fr["lk_tvec"]),
                                "phone": "03", "frame": fr["lk_frame"]},
                         "meta": {"n": n, "k": clip["k"][0] + n, "frame": fr["frame"], "photo": fr["file"],
                                  "dt_ms": fr["dt_ms"], "lk_frame": fr["lk_frame"]}})
    return jobs


# ------------------------------------------------------------------------------------------------ image helpers
def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> None:
    """rgb (H, W, 3) straight (un-premultiplied) float sRGB, alpha (H, W) float -> 8-bit RGBA PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(alpha, 0, 1)
    rgb = np.where(a[..., None] > 1 / 512, np.clip(rgb, 0, 1), 0.0)
    out = np.concatenate([rgb[..., ::-1], a[..., None]], -1)
    cv2.imwrite(str(path), (out * 255 + 0.5).astype(np.uint8))


def write_roles(path: Path, roles: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), roles.astype(np.uint8))


def placement(alpha: np.ndarray, roles: np.ndarray, work: Path, phone: str, frame: int) -> dict | None:
    """IoU of the render's silhouette (alpha > 0.5) with the union of the SAM masks of the photo, and per dancer
    (roles == 1 / 2 vs mask 01 / 02; the follow's mask includes her hair, the avatar has none)."""
    H, W = alpha.shape
    m = {r: read_mask(work, phone, frame, r, (H, W)) for r in ROLES}
    if any(x is None for x in m.values()):
        return None
    sil = alpha > 0.5
    uni = m["lead"] | m["follow"]

    def iou(a, b):
        u = (a | b).sum()
        return round(float((a & b).sum() / u), 3) if u else None
    return {"iou_union": iou(sil, uni), "iou_lead": iou(roles == 1, m["lead"]), "iou_follow": iou(roles == 2, m["follow"]),
            "mask_px": int(uni.sum()), "render_px": int(sil.sum())}


# ------------------------------------------------------------------------------------------------ the current avatar
class _TexGeo:
    """The bits of avatar_texture._Geo that avatar_texture.render reads, for a seam-split skin whose texture is the
    shipped albedo PNG itself (texture() returns it: no per-texel colours)."""

    def __init__(self, torch, faces: np.ndarray, uv: np.ndarray, device):
        self.faces = torch.as_tensor(np.array(faces), dtype=torch.long, device=device)
        self.ft = self.faces
        self.vt = torch.as_tensor(np.array(uv), dtype=torch.float32, device=device)

    def texture(self, img):
        return img


class AvatarRenderer:
    method = "avatar"

    def __init__(self, take: str, device: str = "cuda", ss: int = 2, capture: str = CAPTURE):
        import torch

        from ..export_unity import MIRROR, fk_unity, full_pose, read_skin, rotvec_to_quat, to_unity_points, to_unity_quats
        self.torch, self.dev, self.ss = torch, device, int(ss)
        self.work = PACKAGE_ROOT / "work" / take
        exp = self.work / "export" / capture
        cap = json.loads((exp / "capture.json").read_text())
        self.mirror = MIRROR
        self.skin, self.P, self.R, self.t = {}, {}, {}, {}
        self.alb, self.inputs, self.fk_err_mm = {}, [], {}
        faces = uv = None
        for r in ROLES:
            skin_p = exp / cap["smplx_skin"][r]
            alb_p = exp / cap["smplx_albedo"][r]
            fit_p = self.work / "smplx" / f"{r}.npz"
            _, sk = read_skin(skin_p)
            with np.load(fit_p, allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
            if not np.allclose(np.asarray(sk.betas, float), np.asarray(d["betas"], float), atol=1e-5):
                raise ValueError(f"{r}: the shipped skin's betas differ from smplx/{r}.npz")
            quats_u = to_unity_quats(rotvec_to_quat(full_pose(d)))
            transl_u = to_unity_points(np.asarray(d["transl"], float))
            P, R = fk_unity(sk.joints.astype(float), sk.parents, transl_u, quats_u)
            self.fk_err_mm[r] = round(1e3 * float(np.abs(P - to_unity_points(np.asarray(d["joints"], float)[:, :55])).max()), 3)
            self.skin[r], self.P[r], self.R[r], self.t[r] = sk, P, R, np.asarray(d["t"], float)
            f_world = np.asarray(sk.faces)[:, [0, 2, 1]]       # back to the right-handed winding
            if faces is None:
                faces, uv = f_world, np.asarray(sk.uv)
            elif not (np.array_equal(faces, f_world) and np.allclose(uv, sk.uv)):
                raise ValueError("the two shipped skins have different topology / UVs")
            img = cv2.imread(str(alb_p), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
            self.alb[r] = torch.as_tensor(img.transpose(2, 0, 1).copy(), device=device)
            for p in (skin_p, alb_p, fit_p):
                self.inputs.append({"file": _rel(p), "md5": _md5(p), "bytes": p.stat().st_size})
        self.geo = _TexGeo(torch, faces, uv, device)
        self._cache: dict = {}
        self.albedo_variant = cap.get("smplx_albedo_variant")
        self.unity_motion_dir = cap.get("provenance", {}).get("smplx_dir")

    def _lbs(self, r: str, k: int) -> np.ndarray:
        key = (r, k)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            sk, P, R = self.skin[r], self.P[r][k], self.R[r][k]
            rel = sk.vertices[:, None, :].astype(float) - sk.joints[None].astype(float)          # (V, 55, 3)
            posed = np.einsum("jab,vjb->vja", R, rel) + P[None]
            self._cache[key] = np.einsum("vj,vja->va", sk.weights.astype(float), posed) * self.mirror  # world frame
        return self._cache[key]

    def verts(self, r: str, t: float) -> np.ndarray:
        tg = self.t[r]
        k = int(np.clip(np.searchsorted(tg, t) - 1, 0, len(tg) - 2))
        a = float(np.clip((t - tg[k]) / (tg[k + 1] - tg[k]), 0.0, 1.0))
        return (1 - a) * self._lbs(r, k) + a * self._lbs(r, k + 1)

    def render(self, job: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        from ..avatar_texture import render as tex_render
        c, ss = job["cam"], self.ss
        W, H = c["W"], c["H"]
        Ks = zoom_K(c["K"], (0, 0, W, H), W * ss, H * ss)
        posed = {r: self.verts(r, job["t"]) for r in ROLES}
        with self.torch.no_grad():
            img, idm = tex_render(self.torch, self.geo, posed, self.alb, Ks, c["R"], c["tvec"], 0.0, H * ss, W * ss,
                                  self.dev)
        cov = (idm >= 0).astype(np.float32)
        area = cv2.INTER_AREA
        prem = cv2.resize(img.astype(np.float32) * cov[..., None], (W, H), interpolation=area)
        alpha = cv2.resize(cov, (W, H), interpolation=area)
        rgb = prem / np.maximum(alpha, 1e-6)[..., None]
        own = np.stack([cv2.resize((idm == n).astype(np.float32), (W, H), interpolation=area) for n in range(2)], -1)
        roles = np.where(alpha > 0.5, 1 + own.argmax(-1), 0).astype(np.uint8)
        return rgb, alpha, roles, {"t_posed": round(job["t"], 5), "ss": ss}

    def describe(self) -> dict:
        return {"what": "the CURRENT shipped avatar: the Unity export's shaped + shape-edited SMPL-X skin "
                        f"(export/{CAPTURE}/smplx_<role>_skin.bin) with its albedo (<role>_albedo.png = "
                        "avatar/<role>_albedo_face.png, the 'face' portrait variant), plain LBS exactly like "
                        "SmplxAvatar.cs, unlit, rasterised with avatar_texture.render (both dancers, mutual occlusion)",
                "caption": AVATAR_CAPTION,
                "unity_adds": "hair cards with tip bloom (avatar/hair_groom.json), procedural shoes (shoes.json), "
                              "procedural eye blinks, and partial scene shading: HM_AvatarLit.shader multiplies the "
                              "albedo by lerp(1, main light x wrap Lambert (wrap 0.5) + SH ambient, _LightInfluence) "
                              "with _LightInfluence 0.35 for textured bodies (SmplxAvatar.cs); this offline render is "
                              "unlit",
                "pose": "reference fit smplx/<role>.npz (MAMMA locked-head SMPL-X; head fix not adopted) at the view's "
                        "time, vertices lerped between the bracketing 30 Hz grid frames (Unity plays "
                        f"{self.unity_motion_dir}/, the refined motion; the comparison poses every method with the "
                        "reference fit)",
                "fk_vs_fit_joints_mm": self.fk_err_mm, "albedo_variant": self.albedo_variant,
                "unity_reference": [_rel(p) for p in sorted((self.work / "review").glob("unity_closeup_*_t23.0s.png"))],
                "unity_reference_note": "real Unity screenshots of these avatars (04_LarissaKaduDemo, 2026-10-07, "
                                        "scene 23.0 s, before the 2026-10-08 face-shape edit, HUD overlays)",
                "antialiasing": f"{self.ss} x {self.ss} supersampled, box-filtered (Unity: MSAA + mip-maps)",
                "inputs": self.inputs,
                "spec_note": "views.json renderers.avatar.inputs lists avatar/<role>_albedo.png (the plain bake); the "
                             "export ships avatar/<role>_albedo_face.png (md5 equal to export/" + CAPTURE +
                             "/<role>_albedo.png) - this track renders the shipped one"}


# ------------------------------------------------------------------------------------------------ AGS splats
class AGSRenderer:
    method = "ags_8phone"

    def __init__(self, take: str, device: str = "cuda"):
        import torch

        from ..dancer_splats.data import load_bundle
        from ..dancer_splats.model import AGSModel
        self.torch = torch
        self.work = PACKAGE_ROOT / "work" / take
        out = PACKAGE_ROOT / "work" / OTHER / "dancer_splats"
        self.B = load_bundle(out / "bundle")
        self.model = AGSModel.load(self.B, out / "model.pt", device)
        self.by_cam = self.model.cam_app_index()
        self.sh = self.model.cfg.sh_degree
        self.bview = {(v["cam"], int(v["frame"])): v for v in self.B["views"]}
        self.inputs = [{"file": _rel(p), "md5": _md5(p), "bytes": p.stat().st_size}
                       for p in (out / "model.pt", out / "pose_delta.npz", out / "bundle" / "bodies.npz",
                                 out / "bundle" / "views.json")]
        self.tg = np.asarray(self.model.t_grid, float)

    def _view(self, job: dict) -> tuple[dict, str]:
        lk, c = job["lk"], job["cam"]
        v = {"i": -1, "cam": "novel", "split": "novel", "t": float(lk["t"]), "W": c["W"], "H": c["H"]}
        K, R, tv = lk["K"], lk["R"], lk["tvec"]
        bv = self.bview.get((lk["phone"], int(lk["frame"]))) if lk.get("frame") is not None else None
        if bv is not None:
            v.update({"i": int(bv["i"]), "cam": bv["cam"], "split": bv["split"]})
            if K is None:  # the bundle crop's K + its crop origin = the other take's full-frame K
                K = np.asarray(bv["K"], float).copy()
                K[0, 2] += bv["crop"][0]
                K[1, 2] += bv["crop"][1]
            if R is None:
                R, tv = np.asarray(bv["R"], float), np.asarray(bv["t_w"], float)
        if K is None or R is None:
            raise ValueError(f"{job['id']}: no camera of the other take")
        v.update({"K": np.asarray(K, float).tolist(), "R": np.asarray(R, float).tolist(),
                  "t_w": np.asarray(tv, float).tolist()})
        j = self.model.app_index.get(v["i"])
        ct = ("own (trained view)" if j is not None and v["split"] == "train"
              else "phone mean" if self.by_cam.get(v["cam"]) else "identity")
        return v, ct

    def render(self, job: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        from ..dancer_splats.evaluate import render_view
        v, ct = self._view(job)
        if not (self.tg[0] <= v["t"] <= self.tg[-1]):
            raise ValueError(f"{job['id']}: t_lk {v['t']:.3f} outside AGS's grid {self.tg[0]:.3f}-{self.tg[-1]:.3f}")
        r = render_view(self.model, v, self.by_cam, self.sh)
        alpha = r["alpha"]
        rgb = r["rgb"] / np.maximum(alpha, 1e-6)[..., None]       # composite on black = premultiplied
        ro = r["roles"]                                           # lead, follow body, hair alphas
        own = np.stack([ro[..., 0], ro[..., 1] + ro[..., 2]], -1)
        roles = np.where(alpha > 0.5, 1 + own.argmax(-1), 0).astype(np.uint8)
        return rgb, alpha, roles, {"t_lk": round(v["t"], 5), "colour_transform": ct, "bundle_view": v["i"],
                                   "bundle_split": v["split"],
                                   "lk_camera": {"K": np.round(v["K"], 6).tolist(), "R": np.round(v["R"], 9).tolist(),
                                                 "tvec": np.round(v["t_w"], 6).tolist()},
                                   "lk_C": np.round(-np.asarray(v["R"]).T @ np.asarray(v["t_w"]), 4).tolist()}

    def describe(self) -> dict:
        return {"what": "AGS avatar-guided dancer splats (dancer_splats, 8 phones, trained 2026-10-07 on take "
                        "larissa_kadu 58.0-61.5 s = the same source videos), gsplat antialiased, local 2080 Ti",
                "gaussians": {k: len(g) for k, g in self.model.groups.items()},
                "pose": "AGS's own take's fit (larissa_kadu smplx/ in the bundle) + its trained pose corrections "
                        "(model.pt; pose_delta.npz is the same values exported) - NOT larissa_kadu_38's reference fit",
                "mapping": "real views / phone 03's clip: the other take's own camera and time of the same photo "
                           "(views.json lk, per frame); novel cameras (turntable, VR seat): the body-fitted Sim3 "
                           "world_larissa_kadu_38 -> world_larissa_kadu (s 1.04518, 1.24 deg, mean joint residual "
                           "12.5 mm) at t_lk = t - 6 ms",
                "colour": "a training photo: its own trained colour transform; a held-out photo of a training phone: "
                          "the phone's mean transform; phone 03 and novel cameras: identity (AGS eval convention)",
                "inputs": self.inputs}


# ------------------------------------------------------------------------------------------------ driver
def method_dir(take: str, method: str) -> Path:
    return compare_dir(take) / "renders" / method


def _load_manifest(p: Path) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def render_method(take: str, method: str, groups=GROUPS, ss: int = 2, device: str = "cuda", limit: int | None = None,
                  ids: list[str] | None = None, log=print) -> dict:
    from .. import gpu
    spec = load_spec(take)
    jobs = build_jobs(spec, groups)
    if ids:
        jobs = [j for j in jobs if j["id"] in ids]
    if limit:
        jobs = jobs[:limit]
    out = method_dir(take, method)
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "manifest.json"
    man = _load_manifest(mp)
    work = PACKAGE_ROOT / "work" / take
    t0 = time.time()
    with gpu.exclusive(f"compare-{method}", timeout_s=1800):
        rend = AvatarRenderer(take, device, ss) if method == "avatar" else AGSRenderer(take, device)
        log(f"[compare-{method}] {len(jobs)} renders ({', '.join(groups)}), loaded in {time.time() - t0:.1f} s")
        views = man.get("views", {})
        for n, j in enumerate(jobs):
            ts = time.time()
            rgb, alpha, roles, info = rend.render(j)
            write_rgba(out / j["file"], rgb, alpha)
            rec = {"group": j["group"], "file": j["file"], "W": j["cam"]["W"], "H": j["cam"]["H"],
                   "t": round(j["t"], 5), "lk_t": None if j["lk"]["t"] is None else round(float(j["lk"]["t"]), 5),
                   **info, "meta": j["meta"], "coverage_px": int((alpha > 0.5).sum()),
                   "seconds": round(time.time() - ts, 2)}
            if not j["group"].startswith("clip"):
                write_roles(out / "roles" / j["file"], roles)
                rec["roles_file"] = f"roles/{j['file']}"
                rec["camera"] = {"K": np.round(j["cam"]["K"], 6).tolist(), "R": np.round(j["cam"]["R"], 9).tolist(),
                                 "tvec": np.round(j["cam"]["tvec"], 6).tolist()}
            if j["group"] == "real":
                rec["placement_vs_sam_masks"] = placement(alpha, roles, work, j["meta"]["phone"], j["meta"]["frame"])
            views[j["id"]] = rec
            if n % 20 == 0 or n == len(jobs) - 1:
                log(f"[compare-{method}] {n + 1}/{len(jobs)} {j['id']} {rec['seconds']:.2f} s")
                if method == "ags_8phone":
                    rend.torch.cuda.empty_cache()
        desc = rend.describe()
        rend.torch.cuda.empty_cache()
    man.update({"schema": SCHEMA, "method": method, "take": take, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "code": "dancecap/dancecap/mono4d/appearance_avatar_ags.py", "spec": _rel(compare_dir(take) / "views.json"),
                "spec_created": spec["created"], "conventions": {
                    "image": "RGBA PNG, straight (un-premultiplied) colour, transparent background; roles/<id>.png "
                             "0 none, 1 lead, 2 follow (incl. her hair)",
                    "camera": "pinhole: the real phones' k1 (SIMPLE_RADIAL) is ignored (<= ~7 px at phone 02's corners)",
                    "clip_index": "clip_<camera>/<n:04d>.png, n = 0..104 = views.json clip.t[n] (grid k 836 + n); "
                                  "clip_cam03 frame n = clip.cam03.frames[n] (its own photo time)",
                    "closeups": f"cu_<instant>_c<phone>_<box>.png: the spec's box through a zoomed camera, longer side "
                                f"{CU_SIZE} px; the photo-resolution crop is the same box of <view_id>.png"},
                **desc})
    man["views"] = dict(sorted(views.items()))
    man["counts"] = {g: sum(1 for v in man["views"].values() if v["group"] == g) for g in GROUPS}
    pl = [v["placement_vs_sam_masks"] for v in man["views"].values() if v.get("placement_vs_sam_masks")]
    if pl:
        man["placement_summary"] = {k: round(float(np.median([p[k] for p in pl if p[k] is not None])), 3)
                                    for k in ("iou_union", "iou_lead", "iou_follow")}
    mp.write_text(json.dumps(man, indent=1))
    log(f"[compare-{method}] done {len(jobs)} renders in {time.time() - t0:.0f} s -> {out}")
    return man


# ------------------------------------------------------------------------------------------------ checks
def _read_rgba(p: Path) -> np.ndarray | None:
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return None if im is None else im


def _over(bg: np.ndarray, rgba: np.ndarray | None, k: float = 1.0) -> np.ndarray:
    if rgba is None:
        return bg.copy()
    a = rgba[..., 3:4].astype(np.float32) / 255 * k
    return (bg.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a).astype(np.uint8)


def _lab(img, text, org=(6, 20), s=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _skel(img, fits, t, K, R, tv, col_scale=1.0):
    from .appearance_compare import SMPL_PARENTS
    for r in ROLES:
        uv, z = project(fits[r].joints(t)[:22], K, R, tv)
        for j, p in enumerate(SMPL_PARENTS):
            if p >= 0 and z[j] > 0 and z[p] > 0:
                cv2.line(img, tuple(int(x) for x in uv[j]), tuple(int(x) for x in uv[p]), COL[r],
                         max(1, int(2 * col_scale)), cv2.LINE_AA)


def _iou(a: np.ndarray, b: np.ndarray) -> float | None:
    u = (a | b).sum()
    return round(float((a & b).sum() / u), 3) if u else None


def check(take: str, log=print) -> dict:
    """renders/_check/: per real view photo | avatar over photo | AGS over photo (reference joints + mask outlines);
    novel views avatar | AGS | silhouette overlap; clip strips. Numbers -> renders/_check/check.json."""
    spec = load_spec(take)
    work = PACKAGE_ROOT / "work" / take
    root = compare_dir(take) / "renders"
    out = root / "_check"
    out.mkdir(parents=True, exist_ok=True)
    fits = {r: Fit(work / "smplx" / f"{r}.npz") for r in ROLES}
    res: dict = {"real": {}, "novel_avatar_vs_ags_iou": {}, "novel_avatar_inside_ags": {}, "clip_avatar_vs_ags_iou": {},
                 "clip_cam03_iou_vs_sam_masks": {m: {} for m in METHODS},
                 "how": "IoU of silhouettes (alpha > 0.5); avatar_inside_ags = fraction of the avatar's silhouette that "
                        "AGS also covers (AGS has the follow's hair and a halo the avatar lacks, so its silhouette is "
                        "bigger: this is the mapping check for the Sim3-mapped novel cameras)"}
    by_inst: dict = {}
    for v in spec["real_views"]:
        photo = cv2.imread(str(PACKAGE_ROOT / v["file"]))
        s = max(1, int(round(max(v["W"], v["H"]) / 960)))
        tiles = []
        for m in ("photo",) + METHODS:
            img = photo.copy() if m == "photo" else _over(photo, _read_rgba(root / m / f"{v['id']}.png"), 0.85)
            for r in ROLES:
                mk = read_mask(work, v["cam"], v["frame"], r, (v["H"], v["W"]))
                if mk is not None:
                    cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cs, -1, COL[r], s, cv2.LINE_AA)
            if m != "photo":
                _skel(img, fits, v["t"], v["K"], v["R"], v["tvec"], s)
            x0, y0, x1, y1 = v["crop"] or [0, 0, v["W"], v["H"]]
            tile = img[y0:y1, x0:x1]
            tile = cv2.resize(tile, (int(round(tile.shape[1] * 520 / tile.shape[0])), 520), interpolation=cv2.INTER_AREA)
            tiles.append(_lab(tile, f"{v['id']} {m}"))
        sheet = np.hstack(tiles)
        cv2.imwrite(str(out / f"{v['id']}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        by_inst.setdefault(v["instant"], []).append(sheet)
        res["real"][v["id"]] = {m: (json.loads((root / m / "manifest.json").read_text())["views"].get(v["id"], {})
                                    .get("placement_vs_sam_masks") if (root / m / "manifest.json").exists() else None)
                                for m in METHODS}
    for k, rows in by_inst.items():
        W = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.zeros((r.shape[0], W - r.shape[1], 3), np.uint8)]) for r in rows]
        cv2.imwrite(str(out / f"sheet_{k}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
    # novel stills: the two methods' silhouettes on the same camera (mapping check: AGS goes through the Sim3)
    tiles = []
    nov = [(v["id"], v) for v in spec["novel"]["turntable"]["views"]] + \
          [(f"vr_seat_{x['id']}", spec["novel"]["vr_seat"]) for x in spec["instants"]]
    for vid, cam in nov:
        ims = {m: _read_rgba(root / m / f"{vid}.png") for m in METHODS}
        if any(i is None for i in ims.values()):
            continue
        sa, sg = ims["avatar"][..., 3] > 127, ims["ags_8phone"][..., 3] > 127
        res["novel_avatar_vs_ags_iou"][vid] = _iou(sa, sg)
        res["novel_avatar_inside_ags"][vid] = round(float((sa & sg).sum() / max(sa.sum(), 1)), 3)
        bg = np.full(sa.shape + (3,), 40, np.uint8)
        ov = bg.copy()
        ov[sa] = (0, 180, 255)
        ov[sg] = (255, 120, 0)
        ov[sa & sg] = (230, 230, 230)
        row = [_over(bg, ims["avatar"]), _over(bg, ims["ags_8phone"]), ov]
        row = [cv2.resize(x, (300, 300), interpolation=cv2.INTER_AREA) for x in row]
        _lab(row[0], f"{vid} avatar")
        _lab(row[1], "ags_8phone")
        _lab(row[2], f"IoU {res['novel_avatar_vs_ags_iou'][vid]}")
        tiles.append(np.hstack(row))
    if tiles:
        cols = 2
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(out / "novel_avatar_vs_ags.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
    # clips: every 15th frame, photo / avatar / AGS on phone 03, avatar / AGS from the VR seat
    c3 = spec["clip"]["cam03"]["frames"]
    rows = {"photo03": [], "avatar03": [], "ags03": [], "avatar_vr": [], "ags_vr": []}
    for n in range(0, len(c3), 15):
        ph = cv2.imread(str(PACKAGE_ROOT / c3[n]["file"]))
        bgv = np.full((1080, 1080, 3), 40, np.uint8)
        a3 = _read_rgba(root / "avatar" / f"clip_cam03/{n:04d}.png")
        g3 = _read_rgba(root / "ags_8phone" / f"clip_cam03/{n:04d}.png")
        av = _read_rgba(root / "avatar" / f"clip_vr_seat/{n:04d}.png")
        gv = _read_rgba(root / "ags_8phone" / f"clip_vr_seat/{n:04d}.png")
        rows["photo03"].append(_lab(cv2.resize(ph, (240, 424)), f"03 n{n}"))
        rows["avatar03"].append(_lab(cv2.resize(_over(ph, a3, 0.85), (240, 424)), "avatar"))
        rows["ags03"].append(_lab(cv2.resize(_over(ph, g3, 0.85), (240, 424)), "ags"))
        rows["avatar_vr"].append(_lab(cv2.resize(_over(bgv, av), (240, 240)), f"VR n{n} avatar"))
        rows["ags_vr"].append(_lab(cv2.resize(_over(bgv, gv), (240, 240)), "VR ags"))
        for tag, (a, g) in (("cam03", (a3, g3)), ("vr_seat", (av, gv))):
            if a is not None and g is not None:
                res["clip_avatar_vs_ags_iou"][f"{tag}_{n:04d}"] = _iou(a[..., 3] > 127, g[..., 3] > 127)
    if rows["photo03"]:
        top = np.vstack([np.hstack(rows[k]) for k in ("photo03", "avatar03", "ags03")])
        bot = np.vstack([np.hstack(rows[k]) for k in ("avatar_vr", "ags_vr")])
        W = max(top.shape[1], bot.shape[1])
        pad = lambda x: np.hstack([x, np.zeros((x.shape[0], W - x.shape[1], 3), np.uint8)])  # noqa: E731
        cv2.imwrite(str(out / "clips.jpg"), np.vstack([pad(top), pad(bot)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
    for n, fr in enumerate(c3):  # every clip frame on phone 03 vs that photo's SAM masks (the track's placement)
        for m in METHODS:
            im = _read_rgba(root / m / f"clip_cam03/{n:04d}.png")
            if im is None:
                continue
            mk = [read_mask(work, "03", fr["frame"], r, im.shape[:2]) for r in ROLES]
            if all(x is not None for x in mk):
                res["clip_cam03_iou_vs_sam_masks"][m][f"{n:04d}"] = _iou(im[..., 3] > 127, mk[0] | mk[1])
    for k in ("novel_avatar_vs_ags_iou", "novel_avatar_inside_ags", "clip_avatar_vs_ags_iou"):
        vals = [x for x in res[k].values() if x is not None]
        res[k + "_median"] = round(float(np.median(vals)), 3) if vals else None
    res["clip_cam03_iou_vs_sam_masks_summary"] = {
        m: {"median": round(float(np.median(list(d.values()))), 3), "min": round(float(min(d.values())), 3),
            "n": len(d)} for m, d in res["clip_cam03_iou_vs_sam_masks"].items() if d}
    (out / "check.json").write_text(json.dumps(res, indent=1))
    for m in METHODS:  # the placement summary goes into each method's manifest too
        mp = root / m / "manifest.json"
        if mp.exists():
            man = json.loads(mp.read_text())
            man["check"] = {"file": _rel(out / "check.json"), "sheets": _rel(out),
                            "clip_cam03_iou_vs_sam_masks": res["clip_cam03_iou_vs_sam_masks_summary"].get(m),
                            "novel_avatar_inside_ags_median": res["novel_avatar_inside_ags_median"],
                            "novel_avatar_vs_ags_iou_median": res["novel_avatar_vs_ags_iou_median"]}
            mp.write_text(json.dumps(man, indent=1))
    log(f"[compare-check] -> {out}")
    return res


def _needs_cuda_env() -> bool:
    import os
    import shutil

    from ..env import IS_WINDOWS
    return bool(IS_WINDOWS and shutil.which("cl") is None and not os.environ.get("DANCECAP_CUDA_ENV"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.appearance_avatar_ags", description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["render", "check", "jobs"])
    ap.add_argument("--take", default="larissa_kadu_38")
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--ids", help="comma-separated view ids (render only these)")
    ap.add_argument("--ss", type=int, default=2, help="avatar supersampling factor")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    groups = tuple(g.strip() for g in a.groups.split(",") if g.strip())
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        ap.error(f"unknown groups {bad}; choose from {GROUPS}")
    if a.action == "jobs":
        js = build_jobs(load_spec(a.take), groups)
        print(json.dumps({g: sum(1 for j in js if j["group"] == g) for g in groups}))
    elif a.action == "render":
        if not a.method:
            ap.error("render needs --method")
        if a.method == "ags_8phone" and a.device == "cuda" and _needs_cuda_env():
            # gsplat's cached JIT extension still needs ninja + MSVC on PATH: re-run inside the CUDA build env
            import os
            import subprocess
            import sys

            from ..env import cuda_wrapper
            argv = sys.argv[1:] if argv is None else list(argv)
            r = subprocess.run(cuda_wrapper(["-m", "dancecap.mono4d.appearance_avatar_ags", *argv]), cwd=PACKAGE_ROOT,
                               env={**os.environ, "DANCECAP_CUDA_ENV": "1", "PYTHONUNBUFFERED": "1"})
            return r.returncode
        render_method(a.take, a.method, groups, a.ss, a.device, a.limit,
                      [x.strip() for x in a.ids.split(",")] if a.ids else None)
    else:
        check(a.take)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
