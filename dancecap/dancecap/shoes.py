"""Sneakers for the dancers' Unity avatars: per-dancer shoe style + colours -> avatar/shoes.json (+ the capture folders).

    python -m dancecap.shoes larissa_kadu                    -> avatar/shoes.json (baked albedo + video samples),
                                                                review/realism/shoes_palette.jpg, copies into every
                                                                capture of the take
    python -m dancecap.shoes larissa_kadu --export-only [--capture 02_LarissaKadu]
    python -m dancecap.shoes larissa_kadu --sheet            -> review/realism/shoes_vs_video.jpg from the Unity review
                                                                shots (head-movement Tools/playtest_shoes.ps1)

The geometry is NOT here: head-movement Assets/Shoes builds a procedural skinned sneaker per foot at load from that
dancer's fitted SMPL-X rest foot (SneakerBuilder.Measure), so the json carries only what the mesh cannot know - the
style (sole stack, toe spring) and the colours. The foot measurements below are reported for review only.

Inputs (nothing take-specific in the code): the capture's smplx_<role>_skin.bin + smplx_<role>.bin + times.json
(StreamingAssets/<capture>, Unity coordinates), work/<take>/avatar/<role>_albedo.png + <role>_coverage.png (the
texture bake), frames/<cam>/*.jpg + dancers/<cam>/masks (video), cameras/<cam>.npz + world_from_sfm.json; optional
takes/<take>.toml [dancers.<role>] shoe_style (style only - never colours in a tracked file) and shoe_geometry
(measured sole stack: {sole_heel_m, sole_fore_m, toe_spring_m, heel_bevel_m}, overrides the style's numbers; applied
by every export too, so `--export-only` picks a toml change up without re-sampling the colours), and
work/<take>/avatar/shoes_override.json ({"<role>": {"<field>": "#rrggbb" | number | style}}).

Size: the sneaker is built around the fitted rest foot, so its length / width follow the foot. To match the shoes
measured in the video (python -m dancecap.shoe_size), the exporter shrinks the rest feet (dancecap.shape_edits:
[dancers.<role>.shape_edits] shoe_length_m / foot_height_scale) - shoes.json carries no size.

Method (CPU, < 1 min)
  1. Foot sets: split vertices with w(ankle) + w(foot) >= 0.5 (SMPL-X joints 7/10 left, 8/11 right); sole plane =
     their lowest rest y; foot axis heel -> toe-midpoint from the SMPL-X landmark vertices. Bands by rest height h:
     SOLE = side-facing (|n_y| < 0.55) vertices with h < 2.2 cm, UPPER = 2.5..6 cm (any normal) plus the toe-box top
     (n_y > 0.5, x > 55 % of the foot), HEEL = x < 12 % and h 4..8 cm.
  2. Bake: the foot triangles of each band rasterised on the albedo's UV layout, gated by the coverage map -> texels.
  3. Video: every --step-s seconds of every camera, both dancers' feet posed by linear blend skinning (exactly
     SmplxAvatar's FK), projected (SIMPLE_RADIAL), front-facing triangles painted far-to-near (both dancers' feet
     occlude each other), pixels kept inside the dancer's own mask and outside the partner's (dilated); a view-foot
     counts only when >= 70 % of its projected foot lies in the own mask (the fit is on the shoe there).
  4. Colours: 2-means in linear RGB per band. upper = dominant cluster of the upper band; accent = the other cluster
     when it holds >= 10 % with dE76 >= 20; sole = dominant cluster of the sole band unless it is skin-like (hue 5-35
     deg, saturation > 0.25: red hair / skin bleed of the bake) - then the video's. Video colours are graded to the
     bake's colour space (per-channel gain from the brightest dancer's upper, bake / video), so shoes and body share
     the studio grade of the albedo. That grade comes from a shoe upper itself, so it cannot take the warm cast of the
     bake / studio light out of white parts: a bright low-chroma colour (L* >= 62, C* <= 22) is balanced to a neutral
     white (L* >= 91, C* <= 2) before the derived colours are made from it (neutral_whites). Each field records its
     source and a confidence.
  5. Style: toml shoe_style, else automatic - light upper + light sole -> court_platform (chunky cupsole), dark upper
     + light sole -> runner, else neutral.

shoes.json (git-ignored: work/ and StreamingAssets/ only - the colours are sampled from the dancers):
  {"format": "dancecap shoes", "version": 1, "dancers": {"<role>": {"style", "upper", "sole", "outsole", "accent",
   "lace", "lining", "heel_tab" (#rrggbb sRGB), "sole_heel_m", "sole_fore_m", "toe_spring_m", "heel_bevel_m",
   "source": {field: text}, "confidence": {field: 0..1}, "measured": {...}}}, "likeness": "..."}
"""
from __future__ import annotations

import argparse
import colorsys
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from .export_unity import MIRROR, Skin, fk_unity, read_motion, read_skin
from .take import WORKSPACE, Take

ROLES = ("lead", "follow")
PIDS = {"lead": 1, "follow": 2}
FOOT = {"L": (7, 10), "R": (8, 11)}  # (ankle, foot = ball of the foot) SMPL-X joints
LANDMARKS = {"L": {"big": 5770, "small": 5780, "heel": 8846}, "R": {"big": 8463, "small": 8474, "heel": 8635}}
UNITY_STREAMING = WORKSPACE / "head-movement" / "Assets" / "StreamingAssets"
VERSION = 1
FIELDS = ("upper", "sole", "sole_heel", "outsole", "accent", "lace", "lining", "heel_tab")

# sole stack / shape per style (metres). The outsole bottom is the fitted foot's sole plane: dancecap's world scale
# treats the fitted mesh sole as the shoe sole (CONTRACTS.md, shoe_sole_m), so nothing is ever added below it.
STYLES = {
    "court_platform": {"sole_heel_m": 0.040, "sole_fore_m": 0.035, "toe_spring_m": 0.006, "heel_bevel_m": 0.004},
    "runner": {"sole_heel_m": 0.032, "sole_fore_m": 0.022, "toe_spring_m": 0.012, "heel_bevel_m": 0.008},
    "neutral": {"sole_heel_m": 0.030, "sole_fore_m": 0.024, "toe_spring_m": 0.009, "heel_bevel_m": 0.005},
}
# generic palette when nothing is measured (also the Unity default): off-white leather, white sole, grey outsole
DEFAULT = {"upper": "#d9d6cf", "sole": "#ecebe6", "sole_heel": "#ecebe6", "outsole": "#5a5a5a", "accent": "#9a9a9a", "lace": "#d9d6cf",
           "lining": "#b8b4ac", "heel_tab": "#8a8a8a"}

GEOMETRY = ("sole_heel_m", "sole_fore_m", "toe_spring_m", "heel_bevel_m")  # toml shoe_geometry keys (metres)
GEOMETRY_RANGE = (0.0, 0.07)

SOLE_H = (0.0, 0.022)
UPPER_H = (0.025, 0.060)
HEEL_H = (0.040, 0.080)


# ------------------------------------------------------------------------------------------------ colour helpers

def srgb_to_linear(x):
    x = np.asarray(x, float)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, float), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def to_hex(rgb_lin) -> str:
    s = np.round(linear_to_srgb(rgb_lin) * 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*s)


def from_hex(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return srgb_to_linear(np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0)


def lab(rgb_lin) -> np.ndarray:
    """linear sRGB (..., 3) -> CIE L*a*b* (D65)."""
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = np.asarray(rgb_lin, float) @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_linear(L, a, b) -> np.ndarray:
    """CIE L*a*b* (D65) -> linear sRGB (clipped to 0..1)."""
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    d = 6 / 29
    xyz = np.array([fx, fy, fz])
    xyz = np.where(xyz > d, xyz ** 3, 3 * d * d * (xyz - 4 / 29)) * np.array([0.95047, 1.0, 1.08883])
    Minv = np.linalg.inv(np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]))
    return np.clip(Minv @ xyz, 0, 1)


# White sneaker parts read warm in the texture bake and in the warm-lit video (and the video -> bake grade is taken from
# a shoe upper itself, so it cannot take the cast out): a bright, low-chroma colour is a white part seen under warm
# light. It is balanced to a neutral white - chroma down to WHITE_C_MAX, lightness up to WHITE_L_MIN (the video's white
# toes read L* 87-96 with blue/red ~0.95 next to the dancers' light-blue jeans) - and the colours derived from it follow.
WHITE_L, WHITE_C = 62.0, 22.0  # L* at least, C* at most: a white part under warm light
WHITE_L_MIN, WHITE_C_MAX = 91.0, 2.0
def neutral_whites(col: dict, src: dict, fields: tuple = ("upper", "sole")) -> list[str]:
    """balance the white-ish ones of `fields` to neutral white in place; returns the fields changed"""
    out = []
    for f in fields:
        if f not in col:
            continue
        L, a, b = (float(x) for x in lab(col[f]))
        C = float(np.hypot(a, b))
        if L < WHITE_L or C > WHITE_C:
            continue
        k = min(1.0, WHITE_C_MAX / max(C, 1e-6))
        L2 = max(L, WHITE_L_MIN)
        col[f] = lab_to_linear(L2, a * k, b * k)
        src[f] = f"{src.get(f, '')}; white-balanced L* {L:.0f}->{L2:.0f} C* {C:.0f}->{C * k:.0f}".lstrip("; ")
        out.append(f)
    return out


def delta_e(a, b) -> float:
    return float(np.linalg.norm(lab(a) - lab(b)))


def luminance(rgb_lin) -> float:
    return float(np.dot(rgb_lin, [0.2126, 0.7152, 0.0722]))


def skin_like(rgb_lin) -> bool:
    """warm, saturated (skin / red hair bleed of the bake): hue 5-35 deg, HSV saturation > 0.25"""
    h, s, v = colorsys.rgb_to_hsv(*linear_to_srgb(rgb_lin))
    return 5 / 360 <= h <= 35 / 360 and s > 0.25 and v > 0.25


def kmeans(X: np.ndarray, k: int, it: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """deterministic k-means (init at luminance quantiles) -> centres (k,3) sorted by share desc, labels"""
    X = np.asarray(X, float)
    lum = X @ np.array([0.2126, 0.7152, 0.0722])
    order = np.argsort(lum)
    C = np.stack([X[order[int((i + 0.5) / k * (len(X) - 1))]] for i in range(k)])
    lab_ = np.zeros(len(X), int)
    for _ in range(it):
        lab_ = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C2 = np.stack([np.median(X[lab_ == i], 0) if (lab_ == i).any() else C[i] for i in range(k)])
        if np.allclose(C2, C):
            break
        C = C2
    share = np.bincount(lab_, minlength=k)
    rank = np.argsort(-share)
    remap = np.empty(k, int)
    remap[rank] = np.arange(k)
    return C[rank], remap[lab_]


def clusters(X: np.ndarray, k: int = 2, merge_de: float = 8.0) -> list[dict]:
    """[{rgb (linear), hex, share}] sorted by share; [] for too few samples"""
    if X is None or len(X) < 20:
        return []
    k = min(k, len(X) // 10)
    C, lab_ = kmeans(X, max(1, k))
    n = np.bincount(lab_, minlength=len(C)) / len(lab_)
    out = [{"rgb": C[i], "hex": to_hex(C[i]), "share": float(n[i])} for i in range(len(C))]
    # one colour split by noise / shading into near-identical clusters -> merge (dE76 < merge_de)
    if len(out) > 1 and all(delta_e(out[0]["rgb"], c["rgb"]) < merge_de for c in out[1:]):
        m = np.median(X, 0)
        return [{"rgb": m, "hex": to_hex(m), "share": 1.0}]
    return out


# ------------------------------------------------------------------------------------------------ feet

def split_index(skin: Skin) -> dict:
    """SMPL-X vertex id -> first split vertex (v2 skins); identity on v1"""
    if skin.vertex_ids is None:
        return {}
    out = {}
    for i in range(len(skin.vertex_ids) - 1, -1, -1):
        out[int(skin.vertex_ids[i])] = i
    return out


def foot_frame(skin: Skin, side: str, split: dict | None = None) -> dict:
    """Per foot (rest, Unity coords): vertex set, sole plane, forward / lateral axes, local (x from the heel, s
    medial-positive, h above the sole) of every foot vertex, joints."""
    a, f = FOOT[side]
    split = split_index(skin) if split is None else split
    w = skin.weights[:, a] + skin.weights[:, f]
    idx = np.nonzero(w >= 0.5)[0]
    V = np.asarray(skin.vertices, float)
    lm = {k: V[split.get(v, v)] for k, v in LANDMARKS[side].items()}
    y0 = float(V[idx, 1].min())
    fwd = (lm["big"] + lm["small"]) / 2 - lm["heel"]
    fwd[1] = 0
    fwd /= np.linalg.norm(fwd)
    side_ax = np.cross([0.0, 1.0, 0.0], fwd)
    side_ax /= np.linalg.norm(side_ax)
    medial = 1.0 if np.dot(lm["big"] - lm["small"], side_ax) > 0 else -1.0
    rel = V[idx] - lm["heel"]
    x = rel @ fwd
    x0 = float(x.min())
    return {"side": side, "idx": idx, "y0": y0, "fwd": fwd, "side_ax": side_ax * medial, "heel": lm["heel"],
            "x": x - x0, "s": rel @ (side_ax * medial), "h": V[idx, 1] - y0, "x0": x0, "L": float(x.max() - x0),
            "ankle": np.asarray(skin.joints[a], float), "ball": np.asarray(skin.joints[f], float)}


def measure(skin: Skin, side: str, split: dict | None = None) -> dict:
    """Review numbers of one rest foot (the Unity builder measures the same way at load)."""
    ff = foot_frame(skin, side, split)
    x, s, h, L = ff["x"], ff["s"], ff["h"], ff["L"]
    ball_x = float((ff["ball"] - ff["heel"]) @ ff["fwd"] - ff["x0"])
    near = np.abs(x - ball_x) < 0.02
    plantar = h < 0.03
    return {"length_m": round(L, 4), "ball_from_heel_m": round(ball_x, 4),
            "ball_width_m": round(float(s[near & plantar].max() - s[near & plantar].min()), 4) if (near & plantar).any() else None,
            "ankle_height_m": round(float(ff["ankle"][1] - ff["y0"]), 4),
            "ball_joint_height_m": round(float(ff["ball"][1] - ff["y0"]), 4),
            "foot_vertices": int(len(ff["idx"])), "top_m": round(float(h.max()), 4)}


def bands(skin: Skin, ff: dict) -> dict:
    """foot-vertex masks (over ff["idx"]) of the colour bands"""
    x, h, L = ff["x"], ff["h"], ff["L"]
    ny = np.asarray(skin.normals, float)[ff["idx"], 1] if skin.normals is not None else np.zeros(len(x))
    sole = (h >= SOLE_H[0]) & (h < SOLE_H[1]) & (np.abs(ny) < 0.55) & (x > 0.04 * L) & (x < 0.96 * L)
    upper = ((h >= UPPER_H[0]) & (h < UPPER_H[1]) & (x > 0.12 * L)) | ((ny > 0.5) & (x > 0.55 * L) & (h > 0.012))
    heel = (x < 0.12 * L) & (h >= HEEL_H[0]) & (h < HEEL_H[1])
    outsole = (ny < -0.6) & (h < 0.02)
    sole_rear = sole & (x < 0.35 * L)
    return {"sole": sole, "sole_rear": sole_rear, "outsole": outsole & ~sole, "upper": upper & ~sole & ~heel,
            "heel": heel}


# ------------------------------------------------------------------------------------------------ bake texels

def bake_texels(skin: Skin, albedo: np.ndarray, coverage: np.ndarray | None) -> dict:
    """{band: (N,3) linear RGB texels} of both feet on the baked albedo (H,W,3 sRGB uint8)"""
    H, W = albedo.shape[:2]
    out = {"sole": [], "sole_rear": [], "outsole": [], "upper": [], "heel": []}
    if skin.uv is None:
        return {k: np.zeros((0, 3)) for k in out}
    split = split_index(skin)
    F = np.asarray(skin.faces)
    for side in FOOT:
        ff = foot_frame(skin, side, split)
        bm = bands(skin, ff)
        for band, sel in bm.items():
            vmask = np.zeros(len(skin.vertices), bool)
            vmask[ff["idx"][sel]] = True
            tris = F[vmask[F].all(1)]
            m = np.zeros((H, W), np.uint8)
            uv = np.asarray(skin.uv, float)
            for tri in tris:
                pts = np.stack([uv[tri, 0] * (W - 1), (1 - uv[tri, 1]) * (H - 1)], 1)
                cv2.fillConvexPoly(m, np.round(pts * 4).astype(np.int32), 255, cv2.LINE_8, 2)
            ok = m > 0
            if coverage is not None:
                ok &= coverage > 0
            out[band].append(srgb_to_linear(albedo[ok].astype(float) / 255.0))
    return {k: np.concatenate(v) if v else np.zeros((0, 3)) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ video samples

class Posed:
    """A dancer's capture skin + motion: posed foot vertices per motion frame (SmplxAvatar's FK + LBS)."""

    def __init__(self, capture: Path, role: str):
        _, self.skin = read_skin(capture / f"smplx_{role}_skin.bin")
        mh, fr = read_motion(capture / f"smplx_{role}.bin")
        T = mh["frames"]
        self.P, self.R = fk_unity(np.asarray(self.skin.joints, float), np.asarray(self.skin.parents),
                                  fr[:, :3].astype(float), fr[:, 3:3 + 55 * 4].reshape(T, 55, 4).astype(float))
        self.T = T
        split = split_index(self.skin)
        self.ff = {s: foot_frame(self.skin, s, split) for s in FOOT}
        self.bands = {s: bands(self.skin, self.ff[s]) for s in FOOT}
        F = np.asarray(self.skin.faces)
        self.tris = {}
        for s, ff in self.ff.items():
            m = np.zeros(len(self.skin.vertices), bool)
            m[ff["idx"]] = True
            self.tris[s] = F[m[F].all(1)]
        allidx = np.unique(np.concatenate([ff["idx"] for ff in self.ff.values()]))
        self.sub = allidx
        self.where = np.full(len(self.skin.vertices), -1)
        self.where[allidx] = np.arange(len(allidx))
        self.rel = np.asarray(self.skin.vertices, float)[allidx][:, None, :] - np.asarray(self.skin.joints, float)[None]
        self.w = np.asarray(self.skin.weights, float)[allidx]

    def world(self, k: int) -> np.ndarray:
        """posed foot-subset vertices at motion frame k in the right-handed world (cameras' frame)"""
        posed = np.einsum("jab,vjb->vja", self.R[k], self.rel) + self.P[k][None]
        return np.einsum("vj,vja->va", self.w, posed) * MIRROR


def _project(X, K, R, t, k1):
    Xc = X @ np.asarray(R).T + np.asarray(t)
    z = Xc[:, 2]
    zc = np.maximum(z, 1e-4)
    x, y = Xc[:, 0] / zc, Xc[:, 1] / zc
    d = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], 1), z


def video_samples(take: Take, capture: Path, posed: dict, step_s: float = 0.5, cams: list[str] | None = None,
                  min_inside: float = 0.7, max_views: int = 200) -> tuple[dict, list[dict]]:
    """{role: {band: (N,3) linear RGB}} from the video + per-view-foot records (for the review crops)"""
    from .avatar_texture import camera_views
    work = take.work
    times = np.asarray(json.loads((capture / "times.json").read_text()), float)
    cams = cams or [s.id for s in take.sources if (work / "frames" / s.id / "frames.json").exists()
                    and (work / "cameras" / f"{s.id}.npz").exists()]
    views, _ = camera_views(work, cams, float(times[0]), float(times[-1]), step_s)
    views = views[:max_views]
    order = ("sole", "outsole", "heel", "upper")  # painting band of a triangle: the first with >= 2 of its vertices
    out = {r: {b: [] for b in order + ("sole_rear",)} for r in posed}
    recs = []
    for v in views:
        if not v["ok"]:
            continue
        img = cv2.imread(str(work / "frames" / v["cam"] / f"{v['i']:06d}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        dd = work / "dancers" / v["cam"] / "masks"
        masks = {}
        for r in posed:
            m = cv2.imread(str(dd / f"mask_{v['i']:04d}_{PIDS[r]:02d}.png"), cv2.IMREAD_GRAYSCALE)
            masks[r] = np.zeros((H, W), bool) if m is None else m > 127
        k = int(np.clip(np.searchsorted(times, v["t"]), 0, len(times) - 1))
        if k > 0 and abs(times[k - 1] - v["t"]) < abs(times[k] - v["t"]):
            k -= 1
        if abs(times[k] - v["t"]) > 0.02:
            continue
        cam_c = -np.asarray(v["R"]).T @ np.asarray(v["t_w"])
        # painter's z-buffer of every foot triangle of both dancers (front-facing only)
        polys = []
        geo = {}
        for r, ps in posed.items():
            Xw = ps.world(k)
            px, z = _project(Xw, v["K"], v["R"], v["t_w"], v["k1"])
            geo[r] = (px, z)
            for s, tris in ps.tris.items():
                tl = ps.where[tris]
                a, b, c = Xw[tl[:, 0]], Xw[tl[:, 1]], Xw[tl[:, 2]]
                # skin faces carry Unity (reversed) winding: world-outward normal = (c - a) x (b - a)
                n = np.cross(c - a, b - a)
                front = np.einsum("ij,ij->i", n, cam_c - a) > 0
                zt = z[tl].mean(1)
                vb = {band: np.zeros(len(ps.skin.vertices), bool) for band in ps.bands[s]}
                for band, sel in ps.bands[s].items():
                    vb[band][ps.ff[s]["idx"][sel]] = True
                for ti in np.nonzero(front & (z[tl].min(1) > 0.1))[0]:
                    band = next((bn for bn in order if vb[bn][tris[ti]].sum() >= 2), "other")
                    if band == "sole" and vb["sole_rear"][tris[ti]].sum() >= 2:
                        band = "sole_rear"
                    polys.append((zt[ti], r, s, band, px[tl[ti]]))
        if not polys:
            continue
        polys.sort(key=lambda p: -p[0])
        keys = sorted({(p[1], p[2], p[3]) for p in polys})
        code = {kk: i + 1 for i, kk in enumerate(keys)}
        lab_img = np.zeros((H, W), np.uint8)
        for zt, r, s, band, P2 in polys:
            cv2.fillConvexPoly(lab_img, np.round(P2 * 4).astype(np.int32), code[(r, s, band)], cv2.LINE_8, 2)
        for r in posed:
            partner = [q for q in posed if q != r]
            occ = np.zeros((H, W), np.uint8)
            for q in partner:
                occ |= masks[q].astype(np.uint8)
            occ = cv2.dilate(occ, np.ones((5, 5), np.uint8)) > 0
            for s in FOOT:
                region = np.isin(lab_img, [code[kk] for kk in keys if kk[0] == r and kk[1] == s])
                n_all = int(region.sum())
                if n_all < 60:
                    continue
                inside = float((region & masks[r]).sum()) / n_all
                rec = {"cam": v["cam"], "i": v["i"], "t": v["t"], "k": k, "role": r, "side": s, "px": n_all,
                       "inside": round(inside, 3), "used": False}
                ys, xs = np.nonzero(region)
                rec["box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                recs.append(rec)
                if inside < min_inside:
                    continue
                rec["used"] = True
                for band in order + ("sole_rear",):
                    kk = (r, s, band)
                    if kk not in code:
                        continue
                    m = (lab_img == code[kk]).astype(np.uint8)
                    m = cv2.erode(m, np.ones((3, 3), np.uint8)) > 0
                    m &= masks[r] & ~occ
                    if m.sum() >= 15:
                        px_ = srgb_to_linear(img[m][:, ::-1].astype(float) / 255.0)
                        out[r][band].append(px_)
                        if band == "sole_rear":
                            out[r]["sole"].append(px_)  # the rear third is part of the sole band too
                        rec[band] = int(m.sum())
    res = {r: {b: np.concatenate(v) if v else np.zeros((0, 3)) for b, v in d.items()} for r, d in out.items()}
    return res, recs


# ------------------------------------------------------------------------------------------------ palette

def shade(rgb_lin, f: float) -> np.ndarray:
    return np.clip(np.asarray(rgb_lin, float) * f, 0, 1)


def palette(bake: dict, video: dict | None, gain: np.ndarray | None) -> tuple[dict, dict, dict]:
    """(colours {field: linear rgb}, source {field: text}, confidence {field: 0..1}) of one dancer"""
    col, src, conf = {}, {}, {}
    vid = {b: (x * gain if gain is not None else x) for b, x in (video or {}).items()}
    bu, bs, bh = clusters(bake.get("upper")), clusters(bake.get("sole")), clusters(bake.get("heel"))
    vu, vs = clusters(vid.get("upper")), clusters(vid.get("sole"))

    # upper: bake when it has texels (shares the body's grade), video otherwise; flag disagreement
    if bu:
        col["upper"], src["upper"] = bu[0]["rgb"], f"bake upper band k-means n={len(bake['upper'])} share {bu[0]['share']:.2f}"
        conf["upper"] = min(1.0, 0.4 + bu[0]["share"])
        if vu:
            d = delta_e(bu[0]["rgb"], vu[0]["rgb"])
            src["upper"] += f"; video {vu[0]['hex']} dE {d:.0f}"
            if d > 25:
                conf["upper"] *= 0.7
    elif vu:
        col["upper"], src["upper"], conf["upper"] = vu[0]["rgb"], f"video upper n={len(vid['upper'])}", 0.6
    # sole: bake unless skin-like / the band is split against the upper; then the video's
    sole_bake_ok = bool(bs) and not skin_like(bs[0]["rgb"]) and bs[0]["share"] >= 0.6
    if sole_bake_ok and vs and delta_e(bs[0]["rgb"], vs[0]["rgb"]) > 35:
        sole_bake_ok = False  # the bake's band is contaminated (e.g. trouser hem / skin bleed into the sole UVs)
    if sole_bake_ok:
        col["sole"], src["sole"] = bs[0]["rgb"], f"bake sole band k-means n={len(bake['sole'])} share {bs[0]['share']:.2f}"
        conf["sole"] = min(1.0, 0.3 + bs[0]["share"])
    elif vs:
        # the brightest cluster with >= 25 % share: midsoles are light, the dark rest is floor / outsole / shadow
        c = max((c for c in vs if c["share"] >= 0.25), key=lambda c: luminance(c["rgb"]))
        col["sole"], src["sole"] = c["rgb"], f"video sole band n={len(vid['sole'])} share {c['share']:.2f}" + \
            (f" (bake rejected: {bs[0]['hex']})" if bs else "")
        conf["sole"] = 0.5 + 0.3 * c["share"]
    else:
        col["sole"], src["sole"], conf["sole"] = from_hex(DEFAULT["sole"]), "default", 0.0
    if "upper" not in col:
        col["upper"], src["upper"], conf["upper"] = from_hex(DEFAULT["upper"]), "default", 0.0
    up0 = col["upper"]  # as sampled: a second cluster is an accent only if it differs from this (not a shading of it)
    neutral_whites(col, src, ("upper", "sole"))  # before anything is derived from them
    up, so = col["upper"], col["sole"]
    light_upper = luminance(up) > 0.35
    # accent: second cluster of the upper band (video first: the bake smooths a thin stripe away)
    acc = None
    for cl, name in ((vu, "video"), (bu, "bake")):
        if len(cl) > 1 and cl[1]["share"] >= 0.10 and delta_e(cl[1]["rgb"], up0) >= 20:
            acc = (cl[1]["rgb"], f"{name} upper 2nd cluster share {cl[1]['share']:.2f}")
            break
    if acc is None:
        acc = (so if not light_upper else shade(up, 0.55), "derived (sole colour on a dark upper, darker upper on a light one)")
    col["accent"], src["accent"], conf["accent"] = acc[0], acc[1], 0.6 if "cluster" in acc[1] else 0.2
    # heel tab: a cluster of the heel band that stands out from the upper (video first, then the bake)
    ht = None
    for cl, name in ((clusters(vid.get("heel")), "video"), (bh, "bake")):
        ht = next((c for c in cl if c["share"] >= 0.15 and delta_e(c["rgb"], up0) >= 25), None)
        if ht is not None:
            ht = (ht["rgb"], f"{name} heel band cluster share {ht['share']:.2f}", 0.5)
            break
    col["heel_tab"], src["heel_tab"], conf["heel_tab"] = ht or (
        col["accent"] if light_upper else shade(up, 0.7), "derived (accent / darker upper)", 0.2)
    # sole heel (rear third of the midsole: a contrasting heel unit on runners): the video's rear band when its
    # dominant colour differs from the sole
    vr = clusters(vid.get("sole_rear"))
    if vr and vr[0]["share"] >= 0.5 and len(vid["sole_rear"]) >= 150 and delta_e(vr[0]["rgb"], so) > 30:
        col["sole_heel"], src["sole_heel"], conf["sole_heel"] = vr[0]["rgb"],             f"video rear sole band n={len(vid['sole_rear'])} share {vr[0]['share']:.2f}", 0.4 + 0.3 * vr[0]["share"]
        neutral_whites(col, src, ("sole_heel",))
    else:
        col["sole_heel"], src["sole_heel"], conf["sole_heel"] = so, "= sole", conf["sole"]
    # outsole (seen only when a foot is lifted towards a camera): the video's plantar band, else the bake's (never a
    # skin-like bake colour: that is bleed), else derived
    vo, bo = clusters(vid.get("outsole")), clusters(bake.get("outsole"))
    if vo and len(vid["outsole"]) >= 150:
        col["outsole"], src["outsole"], conf["outsole"] = vo[0]["rgb"],             f"video plantar band n={len(vid['outsole'])} share {vo[0]['share']:.2f}", 0.3 + 0.4 * vo[0]["share"]
        neutral_whites(col, src, ("outsole",))
    elif bo and len(bake["outsole"]) >= 300 and not skin_like(bo[0]["rgb"]):
        col["outsole"], src["outsole"], conf["outsole"] = bo[0]["rgb"],             f"bake plantar band n={len(bake['outsole'])} share {bo[0]['share']:.2f}", 0.3
        neutral_whites(col, src, ("outsole",))
    else:
        col["outsole"], src["outsole"], conf["outsole"] = shade(so, 0.72) if light_upper else shade(up, 0.9),             "derived from upper / sole", 0.1
    # never seen: laces, lining - derived
    col["lace"] = shade(up, 0.92) if light_upper else shade(up, 0.8)
    col["lining"] = shade(up, 0.78) if light_upper else shade(up, 1.3)
    for f in ("lace", "lining"):
        src[f], conf[f] = "derived from upper", 0.2
    return col, src, conf


def auto_style(col: dict) -> str:
    lu, ls = luminance(col["upper"]), luminance(col["sole"])
    if lu > 0.35 and ls > 0.35:
        return "court_platform"
    if lu < 0.12 and ls > 0.35:
        return "runner"
    return "neutral"


def video_gain(bake: dict, video: dict) -> tuple[np.ndarray | None, str]:
    """per-channel gain video -> bake grade from the dancer whose upper is brightest in the bake"""
    best = None
    for r in bake:
        bu, vu = clusters(bake[r].get("upper")), clusters(video.get(r, {}).get("upper"))
        if bu and vu and luminance(bu[0]["rgb"]) > 0.2:
            if best is None or luminance(bu[0]["rgb"]) > best[0]:
                best = (luminance(bu[0]["rgb"]), r, bu[0]["rgb"] / np.maximum(vu[0]["rgb"], 1e-4))
    if best is None:
        return None, "none (no light upper seen in both the bake and the video)"
    g = np.clip(best[2], 0.5, 2.0)
    return g, f"bake / video of the {best[1]}'s upper: {np.round(g, 3).tolist()}"


# ------------------------------------------------------------------------------------------------ build

def toml_geometry(take_cfg: dict, role: str) -> dict:
    """takes toml [dancers.<role>] shoe_geometry -> {field: metres} (validated)."""
    g = (((take_cfg or {}).get("dancers") or {}).get(role) or {}).get("shoe_geometry") or {}
    out = {}
    for k, v in g.items():
        if k not in GEOMETRY:
            raise ValueError(f"{role}: shoe_geometry key {k!r} not one of {GEOMETRY}")
        v = float(v)
        if not GEOMETRY_RANGE[0] <= v <= GEOMETRY_RANGE[1]:
            raise ValueError(f"{role}: shoe_geometry {k} = {v} outside {GEOMETRY_RANGE} m")
        out[k] = v
    return out


def apply_geometry(shoes: dict, take_cfg: dict) -> list[str]:
    """Set the toml's shoe_geometry on a shoes.json dict in place (source noted). -> changed 'role.field' names."""
    changed = []
    for r, d in (shoes.get("dancers") or {}).items():
        g = toml_geometry(take_cfg, r)
        for k, v in g.items():
            if d.get(k) != v:
                changed.append(f"{r}.{k}")
            d[k] = v
        if g:
            d.setdefault("source", {})["geometry"] = "takes toml shoe_geometry: " + ", ".join(
                f"{k} {v:.3f}" for k, v in g.items())
    return changed


_HEX_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")


def apply_override(shoes: dict, override: dict) -> list[str]:
    """Set work/<take>/avatar/shoes_override.json ({"<role>": {"<field>": "#rrggbb" | number | style}}, the owner's decisions: "Larissa's
    shoes are white, Kadu's are black") on a shoes.json dict in place. build() applies the same file while it samples; every export applies
    it again, so a seeded or re-sampled shoes.json cannot drift from the decision and `--export-only` picks a change up. Colours are
    validated; a numeric field is one of the style's numbers. -> changed 'role.field' names."""
    changed = []
    for r, ov in (override or {}).items():
        d = (shoes.get("dancers") or {}).get(r)
        if d is None or not isinstance(ov, dict):
            continue
        for f, v in ov.items():
            if f == "style":
                if v not in STYLES:
                    raise ValueError(f"{r}: unknown shoe style {v!r} ({'|'.join(STYLES)})")
            elif f in FIELDS:
                if not isinstance(v, str) or not _HEX_COLOUR.match(v):
                    raise ValueError(f"{r}: shoes_override {f} must be a \"#rrggbb\" colour, got {v!r}")
                v = v.lower()
            elif f in STYLES["neutral"]:
                v = float(v)
            else:
                raise ValueError(f"{r}: shoes_override key {f!r} is not a colour field, a style number or 'style'")
            if d.get(f) != v:
                changed.append(f"{r}.{f}")
            d[f] = v
            if f in FIELDS:
                d.setdefault("source", {})[f] = "override (shoes_override.json: owner decision)"
                d.setdefault("confidence", {})[f] = 1.0
    return changed


def read_override(take) -> dict:
    p = take.work / "avatar" / "shoes_override.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def captures_for(take: str, streaming: Path = UNITY_STREAMING) -> list[Path]:
    out = []
    for d in sorted(Path(streaming).iterdir()) if Path(streaming).exists() else []:
        cj = d / "capture.json"
        if d.is_dir() and cj.exists():
            try:
                man = json.loads(cj.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if (man.get("provenance") or {}).get("take") == take:
                out.append(d)
    return out


def _read_rgb(p: Path) -> np.ndarray | None:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if img is None else img[:, :, ::-1]


def build(take: Take, capture: Path | None = None, video: bool = True, step_s: float = 0.5) -> tuple[dict, dict]:
    caps = [capture] if capture else captures_for(take.name)
    if not caps:
        raise FileNotFoundError(f"no StreamingAssets capture with provenance.take = {take.name} (export it first)")
    cap = Path(caps[0])
    av = take.work / "avatar"
    posed = {r: Posed(cap, r) for r in ROLES if (cap / f"smplx_{r}_skin.bin").exists()}
    bake = {}
    for r, ps in posed.items():
        alb = _read_rgb(av / f"{r}_albedo.png")
        cov = cv2.imread(str(av / f"{r}_coverage.png"), cv2.IMREAD_GRAYSCALE)
        bake[r] = bake_texels(ps.skin, alb, cov) if alb is not None else {}
    vid, recs = ({}, [])
    if video:
        vid, recs = video_samples(take, cap, posed, step_s=step_s)
    gain, gain_src = video_gain(bake, vid) if vid else (None, "no video")
    override = {}
    op = av / "shoes_override.json"
    if op.exists():
        override = json.loads(op.read_text(encoding="utf-8"))
    out = {"format": "dancecap shoes", "version": VERSION, "take": take.name, "capture_skins": cap.name,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "video_grade": gain_src, "dancers": {},
           "likeness": "shoe colours sampled from the dancers' video / texture bake: work/ and git-ignored "
                       "StreamingAssets only, never a tracked file"}
    for r, ps in posed.items():
        col, src, conf = palette(bake.get(r, {}), vid.get(r), gain)
        dcfg = (take.cfg.get("dancers") or {}).get(r) or {}
        style = dcfg.get("shoe_style") or auto_style(col)
        style_src = "takes toml shoe_style" if dcfg.get("shoe_style") else "auto (upper / sole lightness)"
        ov = override.get(r) or {}
        if "style" in ov:
            style, style_src = ov["style"], "override"
        if style not in STYLES:
            raise ValueError(f"{r}: unknown shoe style {style!r} ({'|'.join(STYLES)})")
        d = {"style": style, **{f: to_hex(col[f]) for f in FIELDS}, **STYLES[style], **toml_geometry(take.cfg, r)}
        for f in FIELDS:
            if f in ov:
                d[f], src[f], conf[f] = ov[f], "override", 1.0
        for f in STYLES[style]:
            if f in ov:
                d[f] = float(ov[f])
        d["source"] = {"style": style_src, **src}
        if toml_geometry(take.cfg, r):
            d["source"]["geometry"] = "takes toml shoe_geometry: " + ", ".join(
                f"{k} {v:.3f}" for k, v in toml_geometry(take.cfg, r).items())
        d["confidence"] = {f: round(float(conf[f]), 2) for f in FIELDS}
        d["samples"] = {"bake": {b: int(len(x)) for b, x in bake.get(r, {}).items()},
                        "video": {b: int(len(x)) for b, x in vid.get(r, {}).items()},
                        "video_views_used": sum(1 for x in recs if x["role"] == r and x["used"])}
        d["clusters"] = {f"{kind}_{b}": [{"hex": c["hex"], "share": round(c["share"], 3)} for c in clusters(x)]
                         for kind, src_ in (("bake", bake.get(r, {})), ("video", vid.get(r, {})))
                         for b, x in src_.items()}
        d["measured"] = {s: measure(ps.skin, s) for s in FOOT}
        out["dancers"][r] = d
    extra = {"recs": recs, "capture": cap, "posed": posed}
    return out, extra


# ------------------------------------------------------------------------------------------------ review

def swatch_row(d: dict, w: int = 760, h: int = 60) -> np.ndarray:
    img = np.full((h + 22, w, 3), 24, np.uint8)
    n = len(FIELDS)
    cw = w // n
    for j, f in enumerate(FIELDS):
        c = np.round(linear_to_srgb(from_hex(d[f])) * 255).astype(int)[::-1]
        cv2.rectangle(img, (j * cw + 2, 2), ((j + 1) * cw - 2, h), tuple(int(x) for x in c), -1)
        cv2.putText(img, f"{f} {d[f]}", (j * cw + 4, h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def video_crops(take: Take, recs: list[dict], role: str, n: int = 4, size: int = 240) -> list[np.ndarray]:
    """crops around both feet of a dancer in the best-covered views (largest projected feet, both used)"""
    by_view = {}
    for r in recs:
        if r["role"] == role and r["used"]:
            by_view.setdefault((r["cam"], r["i"]), []).append(r)
    ranked = sorted(by_view.items(), key=lambda kv: (-len(kv[1]), -sum(x["px"] for x in kv[1])))
    out, cams_used = [], set()
    for (cam, i), rs in ranked:
        if cam in cams_used:
            continue
        img = cv2.imread(str(take.work / "frames" / cam / f"{i:06d}.jpg"))
        if img is None:
            continue
        x0 = min(r["box"][0] for r in rs); y0 = min(r["box"][1] for r in rs)
        x1 = max(r["box"][2] for r in rs); y1 = max(r["box"][3] for r in rs)
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        half = max(x1 - x0, y1 - y0) * 0.75 + 20
        a = np.clip((c - half).astype(int), 0, [img.shape[1] - 1, img.shape[0] - 1])
        b = np.clip((c + half).astype(int), 0, [img.shape[1], img.shape[0]])
        crop = cv2.resize(img[a[1]:b[1], a[0]:b[0]], (size, size))
        cv2.putText(crop, f"cam {cam} f{i}", (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(crop)
        cams_used.add(cam)
        if len(out) >= n:
            break
    return out


def palette_sheet(take: Take, shoes: dict, recs: list[dict], path: Path) -> Path:
    rows = []
    for r, d in shoes["dancers"].items():
        crops = video_crops(take, recs, r)
        head = np.full((30, 960, 3), 16, np.uint8)
        cv2.putText(head, f"{r}: style {d['style']} | sole heel {d['sole_heel_m']*100:.1f} cm fore "
                    f"{d['sole_fore_m']*100:.1f} cm | video views {d['samples']['video_views_used']}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        strip = np.hstack(crops + [np.zeros((240, 240, 3), np.uint8)] * (4 - len(crops))) if crops else \
            np.zeros((240, 960, 3), np.uint8)
        sw = swatch_row(d, 960)
        rows += [head, strip, sw]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


SHEET_SHOTS = ("before_{r}_front34", "after_{r}_standing_front34", "after_{r}_stepping_side", "after_{r}_toes_low",
               "after_{r}_standing_front34_op03")


def _unity_tile(path: Path, size: int, label: str, crop: float = 0.55) -> np.ndarray | None:
    """centre crop of a Unity review shot (the review camera centres the shoes), square, labelled"""
    img = cv2.imread(str(path))
    if img is None:
        return None
    H, W = img.shape[:2]
    side = int(min(H, W) * crop * 1.2)
    x0, y0 = max(0, W // 2 - side // 2), max(0, H // 2 - side // 2)
    img = cv2.resize(img[y0:y0 + side, x0:x0 + side], (size, size))
    cv2.rectangle(img, (0, size - 22), (size, size), (0, 0, 0), -1)
    cv2.putText(img, label, (6, size - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def vs_video_sheet(take: Take, out: Path | None = None, tile: int = 300) -> Path:
    """Unity close-ups (review/realism/shoes_{before,after}_<role>_*.png from head-movement Tools/playtest_shoes.ps1,
    centre-cropped on the shoes) next to video crops of the dancer's real sneakers (shoes_recs.json of the last run):
    per dancer: 2 video crops | bare SMPL-X feet before | sneakers standing / stepping / on the toes / at the default
    avatar opacity."""
    rdir = take.work / "review" / "realism"
    recs = json.loads((rdir / "shoes_recs.json").read_text(encoding="utf-8")) if (rdir / "shoes_recs.json").exists() else []
    rows = []
    for r in ROLES:
        tiles = []
        for name in SHEET_SHOTS:
            n = name.format(r=r)
            t = _unity_tile(rdir / f"shoes_{n}.png", tile, n.replace(f"_{r}", "").replace("_", " "))
            if t is not None:
                tiles.append(t)
        crops = video_crops(take, recs, r, n=2, size=tile)
        if not tiles and not crops:
            continue
        row = np.hstack([cv2.copyMakeBorder(t, 0, 0, 0, 4, cv2.BORDER_CONSTANT) for t in crops + tiles])
        head = np.full((28, row.shape[1], 3), 16, np.uint8)
        cv2.putText(head, f"{r}: video crops (left) | Unity: bare SMPL-X feet before, procedural sneakers after (right)",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        rows += [head, row]
    if not rows:
        raise FileNotFoundError(f"no shoes_*.png in {rdir} (run head-movement Tools/playtest_shoes.ps1)")
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    out = out or rdir / "shoes_vs_video.jpg"
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


# ------------------------------------------------------------------------------------------------ export

def export(take: Take, capture: str | None = None, streaming: Path = UNITY_STREAMING) -> list[Path]:
    """Write avatar/shoes.json (+ the toml's shoe_geometry) into StreamingAssets/<capture>/ (and
    work/<take>/export/<capture>/ when it exists). Re-run after every `python -m dancecap export` (that rewrites the
    capture folder)."""
    src = take.work / "avatar" / "shoes.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing (run python -m dancecap.shoes {take.name})")
    shoes = json.loads(src.read_text(encoding="utf-8"))
    apply_geometry(shoes, getattr(take, "cfg", None) or {})
    apply_override(shoes, read_override(take))
    text = json.dumps(shoes, indent=1)
    dirs = [Path(streaming) / capture] if capture else captures_for(take.name, streaming)
    out = []
    for d in dirs:
        if not (d / "capture.json").exists():
            raise FileNotFoundError(f"{d}/capture.json missing (export the capture first)")
        for dst in (d, take.work / "export" / d.name):
            if dst.exists():
                (dst / "shoes.json").write_text(text, encoding="utf-8")
                out.append(dst / "shoes.json")
        from . import export_unity as eu
        if set(eu.splat_roles(d)) >= set(eu.ROLES):
            # owner rule 2026-10-23: both dancers are splats (the sneakers are mesh-avatar only): archived, not left in the capture
            arch = eu.archive_mesh_avatars(d, take=take.name)
            out = [p for p in out if p.exists()] + [eu._abs(m["to"]) for m in arch["moved"]]
    return out


def run(take: Take, capture: str | None = None, video: bool = True, do_export: bool = True, step_s: float = 0.5) -> dict:
    cap = UNITY_STREAMING / capture if capture else None
    shoes, extra = build(take, cap, video=video, step_s=step_s)
    out = take.work / "avatar" / "shoes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(shoes, indent=1), encoding="utf-8")
    rdir = take.work / "review" / "realism"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "shoes_recs.json").write_text(json.dumps(extra["recs"], indent=0), encoding="utf-8")
    print(f"[shoes] {out}")
    for r, d in shoes["dancers"].items():
        print(f"[shoes]  {r}: {d['style']} upper {d['upper']} sole {d['sole']} accent {d['accent']} "
              f"heel_tab {d['heel_tab']} (views {d['samples']['video_views_used']}; {d['source']['sole']})")
    print(f"[shoes] review {palette_sheet(take, shoes, extra['recs'], rdir / 'shoes_palette.jpg')}")
    if do_export:
        for p in export(take, capture):
            print(f"[shoes] -> {p}")
    return shoes


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.shoes", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--capture", help="capture folder name (default: every capture of the take)")
    ap.add_argument("--no-video", action="store_true", help="colours from the texture bake only")
    ap.add_argument("--step-s", type=float, default=0.5, help="video: one frame per camera every n seconds")
    ap.add_argument("--export-only", action="store_true", help="only copy avatar/shoes.json into the captures")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--sheet", action="store_true", help="only compose review/realism/shoes_vs_video.jpg")
    a = ap.parse_args(argv)
    take = Take.load(a.take)
    if a.sheet:
        print(f"[shoes] {vs_video_sheet(take)}")
        return
    if a.export_only:
        for p in export(take, a.capture):
            print(f"[shoes] -> {p}")
        return
    run(take, a.capture, video=not a.no_video, do_export=not a.no_export, step_s=a.step_s)


if __name__ == "__main__":
    main()
