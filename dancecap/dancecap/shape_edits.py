"""Per-dancer rest-shape edits of the exported SMPL-X skin (applied by export_unity.export on every export).

The fitted betas stay untouched - the motion, the joints and every analysis keep using them. Only the rest vertices of
the Unity skin move, before the UV seam split, so texture UVs, LBS weights, rest joints (and the exporter's FK check)
are unchanged. Configured per dancer in the take TOML:

    [dancers.follow.shape_edits]
    midsection_scale = 0.75                 # waist / abdomen (lower ribs .. hips): width AND depth x 0.75
                                            # ([width, depth] for different factors)
    midsection_reference = "larissa_kadu"   # optional: the scale is relative to THAT take's unedited body (same role,
                                            # its smplx/): final band width / depth = scale x the reference's
    midsection_depth_pivot = 0.5            # optional: where the depth shrinks to, 0 = the back surface stays (all
                                            # of it comes off the front), 0.5 = the section's middle (default)
    shoe_length_m = 0.273                   # optional: the sneaker's outer length (heel back .. toe tip) measured in
                                            # the video in THIS take's world (python -m dancecap.shoe_size); the feet
                                            # are shortened (never lengthened) so the Unity sneaker, which is built
                                            # around the rest foot, comes out this long
    foot_height_scale = 0.85                # optional: height of the foot below the ankle (sole stays on the floor)
    foot_scale = [0.97, 0.97, 0.85]         # optional explicit [length, width, height] (overrides the two above)

    [dancers.follow.shape_edits.face]       # optional: the LOWER FACE (every factor defaults to 1 = unchanged)
    lower_face_height_scale = 0.94          # subnasale .. menton height (upper lip, mouth, chin), toward the subnasale
    chin_height_scale = 0.90                # lower-lip fold .. menton height (the chin alone), on top of the above
    chin_width_scale = 0.88                 # jawline taper: x 1 at the jaw angles -> this factor at the chin and below
    jaw_width_scale = 1.0                   # width of the whole jaw below the jaw angles (gonial width)
    cheek_scale = 1.0                       # width of the lower cheeks (cheekbone line .. jaw angles)
    lower_face_depth_scale = 0.97           # chin projection: depth about the jaw-angle plane, 0 at the lips -> full
                                            # at the chin
    submental_height_scale = 0.7            # the under-chin band (menton .. FACE_THROAT_M below it): < 1 lifts the
                                            # throat toward the chin - a sharper jaw-neck angle (defined jawline)
    # petite taper (added 2026-10-08, face_shape_v4; defaults = no change):
    jaw_taper = 0.92                        # V-taper: width x 1 at the cheekbone line -> this factor at the jaw angles
                                            # and below (jaw angles in, lower cheeks slimmer; the posterior jaw under
                                            # the ear lobes moves too, the ears do not)
    cheekbone_keep = 0.5                    # 0..1: how long the cheeks stay full before the taper (exponent 1 + 2 k
                                            # on the cosine profile)
    chin_point = 0.4                        # 0..1: a smaller, slightly pointed chin (extra narrowing of the chin
                                            # proper, up to 30 % at 1; fades out under the chin and towards the jaw angles)
    neck_under_jaw = 0.96                   # slimmer under-jaw / front of the neck: width and front depth of the neck
                                            # below the menton (nape, hair roots and the neck-shoulder seam stay)
    chin_forward_mm = 4.0                   # -6..12 mm: the chin proper (lower-lip fold .. menton, |x| < 3 cm) moves
                                            # forward by this much (cosine out above, under the chin and sideways). The
                                            # identity transplant sets the chin back (its depth is the least constrained
                                            # by a front photo); this gives a small chin that does not recede
    # identity transplant (dancecap.face_identity; applied BEFORE the factors above, which then act on the new face):
    identity = "work/larissa_kadu_38/face_shape/follow_identity.npz"   # FLAME identity fitted to the dancer's photos
                                            # (path relative to dancecap/ or absolute; git-ignored likeness data)
    identity_weight = 1.0                   # 0..1.5 blend toward it (re-targeted to THIS take's rest head; cached in
                                            # work/<take>/face_shape/<role>_identity_disp.npz)
    identity_mouth = 0.5                    # 0..1: share of the identity's move that the lips / mouth take (0 keeps the
                                            # mouth where the texture has it; the identity moves the whole mouth a few mm
                                            # with the shorter lower face, which would squeeze the philtrum under the fixed
                                            # nose). Blends out over 14 mm of surface around the lips
    identity_neck = 1.0                     # 0..1: extra damping of it under the chin / on the neck (1 = none: the neck
                                            # part is already a biharmonic fade-out of the chin's move, see below)

Midsection: a band from the rest joints, D = spine2.y - pelvis.y: weight 0 below pelvis - 0.15 D (the hips), 1 from
pelvis + 0.25 D to spine2 - 0.05 D (abdomen and waist), 0 above spine2 + 0.25 D (under the bust), cosine ramps in
between (C1: no kinks). Inside, x and z are scaled about the torso's cross-section centre (x: the middle of the mesh
section every 1 cm; z: back + depth_pivot x depth; Gaussian-smoothed over height), y is unchanged. Vertices skinned
to the arms / hands / collars are left alone (tapered by that weight).

Feet: per foot about the ankle's vertical line on the sole plane (the lowest foot vertex), in the foot frame of
shoes.foot_frame (heel -> toe-midpoint landmarks): x along the foot x length scale, across x width scale, height above
the sole x height scale - so the sole stays on the floor. Weight = w(ankle) + w(foot), tapered to 0 from 1.5 cm below
to 3.5 cm above the ankle joint's height: the leg is untouched. The bones are not moved (the toe joint keeps its
fitted position; at a 3 % shorter foot it sits ~4 mm ahead of the mesh's ball crease).

Face (lower face only; added 2026-10-08 for a follow whose avatar face read "inflated" at the jaw: the MAMMA
locked-head SMPL-X keeps the template's face proportions, only scaled with the head). Anchors are FLAME 2020's static
iBUG-68 landmarks carried onto SMPL-X (FACE_LMK: vertex triples + barycentric weights, so nothing FLAME is read at
run time): nasion, subnasale (sn), lower-lip fold (li), stomion (sto), menton (me), the jaw angles (go: iBUG 4 / 12),
the cheek contour (zy: iBUG 1 / 15) and the ear-level contour (iBUG 0 / 16). In the rest pose (y up, z forward):
  heights  y' = y + D(y): D integrates a compression rate (1 - lower_face_height_scale) over [me, sn] plus
           (1 - chin_height_scale) over [me, li] plus (1 - submental_height_scale) over [me - FACE_THROAT_M, me],
           each window with cosine edges (+-FACE_RAMP_M): C1, zero above the subnasale, constant below the throat
           level (the under-chin moves up with the chin and the throat comes along; the neck fade below stretches the
           front of the neck instead of folding the jaw-neck angle)
  widths   x' = mid + f (x - mid), f = jaw_width_scale on the jaw below the angles (fading out halfway up to sn) x
           chin_width_scale tapered in from the angles (go: 1) to the menton (and below) x cheek_scale on the band
           between the jaw angles and the cheekbone line
  depth    z' = z_go + g (z - z_go), g = lower_face_depth_scale tapered in from the stomion (1) to the menton
Every field is multiplied by one C1 mask: front (z from the ear-level contour - 5 mm -> + 25 mm: ears, nape, scalp and
the hair roots stay; widened to -3 .. +4.5 cm on the neck below the throat level), neck (full down to FACE_THROAT_M
under the menton, then fading over FACE_NECK_FADE_M - or pi x
the throat's lift when that is longer - down the front of the neck), lateral (|x - mid| < 8.5 .. 10.5 cm) and 1 - arm weight. Eyes, brows, nose, forehead, cranium and scalp lie
above every field (exactly unchanged; cheek_scale < 1 is the only factor reaching up to the cheekbone line). UVs, LBS
weights and joints are untouched; blendshape deltas (face_export) are added on top as before.
Petite taper (jaw_taper, cheekbone_keep, chin_point, neck_under_jaw), widths about the midline, on a second C1 mask
(face_taper_mask: the face mask with its front edge moved 1.2 cm back below the ear lobes - full 2.4 cm under the lobe
bottom, y_lobe = sto + 0.32 (sn - sto) - so the posterior jaw / jaw angles can come in while the ears and the nape stay,
and its neck fade shortened to FACE_TAPER_FADE_M = 4 cm: the taper is a width change of the jaw and the upper neck, the
neck-shoulder seam stays put):
  jaw_taper       f = 1 - (1 - jaw_taper) t^(1 + 2 cheekbone_keep), t = cosine ramp 0 at the cheekbone line (zy) -> 1
                  at the jaw angles (go) and below (the chin, the under-chin and - fading with the mask - the neck)
  chin_point      f = 1 - 0.3 chin_point (1 - ramp(li .. me)) ramp(me - 2.5 cm .. me) (fade under the chin) on the chin
                  proper only (cosine out between |x - midline| = 3 and 5.5 cm: the jaw angles / neck sides are left)
  neck_under_jaw  below the menton (cosine in over FACE_NECK_RAMP_M = 3 cm): width f = neck_under_jaw and the front
                  of the neck pulled back toward z_ear - 12 mm by the same factor (C1 soft clamp: nothing behind)
Nothing above the cheekbone line moves (taper) - eyes, brows, nose, forehead, cranium, scalp, ears.

Identity (`identity*`, dancecap.face_identity; needs the FLAME files once per rest head, then the per-take cache): a
per-vertex displacement of the head toward the dancer's fitted FLAME identity - lower face, cheeks, jaw, chin and the
under-chin; exactly 0 on the eye region / eyeballs / brows, forehead, nose, scalp, ears and the neck-shoulder seam
(the transition from the protected parts is at least 3.8 cm of surface long from the ears, 1.2 cm from the nose: no
fold at the ear root or the nostrils). The neck below the chin is not taken from FLAME's shape prior (the photos do not
constrain it, and it would print its own larynx relief into the throat as a ledge): it is the BIHARMONIC (thin-plate:
value and slope continuous) extension of the chin's move down to the seam. identity_mouth keeps part of the mouth in
place (the transplant moves it with the lower face). It is applied first (re-targeted to every take's rest head and
cached: 02 has other betas than 04); the factors above then act on the new face.

    python -m dancecap.shape_edits <take> [--role follow]     -> report (before / after numbers, no files written)
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

ROLES = ("lead", "follow")
J_PELVIS, J_SPINE1, J_SPINE2, J_SPINE3 = 0, 3, 6, 9
ARM_JOINTS = (13, 14, 16, 17, 18, 19, 20, 21) + tuple(range(25, 55))  # collars, arms, hands
FOOT = {"L": (7, 10), "R": (8, 11)}  # (ankle, foot) SMPL-X joints
# band (fractions of D = spine2.y - pelvis.y): zero / full / full / zero
BAND = (("pelvis", -0.15), ("pelvis", 0.25), ("spine2", -0.05), ("spine2", 0.25))
# Unity sneaker outer mesh length minus the rest foot length L, per style (head-movement Assets/Shoes SneakerShape:
# heel margin 6 mm + toe margin 11 mm (+3 mm on platforms) + the outsole's bevel / flare; measured on the built
# meshes of 02_LarissaKadu in edit mode, 2026-10-07: court_platform 30.2 / 30.2 mm, runner 25.9 / 25.9 mm)
# 2026-10-07 evening the SneakerShape margins were cut (toe 11 -> 7, heel 6 -> 4, platform extra 3 -> 1 mm; sole flare
# ~1 mm less): ~9 mm less on the platform style, ~7 mm on the others (estimated from the margins, not re-measured)
SNEAKER_ALLOWANCE_M = {"court_platform": 0.0212, "runner": 0.0189, "neutral": 0.020}
FOOT_SCALE_RANGE = (0.80, 1.0)


# ------------------------------------------------------------------------------------------------ config

def take_toml(take: str) -> dict:
    p = PACKAGE_ROOT / "takes" / f"{take}.toml"
    return tomllib.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def config_for(take: str, cfg: dict | None = None) -> dict:
    """{role: shape_edits table} of the take TOML (roles without a table are absent)."""
    cfg = take_toml(take) if cfg is None else cfg
    out = {}
    for r in ROLES:
        se = ((cfg.get("dancers") or {}).get(r) or {}).get("shape_edits")
        if se:
            out[r] = dict(se)
    return out


def _pair(x, name: str) -> tuple[float, float]:
    v = np.atleast_1d(np.asarray(x, float))
    if v.size == 1:
        return float(v[0]), float(v[0])
    if v.size == 2:
        return float(v[0]), float(v[1])
    raise ValueError(f"{name}: a number or [width, depth], got {x!r}")


# ------------------------------------------------------------------------------------------------ midsection

def _ramp(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """0 at x <= a, 1 at x >= b, cosine in between (C1)."""
    t = np.clip((np.asarray(x, float) - a) / max(b - a, 1e-9), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def band(J: np.ndarray) -> tuple[float, float, float, float]:
    """(zero below, full from, full to, zero above) heights of the midsection band from the rest joints."""
    D = float(J[J_SPINE2, 1] - J[J_PELVIS, 1])
    if D <= 0.05:
        raise ValueError(f"spine2 is not above the pelvis in the rest pose (D = {D:.3f} m)")
    ref = {"pelvis": float(J[J_PELVIS, 1]), "spine2": float(J[J_SPINE2, 1])}
    return tuple(ref[j] + f * D for j, f in BAND)  # type: ignore[return-value]


def band_weight(y: np.ndarray, b: tuple[float, float, float, float]) -> np.ndarray:
    return _ramp(y, b[0], b[1]) * (1.0 - _ramp(y, b[2], b[3]))


def arm_weight(weights: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(weights, float)[:, list(ARM_JOINTS)].sum(1), 0.0, 1.0)


def slice_centres(v: np.ndarray, faces: np.ndarray, mask: np.ndarray, y_lo: float, y_hi: float, step: float = 0.01,
                  sigma: float = 0.02, depth_pivot: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scaling centre of the mesh section through the masked faces every `step` m of height: x = bounding-box middle,
    z = back + depth_pivot x depth (SMPL-X faces +z: 0 keeps the back surface, 0.5 = middle, 1 keeps the front),
    Gaussian-smoothed over height (sigma m). -> (ys, cx, cz)."""
    ys = np.arange(y_lo, y_hi + 0.5 * step, step)
    cx, cz = np.full(len(ys), np.nan), np.full(len(ys), np.nan)
    for i, y in enumerate(ys):
        P = section_points(v, faces, mask, float(y))
        if len(P) >= 8:
            cx[i] = 0.5 * (P[:, 0].min() + P[:, 0].max())
            cz[i] = P[:, 2].min() + depth_pivot * np.ptp(P[:, 2])
    ok = np.isfinite(cx)
    if ok.sum() < 2:
        raise ValueError("too few torso vertices in the midsection band")
    cx, cz = np.interp(ys, ys[ok], cx[ok]), np.interp(ys, ys[ok], cz[ok])
    k = np.arange(-3 * sigma, 3 * sigma + 0.5 * step, step)
    g = np.exp(-0.5 * (k / sigma) ** 2)
    g /= g.sum()
    pad = len(k) // 2
    sm = lambda a: np.convolve(np.pad(a, pad, mode="edge"), g, mode="valid")  # noqa: E731
    return ys, sm(cx), sm(cz)


def section_points(v: np.ndarray, faces: np.ndarray, keep: np.ndarray, y: float) -> np.ndarray:
    """Points where the mesh edges of faces with all three vertices in `keep` cross the plane at height y."""
    f = faces[keep[faces].all(1)]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    a, b = v[e[:, 0]], v[e[:, 1]]
    da, db = a[:, 1] - y, b[:, 1] - y
    cross = (da * db < 0)
    t = da[cross] / (da[cross] - db[cross])
    return a[cross] + t[:, None] * (b[cross] - a[cross])


def midsection_dims(v: np.ndarray, J: np.ndarray, weights: np.ndarray, faces: np.ndarray, step: float = 0.01) -> dict:
    """Width (x range) and depth (z range) of the torso's mesh cross-sections every `step` over the band's
    full-weight plateau: the waist (narrowest section) and the plateau means. Heights relative to the lowest vertex."""
    b = band(J)
    torso = arm_weight(weights) < 0.5
    y0 = float(v[:, 1].min())
    rows = []
    for y in np.arange(b[1], b[2] + 0.5 * step, step):
        P = section_points(v, faces, torso, float(y))
        if len(P) >= 8:
            rows.append((y, np.ptp(P[:, 0]), np.ptp(P[:, 2])))
    if not rows:
        raise ValueError("no torso cross-section on the midsection plateau")
    a = np.asarray(rows)
    i = int(np.argmin(a[:, 1]))
    return {"band_m": [round(x - y0, 4) for x in b], "waist_height_m": round(float(a[i, 0] - y0), 4),
            "waist_width_m": round(float(a[i, 1]), 4), "waist_depth_m": round(float(a[i, 2]), 4),
            "plateau_width_mean_m": round(float(a[:, 1].mean()), 4),
            "plateau_depth_mean_m": round(float(a[:, 2].mean()), 4), "sections": len(rows)}


def scale_midsection(v: np.ndarray, J: np.ndarray, weights: np.ndarray, faces: np.ndarray, sw: float, sd: float,
                     depth_pivot: float = 0.5) -> np.ndarray:
    """Vertices with the midsection band's width scaled by sw and depth by sd (see the module docstring)."""
    b = band(J)
    keep = 1.0 - arm_weight(weights)
    w = band_weight(v[:, 1], b) * keep
    torso = keep > 0.5
    ys, cx, cz = slice_centres(v, faces, torso, b[0] - 0.03, b[3] + 0.03, depth_pivot=depth_pivot)
    yc = np.clip(v[:, 1], ys[0], ys[-1])
    ccx, ccz = np.interp(yc, ys, cx), np.interp(yc, ys, cz)
    out = np.array(v, float, copy=True)
    out[:, 0] = ccx + (1.0 - (1.0 - sw) * w) * (v[:, 0] - ccx)
    out[:, 2] = ccz + (1.0 - (1.0 - sd) * w) * (v[:, 2] - ccz)
    return out


# ------------------------------------------------------------------------------------------------ feet

def _foot_frame(v: np.ndarray, J: np.ndarray, weights: np.ndarray, side: str) -> dict:
    """shoes.foot_frame on an unsplit world-frame mesh (same landmarks / axes the sneaker builder uses)."""
    from .export_unity import Skin
    from .shoes import foot_frame
    sk = Skin(vertices=v, faces=np.zeros((0, 3), np.int64), weights=weights, joints=J, parents=np.zeros(55, int),
              model="", betas=np.zeros(0))
    return foot_frame(sk, side, split={})


def foot_lengths(v: np.ndarray, J: np.ndarray, weights: np.ndarray) -> dict:
    return {s: round(float(_foot_frame(v, J, weights, s)["L"]), 4) for s in FOOT}


def scale_feet(v: np.ndarray, J: np.ndarray, weights: np.ndarray, sx: float, sz: float, sy: float) -> np.ndarray:
    """Both feet scaled about the ankle's vertical line on the sole plane (see the module docstring)."""
    out = np.array(v, float, copy=True)
    for side, (ja, jf) in FOOT.items():
        ff = _foot_frame(v, J, weights, side)
        y0 = ff["y0"]
        fwd, lat = np.asarray(ff["fwd"], float), np.asarray(ff["side_ax"], float)
        piv = np.array([J[ja, 0], y0, J[ja, 2]])
        ankle_h = float(J[ja, 1] - y0)
        wf = np.clip(weights[:, ja] + weights[:, jf], 0.0, 1.0)
        h = v[:, 1] - y0
        wf = wf * (1.0 - _ramp(h, ankle_h - 0.015, ankle_h + 0.035))
        idx = np.nonzero(wf > 1e-6)[0]
        rel = v[idx] - piv
        x, s = rel @ fwd, rel @ lat
        tgt = piv + (sx * x)[:, None] * fwd + (sz * s)[:, None] * lat + np.c_[np.zeros(len(idx)), sy * h[idx],
                                                                               np.zeros(len(idx))]
        # tgt's y: piv.y (= y0) + sy * h; fwd / lat are horizontal. Displacements ADD: a few vertices carry a little
        # weight of the other foot's joints (w 0.001), so overwriting would undo the first foot's edit
        out[idx] += wf[idx, None] * (tgt - v[idx])
    return out


def shoe_style(work: Path, role: str) -> str:
    p = work / "avatar" / "shoes.json"
    if p.exists():
        try:
            return str(json.loads(p.read_text(encoding="utf-8"))["dancers"][role]["style"])
        except (KeyError, ValueError):
            pass
    return "neutral"


# ------------------------------------------------------------------------------------------------ face (lower face)

# FLAME 2020 static iBUG-68 landmarks on SMPL-X: name -> (iBUG index, SMPL-X vertices, barycentric weights). Derived
# 2026-10-08 from FLAME's landmark_embedding (full_lmk_faces_idx / full_lmk_bary_coords) through
# models/flame/SMPL-X__FLAME_vertex_ids.npy (both git-ignored; tests/test_shape_edits_face.py re-derives the table when
# they are installed). Same topology for the plain and MAMMA's locked-head SMPL-X (10475 vertices).
FACE_LMK = {
    "c0": (0, (2210, 2212, 2213), (0.97808, 0.01847, 0.00345)),
    "c16": (16, (728, 731, 730), (0.97808, 0.00345, 0.01847)),
    "c1": (1, (1935, 2705, 1963), (0.51624, 0.11194, 0.37181)),
    "c15": (15, (568, 1569, 560), (0.37181, 0.11194, 0.51624)),
    "c4": (4, (8736, 8774, 8773), (0.37337, 0.36149, 0.26514)),
    "c12": (12, (9102, 9033, 9101), (0.36149, 0.37337, 0.26514)),
    "c5": (5, (8740, 8738, 8772), (0.60139, 0.15376, 0.24486)),
    "c11": (11, (9098, 9035, 9039), (0.24488, 0.15371, 0.60141)),
    "c6": (6, (8743, 8769, 8770), (0.40301, 0.09364, 0.50334)),
    "c10": (10, (9095, 9089, 9046), (0.50334, 0.09364, 0.40301)),
    "c7": (7, (8766, 8746, 8744), (0.57491, 0.26649, 0.15860)),
    "c9": (9, (9086, 9047, 9050), (0.57491, 0.15860, 0.26649)),
    "menton": (8, (9082, 8765, 8793), (0.14025, 0.65152, 0.20823)),
    "nasion": (27, (2194, 8997, 8986), (0.00000, 0.97400, 0.02600)),
    "nose_tip": (30, (8970, 9008, 1819), (0.86735, 0.12562, 0.00703)),
    "subnasale": (33, (1692, 9000, 8951), (0.00482, 0.69911, 0.29607)),
    "eye_r_out": (36, (2440, 2428, 2437), (0.12233, 0.17113, 0.70653)),
    "eye_l_out": (45, (1182, 1175, 1154), (0.12233, 0.70653, 0.17113)),
    "mouth_r": (48, (2845, 2715, 2714), (0.47118, 0.50573, 0.02309)),
    "mouth_l": (54, (1730, 1578, 1579), (0.47118, 0.02309, 0.50573)),
    "upper_lip": (51, (1657, 8987, 8990), (0.00028, 0.81925, 0.18046)),
    "lower_lip": (57, (1850, 8950, 8947), (0.00011, 0.03358, 0.96631)),
    "stomion_up": (62, (8977, 8975, 2786), (0.97835, 0.02165, 0.0)),
    "stomion_lo": (66, (8948, 8953, 2937), (0.02154, 0.97845, 0.00001)),
}
FACE_KEYS = ("lower_face_height_scale", "chin_height_scale", "chin_width_scale", "jaw_width_scale", "cheek_scale",
             "lower_face_depth_scale", "submental_height_scale", "jaw_taper", "cheekbone_keep", "chin_point",
             "neck_under_jaw", "chin_forward_mm", "identity_weight", "identity_neck", "identity_mouth")
FACE_STR_KEYS = ("identity",)
FACE_RANGE = (0.7, 1.15)       # allowed factors (submental_height_scale: FACE_SUBMENTAL_RANGE - a thin band)
FACE_SUBMENTAL_RANGE = (0.4, 1.15)
FACE_RANGES = {"submental_height_scale": FACE_SUBMENTAL_RANGE, "jaw_taper": (0.7, 1.0), "cheekbone_keep": (0.0, 1.0),
               "chin_point": (0.0, 1.0), "neck_under_jaw": (0.7, 1.0), "identity_weight": (0.0, 1.5),
               "identity_neck": (0.0, 1.0), "identity_mouth": (0.0, 1.0), "chin_forward_mm": (-6.0, 12.0)}
FACE_DEFAULTS = {"cheekbone_keep": 0.0, "chin_point": 0.0, "identity_weight": 1.0, "identity_neck": 1.0,
                 "identity_mouth": 1.0, "chin_forward_mm": 0.0}  # others: 1.0
FACE_IDENTITY_NECK_M = 0.015   # identity_neck applies from this far under the menton down (cosine in from the menton)
FACE_TAPER_BACK_M = 0.012      # taper mask: front edge moved back by this much below the ear lobes (1.2 cm: the jaw angles
                               # come in, the skin behind the jaw - mastoid / neck top, FLAME "scalp" mask - hardly moves)
FACE_TAPER_FADE_M = 0.040      # taper mask: fade down the neck below the throat level (the height edits' fade is longer:
                               # the taper is a width change of the jaw and the upper neck only, the seam stays put)
FACE_LOBE_FRAC = 0.32          # ear-lobe bottom height between the stomion (0) and the subnasale (1) (SMPL-X ear)
FACE_CHIN_POINT_MAX = 0.3      # chin_point = 1: the chin's width x 0.7 at the menton level
FACE_CHIN_POINT_X_M = (0.030, 0.055)  # ... only on the chin proper: full for |x - midline| < 3 cm, 0 beyond 5.5 cm
                                      # (the jaw angles and the sides of the neck are not 'the chin')
FACE_CHIN_FWD_TOP_M = 0.016    # chin_forward_mm: full this far under the lower-lip fold (0 at the fold itself) ...
FACE_CHIN_FWD_UNDER_M = 0.030  # ... and again 0 this far under the menton (the shift fades out down the under-chin)
FACE_NECK_AXIS_BACK_M = 0.012  # neck_under_jaw: the front of the neck is pulled toward z_ear - this
FACE_NECK_RAMP_M = 0.03        # neck_under_jaw: cosine in from the menton level down to this far under it
FACE_RAMP_M = 0.004            # half-width of the cosine edges of the height windows
FACE_THROAT_M = 0.015          # the under-chin band: the menton .. this far below it (submental_height_scale) ...
FACE_NECK_FADE_M = 0.06        # ... below it every field fades out down the front of the neck over this length
FACE_FRONT_M = (-0.005, 0.025)  # front mask: 0 .. 1 over this z range relative to the ear-level contour (iBUG 0 / 16)
FACE_FRONT_NECK_M = (-0.03, 0.045)  # ... widened below the throat level (the lifted neck: less shear on its sides)
FACE_LATERAL_M = (0.085, 0.105)  # lateral mask: 1 .. 0 over this |x - midline| range (never the shoulders / hands)


def face_landmarks(v: np.ndarray) -> dict:
    """{name: (3,) position} of FACE_LMK on SMPL-X-topology vertices v (unsplit)."""
    v = np.asarray(v, float)
    return {n: (v[list(ids)] * np.asarray(b, float)[:, None]).sum(0) for n, (_, ids, b) in FACE_LMK.items()}


def face_frame(v: np.ndarray) -> dict:
    """Anchor heights / planes of the lower-face edit (rest pose, y up, z forward), metres."""
    L = face_landmarks(v)
    y = {n: float(p[1]) for n, p in L.items()}
    fr = {"x_mid": float(np.mean([L[n][0] for n in ("nasion", "subnasale", "menton", "c1", "c15", "c4", "c12")])),
          "y_n": y["nasion"], "y_zy": 0.5 * (y["c1"] + y["c15"]), "y_sn": y["subnasale"],
          "y_sto": 0.5 * (y["stomion_up"] + y["stomion_lo"]), "y_li": y["lower_lip"], "y_go": 0.5 * (y["c4"] + y["c12"]),
          "y_me": y["menton"], "z_ear": 0.5 * float(L["c0"][2] + L["c16"][2]),
          "z_go": 0.5 * float(L["c4"][2] + L["c12"][2])}
    order = ("y_me", "y_go", "y_li", "y_sto", "y_sn", "y_zy", "y_n")
    if any(fr[a] >= fr[b] for a, b in zip(order, order[1:])):
        raise ValueError(f"face landmarks out of order (not an upright SMPL-X rest head?): "
                         f"{ {k: round(fr[k], 4) for k in order} }")
    return fr


def face_measures(v: np.ndarray) -> dict:
    """Lower-face proportions (mm; widths = x extent, heights = y extent, the rest head is upright)."""
    L = face_landmarks(v)

    def wx(a, b):
        return round(1000 * abs(float(L[a][0] - L[b][0])), 2)

    def hy(a, b):
        return round(1000 * float(L[a][1] - L[b][1]), 2)

    sto = 0.5 * (L["stomion_up"] + L["stomion_lo"])
    m = {"face_height_n_me_mm": hy("nasion", "menton"), "lower_face_sn_me_mm": hy("subnasale", "menton"),
         "upper_lip_sn_sto_mm": round(1000 * float(L["subnasale"][1] - sto[1]), 2),
         "chin_sto_me_mm": round(1000 * float(sto[1] - L["menton"][1]), 2), "chin_li_me_mm": hy("lower_lip", "menton"),
         "outer_canthal_mm": wx("eye_r_out", "eye_l_out"), "cheek_c1_15_mm": wx("c1", "c15"),
         "jaw_angle_c4_12_mm": wx("c4", "c12"), "c5_11_mm": wx("c5", "c11"), "chin_c6_10_mm": wx("c6", "c10"),
         "chin_c7_9_mm": wx("c7", "c9"), "mouth_mm": wx("mouth_r", "mouth_l"),
         "chin_projection_vs_jaw_angle_mm": round(1000 * float(L["menton"][2] - 0.5 * (L["c4"][2] + L["c12"][2])), 2)}
    m["ratios"] = {"face_height_over_occ": round(m["face_height_n_me_mm"] / m["outer_canthal_mm"], 4),
                   "lower_face_over_occ": round(m["lower_face_sn_me_mm"] / m["outer_canthal_mm"], 4),
                   "cheek_over_face_height": round(m["cheek_c1_15_mm"] / m["face_height_n_me_mm"], 4),
                   "jaw_angle_over_cheek": round(m["jaw_angle_c4_12_mm"] / m["cheek_c1_15_mm"], 4),
                   "chin_c6_10_over_cheek": round(m["chin_c6_10_mm"] / m["cheek_c1_15_mm"], 4),
                   "chin_c7_9_over_cheek": round(m["chin_c7_9_mm"] / m["cheek_c1_15_mm"], 4),
                   "chin_sto_me_over_upper_lip": round(m["chin_sto_me_mm"] / m["upper_lip_sn_sto_mm"], 4)}
    return m


def face_factors(cfg: dict) -> dict:
    """The face table's numeric factors (all FACE_KEYS; default 1, FACE_DEFAULTS) - validated."""
    face = cfg.get("face") or {}
    unknown = set(face) - set(FACE_KEYS) - set(FACE_STR_KEYS)
    if unknown:
        raise ValueError(f"shape_edits.face: unknown keys {sorted(unknown)} "
                         f"(known: {', '.join(FACE_KEYS + FACE_STR_KEYS)})")
    f = {k: float(face.get(k, FACE_DEFAULTS.get(k, 1.0))) for k in FACE_KEYS}
    rng = {k: FACE_RANGES.get(k, FACE_RANGE) for k in FACE_KEYS}
    bad = {k: x for k, x in f.items() if not rng[k][0] <= x <= rng[k][1]}
    if bad:
        raise ValueError(f"shape_edits.face: {bad} outside the allowed range ({ {k: rng[k] for k in bad} })")
    return f


def face_identity_cfg(cfg: dict) -> str | None:
    """The face table's identity file (None = no identity transplant)."""
    ident = (cfg.get("face") or {}).get("identity")
    if ident is not None and not isinstance(ident, str):
        raise ValueError(f"shape_edits.face.identity: a path string, got {ident!r}")
    return ident or None


def identity_neck_factor(v: np.ndarray, neck: float) -> np.ndarray:
    """(V,) factor on the identity displacement: 1 above the menton -> `neck` from FACE_IDENTITY_NECK_M under it down
    (cosine, C1). The fitted FLAME identity is constrained by the photos down to the jaw line only; its under-chin /
    neck comes from the shape prior and moves the front of the neck by up to ~2 cm."""
    fr = face_frame(v)
    return 1.0 - (1.0 - neck) * (1.0 - _ramp(np.asarray(v, float)[:, 1], fr["y_me"] - FACE_IDENTITY_NECK_M, fr["y_me"]))


def _taper_active(f: dict) -> bool:
    return f["jaw_taper"] != 1.0 or f["chin_point"] != 0.0 or f["neck_under_jaw"] != 1.0


def _window(y: np.ndarray, lo: float, hi: float, a: float) -> np.ndarray:
    """1 on [lo, hi], cosine edges of half-width a centred on lo and hi (C1)."""
    return _ramp(y, lo - a, lo + a) * (1.0 - _ramp(y, hi - a, hi + a))


def face_lift(y: np.ndarray, fr: dict, s_lower: float, s_chin: float, s_sub: float = 1.0, a: float = FACE_RAMP_M,
              step: float = 1e-4) -> np.ndarray:
    """Upward displacement D(y) of the height edit (before the mask): the integral from y up of the compression rate
    (1 - s_lower) on [me, sn] + (1 - s_chin) on [me, li] + (1 - s_sub) on [me - FACE_THROAT_M, me]. 0 above sn + a,
    constant below me - FACE_THROAT_M - a."""
    y = np.asarray(y, float)
    y_th = fr["y_me"] - FACE_THROAT_M
    top = fr["y_sn"] + 2 * a
    bot = min(float(y.min()), y_th - 2 * a) - step
    g = np.arange(bot, top + step, step)
    c = ((1.0 - s_lower) * _window(g, fr["y_me"], fr["y_sn"], a) + (1.0 - s_chin) * _window(g, fr["y_me"], fr["y_li"], a)
         + (1.0 - s_sub) * _window(g, y_th, fr["y_me"], a))
    seg = 0.5 * (c[1:] + c[:-1]) * step
    D = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]
    return np.interp(y, g, D)


def face_neck_fade(fr: dict, f: dict) -> float:
    """Length (m) of the fade down the neck: FACE_NECK_FADE_M, longer for a large lift (>= pi x the throat's lift,
    so the stretch of the front of the neck stays under ~50 %)."""
    y_th = fr["y_me"] - FACE_THROAT_M
    lift = float(face_lift(np.array([y_th - 2 * FACE_RAMP_M]), fr, f["lower_face_height_scale"],
                           f["chin_height_scale"], f["submental_height_scale"])[0])
    return max(FACE_NECK_FADE_M, np.pi * abs(lift))


def face_mask(v: np.ndarray, weights: np.ndarray, fr: dict, fade: float = FACE_NECK_FADE_M) -> np.ndarray:
    """(V,) the C1 mask every face field is multiplied by (front x neck x lateral x non-arm)."""
    v = np.asarray(v, float)
    y_th = fr["y_me"] - FACE_THROAT_M
    b = _ramp(v[:, 1], y_th, fr["y_me"])                      # 1 = face (above the menton), 0 = neck
    z0 = fr["z_ear"] + b * FACE_FRONT_M[0] + (1 - b) * FACE_FRONT_NECK_M[0]
    z1 = fr["z_ear"] + b * FACE_FRONT_M[1] + (1 - b) * FACE_FRONT_NECK_M[1]
    m = np.clip((v[:, 2] - z0) / (z1 - z0), 0.0, 1.0)
    m = 0.5 - 0.5 * np.cos(np.pi * m)
    m = m * _ramp(v[:, 1], y_th - fade, y_th)
    m = m * (1.0 - _ramp(np.abs(v[:, 0] - fr["x_mid"]), *FACE_LATERAL_M))
    return m * (1.0 - arm_weight(weights))


def face_weights(y: np.ndarray, fr: dict) -> dict:
    """Height profiles (before the mask) of the width / depth factors."""
    y = np.asarray(y, float)
    mid = 0.5 * (fr["y_go"] + fr["y_sn"])
    return {"jaw": 1.0 - _ramp(y, fr["y_go"], mid),
            "chin": 1.0 - _ramp(y, fr["y_me"], fr["y_go"]),
            "cheek": _ramp(y, fr["y_go"], mid) * (1.0 - _ramp(y, fr["y_sn"], fr["y_zy"])),
            "depth": 1.0 - _ramp(y, fr["y_me"], fr["y_sto"])}


def face_taper_mask(v: np.ndarray, weights: np.ndarray, fr: dict, fade: float = FACE_TAPER_FADE_M) -> np.ndarray:
    """(V,) C1 mask of the petite-taper fields: face_mask with its front edge FACE_TAPER_BACK_M further back below
    the ear lobes (the posterior jaw / jaw angles), unchanged at and above the lobes (ears) and on the neck."""
    v = np.asarray(v, float)
    y_th = fr["y_me"] - FACE_THROAT_M
    y_lobe = fr["y_sto"] + FACE_LOBE_FRAC * (fr["y_sn"] - fr["y_sto"])
    b = _ramp(v[:, 1], y_th, fr["y_me"])
    back = (1.0 - _ramp(v[:, 1], y_lobe - 0.024, y_lobe - 0.008)) * FACE_TAPER_BACK_M * b   # full below the lobe
    z0 = fr["z_ear"] + b * FACE_FRONT_M[0] + (1 - b) * FACE_FRONT_NECK_M[0] - back
    z1 = fr["z_ear"] + b * FACE_FRONT_M[1] + (1 - b) * FACE_FRONT_NECK_M[1] - back
    m = np.clip((v[:, 2] - z0) / (z1 - z0), 0.0, 1.0)
    m = 0.5 - 0.5 * np.cos(np.pi * m)
    m = m * _ramp(v[:, 1], y_th - fade, y_th)
    m = m * (1.0 - _ramp(np.abs(v[:, 0] - fr["x_mid"]), *FACE_LATERAL_M))
    return m * (1.0 - arm_weight(weights))


def face_taper_weights(y: np.ndarray, fr: dict, keep: float = 0.0) -> dict:
    """Height profiles (before the taper mask) of the petite-taper fields."""
    y = np.asarray(y, float)
    t = 1.0 - _ramp(y, fr["y_go"], fr["y_zy"])                 # 0 at the cheekbone line and above, 1 at the jaw angles
    return {"taper": t ** (1.0 + 2.0 * keep),
            "point": (1.0 - _ramp(y, fr["y_me"], fr["y_li"])) * _ramp(y, fr["y_me"] - 0.025, fr["y_me"]),
            "neck": 1.0 - _ramp(y, fr["y_me"] - FACE_NECK_RAMP_M, fr["y_me"])}


def chin_forward_weight(v: np.ndarray, fr: dict) -> np.ndarray:
    """(V,) C1 weight of the forward chin shift (chin_forward_mm), before the face mask: 1 on the chin proper (from
    FACE_CHIN_FWD_TOP_M under the lower-lip fold down to the menton, |x - midline| < 3 cm), cosine to 0 above (at the
    lower-lip fold: the lips and the mentolabial fold stay) and under the chin (FACE_CHIN_FWD_UNDER_M below the menton:
    the throat and the neck stay) and out to 5.5 cm from the midline (the jaw angles stay)."""
    v = np.asarray(v, float)
    y = v[:, 1]
    top = min(FACE_CHIN_FWD_TOP_M, 0.7 * (fr["y_li"] - fr["y_me"]))
    wy = (1.0 - _ramp(y, fr["y_li"] - top, fr["y_li"])) * _ramp(y, fr["y_me"] - FACE_CHIN_FWD_UNDER_M, fr["y_me"] - 0.004)
    return wy * (1.0 - _ramp(np.abs(v[:, 0] - fr["x_mid"]), *FACE_CHIN_POINT_X_M))


def _soft_pos(d: np.ndarray, a: float = 0.01) -> np.ndarray:
    """C1 max(d, 0): 0 for d <= 0, d^2 / 2a up to a, then d - a / 2."""
    d = np.asarray(d, float)
    return np.where(d <= 0, 0.0, np.where(d < a, d * d / (2 * a), d - a / 2))


def edit_face(v: np.ndarray, weights: np.ndarray, f: dict, fr: dict | None = None) -> np.ndarray:
    """Vertices with the lower-face factors f (face_factors; the geometric ones - the identity transplant is applied by
    apply_vertices) applied (see the module docstring). All fields are evaluated on the input positions and added (no
    order dependence). fr = face_frame of the head the points belong to (default: of v itself, so v must then be the
    whole SMPL-X mesh; pass the frame to edit a subset or a synthetic point set)."""
    v = np.asarray(v, float)
    fr = face_frame(v) if fr is None else fr
    fade = face_neck_fade(fr, f)
    m = face_mask(v, weights, fr, fade)
    w = face_weights(v[:, 1], fr)
    out = np.array(v, copy=True)
    out[:, 1] += m * face_lift(v[:, 1], fr, f["lower_face_height_scale"], f["chin_height_scale"],
                               f["submental_height_scale"])
    fx = ((1.0 - (1.0 - f["jaw_width_scale"]) * w["jaw"]) * (1.0 - (1.0 - f["chin_width_scale"]) * w["chin"])
          * (1.0 - (1.0 - f["cheek_scale"]) * w["cheek"]))
    out[:, 0] -= m * (1.0 - fx) * (v[:, 0] - fr["x_mid"])       # displacements: exactly 0 where the mask is 0
    gz = 1.0 - (1.0 - f["lower_face_depth_scale"]) * w["depth"]
    out[:, 2] -= m * (1.0 - gz) * (v[:, 2] - fr["z_go"])
    if _taper_active(f):
        mt = face_taper_mask(v, weights, fr, FACE_TAPER_FADE_M)
        tw = face_taper_weights(v[:, 1], fr, f["cheekbone_keep"])
        ft = (1.0 - (1.0 - f["jaw_taper"]) * tw["taper"]) * (1.0 - (1.0 - f["neck_under_jaw"]) * tw["neck"])
        dx = v[:, 0] - fr["x_mid"]
        fp = FACE_CHIN_POINT_MAX * f["chin_point"] * tw["point"] * (1.0 - _ramp(np.abs(dx), *FACE_CHIN_POINT_X_M))
        out[:, 0] -= mt * (1.0 - ft * (1.0 - fp)) * dx
        zc = fr["z_ear"] - FACE_NECK_AXIS_BACK_M
        out[:, 2] -= mt * (1.0 - f["neck_under_jaw"]) * tw["neck"] * _soft_pos(v[:, 2] - zc)
    if f["chin_forward_mm"] != 0.0:
        out[:, 2] += 1e-3 * f["chin_forward_mm"] * m * chin_forward_weight(v, fr)
    return out


def face_protected(v: np.ndarray, weights: np.ndarray) -> dict:
    """{region: (V,) bool} that the face edit must leave in place: above the cheekbone line (eyes, brows, nose
    bridge, forehead, cranium, scalp) and behind the front mask (ears, the back of the head and the neck: hair
    roots, nape)."""
    fr = face_frame(v)
    v = np.asarray(v, float)
    head = np.asarray(weights, float)[:, [15, 22, 23, 24]].sum(1) > 0.5
    return {"above_cheekbones": head & (v[:, 1] > fr["y_zy"]),
            "nose": head & (v[:, 1] > fr["y_sn"] + 2 * FACE_RAMP_M) & (np.abs(v[:, 0] - fr["x_mid"]) < 0.02)
            & (v[:, 2] > fr["z_go"] + 0.03),
            "behind_front_mask": (head & (v[:, 2] < fr["z_ear"] + FACE_FRONT_M[0]) & (v[:, 1] > fr["y_me"]))
            | (v[:, 2] < fr["z_ear"] + FACE_FRONT_NECK_M[0])}


# ------------------------------------------------------------------------------------------------ apply

def reference_dims(ref_take: str, role: str, smplx_dir: str = "smplx") -> dict:
    """Unedited midsection of another take's fitted body (its smplx/<role>.npz betas on the same model choice)."""
    from .export_unity import model_for, shaped_skin
    p = PACKAGE_ROOT / "work" / ref_take / smplx_dir / f"{role}.npz"
    if not p.exists():
        raise FileNotFoundError(f"midsection_reference: {p} missing")
    with np.load(p, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    sk = shaped_skin(model_for(d), d["betas"])
    return midsection_dims(sk.vertices, sk.joints, sk.weights, sk.faces)


def plan(cfg: dict, v: np.ndarray, J: np.ndarray, weights: np.ndarray, faces: np.ndarray, role: str,
         work: Path) -> dict:
    """The factors one dancer's config resolves to on this rest body (no vertices moved)."""
    out: dict = {}
    if "midsection_scale" in cfg:
        sw, sd = _pair(cfg["midsection_scale"], "midsection_scale")
        own = midsection_dims(v, J, weights, faces)
        out["midsection"] = {"requested": [sw, sd], "own": own}
        ref = cfg.get("midsection_reference")
        if ref:
            rd = reference_dims(str(ref), role)
            sw = sw * rd["plateau_width_mean_m"] / own["plateau_width_mean_m"]
            sd = sd * rd["plateau_depth_mean_m"] / own["plateau_depth_mean_m"]
            out["midsection"]["reference"] = {"take": str(ref), **rd}
        if not (0.5 <= sw <= 1.2 and 0.5 <= sd <= 1.2):
            raise ValueError(f"{role}: midsection factors {sw:.3f} / {sd:.3f} outside 0.5..1.2 - check the config")
        out["midsection"]["width_scale"], out["midsection"]["depth_scale"] = round(sw, 5), round(sd, 5)
        piv = float(cfg.get("midsection_depth_pivot", 0.5))
        if not 0.0 <= piv <= 1.0:
            raise ValueError(f"{role}: midsection_depth_pivot {piv} outside 0..1")
        out["midsection"]["depth_pivot"] = piv
    fs = cfg.get("foot_scale")
    if fs is not None:
        f = np.atleast_1d(np.asarray(fs, float))
        if f.size == 1:
            f = np.repeat(f, 3)
        if f.size != 3:
            raise ValueError(f"{role}: foot_scale must be a number or [length, width, height]")
        out["feet"] = {"length_scale": float(f[0]), "width_scale": float(f[1]), "height_scale": float(f[2]),
                       "source": "foot_scale"}
    elif "shoe_length_m" in cfg or "foot_height_scale" in cfg:
        L = foot_lengths(v, J, weights)
        Lm = 0.5 * (L["L"] + L["R"])
        style = shoe_style(work, role)
        allow = SNEAKER_ALLOWANCE_M.get(style, SNEAKER_ALLOWANCE_M["neutral"])
        sx = 1.0
        if "shoe_length_m" in cfg:
            target = float(cfg["shoe_length_m"])
            sx = float(np.clip((target - allow) / Lm, *FOOT_SCALE_RANGE))
        sy = float(cfg.get("foot_height_scale", 1.0))
        out["feet"] = {"length_scale": round(sx, 5), "width_scale": round(sx, 5), "height_scale": sy,
                       "source": "shoe_length_m" if "shoe_length_m" in cfg else "foot_height_scale",
                       "shoe_length_target_m": cfg.get("shoe_length_m"), "style": style, "allowance_m": allow,
                       "foot_length_m": L,
                       "sneaker_length_before_m": round(Lm + allow, 4), "sneaker_length_after_m": round(sx * Lm + allow, 4)}
    if "feet" in out:
        f = out["feet"]
        if not all(0.6 <= f[k] <= 1.2 for k in ("length_scale", "width_scale", "height_scale")):
            raise ValueError(f"{role}: foot scales {f} outside 0.6..1.2")
    if cfg.get("face"):
        out["face"] = {"factors": face_factors(cfg), "own": face_measures(v),
                       "frame_mm": {k: round(1000 * x, 2) for k, x in face_frame(v).items()}}
        ident = face_identity_cfg(cfg)
        if ident:
            out["face"]["identity"] = {"file": ident}
    return out


def apply_vertices(cfg: dict, v: np.ndarray, J: np.ndarray, weights: np.ndarray, faces: np.ndarray, role: str,
                   work: Path) -> tuple[np.ndarray, dict]:
    """-> (edited rest vertices, report). Order: midsection, feet, then the lower face (disjoint regions)."""
    p = plan(cfg, v, J, weights, faces, role, work)
    out = np.asarray(v, float)
    if "midsection" in p:
        m = p["midsection"]
        out = scale_midsection(out, J, weights, faces, m["width_scale"], m["depth_scale"], m["depth_pivot"])
        m["after"] = midsection_dims(out, J, weights, faces)
        m["ratio_after_vs_own"] = [round(m["after"]["plateau_width_mean_m"] / m["own"]["plateau_width_mean_m"], 4),
                                   round(m["after"]["plateau_depth_mean_m"] / m["own"]["plateau_depth_mean_m"], 4)]
        if "reference" in m:
            r = m["reference"]
            m["ratio_after_vs_reference"] = [round(m["after"]["plateau_width_mean_m"] / r["plateau_width_mean_m"], 4),
                                             round(m["after"]["plateau_depth_mean_m"] / r["plateau_depth_mean_m"], 4)]
    if "feet" in p:
        f = p["feet"]
        before = foot_lengths(out, J, weights)
        out = scale_feet(out, J, weights, f["length_scale"], f["width_scale"], f["height_scale"])
        f["foot_length_before_m"], f["foot_length_after_m"] = before, foot_lengths(out, J, weights)
    if "face" in p:
        fc = p["face"]
        before = out
        if "identity" in fc:
            from . import face_identity as fi
            wid = fc["factors"]["identity_weight"]
            if wid > 0:
                disp, mouth_w, irep = fi.cached_displacement(out, fc["identity"]["file"], work, role, wid)
                disp = disp * identity_neck_factor(out, fc["factors"]["identity_neck"])[:, None]
                disp = disp * (1.0 - (1.0 - fc["factors"]["identity_mouth"]) * mouth_w)[:, None]
                out = out + disp
                fc["identity"].update(irep)
                fc["identity"]["max_move_mm"] = round(float(np.linalg.norm(disp, axis=1).max()) * 1000, 2)
        out = edit_face(out, weights, fc["factors"])
        fc["after"] = face_measures(out)
        fc["ratio_after_vs_own"] = {k: round(fc["after"][k] / fc["own"][k], 4) for k in fc["own"]
                                    if k.endswith("_mm") and abs(fc["own"][k]) > 1e-9}
        mv = np.linalg.norm(out - before, axis=1)
        fc["moved_vertices"] = int((mv > 1e-6).sum())
        fc["max_move_mm"] = round(float(mv.max()) * 1000, 2)
        fc["protected_max_move_mm"] = {k: round(1000 * float(mv[s].max()), 4) if s.any() else 0.0
                                       for k, s in face_protected(before, weights).items()}
        from . import face_identity as fi
        if fi.flame_available() and len(before) == fi.N_SX:
            fc["protected_flame_max_move_mm"] = fi.protected_moves(before, out)
    moved = np.linalg.norm(out - v, axis=1)
    p["moved_vertices"] = int((moved > 1e-6).sum())
    p["max_move_mm"] = round(float(moved.max()) * 1000, 2)
    p["config"] = {k: (list(v_) if isinstance(v_, (list, tuple)) else v_) for k, v_ in cfg.items()}
    return out, p


def apply(skin, cfg: dict, role: str, work: Path):
    """export_unity.Skin (unsplit, world frame) -> (edited Skin, report). Joints / weights / faces unchanged."""
    from dataclasses import replace
    v, rep = apply_vertices(cfg, skin.vertices, skin.joints, skin.weights, skin.faces, role, work)
    return replace(skin, vertices=v), rep


def summary(rep: dict) -> dict:
    """The compact version recorded in the skin header / capture.json provenance."""
    s: dict = {"config": rep.get("config")}
    if "midsection" in rep:
        m = rep["midsection"]
        s["midsection"] = {"width_scale": m["width_scale"], "depth_scale": m["depth_scale"],
                           "waist_width_m": [m["own"]["waist_width_m"], m["after"]["waist_width_m"]],
                           "waist_depth_m": [m["own"]["waist_depth_m"], m["after"]["waist_depth_m"]]}
        if "reference" in m:
            s["midsection"]["reference"] = m["reference"]["take"]
    if "feet" in rep:
        f = rep["feet"]
        s["feet"] = {k: f[k] for k in ("length_scale", "width_scale", "height_scale")}
        if "sneaker_length_after_m" in f:
            s["feet"]["sneaker_length_m"] = [f["sneaker_length_before_m"], f["sneaker_length_after_m"]]
    if "face" in rep:
        fc = rep["face"]
        s["face"] = {"factors": {k: x for k, x in fc["factors"].items() if x != FACE_DEFAULTS.get(k, 1.0)}}
        if "identity" in fc:
            s["face"]["identity"] = {k: fc["identity"].get(k) for k in ("file", "weight", "max_move_mm")}
        for k in ("face_height_n_me_mm", "lower_face_sn_me_mm", "chin_c6_10_mm", "jaw_angle_c4_12_mm"):
            if "after" in fc:
                s["face"][k] = [fc["own"][k], fc["after"][k]]
    return s


# ------------------------------------------------------------------------------------------------ partner contact

HANDS = {"right": (21,) + tuple(range(40, 55)), "left": (20,) + tuple(range(25, 40))}


def _posed_frames(d: dict, v_rest: np.ndarray, skin) -> np.ndarray:
    """(T, V, 3) LBS-posed vertices (world frame) of a fitted dancer with rest vertices v_rest (the exporter's FK)."""
    from .export_unity import fk_unity, full_pose, rotvec_to_quat, to_unity_points, to_unity_quats
    J_u = to_unity_points(skin.joints)
    q = to_unity_quats(rotvec_to_quat(full_pose(d)))
    P, R = fk_unity(J_u, skin.parents, to_unity_points(np.asarray(d["transl"], float)), q)
    W = np.asarray(skin.weights, float)
    rel = to_unity_points(v_rest)[:, None, :] - J_u[None]
    out = np.empty((len(P), len(v_rest), 3))
    for k in range(len(P)):
        posed = np.einsum("jab,vjb->vja", R[k], rel) + P[k][None]
        out[k] = np.einsum("vj,vja->va", W, posed)
    return to_unity_points(out)  # back to the world frame (the mirror is its own inverse)


def contact_gap(take: str, cfg: dict | None = None, smplx_dir: str = "smplx", near_m: float = 0.02) -> dict:
    """How the follow's edit moves her torso surface away from (or into) the lead's hands, frame by frame.

    Signed distance of each lead hand (min over its vertices) to the follow's torso (vertices off her arms; sign from
    the nearest vertex's posed normal: negative = the hand is inside her) with her unedited and her edited rest
    shape. Contact frames = unedited distance < near_m. -> per hand: frames, distances before / after (m)."""
    from scipy.spatial import cKDTree

    from .export_unity import model_for, shaped_skin, vertex_normals
    work = PACKAGE_ROOT / "work" / take
    cfg = config_for(take).get("follow", {}) if cfg is None else cfg
    d = {}
    for r in ROLES:
        with np.load(work / smplx_dir / f"{r}.npz", allow_pickle=True) as z:
            d[r] = {k: z[k] for k in z.files}
    sk = {r: shaped_skin(model_for(d[r]), d[r]["betas"]) for r in ROLES}
    v_edit, rep = apply_vertices(cfg, sk["follow"].vertices, sk["follow"].joints, sk["follow"].weights,
                                 sk["follow"].faces, "follow", work)
    torso = np.nonzero(arm_weight(sk["follow"].weights) < 0.5)[0]
    lead = _posed_frames(d["lead"], sk["lead"].vertices, sk["lead"])
    dom = np.asarray(sk["lead"].weights).argmax(1)
    hand_v = {h: np.nonzero(np.isin(dom, js))[0] for h, js in HANDS.items()}
    res: dict = {"near_m": near_m, "edit": summary(rep), "hands": {}}
    dist = {h: {"before": [], "after": []} for h in HANDS}
    for tag, vr in (("before", sk["follow"].vertices), ("after", v_edit)):
        fol = _posed_frames(d["follow"], vr, sk["follow"])
        for k in range(len(fol)):
            n = vertex_normals(fol[k], sk["follow"].faces)
            tree = cKDTree(fol[k][torso])
            for h, hv in hand_v.items():
                dd, ii = tree.query(lead[k][hv])
                j = torso[ii]
                s = np.sign(np.einsum("ij,ij->i", lead[k][hv] - fol[k][j], n[j]))
                dist[h][tag].append(float((dd * np.where(s < 0, -1.0, 1.0)).min()))
    t = np.asarray(d["lead"]["t"], float)
    for h in HANDS:
        b, a = np.asarray(dist[h]["before"]), np.asarray(dist[h]["after"])
        c = b < near_m
        e = {"contact_frames": int(c.sum()), "frames": len(b)}
        if c.any():
            g = a[c] - b[c]
            e.update({"gap_change_mm_median": round(1000 * float(np.median(g)), 1),
                      "gap_change_mm_max": round(1000 * float(g.max()), 1),
                      "before_mm_median": round(1000 * float(np.median(b[c])), 1),
                      "after_mm_median": round(1000 * float(np.median(a[c])), 1),
                      "penetrating_before": int((b[c] < 0).sum()), "penetrating_after": int((a[c] < 0).sum()),
                      "floating_after_gt_10mm": int((a[c] > 0.010).sum()),
                      "floating_after_gt_20mm": int((a[c] > 0.020).sum())})
        res["hands"][h] = e
        res["hands"][h]["per_frame_mm"] = {"t": np.round(t, 3).tolist(), "before": np.round(1000 * b, 1).tolist(),
                                          "after": np.round(1000 * a, 1).tolist()}
    return res


def main(argv: list[str] | None = None) -> None:
    from .export_unity import model_for, shaped_skin
    ap = argparse.ArgumentParser(prog="python -m dancecap.shape_edits", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--role", choices=ROLES)
    ap.add_argument("--smplx-dir", default="smplx")
    ap.add_argument("--contact", action="store_true", help="also: the lead's hands vs the follow's torso, before / "
                                                           "after her edit (contact_gap)")
    ap.add_argument("--out", help="write the report json here")
    a = ap.parse_args(argv)
    work = PACKAGE_ROOT / "work" / a.take
    cfgs = config_for(a.take)
    res: dict = {}
    for r in ([a.role] if a.role else ROLES):
        if r not in cfgs:
            res[r] = "no [dancers.%s.shape_edits]" % r
            continue
        with np.load(work / a.smplx_dir / f"{r}.npz", allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        sk = shaped_skin(model_for(d), d["betas"])
        _, rep = apply(sk, cfgs[r], r, work)
        res[r] = rep
    if a.contact and "follow" in cfgs:
        res["contact"] = contact_gap(a.take, cfgs["follow"], a.smplx_dir)
    text = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    brief = {k: v for k, v in res.items() if k != "contact"}
    if "contact" in res:
        brief["contact"] = {h: {kk: vv for kk, vv in e.items() if kk != "per_frame_mm"}
                            for h, e in res["contact"]["hands"].items()}
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
