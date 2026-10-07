"""Hair groom for the follow's Unity hair: guide-strand roots on her SMPL-X scalp, per-root lengths, volume,
colour and collision capsules, measured from the take -> avatar/hair_groom.json (+ the capture folders).

    python -m dancecap.hair_groom larissa_kadu                 -> avatar/hair_groom.json, review/hair_groom.jpg,
                                                                  copies into every capture of the take
    python -m dancecap.hair_groom larissa_kadu --export-only [--capture 02_LarissaKadu]
    python -m dancecap.hair_groom larissa_kadu --sheet         -> review/hair_sheet.jpg from the Unity review shots
                                                                  (head-movement Tools/playtest_hair.ps1)

Inputs (nothing take-specific in the code): smplx/follow.npz (her fit; the SMPL-X model it names), cameras via
mamma_dyn.load_tracks (sync-corrected world cameras), frames/<cam>, dancers/<cam>/{hair,body,masks} follow masks;
optional avatar/hair_reference.json (the 3D hair-hull measurements of dancecap.hair_reference) - when present it is
the length / tip-level source and this stage's 2D measurement is reported next to it as a cross-check.

Method (CPU, ~1 min)
  1. Rest model: her shaped SMPL-X T-pose (betas of the fit, expression 0, sole at y = 0). Head = vertices whose
     dominant LBS joint is the head (15); head sphere = least-squares sphere through the cranium (head vertices above
     the head joint). Scalp = head vertices above a hairline elevation that depends on the azimuth around the head
     (forehead ~50 deg, temples, above the ears ~20 deg, nape ~-35 deg; 0 deg = facing direction).
  2. Roots: area-uniform candidates on the scalp triangles, farthest-point sampled down to --guides roots (even
     coverage); each root = SMPL-X triangle (3 vertex ids) + barycentric weights, so Unity places it on the exported
     skin of the same model and betas (skin vertex_ids map split vertices back to SMPL-X vertices).
  3. Lengths per root (layered cut): a strand falls over the skull to the head's equator and then hangs to the tip
     level of its azimuth, L = (y_root - y_tip(az)) + r_head (el - sin el); y_tip blends from the FRONT level
     (face-framing sections) to the BACK level, then all lengths are scaled so the crown strand matches the measured
     crown-to-tip back strand; +-5 % seeded jitter (piecey layers). Levels come from hair_reference.json (3D hull)
     when present, else from step 4.
  4. 2D measurement relative to the PROJECTED SMPL-X head, every --step-th frame of every camera: her posed head
     vertices projected (radius r_px = sqrt(hull area / pi), metres per pixel at head depth), hair-mask pixels ->
     drop below the head centre along the projected body-down axis (p98), reach (p98 distance), width in a band
     0.20-0.30 m below the head centre, hair area in head areas; calm views (head angular speed < 90 deg/s) give the
     length estimate (median over views of the p98 drop: projections only shorten), turn views the spread.
  5. Colour: hair-mask pixels (eroded 2 px) of the frame JPEGs, white-balanced per view on her white top (bright,
     unsaturated body-mask pixels inside the projected torso quad) to an assumed albedo of 0.80 -> albedo-like linear
     sRGB; zones along the visible hair (root < 25 % < mid < 75 % < tip of the head-centre distance range); per-camera
     medians, median across cameras; shadow / highlight = luminance bands; strand palette = luminance quantiles.
  6. Colliders: capsules between SMPL-X joints (+ rest-frame offsets) with radii from her rest mesh (p40 distance of
     the segment's dominant vertices): head sphere, neck, shoulder bar, chest and mid-back bars (no arms).
  7. Motion: head angular speed per grid frame (Kabsch of the posed head vertices) -> the fastest head moments
     (review screenshots) and the turn / calm split.

Coordinates in the json: "rest" = her rest model, Y-up right-handed, metres, sole at y = 0; "unity" vectors =
(x, y, -z) relative to the head joint (the skin's head bone has an identity rest rotation, so head-bone space =
rest axes). Likeness data (her colours, shape): work/ and git-ignored StreamingAssets only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .take import WORKSPACE, Take

ROLE, PID = "follow", 2
J_PELVIS, J_LHIP, J_RHIP, J_SPINE2, J_SPINE3, J_NECK, J_HEAD = 0, 1, 2, 6, 9, 12, 15
J_LCOLLAR, J_RCOLLAR, J_LSH, J_RSH, J_LELB, J_RELB, J_JAW = 13, 14, 16, 17, 18, 19, 22
HEAD_JOINTS = (15, 23, 24)  # head + eyes (the jaw region is the chin, never scalp)
UNITY_STREAMING = WORKSPACE / "head-movement" / "Assets" / "StreamingAssets"
VERSION = 1

# hairline: minimum elevation (deg, from the head-sphere centre) of scalp vertices vs |azimuth| (0 = facing)
HAIRLINE_EL = ((0, 50), (30, 45), (55, 34), (75, 24), (100, 20), (120, 2), (150, -26), (180, -34))
# original head-movement tip bloom (docs/HAIR_REFERENCE.md section 5): linear white emission along the strand
TIP_EMISSION = {"s": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "e": [0.0, 0.012, 0.024, 0.219, 1.398, 3.10],
                "source": "LineRenderer.BakeMesh vertex colours x 3.1 of the original HairSimulation (HAIR_REFERENCE.md 5)"}


# ------------------------------------------------------------------------------------------------ small helpers

def srgb_to_linear(x):
    x = np.asarray(x, float)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, float), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def luminance(rgb):
    return np.asarray(rgb, float) @ np.array([0.2126, 0.7152, 0.0722])


def colour(rgb_lin) -> dict:
    from .hair_reference import colour_entry
    return colour_entry(rgb_lin)


def r3(v, n: int = 4) -> list:
    return [round(float(x), n) for x in np.asarray(v).reshape(-1)]


def to_unity(v) -> np.ndarray:
    v = np.array(v, float)
    v[..., 2] *= -1
    return v


def hairline_el(abs_az_deg) -> np.ndarray:
    a, e = np.array(HAIRLINE_EL, float).T
    return np.interp(np.abs(abs_az_deg), a, e)


def fit_sphere(P: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares sphere |p - c| = r through points (N,3)."""
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3]
    return c, float(np.sqrt(max(x[3] + c @ c, 1e-12)))


def az_el(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth (deg, 0 = +z facing, +90 = +x = her left) and elevation (deg) of directions (N,3)."""
    n = np.linalg.norm(d, axis=-1)
    return np.degrees(np.arctan2(d[..., 0], d[..., 2])), np.degrees(np.arcsin(np.clip(d[..., 1] / np.maximum(n, 1e-9), -1, 1)))


def farthest_points(P: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Indices of k farthest-point samples of P (N,3) (deterministic start: the point nearest the centroid + seed)."""
    rng = np.random.default_rng(seed)
    k = min(k, len(P))
    first = int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1))) if seed == 0 else int(rng.integers(len(P)))
    idx = [first]
    d = np.linalg.norm(P - P[first], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.array(idx)


def strand_length(y_root, el_deg, y_tip, r_head) -> np.ndarray:
    """Over the skull to the equator (arc r el, drop r sin el), then straight down to the tip level."""
    el = np.radians(np.maximum(np.asarray(el_deg, float), 0.0))
    return np.maximum(np.asarray(y_root, float) - np.asarray(y_tip, float), 0.05) + r_head * (el - np.sin(el))


def tip_level(abs_az_deg, y_front: float, y_back: float, front_deg: float = 50.0, back_deg: float = 140.0):
    """Tip height per root azimuth: front level up to front_deg, back level beyond back_deg, smoothstep between."""
    x = np.clip((np.abs(abs_az_deg) - front_deg) / (back_deg - front_deg), 0, 1)
    s = x * x * (3 - 2 * x)
    return y_front + (y_back - y_front) * s


def kabsch_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Rotation angle (deg) of the best rigid fit A -> B (rows are points)."""
    a, b = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def segment_distance(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


# ------------------------------------------------------------------------------------------------ rest model

class RestModel:
    """Her shaped SMPL-X rest (T) pose, sole at y = 0."""

    def __init__(self, work: Path):
        from .avatar_texture import load_smplx, model_dir_for
        from .avatar_uv import model_npz, vertex_normals
        d = load_smplx(work, ROLE)
        self.model_path = model_npz(model_dir_for(d))
        m = np.load(self.model_path, allow_pickle=True)
        betas = np.asarray(d["betas"], float).reshape(-1)
        nb = int(d.get("num_betas", len(betas)))
        v = np.asarray(m["v_template"], float) + np.asarray(m["shapedirs"], float)[:, :, :nb] @ betas[:nb]
        J = np.asarray(m["J_regressor"], float) @ v
        y0 = v[:, 1].min()
        v[:, 1] -= y0
        J[:, 1] -= y0
        self.v, self.J, self.y0 = v, J[:55], float(y0)
        self.f = np.asarray(m["f"], np.int64)
        self.w = np.asarray(m["weights"], float)
        self.dom = self.w.argmax(1)
        self.n = vertex_normals(v, self.f)
        self.height = float(v[:, 1].max())
        self.model = str(d.get("model", ""))
        self.betas = betas[:nb]
        self.d = d


def scalp(rm: RestModel) -> dict:
    """Head sphere + scalp vertices / faces of the rest model."""
    head = np.isin(rm.dom, HEAD_JOINTS)
    cran = head & (rm.v[:, 1] > rm.J[J_HEAD, 1] + 0.02)
    c, r = fit_sphere(rm.v[cran])
    az, el = az_el(rm.v - c)
    sc = head & (el >= hairline_el(az))
    faces = np.nonzero(sc[rm.f].all(1))[0]
    return {"centre": c, "radius": r, "head_v": head, "scalp_v": sc, "faces": faces, "az": az, "el": el,
            "crown_y": float(rm.v[head, 1].max())}


def sample_roots(rm: RestModel, sp: dict, n_roots: int, seed: int = 0, oversample: int = 60) -> dict:
    rng = np.random.default_rng(seed)
    F = rm.f[sp["faces"]]
    A = 0.5 * np.linalg.norm(np.cross(rm.v[F[:, 1]] - rm.v[F[:, 0]], rm.v[F[:, 2]] - rm.v[F[:, 0]]), axis=1)
    n_cand = max(n_roots * oversample, 2000)
    fi = rng.choice(len(F), n_cand, p=A / A.sum())
    u, w = rng.random(n_cand), rng.random(n_cand)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    bary = np.c_[1 - u - w, u, w]
    P = (rm.v[F[fi]] * bary[:, :, None]).sum(1)
    keep = farthest_points(P, n_roots)
    fi, bary, P = fi[keep], bary[keep], P[keep]
    # order: back to front around the head, crown first within a ring (stable, readable ids)
    az, el = az_el(P - sp["centre"])
    order = np.lexsort((-el, -np.abs(az)))
    fi, bary, P, az, el = fi[order], bary[order], P[order], az[order], el[order]
    N = (rm.n[F[fi]] * bary[:, :, None]).sum(1)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return {"face": sp["faces"][fi], "vid": F[fi], "bary": bary, "pos": P, "normal": N, "az": az, "el": el,
            "scalp_area_m2": float(A.sum())}


def colliders(rm: RestModel, sp: dict) -> list[dict]:
    """Capsules (bone a + offset a -> bone b + offset b, radius) from the rest mesh; offsets in rest axes."""
    v, J, dom = rm.v, rm.J, rm.dom
    half_sh = 0.5 * abs(J[J_LSH, 0] - J[J_RSH, 0])

    def radius(sel, a, b, q=40):
        d = segment_distance(v[sel], a, b)
        return float(np.percentile(d, q)) if len(d) else 0.05

    out = [{"name": "head", "a": J_HEAD, "oa": sp["centre"] - J[J_HEAD], "b": J_HEAD, "ob": sp["centre"] - J[J_HEAD],
            "r": float(np.percentile(np.linalg.norm(v[sp["scalp_v"]] - sp["centre"], axis=1), 40))}]
    neck_sel = dom == J_NECK
    out.append({"name": "neck", "a": J_NECK, "oa": np.zeros(3), "b": J_HEAD, "ob": np.zeros(3),
                "r": radius(neck_sel, J[J_NECK], J[J_HEAD])})
    for name, j, sel_j, frac in (("shoulders", J_SPINE3, (J_SPINE3, J_LCOLLAR, J_RCOLLAR), 0.75),
                                 ("chest", J_SPINE3, (J_SPINE3,), 0.55), ("midback", J_SPINE2, (J_SPINE2,), 0.5)):
        dy = (J[J_LSH, 1] - J[j, 1]) if name == "shoulders" else 0.0
        oa, ob = np.array([frac * half_sh, dy, 0.0]), np.array([-frac * half_sh, dy, 0.0])
        a, b = J[j] + oa, J[j] + ob
        sel = np.isin(dom, sel_j) & (np.abs(v[:, 1] - a[1]) < 0.06)
        out.append({"name": name, "a": j, "oa": oa, "b": j, "ob": ob, "r": radius(sel, a, b, 35)})
    # no arm capsules: a raised arm next to the head traps strands between two capsules (jitter); hair passing
    # through an arm for a frame is the lesser artefact
    for c in out:
        c["r"] = round(float(np.clip(c["r"], 0.03, 0.16)), 4)
        c["oa_unity"], c["ob_unity"] = r3(to_unity(c.pop("oa"))), r3(to_unity(c.pop("ob")))
    return out


# ------------------------------------------------------------------------------------------------ 2D measurement

def head_speed(body, head_v: np.ndarray) -> np.ndarray:
    """|angular velocity| of her head (deg/s) per grid frame (central, Kabsch of the posed head vertices)."""
    V = body.verts[:, head_v]
    T = len(body.t)
    w = np.zeros(T)
    for k in range(T):
        a, b = max(k - 1, 0), min(k + 1, T - 1)
        if b > a:
            w[k] = kabsch_angle_deg(V[a], V[b]) / (body.t[b] - body.t[a])
    return w


def peaks(t: np.ndarray, w: np.ndarray, n: int = 3, sep_s: float = 0.5) -> list[dict]:
    out = []
    for k in np.argsort(-w):
        if all(abs(t[k] - p["t"]) >= sep_s for p in out):
            out.append({"t": round(float(t[k]), 4), "k": int(k), "deg_s": round(float(w[k]), 1)})
        if len(out) >= n:
            break
    return out


def measure_views(take: Take, rm: RestModel, sp: dict, step: int, cams: tuple = (), calm_dps: float = 90.0,
                  turn_dps: float = 180.0) -> dict:
    from .avatar_texture import load_bodies
    from .face_common import project
    from .mamma_dyn import load_tracks
    work = take.work
    tracks = load_tracks(work)
    if cams:
        tracks = {c: v for c, v in tracks.items() if c in cams}
    bodies, binfo = load_bodies(work, "cpu")
    body = bodies[ROLE]
    hv = sp["head_v"]
    w = head_speed(body, hv)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    views, samples = [], []
    for cam, tr in sorted(tracks.items()):
        dd = work / "dancers" / cam
        for i in range(0, len(tr.t), step):
            k = int(np.argmin(np.abs(body.t - tr.t[i])))
            if abs(body.t[k] - tr.t[i]) > 0.02 or not bool(tr.ok[i]):
                continue
            hair = cv2.imread(str(dd / "hair" / f"mask_{i:04d}_{PID:02d}.png"), cv2.IMREAD_GRAYSCALE)
            bodym = cv2.imread(str(dd / "body" / f"mask_{i:04d}_{PID:02d}.png"), cv2.IMREAD_GRAYSCALE)
            if hair is None or bodym is None:
                continue
            hair, bodym = hair > 127, bodym > 127
            if hair.sum() < 150:
                continue
            K, R, tv, k1 = tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i])
            Vh = body.verts[k, hv].astype(float)
            px, z = project(Vh, K, R, tv, k1)
            if (z <= 0.3).any():
                continue
            H, W = hair.shape
            hull = cv2.convexHull(px.astype(np.float32))
            r_px = float(np.sqrt(max(cv2.contourArea(hull), 1.0) / np.pi))
            hc = px.mean(0)
            if not (0 <= hc[0] < W and 0 <= hc[1] < H):
                continue
            mpp = float(np.median(z)) / float(K[0, 0])
            c3 = Vh.mean(0)
            pd, _ = project(np.array([c3, c3 - [0, 0.5, 0]]), K, R, tv, k1)
            down = pd[1] - pd[0]
            down /= max(np.linalg.norm(down), 1e-9)
            ys, xs = np.nonzero(hair)
            rel = np.c_[xs - hc[0], ys - hc[1]]
            along = (rel @ down) * mpp
            across = (rel @ np.array([-down[1], down[0]])) * mpp
            dist = np.hypot(rel[:, 0], rel[:, 1])
            band = (along > 0.20) & (along < 0.30)
            width = float(np.percentile(across[band], 97) - np.percentile(across[band], 3)) if band.sum() > 30 else None
            v = {"cam": cam, "i": i, "t": float(tr.t[i]), "k": k, "head_dps": round(float(w[k]), 1),
                 "r_head_px": round(r_px, 1), "m_per_px": mpp, "r_head_m": round(r_px * mpp, 4),
                 "drop_m": round(float(np.percentile(along, 98)), 4),
                 "reach_m": round(float(np.percentile(dist, 98) * mpp), 4),
                 "width_0p25_m": None if width is None else round(width, 4),
                 "area_head_areas": round(float(hair.sum() / (np.pi * r_px ** 2)), 3), "n_hair_px": int(hair.sum())}
            views.append(v)
            # colour sample
            img = cv2.imread(tr.files[i])
            if img is None or img.shape[:2] != hair.shape:
                continue
            Jp, _ = project(body.joints[k, :22], K, R, tv, k1)
            q = np.array([Jp[J_LSH], Jp[J_RSH], Jp[J_RHIP], Jp[J_LHIP]], float)
            q = (q.mean(0) + 0.8 * (q - q.mean(0))).astype(np.int32)
            tq = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(tq, q, 1)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            top = tq.astype(bool) & cv2.erode(bodym.astype(np.uint8), k3, iterations=2).astype(bool) & \
                (hsv[..., 1] < 60) & (img.max(2) < 250) & ~cv2.dilate(hair.astype(np.uint8), k3, iterations=3).astype(bool)
            if top.sum() < 60:
                continue
            rgb = srgb_to_linear(img[:, :, ::-1] / 255.0)
            tvv = hsv[..., 2][top]
            sel = top & (hsv[..., 2] >= np.percentile(tvv, 40)) & (hsv[..., 2] <= np.percentile(tvv, 95))
            white = np.median(rgb[sel], 0)
            if luminance(white) < 0.05:
                continue
            he = cv2.erode(hair.astype(np.uint8), k3, iterations=2).astype(bool)
            if he.sum() < 50:
                continue
            ys, xs = np.nonzero(he)
            d = np.hypot(xs - hc[0], ys - hc[1])
            lo, hi = r_px * 0.6, max(np.percentile(d, 98), r_px * 0.6 + 1)
            s = np.clip((d - lo) / (hi - lo), 0, 1)
            wb = rgb[ys, xs] * (0.80 / np.maximum(white, 1e-4))
            samples.append({"cam": cam, "s": s.astype(np.float32), "rgb": wb.astype(np.float32)})
    return {"views": views, "samples": samples, "head_dps": w, "t": body.t, "bodies": binfo,
            "calm_dps": calm_dps, "turn_dps": turn_dps}


def palette(samples: list[dict], rng_seed: int = 0) -> dict:
    """Per-camera zone medians -> median across cameras; shadow / highlight bands; strand palette."""
    per_cam: dict[str, dict] = {}
    for s in samples:
        pc = per_cam.setdefault(s["cam"], {"root": [], "mid": [], "tip": [], "all": []})
        pc["root"].append(s["rgb"][s["s"] < 0.25])
        pc["mid"].append(s["rgb"][(s["s"] >= 0.25) & (s["s"] < 0.75)])
        pc["tip"].append(s["rgb"][s["s"] >= 0.75])
        pc["all"].append(s["rgb"])
    zones = {z: [] for z in ("root", "mid", "tip", "shadow", "highlight")}
    pooled = []
    rng = np.random.default_rng(rng_seed)
    for cam, pc in sorted(per_cam.items()):
        for z in ("root", "mid", "tip"):
            a = np.concatenate(pc[z]) if pc[z] else np.zeros((0, 3))
            if len(a) >= 30:
                zones[z].append(np.median(a, 0))
        a = np.concatenate(pc["all"])
        if len(a) > 100:
            Y = luminance(a)
            lo, hi = np.percentile(Y, [5, 15]), np.percentile(Y, [90, 99])
            zones["shadow"].append(np.median(a[(Y >= lo[0]) & (Y <= lo[1])], 0))
            zones["highlight"].append(np.median(a[(Y >= hi[0]) & (Y <= hi[1])], 0))
            # equal weight per camera in the pooled strand palette
            pooled.append(a[rng.choice(len(a), min(len(a), 4000), replace=False)])
    out = {}
    for z, vals in zones.items():
        if vals:
            A = np.array(vals)
            out[z] = {**colour(np.median(A, 0)), "n_cams": len(A)}
    strand = []
    if pooled:
        P = np.concatenate(pooled)
        Y = luminance(P)
        qs = np.percentile(Y, np.linspace(10, 90, 9))
        for a, b in zip(qs[:-1], qs[1:]):
            sel = (Y >= a) & (Y <= b)
            if sel.sum():
                strand.append(r3(np.median(P[sel], 0)))
    return {"zones": out, "strand_palette_linear": strand, "n_cams": len(per_cam), "n_views": len(samples)}


# ------------------------------------------------------------------------------------------------ groom

def build(take: Take, guides: int = 160, segments: int = 16, step: int = 6, cams: tuple = (), seed: int = 0) -> dict:
    t0 = time.time()
    work = take.work
    rm = RestModel(work)
    sp = scalp(rm)
    roots = sample_roots(rm, sp, guides, seed)
    meas = measure_views(take, rm, sp, step, cams)
    views = meas["views"]
    calm = [v for v in views if v["head_dps"] < meas["calm_dps"]]
    turn = [v for v in views if v["head_dps"] > meas["turn_dps"]]
    c, r_head = sp["centre"], sp["radius"]
    crown = sp["crown_y"]

    def med(vs, key, q=50):
        a = [v[key] for v in vs if v[key] is not None]
        return round(float(np.percentile(a, q)), 4) if a else None
    m2d = {"views": len(views), "calm_views": len(calm), "turn_views": len(turn),
           "cams": sorted({v["cam"] for v in views}),
           "head_radius_projected_m_median": med(views, "r_head_m"),
           "drop_below_head_centre_m": {"calm_median": med(calm, "drop_m"), "calm_p75": med(calm, "drop_m", 75),
                                        "turn_median": med(turn, "drop_m")},
           "reach_from_head_centre_m": {"calm_median": med(calm, "reach_m"), "turn_median": med(turn, "reach_m"),
                                        "turn_p90": med(turn, "reach_m", 90)},
           "width_0p25_below_head_m": {"calm_median": med(calm, "width_0p25_m"), "turn_median": med(turn, "width_0p25_m")},
           "area_in_head_areas": {"calm_median": med(calm, "area_head_areas"), "turn_median": med(turn, "area_head_areas")},
           "note": "2D, relative to the projected SMPL-X head: drops / widths are projections (foreshortening only "
                   "shortens), so the calm p75 is the 2D length estimate"}
    # tip levels: 3D reference when present, else the 2D calm drop
    ref_p = work / "avatar" / "hair_reference.json"
    ref = json.loads(ref_p.read_text(encoding="utf-8")) if ref_p.exists() else None
    shoulder_y = float(rm.J[[J_LSH, J_RSH], 1].mean())
    if ref is not None and abs(ref["scale"]["rest_height_world_m"] - rm.height) < 0.01:
        L = ref["shape"]["length"]
        y_back = float(L["tip_level_back_rest_m"]["median"])
        y_front = float(L["tip_level_front_rest_m"]["median"])
        crown_len = float(L["strand_length_crown_to_tip_back"]["world_m"])
        length_source = "hair_reference.json (3D hair hull + ray-cast, calm frames)"
    else:
        drop = m2d["drop_below_head_centre_m"]["calm_p75"] or 0.45
        y_back = float(c[1] - drop)
        y_front = shoulder_y - 0.06
        crown_len = None
        length_source = "2D masks (calm p75 drop below the projected head centre)" + \
            ("" if ref is None else "; hair_reference.json is for another rest height")
    y_tip = tip_level(roots["az"], y_front, y_back)
    el_root = az_el(roots["pos"] - c)[1]
    raw = strand_length(roots["pos"][:, 1], el_root, y_tip, r_head)
    crown_raw = float(strand_length(crown, 90.0, y_back, r_head))
    k_len = crown_len / crown_raw if crown_len else 1.0
    rng = np.random.default_rng(seed + 1)
    lengths = raw * k_len * (1 + 0.05 * rng.uniform(-1, 1, len(raw)))
    # 2D cross-check of the chosen back level: head-centre drop implied vs measured (calm p75)
    implied_drop = float(c[1] - y_back)
    pal = palette(meas["samples"])
    ref_cmp = None
    if ref is not None:
        from .hair_reference import delta_e76
        ro = ref["colour"]["video"]["overall"]
        ref_cmp = {z: round(delta_e76(pal["zones"][z]["linear_srgb"], ro[z]["linear_srgb"]), 1)
                   for z in ("root", "mid", "tip", "shadow", "highlight") if z in pal["zones"] and z in ro}
    head_t, head_w = meas["t"], meas["head_dps"]
    mot_ref = (ref or {}).get("shape", {}).get("motion", {})
    standoff = 0.03
    if ref is not None:
        so = [x for x in ref["shape"]["volume"]["back_layer_standoff_world_m"] if x is not None]
        standoff = round(float(np.clip(0.6 * np.median(so), 0.015, 0.045)), 4) if so else standoff  # hull = upper bound
    Jh = rm.J[J_HEAD]
    guides_out = [{"vid": [int(x) for x in roots["vid"][g]], "bary": r3(roots["bary"][g], 5),
                   "len": round(float(lengths[g]), 4),
                   "pos_unity": r3(to_unity(roots["pos"][g] - Jh)), "normal_unity": r3(to_unity(roots["normal"][g])),
                   "az": round(float(roots["az"][g]), 1), "el": round(float(roots["el"][g]), 1)}
                  for g in range(len(lengths))]
    zones = pal["zones"]
    groom = {
        "version": VERSION, "take": take.name, "role": ROLE, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "likeness": "her hair colour and shape: work/ and git-ignored StreamingAssets only",
        "model": {"path": str(rm.model_path.relative_to(WORKSPACE)) if rm.model_path.is_relative_to(WORKSPACE)
                  else str(rm.model_path), "fit_model": rm.model, "num_betas": int(len(rm.betas)),
                  "rest_height_m": round(rm.height, 4), "head_joint_rest": r3(rm.J[J_HEAD]),
                  "head_sphere_centre_unity": r3(to_unity(c - Jh)), "head_radius_m": round(r_head, 4),
                  "crown_rest_y": round(crown, 4), "scalp_area_m2": round(roots["scalp_area_m2"], 4),
                  "scalp_vertices": int(sp["scalp_v"].sum()), "hairline_el_deg_vs_abs_az": [list(x) for x in HAIRLINE_EL]},
        "strands": {"guides": len(guides_out), "segments": segments, "children": 3,
                    "child_spread_m": round(float(np.sqrt(roots["scalp_area_m2"] / max(len(guides_out), 1)) * 0.55), 4),
                    "tip_clump": 0.6, "width_root_m": 0.028, "width_tip_m": 0.008,
                    "note": "guides are simulated; each renders as 1 + children view-facing ribbons (child offsets in "
                            "the guide frame, clumping toward the tips = piecey ends)"},
        "guides": guides_out,
        "shape": {"length_source": length_source, "tip_level_back_rest_m": round(y_back, 4),
                  "tip_level_front_rest_m": round(y_front, 4),
                  "crown_strand_m": round(float(crown_raw * k_len), 4), "length_scale": round(float(k_len), 4),
                  "length_m": {"min": round(float(lengths.min()), 4), "median": round(float(np.median(lengths)), 4),
                               "max": round(float(lengths.max()), 4)},
                  "standoff_m": standoff, "part_x_unity": 0.025, "part_side": "her left (+x rest = +x unity)",
                  "implied_drop_below_head_centre_m": round(implied_drop, 4),
                  "measured_2d": m2d, "cut": "layered: front sections to the collarbone, back to between the shoulder "
                                             "blades and the mid-back, +-5 % jitter"},
        "colour": {"source": "dancers hair masks (eroded 2 px) white-balanced on her top (albedo 0.80), median of "
                             "per-camera zone medians", **{z: zones[z] for z in zones},
                   "strand_palette_linear": pal["strand_palette_linear"], "n_cams": pal["n_cams"],
                   "n_views": pal["n_views"], "vs_hair_reference_dE76": ref_cmp},
        "render": {"tip_emission": TIP_EMISSION, "kajiya_kay": {"primary_shift": -0.06, "secondary_shift": 0.08,
                                                               "primary_exponent": 90, "secondary_exponent": 18,
                                                               "specular": 0.25}},
        "colliders": colliders(rm, sp),
        "dynamics": {"gravity_mps2": 9.81, "substeps": 8, "iterations": 2, "damping_per_s": 1.6,
                     "bend_compliance": 2e-5, "shape_stiffness_root": 0.5, "shape_falloff": 0.18, "friction": 0.3, "particle_radius_m": 0.008,
                     "targets": {"turn_spread_r95_m": (mot_ref.get("spread_radius_r95_world_m") or {}).get("turn_median"),
                                 "trailing_angle_deg": (mot_ref.get("trailing_angle_deg_turns") or {}).get("median"),
                                 "lag_s": (mot_ref.get("lag_time_s") or {}).get("phase_alignment")}},
        "motion": {"head_speed_dps": {"p50": round(float(np.median(head_w)), 1), "p90": round(float(np.percentile(head_w, 90)), 1),
                                      "max": round(float(head_w.max()), 1)},
                   "fastest_head_moments_reference_t": peaks(head_t, head_w, 3), "grid_t0": round(float(head_t[0]), 4),
                   "grid_frames": int(len(head_t))},
        "runtime_s": None,
    }
    groom["runtime_s"] = round(time.time() - t0, 1)
    return groom, {"rm": rm, "sp": sp, "roots": roots, "meas": meas, "lengths": lengths}


# ------------------------------------------------------------------------------------------------ review

def review(groom: dict, extra: dict, path: Path, take: Take) -> None:
    """Scalp + roots + combed-down strands on the rest head (front / side / back), colour swatches, 2D measurement
    overlays on calm / turn frames."""
    rm, sp, roots, lengths = extra["rm"], extra["sp"], extra["roots"], extra["lengths"]
    c = sp["centre"]
    S = 360
    panels = []
    for name in ("front", "side (her left)", "back"):
        img = np.full((S + 140, S, 3), 24, np.uint8)
        sel = rm.v[:, 1] > rm.J[J_NECK, 1] - 0.45
        scale = S / 0.75

        def P(p, name=name):
            # image x of an orthographic view: front = her left on the right, back = mirrored, side from her left
            x = {"front": p[..., 0] - c[0], "back": -(p[..., 0] - c[0]), "side (her left)": -(p[..., 2] - c[2])}[name]
            return np.stack([S / 2 + x * scale, 70 + (c[1] + 0.12 - p[..., 1]) * scale], -1).astype(int)
        vis = sel & ((rm.n[:, 2] > 0) if name == "front" else (rm.n[:, 2] < 0) if name == "back" else (rm.n[:, 0] > 0))
        q = P(rm.v[vis])
        col = np.where(sp["scalp_v"][vis, None], [60, 60, 140], [90, 90, 90])
        for (x, y), cc in zip(q, col):
            if 0 <= x < S and 0 <= y < S + 140:
                img[y, x] = cc
        # combed-down preview strands: root -> outward 2 cm -> straight down to the tip level of its length
        facing = {"front": roots["normal"][:, 2] > -0.2, "back": roots["normal"][:, 2] < 0.2,
                  "side (her left)": roots["normal"][:, 0] > -0.2}[name]
        for g in np.nonzero(facing)[0]:
            p0 = roots["pos"][g]
            p1 = p0 + roots["normal"][g] * 0.02
            rest_len = lengths[g] - 0.02
            p2 = p1 + np.array([0, -rest_len, 0])
            a, b, d = P(p0), P(p1), P(p2)
            cv2.line(img, tuple(a), tuple(b), (40, 70, 200), 1, cv2.LINE_AA)
            cv2.line(img, tuple(b), tuple(d), (50, 80, 190), 1, cv2.LINE_AA)
            cv2.circle(img, tuple(a), 2, (80, 220, 255), -1)
        for lvl, txt in ((groom["shape"]["tip_level_back_rest_m"], "back tips"),
                         (groom["shape"]["tip_level_front_rest_m"], "front tips")):
            y = int(70 + (c[1] + 0.12 - lvl) * scale)
            cv2.line(img, (0, y), (S, y), (0, 200, 0) if txt == "back tips" else (0, 160, 200), 1)
            cv2.putText(img, f"{txt} {lvl:.2f} m", (4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
        cv2.putText(img, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        panels.append(img)
    row1 = np.hstack(panels)
    # colour swatches
    sw = np.full((140, row1.shape[1], 3), 24, np.uint8)
    x = 8
    for z in ("shadow", "root", "mid", "tip", "highlight"):
        e = groom["colour"].get(z)
        if e:
            cv2.rectangle(sw, (x, 10), (x + 110, 80), tuple(int(v) for v in e["srgb8"][::-1]), -1)
            cv2.putText(sw, f"{z} {e['hex']}", (x, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
            x += 125
    for j, cl in enumerate(groom["colour"]["strand_palette_linear"]):
        s8 = np.round(linear_to_srgb(cl) * 255).astype(int)
        cv2.rectangle(sw, (x + j * 22, 10), (x + j * 22 + 20, 80), tuple(int(v) for v in s8[::-1]), -1)
    cv2.putText(sw, "strand palette", (x, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
    dE = groom["colour"].get("vs_hair_reference_dE76")
    sh = groom["shape"]
    txt = [f"{groom['strands']['guides']} guides x {groom['strands']['segments']} seg, lengths {sh['length_m']['min']:.2f}-"
           f"{sh['length_m']['max']:.2f} m (crown {sh['crown_strand_m']:.2f}), standoff {sh['standoff_m'] * 100:.1f} cm; "
           f"length: {sh['length_source']}",
           f"2D calm drop below head centre p75 {sh['measured_2d']['drop_below_head_centre_m']['calm_p75']} m vs groom "
           f"{sh['implied_drop_below_head_centre_m']} m; colour vs hair_reference dE76 {dE}"]
    for j, t_ in enumerate(txt):
        cv2.putText(sw, t_, (8, 116 + 16 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    # measurement overlays: one calm and one turn view per a few cameras
    meas = extra["meas"]
    from .face_common import project
    from .mamma_dyn import load_tracks
    tracks = load_tracks(take.work)
    tiles = []
    vs = sorted(meas["views"], key=lambda v: v["head_dps"])
    pick = [vs[len(vs) // 4], vs[len(vs) // 2]] + vs[-2:] if len(vs) >= 4 else vs
    body = None
    for v in pick:
        tr = tracks[v["cam"]]
        img = cv2.imread(tr.files[v["i"]])
        hair = cv2.imread(str(take.work / "dancers" / v["cam"] / "hair" / f"mask_{v['i']:04d}_{PID:02d}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if img is None or hair is None:
            continue
        if body is None:
            from .avatar_texture import load_bodies
            body = load_bodies(take.work, "cpu")[0][ROLE]
        Vh = body.verts[v["k"], sp["head_v"]].astype(float)
        px, _ = project(Vh, tr.K[v["i"]], tr.R[v["i"]], tr.tvec[v["i"]], float(tr.k1[v["i"]]))
        hc = px.mean(0)
        over = img.copy()
        over[hair > 127] = (0.5 * over[hair > 127] + [0, 0, 127]).astype(np.uint8)
        cv2.circle(over, tuple(int(a) for a in hc), int(v["r_head_px"]), (255, 255, 0), 1, cv2.LINE_AA)
        r_drop = v["drop_m"] / v["m_per_px"]
        cv2.circle(over, tuple(int(a) for a in hc), int(r_drop), (0, 255, 0), 1, cv2.LINE_AA)
        side = int(max(r_drop * 1.25, 60))
        x0, y0 = int(max(hc[0] - side, 0)), int(max(hc[1] - side * 0.6, 0))
        crop = over[y0:int(min(hc[1] + side * 1.4, img.shape[0])), x0:int(min(hc[0] + side, img.shape[1]))]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (int(crop.shape[1] * 300 / crop.shape[0]), 300))
        cv2.putText(crop, f"cam {v['cam']} f{v['i']} {v['head_dps']:.0f} deg/s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        cv2.putText(crop, f"drop {v['drop_m']:.2f} m reach {v['reach_m']:.2f} m", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0), 1)
        tiles.append(crop)
    rows = [row1, sw]
    if tiles:
        strip = np.hstack(tiles)
        Wt = row1.shape[1]
        strip = cv2.resize(strip, (Wt, int(strip.shape[0] * Wt / strip.shape[1]))) if strip.shape[1] > Wt else \
            np.hstack([strip, np.full((strip.shape[0], Wt - strip.shape[1], 3), 24, np.uint8)])
        cv2.putText(strip, "hair mask (red), projected SMPL-X head (cyan), p98 drop (green)", (4, strip.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        rows.append(strip)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


# ------------------------------------------------------------------------------------------------ export

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


def export(take: Take, capture: str | None = None, streaming: Path = UNITY_STREAMING) -> list[Path]:
    """Copy avatar/hair_groom.json into StreamingAssets/<capture>/ (and work/<take>/export/<capture>/ when it
    exists). Re-run after every `python -m dancecap export` (that rewrites the capture folder)."""
    src = take.work / "avatar" / "hair_groom.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing (run python -m dancecap.hair_groom {take.name})")
    dirs = [Path(streaming) / capture] if capture else captures_for(take.name, streaming)
    out = []
    for d in dirs:
        if not (d / "capture.json").exists():
            raise FileNotFoundError(f"{d}/capture.json missing (export the capture first)")
        for dst in (d, take.work / "export" / d.name):
            if dst.exists():
                shutil.copy2(src, dst / "hair_groom.json")
                out.append(dst / "hair_groom.json")
    return out


def sheet(take: Take, out: Path | None = None, tile_w: int = 640) -> Path:
    """Labelled contact sheet of the Unity hair review shots (Tools/playtest_hair.ps1 writes review/hair_peak*.png and
    review/hair_shots.json): one row per fastest head moment x tip bloom on / off, one column per view."""
    rdir = take.work / "review"
    meta = json.loads((rdir / "hair_shots.json").read_text(encoding="utf-8-sig"))
    shots = meta["shots"]
    rows: dict[tuple, list] = {}
    for s in shots:
        rows.setdefault((s["peak"], s["tipGlow"]), []).append(s)
    sw = meta.get("sweep") or {}
    lines = [f"{meta.get('capture')} - follow hair (Unity, Assets/Hair): groomed guide strands, Kajiya-Kay ribbons",
             f"sweep {sw.get('frames')} frames: spread r95 calm {sw.get('spreadR95CalmMedianM', float('nan')):.3f} m, "
             f"turn {sw.get('spreadR95TurnMedianM', float('nan')):.3f} m (p90 {sw.get('spreadR95TurnP90M', float('nan')):.3f}), "
             f"trailing {sw.get('trailingDegTurnMedian', float('nan')):.0f} deg; CPU {sw.get('stepMsMean', float('nan')):.2f} ms "
             f"per 30 Hz step (editor), stretch {sw.get('maxStretch', float('nan')):.3f}, resets {sw.get('strandResets')}"]
    blocks = []
    head = np.full((70, tile_w * 3, 3), 20, np.uint8)
    for j, t in enumerate(lines):
        cv2.putText(head, t, (10, 26 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    blocks.append(head)
    for key in sorted(rows, key=lambda k: (k[0], -k[1])):
        tiles = []
        for s in sorted(rows[key], key=lambda s: s["azimuth"], reverse=True):
            img = cv2.imread(str(rdir / s["file"]))
            if img is None:
                continue
            img = cv2.resize(img, (tile_w, int(img.shape[0] * tile_w / img.shape[1])))
            lab = (f"peak {s['peak']}: f{s['frame']} {s['audioTime']:.2f}s {s['degPerS']:.0f} deg/s | "
                   f"{s['view']} az {s['azimuth']:.0f} | bloom {'on' if s['tipGlow'] else 'off'}")
            cv2.rectangle(img, (0, img.shape[0] - 30), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
            cv2.putText(img, lab, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        if tiles:
            while len(tiles) < 3:
                tiles.append(np.zeros_like(tiles[0]))
            blocks.append(np.hstack(tiles[:3]))
    out = out or rdir / "hair_sheet.jpg"
    cv2.imwrite(str(out), np.vstack(blocks), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def run(take: Take, guides: int, step: int, cams: tuple, do_export: bool = True) -> dict:
    groom, extra = build(take, guides=guides, step=step, cams=cams)
    out = take.work / "avatar" / "hair_groom.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(groom, indent=1), encoding="utf-8")
    review(groom, extra, take.work / "review" / "hair_groom.jpg", take)
    print(f"[hair_groom] {out}: {groom['strands']['guides']} guides, lengths {groom['shape']['length_m']}, "
          f"colour mid {groom['colour'].get('mid', {}).get('hex')}, dE vs reference {groom['colour']['vs_hair_reference_dE76']}")
    print(f"[hair_groom] fastest head moments: {groom['motion']['fastest_head_moments_reference_t']}")
    if do_export:
        for p in export(take):
            print(f"[hair_groom] -> {p}")
    return groom


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.hair_groom", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--guides", type=int, default=160, help="guide strands (Quest budget <= 200)")
    ap.add_argument("--step", type=int, default=6, help="every n-th camera frame for the 2D measurement")
    ap.add_argument("--cams", nargs="*", default=[])
    ap.add_argument("--export-only", action="store_true", help="only copy avatar/hair_groom.json into the captures")
    ap.add_argument("--capture", help="capture folder name (default: every capture of the take)")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--sheet", action="store_true", help="only compose review/hair_sheet.jpg from the Unity review shots")
    a = ap.parse_args(argv)
    take = Take.load(a.take)
    if a.sheet:
        print(f"[hair_groom] {sheet(take)}")
        return
    if a.export_only:
        for p in export(take, a.capture):
            print(f"[hair_groom] -> {p}")
        return
    if not 50 <= a.guides <= 200:
        raise SystemExit("--guides must be 50..200 (Quest budget: <= 200 guides x 16 segments)")
    run(take, a.guides, a.step, tuple(a.cams), do_export=not a.no_export)


if __name__ == "__main__":
    main()
