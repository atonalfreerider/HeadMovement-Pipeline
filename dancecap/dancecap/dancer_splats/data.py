"""Prep of the dancer-splat training bundle (local, CPU) and its loader (anywhere: Windows or a Linux pod).

work/<take>/dancer_splats/bundle/ (everything the pod needs; no SMPL-X model, no frame store):
  manifest.json   version, take, prep config, input mtimes (freshness), counts, bytes
  views.json      one entry per crop: images/<i>.jpg (undistorted crop), labels/<i>.png, cam, frame, t_frame, t (sync-
                  corrected time the bodies are posed at), K (crop, pinhole, pixel centres at integers), R, t_w (world
                  camera), W, H, crop box, split (train | test_cam | test_time), eval_train, review, swap_prob,
                  sharpness weight, camera type, inter-ocular distance per dancer, face_hd
  bodies.npz      window grid t_grid; per dancer the fit's posed vertices (MAMMA + adopted head fix + shipped face
                  expression), the global rotations Rg and posed positions Jp of the 55 joints, the shaped rest mesh;
                  shared faces / parents / top-4 LBS weights / vt / ft
  init_<role>.npz area-uniform surface samples (face, barycentrics), avatar albedo colour, region bits
  hair.npz        groom guides as rest polylines in the head frame, segment lengths, palette, colliders
  eval/           the textured avatar (albedo on the same posed SMPL-X) rendered from every held-out view
                  (avatar_<i>.jpg + role map avatar_<i>.png) and from a fixed novel camera (temporal check)
  novel.json      the novel camera (halfway between novel_pair) and the avatar's frame-to-frame change there
Labels: 0 background, 1 lead, 2 follow body, 3 follow hair, 4 both dancers' masks, 255 ignore (other people, phone
screens, outside the undistorted frame).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from . import (BUNDLE_VERSION, HEAD_JOINT, L_BG, L_BOTH, L_FOLLOW, L_HAIR, L_IGNORE, L_LEAD, PIDS, R_EYE, R_FACE,
               R_FOOT, R_HAND, R_HEAD, R_SKIN, ROLES, SplatConfig, prep_dict)
from .binding import rest_strand, triangle_area_samples

REVIEW_TIMES = (58.3, 59.18, 59.9, 60.4, 60.88, 61.41)  # incl. her two fastest head turns (59.18 s, 60.88 s)


# ------------------------------------------------------------------------------------------------ labels

def encode_labels(lead: np.ndarray, follow: np.ndarray, hair: np.ndarray | None, people: np.ndarray | None,
                  valid: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel training label from the boolean masks (all (H, W)): hair = the follow's hair mask (outside the lead);
    both dancers' masks overlapping -> L_BOTH (foreground, owner unknown); other people / phone screens and pixels
    outside the undistorted frame -> L_IGNORE."""
    lead, follow = lead.astype(bool), follow.astype(bool)
    L = np.full(lead.shape, L_BG, np.uint8)
    L[lead] = L_LEAD
    L[follow] = L_FOLLOW
    if hair is not None:
        L[hair.astype(bool) & ~lead] = L_HAIR
    L[lead & follow] = L_BOTH
    if people is not None:
        L[people.astype(bool) & ~(lead | follow)] = L_IGNORE
    if valid is not None:
        L[~valid.astype(bool)] = L_IGNORE
    return L


def others_ignore(people: np.ndarray | None, dancers: np.ndarray, margin_px: int = 10) -> np.ndarray | None:
    """Ignore pixels = other people / phone screens with their margin, NOT the ring the people mask's dilation
    (detect.py dilate_px = 10) puts around the dancers: erode the mask back, drop the dancers' own pixels and a
    margin_px band around them (the detector's person masks are looser than SAM's), dilate the rest again. Without
    this the silhouette loss could not see the dancers' outline."""
    if people is None:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
    core = cv2.erode(people.astype(np.uint8), k) > 0
    near = cv2.dilate(dancers.astype(np.uint8), k) > 0
    others = core & ~near
    return (cv2.dilate(others.astype(np.uint8), k) > 0) & ~dancers.astype(bool)


def decode_labels(L):
    """numpy or torch label image -> dict of boolean masks: fg, valid, lead, follow (body), hair, both."""
    return {"fg": (L >= L_LEAD) & (L <= L_BOTH), "valid": L != L_IGNORE, "lead": L == L_LEAD,
            "follow": L == L_FOLLOW, "hair": L == L_HAIR, "both": L == L_BOTH}


# ------------------------------------------------------------------------------------------------ cameras

def crop_K(K: np.ndarray, x0: float, y0: float) -> np.ndarray:
    K = np.asarray(K, float).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    return K


def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Image resized by 1/sx, 1/sy (pixel centres at integers)."""
    K = np.asarray(K, float).copy()
    K[0, 0] /= sx
    K[1, 1] /= sy
    K[0, 2] = (K[0, 2] + 0.5) / sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) / sy - 0.5
    return K


def project(P: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, k1: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    Xc = P @ R.T + t
    z = Xc[:, 2]
    x, y = Xc[:, 0] / np.maximum(z, 1e-6), Xc[:, 1] / np.maximum(z, 1e-6)
    d = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], 1), z


def undistort(img: np.ndarray, K: np.ndarray, k1: float, interp=cv2.INTER_LINEAR, border=0) -> np.ndarray:
    if abs(k1) < 1e-9:
        return img
    mx, my = cv2.initUndistortRectifyMap(K, np.array([k1, 0, 0, 0, 0.0]), None, K, (img.shape[1], img.shape[0]),
                                         cv2.CV_32FC1)
    return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def lerp_index(t_grid: np.ndarray, t: float) -> tuple[int, float]:
    """(k, a): t lies a of the way from grid frame k to k + 1 (clamped to the grid)."""
    k = int(np.clip(np.searchsorted(t_grid, t) - 1, 0, len(t_grid) - 2))
    a = float(np.clip((t - t_grid[k]) / max(t_grid[k + 1] - t_grid[k], 1e-9), 0.0, 1.0))
    return k, a


def at_time(arr: np.ndarray, t_grid: np.ndarray, t: float) -> np.ndarray:
    k, a = lerp_index(t_grid, t)
    return (1 - a) * arr[k] + a * arr[k + 1]


# ------------------------------------------------------------------------------------------------ bodies

def _parents(m) -> np.ndarray:
    p = np.asarray(m["kintree_table"][0]).astype(np.int64)
    p[0] = -1
    return p


def load_fits(take: str, work: Path) -> tuple[dict, dict]:
    """The take's body motion: [export] smplx_dir (else smplx/), the adopted head fix, the shipped face (smplx_face)."""
    from ..avatar_texture import model_dir_for
    from ..export_unity import resolve_smplx_dir
    sdir, src = resolve_smplx_dir(take)
    fits, info = {}, {"smplx_dir": sdir, "smplx_dir_source": src, "roles": {}}
    hf = work / "avatar" / "head_fix.npz"
    fix = dict(np.load(hf)) if hf.exists() else None
    for r in ROLES:
        p = work / sdir / f"{r}.npz"
        with np.load(p, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        ri = {"file": str(p.relative_to(work).as_posix()), "head_fix": False, "face": None}
        if fix is not None and bool(fix.get(f"{r}_adopted", False)) and len(fix[f"{r}_neck_pose"]) == len(d["t"]):
            bp = np.asarray(d["body_pose"], float).copy()
            bp[:, 33:36] = fix[f"{r}_neck_pose"]
            d["body_pose"] = bp
            ri["head_fix"] = True
        fp = work / "smplx_face" / f"{r}.npz"
        if fp.exists():
            with np.load(fp, allow_pickle=True) as z:
                fz = {k: z[k] for k in z.files}
            if len(fz["t"]) == len(d["t"]) and np.allclose(fz["t"], d["t"], atol=1e-4):
                d["expression"] = np.asarray(fz["expression"], float)
                d["jaw_pose"] = np.asarray(fz["jaw_pose"], float)
                ri["face"] = {"file": f"smplx_face/{r}.npz", "shipped": str(fz.get("shipped", "")),
                              "n_expr": int(np.asarray(fz["expression"]).shape[1])}
        d["_model_dir"] = model_dir_for(d)
        fits[r] = d
        info["roles"][r] = ri
    t0 = np.asarray(fits[ROLES[0]]["t"], float)
    for r in ROLES[1:]:
        if len(fits[r]["t"]) != len(t0) or not np.allclose(fits[r]["t"], t0, atol=1e-4):
            raise ValueError("dancer_splats: lead and follow fits are on different time grids")
    return fits, info


def pose_fit(d: dict, ks: np.ndarray, device: str = "cpu") -> dict:
    """Posed vertices, global joint rotations and posed joints of grid frames ks; the shaped rest mesh; model data."""
    import smplx
    import torch
    from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints
    md = Path(d["_model_dir"])
    T = len(ks)
    nb = int(d.get("num_betas", len(np.asarray(d["betas"]).reshape(-1))))
    ne = int(np.asarray(d["expression"]).shape[1]) if "expression" in d else 10
    flat = bool(d.get("flat_hand_mean", True))
    m = smplx.create(str(md), model_type="smplx", gender="neutral", ext="npz", use_pca=False, flat_hand_mean=flat,
                     num_betas=nb, num_expression_coeffs=ne, batch_size=T)
    f = lambda a: torch.tensor(np.asarray(a, np.float64)[ks], dtype=torch.float32)  # noqa: E731
    z3 = np.zeros((len(d["t"]), 3))
    betas = torch.tensor(np.asarray(d["betas"], np.float64).reshape(-1)[:nb], dtype=torch.float32)[None].expand(T, -1)
    expr = f(d["expression"]) if "expression" in d else torch.zeros(T, ne)
    with torch.no_grad():
        o = m(betas=betas, expression=expr, global_orient=f(d["global_orient"]), body_pose=f(d["body_pose"]),
              jaw_pose=f(d.get("jaw_pose", z3)), leye_pose=f(d.get("leye_pose", z3)), reye_pose=f(d.get("reye_pose", z3)),
              left_hand_pose=f(d["left_hand_pose"]), right_hand_pose=f(d["right_hand_pose"]), transl=f(d["transl"]),
              return_full_pose=True)
        rot = batch_rodrigues(o.full_pose.reshape(-1, 3)).reshape(T, -1, 3, 3)
        shape_c = torch.cat([betas, expr], 1)
        dirs = torch.cat([m.shapedirs, m.expr_dirs], -1)
        v_sh = m.v_template[None] + blend_shapes(shape_c, dirs)
        J = vertices2joints(m.J_regressor, v_sh)
        Jp, A = batch_rigid_transform(rot, J, m.parents)
        Jp = Jp + f(d["transl"])[:, None]
        err = float((Jp[:, :55] - o.joints[:, :55]).abs().max())
        if err > 2e-4:
            raise ValueError(f"dancer_splats: joint chain misses the SMPL-X joints by {err * 1000:.2f} mm")
        v_rest = (m.v_template + blend_shapes(betas[:1], m.shapedirs)[0]).numpy()  # shaped T-pose, no expression
    mz = np.load(md / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    return {"verts": o.vertices.numpy().astype(np.float32), "Rg": A[:, :55, :3, :3].numpy().astype(np.float32),
            "Jp": Jp[:, :55].numpy().astype(np.float32), "v_rest": v_rest.astype(np.float32),
            "J_rest": (np.asarray(mz["J_regressor"], float) @ v_rest).astype(np.float32)[:55],
            "faces": np.asarray(mz["f"], np.int64), "vt": np.asarray(mz["vt"], np.float32),
            "ft": np.asarray(mz["ft"], np.int64), "weights": np.asarray(mz["weights"], np.float32),
            "parents": _parents(mz), "model": mz, "chain_err_m": err,
            "quality": np.asarray(d.get("quality", np.ones(len(d["t"]))), float)[ks],
            "reproj_px": np.asarray(d.get("reproj_px", np.zeros(len(d["t"]))), float)[ks]}


def vertex_region_bits(mz, weights: np.ndarray) -> np.ndarray:
    from ..avatar_uv import vertex_regions
    reg = vertex_regions(mz)
    dom = weights.argmax(1)
    hand = np.isin(dom, (20, 21)) | (dom >= 25)
    foot = np.isin(dom, (7, 8, 10, 11))
    skin = reg["head"] | hand | np.isin(dom, (12, 18, 19))
    bits = (reg["head"] * R_HEAD + reg["face"] * R_FACE + reg["eye"] * R_EYE + hand * R_HAND + skin * R_SKIN
            + foot * R_FOOT)
    return bits.astype(np.uint8)


def sample_bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """img (H, W, C) float, xy (N, 2) pixel coords -> (N, C) (border clamp)."""
    H, W = img.shape[:2]
    x = np.clip(xy[:, 0], 0, W - 1.001)
    y = np.clip(xy[:, 1], 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1] + (1 - fx) * fy * img[y0 + 1, x0]
            + fx * fy * img[y0 + 1, x0 + 1])


def init_samples(pf: dict, albedo: np.ndarray | None, n: int, seed: int) -> dict:
    """Area-uniform samples of the shaped rest surface with the avatar albedo colour (sRGB) and region bits."""
    faces = pf["faces"]
    fi, bary = triangle_area_samples(pf["v_rest"], faces, n, seed)
    vbits = vertex_region_bits(pf["model"], pf["weights"])
    bits = np.zeros(n, np.uint8)
    for b in (R_HEAD, R_FACE, R_EYE, R_HAND, R_SKIN, R_FOOT):
        on = ((vbits[faces[fi]] & b) > 0).astype(np.float32)
        bits |= np.where((on * bary).sum(1) > 0.5, b, 0).astype(np.uint8)
    if albedo is not None:
        uv = (pf["vt"][pf["ft"][fi]] * bary[..., None]).sum(1)
        S = albedo.shape[0]
        xy = np.stack([uv[:, 0] * S - 0.5, (1 - uv[:, 1]) * S - 0.5], 1)
        rgb = sample_bilinear(albedo, xy)
    else:
        rgb = np.full((n, 3), 0.5)
    return {"face": fi.astype(np.int32), "bary": bary.astype(np.float32), "rgb": rgb.astype(np.float32), "region": bits}


def read_albedo(p: Path) -> np.ndarray | None:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if img is None else img[:, :, ::-1].astype(np.float32) / 255.0


# ------------------------------------------------------------------------------------------------ hair

def _flip_z(u) -> np.ndarray:
    """hair_groom.to_unity is its own inverse (z -> -z): Unity head-bone offsets back to the rest frame."""
    v = np.array(u, float)
    v[..., 2] *= -1
    return v


def groom_colliders(groom: dict, J_rest: np.ndarray) -> dict:
    """The groom's capsules: bone a + offset a -> bone b + offset b (offsets in rest axes = to_unity^-1 of
    oa_unity / ob_unity), radius, first strand point pushed (hair_groom.colliders: arms 5, the rest 0)."""
    cols = [c for c in groom.get("colliders", []) if isinstance(c.get("a"), int) and isinstance(c.get("b"), int)]
    a = np.array([c["a"] for c in cols], np.int64)
    b = np.array([c["b"] for c in cols], np.int64)
    oa = np.array([_flip_z(c.get("oa_unity", (0, 0, 0))) for c in cols], float).reshape(-1, 3)
    ob = np.array([_flip_z(c.get("ob_unity", (0, 0, 0))) for c in cols], float).reshape(-1, 3)
    r = np.array([float(c["r"]) for c in cols], float)
    fp = np.array([int(c.get("first_point", 0)) for c in cols], np.int64)
    J = np.asarray(J_rest, float)
    return {"a": a, "b": b, "oa": oa, "ob": ob, "r": r, "first_point": fp, "names": [c.get("name", "") for c in cols],
            "A_rest": (J[a] + oa) if len(a) else np.zeros((0, 3)), "B_rest": (J[b] + ob) if len(b) else np.zeros((0, 3))}


def _push_out_capsules(X: np.ndarray, A: np.ndarray, Bc: np.ndarray, r: np.ndarray, fp: np.ndarray,
                       margin: float = 0.003) -> np.ndarray:
    """Strand nodes X (G, N, 3) pushed radially out of each capsule (A_c -> B_c, radius r_c) from node fp_c on."""
    for c in range(len(r)):
        P = X[:, fp[c]:]
        ab = Bc[c] - A[c]
        tt = np.clip(((P - A[c]) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0) if ab @ ab > 1e-12 else np.zeros(P.shape[:2])
        Q = A[c] + tt[..., None] * ab
        d = P - Q
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        inside = n[..., 0] < r[c] + margin
        dirn = np.where(n > 1e-9, d / np.maximum(n, 1e-9), np.array([0.0, 0.0, -1.0]))
        P[inside] = (Q + dirn * (r[c] + margin))[inside]
    return X


def drape_rest(nodes: np.ndarray, seg: np.ndarray, centre: np.ndarray, r_head: float, cols: dict,
               hold_above_y: float, iters: int = 160, g_step: float = 0.004, clearance: float = 0.005) -> np.ndarray:
    """Short position-based settle of the rest strands in the T-pose (prep, numpy): the part over the skull (initial
    node height above hold_above_y) keeps its comb, the rest falls under gravity over the head sphere and the groom's
    capsules (with their offsets: head, neck, shoulder bar, chest, mid-back; arms from point 5), segment lengths kept
    exactly (follow-the-leader from the fixed root). A stand-in for the Unity settle, so the photos refine a drape
    that already clears the trapezius instead of strands through it."""
    X0 = np.asarray(nodes, float)
    X = X0.copy()
    hold = X0[..., 1] > hold_above_y
    hold[:, 0] = True
    A, Bc, rr, fp = cols["A_rest"], cols["B_rest"], cols["r"], np.maximum(cols["first_point"], 1)
    rs = r_head + clearance

    def project():
        d = X[:, 1:] - centre
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        X[:, 1:] = np.where(n < rs, centre + d / np.maximum(n, 1e-9) * rs, X[:, 1:])
        _push_out_capsules(X, A, Bc, rr, fp)
        for i in range(1, X.shape[1]):  # inextensible: follow-the-leader from the root
            e = X[:, i] - X[:, i - 1]
            X[:, i] = X[:, i - 1] + e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-9) * seg[:, i - 1:i]

    for _ in range(iters):
        X[..., 1] -= np.where(hold, 0.0, g_step)
        X[hold] = X0[hold]
        project()
    return X


def hair_rest(groom: dict, pf: dict) -> dict:
    """The groom's guides as rest polylines in the head joint's rest frame (rest rotations are the identity).

    hair_groom.json frames: model.head_sphere_centre_unity, guides[].pos_unity and colliders[].o?_unity are Unity
    offsets FROM THE HEAD JOINT (to_unity(c - J_head), z flipped); model.head_joint_rest is the head joint in the groom's
    RestModel (soles at y = 0). The bundle's rest frame is the shaped SMPL-X T-pose (soles at y0 = min v_rest.y), so
    rest = J_rest[15] + flip_z(offset); head_joint_rest - (0, -y0, 0) must equal J_rest[15] (same fit + shape)."""
    from ..avatar_uv import vertex_normals
    from ..hair_groom import linear_to_srgb
    v = pf["v_rest"].astype(float)
    J_rest = np.asarray(pf["J_rest"], float)
    J15 = J_rest[HEAD_JOINT]
    y0 = float(v[:, 1].min())
    hj = groom["model"].get("head_joint_rest")
    if hj is not None:
        err = float(np.linalg.norm(np.asarray(hj, float) + np.array([0.0, y0, 0.0]) - J15))
        if err > 0.01:
            raise ValueError(f"dancer_splats: hair_groom.json's head joint misses this fit's rest head joint by "
                             f"{err * 100:.1f} cm (groom made for another fit / shape): re-run hair_groom")
    centre = J15 + _flip_z(groom["model"]["head_sphere_centre_unity"])
    r_head = float(groom["model"]["head_radius_m"])
    S = int(groom["strands"]["segments"])
    n = vertex_normals(v, pf["faces"])
    standoff = float(groom.get("shape", {}).get("standoff_m", 0.01))
    nodes, roots, root_err = [], [], []
    for g in groom["guides"]:
        vid, b = np.asarray(g["vid"], int), np.asarray(g["bary"], float)
        root = (v[vid] * b[:, None]).sum(0)
        nr = (n[vid] * b[:, None]).sum(0)
        nr /= max(np.linalg.norm(nr), 1e-9)
        nodes.append(rest_strand(root, nr, centre, r_head, float(g["len"]), S + 1, standoff))
        roots.append(root)
        if "pos_unity" in g:
            root_err.append(float(np.linalg.norm(J15 + _flip_z(g["pos_unity"]) - root)))
    if root_err and float(np.median(root_err)) > 0.01:
        raise ValueError(f"dancer_splats: the groom's root positions miss the scalp of this fit by "
                         f"{np.median(root_err) * 100:.1f} cm (median): re-run hair_groom")
    nodes = np.array(nodes)
    seg = np.linalg.norm(np.diff(nodes, axis=1), axis=2)
    cols = groom_colliders(groom, J_rest)
    straight = nodes
    nodes = drape_rest(nodes, seg, centre, r_head, cols, hold_above_y=float(centre[1]))
    pal = np.asarray(groom["colour"]["strand_palette_linear"], float)
    st = groom["strands"]
    cl = np.c_[cols["a"], cols["b"], cols["r"], cols["first_point"]].astype(np.float32).reshape(-1, 4)
    return {"rest_local": (nodes - J15).astype(np.float32), "roots_local": (np.array(roots) - J15).astype(np.float32),
            "root_vid": np.array([g["vid"] for g in groom["guides"]], np.int32),
            "root_bary": np.array([g["bary"] for g in groom["guides"]], np.float32),
            "seg_len": seg.astype(np.float32),
            "palette_srgb": np.clip(linear_to_srgb(pal), 0, 1).astype(np.float32),
            "children": np.array(int(st.get("children", 3))), "child_spread": np.array(float(st.get("child_spread_m", 0.008))),
            "width_root": np.array(float(st.get("width_root_m", 0.028))),
            "width_tip": np.array(float(st.get("width_tip_m", 0.008))),
            "head_centre_local": (centre - J15).astype(np.float32), "head_radius": np.array(r_head),
            "colliders": cl, "collider_oa": cols["oa"].astype(np.float32).reshape(-1, 3),
            "collider_ob": cols["ob"].astype(np.float32).reshape(-1, 3),
            "root_check_mm": np.array(round(float(np.median(root_err)) * 1000, 2) if root_err else -1.0),
            "drape_shift_m": np.array(round(float(np.linalg.norm(nodes - straight, axis=-1).max()), 4)),
            "role": np.array(groom.get("role", "follow"))}


# ------------------------------------------------------------------------------------------------ avatar render

def render_avatar(torch, verts: dict, faces, ft, vt, albedo: dict, K, R, t, H: int, W: int):
    """The textured avatar (albedo on the posed SMPL-X, no lighting, no hair) through a pinhole crop camera:
    (H, W, 3) float sRGB in [0, 1] (black background), role map (H, W) int8 (-1 background, 0 lead, 1 follow)."""
    from ..avatar_raster import EMPTY_Z, pixel_bary, rasterize
    dev = faces.device
    Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=dev)
    Rt = torch.as_tensor(np.asarray(R), dtype=torch.float32, device=dev)
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=dev)
    layers = []
    for n, (r, V) in enumerate(verts.items()):
        Vt = torch.as_tensor(np.asarray(V), dtype=torch.float32, device=dev)
        Xc = Vt @ Rt.T + tt
        z = Xc[:, 2]
        px = torch.stack([Kt[0, 0] * Xc[:, 0] / z.clamp_min(1e-4) + Kt[0, 2],
                          Kt[1, 1] * Xc[:, 1] / z.clamp_min(1e-4) + Kt[1, 2]], 1)
        zb, fid = rasterize(torch, px, z, faces, H, W)
        layers.append((r, zb, fid, px, z))
    zs = torch.stack([lay[1] for lay in layers])
    zmin, win = zs.min(0)
    covered = zmin < EMPTY_Z
    img = torch.zeros((H, W, 3), device=dev)
    idm = torch.full((H, W), -1, device=dev, dtype=torch.int8)
    for n, (r, zb, fid, px, z) in enumerate(layers):
        sel = covered & (win == n)
        if not sel.any():
            continue
        bary = pixel_bary(torch, px, z, faces, torch.where(sel, fid, torch.full_like(fid, -1)))
        ys, xs = torch.nonzero(sel, as_tuple=True)
        f = fid[ys, xs]
        uv = (vt[ft[f]] * bary[ys, xs][..., None]).sum(1)
        g = torch.stack([2 * uv[:, 0] - 1, 1 - 2 * uv[:, 1]], 1)[None, None]
        col = torch.nn.functional.grid_sample(albedo[r][None], g, mode="bilinear", padding_mode="border",
                                              align_corners=False)[0, :, 0].T
        img[ys, xs] = col
        idm[ys, xs] = n
    return img.clamp(0, 1).cpu().numpy(), idm.cpu().numpy()


# ------------------------------------------------------------------------------------------------ views

def _box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _union_box(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _read_mask(p: Path, shape) -> np.ndarray | None:
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def _roles_swap(work: Path) -> dict:
    p = work / "dancers" / "roles.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    return {c: {f["i"]: float(f.get("swap_prob", 0.0)) for f in v["frames"]} for c, v in r.get("cams", {}).items()}


def take_cams(take: str, work: Path, want: tuple[str, ...] = ()) -> list[str]:
    from ..take import Take
    tk = Take.load(take)
    cams = [s.id for s in tk.sources if (work / "cameras" / f"{s.id}.npz").exists()
            and (work / "frames" / s.id / "frames.json").exists()]
    if want:
        missing = sorted(set(want) - set(cams))
        if missing:
            raise SystemExit(f"dancer_splats: cameras {missing} have no cameras/<cam>.npz or frames")
        cams = [c for c in cams if c in want]
    return cams


def _camera_types(take: str, cams: list[str]) -> dict:
    try:
        from ..take import Take
        tps = Take.load(take).camera_types(cams)
        return {c: {"mount": t.mount, "zoom": bool(t.zoom), "source": t.source} for c, t in tps.items()}
    except Exception:  # noqa: BLE001 - informational only
        return {c: {"mount": "handheld", "zoom": True, "source": "default"} for c in cams}


# ------------------------------------------------------------------------------------------------ prepare

def _prep_inputs(work: Path, info: dict, cams: list[str]) -> dict:
    files = [work / info["smplx_dir"] / f"{r}.npz" for r in ROLES] + [
        work / "avatar" / "head_fix.npz", work / "avatar" / "hair_groom.json", work / "sync" / "offsets.json",
        work / "cameras" / "world_from_sfm.json", work / "dancers" / "roles.json"] + [
        work / "smplx_face" / f"{r}.npz" for r in ROLES] + [work / "avatar" / f"{r}_albedo.png" for r in ROLES] + [
        work / "cameras" / f"{c}.npz" for c in cams]
    return {f.relative_to(work).as_posix(): round(f.stat().st_mtime, 3) for f in files if f.exists()}


def prep_code_sha1() -> str:
    """The prep's own code: a changed crop / label / posing rule makes every bundle stale."""
    import hashlib
    h = hashlib.sha1()
    for f in ("data.py", "binding.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:12]


def bundle_fresh(work: Path, bundle: Path, cfg: SplatConfig, take: str) -> bool:
    mp = bundle / "manifest.json"
    if not mp.exists():
        return False
    m = json.loads(mp.read_text())
    try:
        _, info = load_fits(take, work)
        cams = m.get("cams") or []
    except Exception:  # noqa: BLE001
        return False
    return (m.get("version") == BUNDLE_VERSION and m.get("prep_config") == json.loads(json.dumps(prep_dict(cfg)))
            and m.get("inputs") == _prep_inputs(work, info, cams) and m.get("code") == prep_code_sha1())


def smplx_fresh(work: Path, sdir: str) -> dict:
    w = work / "cameras" / "world_from_sfm.json"
    sm = [work / sdir / f"{r}.npz" for r in ROLES]
    ts = min(p.stat().st_mtime for p in sm)
    tw = w.stat().st_mtime if w.exists() else None
    return {"stale": bool(tw is not None and ts < tw), "smplx_mtime": ts, "world_from_sfm_mtime": tw}


def prepare(take: str, cfg: SplatConfig, out: Path | None = None, log=print, force: bool = False,
            allow_stale: bool = False, device: str = "cpu") -> dict:
    """Write <out>/bundle (default work/<take>/dancer_splats/bundle). Skipped when fresh (force rewrites)."""
    import shutil

    import torch

    from ..mamma_dyn import apply_offset, read_sync_offsets, world_cameras
    from ..take import PACKAGE_ROOT
    work = PACKAGE_ROOT / "work" / take
    out = Path(out) if out is not None else work / "dancer_splats"
    bundle = out / "bundle"
    if not force and bundle_fresh(work, bundle, cfg, take):
        m = json.loads((bundle / "manifest.json").read_text())
        log(f"[dancer_splats] prepare: bundle up to date ({m['n_views']} views, {m['bytes'] / 1e6:.0f} MB) -> {bundle}")
        return m
    t_start = time.time()
    fits, finfo = load_fits(take, work)
    fr = smplx_fresh(work, finfo["smplx_dir"])
    if fr["stale"] and not allow_stale:
        raise SystemExit(f"dancer_splats: {finfo['smplx_dir']}/ is older than cameras/world_from_sfm.json (a fit in a "
                         "previous world frame): re-run the MAMMA fit or pass --allow-stale")
    cams = take_cams(take, work, tuple(cfg.cams))
    t_all = np.asarray(fits["lead"]["t"], float)
    w0, w1 = float(cfg.window[0]), float(cfg.window[1])
    inside = np.nonzero((t_all >= w0 - 1e-6) & (t_all <= w1 + 1e-6))[0]
    if len(inside) < 2:
        raise SystemExit(f"dancer_splats: window {w0}-{w1} s has fewer than 2 fit frames ({t_all[0]:.2f}-{t_all[-1]:.2f})")
    ks = np.arange(max(0, inside[0] - 1), min(len(t_all), inside[-1] + 2))
    t_grid = t_all[ks]
    k_win = inside - ks[0]                      # grid indices (into t_grid) inside the window: the exported frames
    log(f"[dancer_splats] prepare {take}: window {w0}-{w1} s = {len(inside)} fit frames, cams {','.join(cams)}, "
        f"held out {','.join(cfg.holdout_cams) or 'none'}; motion {finfo['smplx_dir']} ({finfo['smplx_dir_source']})")
    pfs = {r: pose_fit(fits[r], ks) for r in ROLES}
    log(f"[dancer_splats] bodies posed: chain vs SMPL-X joints <= "
        f"{max(p['chain_err_m'] for p in pfs.values()) * 1000:.3f} mm; head fix "
        f"{[r for r in ROLES if finfo['roles'][r]['head_fix']]}; face {[r for r in ROLES if finfo['roles'][r]['face']]}")
    if bundle.exists():
        shutil.rmtree(bundle)
    for sub in ("images", "labels", "eval"):
        (bundle / sub).mkdir(parents=True)
    # ---- bodies
    p0 = pfs["lead"]
    W = p0["weights"]
    w_idx = np.argsort(-W, 1)[:, :4]
    w_val = np.take_along_axis(W, w_idx, 1)
    w_val /= w_val.sum(1, keepdims=True)
    bod = {"t_grid": t_grid.astype(np.float64), "k_window": k_win.astype(np.int32), "faces": p0["faces"].astype(np.int32),
           "parents": p0["parents"].astype(np.int32), "w_idx": w_idx.astype(np.int32), "w_val": w_val.astype(np.float32),
           "vt": p0["vt"], "ft": p0["ft"].astype(np.int32), "window": np.array([w0, w1])}
    for r in ROLES:
        for k in ("verts", "Rg", "Jp", "v_rest", "J_rest", "quality", "reproj_px"):
            bod[f"{r}_{k}"] = pfs[r][k]
    np.savez_compressed(bundle / "bodies.npz", **bod)
    # ---- init samples + albedos
    albedo = {}
    for i, r in enumerate(ROLES):
        albedo[r] = read_albedo(work / "avatar" / f"{r}_albedo.png")
        s = init_samples(pfs[r], albedo[r], cfg.init_samples, cfg.seed + i)
        np.savez_compressed(bundle / f"init_{r}.npz", **s)
    # ---- hair
    gp = work / "avatar" / "hair_groom.json"
    hair_info = None
    if gp.exists():
        groom = json.loads(gp.read_text())
        role = groom.get("role", "follow")
        hr = hair_rest(groom, pfs[role])
        np.savez_compressed(bundle / "hair.npz", **hr)
        hair_info = {"role": role, "guides": int(hr["rest_local"].shape[0]), "segments": int(hr["seg_len"].shape[1]),
                     "length_m": [round(float(hr["seg_len"].sum(1).min()), 3), round(float(hr["seg_len"].sum(1).max()), 3)]}
    # ---- views
    sim3_p = work / "cameras" / "world_from_sfm.json"
    sim3 = json.loads(sim3_p.read_text()) if sim3_p.exists() else None
    offs = read_sync_offsets(work) if (work / "sync" / "offsets.json").exists() else {}
    swaps = _roles_swap(work)
    ctypes = _camera_types(take, cams)
    hair_role = hair_info["role"] if hair_info else "follow"
    views = []
    n_skip = {"camera_not_ok": 0, "no_mask": 0}
    dt = float(np.median(np.diff(t_all)))
    win_rel = lambda t: (t - t_all[inside[0]]) / dt  # noqa: E731  (grid frames from the window's first)
    for cam in cams:
        z = np.load(work / "cameras" / f"{cam}.npz")
        Rw, tw = world_cameras(z["R"], z["tvec"], sim3)
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        okc = z["ok"].astype(bool) if "ok" in z.files else np.ones(len(Rw), bool)
        k1s = z["dist"] if "dist" in z.files else np.zeros(len(Rw))
        f_min = float(np.min(z["K"][okc][:, 0, 0])) if okc.any() else float(z["K"][0, 0, 0])
        tf = np.array([float(f["t"]) for f in meta["frames"]])
        tc = apply_offset(tf, offs.get(cam)) if cam in offs else tf
        keep = None
        if len(tc) > 2 and float(np.median(np.diff(tc))) < 0.8 * dt:  # a 59.94 fps camera: the frame nearest
            keep = {int(np.argmin(np.abs(tc - tg))) for tg in t_all[inside]}  # each 30 Hz grid instant
        for i, fr_ in enumerate(meta["frames"]):
            t_frame = float(tf[i])
            t = float(tc[i])
            if not (w0 <= t <= w1) or i >= len(Rw) or (keep is not None and i not in keep):
                continue
            if not okc[i]:
                n_skip["camera_not_ok"] += 1
                continue
            img = cv2.imread(str(work / "frames" / cam / fr_["file"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H0, W0 = img.shape[:2]
            dd = work / "dancers" / cam
            lead = _read_mask(dd / "masks" / f"mask_{i:04d}_{PIDS['lead']:02d}.png", (H0, W0))
            fol = _read_mask(dd / "masks" / f"mask_{i:04d}_{PIDS['follow']:02d}.png", (H0, W0))
            if lead is None and fol is None:
                n_skip["no_mask"] += 1
                continue
            lead = np.zeros((H0, W0), bool) if lead is None else lead
            fol = np.zeros((H0, W0), bool) if fol is None else fol
            hair = _read_mask(dd / "hair" / f"mask_{i:04d}_{PIDS[hair_role]:02d}.png", (H0, W0))
            if hair is None:
                body = _read_mask(dd / "body" / f"mask_{i:04d}_{PIDS[hair_role]:02d}.png", (H0, W0))
                full = fol if hair_role == "follow" else lead
                hair = (full & ~body) if body is not None else None
            people = others_ignore(_read_mask(work / "people_masks" / cam / f"{i:06d}.png", (H0, W0)), lead | fol)
            K = np.asarray(z["K"][i], float)
            k1 = float(k1s[i])
            img = undistort(img, K, k1)
            und = lambda m: None if m is None else undistort(m.astype(np.uint8) * 255, K, k1, cv2.INTER_NEAREST) > 127  # noqa: E731
            lead, fol, hair, people = und(lead), und(fol), und(hair), und(people)
            valid = undistort(np.full((H0, W0), 255, np.uint8), K, k1, cv2.INTER_NEAREST) > 127
            # crop: the dancers' masks + their projected posed bodies
            box = _union_box(_box(lead), _box(fol))
            pv = {r: at_time(pfs[r]["verts"], t_grid, t) for r in ROLES}
            for r in ROLES:
                px, zz = project(pv[r], K, Rw[i], tw[i])
                px = px[zz > 0.1]
                if len(px):
                    box = _union_box(box, (int(px[:, 0].min()), int(px[:, 1].min()), int(px[:, 0].max()) + 1,
                                           int(px[:, 1].max()) + 1))
            pad = cfg.crop_pad * max(box[2] - box[0], box[3] - box[1])
            x0, y0 = int(max(0, math.floor(box[0] - pad))), int(max(0, math.floor(box[1] - pad)))
            x1, y1 = int(min(W0, math.ceil(box[2] + pad))), int(min(H0, math.ceil(box[3] + pad)))
            if x1 - x0 < 16 or y1 - y0 < 16:
                n_skip["no_mask"] += 1
                continue
            crop = lambda a: None if a is None else a[y0:y1, x0:x1]  # noqa: E731
            im = img[y0:y1, x0:x1]
            L = encode_labels(crop(lead), crop(fol), crop(hair), crop(people), crop(valid))
            Kc = crop_K(K, x0, y0)
            Hc, Wc = im.shape[:2]
            if cfg.max_side and max(Hc, Wc) > cfg.max_side:
                f = max(Hc, Wc) / cfg.max_side
                W2, H2 = max(16, round(Wc / f)), max(16, round(Hc / f))
                Kc = scale_K(Kc, Wc / W2, Hc / H2)
                im = cv2.resize(im, (W2, H2), interpolation=cv2.INTER_AREA)
                L = cv2.resize(L, (W2, H2), interpolation=cv2.INTER_NEAREST)
                Hc, Wc = H2, W2
            fg = decode_labels(L)["fg"]
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            sel = cv2.dilate(fg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            sharp = float(lap[sel].var()) if sel.sum() > 50 else 0.0
            # inter-ocular distance of each dancer from the posed eye joints (23, 24)
            iod = {}
            for r in ROLES:
                eyes = at_time(pfs[r]["Jp"][:, 23:25], t_grid, t)
                pe, ze = project(eyes, Kc, Rw[i], tw[i])
                iod[r] = round(float(np.linalg.norm(pe[0] - pe[1])), 2) if (ze > 0.1).all() else 0.0
            j = len(views)
            cv2.imwrite(str(bundle / "images" / f"{j:04d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(bundle / "labels" / f"{j:04d}.png"), L, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            kk = int(round(win_rel(t)))
            split = ("test_cam" if cam in cfg.holdout_cams else
                     "test_time" if cfg.holdout_every and kk % cfg.holdout_every == cfg.holdout_every // 2 else "train")
            views.append({"i": j, "name": f"images/{j:04d}.jpg", "labels": f"labels/{j:04d}.png", "cam": cam,
                          "frame": i, "t_frame": round(t_frame, 5), "t": round(t, 5), "K": Kc.tolist(),
                          "R": Rw[i].tolist(), "t_w": tw[i].tolist(), "W": int(Wc), "H": int(Hc),
                          "crop": [x0, y0, x1, y1], "native": [W0, H0], "split": split, "grid_rel": kk,
                          "swap_prob": round(swaps.get(cam, {}).get(i, 0.0), 3), "sharpness": round(sharp, 2),
                          "zoom": round(float(K[0, 0] / f_min), 3), "camera_type": ctypes.get(cam),
                          "iod_px": iod, "face_hd": bool(max(iod.values()) >= cfg.face_iod_px),
                          "fg_frac": round(float(fg.mean()), 4), "eval_train": False, "review": False})
        log(f"[dancer_splats] prepare: cam {cam}: {sum(v['cam'] == cam for v in views)} views")
    if not views:
        raise SystemExit("dancer_splats: no views in the window")
    # per-camera sharpness weight (motion-blurred / shaken frames count less)
    for cam in cams:
        s = [v["sharpness"] for v in views if v["cam"] == cam and v["sharpness"] > 0]
        med = float(np.median(s)) if s else 1.0
        for v in views:
            if v["cam"] == cam:
                v["weight"] = round(float(np.clip(math.sqrt(max(v["sharpness"], 1e-6) / max(med, 1e-6)), 0.5, 1.0)), 3)
    tr = [v for v in views if v["split"] == "train"]
    for v in tr[::max(1, len(tr) // max(1, cfg.eval_train_views))][:cfg.eval_train_views]:
        v["eval_train"] = True
    hold = [v for v in views if v["split"] == "test_cam"]
    for tr_ in REVIEW_TIMES:
        if hold and w0 <= tr_ <= w1:
            min(hold, key=lambda v: abs(v["t"] - tr_))["review"] = True
    if not hold:  # no held-out camera (smoke): review the in-sample eval views
        for v in [v for v in views if v["eval_train"]][:3]:
            v["review"] = True
    # ---- avatar renders of the held-out views + the novel camera
    dev = torch.device(device)
    faces_t = torch.as_tensor(bod["faces"], dtype=torch.long, device=dev)
    ft_t = torch.as_tensor(bod["ft"], dtype=torch.long, device=dev)
    vt_t = torch.as_tensor(bod["vt"], dtype=torch.float32, device=dev)
    alb = {r: torch.as_tensor((albedo[r] if albedo[r] is not None else np.full((8, 8, 3), 0.5, np.float32))
                              .transpose(2, 0, 1).copy(), device=dev) for r in ROLES}
    t_av = time.time()
    n_av = 0
    for v in views:
        if v["split"] == "train" and not v["eval_train"]:
            continue
        pv = {r: at_time(pfs[r]["verts"], t_grid, v["t"]) for r in ROLES}
        img, idm = render_avatar(torch, pv, faces_t, ft_t, vt_t, alb, v["K"], v["R"], v["t_w"], v["H"], v["W"])
        cv2.imwrite(str(bundle / "eval" / f"avatar_{v['i']:04d}.jpg"), (img[:, :, ::-1] * 255).round().astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(bundle / "eval" / f"avatar_{v['i']:04d}.png"), (idm.astype(np.int16) + 1).astype(np.uint8))
        n_av += 1
    log(f"[dancer_splats] prepare: avatar reference renders for {n_av} views in {time.time() - t_av:.0f}s")
    novel = novel_camera(views, cfg, pfs, t_grid, k_win)
    if novel is not None:
        prev, ch = None, []
        for n_k, k in enumerate(k_win):
            pv = {r: pfs[r]["verts"][k] for r in ROLES}
            img, idm = render_avatar(torch, pv, faces_t, ft_t, vt_t, alb, novel["K"], novel["R"], novel["t_w"],
                                     novel["H"], novel["W"])
            if prev is not None:
                sel = (idm >= 0) | (prev[1] >= 0)
                if sel.any():
                    ch.append(float(np.abs(img - prev[0])[sel].mean()))
            if n_k in (len(k_win) // 4, len(k_win) // 2, 3 * len(k_win) // 4):
                cv2.imwrite(str(bundle / "eval" / f"novel_avatar_{int(k):04d}.jpg"),
                            (img[:, :, ::-1] * 255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
            prev = (img, idm)
        novel["avatar_change"] = round(float(np.mean(ch)), 5) if ch else None
        (bundle / "novel.json").write_text(json.dumps(novel, indent=1))
    (bundle / "views.json").write_text(json.dumps(views))
    nbytes = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file())
    man = {"version": BUNDLE_VERSION, "take": take, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "prep_config": json.loads(json.dumps(prep_dict(cfg))), "inputs": _prep_inputs(work, finfo, cams),
           "code": prep_code_sha1(),
           "cams": cams, "fits": finfo, "smplx_fresh": {"stale": fr["stale"]},
           "n_views": len(views), "n_train": len(tr), "n_test_cam": len(hold),
           "n_test_time": sum(v["split"] == "test_time" for v in views),
           "grid_frames": int(len(t_grid)), "window_frames": int(len(k_win)), "hair": hair_info,
           "skipped": n_skip, "resolutions": sorted({f"{v['cam']}:{v['W']}x{v['H']}" for v in views})[:40],
           "shape_edits": "not applied: the splats follow the photos (shape_edits only reshape the Unity skins)",
           "bytes": nbytes, "prep_s": round(time.time() - t_start, 1)}
    (bundle / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"[dancer_splats] prepare: {len(views)} views ({len(tr)} train, {man['n_test_cam']} held-out camera, "
        f"{man['n_test_time']} held-out instants), {nbytes / 1e6:.1f} MB in {man['prep_s']:.0f}s -> {bundle}")
    return man


def novel_camera(views: list[dict], cfg: SplatConfig, pfs: dict, t_grid: np.ndarray, k_win: np.ndarray) -> dict | None:
    """A fixed camera halfway between the two novel_pair cameras (mid window), looking at the couple; the crop covers
    both dancers over the whole window. None when a camera of the pair has no view."""
    a, b = cfg.novel_pair
    mid = 0.5 * (cfg.window[0] + cfg.window[1])
    va = [v for v in views if v["cam"] == a]
    vb = [v for v in views if v["cam"] == b]
    if not va or not vb:
        return None
    va = min(va, key=lambda v: abs(v["t"] - mid))
    vb = min(vb, key=lambda v: abs(v["t"] - mid))
    C = [-np.asarray(v["R"]).T @ np.asarray(v["t_w"]) for v in (va, vb)]
    centre = 0.5 * (C[0] + C[1])
    allv = np.concatenate([pfs[r]["verts"][k_win].reshape(-1, 3) for r in ROLES])
    target = np.median(allv, 0)
    zax = target - centre
    zax /= np.linalg.norm(zax)
    xax = np.cross(zax, np.array([0.0, 1.0, 0.0]))
    xax /= np.linalg.norm(xax)
    yax = np.cross(zax, xax)
    R = np.stack([xax, yax, zax])
    t = -R @ centre
    # natives of the pair's views: focal at native resolution (crop K keeps fx)
    f = 0.5 * (va["K"][0][0] + vb["K"][1][1])
    K = np.array([[f, 0, 0.0], [0, f, 0.0], [0, 0, 1.0]])
    px, z = project(allv[::7], K, R, t)
    px = px[z > 0.1]
    lo, hi = px.min(0), px.max(0)
    pad = 0.08 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    s = min(1.0, 900.0 / max(hi - lo))
    K = np.array([[f * s, 0, -lo[0] * s], [0, f * s, -lo[1] * s], [0, 0, 1.0]])
    W, H = int(math.ceil((hi[0] - lo[0]) * s)), int(math.ceil((hi[1] - lo[1]) * s))
    return {"between": [a, b], "K": K.tolist(), "R": R.tolist(), "t_w": t.tolist(), "W": W, "H": H,
            "grid": [int(k) for k in k_win], "t": [round(float(t_grid[k]), 5) for k in k_win]}


# ------------------------------------------------------------------------------------------------ loader

def load_bundle(bundle: Path) -> dict:
    """Everything train / eval / export read (numpy); images and labels stay on disk (read_view)."""
    bundle = Path(bundle)
    mp = bundle / "manifest.json"
    if not mp.exists():
        raise FileNotFoundError(f"dancer_splats: no bundle at {bundle}; run `prepare` locally first")
    man = json.loads(mp.read_text())
    if man.get("version") != BUNDLE_VERSION:
        raise ValueError(f"dancer_splats: bundle version {man.get('version')} != {BUNDLE_VERSION}; rerun prepare")
    with np.load(bundle / "bodies.npz") as z:
        bodies = {k: z[k] for k in z.files}
    init = {}
    for r in ROLES:
        with np.load(bundle / f"init_{r}.npz") as z:
            init[r] = {k: z[k] for k in z.files}
    hair = None
    if (bundle / "hair.npz").exists():
        with np.load(bundle / "hair.npz") as z:
            hair = {k: z[k] for k in z.files}
    novel = json.loads((bundle / "novel.json").read_text()) if (bundle / "novel.json").exists() else None
    return {"dir": bundle, "manifest": man, "views": json.loads((bundle / "views.json").read_text()),
            "bodies": bodies, "init": init, "hair": hair, "novel": novel}


def read_view(bundle: Path, v: dict) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, 3) uint8 RGB crop, (H, W) uint8 labels."""
    img = cv2.imread(str(Path(bundle) / v["name"]), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(Path(bundle) / v["name"])
    L = cv2.imread(str(Path(bundle) / v["labels"]), cv2.IMREAD_GRAYSCALE)
    return img[:, :, ::-1].copy(), L


def read_avatar(bundle: Path, v: dict) -> tuple[np.ndarray, np.ndarray] | None:
    p = Path(bundle) / "eval" / f"avatar_{v['i']:04d}.jpg"
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
    idm = cv2.imread(str(p.with_suffix(".png")), cv2.IMREAD_GRAYSCALE).astype(np.int16) - 1
    return img, idm
