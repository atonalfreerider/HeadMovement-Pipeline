"""Photoreal faces for the SMPL-X avatars from one studio portrait per dancer (likeness data: local only).

    python -m dancecap.avatar_face <take>                       portrait fit + UV projection + composite onto the bake
    python -m dancecap.avatar_face <take> --composite           reuse the saved projection (after a re-bake)
    python -m dancecap.avatar_face <take> --bake follow=X.png   composite onto any baked albedo (2048 SMPL-X UV)

Outputs NEXT TO the bake (never over it): <stem>_face.png (portrait look) and <stem>_face_videolook.png (video-matched
look, from face/look.json); default bake avatar/<role>_albedo.png -> avatar/<role>_albedo_face[_videolook].png (also
copied to the legacy avatar/face/<role>_albedo_face.png). Per take in avatar/face/: <role>_portrait_fit.npz,
_portrait_uv.npz (texels, rgb, w, labels), _face_layer.png (RGBA), _texel_masks.npz, _eyes.json, report.json.

Inputs: the portraits <source_dir>/<name>.jpg (takes/<take>.toml [dancers.<role>].name, or [dancers.<role>].portrait),
their MediaPipe landmarks (face_worker.py in .venv-face, run here as a subprocess when missing: seconds, CPU), the
dancer's SMPL-X betas (smplx/<role>.npz, MAMMA locked-head model) and the baked albedo.

Method
  1. Fit (weak perspective): head rotation, scale, 2D offset, jaw and N_EXPR SMPL-X expression coefficients so that
     the avatar's 52 iBUG landmarks (face_common) project onto the portrait's MediaPipe landmarks (mapped to iBUG).
     Identity differences the locked head cannot express remain as landmark residuals.
  2. Residual warp: a thin-plate spline through those residuals moves every sample point so that each avatar
     landmark samples exactly the portrait's landmark (eyes land on the mesh's eyes in UV).
  3. Portrait mask: face oval (MediaPipe silhouette, top extended forehead_extend_iod up the forehead) x skin-vs-
     occluder colour test (Lab Gaussians: skin seeds on the cheeks/nose vs pixels in a ring outside the oval = hair /
     arm / background), with eyes, brows and lips always kept and the inner mouth (teeth) inpainted from the lips;
     lead: the lower face is always kept (beard). Texel labels (lips / chin-beard / cheeks) come from the same
     landmarks.
  4. UV projection: every head texel of the 2048 atlas is placed on the fitted mesh, tested for visibility (z-buffer
     of the fitted mesh in the portrait view) and weighted by facing (cos 0.25 -> 0.6) x mask. Low-weight texels take
     the mirrored texel's portrait colour at 0.7 weight (turned heads).
  5. Composite onto the bake: colour transfer to the bake's lighting (linear RGB; global robust gain on trusted skin
     texels + a smooth quadratic gain field over the face, clipped), then multi-band (Laplacian pyramid) blending.
  6. Skin repair (follow by default): face texels the portrait does not cover whose bake colour is hair-like (closer
     to the scalp's colour than to the face's skin) take the mirrored texel or a push-pull skin fill, feathered.
  7. Mouth interior: the lips' inner surfaces (never visible with the mouth closed) are painted with the lip colour,
     darkening inwards - otherwise they show as an orange slit when the jaw / smile opens the mouth.
  8. Procedural eyeballs (portrait iris / sclera colours). 9. Video-matched variant (face_look targets: the follow's
     lip colour change attributable to makeup, the lead's chin beard lightness): masked Lab shifts. The user picks.
No generative restoration (GFPGAN / CodeFormer / diffusion) is used: they change identity. Upscaling is bicubic.

Review: review/face_portrait_fit.jpg (fit, residuals, mask, re-render from the portrait view),
        review/avatar_face_portrait.jpg (reference | old | new | video-look at 3 head angles, both dancers),
        review/avatar_face_turntable.jpg, review/avatar_face_video.jpg (video crop | old | new, same pose + camera),
        review/avatar_face_uv.jpg.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .face_common import (LABEL, MP_TO_IBUG68, ROLES, LandmarkEmbedding, crop_pad, label, smplx_model_npz)
from .take import WORKSPACE, Take

FACE_VENV_PY = WORKSPACE / ".venv-face" / "Scripts" / "python.exe"

# MediaPipe index sets
MP_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
           149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
MP_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MP_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
MP_EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MP_EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
MP_BROW_R = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
MP_BROW_L = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
MP_IRIS_R = [468, 469, 470, 471, 472]
MP_IRIS_L = [473, 474, 475, 476, 477]
MP_SKIN_SEEDS = [50, 280, 4, 151, 205, 425, 9, 117, 346]
MP_LOWER_FACE = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323,
                 454, 352, 345, 280, 425, 411, 2, 187, 205, 50, 116, 123]
# chin / jaw beard region: lower outer lip (right corner -> left corner) + the jaw line back (texel label LAB_BEARD)
MP_CHIN_REGION = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                  176, 149, 150, 136, 172, 58, 132]
MP_CHEEKS = (50, 280)
LAB_LIPS, LAB_BEARD, LAB_CHEEK = 1, 2, 8     # per-texel portrait region labels (bits)
LAB_MESH_LIPS = 16                           # the mesh's visible lips (mouth_masks), not from the portrait


@dataclass
class FaceTexConfig:
    n_expr: int = 20
    expr_prior: float = 3e-3
    jaw_prior: float = 1e-2
    cos_lo: float = 0.25
    cos_hi: float = 0.60
    mirror_weight: dict = field(default_factory=lambda: {"lead": 0.7, "follow": 0.95})   # mirrored portrait samples
                                                 # vs the bake (follow: her turned-away side's bake is hair)
    mirror_blend_sigma: float = 6.0             # texels: the own -> mirrored hand-over is smoothed in UV (a narrow
                                                 # switch where the facing drops fast drew a line along the nose)
    tps_smooth: float = 2.0                      # px^2 regularisation of the residual spline
    skin_maha: float = 3.0                       # skin Mahalanobis gate (Lab a,b + 0.5 L)
    keep_lower_face: dict = field(default_factory=lambda: {"lead": True, "follow": False})
    forehead_extend_iod: float = 0.30            # MediaPipe's oval stops mid-forehead: extend its top (x IOD)
    skin_repair: dict = field(default_factory=lambda: {"lead": False, "follow": True})   # hair baked onto the face
    refill_base: dict = field(default_factory=lambda: {"lead": False, "follow": True})   # portrait over its own skin
                                                 # extrapolation instead of the bake inside the face (refill_face_base)
    refill_ramp: float = 12.0                    # texels: the refilled face fades into the bake (hairline, sideburns)
    repair_maha: float = 3.0                     # bake texel != skin beyond this (Lab, 0.3 L) and closer to hair
    repair_feather: int = 8                      # texels
    mouth_interior: bool = True                  # paint the lips' inner surfaces dark (no orange slit when it opens)
    symmetrize_head: dict = field(default_factory=lambda: {"lead": True, "follow": True})    # L/R low frequencies
    lip_L_frac: float = 0.5                      # video look: fraction of the controlled lip lightness change used
    lip_colour: dict = field(default_factory=lambda: {"lead": False, "follow": True})   # portrait look: the mesh's
                                                 # lips take the portrait's lip-vs-cheek colour (lead: off - the
                                                 # moustache overlaps his lips and the portrait colour already holds)
    lip_keep_detail: float = 0.6                 # ... keeping this fraction of their own variation
    smooth_repair: dict = field(default_factory=lambda: {"lead": False, "follow": True})   # gradient-domain clean-up of
    smooth_scalp: dict = field(default_factory=lambda: {"lead": True, "follow": False})    # the repair / the scalp band
    smooth_band_alpha: float = 0.9               # follow: the clean-up also covers the portrait border up to this alpha
    grad_clip: float = 0.025                     # per-texel log-colour step kept by gradient_smooth (larger = a seam)
    scalp_clip: float = 0.012                    # ... on the lead's scalp band (the bake seam is soft: a few texels wide)
    streaks: dict = field(default_factory=lambda: {"lead": True, "follow": False})         # light lines in the beard
    pad_islands: bool = True                     # grow the UV islands into the uncovered texels (no pale lip line)
    feather_texels: int = 28                     # alpha ramp at the portrait region's border (~1 cm)
    gain_clip: tuple = (0.6, 1.6)
    pyramid_levels: int = 5
    raster_scale: int = 4


# ------------------------------------------------------------------------------------------------ inputs


def portrait_path(take: Take, role: str) -> Path:
    d = take.cfg.get("dancers", {}).get(role, {})
    src = take.sources[0].path.parent          # the take's source_dir (resolved by Take.load)
    if d.get("portrait"):
        p = Path(d["portrait"])
        return p if p.is_absolute() else (src / p).resolve()
    return src / f"{str(d.get('name', role)).lower()}.jpg"


def portrait_landmarks(take: Take, role: str, force: bool = False) -> dict:
    out = take.work / "face" / f"portrait_{role}.npz"
    img = portrait_path(take, role)
    if force or not out.exists() or out.stat().st_mtime < img.stat().st_mtime:
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(FACE_VENV_PY), "-m", "dancecap.face_worker", "image", str(img), str(out), "--upscale", "3"],
                       cwd=str(WORKSPACE / "dancecap"), check=True, capture_output=True)
    with np.load(out) as z:
        d = {k: z[k] for k in z.files}
    if not bool(d["mp_ok"]):
        raise RuntimeError(f"MediaPipe found no face in {img}")
    return d


# ------------------------------------------------------------------------------------------------ fit


class HeadModel:
    """Rest-pose SMPL-X (locked head, the dancer's betas) with jaw + expression free."""

    def __init__(self, betas: np.ndarray, n_expr: int):
        import smplx
        import torch
        from .mamma_dyn import SMPLX_MODELS
        self.torch = torch
        nb = len(betas)
        self.m = smplx.create(str(SMPLX_MODELS), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                              flat_hand_mean=True, num_betas=nb, num_expression_coeffs=n_expr, batch_size=1)
        self.betas = torch.tensor(np.asarray(betas, np.float32))[None]
        self.n_expr = n_expr
        self.faces = np.asarray(self.m.faces, np.int64)

    def verts(self, expr, jaw):
        o = self.m(betas=self.betas, expression=expr[None], jaw_pose=jaw[None], return_verts=True)
        return o.vertices[0]

    def landmark_fn(self, emb: LandmarkEmbedding):
        """Fast landmarks(expr, jaw) (52,3) for the fit: shaped template + expression blendshapes on the landmark
        triangles, jaw = rotation of the jaw-skinned part about joint 22 (pose correctives ignored; the final mesh
        comes from the full model)."""
        torch = self.torch
        m = self.m
        with torch.no_grad():
            vs = m.v_template + torch.einsum("vdk,k->vd", m.shapedirs, self.betas[0])
            J22 = (m.J_regressor[22][:, None] * vs).sum(0)
        ids = torch.as_tensor(emb.faces.reshape(-1))
        vs_l = vs[ids].reshape(52, 3, 3)
        E_l = m.expr_dirs[ids].reshape(52, 3, 3, -1)
        w_l = m.lbs_weights[ids, 22].reshape(52, 3, 1)
        bary = torch.tensor(emb.bary, dtype=torch.float32)

        def fn(expr, jaw):
            V = vs_l + torch.einsum("lkdn,n->lkd", E_l, expr)
            R = _rotmat(torch, jaw)
            V = V + w_l * ((V - J22) @ R.T - (V - J22))
            return (V * bary[..., None]).sum(1)
        return fn


from scipy.spatial.transform import Rotation  # noqa: E402


def _rotmat(torch, rv):
    th = rv.norm() + 1e-9
    k = rv / th
    Kx = torch.zeros((3, 3), dtype=rv.dtype)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, dtype=rv.dtype) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def project_weak(torch, P, R, s, t2):
    """Model points (N,3) (y up, +z towards the camera) -> image px (N,2) (y down) and depth (N,) (larger = farther)."""
    Q = P @ R.T
    px = torch.stack([Q[:, 0], -Q[:, 1]], 1) * s + t2
    return px, 10.0 - Q[:, 2]


def ortho_init(P3: np.ndarray, lmk2d: np.ndarray, emb: LandmarkEmbedding) -> tuple[np.ndarray, float, np.ndarray]:
    """Scaled-orthographic camera from the rigid landmarks (brows, nose, eyes): affine least squares, then the
    nearest scaled rotation (y flipped to the image's y-down)."""
    rigid = emb.index(list(range(17, 48)))
    X = P3[rigid] - P3[rigid].mean(0)
    x = lmk2d[rigid] - lmk2d[rigid].mean(0)
    x = x * np.array([1.0, -1.0])                      # image y down -> model y up
    M = np.linalg.lstsq(X, x, rcond=None)[0].T         # (2,3)
    U, Sv, Vt = np.linalg.svd(M, full_matrices=False)
    s = float(Sv.mean())
    r12 = U @ Vt
    r3 = np.cross(r12[0], r12[1])
    R = np.stack([r12[0], r12[1], r3])
    if np.linalg.det(R) < 0:
        R[2] *= -1
    t = lmk2d[rigid].mean(0) - s * np.array([1.0, -1.0]) * (P3[rigid].mean(0) @ R.T)[:2]
    return R, s, t


def fit_portrait(head: HeadModel, emb: LandmarkEmbedding, lmk2d: np.ndarray, cfg: FaceTexConfig) -> dict:
    """lmk2d (52,2) portrait landmarks in IBUG52 order -> fitted rotation/scale/offset/jaw/expression + residuals."""
    torch = head.torch
    L = torch.tensor(lmk2d, dtype=torch.float32)
    iod = float(np.linalg.norm(lmk2d[emb.index([36])[0]] - lmk2d[emb.index([45])[0]]))
    fb = torch.as_tensor(emb.faces)
    bb = torch.tensor(emb.bary, dtype=torch.float32)
    w = torch.ones(52)
    w[torch.as_tensor(emb.index(range(36, 48)))] = 2.0     # eyes
    w[torch.as_tensor(emb.index(range(48, 68)))] = 1.5     # mouth
    w[0] = 0.5                                              # chin (contour)

    def lmk3d(V):
        return (V[fb] * bb[..., None]).sum(1)

    best = None
    lfn = head.landmark_fn(emb)
    with torch.no_grad():
        P0 = lfn(torch.zeros(head.n_expr), torch.zeros(3))
    R0, s_init, t_init = ortho_init(P0.numpy(), lmk2d, emb)
    rv_init = Rotation.from_matrix(R0).as_rotvec()
    for drv in ((0, 0, 0), (0, 0.3, 0), (0, -0.3, 0)):
        rv = torch.tensor(rv_init + np.array(drv), dtype=torch.float32, requires_grad=True)
        ls = torch.tensor(np.log(s_init), dtype=torch.float32, requires_grad=True)
        t2 = torch.tensor(t_init, dtype=torch.float32, requires_grad=True)
        expr = torch.zeros(head.n_expr, requires_grad=True)
        jaw = torch.zeros(3, requires_grad=True)
        for stage, (params, iters, lr) in enumerate((([rv, ls, t2], 150, 0.01), ([rv, ls, t2, expr, jaw], 400, 0.01))):
            opt = torch.optim.Adam(params, lr=lr)
            for _ in range(iters):
                opt.zero_grad()
                Pl = lfn(expr, jaw) if stage else P0
                px, _ = project_weak(torch, Pl, _rotmat(torch, rv), ls.exp(), t2)
                e = ((px - L) / iod).pow(2).sum(1)
                loss = (w * e).sum() / w.sum() + cfg.expr_prior * expr.pow(2).sum() + \
                    cfg.jaw_prior * (jaw[1:].pow(2).sum() * 10 + jaw[0].pow(2))
                loss.backward()
                opt.step()
        if best is None or loss.item() < best[0]:
            best = (loss.item(), rv.detach().clone(), ls.detach().clone(), t2.detach().clone(), expr.detach().clone(),
                    jaw.detach().clone())
    _, rv, ls, t2, expr, jaw = best
    with torch.no_grad():
        V = head.verts(expr, jaw)
        R = _rotmat(torch, rv)
        px, _ = project_weak(torch, lmk3d(V), R, ls.exp(), t2)
        fast_err = float((lfn(expr, jaw) - lmk3d(V)).norm(dim=1).max())
    res = lmk2d - px.numpy()
    return {"R": R.numpy(), "s": float(ls.exp()), "t2": t2.numpy(), "expr": expr.numpy(), "jaw": jaw.numpy(),
            "verts": V.numpy(), "lmk_proj": px.numpy(), "residual": res, "iod_px": iod,
            "rms_px": float(np.sqrt((res ** 2).sum(1).mean())), "rms_iod": float(np.sqrt((res ** 2).sum(1).mean()) / iod),
            "yaw_deg": float(np.degrees(np.arcsin(np.clip(R[0, 2].item(), -1, 1)))),
            "fast_model_err_mm": fast_err * 1000}


# ------------------------------------------------------------------------------------------------ portrait mask


def _poly_mask(shape, pts, dilate: int = 0):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    if dilate > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    return m.astype(bool)


def _hull(pts):
    return cv2.convexHull(np.round(pts).astype(np.float32)).reshape(-1, 2)


def extend_forehead(mp: np.ndarray, extend_iod: float) -> np.ndarray:
    """MediaPipe face-oval points with the part above the eyes pushed up the face by up to extend_iod x IOD (the oval
    ends mid-forehead; the hairline is ~0.4-0.5 IOD higher). Smooth: 0 at the eye line, full at the oval top."""
    pts = mp[MP_OVAL].astype(np.float64).copy()
    if extend_iod <= 0:
        return pts
    iod = float(np.linalg.norm(mp[33] - mp[263]))
    up = mp[10] - mp[152]
    up /= max(np.linalg.norm(up), 1e-9)
    eye = 0.5 * (mp[33] + mp[263])
    h = (pts - eye) @ up
    htop = max(float(h.max()), 1e-6)
    wgt = np.clip(h / htop, 0, 1) ** 1.5
    return pts + (extend_iod * iod * wgt)[:, None] * up[None]


def portrait_labels(shape, mp: np.ndarray) -> np.ndarray:
    """uint8 label image of the portrait: LAB_LIPS (outer minus inner lip), LAB_BEARD (chin / jaw region below the
    lower lip), LAB_CHEEK (cheek discs used as the skin reference by face_look)."""
    lab = np.zeros(shape[:2], np.uint8)
    iod = float(np.linalg.norm(mp[33] - mp[263]))
    lips = _poly_mask(shape, mp[MP_LIPS_OUTER]) & ~_poly_mask(shape, mp[MP_LIPS_INNER])
    chin = _poly_mask(shape, mp[MP_CHIN_REGION]) & ~_poly_mask(shape, mp[MP_LIPS_OUTER])
    cheek = np.zeros(shape[:2], np.uint8)
    for k in MP_CHEEKS:
        cv2.circle(cheek, tuple(np.round(mp[k]).astype(int)), max(1, int(round(0.12 * iod))), 1, -1)
    lab[lips] |= LAB_LIPS
    lab[chin] |= LAB_BEARD
    lab[cheek.astype(bool)] |= LAB_CHEEK
    return lab


def portrait_mask(img: np.ndarray, mp: np.ndarray, role: str, cfg: FaceTexConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> (mask float32 0..1, portrait with the inner mouth inpainted, info). mp (478,2) px."""
    H, W = img.shape[:2]
    iod = float(np.linalg.norm(mp[33] - mp[263]))
    oval_pts = extend_forehead(mp, cfg.forehead_extend_iod)
    oval = _poly_mask(img.shape, oval_pts)
    keep = np.zeros((H, W), bool)
    for idx, dil in ((MP_EYE_R, 0.06), (MP_EYE_L, 0.06), (MP_BROW_R, 0.05), (MP_BROW_L, 0.05), (MP_LIPS_OUTER, 0.03)):
        keep |= _poly_mask(img.shape, mp[idx], int(round(dil * iod)))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    feat = np.stack([lab[..., 0] * 0.3, lab[..., 1], lab[..., 2]], -1)   # shading moves L most: down-weight it
    seeds = np.zeros((H, W), np.uint8)
    for k in MP_SKIN_SEEDS:
        cv2.circle(seeds, tuple(np.round(mp[k]).astype(int)), max(2, int(0.07 * iod)), 1, -1)
    seeds = seeds.astype(bool) & oval & ~keep

    def gauss(X):
        mu, C = X.mean(0), np.cov(X.T) + np.eye(3) * 4.0
        Ci = np.linalg.inv(C)
        d = feat - mu
        m2 = np.einsum("hwi,ij,hwj->hw", d, Ci, d)
        return np.sqrt(m2), -0.5 * m2 - 0.5 * np.log(np.linalg.det(C))
    maha, ll_skin = gauss(feat[seeds])
    # occluder model: pixels in a ring outside the oval that are clearly not skin (hair, arm shadow, background)
    ring = _poly_mask(img.shape, oval_pts, int(0.45 * iod)) & ~_poly_mask(img.shape, oval_pts, int(0.08 * iod))
    occ = ring & (maha > cfg.skin_maha + 1)
    if occ.sum() > 30:
        _, ll_occ = gauss(feat[occ])
        skin = (maha < cfg.skin_maha) | ((ll_skin > ll_occ) & (maha < 2 * cfg.skin_maha))
    else:
        skin = maha < cfg.skin_maha
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    m = oval & (skin | keep)
    if cfg.keep_lower_face.get(role):
        m |= oval & _poly_mask(img.shape, mp[MP_LOWER_FACE])
    # largest connected component + hole fill
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n > 1:
        m = lab_cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ff = (~m).astype(np.uint8)
    cv2.floodFill(ff, None, (0, 0), 0)
    m = m | ff.astype(bool)
    # inner mouth (teeth / dark gap) -> inpaint from the lips so no teeth are painted onto the lip surface
    inner = _poly_mask(img.shape, mp[MP_LIPS_INNER], 1)
    img2 = cv2.inpaint(img, inner.astype(np.uint8), 3, cv2.INPAINT_TELEA) if inner.any() else img.copy()
    # soft edge (1.5 px at portrait resolution)
    mf = cv2.GaussianBlur(cv2.erode(m.astype(np.float32), np.ones((3, 3), np.uint8)), (0, 0), 1.2)
    info = {"oval_px": int(oval.sum()), "mask_px": int(m.sum()), "kept_fraction_of_oval": round(float(m[oval].mean()), 3),
            "skin_seed_px": int(seeds.sum()), "occluder_px": int(occ.sum()), "inner_mouth_px": int(inner.sum()), "iod_px": round(iod, 1)}
    return mf, img2, info


# ------------------------------------------------------------------------------------------------ projection


def _tps(src: np.ndarray, vals: np.ndarray, smooth: float):
    from scipy.interpolate import RBFInterpolator
    return RBFInterpolator(src, vals, kernel="thin_plate_spline", smoothing=smooth)


def _bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """img (H,W[,C]) float, xy (N,2) -> (N[,C]) with border clamp."""
    n, cols = len(xy), 16384                     # remap maps must stay below SHRT_MAX columns
    rows = -(-n // cols)
    pad = np.zeros((rows * cols, 2), np.float32)
    pad[:n] = xy
    m = cv2.remap(img.astype(np.float32), pad[:, 0].reshape(rows, cols), pad[:, 1].reshape(rows, cols),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return m.reshape((rows * cols,) + m.shape[2:])[:n]


def project_portrait(atlas, head: HeadModel, fit: dict, img: np.ndarray, mask: np.ndarray, emb: LandmarkEmbedding,
                     cfg: FaceTexConfig, labels: np.ndarray | None = None) -> dict:
    """Per head texel: portrait colour (sRGB float), weight, sample position."""
    import torch

    from .avatar_raster import rasterize
    from .avatar_uv import vertex_normals
    H, W = img.shape[:2]
    head_t = np.flatnonzero((atlas.region & 1) > 0)
    V = fit["verts"].astype(np.float64)
    R, s, t2 = fit["R"], fit["s"], fit["t2"]
    Q = V @ R.T
    pxv = np.stack([Q[:, 0], -Q[:, 1]], 1) * s + t2
    zv = 10.0 - Q[:, 2]
    S = cfg.raster_scale
    # the eyeballs are separate spheres that bulge through the lid openings: they must not hide the lids/lashes
    from .avatar_uv import vertex_regions
    eye_v = vertex_regions(np.load(smplx_model_npz(), allow_pickle=True))["eye"]
    occluders = head.faces[~eye_v[head.faces].all(1)]
    zb, fid = rasterize(torch, torch.tensor(pxv * S, dtype=torch.float32), torch.tensor(zv, dtype=torch.float32),
                        torch.as_tensor(occluders), H * S, W * S)
    zb = zb.numpy()
    tri = atlas.faces[atlas.face[head_t]]
    b = atlas.bary[head_t]
    P = np.einsum("nkd,nk->nd", V[tri], b)
    N = np.einsum("nkd,nk->nd", vertex_normals(V, head.faces)[tri], b)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    Qp = P @ R.T
    px = np.stack([Qp[:, 0], -Qp[:, 1]], 1) * s + t2
    z = 10.0 - Qp[:, 2]
    cos = (N @ R.T)[:, 2]
    # visibility: texel depth vs the fitted mesh's z-buffer at its projected pixel (6 mm tolerance)
    ui = np.clip(np.round(px[:, 0] * S).astype(int), 0, W * S - 1)
    vi = np.clip(np.round(px[:, 1] * S).astype(int), 0, H * S - 1)
    zmin = zb[vi, ui]
    vis = (zmin < 1e8) & (z <= zmin + 0.006)
    # residual spline: sample = projection + d(projection)
    tps = _tps(fit["lmk_proj"], fit["residual"], cfg.tps_smooth)
    samp = px + tps(px)
    inb = (samp[:, 0] >= 0) & (samp[:, 0] <= W - 1) & (samp[:, 1] >= 0) & (samp[:, 1] <= H - 1)
    face_bit = (atlas.region[head_t] & 2) > 0
    rgb = _bilinear(img[:, :, ::-1].astype(np.float32) / 255.0, samp)
    mk = _bilinear(mask, samp)
    facing = np.clip((cos - cfg.cos_lo) / (cfg.cos_hi - cfg.cos_lo), 0, 1)
    wgt = facing * mk * vis * inb * face_bit
    out = {"texels": head_t, "rgb": rgb.astype(np.float32), "w": wgt.astype(np.float32), "samp": samp.astype(np.float32),
           "cos": cos.astype(np.float32), "vis": vis, "texels_weighted": int((wgt > 0.05).sum())}
    if labels is not None:
        ui = np.clip(np.round(samp[:, 0]).astype(int), 0, W - 1)
        vi = np.clip(np.round(samp[:, 1]).astype(int), 0, H - 1)
        lab = np.where(vis & inb & (cos > 0.15) & face_bit, labels[vi, ui], 0).astype(np.uint8)
        # turned-away texels take the label of their mirror texel (lips / beard / cheeks are left-right symmetric)
        full = np.zeros(atlas.n, np.uint8)
        full[head_t] = lab
        ok_own = np.zeros(atlas.n, bool)
        ok_own[head_t] = vis & inb & (cos > 0.15)
        mir = atlas.mirror[head_t]
        use_m = ~ok_own[head_t] & (mir >= 0)
        lab[use_m] = full[mir[use_m]]
        out["labels"] = lab
    return out


def mirror_fill(atlas, proj: dict, cfg: FaceTexConfig, role: str = "lead") -> dict:
    """Texels whose own weight is low take the mirrored texel's portrait colour (weight x mirror_weight[role]); colour
    AND weight hand over smoothly (UV blur of the mirror share): a step in either drew a line along the nose."""
    n = atlas.n
    w_all = np.zeros(n, np.float32)
    c_all = np.zeros((n, 3), np.float32)
    w_all[proj["texels"]] = proj["w"]
    c_all[proj["texels"]] = proj["rgb"]
    mir = atlas.mirror[proj["texels"]]
    ok = mir >= 0
    wm = np.zeros(len(mir), np.float32)
    mw = cfg.mirror_weight.get(role, 0.7) if isinstance(cfg.mirror_weight, dict) else float(cfg.mirror_weight)
    wm[ok] = w_all[mir[ok]] * mw
    cm = np.zeros((len(mir), 3), np.float32)
    cm[ok] = c_all[mir[ok]]
    w0 = proj["w"]
    # smooth hand-over from the own sample to the mirrored one (a hard switch left a seam down the nose / forehead
    # where the portrait's lighting differs between the two sides): mirror share 0 while wm <= 0.6 w0, 1 from 1.4 w0
    ratio = wm / np.maximum(w0, 1e-6)
    t = np.clip((ratio - 0.6) / 0.8, 0, 1)
    t = t * t * (3 - 2 * t)
    t = np.where(wm > 0.02, t, 0.0)
    if cfg.mirror_blend_sigma > 0:                 # spatially smooth hand-over (normalised blur over the head texels)
        S = atlas.size
        yx = atlas.texel_yx[proj["texels"]]
        ti = np.zeros((S, S), np.float32)
        wi = np.zeros((S, S), np.float32)
        ti[yx[:, 0], yx[:, 1]] = t
        wi[yx[:, 0], yx[:, 1]] = 1
        sb = cv2.GaussianBlur(ti, (0, 0), cfg.mirror_blend_sigma) / np.maximum(
            cv2.GaussianBlur(wi, (0, 0), cfg.mirror_blend_sigma), 1e-6)
        t = np.where(wm > 0.02, np.clip(sb[yx[:, 0], yx[:, 1]], 0, 1), 0.0)
    rgb = (1 - t[:, None]) * proj["rgb"] + t[:, None] * cm
    out = dict(proj)
    out["rgb"] = rgb.astype(np.float32)
    out["w"] = ((1 - t) * w0 + t * wm).astype(np.float32)
    out["mirrored"] = int((t > 0.5).sum())
    return out


# ------------------------------------------------------------------------------------------------ composite


def _lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def _fill_masked(img: np.ndarray, m: np.ndarray, iters: int = 6) -> np.ndarray:
    """Push-pull fill of img outside mask m (both HxW[xC]) so pyramids do not see the background."""
    if m.all():
        return img
    if iters == 0 or min(m.shape) < 4:
        mean = img[m].mean(0) if m.any() else np.zeros(img.shape[2:], img.dtype)
        out = img.copy()
        out[~m] = mean
        return out
    w = m.astype(np.float32)
    small = cv2.resize(img * w[..., None], (max(1, img.shape[1] // 2), max(1, img.shape[0] // 2)),
                       interpolation=cv2.INTER_AREA)
    ws = cv2.resize(w, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    small = small / np.maximum(ws[..., None], 1e-6)
    small = _fill_masked(small, ws > 1e-3, iters - 1)
    up = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.where(m[..., None], img, up)


def _laplacian_blend(a: np.ndarray, b: np.ndarray, alpha: np.ndarray, levels: int) -> np.ndarray:
    ga, gb, gm = [a], [b], [alpha]
    for _ in range(levels):
        ga.append(cv2.pyrDown(ga[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        gm.append(cv2.pyrDown(gm[-1]))
    out = ga[-1] * (1 - gm[-1][..., None]) + gb[-1] * gm[-1][..., None]
    for k in range(levels - 1, -1, -1):
        size = (ga[k].shape[1], ga[k].shape[0])
        la = ga[k] - cv2.pyrUp(ga[k + 1], dstsize=size)
        lb = gb[k] - cv2.pyrUp(gb[k + 1], dstsize=size)
        out = cv2.pyrUp(out, dstsize=size) + la * (1 - gm[k][..., None]) + lb * gm[k][..., None]
    return out


def composite(atlas, bake_bgr: np.ndarray, proj: dict, cfg: FaceTexConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Colour-transfer the portrait texels to the bake's lighting and blend -> (albedo BGR uint8, layer RGBA, info)."""
    S = atlas.size
    yx = atlas.texel_yx[proj["texels"]]
    y0, x0 = yx.min(0)
    y1, x1 = yx.max(0) + 1
    hh, ww = y1 - y0, x1 - x0
    ly, lx = yx[:, 0] - y0, yx[:, 1] - x0
    bake = bake_bgr[y0:y1, x0:x1, ::-1].astype(np.float32) / 255.0
    island = np.zeros((hh, ww), bool)
    island[ly, lx] = True
    por = np.zeros((hh, ww, 3), np.float32)
    wim = np.zeros((hh, ww), np.float32)
    por[ly, lx] = proj["rgb"]
    wim[ly, lx] = proj["w"]
    have = wim > 0.05
    # trusted texels for the colour transfer: portrait weight high and the bake looks like skin there
    bl, pl = _lin(bake), _lin(por)
    trust = (wim > 0.6) & island
    bl_s = cv2.GaussianBlur(bl, (0, 0), 6)
    pl_s = cv2.GaussianBlur(np.where(have[..., None], pl, 0), (0, 0), 6) / \
        np.maximum(cv2.GaussianBlur(have.astype(np.float32), (0, 0), 6)[..., None], 1e-6)
    Yb = bl_s @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    med, mad = np.median(Yb[trust]), np.median(np.abs(Yb[trust] - np.median(Yb[trust]))) + 1e-6
    chroma = bl_s / np.maximum(bl_s.sum(-1, keepdims=True), 1e-6)
    cmed = np.median(chroma[trust], 0)
    trust &= (np.abs(Yb - med) < 2.5 * 1.4826 * mad) & (np.abs(chroma - cmed).sum(-1) < 0.06)
    ratio = np.log(np.maximum(bl_s, 1e-4)) - np.log(np.maximum(pl_s, 1e-4))
    g_global = np.median(ratio[trust], 0)
    # smooth quadratic gain field per channel (robust IRLS) on the trusted texels
    yy, xx = np.nonzero(trust)
    u, v = (xx - ww / 2) / ww, (yy - hh / 2) / hh
    A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)
    coef = []
    for c in range(3):
        r = ratio[yy, xx, c] - g_global[c]
        wt = np.ones_like(r)
        cf = np.zeros(6)
        for _ in range(5):
            cf = np.linalg.lstsq(A * wt[:, None], r * wt, rcond=None)[0]
            e = r - A @ cf
            sc = 1.4826 * np.median(np.abs(e)) + 1e-6
            wt = 1 / np.maximum(1, np.abs(e) / (1.5 * sc))
        coef.append(cf)
    Y, X = np.mgrid[0:hh, 0:ww]
    U, Vv = (X - ww / 2) / ww, (Y - hh / 2) / hh
    AA = np.stack([np.ones_like(U), U, Vv, U * U, U * Vv, Vv * Vv], -1)
    field = np.stack([AA @ coef[c] for c in range(3)], -1)
    gain = (np.exp(g_global) * np.clip(np.exp(field), *cfg.gain_clip)).astype(np.float32)   # clip relative to global
    por_c = _srgb(pl * gain).astype(np.float32)
    # alpha: weight x feather from the border of the usable region (distance in texels)
    # feather from the OUTER border only: the eye openings and the mouth slit are holes in the portrait region, and
    # feathering from them faded the lids and lips back to the bake (alpha ~0.2 on the lips before 2026-10-06)
    outside = (~have).astype(np.uint8)
    pad = cv2.copyMakeBorder(outside, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    cv2.floodFill(pad, None, (0, 0), 2)
    holes = pad[1:-1, 1:-1] == 1
    dist = cv2.distanceTransform((have | holes).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(wim, 0, 1) * np.clip(dist / cfg.feather_texels, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 2.0) * island
    a_f = _fill_masked(bake, island)
    b_f = _fill_masked(np.where(have[..., None], por_c, bake), island | have)
    blended = _laplacian_blend(a_f, b_f, alpha.astype(np.float32), cfg.pyramid_levels)
    blended = np.clip(blended, 0, 1)
    out = bake_bgr.copy()
    sub = out[y0:y1, x0:x1]
    sub[island] = (blended[island][:, ::-1] * 255 + 0.5).astype(np.uint8)
    layer = np.zeros((S, S, 4), np.uint8)
    lay = layer[y0:y1, x0:x1]
    lay[..., :3] = (np.clip(por_c, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8)
    lay[..., 3] = (alpha * 255 + 0.5).astype(np.uint8)
    info = {"trusted_texels": int(trust.sum()), "gain_global_lin": np.round(np.exp(g_global), 3).tolist(),
            "gain_field_range": [np.round(gain[have].min(0), 3).tolist(), np.round(gain[have].max(0), 3).tolist()],
            "alpha_texels_gt_0.5": int((alpha > 0.5).sum()), "bbox_yx": [int(y0), int(x0), int(y1), int(x1)]}
    return out, layer, info


# ------------------------------------------------------------------------------------------------ post-processing


def _texel_values(atlas, img_bgr: np.ndarray) -> np.ndarray:
    """(N,3) sRGB float 0..1 of every atlas texel."""
    return img_bgr[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1], ::-1].astype(np.float32) / 255.0


def _lab(rgb01: np.ndarray) -> np.ndarray:
    """sRGB 0..1 (..., 3) -> CIE Lab (L 0..100, a, b ~ -128..127), OpenCV float convention."""
    sh = rgb01.shape
    return cv2.cvtColor(np.ascontiguousarray(rgb01.reshape(-1, 1, 3), np.float32), cv2.COLOR_RGB2Lab).reshape(sh)


def _rgb(lab: np.ndarray) -> np.ndarray:
    sh = lab.shape
    return np.clip(cv2.cvtColor(np.ascontiguousarray(lab.reshape(-1, 1, 3), np.float32), cv2.COLOR_Lab2RGB), 0, 1).reshape(sh)


def _gauss_stats(X: np.ndarray):
    mu = np.median(X, 0)
    C = np.cov((X - mu).T) + np.eye(X.shape[1]) * 2.0
    return mu, np.linalg.inv(C), float(np.log(np.linalg.det(C)))


def _maha(X, st):
    d = X - st[0]
    return np.sqrt(np.einsum("ni,ij,nj->n", d, st[1], d))


def _uv_mask_ops(atlas, sel: np.ndarray, close: int = 3, open_: int = 2, dilate: int = 0) -> np.ndarray:
    """Morphological clean-up of a texel selection in UV image space -> texel bool."""
    S = atlas.size
    im = np.zeros((S, S), np.uint8)
    im[atlas.texel_yx[sel, 0], atlas.texel_yx[sel, 1]] = 1
    if close:
        im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1,) * 2))
    if open_:
        im = cv2.morphologyEx(im, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_ + 1,) * 2))
    if dilate:
        im = cv2.dilate(im, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2))
    return im[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]] > 0


def refill_face_base(atlas, alb_bgr: np.ndarray, layer: np.ndarray, alpha_t: np.ndarray, exclude: np.ndarray,
                     cfg: FaceTexConfig) -> tuple[np.ndarray, dict]:
    """Where the bake under the face is unusable (the follow: her hair and ghosting were baked across it), the portrait
    is blended over a smooth extrapolation of its own skin instead of over the bake: inside the face region (the
    smoothed face bit) every texel becomes alpha x portrait + (1 - alpha) x fill, the fill being a push-pull
    extension of the portrait's confident skin texels (alpha > 0.9, no eyes / brows / lips / nostrils). The region's
    border ramps over cfg.refill_ramp texels; outside it the composite is unchanged. -> (albedo, info)."""
    S = atlas.size
    por = _lin(layer[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1], :3][:, ::-1].astype(np.float32) / 255.0)
    head = ((atlas.region & 1) > 0) & ((atlas.region & 4) == 0)
    fm = np.zeros((S, S), np.float32)
    fbit = (atlas.region & 2) > 0
    fm[atlas.texel_yx[fbit, 0], atlas.texel_yx[fbit, 1]] = 1
    region_s = cv2.GaussianBlur(fm, (0, 0), 20.0 * S / 2048)
    face = (region_s[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]] > 0.5) & head
    src = face & (alpha_t > 0.9) & ~exclude
    if src.sum() < 500:
        return alb_bgr, {"skipped": "too few confident portrait skin texels"}
    img = np.zeros((S, S, 3), np.float32)
    have = np.zeros((S, S), bool)
    img[atlas.texel_yx[src, 0], atlas.texel_yx[src, 1]] = por[src]
    have[atlas.texel_yx[src, 0], atlas.texel_yx[src, 1]] = True
    yx = atlas.texel_yx[head]
    y0, x0 = yx.min(0)
    y1, x1 = yx.max(0) + 1
    sub = _fill_masked(img[y0:y1, x0:x1], have[y0:y1, x0:x1], iters=8)
    sub = cv2.GaussianBlur(sub, (0, 0), 3.0)
    fill = np.zeros((atlas.n, 3), np.float32)
    fill[head] = sub[atlas.texel_yx[head, 0] - y0, atlas.texel_yx[head, 1] - x0]
    fm2 = np.zeros((S, S), np.uint8)
    fm2[atlas.texel_yx[face, 0], atlas.texel_yx[face, 1]] = 1
    r = np.clip(cv2.distanceTransform(fm2, cv2.DIST_L2, 5) / max(cfg.refill_ramp, 1.0), 0, 1)[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]] * face
    a = np.clip(alpha_t, 0, 1)[:, None]
    lin = _lin(_texel_values(atlas, alb_bgr))
    new = a * por + (1 - a) * fill
    out_lin = lin * (1 - r[:, None]) + new * r[:, None]
    sel = (r > 1e-3) & (alpha_t < 0.999)
    out = alb_bgr.copy()
    out[atlas.texel_yx[sel, 0], atlas.texel_yx[sel, 1]] = (_srgb(out_lin[sel])[:, ::-1] * 255 + 0.5).astype(np.uint8)
    return out, {"texels": int(sel.sum()), "source_texels": int(src.sum()),
                 "bake_share_removed_median": round(float(np.median((1 - alpha_t[sel]) * r[sel])), 3)}


def repair_face_skin(atlas, alb_bgr: np.ndarray, alpha_t: np.ndarray, cfg: FaceTexConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Hair painted onto the face by the video bake (long hair sweeping across it): face-region texels the portrait
    does not cover (alpha < 0.5) whose colour is far from the face's skin (Lab Mahalanobis, L down-weighted) and closer
    to the scalp's hair colour are replaced by the mirrored texel when that one is skin, else by a smooth push-pull
    skin fill, with a feathered edge. alpha_t: (N,) portrait alpha per atlas texel. -> (albedo, repaired (N,) bool,
    info)."""
    rgb = _texel_values(atlas, alb_bgr)
    lab = _lab(rgb)
    feat = np.stack([lab[:, 0] * 0.3, lab[:, 1], lab[:, 2]], 1)
    head = (atlas.region & 1) > 0
    eye = (atlas.region & 4) > 0
    # the face bit comes from per-vertex flags: its top edge is a staircase in UV - smooth it
    S = atlas.size
    fm = np.zeros((S, S), np.float32)
    fbit = (atlas.region & 2) > 0
    fm[atlas.texel_yx[fbit, 0], atlas.texel_yx[fbit, 1]] = 1
    face = (cv2.GaussianBlur(fm, (0, 0), 20.0 * S / 2048)[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]] > 0.5) & head
    trusted = face & (alpha_t > 0.8) & ~eye
    Lt = lab[trusted, 0]
    trusted &= (lab[:, 0] >= np.percentile(Lt, 15)) & (lab[:, 0] <= np.percentile(Lt, 97))   # no brows / lashes / glare
    scalp = head & ~fbit & ~eye
    if trusted.sum() < 500 or scalp.sum() < 500:
        return alb_bgr, np.zeros(atlas.n, bool), {"repaired_texels": 0, "skipped": "too few skin / scalp texels"}
    st_s, st_h = _gauss_stats(feat[trusted]), _gauss_stats(feat[scalp])
    m_s, m_h = _maha(feat, st_s), _maha(feat, st_h)
    ll_s, ll_h = -0.5 * m_s ** 2 - 0.5 * st_s[2], -0.5 * m_h ** 2 - 0.5 * st_h[2]
    cand = face & ~eye & (alpha_t < 0.5) & (m_s > cfg.repair_maha) & (ll_h > ll_s)
    cand = _uv_mask_ops(atlas, cand, close=4, open_=2, dilate=3) & face & ~eye & (alpha_t < 0.5)
    good = face & ~eye & ~cand & ((alpha_t > 0.3) | (m_s < cfg.repair_maha))
    lin = _lin(rgb)
    fill = lin.copy()
    mir = atlas.mirror
    use_m = cand & (mir >= 0)
    use_m[use_m] = good[mir[use_m]]
    fill[use_m] = lin[mir[use_m]]
    rest = cand & ~use_m
    if rest.any():
        S = atlas.size
        img = np.zeros((S, S, 3), np.float32)
        have = np.zeros((S, S), bool)
        src = good | use_m
        img[atlas.texel_yx[src, 0], atlas.texel_yx[src, 1]] = fill[src]
        have[atlas.texel_yx[src, 0], atlas.texel_yx[src, 1]] = True
        yx = atlas.texel_yx[head]
        y0, x0 = yx.min(0)
        y1, x1 = yx.max(0) + 1
        sub = _fill_masked(img[y0:y1, x0:x1], have[y0:y1, x0:x1], iters=8)
        ry, rx = atlas.texel_yx[rest, 0] - y0, atlas.texel_yx[rest, 1] - x0
        fill[rest] = sub[ry, rx]
    # feathered blend: distance into the repaired region (texels)
    cm = np.zeros((S, S), np.uint8)
    cm[atlas.texel_yx[cand, 0], atlas.texel_yx[cand, 1]] = 1
    dist = cv2.distanceTransform(cm, cv2.DIST_L2, 5)
    a = np.clip(dist / max(cfg.repair_feather, 1), 0, 1)
    a = cv2.GaussianBlur(a, (0, 0), 1.5)[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]] * face
    out_lin = lin * (1 - a[:, None]) + fill * a[:, None]
    out = alb_bgr.copy()
    sel = a > 1e-3
    out[atlas.texel_yx[sel, 0], atlas.texel_yx[sel, 1]] = (_srgb(out_lin[sel])[:, ::-1] * 255 + 0.5).astype(np.uint8)
    return out, cand, {"repaired_texels": int(cand.sum()), "repaired_fraction_of_face": round(float(cand[face].mean()), 4),
                       "mirrored_texels": int(use_m.sum()), "pushpull_texels": int(rest.sum())}


def symmetrize_head(atlas, alb_bgr: np.ndarray, V: np.ndarray, sigma: float = 15.0, ramp: int = 40) -> tuple[np.ndarray, dict]:
    """Even out the bake's left / right lighting on the head (a hard seam down the forehead and scalp where the
    cameras covering each side disagree): per side of the head (3D x of the rest surface) a normalised low-pass of the
    linear colour; every head texel with a mirror partner is rescaled so its low frequencies become the average of
    both sides (gain clipped 0.6..1.6); detail (high frequencies) is kept. Ramped to 0 over `ramp` texels towards the
    neck. Eyeballs untouched."""
    S = atlas.size
    head = ((atlas.region & 1) > 0) & ((atlas.region & 4) == 0)
    P = atlas.surface(V)
    lin = _lin(_texel_values(atlas, alb_bgr))
    left = P[:, 0] >= 0
    lows = []
    for side in (left, ~left):                    # normalised low-pass of each side, defined over the whole image
        m = head & side
        img = np.zeros((S, S, 3), np.float32)
        wim = np.zeros((S, S), np.float32)
        img[atlas.texel_yx[m, 0], atlas.texel_yx[m, 1]] = lin[m]
        wim[atlas.texel_yx[m, 0], atlas.texel_yx[m, 1]] = 1
        num = cv2.GaussianBlur(img, (0, 0), sigma)
        den = cv2.GaussianBlur(wim, (0, 0), sigma)
        lows.append((num / np.maximum(den[..., None], 1e-6), den))
    at = lambda k, idx: lows[k][0][atlas.texel_yx[idx, 0], atlas.texel_yx[idx, 1]]
    own = np.where(left[:, None], at(0, np.arange(atlas.n)), at(1, np.arange(atlas.n)))
    # the other side's level: at the mirror texel when the mirror is valid and really on the other side, else (texels
    # on / next to the centre line, whose nearest mirror sample can be themselves) the other side's low-pass here
    mir = atlas.mirror
    other_k = np.where(left, 1, 0)
    good_m = head & (mir >= 0)
    good_m[good_m] = head[mir[good_m]] & (left[mir[good_m]] != left[good_m])
    other = np.where(other_k[:, None] == 0, at(0, np.arange(atlas.n)), at(1, np.arange(atlas.n)))
    other[good_m] = np.where(other_k[good_m, None] == 0, at(0, mir[good_m]), at(1, mir[good_m]))
    den_other = np.where(other_k == 0, lows[0][1][atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]],
                         lows[1][1][atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]])
    ok = head & (good_m | (den_other > 0.2))
    gain = np.ones_like(lin)
    gain[ok] = np.clip(0.5 * (own[ok] + other[ok]) / np.maximum(own[ok], 1e-4), 0.6, 1.6)
    hm = np.zeros((S, S), np.uint8)
    hm[atlas.texel_yx[head, 0], atlas.texel_yx[head, 1]] = 1
    dist = cv2.distanceTransform(cv2.copyMakeBorder(hm, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1), cv2.DIST_L2, 5)
    dist = dist[1:-1, 1:-1][atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]]
    w = np.clip(dist / ramp, 0, 1) * ok
    g = 1 + w[:, None] * (gain - 1)
    out = alb_bgr.copy()
    sel = w > 1e-3
    out[atlas.texel_yx[sel, 0], atlas.texel_yx[sel, 1]] = (_srgb(lin[sel] * g[sel])[:, ::-1] * 255 + 0.5).astype(np.uint8)
    lr = lambda a: (float(np.median(a[head & (P[:, 0] >= 0)] @ [0.2126, 0.7152, 0.0722])),
                    float(np.median(a[head & (P[:, 0] < 0)] @ [0.2126, 0.7152, 0.0722])))
    b, a_ = lr(lin), lr(lin * g)
    return out, {"texels": int(sel.sum()), "luma_left_right_before": [round(b[0], 4), round(b[1], 4)],
                 "luma_left_right_after": [round(a_[0], 4), round(a_[1], 4)]}


def mouth_interior_texels(atlas, V: np.ndarray, faces: np.ndarray, emb: LandmarkEmbedding) -> np.ndarray:
    """Texels of the lips' inner surfaces: inside the mouth box of the closed rest head (V) and seen from none of 15
    frontal views (yaw -50..50, pitch -25..25). When the mouth opens (jaw, smile) these show; the bake / portrait have
    no colour for them (the 'orange slit')."""
    import torch

    from .avatar_raster import rasterize
    from .avatar_uv import vertex_regions
    L = emb.points(V)
    i48, i54, i51, i57 = emb.index([48, 54, 51, 57])
    P = atlas.surface(V)
    head = (atlas.region & 1) > 0
    xlo, xhi = sorted((L[i48, 0], L[i54, 0]))
    box = head & (P[:, 0] > xlo - 0.004) & (P[:, 0] < xhi + 0.004) & (P[:, 1] > L[i57, 1] - 0.004) & \
        (P[:, 1] < L[i51, 1] + 0.004) & (P[:, 2] > L[i51, 2] - 0.03)
    eye_v = vertex_regions(np.load(smplx_model_npz(), allow_pickle=True))["eye"]
    occ = torch.as_tensor(faces[~eye_v[faces].all(1)])
    center = L[[i48, i54, i51, i57]].mean(0)
    size, s = 320, 320 / 0.09                          # 9 cm window around the mouth
    seen = np.zeros(atlas.n, bool)
    idx = np.flatnonzero(box)
    for yaw in (-50, -25, 0, 25, 50):
        for pitch in (-25, 0, 25):
            a, b = np.radians(yaw), np.radians(pitch)
            dv = np.array([np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)])     # towards the camera
            xv = np.cross([0.0, 1.0, 0.0], dv)
            xv /= np.linalg.norm(xv)
            R = np.stack([xv, np.cross(dv, xv), dv])                                         # model -> view
            Q = (V - center) @ R.T
            px = np.stack([Q[:, 0], -Q[:, 1]], 1) * s + size / 2
            z = 10.0 - Q[:, 2]
            zb, _ = rasterize(torch, torch.tensor(px, dtype=torch.float32), torch.tensor(z, dtype=torch.float32), occ,
                              size, size)
            zb = zb.numpy()
            Qt = (P[idx] - center) @ R.T
            u = np.round(Qt[:, 0] * s + size / 2).astype(int)
            v = np.round(-Qt[:, 1] * s + size / 2).astype(int)
            ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
            zt = 10.0 - Qt[:, 2]
            vis = np.zeros(len(idx), bool)
            vis[ok] = zt[ok] <= zb[v[ok], u[ok]] + 0.0007
            seen[idx[vis]] = True
    return box & ~seen


def paint_mouth_interior(atlas, alb_bgr: np.ndarray, interior: np.ndarray,
                         slit: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Inner lip surfaces: the median colour of the lip ring around them, darkening with depth (linear x0.55 at the
    rim -> x0.15 deep inside). slit: the lips' contact band (just behind the inner-lip contour, seen through the gap
    of the closed mouth) - painted at the rim's x0.55, so the closed mouth shows a dark line, not a pale one."""
    S = atlas.size
    slit = np.zeros(atlas.n, bool) if slit is None else slit & ~interior
    im = np.zeros((S, S), np.uint8)
    im[atlas.texel_yx[interior, 0], atlas.texel_yx[interior, 1]] = 1
    both = im.copy()
    both[atlas.texel_yx[slit, 0], atlas.texel_yx[slit, 1]] = 1
    ring_im = cv2.dilate(both, np.ones((7, 7), np.uint8)) & (1 - both)
    idx_img = atlas.index_image()
    ring = idx_img[ring_im > 0]
    ring = ring[ring >= 0]
    if not interior.any() or len(ring) < 10:
        return alb_bgr, {"mouth_interior_texels": 0}
    lin = _lin(_texel_values(atlas, alb_bgr))
    c = np.median(lin[ring], 0)
    dist = cv2.distanceTransform(im, cv2.DIST_L2, 3)[atlas.texel_yx[interior, 0], atlas.texel_yx[interior, 1]]
    k = (0.55 - 0.40 * np.clip(dist / 6.0, 0, 1))[:, None]
    out = alb_bgr.copy()
    out[atlas.texel_yx[interior, 0], atlas.texel_yx[interior, 1]] = (_srgb(c[None] * k)[:, ::-1] * 255 + 0.5).astype(np.uint8)
    if slit.any():
        out[atlas.texel_yx[slit, 0], atlas.texel_yx[slit, 1]] = (_srgb(c[None] * 0.55)[:, ::-1] * 255 + 0.5).astype(np.uint8)
    return out, {"mouth_interior_texels": int(interior.sum()), "slit_texels": int(slit.sum()),
                 "lip_ring_colour_srgb": np.round(_srgb(c), 3).tolist()}


def lip_geometry(atlas, V: np.ndarray, emb: LandmarkEmbedding, dilate_m: float = 0.001):
    """Texels inside the outer-lip contour (iBUG 48-59) of the rest head V seen from the front (x, y; the polygon grown
    by dilate_m), every texel's depth behind the inner-lip contour (60-67, z interpolated over x; metres, positive =
    into the mouth) and its 3D distance to that contour (closed polyline; inf outside the polygon).
    -> (inside (N,) bool, depth (N,), dist_inner (N,))."""
    from matplotlib.path import Path as MPath
    L = emb.points(V)
    P = atlas.surface(V)
    outer = L[emb.index(range(48, 60))]
    inner = L[emb.index(range(60, 68))]
    c = outer[:, :2].mean(0)
    rad = outer[:, :2] - c
    poly = c + rad * (1 + dilate_m / np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-6))
    head = (atlas.region & 1) > 0
    box = head & (np.abs(P[:, 0] - c[0]) < 0.05) & (np.abs(P[:, 1] - c[1]) < 0.035) & (P[:, 2] > outer[:, 2].min() - 0.035)
    idx = np.flatnonzero(box)
    inside = np.zeros(atlas.n, bool)
    inside[idx] = MPath(poly).contains_points(P[idx, :2])
    o = np.argsort(inner[:, 0])
    depth = np.interp(P[:, 0], inner[o, 0], inner[o, 2]) - P[:, 2]
    dist = np.full(atlas.n, np.inf)
    ii = np.flatnonzero(inside)
    Q = P[ii]
    best = np.full(len(ii), np.inf)
    for a_, b_ in zip(inner, np.roll(inner, -1, 0)):
        e = b_ - a_
        u = np.clip(((Q - a_) @ e) / max(float(e @ e), 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(a_ + u[:, None] * e - Q, axis=1))
    dist[ii] = best
    return inside, depth, dist


def mouth_masks(atlas, V: np.ndarray, faces: np.ndarray, emb: LandmarkEmbedding, slit_mm: tuple = (0.3, 1.5),
                contact_mm: float = 0.8) -> dict:
    """Mesh-defined mouth texels of the rest head: interior (the lips' inner surfaces: unseen from 15 frontal views, or
    > slit_mm[1] behind the inner-lip contour), slit (slit_mm[0]..[1] behind it, or within contact_mm of the contour:
    the contact band a closed mouth shows as its dark line) and lips (the visible vermilion inside the outer-lip
    contour, as the MESH has it - the portrait's smiling lips are warped thinner, so their label under-covers the
    mesh's lips)."""
    interior0 = mouth_interior_texels(atlas, V, faces, emb)
    inside, depth, dist = lip_geometry(atlas, V, emb)
    interior = interior0 | (inside & (depth > slit_mm[1] / 1000))
    slit = inside & ~interior & ((depth > slit_mm[0] / 1000) | (dist < contact_mm / 1000))
    lips = inside & ~interior & ~slit
    return {"interior": interior, "slit": slit, "lips": lips}


def _uv_weight(atlas, sel: np.ndarray, ramp: float = 2.0, blur: float = 0.7) -> np.ndarray:
    """Per-texel weight of a texel selection: 1 inside, ramping to 0 over `ramp` texels at its border (inside only)."""
    S = atlas.size
    im = np.zeros((S, S), np.uint8)
    im[atlas.texel_yx[sel, 0], atlas.texel_yx[sel, 1]] = 1
    d = cv2.distanceTransform(im, cv2.DIST_L2, 3)
    w = np.clip(d / max(ramp, 1e-6), 0, 1).astype(np.float32)
    if blur > 0:
        w = cv2.GaussianBlur(w, (0, 0), blur) * im
    return w[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]]


def portrait_lip_contrast(img_bgr: np.ndarray, mp: np.ndarray, gain_lin=None) -> dict:
    """The portrait's lip colour relative to its cheeks (Lab median differences dL, da, db; same regions as face_look:
    portrait_labels lips / cheek discs), after the composite's global linear gain (the colour transfer to the bake's
    lighting), so it is directly a target for the texture."""
    lab_lbl = portrait_labels(img_bgr.shape, mp)
    lin = _lin(img_bgr[:, :, ::-1].astype(np.float32) / 255.0)
    if gain_lin is not None:
        lin = lin * np.asarray(gain_lin, np.float32)
    lab = _lab(_srgb(lin).astype(np.float32))
    lips, cheek = (lab_lbl & LAB_LIPS) > 0, (lab_lbl & LAB_CHEEK) > 0
    if lips.sum() < 5 or cheek.sum() < 5:
        return {}
    d = np.median(lab[lips], 0) - np.median(lab[cheek], 0)
    return {"dL": round(float(d[0]), 2), "da": round(float(d[1]), 2), "db": round(float(d[2]), 2),
            "lips_px": int(lips.sum()), "cheek_px": int(cheek.sum())}


def _contrast(lab: np.ndarray, region: np.ndarray, cheek: np.ndarray) -> np.ndarray:
    return np.median(lab[region], 0) - np.median(lab[cheek], 0)


def set_region_colour(atlas, alb_bgr: np.ndarray, region: np.ndarray, cheek: np.ndarray, target: dict,
                      keep_detail: float = 0.6, ramp: float = 2.0) -> tuple[np.ndarray, dict]:
    """Recolour a texel region so its median Lab minus the cheeks' median equals target {dL, da, db} (missing keys:
    unchanged), keeping keep_detail of its own variation around its median; the border ramps over `ramp` texels.
    -> (albedo, {before, target, after})."""
    rgb = _texel_values(atlas, alb_bgr)
    lab = _lab(rgb)
    if region.sum() < 10 or cheek.sum() < 10:
        return alb_bgr, {"skipped": "region or cheeks too small"}
    before = _contrast(lab, region, cheek)
    c = np.median(lab[cheek], 0)
    cur = np.median(lab[region], 0)
    goal = cur.copy()
    for i, k in enumerate(("dL", "da", "db")):
        if k in target and target[k] is not None:
            goal[i] = c[i] + float(target[k])
    w = _uv_weight(atlas, region, ramp)
    sel = w > 1e-3
    new = goal[None] + keep_detail * (lab[sel] - cur[None])
    out_lab = lab.copy()
    out_lab[sel] = lab[sel] + w[sel, None] * (new - lab[sel])
    out = alb_bgr.copy()
    out[atlas.texel_yx[sel, 0], atlas.texel_yx[sel, 1]] = (_rgb(out_lab[sel])[:, ::-1] * 255 + 0.5).astype(np.uint8)
    after = _contrast(_lab(_texel_values(atlas, out)), region, cheek)
    rnd = lambda v: dict(zip(("dL", "da", "db"), (round(float(x), 2) for x in v)))
    return out, {"texels": int(region.sum()), "before": rnd(before), "target": {k: target.get(k) for k in ("dL", "da", "db")},
                 "after": rnd(after)}


def look_targets(work: Path, role: str, cfg: FaceTexConfig) -> dict | None:
    """Video-matched look from face/look.json (face_look), as targets relative to the cheeks:
      follow lips: the colour change attributable to makeup = the portrait -> video change minus the same change of the
               lead's natural lips (the capture control); a* fully, L by lip_L_frac. Expressed as ratios of the
               portrait's own lip contrast at video scale (scale-free: the texture's portrait-look lip target x ratio).
      lead chin: the video's beard darkness vs the cheeks (no control), scaled to full resolution by the portrait's
               native / video-scale ratio -> an absolute dL target for the texture.
    None when not measured."""
    p = work / "face" / "look.json"
    if not p.exists():
        return None
    lk = json.loads(p.read_text())
    rl = lk.get("roles", {}).get(role)
    if not rl or "compare" not in rl:
        return None
    cmp_, nat, deg = rl["compare"], rl.get("portrait_native", {}), rl.get("portrait_degraded", {})
    out = {"source": str(p)}
    ctrl = lk["roles"].get("lead", {}).get("compare", {})
    if role == "follow" and "lip_da" in cmp_ and "lip_da" in ctrl and abs(deg.get("lip_da", 0)) > 1:
        loss_a = ctrl["lip_da"]["portrait_degraded"] - ctrl["lip_da"]["video_median"]     # capture loss, natural lips
        t_a = cmp_["lip_da"]["video_median"] + loss_a                                     # makeup-free, video scale
        out["lip_ratio_a"] = round(float(np.clip(t_a / deg["lip_da"], 0, 1.5)), 3)
        out["lip_target_videoscale_da"] = round(float(t_a), 2)
        if "lip_dL" in cmp_ and "lip_dL" in ctrl and abs(deg.get("lip_dL", 0)) > 1:
            mk = (cmp_["lip_dL"]["video_median"] - deg["lip_dL"]) - \
                 (ctrl["lip_dL"]["video_median"] - ctrl["lip_dL"]["portrait_degraded"])
            t_L = deg["lip_dL"] + cfg.lip_L_frac * mk
            out["lip_ratio_L"] = round(float(np.clip(t_L / deg["lip_dL"], 0, 1.5)), 3)
            out["lip_target_videoscale_dL"] = round(float(t_L), 2)
    if role == "lead" and "chin_dL" in cmp_ and cmp_["chin_dL"].get("differs") and abs(deg.get("chin_dL", 0)) > 1:
        scale = nat.get("chin_dL", deg["chin_dL"]) / deg["chin_dL"]
        out["chin_dL"] = round(float(cmp_["chin_dL"]["video_median"] * scale), 2)
        out["chin_video_median"] = cmp_["chin_dL"]["video_median"]
    return out if len(out) > 1 else None


def region_stats(atlas, alb_bgr: np.ndarray, labels_t: np.ndarray) -> dict:
    """Lip a* / L and chin L relative to the cheeks, measured on the texture (same definitions as face_look; 'lips' =
    the portrait's projected lip label, 'mesh_lips' = the mesh's visible lips)."""
    lab = _lab(_texel_values(atlas, alb_bgr))
    cheek = (labels_t & LAB_CHEEK) > 0
    out = {}
    if cheek.sum() < 20:
        return out
    c = np.median(lab[cheek], 0)
    for name, bit in (("lips", LAB_LIPS), ("mesh_lips", LAB_MESH_LIPS), ("chin", LAB_BEARD)):
        m = (labels_t & bit) > 0
        if m.sum() >= 20:
            d = np.median(lab[m], 0) - c
            out[f"{name}_dL"], out[f"{name}_da"], out[f"{name}_db"] = (round(float(v), 2) for v in d)
    return out


def _beard_blend(atlas, alb_bgr: np.ndarray, labels_t: np.ndarray, f: float) -> np.ndarray:
    """Blend the chin beard's dark texels toward the cheek skin by fraction f (broad feather, no region edge)."""
    lab = _lab(_texel_values(atlas, alb_bgr))
    cheek = (labels_t & LAB_CHEEK) > 0
    c = np.median(lab[cheek], 0) if cheek.sum() >= 20 else np.median(lab[(atlas.region & 2) > 0], 0)
    S = atlas.size
    im = np.zeros((S, S), np.float32)
    m = (labels_t & LAB_BEARD) > 0
    im[atlas.texel_yx[m, 0], atlas.texel_yx[m, 1]] = 1
    im = cv2.dilate(im, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    w = np.clip(cv2.GaussianBlur(im, (0, 0), 12.0), 0, 1)[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]]
    dark = np.clip((c[0] - lab[:, 0] - 4.0) / 20.0, 0, 1)            # only the dark (beard) texels move
    fw = (f * w * dark)[:, None]
    sel = fw[:, 0] > 1e-4
    out_lab = lab[sel] * (1 - fw[sel]) + c[None] * fw[sel]
    out = alb_bgr.copy()
    out[atlas.texel_yx[sel, 0], atlas.texel_yx[sel, 1]] = (_rgb(out_lab)[:, ::-1] * 255 + 0.5).astype(np.uint8)
    return out


def video_look(atlas, alb_bgr: np.ndarray, labels_t: np.ndarray, tgt: dict,
               lip_target: dict | None = None) -> tuple[np.ndarray, dict]:
    """Apply the video-matched look to the composited portrait-look albedo, aimed at targets relative to the cheeks
    and measured on this texture (2026-10-07: fixed shifts overshot):
      lips (follow)  mesh-lip colour = lip_target (the portrait look's dL/da/db target) x the look's ratios
      chin (lead)    beard blended toward the cheek skin by the fraction that brings the texture's chin dL to the
                     target (bisection on the measured value)
    -> (albedo, info with before / target / achieved)."""
    info = {"before": region_stats(atlas, alb_bgr, labels_t), "targets": {k: v for k, v in tgt.items() if k != "source"}}
    out = alb_bgr
    if lip_target and ("lip_ratio_a" in tgt or "lip_ratio_L" in tgt):
        lips = (labels_t & LAB_MESH_LIPS) > 0
        cheek = (labels_t & LAB_CHEEK) > 0
        ra, rL = tgt.get("lip_ratio_a", 1.0), tgt.get("lip_ratio_L", 1.0)
        t = {"dL": round(lip_target["dL"] * rL, 2), "da": round(lip_target["da"] * ra, 2), "db": round(lip_target["db"] * ra, 2)}
        out, info["lips"] = set_region_colour(atlas, out, lips, cheek, t)
    if "chin_dL" in tgt:
        target = float(tgt["chin_dL"])
        chin = lambda im: region_stats(atlas, im, labels_t).get("chin_dL")
        c0 = chin(out)
        if c0 is not None and c0 < target:                       # darker than the video: lighten
            lo, hi = 0.0, 1.0
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if chin(_beard_blend(atlas, out, labels_t, mid)) < target:
                    lo = mid
                else:
                    hi = mid
            f = 0.5 * (lo + hi)
            out = _beard_blend(atlas, out, labels_t, f)
            info["chin"] = {"before_dL": c0, "target_dL": target, "blend_toward_skin": round(f, 3), "after_dL": chin(out)}
        else:
            info["chin"] = {"before_dL": c0, "target_dL": target, "blend_toward_skin": 0.0, "after_dL": c0,
                            "note": "texture already at or lighter than the target"}
    info["after"] = region_stats(atlas, out, labels_t)
    info["changed_texels"] = int((np.abs(out.astype(int) - alb_bgr.astype(int)).max(-1) > 0).sum())
    return out, info


def pad_uncovered(atlas, img_bgr: np.ndarray) -> np.ndarray:
    """Grow every UV island into the uncovered texels (push-pull fill) so bilinear / mip-mapped sampling at island
    borders and holes (the mouth opening) never picks up the background colour (a pale line along the closed lips)."""
    cov = np.zeros((atlas.size, atlas.size), bool)
    cov[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]] = True
    f = _fill_masked(img_bgr.astype(np.float32), cov, iters=11)
    return np.where(cov[..., None], img_bgr, np.clip(f + 0.5, 0, 255).astype(np.uint8))


def gradient_smooth(atlas, alb_bgr: np.ndarray, region: np.ndarray, anchor: np.ndarray, clip: float = 0.025,
                    lam: float = 1e-3) -> tuple[np.ndarray, dict]:
    """Gradient-domain clean-up of a texel region (Poisson, in log linear RGB on the UV grid): the region's colours f
    solve sum_q (f_p - f_q) = sum_q clip(s_p - s_q) over 4-neighbour texels q, with s the current colours and the
    per-texel steps clipped to +-clip (hard seams, patch borders and stray hair strokes are steps; skin detail is
    not). Neighbours outside the region that are in `anchor` are fixed (Dirichlet: the region meets them without a
    step); other neighbours (hair, uncovered) are left free (Neumann). lam ties f weakly to s (keeps isolated parts
    well-posed). -> (albedo, info)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    S = atlas.size
    idx_img = atlas.index_image()
    reg = region & (atlas.region & 4 == 0)
    if reg.sum() < 10:
        return alb_bgr, {"texels": 0}
    yx = atlas.texel_yx[reg]
    unk = -np.ones(atlas.n, np.int64)
    unk[np.flatnonzero(reg)] = np.arange(len(yx))
    s = np.log(np.maximum(_lin(_texel_values(atlas, alb_bgr)), 1e-4)).astype(np.float64)
    rows, cols, vals = [], [], []
    rhs = np.zeros((len(yx), 3))
    diag = np.full(len(yx), lam)
    rhs += lam * s[reg]
    p_t = np.flatnonzero(reg)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        qy, qx = yx[:, 0] + dy, yx[:, 1] + dx
        ok = (qy >= 0) & (qy < S) & (qx >= 0) & (qx < S)
        q = np.full(len(yx), -1, np.int64)
        q[ok] = idx_img[qy[ok], qx[ok]]
        has = q >= 0
        qq = np.where(has, q, 0)
        g = np.clip(s[p_t] - s[qq], -clip, clip)
        inner = has & (unk[qq] >= 0)
        fixed = has & ~inner & anchor[qq]
        use = inner | fixed
        diag += use
        rhs[use] += g[use]
        rhs[fixed] += s[qq[fixed]]
        i = np.flatnonzero(inner)
        rows.append(i)
        cols.append(unk[qq[inner]])
        vals.append(-np.ones(len(i)))
    A = sp.csr_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(len(yx))]),
                                                       np.concatenate(cols + [np.arange(len(yx))]))),
                      shape=(len(yx), len(yx)))
    f = spsolve(A.tocsc(), rhs)
    if f.ndim == 1:
        f = f[:, None]
    out = alb_bgr.copy()
    out[yx[:, 0], yx[:, 1]] = (_srgb(np.exp(f))[:, ::-1] * 255 + 0.5).astype(np.uint8)
    d = np.abs(f - s[reg])
    return out, {"texels": int(len(yx)), "change_log_p50_p95": [round(float(np.percentile(d, 50)), 4),
                                                                round(float(np.percentile(d, 95)), 4)]}


def seam_metrics(atlas, alb_bgr: np.ndarray, V: np.ndarray, alpha_t: np.ndarray, repaired: np.ndarray,
                 skin: np.ndarray | None = None, brow_y: float | None = None, band: int = 6) -> dict:
    """Luma steps of the head texture in % of the local luma (low-passed, sigma 2 texels), medians and p90:
      vertical_seam_scalp / _face  per 1 cm height bin of the front of the head (|x| < 5 cm on the rest surface; above
                     brow_y + 1 cm / from the nose tip (brow_y - 5.5 cm: no beard) to brow_y), the largest step
                     between adjacent 4 mm x-bins -
                     a hard seam (the bake's L/R camera boundary, a mirror switch) shows as a large step, smooth shading
                     as a small one
      portrait_border  across the portrait's alpha-0.5 contour (texels band..2 band inside vs the nearest outside)
      repair_border    across the skin-repair border (likewise)
    Only `skin` texels (no eyes / mouth / brows / hair) are used when given."""
    from scipy.spatial import cKDTree
    S = atlas.size
    lin = _lin(_texel_values(atlas, alb_bgr)) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    head = ((atlas.region & 1) > 0) & ((atlas.region & 4) == 0)
    use = head if skin is None else head & skin
    im = np.zeros((S, S), np.float32)
    wim = np.zeros((S, S), np.float32)
    im[atlas.texel_yx[use, 0], atlas.texel_yx[use, 1]] = lin[use]
    wim[atlas.texel_yx[use, 0], atlas.texel_yx[use, 1]] = 1
    Y = (cv2.GaussianBlur(im, (0, 0), 2.0) / np.maximum(cv2.GaussianBlur(wim, (0, 0), 2.0), 1e-6))[
        atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]]
    P = atlas.surface(V)
    out = {}
    pm = lambda v: [round(float(np.median(v)), 2), round(float(np.percentile(v, 90)), 2)] if len(v) else None
    front = use & (P[:, 2] > 0.0) & (np.abs(P[:, 0]) < 0.05)
    by = float(np.median(P[head, 1])) if brow_y is None else brow_y
    for name, lo, hi in (("vertical_seam_scalp", by + 0.01, P[head, 1].max()), ("vertical_seam_face", by - 0.055, by)):
        steps = []
        for y in np.arange(lo, hi, 0.01):
            b = front & (P[:, 1] >= y) & (P[:, 1] < y + 0.01)
            if b.sum() < 100:
                continue
            xb = np.floor(P[b, 0] / 0.004).astype(int)
            med = {k: np.median(Y[b][xb == k]) for k in np.unique(xb) if (xb == k).sum() >= 5}
            ks = sorted(med)
            st = [abs(med[k + 1] - med[k]) / max(0.5 * (med[k + 1] + med[k]), 1e-6) * 100 for k in ks if k + 1 in med]
            if st:
                steps.append(max(st))
        out[f"{name}_max_step_pct_median_p90"] = pm(steps)

    def border(mask_in):
        m = np.zeros((S, S), np.uint8)
        m[atlas.texel_yx[mask_in, 0], atlas.texel_yx[mask_in, 1]] = 1
        cov = np.zeros((S, S), np.uint8)
        cov[atlas.texel_yx[head, 0], atlas.texel_yx[head, 1]] = 1
        din = cv2.distanceTransform(m, cv2.DIST_L2, 3)[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]]
        dout = cv2.distanceTransform(((1 - m) * cov).astype(np.uint8), cv2.DIST_L2, 3)[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]]
        bi = np.flatnonzero(use & mask_in & (din >= band) & (din < 2 * band))
        bo = np.flatnonzero(use & ~mask_in & (dout >= band) & (dout < 2 * band))
        if len(bi) < 20 or len(bo) < 20:
            return None
        dd, j = cKDTree(atlas.texel_yx[bo].astype(float)).query(atlas.texel_yx[bi].astype(float))
        ok = dd < 4 * band
        if ok.sum() < 20:
            return None
        a_, b_ = Y[bi[ok]], Y[bo[j[ok]]]
        return pm(np.abs(a_ - b_) / np.maximum(0.5 * (a_ + b_), 1e-6) * 100)
    face = (atlas.region & 2) > 0
    out["portrait_border_step_pct_median_p90"] = border(face & (alpha_t > 0.5))
    if repaired.any():
        out["repair_border_step_pct_median_p90"] = border(repaired)
    return out


def _ear_clip(pts2: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon (n,2) in its own winding order (ear clipping) -> index triples."""
    idx = list(range(len(pts2)))
    area = 0.5 * sum(pts2[i, 0] * pts2[(i + 1) % len(pts2), 1] - pts2[(i + 1) % len(pts2), 0] * pts2[i, 1]
                     for i in range(len(pts2)))
    sgn = 1.0 if area > 0 else -1.0
    cross = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts2[i0], pts2[i1], pts2[i2]
            if sgn * cross(a, b, c) <= 1e-12:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = pts2[j]
                if sgn * cross(a, b, p) >= 0 and sgn * cross(b, c, p) >= 0 and sgn * cross(c, a, p) >= 0:
                    inside = True
                    break
            if not inside:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            break                                   # degenerate: fan the rest
    for k in range(1, len(idx) - 1):
        tris.append((idx[0], idx[k], idx[k + 1]))
    return tris


_CAP: dict = {}


def mouth_cap(model_npz: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """SMPL-X has one open boundary: the mouth bag ~16 mm behind the lips (32 edges). Through the lip gap of a closed
    mouth (~1.2 mm on the rest head) a renderer sees past it into the inside of the back of the head - a pale line
    along the lips. This caps it: the boundary loop ear-clipped (in its best-fit plane) with the winding that matches
    the neighbouring faces. -> (faces (k,3) SMPL-X vertex ids, ft (k,3) the boundary vertices' vt ids from their own
    faces, so the cap shows the painted mouth interior)."""
    path = Path(model_npz or smplx_model_npz())
    if str(path) in _CAP:
        return _CAP[str(path)]
    m = np.load(path, allow_pickle=True)
    f, ft, v = np.asarray(m["f"], np.int64), np.asarray(m["ft"], np.int64), np.asarray(m["v_template"], float)
    und = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    u, cnt = np.unique(und, axis=0, return_counts=True)
    bset = {tuple(e) for e in u[cnt == 1]}
    nxt, vt_of = {}, {}
    for fi in range(len(f)):
        for c in range(3):
            a, b = int(f[fi, c]), int(f[fi, (c + 1) % 3])
            if (min(a, b), max(a, b)) in bset:
                nxt[a] = b                              # directed as in its face
                vt_of[a] = int(ft[fi, c])
                vt_of.setdefault(b, int(ft[fi, (c + 1) % 3]))
    if not nxt:
        out = (np.zeros((0, 3), np.int64), np.zeros((0, 3), np.int64))
        _CAP[str(path)] = out
        return out
    start = next(iter(nxt))
    loop = [start]
    while nxt.get(loop[-1], start) != start and len(loop) <= len(nxt):
        loop.append(nxt[loop[-1]])
    loop = loop[::-1]                                   # the cap runs every boundary edge the other way
    P = v[loop]
    c0 = P.mean(0)
    _, _, vt_ = np.linalg.svd(P - c0, full_matrices=False)
    pts2 = (P - c0) @ vt_[:2].T
    tris = np.array([[loop[a], loop[b], loop[c]] for a, b, c in _ear_clip(pts2)], np.int64)
    tft = np.array([[vt_of[x] for x in t] for t in tris], np.int64)
    _CAP[str(path)] = (tris, tft)
    return tris, tft


def capped_atlas(atlas):
    """A shallow copy of the UV atlas whose faces / ft include the mouth cap (for renders; texel lookups unchanged)."""
    import copy
    tris, tft = mouth_cap()
    a = copy.copy(atlas)
    if len(tris):
        a.faces = np.concatenate([atlas.faces, tris.astype(atlas.faces.dtype)], 0)
        a.ft = np.concatenate([atlas.ft, tft.astype(atlas.ft.dtype)], 0)
    return a


def face_outputs(bake: Path) -> dict:
    """Output paths next to a baked texture (never the input itself)."""
    bake = Path(bake)
    if bake.stem.endswith(("_face", "_face_videolook")):
        raise ValueError(f"{bake} is already a face composite: give the plain bake")
    out = {"portrait": bake.with_name(f"{bake.stem}_face.png"),
           "video": bake.with_name(f"{bake.stem}_face_videolook.png")}
    for v in out.values():
        if v.resolve() == bake.resolve():
            raise ValueError("output would overwrite the input bake")
    return out


# ------------------------------------------------------------------------------------------------ eyes


def iris_colours(img: np.ndarray, mp: np.ndarray) -> dict:
    """Median iris / sclera colours (sRGB 0..1) from MediaPipe's iris landmarks (for the eyeball texture)."""
    out = {}
    iod = float(np.linalg.norm(mp[33] - mp[263]))
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    irises, scleras = [], []
    for iris, eye in ((MP_IRIS_R, MP_EYE_R), (MP_IRIS_L, MP_EYE_L)):
        c = mp[iris[0]]
        rad = max(1.0, float(np.mean(np.linalg.norm(mp[iris[1:]] - c, axis=1))))
        eye_m = _poly_mask(img.shape, mp[eye])
        disk = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(disk, tuple(np.round(c).astype(int)), max(1, int(round(rad * 0.8))), 1, -1)
        irises.append(rgb[disk.astype(bool) & eye_m])
        scl = eye_m & ~cv2.dilate(disk, np.ones((3, 3), np.uint8)).astype(bool)
        scleras.append(rgb[scl])
    ir = np.concatenate(irises) if sum(len(x) for x in irises) else np.zeros((1, 3))
    sc = np.concatenate(scleras) if sum(len(x) for x in scleras) else np.ones((1, 3)) * 0.8
    out["iris"] = np.median(ir, 0).round(3).tolist()
    # sclera: brightest half (lids/lashes darken the rest)
    Ys = sc @ np.array([0.2126, 0.7152, 0.0722])
    out["sclera"] = np.median(sc[Ys >= np.median(Ys)], 0).round(3).tolist() if len(sc) > 1 else [0.8, 0.78, 0.76]
    out["iris_px"] = round(float(np.mean([np.mean(np.linalg.norm(mp[i[1:]] - mp[i[0]], axis=1))
                                          for i in (MP_IRIS_R, MP_IRIS_L)])), 2)
    out["iod_px"] = round(iod, 1)
    return out


def paint_eyeballs(atlas, albedo_bgr: np.ndarray, cols: dict, model_npz: Path,
                   gain_lin: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Procedural eyeballs: sclera / iris (half-angle 28 deg) / pupil (10 deg) around each eyeball's rest-pose
    forward axis (+z) from the eye joint (23 / 24, the eyeball centre). Portrait colours are moved to the bake's
    lighting with the face's global linear gain. Replaces the eyeball texels (region bit 4)."""
    m = np.load(model_npz, allow_pickle=True)
    vt = np.asarray(m["v_template"], float)
    w = np.asarray(m["weights"], float)
    J = np.asarray(m["J_regressor"], float) @ vt
    eye_t = np.flatnonzero((atlas.region & 4) > 0)
    P = np.einsum("nkd,nk->nd", vt[atlas.faces[atlas.face[eye_t]]], atlas.bary[eye_t])
    vw = np.einsum("nkj,nk->nj", w[atlas.faces[atlas.face[eye_t]]], atlas.bary[eye_t])
    out = albedo_bgr.copy()
    g = np.ones(3) if gain_lin is None else np.asarray(gain_lin, float)
    sclera = _srgb(_lin(np.array(cols["sclera"])) * g * 0.8)      # lids shade the visible sclera
    iris = _srgb(_lin(np.array(cols["iris"])) * g)
    for j in (23, 24):
        sel = vw[:, j] > 0.5
        if not sel.any():
            continue
        d = P[sel] - J[j]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        ang = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))     # 0 = straight ahead
        col = np.tile(sclera, (len(d), 1))
        t_iris = np.clip((ang - 27.0) / 3.0, 0, 1)[:, None]       # iris ~ 28 deg half-angle, 3 deg soft limbus
        col = iris * (1 - t_iris) + col * t_iris
        limbal = np.exp(-((ang - 28.0) / 2.0) ** 2)[:, None] * 0.4
        col = col * (1 - limbal)
        pupil = np.clip((ang - 10.0) / 2.0, 0, 1)[:, None]
        col = np.array([0.03, 0.03, 0.03]) * (1 - pupil) + col * pupil
        yx = atlas.texel_yx[eye_t[sel]]
        out[yx[:, 0], yx[:, 1]] = (np.clip(col[:, ::-1], 0, 1) * 255 + 0.5).astype(np.uint8)
    return out, int(len(eye_t))


# ------------------------------------------------------------------------------------------------ run


def _hair_like(atlas, alb_bgr: np.ndarray, alpha_t: np.ndarray, cfg: FaceTexConfig) -> np.ndarray | None:
    """Texels whose colour is far from the face's skin and closer to the scalp's (repair_face_skin's test)."""
    lab = _lab(_texel_values(atlas, alb_bgr))
    feat = np.stack([lab[:, 0] * 0.3, lab[:, 1], lab[:, 2]], 1)
    head, eye, fbit = (atlas.region & 1) > 0, (atlas.region & 4) > 0, (atlas.region & 2) > 0
    trusted = fbit & (alpha_t > 0.8) & ~eye
    scalp = head & ~fbit & ~eye
    if trusted.sum() < 500 or scalp.sum() < 500:
        return None
    st_s, st_h = _gauss_stats(feat[trusted]), _gauss_stats(feat[scalp])
    m_s, m_h = _maha(feat, st_s), _maha(feat, st_h)
    return (m_s > cfg.repair_maha) & (-0.5 * m_h ** 2 - 0.5 * st_h[2] > -0.5 * m_s ** 2 - 0.5 * st_s[2])


def _dark_features(atlas, alb_bgr: np.ndarray, sel: np.ndarray, sigma: float = 8.0, thr: float = 10.0) -> np.ndarray:
    """Texels (within sel) darker than their local (Gaussian sigma texels) mean luminance by > thr L: brows, lashes,
    nostrils - kept out of the gradient clean-up."""
    S = atlas.size
    L = _lab(_texel_values(atlas, alb_bgr))[:, 0]
    im = np.zeros((S, S), np.float32)
    w = np.zeros((S, S), np.float32)
    head = (atlas.region & 1) > 0
    im[atlas.texel_yx[head, 0], atlas.texel_yx[head, 1]] = L[head]
    w[atlas.texel_yx[head, 0], atlas.texel_yx[head, 1]] = 1
    loc = (cv2.GaussianBlur(im, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(w, (0, 0), sigma), 1e-6))[
        atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]]
    return _uv_mask_ops(atlas, sel & (L < loc - thr), close=0, open_=0, dilate=2)


def remove_streaks(atlas, alb_bgr: np.ndarray, region: np.ndarray, ksize: int = 11, thr: float = 7.0) -> tuple[np.ndarray, dict]:
    """Thin light lines inside a region (a bake seam through the beard): texels lighter than their local median
    (ksize texels) by > thr L take that median colour (1-texel dilation, feathered)."""
    S = atlas.size
    if region.sum() < 50:
        return alb_bgr, {"texels": 0}
    yx = atlas.texel_yx[region]
    y0, x0 = np.maximum(yx.min(0) - ksize, 0)
    y1, x1 = np.minimum(yx.max(0) + ksize + 1, S)
    crop = alb_bgr[y0:y1, x0:x1]
    med = cv2.medianBlur(crop, ksize)
    Lc = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)[..., 0].astype(np.float32) * 100 / 255
    Lm = cv2.cvtColor(med, cv2.COLOR_BGR2Lab)[..., 0].astype(np.float32) * 100 / 255
    rm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    rm[yx[:, 0] - y0, yx[:, 1] - x0] = 1
    cand = ((Lc - Lm > thr) & (rm > 0)).astype(np.uint8)
    cand = cv2.dilate(cand, np.ones((3, 3), np.uint8)) & rm
    w = cv2.GaussianBlur(cand.astype(np.float32), (0, 0), 0.8)
    w = np.maximum(w, cand.astype(np.float32))[..., None]
    out = alb_bgr.copy()
    out[y0:y1, x0:x1] = np.clip(crop * (1 - w) + med * w + 0.5, 0, 255).astype(np.uint8)
    return out, {"texels": int(cand.sum()), "ksize": ksize, "thr_L": thr}


def postprocess(atlas, emb: LandmarkEmbedding, head: HeadModel, bake: np.ndarray, proj: dict, img2: np.ndarray,
                mp: np.ndarray, role: str, cfg: FaceTexConfig, eyes_json: Path | None = None,
                compare: dict | None = None):
    """Composite the portrait projection onto the bake and clean it up -> (albedo BGR, composite info, alpha (N,),
    labels (N,), repaired (N,), step infos incl. '_layer'). Steps: composite, skin repair (follow), head
    symmetrisation (lead), gradient-domain clean-up (follow: repair + portrait border; lead: scalp band with the bake
    seam), lip colour (the mesh's lips take the portrait's lip contrast), mouth interior + slit, beard streaks (lead),
    eyeballs; seam metrics after each stage (and of the `compare` albedos {name: BGR}, e.g. the bake and a previous
    face texture, with the same skin mask)."""
    import torch
    steps: dict = {}
    alb, layer, cinfo = composite(atlas, bake, proj, cfg)
    steps["_layer"] = layer
    alpha_t = layer[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1], 3].astype(np.float32) / 255.0
    labels_t = np.zeros(atlas.n, np.uint8)
    labels_t[proj["texels"]] = proj["labels"]
    with torch.no_grad():
        V0 = head.verts(torch.zeros(cfg.n_expr), torch.zeros(3)).numpy().astype(np.float64)
    mm = mouth_masks(atlas, V0, head.faces, emb)
    labels_t[mm["lips"]] |= LAB_MESH_LIPS
    repaired = np.zeros(atlas.n, bool)
    snaps = [("composite", alb)]
    head_m = ((atlas.region & 1) > 0) & ((atlas.region & 4) == 0)
    if cfg.refill_base.get(role):
        excl = mm["lips"] | mm["slit"] | mm["interior"] | \
            _uv_mask_ops(atlas, (labels_t & LAB_LIPS) > 0, close=0, open_=0, dilate=4) | _dark_features(atlas, alb, head_m)
        alb, steps["refill_base"] = refill_face_base(atlas, alb, layer, alpha_t, excl, cfg)
        snaps.append(("refill_base", alb))
    if cfg.skin_repair.get(role):
        alb, repaired, steps["skin_repair"] = repair_face_skin(atlas, alb, alpha_t, cfg)
    if cfg.symmetrize_head.get(role):
        alb, steps["symmetrize_head"] = symmetrize_head(atlas, alb, V0)
    snaps.append(("repair_symmetrize", alb))
    mouth = mm["lips"] | mm["slit"] | mm["interior"]
    gs = {}
    if cfg.smooth_repair.get(role) and repaired.any():
        face = (atlas.region & 2) > 0
        band = face & (alpha_t > 0.03) & (alpha_t < cfg.smooth_band_alpha)
        reg = _uv_mask_ops(atlas, repaired, close=0, open_=0, dilate=4) | band
        reg &= face & head_m & ~mouth
        reg &= ~_dark_features(atlas, alb, reg & (alpha_t > 0.3))
        hair = _hair_like(atlas, alb, alpha_t, cfg)
        anchor = head_m & ~reg & (np.ones(atlas.n, bool) if hair is None else ~hair)
        alb, gs["follow_repair"] = gradient_smooth(atlas, alb, reg, anchor | ~head_m, cfg.grad_clip)
    if cfg.smooth_scalp.get(role):
        P = atlas.surface(V0)
        L = emb.points(V0)
        brow_y = float(L[emb.index(range(17, 27)), 1].max())
        reg = head_m & (P[:, 1] > brow_y + 0.012) & (np.abs(P[:, 0]) < 0.06) & (alpha_t < 0.9) & ~mouth
        alb, gs["scalp_band"] = gradient_smooth(atlas, alb, reg, ~reg, cfg.scalp_clip)
    if gs:
        steps["gradient_smooth"] = gs
        snaps.append(("gradient_smooth", alb))
    cheek = (labels_t & LAB_CHEEK) > 0
    if cfg.lip_colour.get(role):
        tgt = portrait_lip_contrast(img2, mp, cinfo["gain_global_lin"])
        if tgt:
            alb, steps["lip_colour"] = set_region_colour(atlas, alb, mm["lips"], cheek, tgt, cfg.lip_keep_detail, ramp=3.0)
            steps["lip_colour"]["portrait_contrast_after_gain"] = tgt
    if cfg.mouth_interior:
        alb, steps["mouth_interior"] = paint_mouth_interior(atlas, alb, mm["interior"], mm["slit"])
        # light specks at the lip contact / corners (the portrait's inner-mouth highlights, warped) read as a pale
        # line along the closed mouth: the lip texels next to the slit take their local median where lighter
        rim = _uv_mask_ops(atlas, mm["slit"] | mm["interior"], close=0, open_=0, dilate=5) & ~mm["interior"] & head_m
        alb, steps["lip_specks"] = remove_streaks(atlas, alb, rim, ksize=7, thr=5.0)
    if cfg.streaks.get(role):
        P = atlas.surface(V0)
        L = emb.points(V0)
        chin = L[emb.index([8])[0]]
        low = head_m & (P[:, 1] < L[emb.index([57])[0], 1] - 0.004) & (np.abs(P[:, 0]) < 0.055) & (P[:, 2] > chin[2] - 0.07)
        beard = _uv_mask_ops(atlas, (labels_t & LAB_BEARD) > 0, close=0, open_=0, dilate=15) | low
        alb, steps["streaks"] = remove_streaks(atlas, alb, beard & head_m & ~mouth)
    if eyes_json is not None and Path(eyes_json).exists():
        alb, n_eye = paint_eyeballs(atlas, alb, json.loads(Path(eyes_json).read_text()), smplx_model_npz(),
                                    np.asarray(cinfo["gain_global_lin"]))
        cinfo["eyeball_texels"] = n_eye
    snaps.append(("final", alb))
    # one skin mask for every stage: head minus eyes, mouth, dark features (brows, lashes, nostrils) and hair colour
    hair = _hair_like(atlas, alb, alpha_t, cfg)
    skin = head_m & ~mouth & ~_dark_features(atlas, alb, head_m) & (np.ones(atlas.n, bool) if hair is None else ~hair)
    brow_y = float(emb.points(V0)[emb.index(range(17, 27)), 1].max())
    snaps = [(f"compare: {k}", v) for k, v in (compare or {}).items() if v is not None and v.shape == alb.shape] + snaps
    steps["seams"] = {k: seam_metrics(atlas, a_, V0, alpha_t, repaired, skin, brow_y) for k, a_ in snaps}
    steps["mouth_masks"] = {k: int(v.sum()) for k, v in mm.items()}
    return alb, cinfo, alpha_t, labels_t, repaired, steps


def run(take: Take, roles=ROLES, cfg: FaceTexConfig | None = None, composite_only: bool = False,
        eyes: bool = True, bakes: dict | None = None, review: bool = True) -> dict:
    """bakes {role: png}: any baked albedo in the SMPL-X 2048 UV layout (default avatar/<role>_albedo.png); the results
    are written NEXT to it as <stem>_face.png (portrait look) and <stem>_face_videolook.png (video-matched look), never
    over it. Portrait fit / projection / layer files live in avatar/face/ (they depend on the portrait + fit only)."""
    import torch
    torch.set_num_threads(1)                   # the PC's 4 CPUs are shared: one thread (the rasteriser is small)

    from .avatar_uv import build_atlas
    from .mamma_dyn import SMPLX_MODELS
    cfg = cfg or FaceTexConfig()
    work = take.work
    od = work / "avatar" / "face"
    od.mkdir(parents=True, exist_ok=True)
    atlas = build_atlas(SMPLX_MODELS, 2048)
    emb = LandmarkEmbedding()
    report = {"take": take.name, "config": asdict(cfg), "roles": {}}
    fits, tex = {}, {}
    for r in roles:
        t0 = time.time()
        default_bake = work / "avatar" / f"{r}_albedo.png"
        bake_p = Path(bakes[r]) if bakes and bakes.get(r) else default_bake
        outs = face_outputs(bake_p)
        bake = cv2.imread(str(bake_p))
        if bake is None:
            raise FileNotFoundError(bake_p)
        if bake.shape[:2] != (atlas.size, atlas.size):
            raise ValueError(f"{bake_p}: {bake.shape[:2]} is not the {atlas.size}^2 SMPL-X UV layout")
        uvp = od / f"{r}_portrait_uv.npz"
        rr = {"portrait": str(portrait_path(take, r)), "bake": str(bake_p),
              "bake_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(bake_p.stat().st_mtime))}
        d = np.load(work / "smplx" / f"{r}.npz", allow_pickle=True)
        head = HeadModel(np.asarray(d["betas"], np.float32), cfg.n_expr)
        proj = None
        if composite_only and uvp.exists():
            with np.load(uvp) as z:
                proj = {k: z[k] for k in z.files}
            if "labels" not in proj:
                proj = None                      # written before the texel labels existed: redo the projection
        img = cv2.imread(str(portrait_path(take, r)))
        det = portrait_landmarks(take, r)
        mp = det["mp_lmk"][:, :2].astype(np.float64)
        mask, img2, minfo = portrait_mask(img, mp, r, cfg)
        if proj is None:
            lmk68 = mp[MP_TO_IBUG68]
            fit = fit_portrait(head, emb, lmk68[emb.ibug], cfg)
            proj = project_portrait(atlas, head, fit, img2, mask, emb, cfg, labels=portrait_labels(img.shape, mp))
            proj = mirror_fill(atlas, proj, cfg, r)
            np.savez_compressed(uvp, texels=proj["texels"], rgb=proj["rgb"], w=proj["w"], labels=proj["labels"])
            np.savez(od / f"{r}_portrait_fit.npz", R=fit["R"], s=fit["s"], t2=fit["t2"], expr=fit["expr"], jaw=fit["jaw"],
                     residual=fit["residual"], lmk_proj=fit["lmk_proj"], iod_px=fit["iod_px"])
            eyec = iris_colours(img, mp)
            rr.update({"fit": {"rms_px": round(fit["rms_px"], 2), "rms_over_iod": round(fit["rms_iod"], 3),
                               "iod_px": round(fit["iod_px"], 1), "yaw_deg": round(fit["yaw_deg"], 1),
                               "jaw_open_deg": round(float(np.degrees(fit["jaw"][0])), 1),
                               "expr_norm": round(float(np.linalg.norm(fit["expr"])), 3),
                               "residual_max_px": round(float(np.linalg.norm(fit["residual"], axis=1).max()), 2)},
                       "mask": minfo, "texels_weighted": proj["texels_weighted"], "mirrored": proj.get("mirrored", 0),
                       "eyes": eyec})
            fits[r] = (fit, img, img2, mask, mp, head, proj)
            json.dump(eyec, open(od / f"{r}_eyes.json", "w"))
        prev = sorted(od.glob(f"prev_*/{r}_albedo_face.png"))           # an earlier face texture kept for comparison
        compare = {"bake": bake, **({f"previous face ({prev[-1].parent.name})": cv2.imread(str(prev[-1]))} if prev else {})}
        alb, rr["composite"], alpha_t, labels_t, repaired, steps = postprocess(
            atlas, emb, head, bake, proj, img2, mp, r, cfg, eyes_json=od / f"{r}_eyes.json" if eyes else None,
            compare=compare)
        rr.update(steps)
        layer = rr.pop("_layer")
        alb_out = pad_uncovered(atlas, alb) if cfg.pad_islands else alb
        cv2.imwrite(str(outs["portrait"]), alb_out)
        if bake_p.resolve() == default_bake.resolve():
            cv2.imwrite(str(od / f"{r}_albedo_face.png"), alb_out)      # legacy path (face_smirk / older reviews)
        cv2.imwrite(str(od / f"{r}_face_layer.png"), layer)
        np.savez_compressed(od / f"{r}_texel_masks.npz", labels=labels_t, repaired=repaired, alpha=alpha_t)
        rr["portrait_look_texture_stats"] = region_stats(atlas, alb, labels_t)
        tgt = look_targets(work, r, cfg)
        if tgt:
            alb_v, rr["video_look"] = video_look(atlas, alb, labels_t, tgt, rr.get("lip_colour", {}).get("target"))
            cv2.imwrite(str(outs["video"]), pad_uncovered(atlas, alb_v) if cfg.pad_islands else alb_v)
        else:
            rr["video_look"] = None
            outs.pop("video")
        rr["outputs"] = {k: str(v) for k, v in outs.items()}
        tex[r] = {"bake": bake_p, **outs}
        rr["runtime_s"] = round(time.time() - t0, 1)
        report["roles"][r] = rr
        print(f"[avatar_face] {r}: {json.dumps({k: v for k, v in rr.items() if k in ('fit', 'skin_repair', 'gradient_smooth', 'lip_colour', 'mouth_interior', 'streaks', 'seams', 'video_look', 'outputs')}, default=str)}", flush=True)
    (od / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if review:
        rev = work / "review"
        rev.mkdir(parents=True, exist_ok=True)
        paths = []
        if fits:
            paths.append(_fit_sheet(fits, atlas, emb, rev / "face_portrait_fit.jpg", take.name))
        paths.append(_uv_sheet(work, atlas, tex, rev / "avatar_face_uv.jpg", take.name))
        paths.append(review_angles(take, tex, rev / "avatar_face_portrait.jpg", cfg))
        paths.append(review_turntable(take, tex, rev / "avatar_face_turntable.jpg"))
        if (work / "face" / "survey.npz").exists():
            paths.append(review_video(take, tex, rev / "avatar_face_video.jpg"))
            report["likeness"] = likeness(take, tex, rev / "avatar_face_likeness.jpg")
            if report["likeness"].get("review"):
                paths.append(Path(report["likeness"]["review"]))
            print("[avatar_face] likeness: " + json.dumps({r: v["variants"] for r, v in report["likeness"].items()
                                                            if isinstance(v, dict) and "variants" in v}), flush=True)
        else:
            print("[avatar_face] review/avatar_face_video.jpg skipped: no face/survey.npz yet (face_survey / face_pod)")
        report["review"] = [str(p) for p in paths]
        (od / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


# ------------------------------------------------------------------------------------------------ review


def _render_portrait_view(head, fit, atlas, albedo_bgr, H, W, scale=4, verts=None):
    """Re-render the fitted head textured with albedo_bgr from the portrait's weak-perspective camera."""
    import torch

    from .avatar_raster import pixel_bary, rasterize
    V = (fit["verts"] if verts is None else verts).astype(np.float64)
    Q = V @ fit["R"].T
    px = (np.stack([Q[:, 0], -Q[:, 1]], 1) * fit["s"] + fit["t2"]) * scale
    z = 10.0 - Q[:, 2]
    pxt, zt = torch.tensor(px, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)
    faces = torch.as_tensor(head.faces)
    zb, fid = rasterize(torch, pxt, zt, faces, H * scale, W * scale)
    bary = pixel_bary(torch, pxt, zt, faces, fid)
    cov = (fid >= 0).numpy()
    ys, xs = np.nonzero(cov)
    f = fid.numpy()[ys, xs]
    uv = (atlas.vt[atlas.ft[f]] * bary.numpy()[ys, xs][..., None]).sum(1)
    S = albedo_bgr.shape[0]
    mx = (uv[:, 0] * S - 0.5).astype(np.float32)
    my = ((1 - uv[:, 1]) * S - 0.5).astype(np.float32)
    col = np.clip(_bilinear(albedo_bgr.astype(np.float32), np.stack([mx, my], 1)), 0, 255).astype(np.uint8)
    img = np.full((H * scale, W * scale, 3), 40, np.uint8)
    img[ys, xs] = col
    return cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_AREA), cv2.resize(cov.astype(np.uint8), (W * 2, H * 2)) > 0


def _fit_sheet(fits, atlas, emb, path: Path, take: str) -> Path:
    rows = []
    for r, (fit, img, img2, mask, mp, head, proj) in fits.items():
        H, W = img.shape[:2]
        lo, hi = mp.min(0), mp.max(0)
        c, side = (lo + hi) / 2, max(hi - lo) * 1.35
        box = (int(c[0] - side / 2), int(c[1] - side / 2), int(c[0] + side / 2), int(c[1] + side / 2))
        T = 360
        sc = T / (box[2] - box[0])
        a = crop_pad(img, box, T)
        for p in fit["lmk_proj"]:
            cv2.circle(a, tuple(np.round((p - box[:2]) * sc).astype(int)), 2, (0, 0, 255), -1)
        for p in fit["lmk_proj"] + fit["residual"]:
            cv2.circle(a, tuple(np.round((p - box[:2]) * sc).astype(int)), 2, (0, 255, 0), -1)
        label(a, "portrait: detected (green) vs fitted avatar landmarks (red)", (4, 14), scale=0.38)
        label(a, f"rms {fit['rms_px']:.1f}px = {fit['rms_iod'] * 100:.1f}% IOD, yaw {fit['yaw_deg']:.0f}", (4, T - 8), scale=0.42)
        mk = crop_pad((np.dstack([mask] * 3) * img2).astype(np.uint8), box, T)
        cv2.polylines(mk, [np.round((extend_forehead(mp, FaceTexConfig().forehead_extend_iod) - box[:2]) * sc)
                           .astype(np.int32)], True, (0, 200, 255), 1)
        label(mk, "mask x portrait (orange: face oval + forehead)", (4, 14), scale=0.4)
        alb = cv2.imread(str(Path(path).parents[1] / "avatar" / "face" / f"{r}_albedo_face.png"))
        bake = cv2.imread(str(Path(path).parents[1] / "avatar" / f"{r}_albedo.png"))
        ren_new, cov = _render_portrait_view(head, fit, atlas, alb, H, W)
        ren_old, _ = _render_portrait_view(head, fit, atlas, bake, H, W)
        box2 = tuple(int(v * 2) for v in box)
        b = crop_pad(ren_old, box2, T)
        label(b, "video bake on the fitted head (portrait view)", (4, 14), scale=0.4)
        n = crop_pad(ren_new, box2, T)
        label(n, "portrait face on the avatar (portrait view)", (4, 14), scale=0.4)
        row = np.concatenate([a, mk, b, n], 1)
        head_strip = np.full((22, row.shape[1], 3), 20, np.uint8)
        label(head_strip, LABEL[r], (6, 16), scale=0.5)
        rows.append(np.concatenate([head_strip, row], 0))
    sheet = np.concatenate(rows, 0)
    top = np.full((28, sheet.shape[1], 3), 10, np.uint8)
    label(top, f"{take}: portrait -> SMPL-X face fit and projection (likeness data, local only)", (6, 19), scale=0.5)
    cv2.imwrite(str(path), np.concatenate([top, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def _uv_sheet(work: Path, atlas, tex: dict, path: Path, take: str) -> Path:
    rows = []
    for r, tx in tex.items():
        bake = cv2.imread(str(tx["bake"]))
        new = cv2.imread(str(tx["portrait"]))
        vid = cv2.imread(str(tx["video"])) if tx.get("video") else None
        mk = np.load(work / "avatar" / "face" / f"{r}_texel_masks.npz")
        yx = atlas.texel_yx[(atlas.region & 1) > 0]
        y0, x0 = yx.min(0)
        y1, x1 = yx.max(0) + 1
        S = atlas.size
        a = np.zeros((S, S), np.uint8)
        a[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]] = (mk["alpha"] * 255).astype(np.uint8)
        al = cv2.applyColorMap(a, cv2.COLORMAP_VIRIDIS)
        rep = mk["repaired"]
        al[atlas.texel_yx[rep, 0], atlas.texel_yx[rep, 1]] = (60, 60, 255)
        for bit, col in ((LAB_LIPS, (255, 0, 255)), (LAB_BEARD, (0, 200, 255)), (LAB_CHEEK, (255, 255, 0))):
            m = (mk["labels"] & bit) > 0
            ov = np.zeros((S, S), np.uint8)
            ov[atlas.texel_yx[m, 0], atlas.texel_yx[m, 1]] = 255
            cnt, _ = cv2.findContours(ov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(al, cnt, -1, col, 2)
        tiles = [bake[y0:y1, x0:x1], new[y0:y1, x0:x1]] + ([vid[y0:y1, x0:x1]] if vid is not None else []) + [al[y0:y1, x0:x1]]
        names = ["video bake (head island)", "portrait look (+ repair, mouth, eyes)"] + \
            (["video-matched look"] if vid is not None else []) + \
            ["portrait alpha; red = skin repair; outlines: lips / beard / cheeks"]
        tiles = [cv2.resize(t, (int(t.shape[1] * 0.55), int(t.shape[0] * 0.55)), interpolation=cv2.INTER_AREA) for t in tiles]
        for t, s_ in zip(tiles, names):
            label(t, s_, (6, 18), scale=0.5)
        row = np.concatenate(tiles, 1)
        hs = np.full((22, row.shape[1], 3), 20, np.uint8)
        label(hs, LABEL[r], (6, 16), scale=0.5)
        rows.append(np.concatenate([hs, row], 0))
    w = max(rw.shape[1] for rw in rows)
    sheet = np.concatenate([np.pad(rw, ((0, 0), (0, w - rw.shape[1]), (0, 0))) for rw in rows], 0)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def _tex_cols(torch, atlas, p: Path, dev):
    im = cv2.imread(str(p))[:, :, ::-1].astype(np.float32) / 255
    return torch.as_tensor(im[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]], device=dev)


def _best_video_view(work: Path, role: str, yaw_target: float, tol: float = 15.0):
    """Usable face_survey view with |yaw - target| < tol and the largest IOD: (cam, i, lmk (52,2), yaw, iod) or None."""
    if not (work / "face" / "survey.npz").exists():
        return None
    z = np.load(work / "face" / "survey.npz")
    best = None
    for k in z.files:
        if not (k.startswith(f"{role}_") and k.endswith("_base")):
            continue
        c = k.split("_")[1]
        base, iod, yaw = z[f"{role}_{c}_base"], z[f"{role}_{c}_iod"], z[f"{role}_{c}_yaw"]
        ok = np.flatnonzero(base & (np.abs(yaw - yaw_target) < tol))
        if len(ok):
            j = ok[np.argmax(iod[ok])]
            if best is None or iod[j] > best[4]:
                best = (c, int(z[f"{role}_{c}_i"][j]), z[f"{role}_{c}_lmk"][j], float(yaw[j]), float(iod[j]))
    return best


def review_angles(take: Take, tex: dict, path: Path, cfg: FaceTexConfig | None = None, tile: int = 300) -> Path:
    """Per dancer, 3 head angles (the portrait's own view, frontal, the other three-quarter side): reference (the
    portrait, else the best video crop at that angle) | avatar with the video bake (old) | portrait look (new) |
    video-matched look. Neutral expression except in the portrait row (the portrait's fitted expression)."""
    import torch

    from .avatar_texture import _Geo
    from .avatar_uv import build_atlas
    from .face_expr import render_front
    from .mamma_dyn import SMPLX_MODELS
    cfg = cfg or FaceTexConfig()
    work = take.work
    atlas = build_atlas(SMPLX_MODELS, 2048)
    dev = "cpu"
    geo = _Geo(capped_atlas(atlas), dev)
    emb = LandmarkEmbedding()
    blocks = []
    for r, tx in tex.items():
        kinds = [k for k in ("bake", "portrait", "video") if tx.get(k)]
        imgs = {k: cv2.imread(str(tx[k])) for k in kinds}
        cols = {k: _tex_cols(torch, atlas, tx[k], dev) for k in kinds}
        d = np.load(work / "smplx" / f"{r}.npz", allow_pickle=True)
        head = HeadModel(np.asarray(d["betas"], np.float32), cfg.n_expr)
        pf = np.load(work / "avatar" / "face" / f"{r}_portrait_fit.npz")
        with torch.no_grad():
            Vp = head.verts(torch.tensor(pf["expr"][:cfg.n_expr], dtype=torch.float32),
                            torch.tensor(pf["jaw"], dtype=torch.float32)).numpy()
            V0 = head.verts(torch.zeros(cfg.n_expr), torch.zeros(3)).numpy().astype(np.float64)
        fit = {"R": pf["R"], "s": float(pf["s"]), "t2": pf["t2"], "verts": Vp}
        cam_yaw = float(np.degrees(np.arctan2(pf["R"][2, 0], pf["R"][2, 2])))
        img = cv2.imread(str(portrait_path(take, r)))
        H, W = img.shape[:2]
        det = portrait_landmarks(take, r)
        mp = det["mp_lmk"][:, :2]
        lo, hi = mp.min(0), mp.max(0)
        c, side = (lo + hi) / 2, max(hi - lo) * 1.5
        box = (int(c[0] - side / 2), int(c[1] - side / 2), int(c[0] + side / 2), int(c[1] + side / 2))
        rows = []
        row = [crop_pad(img, box, tile)]
        label(row[0], f"portrait (camera yaw {cam_yaw:+.0f})", (4, 16), scale=0.45)
        box2 = tuple(int(v * 2) for v in box)
        for k in kinds:
            ren, _ = _render_portrait_view(head, fit, atlas, imgs[k], H, W)
            row.append(crop_pad(ren, box2, tile))
        rows.append(row)
        L0 = emb.points(V0)
        ctr = L0.mean(0) + np.array([0, 0.015, -0.03])
        for yaw in (0.0, -cam_yaw):
            ref = _best_video_view(work, r, yaw)
            if ref is not None:
                fr_img = cv2.imread(str(work / "frames" / ref[0] / f"{ref[1]:06d}.jpg"))
                l_lo, l_hi = ref[2].min(0), ref[2].max(0)
                sd = max(l_hi - l_lo) * 1.9
                cc = (l_lo + l_hi) / 2
                rb = (int(cc[0] - sd / 2), int(cc[1] - sd / 2 - 0.1 * sd), int(cc[0] + sd / 2), int(cc[1] + sd / 2 - 0.1 * sd))
                t0 = crop_pad(fr_img, rb, tile)
                label(t0, f"video cam {ref[0]} #{ref[1]} yaw {ref[3]:+.0f} IOD {ref[4]:.0f}px", (4, 16), scale=0.42)
            else:
                t0 = np.full((tile, tile, 3), 30, np.uint8)
                label(t0, f"no usable video view at yaw {yaw:+.0f}", (4, 16), scale=0.42)
            row = [t0]
            for k in kinds:
                row.append(render_front(torch, geo, V0, cols[k], tile, dev, yaw_deg=yaw, center=ctr, half=0.125))
            rows.append(row)
        names = {"bake": "OLD: video bake", "portrait": "NEW: portrait look", "video": "NEW: video-matched look"}
        for rw, ang in zip(rows, ("portrait view, portrait expression", "front, neutral", "other side, neutral")):
            for t, k in zip(rw[1:], kinds):
                label(t, names[k], (4, 16), scale=0.45)
                label(t, ang, (4, tile - 8), scale=0.4)
        blk = np.concatenate([np.concatenate(rw, 1) for rw in rows], 0)
        hs = np.full((24, blk.shape[1], 3), 20, np.uint8)
        label(hs, LABEL[r], (6, 17), scale=0.55)
        blocks.append(np.concatenate([hs, blk], 0))
    w = max(b.shape[1] for b in blocks)
    sheet = np.concatenate([np.pad(b, ((0, 6), (0, w - b.shape[1]), (0, 0))) for b in blocks], 0)
    top = np.full((30, w, 3), 10, np.uint8)
    label(top, f"{take.name}: reference | avatar face before / after (unlit albedo, no hair strands; likeness data, "
               f"local only)", (6, 20), scale=0.5)
    cv2.imwrite(str(path), np.concatenate([top, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def review_turntable(take: Take, tex: dict, path: Path, tile: int = 210, n: int = 8) -> Path:
    """Head turntable (rest pose, neutral face, unlit): per dancer the old (bake) and new (portrait look) avatar."""
    import torch

    from .avatar_texture import _Geo
    from .avatar_uv import build_atlas
    from .face_expr import render_front
    from .mamma_dyn import SMPLX_MODELS
    work = take.work
    atlas = build_atlas(SMPLX_MODELS, 2048)
    geo = _Geo(capped_atlas(atlas), "cpu")
    emb = LandmarkEmbedding()
    rows = []
    for r, tx in tex.items():
        d = np.load(work / "smplx" / f"{r}.npz", allow_pickle=True)
        head = HeadModel(np.asarray(d["betas"], np.float32), 10)
        with torch.no_grad():
            V0 = head.verts(torch.zeros(10), torch.zeros(3)).numpy().astype(np.float64)
        ctr = emb.points(V0).mean(0) + np.array([0, 0.0, -0.05])
        for k, nm in (("bake", "OLD video bake"), ("portrait", "NEW portrait face")):
            col = _tex_cols(torch, atlas, tx[k], "cpu")
            tiles = []
            for j in range(n):
                yaw = 360.0 * j / n
                t = render_front(torch, geo, V0, col, tile, "cpu", yaw_deg=yaw, pitch_deg=8, center=ctr, half=0.16)
                label(t, f"{yaw:.0f} deg", (4, tile - 8), scale=0.4)
                tiles.append(t)
            row = np.concatenate(tiles, 1)
            label(row, f"{LABEL[r]} - {nm}", (6, 16), scale=0.5)
            rows.append(row)
    sheet = np.concatenate(rows, 0)
    top = np.full((28, sheet.shape[1], 3), 10, np.uint8)
    label(top, f"{take.name}: head turntable, rest pose, neutral face, unlit albedo (Larissa's hair strands are not "
               f"drawn)", (6, 19), scale=0.5)
    cv2.imwrite(str(path), np.concatenate([top, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def review_video(take: Take, tex: dict, path: Path, n: int = 4, tile: int = 240) -> Path:
    """Video crop | avatar with the bake | portrait look | video-matched look, same pose + camera, at the frames where
    each face is seen best (face_survey). Rendered at crop resolution (sharp)."""
    import torch

    from .avatar_texture import _Geo, load_bodies
    from .avatar_uv import build_atlas
    from .face_common import cameras_world
    from .face_expr import render_crop
    from .mamma_dyn import SMPLX_MODELS
    work = take.work
    z = np.load(work / "face" / "survey.npz")
    bodies, _ = load_bodies(work)
    atlas = build_atlas(SMPLX_MODELS, 2048)
    dev = "cpu"
    geo = _Geo(capped_atlas(atlas), dev)
    kinds = ("bake", "portrait", "video")
    cols = {k: {} for k in kinds}
    for r in ROLES:
        tx = tex.get(r) or {"bake": work / "avatar" / f"{r}_albedo.png"}
        for k in kinds:
            p = tx.get(k) or tx.get("portrait") or tx["bake"]
            cols[k][r] = _tex_cols(torch, atlas, p, dev)
    cams = sorted({k.split("_")[1] for k in z.files if k.startswith("lead_") and k.split("_")[1].isdigit()})
    camw = cameras_world(work, cams)
    rows = []
    for r in tex:
        cand = []
        for c in cams:
            if f"{r}_{c}_base" not in z.files:
                continue
            base, iod, ang = z[f"{r}_{c}_base"], z[f"{r}_{c}_iod"], z[f"{r}_{c}_angle"]
            for k in np.flatnonzero(base):
                cand.append((iod[k] * np.cos(np.radians(ang[k])), c, int(z[f"{r}_{c}_i"][k]), float(z[f"{r}_{c}_t"][k]),
                             z[f"{r}_{c}_lmk"][k]))
        cand.sort(key=lambda x: -x[0])
        picked = []
        for cd in cand:
            if all(cd[1] != p[1] or abs(cd[3] - p[3]) > 1.0 for p in picked):
                picked.append(cd)
            if len(picked) == n:
                break
        for sc, c, i, t, lm in picked:
            img = cv2.imread(str(work / "frames" / c / f"{i:06d}.jpg"))
            ci = camw[c]
            posed = {rr: b.at(t)[0] for rr, b in bodies.items()}
            lo, hi = lm.min(0), lm.max(0)
            side = max(hi - lo) * 2.3
            cxy = (lo + hi) / 2
            box = (int(cxy[0] - side / 2), int(cxy[1] - side / 2), int(cxy[0] + side / 2), int(cxy[1] + side / 2))
            tiles = [crop_pad(img, box, tile)]
            with torch.no_grad():
                for k in kinds:
                    tiles.append(render_crop(torch, geo, posed, cols[k], ci["K"][i], ci["R"][i], ci["t"][i], ci["k1"][i],
                                             box, tile, dev))
            label(tiles[0], f"cam {c} #{i} IOD {np.linalg.norm(lm[1 + 19] - lm[1 + 28]):.0f}px", (4, 14), scale=0.42)
            for tl, nm in zip(tiles[1:], ("OLD video bake", "NEW portrait look", "NEW video-matched look")):
                label(tl, nm, (4, 14), scale=0.42)
            row = np.concatenate(tiles, 1)
            hs = np.full((20, row.shape[1], 3), 20, np.uint8)
            label(hs, f"{LABEL[r]}  t={t:.2f}s", (6, 15), scale=0.45)
            rows.append(np.concatenate([hs, row], 0))
    w = max(rw.shape[1] for rw in rows)
    half = (len(rows) + 1) // 2
    colA = np.concatenate([np.pad(rw, ((0, 3), (0, w - rw.shape[1]), (0, 0))) for rw in rows[:half]], 0)
    colB = np.concatenate([np.pad(rw, ((0, 3), (0, w - rw.shape[1]), (0, 0))) for rw in rows[half:]], 0)
    colB = np.pad(colB, ((0, colA.shape[0] - colB.shape[0]), (0, 0), (0, 0)))
    sheet = np.concatenate([colA, np.zeros((colA.shape[0], 8, 3), np.uint8), colB], 1)
    top = np.full((28, sheet.shape[1], 3), 10, np.uint8)
    label(top, f"{take.name}: photo | avatar textures, same pose + camera (unlit; body fit as exported, neutral face)",
          (6, 19), scale=0.5)
    cv2.imwrite(str(path), np.concatenate([top, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def likeness(take: Take, tex: dict, path: Path | None = None, n: int = 8, min_iod: float = 15.0,
             max_angle: float = 50.0, blur_px: float = 0.7) -> dict:
    """Likeness check against the video: at the n best face views per dancer (face_survey: IOD >= min_iod px, within
    max_angle of frontal, >= 0.3 s apart) the avatar is rendered at the video's own resolution (1 px = 1 px, same pose
    + camera, blurred blur_px like the footage) with each texture variant, and compared over the face (the convex hull
    of the 52 detected landmarks, avatar pixels only) after a per-channel least-squares gain (lighting is not
    modelled): gain-fitted mean abs error (sRGB 0-255, lower = closer) and the luminance correlation NCC (structure,
    higher = closer). Twice: 'posed' (the render as the body fit places the head) and 'aligned' (the render first moved
    by the 2D similarity that maps its 52 projected landmarks onto the detected ones: the body fit's head-pose error,
    15-25 % of these small faces, would otherwise dominate - 'aligned' is the appearance / likeness number). The video
    bake was built FROM these frames with the posed geometry (in-sample for it, especially 'posed'); the portrait
    textures never saw them (their colour transfer uses the bake's average lighting only).
    -> {role: {variants: {variant: {posed: {...}, aligned: {...}}}, views: [...]}}; optional sheet (aligned): crop |
    each variant after the gain (x4)."""
    import torch

    from .avatar_texture import _Geo, load_bodies, render
    from .avatar_uv import build_atlas
    from .face_common import cameras_world, project
    from .face_expr import _crop_camera
    from .mamma_dyn import SMPLX_MODELS
    work = take.work
    if not (work / "face" / "survey.npz").exists() or not (work / "face" / "detections.npz").exists():
        return {"skipped": "no face/survey.npz + detections.npz (face_survey / face_worker)"}
    bodies, _ = load_bodies(work)
    atlas = build_atlas(SMPLX_MODELS, 2048)
    geo = _Geo(capped_atlas(atlas), "cpu")
    prev = sorted((work / "avatar" / "face").glob("prev_*"))
    out, sheet_rows = {}, []
    from .face_eval import load as load_detections
    fd = load_detections(work)
    cams = sorted({str(c) for c in fd["cam"]})
    camw = cameras_world(work, cams)
    emb = LandmarkEmbedding()

    def score(ren, rid, vid, hull):
        m = hull & (rid == 0)
        if m.sum() < 30:
            return None
        rl, vl = _lin(np.clip(ren[m], 0, 1)), _lin(vid[m])
        g = (rl * vl).sum(0) / np.maximum((rl * rl).sum(0), 1e-9)
        fitted = _srgb(np.clip(rl * g, 0, 1))
        Yr, Yv = fitted @ np.array([0.2126, 0.7152, 0.0722]), vid[m] @ np.array([0.2126, 0.7152, 0.0722])
        ncc = float(np.corrcoef(Yr, Yv)[0, 1]) if Yr.std() > 1e-6 and Yv.std() > 1e-6 else 0.0
        show = np.full_like(vid, 40 / 255)
        show[m] = fitted
        return float(np.abs(fitted - vid[m]).mean() * 255), ncc, show
    for r in tex:
        variants = {"bake": tex[r]["bake"]}
        if prev and (prev[-1] / f"{r}_albedo_face.png").exists():
            variants[f"previous face ({prev[-1].name})"] = prev[-1] / f"{r}_albedo_face.png"
        variants["portrait look"] = tex[r]["portrait"]
        if tex[r].get("video"):
            variants["video-matched look"] = tex[r]["video"]
        other = [o for o in ROLES if o != r][0]
        cols = {k: _tex_cols(torch, atlas, p, "cpu") for k, p in variants.items()}
        col_o = _tex_cols(torch, atlas, work / "avatar" / f"{other}_albedo.png", "cpu")
        cand = []
        # the DETECTED landmarks (MediaPipe -> iBUG 52, face_worker / face_eval), where the face really is in the
        # photo; views whose detection disagrees with the fitted head by > 0.25 IOD (a wrong face / a hand) are skipped
        sel = np.flatnonzero((fd["role"] == r) & fd["base"] & fd["mp_ok"] & (fd["iod"] >= min_iod) &
                             (fd["angle"] <= max_angle))
        for n_ in sel:
            raw = np.nanmean(np.linalg.norm(fd["model52"][n_] - fd["mp52"][n_], axis=-1)) / max(float(fd["iod"][n_]), 1)
            if not np.isfinite(raw) or raw > 0.25 or str(fd["cam"][n_]) not in camw:
                continue
            cand.append((float(fd["iod"][n_] * np.cos(np.radians(fd["angle"][n_]))), str(fd["cam"][n_]), int(fd["i"][n_]),
                         float(fd["t"][n_]), np.asarray(fd["mp52"][n_], float), float(fd["iod"][n_])))
        cand.sort(key=lambda x: -x[0])
        picked = []
        for cd in cand:
            if all(abs(cd[3] - p[3]) >= 0.3 for p in picked):
                picked.append(cd)
            if len(picked) == n:
                break
        res = {k: {m_: {"mae": [], "ncc": []} for m_ in ("posed", "aligned")} for k in variants}
        views = []
        for sc, c, i, t, lm, iod in picked:
            img = cv2.imread(str(work / "frames" / c / f"{i:06d}.jpg"))
            ci = camw[c]
            pr, po = bodies[r].at(t), bodies[other].at(t)
            if img is None or pr is None:
                continue
            lo, hi = lm.min(0), lm.max(0)
            side = int(np.ceil(max(hi - lo) * 1.6))
            cxy = (lo + hi) / 2
            box = (int(round(cxy[0] - side / 2)), int(round(cxy[1] - side / 2)))
            box = (box[0], box[1], box[0] + side, box[1] + side)
            vid = crop_pad(img, box)[:, :, ::-1].astype(np.float32) / 255
            lm_c = lm - np.array(box[:2], float)
            hull = np.zeros((side, side), np.uint8)
            cv2.fillConvexPoly(hull, cv2.convexHull(np.round(lm_c).astype(np.int32)), 1)
            hull = cv2.erode(hull, np.ones((3, 3), np.uint8)) > 0
            Kc = _crop_camera(ci["K"][i], box, side)
            lm_r, _ = project(emb.points(np.asarray(pr[0], float)), Kc, np.asarray(ci["R"][i], float),
                              np.asarray(ci["t"][i], float), float(ci["k1"][i]))
            A, _ = cv2.estimateAffinePartial2D(lm_r.astype(np.float32), lm_c.astype(np.float32), method=cv2.LMEDS)
            posed = {r: pr[0]} if po is None else {r: pr[0], other: po[0]}
            vrec = {"cam": c, "i": i, "t": round(t, 3), "iod": round(iod, 1),
                    "align_px": None if A is None else round(float(np.abs(lm_r @ A[:, :2].T + A[:, 2] - lm_c).mean()), 2)}
            row, ok, scores = [vid], True, {}
            for k in variants:
                with torch.no_grad():
                    ren, rid = render(torch, geo, posed, {r: cols[k], other: col_o}, Kc, ci["R"][i], ci["t"][i],
                                      float(ci["k1"][i]), side, side, "cpu")
                ren = np.asarray(ren, np.float32)
                rid = np.asarray(rid)
                if blur_px > 0:
                    ren = cv2.GaussianBlur(ren, (0, 0), blur_px)
                sp_ = score(ren, rid, vid, hull)
                sa_ = None
                if A is not None:
                    ren_a = cv2.warpAffine(ren, A, (side, side), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
                    rid_a = cv2.warpAffine(rid.astype(np.float32), A, (side, side), flags=cv2.INTER_NEAREST,
                                           borderValue=-1).astype(np.int64)
                    sa_ = score(ren_a, rid_a, vid, hull)
                if sp_ is None or sa_ is None:
                    ok = False
                    break
                scores[k] = (sp_, sa_)
            if not ok:
                continue
            for k, (sp_, sa_) in scores.items():
                for m_, sc_ in (("posed", sp_), ("aligned", sa_)):
                    res[k][m_]["mae"].append(sc_[0])
                    res[k][m_]["ncc"].append(sc_[1])
                vrec[k] = {"posed": {"mae": round(sp_[0], 2), "ncc": round(sp_[1], 3)},
                           "aligned": {"mae": round(sa_[0], 2), "ncc": round(sa_[1], 3)}}
                row.append(sa_[2])
            views.append(vrec)
            if path is not None:
                tiles = []
                for j, im in enumerate(row):
                    u8 = cv2.resize((np.clip(im, 0, 1)[:, :, ::-1] * 255).astype(np.uint8), (side * 4, side * 4),
                                    interpolation=cv2.INTER_NEAREST)
                    u8 = cv2.resize(u8, (200, 200), interpolation=cv2.INTER_AREA)
                    if j == 0:
                        label(u8, f"cam {c} #{i} IOD {iod:.0f}", (3, 13), scale=0.38)
                    else:
                        k = list(variants)[j - 1]
                        label(u8, k[:24], (3, 13), scale=0.36)
                        label(u8, f"MAE {vrec[k]['aligned']['mae']:.1f} NCC {vrec[k]['aligned']['ncc']:.2f}", (3, 194),
                              scale=0.38)
                    tiles.append(u8)
                hs = np.full((18, 200 * len(tiles), 3), 20, np.uint8)
                label(hs, f"{LABEL[r]} t={t:.2f}s (aligned on the landmarks)", (4, 13), scale=0.4)
                sheet_rows.append(np.concatenate([hs, np.concatenate(tiles, 1)], 0))
        med = lambda v: round(float(np.median(v)), 3) if v else None
        summ = {k: {m_: {"mae_median": med(v[m_]["mae"]), "ncc_median": med(v[m_]["ncc"]), "views": len(v[m_]["mae"])}
                    for m_ in v} for k, v in res.items()}
        for k, v in res.items():
            if k == "bake":
                continue
            for m_ in v:
                base = res["bake"][m_]["mae"]
                if v[m_]["mae"] and len(v[m_]["mae"]) == len(base):
                    summ[k][m_]["views_closer_than_bake"] = int(sum(a_ < b_ for a_, b_ in zip(v[m_]["mae"], base)))
        out[r] = {"variants": summ, "views": views}
    if path is not None and sheet_rows:
        w = max(x.shape[1] for x in sheet_rows)
        sheet = np.concatenate([np.pad(x, ((0, 3), (0, w - x.shape[1]), (0, 0))) for x in sheet_rows], 0)
        top = np.full((44, w, 3), 10, np.uint8)
        label(top, f"{take.name}: likeness vs the video - photo crop | avatar rendered at the photo's resolution with each "
                   f"texture, moved onto the detected landmarks, after a per-channel gain", (6, 17), scale=0.42)
        label(top, "MAE = mean abs error over the face (0-255, lower = closer); NCC = luminance correlation (higher = "
                   "closer). The bake was built from these frames (in-sample).", (6, 36), scale=0.42)
        cv2.imwrite(str(path), np.concatenate([top, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
        out["review"] = str(path)
    return out


def review_only(take: Take, roles=ROLES, bakes: dict | None = None) -> int:
    """Redraw the review sheets (UV, angles, turntable, video, likeness) from the textures already written."""
    from .avatar_uv import build_atlas
    from .mamma_dyn import SMPLX_MODELS
    work = take.work
    rev = work / "review"
    tex = {}
    for r in roles:
        b = Path(bakes[r]) if bakes and bakes.get(r) else work / "avatar" / f"{r}_albedo.png"
        o = face_outputs(b)
        tex[r] = {"bake": b, "portrait": o["portrait"], **({"video": o["video"]} if o["video"].exists() else {})}
    atlas = build_atlas(SMPLX_MODELS, 2048)
    paths = [_uv_sheet(work, atlas, tex, rev / "avatar_face_uv.jpg", take.name),
             review_angles(take, tex, rev / "avatar_face_portrait.jpg"), review_turntable(take, tex, rev / "avatar_face_turntable.jpg")]
    if (work / "face" / "survey.npz").exists():
        paths.append(review_video(take, tex, rev / "avatar_face_video.jpg"))
        lk = likeness(take, tex, rev / "avatar_face_likeness.jpg")
        rp = work / "avatar" / "face" / "report.json"
        if rp.exists():
            rep = json.loads(rp.read_text())
            rep["likeness"] = lk
            rp.write_text(json.dumps(rep, indent=1, default=float))
        print("[avatar_face] likeness: " + json.dumps({r: v["variants"] for r, v in lk.items() if isinstance(v, dict) and "variants" in v}))
    print(json.dumps([str(p) for p in paths], indent=1))
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--roles", nargs="*", default=list(ROLES))
    ap.add_argument("--bake", action="append", default=[], metavar="ROLE=PNG",
                    help="composite onto this baked albedo instead of avatar/<role>_albedo.png (output next to it)")
    ap.add_argument("--composite", action="store_true", help="reuse the saved portrait projection (skip the fit)")
    ap.add_argument("--no-eyes", action="store_true")
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--review-only", action="store_true", help="redraw the review sheets + likeness from the written textures")
    a = ap.parse_args(argv)
    bakes = {}
    for b in a.bake:
        role, _, p = b.partition("=")
        if role not in ROLES or not p:
            raise SystemExit(f"--bake expects ROLE=PNG with ROLE in {ROLES}, got {b!r}")
        bakes[role] = p
    roles = tuple(a.roles)
    if a.review_only:
        return review_only(Take.load(a.take), roles, bakes)
    rep = run(Take.load(a.take), roles, composite_only=a.composite, eyes=not a.no_eyes, bakes=bakes, review=not a.no_review)
    print(json.dumps({r: v.get("outputs") for r, v in rep["roles"].items()}, indent=1))
    print(json.dumps(rep.get("review"), indent=1))


if __name__ == "__main__":
    sys.exit(main())
