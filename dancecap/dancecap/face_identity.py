"""Dancer face identity: a FLAME 2020 identity shape fitted to a dancer's photos, transplanted onto the exported
SMPL-X rest head (shape_edits face block `identity = ...`; applied by export_unity.export on every export).

Why: MAMMA's locked-head SMPL-X keeps the template face (scaled with the head), so every avatar has the same generic
face. This module estimates the dancer's own FLAME identity (300 shape coefficients) from her portrait + the clearest
video faces and moves the avatar's lower face / cheeks / jaw / chin / under-chin to it. Only the rest vertices of
the follow's skin move (before the UV split): UVs, LBS weights, joints and motion are untouched, and the eyes, brows,
forehead, scalp (hair roots), ears and the neck seam stay exactly where they are (the face texture keeps lining up).

1. FIT (one-off per dancer, `python -m dancecap.face_identity fit <landmarks.json> <identity.npz>`): per image the
   SMIRK encoder (external/smirk, MICA-distilled shape branch) gives FLAME shape / expression / jaw / head pose;
   then a 2D landmark fit refines every image (shape coefficients with a prior around SMIRK's, expression, jaw,
   head rotation, orthographic camera) against MediaPipe (SMIRK's 105-point embedding) and the jaw contour (iBUG
   0-16, FLAME's yaw-dependent contour embedding, RTMW + MediaPipe). The per-image shape codes are combined robustly
   (inliers by fit residual and by distance to the weighted median, portrait weighted x 3): the identity = that
   shape with ZERO expression and jaw (neutral, closed mouth). landmarks.json: [{name, ok, mp478 (x, y, z px of the
   image), rtmw68, rtmw68_score, smirk_center, smirk_side, file}, ...] (review tool detect_lm.py, .venv-face).
2. TRANSPLANT (`displacement`): the rest head's FLAME-topology vertices (SMPL-X__FLAME_vertex_ids) are fitted by a
   FLAME head C = s R FLAME(beta_cur) + t (its "FLAME equivalent"), the identity head T = FLAME(beta_id) is aligned to
   FLAME(beta_cur) by a similarity on the stable regions (eye region, nose bridge, forehead: keeps the outer-canthal
   width and the eye positions), and D = T_aligned - FLAME(beta_cur) is mapped back (s R) and blended with the
   transplant mask (`transplant_mask`: FLAME topology, 0 on the eye region / forehead / scalp / ears / eyeballs / nose /
   nape / neck seam, 1 on the lower face below the subnasale and the under-chin, but only where that is >= 3.8 cm of
   surface from the ears / seam / eyes and >= 1.2 cm from the nose; harmonic in between, then smoothstep: C1, no fold
   at the ear root, no step at the nostrils). The neck is then re-interpolated (`neck_extension`: biharmonic extension
   of the face's move down to the seam, so the throat carries no FLAME-prior relief) and `mouth_weight` (1 on the lips,
   0 at 14 mm) is returned beside the displacement for the face block's `identity_mouth`. The displacement is
   re-targeted to every take's own rest head (02 has other betas than 04).
3. CACHE: FLAME is git-ignored and not installed everywhere, so each take keeps its retargeted displacement (+ the
   mouth weight) in work/<take>/face_shape/<role>_identity_disp.npz with the rest head (matched to 0.05 mm, wherever it
   stands), the identity file and the mask parameters (MASK_DEFAULT, CACHE_VERSION); export uses the cache when they match and
   recomputes it (FLAME needed) otherwise.
   `python -m dancecap.face_identity retarget <take>` (re)builds it.

Needs (git-ignored, never redistributed): models/flame/FLAME2020/generic_model.pkl, models/flame/
SMPL-X__FLAME_vertex_ids.npy, external/smirk (assets: landmark embeddings, FLAME masks; pretrained SMIRK_em1.pt for
`fit` only). The identity file (shape code + fit report) is dancer likeness data: keep it under dancecap/work.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE

FLAME_PKL = WORKSPACE / "models" / "flame" / "FLAME2020" / "generic_model.pkl"
SX_FLAME_IDS = WORKSPACE / "models" / "flame" / "SMPL-X__FLAME_vertex_ids.npy"
SMIRK_DIR = WORKSPACE / "external" / "smirk"
N_SHAPE, N_EXP = 300, 50
N_SX = 10475
# MediaPipe 478 -> iBUG-68 (face_common.MP_TO_IBUG68), contour part
MP_CONTOUR = np.array([162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389])
CACHE_VERSION = 5   # 2: mask with surface clearance from the ears / seam / nose; 3: biharmonic neck extension; 4: + mouth_w;
                    # 5: head matched by shape (translation-invariant, 0.05 mm), not by a hash


# ------------------------------------------------------------------------------------------------ FLAME (numpy)

class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.__dict__.update(s if isinstance(s, dict) else {"state": s})


class _NoChumpy(pickle.Unpickler):
    def find_class(self, mod, name):
        return _Stub if mod.startswith("chumpy") else super().find_class(mod, name)


def flame_available() -> bool:
    return FLAME_PKL.exists() and SX_FLAME_IDS.exists() and (SMIRK_DIR / "assets" / "FLAME_masks").exists()


@lru_cache(maxsize=1)
def flame() -> dict:
    """FLAME 2020 generic: template, shape (300) / expression (100) dirs, jaw skinning, joints regressor, faces,
    landmark embeddings and the FLAME region masks (SMIRK assets)."""
    with open(FLAME_PKL, "rb") as fh:
        d = _NoChumpy(fh, encoding="latin1").load()
    arr = lambda v: np.asarray(v.__dict__.get("x")) if isinstance(v, _Stub) else (  # noqa: E731
        np.asarray(v.todense()) if hasattr(v, "todense") else np.asarray(v))
    F = {k: arr(d[k]).astype(np.float64) for k in ("v_template", "shapedirs", "J_regressor", "weights")}
    F["f"] = arr(d["f"]).astype(np.int64)
    a = SMIRK_DIR / "assets"
    lm = np.load(a / "landmark_embedding.npy", allow_pickle=True, encoding="latin1")[()]
    F["lmk_full_f"] = np.asarray(lm["full_lmk_faces_idx"])[0].astype(np.int64)
    F["lmk_full_b"] = np.asarray(lm["full_lmk_bary_coords"])[0].astype(np.float64)
    F["lmk_dyn_f"] = np.asarray(lm["dynamic_lmk_faces_idx"]).astype(np.int64)
    F["lmk_dyn_b"] = np.asarray(lm["dynamic_lmk_bary_coords"]).astype(np.float64)
    mp = np.load(a / "mediapipe_landmark_embedding" / "mediapipe_landmark_embedding.npz")
    F["mp_f"], F["mp_b"], F["mp_idx"] = (mp["lmk_face_idx"].astype(np.int64), mp["lmk_b_coords"].astype(np.float64),
                                         mp["landmark_indices"].astype(np.int64))
    F["leye"], F["reye"] = np.load(a / "l_eyelid.npy").astype(np.float64), np.load(a / "r_eyelid.npy").astype(np.float64)
    with open(a / "FLAME_masks" / "FLAME_masks.pkl", "rb") as fh:
        F["masks"] = {k: np.asarray(v, np.int64) for k, v in pickle.load(fh, encoding="latin1").items()}
    F["sx_ids"] = np.load(SX_FLAME_IDS).astype(np.int64)
    return F


def flame_shape(beta: np.ndarray, F: dict | None = None) -> np.ndarray:
    """(5023, 3) neutral FLAME head (zero expression / pose) for shape coefficients beta (<= 300)."""
    F = flame() if F is None else F
    b = np.asarray(beta, float).reshape(-1)
    return F["v_template"] + F["shapedirs"][:, :, :len(b)] @ b


def flame_expression(V: np.ndarray, psi: np.ndarray, jaw: np.ndarray, eyelid=None, F: dict | None = None) -> np.ndarray:
    """Neutral head V (5023,3) + expression psi (<= 100) + jaw rotation (axis-angle) (+ eyelids) -> (5023,3)."""
    from scipy.spatial.transform import Rotation
    F = flame() if F is None else F
    p = np.asarray(psi, float).reshape(-1)
    W = V + F["shapedirs"][:, :, 300:300 + len(p)] @ p
    if eyelid is not None:
        W = W + eyelid[0] * F["leye"] + eyelid[1] * F["reye"]
    J = F["J_regressor"] @ W
    R = Rotation.from_rotvec(np.asarray(jaw, float)).as_matrix()
    wj = F["weights"][:, 2:3]
    return W + wj * ((W - J[2]) @ R.T + J[2] - W)


def lmk68(V: np.ndarray, F: dict | None = None) -> np.ndarray:
    F = flame() if F is None else F
    return np.einsum("lkd,lk->ld", V[F["f"][F["lmk_full_f"]]], F["lmk_full_b"])


# ------------------------------------------------------------------------------------------------ alignment

def similarity(A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None, scale: bool = True):
    """(a, Q, b) minimising sum w |a Q A_i + b - B_i|^2 (Umeyama)."""
    w = np.ones(len(A)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ma, mb = w @ A, w @ B
    Ac, Bc = A - ma, B - mb
    C = (Bc * w[:, None]).T @ Ac
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    Q = U @ D @ Vt
    a = float(np.trace(np.diag(S) @ D) / (w @ (Ac ** 2).sum(1))) if scale else 1.0
    return a, Q, mb - a * Q @ ma


def scale_translation(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray]:
    """(a, b) minimising sum |a A_i + b - B_i|^2 (no rotation)."""
    ma, mb = A.mean(0), B.mean(0)
    a = float(((A - ma) * (B - mb)).sum() / ((A - ma) ** 2).sum())
    return a, mb - a * ma


def _fit_weights(F: dict) -> np.ndarray:
    """Per-vertex weights for fitting a FLAME head to a SMPL-X head (no eyeballs; face x 2)."""
    w = np.ones(len(F["v_template"]))
    w[F["masks"]["face"]] = 2.0
    for k in ("left_eyeball", "right_eyeball"):
        w[F["masks"][k]] = 0.0
    return w


def fit_equivalent(H: np.ndarray, n_shape: int = 300, lam: float = 1e-4, iters: int = 12, F: dict | None = None) -> dict:
    """FLAME equivalent of a (5023,3) FLAME-topology head H: H ~ s R FLAME(beta) + t (similarity + ridge on beta).
    -> {beta, s, R, t, C (canonical FLAME(beta)), rms_mm, max_mm}."""
    F = flame() if F is None else F
    vt, S = F["v_template"], F["shapedirs"][:, :, :n_shape]
    w = _fit_weights(F)
    sw = np.sqrt(w)[:, None, None]
    A = (S * sw).reshape(-1, n_shape)
    beta = np.zeros(n_shape)
    for _ in range(iters):
        C = vt + S @ beta
        s, R, t = similarity(C, H, w)
        Hc = ((H - t) @ R) / s                       # H mapped into the canonical frame
        b = ((Hc - vt) * sw[:, :, 0]).reshape(-1)
        beta = np.linalg.solve(A.T @ A + lam * np.eye(n_shape), A.T @ b)
    C = vt + S @ beta
    s, R, t = similarity(C, H, w)
    res = np.linalg.norm(s * C @ R.T + t - H, axis=1)[w > 0]
    return {"beta": beta, "s": s, "R": R, "t": t, "C": C, "rms_mm": float(np.sqrt((res ** 2).mean()) * 1000),
            "max_mm": float(res.max() * 1000)}


def stable_ids(F: dict | None = None) -> np.ndarray:
    """FLAME vertices that anchor the identity (eye region, forehead, nose bridge above the nose tip)."""
    F = flame() if F is None else F
    vt = F["v_template"]
    L = lmk68(vt, F)
    nose = F["masks"]["nose"]
    bridge = nose[vt[nose, 1] > 0.5 * (L[27, 1] + L[30, 1])]
    return np.unique(np.concatenate([F["masks"]["eye_region"], F["masks"]["forehead"], bridge]))


# ------------------------------------------------------------------------------------------------ transplant mask

def _cot_laplacian(V: np.ndarray, f: np.ndarray):
    import scipy.sparse as sp
    n = len(V)
    I, J, W = [], [], []
    for k in range(3):
        a, b, c = f[:, k], f[:, (k + 1) % 3], f[:, (k + 2) % 3]
        u, v = V[b] - V[a], V[c] - V[a]
        cot = (u * v).sum(1) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-12)
        cot = np.clip(cot, 1e-3, 1e3) * 0.5
        I += [b, c]
        J += [c, b]
        W += [cot, cot]
    Wm = sp.coo_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr()
    return sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm


MASK_DEFAULT = {"throat_drop_m": 0.07, "nape_back_m": 0.02, "clear_m": 0.038, "nose_clear_m": 0.012,
                "neck_half_width_m": 0.040, "neck_extension": "biharmonic",
                "mouth_blend_m": 0.014}


def _geodesic(V: np.ndarray, f: np.ndarray, src: np.ndarray) -> np.ndarray:
    """(n,) surface (edge-graph) distance from the vertex set `src` (bool mask)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    g = sp.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V), len(V))).tocsr()
    return dijkstra(g, directed=False, indices=np.nonzero(src)[0], min_only=True)


def _sets(throat_drop_m: float, nape_back_m: float) -> dict:
    """FLAME-topology vertex sets of the transplant: zero (protected: eye region, forehead, scalp, eyeballs, ears, the
    neck seam ring, the nape, the neck lower than throat_drop_m under the menton), neck (all 'neck' vertices) and
    the landmark heights."""
    F = flame()
    vt, M = F["v_template"], F["masks"]
    L = lmk68(vt, F)
    y_me = L[8, 1]
    z_ear = 0.5 * (L[0, 2] + L[16, 2])
    zero = np.zeros(len(vt), bool)
    for k in ("eye_region", "forehead", "scalp", "left_eyeball", "right_eyeball", "left_ear", "right_ear", "boundary"):
        zero[M[k]] = True
    neck = np.zeros(len(vt), bool)
    neck[M["neck"]] = True
    zero |= neck & (vt[:, 2] < z_ear - nape_back_m)
    zero |= neck & (vt[:, 1] < y_me - throat_drop_m)
    return {"zero": zero, "neck": neck, "y_sn": L[33, 1], "y_me": y_me, "z_go": 0.5 * (L[4, 2] + L[12, 2])}


@lru_cache(maxsize=4)
def transplant_mask(throat_drop_m: float = MASK_DEFAULT["throat_drop_m"],
                    nape_back_m: float = MASK_DEFAULT["nape_back_m"], clear_m: float = MASK_DEFAULT["clear_m"],
                    nose_clear_m: float = MASK_DEFAULT["nose_clear_m"],
                    neck_half_width_m: float = MASK_DEFAULT["neck_half_width_m"]) -> np.ndarray:
    """(5023,) weights in [0, 1] on FLAME topology (template frame: y up, z forward).
    0: eye region, forehead, scalp, eyeballs, ears, the neck seam ring (FLAME 'boundary'), the nape (neck behind the
    ear plane - nape_back_m), the neck lower than throat_drop_m under the menton and the whole nose; 1: the lower
    face (face / lips below the subnasale) and the under-chin (the neck within neck_half_width_m of the midline and
    1.2 cm in front of the jaw-angle plane, down to 1.2 cm under the menton: a patch with no straight lateral edge),
    but only where it is at least clear_m (surface distance) from the ears / scalp / seam / eyes and nose_clear_m
    from the nose (the transition between the two sets is therefore never shorter than that: no fold-overs of the
    cheek skin against the ear, no step at the nostrils); harmonic in between (cotan Laplace), then smoothstep (C1 at
    both constraint sets). The neck itself is re-interpolated by neck_extension() in displacement()."""
    import scipy.sparse.linalg as spla
    F = flame()
    vt, f, M = F["v_template"], F["f"], F["masks"]
    st = _sets(throat_drop_m, nape_back_m)
    zero, neck = st["zero"].copy(), st["neck"]
    nose = np.zeros(len(vt), bool)
    nose[M["nose"]] = True
    one = np.zeros(len(vt), bool)
    one[np.concatenate([M["face"], M["lips"]])] = True
    one &= vt[:, 1] < st["y_sn"]
    one |= neck & (vt[:, 2] > st["z_go"] + 0.012) & (np.abs(vt[:, 0]) < neck_half_width_m) & (vt[:, 1] > st["y_me"] - 0.012)
    one &= ~zero & ~nose
    one &= _geodesic(vt, f, zero) >= clear_m
    one &= _geodesic(vt, f, nose) >= nose_clear_m
    zero |= nose
    Lp = _cot_laplacian(vt, f).tocsr()
    fixed = zero | one
    free = ~fixed
    x = np.where(one, 1.0, 0.0)
    rhs = -Lp[free][:, fixed] @ x[fixed]
    x[free] = spla.spsolve(Lp[free][:, free].tocsc(), rhs)
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def neck_extension(D: np.ndarray, throat_drop_m: float = MASK_DEFAULT["throat_drop_m"],
                   nape_back_m: float = MASK_DEFAULT["nape_back_m"]) -> np.ndarray:
    """(5023,3) displacement with the neck re-interpolated: every 'neck' vertex that is not protected (zero set) is
    replaced by the BIHARMONIC extension (thin-plate, cotan Laplacian L and lumped mass M: minimise |M^-1 L d|^2, so
    value AND slope are continuous with the face vertices it starts from and with the zero ring / nape it ends on)
    of the displacement D of all other vertices. The photos constrain the face down to the jaw line, not the neck:
    the FLAME shape prior would otherwise print its own larynx / sterno-cleido-mastoid relief into the neck as a
    ledge along the front of the throat; the extension carries the chin's move down the neck and fades it to nothing
    at the seam without any crease."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    F = flame()
    vt, f = F["v_template"], F["f"]
    st = _sets(throat_drop_m, nape_back_m)
    free = st["neck"] & ~st["zero"]
    fixed = ~free
    L = _cot_laplacian(vt, f).tocsr()
    T = vt[f]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    mass = np.zeros(len(vt))
    for k in range(3):
        np.add.at(mass, f[:, k], area / 3.0)
    Q = (L.T @ sp.diags(1.0 / np.maximum(mass, 1e-12)) @ L).tocsr()
    out = np.array(D, float, copy=True)
    lu = spla.splu(Q[free][:, free].tocsc())
    for c in range(3):
        out[free, c] = lu.solve(-(Q[free][:, fixed] @ D[fixed, c]))
    return out


def mouth_weight(blend_m: float = MASK_DEFAULT["mouth_blend_m"]) -> np.ndarray:
    """(5023,) 1 on the lips (FLAME 'lips' mask) -> 0 blend_m (surface distance) away, smoothstep (C1). Lets the face
    block keep part of the dancer's own mouth where it is (shape_edits `identity_mouth`): the identity moves the whole
    mouth a few mm with the lower face, which squeezes the philtrum between the fixed nose and the moved lips."""
    F = flame()
    lips = np.zeros(len(F["v_template"]), bool)
    lips[F["masks"]["lips"]] = True
    t = np.clip(_geodesic(F["v_template"], F["f"], lips) / blend_m, 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------------------------------------ displacement

def head_ids() -> np.ndarray:
    return flame()["sx_ids"]


def displacement(v_sx: np.ndarray, beta_id: np.ndarray, weight: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> (disp (10475,3), mouth_w (10475,), report): the rest-vertex displacement of an SMPL-X head toward the identity
    beta_id (see the module docstring) and the per-vertex weight of the mouth (mouth_weight)."""
    F = flame()
    ids = F["sx_ids"]
    H = np.asarray(v_sx, float)[ids]
    eq = fit_equivalent(H, F=F)
    T = flame_shape(beta_id, F)
    st = stable_ids(F)
    # both heads are canonical FLAME heads (same frame): scale + translation only on the stable regions. A rotation
    # fitted to the small eye / forehead / bridge patch is poorly conditioned in pitch and would swing the chin and
    # the under-chin by centimetres.
    a, b = scale_translation(T[st], eq["C"][st])
    Ta = a * T + b
    D = Ta - eq["C"]
    m = transplant_mask()
    Dw = eq["s"] * D @ eq["R"].T
    Dm = neck_extension(m[:, None] * Dw)
    out = np.zeros((len(v_sx), 3))
    out[ids] = weight * Dm
    mw = np.zeros(len(v_sx))
    mw[ids] = mouth_weight()
    Lc, La = lmk68(eq["C"], F), lmk68(Ta, F)
    rep = {"equivalent_rms_mm": round(eq["rms_mm"], 3), "equivalent_max_mm": round(eq["max_mm"], 3),
           "equivalent_scale": round(eq["s"], 5), "align_scale": round(a, 5),
           "stable_residual_mm": round(float(np.sqrt((np.linalg.norm(Ta[st] - eq["C"][st], axis=1) ** 2).mean())) * 1000
                                       * eq["s"], 3),
           "occ_mm_head_vs_identity": [round(float(np.linalg.norm(Lc[36] - Lc[45])) * 1000 * eq["s"], 2),
                                       round(float(np.linalg.norm(La[36] - La[45])) * 1000 * eq["s"], 2)],
           "max_move_mm": round(float(np.linalg.norm(out, axis=1).max()) * 1000, 2), "weight": weight}
    return out, mw, rep


def _head_rel(v_sx: np.ndarray) -> np.ndarray:
    """Translation-invariant fingerprint of a rest head that does not need FLAME: every SMPL-X vertex above the 88th
    height percentile (head + neck), relative to their mean (the exporter's skin and hair_groom.RestModel put the same
    body at different heights)."""
    v = np.asarray(v_sx, float)
    sel = v[:, 1] > np.percentile(v[:, 1], 88)
    return (v[sel] - v[sel].mean(0)).astype(np.float32)


HEAD_MATCH_M = 5e-5      # a cached displacement is reused when the head matches to 0.05 mm


def resolve_identity(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PACKAGE_ROOT / p


def load_identity(path: str | Path) -> dict:
    p = resolve_identity(path)
    if not p.exists():
        raise FileNotFoundError(f"shape_edits.face.identity: {p} missing")
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["sha"] = hashlib.sha1(p.read_bytes()).hexdigest()[:16]
    return d


def cached_displacement(v_sx: np.ndarray, identity: str | Path, work: Path | None, role: str,
                        weight: float = 1.0, write: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """displacement(...) through the per-take cache work/<take>/face_shape/<role>_identity_disp.npz (keyed on the
    rest head, the identity file and the mask parameters). Recomputes (FLAME needed) on a miss.
    -> (weight x disp, mouth_w, report)."""
    ident = load_identity(identity)
    rel = _head_rel(v_sx)
    mask_key = json.dumps(MASK_DEFAULT, sort_keys=True)
    cache = (work / "face_shape" / f"{role}_identity_disp.npz") if work is not None else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if (int(z["version"]) == CACHE_VERSION and str(z["identity_sha"]) == ident["sha"]
                    and str(z["mask_params"]) == mask_key and len(z["disp"]) == len(v_sx) and z["head_rel"].shape == rel.shape
                    and float(np.abs(z["head_rel"] - rel).max()) < HEAD_MATCH_M):
                rep = json.loads(str(z["report"]))
                rep.update(weight=weight, cache=str(cache), cache_hit=True)
                rep["max_move_mm"] = round(rep["max_move_mm_at_1"] * abs(weight), 2)
                return weight * z["disp"].astype(np.float64), z["mouth_w"].astype(np.float64), rep
    if not flame_available():
        raise FileNotFoundError(
            f"face identity: no valid cache {cache} for this rest head and FLAME is not installed ({FLAME_PKL}); run "
            f"`python -m dancecap.face_identity retarget <take>` on a machine with FLAME")
    d1, mw, rep = displacement(v_sx, ident["beta"], 1.0)
    rep["max_move_mm_at_1"] = rep.pop("max_move_mm")
    if cache is not None and write:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, disp=d1.astype(np.float32), mouth_w=mw.astype(np.float32), head_rel=rel,
                            identity_sha=np.array(ident["sha"]), identity=np.array(str(identity)),
                            version=np.array(CACHE_VERSION), mask_params=np.array(mask_key),
                            report=np.array(json.dumps(rep)))
    rep.update(weight=weight, cache=str(cache) if cache is not None else None, cache_hit=False)
    rep["max_move_mm"] = round(rep["max_move_mm_at_1"] * abs(weight), 2)
    return weight * d1.astype(np.float32).astype(np.float64), mw.astype(np.float32).astype(np.float64), rep


PROTECTED_REGIONS = ("eye_region", "left_eyeball", "right_eyeball", "forehead", "scalp", "left_ear", "right_ear",
                     "boundary")


def protected_moves(v0: np.ndarray, v1: np.ndarray) -> dict:
    """{FLAME region: max vertex move (mm)} between two SMPL-X rest meshes for the regions the face edits must keep."""
    F = flame()
    mv = np.linalg.norm(np.asarray(v1, float) - np.asarray(v0, float), axis=1)[F["sx_ids"]]
    return {k: round(1000 * float(mv[F["masks"][k]].max()), 4) for k in PROTECTED_REGIONS}


# ------------------------------------------------------------------------------------------------ fit (torch)

def _rodrigues(torch, r):
    th = torch.linalg.norm(r, dim=-1, keepdim=True).clamp_min(1e-8)
    k = r / th
    K = torch.zeros(r.shape[:-1] + (3, 3), dtype=r.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    I = torch.eye(3, dtype=r.dtype).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


def _yaw_index(R: np.ndarray) -> int:
    """FLAME dynamic contour bin of a head rotation (R maps canonical -> camera; SMIRK's FLAME.py convention)."""
    yaw = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])))  # SMIRK FLAME convention (no sign flip)
    a = int(round(min(yaw, 39.0)))
    if a >= 0:
        return a
    return 78 if a < -39 else 39 - a


def fit_images(lmk: list[dict], device: str = "cpu", portrait_weight: float = 3.0, iters: int = 600,
               shape_prior: float = 2e-5, verbose: bool = True) -> dict:
    """SMIRK + per-image landmark refinement + robust combination (module docstring, step 1)."""
    import cv2
    import torch

    from .face_smirk import encode, load_smirk, warp_crop
    torch.set_num_threads(2)
    F = flame()
    recs = [r for r in lmk if r.get("ok")]
    crops = []
    for r in recs:
        img = cv2.imread(r["file"])
        crops.append(warp_crop(img, np.asarray(r["smirk_center"]), float(r["smirk_side"])))
    enc = load_smirk(device)
    sm = encode(enc, crops, device)
    del enc
    dt = torch.float64
    vt = torch.tensor(F["v_template"], dtype=dt)
    S = torch.tensor(F["shapedirs"][:, :, :N_SHAPE], dtype=dt)
    E = torch.tensor(F["shapedirs"][:, :, 300:300 + N_EXP], dtype=dt)
    Jr = torch.tensor(F["J_regressor"], dtype=dt)
    wj = torch.tensor(F["weights"][:, 2:3], dtype=dt)
    le, re_ = torch.tensor(F["leye"], dtype=dt), torch.tensor(F["reye"], dtype=dt)
    fT = torch.tensor(F["f"])
    mpf, mpb = torch.tensor(F["f"][F["mp_f"]]), torch.tensor(F["mp_b"], dtype=dt)
    per = []
    for k, r in enumerate(recs):
        c, side = np.asarray(r["smirk_center"], float), float(r["smirk_side"])
        nz = lambda p: (np.asarray(p, float)[:, :2] - c) / (side / 2)  # noqa: E731  normalised crop coords
        mp = nz(r["mp478"])
        t_mp = torch.tensor(mp[F["mp_idx"]], dtype=dt)
        rt = np.asarray(r["rtmw68"], float)
        rs = np.asarray(r["rtmw68_score"], float)
        t_ct = torch.tensor(np.stack([nz(rt[:17]), mp[MP_CONTOUR]]), dtype=dt)       # (2, 17, 2)
        # MediaPipe's face oval is not the iBUG contour (its upper points sit on the temples): RTMW leads, MediaPipe
        # only steadies it (x 0.25)
        w_ct = torch.tensor(np.stack([np.clip((rs[:17] - 0.3) / 0.4, 0, 1), np.full(17, 0.25)]), dtype=dt)
        occ = float(np.linalg.norm(mp[33] - mp[263]))
        b0 = torch.tensor(sm["shape_params"][k], dtype=dt)
        beta = b0.clone().requires_grad_(True)
        psi = torch.tensor(sm["expression_params"][k], dtype=dt).requires_grad_(True)
        jaw = torch.tensor(sm["jaw_params"][k], dtype=dt).requires_grad_(True)
        eyel = torch.tensor(sm["eyelid_params"][k], dtype=dt)
        rot = torch.tensor(sm["pose_params"][k], dtype=dt).requires_grad_(True)
        cam = torch.tensor(sm["cam"][k], dtype=dt).requires_grad_(True)

        def model(beta, psi, jaw, rot):
            V = vt + S @ beta + E @ psi + eyel[0] * le + eyel[1] * re_
            J = Jr @ V
            Rj = _rodrigues(torch, jaw)
            V = V + wj * ((V - J[2]) @ Rj.T + J[2] - V)
            Rg = _rodrigues(torch, rot)
            return (V - J[0]) @ Rg.T + J[0], Rg

        def proj(P):
            return torch.stack([cam[0] * (P[..., 0] + cam[1]), -cam[0] * (P[..., 1] + cam[2])], -1)

        def lmks(V, Rg):
            pm = (V[mpf] * mpb[..., None]).sum(1)
            di = _yaw_index(Rg.detach().numpy())
            fd = torch.tensor(F["f"][F["lmk_dyn_f"][di]])
            pc = (V[fd] * torch.tensor(F["lmk_dyn_b"][di], dtype=dt)[..., None]).sum(1)
            return proj(pm), proj(pc)

        def loss_fn(stage):
            V, Rg = model(beta, psi, jaw, rot)
            pm, pc = lmks(V, Rg)
            e_mp = torch.linalg.norm(pm - t_mp, dim=-1) / occ
            e_ct = torch.linalg.norm(pc[None] - t_ct, dim=-1) / occ
            hub = lambda e, d=0.04: torch.where(e < d, 0.5 * e ** 2, d * (e - 0.5 * d))  # noqa: E731
            L = hub(e_mp).mean() + 0.6 * (w_ct * hub(e_ct)).sum() / w_ct.sum()
            if stage > 0:
                L = L + shape_prior * (((beta - b0) ** 2).sum() + 0.1 * (beta ** 2).sum()) \
                    + 1e-5 * (psi ** 2).sum() + 1e-2 * (jaw ** 2).sum()
            return L, e_mp, e_ct

        opt = torch.optim.Adam([rot, cam], lr=0.01)
        for it in range(150):
            opt.zero_grad()
            L, *_ = loss_fn(0)
            L.backward()
            opt.step()
        opt = torch.optim.Adam([{"params": [rot, cam], "lr": 0.005}, {"params": [beta], "lr": 0.02},
                                {"params": [psi, jaw], "lr": 0.01}])
        for it in range(iters):
            opt.zero_grad()
            L, e_mp, e_ct = loss_fn(1)
            L.backward()
            opt.step()
            with torch.no_grad():
                jaw[0].clamp_(min=0.0)
        with torch.no_grad():
            _, e_mp, e_ct = loss_fn(1)
            V, Rg = model(beta, psi, jaw, rot)
        rec = {"name": r["name"], "file": r["file"], "beta": beta.detach().numpy().copy(),
               "beta_smirk": sm["shape_params"][k].astype(np.float64), "psi": psi.detach().numpy().copy(),
               "jaw": jaw.detach().numpy().copy(), "eyelid": sm["eyelid_params"][k].astype(np.float64),
               "rot": rot.detach().numpy().copy(), "R": Rg.numpy().copy(), "cam": cam.detach().numpy().copy(),
               "err_mp_occ": float(e_mp.mean()), "err_contour_occ": float(e_ct[1].mean()),
               "err_contour_rtmw_occ": float((w_ct[0] * e_ct[0]).sum() / w_ct[0].sum().clamp_min(1e-9)),
               "yaw_bin": _yaw_index(Rg.numpy()), "occ_norm": occ, "crop": crops[k]}
        per.append(rec)
        if verbose:
            print(f"[face_identity] {r['name']}: mp {rec['err_mp_occ']:.3f} contour {rec['err_contour_occ']:.3f} "
                  f"(rtmw {rec['err_contour_rtmw_occ']:.3f}) x occ; |beta - smirk| "
                  f"{np.linalg.norm(rec['beta'] - rec['beta_smirk']):.2f}", flush=True)
    return combine(per, portrait_weight)


def combine(per: list[dict], portrait_weight: float = 3.0, max_err: float = 2.0, max_dist: float = 2.5) -> dict:
    """Robust combination of the per-image shape codes: residual inliers (fit error <= max_err x the median), then
    shape-space inliers (|beta_i - weighted median| <= max_dist x the median distance), weighted mean."""
    B = np.stack([p["beta"] for p in per])
    w = np.array([portrait_weight if p["name"] == "portrait" else 1.0 for p in per])
    err = np.array([p["err_mp_occ"] + 0.6 * p["err_contour_rtmw_occ"] for p in per])
    ok = err <= max_err * np.median(err)
    med = np.array([_wmedian(B[ok, j], w[ok]) for j in range(B.shape[1])])
    dist = np.linalg.norm(B - med, axis=1)
    ok &= dist <= max_dist * max(np.median(dist[ok]), 1e-6)
    beta = (w[ok, None] * B[ok]).sum(0) / w[ok].sum()
    return {"beta": beta, "per_image": per, "inlier": ok, "weights": w, "fit_err": err, "dist_to_median": dist}


def _wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def save_identity(res: dict, out: Path, meta: dict) -> Path:
    per = res["per_image"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, beta=res["beta"].astype(np.float64), n_shape=np.array(N_SHAPE), model=np.array("FLAME2020 generic"),
        images=np.array([p["name"] for p in per]), inlier=res["inlier"], image_weight=res["weights"],
        per_image_beta=np.stack([p["beta"] for p in per]), per_image_beta_smirk=np.stack([p["beta_smirk"] for p in per]),
        per_image_psi=np.stack([p["psi"] for p in per]), per_image_jaw=np.stack([p["jaw"] for p in per]),
        per_image_eyelid=np.stack([p["eyelid"] for p in per]), per_image_R=np.stack([p["R"] for p in per]),
        per_image_cam=np.stack([p["cam"] for p in per]), fit_err=res["fit_err"], dist_to_median=res["dist_to_median"],
        meta=np.array(json.dumps(meta)))
    return out


# ------------------------------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m dancecap.face_identity", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("fit", help="landmarks.json -> identity npz")
    a1.add_argument("landmarks")
    a1.add_argument("out")
    a1.add_argument("--exclude", default="", help="comma-separated image names to leave out")
    a1.add_argument("--portrait-weight", type=float, default=3.0)
    a2 = sub.add_parser("retarget", help="(re)build the take's displacement cache for its configured identity")
    a2.add_argument("take")
    a2.add_argument("--role", default="follow")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        lmk = json.loads(Path(a.landmarks).read_text())
        ex = set(filter(None, a.exclude.split(",")))
        lmk = [r for r in lmk if r.get("name") not in ex]
        res = fit_images(lmk, portrait_weight=a.portrait_weight)
        p = save_identity(res, Path(a.out), {"landmarks": str(a.landmarks), "excluded": sorted(ex)})
        print(json.dumps({"out": str(p), "inlier": dict(zip([q["name"] for q in res["per_image"]],
                                                              res["inlier"].tolist()))}, indent=1))
    elif a.cmd == "retarget":
        from . import shape_edits as se
        from .export_unity import model_for, resolve_smplx_dir, shaped_skin
        work = PACKAGE_ROOT / "work" / a.take
        cfg = se.config_for(a.take).get(a.role) or {}
        face = cfg.get("face") or {}
        if not face.get("identity"):
            sys.exit(f"{a.take}: no [dancers.{a.role}.shape_edits.face] identity")
        sdir, _ = resolve_smplx_dir(a.take, None)
        with np.load(work / sdir / f"{a.role}.npz", allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        sk = shaped_skin(model_for(d), d["betas"])
        _, _, rep = cached_displacement(sk.vertices, face["identity"], work, a.role, 1.0)
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
