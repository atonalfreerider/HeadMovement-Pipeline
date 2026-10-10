"""Hair-aware photoreal texture for the follow, seam-free two-band fusion, and a hair colour sample for Unity.

Hooks used by avatar_texture.py (TextureConfig.hair_mode = "exclude", blend = "multiband"; the previous behaviour is
hair_mode = "soft", blend = "mean"):

  1. Per view (hair_maps): hair pixels = dancers/<cam>/hair mask (the SAM2 full mask split by dancers_hair.py; derived
     as full - body when only body/ exists) plus strongly hair-coloured pixels of her full mask near it (per-camera
     Lab model P(hair | colour) of avatar_texture.hair_model: catches strands the split missed). A distance ramp
     around them (zero within r0, full weight beyond r1; both scale with her mask height) gives every pixel a keep
     weight: 0 on the hair and on its mixed border (pink hair-over-white, chroma bleed, motion blur).
  2. Per texel (exclude_hair): body texels (everything except her scalp under the hair) multiply their sample weight
     by keep, so red hair never paints chest, shoulders, back or face; a texel seen only through hair gets no samples
     and is filled. Scalp texels keep their hair samples (hair colour belongs there; it roots the Unity hair).
     Each sample also adds to the texel's hair evidence (weight on hair pixels / all weight).
  3. Scalp region (hair_region_from_evidence): the ANATOMICAL scalp (scalp_vertices: head minus face minus ears minus
     everything below the eye line in front of the ear; 98 % inside FLAME's scalp mask) with only its hairline learned:
     vertices more than hairline_margin_m (1.5 cm) inside it are always hair, the 1.5 cm band on either side of its
     boundary (outside only where hairline_ok: not the face core, ears, cheek / jaw / temple below the eye line)
     follows the samples' hair evidence. Hanging hair over a cheek, an ear or the neck is therefore never scalp (the
     first version learned the region purely from evidence and grew a hair 'bob' over her right cheek and ear). Pass 0
     uses the anatomical scalp; every later pass uses the previous pass's evidence. Where hanging hair covered the
     body in the photos is published separately as <role>_hair_cover.png (evidence fraction per texel).
  4. Fill (fill_region_aware): scalp and body are filled separately (hair never diffuses into skin and vice versa).
     Before the fill (fill_guards) a measured upper-body / neck / non-scalp head texel that is still DARK
     hair-coloured is dropped (hair that leaked through the exclusion), and hair-coloured texels or texels mostly seen
     through her hair may not seed a fill. Body holes use a class-aware fill (fill_classed): the measured vertex
     colours of the good texels are clustered (white top / skin / jeans ...; a hair-coloured class seeds nothing),
     each hole vertex takes the class of its nearest measured vertex on the mesh and its colour diffuses only inside
     that class, neck / nape / face / ear holes take the skin class, and the left/right mirror never copies a
     hair-coloured texel. The hairline is feathered over about one mesh edge.
  5. Seams (multiband): per camera texture layers (that camera's inlier samples, exposure-corrected). High
     frequencies are combined with the sharp per-texel weights (face exponent 4: the best view keeps the face
     sharp) smoothed over ~2 texels, low frequencies (Gaussian sigma = band_sigma_frac x atlas size) with the same
     weights low-passed over the surface, so exposure / white balance / shading differences between views blend over
     centimetres instead of stepping at view boundaries. The low-pass is mask-normalised inside each UV island x
     hair-region group. Measured on larissa_kadu (same samples, 2048^2): luminance steps across view boundaries in
     excess of the interior texel-to-texel variation -26 % (1 texel) / -32 % (4 texels) on the follow, 0 / -9 % on
     the lead; steps that remain are content mismatch between views (fit misregistration, cloth moving), not exposure.
     Face texels (face_sharp_fusion, default) keep the one-band sharp fusion (best view ^4 at all frequencies),
     feathered over ~4 texels: mixing one camera's detail with another camera's local mean put a red-brown rim on the
     lead's eye.
  Off-body hair (offbody_hair, TextureConfig.hair_offbody): the split misses motion-blurred hair - in a hair whip the
     outer, smeared part of the whip is labelled BODY (larissa_kadu_38 review/dancers_body_hair.jpg: cam07 #372, cam05
     #342, cam01 #396, cam09 #525, cam04 #607 beside her neck and shoulder). In the bake every pixel of her full mask
     that lies outside her posed model's silhouette (dilated by hair_offbody_margin_frac x her mask height) and is
     connected to her hair is hair whatever its colour, and joins the hair mask before the colour step and the
     exclusion ramp. A view where that off-body hair is large (>= hair_whip_area_frac of her mask: a whip) gets its
     exclusion radii multiplied by hair_whip_mult, so smeared hair over her shoulders / arms next to the whip is
     excluded too. Views and pixels are counted in report.json passes[].hair_offbody.
  Face: the dancers-stage hair split often covers her warm-lit face, so inside her projected face (FaceProjector)
     hair = mask AND a face-aware colour model (hair_lut_face_aware), other face pixels get a relative weight, the
     face-core texels use a lighter luminance quantile as the fusion's inlier reference (TextureConfig.skin_ref_q) and
     a face-core texel still fused dark hair-coloured is refilled (dark_hair_guard).
  6. Hair sample (hair_sample): exposure-corrected colours of hair pixels from the bake's views -> follow_hair.json
     (palette, root -> tip gradient) and follow_hair_strip.png (256 x 512 strand strip, row 0 = root) for the Unity
     hair cards / strands.

Review / validation (writes only under the given output dirs):
    python -m dancecap.avatar_hair metrics <avatar dir>                      # hair-coloured texel fractions
    python -m dancecap.avatar_hair compare <take> --before DIR --after DIR --review DIR [--heldout-cam C] [--label L]
                                                                             [--no-sheets] [--json F]
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import os

import numpy as np

UPPER_JOINTS = (6, 9, 12, 13, 14, 16, 17)  # spine2, spine3, neck, collars, shoulders (SMPL-X joint ids)
SHOULDER_JOINTS = (13, 14, 16, 17)
FOREARM_JOINTS = (18, 19, 20, 21)


# ------------------------------------------------------------------------------------------------ per view


def offbody_hair(full: np.ndarray, sil: np.ndarray, hair: np.ndarray | None, margin_px: float) -> np.ndarray:
    """Her full-mask pixels outside her posed model's silhouette (dilated by margin_px) that are connected to her hair
    (components touching the hair mask, 3 px tolerance) -> bool (H, W). The smeared outer part of a hair whip that the
    dancers-stage split labelled body; a misfit hand or foot far from her hair is not included."""
    fm = full > 0
    if not fm.any() or hair is None or not np.any(hair):
        return np.zeros(fm.shape, bool)
    r = max(1, int(round(margin_px)))
    sil_d = cv2.dilate((sil > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    off = fm & (sil_d == 0)
    if not off.any():
        return off
    n, lab = cv2.connectedComponents((off | (hair > 0)).astype(np.uint8), connectivity=8)
    touch = np.unique(lab[cv2.dilate((hair > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0])
    return off & np.isin(lab, touch[touch > 0])


def hair_maps(full: np.ndarray, hair: np.ndarray | None, p_hair: np.ndarray | None, dilate_frac: float = 0.006,
              ramp_frac: float = 0.006, p_hard: float = 0.6, near_frac: float = 0.06,
              face: np.ndarray | None = None, p_face: np.ndarray | None = None,
              extra_hair: np.ndarray | None = None, whip_mult: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """-> keep (H, W) float32 in [0, 1] = weight multiplier for body texels, hair (H, W) uint8 0/1 = hair pixels.

    full: her full mask (body + hair), hair: her hair mask (or None), p_hair: P(hair | colour) (or None), face: her
    projected face + throat (FaceProjector) where the hair mask is not trusted (the dancers-stage split often marks
    warm-lit face skin as hair): there a pixel is hair only when the mask AND the face-aware colour model p_face
    (hair_lut_face_aware; default p_hair) agree, and other face pixels get a relative (1 - p_face)^2 weight.
    extra_hair: pixels known to be hair whatever their colour (offbody_hair), joined to the hair mask outside the face;
    whip_mult scales both exclusion radii (hair-whip views)."""
    H, W = full.shape[:2]
    fm = full > 0
    mask = (hair > 0) & fm if hair is not None else np.zeros((H, W), bool)
    fd = (face > 0) if face is not None else np.zeros((H, W), bool)
    hb = mask & ~fd  # outside her face / throat the hair mask is trusted
    if extra_hair is not None:
        hb |= (extra_hair > 0) & fm & ~fd
    ys = np.nonzero(fm.any(1))[0]
    h = float(ys[-1] - ys[0] + 1) if len(ys) else 0.0
    if p_hair is not None and h > 0:
        strong = (p_hair >= p_hard) & fm & ~fd
        if hb.any():
            d_h = cv2.distanceTransform((~hb).astype(np.uint8), cv2.DIST_L2, 5)
            strong &= d_h <= max(3.0, near_frac * h)
        else:  # no hair mask (keypoints unknown): hair-coloured pixels in the top 45 % of her mask
            strong[int(ys[0] + 0.45 * h):] = False
        hb |= strong
    keep = np.ones((H, W), np.float32)
    if hb.any():
        r0 = max(2.0, dilate_frac * h) * max(1.0, whip_mult)
        r1 = r0 + max(2.0, ramp_frac * h) * max(1.0, whip_mult)
        d = cv2.distanceTransform((~hb).astype(np.uint8), cv2.DIST_L2, 5)
        keep = np.clip((d - r0) / (r1 - r0), 0, 1).astype(np.float32)
    if p_hair is not None:  # the old soft colour rule stays for stray hair-coloured pixels far from her hair
        keep *= (np.clip((1 - p_hair) / 0.5, 0, 1) ** 2).astype(np.float32)
    pf = p_face if p_face is not None else p_hair
    if fd.any():  # face / throat: no hard cut from the (unreliable) mask; hair there = mask AND clearly hair-coloured
        inside = fd & mask & (pf >= 0.8) if pf is not None else np.zeros((H, W), bool)
        soft = ((1 - pf[fd]) ** 2).astype(np.float32) if pf is not None else 1.0
        keep[fd] = np.where(inside[fd], 0.0, soft)  # relative weighting; the fusion's light-quantile reference on
        hb |= inside                                 # face texels (ref_q) then keeps skin over darker hair samples
    return keep, hb.astype(np.uint8)


def exclude_hair(torch, store, w, px, keep_map, hair_map, hair_tex, sample):
    """Bake hook for one view of a hair role: accumulate hair evidence (store.hair_ev (2, N): weight on hair pixels,
    all weight) and zero/ramp the weights of body texels on and around hair pixels. -> new weights (N,)."""
    dev = w.device
    keep = sample(torch, torch.as_tensor(keep_map, device=dev), px)
    on_hair = sample(torch, torch.as_tensor(hair_map, device=dev), px).float()
    if getattr(store, "hair_ev", None) is not None:
        store.hair_ev[0] += w * on_hair
        store.hair_ev[1] += w
    return torch.where(hair_tex, w, w * keep)


def collect_hair_pixels(out: list, img_rgb: np.ndarray, hair_map: np.ndarray, head_px, head_z: float, f: float,
                        vi: int, max_px: int = 800, rng=None) -> int:
    """Append (view id, sRGB uint8 (n, 3), distance from the head centre in metres at head depth (n,)) of up to
    max_px interior hair pixels of one view (eroded by 1 px: no mixed border pixels)."""
    if hair_map is None or not hair_map.any() or not np.isfinite(head_z) or head_z <= 0.1:
        return 0
    core = cv2.erode(hair_map.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.nonzero(core)
    if len(xs) < 20:
        return 0
    rng = rng or np.random.default_rng(vi)
    if len(xs) > max_px:
        sel = rng.choice(len(xs), max_px, replace=False)
        ys, xs = ys[sel], xs[sel]
    d_m = np.hypot(xs - float(head_px[0]), ys - float(head_px[1])) * head_z / f
    out.append((vi, img_rgb[ys, xs].astype(np.uint8), d_m.astype(np.float32)))
    return len(xs)


def skin_vertex_ids(model_dir: Path) -> np.ndarray:
    """SMPL-X vertices of the front of the face (forehead to chin, |x| < 6.5 cm of the head joint): skin that her
    hair may cross but never covers for long, and that the dancers-stage hair split often swallows."""
    from .avatar_uv import model_npz, vertex_regions
    m = np.load(model_npz(model_dir), allow_pickle=True)
    vt = np.asarray(m["v_template"], float)
    J = np.asarray(m["J_regressor"], float) @ vt
    reg = vertex_regions(m)
    return np.flatnonzero(reg["face"] & (np.abs(vt[:, 0] - J[15, 0]) < 0.065))


class FaceProjector:
    """Image mask of the follow's visible face + throat for a view: the front-facing SMPL-X triangles of those
    regions (posed at the view's time), rasterised exactly (no hull: hair beside the face stays outside), plus a
    margin of dilate_frac x the projected face height for pose / registration error."""

    def __init__(self, body, model_dir: Path, dilate_frac: float = 0.04):
        from .avatar_uv import model_npz
        m = np.load(model_npz(model_dir), allow_pickle=True)
        self.body = body
        faces = np.asarray(m["f"], np.int64)
        inset = np.zeros(len(m["v_template"]), bool)
        inset[skin_vertex_ids(model_dir)] = True
        self.tri = faces[inset[faces].all(1)]
        self.dilate_frac = dilate_frac

    def mask(self, v: dict, H: int, W: int) -> np.ndarray | None:
        p = self.body.at(v["t"])
        if p is None:
            return None
        V = np.asarray(p[0], np.float64)
        T = V[self.tri]  # (F, 3, 3)
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        R, t, K = np.asarray(v["R"]), np.asarray(v["t_w"]), np.asarray(v["K"])
        C = -R.T @ t
        nf = n.sum(0)  # area-weighted mean normal of the face: turned away (> ~84 deg) -> the head hides it
        dc = C - T.reshape(-1, 3).mean(0)
        if nf @ dc < 0.1 * np.linalg.norm(nf) * np.linalg.norm(dc):
            return None
        facing = ((C - T.mean(1)) * n).sum(1) > 0
        if facing.sum() < 20:
            return None
        Xc = T[facing].reshape(-1, 3) @ R.T + t
        if (Xc[:, 2] < 0.1).any():
            return None
        x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        d = 1 + float(v.get("k1", 0.0)) * (x * x + y * y)
        uv = np.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], 1)
        out = np.zeros((H, W), np.uint8)
        cv2.fillPoly(out, list(np.round(uv * 4).astype(np.int32).reshape(-1, 3, 2)), 1, lineType=cv2.LINE_8, shift=2)
        r = int(round(self.dilate_frac * float(np.ptp(uv[:, 1]))))
        if r >= 1:
            out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return out


def hair_lut_face_aware(work: Path, cam: str, frame_step: int, face_fn, role: str = "follow",
                        pids: dict | None = None) -> tuple[np.ndarray | None, dict]:
    """avatar_texture.hair_model with her projected face + throat as skin: per-camera P(hair | Lab colour) from
    dancers/<cam>/hair pixels OUTSIDE the (dilated) face (hair seeds) vs the rest of both dancers plus the face/throat
    pixels of her full mask (non-hair seeds). face_fn(i, H, W) -> mask or None."""
    from .avatar_texture import HAIR_BIN, PIDS, _read_mask
    pids = pids or PIDS
    dd = work / "dancers" / cam
    pid = pids[role]
    if not (dd / "hair").exists():
        return None, {"cam": cam, "status": "no hair masks"}
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    nb = 256 // HAIR_BIN
    Hh, Hn, Hf = np.zeros((nb,) * 3), np.zeros((nb,) * 3), np.zeros((nb,) * 3)
    k15 = np.ones((15, 15), np.uint8)
    n_face = 0
    for i in range(0, len(meta["frames"]), frame_step):
        hp = dd / "hair" / f"mask_{i:04d}_{pid:02d}.png"
        img = cv2.imread(str(work / "frames" / cam / f"{i:06d}.jpg"))
        if img is None or not hp.exists():
            continue
        shp = img.shape[:2]
        lab = (cv2.cvtColor(img, cv2.COLOR_BGR2LAB) // HAIR_BIN).reshape(-1, 3)
        hair = _read_mask(hp, shp).astype(bool)
        fol = _read_mask(dd / "masks" / f"mask_{i:04d}_{pid:02d}.png", shp).astype(bool)
        lead = _read_mask(dd / "masks" / f"mask_{i:04d}_{3 - pid:02d}.png", shp).astype(bool)
        face = face_fn(i, *shp)
        face = face.astype(bool) & fol if face is not None else np.zeros(shp, bool)
        if face.any():
            n_face += 1
        fdil = cv2.dilate(face.astype(np.uint8), k15).astype(bool)
        non = fol & ~cv2.dilate(hair.astype(np.uint8), k15).astype(bool)
        ys = np.nonzero(fol.any(1))[0]
        if len(ys):
            non[: int(ys[0] + 0.45 * (ys[-1] - ys[0]))] = False
        non = non | lead
        for Hx, msk in ((Hh, hair & ~fdil), (Hn, non), (Hf, face)):
            q = lab[msk.ravel()]
            np.add.at(Hx, (q[:, 0], q[:, 1], q[:, 2]), 1)
    if Hh.sum() < 2000 or Hn.sum() < 2000:
        return None, {"cam": cam, "status": "too few pixels", "hair_px": int(Hh.sum()), "other_px": int(Hn.sum())}
    from scipy.ndimage import gaussian_filter
    dh = gaussian_filter(Hh / Hh.sum(), 1.0)
    dn = gaussian_filter(Hn / Hn.sum(), 1.0)
    if Hf.sum() > 500:  # face / throat skin: its own class, as likely as the other non-hair colours together
        dn = np.maximum(dn, gaussian_filter(Hf / Hf.sum(), 1.0))
    lut = (dh / (dh + dn + 1e-7)).astype(np.float32)
    on = lambda Hx: round(float((Hx * (lut > 0.5)).sum() / max(Hx.sum(), 1)), 3)
    return lut, {"cam": cam, "status": "ok", "face_aware": True, "hair_px": int(Hh.sum()), "other_px": int(Hn.sum()),
                 "face_px": int(Hf.sum()), "frames_with_face": n_face, "hair_recall": on(Hh),
                 "other_false_pos": on(Hn), "face_marked_hair": on(Hf)}


# ------------------------------------------------------------------------------------------------ texel regions


def scalp_vertices(m) -> dict:
    """Anatomical scalp on the SMPL-X template (rest pose, metres; h = head joint, e = eye level):
    head minus face (avatar_uv.vertex_regions: the face reaches e + 5.7 cm at the front) minus the ears
    (|x - h_x| > 5.5 cm, e - 7 cm < y < e + 1 cm, h_z - 4 cm < z < h_z + 1.5 cm: 97 % recall / 97 % precision against
    FLAME's ear masks mapped through SMPL-X__FLAME_vertex_ids) minus everything below the eye line in front of the
    back edge of the ear (temple, cheek, jaw: z > h_z - 3.5 cm and y < e). Checked against FLAME's 'scalp' mask on the
    same correspondence: 98 % of these vertices are FLAME scalp, 91 % of FLAME's scalp inside the SMPL-X head is
    covered (the rest is forehead / nape boundary). Also -> `hairline_ok`: where the evidence-learned hairline may
    extend beyond the scalp (head, not the face core, not the ears, not below the eye line in front of the ear)."""
    from .avatar_uv import vertex_regions
    vt = np.asarray(m["v_template"], float)
    J = np.asarray(m["J_regressor"], float) @ vt
    vreg = vertex_regions(m)
    h, e = J[15], float(J[23:25, 1].mean())
    dx, dy, dz = vt[:, 0] - h[0], vt[:, 1] - e, vt[:, 2] - h[2]
    head = vreg["head"]
    ear = head & (np.abs(dx) > 0.055) & (dy > -0.07) & (dy < 0.01) & (dz > -0.04) & (dz < 0.015)
    low_front = (dz > -0.035) & (dy < 0.0)
    face_core = vreg["face"] & (np.abs(dx) < 0.055) & (dy < 0.03)
    return {"scalp": head & ~vreg["face"] & ~ear & ~low_front, "ear": ear,
            "hairline_ok": head & ~face_core & ~ear & ~low_front, "head": head}


def _geodesic_to_boundary(faces: np.ndarray, verts: np.ndarray, inside: np.ndarray, within: np.ndarray) -> np.ndarray:
    """Surface distance (m, along mesh edges of the rest template) of every vertex in `within` to the boundary of
    `inside` (both sides of a crossing edge are at 0); inf elsewhere."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    E = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    E = E[within[E[:, 0]] & within[E[:, 1]]]
    L = np.linalg.norm(verts[E[:, 0]] - verts[E[:, 1]], axis=1)
    n = len(verts)
    G = coo_matrix((np.concatenate([L, L]), (np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]]))),
                   shape=(n, n)).tocsr()
    cross = inside[E[:, 0]] != inside[E[:, 1]]
    src = np.unique(E[cross].ravel())
    if not len(src):
        return np.full(n, np.inf)
    d = dijkstra(G, directed=False, indices=src, min_only=True)
    d[~within] = np.inf
    return d


def texel_regions(atlas, model_dir: Path) -> dict:
    """Per-texel masks on the SMPL-X UV layout: head / face / face_core / anatomical scalp (`prior`, scalp_vertices)
    / ears / upper body (chest, shoulders, back, neck) / shoulders / forearms / front, the dominant LBS joint and the
    rest-pose surface point; vertex-level scalp / hairline_ok and the signed surface distance to the scalp boundary
    (scalp_dist_v: > 0 inside) for hair_region_from_evidence."""
    from .avatar_uv import model_npz, vertex_regions
    m = np.load(model_npz(model_dir), allow_pickle=True)
    w = np.asarray(m["weights"], float)
    vt = np.asarray(m["v_template"], float)
    J = np.asarray(m["J_regressor"], float) @ vt
    tri = atlas.faces[atlas.face]
    dom = w.argmax(1)[tri[np.arange(atlas.n), atlas.bary.argmax(1)]]
    P0 = atlas.surface(vt)
    head = (atlas.region & 1) > 0
    face = (atlas.region & 2) > 0
    eye_y = float(J[23:25, 1].mean())
    face_core = face & (np.abs(P0[:, 0] - J[15, 0]) < 0.055) & (P0[:, 1] < eye_y + 0.03)
    vreg = vertex_regions(m)
    sv = scalp_vertices(m)
    agg = lambda flag: (flag[tri].astype(np.float32) * atlas.bary).sum(1) > 0.5
    low_front = head & (P0[:, 2] - J[15, 2] > -0.035) & (P0[:, 1] < eye_y)  # texel test of the vertex rule
    scalp = agg(sv["scalp"]) & head & ~face_core & ~low_front
    ear = agg(sv["ear"]) & head
    faces = np.asarray(m["f"], np.int64)
    d = _geodesic_to_boundary(faces, vt, sv["scalp"], sv["head"])
    neck = (dom == 12) & ~head
    return {"dom": dom, "P0": P0, "head": head, "face": face, "face_core": face_core, "prior": scalp, "ear": ear,
            "low_front": low_front,
            "skin_pref": face_core,
            "upper": np.isin(dom, UPPER_JOINTS) & ~head, "shoulders": np.isin(dom, SHOULDER_JOINTS) & ~head,
            "neck": neck, "forearm": np.isin(dom, FOREARM_JOINTS),
            "front": P0[:, 2] > J[dom, 2], "head_v": vreg["head"], "n_verts": len(vt),
            "scalp_v": sv["scalp"], "hairline_ok_v": sv["hairline_ok"], "ear_v": sv["ear"],
            "scalp_dist_v": np.where(sv["scalp"], d, -d)}


def dark_hair_guard(rgb_lin: np.ndarray, conf: np.ndarray, where: np.ndarray) -> tuple[np.ndarray, int]:
    """Last guard on the face core: a fused texel that is still dark hair-coloured (red_cast and L* < 45) is
    treated as unmeasured, so the fill (left/right mirror, then the mesh) replaces it. -> (conf, texels dropped)"""
    from .avatar_texture import linear_to_srgb
    idx = np.flatnonzero(where & (conf > 0))
    if not len(idx):
        return conf, 0
    lab = lab_of(linear_to_srgb(rgb_lin[idx]))
    bad = idx[red_cast(lab) & (lab[:, 0] < 45)]
    out = conf.copy()
    out[bad] = 0.0
    return out, int(len(bad))


def prior_region(regions: dict) -> tuple[np.ndarray, np.ndarray]:
    b = regions["prior"].copy()
    return b, b.astype(np.float32)


def _vertex_adjacency(faces: np.ndarray, n_verts: int):
    from scipy.sparse import coo_matrix
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    A = coo_matrix((np.ones(len(i)), (i, j)), shape=(n_verts, n_verts)).tocsr()
    A.data[:] = 1.0
    return A


def _to_vertices(atlas, values: np.ndarray, weights: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """Bary-weighted pooling of per-texel values onto mesh vertices -> (sum w*v, sum w)."""
    tri = atlas.faces[atlas.face]
    wt = atlas.bary * weights[:, None]
    sw = np.bincount(tri.ravel(), wt.ravel(), n_verts)
    sv = np.bincount(tri.ravel(), (wt * values[:, None]).ravel(), n_verts)
    return sv, sw


def _from_vertices(atlas, vals: np.ndarray) -> np.ndarray:
    return np.einsum("nk,nk->n", vals[atlas.faces[atlas.face]], atlas.bary)


def hair_region_from_evidence(atlas, ev_hair: np.ndarray, ev_all: np.ndarray, regions: dict, thresh: float = 0.5,
                              prior_strength: float = 0.05, smooth: int = 3, feather: float = 0.15,
                              margin_m: float = 0.015) -> tuple:
    """-> (binary scalp-under-hair texels (N,) bool, soft weight (N,) float32 for the feathered hairline, stats).

    The region is the ANATOMICAL scalp (texel_regions prior = scalp_vertices) with only its hairline learned from the
    data: vertices deeper than margin_m inside the scalp are always hair (hair roots there whatever the photos show),
    vertices within margin_m of the scalp boundary - inside it, or outside it where hairline_ok (head, not the face
    core, not the ears, not below the eye line in front of the ear) - are hair when their evidence says so, everything
    else never. Hanging hair over a cheek, ear, jaw or the neck therefore never becomes scalp: those texels stay body
    texels (hair excluded, filled from skin); where hanging hair covered the body is published separately
    (hair_cover). Evidence per vertex = (sum w*hair + k*prior) / (sum w + k), k = prior_strength x the median vertex
    weight, then `smooth` neighbour-averaging rounds on the head vertices."""
    n_verts = regions["n_verts"]
    head_v = regions["head_v"]
    scalp_v = regions["scalp_v"]
    sd = regions["scalp_dist_v"]
    hv, wv = _to_vertices(atlas, np.where(ev_all > 0, ev_hair / np.maximum(ev_all, 1e-20), 0.0), ev_all, n_verts)
    prior_v = scalp_v.astype(float)
    k = prior_strength * (np.median(wv[head_v & (wv > 0)]) if (head_v & (wv > 0)).any() else 1.0)
    fv = (hv + k * prior_v) / (wv + k)
    fv[~head_v] = 0.0
    A = _vertex_adjacency(atlas.faces, n_verts)
    deg = np.asarray(A.sum(1)).ravel()
    for _ in range(smooth):
        sm = (fv + A @ fv) / (1 + deg)
        fv = np.where(head_v, sm, 0.0)
    core = scalp_v & (sd > margin_m)
    band = (scalp_v & (sd <= margin_m)) | (~scalp_v & regions["hairline_ok_v"] & (-sd <= margin_m))
    val = np.where(core, 1.0, np.where(band, fv, 0.0))
    never = ~regions["head"] | regions["face_core"] | regions["ear"] | regions.get("low_front", False)
    soft = _from_vertices(atlas, val)
    soft[never] = 0.0
    hair = soft > thresh
    s = np.clip((soft - (thresh - feather)) / (2 * feather), 0, 1).astype(np.float32)
    s[never] = 0.0
    head = regions["head"]
    ev_t = _from_vertices(atlas, fv)
    stats = {"head_texels_hair_fraction": round(float(hair[head].mean()), 4),
             "prior_scalp_hair_fraction": round(float(hair[regions["prior"]].mean()), 4),
             "outside_scalp_hair_fraction_of_head": round(float((hair & ~regions["prior"])[head].mean()), 4),
             "face_hair_fraction": round(float(hair[regions["face"]].mean()), 4),
             "face_outside_core_hair_fraction": round(float(hair[regions["face"] & ~regions["face_core"]].mean()), 4),
             "ear_hair_fraction": round(float(hair[regions["ear"]].mean()), 4) if regions["ear"].any() else 0.0,
             "evidence_above_thresh_outside_allowed_head_fraction": round(float(
                 ((ev_t > thresh) & ~hair)[head & ~regions["prior"]].mean()), 4),
             "evidence_texels": int((ev_all[head] > 0).sum())}
    return hair, s, stats


def hair_cover(ev_hair: np.ndarray, ev_all: np.ndarray) -> np.ndarray:
    """Per texel the share of the sample weight that saw her hair over it (0 where never seen): where her hanging
    hair covered the body / face in the capture. Not where hair roots (that is the scalp region)."""
    return np.where(ev_all > 0, ev_hair / np.maximum(ev_all, 1e-20), 0.0).astype(np.float32)


def island_ids(atlas) -> np.ndarray:
    """UV island of every texel (connected components of the vt graph)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ft = np.asarray(atlas.ft)
    nvt = len(atlas.vt)
    i = np.concatenate([ft[:, 0], ft[:, 1], ft[:, 2]])
    j = np.concatenate([ft[:, 1], ft[:, 2], ft[:, 0]])
    _, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(nvt, nvt)), directed=False)
    return lab[ft[atlas.face, 0]].astype(np.int32)


def texel_groups(islands: np.ndarray, hair: np.ndarray | None) -> np.ndarray:
    """Low-pass groups: UV island x (scalp under hair | body)."""
    return islands * 2 + (hair.astype(np.int32) if hair is not None else 0)


# ------------------------------------------------------------------------------------------------ hole filling


def fill_classed(atlas, rgb_lin: np.ndarray, conf: np.ndarray, n_verts: int, conf_ref: float | None = None,
                 classes: int = 4, iters: int = 400, bad: np.ndarray | None = None,
                 skin_only: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """avatar_texture.fill_holes with a class-aware diffusion: -> filled linear RGB (N,3), alpha (N,), source (N,)
    uint8 0 measured / 1 mirror / 2 filled. Hole vertices take the colour class (k-means in Lab of the measured
    vertex colours) of their nearest measured vertex (mesh hops) and diffuse only from vertices of that class.

    bad (N,) bool: measured texels that may NOT seed a fill (hair-coloured / mostly seen through hair): they keep
    their own colour but are never mirrored onto a hole, never feed the vertex colours the diffusion starts from, and
    a k-means class whose centre is hair-coloured (red_cast) seeds nothing. skin_only (N,) bool: hole texels that
    take a skin colour whatever their nearest measured neighbour is (neck / nape / face / ears): a colour field
    diffused only from measured, non-bad vertices of the skin classes (warm, not hair-coloured k-means centres with
    a* > 5 and b* > 8, each holding >= 5 % of the skin-zone seeds; lit and shadowed skin usually split into two;
    fallback the most common class there). A vertex its class cannot reach gets the class's median colour (not the
    global mean)."""
    from scipy.sparse import diags
    from scipy.sparse.csgraph import dijkstra

    from .avatar_texture import linear_to_srgb
    ref = conf_ref if conf_ref is not None else (np.median(conf[conf > 0]) if (conf > 0).any() else 1.0)
    alpha = np.clip(conf / (0.05 * ref), 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    good = np.ones(atlas.n, bool) if bad is None else ~np.asarray(bad, bool)
    out = rgb_lin.copy()
    src = np.zeros(atlas.n, np.uint8)
    m = atlas.mirror  # 1) left/right mirror (as fill_holes), from good texels only
    mm = np.clip(m, 0, None)
    am = np.where(m >= 0, alpha[mm] * good[mm], 0)
    use = (alpha < 1) & (am > alpha)
    mc = rgb_lin[mm]
    out[use] = alpha[use, None] * rgb_lin[use] + (1 - alpha[use, None]) * mc[use]
    alpha2 = np.where(use, alpha + (1 - alpha) * am, alpha)
    src[use & (alpha < 0.5)] = 1
    tri = atlas.faces[atlas.face]  # 2) vertex colours of the measured / mirrored good texels
    wt = atlas.bary * (alpha2 * good)[:, None] ** 2
    wv = np.bincount(tri.ravel(), wt.ravel(), n_verts)
    cv = np.stack([np.bincount(tri.ravel(), (wt * out[:, ch, None]).ravel(), n_verts) for ch in range(3)], 1)
    known = wv > 1e-6
    col = np.where(known[:, None], cv / np.maximum(wv, 1e-12)[:, None], 0)
    vstat = {}
    col_skin, sk_t = None, None
    if (~known).any() and known.sum() >= max(classes, 1):
        A = _vertex_adjacency(atlas.faces, n_verts)
        kidx = np.flatnonzero(known)
        lab_known = np.zeros(len(kidx), np.int32)
        red_cls, cen = set(), None
        if classes > 1:
            lab = cv2.cvtColor(linear_to_srgb(col[kidx]).astype(np.float32)[None], cv2.COLOR_RGB2LAB)[0]
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            cv2.setRNGSeed(0)
            _, lk, cen = cv2.kmeans(lab.astype(np.float32), classes, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            lab_known = lk.ravel().astype(np.int32)
            red_cls = {int(k) for k in np.flatnonzero(red_cast(cen))}
            vstat["class_centres_lab"] = np.round(cen, 1).tolist()
            vstat["red_classes"] = sorted(red_cls)
        if red_cls:  # a hair-coloured class seeds nothing: its vertices count as unknown for the diffusion
            keep = ~np.isin(lab_known, list(red_cls))
            known[kidx[~keep]] = False
            kidx, lab_known = kidx[keep], lab_known[keep]
        _, _, srcv = dijkstra(A, directed=False, indices=kidx, unweighted=True, min_only=True,
                              return_predecessors=True)
        label = np.full(n_verts, -1, np.int32)
        label[kidx] = lab_known
        reach = srcv >= 0
        pos = np.searchsorted(kidx, srcv[reach])
        label[reach] = lab_known[pos]
        label[kidx] = lab_known
        skin_k = None
        if skin_only is not None and np.asarray(skin_only).any() and classes > 1:
            sk_t = np.asarray(skin_only, bool)
            sk_v = np.bincount(tri.ravel(), (atlas.bary * sk_t[:, None]).ravel(), n_verts) > 0.5 * np.bincount(
                tri.ravel(), atlas.bary.ravel(), n_verts).clip(min=1e-9)
            seeds = lab_known[sk_v[kidx]]
            if len(seeds) >= 10:
                cnt = np.bincount(seeds, minlength=classes)
                # skin = warm, not hair-coloured classes (lit and shadowed skin often split in two); a class must
                # hold >= 5 % of the skin-zone seeds. Fallback: the most common class there.
                warm = [k for k in range(classes) if cen is not None and cen[k, 1] > 5 and cen[k, 2] > 8
                        and k not in red_cls and cnt[k] >= 0.05 * len(seeds)]
                skin_ks = warm or [int(cnt.argmax())]
                skin_k = int(max(skin_ks, key=lambda k: cnt[k]))
                label[sk_v & ~known] = skin_k
                vstat["skin_classes"] = [int(k) for k in skin_ks]
                vstat["skin_only_hole_vertices"] = int((sk_v & ~known).sum())
        c = col.copy()
        have = known.astype(float)
        cls_med = {int(k): np.median(col[kidx[lab_known == k]], 0) for k in np.unique(lab_known)}
        for k in np.unique(label[label >= 0]):
            mk = (label == k).astype(float)
            Ak = diags(mk) @ A @ diags(mk)
            hk = have * mk
            todo = (~known) & (label == k)
            for _ in range(iters):
                num = Ak @ (c * hk[:, None])
                den = Ak @ hk
                upd = todo & (hk == 0) & (den > 0)
                if not upd.any():
                    break
                c[upd] = num[upd] / den[upd, None]
                hk = np.where(upd, 1.0, hk)
            stuck = todo & (hk == 0)
            if stuck.any() and int(k) in cls_med:  # no path inside the class: the class's median colour
                c[stuck] = cls_med[int(k)]
                hk = np.where(stuck, 1.0, hk)
            degk = np.asarray(Ak.sum(1)).ravel()
            sel = todo & (degk > 0)
            for _ in range(20):  # a few smoothing passes on the filled part of the class only
                c[sel] = (Ak @ c)[sel] / degk[sel, None]
            have = np.maximum(have, hk)
        left = (have == 0)
        if left.any():  # unreachable vertices (separate components without data): median of the good seeds
            c[left] = np.median(col[kidx], 0) if len(kidx) else col[known].mean(0)
        if skin_k is not None:  # skin-zone holes: a skin-only colour field (other classes' vertices do not feed it)
            seed_s = known & np.isin(label, skin_ks)
            todo = sk_v & ~seed_s
            mz = (sk_v | seed_s).astype(float)
            Az = diags(mz) @ A @ diags(mz)
            cs = np.where(seed_s[:, None], c, 0.0)
            hs = seed_s.astype(float)
            for _ in range(iters):
                num = Az @ (cs * hs[:, None])
                den = Az @ hs
                upd = todo & (hs == 0) & (den > 0)
                if not upd.any():
                    break
                cs[upd] = num[upd] / den[upd, None]
                hs = np.where(upd, 1.0, hs)
            cs[todo & (hs == 0)] = cls_med.get(skin_k, np.median(c[seed_s], 0) if seed_s.any() else c.mean(0))
            degz = np.asarray(Az.sum(1)).ravel()
            sel = todo & (degz > 0)
            for _ in range(20):
                cs[sel] = (Az @ cs)[sel] / degz[sel, None]
            col_skin = cs
        col = c
    vc = np.einsum("nkd,nk->nd", col[tri], atlas.bary)
    if col_skin is not None:
        hole_skin = sk_t & (alpha2 < 1)
        vc[hole_skin] = np.einsum("nkd,nk->nd", col_skin[tri[hole_skin]], atlas.bary[hole_skin])
    out = alpha2[:, None] * out + (1 - alpha2[:, None]) * vc
    src[(alpha2 < 0.5) & (src == 0)] = 2
    fill_classed.last_stats = vstat
    return out, alpha, src


def fill_region_aware(atlas, rgb_lin: np.ndarray, conf: np.ndarray, n_verts: int, conf_ref: float | None,
                      hair: np.ndarray, soft: np.ndarray, classes: int = 4, bad: np.ndarray | None = None,
                      skin_only: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalp-under-hair and body filled separately, then feathered at the hairline with `soft`. bad / skin_only:
    see fill_classed (body fill only)."""
    from .avatar_texture import fill_holes
    body_conf = np.where(hair, 0.0, conf)
    if classes > 1:
        fb, ab, sb = fill_classed(atlas, rgb_lin, body_conf, n_verts, conf_ref, classes, bad=bad,
                                  skin_only=None if skin_only is None else skin_only & ~hair)
    else:
        fb, ab, sb = fill_holes(atlas, rgb_lin, body_conf, n_verts, conf_ref)
    if not hair.any():
        return fb, ab, sb
    fh, ah_, sh = fill_holes(atlas, rgb_lin, np.where(hair, conf, 0.0), n_verts, conf_ref)
    s = np.asarray(soft, np.float32)[:, None]
    out = s * fh + (1 - s) * fb
    return out, np.where(hair, ah_, ab), np.where(hair, sh, sb).astype(np.uint8)


def fill_guards(rgb_lin: np.ndarray, conf: np.ndarray, regions: dict, hair: np.ndarray,
                ev: np.ndarray | None = None, ev_bad: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Hair-role body texels before the fill -> (conf with leaked dark hair dropped, bad seeds, skin_only, stats).

    Leak guard: a measured body texel of the upper body / neck / non-scalp head that is still DARK hair-coloured
    (red_cast and L* < 45 - her top is white, her skin has b* > a*) is hair that leaked through the exclusion: it is
    dropped and refilled (this extends the old face-core dark_hair_guard; the counts are reported because the
    dark-hair texel metric is then ~0 there by construction). Bad seeds: every red_cast measured texel (hair-over-
    white, but also the real pink print on her top, which stays where it was measured but is not copied into holes)
    and texels whose sample weight was mostly on her hair (evidence > ev_bad). skin_only: neck + non-scalp head."""
    from .avatar_texture import linear_to_srgb
    body = ~hair
    where = (regions["upper"] | regions["neck"] | regions["head"]) & body
    meas = conf > 0
    lab = lab_of(linear_to_srgb(rgb_lin))
    rc = red_cast(lab)
    dark = rc & (lab[:, 0] < 45)
    drop = where & meas & dark
    conf2 = np.where(drop, 0.0, conf)
    bad = body & meas & rc
    if ev is not None:
        frac = np.where(ev[1] > 0, ev[0] / np.maximum(ev[1], 1e-20), 0.0)
        bad |= body & meas & (frac > ev_bad)
    skin_only = (regions["neck"] | regions["head"]) & body
    names = {"upper": regions["upper"], "neck": regions["neck"], "head_not_scalp": regions["head"] & body}
    stats = {"leak_guard_dropped": {k: int((drop & v).sum()) for k, v in names.items()},
             "leak_guard_dropped_pct_of_measured": {k: round(100 * float((drop & v).sum() / max((meas & v).sum(), 1)), 2)
                                                    for k, v in names.items()},
             "bad_seed_texels": int(bad.sum()), "skin_only_texels": int(skin_only.sum())}
    return conf2, bad, skin_only, stats


# ------------------------------------------------------------------------------------------------ multi-band fusion


def lowpass_groups(torch, vals, wts, yx: np.ndarray, groups: np.ndarray, sigma: float, pool: int = 4):
    """Mask-normalised Gaussian low-pass of per-texel values inside each texel group (no blur across UV islands or
    the hairline). vals (C, N, ch), wts (C, N) >= 0 -> (C, N, ch) = G*(w v) / G*(w) and (C, N) = G*(w) (mean
    weight density in the window). Computed on a grid pooled by `pool` texels (sigma in texels)."""
    F = torch.nn.functional
    dev = vals.device
    C, N, ch = vals.shape
    p = max(1, min(pool, int(sigma // 2)))
    sp = sigma / p
    rad = max(1, int(np.ceil(3 * sp)))
    xs = torch.arange(-rad, rad + 1, device=dev, dtype=torch.float32)
    k1 = torch.exp(-0.5 * (xs / sp) ** 2)
    k1 = k1 / k1.sum()
    out_v = torch.zeros_like(vals)
    out_w = torch.zeros_like(wts)
    g_t = torch.as_tensor(np.asarray(groups), device=dev)
    yx_t = torch.as_tensor(np.asarray(yx), device=dev).long()
    nch = C * (ch + 1)
    for g in np.unique(groups):
        idx = torch.nonzero(g_t == int(g)).squeeze(1)
        y, x = yx_t[idx, 0], yx_t[idx, 1]
        py, px = y // p, x // p
        y0, x0 = int(py.min()) - rad - 1, int(px.min()) - rad - 1
        hb, wb = int(py.max()) - y0 + rad + 2, int(px.max()) - x0 + rad + 2
        lin = (py - y0) * wb + (px - x0)
        w = wts[:, idx]
        stack = torch.cat([(vals[:, idx] * w[..., None]).permute(0, 2, 1), w[:, None]], 1)  # (C, ch+1, n)
        canvas = torch.zeros((nch, hb * wb), device=dev)
        canvas.index_add_(1, lin, stack.reshape(nch, -1).float())
        img = canvas.view(1, nch, hb, wb) / (p * p)
        img = F.conv2d(img, k1.view(1, 1, 1, -1).expand(nch, 1, 1, -1), padding=(0, rad), groups=nch)
        img = F.conv2d(img, k1.view(1, 1, -1, 1).expand(nch, 1, -1, 1), padding=(rad, 0), groups=nch)
        gy = ((y.float() + 0.5) / p - 0.5 - y0) / (hb - 1) * 2 - 1  # texel centre in pooled pixels
        gx = ((x.float() + 0.5) / p - 0.5 - x0) / (wb - 1) * 2 - 1
        smp = F.grid_sample(img, torch.stack([gx, gy], 1)[None, None], mode="bilinear",
                            align_corners=True)[0, :, 0].view(C, ch + 1, -1)
        den = smp[:, ch]
        out_v[:, idx] = (smp[:, :ch] / den.clamp(min=1e-30)[:, None]).permute(0, 2, 1)
        out_w[:, idx] = den
    return out_v, out_w


def multiband(torch, layers, atlas, groups: np.ndarray | None, expo, sigma: float, sigma_high: float = 0.0,
              sharp_where=None, feather: float = 4.0):
    """Two-band fusion of per-camera layers -> linear RGB (N, 3).

    layers = (T (C,N,3) per-camera colour, Ws (C,N) sum of sharpened inlier weights, Q (C,N) sum of inlier weights);
    expo (N,) sharpening exponent per texel. high = sum_c Ws_c (T_c - LP T_c) / sum Ws (= the old sharp fusion's
    detail), low = sum_c Wl_c LP T_c / sum Wl with Wl_c = (LP Q_c / max_c LP Q_c) ^ expo: smooth across view
    boundaries, still favouring the best view (the face keeps its best camera at low frequencies too)."""
    T, Ws, Q = layers
    has = (Q > 0).float()
    grp = groups if groups is not None else island_ids(atlas)
    lpT, lpM = lowpass_groups(torch, T, has, atlas.texel_yx, grp, sigma)
    lpQ, _ = lowpass_groups(torch, Q[..., None], torch.ones_like(Q), atlas.texel_yx, grp, sigma)
    lq = lpQ[..., 0] * (lpM > 0.02)
    rel = lq / lq.max(0).values.clamp(min=1e-30)[None]
    ex = expo if torch.is_tensor(expo) else torch.full((T.shape[1],), float(expo), device=T.device)
    wl = rel.clamp(min=0) ** ex[None]
    ws = Ws * has
    if sigma_high > 0:  # high band: the sharp weights smoothed over a few texels (no per-texel camera flicker)
        wsm, _ = lowpass_groups(torch, ws[..., None], torch.ones_like(Q), atlas.texel_yx, grp, sigma_high, pool=1)
        ws = wsm[..., 0] * has
    wl = torch.where((wl.sum(0) > 0)[None], wl, ws)  # isolated texels (no camera covers 2 % of the window): sharp
    low = (wl[..., None] * lpT).sum(0) / wl.sum(0).clamp(min=1e-30)[:, None]
    high = (ws[..., None] * (T - lpT)).sum(0) / ws.sum(0).clamp(min=1e-30)[:, None]
    out = low + high
    if sharp_where is not None and bool(torch.as_tensor(sharp_where).any()):
        # one-band sharp fusion on these texels (the face: its best view at both frequencies), feathered over
        # `feather` texels inside the low-pass groups so no ring appears at the boundary
        w0 = Ws * has
        sharp = (w0[..., None] * T).sum(0) / w0.sum(0).clamp(min=1e-30)[:, None]
        m = torch.as_tensor(sharp_where, device=T.device).float()
        f, _ = lowpass_groups(torch, m[None, :, None], torch.ones((1, T.shape[1]), device=T.device), atlas.texel_yx,
                              grp, feather, pool=1)
        f = ((f[0, :, 0] - 0.2) / 0.6).clamp(0, 1)  # ~1 sigma ramp centred on the boundary
        f = torch.where(w0.sum(0) > 0, f, torch.zeros_like(f))
        out = out * (1 - f[:, None]) + sharp * f[:, None]
    none = ws.sum(0) <= 0
    out[none] = 0.0
    return out.clamp(min=0)


# ------------------------------------------------------------------------------------------------ hair sample


def hair_sample(pixels: list, view_gain: np.ndarray, out_dir: Path, role: str = "follow", scalp_srgb=None,
                strip_hw: tuple = (512, 256), palette_k: int = 6) -> dict:
    """<role>_hair.json + <role>_hair_strip.png from collect_hair_pixels samples, exposure-corrected with the
    bake's per-view linear gains (same normalisation as the body texture)."""
    from .avatar_texture import linear_to_srgb, srgb_to_linear
    if not pixels:
        return {"status": "no hair pixels"}
    vid = np.concatenate([np.full(len(p[1]), p[0], np.int64) for p in pixels])
    rgb = np.concatenate([p[1] for p in pixels]).astype(np.float32) / 255.0
    dm = np.concatenate([p[2] for p in pixels])
    lin = np.clip(srgb_to_linear(rgb) * view_gain[np.clip(vid, 0, len(view_gain) - 1)], 0, 1)
    srgb = linear_to_srgb(lin).astype(np.float32)
    L = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    p95 = float(np.percentile(dm, 95))
    s = np.clip(dm / max(p95, 1e-6), 0, 1)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(srgb), min(len(srgb), 60000), replace=False)
    lab = cv2.cvtColor(srgb[sub][None], cv2.COLOR_RGB2LAB)[0]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.05)
    cv2.setRNGSeed(0)
    _, lk, cen = cv2.kmeans(lab.astype(np.float32), palette_k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    frac = np.bincount(lk.ravel(), minlength=palette_k) / len(lk)
    cen_rgb = cv2.cvtColor(cen[None].astype(np.float32), cv2.COLOR_LAB2RGB)[0]
    order = np.argsort(cen[:, 0])
    to8 = lambda c: [int(round(float(v) * 255)) for v in np.clip(c, 0, 1)]
    palette = [{"srgb8": to8(cen_rgb[i]), "linear": [round(float(v), 4) for v in srgb_to_linear(np.clip(cen_rgb[i], 0, 1))],
                "fraction": round(float(frac[i]), 4)} for i in order]
    nb = 8
    grad = []
    for b in range(nb):
        sel = (s >= b / nb) & (s < (b + 1) / nb + (1e-6 if b == nb - 1 else 0))
        if sel.sum() >= 20:
            grad.append({"t": round((b + 0.5) / nb, 4), "srgb8": to8(linear_to_srgb(np.median(lin[sel], 0))),
                         "n": int(sel.sum())})
    # strand strip: rows root -> tip (64 bins along the hair, interpolated), columns = luminance quantiles in a fixed
    # shuffled order (each column a strand of consistent relative brightness)
    Hs, Ws_ = strip_hw
    nrow = 64
    q = (np.arange(Ws_) + 0.5) / Ws_
    perm = rng.permutation(Ws_)
    rows = np.zeros((nrow, Ws_, 3), np.float32)
    have = np.zeros(nrow, bool)
    for b in range(nrow):
        sel = (s >= b / nrow) & (s <= (b + 1) / nrow)
        if sel.sum() >= 30:
            o = np.argsort(L[sel])
            li = lin[sel][o]
            idx = np.clip((q * len(li)).astype(int), 0, len(li) - 1)
            rows[b] = li[idx][perm]
            have[b] = True
    if have.any():
        for ch in range(3):  # fill empty bins along the hair from their neighbours
            for j in range(Ws_):
                rows[~have, j, ch] = np.interp(np.flatnonzero(~have), np.flatnonzero(have), rows[have, j, ch])
    strip = cv2.resize(rows, (Ws_, Hs), interpolation=cv2.INTER_LINEAR)
    strip = cv2.GaussianBlur(strip, (1, 0), sigmaX=0.1, sigmaY=Hs / nrow)
    strip8 = (np.clip(linear_to_srgb(np.clip(strip, 0, 1)), 0, 1) * 255 + 0.5).astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    strip_name = f"{role}_hair_strip.png"
    cv2.imwrite(str(out_dir / strip_name), strip8[:, :, ::-1])
    med = linear_to_srgb(np.median(lin, 0))
    third = lambda a, b: to8(linear_to_srgb(np.median(lin[(s >= a) & (s < b)], 0))) if ((s >= a) & (s < b)).sum() else None
    rep = {"role": role, "what": "hair colour sample for the Unity hair cards / strands: exposure-normalised like the "
                                 "body albedo (per-view gains of the bake), sRGB 8-bit + linear",
           "n_pixels": int(len(srgb)), "n_views": int(len(np.unique(vid))),
           "hair_extent_from_head_centre_m_p95": round(p95, 3),
           "hair_extent_note": "95th percentile of the hair pixels' image distance from the projected head centre, "
                               "in metres at head depth: a lower bound on the visible spread, NOT the strand length "
                               "(hair_reference.json measures length below the crown)",
           "mean_srgb8": to8(linear_to_srgb(lin.mean(0))), "median_srgb8": to8(med),
           "root_srgb8": third(0.0, 0.2), "mid_srgb8": third(0.4, 0.6), "tip_srgb8": third(0.8, 1.01),
           "palette": palette, "gradient": grad, "strip": strip_name,
           "strip_layout": f"{Ws_} x {Hs} sRGB PNG; row 0 = root (scalp), last row = tips; columns = strands "
                           "(luminance quantiles in a fixed shuffled order)",
           "scalp_albedo_mean_srgb8": to8(scalp_srgb) if scalp_srgb is not None else None}
    (out_dir / f"{role}_hair.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------------------------------------ metrics


def lab_of(srgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(np.asarray(srgb, np.float32), 0, 1)[None], cv2.COLOR_RGB2LAB)[0]


def red_cast(lab: np.ndarray) -> np.ndarray:
    """Hair-coloured (red / maroon / pink hair-over-white) in CIELAB: a* > 9 and a* - b* > 3. Her skin has b* > a*,
    the white top and jeans a* < 9 (calibrated on the follow's crown 90 %, forearms 3 %, lead upper body 0.2 %).
    HM_RED_TOP=1 (carlos_aline, C7): the follow's TOP is red and her hair black, so a red-cast texel is cloth, not leaked hair: the rule is off
    (the guards then rest on the hair evidence alone: fill_guards' ev_bad, the hair masks); the recap / demo leave it unset."""
    if os.environ.get("HM_RED_TOP") == "1":
        return np.zeros(len(lab), bool)
    return (lab[:, 1] > 9) & (lab[:, 1] - lab[:, 2] > 3)


def hair_residue(lab: np.ndarray) -> np.ndarray:
    """Hair residue on a light top / skin, including DESATURATED and DARKENED hair (hair-over-white mixes, blur):
    pinkish-mauve and darker than the clean top (a* - b* > 1.5, a* > 6.5, L* < 72) or dark hair (red_cast, L* < 45).
    Her clean white top is about (79, 5, 6), skin has b* > a*. Calibrated on larissa_kadu_38's follow (back texels
    hair-covered in > 30 % of the sample weight 14 %, never covered 1.5 %; the lead's back 0.5 %). The pink print on
    her top's front also counts: compare the back."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    return ((a - b > 1.5) & (a > 6.5) & (L < 72)) | (red_cast(lab) & (L < 45))


def texture_metrics(srgb: np.ndarray, regions: dict, hair: np.ndarray | None = None) -> dict:
    """Percent of texels classified hair-coloured (red_cast; dark_hair = red_cast and L* < 45; hair_residue incl.
    desaturated / darkened hair, see hair_residue) per body region."""
    lab = lab_of(srgb)
    rc = red_cast(lab)
    res = hair_residue(lab)
    dark = rc & (lab[:, 0] < 45)
    sel = {"chest_shoulders_back": regions["upper"], "chest": regions["upper"] & regions["front"] & ~regions["neck"],
           "back": regions["upper"] & ~regions["front"] & ~regions["neck"], "shoulders": regions["shoulders"],
           "neck": regions["neck"], "face_core": regions["face_core"], "forearms_baseline": regions["forearm"]}
    if hair is not None:
        sel["face_not_hair"] = regions["face"] & ~hair
    out = {}
    for k, m in sel.items():
        if m.any():
            out[k] = {"hair_coloured_pct": round(100 * float(rc[m].mean()), 2),
                      "dark_hair_pct": round(100 * float(dark[m].mean()), 2),
                      "hair_residue_pct": round(100 * float(res[m].mean()), 2), "texels": int(m.sum())}
    return out


# ------------------------------------------------------------------------------------------------ CLI


def _metrics_cli(d: Path) -> dict:
    from .avatar_uv import build_atlas
    from .mamma_dyn import SMPLX_MODELS
    rep = {}
    for r in ("lead", "follow"):
        z = np.load(d / f"{r}_texels.npz")
        S = int(round(np.sqrt(len(z["rgb_srgb"]) / 0.734)))  # atlas size from the texel count (~73 % filled)
        size = min((512, 1024, 2048, 4096), key=lambda s: abs(s - S))
        atlas = build_atlas(SMPLX_MODELS, size)
        reg = texel_regions(atlas, SMPLX_MODELS)
        hair = (np.asarray(z["hair_region"]) > 127) if "hair_region" in z.files else None
        rep[r] = texture_metrics(np.asarray(z["rgb_srgb"], np.float32), reg, hair)
    return rep


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="python -m dancecap.avatar_hair")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("metrics", help="hair-coloured texel fractions of an avatar output dir")
    sp.add_argument("dir")
    sp = sub.add_parser("compare", help="before/after review sheets + held-out photometric error")
    sp.add_argument("take")
    sp.add_argument("--before", required=True)
    sp.add_argument("--after", required=True)
    sp.add_argument("--review", required=True, help="directory for the review images")
    sp.add_argument("--extra", action="append", default=[], help="LABEL=DIR more avatar dirs for the metrics table")
    sp.add_argument("--heldout-cam", help="camera left out of both bakes (leave-one-camera-out check)")
    sp.add_argument("--per-cam", type=int, default=3, help="held-out frames per camera")
    sp.add_argument("--json", help="write the comparison numbers here")
    sp.add_argument("--label", help="take label on the sheets (e.g. 'larissa_kadu (6 s dev take)')")
    sp.add_argument("--no-sheets", action="store_true", help="numbers only, no review images")
    sp.add_argument("--device", help="cuda|cpu")
    a = p.parse_args(argv)
    if a.cmd == "metrics":
        print(json.dumps(_metrics_cli(Path(a.dir)), indent=1))
        return
    from .avatar_hair_review import compare
    rep = compare(a.take, Path(a.before), Path(a.after), Path(a.review), extra=dict(e.split("=", 1) for e in a.extra),
                  heldout_cam=a.heldout_cam, per_cam=a.per_cam, tag=f"_loco{a.heldout_cam}" if a.heldout_cam else "",
                  label=a.label, sheets=not a.no_sheets, device=a.device)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep.get("summary", rep), indent=1, default=float))


if __name__ == "__main__":
    main()
