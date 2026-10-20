r"""Render track 'hybrid' of the appearance comparison (2026-10-19): a Quest-native textured skinned MESH avatar whose
appearance comes from LHM++ - the middle option between the current footage-baked avatar and moving Gaussian splats.

    python -m dancecap.mono4d.appearance_hybrid all       # fit, facereg, bake (+ fidelity), shell, render, score, check,
                                                          # manifest - about 25 min on the shared 2080 Ti, no pod
    python -m dancecap.mono4d.appearance_hybrid fit | facereg | bake | fidelity | shell | score | check | manifest
    python -m dancecap.mono4d.appearance_hybrid render [--method hybrid_mesh|hybrid_mesh_shell] [--groups ...] [--ids ...]
    (fit / facereg / bake / fidelity / shell / all re-run themselves inside the MSVC + CUDA env on Windows: gsplat's JIT)

Splats: gs_source() - the compare track's improved one-phone LHM++ splats (compare/lhmpp/pod_out/gs/<role>_lhmpp_1phone
_cano.ply: 8 phone-06 crops without white-patch crops, shape mode ref) when present, else mono4d/lhmpp/pod_out/gs/
<role>_w8_ref_cano.ply. 160k Gaussians each, standard 3DGS PLY (SH DC colour, logit opacity, log scales, wxyz quats).

Steps
  1. fit      LHM++'s canonical Gaussians are posed (LHM++ @ 906b5d9, base_gs_render._prepare_smplx_data) with root 0,
              trans 0, hips z +-15 deg, shoulders z -+30 deg, everything else 0, betas = reference betas[:10]. The SHIPPED
              skin (export/04_LarissaKaduDemo/smplx_<role>_skin.bin: 16 betas + shape edits, seam-split SMPL-X, Unity
              UVs) is LBS-posed with exactly that pose (LHMPP_CANO_POSE) and only a translation is fitted (robust two-sided
              nearest-point distance, CPU; a free 22-joint fit drifted into twisted hips / a 41 deg head pitch and was
              dropped) -> hybrid/apose_<role>.npz/json, check/apose_<role>.jpg (splats + skin outline + joints).
  1b. facereg LHM++'s face is not where the skin's eye openings / nose / mouth are (the lead's LHM++ eyes sat ~1.4 cm
              high, on the skin's brow; the follow's profile 1.1 cm forward): a similarity found by masked NCC of
              high-passed frontal + profile renders against the skin wearing the CURRENT shipped albedo (whose portrait
              face is painted into the skin's own openings) moves the head Gaussians (blended by head skinning weight)
              -> face_reg_<role>.npz/json, check/face_reg_<role>.jpg. load_gs() applies it.
  2. bake     gsplat (LHM++'s renderer settings: classic, eps2d 0.3) renders RGB + expected depth of the (registered)
              canonical splats from 48 orbit cameras (4 elevations x 12 azimuths) + 14-direction close-ups of head, hands
              and feet. Every texel of the skin's 2048 x 2048 UV atlas (the shipped albedo's layout) is a point on the
              A-posed skin. Pass 1: its offset along the normal to the splat surface (expected depth, near-frontal
              mesh-visible views) -> a smoothed per-vertex offset field. Pass 2: the texel point pushed out by that offset
              is projected into every render; visible when the render's expected depth agrees; weight = depth agreement x
              cos^p (p 4 body, 12 head; face / eyeball texels use the normal tilted towards the face direction, so the
              SMPL-X nostril / eyelid / lip rims do not pick oblique colours) x resolution x alpha^2; weighted mean in
              linear RGB; holes filled like the footage bake (avatar_texture.fill_holes: mirror, then mesh diffusion) +
              gutter -> hybrid/<role>_albedo.png (Unity format: the same SMPL-X UV layout, 2048, PNG row 0 = v 1),
              <role>_coverage.png, <role>_offset.png, <role>_offsets.npz, bake_<role>.json.
      fidelity the A-posed skin with the new albedo vs the splats from cameras between the bake cameras ->
              check/fidelity_<role>.jpg + fidelity_<role>.json (what the bake keeps / loses).
  3. shell    (optional) the skin subdivided once (midpoints; LBS weights / UVs interpolated) and pushed out along a
              smoothed normal field by the offset field (clothes <= 3.5 cm, the follow's hair <= 12 cm on head / upper
              back, the face <= 4 mm, a stretch limiter so curved parts such as the ears do not balloon) ->
              hybrid/smplx_<role>_skin.bin (the export's DCXS v2 skin format, Unity coordinates), then the same bake on
              that geometry -> hybrid/shell/<role>_albedo.png (bake_shell_<role>.json).
  4. render   every view of compare/views.json (appearance_avatar_ags.build_jobs) with the SAME renderer as the current
              avatar (appearance_avatar_ags.AvatarRenderer: plain LBS like SmplxAvatar.cs, the reference fit, unlit,
              avatar_texture.render, 2 x 2 supersampled); only the albedo (and for hybrid_mesh_shell the skin) differ
              -> compare/renders/<method>/ (same file names / manifest schema as the other tracks; manifest seen_by says
              which phone-06 frames LHM++ saw).
      score   masked PSNR / SSIM vs the 18 real photos (no LPIPS locally) for every method present -> hybrid/scores.json.
      check   hybrid/check/: real_<instant>.jpg, turntable.jpg, closeups.jpg, clips.jpg, albedo_atlases.jpg.
      manifest hybrid/manifest.json: the files and how the Unity build would load them.

Likeness data (splats, renders, textures) stays under work/ (git-ignored). LHM++ weights are CC BY-NC 4.0: the albedo
is derived from their output - research use only.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT
from .appearance_compare import AVATAR_CAPTION, ROLES, _rel, compare_dir

TAKE = "larissa_kadu_38"
CAPTURE = "04_LarissaKaduDemo"
LHMPP_GS = PACKAGE_ROOT / "work" / TAKE / "mono4d" / "lhmpp" / "pod_out" / "gs"
SH_C0 = 0.28209479177387814
SIZE = 2048
FACE_FORWARD_TILT = 2.0   # face texels: view weights use normalize(n + 2 * forward) (the canonical head faces +z)
SHELL_SMOOTH_ITERS = 30    # shell: Laplacian smoothing passes of the per-vertex offset field (steep cliffs stretch the texture)
SHELL_MAX_STRETCH = 2.5    # shell: max area growth of a face from the push (more = streaky texture)
SHELL_NORMAL_SMOOTH = 40   # shell: smoothing passes of the push direction (on the once-subdivided mesh)
MIRROR = np.array([1.0, 1.0, -1.0])
METHODS = ("hybrid_mesh", "hybrid_mesh_shell")


def log(msg: str) -> None:
    print(f"[hybrid] {msg}", flush=True)


def hybrid_dir(take: str = TAKE) -> Path:
    return compare_dir(take) / "hybrid"


# ------------------------------------------------------------------------------------------------ inputs
def gs_source(role: str, take: str = TAKE) -> Path:
    """Prefer the compare track's improved one-phone LHM++ splats when they exist, else the 2026-10-19 w8_ref."""
    cdir = compare_dir(take)
    for p in (cdir / "renders" / "lhmpp" / f"{role}_lhmpp_1phone_cano.ply",
              cdir / "renders" / "lhmpp_1phone" / f"{role}_cano.ply",
              cdir / "lhmpp" / "pod_out" / "gs" / f"{role}_lhmpp_1phone_cano.ply"):
        if p.exists():
            return p
    return LHMPP_GS / f"{role}_w8_ref_cano.ply"


def read_gs(path: Path) -> dict:
    """3DGS PLY (LHM++ GaussianModel.save_ply) -> xyz, rgb (SH DC -> [0, 1]), opacity (sigmoid), scales (exp), quats wxyz."""
    from plyfile import PlyData
    d = PlyData.read(str(path))["vertex"].data
    g = lambda *n: np.stack([np.asarray(d[k], np.float32) for k in n], 1)  # noqa: E731
    q = g("rot_0", "rot_1", "rot_2", "rot_3")
    return {"xyz": g("x", "y", "z"), "rgb": np.clip(g("f_dc_0", "f_dc_1", "f_dc_2") * SH_C0 + 0.5, 0, 1),
            "opacity": 1 / (1 + np.exp(-np.asarray(d["opacity"], np.float32))),
            "scales": np.exp(g("scale_0", "scale_1", "scale_2")),
            "quats": q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)}


def load_skin(role: str, take: str = TAKE) -> dict:
    """The shipped skin in the right-handed world frame (un-mirrored, right-handed winding)."""
    from ..export_unity import read_skin
    p = PACKAGE_ROOT / "work" / take / "export" / CAPTURE / f"smplx_{role}_skin.bin"
    h, sk = read_skin(p)
    return {"path": p, "header": h, "V": np.asarray(sk.vertices, float) * MIRROR,
            "faces": np.asarray(sk.faces, np.int64)[:, [0, 2, 1]], "W": np.asarray(sk.weights, float),
            "J": np.asarray(sk.joints, float) * MIRROR, "parents": np.asarray(sk.parents, np.int64),
            "uv": np.asarray(sk.uv, float), "vid": np.asarray(sk.vertex_ids, np.int64), "skin": sk}


# ------------------------------------------------------------------------------------------------ kinematics (torch)
def rodrigues_t(torch, rv):
    th = torch.sqrt((rv * rv).sum(-1, keepdim=True) + 1e-16)
    k = rv / th
    K = torch.zeros(rv.shape[:-1] + (3, 3), dtype=rv.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    eye = torch.eye(3, dtype=rv.dtype).expand(K.shape)
    return eye + s * K + (1 - c) * (K @ K)


def fk_t(torch, J, parents, Rl, transl):
    Rg, P = [None] * len(parents), [None] * len(parents)
    for j, p in enumerate(parents):
        if p < 0:
            Rg[j], P[j] = Rl[j], J[j] + transl
        else:
            Rg[j] = Rg[p] @ Rl[j]
            P[j] = P[p] + Rg[p] @ (J[j] - J[p])
    return torch.stack(P), torch.stack(Rg)


def lbs_t(torch, V, W, J, P, Rg):
    rel = V[:, None, :] - J[None]
    posed = torch.einsum("jab,vjb->vja", Rg, rel) + P[None]
    return torch.einsum("vj,vja->va", W, posed)


def lbs_np(V, W, J, parents, rv, transl):
    import torch
    with torch.no_grad():
        Rl = rodrigues_t(torch, torch.as_tensor(rv, dtype=torch.float64))
        P, Rg = fk_t(torch, torch.as_tensor(J), list(parents), Rl, torch.as_tensor(transl, dtype=torch.float64))
        Vp = lbs_t(torch, torch.as_tensor(V), torch.as_tensor(W), torch.as_tensor(J), P, Rg)
    return Vp.numpy(), P.numpy(), Rg.numpy()


def vertex_normals(V, F):
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    out = np.zeros_like(V)
    for k in range(3):
        np.add.at(out, F[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def unsplit_normals(V, F, vid):
    """Normals of the seam-split mesh computed on the unsplit topology (no seams in the normal field)."""
    n_u = int(vid.max()) + 1
    Vu = np.zeros((n_u, 3))
    Vu[vid] = V
    Nu = vertex_normals(Vu, vid[F])
    return Nu[vid]


# ------------------------------------------------------------------------------------------------ cameras / gsplat
def look_at(C, P, up=(0.0, 1.0, 0.0)):
    z = np.asarray(P, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    u = np.asarray(up, float)
    if abs(z @ u) > 0.98:
        u = np.array([0.0, 0.0, 1.0])
    x = np.cross(-u, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ np.asarray(C, float)


def cam(C, target, W, H, f):
    R, t = look_at(C, target)
    return {"K": np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]]), "R": R, "tvec": t, "W": W, "H": H,
            "C": np.asarray(C, float)}


class GSRenderer:
    """gsplat rasterisation of one canonical splat set with the LHM++ renderer's settings (classic, eps2d 0.3)."""

    def __init__(self, G: dict, device="cuda"):
        import torch
        from gsplat import rasterization
        self.torch, self.rast, self.dev = torch, rasterization, device
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
        self.means, self.quats, self.scales = T(G["xyz"]), T(G["quats"]), T(G["scales"])
        self.opac, self.rgb = T(G["opacity"]), T(G["rgb"])

    def render(self, c: dict) -> dict:
        torch = self.torch
        vm = torch.eye(4, device=self.dev)
        vm[:3, :3] = torch.as_tensor(c["R"], dtype=torch.float32, device=self.dev)
        vm[:3, 3] = torch.as_tensor(c["tvec"], dtype=torch.float32, device=self.dev)
        K = torch.as_tensor(c["K"], dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            out, alpha, _ = self.rast(means=self.means, quats=self.quats, scales=self.scales, opacities=self.opac,
                                      colors=self.rgb, viewmats=vm[None], Ks=K[None], width=int(c["W"]),
                                      height=int(c["H"]), near_plane=0.01, far_plane=100.0, eps2d=0.3,
                                      render_mode="RGB+ED", rasterize_mode="classic", packed=False)
        a = alpha[0, ..., 0]
        return {"rgb_prem": out[0, ..., :3], "depth": out[0, ..., 3], "alpha": a}


# ------------------------------------------------------------------------------------------------ step 1: A-pose fit
# LHM++ @ 906b5d9, core/models/rendering/base_gs_render.py _prepare_smplx_data: the canonical Gaussians animate_gs_model
# returns (the last, appended view; saved as <role>_<variant>_cano.ply by our pod script) are posed with root 0, trans 0,
# jaw / eyes / hands / expression 0, betas = the betas fed in (reference betas[:10] in shape mode 'ref') and body_pose 0
# except  body_pose[0, 2] = +pi/12 (L hip), body_pose[1, 2] = -pi/12 (R hip), body_pose[15, 2] = -pi/6 (L shoulder),
# body_pose[16, 2] = +pi/6 (R shoulder)   (body_pose index i = SMPL-X joint i + 1).
LHMPP_CANO_POSE = {1: (0.0, 0.0, math.pi / 12), 2: (0.0, 0.0, -math.pi / 12),
                   16: (0.0, 0.0, -math.pi / 6), 17: (0.0, 0.0, math.pi / 6)}


def fit_apose(role: str, take: str = TAKE, iters: int = 150, log=log) -> dict:
    """The shipped skin in LHM++'s canonical pose (LHMPP_CANO_POSE, from its source) + a fitted translation (the skin's
    root joint comes from 16 betas + shape edits, LHM++'s from betas[:10]): robust two-sided nearest-point distance between
    the opaque Gaussian centres and the skin vertices (CPU)."""
    import torch
    from scipy.spatial import cKDTree
    sk = load_skin(role, take)
    src = gs_source(role, take)
    G = read_gs(src)
    keep = G["opacity"] > 0.5
    Gp = G["xyz"][keep].astype(np.float64)
    rng = np.random.default_rng(0)
    Gs = Gp[rng.choice(len(Gp), min(40000, len(Gp)), replace=False)]
    treeG = cKDTree(Gp)
    V0, W, J = (torch.as_tensor(sk[k]) for k in ("V", "W", "J"))
    parents = list(sk["parents"])
    rv_full = np.zeros((55, 3))
    for j, v in LHMPP_CANO_POSE.items():
        rv_full[j] = v
    Rl = rodrigues_t(torch, torch.as_tensor(rv_full))
    tr = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([tr], lr=0.002)
    sig = 0.03
    rho = lambda d2: d2 / (d2 + sig * sig)  # noqa: E731
    Gs_t, Gp_t = torch.as_tensor(Gs), torch.as_tensor(Gp)
    hist = []
    for it in range(iters):
        P, Rg = fk_t(torch, J, parents, Rl, tr)
        V = lbs_t(torch, V0, W, J, P, Rg)
        Vd = V.detach().numpy()
        _, iG = treeG.query(Vd, workers=-1)
        l1 = rho(((V - Gp_t[iG]) ** 2).sum(1)).mean()
        _, iV = cKDTree(Vd).query(Gs, workers=-1)
        l2 = rho(((Gs_t - V[iV]) ** 2).sum(1)).mean()
        loss = l1 + l2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append([it, round(float(l1), 5), round(float(l2), 5)])
            log(f"{role} fit it {it}: mesh->gs {float(l1):.4f} gs->mesh {float(l2):.4f} transl {tr.detach().numpy()}")
    Va, Pa, Ra = lbs_np(sk["V"], sk["W"], sk["J"], sk["parents"], rv_full, tr.detach().numpy())
    d_vg, _ = treeG.query(Va, workers=-1)
    d_gv, _ = cKDTree(Va).query(Gp, workers=-1)
    head = sk["W"][:, [15, 22, 23, 24]].sum(1) > 0.5
    d_head, _ = treeG.query(Va[head], workers=-1)
    rep = {"role": role, "splats": _rel(src), "skin": _rel(sk["path"]), "gaussians": int(len(G["xyz"])),
           "gaussians_opaque": int(keep.sum()), "iters": iters, "history": hist,
           "pose": "LHM++ canonical (base_gs_render._prepare_smplx_data): hips z +-15 deg, shoulders z -+30 deg",
           "pose_deg": {str(j): np.round(np.degrees(v), 1).tolist() for j, v in LHMPP_CANO_POSE.items()},
           "transl_m": np.round(tr.detach().numpy(), 4).tolist(),
           "skin_vertex_to_splats_mm": {"median": round(1e3 * float(np.median(d_vg)), 1),
                                        "p90": round(1e3 * float(np.percentile(d_vg, 90)), 1)},
           "head_vertex_to_splats_mm": {"median": round(1e3 * float(np.median(d_head)), 1),
                                        "p90": round(1e3 * float(np.percentile(d_head, 90)), 1)},
           "splats_to_skin_mm": {"median": round(1e3 * float(np.median(d_gv)), 1),
                                 "p90": round(1e3 * float(np.percentile(d_gv, 90)), 1),
                                 "p99": round(1e3 * float(np.percentile(d_gv, 99)), 1)}}
    out = hybrid_dir(take)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"apose_{role}.npz", rv=rv_full, transl=tr.detach().numpy(), V=Va, P=Pa, R=Ra)
    (out / f"apose_{role}.json").write_text(json.dumps(rep, indent=1))
    log(f"{role} A-pose: transl {rep['transl_m']}; skin->splats {rep['skin_vertex_to_splats_mm']} head "
        f"{rep['head_vertex_to_splats_mm']} splats->skin {rep['splats_to_skin_mm']}")
    return rep


def load_apose(role: str, take: str = TAKE) -> dict:
    z = np.load(hybrid_dir(take) / f"apose_{role}.npz")
    return {k: z[k] for k in z.files}


def _outline(img, sil, col, th=1):
    cs, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, col, th, cv2.LINE_AA)


def mesh_mask(V, F, c) -> tuple[np.ndarray, np.ndarray]:
    """Exact z-buffer of a mesh in camera c (avatar_raster, CPU torch) -> (H, W) depth (inf empty), face id."""
    import torch

    from ..avatar_raster import EMPTY_Z, rasterize
    Xc = V @ c["R"].T + c["tvec"]
    z = Xc[:, 2]
    px = np.stack([c["K"][0, 0] * Xc[:, 0] / z + c["K"][0, 2], c["K"][1, 1] * Xc[:, 1] / z + c["K"][1, 2]], 1)
    zb, fid = rasterize(torch, torch.as_tensor(px, dtype=torch.float32), torch.as_tensor(z, dtype=torch.float32),
                        torch.as_tensor(F), int(c["H"]), int(c["W"]))
    zb = zb.numpy()
    return np.where(zb < EMPTY_Z, zb, np.inf), fid.numpy()


def apose_check(role: str, take: str = TAKE, device="cuda") -> Path:
    """Front / left / back / right / head renders of the canonical splats with the fitted A-pose skin's outline (cyan),
    its joints (yellow) and a semi-transparent wireframe -> hybrid/check/apose_<role>.jpg."""
    sk, A = load_skin(role, take), load_apose(role, take)
    G = read_gs(gs_source(role, take))
    gr = GSRenderer(G, device)
    V, F = A["V"], sk["faces"]
    ctr = 0.5 * (V.max(0) + V.min(0))
    hgt = float(V[:, 1].max() - V[:, 1].min())
    tiles = []
    for az, target, dist, size in ((0, ctr, 2.4 * hgt, 900), (90, ctr, 2.4 * hgt, 900), (180, ctr, 2.4 * hgt, 900),
                                   (270, ctr, 2.4 * hgt, 900), (20, A["P"][15] + [0, 0.05, 0], 0.55, 900)):
        f = size / 2 / math.tan(math.radians(12))
        C = target + dist * np.array([math.sin(math.radians(az)), 0.05, math.cos(math.radians(az))])
        c = cam(C, target, size, size, f)
        r = gr.render(c)
        a = r["alpha"].cpu().numpy()
        rgb = r["rgb_prem"].cpu().numpy() + (1 - a[..., None])  # over white
        img = (np.clip(rgb, 0, 1)[..., ::-1] * 255).astype(np.uint8).copy()
        zb, fid = mesh_mask(V, F, c)
        wire = img.copy()
        Xc = V @ c["R"].T + c["tvec"]
        px = np.stack([f * Xc[:, 0] / Xc[:, 2] + c["K"][0, 2], f * Xc[:, 1] / Xc[:, 2] + c["K"][1, 2]], 1)
        vis_f = np.unique(fid[fid >= 0])
        for tri in F[vis_f]:
            cv2.polylines(wire, [np.round(px[tri] * 4).astype(np.int32)], True, (255, 200, 0), 1, cv2.LINE_AA, shift=2)
        img = cv2.addWeighted(img, 0.8, wire, 0.2, 0)
        _outline(img, np.isfinite(zb), (255, 255, 0), 2)
        Pj = A["P"][:25] @ c["R"].T + c["tvec"]
        for j in range(25):
            cv2.circle(img, (int(f * Pj[j, 0] / Pj[j, 2] + c["K"][0, 2]), int(f * Pj[j, 1] / Pj[j, 2] + c["K"][1, 2])),
                       4, (0, 220, 255), -1, cv2.LINE_AA)
        sil_g = a > 0.5
        iou = float((sil_g & np.isfinite(zb)).sum() / max((sil_g | np.isfinite(zb)).sum(), 1))
        cv2.putText(img, f"{role} az {az} IoU(splat a>0.5, skin) {iou:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA))
    out = hybrid_dir(take) / "check" / f"apose_{role}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


# ------------------------------------------------------------------------------------------------ step 1b: face register
HEAD_J = (15, 22, 23, 24)


def _head_weight(sk: dict, V: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per point: the head (+ half the neck) skinning weight of the nearest A-posed skin vertices (4-NN mean)."""
    from scipy.spatial import cKDTree
    W = sk["W"]
    wv = np.clip(W[:, list(HEAD_J)].sum(1) + 0.5 * W[:, 12], 0, 1)
    _, i = cKDTree(V).query(X, k=4, workers=-1)
    return wv[i].mean(1)


FACE_DIST = 0.9


def _face_cams(A: dict, S: int = 512, span: float = 0.30, dist: float = FACE_DIST):
    hc = A["P"][15] + np.array([0.0, 0.05, 0.0])
    f = S * dist / span
    return hc, f, {"front": cam(hc + [0, 0, dist], hc, S, S, f), "side": cam(hc + [dist, 0, 0], hc, S, S, f)}


def _hp(gray: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), sigma)
    return g - cv2.GaussianBlur(g, (0, 0), 12)


def vertex_regions_split(sk: dict) -> dict:
    from ..avatar_uv import vertex_regions
    from ..export_unity import MAMMA_MODEL
    reg = vertex_regions(np.load(MAMMA_MODEL, allow_pickle=True))
    return {k: v[sk["vid"]] for k, v in reg.items()}


def _apply_reg(G: dict, k: float, t: np.ndarray, hc: np.ndarray, w: np.ndarray) -> dict:
    Xn = hc + k * (G["xyz"] - hc) + t
    out = dict(G)
    out["xyz"] = (G["xyz"] + w[:, None] * (Xn - G["xyz"])).astype(np.float32)
    out["scales"] = (G["scales"] * (1 + w[:, None] * (k - 1))).astype(np.float32)
    return out


def load_gs(role: str, take: str = TAKE, registered: bool = True) -> dict:
    """The canonical splats, with the face registration applied when hybrid/face_reg_<role>.npz exists."""
    G = read_gs(gs_source(role, take))
    p = hybrid_dir(take) / f"face_reg_{role}.npz"
    if registered and p.exists():
        z = np.load(p)
        if len(z["w"]) == len(G["xyz"]):
            G = _apply_reg(G, float(z["k"]), z["t"], z["hc"], z["w"])
            G["registered"] = True
    return G


def face_register(role: str, take: str = TAKE, device="cuda", log=log) -> dict:
    """LHM++ places its face where its own head prior puts it; the shipped skin has its eye openings / nose / mouth where
    the MAMMA-fitted SMPL-X head has them. They differ by up to ~2.5 cm (the lead's LHM++ eyes land on the skin's brow,
    the skin's eye openings then show bare eyeballs). Registration in the canonical A-pose (both upright, facing +z):
    the skin with the CURRENT shipped albedo (its portrait face is painted into the skin's own eye openings) vs the
    splats, frontal + profile renders, high-passed grey, brute-force masked normalised cross-correlation over scale
    (0.80-1.20) x translation inside the skin's face region -> a similarity (scale about the head centre + translation)
    applied to the head Gaussians, blended by the head skinning weight (neck 0.5) -> hybrid/face_reg_<role>.npz/json +
    check/face_reg_<role>.jpg (skin + current albedo | splats before | after, skin edges in cyan)."""
    import torch

    from ..avatar_texture import render as tex_render
    from .appearance_avatar_ags import _TexGeo
    sk, A = load_skin(role, take), load_apose(role, take)
    G = read_gs(gs_source(role, take))
    hc, f, cams_ = _face_cams(A)
    alb_p = PACKAGE_ROOT / "work" / take / "export" / CAPTURE / f"{role}_albedo.png"
    img = cv2.imread(str(alb_p), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
    alb = torch.as_tensor(img.transpose(2, 0, 1).copy(), device=device)
    geo = _TexGeo(torch, sk["faces"], sk["uv"], device)
    reg = vertex_regions_split(sk)

    def renders(Gx, c):
        r = GSRenderer(Gx, device).render(c)
        a = r["alpha"].cpu().numpy()
        gs = r["rgb_prem"].cpu().numpy() + 0.5 * (1 - a[..., None])
        with torch.no_grad():
            m, idm = tex_render(torch, geo, {role: A["V"]}, {role: alb}, c["K"], c["R"], c["tvec"], 0.0, c["H"], c["W"],
                                device)
        m = np.where((idm >= 0)[..., None], m, 0.5)
        _, fid = mesh_mask(A["V"], sk["faces"], c)
        fm = (fid >= 0) & reg["face"][sk["faces"][np.clip(fid, 0, None)]].all(-1)
        return gs.astype(np.float32), m.astype(np.float32), fm

    def match(gs, m, fm, scales, max_shift_px):
        """Best (ncc, k, shift): the splat image scaled by k about the image centre, then shifted by -shift, matches the
        skin's face. A splat pixel p maps to k (p - c0) + c0 - shift."""
        gm = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        gg = cv2.cvtColor(gs, cv2.COLOR_RGB2GRAY)
        tpl_full = _hp(gm, 3.0)
        ys, xs = np.nonzero(fm)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tpl = tpl_full[y0:y1, x0:x1].astype(np.float32)
        tmask = fm[y0:y1, x0:x1].astype(np.float32)
        H, W = gg.shape
        c0 = np.array([W / 2 - 0.5, H / 2 - 0.5])
        hp = _hp(gg, 2.0)
        best = None
        for k in scales:
            M = np.array([[k, 0, (1 - k) * c0[0]], [0, k, (1 - k) * c0[1]]])
            warped = cv2.warpAffine(hp, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
            res = np.nan_to_num(cv2.matchTemplate(warped, tpl, cv2.TM_CCORR_NORMED, mask=tmask), nan=-1.0)
            yy, xx = np.mgrid[0:res.shape[0], 0:res.shape[1]]
            res[(np.abs(yy - y0) > max_shift_px) | (np.abs(xx - x0) > max_shift_px)] = -1
            j = np.unravel_index(np.argmax(res), res.shape)
            if best is None or res[j] > best[0]:
                best = (float(res[j]), float(k), int(j[1] - x0), int(j[0] - y0))
        return {"ncc": round(best[0], 4), "k": best[1], "shift_px": [best[2], best[3]]}

    w = _head_weight(sk, A["V"], G["xyz"].astype(float))
    gs0, m0, fm0 = renders(G, cams_["front"])
    fr = match(gs0, m0, fm0, np.round(np.arange(0.80, 1.201, 0.02), 3), 70)
    k = fr["k"]
    # frontal camera (looking -z): u = +x, v = -y; a splat pixel moves by -shift (after the scale about the centre)
    t_front = np.array([-fr["shift_px"][0] / f * FACE_DIST, fr["shift_px"][1] / f * FACE_DIST, 0.0])
    G1 = _apply_reg(G, k, t_front, hc, w)
    gs1, m1, fm1 = renders(G1, cams_["side"])
    sr = match(gs1, m1, fm1, [1.0], 45)
    # side camera (at +x, looking -x): u = -z; the splat pixel moves by -shift_u -> dz = +shift_u metres-scaled
    dz = float(np.clip(sr["shift_px"][0] / f * FACE_DIST, -0.025, 0.025))
    t = t_front + np.array([0.0, 0.0, dz])
    G2 = _apply_reg(G, k, t, hc, w)
    gsf, mf, fmf = renders(G2, cams_["front"])
    gss, ms, fms = renders(G2, cams_["side"])
    fr2 = match(gsf, mf, fmf, [1.0], 25)
    sr2 = match(gss, ms, fms, [1.0], 25)
    rep = {"role": role, "splats": _rel(gs_source(role, take)), "reference": _rel(alb_p),
           "front": fr, "side": sr, "scale": round(float(k), 4), "translation_m": np.round(t, 4).tolist(),
           "residual_px_after": {"front": fr2["shift_px"], "side": sr2["shift_px"]},
           "ncc_after": {"front": fr2["ncc"], "side": sr2["ncc"]},
           "mm_per_px": round(float(FACE_DIST / f * 1e3), 3), "gaussians_moved": int((w > 0.05).sum())}
    np.savez(hybrid_dir(take) / f"face_reg_{role}.npz", k=k, t=t, hc=hc, w=w)
    (hybrid_dir(take) / f"face_reg_{role}.json").write_text(json.dumps(rep, indent=1))
    rows = []
    gsb_side, ms0, _ = renders(G, cams_["side"])
    for gsb, mm, gsa in ((gs0, m0, gsf), (gsb_side, ms0, gss)):
        edges = cv2.Canny((cv2.cvtColor(mm, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8), 30, 80) > 0
        tiles = []
        for im, lab in ((mm, "skin + current albedo"), (gsb, "LHM++ splats before"), (gsa, "after registration")):
            t8 = (np.clip(im, 0, 1)[..., ::-1] * 255).astype(np.uint8).copy()
            t8[edges] = (255, 255, 0)
            tiles.append(_lab(t8, lab))
        rows.append(np.hstack(tiles))
    (hybrid_dir(take) / "check").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(hybrid_dir(take) / "check" / f"face_reg_{role}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    log(f"{role} face registration: {json.dumps(rep)}")
    return rep


# ------------------------------------------------------------------------------------------------ step 2: bake
def bake_cameras(V: np.ndarray, P: np.ndarray, sk: dict) -> list[dict]:
    """48 orbit cameras + 14-direction close-ups of the head, hands and feet around the A-posed skin."""
    cams = []
    lo, hi = V.min(0), V.max(0)
    ctr = 0.5 * (lo + hi)
    half = 0.5 * float(np.linalg.norm(hi - lo))
    S = 1280
    fov = math.radians(36)
    dist = half / math.sin(fov / 2) * 1.02
    f = S / 2 / math.tan(fov / 2)
    for el in (-35, -5, 25, 55):
        for k in range(12):
            az = math.radians(30 * k + (15 if el in (-5, 55) else 0))
            e = math.radians(el)
            C = ctr + dist * np.array([math.cos(e) * math.sin(az), math.sin(e), math.cos(e) * math.cos(az)])
            cams.append({**cam(C, ctr, S, S, f), "id": f"orbit_e{el}_a{int(math.degrees(az))}", "part": "body"})
    dirs = [np.array(d, float) for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    dirs += [np.array(d, float) / math.sqrt(3) for d in
             ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1))]
    W = sk["W"]
    parts = {"head": ([15, 22, 23, 24, 12], 0.17, 896), "lhand": (list(range(20, 21)) + list(range(25, 40)), 0.13, 640),
             "rhand": (list(range(21, 22)) + list(range(40, 55)), 0.13, 640), "lfoot": ([7, 10], 0.16, 640),
             "rfoot": ([8, 11], 0.16, 640)}
    for name, (joints, rad, size) in parts.items():
        sel = W[:, joints].sum(1) > 0.5
        pc = 0.5 * (V[sel].max(0) + V[sel].min(0))
        r = max(rad, 0.5 * float(np.linalg.norm(V[sel].max(0) - V[sel].min(0))))
        fv = math.radians(30)
        d = r / math.sin(fv / 2) * 1.05
        fp = size / 2 / math.tan(fv / 2)
        for n, dvec in enumerate(dirs):
            C = pc + d * dvec
            if np.min(np.linalg.norm(V - C, axis=1)) < 0.12:   # inside / against the body: useless
                continue
            cams.append({**cam(C, pc, size, size, fp), "id": f"{name}_{n:02d}", "part": name})
    return cams


def _bilinear(torch, img, px):
    """img (H, W, C) -> (N, C) bilinear at pixel coords px (N, 2); out-of-image -> 0."""
    H, W = img.shape[:2]
    g = torch.stack([(px[:, 0] + 0.5) / W * 2 - 1, (px[:, 1] + 0.5) / H * 2 - 1], 1)[None, None]
    return torch.nn.functional.grid_sample(img.permute(2, 0, 1)[None], g, mode="bilinear", padding_mode="zeros",
                                           align_corners=False)[0, :, 0].T


def _texel_atlas(sk: dict, size: int, key: str = ""):
    """UVAtlas of the skin's own UVs (faces = the unsplit vertex ids, so hole diffusion crosses seams). key "" = the
    shipped skin (SMPL-X ids); a shell skin passes its own key (midpoint ids >= 10475 take the nearest original vertex's
    head / face / eye flags)."""
    from scipy.spatial import cKDTree

    from ..avatar_uv import UVAtlas, rasterize_uv, vertex_regions
    from ..export_unity import MAMMA_MODEL
    cache = hybrid_dir() / f"_atlas_{size}{('_' + key) if key else ''}.npz"
    F = sk["faces"]
    if cache.exists() and cache.stat().st_mtime >= Path(sk["path"]).stat().st_mtime:
        z = np.load(cache)
        yx, face, bary, mirror, region = z["yx"], z["face"], z["bary"], z["mirror"], z["region"]
    else:
        yx, face, bary = rasterize_uv(sk["uv"], F, size)
        P0 = np.einsum("nkd,nk->nd", sk["V"][F[face]], bary)
        d, j = cKDTree(P0).query(P0 * np.array([-1.0, 1.0, 1.0]), workers=-1)
        mirror = np.where(d < 0.01, j, -1).astype(np.int32)
        m = np.load(MAMMA_MODEL, allow_pickle=True)
        reg = vertex_regions(m)
        n_s = len(m["v_template"])
        vid = sk["vid"]
        orig = vid < n_s
        near = np.arange(len(vid))
        if not orig.all():
            _, j = cKDTree(sk["V"][orig]).query(sk["V"][~orig], workers=-1)
            near[~orig] = np.flatnonzero(orig)[j]
        reg = {k: v[vid[near]] for k, v in reg.items()}             # per split vertex
        tri = F[face]
        agg = lambda flag: (flag[tri].astype(np.float32) * bary).sum(1) > 0.5  # noqa: E731
        region = (agg(reg["head"]) * 1 + agg(reg["face"]) * 2 + agg(reg["eye"]) * 4).astype(np.uint8)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, yx=yx, face=face, bary=bary, mirror=mirror, region=region)
    atlas = UVAtlas(size, yx, face, bary, mirror, region, sk["vid"][F], F, sk["uv"])
    return atlas


def _smooth_on_mesh(vals: np.ndarray, known_w: np.ndarray, F: np.ndarray, n: int, iters: int) -> np.ndarray:
    """Weighted Laplacian smoothing of per-vertex values over an (unsplit) mesh; unknown vertices filled by diffusion."""
    from ..avatar_texture import _vertex_adjacency
    A = _vertex_adjacency(F, n)
    w = known_w.astype(float)
    v = np.where(w > 0, vals, 0.0)
    for _ in range(iters):
        num = A @ (v * w) + v * w * 2
        den = A @ w + w * 2
        v = np.where(den > 1e-9, num / np.maximum(den, 1e-12), v)
        w = np.minimum(1.0, A @ w + w)
    return v


def load_shell_skin(role: str, take: str = TAKE) -> dict:
    """hybrid/smplx_<role>_skin.bin in the world frame (like load_skin)."""
    from ..export_unity import read_skin
    p = hybrid_dir(take) / f"smplx_{role}_skin.bin"
    h, sk = read_skin(p)
    return {"path": p, "header": h, "V": np.asarray(sk.vertices, float) * MIRROR,
            "faces": np.asarray(sk.faces, np.int64)[:, [0, 2, 1]], "W": np.asarray(sk.weights, float),
            "J": np.asarray(sk.joints, float) * MIRROR, "parents": np.asarray(sk.parents, np.int64),
            "uv": np.asarray(sk.uv, float), "vid": np.asarray(sk.vertex_ids, np.int64), "skin": sk}


def bake(role: str, take: str = TAKE, size: int = SIZE, device="cuda", shell: bool = False, log=log) -> dict:
    """shell=False: the shipped skin, texel points pushed out to the splat surface (pass 1) -> hybrid/<role>_albedo.png.
    shell=True: the shell skin (shell_skin) A-posed with the same pose; its surface already stands where the clothes /
    hair are; pass 1 only finds the residual offset to the splat surface -> hybrid/shell/<role>_albedo.png (same UV
    layout)."""
    import torch

    from ..avatar_texture import _lin_t, fill_holes, gutter, linear_to_srgb
    t0 = time.time()
    out = hybrid_dir(take) / "shell" if shell else hybrid_dir(take)
    A = load_apose(role, take)
    if shell:
        sk = load_shell_skin(role, take)
        V = lbs_np(sk["V"], sk["W"], sk["J"], sk["parents"], A["rv"], A["transl"])[0]
        atlas = _texel_atlas(sk, size, key=f"shell_{role}")
    else:
        sk = load_skin(role, take)
        V = A["V"]
        atlas = _texel_atlas(sk, size)
    src = gs_source(role, take)
    G = load_gs(role, take)
    gr = GSRenderer(G, device)
    F, vid = sk["faces"], sk["vid"]
    Nrm = unsplit_normals(V, F, vid)
    tri = F[atlas.face]
    X = np.einsum("nkd,nk->nd", V[tri], atlas.bary)
    Nt = np.einsum("nkd,nk->nd", Nrm[tri], atlas.bary)
    Nt /= np.maximum(np.linalg.norm(Nt, axis=1, keepdims=True), 1e-12)
    head = (atlas.region & 1) > 0
    face_t = (atlas.region & 6) > 0                       # front of the head + eyeballs
    Nw = Nt.copy()                                       # normal used for the view weights
    Nw[face_t] = Nt[face_t] + FACE_FORWARD_TILT * np.array([0.0, 0.0, 1.0])
    Nw /= np.maximum(np.linalg.norm(Nw, axis=1, keepdims=True), 1e-12)
    cams = bake_cameras(V, A["P"], sk)
    log(f"{role}: {atlas.n} texels, {len(cams)} bake cameras, splats {_rel(src)}")
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    Xt, Nt_t = T(X), T(Nt)
    renders = []
    for c in cams:
        r = gr.render(c)
        renders.append({k: v for k, v in r.items()})
    t_r = time.time() - t0
    # ---- pass 1: offset of the splat surface along the normal (near-frontal views, mesh-visible texels only)
    num_d = torch.zeros(atlas.n, device=device)
    den_d = torch.zeros(atlas.n, device=device)
    for c, r in zip(cams, renders):
        zb, _ = mesh_mask(V, F, c)
        zb_t = T(np.where(np.isfinite(zb), zb, 1e9))[..., None]
        Rt, tt, Kt = T(c["R"]), T(c["tvec"]), T(c["K"])
        Xc = Xt @ Rt.T + tt
        z = Xc[:, 2]
        px = torch.stack([Kt[0, 0] * Xc[:, 0] / z + Kt[0, 2], Kt[1, 1] * Xc[:, 1] / z + Kt[1, 2]], 1)
        vdir = torch.nn.functional.normalize(T(c["C"])[None] - Xt, dim=1)
        cos = (vdir * Nt_t).sum(1)
        zmesh = _bilinear(torch, zb_t, px)[:, 0]
        a = _bilinear(torch, r["alpha"][..., None], px)[:, 0]
        ed = _bilinear(torch, r["depth"][..., None], px)[:, 0] / a.clamp(min=1e-6)
        ok = (cos > 0.6) & (z <= zmesh + 0.01) & (a > 0.95)
        delta = (z - ed) * cos
        w = torch.where(ok & (delta > -0.05) & (delta < 0.20), cos ** 8, torch.zeros_like(cos))
        num_d += w * delta
        den_d += w
    d_tex = (num_d / den_d.clamp(min=1e-9)).cpu().numpy()
    w_tex = den_d.cpu().numpy()
    # texel offsets -> per (unsplit) vertex, smoothed, back to texels
    vu = vid[tri]
    n_u = int(vid.max()) + 1
    acc, wacc = np.zeros(n_u), np.zeros(n_u)
    for k in range(3):
        np.add.at(acc, vu[:, k], atlas.bary[:, k] * w_tex * d_tex)
        np.add.at(wacc, vu[:, k], atlas.bary[:, k] * w_tex)
    d_v = np.where(wacc > 1e-9, acc / np.maximum(wacc, 1e-12), 0.0)
    known = (wacc > 1e-6 * max(wacc.max(), 1e-12)).astype(float)
    d_v = _smooth_on_mesh(d_v, known, vid[F], n_u, iters=6)
    d_v = np.clip(d_v, -0.03, 0.15)
    d_tex_s = np.einsum("nk,nk->n", d_v[vu], atlas.bary)
    Xs = T(X + Nt * d_tex_s[:, None])
    Nw_t = T(Nw)
    # ---- pass 2: colour of the pushed-out texel points
    p_exp = torch.where(T(head) > 0, torch.full((atlas.n,), 12.0, device=device), torch.full((atlas.n,), 4.0, device=device))
    acc_c = torch.zeros(atlas.n, 3, device=device)
    acc_w = torch.zeros(atlas.n, device=device)
    best_w = torch.zeros(atlas.n, device=device)
    n_seen = torch.zeros(atlas.n, device=device)
    for c, r in zip(cams, renders):
        Rt, tt, Kt = T(c["R"]), T(c["tvec"]), T(c["K"])
        Xc = Xs @ Rt.T + tt
        z = Xc[:, 2]
        px = torch.stack([Kt[0, 0] * Xc[:, 0] / z + Kt[0, 2], Kt[1, 1] * Xc[:, 1] / z + Kt[1, 2]], 1)
        vdir = torch.nn.functional.normalize(T(c["C"])[None] - Xs, dim=1)
        cos = (vdir * Nw_t).sum(1)
        a = _bilinear(torch, r["alpha"][..., None], px)[:, 0]
        ed = _bilinear(torch, r["depth"][..., None], px)[:, 0] / a.clamp(min=1e-6)
        col = _bilinear(torch, r["rgb_prem"], px) / a.clamp(min=1e-6)[:, None]
        foot = z / Kt[0, 0]                                  # metres per pixel at the texel
        tol = torch.clamp(3 * foot / cos.clamp(min=0.2), min=0.012)
        vis = torch.exp(-((z - ed) / tol) ** 2)
        res = (0.0016 / foot).clamp(max=4.0) ** 2              # 1.6 mm/px (the orbit renders) = 1
        w = torch.where((cos > 0.15) & (a > 0.5), vis * cos.clamp(min=0) ** p_exp * res * a * a, torch.zeros_like(cos))
        acc_c += w[:, None] * _lin_t(torch, col.clamp(0, 1))
        acc_w += w
        best_w = torch.maximum(best_w, w)
        n_seen += (w > 0.05 * w.max().clamp(min=1e-12)).float()
    rgb_lin = (acc_c / acc_w.clamp(min=1e-12)[:, None]).cpu().numpy()
    conf = acc_w.cpu().numpy()
    filled, alpha_meas, srcmap = fill_holes(atlas, rgb_lin, conf, n_u)
    srgb = linear_to_srgb(np.clip(filled, 0, 1))
    img = atlas.to_image(srgb.astype(np.float32))
    cov = atlas.to_image(np.ones(atlas.n, np.float32)) > 0
    img = gutter(img, cov, 16)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{role}_albedo.png"), (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))
    covimg = atlas.to_image((alpha_meas * 255).astype(np.uint8))
    cv2.imwrite(str(out / f"{role}_coverage.png"), covimg)
    offimg = atlas.to_image(np.clip(d_tex_s / 0.15 * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(str(out / f"{role}_offset.png"), offimg)
    if not shell:
        np.savez(out / f"{role}_offsets.npz", d_v=d_v.astype(np.float32), vertex_weight=wacc.astype(np.float32))
    # thumbnails of a few bake renders (what the texture was made from)
    th = []
    for i in [0, 3, 6, 9, 13, 19]:
        a = renders[i]["alpha"].cpu().numpy()
        rgb = renders[i]["rgb_prem"].cpu().numpy() + (1 - a[..., None])
        th.append(cv2.resize((np.clip(rgb, 0, 1)[..., ::-1] * 255).astype(np.uint8), (320, 320), interpolation=cv2.INTER_AREA))
    (hybrid_dir(take) / "check").mkdir(exist_ok=True)
    cv2.imwrite(str(hybrid_dir(take) / "check" / f"bake_views_{role}.jpg"), np.hstack(th), [cv2.IMWRITE_JPEG_QUALITY, 85])
    meas = alpha_meas > 0.5
    rep = {"role": role, "mode": "shell" if shell else "skin", "albedo": _rel(out / f"{role}_albedo.png"),
           "splats": _rel(src), "face_registered": bool(G.get("registered")), "skin": _rel(sk["path"]),
           "atlas": size, "texels": int(atlas.n),
           "cameras": {p: sum(1 for c in cams if c["part"] == p) for p in sorted({c["part"] for c in cams})},
           "measured_frac": round(float(meas.mean()), 4),
           "filled": {"mirror": int((srcmap == 1).sum()), "diffused": int((srcmap == 2).sum())},
           "views_per_texel_median": float(np.median(n_seen.cpu().numpy()[meas])) if meas.any() else 0,
           "offset_m": {"median": round(float(np.median(d_v)), 4), "p95": round(float(np.percentile(d_v, 95)), 4),
                        "max": round(float(d_v.max()), 4), "min": round(float(d_v.min()), 4)},
           "head_measured_frac": round(float(meas[head].mean()), 4),
           "render_s": round(t_r, 1), "seconds": round(time.time() - t0, 1)}
    (hybrid_dir(take) / f"bake_{'shell_' if shell else ''}{role}.json").write_text(json.dumps(rep, indent=1))
    log(f"{role} bake: {json.dumps(rep)}")
    return rep


def fidelity_check(role: str, take: str = TAKE, device="cuda", albedo: Path | None = None) -> dict:
    """What the bake keeps: the canonical splats (gsplat) vs the A-posed skin with the baked albedo (avatar_texture.render)
    from cameras BETWEEN the bake cameras (body front / back / side, head front / three-quarter / profile) ->
    hybrid/check/fidelity_<role>.jpg (splats | baked mesh | |difference| x 3) and PSNR where both cover."""
    import torch

    from ..avatar_texture import render as tex_render
    from .appearance_avatar_ags import _TexGeo
    sk, A = load_skin(role, take), load_apose(role, take)
    gr = GSRenderer(load_gs(role, take), device)
    alb_p = albedo or hybrid_dir(take) / f"{role}_albedo.png"
    img = cv2.imread(str(alb_p), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
    alb = torch.as_tensor(img.transpose(2, 0, 1).copy(), device=device)
    geo = _TexGeo(torch, sk["faces"], sk["uv"], device)
    V = A["V"]
    ctr = 0.5 * (V.max(0) + V.min(0))
    head = A["P"][15] + np.array([0, 0.06, 0])
    views = [("body front", ctr, 7, 3.4, 800), ("body back", ctr, 187, 3.4, 800), ("body side", ctr, 97, 3.4, 800),
             ("head front", head, 7, 0.75, 700), ("head 3/4", head, 40, 0.75, 700), ("head profile", head, 82, 0.75, 700)]
    rows, scores = [], {}
    for name, tgt, az, dist, S in views:
        f = S / 2 / math.tan(math.radians(18 if name.startswith("body") else 12))
        C = tgt + dist * np.array([math.sin(math.radians(az)), 0.12 if name.startswith("body") else 0.03,
                                   math.cos(math.radians(az))])
        c = cam(C, tgt, S, S, f)
        r = gr.render(c)
        a = r["alpha"].cpu().numpy()
        gs_rgb = r["rgb_prem"].cpu().numpy() / np.maximum(a, 1e-6)[..., None]
        with torch.no_grad():
            m_rgb, idm = tex_render(torch, geo, {role: V}, {role: alb}, c["K"], c["R"], c["tvec"], 0.0, S, S, device)
        cov = idm >= 0
        both = cov & (a > 0.95)
        mse = float(((m_rgb - gs_rgb)[both] ** 2).mean()) if both.any() else float("nan")
        scores[name] = {"psnr_db": round(10 * math.log10(1 / max(mse, 1e-10)), 2), "mesh_px": int(cov.sum()),
                        "splat_px": int((a > 0.5).sum()), "both_px": int(both.sum())}
        bg = np.full((S, S, 3), 0.43, np.float32)
        t1 = np.where(a[..., None] > 0, gs_rgb * a[..., None] + bg * (1 - a[..., None]), bg)
        t2 = np.where(cov[..., None], m_rgb, bg)
        t3 = np.where(both[..., None], np.clip(np.abs(m_rgb - gs_rgb) * 3, 0, 1), 0.0)
        row = [(np.clip(t, 0, 1)[..., ::-1] * 255).astype(np.uint8).copy() for t in (t1, t2, t3)]
        _lab(row[0], f"{role} {name}: LHM++ splats")
        _lab(row[1], "baked mesh (A-pose)")
        _lab(row[2], f"|diff| x3, PSNR {scores[name]['psnr_db']} dB")
        rows.append(cv2.resize(np.hstack(row), (3 * 420, 420), interpolation=cv2.INTER_AREA))
    out = hybrid_dir(take) / "check" / f"fidelity_{role}{'' if albedo is None else '_' + Path(albedo).stem}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (hybrid_dir(take) / f"fidelity_{role}.json").write_text(json.dumps(scores, indent=1))
    log(f"{role} fidelity {json.dumps(scores)} -> {out}")
    return scores


# ------------------------------------------------------------------------------------------------ step 3: shell skin
def subdivide(sk: dict) -> dict:
    """One midpoint subdivision of the seam-split skin (each triangle -> 4); vertex attributes averaged per edge. A seam
    edge exists in two split copies: both get the same position (no cracks)."""
    F = sk["faces"]
    nV = len(sk["V"])
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    es = np.sort(e, 1)
    uniq, inv = np.unique(es, axis=0, return_inverse=True)
    inv = inv.reshape(3, -1).T + nV              # (F, 3) midpoint ids of edges 01, 12, 20
    a, b = uniq[:, 0], uniq[:, 1]
    mid = lambda X: np.concatenate([X, 0.5 * (X[a] + X[b])])  # noqa: E731
    out = dict(sk)
    out["V"], out["W"], out["uv"] = mid(sk["V"]), mid(sk["W"]), mid(sk["uv"])
    # unsplit identity of a midpoint = its (unsplit) edge, so seam copies share it
    vu = sk["vid"]
    ue = np.sort(np.stack([vu[a], vu[b]], 1), 1)
    uu, uinv = np.unique(ue, axis=0, return_inverse=True)
    n_u = int(vu.max()) + 1
    out["vid"] = np.concatenate([vu, n_u + uinv.ravel()])
    out["edge_ends_unsplit"] = np.concatenate([np.stack([np.arange(n_u), np.arange(n_u)], 1), uu])
    m01, m12, m20 = inv[:, 0], inv[:, 1], inv[:, 2]
    v0, v1, v2 = F[:, 0], F[:, 1], F[:, 2]
    out["faces"] = np.concatenate([np.stack([v0, m01, m20], 1), np.stack([m01, v1, m12], 1),
                                   np.stack([m20, m12, v2], 1), np.stack([m01, m12, m20], 1)])
    return out


def shell_skin(role: str, take: str = TAKE, max_body: float = 0.035, max_hair: float = 0.12, max_face: float = 0.004,
               log=log) -> dict:
    """Subdivided skin pushed out along its rest normals by the bake's offset field -> hybrid/smplx_<role>_skin.bin."""
    from ..export_unity import Skin, unity_skin, write_skin
    sk = load_skin(role, take)
    z = np.load(hybrid_dir(take) / f"{role}_offsets.npz")
    d_v = np.clip(z["d_v"].astype(float), 0.0, None)
    # hair may stand off the head / upper back further than clothes stand off the body
    W = sk["W"]
    n_u = int(sk["vid"].max()) + 1
    w_u = np.zeros((n_u, W.shape[1]))
    w_u[sk["vid"]] = W
    upper = w_u[:, [9, 12, 13, 14, 15, 22, 23, 24]].sum(1)             # spine3, neck, collars, head, jaw, eyes
    cap = max_body + (max_hair - max_body) * np.clip((upper - 0.3) / 0.4, 0, 1) if role == "follow" else \
        np.full(n_u, max_body)
    reg = vertex_regions_split(sk)
    face_u = np.zeros(n_u)
    face_u[sk["vid"]] = (reg["face"] | reg["eye"]).astype(float)
    F_u = sk["vid"][sk["faces"]]
    face_u = np.clip(_smooth_on_mesh(face_u, np.ones(n_u), F_u, n_u, iters=4) * 1.5, 0, 1)
    cap = cap * (1 - face_u) + max_face * face_u            # the face (and a ramp around it) stays on the skin
    d_v = np.minimum(d_v, cap)
    d_v = _smooth_on_mesh(d_v, np.ones(n_u), F_u, n_u, iters=SHELL_SMOOTH_ITERS)
    sub = subdivide(sk)
    ends = sub["edge_ends_unsplit"]
    d_sub_u = 0.5 * (d_v[ends[:, 0]] + d_v[ends[:, 1]])
    d_sub = d_sub_u[sub["vid"]]
    # push along a SMOOTHED normal field: raw normals of folded parts (ears, nostrils, fingers) fan out and inflate them
    from ..avatar_texture import _vertex_adjacency
    n_all = int(sub["vid"].max()) + 1
    Nu = np.zeros((n_all, 3))
    Nu[sub["vid"]] = unsplit_normals(sub["V"], sub["faces"], sub["vid"])
    Adj = _vertex_adjacency(sub["vid"][sub["faces"]], n_all)
    for _ in range(SHELL_NORMAL_SMOOTH):
        Nu = Adj @ Nu + Nu
        Nu /= np.maximum(np.linalg.norm(Nu, axis=1, keepdims=True), 1e-12)
    # stretch limiter: inflating a strongly curved part (the ears above all) blows its few texels up into streaks;
    # where a vertex's faces would grow by more than SHELL_MAX_STRETCH in area its push is reduced
    Vu = np.zeros((n_all, 3))
    Vu[sub["vid"]] = sub["V"]
    Fu = sub["vid"][sub["faces"]]

    def areas(X):
        t = X[Fu]
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    a0 = np.maximum(areas(Vu), 1e-12)
    limited = np.zeros(n_all, bool)
    for _ in range(12):
        ratio = areas(Vu + Nu * d_sub_u[:, None]) / a0
        vmax = np.zeros(n_all)
        for k in range(3):
            np.maximum.at(vmax, Fu[:, k], ratio)
        bad = vmax > SHELL_MAX_STRETCH
        if not bad.any():
            break
        limited |= bad
        d_sub_u = np.where(bad, 0.7 * d_sub_u, d_sub_u)
        d_sub_u = np.minimum(d_sub_u, _smooth_on_mesh(d_sub_u, np.ones(n_all), Fu, n_all, iters=2))
    d_sub = d_sub_u[sub["vid"]]
    Nrm = Nu[sub["vid"]]
    sub["V"] = sub["V"] + Nrm * d_sub[:, None]
    Nrm2 = unsplit_normals(sub["V"], sub["faces"], sub["vid"])
    base: Skin = sk["skin"]
    skin_w = Skin(vertices=sub["V"], faces=sub["faces"], weights=sub["W"], joints=sk["J"], parents=sk["parents"],
                  model=base.model, betas=base.betas, uv=sub["uv"], normals=Nrm2, vertex_ids=sub["vid"],
                  edits={**(base.edits or {}), "hybrid_shell": {
                      "source": "LHM++ canonical splats, appearance_hybrid.shell_skin", "subdivided": 1,
                      "offset_m_max": round(float(d_sub.max()), 4), "offset_m_median": round(float(np.median(d_sub)), 4),
                      "stretch_limited_vertices": int(limited.sum())}})
    p = hybrid_dir(take) / f"smplx_{role}_skin.bin"
    h = write_skin(p, role, unity_skin(skin_w))
    log(f"{role} shell skin: {int(limited.sum())} vertices stretch-limited; {len(sub['V'])} vertices, "
        f"{len(sub['faces'])} faces, offset median "
        f"{np.median(d_sub) * 1e3:.1f} mm, max {d_sub.max() * 1e3:.1f} mm -> {p}")
    return {"file": _rel(p), "vertices": h["vertices"], "faces": h["faces"],
            "offset_mm": {"median": round(float(np.median(d_sub)) * 1e3, 1),
                          "p95": round(float(np.percentile(d_sub, 95)) * 1e3, 1),
                          "max": round(float(d_sub.max()) * 1e3, 1)}}


# ------------------------------------------------------------------------------------------------ step 4: render
def make_renderer(method: str, take: str = TAKE, device="cuda", ss: int = 2):
    import torch

    from .appearance_avatar_ags import AvatarRenderer, _TexGeo, _md5
    from ..export_unity import read_skin
    R = AvatarRenderer(take, device, ss)
    hd = hybrid_dir(take)
    R.hybrid_inputs = []
    for r in ROLES:
        p = (hd / "shell" if method == "hybrid_mesh_shell" else hd) / f"{r}_albedo.png"
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
        R.alb[r] = torch.as_tensor(img.transpose(2, 0, 1).copy(), device=device)
        R.hybrid_inputs.append({"file": _rel(p), "md5": _md5(p), "bytes": p.stat().st_size})
    if method == "hybrid_mesh_shell":
        faces = uv = None
        for r in ROLES:
            p = hd / f"smplx_{r}_skin.bin"
            _, sk = read_skin(p)
            R.skin[r] = sk
            f_world = np.asarray(sk.faces)[:, [0, 2, 1]]
            if faces is None:
                faces, uv = f_world, np.asarray(sk.uv)
            R.hybrid_inputs.append({"file": _rel(p), "md5": _md5(p), "bytes": p.stat().st_size})
        R.geo = _TexGeo(torch, faces, uv, device)
        R._cache.clear()
    R.method = method
    return R


def lhmpp_inputs(take: str, spec: dict) -> dict:
    """Which photos the appearance came from: the LHM++ reference crops of the splats used (compare/lhmpp/input/refs.json
    for lhmpp_1phone, else the w8 set), and per real view of the spec the nearest input frame of the same phone."""
    src = gs_source(ROLES[0], take)
    var = "lhmpp_1phone" if "lhmpp_1phone" in src.name else None
    refs_p = compare_dir(take) / "lhmpp" / "input" / "refs.json"
    out = {"splats": {r: _rel(gs_source(r, take)) for r in ROLES}, "inputs": {}, "real_views": {}}
    if var and refs_p.exists():
        R = json.loads(refs_p.read_text())[var]
        out["inputs"] = {r: [[f["cam"], int(f["i"])] for f in R[r]["frames"]] for r in ROLES}
    else:
        out["note"] = "w8_ref: 8 crops of phone 06, 58-62 s (mono4d/lhmpp/input/refs.json)"
        return out
    for v in spec["real_views"]:
        rec = {}
        for r in ROLES:
            same = [i for c, i in out["inputs"][r] if c == v["cam"]]
            if same:
                d = min(abs(i - int(v["frame"])) for i in same)
                rec[r] = "INPUT CROP (this exact frame)" if d == 0 else f"phone is an input phone, nearest input frame {d} frames away"
            else:
                rec[r] = "phone not an input"
        out["real_views"][v["id"]] = rec
    out["note"] = ("the hybrids' appearance saw only these phone-06 crops; phones 02 / 03 never entered (the current "
                   "avatar's footage bake used all 8 phones, so it is in-sample on every real view)")
    return out


def render_method(method: str, take: str = TAKE, groups=None, ss: int = 2, device="cuda", ids=None, log=log) -> dict:
    from .. import gpu
    from .appearance_avatar_ags import GROUPS, SCHEMA, CU_SIZE, build_jobs, load_spec, placement, write_rgba, write_roles
    groups = groups or GROUPS
    spec = load_spec(take)
    jobs = build_jobs(spec, groups)
    if ids:
        jobs = [j for j in jobs if j["id"] in ids]
    out = compare_dir(take) / "renders" / method
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {}
    work = PACKAGE_ROOT / "work" / take
    t0 = time.time()
    with gpu.exclusive(f"compare-{method}", timeout_s=3600):
        rend = make_renderer(method, take, device, ss)
        views = man.get("views", {})
        for n, j in enumerate(jobs):
            ts = time.time()
            rgb, alpha, roles, info = rend.render(j)
            write_rgba(out / j["file"], rgb, alpha)
            rec = {"group": j["group"], "file": j["file"], "W": j["cam"]["W"], "H": j["cam"]["H"], "t": round(j["t"], 5),
                   **info, "meta": j["meta"], "coverage_px": int((alpha > 0.5).sum()), "seconds": round(time.time() - ts, 2)}
            if not j["group"].startswith("clip"):
                write_roles(out / "roles" / j["file"], roles)
                rec["roles_file"] = f"roles/{j['file']}"
                rec["camera"] = {"K": np.round(j["cam"]["K"], 6).tolist(), "R": np.round(j["cam"]["R"], 9).tolist(),
                                 "tvec": np.round(j["cam"]["tvec"], 6).tolist()}
            if j["group"] == "real":
                rec["placement_vs_sam_masks"] = placement(alpha, roles, work, j["meta"]["phone"], j["meta"]["frame"])
            views[j["id"]] = rec
            if n % 25 == 0 or n == len(jobs) - 1:
                log(f"[{method}] {n + 1}/{len(jobs)} {j['id']} {rec['seconds']:.2f} s")
        base = rend.describe()
    shell = method == "hybrid_mesh_shell"
    bk = "bake_shell_" if shell else "bake_"
    bakes = {r: json.loads((hybrid_dir(take) / f"{bk}{r}.json").read_text()) for r in ROLES
             if (hybrid_dir(take) / f"{bk}{r}.json").exists()}
    freg = {r: json.loads((hybrid_dir(take) / f"face_reg_{r}.json").read_text()) for r in ROLES
            if (hybrid_dir(take) / f"face_reg_{r}.json").exists()}
    fits = {r: json.loads((hybrid_dir(take) / f"apose_{r}.json").read_text()) for r in ROLES
            if (hybrid_dir(take) / f"apose_{r}.json").exists()}
    man.update({
        "schema": SCHEMA, "method": method, "take": take, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": "dancecap/dancecap/mono4d/appearance_hybrid.py (renderer: appearance_avatar_ags.AvatarRenderer)",
        "spec": _rel(compare_dir(take) / "views.json"), "spec_created": spec["created"],
        "what": ("HYBRID: the shipped SMPL-X skin (export/" + CAPTURE + "/smplx_<role>_skin.bin" +
                 (", subdivided once and pushed out along smoothed rest normals by LHM++'s clothes / hair offset field "
                  "(face kept on the skin, stretch-limited): hybrid/smplx_<role>_skin.bin" if shell else "") +
                 ") with an albedo BAKED FROM LHM++'s canonical Gaussians (lhmpp_1phone, face registered onto the "
                 "skin's eye openings / nose / mouth; gsplat renders from 48 orbit + head / hand / foot close-up cameras "
                 "onto the SMPL-X UV atlas: hybrid/" + ("shell/" if shell else "") + "<role>_albedo.png), skinned and "
                 "rendered exactly like the current avatar (plain LBS, reference fit, unlit, avatar_texture.render)"),
        "caption": "offline render of a skinned mesh with an LHM++-baked texture (Unity-loadable; no hair cards drawn)",
        "avatar_caption_for_reference": AVATAR_CAPTION,
        "pose": base["pose"], "antialiasing": base["antialiasing"], "fk_vs_fit_joints_mm": base["fk_vs_fit_joints_mm"],
        "appearance_source": {r: fits.get(r, {}).get("splats") for r in ROLES},
        "seen_by": lhmpp_inputs(take, spec),
        "apose_fit": {r: {k: fits[r][k] for k in ("pose_deg", "transl_m", "skin_vertex_to_splats_mm", "splats_to_skin_mm")}
                      for r in fits},
        "bake": bakes, "face_registration": {r: {k: freg[r][k] for k in ("scale", "translation_m", "front", "side")}
                                              for r in freg},
        "inputs": base["inputs"] + rend.hybrid_inputs,
        "conventions": {"image": "RGBA PNG, straight colour, transparent background; roles/<id>.png 0 none, 1 lead, 2 "
                                 "follow", "camera": "pinhole (k1 ignored)",
                        "clip_index": "clip_<camera>/<n:04d>.png, n = 0..104 = views.json clip.t[n]",
                        "closeups": f"cu_<instant>_c<phone>_<box>.png, longer side {CU_SIZE} px"}})
    man["views"] = dict(sorted(views.items()))
    man["counts"] = {g: sum(1 for v in man["views"].values() if v["group"] == g) for g in GROUPS}
    pl = [v["placement_vs_sam_masks"] for v in man["views"].values() if v.get("placement_vs_sam_masks")]
    if pl:
        man["placement_summary"] = {k: round(float(np.median([p[k] for p in pl if p[k] is not None])), 3)
                                    for k in ("iou_union", "iou_lead", "iou_follow")}
    mp.write_text(json.dumps(man, indent=1))
    log(f"[{method}] {len(jobs)} renders in {time.time() - t0:.0f} s -> {out}")
    return man


# ------------------------------------------------------------------------------------------------ review sheets
def _over(bg, rgba, k=1.0):
    if rgba is None:
        return bg.copy()
    a = rgba[..., 3:4].astype(np.float32) / 255 * k
    return (bg.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a).astype(np.uint8)


def _lab(img, text, org=(6, 22), s=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def check(take: str = TAKE, methods=("avatar", "hybrid_mesh", "hybrid_mesh_shell")) -> list[Path]:
    """hybrid/check/: per real view photo crop | each method (on a grey background), turntable rows, close-ups, the
    albedo atlases side by side."""
    from .appearance_avatar_ags import load_spec
    spec = load_spec(take)
    root = compare_dir(take) / "renders"
    out = hybrid_dir(take) / "check"
    out.mkdir(parents=True, exist_ok=True)
    methods = [m for m in methods if (root / m).exists()]
    written = []
    rd = lambda p: cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # noqa: E731
    by_inst: dict = {}
    for v in spec["real_views"]:
        photo = cv2.imread(str(PACKAGE_ROOT / v["file"]))
        x0, y0, x1, y1 = v["crop"] or [0, 0, v["W"], v["H"]]
        tiles = [_lab(photo[y0:y1, x0:x1].copy(), f"{v['id']} photo")]
        for m in methods:
            bg = np.full_like(photo, 110)
            tiles.append(_lab(_over(bg, rd(root / m / f"{v['id']}.png"))[y0:y1, x0:x1].copy(), m))
        tiles = [cv2.resize(t, (int(round(t.shape[1] * 560 / t.shape[0])), 560), interpolation=cv2.INTER_AREA) for t in tiles]
        by_inst.setdefault(v["instant"], []).append(np.hstack(tiles))
    for k, rows in by_inst.items():
        W = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.zeros((r.shape[0], W - r.shape[1], 3), np.uint8)]) for r in rows]
        p = out / f"real_{k}.jpg"
        cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 86])
        written.append(p)
    rows = []
    for m in methods:
        row = []
        for v in spec["novel"]["turntable"]["views"][::2]:
            im = rd(root / m / f"{v['id']}.png")
            bg = np.full((1080, 1080, 3), 110, np.uint8)
            row.append(_lab(cv2.resize(_over(bg, im), (360, 360), interpolation=cv2.INTER_AREA), f"{m} {v['id']}"))
        rows.append(np.hstack(row))
    if rows:
        p = out / "turntable.jpg"
        cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 86])
        written.append(p)
    rows = []
    for c in spec["closeups"][:3]:
        for phone, cc in c["cams"].items():
            v = next(x for x in spec["real_views"] if x["id"] == cc["view_id"])
            photo = cv2.imread(str(PACKAGE_ROOT / v["file"]))
            for name, b in cc["boxes"].items():
                if not b.get("box"):
                    continue
                x0, y0, x1, y1 = b["box"]
                crop = photo[max(y0, 0):y1, max(x0, 0):x1]
                row = [_lab(cv2.resize(crop, (300, 300), interpolation=cv2.INTER_CUBIC), f"{c['instant']} c{phone} {name}", s=0.45)]
                for m in methods:
                    im = rd(root / m / f"cu_{c['instant']}_c{phone}_{name}.png")
                    bg = np.full((300, 300, 3), 110, np.uint8)
                    row.append(_lab(_over(bg, None if im is None else cv2.resize(im, (300, 300), interpolation=cv2.INTER_AREA)), m, s=0.45))
                rows.append(np.hstack(row))
    if rows:
        p = out / "closeups.jpg"
        cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 86])
        written.append(p)
    # clips: every 15th frame; phone 03's own track over the photo, and the VR seat on grey
    c3 = spec["clip"]["cam03"]["frames"]
    cms = methods + [m for m in ("lhmpp_1phone",) if (root / m).exists()]
    r03 = {m: [] for m in ["photo"] + cms}
    rvr = {m: [] for m in cms}
    for n in range(0, len(c3), 15):
        ph = cv2.imread(str(PACKAGE_ROOT / c3[n]["file"]))
        r03["photo"].append(_lab(cv2.resize(ph, (240, 424), interpolation=cv2.INTER_AREA), f"03 n{n}", s=0.45))
        for m in cms:
            im = rd(root / m / "clip_cam03" / f"{n:04d}.png")
            r03[m].append(_lab(cv2.resize(_over(ph, im, 0.9), (240, 424), interpolation=cv2.INTER_AREA), m, s=0.4))
            iv = rd(root / m / "clip_vr_seat" / f"{n:04d}.png")
            bg = np.full((1080, 1080, 3), 110, np.uint8)
            rvr[m].append(_lab(cv2.resize(_over(bg, iv), (300, 300), interpolation=cv2.INTER_AREA), f"VR n{n} {m}", s=0.4))
    top = np.vstack([np.hstack(v) for v in r03.values()])
    bot = np.vstack([np.hstack(v) for v in rvr.values()])
    Wd = max(top.shape[1], bot.shape[1])
    pad = lambda x: np.hstack([x, np.zeros((x.shape[0], Wd - x.shape[1], 3), np.uint8)])  # noqa: E731
    p = out / "clips.jpg"
    cv2.imwrite(str(p), np.vstack([pad(top), pad(bot)]), [cv2.IMWRITE_JPEG_QUALITY, 84])
    written.append(p)
    tiles = []
    for r in ROLES:
        for p in (PACKAGE_ROOT / "work" / take / "export" / CAPTURE / f"{r}_albedo.png", hybrid_dir(take) / f"{r}_albedo.png"):
            im = cv2.imread(str(p))
            if im is not None:
                tiles.append(_lab(cv2.resize(im, (700, 700), interpolation=cv2.INTER_AREA),
                                  f"{r} {'current (footage bake, face variant)' if 'export' in str(p) else 'hybrid (LHM++ bake)'}"))
    if tiles:
        p = out / "albedo_atlases.jpg"
        cv2.imwrite(str(p), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 86])
        written.append(p)
    return written


def score(take: str = TAKE, methods=("avatar", "hybrid_mesh", "hybrid_mesh_shell", "lhmpp_1phone", "ags_8phone"),
          log=log) -> dict:
    """Photo agreement on the 18 real views (local CPU, no LPIPS here): per view and dancer, a crop around the dancer's
    SAM mask (+10 %), the render composited on black (uncovered = black, so missing hair / clothes volume costs), PSNR /
    SSIM inside the mask, raw and after a least-squares colour fit (_cc) - dancer_splats.evaluate._crop_scores, the
    metric code of the other tracks. -> hybrid/scores.json. Caveats per view in `seen_by` (spec): the current avatar's
    bake saw every phone; LHM++ / the hybrids saw phone 06's crops (i2_c06, i4_c06 are input crops of lhmpp_1phone)."""
    from ..dancer_splats.evaluate import _crop_scores
    from .appearance_avatar_ags import load_spec
    from .appearance_compare import read_mask
    spec = load_spec(take)
    work = PACKAGE_ROOT / "work" / take
    root = compare_dir(take) / "renders"
    methods = [m for m in methods if (root / m).exists()]
    per: dict = {m: {} for m in methods}
    for v in spec["real_views"]:
        photo = cv2.imread(str(PACKAGE_ROOT / v["file"]))[:, :, ::-1].astype(np.float32) / 255
        H, W = photo.shape[:2]
        for r in ROLES:
            mk = read_mask(work, v["cam"], v["frame"], r, (H, W))
            if mk is None or mk.sum() < 200:
                continue
            ys, xs = np.nonzero(mk)
            py, px = int(0.1 * (ys.max() - ys.min())), int(0.1 * (xs.max() - xs.min()))
            y0, y1 = max(ys.min() - py, 0), min(ys.max() + py + 1, H)
            x0, x1 = max(xs.min() - px, 0), min(xs.max() + px + 1, W)
            gt, sel = photo[y0:y1, x0:x1], mk[y0:y1, x0:x1]
            for m in methods:
                im = cv2.imread(str(root / m / f"{v['id']}.png"), cv2.IMREAD_UNCHANGED)
                if im is None or im.shape[:2] != (H, W):
                    continue
                a = im[..., 3:4].astype(np.float32) / 255
                pred = (im[..., :3][..., ::-1].astype(np.float32) / 255 * a)[y0:y1, x0:x1]
                sc = _crop_scores(pred, gt, sel, None, "cpu")
                if sc:
                    sc["coverage"] = round(float((a[y0:y1, x0:x1, 0] > 0.5)[sel].mean()), 3)
                    per[m][f"{v['id']}_{r}"] = {k: (round(float(x), 4) if isinstance(x, float) else x) for k, x in sc.items()}
    summ = {}
    for m in methods:
        d = per[m]
        if not d:
            continue
        summ[m] = {}
        for subset, keys in (("all", list(d)), ("cam03", [k for k in d if "_c03_" in k]),
                             ("lead", [k for k in d if k.endswith("_lead")]), ("follow", [k for k in d if k.endswith("_follow")])):
            if keys:
                summ[m][subset] = {q: round(float(np.median([d[k][q] for k in keys])), 3)
                                   for q in ("psnr", "ssim", "psnr_cc", "ssim_cc", "coverage")} | {"n": len(keys)}
    out = {"how": score.__doc__.split("\n")[0] + " ...", "summary_median": summ, "per_view": per}
    (hybrid_dir(take) / "scores.json").write_text(json.dumps(out, indent=1))
    for m, sm in summ.items():
        log(f"score {m}: {json.dumps(sm['all'])}  cam03 {json.dumps(sm.get('cam03'))}")
    return out


def write_manifest(take: str = TAKE) -> Path:
    """hybrid/manifest.json: every file of the hybrid avatar, what it is, and how the Unity build would load it."""
    from .appearance_avatar_ags import _md5
    hd = hybrid_dir(take)
    exp = PACKAGE_ROOT / "work" / take / "export" / CAPTURE

    def f(p: Path, what: str) -> dict:
        return {"file": _rel(p), "what": what, "bytes": p.stat().st_size, "md5": _md5(p)} if p.exists() else \
            {"file": _rel(p), "what": what, "missing": True}
    files = []
    for r in ROLES:
        files += [f(hd / f"{r}_albedo.png", f"{r}: 2048 x 2048 sRGB albedo baked from LHM++ on the SHIPPED skin's SMPL-X UV "
                                          f"layout (= the layout of export/{CAPTURE}/{r}_albedo.png; PNG row 0 = v 1)"),
                  f(hd / f"{r}_coverage.png", f"{r}: 0-255 measured share per texel (0 = filled: mirror / diffusion)"),
                  f(hd / f"{r}_offset.png", f"{r}: splat surface offset along the normal, 0-255 = 0-15 cm (negative clipped)"),
                  f(hd / f"smplx_{r}_skin.bin", f"{r}: SHELL skin, DCXS v2 like export/{CAPTURE}/smplx_{r}_skin.bin (Unity "
                                                "coordinates), subdivided once (43554 vertices) and pushed out by the "
                                                "offset field; vertex_ids of midpoints are >= 10475 (edge ids)"),
                  f(hd / "shell" / f"{r}_albedo.png", f"{r}: albedo baked on the shell skin (same UV layout)")]
    man = {"schema": "dancecap.mono4d.appearance_hybrid.hybrid/1", "take": take,
           "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "code": "dancecap/dancecap/mono4d/appearance_hybrid.py",
           "files": files,
           "unity": {
               "hybrid_mesh": f"drop-in: copy hybrid/<role>_albedo.png over export/{CAPTURE}/<role>_albedo.png (and its "
                              "StreamingAssets mirror) - same skin, same UVs, same 2048 PNG; hair cards / shoes / blinks "
                              "work unchanged (the hair texture is painted on the scalp and back, the cards add volume)",
               "hybrid_mesh_shell": f"replace export/{CAPTURE}/smplx_<role>_skin.bin by hybrid/smplx_<role>_skin.bin and "
                                    "the albedo by hybrid/shell/<role>_albedo.png. NOT tested in Unity. Caveats: SmplxFace "
                                    "blendshapes (blinks, expressions) and HeadPropGeometry key on SMPL-X vertex ids, so "
                                    "the new midpoint vertices would not follow eyelid / face deltas (the face is kept "
                                    "within 4 mm of the skin, but blinks would tear); SneakerBuilder reads the pushed-out "
                                    "foot vertices; 2 x 43554 vertices / 2 x 83752 triangles (UInt32 index buffer already)"},
           "licence": "the albedos / shells are derived from LHM++ output (weights CC BY-NC 4.0) and SMPL-X (non-commercial):"
                      " research use only, never committed (work/ is git-ignored)",
           "reports": {k: _rel(hd / k) for k in sorted(x.name for x in hd.glob("*.json") if x.name != "manifest.json")},
           "check_sheets": sorted(_rel(x) for x in (hd / "check").glob("*.jpg")),
           "export_reference": {r: f(exp / f"{r}_albedo.png", "the current shipped albedo") for r in ROLES}}
    p = hd / "manifest.json"
    p.write_text(json.dumps(man, indent=1))
    return p


# ------------------------------------------------------------------------------------------------ driver
def _needs_cuda_env() -> bool:
    import os
    import shutil

    from ..env import IS_WINDOWS
    return bool(IS_WINDOWS and shutil.which("cl") is None and not os.environ.get("DANCECAP_CUDA_ENV"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.appearance_hybrid", description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["fit", "facereg", "bake", "fidelity", "shell", "render", "check", "score", "manifest", "all"])
    ap.add_argument("--take", default=TAKE)
    ap.add_argument("--roles", default=",".join(ROLES))
    ap.add_argument("--method", choices=METHODS, default="hybrid_mesh")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    roles = [r for r in a.roles.split(",") if r]
    if a.action in ("fit", "facereg", "bake", "fidelity", "shell", "all") and a.device == "cuda" and _needs_cuda_env():
        import os
        import subprocess
        import sys

        from ..env import cuda_wrapper
        argv = sys.argv[1:] if argv is None else list(argv)
        r = subprocess.run(cuda_wrapper(["-m", "dancecap.mono4d.appearance_hybrid", *argv]), cwd=PACKAGE_ROOT,
                           env={**os.environ, "DANCECAP_CUDA_ENV": "1", "PYTHONUNBUFFERED": "1"})
        return r.returncode
    from .. import gpu
    if a.action in ("fit", "all"):
        for r in roles:
            fit_apose(r, a.take, a.iters)
        with gpu.exclusive("compare-hybrid-fitcheck", timeout_s=3600):
            for r in roles:
                log(f"check -> {apose_check(r, a.take, a.device)}")
    if a.action in ("facereg", "all"):
        with gpu.exclusive("compare-hybrid-facereg", timeout_s=3600):
            for r in roles:
                face_register(r, a.take, a.device)
    if a.action in ("bake", "all"):
        with gpu.exclusive("compare-hybrid-bake", timeout_s=3600):
            for r in roles:
                bake(r, a.take, device=a.device)
                fidelity_check(r, a.take, a.device)
    if a.action == "fidelity":
        with gpu.exclusive("compare-hybrid-fidelity", timeout_s=3600):
            for r in roles:
                fidelity_check(r, a.take, a.device)
    if a.action in ("shell", "all"):
        for r in roles:
            shell_skin(r, a.take)
        with gpu.exclusive("compare-hybrid-bake-shell", timeout_s=3600):
            for r in roles:
                bake(r, a.take, device=a.device, shell=True)
    if a.action in ("render", "all"):
        groups = tuple(g for g in a.groups.split(",") if g) if a.groups else None
        ms = METHODS if a.action == "all" else (a.method,)
        for m in ms:
            render_method(m, a.take, groups, device=a.device, ids=a.ids.split(",") if a.ids else None)
    if a.action in ("score", "all"):
        score(a.take)
    if a.action in ("check", "all"):
        for p in check(a.take):
            log(f"sheet -> {p}")
    if a.action in ("manifest", "all"):
        log(f"manifest -> {write_manifest(a.take)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
