"""Gated route (FLAME): SMIRK expression encoder on the video faces, FLAME -> SMPL-X expression transfer, and a
head-to-head robustness test against MediaPipe on the same degraded crops.

    python -m dancecap.face_smirk <take> run        face/jobs.json -> face/smirk.npz (per job: FLAME expression 50,
                                                    jaw 3, eyelids 2, mouth metrics, SMPL-X expression via the mesh)
    python -m dancecap.face_smirk <take> degrade    face/degrade.json + review/face_degrade.png: large faces shrunk
                                                    to IOD 8/12/16/20 px (area resize + JPEG q70), both methods
                                                    compared with their own full-resolution answer

Needs (git-ignored, never redistributed): HeadMovement/models/flame/FLAME2020/generic_model.pkl (FLAME 2020,
flame.is.tue.mpg.de, user download), models/flame/SMPL-X__FLAME_vertex_ids.npy (SMPL-X site, "SMPL-X to FLAME
vertex correspondence"), external/smirk (github.com/georgeretsi/smirk, MIT, commit c7de404) with
pretrained_models/SMIRK_em1.pt (140 MB, the README's Google-Drive link). SMIRK's own FLAME class needs chumpy; we
load the pickle with a stub unpickler instead (no chumpy, no pytorch3d: the renderer is not used).

FLAME -> SMPL-X: the SMPL-X 1.1 expression basis is NOT FLAME 2020's (measured through the vertex correspondence:
only component 0 matches, cos 0.93; FLAME's first 10 directions are 67 % inside SMPL-X's first 10, 80 % inside all
100). So codes are transferred through geometry: SMPL-X expression = least squares fit to the FLAME expression
displacement on the 5023 corresponding vertices (face/lips/eye-region weighted); jaw copied (same rest frame).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .face_common import IBUG52, LABEL, MP_TO_IBUG68, ROLES, smplx_model_npz
from .take import WORKSPACE, Take

SMIRK_DIR = WORKSPACE / "external" / "smirk"
FLAME_PKL = WORKSPACE / "models" / "flame" / "FLAME2020" / "generic_model.pkl"
SX_FLAME_IDS = WORKSPACE / "models" / "flame" / "SMPL-X__FLAME_vertex_ids.npy"
N_SX = 20


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.__dict__.update(s if isinstance(s, dict) else {"state": s})


class _NoChumpy(pickle.Unpickler):
    def find_class(self, mod, name):
        return _Stub if mod.startswith("chumpy") else super().find_class(mod, name)


def load_flame(path: Path = FLAME_PKL) -> dict:
    d = _NoChumpy(open(path, "rb"), encoding="latin1").load()
    arr = lambda v: np.asarray(v.__dict__.get("x")) if isinstance(v, _Stub) else (
        np.asarray(v.todense()) if hasattr(v, "todense") else np.asarray(v))
    return {k: arr(d[k]) for k in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f")}


class FlameGeo:
    """Rest-pose FLAME with expression (50), jaw (3) and SMIRK's eyelid blendshapes; landmarks; SMPL-X transfer."""

    def __init__(self, n_sx: int = N_SX):
        F = load_flame()
        self.vt = F["v_template"].astype(np.float64)
        self.E = F["shapedirs"][:, :, 300:350].astype(np.float64)
        self.W = F["weights"][:, 2].astype(np.float64)                      # jaw skinning
        self.Jjaw = F["J_regressor"][2] @ self.vt
        self.f = F["f"].astype(np.int64)
        a = SMIRK_DIR / "assets"
        self.leye, self.reye = np.load(a / "l_eyelid.npy"), np.load(a / "r_eyelid.npy")
        lm = np.load(a / "landmark_embedding.npy", allow_pickle=True, encoding="latin1")[()]
        self.lmk_f = np.asarray(lm["full_lmk_faces_idx"])[0].astype(np.int64)
        self.lmk_b = np.asarray(lm["full_lmk_bary_coords"])[0].astype(np.float64)
        masks = pickle.load(open(a / "FLAME_masks" / "FLAME_masks.pkl", "rb"), encoding="latin1")
        w = np.full(len(self.vt), 0.1)
        for k, s in (("face", 1.0), ("lips", 3.0), ("eye_region", 2.0)):
            w[np.asarray(masks[k])] = np.maximum(w[np.asarray(masks[k])], s)
        for k in ("left_eyeball", "right_eyeball", "neck", "scalp", "boundary"):
            w[np.asarray(masks[k])] = 0.0
        self.vw = w
        ids = np.load(SX_FLAME_IDS)
        m = np.load(smplx_model_npz(), allow_pickle=True)
        S = np.asarray(m["shapedirs"], float)[ids][:, :, 300:300 + n_sx]       # (5023, 3, n)
        A = (S * np.sqrt(w)[:, None, None]).reshape(-1, n_sx)
        self.n_sx = n_sx
        self.A = A
        self.sx_pinv = np.linalg.solve(A.T @ A + 1e-6 * np.eye(n_sx), A.T)     # ridge pseudo-inverse
        self.sqrt_w = np.sqrt(w)

    def verts(self, exp, jaw, eyelid):
        """exp (N,50), jaw (N,3), eyelid (N,2) -> (N,5023,3) rest-pose head (no identity, no head pose)."""
        from scipy.spatial.transform import Rotation
        V = self.vt[None] + np.einsum("vdk,nk->nvd", self.E, exp)
        V = V + eyelid[:, 0, None, None] * self.leye[None] + eyelid[:, 1, None, None] * self.reye[None]
        R = Rotation.from_rotvec(jaw).as_matrix()
        Vc = V - self.Jjaw
        return V + self.W[None, :, None] * (np.einsum("nij,nvj->nvi", R, Vc) - Vc)

    def landmarks(self, V):
        return np.einsum("nlkd,lk->nld", V[:, self.f[self.lmk_f]], self.lmk_b)

    def metrics(self, exp, jaw, eyelid) -> dict:
        V = self.verts(exp, jaw, eyelid)
        L = self.landmarks(V) * 1000.0                                       # mm
        L0 = self.landmarks(self.verts(np.zeros((1, 50)), np.zeros((1, 3)), np.zeros((1, 2))))[0] * 1000.0
        gap = (np.linalg.norm(L[:, 62] - L[:, 66], axis=1) + np.linalg.norm(L[:, 61] - L[:, 67], axis=1) +
               np.linalg.norm(L[:, 63] - L[:, 65], axis=1)) / 3
        width = np.linalg.norm(L[:, 48] - L[:, 54], axis=1)
        smile = ((L[:, 48, 1] - L0[48, 1]) + (L[:, 54, 1] - L0[54, 1])) / 2 + (width - np.linalg.norm(L0[48] - L0[54])) / 2
        brow = ((L[:, 17:27, 1] - L0[17:27, 1]).mean(1))
        return {"mouth_open_mm": gap, "mouth_width_mm": width, "smile_mm": smile, "brow_mm": brow,
                "jaw_deg": np.degrees(jaw[:, 0])}

    def to_smplx(self, exp, eyelid) -> tuple[np.ndarray, np.ndarray]:
        """FLAME expression (+ eyelids) displacement -> SMPL-X expression (N, n_sx) and relative fit residual."""
        D = np.einsum("vdk,nk->nvd", self.E, exp) + eyelid[:, 0, None, None] * self.leye[None] + \
            eyelid[:, 1, None, None] * self.reye[None]
        b = (D * self.sqrt_w[None, :, None]).reshape(len(D), -1)
        x = b @ self.sx_pinv.T
        res = np.linalg.norm(x @ self.A.T - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-12)
        return x, res


# ------------------------------------------------------------------------------------------------ SMIRK


def load_smirk(device: str = "cpu"):
    import timm
    import torch
    sys.path.insert(0, str(SMIRK_DIR))
    orig = timm.create_model
    timm.create_model = lambda name, pretrained=True, **kw: orig(name, pretrained=False, **kw)  # weights come from the ckpt
    try:
        from src.smirk_encoder import SmirkEncoder
        enc = SmirkEncoder()
    finally:
        timm.create_model = orig
    ck = torch.load(SMIRK_DIR / "pretrained_models" / "SMIRK_em1.pt", map_location="cpu")
    enc.load_state_dict({k.replace("smirk_encoder.", ""): v for k, v in ck.items() if "smirk_encoder" in k})
    return enc.eval().to(device)


def smirk_box(mp478: np.ndarray | None, model52: np.ndarray, calib: dict) -> tuple[np.ndarray, float]:
    """SMIRK's crop (demo.crop_face, scale 1.4 on the MediaPipe landmark box); without MediaPipe the box is
    predicted from the projected SMPL-X landmarks with the calibrated centre offset / size ratio."""
    if mp478 is not None and np.isfinite(mp478).all():
        lo, hi = mp478.min(0), mp478.max(0)
        return (lo + hi) / 2, float(((hi - lo).sum() / 2) * 1.4)
    lo, hi = model52.min(0), model52.max(0)
    s = (hi - lo).sum() / 2
    return (lo + hi) / 2 + np.asarray(calib["offset"]) * s, float(s * calib["ratio"] * 1.4)


def warp_crop(img, center, size, out=224):
    s = out / size
    A = np.array([[s, 0, out / 2 - s * center[0]], [0, s, out / 2 - s * center[1]]])
    return cv2.warpAffine(img, A, (out, out), flags=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA,
                          borderMode=cv2.BORDER_CONSTANT)


def encode(enc, crops_bgr: list[np.ndarray], device: str = "cpu", bs: int = 32) -> dict:
    import torch
    outs = {k: [] for k in ("expression_params", "jaw_params", "eyelid_params", "pose_params", "cam", "shape_params")}
    with torch.no_grad():
        for a in range(0, len(crops_bgr), bs):
            x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr[a:a + bs]]).astype(np.float32) / 255
            o = enc(torch.as_tensor(x).permute(0, 3, 1, 2).to(device))
            for k in outs:
                outs[k].append(o[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def calibrate_crop(det, jobs) -> dict:
    """MediaPipe box vs projected SMPL-X landmark box (centre offset in units of the SMPL-X box size, size ratio)."""
    offs, ratios = [], []
    for k, j in enumerate(jobs):
        if not det["mp_ok"][k]:
            continue
        mp = det["mp_lmk"][k, :, :2]
        m52 = np.asarray(j["lmk_model"])
        lo, hi = mp.min(0), mp.max(0)
        l2, h2 = m52.min(0), m52.max(0)
        s2 = (h2 - l2).sum() / 2
        offs.append(((lo + hi) / 2 - (l2 + h2) / 2) / s2)
        ratios.append(((hi - lo).sum() / 2) / s2)
    return {"offset": np.median(offs, 0).tolist() if offs else [0.0, -0.1],
            "ratio": float(np.median(ratios)) if ratios else 1.3, "n": len(offs)}


def run_jobs(take: Take, device: str = "cpu") -> dict:
    work = take.work
    jobs = json.loads((work / "face" / "jobs.json").read_text())["jobs"]
    det = np.load(work / "face" / "detections.npz")
    calib = calibrate_crop(det, jobs)
    enc = load_smirk(device)
    geo = FlameGeo()
    crops, src = [], []
    cache = {}
    for k, j in enumerate(jobs):
        key = (j["cam"], j["i"])
        if key not in cache:
            cache.clear()
            cache[key] = cv2.imread(str(work / "frames" / j["cam"] / f"{j['i']:06d}.jpg"))
        mp = det["mp_lmk"][k, :, :2] if det["mp_ok"][k] else None
        c, s = smirk_box(mp, np.asarray(j["lmk_model"]), calib)
        crops.append(warp_crop(cache[key], c, s))
        src.append("mediapipe" if mp is not None else "smplx")
    t0 = time.time()
    o = encode(enc, crops, device)
    met = geo.metrics(o["expression_params"], o["jaw_params"], o["eyelid_params"])
    sx, sx_res = geo.to_smplx(o["expression_params"], o["eyelid_params"])
    out = {**o, **met, "smplx_expression": sx, "smplx_transfer_residual": sx_res, "crop_source": np.array(src)}
    np.savez_compressed(work / "face" / "smirk.npz", **out)
    np.save(work / "face" / "smirk_crops_preview.npy", np.stack(crops[:: max(1, len(crops) // 64)]))
    return {"jobs": len(jobs), "calib": calib, "runtime_s": round(time.time() - t0, 1),
            "crop_from_mediapipe": int(sum(s == "mediapipe" for s in src)),
            "flame_to_smplx_relative_residual_median": round(float(np.median(sx_res)), 3)}


# ------------------------------------------------------------------------------------------------ degrade test


LEVELS = (8, 12, 16, 20)


def degrade(img_crop: np.ndarray, factor: float, q: int = 70) -> np.ndarray:
    h, w = img_crop.shape[:2]
    small = cv2.resize(img_crop, (max(4, int(round(w * factor))), max(4, int(round(h * factor)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
    small = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def degrade_test(take: Take, min_iod: float = 24.0, device: str = "cpu") -> dict:
    """Large usable faces -> shrunk to LEVELS (IOD px) -> SMIRK and MediaPipe on identical crops. Ground truth =
    each method's own full-resolution answer (consistency under shrinking, i.e. what our tiny faces cost)."""
    import subprocess

    from .avatar_face import FACE_VENV_PY
    from .face_eval import load
    work = take.work
    d = load(work)
    det = np.load(work / "face" / "detections.npz")
    jobs = d["jobs"]
    sel = np.flatnonzero(d["base"] & d["mp_ok"] & (d["iod"] >= min_iod))
    calib = calibrate_crop(det, jobs)
    # one 512 px context crop per face (SMIRK box x 1.0, same for both methods), degraded per level
    ctx, meta = [], []
    for k in sel:
        j = jobs[k]
        img = cv2.imread(str(work / "frames" / j["cam"] / f"{j['i']:06d}.jpg"))
        c, s = smirk_box(None, np.asarray(j["lmk_model"]), calib)      # model-based box at every level (fair)
        ctx.append(warp_crop(img, c, s * 1.25, 448))
        meta.append({"k": int(k), "role": j["role"], "cam": j["cam"], "i": j["i"], "iod": j["iod"], "box_side": s * 1.25})
    levels = {"full": 1.0}
    variants = {"full": ctx}
    for L in LEVELS:
        variants[f"iod{L}"] = [degrade(cimg, L / m["iod"]) if L < m["iod"] else cimg for cimg, m in zip(ctx, meta)]
    # SMIRK: centre 224 crop of the 448 context = the SMIRK box
    enc = load_smirk(device)
    geo = FlameGeo()
    res = {}
    for name, crops in variants.items():
        o = encode(enc, [cv2.resize(c[56:392, 56:392], (224, 224), interpolation=cv2.INTER_AREA) for c in crops], device)
        res[f"smirk_{name}"] = {**o, **geo.metrics(o["expression_params"], o["jaw_params"], o["eyelid_params"])}
    # MediaPipe in .venv-face on the same crops
    tmp = work / "face" / "degrade_crops.npz"
    np.savez_compressed(tmp, **{n: np.stack(v) for n, v in variants.items()})
    outp = work / "face" / "degrade_mp.npz"
    subprocess.run([str(FACE_VENV_PY), "-m", "dancecap.face_worker", "crops", str(tmp), str(outp)],
                   cwd=str(WORKSPACE / "dancecap"), check=True, capture_output=True)
    mpz = np.load(outp)
    names = [str(x) for x in mpz["blend_names"]]
    rep = {"faces": len(meta), "roles": {r: int(sum(m["role"] == r for m in meta)) for r in ROLES},
           "levels_iod_px": list(LEVELS), "calib": calib, "smirk": {}, "mediapipe": {}}

    def stats(full, lev, ok=None):
        ok = np.ones(len(full), bool) if ok is None else ok
        if ok.sum() < 3:
            return {"n": int(ok.sum())}
        f, l = full[ok], lev[ok]
        sd = float(np.std(full)) + 1e-9
        return {"n": int(ok.sum()), "corr": round(float(np.corrcoef(f, l)[0, 1]), 3) if np.std(l) > 1e-9 else None,
                "mae_over_sd": round(float(np.mean(np.abs(f - l)) / sd), 3), "sd_full": round(sd, 4)}
    for L in LEVELS:
        a, b = res["smirk_full"], res[f"smirk_iod{L}"]
        rep["smirk"][f"iod{L}"] = {m: stats(a[m], b[m]) for m in ("mouth_open_mm", "smile_mm", "jaw_deg", "brow_mm")}
        rep["smirk"][f"iod{L}"]["expr_code_corr"] = round(float(np.mean(
            [np.corrcoef(a["expression_params"][n], b["expression_params"][n])[0, 1] for n in range(len(meta))])), 3)
        okf, okl = mpz["full_ok"], mpz[f"iod{L}_ok"]
        both = okf & okl
        g = lambda z, n: z[:, names.index(n)]
        mf, ml = mpz["full_blend"], mpz[f"iod{L}_blend"]
        rep["mediapipe"][f"iod{L}"] = {
            "detect_rate": round(float(okl.mean()), 3),
            "jawOpen": stats(g(mf, "jawOpen"), g(ml, "jawOpen"), both),
            "mouthSmile": stats((g(mf, "mouthSmileLeft") + g(mf, "mouthSmileRight")) / 2,
                                (g(ml, "mouthSmileLeft") + g(ml, "mouthSmileRight")) / 2, both),
            "browInnerUp": stats(g(mf, "browInnerUp"), g(ml, "browInnerUp"), both)}
    rep["mediapipe"]["full_detect_rate"] = round(float(mpz["full_ok"].mean()), 3)
    (work / "face" / "degrade.json").write_text(json.dumps(rep, indent=1))
    rep["review"] = [str(_degrade_plot(rep, work / "review" / "face_degrade.png", take.name)),
                     str(_degrade_sheet(variants, res, meta, mpz, work / "review" / "face_degrade_crops.jpg", take.name))]
    (work / "face" / "degrade.json").write_text(json.dumps(rep, indent=1))
    return rep


def _degrade_plot(rep, path: Path, take: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = rep["levels_iod_px"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    col = {"mouth_open_mm": "C0", "smile_mm": "C1", "jaw_deg": "C2", "jawOpen": "C3", "mouthSmile": "C4"}
    for key, lab in (("mouth_open_mm", "SMIRK mouth opening"), ("smile_mm", "SMIRK smile"), ("jaw_deg", "SMIRK jaw")):
        ax[0].plot(L, [rep["smirk"][f"iod{x}"][key].get("corr") or np.nan for x in L], "o-", color=col[key], label=lab)
    for key, lab in (("jawOpen", "MediaPipe jawOpen"), ("mouthSmile", "MediaPipe mouthSmile")):
        ax[0].plot(L, [rep["mediapipe"][f"iod{x}"][key].get("corr") or np.nan for x in L], "s--", color=col[key], label=lab)
    ax[0].set_title("correlation with the method's own full-res answer", fontsize=9)
    ax[0].set_xlabel("simulated IOD (px)")
    ax[0].set_ylim(-0.2, 1.05)
    ax[0].legend(fontsize=7)
    for key, lab in (("mouth_open_mm", "SMIRK mouth opening"), ("smile_mm", "SMIRK smile")):
        ax[1].plot(L, [rep["smirk"][f"iod{x}"][key].get("mae_over_sd") or np.nan for x in L], "o-", color=col[key], label=lab)
    for key, lab in (("jawOpen", "MediaPipe jawOpen (near-constant signal)"), ("mouthSmile", "MediaPipe mouthSmile")):
        ax[1].plot(L, [rep["mediapipe"][f"iod{x}"][key].get("mae_over_sd") or np.nan for x in L], "s--", color=col[key],
                   label=lab)
    ax[1].axhline(1.0, color="grey", lw=0.6)
    ax[1].set_title("mean abs error / spread of the full-res signal (1 = no information)", fontsize=9)
    ax[1].set_xlabel("simulated IOD (px)")
    ax[1].legend(fontsize=7)
    ax[2].bar([str(x) for x in L], [rep["mediapipe"][f"iod{x}"]["detect_rate"] for x in L], color="C0", label="MediaPipe")
    ax[2].bar([str(x) for x in L], [0] * len(L))
    ax[2].axhline(1.0, color="C1", lw=2, label="SMIRK (no detector: always answers)")
    ax[2].set_title("face found (MediaPipe) on the shrunk crops", fontsize=9)
    ax[2].set_ylim(0, 1.1)
    ax[2].legend(fontsize=7)
    fig.suptitle(f"{take}: {rep['faces']} large video faces shrunk to small sizes - SMIRK (FLAME) vs MediaPipe")
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path


def _render_flame(geo: FlameGeo, exp, jaw, eyelid, size=160) -> np.ndarray:
    """Frontal shaded render of the rest-pose FLAME head (orthographic, Lambert) with the given expression."""
    import torch

    from .avatar_raster import rasterize
    V = geo.verts(exp[None], jaw[None], eyelid[None])[0]
    c = V.mean(0)
    s = size / 0.24
    px = np.stack([(V[:, 0] - c[0]) * s + size / 2, -(V[:, 1] - c[1] - 0.01) * s + size / 2], 1)
    z = 10.0 - V[:, 2]
    zb, fid = rasterize(torch, torch.tensor(px, dtype=torch.float32), torch.tensor(z, dtype=torch.float32),
                        torch.as_tensor(geo.f), size, size)
    fid = fid.numpy()
    tri = V[geo.f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    shade = np.clip(n @ np.array([0.3, 0.4, 0.87]), 0.05, 1)
    img = np.full((size, size, 3), 30, np.uint8)
    cov = fid >= 0
    img[cov] = (np.clip(shade[fid[cov]], 0, 1)[:, None] * np.array([200, 205, 215])).astype(np.uint8)
    return img


def _degrade_sheet(variants, res, meta, mpz, path: Path, take: str, n: int = 5, tile: int = 150) -> Path:
    from .face_common import label
    geo = FlameGeo()
    order = np.argsort(-res["smirk_full"]["mouth_open_mm"])
    pick = list(order[:2]) + list(order[len(order) // 2:len(order) // 2 + 1]) + list(order[-2:])
    rows = []
    for k in pick[:n]:
        tiles = []
        for name in ["full"] + [f"iod{L}" for L in LEVELS]:
            c = cv2.resize(variants[name][k][56:392, 56:392], (tile, tile), interpolation=cv2.INTER_AREA)
            o = res[f"smirk_{name}"]
            label(c, name, (3, 12), scale=0.38)
            ok = bool(mpz[f"{name}_ok"][k])
            label(c, f"MP {'ok' if ok else '--'}", (3, tile - 5), scale=0.38)
            r = _render_flame(geo, o["expression_params"][k], o["jaw_params"][k], o["eyelid_params"][k], tile)
            label(r, f"open {o['mouth_open_mm'][k]:.0f}mm smile {o['smile_mm'][k]:+.1f}", (3, tile - 5), scale=0.36)
            tiles += [c, r, np.zeros((tile, 3, 3), np.uint8)]
        row = np.concatenate(tiles, 1)
        hs = np.full((16, row.shape[1], 3), 20, np.uint8)
        m = meta[k]
        label(hs, f"{LABEL[m['role']]} cam {m['cam']} #{m['i']} (IOD {m['iod']:.0f}px)", (3, 12), scale=0.4)
        rows.append(np.concatenate([hs, row], 0))
    sheet = np.concatenate(rows, 0)
    top = np.full((24, sheet.shape[1], 3), 10, np.uint8)
    label(top, f"{take}: shrunk crop | SMIRK FLAME expression (neutral identity, frontal) at each size", (4, 17), scale=0.45)
    cv2.imwrite(str(path), np.concatenate([top, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("what", choices=["run", "degrade", "agree", "review", "all"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    take = Take.load(a.take)
    if a.what in ("run", "all"):
        print(json.dumps(run_jobs(take, a.device), indent=1))
    if a.what in ("degrade", "all"):
        print(json.dumps(degrade_test(take, device=a.device), indent=1))
    if a.what in ("agree", "all"):
        print(json.dumps(agreement(take), indent=1))
    if a.what in ("review", "all"):
        print(review_avatar(take))


def agreement(take: Take) -> dict:
    """Real-video checks for SMIRK vs MediaPipe (usable views): (a) the same dancer seen by two cameras at the same
    instant (+-20 ms) should give the same expression; (b) SMIRK vs MediaPipe on the same view (independent methods);
    (c) frame-to-frame jitter in the same camera relative to the signal's spread."""
    from .face_eval import _mouth_open, load
    work = take.work
    d = load(work)
    z = np.load(work / "face" / "smirk.npz")
    names = d["blend_names"]
    g = lambda n: d["mp_blend"][:, names.index(n)]
    sig = {"smirk_mouth_open_mm": z["mouth_open_mm"], "smirk_smile_mm": z["smile_mm"], "smirk_jaw_deg": z["jaw_deg"],
           "mp_jawOpen": g("jawOpen"), "mp_mouthSmile": (g("mouthSmileLeft") + g("mouthSmileRight")) / 2,
           "mp_lip_gap": _mouth_open(d["mp52"])}
    is_mp = lambda s: s.startswith("mp_")
    out = {"cross_camera": {}, "smirk_vs_mp_same_view": {}, "jitter_over_sd": {}, "spread": {}}
    for r in ROLES:
        base = (d["role"] == r) & d["base"]
        for min_iod in (10, 20):
            idx = np.flatnonzero(base & (d["iod"] >= min_iod))
            pairs = [(p, q) for p in idx for q in idx if p < q and d["cam"][p] != d["cam"][q] and abs(d["t"][p] - d["t"][q]) < 0.02]
            res = {}
            for s, v in sig.items():
                pp = [(p, q) for p, q in pairs if (not is_mp(s)) or (d["mp_ok"][p] and d["mp_ok"][q])]
                if len(pp) >= 5:
                    a = np.array([[v[p], v[q]] for p, q in pp])
                    res[s] = {"pairs": len(pp), "corr": round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3),
                              "mean_abs_diff_over_sd": round(float(np.abs(a[:, 0] - a[:, 1]).mean() / (np.std(v[base]) + 1e-9)), 3)}
                else:
                    res[s] = {"pairs": len(pp)}
            out["cross_camera"][f"{r}_iod{min_iod}"] = res
        ok = base & d["mp_ok"]
        for a_, b_ in (("smirk_mouth_open_mm", "mp_lip_gap"), ("smirk_mouth_open_mm", "mp_jawOpen"),
                       ("smirk_smile_mm", "mp_mouthSmile")):
            if ok.sum() > 5:
                out["smirk_vs_mp_same_view"][f"{r}: {a_} ~ {b_}"] = {
                    "n": int(ok.sum()), "corr": round(float(np.corrcoef(sig[a_][ok], sig[b_][ok])[0, 1]), 3)}
        order = np.lexsort((d["i"], d["cam"], d["role"]))
        for s, v in sig.items():
            dif = []
            for a_, b_ in zip(order[:-1], order[1:]):
                if d["role"][a_] == r and d["role"][b_] == r and d["cam"][a_] == d["cam"][b_] and d["i"][b_] - d["i"][a_] == 1 \
                        and d["base"][a_] and d["base"][b_] and (not is_mp(s) or (d["mp_ok"][a_] and d["mp_ok"][b_])):
                    dif.append(abs(v[b_] - v[a_]))
            sd = np.std(v[base & (d["mp_ok"] if is_mp(s) else True)]) + 1e-9
            out["jitter_over_sd"][f"{r} {s}"] = round(float(np.median(dif) / sd), 3) if dif else None
            out["spread"][f"{r} {s}"] = round(float(sd), 4)
    (work / "face" / "smirk_agreement.json").write_text(json.dumps(out, indent=1))
    return out


def review_avatar(take: Take, n: int = 4, tile: int = 220, n_sx: int = N_SX) -> Path:
    """Video | avatar neutral | landmark fit (face_expr) | SMIRK -> SMPL-X (same neck correction) on the largest
    usable faces where SMIRK's mouth differs most from its median. Portrait-face textures when present."""
    import smplx
    import torch

    from .avatar_texture import _Geo, render
    from .avatar_uv import build_atlas
    from .face_common import cameras_world, crop_pad, label
    from .face_eval import load
    from .mamma_dyn import SMPLX_MODELS
    work = take.work
    d = load(work)
    z = np.load(work / "face" / "smirk.npz")
    atlas = build_atlas(SMPLX_MODELS, 2048)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geo = _Geo(atlas, dev)
    cols = {}
    for r in ROLES:
        p = work / "avatar" / "face" / f"{r}_albedo_face.png"
        p = p if p.exists() else work / "avatar" / f"{r}_albedo.png"
        im = cv2.imread(str(p))[:, :, ::-1].astype(np.float32) / 255
        cols[r] = torch.as_tensor(im[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]], device=dev)
    camw = cameras_world(work, sorted(set(d["cam"])))
    m = smplx.create(str(SMPLX_MODELS), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                     flat_hand_mean=True, num_betas=16, num_expression_coeffs=n_sx, batch_size=1)
    f = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float32)[None]

    def pose(s, gk, neck=None, jaw=np.zeros(3), expr=np.zeros(n_sx)):
        bp = np.asarray(s["body_pose"][gk], float).copy()
        if neck is not None:
            bp[33:36] = neck
        with torch.no_grad():
            o = m(betas=f(np.asarray(s["betas"]).reshape(-1)[:16]), global_orient=f(s["global_orient"][gk]),
                  body_pose=f(bp), jaw_pose=f(jaw), expression=f(expr), transl=f(s["transl"][gk]),
                  left_hand_pose=f(s["left_hand_pose"][gk]), right_hand_pose=f(s["right_hand_pose"][gk]),
                  leye_pose=f(np.zeros(3)), reye_pose=f(np.zeros(3)))
        return o.vertices[0].numpy()

    rows = []
    for r in ROLES:
        sm = {k: v for k, v in np.load(work / "smplx" / f"{r}.npz", allow_pickle=True).items()}
        other = [o for o in ROLES if o != r][0]
        so = {k: v for k, v in np.load(work / "smplx" / f"{other}.npz", allow_pickle=True).items()}
        ex = np.load(work / "face" / f"expr_{r}.npz")
        idx = np.flatnonzero((d["role"] == r) & d["base"] & (d["iod"] >= 18))
        if not len(idx):
            continue
        idx = idx[np.argsort(-d["iod"][idx] * np.abs(z["mouth_open_mm"][idx] - np.median(z["mouth_open_mm"][idx])))]
        picked = []
        for k in idx:
            if all(abs(d["t"][k] - d["t"][q]) > 0.4 for q in picked):
                picked.append(k)
            if len(picked) == n:
                break
        tiles = []
        for k in picked:
            gk = int(np.argmin(np.abs(np.asarray(sm["t"]) - d["t"][k])))
            c, i = d["cam"][k], int(d["i"][k])
            img = cv2.imread(str(work / "frames" / c / f"{i:06d}.jpg"))
            H, W = img.shape[:2]
            ci = camw[c]
            ex_e = np.zeros(n_sx)
            ex_e[:ex["expression"].shape[1]] = ex["expression"][gk]
            variants = [("neutral", pose(sm, gk, ex["neck_pose"][gk])),
                        ("landmark fit", pose(sm, gk, ex["neck_pose"][gk], ex["jaw_pose"][gk], ex_e)),
                        ("SMIRK", pose(sm, gk, ex["neck_pose"][gk], z["jaw_params"][k], z["smplx_expression"][k][:n_sx]))]
            Vo = pose(so, gk)
            lm = d["model52"][k]
            lo, hi = lm.min(0), lm.max(0)
            side = max(hi - lo) * 2.4
            cx = (lo + hi) / 2
            box = (int(cx[0] - side / 2), int(cx[1] - side / 2), int(cx[0] + side / 2), int(cx[1] + side / 2))
            ph = crop_pad(img, box, tile)
            label(ph, f"cam {c} #{i} IOD {d['iod'][k]:.0f}", (4, 14), scale=0.4)
            tiles.append(ph)
            for name, V in variants:
                ren, rid = render(torch, geo, {r: V, other: Vo}, cols, ci["K"][i], ci["R"][i], ci["t"][i],
                                  float(ci["k1"][i]), H, W, dev)
                ren8 = (np.clip(ren, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
                t_ = crop_pad(np.where((rid >= 0)[..., None], ren8, 40).astype(np.uint8), box, tile)
                label(t_, name, (4, 14), scale=0.4)
                if name == "SMIRK":
                    label(t_, f"open {z['mouth_open_mm'][k]:.0f}mm smile {z['smile_mm'][k]:+.0f}", (4, tile - 6), scale=0.38)
                if name == "landmark fit":
                    label(t_, f"jaw {np.degrees(ex['jaw_pose'][gk, 0]):.0f}deg conf {ex['confidence'][gk]:.1f}",
                          (4, tile - 6), scale=0.38)
                tiles.append(t_)
            tiles.append(np.zeros((tile, 4, 3), np.uint8))
        row = np.concatenate(tiles, 1)
        hs = np.full((20, row.shape[1], 3), 20, np.uint8)
        label(hs, LABEL[r], (5, 15), scale=0.45)
        rows.append(np.concatenate([hs, row], 0))
    w = max(rw.shape[1] for rw in rows)
    sheet = np.concatenate([np.pad(rw, ((0, 4), (0, w - rw.shape[1]), (0, 0))) for rw in rows], 0)
    top = np.full((26, w, 3), 10, np.uint8)
    label(top, f"{take.name}: photo | neutral | landmark-fit expression (ungated) | SMIRK expression (FLAME, gated) "
               f"- same neck correction", (5, 18), scale=0.45)
    path = work / "review" / "face_expr_smirk_vs_landmarks.jpg"
    cv2.imwrite(str(path), np.concatenate([top, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


if __name__ == "__main__":
    sys.exit(main())
