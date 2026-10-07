"""Per-frame facial expression (SMPL-X jaw + expression) for the avatars, from the multi-view video landmarks.

    python -m dancecap.face_expr <take>              -> work/<take>/smplx_face/{lead,follow}.npz, face_blendshapes.npz,
                                                        report.json, review/face_expr.jpg, review/face_expr_confidence.png
    python -m dancecap.face_expr <take> --research   -> the earlier absolute-position fit: face/expr_<role>.npz,
                                                        expr.json, review/face_expr_research*.{jpg,png}
    (needs face/survey.npz + jobs.json (face_survey), face/detections.npz (face_worker, MediaPipe; on a pod for long
     takes: dancecap.face_pod), optionally the portrait fit (avatar_face) for the identity offsets and the gated
     FLAME eyelid assets for the blink blendshapes)

The final route (build, below) is described at its section header. The research fit:
MAMMA fits no face: jaw is excluded from its optimiser (optimization.py skips "jaw"), expression is never a parameter
(always zeros) and the eye rotations are only weakly constrained. So per dancer and 30 Hz grid frame k we fit

    P_k = J12_k + Rot(w_k) (R_k (L(e_k, j_k) + D) + t_k - J12_k)

  L(e, j)   the 52 iBUG landmarks of the dancer's rest SMPL-X head with expression e (N_EXPR) and jaw rotation j
  D         identity offsets of the landmarks (from the portrait fit residuals, avatar_face; 0 without a portrait)
  R_k, t_k  the fitted body's rigid head motion (rest -> posed landmarks), J12_k its neck joint
  w_k       neck correction (world rotation about the neck joint, as avatar_head.py; -> neck_pose_with)
to the MediaPipe landmarks of every usable view (face_survey base flag, MediaPipe face within 0.5 IOD of the
expected position) with a robust (Cauchy, 0.1 IOD) reprojection loss in IOD units, temporal smoothness, and priors
pulling e towards a per-dancer resting expression e0 (also fitted) and w towards 0. Frames without observations
are carried by the smoothness and relax to e0 (confidence 0).

Checks: in-sample and leave-one-camera-out landmark errors (all / mouth points) for neutral, head-only and full
models; the held-out mouth error is the honest measure of whether expressions are real signal or noise.
larissa_kadu (2026-10-06): in-sample mouth NME 15.7 -> 4.0 -> 3.4 % IOD (lead) but held-out cameras do NOT improve
(06: 12.7 -> 11.9 -> 12.3 %; 04/05/07 and every follow camera get worse, also with 10-100x stronger priors): most
frames are seen by one camera, so the neck/expression corrections overfit that view. Do not adopt neck_pose from
here; treat expression as single-view (hero camera) semantics only.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .face_common import LABEL, ROLES, LandmarkEmbedding, cameras_world, smplx_model_npz
from .take import Take

MOUTH = tuple(range(48, 68))


@dataclass
class ExprConfig:
    n_expr: int = 10
    jaw_dims: int = 1               # 1 = opening only (x rotation); 3 = full
    cauchy_iod: float = 0.10
    expr_prior: float = 0.02        # |e - e0|^2
    e0_prior: float = 0.01
    jaw_prior: float = 0.5
    smooth_expr: float = 1.0        # |e_k+1 - e_k|^2 (30 Hz)
    smooth_jaw: float = 20.0
    neck_prior: float = 2.0         # |w|^2 (rad)
    smooth_neck: float = 20.0
    iters: int = 200
    lr: float = 0.02
    min_mp_dist_iod: float = 0.5
    device: str = "auto"            # auto = cuda when available (tiny: < 100 MB), the CPU is often saturated
    loo_min_views: int = 5


def rodrigues(torch, rv):
    """(T,3) axis-angle -> (T,3,3)."""
    th = rv.norm(dim=1, keepdim=True).clamp(min=1e-9)
    k = rv / th
    z = torch.zeros_like(k[:, 0])
    Kx = torch.stack([z, -k[:, 2], k[:, 1], k[:, 2], z, -k[:, 0], -k[:, 1], k[:, 0], z], 1).view(-1, 3, 3)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=rv.device)[None] + s * Kx + (1 - c) * Kx @ Kx


def batched_landmarks(head, emb: LandmarkEmbedding, device: str = "cpu"):
    """fn(expr (T,E), jaw (T,3)) -> (T,52,3): HeadModel.landmark_fn for a batch of frames."""
    import torch
    m = head.m
    with torch.no_grad():
        vs = m.v_template + torch.einsum("vdk,k->vd", m.shapedirs, head.betas[0])
        J22 = (m.J_regressor[22][:, None] * vs).sum(0).to(device)
    ids = torch.as_tensor(emb.faces.reshape(-1))
    vs_l = vs[ids].reshape(52, 3, 3).detach().to(device)
    E_l = m.expr_dirs[ids].reshape(52, 3, 3, -1).detach().to(device)
    w_l = m.lbs_weights[ids, 22].reshape(52, 3, 1).detach().to(device)
    bary = torch.tensor(emb.bary, dtype=torch.float32, device=device)

    def fn(expr, jaw):
        V = vs_l[None] + torch.einsum("lkdn,tn->tlkd", E_l, expr)
        R = rodrigues(torch, jaw)
        Vc = V - J22
        V = V + w_l[None] * (torch.einsum("tij,tlkj->tlki", R, Vc) - Vc)
        return (V * bary[None, ..., None]).sum(2)
    return fn


def observations(work: Path, role: str, grid_t: np.ndarray, max_dt: float = 0.02):
    """-> list of dict(k grid frame, cam, i, K, R, t, k1, uv (52,2), iod) for usable MediaPipe detections."""
    from .face_eval import load
    d = load(work)
    cams = sorted(set(d["cam"]))
    camw = cameras_world(work, cams)
    obs = []
    for n in np.flatnonzero((d["role"] == role) & d["base"] & d["mp_ok"]):
        k = int(np.argmin(np.abs(grid_t - d["t"][n])))
        if abs(grid_t[k] - d["t"][n]) > max_dt:
            continue
        c, i = d["cam"][n], int(d["i"][n])
        ci = camw[c]
        obs.append({"k": k, "cam": c, "i": i, "K": ci["K"][i], "R": ci["R"][i], "t": ci["t"][i], "k1": float(ci["k1"][i]),
                    "uv": d["mp52"][n].astype(np.float64), "iod": float(d["iod"][n])})
    return obs


def head_tracks(work: Path, role: str, emb: LandmarkEmbedding, L_rest: np.ndarray):
    """Per grid frame: R_k, t_k (rest -> posed landmarks, rigid upper face), neck joint J12_k, posed landmarks."""
    from .avatar_texture import load_bodies
    bodies, _ = load_bodies(work, head_fix=False)
    b = bodies[role]
    P = emb.points(b.verts.astype(np.float64))
    T = len(b.t)
    Rk, tk = np.zeros((T, 3, 3)), np.zeros((T, 3))
    from .face_common import kabsch
    rigid = np.r_[1:32]
    for k in range(T):
        Rk[k], tk[k] = kabsch(L_rest[rigid], P[k, rigid])
    return b.t, Rk, tk, b.joints[:, 12], P


def identity_offsets(work: Path, role: str, emb: LandmarkEmbedding) -> np.ndarray | None:
    """Portrait fit residuals lifted to the head's rest frame (z = 0): D (52,3) in metres, or None."""
    p = work / "avatar" / "face" / f"{role}_portrait_fit.npz"
    if not p.exists():
        return None
    z = np.load(p)
    R, s, res = z["R"], float(z["s"]), z["residual"]
    d_img = np.stack([res[:, 0], -res[:, 1], np.zeros(len(res))], 1) / s
    return d_img @ R          # R maps rest -> camera; rows: R^T applied


def fit_role(work: Path, role: str, cfg: ExprConfig, holdout: str | None = None, mode: str = "full",
             cache: dict | None = None) -> dict:
    """mode: 'neutral' (nothing fitted), 'head' (neck correction only), 'full' (neck + jaw + expression)."""
    import torch

    from .avatar_face import HeadModel
    cache = cache if cache is not None else {}
    if "head" not in cache:
        emb = LandmarkEmbedding()
        d = np.load(work / "smplx" / f"{role}.npz", allow_pickle=True)
        head = HeadModel(np.asarray(d["betas"], np.float32), cfg.n_expr)
        lfn = head.landmark_fn(emb)
        with torch.no_grad():
            L_rest = lfn(torch.zeros(cfg.n_expr), torch.zeros(3)).double().numpy()
        grid_t, Rk, tk, J12, Pk = head_tracks(work, role, emb, L_rest)
        D = identity_offsets(work, role, emb)
        cache.update(emb=emb, head=head, lfn=lfn, L_rest=L_rest, grid_t=grid_t, Rk=Rk, tk=tk, J12=J12, Pk=Pk,
                     D=np.zeros((52, 3)) if D is None else D, has_D=D is not None,
                     obs=observations(work, role, grid_t))
    c = cache
    T = len(c["grid_t"])
    obs_all = c["obs"]
    obs = [o for o in obs_all if o["cam"] != holdout]
    test = [o for o in obs_all if o["cam"] == holdout] if holdout else obs_all
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    E = cfg.n_expr
    e = torch.zeros((T, E), requires_grad=True, device=dev)
    e0 = torch.zeros(E, requires_grad=True, device=dev)
    jaw = torch.zeros((T, cfg.jaw_dims), requires_grad=True, device=dev)
    w = torch.zeros((T, 3), requires_grad=True, device=dev)
    Rk, tk, J12, D = f32(c["Rk"]), f32(c["tk"]), f32(c["J12"]), f32(c["D"])

    blfn = c.setdefault(f"blfn_{dev}", batched_landmarks(c["head"], c["emb"], dev))

    def landmarks(e_, jaw_, w_):
        """World landmarks for all frames (T,52,3)."""
        j3 = torch.cat([jaw_, torch.zeros((T, 3 - cfg.jaw_dims), device=dev)], 1) if cfg.jaw_dims < 3 else jaw_
        Lr = blfn(e_, j3) + D
        Pw = torch.einsum("tij,tlj->tli", Rk, Lr) + tk[:, None]
        Rw = rodrigues(torch, w_)
        return J12[:, None] + torch.einsum("tij,tlj->tli", Rw, Pw - J12[:, None])

    def batch(ol):
        return {"k": torch.as_tensor([o["k"] for o in ol], device=dev), "K": f32([o["K"] for o in ol]),
                "R": f32([o["R"] for o in ol]), "t": f32([o["t"] for o in ol]), "k1": f32([o["k1"] for o in ol]),
                "uv": f32([o["uv"] for o in ol]), "iod": f32([o["iod"] for o in ol])}

    def residuals(P, b):
        """(n,52) reprojection error in IOD units."""
        Xc = torch.einsum("nij,nlj->nli", b["R"], P[b["k"]]) + b["t"][:, None]
        z = Xc[..., 2].clamp(min=0.05)
        x, y = Xc[..., 0] / z, Xc[..., 1] / z
        dd = 1 + b["k1"][:, None] * (x * x + y * y)
        u = b["K"][:, 0, 0, None] * x * dd + b["K"][:, 0, 2, None]
        v = b["K"][:, 1, 1, None] * y * dd + b["K"][:, 1, 2, None]
        return torch.sqrt((u - b["uv"][..., 0]) ** 2 + (v - b["uv"][..., 1]) ** 2 + 1e-12) / b["iod"][:, None]

    params = [] if mode == "neutral" else ([w] if mode == "head" else [w, e, e0, jaw])
    t0 = time.time()
    if params and obs:
        opt = torch.optim.Adam(params, lr=cfg.lr)
        b = batch(obs)
        for it in range(cfg.iters):
            opt.zero_grad()
            P = landmarks(e, jaw, w)
            r = residuals(P, b)
            loss = (torch.log1p((r / cfg.cauchy_iod) ** 2) * cfg.cauchy_iod ** 2).sum() / len(obs)
            loss = loss + (cfg.expr_prior * (e - e0).pow(2).sum() + cfg.jaw_prior * jaw.pow(2).sum()
                           + cfg.neck_prior * w.pow(2).sum()) / T \
                + (cfg.smooth_expr * (e[1:] - e[:-1]).pow(2).sum() + cfg.smooth_jaw * (jaw[1:] - jaw[:-1]).pow(2).sum()
                   + cfg.smooth_neck * (w[1:] - w[:-1]).pow(2).sum()) / T + cfg.e0_prior * e0.pow(2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                jaw[:, 0].clamp_(0.0, 0.45)     # jaw opens (positive x), at most ~26 deg
    with torch.no_grad():
        P = landmarks(e, jaw, w)
        errs = []
        if test:
            rt = residuals(P, batch(test)).cpu().numpy()
            errs = [(o["k"], o["cam"], rt[n]) for n, o in enumerate(test)]
    n_obs = np.zeros(T, int)
    for o in obs:
        n_obs[o["k"]] += 1
    mouth_pos = [1 + (m - 17) for m in MOUTH]
    all_e = np.concatenate([r for _, _, r in errs]) if errs else np.array([np.nan])
    mouth_e = np.concatenate([r[mouth_pos] for _, _, r in errs]) if errs else np.array([np.nan])
    return {"expr": e.detach().cpu().numpy(), "e0": e0.detach().cpu().numpy(), "jaw": jaw.detach().cpu().numpy(),
            "neck_delta": w.detach().cpu().numpy(), "n_obs": n_obs, "t": c["grid_t"],
            "nme_all": float(np.median(all_e)), "nme_mouth": float(np.median(mouth_e)), "n_test_views": len(errs),
            "n_fit_views": len(obs), "frames_observed": int((n_obs > 0).sum()), "runtime_s": round(time.time() - t0, 1),
            "identity_offsets": bool(c["has_D"])}


def run(take: Take, cfg: ExprConfig | None = None, roles=ROLES, loo: bool = True) -> dict:
    cfg = cfg or ExprConfig()
    work = take.work
    out_dir = work / "face"
    rep = {"take": take.name, "config": asdict(cfg), "roles": {}}
    for r in roles:
        cache: dict = {}
        res = {m: fit_role(work, r, cfg, mode=m, cache=cache) for m in ("neutral", "head", "full")}
        full = res["full"]
        conf = np.clip(np.convolve((full["n_obs"] > 0).astype(float), np.ones(9) / 9, "same") * 2, 0, 1)
        jaw3 = np.zeros((len(full["t"]), 3))
        jaw3[:, :cfg.jaw_dims] = full["jaw"]
        d = np.load(work / "smplx" / f"{r}.npz", allow_pickle=True)
        from .avatar_head import neck_pose_with
        np.savez(out_dir / f"expr_{r}.npz", t=full["t"], expression=full["expr"], expression_rest=full["e0"],
                 jaw_pose=jaw3, neck_delta_rotvec=full["neck_delta"], neck_pose=neck_pose_with(dict(d), full["neck_delta"]),
                 n_obs=full["n_obs"], confidence=conf, n_expr=np.array(cfg.n_expr),
                 note=np.array("SMPL-X expression (first n_expr of 100) + jaw_pose; neck_pose = body_pose[:, 33:36] "
                               "with the face-landmark neck correction; confidence 0 = no observation (relaxed to "
                               "expression_rest)"))
        rr = {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()
                  if k in ("nme_all", "nme_mouth", "n_test_views", "n_fit_views", "frames_observed", "runtime_s",
                           "identity_offsets")} for m, x in res.items()}
        rr["jaw_open_deg_p50_p95"] = [round(float(np.degrees(np.percentile(full["jaw"][:, 0], q))), 1) for q in (50, 95)]
        rr["expr_norm_p50_p95"] = [round(float(np.percentile(np.linalg.norm(full["expr"], axis=1), q)), 2) for q in (50, 95)]
        rr["e0"] = np.round(full["e0"], 2).tolist()
        rr["neck_correction_deg_p50"] = round(float(np.degrees(np.median(np.linalg.norm(full["neck_delta"], axis=1)))), 1)
        if loo:
            cams = sorted({o["cam"] for o in cache["obs"]})
            lo = {}
            for c in cams:
                n_c = sum(o["cam"] == c for o in cache["obs"])
                if n_c < cfg.loo_min_views:
                    continue
                lo[c] = {m: {k: round(v, 4) for k, v in fit_role(work, r, cfg, holdout=c, mode=m, cache=cache).items()
                             if k in ("nme_all", "nme_mouth")} for m in ("neutral", "head", "full")}
                lo[c]["views"] = n_c
            rr["leave_one_camera_out"] = lo
        rep["roles"][r] = rr
        print(f"[face_expr] {r}: {json.dumps(rr)}", flush=True)
    (out_dir / "expr.json").write_text(json.dumps(rep, indent=1))
    rev = work / "review"
    rep["review"] = [str(_curves(work, roles, rev / "face_expr_research_curves.png", take.name))]
    try:
        rep["review"].append(str(review_frames(take, roles, rev / "face_expr_research.jpg")))
    except FileNotFoundError as ex:
        print(f"[face_expr] review frames skipped: {ex}")
    (out_dir / "expr.json").write_text(json.dumps(rep, indent=1))
    return rep


def _curves(work: Path, roles, path: Path, take: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(roles), 1, figsize=(14, 3.6 * len(roles)), squeeze=False)
    for n, r in enumerate(roles):
        z = np.load(work / "face" / f"expr_{r}.npz")
        a = ax[n, 0]
        a.plot(z["t"], np.degrees(z["jaw_pose"][:, 0]), label="jaw opening (deg)", color="C3")
        for k in range(min(4, z["expression"].shape[1])):
            a.plot(z["t"], z["expression"][:, k] * 5, lw=0.8, label=f"expression {k} (x5)")
        a.fill_between(z["t"], -10, -10 + 3 * z["confidence"], color="k", alpha=0.3, label="observed (confidence)")
        a.set_title(f"{LABEL[r]}: fitted jaw / expression (grey bar = frames with a usable MediaPipe face)", fontsize=9)
        a.legend(fontsize=7, ncol=4)
        a.set_xlabel("reference time (s)")
    fig.suptitle(f"{take}: per-frame face fit from the video landmarks")
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path


def posed_vertices(work: Path, role: str, with_expr: bool, face_dir: Path | None = None):
    """(T,V,3) vertices of the dancer with (or without) the fitted jaw / expression / neck correction."""
    import smplx
    import torch

    from .mamma_dyn import SMPLX_MODELS
    d = {k: v for k, v in np.load(work / "smplx" / f"{role}.npz", allow_pickle=True).items()}
    T = len(d["t"])
    z = np.load((face_dir or work / "face") / f"expr_{role}.npz") if with_expr else None
    ne = int(z["n_expr"]) if z is not None else 10
    m = smplx.create(str(SMPLX_MODELS), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                     flat_hand_mean=True, num_betas=int(d.get("num_betas", 16)), num_expression_coeffs=ne, batch_size=T)
    bp = np.asarray(d["body_pose"], float).copy()
    jaw = np.zeros((T, 3))
    ex = np.zeros((T, ne))
    if z is not None:
        bp[:, 33:36] = z["neck_pose"]
        jaw, ex = z["jaw_pose"], z["expression"]
    f = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float32)
    with torch.no_grad():
        o = m(betas=f(np.asarray(d["betas"]).reshape(-1))[None].expand(T, -1), global_orient=f(d["global_orient"]),
              body_pose=f(bp), jaw_pose=f(jaw), leye_pose=f(np.zeros((T, 3))), reye_pose=f(np.zeros((T, 3))),
              left_hand_pose=f(d["left_hand_pose"]), right_hand_pose=f(d["right_hand_pose"]), transl=f(d["transl"]),
              expression=f(ex))
    return np.asarray(d["t"], float), o.vertices.numpy()


def review_frames(take: Take, roles, path: Path, n: int = 4, tile: int = 240) -> Path:
    """Video | avatar (portrait face, neutral) | avatar (portrait face + fitted expression), frames where the fit
    moved the face most and the face was observed."""
    import cv2
    import torch

    from .avatar_texture import _Geo, render
    from .avatar_uv import build_atlas
    from .face_common import crop_pad, label
    from .mamma_dyn import SMPLX_MODELS
    work = take.work
    atlas = build_atlas(SMPLX_MODELS, 2048)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geo = _Geo(atlas, dev)
    cols = {}
    for r in ROLES:
        p = work / "avatar" / "face" / f"{r}_albedo_face.png"
        if not p.exists():
            p = work / "avatar" / f"{r}_albedo.png"
        im = cv2.imread(str(p))[:, :, ::-1].astype(np.float32) / 255
        cols[r] = torch.as_tensor(im[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]], device=dev)
    V = {r: {flag: posed_vertices(work, r, flag)[1] for flag in (False, True)} for r in ROLES}
    sv = np.load(work / "face" / "survey.npz")
    cams = sorted({k.split("_")[1] for k in sv.files if k.startswith("lead_") and k.split("_")[1].isdigit()})
    camw = cameras_world(work, cams)
    emb = LandmarkEmbedding()
    rows = []
    for r in roles:
        z = np.load(work / "face" / f"expr_{r}.npz")
        move = np.linalg.norm(emb.points(V[r][True]) - emb.points(V[r][False]), axis=2).max(1)
        cand = []
        for c in cams:
            if f"{r}_{c}_base" not in sv.files:
                continue
            for q in np.flatnonzero(sv[f"{r}_{c}_base"] & (sv[f"{r}_{c}_iod"] >= 15)):
                tq = float(sv[f"{r}_{c}_t"][q])
                k = int(np.argmin(np.abs(z["t"] - tq)))
                if abs(z["t"][k] - tq) < 0.02 and z["n_obs"][k] > 0:
                    cand.append((move[k] * sv[f"{r}_{c}_iod"][q], c, int(sv[f"{r}_{c}_i"][q]), k, sv[f"{r}_{c}_lmk"][q]))
        cand.sort(key=lambda x: -x[0])
        picked = []
        for cd in cand:
            if all(abs(cd[3] - p[3]) > 10 for p in picked):
                picked.append(cd)
            if len(picked) == n:
                break
        tiles = []
        for sc, c, i, k, lm in picked:
            img = cv2.imread(str(work / "frames" / c / f"{i:06d}.jpg"))
            H, W = img.shape[:2]
            ci = camw[c]
            lo, hi = lm.min(0), lm.max(0)
            side = max(hi - lo) * 2.4
            cx = (lo + hi) / 2
            box = (int(cx[0] - side / 2), int(cx[1] - side / 2), int(cx[0] + side / 2), int(cx[1] + side / 2))
            trip = [crop_pad(img, box, tile)]
            for flag in (False, True):
                posed = {rr: V[rr][flag if rr == r else False][k] for rr in ROLES}
                ren, rid = render(torch, geo, posed, cols, ci["K"][i], ci["R"][i], ci["t"][i], float(ci["k1"][i]), H, W, dev)
                ren8 = (np.clip(ren, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
                trip.append(crop_pad(np.where((rid >= 0)[..., None], ren8, 40).astype(np.uint8), box, tile))
            label(trip[0], f"cam {c} #{i} t={z['t'][k]:.2f}", (4, 14), scale=0.42)
            label(trip[1], "neutral face", (4, 14), scale=0.42)
            label(trip[2], f"fitted jaw {np.degrees(z['jaw_pose'][k, 0]):.0f}deg |e| {np.linalg.norm(z['expression'][k]):.1f}",
                  (4, 14), scale=0.42)
            tiles += trip + [np.zeros((tile, 4, 3), np.uint8)]
        if tiles:
            row = np.concatenate(tiles, 1)
            hs = np.full((22, row.shape[1], 3), 20, np.uint8)
            label(hs, LABEL[r], (6, 16), scale=0.5)
            rows.append(np.concatenate([hs, row], 0))
    if not rows:
        raise FileNotFoundError("no observed frames to show")
    w = max(rw.shape[1] for rw in rows)
    sheet = np.concatenate([np.pad(rw, ((0, 4), (0, w - rw.shape[1]), (0, 0))) for rw in rows], 0)
    top = np.full((28, w, 3), 10, np.uint8)
    label(top, f"{take.name}: photo | avatar neutral face | avatar with the fitted jaw + expression + neck correction",
          (6, 19), scale=0.5)
    cv2.imwrite(str(path), np.concatenate([top, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


# ================================================================================================================
# Final per-frame face for the avatars: work/<take>/smplx_face/{lead,follow}.npz (python -m dancecap.face_expr <take>)
# ================================================================================================================
#
# Shape, not position: the held-out errors of the research fit above are dominated by head pose / position error
# (the body fit's head is good to ~4-5 px, which is 15-25 % of a 20-30 px face), so mouth landmarks cannot be compared
# in absolute image position. Every observation is therefore first aligned to the model by the 2D similarity that
# best maps the model's RIGID anchors (nose bridge + tip, eye corners) onto the detected ones; only the remaining
# shape differences (mouth, chin, brows) drive the expression. No neck correction is fitted (never adopted).
#
# Honest hybrid (research verdict 2026-10-06):
#   expression_rest e0 / jaw_rest   fitted from every confident view of the dancer, applied all the time
#   captured layer                  per-frame jaw opening + expression, Gaussian-smoothed (smooth_s), deviation from
#                                   e0 clamped (max_dev), jaw clamped to [0, jaw_max_deg]
#   confidence c_k                  max over the frame's views of q = size (IOD min_iod -> full_iod px) x frontal
#                                   (angle full_angle -> max_angle deg) x agreement (MediaPipe within max_raw_nme IOD
#                                   of the projected SMPL-X face, else it locked onto a hand / the partner), held
#                                   +-conf_hold_s and smoothed
#   final                           e = e0 + c (e_captured - e0), jaw = jaw_rest + c (jaw_captured - jaw_rest)
#   eyelid (stylisation)            procedural blinks every blink_every_s (seeded per take + dancer); weights for the
#                                   FLAME eyelid blendshapes moved onto SMPL-X (face_blendshapes.npz); MAMMA's eyes and
#                                   these blinks are NOT captured
# Checks: leave-one-camera-out on the aligned shape (mouth NME in % IOD, mouth-opening and mouth-width correlation
# with the held-out detections) for neutral / rest / per-frame / final -> smplx_face/report.json.

RIGID_ANCHORS = (27, 28, 29, 30, 31, 33, 35, 36, 39, 42, 45)
IBUG_INNER_LIPS = tuple(range(60, 68))
BROWS = tuple(range(17, 27))
EYELIDS = (37, 38, 40, 41, 43, 44, 46, 47)
FLAME_IDS = ("models", "flame", "SMPL-X__FLAME_vertex_ids.npy")
EYELID_ASSETS = ("external", "smirk", "assets")


@dataclass
class FaceConfig:
    n_expr: int = 10
    min_iod: float = 12.0          # view quality ramps from min_iod to full_iod px (outer eye corners)
    full_iod: float = 20.0
    full_angle: float = 35.0       # ... and from full_angle to max_angle deg off frontal
    max_angle: float = 60.0
    max_raw_nme: float = 0.25      # MediaPipe vs projected SMPL-X face (IOD): beyond = wrong face / a hand
    cauchy_iod: float = 0.06
    w_mouth: float = 1.0
    w_inner_lips: float = 1.0      # inner lip contour (60-67): sets the lip gap
    w_brow: float = 0.5
    w_anchor: float = 0.3
    expr_prior: float = 0.05       # |e_k - e0|^2
    e0_prior: float = 0.02         # |e0|^2
    jaw_prior: float = 30.0        # (a_k - a0)^2, rad
    smooth_expr: float = 2.0       # |e_k+1 - e_k|^2 (30 Hz)
    smooth_jaw: float = 200.0
    iters: int = 300
    lr: float = 0.02
    smooth_s: float = 0.10         # Gaussian sigma of the captured layer (window ~ +-2 sigma = 0.4 s)
    conf_hold_s: float = 0.10
    jaw_max_deg: float = 6.0
    max_dev: float = 1.5           # |e - e0| clamp (SMPL-X expression units)
    blink_every_s: tuple = (2.5, 4.5)
    blink_close_s: float = 0.075
    blink_hold_s: float = 0.03
    blink_open_s: float = 0.15
    seed: int = 7
    gaze_stylise: bool = True          # STYLISATION (not capture): when the head pitches down, the eyes look further
    eye_pitch_gain: float = 0.4        # down (gain x pitch beyond eye_pitch_start_deg, <= eye_pitch_max_deg) and the
    eye_pitch_start_deg: float = 5.0   # upper lids lower (eyelid weight up to droop_max from droop_start_deg over
    eye_pitch_max_deg: float = 20.0    # droop_span_deg) - MAMMA's eyes stay at ~1 deg, so a lowered head stared
    droop_start_deg: float = 10.0
    droop_span_deg: float = 40.0
    droop_max: float = 0.35
    loo_min_views: int = 5
    block_s: float = 0.2           # temporal hold-out blocks (alternate blocks held out, 2 folds)
    adopt_rest_gain_pp: float = 0.3    # model selection on the hold-out checks (adoption(); mouth NME, % IOD)
    adopt_min_gain_pp: float = 0.3
    adopt_max_cam_loss_pp: float = 0.3
    adopt_open_rel: float = 1.5        # a layer may raise the held-out mouth-opening error (lip gap / mouth width) to at
    adopt_open_floor: float = 0.005    # most max(rel x the simpler model's error, that error + floor) (2026-10-07: the
                                       # absolute 0.05 let a rest smile that tripled the error through)
    w_open: float = 0.0                # optional lip-gap term (model opening vs each detection's, Cauchy open_cauchy);
    open_cauchy: float = 0.03          # off: on larissa_kadu it made held-out openings worse (bimodal: open / closed smiles)
    w_closed: float = 0.01             # 'rest_closed' candidate: the resting expression keeps the neutral lip gap
                                       # (per mm of 3D gap change) - a smile of the mouth corners without parted lips
    covis_min_views: int = 5           # per-frame capture is scored on held-out views whose frame another camera saw
    beard_roles: tuple = ("lead",)     # chin + outer lower lip down-weighted (face_look: Kadu's chin beard)
    identity_offsets: bool = False     # portrait residuals as landmark offsets: off - the avatar mesh does not carry
                                       # them (only the texture warp does), and the follow's portrait residual holds
                                       # her open-lipped smile; without them e0 absorbs the visible identity + rest face
    device: str = "cpu"            # tiny problem: CPU keeps the shared GPU free
    threads: int = 1               # torch CPU threads: >1 thrashes when other jobs load the 4-CPU PC (30x slower)


def _ibug_pos(ids) -> np.ndarray:
    """iBUG ids -> positions in LMK52 (chin 8 first, then 17..67)."""
    return np.array([0 if k == 8 else 1 + (k - 17) for k in ids])


def point_weights(cfg: FaceConfig, role: str = "") -> np.ndarray:
    w = np.zeros(52)
    w[_ibug_pos(RIGID_ANCHORS)] = cfg.w_anchor
    w[_ibug_pos(BROWS)] = cfg.w_brow
    w[_ibug_pos((8,) + MOUTH)] = cfg.w_mouth
    w[_ibug_pos(IBUG_INNER_LIPS)] = cfg.w_inner_lips
    if role in cfg.beard_roles:          # the beard drags MediaPipe's chin / outer lower lip down (reads as an open jaw)
        w[_ibug_pos((8,))] = 0.0
        w[_ibug_pos((55, 56, 57, 58, 59))] = 0.5 * cfg.w_mouth
    return w


_OPEN_PAIRS = ((62, 66), (61, 67), (63, 65))


def mouth_open_torch(torch, P):
    """face_eval._mouth_open for torch (..., 52, 2): mean inner-lip gap / mouth width."""
    gap = sum((P[..., a, :] - P[..., b, :]).norm(dim=-1) for a, b in
              ((int(_ibug_pos((u,))[0]), int(_ibug_pos((v,))[0])) for u, v in _OPEN_PAIRS)) / 3
    i48, i54 = (int(x) for x in _ibug_pos((48, 54)))
    return gap / (P[..., i48, :] - P[..., i54, :]).norm(dim=-1).clamp(min=1e-6)


def sim_align(torch, P, U, wa):
    """P, U (n, L, 2), wa (L,) anchor weights: P mapped onto U by the least-squares 2D similarity of the anchors."""
    wn = wa / wa.sum()
    mp = (P * wn[None, :, None]).sum(1, keepdim=True)
    mu = (U * wn[None, :, None]).sum(1, keepdim=True)
    Pc, Uc = P - mp, U - mu
    den = (wn[None] * (Pc ** 2).sum(-1)).sum(1).clamp(min=1e-9)
    a = (wn[None] * (Pc[..., 0] * Uc[..., 0] + Pc[..., 1] * Uc[..., 1])).sum(1) / den
    b = (wn[None] * (Pc[..., 0] * Uc[..., 1] - Pc[..., 1] * Uc[..., 0])).sum(1) / den
    x = a[:, None] * Pc[..., 0] - b[:, None] * Pc[..., 1]
    y = b[:, None] * Pc[..., 0] + a[:, None] * Pc[..., 1]
    return torch.stack([x, y], -1) + mu


def view_quality(iod, angle, raw_nme, cfg: FaceConfig) -> np.ndarray:
    q_iod = np.clip((np.asarray(iod) - cfg.min_iod) / (cfg.full_iod - cfg.min_iod), 0, 1)
    q_ang = np.clip((cfg.max_angle - np.asarray(angle)) / (cfg.max_angle - cfg.full_angle), 0, 1)
    agree = np.nan_to_num(np.asarray(raw_nme, float), nan=9.0) < cfg.max_raw_nme
    return q_iod * q_ang * agree


def confidence_curve(k: np.ndarray, q: np.ndarray, T: int, fps: float, cfg: FaceConfig) -> np.ndarray:
    """Per-frame confidence from the views (frame index k, quality q): max per frame, held and smoothed."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    c = np.zeros(T)
    np.maximum.at(c, np.asarray(k, int), np.asarray(q, float))
    hold = max(1, int(round(cfg.conf_hold_s * fps)))
    c = maximum_filter1d(c, 2 * hold + 1, mode="constant")
    return np.clip(gaussian_filter1d(c, hold, mode="constant"), 0, 1)


def blend_final(e_cap, jaw_cap, e0, jaw0, conf, fps: float, cfg: FaceConfig):
    """Captured layer (T,E) / (T,) rad -> smoothed, clamped, confidence-blended expression (T,E) and jaw (T,) rad."""
    from scipy.ndimage import gaussian_filter1d
    sig = max(cfg.smooth_s * fps, 1e-3)
    e_sm = gaussian_filter1d(np.asarray(e_cap, float), sig, axis=0, mode="nearest")
    j_sm = gaussian_filter1d(np.asarray(jaw_cap, float), sig, mode="nearest")
    dev = e_sm - e0[None]
    n = np.linalg.norm(dev, axis=1, keepdims=True)
    dev *= np.minimum(1.0, cfg.max_dev / np.maximum(n, 1e-9))
    c = np.asarray(conf, float)
    jmax = np.radians(cfg.jaw_max_deg)
    j0 = float(np.clip(jaw0, 0, jmax))
    expr = e0[None] + c[:, None] * dev
    jaw = np.clip(j0 + c * (np.clip(j_sm, 0, jmax) - j0), 0, jmax)
    return expr, jaw, e_sm, j_sm


def gaze_stylisation(Rk: np.ndarray, fps: float, cfg: FaceConfig) -> dict:
    """Stylised eyes from the head's pitch (Rk: (T,3,3) rest -> posed head rotation; rest SMPL-X looks along +z, y up):
    eye pitch (rad, + = down about the head's x axis) and upper-lid droop (0..droop_max), smoothed 0.1 s."""
    from scipy.ndimage import gaussian_filter1d
    fwd = np.asarray(Rk, float)[:, :, 2]
    down = np.degrees(np.arcsin(np.clip(-fwd[:, 1], -1, 1)))
    down = gaussian_filter1d(down, max(0.1 * fps, 1e-3), mode="nearest")
    pitch = np.clip(cfg.eye_pitch_gain * (down - cfg.eye_pitch_start_deg), 0, cfg.eye_pitch_max_deg)
    droop = np.clip((down - cfg.droop_start_deg) / cfg.droop_span_deg, 0, 1) * cfg.droop_max
    return {"head_pitch_down_deg": down, "eye_pitch": np.radians(pitch), "droop": droop}


def blink_schedule(t: np.ndarray, cfg: FaceConfig, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Procedural blinks (stylisation, not capture): weights (T,) in [0, 1] on the grid t, blink start times."""
    import zlib
    rng = np.random.default_rng(cfg.seed + zlib.crc32(key.encode()) % 100000)
    lo, hi = cfg.blink_every_s
    starts = []
    s = float(t[0]) + rng.uniform(0.3, hi)
    while s < t[-1]:
        starts.append(s)
        s += rng.uniform(lo, hi)
    w = np.zeros(len(t))
    c, h, o = cfg.blink_close_s, cfg.blink_hold_s, cfg.blink_open_s
    for s in starts:
        x = np.asarray(t, float) - s
        up = np.clip(x / c, 0, 1)
        down = np.clip(1 - (x - c - h) / o, 0, 1)
        prof = np.where(x < 0, 0.0, np.where(x < c, 0.5 - 0.5 * np.cos(np.pi * up),
                                             np.where(x < c + h, 1.0, 0.5 - 0.5 * np.cos(np.pi * down))))
        w = np.maximum(w, prof)
    return w, np.array(starts)


_LOAD_CACHE: dict = {}


def prepare(work: Path, role: str, cfg: FaceConfig) -> dict:
    """Head tracks, landmark model and the dancer's MediaPipe views (vectorised) for the aligned fit."""
    import torch

    from .avatar_face import HeadModel
    from .face_eval import load
    emb = LandmarkEmbedding()
    d = np.load(work / "smplx" / f"{role}.npz", allow_pickle=True)
    head = HeadModel(np.asarray(d["betas"], np.float32), cfg.n_expr)
    lfn = head.landmark_fn(emb)
    with torch.no_grad():
        L_rest = lfn(torch.zeros(cfg.n_expr), torch.zeros(3)).double().numpy()
    grid_t, Rk, tk, _, _ = head_tracks(work, role, emb, L_rest)
    D = identity_offsets(work, role, emb) if cfg.identity_offsets else None
    key = str(work.resolve())
    if key not in _LOAD_CACHE:
        _LOAD_CACHE.clear()
        _LOAD_CACHE[key] = load(work)
    fd = _LOAD_CACHE[key]
    sel = np.flatnonzero((fd["role"] == role) & fd["base"] & fd["mp_ok"])
    camw = cameras_world(work, sorted(set(fd["cam"][sel]))) if len(sel) else {}
    rows = []
    for n in sel:
        k = int(np.argmin(np.abs(grid_t - fd["t"][n])))
        if abs(grid_t[k] - fd["t"][n]) > 0.02:
            continue
        rows.append((n, k))
    n_ = np.array([r[0] for r in rows], int)
    k_ = np.array([r[1] for r in rows], int)
    raw = np.nanmean(np.linalg.norm(fd["model52"][n_] - fd["mp52"][n_], axis=-1), -1) / np.maximum(fd["iod"][n_], 1) \
        if len(n_) else np.zeros(0)
    cam = fd["cam"][n_]
    ii = fd["i"][n_].astype(int)
    obs = {"k": k_, "cam": cam, "i": ii, "t": fd["t"][n_], "uv": fd["mp52"][n_].astype(np.float64),
           "iod": fd["iod"][n_].astype(float), "angle": fd["angle"][n_].astype(float), "raw_nme": raw,
           "K": np.array([camw[c]["K"][i] for c, i in zip(cam, ii)]).reshape(-1, 3, 3),
           "R": np.array([camw[c]["R"][i] for c, i in zip(cam, ii)]).reshape(-1, 3, 3),
           "tc": np.array([camw[c]["t"][i] for c, i in zip(cam, ii)]).reshape(-1, 3),
           "k1": np.array([float(camw[c]["k1"][i]) for c, i in zip(cam, ii)])}
    obs["q"] = view_quality(obs["iod"], obs["angle"], obs["raw_nme"], cfg)
    dt = np.diff(grid_t)
    return {"role": role, "emb": emb, "head": head, "grid_t": grid_t, "fps": float(1 / np.median(dt)) if len(dt) else 30.0,
            "Rk": Rk, "tk": tk, "D": np.zeros((52, 3)) if D is None else D, "has_D": D is not None, "obs": obs,
            "smplx": {k: d[k] for k in d.files}}


def lip_gap_3d(torch, L):
    """(T,52,3) landmarks -> (T,) mean 3D inner-lip gap (iBUG 62-66, 61-67, 63-65), metres."""
    return sum((L[:, int(_ibug_pos((a,))[0])] - L[:, int(_ibug_pos((b,))[0])]).norm(dim=-1) for a, b in _OPEN_PAIRS) / 3


def fit_aligned(prep: dict, cfg: FaceConfig, train: np.ndarray | None = None, mode: str = "frame") -> dict:
    """mode 'neutral' (e = 0, jaw = 0), 'rest' (constant e0, jaw0), 'rest_closed' (constant e0, jaw0 keeping the
    neutral face's 3D lip gap), 'frame' (per-frame e_k, jaw_k + e0, jaw0).
    train: boolean mask over prep['obs'] (default: every view with q > 0)."""
    import torch
    torch.manual_seed(0)
    obs = prep["obs"]
    T, E = len(prep["grid_t"]), cfg.n_expr
    dev = cfg.device
    f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    blfn = prep.setdefault(f"_blfn_{dev}", batched_landmarks(prep["head"], prep["emb"], dev))
    Rk, tk, D = f32(prep["Rk"]), f32(prep["tk"]), f32(prep["D"])
    tr = (obs["q"] > 0) if train is None else (np.asarray(train, bool) & (obs["q"] > 0))
    e = torch.zeros((T, E), device=dev, requires_grad=True)
    a = torch.zeros(T, device=dev, requires_grad=True)
    e0 = torch.zeros(E, device=dev, requires_grad=True)
    a0 = torch.zeros((), device=dev, requires_grad=True)
    wpt = f32(point_weights(cfg, prep["role"]))
    wanc = torch.zeros(52, device=dev)
    wanc[torch.as_tensor(_ibug_pos(RIGID_ANCHORS))] = 1.0

    def frames(e_, a_, e0_, a0_):
        if mode == "neutral":
            return torch.zeros((T, E), device=dev), torch.zeros(T, device=dev)
        if mode in ("rest", "rest_closed"):
            return e0_[None].expand(T, E), a0_.expand(T)
        return e_, a_

    def world(ee, aa):
        jaw3 = torch.stack([aa, torch.zeros_like(aa), torch.zeros_like(aa)], 1)
        Lr = blfn(ee, jaw3) + D
        return torch.einsum("tij,tlj->tli", Rk, Lr) + tk[:, None]

    pcache: dict = {}

    def proj(P, idx):
        key = np.asarray(idx).tobytes()
        if key not in pcache:
            pcache.clear()
            pcache[key] = ({k: f32(obs[k][idx]) for k in ("K", "R", "tc", "k1")}, torch.as_tensor(obs["k"][idx], device=dev))
        b, kk = pcache[key]
        Xc = torch.einsum("nij,nlj->nli", b["R"], P[kk]) + b["tc"][:, None]
        z = Xc[..., 2].clamp(min=0.05)
        x, y = Xc[..., 0] / z, Xc[..., 1] / z
        dd = 1 + b["k1"][:, None] * (x * x + y * y)
        u = b["K"][:, 0, 0, None] * x * dd + b["K"][:, 0, 2, None]
        v = b["K"][:, 1, 1, None] * y * dd + b["K"][:, 1, 2, None]
        return torch.stack([u, v], -1)

    idx_tr = np.flatnonzero(tr)
    params = {"neutral": [], "rest": [e0, a0], "rest_closed": [e0, a0], "frame": [e, a, e0, a0]}[mode]
    if mode == "rest_closed":
        with torch.no_grad():
            gap_neutral = lip_gap_3d(torch, blfn(torch.zeros((1, E), device=dev), torch.zeros((1, 3), device=dev)))
    t0 = time.time()
    if params and len(idx_tr):
        U = f32(obs["uv"][idx_tr])
        iod = f32(obs["iod"][idx_tr])
        qn = f32(obs["q"][idx_tr])
        open_d = mouth_open_torch(torch, U)
        opt = torch.optim.Adam(params, lr=cfg.lr)
        for _ in range(cfg.iters):
            opt.zero_grad()
            ee, aa = frames(e, a, e0, a0)
            Pa = sim_align(torch, proj(world(ee, aa), idx_tr), U, wanc)
            r = (Pa - U).norm(dim=-1) / iod[:, None]
            rho = torch.log1p((r / cfg.cauchy_iod) ** 2) * cfg.cauchy_iod ** 2
            loss = (qn[:, None] * wpt[None] * rho).sum() / (qn.sum() * wpt.sum())
            if cfg.w_open > 0:
                do = (mouth_open_torch(torch, Pa) - open_d) / cfg.open_cauchy
                loss = loss + cfg.w_open * (qn * torch.log1p(do ** 2)).sum() / qn.sum() * cfg.open_cauchy ** 2
            if mode == "rest_closed":
                j3 = torch.stack([a0, torch.zeros_like(a0), torch.zeros_like(a0)])[None]
                loss = loss + cfg.w_closed * (((lip_gap_3d(torch, blfn(e0[None], j3)) - gap_neutral) / 0.001) ** 2).sum()
            loss = loss + cfg.e0_prior * e0.pow(2).sum() / 100
            if mode == "frame":
                loss = loss + (cfg.expr_prior * (e - e0).pow(2).sum() + cfg.jaw_prior * (a - a0).pow(2).sum()
                               + cfg.smooth_expr * (e[1:] - e[:-1]).pow(2).sum()
                               + cfg.smooth_jaw * (a[1:] - a[:-1]).pow(2).sum()) / T / 100
            loss.backward()
            opt.step()
            with torch.no_grad():
                a.clamp_(0.0, 0.45)
                a0.clamp_(0.0, 0.45)
    with torch.no_grad():
        ee, aa = frames(e, a, e0, a0)
    return {"expr": ee.detach().cpu().numpy().astype(np.float64), "jaw": aa.detach().cpu().numpy().astype(np.float64),
            "e0": e0.detach().cpu().numpy().astype(np.float64), "jaw0": float(a0.detach()), "n_train": int(len(idx_tr)),
            "runtime_s": round(time.time() - t0, 1), "_world": world, "_proj": proj, "_wanc": wanc}


def aligned_errors(prep: dict, cfg: FaceConfig, expr: np.ndarray, jaw: np.ndarray, idx: np.ndarray, world=None,
                   proj=None) -> dict:
    """Aligned shape errors of views idx for given per-frame expression (T,E) and jaw opening (T,) rad."""
    import torch

    from .face_eval import _mouth_open
    if len(idx) == 0:
        return {"n": 0}
    obs = prep["obs"]
    f32 = lambda a_: torch.tensor(np.asarray(a_), dtype=torch.float32)
    if world is None:
        w = fit_aligned(prep, cfg, train=np.zeros(len(obs["k"]), bool), mode="neutral")
        world, proj = w["_world"], w["_proj"]
    wanc = torch.zeros(52)
    wanc[torch.as_tensor(_ibug_pos(RIGID_ANCHORS))] = 1.0
    with torch.no_grad():
        U = f32(obs["uv"][idx])
        Pa = sim_align(torch, proj(world(f32(expr), f32(jaw)), idx), U, wanc).numpy()
    U = U.numpy()
    iod = obs["iod"][idx]
    mpos = _ibug_pos(MOUTH)
    nme_mouth = np.linalg.norm(Pa[:, mpos] - U[:, mpos], axis=-1).mean(1) / iod
    open_p, open_d = _mouth_open(Pa), _mouth_open(U)
    i48, i54, i36, i45 = _ibug_pos((48, 54, 36, 45))
    width = lambda X: np.linalg.norm(X[:, i48] - X[:, i54], axis=-1) / np.linalg.norm(X[:, i36] - X[:, i45], axis=-1)
    wid_p, wid_d = width(Pa), width(U)
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 1e-9 and np.std(y) > 1e-9 else None
    return {"n": int(len(idx)), "nme_mouth_med": float(np.median(nme_mouth)), "nme_mouth_mean": float(np.mean(nme_mouth)),
            "open_corr": corr(open_p, open_d), "width_corr": corr(wid_p, wid_d),
            "open_bias": float(np.median(open_p - open_d)), "width_bias": float(np.median(wid_p - wid_d)),
            "_nme": nme_mouth, "_open": (open_p, open_d), "_width": (wid_p, wid_d)}


def holdout_check(prep: dict, cfg: FaceConfig, splits: list[tuple[str, np.ndarray, np.ndarray]]) -> dict:
    """Score views held out of the fit: per split (name, train mask, test mask) the neutral face, the resting
    expression, the raw per-frame fit, the final face (rest + confidence-gated captured layer) and the final face
    with the jaw kept at rest ('final_expr')."""
    obs = prep["obs"]
    T = len(prep["grid_t"])
    jmax = np.radians(cfg.jaw_max_deg)
    modes = ("neutral", "rest", "rest_closed", "frame", "final", "final_expr")
    new = lambda: {m: {"nme": [], "open": [[], []], "width": [[], []]} for m in modes}
    out, pooled, pooled_cv = {}, new(), new()
    for name, train, test_m in splits:
        test = np.flatnonzero(test_m)
        if len(test) < cfg.loo_min_views or not train.any():
            continue
        fr = fit_aligned(prep, cfg, train, "frame")
        rs = fit_aligned(prep, cfg, train, "rest")
        rc = fit_aligned(prep, cfg, train, "rest_closed")
        conf = confidence_curve(obs["k"][train], obs["q"][train], T, prep["fps"], cfg)
        ef, jf, _, _ = blend_final(fr["expr"], fr["jaw"], fr["e0"], fr["jaw0"], conf, prep["fps"], cfg)
        jr = np.full(T, float(np.clip(fr["jaw0"], 0, jmax)))
        # 'rest' is exactly what build() ships when only the rest layer is adopted: the rest-mode fit's e0 / jaw0
        cands = {"neutral": (np.zeros_like(fr["expr"]), np.zeros(T)),
                 "rest": (rs["expr"], np.clip(rs["jaw"], 0, jmax)),
                 "rest_closed": (rc["expr"], np.clip(rc["jaw"], 0, jmax)),
                 "frame": (fr["expr"], fr["jaw"]), "final": (ef, jf), "final_expr": (ef, jr)}
        # co-visible test views: another camera saw the same frame in the training set, so the per-frame layer had
        # direct evidence for that instant (all other held-out views only test temporal interpolation)
        cv = np.isin(obs["k"][test], obs["k"][train])
        res = {}
        for m, (ee, jj) in cands.items():
            er = aligned_errors(prep, cfg, ee, jj, test, fr["_world"], fr["_proj"])
            res[m] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in er.items() if not k.startswith("_")}
            for pl, sel in ((pooled, np.ones(len(test), bool)), (pooled_cv, cv)):
                pl[m]["nme"].append(er["_nme"][sel])
                for key in ("open", "width"):
                    pl[m][key][0].append(er[f"_{key}"][0][sel])
                    pl[m][key][1].append(er[f"_{key}"][1][sel])
        res["views"] = int(len(test))
        res["covisible_views"] = int(cv.sum())
        out[name] = res

    def summarise(pl, min_views):
        summ = {}
        for m, p in pl.items():
            if not p["nme"]:
                continue
            nme = np.concatenate(p["nme"])
            if len(nme) < min_views:
                continue
            op = [np.concatenate(x) for x in p["open"]]
            wd = [np.concatenate(x) for x in p["width"]]
            cc = lambda x, y: round(float(np.corrcoef(x, y)[0, 1]), 3) if len(x) > 2 and np.std(x) > 1e-9 else None
            summ[m] = {"views": int(len(nme)), "mouth_nme_pct_iod_median": round(float(np.median(nme)) * 100, 2),
                       "mouth_open_corr": cc(*op), "mouth_width_corr": cc(*wd),
                       "mouth_open_abs_err_median": round(float(np.median(np.abs(op[0] - op[1]))), 4)}
        return summ
    n_cv = int(sum(r["covisible_views"] for r in out.values()))
    return {"splits": out, "pooled": summarise(pooled, 1),
            "covisible": {"views": n_cv, "testable": n_cv >= cfg.covis_min_views,
                          "pooled": summarise(pooled_cv, cfg.covis_min_views)}}


def loo_check(prep: dict, cfg: FaceConfig) -> dict:
    """Cross-camera (each camera held out in turn) and temporal (alternate block_s blocks held out, 2 folds) checks.
    Few frames are seen by two cameras, so the camera check mostly tests the resting expression + temporal
    propagation; the time-block check tests whether the per-frame layer is consistent signal rather than jitter."""
    obs = prep["obs"]
    usable = obs["q"] > 0
    cams = [(f"cam{c}", usable & (obs["cam"] != c), usable & (obs["cam"] == c)) for c in sorted(set(obs["cam"][usable]))]
    blk = (np.asarray(obs["k"]) // max(1, int(round(cfg.block_s * prep["fps"])))) % 2
    times = [(f"blocks{b}", usable & (blk != b), usable & (blk == b)) for b in (0, 1)]
    return {"cross_camera": holdout_check(prep, cfg, cams), "time_blocks": holdout_check(prep, cfg, times)}


def adoption(check: dict | None, cfg: FaceConfig) -> dict:
    """Model selection on the hold-out checks, simplest first (mouth NME in % IOD, pooled):
      rest       a resting expression e0 / jaw0 replaces the neutral face when it beats neutral on held-out CAMERAS by
                 >= adopt_rest_gain_pp and keeps the held-out mouth-opening error within open_allowed(neutral's) =
                 max(adopt_open_rel x it, it + adopt_open_floor); candidates 'rest' (free) and 'rest_closed' (lip gap
                 of the neutral face kept): the eligible one with the lower held-out camera error ('rest_kind')
      expression the captured per-frame layer (on top of an adopted rest) when it beats rest on held-out TIME BLOCKS
                 by >= adopt_min_gain_pp, costs <= adopt_max_cam_loss_pp on held-out cameras, keeps the opening
                 error within open_allowed(rest's) and - when enough held-out views are co-visible (another camera saw
                 that frame) - is not worse than rest on those views (the only test of per-frame capture accuracy)
      jaw        the captured jaw on top of an adopted expression layer when it does not worsen either check
    The adopted candidate's name is returned as 'shipped' (build() ships exactly that recipe, fitted on all views)."""
    out = {"rest": False, "expression": False, "jaw": False, "shipped": "neutral"}
    if not check:
        out["reason"] = "no hold-out check run: neutral face"
        return out
    tb, cc = check["time_blocks"]["pooled"], check["cross_camera"]["pooled"]
    if not all(m in d for d in (tb, cc) for m in ("neutral", "rest", "final", "final_expr")):
        out["reason"] = "too few views for the hold-out checks: neutral face"
        return out
    nme = lambda d, m: d[m]["mouth_nme_pct_iod_median"]
    oe = lambda d, m: d[m]["mouth_open_abs_err_median"]
    allowed = lambda base: max(cfg.adopt_open_rel * base, base + cfg.adopt_open_floor)
    out["rest_open_err_allowed"] = round(allowed(oe(cc, "neutral")), 4)
    out["rest_candidates"], best = {}, None
    for m in ("rest", "rest_closed"):
        if m not in cc or m not in tb:
            continue
        gain = round(nme(cc, "neutral") - nme(cc, m), 2)
        ok = bool(gain >= cfg.adopt_rest_gain_pp and oe(cc, m) <= out["rest_open_err_allowed"])
        why = None if ok else ("no held-out gain over neutral" if gain < cfg.adopt_rest_gain_pp
                               else "held-out mouth-opening error above allowed")
        out["rest_candidates"][m] = {"gain_cam_pp": gain, "open_err_cam": oe(cc, m), "eligible": ok, "why_not": why}
        if ok and (best is None or nme(cc, m) < nme(cc, best)):
            best = m
    out["rest"] = best is not None
    out["rest_kind"] = best
    ref = best or "rest"
    out["rest_gain_cam_pp"] = round(nme(cc, "neutral") - nme(cc, ref), 2)
    out["rest_open_err"] = [oe(cc, "neutral"), oe(cc, ref)]
    out["expr_gain_time_pp"] = round(nme(tb, ref) - nme(tb, "final_expr"), 2)
    out["expr_loss_cam_pp"] = round(nme(cc, "final_expr") - nme(cc, ref), 2)
    cv = check["cross_camera"].get("covisible", {})
    cvp = cv.get("pooled", {})
    out["covisible_views"] = int(cv.get("views", 0))
    out["per_frame_capture_testable"] = bool(cv.get("testable") and "final_expr" in cvp and ref in cvp)
    cv_ok = True
    if out["per_frame_capture_testable"]:
        out["expr_gain_covisible_pp"] = round(nme(cvp, ref) - nme(cvp, "final_expr"), 2)
        cv_ok = out["expr_gain_covisible_pp"] >= 0
    out["expression"] = bool(out["rest"] and out["expr_gain_time_pp"] >= cfg.adopt_min_gain_pp
                             and out["expr_loss_cam_pp"] <= cfg.adopt_max_cam_loss_pp
                             and oe(tb, "final_expr") <= allowed(oe(tb, ref)) and cv_ok)
    out["jaw"] = bool(out["expression"] and oe(tb, "final") <= oe(tb, "final_expr") and oe(cc, "final") <= oe(cc, "final_expr")
                      and nme(tb, "final") <= nme(tb, "final_expr") + 0.05)
    out["shipped"] = "final" if out["jaw"] else "final_expr" if out["expression"] else best if out["rest"] else "neutral"
    why_not_rest = "; ".join(f"{m}: {c['why_not']} (gain {c['gain_cam_pp']} pp, opening error {c['open_err_cam']} vs "
                             f"allowed {out['rest_open_err_allowed']})" for m, c in out["rest_candidates"].items())
    out["reason"] = ((f"resting expression adopted ({best})" if out["rest"] else
                      f"resting expression NOT adopted ({why_not_rest}): neutral face") + ("; captured per-frame expression adopted" if out["expression"] else
                                         "; no per-frame expression") +
                     ("; captured jaw adopted" if out["jaw"] else "; jaw at rest") +
                     ("" if out["per_frame_capture_testable"] else
                      f"; per-frame capture accuracy NOT testable on this take ({out['covisible_views']} held-out views "
                      f"co-visible with another camera, < {cfg.covis_min_views})"))
    return out


def ship_rest(fits: dict, adopt: dict, cfg: FaceConfig) -> tuple[np.ndarray, float]:
    """The rest face of the adopted candidate, exactly as holdout_check built it (fits: mode -> fit_aligned result on
    all views): 'rest' / 'rest_closed' = that rest-mode fit's e0 / jaw0; 'final_expr' / 'final' = the per-frame fit's
    anchor e0 / jaw0; else neutral. -> (e_rest, jaw_rest rad)."""
    jmax = np.radians(cfg.jaw_max_deg)
    src = "frame" if adopt.get("expression") else adopt.get("rest_kind") if adopt.get("rest") else None
    if src is None:
        return np.zeros_like(np.asarray(fits["frame"]["e0"], float)), 0.0
    return np.asarray(fits[src]["e0"], float), float(np.clip(fits[src]["jaw0"], 0, jmax))


LIMITATIONS = (
    "Held-out cameras: most frames are seen by one camera, so on held-out cameras the per-frame layer is mostly "
    "temporal interpolation from other instants; it measures capture accuracy only on co-visible views (another camera "
    "saw the same frame in training) - see holdout.cross_camera.covisible and adoption.per_frame_capture_testable.",
    "Time blocks (0.2 s, alternating) test whether the per-frame layer is consistent signal rather than jitter, not "
    "whether it is accurate at a given instant.",
    "Faces are 8-30 px between the eyes in this video: MediaPipe's mouth shape noise is of the order of the expression "
    "signal; candidates are adopted only when they beat the simpler model on held-out views.",
    "'shipped' is the adopted candidate refitted on all views; shipped_heldout quotes that candidate's own hold-out "
    "numbers (the candidate 'final' in the hold-out tables is the gated per-frame layer, shipped only if adopted).",
    "Blinks (eyelid) are procedural stylisation; eye rotations are the body fit's (no gaze).")


def face_blendshapes(work: Path, n_expr: int, out: Path) -> dict:
    """Head-vertex blendshapes for Unity: SMPL-X expression 0..n_expr-1 (+ FLAME eyelids moved to SMPL-X when the
    gated files are present). Deltas in metres, model axes (rest pose, y up, +z forward; Unity: (x, y, -z))."""
    from .take import WORKSPACE
    m = np.load(smplx_model_npz(), allow_pickle=True)
    sd = np.asarray(m["shapedirs"], np.float32)
    nb = sd.shape[2] - 100                                       # SMPL-X: 300 shape + 100 expression
    ex = sd[:, :, nb:nb + n_expr]                                # (V,3,E)
    names = [f"expr{k:02d}" for k in range(n_expr)]
    deltas = [ex[:, :, k] for k in range(n_expr)]
    fl_ids = WORKSPACE.joinpath(*FLAME_IDS)
    assets = WORKSPACE.joinpath(*EYELID_ASSETS)
    eyelids = fl_ids.exists() and (assets / "l_eyelid.npy").exists() and (assets / "r_eyelid.npy").exists()
    if eyelids:
        ids = np.load(fl_ids).astype(np.int64)
        for side in ("l", "r"):
            dl = np.zeros_like(ex[:, :, 0])
            dl[ids] = np.load(assets / f"{side}_eyelid.npy").astype(np.float32)
            deltas.append(dl)
            names.append(f"eyelid_{'left' if side == 'l' else 'right'}")
    Dl = np.stack(deltas)                                        # (S,V,3)
    keep = np.flatnonzero(np.abs(Dl).max((0, 2)) > 1e-7)
    np.savez_compressed(out, vertex_ids=keep.astype(np.int32), names=np.array(names), deltas=Dl[:, keep].astype(np.float32),
                        note=np.array("SMPL-X expression dirs (model shapedirs[:, :, 300 + k]) and, when present, FLAME "
                                      "eyelid blendshapes (SMIRK assets, FLAME licence) mapped 1:1 by "
                                      "SMPL-X__FLAME_vertex_ids.npy; deltas in metres in the model's rest axes; "
                                      "likeness-free but model-licensed: never commit"))
    return {"names": names, "vertices": int(len(keep)), "eyelids": bool(eyelids),
            "max_delta_mm": [round(float(np.linalg.norm(Dl[s], axis=1).max() * 1000), 2) for s in range(len(names))]}


def build(take: Take, cfg: FaceConfig | None = None, roles=ROLES, loo: bool = True, review: bool = True) -> dict:
    """-> work/<take>/smplx_face/{lead,follow}.npz, face_blendshapes.npz, report.json (+ review images)."""
    import torch
    cfg = cfg or FaceConfig()
    if cfg.device == "cpu":
        torch.set_num_threads(cfg.threads)
    work = take.work
    od = work / "smplx_face"
    od.mkdir(parents=True, exist_ok=True)
    rep = {"take": take.name, "config": asdict(cfg), "limitations": list(LIMITATIONS), "roles": {}}
    bs = face_blendshapes(work, cfg.n_expr, od / "face_blendshapes.npz")
    rep["blendshapes"] = bs
    preps = {}
    for r in roles:
        t0 = time.time()
        prep = prepare(work, r, cfg)
        preps[r] = prep
        obs, T, fps = prep["obs"], len(prep["grid_t"]), prep["fps"]
        fr = fit_aligned(prep, cfg, mode="frame")
        rs = fit_aligned(prep, cfg, mode="rest")
        rc = fit_aligned(prep, cfg, mode="rest_closed")
        conf = confidence_curve(obs["k"], obs["q"], T, fps, cfg)
        expr, jaw, e_sm, j_sm = blend_final(fr["expr"], fr["jaw"], fr["e0"], fr["jaw0"], conf, fps, cfg)
        check = loo_check(prep, cfg) if loo else None
        adopt = adoption(check, cfg)
        fits = {"frame": fr, "rest": rs, "rest_closed": rc}
        e_rest, jaw_rest = ship_rest(fits, adopt, cfg)
        rk = adopt.get("rest_kind") or "rest"          # the rest candidate reported as 'expression_rest_fitted'
        rsel = fits[rk]
        if not adopt["expression"]:
            expr = np.repeat(e_rest[None], T, 0)
        if not adopt["jaw"]:
            jaw = np.full(T, jaw_rest)
        blink, starts = blink_schedule(prep["grid_t"], cfg, f"{take.name}/{r}")
        leye, reye = np.asarray(prep["smplx"]["leye_pose"], float).copy(), np.asarray(prep["smplx"]["reye_pose"], float).copy()
        gz = gaze_stylisation(prep["Rk"], fps, cfg)
        lid = blink
        if cfg.gaze_stylise:
            leye[:, 0] += gz["eye_pitch"]                      # small angles: the body fit's eyes are ~1 deg
            reye[:, 0] += gz["eye_pitch"]
            lid = np.maximum(blink, gz["droop"])
        n_views = np.bincount(obs["k"][obs["q"] > 0], minlength=T)
        jaw3 = np.zeros((T, 3))
        jaw3[:, 0] = jaw
        shipped = adopt["shipped"]
        np.savez(od / f"{r}.npz", t=prep["grid_t"], jaw_pose=jaw3, expression=expr.astype(np.float32),
                 confidence=conf.astype(np.float32), expression_rest=e_rest.astype(np.float32),
                 expression_rest_fitted=rsel["e0"].astype(np.float32),
                 jaw_rest_fitted_deg=np.array(np.degrees(rsel["jaw0"])), rest_fitted_kind=np.array(rk),
                 expression_rest_free=rs["e0"].astype(np.float32), expression_rest_closed=rc["e0"].astype(np.float32),
                 expression_anchor_perframe=fr["e0"].astype(np.float32),
                 jaw_rest_deg=np.array(np.degrees(jaw_rest)), adopted_rest=np.array(adopt["rest"]),
                 adopted_expression=np.array(adopt["expression"]), adopted_jaw=np.array(adopt["jaw"]),
                 shipped=np.array(shipped), expression_captured=e_sm.astype(np.float32),
                 jaw_captured_deg=np.degrees(j_sm).astype(np.float32), eyelid=np.stack([lid, lid], 1).astype(np.float32),
                 eyelid_names=np.array(["eyelid_left", "eyelid_right"]), blink_starts=starts,
                 leye_pose=leye.astype(np.float32), reye_pose=reye.astype(np.float32),
                 eye_pitch_stylised_deg=np.degrees(gz["eye_pitch"] if cfg.gaze_stylise else 0 * gz["eye_pitch"]).astype(np.float32),
                 eyelid_droop_stylised=(gz["droop"] if cfg.gaze_stylise else 0 * gz["droop"]).astype(np.float32),
                 head_pitch_down_deg=gz["head_pitch_down_deg"].astype(np.float32), gaze_stylised=np.array(cfg.gaze_stylise),
                 n_views=n_views.astype(np.int16), n_expr=np.array(cfg.n_expr),
                 source=np.array(f"face_expr.build: MediaPipe 478 -> iBUG 52, similarity-aligned shape fit, "
                                 f"{int((obs['q'] > 0).sum())} views; jaw joint 22 = x-rotation only; shipped = the "
                                 f"hold-out candidate '{shipped}' refitted on all views"),
                 note=np.array("expression / jaw_pose drive the avatar (SMPL-X expression[:n_expr] + jaw joint 22); "
                               "shipped = the adopted hold-out candidate (neutral | rest / rest_closed = that "
                               "rest-mode fit (expression_rest_free / _closed) | final_expr / final = the per-frame fit, "
                               "anchored at expression_anchor_perframe); confidence 0 = rest expression only; "
                               "adopted_* False = that layer failed the hold-out checks (rest -> neutral face, "
                               "expression/jaw -> rest value); eyelid = PROCEDURAL blinks + lid droop when the head "
                               "pitches down (stylisation, not capture); leye/reye = the body fit's (MAMMA, ~1 deg) + "
                               "eye_pitch_stylised_deg (the eyes follow a lowered head partway; stylisation, no gaze target)"))
        with torch.no_grad():
            hv = lambda e_, j_: prep["head"].verts(torch.tensor(np.asarray(e_, np.float32)),
                                                   torch.tensor([float(j_), 0.0, 0.0])).numpy()
            V0 = hv(np.zeros(cfg.n_expr), 0.0)
            move_rest = float(np.linalg.norm(hv(e_rest, jaw_rest) - V0, axis=1).max() * 1000)
            move_cand = {m: round(float(np.linalg.norm(
                hv(fits[m]["e0"], np.clip(fits[m]["jaw0"], 0, np.radians(cfg.jaw_max_deg))) - V0, axis=1).max() * 1000), 2)
                for m in ("rest", "rest_closed")}
        rr = {"views_usable": int((obs["q"] > 0).sum()), "views_all": int(len(obs["q"])),
              "views_by_cam": {c: int(((obs["cam"] == c) & (obs["q"] > 0)).sum()) for c in sorted(set(obs["cam"]))},
              "frames": T, "frames_conf_gt_0.5": int((conf > 0.5).sum()), "frames_conf_gt_0": int((conf > 0.01).sum()),
              "shipped": shipped,
              "rest_fits": {m: {"e0": np.round(fits[m]["e0"], 3).tolist(),
                                "e0_norm": round(float(np.linalg.norm(fits[m]["e0"])), 3),
                                "jaw0_deg": round(float(np.degrees(fits[m]["jaw0"])), 2),
                                "max_vertex_move_mm": move_cand[m]} for m in ("rest", "rest_closed")},
              "e0_perframe_anchor": np.round(fr["e0"], 3).tolist(),
              "e0_perframe_anchor_norm": round(float(np.linalg.norm(fr["e0"])), 3),
              "e_rest_used": np.round(e_rest, 3).tolist(), "e_rest_used_norm": round(float(np.linalg.norm(e_rest)), 3),
              "jaw_rest_used_deg": round(float(np.degrees(jaw_rest)), 2),
              "shipped_rest_max_vertex_move_mm": round(move_rest, 2),
              "jaw_final_deg_p50_p95_max": [round(float(np.degrees(np.percentile(jaw, q))), 2) for q in (50, 95, 100)],
              "expr_dev_final_p50_p95": [round(float(np.percentile(np.linalg.norm(expr - e_rest, axis=1), q)), 3)
                                         for q in (50, 95)],
              "expr_dev_captured_p50_p95": [round(float(np.percentile(np.linalg.norm(e_sm - fr["e0"], axis=1), q)), 3)
                                            for q in (50, 95)],
              "blinks": int(len(starts)), "identity_offsets_used": bool(prep["has_D"]),
              "gaze_stylised": bool(cfg.gaze_stylise),
              "head_pitch_down_deg_p50_p95": [round(float(np.percentile(gz["head_pitch_down_deg"], q)), 1) for q in (50, 95)],
              "eye_pitch_stylised_deg_max": round(float(np.degrees(gz["eye_pitch"]).max()), 1) if cfg.gaze_stylise else 0.0,
              "eyelid_droop_frames": int((gz["droop"] > 0.05).sum()) if cfg.gaze_stylise else 0,
              "adoption": adopt, "fit_runtime_s": round(fr["runtime_s"] + rs["runtime_s"] + rc["runtime_s"], 1)}
        if check:
            # the shipped object's own held-out numbers (the candidate of the same name, fitted without the held-out
            # views) - so the report never quotes a model that was not shipped
            rr["shipped_heldout"] = {kind: check[kind]["pooled"].get(shipped) for kind in ("cross_camera", "time_blocks")}
            rr["shipped_heldout"]["covisible"] = check["cross_camera"]["covisible"]["pooled"].get(shipped)
        pf = work / "avatar" / "face" / f"{r}_portrait_fit.npz"
        if pf.exists():
            pe = np.load(pf)["expr"][:cfg.n_expr]
            rr["portrait_expr_first"] = np.round(pe, 3).tolist()
            cos = lambda a_, b_: round(float(a_ @ b_ / max(np.linalg.norm(a_) * np.linalg.norm(b_), 1e-9)), 3)
            rr["cos_rest_fits_vs_portrait"] = {m: cos(pe, fits[m]["e0"]) for m in ("rest", "rest_closed")}
            rr["cos_shipped_rest_vs_portrait"] = cos(pe, e_rest) if np.linalg.norm(e_rest) > 0 else None
        ins = {}
        for m, (ee, jj) in {"neutral": (np.zeros_like(expr), np.zeros(T)),
                            **{m: (np.repeat(fits[m]["e0"][None], T, 0),
                                   np.full(T, float(np.clip(fits[m]["jaw0"], 0, np.radians(cfg.jaw_max_deg)))))
                               for m in ("rest", "rest_closed")},
                            "shipped": (expr, jaw)}.items():
            er = aligned_errors(prep, cfg, ee, jj, np.flatnonzero(obs["q"] > 0), fr["_world"], fr["_proj"])
            ins[m] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in er.items() if not k.startswith("_")}
        rr["in_sample"] = ins
        if check:
            rr["holdout"] = check
        rr["runtime_s"] = round(time.time() - t0, 1)
        rep["roles"][r] = rr
        print(f"[face_expr] {r}: " + json.dumps({k: v for k, v in rr.items() if k != "holdout"}), flush=True)
        if check:
            for kind in ("cross_camera", "time_blocks"):
                print(f"[face_expr] {r} {kind} pooled: {json.dumps(check[kind]['pooled'])}", flush=True)
            print(f"[face_expr] {r} cross_camera covisible: {json.dumps(check['cross_camera']['covisible'])}", flush=True)
    (od / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    if review:
        rev = work / "review"
        rev.mkdir(parents=True, exist_ok=True)
        paths = [confidence_plot(take, roles, rev / "face_expr_confidence.png")]
        try:
            paths.append(review_final(take, roles, rev / "face_expr.jpg", preps=preps))
        except FileNotFoundError as ex:
            print(f"[face_expr] review skipped: {ex}")
        rep["review"] = [str(p) for p in paths]
        (od / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def confidence_plot(take: Take, roles, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    work = take.work
    fig, ax = plt.subplots(len(roles), 1, figsize=(14, 3.9 * len(roles)), squeeze=False)
    for n, r in enumerate(roles):
        z = np.load(work / "smplx_face" / f"{r}.npz")
        t = z["t"]
        a = ax[n, 0]
        a.fill_between(t, 0, z["confidence"], color="0.75", label="confidence (0 = rest expression only)")
        a.plot(t, np.degrees(z["jaw_pose"][:, 0]) / 6, color="C3", lw=1.6, label="jaw opening (deg / 6)")
        a.plot(t, z["jaw_captured_deg"] / 6, color="C3", lw=0.7, ls="--", label="jaw captured, before gating (deg / 6)")
        ad = f"rest expression {'ADOPTED' if bool(z['adopted_rest']) else 'not adopted (neutral)'}, per-frame "              f"expression {'ADOPTED' if bool(z['adopted_expression']) else 'not adopted'}, jaw "              f"{'ADOPTED' if bool(z['adopted_jaw']) else 'at rest'}"
        dev = np.linalg.norm(z["expression"] - z["expression_rest"][None], axis=1)
        devc = np.linalg.norm(z["expression_captured"] - z["expression_rest"][None], axis=1)
        a.plot(t, dev, color="C0", lw=1.6, label="|expression - rest| final")
        a.plot(t, devc, color="C0", lw=0.7, ls="--", label="|expression - rest| captured, before gating")
        a.plot(t, z["eyelid"][:, 0] * 0.25 - 0.3, color="C2", lw=1.0, label="eyelids: procedural blinks + droop when the head lowers (stylisation)")
        nv = z["n_views"]
        a.scatter(t[nv > 0], np.full((nv > 0).sum(), 1.1), s=6 + 6 * nv[nv > 0], c="k", marker="|",
                  label="frames with usable MediaPipe views (tick size = number)")
        a.set_ylim(-0.35, 1.6)
        a.set_title(f"{LABEL[r]}: final face curves on the fit grid (smplx_face/{r}.npz) - {ad}", fontsize=10)
        a.legend(fontsize=7, ncol=3, loc="upper right", framealpha=0.85)
        a.set_xlabel("reference time (s)")
    fig.suptitle(f"{take.name}: per-frame face - confidence, jaw, expression deviation from the resting expression, blinks")
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path


def face_vertices(work: Path, role: str, frames: np.ndarray, face: dict | None, n_expr: int = 10) -> np.ndarray:
    """(n,V,3) posed vertices of the body fit at grid frames, with the face block (jaw / expression; the eyelid
    blendshapes are shown in the rest-pose close-ups only) or without (face None = as exported today: jaw 0,
    expression 0)."""
    import smplx
    import torch

    from .mamma_dyn import SMPLX_MODELS
    d = {k: v for k, v in np.load(work / "smplx" / f"{role}.npz", allow_pickle=True).items()}
    frames = np.asarray(frames, int)
    n = len(frames)
    m = smplx.create(str(SMPLX_MODELS), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                     flat_hand_mean=True, num_betas=int(d.get("num_betas", 16)), num_expression_coeffs=n_expr, batch_size=n)
    f = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float32)
    jaw = np.zeros((n, 3)) if face is None else face["jaw_pose"][frames]
    ex = np.zeros((n, n_expr)) if face is None else face["expression"][frames][:, :n_expr]
    with torch.no_grad():
        o = m(betas=f(np.asarray(d["betas"]).reshape(-1))[None].expand(n, -1), global_orient=f(d["global_orient"][frames]),
              body_pose=f(d["body_pose"][frames]), jaw_pose=f(jaw),
              leye_pose=f((d if face is None else face)["leye_pose"][frames]),
              reye_pose=f((d if face is None else face)["reye_pose"][frames]), left_hand_pose=f(d["left_hand_pose"][frames]),
              right_hand_pose=f(d["right_hand_pose"][frames]), transl=f(d["transl"][frames]), expression=f(ex))
    return o.vertices.numpy()


def rest_head(work: Path, role: str, face: dict | None, k: int | None, n_expr: int = 10, eyelid: float = 0.0):
    """Rest-pose vertices (V,3) of the dancer with the frame's face block (or none) for close-up renders."""
    import torch

    from .avatar_face import HeadModel
    d = np.load(work / "smplx" / f"{role}.npz", allow_pickle=True)
    head = HeadModel(np.asarray(d["betas"], np.float32), n_expr)
    e = torch.zeros(n_expr) if face is None else torch.tensor(face["expression"][k][:n_expr], dtype=torch.float32)
    j = torch.zeros(3) if face is None else torch.tensor(face["jaw_pose"][k], dtype=torch.float32)
    with torch.no_grad():
        V = head.verts(e, j).numpy().astype(np.float64)
    if eyelid > 0:
        bs = np.load(work / "smplx_face" / "face_blendshapes.npz")
        names = [str(x) for x in bs["names"]]
        if "eyelid_left" in names:
            ids = bs["vertex_ids"]
            V[ids] += eyelid * (bs["deltas"][names.index("eyelid_left")] + bs["deltas"][names.index("eyelid_right")])
    return V, head


def _crop_camera(K, box, tile):
    """Intrinsics that render only the square box (x0, y0, x1, y1) at tile x tile px."""
    s = tile / float(box[2] - box[0])
    K2 = np.array(K, float).copy()
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] = (K2[0, 2] - box[0]) * s
    K2[1, 2] = (K2[1, 2] - box[1]) * s
    return K2


def render_crop(torch, geo, posed: dict, cols: dict, K, R, t, k1, box, tile: int, device, bg: int = 40) -> np.ndarray:
    from .avatar_texture import render
    ren, rid = render(torch, geo, posed, cols, _crop_camera(K, box, tile), R, t, float(k1), tile, tile, device)
    ren8 = (np.clip(ren, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    return np.where((rid >= 0)[..., None], ren8, bg).astype(np.uint8)


_HEAD_V: list = []


def _head_vertices() -> np.ndarray:
    if not _HEAD_V:
        from .avatar_uv import vertex_regions
        _HEAD_V.append(vertex_regions(np.load(smplx_model_npz(), allow_pickle=True))["head"])
    return _HEAD_V[0]


def render_front(torch, geo, V, col, tile: int, device, yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                 center=None, half: float = 0.13, bg: int = 40) -> np.ndarray:
    """Close-up of a rest-pose head (SMPL-X looks along +z) from a virtual camera at yaw / pitch, 1.2 m away."""
    from .avatar_texture import render
    if center is None:
        center = V[_head_vertices()].mean(0) + np.array([0, -0.02, 0])
    a, p = np.radians(yaw_deg), np.radians(pitch_deg)
    dirv = np.array([np.sin(a) * np.cos(p), np.sin(p), np.cos(a) * np.cos(p)])
    C = center + dirv * 1.2
    z = center - C
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ C
    f = tile / 2 * 1.2 / half
    K = np.array([[f, 0, tile / 2], [0, f, tile / 2], [0, 0, 1.0]])
    ren, rid = render(torch, geo, {"x": V}, {"x": col}, K, R, t, 0.0, tile, tile, device)
    ren8 = (np.clip(ren, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    return np.where((rid >= 0)[..., None], ren8, bg).astype(np.uint8)


def albedo_colors(torch, work: Path, atlas, role: str, device, which: str = "face"):
    """Per-texel sRGB colours of the avatar texture: 'face' = the portrait-face texture (avatar_face), else the bake."""
    import cv2
    p = work / "avatar" / f"{role}_albedo_face.png"
    if not p.exists():
        p = work / "avatar" / "face" / f"{role}_albedo_face.png"
    if which != "face" or not p.exists():
        p = work / "avatar" / f"{role}_albedo.png"
    im = cv2.imread(str(p))[:, :, ::-1].astype(np.float32) / 255
    return torch.as_tensor(im[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]], device=device)


def pick_instants(work: Path, role: str, face: dict, obs: dict, n: int = 4, min_gap_s: float = 0.5) -> list[dict]:
    """Instants to review: the best (largest IOD x quality) view of frames, preferring confident frames where the
    final face moves most from the rest expression, spaced >= min_gap_s."""
    t = face["t"]
    move = np.linalg.norm(face["expression"] - face["expression_rest"][None], axis=1) + \
        0.1 * np.abs(np.degrees(face["jaw_pose"][:, 0]) - float(face["jaw_rest_deg"]))
    cand = []
    for j in range(len(obs["k"])):
        if obs["iod"][j] < 12:
            continue
        k = int(obs["k"][j])
        score = obs["iod"][j] * (0.3 + obs["q"][j]) * (0.5 + face["confidence"][k] * move[k])
        cand.append((score, j, k))
    cand.sort(key=lambda x: -x[0])
    picked = []
    for sc, j, k in cand:
        if all(abs(t[k] - t[p["k"]]) >= min_gap_s for p in picked):
            picked.append({"j": j, "k": k, "cam": str(obs["cam"][j]), "i": int(obs["i"][j]), "iod": float(obs["iod"][j]),
                           "q": float(obs["q"][j])})
        if len(picked) == n:
            break
    return sorted(picked, key=lambda p: p["k"])


def _component_rows(work: Path, roles, geo, cols: dict, width: int, dev) -> list[np.ndarray]:
    """Per dancer, frontal close-ups: neutral | the adopted rest face | the fitted rest expression x2 (direction,
    exaggerated) | eyelid blendshapes at 1 (blink, stylisation) - so the face block's parts can be judged at all."""
    import torch

    from .face_common import label
    out = []
    tile = width // 4
    for r in roles:
        face = {k: v for k, v in np.load(work / "smplx_face" / f"{r}.npz").items()}
        k = int(np.argmax(face["confidence"]))
        V0, _ = rest_head(work, r, None, None)
        V1, _ = rest_head(work, r, face, k)
        f2 = dict(face)
        f2["expression"] = np.repeat(2 * face["expression_rest_fitted"][None], len(face["t"]), 0)
        f2["jaw_pose"] = np.zeros_like(face["jaw_pose"])
        f2["jaw_pose"][:, 0] = 2 * np.radians(float(face["jaw_rest_fitted_deg"]))
        V2, _ = rest_head(work, r, f2, k)
        V3, _ = rest_head(work, r, None, None, eyelid=1.0)
        disp = float(np.linalg.norm(V1 - V0, axis=1).max() * 1000)
        tiles = [render_front(torch, geo, V, cols[r], tile, dev, yaw_deg=12, half=0.11) for V in (V0, V1, V2, V3)]
        label(tiles[0], "neutral (as exported today)", (4, 14), scale=0.4)
        label(tiles[1], "final rest face" + ("" if bool(face["adopted_rest"]) else " = neutral"), (4, 14), scale=0.4)
        label(tiles[1], f"max vertex move {disp:.1f} mm", (4, tile - 8), scale=0.4)
        label(tiles[2], "fitted rest expression x2 (exaggerated)", (4, 14), scale=0.4)
        label(tiles[3], "eyelid blendshapes = 1 (blink, stylised)", (4, 14), scale=0.4)
        row = np.concatenate(tiles, 1)
        hs = np.full((20, row.shape[1], 3), 20, np.uint8)
        label(hs, f"{LABEL[r]}: face block parts (rest pose, front)", (6, 15), scale=0.45)
        out.append(np.concatenate([hs, row], 0))
    return out


def review_final(take: Take, roles, path: Path, n: int = 4, tile: int = 230, preps: dict | None = None) -> Path:
    """Per dancer n instants: video crop | avatar today (no face block) | avatar with the final face (same camera) |
    close-up front, today | close-up front, final face. Avatar textures: the portrait face (avatar_face) when present."""
    import cv2
    import torch

    from .avatar_texture import _Geo
    from .avatar_uv import build_atlas
    from .face_common import crop_pad, label
    from .mamma_dyn import SMPLX_MODELS
    work = take.work
    from .avatar_face import capped_atlas
    atlas = build_atlas(SMPLX_MODELS, 2048)
    dev = "cpu"
    geo = _Geo(capped_atlas(atlas), dev)          # mouth bag capped: no see-through line between the lips
    cols = {r: albedo_colors(torch, work, atlas, r, dev) for r in ROLES}
    preps = preps or {}
    rows = []
    for r in roles:
        face = {k: v for k, v in np.load(work / "smplx_face" / f"{r}.npz").items()}
        obs = (preps.get(r) or prepare(work, r, FaceConfig()))["obs"]
        inst = pick_instants(work, r, face, obs, n)
        if not inst:
            continue
        ks = np.array([p["k"] for p in inst])
        other = [o for o in ROLES if o != r][0]
        V_none = face_vertices(work, r, ks, None)
        V_face = face_vertices(work, r, ks, face)
        V_oth = face_vertices(work, other, ks, None)
        camw = cameras_world(work, sorted({p["cam"] for p in inst}))
        for j, p in enumerate(inst):
            c, i, k = p["cam"], p["i"], p["k"]
            img = cv2.imread(str(work / "frames" / c / f"{i:06d}.jpg"))
            lm = obs["uv"][p["j"]]
            lo, hi = lm.min(0), lm.max(0)
            side = max(hi - lo) * 2.3
            cx = (lo + hi) / 2
            box = (int(cx[0] - side / 2), int(cx[1] - side / 2), int(cx[0] + side / 2), int(cx[1] + side / 2))
            ci = camw[c]
            tiles = [crop_pad(img, box, tile)]
            for V in (V_none[j], V_face[j]):
                tiles.append(render_crop(torch, geo, {r: V, other: V_oth[j]}, cols, ci["K"][i], ci["R"][i], ci["t"][i],
                                         ci["k1"][i], box, tile, dev))
            Vr0, _ = rest_head(work, r, None, None)
            Vr1, _ = rest_head(work, r, face, k, eyelid=float(face["eyelid"][k, 0]))
            for V in (Vr0, Vr1):
                tiles.append(render_front(torch, geo, V, cols[r], tile, dev, yaw_deg=15 if r == "lead" else -15))
            jd = np.degrees(face["jaw_pose"][k, 0])
            dv = np.linalg.norm(face["expression"][k] - face["expression_rest"])
            label(tiles[0], f"cam {c} #{i} t={face['t'][k]:.2f} IOD {p['iod']:.0f}", (4, 14), scale=0.4)
            label(tiles[1], "today: no face block", (4, 14), scale=0.4)
            label(tiles[2], f"final: conf {face['confidence'][k]:.2f}", (4, 14), scale=0.4)
            label(tiles[2], f"jaw {jd:.1f}deg |e-e0| {dv:.2f}", (4, tile - 8), scale=0.4)
            label(tiles[3], "close-up, today", (4, 14), scale=0.4)
            label(tiles[4], "close-up, final face", (4, 14), scale=0.4)
            if face["eyelid"][k, 0] > 0.05:
                label(tiles[4], f"eyelids {face['eyelid'][k, 0]:.2f} (stylised blink / droop)", (4, tile - 8), scale=0.4)
            sep = np.zeros((tile, 3, 3), np.uint8)
            row = np.concatenate([tiles[0], sep, tiles[1], tiles[2], sep, tiles[3], tiles[4]], 1)
            hs = np.full((20, row.shape[1], 3), 20, np.uint8)
            label(hs, f"{LABEL[r]}  view quality {p['q']:.2f}", (6, 15), scale=0.45)
            rows.append(np.concatenate([hs, row], 0))
    if not rows:
        raise FileNotFoundError("no views to show")
    rows += _component_rows(work, roles, geo, cols, rows[0].shape[1], dev)
    w = max(rw.shape[1] for rw in rows)
    sheet = np.concatenate([np.pad(rw, ((0, 3), (0, w - rw.shape[1]), (0, 0))) for rw in rows], 0)
    top = np.full((44, w, 3), 10, np.uint8)
    label(top, f"{take.name}: video | avatar as exported today (jaw 0, expression 0) | with the final face block "
               f"(same pose + camera)", (6, 17), scale=0.45)
    label(top, "| close-ups (rest pose): today vs final. Final = the adopted hold-out candidate (neutral / rest-mode fit "
               "/ gated per-frame layer); blinks are stylisation", (6, 37), scale=0.45)
    cv2.imwrite(str(path), np.concatenate([top, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="per-frame jaw + expression for the avatars (smplx_face/)")
    ap.add_argument("take")
    ap.add_argument("--roles", nargs="*", default=list(ROLES))
    ap.add_argument("--no-loo", action="store_true", help="skip the leave-one-camera-out check")
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--review-only", action="store_true", help="redraw the review images from smplx_face/")
    ap.add_argument("--research", action="store_true",
                    help="the earlier absolute-position research fit (face/expr_<role>.npz, neck correction, LOO)")
    a = ap.parse_args(argv)
    take = Take.load(a.take)
    if a.review_only:
        import torch
        torch.set_num_threads(FaceConfig().threads)
        rev = take.work / "review"
        # the blendshapes are cheap: rewritten here so a pod run (no gated FLAME files there) gains the eyelids
        bs = face_blendshapes(take.work, FaceConfig().n_expr, take.work / "smplx_face" / "face_blendshapes.npz")
        print("[face_expr] blendshapes:", json.dumps(bs))
        rep = {"review": [str(confidence_plot(take, tuple(a.roles), rev / "face_expr_confidence.png")),
                          str(review_final(take, tuple(a.roles), rev / "face_expr.jpg"))]}
    elif a.research:
        rep = run(take, roles=tuple(a.roles), loo=not a.no_loo)
    else:
        rep = build(take, roles=tuple(a.roles), loo=not a.no_loo, review=not a.no_review)
    print("review:", rep.get("review"))


if __name__ == "__main__":
    sys.exit(main())
