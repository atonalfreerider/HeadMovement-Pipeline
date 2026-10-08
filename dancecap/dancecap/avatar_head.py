"""Head-orientation refinement for texturing: MAMMA's SMPL-X head is often turned/tilted away from the real one
(its face points miss the YOLO COCO face keypoints by 1-3 inter-eye distances), which smears the face texture.

Per 30 Hz grid frame we fit a world rotation dR about the neck joint (joint 12) to the multi-view COCO face
keypoints (nose, eyes, ears) of that dancer, jointly over the window with a temporal smoothness term and a weak
prior towards zero. dR is applied as G12' = dR G12, i.e. the neck's LOCAL rotation becomes
L12' = G9^T dR G9 L12 (G9 = global rotation of spine3), so FK/skinning stay exact SMPL-X.

    avatar/head_fix.npz   t, delta_rotvec (T,3) world, neck_pose (T,3) corrected local axis-angle of joint 12,
                          err_before_m / err_after_m (T,) median face-keypoint error in metres at the head,
                          n_obs (T,) observations used
The smplx/ npz files are NOT changed (owned by the mamma stage); avatar_texture poses the bodies with this fix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mamma_dyn import world_cameras

SMPLX_FACE = (55, 56, 57, 58, 59)   # smplx joints: nose, right_eye, left_eye, right_ear, left_ear
COCO_FACE = (0, 2, 1, 4, 3)         # same points in COCO-17 order
CHAIN_TO_SPINE3 = (0, 3, 6, 9)      # pelvis -> spine1 -> spine2 -> spine3
PIDS = {"lead": 1, "follow": 2}


def _rotvec_to_mat(rv: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(rv).reshape(-1, 3)).as_matrix().reshape(np.shape(rv)[:-1] + (3, 3))


def _box_iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def face_observations(work: Path, role: str, t_grid: np.ndarray, cams: list[str], max_dt: float = 0.02,
                      min_conf: float = 0.5) -> list[list[dict]]:
    """Per grid frame: [{cam, K, R, t, uv (5,2), conf (5,)}] of the role's COCO face keypoints (detection matched to
    the role's box in dancers/roles.json, IoU > 0.5; camera frame within max_dt of the grid time)."""
    sim3_p = work / "cameras" / "world_from_sfm.json"
    sim3 = json.loads(sim3_p.read_text()) if sim3_p.exists() else None
    roles = json.loads((work / "dancers" / "roles.json").read_text())
    from .avatar_texture import view_time_offsets
    from .mamma_dyn import apply_offset
    offs = view_time_offsets(work)
    out: list[list[dict]] = [[] for _ in t_grid]
    for c in cams:
        pf, cf = work / "people" / f"{c}.npz", work / "cameras" / f"{c}.npz"
        if not pf.exists() or not cf.exists() or c not in roles["cams"]:
            continue
        z = np.load(cf)
        Rw, tw = world_cameras(z["R"], z["tvec"], sim3)
        ok_cam = z["ok"] if "ok" in z.files else np.ones(len(Rw), bool)
        with np.load(pf) as zz:  # read once: indexing a lazy NpzFile per frame re-decompresses the whole array
            pp = {k: zz[k] for k in ("valid", "boxes", "kpts") if k in zz.files}
        if "kpts" not in pp:
            continue
        frames = roles["cams"][c]["frames"]
        ts = np.array([float(f["t"]) for f in json.loads((work / "frames" / c / "frames.json").read_text())["frames"]])
        if c in offs:  # the fitted bodies are on the sync-corrected grid (avatar_texture.view_time_offsets)
            ts = apply_offset(ts, offs[c])
        for k, t in enumerate(t_grid):
            i = int(np.argmin(np.abs(ts - t)))
            if abs(ts[i] - t) > max_dt or not ok_cam[i] or i >= len(frames):
                continue
            fr = frames[i]
            if not fr.get(role) or fr.get("swap_prob", 0.0) > 0.2:
                continue
            box = fr[role].get("box")
            if box is None or fr[role].get("geo_veto"):  # wrong-person span removed by dancers_geoveto
                continue
            valid = pp["valid"][i]
            bx, kp = pp["boxes"][i][valid], pp["kpts"][i][valid]
            if len(bx) == 0:
                continue
            ious = [_box_iou(b, box) for b in bx]
            j = int(np.argmax(ious))
            if ious[j] < 0.5:
                continue
            obs = kp[j][list(COCO_FACE)]
            conf = np.where(obs[:, 2] >= min_conf, obs[:, 2], 0.0)
            if (conf > 0).sum() < 2:
                continue
            out[k].append({"cam": c, "K": z["K"][i], "R": Rw[i], "t": tw[i], "uv": obs[:, :2].astype(float),
                           "conf": conf})
    return out


def _residual_terms(obs: list[list[dict]]):
    rows = []
    for k, ol in enumerate(obs):
        for o in ol:
            for p in range(5):
                if o["conf"][p] > 0:
                    rows.append((k, p, o))
    return rows


def fit_head(face_pts: np.ndarray, neck: np.ndarray, obs: list[list[dict]], smooth: float = 0.15,
             prior: float = 0.02, f_scale: float = 0.02, max_iter: int = 60) -> tuple[np.ndarray, dict]:
    """face_pts (T,5,3) and neck joint (T,3) of the current fit -> delta rotvec (T,3) minimising the face keypoints'
    reprojection error in metres at the head (px x depth / f), soft-L1, + smooth*|w_k+1 - w_k| + prior*|w_k|."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    T = len(face_pts)
    rows = _residual_terms(obs)
    n_obs = np.zeros(T, int)
    for k, _, _ in rows:
        n_obs[k] += 1
    if not rows:
        return np.zeros((T, 3)), {"status": "no observations"}
    ks = np.array([r[0] for r in rows])
    ps = np.array([r[1] for r in rows])
    K = np.stack([r[2]["K"] for r in rows])
    R = np.stack([r[2]["R"] for r in rows])
    tt = np.stack([r[2]["t"] for r in rows])
    uv = np.stack([r[2]["uv"][r[1]] for r in rows])
    cw = np.sqrt(np.array([r[2]["conf"][r[1]] for r in rows]))
    P0 = face_pts[ks, ps] - neck[ks]  # lever arms

    def project(w):
        Rd = _rotvec_to_mat(w[ks])
        X = neck[ks] + np.einsum("nij,nj->ni", Rd, P0)
        Xc = np.einsum("nij,nj->ni", R, X) + tt
        z = np.maximum(Xc[:, 2], 0.1)
        px = Xc[:, :2] / z[:, None] * K[:, [0, 1], [0, 1]] + K[:, :2, 2]
        return px, z

    def fun(x):
        w = x.reshape(T, 3)
        px, z = project(w)
        e = (px - uv) * (z / K[:, 0, 0])[:, None] * cw[:, None]  # metres at the head
        return np.concatenate([e.ravel(), (smooth * np.diff(w, axis=0)).ravel(), (prior * w).ravel()])

    n_res = 2 * len(rows) + 3 * (T - 1) + 3 * T
    S = lil_matrix((n_res, 3 * T), dtype=np.int8)
    for n, k in enumerate(ks):
        S[2 * n:2 * n + 2, 3 * k:3 * k + 3] = 1
    base = 2 * len(rows)
    for k in range(T - 1):
        S[base + 3 * k:base + 3 * k + 3, 3 * k:3 * k + 6] = 1
    base += 3 * (T - 1)
    for k in range(T):
        S[base + 3 * k:base + 3 * k + 3, 3 * k:3 * k + 3] = 1
    res = least_squares(fun, np.zeros(3 * T), jac_sparsity=S, loss="soft_l1", f_scale=f_scale, max_nfev=max_iter,
                        x_scale=0.1)
    w = res.x.reshape(T, 3)

    def per_frame(wv):
        px, z = project(wv)
        e = np.linalg.norm(px - uv, axis=1) * z / K[:, 0, 0]
        return np.array([np.median(e[ks == k]) if (ks == k).any() else np.nan for k in range(T)]), e

    before, eb = per_frame(np.zeros((T, 3)))
    after, ea = per_frame(w)
    cams = np.array([r[2]["cam"] for r in rows])
    per_cam = {c: {"before_cm": round(float(np.median(eb[cams == c])) * 100, 2),
                   "after_cm": round(float(np.median(ea[cams == c])) * 100, 2), "n": int((cams == c).sum())}
               for c in sorted(set(cams))}
    info = {"status": "ok", "n_obs": int(len(rows)), "frames_with_obs": int((n_obs > 0).sum()),
            "median_err_before_cm": round(float(np.nanmedian(before)) * 100, 2),
            "median_err_after_cm": round(float(np.nanmedian(after)) * 100, 2),
            "median_correction_deg": round(float(np.degrees(np.median(np.linalg.norm(w, axis=1)))), 1),
            "p90_correction_deg": round(float(np.degrees(np.percentile(np.linalg.norm(w, axis=1), 90))), 1),
            "per_cam": per_cam, "cost": float(res.cost), "nfev": int(res.nfev)}
    return w, {**info, "_before": before, "_after": after, "_n_obs": n_obs}


def neck_pose_with(d: dict, delta: np.ndarray) -> np.ndarray:
    """Corrected local axis-angle of joint 12 (neck) so that its global rotation becomes dR G12."""
    from scipy.spatial.transform import Rotation
    T = len(d["t"])
    bp = np.asarray(d["body_pose"], float).reshape(T, 21, 3)
    G9 = _rotvec_to_mat(np.asarray(d["global_orient"], float))
    for j in CHAIN_TO_SPINE3[1:]:
        G9 = G9 @ _rotvec_to_mat(bp[:, j - 1])
    L12 = _rotvec_to_mat(bp[:, 11])
    L12n = np.transpose(G9, (0, 2, 1)) @ _rotvec_to_mat(delta) @ G9 @ L12
    return Rotation.from_matrix(L12n).as_rotvec()


def face_points(d: dict, model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(T,5,3) nose/eyes/ears and (T,3) neck joint of the stored fit (smplx library FK)."""
    import smplx
    import torch
    T = len(d["t"])
    nb = int(d.get("num_betas", len(d["betas"])))
    m = smplx.create(str(model_dir), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                     flat_hand_mean=bool(d.get("flat_hand_mean", True)), num_betas=nb, batch_size=T)
    f = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float32)
    z3 = np.zeros((T, 3))
    with torch.no_grad():
        o = m(betas=f(np.asarray(d["betas"]).reshape(-1)[:nb])[None].expand(T, -1), global_orient=f(d["global_orient"]),
              body_pose=f(d["body_pose"]), jaw_pose=f(d.get("jaw_pose", z3)), leye_pose=f(d.get("leye_pose", z3)),
              reye_pose=f(d.get("reye_pose", z3)), left_hand_pose=f(d["left_hand_pose"]),
              right_hand_pose=f(d["right_hand_pose"]), transl=f(d["transl"]))
    J = o.joints.numpy().astype(np.float64)
    return J[:, list(SMPLX_FACE)], J[:, 12]


def run_head_fix(work: Path, d: dict, role: str, model_dir: Path, cams: list[str], holdout: str | None = None,
                 **kw) -> tuple[np.ndarray, dict]:
    """Fit the correction for one dancer; with holdout=<cam> that camera is left out of the fit and only scored."""
    pts, neck = face_points(d, model_dir)
    obs = face_observations(work, role, np.asarray(d["t"], float), cams)
    fit_obs = [[o for o in ol if o["cam"] != holdout] for ol in obs] if holdout else obs
    delta, info = fit_head(pts, neck, fit_obs, **kw)
    if holdout:
        ho = [[o for o in ol if o["cam"] == holdout] for ol in obs]
        if any(ho):
            rows = _residual_terms(ho)
            e0, e1 = [], []
            for wv, acc in ((np.zeros_like(delta), e0), (delta, e1)):
                for (k, p, o) in rows:
                    X = neck[k] + _rotvec_to_mat(wv[k]) @ (pts[k, p] - neck[k])
                    Xc = o["R"] @ X + o["t"]
                    px = Xc[:2] / Xc[2] * o["K"][0, 0] + o["K"][:2, 2]
                    acc.append(np.linalg.norm(px - o["uv"][p]) * Xc[2] / o["K"][0, 0])
            info["holdout"] = {"cam": holdout, "n": len(rows), "before_cm": round(float(np.median(e0)) * 100, 2),
                               "after_cm": round(float(np.median(e1)) * 100, 2)}
    return delta, info
