"""The camera evidence of the "sides" step (DESIGN 2.2): where the joints really are, which views see them, and what each view says.

    triangulate_all       vectorised RANSAC triangulation of all 17 COCO joints at 30 Hz (hand_contact.triangulate_joint's rules: seed
                          pairs >= 8 deg apart, inliers 14 px x f/1000, final DLT on the inliers) + the covariance
                          S = (sum_c J_c^T J_c / sigma_c^2)^-1 (sigma_c = 4 px x f/1000, eigenvalues floored at (2 cm)^2), an identity
                          gate of 0.60 m to the base joint (role swaps only; never the 0.30 m TriCfg gate) and a per-joint reach
    coco_bias17           the COCO -> SMPL-X offset of each joint in the dancer's yaw frame, on free frames (hand_contact.calibrate_coco_bias
                          generalised to 17 joints)
    visible               per view, is a point visible under a pose (ray test against both dancers' capsules: zouk_contacts.joint_visible
                          generalised, the occluders a parameter)
    retriangulate_consistent  drop the detections of views that predict the joint hidden, triangulate again
    face_bits / lr_bits   per view: her / his face seen (+1) / hidden (-1); the left/right order of shoulders + hips (jl_facing.lr_rho)
    crossing_votes        per view: who is in front where two segments cross in the image (s3D = o_c x sigma_c, relations.md 2.2)
    missing_term          P(detection | visible / hidden) log-likelihoods (MNAR)
    cue_reliability       q_c of a view's ordinal cue from frames whose 3D margin is decisive
    heldout_plan          the phone held out of the solve per 10-s block

`run(take)` writes work/<take>/sides/evidence/ (tri_<role>.npz, tri_<role>_ho.npz, coco_bias.json, views_<cam>.npz, capsules.json,
sites.json, evidence.json). The keys are listed in the notes of the step (build_A.md) and in `TRI_KEYS` / `VIEW_KEYS` below.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import geom
from .adapters import PIDX, MultiView, adapter_for
from .config import SidesCfg, load_cfg

TRI_KEYS = ("t", "cams", "X", "Xd", "Xs", "valid_s", "S", "nv", "inl", "err", "X_all", "nv_all", "dropped", "ho")
VIEW_KEYS = ("uv", "conf", "det", "face", "face_conf", "lr", "part_vis", "parts", "vis_pred", "vis_palm", "ok", "C", "f")
PARTS = ("body", "left_hand", "right_hand", "left_foot", "right_foot", "head")


def log(*a) -> None:
    print("[sides.evidence]", *a, flush=True)


def write_json(p: Path, obj) -> Path:
    from .. import hand_contact as hc
    return hc.write_json(Path(p), obj)


def sha256(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ triangulation core (arrays only)

def _dlt_batch(P: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """P (N,V,3,4), x (N,V,2), w (N,V) 0/1 -> X (N,3) (hand_contact._dlt on the rows with w = 1; zero rows leave the null space alone)"""
    A1 = x[..., 0, None] * P[:, :, 2, :] - P[:, :, 0, :]
    A2 = x[..., 1, None] * P[:, :, 2, :] - P[:, :, 1, :]
    A = np.concatenate([A1 * w[..., None], A2 * w[..., None]], 1)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1]
    ok = np.abs(Xh[:, 3]) > 1e-12
    X = np.full((len(P), 3), np.nan)
    X[ok] = Xh[ok, :3] / Xh[ok, 3:4]
    return X


def _reproj(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """P (N,V,3,4), X (N,3) -> pixels (N,V,2) (inf behind / at the camera, as hand_contact._reproj)"""
    h = np.einsum("nvij,nj->nvi", P[..., :3], X) + P[..., 3]
    good = h[..., 2] > 1e-6
    uv = np.where(good[..., None], h[..., :2] / np.where(good, h[..., 2], 1.0)[..., None], np.inf)
    return uv


def _tri_chunk(args) -> dict:
    """one block of (frame, joint) problems: P (N,V,3,4), C (N,V,3), f (N,V), x (N,V,2), valid (N,V) -> X, nv, inl, err, S"""
    P, C, f, x, valid, inlier_px, min_angle, min_views, sigma_px, floor = args
    N, V = valid.shape
    best_n = np.zeros(N, int)
    best_m = np.full(N, np.inf)
    best_inl = np.zeros((N, V), bool)
    cos_min = np.cos(np.radians(min_angle))
    for a in range(V):
        for b in range(a + 1, V):
            m = valid[:, a] & valid[:, b]
            if not m.any():
                continue
            idx = np.flatnonzero(m)
            Xa = _dlt_batch(P[idx][:, [a, b]], x[idx][:, [a, b]], np.ones((len(idx), 2)))
            fin = np.isfinite(Xa).all(1)
            ra, rb = Xa - C[idx, a], Xa - C[idx, b]
            with np.errstate(invalid="ignore", divide="ignore"):
                cs = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1))
            # hand_contact: ang = degrees(arccos(clip(cos))); skip when ang < min_angle
            ang_ok = fin & (np.degrees(np.arccos(np.clip(np.nan_to_num(cs, nan=1.0), -1, 1))) >= min_angle)
            uv = _reproj(P[idx], np.nan_to_num(Xa))
            with np.errstate(invalid="ignore"):
                e = np.linalg.norm(uv - x[idx], axis=-1) / f[idx]
            inl = valid[idx] & (e < inlier_px) & ang_ok[:, None]
            n = inl.sum(1)
            em = np.where(inl, e, np.nan)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(em, 1)
            cand = ang_ok & (n >= min_views)
            better = cand & ((n > best_n[idx]) | ((n == best_n[idx]) & (med < best_m[idx])))
            # hand_contact keeps the first of equal scores: '>' on (n, -median)
            bi = idx[better]
            best_n[bi] = n[better]
            best_m[bi] = med[better]
            best_inl[bi] = inl[better]
    X = np.full((N, 3), np.nan)
    nv = np.zeros(N, np.int8)
    err = np.full(N, np.nan)
    S = np.full((N, 3, 3), np.nan)
    has = best_n >= min_views
    if has.any():
        idx = np.flatnonzero(has)
        w = best_inl[idx].astype(float)
        Xr = _dlt_batch(P[idx], x[idx], w)
        fin = np.isfinite(Xr).all(1)
        idx, w, Xr = idx[fin], w[fin], Xr[fin]
        uv = _reproj(P[idx], Xr)
        e = np.linalg.norm(uv - x[idx], axis=-1)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            er = np.nanmedian(np.where(w > 0, e, np.nan), 1)
        X[idx], nv[idx], err[idx] = Xr, w.sum(1).astype(np.int8), er
        # covariance: J_c = d(pixel)/dX = (P_row - pixel * P_row3) / h3, information sum_c J^T J / (sigma_px f_c)^2
        h = np.einsum("nvij,nj->nvi", P[idx][..., :3], Xr) + P[idx][..., 3]
        h2 = np.where(np.abs(h[..., 2]) > 1e-9, h[..., 2], 1e-9)
        u, v = h[..., 0] / h2, h[..., 1] / h2
        Pr = P[idx][..., :3]
        use = (w > 0)[..., None]
        with np.errstate(all="ignore"):
            Ju = np.where(use, (Pr[:, :, 0] - u[..., None] * Pr[:, :, 2]) / h2[..., None], 0.0)
            Jv = np.where(use, (Pr[:, :, 1] - v[..., None] * Pr[:, :, 2]) / h2[..., None], 0.0)
            sc = np.where(w > 0, w / np.maximum(sigma_px * f[idx], 1e-9) ** 2, 0.0)
        info = np.einsum("nv,nvi,nvj->nij", sc, Ju, Ju) + np.einsum("nv,nvi,nvj->nij", sc, Jv, Jv)
        info += np.eye(3)[None] * 1e-6
        Sg = np.linalg.inv(info)
        lam, Q = np.linalg.eigh(0.5 * (Sg + np.swapaxes(Sg, 1, 2)))
        lam = np.maximum(lam, floor ** 2)
        S[idx] = np.einsum("nij,nj,nkj->nik", Q, lam, Q)
        best_inl[idx] = w > 0
    best_inl[~np.isfinite(X).all(1)] = False
    return {"X": X, "nv": nv, "inl": best_inl, "err": err, "S": S}


def triangulate_all(P: np.ndarray, C: np.ndarray, f: np.ndarray, uv: np.ndarray, valid: np.ndarray, cfg: SidesCfg,
                    ref: np.ndarray | None = None, anchors: np.ndarray | None = None, reach: np.ndarray | None = None,
                    cams_mask: np.ndarray | None = None, min_angle: float | None = None, min_views: int = 2,
                    block: int = 600, workers: int = 1) -> dict:
    """Vectorised RANSAC triangulation of J joints over T frames from V views.

    P (V,T,3,4) grid projection matrices, C (V,T,3) camera centres, f (V,T) focal / 1000, uv (V,T,J,2) undistorted detections,
    valid (V,T,J) usable detections, ref (T,J,3) the base joints (identity gate), anchors (T,J,3) + reach (J,) the reach gate,
    cams_mask (T,V) views allowed (False = held out) -> {"X" (T,J,3), "nv" (T,J), "inl" (T,J,V), "err" (T,J) px, "S" (T,J,3,3)}"""
    V, T, J = valid.shape
    ok = valid.copy()
    if cams_mask is not None:
        ok &= np.asarray(cams_mask, bool).T[:, :, None]
    ang = cfg.min_angle_deg if min_angle is None else min_angle
    out = {"X": np.full((T, J, 3), np.nan), "nv": np.zeros((T, J), np.int8), "inl": np.zeros((T, J, V), bool),
           "err": np.full((T, J), np.nan), "S": np.full((T, J, 3, 3), np.nan)}
    jobs = []
    for a in range(0, T, block):
        b = min(T, a + block)
        n = b - a
        Pb = np.broadcast_to(np.swapaxes(P[:, a:b], 0, 1)[:, None], (n, J, V, 3, 4)).reshape(n * J, V, 3, 4)
        Cb = np.broadcast_to(np.swapaxes(C[:, a:b], 0, 1)[:, None], (n, J, V, 3)).reshape(n * J, V, 3)
        fb = np.broadcast_to(np.swapaxes(f[:, a:b], 0, 1)[:, None], (n, J, V)).reshape(n * J, V)
        xb = np.transpose(uv[:, a:b], (1, 2, 0, 3)).reshape(n * J, V, 2)
        vb = np.transpose(ok[:, a:b], (1, 2, 0)).reshape(n * J, V)
        jobs.append((a, b, (np.ascontiguousarray(Pb), np.ascontiguousarray(Cb), np.ascontiguousarray(fb), np.nan_to_num(xb), vb,
                            cfg.inlier_px, ang, min_views, cfg.sigma_px, cfg.sigma_floor_m)))
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_tri_chunk, [j[2] for j in jobs]))
    else:
        res = [_tri_chunk(j[2]) for j in jobs]
    for (a, b, _), r in zip(jobs, res):
        n = b - a
        out["X"][a:b] = r["X"].reshape(n, J, 3)
        out["nv"][a:b] = r["nv"].reshape(n, J)
        out["inl"][a:b] = r["inl"].reshape(n, J, V)
        out["err"][a:b] = r["err"].reshape(n, J)
        out["S"][a:b] = r["S"].reshape(n, J, 3, 3)
    bad = np.zeros((T, J), bool)
    if ref is not None:
        with np.errstate(invalid="ignore"):
            bad |= np.linalg.norm(out["X"] - ref, axis=-1) > cfg.identity_gate_m
    if anchors is not None and reach is not None:
        with np.errstate(invalid="ignore"):
            bad |= np.linalg.norm(out["X"] - anchors, axis=-1) > np.asarray(reach)[None]
    bad &= np.isfinite(out["X"]).all(-1)
    out["X"][bad] = np.nan
    out["nv"][bad] = 0
    out["inl"][bad] = False
    out["err"][bad] = np.nan
    out["S"][bad] = np.nan
    out["gated"] = bad
    return out


# ------------------------------------------------------------------------------------------------ COCO -> SMPL-X bias (17 joints)

def yaw_axes(J55: np.ndarray) -> np.ndarray:
    """(T,3,3) columns = the dancer's horizontal left, up, front (zouk_contacts.torso_frame)"""
    from .. import zouk_contacts as zc
    return zc.torso_frame(np.asarray(J55, float))


def free_frames(JA: np.ndarray, JB: np.ndarray, min_m: float) -> np.ndarray:
    d = JA[:, 0] - JB[:, 0]
    return np.hypot(d[:, 0], d[:, 2]) > min_m


def coco_bias17(X: np.ndarray, nv: np.ndarray, J55: np.ndarray, J55_other: np.ndarray, cfg: SidesCfg, seed: int = 0) -> dict:
    """median of (triangulated COCO - its SMPL-X joint) in the dancer's yaw frame on free frames with >= bias_min_views views
    -> {"bias" (17,3) [left, up, front] m, "n" (17,), "iqr_cm" (17,3)}"""
    ref = J55[:, list(geom.C2S)]
    free = free_frames(J55, J55_other, cfg.bias_free_m)
    Fr = yaw_axes(J55)
    rng = np.random.default_rng(seed)
    b = np.zeros((17, 3))
    n = np.zeros(17, int)
    iqr = np.zeros((17, 3))
    for j in range(17):
        ok = free & (nv[:, j] >= cfg.bias_min_views) & np.isfinite(X[:, j]).all(1)
        ks = np.flatnonzero(ok)
        if len(ks) > cfg.bias_max_frames:
            ks = np.sort(rng.choice(ks, cfg.bias_max_frames, replace=False))
        if len(ks) < 20:
            continue
        loc = np.einsum("kji,kj->ki", Fr[ks], X[ks, j] - ref[ks, j])
        b[j] = np.median(loc, 0)
        n[j] = len(ks)
        iqr[j] = (np.percentile(loc, 75, 0) - np.percentile(loc, 25, 0)) * 100
    return {"bias": b, "n": n, "iqr_cm": iqr}


def tri_yaw_axes(X: np.ndarray, nv: np.ndarray, J55: np.ndarray, pair: tuple = (5, 6), wmin: float = 0.15) -> np.ndarray:
    """(T,3,3) the dancer's yaw frame from the TRIANGULATED joint pair (shoulders 5/6 or hips 11/12) where both are seen, else from the
    base (zouk_contacts.torso_frame)"""
    Fr = yaw_axes(J55)
    up = np.array([0.0, 1.0, 0.0])
    a, b = pair
    d = X[:, a] - X[:, b]
    d[:, 1] = 0
    w = np.linalg.norm(d, axis=1)
    ok = (nv[:, a] >= 2) & (nv[:, b] >= 2) & np.isfinite(w) & (w > wmin)
    x = d[ok] / w[ok, None]
    Fr[ok] = np.stack([x, np.tile(up, (len(x), 1)), np.cross(x, up)], 2)
    return Fr


def debias(X: np.ndarray, J55: np.ndarray, bias: np.ndarray, nv: np.ndarray | None = None) -> np.ndarray:
    """X - the per-joint bias rotated into the dancer's yaw frame. With nv: the frame of the joint's own triangulated line (hips, knees,
    ankles: the hip line; the rest: the shoulder line), so the outward hip bias never rotates the hip line - in the base's frame it did
    wherever the base heading is wrong (recap: hip-line heading p90 42 deg debiased in the base frame, 23 deg raw)"""
    b = np.asarray(bias, float)
    if nv is None:
        return X - np.einsum("kij,pj->kpi", yaw_axes(J55), b)
    Fs = tri_yaw_axes(X, nv, J55, (5, 6), 0.15)
    Fh = tri_yaw_axes(X, nv, J55, (11, 12), 0.08)
    out = X - np.einsum("kij,pj->kpi", Fs, b)
    out[:, 11:] = X[:, 11:] - np.einsum("kij,pj->kpi", Fh, b[11:])
    return out


def smooth_all(X: np.ndarray, nv: np.ndarray, min_views: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """hand_contact.smooth_track per joint (5-frame median, gaps <= 8 frames bridged, Gaussian sigma 2) -> (Xs, valid)"""
    from .. import hand_contact as hc
    Xs = np.full_like(X, np.nan)
    val = np.zeros(X.shape[:2], bool)
    for j in range(X.shape[1]):
        x = np.where((nv[:, j] >= min_views)[:, None], X[:, j], np.nan)
        if np.isfinite(x).all(1).sum() < 3:
            continue
        y, v = hc.smooth_track(x)
        Xs[:, j] = np.where(v[:, None], y, np.nan)
        val[:, j] = v
    return Xs, val


# ------------------------------------------------------------------------------------------------ capsules, sites, visibility

def capsule_set(betas: dict, cache: Path | None = None) -> dict:
    """{role: capsule body} of zouk_contacts.fit_capsules (the shaped rest mesh), cached in `cache` keyed by the betas"""
    from .. import zouk_contacts as zc
    key = {r: [round(float(x), 8) for x in np.asarray(b, float).ravel()] for r, b in betas.items()}
    if cache is not None and cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("betas") == key:
            return {r: zc._caps_from_json(d["bodies"][r]) for r in betas}
    bodies = {r: zc.fit_capsules(np.asarray(b, float)) for r, b in betas.items()}
    if cache is not None:
        write_json(cache, {"what": "capsule bodies (zouk_contacts.fit_capsules) of each dancer's shaped rest mesh", "betas": key,
                           "bodies": {r: zc._jsonable_caps(b) for r, b in bodies.items()}})
    return bodies


def site_set(betas: dict, cache: Path | None = None) -> dict:
    """{role: {site: {"joint", "p", "n", ...}}} of zouk_contacts.fit_sites, cached like capsule_set"""
    from .. import zouk_contacts as zc
    key = {r: [round(float(x), 8) for x in np.asarray(b, float).ravel()] for r, b in betas.items()}
    if cache is not None and cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("betas") == key:
            return d["sites"]
    sites = {r: zc.fit_sites(np.asarray(b, float)) for r, b in betas.items()}
    if cache is not None:
        write_json(cache, {"what": "contact sites (zouk_contacts.fit_sites)", "betas": key, "sites": sites})
    return sites


def rest_joints(betas: dict, cache: Path | None = None) -> dict:
    """{role: (55,3) joints of the shaped rest pose} (hand_contact.Body at pose 0), cached like capsule_set"""
    key = {r: [round(float(x), 8) for x in np.asarray(b, float).ravel()] for r, b in betas.items()}
    if cache is not None and cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("betas") == key:
            return {r: np.asarray(v, float) for r, v in d["joints"].items()}
    from .. import hand_contact as hc
    out = {}
    for r, b in betas.items():
        body = hc.Body({"betas": np.asarray(b, float), "t": np.zeros(1)}, "cpu")
        out[r] = body.forward_np(np.zeros((1, 55, 3)), np.zeros((1, 3)), verts=False)["joints"][0]
    if cache is not None:
        write_json(cache, {"what": "rest-pose joints (pose 0) of each dancer's shaped SMPL-X body", "betas": key,
                           "joints": {r: v.tolist() for r, v in out.items()}})
    return out


def body_capsule_index(caps: dict) -> np.ndarray:
    """the capsules that stand for the body (the finger phalanges, which need fingertips, are left out)"""
    import re
    return np.array([i for i, n in enumerate(caps["names"]) if not re.match(r"(index|middle|ring|pinky|thumb)\d", n)], int)


def posed_capsules(caps: dict, J55: np.ndarray, Rg22: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """capsule end points (T,K,3) x 2, radii (K,), parts [K] of the body capsules posed on joints (T,55,3) + world rotations (T,22,3,3)"""
    from .. import zouk_contacts as zc
    idx = body_capsule_index(caps)
    sub = {"A": [caps["A"][i] for i in idx], "B": [caps["B"][i] for i in idx]}
    R = np.zeros((len(J55), 55, 3, 3))
    R[:, :22] = Rg22
    R[:, 22:] = np.eye(3)
    A, B = zc.capsule_points(sub, J55, R, np.zeros((len(J55), 10, 3)))
    return A, B, np.asarray(caps["r"], float)[idx], [caps["parts"][i] for i in idx]


# own capsule parts that never hide a joint (the joint sits on / in them)
_OWN_EXCL = {0: ("neck_head",), 1: ("neck_head",), 2: ("neck_head",), 3: ("neck_head",), 4: ("neck_head",),
             5: ("torso", "shoulder_L", "upper_arm_L"), 6: ("torso", "shoulder_R", "upper_arm_R"),
             7: ("upper_arm_L", "forearm_L"), 8: ("upper_arm_R", "forearm_R"), 9: ("forearm_L", "hand_L"), 10: ("forearm_R", "hand_R"),
             11: ("torso", "leg_L"), 12: ("torso", "leg_R"), 13: ("leg_L",), 14: ("leg_R",), 15: ("leg_L",), 16: ("leg_R",)}
PALM_EXCL = {"left": ("forearm_L", "hand_L"), "right": ("forearm_R", "hand_R")}


def exclusion(parts_own: list, parts_other: list, excl_own: list[tuple]) -> np.ndarray:
    """(P, K_own + K_other) True where capsule k cannot hide point p (its own parts in excl_own[p])"""
    P = len(excl_own)
    E = np.zeros((P, len(parts_own) + len(parts_other)), bool)
    for p, ex in enumerate(excl_own):
        for k, pt in enumerate(parts_own):
            if pt in ex:
                E[p, k] = True
    return E


def ray_blocked(Cc: np.ndarray, X: np.ndarray, OA: np.ndarray, OB: np.ndarray, r: np.ndarray, margin: float,
                excl: np.ndarray | None = None, block: int = 1500) -> np.ndarray:
    """camera centres (T,3), points (T,P,3), occluder capsules (T,K,3) x 2 + radii (K,) -> (T,P): the segment camera -> point (stopped
    `margin` short of it) passes through a capsule that is not excluded for that point (excl (P,K))"""
    from .. import zouk_contacts as zc
    T, P = X.shape[:2]
    out = np.zeros((T, P), bool)
    for a in range(0, T, block):
        b = min(T, a + block)
        d = X[a:b] - Cc[a:b, None]
        L = np.linalg.norm(d, axis=-1, keepdims=True)
        Xs = Cc[a:b, None] + d * np.maximum(L - margin, 0) / np.maximum(L, 1e-9)
        dist, _, _ = zc.seg_seg(Cc[a:b, None, None], Xs[:, :, None], OA[a:b, None], OB[a:b, None])
        hit = dist < r[None, None]
        if excl is not None:
            hit &= ~excl[None]
        out[a:b] = hit.any(-1)
    return out


def visible(obs: dict, cams: list, X: np.ndarray, OA: np.ndarray, OB: np.ndarray, r: np.ndarray, excl: np.ndarray | None,
            margin: float = 0.035, frames: np.ndarray | None = None) -> np.ndarray:
    """(T,P,V) camera v sees point X[t,p]: the camera has the frame, the point is in front of it, the ray is not blocked"""
    T, P = X.shape[:2]
    ks = np.arange(T) if frames is None else np.asarray(frames)
    out = np.zeros((len(ks), P, len(cams)), bool)
    fin = np.isfinite(X).all(-1)
    Xn = np.nan_to_num(X)
    for v, c in enumerate(cams):
        o = obs[c]
        Cc = o["C"][ks]
        blk = ray_blocked(Cc, Xn, OA, OB, r, margin, excl)
        Xc = np.einsum("kij,kpj->kpi", o["R"][ks], Xn) + o["t"][ks][:, None]
        out[:, :, v] = o["ok"][ks][:, None] & (Xc[..., 2] > 0.1) & ~blk & fin
    return out


# ------------------------------------------------------------------------------------------------ per-view bits

def face_bits(conf: np.ndarray, det: np.ndarray, cfg: SidesCfg) -> np.ndarray:
    """conf (T,17), det (T,) -> (T,) +1 face seen (nose >= 0.7 and an eye >= 0.6), -1 hidden (nose and eyes < 0.2), 0 no vote; only where
    the body is detected (person conf and the mean of shoulders / elbows / wrists / hips >= 0.5)"""
    body = (det >= cfg.det_conf) & (conf[:, 5:13].mean(1) >= 0.5)
    nose, eye = conf[:, 0], np.maximum(conf[:, 1], conf[:, 2])
    seen = (nose >= cfg.face_nose) & (eye >= cfg.face_eye)
    hid = (nose < cfg.face_hidden) & (eye < cfg.face_hidden)
    out = np.zeros(len(conf), np.int8)
    out[body & seen] = 1
    out[body & hid] = -1
    return out


def lr_bits(uv: np.ndarray, conf: np.ndarray, det: np.ndarray, cfg: SidesCfg) -> np.ndarray:
    """jl_facing.lr_rho of the detected shoulders and hips: > 0 = the body's left on the image's right = front to the camera (NaN when
    any of the four is missing)"""
    ls, rs, lh, rh = uv[:, 5], uv[:, 6], uv[:, 11], uv[:, 12]
    torso = np.linalg.norm(0.5 * (ls + rs) - 0.5 * (lh + rh), axis=-1)
    dx = 0.5 * ((ls[:, 0] - rs[:, 0]) + (lh[:, 0] - rh[:, 0]))
    rho = dx / np.maximum(torso, 1e-6)
    ok = (det >= cfg.det_conf) & (conf[:, [5, 6, 11, 12]] >= 0.5).all(1) & np.isfinite(rho)
    return np.where(ok, rho, np.nan)


def seg_cross_2d(a0, a1, b0, b1) -> tuple[np.ndarray, np.ndarray]:
    """2D segments (n,2) -> (they properly cross, the 2D cross product of their directions)"""
    def cr(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    da, db = a1 - a0, b1 - b0
    c = cr(da, db)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = cr(b0 - a0, db) / c
        t = cr(b0 - a0, da) / c
    hit = np.isfinite(s) & np.isfinite(t) & (s > 0) & (s < 1) & (t > 0) & (t < 1) & (np.abs(c) > 1e-9)
    return hit, c


def crossing_votes(uvA: np.ndarray, cfA: np.ndarray, uvB: np.ndarray, cfB: np.ndarray, gap: float = 0.05) -> np.ndarray:
    """per view and frame, the 1-bit vote on the 3D side s3D = sign((u x v).(A0 - B0)) of segments A = (A0, A1), B = (B0, B1)
    (relations.md 2.2): where their images cross, sigma_c = sign(n . w_c) = -sign(2D cross of the pixel directions) (pixel y down)
    and o_c = +1 when A is in front (its two joints' mean COCO confidence higher by > gap), -1 when B is -> vote o_c sigma_c in
    {-1, 0, +1}. uvX (n,2,2) the two end points in pixels, cfX (n,2) their confidences."""
    hit, c = seg_cross_2d(uvA[:, 0], uvA[:, 1], uvB[:, 0], uvB[:, 1])
    sig = -np.sign(c)
    dc = np.nanmean(cfA, 1) - np.nanmean(cfB, 1)
    o = np.where(dc > gap, 1, np.where(dc < -gap, -1, 0))
    return np.where(hit, o * sig, 0).astype(np.int8)


def cue_reliability(votes: np.ndarray, labels: np.ndarray, prior: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """votes (T,V) in {-1,0,1}, labels (T,) in {-1,0,1} (0 = no label) -> (q (V,), n (V,)): the share of votes that agree with the
    label (Beta(prior, prior) smoothed)"""
    lab = np.asarray(labels)[:, None]
    m = (votes != 0) & (lab != 0)
    agree = ((votes == lab) & m).sum(0)
    n = m.sum(0)
    return (agree + prior) / (n + 2 * prior), n


def missing_term(detected: np.ndarray, vis: np.ndarray, p_det_vis: np.ndarray, p_det_hid: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """log P(det | the hypothesis' visibility) per view: detected (n,V) bool, vis (n,V) bool predicted visible, rates (V,)"""
    a = np.clip(np.asarray(p_det_vis, float), floor, 1 - floor)[None]
    b = np.clip(np.asarray(p_det_hid, float), floor, 1 - floor)[None]
    p = np.where(vis, a, b)
    return np.where(detected, np.log(p), np.log(1 - p))


def heldout_plan(t: np.ndarray, seen: np.ndarray, cfg: SidesCfg) -> np.ndarray:
    """seen (T,V) the dancer detected per view -> (T,) index of the view held out of the solve (-1 = none): in each block of
    heldout_block_s where >= heldout_min_cams views see the dancer (on >= heldout_min_share of its frames), view seen[block % n]"""
    blk = np.floor(np.asarray(t) / cfg.heldout_block_s).astype(int)
    out = np.full(len(t), -1, np.int8)
    for b in np.unique(blk):
        m = blk == b
        share = seen[m].mean(0)
        cams = np.flatnonzero(share >= cfg.heldout_min_share)
        if len(cams) >= cfg.heldout_min_cams:
            out[m] = cams[int(b) % len(cams)]
    return out


def part_means(vis: np.ndarray, parts: dict) -> np.ndarray:
    """MAMMA visibility (T,2,512) -> (T,2,len(PARTS)) mean per part (body / hands / feet / head)"""
    out = np.zeros(vis.shape[:2] + (len(PARTS),), np.float32)
    for i, p in enumerate(PARTS):
        if p in parts:
            out[..., i] = vis[..., parts[p]].astype(np.float32).mean(-1)
    return out


# ------------------------------------------------------------------------------------------------ the evidence step

def _out_dir(ad: MultiView) -> Path:
    return ad.work / "sides" / "evidence"


def run(take: str, cfg: SidesCfg | None = None, out_dir: Path | None = None, t_range: tuple | None = None, workers: int | None = None) -> dict:
    """the evidence step of a take -> work/<take>/sides/evidence/ (see the module docstring); t_range = (t0, t1) dance seconds limits
    the triangulated frames (a development run; the arrays keep the full grid, NaN outside)"""
    from .. import hand_contact as hc
    t_start = time.time()
    cfg = cfg or load_cfg(take)
    workers = int(workers or cfg.workers or 1)
    ad = adapter_for(take, cfg)
    out_dir = Path(out_dir or _out_dir(ad))
    out_dir.mkdir(parents=True, exist_ok=True)
    roles = list(cfg.roles)
    base = ad.base()
    T, tg = ad.T, ad.t
    cams = ad.cams
    V = len(cams)
    sel = np.ones(T, bool) if t_range is None else (tg >= t_range[0]) & (tg < t_range[1])
    ks = np.flatnonzero(sel)
    log(f"{take}: {ad.kind}, cams {cams}, T {T}, frames {len(ks)}, workers {workers}")
    obs = ad.obs()
    coco = ad.coco_tracks()
    log("obs + COCO synced", round(time.time() - t_start, 1), "s")
    rep = {"take": take, "kind": ad.kind, "cams": cams, "T": T, "frames": [int(ks[0]), int(ks[-1])] if len(ks) else None,
           "cfg": cfg.to_json(), "base": str(ad.base_dir), "inputs_sha256": {r: sha256(ad.base_dir / f"{r}.npz") for r in roles},
           "obs_dir": str(ad.obs_dir()), "sync": "owning chunk video offsets (segment_owner + mamma_dyn.apply_offset)",
           "sync_dt_ms_p95": {c: float(np.percentile(coco[c]["dt"][np.isfinite(coco[c]["dt"])], 95) * 1000) if np.isfinite(coco[c]["dt"]).any()
                              else None for c in cams}}
    rep["role_check"] = ad.role_swap_check()
    log("role check", {c: v["stored_better"] for c, v in rep["role_check"].items()})
    # poses, capsules, sites
    J55 = {r: np.asarray(base[r]["joints"], float) for r in roles}
    Rg = {r: geom.global_rotations(base[r]["global_orient"], base[r]["body_pose"]) for r in roles}
    betas = {r: np.asarray(base[r]["betas"], float) for r in roles}
    caps = capsule_set(betas, out_dir / "capsules.json")
    site_set(betas, out_dir / "sites.json")
    rest_joints(betas, out_dir / "rest.json")
    posed ={r: posed_capsules(caps[r], J55[r], Rg[r]) for r in roles}
    log("capsules + sites", round(time.time() - t_start, 1), "s")
    # per-view arrays
    Pm = np.stack([ad.proj_mats(c) for c in cams])                       # (V,T,3,4)
    Cc = np.stack([obs[c]["C"] for c in cams])
    fc = np.stack([obs[c]["f"] for c in cams])
    try:
        parts = hc.landmark_parts()
    except Exception as e:  # pragma: no cover - the part file is missing
        log("no MAMMA part index:", e)
        parts = {}
    tri = {}
    vis_pred = {}
    for r in roles:
        other = [x for x in roles if x != r][0]
        p = PIDX[r]
        uv = np.stack([coco[c]["uv"][:, p] for c in cams])               # (V,T,17,2)
        valid = np.stack([ad.coco_valid(c, r) for c in cams])            # (V,T,17)
        valid &= sel[None, :, None]
        ref = J55[r][:, list(geom.C2S)]
        anchors = np.stack([J55[r][:, geom.REACH[j][0]] for j in range(17)], 1)
        reach = np.array([geom.REACH[j][1] for j in range(17)])
        t0 = time.time()
        first = triangulate_all(Pm, Cc, fc, uv, valid, cfg, ref, anchors, reach, workers=workers)
        log(r, "triangulated (all detections)", round(time.time() - t0, 1), "s; joints with >= 2 views",
            int((first["nv"] >= 2).sum()), ">= 3", int((first["nv"] >= 3).sum()))
        # visibility of each joint in each view under the base (both dancers' capsules, own limb excluded)
        A1, B1, r1, pt1 = posed[r]
        A2, B2, r2, pt2 = posed[other]
        OA, OB, Orad = np.concatenate([A1, A2], 1), np.concatenate([B1, B2], 1), np.concatenate([r1, r2]) * cfg.occl_r_scale
        ex = exclusion(pt1, pt2, [_OWN_EXCL[j] for j in range(17)])
        t0 = time.time()
        vp = np.zeros((T, 17, V), bool)
        vp[ks] = visible(obs, cams, ref[ks], OA[ks], OB[ks], Orad, ex, cfg.occl_margin_m, frames=ks)
        palm = np.stack([geom.palm_from_joints(J55[r], s) for s in ("left", "right")], 1)
        exp_ = exclusion(pt1, pt2, [PALM_EXCL["left"], PALM_EXCL["right"]])
        vpalm = np.zeros((T, 2, V), bool)
        vpalm[ks] = visible(obs, cams, palm[ks], OA[ks], OB[ks], Orad, exp_, cfg.occl_margin_m, frames=ks)
        vis_pred[r] = (vp, vpalm)
        log(r, "visibility", round(time.time() - t0, 1), "s; predicted visible share", round(float(vp[ks].mean()), 3))
        # the consistent pass tests each view at the TRIANGULATED point (all detections) where there is one, else at the base joint:
        # testing the base joint drops good views exactly where the base is wrong (recap 57.5 s: cam 00 / 06 dropped, a 2-view point
        # 45 cm off the 3-view one)
        if cfg.vis_point == "tri":
            P0 = np.where(np.isfinite(first["X"]).all(-1)[..., None], first["X"], ref)
            vt = np.zeros((T, 17, V), bool)
            vt[ks] = visible(obs, cams, P0[ks], OA[ks], OB[ks], Orad, ex, cfg.occl_margin_m, frames=ks)
        else:
            vt = vp
        if cfg.vis_consistent:
            valid2 = valid & np.transpose(vt, (2, 0, 1))
            res = triangulate_all(Pm, Cc, fc, uv, valid2, cfg, ref, anchors, reach, workers=workers)
            dropped = np.transpose(valid & ~valid2, (1, 2, 0))
        else:
            res, dropped = first, np.zeros((T, 17, V), bool)
        # held-out phone (its COCO stays out of the solve's terms and targets)
        seen = np.stack([(coco[c]["det"][:, p] >= cfg.det_conf) & obs[c]["ok"] for c in cams], 1)
        ho = heldout_plan(tg, seen, cfg)
        mask = np.ones((T, V), bool)
        mask[np.arange(T)[ho >= 0], ho[ho >= 0]] = False
        res_ho = triangulate_all(Pm, Cc, fc, uv, (valid & np.transpose(vt, (2, 0, 1))) if cfg.vis_consistent else valid, cfg, ref,
                                 anchors, reach, cams_mask=mask, workers=workers)
        tri[r] = {"res": res, "first": first, "dropped": dropped, "ho": ho, "res_ho": res_ho}
        log(r, "re-triangulated + held-out", round(time.time() - t0, 1), "s")
    # bias + debiased + smoothed, written per role
    bias_rep = {}
    for r in roles:
        other = [x for x in roles if x != r][0]
        res = tri[r]["res"]
        b = coco_bias17(res["X"], res["nv"], J55[r], J55[other], cfg)
        bias_rep[r] = b
        for tag, rr in (("", res), ("_ho", tri[r]["res_ho"])):
            Xd = debias(rr["X"], J55[r], b["bias"], rr["nv"])
            Xs, vs = smooth_all(Xd, rr["nv"])
            d = {"t": tg, "cams": np.array(cams), "X": rr["X"].astype(np.float32), "Xd": Xd.astype(np.float32), "Xs": Xs.astype(np.float32),
                 "valid_s": vs, "S": rr["S"].astype(np.float32), "nv": rr["nv"], "inl": rr["inl"], "err": rr["err"].astype(np.float32),
                 "ho": tri[r]["ho"]}
            if tag == "":
                d.update(X_all=tri[r]["first"]["X"].astype(np.float32), nv_all=tri[r]["first"]["nv"], dropped=tri[r]["dropped"])
            np.savez_compressed(out_dir / f"tri_{r}{tag}.npz", **d)
    write_json(out_dir / "coco_bias.json", {"what": "triangulated COCO minus its SMPL-X joint (geom.C2S) in the dancer's yaw frame "
                                                    "[left, up, front] m, median on free frames (pelvises > bias_free_m apart, >= "
                                                    "bias_min_views views); subtracted in Xd / Xs", "joints": list(geom.COCO),
                                            "bias": {r: v["bias"] for r, v in bias_rep.items()}, "n": {r: v["n"] for r, v in bias_rep.items()},
                                            "iqr_cm": {r: v["iqr_cm"] for r, v in bias_rep.items()}})
    # per-view files
    for v, c in enumerate(cams):
        co = coco[c]
        face = np.stack([face_bits(co["conf"][:, PIDX[r]], co["det"][:, PIDX[r]], cfg) for r in roles], 1)
        lr = np.stack([lr_bits(co["uv"][:, PIDX[r]], co["conf"][:, PIDX[r]], co["det"][:, PIDX[r]], cfg) for r in roles], 1)
        pv = part_means(obs[c]["vis"], parts)
        np.savez_compressed(out_dir / f"views_{c}.npz", uv=co["uv"], conf=co["conf"].astype(np.float16), det=co["det"].astype(np.float16),
                            face=face, face_conf=co["conf"][:, :, :5].astype(np.float16), lr=lr.astype(np.float16), part_vis=pv.astype(np.float16),
                            parts=np.array(PARTS), vis_pred=np.stack([vis_pred[r][0][:, :, v] for r in roles], 1),
                            vis_palm=np.stack([vis_pred[r][1][:, :, v] for r in roles], 1), ok=obs[c]["ok"], C=obs[c]["C"].astype(np.float32),
                            f=obs[c]["f"].astype(np.float32))
    # coverage
    cov = {}
    for r in roles:
        res = tri[r]["res"]
        cov[r] = {"ge2": {geom.COCO[j]: int((res["nv"][ks, j] >= 2).sum()) for j in range(17)},
                  "ge3": {geom.COCO[j]: int((res["nv"][ks, j] >= 3).sum()) for j in range(17)},
                  "dropped_as_hidden": int(tri[r]["dropped"][ks].sum()),
                  "err_px_p50": float(np.nanmedian(res["err"][ks])) if np.isfinite(res["err"][ks]).any() else None,
                  "heldout_frames": int((tri[r]["ho"][ks] >= 0).sum())}
    rep["coverage"] = cov
    rep["bias_n"] = {r: v["n"] for r, v in bias_rep.items()}
    rep["files"] = sorted(p.name for p in out_dir.iterdir())
    rep["seconds"] = round(time.time() - t_start, 1)
    write_json(out_dir / "evidence.json", rep)
    log("done", rep["seconds"], "s")
    return rep


def load_tri(work: Path, role: str, heldout: bool = False) -> dict:
    p = Path(work) / "sides" / "evidence" / f"tri_{role}{'_ho' if heldout else ''}.npz"
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_views(work: Path, cams: list) -> dict:
    out = {}
    for c in cams:
        with np.load(Path(work) / "sides" / "evidence" / f"views_{c}.npz", allow_pickle=False) as z:
            out[c] = {k: z[k] for k in z.files}
    return out
