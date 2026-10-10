"""Attach a camera that lives in a SUB-MODEL frame to the room frame (rotation only), checked on PEOPLE without identity labels.

The two recap phones that face the Dance Precisions wall (05, 06) see none of the room model's main images: they are localised
in the demo's sub-models (cameras/sfm_work/sub_05, sub_06), in a frame of their own. That frame is not metric, and its camera
CENTRES are unreliable (the 616-point sub-model of 05 lets the centre of a phone that is held in one place wander by 16 units; the
orientation is good: 0.03-0.2 deg between chunks). The demo attached its own 05 / 06 by ROTATION ONLY: one constant rotation Rs from
the sub frame into the room frame and ONE fixed centre per phone, solved from the two dancers' identity-labelled keypoints. The recap
has no labels yet, and its phones 05 / 06 stand where the demo's did, so the attach is
    R_room(t) = R_sub(t) Rt^T ,  C_room = C        with (Rt, C) = the demo's attach of the same sub-model (demo_prior)
and it is CHECKED on the recap's own people, with no identities: people are triangulated from the main cameras (00-03, which are
localised directly in the room model; every pair matches its people by the epipolar cost and triangulates), projected through the
attached camera, and matched to the detected people of the attached view. The share of visible 3D people that land within a few
pixels of a detected person is the evidence (reported with the same share for random poses and for the prior moved by 0.5 m / 2 deg:
it must collapse). A refinement on that score is adopted only if it improves the HELD-OUT views.

    python -m dancecap.cameras.attach_people 05 [--step 1.5]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Ro

TAU_PX = 25.0           # truncation of one joint's epipolar distance
MIN_JOINTS = 6          # confident joints in both persons for a pair to count
CONF = 0.5


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def undistort_norm(xy: np.ndarray, K: np.ndarray, k1: float) -> np.ndarray:
    d = (np.asarray(xy, float) - [K[0, 2], K[1, 2]]) / K[0, 0]
    u = d.copy()
    for _ in range(6):
        u = d / (1 + k1 * (u ** 2).sum(-1, keepdims=True))
    return u


@dataclass
class Frame:
    """The people of one (camera, time): normalised homogeneous joints (P, 17, 3) and a validity mask (P, 17)."""
    xh: np.ndarray
    valid: np.ndarray
    f: float


def make_frame(persons: np.ndarray, K: np.ndarray, k1: float) -> Frame | None:
    keep = (persons[:, :, 2] >= CONF).sum(1) >= MIN_JOINTS
    if not keep.any():
        return None
    p = persons[keep]
    n = undistort_norm(p[:, :, :2].reshape(-1, 2).astype(float), K, k1).reshape(len(p), 17, 2)
    return Frame(np.concatenate([n, np.ones((len(p), 17, 1))], -1), p[:, :, 2] >= CONF, float(K[0, 0]))


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def pair_costs(E: np.ndarray, fs: Frame, fa: Frame) -> np.ndarray:
    """(Ps, Pa) mean truncated epipolar distance (px) of every person pair; +inf where fewer than MIN_JOINTS joints are shared.
    E maps rays of view s to epipolar lines in view a: x_a^T E x_s = 0."""
    la = fs.xh @ E.T                          # (Ps,17,3) epipolar lines in A
    ls = fa.xh @ E                            # (Pa,17,3) epipolar lines in S  (E^T x_a)
    na = np.maximum(np.hypot(la[..., 0], la[..., 1]), 1e-12)
    ns = np.maximum(np.hypot(ls[..., 0], ls[..., 1]), 1e-12)
    num = np.abs(np.einsum("qkc,pkc->pqk", fa.xh, la))         # (Ps,Pa,17)
    d = 0.5 * (num / na[:, None, :] * fa.f + num / ns[None, :, :] * fs.f)
    d = np.minimum(d, TAU_PX)
    v = fs.valid[:, None, :] & fa.valid[None, :, :]
    cnt = v.sum(-1)
    c = (d * v).sum(-1) / np.maximum(cnt, 1)
    return np.where(cnt >= MIN_JOINTS, c, np.inf)


# ------------------------------------------------------------------------------------------------ 3D people of the main cameras


@dataclass
class People3D:
    """People triangulated from the main cameras at one time (world metres): X (P,17,3), ok (P,17) joints seen by both views."""
    t: float
    X: np.ndarray
    ok: np.ndarray


def triangulate_people(mains: dict, main_people: dict, times: np.ndarray, pair_px: float = 6.0, max_gap: float = 0.04,
                       dedupe_m: float = 0.35) -> list[People3D]:
    """Identity-free 3D people at `times`: every pair of main cameras matches its people by the epipolar cost (< pair_px) and
    triangulates the joints both views see; a person found by several pairs is kept once (feet closer than dedupe_m)."""
    out = []
    names = sorted(mains)
    for t in times:
        idx = {}
        for c in names:
            A = mains[c]
            j = int(np.argmin(np.abs(A["t"] - t)))
            if abs(A["t"][j] - t) <= max_gap and A["ok"][j] and int(A["frame_index"][j]) < len(main_people[c]):
                idx[c] = j
        people = []
        for ai in range(len(names)):
            for bi in range(ai + 1, len(names)):
                a, b = names[ai], names[bi]
                if a not in idx or b not in idx:
                    continue
                A, B, ja, jb = mains[a], mains[b], idx[a], idx[b]
                ppA, ppB = main_people[a][int(A["frame_index"][ja])], main_people[b][int(B["frame_index"][jb])]
                if len(ppA) == 0 or len(ppB) == 0:
                    continue
                Ra, ta, Ka, ka = A["R"][ja], A["tvec"][ja], A["K"][ja], float(A["dist"][ja])
                Rb, tb, Kb, kb = B["R"][jb], B["tvec"][jb], B["K"][jb], float(B["dist"][jb])
                fa, fb = make_frame(ppA, Ka, ka), make_frame(ppB, Kb, kb)
                if fa is None or fb is None:
                    continue
                Rab = Rb @ Ra.T
                E = skew(tb - Rab @ ta) @ Rab
                c_ = pair_costs(E, fa, fb)
                pa = ppA[(ppA[:, :, 2] >= CONF).sum(1) >= MIN_JOINTS]
                pb = ppB[(ppB[:, :, 2] >= CONF).sum(1) >= MIN_JOINTS]
                for p in range(len(pa)):
                    q = int(np.argmin(c_[p]))
                    if not np.isfinite(c_[p, q]) or c_[p, q] > pair_px:
                        continue
                    both = (pa[p, :, 2] >= CONF) & (pb[q, :, 2] >= CONF)
                    if both.sum() < 8:
                        continue
                    X = np.full((17, 3), np.nan)
                    X[both] = triangulate_pair(Ra, ta, Ka, ka, pa[p, both, :2].astype(float), Rb, tb, Kb, kb, pb[q, both, :2].astype(float))
                    if np.isfinite(X[:, 1]).sum() >= 8 and -0.5 < np.nanmin(X[:, 1]) and np.nanmax(X[:, 1]) < 2.6:
                        people.append((float(both.sum()), X, both))
        people.sort(key=lambda r: -r[0])
        kept = []
        for _, X, both in people:
            foot = np.nanmean(X[[15, 16] if both[15] or both[16] else list(np.flatnonzero(both))], axis=0)[[0, 2]]
            if all(np.linalg.norm(foot - f) > dedupe_m for f, _, _ in kept):
                kept.append((foot, X, both))
        if kept:
            out.append(People3D(float(t), np.array([k[1] for k in kept]), np.array([k[2] for k in kept])))
    return out


def project_world(X: np.ndarray, R: np.ndarray, C: np.ndarray, K: np.ndarray, k1: float) -> tuple[np.ndarray, np.ndarray]:
    """(uv, depth) of world points X (...,3) through a camera (R world->cam, centre C)."""
    Xc = (X - C) @ R.T
    z = Xc[..., 2]
    zz = np.where(z > 1e-6, z, 1e-6)
    x, y = Xc[..., 0] / zz, Xc[..., 1] / zz
    d = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], -1), z


@dataclass
class View:
    """The attached camera at one time: sub-frame rotation, intrinsics, detected people (P,17,3 px) and the matching 3D people."""
    t: float
    i: int
    Rs: np.ndarray
    K: np.ndarray
    k1: float
    det: np.ndarray
    p3: People3D
    size: tuple


def build_views(S: dict, S_people: list, p3: list[People3D], max_gap: float = 0.04) -> list[View]:
    out = []
    size = (int(S["native_size"][0]), int(S["native_size"][1]))
    for q in p3:
        i = int(np.argmin(np.abs(S["t"] - q.t)))
        if abs(S["t"][i] - q.t) > max_gap or not S["ok"][i]:
            continue
        n = int(S["frame_index"][i])
        if n >= len(S_people) or len(S_people[n]) == 0:
            continue
        out.append(View(q.t, i, S["R"][i], S["K"][i], float(S["dist"][i]), S_people[n], q, size))
    return out


def triangulate_pair(Ra, ta, Ka, ka, xa, Rb, tb, Kb, kb, xb) -> np.ndarray:
    """DLT of corresponding joints (N,2 px each) from two cameras (cam_from_world R, t); returns (N,3)."""
    ua, ub = undistort_norm(xa, Ka, ka), undistort_norm(xb, Kb, kb)
    Pa, Pb = np.hstack([Ra, ta[:, None]]), np.hstack([Rb, tb[:, None]])
    A = np.stack([ua[:, :1] * Pa[2] - Pa[0], ua[:, 1:] * Pa[2] - Pa[1], ub[:, :1] * Pb[2] - Pb[0], ub[:, 1:] * Pb[2] - Pb[1]], 1)
    _, _, vt = np.linalg.svd(A)
    X = vt[:, -1]
    return X[:, :3] / np.where(np.abs(X[:, 3:]) < 1e-12, 1e-12, X[:, 3:])


def triangulation_stats(poseA: tuple, poseB: tuple, ppA: np.ndarray, ppB: np.ndarray, max_pair_px: float = 8.0) -> list[dict]:
    """People matched between two views by the epipolar cost (< max_pair_px) and triangulated: a list of
    {height (nose - lowest ankle, m), ankle (lowest ankle height above the floor, m), cost, foot_xz}. Poses: (R, t, K, k1) in the world
    frame (Y up, metres)."""
    Ra, ta, Ka, ka = poseA
    Rb, tb, Kb, kb = poseB
    fa, fb = make_frame(ppA, Ka, ka), make_frame(ppB, Kb, kb)
    if fa is None or fb is None:
        return []
    Rab = Rb @ Ra.T
    E = skew(tb - Rab @ ta) @ Rab
    c = pair_costs(E, fa, fb)
    keepA = ((ppA[:, :, 2] >= CONF).sum(1) >= MIN_JOINTS)
    keepB = ((ppB[:, :, 2] >= CONF).sum(1) >= MIN_JOINTS)
    pa, pb = ppA[keepA], ppB[keepB]
    out = []
    for p in range(len(pa)):
        q = int(np.argmin(c[p]))
        if not np.isfinite(c[p, q]) or c[p, q] > max_pair_px:
            continue
        both = (pa[p, :, 2] >= CONF) & (pb[q, :, 2] >= CONF)
        idx = list(np.flatnonzero(both))
        if len(idx) < 8 or not {0, 15, 16} <= set(idx):
            continue
        X = triangulate_pair(Ra, ta, Ka, ka, pa[p, both, :2].astype(float), Rb, tb, Kb, kb, pb[q, both, :2].astype(float))
        y = {k: X[idx.index(k), 1] for k in (0, 15, 16)}
        lo = min(y[15], y[16])
        out.append({"height": float(y[0] - lo), "ankle": float(lo), "cost": float(c[p, q]), "foot_xz": X[idx.index(15)][[0, 2]].tolist()})
    return out


def summarise_people(rows: list[dict]) -> dict:
    h = np.array([r["height"] for r in rows])
    f = np.array([r["ankle"] for r in rows])
    if not len(h):
        return {"persons": 0}
    return {"persons": int(len(h)), "nose_to_ankle_median_m": round(float(np.median(h)), 3),
            "share_1.0_to_2.0_m": round(float(((h > 1.0) & (h < 2.0)).mean()), 3),
            "ankle_height_median_m": round(float(np.median(f)), 3), "ankle_height_p10_p90_m": [round(float(np.percentile(f, 10)), 3), round(float(np.percentile(f, 90)), 3)],
            "share_ankle_within_0.3m_of_floor": round(float((np.abs(f) < 0.3).mean()), 3)}


DEMO_WORK = "larissa_kadu_38"      # the reference take of the world package: its attach of phones 05 / 06 (rotation-only)


def demo_prior(cam: str, world: dict) -> tuple[np.ndarray, np.ndarray]:
    """(Rt, C): the demo's own attach of the phone in the same sub-model, in the room's WORLD frame. The demo localised its phone `cam`
    in the same sub-model (cameras/sfm_work/sub_<cam>) and attached it by ROTATION ONLY: a constant Rs (sub frame -> SfM frame; its
    per-frame poses differ from the sub-frame ones by exactly one rotation, checked to 0.01 deg) and one centre. Both are read back
    from the demo's caches; Rt = R_world_from_sfm Rs, C = world centre."""
    import pickle

    from .. import recap_cameras as rc
    base = rc.PACKAGE_ROOT / "work" / DEMO_WORK / "cameras"
    sub = pickle.loads((base / "cache" / f"locsub_{cam}.pkl").read_bytes())["value"]
    main = np.load(base / "cache" / f"loc_{cam}.npz", allow_pickle=True)
    idx = [i for i, p in enumerate(sub) if p.ok and main["ok"][i]]
    Rs = np.einsum("fji,fjk->fik", main["R"][idx], np.array([sub[i].R for i in idx]))        # R_main^T R_sub per frame
    Rs_mean = Ro.from_matrix(Rs).mean().as_matrix()
    spread = max(float(np.degrees(np.linalg.norm(Ro.from_matrix(R @ Rs_mean.T).as_rotvec()))) for R in Rs[::25])
    if spread > 0.01:
        raise RuntimeError(f"the demo's attach of {cam} is not rotation-only (spread {spread:.3f} deg)")
    rep = json.loads((base / "report.json").read_text())["attach"][cam]
    s, Rw, tt = float(world["s"]), np.asarray(world["R"]), np.asarray(world["t"])
    return Rw @ Rs_mean, s * Rw @ np.array(rep["camera_centre_sfm"]) + tt


def people_hits(Rt: np.ndarray, C: np.ndarray, views: list[View], thr: tuple[float, float]) -> dict:
    """Among the 3D people that project into the attached camera's picture (>= 8 joints inside): the median distance (px) to the best
    matching detected person and the share closer than the two thresholds."""
    d = []
    for v in views:
        Rm = v.Rs @ Rt.T
        uv, z = project_world(np.nan_to_num(v.p3.X), Rm, C, v.K, v.k1)
        W, H = v.size
        okj = v.p3.ok & (z > 0.3) & (uv[..., 0] > 0) & (uv[..., 0] < W) & (uv[..., 1] > 0) & (uv[..., 1] < H)
        vis = okj.sum(1) >= 8
        if not vis.any():
            continue
        det = v.det[:, :, :2]
        dok = v.det[:, :, 2] >= CONF
        dist = np.linalg.norm(uv[:, None] - det[None], axis=-1)
        valid = okj[:, None, :] & dok[None]
        cnt = valid.sum(-1)
        m = np.where(cnt >= 6, (np.minimum(dist, 300) * valid).sum(-1) / np.maximum(cnt, 1), np.inf)
        b = m.min(1) if m.shape[1] else np.full(len(uv), np.inf)
        d.extend(np.where(np.isfinite(b[vis]), b[vis], 300.0).tolist())
    d = np.array(d)
    if not len(d):
        return {"visible": 0}
    return {"visible": int(len(d)), "median_px": round(float(np.median(d)), 1), f"share_under_{thr[0]:.0f}px": round(float((d < thr[0]).mean()), 3),
            f"share_under_{thr[1]:.0f}px": round(float((d < thr[1]).mean()), 3)}


def kernel_refine(views: list[View], Rt0: np.ndarray, C0: np.ndarray, width: int, max_rot_deg: float = 4.0, max_shift_m: float = 0.6):
    """Refine (rotation correction, centre) on a smooth score of the people reprojections (Gaussian kernel of the best-match distance,
    sigma 25 -> 8 px at 480 px width, scaled), bounded around the prior. Returns (Rt, C, score)."""
    from scipy.optimize import minimize
    k = width / 480.0
    lim = np.r_[[np.radians(max_rot_deg)] * 3, [max_shift_m] * 3]

    def poses(x):
        x = np.clip(x, -lim, lim)
        return Ro.from_rotvec(x[:3]).as_matrix() @ Rt0, C0 + x[3:6] * np.array([1, 0.5, 1])

    def dists(x):
        Rt, C = poses(x)
        out = []
        for v in views:
            Rm = v.Rs @ Rt.T
            uv, z = project_world(np.nan_to_num(v.p3.X), Rm, C, v.K, v.k1)
            dok = v.det[:, :, 2] >= CONF
            dist = np.linalg.norm(uv[:, None] - v.det[None, :, :, :2], axis=-1)
            valid = v.p3.ok[:, None, :] & dok[None] & (z[:, None, :] > 0.3)
            cnt = valid.sum(-1)
            m = np.where(cnt >= 6, (np.minimum(dist, 300) * valid).sum(-1) / np.maximum(cnt, 1), np.inf)
            b = m.min(1) if m.shape[1] else np.full(len(uv), np.inf)
            out.append(np.where(np.isfinite(b), b, 300.0))
        return np.concatenate(out)
    x = np.zeros(6)
    r = None
    for sg in (25, 16, 11, 8):
        sg *= k
        r = minimize(lambda y: -float(np.mean(np.exp(-0.5 * (dists(y) / sg) ** 2))), x, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 500,
                              "initial_simplex": np.vstack([x, x + np.diag([0.01, 0.01, 0.01, 0.1, 0.05, 0.1])])})
        x = r.x
    Rt, C = poses(x)
    return Rt, C, float(-r.fun)


def attach_camera(cam: str, step_s: float = 1.5, seed: int = 0, log=print) -> dict:
    """Attach the sub-frame camera `cam` (cameras/<cam>_subframe.npz) to the room frame with the demo's rotation-only geometry (demo_prior),
    CHECKED and, when the check supports it, refined on the recap's own people: the people triangulated by the main cameras (no
    identities) must project onto the detected people of the attached view. Writes cameras/<cam>.npz (SfM frame, attached = True) +
    <cam>_world.npz + attach_<cam>.json (the evidence)."""
    from .. import recap_cameras as rc
    from . import newcam
    table = json.loads(rc.CAMS_JSON.read_text())["cams"]
    world = rc.world_frame()
    S = dict(np.load(rc.CAM_DIR / f"{cam}_subframe.npz", allow_pickle=True))
    mains = {c: dict(np.load(rc.CAM_DIR / f"{c}_world.npz", allow_pickle=True)) for c in ("00", "01", "02", "03")
             if (rc.CAM_DIR / f"{c}_world.npz").exists()}
    S_people = newcam.load_pose_frames(rc.SOURCE / f"{table[cam]['file']}.json")
    main_people = {c: newcam.load_pose_frames(rc.SOURCE / f"{table[c]['file']}.json") for c in mains}
    ok = S["ok"].astype(bool)
    lo = max(np.nanmin(S["t"][ok]), *[np.nanmin(m["t"][m["ok"]]) for m in mains.values()]) + 1
    hi = min(np.nanmax(S["t"][ok]), *[np.nanmax(m["t"][m["ok"]]) for m in mains.values()]) - 1
    p3 = triangulate_people(mains, main_people, np.arange(lo, hi, step_s))
    views = build_views(S, S_people, p3)
    W = int(S["native_size"][0])
    thr = (0.025 * W, 0.04 * W)                        # 12 / 19 px at 480 px width
    Rt0, C0 = demo_prior(cam, world)
    log(f"[attach_people] {cam}: {len(views)} views with identity-free 3D people ({np.mean([len(v.p3.X) for v in views]):.1f} per view); "
        f"demo prior centre {np.round(C0, 2).tolist()}")
    prior = people_hits(Rt0, C0, views, thr)
    rng = np.random.default_rng(seed)
    null = [people_hits(Ro.from_rotvec(rng.normal(0, 0.14, 3)).as_matrix() @ Rt0, C0 + rng.normal(0, [1.5, 0.3, 1.5]), views[::3], thr)
            for _ in range(12)]
    key2 = f"share_under_{thr[1]:.0f}px"
    sens = {}
    for name, (dR, dC) in {"centre_x+0.5m": (np.eye(3), [0.5, 0, 0]), "centre_z+0.5m": (np.eye(3), [0, 0, 0.5]),
                           "yaw_2deg": (Ro.from_euler("y", 2, degrees=True).as_matrix(), [0, 0, 0]),
                           "pitch_2deg": (Ro.from_euler("x", 2, degrees=True).as_matrix(), [0, 0, 0])}.items():
        sens[name] = people_hits(dR @ Rt0, C0 + np.array(dC), views[::3], thr).get(key2)
    # held-out refinement: fit on the even views, judge on the odd ones; adopt only a real improvement
    Rt1, C1, _ = kernel_refine(views[0::2], Rt0, C0, W)
    held_prior, held_ref = people_hits(Rt0, C0, views[1::2], thr), people_hits(Rt1, C1, views[1::2], thr)
    adopt = held_ref.get(key2, 0) >= 1.10 * held_prior.get(key2, 0) and held_ref.get(key2, 0) > held_prior.get(key2, 0) + 0.02
    if adopt:
        Rt, C, _ = kernel_refine(views, Rt0, C0, W)
    else:
        Rt, C = Rt0, C0
    final = people_hits(Rt, C, views, thr)
    # export: pose per frame in the world frame, then the SfM frame
    Rm = S["R"] @ Rt.T                                           # world -> camera
    tw = -np.einsum("fij,j->fi", Rm, C)
    s, Rw, tt = float(world["s"]), np.asarray(world["R"]), np.asarray(world["t"])
    C_sfm = (C - tt) @ Rw / s                                    # X_sfm = Rw^T (X_w - t) / s
    R_sfm = Rm @ Rw
    t_sfm = -np.einsum("fij,j->fi", R_sfm, C_sfm)
    base = {k: S[k] for k in ("t", "K", "dist", "ok", "pts", "frame_index", "source", "interpolated", "localised", "work_size", "native_size")}
    out_sfm = {**base, "R": R_sfm, "tvec": t_sfm, "reproj_px": S["reproj_px"], "n_inliers": S["n_inliers"], "attached": np.bool_(True),
               "axis_sd": S["axis_sd"], "recentred": S["recentred"], "camera_class": S["camera_class"], "type_model": np.str_("fixed_focal_rotation_only"),
               "type_held": np.zeros(len(S["t"]), bool), "frame": np.str_("sfm")}
    np.savez(rc.CAM_DIR / f"{cam}.npz", **out_sfm)
    np.savez(rc.CAM_DIR / f"{cam}_world.npz", t=S["t"], K=S["K"], R=Rm, tvec=tw, C=np.tile(C, (len(S["t"]), 1)), dist=S["dist"], ok=S["ok"],
             frame_index=S["frame_index"], pts=S["pts"], interpolated=S["interpolated"],
             note=np.str_("world frame of the room model (Y up, m); ROTATION-ONLY attach: ONE centre C for the whole video (attached camera)"))
    rep = {"cam": cam, "method": "demo rotation-only attach geometry (same sub-model), checked/refined on the recap's people (no identity labels)",
           "views": len(views), "threshold_px": list(thr), "prior": prior,
           "random_pose_null": {"median_px": round(float(np.median([n.get("median_px", 300) for n in null])), 1),
                                key2: round(float(np.mean([n.get(key2, 0) for n in null])), 3)},
           "sensitivity_share": {**sens, "prior": prior.get(key2)}, "held_out_prior": held_prior, "held_out_refined": held_ref,
           "refinement_adopted": bool(adopt), "final": final, "centre_world_m": C.tolist(), "centre_sfm": C_sfm.tolist(),
           "rotation_vs_prior_deg": round(float(np.degrees(np.linalg.norm(Ro.from_matrix(Rt @ Rt0.T).as_rotvec()))), 3),
           "centre_shift_m": round(float(np.linalg.norm(C - C0)), 3)}
    (rc.CAM_DIR / f"attach_{cam}.json").write_text(json.dumps(rep, indent=1, default=float))
    log(f"[attach_people] {cam}: prior {prior} | random-pose null {rep['random_pose_null']} | sensitivity {rep['sensitivity_share']} | "
        f"refinement adopted {adopt} | final {final}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.attach_people", description=__doc__.split("\n")[0])
    ap.add_argument("cam")
    ap.add_argument("--step", type=float, default=1.5, help="seconds between the times used for the 3D people")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    attach_camera(a.cam, a.step, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
