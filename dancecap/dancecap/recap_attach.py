"""Recap attached phones 05 / 06 made precise by the TEACHERS (agent R3; R2's NOTES_R2_cameras.md section 4 asked for exactly this).

R2 attached the two phones that face the Dance Precisions wall to the room frame by ROTATION ONLY with the demo's geometry: per frame R(t) = R_sub(t) Rt^T
and ONE centre C (yaw good to 2-4 deg, centre to about 0.3 m, validated on people without identities). With the teachers' 3D tracks (precise: four
well-localised phones, triangulated joints) there is a much better observable: the teachers' joints seen by the attached phone. Two constants are re-fitted,

    R'(t) = R(t) Q          Q = a constant rotation of the world frame (3 parameters),     C' = a constant centre (3 parameters),

minimising a robust (soft-L1) reprojection of the teachers' triangulated 3D joints against the JSON joints of the persons associated to them, residuals
measured in 5 % of the person's size. The per-frame rotation R(t) of the sub-model track (the phone sways and turns) is kept: only the attach changes.
It is validated on HELD-OUT blocks (fit on every other 20 s block, judged on the others) and adopted only if it improves them.

    python -m dancecap.recap_attach refine 05 06      -> people/attach/<cam>.json (before / after), <cam>.npz + <cam>_world.npz (R2's schema, refined)
                                                         and people/attach/assoc_<cam>.npz (the teachers re-associated with the refined pose)

Nothing of R2's is overwritten: the orchestrator decides whether the refined files replace cameras/05.npz and 06.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Ro

from . import recap_people as rp

OUT = rp.PEOPLE / "attach"
BLOCK_S = 20.0


def observations(cam: rp.Cam, A: rp.Assoc, tr: rp.Track3D, step: int = 6, kinds=(2, 3), conf: float = 0.5, max_cost: float = 0.30) -> dict:
    """Teacher joint pairs (3D from the track, 2D from the JSON) of every `step`-th frame of the camera: arrays over M frame-slots."""
    rows = []
    for i in range(0, len(cam.t), step):
        if not cam.ok[i]:
            continue
        P = cam.persons(i)
        for s in range(2):
            r = int(A.row[i, s])
            if r < 0 or r >= len(P) or A.kind[i, s] not in kinds or not (A.cost[i, s] <= max_cost):
                continue
            X, kd = tr.at(s, float(cam.t[i]))
            if X is None:
                continue
            v = (P[r, :, 2] >= conf) & np.isfinite(X[:, 0])
            if v.sum() < 6:
                continue
            rows.append((i, s, X, P[r], v))
    if not rows:
        raise RuntimeError("no teacher observations")
    return {"i": np.array([r[0] for r in rows]), "slot": np.array([r[1] for r in rows]), "X": np.stack([r[2] for r in rows]).astype(float),
            "uv": np.stack([r[3][:, :2] for r in rows]).astype(float), "valid": np.stack([r[4] for r in rows]),
            "size": np.array([rp.person_height_px(r[3]) for r in rows]), "t": cam.t[[r[0] for r in rows]]}


def reproject(params: np.ndarray, cam: rp.Cam, ob: dict, with_depth: bool = False):
    """(M,17,2) projection of the observed 3D joints through the attach (Q, C) = params (+ the depth (M,17) when asked)."""
    Q = Ro.from_rotvec(params[:3]).as_matrix()
    C = params[3:6]
    i = ob["i"]
    Rn = cam.R[i] @ Q
    Xc = np.einsum("mij,mkj->mki", Rn, ob["X"] - C)
    z = np.where(Xc[..., 2] > 1e-6, Xc[..., 2], 1e-6)
    x, y = Xc[..., 0] / z, Xc[..., 1] / z
    d = 1 + cam.k1[i][:, None] * (x * x + y * y)
    f, cx, cy = cam.K[i][:, 0, 0][:, None], cam.K[i][:, 0, 2][:, None], cam.K[i][:, 1, 2][:, None]
    uv = np.stack([f * x * d + cx, f * y * d + cy], -1)
    return (uv, Xc[..., 2]) if with_depth else uv


def residuals(params: np.ndarray, cam: rp.Cam, ob: dict, sel: np.ndarray) -> np.ndarray:
    """Reprojection residuals in 5 % of the person's size. A joint behind the camera (a 3D outlier) carries no information and is left out; the
    rest is clipped at 40 units so one wild joint cannot dominate."""
    uv, z = reproject(params, cam, {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(ob["i"]) else v) for k, v in ob.items()}, True)
    d = (uv - ob["uv"][sel]) / (0.05 * np.maximum(ob["size"][sel], 40.0))[:, None, None]
    ok = ob["valid"][sel] & (z > 0.3)
    return np.where(ok[..., None], np.clip(d, -40.0, 40.0), 0.0).reshape(-1)


def stats(params: np.ndarray, cam: rp.Cam, ob: dict, sel: np.ndarray) -> dict:
    uv = reproject(params, cam, {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(ob["i"]) else v) for k, v in ob.items()})
    d = np.linalg.norm(uv - ob["uv"][sel], axis=-1)
    v = ob["valid"][sel]
    rel = d / np.maximum(ob["size"][sel], 40.0)[:, None]
    per = np.array([np.median(rel[m][vv]) for m, vv in zip(range(len(d)), v) if vv.any()])
    return {"frame_slots": int(sel.sum()), "median_joint_px": round(float(np.median(d[v])), 2), "median_joint_rel": round(float(np.median(rel[v])), 4),
            "share_person_median_rel_below_5pct": round(float((per < 0.05).mean()), 3), "share_person_median_rel_below_10pct": round(float((per < 0.10).mean()), 3)}


def robust_cost(params: np.ndarray, cam: rp.Cam, ob: dict, sel: np.ndarray) -> float:
    r = residuals(params, cam, ob, sel)
    return float(np.sum(2.0 * (np.sqrt(1.0 + r * r) - 1.0)))


def fit(cam: rp.Cam, ob: dict, sel: np.ndarray, x0: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Robust fit from R2's attach: a persons whose joints are far from the projection at the current parameters (a wrong association) leave the
    set of each round (median joint distance > 12 % of the person's size, then 8 %, then 6 %); each round is Nelder-Mead on the soft-L1 cost (the
    landscape is flat far from the minimum and a gradient step can leave it) followed by a bounded Gauss-Newton polish in a small box."""
    from scipy.optimize import minimize
    x = x0.copy()
    use = sel.copy()
    for thr in (0.12, 0.08, 0.06)[:rounds]:
        uv = reproject(x, cam, {k: (v[use] if isinstance(v, np.ndarray) and len(v) == len(ob["i"]) else v) for k, v in ob.items()})
        d = np.linalg.norm(uv - ob["uv"][use], axis=-1)
        rel = np.where(ob["valid"][use], d, np.nan) / np.maximum(ob["size"][use], 40.0)[:, None]
        med = np.nanmedian(rel, axis=1)
        keep = np.flatnonzero(use)[med <= thr]
        use = np.zeros(len(sel), bool)
        use[keep] = True
        sim = np.vstack([x] + [x + np.r_[np.eye(3)[k] * 0.01, np.zeros(3)] if k < 3 else x + np.r_[np.zeros(3), np.eye(3)[k - 3] * 0.1] for k in range(6)])
        res = minimize(robust_cost, x, args=(cam, ob, use), method="Nelder-Mead", options={"initial_simplex": sim, "maxiter": 1500, "xatol": 1e-4, "fatol": 1e-3})
        x = res.x
        lo = np.r_[x[:3] - 0.03, x[3:] - 0.15]
        hi = np.r_[x[:3] + 0.03, x[3:] + 0.15]
        x = least_squares(residuals, x, args=(cam, ob, use), loss="soft_l1", f_scale=1.0, bounds=(lo, hi), x_scale=np.array([0.01] * 3 + [0.05] * 3), max_nfev=60).x
    return x


def refine_camera(cam_id: str, step: int = 6, log=rp.log) -> dict:
    cam = rp.load_cam(cam_id)
    A = rp.Assoc.load(rp.PEOPLE / f"assoc_{cam_id}.npz")
    tr = rp.Track3D.load(rp.PEOPLE / "track3d.npz")
    ob = observations(cam, A, tr, step)
    x0 = np.r_[np.zeros(3), cam.C[0]]
    block = (ob["t"] // BLOCK_S).astype(int)
    train, test = (block % 2 == 0), (block % 2 == 1)
    before = {"all": stats(x0, cam, ob, np.ones(len(block), bool)), "train": stats(x0, cam, ob, train), "test": stats(x0, cam, ob, test)}
    x_tr = fit(cam, ob, train, x0)
    held = stats(x_tr, cam, ob, test)
    x_all = fit(cam, ob, np.ones(len(block), bool), x0)
    after = {"all": stats(x_all, cam, ob, np.ones(len(block), bool))}
    adopt = held["median_joint_rel"] < before["test"]["median_joint_rel"] * 0.9
    rot_deg = float(np.degrees(np.linalg.norm(x_all[:3])))
    rep = {"cam": cam_id, "observations": int(len(block)), "frames_step": step, "before": before, "fit_on_train_judged_on_test": held, "after_all": after,
           "adopted": bool(adopt), "rotation_correction_deg": round(rot_deg, 3), "centre_world_m_before": np.round(x0[3:], 3).tolist(),
           "centre_world_m_after": np.round(x_all[3:], 3).tolist(), "centre_shift_m": round(float(np.linalg.norm(x_all[3:] - x0[3:])), 3),
           "params": x_all.tolist(), "params_train_only": x_tr.tolist(), "block_s": BLOCK_S,
           "note": "residuals: 3D teacher joints (4 localised phones) vs the JSON joints of the associated persons, in 5 % of the person's size; soft-L1"}
    log(f"{cam_id}: median joint error {before['all']['median_joint_px']} -> {after['all']['median_joint_px']} px "
        f"(held out {before['test']['median_joint_rel']:.3f} -> {held['median_joint_rel']:.3f} of the person's size), rotation {rot_deg:.2f} deg, centre moved {rep['centre_shift_m']} m, "
        f"{'ADOPT' if adopt else 'keep R2'}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{cam_id}.json").write_text(json.dumps(rep, indent=1))
    # the refined cameras in R2's schema (world and SfM frames)
    from .mamma_dyn import sfm_cameras
    Q = Ro.from_rotvec(x_all[:3]).as_matrix()
    Rn = cam.R @ Q
    C = np.tile(x_all[3:], (len(cam.t), 1))
    tvec = -np.einsum("fij,fj->fi", Rn, C)
    world = json.loads((rp.WORK / "cameras" / "world_from_sfm.json").read_text())
    zw = np.load(rp.WORK / "cameras" / f"{cam_id}_world.npz", allow_pickle=True)
    np.savez(OUT / f"{cam_id}_world.npz", **{**{k: zw[k] for k in zw.files}, "R": Rn, "tvec": tvec, "C": C,
                                              "note": np.str_("R3 refinement of R2's attach by the teachers (recap_attach): x_cam = R X_world + tvec")})
    z = np.load(rp.WORK / "cameras" / f"{cam_id}.npz", allow_pickle=True)
    Rs, ts = sfm_cameras(Rn, tvec, world)
    np.savez(OUT / f"{cam_id}.npz", **{**{k: z[k] for k in z.files}, "R": Rs, "tvec": ts})
    # re-association with the refined pose
    cam2 = rp.load_cam(cam_id)
    cam2.R, cam2.tvec, cam2.C = Rn, tvec, C
    A2 = rp.fill_by_2d_continuity(cam2, rp.associate(cam2, tr, gate=0.22))
    A2.save(OUT / f"assoc_{cam_id}.npz")
    ok = cam2.ok
    rep["association_with_refined_pose"] = {"gate": 0.22, "lead_slot_A": round(float((A2.row[ok, 0] >= 0).mean()), 3), "slot_B": round(float((A2.row[ok, 1] >= 0).mean()), 3),
                                            "both": round(float(((A2.row[ok, 0] >= 0) & (A2.row[ok, 1] >= 0)).mean()), 3),
                                            "median_cost": round(float(np.nanmedian(A2.cost[A2.row >= 0])), 4),
                                            "r2_pose_for_comparison": {"slot_A": round(float((A.row[ok, 0] >= 0).mean()), 3), "slot_B": round(float((A.row[ok, 1] >= 0).mean()), 3),
                                                                       "median_cost": round(float(np.nanmedian(A.cost[A.row >= 0])), 4)}}
    (OUT / f"{cam_id}.json").write_text(json.dumps(rep, indent=1))
    log(f"{cam_id}: re-associated with the refined pose at gate 0.22: {rep['association_with_refined_pose']}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.recap_attach", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("refine",))
    ap.add_argument("cams", nargs="*", default=list(rp.ATTACHED))
    ap.add_argument("--step", type=int, default=6)
    a = ap.parse_args(argv)
    for c in a.cams:
        refine_camera(c, a.step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
