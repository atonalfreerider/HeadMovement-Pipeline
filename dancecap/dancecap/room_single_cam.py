"""New room analysis from ONE handheld, zooming phone video (john_lula): cameras, world frame, room model, review images.

    python -m dancecap.room_single_cam hypotheses               skeleton solutions at several 1x focals -> room_work/ptz/final_s*.npz
    python -m dancecap.room_single_cam lines                    LSD line segments of the skeleton frames, people masked (local, ~1 min)
    python -m dancecap.room_single_cam solve   [--f1x 806.7]    per-frame camera + world frame -> work/john_lula/cameras/
    python -m dancecap.room_single_cam variants                 focal hypotheses -> cameras/uncertainty_variants.json (the ambiguity table)
    python -m dancecap.room_single_cam room                     planes + textured floor / wall mesh -> room/
    python -m dancecap.room_single_cam review                   the room projected into frames -> review/room/

ONE camera cannot triangulate. What fixes the geometry (work/john_lula/NOTES_J2.md has the full account):
  rotation + zoom   the camera pans / tilts / zooms about an (almost) fixed centre: dancecap.cameras.ptz tracks R_i and f_i of every frame from the static
                    background (people masked), ~1 px; the focal is only known up to one scale (the pure-rotation cost is flat in f);
  focal             the 1x focal comes from the lens (distortion k1 -> 0 at 807 px), monocular geometry (MoGe-2: 773-855 px on the wide frames, VGGT: 912) and the
                    phone class; carried through as variants (`variants`);
  gravity / floor   the floor normal of a monocular metric-geometry model (MoGe-2) on 16 wide frames, agreeing to 0.1-1.0 deg after the rotations of the track
                    (and with the horizon of the people regression);
  metric scale      the dancers' statures (John 1.803 m, Lula 1.702 m): the camera height from every upright person's head / feet rays (independent of the
                    focal), then every distance (proportional to the focal);
  room axes         the dominant family of horizontal lines (floor boards, ceiling slats).
Heavy compute (per-frame tracking, 2D keypoints, monocular geometry) ran on a Runpod pod (dancecap.cameras.single_cam_pod).
"""
from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import cv2
import numpy as np

from .cameras import couple as CP
from .cameras import lines as LN
from .cameras import manhattan as MN
from .cameras import ptz as P
from .cameras import ptz_feats as F
from .cameras import ptz_run as PR
from .cameras import room_draw as D
from .cameras import single_view as SV
from .take import PACKAGE_ROOT

TAKE = "john_lula"
WORK = PACKAGE_ROOT / "work" / TAKE
RW = WORK / "room_work"                # intermediate data (git-ignored)
POD = RW / "pod"                       # what the pod job returned: tracks' matches, 2D keypoints, monocular geometry
CAM_DIR = WORK / "cameras"
ROOM_DIR = WORK / "room"
REVIEW_DIR = WORK / "review" / "room"
VIDEO = PACKAGE_ROOT.parent / "dances" / "John-Lula" / "john_lula_jnj.mp4"
SIZE = (538, 960)
FPS = 14997 / 500
SKEL_STEP = 15
YOLO_SEG = PACKAGE_ROOT / "weights" / "yolo11x-seg.pt"
F_MAIN_1X = 806.7                      # px of the 538 x 960 frame (final_s0.800.npz); see NOTES_J2.md for how it was chosen
ZOOM_RATIO_SPLIT = 1.3                 # a frame is "2x" when its focal is above 1.3 x the 1x focal


def log(msg: str) -> None:
    print(f"[room_single_cam {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def statures() -> dict[str, float]:
    cfg = tomllib.loads((PACKAGE_ROOT / "takes" / f"{TAKE}.toml").read_text(encoding="utf-8"))
    return {k: float(v["stature_m"]) for k, v in cfg["dancers"].items()}


# ------------------------------------------------------------------------------------------------ focal hypotheses


def stage_hypotheses(s_list=(0.70, 0.75, 0.80, 0.85, 0.90, 1.00), out_dir: Path | None = None) -> dict:
    """Skeleton solutions at several 1x focals. The rotation-only cost is nearly flat in the focal (a pure pan), so the focal is a HYPOTHESIS: for each
    scale s of the skeleton's first solution (f = s x 1008.4 px at 1x) the rotations are re-solved with every focal fixed at s x f_skeleton (zoom ratios
    kept), starting from the skeleton's rotations / s, first without lens, then k1, then k1 only on the inlier pairs -> ptz/final_s<s>.npz.
    The solve of the main hypothesis is the skeleton the whole track is refined against."""
    out_dir = out_dir or RW / "ptz"
    sk = np.load(out_dir / "skeleton.npz")
    pr = np.load(out_dir / "pairs.npz")
    n = len(sk["frame"])
    pairs = P.Pairs(pr["i"], pr["j"], pr["p"], pr["q"])
    lens0 = P.Lens(*SIZE, float(sk["lens"][0]), float(sk["lens"][1]))
    keep = P.reject_outliers(P.PtzSolution(sk["rv"], sk["f"], lens0), pairs, 2.5)
    sel = np.where(keep)[0][::3]
    pk = P.Pairs(pairs.i[sel], pairs.j[sel], pairs.p[sel], pairs.q[sel])
    res = {}
    for s in s_list:
        t0 = time.time()
        sol = P.solve_ptz(n, pk, sk["rv"] / s, sk["f"] * s, P.Lens(*SIZE, 0.0, 0.0), free=(), max_nfev=30, fix_f=True)
        sol = P.solve_ptz(n, pk, sol.rv, sol.f, sol.lens, free=("k1",), max_nfev=30, fix_f=True)
        lens = P.Lens(*SIZE, sol.lens.k1, 0.0)
        keep = P.reject_outliers(P.PtzSolution(sol.rv, sol.f, lens), pairs, 3.0)
        pa = P.Pairs(pairs.i[keep], pairs.j[keep], pairs.p[keep], pairs.q[keep])
        sol = P.solve_ptz(n, pa, sol.rv, sol.f, lens, free=("k1",), max_nfev=40, fix_f=True)
        k2 = P.reject_outliers(sol, pa, 2.0)
        pb = P.Pairs(pa.i[k2], pa.j[k2], pa.p[k2], pa.q[k2])
        sol = P.solve_ptz(n, pb, sol.rv, sol.f, sol.lens, free=("k1",), max_nfev=40, fix_f=True)
        np.savez_compressed(out_dir / f"final_s{s:.3f}.npz", frame=sk["frame"], rv=sol.rv, f=sol.f, lens=np.array([sol.lens.k1, 0.0, 0.0, 0.0]),
                            size=sk["size"], median=sol.median_px, rms=sol.rms_px, free=np.array(["k1"]))
        res[f"{s:.2f}"] = {"f1x_px": float(sol.f.min()), "k1": float(sol.lens.k1), "median_px": float(sol.median_px), "p90_px": float(sol.p90_px), "pairs": len(pb)}
        log(f"s={s:.2f}: " + json.dumps(res[f"{s:.2f}"]) + f" ({time.time() - t0:.0f} s)")
    return res


# ------------------------------------------------------------------------------------------------ inputs


F_SKELETON_1X = 1008.4                 # the 1x focal of the skeleton's first solution; the hypotheses are s x this


def skeleton_path(f1x: float) -> Path:
    return RW / "ptz" / f"final_s{round(f1x / F_SKELETON_1X, 2):.3f}.npz"


def load_skeleton(path: Path):
    rv, f, lens = PR.load_solution(path)
    return rv, f, lens


def load_track(skel: Path, tag: str, frames: list[int] | None = None, force: bool = False) -> dict:
    """Per-frame (R, f) of every video frame against the skeleton solution `skel` (from the stored pod matches; CPU, ~2 min for 2643 frames)."""
    out = RW / "final" / f"track_{tag}.npz"
    if out.exists() and not force and frames is None:
        return dict(np.load(out))
    rv, f, lens = load_skeleton(skel)
    m = PR.load_matches(sorted(POD.glob("track_s?_matches.npz")), len(rv))
    res = PR.refine_tracks(m, rv, f, lens, SKEL_STEP, frames=frames)
    res["lens"] = np.array([lens.k1, lens.k2, lens.ox, lens.oy])
    res["size"] = np.array(SIZE)
    if frames is None:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **res)
    return res


def smooth_focal(f: np.ndarray, k: int = 5) -> np.ndarray:
    """Median filter of log f over k frames: removes the per-frame focal jitter of the track, keeps the 0.7 s zoom steps."""
    lf = np.log(f)
    pad = k // 2
    x = np.pad(lf, pad, mode="edge")
    return np.exp(np.array([np.median(x[i:i + k]) for i in range(len(lf))]))


# ------------------------------------------------------------------------------------------------ monocular geometry (MoGe-2)


def moge_floor(trk: dict, f1x: float, moge_path: Path | None = None, seed: int = 0) -> dict:
    """Floor plane of every MoGe-2 frame (camera frame of the model, metric of the model): the unit UP normal, the camera height, the inlier share,
    the model's focal; and the normal re-expressed for the track's focal (rescale_up) and carried to the reference frame with the frame's rotation."""
    m = np.load(moge_path or POD / "moge.npz")
    pts, mask, fr = m["points"].astype(np.float32), m["mask"], m["frame"]
    H = pts.shape[1]
    rng = np.random.default_rng(seed)
    R = P.rotations(trk["rv"])
    out = {"frame": [], "up_cam": [], "h_model": [], "inl": [], "f_model": [], "up_ref": [], "f_track": [], "state_2x": []}
    for i, f0 in enumerate(fr):
        X, mk = pts[i], mask[i] & np.isfinite(pts[i]).all(-1)
        sel = mk & (np.arange(H)[:, None] > 0.45 * H)
        Pf = X[sel]
        if len(Pf) < 500:
            continue
        n, d, inl = SV.plane_ransac(Pf, rng, tol=0.06, iters=200, up_prior=np.array([0, -1.0, 0]))
        j = int(np.argmin(np.abs(trk["frame"] - f0)))
        f_trk = float(trk["f"][j])
        f_mod = float(m["intrinsics"][i][0, 0] * m["size"][0])
        up_t = SV.rescale_up(n, f_mod, f_trk)
        out["frame"].append(int(f0)); out["up_cam"].append(n); out["h_model"].append(abs(d)); out["inl"].append(float(inl.mean()))
        out["f_model"].append(f_mod); out["up_ref"].append(R[j].T @ up_t); out["f_track"].append(f_trk)
        out["state_2x"].append(f_trk > ZOOM_RATIO_SPLIT * f1x)
    return {k: np.array(v) for k, v in out.items()}


def gravity_from_moge(mg: dict, only_1x: bool = True) -> tuple[np.ndarray, dict]:
    """UP in the reference frame: trimmed mean of the per-frame floor normals (frames of the wide state), with the spread in degrees."""
    sel = ~mg["state_2x"] if only_1x else np.ones(len(mg["frame"]), bool)
    U = mg["up_ref"][sel]
    u = U.mean(0)
    u /= np.linalg.norm(u)
    for _ in range(3):
        ang = np.degrees(np.arccos(np.clip(U @ u, -1, 1)))
        keep = ang < max(2.0, 2.5 * np.median(ang))
        u = U[keep].mean(0)
        u /= np.linalg.norm(u)
    ang = np.degrees(np.arccos(np.clip(mg["up_ref"] @ u, -1, 1)))
    return u, {"n_frames_1x": int(sel.sum()), "spread_deg_1x_median": float(np.median(ang[sel])), "spread_deg_1x_max": float(ang[sel].max()),
               "spread_deg_2x": [round(float(a), 2) for a in ang[~sel]] if (~sel).any() else [], "frames": mg["frame"].tolist(),
               "angle_to_mean_deg": ang.round(2).tolist()}


# ------------------------------------------------------------------------------------------------ people


def people(trk: dict, up_ref: np.ndarray, h_prov: float = 1.43, pose_path: Path | None = None) -> dict:
    """Detections with the quantities of the metric analysis, the dancing couple picked, rays in the reference frame.

    The couple: candidate pairs (a dark torso overlapping a lighter one, dancecap.cameras.couple) of every frame, then a dynamic programme that
    prefers big pairs and a pair whose floor position (provisional camera height `h_prov`, the gravity `up_ref`) moves continuously - the other
    couples dancing on the same floor and the dark-shirted spectators are elsewhere."""
    p = np.load(pose_path or POD / "pose.npz")
    kp, box, fr = p["kp"], p["box"], p["frame"]
    nfr = int(p["n_frames"])
    mk = SV.mean_pose_keypoints(kp)
    ok = (mk["ank_low"][:, 2] > 0.5) & (mk["eye"][:, 2] > 0.5) & (mk["sh"][:, 2] > 0.5) & (mk["hip"][:, 2] > 0.4)
    ok &= (mk["ank_low"][:, 1] < SIZE[1] - 10) & (mk["ank_low"][:, 0] > 3) & (mk["ank_low"][:, 0] < SIZE[0] - 3) & (box[:, 3] - box[:, 1] > 60)
    ok &= np.isin(fr, trk["frame"])                      # a track of a frame subset (the focal variants) only has the rotation of those frames
    lens = P.Lens(*SIZE, float(trk["lens"][0]), float(trk["lens"][1]))
    rowall = np.clip(np.searchsorted(trk["frame"], fr), 0, len(trk["frame"]) - 1)
    Rall = P.rotations(trk["rv"])[rowall]
    fall = trk["f"][rowall]
    # provisional floor position of every detection (feet, or the bottom of the box when the ankles are not seen)
    foot_pix = np.where((mk["ank_low"][:, 2] > 0.3)[:, None], mk["ank_low"][:, :2], np.stack([0.5 * (box[:, 0] + box[:, 2]), box[:, 3]], 1))
    ray = SV.rays_in_reference(foot_pix, Rall, fall, lens)
    up = up_ref if np.median(ray @ up_ref) < 0 else -up_ref
    Wh = np.stack(SV.horizon_basis(up), 1)
    rw = ray @ Wh
    below = rw[:, 1] < -1e-3
    pos = np.full((len(fr), 2), 60.0)
    pf = SV.floor_points(rw[below], h_prov)
    pos[below] = np.clip(pf[:, [0, 2]], -60, 60)
    cands = CP.candidates(fr, box, p["rgb"], nfr)
    redness = (np.nan_to_num(p["rgb"][:, 0]) - np.nan_to_num(p["rgb"][:, 2])) > 20
    lead, foll = CP.track_pairs(cands, pos, redness)
    score = np.array([max([c[0] for c in cc] or [0.0]) for cc in cands])
    role = np.zeros(len(fr), np.int8)
    role[lead[lead >= 0]] = 1
    role[foll[foll >= 0]] = 2
    idx = np.where(ok)[0]
    row = rowall[idx]
    R = Rall[idx]
    f = fall[idx]
    rays = lambda pix: SV.rays_in_reference(pix, R, f, lens)          # noqa: E731
    foot_seen = np.isin(fr, trk["frame"]) & (mk["ank_low"][:, 2] > 0.3) & (mk["ank_low"][:, 1] < SIZE[1] - 10) & (mk["ank_low"][:, 0] > 3) & (mk["ank_low"][:, 0] < SIZE[0] - 3)
    return {"idx": idx, "frame": fr[idx], "role": role[idx], "box": box[idx], "foot": rays(mk["ank_low"][idx, :2]), "eye": rays(mk["eye"][idx, :2]),
            "sh": rays(mk["sh"][idx, :2]), "hip": rays(mk["hip"][idx, :2]), "row": row, "f": f, "pair_score": score, "lead": lead, "foll": foll,
            # every detection (feet only): the couple's floor positions need no eyes / hips
            "all_frame": fr, "all_role": role, "all_foot": SV.rays_in_reference(mk["ank_low"][:, :2], Rall, fall, lens), "all_foot_seen": foot_seen,
            "all_row": rowall}


def height_samples(pp: dict, up_ref: np.ndarray, stat: dict[str, float], pop_stature: float = 1.70, upright_deg: float = 7.0) -> dict:
    """Camera height implied by every upright person (head / feet rays; the dancers with their own stature, everybody else with `pop_stature`)."""
    up = up_ref if np.median(pp["foot"] @ up_ref) < 0 else -up_ref
    x, y, z = SV.horizon_basis(up)
    Wm = np.stack([x, y, z], 1)
    foot, eye = pp["foot"] @ Wm, pp["eye"] @ Wm
    n = SV.person_axis_normals(pp["foot"], pp["sh"])
    upr = np.abs(n @ y) < np.sin(np.radians(upright_deg))
    He = np.where(pp["role"] == 1, SV.eye_height(stat["lead"]), np.where(pp["role"] == 2, SV.eye_height(stat["follow"]), SV.eye_height(pop_stature)))
    h = SV.camera_height(foot, eye, He)
    return {"h": h, "upright": upr, "foot_w": foot, "eye_w": eye, "Wm": Wm, "up": up}


def summarize_height(h: np.ndarray, pp: dict, hs: dict, two_x: np.ndarray) -> dict:
    ok = np.isfinite(h) & hs["upright"] & (h > 0.6) & (h < 2.6)
    out = {}
    for name, m in (("lead_1x", (pp["role"] == 1) & ~two_x), ("follow_1x", (pp["role"] == 2) & ~two_x), ("lead_2x", (pp["role"] == 1) & two_x),
                    ("follow_2x", (pp["role"] == 2) & two_x), ("population_1x", (pp["role"] == 0) & ~two_x), ("population_2x", (pp["role"] == 0) & two_x)):
        v = h[m & ok]
        out[name] = {"n": int(len(v)), "median": float(np.median(v)) if len(v) else None,
                     "p25": float(np.percentile(v, 25)) if len(v) else None, "p75": float(np.percentile(v, 75)) if len(v) else None}
    dn = [out[k]["median"] for k in ("lead_1x", "follow_1x", "lead_2x", "follow_2x") if out[k]["n"] >= 30]
    out["dancers_pooled_median"] = float(np.median(dn)) if dn else None
    out["dancers_role_medians"] = dn
    return out


def floor_positions(hs: dict, h_cam: float) -> np.ndarray:
    """(M,3) floor points of the feet (ankle height) in the world axes, relative to the point of the floor below the camera."""
    return SV.floor_points(hs["foot_w"], h_cam)


# ------------------------------------------------------------------------------------------------ lines


def stage_lines(a=None) -> dict:
    """LSD segments of the skeleton frames (every 15th video frame), people masked with the YOLO11x-seg masks -> room_work/lines.npz."""
    idx = [int(k) for k in np.load(RW / "ptz" / "skeleton.npz")["frame"]]
    frames = list(F.read_video_frames(str(VIDEO), *SIZE, frames=idx))
    imgs = [f for _, f in frames]
    masks = []
    for b in range(0, len(imgs), 8):
        pm = F.person_masks_yolo(imgs[b:b + 8], str(YOLO_SEG))
        masks += [F.static_ignore(im.shape[:2], m, []) for im, m in zip(imgs[b:b + 8], pm)]
    segs = [LN.detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), mk) for (_, im), mk in zip(frames, masks)]
    cnt = np.array([len(s) for s in segs])
    np.savez_compressed(RW / "lines.npz", frame=np.array(idx), seg=np.concatenate(segs), count=cnt)
    return {"frames": len(idx), "segments_median": float(np.median(cnt))}


def line_azimuth(skel: Path, up_ref: np.ndarray, tau_deg: float = 1.2) -> dict:
    """Dominant horizontal family of straight background lines (floor boards, ceiling slats ...) about the given vertical."""
    ln = np.load(RW / "lines.npz")
    frames, seg, cnt = ln["frame"], ln["seg"], ln["count"]
    off = np.concatenate([[0], np.cumsum(cnt)])
    rv, f, lens = load_skeleton(skel)
    R = P.rotations(rv)
    c = np.array([SIZE[0] / 2.0, SIZE[1] / 2.0])
    Ns, Ws = [], []
    for i in range(len(frames)):
        s = seg[off[i]:off[i + 1]]
        if len(s):
            n, _ = LN.plane_normals(s, float(f[i]), c, lens.k1, lens.k2)
            Ns.append(n @ R[i])
            Ws.append(np.minimum(s[:, 4], 200.0))
    N, W = np.concatenate(Ns), np.concatenate(Ws)
    fr = MN.fit_azimuth(N, W, up_ref, tau_deg=tau_deg)
    v, a, b = fr.M.T
    ang = lambda x: np.degrees(np.arcsin(np.clip(np.abs(N @ x), 0, 1)))        # noqa: E731
    return {"b": b, "a": a, "share_b": float((W * (ang(b) < 1.5)).sum() / W.sum()), "share_a": float((W * (ang(a) < 1.5)).sum() / W.sum()),
            "share_v": float((W * (ang(v) < 1.5)).sum() / W.sum()), "n_lines": int(len(N)), "tau_deg": tau_deg,
            "b_within_1p5deg_unweighted": float((ang(b) < 1.5).mean())}


# ------------------------------------------------------------------------------------------------ solve


def solve(f1x: float = F_MAIN_1X, tag: str | None = None, write: bool = True, trk: dict | None = None) -> dict:
    tag = tag or f"s{round(f1x / F_SKELETON_1X, 2):.3f}"
    skel = skeleton_path(f1x)
    trk = trk or load_track(skel, tag)
    stat = statures()
    n = len(trk["frame"])
    f_s = smooth_focal(trk["f"])
    trk = dict(trk, f=f_s)
    fmain = float(np.median(f_s[f_s < ZOOM_RATIO_SPLIT * f_s.min()]))
    two_x_row = f_s > ZOOM_RATIO_SPLIT * fmain
    mg = moge_floor(trk, fmain)
    up_ref, grav = gravity_from_moge(mg)
    pp = people(trk, up_ref)
    hs = height_samples(pp, up_ref, stat)
    two_x = two_x_row[pp["row"]]
    hsum = summarize_height(hs["h"], pp, hs, two_x)
    h_cam = hsum["dancers_pooled_median"]
    # camera-centred floor positions of the dancers' feet (picked couple, wide state, ankles seen), the origin = their mean position
    Wh_ = hs["Wm"]
    sel = (pp["all_role"] > 0) & pp["all_foot_seen"] & ~two_x_row[pp["all_row"]]
    foot_h = pp["all_foot"][sel] @ Wh_
    below = foot_h[:, 1] < -1e-3
    foot_h = foot_h[below]
    pos_all = SV.floor_points(foot_h, h_cam)
    roles_sel = pp["all_role"][sel][below]
    frames_sel = pp["all_frame"][sel][below]
    med = np.median(pos_all[:, [0, 2]], 0)
    mad = 1.4826 * np.median(np.abs(pos_all[:, [0, 2]] - med), 0)
    keep = (np.abs(pos_all[:, [0, 2]] - med) < 4 * np.maximum(mad, 0.3)).all(1)
    origin = np.array([np.mean(pos_all[keep, 0]), 0.0, np.mean(pos_all[keep, 2])])
    ctr = pos_all
    az = line_azimuth(skel, up_ref)
    fwd_ref = np.array([0.0, 0.0, 1.0])
    x_dir = az["a"] if az["share_a"] >= az["share_b"] else az["b"]            # the family with most line weight = the floor boards / ceiling slats
    Wm = SV.world_basis(up_ref, x_dir if x_dir[0] > 0 else -x_dir, fwd_ref)
    # the positions above are in the horizon_basis axes (X along the reference x): express them in the final world axes
    Wh = hs["Wm"]
    rot = Wm.T @ Wh                                              # horizon-basis coordinates -> final world coordinates
    origin_w = rot @ origin
    C = np.array([-origin_w[0], h_cam, -origin_w[2]])
    lens = P.Lens(*SIZE, float(trk["lens"][0]), float(trk["lens"][1]))
    k1 = float(trk["lens"][0])
    # frame flags: a bad solve (high residual, few inliers) is replaced by the interpolation of its ok neighbours when they are within 1 s
    rms = trk["rms_px"]
    ok = np.isfinite(rms) & (rms < 4.5) & (trk["n_inl"] >= 80)
    rv_f, f_f, ok, interp = SV.fill_bad_frames(trk["rv"], f_s, ok, max_gap=int(round(FPS)))
    R, tvec, K, dist, Cn = SV.camera_arrays(rv_f, f_f, Wm, C, SIZE, k1)
    dist_to_dancers = float(np.hypot(C[0], C[2]))
    # statistics of the dancers' positions in the world
    ctr_w = (rot @ ctr.T).T - origin_w
    dwr = np.hypot(ctr_w[:, 0] - C[0], ctr_w[:, 2] - C[2])
    n_feet = {"lead": int((roles_sel[keep] == 1).sum()), "follow": int((roles_sel[keep] == 2).sum())}
    rig = {"schema": "john_lula.camera_rig/1", "model": "handheld phone: fixed centre, per-frame rotation and focal (zoom), one k1; see NOTES_J2.md",
           "image_px": list(SIZE), "principal_point_px": [SIZE[0] / 2, SIZE[1] / 2], "k1": k1, "k2_fitted_then_ignored": float(trk["lens"][1]),
           "focal_1x_px": fmain, "focal_2x_px": float(np.median(f_s[two_x_row])) if two_x_row.any() else None,
           "zoom_ratio": float(np.median(f_s[two_x_row]) / fmain) if two_x_row.any() else None,
           "centre_world_m": C.tolist(), "camera_height_m": h_cam, "dancers_mean_distance_m": dist_to_dancers,
           "dancer_distance_percentiles_m": np.percentile(dwr[keep], [5, 25, 50, 75, 95]).round(2).tolist(),
           "dancer_centre_extent_x_m": np.percentile(ctr_w[keep, 0], [5, 95]).round(2).tolist(),
           "dancer_centre_extent_z_m": np.percentile(ctr_w[keep, 2], [5, 95]).round(2).tolist(),
           "world_axes_in_reference_frame": {"X": Wm[:, 0].tolist(), "Y": Wm[:, 1].tolist(), "Z": Wm[:, 2].tolist()},
           "gravity": {"source": "MoGe-2 floor normals of 16 wide frames carried through the track", **{k: v for k, v in grav.items() if k != "angle_to_mean_deg"}},
           "x_axis": {"source": "dominant family of horizontal lines (floor boards, ceiling slats)", "direction_ref": x_dir.tolist(), **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in az.items()}},
           "height": {"definition": "head / feet rays of every upright person, eye = 0.936 stature, ankle point 0.09 m; the dancers with their statures", **hsum},
           "statures_m": stat, "origin_rule": "trimmed mean of the picked dancers' floor positions (lower ankle of each, wide frames, 4 MAD trim)", "origin_feet_samples": n_feet,
           "moge": {"camera_height_model_m_median": float(np.median(mg["h_model"])), "focal_model_px_1x": float(np.median(mg["f_model"][~mg["state_2x"]])),
                    "note": "the model's own metric scale; the people-based height is used (see NOTES_J2.md section 4)"},
           "track": {"frames": int(n), "ok_frames": int(ok.sum()), "interpolated_frames": int(interp.sum()), "reproj_median_px": float(np.nanmedian(rms)), "reproj_p90_px": float(np.nanpercentile(rms, 90)),
                     "skeleton_median_px": float(np.load(skel)["median"]) if "median" in np.load(skel).files else None, "skeleton": skel.name}}
    res = {"R": R, "tvec": tvec, "K": K, "dist": dist, "C": Cn, "ok": ok, "interp": interp, "frame": trk["frame"], "f": f_f, "rig": rig, "Wm": Wm, "up_ref": up_ref,
           "rv": rv_f, "rms": rms, "n_inl": trk["n_inl"], "hs": hs, "pp": pp, "pos_rot": rot, "origin_w": origin_w, "h_cam": h_cam, "mg": mg,
           "two_x_row": two_x_row}
    if write:
        write_cameras(res, trk)
    return res


def write_cameras(res: dict, trk: dict) -> None:
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    n = len(res["frame"])
    t = res["frame"] / FPS
    common = dict(t=t, K=res["K"], R=res["R"], tvec=res["tvec"], dist=res["dist"], ok=res["ok"], reproj_px=res["rms"].astype(np.float32),
                  n_inliers=res["n_inl"].astype(np.int32))
    extra = dict(frame_index=res["frame"].astype(np.int32), interpolated=res["interp"], focal_px=res["f"].astype(np.float32), source=np.array(["track"] * n),
                 camera_class=np.array("handheld_zoom"), type_model=np.array("fixed_centre"), centre_world=res["C"][0], pp_px=np.array([SIZE[0] / 2, SIZE[1] / 2]),
                 k2_ignored=np.array(float(trk["lens"][1])), focal_hypothesis_px=np.array(res["rig"]["focal_1x_px"]), attached=np.array(False),
                 t_clock=np.array("video time of the single camera = the take's reference clock (identity clock); frame k at k / 29.994 s"),
                 pts=t, native_size=np.array(SIZE, np.int32), work_size=np.array(SIZE, np.int32),
                 camera_height_m=np.array(res["h_cam"]), zoom_2x=res["two_x_row"])
    np.savez_compressed(CAM_DIR / "01.npz", **common, **extra)
    np.savez_compressed(CAM_DIR / "01_world.npz", t=t, R=res["R"], tvec=res["tvec"], C=res["C"], K=res["K"], dist=res["dist"], ok=res["ok"])
    (CAM_DIR / "world_from_sfm.json").write_text(json.dumps({"s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "gravity_source": "MoGe-2 floor normals + track",
                                                              "scale_source": "dancers' statures (camera height from head / feet rays)",
                                                              "floor_source": "plane y = 0 (feet on the floor, ankle point 0.09 m)",
                                                              "x_axis": "dominant horizontal line family (floor boards)",
                                                              "note": "no SfM model exists: the cameras are stored in the world frame, this file is the identity"}, indent=1))
    (CAM_DIR / "rig.json").write_text(json.dumps(_jsonable(res["rig"]), indent=1))
    np.savez_compressed(CAM_DIR / "dancers_world.npz", **dancers_world(res))
    log(f"wrote cameras: {n} frames, ok {int(res['ok'].sum())}")


def dancers_world(res: dict) -> dict:
    """Floor positions (world, m) of the ankles of every detection that went into the analysis: frame, role (0 other, 1 lead, 2 follow), X, Z, upright."""
    pp, hs = res["pp"], res["hs"]
    pos = (res["pos_rot"] @ floor_positions(hs, res["h_cam"]).T).T - res["origin_w"]
    # trusted = a picked dancer of the wide state within 4 MAD of the cluster of the picked dancers (the pick is a heuristic: other couples slip in)
    wide = ~res["two_x_row"][pp["row"]]
    pk = (pp["role"] > 0) & wide
    med = np.median(pos[pk][:, [0, 2]], 0)
    mad = np.maximum(1.4826 * np.median(np.abs(pos[pk][:, [0, 2]] - med), 0), 0.3)
    trusted = pk & (np.abs(pos[:, [0, 2]] - med) < 4 * mad).all(1)
    return {"frame": pp["frame"].astype(np.int32), "role": pp["role"], "X": pos.astype(np.float32), "upright": hs["upright"],
            "h_sample": hs["h"].astype(np.float32), "trusted": trusted}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ------------------------------------------------------------------------------------------------ the focal / distance / height table


def variants(f_list=(0.70, 0.75, 0.80, 0.85, 0.90, 1.00), every: int = 6) -> dict:
    """Re-solve the track at other 1x focals (skeleton re-solved with that focal fixed, per-frame track every `every`-th frame) and report what changes:
    camera height, distance to the dancers, pitch of the camera at the first frame, the lens k1 and the pure-rotation cost. The dolly-zoom ambiguity."""
    stat = statures()
    out = []
    sub = list(range(0, 2643, every))
    for f1x in f_list:
        skel = skeleton_path(float(f1x) * F_SKELETON_1X)
        if not skel.exists():
            log(f"no skeleton solution for f = {f1x} ({skel.name}), skipped")
            continue
        trk = load_track(skel, "tmp", frames=sub)
        trk["f"] = smooth_focal(trk["f"], 3)
        fmain = float(np.median(trk["f"][trk["f"] < ZOOM_RATIO_SPLIT * trk["f"].min()]))
        two = trk["f"] > ZOOM_RATIO_SPLIT * fmain
        mg = moge_floor(trk, fmain)
        up_ref, grav = gravity_from_moge(mg)
        pp = people(trk, up_ref)
        hs = height_samples(pp, up_ref, stat)
        hsum = summarize_height(hs["h"], pp, hs, two[pp["row"]])
        h_cam = hsum["dancers_pooled_median"]
        sel = (pp["all_role"] > 0) & pp["all_foot_seen"] & ~two[pp["all_row"]]
        fh = pp["all_foot"][sel] @ hs["Wm"]
        fh = fh[fh[:, 1] < -1e-3]
        pos = SV.floor_points(fh, h_cam)
        d = pos[:, [0, 2]]
        med = np.median(d, 0)
        mean_pos = d[(np.abs(d - med) < 4 * np.maximum(1.4826 * np.median(np.abs(d - med), 0), 0.3)).all(1)].mean(0)
        dist = float(np.hypot(*mean_pos))
        # pitch of the camera at frame 0 relative to the horizon
        pitch0 = float(np.degrees(np.arcsin(-(P.rotations(trk["rv"][:1])[0] @ up_ref)[2])))
        sk = np.load(skel)
        out.append({"f1x_px": fmain, "f2x_px": float(np.median(trk["f"][two])) if two.any() else None, "lens_k1": float(trk["lens"][0]),
                    "rotation_fit_median_px": float(sk["median"]) if "median" in sk.files else None,
                    "camera_height_m": h_cam, "population_height_m": hsum["population_1x"]["median"], "lead_1x": hsum["lead_1x"]["median"],
                    "follow_1x": hsum["follow_1x"]["median"], "distance_to_dancers_m": dist, "camera_pitch_down_at_frame0_deg": pitch0,
                    "gravity_spread_deg": grav["spread_deg_1x_median"]})
        log(json.dumps(out[-1]))
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    (CAM_DIR / "uncertainty_variants.json").write_text(json.dumps(_jsonable({"variants": out, "frames_every": every}), indent=1))
    return {"variants": out}


# ------------------------------------------------------------------------------------------------ review


def load_cam() -> dict:
    d = dict(np.load(CAM_DIR / "01.npz", allow_pickle=True))
    d["C"] = d["centre_world"]
    return d


def cam_dict(arr: dict, i: int) -> dict:
    return dict(R=arr["R"][i], tvec=arr["tvec"][i], f=float(arr["K"][i, 0, 0]), c=(float(arr["K"][i, 0, 2]), float(arr["K"][i, 1, 2])), k1=float(arr["dist"][i]))


def read_frames(frames: list[int]) -> dict[int, np.ndarray]:
    return dict(F.read_video_frames(str(VIDEO), *SIZE, frames=frames))


def draw_room_model(im: np.ndarray, cd: dict, model: dict | None) -> None:
    """Floor grid (every metre; every 5th line heavier), world axes at the origin and, when a room model exists, the walls and ceiling outlines."""
    D.draw_floor_grid(im, cd, (-6, 8), (-9, 3), 1.0, color=(0, 220, 255), thickness=1, major=5)      # 1 m cells (every 5th line heavier)
    D.draw_axes(im, cd, (0, 0, 0), 1.5)
    if model:
        for q, col in model.get("quads", []):
            D.draw_quad(im, np.array(q), cd, tuple(col), 2)


def stage_review(a=None, model: dict | None = None, times=(0.5, 5, 10, 15, 22, 30, 40, 48, 56, 66, 76, 84), fname: str = "contact.jpg") -> dict:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    arr = load_cam()
    dw = np.load(CAM_DIR / "dancers_world.npz")
    frames = [int(round(t * FPS)) for t in times]
    got = read_frames(frames)
    tiles, labels = [], []
    stat = statures()
    for t, fi in zip(times, frames):
        im = got[fi].copy()
        i = int(np.argmin(np.abs(arr["frame_index"] - fi)))
        cd = cam_dict(arr, i)
        draw_room_model(im, cd, model)
        sel = (np.abs(dw["frame"] - fi) <= 6) & (dw["role"] > 0) & dw["trusted"]
        for X, role in zip(dw["X"][sel], dw["role"][sel]):
            col = (255, 80, 0) if role == 1 else (80, 80, 255)
            D.draw_points(im, X[None], cd, col, 5)
            top = np.array([X[0], stat["lead" if role == 1 else "follow"], X[2]])
            D.draw_segment3d(im, np.array([X[0], 0.0, X[2]]), top, cd, col, 2, 20)
        tiles.append(im)
        labels.append(f"t={t}s f={cd['f']:.0f}px {'ok' if arr['ok'][i] else 'NOT OK'}")
    cv2.imwrite(str(REVIEW_DIR / fname), D.contact_sheet(tiles, 4, 540, labels), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"files": [fname]}


# ------------------------------------------------------------------------------------------------ CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lines")
    sub.add_parser("hypotheses")
    s = sub.add_parser("solve")
    s.add_argument("--f1x", type=float, default=F_MAIN_1X)
    sub.add_parser("variants")
    sub.add_parser("room")
    sub.add_parser("evidence")
    sub.add_parser("review")
    a = ap.parse_args(argv)
    if a.cmd == "lines":
        print(json.dumps(stage_lines()))
    elif a.cmd == "hypotheses":
        print(json.dumps(stage_hypotheses(), indent=1))
    elif a.cmd == "solve":
        r = solve(a.f1x)
        print(json.dumps(_jsonable(r["rig"]), indent=1))
    elif a.cmd == "variants":
        variants()
    elif a.cmd == "room":
        from . import room_single_cam_mesh as RM
        print(json.dumps(_jsonable(RM.stage_room()["report"])))
    elif a.cmd == "evidence":
        from . import room_single_cam_plots as PL
        print(json.dumps(_jsonable(PL.main_plots()), indent=1))
    elif a.cmd == "review":
        mp = ROOM_DIR / "room_model.json"
        model = json.loads(mp.read_text()) if mp.exists() else None
        print(json.dumps(stage_review(model=model)))
        print(json.dumps(stage_review(model=model, times=(2, 8, 12, 17, 20, 27, 34, 44, 52, 60, 72, 80), fname="contact_b.jpg")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
