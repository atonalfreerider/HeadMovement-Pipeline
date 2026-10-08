"""Post-fit motion clean-up of a smplx-contract fit: temporal filter + floor contact, image-aware.

    python -m dancecap.pose_refine <take> [--src smplx] [--dst smplx_refined] [--mode refine]
                                   [--root-hz 5] [--rot-hz 8] [--w-floor 1] [--floor-mm 5] [--w-feet 1]
                                   [--w-slip 0] [--slip-max-v 0.15] [--w-anchor 1] [--anchor-mm 10] [--root-xz]
                                   [--iters 400] [--device cpu|cuda]

MAMMA has no floor term and fuses noisy per-frame evidence, so a fused fit floats / sinks its planted feet by a few cm
(slow per-dancer drift + pelvis-coupled wobble: scale/floor_diagnosis.json) and carries 4.5-12 Hz rigid root jitter.
This module writes a SEPARATE contract folder (default work/<take>/smplx_refined/; smplx/ is never touched) with the
same keys as the source plus 'smoothing' (json str: parameters, per-step magnitudes, energy history) and 'stance'
(T, 2) bool (left, right). Modes (each includes the steps before it in this list where marked):

  filter   1. zero-phase low-pass (Butterworth 2nd order, filtfilt) of the root translation at --root-hz and of every
              body / hand joint rotation (sign-continuous unit quaternions, renormalised) at --rot-hz. Default cutoffs
              sit at or above Winter's residual analysis on larissa_kadu_38 (root 3.5-4.5 Hz with 1-2.7 mm noise;
              wrists / head / ankles relative to the pelvis 6.5-8 Hz); 5 Hz keeps the 2.5 Hz eighth-note bounce at
              >= 94 %.
           2. stance per foot on the filtered motion: MAMMA's floor-contact score > 0.5 AND the sole within 30 mm of
              the lower foot AND the loaded sole point slower than 1.0 m/s horizontally (zouk stance feet pivot and
              slide: timing.TimingConfig.contact_vxz); 3-frame median, runs >= 3 frames.
  root     + 3. per-frame vertical offset = mean lowest-sole height of the stance feet, bridged linearly over frames
              without stance, low-passed (--vert-hz, default --root-hz), subtracted from the root height.
  ik       filter + 4. foot lock: in each stance run the sole-centre horizontal path is replaced by its 1.5 Hz low-pass
              (slow slides / pivots stay, frame-to-frame slip goes) and the lowest sole point is put on y = 0, applied
              as a rigid foot translation through analytic two-bone leg IK (hip + knee change, the ankle keeps its world
              orientation), ramped over 3 frames at the run ends.  root+ik = 3 then 4.
  refine   filter + a small MAP problem per dancer (torch): free per-frame root offset (vertical only unless
           --root-xz) and hip / knee / ankle rotation offsets (6 x 3, left-multiplied onto the local rotations);
           energy = 2D data + w_floor x stance lowest sole point / --floor-mm + penetration / 5 mm + w_slip x slip /
           3 mm + w_anchor x horizontal ankle / foot joint shift from the filtered input / --anchor-mm + second
           differences of the offsets (root / 1 mm, rotations / 0.01 rad) + offset prior (/ 0.15 rad).
           2D data, per camera: the per-frame mean Geman-McClure error (sigma 10 px) of MAMMA's 512 points vs the SAME
           ma_2d landmarks MAMMA fitted (per-frame cameras), PLUS w_feet x the same mean over each foot's own points
           (mamma_dyn.part_index: 44 + 44 of the 512), so the feet's evidence is not diluted 88:512 by the rest of
           the body. Slip: horizontal motion of the loaded sole point between consecutive stance frames, only where
           the filtered loaded point is slower than --slip-max-v (near-static feet; zouk stance feet pivot and slide,
           so pinning every stance frame drags the feet sideways); off by default (--w-slip 0).
           The weights are a TRADE-OFF, not a guarantee: the floor term is a hard pull and costs image agreement on the
           feet where the cameras put a planted foot above / below y = 0. Measure every variant with
           `python -m dancecap.pose_quality <take> --smplx-dir <dst>` (feet-only reprojection, joint displacement vs
           smplx/, slip on fixed frames) and compare them with `pose_quality <take> --variants ...`.
           First adopted run (smplx_refined, 2026-10-07; = --w-feet 0 --slip-max-v 0 --w-anchor 0 --root-xz): slip
           on every stance frame moved the stance ankles 27-35 mm sideways (median) and raised the feet-only
           reprojection 7.6 / 8.6 -> 9.0 / 9.6 px (lead / follow) - superseded. Re-tuned (2026-10-07, 'post' review;
           8 variants compared in work/<take>/quality/variants.json, review/pose_refine_variants.png): the DEFAULTS
           (= smplx_rf2_noslip, adopted for larissa_kadu_38's viewer) keep the stance ankles within 2.5 / 2.8 mm
           (median, p95 6.5 / 7.2) of the filtered fit horizontally, floats > 15 mm 22 / 29 % -> 0.6 / 1.5 %, sinks
           13 / 9 % -> 0, all-points reprojection 5.71 / 7.32 -> 6.05 / 7.42 px (cams 02 / 03 / 04 / 09 pay in every
           variant, 01 / 05 / 06 / 07 gain: the cameras disagree about foot height), feet-only 7.57 / 8.56 -> 7.43 /
           8.37 px (in-sample: the feet term fits the same landmarks). A whole-body vertical shift (mode root) costs
           +1.15 / +0.96 px; --w-feet 10 / 30 bend the legs further (leg-rotation offsets p95 11-16 deg) to fit the
           same feet landmarks in-sample and move the stance ankles 8-12 mm sideways.

Plain linear-blend skinning without pose correctives (as physics_body and the Unity skin); the 512 points are
pre-blended per joint (D @ W and D @ (W * v_rest)). Grid inputs: mamma/ma_cap/<seq>/gt/cam*.npz (cam_int, cam_ext
per frame) and mamma/ma_2d/<seq>/cam*.npz (landmarks, visibilities), seq = mamma/select.json's chosen set; the fit's
frame i is grid frame i (full-span fits only: Problem refuses a fit whose t differs from the grid's t).
Every mode recomputes the per-frame image-quality fields of the output (reproj_px, reproj_parts_px, quality,
quality_parts: mamma_dyn.reprojection_parts with smplx FK + correctives, as mamma_dyn.export_smplx) for the NEW motion;
contact / floor_contact (MAMMA's image-based contact scores), repaired, suspect and reproj_px_raw (MAMMA's pre-repair
residual) are the source fit's, carried unchanged (listed in 'smoothing').
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot

from . import physics_body as pb
from .take import PACKAGE_ROOT

ROLES = ("lead", "follow")
PIDX = {"lead": 0, "follow": 1}
LEG = {"left": (1, 4, 7), "right": (2, 5, 8)}  # hip, knee, ankle (SMPL-X joint ids)
LEGJ = [1, 2, 4, 5, 7, 8]
ANCHOR = [7, 8, 10, 11]  # ankles + feet (horizontal anchor)
CONTACT_V = 1.0  # m/s
MODES = ("filter", "root", "ik", "root+ik", "refine")


# ------------------------------------------------------------------------------------------------ filter + stance

def lowpass(x: np.ndarray, fc: float, fs: float, axis: int = 0) -> np.ndarray:
    b, a = butter(2, fc / (fs / 2))
    return filtfilt(b, a, x, axis=axis, padtype="odd", padlen=min(27, x.shape[axis] - 1))


def filter_rotations(aa: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """(T, 3) axis-angle -> low-passed (T, 3) via sign-continuous unit quaternions."""
    q = Rot.from_rotvec(aa).as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    qf = lowpass(q, fc, fs)
    qf /= np.linalg.norm(qf, axis=1, keepdims=True)
    return Rot.from_quat(qf).as_rotvec()


def runs_of(m: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def fk(body, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, p = pb.forward_kinematics(body, pb.full_pose(z), np.asarray(z["transl"], float))
    return R, p, np.einsum("tjab,jb->tja", R, body.j_rest) + p


def soles(body, R, p, idx) -> dict:
    return {f: pb.lbs_points(body, R, p, idx[f]) for f in ("left", "right")}


def stance_mask(z, body, R, p, idx, t, aux: dict | None = None) -> tuple[dict, dict]:
    """Stance per foot (see the module doc, step 2). aux (optional) receives 'speed': {foot: (T,) m/s} of the
    6 Hz low-passed loaded sole point."""
    from .physics import mamma_foot_contact
    sp_ = soles(body, R, p, idx)
    low = {f: sp_[f][..., 1].min(1) for f in sp_}
    lower = np.minimum(low["left"], low["right"])
    mc = mamma_foot_contact(z, body)
    fs = 1.0 / float(np.median(np.diff(t)))
    st = {}
    for k, f in enumerate(("left", "right")):
        pts = sp_[f]
        w = (pts[..., 1] <= low[f][:, None] + 0.01).astype(float)  # loaded point: within 1 cm of the lowest
        cen = (pts * w[..., None]).sum(1) / w.sum(1)[:, None]
        v = np.linalg.norm(np.gradient(lowpass(cen[:, [0, 2]], 6.0, fs), t, axis=0), axis=1)
        if aux is not None:
            aux.setdefault("speed", {})[f] = v
        con = (mc[:, k] > 0.5) if mc is not None else np.ones(len(t), bool)
        m = median_filter((con & (low[f] < lower + 0.03) & (v < CONTACT_V)).astype(np.uint8), size=3,
                          mode="nearest").astype(bool)
        keep = np.zeros_like(m)
        for a, b in runs_of(m, 3):
            keep[a:b + 1] = True
        st[f] = keep
    return st, low


def ramp_weights(T: int, runs: list[tuple[int, int]], n: int = 3) -> np.ndarray:
    w = np.zeros(T)
    for a, b in runs:
        w[a:b + 1] = 1.0
        for k in range(1, n + 1):
            r = 0.5 * (1 + np.cos(np.pi * k / (n + 1)))
            if a - k >= 0:
                w[a - k] = max(w[a - k], r)
            if b + k < T:
                w[b + k] = max(w[b + k], r)
    return w


def min_rot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(N,3,3) minimal rotations taking directions u to v."""
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    ax = np.cross(u, v)
    s = np.linalg.norm(ax, axis=-1, keepdims=True)
    ang = np.arctan2(s, np.sum(u * v, -1, keepdims=True))
    return Rot.from_rotvec(np.where(s > 1e-12, ax / np.maximum(s, 1e-12) * ang, 0.0)).as_matrix()


def leg_ik(body, z, R, J, foot: str, delta: np.ndarray, frames: np.ndarray) -> dict:
    """Move the ankle of `foot` by delta (F,3) on `frames` with hip + knee rotations; ankle world orientation kept.
    Writes z['body_pose'] in place."""
    h, k, a = LEG[foot]
    T = len(z["t"])
    H, K, A = J[frames, h], J[frames, k], J[frames, a]
    la, lb = np.linalg.norm(K - H, axis=1), np.linalg.norm(A - K, axis=1)
    At = A + delta
    c = np.linalg.norm(At - H, axis=1)
    cmin, cmax = np.abs(la - lb) + 1e-4, la + lb - 1e-4
    clamped = (c > cmax) | (c < cmin)
    c = np.clip(c, cmin, cmax)
    d = (At - H) / np.linalg.norm(At - H, axis=1, keepdims=True)
    At = H + c[:, None] * d
    pole = (K - H) - np.sum((K - H) * d, 1, keepdims=True) * d
    pn = np.linalg.norm(pole, axis=1, keepdims=True)
    fwd = R[frames, h][:, :, 2]
    pole = np.where(pn > 1e-4, pole / np.maximum(pn, 1e-12), fwd - np.sum(fwd * d, 1, keepdims=True) * d)
    pole /= np.linalg.norm(pole, axis=1, keepdims=True)
    al = np.arccos(np.clip((la ** 2 + c ** 2 - lb ** 2) / (2 * la * c), -1, 1))
    Kt = H + la[:, None] * (np.cos(al)[:, None] * d + np.sin(al)[:, None] * pole)
    Q1, Q2 = min_rot(K - H, Kt - H), min_rot(A - K, At - Kt)
    Rh2, Rk2 = Q1 @ R[frames, h], Q2 @ R[frames, k]
    loc = {h: np.swapaxes(R[frames, body.parents[h]], -1, -2) @ Rh2, k: np.swapaxes(Rh2, -1, -2) @ Rk2,
           a: np.swapaxes(Rk2, -1, -2) @ R[frames, a]}
    bp = np.asarray(z["body_pose"], float).reshape(T, 21, 3).copy()
    for j, m in loc.items():
        bp[frames, j - 1] = Rot.from_matrix(m).as_rotvec()
    z["body_pose"] = bp.reshape(T, 63)
    ang = lambda Q: np.degrees(np.linalg.norm(Rot.from_matrix(Q).as_rotvec(), axis=1))
    return {"frames": int(len(frames)), "clamped": int(clamped.sum()),
            "hip_change_deg_p95": round(float(np.percentile(ang(Q1), 95)), 2) if len(frames) else 0.0,
            "knee_change_deg_p95": round(float(np.percentile(ang(Q2), 95)), 2) if len(frames) else 0.0}


def filter_and_contacts(z: dict, mode: str, root_hz: float, rot_hz: float, vert_hz: float | None = None,
                        lock_hz: float = 1.5, aux: dict | None = None) -> tuple[dict, dict, dict]:
    """Modes filter / root / ik / root+ik (steps 1-4). -> (new npz dict, report, stance masks); aux: see stance_mask
    (speeds on the filtered motion)."""
    z = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in z.items()}
    t = np.asarray(z["t"], float)
    T = len(t)
    fs = 1.0 / float(np.median(np.diff(t)))
    vert_hz = vert_hz or root_hz
    rep: dict = {"mode": mode, "root_hz": root_hz, "rot_hz": rot_hz}
    raw_tr = np.asarray(z["transl"], float).copy()
    z["transl"] = lowpass(raw_tr, root_hz, fs)
    z["global_orient"] = filter_rotations(np.asarray(z["global_orient"], float), rot_hz, fs)
    for key, n in (("body_pose", 21), ("left_hand_pose", 15), ("right_hand_pose", 15)):
        a = np.asarray(z[key], float).reshape(T, n, 3)
        z[key] = np.stack([filter_rotations(a[:, j], rot_hz, fs) for j in range(n)], 1).reshape(T, n * 3)
    rep["transl_change_mm_rms"] = round(float(1000 * np.sqrt(np.mean((z["transl"] - raw_tr) ** 2))), 2)
    body = pb.load_body(z)
    idx = {f: body.sole_vertices(f, 0.02) for f in ("left", "right")}
    R, p, J = fk(body, z)
    st, low = stance_mask(z, body, R, p, idx, t, aux)
    rep["stance_frac"] = {f: round(float(st[f].mean()), 3) for f in st}
    if "root" in mode:
        num = np.where(st["left"], low["left"], 0) + np.where(st["right"], low["right"], 0)
        cnt = st["left"].astype(int) + st["right"].astype(int)
        has = cnt > 0
        e = np.interp(t, t[has], num[has] / cnt[has])
        e = lowpass(e, vert_hz, fs)
        z["transl"] = z["transl"] - np.outer(e, [0.0, 1.0, 0.0])
        rep["vertical_offset_mm"] = {"vert_hz": vert_hz, "median": round(float(1000 * np.median(e)), 1),
                                     "p5": round(float(1000 * np.percentile(e, 5)), 1),
                                     "p95": round(float(1000 * np.percentile(e, 95)), 1),
                                     "no_stance_frames": int((~has).sum())}
        R, p, J = fk(body, z)
    if "ik" in mode:
        rep["ik"] = {}
        sp_ = soles(body, R, p, idx)
        for f in ("left", "right"):
            runs = runs_of(st[f], 3)
            delta = np.zeros((T, 3))
            lowf, cen = sp_[f][..., 1].min(1), sp_[f].mean(1)
            for a, b in runs:
                seg = cen[a:b + 1, [0, 2]]
                smv = lowpass(seg, lock_hz, fs) if len(seg) >= 8 else np.repeat(seg.mean(0, keepdims=True), len(seg), 0)
                delta[a:b + 1, 0], delta[a:b + 1, 2] = smv[:, 0] - seg[:, 0], smv[:, 1] - seg[:, 1]
                delta[a:b + 1, 1] = -lowf[a:b + 1]
            w = ramp_weights(T, runs, 3)
            filled = delta.copy()
            for a, b in runs:  # hold the run-end correction and fade it out over 3 frames
                for kk in range(1, 4):
                    if a - kk >= 0 and not st[f][a - kk]:
                        filled[a - kk] = delta[a]
                    if b + kk < T and not st[f][b + kk]:
                        filled[b + kk] = delta[b]
            delta = filled * w[:, None]
            frames = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-5)
            rep["ik"][f] = leg_ik(body, z, R, J, f, delta[frames], frames)
            R, p, J = fk(body, z)
    jo = np.asarray(z["joints"], float)
    z["joints"] = np.concatenate([J, jo[:, 55:]], 1) if jo.shape[1] > 55 else J
    return z, rep, st


# ------------------------------------------------------------------------------------------------ image-aware refine

def _rodrigues(rv):
    import torch
    th = torch.linalg.norm(rv, dim=-1, keepdim=True).clamp_min(1e-12)
    k = rv / th
    K = torch.zeros(rv.shape[:-1] + (3, 3), dtype=rv.dtype, device=rv.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, dtype=rv.dtype, device=rv.device) + s * K + (1 - c) * (K @ K)


def grid_seq(work: Path) -> str:
    sel = json.loads((work / "mamma" / "select.json").read_text())
    return f"{work.name}_{sel['chosen']}"


class Problem:
    """Root + leg offsets of one dancer against the grid landmarks of every camera of the chosen fit set."""

    def __init__(self, z: dict, role: str, work: Path, stance: dict, device: str = "cpu"):
        import torch
        from .mamma_dyn import _load_verts512, part_index
        f32 = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
        self.torch, self.dev = torch, device
        body = pb.load_body(z)
        self.T = T = len(z["t"])
        parts = part_index()
        self.feet = {f: torch.tensor(parts[f"{f}_foot"], device=device) for f in ("left", "right")
                     if f"{f}_foot" in parts}
        D = _load_verts512()                                       # (512, V)
        V = D.shape[1]                                             # physics_body adds a mouth-cap centre vertex
        W = body.weights
        self.Bw = f32(D @ W[:V])                                   # (512, 55)
        self.A165 = f32((D @ (W[:V, :, None] * body.v_rest[:V, None, :]).reshape(V, 165)).T.copy())  # ((j,b),512)
        self.sole = {f: (f32(W[i]), f32(body.v_rest[i])) for f in ("left", "right")
                     for i in [body.sole_vertices(f, 0.02)]}
        self.parents = body.parents
        self.Jrest = f32(body.j_rest)
        self.Rl0 = _rodrigues(f32(pb.full_pose(z)))
        self.tr0 = f32(np.asarray(z["transl"], float))
        seq = grid_seq(work)
        gt, lm = work / "mamma" / "ma_cap" / seq / "gt", work / "mamma" / "ma_2d" / seq
        p = PIDX[role]
        self.cam_names, self.cams = [], []
        tz = np.asarray(z["t"], float)
        for f in sorted(gt.glob("cam*.npz")):
            g, l = np.load(f), np.load(lm / f.name)
            if "t" in g.files:  # the fit's frame i must be grid frame i (a partial / chunked fit is refused)
                tg = np.asarray(g["t"], float)
                if len(tg) < T or not np.allclose(tg[:T], tz, atol=1e-4):
                    raise ValueError(f"{f}: grid t ({len(tg)} frames from {tg[0]:.4f} s) does not match the fit's t "
                                     f"({T} frames from {tz[0]:.4f} s): pose_refine needs a full-span fit on this grid")
            E = np.asarray(g["cam_ext"], float)[:T]
            obs = l["landmarks"][:T, p, :, :2]
            vis = (l["visibilities"][:T, p] > 0.5) & (np.abs(obs).sum(-1) > 0)
            self.cam_names.append(f.stem[3:])
            self.cams.append((f32(np.asarray(g["cam_int"], float)[:T]), f32(E[:, :3, :3]), f32(E[:, :3, 3]),
                              f32(obs), torch.tensor(vis, device=device)))
        self.stance = {f: torch.tensor(stance[f], device=device) for f in stance}

    def forward(self, dp, dr):
        torch = self.torch
        Rl = self.Rl0.clone()
        Rl[:, LEGJ] = _rodrigues(dr) @ self.Rl0[:, LEGJ]
        T, J = self.T, self.Jrest
        Rw, pj = [None] * 55, [None] * 55
        tr = self.tr0 + dp
        for j in range(55):
            par = self.parents[j]
            if par < 0:
                Rw[j], pj[j] = Rl[:, j], J[j] + tr
            else:
                Rw[j] = Rw[par] @ Rl[:, j]
                pj[j] = pj[par] + (Rw[par] @ (J[j] - J[par]))
        R = torch.stack(Rw, 1)
        P = torch.stack(pj, 1)
        p = P - (R @ J[None, :, :, None])[..., 0]
        pts = (R.permute(0, 2, 1, 3).reshape(T, 3, 165) @ self.A165).transpose(1, 2) + self.Bw @ p
        R9 = R.reshape(T, 55, 9)
        soles_ = {f: ((Ws @ R9).reshape(T, -1, 3, 3) @ xs[None, :, :, None])[..., 0] + Ws @ p
                  for f, (Ws, xs) in self.sole.items()}
        return pts, soles_, Rl, P

    def data_term(self, pts, sigma: float = 10.0, w_feet: float = 0.0):
        """Sum over cameras of the mean (over frames with >= 3 visible points) per-frame mean GM error of all 512
        points, + w_feet x the same for each foot's own points. -> (total, per-camera all-points, per-camera feet)."""
        tot, per, per_f = 0.0, [], []
        for K, Rc, tc, obs, vis in self.cams:
            Xc = (Rc[:, None] @ pts[..., None])[..., 0] + tc[:, None]
            z = Xc[..., 2:3]
            uv = (K[:, None] @ Xc[..., None])[..., 0]
            r2 = ((uv[..., :2] / z.clamp_min(1e-3) - obs) ** 2).sum(-1)
            m = vis & (z[..., 0] > 0)
            gm = (r2 / (r2 + sigma ** 2)) * m
            n = m.sum(1)
            fr = gm.sum(1) / n.clamp_min(1)
            e = fr[n >= 3].mean() if bool((n >= 3).any()) else self.torch.zeros((), device=pts.device)
            ef = 0.0
            for idx in self.feet.values():
                nf = m[:, idx].sum(1)
                if (nf >= 3).any():
                    ef = ef + (gm[:, idx].sum(1) / nf.clamp_min(1))[nf >= 3].mean()
            tot = tot + e + (w_feet * ef if w_feet else 0.0)
            per.append(float(e))
            per_f.append(float(ef))
        return tot, per, per_f


def lowest(y):
    """Lowest sole point per frame (hard min: a sum-based soft-min sits tau * log(n near the bottom) BELOW the true
    minimum, which left the refined soles ~7 mm above the floor in the first larissa_kadu_38 run)."""
    return y.min(dim=-1).values


def refine_dancer(z: dict, role: str, work: Path, stance: dict, iters: int = 400, w_floor: float = 1.0,
                  device: str = "cpu", log=print, w_feet: float = 1.0, slip_max_v: float | None = 0.15,
                  w_slip: float = 0.0, floor_mm: float = 5.0, root_xz: bool = False,
                  speed: dict | None = None, w_anchor: float = 1.0, anchor_mm: float = 10.0) -> tuple[dict, dict]:
    """speed: {foot: (T,) m/s} loaded-point speed of the filtered motion (stance_mask aux); with slip_max_v the slip
    term only acts between consecutive stance frames that are both slower than slip_max_v (None: every stance pair).
    w_anchor: horizontal anchor of the ankles and feet (joints 7, 8, 10, 11) to their filtered input positions, scale
    anchor_mm: the floor fix is vertical, so sideways foot motion needs the images' (or the slip term's) support."""
    import torch
    prob = Problem(z, role, work, stance, device)
    T = prob.T
    dp = torch.zeros((T, 3 if root_xz else 1), device=device, requires_grad=True)
    dr = torch.zeros((T, len(LEGJ), 3), device=device, requires_grad=True)
    zero = torch.zeros((T, 1), device=device)
    root = (lambda: dp) if root_xz else (lambda: torch.cat([zero, dp, zero], 1))
    opt = torch.optim.Adam([{"params": [dp], "lr": 1e-3}, {"params": [dr], "lr": 5e-3}])
    both = {}
    for f in prob.stance:
        m = prob.stance[f]
        if slip_max_v is not None and speed is not None:
            m = m & torch.tensor(np.asarray(speed[f]) < slip_max_v, device=device)
        both[f] = m[1:] & m[:-1]
    with torch.no_grad():
        pts0, _, _, P0 = prob.forward(root(), dr)
        d0, per0, perf0 = prob.data_term(pts0, w_feet=w_feet)
        A0 = P0[:, ANCHOR][..., [0, 2]].clone()
    hist, t0 = [], time.time()
    for it in range(iters + 1):
        opt.zero_grad()
        pts, sol, _, P = prob.forward(root(), dr)
        e_data, _, _ = prob.data_term(pts, w_feet=w_feet)
        e_floor = e_pen = e_slip = torch.zeros((), device=device)
        for f, s in sol.items():
            y = lowest(s[..., 1])
            if prob.stance[f].any():
                e_floor = e_floor + ((y[prob.stance[f]] / (floor_mm / 1000.0)) ** 2).mean()
            e_pen = e_pen + ((torch.relu(-y) / 0.005) ** 2).mean()
            if w_slip and both[f].any():
                w = torch.softmax(-s[..., 1] / 0.005, dim=-1)
                c = (w[..., None] * s).sum(1)[:, [0, 2]]
                e_slip = e_slip + (((c[1:] - c[:-1])[both[f]] / 0.003) ** 2).sum(-1).mean()
        a_p = dp[2:] - 2 * dp[1:-1] + dp[:-2]
        a_r = dr[2:] - 2 * dr[1:-1] + dr[:-2]
        e_smooth = ((a_p / 0.001) ** 2).sum(-1).mean() + ((a_r / 0.01) ** 2).sum(-1).mean()
        e_prior = ((dr / 0.15) ** 2).sum(-1).mean()
        e_anchor = (((P[:, ANCHOR][..., [0, 2]] - A0) / (anchor_mm / 1000.0)) ** 2).sum(-1).mean() if w_anchor             else torch.zeros((), device=device)
        loss = e_data + w_floor * e_floor + e_pen + w_slip * e_slip + e_smooth + e_prior + w_anchor * e_anchor
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == iters:
            row = {"it": it, "loss": round(float(loss), 4), "data": round(float(e_data), 4),
                   "floor": round(float(e_floor), 4), "pen": round(float(e_pen), 4), "slip": round(float(e_slip), 4),
                   "smooth": round(float(e_smooth), 4), "prior": round(float(e_prior), 4),
                   "anchor": round(float(e_anchor), 4), "s": round(time.time() - t0, 1)}
            hist.append(row)
            log(f"[pose_refine] {role} {row}")
    with torch.no_grad():
        pts, _, Rl, _ = prob.forward(root(), dr)
        d1, per1, perf1 = prob.data_term(pts, w_feet=w_feet)
    out = dict(z)
    dpn = root().detach().cpu().numpy().astype(np.float64)
    drn = dr.detach().cpu().numpy().astype(np.float64)
    out["transl"] = np.asarray(z["transl"], float) + dpn
    bp = np.asarray(z["body_pose"], float).reshape(T, 21, 3).copy()
    Rln = Rl.detach().cpu().numpy().astype(np.float64)
    for j in LEGJ:
        bp[:, j - 1] = Rot.from_matrix(Rln[:, j]).as_rotvec()
    out["body_pose"] = bp.reshape(T, 63)
    body = pb.load_body(z)
    _, _, J2 = fk(body, out)
    jo = np.asarray(z["joints"], float)
    out["joints"] = np.concatenate([J2, jo[:, 55:]], 1) if jo.shape[1] > 55 else J2
    rep = {"iters": iters, "w_floor": w_floor, "floor_mm": floor_mm, "w_feet": w_feet, "w_slip": w_slip,
           "slip_max_v": slip_max_v, "root_xz": root_xz, "w_anchor": w_anchor, "anchor_mm": anchor_mm,
           "device": device, "history": hist,
           "slip_pairs": {f: int(both[f].sum()) for f in both},
           "data_term_before_after": [round(float(d0), 4), round(float(d1), 4)],
           "data_per_cam_before_after": {c: [round(a, 4), round(b, 4)] for c, a, b in zip(prob.cam_names, per0, per1)},
           "feet_data_per_cam_before_after": {c: [round(a, 4), round(b, 4)]
                                              for c, a, b in zip(prob.cam_names, perf0, perf1)},
           "root_offset_mm": {"median_abs_y": round(float(1000 * np.median(np.abs(dpn[:, 1]))), 1),
                              "p95_norm": round(float(1000 * np.percentile(np.linalg.norm(dpn, axis=1), 95)), 1)},
           "leg_rot_offset_deg_p95": round(float(np.degrees(np.percentile(np.linalg.norm(drn, axis=-1), 95))), 2)}
    return out, rep


# ------------------------------------------------------------------------------------------------ per-frame quality

CARRIED = ("contact", "floor_contact", "repaired", "suspect", "reproj_px_raw")


def requality(z: dict, role: str, work: Path) -> dict:
    """Recompute reproj_px, reproj_parts_px, quality, quality_parts of a (modified) contract dict for its own motion,
    exactly as mamma_dyn.export_smplx does (smplx FK with correctives, MAMMA's 512 points vs the chosen grid set's
    landmarks, all cameras). Frames flagged 'repaired' (interpolated by export_smplx) keep quality <= 0.1."""
    from . import mamma_dyn as md
    T = len(z["t"])
    pose = np.concatenate([np.asarray(z[k], float).reshape(T, -1) for k in
                           ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose",
                            "right_hand_pose")], 1)
    verts, _ = md._forward(pose, np.asarray(z["transl"], float), np.asarray(z["betas"], float), True)
    pts = np.matmul(md._load_verts512(), verts)
    del verts
    seq = grid_seq(work)
    gt, lm = work / "mamma" / "ma_cap" / seq / "gt", work / "mamma" / "ma_2d" / seq
    cams = sorted(p.stem for p in gt.glob("cam*.npz"))
    parts = md.part_index()
    rp = md.reprojection_parts(pts, cams, gt, lm, PIDX[role], parts, slice(0, T))
    part_px = np.stack([rp[k] for k in md.PARTS], 1)
    core = np.stack([rp[k] for k in md.PARTS if "hand" not in k], 1)
    worst = np.nanmax(np.where(np.isfinite(core), core, -np.inf), 1)
    worst = np.where(np.isfinite(worst), np.maximum(worst, np.nan_to_num(rp["all"], nan=0.0)), rp["all"])
    quality = np.exp(-np.nan_to_num(worst, nan=1e3) / 10.0)
    if "repaired" in z:
        bad = np.asarray(z["repaired"], bool)
        quality[bad] = np.minimum(quality[bad], 0.1)
    out = dict(z)
    out.update(reproj_px=rp["all"], reproj_parts_px=part_px.astype(np.float32), parts=np.array(md.PARTS),
               quality=quality, quality_parts=np.exp(-np.nan_to_num(part_px, nan=1e3) / 10.0).astype(np.float32))
    return out


# ------------------------------------------------------------------------------------------------ run

def run(take: str, src: str = "smplx", dst: str = "smplx_refined", mode: str = "refine", root_hz: float = 5.0,
        rot_hz: float = 8.0, vert_hz: float | None = None, w_floor: float = 1.0, iters: int = 400,
        device: str = "cpu", w_feet: float = 1.0, slip_max_v: float | None = 0.15, w_slip: float = 0.0,
        floor_mm: float = 5.0, root_xz: bool = False, quality: bool = True, w_anchor: float = 1.0,
        anchor_mm: float = 10.0) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if dst in ("smplx", src):
        raise ValueError("pose_refine writes a separate folder (never smplx/ or the source)")
    work = PACKAGE_ROOT / "work" / take
    out = work / dst
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for r in ROLES:
        z = dict(np.load(work / src / f"{r}.npz", allow_pickle=True))
        z.pop("smoothing", None)
        z.pop("stance", None)
        aux: dict = {}
        zf, frep, st = filter_and_contacts(z, "filter" if mode == "refine" else mode, root_hz, rot_hz, vert_hz,
                                           aux=aux)
        meta = {"source": f"work/{take}/{src}/{r}.npz", "mode": mode, "filter": frep,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": "dancecap.pose_refine"}
        if mode == "refine":
            zf, meta["refine"] = refine_dancer(zf, r, work, st, iters, w_floor, device, w_feet=w_feet,
                                               slip_max_v=slip_max_v, w_slip=w_slip, floor_mm=floor_mm,
                                               root_xz=root_xz, speed=aux.get("speed"), w_anchor=w_anchor,
                                               anchor_mm=anchor_mm)
        if quality:
            zf = requality(zf, r, work)
            meta["per_frame_quality"] = ("reproj_px, reproj_parts_px, quality, quality_parts recomputed for this "
                                         "motion (pose_refine.requality = mamma_dyn.export_smplx's definition)")
        else:
            for k in ("reproj_px", "reproj_parts_px", "quality", "quality_parts"):
                zf.pop(k, None)
            meta["per_frame_quality"] = "dropped (--no-quality): the source's values described another motion"
        meta["carried_from_source"] = [k for k in CARRIED if k in zf]
        zf["smoothing"] = np.array(json.dumps(meta))
        zf["stance"] = np.stack([st["left"], st["right"]], 1)
        np.savez_compressed(out / f"{r}.npz", **zf)
        summary[r] = meta
    (out / "pose_refine.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pose_refine", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--src", default="smplx")
    ap.add_argument("--dst", default="smplx_refined")
    ap.add_argument("--mode", default="refine", choices=MODES)
    ap.add_argument("--root-hz", type=float, default=5.0)
    ap.add_argument("--rot-hz", type=float, default=8.0)
    ap.add_argument("--vert-hz", type=float, default=None)
    ap.add_argument("--w-floor", type=float, default=1.0)
    ap.add_argument("--floor-mm", type=float, default=5.0, help="scale of the stance floor term (mm)")
    ap.add_argument("--w-feet", type=float, default=1.0,
                    help="weight of each foot's own 2D data term (0 = all 512 points only, the first run)")
    ap.add_argument("--w-slip", type=float, default=0.0,
                    help="slip term weight (0 = none: the adopted larissa_kadu_38 setting; 0.5 = variant rf2)")
    ap.add_argument("--slip-max-v", type=float, default=0.15,
                    help="slip term only where the filtered loaded sole point is slower than this (m/s); "
                         "<= 0: every stance frame (the first run)")
    ap.add_argument("--root-xz", action="store_true", help="free horizontal root offsets too (first run)")
    ap.add_argument("--w-anchor", type=float, default=1.0,
                    help="horizontal anchor of ankles + feet to the filtered input (0 = none, the first run)")
    ap.add_argument("--anchor-mm", type=float, default=10.0)
    ap.add_argument("--no-quality", action="store_true",
                    help="drop the per-frame quality fields instead of recomputing them (no smplx FK)")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    a = ap.parse_args(argv)
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    s = run(a.take, a.src, a.dst, a.mode, a.root_hz, a.rot_hz, a.vert_hz, a.w_floor, a.iters, a.device,
            w_feet=a.w_feet, slip_max_v=a.slip_max_v if a.slip_max_v > 0 else None, w_slip=a.w_slip,
            floor_mm=a.floor_mm, root_xz=a.root_xz, quality=not a.no_quality, w_anchor=a.w_anchor,
            anchor_mm=a.anchor_mm)
    for r, m in s.items():
        brief = {k: v for k, v in m.get("refine", {}).items()
                 if k not in ("history", "data_per_cam_before_after", "feet_data_per_cam_before_after")}
        print(f"[pose_refine] {r}: filter {json.dumps(m['filter'])} refine {json.dumps(brief)}")


if __name__ == "__main__":
    main()
