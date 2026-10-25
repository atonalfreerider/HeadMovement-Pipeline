"""Geometry of the "sides" relations (DESIGN 2.1): frames, signed margins R1-R16, the Gauss linking integral (GLI), pass-through events,
the signed closest-point separation (F2), arm swivel, swing-twist, chain twist / wraps, headings, unwrapping.

Every function takes `xp=np` or `xp=torch` and uses the same formula in both (torch: float64 for G and d_s, differentiable).
World: y up, metres. SMPL-X body: +x the body's left, +y up, +z front.

Point sets: a dancer is a dict of named points (n,3): shL shR elL elR wrL wrR hipL hipR knL knR anL anR head palmL palmR (+ ftL ftR
the foot / toe points when known). `points_from_smplx` builds it from the 55 SMPL-X joints, `points_from_coco` from COCO-17.
"""
from __future__ import annotations

import math

import numpy as np

# SMPL-X joints (zouk_contacts.py:104-111)
PELVIS, SPINE1, SPINE2, SPINE3, NECK, HEAD = 0, 3, 6, 9, 12, 15
HIP = {"left": 1, "right": 2}
KNEE = {"left": 4, "right": 5}
ANKLE = {"left": 7, "right": 8}
FOOT = {"left": 10, "right": 11}
COLLAR = {"left": 13, "right": 14}
SHOULDER = {"left": 16, "right": 17}
ELBOW = {"left": 18, "right": 19}
WRIST = {"left": 20, "right": 21}
PARENTS22 = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

# COCO-17
COCO = ("nose", "leye", "reye", "lear", "rear", "shL", "shR", "elL", "elR", "wrL", "wrR", "hipL", "hipR", "knL", "knR", "anL", "anR")
CI = {n: i for i, n in enumerate(COCO)}
# COCO joint -> SMPL-X joint it is compared with (face points -> the head joint)
C2S = (HEAD, 23, 24, HEAD, HEAD, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8)
# COCO joint -> SMPL-X anchor joint of its reach gate, and the reach (m)
REACH = {0: (NECK, 0.40), 1: (NECK, 0.40), 2: (NECK, 0.40), 3: (NECK, 0.40), 4: (NECK, 0.40), 5: (SPINE3, 0.40), 6: (SPINE3, 0.40),
         7: (16, 0.50), 8: (17, 0.50), 9: (16, 0.85), 10: (17, 0.85), 11: (PELVIS, 0.35), 12: (PELVIS, 0.35), 13: (1, 0.70), 14: (2, 0.70),
         15: (1, 1.15), 16: (2, 1.15)}
UP = np.array([0.0, 1.0, 0.0])


# ------------------------------------------------------------------------------------------------ numpy / torch shim

class _Ops:
    def __init__(self, xp):
        self.xp = xp
        self.np = xp is np

    def cross(self, a, b):
        return np.cross(a, b) if self.np else self.xp.cross(a, b, dim=-1)

    def dot(self, a, b):
        return (a * b).sum(-1)

    def norm(self, a, keepdims=False):
        if self.np:
            return np.linalg.norm(a, axis=-1, keepdims=keepdims)
        return self.xp.linalg.norm(a, dim=-1, keepdim=keepdims)

    def clip(self, a, lo, hi):
        return np.clip(a, lo, hi) if self.np else self.xp.clamp(a, lo, hi)

    def maximum(self, a, b):
        if self.np:
            return np.maximum(a, b)
        if not self.xp.is_tensor(b):
            return self.xp.clamp_min(a, b)
        return self.xp.maximum(a, b)

    def asin(self, a):
        return np.arcsin(a) if self.np else self.xp.asin(a)

    def atan2(self, a, b):
        return np.arctan2(a, b) if self.np else self.xp.atan2(a, b)

    def sign(self, a):
        return np.sign(a) if self.np else self.xp.sign(a)

    def where(self, c, a, b):
        return np.where(c, a, b) if self.np else self.xp.where(c, a, b)

    def stack(self, xs, axis):
        return np.stack(xs, axis) if self.np else self.xp.stack(xs, dim=axis)

    def abs(self, a):
        return np.abs(a) if self.np else self.xp.abs(a)

    def sqrt(self, a):
        return np.sqrt(a) if self.np else self.xp.sqrt(a)

    def zeros_like(self, a):
        return np.zeros_like(a) if self.np else self.xp.zeros_like(a)

    def const(self, v, like):
        if self.np:
            return np.asarray(v, float)
        return self.xp.as_tensor(np.asarray(v, float), dtype=like.dtype, device=like.device)


def _ops(xp) -> _Ops:
    return _Ops(xp)


def unit(v, xp=np, eps: float = 1e-12):
    o = _ops(xp)
    return v / o.maximum(o.norm(v, keepdims=True), eps)


# ------------------------------------------------------------------------------------------------ headings and unwrapping

def wrap_deg(a):
    """to [-180, 180)"""
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def heading(v, xp=np, min_h: float = 1e-6):
    """yaw (deg) of the horizontal part of v: atan2(x, z); NaN (numpy) where the horizontal part is shorter than min_h"""
    o = _ops(xp)
    a = o.atan2(v[..., 0], v[..., 2]) * (180.0 / math.pi)
    if o.np:
        h = np.hypot(v[..., 0], v[..., 2])
        return np.where(h > min_h, a, np.nan)
    return a


def line_heading(L, R, xp=np):
    """facing of a left-right point pair: forward = (L - R) x up -> yaw deg"""
    o = _ops(xp)
    d = L - R
    f = o.stack([-d[..., 2], o.zeros_like(d[..., 0]), d[..., 0]], -1)
    return heading(f, xp)


def unwrap_deg(a, xp=np):
    """unwrap (deg); numpy: NaN gaps are bridged (the jump across a gap taken as the shortest one); torch: no NaN expected"""
    if xp is np:
        a = np.asarray(a, float)
        out = np.full_like(a, np.nan)
        ok = np.flatnonzero(np.isfinite(a))
        if len(ok):
            out[ok] = np.degrees(np.unwrap(np.radians(a[ok])))
        return out
    d = a[1:] - a[:-1]
    d = d - 360.0 * xp.round(d / 360.0)
    return xp.cat([a[:1], a[:1] + xp.cumsum(d, 0)])


# ------------------------------------------------------------------------------------------------ point sets

def palm_from_joints(J, side: str):
    """palm centre from the 55 SMPL-X joints (hand_contact.palm_centre: wrist + the four MCP joints)"""
    from .. import hand_contact as hc
    return hc.palm_centre(np.asarray(J), side)


def points_from_smplx(J, R=None) -> dict:
    """named points (n,3) of one dancer from SMPL-X joints (n,55,3) (numpy)"""
    J = np.asarray(J, float)
    P = {"shL": J[:, 16], "shR": J[:, 17], "elL": J[:, 18], "elR": J[:, 19], "wrL": J[:, 20], "wrR": J[:, 21], "hipL": J[:, 1], "hipR": J[:, 2],
         "knL": J[:, 4], "knR": J[:, 5], "anL": J[:, 7], "anR": J[:, 8], "head": J[:, HEAD], "ftL": J[:, 10], "ftR": J[:, 11],
         "pelvis": J[:, PELVIS], "neck": J[:, NECK]}
    if J.shape[1] >= 55:
        P["palmL"] = palm_from_joints(J, "left")
        P["palmR"] = palm_from_joints(J, "right")
    else:
        P["palmL"] = palm_from_forearm(P["elL"], P["wrL"])
        P["palmR"] = palm_from_forearm(P["elR"], P["wrR"])
    return P


def palm_from_forearm(E, W, ahead_m: float = 0.08, xp=np):
    """a palm centre `ahead_m` beyond the wrist along the forearm (COCO has no palm)"""
    return W + ahead_m * unit(W - E, xp)


def points_from_coco(X, ahead_m: float = 0.08) -> dict:
    """named points from COCO-17 (n,17,3) (NaN where missing); head = mean of the visible face points"""
    X = np.asarray(X, float)
    P = {n: X[:, CI[n]] for n in ("shL", "shR", "elL", "elR", "wrL", "wrR", "hipL", "hipR", "knL", "knR", "anL", "anR")}
    with np.errstate(invalid="ignore"):
        face = X[:, :5]
        okf = np.isfinite(face[..., 0])
        P["head"] = np.where(okf.any(1)[:, None], np.nansum(np.where(okf[..., None], face, 0), 1) / np.maximum(okf.sum(1), 1)[:, None], np.nan)
    P["palmL"] = palm_from_forearm(P["elL"], P["wrL"], ahead_m)
    P["palmR"] = palm_from_forearm(P["elR"], P["wrR"], ahead_m)
    P["pelvis"] = 0.5 * (P["hipL"] + P["hipR"])
    return P


# ------------------------------------------------------------------------------------------------ frames

def tilted_frame(shL, shR, hipL, hipR, xp=np) -> dict:
    """origin = mean of shoulders and hips; up a = mid-shoulders - mid-hips; left x = L_sh - R_sh without its a-part; front z = x x a"""
    o_ = _ops(xp)
    o = 0.25 * (shL + shR + hipL + hipR)
    a = unit(0.5 * (shL + shR) - 0.5 * (hipL + hipR), xp)
    x = shL - shR
    x = unit(x - o_.dot(x, a)[..., None] * a, xp)
    z = o_.cross(x, a)
    return {"o": o, "x": x, "up": a, "z": z}


def yaw_frame(shL, shR, xp=np) -> dict:
    """zouk_contacts.torso_frame from the shoulders: x = horizontal left, up = world y, z = x x up (front), origin mid-shoulders"""
    o_ = _ops(xp)
    d = shL - shR
    x = o_.stack([d[..., 0], o_.zeros_like(d[..., 0]), d[..., 2]], -1)
    x = unit(x, xp)
    up = o_.const(UP, x) + o_.zeros_like(x)
    z = o_.cross(x, up)
    return {"o": 0.5 * (shL + shR), "x": x, "up": up, "z": z}


def frames(P: dict, xp=np) -> dict:
    """{"tilt": tilted_frame, "yaw": yaw_frame} of a dancer's points"""
    return {"tilt": tilted_frame(P["shL"], P["shR"], P["hipL"], P["hipR"], xp), "yaw": yaw_frame(P["shL"], P["shR"], xp)}


# ------------------------------------------------------------------------------------------------ the Gauss linking integral

def gli_seg(p1, p2, p3, p4, xp=np):
    """Gauss linking integral of segment p1p2 with p3p4 (Klenin & Langowski 2000; relations.md 1.1), broadcast over leading dims.
    = sign((u x v).(A0 - B0)) x (solid angle of the view directions in which the two segments cross) / 4 pi"""
    o = _ops(xp)
    r13, r14, r23, r24 = p3 - p1, p4 - p1, p3 - p2, p4 - p2

    def nrm(a, b):
        return unit(o.cross(a, b), xp)
    n1, n2, n3, n4 = nrm(r13, r14), nrm(r14, r24), nrm(r24, r23), nrm(r23, r13)

    def a_(u, v):
        return o.asin(o.clip(o.dot(u, v), -1.0 + 1e-12, 1.0 - 1e-12))
    om = a_(n1, n2) + a_(n2, n3) + a_(n3, n4) + a_(n4, n1)
    s = o.sign(o.dot(o.cross(p4 - p3, p2 - p1), r13))
    return om * s / (4.0 * math.pi)


def gli_chain(A, B, xp=np, pairs: bool = False):
    """A (..., nA, 3), B (..., nB, 3) polylines -> G (...) summed over the (nA-1)(nB-1) segment pairs (+ the per-pair matrix
    (..., nA-1, nB-1) when pairs=True: which pair is crossed)"""
    o = _ops(xp)
    rows = []
    for i in range(A.shape[-2] - 1):
        row = [gli_seg(A[..., i, :], A[..., i + 1, :], B[..., j, :], B[..., j + 1, :], xp) for j in range(B.shape[-2] - 1)]
        rows.append(o.stack(row, -1))
    M = o.stack(rows, -2)
    G = M.sum(-1).sum(-1)
    return (G, M) if pairs else G


def passthrough_events(G, jump: float = 0.05, merge: int = 1) -> np.ndarray:
    """frame indices k (of the frame AFTER the jump) where G changes sign between k-1 and k with |G_k - G_{k-1}| > jump (a pass-through;
    going around a bone end changes the sign through 0 with a small step); events within `merge` frames of the previous one are merged.
    NaN frames never make an event."""
    G = np.asarray(G, float)
    d = np.diff(G)
    with np.errstate(invalid="ignore"):
        ev = np.flatnonzero((np.sign(G[1:]) != np.sign(G[:-1])) & (np.abs(d) > jump) & np.isfinite(d))
    out = [int(e) + 1 for i, e in enumerate(ev) if i == 0 or e - ev[i - 1] > merge]
    return np.asarray(out, int)


# ------------------------------------------------------------------------------------------------ closest points / signed separation (F2)

def closest_params(p1, q1, p2, q2, xp=np):
    """closest points of segments p1-q1 and p2-q2 -> (s, t, distance) (zouk_contacts.seg_seg)"""
    from .. import zouk_contacts as zc
    d, s, t = zc.seg_seg(p1, q1, p2, q2, xp)
    return s, t, d


def signed_sep(p1, q1, p2, q2, xp=np, lo: float = 0.02, hi: float = 0.98):
    """signed closest-point separation d_s = (c_A - c_B) . unit(u x v) of segments A = p1-q1, B = p2-q2 (DESIGN F2)
    -> (d_s, s, t, distance, interior) ; interior = both closest points strictly inside (lo, hi). sign(d_s) = the GLI sign of the pair
    wherever the closest points are interior, and it flips exactly at a through-crossing (the bone axes pass each other)."""
    o = _ops(xp)
    s, t, dist = closest_params(p1, q1, p2, q2, xp)
    u, v = q1 - p1, q2 - p2
    cA = p1 + u * s[..., None]
    cB = p2 + v * t[..., None]
    n = unit(o.cross(u, v), xp)
    ds = o.dot(cA - cB, n)
    interior = (s > lo) & (s < hi) & (t > lo) & (t < hi)
    return ds, s, t, dist, interior


# ------------------------------------------------------------------------------------------------ arm swivel (R10)

def swivel(S, E, W, xp=np):
    """angle (deg) of the elbow about the shoulder-wrist axis: 0 = below the line (toward -y), +-180 = above; and the weight
    sin^2(flexion) (0 for a straight arm, where the angle is undefined) -> (phi, weight)"""
    o = _ops(xp)
    a = unit(W - S, xp)
    e = E - S
    e = e - o.dot(e, a)[..., None] * a
    down = o.const([0.0, -1.0, 0.0], a) + o.zeros_like(a)
    d = down - o.dot(down, a)[..., None] * a
    alt = o.const([0.0, 0.0, 1.0], a) + o.zeros_like(a)
    alt = alt - o.dot(alt, a)[..., None] * a
    dn = o.norm(d, keepdims=True)
    d = o.where(dn > 0.1, d, alt)
    d = unit(d, xp)
    phi = o.atan2(o.dot(o.cross(d, e), a), o.dot(d, e)) * (180.0 / math.pi)
    u1, u2 = unit(E - S, xp), unit(W - E, xp)
    c = o.clip(o.dot(u1, u2), -1.0, 1.0)
    w = 1.0 - c * c                                    # sin^2 of the flexion
    return phi, w


def swivel_valid(S, W, xp=np, max_axis_y: float = 0.85):
    """frames where the swivel's reference (down, projected off the shoulder-wrist axis) is stable: the axis is not within ~30 deg of
    vertical (an arm raised overhead switches the reference and the angle jumps without any motion)"""
    a = unit(W - S, xp)
    return _ops(xp).abs(a[..., 1]) < max_axis_y


# ------------------------------------------------------------------------------------------------ rotations, swing-twist

def mat_to_quat(R, xp=np):
    """(...,3,3) -> (...,4) unit quaternions (w, x, y, z) with w >= 0 (Shepperd: the largest component first, so 180-degree rotations
    keep their relative signs), branch-free (numpy / torch)"""
    o = _ops(xp)
    r = lambda i, j: R[..., i, j]
    m00, m11, m22 = r(0, 0), r(1, 1), r(2, 2)
    mags = [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22]
    sq = [o.sqrt(o.maximum(m, 1e-12)) * 0.5 for m in mags]
    a, b, c, d = (r(2, 1) - r(1, 2)), (r(0, 2) - r(2, 0)), (r(1, 0) - r(0, 1)), None
    s01, s02, s12 = r(0, 1) + r(1, 0), r(0, 2) + r(2, 0), r(1, 2) + r(2, 1)
    cand = [o.stack([sq[0], a / (4 * sq[0]), b / (4 * sq[0]), c / (4 * sq[0])], -1),
            o.stack([a / (4 * sq[1]), sq[1], s01 / (4 * sq[1]), s02 / (4 * sq[1])], -1),
            o.stack([b / (4 * sq[2]), s01 / (4 * sq[2]), sq[2], s12 / (4 * sq[2])], -1),
            o.stack([c / (4 * sq[3]), s02 / (4 * sq[3]), s12 / (4 * sq[3]), sq[3]], -1)]
    M = o.stack(mags, -1)
    best = M.argmax(-1) if o.np else M.argmax(-1)
    q = cand[0]
    for i in (1, 2, 3):
        sel = (best == i)[..., None]
        q = o.where(sel, cand[i], q)
    q = q * o.where(q[..., :1] < 0, -1.0, 1.0)
    return q / o.maximum(o.norm(q, keepdims=True), 1e-12)


def rotvec_to_mat(rv, xp=np):
    """axis-angle (...,3) -> (...,3,3) (Rodrigues; numpy / torch)"""
    o = _ops(xp)
    th = o.norm(rv, keepdims=True)
    k = rv / o.maximum(th, 1e-12)
    th = th[..., 0]
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = o.zeros_like(kx)
    Kx = o.stack([o.stack([z, -kz, ky], -1), o.stack([kz, z, -kx], -1), o.stack([-ky, kx, z], -1)], -2)
    if o.np:
        I = np.broadcast_to(np.eye(3), Kx.shape)
        s, c = np.sin(th)[..., None, None], np.cos(th)[..., None, None]
    else:
        I = xp.eye(3, dtype=rv.dtype, device=rv.device).expand(Kx.shape)
        s, c = xp.sin(th)[..., None, None], xp.cos(th)[..., None, None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


def swing_twist(R, axis, xp=np):
    """split rotations R (...,3,3) into swing x twist about `axis` (3,) (R = R_swing @ R_twist) -> (twist_deg (...), R_swing)"""
    o = _ops(xp)
    q = mat_to_quat(R, xp)
    ax = axis if (not o.np and xp.is_tensor(axis)) else o.const(axis, R)
    ax = unit(ax, xp)
    p = (q[..., 1:] * ax).sum(-1)
    tw = 2.0 * o.atan2(p, q[..., 0])                 # radians in (-2pi, 2pi]
    twd = tw * (180.0 / math.pi)
    twd = ((twd + 180.0) % 360.0) - 180.0
    Rt = rotvec_to_mat(ax * (twd * (math.pi / 180.0))[..., None], xp)
    Rs = R @ (Rt.transpose(-1, -2) if not o.np else np.swapaxes(Rt, -1, -2))
    return twd, Rs


def chain_twist(Rloc, axes, xp=np) -> dict:
    """Rloc (..., n, 3, 3) local rotations of a chain, axes (n,3) rest bone axes -> {"twist" (..., n) deg, "total", "wrapped"
    (wrap_360 of the total), "wrap" (total - wrapped: a multiple of 360 where the chain is wound, R16)}"""
    o = _ops(xp)
    tws = []
    for i in range(Rloc.shape[-3]):
        tw, _ = swing_twist(Rloc[..., i, :, :], axes[i], xp)
        tws.append(tw)
    tw = o.stack(tws, -1)
    tot = tw.sum(-1)
    wr = ((tot + 180.0) % 360.0) - 180.0
    return {"twist": tw, "total": tot, "wrapped": wr, "wrap": tot - wr}


def global_rotations(global_orient, body_pose) -> np.ndarray:
    """(T,3), (T,63) axis-angle -> (T,22,3,3) world rotations of the 22 body joints (numpy FK of the kintree)"""
    T = len(global_orient)
    loc = np.concatenate([np.asarray(global_orient, float)[:, None], np.asarray(body_pose, float).reshape(T, 21, 3)], 1)
    Rl = rotvec_to_mat(loc)
    Rg = np.zeros_like(Rl)
    for j in range(22):
        Rg[:, j] = Rl[:, j] if PARENTS22[j] < 0 else Rg[:, PARENTS22[j]] @ Rl[:, j]
    return Rg


def local_rotations(global_orient, body_pose) -> np.ndarray:
    T = len(global_orient)
    loc = np.concatenate([np.asarray(global_orient, float)[:, None], np.asarray(body_pose, float).reshape(T, 21, 3)], 1)
    return rotvec_to_mat(loc)


def segment_headings(Rg: np.ndarray, J: np.ndarray) -> dict:
    """R15: yaw (deg, wrapped) of the pelvis / spine3 / head forward axes, the hip line, each thigh / shank / foot frame (numpy)"""
    z = np.array([0.0, 0.0, 1.0])
    out = {"pelvis": heading(Rg[:, PELVIS] @ z), "spine3": heading(Rg[:, SPINE3] @ z), "head": heading(Rg[:, HEAD] @ z),
           "hipline": line_heading(J[:, 1], J[:, 2]), "shoulderline": line_heading(J[:, 16], J[:, 17])}
    for s, x in (("left", "L"), ("right", "R")):
        out[f"thigh{x}"] = heading(Rg[:, HIP[s]] @ z)
        out[f"shank{x}"] = heading(Rg[:, KNEE[s]] @ z)
        out[f"foot{x}"] = heading(Rg[:, ANKLE[s]] @ z)
    return out


# ------------------------------------------------------------------------------------------------ the relation margins R1-R13

def _h(v, xp=np):
    """horizontal part of v"""
    o = _ops(xp)
    return o.stack([v[..., 0], o.zeros_like(v[..., 0]), v[..., 2]], -1)


def between_margin(K, P0, P1, xp=np):
    """R11: K between P0 and P1 horizontally: m = min(t, 1 - t) |a| (positive = between), and |a|"""
    o = _ops(xp)
    a = _h(P1 - P0, xp)
    k = _h(K - P0, xp)
    L = o.norm(a)
    t = o.dot(k, a) / o.maximum(L * L, 1e-9)
    m = o.where(t < 1.0 - t, t, 1.0 - t) * L
    return m, L


def r8_angle(p, A, B, xp=np):
    """R8: point p against segment A-B (her forearm): (alpha deg: +90 = directly above, -90 below, 0 beside; closest parameter t; dist)"""
    o = _ops(xp)
    from .. import zouk_contacts as zc
    d, _, t = zc.seg_seg(p, p, A, B, xp)
    f = unit(B - A, xp)
    c = A + (B - A) * t[..., None]
    up = o.const(UP, f) + o.zeros_like(f)
    b1 = up - o.dot(up, f)[..., None] * f
    alt = o.const([1.0, 0.0, 0.0], f) + o.zeros_like(f)
    b1 = o.where(o.norm(b1, keepdims=True) > 0.2, b1, alt - o.dot(alt, f)[..., None] * f)
    b1 = unit(b1, xp)
    b2 = o.cross(f, b1)
    r = p - c
    al = o.atan2(o.dot(r, b1), o.abs(o.dot(r, b2))) * (180.0 / math.pi)
    return al, t, d


ARM_CHAIN = {"L": ("shL", "elL", "wrL"), "R": ("shR", "elR", "wrR")}


def arm_chain(P: dict, side: str, xp=np):
    o = _ops(xp)
    return o.stack([P[n] for n in ARM_CHAIN[side]], -2)


def chain_distance(A, B, xp=np):
    """minimum distance between the segments of two chains (..., nA, 3), (..., nB, 3)"""
    from .. import zouk_contacts as zc
    o = _ops(xp)
    ds = []
    for i in range(A.shape[-2] - 1):
        for j in range(B.shape[-2] - 1):
            d, _, _ = zc.seg_seg(A[..., i, :], A[..., i + 1, :], B[..., j, :], B[..., j + 1, :], xp)
            ds.append(d)
    D = o.stack(ds, -1)
    return D.min(-1) if o.np else D.min(-1).values


# relation ids of the arm pairings (his side x her side) used by R9
R9_PAIRS = {"R9_RL": ("R", "L"), "R9_LR": ("L", "R"), "R9_RR": ("R", "R"), "R9_LL": ("L", "L")}


def margins(PL: dict, PF: dict, xp=np, half_width_m: float = 0.14) -> dict:
    """the signed margins and gates of the couple relations from the two dancers' point sets (numpy or torch, same formulas)
    -> {id: {"m": margin, "gate": defined-when (bool), "aux": {...}}} for R1 R2 R4(fb, fl) R5 R6 R7 R8 R9_* R11 R12 R13 (R10 self, see swivel).
    Gates here are geometric only (from the same points); the caller decides which point set says "defined"."""
    o = _ops(xp)
    out = {}
    fL, fF = frames(PL, xp), frames(PF, xp)
    cL, cF = 0.5 * (PL["hipL"] + PL["hipR"]), 0.5 * (PF["hipL"] + PF["hipR"])
    dLF = _h(cF - cL, xp)
    pd = o.norm(dLF)
    dn = dLF / o.maximum(pd, 1e-9)[..., None]
    zL, zF = fL["yaw"]["z"], fF["yaw"]["z"]
    out["R1_lead"] = {"m": o.dot(zL, dn), "gate": (pd > 0.15) & (pd < 1.2)}
    out["R1_follow"] = {"m": -o.dot(zF, dn), "gate": (pd > 0.15) & (pd < 1.2)}
    out["R2"] = {"m": -o.dot(zL, zF), "gate": pd < 9.0}            # + = face to face, - = same way, |m| < 0.35 = side
    # R4: his right palm on her torso (tilted frame); m_fb + = front, m_fl + = beyond her half-width (her side)
    tF = fF["tilt"]
    r = PL["palmR"] - tF["o"]
    out["R4_fb"] = {"m": o.dot(r, tF["z"]), "gate": pd < 9.0}
    out["R4_fl"] = {"m": o.abs(o.dot(r, tF["x"])) - half_width_m, "gate": pd < 9.0}
    # R5: her left hand on his right shoulder: front / back (his yaw front) and top
    rr = PF["palmL"] - PL["shR"]
    near5 = o.norm(rr) < 0.15
    out["R5_fb"] = {"m": o.dot(rr, zL), "gate": near5}
    out["R5_top"] = {"m": rr[..., 1], "gate": near5}
    # R6 / R7: his right elbow vs her left elbow
    de = PL["elR"] - PF["elL"]
    ed = o.norm(de)
    out["R6"] = {"m": de[..., 1], "gate": ed < 0.30}
    out["R7"] = {"m": o.dot(de, fF["yaw"]["x"]), "gate": ed < 0.30}
    # R8: his right elbow over / under her left forearm
    al, t8, d8 = r8_angle(PL["elR"], PF["elL"], PF["wrL"], xp)
    out["R8"] = {"m": al, "gate": (t8 > 0.05) & (t8 < 0.95) & (d8 < 0.15)}
    # R9: arm chains (GLI) for the four pairings
    for rid, (sa, sb) in R9_PAIRS.items():
        A, B = arm_chain(PL, sa, xp), arm_chain(PF, sb, xp)
        G = gli_chain(A, B, xp)
        dmin = chain_distance(A, B, xp)
        out[rid] = {"m": G, "gate": (o.abs(G) > 0.02) | (dmin < 0.10), "aux": {"dist": dmin}}
    # R11: knee between the partner's knees (horizontal)
    m11, L11 = between_margin(PL["knR"], PF["knR"], PF["knL"], xp)
    out["R11_lead"] = {"m": m11, "gate": (pd < 0.5) & (L11 > 0.10)}
    m11b, L11b = between_margin(PF["knR"], PL["knR"], PL["knL"], xp)
    out["R11_follow"] = {"m": m11b, "gate": (pd < 0.5) & (L11b > 0.10)}
    # R12: order along the couple's lateral axis: his knee / ankle minus hers, for the 4 cross pairs each
    lat = o.cross(dn, o.const(UP, dn) + o.zeros_like(dn))
    for kind, (a, b) in (("kn", ("knL", "knR")), ("an", ("anL", "anR"))):
        for hs in (a, b):
            for fs in (a, b):
                out[f"R12_{hs[-1]}{fs[-1]}_{kind}"] = {"m": o.dot(PL[hs] - PF[fs], lat), "gate": pd < 0.5}
    # R13: head side in embraces
    dh = PF["head"] - PL["head"]
    out["R13"] = {"m": o.dot(dh, fL["yaw"]["x"]), "gate": o.norm(dh) < 0.35}
    return out


# relation table: id -> (kind, dead zone for deciding, dead zone in the fit, band key)
#   kind "hinge" = enforced when decided, "position" = evidence only (hinge only when decided), "hold" = R9 (E_sep / E_gli)
REL_TABLE = {
    "R1_lead": ("hinge", 0.20, 0.05, None), "R1_follow": ("hinge", 0.20, 0.05, None), "R2": ("hinge", 0.35, 0.05, None),
    "R5_fb": ("hinge", 0.03, 0.01, None),
    "R6": ("position", 0.04, 0.01, "R9_RL"), "R7": ("hinge", 0.06, 0.015, "R9_RL"), "R8": ("hinge", 25.0, 8.0, "R9_RL"),
    "R9_RL": ("hold", 0.02, 0.0, "R9_RL"), "R9_LR": ("hold", 0.02, 0.0, "R9_LR"), "R9_RR": ("hold", 0.02, 0.0, "R9_RR"),
    "R9_LL": ("hold", 0.02, 0.0, "R9_LL"),
    "R11_lead": ("hinge", 0.04, 0.01, "legs"), "R11_follow": ("hinge", 0.04, 0.01, "legs"),
    "R13": ("position", 0.04, 0.01, None),
}
for _k in ("kn", "an"):
    for _a in "LR":
        for _b in "LR":
            REL_TABLE[f"R12_{_a}{_b}_{_k}"] = ("hinge", 0.04, 0.01, "legs")
# the joints (dancer, COCO index) each margin depends on, for its noise sigma_m^2 = sum a_i^T S_i a_i
REL_JOINTS = {
    "R1_lead": [("L", "shL"), ("L", "shR"), ("L", "hipL"), ("L", "hipR"), ("F", "hipL"), ("F", "hipR")],
    "R1_follow": [("F", "shL"), ("F", "shR"), ("F", "hipL"), ("F", "hipR"), ("L", "hipL"), ("L", "hipR")],
    "R2": [("L", "shL"), ("L", "shR"), ("F", "shL"), ("F", "shR")],
    "R4_fb": [("L", "elR"), ("L", "wrR"), ("F", "shL"), ("F", "shR"), ("F", "hipL"), ("F", "hipR")],
    "R4_fl": [("L", "elR"), ("L", "wrR"), ("F", "shL"), ("F", "shR"), ("F", "hipL"), ("F", "hipR")],
    "R5_fb": [("F", "elL"), ("F", "wrL"), ("L", "shR"), ("L", "shL")],
    "R6": [("L", "elR"), ("F", "elL")], "R7": [("L", "elR"), ("F", "elL"), ("F", "shL"), ("F", "shR")],
    "R8": [("L", "elR"), ("F", "elL"), ("F", "wrL")],
    "R11_lead": [("L", "knR"), ("F", "knL"), ("F", "knR")], "R11_follow": [("F", "knR"), ("L", "knL"), ("L", "knR")],
    "R13": [("L", "shL"), ("L", "shR")],
}
for _rid, (_a, _b) in R9_PAIRS.items():
    REL_JOINTS[_rid] = [("L", n) for n in ARM_CHAIN[_a]] + [("F", n) for n in ARM_CHAIN[_b]]
for _k in ("kn", "an"):
    for _a in "LR":
        for _b in "LR":
            REL_JOINTS[f"R12_{_a}{_b}_{_k}"] = [("L", f"{_k}{_a}"), ("F", f"{_k}{_b}"), ("L", "hipL"), ("L", "hipR"), ("F", "hipL"),
                                                ("F", "hipR")]
