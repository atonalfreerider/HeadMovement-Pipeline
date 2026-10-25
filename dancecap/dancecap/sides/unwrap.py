"""R16 chain wraps and the spine split (DESIGN 2.4 "Wraps", 4.1 `unwrap.py`). Pure numpy + scipy, no take constants.

A *wrap* is the candy-wrapper skin: the local twists of a chain (spine1-3, neck + head, hip-knee-ankle) are huge and cancel, so the
joints sit where they should while the skin between two joints is twisted by up to 360 deg. Neither a joint test nor a reprojection
sees it. R16 measures it per frame as w = sum(local twists) - Sigma*, where every local twist is wrapped to (-180, 180] and Sigma* is
the representative of the sum mod 360 nearest the chain's *measured* relative heading (from the fit's own joints: shoulder line minus
hip line for the spine); |w| >= 300 deg is a wrap.

`unwrap_chain` removes it in closed form without moving any joint: every intermediate joint's GLOBAL frame is turned about its own
bone (G_j' = G_j T(a_j, phi_j)), which leaves the child joint where it was, and the chain's last joint keeps its world orientation
exactly (everything above it is byte-for-byte the same rotation). The phi_j are chosen so the local twists become shares of Sigma*.
`split_spine` is the L0 fallback for an over-twist that is not a wrap: the three spine rotations become the equal geodesic split of
their product (spine3's world rotation kept), accepted only when every spine joint ends <= 35 deg and no joint moves > 3 cm.

Pose layout: (T, J, 3) axis-angle local rotations, J >= 22, SMPL-X order (0 = global orient, 1..21 = body_pose). Rest joints: the
shaped rest joints (>= 22, 3), e.g. `stitch.rest_joints(betas)`.

The headings / wrapping come from `sides.geom` (wrap_deg, line_heading, unit); the thresholds from `SidesCfg.unwrap_cfg()`."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import geom
from .geom import PARENTS22
# SMPL-X body joints used here
PELVIS, L_HIP, R_HIP, SPINE1, L_KNEE, R_KNEE, SPINE2, L_ANKLE, R_ANKLE, SPINE3, L_FOOT, R_FOOT, NECK = range(13)
HEAD, L_SHOULDER, R_SHOULDER, L_EYE, R_EYE = 15, 16, 17, 23, 24


@dataclass(frozen=True)
class Chain:
    """A serial chain proximal -> distal. Every joint but the last has exactly one child, the next joint. axes[i] = (a, b): joint i's
    twist axis is the rest bone a -> b (the last joint uses the bone INTO it, so the closed-form unwrap sets its twist exactly)."""
    name: str
    joints: tuple
    axes: tuple
    shares: tuple                 # how Sigma* is split over the joints (sums to 1)
    limit_deg: tuple              # per-joint twist gate (DESIGN 3.4), the region an unwrap run grows into
    total_limit_deg: float        # |sum| gate (DESIGN 3.4)
    guard: str                    # "spine" | "neck" | "limit": how Sigma* is checked against the joints


CHAINS = {
    "spine": Chain("spine", (SPINE1, SPINE2, SPINE3), ((SPINE1, SPINE2), (SPINE2, SPINE3), (SPINE2, SPINE3)),
                   (0.3, 0.35, 0.35), (35.0, 35.0, 35.0), 70.0, "spine"),
    "neck": Chain("neck", (NECK, HEAD), ((NECK, HEAD), (NECK, HEAD)), (0.5, 0.5), (70.0, 70.0), 70.0, "neck"),
    "leg_l": Chain("leg_l", (L_HIP, L_KNEE, L_ANKLE), ((L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_KNEE, L_ANKLE)),
                   (0.6, 0.1, 0.3), (50.0, 45.0, 40.0), 90.0, "limit"),
    "leg_r": Chain("leg_r", (R_HIP, R_KNEE, R_ANKLE), ((R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_KNEE, R_ANKLE)),
                   (0.6, 0.1, 0.3), (50.0, 45.0, 40.0), 90.0, "limit"),
}


@dataclass(frozen=True)
class UnwrapCfg:
    """Thresholds of the unwrap step ([I] = starting values). The pipeline builds it from SidesCfg (`SidesCfg.unwrap_cfg()`, the
    take's [sides] table); the defaults here equal SidesCfg's."""
    wrap_deg: float = 300.0          # R16: |w| >= this is a wrap (DESIGN 2.1)
    candy_deg: float = 90.0          # [I] a single local twist beyond this is never anatomical: re-expressed like a wrap
    guard_deg: float = 30.0          # DESIGN: Sigma* within 30 deg of the measured relative heading
    guard_total_deg: float = 90.0    # [I] and |Sigma*| <= this (a torso turned 150 deg against the pelvis is a facing error, not a wrap)
    ramp_frames: int = 6             # [I] cosine ramp of the twist change outside each run (the skin must not pop)
    split_joint_max_deg: float = 35.0   # DESIGN: split accepted only if every spine joint <= 35 deg ...
    split_move_max_m: float = 0.03      # ... and joints move <= 3 cm
    split_ramp_frames: int = 4       # [I]
    guard_mode: str = "flag"         # [I] "flag": re-express the whole event, report guard failures as not-pure (they need the yaw /
                                     # solver windows); "gate": DESIGN-strict, re-express only guard-ok frames (see build_P.md)
    chains: tuple = ("spine", "neck", "leg_l", "leg_r")


# ------------------------------------------------------------------------------------------------ small helpers (sides.geom)

_wrap = geom.wrap_deg
_unit = geom.unit
_line_heading = geom.line_heading


def pose_from_fit(fit: dict) -> np.ndarray:
    """(T, 22, 3) local axis-angle of the body: global_orient + body_pose"""
    go = np.asarray(fit["global_orient"], float)
    T = len(go)
    return np.concatenate([go[:, None], np.asarray(fit["body_pose"], float).reshape(T, 21, 3)], 1)


def rotmats(pose: np.ndarray) -> np.ndarray:
    p = np.asarray(pose, float)
    return Rot.from_rotvec(p.reshape(-1, 3)).as_matrix().reshape(p.shape[:-1] + (3, 3))


def rotvecs(R: np.ndarray) -> np.ndarray:
    return Rot.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def axis_rot(axis: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """(T, 3, 3) rotations about a fixed unit axis by deg (T,)"""
    d = np.radians(np.asarray(deg, float))
    return Rot.from_rotvec(np.asarray(axis, float)[None] * d[:, None]).as_matrix()


def twist_deg(R: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """swing-twist (R = S T(axis, tw), the twist applied first, S perpendicular to the axis): tw in (-180, 180] deg"""
    q = Rot.from_matrix(R.reshape(-1, 3, 3)).as_quat()           # x y z w
    p = q[:, :3] @ np.asarray(axis, float)
    return _wrap(2.0 * np.degrees(np.arctan2(p, q[:, 3]))).reshape(R.shape[:-2])


def fk22(R: np.ndarray, rest: np.ndarray, transl: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """global rotations (T, 22, 3, 3) and joint positions (T, 22, 3) of the body from local rotations (T, >=22, 3, 3) and the shaped
    rest joints (SMPL-X: J_0 = r_0 + transl, J_j = J_p + G_p (r_j - r_p))"""
    T = R.shape[0]
    rest = np.asarray(rest, float)
    G = np.zeros((T, 22, 3, 3))
    X = np.zeros((T, 22, 3))
    tr = np.zeros((T, 3)) if transl is None else np.asarray(transl, float)
    for j in range(22):
        p = PARENTS22[j]
        if p < 0:
            G[:, j] = R[:, j]
            X[:, j] = rest[j] + tr
        else:
            G[:, j] = G[:, p] @ R[:, j]
            X[:, j] = X[:, p] + np.einsum("tab,b->ta", G[:, p], rest[j] - rest[p])
    return G, X


def chain_axes(rest: np.ndarray, chain: Chain) -> np.ndarray:
    rest = np.asarray(rest, float)
    return np.stack([_unit(rest[b] - rest[a]) for a, b in chain.axes])


def chain_twists(R: np.ndarray, rest: np.ndarray, chain: Chain) -> np.ndarray:
    """(T, n) local twists (deg) of the chain's joints about their rest bone axes"""
    ax = chain_axes(rest, chain)
    return np.stack([twist_deg(R[:, j], ax[i]) for i, j in enumerate(chain.joints)], 1)


def relative_heading(joints: np.ndarray, chain: Chain) -> np.ndarray:
    """the chain's twist as the joints show it (deg), NaN when the chain has no such measure ("limit" chains)"""
    J = np.asarray(joints, float)
    if chain.guard == "spine":
        return _wrap(_line_heading(J[:, L_SHOULDER], J[:, R_SHOULDER]) - _line_heading(J[:, L_HIP], J[:, R_HIP]))
    if chain.guard == "neck":
        if J.shape[1] <= R_EYE:
            return np.full(len(J), np.nan)
        return _wrap(_line_heading(J[:, L_EYE], J[:, R_EYE]) - _line_heading(J[:, L_SHOULDER], J[:, R_SHOULDER]))
    return np.full(len(J), np.nan)


def runs_of(mask: np.ndarray) -> list:
    """[(a, b)] inclusive index runs of a boolean mask"""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def cos_ramp_up(n: int) -> np.ndarray:
    """n weights strictly inside (0, 1) rising toward the run (zero slope at both ends)"""
    x = (np.arange(n) + 0.5) / max(n, 1)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def run_weights(core: np.ndarray, n_ramp: int) -> np.ndarray:
    """1 on the core frames, cosine ramps of n_ramp frames outside each run (max where ramps of two runs meet)"""
    T = len(core)
    w = np.asarray(core, float).copy()
    up = cos_ramp_up(n_ramp)
    for a, b in runs_of(core):
        for i in range(n_ramp):
            k0, k1 = a - n_ramp + i, b + n_ramp - i
            if 0 <= k0 < T:
                w[k0] = max(w[k0], up[i])
            if 0 <= k1 < T:
                w[k1] = max(w[k1], up[i])
    return w


# ------------------------------------------------------------------------------------------------ R16

def chain_wraps(fit: dict | None = None, *, pose: np.ndarray | None = None, rest: np.ndarray | None = None,
                joints: np.ndarray | None = None, cfg: UnwrapCfg = UnwrapCfg(), chains: tuple | None = None) -> dict:
    """R16 per chain and frame (DESIGN 2.1): Sigma* = wrap360(sum of the wrapped local twists), w = sum - Sigma*, wrapped = |w| >= 300.
    Sigma* is a continuous function of the rotations (a representation flip of one joint changes the sum by 360, not Sigma*), so it is
    the unwrap's target total. Give a fit (global_orient, body_pose, joints, betas) or pose (T, >=22, 3) + rest (+ joints for the
    guard; without them the joints come from fk22). -> {chain: {twists (T, n), sum, sigma_star, h_rel, w, wrapped, candy, over,
    guard_ok}}: candy = a single twist > candy_deg; over = beyond the chain's 3.4 limits; guard_ok = the DESIGN guard (Sigma* within
    guard_deg of the measured relative heading, |Sigma*| <= guard_total_deg): where it fails the chain is not a *pure* wrap."""
    if pose is None:
        pose = pose_from_fit(fit)
    if rest is None:
        rest = _rest_from_fit(fit)
    R = rotmats(pose)
    if joints is None:
        joints = fit["joints"] if fit is not None and "joints" in fit else fk22(R, rest)[1]
    out = {}
    for name in (chains or cfg.chains):
        ch = CHAINS[name]
        tw = chain_twists(R, rest, ch)
        s = tw.sum(1)
        h = relative_heading(joints, ch)
        star = _wrap(s)
        w = s - star
        wrapped = np.abs(w) >= cfg.wrap_deg
        candy = (np.abs(tw) > cfg.candy_deg).any(1)
        over = (np.abs(tw) > np.asarray(ch.limit_deg)[None]).any(1) | (np.abs(s) > ch.total_limit_deg)
        guard = np.abs(star) <= cfg.guard_total_deg
        if ch.guard in ("spine", "neck"):
            guard &= np.isfinite(h) & (np.abs(_wrap(star - np.nan_to_num(h))) <= cfg.guard_deg)
        out[name] = {"twists": tw, "sum": s, "sigma_star": star, "h_rel": h, "w": w, "wrapped": wrapped, "candy": candy,
                     "over": over, "guard_ok": guard}
    return out


def unwrap_runs(d: dict, cfg: UnwrapCfg = UnwrapCfg()) -> np.ndarray:
    """The frames an unwrap re-expresses for one chain (a chain_wraps entry): every run of over-limit / wrapped / candy frames that
    holds at least one wrap or candy frame (the twist builds up through the over-limit frames, so the whole event is re-expressed at
    once and no skin pops inside it). guard_mode "gate" (DESIGN-strict) drops the frames whose guard fails."""
    core0 = d["wrapped"] | d["candy"]
    lab = np.zeros(len(core0), bool)
    for a, b in runs_of(core0 | d["over"]):
        if core0[a:b + 1].any():
            lab[a:b + 1] = True
    if cfg.guard_mode == "gate":
        lab &= d["guard_ok"]
    return lab


def _run_target(star: np.ndarray, w: np.ndarray) -> np.ndarray:
    """the target total per frame: Sigma* made continuous along each weighted run (it is continuous except where it crosses
    +-180), on the 360 branch with the smallest median magnitude"""
    out = star.copy()
    for a, b in runs_of(w > 0):
        seg = np.degrees(np.unwrap(np.radians(star[a:b + 1])))
        seg -= 360.0 * np.round(np.median(seg) / 360.0)
        out[a:b + 1] = seg
    return out


def _rest_from_fit(fit: dict) -> np.ndarray:
    from .. import stitch   # the SMPL-X model file is needed only here
    return stitch.rest_joints(np.asarray(fit["betas"], float))


def _continuous(phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """phi (T,) deg made continuous along each nonzero-weight run, on the branch whose run ends are the shortest twists"""
    out = phi.copy()
    for a, b in runs_of(w > 0):
        seg = np.degrees(np.unwrap(np.radians(phi[a:b + 1])))
        seg -= 360.0 * np.round(seg[0] / 360.0)        # the ramp-in end is the short way round
        out[a:b + 1] = seg
    return out


def unwrap_chain(pose: np.ndarray, rest: np.ndarray, chain: Chain | str, weight: np.ndarray, sigma_star: np.ndarray) -> tuple[np.ndarray, dict]:
    """Closed-form re-expression of one chain. weight (T,) in [0, 1]: 1 = the twists become shares of sigma_star (T,), 0 = untouched,
    in between the same twist moves scaled (ramps). No joint moves (up to float round-off) and the last joint's world rotation is kept.
    -> (new pose (T, J, 3), {twists_before, twists_after})"""
    ch = CHAINS[chain] if isinstance(chain, str) else chain
    pose = np.asarray(pose, float)
    R = rotmats(pose)
    ax = chain_axes(rest, ch)
    w = np.clip(np.asarray(weight, float), 0.0, 1.0)
    act = w > 0
    before = chain_twists(R, rest, ch)
    Rn = R.copy()
    if not act.any():
        return pose.copy(), {"twists_before": before, "twists_after": before.copy()}
    idx = np.flatnonzero(act)
    n = len(ch.joints)
    target = np.asarray(sigma_star, float)[:, None] * np.asarray(ch.shares, float)[None]
    # full (w = 1) phis, computed in sequence: joint i sees the left factor T(a_{i-1}, -phi_{i-1}) of the previous joint's move
    phi = np.zeros((len(R), n - 1))
    left = np.repeat(np.eye(3)[None], len(R), 0)
    for i in range(n - 1):
        j = ch.joints[i]
        Ri = left[idx] @ R[idx, j]
        tau = twist_deg(Ri, ax[i])
        phi[idx, i] = _wrap(target[idx, i] - tau)
        phi[:, i] = _continuous(phi[:, i], w)
        left = axis_rot(ax[i], -phi[:, i] * w)
    # apply the scaled moves
    prev = np.repeat(np.eye(3)[None], len(idx), 0)
    for i in range(n):
        j = ch.joints[i]
        Ri = prev @ R[idx, j]
        if i < n - 1:
            Ri = Ri @ axis_rot(ax[i], phi[idx, i] * w[idx])
            prev = axis_rot(ax[i], -phi[idx, i] * w[idx])
        Rn[idx, j] = Ri
    newpose = pose.copy()
    for j in ch.joints:
        newpose[idx, j] = rotvecs(Rn[idx, j])
    return newpose, {"twists_before": before, "twists_after": chain_twists(Rn, rest, ch)}


def split_spine(pose: np.ndarray, rest: np.ndarray, transl: np.ndarray | None, core: np.ndarray,
                cfg: UnwrapCfg = UnwrapCfg()) -> tuple[np.ndarray, dict]:
    """L0 equal geodesic split of spine1-3 on the `core` frames (cosine ramps outside), spine3's world rotation kept; each run is
    accepted only when every spine twist ends <= split_joint_max_deg and no body joint moves more than split_move_max_m, otherwise
    that run (with its ramps) is left as it was. -> (new pose, {runs: [{a, b, accepted, max_twist_deg, max_move_m}]})"""
    pose = np.asarray(pose, float)
    ch = CHAINS["spine"]
    R = rotmats(pose)
    _, X0 = fk22(R, rest, transl)
    out = pose.copy()
    rep = {"runs": []}
    ax = chain_axes(rest, ch)
    for a, b in runs_of(core):
        m = np.zeros(len(pose), bool)
        m[a:b + 1] = True
        w = run_weights(m, cfg.split_ramp_frames)
        idx = np.flatnonzero(w > 0)
        R1, R2, R3 = R[idx, SPINE1], R[idx, SPINE2], R[idx, SPINE3]
        prod = R1 @ R2 @ R3
        Q = Rot.from_rotvec(Rot.from_matrix(prod).as_rotvec() / 3.0).as_matrix()
        ww = w[idx]
        s1 = (Rot.from_matrix(R1) * Rot.from_rotvec(Rot.from_matrix(np.swapaxes(R1, 1, 2) @ Q).as_rotvec() * ww[:, None])).as_matrix()
        s2 = (Rot.from_matrix(R2) * Rot.from_rotvec(Rot.from_matrix(np.swapaxes(R2, 1, 2) @ Q).as_rotvec() * ww[:, None])).as_matrix()
        s3 = np.swapaxes(s1 @ s2, 1, 2) @ prod          # spine3's world rotation exactly as before
        Rn = R[idx].copy()
        Rn[:, SPINE1], Rn[:, SPINE2], Rn[:, SPINE3] = s1, s2, s3
        _, X1 = fk22(Rn, rest, None if transl is None else np.asarray(transl, float)[idx])
        move = float(np.linalg.norm(X1 - X0[idx], axis=-1).max())
        tw = np.stack([twist_deg(Rn[:, j], ax[i]) for i, j in enumerate(ch.joints)], 1)
        core_i = (idx >= a) & (idx <= b)
        mt = float(np.abs(tw[core_i]).max())
        ok = (mt <= cfg.split_joint_max_deg) and (move <= cfg.split_move_max_m)
        rep["runs"].append({"a": a, "b": b, "accepted": bool(ok), "max_twist_deg": round(mt, 2), "max_move_m": round(move, 4)})
        if ok:
            for jj, Rj in ((SPINE1, s1), (SPINE2, s2), (SPINE3, s3)):
                out[idx, jj] = rotvecs(Rj)
    return out, rep


def unwrap_fit(fit: dict, cfg: UnwrapCfg = UnwrapCfg(), rest: np.ndarray | None = None, fk=None, split: bool = True,
               frames: np.ndarray | None = None) -> tuple[dict, dict]:
    """The prepare-step rung: the R16 re-expression of every chain (each event = a run of over-limit frames holding a wrap or a candy
    twist, target total Sigma* made continuous along the run, cosine ramps), then (split=True) the spine split on the spine frames
    still over the limits. frames (T,) bool limits both to those frames (None = all). Returns (new fit dict: global_orient,
    body_pose, joints replaced; everything else shared with the input, report). `fk(new_fit, idx) -> joints (len(idx), J, 3)`
    recomputes the joints of the changed frames (default: `stitch.fk_joints` of all 55 joints with the fit's betas)."""
    if rest is None:
        rest = _rest_from_fit(fit)
    pose = pose_from_fit(fit)
    T = len(pose)
    allow = np.ones(T, bool) if frames is None else np.asarray(frames, bool)
    joints = np.asarray(fit["joints"], float)
    rep = {"chains": {}, "split": None, "cfg": {"guard_mode": cfg.guard_mode, "ramp_frames": cfg.ramp_frames}}
    cur = pose.copy()
    changed = np.zeros(T, bool)
    det = chain_wraps(pose=cur, rest=rest, joints=joints, cfg=cfg)
    for name in cfg.chains:
        d = det[name]
        lab = unwrap_runs(d, cfg) & allow
        # runs whose ramps would touch are one event
        merged = np.zeros(T, bool)
        rr = runs_of(lab)
        i = 0
        while i < len(rr):
            a, b = rr[i]
            while i + 1 < len(rr) and rr[i + 1][0] - b <= 2 * cfg.ramp_frames:
                i += 1
                b = rr[i][1]
            merged[a:b + 1] = True
            i += 1
        crep = {"wrapped_frames_before": int((d["wrapped"] & allow).sum()),
                "candy_frames_before": int(((d["wrapped"] | d["candy"]) & allow).sum()), "runs": [], "refused": []}
        for a, b in runs_of(merged):
            m = np.zeros(T, bool)
            m[a:b + 1] = True
            w = run_weights(m, cfg.ramp_frames) * allow
            span = w > 0
            target = _run_target(d["sigma_star"], w)
            trial, _ = unwrap_chain(cur, rest, name, w, target)
            aft = chain_wraps(pose=trial, rest=rest, joints=joints, cfg=cfg, chains=(name,))[name]
            mb, ma = float(np.abs(d["twists"][span]).max()), float(np.abs(aft["twists"][span]).max())
            row = {"a": a, "b": b, "wraps_before": int(d["wrapped"][span].sum()), "wraps_after": int(aft["wrapped"][span].sum()),
                   "candy_before": int(d["candy"][span].sum()), "candy_after": int(aft["candy"][span].sum()),
                   "max_twist_before_deg": round(mb, 1), "max_twist_after_deg": round(ma, 1),
                   "max_abs_target_deg": round(float(np.abs(target[span]).max()), 1),
                   "not_pure_frames": int((m & ~d["guard_ok"]).sum())}
            # a re-expression must clear the wraps and must not make a twist worse; an event whose total twist really passes
            # +-180 cannot be cleared this way (a facing / yaw error: the yaw and solver windows own it)
            ok = row["wraps_after"] == 0 and ma <= mb + 1e-6 and row["candy_after"] <= row["candy_before"]
            if ok:
                cur = trial
                changed |= span
                crep["runs"].append(row)
            else:
                crep["refused"].append(row)
        after = chain_wraps(pose=cur, rest=rest, joints=joints, cfg=cfg, chains=(name,))[name]
        crep.update({"core_frames": int(sum(r["b"] - r["a"] + 1 for r in crep["runs"])),
                     "not_pure_frames": int(sum(r["not_pure_frames"] for r in crep["runs"])),
                     "wrapped_frames_after": int((after["wrapped"] & allow).sum()),
                     "candy_frames_after": int(((after["wrapped"] | after["candy"]) & allow).sum())})
        rep["chains"][name] = crep
    spine = chain_wraps(pose=cur, rest=rest, joints=joints, cfg=cfg, chains=("spine",))["spine"]
    sp_core = spine["over"] & ~spine["wrapped"] & allow
    if split and sp_core.any():
        transl = np.asarray(fit["transl"], float) if "transl" in fit else None
        cur2, srep = split_spine(cur, rest, transl, sp_core, cfg)
        changed |= np.any(np.abs(cur2 - cur) > 0, axis=(1, 2))
        cur = cur2
        rep["split"] = srep
    new = dict(fit)
    new["global_orient"] = cur[:, 0].copy()
    new["body_pose"] = cur[:, 1:22].reshape(T, 63).copy()
    idx = np.flatnonzero(changed)
    newJ = joints.copy()
    if len(idx):
        if fk is None:
            from .. import stitch
            z = {k: np.asarray(new[k], float)[idx] for k in (*stitch.POSE_KEYS, "transl")}
            J55 = stitch.fk_joints(stitch.pose165(z), np.asarray(fit["transl"], float)[idx], np.asarray(fit["betas"], float))
        else:
            J55 = fk(new, idx)
        newJ[idx, :J55.shape[1]] = J55
        rep["max_joint_move_m"] = round(float(np.linalg.norm(newJ[idx, :22] - joints[idx, :22], axis=-1).max()), 5)
    else:
        rep["max_joint_move_m"] = 0.0
    new["joints"] = newJ
    rep["changed_frames"] = int(changed.sum())
    return new, rep
