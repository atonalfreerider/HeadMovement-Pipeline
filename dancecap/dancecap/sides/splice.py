"""Spin splice (DESIGN 2.4 "Splice", 4.1 `splice.py`): where the stitched fit winds a spin the wrong way and an overlapping chunk
solution of the same frames gets it right, hand those frames to that chunk, for both dancers (they were solved together).

* `spin_defects`: |psi_base - psi*| > 45 deg for >= 3 frames, or >= 2 views with contradicted face bits.
* `score_alternatives`: every candidate (the base and each overlapping chunk) per frame by its yaw log-likelihood (any sum of
  evidence terms, e.g. `von_mises_ll`, `face_bits_ll`) minus a weighted 2D cost (MAMMA 512 / COCO px).
* `choose_exit_all_segments` (the F1 rule): the first frame from which, for >= 0.3 s and for every dancer, all body joints agree
  (p95 <= 60 mm) and the pelvis, spine3 and both feet agree mod 360 within 15 deg - not the pelvis agreement alone, which comes
  earlier with a foot still wound.
* `splice`: entry = an offset cosine ramp over >= 7 frames (the alternative is turned about its pelvis vertical, and every local
  rotation and the root position offset, onto the base at the entry frame, the offset easing to 0 with zero slope at both ends);
  exit = a >= 6-frame slerp to the base on the nearest rotation branch (`stitch.blend_rotvec`).

Grid convention: base arrays are on the take grid (frame k = index); an alternative's local frame i is grid frame i + offset (the
stitch.json `frame_offset`). No take constants. Headings / runs come from sides.geom / sides.hmm; the thresholds from
`SidesCfg.splice_cfg()`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import geom, hmm

POSE_KEYS = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose")
FIT_KEYS = ("repaired", "suspect", "reproj_px_raw", "contact", "floor_contact", "quality", "quality_parts", "reproj_parts_px",
            "reproj_px", "stance")       # per-frame values that describe the fit itself: taken from the source that dominates
SRC_BASE, SRC_ENTRY, SRC_ALT, SRC_EXIT = 0, 1, 2, 3


@dataclass(frozen=True)
class SpliceCfg:
    """DESIGN 2.4 values ([I] where marked). The pipeline builds it from SidesCfg (`SidesCfg.splice_cfg()`); the defaults equal
    SidesCfg's."""
    hz: float = 30.0
    defect_deg: float = 45.0
    defect_min_frames: int = 3
    defect_min_views: int = 2
    defect_merge_gap: int = 2            # [I] frames
    entry_frames: int = 7                # >= 7
    exit_p95_m: float = 0.060
    exit_seg_deg: float = 15.0
    exit_hold_s: float = 0.3
    exit_slerp_frames: int = 6           # >= 6
    exit_relative: bool = True           # spine3 / feet compared as (segment - pelvis), the pelvis absolutely (F1, f1_exit.py)
    foot_min_h_m: float = 0.03           # [I] a foot vector shorter than this horizontally has no heading (counts as agreeing)


# ------------------------------------------------------------------------------------------------ small helpers (sides.geom / hmm)

_wrap = geom.wrap_deg


def _heading(v: np.ndarray, min_h: float = 1e-9) -> np.ndarray:
    return geom.heading(v, min_h=min_h)


_line_heading = geom.line_heading


def _runs(mask: np.ndarray) -> list:
    return hmm.runs(mask)


def entry_ramp(n: int) -> np.ndarray:
    """offset strength over the entry: 1 at the entry frame (the output equals the base there) down to 0 at frame n-1 (the
    alternative itself), a cosine with zero slope at both ends (C1)"""
    if n < 2:
        return np.zeros(max(n, 0))
    return 0.5 * (1.0 + np.cos(np.pi * np.arange(n) / (n - 1)))


def exit_weights(n: int) -> np.ndarray:
    """weight of the base over the exit slerp: strictly inside (0, 1), cosine (= stitch.cosine_weights)"""
    x = (np.arange(n) + 0.5) / max(n, 1)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def segment_headings(J: np.ndarray, foot_min_h: float = 0.03) -> dict:
    """world headings (deg, wrapped) of the segments the exit rule compares, from SMPL-X joints (T, >=22, 3): pelvis (hip line),
    spine3 (shoulder line), each foot (ankle -> foot joint, NaN when it points down)"""
    J = np.asarray(J, float)
    fl, fr = J[:, 10] - J[:, 7], J[:, 11] - J[:, 8]
    return {"pelvis": _line_heading(J[:, 1], J[:, 2]), "spine3": _line_heading(J[:, 16], J[:, 17]),
            "foot_l": _heading(fl, foot_min_h), "foot_r": _heading(fr, foot_min_h)}


def foot_minus_pelvis(J: np.ndarray, side: str = "l", foot_min_h: float = 0.03) -> np.ndarray:
    """the scorecard's foot-minus-pelvis yaw (deg): ankle->foot heading minus the hip-line heading"""
    s = segment_headings(J, foot_min_h)
    return _wrap(s["foot_" + side] - s["pelvis"])


# ------------------------------------------------------------------------------------------------ defects and scores

def spin_defects(psi_base: np.ndarray, psi_star: np.ndarray, face_contra: np.ndarray | None = None,
                 cfg: SpliceCfg = SpliceCfg()) -> dict:
    """DESIGN 2.4 spin defect per frame: |wrap(psi_base - psi*)| > defect_deg on >= defect_min_frames consecutive frames (psi* NaN =
    no evidence), or face_contra (T,) >= defect_min_views (views whose face bit contradicts the base's head). COCO-3D hip lines alone
    must never be passed as psi* (they flip on back views): psi* is the yaw HMM's output. -> {mask, runs [(a, b)], err}"""
    e = np.abs(_wrap(np.asarray(psi_base, float) - np.asarray(psi_star, float)))
    big = np.nan_to_num(e, nan=0.0) > cfg.defect_deg
    m = np.zeros(len(e), bool)
    for a, b in _runs(big):
        if b - a + 1 >= cfg.defect_min_frames:
            m[a:b + 1] = True
    if face_contra is not None:
        m |= np.asarray(face_contra) >= cfg.defect_min_views
    # merge short gaps
    rr = _runs(m)
    out = []
    for a, b in rr:
        if out and a - out[-1][1] - 1 <= cfg.defect_merge_gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    mask = np.zeros(len(e), bool)
    for a, b in out:
        mask[a:b + 1] = True
    return {"mask": mask, "runs": out, "err": e}


def von_mises_ll(psi_star: np.ndarray, kappa: np.ndarray | float) -> Callable:
    """log-likelihood term of a heading track under a von Mises around psi* (NaN = no evidence -> 0); kappa per frame or scalar
    (kappa = 1 / sigma_rad^2 for a sigma of a few tens of degrees)"""
    ps = np.asarray(psi_star, float)
    k = np.broadcast_to(np.asarray(kappa, float), ps.shape)

    def ll(h):
        v = k * (np.cos(np.radians(np.asarray(h, float) - ps)) - 1.0)
        return np.where(np.isfinite(v), v, 0.0)
    return ll


def face_bits_ll(labels: np.ndarray, azimuth: np.ndarray, q: np.ndarray | float = 0.9) -> Callable:
    """log-likelihood of per-view face bits for a HEAD heading track: labels (T, C) +1 face seen / -1 hidden / 0 no bit, azimuth
    (T, C) the heading from the head toward camera c; a bit is the half-plane |wrap(azimuth - h)| < 90 (seen) with reliability q"""
    lab = np.asarray(labels)
    az = np.asarray(azimuth, float)
    qq = np.broadcast_to(np.asarray(q, float), (lab.shape[1],))[None]

    def ll(h):
        facing = np.abs(_wrap(az - np.asarray(h, float)[:, None])) < 90.0
        pred = np.where(facing, 1, -1)
        ok = np.isfinite(az) & (lab != 0)
        agree = pred == lab
        v = np.where(agree, np.log(qq), np.log(1.0 - qq))
        return np.where(ok, v, 0.0).sum(1)
    return ll


def face_contradictions(labels: np.ndarray, azimuth: np.ndarray, head_heading: np.ndarray, seen_max: float = 110.0,
                        hidden_min: float = 70.0) -> np.ndarray:
    """(T,) number of views whose face bit contradicts a head heading track (spins.md s6: seen but turned > 110 deg away, or hidden
    while turned < 70 deg toward the camera)"""
    ang = np.abs(_wrap(np.asarray(azimuth, float) - np.asarray(head_heading, float)[:, None]))
    lab = np.asarray(labels)
    v = ((lab == 1) & (ang > seen_max)) | ((lab == -1) & (ang < hidden_min))
    return (v & np.isfinite(ang)).sum(1)


def score_alternatives(cands: dict, terms: list, cost2d: dict | None = None, w2d: float = 0.0, runs: list | None = None) -> dict:
    """cands {name: {signal: heading (T,) deg, NaN where the candidate has no frame}}, terms [(signal, ll_fn, weight)], cost2d
    {name: (T,) px} (optional, weighted by w2d nats per px). -> {names, score (n, T) (NaN where a candidate is absent), best (T,)
    index (-1 where none), runs: [{a, b, totals {name: sum over the frames every candidate covers}, best}]}"""
    names = list(cands)
    T = len(next(iter(next(iter(cands.values())).values())))
    S = np.full((len(names), T), np.nan)
    for i, n in enumerate(names):
        tot = np.zeros(T)
        present = np.ones(T, bool)
        for sig, fn, wt in terms:
            h = np.asarray(cands[n][sig], float)
            present &= np.isfinite(h)
            tot += wt * fn(np.nan_to_num(h))
        if cost2d is not None and n in cost2d:
            c = np.asarray(cost2d[n], float)
            tot -= w2d * np.nan_to_num(c)
        S[i] = np.where(present, tot, np.nan)
    best = np.where(np.isfinite(S).any(0), np.nanargmax(np.where(np.isfinite(S), S, -np.inf), 0), -1)
    out = {"names": names, "score": S, "best": best, "runs": []}
    for a, b in (runs or []):
        sl = S[:, a:b + 1]
        both = np.isfinite(sl).all(0)
        tot = {n: float(sl[i, both].sum()) for i, n in enumerate(names)}
        srt = sorted(tot.values(), reverse=True)
        out["runs"].append({"a": a, "b": b, "frames_compared": int(both.sum()), "totals": tot,
                            "best": max(tot, key=tot.get) if both.any() else None,
                            "margin": float(srt[0] - srt[1]) if both.any() and len(srt) > 1 else None})
    return out


# ------------------------------------------------------------------------------------------------ exit rule (F1)

def agreement(base_J: np.ndarray, alt_J: np.ndarray, cfg: SpliceCfg = SpliceCfg()) -> dict:
    """per frame: p95 (m) of the body-joint distance and the |wrapped| heading difference of each exit segment between two joint
    tracks on the same frames (NaN rows = absent). -> {p95, seg {name: deg}, ok (T,)}"""
    A, B = np.asarray(base_J, float), np.asarray(alt_J, float)
    d = np.linalg.norm(A[:, :22] - B[:, :22], axis=-1)
    p95 = np.percentile(d, 95, axis=1)
    sa, sb = segment_headings(A, cfg.foot_min_h_m), segment_headings(B, cfg.foot_min_h_m)
    if cfg.exit_relative:      # the pelvis in the world, every other segment's winding against its own pelvis (as F1 measured it)
        seg = {k: np.abs(_wrap(sa[k] - sb[k])) if k == "pelvis" else
               np.abs(_wrap((sa[k] - sa["pelvis"]) - (sb[k] - sb["pelvis"]))) for k in sa}
    else:
        seg = {k: np.abs(_wrap(sa[k] - sb[k])) for k in sa}
    ok = np.isfinite(p95) & (p95 <= cfg.exit_p95_m)
    for k, v in seg.items():
        both_nan = ~np.isfinite(sa[k]) & ~np.isfinite(sb[k])
        ok &= (np.nan_to_num(v, nan=999.0) <= cfg.exit_seg_deg) | both_nan
    return {"p95": p95, "seg": seg, "ok": ok}


def choose_exit_all_segments(base_J: dict, alt_J: dict, start: int, cfg: SpliceCfg = SpliceCfg(), stop: int | None = None) -> dict:
    """The F1 exit: the first frame k >= start such that frames k .. k + hold - 1 agree for EVERY dancer (all body joints p95 <=
    exit_p95_m, pelvis / spine3 / both feet within exit_seg_deg mod 360), and the exit slerp k .. k + exit_slerp_frames - 1 lies
    inside the alternative. base_J / alt_J {role: (T, >=22, 3)} on the same frames (alt NaN where absent); stop = one past the last
    frame the alternative covers (default: its last finite frame + 1). -> {k_exit (None if never), hold, per_role {p95, seg, ok},
    first_pelvis_only (the pelvis-only rule's answer, for the report)}"""
    hold = int(round(cfg.exit_hold_s * cfg.hz))
    roles = list(base_J)
    per = {r: agreement(base_J[r], alt_J[r], cfg) for r in roles}
    ok = np.logical_and.reduce([per[r]["ok"] for r in roles])
    if stop is None:
        fin = np.flatnonzero(np.isfinite(np.asarray(alt_J[roles[0]], float)[:, 0, 0]))
        stop = int(fin[-1]) + 1 if len(fin) else 0
    last = stop - max(hold, cfg.exit_slerp_frames)
    k_exit = None
    for k in range(max(start, 0), last + 1):
        if ok[k:k + hold].all():
            k_exit = k
            break
    # what a pelvis-heading-only exit would have picked (DESIGN: all three designs exited there, with a foot still wound)
    pel = np.logical_and.reduce([np.nan_to_num(per[r]["seg"]["pelvis"], nan=999.0) <= cfg.exit_seg_deg for r in roles])
    k_pel = None
    for k in range(max(start, 0), last + 1):
        if pel[k:k + hold].all():
            k_pel = k
            break
    return {"k_exit": k_exit, "hold": hold, "per_role": per, "ok": ok, "first_pelvis_only": k_pel}


# ------------------------------------------------------------------------------------------------ the splice

def _mats(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return Rot.from_rotvec(v.reshape(-1, 3)).as_matrix().reshape(v.shape[:-1] + (3, 3))


def _vecs(R: np.ndarray) -> np.ndarray:
    return Rot.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def _nearest_rep(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    from ..stitch import nearest_rep
    return nearest_rep(v, ref)


def _blend_rotvec(a, b, w):
    from ..stitch import blend_rotvec
    return blend_rotvec(a, b, w)


def _default_fk(pose165: np.ndarray, transl: np.ndarray, betas: np.ndarray) -> np.ndarray:
    from ..stitch import fk_joints
    return fk_joints(pose165, transl, betas)


def _pose165(d: dict, idx: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(d[k], float)[idx].reshape(len(idx), -1) for k in POSE_KEYS], 1)


def yaw_of(R: np.ndarray) -> np.ndarray:
    """heading (deg) of a rotation's forward (+z) axis"""
    return _heading(np.asarray(R, float)[..., :, 2])


def splice(base: dict, alt: dict, offset: int, k_entry: int, k_exit: int, cfg: SpliceCfg = SpliceCfg(), fk: Callable | None = None,
           alt_name: str = "alt") -> tuple[dict, dict]:
    """Splice one alternative into the base for every role in `base`. base {role: arrays on the grid}, alt {role: arrays of the
    chunk}, offset = grid frame of the chunk's local frame 0. Frames: [k_entry, k_entry + entry_frames) entry ramp (the offset
    eases out), [.., k_exit) the alternative, [k_exit, k_exit + exit_slerp_frames) the slerp to the base, the base after.
    fk(pose165, transl, betas) -> joints (n, 55, 3) recomputes the joints of ramp frames (default stitch.fk_joints).
    -> ({role: full arrays (copies)}, info {k0, k1, src (n,), per role offsets})"""
    fk = fk or _default_fk
    ne, nx = cfg.entry_frames, cfg.exit_slerp_frames
    k1 = k_exit + nx                                    # one past the last frame that differs from the base
    n_alt = len(next(iter(alt.values()))["transl"])
    if k_entry < offset or k1 > offset + n_alt:
        raise ValueError(f"splice frames {k_entry}..{k1} not inside the alternative ({offset}..{offset + n_alt})")
    if k_exit < k_entry + ne:
        raise ValueError(f"exit {k_exit} inside the entry ramp ({k_entry}..{k_entry + ne})")
    ks = np.arange(k_entry, k1)
    n = len(ks)
    src = np.full(n, SRC_ALT, np.int8)
    src[:ne] = SRC_ENTRY
    src[n - nx:] = SRC_EXIT
    a_ent = np.zeros(n)
    a_ent[:ne] = entry_ramp(ne)
    w_base = np.zeros(n)
    w_base[n - nx:] = exit_weights(nx)
    w_alt = 1.0 - np.maximum(a_ent, w_base)          # informational: how much of the frame is the alternative
    outs, info = {}, {"k0": int(k_entry), "k1": int(k1), "k_exit": int(k_exit), "src": src, "w_alt": w_alt, "roles": {}}
    for role, B in base.items():
        A = alt[role]
        ia = ks - offset
        out = {k: np.array(v, copy=True) for k, v in B.items()}
        T = len(B["transl"])
        e = 0                                            # index of the entry frame in ks
        # --- per-joint offsets at the entry frame (left-multiplied: global orient in the world, the rest in their parent frames)
        rinfo = {}
        for key in POSE_KEYS:
            if key not in B or key not in A:
                continue
            nb = np.asarray(B[key], float).reshape(T, -1, 3)
            na = np.asarray(A[key], float).reshape(len(A[key]), -1, 3)
            Rb_e, Ra = _mats(nb[k_entry]), _mats(na[ia])            # (J,3,3), (n,J,3,3)
            delta = Rb_e @ np.swapaxes(Ra[e], -1, -2)               # R_base(e) R_alt(e)^T per joint
            dv = _vecs(delta)                                       # (J,3), angle <= pi (the short way)
            Rout = Ra.copy()
            for i in range(ne):
                Rout[i] = _mats(dv * a_ent[i]) @ Ra[i]
            if key == "global_orient":
                rinfo["entry_offset_deg"] = round(float(np.degrees(np.linalg.norm(dv[0]))), 2)
                rinfo["entry_offset_yaw_deg"] = round(float(_wrap(yaw_of(Rb_e[0]) - yaw_of(Ra[e, 0]))), 2)
            else:
                ang = np.degrees(np.linalg.norm(dv, axis=-1))
                rinfo.setdefault("entry_offset_local_max_deg", {})[key] = round(float(ang.max()), 2)
            vout = _vecs(Rout)
            # exit slerp to the base (nearest branch, shortest arc)
            vb = nb[ks[n - nx:]]
            vout[n - nx:] = _blend_rotvec(vout[n - nx:], vb, w_base[n - nx:])
            # parameter continuity: each frame on the branch nearest the previous output frame
            prev = nb[k_entry - 1] if k_entry > 0 else vout[0]
            for i in range(n):
                vout[i] = _nearest_rep(vout[i], prev)
                prev = vout[i]
            out[key] = np.asarray(out[key]).copy()
            out[key][ks] = vout.reshape((n,) + np.asarray(B[key]).shape[1:]).astype(np.asarray(B[key]).dtype)
        # --- root position: the alternative's, plus the entry offset easing out; linear blend to the base at the exit
        tb, ta = np.asarray(B["transl"], float), np.asarray(A["transl"], float)
        dtr = tb[k_entry] - ta[ia[e]]
        tr = ta[ia] + a_ent[:, None] * dtr[None]
        tr[n - nx:] = (1 - w_base[n - nx:, None]) * tr[n - nx:] + w_base[n - nx:, None] * tb[ks[n - nx:]]
        out["transl"] = np.asarray(out["transl"]).copy()
        out["transl"][ks] = tr.astype(np.asarray(B["transl"]).dtype)
        rinfo["entry_offset_transl_m"] = [round(float(x), 4) for x in dtr]
        # --- joints: the alternative's own on its pure frames, FK on the ramps
        J = np.asarray(out["joints"], float).copy()
        J[ks[src == SRC_ALT]] = np.asarray(A["joints"], float)[ia[src == SRC_ALT]]
        rk = ks[src != SRC_ALT]
        J[rk, :55] = fk(_pose165(out, rk), np.asarray(out["transl"], float)[rk], np.asarray(B["betas"], float))
        out["joints"] = J.astype(np.asarray(B["joints"]).dtype)
        # --- per-frame fit descriptors from the dominant source; provenance
        dom_alt = w_alt >= 0.5
        for key in FIT_KEYS:
            if key in B and key in A and np.asarray(B[key]).shape[1:] == np.asarray(A[key]).shape[1:] and len(B[key]) == T:
                arr = np.array(out[key], copy=True)
                arr[ks[dom_alt]] = np.asarray(A[key])[ia[dom_alt]]
                out[key] = arr
        if "segment_owner" in B:
            so = np.array(B["segment_owner"], dtype=f"<U{max(len(alt_name), np.asarray(B['segment_owner']).dtype.itemsize // 4)}")
            so[ks[dom_alt]] = alt_name
            out["segment_owner"] = so
        outs[role] = out
        info["roles"][role] = rinfo
    return outs, info


def splice_window_arrays(spliced: dict, info: dict) -> dict:
    """the frames a splice changed, flattened for an npz: k, src, w_alt and per role <role>_<key> for the pose keys, transl, joints,
    stance, betas"""
    k0, k1 = info["k0"], info["k1"]
    ks = np.arange(k0, k1)
    z = {"k": ks, "src": info["src"], "w_alt": info["w_alt"]}
    for role, d in spliced.items():
        for key in (*POSE_KEYS, "transl", "joints", "stance"):
            if key in d:
                z[f"{role}_{key}"] = np.asarray(d[key])[ks]
        z[f"{role}_betas"] = np.asarray(d["betas"])
        if "t" in d:
            z["t"] = np.asarray(d["t"])[ks]
    return z


def apply_window(base: dict, z: dict, role: str) -> dict:
    """a copy of one role's base arrays with a saved splice window (`splice_window_arrays` / spin_*.npz) written in: the pose keys,
    transl, joints and stance of frames z["k"]; segment_owner = z["alt_name"] where the alternative dominates (w_alt >= 0.5)"""
    ks = np.asarray(z["k"], int)
    out = {k: np.array(v, copy=True) for k, v in base.items()}
    for key in (*POSE_KEYS, "transl", "joints", "stance"):
        zk = f"{role}_{key}"
        if zk in z and key in out:
            out[key][ks] = np.asarray(z[zk]).astype(out[key].dtype)
    if "segment_owner" in out and "alt_name" in z:
        name = str(np.asarray(z["alt_name"]))
        so = np.array(out["segment_owner"], dtype=f"<U{max(len(name), out['segment_owner'].dtype.itemsize // 4)}")
        so[ks[np.asarray(z["w_alt"]) >= 0.5]] = name
        out["segment_owner"] = so
    return out
