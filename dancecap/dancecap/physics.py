"""Physics (L0 centroidal dynamics) of the two dancers from the fused SMPL-X fits.

Inputs (work/<take>/): smplx/{lead,follow}.npz (contract: SMPL-X params, Y-up world metres, floor y=0), an
optional beats json (zouk format, as timing), optional dancer masses / statures. Outputs: physics/physics.npz
(full arrays), physics/physics.json (compact per-frame record for Unity), physics/qa_physics.png.

What is reported, and why only this (research verdict): partner tension / compression is NOT identifiable
from video kinematics in general - with both partners on the floor the contact problem is statically
indeterminate (130-270 N of freedom) and kinematic acceleration noise (30-150 N) exceeds the expected signal
(10-30 N). The honest deliverable is the centroidal level:

  * body model (physics_body): SMPL-X shape -> 55 rigid mass elements (voxelised closed mesh, LBS-argmax
    labels, uniform density scaled to the total mass); mass = user input, else volume x 1010 kg/m^3 (+-10%).
  * per dancer per frame: COM, vCOM, aCOM from a quintic smoothing spline on the real timestamps (cutoff
    5 Hz, physics_spline); F_net = m (a_com - g) = all non-gravitational external force (floor + partner),
    with a 95 % band (white-noise propagation of the 4th-difference noise estimate, a moving-block residual
    bootstrap, and the 4-6 Hz cutoff sensitivity); whole-body angular momentum L_G and dL/dt (segment COM
    velocities + segment spins), the vertical component (turns); foot contacts and the support polygon from
    the posed soles; extrapolated COM (Hof 2005) and the balance margin (signed distance XCoM -> polygon).
  * couple GRF = F_net(lead) + F_net(follow): partner forces cancel (Newton 3), so this IS identifiable.
  * beat-window impulses  m dv - m g dt  per beat interval (velocity at the beat boundaries only, so
    per-frame acceleration noise integrates out).
  * partner force: an interval from a linear program over every contact force distribution consistent with
    both dancers' F_net and dL/dt (bands as tolerances), unilateral friction-pyramid foot contacts at the
    support polygon corners, free 3D forces at the active hand/torso connections. It is labelled
    'identifiable' only when a dancer is in flight or single support AND the interval is narrower than
    cfg.identifiable_width_n; everything else is 'not identifiable'. head-movement's PartnerConnection
    spring-damper signal stays a labelled heuristic for visualisation, not a force estimate.

    python -m dancecap physics synthetic --validate     -> work/synthetic/physics/ + validation.json (gate G6)
    python -m dancecap physics larissa_kadu --mass-lead 78 --mass-follow 56
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import physics_body as pb
from .physics_spline import Smoother
from .take import PACKAGE_ROOT

ROLES = ("lead", "follow")
G = pb.G
G_VEC = np.array([0.0, -G, 0.0])
Z95 = 1.96
FLAG_BITS = {"edge": 1, "flight": 2, "single_support": 4, "double_support": 8, "xcom_outside": 16,
             "low_quality": 32, "partner_identifiable": 64, "partner_no_contact": 128,
             "partner_inconsistent": 256}
PARTNER_STATUS = {0: "not_identifiable", 1: "identifiable", 2: "no_contact", 3: "inconsistent"}
# candidate partner connections (lead points x follow points, contact distance m), as PartnerConnection.cs
LEAD_POINTS = {"lead_l_hand": "hand_l", "lead_r_hand": "hand_r", "lead_chest": 9}
FOLLOW_POINTS = {"follow_l_hand": "hand_l", "follow_r_hand": "hand_r", "follow_spine3": 9, "follow_neck": 12,
                 "follow_l_shoulder": 16, "follow_r_shoulder": 17}
CONTACT_DIST = {"hand": 0.15, "lead_chest": 0.40}


@dataclass
class PhysicsConfig:
    cutoff_hz: float = 5.0
    cutoff_sens: tuple[float, float] = (4.0, 6.0)
    n_boot: int = 200
    block_s: float = 0.2            # moving-block bootstrap block length
    edge_s: float = 0.25            # spline boundary region flagged at both ends
    contact_h: float = 0.03         # sole height above the floor estimate counted as contact (m)
    contact_v: float = 0.5          # max horizontal sole speed in contact (m/s)
    contact_pts_h: float = 0.015    # sole vertices within this of the lowest contact point span the polygon
    mu: float = 0.6                 # floor friction (circumscribed pyramid -> conservative intervals)
    torsion_r: float = 0.05         # free vertical moment <= r * normal force
    partner_cap_n: float = 400.0    # box bound per partner force component (LP boundedness)
    identifiable_width_n: float = 40.0
    tol_min_n: float = 5.0          # floors for the LP equality tolerances (model error)
    tol_min_nm: float = 3.0
    voxel_m: float = 0.007
    low_quality: float = 0.3
    seed: int = 0


# ----------------------------------------------------------------------------------------------- helpers

def _vee(W: np.ndarray) -> np.ndarray:
    return 0.5 * np.stack([W[..., 2, 1] - W[..., 1, 2], W[..., 0, 2] - W[..., 2, 0], W[..., 1, 0] - W[..., 0, 1]], -1)


def estimate_noise(x: np.ndarray) -> float:
    """Robust white-noise SD of a smooth series (T,): MAD of 4th differences / sqrt(70) (as timing)."""
    d = np.diff(x, 4)
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / math.sqrt(70.0))


def block_bootstrap(op: np.ndarray, resid: np.ndarray, n: int, block: int, rng) -> np.ndarray:
    """SD over moving-block resamples of the residuals (N, C) pushed through the linear operator op (M, N)."""
    N = len(resid)
    nb = int(math.ceil(N / block))
    out = np.empty((n, op.shape[0], resid.shape[1]))
    for b in range(n):
        starts = rng.integers(0, N - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:N]
        r = resid[idx]
        out[b] = op @ (r - r.mean(0))
    return out.std(0)


def signed_distance_polygon(pt: np.ndarray, poly: np.ndarray) -> float:
    """Signed distance (m) of a 2D point to a convex polygon (CCW or CW), positive inside; NaN if no polygon."""
    poly = poly[np.isfinite(poly[:, 0])]
    if len(poly) == 0:
        return float("nan")
    if len(poly) == 1:
        return -float(np.linalg.norm(pt - poly[0]))
    a, b = poly, np.roll(poly, -1, 0)
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    s = np.clip(((pt - a) * ab).sum(1) / L2, 0, 1)
    d = np.linalg.norm(a + s[:, None] * ab - pt, axis=1).min()
    if len(poly) < 3:
        return -float(d)
    cross = ab[:, 0] * (pt[1] - a[:, 1]) - ab[:, 1] * (pt[0] - a[:, 0])
    inside = np.all(cross >= 0) or np.all(cross <= 0)
    return float(d if inside else -d)


def hull2d(pts: np.ndarray, max_pts: int = 16) -> np.ndarray:
    """Convex hull (CCW) of 2D points, at most max_pts vertices (farthest-point subsampling of the hull)."""
    from scipy.spatial import ConvexHull, QhullError
    if len(pts) < 3:
        return pts
    try:
        h = pts[ConvexHull(pts).vertices]
    except QhullError:
        d = pts - pts.mean(0)
        ax = np.linalg.svd(d, full_matrices=False)[2][0]
        pr = d @ ax
        return pts[[pr.argmin(), pr.argmax()]]
    if len(h) > max_pts:
        keep = [0]
        dist = np.linalg.norm(h - h[0], axis=1)
        for _ in range(max_pts - 1):
            keep.append(int(dist.argmax()))
            dist = np.minimum(dist, np.linalg.norm(h - h[keep[-1]], axis=1))
        h = h[np.sort(keep)]
    return h


# ----------------------------------------------------------------------------------------------- one dancer

def analyze_dancer(npz: dict, cfg: PhysicsConfig, mass: float | None = None, stature: float | None = None,
                   rng=None) -> dict:
    """Centroidal quantities of one dancer (all arrays on npz['t'])."""
    rng = rng or np.random.default_rng(cfg.seed)
    t = np.asarray(npz["t"], float)
    T = len(t)
    body = pb.segment_masses(pb.load_body(npz), mass, voxel=cfg.voxel_m)
    M = body.mass
    R, p = pb.forward_kinematics(body, pb.full_pose(npz), np.asarray(npz["transl"], float))
    fk_err = None
    if "joints" in npz:
        J = np.einsum("tjab,jb->tja", R, body.j_rest) + p
        fk_err = float(np.abs(J - np.asarray(npz["joints"], float)[:, :55]).max())
    c, I = pb.element_world(body, R, p)                     # (T,55,3), (T,55,3,3)
    m = body.elem_mass
    com_raw = (m[None, :, None] * c).sum(1) / M
    qual = np.asarray(npz["quality"], float) if "quality" in npz else np.ones(T)

    S = Smoother(t, cfg.cutoff_hz)
    com, vcom, acom = S(com_raw, 0), S(com_raw, 1), S(com_raw, 2)
    f_net = M * (acom - G_VEC)

    # noise band: white-noise propagation, moving-block residual bootstrap, cutoff sensitivity
    resid = com_raw - com
    sd_pos = np.array([estimate_noise(com_raw[:, k]) for k in range(3)])
    sd_white = sd_pos[None] * S.noise_gain(2)[:, None]
    dt = float(np.median(np.diff(t)))
    block = max(2, int(round(cfg.block_s / dt)))
    sd_boot = block_bootstrap(S.operator(2), resid, cfg.n_boot, block, rng)
    a_lo = Smoother(t, cfg.cutoff_sens[0])(com_raw, 2)
    a_hi = Smoother(t, cfg.cutoff_sens[1])(com_raw, 2)
    sens = 0.5 * np.abs(a_hi - a_lo)
    a_band = np.sqrt((Z95 * np.maximum(sd_white, sd_boot)) ** 2 + sens ** 2)

    # angular momentum about the COM: element translation + element spin
    cs = S(c.reshape(T, -1), 0).reshape(T, 55, 3)
    cv = S(c.reshape(T, -1), 1).reshape(T, 55, 3)
    Rd = S(R.reshape(T, -1), 1).reshape(T, 55, 3, 3)
    omega = _vee(Rd @ np.swapaxes(R, -1, -2))
    L = (m[None, :, None] * np.cross(cs - com[:, None], cv - vcom[:, None])).sum(1) \
        + np.einsum("tjab,tjb->ta", I, omega)
    dL = S(L, 1)
    rel = c - com_raw[:, None]
    I_com = I.sum(1) + (m[None, :, None, None] * ((rel ** 2).sum(-1)[..., None, None] * np.eye(3)
                                                     - rel[..., :, None] * rel[..., None, :])).sum(1)
    spin_rate = L[:, 1] / np.maximum(I_com[:, 1, 1], 1e-6)  # rad/s about the vertical (+Y, CCW seen from above)
    # dL/dt band: bootstrap of the element-position residuals through the translational term
    resid_c = (c - cs).reshape(T, -1)
    a_c_noise = block_bootstrap(S.operator(2), resid_c, max(30, cfg.n_boot // 4), block, rng).reshape(T, 55, 3)
    dL_band = Z95 * np.sqrt(((m[None, :, None] * np.linalg.norm(rel, axis=-1, keepdims=True) * a_c_noise) ** 2)
                            .sum(1))

    # contacts and support polygon from the posed soles
    soles = {f: body.sole_vertices(f) for f in ("left", "right")}
    sole_pos = {f: pb.lbs_points(body, R, p, idx) for f, idx in soles.items()}   # (T, n, 3)
    # lowest sole point per foot, spline-smoothed so per-frame pose noise does not fragment the contacts
    low = S(np.stack([sole_pos[f][:, :, 1].min(1) for f in ("left", "right")], 1))   # (T, 2)
    floor_y = float(np.percentile(low.min(1), 2))
    contact = np.zeros((T, 2), bool)
    for k, f in enumerate(("left", "right")):
        cen = sole_pos[f].mean(1)
        sp = np.linalg.norm(S(cen, 1)[:, [0, 2]], axis=1)
        contact[:, k] = (low[:, k] - floor_y < cfg.contact_h) & (sp < cfg.contact_v)
    from scipy.ndimage import median_filter
    contact = median_filter(contact.astype(np.uint8), size=(3, 1), mode="nearest").astype(bool)
    support = np.full((T, 16, 2), np.nan)
    corners = []   # per frame list of (x,y,z) foot contact points for the partner LP
    for i in range(T):
        pts2, pts3 = [], []
        for k, f in enumerate(("left", "right")):
            if not contact[i, k]:
                continue
            sp_ = sole_pos[f][i]
            sel = sp_[sp_[:, 1] < sp_[:, 1].min() + cfg.contact_pts_h][:, [0, 2]]  # the loaded part of the sole
            pts2.append(sel)
            h = hull2d(sel, 4)  # <= 4 corners per foot for the contact-force LP
            pts3.append(np.stack([h[:, 0], np.full(len(h), floor_y), h[:, 1]], 1))
        corners.append(np.concatenate(pts3) if pts3 else np.zeros((0, 3)))
        if pts2:
            h = hull2d(np.concatenate(pts2), 16)
            support[i, :len(h)] = h
    leg = pb.rest_com_height(body)
    w0 = math.sqrt(G / leg)
    xcom = com[:, [0, 2]] + vcom[:, [0, 2]] / w0
    margin_x = np.array([signed_distance_polygon(xcom[i], support[i]) for i in range(T)])
    margin_c = np.array([signed_distance_polygon(com[i, [0, 2]], support[i]) for i in range(T)])

    flags = np.zeros(T, np.int64)
    edge = (t < t[0] + cfg.edge_s) | (t > t[-1] - cfg.edge_s)
    nc = contact.sum(1)
    flags |= np.where(edge, FLAG_BITS["edge"], 0)
    flags |= np.where(nc == 0, FLAG_BITS["flight"], 0)
    flags |= np.where(nc == 1, FLAG_BITS["single_support"], 0)
    flags |= np.where(nc == 2, FLAG_BITS["double_support"], 0)
    flags |= np.where(np.nan_to_num(margin_x, nan=1.0) < 0, FLAG_BITS["xcom_outside"], 0)
    flags |= np.where(qual < cfg.low_quality, FLAG_BITS["low_quality"], 0)

    com_mesh = pb.posed_mesh_com(body, R, p)
    info = pb.summary(body)
    info.update({
        "fk_max_err_mm": None if fk_err is None else round(fk_err * 1000, 4),
        "floor_y_est_m": round(floor_y, 4),
        "com_noise_sd_mm": [round(float(x) * 1000, 3) for x in sd_pos],
        "com_vs_posed_mesh_com_mm": {"rms": round(float(np.sqrt(((com_mesh - com_raw) ** 2).sum(1).mean())) * 1000, 2),
                                     "max": round(float(np.linalg.norm(com_mesh - com_raw, axis=1).max()) * 1000, 2)},
        "xcom_omega0": round(w0, 4), "xcom_pendulum_l_m": round(leg, 4),
        "contact_fraction": {"flight": float(np.mean(nc == 0)), "single": float(np.mean(nc == 1)),
                             "double": float(np.mean(nc == 2))},
        "f_net_band95_median_n": [round(float(x), 1) for x in np.median(M * a_band, 0)],
    })
    if stature is not None:
        info["stature_input_m"] = stature
        info["stature_rel_diff"] = round(body.stature / stature - 1, 4)
        info["stature_warning"] = abs(body.stature / stature - 1) > 0.05
    return {"t": t, "mass": M, "body": body, "info": info, "S": S, "com_raw": com_raw, "com": com, "vcom": vcom,
            "acom": acom, "sd_pos": sd_pos, "f_net": f_net, "f_net_band": M * a_band, "f_sd_white": M * sd_white,
            "f_sd_boot": M * sd_boot, "f_cutoff_sens": M * sens, "L": L, "dL": dL, "dL_band": dL_band,
            "L_vertical": L[:, 1], "spin_rate": spin_rate, "contact": contact, "support": support,
            "foot_corners": corners, "xcom": xcom, "margin_xcom": margin_x, "margin_com": margin_c,
            "flags": flags, "quality": qual, "com_mesh": com_mesh, "joints": np.einsum("tjab,jb->tja", R, body.j_rest) + p,
            "elem_com": c}


# ----------------------------------------------------------------------------------------------- beats

def beat_impulses(res: dict, beat_t: np.ndarray, cfg: PhysicsConfig, rng=None) -> dict:
    """Impulse of the non-gravitational external force per beat interval: J = m dv - m g dt (N s), its mean
    force J/dt, and a 95 % band from the residual bootstrap (velocity at the interval ends only)."""
    rng = rng or np.random.default_rng(cfg.seed)
    t = res["lead"]["t"]
    lo, hi = t[0] + cfg.edge_s, t[-1] - cfg.edge_s
    b = np.asarray(beat_t, float)
    b = b[(b >= lo) & (b <= hi)]
    out = {"t0": b[:-1], "t1": b[1:]}
    if len(b) < 2:
        return out
    dt = np.diff(b)
    tot = 0
    var = 0
    for role in ROLES:
        r = res[role]
        S = r["S"]
        V = S.operator(1, b)                                   # (B+1, N)
        v = V @ r["com_raw"]
        J = r["mass"] * np.diff(v, axis=0) - r["mass"] * G_VEC[None] * dt[:, None]
        D = V[1:] - V[:-1]
        block = max(2, int(round(cfg.block_s / float(np.median(np.diff(t))))))
        sd_boot = block_bootstrap(D, r["com_raw"] - r["com"], cfg.n_boot, block, rng)
        # the residual lacks the sub-cutoff part of the noise, which dominates velocity differences: also propagate
        # the white-noise estimate and keep the larger
        sd_white = r["sd_pos"][None] * np.sqrt((D ** 2).sum(1))[:, None]
        sd = r["mass"] * np.maximum(sd_boot, sd_white)
        out[f"impulse_{role}"] = J
        out[f"impulse_{role}_band"] = Z95 * sd
        out[f"mean_force_{role}"] = J / dt[:, None]
        tot = tot + J
        var = var + sd ** 2
    out["impulse_couple"] = tot
    out["impulse_couple_band"] = Z95 * np.sqrt(var)
    out["mean_force_couple"] = tot / dt[:, None]
    return out


def cycle_grf(imp: dict, beats_t: np.ndarray, m_total: float, per: int = 8) -> dict:
    """Mean couple vertical GRF over whole beat cycles (measures of `per` beat entries) / ((mA+mB) g)."""
    if "impulse_couple" not in imp:
        return {"n_cycles": 0}
    t0, t1 = imp["t0"], imp["t1"]
    first = np.searchsorted(np.asarray(beats_t), t0[0])
    start = (-first) % per  # align to measure boundaries of the full grid
    rows = []
    for s in range(start, len(t0) - per + 1, per):
        Jy = imp["impulse_couple"][s:s + per, 1].sum()
        band = math.sqrt((imp["impulse_couple_band"][s:s + per, 1] ** 2).sum())  # conservative-ish (indep.)
        dur = t1[s + per - 1] - t0[s]
        rows.append({"t0": float(t0[s]), "t1": float(t1[s + per - 1]), "grf_y_over_mg": float(Jy / dur / (m_total * G)),
                     "band95_rel": float(band / dur / (m_total * G))})
    if not rows:
        return {"n_cycles": 0}
    r = np.array([x["grf_y_over_mg"] for x in rows])
    return {"n_cycles": len(rows), "per_beats": per, "cycles": rows, "mean_ratio": float(r.mean()),
            "max_abs_dev": float(np.abs(r - 1).max()), "beat_max_abs_dev":
                float(np.abs(imp["mean_force_couple"][:, 1] / (m_total * G) - 1).max())}


# ----------------------------------------------------------------------------------------------- partner

def _hand_center(joints: np.ndarray, side: str) -> np.ndarray:
    hand = range(25, 40) if side == "l" else range(40, 55)
    wrist = 20 if side == "l" else 21
    return 0.5 * joints[:, wrist] + 0.5 * joints[:, list(hand)].mean(1)


def _points(joints: np.ndarray, spec: dict) -> dict:
    out = {}
    for name, j in spec.items():
        out[name] = _hand_center(joints, j[-1]) if isinstance(j, str) else joints[:, j]
    return out


def partner_connections(lead_j: np.ndarray, follow_j: np.ndarray) -> list[dict]:
    """Per-frame active connections: lead hands / chest vs the nearest follow point within the contact distance."""
    lp, fp = _points(lead_j, LEAD_POINTS), _points(follow_j, FOLLOW_POINTS)
    conns = []
    for ln, lpos in lp.items():
        names = ["follow_spine3"] if ln == "lead_chest" else list(fp)
        dists = np.stack([np.linalg.norm(lpos - fp[n], axis=1) for n in names], 1)
        k = dists.argmin(1)
        dmin = dists[np.arange(len(k)), k]
        lim = CONTACT_DIST["lead_chest"] if ln == "lead_chest" else CONTACT_DIST["hand"]
        fpos = np.stack([fp[names[kk]][i] for i, kk in enumerate(k)])
        conns.append({"name": ln, "active": dmin < lim, "dist": dmin, "point": 0.5 * (lpos + fpos),
                      "follow_point": [names[kk] for kk in k]})
    return conns


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def partner_interval(res: dict, cfg: PhysicsConfig) -> dict:
    """Bounds on the partner force along the follow->lead COM axis (force ON the follow; + = tension/pull)."""
    from scipy.optimize import linprog
    L, F = res["lead"], res["follow"]
    T = len(L["t"])
    conns = partner_connections(L["joints"], F["joints"])
    mu, r_t = cfg.mu, cfg.torsion_r
    edges = np.array([[sx * mu, 1.0, sz * mu] for sx in (-1, 1) for sz in (-1, 1)])  # circumscribed pyramid
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    lo2, hi2 = lo.copy(), hi.copy()
    relaxed = np.zeros(T, bool)  # feasible only with the doubled tolerance
    status = np.zeros(T, np.int64)
    axis = L["com"] - F["com"]
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
    n_conn = np.zeros(T, np.int64)
    for i in range(T):
        act = [c for c in conns if c["active"][i]]
        n_conn[i] = len(act)
        if not act:
            status[i] = 2
            lo[i] = hi[i] = lo2[i] = hi2[i] = 0.0
            continue
        dancers = []  # LP columns: [beta_L, tau_L, beta_F, tau_F, F_k (3 per active connection)]
        for d, sgn in ((L, 1.0), (F, -1.0)):
            P = d["foot_corners"][i]
            dancers.append((d, sgn, P))
        nb = [len(P) * 4 for _, _, P in dancers]
        nk = 3 * len(act)
        nvar = nb[0] + 1 + nb[1] + 1 + nk
        A_eq_rows, b_rows, tol_rows = [], [], []
        A_ub_extra, b_ub_extra = [], []
        off = 0
        for di, (d, sgn, P) in enumerate(dancers):
            Gd = d["com"][i]
            Af = np.zeros((3, nvar))
            Am = np.zeros((3, nvar))
            for pi_, pt in enumerate(P):
                for e, de in enumerate(edges):
                    col = off + pi_ * 4 + e
                    Af[:, col] = de
                    Am[:, col] = np.cross(pt - Gd, de)
            tau_col = off + nb[di]
            Am[1, tau_col] = 1.0
            for k, c in enumerate(act):
                cols = slice(nb[0] + 1 + nb[1] + 1 + 3 * k, nb[0] + 1 + nb[1] + 1 + 3 * k + 3)
                Af[:, cols] = sgn * np.eye(3)
                Am[:, cols] = sgn * _skew(c["point"][i] - Gd)
            A_eq_rows += [Af, Am]
            b_rows += [d["f_net"][i], d["dL"][i]]
            tol_rows += [np.maximum(d["f_net_band"][i], cfg.tol_min_n), np.maximum(d["dL_band"][i], cfg.tol_min_nm)]
            # |tau| <= r * sum(beta) (each edge has unit normal component)
            for sgn_t in (1.0, -1.0):
                row = np.zeros(nvar)
                row[tau_col] = sgn_t
                row[off:off + nb[di]] = -r_t
                A_ub_extra.append(row)
                b_ub_extra.append(0.0)
            off += nb[di] + 1
        A = np.concatenate(A_eq_rows)
        b = np.concatenate(b_rows)
        tol = np.concatenate(tol_rows)
        A_ub = np.concatenate([A, -A, np.array(A_ub_extra)])
        bounds = []
        for di in range(2):
            bounds += [(0, None)] * nb[di] + [(None, None)]
        bounds += [(-cfg.partner_cap_n, cfg.partner_cap_n)] * nk
        cobj = np.zeros(nvar)
        u = axis[i]
        for k in range(len(act)):
            cols = slice(nb[0] + 1 + nb[1] + 1 + 3 * k, nb[0] + 1 + nb[1] + 1 + 3 * k + 3)
            cobj[cols] = -u  # force on follow = -F_k (F_k acts on the lead); tension = (-sum F_k) . u
        def solve(k_tol):
            b_ub = np.concatenate([b + k_tol * tol, -(b - k_tol * tol), np.array(b_ub_extra)])
            r1 = linprog(cobj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            r2 = linprog(-cobj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            return (r1.fun, -r2.fun) if r1.status == 0 and r2.status == 0 else None

        # 95 % bands as tolerances; the doubled (~4 sigma) tolerance guards against intervals that are only narrow
        # because the problem is barely feasible (noise), and it decides identifiability.
        i1, i2 = solve(1.0), solve(2.0)
        if i2 is None:
            status[i] = 3
            continue
        lo[i], hi[i] = i1 if i1 is not None else i2
        lo2[i], hi2[i] = i2
        if i1 is None:
            relaxed[i] = True
    width = hi - lo
    supp_ok = np.array([(L["contact"][i].sum() <= 1) or (F["contact"][i].sum() <= 1) for i in range(T)])
    ident = (status == 0) & supp_ok & (hi2 - lo2 <= cfg.identifiable_width_n)
    status[ident] = 1
    return {"lo": lo, "hi": hi, "width": width, "lo_robust": lo2, "hi_robust": hi2, "relaxed": relaxed,
            "status": status, "axis": axis, "n_connections": n_conn,
            "connections": {c["name"]: c["active"] for c in conns}}


# ----------------------------------------------------------------------------------------------- top level

def analyze(npzs: dict, cfg: PhysicsConfig, beats_t: np.ndarray | None = None, masses: dict | None = None,
            statures: dict | None = None, partner: bool = True) -> dict:
    rng = np.random.default_rng(cfg.seed)
    masses, statures = masses or {}, statures or {}
    res = {role: analyze_dancer(npzs[role], cfg, masses.get(role), statures.get(role), rng) for role in ROLES}
    tL, tF = res["lead"]["t"], res["follow"]["t"]
    if len(tL) != len(tF) or np.abs(tL - tF).max() > 1e-6:
        raise ValueError("lead and follow must share the master time grid (contract: 30 Hz grid)")
    m_tot = res["lead"]["mass"] + res["follow"]["mass"]
    res["grf_couple"] = res["lead"]["f_net"] + res["follow"]["f_net"]
    res["grf_couple_band"] = np.sqrt(res["lead"]["f_net_band"] ** 2 + res["follow"]["f_net_band"] ** 2)
    res["m_total"] = m_tot
    if beats_t is not None:
        res["beats"] = beat_impulses(res, beats_t, cfg, rng)
        res["cycles"] = cycle_grf(res["beats"], beats_t, m_tot)
    if partner:
        res["partner"] = partner_interval(res, cfg)
        p = res["partner"]
        bits = np.where(p["status"] == 1, FLAG_BITS["partner_identifiable"], 0) \
            | np.where(p["status"] == 2, FLAG_BITS["partner_no_contact"], 0) \
            | np.where(p["status"] == 3, FLAG_BITS["partner_inconsistent"], 0)
        for role in ROLES:
            res[role]["flags"] = res[role]["flags"] | bits
    return res


NOTES = [
    "World: right-handed Y-up metres, floor y=0 (Unity = (x, y, -z)). Forces in N on the dancer; F_net = m (a_com - g) "
    "is the total non-gravitational external force (floor + partner); at rest F_net = (0, m g, 0).",
    "Derivatives: penalised quintic smoothing spline on the real timestamps (penalty on f''', gain 1/(1+(f/fc)^6), "
    "fc = cutoff_hz). The first/last edge_s seconds are flagged 'edge' (spline boundary).",
    "f_net_band = 95 % half-width per axis: 1.96 x max(white-noise propagation of the 4th-difference COM noise, moving-"
    "block residual bootstrap) combined with half the 4-6 Hz cutoff spread. It EXCLUDES the mass scale uncertainty "
    "(mass_rel_sigma, multiplicative on every force).",
    "grf_couple = F_net(lead) + F_net(follow): partner forces cancel, so the couple's floor force is identifiable.",
    "Beat impulses J = m (v(t1) - v(t0)) - m g (t1 - t0) per beat-grid interval use the spline velocity at the interval "
    "ends only; mean force = J / dt. More robust than per-frame forces.",
    "XCoM (Hof 2005) = COM_xz + vCOM_xz / omega0, omega0 = sqrt(g / l), l = standing COM height of the shaped body; "
    "xcom_xz / support_xz are floor-plane (x, z) points (y = 0); "
    "balance margin = signed distance XCoM -> support polygon (convex hull of the posed sole vertices of feet in "
    "contact), positive inside, NaN in flight.",
    "L = whole-body angular momentum about the COM (element translation + spin, kg m^2/s); L_vertical = Y component "
    "(+ = counter-clockwise seen from above), spin_rate = L_vertical / I_yy(COM).",
    "Partner force: NOT identifiable in general (indeterminacy 130-270 N, kinematic noise 30-150 N vs signal 10-30 N). "
    "The interval is the LP range of the force ON THE FOLLOW along the follow->lead COM axis (+ = tension) over all "
    "contact force distributions consistent with both F_net and dL/dt within their bands (friction pyramid mu, "
    "unilateral feet at the sole contact hull corners, free forces at active hand/chest connections, each "
    "component capped). 'identifiable' only with a dancer in flight/single support and width <= identifiable_width_n.",
    "head-movement PartnerConnection (spring-damper on frame length) is a HEURISTIC visualisation, not this estimate.",
    "Segment masses: LBS-argmax regions of the voxelised shaped mesh with uniform density; de Leva (1996) deviations "
    "are logged per dancer (trunk/thigh differ by construction: gluteal mass is skinned to the pelvis).",
]


def _r(x, nd):
    a = np.round(np.asarray(x, float), nd)
    return np.where(np.isfinite(a), a, np.nan).tolist()


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def to_json(res: dict, cfg: PhysicsConfig, meta: dict) -> dict:
    t = res["lead"]["t"]
    dancers = {}
    for role in ROLES:
        r = res[role]
        sup = [[q for q in poly.tolist() if math.isfinite(q[0])] for poly in np.round(r["support"], 3)]
        dancers[role] = {
            "mass_kg": round(r["mass"], 2), "mass_rel_sigma": r["body"].mass_rel_sigma,
            "mass_source": r["body"].mass_source,
            "com": _r(r["com"], 4), "vcom": _r(r["vcom"], 3), "xcom_xz": _r(r["xcom"], 4),
            "f_net": _r(r["f_net"], 1), "f_net_band": _r(r["f_net_band"], 1),
            "balance_margin": _r(r["margin_xcom"], 4), "com_margin": _r(r["margin_com"], 4),
            "contact": [("L" if a else "") + ("R" if b else "") for a, b in r["contact"]],
            "support_xz": sup, "L_vertical": _r(r["L_vertical"], 2), "spin_rate": _r(r["spin_rate"], 3),
            "flags": r["flags"].tolist(), "info": r["info"],
        }
    out = {
        "version": 1, "stage": "physics", "level": "L0 centroidal dynamics",
        "world": "right-handed Y-up metres, floor y=0; Unity = (x, y, -z)", "units": "m, m/s, N, N s, kg m^2/s",
        "t": _r(t, 4), "flag_bits": FLAG_BITS, "dancers": dancers,
        "couple": {"m_total_kg": round(res["m_total"], 2), "mg_n": round(res["m_total"] * G, 1),
                   "grf": _r(res["grf_couple"], 1), "grf_band": _r(res["grf_couple_band"], 1)},
        "config": asdict(cfg), "notes": NOTES, **meta,
    }
    if "beats" in res and "impulse_couple" in res["beats"]:
        b = res["beats"]
        out["beats"] = [{"t0": round(float(b["t0"][k]), 4), "t1": round(float(b["t1"][k]), 4),
                         "impulse": {x: _r(b[f"impulse_{x}"][k], 2) for x in ROLES + ("couple",)},
                         "impulse_band": {x: _r(b[f"impulse_{x}_band"][k], 2) for x in ROLES + ("couple",)},
                         "mean_force": {x: _r(b[f"mean_force_{x}"][k], 1) for x in ROLES + ("couple",)}}
                        for k in range(len(b["t0"]))]
        out["cycles"] = res["cycles"]
    if "partner" in res:
        p = res["partner"]
        out["partner"] = {
            "quantity": "force on the follow along the follow->lead COM axis (N), + = tension (pull), - = compression",
            "interval": [None if p["status"][i] in (0, 3) else [round(float(p["lo_robust"][i]), 1),
                                                                    round(float(p["hi_robust"][i]), 1)]
                         for i in range(len(t))],  # only where identifiable (robust ~4 sigma interval) or no contact
            "lp_range": [[None if not math.isfinite(p["lo"][i]) else round(float(p["lo"][i]), 1),
                          None if not math.isfinite(p["hi"][i]) else round(float(p["hi"][i]), 1)] for i in range(len(t))],
            "status": [PARTNER_STATUS[int(s)] for s in p["status"]],
            "identifiable_fraction": float(np.mean(p["status"] == 1)),
            "heuristic": "head-movement PartnerConnection spring-damper signal: heuristic visualisation only",
        }
    return _clean(out)


def save(res: dict, out_dir: Path, cfg: PhysicsConfig, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrs = {"t": res["lead"]["t"], "grf_couple": res["grf_couple"], "grf_couple_band": res["grf_couple_band"],
            "m_total": res["m_total"]}
    keys = ("com_raw", "com", "vcom", "acom", "f_net", "f_net_band", "f_sd_white", "f_sd_boot", "f_cutoff_sens", "L",
            "dL", "dL_band", "L_vertical", "spin_rate", "contact", "support", "xcom", "margin_xcom", "margin_com",
            "flags", "com_mesh")
    for role in ROLES:
        r = res[role]
        for k in keys:
            arrs[f"{role}_{k}"] = r[k]
        arrs[f"{role}_mass"] = r["mass"]
        arrs[f"{role}_elem_mass"] = r["body"].elem_mass
    for k, v in res.get("beats", {}).items():
        arrs[f"beats_{k}"] = v
    if "partner" in res:
        for k in ("lo", "hi", "lo_robust", "hi_robust", "relaxed", "status", "axis", "n_connections"):
            arrs[f"partner_{k}"] = res["partner"][k]
    np.savez_compressed(out_dir / "physics.npz", **arrs)
    js = to_json(res, cfg, meta)
    (out_dir / "physics.json").write_text(json.dumps(js, separators=(",", ":")))
    return out_dir / "physics.json"


def plot_qa(res: dict, path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = res["lead"]["t"]
    col = {"lead": "#1f4e9c", "follow": "#c0392b"}
    fig, ax = plt.subplots(6, 1, figsize=(13, 16), sharex=True)
    for role in ROLES:
        r = res[role]
        y0 = np.mean(r["com"][:, 1])
        ax[0].plot(t, 100 * (r["com"][:, 1] - y0), color=col[role], label=f"{role} COM y - {y0:.3f} m")
        ax[0].plot(t, 100 * (r["com_raw"][:, 1] - y0), color=col[role], lw=0.5, alpha=0.5)
        mg = r["mass"] * G
        y, b = r["f_net"][:, 1] / mg, r["f_net_band"][:, 1] / mg
        ax[1].plot(t, y, color=col[role], label=f"{role} F_net,y / mg")
        ax[1].fill_between(t, y - b, y + b, color=col[role], alpha=0.15)
        ax[3].plot(t, r["margin_xcom"] * 100, color=col[role], label=f"{role} XCoM margin")
        ax[3].plot(t, r["margin_com"] * 100, color=col[role], ls=":", lw=0.8)
        ax[4].plot(t, r["L_vertical"], color=col[role], label=f"{role} L_vertical")
        base = -3.0 - (2.4 if role == "follow" else 0.0)
        for k in range(2):  # contact rasters below the curves: left (lower) / right foot
            ax[0].fill_between(t, base + 1.1 * k, base + 1.1 * k + r["contact"][:, k], color=col[role],
                               alpha=0.5 if k == 0 else 0.25, step="mid", lw=0)
    mg = res["m_total"] * G
    y, b = res["grf_couple"][:, 1] / mg, res["grf_couple_band"][:, 1] / mg
    ax[2].plot(t, y, color="k", label="couple GRF_y / (m_A+m_B) g")
    ax[2].fill_between(t, y - b, y + b, color="k", alpha=0.12)
    if "beats" in res and "mean_force_couple" in res["beats"]:
        bt = res["beats"]
        ax[2].hlines(bt["mean_force_couple"][:, 1] / mg, bt["t0"], bt["t1"], color="tab:green", lw=2,
                     label="beat-interval mean (impulse)")
        for cyc in res.get("cycles", {}).get("cycles", []):
            ax[2].hlines(cyc["grf_y_over_mg"], cyc["t0"], cyc["t1"], color="tab:orange", lw=2.5)
    ax[2].axhline(1, color="gray", lw=0.6)
    ax[2].axhline(1.03, color="gray", lw=0.4, ls="--")
    ax[2].axhline(0.97, color="gray", lw=0.4, ls="--")
    ax[3].axhline(0, color="gray", lw=0.6)
    if "partner" in res:
        p = res["partner"]
        okk = p["status"] == 0
        ax[5].fill_between(t, np.where(okk, p["lo"], np.nan), np.where(okk, p["hi"], np.nan), color="gray", alpha=0.3,
                           label="LP range (not identifiable)", step="mid")
        idn = p["status"] == 1
        ax[5].fill_between(t, np.where(idn, p["lo"], np.nan), np.where(idn, p["hi"], np.nan), color="tab:purple",
                           alpha=0.7, label="identifiable interval", step="mid")
        ax[5].plot(t, np.where(p["status"] == 3, 0, np.nan), "rx", ms=3, label="inconsistent")
        ax[5].set_ylim(-450, 450)
    labels = ["COM height (cm); contact L/R rasters", "F_net,y / m g (95% band)", "couple vertical GRF", "balance margin (cm)",
              "L vertical (kg m^2/s)", "partner force on follow (N, + tension)"]
    for a, lab in zip(ax, labels):
        a.set_ylabel(lab, fontsize=8)
        a.legend(fontsize=7, loc="upper right")
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("t (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def find_beats(work: Path) -> Path | None:
    for p in (work / "timing" / "beats.json", work / "beats.json", work / "zouk-time-analysis.json"):
        if p.exists():
            return p
    return None


def load_beat_times(path: Path) -> np.ndarray:
    arr = np.asarray(json.loads(Path(path).read_text()), float).reshape(-1, 2)
    return arr[:, 0]


def run(work: Path, cfg: PhysicsConfig | None = None, beats_path: Path | None = None, masses: dict | None = None,
        statures: dict | None = None, out_dir: Path | None = None, npzs: dict | None = None, plot: bool = True) -> dict:
    cfg = cfg or PhysicsConfig()
    npzs = npzs or {role: load_npz(work / "smplx" / f"{role}.npz") for role in ROLES}
    bp = beats_path or find_beats(work)
    beats_t = load_beat_times(bp) if bp else None
    res = analyze(npzs, cfg, beats_t, masses, statures)
    out_dir = out_dir or work / "physics"
    meta = {"take": work.name, "beats": None if bp is None else str(bp), "warnings": []}
    if bp is None:
        meta["warnings"].append("no beats json: beat impulses skipped")
    for role in ROLES:
        if res[role]["info"].get("stature_warning"):
            meta["warnings"].append(f"{role}: mesh stature differs from the input by "
                                    f"{100 * res[role]['info']['stature_rel_diff']:+.1f} %")
    res["_json"] = save(res, out_dir, cfg, meta)
    if plot:
        plot_qa(res, out_dir / "qa_physics.png", f"{work.name}: L0 centroidal physics")
    return res


# ----------------------------------------------------------------------------------------------- validation

def add_noise(npz: dict, rng, transl_m: float, rot_deg: float) -> dict:
    """Gaussian noise on the root translation and every joint rotation (axis-angle composition)."""
    from scipy.spatial.transform import Rotation
    d = dict(npz)
    T = len(d["t"])
    d["transl"] = d["transl"] + rng.normal(0, transl_m, (T, 3))
    for k, n in (("global_orient", 1), ("body_pose", 21)):
        rv = np.asarray(d[k], float).reshape(T * n, 3)
        noise = Rotation.from_rotvec(rng.normal(0, math.radians(rot_deg), (T * n, 3)))
        d[k] = (noise * Rotation.from_rotvec(rv)).as_rotvec().reshape(T, 3 * n)
    d.pop("joints", None)
    return d


def gate_g6(res: dict, tol: float = 0.03) -> dict:
    cyc = res.get("cycles", {})
    ok = cyc.get("n_cycles", 0) > 0 and cyc["max_abs_dev"] <= tol
    return {"gate": "G6", "tol": tol, "passed": bool(ok), "n_cycles": cyc.get("n_cycles", 0),
            "mean_ratio": cyc.get("mean_ratio"), "max_abs_dev_per_cycle": cyc.get("max_abs_dev"),
            "max_abs_dev_per_beat": cyc.get("beat_max_abs_dev")}


def validate_synthetic(cfg: PhysicsConfig | None = None, noise=((0.01, 1.0),), seeds=(0, 1, 2)) -> dict:
    """Run on work/synthetic (clean + noisy) and check: masses 60-90 kg, de Leva deviations, COM plausibility,
    F_net,y per frame averages ~ m g, gate G6 (couple vertical GRF over beat cycles within 3 % of (mA+mB) g)."""
    cfg = cfg or PhysicsConfig()
    work = PACKAGE_ROOT / "work" / "synthetic"
    res = run(work, cfg)
    rep = {"clean": _checks(res)}
    rep["clean"]["G6"] = gate_g6(res)
    npzs = {role: load_npz(work / "smplx" / f"{role}.npz") for role in ROLES}
    beats_t = load_beat_times(find_beats(work))
    for tr, rot in noise:
        rows = []
        for s in seeds:
            rng = np.random.default_rng(100 + s)
            noisy = {role: add_noise(npzs[role], rng, tr, rot) for role in ROLES}
            r = analyze(noisy, cfg, beats_t, partner=(s == seeds[0]))
            row = {"seed": s, "G6": gate_g6(r), **{k: v for k, v in _checks(r).items() if k != "dancers"}}
            row["f_net_y_band95_median_n"] = {role: float(np.median(r[role]["f_net_band"][:, 1])) for role in ROLES}
            row["f_net_y_err_rms_n"] = {role: float(np.sqrt(np.mean((r[role]["f_net"][:, 1] - res[role]["f_net"][:, 1]) ** 2)))
                                        for role in ROLES}
            row["f_net_y_err_in_band"] = {role: float(np.mean(np.abs(r[role]["f_net"][:, 1] - res[role]["f_net"][:, 1])
                                                            <= r[role]["f_net_band"][:, 1])) for role in ROLES}
            if "partner" in r:
                row["partner_identifiable_fraction"] = float(np.mean(r["partner"]["status"] == 1))
                row["partner_median_width_n"] = float(np.nanmedian(r["partner"]["width"]))
            if s == seeds[0]:
                plot_qa(r, work / "physics" / f"qa_physics_noise_{int(tr * 1000)}mm_{rot:g}deg.png",
                        f"synthetic + noise {tr * 1000:.0f} mm / {rot:g} deg")
            rows.append(row)
        rep[f"noise_{tr * 1000:.0f}mm_{rot:g}deg"] = rows
    gates = [rep["clean"]["G6"]["passed"]] + [r["G6"]["passed"] for k, v in rep.items() if k.startswith("noise")
                                              for r in v]
    rep["G6_passed_all"] = bool(all(gates))
    (work / "physics" / "validation.json").write_text(json.dumps(_clean(rep), indent=1))
    return rep


def _checks(res: dict) -> dict:
    out = {"dancers": {}}
    t = res["lead"]["t"]
    inner = (t > t[0] + 0.5) & (t < t[-1] - 0.5)
    for role in ROLES:
        r = res[role]
        info = r["info"]
        h = info["stature_m"]
        out["dancers"][role] = {
            "mass_kg": r["mass"], "mass_ok_60_90": 60 <= r["mass"] <= 90,
            "de_leva_max_abs_diff_pct_points": info["de_leva"]["max_abs_diff_pct_points"],
            "com_height_over_stature": float(np.mean(r["com"][inner, 1]) / h),
            "com_speed_max": float(np.linalg.norm(r["vcom"][inner], axis=1).max()),
            "f_net_y_mean_over_mg": float(np.mean(r["f_net"][inner, 1]) / (r["mass"] * G)),
            "contact_fraction": info["contact_fraction"],
            "com_vs_mesh_mm": info["com_vs_posed_mesh_com_mm"],
        }
    out["couple_grf_y_mean_over_mg"] = float(np.mean(res["grf_couple"][inner, 1]) / (res["m_total"] * G))
    if "partner" in res:
        out["partner_status_fraction"] = {PARTNER_STATUS[k]: float(np.mean(res["partner"]["status"] == k))
                                          for k in PARTNER_STATUS}
    return out
