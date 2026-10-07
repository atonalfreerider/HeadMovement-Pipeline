"""world_from_sfm bootstrap: gravity, floor, metric scale and origin from the cameras and the two dancers.

  gravity  the building's vertical (vertical vanishing points of the main-model cameras, vertical.py) when it is
           available and consistent; it is checked against the mean camera up vector (phones are held upright:
           -row 1 of R) and against the plane through the dancers' triangulated ankle keypoints (COCO 15/16,
           people/<cam>.npz, dancer boxes from dancers/roles.json; the LOW ankle points). Fallback: ankle plane.
           (larissa_kadu: the ankle plane and the sparse floor near the dancers are ~5 deg off the building
           vertical, while the building's horizontal edges are perpendicular to it within 0.1 deg: the floor
           patch, not the room, is what the reconstruction gets wrong - see report.json diagnostics.)
  floor    ankle plane (normal = gravity) lowered by the ankle-joint height (0.039 x stature + shoe sole, default
           2.5 cm sneaker sole, takes/<take>.toml [dancers.<role>] shoe_sole_m), then refined by the densest height
           band of the sparse SfM points near it within 3 m of the dancers (scuffs). Points far below the floor are
           mirror / glossy-floor reflections (counted).
  scale    scale_rule "chain" (default since 2026-10-06): each dancer's UNFOLDED keypoint chain = median leg
           (hip-knee + knee-ankle, both sides) + median trunk (mid-hip -> mid-shoulder) + median mid-shoulder ->
           mid-eye, in SfM units, vs CHAIN_FRAC x stature. Segment lengths change little when the knees bend or the
           dancer crouches, so the rule does not need upright frames (zouk: the lead of larissa_kadu is never upright
           in 6 s - median knee angle 147 deg); trunk and neck are not rigid, so it is pose-ROBUST, not invariant
           (chain_measure). Both dancers are combined by an inverse-variance log mean whose per-dancer variance is the
           block-bootstrap SE plus SCALE_SYS_REL (1 %, stature rounding / self-report and individual proportions) in
           quadrature; diagnostics.scale_disagreement > 1.5 % means one stature (or one dancer's triangulation) is
           inconsistent.
           CHAIN_FRAC = 0.900 is a CALIBRATION, not a validated constant: it is the value that reproduces larissa_kadu's
           MAMMA-betas scale (rest height = stature + 2.5 cm sole; 0.90024 exactly, per dancer 0.9016 / 0.8989) on the
           56-62 s window. A chain-vs-betas comparison on that window is therefore circular. The independent prior is
           anthropometric, 0.895 +- 0.01 (eye - ankle 0.894 H on SMPL-X / 0.897 H Drillis & Contini, COCO ankle ~1 cm
           above the joint, shoulder -> eye ~1 % longer than its vertical part): with it the same window gives a scale
           0.75 % smaller. Expect the chain rule to be good to about +-1 % for other dancers, PLUS the absolute
           uncertainty of the statures themselves (SCALE_ABS_REL).
           Sole coupling: the calibration assumed target = stature + 2.5 cm (CHAIN_FRAC_SOLE_M), so CHAIN_FRAC moves
           with the take's shoe_sole_m exactly like the betas scale does. The rule therefore sets the chain to
           CHAIN_FRAC x (stature + shoe_sole_m - CHAIN_FRAC_SOLE_M) (chain_stature; = stature at the default sole), and
           the chain stays a consistent cross-check of a betas reference rebuilt with another sole. Read physically:
           the chain (ankle keypoint -> eye, both raised by the shoe) is anatomical, i.e. ~0.90 x the BAREFOOT height;
           shoe_sole_m 0 = "statures measured in these ~2.5 cm sneakers" -> barefoot = stature - 2.5 cm. Neither the
           chain nor the anthropometric prior can tell barefoot from shod statures: both scale with the same height.
           ABSOLUTE scale (any rule): statures are whole inches and probably self-reported (+-0.7 %), and whether they
           are barefoot heights with the fitted SMPL-X sole at the sneaker sole (target = stature + shoe_sole_m) is
           unconfirmed: sole 2.5 cm vs 0 moves the scale by 1.5 %. SCALE_ABS_REL = 1.5 % is recorded in diagnostics.
           A take re-cut from recordings that already have a betas-corrected world should INHERIT it instead
           ([cameras] scale_reference, world_scale.adopt_reference: same physical camera frames -> SfM similarity);
           the chain then is a cross-check only (diagnostics.scale_reference.chain_vs_reference).
           scale_rule "eye_p90" (the larissa_kadu world of 2026-10-06; keep for reproduction only): eye-above-ankle
           height, 90th percentile over the window, vs 0.897 x stature. It assumed the 90th percentile is upright;
           bent-knee dancing made the larissa_kadu world 4.8 % too large (bodies 1.845 / 1.747 m for 1.727 / 1.651 m).
           Select per take with [cameras] scale_rule = "eye_p90" in takes/<take>.toml.
           Post-fit check / correction: cameras/world_scale.py (betas rest height vs stature + sole, rescale()).
           Statures come from takes/<take>.toml [dancers] (user supplied) or default to lead 1.75 / follow 1.65 m
           ("assumed"). G2a gates on all of this: scale_checks() (cameras/report.json gates_G2a.world_scale*).
  axes     Y = up (floor normal), origin = dancers' mean floor position, X = horizontal projection of
           camera 06's mean right vector (reference clock camera), Z = X x Y (right-handed).
X_world = s R (X_sfm) + t  (contract: cameras/world_from_sfm.json).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

EYE_MINUS_ANKLE = 0.936 - 0.039  # fraction of stature
ANKLE_FRAC = 0.039
SOLE_M = 0.025
DEFAULT_STATURE = {"lead": 1.75, "follow": 1.65}
KP_EYES, KP_ANKLES = (1, 2), (15, 16)
KP_SHOULDERS, KP_HIPS, KP_KNEES = (5, 6), (11, 12), (13, 14)
KP_SCALE = KP_EYES + KP_SHOULDERS + KP_HIPS + KP_KNEES + KP_ANKLES  # what bootstrap() needs (both rules)
CHAIN_FRAC = 0.900  # unfolded COCO chain ankle..eye / stature: CALIBRATED on larissa_kadu's betas (module docstring)
CHAIN_FRAC_SOLE_M = 0.025  # ... with target = stature + THIS sole: the chain rule uses stature + sole - CHAIN_FRAC_SOLE_M
CHAIN_FRAC_PRIOR = (0.895, 0.010)  # independent anthropometric value +- 1 sigma
SCALE_RULES = ("chain", "eye_p90")
SCALE_AGREE_REL = 0.015  # two dancers' scale estimates should agree within this
SCALE_SYS_REL = 0.010  # per-dancer systematic error (stature rounding / self-report, proportions), added in quadrature
SCALE_ABS_REL = 0.015  # absolute scale uncertainty from the statures (barefoot? sole in the mesh? self-reported)
SCALE_MAX_REL_SE = 0.0075  # gate: per-dancer statistical chain SE
SCALE_MAX_HALVES = 0.02  # gate: first vs second half of the window, per dancer
SCALE_MAX_VS_REFERENCE = 0.02  # gate: chain rule vs an adopted reference scale


def undistort_pts(xy: np.ndarray, K: np.ndarray, k1: float, iters: int = 5) -> np.ndarray:
    """Pixel (OpenCV convention) -> normalised camera coordinates (SIMPLE_RADIAL inverse, fixed point)."""
    x = (xy[..., 0] - K[0, 2]) / K[0, 0]
    y = (xy[..., 1] - K[1, 2]) / K[1, 1]
    xu, yu = x.copy(), y.copy()
    for _ in range(iters):
        d = 1 + k1 * (xu * xu + yu * yu)
        xu, yu = x / d, y / d
    return np.stack([xu, yu], -1)


def triangulate(nxy: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation from normalised coordinates nxy (V,2) and poses R (V,3,3), t (V,3)."""
    A = []
    for (x, y), Ri, ti in zip(nxy, R, t):
        P = np.hstack([Ri, ti[:, None]])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    return X[:3] / X[3]


def triangulate_robust(xy: np.ndarray, K: np.ndarray, k1: np.ndarray, R: np.ndarray, t: np.ndarray,
                       max_px: float = 8.0, min_views: int = 2) -> tuple[np.ndarray | None, float, int]:
    """Triangulate one point seen in V views (pixels), dropping the worst view while its error > max_px."""
    idx = list(range(len(xy)))
    nxy = np.array([undistort_pts(xy[i], K[i], k1[i]) for i in idx])
    while len(idx) >= min_views:
        X = triangulate(nxy[idx], R[idx], t[idx])
        errs = []
        for i in idx:
            Xc = R[i] @ X + t[i]
            if Xc[2] <= 0:
                errs.append(1e6)
                continue
            u = Xc[:2] / Xc[2]
            errs.append(np.linalg.norm((u - nxy[i]) * K[i][0, 0]))
        errs = np.asarray(errs)
        if errs.max() <= max_px:
            return X, float(np.median(errs)), len(idx)
        if len(idx) == min_views:
            break
        idx.pop(int(np.argmax(errs)))
    return None, float("nan"), 0


def fit_plane(P: np.ndarray, thr: float, prior_n: np.ndarray | None = None, max_angle_deg: float = 20.0,
              iters: int = 2000, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane n.x = d (n unit, oriented like prior_n) + least-squares refit. Returns (n, d, inliers)."""
    rng = np.random.default_rng(seed)
    best = None
    cos_max = np.cos(np.radians(max_angle_deg))
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if prior_n is not None:
            if np.dot(n, prior_n) < 0:
                n = -n
            if np.dot(n, prior_n) < cos_max:
                continue
        d = np.dot(n, s[0])
        inl = np.abs(P @ n - d) < thr
        if best is None or inl.sum() > best[2].sum():
            best = (n, d, inl)
    if best is None:
        raise ValueError("no plane")
    n, d, inl = best
    for _ in range(3):  # LS refit on inliers
        c = P[inl].mean(0)
        _, _, vt = np.linalg.svd(P[inl] - c)
        n = vt[-1] * (1 if prior_n is None or np.dot(vt[-1], prior_n) >= 0 else -1)
        d = float(np.dot(n, c))
        inl = np.abs(P @ n - d) < thr
    return n, d, inl


def dancer_keypoints(work: Path, cam: str, n_frames: int) -> dict[str, np.ndarray]:
    """{'lead': (F,17,3), 'follow': (F,17,3)} COCO keypoints of the dancers (zeros where unknown).

    Dancer boxes come from dancers/roles.json; without it the two largest boxes near the image centre are used
    (role unknown -> both under 'lead'/'follow' by size)."""
    from ..detect import iou_matrix
    pp = np.load(work / "people" / f"{cam}.npz")
    boxes, valid, kpts = pp["boxes"], pp["valid"], pp["kpts"]
    out = {r: np.zeros((n_frames, 17, 3), np.float32) for r in ("lead", "follow")}
    roles_p = work / "dancers" / "roles.json"
    roles = json.loads(roles_p.read_text())["cams"].get(cam) if roles_p.exists() else None
    for i in range(min(n_frames, len(boxes))):
        b = boxes[i][valid[i]]
        k = kpts[i][valid[i]]
        if len(b) == 0:
            continue
        if roles is not None:
            fr = roles["frames"][i] if i < len(roles["frames"]) else None
            for role in ("lead", "follow"):
                rb = (fr or {}).get(role)
                if not rb or rb.get("box") is None:
                    continue
                iou = iou_matrix(np.asarray(rb["box"], float)[None], b)[0]
                j = int(np.argmax(iou))
                if iou[j] > 0.5:
                    out[role][i] = k[j]
        else:
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            for role, j in zip(("lead", "follow"), np.argsort(-area)[:2]):
                out[role][i] = k[j]
    return out


def triangulate_dancers(work: Path, tracks: dict, grid_dt: float = 1 / 30, max_dt: float = 0.02,
                        min_conf: float = 0.5, kps: tuple[int, ...] = KP_SCALE,
                        return_grid: bool = False):
    """Triangulated COCO keypoints `kps` of both dancers on a 30 Hz grid (SfM frame).

    tracks[cam] = dict(t, K, k1, R, tvec, ok). Returns {role: {kp: (T,3) NaN where unknown}} (+ grid)."""
    t0 = max(tr["t"][0] for tr in tracks.values())
    t1 = min(tr["t"][-1] for tr in tracks.values())
    grid = np.arange(t0, t1, grid_dt)
    kp = {cam: dancer_keypoints(work, cam, len(tr["t"])) for cam, tr in tracks.items()}
    out = {r: {k: np.full((len(grid), 3), np.nan) for k in kps} for r in ("lead", "follow")}
    for g, tg in enumerate(grid):
        sel = []
        for cam, tr in tracks.items():
            i = int(np.argmin(np.abs(tr["t"] - tg)))
            if abs(tr["t"][i] - tg) <= max_dt and tr["ok"][i]:
                sel.append((cam, i))
        if len(sel) < 2:
            continue
        for role in ("lead", "follow"):
            for k in kps:
                v = [(cam, i) for cam, i in sel if kp[cam][role][i, k, 2] >= min_conf]
                if len(v) < 2:
                    continue
                xy = np.array([kp[cam][role][i, k, :2] for cam, i in v], float)
                K = np.array([tracks[cam]["K"][i] for cam, i in v])
                k1 = np.array([tracks[cam]["k1"][i] for cam, i in v])
                R = np.array([tracks[cam]["R"][i] for cam, i in v])
                t = np.array([tracks[cam]["tvec"][i] for cam, i in v])
                X, _, nv = triangulate_robust(xy, K, k1, R, t)
                if X is not None and nv >= 2:
                    out[role][k][g] = X
    return (out, grid) if return_grid else out


def mode_1d(h: np.ndarray, width: float) -> tuple[float, np.ndarray]:
    """Densest window of `width` in 1-D samples h -> (mean of the window's samples, inlier mask)."""
    hs = np.sort(h)
    j = np.searchsorted(hs, hs + width, side="right")
    k = int(np.argmax(j - np.arange(len(hs))))
    m = float(hs[k:j[k]].mean())
    inl = np.abs(h - m) <= width / 2
    for _ in range(3):  # re-centre the window on its mean
        if not inl.any():
            break
        m = float(h[inl].mean())
        inl = np.abs(h - m) <= width / 2
    return m, inl


def chain_measure(kp: dict, n_boot: int = 200, block: int = 30, min_n: int = 10, seed: int = 0,
                  eyes: str = "any") -> dict | None:
    """Pose-invariant size of one dancer from triangulated COCO keypoints {kp: (T,3), NaN = unknown} (any units):
    unfolded chain = median leg (hip-knee + knee-ankle, both sides pooled) + median trunk (mid-hip -> mid-shoulder)
    + median mid-shoulder -> mid-eye. Medians per segment, so outlier frames (mis-triangulations, identity swaps) do
    not matter. Legs are rigid; trunk (spine flexion, cambre) and shoulder-eye (neck flexion, zouk head movements)
    are NOT, so the chain is pose-robust, not pose-invariant (larissa_kadu 56-62 s: halves of the window differ by
    1.8 % lead / 3.6 % follow; the follow's shoulder-eye p10-p90 spans 0.19-0.33 SfM units).
    eyes: "any" (default; one eye when the other is missing) | "both" (frames with both eyes only; falls back to
    "any" with < max(min_n, 20) such frames). On larissa_kadu "both" was worse, not better: the follow's both-eye
    frames cluster in the first half (43 / 15) and in head-forward poses, so it moved her per-dancer chain/stature
    1.5 % away from her betas and the lead had no both-eye frame in the first half; one-eye frames read long because
    of the POSE they occur in (turned / tilted head), not because of the single eye (lead: +0.6 %). Both variants
    are always recorded (shoulder_eye_both, shoulder_eye_one_eye).
    rel_se = block-bootstrap (blocks of `block` frames) standard error of the chain relative to its value (floor
    0.1 %; statistical only - see SCALE_SYS_REL). Stability diagnostics: halves (the chain of the first / second half
    of the frames) and halves_rel_diff, per-segment p10/p90 (pct), leg_lr (left / right leg medians),
    per_frame_median (median of per-frame sums over frames with every segment). None when a keypoint is absent or a
    segment has < min_n frames."""
    if any(k not in kp for k in KP_SCALE):
        return None

    def seg(a, b):
        return np.linalg.norm(np.asarray(kp[a], float) - np.asarray(kp[b], float), axis=1)

    legs = np.stack([seg(11, 13) + seg(13, 15), seg(12, 14) + seg(14, 16)], 1)
    hip = 0.5 * (np.asarray(kp[11], float) + np.asarray(kp[12], float))
    sh = 0.5 * (np.asarray(kp[5], float) + np.asarray(kp[6], float))
    e1, e2 = np.asarray(kp[1], float), np.asarray(kp[2], float)
    both = np.isfinite(e1).all(1) & np.isfinite(e2).all(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eye = np.nanmean(np.stack([e1, e2]), 0)
    se = np.linalg.norm(eye - sh, axis=1)
    se_both = np.where(both, se, np.nan)
    if eyes not in ("any", "both"):
        raise ValueError(f"eyes {eyes!r} not in ('any', 'both')")
    if eyes == "both" and np.isfinite(se_both).sum() < max(min_n, 20):
        eyes = "any"
    parts = {"leg": legs, "trunk": np.linalg.norm(sh - hip, axis=1)[:, None],
             "shoulder_eye": (se_both if eyes == "both" else se)[:, None]}
    n = {k: int(np.isfinite(v).sum()) for k, v in parts.items()}
    if min(n.values()) < min_n:
        return None

    def total(idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(sum(np.nanmedian(v[idx]) for v in parts.values()))

    T = len(hip)
    c = total(np.arange(T))
    rng = np.random.default_rng(seed)
    nb = max(1, int(np.ceil(T / block)))
    boots = []
    for _ in range(n_boot):
        st = rng.integers(0, max(1, T - block + 1), nb)
        boots.append(total(np.concatenate([np.arange(s0, min(T, s0 + block)) for s0 in st])))
    boots = np.asarray(boots)
    boots = boots[np.isfinite(boots)]
    rel_se = float(np.std(boots) / c) if len(boots) > 10 else 0.05
    halves = [total(np.arange(0, T // 2)), total(np.arange(T // 2, T))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = {k: float(np.nanmedian(v)) for k, v in parts.items()}
        pct = {k: [float(np.nanpercentile(v, 10)), float(np.nanpercentile(v, 90))] for k, v in parts.items()}
        pf = np.nanmean(legs, 1) + parts["trunk"][:, 0] + parts["shoulder_eye"][:, 0]
    se_one = se[np.isfinite(se) & ~both]
    se_two = se[np.isfinite(se) & both]
    return {"chain": c, **med, "n": n, "rel_se": max(rel_se, 1e-3),
            "halves": halves, "halves_rel_diff": float(abs(halves[1] / halves[0] - 1)) if all(
                np.isfinite(halves)) and halves[0] > 0 else None,
            "pct": pct, "leg_lr": [float(np.nanmedian(legs[:, 0])), float(np.nanmedian(legs[:, 1]))],
            "per_frame_median": float(np.nanmedian(pf)) if np.isfinite(pf).any() else None,
            "shoulder_eye_eyes": eyes,
            "shoulder_eye_both": {"median": float(np.median(se_two)) if len(se_two) else None, "n": int(len(se_two))},
            "shoulder_eye_one_eye": {"median": float(np.median(se_one)) if len(se_one) else None, "n": int(len(se_one))}}


def chain_stature(stature: float, sole: float = SOLE_M) -> float:
    """The height CHAIN_FRAC multiplies: stature + shoe_sole_m - CHAIN_FRAC_SOLE_M (= stature at the default sole).
    Keeps the chain rule in the convention of the betas check (target = stature + sole) whatever the sole is."""
    return float(stature) + float(sole) - CHAIN_FRAC_SOLE_M


def chain_scale(dancers: dict, statures: dict[str, float], soles: float | dict = SOLE_M) -> tuple[dict, dict]:
    """The chain rule on triangulated keypoints {role: {kp: (T,3)}} (SfM units): per dancer chain_measure + s =
    CHAIN_FRAC x chain_stature / chain, and the inverse-variance log mean over the dancers (variance = bootstrap SE^2
    + SCALE_SYS_REL^2). Returns (chains {role: ...}, combined {s, weights, sys_rel, rel_se_combined, s_equal_weight,
    chain_frac, disagreement, agreement_ok}); combined is {} when no dancer has a chain."""
    sd = soles if isinstance(soles, dict) else {r: float(soles) for r in ("lead", "follow")}
    chains = {}
    for role in ("lead", "follow"):
        if role not in statures:
            continue
        cm = chain_measure(dancers.get(role, {}))
        if cm is None:
            continue
        sole = float(sd.get(role, SOLE_M))
        hs = chain_stature(statures[role], sole)
        chains[role] = {**cm, "s": CHAIN_FRAC * hs / cm["chain"], "stature_m": statures[role], "sole_m": sole,
                        "chain_stature_m": hs}
    if not chains:
        return chains, {}
    var = {r: v["rel_se"] ** 2 + SCALE_SYS_REL ** 2 for r, v in chains.items()}  # statistical + systematic
    w = {r: 1.0 / var[r] for r in chains}
    s = float(np.exp(sum(w[r] * np.log(v["s"]) for r, v in chains.items()) / sum(w.values())))
    comb = {"s": s, "weights": {r: float(w[r] / sum(w.values())) for r in w}, "sys_rel": SCALE_SYS_REL,
            "rel_se_combined": float(1.0 / np.sqrt(sum(w.values()))),
            "s_equal_weight": float(np.exp(np.mean([np.log(v["s"]) for v in chains.values()]))),
            "chain_frac": {"value": CHAIN_FRAC, "kind": "calibrated on larissa_kadu c8 MAMMA betas (56-62 s)",
                           "calibration_sole_m": CHAIN_FRAC_SOLE_M,
                           "chain_stature_m": {r: v["chain_stature_m"] for r, v in chains.items()},
                           "prior": list(CHAIN_FRAC_PRIOR), "s_with_prior": s * CHAIN_FRAC_PRIOR[0] / CHAIN_FRAC,
                           "note": "s_with_prior is independent of the betas, NOT of the statures: like the chain it "
                                   "scales with the dancers' height, so it cannot tell barefoot from shod statures"}}
    if len(chains) == 2:
        a, b = (v["s"] for v in chains.values())
        comb["disagreement"] = float(abs(a - b) / s)
        comb["agreement_ok"] = bool(abs(a - b) / s <= SCALE_AGREE_REL)
    return chains, comb


def bootstrap(tracks: dict, dancers: dict, points: np.ndarray, statures: dict[str, float],
              stature_source: str, x_cam: str = "06", up_cams: list[str] | None = None,
              vertical: dict | None = None, max_vertical_vs_cameras_deg: float = 15.0,
              floor_radius_m: float = 3.0, scale_rule: str = "chain", sole_m: float | dict = SOLE_M) -> dict:
    """Compute the Sim3 (s, R, t) SfM -> world plus diagnostics.

    sole_m: shoe sole (m) under the dancers' feet, one value or {role: m} (sole_from_take): the floor offset uses
    the mean, the chain rule each dancer's own (chain_stature).

    vertical: architecture vertical {"u": unit (SfM frame), "n_cams", "max_cam_deg", ...} (vertical.fuse). When
    given and plausible (within max_vertical_vs_cameras_deg of the mean camera up, >= 2 cameras agreeing within
    2 deg) it IS gravity: world Y = u. The ankle plane and the sparse floor then only set the floor OFFSET (normal
    fixed to u, densest height band of the floor points within floor_radius_m of the dancers), and their own
    tilts are recorded as diagnostics. Without it: gravity = normal of the dancers' ankle plane (old behaviour).

    scale_rule: "chain" (default; pose-invariant segment lengths, needs KP_SCALE keypoints - falls back to
    "eye_p90" when they are missing) or "eye_p90" (the old rule). Both are always computed when possible and
    recorded in diagnostics (scale_chain / scale_eye_p90); scale_per_dancer = the adopted rule's estimates."""
    if scale_rule not in SCALE_RULES:
        raise ValueError(f"scale_rule {scale_rule!r} not in {SCALE_RULES}")
    soles = ({r: float(sole_m.get(r, SOLE_M)) for r in ("lead", "follow")} if isinstance(sole_m, dict)
             else {r: float(sole_m) for r in ("lead", "follow")})
    sole_mean = float(np.mean(list(soles.values())))
    ups = []
    for cam, tr in tracks.items():
        if up_cams is not None and cam not in up_cams:
            continue
        ups.append(-tr["R"][tr["ok"], 1, :])
    up_cam = np.concatenate(ups).mean(0)
    up_cam /= np.linalg.norm(up_cam)
    diag: dict = {"up_from_cameras": up_cam.tolist()}

    ank = np.concatenate([dancers[r][k] for r in dancers for k in KP_ANKLES])
    ank = ank[np.isfinite(ank).all(1)]
    diag["ankle_points"] = int(len(ank))
    if len(ank) < 20:
        raise ValueError(f"only {len(ank)} triangulated ankle points: cannot bootstrap the floor")
    h = ank @ up_cam
    low = ank[h <= np.percentile(h, 40)]
    spread = np.percentile(h, 90) - np.percentile(h, 10)
    n_ank, d_ank, inl = fit_plane(low, thr=max(0.05 * spread, 1e-6), prior_n=up_cam, max_angle_deg=25)
    diag["ankle_plane_inliers"] = int(inl.sum())
    diag["ankle_plane_vs_cameras_deg"] = _ang(n_ank, up_cam)
    diag["ankle_plane_normal_sfm"] = n_ank.tolist()
    n, d = n_ank, d_ank
    gravity_source = "ankle plane (dancers' triangulated low ankle keypoints), init mean camera up"
    use_vertical = False
    if vertical is not None:
        u = np.asarray(vertical["u"], float)
        u = u / np.linalg.norm(u) * (1 if np.dot(u, up_cam) >= 0 else -1)
        diag["vertical"] = {"u": u.tolist(), "vs_cameras_up_deg": _ang(u, up_cam),
                            "vs_ankle_plane_deg": _ang(u, n_ank),
                            **{k: v for k, v in vertical.items() if k != "u"}}
        use_vertical = (_ang(u, up_cam) <= max_vertical_vs_cameras_deg and vertical.get("n_cams", 0) >= 2
                        and vertical.get("max_cam_deg", 99.0) <= 2.0)
        diag["vertical"]["adopted"] = bool(use_vertical)
        if use_vertical:
            n = u
            d, _ = mode_1d(low @ n, max(0.1 * spread, 1e-6))  # ankle plane offset, normal fixed
            gravity_source = (f"architecture vertical: vertical vanishing points (LSD segments not below the "
                              f"horizon, people masked) of {vertical.get('n_cams')} main-model cameras, fused "
                              f"(cameras agree within {vertical.get('max_cam_deg', float('nan')):.2f} deg)")
    if not use_vertical and diag["ankle_plane_vs_cameras_deg"] > 15:  # implausible: phones are roughly upright
        n, gravity_source = up_cam, "mean camera up vector (ankle plane disagreed by > 15 deg)"
        d = float(np.percentile(h, 20))
    diag["gravity_vs_cameras_deg"] = _ang(n, up_cam)
    diag["gravity_vs_ankle_plane_deg"] = _ang(n, n_ank)

    # scale (a) old rule: eye-above-ankle-plane heights, 90th percentile = "upright"
    scales = {}
    for role in ("lead", "follow"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eyes = np.nanmean(np.stack([dancers[role][k] for k in KP_EYES]), 0)
        he = eyes @ n - d
        he = he[np.isfinite(he)]
        if len(he) < 10:
            continue
        h90 = float(np.percentile(he, 90))
        scales[role] = {"eye_above_ankle_sfm": h90, "n": int(len(he)),
                        "s": EYE_MINUS_ANKLE * statures[role] / h90, "stature_m": statures[role]}
    # (b) pose-invariant unfolded chain, joint inverse-variance log least squares over both dancers
    chains, chain_comb = chain_scale(dancers, statures, soles)
    diag["scale_eye_p90"] = scales
    diag["scale_chain"] = chains
    rule = scale_rule
    if rule == "chain" and not chains:
        rule = "eye_p90"
        diag["scale_rule_fallback"] = "chain: keypoints for the unfolded chain missing -> eye_p90"
    use = chains if rule == "chain" else scales
    if not use:
        raise ValueError("no dancer eye heights: cannot set the metric scale")
    if rule == "chain":
        s = chain_comb["s"]
        diag["scale_weights"] = chain_comb["weights"]
        diag["scale_sys_rel"] = chain_comb["sys_rel"]
        diag["scale_rel_se_combined"] = chain_comb["rel_se_combined"]  # chain rule, relative to CHAIN_FRAC
        diag["scale_s_equal_weight"] = chain_comb["s_equal_weight"]
        diag["chain_frac"] = chain_comb["chain_frac"]
    else:
        s = float(np.mean([v["s"] for v in use.values()]))
    diag["scale_rule"] = rule
    diag["scale_rule_requested"] = scale_rule
    diag["scale_per_dancer"] = use
    diag["scale_abs_uncertainty_rel"] = SCALE_ABS_REL
    diag["scale_abs_verified"] = False  # statures barefoot? sole in the mesh? (no independent metric object yet)
    diag["sole_m"] = float(sole_mean)
    diag["soles_m"] = soles
    if len(use) == 2:
        a, b = (v["s"] for v in use.values())
        diag["scale_disagreement"] = float(abs(a - b) / s)
        diag["scale_agreement_ok"] = bool(abs(a - b) / s <= SCALE_AGREE_REL)

    # floor = ankle plane lowered by the ankle height; refine with sparse floor points
    ankle_h = ANKLE_FRAC * np.mean(list(statures.values())) + sole_mean
    d_floor = d - ankle_h / s
    floor_source = "ankle plane - ankle height"
    centre = ank.mean(0)
    if points is not None and len(points) > 50:
        dist = (points @ n - d_floor) * s  # metres above floor
        sel = np.abs(dist) < 0.06
        if use_vertical:
            rad = np.linalg.norm((points - centre) - np.outer((points - centre) @ n, n), axis=1) * s
            sel &= rad < floor_radius_m
        near = points[sel]
        diag["sfm_points_near_floor"] = int(len(near))
        if len(near) > 100:
            n2, d2, inl2 = fit_plane(near, thr=0.015 / s, prior_n=n, max_angle_deg=6)
            off = (d2 - d_floor) * s
            diag["sfm_floor_normal_sfm"] = n2.tolist()
            diag["sfm_floor_plane"] = {"inliers": int(inl2.sum()), "tilt_vs_gravity_deg": _ang(n2, n),
                                       "tilt_vs_ankle_deg": _ang(n2, n_ank), "offset_vs_ankle_m": float(off)}
            if use_vertical:
                hf, inl_f = mode_1d(near @ n, 0.03 / s)
                off_f = (hf - d_floor) * s
                diag["sfm_floor_fixed_normal"] = {"inliers": int(inl_f.sum()), "offset_vs_ankle_m": float(off_f),
                                                  "radius_m": floor_radius_m}
                if inl_f.sum() > 100 and abs(off_f) < 0.08:
                    d_floor = hf
                    floor_source = (f"sparse SfM floor points within {floor_radius_m:g} m of the dancers, normal = "
                                    "gravity (densest 3 cm height band; init: dancers' ankle plane)")
            elif inl2.sum() > 100 and abs(off) < 0.08:
                n, d_floor = n2, d2
                floor_source = "RANSAC plane through sparse SfM floor points (init: dancers' ankle plane)"
        below = (points @ n - d_floor) * s < -0.15
        diag["points_below_floor"] = int(below.sum())  # glossy floor / mirror reflections
        diag["points_total"] = int(len(points))

    # origin: dancers' mean floor position
    ank_low = ank[(ank @ n - d_floor) * s < 0.2]
    o = (ank_low if len(ank_low) else ank).mean(0)
    o = o - (np.dot(o, n) - d_floor) * n
    lowa = (ank @ n - d_floor) * s  # how well y = 0 fits under the dancers (world metres)
    lowa = lowa[lowa < np.percentile(lowa, 30)]
    diag["low_ankle_height_m"] = {"p10": float(np.percentile(lowa, 10)), "median": float(np.median(lowa)),
                                  "expected_ankle_joint": float(ankle_h)}
    if points is not None and len(points) > 50:
        hp = (points @ n - d_floor) * s
        rp = np.linalg.norm((points - o) - np.outer((points - o) @ n, n), axis=1) * s
        prof = {}
        for r0, r1 in ((0, 1.5), (1.5, 3.0), (3.0, 5.0)):
            m = (np.abs(hp) < 0.15) & (rp >= r0) & (rp < r1)
            if m.sum() >= 20:
                prof[f"{r0:g}-{r1:g}m"] = {"n": int(m.sum()), "median_height_m": float(np.median(hp[m])),
                                           "mad_m": float(np.median(np.abs(hp[m] - np.median(hp[m]))))}
        diag["floor_points_height_by_radius"] = prof
    # X axis from camera x_cam's mean right vector
    good = [c for c in sorted(tracks) if tracks[c]["ok"].any() and (up_cams is None or c in up_cams)]
    cam = x_cam if x_cam in good else good[0]
    r = tracks[cam]["R"][tracks[cam]["ok"], 0, :].mean(0)
    x = r - np.dot(r, n) * n
    x /= np.linalg.norm(x)
    z = np.cross(x, n)
    Rw = np.stack([x, n, z])
    t = -s * Rw @ o
    st_txt = f"{stature_source}: " + ", ".join(f"{k} {v:.3f} m" for k, v in statures.items())
    if rule == "chain":
        scale_source = (f"dancers' unfolded keypoint chain (median leg + trunk + shoulder-eye) = {CHAIN_FRAC:.3f} x "
                        f"(stature + shoe sole - {100 * CHAIN_FRAC_SOLE_M:.1f} cm) (CHAIN_FRAC calibrated on "
                        f"larissa_kadu betas), inverse-variance log mean over "
                        f"{len(use)} dancer(s) incl. {100 * SCALE_SYS_REL:.0f} % systematic ({st_txt})")
    else:
        scale_source = f"dancers' eye-above-ankle height vs stature ({st_txt})"
    return {"s": s, "R": Rw.tolist(), "t": t.tolist(), "gravity_source": gravity_source,
            "scale_source": scale_source, "scale_rule": rule,
            "floor_source": floor_source, "x_axis": f"camera {cam} mean right vector (horizontal)",
            "diagnostics": diag}


def rescale(sim3: dict, f: float, reason: str) -> dict:
    """World frame scaled by f about the world origin: X_world' = f X_world, i.e. s' = f s, t' = f t, R kept.

    The origin lies on the floor, so y = 0 stays the floor, the axes and the origin stay put and every camera keeps
    its image projections (cameras/<cam>.npz are SfM-frame and unchanged; world cameras derive from this Sim3).
    Anything already expressed in WORLD metres (smplx/, mamma inputs, physics, timing, export) is stale afterwards.
    The history is appended to diagnostics.rescaled."""
    if not np.isfinite(f) or f <= 0:
        raise ValueError(f"rescale factor must be > 0, got {f}")
    out = json.loads(json.dumps(sim3, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))  # deep copy
    out["s"] = float(sim3["s"]) * f
    out["t"] = (np.asarray(sim3["t"], float) * f).tolist()
    out.setdefault("diagnostics", {}).setdefault("rescaled", []).append(
        {"factor": float(f), "s_before": float(sim3["s"]), "s_after": out["s"], "reason": reason})
    out["scale_source"] = f"{sim3.get('scale_source', '?')}; then x {f:.5f} ({reason})"
    return out


def _ang(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def statures_from_take(cfg: dict) -> tuple[dict[str, float], str]:
    d = cfg.get("dancers", {})
    st = {r: float(d[r]["stature_m"]) for r in ("lead", "follow") if r in d and "stature_m" in d[r]}
    if len(st) == 2:
        return st, "user-supplied statures (takes/<take>.toml)"
    return dict(DEFAULT_STATURE), "assumed statures"


def scale_rule_from_take(cfg: dict) -> str:
    """takes/<take>.toml [cameras] scale_rule ("chain" default | "eye_p90" = the pre-2026-10-06 rule)."""
    rule = str(cfg.get("cameras", {}).get("scale_rule", "chain"))
    if rule not in SCALE_RULES:
        raise ValueError(f"[cameras] scale_rule {rule!r} not in {SCALE_RULES}")
    return rule


def sole_from_take(cfg: dict) -> tuple[dict[str, float], str]:
    """Shoe sole (m) per dancer: takes/<take>.toml [dancers.<role>] shoe_sole_m, else [dancers] shoe_sole_m, else
    SOLE_M ("assumed"). It is what separates the stature (barefoot, crown) from the fitted mesh's crown-to-sole height
    (the mesh sole sits on the shoe sole): set 0 when the statures were measured in the dance shoes."""
    d = cfg.get("dancers", {})
    out, user = {}, True
    for r in ("lead", "follow"):
        v = d.get(r, {}).get("shoe_sole_m", d.get("shoe_sole_m"))
        if v is None:
            v, user = SOLE_M, False
        out[r] = float(v)
    if any(not (0.0 <= v <= 0.1) for v in out.values()):
        raise ValueError(f"shoe_sole_m out of range 0..0.1 m: {out}")
    return out, ("takes/<take>.toml shoe_sole_m" if user else f"assumed {SOLE_M * 100:.1f} cm sneaker sole")


def roles_coverage(work: Path, cams: list[str]) -> dict:
    """Per camera: does dancers/roles.json label lead / follow (dancer_keypoints) or does the size fallback apply?
    {cam: {"source": "roles.json" | "fallback: two largest boxes", "frames": n, "lead": n with a box, "follow": n}}."""
    p = Path(work) / "dancers" / "roles.json"
    roles = json.loads(p.read_text())["cams"] if p.exists() else {}
    out = {}
    for cam in cams:
        r = roles.get(cam)
        if r is None:
            out[cam] = {"source": "fallback: two largest boxes (no dancers/roles.json entry)",
                        "roles_json": p.exists()}
            continue
        fr = r.get("frames", [])
        has = lambda role: sum(1 for f in fr if (f or {}).get(role) and f[role].get("box") is not None)  # noqa: E731
        out[cam] = {"source": "roles.json", "frames": len(fr), "lead": has("lead"), "follow": has("follow")}
    return out


def scale_checks(sim3: dict) -> dict:
    """G2a gates on the metric scale from world_from_sfm diagnostics (cameras/report.json gates_G2a):
    world_scale        the ADOPTED scale is trustworthy: the requested rule ran (no fallback), every main camera's
                       dancers are role-labelled (dancers/roles.json; the chain is per role), and either a configured
                       reference was adopted with the chain within SCALE_MAX_VS_REFERENCE of it, or (no reference)
                       the chain itself passes world_scale_chain.
    world_scale_chain  quality of the chain measurement: dancers agree (<= SCALE_AGREE_REL), per-dancer statistical
                       SE <= SCALE_MAX_REL_SE, first vs second half of the window <= SCALE_MAX_HALVES per dancer."""
    d = sim3.get("diagnostics", {})
    ch = d.get("scale_chain", {})
    req = d.get("scale_rule_requested", sim3.get("scale_rule"))
    rule_ok = sim3.get("scale_rule") == req and "scale_rule_fallback" not in d
    cov = d.get("dancer_roles")
    roles_ok = cov is not None and all(v.get("source") == "roles.json" and v.get("lead", 0) > 0 and v.get("follow", 0) > 0
                                       for v in cov.values())
    halves = {r: v.get("halves_rel_diff") for r, v in ch.items()}
    rel_se = {r: v.get("rel_se") for r, v in ch.items()}
    chain = {"agreement": {"value": d.get("scale_disagreement"), "threshold": SCALE_AGREE_REL,
                           "passed": bool(d.get("scale_disagreement") is not None
                                          and d["scale_disagreement"] <= SCALE_AGREE_REL)},
             "rel_se": {"value": rel_se, "threshold": SCALE_MAX_REL_SE,
                        "passed": len(ch) == 2 and all(x is not None and x <= SCALE_MAX_REL_SE for x in rel_se.values())},
             "halves": {"value": halves, "threshold": SCALE_MAX_HALVES,
                        "passed": len(ch) == 2 and all(x is not None and x <= SCALE_MAX_HALVES for x in halves.values())}}
    chain_ok = all(v["passed"] for v in chain.values())
    ref = d.get("scale_reference")
    base = {"rule_as_requested": {"value": [req, sim3.get("scale_rule")], "passed": bool(rule_ok)},
            "roles_coverage": {"value": cov, "passed": bool(roles_ok),
                               "note": "every main camera needs dancers/roles.json lead+follow boxes: run "
                                       "'python -m dancecap dancers <take>' before 'cameras'"}}
    if ref is not None:
        cv = ref.get("chain_vs_reference")
        base["reference"] = {"value": {k: ref.get(k) for k in ("adopted", "why", "factor", "chain_vs_reference",
                                                               "gravity_vs_inherited_deg", "resid_rel", "n_cams")},
                             "passed": bool(ref.get("adopted")) and cv is not None and abs(cv) <= SCALE_MAX_VS_REFERENCE,
                             "threshold": f"geometry gates (adopt_reference) + |chain / reference - 1| <= "
                                          f"{SCALE_MAX_VS_REFERENCE:.0%}",
                             "note": "chain_vs_reference is a COARSE sanity check: the cameras stage triangulates on the "
                                     "audio clocks before sync.py (phone A/V offsets of tens of ms add limb-motion noise "
                                     "~1 %; re-measure after sync: world_scale chain <take>). It is not circular-free "
                                     "either (CHAIN_FRAC was calibrated on the reference's betas)"}
    else:
        base["chain_is_the_scale"] = {"passed": chain_ok, "note": "no [cameras] scale_reference: the chain IS the scale"}
    val = {"s": sim3.get("s"), "rule": sim3.get("scale_rule"),
           "adopted": "reference" if ref and ref.get("adopted") else sim3.get("scale_rule"),
           "abs_uncertainty_rel": d.get("scale_abs_uncertainty_rel", SCALE_ABS_REL),
           "abs_verified": bool(d.get("scale_abs_verified", False))}
    return {"world_scale": {"passed": all(v["passed"] for v in base.values()), "value": val, "checks": base,
                            "note": "passed = SELF-CONSISTENT, not absolutely verified: the absolute scale rests on "
                                    "the statures (+-1.5 %: barefoot? sole in the mesh?) and no independent metric "
                                    "object has been measured"},
            "world_scale_chain": {"passed": chain_ok, "value": {k: v["value"] for k, v in chain.items()},
                                  "checks": chain}}


def apply(sim3: dict, X: np.ndarray) -> np.ndarray:
    return float(sim3["s"]) * X @ np.asarray(sim3["R"]).T + np.asarray(sim3["t"])
