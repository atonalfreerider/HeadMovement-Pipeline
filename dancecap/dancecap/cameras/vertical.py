"""Architecture-based vertical: vertical vanishing points from wall/door/window/pillar line segments.

The floor plane and the dancers' ankles only see the floor patch around the couple; the building's vertical
edges (pillars, door and window frames, shutter rails, banner edges) are an independent and much denser
gravity reference. Per frame:

  segments  LSD (OpenCV) on the canonical frame, people/phone/caption mask (people_masks/) dilated, short
            segments dropped; only segments whose endpoints both lie ABOVE the horizon of the current camera
            (floor tape and scuffs running away from the camera look vertical in the image but are horizontal)
  lines     each segment and the camera centre span a plane with unit normal n = p1 x p2 (undistorted
            normalised coordinates); a vertical edge's plane contains the vertical direction v: n . v = 0
  solve     v = argmin sum_i w_i (n_i . v)^2 (smallest eigenvector), IRLS with a Tukey weight on the angular
            residual asin|n . v|, starting from the prior vertical (camera up from the current world frame);
            w_i = segment length (long edges are more precise)
  frame     v_cam is oriented up (camera -Y side) and mapped to the SfM frame: u = R^T v_cam

Per camera the frame verticals are fused by a robust mean on the sphere; across cameras by a robust mean of the
per-camera verticals (each camera counted once). Wall planes in the sparse model are a second, 3D check.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .world import undistort_pts


def detect_segments(img: np.ndarray, mask: np.ndarray | None, min_len_frac: float = 0.035,
                    mask_dilate_frac: float = 0.012, work_long: int = 1920) -> np.ndarray:
    """LSD segments (N, 4) x1 y1 x2 y2 in native pixels that do not touch the (dilated) ignore mask."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape
    sc = min(1.0, work_long / max(h, w))
    gs = cv2.resize(g, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else g
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    seg = lsd.detect(gs)[0]
    if seg is None:
        return np.zeros((0, 4))
    seg = seg.reshape(-1, 4).astype(float) / sc
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    seg = seg[L >= min_len_frac * max(h, w)]
    if mask is not None and mask.any() and len(seg):
        r = max(1, int(round(mask_dilate_frac * max(h, w))))
        m = cv2.dilate((mask > 127).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        s = np.linspace(0, 1, 12)
        x = np.clip(np.round(seg[:, None, 0] + s * (seg[:, None, 2] - seg[:, None, 0])), 0, w - 1).astype(int)
        y = np.clip(np.round(seg[:, None, 1] + s * (seg[:, None, 3] - seg[:, None, 1])), 0, h - 1).astype(int)
        seg = seg[~m[y, x].any(1)]
    return seg


def segment_normals(seg: np.ndarray, K: np.ndarray, k1: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpretation-plane normals (N,3), lengths (px) and normalised endpoints (N,2,3) of segments."""
    p = undistort_pts(seg.reshape(-1, 2), K, k1).reshape(-1, 2, 2)
    P = np.concatenate([p, np.ones((len(p), 2, 1))], 2)
    n = np.cross(P[:, 0], P[:, 1])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return n, L, P


def _tukey(r: np.ndarray, c: float) -> np.ndarray:
    return np.where(r < c, (1 - (r / c) ** 2) ** 2, 0.0)


def solve_vertical(n: np.ndarray, w: np.ndarray, v0: np.ndarray, gate_deg: float = 12.0, c_deg: float = 3.0,
                   iters: int = 15, min_inliers: int = 6, n_hyp: int = 80, seed: int = 0) -> dict | None:
    """Robust vertical vanishing direction from interpretation-plane normals n (N,3), weights w (N,).

    Hypotheses v = n_i x n_j from pairs of the longest candidate segments within gate_deg of the prior v0 (no
    dependence on an accurate prior), scored by a Tukey-weighted inlier count; the best is refined by IRLS
    (smallest eigenvector of sum w_i rho_i n_i n_i^T). Returns v (oriented like v0) + statistics, or None."""
    v0 = v0 / np.linalg.norm(v0)
    sel = np.abs(n @ v0) < np.sin(np.radians(gate_deg))
    if sel.sum() < min_inliers:
        return None
    n, w = n[sel], w[sel]
    c = np.sin(np.radians(c_deg))
    top = np.argsort(-w)[:n_hyp]
    cand = [v0]
    for a in range(len(top)):
        V = np.cross(n[top[a]], n[top[a + 1:]])
        nv = np.linalg.norm(V, axis=1)
        V = V[nv > 0.05] / nv[nv > 0.05, None]  # nearly parallel planes give a poorly defined intersection
        V *= np.sign(V @ v0)[:, None]
        cand.extend(V[V @ v0 > np.cos(np.radians(gate_deg))])
    cand = np.asarray(cand)
    if len(cand) > 4000:
        cand = cand[np.random.default_rng(seed).choice(len(cand), 4000, replace=False)]
    score = (_tukey(np.abs(n @ cand.T), c) * w[:, None]).sum(0)
    v = cand[int(np.argmax(score))]
    for _ in range(iters):
        rw = w * _tukey(np.abs(n @ v), c)
        ev, evec = np.linalg.eigh((n * rw[:, None]).T @ n)
        vn = evec[:, 0] * (1 if np.dot(evec[:, 0], v) >= 0 else -1)
        done = np.dot(vn, v) > 1 - 1e-13
        v = vn
        if done:
            break
    r_deg = np.degrees(np.arcsin(np.clip(np.abs(n @ v), 0, 1)))
    inl = r_deg < c_deg / 2
    if inl.sum() < min_inliers:
        return None
    ev = np.linalg.eigvalsh((n[inl] * w[inl, None]).T @ n[inl])
    s2 = float(np.sum(w[inl] * (n[inl] @ v) ** 2) / max(w[inl].sum(), 1e-12))  # weighted residual variance
    sigma = np.degrees(np.sqrt(s2 * w[inl].sum() / max(ev[1], 1e-12) / max(inl.sum() - 2, 1)))
    return {"v": v, "n_inliers": int(inl.sum()), "n_candidates": int(len(n)),
            "resid_deg_median": float(np.median(r_deg[inl])), "sigma_deg": float(sigma),
            "cond": float(ev[1] / max(ev[2], 1e-12)), "inlier_mask_sel": inl, "sel": sel}


def above_horizon(P: np.ndarray, v: np.ndarray, margin_deg: float = 2.0) -> np.ndarray:
    """(N,) segment is not entirely below the horizon of vertical v (camera frame, v up): at least one endpoint
    (normalised, (N,2,3)) is margin_deg above it. The floor is always below the horizon, so floor tape and scuff
    lines (vertical-looking in the image but horizontal in 3D) never pass."""
    d = P / np.linalg.norm(P, axis=2, keepdims=True)
    return ((d @ v) > np.sin(np.radians(margin_deg))).any(1)


def vertical_from_segments(seg: np.ndarray, K: np.ndarray, k1: float, v_prior: np.ndarray, gate_deg: float = 12.0,
                           first_margin_deg: float = -4.0, margin_deg: float = 2.0) -> dict | None:
    """Two passes: the first admits segments down to |first_margin_deg| below the PRIOR horizon (the prior may be
    several degrees off), the second redraws the horizon with the first estimate and keeps only segments reaching
    margin_deg above it. Returns solve_vertical's dict (+ 'used' mask over seg) or None."""
    if len(seg) < 6:
        return None
    n, L, P = segment_normals(seg, K, k1)
    v = v_prior / np.linalg.norm(v_prior)
    res = None
    for margin in (first_margin_deg, margin_deg):
        up = above_horizon(P, v, margin)
        r = solve_vertical(n[up], L[up], v, gate_deg=gate_deg if res is None else 4.0)
        if r is None:
            break
        res, v = r, r["v"]
    if res is None:
        return None
    rr = np.degrees(np.arcsin(np.clip(np.abs(n @ res["v"]), 0, 1)))
    res["used"] = above_horizon(P, res["v"], margin_deg) & (rr < 1.5)
    return res


def frame_vertical(img: np.ndarray, mask: np.ndarray | None, K: np.ndarray, k1: float, v_prior: np.ndarray,
                   gate_deg: float = 12.0, return_segments: bool = False) -> dict | None:
    """Vertical direction in the CAMERA frame (oriented like v_prior, i.e. up) for one frame."""
    seg = detect_segments(img, mask)
    res = vertical_from_segments(seg, K, k1, v_prior, gate_deg)
    if res is None:
        return None
    res["segments_total"] = int(len(seg))
    if return_segments:
        res["seg"], res["seg_used"] = seg, res["used"]
    return res


def sphere_mean(U: np.ndarray, w: np.ndarray | None = None, tukey_deg: float = 2.0, iters: int = 10) -> dict:
    """Robust mean direction of unit vectors U (N,3) (Tukey IRLS on the angle) + spread statistics."""
    w = np.ones(len(U)) if w is None else np.asarray(w, float)
    m = (U * w[:, None]).sum(0)
    m /= np.linalg.norm(m)
    for _ in range(iters):
        a = np.degrees(np.arccos(np.clip(U @ m, -1, 1)))
        c = 4.685 * max(tukey_deg / 2, 1.4826 * np.median(a) if len(a) > 2 else tukey_deg / 2)
        rw = w * np.where(a < c, (1 - (a / c) ** 2) ** 2, 0.0)
        if rw.sum() <= 0:
            break
        m2 = (U * rw[:, None]).sum(0)
        m2 /= np.linalg.norm(m2)
        if np.dot(m2, m) > 1 - 1e-14:
            m = m2
            break
        m = m2
    a = np.degrees(np.arccos(np.clip(U @ m, -1, 1)))
    return {"u": m, "n": int(len(U)), "spread_deg_median": float(np.median(a)) if len(a) else float("nan"),
            "spread_deg_p90": float(np.percentile(a, 90)) if len(a) else float("nan"),
            "sem_deg": float(1.2533 * np.median(a) / np.sqrt(max(len(a), 1))) if len(a) else float("nan")}


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def camera_verticals(work: Path, cam: str, track: dict, up_ref: np.ndarray, step: int = 2,
                     frames: list[int] | None = None, min_inliers: int = 8, n_boot: int = 200) -> dict:
    """Architecture vertical of one camera in its reference frame (main SfM frame, or a sub-model frame).

    track: dict(K (F,3,3), k1 (F,), R (F,3,3) reference->camera, ok (F,) optional); up_ref: prior up in the
    reference frame (only gates candidate segments by +-12 deg and draws the horizon). Every `step`-th ok window
    frame is used. Returns per-frame verticals (frames with >= min_inliers segments) AND a pooled estimate: all
    frames' segment normals mapped into the reference frame (n_ref = R^T n_cam) and solved together - phones
    that see only a few vertical edges per frame (05, 06) still get a precise vertical. sigma of the pooled
    estimate = block bootstrap over frames."""
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    idx = frames if frames is not None else list(range(0, len(meta["frames"]), step))
    up_ref = up_ref / np.linalg.norm(up_ref)
    out_i, v_cam, u_ref, stats = [], [], [], []
    pool_n, pool_w, pool_f = [], [], []
    for i in idx:
        if track.get("ok") is not None and not track["ok"][i]:
            continue
        img = cv2.imread(str(work / "frames" / cam / meta["frames"][i]["file"]))
        mp = work / "people_masks" / cam / (Path(meta["frames"][i]["file"]).stem + ".png")
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
        R = track["R"][i]
        K, k1 = track["K"][i], float(track["k1"][i])
        seg = detect_segments(img, mask)
        if len(seg) == 0:
            continue
        n, L, P = segment_normals(seg, K, k1)
        v_prior = R @ up_ref
        res = frame_vertical(img, mask, K, k1, v_prior) if len(seg) >= 6 else None
        v_h = res["v"] if res is not None else v_prior
        keep = above_horizon(P, v_h) & (np.abs(n @ v_prior) < np.sin(np.radians(12)))
        pool_n.append(n[keep] @ R), pool_w.append(L[keep]), pool_f.append(np.full(int(keep.sum()), i))
        if res is None or res["n_inliers"] < min_inliers:
            continue
        out_i.append(i), v_cam.append(res["v"]), u_ref.append(R.T @ res["v"])
        stats.append((res["n_inliers"], res["resid_deg_median"], res["sigma_deg"]))
    st = np.asarray(stats).reshape(-1, 3)
    out = {"frames": np.asarray(out_i, int), "v_cam": np.asarray(v_cam).reshape(-1, 3),
           "u_ref": np.asarray(u_ref).reshape(-1, 3), "n_inliers": st[:, 0], "resid_deg": st[:, 1],
           "sigma_deg": st[:, 2], "pooled": None}
    if pool_n:
        N, W, Fi = np.concatenate(pool_n), np.concatenate(pool_w), np.concatenate(pool_f)
        res = solve_vertical(N, W, up_ref, n_hyp=120)
        if res is not None:
            rng = np.random.default_rng(0)
            fr = np.unique(Fi)
            boots = []
            for _ in range(n_boot if len(fr) >= 5 else 0):
                pick = rng.choice(fr, len(fr))
                cnt = np.bincount(np.searchsorted(fr, pick), minlength=len(fr))
                wb = W * cnt[np.searchsorted(fr, Fi)]
                rb = solve_vertical(N, wb, res["v"], gate_deg=6.0, n_hyp=0)
                if rb is not None:
                    boots.append(rb["v"])
            sig = float(np.degrees(np.sqrt(np.mean(np.arccos(np.clip(np.asarray(boots) @ res["v"], -1, 1)) ** 2))
                                   / np.sqrt(2))) if len(boots) > 10 else float("nan")  # per-axis std
            used = np.zeros(len(N), bool)
            used[np.nonzero(res["sel"])[0][res["inlier_mask_sel"]]] = True
            # frame-to-frame scatter of the single-frame verticals: an honest (if pessimistic) error of the pooled
            # one; the bootstrap ignores biases shared by all segments of a frame (e.g. pitch for a narrow FOV)
            sig_fr = float("nan")
            if len(out["u_ref"]) >= 5:
                a = np.degrees(np.arccos(np.clip(out["u_ref"] @ res["v"], -1, 1)))
                sig_fr = float(np.median(a) / 1.177 / np.sqrt(len(a)))
            out["pooled"] = {"u": res["v"], "n_segments": int(res["n_inliers"]), "n_frames": int(len(np.unique(Fi[used]))),
                             "resid_deg_median": res["resid_deg_median"], "sigma_deg": sig,
                             "sigma_frames_deg": sig_fr, "frames_total": int(len(fr))}
    return out


def fuse(per_cam: dict[str, dict], cams: list[str]) -> dict:
    """Robust mean over cameras of the pooled per-camera verticals (one vote per camera, reference = SfM frame)."""
    cam_u, info = {}, {}
    for cam in cams:
        d = per_cam.get(cam)
        if d is None or d.get("pooled") is None:
            continue
        cam_u[cam] = d["pooled"]["u"]
        info[cam] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d["pooled"].items()}
        if len(d["u_ref"]) >= 3:
            m = sphere_mean(d["u_ref"])
            info[cam]["per_frame"] = {"n": m["n"], "vs_pooled_deg": angle_deg(m["u"], d["pooled"]["u"]),
                                      "spread_deg_median": m["spread_deg_median"]}
    if not cam_u:
        raise ValueError("no camera with enough vertical vanishing points")
    # one vote per camera, weighted by its uncertainty (narrow-FOV / few-edge cameras such as a zoomed-in phone
    # count less): sigma = max(bootstrap, frame scatter, 0.1 deg)
    sig = np.array([max(info[c].get("sigma_deg") or 0.0, np.nan_to_num(info[c].get("sigma_frames_deg"), nan=0.0), 0.1)
                    for c in cam_u])
    allm = sphere_mean(np.array(list(cam_u.values())), w=1.0 / sig ** 2, tukey_deg=1.5)
    for cam, u in cam_u.items():
        info[cam]["vs_fused_deg"] = angle_deg(u, allm["u"])
    for cam, sg in zip(cam_u, sig):
        info[cam]["fusion_weight_sigma_deg"] = float(sg)
    return {"u": allm["u"], "cams": info, "spread_deg_median": allm["spread_deg_median"],
            "max_cam_deg": max(info[c]["vs_fused_deg"] for c in cam_u), "n_cams": len(cam_u)}


def wall_planes(points: np.ndarray, up: np.ndarray, scale: float, n_planes: int = 4, thr: float = 0.02,
                min_inliers: int = 300, seed: int = 0) -> list[dict]:
    """Largest near-vertical planes (|n . up| < sin 20 deg) in a point cloud; thr in world-ish metres x scale."""
    from .world import fit_plane
    out = []
    P = points.copy()
    rng = np.random.default_rng(seed)
    for k in range(n_planes):
        if len(P) < min_inliers:
            break
        best = None
        for _ in range(3000):
            s = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0])
            if np.linalg.norm(n) < 1e-12:
                continue
            n /= np.linalg.norm(n)
            if abs(n @ up) > np.sin(np.radians(20)):
                continue
            inl = np.abs((P - s[0]) @ n) < thr / scale
            if best is None or inl.sum() > best[1].sum():
                best = (n, inl)
        if best is None or best[1].sum() < min_inliers:
            break
        n, d, inl = fit_plane(P[best[1]], thr / scale, prior_n=best[0], max_angle_deg=10, iters=200, seed=k)
        Q = P[best[1]][inl]
        out.append({"n": n, "inliers": int(len(Q)), "tilt_from_vertical_deg": abs(90 - angle_deg(n, up)),
                    "centroid": Q.mean(0)})
        keep = np.ones(len(P), bool)
        keep[np.nonzero(best[1])[0][inl]] = False
        P = P[keep]
    return out


def _plumb(p_px: np.ndarray, v: np.ndarray, K: np.ndarray, k1: float, length: float = 1.0, n: int = 24):
    """Image polyline of the 3D line through the ray of pixel p (at depth 1) along direction v (camera frame)."""
    x = undistort_pts(p_px[None], K, k1)[0]
    X = np.array([x[0], x[1], 1.0])[None] + np.linspace(-length, length, n)[:, None] * v[None]
    z = X[:, 2]
    ok = z > 0.05
    xn, yn = X[:, 0] / np.where(ok, z, 1), X[:, 1] / np.where(ok, z, 1)
    d = 1 + k1 * (xn * xn + yn * yn)
    return np.stack([K[0, 0] * xn * d + K[0, 2], K[1, 1] * yn * d + K[1, 2]], 1)[ok]


def draw_check(img: np.ndarray, seg: np.ndarray, used: np.ndarray, K: np.ndarray, k1: float,
               lines: dict[str, tuple[np.ndarray, tuple[int, int, int]]], mask: np.ndarray | None = None,
               n_anchor: int = 6) -> np.ndarray:
    """Review overlay (no text): segments used for the vertical (yellow, others not drawn), people darkened, and
    for each direction in `lines` {name: (v_cam, BGR)} a full-height plumb line through the midpoints of the
    longest used segments (spread across the image): a correct vertical runs ALONG the yellow edges."""
    im = img.copy()
    h, w = im.shape[:2]
    if mask is not None:
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 127
        im[m] = (im[m] * 0.45).astype(np.uint8)
    th = max(1, round(max(h, w) / 600))
    for s in seg[used]:
        cv2.line(im, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 230, 255), 2 * th, cv2.LINE_AA)
    U = seg[used]
    anchors = []
    if len(U):
        L = np.hypot(U[:, 2] - U[:, 0], U[:, 3] - U[:, 1])
        for j in np.argsort(-L):
            mid = (U[j, :2] + U[j, 2:]) / 2
            if all(abs(mid[0] - a[0]) > w / (n_anchor + 2) for a in anchors):
                anchors.append(mid)
            if len(anchors) == n_anchor:
                break
    for ax in np.linspace(0.1, 0.9, n_anchor):  # fill up with a regular grid at the upper third
        if len(anchors) >= n_anchor:
            break
        if all(abs(ax * w - a[0]) > w / (n_anchor + 2) for a in anchors):
            anchors.append(np.array([ax * w, 0.3 * h]))
    span = 1.5 * max(h, w) / K[0, 0]
    for k, (name, (v, col)) in enumerate(lines.items()):
        for a in anchors:
            pl = _plumb(np.asarray(a, float), v, K, k1, length=span, n=40)
            pl = pl[(np.abs(pl) < 1e5).all(1)]
            if len(pl) > 1:
                cv2.polylines(im, [np.round(pl * 16).astype(np.int32)], False, col, th, cv2.LINE_AA, shift=4)
    return im


def pooled_normals(work: Path, cam: str, track: dict, frames: list[int], v_ref: np.ndarray,
                   exclude_vertical_deg: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Interpretation-plane normals in the reference frame (R^T n) + lengths of NON-vertical segments that are not
    entirely below the horizon (walls, ceiling edges, banners, shutter slats), pooled over `frames`."""
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    Ns, Ws = [], []
    for i in frames:
        img = cv2.imread(str(work / "frames" / cam / meta["frames"][i]["file"]))
        mp = work / "people_masks" / cam / (Path(meta["frames"][i]["file"]).stem + ".png")
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
        seg = detect_segments(img, mask)
        if not len(seg):
            continue
        n, L, P = segment_normals(seg, track["K"][i], float(track["k1"][i]))
        R = track["R"][i]
        v = R @ v_ref
        keep = above_horizon(P, v, 3.0) & (np.abs(n @ v) > np.sin(np.radians(exclude_vertical_deg)))
        Ns.append(n[keep] @ R), Ws.append(L[keep])
    return (np.concatenate(Ns) if Ns else np.zeros((0, 3))), (np.concatenate(Ws) if Ws else np.zeros(0))


def horizontal_directions(N: np.ndarray, W: np.ndarray, u: np.ndarray, n_dirs: int = 3, max_elev_deg: float = 15.0,
                          c_deg: float = 1.5, n_pairs: int = 30000, seed: int = 0) -> list[dict]:
    """Dominant 3D line directions d (vanishing points) within max_elev_deg of the plane perpendicular to u, from
    pooled interpretation-plane normals N (reference frame). For a metric (undistorted) reconstruction the
    building's horizontal edges are perpendicular to its vertical: elevation ~ 0."""
    rng = np.random.default_rng(seed)
    c = np.sin(np.radians(c_deg))
    out = []
    alive = np.ones(len(N), bool)
    for _ in range(n_dirs):
        idx = np.nonzero(alive)[0]
        if len(idx) < 20:
            break
        p = rng.choice(idx, (n_pairs, 2), p=W[idx] / W[idx].sum())
        D = np.cross(N[p[:, 0]], N[p[:, 1]])
        nd = np.linalg.norm(D, axis=1)
        D = D[nd > 0.1] / nd[nd > 0.1, None]
        D = D[np.abs(D @ u) < np.sin(np.radians(max_elev_deg))]
        if not len(D):
            break
        sub = D[rng.choice(len(D), min(3000, len(D)), replace=False)]
        score = (_tukey(np.abs(N[idx] @ sub.T), c) * W[idx, None]).sum(0)
        d = sub[int(np.argmax(score))]
        for _ in range(15):
            rw = W[idx] * _tukey(np.abs(N[idx] @ d), c)
            ev, evec = np.linalg.eigh((N[idx] * rw[:, None]).T @ N[idx])
            d = evec[:, 0] * (1 if evec[:, 0] @ d >= 0 else -1)
        r = np.abs(N @ d)
        inl = alive & (r < c / 2)
        out.append({"d": d, "n_segments": int(inl.sum()), "elev_vs_u_deg": float(np.degrees(np.arcsin(abs(d @ u))))})
        alive &= r >= c
    return out
