"""Real shoe sizes from the video: outer length, width and toe-box height of each dancer's sneakers, in the take's world.

    python -m dancecap.shoe_size <take>                       -> work/<take>/avatar/shoe_size.json (+ review crops)
    python -m dancecap.shoe_size <take> --detect [--step 4]   first run RTMW-x whole-body on every step-th frame of
                                                              every camera (when the take has no hands/detections.npz)
    python -m dancecap.shoe_size <take> --review <dir>        review crops into <dir> (default work/<take>/review/demo)

The numbers feed takes/<take>.toml [dancers.<role>.shape_edits] shoe_length_m (dancecap.shape_edits shrinks the rest
feet so the Unity sneaker, built around the foot, comes out that long) and [dancers.<role>] shoe_geometry.

Method (CPU, ~1 min on 8 cameras x 6 s; nothing is fitted - the SMPL-X feet are not used):
  1. Keypoints (--keypoints auto: the first available): RTMW-x 384x288 whole-body foot points (heel, big toe, small
     toe) + ankles, top-down on each dancer's body box (refine_hands' hands/detections.npz body jobs, or
     avatar/shoe_rtmw.npz from --detect, which runs dancecap.shoe_size worker in .venv-face like dancecap.hand_worker;
     CPU-only there: ~1 s per box); else MAMMA's dense 2D landmarks (mamma/ma_2d_native/<cam>.npz, every frame of every
     camera): the SMPL-X foot vertices among its 512 (mamma verts_512 one-hot rows) at the heel bottom, the toe tip, the
     lateral forefoot and the back of the ankle (all of them: its visibilities are occlusion probabilities, so the
     triangulation's worst-view rejection does the filtering). MAMMA's points lie on the BODY (the foot inside
     the shoe), RTMW's inside the shoe outline: either only gives the axis and the planted frames here.
  2. Triangulation on the capture grid: undistorted (SIMPLE_RADIAL) keypoints, linear between consecutive frames of
     each camera on its sync-corrected clock, weighted DLT over >= 3 cameras with worst-view rejection (> max_px).
  3. Planted feet only: heel < 7 cm and toe < 6 cm above the floor, both moved < 8 mm since the previous grid frame.
     The keypoints sit INSIDE the shoe (the heel point ~1-2 cm in front of the heel counter, the toe point behind the
     toe cap): their distance is a lower bound, reported but not used.
  4. Silhouettes (the shoe itself): along the triangulated foot axis on the floor, the dancer's own SAM2 mask is
     sampled every 2 mm at 1.2 / 2.2 / 3.2 cm above the floor in views seeing the foot from the side (|axis . ray| <
     0.5); the longest mask run through the foot's middle is the outer length (heel counter .. toe tip). Width: across
     the axis at 68 % of the length in views along the foot (|axis . ray| > 0.85). Toe-box height: the mask's top
     above the floor at 75 / 85 / 92 % of the length in side views. Views where the partner's mask or the dancer's
     other foot touches the samples are skipped. Hi-res views (>= 250 px/m at the foot) are preferred when there are
     at least 8; mask edges are ~1 px soft, so the numbers carry ~+-3 mm (hi-res) to +-6 mm (480 px views).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE

ROLES = ("lead", "follow")
PIDS = {"lead": 1, "follow": 2}
KP = {"L": (19, 17, 18, 15), "R": (22, 20, 21, 16)}  # COCO-WholeBody (person's side): heel, big toe, small toe, ankle
RTMW_ONNX = PACKAGE_ROOT / "weights" / "face" / "rtmw_x384" / "end2end.onnx"
# MAMMA 512-landmark ids (mamma data/body_models/downsampled_verts/verts_512.pkl rows = one SMPL-X vertex each;
# landmarks/lib/datasets/_smplx_512_body_parts.json left_feet / right_feet) in the KP slot order: heel bottom (x 0.9 cm
# from the heel, 1.7 cm up on the template), toe tip (x = L, 0.5 cm up), lateral forefoot, back of the ankle (9.8 cm up)
MA2D_FOOT = {"L": (221, 51, 2, 6), "R": (477, 307, 258, 262)}
FACE_PY = WORKSPACE / ".venv-face" / "Scripts" / "python.exe"
HEIGHT_STATIONS = (0.75, 0.85, 0.92)
WIDTH_STATION = 0.68
HIRES_PPM = 250.0


# ------------------------------------------------------------------------------------------------ keypoints

def load_ma2d(work: Path, min_vis: float = 0.0) -> dict:
    """MAMMA's native 2D landmarks -> the load_keypoints format (score 5 where visibility >= min_vis, else 0).
    MAMMA's visibilities are self-occlusion probabilities (median 0.13 over all landmarks), not confidences: every
    landmark is used and the triangulation's worst-view rejection drops the bad ones."""
    out = {}
    ids = list(MA2D_FOOT["L"]) + list(MA2D_FOOT["R"])
    for p in sorted((work / "mamma" / "ma_2d_native").glob("[0-9][0-9].npz")):
        with np.load(p) as z:
            L, V = z["landmarks"][:, :, ids, :2].astype(float), z["visibilities"][:, :, ids].astype(float)
        for n, r in enumerate(ROLES):  # person axis = mamma_dyn.PIDS (1 lead, 2 follow)
            sc = np.where((V[:, n] >= min_vis) & np.isfinite(L[:, n]).all(-1), 5.0, 0.0)
            ok = sc.sum(1) > 0
            out[(r, p.stem)] = (np.nonzero(ok)[0], L[ok, n], sc[ok])
    if not out:
        raise FileNotFoundError(f"{work}/mamma/ma_2d_native/*.npz missing")
    return out


def load_keypoints(work: Path, source: str = "auto") -> tuple[dict, str]:
    """{(role, cam): (i (n,), kp (n,8,2) distorted px, score (n,8))} of the foot + ankle points, and the source."""
    if source == "ma2d":
        return load_ma2d(work), "mamma/ma_2d_native"
    for p in (work / "hands" / "detections.npz", work / "avatar" / "shoe_rtmw.npz"):
        if p.exists():
            with np.load(p) as z:
                if "body_cam" not in z.files or not len(z["body_cam"]):
                    continue
                cam, ii, role = z["body_cam"], z["body_i"], z["body_role"]
                ids = list(KP["L"]) + list(KP["R"])
                kp, sc = z["rtmw_kp"][:, ids].astype(float), z["rtmw_score"][:, ids].astype(float)
            out = {}
            for r in ROLES:
                for c in sorted(set(cam.tolist())):
                    sel = (cam == c) & (role == r)
                    if sel.any():
                        out[(r, str(c))] = (ii[sel].astype(int), kp[sel], sc[sel])
            return out, str(p.relative_to(work))
    if source == "auto" and (work / "mamma" / "ma_2d_native").exists():
        return load_ma2d(work), "mamma/ma_2d_native"
    raise FileNotFoundError(f"no RTMW body keypoints / ma_2d landmarks for {work.name}: run with --detect")


def grid_times(work: Path) -> np.ndarray:
    p = work / "smplx" / "lead.npz"
    if p.exists():
        with np.load(p) as z:
            return np.asarray(z["t"], float)
    prep = work / "mamma" / "prepare.json"
    if prep.exists():
        return np.asarray(json.loads(prep.read_text())["grid"]["t"], float)
    raise FileNotFoundError(f"{work.name}: no smplx/lead.npz or mamma/prepare.json for the time grid")


def _dlt(Ps: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    from .refine_hands import _dlt as dlt
    return dlt(Ps, x, w)


def triangulate(work: Path, kps: dict, grid: np.ndarray, min_score: float = 3.0, max_px: float = 6.0,
                min_views: int = 3) -> tuple[dict, dict, dict]:
    """-> ({(role, side): X (T,4,3) world [heel, big toe, small toe, ankle]}, {(role, side): views (T,)}, tracks)."""
    from . import mamma_dyn as md
    from .refine_hands import _proj, grid_cameras
    cams = sorted({c for _, c in kps})
    tracks = md.load_tracks(work, cams)
    gc = grid_cameras(work, tracks, grid)
    T = len(grid)
    X, NV = {}, {}
    for r in ROLES:
        for s, cols in (("L", slice(0, 4)), ("R", slice(4, 8))):
            obs = {}
            for c in cams:
                if (r, c) not in kps:
                    continue
                ii, kp, sc = kps[(r, c)]
                tr = tracks[c]
                F = len(tr.t)
                K, W = np.zeros((F, 4, 2)), np.zeros((F, 4))
                for n, i in enumerate(ii):
                    if i >= F:
                        continue
                    K[i] = md.undistort_simple_radial(kp[n, cols], tr.K[i], float(tr.k1[i]))
                    W[i] = (sc[n, cols] >= min_score).astype(float)
                i0, i1, a, d0, d1, gap = md._brackets(tr.t, grid, tr.frame_dt)
                two = (W[i0] > 0) & (W[i1] > 0) & (gap[:, None] <= 1.5)
                lin = (1 - a)[:, None, None] * K[i0] + a[:, None, None] * K[i1]
                obs[c] = (np.where(two[..., None], lin, 0.0), two.astype(float))
            Xs, nv = np.full((T, 4, 3), np.nan), np.zeros(T, int)
            for k in range(T):
                vs = [c for c in obs if gc[c]["ok"][k] and obs[c][1][k].sum() >= 3]
                if len(vs) < min_views:
                    continue
                Ps = np.asarray([gc[c]["K"][k] @ np.concatenate([gc[c]["R"][k], gc[c]["t"][k][:, None]], 1) for c in vs])
                x = np.stack([obs[c][0][k] for c in vs])
                w = np.stack([obs[c][1][k] for c in vs])
                inl = np.ones(len(vs), bool)
                Xk = None
                while inl.sum() >= min_views:
                    Xk = _dlt(Ps[inl], x[inl], w[inl])
                    if not np.isfinite(Xk).all():
                        Xk = None
                        break
                    e = np.array([np.median(np.linalg.norm(_proj(Ps[v], Xk) - x[v], axis=1)) if inl[v] else -1.0
                                  for v in range(len(vs))])
                    worst = int(np.argmax(e))
                    if e[worst] <= max_px:
                        break
                    inl[worst] = False
                    Xk = None
                if Xk is not None and inl.sum() >= min_views:
                    Xs[k], nv[k] = Xk, int(inl.sum())
            X[(r, s)], NV[(r, s)] = Xs, nv
    return X, NV, tracks


def smooth_track(X: np.ndarray, half: int = 2) -> np.ndarray:
    """Centred moving median over 2 half + 1 grid frames (NaN-aware; NaN where fewer than half + 1 are finite): the
    per-frame triangulations jitter by ~1 cm (MAMMA landmarks), a planted foot does not."""
    out = np.full_like(X, np.nan)
    for k in range(len(X)):
        w = X[max(0, k - half):k + half + 1]
        ok = np.isfinite(w).all(axis=tuple(range(1, w.ndim)))
        if ok.sum() >= half + 1 and np.isfinite(X[k]).all():
            out[k] = np.median(w[ok], axis=0)
    return out


def planted(X: np.ndarray, heel_max: float = 0.07, toe_max: float = 0.06, step_max: float = 0.006) -> np.ndarray:
    """(T,) grid frames where the foot is flat on the floor and still: on the smoothed track (smooth_track), heel / toe
    low, both moved < step_max since the previous grid frame AND until the next one (a moving foot smears in the video
    and its mask runs long)."""
    X = smooth_track(X)
    ok = np.isfinite(X).all((1, 2)) & (X[:, 0, 1] < heel_max) & (X[:, 1, 1] < toe_max)
    step = np.full(len(X) + 1, np.inf)
    step[1:-1] = np.linalg.norm(X[1:, :2] - X[:-1, :2], axis=2).max(1)
    mv = np.fmax(step[:-1], step[1:])
    return ok & (mv < step_max)


# ------------------------------------------------------------------------------------------------ silhouettes

class View:
    """One camera frame: projection with SIMPLE_RADIAL distortion, the two dancers' masks."""

    def __init__(self, work: Path, tr, i: int):
        import cv2
        self.R, self.t, self.K, self.k1 = tr.R[i], tr.tvec[i], tr.K[i], float(tr.k1[i])
        self.C = -self.R.T @ self.t
        self.cam, self.i = tr.name, i
        md_ = work / "dancers" / tr.name / "masks"
        self.mask = {r: cv2.imread(str(md_ / f"mask_{i:04d}_{PIDS[r]:02d}.png"), cv2.IMREAD_GRAYSCALE) for r in ROLES}

    def proj(self, P: np.ndarray) -> np.ndarray:
        Xc = np.atleast_2d(P) @ self.R.T + self.t
        x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        d = 1 + self.k1 * (x * x + y * y)
        return np.stack([self.K[0, 0] * x * d + self.K[0, 2], self.K[1, 1] * y * d + self.K[1, 2]], 1)

    def sample(self, role: str, P: np.ndarray) -> np.ndarray | None:
        m = self.mask[role]
        if m is None:
            return None
        ij = np.round(self.proj(P)).astype(int)
        H, W = m.shape
        if not ((ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)).all():
            return None
        return m[ij[:, 1], ij[:, 0]] > 127

    def ppm(self, M: np.ndarray) -> float:
        p = self.proj(np.stack([M, M + [0.0, 0.1, 0.0]]))
        return float(np.linalg.norm(p[1] - p[0]) * 10.0)


def _run(on: np.ndarray, c0: int, slack: int = 5) -> tuple[int, int] | None:
    """The mask run through sample c0 (or within +-slack of it); None if it touches either end."""
    if not on[max(0, c0 - slack):c0 + slack].any():
        return None
    if not on[c0]:
        cand = np.nonzero(on[max(0, c0 - slack):c0 + slack])[0]
        c0 = int(max(0, c0 - slack) + cand[np.argmin(np.abs(cand + max(0, c0 - slack) - c0))])
    a = c0
    while a > 0 and on[a - 1]:
        a -= 1
    b = c0
    while b < len(on) - 1 and on[b + 1]:
        b += 1
    if a == 0 or b == len(on) - 1:
        return None
    return a, b


def silhouettes(work: Path, X: dict, tracks: dict, grid: np.ndarray, step: float = 0.002) -> list[dict]:
    """Per planted (role, side, grid frame) and usable view: length / width / toe-heights records (metres)."""
    recs = []
    cache: dict = {}

    def view(c, i):
        if (c, i) not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[(c, i)] = View(work, tracks[c], i)
        return cache[(c, i)]

    Xsm = {key: smooth_track(x) for key, x in X.items()}
    for (r, s), Xs in Xsm.items():
        other = Xsm[(r, "R" if s == "L" else "L")]
        partner = [Xsm[(q, ss)] for q in ROLES if q != r for ss in ("L", "R")]
        for k in np.nonzero(planted(X[(r, s)]))[0]:
            heel, toe = Xs[k, 0], Xs[k, 1]
            d = toe - heel
            d[1] = 0.0
            d /= np.linalg.norm(d)
            n = np.cross([0.0, 1.0, 0.0], d)
            M = 0.5 * (heel + toe)
            M[1] = 0.0
            for c, tr in tracks.items():
                i = int(np.argmin(np.abs(tr.t - grid[k])))
                if abs(tr.t[i] - grid[k]) > 0.5 * tr.frame_dt or not tr.ok[i]:
                    continue
                v = view(c, i)
                if v.mask[r] is None:
                    continue
                ray = M - v.C
                ray /= np.linalg.norm(ray)
                cosang = abs(float(ray @ d))
                ppm = v.ppm(M)

                def clear(P):
                    """no partner pixels on the samples, the other foot's keypoints off them"""
                    q = v.sample([x for x in ROLES if x != r][0], P)
                    if q is None or q.any():
                        return False
                    p2 = v.proj(P)
                    lo, hi = p2.min(0) - 6, p2.max(0) + 6
                    for Y in [other] + partner:
                        if np.isfinite(Y[k, :3]).all():
                            pp = v.proj(Y[k, :3])
                            if ((pp >= lo) & (pp <= hi)).all(1).any():
                                return False
                    return True

                base = {"role": r, "side": s, "k": int(k), "t": float(grid[k]), "cam": c, "i": i,
                        "ppm": round(ppm, 1), "kp_length": float(np.linalg.norm(toe - heel))}
                if cosang < 0.5:  # side view: length, then toe-box heights on that length
                    u = np.arange(-0.25, 0.25, step)
                    best = None
                    for h in (0.012, 0.022, 0.032):
                        P = M[None] + u[:, None] * d[None] + [0.0, h, 0.0]
                        if not clear(P):
                            best = None
                            break
                        on = v.sample(r, P)
                        run = None if on is None else _run(on, len(u) // 2)
                        if run is not None and (best is None or run[1] - run[0] > best[1] - best[0]):
                            best = run
                    if best is not None and not 1.0 <= (u[best[1]] - u[best[0]] + step) / base["kp_length"] <= 1.45:
                        best = None  # the keypoints sit inside the shoe: outer length 1.0-1.45 x their distance
                    if best is not None:
                        L = float(u[best[1]] - u[best[0]] + step)
                        x0 = M + u[best[0]] * d
                        rec = {**base, "kind": "length", "value": L, "cos": round(cosang, 3),
                               "ends": [(M + u[best[0]] * d).tolist(), (M + u[best[1]] * d).tolist()]}
                        recs.append(rec)
                        hh = np.arange(0.004, 0.16, step)
                        for f in HEIGHT_STATIONS:
                            P = (x0 + f * L * d)[None] + hh[:, None] * np.array([0.0, 1.0, 0.0])[None]
                            on = v.sample(r, P)
                            if on is None or not on[:3].any():
                                continue
                            b = int(np.argmax(on))
                            while b < len(hh) - 1 and on[b + 1]:
                                b += 1
                            if b == len(hh) - 1:
                                continue
                            q = v.sample([x for x in ROLES if x != r][0], P[:b + 3])
                            if q is None or q.any():
                                continue
                            recs.append({**base, "kind": f"height@{f:.2f}", "value": float(hh[b] + step)})
                elif cosang > 0.85:  # along the foot: width at the ball
                    u = np.arange(-0.12, 0.12, step)
                    ws = []
                    for h in (0.015, 0.025):
                        P = (M + (WIDTH_STATION - 0.5) * base["kp_length"] * d)[None] + u[:, None] * n[None] + [0, h, 0]
                        if not clear(P):
                            ws = []
                            break
                        on = v.sample(r, P)
                        run = None if on is None else _run(on, len(u) // 2, slack=3)
                        if run is not None:
                            ws.append(float(u[run[1]] - u[run[0]] + step))
                    if ws and 0.07 <= max(ws) <= 0.16:
                        recs.append({**base, "kind": "width", "value": max(ws), "cos": round(cosang, 3)})
    return recs


def summarise(recs: list[dict], X: dict, min_hires: int = 8) -> dict:
    out = {}
    for r in ROLES:
        e: dict = {}
        for kind in ["length", "width"] + [f"height@{f:.2f}" for f in HEIGHT_STATIONS]:
            rr = [x for x in recs if x["role"] == r and x["kind"] == kind]
            hi = [x for x in rr if x["ppm"] >= HIRES_PPM]
            use = hi if len(hi) >= min_hires else rr
            if not use:
                continue
            vals = np.array([x["value"] for x in use])
            e[kind] = {"m": round(float(np.median(vals)), 4), "p25": round(float(np.percentile(vals, 25)), 4),
                       "p75": round(float(np.percentile(vals, 75)), 4), "n": len(use), "n_all": len(rr),
                       "views": "hi-res" if use is hi else "all",
                       "per_side": {s: round(float(np.median([x["value"] for x in use if x["side"] == s])), 4)
                                    for s in ("L", "R") if any(x["side"] == s for x in use)},
                       "cams": sorted({x["cam"] for x in use})}
        kl = [float(np.linalg.norm(X[(r, s)][k, 1] - X[(r, s)][k, 0])) for s in ("L", "R")
              for k in np.nonzero(planted(X[(r, s)]))[0]]
        if kl:
            e["keypoint_heel_toe_m"] = {"m": round(float(np.median(kl)), 4), "n": len(kl),
                                        "note": "heel -> toe keypoints on planted feet: the points sit inside the shoe "
                                                "(lower bound, not used)"}
        out[r] = e
    return out


# ------------------------------------------------------------------------------------------------ review

def review_sheet(work: Path, recs: list[dict], X: dict, tracks: dict, path: Path, per_role: int = 4, tile: int = 260
                 ) -> Path:
    """Crops of the best side views per dancer: keypoints (red heel, green big toe, cyan small toe) and the measured
    silhouette length (magenta, 2 cm above the floor)."""
    import cv2
    rows = []
    for r in ROLES:
        rr = sorted([x for x in recs if x["role"] == r and x["kind"] == "length"], key=lambda x: -x["ppm"])
        tiles, seen = [], set()
        for x in rr:
            key = (x["cam"], x["side"])
            if key in seen:
                continue
            seen.add(key)
            tr = tracks[x["cam"]]
            img = cv2.imread(str(work / "frames" / x["cam"] / f"{x['i']:06d}.jpg"))
            if img is None:
                continue
            v = View(work, tr, x["i"])
            ends = v.proj(np.asarray(x["ends"]) + [0.0, 0.02, 0.0])
            kp = v.proj(X[(r, x["side"])][x["k"], :3])
            cxy = ends.mean(0)
            sc = 0.62 * tile / max(8.0, float(np.linalg.norm(ends[1] - ends[0])))
            A = np.array([[sc, 0, -cxy[0] * sc + tile / 2], [0, sc, -cxy[1] * sc + tile / 2]])
            crop = cv2.warpAffine(img, A, (tile, tile), flags=cv2.INTER_CUBIC)
            q = lambda p: tuple(int(round(z)) for z in (A[:, :2] @ p + A[:, 2]))  # noqa: E731
            cv2.line(crop, q(ends[0]), q(ends[1]), (255, 0, 255), 2, cv2.LINE_AA)
            for p, col in zip(kp, [(0, 0, 255), (0, 255, 0), (255, 255, 0)]):
                cv2.circle(crop, q(p), 4, col, 2, cv2.LINE_AA)
            cv2.rectangle(crop, (0, tile - 20), (tile, tile), (0, 0, 0), -1)
            cv2.putText(crop, f"{r} {x['side']} cam{x['cam']} sil {100 * x['value']:.1f} kp {100 * x['kp_length']:.1f} cm",
                        (4, tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
            if len(tiles) >= per_role:
                break
        tiles += [np.zeros((tile, tile, 3), np.uint8)] * (per_role - len(tiles))
        rows.append(np.hstack(tiles))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


# ------------------------------------------------------------------------------------------------ detection (.venv-face)

def detect_jobs(work: Path, step: int = 4, pad: float = 0.12) -> list[dict]:
    """Body boxes of both dancers from their masks on every step-th frame of every camera."""
    import cv2
    jobs = []
    for cam_dir in sorted((work / "dancers").iterdir()):
        md_ = cam_dir / "masks"
        if not md_.exists() or not (work / "frames" / cam_dir.name / "frames.json").exists():
            continue
        n = len(json.loads((work / "frames" / cam_dir.name / "frames.json").read_text())["frames"])
        for i in range(0, n, step):
            for r in ROLES:
                m = cv2.imread(str(md_ / f"mask_{i:04d}_{PIDS[r]:02d}.png"), cv2.IMREAD_GRAYSCALE)
                if m is None or not m.any():
                    continue
                x, y, w, h = cv2.boundingRect((m > 127).astype(np.uint8))
                jobs.append({"cam": cam_dir.name, "i": i, "role": r,
                             "box": [x - pad * w, y - pad * h, x + w * (1 + pad), y + h * (1 + pad)]})
    return jobs


def worker(jobs_path: Path, out_path: Path, device: str = "cuda") -> None:
    """RTMW-x on the jobs (runs in .venv-face: rtmlib + onnxruntime). Output keys like hand_worker's body jobs."""
    import cv2
    from rtmlib import RTMPose
    from .hand_worker import _ort_cuda
    if device == "cuda" and not _ort_cuda():
        print("[shoe_size worker] onnxruntime CUDA provider unavailable - CPU", flush=True)
        device = "cpu"
    J = json.loads(Path(jobs_path).read_text())
    work = Path(J["work"])
    jobs = J["jobs"]
    body = RTMPose(onnx_model=str(RTMW_ONNX), model_input_size=(288, 384), backend="onnxruntime", device=device)
    n = len(jobs)
    kp, sc = np.full((n, 133, 2), np.nan, np.float32), np.full((n, 133), np.nan, np.float32)
    order = sorted(range(n), key=lambda k: (jobs[k]["cam"], jobs[k]["i"]))
    img, key, t0 = None, None, time.time()
    for c, k in enumerate(order):
        j = jobs[k]
        if key != (j["cam"], j["i"]):
            key = (j["cam"], j["i"])
            img = cv2.imread(str(work / "frames" / j["cam"] / f"{j['i']:06d}.jpg"))
        a, b = body(img, bboxes=[j["box"]])
        kp[k], sc[k] = a[0], b[0]
        if c % 500 == 0:
            print(f"[shoe_size worker] {c}/{n} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out_path, body_cam=np.array([j["cam"] for j in jobs]),
                        body_i=np.array([j["i"] for j in jobs], np.int32),
                        body_role=np.array([j["role"] for j in jobs]), rtmw_kp=kp, rtmw_score=sc)
    print(f"[shoe_size worker] {n} boxes in {time.time() - t0:.0f}s on {device} -> {out_path}", flush=True)


def detect(work: Path, step: int = 4, device: str = "cuda") -> Path:
    jobs = detect_jobs(work, step)
    out = work / "avatar" / "shoe_rtmw.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    jp = work / "avatar" / "shoe_rtmw_jobs.json"
    jp.write_text(json.dumps({"work": str(work), "step": step, "jobs": jobs}))
    print(f"[shoe_size] RTMW on {len(jobs)} body boxes (every {step}th frame) in .venv-face ...", flush=True)
    subprocess.run([str(FACE_PY), "-m", "dancecap.shoe_size", "worker", str(jp), str(out), "--device", device],
                   cwd=str(PACKAGE_ROOT), check=True)
    return out


# ------------------------------------------------------------------------------------------------ run

def run(take: str, review: Path | None = None, keypoints: str = "auto") -> dict:
    work = PACKAGE_ROOT / "work" / take
    t0 = time.time()
    kps, src = load_keypoints(work, keypoints)
    grid = grid_times(work)
    X, NV, tracks = triangulate(work, kps, grid)
    recs = silhouettes(work, X, tracks, grid)
    res = {"take": take, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "keypoints": src,
           "world": "this take's world frame (metres); larissa_kadu_38's world = larissa_kadu's x 0.95446",
           "keypoint_source": keypoints,
           "triangulated_frames": {f"{r}_{s}": int((v >= 3).sum()) for (r, s), v in NV.items()},
           "planted_frames": {f"{r}_{s}": int(planted(x).sum()) for (r, s), x in X.items()},
           "dancers": summarise(recs, X), "seconds": round(time.time() - t0, 1),
           "method": "outer length = SAM2 mask run along the triangulated foot axis at 1.2-3.2 cm (side views); "
                     "width across it at 68 % (views along the foot); toe-box height = mask top at 75/85/92 %"}
    av = work / "avatar"
    av.mkdir(parents=True, exist_ok=True)
    (av / "shoe_size.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (av / "shoe_size_records.json").write_text(json.dumps(recs), encoding="utf-8")
    review = review or work / "review" / "demo"
    res["review"] = str(review_sheet(work, recs, X, tracks, Path(review) / f"shoe_size_{take}.jpg"))
    return res


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "worker":
        ap = argparse.ArgumentParser(prog="python -m dancecap.shoe_size worker")
        ap.add_argument("_w")
        ap.add_argument("jobs")
        ap.add_argument("out")
        ap.add_argument("--device", default="cuda")
        a = ap.parse_args(argv)
        worker(Path(a.jobs), Path(a.out), a.device)
        return
    ap = argparse.ArgumentParser(prog="python -m dancecap.shoe_size", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--detect", action="store_true", help="run RTMW-x first (avatar/shoe_rtmw.npz)")
    ap.add_argument("--step", type=int, default=4, help="--detect: every n-th frame of every camera")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--review", help="review crops directory (default work/<take>/review/demo)")
    ap.add_argument("--keypoints", choices=("auto", "rtmw", "ma2d"), default="auto",
                    help="auto: RTMW (hands/detections.npz or avatar/shoe_rtmw.npz), else MAMMA's ma_2d landmarks")
    a = ap.parse_args(argv)
    work = PACKAGE_ROOT / "work" / a.take
    if a.detect:
        detect(work, a.step, a.device)
    res = run(a.take, Path(a.review) if a.review else None, a.keypoints)
    print(json.dumps({k: v for k, v in res.items() if k != "dancers"}, indent=1))
    for r, e in res["dancers"].items():
        print(f"[shoe_size] {r}: " + ", ".join(f"{k} {v['m'] * 100:.1f} cm (n {v['n']}, {v['views']})"
                                               for k, v in e.items() if isinstance(v, dict) and "views" in v)
              + (f"; keypoints heel-toe {e['keypoint_heel_toe_m']['m'] * 100:.1f} cm" if "keypoint_heel_toe_m" in e else ""))


if __name__ == "__main__":
    main()
