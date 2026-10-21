r"""john_lula: does TrackEverything (or the CoTracker3 tracker it is compared with) help with FOOT SLIP? (2026-10-20)

Owner, 2026-10-20: "https://github.com/ayushjain1144/trackeverything - research this one and determine if it will help with
foot slip, monocular grounding, etc".

TrackEverything (CMU + Meta, arXiv 2609.30222, code CC BY-NC 4.0) tracks dense 3D points from RGB + depth + cameras; its
weights (katefgroup/TrackEverything) sit behind a Hugging Face click-through gate that the pods' token has NOT accepted
(auth_check 403, 2026-10-20). The pre-registered fallback STAGE 0 is run instead: CoTracker3 offline (the 2D tracker that
TrackEverything ties on TAPVid-3D Panoptic Studio: APD-P 30.1 vs 29.6 with the same geometry, paper Table 1) with the SAME
queries on our native frames, against RTMW (the existing baseline the foot lock uses). If CoTracker3 does not beat RTMW
on F1 / F2, TrackEverything is refuted for feet without running it.

WINDOW  frames 450-674 (15.00-22.47 s, 225 frames): 10 lead + 9 follow jl_footlock stance runs (best fit), Lula's front /
        back flips, the wide state (dancers ~310-330 px) and the zoomed state (640-710 px), the zoom (a 0.8 s ramp, 17.8-
        18.6 s, focal 809 -> 1600 px; cameras/01.npz zoom_2x flips at frame 543), camera reprojection median 1.5 px.
QUERIES (a) RTMW foot points (people/teachers_01.npz kp_wb ankle / big toe / small toe / heel, both dancers and feet,
        score > 0.5) seeded every 4 frames; (b) 64 points per dancer inside the own SAM mask between the RTMW shoulder and
        hip lines, seeded at 16.3, 18.7 and 20.4 s.
ARMS    ct3_512 (the predictor's 384 x 512 resize: our portrait frames squashed 2.5x vertically, as TrackEverything's default),
        ct3_native (536 x 960), both cut at the zoom flag; ct3_full (native, uncut); rtmw (no compute).
METRICS (all through jl_footlock.rotation_homography and REF_H_PX)
  F1  within-stance jitter: per jl_footlock stance run (best fit) and foot, the keypoints' positions mapped into the run's
      first frame by the rotation homography (camera rotation / zoom about its fixed centre removed), scaled to a 400 px
      person, RMS about the run's median (px); mm at 5 mm / px across (image x) and 18 mm / px along the ray (image y), (e).
      For a tracker: the query seeded inside the run nearest its centre, followed over the whole run (vis > 0.9).
  F2  stance coverage: per foot and frame the compensated speed (median over the 4 points, 3-frame median), still below a
      threshold for >= MIN_RUN frames (1-frame gaps closed); share of the visible-foot frames (RTMW own-dancer valid). The
      threshold of each tracker is set so that its false-still rate on SWING frames (render-driver lowest sole vertex
      > 8 cm AND RTMW compensated displacement > 15 px over 5 frames) equals RTMW's at IMG_STILL_PX = 2.5. A tracker's
      speed at frame t uses the query seeded latest at or before t - 1 (re-seeded every 4 frames).
  S1  torso query survival (visibility > 0.9 and on the own SAM mask) at the image-evidence 90 deg (first SIDE) and 180 deg
      (first opposite class) after each seed (facing/posterior.npz). A 2D tracker cannot give S2 (3D yaw).

    python -m dancecap.mono4d.jl_trackeverything prep      # input/ for the pod (frames, queries, meta)
    python -m dancecap.mono4d.jl_trackeverything analyze   # pod_out/*.npz -> metrics.json
    python -m dancecap.mono4d.jl_trackeverything sheets    # overlay video + sheets (videos/, sheets/)
Outputs: work/john_lula/realism/trackeverything/. Likeness data stays in work/. Reads only; writes nothing else.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import jl_footlock as fl

W0, W1 = 450, 674
CUT = 543                                  # cameras/01.npz zoom_2x flips here (18.10 s; the zoom ramp is 534-558)
FPS = 29.994
SEED_EVERY = 4
TORSO_T = (16.3, 18.7, 20.4)
TORSO_N = 64
KP_SEED = 0.5
OUT = fl.WORK / "realism" / "trackeverything"
INP = OUT / "input"
POD = OUT / "pod_out"
ARMS = ("rtmw", "ct3_512", "ct3_native", "ct3_full")
VIS_THR = 0.9                              # CoTracker3 predictor's own visibility rule
SWING_H_M = 0.08
SWING_PX = 15.0
MM_ACROSS, MM_ALONG = 5.0, 18.0
WB_TORSO = (5, 6, 12, 11)                  # COCO-WholeBody: L shoulder, R shoulder, R hip, L hip (a quad)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def frames() -> np.ndarray:
    return np.arange(W0, W1 + 1)


def segments() -> dict:
    return {"cut": [[W0, CUT - 1], [CUT, W1]], "full": [[W0, W1]]}


def _mask(M, keys, fr: int, m: int):
    import cv2
    k = f"m_{fr}_{m}"
    if k not in keys:
        return None
    mm = cv2.imdecode(np.frombuffer(M[k].tobytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    return None if mm is None else (mm > 0)


def farthest(P: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(P)))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(min(n, len(P)) - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return P[np.array(idx)]


# ================================================================================================ prep
def prep() -> dict:
    import cv2
    T_ = np.load(fl.WORK / "people" / "teachers_01.npz", allow_pickle=True)
    M = np.load(fl.WORK / "people" / "masks_01.npz")
    keys = set(M.files)
    kw = np.asarray(T_["kp_wb"], float)
    ids = np.array([[fl.KP[f][n] for n in fl.KP_NAMES] for f in fl.FEET])     # (2 video feet, 4)
    rows = []                                                                  # frame, x, y, kind, dancer, vfoot, kp, score
    for s in range(W0, W1 + 1, SEED_EVERY):
        for w in (0, 1):
            for v in (0, 1):
                for q in range(4):
                    x, y, sc = kw[s, w, ids[v, q]]
                    if sc > KP_SEED and np.isfinite(x) and 0 <= x < 538 and 0 <= y < 960:
                        rows.append((s, x, y, 0, w, v, q, sc))
    n_feet = len(rows)
    torso_info = {}
    for ts in TORSO_T:
        s = int(round(ts * FPS))
        for w in (0, 1):
            m = _mask(M, keys, s, w)
            mp = _mask(M, keys, s, 1 - w)
            if m is None:
                torso_info[f"{s}_{w}"] = "no mask"
                continue
            m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            if mp is not None:
                m &= ~(cv2.dilate(mp.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
            quad = kw[s, w, list(WB_TORSO)]
            poly = np.zeros_like(m, np.uint8)
            if (quad[:, 2] > 0.3).all():
                cv2.fillPoly(poly, [np.round(quad[:, :2]).astype(np.int32)], 1)
                how = "rtmw quad"
            else:
                ys = quad[:, 1][quad[:, 2] > 0.3]
                y0, y1 = (ys.min(), ys.max()) if len(ys) >= 2 else (0, 0)
                poly[int(y0):int(y1) + 1] = 1
                how = "rtmw band"
            cand = np.argwhere(m & (poly > 0))[:, ::-1].astype(float)
            if len(cand) < TORSO_N:
                torso_info[f"{s}_{w}"] = f"{how}: only {len(cand)} px"
                continue
            P = farthest(cand, TORSO_N, seed=s * 2 + w)
            for x, y in P:
                rows.append((s, x, y, 1, w, -1, -1, 1.0))
            torso_info[f"{s}_{w}"] = f"{how}: {len(cand)} px -> {len(P)} points"
    R = np.array(rows, float)
    INP.mkdir(parents=True, exist_ok=True)
    (INP / "frames").mkdir(exist_ok=True)
    for f in frames():
        dst = INP / "frames" / f"{f:06d}.jpg"
        if not dst.exists():
            shutil.copy2(fl.WORK / "frames" / "01" / f"{f:06d}.jpg", dst)
    np.savez_compressed(INP / "queries.npz", q_frame=R[:, 0].astype(np.int32), q_xy=R[:, 1:3].astype(np.float32),
                        q_kind=R[:, 3].astype(np.int8), q_dancer=R[:, 4].astype(np.int8), q_vfoot=R[:, 5].astype(np.int8),
                        q_kp=R[:, 6].astype(np.int8), q_score=R[:, 7].astype(np.float32))
    meta = {"take": fl.TAKE, "cam": fl.CAM, "frames": [int(f) for f in frames()], "segments": segments(), "cut": CUT,
            "seed_every": SEED_EVERY, "torso_seed_frames": [int(round(t * FPS)) for t in TORSO_T],
            "n_queries": int(len(R)), "n_feet": int(n_feet), "n_torso": int(len(R) - n_feet), "torso": torso_info,
            "frame_size_wh": [538, 960]}
    (INP / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"prep: {len(R)} queries ({n_feet} feet, {len(R) - n_feet} torso), {torso_info}")
    return meta


# ================================================================================================ analysis helpers
def comp_to(uv: np.ndarray, K: np.ndarray, R: np.ndarray, r: int) -> np.ndarray:
    """(T, 2) pixels (window lattice) -> the same points mapped into frame r by the rotation homography."""
    H = fl.rotation_homography(K, R, K[r][None], R[r][None])
    return fl.apply_h(H, uv)


def speed_series(uv: np.ndarray, ok: np.ndarray, K: np.ndarray, R: np.ndarray, hpx: np.ndarray) -> np.ndarray:
    """(T, 4, 2) points, (T, 4) validity -> (T,) median over points of the compensated speed at a 400 px person."""
    sp = np.full(uv.shape[:2], np.nan)
    for q in range(uv.shape[1]):
        u = np.where(ok[:, q, None], uv[:, q], np.nan)
        sp[:, q] = fl.compensated_speed(u, K, R) * fl.REF_H_PX / hpx
    with np.errstate(all="ignore"):
        return np.nanmedian(sp, axis=1)


def still_mask(sp_m3: np.ndarray, thr: float) -> np.ndarray:
    m = np.isfinite(sp_m3) & (sp_m3 < thr)
    m = fl.close_gaps(m, 1)
    out = np.zeros_like(m)
    for a, b in fl.runs_of(m, fl.MIN_RUN):
        out[a:b + 1] = True
    return out


class Tracks:
    """One tracker arm on the window lattice: per query (N, F, 2) tracks + vis; helper views per fit foot."""

    def __init__(self, name: str, Q: dict, meta: dict):
        z = np.load(POD / f"{name}.npz")
        self.name = name
        self.tr = z["tracks"].astype(float)
        self.vis = z["vis"].astype(float) > VIS_THR
        self.Q = Q
        segs = meta["segments"]["full" if name == "ct3_full" else "cut"]
        self.seg_of = np.zeros(len(frames()), int)
        for i, (a, b) in enumerate(segs):
            self.seg_of[a - W0:b - W0 + 1] = i

    def query(self, s: int, w: int, v: int, q: int) -> int | None:
        Q = self.Q
        hit = np.where((Q["q_kind"] == 0) & (Q["q_frame"] == s) & (Q["q_dancer"] == w) & (Q["q_vfoot"] == v) & (Q["q_kp"] == q))[0]
        return int(hit[0]) if len(hit) else None


# ================================================================================================ analyze
def analyze() -> dict:
    meta = json.loads((INP / "meta.json").read_text())
    Qz = np.load(INP / "queries.npz")
    Q = {k: Qz[k] for k in Qz.files}
    vf = fl.video_feet()
    cams = fl.load_cams()
    fits = fl.load_fit(fl.BEST)
    fr = frames()
    K, R = cams["K"][fr].astype(float), cams["R"][fr].astype(float)
    arms = {a: Tracks(a, Q, meta) for a in ARMS[1:] if (POD / f"{a}.npz").exists()}
    pod = json.loads((POD / "ct3_pod.json").read_text()) if (POD / "ct3_pod.json").exists() else {}
    import cv2
    Mz = np.load(fl.WORK / "people" / "masks_01.npz")
    mkeys = set(Mz.files)
    # own / partner mask distance for the tracker points (the same own_dancer test as RTMW's)
    dts = {}
    for f in fr:
        for m in (0, 1):
            mm = _mask(Mz, mkeys, int(f), m)
            dts[(int(f), m)] = None if mm is None or not mm.any() else cv2.distanceTransform((~mm).astype(np.uint8), cv2.DIST_L2, 3)

    def mask_d(f: int, m: int, xy: np.ndarray) -> np.ndarray:
        dt = dts[(f, m)]
        if dt is None:
            return np.full(len(xy), 255.0)
        ok = np.isfinite(xy).all(1)
        xi = np.clip(np.round(np.nan_to_num(xy[:, 0])).astype(int), 0, 537)
        yi = np.clip(np.round(np.nan_to_num(xy[:, 1])).astype(int), 0, 959)
        inb = ok & (xy[:, 0] >= 0) & (xy[:, 0] < 538) & (xy[:, 1] >= 0) & (xy[:, 1] < 960)
        return np.where(inb, np.minimum(dt[yi, xi], 255.0), 255.0)

    res = {"window": [W0, W1], "t": [round(float(cams["t"][W0]), 3), round(float(cams["t"][W1]), 3)], "cut": CUT,
           "params": {"VIS_THR": VIS_THR, "REF_H_PX": fl.REF_H_PX, "IMG_STILL_PX": fl.IMG_STILL_PX, "MIN_RUN": fl.MIN_RUN,
                      "SWING_H_M": SWING_H_M, "SWING_PX": SWING_PX, "mm_per_px_across_along": [MM_ACROSS, MM_ALONG]},
           "pod": pod, "roles": {}}
    per_arm_feet = {}                                         # for sheets / video: role -> foot -> arm -> (F, 4, 2), ok
    for role in fl.ROLES:
        w = fl.WHO[role]
        Fz = fl.Fit(fits[role])
        ev = fl.evidence(Fz, role, vf, cams)
        runs_all = fl.stance_runs(Fz, ev)
        runs = {f: [(a, b) for a, b in runs_all[f] if a >= W0 and b <= W1] for f in fl.FEET}
        swap = ev["swap"][fr]
        hpx = np.where(np.isfinite(vf["box_h"][fr, w]) & (vf["box_h"][fr, w] > 50), vf["box_h"][fr, w], fl.REF_H_PX)
        rr = {"runs": {f: runs[f] for f in fl.FEET}, "feet": {}}
        per_arm_feet[role] = {}
        for f in fl.FEET:
            fi = fl.FEET.index(f)
            rt_uv = ev["feet"][f]["uv"][fr]                     # (F, 4, 2) RTMW assigned to the fit foot
            rt_ok = ev["feet"][f]["valid"][fr]
            visible = rt_ok.any(1)
            sole = Fz.sole_low[f][fr]
            # --- tracker per-frame views: query seeded latest at or before t-1 (same segment), its points at t-1 and t
            views = {"rtmw": (rt_uv, rt_ok, rt_uv, rt_ok)}           # (uv_t, ok_t, uv_prev, ok_prev)
            for an, A in arms.items():
                cur = np.full((len(fr), 4, 2), np.nan)
                prv = np.full((len(fr), 4, 2), np.nan)
                okc = np.zeros((len(fr), 4), bool)
                okp = np.zeros((len(fr), 4), bool)
                for i in range(1, len(fr)):
                    t = fr[i]
                    s = W0 + ((t - 1 - W0) // SEED_EVERY) * SEED_EVERY
                    if A.seg_of[s - W0] != A.seg_of[i] or A.seg_of[i - 1] != A.seg_of[i]:
                        continue
                    v = (1 - fi) if swap[s - W0] else fi
                    for q in range(4):
                        qi = A.query(s, w, v, q)
                        if qi is None or not ev["feet"][f]["valid"][s, q]:
                            continue
                        cur[i, q], prv[i, q] = A.tr[qi, i], A.tr[qi, i - 1]
                        okc[i, q], okp[i, q] = A.vis[qi, i], A.vis[qi, i - 1]
                # own-dancer test on the tracker points (as RTMW's)
                for i in range(len(fr)):
                    d_own = mask_d(int(fr[i]), w, cur[i])
                    d_par = mask_d(int(fr[i]), 1 - w, cur[i])
                    okc[i] &= fl.own_dancer(d_own, d_par)
                views[an] = (cur, okc, prv, okp)
            # --- speeds
            sp = {}
            for an, (cur, okc, prv, okp) in views.items():
                if an == "rtmw":
                    s_ = speed_series(cur, okc, K, R, hpx)
                else:
                    s_ = np.full(len(fr), np.nan)
                    spq = np.full((len(fr), 4), np.nan)
                    for i in range(1, len(fr)):
                        for q in range(4):
                            if okc[i, q] and okp[i, q]:
                                Hh = fl.rotation_homography(K[i - 1], R[i - 1], K[i], R[i])
                                pred = fl.apply_h(Hh, prv[i, q][None])[0]
                                spq[i, q] = np.linalg.norm(cur[i, q] - pred) * fl.REF_H_PX / hpx[i]
                    with np.errstate(all="ignore"):
                        s_ = np.nanmedian(spq, axis=1)
                sp[an] = fl.nanmedian_filter(s_, 1)
            # --- swing frames: sole > 8 cm and RTMW compensated displacement > 15 px (400 px person) over 5 frames
            disp = np.full(len(fr), np.nan)
            for i in range(2, len(fr) - 2):
                ok2 = rt_ok[i - 2] & rt_ok[i + 2]
                if ok2.any():
                    Hh = fl.rotation_homography(K[i - 2], R[i - 2], K[i + 2], R[i + 2])
                    d = np.linalg.norm(rt_uv[i + 2][ok2] - fl.apply_h(Hh, rt_uv[i - 2][ok2]), axis=1)
                    disp[i] = np.median(d) * fl.REF_H_PX / hpx[i]
            # pre-registered: the render driver's lowest sole vertex > 8 cm. Zouk feet glide: in this window that never
            # happens (0 frames), so the substitute is the render driver's foot point moving > 8 cm horizontally over
            # the same 5 frames (both definitions reported; the one used is res['swing_definition']).
            fp = Fz.foot_point(f)[fr][:, [0, 2]]
            fmove = np.full(len(fr), np.nan)
            fmove[2:-2] = np.linalg.norm(fp[4:] - fp[:-4], axis=1)
            swing_pre = (sole > SWING_H_M) & (disp > SWING_PX) & visible
            swing_alt = (fmove > SWING_H_M) & (disp > SWING_PX) & visible
            swing = swing_pre if swing_pre.sum() >= 20 else swing_alt
            low = (sole < fl.FIT_LOW_M) & visible
            # --- F2 at fixed 2.5 px and at RTMW's false-still rate
            f2 = {"n_visible": int(visible.sum()), "n_swing": int(swing.sum()), "n_swing_preregistered": int(swing_pre.sum()),
                  "n_swing_substitute": int(swing_alt.sum()), "swing_used": "preregistered" if swing is swing_pre else "substitute",
                  "n_fit_low": int(low.sum())}
            base_still = still_mask(sp["rtmw"], fl.IMG_STILL_PX)
            fs_rt = float(base_still[swing].mean()) if swing.any() else float("nan")
            for an in views:
                st = still_mask(sp[an], fl.IMG_STILL_PX)
                thr_eq = fl.IMG_STILL_PX
                if an != "rtmw" and swing.any():
                    best = None
                    for thr in np.arange(0.5, 8.0001, 0.05):
                        fs = still_mask(sp[an], thr)[swing].mean()
                        if fs <= fs_rt + 1e-9:
                            best = thr
                    thr_eq = float(best) if best is not None else 0.5
                st_eq = still_mask(sp[an], thr_eq)
                f2[an] = {"cover_2p5": round(float(st[visible].mean()), 4) if visible.any() else None,
                          "false_still_2p5": round(float(st[swing].mean()), 4) if swing.any() else None,
                          "thr_equal_false_still": round(thr_eq, 2),
                          "cover_eq": round(float(st_eq[visible].mean()), 4) if visible.any() else None,
                          "false_still_eq": round(float(st_eq[swing].mean()), 4) if swing.any() else None,
                          "cover_eq_fit_low": round(float(st_eq[low].mean()), 4) if low.any() else None,
                          "speed_frames": int(np.isfinite(sp[an][visible]).sum())}
            # --- F1 per run
            f1 = {an: [] for an in views}
            for a, b in runs[f]:
                ia, ib = a - W0, b - W0
                for an, (cur, okc, prv, okp) in views.items():
                    if an == "rtmw":
                        pts = [(rt_uv[ia:ib + 1, q], rt_ok[ia:ib + 1, q], q) for q in range(4)]
                    else:
                        A = arms[an]
                        pts = []
                        seeds = [s for s in range(W0, W1 + 1, SEED_EVERY) if a <= s <= b]
                        seeds.sort(key=lambda s: abs(s - 0.5 * (a + b)))
                        for q in range(4):
                            for s in seeds:
                                v = (1 - fi) if swap[s - W0] else fi
                                qi = A.query(s, w, v, q)
                                if qi is None or not ev["feet"][f]["valid"][s, q]:
                                    continue
                                same = A.seg_of[ia:ib + 1] == A.seg_of[s - W0]
                                pts.append((A.tr[qi, ia:ib + 1], A.vis[qi, ia:ib + 1] & same, q))
                                break
                    devs, devs_toe = [], []
                    for uv, ok, q in pts:
                        if ok.sum() < 3:
                            continue
                        u = np.where(ok[:, None], comp_to(np.nan_to_num(uv), K[ia:ib + 1], R[ia:ib + 1], 0), np.nan)
                        u = u * fl.REF_H_PX / float(np.median(hpx[ia:ib + 1]))
                        d = u - np.nanmedian(u, axis=0)
                        d = d[np.isfinite(d).all(1)]
                        devs.append(d)
                        if q in fl.TRACK_KP:
                            devs_toe.append(d)
                    if not devs:
                        f1[an].append(None)
                        continue
                    D = np.concatenate(devs)
                    Dt = np.concatenate(devs_toe) if devs_toe else np.zeros((0, 2))
                    f1[an].append({"run": [int(a), int(b)], "n_pts": len(devs), "n": int(len(D)),
                                   "rms_px": round(float(np.sqrt((D ** 2).sum(1).mean())), 3),
                                   "rms_mm": round(float(np.sqrt(((D[:, 0] * MM_ACROSS) ** 2 + (D[:, 1] * MM_ALONG) ** 2).mean())), 2),
                                   "rms_px_toes": round(float(np.sqrt((Dt ** 2).sum(1).mean())), 3) if len(Dt) else None})
            rr["feet"][f] = {"F1": f1, "F2": f2}
            per_arm_feet[role][f] = {an: (v_[0], v_[1]) for an, v_ in views.items()}
            per_arm_feet[role][f]["_speed"] = sp
            per_arm_feet[role][f]["_swing"] = swing
        res["roles"][role] = rr
    # --- summary per arm
    summ = {}
    for an in ["rtmw"] + list(arms):
        rows = [r for role in fl.ROLES for f in fl.FEET for r in res["roles"][role]["feet"][f]["F1"][an] if r]
        allr = [r for role in fl.ROLES for f in fl.FEET for r in res["roles"][role]["feet"][f]["F1"]["rtmw"] if r]
        cov = [res["roles"][role]["feet"][f]["F2"][an] for role in fl.ROLES for f in fl.FEET]
        nvis = [res["roles"][role]["feet"][f]["F2"]["n_visible"] for role in fl.ROLES for f in fl.FEET]
        nsw = [res["roles"][role]["feet"][f]["F2"]["n_swing"] for role in fl.ROLES for f in fl.FEET]
        wavg = lambda key, wts: round(float(np.nansum([(c[key] if c[key] is not None else np.nan) * n for c, n in zip(cov, wts)]) / max(sum(wts), 1)), 4)
        summ[an] = {"F1_runs_scored": len(rows), "F1_runs_total": len(allr),
                    "F1_median_rms_px": round(float(np.median([r["rms_px"] for r in rows])), 3) if rows else None,
                    "F1_median_rms_mm": round(float(np.median([r["rms_mm"] for r in rows])), 2) if rows else None,
                    "F1_pooled_rms_px": round(float(np.sqrt(np.sum([r["rms_px"] ** 2 * r["n"] for r in rows]) / max(np.sum([r["n"] for r in rows]), 1))), 3) if rows else None,
                    "F1_median_rms_px_toes": round(float(np.median([r["rms_px_toes"] for r in rows if r["rms_px_toes"] is not None])), 3) if rows else None,
                    "F2_cover_2p5": wavg("cover_2p5", nvis), "F2_false_still_2p5": wavg("false_still_2p5", nsw),
                    "F2_cover_eq": wavg("cover_eq", nvis), "F2_false_still_eq": wavg("false_still_eq", nsw)}
    # paired F1 (runs scored by both rtmw and the arm)
    for an in arms:
        pr = []
        for role in fl.ROLES:
            for f in fl.FEET:
                for r0, r1 in zip(res["roles"][role]["feet"][f]["F1"]["rtmw"], res["roles"][role]["feet"][f]["F1"][an]):
                    if r0 and r1:
                        pr.append(r1["rms_px"] / max(r0["rms_px"], 1e-9))
        summ[an]["F1_paired_ratio_median"] = round(float(np.median(pr)), 3) if pr else None
        summ[an]["F1_paired_n"] = len(pr)
    res["summary"] = summ
    # --- S1 torso survival
    res["S1"] = s1(arms, Q, mask_d, cams)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    np.savez_compressed(OUT / "analysis_cache.npz", **{f"{role}_{f}_{an}_{k}": np.asarray(v[i])
                                                        for role in per_arm_feet for f in per_arm_feet[role]
                                                        for an, v in per_arm_feet[role][f].items() if not an.startswith("_")
                                                        for i, k in enumerate(("uv", "ok"))},
                        **{f"{role}_{f}_speed_{an}": s for role in per_arm_feet for f in per_arm_feet[role]
                           for an, s in per_arm_feet[role][f]["_speed"].items()},
                        **{f"{role}_{f}_swing": per_arm_feet[role][f]["_swing"] for role in per_arm_feet for f in per_arm_feet[role]})
    log(json.dumps(summ, indent=1))
    return res


def s1(arms: dict, Q: dict, mask_d, cams) -> dict:
    P = np.load(fl.WORK / "realism" / "v2" / "facing" / "posterior.npz")
    post = P["post"]
    out = {}
    for s in sorted(set(Q["q_frame"][Q["q_kind"] == 1].tolist())):
        for w in (0, 1):
            sel = np.where((Q["q_kind"] == 1) & (Q["q_frame"] == s) & (Q["q_dancer"] == w))[0]
            if not len(sel):
                continue
            c0 = int(post[s, w].argmax())
            ev = {"seed_class": ["front", "side", "back"][c0], "seed_conf": round(float(post[s, w].max()), 3)}
            t90 = t180 = None
            for t in range(s + 1, W1 + 1):
                c = int(post[t, w].argmax())
                if t90 is None and c == 1:
                    t90 = t
                if c0 != 1 and c == 2 - c0 and post[t, w].max() > 0.8:
                    t180 = t
                    break
            ev["t90"], ev["t180"] = t90, t180
            for an, A in arms.items():
                r = {}
                for lab, t in (("seed+15", min(s + 15, W1)), ("90", t90), ("180", t180)):
                    if t is None:
                        continue
                    i = t - W0
                    same = A.seg_of[i] == A.seg_of[s - W0]
                    xy = A.tr[sel, i]
                    on = mask_d(int(t), w, xy) <= 3
                    alive = A.vis[sel, i] & on & same
                    r[lab] = {"frame": int(t), "vis": round(float((A.vis[sel, i] & same).mean()), 3),
                              "vis_on_mask": round(float(alive.mean()), 3), "on_mask": round(float((on & same).mean()), 3)}
                ev[an] = r
            out[f"{s}_{['lead', 'follow'][w]}"] = ev
    return out


# ================================================================================================ sheets / video
def sheets() -> None:
    import cv2
    res = json.loads((OUT / "metrics.json").read_text())
    C = np.load(OUT / "analysis_cache.npz")
    Qz = np.load(INP / "queries.npz")
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    arms = [a for a in ARMS if f"lead_left_{a}_uv" in C.files]
    col = {"rtmw": (0, 200, 255), "ct3_512": (255, 160, 0), "ct3_native": (60, 230, 60), "ct3_full": (230, 60, 230)}
    fr = frames()
    # overlay video: RTMW | ct3_native, feet points coloured, stance frames (RTMW still) marked
    vw = cv2.VideoWriter(str(OUT / "videos" / "feet_rtmw_vs_ct3.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS / 2, (538 * 2, 960))
    show = ["rtmw", "ct3_native" if "ct3_native" in arms else arms[-1]]
    for i, f in enumerate(fr):
        img = cv2.imread(str(INP / "frames" / f"{f:06d}.jpg"))
        panels = []
        for an in show:
            p = img.copy()
            for role in fl.ROLES:
                for foot in fl.FEET:
                    uv = C[f"{role}_{foot}_{an}_uv"][i]
                    ok = C[f"{role}_{foot}_{an}_ok"][i]
                    spd = C[f"{role}_{foot}_speed_{an}"][i]
                    still = np.isfinite(spd) and spd < fl.IMG_STILL_PX
                    for q in range(4):
                        if ok[q] and np.isfinite(uv[q]).all():
                            cv2.circle(p, (int(round(uv[q, 0])), int(round(uv[q, 1]))), 4 if still else 2, col[an], -1 if still else 1, cv2.LINE_AA)
            cv2.putText(p, f"{an}  {f} {f / FPS:5.2f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(p, "filled = still (< 2.5 px)", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(p)
        vw.write(np.hstack(panels))
    vw.release()
    # speed sheet: per role / foot, the compensated speed of each arm over the window, swing frames shaded
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    t = fr / FPS
    for k, (role, foot) in enumerate([(r, f) for r in fl.ROLES for f in fl.FEET]):
        ax = axs[k]
        sw = C[f"{role}_{foot}_swing"]
        ax.fill_between(t, 0, 12, where=sw, color="0.85", step="mid", label="swing (fit foot moves > 8 cm and RTMW > 15 px / 5 fr)")
        for a, b in res["roles"][role]["runs"][foot]:
            ax.axvspan(a / FPS, b / FPS, color=(0.8, 1.0, 0.8), alpha=0.6)
        for an in arms:
            ax.plot(t, C[f"{role}_{foot}_speed_{an}"], color=np.array(col[an][::-1]) / 255, lw=1.2, label=an)
        ax.axhline(fl.IMG_STILL_PX, color="k", ls="--", lw=0.8)
        ax.axvline(CUT / FPS, color="r", lw=0.8)
        ax.set_ylim(0, 12)
        ax.set_ylabel(f"{fl.NAME[role]} {foot}\npx/frame @400px")
        if k == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=3)
    axs[-1].set_xlabel("s (green = jl_footlock stance runs; red = zoom flag)")
    fig.suptitle("john_lula 15.0-22.5 s: camera-compensated foot speed, RTMW vs CoTracker3 arms")
    fig.tight_layout()
    fig.savefig(OUT / "sheets" / "foot_speed.png", dpi=90)
    plt.close(fig)
    # stance-run jitter sheet: for 6 runs, crops with the compensated points of each arm
    tiles = []
    for role in fl.ROLES:
        for foot in fl.FEET:
            for a, b in res["roles"][role]["runs"][foot][:2]:
                m = (a + b) // 2
                img = cv2.imread(str(INP / "frames" / f"{m:06d}.jpg"))
                uv = C[f"{role}_{foot}_rtmw_uv"][m - W0]
                c = np.nanmean(uv, axis=0) if np.isfinite(uv).any() else np.array([269, 700])
                cx, cy = int(np.clip(c[0], 60, 478)), int(np.clip(c[1], 60, 900))
                crop = img[cy - 60:cy + 60, cx - 60:cx + 60].copy()
                crop = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_CUBIC)
                for an in arms:
                    U = C[f"{role}_{foot}_{an}_uv"][a - W0:b - W0 + 1]
                    O = C[f"{role}_{foot}_{an}_ok"][a - W0:b - W0 + 1]
                    for j in range(len(U)):
                        for q in fl.TRACK_KP:
                            if O[j, q] and np.isfinite(U[j, q]).all():
                                x = (U[j, q, 0] - (cx - 60)) * 2
                                y = (U[j, q, 1] - (cy - 60)) * 2
                                cv2.circle(crop, (int(x), int(y)), 2, col[an], -1, cv2.LINE_AA)
                cv2.putText(crop, f"{fl.NAME[role][0]} {foot[0]} {a / FPS:.2f}-{b / FPS:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(crop)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        sheet = np.vstack(rows)
        leg = np.zeros((28, sheet.shape[1], 3), np.uint8)
        x = 6
        for an in arms:
            cv2.putText(leg, an, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col[an], 1, cv2.LINE_AA)
            x += 160
        cv2.imwrite(str(OUT / "sheets" / "stance_runs_toes.jpg"), np.vstack([leg, sheet]))
    # torso spin sheet: Lula's / John's torso queries through the image-evidence turns (ct3_native)
    an = "ct3_native" if (POD / "ct3_native.npz").exists() else None
    if an:
        z = np.load(POD / f"{an}.npz")
        trk, vis = z["tracks"], z["vis"].astype(float) > VIS_THR
        Mz = np.load(fl.WORK / "people" / "masks_01.npz")
        mkeys = set(Mz.files)
        rows = []
        for key, ev in res["S1"].items():
            s, role = int(key.split("_")[0]), key.split("_")[1]
            w = fl.WHO[role]
            sel = np.where((Qz["q_kind"] == 1) & (Qz["q_frame"] == s) & (Qz["q_dancer"] == w))[0]
            ts = sorted(set([s, min(s + 5, W1)] + [t for t in (ev.get("t90"), ev.get("t180")) if t] + [min(s + 30, W1)]))
            tiles = []
            for t in ts:
                img = cv2.imread(str(INP / "frames" / f"{t:06d}.jpg"))
                m = _mask(Mz, mkeys, t, w)
                xy = trk[sel, t - W0]
                ok = np.isfinite(xy).all(1)
                if m is not None and m.any():
                    yy, xx = np.nonzero(m)
                    x0, x1, y0, y1 = xx.min(), xx.max(), yy.min(), yy.max()
                else:
                    x0, x1, y0, y1 = 0, 537, 0, 959
                for j in range(len(sel)):
                    if not ok[j]:
                        continue
                    x, y = int(round(xy[j, 0])), int(round(xy[j, 1]))
                    on = m is not None and 0 <= x < 538 and 0 <= y < 960 and m[y, x]
                    c = (60, 230, 60) if vis[sel[j], t - W0] and on else ((0, 0, 255) if vis[sel[j], t - W0] else (160, 160, 160))
                    cv2.circle(img, (x, y), 3, c, -1, cv2.LINE_AA)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                hh = max(int((y1 - y0) * 0.6), 120)
                crop = img[max(0, cy - hh):min(960, cy + hh), max(0, cx - hh // 2):min(538, cx + hh // 2)]
                crop = cv2.resize(crop, (180, 360))
                post = np.load(fl.WORK / "realism" / "v2" / "facing" / "posterior.npz")["post"][t, w]
                cv2.putText(crop, f"{t / FPS:.2f}s {'FSB'[int(post.argmax())]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(crop)
            while len(tiles) < 5:
                tiles.append(np.zeros_like(tiles[0]))
            row = np.hstack(tiles[:5])
            lab = np.zeros((22, row.shape[1], 3), np.uint8)
            cv2.putText(lab, f"{fl.NAME[role]} torso seeds {s / FPS:.1f}s ({an}; green vis+on mask, red vis off mask, grey invisible)",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(np.vstack([lab, row]))
        if rows:
            cv2.imwrite(str(OUT / "sheets" / "torso_spins.jpg"), np.vstack(rows))
    log("sheets written")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "analyze", "sheets"])
    a = ap.parse_args()
    {"prep": prep, "analyze": analyze, "sheets": sheets}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
