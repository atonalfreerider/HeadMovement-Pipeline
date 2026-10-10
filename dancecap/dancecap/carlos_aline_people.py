"""carlos_aline: who the two dancers are in each of the two cameras, per canonical frame (agent C3).

Carlos = lead (pid 1, white shirt), Aline = follow (pid 2, red top). Two tripod cameras (per-frame rotation / focal tracks of C2), YOLO11x-pose of every
person (pose tables of dancecap.carlos_aline_inputs), so the stage is: veto by WHERE a person stands, then decide WHO by colour + tracking.

    zone     the floor point of each person (ankles, else hips at a nominal height) is intersected with the floor plane through the camera's own
             per-frame ray; only persons inside the dance zone (a disc of ZONE_R metres round the world origin = the dancers' mean position; the
             dancers stay within ~2.3 m of it) are candidates: the five judges sit 9 m behind it, the DJ-table crew is further, a hostess
             is gone before the danced span starts.
    colour   torso colour of each candidate (people2d.torso_rgb): chroma "redness". A two-class Gaussian model per camera is learned from the frames
             with two well separated candidates (the redder one is Aline), then used as a log-likelihood ratio per candidate.
    track    ONE Viterbi over the ordered pair (candidate for Carlos, candidate for Aline), each possibly missing: unary = colour, missing and
             unassigned-candidate costs, transition = how far each slot's person moved in the image (relative to its size). A swap at a crossing
             costs the sum of two large jumps; the colour evidence of the neighbouring frames decides it. Forward-backward marginals of the same
             model give the role confidence (the frames around an ambiguous crossing are low-confidence instead of wrong-and-sure).
    verify   both cameras at the same instants: the floor positions of the two roles must agree (a swap in one camera is a disagreement).

    python -m dancecap.carlos_aline_people run [--cams 01 02]     teachers_<cam>.npz (+ teachers.json)
    python -m dancecap.carlos_aline_people verify                 cross-camera agreement, zone, height; people/verification.json
    python -m dancecap.carlos_aline_people review [--times ...]   contact sheets in review/people/

Output contract = recap_people's teachers_<cam>.npz (CONTRACTS.md "Recap people"): t, frame_index, ok, row (F, 2) [lead, follow], box, kp, conf, plus
merged, role_conf, floor (F, 2, 2) metres, size, attached.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .carlos_aline_inputs import CAMS, PEOPLE, SOURCE, VIDEO, WORK
from .recap_people import undistort_norm

ZONE_R = {"01": 4.5, "02": 6.5}   # m round the world origin (the dancers' ankles stay within 2.3 m, the judges are 9 m behind). Camera 02 stands 0.64 m above the floor:
                             # its ankle rays graze the floor and its depths stretch, so its zone is wider and the static veto below does the judging
STATIC_CELL_DEG = 0.3        # a person-instance whose foot ray falls in a cell of this size (and its neighbours) in at least STATIC_MIN frames of the video is furniture
STATIC_MIN = 1700            # (seated judges, the DJ-table crew: thousands of frames; the dancers dwell at most a few hundred frames per cell)
HIP_H = 0.95                 # m, nominal hip height for a person whose ankles are not seen
MAX_CAND = 4                 # candidates kept per frame (largest in the zone)
JOINT_CONF = 0.3
MIN_JOINTS = 6
MISS_COST = 2.0              # per frame and slot with no person
SPARE_COST = 0.3             # a zone candidate nobody took
MOVE_SCALE = 0.10            # image displacement per frame / person size at which the continuity cost is 1
MOVE_CAP = 30.0
REACQ_COST = 1.0             # missing -> present
LOSE_COST = 0.5              # present -> missing
LLR_CLIP = 3.0
W_COLOUR = 1.0
DUP_IOU = 0.8
SIZE_FRAC = 0.6              # a candidate must be at least this share of the neighbourhood's median largest candidate
X_OK_PX = 8.0                # cross-view reprojection (px) up to which a pairing is consistent
X_PEN_PER_PX = 0.125         # penalty per px above X_OK_PX, capped at X_PEN_MAX
X_PEN_MAX = 4.0
X_MIN_CONF = 0.5             # the other camera's dancer must be this sure before it judges a candidate here
VIDEO_OFFSET_S = 0.060       # C2: camera 01's frame at audio time t shows what camera 02's frame at t + 0.060 s shows (to be confirmed by dancecap.sync)


def log(msg: str) -> None:
    print(f"[carlos_aline_people {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ geometry and features (pure numpy)


def floor_point(uv: np.ndarray, conf: np.ndarray, K: np.ndarray, R: np.ndarray, C: np.ndarray, k1: float) -> np.ndarray:
    """(x, z) on the floor plane y = 0 of one person: the mean of the ankle rays (conf >= JOINT_CONF), else the hip rays at HIP_H. NaN when no ray hits the floor."""
    def hit(px: np.ndarray, h: float) -> np.ndarray | None:
        n = undistort_norm(px, K, k1)
        d = (np.c_[n, np.ones(len(n))]) @ R                   # R is world -> camera, so R^T d = d @ R
        dy = d[:, 1]
        t = (h - C[1]) / np.where(np.abs(dy) < 1e-9, 1e-9, dy)
        good = (t > 0) & (dy < 0 if h < C[1] else dy > 0)
        if not good.any():
            return None
        P = C[None] + t[good, None] * d[good]
        return P[:, [0, 2]]

    for ids, h in (((15, 16), 0.0), ((11, 12), HIP_H)):
        ok = [i for i in ids if conf[i] >= JOINT_CONF]
        if ok:
            r = hit(uv[ok], h)
            if r is not None:
                return r.mean(0)
    return np.full(2, np.nan)


def foot_direction(box: np.ndarray, K: np.ndarray, R: np.ndarray, k1: float) -> np.ndarray:
    """(azimuth, elevation) in degrees of the world ray through the bottom centre of a box: a pan / tilt / zoom independent address of where a person
    stands in the room (a seated judge keeps it all through the heat, a dancer does not)."""
    n = undistort_norm(np.array([[(box[0] + box[2]) / 2, box[3]]]), K, k1)[0]
    d = np.r_[n, 1.0] @ R
    d = d / np.linalg.norm(d)
    return np.degrees([np.arctan2(d[0], -d[2]), np.arcsin(np.clip(d[1], -1, 1))])


def static_cells(dirs: np.ndarray, cell: float = STATIC_CELL_DEG) -> dict:
    """Counts of person-instances per angular cell: {(i, j): n} from dirs (N, 2) degrees."""
    ij = np.floor(dirs / cell).astype(int)
    out: dict = {}
    for a, b in map(tuple, ij):
        out[(a, b)] = out.get((a, b), 0) + 1
    return out


def is_static(d: np.ndarray, counts: dict, cell: float = STATIC_CELL_DEG, need: int = STATIC_MIN) -> bool:
    a, b = np.floor(np.asarray(d) / cell).astype(int)
    return sum(counts.get((a + x, b + y), 0) for x in (-1, 0, 1) for y in (-1, 0, 1)) >= need


def redness(rgb: np.ndarray) -> np.ndarray:
    """Chroma redness of a mean torso colour (0-255): (R - (G + B) / 2) / (R + G + B + 60). White / grey / black ~0, a red top 0.15-0.4."""
    rgb = np.asarray(rgb, float)
    return (rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])) / (rgb[..., 0] + rgb[..., 1] + rgb[..., 2] + 60.0)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


def fit_colour_model(red_a: np.ndarray, red_c: np.ndarray) -> dict:
    """Robust two-class model of the redness: Aline (red) and Carlos (white)."""
    def stats(x):
        x = x[np.isfinite(x)]
        med = float(np.median(x))
        sd = max(1.4826 * float(np.median(np.abs(x - med))), 0.03)
        return med, sd
    ma, sa = stats(red_a)
    mc, sc = stats(red_c)
    return {"aline": [ma, sa], "carlos": [mc, sc], "n": [int(np.isfinite(red_a).sum()), int(np.isfinite(red_c).sum())]}


def llr_aline(red: np.ndarray, model: dict) -> np.ndarray:
    """log p(red | Aline) - log p(red | Carlos), clipped; 0 for an unknown colour."""
    ma, sa = model["aline"]
    mc, sc = model["carlos"]
    r = np.asarray(red, float)
    ll = (-0.5 * ((r - ma) / sa) ** 2 - np.log(sa)) - (-0.5 * ((r - mc) / sc) ** 2 - np.log(sc))
    return np.where(np.isfinite(r), np.clip(ll, -LLR_CLIP, LLR_CLIP), 0.0)


def move_cost(prev_c: np.ndarray, prev_h: np.ndarray, cur_c: np.ndarray, cur_h: np.ndarray) -> np.ndarray:
    """(P+1, Q+1) cost of one slot going from candidate j of the previous frame (index 0 = missing) to candidate k of this one."""
    P, Q = len(prev_c), len(cur_c)
    T = np.zeros((P + 1, Q + 1))
    T[0, 1:] = REACQ_COST
    T[1:, 0] = LOSE_COST
    if P and Q:
        d = np.linalg.norm(prev_c[:, None, :] - cur_c[None, :, :], axis=-1) / np.maximum(np.maximum(prev_h[:, None], cur_h[None, :]), 40.0)
        T[1:, 1:] = np.minimum((d / MOVE_SCALE) ** 2, MOVE_CAP)
    return T


def pair_states(m: int) -> np.ndarray:
    """All ordered (carlos, aline) assignments of m candidates, index 0 = missing: both present never on the same candidate. (S, 2) with candidate = index - 1."""
    st = [(c, a) for c in range(m + 1) for a in range(m + 1) if c == 0 or a == 0 or c != a]
    return np.array(st, int)


def unary(states: np.ndarray, llr: np.ndarray, m: int, xpen: np.ndarray | None = None) -> np.ndarray:
    """Cost of every state of a frame with m candidates whose colour log-likelihood ratios (Aline vs Carlos) are llr (m,); xpen (m, 2) = cross-view penalty of
    candidate k as Carlos / as Aline (the other camera's dancers do not agree with it), 0 when unknown."""
    s = np.r_[0.0, llr]
    xp = np.vstack([np.zeros((1, 2)), xpen]) if xpen is not None and len(xpen) else np.zeros((m + 1, 2))
    cost = np.where(states[:, 0] == 0, MISS_COST, W_COLOUR * s[states[:, 0]] + xp[states[:, 0], 0])          # a candidate with a high Aline score is expensive as Carlos
    cost = cost + np.where(states[:, 1] == 0, MISS_COST, -W_COLOUR * s[states[:, 1]] + xp[states[:, 1], 1])
    used = np.zeros((len(states), m + 1), bool)
    used[np.arange(len(states)), states[:, 0]] = True
    used[np.arange(len(states)), states[:, 1]] = True
    cost = cost + SPARE_COST * (~used[:, 1:]).sum(1)
    return cost


def track_pair(frames: list[dict]) -> dict:
    """Viterbi + forward-backward over the frames. Each frame: {"c": (m, 2) centres, "h": (m,) sizes, "llr": (m,)}. Returns per frame the chosen
    candidate of Carlos and Aline (-1 = none) and the marginal probability of that choice for each role."""
    n = len(frames)
    sts = [pair_states(len(f["h"])) for f in frames]
    U = [unary(s, f["llr"], len(f["h"]), f.get("xpen")) for s, f in zip(sts, frames)]
    V = [U[0]]
    back = [None]
    trans = [None]
    for i in range(1, n):
        a, b = frames[i - 1], frames[i]
        Tc = move_cost(a["c"], a["h"], b["c"], b["h"])
        T = Tc[np.ix_(sts[i - 1][:, 0], sts[i][:, 0])] + Tc[np.ix_(sts[i - 1][:, 1], sts[i][:, 1])]
        trans.append(T)
        tot = V[-1][:, None] + T
        back.append(np.argmin(tot, 0))
        V.append(U[i] + tot.min(0))
    path = np.zeros(n, int)
    path[-1] = int(np.argmin(V[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i][path[i]]
    # forward-backward (probabilities exp(-cost), log-sum-exp)
    def lse(x, axis):
        m_ = x.max(axis, keepdims=True)
        return (m_ + np.log(np.exp(x - m_).sum(axis, keepdims=True))).squeeze(axis)
    al = [-U[0]]
    for i in range(1, n):
        al.append(-U[i] + lse(al[-1][:, None] - trans[i], 0))
    be = [None] * n
    be[-1] = np.zeros(len(sts[-1]))
    for i in range(n - 2, -1, -1):
        be[i] = lse(-trans[i + 1] - U[i + 1][None, :] + be[i + 1][None, :], 1)
    cand = np.full((n, 2), -1, int)
    pc = np.zeros((n, 2))
    for i in range(n):
        g = al[i] + be[i]
        g = np.exp(g - g.max())
        g /= g.sum()
        s = sts[i][path[i]]
        cand[i] = s - 1
        for r in range(2):
            pc[i, r] = float(g[sts[i][:, r] == s[r]].sum())
    return {"cand": cand, "p": pc}


# ------------------------------------------------------------------------------------------------ one camera


def load_cam(cam: str) -> dict:
    z = np.load(WORK / "cameras" / f"{cam}_world.npz", allow_pickle=True)
    z0 = np.load(WORK / "cameras" / f"{cam}.npz", allow_pickle=True)
    pt = np.load(PEOPLE / "pose_tables" / f"{cam}.npz")
    return {"id": cam, "t": np.asarray(z["t"], float), "ok": np.asarray(z["ok"], bool), "fi": np.asarray(z0["frame_index"], int), "K": np.asarray(z["K"], float),
            "R": np.asarray(z["R"], float), "C": np.asarray(z["C"], float), "k1": np.asarray(z["dist"], float), "size": tuple(int(x) for x in z0["native_size"]),
            "kp": pt["kp"], "n": pt["n"], "box": pt["box"], "dconf": pt["det_conf"], "rgb": pt["rgb"]}


def static_counts(cam: dict) -> dict:
    """Angular cell counts of every person-instance of the camera's canonical frames (before any selection)."""
    dirs = []
    for i in range(len(cam["t"])):
        if not cam["ok"][i]:
            continue
        r = int(cam["fi"][i])
        for j in range(int(cam["n"][r]) if r < len(cam["n"]) else 0):
            b = cam["box"][r, j]
            if np.isfinite(b).all():
                dirs.append(foot_direction(b, cam["K"][i], cam["R"][i], float(cam["k1"][i])))
    return static_cells(np.array(dirs))


def judges_band(cam: dict, counts: dict, lo_pad: float = 0.1, hi_pad: float = 0.4) -> tuple[float, float]:
    """Elevation band (degrees) of the feet of the static furniture (the judges' row): a person standing in it is at the judges' depth, a dancer never is (the
    dancers' feet are 1-4 degrees lower in the picture). Computed from the camera's own static cells (5th to 95th percentile, padded)."""
    el = []
    for i in range(0, len(cam["t"]), 2):
        if not cam["ok"][i]:
            continue
        r = int(cam["fi"][i])
        for j in range(int(cam["n"][r]) if r < len(cam["n"]) else 0):
            b = cam["box"][r, j]
            if np.isfinite(b).all():
                d = foot_direction(b, cam["K"][i], cam["R"][i], float(cam["k1"][i]))
                if is_static(d, counts):
                    el.append(d[1])
    lo, hi = np.percentile(el, [5, 95])
    log(f"{cam['id']}: judges' feet band {lo - lo_pad:.2f} .. {hi + hi_pad:.2f} deg ({len(el)} static instances)")
    return float(lo - lo_pad), float(hi + hi_pad)


def candidates(cam: dict) -> list[dict]:
    """Per canonical frame the zone candidates: person rows, floor points, boxes, redness. Everything the tracker needs."""
    out = []
    counts = static_counts(cam)
    band = judges_band(cam, counts)
    n_static = 0
    for i in range(len(cam["t"])):
        r = int(cam["fi"][i])
        n = int(cam["n"][r]) if r < len(cam["n"]) else 0
        rows, fl = [], []
        if cam["ok"][i] and n:
            for j in range(n):
                kp = cam["kp"][r, j]
                if int((kp[:, 2] >= JOINT_CONF).sum()) < MIN_JOINTS:
                    continue
                f = floor_point(kp[:, :2], kp[:, 2], cam["K"][i], cam["R"][i], cam["C"][i], float(cam["k1"][i]))
                if not np.isfinite(f).all() or np.hypot(*f) > ZONE_R[cam["id"]]:
                    continue
                fd = foot_direction(cam["box"][r, j], cam["K"][i], cam["R"][i], float(cam["k1"][i]))
                if is_static(fd, counts) or band[0] <= fd[1] <= band[1]:        # furniture, or standing where the judges' feet are (they stand up and applaud at the end)
                    n_static += 1
                    continue
                rows.append(j)
                fl.append(f)
        # duplicates (the detector's double boxes on one body): keep the more confident
        keep = []
        for a, j in enumerate(rows):
            if any(box_iou(cam["box"][r, j], cam["box"][r, rows[b]]) > DUP_IOU and cam["dconf"][r, rows[b]] >= cam["dconf"][r, j] for b in keep):
                continue
            keep = [b for b in keep if not (box_iou(cam["box"][r, j], cam["box"][r, rows[b]]) > DUP_IOU)] + [a]
        rows = [rows[a] for a in keep]
        fl = [fl[a] for a in keep]
        if len(rows) > MAX_CAND:
            size = [cam["box"][r, j, 3] - cam["box"][r, j, 1] for j in rows]
            top = np.argsort(size)[::-1][:MAX_CAND]
            rows, fl = [rows[a] for a in top], [fl[a] for a in top]
        out.append({"i": i, "r": r, "rows": rows, "floor": np.array(fl).reshape(-1, 2)})
    log(f"{cam['id']}: {n_static} in-zone person-instances vetoed as static furniture")
    return size_filter(cam, out)


def size_filter(cam: dict, out: list[dict], win: int = 90, frac: float = SIZE_FRAC) -> list[dict]:
    """Drop candidates much smaller than the dancers of the neighbourhood: a half-hidden judge between the two dancers, a head above a table. The reference is the median
    over +-win frames of the frame's largest candidate (the dancers are the biggest people of the zone)."""
    big = np.array([max([cam["box"][c["r"], j, 3] - cam["box"][c["r"], j, 1] for j in c["rows"]] or [np.nan]) for c in out])
    n_drop = 0
    for i, c in enumerate(out):
        seg = big[max(0, i - win):i + win + 1]
        seg = seg[np.isfinite(seg)]
        if not len(seg):
            continue
        ref = float(np.median(seg))
        keep = [a for a, j in enumerate(c["rows"]) if cam["box"][c["r"], j, 3] - cam["box"][c["r"], j, 1] >= frac * ref]
        n_drop += len(c["rows"]) - len(keep)
        c["rows"] = [c["rows"][a] for a in keep]
        c["floor"] = c["floor"][keep] if len(keep) else np.zeros((0, 2))
    log(f"{cam['id']}: {n_drop} candidates dropped as too small (< {frac:g} x the neighbourhood's largest)")
    return out


def learn_colour(cam: dict, cands: list[dict]) -> dict:
    """The redder of two well separated candidates is Aline: collect the two redness samples and fit the per-camera classes."""
    ra, rc = [], []
    for c in cands:
        if len(c["rows"]) != 2:
            continue
        r = c["r"]
        b = [cam["box"][r, j] for j in c["rows"]]
        h = max(b[0][3] - b[0][1], b[1][3] - b[1][1])
        if box_iou(b[0], b[1]) > 0.02 or np.linalg.norm(c["floor"][0] - c["floor"][1]) < 0.8 or h < 80:
            continue
        x = redness(cam["rgb"][r, c["rows"]])
        if not np.isfinite(x).all() or abs(x[0] - x[1]) < 0.05:
            continue
        ra.append(x.max())
        rc.append(x.min())
    if len(ra) < 20:
        raise RuntimeError(f"camera {cam['id']}: only {len(ra)} frames with two separated dancers, no colour model")
    return fit_colour_model(np.array(ra), np.array(rc))


def prepare_camera(cam_id: str) -> dict:
    cam = load_cam(cam_id)
    cands = candidates(cam)
    model = learn_colour(cam, cands)
    log(f"{cam_id}: {sum(1 for c in cands if c['rows'])}/{len(cands)} frames with a candidate, colour model {model}")
    frames = []
    for c in cands:
        r, rows = c["r"], c["rows"]
        if rows:
            b = cam["box"][r, rows]
            frames.append({"c": np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], "h": b[:, 3] - b[:, 1], "llr": llr_aline(redness(cam["rgb"][r, rows]), model)})
        else:
            frames.append({"c": np.zeros((0, 2)), "h": np.zeros(0), "llr": np.zeros(0)})
    return {"cam": cam, "cands": cands, "model": model, "frames": frames}


def assemble(P: dict) -> dict:
    """The tracker's choice (track_pair on P['frames']) as the teachers_<cam>.npz arrays."""
    cam, cands, frames = P["cam"], P["cands"], P["frames"]
    res = track_pair(frames)
    F = len(cam["t"])
    row = np.full((F, 2), -1, np.int16)
    box = np.full((F, 2, 4), np.nan, np.float32)
    kp = np.full((F, 2, 17, 3), np.nan, np.float32)
    conf = np.zeros((F, 2), np.float32)
    rconf = np.zeros((F, 2), np.float32)
    merged = np.zeros((F, 2), bool)
    floor = np.full((F, 2, 2), np.nan, np.float32)
    llr = np.full((F, 2), np.nan, np.float32)
    for i, c in enumerate(cands):
        r = c["r"]
        for role in range(2):
            k = int(res["cand"][i, role])
            if k < 0:
                continue
            j = c["rows"][k]
            row[i, role] = j
            box[i, role] = cam["box"][r, j]
            kp[i, role] = cam["kp"][r, j]
            floor[i, role] = c["floor"][k]
            llr[i, role] = frames[i]["llr"][k]
            rconf[i, role] = res["p"][i, role]
            q = float(np.clip(np.mean(cam["kp"][r, j, :, 2]) / 0.7, 0, 1))
            conf[i, role] = q * rconf[i, role]
        a_, b_ = row[i]
        wide = [(box[i, r_, 2] - box[i, r_, 0]) / max(box[i, r_, 3] - box[i, r_, 1], 1.0) > 0.8 for r_ in range(2) if row[i, r_] >= 0]
        if a_ >= 0 and b_ >= 0 and box_iou(box[i, 0], box[i, 1]) > 0.5:
            merged[i] = True
            conf[i] *= 0.6
        elif any(wide):                                        # one detection holding both bodies
            for r_ in range(2):
                if row[i, r_] >= 0:
                    merged[i, r_] = True
                    conf[i, r_] *= 0.6
    return {"t": cam["t"], "frame_index": cam["fi"].astype(np.int32), "ok": cam["ok"], "row": row, "box": box, "kp": kp, "conf": conf, "merged": merged,
            "role_conf": rconf, "floor": floor, "llr": llr, "size": np.array(cam["size"], np.int32), "attached": np.array(False)}


def cross_penalties(P: dict, Q: dict, out_q: dict, dt: float) -> int:
    """Fill P['frames'][i]['xpen'] (m, 2): how badly candidate k of this camera fits the OTHER camera's confident Carlos / Aline at the same instant (two-view
    reprojection of the shared joints, tri_cost). Candidates with no confident counterpart get 0 (no information). Returns the number of frames judged."""
    cp, cq = P["cam"], Q["cam"]
    tgt = cp["t"] + dt
    j = np.clip(np.searchsorted(cq["t"], tgt), 1, len(cq["t"]) - 1)
    j = np.where(np.abs(cq["t"][j - 1] - tgt) < np.abs(cq["t"][j] - tgt), j - 1, j)
    n = 0

    def proj(cm, idx):
        R, tv = cm["R"][idx], -cm["R"][idx] @ cm["C"][idx]
        return np.c_[R, tv], float(cm["K"][idx][0, 0])
    for i, c in enumerate(P["cands"]):
        f = P["frames"][i]
        f["xpen"] = np.zeros((len(c["rows"]), 2))
        k = int(j[i])
        if not c["rows"] or abs(cq["t"][k] - tgt[i]) > 0.025 or not (cp["ok"][i] and cq["ok"][k]):
            continue
        Pa, fa = proj(cp, i)
        Pb, fb = proj(cq, k)
        done = False
        for role in range(2):
            if out_q["row"][k, role] < 0 or out_q["role_conf"][k, role] < X_MIN_CONF:
                continue
            ub = undistort_norm(out_q["kp"][k, role, :, :2], cq["K"][k], float(cq["k1"][k]))
            cb = out_q["kp"][k, role, :, 2]
            for a, row in enumerate(c["rows"]):
                kp = cp["kp"][c["r"], row]
                ua = undistort_norm(kp[:, :2], cp["K"][i], float(cp["k1"][i]))
                cost = tri_cost(ua, kp[:, 2], Pa, ub, cb, Pb, fa, fb, min_j=5)
                if np.isfinite(cost):
                    f["xpen"][a, role] = min(max(cost - X_OK_PX, 0.0) * X_PEN_PER_PX, X_PEN_MAX)
                    done = True
        n += int(done)
    return n


def run_all(cams=CAMS, passes: int = 2) -> dict:
    """Pass 1: every camera alone (zone, static veto, size, colour, tracking). Pass 2 (both cameras): the other camera's confident dancers judge this camera's
    candidates through two-view geometry (a judge who stands up at the end, a half-hidden bystander, a swapped role), then the tracker runs again."""
    t0 = time.time()
    P = {c: prepare_camera(c) for c in cams}
    out = {c: assemble(P[c]) for c in cams}
    log("pass 1: " + json.dumps({c: [round(float((out[c]["conf"][:, r] >= 0.5).mean()), 3) for r in range(2)] for c in cams}))
    if len(cams) == 2:
        for n in range(passes - 1):
            judged = {}
            for c, o in ((cams[0], cams[1]), (cams[1], cams[0])):
                dt = VIDEO_OFFSET_S if c == "01" else -VIDEO_OFFSET_S
                judged[c] = cross_penalties(P[c], P[o], out[o], dt)
            out = {c: assemble(P[c]) for c in cams}
            log(f"pass {n + 2}: frames judged by the other camera {judged}; " + json.dumps({c: [round(float((out[c]["conf"][:, r] >= 0.5).mean()), 3) for r in range(2)] for c in cams}))
    res = {}
    for c in cams:
        o = out[c]
        np.savez_compressed(PEOPLE / f"teachers_{c}.npz", **o)
        row, conf, merged = o["row"], o["conf"], o["merged"]
        res[c] = {"frames": int(len(row)), "lead_share": round(float((row[:, 0] >= 0).mean()), 3), "follow_share": round(float((row[:, 1] >= 0).mean()), 3),
                  "both_share": round(float(((row[:, 0] >= 0) & (row[:, 1] >= 0)).mean()), 3), "conf_ge_0.5": [round(float((conf[:, r] >= 0.5).mean()), 3) for r in range(2)],
                  "merged_share": round(float(merged.any(1).mean()), 3), "colour_model": P[c]["model"]}
    res["seconds"] = round(time.time() - t0, 1)
    return res



# ------------------------------------------------------------------------------------------------ verification


def tri_cost(ua: np.ndarray, ca: np.ndarray, Pa: np.ndarray, ub: np.ndarray, cb: np.ndarray, Pb: np.ndarray, fa: float, fb: float, min_j: int = 6) -> float:
    """Median two-view reprojection error (px) of the joints that both views see with confidence >= JOINT_CONF: triangulate each joint (DLT on normalised
    rays) and reproject. NaN when fewer than min_j joints are shared. A pair of views of two DIFFERENT people has no consistent 3D point: large."""
    ok = (ca >= JOINT_CONF) & (cb >= JOINT_CONF)
    if int(ok.sum()) < min_j:
        return float("nan")
    errs = []
    for j in np.flatnonzero(ok):
        A = np.stack([ua[j, 0] * Pa[2] - Pa[0], ua[j, 1] * Pa[2] - Pa[1], ub[j, 0] * Pb[2] - Pb[0], ub[j, 1] * Pb[2] - Pb[1]])
        X = np.linalg.svd(A)[2][-1]
        X = X / X[3] if abs(X[3]) > 1e-12 else X
        ea = Pa @ X
        eb = Pb @ X
        if ea[2] <= 0 or eb[2] <= 0:
            errs.append(1e3)
            continue
        errs.append(max(np.linalg.norm(ea[:2] / ea[2] - ua[j]) * fa, np.linalg.norm(eb[:2] / eb[2] - ub[j]) * fb))
    return float(np.median(errs))


def verify() -> dict:
    """Two views, one pair of cameras: at common instants the two roles' joints, taken from both cameras, must triangulate (reprojection of the SAME-role
    pairing small, of the swapped pairing large). Reports per-second runs where the swapped pairing fits better (a role swap in one camera), the shares of
    frames per camera with both dancers, zone radius and so on. People of the audience never reach here (static veto), so a disagreement is a swap or a wrong
    detection."""
    C = {c: load_cam(c) for c in CAMS}
    T = {c: dict(np.load(PEOPLE / f"teachers_{c}.npz")) for c in CAMS}
    a, b = T["01"], T["02"]
    j = np.clip(np.searchsorted(b["t"], a["t"]), 1, len(b["t"]) - 1)
    j = np.where(np.abs(b["t"][j - 1] - a["t"]) < np.abs(b["t"][j] - a["t"]), j - 1, j)
    close = np.abs(b["t"][j] - a["t"]) < 0.025
    rows = []
    for i in np.flatnonzero(close & a["ok"] & b["ok"][j]):
        k = j[i]
        if min(a["conf"][i].min(), b["conf"][k].min()) < 0.3:
            continue
        cams = []
        for cid, idx, o in (("01", i, a), ("02", k, b)):
            cm = C[cid]
            R, tv = cm["R"][idx], -cm["R"][idx] @ cm["C"][idx]
            P = np.c_[R, tv]
            f = cm["K"][idx][0, 0]
            u = [undistort_norm(o["kp"][idx, r, :, :2], cm["K"][idx], float(cm["k1"][idx])) for r in range(2)]
            cams.append((u, [o["kp"][idx, r, :, 2] for r in range(2)], P, f))
        (ua, ca, Pa, fa), (ub, cb, Pb, fb) = cams
        same = [tri_cost(ua[r], ca[r], Pa, ub[r], cb[r], Pb, fa, fb) for r in range(2)]
        swap = [tri_cost(ua[r], ca[r], Pa, ub[1 - r], cb[1 - r], Pb, fa, fb) for r in range(2)]
        if np.isfinite(same).all() and np.isfinite(swap).all():
            rows.append((a["t"][i], max(same), max(swap), float(np.mean(same)), float(np.mean(swap))))
    R_ = np.array(rows)
    ts, s_same, s_swap = R_[:, 0], R_[:, 1], R_[:, 2]
    swapped = (s_swap < s_same) & (s_same > 8.0)
    rep = {"common_frames": int(close.sum()), "compared": int(len(ts)), "same_pairing_px_median_p90": [round(float(np.median(R_[:, 3])), 2), round(float(np.percentile(R_[:, 3], 90)), 2)],
           "swapped_pairing_px_median": round(float(np.median(R_[:, 4])), 2), "swap_better_share": round(float(swapped.mean()), 4)}
    runs, s0 = [], None
    for q in range(len(ts) + 1):
        on = bool(swapped[q]) if q < len(ts) else False
        if on and s0 is None:
            s0 = q
        elif not on and s0 is not None:
            if q - s0 >= 3:
                runs.append([round(float(ts[s0]), 2), round(float(ts[q - 1]), 2), q - s0])
            s0 = None
    rep["swap_better_runs_t0_t1_frames"] = runs
    per_cam = {}
    for c in CAMS:
        o = T[c]
        good = {name: round(float(((o["conf"][:, r] >= 0.5) & o["ok"]).sum() / max(1, o["ok"].sum())), 3) for r, name in ((0, "carlos"), (1, "aline"))}
        good["both"] = round(float(((o["conf"] >= 0.5).all(1) & o["ok"]).sum() / max(1, o["ok"].sum())), 3)
        z = o["floor"][o["ok"]]
        zz = z[np.isfinite(z).all(-1)]
        rr = np.hypot(zz[..., 0], zz[..., 1])
        good["floor_radius_p50_p99_max_m"] = [round(float(np.percentile(rr, 50)), 2), round(float(np.percentile(rr, 99)), 2), round(float(rr.max()), 2)]
        per_cam[c] = good
    rep["per_camera"] = per_cam
    (PEOPLE / "verification.json").write_text(json.dumps(rep, indent=1))
    log(json.dumps(rep))
    return rep


# ------------------------------------------------------------------------------------------------ review


def decode(cam: str, frame: int, width: int = 640):
    import cv2
    cap = cv2.VideoCapture(str(SOURCE / VIDEO[cam]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    s = width / img.shape[1]
    return cv2.resize(img, (width, int(img.shape[0] * s))), s


def review(times: list[float], out: Path | None = None, cols: int = 3) -> list[Path]:
    import cv2
    out = out or WORK / "review" / "people"
    out.mkdir(parents=True, exist_ok=True)
    T = {c: dict(np.load(PEOPLE / f"teachers_{c}.npz")) for c in CAMS}
    pt = {c: np.load(PEOPLE / "pose_tables" / f"{c}.npz") for c in CAMS}
    cams = {c: load_cam(c) for c in CAMS}
    paths = []
    for c in CAMS:
        tiles = []
        for t in times:
            o = T[c]
            if not (o["t"][0] <= t <= o["t"][-1]):
                continue
            i = int(np.argmin(np.abs(o["t"] - t)))
            r = int(o["frame_index"][i])
            d = decode(c, r)
            if d is None:
                continue
            img, s = d
            # every detection thin grey
            for j in range(int(pt[c]["n"][r])):
                b = pt[c]["box"][r, j] * s
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (150, 150, 150), 1)
            for role, col, name in ((0, (60, 200, 60), "C"), (1, (60, 60, 240), "A")):
                if o["row"][i, role] >= 0:
                    b = o["box"][i, role] * s
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
                    cv2.putText(img, f"{name} {o['conf'][i, role]:.2f}", (int(b[0]), max(12, int(b[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cv2.putText(img, f"{c} t={t:.1f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(img)
        if not tiles:
            continue
        h = max(x.shape[0] for x in tiles)
        tiles = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
        p = out / f"teachers_{c}_{int(times[0])}_{int(times[-1])}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p)
        log(f"review {p}")
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.carlos_aline_people", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run", "verify", "review"))
    ap.add_argument("--cams", nargs="*", default=list(CAMS))
    ap.add_argument("--times", nargs="*", type=float, default=[])
    a = ap.parse_args(argv)
    if a.cmd == "run":
        res = run_all(tuple(a.cams))
        (PEOPLE / "teachers.json").write_text(json.dumps(res, indent=1))
        log(json.dumps(res))
    elif a.cmd == "verify":
        verify()
    else:
        review(a.times or [float(x) for x in range(8, 198, 16)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
