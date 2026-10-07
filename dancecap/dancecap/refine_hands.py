"""Hands of the fitted dancers: how well can the video see fingers, and where do the partners touch?

    python -m dancecap.refine_hands survey  <take>   -> hands/survey.npz, survey.json, jobs.json,
                                                       review/hands_survey.png, review/hands_survey_crops.jpg
    .venv-face\\Scripts\\python.exe -m dancecap.hand_worker <take>     hands/jobs.json -> hands/detections.npz
    python -m dancecap.hand_mano_worker run <take> --model wilor|hamer   (pod)  -> hands/mano_<model>.npz
    python -m dancecap.refine_hands detectors <take>  -> hands/tri.npz, detectors.json, review/hands_detectors.{png,jpg}
    python -m dancecap.refine_hands mano <take>       -> hands/mano_fused_<model>.npz, mano.json, review/hands_mano_*.jpg
    python -m dancecap.refine_hands fingers <take> [--mano wilor --tag hybrid] [--workers N]
                                                      -> hands/fingers[_tag]_<role>.npz, fingers[_tag].json,
                                                         smplx_fingers[_tag|_ik]/, review/hands_fingers[_tag].jpg
    python -m dancecap.refine_hands contact <take> [--smplx-dir DIR]
                                                      -> hands/contact.json, contact.npz, smplx/ (refined),
                                                         review/hands_contact*.{png,jpg}
    python -m dancecap.refine_hands compare <take>    -> hands/compare.json, review/hands_compare.png
    python -m dancecap.refine_hands build <take> [--workers N] [--no-review]   (hands_detail: wrist gate + fingers +
                                                      guarded contact -> smplx_hands/, smplx_detail/, hands/metrics.json)
    python -m dancecap.refine_hands review <take>     -> review/hands_contact.jpg, hands_render.jpg, hands_metrics.png
    The heavy steps run in one Runpod session: python -m dancecap.hands_pod run <take>  (Runpod-first, 2026-10-06).

Inputs: smplx/{lead,follow}.npz (contract: 30 Hz grid, world Y-up metres; MAMMA locked-head SMPL-X, hands absolute,
flat_hand_mean=True), cameras/<cam>.npz + world_from_sfm.json, sync/offsets.json (per-camera residual video offsets,
up to ~60 ms - must be applied for hands), frames/<cam>, dancers/<cam>/masks. Everything is parameterised by the take,
so the same commands run on larissa_kadu_38 once its smplx/ exists.

What MAMMA gives us (read from mamma-turing/optimization, 2026-10-06): the 45-D hand poses ARE optimised (stages 2-3,
all pose keys except jaw), driven only by ma_2d's 63 dense points per hand (no hand keypoint detector: hand_joints2d
is None), with no hand prior (rotation_prior_hands_loss is defined but not configured). Jaw is skipped in the
optimiser (zero), expression is not a parameter. The only partner-contact term is MAMMA's intersection_loss in the
4th stage (limbs only, 25 iterations): SDF repulsion of the partner's 512 sampled points inside one's mesh plus a
weak (x0.2) attraction for points ma_2d marks as contact AND already within 1 cm.

Stages here:
  survey     project each hand (21 points: wrist, thumb CMC/MCP/IP/tip, 4 x MCP/PIP/DIP/tip) of the fit into every
             native frame (sync-corrected time): hand length px, in-image, z-buffer visibility, own/partner mask
             overlap; writes the detector jobs (square hand crops + a body box).
  detectors  open hand keypoint detectors (hand_worker in .venv-face: RTMW-x whole-body, RTMPose-m hand, MediaPipe
             Hand Landmarker) vs the fit: detection rate per camera, 2D agreement, robust multi-view triangulation and
             leave-one-camera-out reprojection (the honest accuracy measure when there is no ground truth).
  mano       MANO regressors (WiLoR, HaMeR; gated MANO files, run on a pod) give a full finger pose per view; finger
             rotations are local (view-independent), so they are fused by robust rotation averaging across cameras
             (weights: hand size, own-detector hit, visibility) and smoothed in time; leave-one-camera-out check.
  fingers    IK of the SMPL-X hand pose (45-D, with the model's MANO PCA as a Mahalanobis prior) and the wrist to the
             multi-view keypoints, temporal smoothness; leave-one-camera-out check against the MAMMA hands.
             --mano wilor: the hybrid route (keypoint IK pulled towards the fused WiLoR pose where it is confident).
  compare    every finger route's leave-one-camera-out error side by side.
  contact    hand-partner contact episodes (hand surface to partner surface distance + multi-view 2D evidence +
             persistence/hysteresis), classified hold / near-miss by target region; and a small IK refinement
             (shoulder/elbow/wrist + finger closure) that closes holds without interpenetration while keeping the
             body's reprojection error (MAMMA's own grid landmarks) within tolerance. Gaps are exact distances to the
             partner's surface, signed by the generalized winding number of the partner's closed body (mesh_contact;
             the nearest-vertex-normal sign of the first version miscounted sinks, 2026-10-07 review).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from . import mamma_dyn as md
from .face_common import LABEL, PIDS, ROLES, crop_pad, label, project
from .hand_kin import (HAND21, HAND_EDGES, SIDES, SmplxKin, WRIST, full_pose, hand_vertex_ids)
from .take import Take

OTHER = {"lead": "follow", "follow": "lead"}
SIDE_COL = {"left": (80, 200, 80), "right": (60, 160, 255)}       # BGR
JOB_MIN_LEN_PX = 8.0       # hand length (wrist -> middle tip) below which no crop is made
CROP_SCALE = 2.2           # crop side = CROP_SCALE x the hand's projected extent (min 1.5 x hand length)
OCC_TOL_M = 0.03


# ------------------------------------------------------------------------------------------------ fit + cameras

class WorkTake:
    """Pod-safe stand-in for Take: these stages only use .name and .work, while Take.load also reads the take's audio
    alignment / source-video files, which never go to a pod."""

    def __init__(self, name: str):
        self.name = name
        self.work = md.take_work(name)


def load_take(name: str):
    try:
        return Take.load(name)
    except (FileNotFoundError, OSError):
        return WorkTake(name)


def _npz(p: Path) -> dict:
    """All arrays of an npz in memory: indexing a lazy NpzFile inside a loop re-reads and decompresses the whole array
    on every access (minutes for the 5779-job detector files)."""
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def fit_dir(work: Path) -> Path:
    return work / "smplx"


def load_fit(work: Path, role: str, smplx_dir: Path | None = None) -> dict:
    with np.load((smplx_dir or fit_dir(work)) / f"{role}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def model_dir(d: dict) -> Path:
    from .avatar_texture import model_dir_for
    return model_dir_for(d)


def fit_cams(work: Path) -> list[str]:
    """Cameras of the adopted MAMMA fit (mamma/select.json), else every camera with a cameras/<cam>.npz."""
    sel = work / "mamma" / "select.json"
    if sel.exists():
        cams = json.loads(sel.read_text()).get("cams")
        if cams:
            return [str(c) for c in cams]
    return sorted(p.stem for p in (work / "cameras").glob("*.npz") if p.stem.isdigit())


class Dancer:
    """A fitted dancer: kinematics, grid hand points, and full-mesh vertices on demand."""

    def __init__(self, work: Path, role: str, smplx_dir: Path | None = None, device: str = "cpu"):
        import torch
        self.role = role
        self.d = load_fit(work, role, smplx_dir)
        self.t = np.asarray(self.d["t"], float)
        self.mdir = model_dir(self.d)
        self.kin = SmplxKin(self.mdir, np.asarray(self.d["betas"], float).reshape(-1), device)
        self.pose = torch.tensor(full_pose(self.d), dtype=torch.float64)
        self.transl = torch.tensor(np.asarray(self.d["transl"], float), dtype=torch.float64)
        with torch.no_grad():
            self.out = self.kin.forward(self.pose, self.transl)
            self.hands = {s: self.kin.hand21(self.out, s).numpy() for s in SIDES}   # (T,21,3)
        self.joints = self.out["joints"].numpy()
        self._verts = None

    def verts(self) -> np.ndarray:
        """(T, V, 3) full posed surface (smplx, incl. pose correctives), cached."""
        if self._verts is None:
            from .avatar_uv import Body
            self._verts = Body(self.d, self.mdir, "cpu").verts.astype(np.float64)
        return self._verts

    def at(self, arr: np.ndarray, t: float) -> np.ndarray | None:
        """Linear interpolation of a (T, ...) grid array at reference time t (None outside the grid)."""
        if t < self.t[0] - 1e-3 or t > self.t[-1] + 1e-3:
            return None
        k = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        a = float(np.clip((t - self.t[k]) / (self.t[k + 1] - self.t[k]), 0, 1))
        return (1 - a) * arr[k] + a * arr[k + 1]


def grid_cameras(work: Path, tracks: dict, grid: np.ndarray) -> dict:
    """Per camera on the 30 Hz grid (pinhole, undistorted - MAMMA's convention): K, R, t, ok."""
    out = {}
    for c, tr in tracks.items():
        out[c] = md.resample_cameras(tr.t, tr.K, tr.R, tr.tvec, tr.ok, grid, tr.frame_dt)
    return out


def _mask(p: Path):
    import cv2
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def _frac_in(mask, px: np.ndarray, dilate_px: int = 0) -> float:
    if mask is None:
        return 0.0
    if dilate_px:
        import cv2
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * dilate_px + 1,) * 2, np.uint8)) > 0
    H, W = mask.shape
    u, v = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not ok.any():
        return 0.0
    return float(mask[v[ok], u[ok]].mean())


def _zbuffer_vis(occ: np.ndarray, pts: np.ndarray, K, R, t, k1, W, H, cell: float) -> np.ndarray:
    """Per point: not behind another surface (vertex splat z-buffer in cell-px cells, 3x3 min filter)."""
    import cv2
    pa, za = project(occ, K, R, t, k1)
    Wc, Hc = int(W / cell) + 2, int(H / cell) + 2
    ia = np.round(pa / cell).astype(int)
    ok = (ia[:, 0] >= 0) & (ia[:, 0] < Wc) & (ia[:, 1] >= 0) & (ia[:, 1] < Hc) & (za > 0.1)
    zb = np.full((Hc, Wc), 1e9, np.float32)
    np.minimum.at(zb, (ia[ok, 1], ia[ok, 0]), za[ok].astype(np.float32))
    zb = -cv2.dilate(-zb, np.ones((3, 3), np.uint8))
    pl, zl = project(pts, K, R, t, k1)
    il = np.round(pl / cell).astype(int)
    inb = (il[:, 0] >= 0) & (il[:, 0] < Wc) & (il[:, 1] >= 0) & (il[:, 1] < Hc)
    vis = np.zeros(len(pts), bool)
    vis[inb] = zl[inb] <= zb[il[inb, 1], il[inb, 0]] + OCC_TOL_M
    return vis


# ------------------------------------------------------------------------------------------------ survey

def run_survey(take: Take, cams: list[str] | None = None, smplx_dir: Path | None = None) -> dict:
    """Project every hand into every native frame: size, visibility, crops for the detector worker."""
    import cv2
    work = take.work
    out = work / "hands"
    out.mkdir(parents=True, exist_ok=True)
    cams = cams or fit_cams(work)
    t0 = time.time()
    tracks = md.load_tracks(work, cams)
    dancers = {r: Dancer(work, r, smplx_dir) for r in ROLES}
    hand_ids = {s: hand_vertex_ids(dancers["lead"].mdir, s) for s in SIDES}
    verts = {r: dancers[r].verts() for r in ROLES}
    nv = verts["lead"].shape[1]
    rec, jobs, body_jobs = {}, [], {}
    for c in cams:
        tr = tracks[c]
        W, H = tr.width, tr.height
        dd = work / "dancers" / c
        for i in range(len(tr.t)):
            t = float(tr.t[i])
            K, R, tt, k1 = tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i])
            masks = {r: _mask(dd / "masks" / f"mask_{i:04d}_{PIDS[r]:02d}.png") for r in ROLES}
            for r in ROLES:
                dn = dancers[r]
                if dn.at(dn.t, t) is None:
                    continue
                Vs = dn.at(verts[r], t)
                Vo = dancers[OTHER[r]].at(verts[OTHER[r]], t)
                for s in SIDES:
                    P = dn.at(dn.hands[s], t)
                    px, z = project(P, K, R, tt, k1)
                    L = float(np.linalg.norm(px[0] - px[12]))
                    ext = float(max(px.max(0) - px.min(0)))
                    inimg = bool(((px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)).all()
                                 and (z > 0.1).all())
                    keep = np.ones(nv, bool)
                    keep[hand_ids[s]] = False
                    cell = float(np.clip(L / 6, 1.5, 6))
                    vis = float(_zbuffer_vis(np.concatenate([Vs[keep], Vo]), P, K, R, tt, k1, W, H, cell).mean()) \
                        if inimg else 0.0
                    own = _frac_in(masks[r], px, 2)
                    partner = _frac_in(masks[OTHER[r]], px, 0)
                    key = (r, s, c)
                    rec.setdefault(key, []).append(dict(i=i, t=t, len_px=L, ext_px=ext, inimg=inimg, vis=vis,
                                                        own=own, partner=partner, ok=bool(tr.ok[i]),
                                                        depth=float(np.median(z)), px=px.astype(np.float32)))
                    centre_in = 0 <= px[:, 0].mean() < W and 0 <= px[:, 1].mean() < H
                    if tr.ok[i] and centre_in and L >= JOB_MIN_LEN_PX:
                        side = max(CROP_SCALE * ext, 1.5 * L, 24.0)
                        cx, cy = (px.min(0) + px.max(0)) / 2
                        jobs.append({"role": r, "side": s, "cam": c, "i": i, "t": round(t, 5), "len_px": round(L, 2),
                                     "box": [round(cx - side / 2, 1), round(cy - side / 2, 1),
                                             round(cx + side / 2, 1), round(cy + side / 2, 1)],
                                     "model_px": np.round(px, 1).tolist()})
                # body box for top-down whole-body detectors: the dancer's full mask, else the projected joints
                if tr.ok[i] and (c, i, r) not in body_jobs:
                    m = masks[r]
                    if m is not None and m.any():
                        x, y, bw, bh = cv2.boundingRect(m.astype(np.uint8))
                        box = [x - 0.08 * bw, y - 0.05 * bh, x + 1.08 * bw, y + 1.05 * bh]
                    else:
                        pj, _ = project(dn.at(dn.joints, t), K, R, tt, k1)
                        lo, hi = pj.min(0), pj.max(0)
                        pad = 0.15 * (hi - lo)
                        box = [*(lo - pad), *(hi + pad)]
                    body_jobs[(c, i, r)] = [round(float(v), 1) for v in box]
    arrays, summary = {}, {"take": take.name, "cams": cams, "roles": {}}
    for (r, s, c), rows in rec.items():
        a = {k: np.array([row[k] for row in rows]) for k in rows[0]}
        for k, v in a.items():
            arrays[f"{r}_{s}_{c}_{k}"] = v
        good = a["ok"] & a["inimg"]
        rs = summary["roles"].setdefault(r, {}).setdefault(s, {})
        rs[c] = {"frames": int(len(a["t"])),
                 "len_px_median": round(float(np.median(a["len_px"][good])), 1) if good.any() else None,
                 "len_px_p10": round(float(np.percentile(a["len_px"][good], 10)), 1) if good.any() else None,
                 "len_px_p90": round(float(np.percentile(a["len_px"][good], 90)), 1) if good.any() else None,
                 "in_image": round(float(good.mean()), 3),
                 "visible_ge_0.8": round(float((good & (a["vis"] >= 0.8)).mean()), 3),
                 "inside_partner_mask_ge_0.3": round(float((good & (a["partner"] >= 0.3)).mean()), 3),
                 "own_mask_median": round(float(np.median(a["own"][good])), 3) if good.any() else None,
                 # finger segments are ~ 1/8 of the hand length and ~ 1/10 wide: px per finger width
                 "finger_width_px_median": round(float(np.median(a["len_px"][good]) / 10.5), 2) if good.any() else None}
    summary["jobs"] = len(jobs)
    summary["body_jobs"] = len(body_jobs)
    summary["runtime_s"] = round(time.time() - t0, 1)
    np.savez_compressed(out / "survey.npz", **arrays)
    bj = [{"cam": c, "i": i, "role": r, "box": b} for (c, i, r), b in sorted(body_jobs.items())]
    (out / "jobs.json").write_text(json.dumps({"take": take.name, "jobs": jobs, "body_jobs": bj}))
    review = work / "review"
    review.mkdir(parents=True, exist_ok=True)
    summary["review"] = [str(_survey_plot(arrays, summary, cams, review / "hands_survey.png", take.name)),
                         str(_survey_crops(work, arrays, cams, review / "hands_survey_crops.jpg", take.name))]
    (out / "survey.json").write_text(json.dumps(summary, indent=1))
    return summary


def _survey_plot(arrays, summary, cams, path: Path, take: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for row, r in enumerate(ROLES):
        for col, s in enumerate(SIDES):
            ax = axes[row, col]
            for n, c in enumerate(cams):
                k = f"{r}_{s}_{c}_"
                if k + "t" not in arrays:
                    continue
                t, L = arrays[k + "t"], arrays[k + "len_px"]
                good = arrays[k + "ok"] & arrays[k + "inimg"] & (arrays[k + "vis"] >= 0.8)
                sc = ax.scatter(t[good], np.full(good.sum(), n), c=np.clip(L[good], 0, 80), cmap="viridis", vmin=0,
                                vmax=80, s=9, marker="s")
                ax.scatter(t[~good], np.full((~good).sum(), n), c="#cccccc", s=4, marker="s")
                st = summary["roles"][r][s][c]
                ax.text(t[-1] + 0.05, n, f"len {st['len_px_median']}px  vis {st['visible_ge_0.8'] * 100:.0f}%",
                        va="center", fontsize=8)
            ax.set_yticks(range(len(cams)))
            ax.set_yticklabels([f"cam {c}" for c in cams])
            ax.invert_yaxis()
            ax.set_title(f"{LABEL[r]} {s} hand: wrist-to-middle-fingertip length (px); grey = occluded/outside", fontsize=9)
            fig.colorbar(sc, ax=ax, fraction=0.03, label="hand length px")
    for ax in axes[1]:
        ax.set_xlabel("reference time (s)")
    fig.suptitle(f"{take}: how big the hands are in every camera (fitted SMPL-X hands projected; a finger is ~1/10 "
                 f"of the hand length wide)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def draw_hand(img, px, color, thick=1, r=1):
    import cv2
    p = np.round(px).astype(int)
    for a, b in HAND_EDGES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), color, thick, cv2.LINE_AA)
    for q in p:
        cv2.circle(img, tuple(q), r, color, -1, cv2.LINE_AA)
    return img


def _survey_crops(work: Path, arrays, cams, path: Path, take: str, tile: int = 150) -> Path:
    """Per dancer/hand and camera: the largest visible hand and a median one, native pixels nearest-upscaled, with the
    fitted hand skeleton drawn in a second copy."""
    import cv2
    rows = []
    for r in ROLES:
        for s in SIDES:
            tiles = []
            for c in cams:
                k = f"{r}_{s}_{c}_"
                if k + "t" not in arrays:
                    continue
                good = np.flatnonzero(arrays[k + "ok"] & arrays[k + "inimg"] & (arrays[k + "vis"] >= 0.8))
                if not len(good):
                    continue
                L = arrays[k + "len_px"]
                med = good[np.argsort(L[good])[len(good) // 2]]
                i = int(arrays[k + "i"][med])
                img = cv2.imread(str(work / "frames" / c / f"{i:06d}.jpg"))
                px = arrays[k + "px"][med]
                ext = max(px.max(0) - px.min(0))
                side = max(CROP_SCALE * ext, 1.5 * L[med], 24)
                cx, cy = (px.min(0) + px.max(0)) / 2
                box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
                raw = crop_pad(img, box, tile, cv2.INTER_NEAREST)
                sc = tile / (box[2] - box[0])
                ov = raw.copy()
                draw_hand(ov, (px - [box[0], box[1]]) * sc, SIDE_COL[s], 1, 2)
                label(raw, f"cam {c}", (3, 13), scale=0.4)
                label(raw, f"{L[med]:.0f}px", (3, tile - 5), scale=0.4)
                tiles.append(np.concatenate([raw, ov], 0))
            if tiles:
                row = np.concatenate(tiles, 1)
                head = np.full((20, row.shape[1], 3), 20, np.uint8)
                label(head, f"{LABEL[r]} {s} hand: median-size visible view per camera (top: video, bottom: fitted "
                            f"MAMMA hand)", (5, 14), scale=0.45)
                rows.append(np.concatenate([head, row], 0))
    w = max(x.shape[1] for x in rows)
    sheet = np.concatenate([np.pad(x, ((0, 4), (0, w - x.shape[1]), (0, 0))) for x in rows], 0)
    head = np.full((28, w, 3), 10, np.uint8)
    label(head, f"{take}: hands as the cameras see them (native pixels, nearest-neighbour upscale)", (6, 19), scale=0.5)
    cv2.imwrite(str(path), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


# ------------------------------------------------------------------------------------------------ detectors

DETECTORS = ("rtmw", "rtmh", "mp")
MANO_MODELS = ("wilor", "hamer")   # MANO regressors (hand_mano_worker, pod); their 2D joints also act as detectors
DET_LABEL = {"rtmw": "RTMW-x whole-body", "rtmh": "RTMPose-m hand", "mp": "MediaPipe Hand Landmarker",
             "fused": "fused (best detector per view)", "wilor": "WiLoR (MANO regressor)",
             "hamer": "HaMeR (MANO regressor)"}


def mano_models_present(work: Path) -> tuple:
    return tuple(m for m in MANO_MODELS if (work / "hands" / f"mano_{m}.npz").exists())
RTMW_HAND = {"left": slice(91, 112), "right": slice(112, 133)}


@dataclass
class DetConfig:
    # confidence floors (detector units): rtmlib SimCC scores for RTMW / RTMPose, MediaPipe handedness score
    min_conf: dict = field(default_factory=lambda: {"rtmw": 0.5, "rtmh": 0.3, "mp": 0.5})
    gate_len: float = 1.0          # detection kept when its wrist is within gate_len x hand length of the fit's wrist
    tri_inlier_len: float = 0.35   # multi-view inlier: reprojection within this x hand length (median over points)
    min_views: int = 2
    mano_floor: float = 0.3        # confidence of a MANO regressor's joints when its own detector did not find the hand


def native_detections(work: Path, cfg: DetConfig | None = None) -> dict:
    """hands/detections.npz -> {(det, role, side, cam): {"i": (n,), "kp": (n,21,2) distorted px, "w": (n,21)}}.
    w = per-point confidence in [0, 1] (0 = unusable): detector score >= floor, wrist gated against the fit."""
    cfg = cfg or DetConfig()
    z = _npz(work / "hands" / "detections.npz")
    jobs = json.loads((work / "hands" / "jobs.json").read_text())["jobs"]
    model = {(j["cam"], j["i"], j["role"], j["side"]): (np.asarray(j["model_px"], float), j["len_px"]) for j in jobs}
    out = {}

    def put(det, r, s, c, i, kp, w):
        key = (det, r, s, c)
        m = model.get((c, int(i), r, s))
        if m is None:
            return
        mpx, L = m
        if not np.isfinite(kp).all() or np.linalg.norm(kp[0] - mpx[0]) > cfg.gate_len * max(L, 8.0):
            w = np.zeros(21)
        out.setdefault(key, {"i": [], "kp": [], "w": [], "model": [], "len": []})
        d = out[key]
        d["i"].append(int(i)), d["kp"].append(np.nan_to_num(kp)), d["w"].append(w)
        d["model"].append(mpx), d["len"].append(L)

    for k in range(len(z["job_cam"])):
        c, i, r, s = str(z["job_cam"][k]), int(z["job_i"][k]), str(z["job_role"][k]), str(z["job_side"][k])
        sc = z["rtmh_score"][k]
        put("rtmh", r, s, c, i, z["rtmh_kp"][k].astype(float),
            np.where(sc >= cfg.min_conf["rtmh"], np.clip(sc, 0, 1), 0.0))
        if z["mp_ok"][k] and z["mp_score"][k] >= cfg.min_conf["mp"]:
            put("mp", r, s, c, i, z["mp_kp"][k][:, :2].astype(float), np.full(21, float(z["mp_score"][k])))
        else:
            put("mp", r, s, c, i, np.full((21, 2), np.nan), np.zeros(21))
    for k in range(len(z["body_cam"])):
        c, i, r = str(z["body_cam"][k]), int(z["body_i"][k]), str(z["body_role"][k])
        for s in SIDES:
            kp, sc = z["rtmw_kp"][k][RTMW_HAND[s]].astype(float), z["rtmw_score"][k][RTMW_HAND[s]]
            put("rtmw", r, s, c, i, kp, np.where(sc >= cfg.min_conf["rtmw"], np.clip(sc / 2.0, 0, 1), 0.0))
    # MANO regressors: their projected joints are a 2D detector too. They always return a hand (a prior-shaped guess
    # when the hand is hidden), so confidence = the WiLoR YOLO detection when it found this hand, else a flat floor.
    for m in mano_models_present(work):
        zm = _npz(work / "hands" / f"mano_{m}.npz")
        for k in range(len(zm["job_cam"])):
            if not zm["ok"][k]:
                continue
            det = bool(zm["box_src"][k] == 1)
            wv = float(np.clip(zm["det_conf"][k], 0.3, 1.0)) if det else cfg.mano_floor
            put(m, str(zm["job_role"][k]), str(zm["job_side"][k]), str(zm["job_cam"][k]), int(zm["job_i"][k]),
                zm["kp2d"][k].astype(float), np.full(21, wv))
    return {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in out.items()}


def to_grid(nat: dict, tracks: dict, grid: np.ndarray) -> dict:
    """Native detections -> undistorted pixels on the 30 Hz grid: {(det, role, side, cam): (kp (T,21,2), w (T,21))}.
    Linear between two consecutive detected frames (gap <= 1.5 frames, confidence = min of both), else the nearest
    detected frame within one frame interval (confidence x 0.7); else missing (w 0)."""
    out = {}
    for key, d in nat.items():
        c = key[3]
        tr = tracks[c]
        F = len(tr.t)
        kp = np.zeros((F, 21, 2))
        w = np.zeros((F, 21))
        ii = d["i"]
        kp[ii], w[ii] = d["kp"], d["w"]
        for f in np.unique(ii):
            if tr.k1[f] != 0:
                kp[f] = md.undistort_simple_radial(kp[f], tr.K[f], float(tr.k1[f]))
        i0, i1, a, d0, d1, gap = md._brackets(tr.t, grid, tr.frame_dt)
        two = (w[i0] > 0) & (w[i1] > 0) & (gap[:, None] <= 1.5)
        lin = (1 - a)[:, None, None] * kp[i0] + a[:, None, None] * kp[i1]
        near0 = (w[i0] > 0) & ((d0 <= d1)[:, None] | (w[i1] <= 0)) & (d0[:, None] <= 1.0 + 1e-6)
        near1 = (w[i1] > 0) & ~near0 & (d1[:, None] <= 1.0 + 1e-6)
        K = np.where(two[..., None], lin, np.where(near0[..., None], kp[i0], kp[i1]))
        W = np.where(two, np.minimum(w[i0], w[i1]), np.where(near0, 0.7 * w[i0], np.where(near1, 0.7 * w[i1], 0.0)))
        out[key] = (K, W)
    return out


def _dlt(Ps: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted DLT: Ps (V,3,4), x (V,N,2), w (V,N) -> (N,3) (NaN where < 2 views)."""
    V, N = x.shape[:2]
    out = np.full((N, 3), np.nan)
    for n in range(N):
        ok = w[:, n] > 0
        if ok.sum() < 2:
            continue
        A = []
        for v in np.flatnonzero(ok):
            P = Ps[v]
            A.append(w[v, n] * (x[v, n, 0] * P[2] - P[0]))
            A.append(w[v, n] * (x[v, n, 1] * P[2] - P[1]))
        _, _, Vt = np.linalg.svd(np.asarray(A))
        X = Vt[-1]
        if abs(X[3]) > 1e-12:
            out[n] = X[:3] / X[3]
    return out


def _proj(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
    y = Xh @ P.T
    return y[:, :2] / np.maximum(y[:, 2:3], 1e-6)


def triangulate_hand(Ps: np.ndarray, x: np.ndarray, w: np.ndarray, L: np.ndarray, cfg: DetConfig):
    """Robust multi-view triangulation of one hand at one instant. Ps (V,3,4) pinhole projections, x (V,21,2), w (V,21),
    L (V,) hand length px per view. Views whose median reprojection error exceeds tri_inlier_len x L are dropped
    one at a time (worst first). -> X (21,3) or None, inlier mask (V,)"""
    inl = (w > 0).sum(1) >= 6
    while inl.sum() >= cfg.min_views:
        X = _dlt(Ps[inl], x[inl], w[inl])
        if np.isnan(X).all():
            return None, inl
        err = np.full(len(Ps), np.inf)
        for v in np.flatnonzero(inl):
            ok = (w[v] > 0) & np.isfinite(X).all(1)
            if ok.sum() >= 4:
                err[v] = np.median(np.linalg.norm(_proj(Ps[v], X[ok]) - x[v, ok], axis=1)) / max(L[v], 4.0)
        worst = int(np.argmax(np.where(inl, err, -np.inf)))
        if err[worst] <= cfg.tri_inlier_len:
            return X, inl
        inl = inl.copy()
        inl[worst] = False
    return None, inl


def projections(gc: dict, cams: list[str], k: int) -> np.ndarray:
    Ps = []
    for c in cams:
        K, R, t = gc[c]["K"][k], gc[c]["R"][k], gc[c]["t"][k]
        Ps.append(K @ np.concatenate([R, t[:, None]], 1))
    return np.asarray(Ps)


def fuse_views(G: dict, dets=DETECTORS) -> dict:
    """Per view and frame: the detector with the highest mean confidence (>= 15 confident points) -> 'fused'."""
    keys = {(r, s, c) for (_, r, s, c) in G}
    out = {}
    for r, s, c in keys:
        cands = [G[(d, r, s, c)] for d in dets if (d, r, s, c) in G]
        if not cands:
            continue
        score = np.stack([np.where((w > 0).sum(1) >= 15, w.mean(1), -1) for _, w in cands], 0)   # (D,T)
        best = score.argmax(0)
        kp = np.stack([cands[b][0][k] for k, b in enumerate(best)])
        w = np.stack([cands[b][1][k] if score[b, k] > 0 else np.zeros(21) for k, b in enumerate(best)])
        out[("fused", r, s, c)] = (kp, w)
    return out


def _hand_len_px(gc, c, k, X21) -> float:
    K, R, t = gc[c]["K"][k], gc[c]["R"][k], gc[c]["t"][k]
    P = K @ np.concatenate([R, t[:, None]], 1)
    p = _proj(P, X21[[0, 12]])
    return float(np.linalg.norm(p[0] - p[1]))


def evaluate_grid(G: dict, dancers: dict, gc: dict, cams: list[str], cfg: DetConfig, det: str) -> dict:
    """Triangulate every hand on every grid frame from detector `det` and score it against the views it did not use.

    Returns per (role, side): X (T,21,3), nviews (T,), and per view-frame records of the leave-one-camera-out test
    (columns: frame, cam index, hand length px, tri px, fit px, tri articulation px, fit articulation px, n other
    views): tri = triangulated from the other views -> this view's detection, fit = the MAMMA hand -> this view's
    detection; articulation = the same after re-anchoring both on the detected wrist (finger shape only)."""
    res = {}
    T = len(dancers["lead"].t)
    for r in ROLES:
        for s in SIDES:
            if not any((det, r, s, c) in G for c in cams):
                continue
            X = np.full((T, 21, 3), np.nan)
            nv = np.zeros(T, int)
            recs = []
            model = dancers[r].hands[s]
            for k in range(T):
                vs = [c for c in cams if (det, r, s, c) in G and gc[c]["ok"][k]]
                if len(vs) < 2:
                    continue
                Ps = projections(gc, vs, k)
                x = np.stack([G[(det, r, s, c)][0][k] for c in vs])
                w = np.stack([G[(det, r, s, c)][1][k] for c in vs])
                L = np.array([_hand_len_px(gc, c, k, model[k]) for c in vs])
                Xk, inl = triangulate_hand(Ps, x, w, L, cfg)
                if Xk is None:
                    continue
                X[k], nv[k] = Xk, int(inl.sum())
                for v in np.flatnonzero(inl):
                    others = inl.copy()
                    others[v] = False
                    if others.sum() < 2:
                        continue
                    Xo = _dlt(Ps[others], x[others], w[others])
                    ok = (w[v] > 0) & np.isfinite(Xo).all(1)
                    if ok.sum() < 8:
                        continue
                    pt = _proj(Ps[v], Xo)
                    pf = _proj(Ps[v], model[k])
                    e_t = np.linalg.norm(pt[ok] - x[v, ok], axis=1)
                    e_f = np.linalg.norm(pf[ok] - x[v, ok], axis=1)
                    okf = ok.copy()
                    okf[0] = False
                    a_t = np.linalg.norm((pt - pt[0] + x[v, 0])[okf] - x[v, okf], axis=1)
                    a_f = np.linalg.norm((pf - pf[0] + x[v, 0])[okf] - x[v, okf], axis=1)
                    recs.append((k, cams.index(vs[v]), L[v], np.median(e_t), np.median(e_f),
                                 np.median(a_t), np.median(a_f), int(others.sum())))
            rec = np.array(recs, float) if recs else np.zeros((0, 8))
            res[(r, s)] = {"X": X, "nviews": nv, "loo": rec}
    return res


def _bone_ratio(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Per frame: median ratio of triangulated to model finger bone lengths (20 hand edges)."""
    e = np.array(HAND_EDGES)
    lx = np.linalg.norm(X[:, e[:, 0]] - X[:, e[:, 1]], axis=-1)
    lm = np.linalg.norm(M[:, e[:, 0]] - M[:, e[:, 1]], axis=-1)
    with np.errstate(all="ignore"):
        return np.nanmedian(lx / np.maximum(lm, 1e-6), axis=1)


def loo_summary(loo: np.ndarray) -> dict:
    out = {}
    if not len(loo):
        return out
    Lv = loo[:, 2]
    for name, sel in (("all", np.ones(len(loo), bool)), ("len_ge_30px", Lv >= 30), ("len_lt_30px", Lv < 30)):
        if sel.sum() < 5:
            continue
        out[name] = {
            "n": int(sel.sum()),
            "tri_px": round(float(np.median(loo[sel, 3])), 2), "fit_px": round(float(np.median(loo[sel, 4])), 2),
            "tri_len": round(float(np.median(loo[sel, 3] / Lv[sel])), 3),
            "fit_len": round(float(np.median(loo[sel, 4] / Lv[sel])), 3),
            "tri_art_len": round(float(np.median(loo[sel, 5] / Lv[sel])), 3),
            "fit_art_len": round(float(np.median(loo[sel, 6] / Lv[sel])), 3),
            "tri_beats_fit": round(float((loo[sel, 3] < loo[sel, 4]).mean()), 3),
            "tri_art_beats_fit": round(float((loo[sel, 5] < loo[sel, 6]).mean()), 3)}
    return out


def run_detectors(take: Take, cfg: DetConfig | None = None, smplx_dir: Path | None = None) -> dict:
    cfg = cfg or DetConfig()
    work = take.work
    out = work / "hands"
    cams = fit_cams(work)
    t0 = time.time()
    tracks = md.load_tracks(work, cams)
    dancers = {r: Dancer(work, r, smplx_dir) for r in ROLES}
    grid = dancers["lead"].t
    gc = grid_cameras(work, tracks, grid)
    nat = native_detections(work, cfg)
    G = to_grid(nat, tracks, grid)
    G.update(fuse_views(G))
    sv = _npz(out / "survey.npz")
    summary = {"take": take.name, "cams": cams, "config": asdict(cfg), "native": {}, "grid": {}}
    # --- native frames: detection rate and agreement with the fit, split by visibility
    for (det, r, s, c), d in sorted(nat.items()):
        k = f"{r}_{s}_{c}_"
        vis_by_i = dict(zip(sv[k + "i"].tolist(), sv[k + "vis"].tolist()))
        vis = np.array([vis_by_i.get(int(i), 0.0) for i in d["i"]])
        det_ok = (d["w"] > 0).sum(1) >= 15
        L = d["len"]
        dist = np.linalg.norm(d["kp"] - d["model"], axis=-1).mean(1) / np.maximum(L, 4)
        art = np.linalg.norm((d["kp"] - d["kp"][:, :1]) - (d["model"] - d["model"][:, :1]), axis=-1)[:, 1:].mean(1) \
            / np.maximum(L, 4)
        e = summary["native"].setdefault(det, {}).setdefault(r, {}).setdefault(s, {})
        vv = vis >= 0.8
        e[c] = {"jobs": int(len(L)), "len_px_median": round(float(np.median(L)), 1),
                "detected": round(float(det_ok.mean()), 3),
                "detected_visible": round(float(det_ok[vv].mean()), 3) if vv.any() else None,
                "detected_occluded": round(float(det_ok[~vv].mean()), 3) if (~vv).any() else None,
                "fit_dist_len_median": round(float(np.median(dist[det_ok])), 3) if det_ok.any() else None,
                "fit_articulation_len_median": round(float(np.median(art[det_ok])), 3) if det_ok.any() else None}
    # --- grid: robust triangulation + leave-one-camera-out
    arrays = {"grid_t": grid}
    for det in (*DETECTORS, *mano_models_present(work), "fused"):
        res = evaluate_grid(G, dancers, gc, cams, cfg, det)
        gs = summary["grid"].setdefault(det, {})
        for (r, s), v in res.items():
            X, nv, loo = v["X"], v["nviews"], v["loo"]
            M = dancers[r].hands[s]
            ok = nv >= 2
            d3 = np.linalg.norm(X - M, axis=-1)                                   # (T,21) m
            art3 = np.linalg.norm((X - X[:, :1]) - (M - M[:, :1]), axis=-1)[:, 1:]
            br = _bone_ratio(X, M)
            arrays[f"{det}_{r}_{s}_X"], arrays[f"{det}_{r}_{s}_nviews"] = X, nv
            arrays[f"{det}_{r}_{s}_loo"] = loo
            entry = {"frames_triangulated": round(float(ok.mean()), 3),
                     "views_median": float(np.median(nv[ok])) if ok.any() else 0,
                     "wrist_vs_fit_cm_median": round(float(np.nanmedian(d3[ok, 0]) * 100), 2) if ok.any() else None,
                     "joints_vs_fit_cm_median": round(float(np.nanmedian(d3[ok]) * 100), 2) if ok.any() else None,
                     "articulation_vs_fit_cm_median": round(float(np.nanmedian(art3[ok]) * 100), 2) if ok.any() else None,
                     "bone_length_ratio_median": round(float(np.nanmedian(br[ok])), 3) if ok.any() else None,
                     "bone_length_ratio_iqr": [round(float(np.nanpercentile(br[ok], q)), 3) for q in (25, 75)]
                     if ok.any() else None,
                     "loo": loo_summary(loo)}
            gs.setdefault(r, {})[s] = entry
    # keep the grid observations for the fingers stage
    obs = {}
    for (det, r, s, c), (kp, w) in G.items():
        obs[f"{det}|{r}|{s}|{c}|kp"], obs[f"{det}|{r}|{s}|{c}|w"] = kp.astype(np.float32), w.astype(np.float32)
    np.savez_compressed(out / "grid_obs.npz", **obs)
    np.savez_compressed(out / "tri.npz", **arrays)
    summary["runtime_s"] = round(time.time() - t0, 1)
    review = work / "review"
    summary["review"] = [str(_detectors_plot(summary, arrays, cams, review / "hands_detectors.png", take.name)),
                         str(_detectors_sheet(work, nat, sv, cams, review / "hands_detectors.jpg", take.name))]
    (out / "detectors.json").write_text(json.dumps(summary, indent=1))
    return summary


def _detectors_plot(summary, arrays, cams, path: Path, take: str) -> Path:
    """Top: detection rate per camera (visible hands) for each detector, with the median hand length. Bottom: leave-
    one-camera-out error of the triangulated hand vs the MAMMA hand (in hand lengths) against hand size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    cols = {"rtmw": "#1f77b4", "rtmh": "#ff7f0e", "mp": "#2ca02c", "fused": "#9467bd", "wilor": "#d62728",
            "hamer": "#8c564b"}
    dets = [d for d in (*DETECTORS, *MANO_MODELS) if d in summary["native"]]
    bw = 0.8 / max(len(dets), 1)
    n = 0
    for r in ROLES:
        for s in SIDES:
            ax = axes[0, n]
            x = np.arange(len(cams))
            for j, det in enumerate(dets):
                e = summary["native"].get(det, {}).get(r, {}).get(s, {})
                y = [e.get(c, {}).get("detected_visible") or 0 for c in cams]
                ax.bar(x + (j - (len(dets) - 1) / 2) * bw, y, bw, color=cols[det], label=DET_LABEL[det])
            lens = [summary["native"]["rtmh"][r][s].get(c, {}).get("len_px_median", 0) for c in cams]
            ax.set_xticks(x)
            ax.set_xticklabels([f"{c}\n{L:.0f}px" for c, L in zip(cams, lens)], fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(f"{LABEL[r]} {s}: detected (>=15 confident pts),\nvisible hands; cam / median hand length",
                         fontsize=9)
            if n == 0:
                ax.legend(fontsize=7)
            ax = axes[1, n]
            for det in (*dets, "fused"):
                loo = arrays.get(f"{det}_{r}_{s}_loo")
                if loo is None or not len(loo):
                    continue
                L = loo[:, 2]
                bins = np.array([0, 15, 25, 35, 50, 80, 200])
                idx = np.digitize(L, bins) - 1
                med_t = [np.median(loo[idx == b, 5] / L[idx == b]) if (idx == b).sum() >= 5 else np.nan
                         for b in range(len(bins) - 1)]
                ax.plot((bins[:-1] + bins[1:]) / 2, med_t, "-o", color=cols[det], ms=4, label=f"{det}: tri (others)")
                if det == "fused":
                    med_f = [np.median(loo[idx == b, 6] / L[idx == b]) if (idx == b).sum() >= 5 else np.nan
                             for b in range(len(bins) - 1)]
                    ax.plot((bins[:-1] + bins[1:]) / 2, med_f, "--k", lw=2, label="MAMMA hand")
            ax.set_xlabel("hand length in the held-out view (px)")
            ax.set_ylabel("finger-shape error / hand length")
            ax.set_ylim(0, 0.6)
            ax.set_title("leave-one-camera-out: predicted vs detected finger\nshape (wrist re-anchored); lower = better",
                         fontsize=9)
            if n == 0:
                ax.legend(fontsize=7)
            n += 1
    fig.suptitle(f"{take}: open hand keypoint detectors on our footage")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _detectors_sheet(work: Path, nat: dict, sv, cams: list[str], path: Path, take: str, tile: int = 170,
                     per_hand: int = 6) -> Path:
    """For each hand: the largest visible views (one per camera, up to per_hand), crop | RTMW | RTMPose | MediaPipe |
    MAMMA fit, so the detectors can be judged by eye."""
    import cv2
    rows = []
    dets = [d for d in (*DETECTORS, *MANO_MODELS) if any(key[0] == d for key in nat)]
    for r in ROLES:
        for s in SIDES:
            cand = []
            for c in cams:
                k = f"{r}_{s}_{c}_"
                if k + "t" not in sv:
                    continue
                good = np.flatnonzero(sv[k + "ok"] & sv[k + "inimg"] & (sv[k + "vis"] >= 0.8))
                if len(good):
                    b = good[np.argmax(sv[k + "len_px"][good])]
                    cand.append((float(sv[k + "len_px"][b]), c, int(sv[k + "i"][b])))
            cand.sort(reverse=True)
            for L, c, i in cand[:per_hand]:
                img = cv2.imread(str(work / "frames" / c / f"{i:06d}.jpg"))
                d0 = nat.get(("rtmh", r, s, c))
                if d0 is None or i not in d0["i"].tolist():
                    continue
                j = d0["i"].tolist().index(i)
                mpx = d0["model"][j]
                ext = max(mpx.max(0) - mpx.min(0))
                side = max(CROP_SCALE * ext, 1.5 * L, 24)
                cx, cy = (mpx.min(0) + mpx.max(0)) / 2
                box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
                sc = tile / (box[2] - box[0])
                base = crop_pad(img, box, tile, cv2.INTER_CUBIC)
                tiles = [label(base.copy(), f"cam {c} f{i} {L:.0f}px", (3, 13), scale=0.38)]
                for det in dets:
                    t = base.copy()
                    d = nat.get((det, r, s, c))
                    if d is not None and i in d["i"].tolist():
                        jj = d["i"].tolist().index(i)
                        if (d["w"][jj] > 0).sum() >= 15:
                            draw_hand(t, (d["kp"][jj] - [box[0], box[1]]) * sc, (0, 255, 255), 1, 2)
                            label(t, f"{det} conf {d['w'][jj].mean():.2f}", (3, 13), scale=0.38)
                        else:
                            label(t, f"{det}: none", (3, 13), (120, 120, 255), scale=0.38)
                    tiles.append(t)
                t = base.copy()
                draw_hand(t, (mpx - [box[0], box[1]]) * sc, SIDE_COL[s], 1, 2)
                label(t, "MAMMA fit", (3, 13), scale=0.38)
                tiles.append(t)
                rows.append(np.concatenate(tiles, 1))
            if cand:
                hdr = np.full((20, tile * (2 + len(dets)), 3), 30, np.uint8)
                label(hdr, f"{LABEL[r]} {s} hand (rows below: largest visible view per camera)", (5, 14), scale=0.45)
                rows.insert(len(rows) - min(len(cand), per_hand), hdr)
    w = max(x.shape[1] for x in rows)
    sheet = np.concatenate([np.pad(x, ((0, 2), (0, w - x.shape[1]), (0, 0))) for x in rows], 0)
    head = np.full((28, w, 3), 10, np.uint8)
    label(head, f"{take}: video crop | {' | '.join(DET_LABEL[d] for d in dets)} | MAMMA hand (largest visible views)",
          (6, 19), scale=0.5)
    cv2.imwrite(str(path), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


# ------------------------------------------------------------------------------------------------ fingers (IK)

@dataclass
class FingerConfig:
    detector: str = "fused"
    sigma_len: float = 0.10       # 2D noise per point = sigma_len x hand length px in that view
    min_len_px: float = 12.0      # views where the hand is smaller than this are ignored
    prior: float = 0.02           # |PCA coefficients|^2 (MANO training distribution, whitened)
    to_init: float = 0.0          # |theta - theta_MAMMA|^2 (rad^2)
    smooth: float = 30.0          # |theta_k+1 - theta_k|^2 (30 Hz)
    accel: float = 30.0           # |theta_k+1 - 2 theta_k + theta_k-1|^2
    wrist_prior: float = 5.0      # |wrist - wrist_MAMMA|^2
    wrist_smooth: float = 30.0
    iters: int = 250
    lr: float = 0.03
    # hybrid route: pull the fingers towards the multi-view fused MANO regressor pose (refine_hands mano), weighted per
    # frame by the fusion confidence (0..1). "" = keypoints only.
    mano_model: str = ""
    mano_weight: float = 1.0
    tag: str = ""                 # output names: fingers[_tag]_<role>.npz, fingers[_tag].json, smplx_fingers[_tag]/
    # npz with <role>_<side>_wrist (T,3) local wrist axis-angles (hands_detail wrist gate): the IK starts from and is
    # regularised towards these wrists instead of MAMMA's. "" = MAMMA's wrists.
    wrist_init: str = ""


_WRIST_INIT: dict = {}


def wrist_init_for(path: str, role: str, side: str) -> np.ndarray | None:
    """(T,3) gated wrist rotations from FingerConfig.wrist_init (cached per path), None when absent."""
    if not path:
        return None
    if path not in _WRIST_INIT:
        p = Path(path)
        _WRIST_INIT[path] = _npz(p) if p.exists() else {}
    v = _WRIST_INIT[path].get(f"{role}_{side}_wrist")
    return None if v is None else np.asarray(v, float)


class HandPrior:
    """MANO pose space from the SMPL-X model file: theta_abs = hands_mean + c @ components, c ~ N(0, I) (the file's
    hands_coeffs have unit variance)."""

    def __init__(self, model_dir: Path, side: str):
        from .hand_kin import model_npz
        m = np.load(model_npz(model_dir), allow_pickle=True)
        cs = "l" if side == "left" else "r"
        self.mean = np.asarray(m["hands_mean" + cs], float)
        C = np.asarray(m["hands_components" + cs], float)
        self.pinv = np.linalg.pinv(C)                     # theta -> coefficients
        self.C = C

    def coeffs(self, theta: np.ndarray) -> np.ndarray:
        return (theta - self.mean) @ self.pinv


def _robust(torch, r2):
    """Geman-McClure on squared normalised residuals (scale 1)."""
    return r2 / (r2 + 1.0)


def fit_hand(dn: Dancer, side: str, obs: dict, gc: dict, cams: list[str], cfg: FingerConfig,
             holdout: str | None = None, device: str = "cpu", target: tuple | None = None) -> dict:
    """IK of one hand over the whole take. obs {cam: (kp (T,21,2) undistorted px, w (T,21))}. target = (theta (T,15,3)
    with NaN where unknown, weight (T,) in 0..1): the fused MANO-regressor pose for the hybrid route (cfg.mano_weight).
    Returns theta (T,15,3), wrist (T,3), per-frame observation count, and the final per-term losses."""
    import torch
    T = len(dn.t)
    kin = dn.kin
    base = kin.hand_base(dn.out, side)
    f0 = 25 if side == "left" else 40
    th0 = dn.pose[:, f0:f0 + 15].clone()
    wr0 = dn.pose[:, WRIST[side]].clone()
    wi = wrist_init_for(cfg.wrist_init, dn.role, side)
    if wi is not None and wi.shape == tuple(wr0.shape):
        wr0 = torch.tensor(wi, dtype=torch.float64)
    prior = HandPrior(dn.mdir, side)
    pmean = torch.tensor(prior.mean, dtype=torch.float64)
    pinv = torch.tensor(prior.pinv, dtype=torch.float64)
    # observations -> tensors (views x T x 21)
    vs = [c for c in cams if c in obs and c != holdout]
    Pm, X, W, S = [], [], [], []
    M0 = dn.hands[side]
    for c in vs:
        kp, w = obs[c]
        K, R, t = gc[c]["K"], gc[c]["R"], gc[c]["t"]
        P = np.einsum("tij,tjk->tik", K, np.concatenate([R, t[..., None]], -1))       # (T,3,4)
        Xh = np.concatenate([M0[:, [0, 12]], np.ones((T, 2, 1))], -1)
        y = np.einsum("tij,tnj->tni", P, Xh)
        p2 = y[..., :2] / np.maximum(y[..., 2:], 1e-6)
        L = np.linalg.norm(p2[:, 0] - p2[:, 1], axis=-1)                                 # (T,) hand length px
        ok = gc[c]["ok"] & (L >= cfg.min_len_px)
        Pm.append(P), X.append(kp), W.append(np.where(ok[:, None], w, 0.0)), S.append(np.maximum(cfg.sigma_len * L, 1.0))
    if not vs:
        return {"theta": th0.numpy(), "wrist": wr0.numpy(), "n_obs": np.zeros(T, int), "views": []}
    f64 = dict(dtype=torch.float64, device=device)
    Pm, X, W, S = (torch.tensor(np.stack(a), **f64) for a in (Pm, X, W, S))
    n_obs = (W.sum(-1) > 0).sum(0).numpy()                                               # (T,) views per frame
    tg = tw = None
    if target is not None and cfg.mano_weight > 0:
        tth, tww = np.asarray(target[0], float), np.asarray(target[1], float)
        okt = np.isfinite(tth).all((1, 2)) & (tww > 0)
        tg = torch.tensor(np.where(okt[:, None, None], tth, 0.0), **f64)
        tw = torch.tensor(np.where(okt, np.clip(tww, 0, 1), 0.0), **f64)
        th0 = torch.where(torch.tensor(okt)[:, None, None], tg, th0)   # start from the fused pose where known
    th = th0.clone().requires_grad_(True)
    wr = wr0.clone().requires_grad_(True)
    opt = torch.optim.Adam([th, wr], lr=cfg.lr)
    wsum = W.sum().clamp(min=1.0)
    for it in range(cfg.iters):
        opt.zero_grad()
        J = kin.hand_fk(base, wr, th)                                                     # (T,21,3)
        Jh = torch.cat([J, torch.ones_like(J[..., :1])], -1)
        y = torch.einsum("vtij,tnj->vtni", Pm, Jh)
        p2 = y[..., :2] / y[..., 2:].clamp(min=1e-6)
        r2 = ((p2 - X) ** 2).sum(-1) / S[..., None] ** 2
        l_rep = (W * _robust(torch, r2)).sum() / wsum
        c = (th.reshape(T, 45) - pmean) @ pinv
        l_pri = cfg.prior * (c ** 2).sum(1).mean()
        l_init = cfg.to_init * ((th - th0) ** 2).sum((1, 2)).mean()
        d1 = th[1:] - th[:-1]
        d2 = th[2:] - 2 * th[1:-1] + th[:-2]
        l_s = cfg.smooth * (d1 ** 2).sum((1, 2)).mean() + cfg.accel * (d2 ** 2).sum((1, 2)).mean()
        l_w = cfg.wrist_prior * ((wr - wr0) ** 2).sum(1).mean() + cfg.wrist_smooth * ((wr[1:] - wr[:-1]) ** 2).sum(1).mean()
        l_m = cfg.mano_weight * (tw * ((th - tg) ** 2).sum((1, 2))).mean() if tg is not None else th.sum() * 0.0
        loss = l_rep + l_pri + l_init + l_s + l_w + l_m
        loss.backward()
        opt.step()
    return {"theta": th.detach().numpy(), "wrist": wr.detach().numpy(), "n_obs": n_obs, "views": vs,
            "loss": {"reproj": float(l_rep), "prior": float(l_pri), "smooth": float(l_s), "wrist": float(l_w),
                     "mano": float(l_m)}}


def heldout_errors(dn: Dancer, side: str, theta: np.ndarray, wrist: np.ndarray, obs_c, gc_c, cfg: FingerConfig):
    """Errors of the fitted and the MAMMA hand in one held-out view: per frame (fit_px, mamma_px, fit_art, mamma_art,
    L) for frames with >= 10 confident detected points. art = finger shape after re-anchoring on the detected wrist."""
    import torch
    base = dn.kin.hand_base(dn.out, side)
    with torch.no_grad():
        Jf = dn.kin.hand_fk(base, torch.tensor(wrist), torch.tensor(theta)).numpy()
    Jm = dn.hands[side]
    kp, w = obs_c
    rows = []
    for k in range(len(dn.t)):
        if not gc_c["ok"][k] or (w[k] > 0).sum() < 10:
            continue
        P = gc_c["K"][k] @ np.concatenate([gc_c["R"][k], gc_c["t"][k][:, None]], 1)
        pf, pm = _proj(P, Jf[k]), _proj(P, Jm[k])
        L = float(np.linalg.norm(pm[0] - pm[12]))
        if L < cfg.min_len_px:
            continue
        ok = w[k] > 0
        okf = ok.copy()
        okf[0] = False
        e = lambda p: float(np.median(np.linalg.norm(p[ok] - kp[k, ok], axis=1)))
        a = lambda p: float(np.median(np.linalg.norm((p - p[0] + kp[k, 0])[okf] - kp[k, okf], axis=1)))
        rows.append((k, e(pf), e(pm), a(pf), a(pm), L))
    return np.array(rows, float).reshape(-1, 6)


def load_grid_obs(work: Path, det: str) -> dict:
    z = np.load(work / "hands" / "grid_obs.npz")
    out = {}
    for key in z.files:
        d, r, s, c, kind = key.split("|")
        if d != det:
            continue
        out.setdefault((r, s), {}).setdefault(c, [None, None])[0 if kind == "kp" else 1] = z[key].astype(float)
    return {k: {c: tuple(v) for c, v in vv.items()} for k, vv in out.items()}


_POOL: dict = {}


def _pool_init(take_name: str, cfg_d: dict, smplx_dir: str | None):
    """Worker process for the parallel leave-one-camera-out fits (spawned; loads the take once)."""
    import torch
    torch.set_num_threads(1)
    take = load_take(take_name)
    work = take.work
    cams = fit_cams(work)
    tracks = md.load_tracks(work, cams)
    dancers = {r: Dancer(work, r, Path(smplx_dir) if smplx_dir else None) for r in ROLES}
    cfg = FingerConfig(**cfg_d)
    _POOL.update(work=work, cams=cams, dancers=dancers, gc=grid_cameras(work, tracks, dancers["lead"].t),
                 obs=load_grid_obs(work, cfg.detector), cfg=cfg,
                 mano=load_mano_targets(work, cfg.mano_model) if cfg.mano_model else None)


def _loo_task(r: str, s: str, c: str):
    P = _POOL
    dn, cfg = P["dancers"][r], P["cfg"]
    o = P["obs"].get((r, s), {})
    tgt = mano_target(P["mano"], r, s, holdout=c) if P["mano"] is not None else None
    f = fit_hand(dn, s, o, P["gc"], P["cams"], cfg, holdout=c, target=tgt)
    return r, s, c, heldout_errors(dn, s, f["theta"], f["wrist"], o[c], P["gc"][c], cfg)


def load_mano_targets(work: Path, model: str) -> dict | None:
    p = work / "hands" / f"mano_fused_{model}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def mano_target(mz: dict | None, r: str, s: str, holdout: str | None = None) -> tuple | None:
    """(theta (T,15,3) with NaN, weight (T,) 0..1) from refine_hands mano; holdout -> the fusion without that camera."""
    if mz is None:
        return None
    key = f"{r}_{s}_theta_wo_{holdout}" if holdout else f"{r}_{s}_theta"
    if key not in mz:
        key = f"{r}_{s}_theta"
    wkey = f"{r}_{s}_conf_wo_{holdout}" if holdout and f"{r}_{s}_conf_wo_{holdout}" in mz else f"{r}_{s}_conf"
    return mz[key], np.clip(mz[wkey], 0, 1)


def write_contract(work: Path, dancers: dict, poses: dict, out_dir: Path, note: str) -> Path:
    """Contract smplx npz per dancer with new hand poses / wrists (poses {role: (T,55,3)}) + FK joints."""
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    for r, P in poses.items():
        dn = dancers[r]
        d = dict(dn.d)
        d["body_pose"] = P[:, 1:22].reshape(len(P), 63)
        d["left_hand_pose"] = P[:, 25:40].reshape(len(P), 45)
        d["right_hand_pose"] = P[:, 40:55].reshape(len(P), 45)
        with torch.no_grad():
            d["joints"] = dn.kin.forward(torch.tensor(P), dn.transl)["joints"].numpy()
        d["source"] = np.array(str(d.get("source", "")) + f" + {note}")
        np.savez_compressed(out_dir / f"{r}.npz", **d)
    return out_dir


def run_fingers(take: Take, cfg: FingerConfig | None = None, loo: bool = True, smplx_dir: Path | None = None,
                workers: int = 1, sheet: bool = True) -> dict:
    """Fit every hand to the multi-view keypoints (+ the fused MANO pose when cfg.mano_model); leave-one-camera-out
    comparison with the MAMMA hands. workers > 1 runs the held-out fits in spawned processes (pods)."""
    cfg = cfg or FingerConfig()
    work = take.work
    out = work / "hands"
    sfx = f"_{cfg.tag}" if cfg.tag else ""
    cams = fit_cams(work)
    t0 = time.time()
    tracks = md.load_tracks(work, cams)
    dancers = {r: Dancer(work, r, smplx_dir) for r in ROLES}
    grid = dancers["lead"].t
    gc = grid_cameras(work, tracks, grid)
    obs = load_grid_obs(work, cfg.detector)
    mz = load_mano_targets(work, cfg.mano_model) if cfg.mano_model else None
    if cfg.mano_model and mz is None:
        raise SystemExit(f"fingers: --mano {cfg.mano_model} needs hands/mano_fused_{cfg.mano_model}.npz "
                         f"(run: refine_hands mano {take.name})")
    summary = {"take": take.name, "config": asdict(cfg), "hands": {}}
    res = {r: {} for r in ROLES}
    for r in ROLES:
        for s in SIDES:
            res[r][s] = fit_hand(dancers[r], s, obs.get((r, s), {}), gc, cams, cfg, target=mano_target(mz, r, s))
    loo_rows: dict = {}
    if loo:
        tasks = [(r, s, c) for r in ROLES for s in SIDES for c in res[r][s]["views"]]
        if workers > 1 and tasks:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_pool_init,
                                     initargs=(take.name, asdict(cfg), str(smplx_dir) if smplx_dir else None)) as ex:
                for r, s, c, e in ex.map(_loo_task, *zip(*tasks)):
                    loo_rows.setdefault((r, s), []).append((c, e))
        else:
            for r, s, c in tasks:
                o = obs.get((r, s), {})
                f = fit_hand(dancers[r], s, o, gc, cams, cfg, holdout=c, target=mano_target(mz, r, s, holdout=c))
                loo_rows.setdefault((r, s), []).append((c, heldout_errors(dancers[r], s, f["theta"], f["wrist"], o[c],
                                                                          gc[c], cfg)))
    for r in ROLES:
        dn = dancers[r]
        for s in SIDES:
            full = res[r][s]
            entry = {"views": full["views"], "frames_with_obs": round(float((full["n_obs"] > 0).mean()), 3),
                     "frames_with_2plus_views": round(float((full["n_obs"] >= 2).mean()), 3),
                     "loss": full.get("loss")}
            f0 = 25 if s == "left" else 40
            th_m = dn.pose[:, f0:f0 + 15].numpy()
            entry["mean_change_deg"] = round(float(np.degrees(np.linalg.norm(full["theta"] - th_m, axis=-1)).mean()), 2)
            entry["wrist_change_deg"] = round(float(np.degrees(np.linalg.norm(
                full["wrist"] - dn.pose[:, WRIST[s]].numpy(), axis=-1)).mean()), 2)
            rows = [np.concatenate([e, np.full((len(e), 1), cams.index(c))], 1) for c, e in loo_rows.get((r, s), [])
                    if len(e)]
            E = np.concatenate(rows, 0) if rows else np.zeros((0, 7))
            full["loo"] = E
            if len(E):
                entry.update(loo_stats(E))
            summary["hands"].setdefault(r, {})[s] = entry
            brief = {k: v for k, v in entry.items() if k.startswith("loo") or k.endswith("deg")}
            print(f"[fingers{sfx}] {r} {s}: {json.dumps(brief)}", flush=True)
        np.savez_compressed(out / f"fingers{sfx}_{r}.npz", t=grid,
                            **{f"{s}_hand_pose": res[r][s]["theta"].reshape(len(grid), 45) for s in SIDES},
                            **{f"{s}_wrist": res[r][s]["wrist"] for s in SIDES},
                            **{f"{s}_n_obs": res[r][s]["n_obs"] for s in SIDES},
                            **{f"{s}_loo": res[r][s].get("loo", np.zeros((0, 7))) for s in SIDES})
    # contract copy with the new fingers + wrists (input of `contact --smplx-dir`, and of the Unity export later)
    poses = {}
    for r in ROLES:
        P = dancers[r].pose.numpy().copy()
        for s in SIDES:
            f0 = 25 if s == "left" else 40
            P[:, f0:f0 + 15] = res[r][s]["theta"]
            P[:, WRIST[s]] = res[r][s]["wrist"]
        poses[r] = P
    summary["smplx_dir"] = str(write_contract(work, dancers, poses, out / f"smplx_fingers{sfx or '_ik'}",
                                              f"dancecap.refine_hands fingers{sfx}"))
    summary["runtime_s"] = round(time.time() - t0, 1)
    (out / f"fingers{sfx}.json").write_text(json.dumps(summary, indent=1))
    summary["review"] = [str(_fingers_sheet(work, dancers, gc, cams, obs, cfg,
                                            work / "review" / f"hands_fingers{sfx}.jpg", take.name))] if sheet else []
    (out / f"fingers{sfx}.json").write_text(json.dumps(summary, indent=1))
    return summary


def loo_stats(E: np.ndarray) -> dict:
    """Rows (frame, fit px, mamma px, fit art, mamma art, L, cam) -> medians by hand size (art in hand lengths)."""
    out = {}
    L = E[:, 5]
    for name, sel in (("all", L > 0), ("len_ge_30px", L >= 30), ("len_lt_30px", L < 30)):
        if sel.sum() < 5:
            continue
        out[f"loo_{name}"] = {
            "n": int(sel.sum()),
            "fit_px": round(float(np.median(E[sel, 1])), 2), "mamma_px": round(float(np.median(E[sel, 2])), 2),
            "fit_art_len": round(float(np.median(E[sel, 3] / L[sel])), 3),
            "mamma_art_len": round(float(np.median(E[sel, 4] / L[sel])), 3),
            "fit_better_frac": round(float((E[sel, 3] < E[sel, 4]).mean()), 3)}
    return out


def _fingers_sheet(work: Path, dancers: dict, gc: dict, cams: list[str], obs: dict, cfg: FingerConfig, path: Path,
                   take: str, tile: int = 180, per_hand: int = 4) -> Path:
    """Largest observed views per hand: crop | detections | MAMMA hand | IK hand."""
    import cv2
    import torch
    rows = []
    sfx = f"_{cfg.tag}" if cfg.tag else ""
    for r in ROLES:
        z = np.load(work / "hands" / f"fingers{sfx}_{r}.npz")
        dn = dancers[r]
        for s in SIDES:
            base = dn.kin.hand_base(dn.out, s)
            with torch.no_grad():
                Jf = dn.kin.hand_fk(base, torch.tensor(z[f"{s}_wrist"]),
                                    torch.tensor(z[f"{s}_hand_pose"].reshape(-1, 15, 3))).numpy()
            Jm = dn.hands[s]
            o = obs.get((r, s), {})
            cand = []
            for c in o:
                kp, w = o[c]
                for k in range(0, len(dn.t), 3):
                    if (w[k] > 0).sum() >= 15 and gc[c]["ok"][k]:
                        P = gc[c]["K"][k] @ np.concatenate([gc[c]["R"][k], gc[c]["t"][k][:, None]], 1)
                        pm = _proj(P, Jm[k])
                        cand.append((float(np.linalg.norm(pm[0] - pm[12])), c, k))
            cand.sort(reverse=True)
            picked, used = [], set()
            for L, c, k in cand:
                if c in used:
                    continue
                used.add(c)
                picked.append((L, c, k))
                if len(picked) >= per_hand:
                    break
            if not picked:
                continue
            hdr = np.full((20, tile * 4, 3), 30, np.uint8)
            what = f"hybrid IK (keypoints + fused {cfg.mano_model})" if cfg.mano_model else "IK fit"
            label(hdr, f"{LABEL[r]} {s} hand: video | detections ({cfg.detector}) | MAMMA | {what}", (5, 14), scale=0.45)
            rows.append(hdr)
            for L, c, k in picked:
                # grid frame -> nearest native frame of this camera (distortion ignored at crop scale)
                tr_t = dn.t[k]
                meta = json.loads((work / "frames" / c / "frames.json").read_text())
                # sync offset applied as in load_tracks
                off = md.read_sync_offsets(work).get(c)
                ts = md.apply_offset(np.array([f["t"] for f in meta["frames"]]), off)
                i = int(np.argmin(np.abs(ts - tr_t)))
                img = cv2.imread(str(work / "frames" / c / f"{i:06d}.jpg"))
                P = gc[c]["K"][k] @ np.concatenate([gc[c]["R"][k], gc[c]["t"][k][:, None]], 1)
                pm, pf = _proj(P, Jm[k]), _proj(P, Jf[k])
                kp, w = o[c][0][k], o[c][1][k]
                ext = max(pm.max(0) - pm.min(0))
                side = max(CROP_SCALE * ext, 1.5 * L, 24)
                cx, cy = (pm.min(0) + pm.max(0)) / 2
                box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
                sc = tile / (box[2] - box[0])
                b = crop_pad(img, box, tile, cv2.INTER_CUBIC)
                t1, t2, t3 = b.copy(), b.copy(), b.copy()
                draw_hand(t1, (kp - [box[0], box[1]]) * sc, (0, 255, 255), 1, 2)
                draw_hand(t2, (pm - [box[0], box[1]]) * sc, SIDE_COL[s], 1, 2)
                draw_hand(t3, (pf - [box[0], box[1]]) * sc, (255, 80, 255), 1, 2)
                label(b, f"cam {c} t={tr_t:.2f}s {L:.0f}px", (3, 13), scale=0.38)
                rows.append(np.concatenate([b, t1, t2, t3], 1))
    w = max(x.shape[1] for x in rows)
    sheet = np.concatenate([np.pad(x, ((0, 2), (0, w - x.shape[1]), (0, 0))) for x in rows], 0)
    head = np.full((28, w, 3), 10, np.uint8)
    label(head, f"{take}: finger IK to multi-view keypoints (largest views; undistortion ignored in the crop)", (6, 19),
          scale=0.5)
    cv2.imwrite(str(path), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


# ------------------------------------------------------------------------------------------------ MANO regressors

@dataclass
class ManoConfig:
    min_len_px: float = 12.0       # views where the hand is shorter than this are not used
    len_ref_px: float = 40.0       # weight ~ (hand length / len_ref)^2, clipped to [0.25, 1.5]^2
    undetected: float = 0.5        # weight factor when the regressor ran on the fitted box (its own detector missed)
    vis_floor: float = 0.1         # weight factor floor for hands the z-buffer says are hidden (survey vis)
    robust_deg: float = 20.0       # Cauchy scale of the per-joint rotation averaging across views
    smooth_frames: float = 1.5     # Gaussian sigma of the temporal smoothing (30 Hz frames)


def _so3(M: np.ndarray) -> np.ndarray:
    """Nearest rotation (SVD projection) of (..., 3, 3)."""
    U, _, Vt = np.linalg.svd(M)
    D = np.ones(M.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


def _geo_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    tr = np.einsum("...ij,...ij->...", A, B)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def _wmedian(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(c, c[-1] / 2), len(x) - 1)])


def robust_rotation_mean(Rs: np.ndarray, w: np.ndarray, scale_deg: float = 20.0, iters: int = 4):
    """Rs (V,J,3,3) per-view rotations, w (V,) weights -> (J,3,3) chordal L2 mean with Cauchy reweighting by the
    geodesic distance to the current mean, and the weighted median per-joint spread (deg; NaN with one view)."""
    W = np.broadcast_to(np.asarray(w, float)[:, None], Rs.shape[:2]).copy()
    for _ in range(iters):
        M = _so3(np.einsum("vj,vjab->jab", W, Rs))
        d = _geo_deg(Rs, M[None])
        W = np.asarray(w, float)[:, None] / (1 + (d / scale_deg) ** 2)
    spread = (np.array([_wmedian(d[:, j], np.asarray(w, float)) for j in range(Rs.shape[1])]) if len(Rs) > 1
              else np.full(Rs.shape[1], np.nan))
    return M, spread


def smooth_rotvecs(theta: np.ndarray, conf: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """theta (T,J,3) axis-angle (NaN where unknown), conf (T,) -> confidence-weighted Gaussian smoothing over time
    (fine for finger joints: small rotations about a fixed rest pose) and the smoothed confidence. NaN where no data
    lies within 3 sigma."""
    T = len(theta)
    ok = np.isfinite(theta).all(tuple(range(1, theta.ndim))) & (conf > 0)
    out = np.full_like(theta, np.nan, dtype=float)
    cs = np.zeros(T)
    if not ok.any():
        return out, cs
    idx = np.arange(T)
    th0 = np.nan_to_num(theta)
    c0 = np.where(ok, conf, 0.0)
    for k in range(T):
        g = np.exp(-0.5 * ((idx - k) / sigma) ** 2)
        g[np.abs(idx - k) > 3 * sigma] = 0.0
        gw = g * c0
        if gw.sum() > 1e-9:
            out[k] = np.tensordot(gw, th0, axes=(0, 0)) / gw.sum()
            cs[k] = gw.sum() / g.sum()
    return out, cs


def mano_views(work: Path, model: str, grid: np.ndarray, cfg: ManoConfig) -> dict:
    """hands/mano_<model>.npz -> {(role, side): {grid frame: [(cam, R15 (15,3,3), orient (3,3), w, L, job index)]}}."""
    from scipy.spatial.transform import Rotation
    z = _npz(work / "hands" / f"mano_{model}.npz")
    jobs = json.loads((work / "hands" / "jobs.json").read_text())["jobs"]
    sv = _npz(work / "hands" / "survey.npz")
    vis = {}
    for key in list(sv):
        if key.endswith("_vis"):
            pre = key[:-3]
            for i, v in zip(sv[pre + "i"].tolist(), sv[key].tolist()):
                vis[(pre, int(i))] = float(v)
    dt = float(np.median(np.diff(grid))) if len(grid) > 1 else 1 / 30
    out: dict = {}
    for n in range(len(z["job_cam"])):
        j = jobs[n]
        if not z["ok"][n] or j["cam"] != str(z["job_cam"][n]) or int(j["i"]) != int(z["job_i"][n]):
            continue
        L = float(j["len_px"])
        if L < cfg.min_len_px:
            continue
        k = int(np.argmin(np.abs(grid - j["t"])))
        if abs(grid[k] - j["t"]) > 0.5 * dt + 1e-3:
            continue
        r, s, c = j["role"], j["side"], j["cam"]
        v = vis.get((f"{r}_{s}_{c}_", int(j["i"])), 1.0)
        w = float(np.clip(L / cfg.len_ref_px, 0.25, 1.5) ** 2 * (1.0 if z["box_src"][n] == 1 else cfg.undetected)
                  * max(v, cfg.vis_floor))
        R15 = Rotation.from_rotvec(z["pose_aa"][n].astype(float)).as_matrix()
        out.setdefault((r, s), {}).setdefault(k, []).append((c, R15, z["orient"][n].astype(float), w, L, n))
    return out


def fuse_mano(views_rs: dict, T: int, gc: dict, cfg: ManoConfig, exclude: str | None = None) -> dict:
    """Per grid frame: robust mean of the views' finger rotations (view-independent local rotations) and of the wrist
    orientation in the world (camera R^T @ regressed orientation), then temporal smoothing."""
    from scipy.spatial.transform import Rotation
    raw = np.full((T, 15, 3), np.nan)
    conf = np.zeros(T)
    nv = np.zeros(T, int)
    spread = np.full((T, 15), np.nan)
    wrist = np.full((T, 3, 3), np.nan)
    for k, vs in views_rs.items():
        vs = [v for v in vs if v[0] != exclude and gc[v[0]]["ok"][k]]
        if not vs:
            continue
        Rs = np.stack([v[1] for v in vs])
        w = np.array([v[3] for v in vs])
        M, sp = robust_rotation_mean(Rs, w, cfg.robust_deg)
        raw[k] = Rotation.from_matrix(M).as_rotvec()
        conf[k], nv[k], spread[k] = w.sum(), len(vs), sp
        Rw = np.stack([gc[v[0]]["R"][k].T @ v[2] for v in vs])[:, None]
        wrist[k] = robust_rotation_mean(Rw, w, cfg.robust_deg)[0][0]
    th, cs = smooth_rotvecs(raw, conf, cfg.smooth_frames)
    return {"theta": th, "conf": cs, "theta_raw": raw, "conf_raw": conf, "nviews": nv, "spread": spread,
            "wrist_world": wrist}


def run_mano(take: Take, models: tuple | None = None, cfg: ManoConfig | None = None,
             smplx_dir: Path | None = None) -> dict:
    """Fuse the per-view MANO regressions (WiLoR / HaMeR) across cameras and time; leave-one-camera-out finger-shape
    error against the held-out view's keypoint detections (same measure as `fingers`)."""
    cfg = cfg or ManoConfig()
    work = take.work
    out = work / "hands"
    models = models or mano_models_present(work)
    if not models:
        raise SystemExit("mano: no hands/mano_<model>.npz (run dancecap.hand_mano_worker on a pod first)")
    cams = fit_cams(work)
    t0 = time.time()
    tracks = md.load_tracks(work, cams)
    dancers = {r: Dancer(work, r, smplx_dir) for r in ROLES}
    grid = dancers["lead"].t
    T = len(grid)
    gc = grid_cameras(work, tracks, grid)
    obs = load_grid_obs(work, "fused") if (out / "grid_obs.npz").exists() else {}
    fcfg = FingerConfig()
    summary = {"take": take.name, "config": asdict(cfg), "models": {}}
    for m in models:
        V = mano_views(work, m, grid, cfg)
        arrays = {"t": grid}
        ms = summary["models"].setdefault(m, {})
        info_p = out / f"mano_{m}.json"
        if info_p.exists():
            ms["worker"] = json.loads(info_p.read_text())
        for r in ROLES:
            dn = dancers[r]
            for s in SIDES:
                f0 = 25 if s == "left" else 40
                th_m = dn.pose[:, f0:f0 + 15].numpy()
                F = fuse_mano(V.get((r, s), {}), T, gc, cfg)
                for k, v in F.items():
                    arrays[f"{r}_{s}_{k}"] = v
                cov = np.isfinite(F["theta"]).all((1, 2))
                e = {"frames_covered": round(float(cov.mean()), 3),
                     "views_median": float(np.median(F["nviews"][F["nviews"] > 0])) if (F["nviews"] > 0).any() else 0}
                if cov.any():
                    from scipy.spatial.transform import Rotation
                    Rf = Rotation.from_rotvec(F["theta"][cov].reshape(-1, 3)).as_matrix()
                    Rm = Rotation.from_rotvec(th_m[cov].reshape(-1, 3)).as_matrix()
                    e["finger_change_vs_mamma_deg"] = round(float(_geo_deg(Rf, Rm).mean()), 2)
                    e["view_spread_deg_median"] = round(float(np.nanmedian(np.nanmean(F["spread"], 1))), 2) \
                        if np.isfinite(F["spread"]).any() else None
                    Rwm = dn.out["R"][:, WRIST[s]].numpy()
                    okw = np.isfinite(F["wrist_world"]).all((1, 2))
                    if okw.any():
                        dev = _geo_deg(F["wrist_world"][okw], Rwm[okw])
                        arrays[f"{r}_{s}_wrist_vs_mamma_deg"] = np.where(okw, 0, np.nan)
                        arrays[f"{r}_{s}_wrist_vs_mamma_deg"][okw] = dev
                        e["wrist_vs_mamma_deg_median"] = round(float(np.median(dev)), 1)
                # leave-one-camera-out: fuse without camera c, judge the finger shape in c
                rows = []
                o = obs.get((r, s), {})
                for c in cams:
                    Fc = fuse_mano(V.get((r, s), {}), T, gc, cfg, exclude=c)
                    arrays[f"{r}_{s}_theta_wo_{c}"], arrays[f"{r}_{s}_conf_wo_{c}"] = Fc["theta"], Fc["conf"]
                    if c not in o:
                        continue
                    covc = np.isfinite(Fc["theta"]).all((1, 2))
                    th = np.where(covc[:, None, None], Fc["theta"], th_m)
                    E = heldout_errors(dn, s, th, dn.pose[:, WRIST[s]].numpy(), o[c], gc[c], fcfg)
                    E = E[covc[E[:, 0].astype(int)]] if len(E) else E
                    if len(E):
                        rows.append(np.concatenate([E, np.full((len(E), 1), cams.index(c))], 1))
                Eall = np.concatenate(rows, 0) if rows else np.zeros((0, 7))
                arrays[f"{r}_{s}_loo"] = Eall
                if len(Eall):
                    e.update(loo_stats(Eall))
                ms.setdefault(r, {})[s] = e
                print(f"[mano] {m} {r} {s}: {json.dumps(e)}", flush=True)
        np.savez_compressed(out / f"mano_fused_{m}.npz", **arrays)
        ms["review"] = str(_mano_sheet(work, take.name, m, dancers, gc, cams, arrays,
                                       work / "review" / f"hands_mano_{m}.jpg"))
    summary["runtime_s"] = round(time.time() - t0, 1)
    (out / "mano.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def _mano_sheet(work: Path, take: str, model: str, dancers: dict, gc: dict, cams: list[str], arrays: dict,
                path: Path, tile: int = 170, per_hand: int = 5) -> Path:
    """Per hand, the largest views (one per camera): video | the regressor's own joints in that view | the multi-view
    fused fingers on the MAMMA wrist | the MAMMA hand."""
    import cv2
    import torch
    z = _npz(work / "hands" / f"mano_{model}.npz")
    jobs = json.loads((work / "hands" / "jobs.json").read_text())["jobs"]
    grid = dancers["lead"].t
    rows = []
    for r in ROLES:
        dn = dancers[r]
        for s in SIDES:
            th = arrays[f"{r}_{s}_theta"]
            cov = np.isfinite(th).all((1, 2))
            if not cov.any():
                continue
            f0 = 25 if s == "left" else 40
            thf = np.where(cov[:, None, None], th, dn.pose[:, f0:f0 + 15].numpy())
            base = dn.kin.hand_base(dn.out, s)
            with torch.no_grad():
                Jf = dn.kin.hand_fk(base, dn.pose[:, WRIST[s]], torch.tensor(thf)).numpy()
            Jm = dn.hands[s]
            cand = {}
            for n in range(len(z["job_cam"])):
                j = jobs[n]
                if j["role"] != r or j["side"] != s or not z["ok"][n]:
                    continue
                k = int(np.argmin(np.abs(grid - j["t"])))
                if not cov[k]:
                    continue
                score = j["len_px"] * (1.5 if z["box_src"][n] == 1 else 1.0)
                if j["cam"] not in cand or score > cand[j["cam"]][0]:
                    cand[j["cam"]] = (score, n, k)
            picked = sorted(cand.values(), reverse=True)[:per_hand]
            if not picked:
                continue
            hdr = np.full((20, tile * 4, 3), 30, np.uint8)
            label(hdr, f"{LABEL[r]} {s} hand: video | {model} in this view | {model} fused over views (MAMMA wrist) | "
                       f"MAMMA", (5, 14), scale=0.42)
            rows.append(hdr)
            for _, n, k in picked:
                j = jobs[n]
                c = j["cam"]
                img = cv2.imread(str(work / "frames" / c / f"{int(j['i']):06d}.jpg"))
                P = gc[c]["K"][k] @ np.concatenate([gc[c]["R"][k], gc[c]["t"][k][:, None]], 1)
                pm, pf = _proj(P, Jm[k]), _proj(P, Jf[k])
                L = float(j["len_px"])
                ext = max(pm.max(0) - pm.min(0))
                side = max(CROP_SCALE * ext, 1.5 * L, 24)
                cx, cy = (pm.min(0) + pm.max(0)) / 2
                box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
                sc = tile / (box[2] - box[0])
                b = crop_pad(img, box, tile, cv2.INTER_CUBIC)
                t1, t2, t3 = b.copy(), b.copy(), b.copy()
                draw_hand(t1, (z["kp2d"][n].astype(float) - [box[0], box[1]]) * sc, (255, 255, 0), 1, 2)
                label(t1, f"{'detected' if z['box_src'][n] == 1 else 'fit box'} {z['det_conf'][n]:.2f}", (3, 13),
                      scale=0.36)
                draw_hand(t2, (pf - [box[0], box[1]]) * sc, (80, 255, 80), 1, 2)
                nv = int(arrays[f"{r}_{s}_nviews"][k])
                sp = arrays[f"{r}_{s}_spread"][k]
                label(t2, f"{nv} views, spread {np.nanmean(sp):.0f}deg" if np.isfinite(sp).any() else f"{nv} view",
                      (3, 13), scale=0.36)
                draw_hand(t3, (pm - [box[0], box[1]]) * sc, SIDE_COL[s], 1, 2)
                label(b, f"cam {c} t={grid[k]:.2f}s {L:.0f}px", (3, 13), scale=0.38)
                rows.append(np.concatenate([b, t1, t2, t3], 1))
    if not rows:
        rows = [np.full((40, tile * 4, 3), 30, np.uint8)]
    w = max(x.shape[1] for x in rows)
    sheet = np.concatenate([np.pad(x, ((0, 2), (0, w - x.shape[1]), (0, 0))) for x in rows], 0)
    head = np.full((28, w, 3), 10, np.uint8)
    label(head, f"{take}: {model} MANO regression per view and fused over the 8 cameras (largest views)", (6, 19),
          scale=0.5)
    cv2.imwrite(str(path), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


METHODS = (("fingers", "keypoint IK (RTMW/RTMPose/MediaPipe fused)"), ("mano:wilor", "WiLoR fused over views"),
           ("mano:hamer", "HaMeR fused over views"), ("fingers_hybrid", "hybrid: keypoint IK + fused WiLoR"))


def run_compare(take: Take) -> dict:
    """One table + chart of the leave-one-camera-out finger-shape error (hand lengths, lower = better) of every finger
    route that has run, against the MAMMA hand on the same held-out views."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    work = take.work
    out = work / "hands"
    res: dict = {}
    for key, name in METHODS:
        if key.startswith("mano:"):
            p = out / "mano.json"
            if not p.exists():
                continue
            src = json.loads(p.read_text())["models"].get(key[5:])
        else:
            p = out / f"{key}.json"
            src = json.loads(p.read_text())["hands"] if p.exists() else None
        if not src:
            continue
        for r in ROLES:
            for s in SIDES:
                e = (src.get(r) or {}).get(s) or {}
                for b in ("all", "len_ge_30px", "len_lt_30px"):
                    st = e.get(f"loo_{b}")
                    if st:
                        res.setdefault(f"{r}_{s}", {}).setdefault(b, {})[key] = {
                            "art_len": st["fit_art_len"], "mamma_art_len": st["mamma_art_len"],
                            "better_than_mamma": st["fit_better_frac"], "n": st["n"]}
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    cols = ["#1f77b4", "#d62728", "#8c564b", "#2ca02c"]
    for ax, (r, s) in zip(axes, [(r, s) for r in ROLES for s in SIDES]):
        d = res.get(f"{r}_{s}", {})
        bins = ["all", "len_ge_30px", "len_lt_30px"]
        x = np.arange(len(bins))
        bw = 0.8 / (len(METHODS) + 1)
        mam = [np.nanmedian([v["mamma_art_len"] for v in d.get(b, {}).values()]) if d.get(b) else np.nan for b in bins]
        ax.bar(x - 0.4 + bw / 2, mam, bw, color="#999999", label="MAMMA hand (same held-out views)")
        for j, (key, name) in enumerate(METHODS):
            y = [d.get(b, {}).get(key, {}).get("art_len", np.nan) for b in bins]
            ax.bar(x - 0.4 + bw * (j + 1.5), y, bw, color=cols[j], label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(["all views", "hand >= 30 px", "hand < 30 px"])
        ax.set_title(f"{LABEL[r]} {s} hand")
        if ax is axes[0]:
            ax.set_ylabel("held-out finger-shape error / hand length (median)")
            ax.legend(fontsize=7)
    fig.suptitle(f"{take.name}: finger routes, leave-one-camera-out (lower = better; wrist re-anchored on the "
                 f"held-out detection)")
    fig.tight_layout()
    path = work / "review" / "hands_compare.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    summary = {"take": take.name, "metric": "median held-out finger-shape error / hand length", "hands": res,
               "review": [str(path)]}
    if (out / "grid_obs.npz").exists():
        try:
            summary["cross_judge"] = cross_judge(take)
            summary["review"].append(summary["cross_judge"].pop("review"))
        except Exception as e:  # the LOO table above stands on its own
            summary["cross_judge_error"] = f"{type(e).__name__}: {e}"
    (out / "compare.json").write_text(json.dumps(summary, indent=1))
    return summary


def cross_judge(take: Take, min_pts: int = 10, min_len_px: float = 12.0) -> dict:
    """Judge every finger route with 2D evidence it never saw. The leave-one-camera-out tables use the same detector
    family for fitting and judging (a detector's biases - RTMW draws small hands as compact fists - agree with
    themselves across views), so here each route's full fit is projected into every view and compared, finger shape
    only (wrist re-anchored, / hand length), with each judge: keypoint detectors ('fused': RTMW/RTMPose/MediaPipe) and
    each MANO regressor's own per-view joints. Cells where the route was fitted to that judge are flagged in-sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    work = take.work
    out = work / "hands"
    cams = fit_cams(work)
    tracks = md.load_tracks(work, cams)
    dancers = {r: Dancer(work, r) for r in ROLES}
    gc = grid_cameras(work, tracks, dancers["lead"].t)
    judges = [j for j in ("fused", *MANO_MODELS) if load_grid_obs(work, j)]
    obs = {j: load_grid_obs(work, j) for j in judges}
    routes = {"mamma": ("MAMMA", set()), "fingers": ("keypoint IK", {"fused"}),
              "fingers_hybrid": ("hybrid IK + WiLoR", {"fused", "wilor"})}
    for m in MANO_MODELS:
        if (out / f"mano_fused_{m}.npz").exists():
            routes[f"mano_{m}"] = (f"{m} fused", {m})
            routes[f"mano_{m}_wrist"] = (f"{m} fused + its wrist rotation", {m})
    joints: dict = {}
    for r in ROLES:
        dn = dancers[r]
        for s in SIDES:
            f0 = 25 if s == "left" else 40
            base = dn.kin.hand_base(dn.out, s)
            th_m = dn.pose[:, f0:f0 + 15].numpy()
            for key in routes:
                if key == "mamma":
                    J = dn.hands[s]
                elif key.startswith("fingers"):
                    p = out / f"{key}_{r}.npz"
                    if not p.exists():
                        continue
                    z = _npz(p)
                    with torch.no_grad():
                        J = dn.kin.hand_fk(base, torch.tensor(z[f"{s}_wrist"]),
                                           torch.tensor(z[f"{s}_hand_pose"].reshape(-1, 15, 3))).numpy()
                else:
                    from scipy.spatial.transform import Rotation
                    z = _npz(out / f"mano_fused_{key.split('_')[1]}.npz")
                    th = z[f"{r}_{s}_theta"]
                    th = np.where(np.isfinite(th).all((1, 2))[:, None, None], th, th_m)
                    wr = dn.pose[:, WRIST[s]].numpy().copy()
                    if key.endswith("_wrist"):   # the regressor's wrist orientation (world) -> local wrist rotation
                        Rw = z[f"{r}_{s}_wrist_world"]
                        okw = np.isfinite(Rw).all((1, 2))
                        Rp = base["Rp"].numpy()
                        wr[okw] = Rotation.from_matrix(np.transpose(Rp[okw], (0, 2, 1)) @ Rw[okw]).as_rotvec()
                    with torch.no_grad():
                        J = dn.kin.hand_fk(base, torch.tensor(wr), torch.tensor(th)).numpy()
                joints[(key, r, s)] = J
    res: dict = {}
    for r in ROLES:
        for s in SIDES:
            for jd in judges:
                o = obs[jd].get((r, s), {})
                rows = {k: [] for k in routes}
                for c, (kp, w) in o.items():
                    for k in range(len(kp)):
                        ok = w[k] > 0
                        if ok.sum() < min_pts or not gc[c]["ok"][k]:
                            continue
                        P = gc[c]["K"][k] @ np.concatenate([gc[c]["R"][k], gc[c]["t"][k][:, None]], 1)
                        pm = _proj(P, dancers[r].hands[s][k])
                        L = float(np.linalg.norm(pm[0] - pm[12]))
                        if L < min_len_px:
                            continue
                        okf = ok.copy()
                        okf[0] = False
                        for key in routes:
                            J = joints.get((key, r, s))
                            if J is None:
                                continue
                            p = _proj(P, J[k])
                            rows[key].append(float(np.median(np.linalg.norm((p - p[0] + kp[k, 0])[okf] - kp[k, okf],
                                                                            axis=1))) / L)
                base_e = np.array(rows["mamma"])
                for key, (name, fitted_to) in routes.items():
                    e = np.array(rows[key])
                    if not len(e):
                        continue
                    res.setdefault(f"{r}_{s}", {}).setdefault(jd, {})[key] = {
                        "art_len": round(float(np.median(e)), 3), "n": int(len(e)),
                        "better_than_mamma": round(float((e < base_e).mean()), 3) if len(e) == len(base_e) else None,
                        "in_sample": jd in fitted_to}
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    keys = list(routes)
    cols = {"mamma": "#999999", "fingers": "#1f77b4", "fingers_hybrid": "#2ca02c", "mano_wilor": "#d62728",
            "mano_hamer": "#8c564b", "mano_wilor_wrist": "#ff9896", "mano_hamer_wrist": "#c49c94"}
    for ax, (r, s) in zip(axes, [(r, s) for r in ROLES for s in SIDES]):
        d = res.get(f"{r}_{s}", {})
        x = np.arange(len(judges))
        bw = 0.8 / max(len(keys), 1)
        for j, key in enumerate(keys):
            y = [d.get(jd, {}).get(key, {}).get("art_len", np.nan) for jd in judges]
            ins = [d.get(jd, {}).get(key, {}).get("in_sample", False) for jd in judges]
            bars = ax.bar(x - 0.4 + bw * (j + 0.5), y, bw, color=cols.get(key, "#444444"), label=routes[key][0])
            for b, i in zip(bars, ins):
                if i:
                    b.set_hatch("//")
                    b.set_alpha(0.45)
        ax.set_xticks(x)
        ax.set_xticklabels([f"judge: {DET_LABEL.get(jd, jd)}" for jd in judges], fontsize=7)
        ax.set_title(f"{LABEL[r]} {s} hand")
        if ax is axes[0]:
            ax.set_ylabel("finger-shape error / hand length (median, all views)")
            ax.legend(fontsize=7)
    fig.suptitle(f"{take.name}: each finger route judged by 2D evidence it was not fitted to (hatched = in-sample, "
                 f"not a fair judge); lower = better")
    fig.tight_layout()
    path = work / "review" / "hands_cross_judge.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return {"judges": judges, "routes": {k: v[0] for k, v in routes.items()}, "hands": res, "review": str(path)}


# ------------------------------------------------------------------------------------------------ contact

@dataclass
class ContactConfig:
    gap_full_m: float = 0.015      # 3D score 1 at/below this hand-to-partner gap
    gap_zero_m: float = 0.06       # 3D score 0 at/above
    view_touch_len: float = 0.15   # a view supports contact: hand points within this x hand length (>= 3 px) of the
                                   # partner's mask (inside counts: the hand is on or behind the partner)
    view_gap_len: float = 0.35     # a view contradicts contact: every hand point further than this from the mask
    on: float = 0.55
    off: float = 0.30
    min_frames: int = 4
    merge_gap: int = 3
    hold_min_s: float = 0.30
    hold_gap_m: float = 0.03
    hold_evidence: float = 0.6
    near_lo_m: float = 0.02        # near-miss: closest approach in [near_lo, near_hi] without a contact episode
    near_hi_m: float = 0.07
    bridge_max_gap: int = 15       # merged hand-hand holds: gaps between the both-records runs bridged when the
    bridge_min_ev: float = 0.6     # better record's evidence is >= min on every gap frame ...
    bridge_med_ev: float = 0.75    # ... and >= med in the median (bridge_runs)


# dominant LBS joint -> partner region (torso split front/back/side by the surface normal)
_REG = {0: "waist", 3: "waist", 6: "back", 9: "back", 12: "neck", 15: "head", 22: "head", 23: "head", 24: "head",
        13: "left shoulder", 14: "right shoulder", 16: "left upper arm", 17: "right upper arm",
        18: "left forearm", 19: "right forearm", 20: "left hand", 21: "right hand",
        1: "left hip", 2: "right hip", 4: "left leg", 5: "right leg", 7: "left leg", 8: "right leg",
        10: "left foot", 11: "right foot"}
for _j in range(25, 40):
    _REG[_j] = "left hand"
for _j in range(40, 55):
    _REG[_j] = "right hand"


def vertex_joint(mdir: Path) -> np.ndarray:
    from .hand_kin import model_npz
    return np.asarray(np.load(model_npz(mdir), allow_pickle=True)["weights"], float).argmax(1)


def region_name(joint: int, normal: np.ndarray, fwd: np.ndarray) -> str:
    base = _REG.get(int(joint), "body")
    if base in ("waist", "back"):
        d = float(normal @ fwd)
        if base == "back":
            return "chest" if d > 0.35 else ("upper back" if d < -0.35 else "side (ribs)")
        return "belly" if d > 0.35 else ("lower back / waist" if d < -0.35 else "waist (side)")
    return base


def torso_forward(joints: np.ndarray) -> np.ndarray:
    """(T,3) horizontal facing direction (left hip - right hip) x up, as avatar_review."""
    fwd = -np.cross(joints[:, 2] - joints[:, 1], np.array([0.0, 1.0, 0.0]))
    return fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)


def mesh_normals(V: np.ndarray, faces: np.ndarray) -> np.ndarray:
    from .avatar_uv import vertex_normals
    return vertex_normals(V, faces)


def hand_partner_geometry(dancers: dict, hand_ids: dict, arm_ids: dict | None = None, device: str | None = None,
                          max_depth: float = 0.16) -> dict:
    """Per (role, side): signed gap (T,) from the hand surface to the partner's surface (negative = penetration depth),
    the number of hand vertices within 1.5 cm, the closest partner vertex and its region. Exact point-to-triangle
    distances; inside = generalized winding number of the partner's closed body > 0.5 (mesh_contact) - the earlier
    nearest-vertex-normal sign was wrong near concave regions (2026-10-07 review). With arm_ids {side: vertex ids}
    also 'arm_gap' / 'arm_pen' (T,): the upper arm + forearm against the partner."""
    from . import mesh_contact as mc
    from scipy.spatial import cKDTree
    out = {}
    T = len(dancers["lead"].t)
    faces = dancers["lead"].kin.faces
    topo = mc.Topology(faces, len(dancers["lead"].kin.weights_np))
    for r in ROLES:
        o = OTHER[r]
        Vo_all = dancers[o].verts()
        Vr_all = dancers[r].verts()
        vj = vertex_joint(dancers[o].mdir)
        fwd = torso_forward(dancers[o].joints)
        res = {s: {"gap": np.full(T, np.nan), "pen": np.zeros(T), "n_close": np.zeros(T, int),
                   "pvert": np.zeros(T, int), "region": np.array([""] * T, dtype="<U24"),
                   "arm_gap": np.full(T, np.nan), "arm_pen": np.zeros(T)} for s in SIDES}
        sets = [(s, "hand", np.asarray(hand_ids[s])) for s in SIDES]
        if arm_ids:
            sets += [(s, "arm", np.asarray(arm_ids[s])) for s in SIDES]
        allp = np.concatenate([x[2] for x in sets])
        bounds = np.cumsum([0] + [len(x[2]) for x in sets])
        for k in range(T):
            Vo = Vo_all[k]
            tree = cKDTree(Vo[topo.used])
            sd_all = mc.signed_distance(Vr_all[k][allp], Vo, topo, device=device, max_depth=max_depth, tree=tree)
            No = None
            for (s, kind, _), a, b in zip(sets, bounds[:-1], bounds[1:]):
                sd = sd_all["sd"][a:b]
                m = int(np.argmin(sd))
                e = res[s]
                if kind == "arm":
                    e["arm_gap"][k], e["arm_pen"][k] = sd[m], max(0.0, -float(sd.min()))
                    continue
                e["gap"][k], e["pen"][k] = sd[m], max(0.0, -float(sd.min()))
                e["n_close"][k] = int((sd < 0.015).sum())
                f = int(sd_all["face"][a + m])
                pv = int(topo.F[f][int(np.argmax(sd_all["bary"][a + m]))])
                e["pvert"][k] = pv
                if No is None:
                    No = mesh_normals(Vo, faces)
                e["region"][k] = region_name(vj[pv], No[pv], fwd[k])
        for s in SIDES:
            out[(r, s)] = res[s]
    return out


def view_evidence(work: Path, dancers: dict, tracks: dict, cams: list[str], cfg: ContactConfig) -> dict:
    """Per (role, side): (T, n_cams) view verdicts on the grid: +1 consistent with contact, -1 contradicts (a visible
    gap to the partner's mask), 0 unknown (camera not ok / hand outside the image)."""
    import cv2
    T = len(dancers["lead"].t)
    grid = dancers["lead"].t
    out = {(r, s): np.zeros((T, len(cams)), np.int8) for r in ROLES for s in SIDES}
    gaps = {(r, s): np.full((T, len(cams)), np.nan) for r in ROLES for s in SIDES}
    for ci, c in enumerate(cams):
        tr = tracks[c]
        near = md.nearest_index(tr.t, grid)
        for k in range(T):
            i = int(near[k])
            if not tr.ok[i] or abs(tr.t[i] - grid[k]) > 1.0 / 30:
                continue
            K, R, tt, k1 = tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i])
            for r in ROLES:
                m = _mask(work / "dancers" / c / "masks" / f"mask_{i:04d}_{PIDS[OTHER[r]]:02d}.png")
                if m is None:
                    continue
                dt = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
                H, W = m.shape
                for s in SIDES:
                    px, z = project(dancers[r].hands[s][k], K, R, tt, k1)
                    if not (((px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)).all() and (z > 0.1).all()):
                        continue
                    L = float(np.linalg.norm(px[0] - px[12]))
                    u = np.clip(np.round(px).astype(int), 0, [W - 1, H - 1])
                    g = float(dt[u[:, 1], u[:, 0]].min())
                    gaps[(r, s)][k, ci] = g / max(L, 4.0)
                    if g <= max(3.0, cfg.view_touch_len * L):
                        out[(r, s)][k, ci] = 1
                    elif g >= cfg.view_gap_len * L:
                        out[(r, s)][k, ci] = -1
    return {"verdict": out, "gap_len": gaps}


def contact_episodes(geo: dict, ev: dict, t: np.ndarray, cfg: ContactConfig) -> tuple[list, dict]:
    from .counterbalance import hysteresis
    episodes, series = [], {}
    dt = float(np.median(np.diff(t)))
    for (r, s), g in geo.items():
        gap = g["gap"]
        s3 = np.clip((cfg.gap_zero_m - gap) / (cfg.gap_zero_m - cfg.gap_full_m), 0, 1)
        v = ev["verdict"][(r, s)]
        pos, neg = (v > 0).sum(1), (v < 0).sum(1)
        evid = np.where(pos + neg > 0, pos / np.maximum(pos + neg, 1), 0.5)       # 0.5 = no view could tell
        score = s3 * (0.4 + 0.6 * evid)
        series[(r, s)] = {"gap": gap, "s3": s3, "evidence": evid, "score": score, "views_for": pos, "views_against": neg}
        for a, b in hysteresis(score, cfg.on, cfg.off, cfg.min_frames, cfg.merge_gap):
            sl = slice(a, b + 1)
            regs, cnt = np.unique(g["region"][sl], return_counts=True)
            region = str(regs[np.argmax(cnt)])
            dur = (b - a + 1) * dt
            med_gap, med_ev = float(np.median(gap[sl])), float(np.median(evid[sl]))
            kind = "hold" if (dur >= cfg.hold_min_s and med_gap <= cfg.hold_gap_m and med_ev >= cfg.hold_evidence) \
                else "touch"
            episodes.append({"role": r, "side": s, "partner": OTHER[r], "region": region,
                             "hand_hand": region.endswith("hand"), "kind": kind, "i0": int(a), "i1": int(b),
                             "t0": round(float(t[a]), 3), "t1": round(float(t[b]), 3), "duration_s": round(dur, 3),
                             "gap_cm_median": round(med_gap * 100, 2), "gap_cm_min": round(float(gap[sl].min()) * 100, 2),
                             "penetration_cm_max": round(float(g["pen"][sl].max()) * 100, 2),
                             "evidence_median": round(med_ev, 3),
                             "views_for_median": float(np.median(pos[sl])), "views_against_median": float(np.median(neg[sl])),
                             "score_mean": round(float(score[sl].mean()), 3)})
        # near misses: closest approach in [near_lo, near_hi] outside every episode of this hand
        inside = np.zeros(len(t), bool)
        for e in episodes:
            if e["role"] == r and e["side"] == s:
                inside[e["i0"]:e["i1"] + 1] = True
        cand = (~inside) & (gap >= cfg.near_lo_m) & (gap <= cfg.near_hi_m)
        for a, b in hysteresis(cand.astype(float), 0.5, 0.5, 3, 1):
            k = a + int(np.argmin(gap[a:b + 1]))
            if gap[k] > cfg.near_hi_m * 0.8:
                continue
            regs, cnt = np.unique(g["region"][a:b + 1], return_counts=True)
            episodes.append({"role": r, "side": s, "partner": OTHER[r], "region": str(regs[np.argmax(cnt)]),
                             "hand_hand": str(regs[np.argmax(cnt)]).endswith("hand"), "kind": "near-miss",
                             "i0": int(a), "i1": int(b), "t0": round(float(t[a]), 3), "t1": round(float(t[b]), 3),
                             "duration_s": round((b - a + 1) * dt, 3), "gap_cm_median": round(float(np.median(gap[a:b + 1])) * 100, 2),
                             "gap_cm_min": round(float(gap[k]) * 100, 2), "penetration_cm_max": 0.0,
                             "evidence_median": round(float(np.median(evid[a:b + 1])), 3),
                             "views_for_median": float(np.median(pos[a:b + 1])),
                             "views_against_median": float(np.median(neg[a:b + 1])), "score_mean": round(float(score[a:b + 1].mean()), 3)})
    episodes.sort(key=lambda e: (e["t0"], e["role"], e["side"]))
    return episodes, series


# ------------------------------------------------------------------------------------------------ contact refinement

@dataclass
class RefineConfig:
    ramp: int = 6                  # frames of blend-in / blend-out around an episode
    contact_target_m: float = 0.004
    w_contact: float = 4000.0      # (softmin gap - target)^2 inside the episode, m^-2
    # penetration (2026-10-07 review: the nearest-vertex normal sign was wrong near concave regions): points inside the
    # partner by the generalized winding number of the partner's closed body, penalised by their squared distance to
    # the exact closest point of the partner's surface (the inside flags are refreshed every nn_every iterations)
    w_pen: float = 20000.0         # hand vertices, m^-2 (sum over vertices, mean over frames)
    w_arm_pen: float = 2000.0      # the moved arm's upper-arm + forearm vertices (every arm_stride-th); 0 in retries
    arm_stride: int = 2
    w_finger: float = 600.0        # fingertips that start within finger_reach of the partner are laid onto it ...
    finger_reach_m: float = 0.05
    finger_facing: float = 0.3     # ... only those whose nearest partner point lies on the palm side (cos > this)
    # finger-finger spacing (the review found the contact step crossing adjacent fingers: no self-collision term):
    # capsule sample points of different fingers stay >= min(r_a + r_b, max(input distance, self_floor (r_a + r_b)))
    w_self: float = 1.0e7
    self_floor: float = 0.95
    # ... and on the skin: a finger vertex may not sink deeper into another finger (nearest vertex of the other
    # finger, signed by its normal, within fself_reach) than it did in the input + 0.5 mm. The capsules alone missed
    # mesh crossings of curled distal phalanges (2026-10-07 smoke test: 4 -> 7 frames with the capsule term at 0)
    w_fself: float = 2.0e7
    fself_reach_m: float = 0.006
    fingers_free: bool = True      # False: the finger pose stays the input's (only shoulder / elbow / wrist move)
    # palm-centre displacement cap (the contact fix moved hands 4-19 cm off where the video shows them)
    w_disp: float = 5.0e7          # relu(|palm - palm_input| - cap)^2, m^-2: a stiff wall (it must beat w_pen)
    disp_cap_m: float = 0.03
    disp_cap_bad_m: float = 0.06   # where the input hand is > kp_bad_len hand lengths from its keypoint detections
    kp_bad_len: float = 1.0
    bridge_contact_mult: float = 2.0   # attraction weight on the bridged frames of a merged hand-hand hold
    sink_only: bool = False        # the conservative attempt: no attraction, only penetration removal, cap sink_cap_m
    sink_cap_m: float = 0.02
    w_rep: float = 1.0             # MAMMA's ma_2d landmarks of the moved arm (Geman-McClure in sigma units)
    w_kp: float = 0.3              # hand keypoint detections (fused) of the contacting hand
    w_arm_prior: float = 30.0      # |delta|^2 (rad^2) on shoulder/elbow/wrist corrections
    w_finger_prior: float = 3.0    # |delta theta|^2 (rad^2)
    w_pca: float = 0.01
    w_smooth: float = 300.0        # |delta_k+1 - delta_k|^2
    iters: int = 200
    lr: float = 0.01
    softmin_tau_m: float = 0.004
    nn_every: int = 20             # refresh closest points / inside flags every n iterations
    max_depth_m: float = 0.16
    # robust penetration penalty: c^2 log(1 + (d/c)^2), ~d^2 for shallow sinks, a weak pull for deep ones (2026-10-07
    # pod run 2: with d^2 a hand sunk 5-9 cm by a body-placement error drove the arm 31 deg off its landmarks against
    # the palm cap; a sink the arm cannot fix within the cap is left and reported, not forced)
    pen_scale_m: float = 0.01
    guard: bool = True             # guard (guard_ok): redo an episode more conservatively when it fails
    keep_failed: bool = False      # the old policy (keep the attempt closest to the guard); default: keep the input
    # the per-frame corrections live in a smooth temporal basis (Gaussian bumps of sigma frames on keys every step
    # frames, rows normalised): the contact terms re-pick nearest surface points every few iterations and, with free
    # per-frame deltas, made the refined hands jitter (larissa_kadu 2026-10-07: wrist 2 -> 14 mm). sigma 0 = per frame.
    basis_step: int = 2
    basis_sigma: float = 2.0


ARM_JOINTS = {s: (16 if s == "left" else 17, 18 if s == "left" else 19, 20 if s == "left" else 21) for s in SIDES}
UPPER_ARM_JOINTS = {s: (16, 18) if s == "left" else (17, 19) for s in SIDES}   # shoulder (upper arm), elbow (forearm)
PALM21 = (0, 5, 9, 13, 17)          # wrist + the four MCPs: the palm centre of the displacement cap
PALM_REST = np.array([0.0, -1.0, 0.0])   # palm-side normal of both SMPL-X hands in the wrist frame (flat T-pose)
# guard: an episode's refinement is accepted when for every moved arm (a) MAMMA's arm + hand landmark error (median,
# all cameras) grows by no more than max(GUARD_REP_PX, GUARD_REP_REL x ref) and (b) in no camera with >= GUARD_CAM_MIN_N
# observations by more than max(GUARD_CAM_PX, GUARD_CAM_REL x ref) (ref = the worse of the input and the MAMMA fit),
# (c) shoulder and elbow turn no more than GUARD_ARM_DEG, the wrist only more when it turns back towards the MAMMA
# fit's wrist (undoing the wrist gate, 2026-10-07 pod run 3), (d) wrist / elbow jitter stays <= max(GUARD_JIT_REL x before,
# before + GUARD_JIT_MM), (e) the palm's p95 displacement on frames with a good input stays within its cap +
# GUARD_DISP_M, (f) finger crossings grow by no more than max(2 frames, 5 %), (g) GUARD_ARM_PEN_CM; otherwise it is
# redone with stronger arm / image / finger priors (multipliers of w_arm_prior + w_finger_prior + w_self, w_rep, and
# of w_contact + w_finger) and the input's finger pose (GUARD_FINGERS_FREE), the last attempt sink-only (contact
# multiplier 0: penetration removal only, tighter cap). No passing attempt: the input pose is kept and the episode
# flagged (keep_failed=True restores the old 'closest attempt').
GUARD_REP_PX, GUARD_REP_REL, GUARD_ARM_DEG = 1.0, 0.08, 25.0
GUARD_WRIST_SLACK_DEG = 5.0   # (c) the wrist may exceed GUARD_ARM_DEG only towards the MAMMA fit's wrist (+ this)
GUARD_CAM_PX, GUARD_CAM_REL, GUARD_CAM_MIN_N = 4.0, 0.25, 20
GUARD_JIT_REL, GUARD_JIT_MM = 1.5, 1.0
GUARD_DISP_M = 0.01
GUARD_ARM_PEN_CM = 1.0     # (g) the moved arm may not end > 1 cm deeper inside the partner than before
GUARD_STEPS = ((1.0, 1.0, 1.0), (6.0, 4.0, 1.0), (6.0, 4.0, 0.0))
GUARD_FINGERS_FREE = (True, False, False)    # the retries keep the input's finger pose (no new finger crossings)


def smooth_basis(K: int, step: int, sigma: float) -> np.ndarray:
    """(K, n_keys) Gaussian interpolation basis (rows sum to 1) on keys every `step` frames (+ the last frame); the
    identity when sigma <= 0."""
    if sigma <= 0 or K <= 2:
        return np.eye(K)
    keys = np.unique(np.concatenate([np.arange(0, K, max(1, step)), [K - 1]]))
    W = np.exp(-0.5 * ((np.arange(K)[:, None] - keys[None]) / sigma) ** 2)
    return W / W.sum(1, keepdims=True)


def attract_mask(ep: dict, T: int) -> np.ndarray:
    """(T,) frames where an episode's contact is attracted: [i0, i1], or only its attract_runs when the episode was
    merged from both hands' records of a hand-hand hold (frames where only one record sees contact - a release or
    re-grip - get no attraction, only the penetration term; short gaps with strong evidence are bridged:
    bridge_runs)."""
    m = np.zeros(T, bool)
    runs = ep.get("attract_runs")
    for a, b in (runs if runs is not None else [(ep["i0"], ep["i1"])]):
        m[max(0, int(a)):min(T - 1, int(b)) + 1] = True
    return m


def bridge_runs(runs: list, evidence: np.ndarray, max_gap: int = 15, min_ev: float = 0.6,
                med_ev: float = 0.75) -> tuple[list, list]:
    """Merge consecutive attract runs of a hand-hand hold across gaps of <= max_gap frames where the better of the two
    hands' records still sees the hold: its 2D evidence (share of deciding views that agree) >= min_ev on every gap
    frame and >= med_ev in the median (<= 1 contradicting view in 4). The review (2026-10-07) found the held hands
    drifting 2-7 cm apart on such gaps while camera 06 shows them clasped. -> (runs, bridged gaps [[a, b], ...])."""
    runs = sorted([list(map(int, r)) for r in runs])
    if not runs:
        return [], []
    out, gaps = [runs[0]], []
    ev = np.asarray(evidence, float)
    for a, b in runs[1:]:
        g0, g1 = out[-1][1] + 1, a - 1
        if g1 < g0:
            out[-1][1] = max(out[-1][1], b)
            continue
        e = ev[g0:g1 + 1]
        if g1 - g0 + 1 <= max_gap and len(e) and np.nanmin(e) >= min_ev and np.nanmedian(e) >= med_ev:
            gaps.append([g0, g1])
            out[-1][1] = b
        else:
            out.append([a, b])
    return out, gaps


def _guard_ref(rec: dict, rr: str, ss: str, key: str = "rep_px") -> float:
    """The guard's reference landmark error: the refinement's input, or the adopted MAMMA fit's (base, when the input
    was already changed by the finger / wrist stage) if that is worse - the arm may give back what the wrist gate
    gained, but never ends worse than the MAMMA fit + tolerance."""
    b = rec.get(f"{rr}_{ss}_{key}_before", np.nan)
    base = rec.get(f"{rr}_{ss}_{key}_base", np.nan)
    if np.isfinite(base) and (not np.isfinite(b) or base > b):
        return float(base)
    return float(b)


def guard_ok(rec: dict, movers, rep_px: float | None = None, rep_rel: float | None = None,
             arm_deg: float | None = None) -> tuple[bool, list[str]]:
    """Guard verdict for one refinement attempt's record (see GUARD_*). Missing keys pass (older records, tests); NaN
    reprojection (no MAMMA landmarks) is judged on the other criteria."""
    rep_px = GUARD_REP_PX if rep_px is None else rep_px
    rep_rel = GUARD_REP_REL if rep_rel is None else rep_rel
    arm_deg = GUARD_ARM_DEG if arm_deg is None else arm_deg
    why = []
    for rr, ss in movers:
        h = f"{rr}_{ss}"
        b, a = _guard_ref(rec, rr, ss), rec.get(f"{h}_rep_px_after", np.nan)
        if np.isfinite(a) and np.isfinite(b) and a > b + max(rep_px, rep_rel * b):
            why.append(f"{rr} {ss} arm landmarks {b:.2f} (ref) -> {a:.2f} px")
        cb, cbase, ca = (rec.get(f"{h}_cam_px_{x}") or {} for x in ("before", "base", "after"))
        for c, (pa, n) in ca.items():
            ref = max([v[0] for v in (cb.get(c), cbase.get(c)) if v is not None and np.isfinite(v[0])], default=np.nan)
            if n >= GUARD_CAM_MIN_N and np.isfinite(ref) and pa > ref + max(GUARD_CAM_PX, GUARD_CAM_REL * ref):
                why.append(f"{rr} {ss} cam {c} landmarks {ref:.1f} (ref) -> {pa:.1f} px")
        if f"{h}_shoulder_elbow_change_deg_max" in rec:     # shoulder / elbow, + the wrist outside its envelope
            d = rec[f"{h}_shoulder_elbow_change_deg_max"]
            if d > arm_deg:
                why.append(f"{rr} {ss} shoulder / elbow turned {d:.1f} deg")
            xw = rec.get(f"{h}_wrist_excess_deg_max", 0.0)
            if xw > 0:
                why.append(f"{rr} {ss} wrist turned {xw + arm_deg:.1f} deg away from the MAMMA fit's wrist")
        else:
            d = rec.get(f"{h}_arm_change_deg_max", 0.0)
            if d > arm_deg:
                why.append(f"{rr} {ss} arm turned {d:.1f} deg")
        for j in ("wrist", "elbow"):
            jb, ja = rec.get(f"{h}_{j}_jitter_mm_before", np.nan), rec.get(f"{h}_{j}_jitter_mm_after", np.nan)
            if np.isfinite(jb) and np.isfinite(ja) and ja > max(GUARD_JIT_REL * jb, jb + GUARD_JIT_MM):
                why.append(f"{rr} {ss} {j} jitter {jb:.2f} -> {ja:.2f} mm")
        dp, cap = rec.get(f"{h}_palm_disp_cm_p95_good", np.nan), rec.get(f"{h}_disp_cap_cm", np.nan)
        if np.isfinite(dp) and np.isfinite(cap) and dp > cap + GUARD_DISP_M * 100:
            why.append(f"{rr} {ss} palm moved {dp:.1f} cm (p95, cap {cap:.1f} cm)")
        xb, xa = rec.get(f"{h}_cross_frames_before"), rec.get(f"{h}_cross_frames_after")
        if xb is not None and xa is not None and xa > xb + max(2, 0.05 * rec.get("window_frames", 0)):
            why.append(f"{rr} {ss} finger crossings {xb} -> {xa} frames")
        ab, aa = rec.get(f"{h}_arm_pen_cm_max_before", np.nan), rec.get(f"{h}_arm_pen_cm_max_after", np.nan)
        if np.isfinite(ab) and np.isfinite(aa) and aa > ab + GUARD_ARM_PEN_CM:
            why.append(f"{rr} {ss} arm sinks deeper into the partner {ab:.1f} -> {aa:.1f} cm")
    return not why, why


def _guard_excess(rec: dict, movers) -> float:
    """How far an attempt is outside the guard (0 = inside), for reporting the least-bad attempt."""
    x = 0.0
    for rr, ss in movers:
        h = f"{rr}_{ss}"
        b, a = _guard_ref(rec, rr, ss), rec.get(f"{h}_rep_px_after", np.nan)
        if np.isfinite(a) and np.isfinite(b):
            tol = max(GUARD_REP_PX, GUARD_REP_REL * b)
            x += max(0.0, a - b - tol) / tol
        x += max(0.0, rec.get(f"{h}_shoulder_elbow_change_deg_max", rec.get(f"{h}_arm_change_deg_max", 0.0))
                 - GUARD_ARM_DEG) / GUARD_ARM_DEG + rec.get(f"{h}_wrist_excess_deg_max", 0.0) / GUARD_ARM_DEG
        for j in ("wrist", "elbow"):
            jb, ja = rec.get(f"{h}_{j}_jitter_mm_before", np.nan), rec.get(f"{h}_{j}_jitter_mm_after", np.nan)
            if np.isfinite(jb) and np.isfinite(ja):
                lim = max(GUARD_JIT_REL * jb, jb + GUARD_JIT_MM)
                x += max(0.0, ja - lim) / lim
    return x


def _ramp(n: int, a: int, b: int, R: int) -> np.ndarray:
    """Window weights over [a-R, b+R] (clipped to [0, n)): 1 inside [a, b], raised-cosine to 0 over R frames."""
    lo, hi = max(0, a - R), min(n - 1, b + R)
    k = np.arange(lo, hi + 1)
    w = np.ones(len(k))
    for j, kk in enumerate(k):
        if kk < a:
            w[j] = 0.5 - 0.5 * np.cos(np.pi * (kk - (a - R - 1)) / (R + 1))
        elif kk > b:
            w[j] = 0.5 - 0.5 * np.cos(np.pi * ((b + R + 1) - kk) / (R + 1))
    return k, np.clip(w, 0, 1)


def ma2d_obs(work: Path) -> tuple[dict, dict] | tuple[None, None]:
    """MAMMA's grid landmarks of the adopted fit: {cam: (lm (T,2,512,2), sigma px (T,2,512), vis (T,2,512))}."""
    sel = work / "mamma" / "select.json"
    tag = json.loads(sel.read_text()).get("chosen") if sel.exists() else None
    d = work / "mamma" / "ma_2d" / f"{work.name}_{tag}" if tag else None
    if d is None or not d.exists():
        return None, None
    out = {}
    for p in sorted(d.glob("cam*.npz")):
        z = np.load(p)
        lm = z["landmarks"].astype(float)
        sig = md.logvar_to_sigma_px(lm[..., 2], md.CROP_PX)   # ma_3d's crop-unit convention (x 512)
        out[p.stem[3:]] = (lm[..., :2], np.maximum(sig, 2.0), z["visibilities"].astype(float))
    return out, {"tag": tag, "dir": str(d)}


def kp_error_frames(J21: np.ndarray, ko: dict, gc: dict, frames: np.ndarray, min_pts: int = 10) -> np.ndarray:
    """(K,) per window frame: median over cameras of the median |projected hand point - keypoint detection| / hand
    length of the hand points J21 (K,21,3) against the fused detections ko {cam: (kp (T,21,2), w (T,21))}; NaN where
    no camera has >= min_pts confident keypoints."""
    out = np.full(len(frames), np.nan)
    for n, k in enumerate(frames):
        e = []
        for c, (kp, w) in ko.items():
            ok = w[k] > 0
            if ok.sum() < min_pts or not gc[c]["ok"][k]:
                continue
            P = gc[c]["K"][k] @ np.concatenate([gc[c]["R"][k], gc[c]["t"][k][:, None]], 1)
            p = _proj(P, J21[n])
            L = max(float(np.linalg.norm(p[0] - p[12])), 8.0)
            e.append(float(np.median(np.linalg.norm(p[ok] - kp[k, ok], axis=1))) / L)
        if e:
            out[n] = float(np.median(e))
    return out


def _finger_pairs(Vh, fid: np.ndarray, reach: float):
    """Finger-skin correspondences of one hand (K,nh,3) torch: for every finger vertex the nearest vertex of ANOTHER
    finger -> (a_idx (n,), b_idx (K,n) long, active (K,n) float: within reach)."""
    import torch
    a_idx = np.flatnonzero(fid > 0)
    A = Vh[:, a_idx].float()
    D = torch.cdist(A, A)                                                           # (K,n,n)
    same = torch.tensor(fid[a_idx][:, None] == fid[a_idx][None])
    D = D.masked_fill(same[None], float("inf"))
    d, j = D.min(2)
    return (torch.tensor(a_idx), torch.tensor(a_idx)[j], (d < reach).double())


def self_thresholds(D0: np.ndarray, lay: dict, floor: float, slack: float = 3e-4) -> np.ndarray:
    """(K,n,n) minimum allowed distance between capsule sample points of different fingers: r_a + r_b when the input
    keeps them apart, else no closer than the input, and at least floor x (r_a + r_b) (an input crossing is pushed
    apart); -1 (no constraint) for same-finger pairs."""
    Ssum = lay["radius"][:, None] + lay["radius"][None]
    thr = np.minimum(Ssum[None], np.maximum(D0, floor * Ssum[None])) - slack
    return np.where(lay["pair"][None], thr, -1.0)


def choose_attempt(attempts: list, movers, keep_failed: bool = False) -> tuple:
    """attempts [(record, result)] in order -> (adopted attempt or None, closest attempt, guard label). The first
    attempt that passes the guard is adopted; when none passes the input pose is kept (None) - unless keep_failed, the
    old policy that kept the attempt closest to the guard (the review found it shipping a hand 14-19 cm off the
    video)."""
    good = [a for a in attempts if a[0].get("guard_ok")]
    closest = min(attempts, key=lambda a: _guard_excess(a[0], movers))
    if good:
        return good[0], closest, "passed"
    if keep_failed:
        return closest, closest, "failed on every attempt: kept the attempt closest to the guard"
    return None, closest, "failed on every attempt: input pose kept"


class _Surface:
    """Closest points + inside flags of a mover's points (contacting hand, its arm) on the partner's body, frozen
    between refreshes as (face, barycentric) so the per-iteration distance |p - q| stays differentiable; q follows the
    partner's moving arm / hand vertices when the partner is a mover too (hand-hand holds)."""

    def __init__(self, faces: np.ndarray, topo, Vbase: np.ndarray, sub_map: np.ndarray | None, device: str,
                 max_depth: float):
        self.F, self.topo, self.Vbase, self.sub_map = faces, topo, Vbase, sub_map
        self.device, self.max_depth = device, max_depth

    def refresh(self, P: np.ndarray, Vt: np.ndarray, faces_all: np.ndarray | None = None, inside_test: bool = True):
        """P (K,n,3) current points, Vt (K,nv,3) current target body: closest points (exact) and inside flags
        (winding number over the closed body; brute force on a pod GPU)."""
        from . import mesh_contact as mc
        r = mc.signed_distance_frames(P, Vt, self.topo, self.F if faces_all is None else faces_all, self.device,
                                      self.max_depth) if inside_test else None
        if r is None:
            r = {"face": [], "bary": [], "dist": []}
            for k in range(len(P)):
                cp = mc.signed_distance(P[k], Vt[k], self.topo, None, self.device, self.max_depth, inside_test=False)
                for key in r:
                    r[key].append(cp[key])
            r = {key: np.stack(v) for key, v in r.items()}
            r["inside"] = np.zeros(r["dist"].shape, bool)
        face, bary, inside = r["face"].astype(np.int64), r["bary"], r["inside"]
        self.face, self.bary, self.inside = face, bary, inside
        K = len(P)
        vids = self.topo.F[face]                                                       # (K,n,3)
        kk = np.arange(K)[:, None, None]
        self.C0 = self.Vbase[kk, vids]                                                 # (K,n,3,3) fixed corners
        self.cpos = self.sub_map[vids] if self.sub_map is not None else None           # (K,n,3) or -1
        self._t = None
        return self

    def q(self, torch, Vsub=None):
        """(K,n,3) closest points on the target with its current mover vertices Vsub (K,n_sub,3) when it moves."""
        if self._t is None:
            mv = self.cpos is not None and (self.cpos >= 0).any()
            K, n = self.face.shape
            self._t = {"C0": torch.tensor(self.C0), "bary": torch.tensor(self.bary)[..., None], "mv": mv}
            if mv:
                self._t["idx"] = torch.tensor(np.maximum(self.cpos, 0).reshape(K, -1))[..., None].expand(-1, -1, 3)
                self._t["m"] = torch.tensor(self.cpos >= 0)[..., None]
        C = self._t["C0"]
        if self._t["mv"] and Vsub is not None:
            K, n = self.face.shape
            g = torch.gather(Vsub, 1, self._t["idx"]).reshape(K, n, 3, 3)
            C = torch.where(self._t["m"], g, C)
        return (self._t["bary"] * C).sum(-2)


class EpisodeRefiner:
    """Small IK for one contact episode: shoulder/elbow/wrist deltas + finger deltas of the contacting hand(s)."""

    def __init__(self, dancers: dict, gc: dict, cams: list[str], ma2d: dict | None, kp_obs: dict, cfg: RefineConfig,
                 device: str | None = None):
        from . import mesh_contact as mc
        self.dn, self.gc, self.cams, self.ma2d, self.kp_obs, self.cfg = dancers, gc, cams, ma2d, kp_obs, cfg
        self.device = device or mc.default_device()
        self.D_idx = np.asarray(md._load_verts512().argmax(1))       # 512 points -> SMPL-X vertex
        self.priors = {(r, s): HandPrior(dancers[r].mdir, s) for r in ROLES for s in SIDES}
        kin = dancers["lead"].kin
        self.faces = kin.faces
        self.topo = mc.Topology(self.faces, len(kin.weights_np))
        self.hand_topo = {}
        for s in SIDES:
            hv = hand_vertex_ids(dancers["lead"].mdir, s)
            self.hand_topo[s] = mc.Topology(self.faces[np.isin(self.faces, hv).all(1)], len(kin.weights_np))
        self.lay = {}
        self.hf = {s: mc.HandFaces(kin.weights_np, self.faces, s) for s in SIDES}
        for r in ROLES:
            k_ = dancers[r].kin
            for s in SIDES:
                self.lay[(r, s)] = mc.capsule_layout(mc.finger_radii(k_.v_shaped_np, k_.J_rest, k_.weights_np,
                                                                     k_.tips[s], s))

    def _setup(self, r: str, s: str, frames: np.ndarray):
        """Moving vertex subset of dancer r side s: arm (shoulder..fingers) + its 512 landmarks."""
        dn = self.dn[r]
        mdir = dn.mdir
        w = dn.kin.weights_np
        sh, el, wr = ARM_JOINTS[s]
        js = [sh, el, wr, *range(25 if s == "left" else 40, (25 if s == "left" else 40) + 15)]
        arm = np.flatnonzero(w[:, js].sum(1) > 0.3)
        hand = hand_vertex_ids(mdir, s)
        lm_pts = np.flatnonzero(np.isin(self.D_idx, arm))
        ids = np.unique(np.concatenate([arm, hand]))
        upper = np.flatnonzero(np.isin(w.argmax(1), UPPER_ARM_JOINTS[s]))[::max(1, self.cfg.arm_stride)]
        upper = upper[np.isin(upper, ids)]
        sub_map = -np.ones(len(w), np.int64)
        sub_map[ids] = np.arange(len(ids))
        return {"ids": ids, "hand_pos": np.searchsorted(ids, hand), "lm_pts": lm_pts,
                "lm_pos": np.searchsorted(ids, self.D_idx[lm_pts]), "frames": frames,
                "arm_pos": np.searchsorted(ids, upper), "sub_map": sub_map}

    def refine(self, ep: dict, partner_hand_side: str | None = None, cfg: RefineConfig | None = None) -> dict:
        """cfg overrides the refiner's config for this call (the guard retries more conservatively). The contact
        (attraction) terms act on attract_mask(ep) frames only (none when cfg.sink_only); penetration is removed over
        the whole ramped window. The dancers must carry the input poses (set_dancer_pose)."""
        import torch
        from . import mesh_contact as mc
        cfg = cfg or self.cfg
        r, s = ep["role"], ep["side"]
        o = OTHER[r]
        T = len(self.dn[r].t)
        frames, ramp = _ramp(T, ep["i0"], ep["i1"], cfg.ramp)
        K = len(frames)
        att = attract_mask(ep, T)[frames] & (not cfg.sink_only)
        bridged = np.zeros(T, bool)
        for a_, b_ in ep.get("bridged_gaps") or []:
            bridged[a_:b_ + 1] = True
        bridged = bridged[frames] & att
        # bridged hold gaps: the video shows the clasp but the input keeps the hands apart (2026-10-07 review: 2-7 cm),
        # so the attraction is stronger there and the palms may move up to the loose cap
        inside = torch.tensor(np.where(bridged, cfg.bridge_contact_mult, 1.0) * att, dtype=torch.float64)
        n_att = max(int(att.sum()), 1)
        wr_t = torch.tensor(ramp, dtype=torch.float64)
        movers = [(r, s)] + ([(o, partner_hand_side)] if partner_hand_side else [])
        B = torch.tensor(smooth_basis(K, cfg.basis_step, cfg.basis_sigma), dtype=torch.float64)
        Vfull = {rr: self.dn[rr].verts()[frames] for rr in ROLES}                    # input bodies on the window
        st, var = {}, []
        for (rr, ss) in movers:
            S = self._setup(rr, ss, frames)
            dn = self.dn[rr]
            kin = SmplxKin(dn.mdir, np.asarray(dn.d["betas"], float).reshape(-1), "cpu", vert_subset=S["ids"])
            S["kin"] = kin
            S["pose0"] = dn.pose[frames].clone()
            S["transl"] = dn.transl[frames]
            S["k_arm"] = torch.zeros((B.shape[1], 3, 3), dtype=torch.float64, requires_grad=True)
            S["k_fing"] = torch.zeros((B.shape[1], 15, 3), dtype=torch.float64, requires_grad=cfg.fingers_free)
            var += [S["k_arm"]] + ([S["k_fing"]] if cfg.fingers_free else [])
            hid_ = hand_vertex_ids(dn.mdir, ss)
            dom_ = dn.kin.weights_np[hid_].argmax(1)
            f0_ = 25 if ss == "left" else 40
            S["hand_fid"] = np.where((dom_ >= f0_) & (dom_ < f0_ + 15), 1 + (dom_ - f0_) // 3, 0)
            with torch.no_grad():
                out0 = kin.forward(S["pose0"], S["transl"], want_verts=True)
                J0 = SmplxKin.hand21(out0, ss).numpy()
            S["V0"] = out0["verts"].numpy()
            S["palm0"] = torch.tensor(J0[:, list(PALM21)].mean(1))
            kerr = kp_error_frames(J0, self.kp_obs.get((rr, ss), {}), self.gc, frames)
            cap = np.where(np.nan_to_num(kerr, nan=0.0) > cfg.kp_bad_len, cfg.disp_cap_bad_m, cfg.disp_cap_m)
            cap = np.where(bridged, cfg.disp_cap_bad_m, cap)
            if cfg.sink_only:
                cap = np.minimum(cap, cfg.sink_cap_m)
            S["cap"], S["kp_err0"] = cap, kerr
            lay = self.lay[(rr, ss)]
            X0 = mc.capsule_points(J0, lay)
            D0 = np.linalg.norm(X0[:, :, None] - X0[:, None], axis=-1)
            S["self_thr"] = torch.tensor(self_thresholds(D0, lay, cfg.self_floor))
            st[(rr, ss)] = S
        # target surfaces: each mover's hand + upper-arm points on the OTHER dancer's closed body; in a hand-hand hold
        # also the partner's hand alone (attraction target)
        for (rr, ss), S in st.items():
            tg = OTHER[rr]
            tm = next(((a, b) for (a, b) in st if a == tg), None)
            sub_map = st[tm]["sub_map"] if tm else None
            S["surf"] = _Surface(self.faces, self.topo, Vfull[tg], sub_map, self.device, cfg.max_depth_m)
            S["tm"], S["tg"] = tm, tg
            if partner_hand_side and tm:
                S["hsurf"] = _Surface(self.faces, self.hand_topo[tm[1]], Vfull[tg], sub_map, self.device,
                                      cfg.max_depth_m)
        opt = torch.optim.Adam(var, lr=cfg.lr)
        reach = {}

        def posed(S, ss):
            P = S["pose0"].clone()
            f0 = 25 if ss == "left" else 40
            for n, j in enumerate(ARM_JOINTS[ss]):
                P[:, j] = P[:, j] + wr_t[:, None] * S["d_arm"][:, n]
            P[:, f0:f0 + 15] = P[:, f0:f0 + 15] + wr_t[:, None, None] * S["d_fing"]
            out = S["kin"].forward(P, S["transl"], want_verts=True)
            return P, out

        def evaluate():
            res = {}
            for (rr, ss), S in st.items():
                S["d_arm"] = torch.einsum("kn,nab->kab", B, S["k_arm"])        # smooth per-frame corrections
                S["d_fing"] = torch.einsum("kn,nab->kab", B, S["k_fing"])
                res[(rr, ss)] = posed(S, ss)
            return res

        def target_now(S, res):
            """(K,nv,3) numpy target body with the partner mover's current vertices."""
            V = Vfull[S["tg"]]
            if S["tm"] is None:
                return V
            V = V.copy()
            V[:, st[S["tm"]]["ids"]] = res[S["tm"]][1]["verts"].detach().numpy()
            return V

        hist = []
        tau = cfg.softmin_tau_m
        for it in range(cfg.iters + 1):
            opt.zero_grad()
            res = evaluate()
            terms = {}
            if it % cfg.nn_every == 0 and cfg.fingers_free and cfg.w_fself > 0:
                for (rr, ss), S in st.items():
                    Vh = res[(rr, ss)][1]["verts"][:, S["hand_pos"]].detach()
                    S["fnn"] = _finger_pairs(Vh, S["hand_fid"], cfg.fself_reach_m)
            if it % cfg.nn_every == 0:
                for (rr, ss), S in st.items():
                    Vv = res[(rr, ss)][1]["verts"].detach().numpy()
                    pts = np.concatenate([Vv[:, S["hand_pos"]], Vv[:, S["arm_pos"]]], 1)
                    Vt = target_now(S, res)
                    S["surf"].refresh(pts, Vt)
                    if "hsurf" in S:
                        S["hsurf"].refresh(Vv[:, S["hand_pos"]], Vt, inside_test=False)
            pen = torch.zeros(K, dtype=torch.float64)
            pen_arm = torch.zeros(K, dtype=torch.float64)
            sds = []
            for (rr, ss), S in st.items():
                Vv = res[(rr, ss)][1]["verts"]
                pts = torch.cat([Vv[:, S["hand_pos"]], Vv[:, S["arm_pos"]]], 1)
                Vsub = res[S["tm"]][1]["verts"] if S["tm"] else None
                q = S["surf"].q(torch, Vsub)
                d = (pts - q).norm(dim=-1).clamp(min=1e-9)
                ins = torch.tensor(S["surf"].inside, dtype=torch.float64)
                nh = len(S["hand_pos"])
                rho = cfg.pen_scale_m ** 2 * torch.log1p((d / cfg.pen_scale_m) ** 2)   # ~d^2 below the scale
                pen = pen + (ins[:, :nh] * rho[:, :nh]).sum(1)
                pen_arm = pen_arm + (ins[:, nh:] * rho[:, nh:]).sum(1)
                if "hsurf" in S:          # hand-hand: distance to the partner's hand, negative inside the partner
                    qh = S["hsurf"].q(torch, Vsub)
                    dh = (Vv[:, S["hand_pos"]] - qh).norm(dim=-1)
                    sds.append(torch.where(ins[:, :nh] > 0, -d[:, :nh], dh))
                else:
                    sds.append(torch.where(ins[:, :nh] > 0, -d[:, :nh], d[:, :nh]))
            sd = torch.cat(sds, 1)
            smin = -tau * torch.logsumexp(-sd / tau, 1)                                 # (K,) soft closest gap
            terms["contact"] = cfg.w_contact * (inside * (smin - cfg.contact_target_m) ** 2).sum() / \
                n_att
            terms["pen"] = cfg.w_pen * (wr_t * pen).sum() / K
            terms["pen_arm"] = cfg.w_arm_pen * (wr_t * pen_arm).sum() / K
            # fingertips onto the partner surface (only fingers that start within reach and face the partner)
            outa = res[(r, s)][1]
            tipv = outa["tips"][:, :5] if s == "left" else outa["tips"][:, 5:]
            if partner_hand_side:
                tgtV = res[(o, partner_hand_side)][1]["verts"][:, st[(o, partner_hand_side)]["hand_pos"]]
            else:
                tgtV = torch.tensor(Vfull[o])
            dmat = torch.cdist(tipv, tgtV)                                                # (K,5,n)
            d_t, j_t = dmat.min(2)
            if it == 0:
                with torch.no_grad():
                    qn = torch.gather(tgtV, 1, j_t[..., None].expand(-1, -1, 3))
                    n_palm = torch.einsum("kab,b->ka", outa["R"][:, WRIST[s]], torch.tensor(PALM_REST))
                    v = qn - tipv
                    cos = (v * n_palm[:, None]).sum(-1) / v.norm(dim=-1).clamp(min=1e-9)
                    reach["m"] = ((d_t < cfg.finger_reach_m) & (cos > cfg.finger_facing)).double()
            terms["fingers"] = cfg.w_finger * (inside[:, None] * reach["m"] * torch.relu(d_t - 0.006) ** 2).sum() / \
                n_att
            # --- image terms: MAMMA's landmarks of the moved arm(s), hand keypoints of the contacting hand
            terms["rep"] = torch.zeros((), dtype=torch.float64)
            terms["kp"] = torch.zeros((), dtype=torch.float64)
            for (rr, ss), S in st.items():
                P, out = res[(rr, ss)]
                if self.ma2d:
                    pid = ROLES.index(rr)
                    Vl = out["verts"][:, S["lm_pos"]]                                    # (K, nl, 3)
                    num, den = 0.0, 0.0
                    for c in self.cams:
                        if c not in self.ma2d:
                            continue
                        lm, sig, vis = self.ma2d[c]
                        g = self.gc[c]
                        x = torch.tensor(lm[frames][:, pid][:, S["lm_pts"]])
                        sg = torch.tensor(sig[frames][:, pid][:, S["lm_pts"]])
                        vv = torch.tensor(vis[frames][:, pid][:, S["lm_pts"]] * g["ok"][frames][:, None])
                        Pm = torch.tensor(np.einsum("tij,tjk->tik", g["K"][frames],
                                                    np.concatenate([g["R"][frames], g["t"][frames][..., None]], -1)))
                        y = torch.einsum("tij,tnj->tni", Pm, torch.cat([Vl, torch.ones_like(Vl[..., :1])], -1))
                        p2 = y[..., :2] / y[..., 2:].clamp(min=1e-6)
                        r2 = ((p2 - x) ** 2).sum(-1) / sg ** 2
                        num = num + (vv * _robust(torch, r2)).sum()
                        den = den + vv.sum()
                    terms["rep"] = terms["rep"] + cfg.w_rep * num / max(float(den), 1.0) * 100
                ko = self.kp_obs.get((rr, ss), {})
                J21 = SmplxKin.hand21(out, ss)
                if ko:
                    num, den = 0.0, 0.0
                    for c, (kp, w) in ko.items():
                        g = self.gc[c]
                        Pm = torch.tensor(np.einsum("tij,tjk->tik", g["K"][frames],
                                                    np.concatenate([g["R"][frames], g["t"][frames][..., None]], -1)))
                        y = torch.einsum("tij,tnj->tni", Pm, torch.cat([J21, torch.ones_like(J21[..., :1])], -1))
                        p2 = y[..., :2] / y[..., 2:].clamp(min=1e-6)
                        L = (p2[:, 0] - p2[:, 12]).norm(dim=-1).detach().clamp(min=8.0)
                        ww = torch.tensor(w[frames] * g["ok"][frames][:, None])
                        r2 = ((p2 - torch.tensor(kp[frames])) ** 2).sum(-1) / (0.1 * L[:, None]) ** 2
                        num = num + (ww * _robust(torch, r2)).sum()
                        den = den + ww.sum()
                    terms["kp"] = terms["kp"] + cfg.w_kp * num / max(float(den), 1.0) * 100
                # finger spacing (capsule proxy of finger-finger collisions)
                X = mc.capsule_points(J21, self.lay[(rr, ss)])
                Dx = (X[:, :, None] - X[:, None]).norm(dim=-1)
                terms[f"self_{rr}_{ss}"] = cfg.w_self * (torch.relu(S["self_thr"] - Dx) ** 2).sum() / 2 / K
                if cfg.fingers_free and cfg.w_fself > 0 and S.get("fnn") is not None:
                    a_i, b_i, act = S["fnn"]
                    Vh = out["verts"][:, S["hand_pos"]]
                    Nh = _torch_normals(Vh, self._hand_faces(rr, ss))
                    pb = torch.gather(Vh, 1, b_i[..., None].expand(-1, -1, 3))
                    nb = torch.gather(Nh, 1, b_i[..., None].expand(-1, -1, 3))
                    sdf = ((Vh[:, a_i] - pb) * nb).sum(-1)                              # (K, n_finger_verts)
                    if "fpen0" not in S:
                        S["fpen0"] = torch.relu(-sdf.detach())
                    terms[f"fself_{rr}_{ss}"] = cfg.w_fself * (act * torch.relu(-sdf - S["fpen0"] - 5e-4) ** 2).sum() / K
                # palm displacement cap
                palm = J21[:, list(PALM21)].mean(1)
                dp = (palm - S["palm0"]).norm(dim=-1)
                terms[f"disp_{rr}_{ss}"] = cfg.w_disp * (wr_t * torch.relu(dp - torch.tensor(S["cap"])) ** 2).sum() / K
                # priors and smoothness
                f0 = 25 if ss == "left" else 40
                pr = self.priors[(rr, ss)]
                th = P[:, f0:f0 + 15].reshape(K, 45)
                cpca = (th - torch.tensor(pr.mean)) @ torch.tensor(pr.pinv)
                terms[f"prior_{rr}_{ss}"] = cfg.w_arm_prior * (S["d_arm"] ** 2).sum((1, 2)).mean() + \
                    cfg.w_finger_prior * (S["d_fing"] ** 2).sum((1, 2)).mean() + cfg.w_pca * (cpca ** 2).sum(1).mean()
                terms[f"smooth_{rr}_{ss}"] = cfg.w_smooth * (((S["d_arm"][1:] - S["d_arm"][:-1]) ** 2).sum((1, 2)).mean()
                                                             + ((S["d_fing"][1:] - S["d_fing"][:-1]) ** 2).sum((1, 2)).mean())
            loss = sum(terms.values())
            if it == 0 or it == cfg.iters:
                hist.append({k: float(v) for k, v in terms.items()})
            if it == cfg.iters:
                break
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = {k: (P.numpy(), {kk: vv.numpy() for kk, vv in out.items() if kk in ("joints", "verts", "tips")})
                     for k, (P, out) in evaluate().items()}
        return {"frames": frames, "ramp": ramp, "final": final, "terms_first": hist[0], "terms_last": hist[-1],
                "setup": {k: {kk: v[kk] for kk in ("ids", "hand_pos", "lm_pts", "lm_pos", "arm_pos", "cap", "kp_err0",
                                                    "V0")}
                          for k, v in st.items()}}

    _hf_cache: dict = {}

    def _hand_faces(self, r: str, s: str):
        """Faces of the hand subset (indices into the hand_pos ordering) for normals."""
        import torch
        key = (r, s)
        if key not in self._hf_cache:
            ids = hand_vertex_ids(self.dn[r].mdir, s)
            f = self.dn[r].kin.faces
            keep = np.isin(f, ids).all(1)
            remap = -np.ones(f.max() + 1, np.int64)
            remap[ids] = np.arange(len(ids))
            self._hf_cache[key] = torch.tensor(remap[f[keep]])
        return self._hf_cache[key]


def _torch_normals(V, faces):
    """(K,n,3) vertices + (F,3) faces -> (K,n,3) area-weighted unit vertex normals."""
    import torch
    v0, v1, v2 = V[:, faces[:, 0]], V[:, faces[:, 1]], V[:, faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    N = torch.zeros_like(V)
    for k in range(3):
        N = N.index_add(1, faces[:, k], fn)
    return N / N.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _arm_rep_px(dn_pose: np.ndarray, r: str, s: str, frames: np.ndarray, dancers: dict, gc: dict, ma2d: dict | None,
                setup: dict, kin: SmplxKin) -> tuple[float, dict]:
    """Median pixel error of MAMMA's visible (vis > 0.5) arm + hand landmarks over the window: all cameras, and per
    camera {cam: (median px, n observations)}."""
    import torch
    if not ma2d:
        return float("nan"), {}
    with torch.no_grad():
        out = kin.forward(torch.tensor(dn_pose), dancers[r].transl[frames], want_verts=True)
    V = out["verts"][:, setup["lm_pos"]].numpy()
    pid = ROLES.index(r)
    errs, per = [], {}
    for c, (lm, sig, vis) in ma2d.items():
        g = gc[c]
        ec = []
        for n, k in enumerate(frames):
            if not g["ok"][k]:
                continue
            ok = vis[k, pid, setup["lm_pts"]] > 0.5
            if not ok.any():
                continue
            P = g["K"][k] @ np.concatenate([g["R"][k], g["t"][k][:, None]], 1)
            p = _proj(P, V[n][ok])
            ec.extend(np.linalg.norm(p - lm[k, pid, setup["lm_pts"]][ok], axis=1).tolist())
        if ec:
            per[c] = (round(float(np.median(ec)), 2), len(ec))
        errs.extend(ec)
    return (float(np.median(errs)) if errs else float("nan")), per


def _gap_series(Vh: np.ndarray, Vo: np.ndarray, faces: np.ndarray, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per frame signed closest gap (m) and max penetration (m) of hand vertices Vh (K,n,3) vs the closed body Vo
    (K,V,3): exact distances, generalized winding number inside test (mesh_contact)."""
    from . import mesh_contact as mc
    topo = mc.Topology(faces, Vo.shape[1])
    sd = mc.signed_distance_frames(Vh, Vo, topo, device=device)["sd"]
    gap = sd.min(1)
    return gap, np.maximum(0.0, -gap)


def set_dancer_pose(dn: Dancer, P: np.ndarray) -> None:
    """Give a Dancer a new full pose (T,55,3): contract pose keys, FK joints, hand points; the mesh cache is dropped so
    the next verts() (a partner surface in a later episode) is the updated body."""
    import torch
    P = np.asarray(P, float)
    T = len(P)
    dn.d = dict(dn.d, global_orient=P[:, 0], body_pose=P[:, 1:22].reshape(T, 63), jaw_pose=P[:, 22],
                leye_pose=P[:, 23], reye_pose=P[:, 24], left_hand_pose=P[:, 25:40].reshape(T, 45),
                right_hand_pose=P[:, 40:55].reshape(T, 45))
    dn.pose = torch.tensor(P, dtype=torch.float64)
    with torch.no_grad():
        dn.out = dn.kin.forward(dn.pose, dn.transl)
        dn.hands = {s: dn.kin.hand21(dn.out, s).numpy() for s in SIDES}
    dn.joints = dn.out["joints"].numpy()
    dn._verts = None


def _jitter_mm(X: np.ndarray) -> float:
    from .hands_detail import hf_rms
    return round(hf_rms(X) * 1000, 2)


def _episode_record(res: dict, ep: dict, n: int, ps: str | None, new_pose: dict, dancers: dict, gc: dict,
                    ma2d: dict | None, geo: dict, ref: EpisodeRefiner, base_pose: dict | None = None) -> dict:
    """Before / after numbers of one refinement attempt: per moved arm MAMMA's arm + hand landmark reprojection (all
    cameras and per camera), arm / finger change, wrist / elbow jitter, palm displacement, finger crossings; for the
    contacting hand the signed gap to the partner's closed body (winding numbers) over the attracted frames."""
    from . import mesh_contact as mc
    r, s = ep["role"], ep["side"]
    frames = res["frames"]
    K = len(frames)
    rec = {"episode": n, "role": r, "side": s, "partner_hand": ps, "frames": [int(frames[0]), int(frames[-1])],
           "window_frames": K, "terms_first": res["terms_first"], "terms_last": res["terms_last"]}
    Vafter = {rr: dancers[rr].verts()[frames].copy() for rr in ROLES}
    for (rr, ss), (P, outF) in res["final"].items():
        Vafter[rr][:, res["setup"][(rr, ss)]["ids"]] = outF["verts"]
    for (rr, ss), (P, outF) in res["final"].items():
        S = res["setup"][(rr, ss)]
        h = f"{rr}_{ss}"
        kin = SmplxKin(dancers[rr].mdir, np.asarray(dancers[rr].d["betas"], float).reshape(-1), "cpu",
                       vert_subset=S["ids"])
        P0 = new_pose[rr][frames]
        for tag, PP in (("before", P0), ("after", P)) + ((("base", base_pose[rr][frames]),) if base_pose else ()):
            a, per = _arm_rep_px(PP, rr, ss, frames, dancers, gc, ma2d, S, kin)
            rec[f"{h}_rep_px_{tag}"] = round(a, 2)
            rec[f"{h}_cam_px_{tag}"] = per
        dpose = np.degrees(np.linalg.norm(P - P0, axis=-1))
        rec[f"{h}_arm_change_deg_max"] = round(float(dpose[:, list(ARM_JOINTS[ss])].max()), 2)
        rec[f"{h}_arm_change_deg_by_joint"] = {n_: round(float(dpose[:, j].max()), 2) for n_, j in
                                               zip(("shoulder", "elbow", "wrist"), ARM_JOINTS[ss])}
        rec[f"{h}_shoulder_elbow_change_deg_max"] = round(float(dpose[:, list(ARM_JOINTS[ss][:2])].max()), 2)
        # the wrist may turn > GUARD_ARM_DEG only back towards the adopted (MAMMA) fit's wrist: per frame the new
        # wrist must not end further from MAMMA's than the input was (+ GUARD_WRIST_SLACK_DEG) - i.e. the contact step
        # may give back what the wrist gate changed, never invent a new wrist
        from scipy.spatial.transform import Rotation as _R
        wj = ARM_JOINTS[ss][2]
        dw = dpose[:, wj]
        if base_pose is not None:
            Rb = _R.from_rotvec(base_pose[rr][frames][:, wj])
            to_base_in = np.degrees((Rb.inv() * _R.from_rotvec(P0[:, wj])).magnitude())
            to_base_out = np.degrees((Rb.inv() * _R.from_rotvec(P[:, wj])).magnitude())
        else:
            to_base_in, to_base_out = np.zeros(len(dw)), dw
        bad_w = (dw > GUARD_ARM_DEG) & (to_base_out > to_base_in + GUARD_WRIST_SLACK_DEG)
        rec[f"{h}_wrist_excess_deg_max"] = round(float(np.where(bad_w, dw - GUARD_ARM_DEG, 0.0).max()), 2)
        rec[f"{h}_wrist_toward_mamma_frames"] = int(((dw > GUARD_ARM_DEG) & ~bad_w).sum())
        rec[f"{h}_arm_change_deg_p95"] = round(float(np.percentile(dpose[:, list(ARM_JOINTS[ss])].max(1), 95)), 2)
        f0 = 25 if ss == "left" else 40
        rec[f"{h}_finger_change_deg_mean"] = round(float(dpose[:, f0:f0 + 15].mean()), 2)
        import torch
        with torch.no_grad():
            J0 = kin.forward(torch.tensor(P0), dancers[rr].transl[frames])["joints"].numpy()
        J1 = outF["joints"]
        for j, jj in (("wrist", WRIST[ss]), ("elbow", 18 if ss == "left" else 19)):
            rec[f"{h}_{j}_jitter_mm_before"] = _jitter_mm(J0[:, jj][:, None])
            rec[f"{h}_{j}_jitter_mm_after"] = _jitter_mm(J1[:, jj][:, None])
        J21a = np.stack([J1[:, k] if k < 55 else outF["tips"][:, (k - 66) % 5 + (0 if ss == "left" else 5)]
                         for k in (HAND21[ss])], 1)
        with torch.no_grad():
            o0 = kin.forward(torch.tensor(P0), dancers[rr].transl[frames])
        J21b = SmplxKin.hand21(o0, ss).numpy()
        disp = np.linalg.norm(J21a[:, list(PALM21)].mean(1) - J21b[:, list(PALM21)].mean(1), axis=1)
        cap = np.asarray(S["cap"])
        good = cap <= cap.min() + 1e-9
        rec[f"{h}_disp_cap_cm"] = round(float(cap.min()) * 100, 2)
        rec[f"{h}_palm_disp_cm_median"] = round(float(np.median(disp)) * 100, 2)
        rec[f"{h}_palm_disp_cm_p95_good"] = round(float(np.percentile(disp[good], 95)) * 100, 2) if good.any() else None
        rec[f"{h}_palm_disp_cm_max"] = round(float(disp.max()) * 100, 2)
        hf = ref.hf[ss]
        Vb = dancers[rr].verts()[frames]
        rec[f"{h}_cross_frames_before"] = int((mc.crossing_series(Vb, hf)[0] > 0).sum())
        rec[f"{h}_cross_frames_after"] = int((mc.crossing_series(Vafter[rr], hf)[0] > 0).sum())
        # arm penetration into the partner (upper arm + forearm), before / after
        up = np.flatnonzero(np.isin(dancers[rr].kin.weights_np.argmax(1), UPPER_ARM_JOINTS[ss]))[::2]
        o_ = OTHER[rr]
        g0 = mc.signed_distance_frames(Vb[:, up], dancers[o_].verts()[frames], ref.topo, device=ref.device)["sd"]
        g1 = mc.signed_distance_frames(Vafter[rr][:, up], Vafter[o_], ref.topo, device=ref.device)["sd"]
        rec[f"{h}_arm_pen_cm_max_before"] = round(max(0.0, -float(g0.min())) * 100, 2)
        rec[f"{h}_arm_pen_cm_max_after"] = round(max(0.0, -float(g1.min())) * 100, 2)
        rec[f"{h}_arm_pen_gt_1cm_before"] = round(float((g0.min(1) < -0.01).mean()), 3)
        rec[f"{h}_arm_pen_gt_1cm_after"] = round(float((g1.min(1) < -0.01).mean()), 3)
    S = res["setup"][(r, s)]
    hid = hand_vertex_ids(dancers[r].mdir, s)
    ins = attract_mask(ep, len(dancers[r].t))[frames]
    if not ins.any():
        ins = (frames >= ep["i0"]) & (frames <= ep["i1"])
    g0, p0 = geo[(r, s)]["gap"][frames], geo[(r, s)]["pen"][frames]
    sd1 = mc.signed_distance_frames(Vafter[r][:, hid], Vafter[OTHER[r]], ref.topo, device=ref.device)["sd"]
    g1 = sd1.min(1)
    p1 = np.maximum(0.0, -g1)
    rec.update({"gap_cm_median_before": round(float(np.median(g0[ins])) * 100, 2),
                "gap_cm_median_after": round(float(np.median(g1[ins])) * 100, 2),
                "in_contact_frac_before": round(float((np.abs(g0[ins]) <= 0.01).mean()), 3),
                "in_contact_frac_after": round(float((np.abs(g1[ins]) <= 0.01).mean()), 3),
                "sunk_gt_1cm_before": round(float((g0[ins] < -0.01).mean()), 3),
                "sunk_gt_1cm_after": round(float((g1[ins] < -0.01).mean()), 3),
                "float_gt_1cm_before": round(float((g0[ins] > 0.01).mean()), 3),
                "float_gt_1cm_after": round(float((g1[ins] > 0.01).mean()), 3),
                "penetration_cm_max_before": round(float(p0[ins].max()) * 100, 2),
                "penetration_cm_max_after": round(float(p1[ins].max()) * 100, 2),
                "penetration_cm_max_window_before": round(float(p0.max()) * 100, 2),
                "penetration_cm_max_window_after": round(float(p1.max()) * 100, 2),
                "attracted_frames": int(ins.sum()), "_gap_after": g1})
    return rec


def merge_hold_records(ep: dict, episodes: list, series: dict, kinds: tuple, ccfg) -> tuple[dict, list]:
    """A hand-hand hold is found from both hands: the union of both records, attracted where both records see contact
    plus short gaps the better record still sees (bridge_runs). -> (merged episode copy, twin records)."""
    r, s = ep["role"], ep["side"]
    ps = "left" if ep["region"].startswith("left") else "right"
    twin = [e for e in episodes if e is not ep and e["kind"] in kinds and e["hand_hand"] and
            e["role"] == OTHER[r] and e["side"] == ps and e["i0"] <= ep["i1"] and e["i1"] >= ep["i0"]]
    if not twin:
        return dict(ep), []
    runs = [[max(ep["i0"], e["i0"]), min(ep["i1"], e["i1"])] for e in twin]
    runs = [x for x in runs if x[0] <= x[1]]
    ev = np.maximum(series[(r, s)]["evidence"], series[(OTHER[r], ps)]["evidence"])
    runs, gaps = bridge_runs(runs, ev, ccfg.bridge_max_gap, ccfg.bridge_min_ev, ccfg.bridge_med_ev)
    i0 = min([ep["i0"]] + [e["i0"] for e in twin])
    i1 = max([ep["i1"]] + [e["i1"] for e in twin])
    return dict(ep, i0=i0, i1=i1, attract_runs=runs, bridged_gaps=gaps, union=[i0, i1], partner_side=ps), twin


def run_contact(take: Take, ccfg: ContactConfig | None = None, rcfg: RefineConfig | None = None, refine: bool = True,
                smplx_dir: Path | None = None, kinds: tuple = ("hold", "touch"), sheets: bool = True) -> dict:
    """Contact episodes of the fit in smplx_dir (default smplx/) and their refinement -> hands/contact.json,
    contact.npz, hands/smplx/ (refined contract fits). sheets=False skips the older review/hands_contact.jpg (the build
    stage draws its own)."""
    import torch
    from . import mesh_contact as mc
    ccfg, rcfg = ccfg or ContactConfig(), rcfg or RefineConfig()
    work = take.work
    out = work / "hands"
    out.mkdir(parents=True, exist_ok=True)
    cams = fit_cams(work)
    t0 = time.time()
    tracks = md.load_tracks(work, cams)
    dancers = {r: Dancer(work, r, smplx_dir) for r in ROLES}
    grid = dancers["lead"].t
    gc = grid_cameras(work, tracks, grid)
    hand_ids = {s: hand_vertex_ids(dancers["lead"].mdir, s) for s in SIDES}
    device = mc.default_device()
    geo = hand_partner_geometry(dancers, hand_ids, device=device)
    ev = view_evidence(work, dancers, tracks, cams, ccfg)
    episodes, series = contact_episodes(geo, ev, grid, ccfg)
    in_dir = Path(smplx_dir) if smplx_dir is not None else fit_dir(work)
    summary = {"take": take.name, "cams": cams, "contact_config": asdict(ccfg), "refine_config": asdict(rcfg),
               "input_fit": str(in_dir), "input_label": _fit_label(work, in_dir), "device": device,
               "gap_method": "exact closest point + generalized winding number (mesh_contact)", "episodes": episodes}
    arrays = {"t": grid}
    for (r, s), sr in series.items():
        for k, v in sr.items():
            arrays[f"{r}_{s}_{k}"] = v
        arrays[f"{r}_{s}_region"] = geo[(r, s)]["region"]
        arrays[f"{r}_{s}_pen"] = geo[(r, s)]["pen"]
        arrays[f"{r}_{s}_view_gap_len"] = ev["gap_len"][(r, s)]
    if refine:
        ma2d, ma_info = ma2d_obs(work)
        summary["ma2d"] = ma_info
        kp_obs = {}
        if (out / "grid_obs.npz").exists():
            kp_obs = load_grid_obs(work, "fused")
        ref = EpisodeRefiner(dancers, gc, cams, ma2d, kp_obs, rcfg, device)
        new_pose = {r: dancers[r].pose.numpy().copy() for r in ROLES}
        base_pose = None   # the adopted fit, when refining a finger / wrist stage's output (guard reference)
        if smplx_dir is not None and Path(smplx_dir).resolve() != fit_dir(work).resolve():
            from .hand_kin import full_pose
            base_pose = {r: full_pose(load_fit(work, r)) for r in ROLES}
        touched = {r: np.zeros(len(grid), bool) for r in ROLES}
        done_pairs = set()
        results = []
        for n, ep in enumerate(episodes):
            if ep["kind"] not in kinds:
                continue
            r, s = ep["role"], ep["side"]
            ps = None
            if ep["hand_hand"]:
                ps = "left" if ep["region"].startswith("left") else "right"
                merged, twin = merge_hold_records(ep, episodes, series, kinds, ccfg)
                if any(id(e) in done_pairs for e in twin):
                    ep["refined"] = "merged with the partner's record of the same hold"
                    continue
                for k_ in ("attract_runs", "bridged_gaps", "union", "partner_side"):
                    if k_ in merged:
                        episodes[n][k_] = merged[k_]
                for e in twin:
                    e["merged_into"] = n
                ep = merged
                done_pairs.add(id(episodes[n]))
            # start from the current (possibly already refined) poses, partner surfaces included
            for rr in ROLES:
                set_dancer_pose(dancers[rr], new_pose[rr])
            movers = [(r, s)] + ([(OTHER[r], ps)] if ps else [])
            attempts = []
            steps = GUARD_STEPS if rcfg.guard else GUARD_STEPS[:1]
            for ai, (fa, fr, fc) in enumerate(steps):
                tt0 = time.time()
                cfg_a = replace(rcfg, w_arm_prior=rcfg.w_arm_prior * fa, w_finger_prior=rcfg.w_finger_prior * fa,
                                w_self=rcfg.w_self * fa, w_rep=rcfg.w_rep * fr, w_contact=rcfg.w_contact * fc,
                                w_finger=rcfg.w_finger * fc, sink_only=fc == 0,
                                w_arm_pen=rcfg.w_arm_pen if ai == 0 else 0.0,
                                fingers_free=rcfg.fingers_free and GUARD_FINGERS_FREE[ai])
                res = ref.refine(ep, ps, cfg_a)
                rec = _episode_record(res, ep, n, ps, new_pose, dancers, gc, ma2d, geo, ref, base_pose)
                ok, why = guard_ok(rec, movers)
                rec.update({"attempt": ai, "multipliers": [fa, fr, fc], "sink_only": fc == 0,
                            "fingers_free": cfg_a.fingers_free, "guard_ok": ok,
                            "guard_why": why, "seconds": round(time.time() - tt0, 1)})
                attempts.append((rec, res))
                print(f"[contact] ep {n} {r} {s} -> {ep['region']} ({ep['kind']}, {ep['t0']:.2f}-{ep['t1']:.2f}s) "
                      f"attempt {ai} x{fa:g}/{fr:g}/{fc:g}{' sink-only' if fc == 0 else ''}"
                      f"{'' if cfg_a.fingers_free else ' fingers kept'}: gap "
                      f"{rec['gap_cm_median_before']} -> {rec['gap_cm_median_after']} cm, pen "
                      f"{rec['penetration_cm_max_before']} -> {rec['penetration_cm_max_after']} cm, arm rep "
                      f"{rec.get(f'{r}_{s}_rep_px_before')} -> {rec.get(f'{r}_{s}_rep_px_after')} px, arm "
                      f"{rec.get(f'{r}_{s}_arm_change_deg_max')} deg, palm p95 "
                      f"{rec.get(f'{r}_{s}_palm_disp_cm_p95_good')} cm, crossings "
                      f"{rec.get(f'{r}_{s}_cross_frames_before')} -> {rec.get(f'{r}_{s}_cross_frames_after')}, "
                      f"guard {'ok' if ok else why} ({rec['seconds']} s)", flush=True)
                if ok:
                    break
            adopt, closest, glabel = choose_attempt(attempts, movers, rcfg.keep_failed)
            good = glabel == "passed"
            rec = dict((adopt or closest)[0])
            rec["guard"] = glabel
            rec["adopted"] = adopt is not None
            rec["attempts"] = [{k: a[0].get(k) for k in a[0] if not k.startswith("terms") and k != "_gap_after"}
                               for a in attempts]
            res = (adopt or closest)[1]
            frames = res["frames"]
            if adopt is not None:
                for (rr, ss), (P, _) in res["final"].items():
                    new_pose[rr][frames] = P
                    touched[rr][frames] |= res["ramp"] > 0
                arrays[f"ep{n}_gap_after"] = rec.pop("_gap_after")
            else:
                rec.pop("_gap_after")
                arrays[f"ep{n}_gap_after"] = geo[(r, s)]["gap"][frames]
            arrays[f"ep{n}_frames"] = frames
            results.append(rec)
            episodes[n]["refined"] = ("yes" if good else "yes (guard failed: closest attempt kept)" if adopt else
                                      "no (guard failed on every attempt: input kept)")
        summary["refined"] = results
        # refined fits: the contract npz with the corrected arm / hand poses (+ FK joints), smplx/ is untouched
        sdir = out / "smplx"
        sdir.mkdir(parents=True, exist_ok=True)
        for r in ROLES:
            d = dict(dancers[r].d)
            P = new_pose[r]
            d["body_pose"] = P[:, 1:22].reshape(len(P), 63)
            d["left_hand_pose"] = P[:, 25:40].reshape(len(P), 45)
            d["right_hand_pose"] = P[:, 40:55].reshape(len(P), 45)
            with torch.no_grad():
                d["joints"] = dancers[r].kin.forward(torch.tensor(P), dancers[r].transl)["joints"].numpy()
            d["hands_refined"] = touched[r]
            d["source"] = np.array(str(d.get("source", "")) + " + dancecap.refine_hands contact")
            np.savez_compressed(sdir / f"{r}.npz", **d)
        for rr in ROLES:
            set_dancer_pose(dancers[rr], new_pose[rr])
    np.savez_compressed(out / "contact.npz", **arrays)
    summary["runtime_s"] = round(time.time() - t0, 1)
    review = work / "review"
    paths = [_contact_timeline(arrays, summary, review / "hands_contact_timeline.png", take.name)]
    if refine and sheets:
        paths += _contact_sheets(work, take.name, summary, tracks, cams, review)
    summary["review"] = [str(p) for p in paths]
    (out / "contact.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def _fit_label(work: Path, d: Path) -> str:
    """Human label of a contract fit directory (for plots): MAMMA fit, fingers + wrist gate, ..."""
    try:
        rel = Path(d).resolve().relative_to(work.resolve()).as_posix()
    except ValueError:
        rel = Path(d).name
    return {"smplx": "MAMMA fit (smplx/)", "hands/smplx_fingers_build": "fingers + wrist gate (contact input)",
            "hands/smplx_fingers_hybrid": "fingers hybrid", "hands/smplx_fingers_ik": "fingers IK"}.get(rel, rel)


def _contact_timeline(arrays: dict, summary: dict, path: Path, take: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = arrays["t"]
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    kc = {"hold": "#2ca02c", "touch": "#ffbf00", "near-miss": "#d62728"}
    n = 0
    for r in ROLES:
        for s in SIDES:
            ax = axes[n]
            gap = arrays[f"{r}_{s}_gap"] * 100
            ax.plot(t, np.clip(gap, -5, 15), color="k", lw=1.2,
                    label=f"gap before refinement (cm): {summary.get('input_label', 'input fit')}")
            ax.plot(t, arrays[f"{r}_{s}_evidence"] * 10, color="#1f77b4", lw=1, alpha=0.8,
                    label="2D evidence x10 (fraction of deciding views that agree)")
            ax.plot(t, arrays[f"{r}_{s}_score"] * 10, color="#9467bd", lw=1, ls="--", label="contact score x10")
            for e in summary["episodes"]:
                if e["role"] == r and e["side"] == s:
                    ax.axvspan(t[e["i0"]], t[e["i1"]], color=kc[e["kind"]], alpha=0.25)
                    ax.text((t[e["i0"]] + t[e["i1"]]) / 2, 13.5, f"{e['kind']}\n{e['region']}", ha="center",
                            va="top", fontsize=7)
            for k in range(200):
                if f"ep{k}_frames" not in arrays:
                    continue
                ref = summary.get("refined", [])
                hit = [x for x in ref if x["episode"] == k]
                if hit and hit[0]["role"] == r and hit[0]["side"] == s:
                    lab = ("gap after refinement (cm)" if hit[0].get("adopted", True) else
                           "gap after refinement (cm): guard failed, input kept")
                    ax.plot(t[arrays[f"ep{k}_frames"]], np.clip(arrays[f"ep{k}_gap_after"] * 100, -5, 15), color="#e377c2",
                            lw=2, label=lab)
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_ylim(-5, 15)
            ax.set_ylabel("cm (negative = sunk in)")
            ax.set_title(f"{LABEL[r]} {s} hand -> {LABEL[OTHER[r]]}", fontsize=10)
            from matplotlib.patches import Patch
            hd_, lb_ = ax.get_legend_handles_labels()
            hd_ += [Patch(color=kc[k_], alpha=0.25) for k_ in kc]
            lb_ += [f"{k_} episode" for k_ in kc]
            ax.legend(hd_, lb_, fontsize=7, loc="upper right", ncol=3)
            n += 1
    axes[-1].set_xlabel("reference time (s)")
    fig.suptitle(f"{take}: hand contact episodes (green hold, amber touch, red near-miss); signed gap = exact distance to "
                 f"the partner, inside by winding number")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def render_meshes(meshes: list, K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int):
    """meshes [(V (n,3), faces (F,3), rgb 0..1)] -> (H,W,3) float image (Lambert, headlight), coverage mask."""
    import torch
    from .avatar_raster import rasterize
    Vs, Fs, Cs, off = [], [], [], 0
    for V, F, col in meshes:
        Vs.append(V)
        Fs.append(F + off)
        Cs.append(np.tile(np.asarray(col, float), (len(F), 1)))
        off += len(V)
    V, F, C = np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Cs)
    Xc = V @ R.T + t
    z = Xc[:, 2]
    px = np.stack([K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-4) + K[0, 2], K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-4) + K[1, 2]], 1)
    zb, fid = rasterize(torch, torch.tensor(px), torch.tensor(z), torch.tensor(F), H, W)
    fid = fid.numpy()
    tri = Xc[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    view = -tri.mean(1)
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    shade = 0.25 + 0.75 * np.abs((n * view).sum(1))
    img = np.zeros((H, W, 3))
    cov = fid >= 0
    img[cov] = C[fid[cov]] * shade[fid[cov], None]
    return img, cov


def _contact_sheets(work: Path, take: str, summary: dict, tracks: dict, cams: list[str], review: Path,
                    max_eps: int = 6, tile: int = 220) -> list[Path]:
    """Per refined episode (up to max_eps, longest first): the mid frame in the two cameras where the hand is largest:
    video crop | MAMMA meshes overlaid | refined meshes overlaid, and a 3D close-up before / after."""
    import cv2
    import torch
    paths = []
    refined = sorted(summary.get("refined", []), key=lambda x: -(x["frames"][1] - x["frames"][0]))[:max_eps]
    if not refined:
        return paths
    orig = {r: Dancer(work, r) for r in ROLES}
    newd = {r: Dancer(work, r, work / "hands" / "smplx") for r in ROLES}
    faces = orig["lead"].kin.faces
    col = {"lead": (0.95, 0.45, 0.35), "follow": (0.35, 0.6, 0.95)}
    rows = []
    for rec in refined:
        ep = summary["episodes"][rec["episode"]]
        r, s = ep["role"], ep["side"]
        k = (ep["i0"] + ep["i1"]) // 2
        tk = orig[r].t[k]
        best = []
        for c in cams:
            tr = tracks[c]
            i = int(md.nearest_index(tr.t, np.array([tk]))[0])
            if not tr.ok[i]:
                continue
            px, z = project(orig[r].hands[s][k], tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i]))
            L = float(np.linalg.norm(px[0] - px[12]))
            if (px[:, 0] >= 0).all() and (px[:, 0] < tr.width).all() and (px[:, 1] >= 0).all() and (px[:, 1] < tr.height).all():
                best.append((L, c, i, px))
        best.sort(key=lambda b: -b[0])
        tiles = []
        for L, c, i, px in best[:2]:
            tr = tracks[c]
            img = cv2.imread(str(work / "frames" / c / f"{i:06d}.jpg"))
            side = max(4.0 * L, 60)
            cx, cy = px.mean(0)
            box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
            crop = crop_pad(img, box, tile, cv2.INTER_CUBIC)
            sc = tile / (box[2] - box[0])
            Kc = tr.K[i].copy()
            Kc[0, 0] *= sc
            Kc[1, 1] *= sc
            Kc[0, 2] = (Kc[0, 2] - box[0]) * sc
            Kc[1, 2] = (Kc[1, 2] - box[1]) * sc
            t_i = tr.t[i]
            for name, src in (("MAMMA", orig), ("refined", newd)):
                meshes = [(src[rr].at(src[rr].verts(), t_i), faces, col[rr]) for rr in ROLES]
                ren, cov = render_meshes(meshes, Kc, tr.R[i], tr.tvec[i], tile, tile)
                ov = crop.astype(float) / 255
                ov[cov] = 0.45 * ov[cov] + 0.55 * ren[cov][:, ::-1]
                ov = (ov * 255).astype(np.uint8)
                label(ov, f"{name}", (3, 13), scale=0.4)
                tiles.append(ov)
            label(crop, f"cam {c} t={t_i:.2f}s", (3, 13), scale=0.4)
            tiles.insert(len(tiles) - 2, crop)
        # 3D close-up around the contact point, before / after
        Vb = orig[r].verts()[k]
        ctr = orig[r].hands[s][k].mean(0)
        fwd = torso_forward(orig[r].joints[k:k + 1])[0]
        for ang in (90.0,):
            a = np.deg2rad(ang)
            Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            C = ctr + Ry @ fwd * 0.45 + np.array([0, 0.12, 0])
            from .avatar_review import _look_at
            Rv, tv = _look_at(C, ctr)
            Kv = np.array([[tile * 1.6, 0, tile / 2], [0, tile * 1.6, tile / 2], [0, 0, 1.0]])
            for name, src in (("MAMMA 3D", orig), ("refined 3D", newd)):
                meshes = [(src[rr].verts()[k], faces, col[rr]) for rr in ROLES]
                ren, cov = render_meshes(meshes, Kv, Rv, tv, tile, tile)
                im = np.where(cov[..., None], ren, 0.12)
                im = (np.clip(im, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
                label(im, name, (3, 13), scale=0.4)
                tiles.append(im)
        row = np.concatenate(tiles, 1) if tiles else None
        if row is None:
            continue
        hdr = np.full((22, row.shape[1], 3), 25, np.uint8)
        label(hdr, f"{LABEL[r]} {s} hand -> {LABEL[OTHER[r]]} {ep['region']} ({ep['kind']}, {ep['t0']:.2f}-{ep['t1']:.2f}s): "
                   f"gap {rec['gap_cm_median_before']} -> {rec['gap_cm_median_after']} cm, penetration "
                   f"{rec['penetration_cm_max_before']} -> {rec['penetration_cm_max_after']} cm, arm landmarks "
                   f"{rec.get(f'{r}_{s}_rep_px_before')} -> {rec.get(f'{r}_{s}_rep_px_after')} px", (5, 15), scale=0.42)
        rows.append(np.concatenate([hdr, row], 0))
    if rows:
        w = max(x.shape[1] for x in rows)
        sheet = np.concatenate([np.pad(x, ((0, 3), (0, w - x.shape[1]), (0, 0))) for x in rows], 0)
        head = np.full((28, w, 3), 10, np.uint8)
        label(head, f"{take}: contact refinement (lead red, follow blue): per camera video | MAMMA | refined; then 3D "
                    f"close-ups", (6, 19), scale=0.5)
        p = review / "hands_contact.jpg"
        cv2.imwrite(str(p), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(p)
    return paths


# ------------------------------------------------------------------------------------------------ CLI

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["survey", "detectors", "mano", "fingers", "contact", "compare", "build", "review"])
    ap.add_argument("take")
    ap.add_argument("--no-loo", action="store_true", help="fingers: skip the leave-one-camera-out fits")
    ap.add_argument("--detector", default="fused", help="fingers: rtmw | rtmh | mp | wilor | hamer | fused")
    ap.add_argument("--mano", default="", help="fingers: hybrid route, pull towards the fused MANO pose of this model "
                                                "(wilor | hamer; needs `mano` first)")
    ap.add_argument("--mano-weight", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="fingers: output name suffix (fingers_<tag>.json, smplx_fingers_<tag>/)")
    ap.add_argument("--workers", type=int, default=1, help="fingers: parallel processes for the held-out fits")
    ap.add_argument("--models", default="", help="mano: comma-separated models (default: every mano_<model>.npz)")
    ap.add_argument("--smplx-dir", default="", help="fingers / contact / mano: start from this contract dir "
                                                     "(e.g. work/<take>/hands/smplx_fingers_hybrid) instead of smplx/")
    ap.add_argument("--no-refine", action="store_true", help="contact: detect episodes only")
    ap.add_argument("--no-review", action="store_true", help="build: no frame-based review images (pod)")
    ap.add_argument("--video-only", action="store_true", help="review: only hands_contact.jpg (video crops, local)")
    ap.add_argument("--mesh-only", action="store_true", help="review: no video crops")
    ap.add_argument("--at", default="", help="review: extra video-crop moments at these reference times (s, comma-sep.)")
    a = ap.parse_args(argv)
    if a.stage in ("build", "review"):   # hands_detail: wrist gate, fingers, contact, smplx_hands/, smplx_detail/
        from . import hands_detail
        return hands_detail.main([a.stage, a.take, "--workers", str(a.workers)] + (["--no-loo"] if a.no_loo else [])
                                 + (["--no-review"] if a.no_review else []) + (["--video-only"] if a.video_only else [])
                                 + (["--mesh-only"] if a.mesh_only else []) + (["--at", a.at] if a.at else []))
    take = load_take(a.take)
    for d in ("hands", "review"):   # a pod gets neither from the bundle (matplotlib cannot save into a missing dir)
        (take.work / d).mkdir(parents=True, exist_ok=True)
    sd = Path(a.smplx_dir) if a.smplx_dir else None
    if sd is not None and not sd.is_absolute():
        sd = (Path.cwd() / sd) if (Path.cwd() / sd).exists() else take.work.parent.parent / sd
    if a.stage == "survey":
        s = run_survey(take)
        for r, v in s["roles"].items():
            for side, cs in v.items():
                print(r, side, {c: (st["len_px_median"], st["visible_ge_0.8"]) for c, st in cs.items()})
    elif a.stage == "detectors":
        s = run_detectors(take)
        print(json.dumps(s["grid"], indent=1))
    elif a.stage == "mano":
        s = run_mano(take, tuple(x for x in a.models.split(",") if x) or None, smplx_dir=sd)
    elif a.stage == "fingers":
        cfg = FingerConfig(detector=a.detector, mano_model=a.mano, mano_weight=a.mano_weight, tag=a.tag)
        s = run_fingers(take, cfg, loo=not a.no_loo, smplx_dir=sd, workers=a.workers)
    elif a.stage == "compare":
        s = run_compare(take)
        print(json.dumps(s["hands"], indent=1))
    else:
        s = run_contact(take, refine=not a.no_refine, smplx_dir=sd)
        for e in s["episodes"]:
            print(e)
    print("review:", s.get("review"), "runtime", s.get("runtime_s"), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
