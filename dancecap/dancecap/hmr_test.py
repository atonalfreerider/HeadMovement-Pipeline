"""lucas_luiza: does a newer monocular HMR method beat our single-camera fit? ExMR now, FlowHMR later (2026-10-27).

Owner, 2026-10-27: licences accepted (ExMR code + weights, Sapiens2, SMPL-X / MANO); FlowHMR is NOT run now (no SMPL-H
neutral model on this PC): ExMR only, the scorer keeps a slot for flowhmr. Simple acceptance: one sheet plus a few numbers.

Each candidate runs on the tripod cut only (camera 02 of lucas_luiza = work/src_cuts/lucas_luiza_h/lucas_luiza_h.mp4,
1280x720, 30/1 fps CFR, the pixels baseline B1 saw), once per dancer per clip, with our teacher boxes and our camera 02
(K, R) injected, and is scored in the held-out Steadicam (camera 01) and against the two-view fit.

    python -m dancecap.hmr_test sheet-frames --take lucas_luiza     # local, read-only: the 12 camera-01 jpgs (job inputs)
    python -m dancecap.hmr_test clips --take lucas_luiza --clips E1,E2,H1,H3 --margin 30    # pod job lucas_luiza_hmr_prep
    python -m dancecap.hmr_test score --take lucas_luiza --methods B0,B1,B2,exmr,flowhmr    # pod job lucas_luiza_hmr_score
    python -m dancecap.hmr_test sheet --take lucas_luiza                                    # pod job lucas_luiza_hmr_score
    (the ExMR runs between them: remote/cand_hmr/exmr_pod.py in job lucas_luiza_exmr)

Clocks: ll grid t = k / 30 (k = i + 507, i = row of the stitch smplx_rf2_noslip); tripod-cut frame n = k - 420
(h time + 14.000 s = ll time, frame exact; the h fit's row is i + 3). Camera 02 row: nearest to t (delta 0, |dt| <= 17
ms); camera 01 row: nearest to mamma_dyn.offset_label(t) within 20 ms, joints interpolated to that row's corrected time.
Clips (ll grid k, both ends included; run with a 30-frame margin on each side, scored on the core):
    E1 3273-3507 closed hold, E2 4500-4818 open spins, H1 2520-2880 embrace / occlusion, H3 5490-5949 fast spins.

Contract: every method becomes camera-frame joints J_c (body joints 0-21, the same order in SMPL-H and SMPL-X) plus the
intrinsics K_used it predicted them with. B0 = the two-view stitch (reference, world joints); B1 = the mono tripod fit
lucas_luiza_h/smplx_rf2_noslip (h world -> h camera, K_used = h K); B2 = SAM 3D Body with the mask prompt
(lucas_luiza_h/realism/sam3db/pod_out/smplx_mask255.npz, K_used = input/meta.json K); exmr = joints_incam[:, :22],
K_used = K_fullimg; flowhmr = <clip>_<role>/joints_cam02.npz (J_c, K) written by its pod adapter, "anchored".
Placements into camera 02 (check error 1, the pelvis RE-RAY): P' = z' K02^-1 (K_used P / P_z), J' = J - P + P' (the pelvis
keeps its pixel in the method's own image; the body keeps its shape). A: z' = P_z f02 / f_used; B: z' = B0's pelvis depth
in camera 02. World: X_w = R02^T (J' - t02). The old slide along the method's own ray (J + (z'/P_z - 1) P) is kept only for
the scorer self-check against the basis numbers.
Metrics per dancer, per clip and pooled over the four clips (good-reference frames; all held-out frames as well):
    M1   held-out camera-01 error, placement B: 12 COCO-mapped joints vs people/teachers_01 kp (joint conf >= 0.5,
         >= 6 joints, dancer conf >= 0.5), per frame median px / teacher box height (%): median, p90, share > 10 %
    M1b  in-view check: the method's joints in its OWN image (its own K, no placement) vs people/teachers_02 kp
    M1c  wrong-dancer share: camera-02 frames whose projection is closer to the partner's kp than to the dancer's own
    M2   PA-MPJPE (22 joints, per-frame similarity Procrustes) vs B0; root-relative world MPJPE after placement B
    M3   placement-A pelvis error and couple-vector (lead - follow pelvis) error vs B0
    M4   pelvis jitter 4.5-12 Hz (analysis_review.root_jitter), ankle / toe skate on B0's stance frames, lowest-joint
         floor spread (reported only)
Pass bar (owner, 2026-10-27): a method "improves accuracy" for a dancer when its pooled M1-B AND its pooled PA-MPJPE are
both no worse than B1 (the current single-camera fit) AND no worse than B2 (the SAM 3D Body prior), on the frames all
of them scored (good-reference frames). B0 is the reference row. The scorer must first reproduce the basis numbers
(SELF_CHECK, check error 10 wording); when it does not, the sheet says "INVALID: scorer" and gives no verdict. A method
whose run failed its integrity checks is "not run correctly", which is not the same as "worse".

OUTPUTS (work/lucas_luiza/hmr_test/, likeness-derived, never committed):
    clips/<clip>/{clip.mp4, boxes.npz, cam02.npz, meta.json, sheet02_<n>.jpg}   (prep)
    exmr/<clip>_<role>/results.npz + exmr/run.json, exmr_smoke/                 (lucas_luiza_exmr)
    score/{score.json, per_frame.npz, sheet_points.npz, sheet.md, sheet.json, contact_cam01.jpg, contact_cam02.jpg}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .carlos_aline_fitcheck import COCO_TO_SMPLX, project
from .jl_fitcheck import BONES, frame_map
from .take import PACKAGE_ROOT, WORKSPACE

WORK = PACKAGE_ROOT / "work"
TAKE, H_TAKE = "lucas_luiza", "lucas_luiza_h"
ROLES = ("lead", "follow")
CLIPS = {"E1": (3273, 3507), "E2": (4500, 4818), "H1": (2520, 2880), "H3": (5490, 5949)}   # ll grid k, core
CLIP_NOTE = {"E1": "closed hold", "E2": "open spins", "H1": "embrace / occlusion", "H3": "fast spins"}
N_FROM_K = 420            # tripod-cut frame n = ll grid k - 420
H_CLOCK_S = 14.0          # h (tripod cut) time + 14.000 s = ll time
FPS = 30.0
NJ = 22                   # body joints 0-21 (SMPL / SMPL-H / SMPL-X order)
MAX_DT = 0.02             # camera-01 and teacher rows (jl_fitcheck.frame_map max_dt; its default is 0.04)
MAX_DT_02 = 0.017         # camera-02 rows of the clips (largest measured: 15.8 ms)
CONF = 0.5
MIN_JOINTS = 6
MIN_BOX_H = 50.0
GROSS_PCT = 10.0
F_SPREAD_MAX = 0.005      # K02 focal within a clip: |f / median - 1| (no zoom step inside a clip)
DIAG_MM = float(np.hypot(36.0, 24.0))   # ExMR's f_mm: full-frame (36 x 24 mm) equivalent over the image diagonal
SHEET_FRACTIONS = (0.25, 0.5, 0.75)
SHEET_SEARCH = 15         # frames either side of a sheet time to find a localised camera-01 row
METHODS = ("B0", "B1", "B2", "exmr", "flowhmr")
CANDIDATES = ("exmr", "flowhmr")
LABEL = {"B0": "B0 two-view", "B1": "B1 mono fit", "B2": "B2 SAM 3D Body", "exmr": "ExMR", "flowhmr": "FlowHMR"}
COLOR = {"B0": (255, 255, 255), "B1": (0, 255, 255), "B2": (255, 255, 0), "exmr": (0, 255, 0), "flowhmr": (255, 0, 255)}
COLOR_NAME = {"B0": "white", "B1": "yellow", "B2": "cyan", "exmr": "green", "flowhmr": "magenta"}   # BGR above
NOT_RUN = {"flowhmr": "not run (owner 2026-10-27: no SMPL-H neutral model; FlowHMR is not run now)"}
COCO = sorted(COCO_TO_SMPLX)
SMX = [COCO_TO_SMPLX[c] for c in COCO]
SCHEMA = "dancecap.hmr_test/1"
# the scorer self-check (check error 10 wording): basis numbers, lead / follow, +- tolerance. PA is the per-frame MEDIAN
# (the mean is 86 / 73); the good-reference count is checked twice: the mask before the ">= 6 joints" filter (676 / 589)
# and the frames B0 actually scored (660 / 578); held-out = every frame B0 scored in camera 01.
SELF_CHECK = (
    ("B0 pooled M1, good-reference frames (%)", "b0_m1_good", (2.66, 2.67), 0.15),
    ("B1 M1 placement B, OLD pelvis slide (%)", "b1_m1_naive_good", (5.47, 5.98), 0.2),
    ("B1 PA-MPJPE, per-frame median (mm)", "b1_pa_good", (81.0, 64.0), 3.0),
    ("good-reference mask, before the >= 6 joint filter (frames)", "n_good_mask", (676, 589), 2),
    ("good-reference frames B0 scored (frames)", "n_good", (660, 578), 2),
    ("held-out frames B0 scored (frames)", "n_held", (901, 1003), 2),
)
B2_FRAME_MAX = 5.0        # B2's camera frame confirmed when its own-image M1b on E1 is below this (measured 1.9 / 1.7 %)


def out_dir(take: str = TAKE) -> Path:
    return WORK / take / "hmr_test"


def log(msg: str) -> None:
    print(f"[hmr_test {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ pure helpers


def clip_cut_range(k0: int, k1: int, margin: int) -> tuple[int, int, int]:
    """Core grid span k0..k1 (both included) -> the tripod-cut frames n0..n1 with the margin, and their count L."""
    n0, n1 = k0 - N_FROM_K - margin, k1 - N_FROM_K + margin
    return n0, n1, n1 - n0 + 1


def grid_index_of_cut(n, kg0: int):
    """Tripod-cut frame n -> ll grid row i (k = n + 420, i = k - kg0)."""
    return np.asarray(n) + N_FROM_K - kg0


def cut_of_grid_index(i, kg0: int):
    """ll grid row i -> tripod-cut frame n (lucas_luiza: n = i + 87)."""
    return np.asarray(i) + kg0 - N_FROM_K


def h_index(tg: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ll grid times -> rows of an h-clock series (h t + 14 s = ll t) and whether that row exists within 1 ms."""
    i = np.round((np.asarray(tg) - H_CLOCK_S - th[0]) * FPS).astype(int)
    ok = (i >= 0) & (i < len(th))
    ok[ok] &= np.abs(th[i[ok]] + H_CLOCK_S - np.asarray(tg)[ok]) < 1e-3
    return np.where(ok, i, 0), ok


def f_mm_of(f_px: float, width: int, height: int) -> float:
    """ExMR's --f_mm (full-frame equivalent) whose estimate_K gives exactly f_px for this image size."""
    return float(f_px) * DIAG_MM / float(np.hypot(width, height))


def reray(Jc: np.ndarray, K_used: np.ndarray, K_new: np.ndarray, z_new) -> np.ndarray:
    """The pelvis re-ray (check error 1). Jc (F, J, 3) camera-frame joints predicted with K_used (F, 3, 3): the pelvis
    keeps its pixel u = K_used P / P_z, moves onto that pixel's ray under K_new at depth z_new (F,), and the body moves
    with it unchanged (translation only): P' = z' K_new^-1 u, J' = J - P + P'."""
    Jc = np.asarray(Jc, float)
    P = Jc[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.einsum("fij,fj->fi", np.asarray(K_used, float), P)
        u = u / u[:, 2:3]
        ray = np.einsum("fij,fj->fi", np.linalg.inv(np.asarray(K_new, float)), u)
        Pn = ray / ray[:, 2:3] * np.asarray(z_new, float)[:, None]
    return Jc - P[:, None] + Pn[:, None]


def placement_a(Jc, K_used, K02) -> np.ndarray:
    """Placement A: the method's own depth, focal-corrected: z' = P_z f02 / f_used (re-rayed)."""
    Jc = np.asarray(Jc, float)
    z = Jc[:, 0, 2] * np.asarray(K02)[:, 0, 0] / np.asarray(K_used)[:, 0, 0]
    return reray(Jc, K_used, K02, z)


def placement_b(Jc, K_used, K02, z_ref) -> np.ndarray:
    """Placement B: the reference's pelvis depth z_ref in camera 02 (re-rayed)."""
    return reray(Jc, K_used, K02, z_ref)


def slide_naive(Jc, z_new) -> np.ndarray:
    """The OLD placement (self-check only): slide along the method's own pelvis ray to depth z_new. Under K02 the image
    then moves away from the centre by f02 / f_used whenever the method's focal is wrong (B1, B2): check error 1."""
    Jc = np.asarray(Jc, float)
    P = Jc[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = np.asarray(z_new, float) / P[:, 2]
    return Jc + ((lam - 1.0)[:, None] * P)[:, None]


def to_world(Jc, R, t) -> np.ndarray:
    """Camera-frame joints (F, J, 3) of a world-to-camera (R, t) per frame -> world: X_w = R^T (J - t)."""
    return np.einsum("fji,fnj->fni", np.asarray(R, float), np.asarray(Jc, float) - np.asarray(t, float)[:, None])


def to_camera(Xw, R, t) -> np.ndarray:
    return np.einsum("fij,fnj->fni", np.asarray(R, float), np.asarray(Xw, float)) + np.asarray(t, float)[:, None]


def interp_grid(J: np.ndarray, tg: np.ndarray, tq: np.ndarray) -> np.ndarray:
    """Linear interpolation of per-grid-frame values J (T, ...) on the uniform grid tg to the times tq; NaN outside."""
    dt = float(tg[1] - tg[0])
    x = (np.asarray(tq, float) - tg[0]) / dt
    i0 = np.floor(x).astype(int)
    ok = (i0 >= 0) & (i0 + 1 < len(tg))
    i0c = np.clip(i0, 0, len(tg) - 2)
    w = (x - i0c).reshape((-1,) + (1,) * (J.ndim - 1))
    out = J[i0c] * (1 - w) + J[i0c + 1] * w
    out[~ok] = np.nan
    return out


def pinhole(Jc: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(J, 3) camera-frame points -> (J, 2) pixels with K (no distortion: a method's own image) and their depth."""
    z = Jc[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        zz = np.where(z > 1e-6, z, np.nan)
        uv = np.stack([K[0, 0] * Jc[:, 0] / zz + K[0, 2], K[1, 1] * Jc[:, 1] / zz + K[1, 2]], -1)
    return uv, z


def box_h(box) -> float:
    return max(float(box[3] - box[1]), MIN_BOX_H)


def frame_score(uv12: np.ndarray, z12: np.ndarray, kp17: np.ndarray, height: float) -> tuple[float, float]:
    """One frame, the 12 COCO-mapped joints (COCO order of COCO_TO_SMPLX): projected uv (12, 2) and depth (12,) against a
    teacher's kp (17, 3) -> (median px error / height in %, median px): joints with conf >= 0.5 that are in front of
    the camera, at least 6 of them; else (nan, nan)."""
    c = kp17[COCO, 2] >= CONF
    if c.sum() < MIN_JOINTS or not np.isfinite(uv12[c]).all() or not np.isfinite(z12[c]).all():
        return np.nan, np.nan
    e = np.linalg.norm(uv12[c] - kp17[COCO][c, :2], axis=1)[z12[c] > 0]
    if len(e) < MIN_JOINTS:
        return np.nan, np.nan
    px = float(np.median(e))
    return 100.0 * px / max(float(height), MIN_BOX_H), px


def teacher_ok(T: dict, row: int, slot: int) -> bool:
    return bool(T["row"][row, slot] >= 0 and T["conf"][row, slot] >= CONF)


def pa_mpjpe(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean joint distance (mm) after the similarity transform (rotation, scale, translation) that best maps X onto Y."""
    mx, my = X.mean(0), Y.mean(0)
    X0, Y0 = X - mx, Y - my
    U, S, Vt = np.linalg.svd(X0.T @ Y0)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    Rm = (U @ D @ Vt).T
    sc = (S * np.diag(D)).sum() / (X0 ** 2).sum()
    return float(np.linalg.norm(sc * X0 @ Rm.T - Y0, axis=1).mean() * 1000.0)


def summary(v) -> dict:
    """{n, median, p90, gross (share > 10 %)} of the finite values."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0, "median": None, "p90": None, "gross": None}
    return {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "gross": float(np.mean(v > GROSS_PCT))}


def wrong_dancer(own_px: float, partner_px: float) -> float:
    """1 when the projection is closer to the partner's kp than to the dancer's own, 0 when not, nan when unknown."""
    if not (np.isfinite(own_px) and np.isfinite(partner_px)):
        return np.nan
    return float(partner_px < own_px)


def no_worse(value, *baselines, ndigits: int) -> bool | None:
    """value <= every baseline at the sheet's display precision (a tie passes); None when anything is missing."""
    if value is None or any(b is None for b in baselines):
        return None
    return all(round(float(value), ndigits) <= round(float(b), ndigits) for b in baselines)


def verdict(m1b: dict, pa: dict) -> dict:
    """The owner's pass bar for one dancer. m1b / pa: {"B1": x, "B2": y, "cand": z} pooled medians on the common frames.
    -> {"improves": bool | None, "m1b_ok", "pa_ok", "text"}"""
    a = no_worse(m1b.get("cand"), m1b.get("B1"), m1b.get("B2"), ndigits=2)
    b = no_worse(pa.get("cand"), pa.get("B1"), pa.get("B2"), ndigits=1)
    if a is None or b is None:
        return {"improves": None, "m1b_ok": a, "pa_ok": b, "text": "no verdict (missing values)"}
    ok = bool(a and b)
    return {"improves": ok, "m1b_ok": a, "pa_ok": b, "text": "improves accuracy" if ok else "not better"}


def self_check(measured: dict) -> dict:
    """measured: {key: (lead, follow)} for the keys of SELF_CHECK -> {"ok", "rows": [...]}; a missing value fails."""
    rows, ok = [], True
    for name, key, target, tol in SELF_CHECK:
        got = measured.get(key, (None, None))
        passed = [g is not None and np.isfinite(g) and abs(float(g) - t) <= tol + 1e-9 for g, t in zip(got, target)]
        ok &= all(passed)
        rows.append({"name": name, "key": key, "target": list(target), "tol": tol,
                     "got": [None if g is None else float(g) for g in got], "ok": all(passed)})
    b2 = measured.get("b2_m1b_e1", (None, None))
    passed = [g is not None and np.isfinite(g) and float(g) <= B2_FRAME_MAX for g in b2]
    ok &= all(passed)
    rows.append({"name": "B2 camera frame: own-image M1b on E1 (%)", "key": "b2_m1b_e1", "target": [B2_FRAME_MAX] * 2,
                 "tol": "max", "got": [None if g is None else float(g) for g in b2], "ok": all(passed)})
    return {"ok": bool(ok), "rows": rows}


def sheet_times(c1_t: np.ndarray, c1_ok: np.ndarray, label_of, clips: dict | None = None) -> list[dict]:
    """The 12 fixed sheet times (25 / 50 / 75 % of each clip's core, chosen before any result exists) and their camera-01
    rows. label_of(t_ref) -> camera-01 frame time (mamma_dyn.offset_label). A time whose camera-01 row is missing or not
    localised moves to the nearest cut frame (+1, -1, +2, ...) that has one."""
    out = []
    for clip, (k0, k1) in (CLIPS if clips is None else clips).items():
        nc0, nc1 = k0 - N_FROM_K, k1 - N_FROM_K
        for frac in SHEET_FRACTIONS:
            n_want = int(round(nc0 + frac * (nc1 - nc0)))
            for d in [0] + [s * j for j in range(1, SHEET_SEARCH + 1) for s in (1, -1)]:
                n = n_want + d
                if not nc0 <= n <= nc1:
                    continue
                t_ref = (n + N_FROM_K) / FPS
                lab = float(np.asarray(label_of(np.array([t_ref])))[0])
                row, ok = frame_map(np.array([lab]), np.asarray(c1_t), max_dt=MAX_DT)
                row = int(row[0])
                if ok[0] and bool(c1_ok[row]):
                    out.append({"clip": clip, "frac": frac, "n": n, "k": n + N_FROM_K, "t": t_ref, "row01": row,
                                "moved": d})
                    break
            else:
                raise SystemExit(f"{clip} {frac}: no localised camera-01 row within {SHEET_SEARCH} frames of n {n_want}")
    return out


# ------------------------------------------------------------------------------------------------ sheet-frames (local)


def cmd_sheet_frames(args) -> None:
    """Read-only: print the job-input entries of the 12 camera-01 jpgs the contact sheet draws on."""
    from . import mamma_dyn as md
    ll = WORK / args.take
    c1 = np.load(ll / "cameras" / "01.npz")
    off1 = md.read_sync_offsets(ll).get("01")
    times = sheet_times(c1["t"], c1["ok"], lambda t: md.offset_label(t, off1))
    entries = [{"local": str(ll / "frames" / "01" / f"{d['row01']:06d}.jpg"),
                "pod": f"dancecap/work/{args.take}/hmr_test/sheet_frames/{d['row01']:06d}.jpg"} for d in times]
    missing = [e["local"] for e in entries if not Path(e["local"]).exists()]
    if missing:
        raise SystemExit(f"missing camera-01 frames: {missing}")
    print(json.dumps({"times": times, "inputs": entries}, indent=1))


# ------------------------------------------------------------------------------------------------ clips (pod, prep)


def _ff(args: list, what: str) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{what} failed (rc {r.returncode}): {r.stderr[-800:]}")
    return r.stdout


def count_frames(path: Path) -> int:
    out = _ff(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
               "stream=nb_read_frames", "-of", "csv=p=0", str(path)], "ffprobe")
    return int(out.strip().split(",")[0])


def video_size(path: Path) -> tuple[int, int, str]:
    out = _ff(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
               "-of", "csv=p=0", str(path)], "ffprobe")
    w, h, r = out.strip().split(",")[:3]
    return int(w), int(h), r


def cut_clip(src: Path, dst: Path, n0: int, n1: int) -> None:
    """Frames n0..n1 (decode order) of src -> dst, 30 fps CFR, libx264 crf 14 (select + setpts; ffmpeg 8 rejects
    -vsync 0, check error 11). setpts=N/(30*TB) alone gives the 30 fps timestamps: an output -r with -fps_mode passthrough
    is refused by ffmpeg 7+ ("contradictory", rc 234; review finding 1)."""
    tmp = dst.with_name(dst.stem + ".part.mp4")
    _ff(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-an",
         "-vf", f"select=between(n\\,{n0}\\,{n1}),setpts=N/(30*TB)", "-fps_mode", "passthrough",
         "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", str(tmp)], f"ffmpeg cut {dst.parent.name}")
    tmp.replace(dst)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-10 else 10.0 * np.log10(255.0 ** 2 / mse)


def read_frames(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Sequential OpenCV decode (a second decoder, independent of ffmpeg's select) of the wanted frame numbers."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    out, i, last = {}, 0, max(wanted)
    while i <= last:
        ok, img = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = img
        i += 1
    cap.release()
    return out


def endpoint_check(src_frames: dict, clip_frames: dict, n0: int, n1: int, L: int) -> dict:
    """Clip frame 0 must match cut frame n0 and clip frame L-1 cut frame n1 better than their neighbours do (and at
    >= 35 dB): catches an off-by-one of the select."""
    res, ok = {}, True
    for j, n in ((0, n0), (L - 1, n1)):
        p = {d: psnr(clip_frames[j], src_frames[n + d]) for d in (-1, 0, 1) if (n + d) in src_frames}
        good = p[0] >= 35.0 and all(p[0] >= v for v in p.values())
        ok &= good
        res[f"clip{j}_vs_n{n}"] = {"psnr_db": {str(d): round(v, 2) for d, v in p.items()}, "ok": good}
    res["ok"] = bool(ok)
    return res


def find_video(take: str, explicit: str | None) -> Path:
    cands = [Path(explicit)] if explicit else []
    cands += [WORKSPACE / "dances" / H_TAKE / f"{H_TAKE}.mp4", WORK / "src_cuts" / H_TAKE / f"{H_TAKE}.mp4"]
    for p in cands:
        if p.exists():
            return p
    raise SystemExit(f"tripod cut not found: {[str(p) for p in cands]}")


def cmd_clips(args) -> None:
    from . import mamma_dyn as md
    import cv2
    ll, h, od = WORK / args.take, WORK / H_TAKE, out_dir(args.take)
    names = [c.strip() for c in args.clips.split(",") if c.strip()]
    src = find_video(args.take, args.video)
    W, H, rate = video_size(src)
    if (W, H) != (1280, 720) or rate not in ("30/1", "30000/1000"):
        raise SystemExit(f"{src}: {W}x{H} @ {rate}, expected the 1280x720 30/1 fps tripod cut")
    th = dict(np.load(h / "people" / "teachers_01.npz"))
    if not np.allclose(th["t"], np.arange(len(th["t"])) / FPS, atol=1e-4):
        raise SystemExit("lucas_luiza_h teachers_01 rows are not the cut frames n / 30")
    sim3 = json.loads((ll / "cameras" / "world_from_sfm.json").read_text())
    c2 = dict(np.load(ll / "cameras" / "02.npz"))
    R2, t2 = md.world_cameras(c2["R"], c2["tvec"], sim3)
    off2 = md.read_sync_offsets(ll).get("02")
    c1 = np.load(ll / "cameras" / "01.npz")
    off1 = md.read_sync_offsets(ll).get("01")
    times = sheet_times(c1["t"], c1["ok"], lambda t: md.offset_label(t, off1),
                        {c: CLIPS[c] for c in names})
    spans = {c: clip_cut_range(*CLIPS[c], args.margin) for c in names}
    wanted = set()
    for n0, n1, _ in spans.values():
        wanted |= {n0 - 1, n0, n0 + 1, n1 - 1, n1, n1 + 1}
    t0 = time.time()
    src_frames = read_frames(src, {n for n in wanted if n >= 0})
    log(f"source decoded to frame {max(wanted)} in {time.time() - t0:.0f} s (OpenCV, for the end-point check)")
    for c in names:
        n0, n1, L = spans[c]
        d = od / "clips" / c
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        cut_clip(src, d / "clip.mp4", n0, n1)
        got = count_frames(d / "clip.mp4")
        if got != L:
            raise SystemExit(f"{c}: clip.mp4 has {got} frames, expected {L}")
        sheet_n = [x["n"] for x in times if x["clip"] == c]
        cf = read_frames(d / "clip.mp4", {0, L - 1} | {n - n0 for n in sheet_n})
        chk = endpoint_check(src_frames, cf, n0, n1, L)
        if not chk["ok"]:
            raise SystemExit(f"{c}: end-point check failed {chk}")
        for n in sheet_n:
            cv2.imwrite(str(d / f"sheet02_{n:06d}.jpg"), cf[n - n0], [cv2.IMWRITE_JPEG_QUALITY, 92])
        sl = slice(n0, n1 + 1)
        boxes = {}
        for slot, role in enumerate(ROLES):
            b = th["box"][sl, slot].astype(np.float64)
            if not np.isfinite(b).all():
                raise SystemExit(f"{c} {role}: teacher box not finite on every frame")
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, W)
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, H)
            boxes[f"{role}_box"] = b.astype(np.float32)
            boxes[f"{role}_present"] = th["row"][sl, slot] >= 0
            boxes[f"{role}_conf"] = th["conf"][sl, slot]
            boxes[f"{role}_kp"] = th["kp"][sl, slot]
            boxes[f"{role}_skel_in_partner"] = th["skel_in_partner"][sl, slot]
        n = np.arange(n0, n1 + 1)
        np.savez(d / "boxes.npz", n=n, **boxes)
        t_ref = (n + N_FROM_K) / FPS
        rows, okdt = frame_map(md.offset_label(t_ref, off2), c2["t"], max_dt=MAX_DT_02)
        if not (okdt.all() and c2["ok"][rows].all()):
            raise SystemExit(f"{c}: camera 02 not localised within {MAX_DT_02 * 1000:.0f} ms on every frame")
        K = c2["K"][rows]
        f = K[:, 0, 0]
        fmed = float(np.median(f))
        if np.abs(f / fmed - 1).max() > F_SPREAD_MAX:
            raise SystemExit(f"{c}: K02 focal varies by {np.abs(f / fmed - 1).max() * 100:.2f} % inside the clip (zoom step)")
        np.savez(d / "cam02.npz", n=n, row=rows, dt=c2["t"][rows] - t_ref, K=K, R=R2[rows], t=t2[rows],
                 dist=c2["dist"][rows])
        k0 = n0 + N_FROM_K
        meta = {"schema": SCHEMA, "clip": c, "note": CLIP_NOTE.get(c, ""), "n0": n0, "n1": n1, "L": L,
                "margin": args.margin, "core_n": [n0 + args.margin, n1 - args.margin], "core_k": list(CLIPS[c]),
                "k0": k0, "i0": k0 - 507, "fps": FPS, "width": W, "height": H, "source": src.name,
                "f02_px": {"median": fmed, "min": float(f.min()), "max": float(f.max())},
                "f_mm": f_mm_of(fmed, W, H), "pp02": [float(np.median(K[:, 0, 2])), float(np.median(K[:, 1, 2]))],
                "cam02_dt_ms_max": float(np.abs(c2["t"][rows] - t_ref).max() * 1000), "frames_check": chk,
                "sheet_n": sheet_n, "raw_detection": {r: float(boxes[f"{r}_present"].mean()) for r in ROLES}}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        log(f"{c}: n {n0}-{n1} ({L} frames) cut and checked in {time.time() - t0:.0f} s; f02 {f.min():.0f}-{f.max():.0f}"
            f" px, f_mm {meta['f_mm']:.2f}; raw detection lead {meta['raw_detection']['lead']:.2f} follow "
            f"{meta['raw_detection']['follow']:.2f}")
    (od / "clips" / "sheet_times.json").write_text(json.dumps(times, indent=1))


# ------------------------------------------------------------------------------------------------ score (pod)


class Ref:
    """Everything the scorer reads once: ll cameras in world metres, sync, teachers, B0 and its frame masks."""

    def __init__(self, take: str = TAKE):
        from . import mamma_dyn as md
        ll = WORK / take
        self.take = take
        sim3 = json.loads((ll / "cameras" / "world_from_sfm.json").read_text())
        self.c1 = dict(np.load(ll / "cameras" / "01.npz"))
        self.c2 = dict(np.load(ll / "cameras" / "02.npz"))
        self.R1, self.t1 = md.world_cameras(self.c1["R"], self.c1["tvec"], sim3)
        self.R2, self.t2 = md.world_cameras(self.c2["R"], self.c2["tvec"], sim3)
        offs = md.read_sync_offsets(ll)
        self.off1, off2 = offs.get("01"), offs.get("02")
        self.T1 = dict(np.load(ll / "people" / "teachers_01.npz"))
        self.T2 = dict(np.load(ll / "people" / "teachers_02.npz"))
        for T, c in ((self.T1, self.c1), (self.T2, self.c2)):
            if len(T["t"]) != len(c["t"]) or not np.allclose(T["t"], c["t"], atol=1e-6):
                raise SystemExit("teacher rows are not the camera rows")
        keep = ("t", "joints", "quality_flag", "repaired", "suspect", "blend_seam", "stance")
        self.B0 = {}
        for r in ROLES:
            with np.load(ll / "smplx_rf2_noslip" / f"{r}.npz", allow_pickle=True) as z:
                self.B0[r] = {k: z[k] for k in keep}
        self.tg = np.asarray(self.B0["lead"]["t"], float)
        if not np.array_equal(self.tg, self.B0["follow"]["t"]) or not np.allclose(np.diff(self.tg), 1 / FPS, atol=1e-4):
            raise SystemExit("B0 grid is not a uniform 30 Hz grid shared by both dancers")
        self.NG = len(self.tg)
        self.k = np.round(self.tg * FPS).astype(int)
        self.kg0 = int(self.k[0])
        self.r1, okdt1 = frame_map(md.offset_label(self.tg, self.off1), self.c1["t"], max_dt=MAX_DT)
        self.ok1 = okdt1 & self.c1["ok"][self.r1]
        self.tau = md.apply_offset(self.c1["t"][self.r1], self.off1)
        self.r2, okdt2 = frame_map(md.offset_label(self.tg, off2), self.c2["t"], max_dt=MAX_DT)
        self.ok2 = okdt2 & self.c2["ok"][self.r2]
        self.K02 = self.c2["K"][self.r2]
        self.R02, self.t02 = self.R2[self.r2], self.t2[self.r2]
        ls1, ls2 = self.T1["lead_slot"][self.r1].astype(int), self.T2["lead_slot"][self.r2].astype(int)
        self.slot1 = {"lead": ls1, "follow": 1 - ls1}
        self.slot2 = {"lead": ls2, "follow": 1 - ls2}
        self.frames = {c: np.arange(k0 - self.kg0, k1 - self.kg0 + 1) for c, (k0, k1) in CLIPS.items()}
        self.clip_id = np.full(self.NG, -1, int)
        for j, fr in enumerate(self.frames.values()):
            self.clip_id[fr] = j
        self.in_clips = self.clip_id >= 0
        # per-frame px_01 / px_02 of the owning chunk (ll2g*/quality/quality_rf2_noslip_frames.npz over owned_k)
        st = json.loads((ll / "stitch.json").read_text())
        px = {(c, r): np.full(self.NG, np.nan) for c in ("01", "02") for r in ROLES}
        for ch in st["chunks"]:
            q = np.load(WORK / f"{take}_{ch['name']}" / "quality" / "quality_rf2_noslip_frames.npz")
            kq = np.round(q["t"] * FPS).astype(int)
            m = (kq >= ch["owned_k"][0]) & (kq <= ch["owned_k"][1]) & (kq - self.kg0 < self.NG)
            for (c, r), arr in px.items():
                arr[kq[m] - self.kg0] = q[f"px_{c}_{r}"][m]
        self.good_mask = {}
        for r in ROLES:
            b = self.B0[r]
            with np.errstate(invalid="ignore"):
                pxm = (px[("01", r)] <= 20) & (px[("02", r)] <= 12)
            tea = np.array([teacher_ok(self.T1, self.r1[i], self.slot1[r][i]) for i in range(self.NG)])
            self.good_mask[r] = ((b["quality_flag"] == 2) & ~b["repaired"] & ~b["suspect"] & (b["blend_seam"] != 2)
                                 & pxm & self.ok1 & tea)
        P0 = {r: to_camera(self.B0[r]["joints"][:, :1], self.R02, self.t02)[:, 0] for r in ROLES}
        self.zB0 = {r: P0[r][:, 2] for r in ROLES}

    # -- the views
    def m1(self, Jw: np.ndarray, role: str) -> tuple[np.ndarray, np.ndarray]:
        """Held-out camera 01: world joints on the grid (NG, >=22, 3), interpolated to each camera-01 row's corrected
        time -> per-frame (% of box height, px) on the clips' core frames."""
        Jt = interp_grid(np.asarray(Jw)[:, :NJ], self.tg, self.tau)
        pct, pxv = np.full(self.NG, np.nan), np.full(self.NG, np.nan)
        T, c = self.T1, self.c1
        for i in np.flatnonzero(self.in_clips & self.ok1):
            k, s = self.r1[i], self.slot1[role][i]
            if not teacher_ok(T, k, s):
                continue
            uv, z = project(Jt[i][SMX], self.R1[k], self.t1[k], c["K"][k], float(c["dist"][k]))
            pct[i], pxv[i] = frame_score(uv, z, T["kp"][k, s], box_h(T["box"][k, s]))
        return pct, pxv

    def m1b_world(self, Jw: np.ndarray, role: str) -> np.ndarray:
        """Camera 02 through the world camera (B0: it has no own K)."""
        out = np.full(self.NG, np.nan)
        T, c = self.T2, self.c2
        for i in np.flatnonzero(self.in_clips & self.ok2):
            k, s = self.r2[i], self.slot2[role][i]
            if not teacher_ok(T, k, s):
                continue
            uv, z = project(np.asarray(Jw)[i][SMX], self.R2[k], self.t2[k], c["K"][k], float(c["dist"][k]))
            out[i] = frame_score(uv, z, T["kp"][k, s], box_h(T["box"][k, s]))[0]
        return out

    def m1b_own(self, Jc: np.ndarray, K: np.ndarray, role: str) -> tuple[np.ndarray, np.ndarray]:
        """Camera 02 in the method's OWN image (its own K, no placement): M1b (%) and M1c (1 = closer to the partner)."""
        m1b, m1c = np.full(self.NG, np.nan), np.full(self.NG, np.nan)
        T = self.T2
        for i in np.flatnonzero(self.in_clips & self.ok2):
            if not np.isfinite(Jc[i, :NJ]).all() or not np.isfinite(K[i]).all():
                continue
            k, s = self.r2[i], self.slot2[role][i]
            uv, z = pinhole(Jc[i][SMX], K[i])
            own = frame_score(uv, z, T["kp"][k, s], box_h(T["box"][k, s])) if teacher_ok(T, k, s) else (np.nan, np.nan)
            m1b[i] = own[0]
            if teacher_ok(T, k, 1 - s):
                other = frame_score(uv, z, T["kp"][k, 1 - s], box_h(T["box"][k, 1 - s]))
                m1c[i] = wrong_dancer(own[1], other[1])
        return m1b, m1c


def load_b1(ref: Ref, role: str) -> tuple[np.ndarray, np.ndarray]:
    """B1: the h world joints (h row = grid i + 3, by time) -> the h tripod camera (lucas_luiza_h/cameras/01.npz)."""
    h = WORK / H_TAKE
    with np.load(h / "smplx_rf2_noslip" / f"{role}.npz", allow_pickle=True) as z:
        J, th = z["joints"][:, :NJ].astype(float), np.asarray(z["t"], float)
    ch = np.load(h / "cameras" / "01.npz")
    ih, ok = h_index(ref.tg, th)
    rh, okc = frame_map(ref.tg - H_CLOCK_S, ch["t"], max_dt=1e-3)
    ok &= okc
    Jc = np.full((ref.NG, NJ, 3), np.nan)
    Jc[ok] = to_camera(J[ih[ok]], ch["R"][rh[ok]], ch["tvec"][rh[ok]])
    K = np.full((ref.NG, 3, 3), np.nan)
    K[ok] = ch["K"][rh[ok]]
    return Jc, K


def load_b2(ref: Ref, role: str) -> tuple[np.ndarray, np.ndarray]:
    """B2: SAM 3D Body (mask prompt) camera-frame joints on cut frame n = grid i + 87, with the K it was given."""
    d = WORK / H_TAKE / "realism" / "sam3db"
    z = np.load(d / "pod_out" / "smplx_mask255.npz")
    meta = json.loads((d / "input" / "meta.json").read_text())
    n = cut_of_grid_index(np.arange(ref.NG), ref.kg0)
    ok = (n >= 0) & (n < len(z["frame_i"]))
    if not np.array_equal(z["frame_i"], np.arange(len(z["frame_i"]))):
        raise SystemExit("B2 frame_i is not the cut frame index")
    Km = np.array([f["K"] for f in meta["frames"]], float)
    Jc = np.full((ref.NG, NJ, 3), np.nan)
    Jc[ok] = z[f"{role}_joints"][n[ok], :NJ]
    K = np.full((ref.NG, 3, 3), np.nan)
    K[ok] = Km[n[ok]]
    return Jc, K


def candidate_status(od: Path, method: str) -> tuple[str, dict]:
    """'ok' | 'not run (...)' | 'not run correctly (...)' from <method>/run.json (or its smoke run)."""
    run = od / method / "run.json"
    if not run.exists():
        smoke = od / f"{method}_smoke" / "run.json"
        if smoke.exists():
            s = json.loads(smoke.read_text())
            bad = {k: v.get("failed") for k, v in s.get("runs", {}).items() if not v.get("ok")}
            return f"not run (smoke {'failed: ' + json.dumps(bad) if bad else 'passed, the run did not finish'})", {}
        return NOT_RUN.get(method, "not run (no outputs)"), {}
    d = json.loads(run.read_text())
    runs = d.get("runs", {})
    want = [f"{c}_{r}" for c in CLIPS for r in ROLES]
    missing = [w for w in want if w not in runs]
    bad = {w: runs[w].get("failed") for w in want if w in runs and not runs[w].get("ok")}
    if missing or bad:
        return f"not run correctly ({'missing ' + ', '.join(missing) + '; ' if missing else ''}{json.dumps(bad)})", d
    return "ok", d


def load_candidate(ref: Ref, od: Path, method: str, role: str) -> tuple[np.ndarray, np.ndarray]:
    """exmr: <clip>_<role>/results.npz (joints_incam, K_fullimg); flowhmr: <clip>_<role>/joints_cam02.npz (J_c, K)."""
    Jc = np.full((ref.NG, NJ, 3), np.nan)
    K = np.full((ref.NG, 3, 3), np.nan)
    for c in CLIPS:
        meta_p = od / "clips" / c / "meta.json"
        if not meta_p.exists():
            continue
        n0 = json.loads(meta_p.read_text())["n0"]
        d = od / method / f"{c}_{role}"
        if method == "exmr" and (d / "results.npz").exists():
            with np.load(d / "results.npz") as z:
                J, KK = z["joints_incam"][:, :NJ].astype(float), z["K_fullimg"].astype(float)
        elif method == "flowhmr" and (d / "joints_cam02.npz").exists():
            with np.load(d / "joints_cam02.npz") as z:
                J, KK = z["J_c"][:, :NJ].astype(float), z["K"].astype(float)
        else:
            continue
        i = grid_index_of_cut(n0 + np.arange(len(J)), ref.kg0)
        m = (i >= 0) & (i < ref.NG)
        Jc[i[m]], K[i[m]] = J[m], KK[m]
    return Jc, K


def m4_clip(ref: Ref, Jw_a: np.ndarray, Jw_b: np.ndarray, role: str, fr: np.ndarray) -> dict:
    """Plausibility on one clip's core frames: 4.5-12 Hz pelvis jitter of the placement-A world pelvis (mm RMS), ankle /
    toe skate on B0's stance frames (cm/s, horizontal, placement B), lowest-joint height spread (p90 - p10, cm)."""
    from .analysis_review import root_jitter
    out = {"jitter_mm": None, "skate_cm_s": None, "floor_spread_cm": None}
    if np.isfinite(Jw_a[fr, 0]).all():
        out["jitter_mm"] = root_jitter({role: {"t": ref.tg[fr], "joints": Jw_a[fr]}})[role]["rms_mm"]
    st = np.asarray(ref.B0[role]["stance"], bool)
    sk = []
    for side, joints in ((0, (7, 10)), (1, (8, 11))):      # stance column 0 = left (ankle 7, foot 10), 1 = right
        both = st[fr[:-1], side] & st[fr[1:], side]
        for j in joints:
            v = np.linalg.norm(Jw_b[fr[1:], j][:, [0, 2]] - Jw_b[fr[:-1], j][:, [0, 2]], axis=1) * FPS * 100
            sk.append(v[both & np.isfinite(v)])
    sk = np.concatenate(sk) if sk else np.array([])
    if len(sk):
        out["skate_cm_s"] = float(np.median(sk))
    y = Jw_b[fr, :NJ, 1]
    low = y[np.isfinite(y).all(1)].min(1) if np.isfinite(y).all(1).any() else np.array([])
    if len(low):
        out["floor_spread_cm"] = float((np.percentile(low, 90) - np.percentile(low, 10)) * 100)
    return out


def cmd_score(args) -> None:
    od = out_dir(args.take)
    sd = od / "score"
    sd.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in ("B0", "B1", "B2"):
        if m not in methods:
            raise SystemExit(f"--methods must include {m}: the pass bar and the self-check need it")
    t0 = time.time()
    ref = Ref(args.take)
    log(f"reference loaded in {time.time() - t0:.0f} s: grid {ref.NG} frames, k {ref.kg0}..{ref.k[-1]}")
    per: dict[str, np.ndarray] = {"tg": ref.tg, "clip_id": ref.clip_id}
    res: dict = {"schema": SCHEMA, "take": args.take, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "clips": {c: {"k": list(CLIPS[c]), "note": CLIP_NOTE[c], "frames": int(len(ref.frames[c]))} for c in CLIPS},
                 "frames": {}, "methods": {}, "couple": {}}
    held, good = {}, {}
    # B0: the reference; its camera-01 score defines the held-out and good-reference frames
    pelA = {}
    for r in ROLES:
        J0 = ref.B0[r]["joints"][:, :NJ].astype(float)
        e0, px0 = ref.m1(J0, r)
        held[r] = np.isfinite(e0) & ref.in_clips
        good[r] = ref.good_mask[r] & held[r]
        per[f"good_mask__{r}"], per[f"good__{r}"], per[f"held__{r}"] = ref.good_mask[r] & ref.in_clips, good[r], held[r]
        per[f"B0__{r}__m1B"], per[f"B0__{r}__m1B_px"] = e0, px0
        per[f"B0__{r}__m1b"] = ref.m1b_world(J0, r)
        pelA[("B0", r)] = J0[:, 0]
        res["frames"][r] = {"good_mask": int((ref.good_mask[r] & ref.in_clips).sum()), "good": int(good[r].sum()),
                            "held": int(held[r].sum()),
                            "clips": {c: {"good_mask": int(ref.good_mask[r][fr].sum()), "good": int(good[r][fr].sum()),
                                          "held": int(held[r][fr].sum())} for c, fr in ref.frames.items()}}
    status = {"B0": "ok (reference: two-view, in-sample in camera 01 where two-view)", "B1": "ok", "B2": "ok"}
    run_info = {}
    for m in methods:
        if m in CANDIDATES:
            status[m], run_info[m] = candidate_status(od, m)
    sheet_pts: dict[str, np.ndarray] = {}
    for m in methods:
        if m == "B0":
            continue
        if m not in ("B1", "B2") + CANDIDATES:
            raise SystemExit(f"unknown method {m}")
        has_any = False
        for r in ROLES:
            if m == "B1":
                Jc, K = load_b1(ref, r)
            elif m == "B2":
                Jc, K = load_b2(ref, r)
            else:
                Jc, K = load_candidate(ref, od, m, r)
            fin = np.isfinite(Jc).all((1, 2)) & np.isfinite(K).all((1, 2))
            if not (fin & ref.in_clips).any():
                continue
            has_any = True
            Jc[~fin], K[~fin] = np.nan, np.nan
            JB = to_world(placement_b(Jc, K, ref.K02, ref.zB0[r]), ref.R02, ref.t02)
            JA = to_world(placement_a(Jc, K, ref.K02), ref.R02, ref.t02)
            per[f"{m}__{r}__m1B"], per[f"{m}__{r}__m1B_px"] = ref.m1(JB, r)
            per[f"{m}__{r}__m1A"] = ref.m1(JA, r)[0]
            if m == "B1":   # the old slide, for the self-check only
                per[f"{m}__{r}__m1B_naive"] = ref.m1(to_world(slide_naive(Jc, ref.zB0[r]), ref.R02, ref.t02), r)[0]
            per[f"{m}__{r}__m1b"], per[f"{m}__{r}__m1c"] = ref.m1b_own(Jc, K, r)
            J0 = ref.B0[r]["joints"][:, :NJ]
            pa, rr = np.full(ref.NG, np.nan), np.full(ref.NG, np.nan)
            for i in np.flatnonzero(fin & ref.in_clips):
                pa[i] = pa_mpjpe(Jc[i], J0[i])
                rr[i] = float(np.linalg.norm((JB[i] - JB[i, 0]) - (J0[i] - J0[i, 0]), axis=1).mean() * 1000)
            per[f"{m}__{r}__pa"], per[f"{m}__{r}__rr"] = pa, rr
            per[f"{m}__{r}__pelA_err"] = np.linalg.norm(JA[:, 0] - J0[:, 0], axis=1)
            pelA[(m, r)] = JA[:, 0]
            res.setdefault("m4", {}).setdefault(m, {})[r] = {c: m4_clip(ref, JA, JB, r, fr) for c, fr in ref.frames.items()}
            sheet_pts[f"_Jc__{m}__{r}"], sheet_pts[f"_K__{m}__{r}"], sheet_pts[f"_JB__{m}__{r}"] = Jc, K, JB
        if not has_any and m in CANDIDATES and status[m] == "ok":
            status[m] = "not run (no outputs)"
        if not has_any and m in ("B1", "B2"):
            raise SystemExit(f"{m}: no frames in the clips")
    # B0 plausibility too (its own numbers as the floor)
    for r in ROLES:
        J0 = ref.B0[r]["joints"][:, :NJ].astype(float)
        res.setdefault("m4", {}).setdefault("B0", {})[r] = {c: m4_clip(ref, J0, J0, r, fr) for c, fr in ref.frames.items()}
    # tables: per clip and pooled
    for m in methods:
        md_: dict = {"status": status.get(m, "ok"), "anchored": m == "flowhmr", "roles": {}}
        for r in ROLES:
            rows = {}
            for c, fr in list(ref.frames.items()) + [("pooled", np.flatnonzero(ref.in_clips))]:
                sel_g = np.zeros(ref.NG, bool)
                sel_g[fr] = True
                g, ho = sel_g & good[r], sel_g & held[r]
                row = {}
                for key in ("m1B", "m1A", "m1b", "m1c", "pa", "rr", "pelA_err", "m1B_naive"):
                    a = per.get(f"{m}__{r}__{key}")
                    if a is None:
                        continue
                    if key == "m1B":
                        row["m1B_good"], row["m1B_held"] = summary(a[g]), summary(a[ho])
                    elif key in ("m1b",):
                        row["m1b"] = summary(a[sel_g])
                    elif key == "m1c":
                        v = a[sel_g & np.isfinite(a)]
                        row["m1c"] = {"n": int(len(v)), "share": float(v.mean()) if len(v) else None}
                    else:
                        row[key] = summary(a[g])
                rows[c] = row
            md_["roles"][r] = rows
        res["methods"][m] = md_
    # M3 couple vector (placement A): frames where both dancers are good-reference and both placed
    for m in methods:
        if m == "B0" or (m, "lead") not in pelA or (m, "follow") not in pelA:
            continue
        cv = np.linalg.norm((pelA[(m, "lead")] - pelA[(m, "follow")])
                            - (pelA[("B0", "lead")] - pelA[("B0", "follow")]), axis=1)
        both = good["lead"] & good["follow"]
        per[f"{m}__couple_A"] = cv
        res["couple"][m] = {c: summary(cv[both & (ref.clip_id == j)]) for j, c in enumerate(CLIPS)}
        res["couple"][m]["pooled"] = summary(cv[both])
    # the self-check
    meas = {"n_good_mask": tuple(res["frames"][r]["good_mask"] for r in ROLES),
            "n_good": tuple(res["frames"][r]["good"] for r in ROLES),
            "n_held": tuple(res["frames"][r]["held"] for r in ROLES)}

    def med(key, mask_of):
        out = []
        for r in ROLES:
            a = per.get(key.format(r=r))
            v = a[mask_of(r)] if a is not None else np.array([])
            v = v[np.isfinite(v)]
            out.append(float(np.median(v)) if len(v) else None)
        return tuple(out)
    meas["b0_m1_good"] = med("B0__{r}__m1B", lambda r: good[r])
    meas["b1_m1_naive_good"] = med("B1__{r}__m1B_naive", lambda r: good[r])
    meas["b1_pa_good"] = med("B1__{r}__pa", lambda r: good[r])
    meas["b2_m1b_e1"] = med("B2__{r}__m1b", lambda r: ref.clip_id == 0)
    res["self_check"] = self_check(meas)
    res["run_info"] = run_info
    log("self-check " + ("PASSED" if res["self_check"]["ok"] else "FAILED") + ": " +
        "; ".join(f"{x['key']} {x['got']}" for x in res["self_check"]["rows"]))
    # sheet points: the 12 fixed times, every method in camera 01 (placement B) and in its own camera-02 image
    times_p = od / "clips" / "sheet_times.json"
    from . import mamma_dyn as md
    times = (json.loads(times_p.read_text()) if times_p.exists() else
             sheet_times(ref.c1["t"], ref.c1["ok"], lambda t: md.offset_label(t, ref.off1)))
    pts: dict[str, np.ndarray] = {}
    Jt_cache = {}
    for e in times:
        i = int(grid_index_of_cut(e["n"], ref.kg0))
        e["i"], e["row02"] = i, int(ref.r2[i])
        if int(ref.r1[i]) != e["row01"]:
            raise SystemExit(f"sheet time {e}: camera-01 row {ref.r1[i]} differs from the shipped frame {e['row01']}")
    for m in methods:
        for r in ROLES:
            if m == "B0":
                Jw, Jc, K = ref.B0[r]["joints"][:, :NJ].astype(float), None, None
            elif f"_JB__{m}__{r}" in sheet_pts:
                Jw, Jc, K = sheet_pts[f"_JB__{m}__{r}"], sheet_pts[f"_Jc__{m}__{r}"], sheet_pts[f"_K__{m}__{r}"]
            else:
                continue
            Jt_cache[(m, r)] = interp_grid(Jw, ref.tg, ref.tau)
            uv1, uv2 = np.full((len(times), NJ, 2), np.nan), np.full((len(times), NJ, 2), np.nan)
            for j, e in enumerate(times):
                i, k1, k2 = e["i"], e["row01"], e["row02"]
                uv1[j] = project(Jt_cache[(m, r)][i], ref.R1[k1], ref.t1[k1], ref.c1["K"][k1], float(ref.c1["dist"][k1]))[0]
                if m == "B0":
                    uv2[j] = project(Jw[i], ref.R2[k2], ref.t2[k2], ref.c2["K"][k2], float(ref.c2["dist"][k2]))[0]
                elif np.isfinite(Jc[i]).all():
                    uv2[j] = pinhole(Jc[i], K[i])[0]
            pts[f"uv01__{m}__{r}"], pts[f"uv02__{m}__{r}"] = uv1, uv2
    pts["kp01"] = np.stack([np.stack([ref.T1["kp"][e["row01"], ref.slot1[r][e["i"]]] for r in ROLES]) for e in times])
    pts["kp02"] = np.stack([np.stack([ref.T2["kp"][e["row02"], ref.slot2[r][e["i"]]] for r in ROLES]) for e in times])
    res["sheet_times"] = times
    np.savez_compressed(sd / "sheet_points.npz", **pts)
    np.savez_compressed(sd / "per_frame.npz", **per)
    (sd / "score.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"score: {sd / 'score.json'} ({time.time() - t0:.0f} s); methods " +
        ", ".join(f"{m}: {res['methods'][m]['status']}" for m in methods))


# ------------------------------------------------------------------------------------------------ sheet (pod)


def common_verdicts(per: dict, status: dict, methods: list[str]) -> dict:
    """The owner's pass bar per candidate and dancer, on the frames B1, B2 and the candidate all scored (good-reference
    frames, M1-B and PA both finite) -> {cand: {role: {...}}}. "b0_m1B": B0's M1-B on the same frames, the reference
    shown next to the verdict (not part of it; B0 is finite on every good-reference frame, and its PA is 0 by
    definition)."""
    out = {}
    for m in methods:
        if m not in CANDIDATES:
            continue
        out[m] = {}
        for r in ROLES:
            if status.get(m) != "ok":
                out[m][r] = {"improves": None, "text": "no verdict: " + str(status.get(m))}
                continue
            mask = per[f"good__{r}"].astype(bool).copy()
            for x in ("B1", "B2", m):
                for key in ("m1B", "pa"):
                    a = per.get(f"{x}__{r}__{key}")
                    mask &= np.isfinite(a) if a is not None else False
            n = int(mask.sum())
            vals = {key: {("cand" if x == m else x): (float(np.median(per[f"{x}__{r}__{key}"][mask])) if n else None)
                          for x in ("B1", "B2", m)} for key in ("m1B", "pa")}
            v = verdict(vals["m1B"], vals["pa"]) if n else {"improves": None, "text": "no verdict (no common frames)"}
            b0 = per.get(f"B0__{r}__m1B")
            b0v = b0[mask][np.isfinite(b0[mask])] if b0 is not None and n else np.array([])
            out[m][r] = {**v, "n_common": n, "m1B": vals["m1B"], "pa": vals["pa"],
                         "b0_m1B": float(np.median(b0v)) if len(b0v) else None}
    return out


def _fmt(x, nd=2, dash="–"):
    return dash if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def draw_contact(times: list[dict], images: list, pts: dict, cam: str, methods: list[str], path: Path) -> None:
    """3 times x 4 clips; each method's skeleton (22 joints, jl_fitcheck.BONES) and the teacher kp as dots."""
    import cv2
    cells = []
    for j, (e, img) in enumerate(zip(times, images)):
        if img is None:
            img = np.full((1280, 720, 3) if cam == "01" else (720, 1280, 3), 60, np.uint8)
            cv2.putText(img, "frame missing", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        img = img.copy()
        for m in methods:
            for r in ROLES:
                uv = pts.get(f"uv{cam}__{m}__{r}")
                if uv is None or not np.isfinite(uv[j]).any():
                    continue
                for a, b in BONES:
                    p, q = uv[j, a], uv[j, b]
                    if np.isfinite(p).all() and np.isfinite(q).all() and np.abs(np.r_[p, q]).max() < 1e5:
                        cv2.line(img, tuple(int(x) for x in p), tuple(int(x) for x in q), COLOR[m], 3, cv2.LINE_AA)
        for si, col in enumerate(((0, 0, 255), (255, 0, 0))):      # teacher kp: lead red, follow blue
            for x, y, c in pts[f"kp{cam}"][j, si]:
                if c >= CONF:
                    cv2.circle(img, (int(x), int(y)), 6, col, -1, cv2.LINE_AA)
        cv2.putText(img, f"{e['clip']} {int(e['frac'] * 100)}% n{e['n']} t{e['t']:.2f}", (16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, f"{e['clip']} {int(e['frac'] * 100)}% n{e['n']} t{e['t']:.2f}", (16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
        cells.append(cv2.resize(img, None, fx=0.375, fy=0.375, interpolation=cv2.INTER_AREA))
    rows = [np.concatenate(cells[i:i + 3], 1) for i in range(0, len(cells), 3)]
    sheet = np.concatenate(rows, 0)
    leg = np.zeros((64, sheet.shape[1], 3), np.uint8)
    x = 8
    for m in methods:
        cv2.putText(leg, LABEL[m], (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[m], 1, cv2.LINE_AA)
        x += 14 + 10 * len(LABEL[m])
    cv2.putText(leg, "teacher kp: lead red, follow blue; " + ("placement B" if cam == "01" else "own image (B0: world camera)"),
                (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.concatenate([sheet, leg], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])


def cmd_sheet(args) -> None:
    import cv2
    od = out_dir(args.take)
    sd = od / "score"
    res = json.loads((sd / "score.json").read_text())
    with np.load(sd / "per_frame.npz") as z:
        per = {k: z[k] for k in z.files}
    methods = list(res["methods"])
    status = {m: res["methods"][m]["status"] for m in methods}
    valid = bool(res["self_check"]["ok"])
    verd = common_verdicts(per, status, methods) if valid else {}
    M = res["methods"]

    def cell(m, r, c, key, sub="median", nd=2):
        row = M.get(m, {}).get("roles", {}).get(r, {}).get(c, {})
        v = row.get(key)
        if v is None:
            return "–"
        if key == "m1c":
            return _fmt(None if v["share"] is None else 100 * v["share"], 1)
        return _fmt(v.get(sub), nd)

    cands = [m for m in methods if m in CANDIDATES]
    lines = [f"# {res['take']}: does a newer monocular HMR method improve accuracy? ({time.strftime('%Y-%m-%d')})", ""]
    if not valid:
        lines += ["**INVALID: scorer.** The self-check did not reproduce the basis numbers (table below): no verdict.", ""]
    else:
        lines += ["**Pass bar (owner, 2026-10-27):** a method improves accuracy for a dancer when its pooled M1-B "
                  "(held-out camera 01, placement B) and its pooled PA-MPJPE are both no worse than B1 (our current "
                  "single-camera fit) and no worse than B2 (the SAM 3D Body prior), on the frames all three scored. "
                  "B0 (the two-view fit) is the reference: its M1-B on the same frames is on each line.", ""]
        for m in cands:
            for r in ROLES:
                v = verd[m][r]
                if v.get("improves") is None:
                    lines.append(f"- **{LABEL[m]}, {r}: {v['text']}**")
                    continue
                a, b = v["m1B"], v["pa"]
                lines.append(f"- **{LABEL[m]}, {r}: {v['text']}** — M1-B {_fmt(a['cand'])} % vs B1 {_fmt(a['B1'])} / "
                             f"B2 {_fmt(a['B2'])} (B0 reference {_fmt(v.get('b0_m1B'))}); PA {_fmt(b['cand'], 1)} mm "
                             f"vs B1 {_fmt(b['B1'], 1)} / B2 {_fmt(b['B2'], 1)} ({v['n_common']} common frames)")
        lines.append("")
    lines += ["Method status: " + "; ".join(f"{LABEL[m]}{' (anchored)' if M[m].get('anchored') else ''}: {status[m]}"
                                         for m in methods), ""]
    if status.get("flowhmr") == "ok":   # check error 13: reported, not corrected
        lines += ["FlowHMR outputs SMPL-H (the others SMPL-X): their joint regressors differ slightly, a small systematic "
                  "offset on its PA that was not measured. Its world placement is anchored (yaw + 2-DoF horizontal "
                  "anchor to the tripod teacher's hip midpoint); the other methods need no anchor.", ""]
    hdr = (["Clip", "Dancer", "good / held"] + [f"M1-B {LABEL[m].split()[0]}" for m in methods]
           + [f"PA {LABEL[m].split()[0]}" for m in methods if m != "B0"]
           + [f"M1b {LABEL[m].split()[0]}" for m in methods] + [f"M1c {LABEL[m].split()[0]}" for m in methods if m != "B0"])
    lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for c in list(CLIPS) + ["pooled"]:
        for r in ROLES:
            fr = res["frames"][r] if c == "pooled" else res["frames"][r]["clips"][c]
            row = [c, r, f"{fr['good']} / {fr['held']}"]
            row += [cell(m, r, c, "m1B_good") for m in methods]
            row += [cell(m, r, c, "pa", nd=0) for m in methods if m != "B0"]
            row += [cell(m, r, c, "m1b") for m in methods]
            row += [cell(m, r, c, "m1c") for m in methods if m != "B0"]
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "M1-B: median camera-01 error in % of the teacher's box height on good-reference frames (placement B, "
              "the pelvis re-ray). PA: per-frame median PA-MPJPE (mm, 22 joints) vs B0 on the same frames. M1b: the "
              "method's joints in its own camera-02 image (its own K) vs teachers_02, all clip frames. M1c: % of camera-02 "
              "frames closer to the partner. B0 is in-sample in camera 01 where it is two-view: the reference floor "
              "(about 3 % / 30 mm), below which methods are not ranked.", ""]
    lines += ["## All held-out frames, placement A, 3D and plausibility (reported, not part of the pass bar)", "",
              "| Method | Dancer | M1-B held-out (median / p90 / >10 %) | M1-A | root-rel MPJPE mm | pelvis A m | "
              "couple vector A m (median / p90) |", "|---|---|---|---|---|---|---|"]
    for m in methods:
        for r in ROLES:
            row = M[m]["roles"][r].get("pooled", {})
            hv = row.get("m1B_held") or {}
            cp = res["couple"].get(m, {}).get("pooled", {}) if r == "lead" else None
            lines.append(f"| {LABEL[m]} | {r} | {_fmt(hv.get('median'))} / {_fmt(hv.get('p90'), 1)} / "
                         f"{_fmt(None if hv.get('gross') is None else 100 * hv['gross'], 0)} % | "
                         f"{_fmt((row.get('m1A') or {}).get('median'))} | {_fmt((row.get('rr') or {}).get('median'), 0)} | "
                         f"{_fmt((row.get('pelA_err') or {}).get('median'))} | "
                         + (f"{_fmt(cp.get('median'))} / {_fmt(cp.get('p90'))} |" if cp is not None else "(the couple: lead row) |"))
    lines += ["", "| Method | Dancer | " + " | ".join(f"{c} jitter mm / skate cm/s / floor spread cm" for c in CLIPS) + " |",
              "|---|---|" + "---|" * len(CLIPS)]
    for m in methods:
        for r in ROLES:
            m4 = res.get("m4", {}).get(m, {}).get(r)
            if not m4:
                continue
            lines.append(f"| {LABEL[m]} | {r} | " + " | ".join(
                f"{_fmt(m4[c]['jitter_mm'], 1)} / {_fmt(m4[c]['skate_cm_s'], 1)} / {_fmt(m4[c]['floor_spread_cm'], 1)}"
                for c in CLIPS) + " |")
    lines += ["", "## Scorer self-check" + ("" if valid else " (FAILED: INVALID: scorer)"), "",
              "| Check | Target lead / follow | Tolerance | Got lead / follow | OK |", "|---|---|---|---|---|"]
    for x in res["self_check"]["rows"]:
        lines.append(f"| {x['name']} | {x['target'][0]} / {x['target'][1]} | {x['tol']} | "
                     f"{_fmt(x['got'][0])} / {_fmt(x['got'][1])} | {'yes' if x['ok'] else 'NO'} |")
    lines += ["", "Contact sheets: contact_cam01.jpg (held-out camera 01, placement B) and contact_cam02.jpg (each "
              "method in its own camera-02 image; B0 through the world camera). " +
              ", ".join(f"{LABEL[m]} {COLOR_NAME[m]}" for m in methods) +
              "; teacher kp: lead red, follow blue.", ""]
    (sd / "sheet.md").write_text("\n".join(lines))
    (sd / "sheet.json").write_text(json.dumps({"schema": SCHEMA, "valid": valid, "status": status, "verdicts": verd,
                                               "self_check": res["self_check"]}, indent=1))
    # contact sheets
    times = res["sheet_times"]
    with np.load(sd / "sheet_points.npz") as z:
        pts = {k: z[k] for k in z.files}
    drawn = [m for m in methods if any(f"uv01__{m}__{r}" in pts for r in ROLES)]
    read = lambda p: cv2.imread(str(p)) if p.exists() else None
    im1 = [read(od / "sheet_frames" / f"{e['row01']:06d}.jpg") for e in times]
    im2 = [read(od / "clips" / e["clip"] / f"sheet02_{e['n']:06d}.jpg") for e in times]
    draw_contact(times, im1, pts, "01", drawn, sd / "contact_cam01.jpg")
    draw_contact(times, im2, pts, "02", drawn, sd / "contact_cam02.jpg")
    log(f"sheet: {sd / 'sheet.md'} ({'valid' if valid else 'INVALID: scorer'}); " +
        "; ".join(f"{m} {r}: {v[r]['text']}" for m, v in verd.items() for r in ROLES))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.hmr_test", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sheet-frames", help="local, read-only: the 12 camera-01 jpgs of the contact sheet (job inputs)")
    p.add_argument("--take", default=TAKE)
    p = sub.add_parser("clips", help="pod: cut the clips and write boxes / camera 02 / meta")
    p.add_argument("--take", default=TAKE)
    p.add_argument("--clips", default=",".join(CLIPS))
    p.add_argument("--margin", type=int, default=30)
    p.add_argument("--video", help="the tripod cut (default: dances/lucas_luiza_h/ or work/src_cuts/lucas_luiza_h/)")
    p = sub.add_parser("score", help="pod: every method's metrics, the self-check, the sheet points")
    p.add_argument("--take", default=TAKE)
    p.add_argument("--methods", default=",".join(METHODS))
    p = sub.add_parser("sheet", help="pod: sheet.md / sheet.json / contact sheets from score/")
    p.add_argument("--take", default=TAKE)
    args = ap.parse_args(argv)
    if args.take != TAKE:
        raise SystemExit(f"only {TAKE} has a reference (luiza_paulo: later, with per-frame K and R for its camera 02)")
    {"sheet-frames": cmd_sheet_frames, "clips": cmd_clips, "score": cmd_score, "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    main()
