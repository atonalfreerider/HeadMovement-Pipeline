"""Hold-out evaluation of the mono4d route on a take captured with many phones. Pure numpy.

Protocol (docs/MONO4D_NOVEL_VIEWS.md, section 6):
  1. reference R: the take's multi-camera fit (larissa_kadu_38/smplx: 8 phones), the best 3D we have
  2. input: ONE phone of that take over a window; the provider sees only that video
  3. the provider renders the ring (MAMMA's input) and, for comparison, every other phone's real pose (views h<cam>)
  4. fit A: MAMMA on the virtual take {input phone + ring}; fit B (baseline): the one-camera fit of the same phone and
     window (dancecap.jl_mono + MAMMA_MONO)
  5. A and B against R: compare_fits(); the generated h-views against the real phones: compare_views()
The route earns more work only if A beats B on the same window (verdict()).

    compare_fits(ref_dir, test_dir)      per role: MPJPE, PA-MPJPE, pelvis (horizontal / vertical), ankle height (mm)
    compare_views(real_work, cam, virtual_work, vcam)   per frame: person count, 2D PCK, mask IoU of the two dancers
    verdict(a, b)                        does fit A beat fit B? (thresholds in VERDICT)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BODY = tuple(range(22))  # SMPL-X body joints 0-21 (pelvis ... wrists); hands and face are judged elsewhere
PELVIS, L_ANKLE, R_ANKLE = 0, 7, 8
ROLES = ("lead", "follow")
VERDICT = {"pa_mpjpe_gain": 0.10, "pelvis_gain": 0.10}  # A must be >= 10 % better than B on both medians


def summary(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "median": None, "mean": None, "p95": None}
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(x.mean()), "p95": float(np.percentile(x, 95))}


def match_times(t_ref: np.ndarray, t_test: np.ndarray, tol: float = 1 / 60) -> tuple[np.ndarray, np.ndarray]:
    """Indices (i_ref, i_test) of the times both series share (nearest neighbour within tol seconds)."""
    t_ref, t_test = np.asarray(t_ref, float), np.asarray(t_test, float)
    j = np.searchsorted(t_test, t_ref)
    j0, j1 = np.clip(j - 1, 0, len(t_test) - 1), np.clip(j, 0, len(t_test) - 1)
    pick = np.where(np.abs(t_test[j0] - t_ref) <= np.abs(t_test[j1] - t_ref), j0, j1)
    ok = np.abs(t_test[pick] - t_ref) <= tol
    return np.nonzero(ok)[0], pick[ok]


def mpjpe(a: np.ndarray, b: np.ndarray, joints=BODY) -> np.ndarray:
    """(T,) mean joint distance over `joints`, metres."""
    j = list(joints)
    return np.linalg.norm(a[:, j] - b[:, j], axis=-1).mean(-1)


def procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """X (J,3) aligned to Y by the best similarity (scale, rotation, translation)."""
    mx, my = X.mean(0), Y.mean(0)
    x, y = X - mx, Y - my
    U, S, Vt = np.linalg.svd(x.T @ y)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max((x ** 2).sum(), 1e-12))
    return s * x @ R + my


def pa_mpjpe(a: np.ndarray, b: np.ndarray, joints=BODY) -> np.ndarray:
    """(T,) MPJPE after a per-frame similarity alignment of a onto b: the pose error alone."""
    j = list(joints)
    return np.array([np.linalg.norm(procrustes(a[i, j], b[i, j]) - b[i, j], axis=-1).mean() for i in range(len(a))])


def load_fit(smplx_dir: Path, role: str) -> dict:
    z = np.load(Path(smplx_dir) / f"{role}.npz", allow_pickle=True)
    return {"t": np.asarray(z["t"], float), "joints": np.asarray(z["joints"], float),
            "quality": np.asarray(z["quality"], float) if "quality" in z.files else None}


def compare_fits(ref_dir: Path, test_dir: Path, window: tuple[float, float] | None = None) -> dict:
    """A test fit against the reference fit on their shared grid times -> per role errors in millimetres."""
    out = {}
    for role in ROLES:
        r, s = load_fit(ref_dir, role), load_fit(test_dir, role)
        ir, it = match_times(r["t"], s["t"])
        if window is not None:
            keep = (r["t"][ir] >= window[0]) & (r["t"][ir] <= window[1])
            ir, it = ir[keep], it[keep]
        if not len(ir):
            out[role] = {"frames": 0}
            continue
        A, B = s["joints"][it], r["joints"][ir]
        d_pel = A[:, PELVIS] - B[:, PELVIS]
        ank = np.abs(A[:, [L_ANKLE, R_ANKLE], 1] - B[:, [L_ANKLE, R_ANKLE], 1]).mean(1)
        out[role] = {"frames": int(len(ir)), "t": [float(r["t"][ir[0]]), float(r["t"][ir[-1]])],
                     "mpjpe_mm": summary(1e3 * mpjpe(A, B)), "pa_mpjpe_mm": summary(1e3 * pa_mpjpe(A, B)),
                     "pelvis_mm": summary(1e3 * np.linalg.norm(d_pel, axis=1)),
                     "pelvis_horizontal_mm": summary(1e3 * np.linalg.norm(d_pel[:, [0, 2]], axis=1)),
                     "pelvis_vertical_mm": summary(1e3 * np.abs(d_pel[:, 1])),
                     "ankle_height_mm": summary(1e3 * ank)}
    return out


def verdict(a: dict, b: dict, th: dict | None = None) -> dict:
    """Does fit A (mono4d) beat fit B (one-camera baseline)? Both are compare_fits() outputs on the same window."""
    th = {**VERDICT, **(th or {})}
    why, rows = [], {}
    for role in ROLES:
        ra, rb = a.get(role) or {}, b.get(role) or {}
        if not ra.get("frames") or not rb.get("frames"):
            why.append(f"{role}: no shared frames")
            continue
        pa_a, pa_b = ra["pa_mpjpe_mm"]["median"], rb["pa_mpjpe_mm"]["median"]
        pe_a, pe_b = ra["pelvis_mm"]["median"], rb["pelvis_mm"]["median"]
        rows[role] = {"pa_mpjpe_mm": [pa_a, pa_b], "pelvis_mm": [pe_a, pe_b],
                      "pa_gain": 1 - pa_a / pa_b, "pelvis_gain": 1 - pe_a / pe_b}
        if rows[role]["pa_gain"] < th["pa_mpjpe_gain"]:
            why.append(f"{role}: PA-MPJPE {pa_a:.0f} mm vs baseline {pa_b:.0f} mm (gain "
                       f"{100 * rows[role]['pa_gain']:.0f} % < {100 * th['pa_mpjpe_gain']:.0f} %)")
        if rows[role]["pelvis_gain"] < th["pelvis_gain"]:
            why.append(f"{role}: pelvis {pe_a:.0f} mm vs baseline {pe_b:.0f} mm (gain "
                       f"{100 * rows[role]['pelvis_gain']:.0f} % < {100 * th['pelvis_gain']:.0f} %)")
    return {"route_beats_baseline": not why, "why": why, "rows": rows, "thresholds": th}


# ---------------------------------------------------------------------------------------------------- 2D views


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def pck(pred: np.ndarray, ref: np.ndarray, conf_ref: np.ndarray, scale: float, alpha: float = 0.1,
        min_conf: float = 0.35) -> tuple[float, float]:
    """(fraction of the reference's confident keypoints within alpha * scale px, their mean error / scale)."""
    ok = conf_ref >= min_conf
    if not ok.any():
        return float("nan"), float("nan")
    d = np.linalg.norm(pred[ok] - ref[ok], axis=-1) / scale
    return float((d <= alpha).mean()), float(d.mean())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 127, b > 127
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def _people(work: Path, cam: str) -> dict:
    z = np.load(Path(work) / "people" / f"{cam}.npz")
    return {k: z[k] for k in z.files}


def _times(work: Path, cam: str) -> np.ndarray:
    return np.array([f["t"] for f in json.loads((Path(work) / "frames" / cam / "frames.json").read_text())["frames"]])


def _union_mask(work: Path, cam: str, i: int):
    from PIL import Image
    d = Path(work) / "dancers" / cam / "masks"
    m = None
    for pid in (1, 2):
        p = d / f"mask_{i:04d}_{pid:02d}.png"
        if p.exists():
            a = np.asarray(Image.open(p).convert("L"))
            m = a if m is None else np.maximum(m, a)
    return m


def compare_views(real_work: Path, cam: str, virtual_work: Path, vcam: str, min_score: float = 0.5) -> dict:
    """A generated view rendered at a real phone's pose vs that phone, on shared times.

    Uses the detect stage's people/<cam>.npz of both (boxes, COCO-17 keypoints) and, when both exist, the dancers
    stage's masks. Per frame: people counted in each (a duplicated dancer shows as extra people), the two largest
    people of the generated view matched to the real one's by box IoU, PCK@0.1 of the box diagonal, mask IoU.
    """
    pr, pv = _people(real_work, cam), _people(virtual_work, vcam)
    ir, iv = match_times(_times(real_work, cam), _times(virtual_work, vcam))
    rows = {"count_real": [], "count_gen": [], "pck": [], "kp_err": [], "mask_iou": []}
    for a, b in zip(ir, iv):
        vr = pr["valid"][a] & (pr["scores"][a] >= min_score)
        vg = pv["valid"][b] & (pv["scores"][b] >= min_score)
        rows["count_real"].append(int(vr.sum()))
        rows["count_gen"].append(int(vg.sum()))
        R_ = np.nonzero(vr)[0]
        G_ = np.nonzero(vg)[0]
        area = lambda bx: (bx[2] - bx[0]) * (bx[3] - bx[1])
        R_ = sorted(R_, key=lambda k: -area(pr["boxes"][a, k]))[:2]
        G_ = list(G_)
        for k in R_:
            if not G_ or "kpts" not in pr or "kpts" not in pv:
                continue
            g = max(G_, key=lambda m: box_iou(pr["boxes"][a, k], pv["boxes"][b, m]))
            if box_iou(pr["boxes"][a, k], pv["boxes"][b, g]) < 0.1:
                rows["pck"].append(0.0)
                continue
            G_.remove(g)
            bx = pr["boxes"][a, k]
            p, e = pck(pv["kpts"][b, g, :, :2], pr["kpts"][a, k, :, :2], pr["kpts"][a, k, :, 2],
                       float(np.hypot(bx[2] - bx[0], bx[3] - bx[1])))
            rows["pck"].append(p)
            rows["kp_err"].append(e)
        mr, mg = _union_mask(real_work, cam, a), _union_mask(virtual_work, vcam, b)
        if mr is not None and mg is not None and mr.shape == mg.shape:
            rows["mask_iou"].append(mask_iou(mr, mg))
    cr, cg = np.array(rows["count_real"]), np.array(rows["count_gen"])
    return {"cam": cam, "vcam": vcam, "frames": int(len(ir)),
            "extra_people_frac": float((cg > cr).mean()) if len(cr) else None,
            "missing_people_frac": float((cg < np.minimum(cr, 2)).mean()) if len(cr) else None,
            "pck_0.1": summary(rows["pck"]), "kp_err_over_diag": summary(rows["kp_err"]),
            "mask_iou": summary(rows["mask_iou"])}
