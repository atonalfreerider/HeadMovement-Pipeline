"""One scoring harness for every one-camera pose candidate (2026-10-19): fit B (MAMMA_MONO), Viggle JST-2, SAM 3D Body,
Multi-HMR 2, CloseApp, Kimodo-smoothed fits, ... all scored the same way against the multi-phone reference fit.

Contract, work/<take>/mono4d/cand/<name>/:
    lead.npz, follow.npz   t (T,) reference seconds = the input phone's frames.json t + its sync delta (frame_to_ref;
                           the reference fit's and load_tracks' clock - raw frame times are ~1 frame off on a phone
                           whose delta is ~-30 ms, and score() flags them); joints (T, J, 3) with J >= 22 in SMPL / SMPL-X body order
                           (0 pelvis ... 21 right wrist; NaN where a joint is not predicted); frame = "world" (metres,
                           CONTRACTS world frame) or "camera" (metres, OpenCV axes of the input phone)
    meta.json              {"name", "method", "input_cam", "frame", "source", "notes", "cost_usd", ...}
Helpers: to_world (camera -> world with the phone's per-frame track), assign_roles (which predicted track is the lead
and which the follow: their 2D projections against dancers/roles.json boxes), save.

score() per role on the shared grid times inside the window:
    pa22_mm     PA-MPJPE on the 22 body joints (pose only: per-frame similarity alignment)
    pa14_mm     the same on the 14 joints every rig defines (Viggle's Mixamo set): comparable across rigs
    rr_mm       root-relative MPJPE (pelvis on pelvis, no rotation or scale): pose AND global body orientation; world
                or camera candidates only (a camera-frame candidate is moved to the world with the phone's track)
    pelvis_mm   metric pelvis position error in the world frame (only for world-frame candidates)
    limb_deg    limb-direction error (upper arms, forearms, thighs, shins) after the per-frame rotation
leaderboard() collects every candidate of a take + the multi-view floor (the 7-phone held-out chunk) into one table.

    python -m dancecap.mono4d cand-score <take> --name <name> [--t0 58 --t1 62]
    python -m dancecap.mono4d leaderboard <take> [--t0 58 --t1 62]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT
from .holdout import load_fit, match_times, procrustes, summary
from .viggle import LIMBS, MIXAMO_TO_SMPLX

J14 = sorted(MIXAMO_TO_SMPLX.values())
J22 = list(range(22))
ROLES = ("lead", "follow")


def cand_dir(take: str, name: str) -> Path:
    return PACKAGE_ROOT / "work" / take / "mono4d" / "cand" / name


def frame_to_ref(take: str, cam: str, t: np.ndarray) -> np.ndarray:
    """frames.json times of one phone -> reference seconds (t + that phone's sync/offsets.json delta), the clock of the
    multi-phone fit and of mamma_dyn.load_tracks."""
    from .. import mamma_dyn as md
    return md.apply_offset(np.asarray(t, float), md.read_sync_offsets(PACKAGE_ROOT / "work" / take).get(cam))


def clock_check(take: str, cam: str, t: np.ndarray, tol: float = 1e-3, ref_grid: np.ndarray | None = None) -> dict:
    """Which clock a candidate's t is on: the fraction of its times within tol of the phone's raw frames.json times
    and of the same times on the reference clock. 'raw' (a missed frame_to_ref) when the raw match clearly wins.
    ref_grid (the reference fit's t): a candidate on that grid is in reference seconds by construction (fit B's
    convention); if the phone's raw frames also happen to lie on it, the clock cannot be told ('ambiguous')."""
    fj = PACKAGE_ROOT / "work" / take / "frames" / cam / "frames.json"
    if not fj.exists() or not len(t):
        return {"clock": "unknown"}
    raw = np.sort([f["t"] for f in json.loads(fj.read_text())["frames"]])
    if len(raw) < 2:
        return {"clock": "unknown"}
    try:
        ref = np.sort(frame_to_ref(take, cam, raw))
    except SystemExit as e:  # a stale sync/offsets.json: report it, never abort the scoring of a diagnostic
        return {"clock": "unknown", "error": str(e)}
    t = np.asarray(t, float)

    def near(grid):
        i = np.clip(np.searchsorted(grid, t), 1, len(grid) - 1)
        return float(np.mean(np.minimum(np.abs(grid[i] - t), np.abs(grid[i - 1] - t)) <= tol))

    m_raw, m_ref = near(raw), near(ref)
    m_grid = near(np.sort(np.asarray(ref_grid, float))) if ref_grid is not None and len(ref_grid) > 1 else 0.0
    shift = float(np.median(ref - raw))
    per = float(np.median(np.diff(raw)))
    if abs(shift) > 2 * tol and abs(shift - per * round(shift / per)) <= tol:
        clock = "ambiguous"  # the delta is a whole number of frames: raw and reference times coincide
    elif m_grid > 0.5:
        clock = "ambiguous" if m_raw > 0.5 else "ref"
    else:
        clock = "raw" if (abs(shift) > 2 * tol and m_raw > 0.5 and m_raw > m_ref) else (
            "ref" if m_ref > 0.5 else "other")
    return {"clock": clock, "raw_match": round(m_raw, 3), "ref_match": round(m_ref, 3), "grid_match": round(m_grid, 3),
            "sync_delta_ms": round(1e3 * shift, 2)}


def to_world(joints_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(T, J, 3) camera-frame points (x_cam = R X + t, per frame) -> world."""
    return np.einsum("fji,fnj->fni", R, joints_cam - t[:, None, :])


def assign_roles(tracks: dict, take: str, cam: str, K: np.ndarray, frame_t: np.ndarray) -> dict:
    """tracks {id: joints_cam (T, J, 3) on frame_t} -> {"lead": id, "follow": id}: each track's projected 2D joints
    box against the lead / follow boxes of dancers/roles.json, mean IoU over frames, best assignment. frame_t is the
    phone's RAW frames.json clock (the clock of roles.json and the masks), not reference seconds."""
    from .holdout import box_iou
    roles = json.loads((PACKAGE_ROOT / "work" / take / "dancers" / "roles.json").read_text())["cams"][cam]["frames"]
    rt = np.array([f["t"] for f in roles])
    score = {}
    for tid, J in tracks.items():
        for role in ROLES:
            ious = []
            for k, tk in enumerate(frame_t):
                i = int(np.argmin(np.abs(rt - tk)))
                b = (roles[i].get(role) or {}).get("box")
                P = J[k][np.isfinite(J[k]).all(1)]
                if b is None or not len(P) or (P[:, 2] <= 0).any():
                    continue
                uv = (K @ (P / P[:, 2:3]).T).T[:, :2]
                ious.append(box_iou(np.array(b, float), np.r_[uv.min(0), uv.max(0)]))
            score[(tid, role)] = float(np.mean(ious)) if ious else 0.0
    ids = list(tracks)
    best, best_s = None, -1.0
    for a in ids:
        for b in ids:
            if a != b and score[(a, "lead")] + score[(b, "follow")] > best_s:
                best, best_s = {"lead": a, "follow": b}, score[(a, "lead")] + score[(b, "follow")]
    if best is None and ids:  # a single track: give it to the role it matches best
        a = ids[0]
        best = {max(ROLES, key=lambda r: score[(a, r)]): a}
    return {**(best or {}), "iou": {f"{k[0]}:{k[1]}": round(v, 3) for k, v in score.items()}}


def save(take: str, name: str, role: str, t: np.ndarray, joints: np.ndarray, frame: str, meta: dict | None = None) -> Path:
    d = cand_dir(take, name)
    d.mkdir(parents=True, exist_ok=True)
    if joints.shape[1] < 22:
        raise ValueError("joints must have >= 22 entries in SMPL body order (NaN for missing)")
    np.savez(d / f"{role}.npz", t=np.asarray(t, float), joints=np.asarray(joints, float), frame=frame)
    m = {"name": name, "frame": frame, **(json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}),
         **(meta or {})}
    (d / "meta.json").write_text(json.dumps(m, indent=1, default=str))
    return d


def _limb_err(A: np.ndarray, B: np.ndarray, js: list[int]) -> list[float]:
    out = []
    for a, b in LIMBS.values():
        if a in js and b in js:
            u, v = A[js.index(b)] - A[js.index(a)], B[js.index(b)] - B[js.index(a)]
            if np.isfinite(u).all() and np.isfinite(v).all():
                c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
                out.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    return out


def score_arrays(t: np.ndarray, J: np.ndarray, frame: str, ref_t: np.ndarray, ref_J: np.ndarray,
                 window: tuple[float, float]) -> dict:
    ir, it = match_times(ref_t, t)
    keep = (ref_t[ir] >= window[0]) & (ref_t[ir] <= window[1])
    ir, it = ir[keep], it[keep]
    rows = {"pa22": [], "pa14": [], "rr": [], "pelvis": [], "limb": []}
    for a, b in zip(it, ir):
        A, B = J[a], ref_J[b]
        for key, js in (("pa22", J22), ("pa14", J14)):
            ok = [j for j in js if np.isfinite(A[j]).all()]
            if len(ok) >= 10:
                rows[key].append(1e3 * np.linalg.norm(procrustes(A[ok], B[ok]) - B[ok], axis=-1).mean())
        if frame == "world" and np.isfinite(A[0]).all():  # orientation is only comparable in the world frame
            ok = [j for j in J22 if np.isfinite(A[j]).all()]
            rows["rr"].append(1e3 * np.linalg.norm((A[ok] - A[0]) - (B[ok] - B[0]), axis=-1).mean())
            if frame == "world":
                rows["pelvis"].append(1e3 * np.linalg.norm(A[0] - B[0]))
        ok = [j for j in J14 if np.isfinite(A[j]).all()]
        if len(ok) >= 10:
            rows["limb"] += _limb_err(procrustes(A[ok], B[ok]), B[ok], ok)
    return {"frames": int(len(ir)), "pa22_mm": summary(rows["pa22"]), "pa14_mm": summary(rows["pa14"]),
            "rr_mm": summary(rows["rr"]), "pelvis_mm": summary(rows["pelvis"]), "limb_deg": summary(rows["limb"])}


def score(take: str, name: str, window: tuple[float, float], reference: Path | None = None) -> dict:
    d = cand_dir(take, name)
    ref = Path(reference) if reference else PACKAGE_ROOT / "work" / take / "smplx"
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {"name": name}
    out = {"name": name, "meta": meta, "window": list(window), "reference": str(ref), "roles": {}}
    for role in ROLES:
        p = d / f"{role}.npz"
        if not p.exists():
            out["roles"][role] = {"frames": 0, "missing": True}
            continue
        z = np.load(p, allow_pickle=True)
        r = load_fit(ref, role)
        t, J, frame = np.asarray(z["t"], float), np.asarray(z["joints"], float), str(z["frame"])
        if meta.get("input_cam"):  # flag (never silently fix) a candidate saved on the raw frame clock
            ck = clock_check(take, meta["input_cam"], t, ref_grid=r["t"])
            out.setdefault("clock", {})[role] = ck
            if ck["clock"] == "raw":
                print(f"[cand] WARNING {name}/{role}: t is on phone {meta['input_cam']}'s raw frames.json clock, not "
                      f"reference seconds (sync delta {ck['sync_delta_ms']:+.1f} ms): save frame_to_ref(t)")
        if frame == "camera":  # through the input phone's track (per-frame R, t) into the world frame
            from .. import mamma_dyn as md
            tr = md.load_tracks(PACKAGE_ROOT / "work" / take, [meta["input_cam"]])[meta["input_cam"]]
            k = np.array([int(np.argmin(np.abs(tr.t - x))) for x in t])
            J, frame = to_world(J, tr.R[k], tr.tvec[k]), "world"
        out["roles"][role] = score_arrays(t, J, frame, r["t"], r["joints"], window)
    (d / "score.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def import_viggle(take: str, run_dir: Path, fps: float = 30.0) -> Path:
    """The Viggle GLBs of a viggle-test run -> candidate 'viggle_jst2' (14 joints, Viggle's own frame: frame 'free')."""
    from .viggle import glb_joints, mixamo_index
    for role in ROLES:
        names, times, P = glb_joints((run_dir / f"{role}_mixamo.glb").read_bytes(), fps=fps)
        tj = json.loads((run_dir / f"{role}.times.json").read_text())
        idx = mixamo_index(names)
        J = np.full((len(times), 22, 3), np.nan)
        for j, n in idx.items():
            J[:, j] = P[:, n]
        save(take, "viggle_jst2", role, frame_to_ref(take, "06", tj["t"][0] + times), J, "free",
             {"method": "Viggle JST-2 Motion 3D (API, Mixamo rig), one masked clip per dancer", "input_cam": "06",
              "source": str(Path(run_dir).resolve()), "cost_usd": 0.40,
              "clock": "t = clip start (06.times.json, frames.json clock) + GLB time + phone 06's sync delta"})
    return cand_dir(take, "viggle_jst2")


def multiview_floor(take: str, tag: str, window: tuple[float, float]) -> dict | None:
    """A multi-view fit of the same window (e.g. the 7-phone held-out chunk) scored like a candidate: the floor."""
    d = PACKAGE_ROOT / "work" / take / "mamma" / "fits" / tag / "smplx"
    if not d.exists():
        return None
    ref = PACKAGE_ROOT / "work" / take / "smplx"
    out = {}
    for role in ROLES:
        s, r = load_fit(d, role), load_fit(ref, role)
        out[role] = score_arrays(s["t"], s["joints"][:, :22], "world", r["t"], r["joints"], window)
    return out


def leaderboard(take: str, window: tuple[float, float], floor_tag: str | None = "h03_f736-968") -> str:
    root = PACKAGE_ROOT / "work" / take / "mono4d" / "cand"
    rows = []
    if floor_tag:
        f = multiview_floor(take, floor_tag, window)
        if f:
            rows.append((f"multi-view floor ({floor_tag}, 7 phones)", f))
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        if (d / "lead.npz").exists() or (d / "follow.npz").exists():
            s = score(take, d.name, window)
            rows.append((d.name, (s["roles"], s.get("clock", {}))))
    fmt = lambda s: "-" if not s or s.get("median") is None else f"{s['median']:.0f}"  # noqa: E731
    lines = [f"# One-camera candidates, {take}, {window[0]:.1f}-{window[1]:.1f} s (medians; reference = 8-phone fit)", "",
             "| candidate | role | frames | PA-MPJPE 22 (mm) | PA-MPJPE 14 (mm) | root-rel (mm) | pelvis world (mm) | limb (deg) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, roles in rows:
        if isinstance(roles, tuple):  # (roles, clock flags) of a candidate
            roles, clocks = roles
            if any(c.get("clock") == "raw" for c in clocks.values()):
                name += " (RAW CLOCK: ~1 frame off)"
        for role in ROLES:
            s = roles.get(role) or {}
            if not s.get("frames"):
                continue
            lines.append(f"| {name} | {role} | {s['frames']} | {fmt(s['pa22_mm'])} | {fmt(s['pa14_mm'])} | {fmt(s['rr_mm'])} | "
                         f"{fmt(s['pelvis_mm'])} | {fmt(s['limb_deg'])} |")
    text = "\n".join(lines) + "\n"
    if root.exists():
        (root / "leaderboard.md").write_text(text)
    return text
