"""john_lula: where is the single-camera fit wrong, and how plausible is its 3D (agent J3b, 2026-10-11).

    python -m dancecap.jl_fitcheck reproj  [--take john_lula] [--src smplx_rf2_noslip] [--out file.json]
    python -m dancecap.jl_fitcheck sanity  [--take john_lula] [--src smplx_rf2_noslip] [--out file.json]
    python -m dancecap.jl_fitcheck overlay --times 0.5 3 ...  [--src ...] [--out dir]

reproj   the fit's SMPL-X body joints (smplx/<role>.npz) projected through the per-frame camera (work/john_lula/cameras/01.npz) against the RTMW body keypoints of J3a
         (people/teachers_01.npz, both confidences >= 0.5, skeleton not lying on the partner), as PERCENT OF THE PERSON'S HEIGHT IN THE PICTURE (the detector's own noise and the
         COCO-vs-SMPL-X joint definitions cost 2-3 %); per second and per role, split by the camera's zoom state. This is the only check that uses the picture: it is a 2D
         check. A monocular fit can match every 2D joint and still be wrong along the optical axis, which only the 3D checks below can hint at.
sanity   plausibility of the 3D: feet on the floor (lowest vertex per frame), speeds (pelvis, ankles), the two bodies against each other (nearest surface points), the rest
         height against the stature, the camera-depth profile of each dancer.
overlay  the skeletons on the source frames (contact sheets), lead green, follow red.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE

DANCE = "john_lula"
WORK = PACKAGE_ROOT / "work" / DANCE
PEOPLE = WORK / "people"
VIDEO = WORKSPACE / "dances" / "John-Lula" / "john_lula_jnj.mp4"
COCO_TO_SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
BAD_PCT = 5.0
CONF = 0.5
ROLES = ("lead", "follow")
BONES = [(0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11), (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]


def project(X: np.ndarray, R: np.ndarray, tvec: np.ndarray, K: np.ndarray, k1: float) -> tuple[np.ndarray, np.ndarray]:
    Xc = X @ R.T + tvec
    z = np.where(Xc[..., 2] > 1e-6, Xc[..., 2], 1e-6)
    x, y = Xc[..., 0] / z, Xc[..., 1] / z
    d = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], -1), Xc[..., 2]


def load_fit(take: str, src: str) -> dict:
    d = PACKAGE_ROOT / "work" / take / src
    return {r: dict(np.load(d / f"{r}.npz", allow_pickle=True)) for r in ROLES}


def frame_map(t: np.ndarray, tc: np.ndarray, max_dt: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Nearest camera frame of each fit instant, and whether it is close enough."""
    j = np.clip(np.searchsorted(tc, t), 1, len(tc) - 1)
    j = np.where(np.abs(tc[j - 1] - t) < np.abs(tc[j] - t), j - 1, j)
    return j, np.abs(tc[j] - t) <= max_dt


def reproj(take: str = DANCE, src: str = "smplx_rf2_noslip", cams_file: str | None = None) -> dict:
    S = load_fit(take, src)
    cam = np.load(cams_file or WORK / "cameras" / "01.npz", allow_pickle=True)
    T_ = np.load(PEOPLE / "teachers_01.npz", allow_pickle=True)
    t = np.asarray(S["lead"]["t"], float)
    j, near = frame_map(t, np.asarray(cam["t"], float))
    ok = np.asarray(cam["ok"], bool)[j] & near
    zoom2 = np.asarray(cam["zoom_2x"], bool)[j] if "zoom_2x" in cam.files else np.zeros(len(t), bool)
    err = {r: np.full(len(t), np.nan) for r in ROLES}
    per_joint = {r: np.full((len(t), len(COCO_TO_SMPLX)), np.nan) for r in ROLES}
    depth_signed = {r: np.full(len(t), np.nan) for r in ROLES}
    for ri, role in enumerate(ROLES):
        J = np.asarray(S[role]["joints"], float)
        for i in np.flatnonzero(ok):
            k = int(j[i])
            fr = int(np.asarray(cam["frame_index"])[k])
            if T_["conf"][fr, ri] < CONF or T_["skel_in_partner"][fr, ri] > 0.5:
                continue
            kp = T_["kp"][fr, ri]
            ids = [c for c in COCO_TO_SMPLX if kp[c, 2] >= CONF]
            if len(ids) < 6:
                continue
            uv, z = project(J[i, [COCO_TO_SMPLX[c] for c in ids]], np.asarray(cam["R"][k], float), np.asarray(cam["tvec"][k], float), np.asarray(cam["K"][k], float),
                            float(np.asarray(cam["dist"][k]).reshape(-1)[0]))
            hpx = max(float(T_["box"][fr, ri, 3] - T_["box"][fr, ri, 1]), 50.0)
            e = 100.0 * np.linalg.norm(uv - kp[ids, :2], axis=-1) / hpx
            m = z > 0
            if m.sum() >= 6:
                err[role][i] = float(np.median(e[m]))
                for c, v in zip(ids, e):
                    per_joint[role][i, list(COCO_TO_SMPLX).index(c)] = v
            depth_signed[role][i] = float((np.asarray(cam["R"][k], float) @ J[i, 0] + np.asarray(cam["tvec"][k], float))[2])
    sec = np.floor(t - t[0]).astype(int)
    out = {"take": take, "src": src, "frames": int(len(t)), "per_second": {}, "summary": {}}
    for role in ROLES:
        secs, vals = [], []
        for s in range(int(sec.max()) + 1):
            m = (sec == s) & np.isfinite(err[role])
            secs.append(round(float(t[0] + s), 1))
            vals.append(round(float(np.median(err[role][m])), 1) if m.sum() >= 5 else None)
        fin = err[role][np.isfinite(err[role])]
        z2 = err[role][np.isfinite(err[role]) & zoom2]
        z1 = err[role][np.isfinite(err[role]) & ~zoom2]
        out["per_second"][role] = {"sec": secs, "pct": vals}
        out["summary"][role] = {"median": round(float(np.median(fin)), 2) if len(fin) else None, "p90": round(float(np.percentile(fin, 90)), 2) if len(fin) else None,
                                "frames": int(len(fin)), "median_wide": round(float(np.median(z1)), 2) if len(z1) else None,
                                "median_tight": round(float(np.median(z2)), 2) if len(z2) else None, "bad_seconds": [s for s, v in zip(secs, vals) if v is not None and v >= BAD_PCT],
                                "seconds_without_reference": [s for s, v in zip(secs, vals) if v is None]}
        names = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
        out["summary"][role]["median_by_joint_pct"] = {n: (round(float(np.nanmedian(per_joint[role][:, q])), 2) if np.isfinite(per_joint[role][:, q]).any() else None) for q, n in enumerate(names)}
    return out


def sanity(take: str = DANCE, src: str = "smplx_rf2_noslip", stride: int = 3) -> dict:
    """3D plausibility. Floor: lowest vertex of every frame (LBS of the shaped rest mesh, no pose correctives: mm-level, cameras.world_scale.floor_stats); speeds from the joints
    (mm / frame -> m/s at the lattice rate); the two bodies' nearest surface points on a sparse vertex set; rest height vs stature (+ sole)."""
    from . import physics_body as pb
    from .cameras import world_scale as ws
    from .cameras.world import sole_from_take
    import tomllib
    S = load_fit(take, src)
    cfg = tomllib.loads((PACKAGE_ROOT / "takes" / f"{DANCE}.toml").read_text(encoding="utf-8"))
    soles, _ = sole_from_take(cfg)
    t = np.asarray(S["lead"]["t"], float)
    dt = float(np.median(np.diff(t)))
    cam = np.load(WORK / "cameras" / "01.npz", allow_pickle=True)
    j, near = frame_map(t, np.asarray(cam["t"], float))
    out = {"take": take, "src": src, "frames": int(len(t)), "per_role": {}}
    verts = {}
    for role in ROLES:
        npz = S[role]
        body = pb.load_body(npz)
        pose = pb.full_pose(npz)
        Rm, p = pb.forward_kinematics(body, pose, np.asarray(npz["transl"], float).reshape(len(pose), 3))
        idx = np.arange(0, len(body.v_rest), 20)
        low, vs = [], []
        for s0 in range(0, len(Rm), 32):
            V = pb.lbs_points(body, Rm[s0:s0 + 32], p[s0:s0 + 32], np.arange(len(body.v_rest)))
            low.append(V[..., 1].min(1))
            vs.append(V[:, idx])
        low = np.concatenate(low)
        verts[role] = np.concatenate(vs)
        J = np.asarray(npz["joints"], float)
        sp = lambda x: np.linalg.norm(np.diff(x, axis=0), axis=-1) / dt     # noqa: E731
        pel, ank = sp(J[:, 0]), np.concatenate([sp(J[:, 7]), sp(J[:, 8])])
        wr = np.concatenate([sp(J[:, 20]), sp(J[:, 21])])
        zc = np.array([(np.asarray(cam["R"][k], float) @ J[i, 0] + np.asarray(cam["tvec"][k], float))[2] for i, k in enumerate(j)])
        H = ws.rest_height(np.asarray(npz["betas"], float), ws.body_model(pb.model_file(npz)))
        stat = float(cfg["dancers"][role]["stature_m"])
        out["per_role"][role] = {
            "lowest_vertex_m": {"p1": round(float(np.percentile(low, 1)), 4), "median": round(float(np.median(low)), 4), "p99": round(float(np.percentile(low, 99)), 4),
                                "share_below_minus_1_5cm": round(float((low < -0.015).mean()), 4), "share_above_10cm": round(float((low > 0.10).mean()), 4),
                                "share_above_25cm": round(float((low > 0.25).mean()), 4)},
            "pelvis_speed_ms": {"median": round(float(np.median(pel)), 2), "p95": round(float(np.percentile(pel, 95)), 2), "max": round(float(pel.max()), 2)},
            "ankle_speed_ms": {"median": round(float(np.median(ank)), 2), "p95": round(float(np.percentile(ank, 95)), 2), "p99": round(float(np.percentile(ank, 99)), 2), "max": round(float(ank.max()), 2)},
            "wrist_speed_ms": {"p95": round(float(np.percentile(wr, 95)), 2), "max": round(float(wr.max()), 2)},
            "rest_height_m": round(H, 4), "stature_plus_sole_m": round(stat + soles[role], 4), "rest_height_ratio": round(H / (stat + soles[role]), 4),
            "pelvis_height_m": {"median": round(float(np.median(J[:, 0, 1])), 3), "min": round(float(J[:, 0, 1].min()), 3)},
            "camera_depth_m": {"p5": round(float(np.percentile(zc, 5)), 2), "median": round(float(np.median(zc)), 2), "p95": round(float(np.percentile(zc, 95)), 2)}}
        out.setdefault("_arrays", {})[role] = {"low": low, "pelvis": J[:, 0]}
    # the couple: nearest surface points of the two bodies
    d = np.empty(len(t))
    for i in range(0, len(t), stride):
        a, b = verts["lead"][i], verts["follow"][i]
        d[i:i + stride] = float(np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1).min()))
    out["couple"] = {"nearest_points_m": {"min": round(float(d.min()), 4), "p1": round(float(np.percentile(d, 1)), 4), "median": round(float(np.median(d)), 3)},
                     "share_closer_than_1cm": round(float((d < 0.01).mean()), 4), "share_closer_than_2cm": round(float((d < 0.02).mean()), 4),
                     "note": "sparse vertex set (every 20th, about 5 cm apart): two surfaces that overlap by a few cm read under 2 cm; an embrace touches by design"}
    pl, pf = out["_arrays"]["lead"]["pelvis"], out["_arrays"]["follow"]["pelvis"]
    out["couple"]["pelvis_distance_m"] = {"min": round(float(np.linalg.norm(pl - pf, axis=1).min()), 3), "median": round(float(np.median(np.linalg.norm(pl - pf, axis=1))), 3)}
    out["john_taller_than_lula_rest_m"] = round(out["per_role"]["lead"]["rest_height_m"] - out["per_role"]["follow"]["rest_height_m"], 4)
    out["_arrays"] = {r: {"low_by_frame_m": [round(float(x), 3) for x in v["low"][::10]]} for r, v in out["_arrays"].items()}
    out["pair_distance_by_frame_m_every10"] = [round(float(x), 3) for x in d[::10]]
    return out


def overlay(take: str, times: list[float], src: str = "smplx_rf2_noslip", out: Path | None = None, width: int = 480, cols: int = 3, joints: bool = True) -> list[Path]:
    """Contact sheets of the fit's skeletons (lead green, follow red) over the source video frames at the given video times."""
    import cv2
    S = load_fit(take, src)
    t = np.asarray(S["lead"]["t"], float)
    cam = np.load(WORK / "cameras" / "01.npz", allow_pickle=True)
    tc = np.asarray(cam["t"], float)
    out = out or WORK / "review" / "fits"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    tiles = []
    for tt in times:
        i = int(np.argmin(np.abs(t - tt)))
        if abs(t[i] - tt) > 0.1:
            continue
        k = int(np.argmin(np.abs(tc - t[i])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.asarray(cam["frame_index"])[k]))
        ok, img = cap.read()
        if not ok:
            continue
        sc = width / img.shape[1]
        img = cv2.resize(img, (width, int(img.shape[0] * sc)))
        for role, col in (("lead", (60, 220, 60)), ("follow", (60, 60, 240))):
            J = np.asarray(S[role]["joints"], float)[i]
            uv, z = project(J, np.asarray(cam["R"][k], float), np.asarray(cam["tvec"][k], float), np.asarray(cam["K"][k], float), float(np.asarray(cam["dist"][k]).reshape(-1)[0]))
            uv = uv * sc
            for a, b in BONES:
                cv2.line(img, (int(uv[a, 0]), int(uv[a, 1])), (int(uv[b, 0]), int(uv[b, 1])), col, 2)
            if joints:
                for q in range(22):
                    cv2.circle(img, (int(uv[q, 0]), int(uv[q, 1])), 2, (255, 255, 255), -1)
        cv2.putText(img, f"t={tt:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(img)
    cap.release()
    paths = []
    per = cols * 2
    for n0 in range(0, len(tiles), per):
        grp = tiles[n0:n0 + per]
        while len(grp) % cols:
            grp.append(np.zeros_like(grp[0]))
        sheet = np.vstack([np.hstack(grp[q:q + cols]) for q in range(0, len(grp), cols)])
        pth = out / f"{take}_{src}_{int(times[n0])}_{int(times[min(len(times), n0 + per) - 1])}.jpg"
        cv2.imwrite(str(pth), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(pth)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_fitcheck", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("reproj", "sanity", "overlay"))
    ap.add_argument("--take", default=DANCE)
    ap.add_argument("--src", default="smplx_rf2_noslip")
    ap.add_argument("--out")
    ap.add_argument("--times", nargs="*", type=float)
    ap.add_argument("--cams-file")
    a = ap.parse_args(argv)
    if a.cmd == "overlay":
        for p in overlay(a.take, a.times or [], a.src, Path(a.out) if a.out else None):
            print(p)
        return 0
    rep = reproj(a.take, a.src, a.cams_file) if a.cmd == "reproj" else sanity(a.take, a.src)
    out = Path(a.out) if a.out else WORK / "review" / f"fitcheck_{a.cmd}_{a.take}_{a.src}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["summary"] if a.cmd == "reproj" else {k: v for k, v in rep.items() if k not in ("pair_distance_by_frame_m_every10", "_arrays")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
