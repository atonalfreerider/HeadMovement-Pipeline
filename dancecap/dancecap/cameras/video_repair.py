"""Repair of a stretch of an exported camera track that the VIDEO contradicts (camera-tracking gate).

larissa_kadu_full cam09, -3.13 to -1.62 s: the localiser finds, for the first 47 frames of the recording, a pose that
fits its static correspondences (450 inliers, 1.2-1.7 px) at f/max-dim 1.2-1.6 and a camera centre 6 units behind the
real one (the wide first seconds do not match the zoomed-in map images: the dolly-zoom ambiguity plus the mirror
images in the glossy floor); the zoom-depth guard then re-centres them and the focal settles at 0.96. At -1.6 s the
exported camera jumps by 20.9 deg and the focal by 14 %, while the video shows a slow pan and no zoom (similarity scale
+0.17 % per 1/6 s over the pair, 323 of 339 matches). cameras/video_motion.py finds such jumps; the repair:

  anchor   the first frame after the jump whose next 20 frames agree with their own neighbours (no jump, inliers >= 400)
  focal    anchor focal x exp(-sum of the video's zoom between the frame and the anchor)   (video_zoom.py: ~0 here)
  centre   the anchor's centre, held (a handheld phone in 1.5 s; the along-axis centre is not observable in the wide
           shot, that is the ambiguity that produced the error)
  rotation chained backward from the anchor through the VIDEO's own frame-to-frame motion (the stored correspondences of
           these frames are a mirror-image mode: with focal and centre fixed they still pull the rotation 21 deg away):
           per video pair (1/6 s) the pure-rotation Kabsch fit of its matched points, slerp between the pair times.

Their static reprojection is not meaningful (reproj_px NaN); they are flagged in the export (`repaired`), never hidden.
Accuracy: ~0.03 deg per pair (video_motion), so ~0.1-0.2 deg over 1.5 s.

    python -m dancecap.cameras.video_repair detect                 # jumps in every exported track (no files written)
    python -m dancecap.cameras.video_repair apply a 09             # rewrite cameras/09.npz of larissa_kadu_full_a (backup kept)
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from . import gate_cameras as gc


def rotation_steps(R: np.ndarray) -> np.ndarray:
    """Angle (deg) between the rotations of adjacent frames, (F-1,)."""
    from scipy.spatial.transform import Rotation
    return np.degrees(np.linalg.norm(Rotation.from_matrix(R[1:] @ np.transpose(R[:-1], (0, 2, 1))).as_rotvec(), axis=1))


def find_jumps(d: dict, vid: dict | None = None, min_deg: float = 3.0, min_focal_pct: float = 0.0) -> list[dict]:
    """Adjacent-frame rotation jumps of an exported track (> min_deg between two frames; a handheld phone does < 1.5)
    and, when the video measurement is given, the video's own zoom over the jump (small = the picture did not jump)."""
    st = rotation_steps(d["R"])
    out = []
    for i in np.flatnonzero(st > min_deg):
        e = {"t": round(float(d["t"][i + 1]), 3), "frame": int(i + 1), "rotation_deg": round(float(st[i]), 2),
             "focal_pct": round(100 * float(d["f"][i + 1] / d["f"][i] - 1), 1)}
        if vid is not None:
            t0 = vid["offset"] + vid["slope"] * vid["pts0"]
            t1 = vid["offset"] + vid["slope"] * vid["pts1"]
            k = np.flatnonzero((t0 <= d["t"][i + 1] + 0.1) & (t1 >= d["t"][i] - 0.1) & vid["ok"])
            if len(k):
                e["video_zoom_pct"] = round(100 * float(np.exp(np.sum(vid["dlog"][k])) - 1), 2)
                e["video_matches"] = int(np.median(vid["inl"][k]))
        out.append(e)
    return out


def anchor_after(d: dict, i_jump: int, n: int = 20, min_inliers: int = 400) -> int:
    """First frame >= i_jump with min_inliers whose next n frames have no rotation step > 1.5 deg."""
    st = rotation_steps(d["R"])
    for i in range(i_jump, len(d["t"]) - n - 1):
        if d["n_inliers_raw"][i] >= min_inliers and d["ok"][i] and not (st[i:i + n] > 1.5).any():
            return i
    raise ValueError("no stable anchor after the jump")


def video_zoom_between(vid: dict, t_a: float, t_b: float) -> float:
    """Sum of the video's log zoom between reference times t_a < t_b (pairs inside; undefined pairs count 0)."""
    t0 = vid["offset"] + vid["slope"] * vid["pts0"]
    t1 = vid["offset"] + vid["slope"] * vid["pts1"]
    k = (t0 >= t_a - 1e-3) & (t1 <= t_b + 1e-3) & vid["ok"]
    return float(np.sum(vid["dlog"][k]))


def relative_rotation(a: np.ndarray, b: np.ndarray, f0: float, f1: float, cx: float, cy: float) -> np.ndarray:
    """Rotation R (x_cam1 = R x_cam0) of a pure-rotation pair from the matches a -> b (native px), focal f0 / f1:
    Kabsch on the normalised bearings (SVD), then one trimmed refit that drops the worst 20 %."""
    ra = np.c_[(a[:, 0] - cx) / f0, (a[:, 1] - cy) / f0, np.ones(len(a))]
    rb = np.c_[(b[:, 0] - cx) / f1, (b[:, 1] - cy) / f1, np.ones(len(b))]
    ra /= np.linalg.norm(ra, axis=1, keepdims=True)
    rb /= np.linalg.norm(rb, axis=1, keepdims=True)
    keep = np.ones(len(a), bool)
    R = np.eye(3)
    for _ in range(2):
        H = ra[keep].T @ rb[keep]
        U, _, Vt = np.linalg.svd(H)
        R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
        e = np.linalg.norm(rb - ra @ R.T, axis=1)
        keep = e <= np.percentile(e, 80)
    return R


def repair_stretch(d: dict, vid: dict, i_end: int, i_anchor: int, start: int = 0) -> dict:
    """Replace frames start..i_end-1 backward from the anchor (see the module docstring). The rotation comes from the
    VIDEO, not from the stored correspondences: those of the first 46 frames of cam09 are a mirror-image mode (fixing
    the centre and the focal, they still pull the rotation 21 deg away). d: gate_cameras.load_cam dict."""
    from scipy.spatial.transform import Rotation, Slerp
    from .video_motion import to_frame_orientation
    cx, cy = d["w"] / 2 - 0.5, d["h"] / 2 - 0.5
    t = d["t"]
    C_a = d["C"][i_anchor].copy()
    f_a = float(d["f"][i_anchor])
    idx = np.arange(start, i_end)
    f_of = lambda tt: f_a * np.exp(-video_zoom_between(vid, tt, t[i_anchor]))  # noqa: E731
    f = np.array([f_of(t[i]) for i in idx])
    t0 = vid["offset"] + vid["slope"] * vid["pts0"]
    t1 = vid["offset"] + vid["slope"] * vid["pts1"]
    # video pairs between the first frame and the anchor, newest first; the chain starts at the pair that ends at the
    # frame nearest to the anchor
    k_end = int(np.argmin(np.abs(t1 - t[i_anchor])))
    if abs(t1[k_end] - t[i_anchor]) > 0.034:
        raise ValueError("the anchor is not on a video sample")
    end_R = d["R"][i_anchor]                      # rotation (world -> camera) at the END of the pair being processed
    times, rots, n_used = [t[i_anchor]], [end_R], []
    for k in range(k_end, -1, -1):
        if t0[k] < t[start] - 0.034:
            break
        a, b = vid["ma"][k], vid["mb"][k]
        m = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
        a, b = (to_frame_orientation(P[m].astype(float), d.get("rotation", 0), d["w"], d["h"]) for P in (a, b))
        if not vid["ok"][k] or len(a) < 15:
            R_rel = np.eye(3)                      # no usable pair: hold the rotation
        else:
            R_rel = relative_rotation(a, b, f_of(t0[k]), f_of(t1[k]), cx, cy)
            n_used.append(len(a))
        end_R = R_rel.T @ end_R                    # x_cam1 = R_rel x_cam0  =>  R_c0 = R_rel^T R_c1
        times.append(t0[k])
        rots.append(end_R)
    order = np.argsort(times)
    times, rots = np.asarray(times)[order], np.stack(rots)[order]
    sl = Slerp(times, Rotation.from_matrix(rots))
    R = sl(np.clip(t[idx], times[0], times[-1])).as_matrix()
    return {"frames": idx, "R": R, "f": f, "C": np.repeat(C_a[None], len(idx), 0), "reproj_px": np.full(len(idx), np.nan),
            "anchor": int(i_anchor), "pairs_used": len(n_used), "matches_median": int(np.median(n_used)) if n_used else 0}


def apply(seg: str, cam: str, backup: str = "s2_gate_backup", log=print) -> dict | None:
    """Detect the start jump of cam in segment seg, repair the stretch before it and rewrite cameras/<cam>.npz."""
    work = gc.seg_work(seg)
    d = gc.load_cam(work, cam)
    vid = gc.load_video(cam)
    jumps = find_jumps(d, vid)
    jumps = [j for j in jumps if j["frame"] <= 0.25 * len(d["t"]) and abs(j.get("video_zoom_pct", 0)) < 5.0]
    if not jumps:
        log(f"[video_repair] {work.name} cam{cam}: no jump the video contradicts")
        return None
    j = jumps[0]
    ia = anchor_after(d, j["frame"])
    t1 = vid["offset"] + vid["slope"] * vid["pts1"]
    k = int(np.flatnonzero(t1 >= d["t"][ia] - 1e-3)[0])    # first video sample at or after the anchor
    ia = int(np.argmin(np.abs(d["t"] - t1[k])))           # the frame at that sample: the chain starts there
    if ia < j["frame"]:
        raise ValueError("anchor before the jump")
    rep = repair_stretch(d, vid, j["frame"], ia)
    p = work / "cameras" / f"{cam}.npz"
    bdir = work / "cameras" / backup
    bdir.mkdir(parents=True, exist_ok=True)
    if not (bdir / p.name).exists():
        shutil.copy2(p, bdir / p.name)
    e = dict(np.load(bdir / p.name, allow_pickle=True))      # always patch the ORIGINAL export, not an earlier repair
    idx = rep["frames"]
    for q, i in enumerate(idx):
        e["R"][i] = rep["R"][q]
        e["tvec"][i] = -rep["R"][q] @ rep["C"][q]
        e["K"][i, 0, 0] = e["K"][i, 1, 1] = rep["f"][q]
        e["ok"][i] = True
        e["reproj_px"][i] = np.nan                       # the static points of these frames are a mirror mode: not used
    flag = np.zeros(len(e["t"]), bool)
    flag[idx] = True
    e["repaired"] = flag
    np.savez(p, **e)
    info = {"seg": seg, "cam": cam, "jump": j, "frames": [int(idx[0]), int(idx[-1])],
            "t": [round(float(d["t"][idx[0]]), 3), round(float(d["t"][idx[-1]]), 3)], "anchor_frame": rep["anchor"],
            "anchor_t": round(float(d["t"][rep["anchor"]]), 3),
            "f_over_maxdim_before": [round(float(d["f"][idx].min() / d["maxdim"]), 3), round(float(d["f"][idx].max() / d["maxdim"]), 3)],
            "f_over_maxdim_after": [round(float(rep["f"].min() / d["maxdim"]), 3), round(float(rep["f"].max() / d["maxdim"]), 3)],
            "video_pairs_used": rep["pairs_used"], "matches_median": rep["matches_median"],
            "centre_held_sfm": [round(float(v), 4) for v in rep["C"][0]]}
    rp = work / "cameras" / "video_repairs.json"
    allr = json.loads(rp.read_text()) if rp.exists() else []
    allr = [r for r in allr if not (r["cam"] == cam and r["seg"] == seg)] + [info]
    rp.write_text(json.dumps(allr, indent=1))
    log(f"[video_repair] {work.name} cam{cam}: repaired t {info['t']} ({len(idx)} frames), focal {info['f_over_maxdim_before']} -> "
        f"{info['f_over_maxdim_after']}, {info['video_pairs_used']} video pairs")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.video_repair", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["detect", "apply"])
    ap.add_argument("seg", nargs="?")
    ap.add_argument("cam", nargs="?")
    a = ap.parse_args(argv)
    if a.what == "detect":
        for seg in ("38",) + gc.SEGS:
            for cam in gc.CAMS:
                d = gc.load_cam(gc.seg_work(seg), cam)
                if d is None:
                    continue
                for j in find_jumps(d, gc.load_video(cam)):
                    print(seg, cam, j)
    else:
        apply(a.seg, a.cam)


if __name__ == "__main__":
    main()
