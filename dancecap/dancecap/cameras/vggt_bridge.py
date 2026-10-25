"""Bridge a moving camera's localisation holes with VGGT's relative camera motion (two-view takes; lucas_luiza 2026-10-25).

Where the room map itself puts a moving camera on the wrong wall (repeated doors / EXIT signs: cameras.anchored refuses
every pose there), no map-based localisation fills the hole. The camera's own frames still overlap from one to the next:
VGGT gives the relative motion of a short clip without matching against the map. The clip spans the hole plus verified
frames on both sides; its groups are chained by their overlapping frames and the chain is aligned to the verified
(anchored) poses on both sides; every bridged pose is then judged by the fixed camera through the dancers (cameras.anchored).

    python -m dancecap.cameras.vggt_bridge run   <take> --cam 01 --span 57:86.5 [--step 4 --group 32 --overlap 8]   # GPU pod
    python -m dancecap.cameras.vggt_bridge align <take> --cam 01 --fixed 02 --npz <run npz>                           # CPU

run writes work/<take>/cameras/vggt/<cam>_<t0>_<t1>.npz: frame (n,) window-frame indices, group (n,), ext (n, 3, 4)
cam_from_world in its group's frame (VGGT: the group's first camera = identity, a scale per group), K (n, 3, 3) in the
original frame's pixels. align prints / writes cameras/qa/vggt_bridge_<cam>_<t0>_<t1>.json: the chain's alignment residual
on the verified poses (each side), and the judge's epi / transfer on the bridged frames.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[vggt_bridge {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _frames(work: Path, cam: str, t0: float, t1: float, step: int) -> list[tuple[int, float, str]]:
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    rows = [(i, float(f["t"]), f["file"]) for i, f in enumerate(meta["frames"]) if t0 <= float(f["t"]) <= t1]
    return rows[::step]


def cmd_run(work: Path, cam: str, t0: float, t1: float, step: int, group: int, overlap: int,
            name: str = "facebook/VGGT-1B") -> Path:
    import torch
    from .geom_priors import ensure
    ensure("from vggt.models.vggt import VGGT", "git+https://github.com/facebookresearch/vggt.git", extra=("einops", "safetensors"))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    rows = _frames(work, cam, t0, t1, step)
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    w0, h0 = int(meta["width"]), int(meta["height"])
    model = VGGT.from_pretrained(name).to("cuda").eval()
    out = {"frame": [], "group": [], "ext": [], "K": []}
    starts = list(range(0, max(1, len(rows) - overlap), group - overlap))
    for g, a in enumerate(starts):
        rs = rows[a:a + group]
        if len(rs) < 2:
            continue
        x = load_and_preprocess_images([str(work / "frames" / cam / f) for _, _, f in rs], mode="pad").to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)
        ex, K = pose_encoding_to_extri_intri(pred["pose_enc"], x.shape[-2:])
        ex, K = ex[0].float().cpu().numpy(), K[0].float().cpu().numpy()
        S = x.shape[-1]  # the padded square (mode pad: the frame scaled to fit S, centred)
        sc = max(w0, h0) / S
        for k, (i, _, _) in enumerate(rs):
            Ko = K[k].copy()
            Ko[:2] *= sc
            Ko[0, 2] -= (max(w0, h0) - w0) / 2.0
            Ko[1, 2] -= (max(w0, h0) - h0) / 2.0
            out["frame"].append(i), out["group"].append(g), out["ext"].append(ex[k]), out["K"].append(Ko)
        log(f"group {g}: frames {rs[0][0]}..{rs[-1][0]} ({len(rs)}), focal {np.median(K[:, 0, 0]) * sc:.0f} px")
        del pred
    p = work / "cameras" / "vggt" / f"{cam}_{int(round(t0 * 1000))}_{int(round(t1 * 1000))}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **{k: np.array(v) for k, v in out.items()}, span=np.array([t0, t1]), step=step, group_size=group, overlap=overlap)
    log(f"-> {p}")
    return p


# ------------------------------------------------------------------------------------------------ align


def _rot_mean(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.sum(Rs, 0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def sim3_from_poses(Ra, Ca, Rb, Cb) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, Q, t) with world_b = s Q world_a + t from the same cameras in two frames: Q from the rotations
    (R_b = R_a Q^T), s and t from the centres (least squares, Q fixed)."""
    Q = _rot_mean(np.einsum("nji,njk->nik", Rb, Ra))  # R_b^T R_a = Q for cam_from_world R_b = R_a Q^T
    A = Ca @ Q.T
    am, bm = A.mean(0), Cb.mean(0)
    da, db = A - am, Cb - bm
    s = float((da * db).sum() / max((da * da).sum(), 1e-12)) if len(A) > 1 else 1.0
    return s, Q, bm - s * am


def cmd_align(work: Path, cam: str, fixed: str, npz: Path, off_ms: float | None, log=log) -> dict:
    from . import anchored as AN
    z = np.load(npz)
    fr, grp, ext = z["frame"], z["group"], z["ext"]
    R = ext[:, :, :3]
    C = -np.einsum("nji,nj->ni", R, ext[:, :, 3])
    # chain the groups: group g+1 into group 0's frame by their shared frames
    Rw, Cw = R.copy(), C.copy()
    groups = sorted(set(grp.tolist()))
    for g in groups[1:]:
        prev = grp == g - 1
        cur = grp == g
        shared = np.intersect1d(fr[prev], fr[cur])
        if len(shared) < 2:
            raise SystemExit(f"[vggt_bridge] groups {g - 1}/{g} share {len(shared)} frames")
        ip = np.array([np.flatnonzero(prev & (fr == f))[0] for f in shared])
        ic = np.array([np.flatnonzero(cur & (fr == f))[0] for f in shared])
        s, Q, t = sim3_from_poses(Rw[ic], Cw[ic], Rw[ip], Cw[ip])  # cur's frame -> prev's (already chained) frame
        Rw[cur] = np.einsum("nij,kj->nik", Rw[cur], Q)  # R_new = R Q^T
        Cw[cur] = s * Cw[cur] @ Q.T + t
    # one pose per frame (the first group's where a frame is in two)
    _, first = np.unique(fr, return_index=True)
    fr, Rw, Cw = fr[first], Rw[first], Cw[first]
    loc = dict(np.load(work / "cameras" / "cache" / f"loc_{cam}.npz"))
    t = np.asarray(loc["t"], float)
    ok = np.asarray(loc["ok"], bool) & np.isin(np.asarray(loc["source"]).astype(str), ("anchor", "cont", "prior"))
    Rl = np.asarray(loc["R"], float)
    Cl = AN.centres(Rl, np.asarray(loc["tvec"], float))
    anc = ok[fr]
    if anc.sum() < 3:
        raise SystemExit(f"[vggt_bridge] only {int(anc.sum())} verified frames in the clip")
    tm = t[fr]
    hole = ~anc
    t_hole = tm[hole]
    side = np.where(tm < (t_hole.min() if len(t_hole) else tm.mean()), 0, 1)
    s, Q, tt = sim3_from_poses(Rw[anc], Cw[anc], Rl[fr][anc], Cl[fr][anc])
    Rm = np.einsum("nij,kj->nik", Rw, Q)
    Cm = s * Cw @ Q.T + tt
    off, why = AN.offset_ms(work, off_ms, cam, fixed)
    J = AN.Judge(work, cam, fixed, off / 1000.0)
    room_unit_m = float(J.world["s"]) if J.world else 1.0
    err_c = np.linalg.norm(Cm[anc] - Cl[fr][anc], axis=1) * room_unit_m
    err_r = AN.rot_deg(Rm[anc], Rl[fr][anc])
    f_map = float(np.median(np.asarray(loc["f"], float)[ok]))
    k1 = float(np.median(np.asarray(loc["k1"], float)[ok]))
    tvec = -np.einsum("nij,nj->ni", Rm, Cm)
    epi, xf, nj = J.judge(tm, J.joints_at(tm), Rm, tvec, np.full(len(fr), f_map), np.full(len(fr), k1))
    epi_l, xf_l, _ = J.judge(tm[anc], J.joints_at(tm[anc]), Rl[fr][anc], np.asarray(loc["tvec"], float)[fr][anc],
                             np.full(int(anc.sum()), f_map), np.full(int(anc.sum()), k1))
    refused = J.refused(epi, xf)
    pct = lambda x: {str(q): round(float(np.nanpercentile(x, q)), 2) for q in (50, 90)} if np.isfinite(x).any() else None  # noqa: E731
    rep = {"npz": str(npz), "frames": int(len(fr)), "verified": int(anc.sum()), "hole_frames": int(hole.sum()),
           "sim3_scale": round(s, 4), "offset_ms": off,
           "anchor_err_m": {"all": pct(err_c), "side0": pct(err_c[side[anc] == 0]), "side1": pct(err_c[side[anc] == 1])},
           "anchor_err_deg": pct(err_r),
           "hole": {"epi_px": pct(epi[hole]), "transfer_px": pct(xf[hole]), "judged": int(np.isfinite(epi[hole]).sum()),
                    "refused": int(refused[hole].sum()), "anchors_by_judge": int(J.anchor(epi[hole], xf[hole]).sum())},
           "verified_frames_judge": {"epi_px": pct(epi_l), "transfer_px": pct(xf_l)},
           "bridged_on_verified": {"epi_px": pct(epi[anc]), "transfer_px": pct(xf[anc])}}
    log(json.dumps(rep, indent=1))
    p = work / "cameras" / "qa" / f"vggt_bridge_{npz.stem}.json"
    p.write_text(json.dumps(rep, indent=1))
    np.savez(work / "cameras" / "vggt" / f"{npz.stem}_aligned.npz", frame=fr, R=Rm, C=Cm, tvec=tvec, epi=epi, xf=xf, verified=anc)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.vggt_bridge", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("run", "align"))
    ap.add_argument("take")
    ap.add_argument("--cam", default="01")
    ap.add_argument("--fixed", default="02")
    ap.add_argument("--span", help="t0:t1 (s, the take's clock)")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--npz")
    ap.add_argument("--offset-ms", type=float, default=None)
    a = ap.parse_args(argv)
    from ..take import Take
    work = Take.load(a.take).work.resolve()
    if a.cmd == "run":
        t0, t1 = (float(x) for x in a.span.split(":"))
        cmd_run(work, a.cam, t0, t1, a.step, a.group, a.overlap)
    else:
        cmd_align(work, a.cam, a.fixed, Path(a.npz), a.offset_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
