"""How big, how frontal and how visible are the dancers' faces in every camera frame?

    python -m dancecap.face_survey <take>        -> work/<take>/face/survey.npz, survey.json, jobs.json
                                                    review/face_survey.png, review/face_survey_crops.jpg

For every canonical frame of every camera and both dancers, the fitted SMPL-X body (smplx/<role>.npz, posed at the
frame's reference time like the texture bake, head fix applied when adopted) gives the 52 iBUG face landmarks
(face_common.LandmarkEmbedding). Projected into the frame they give:
    iod_px     inter-ocular distance (outer eye corners 36-45), the standard face-size measure
    face_px    landmark box height (brow to chin ~ 0.7 x face height)
    angle_deg  angle between the face's forward axis and the direction to the camera (0 = frontal, 90 = profile)
    vis        fraction of landmarks not hidden by the partner's body or the dancer's own arms/hands/torso (coarse
               vertex z-buffer, 3 cm tolerance; the own head is no occluder, profile views are the angle test)
    partner    fraction of landmarks inside the PARTNER's SAM mask (occluded by the partner)
    hair       fraction inside the follow's hair mask (dancers/<cam>/hair; follow only)
    own        fraction inside the dancer's own SAM mask (sanity check of fit + camera; low = misregistered)
Usable tiers: a view is usable when the camera frame is ok, landmarks are inside the image, angle < 60 deg,
vis >= 0.8, partner < 0.2, hair < 0.3, own >= 0.6 and iod >= tier (TIERS px).
jobs.json lists the crops the detector worker (face_worker.py, run in .venv-face) processes: every frame with
iod >= 5 px, angle < 80 deg, inside the image.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .face_common import (LABEL, PIDS, ROLES, LandmarkEmbedding, cameras_world, crop_pad, frames_meta, head_frame,
                          label, project, smplx_model_npz, take_cams)
from .take import Take

TIERS = (10, 20, 30)          # iod px: rough landmarks / landmarks OK / expressions plausible
JOB_MIN_IOD = 5.0
OCC_TOL_M = 0.03


def _mask(p: Path) -> np.ndarray | None:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def _frac_in(mask: np.ndarray | None, px: np.ndarray) -> float:
    if mask is None:
        return 0.0
    H, W = mask.shape
    u, v = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not ok.any():
        return 0.0
    return float(mask[v[ok], u[ok]].mean())


def _zbuffer_vis(verts_all: np.ndarray, lmk: np.ndarray, K, R, t, k1, W, H, cell: float) -> np.ndarray:
    """Per landmark: not behind another surface (vertex splat z-buffer in cell-px cells, 3x3 min filter)."""
    pa, za = project(verts_all, K, R, t, k1)
    Wc, Hc = int(W / cell) + 2, int(H / cell) + 2
    ia = np.round(pa / cell).astype(int)
    ok = (ia[:, 0] >= 0) & (ia[:, 0] < Wc) & (ia[:, 1] >= 0) & (ia[:, 1] < Hc) & (za > 0.1)
    zb = np.full((Hc, Wc), 1e9, np.float32)
    np.minimum.at(zb, (ia[ok, 1], ia[ok, 0]), za[ok].astype(np.float32))
    zb = -cv2.dilate(-zb, np.ones((3, 3), np.uint8))
    pl, zl = project(lmk, K, R, t, k1)
    il = np.round(pl / cell).astype(int)
    inb = (il[:, 0] >= 0) & (il[:, 0] < Wc) & (il[:, 1] >= 0) & (il[:, 1] < Hc)
    vis = np.zeros(len(lmk), bool)
    vis[inb] = zl[inb] <= zb[il[inb, 1], il[inb, 0]] + OCC_TOL_M
    return vis


def run_survey(take: Take, cams: list[str] | None = None, out: Path | None = None, head_fix: bool = True) -> dict:
    from .avatar_texture import load_bodies
    work = take.work
    out = out or work / "face"
    out.mkdir(parents=True, exist_ok=True)
    cams = cams or take_cams(work)
    t0 = time.time()
    bodies, binfo = load_bodies(work, device="cpu", head_fix=head_fix)
    emb = LandmarkEmbedding()
    m = np.load(smplx_model_npz(), allow_pickle=True)
    lmk_rest = emb.points(np.asarray(m["v_template"], float))
    from .avatar_uv import vertex_regions
    not_head = ~vertex_regions(m)["head"]   # own head never occludes its own landmarks (profile = angle test)
    camw = cameras_world(work, cams)
    i36, i45 = emb.index([36, 45])
    rec: dict = {}
    jobs = []
    for c in cams:
        meta = frames_meta(work, c)
        W, H = int(meta["width"]), int(meta["height"])
        ci = camw[c]
        for r in ROLES:
            rows = []
            other = [o for o in ROLES if o != r][0]
            for i, fr in enumerate(meta["frames"]):
                t = float(fr["t"])
                a = bodies[r].at(t)
                if a is None or i >= len(ci["K"]):
                    continue
                V = a[0].astype(np.float64)
                Vo = bodies[other].at(t)[0].astype(np.float64)
                K, R, tt, k1 = ci["K"][i], ci["R"][i], ci["t"][i], float(ci["k1"][i])
                lmk = emb.points(V)
                px, z = project(lmk, K, R, tt, k1)
                iod = float(np.linalg.norm(px[i36] - px[i45]))
                Rh, ctr = head_frame(lmk_rest, lmk)
                C = -R.T @ tt
                d = Rh.T @ ((C - ctr) / np.linalg.norm(C - ctr))
                angle = float(np.degrees(np.arccos(np.clip(d[2], -1, 1))))
                yaw = float(np.degrees(np.arctan2(d[0], d[2])))
                pitch = float(np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2]))))
                inimg = bool(((px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)).all() and (z > 0.1).all())
                cell = float(np.clip(iod / 3, 2, 8))
                vis = float(_zbuffer_vis(np.concatenate([V[not_head], Vo]), lmk, K, R, tt, k1, W, H, cell).mean()) if inimg else 0.0
                dd = work / "dancers" / c
                full_o = _mask(dd / "masks" / f"mask_{i:04d}_{PIDS[other]:02d}.png")
                full_s = _mask(dd / "masks" / f"mask_{i:04d}_{PIDS[r]:02d}.png")
                hair = _mask(dd / "hair" / f"mask_{i:04d}_02.png") if r == "follow" else None
                partner, own = _frac_in(full_o, px), _frac_in(full_s, px)
                hairf = _frac_in(hair, px)
                lo, hi = px.min(0), px.max(0)
                face_px = float(hi[1] - lo[1])
                rows.append(dict(i=i, t=t, iod=iod, face_px=face_px, angle=angle, yaw=yaw, pitch=pitch, vis=vis,
                                 partner=partner, hair=hairf, own=own, inimg=inimg, ok=bool(ci["ok"][i]),
                                 cx=float(px[:, 0].mean()), cy=float(px[:, 1].mean()), lmk=px.astype(np.float32)))
                if inimg and iod >= JOB_MIN_IOD and angle < 80:
                    side = max(hi - lo) * 2.2
                    box_face = [float((lo[0] + hi[0]) / 2 - side / 2), float((lo[1] + hi[1]) / 2 - side / 2)]
                    box_face += [box_face[0] + side, box_face[1] + side]
                    # upper-body box for top-down whole-body models (RTMW): face in the top third, ~25 % of the width
                    bs = side * 2.2
                    cxb, cyb = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2 + 0.30 * bs
                    box_body = [cxb - bs * 0.5, cyb - bs * 0.6, cxb + bs * 0.5, cyb + bs * 0.6]
                    jobs.append({"role": r, "cam": c, "i": i, "t": t, "iod": round(iod, 2), "angle": round(angle, 1),
                                 "box_face": [round(v, 1) for v in box_face], "box_body": [round(v, 1) for v in box_body],
                                 "lmk_model": np.round(px, 1).tolist()})
            if not rows:
                continue
            rec[(r, c)] = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    # usable flags
    summary = {"take": take.name, "cams": cams, "tiers_iod_px": list(TIERS), "bodies": binfo, "roles": {}}
    arrays = {}
    for (r, c), a in rec.items():
        base = a["ok"] & a["inimg"] & (a["angle"] < 60) & (a["vis"] >= 0.8) & (a["partner"] < 0.2) & \
            (a["hair"] < 0.3) & (a["own"] >= 0.6)
        a["base"] = base
        for k, v in a.items():
            arrays[f"{r}_{c}_{k}"] = v
    grid_t = np.asarray(bodies["lead"].t, float)
    for r in ROLES:
        rs = {"cams": {}, "coverage": {}}
        per_cam_usable = {}
        for c in cams:
            a = rec.get((r, c))
            if a is None:
                continue
            st = {"frames": int(len(a["t"])), "iod_median": round(float(np.median(a["iod"])), 1),
                  "iod_p90": round(float(np.percentile(a["iod"], 90)), 1),
                  "iod_max": round(float(a["iod"].max()), 1),
                  "frontal_lt60": round(float((a["angle"] < 60).mean()), 3),
                  "visible": round(float((a["vis"] >= 0.8).mean()), 3),
                  "partner_occluded": round(float((a["partner"] >= 0.2).mean()), 3),
                  "hair_occluded": round(float((a["hair"] >= 0.3).mean()), 3),
                  "own_mask_median": round(float(np.median(a["own"])), 3),
                  "usable_base": round(float(a["base"].mean()), 3)}
            for tier in TIERS:
                st[f"usable_iod{tier}"] = round(float((a["base"] & (a["iod"] >= tier)).mean()), 3)
            rs["cams"][c] = st
            # map the camera's frames onto the 30 Hz grid (nearest within 20 ms)
            k = np.clip(np.searchsorted(grid_t, a["t"]), 1, len(grid_t) - 1)
            k = np.where(np.abs(grid_t[k - 1] - a["t"]) < np.abs(grid_t[k] - a["t"]), k - 1, k)
            near = np.abs(grid_t[k] - a["t"]) < 0.02
            per_cam_usable[c] = {tier: np.zeros(len(grid_t), bool) for tier in TIERS}
            for tier in TIERS:
                u = a["base"] & (a["iod"] >= tier) & near
                per_cam_usable[c][tier][k[u]] = True
        for tier in TIERS:
            n = np.sum([per_cam_usable[c][tier] for c in per_cam_usable], 0) if per_cam_usable else np.zeros(len(grid_t))
            arrays[f"{r}_grid_nviews_iod{tier}"] = n
            rs["coverage"][f"iod{tier}"] = {"grid_frames_ge1_view": round(float((n >= 1).mean()), 3),
                                            "grid_frames_ge2_views": round(float((n >= 2).mean()), 3),
                                            "longest_gap_s": round(_longest_gap(n >= 1, grid_t), 2),
                                            "cams_contributing": sorted(c for c in per_cam_usable
                                                                        if per_cam_usable[c][tier].any())}
        summary["roles"][r] = rs
    arrays["grid_t"] = grid_t
    np.savez_compressed(out / "survey.npz", **arrays)
    summary["jobs"] = len(jobs)
    summary["runtime_s"] = round(time.time() - t0, 1)
    (out / "survey.json").write_text(json.dumps(summary, indent=1))
    (out / "jobs.json").write_text(json.dumps({"take": take.name, "jobs": jobs}))
    review = work / "review"
    review.mkdir(parents=True, exist_ok=True)
    summary["review"] = [str(_plot(rec, arrays, cams, review / "face_survey.png", take.name)),
                         str(_crops_sheet(work, rec, cams, review / "face_survey_crops.jpg", take.name))]
    (out / "survey.json").write_text(json.dumps(summary, indent=1))
    return summary


def _longest_gap(ok: np.ndarray, t: np.ndarray) -> float:
    best, start = 0.0, None
    for k, v in enumerate(ok):
        if not v and start is None:
            start = k
        if (v or k == len(ok) - 1) and start is not None:
            end = k if v else k + 1
            best = max(best, (end - start) / 30.0)
            start = None
    return best


def _plot(rec, arrays, cams, path: Path, take: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5), gridspec_kw={"height_ratios": [3, 1.2]})
    t_grid = arrays["grid_t"]
    for col, r in enumerate(ROLES):
        ax = axes[0, col]
        for n, c in enumerate(cams):
            a = rec.get((r, c))
            if a is None:
                continue
            good = a["base"]
            sc = ax.scatter(a["t"][good], np.full(good.sum(), n), c=np.clip(a["iod"][good], 0, 50), cmap="viridis",
                            vmin=0, vmax=50, s=10, marker="s")
            ax.scatter(a["t"][~good], np.full((~good).sum(), n), c="#cccccc", s=4, marker="s")
            ax.text(t_grid[-1] + 0.05, n, f"med {np.median(a['iod']):.0f}px  usable {a['base'].mean() * 100:.0f}%",
                    va="center", fontsize=8)
        ax.set_yticks(range(len(cams)))
        ax.set_yticklabels([f"cam {c}" for c in cams])
        ax.set_title(f"{LABEL[r]}: inter-ocular distance (px) per frame; grey = not usable "
                     f"(profile/occluded/hair/outside)", fontsize=9)
        ax.set_xlim(t_grid[0], t_grid[-1] + 1.3)
        ax.invert_yaxis()
        fig.colorbar(sc, ax=ax, fraction=0.03, label="IOD px")
        ax2 = axes[1, col]
        for tier, colr in zip((10, 20, 30), ("#9ecae1", "#3182bd", "#08306b")):
            ax2.plot(t_grid, arrays[f"{r}_grid_nviews_iod{tier}"], color=colr, lw=1.2, label=f"usable views, IOD >= {tier}px")
        ax2.set_ylabel("# cameras")
        ax2.set_xlabel("reference time (s)")
        ax2.legend(fontsize=7, loc="upper right")
        ax2.set_xlim(t_grid[0], t_grid[-1] + 1.3)
    fig.suptitle(f"{take}: face size and visibility per camera (from the fitted SMPL-X face landmarks)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _crops_sheet(work: Path, rec, cams, path: Path, take: str, tile: int = 160) -> Path:
    """Per dancer and camera: the largest usable face and a typical (median-IOD) usable face, native pixels upscaled."""
    rows = []
    for r in ROLES:
        tiles = []
        for c in cams:
            a = rec.get((r, c))
            if a is None:
                continue
            idx = np.flatnonzero(a["base"])
            if not len(idx):
                idx = np.flatnonzero(a["inimg"])
            if not len(idx):
                continue
            best = idx[np.argmax(a["iod"][idx])]
            med = idx[np.argsort(a["iod"][idx])[len(idx) // 2]]
            for k, tag in ((best, "max"), (med, "median")):
                img = cv2.imread(str(work / "frames" / c / f"{int(a['i'][k]):06d}.jpg"))
                lm = a["lmk"][k]
                lo, hi = lm.min(0), lm.max(0)
                side = max(hi - lo) * 2.2
                cx, cy = (lo + hi) / 2
                box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
                cr = crop_pad(img, box, tile, cv2.INTER_NEAREST)
                label(cr, f"cam {c} {tag}", (4, 14), scale=0.42)
                label(cr, f"IOD {a['iod'][k]:.0f}px {a['angle'][k]:.0f}deg", (4, tile - 6), scale=0.42)
                tiles.append(cr)
        if tiles:
            row = np.concatenate(tiles, 1)
            head = np.full((22, row.shape[1], 3), 20, np.uint8)
            label(head, LABEL[r], (6, 16), scale=0.5)
            rows.append(np.concatenate([head, row], 0))
    w = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows], 0)
    head = np.full((30, w, 3), 10, np.uint8)
    label(head, f"{take}: faces as the cameras see them (native pixels, nearest-neighbour upscale): largest and "
                f"median usable view per camera", (6, 20), scale=0.5)
    cv2.imwrite(str(path), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--cams", nargs="*")
    ap.add_argument("--no-head-fix", action="store_true")
    a = ap.parse_args(argv)
    s = run_survey(Take.load(a.take), a.cams, head_fix=not a.no_head_fix)
    for r, rs in s["roles"].items():
        print(r, json.dumps(rs["coverage"]))
        for c, st in rs["cams"].items():
            print(f"  cam {c}: iod med {st['iod_median']} p90 {st['iod_p90']} max {st['iod_max']} | usable base "
                  f"{st['usable_base']} iod>=10 {st['usable_iod10']} >=20 {st['usable_iod20']} >=30 {st['usable_iod30']}"
                  f" | hair {st['hair_occluded']} partner {st['partner_occluded']} own {st['own_mask_median']}")
    print("review:", s["review"], "jobs:", s["jobs"], "runtime", s["runtime_s"], "s")


if __name__ == "__main__":
    sys.exit(main())
