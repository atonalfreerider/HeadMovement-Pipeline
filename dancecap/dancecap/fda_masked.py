"""4DAnyone round 2 on lucas_luiza: each dancer ISOLATED with our SAM 2.1 mask, then (A) 4DAnyone's own GVHMR skeleton or
(B) OUR two-view fit as the skeleton. Owner 2026-11-05, after round 1 (docs/FDANYONE_TEST.md: the partner was erased or
merged): "why aren't we using SAM to isolate the follow and the lead and then running 4DAnyone so there is no overlap. we
also have room data for triangulation" -> "yes do E1+E2".

Inputs (pod, dancecap env; `inputs --cam 02|01`), per camera x clip (E1, E2: 121 frames from the core start, as round 1)
x dancer:
  - the dancer's own SAM 2.1 mask (lucas_luiza_h/people/masks_01.npz for the tripod, lucas_luiza_v/... for the Steadicam:
    the single-camera takes' people stage). Frame map: tripod f = round(t * 30) - 420 (the h cut; box IoU 0.89 at offset
    0), Steadicam f = source frame_index - 55 (IoU 0.83), re-searched +-4 per clip. The mask SLOT is chosen per frame by
    box IoU against the dancer's two-view teacher box (slots are colour tracks, not roles).
  - pixels outside the mask (dilated 5 px, 3 px feather) -> grey 128: the partner and the room are gone. Where the
    partner is IN FRONT of the dancer the dancer has a hole; 4DAnyone has to invent it (that is the test).
  - tripod: a 9:16 crop around THIS dancer's mask boxes over the window (round 1 cropped the couple); Steadicam: full frame.
  - GVHMR's tracker gets the mask boxes (round 1's patch).
  - ourpose.npz: the two-view fit (smplx_rf2_noslip) at each input frame (nearest grid frame on the sync-corrected clock),
    the input camera's world R, t, K, the crop, and the HELD-OUT camera's centre (the other real camera, same instants).
  - heldout/<run>.npz: the held-out camera's real frames of this dancer, masked, on a 256 x 512 canvas (for B's score).
Runs (pod, 4DAnyone venv; remote/cand_fda/fda_pod.py pipeline): per input, A = the masked clip as is (6 views, default
ring); then `inject` builds B's motion from A's (same clip identity, ViTPose 2D kept) with OUR fit's pose (10 betas fitted
to the fit's joints), our camera (incam = R X + t) and our K, and aims B's view 0 at the held-out camera (start_yaw and
pitch solved with 4DAnyone's own ring formula in its canonical frame); B reuses that motion (no GVHMR) and generates.
Score: A as round 1 (pose vs B0 / hmr_test method fdam, people per generated frame, partner); B: view 0 against the real
held-out camera (LPIPS, colour-matched PSNR, inside the real dancer mask, frames where the two directions are < 20 deg).

    python -m dancecap.fda_masked job                     # local: two pod jobs (one per camera), run with --parallel
    python -m dancecap.fda_masked inputs --cam 02         # pod
    python -m dancecap.fda_masked score --cam 02          # pod
    python -m dancecap.fda_masked merge                   # local: both cameras' results -> score_m/sheet.md (+ fusion)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .fda_test import (FPS_NUM, L, OUT_H, OUT_W, PAD, TAKE, WORK, agg_views, fill_boxes, k_orig, log, umeyama,
                       views2d, windows)
from .hmr_test import NJ, ROLES, grid_index_of_cut, interp_grid, pa_mpjpe, pinhole, to_world

CLIPS_M = ("E1", "E2")
MASK_TAKE = {"02": "lucas_luiza_h", "01": "lucas_luiza_v"}
MASK_OFF = {"02": 0, "01": -55}
OTHER = {"02": "01", "01": "02"}
CANVAS = (256, 512)   # w, h
DILATE, FEATHER = 5, 3
HELD_MAX_DEG = 20.0


def od() -> Path:
    return WORK / TAKE / "fda_test"


def rname(cam: str, clip: str, role: str) -> str:
    return f"m{cam}_{clip}_{role}"


def mask_base(cam: str, c: dict, rows: np.ndarray) -> np.ndarray:
    return (np.round(c["t"][rows] * 30).astype(int) - 420) if cam == "02" else (c["frame_index"][rows].astype(int))


def box_iou(a, b):
    x0, y0 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    x1, y1 = np.minimum(a[..., 2], b[..., 2]), np.minimum(a[..., 3], b[..., 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    u = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - i
    with np.errstate(invalid="ignore", divide="ignore"):
        return i / u


def role_masks(cam: str, rows: np.ndarray, M, T: dict, c: dict) -> tuple[dict, dict]:
    """{role: (N, H, W) bool masks or None per frame}, info. Offset searched around MASK_OFF, slot by box IoU per frame."""
    import cv2
    base = mask_base(cam, c, rows)
    B = M["box"]
    ls = T["lead_slot"][rows].astype(int)
    tb = {"lead": T["box"][rows, ls], "follow": T["box"][rows, 1 - ls]}
    best = None
    for off in range(MASK_OFF[cam] - 4, MASK_OFF[cam] + 5):
        f = base + off
        if f.min() < 0 or f.max() >= len(B):
            continue
        s = np.nanmean([np.nanmax([box_iou(tb[r], B[f, o]) for o in (0, 1)], 0) for r in ROLES])
        if best is None or s > best[0]:
            best = (float(s), off)
    off = best[1]
    f = base + off
    out, info = {}, {"offset": off, "iou_mean": best[0]}
    # one joint assignment per frame: lead and follow never share a mask (each role alone picked the same slot on 6-38 %
    # of frames in close holds, 2026-11-12: the partner's pixels went into the dancer's clip)
    iou = {r: np.nan_to_num(np.stack([box_iou(tb[r], B[f, o]) for o in (0, 1)], 1), nan=0.0) for r in ROLES}
    straight = iou["lead"][:, 0] + iou["follow"][:, 1] >= iou["lead"][:, 1] + iou["follow"][:, 0]
    slots = {"lead": np.where(straight, 0, 1), "follow": np.where(straight, 1, 0)}
    info["assignment"] = "joint per frame (2x2, max summed box IoU)"
    for r in ROLES:
        ious = iou[r]
        slot = slots[r]
        ms, n_ok = [], 0
        for i in range(len(rows)):
            k = f"m_{f[i]}_{slot[i]}"
            if k in M.files and np.nan_to_num(ious[i, slot[i]]) >= 0.3:
                ms.append(cv2.imdecode(M[k], cv2.IMREAD_GRAYSCALE) > 127)
                n_ok += 1
            else:
                ms.append(None)
        # sanity: SAM sometimes returns a mask that spills over the room (lucas_luiza_h H3 row 5211: the lead's mask covers
        # the whole frame). Keep a mask only when its area is 0.35-1.8 x the role's median in this window and its own
        # tight box overlaps the dancer's teacher box (IoU >= 0.3); the rest are filled from the nearest kept frame.
        areas = np.array([m.sum() if m is not None else np.nan for m in ms], float)
        med = np.nanmedian(areas) if np.isfinite(areas).any() else np.nan
        n_bad = 0
        for i, m in enumerate(ms):
            if m is None:
                continue
            ok = 0.35 * med <= areas[i] <= 1.8 * med and np.nan_to_num(box_iou(tb[r][i], mask_box(m))) >= 0.3
            if not ok:
                ms[i] = None
                n_bad += 1
        out[r] = ms
        info[r] = {"frames_with_mask": n_ok - n_bad, "rejected_masks": n_bad, "slot0_share": float((slot == 0).mean())}
    return out, info


def fill_masks(ms: list) -> list:
    """Missing masks -> the nearest frame's mask (reported)."""
    idx = [i for i, m in enumerate(ms) if m is not None]
    if not idx:
        raise SystemExit("no mask in the window")
    return [ms[i] if ms[i] is not None else ms[min(idx, key=lambda j: abs(j - i))] for i in range(len(ms))]


def soft(m: np.ndarray) -> np.ndarray:
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1))
    d = cv2.dilate(m.astype(np.uint8), k).astype(np.float32)
    return cv2.GaussianBlur(d, (2 * FEATHER + 1, 2 * FEATHER + 1), 0)


def mask_box(m: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(m)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float) if len(xs) else np.full(4, np.nan)


def canvas_crop(img: np.ndarray, m: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Box (5 % margin) letterboxed into CANVAS: (rgb with black outside the mask, mask)."""
    import cv2
    W, H = CANVAS
    x0, y0, x1, y1 = box
    mx, my = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    x0, y0, x1, y1 = x0 - mx, y0 - my, x1 + mx, y1 + my
    s = min(W / (x1 - x0), H / (y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    A = np.array([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]], np.float32)
    rgb = cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
    mm = cv2.warpAffine(m.astype(np.uint8) * 255, A, (W, H), flags=cv2.INTER_LINEAR) > 127
    rgb[~mm] = 0
    return rgb, mm


def write_held(path: Path, om: list, oframes: list, orows: np.ndarray, otau: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """The held-out camera's real frames of one dancer (masked, letterboxed on the canvas) + the mask's box in that
    camera's pixels (the LHM++ baseline crops its render at the real camera with the same box)."""
    from .jl_fitcheck import frame_map
    j, _ = frame_map(tau, otau, max_dt=0.05)
    crops = np.zeros((len(orows), CANVAS[1], CANVAS[0], 3), np.uint8)
    cm = np.zeros((len(orows), CANVAS[1], CANVAS[0]), bool)
    okh = np.zeros(len(orows), bool)
    boxes = np.full((len(orows), 4), np.nan)
    for i in range(len(orows)):
        if om[i] is None or om[i].sum() < 2000:
            continue
        boxes[i] = mask_box(om[i])
        crops[i], cm[i] = canvas_crop(oframes[i], om[i], boxes[i])
        okh[i] = True
    np.savez_compressed(path, crops=crops, masks=cm, ok=okh, otau=otau, tau=tau, j=j, boxes=boxes, rows=orows)
    return okh


def cmd_inputs(args) -> None:
    import cv2
    from . import mamma_dyn as md
    from .jl_fitcheck import frame_map
    cam = args.cam
    ll = WORK / TAKE
    Wn = windows()
    offs = md.read_sync_offsets(ll)
    sim3 = json.loads((ll / "cameras" / "world_from_sfm.json").read_text())
    cams = {k: dict(np.load(ll / "cameras" / f"{k}.npz")) for k in ("01", "02")}
    Rw = {k: md.world_cameras(cams[k]["R"], cams[k]["tvec"], sim3) for k in cams}
    Ts = {k: dict(np.load(ll / "people" / f"teachers_{k}.npz")) for k in cams}
    Ms = {k: np.load(WORK / MASK_TAKE[k] / "people" / "masks_01.npz") for k in cams}
    fit = {r: dict(np.load(ll / "smplx_rf2_noslip" / f"{r}.npz", allow_pickle=True)) for r in ROLES}
    tg = np.asarray(fit["lead"]["t"], float)
    root, held = od() / "inputs_m", od() / "heldout"
    root.mkdir(parents=True, exist_ok=True)
    held.mkdir(parents=True, exist_ok=True)
    rep = {}
    for clip in CLIPS_M:
        rows = np.array(Wn[cam][clip])
        frames = [cv2.imread(str(od() / "frames" / cam / f"{r:06d}.jpg")) for r in rows]
        H, Wd = frames[0].shape[:2]
        masks, info = role_masks(cam, rows, Ms[cam], Ts[cam], cams[cam])
        # held-out camera: same clip window rows of the other camera, its masks
        oc = OTHER[cam]
        orows = np.array(Wn[oc][clip])
        oframes = [cv2.imread(str(od() / "frames" / oc / f"{r:06d}.jpg")) for r in orows]
        omasks, oinfo = role_masks(oc, orows, Ms[oc], Ts[oc], cams[oc])
        tau = md.apply_offset(cams[cam]["t"][rows], offs.get(cam))
        otau = md.apply_offset(cams[oc]["t"][orows], offs.get(oc))
        gi = np.clip(np.round((tau - tg[0]) * 30).astype(int), 0, len(tg) - 1)
        for role in ROLES:
            name = rname(cam, clip, role)
            if getattr(args, "held_only", False):
                write_held(held / f"{name}.npz", omasks[role], oframes, orows, otau, tau)
                log(f"{name}: held-out crops only")
                continue
            rd = root / name
            (rd / "frames").mkdir(parents=True, exist_ok=True)
            ms = fill_masks(masks[role])
            boxes = np.stack([mask_box(m) for m in ms])
            boxes, nfill = fill_boxes(boxes)
            if cam == "02":   # 9:16 static crop around THIS dancer over the window
                x0, y0, x1, y1 = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
                cw, ch = (x1 - x0) * (1 + 2 * PAD), (y1 - y0) * (1 + 2 * PAD)
                ch = min(max(ch, cw * 16 / 9, H * 0.6), 2.0 * H)
                cw = ch * 9 / 16
                ox = (x0 + x1) / 2 - cw / 2
                oy = float(np.clip((y0 + y1) / 2 - ch / 2, 0.0, H - ch)) if ch <= H else (H - ch) / 2
            else:
                ox, oy, cw, ch = 0.0, 0.0, float(Wd), float(H)
            s = OUT_H / ch
            A = np.array([[s, 0, -ox * s], [0, s, -oy * s]], np.float32)
            for i, (fr, m) in enumerate(zip(frames, ms)):
                a = soft(m)[..., None]
                iso = (fr.astype(np.float32) * a + 128.0 * (1 - a)).astype(np.uint8)
                out = cv2.warpAffine(iso, A, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
                cv2.imwrite(str(rd / "frames" / f"{i:04d}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 97])
            num, den = FPS_NUM[cam]
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{num}/{den}", "-i",
                            str(rd / "frames" / "%04d.jpg"), "-c:v", "libx264", "-crf", "12", "-preset", "fast",
                            "-pix_fmt", "yuv420p", str(rd / "clip.mp4")], check=True)
            subprocess.run(["rm", "-rf", str(rd / "frames")], check=True)
            bt = np.stack([(boxes[:, 0] - ox) * s, (boxes[:, 1] - oy) * s, (boxes[:, 2] - ox) * s, (boxes[:, 3] - oy) * s], 1)
            bt[:, [0, 2]] = bt[:, [0, 2]].clip(0, OUT_W - 1)
            bt[:, [1, 3]] = bt[:, [1, 3]].clip(0, OUT_H - 1)
            np.save(rd / f"boxes_{role}.npy", bt.astype(np.float32))
            meta = {"cam": cam, "clip": clip, "role": role, "variant": "masked", "rows": rows.tolist(),
                    "crop": {"ox": ox, "oy": oy, "w": cw, "h": ch, "s": s}, "size": [OUT_W, OUT_H], "src_size": [Wd, H],
                    "fps": [num, den], "boxes_filled": nfill, "mask": {**info, "role": info[role]},
                    "held_cam": oc, "held_rows": orows.tolist(), "held_mask": {**oinfo, "role": oinfo[role]}}
            (rd / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
            # our two-view fit on this clip's instants + cameras
            F = fit[role]
            Rc, tc = Rw[cam][0][rows], Rw[cam][1][rows]
            orw, otw = Rw[oc][0][orows], Rw[oc][1][orows]
            held_c = -np.einsum("fji,fj->fi", orw, otw)
            # held-out centre at the input instants (nearest held-out row in corrected time)
            j, _ = frame_map(tau, otau, max_dt=0.05)
            np.savez_compressed(
                rd / "ourpose.npz", global_orient=F["global_orient"][gi], body_pose=F["body_pose"][gi],
                transl=F["transl"][gi], joints=F["joints"][gi, :NJ], betas=F["betas"], R=Rc, t=tc,
                K=cams[cam]["K"][rows], crop=np.array([s, ox, oy]), held_centre=held_c[j], grid_index=gi,
                dt_ms=(tau - tg[gi]) * 1000)
            okh = write_held(held / f"{name}.npz", omasks[role], oframes, orows, otau, tau)
            rep[name] = {"boxes_filled": nfill, "mask_frames": info[role]["frames_with_mask"], "held_ok": int(okh.sum()),
                         "fit_dt_ms_max": float(np.abs((tau - tg[gi]) * 1000).max())}
            log(f"{name}: crop x {ox:.0f} y {oy:.0f} {cw:.0f}x{ch:.0f}; masks {info[role]['frames_with_mask']}/121 "
                f"(offset {info['offset']}, IoU {info['iou_mean']:.2f}); held-out {oc}: {okh.sum()}/121 frames")
    (root / f"inputs_{cam}.json").write_text(json.dumps(rep, indent=1))


# ------------------------------------------------------------------------------------------------ score (pod) + merge


def cmd_score(args) -> None:
    from . import mamma_dyn as md
    from .hmr_test import Ref
    cam = args.cam
    ref = Ref(TAKE)
    ll = WORK / TAKE
    offs = md.read_sync_offsets(ll)
    c = {"01": ref.c1, "02": ref.c2}
    Rw = {"01": ref.R1, "02": ref.R2}
    rd_root, sd = od() / "runs_m", od() / "score_m"
    sd.mkdir(parents=True, exist_ok=True)
    rep = json.loads((rd_root / "run.json").read_text())
    J0 = {r: ref.B0[r]["joints"][:, :NJ].astype(float) for r in ROLES}
    res = {"cam": cam, "runs": {}, "run_status": {k: {kk: v.get(kk) for kk in ("ok", "failed", "seconds", "inject", "appearance")}
                                                  for k, v in rep["runs"].items()}}
    grid = {}
    hm = WORK / TAKE / "hmr_test" / "fdam"
    hm_runs = {}
    for clip in CLIPS_M:
        for role in ROLES:
            name = rname(cam, clip, role)
            for variant, rn in (("A", name), ("B", name + "_ours")):
                rd = rd_root / rn
                if not rep["runs"].get(rn, {}).get("ok") or not (rd / "fda.npz").exists():
                    res["runs"][rn] = {"status": "not run correctly: " + str(rep["runs"].get(rn, {}).get("failed"))}
                    continue
                meta = json.loads((od() / "inputs_m" / name / "meta.json").read_text())
                fda = dict(np.load(rd / "fda.npz"))
                rows = np.array(meta["rows"])
                tau = md.apply_offset(c[cam]["t"][rows], offs.get(cam))
                B = {r: interp_grid(J0[r], ref.tg, tau) for r in ROLES}
                Jc = fda["J_cam"]
                pa = np.array([pa_mpjpe(Jc[i], B[role][i]) if np.isfinite(B[role][i]).all() else np.nan for i in range(L)])
                Rr = Rw[cam][rows]
                Jw = to_world(Jc, Rr, np.zeros((L, 3)))
                rr_w = Jw - Jw[:, :1]
                rr = np.linalg.norm(rr_w - (B[role] - B[role][:, :1]), axis=-1).mean(-1) * 1000
                okf = np.isfinite(B[role]).all((1, 2))
                sim = umeyama(fda["J_canon"][okf].reshape(-1, 3), B[role][okf].reshape(-1, 3))
                s_, R_, t_ = sim["s"], np.array(sim["R"]), np.array(sim["t"])
                J0c = {r: ((B[r] - t_) @ R_) / s_ for r in ROLES}
                v2 = views2d(fda, dict(np.load(rd / "det.npz")), J0c, role)
                yaw_rel = np.asarray(fda["cam_yaw"], float)
                agg = agg_views(v2["rows"], np.zeros_like(yaw_rel))["all"]
                r_ = {"status": "ok", "variant": variant, "cam": cam, "clip": clip, "role": role,
                      "pa_mm": float(np.nanmedian(pa)), "rr_world_mm": float(np.nanmedian(rr)),
                      "sim3_rms_mm": sim["rms"] * 1000, "views": agg,
                      "seconds": rep["runs"][rn].get("seconds")}
                if variant == "B":
                    r_["inject"] = rep["runs"][rn].get("inject")
                    ap = rd / "appearance.json"
                    r_["appearance"] = json.loads(ap.read_text()) if ap.exists() else None
                res["runs"][rn] = r_
                grid[f"{rn}__gi"] = np.round((tau - ref.tg[0]) * 30).astype(int)
                grid[f"{rn}__rr"] = rr_w
                if variant == "A" and cam == "02":
                    meta_c = json.loads((WORK / TAKE / "hmr_test" / "clips" / clip / "meta.json").read_text())
                    n0, Lc = meta_c["n0"], meta_c["L"]
                    i_clip = grid_index_of_cut(n0 + np.arange(Lc), ref.kg0)
                    Ko = k_orig(fda["K_in"], meta["crop"])
                    J_out, K_out = np.full((Lc, NJ, 3), np.nan), np.full((Lc, 3, 3), np.nan)
                    pos = {int(g): i for i, g in enumerate(grid[f"{rn}__gi"])}
                    for q, g in enumerate(i_clip):
                        if int(g) in pos:
                            J_out[q], K_out[q] = Jc[pos[int(g)]], Ko[pos[int(g)]]
                    (hm / f"{clip}_{role}").mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(hm / f"{clip}_{role}" / "joints_cam02.npz", J_c=J_out, K=K_out)
                    hm_runs[f"{clip}_{role}"] = {"ok": True}
                log(f"{rn}: PA {r_['pa_mm']:.0f} mm; people 1 / 2 / 3+: {agg['people_1']} / {agg['people_2']} / "
                    f"{agg['people_3plus']}" + (f"; appearance {r_.get('appearance', {}) and r_['appearance'].get('summary')}"
                                                 if variant == "B" else ""))
    if hm_runs:   # the clips round 2 does not run (H1, H3) are absent: hmr_test reports per clip
        for c_ in ("H1", "H3"):
            for r in ROLES:
                hm_runs[f"{c_}_{r}"] = {"ok": True, "frames": 0, "note": "not in round 2"}
        (hm / "run.json").write_text(json.dumps({"method": "fdam", "runs": hm_runs}, indent=1))
    np.savez_compressed(sd / f"grid_{cam}.npz", J0_lead=J0["lead"], J0_follow=J0["follow"], **grid)
    (sd / f"score_{cam}.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"score: {sd / f'score_{cam}.json'}")


def _f(x, nd=0, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{100 * x:.0f} %" if pct else f"{x:.{nd}f}"


def cmd_merge(args) -> None:
    """Local: read both cameras' pulled results, the two-view fusion (arithmetic on pulled arrays), the sheet."""
    sd = od() / "score_m"
    res = {cam: json.loads((sd / f"score_{cam}.json").read_text()) for cam in ("02", "01") if (sd / f"score_{cam}.json").exists()}
    G = {cam: dict(np.load(sd / f"grid_{cam}.npz")) for cam in res}
    r1 = json.loads((od() / "score" / "score.json").read_text()) if (od() / "score" / "score.json").exists() else {"runs": {}}
    out = ["# 4DAnyone round 2: each dancer isolated with our SAM mask; A = its own skeleton, B = our two-view fit", "",
           "Round 1 (no masking) for comparison in brackets. PA = PA-MPJPE vs the two-view fit (mm). People = YOLO11x-pose "
           "persons (conf >= 0.5) per generated frame. B appearance: view 0 aimed at the held-out real camera vs that camera's "
           "real frames of the dancer, inside the real mask, frames with < 20 deg between the two directions.", "",
           "| run | variant | PA mm (round 1) | 1 person | 2 people | 3+ | LPIPS | PSNR cc dB | frames | angle deg | seconds |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    fused = {}
    for cam, R in res.items():
        for rn, v in R["runs"].items():
            if v.get("status") != "ok":
                out.append(f"| {rn} | | {v['status']} |||||||| ")
                continue
            r1n = f"c{cam}_{v['clip']}_{v['role']}"
            p1 = r1["runs"].get(r1n, {}).get("pa_mm")
            a = v["views"]
            ap = (v.get("appearance") or {}).get("summary") or {}
            out.append(f"| {rn} | {v['variant']} | {_f(v['pa_mm'])} ({_f(p1)}) | {_f(a['people_1'], pct=True)} | "
                       f"{_f(a['people_2'], pct=True)} | {_f(a['people_3plus'], pct=True)} | {_f(ap.get('lpips'), 3)} | "
                       f"{_f(ap.get('psnr_cc'), 1)} | {ap.get('n', '–')} | {_f(ap.get('angle_deg'))} | {_f(v.get('seconds'))} |")
    # two-view fusion of variant A (each camera's skeleton, root-relative world joints averaged)
    if "02" in G and "01" in G:
        from .hmr_test import pa_mpjpe as pam
        out += ["", "## A, both cameras fused (root-relative world joints averaged; PA mm)", "",
                "| clip, dancer | tripod | Steadicam | fused |", "|---|---|---|---|"]
        for clip in CLIPS_M:
            for role in ROLES:
                a, b = f"m02_{clip}_{role}", f"m01_{clip}_{role}"
                if f"{a}__gi" not in G["02"] or f"{b}__gi" not in G["01"]:
                    continue
                ga, gb = G["02"][f"{a}__gi"], G["01"][f"{b}__gi"]
                ia, ib = {int(g): i for i, g in enumerate(ga)}, {int(g): i for i, g in enumerate(gb)}
                J0 = G["02"][f"J0_{role}"]
                e = {"02": [], "01": [], "f": []}
                for g in sorted(set(ia) & set(ib)):
                    if not 0 <= g < len(J0):
                        continue
                    y = J0[g] - J0[g][:1]
                    xa, xb = G["02"][f"{a}__rr"][ia[g]], G["01"][f"{b}__rr"][ib[g]]
                    for k, x in (("02", xa), ("01", xb), ("f", (xa + xb) / 2)):
                        e[k].append(pam(x, y))
                fused[f"{clip}_{role}"] = {k: float(np.median(v)) for k, v in e.items() if v}
                out.append(f"| {clip} {role} | {_f(fused[f'{clip}_{role}'].get('02'))} | "
                           f"{_f(fused[f'{clip}_{role}'].get('01'))} | {_f(fused[f'{clip}_{role}'].get('f'))} |")
    (sd / "sheet.md").write_text("\n".join(out) + "\n")
    (sd / "merged.json").write_text(json.dumps({"cams": res, "fused": fused}, indent=1, default=float))
    print("\n".join(out))


# ------------------------------------------------------------------------------------------------ job (local)


def cmd_job(args) -> None:
    base = json.loads((WORK / TAKE / "jobs" / "pod" / "lucas_luiza_fda.json").read_text())
    keep = [x for x in base["inputs"] if not (isinstance(x, dict) and "/fda_test/frames/" in x["pod"])]
    Wn = windows()
    frames = []
    for cam in ("02", "01"):
        rows = sorted({r for c in CLIPS_M for r in Wn[cam][c]})
        frames += [{"local": str(WORK / TAKE / "frames" / cam / f"{r:06d}.jpg"),
                    "pod": f"dancecap/work/{TAKE}/fda_test/frames/{cam}/{r:06d}.jpg"} for r in rows]
    masks = [f"dancecap/work/{MASK_TAKE[k]}/people/masks_01.npz" for k in ("02", "01")]
    py, pod = "/workspace/fda/venv/bin/python", "remote/cand_fda/fda_pod.py"
    I, R, Hd = f"work/{TAKE}/fda_test/inputs_m", f"work/{TAKE}/fda_test/runs_m", f"work/{TAKE}/fda_test/heldout"
    paths = []
    for cam in ("02", "01"):
        names = [rname(cam, c, r) for c in CLIPS_M for r in ROLES]
        cmds = [
            {"name": "install", "timeout_min": 30, "cmd": "bash remote/cand_fda/install.sh"},
            {"name": "inputs", "timeout_min": 20, "cmd": f"python -m dancecap.fda_masked inputs --cam {cam}"},
            {"name": "bodymodels", "timeout_min": 2, "cmd": f"{py} {pod} place-models"},
            {"name": "pipeline", "timeout_min": args.run_minutes,
             "cmd": f"{py} {pod} pipeline --inputs {I} --out {R} --only {','.join(names)} --views 6 "
                    f"--budget-min {args.run_minutes - 4}"},
            {"name": "extract", "timeout_min": 10, "cmd": f"{py} {pod} extract --inputs {I} --out {R}"},
            {"name": "detect", "timeout_min": 10, "cmd": f"{py} {pod} detect --out {R} --every 4"},
            {"name": "appearance", "timeout_min": 10, "cmd": f"{py} {pod} appearance --inputs {I} --out {R} --heldout {Hd}"},
            {"name": "score", "timeout_min": 15, "cmd": f"python -m dancecap.fda_masked score --cam {cam}"},
        ]
        if cam == "02":
            cmds += [{"name": "hmr_score", "timeout_min": 15, "cmd": f"python -m dancecap.hmr_test score --take {TAKE} "
                      "--methods B0,B1,B2,exmr,fda,fdam --out-sub fdam_score"},
                     {"name": "hmr_sheet", "timeout_min": 10,
                      "cmd": f"python -m dancecap.hmr_test sheet --take {TAKE} --out-sub fdam_score"}]
        extra = []
        if cam == "02":   # round 1's camera-02 joints, so hmr_test can put fda and fdam side by side
            extra = [f"dancecap/work/{TAKE}/hmr_test/fda/"]
        job = {**{k: base[k] for k in ("schema", "take", "cwd", "needs", "base", "models", "min_vram_gb", "exclude", "env")},
               "name": f"lucas_luiza_fdam_c{cam}", "max_minutes": args.max_minutes,
               "inputs": keep + frames + masks + extra, "commands": cmds,
               "outputs": [f"dancecap/work/{TAKE}/fda_test/runs_m/", f"dancecap/work/{TAKE}/fda_test/score_m/",
                           f"dancecap/work/{TAKE}/fda_test/inputs_m/", f"dancecap/work/{TAKE}/hmr_test/fdam/",
                           f"dancecap/work/{TAKE}/hmr_test/fdam_score/"],
               "verify": [f"dancecap/work/{TAKE}/fda_test/score_m/score_{cam}.json"],
               "notes": ["4DAnyone round 2 (dancecap.fda_masked): SAM-isolated dancers, A = its GVHMR skeleton, B = our "
                         "two-view fit injected; camera " + cam + "; same licences as round 1 (lucas_luiza_fda.json)."]}
        p = WORK / TAKE / "jobs" / "pod" / f"lucas_luiza_fdam_c{cam}.json"
        p.write_text(json.dumps(job, indent=1))
        paths.append(p)
        log(f"job {p}: {len(names)} inputs x 2 variants, {len(frames)} frames")
    print(",".join(str(p) for p in paths))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("job")
    p.add_argument("--max-minutes", type=int, default=170)
    p.add_argument("--run-minutes", type=int, default=120)
    p = sub.add_parser("inputs")
    p.add_argument("--cam", required=True, choices=["02", "01"])
    p.add_argument("--held-only", action="store_true", help="only heldout/<run>.npz (the LHM++ baseline job)")
    p = sub.add_parser("score")
    p.add_argument("--cam", required=True, choices=["02", "01"])
    sub.add_parser("merge")
    a = ap.parse_args(argv)
    {"job": cmd_job, "inputs": cmd_inputs, "score": cmd_score, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
