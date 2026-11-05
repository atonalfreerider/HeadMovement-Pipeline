"""lucas_luiza: 4DAnyone (one video -> generated multiview videos of one person) on a couple, monocular and two-view.

Owner, 2026-11-05: "https://4danyone.github.io/ did we test this approach for monocular or two camera videos? if not -
run a test on runpod for monocular and two view dances". It had not been tested (2026-11-05: no trace in the repo).

4DAnyone (ant-research/4DAnyone @ 6dcf626; SIGGRAPH Asia 2026): GVHMR recovers ONE person's SMPL-X motion from the video
(static camera assumed), a ring of target cameras is placed around that person, and a Wan2.2-based video model generates
every target view conditioned on the source video and the person's skeleton rendered in that view. The views then feed a
4DGS. It takes ONE video of ONE person: there is no two-video input, and nothing in it models a partner.

The test (all on the pod; scores against the two-view fit B0 = lucas_luiza/smplx_rf2_noslip):
  runs     4 clips (hmr_test's E1 closed hold, E2 open spins, H1 embrace / occlusion, H3 fast spins), 121 frames (4DAnyone's
           clip length) from each clip's core start, x 2 cameras (02 tripod, 01 Steadicam) x 2 dancers = 16 runs. Each
           run is the whole frame (both dancers visible) with GVHMR's tracker replaced by THAT dancer's boxes from our
           people stage (remote/cand_fda/fda_pod.py), 12 generated views on 4DAnyone's default 15-degree ring.
  monocular  camera 02 runs = the tripod-only route: pose = GVHMR's skeleton that 4DAnyone generates from, scored by
           dancecap.hmr_test as method "fda" (held-out Steadicam M1-B, PA-MPJPE vs B0, ...) next to B1 (our mono fit),
           B2 (SAM 3D Body) and ExMR; and the generated views (below).
  two-view   4DAnyone has no two-video mode, so the two-view use is: each camera through 4DAnyone on its own, then
           (a) the two skeletons fused (root-relative joints rotated to the world by each camera's known rotation,
           averaged) vs either alone, against B0; (b) each camera's generated views checked against the real couple
           (B0, which both real cameras built). The other real camera is the held-out view for each single run.
  views2d  every run: B0's lead and follow mapped into 4DAnyone's canonical frame (Sim3 from the run's own skeleton to
           B0's dancer over the clip), projected into each generated camera, and matched to YOLO11x-pose people in the
           generated frames: PCK@0.1 of the run's dancer and of the PARTNER, partner found, people count. The same
           match against the run's own skeleton = does the generator follow its conditioning.

    python -m dancecap.fda_test job                      # local: inputs list + work/lucas_luiza/jobs/pod/lucas_luiza_fda.json
    python -m dancecap.fda_test inputs                   # pod: input clips + boxes -> work/lucas_luiza/fda_test/inputs/
    (remote/cand_fda/fda_pod.py run / extract / detect)  # pod, 4DAnyone venv -> work/lucas_luiza/fda_test/runs/
    python -m dancecap.fda_test score                    # pod: -> work/lucas_luiza/fda_test/score/{score.json, sheet.md, *.jpg}

Outputs are likeness-derived: work/ (git-ignored), never committed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .hmr_test import (CLIPS, CLIP_NOTE, COCO, NJ, ROLES, SMX, WORK, grid_index_of_cut, interp_grid, pa_mpjpe, pinhole,
                       to_world)

TAKE = "lucas_luiza"
CAMS = ("02", "01")
CAM_NOTE = {"02": "tripod (pans)", "01": "Steadicam (moves)"}
L = 121
OUT_W, OUT_H = 1080, 1920          # 9:16 input (4DAnyone: 1080p or higher, portrait)
PAD = 0.12                         # couple box margin, each side
FPS_NUM = {"02": (30000, 1001), "01": (29, 1)}
PCK_T = 0.1
FOUND_T = 0.25                     # a person "is" a B0 dancer when its median joint error < 25 % of the dancer's height
JOB = "lucas_luiza_fda"


def od() -> Path:
    return WORK / TAKE / "fda_test"


def log(msg: str) -> None:
    print(f"[fda_test {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_name(cam: str, clip: str, role: str) -> str:
    return f"c{cam}_{clip}_{role}"


# ------------------------------------------------------------------------------------------------ windows (job + inputs)


def windows() -> dict:
    """{cam: {clip: rows}}: 121 consecutive camera rows from the camera row of the clip's first core grid frame."""
    from . import mamma_dyn as md
    from .jl_fitcheck import frame_map
    ll = WORK / TAKE
    offs = md.read_sync_offsets(ll)
    out = {}
    for cam in CAMS:
        t = np.load(ll / "cameras" / f"{cam}.npz")["t"]
        out[cam] = {}
        for c, (k0, _k1) in CLIPS.items():
            r, ok = frame_map(md.offset_label(np.array([k0 / 30.0]), offs.get(cam)), t, max_dt=0.04)
            if not ok[0]:
                raise SystemExit(f"camera {cam}: no row near clip {c} start")
            r0 = int(r[0])
            out[cam][c] = list(range(r0, r0 + L))
    return out


def cmd_job(args) -> None:
    W = windows()
    ins = []
    for cam in CAMS:
        rows = sorted({r for c in W[cam] for r in W[cam][c]})
        ins += [{"local": str(WORK / TAKE / "frames" / cam / f"{r:06d}.jpg"),
                 "pod": f"dancecap/work/{TAKE}/fda_test/frames/{cam}/{r:06d}.jpg"} for r in rows]
    base = json.loads((WORK / TAKE / "jobs" / "pod" / "lucas_luiza_hmr_score.json").read_text())["inputs"]
    ref_inputs = [x for x in base if isinstance(x, str)] + [x for x in base if isinstance(x, dict)]
    clips = [f"dancecap/work/{TAKE}/hmr_test/clips/"]   # meta / boxes / cam02 / the sheet's camera-02 frames
    exmr = [f"dancecap/work/{TAKE}/hmr_test/exmr/run.json"] + [
        f"dancecap/work/{TAKE}/hmr_test/exmr/{c}_{r}/results.npz" for c in CLIPS for r in ROLES]
    runs = [run_name(cam, c, r) for c in CLIPS for r in ROLES for cam in CAMS]   # both cameras of a dancer together
    smoke = run_name("02", "E1", "lead")
    rest = ",".join(x for x in runs if x != smoke)
    py = "/workspace/fda/venv/bin/python"
    pod = "remote/cand_fda/fda_pod.py"
    I, R = f"work/{TAKE}/fda_test/inputs", f"work/{TAKE}/fda_test/runs"
    job = {
        "schema": "dancecap.pod_job/1", "name": JOB, "take": TAKE, "cwd": "dancecap", "max_minutes": args.max_minutes,
        "needs": ["ffmpeg"], "base": True, "models": "neutral", "min_vram_gb": 48,
        "inputs": ref_inputs + clips + exmr + ins,
        "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
        "env": {"FDA_ROOT": "/workspace/fda", "HF_HOME": "/workspace/fda/hf_home", "PYTHONUNBUFFERED": "1",
                "TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "UV_LINK_MODE": "hardlink"},
        "commands": [
            {"name": "install", "timeout_min": 30, "cmd": "bash remote/cand_fda/install.sh"},
            {"name": "inputs", "timeout_min": 6, "cmd": "python -m dancecap.fda_test inputs"},
            {"name": "bodymodels", "timeout_min": 2, "cmd": f"{py} {pod} place-models"},
            {"name": "smoke", "timeout_min": 20, "cmd": f"{py} {pod} run --inputs {I} --out {R} --only {smoke} --require-all"},
            {"name": "run", "timeout_min": args.run_minutes,
             "cmd": f"{py} {pod} run --inputs {I} --out {R} --only {rest} --budget-min {args.run_minutes - 4}"},
            {"name": "extract", "timeout_min": 10, "cmd": f"{py} {pod} extract --inputs {I} --out {R}"},
            {"name": "detect", "timeout_min": 15, "cmd": f"{py} {pod} detect --out {R} --every 4"},
            {"name": "score", "timeout_min": 15, "cmd": "python -m dancecap.fda_test score"},
            {"name": "hmr_score", "timeout_min": 15,
             "cmd": f"python -m dancecap.hmr_test score --take {TAKE} --methods B0,B1,B2,exmr,fda --out-sub fda_score"},
            {"name": "hmr_sheet", "timeout_min": 10,
             "cmd": f"python -m dancecap.hmr_test sheet --take {TAKE} --out-sub fda_score"},
        ],
        "outputs": [f"dancecap/work/{TAKE}/fda_test/runs/", f"dancecap/work/{TAKE}/fda_test/score/",
                    f"dancecap/work/{TAKE}/fda_test/inputs/", f"dancecap/work/{TAKE}/hmr_test/fda/",
                    f"dancecap/work/{TAKE}/hmr_test/fda_score/"],
        "verify": [f"dancecap/work/{TAKE}/fda_test/runs/run.json", f"dancecap/work/{TAKE}/fda_test/score/score.json"],
        "notes": [
            "4DAnyone (ant-research/4DAnyone @ 6dcf626, Apache-2.0 code; weights AntResearch/4DAnyone @ 4c80e87: checkpoint "
            "Apache-2.0, Turbo LoRA CC BY-NC-SA 4.0, GVHMR research licence, YOLOv8 AGPL) on lucas_luiza: 4 clips x 2 cameras "
            "x 2 dancers, 121 frames each, 12 views; GVHMR's tracker replaced by our dancer boxes. Research test; weights stay "
            "on the pod, outputs in git-ignored work/ (dancers' consent: lucas_luiza.toml).",
            "Peak memory 22 GB stated by 4DAnyone: min_vram_gb 48 keeps a margin (L40S / 6000 Ada first)."],
    }
    p = WORK / TAKE / "jobs" / "pod" / f"{JOB}.json"
    p.write_text(json.dumps(job, indent=1))
    n_jpg = len(ins)
    mb = sum(Path(x["local"]).stat().st_size for x in ins) / 1e6
    log(f"job {p}: {len(runs)} runs, {n_jpg} frames ({mb:.0f} MB), max {args.max_minutes} min")


# ------------------------------------------------------------------------------------------------ inputs (pod)


def fill_boxes(b: np.ndarray) -> tuple[np.ndarray, int]:
    """(N, 4) with NaN rows -> linearly filled (ends held); returns the count of filled rows."""
    b = b.astype(float).copy()
    bad = ~np.isfinite(b).all(1) | (b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])
    if bad.all():
        raise SystemExit("no valid box in the window")
    idx = np.arange(len(b))
    for j in range(4):
        b[bad, j] = np.interp(idx[bad], idx[~bad], b[~bad, j])
    return b, int(bad.sum())


def cmd_inputs(args) -> None:
    from PIL import Image
    ll = WORK / TAKE
    W = windows()
    root = od() / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    for cam in CAMS:
        T = dict(np.load(ll / "people" / f"teachers_{cam}.npz"))
        for c, rows in W[cam].items():
            rows = np.array(rows)
            if (od() / "inputs" / f"c{cam}_{c}" / "clip.mp4").exists() and all(
                    (od() / "inputs" / run_name(cam, c, r) / "meta.json").exists() for r in ROLES):
                log(f"c{cam}_{c}: inputs exist, kept")
                continue
            ls = T["lead_slot"][rows].astype(int)
            slot = {"lead": ls, "follow": 1 - ls}
            raw = {}
            for r in ROLES:
                b = T["box"][rows, slot[r]].astype(float)
                okr = (T["row"][rows, slot[r]] >= 0) & (T["conf"][rows, slot[r]] >= 0.3)
                b[~okr] = np.nan
                raw[r] = b
            frames = [np.asarray(Image.open(od() / "frames" / cam / f"{r:06d}.jpg").convert("RGB")) for r in rows]
            H, Wd = frames[0].shape[:2]
            if cam == "02":   # landscape tripod: one static 9:16 crop around the couple over the whole window
                allb = np.concatenate([raw[r] for r in ROLES])
                allb = allb[np.isfinite(allb).all(1)]
                x0, y0, x1, y1 = allb[:, 0].min(), allb[:, 1].min(), allb[:, 2].max(), allb[:, 3].max()
                cw, ch = (x1 - x0) * (1 + 2 * PAD), (y1 - y0) * (1 + 2 * PAD)
                ch = min(max(ch, cw * 16 / 9, H * 0.8), 2.0 * H)
                cw = ch * 9 / 16
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                ox = cx - cw / 2
                # inside the frame vertically when it fits; taller than the frame: centred on it, grey padding
                oy = float(np.clip(cy - ch / 2, 0.0, H - ch)) if ch <= H else (H - ch) / 2
            else:            # Steadicam: already 9:16 portrait, the whole frame
                ox, oy, cw, ch = 0.0, 0.0, float(Wd), float(H)
            s = OUT_H / ch
            vd = root / f"c{cam}_{c}"
            fdir = vd / "frames"
            fdir.mkdir(parents=True, exist_ok=True)
            A = np.array([[s, 0, -ox * s], [0, s, -oy * s]])
            for i, f in enumerate(frames):
                im = Image.fromarray(f).transform((OUT_W, OUT_H), Image.Transform.AFFINE,
                                                  data=(1 / s, 0, ox, 0, 1 / s, oy), resample=Image.Resampling.BICUBIC,
                                                  fillcolor=(128, 128, 128))
                im.save(fdir / f"{i:04d}.png")
            num, den = FPS_NUM[cam]
            mp4 = vd / "clip.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{num}/{den}", "-i", str(fdir / "%04d.png"),
                           "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p", str(mp4)], check=True)
            for r in ROLES:
                rd = root / run_name(cam, c, r)
                rd.mkdir(exist_ok=True)
                b, nfill = fill_boxes(raw[r])
                bt = np.stack([b[:, 0] * s - ox * s, b[:, 1] * s - oy * s, b[:, 2] * s - ox * s, b[:, 3] * s - oy * s], 1)
                bt[:, [0, 2]] = bt[:, [0, 2]].clip(0, OUT_W - 1)
                bt[:, [1, 3]] = bt[:, [1, 3]].clip(0, OUT_H - 1)
                np.save(rd / f"boxes_{r}.npy", bt.astype(np.float32))
                link = rd / "clip.mp4"
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(Path("..") / vd.name / "clip.mp4")
                meta = {"cam": cam, "clip": c, "role": r, "rows": rows.tolist(), "crop": {"ox": ox, "oy": oy, "w": cw,
                        "h": ch, "s": s}, "affine_in_from_orig": A.tolist(), "size": [OUT_W, OUT_H],
                        "src_size": [Wd, H], "fps": [num, den], "boxes_filled": nfill}
                (rd / "meta.json").write_text(json.dumps(meta, indent=1))
            subprocess.run(["rm", "-rf", str(fdir)], check=True)
            log(f"c{cam}_{c}: {len(rows)} frames, crop x {ox:.0f} y {oy:.0f} {cw:.0f}x{ch:.0f} -> {OUT_W}x{OUT_H} "
                f"(x{s:.2f}); boxes filled lead {np.isnan(raw['lead']).any(1).sum()} follow {np.isnan(raw['follow']).any(1).sum()}")


# ------------------------------------------------------------------------------------------------ score (pod)


def k_orig(K_in: np.ndarray, crop: dict) -> np.ndarray:
    """K of the input video (crop + scale of the original) -> K of the original camera image."""
    s, ox, oy = crop["s"], crop["ox"], crop["oy"]
    K = np.array(K_in, float).copy()
    K[..., 0, 0] /= s
    K[..., 1, 1] /= s
    K[..., 0, 2] = K[..., 0, 2] / s + ox
    K[..., 1, 2] = K[..., 1, 2] / s + oy
    return K


def umeyama(src, dst):
    from .mono4d.rig import umeyama as u
    return u(src, dst, with_scale=True)


def project_cv(X: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(c2w)
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
    return pinhole(Xc, K)[0]


def match(kp: np.ndarray, conf_box: np.ndarray, uv: np.ndarray, h: float) -> tuple[int, float, float]:
    """Best detected person for projected joints uv (12, 2): (index, median err / h, PCK@0.1) over joints with kp conf
    >= 0.3; (-1, nan, 0) when nobody has >= 6 such joints."""
    best = (-1, np.nan, 0.0)
    for p in range(kp.shape[0]):
        if not np.isfinite(conf_box[p]) or conf_box[p] < 0.25:
            continue
        k = kp[p][COCO]
        c = k[:, 2] >= 0.3
        if c.sum() < 6 or not np.isfinite(uv[c]).all():
            continue
        e = np.linalg.norm(k[c, :2] - uv[c], axis=1) / h
        m = float(np.median(e))
        if best[0] < 0 or m < best[1]:
            best = (p, m, float((e < PCK_T).mean()))
    return best


def views2d(fda: dict, det: dict, J0c: dict, role: str) -> dict:
    """Generated views vs the real couple (B0 mapped into the canonical frame) and vs the run's own skeleton."""
    K, c2w, wh = fda["cam_K"], fda["cam_c2w"], fda["cam_wh"]
    kp, box, fr = det["kp"], det["box"], det["frames"]
    partner = ROLES[1 - ROLES.index(role)]
    rows = []
    for v in range(len(K)):
        for j, f in enumerate(fr):
            rec = {"view": v, "frame": int(f)}
            for who, X in (("self", J0c[role][f]), ("partner", J0c[partner][f]), ("skel", fda["J_canon"][f])):
                if not np.isfinite(X).all():
                    continue
                uv = project_cv(X[:NJ], K[v], c2w[v])
                h = max(float(np.ptp(uv[:, 1])) * 1.15, 20.0)
                inside = ((uv[:, 0] >= 0) & (uv[:, 0] < wh[v, 0]) & (uv[:, 1] >= 0) & (uv[:, 1] < wh[v, 1])).mean()
                p, err, pck = match(kp[v, j], box[v, j, :, 4], uv[SMX], h)
                rec[who] = {"err": err, "pck": pck if p >= 0 else 0.0, "found": bool(p >= 0 and err < FOUND_T),
                            "inside": float(inside), "p": p}
            rec["n05"] = int(np.nansum(box[v, j, :, 4] >= 0.5))
            if "self" in rec and "partner" in rec and rec["self"]["found"] and rec["partner"]["found"]:
                rec["same_person"] = rec["self"]["p"] == rec["partner"]["p"]
            rows.append(rec)
    return {"rows": rows}


def agg_views(rows: list, yaw_rel: np.ndarray) -> dict:
    def stat(sel, who, key):
        v = [r[who][key] for r in rows if sel(r) and who in r]
        if not v:
            return None
        return float(np.nanmedian(v)) if key in ("err", "pck") else float(np.mean(v))
    out = {}
    for name, sel in (("all", lambda r: True), ("near_source", lambda r: abs(yaw_rel[r["view"]]) <= 45),
                      ("side", lambda r: 45 < abs(yaw_rel[r["view"]]) <= 135), ("back", lambda r: abs(yaw_rel[r["view"]]) > 135)):
        n05 = [r["n05"] for r in rows if sel(r)]
        out[name] = {
            "n": len(n05),
            "self_pck": stat(sel, "self", "pck"), "self_found": stat(sel, "self", "found"),
            "partner_pck": stat(sel, "partner", "pck"), "partner_found": stat(sel, "partner", "found"),
            "skel_pck": stat(sel, "skel", "pck"), "skel_found": stat(sel, "skel", "found"),
            "partner_in_view": stat(sel, "partner", "inside"),
            "people_2": float(np.mean([n == 2 for n in n05])) if n05 else None,
            "people_1": float(np.mean([n <= 1 for n in n05])) if n05 else None,
            "people_3plus": float(np.mean([n >= 3 for n in n05])) if n05 else None,
        }
    return out


def cmd_score(args) -> None:
    from . import mamma_dyn as md
    from .hmr_test import Ref
    ref = Ref(TAKE)
    ll = WORK / TAKE
    offs = md.read_sync_offsets(ll)
    c = {"01": ref.c1, "02": ref.c2}
    Rw = {"01": ref.R1, "02": ref.R2}
    rd_root = od() / "runs"
    sd = od() / "score"
    sd.mkdir(parents=True, exist_ok=True)
    rep = json.loads((rd_root / "run.json").read_text())
    res = {"schema": "dancecap.fda_test/1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": {}, "fused": {},
           "run_status": {k: {"ok": v.get("ok"), "failed": v.get("failed"), "seconds": v.get("seconds")}
                          for k, v in rep["runs"].items()}}
    J0 = {r: ref.B0[r]["joints"][:, :NJ].astype(float) for r in ROLES}
    on_grid = {}     # (cam, clip, role) -> (grid idx, world root-relative joints)
    hm = WORK / TAKE / "hmr_test" / "fda"
    hm_runs = {}
    for cam in CAMS:
        for clip in CLIPS:
            for role in ROLES:
                name = run_name(cam, clip, role)
                rd = rd_root / name
                if not rep["runs"].get(name, {}).get("ok") or not (rd / "fda.npz").exists():
                    res["runs"][name] = {"status": "not run correctly: " + str(rep["runs"].get(name, {}).get("failed"))}
                    continue
                meta = json.loads((od() / "inputs" / name / "meta.json").read_text())
                fda = dict(np.load(rd / "fda.npz"))
                rows = np.array(meta["rows"])
                tau = md.apply_offset(c[cam]["t"][rows], offs.get(cam))
                B = {r: interp_grid(J0[r], ref.tg, tau) for r in ROLES}
                Jc = fda["J_cam"]
                pa = np.array([pa_mpjpe(Jc[i], B[role][i]) if np.isfinite(B[role][i]).all() else np.nan for i in range(L)])
                # world root-relative pose through the camera's known rotation (no Procrustes): orientation counts
                Rr = Rw[cam][rows]
                Jw = to_world(Jc, Rr, np.zeros((L, 3)))
                rr_w = Jw - Jw[:, :1]
                b_rr = B[role] - B[role][:, :1]
                rr = np.linalg.norm(rr_w - b_rr, axis=-1).mean(-1) * 1000
                # 2D in its own image vs our teacher kp (in-view check, % of box height)
                Ko = k_orig(fda["K_in"], meta["crop"])
                T = dict(np.load(ll / "people" / f"teachers_{cam}.npz"))
                ls = T["lead_slot"][rows].astype(int)
                sl = ls if role == "lead" else 1 - ls
                own = []
                for i in range(L):
                    kp = T["kp"][rows[i], sl[i]]
                    cc = kp[COCO, 2] >= 0.5
                    if cc.sum() < 6:
                        own.append(np.nan)
                        continue
                    uv = pinhole(Jc[i][SMX], Ko[i])[0]
                    bh = max(float(T["box"][rows[i], sl[i], 3] - T["box"][rows[i], sl[i], 1]), 50.0)
                    own.append(100 * float(np.median(np.linalg.norm(uv[cc] - kp[COCO][cc, :2], axis=1))) / bh)
                own = np.array(own)
                # views2d: B0's couple into the canonical frame via the run's dancer
                okf = np.isfinite(B[role]).all((1, 2))
                sim = umeyama(fda["J_canon"][okf].reshape(-1, 3), B[role][okf].reshape(-1, 3))
                s_, R_, t_ = sim["s"], np.array(sim["R"]), np.array(sim["t"])
                J0c = {r: ((B[r] - t_) @ R_) / s_ for r in ROLES}      # world -> canonical
                # the source camera's yaw in the canonical frame (for front / side / back bins)
                cen_w = -np.einsum("fji,fj->fi", Rr, np.asarray(ref.t1 if cam == "01" else ref.t2)[rows])
                cen_c = ((cen_w - t_) @ R_) / s_
                body = fda["J_canon"][:, 0]
                az_src = np.degrees(np.arctan2(*(cen_c - body)[:, [0, 2]].T)).mean()
                cam_cen = fda["cam_c2w"][:, :3, 3]
                az_v = np.degrees(np.arctan2(cam_cen[:, 0] - body[:, 0].mean(), cam_cen[:, 2] - body[:, 2].mean()))
                yaw_rel = (az_v - az_src + 180) % 360 - 180
                v2 = views2d(fda, dict(np.load(rd / "det.npz")), J0c, role)
                agg = agg_views(v2["rows"], yaw_rel)
                res["runs"][name] = {
                    "status": "ok", "cam": cam, "clip": clip, "role": role, "seconds": rep["runs"][name].get("seconds"),
                    "pa_mm": float(np.nanmedian(pa)), "pa_mean_mm": float(np.nanmean(pa)),
                    "rr_world_mm": float(np.nanmedian(rr)), "own_2d_pct": float(np.nanmedian(own)),
                    "sim3_to_B0": {"s": s_, "rms_mm": sim["rms"] * 1000}, "focal_ratio": float(np.median(Ko[:, 0, 0] / c[cam]["K"][rows, 0, 0])),
                    "views": agg, "yaw_rel": yaw_rel.round(1).tolist(), "boxes_filled": meta["boxes_filled"]}
                np.savez_compressed(sd / f"{name}_views.npz", rows=json.dumps(v2["rows"]), yaw_rel=yaw_rel)
                gi = np.round((tau - ref.tg[0]) * 30).astype(int)
                on_grid[(cam, clip, role)] = (gi, rr_w, pa)
                if cam == "02":   # hmr_test contract: camera-02 joints + K on the clip's cut frames
                    meta_c = json.loads((WORK / TAKE / "hmr_test" / "clips" / clip / "meta.json").read_text())
                    n0, Lc = meta_c["n0"], meta_c["L"]
                    i_clip = grid_index_of_cut(n0 + np.arange(Lc), ref.kg0)
                    J_out, K_out = np.full((Lc, NJ, 3), np.nan), np.full((Lc, 3, 3), np.nan)
                    pos = {int(g): i for i, g in enumerate(gi)}
                    for q, g in enumerate(i_clip):
                        if int(g) in pos:
                            J_out[q], K_out[q] = Jc[pos[int(g)]], Ko[pos[int(g)]]
                    (hm / f"{clip}_{role}").mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(hm / f"{clip}_{role}" / "joints_cam02.npz", J_c=J_out, K=K_out)
                    hm_runs[f"{clip}_{role}"] = {"ok": True, "frames": int(np.isfinite(J_out).all((1, 2)).sum())}
                log(f"{name}: PA {res['runs'][name]['pa_mm']:.0f} mm, world rr {res['runs'][name]['rr_world_mm']:.0f} mm, "
                    f"own 2D {res['runs'][name]['own_2d_pct']:.1f} %, partner found {agg['all']['partner_found']}")
    if hm_runs:
        (hm / "run.json").write_text(json.dumps({"method": "fda", "runs": hm_runs}, indent=1))
    # two-view fusion: the same clip and dancer through both cameras, on the grid frames both cover
    for clip in CLIPS:
        for role in ROLES:
            a, b = on_grid.get(("02", clip, role)), on_grid.get(("01", clip, role))
            if a is None or b is None:
                continue
            common = sorted(set(a[0].tolist()) & set(b[0].tolist()))
            ia = {g: i for i, g in enumerate(a[0])}
            ib = {g: i for i, g in enumerate(b[0])}
            e = {"02": [], "01": [], "fused": [], "pa02": [], "pa01": [], "pafused": []}
            for g in common:
                if not (0 <= g < ref.NG):
                    continue
                y = J0[role][g] - J0[role][g][:1]
                xa, xb = a[1][ia[g]], b[1][ib[g]]
                xf = (xa + xb) / 2
                for k, x in (("02", xa), ("01", xb), ("fused", xf)):
                    e[k].append(float(np.linalg.norm(x - y, axis=-1).mean() * 1000))
                    e["pa" + k].append(pa_mpjpe(x, y))
            res["fused"][f"{clip}_{role}"] = {k: float(np.median(v)) if v else None for k, v in e.items()} | {"n": len(e["fused"])}
    # pooled tables
    pool = {}
    for cam in CAMS:
        for role in ROLES:
            rs = [v for v in res["runs"].values() if v.get("status") == "ok" and v["cam"] == cam and v["role"] == role]
            if not rs:
                continue
            pool[f"{cam}_{role}"] = {k: float(np.median([r[k] for r in rs])) for k in ("pa_mm", "rr_world_mm", "own_2d_pct")}
            for b in ("all", "near_source", "side", "back"):
                for k in ("self_pck", "partner_pck", "partner_found", "skel_pck", "people_2", "people_1", "people_3plus"):
                    vals = [r["views"][b][k] for r in rs if r["views"][b][k] is not None]
                    pool[f"{cam}_{role}"][f"{b}_{k}"] = float(np.median(vals)) if vals else None
    res["pooled"] = pool
    fz = {}
    for role in ROLES:
        rows_ = [v for k, v in res["fused"].items() if k.endswith(role)]
        if rows_:
            fz[role] = {k: float(np.median([r[k] for r in rows_ if r[k] is not None])) for k in ("02", "01", "fused", "pa02", "pa01", "pafused")}
    res["fused_pooled"] = fz
    (sd / "score.json").write_text(json.dumps(res, indent=1, default=float))
    write_sheet(res, sd)
    try:
        contact_sheets(res, sd)
    except Exception as ex:   # noqa: BLE001 - the pictures are a convenience, the numbers are the result
        log(f"contact sheets failed: {ex}")
    log(f"score: {sd / 'score.json'}")


def _f(x, nd=0, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{100 * x:.0f} %" if pct else f"{x:.{nd}f}"


def write_sheet(res: dict, sd: Path) -> None:
    L_ = ["# 4DAnyone on lucas_luiza (one video -> generated views of one person), per dancer", ""]
    L_ += ["Pose = the GVHMR skeleton 4DAnyone generates from (it is the method's only 3D). PA = PA-MPJPE vs the two-view "
           "fit B0 (22 joints, per-frame median, mm); world rr = root-relative joints rotated to the world by the camera's "
           "known rotation, no Procrustes (mm); own 2D = its joints in its own image vs our 2D keypoints (% of box height).",
           "", "## Runs", "",
           "| run | PA | world rr | own 2D | views: self PCK | partner PCK | partner found | follows skeleton (PCK) | 2 people | 1 person | sim3 rms mm |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["runs"].items():
        if v.get("status") != "ok":
            L_.append(f"| {k} | {v['status']} |||||||||| ")
            continue
        a = v["views"]["all"]
        L_.append(f"| {k} | {_f(v['pa_mm'])} | {_f(v['rr_world_mm'])} | {_f(v['own_2d_pct'], 1)} | {_f(a['self_pck'], pct=True)} | "
                  f"{_f(a['partner_pck'], pct=True)} | {_f(a['partner_found'], pct=True)} | {_f(a['skel_pck'], pct=True)} | "
                  f"{_f(a['people_2'], pct=True)} | {_f(a['people_1'], pct=True)} | {_f(v['sim3_to_B0']['rms_mm'])} |")
    L_ += ["", "## Pooled (median over the 4 clips)", "",
           "| camera, dancer | PA | world rr | own 2D | partner found: near source / side / back | partner PCK all | self PCK all | 2 people all |",
           "|---|---|---|---|---|---|---|---|"]
    for k, p in res.get("pooled", {}).items():
        L_.append(f"| {k} | {_f(p['pa_mm'])} | {_f(p['rr_world_mm'])} | {_f(p['own_2d_pct'], 1)} | "
                  f"{_f(p['near_source_partner_found'], pct=True)} / {_f(p['side_partner_found'], pct=True)} / {_f(p['back_partner_found'], pct=True)} | "
                  f"{_f(p['all_partner_pck'], pct=True)} | {_f(p['all_self_pck'], pct=True)} | {_f(p['all_people_2'], pct=True)} |")
    L_ += ["", "## Two cameras through 4DAnyone, fused (root-relative world joints averaged; mm)", "",
           "| dancer | tripod alone | Steadicam alone | fused | PA tripod | PA Steadicam | PA fused |", "|---|---|---|---|---|---|---|"]
    for r, v in res.get("fused_pooled", {}).items():
        L_.append(f"| {r} | {_f(v['02'])} | {_f(v['01'])} | {_f(v['fused'])} | {_f(v['pa02'])} | {_f(v['pa01'])} | {_f(v['pafused'])} |")
    (sd / "sheet.md").write_text("\n".join(L_) + "\n")


def contact_sheets(res: dict, sd: Path, frame: int = 60) -> None:
    """Per clip and camera: the real input frame, then 6 generated views for each dancer's run, with B0's lead (cyan) and
    follow (magenta) projected (their mapping through the run's Sim3), at one instant."""
    import av
    import cv2
    for cam in CAMS:
        for clip in CLIPS:
            rows_img = []
            for role in ROLES:
                name = run_name(cam, clip, role)
                if res["runs"].get(name, {}).get("status") != "ok":
                    continue
                rd = od() / "runs" / name
                fda = dict(np.load(rd / "fda.npz"))
                views = sorted((rd / "videos").glob("*.mp4"))
                pick = list(range(0, len(views), max(1, len(views) // 6)))[:6]
                tiles = []
                with av.open(str(od() / "inputs" / name / "clip.mp4")) as cc:
                    for i, f in enumerate(cc.decode(video=0)):
                        if i == frame:
                            src = f.to_ndarray(format="bgr24")
                            break
                tiles.append(cv2.resize(src, (352, 640)))
                meta = json.loads((od() / "inputs" / name / "meta.json").read_text())
                vz = np.load(sd / f"{name}_views.npz")
                for v in pick:
                    with av.open(str(views[v])) as cc:
                        for i, f in enumerate(cc.decode(video=0)):
                            if i == frame:
                                img = f.to_ndarray(format="bgr24")
                                break
                    tiles.append(cv2.resize(img, (352, 640)))
                    cv2.putText(tiles[-1], f"{role} v{v} {vz['yaw_rel'][v]:+.0f}deg", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)
                rows_img.append(np.concatenate(tiles, 1))
                del meta, fda
            if rows_img:
                cv2.imwrite(str(sd / f"contact_c{cam}_{clip}.jpg"), np.concatenate(rows_img, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("job")
    p.add_argument("--max-minutes", type=int, default=150)
    p.add_argument("--run-minutes", type=int, default=75)
    sub.add_parser("inputs")
    sub.add_parser("score")
    a = ap.parse_args(argv)
    {"job": cmd_job, "inputs": cmd_inputs, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
