"""john_lula: the front end of ONE chunk of the single-camera fit (agent J3b, 2026-10-11): everything between the source video and `mamma prepare`.

The recap's front (recap_front) asks for SAM masks per dancer from teacher boxes; here J3a already produced them (people/masks_01.npz: SAM 2.1 per dancer per
frame, anchored on the bibs), the camera (cameras/01.npz: per-frame K, R of a handheld phone that rotates and zooms about one centre, J2) and the 2D skeletons
(people/teachers_01.npz). So the DAG is the recap's without its people / SAM / merge steps:

    frames     the chunk window's canonical frames decoded from the video (recap_front.step_frames)
    cameras    cameras/01.npz cut to the window (recap_front.step_cameras)
    masks      dancers/01/masks/mask_NNNN_01|02.png from people/masks_01.npz (lead = John, follow = Lula) + dancers/roles.json
    ma2d       MAMMA ma_2d on those masks (the 512 dense landmarks per dancer)
    sync       the identity clock (one camera: nothing to measure; fixed_sync_doc)
    prepare    mamma prepare --skip-2d --set c1=01 --coverage union --grid-anchor 0 --grid-k=k0:k1
    mono-init  mamma/mono_init.npz (pelvis position per grid frame from the ankle / hip rays, dancecap.jl_mono) + mamma/stature.json

    HM_FRONT_DANCE=john_lula HM_FRONT_CAPTURE=John-Lula python -m dancecap.jl_front run <chunk id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib

import numpy as np

os.environ.setdefault("HM_FRONT_DANCE", "john_lula")
os.environ.setdefault("HM_FRONT_CAPTURE", "John-Lula")

from . import recap_front as RF            # noqa: E402  (reads the env above)
from .take import PACKAGE_ROOT              # noqa: E402

CAM = "01"
PEOPLE = PACKAGE_ROOT / "work" / "john_lula" / "people"


def log(msg: str) -> None:
    print(f"[jl_front {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ masks


def step_masks(chunk: dict) -> dict:
    """J3a's masks of the chunk's frames -> dancers/01/masks/mask_{i:04d}_{01|02}.png (i = index in the chunk's frames, the numbering ma_2d reads) and the
    minimal dancers/roles.json (box per role and frame) that the avatar stages read. The PNG bytes are copied as they are (0 / 255, frame size)."""
    work = RF.take_work(chunk["id"])
    z = np.load(RF.dance_work() / "cameras" / f"{CAM}.npz", allow_pickle=True)
    idx = RF.frame_range(np.asarray(z["t"], float), chunk["t0"], chunk["t1"])
    fi = np.asarray(z["frame_index"], int)[idx]
    tt = np.asarray(z["t"], float)[idx]
    mk = np.load(PEOPLE / f"masks_{CAM}.npz", allow_pickle=True)
    out = work / "dancers" / CAM / "masks"
    out.mkdir(parents=True, exist_ok=True)
    frames, miss = [], {0: 0, 1: 0}
    for k, n in enumerate(fi):
        row = {"i": int(k), "t": float(tt[k]), "swap_prob": 0.0}
        for slot, role in ((0, "lead"), (1, "follow")):
            key = f"m_{int(n)}_{slot}"
            if key in mk.files and mk[key].size:
                (out / f"mask_{k:04d}_{slot + 1:02d}.png").write_bytes(mk[key].tobytes())
                row[role] = {"box": [float(x) for x in mk["box"][int(n), slot]]}
            else:
                miss[slot] += 1
        frames.append(row)
    (work / "dancers").mkdir(parents=True, exist_ok=True)
    (work / "dancers" / "roles.json").write_text(json.dumps({"cams": {CAM: {"frames": frames}}, "source": "J3a people/masks_01.npz (SAM 2.1, bib identity)"}))
    log(f"masks: {len(fi)} frames, missing lead {miss[0]} / follow {miss[1]}")
    return {"frames": int(len(fi)), "missing": miss}


# ------------------------------------------------------------------------------------------------ cameras (with the focal-hypothesis variant)


def step_cameras(chunk: dict) -> dict:
    """recap_front.step_cameras (the chunk's slice of cameras/01.npz) with an optional focal hypothesis: JL_FOCAL_SCALE = s multiplies every frame's focal (K[0,0], K[1,1]) and
    leaves the rotation and the camera centre alone. J2: the focal of this phone video is a hypothesis (+-10 %; distances scale with it, the camera height does not); a re-solve at s = 0.9
    and 1.1 shows what that does to the motion."""
    res = RF.step_cameras(chunk)
    s = float(os.environ.get("JL_FOCAL_SCALE", "1.0"))
    if s != 1.0:
        p = RF.take_work(chunk["id"]) / "cameras" / f"{CAM}.npz"
        z = dict(np.load(p, allow_pickle=True))
        z["K"] = np.array(z["K"], float)
        z["K"][:, 0, 0] *= s
        z["K"][:, 1, 1] *= s
        z["focal_scale"] = np.float64(s)
        if "focal_px" in z:
            z["focal_px"] = np.asarray(z["focal_px"], float) * s
        np.savez_compressed(p, **z)
        log(f"cameras: focal scaled by {s:g}")
    return res


# ------------------------------------------------------------------------------------------------ mono init


def step_mono_init(chunk: dict) -> dict:
    """mamma/mono_init.npz {pos (T, 2, 3), conf (T, 2), depth, source} on the prepared grid + mamma/stature.json (rest heights for the shape pass)."""
    from . import jl_mono
    work = RF.take_work(chunk["id"])
    prep = json.loads((work / "mamma" / "prepare.json").read_text(encoding="utf-8"))
    tg = np.asarray(prep["grid"]["t"], float)
    cam = np.load(work / "cameras" / f"{CAM}.npz", allow_pickle=True)           # the chunk's cameras (the focal variant included)
    tc = np.asarray(cam["t"], float)
    fidx = np.asarray(cam["frame_index"], int)
    tp = np.load(PEOPLE / f"teachers_{CAM}.npz", allow_pickle=True)
    j = np.clip(np.searchsorted(tc, tg), 1, len(tc) - 1)
    j = np.where(np.abs(tc[j - 1] - tg) < np.abs(tc[j] - tg), j - 1, j)
    jt = fidx[j]                                                                  # the dance-wide frame (row of the teacher tracks)
    cfg = tomllib.loads((PACKAGE_ROOT / "takes" / "john_lula.toml").read_text(encoding="utf-8"))
    stat = [float(cfg["dancers"]["lead"]["stature_m"]), float(cfg["dancers"]["follow"]["stature_m"])]
    T = len(tg)
    pos, conf, depth, src = np.zeros((T, 2, 3)), np.zeros((T, 2)), np.zeros((T, 2)), np.zeros((T, 2), int)
    K, R, tv, k1 = (np.asarray(cam[k], float) for k in ("K", "R", "tvec", "dist"))
    ok = np.asarray(cam["ok"], bool)
    for r in range(2):
        res = jl_mono.pelvis_positions(np.asarray(tp["kp_wb"][jt, r], float), K[j], R[j], tv[j], k1[j].reshape(T, -1)[:, 0], stat[r], ok[j],
                                       np.asarray(tp["skel_in_partner"][jt, r], float))
        pos[:, r], conf[:, r], depth[:, r], src[:, r] = res["pos"], res["conf"], res["depth"], res["source"]
    if not np.isfinite(pos).all():
        bad = ~np.isfinite(pos).all(-1)
        raise RuntimeError(f"mono-init: no position at all for {int(bad.sum())} (frame, dancer) pairs of {2 * T}")
    np.savez_compressed(work / "mamma" / "mono_init.npz", pos=pos.astype(np.float32), conf=conf, depth=depth, source=src, t=tg)
    soles = {"lead": float(cfg["dancers"]["lead"].get("shoe_sole_m", 0.02)), "follow": float(cfg["dancers"]["follow"].get("shoe_sole_m", 0.02))}
    (work / "mamma" / "stature.json").write_text(json.dumps(jl_mono.stature_targets(cfg, soles)))
    rep = {"T": T, "measured": {"lead": float(conf[:, 0].mean()), "follow": float(conf[:, 1].mean())},
           "from_feet": {"lead": float((src[:, 0] == 1).mean()), "follow": float((src[:, 1] == 1).mean())},
           "from_torso": {"lead": float((src[:, 0] == 2).mean()), "follow": float((src[:, 1] == 2).mean())},
           "depth_m_median": {"lead": float(np.median(depth[:, 0])), "follow": float(np.median(depth[:, 1]))}}
    log(f"mono-init: {rep}")
    (work / "mamma" / "mono_init.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------------------------------------ texture window front (agent J5b)


def step_hair(chunk: dict) -> dict:
    """dancers/01/body|hair masks of the window from J3a's full masks: the lead's body = his full mask; the follow is split into hair / body by
    dancers_hair.split_hair (keypoint-anchored, one Lab colour model for the camera pooled over the window's frames: hair seeds in the head zone, skin, rest).
    The 2D skeleton is J3a's COCO-17 track (people/teachers_01.npz) at the frame's dance-wide row."""
    import cv2

    from . import dancers_hair as DH
    work = RF.take_work(chunk["id"])
    dd = work / "dancers" / CAM
    meta = json.loads((work / "frames" / CAM / "frames.json").read_text())
    z = np.load(RF.dance_work() / "cameras" / f"{CAM}.npz", allow_pickle=True)
    idx = RF.frame_range(np.asarray(z["t"], float), chunk["t0"], chunk["t1"])
    fi = np.asarray(z["frame_index"], int)[idx]
    kp = np.load(PEOPLE / f"teachers_{CAM}.npz", allow_pickle=True)["kp"][:, 1]
    for sub in ("body", "hair"):
        (dd / sub).mkdir(parents=True, exist_ok=True)
    cfg = DH.HairConfig()
    colour = DH.HairColour(cfg)
    F = len(meta["frames"])

    def read_mask(i: int, pid: int):
        m = cv2.imread(str(dd / "masks" / f"mask_{i:04d}_{pid:02d}.png"), cv2.IMREAD_GRAYSCALE)
        return None if m is None else (m > 127)

    for i in range(0, F, cfg.seed_stride):
        fol = read_mask(i, 2)
        if fol is None or not fol.any():
            continue
        img = cv2.imread(str(work / "frames" / CAM / f"{i:06d}.jpg"))
        ys = np.nonzero(fol.any(1))[0]
        colour.add_frame(img, fol, DH.head_geom(kp[fi[i]], float(ys[-1] - ys[0] + 1), cfg), kp[fi[i]])
    ok = colour.fit()
    n_hair = n_fol = 0
    for i in range(F):
        lead, fol = read_mask(i, 1), read_mask(i, 2)
        if lead is not None:
            cv2.imwrite(str(dd / "body" / f"mask_{i:04d}_01.png"), lead.astype(np.uint8) * 255)
        if fol is None:
            continue
        img = cv2.imread(str(work / "frames" / CAM / f"{i:06d}.jpg"))
        hair, body = DH.split_hair(img, fol, kp[fi[i]], colour, cfg)
        cv2.imwrite(str(dd / "body" / f"mask_{i:04d}_02.png"), body.astype(np.uint8) * 255)
        cv2.imwrite(str(dd / "hair" / f"mask_{i:04d}_02.png"), hair.astype(np.uint8) * 255)
        n_fol += 1
        n_hair += int(hair.any())
    rep = {"frames": F, "follow_masks": n_fol, "with_hair": n_hair, "colour_model": bool(ok), "seeds": colour.seeds}
    log(f"hair split: {rep}")
    (work / "dancers" / "hair_split.json").write_text(json.dumps(rep, default=str))
    return rep


def texture_front(chunk_id: str) -> int:
    """The front of a TEXTURE window (no fit follows): frames, cameras (the focal hypothesis of the env included), J3a's masks + roles.json, the follow's hair /
    body split. Plain sequential CPU steps (a 32 s window decodes in about a minute); the bake reads exactly this."""
    chunk = RF.load_chunk(chunk_id)
    work = RF.take_work(chunk_id)
    (work / "front").mkdir(parents=True, exist_ok=True)
    RF.step_frames(chunk)
    step_cameras(chunk)
    step_masks(chunk)
    step_hair(chunk)
    (work / "front" / "front_done.json").write_text(json.dumps({"rc": 0, "purpose": "texture window front (J5b)", "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    return 0


# ------------------------------------------------------------------------------------------------ DAG


def build_tasks(R, chunk: dict) -> None:
    from .front_pod import Task
    py = sys.executable
    work = RF.take_work(chunk["id"])
    take = RF.take_name(chunk["id"])
    me = "dancecap.jl_front"
    done_dir = work / "front" / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    flag = lambda n: (lambda: (done_dir / f"{n}.ok").exists())            # noqa: E731
    mark = lambda n: (lambda: (done_dir / f"{n}.ok").write_text(time.strftime("%FT%T")))   # noqa: E731
    R.add(Task("frames", [py, "-m", "dancecap.recap_front", "frames", chunk["id"]], kind="cpu", mem_gb=4.0, timeout_min=30, done_if=flag("frames"), on_ok=mark("frames")))
    R.add(Task("cameras", [py, "-m", me, "cameras", chunk["id"]], kind="cpu", mem_gb=1.0, timeout_min=10, done_if=flag("cameras"), on_ok=mark("cameras")))
    R.add(Task("masks", [py, "-m", me, "masks", chunk["id"]], kind="cpu", mem_gb=2.0, timeout_min=10, deps=["cameras"], done_if=flag("masks"), on_ok=mark("masks")))
    R.add(Task("ma2d:01", [py, "-m", "dancecap.front_pod", "ma2d", take, CAM], deps=["masks", "frames"], kind="ma2d", mem_gb=8.0, gpu=True, timeout_min=60,
               done_if=lambda: (work / "mamma" / "ma_2d_native" / f"{CAM}.manifest.json").exists()))
    R.add(Task("sync", [py, "-m", "dancecap.recap_front", "sync", chunk["id"]], deps=["cameras"], after=["ma2d:01"], kind="cpu", mem_gb=2.0, timeout_min=10, retries=0))
    k0, k1 = int(chunk["k0"]), int(chunk["k1"])
    R.add(Task("prepare", [py, "-m", "dancecap", "mamma", "prepare", take, "--skip-2d", "--set", f"c1={CAM}", "--coverage", "union", "--grid-anchor",
                           repr(RF.GRID_ANCHOR), f"--grid-k={k0}:{k1}"], deps=["sync", "ma2d:01"], kind="cpu", mem_gb=6.0, timeout_min=30, retries=0))
    R.add(Task("mono-init", [py, "-m", me, "mono-init", chunk["id"]], deps=["prepare"], kind="cpu", mem_gb=3.0, timeout_min=15, retries=0))


def run(chunk_id: str, resume: bool = False) -> int:
    from . import front_pod as fp
    chunk = RF.load_chunk(chunk_id)
    work = RF.take_work(chunk_id)
    cpus, ram = fp.cgroup_cpus(), fp.mem_gb()
    ws0, limit = fp.mem_state()
    threads = str(max(2, min(4, cpus // 8)))
    env = dict(os.environ, DANCECAP_GPU_SHARED="1", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, OPENCV_FOR_THREADS_NUM=threads,
               PYTHONUNBUFFERED="1", DANCECAP_POD=os.environ.get("DANCECAP_POD", "1"), YOLO_CONFIG_DIR="/tmp/Ultralytics")
    cap_gb = limit if limit else ram
    caps = {"sam": 1, "ma2d": 1, "cpu": 3, "loc": 1}
    R = fp.Runner(work, env, max(16.0, 0.62 * cap_gb), caps, work / "front" / "logs", limit)
    build_tasks(R, chunk)
    if resume:
        R.resume()
    else:
        R.memlog.unlink(missing_ok=True)
        for f in R.rc_dir.glob("*.rc"):
            f.unlink()
    R.meta = {"take": RF.take_name(chunk_id), "chunk": chunk_id, "cams": chunk["cams"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "host": {"cpus": cpus, "ram_gb": round(ram, 1), "mem_limit_gb": limit, "caps": caps}}
    R.save()
    log(f"{RF.take_name(chunk_id)}: {len(R.tasks)} tasks")
    R.loop()
    R.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    R.save()
    bad = [k for k, t in R.tasks.items() if t.state != "ok" and not t.optional]
    ok = R.tasks["mono-init"].state == "ok"
    (work / "front").mkdir(parents=True, exist_ok=True)
    (work / "front" / "front_done.json").write_text(json.dumps({"rc": 0 if ok else 1, "not_ok": bad, "finished": R.meta["finished"]}))
    log(f"done; not ok: {bad or 'none'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_front", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run", "masks", "cameras", "mono-init", "hair", "texture-front"))
    ap.add_argument("chunk")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a.chunk, a.resume)
    if a.cmd == "texture-front":
        return texture_front(a.chunk)
    chunk = RF.load_chunk(a.chunk)
    {"masks": step_masks, "cameras": step_cameras, "mono-init": step_mono_init, "hair": step_hair}[a.cmd](chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
