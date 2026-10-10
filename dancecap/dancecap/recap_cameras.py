"""Recap cameras (agent R2): upload of the class recap to the hm-farm volume, camera-localisation jobs on Runpod, pull, merge, QA.

    python -m dancecap.recap_cameras upload [--keep-gateway]      videos + compressed 2D pose JSONs -> volume src/<capture>/
    python -m dancecap.recap_cameras table                        write cameras/recap_cams.json (display sizes, clocks, fps)

The localisation itself is dancecap.cameras.newcam (it runs on the pod); this module is the PC-side orchestration:
volume upload (resumable, hash-diffed, striped over several ssh streams), job specs for the farm worker, pulls, merge of
the 60 s chunks into the per-camera tracks, quality report and review images. Nothing here runs heavy compute locally.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
import numpy as np
from pathlib import Path

from .take import PACKAGE_ROOT

TAKE = "larissa_kadu_recap"
CAPTURE = "Larissa-Kadu-recap"            # volume src/<capture>/ = Path(take source_dir).name (farm convention)
WORK = PACKAGE_ROOT / "work" / TAKE
SOURCE = PACKAGE_ROOT.parent / "dances" / CAPTURE   # the git-ignored junction to the owner's folder
ALIGNMENT = PACKAGE_ROOT.parent / "prototype" / "larissa_kadu" / "alignment_recap.json"
CAMS_JSON = WORK / "cameras" / "recap_cams.json"


def log(msg: str) -> None:
    print(f"[recap_cameras {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ camera table


def camera_table() -> dict:
    """{cam: {file, w, h, rotation, fps, frames, duration, slope, offset, accepted, ...}} of all seven videos (DISPLAY sizes).

    Tony (04) is excluded from the take (not placed on the master clock) but is still a camera to localise, so the table is built
    from the alignment file and the files themselves, not from Take.load."""
    from .ingest import probe
    al = json.loads(ALIGNMENT.read_text())
    out: dict = {}
    for r in al["sources"]:
        p = SOURCE / r["name"]
        if not p.exists():
            raise FileNotFoundError(p)
        pr = probe(p)
        out[r["id"]] = {"id": r["id"], "file": r["name"], "width": pr["width"], "height": pr["height"],
                        "rotation": pr["rotation"], "avg_fps": pr["avg_fps"], "duration": r.get("duration"),
                        "slope": r.get("slope"), "offset": r.get("offset"), "accepted": bool(r.get("accepted", True)),
                        "offset_source": r.get("offset_source"), "bytes": p.stat().st_size}
    return out


def write_table() -> dict:
    t = camera_table()
    CAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    CAMS_JSON.write_text(json.dumps({"take": TAKE, "capture": CAPTURE, "cams": t}, indent=1))
    return t


# ------------------------------------------------------------------------------------------------ upload


def gzip_poses(out_dir: Path) -> list[Path]:
    """Compress the per-video 2D pose JSONs (420 MB of text) for the upload; returns the .json.gz paths (cached by size+mtime)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for src in sorted(SOURCE.glob("*.mp4.json")):
        dst = out_dir / (src.name + ".gz")
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            res.append(dst)
            continue
        t0 = time.time()
        tmp = dst.with_suffix(".part")
        with open(src, "rb") as fi, gzip.open(tmp, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo, 1 << 22)
        tmp.replace(dst)
        log(f"gzip {src.name}: {src.stat().st_size / 1e6:.0f} -> {dst.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f} s")
        res.append(dst)
    return res


def upload(keep_gateway: bool = False, with_poses: bool = True, link=None) -> dict:
    """Push the 7 videos (+ compressed pose JSONs) to src/<capture>/ once. Resumable and hash-diffed: a second call sends nothing."""
    from .farm import pods, volume
    t0 = time.time()
    table = write_table()
    items = [(SOURCE / c["file"], f"src/{CAPTURE}/{c['file']}") for c in table.values()]
    if with_poses:
        for p in gzip_poses(WORK / "cameras" / "upload_stage"):
            items.append((p, f"src/{CAPTURE}/poses/{p.name}"))
    total = sum(p.stat().st_size for p, _ in items)
    log(f"{len(items)} files, {total / 1e6:.0f} MB to src/{CAPTURE}/")
    created = None
    pod = None
    try:
        if link is None:
            pod, created = pods.gateway(minutes=60, vcpu=2, reuse=False, owner="R2-recap-cameras")
            link = volume.Link(pod)
            log(f"gateway pod {pod.id} ({pod.gpu}, ${pod.cost_per_hr:.3f}/h)")
        last = {}

        def progress(rel, done, size):
            now = time.time()
            if now - last.get(rel, 0) > 20:
                last[rel] = now
                log(f"  {Path(rel).name}: {done / 1e6:.0f}/{size / 1e6:.0f} MB ({done / max(1e-9, now - t0) / 1e6:.2f} MB/s overall since start)")

        res = []
        for p, rel in sorted(items, key=lambda x: x[0].stat().st_size):
            r = link.push_file(p, rel, progress=progress)
            res.append(r)
            log(f"{rel}: {'already there' if r['skipped'] else 'sent'} {r['bytes'] / 1e6:.0f} MB in {r['seconds']} s")
        manifest = {"capture": CAPTURE, "uploaded": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "files": [{"path": rel, "bytes": p.stat().st_size, "sha256": volume.sha256_file(p)} for p, rel in items],
                    "note": "videos as filmed; poses/*.json.gz = the owner's per-video 2D pose JSONs, gzip level 6 (display pixels)"}
        link.write_text(f"src/{CAPTURE}/UPLOAD.json", json.dumps(manifest, indent=1))
    finally:
        if created and pod is not None and not keep_gateway:
            try:
                pods.delete(pod.id, reason="recap upload done")
                log(f"gateway pod {pod.id} deleted")
            except Exception as e:  # noqa: BLE001
                log(f"WARNING: gateway pod {pod.id} NOT confirmed deleted: {e}")
    out = {"files": len(items), "bytes": total, "seconds": round(time.time() - t0, 1), "results": res}
    (WORK / "cameras").mkdir(parents=True, exist_ok=True)
    (WORK / "cameras" / "upload_result.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------ jobs (farm worker specs)

WORLD_DANCE = "larissa_kadu_full"
WORLD_ID = "612dfea3d607"
MODELS = {"00": "main", "01": "main", "02": "main", "03": "main", "04": "main", "05": "sub_05", "06": "sub_06"}
CAMS_REL = "dancecap/work/larissa_kadu_recap/cameras"            # workspace-relative (the pod's /opt/hm/ws)
STATE = WORK / "cameras" / "run_state.json"
CAM_DIR = WORK / "cameras"


def world_info() -> dict:
    m = json.loads((PACKAGE_ROOT / "worlds" / WORLD_DANCE / WORLD_ID / "manifest.json").read_text())
    return {"world_id": m["world_id"], "world_sha256": m["world_sha256"], "pose_id": m["pose_id"]}


def camera_chunks(cam: dict, windows: list[dict]) -> list[int]:
    """Chunk indices in which this camera records (master clock); a camera without a clock = one chunk (its whole video)."""
    if cam.get("offset") is None:
        return [0]
    t0 = cam["offset"]
    t1 = cam["slope"] * cam["duration"] + cam["offset"]
    return [w["k"] for w in windows if w["core"][1] > t0 + 1.0 and w["core"][0] < t1 - 1.0]


def job_specs(cams: list[str], stride: int, code: dict | None = None, only_chunks: list[int] | None = None,
              calib_only: bool = False, max_anchors: int | None = None, models: dict | None = None, mode: str = "camera",
              procs: int = 3, reuse_calib: bool = False, job_suffix: str = "") -> list[dict]:
    """dancecap.farm.job/1 specs. mode "camera" (default): ONE job per camera = calibration + its 60 s chunks tracked `procs` at
    a time on the pod's GPU (newcam run); mode "chunk": a calibration job per camera, then one track job per (camera, chunk)."""
    from .cameras import newcam
    from .farm import worker
    table = json.loads(CAMS_JSON.read_text())["cams"]
    wi = world_info()
    windows = newcam.chunk_windows()
    models = {**MODELS, **(models or {})}
    specs = []
    for c in cams:
        cam = table[c]
        model = models[c]
        cj = f"{CAMS_REL}/recap_cams.json"
        env_cmd = "--world $HM_WORLD_DIR"
        cal_steps = [{"name": "calib", "cmd": f"python -m dancecap.cameras.newcam calib {c} --model {model} {env_cmd}"
                      + (f" --max-anchors {max_anchors}" if max_anchors else ""), "timeout_min": 60}]
        if mode == "camera" and not calib_only:
            ks = [k for k in camera_chunks(cam, windows) if only_chunks is None or k in only_chunks]
            steps = [{"name": "run", "cmd": f"python -m dancecap.cameras.newcam run {c} --model {model} {env_cmd} --stride {stride} "
                      f"--procs {procs} --chunks {','.join(str(k) for k in ks)}"
                      + (f" --max-anchors {max_anchors}" if max_anchors else ""), "timeout_min": 120}]
            outs = [{"path": f"{CAMS_REL}/calib/{c}.json"}] + [{"path": f"{CAMS_REL}/chunks/{c}_{k}.npz"} for k in ks]
            ins = [{"path": cj, "local": str(CAMS_JSON)}]
            cal_local = CAM_DIR / "calib" / f"{c}.json"
            if reuse_calib and cal_local.exists():          # a finished calibration is an input: the job tracks only
                ins.append({"path": f"{CAMS_REL}/calib/{c}.json", "local": str(cal_local)})
            spec = worker.analyse_spec(f"rc-cam-{c}{job_suffix}", WORLD_DANCE, steps, outs, inputs=ins,
                                       max_minutes=120, priority=30 if model == "main" else 10, gpu=True, kind="front",
                                       world_id=wi["world_id"], world_sha256=wi["world_sha256"])
            spec.update({"pose_id": wi["pose_id"], "code": code or {}, "retries": 2, "min_vram_gb": 10})
            specs.append(spec)
            continue
        cid = f"rc-calib-{c}"
        spec = worker.analyse_spec(cid, WORLD_DANCE, cal_steps, [{"path": f"{CAMS_REL}/calib/{c}.json"}],
                                   inputs=[{"path": cj, "local": str(CAMS_JSON)}], max_minutes=60, priority=50, gpu=True, kind="front",
                                   world_id=wi["world_id"], world_sha256=wi["world_sha256"])
        spec.update({"pose_id": wi["pose_id"], "code": code or {}, "retries": 2, "min_vram_gb": 10})
        specs.append(spec)
        if calib_only:
            continue
        for k in camera_chunks(cam, windows):
            if only_chunks is not None and k not in only_chunks:
                continue
            tid = f"rc-trk-{c}-{k}"
            steps = [{"name": "track", "cmd": f"python -m dancecap.cameras.newcam track {c} {k} --model {model} {env_cmd} --stride {stride}",
                      "timeout_min": 90}]
            t = worker.analyse_spec(tid, WORLD_DANCE, steps, [{"path": f"{CAMS_REL}/chunks/{c}_{k}.npz"}],
                                    after=[cid], inputs=[{"path": cj, "local": str(CAMS_JSON)},
                                                         {"path": f"chunks/{cid}/out/result.tar", "archive": True}],
                                    max_minutes=90, priority=10, gpu=True, kind="front",
                                    world_id=wi["world_id"], world_sha256=wi["world_sha256"])
            t.update({"pose_id": wi["pose_id"], "code": code or {}, "retries": 2, "min_vram_gb": 10})
            specs.append(t)
    return specs


def n_frames(cam: dict, stride: int) -> int:
    """Localised frames of one camera over the take window at the given stride."""
    if cam.get("offset") is None:
        return int(cam["duration"] * min(30.0, cam["avg_fps"]) / stride)
    t0 = max(0.0, cam["offset"])
    t1 = min(360.48, cam["slope"] * cam["duration"] + cam["offset"])
    return int(max(0.0, t1 - t0) * min(30.0, cam["avg_fps"]) / stride)


def estimate(cams: list[str], stride: int, s_per_frame: float = 0.17, price_hr: float = 0.89, calib_min: float = 3.5,
             overhead_s: float = 60.0, procs: float = 2.0) -> dict:
    """GPU-hours and dollars of a run: localised frames x seconds per frame + calibration + per-job overhead (pod-side load)."""
    from .cameras import newcam
    table = json.loads(CAMS_JSON.read_text())["cams"]
    windows = newcam.chunk_windows()
    frames = sum(n_frames(table[c], stride) for c in cams)
    n_j = sum(len(camera_chunks(table[c], windows)) + 1 for c in cams)
    gpu_s = frames * s_per_frame / max(1, procs) + len(cams) * calib_min * 60 + len(cams) * overhead_s
    return {"cams": cams, "stride": stride, "localised_frames": frames, "jobs": len(cams), "chunks": n_j - len(cams),
            "s_per_frame_assumed": s_per_frame, "parallel_procs_assumed": procs,
            "gpu_hours": round(gpu_s / 3600, 2), "usd": round(gpu_s / 3600 * price_hr, 2), "price_hr": price_hr}


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {"pods": {}, "jobs": {}, "pulled": []}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str))
    tmp.replace(STATE)


def poll(link, ids: list[str]) -> dict[str, str]:
    """{job id: queued | claimed | running | done | failed} in ONE ssh call."""
    if not ids:
        return {}
    cmd = ("cd /workspace/hm/chunks 2>/dev/null; for id in " + " ".join(ids) + "; do "
           "if [ -f $id/done.json ]; then echo \"$id done\"; elif [ -f $id/failed.json ]; then echo \"$id failed\"; "
           "elif [ -f $id/heartbeat ]; then echo \"$id running\"; elif [ -d $id/claim.lock ]; then echo \"$id claimed\"; "
           "else echo \"$id queued\"; fi; done")
    out = link.sh(cmd, 120)
    return {ln.split()[0]: ln.split()[1] for ln in out.splitlines() if len(ln.split()) == 2}


def _pod_alive(pid: str) -> bool:
    from . import remote
    d = remote.listed_pods()
    return True if d is None else pid in d


def run_jobs(specs: list[dict], n_pods: int, minutes: float, hard_minutes: float, max_cost: float, gpus: list[str] | None = None,
             poll_s: float = 30.0, keep_pods: bool = False, log_fn=log) -> dict:
    """Queue `specs`, start GPU pods, wait for every job, pull each finished job's outputs at once, delete the pods.

    Pods created here are recorded in cameras/run_state.json and deleted in the `finally` (and by their own kill switch at
    `minutes` / `hard_minutes` at the latest). Returns {job id: status}."""
    from .farm import pods, volume
    st = load_state()
    ids = [s["id"] for s in specs]
    gw = None
    mine: list[str] = []
    try:
        gw, _ = pods.gateway(minutes=45, vcpu=2, reuse=False, owner="R2-recap-cameras")
        mine.append(gw.id)
        st["pods"][gw.id] = {"role": "gateway", "started": time.time(), "price_hr": gw.cost_per_hr}
        save_state(st)
        link = volume.Link(gw)
        log_fn(f"gateway {gw.id}; syncing world and code")
        volume.sync_world(link, WORLD_DANCE, WORLD_ID)
        info = volume.sync_code(link, make_current=False)
        code = {"git_sha": info.get("git_sha"), "dirty": info.get("dirty"), "bundle_sha256": info.get("bundle_sha256")}
        for sp in specs:
            sp["code"] = code
        for sp in specs:
            volume.submit_job(link, sp)
            st["jobs"][sp["id"]] = {"status": "queued", "queued": time.time()}
        save_state(st)
        log_fn(f"{len(specs)} jobs queued; code {info['key']}")
        launched = pods.launch(n_pods, gpus=gpus, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds="front",
                               idle_min=8, prefix="hm-farm-rcam", owner="R2-recap-cameras")
        for l in launched:
            if l.get("pod_id"):
                mine.append(l["pod_id"])
                st["pods"][l["pod_id"]] = {"role": "gpu", "gpu": l.get("gpu"), "price_hr": l.get("cost_per_hr"), "started": time.time(),
                                           "kill_at": l.get("kill_at")}
            else:
                log_fn(f"launch problem: {l}")
        save_state(st)
        if not any(l.get("pod_id") for l in launched):
            raise RuntimeError(f"no GPU pod could be started: {launched}")
        pulled: set[str] = set(st.get("pulled", []))
        t_last = 0.0
        while True:
            status = poll(link, ids)
            for jid, stt in status.items():
                st["jobs"].setdefault(jid, {})["status"] = stt
            for j in [j for j, stt in status.items() if stt == "done" and j not in pulled]:
                try:
                    r = volume.pull_result(link, j, PACKAGE_ROOT.parent)
                    pulled.add(j)
                    log_fn(f"pulled {j}: {r['files']} files {r['bytes'] / 1e6:.1f} MB")
                except Exception as e:  # noqa: BLE001
                    log_fn(f"pull {j} failed: {e}")
            st["pulled"] = sorted(pulled)
            save_state(st)
            counts = {k: sum(1 for v in status.values() if v == k) for k in ("queued", "claimed", "running", "done", "failed")}
            if time.time() - t_last > 120:
                t_last = time.time()
                log_fn(f"jobs {counts}")
            if all(status.get(j) in ("done", "failed") for j in ids) and all(j in pulled or status.get(j) == "failed" for j in ids):
                return status
            live = [p for p in mine if p != gw.id]
            if live and not any(_pod_alive(p) for p in live):
                raise RuntimeError("every GPU pod of this run is gone while jobs are still open")
            time.sleep(poll_s)
    finally:
        if not keep_pods:
            for pid in reversed(mine):
                try:
                    pods.delete(pid, reason="recap cameras run finished")
                    log_fn(f"pod {pid} deleted")
                    st["pods"].get(pid, {})["deleted"] = time.time()
                except Exception as e:  # noqa: BLE001
                    log_fn(f"WARNING pod {pid} NOT confirmed deleted: {e}")
            save_state(st)


# ------------------------------------------------------------------------------------------------ merge / report / review (PC)

CAM_DIR = WORK / "cameras"
WORLD_DATA = PACKAGE_ROOT / "work" / "worlds" / WORLD_DANCE / WORLD_ID
REVIEW = WORK / "review" / "cameras"


def world_frame() -> dict:
    return json.loads((WORLD_DATA / "ref" / "cameras" / "world_from_sfm.json").read_text())


def merge_all(cams: list[str], log_fn=log) -> dict:
    """chunks/<cam>_<k>.npz -> cameras/<cam>.npz (+ _world.npz for the main-model cameras) and a quality record per camera.
    Cameras of a sub-model (05, 06) are exported in their SUB frame (attached = False, frame = sub_NN: not the room frame)."""
    from .cameras import newcam_report as R
    table = json.loads(CAMS_JSON.read_text())["cams"]
    world = world_frame()
    out = {}
    for c in cams:
        files = sorted((CAM_DIR / "chunks").glob(f"{c}_*.npz"))
        cal_p = CAM_DIR / "calib" / f"{c}.json"
        if not files or not cal_p.exists():
            out[c] = {"cam": c, "file": table[c]["file"], "verdict": "unusable", "why": ["no localisation output"], "frames": 0}
            probe = CAM_DIR / "tony_probe"
            if c == "04" and probe.is_dir():       # evidence: 19 anchors of the clip against every map model (cameras/newcam calib)
                ev = {}
                for m in ("main", "sub_05", "sub_06"):
                    p = probe / m / "04.json"
                    if p.exists():
                        d = json.loads(p.read_text())
                        a = d["anchors"]
                        ev[m] = {"anchors": d["n_anchors"], "localised": d["n_ok"], "inliers_max": max([x["n_inliers"] for x in a] or [0]),
                                 "best_reference_correspondences_max": max([x["top_counts"][0] for x in a if x["top_counts"]] or [0]),
                                 "map_images": 114 if m == "main" else 19}
                out[c]["map_registration_evidence"] = ev
                out[c]["why"] = ["the clip does not register in the room model: " + "; ".join(
                    f"{m}: {e['localised']}/{e['anchors']} anchors, at most {e['inliers_max']} inliers" for m, e in ev.items())
                    + " (a registration needs >= 40; the clip faces the logo wall from the banner side and shares no view with the demo's images)",
                    "not placed on the master clock (R1: its speech matches no other phone; its audience is a few standing people, not the seated crowd)",
                    "an own sub-model would still need a time (people) link to the room frame: not built"]
            continue
        calib = json.loads(cal_p.read_text())
        chunks = []
        for p in files:
            d = R.load_chunk(p)
            d["chunk"] = int(d["chunk"]) if "chunk" in d else int(p.stem.split("_")[1])
            chunks.append(d)
        merged = R.merge_chunks(chunks)
        in_room = calib.get("model") == "main"
        # a sub-model camera is NOT in the room frame: its track goes to <cam>_subframe.npz and cameras/<cam>.npz is written only by
        # dancecap.cameras.attach_people (a downstream reader of cameras/<cam>.npz must never get a pose in the wrong frame)
        R.export_track(table[c], calib, merged, CAM_DIR, world if in_room else None, attached=False,
                       frame_name="sfm" if in_room else str(calib.get("model")), stem=c if in_room else f"{c}_subframe")
        q = R.camera_quality(table[c], calib, merged, world if in_room else None, scale_m=float(world["s"]) if in_room else None)
        q.update(R.verdict(q, in_world=in_room))
        q["chunks"] = {str(d["chunk"]): {"frames": int(len(d["n"])), "seconds": d["stats"].get("seconds"),
                                         "stats": {k: v for k, v in d["stats"].items() if k != "refine"}} for d in chunks}
        q["frame_of_poses"] = "room model SfM frame (+ _world.npz metric world frame)" if in_room else f"{calib.get('model')} own frame (NOT the room frame)"
        att_p = CAM_DIR / f"attach_{c}.json"
        if not in_room and att_p.exists():
            att = json.loads(att_p.read_text())
            fin = att["final"]
            k2 = [k for k in fin if k.startswith("share_under_")][-1]
            q["attach"] = {k: att[k] for k in ("method", "views", "threshold_px", "prior", "final", "random_pose_null", "sensitivity_share",
                                               "refinement_adopted", "centre_world_m", "rotation_vs_prior_deg", "centre_shift_m")}
            q["verdict"] = "partial"
            q["why"] = [f"attached to the room frame by ROTATION ONLY (one fixed centre {np.round(att['centre_world_m'], 2).tolist()} m, the demo's geometry): "
                        f"{100 * fin[k2]:.0f} % of the visible 3D people (triangulated by the main cameras) project within {k2.split('_')[2]} of a detected person "
                        f"(median {fin['median_px']} px), random pose {100 * att['random_pose_null'][k2]:.0f} %; yaw uncertainty 2-4 deg, centre +-0.3 m: "
                        "usable only as a low-weight view (attached = True)"]
            q["frame_of_poses"] = "room model SfM frame, ROTATION-ONLY attach (attached = True); the raw sub-model-frame track is <cam>_subframe.npz"
        out[c] = q
        log_fn(f"{c}: {q['frames_ok']}/{q['frames']} frames ok ({100 * q['ok_fraction']:.1f} %), verdict {q['verdict']} {q.get('why')}")
    shutil.copyfile(WORLD_DATA / "ref" / "cameras" / "world_from_sfm.json", CAM_DIR / "world_from_sfm.json")
    return out


def coverage_report(quality: dict, window=(0.0, 360.48)) -> dict:
    """Per second of the master timeline: which cameras have an ok frame (from cameras/<cam>.npz)."""
    t_s = np.arange(window[0], window[1], 1.0)
    cov = {}
    for c in quality:
        p = CAM_DIR / f"{c}.npz"
        if not p.exists() or quality[c].get("frames", 0) == 0:
            continue
        z = np.load(p, allow_pickle=True)
        t, ok = z["t"], z["ok"]
        if not np.isfinite(t).any():
            continue
        row = np.zeros(len(t_s), bool)
        for i, s in enumerate(t_s):
            m = (t >= s) & (t < s + 1.0)
            row[i] = bool(m.any() and ok[m].mean() >= 0.5)
        cov[c] = row
    n_cams = sum(cov.values()) if cov else np.zeros(len(t_s), int)
    rep = {"seconds": len(t_s), "per_camera_fraction": {c: round(float(r.mean()), 3) for c, r in cov.items()},
           "cameras_per_second_counts": {str(k): int((n_cams == k).sum()) for k in range(0, len(cov) + 1)},
           "seconds_with_3_or_more": int((n_cams >= 3).sum()), "seconds_with_less_than_2": int((n_cams < 2).sum()),
           "first_second_with_3": float(t_s[np.argmax(n_cams >= 3)]) if (n_cams >= 3).any() else None}
    rep["_rows"] = {c: r for c, r in cov.items()}
    return rep


def coverage_png(rep: dict, out: Path, window=(0.0, 360.48)) -> None:
    import cv2
    rows = rep.get("_rows", {})
    if not rows:
        return
    n = int(window[1] - window[0])
    W, H, pad = 1500, 38, 70
    img = np.full((H * (len(rows) + 1) + 30, W + pad + 10, 3), 25, np.uint8)
    cams = sorted(rows)
    for i, c in enumerate(cams):
        y = 10 + i * H
        cv2.putText(img, f"cam {c}", (4, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for s in range(n):
            x0, x1 = pad + int(s * W / n), pad + int((s + 1) * W / n)
            img[y:y + H - 8, x0:max(x1, x0 + 1)] = (60, 170, 60) if rows[c][s] else (50, 50, 160)
    y = 10 + len(cams) * H
    cnt = sum(rows.values())
    cv2.putText(img, "n cams", (4, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for s in range(n):
        x0, x1 = pad + int(s * W / n), pad + int((s + 1) * W / n)
        v = int(cnt[s])
        shade = int(min(255, 60 + 30 * v))
        img[y:y + H - 8, x0:max(x1, x0 + 1)] = (shade, shade, shade)
    for s in range(3, n, 15):                                  # the counts, after every cell is filled
        x0 = pad + int(s * W / n)
        v = int(cnt[s])
        cv2.putText(img, str(v), (x0, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if v >= 4 else (255, 255, 255), 2, cv2.LINE_AA)
    for s in range(0, n + 1, 60):
        x = pad + int(s * W / n)
        cv2.line(img, (x, 5), (x, img.shape[0] - 22), (200, 200, 200), 1)
        cv2.putText(img, f"{s}s", (x - 10, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)


def review_all(cams: list[str], log_fn=log) -> dict:
    """Contact sheets (8 frames spread over the class, room model drawn in) per camera -> review/cameras/<cam>_contact.jpg."""
    from .cameras import newcam_report as R
    from .cameras.newcam import MapRefs
    table = json.loads(CAMS_JSON.read_text())["cams"]
    world = world_frame()
    maps: dict[str, object] = {}
    res = {}
    for c in cams:
        p = CAM_DIR / f"{c}.npz"
        if not p.exists():
            continue
        tr = dict(np.load(p, allow_pickle=True))
        model = "main" if bool(tr.get("attached", False)) or str(tr.get("frame", "sfm")) == "sfm" else MODELS[c]
        if model not in maps:
            maps[model] = MapRefs(WORLD_DATA, model)
        mp = maps[model]
        res[c] = R.contact_sheet(table[c], SOURCE / table[c]["file"], tr, mp.points, mp.rgb, world if model == "main" else None,
                                 REVIEW / f"{c}_contact.jpg")
        log_fn(f"{c}: contact sheet {REVIEW / (c + '_contact.jpg')} at {np.round(res[c], 1).tolist()}")
    return res


def epipolar_all(pairs: list[tuple[str, str]], n_times: int = 40, log_fn=log) -> dict:
    """Identity-free epipolar consistency of camera pairs from the owner's 2D pose JSONs (see newcam_report.epipolar_check)."""
    from .cameras import newcam, newcam_report as R
    poses: dict[str, list] = {}
    tracks: dict[str, dict] = {}
    out = {}
    for a, b in pairs:
        for c in (a, b):
            if c not in tracks:
                tracks[c] = dict(np.load(CAM_DIR / f"{c}.npz", allow_pickle=True))
            if c not in poses:
                table = json.loads(CAMS_JSON.read_text())["cams"]
                poses[c] = newcam.load_pose_frames(SOURCE / f"{table[c]['file']}.json")
        ta, tb = tracks[a], tracks[b]
        lo = max(np.nanmin(ta["t"]), np.nanmin(tb["t"])) + 1.0
        hi = min(np.nanmax(ta["t"]), np.nanmax(tb["t"])) - 1.0
        times = np.linspace(lo, hi, n_times)
        r = R.epipolar_check(ta, tb, poses[a], poses[b], times)
        out[f"{a}-{b}"] = r
        log_fn(f"epipolar {a}-{b}: share of persons under 6 px: real {r['real']['under_6px']} (n {r['real']['n_persons']}), "
               + ", ".join(f"{k} {v['under_6px']}" for k, v in r.items() if k != "real"))
    return out


def world_check(pairs: list[tuple[str, str]], n_times: int = 60, log_fn=log) -> dict:
    """Independent check of the main cameras AND the metric world frame: people matched between two cameras (epipolar cost < 6 px, no
    identities) are triangulated in the world frame; their ankles must stand on the floor (world y = 0) and the standing ones must be
    person-sized. The poses come from static features only, so this is not circular."""
    from .cameras import attach_people as ap, newcam
    table = json.loads(CAMS_JSON.read_text())["cams"]
    tracks, people, out = {}, {}, {}
    for a, b in pairs:
        for c in (a, b):
            if c not in tracks:
                tracks[c] = dict(np.load(CAM_DIR / f"{c}_world.npz", allow_pickle=True))
                people[c] = newcam.load_pose_frames(SOURCE / f"{table[c]['file']}.json")
        A, B = tracks[a], tracks[b]
        lo = max(np.nanmin(A["t"]), np.nanmin(B["t"])) + 1
        hi = min(np.nanmax(A["t"]), np.nanmax(B["t"])) - 1
        rows = []
        for t in np.linspace(lo, hi, n_times):
            ia, ib = int(np.argmin(np.abs(A["t"] - t))), int(np.argmin(np.abs(B["t"] - t)))
            if not (A["ok"][ia] and B["ok"][ib]):
                continue
            rows += ap.triangulation_stats((A["R"][ia], A["tvec"][ia], A["K"][ia], float(A["dist"][ia])),
                                           (B["R"][ib], B["tvec"][ib], B["K"][ib], float(B["dist"][ib])),
                                           people[a][int(A["frame_index"][ia])], people[b][int(B["frame_index"][ib])], 6.0)
        s = ap.summarise_people(rows)
        stand = [r_ for r_ in rows if r_["height"] > 1.35]
        s["standing_persons"] = len(stand)
        s["standing_height_median_m"] = round(float(np.median([r_["height"] for r_ in stand])), 3) if stand else None
        out[f"{a}-{b}"] = s
        log_fn(f"people on the floor {a}-{b}: {s}")
    return out


def finalize(cams: list[str], epi_pairs: list[tuple[str, str]] | None = None, review: bool = True, log_fn=log) -> dict:
    """merge -> cameras/<cam>.npz + report.json (quality, verdicts, coverage, epipolar checks, spend) + review images."""
    quality = merge_all(cams, log_fn)
    cov = coverage_report(quality)
    coverage_png(cov, REVIEW / "coverage.png")
    rows = cov.pop("_rows", None)
    epi = epipolar_all(epi_pairs, log_fn=log_fn) if epi_pairs else {}
    wchk = world_check(epi_pairs, log_fn=log_fn) if epi_pairs else {}
    if review:
        review_all([c for c in cams if quality.get(c, {}).get("frames", 0)], log_fn)
    rep = {"take": TAKE, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "world": world_info(), "cameras": quality, "coverage": cov,
           "epipolar": epi, "people_on_floor": wchk,
           "interpolation_check": json.loads((CAM_DIR / "interpolation_check.json").read_text()) if (CAM_DIR / "interpolation_check.json").exists() else None,
           "settings": {"stride": 2, "chunk_s": 60.08, "overlap_s": 4.0, "long_side": 1296,
                                         "model": "fixed focal + k1 per camera, per-frame 6-dof pose, handheld"}}
    (CAM_DIR / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def job_log(jid: str, n: int = 25) -> str:
    """Tail of a job's step logs + its heartbeat, through a running pod of this run (cameras/run_state.json)."""
    from .farm import pods, volume
    st = load_state()
    cands = [k for k, v in st.get("pods", {}).items() if not v.get("deleted")]
    last = None
    for pid in sorted(cands, key=lambda k: st["pods"][k].get("role") != "gateway"):
        try:
            link = volume.Link(pods.attach(pid))
            out = link.sh(f"cd /workspace/hm/chunks/{jid} 2>/dev/null && (for f in log/*.log; do echo \"== $f\"; tail -n {n} $f; done; "
                          f"echo == heartbeat; cat heartbeat 2>/dev/null; echo; echo == done; cat done.json 2>/dev/null | head -c 1500; "
                          f"cat failed.json 2>/dev/null | head -c 1500)", 90)
            return out
        except Exception as e:  # noqa: BLE001
            last = e
    return f"no pod could be read: {last}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.recap_cameras", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload")
    u.add_argument("--keep-gateway", action="store_true")
    u.add_argument("--no-poses", action="store_true")
    sub.add_parser("table")
    fz = sub.add_parser("finalize")
    fz.add_argument("--cams", default="00,01,02,03,05,06,04")
    fz.add_argument("--epipolar", default="", help="pairs like 00-01,01-02")
    fz.add_argument("--no-review", action="store_true")
    jl = sub.add_parser("log")
    jl.add_argument("job")
    jl.add_argument("-n", type=int, default=25)
    for nm in ("plan", "run"):
        r = sub.add_parser(nm)
        r.add_argument("--cams", default="00,01,02,03", help="comma list of camera ids")
        r.add_argument("--stride", type=int, default=2)
        r.add_argument("--chunks", default=None, help="comma list of chunk indices (default: all the camera records in)")
        r.add_argument("--calib-only", action="store_true")
        r.add_argument("--max-anchors", type=int, default=None)
        r.add_argument("--s-per-frame", type=float, default=0.17, help="assumed seconds per localised frame (estimate)")
        r.add_argument("--mode", default="camera", choices=["camera", "chunk"])
        r.add_argument("--reuse-calib", action="store_true", help="upload an existing cameras/calib/<cam>.json and skip the calibration")
        r.add_argument("--suffix", default="", help="job id suffix (a second run of the same camera)")
        r.add_argument("--procs", type=int, default=3)
        if nm == "run":
            r.add_argument("--pods", type=int, default=1)
            r.add_argument("--gpus", default="4090", help="comma list of GPU priority (substring of the Runpod name)")
            r.add_argument("--minutes", type=float, default=60)
            r.add_argument("--hard-minutes", type=float, default=90)
            r.add_argument("--max-cost", type=float, default=2.5)
            r.add_argument("--keep-pods", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "upload":
        upload(a.keep_gateway, not a.no_poses)
    elif a.cmd == "finalize":
        pairs = [tuple(x.split("-")) for x in a.epipolar.split(",") if x]
        finalize([c for c in a.cams.split(",") if c], pairs, not a.no_review)
    elif a.cmd == "log":
        print(job_log(a.job, a.n))
    elif a.cmd in ("plan", "run"):
        cams = [c.strip() for c in a.cams.split(",") if c.strip()]
        chunks = [int(x) for x in a.chunks.split(",")] if a.chunks else None
        est = estimate(cams, a.stride, a.s_per_frame, procs=min(a.procs, 2.5))
        specs = job_specs(cams, a.stride, only_chunks=chunks, calib_only=a.calib_only, max_anchors=a.max_anchors, mode=a.mode, procs=a.procs, reuse_calib=a.reuse_calib, job_suffix=a.suffix)
        est["jobs_planned"] = [s["id"] for s in specs]
        print(json.dumps(est, indent=1))
        if a.cmd == "run":
            from .farm import pods as _pods
            gp = {"4090": "NVIDIA GeForce RTX 4090", "l4": "NVIDIA L4", "a4500": "NVIDIA RTX A4500"}
            gpus = [gp.get(g.strip().lower(), g.strip()) for g in a.gpus.split(",")]
            res = run_jobs(specs, a.pods, a.minutes, a.hard_minutes, a.max_cost, gpus=gpus, keep_pods=a.keep_pods)
            print(json.dumps(res, indent=1))
    elif a.cmd == "table":
        t = write_table()
        for c, r in t.items():
            print(c, r["file"], f"{r['width']}x{r['height']}", f"rot {r['rotation']}", f"{r['avg_fps']:.2f} fps", r["offset"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
