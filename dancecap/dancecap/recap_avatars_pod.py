"""Pod side of the recap avatars (agent A1): job specs of kind `avatar`, the one-job-one-pod runner and the steps a pod executes.

Design (the farm's, FARM.md 5-6; a dedicated queue is NOT needed because the kind is dedicated):
  * ONE job spec (dancecap.farm.job/1) of kind `avatar`, id `ra-<stage>-<n>`; no farm worker claims that kind (the fit pods run
    --kinds front,fit,refine,full,asr; the analysis CPU pods --kinds analyse), only the pod THIS module starts (HM_KINDS=avatar).
  * inputs: ONE deterministic tar of the small inputs (landmarks, face frames, portraits, betas), the big shared files live on the
    volume (models/flame, models/smirk, src/Larissa-Kadu-recap/...): uploaded once, hash-diffed.
  * the pod: a CPU pod (cpu5c x 8 vCPU, ~0.28 USD/h) for the identity fit (no GPU needed: ~20 images x 750 iterations of a 5k-vertex
    FLAME), a 4090 pod (0.89 USD/h) for the texture stages. Ledger owner A1-recap-avatars; every pod created here is deleted in the
    `finally`; the on-pod kill switch (HM_KILL_AT / HM_HARD_CEIL) is the backstop.
  * the code bundle is pinned by the job spec (volume.sync_code(make_current=False): the volume's code/CURRENT is never moved).

    python -m dancecap.recap_avatars face-identity [--max-cost 1.0] [--dry-run]
    python -m dancecap.recap_avatars pod-identity            (the pod-side step; reads the job's input tar)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import recap_avatars as RA
from .take import PACKAGE_ROOT, WORKSPACE

STATE = RA.AV / "pod_state.json"
POD_WS = "/opt/hm/ws"


def log(msg: str) -> None:
    print(f"[recap_avatars_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ models on the volume

def model_items() -> list[tuple[Path, str]]:
    """(local file, volume-relative path) of what only this PC has and the identity fit needs: FLAME 2020 + the SMPL-X <-> FLAME
    vertex map (gated, non-commercial research licence: the owner's private volume only), SMIRK (MIT, the code + its assets + the
    140 MB checkpoint of the README's drive link). Uploaded ONCE (hash-diffed); never into a repo."""
    fl = WORKSPACE / "models" / "flame"
    sm = WORKSPACE / "external" / "smirk"
    items = [(fl / "FLAME2020" / "generic_model.pkl", "models/flame/FLAME2020/generic_model.pkl"),
             (fl / "SMPL-X__FLAME_vertex_ids.npy", "models/flame/SMPL-X__FLAME_vertex_ids.npy")]
    for sub in ("src", "assets"):
        items += [(f, f"models/smirk/{f.relative_to(sm).as_posix()}") for f in sorted((sm / sub).rglob("*"))
                  if f.is_file() and "__pycache__" not in f.parts]
    items += [(sm / "pretrained_models" / "SMIRK_em1.pt", "models/smirk/pretrained_models/SMIRK_em1.pt"),
              (sm / "LICENSE", "models/smirk/LICENSE")]
    missing = [str(p) for p, _ in items if not p.exists()]
    if missing:
        raise FileNotFoundError(f"identity fit models missing locally: {missing}")
    return items


def ensure_models(link, progress=None) -> dict:
    return link.push_files(model_items(), progress=progress)


# ------------------------------------------------------------------------------------------------ the identity job

def identity_archive() -> dict:
    """The small inputs as ONE deterministic tar input (the uploader's parallel tar batches race on parent directories of many small
    files): landmarks_<role>.json, the chosen frames, the two portraits (copied under avatar/face_id/portrait_<role>.<ext> inside the
    tar only), betas.json (the pinned betas of the shape pass). Workspace-relative arcnames."""
    from .farm import chunks as C
    items: list[tuple[Path, str]] = []
    for role in RA.ROLES:
        lm = RA.FACE_ID / f"landmarks_{role}.json"
        items.append((lm, f"{RA.REL}/avatar/face_id/{lm.name}"))
        for r in json.loads(lm.read_text()):
            f = RA.WORK / r["file"]
            if r["name"] == "portrait":
                f = RA.portrait_path(role)
            items.append((f, f"{RA.REL}/{r['file']}"))
    betas = RA.AV / "face_id" / "betas.json"
    betas.write_text(json.dumps({r: [float(x) for x in b] for r, b in RA.pinned_betas().items()}, indent=1))
    items.append((betas, f"{RA.REL}/avatar/betas.json"))
    items = sorted({a: (s, a) for s, a in items}.values(), key=lambda x: x[1])
    dest = RA.FACE_ID / "in" / "faceid_in.tar"
    C.pack_tar(dest, items)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {"path": f"{RA.REL}/avatar/face_id/faceid_in.tar", "archive": True, "sha256": h, "bytes": dest.stat().st_size,
            "local": str(dest), "files": len(items)}


def identity_steps() -> list[dict]:
    """The worker runs steps in <ws>/dancecap with the volume's venv first on PATH and $HM_ROOT set."""
    links = ("set -e; cd ..; mkdir -p models external; ln -sfn $HM_ROOT/models/flame models/flame; "
             "ln -sfn $HM_ROOT/models/smirk external/smirk; "
             "ls models/flame/FLAME2020/generic_model.pkl external/smirk/pretrained_models/SMIRK_em1.pt > /dev/null; "
             "python -c 'import timm' 2>/dev/null || pip install -q 'timm==0.9.16'")
    return [{"name": "links", "cmd": links, "timeout_min": 8, "optional": False},
            {"name": "identity", "cmd": "python -m dancecap.recap_avatars pod-identity", "timeout_min": 40, "optional": False}]


def identity_outputs() -> list[dict]:
    rows = []
    for role in RA.ROLES:
        rows += [f"{RA.REL}/face_shape/{role}_identity.npz", f"{RA.REL}/face_shape/{role}_identity_disp.npz"]
        rows += [f"{RA.REL}/face_shape/{role}_identity_{v}.npz" for v in ("portrait", "video")]
    rows += [f"{RA.REL}/avatar/face_id/fit_report.json"]
    return [{"path": p, **({"optional": True} if p.endswith(("_portrait.npz", "_video.npz")) else {})} for p in rows]


def identity_spec(job_id: str, code: dict | None = None) -> dict:
    from .farm import worker
    inp = identity_archive()
    spec = worker.analyse_spec(job_id, RA.TAKE, identity_steps(), identity_outputs(), inputs=[inp], max_minutes=45, priority=10, gpu=False,
                               kind=RA.KIND)
    spec.update({"code": code or {}, "retries": 1, "needs_gpu": False, "env": {"DANCECAP_FARM_DANCE": RA.TAKE, "OMP_NUM_THREADS": "6"}})
    return spec


def estimate(minutes_fit: float = 12.0) -> dict:
    mins = {"pod start + env + code": 2.0, "inputs (tar, models on the volume)": 1.0, "SMIRK + landmark fits (2 teachers, ~20 images)": minutes_fit,
            "retarget (FLAME fit to the rest heads)": 1.5}
    tot = sum(mins.values())
    return {"minutes": mins, "total_min": round(tot, 1), "price_hr": RA.PRICE_CPU_8, "usd": round(tot / 60 * RA.PRICE_CPU_8, 3)}


# ------------------------------------------------------------------------------------------------ the pod-side step

def pod_identity(portrait_share: float = 0.5, work: Path | None = None, iters: int = 600, limit: int | None = None) -> dict:
    """On the pod: per teacher, the SMIRK + landmark fit of every image ONCE (fit_images), then three robust combinations of the per-image
    codes: `mixed` (the portrait carries `portrait_share` of the total weight: the shipped identity), `portrait` (the portrait alone) and
    `video` (the video frames alone: the consistency check of the two sources); the mixed identity is retargeted to the REST head of the
    pinned betas (face_identity.cached_displacement -> face_shape/<role>_identity_disp.npz, the cache the export reads).
    work / iters / limit exist for a local smoke test (a scratch work dir, a few iterations, the first `limit` images)."""
    from . import export_unity as eu
    from . import face_identity as fi
    work = work or RA.WORK
    betas = json.loads((work / "avatar" / "betas.json").read_text())
    (work / "face_shape").mkdir(parents=True, exist_ok=True)
    report: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "portrait_share": portrait_share, "roles": {}}
    for role in RA.ROLES:
        lm = json.loads((work / "avatar" / "face_id" / f"landmarks_{role}.json").read_text())
        lm = lm[:limit] if limit else lm
        for r in lm:
            r["file"] = str(work / r["file"])
        t0 = time.time()
        res = fi.fit_images(lm, device="cpu", portrait_weight=1.0, iters=iters)
        per = res["per_image"]
        n_video = sum(1 for p in per if p["name"] != "portrait")
        pw = max(1.0, portrait_share / max(1e-6, 1 - portrait_share) * n_video)
        variants = {"mixed": fi.combine(per, pw), "portrait": fi.combine([p for p in per if p["name"] == "portrait"], 1.0)}
        if n_video:
            variants["video"] = fi.combine([p for p in per if p["name"] != "portrait"], 1.0)
        files = {}
        for v, rv in variants.items():
            out = work / "face_shape" / (f"{role}_identity.npz" if v == "mixed" else f"{role}_identity_{v}.npz")
            fi.save_identity(rv, out, {"landmarks": f"landmarks_{role}.json", "variant": v, "portrait_weight": pw if v == "mixed" else None,
                                       "recap": True})
            files[v] = out.name
        b = {v: variants[v]["beta"] for v in variants}
        rep = {"images": [p["name"] for p in per], "inlier": dict(zip([p["name"] for p in per], variants["mixed"]["inlier"].tolist())),
               "fit_err": dict(zip([p["name"] for p in per], [round(float(x), 4) for x in variants["mixed"]["fit_err"]])),
               "dist_to_median": dict(zip([p["name"] for p in per], [round(float(x), 3) for x in variants["mixed"]["dist_to_median"]])),
               "portrait_weight": pw, "beta_norm": {v: round(float(np.linalg.norm(x)), 3) for v, x in b.items()},
               "beta_distance": {f"{a}-{c}": round(float(np.linalg.norm(b[a] - b[c])), 3) for a in b for c in b if a < c},
               "files": files, "seconds_fit": round(time.time() - t0, 1)}
        sk = eu.shaped_skin(eu.model_for({"model": "smplx_locked_head"}), np.asarray(betas[role], float))
        t1 = time.time()
        _, _, rrep = fi.cached_displacement(sk.vertices, work / "face_shape" / f"{role}_identity.npz", work, role, 1.0)
        rep["retarget"] = {k: rrep.get(k) for k in ("equivalent_rms_mm", "equivalent_max_mm", "stable_residual_mm", "max_move_mm",
                                                    "occ_mm_head_vs_identity", "cache_hit")}
        rep["retarget"]["seconds"] = round(time.time() - t1, 1)
        report["roles"][role] = rep
        print(json.dumps({role: rep}, default=float), flush=True)
    (work / "avatar" / "face_id" / "fit_report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


# ------------------------------------------------------------------------------------------------ one job, one pod

def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {"pods": {}, "jobs": {}, "pulled": []}


def save_state(st: dict) -> None:
    """Merge into the file on disk (parallel texture jobs, one process each, share it: pods / jobs by key, `pulled` as a union)."""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    disk = load_state()
    merged = {"pods": {**disk.get("pods", {}), **st.get("pods", {})}, "jobs": {**disk.get("jobs", {}), **st.get("jobs", {})},
              "pulled": sorted(set(disk.get("pulled", [])) | set(st.get("pulled", [])))}
    tmp = STATE.with_name(f"{STATE.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(merged, indent=1, default=str))
    tmp.replace(STATE)


def run_job(spec: dict, *, cpu: bool, vcpu: int = 8, gpus: list[str] | None = None, minutes: float = 40, hard_minutes: float = 70,
            max_cost: float = 1.0, upload_models: bool = False, poll_s: float = 30.0, log_fn=log, kind: str | None = None) -> dict:
    """Queue `spec` on the volume, start ONE pod of kind `avatar` (CPU or GPU), wait, pull the result, delete every pod this call
    created (also on an error). Budget gate: pods.launch / launch_cpu refuse when the worst case (price x hard minutes) exceeds
    max_cost, the account balance - 2 USD or the ledger cap. kind: the worker kind of THIS job's pod (default `avatar`; several jobs run in
    parallel with a kind each, e.g. `avatartx3`, so a pod can only claim the job its own process started and waits for)."""
    from . import recap_cameras as rc
    from .farm import pods, volume
    st = load_state()
    mine: list[str] = []
    gw = None
    try:
        gw, _ = pods.gateway(minutes=45, vcpu=2, reuse=False, owner=RA.OWNER)   # never borrow a pod of another run
        mine.append(gw.id)
        st["pods"][gw.id] = {"role": "gateway", "started": time.time(), "price_hr": gw.cost_per_hr}
        save_state(st)
        link = volume.Link(gw)
        info = volume.sync_code(link, make_current=False)
        spec["code"] = {"git_sha": info.get("git_sha"), "dirty": info.get("dirty"), "bundle_sha256": info.get("bundle_sha256")}
        if upload_models:
            t0 = time.time()
            r = ensure_models(link)
            log_fn(f"models on the volume: {r} in {time.time() - t0:.0f} s")
        volume.submit_job(link, spec)
        st["jobs"][spec["id"]] = {"status": "queued", "queued": time.time(), "code": info.get("key")}
        save_state(st)
        kind = kind or spec.get("kind") or RA.KIND
        log_fn(f"job {spec['id']} queued (code {info['key']}); starting one {'CPU' if cpu else 'GPU'} pod of kind {kind}")
        if cpu:
            launched = pods.launch_cpu(1, vcpu=vcpu, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds=kind, idle_min=4,
                                       prefix="hm-farm-ravatar", owner=RA.OWNER)
        else:
            launched = pods.launch(1, gpus=gpus, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds=kind, idle_min=4,
                                   prefix="hm-farm-ravatar", owner=RA.OWNER)
        for l in launched:
            if l.get("pod_id"):
                mine.append(l["pod_id"])
                st["pods"][l["pod_id"]] = {"role": "cpu" if cpu else "gpu", "gpu": l.get("gpu"), "price_hr": l.get("cost_per_hr"),
                                           "started": time.time(), "kill_at": l.get("kill_at")}
            else:
                log_fn(f"launch problem: {l}")
        save_state(st)
        if not any(l.get("pod_id") for l in launched):
            raise RuntimeError(f"no pod could be started: {launched}")
        t_last = 0.0
        while True:
            status = rc.poll(link, [spec["id"]]).get(spec["id"], "queued")
            st["jobs"][spec["id"]]["status"] = status
            save_state(st)
            if time.time() - t_last > 90:
                t_last = time.time()
                log_fn(f"job {spec['id']}: {status}")
            if status == "done":
                r = volume.pull_result(link, spec["id"], PACKAGE_ROOT.parent)
                log_fn(f"pulled {spec['id']}: {r['files']} files {r['bytes'] / 1e6:.1f} MB")
                st["pulled"].append(spec["id"])
                save_state(st)
                return {**r, "status": "done"}
            if status == "failed":
                try:
                    f = link.read_json(f"chunks/{spec['id']}/failed.json")
                    log_fn("failed.json: " + json.dumps(f)[:2500])
                    tail = link.sh(f"tail -n 40 /workspace/hm/chunks/{spec['id']}/log/*.log 2>/dev/null | tail -n 60", 60)
                    log_fn("log tail:\n" + tail[-3000:])
                except Exception as e:  # noqa: BLE001
                    log_fn(f"cannot read failed.json: {e}")
                return {"status": "failed"}
            live = [p for p in mine if p != gw.id]
            if live and not any(rc._pod_alive(p) for p in live):
                raise RuntimeError("the worker pod is gone while the job is still open")
            time.sleep(poll_s)
    finally:
        for pid in reversed(mine):
            try:
                pods.delete(pid, reason="recap avatars run finished")
                log_fn(f"pod {pid} deleted")
                st["pods"].get(pid, {})["deleted"] = time.time()
            except Exception as e:  # noqa: BLE001
                log_fn(f"WARNING pod {pid} NOT confirmed deleted: {e}")
        save_state(st)


def owner_spend() -> float:
    """USD booked in the ledger under this module's owner (rows closed with a cost, plus open rows at their rate x age)."""
    from . import remote
    tot = 0.0
    for r in remote.ledger():
        if r.get("owner") == RA.OWNER:
            tot += float(r.get("cost") or 0.0)
    return round(tot, 3)


# ------------------------------------------------------------------------------------------------ the pose-dependent texture job

TEX_CHUNK = "shape"       # the dry run uses the shape pass's window (100-130 s, six phones, real recap data and a real fit)


def texture_take(chunk_id: str = TEX_CHUNK) -> str:
    return f"{RA.TAKE}_{chunk_id}"


def window_cams(t0: float, t1: float) -> tuple[list[str], dict]:
    """The phones of a texture window by the farm's own camera rule (farm.recap_profile.choose_cams on R3's teacher coverage)."""
    from .farm import recap_profile as rp
    return rp.choose_cams(rp.window_stats(rp.load_teachers(), t0, t1))


def window_job(window_id: str, t0: float, t1: float, cams: list[str] | None = None, base_id: str = TEX_CHUNK) -> dict:
    """The farm job spec of a texture window = the shape job's spec (sources, inputs, steps' shape) with the window's own chunk dict: chunk-take
    and recap_front read ONLY id / take / k0 / k1 / cams / sources from it, so the front of any window of the class runs unchanged."""
    import copy
    import math
    base = json.loads((RA.WORK / "farm" / "jobs" / f"{base_id}.json").read_text())
    job = copy.deepcopy(base)
    k0, k1 = int(math.ceil(t0 * 30.0 - 1e-6)), int(math.floor(t1 * 30.0 + 1e-6)) - 1
    cams = list(cams) if cams else window_cams(t0, t1)[0]
    take = f"{RA.TAKE}_{window_id}"
    job.update({"id": window_id, "take": take, "cams": cams, "kind": "front", "what": "texture window", "stages": ["front"], "after": []})
    job["chunk"].update({"id": window_id, "k0": k0, "k1": k1, "core_k0": k0, "core_k1": k1, "overlap_k": 0, "take": take, "cams": cams,
                         "T": k1 - k0 + 1, "core_s": round((k1 - k0 + 1) / 30.0, 2), "origin": "texture window"})
    return job


def slice_smplx(src_dir: Path, dst_dir: Path, t0: float, t1: float) -> dict:
    """work/<take>/smplx/{lead,follow}.npz cut to the master-time window [t0, t1] (every array whose first axis is the frame axis)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for r in RA.ROLES:
        z = np.load(src_dir / f"{r}.npz", allow_pickle=True)
        t = np.asarray(z["t"], float)
        m = (t >= t0 - 1e-6) & (t <= t1 + 1e-6)
        if m.sum() < 30:
            raise ValueError(f"{src_dir}/{r}.npz has {int(m.sum())} frames in [{t0:.2f}, {t1:.2f}] s")
        out = {k: (z[k][m] if (z[k].ndim >= 1 and z[k].shape[0] == len(t)) else z[k]) for k in z.files}
        np.savez_compressed(dst_dir / f"{r}.npz", **out)
        info[r] = int(m.sum())
    return info


def texture_archive(chunk_id: str = TEX_CHUNK, job_file: Path | None = None, smplx_dir: Path | None = None, sync_file: Path | None = None) -> dict:
    """The small inputs of the texture job as ONE tar: the window's farm job spec (chunk-take / recap_front read it), the fit's smplx/ and the
    sync offsets (the phones' video offsets). Defaults = the shape window's own files (the dry run); the real run passes the sliced stitched
    smplx and the offsets of the chunk the window lies in."""
    from .farm import chunks as C
    tk = texture_take(chunk_id)
    wk = PACKAGE_ROOT / "work" / tk
    job_file = job_file or RA.WORK / "farm" / "jobs" / f"{chunk_id}.json"
    smplx_dir = smplx_dir or wk / "smplx"
    sync_file = sync_file or wk / "sync" / "offsets.json"
    items = [(job_file, f"{RA.REL}/farm/jobs/{chunk_id}.json")]
    items += [(smplx_dir / f"{r}.npz", f"dancecap/work/{tk}/smplx/{r}.npz") for r in RA.ROLES]
    items += [(sync_file, f"dancecap/work/{tk}/sync/offsets.json")]
    miss = [str(src) for src, _ in items if not src.exists()]
    if miss:
        raise FileNotFoundError(f"texture job inputs missing: {miss}")
    dest = RA.AV / "texture" / "in" / f"texture_in_{chunk_id}.tar"
    C.pack_tar(dest, items)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {"path": f"dancecap/work/{tk}/texture_in.tar", "archive": True, "sha256": h, "bytes": dest.stat().st_size, "local": str(dest), "files": len(items)}


def shared_recap_inputs(base_id: str = TEX_CHUNK) -> list[dict]:
    """The rows of the chunk's farm job spec that name SHARED volume files (recap_in.tar: R2's camera tracks, R3's teacher tracks and pose
    tables; already uploaded by the farm run): copied as they are, without a `local` key (nothing is uploaded again). The videos are read
    from the volume's src/ by recap_front and need no row."""
    spec = json.loads((RA.WORK / "farm" / "jobs" / f"{base_id}.json").read_text())
    rows = [{k: v for k, v in r.items() if k != "local"} for r in spec["inputs"] if r["path"].endswith("recap_in.tar")]
    if not rows:
        raise RuntimeError(f"farm/jobs/{base_id}.json lists no recap_in.tar input")
    return rows


def texture_steps(chunk_id: str = TEX_CHUNK, step_ms: float = 100.0, hair_ref: bool = False, top_k: int = 4, body_ref_q: float | None = None,
                  mask: str | None = None) -> list[dict]:
    tk = texture_take(chunk_id)
    w = f"dancecap/work/{tk}"
    steps = [
        {"name": "chunk_take", "cmd": f"python -m dancecap.farm chunk-take {chunk_id}", "timeout_min": 3, "optional": False},
        {"name": "weights", "cmd": "python -m dancecap.front_full fetch-weights", "timeout_min": 12, "optional": True},
        {"name": "front", "cmd": f"python -m dancecap.recap_avatars pod-texture-front {chunk_id}", "timeout_min": 70, "optional": False},
        {"name": "stage", "cmd": f"cd ..; touch {w}/smplx/*.npz; ls {w}/dancers/roles.json {w}/sync/offsets.json", "timeout_min": 3, "optional": False},
    ] + ([{"name": "maskcams", "cmd": f"python -m dancecap.recap_avatars pod-mask-cams {chunk_id} --mask {mask}", "timeout_min": 3, "optional": False}] if mask else []) + [
        {"name": "bake", "cmd": f"DANCECAP_POD=1 python -m dancecap texture {tk} --preset pod --step-ms {step_ms:g} --top-k {top_k} --align-passes 2 "
                               + (f"--body-ref-q {body_ref_q:g} " if body_ref_q else "") +
                               "--hair-mode exclude --blend multiband --no-lock --allow-stale", "timeout_min": 60, "optional": False},
    ]
    if hair_ref:    # optional, NOT exercised on the recap yet: the 3D hair hull every 10th grid frame (the demo's groom is the seed otherwise)
        steps.append({"name": "hair_ref", "cmd": f"python -m dancecap.hair_reference {tk} --step 10 --device cuda --no-isolate-gpu --no-review --no-mp",
                      "timeout_min": 30, "optional": True})
    return steps


def texture_outputs(chunk_id: str = TEX_CHUNK) -> list[dict]:
    from .avatar_pod import OUTPUTS
    w = f"dancecap/work/{texture_take(chunk_id)}"
    must = {"avatar/lead_albedo.png", "avatar/follow_albedo.png", "avatar/report.json"}
    rows = [{"path": f"{w}/{o}", **({} if o in must else {"optional": True})} for o in OUTPUTS]
    rows += [{"path": f"{w}/{o}", "optional": True} for o in ("dancers/roles.json", "dancers/report.json", "front/front_done.json",
                                                              "front/front_pod.json", "frames/01/frames.json", "avatar/hair_reference.json", "avatar/hair_ref/observations.json")]
    return rows


def texture_spec(job_id: str, chunk_id: str = TEX_CHUNK, step_ms: float = 100.0, code: dict | None = None, job_file: Path | None = None,
                 smplx_dir: Path | None = None, sync_file: Path | None = None, hair_ref: bool = False, top_k: int = 4,
                 body_ref_q: float | None = None, mask: str | None = None, kind: str = RA.KIND) -> dict:
    from .farm import worker
    inputs = [texture_archive(chunk_id, job_file, smplx_dir, sync_file)] + shared_recap_inputs()
    spec = worker.analyse_spec(job_id, RA.TAKE, texture_steps(chunk_id, step_ms, hair_ref, top_k, body_ref_q, mask), texture_outputs(chunk_id), inputs=inputs,
                               max_minutes=95, priority=10, gpu=True, kind=kind)
    spec.update({"code": code or {}, "retries": 0, "needs_gpu": True, "min_vram_gb": 24,
                 "env": {"DANCECAP_FARM_DANCE": RA.TAKE, "DANCECAP_POD": "1"}})
    return spec


def pod_texture_front(chunk_id: str = TEX_CHUNK, resume: bool = False) -> int:
    """recap_front's front DAG of the chunk WITHOUT its MAMMA tail: frames, cameras, people, SAM 2.1 masks + body / hair masks of every phone
    (the teacher tracks as candidate tracks) and `merge` (dancers/roles.json) - what the texture bake reads. ma_2d / sync / prepare are
    left out: the fit already exists (smplx/ and sync/offsets.json come with the job's inputs). Same Runner, same memory budget and caps."""
    from . import front_pod as fp
    from . import recap_front as rf
    chunk = rf.load_chunk(chunk_id)
    work = rf.take_work(chunk_id)
    cpus, ram = fp.cgroup_cpus(), fp.mem_gb()
    _, limit = fp.mem_state()
    threads = str(max(2, min(4, cpus // 8)))
    import os
    env = dict(os.environ, DANCECAP_GPU_SHARED="1", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
               OPENCV_FOR_THREADS_NUM=threads, PYTHONUNBUFFERED="1", DANCECAP_POD=os.environ.get("DANCECAP_POD", "1"), YOLO_CONFIG_DIR="/tmp/Ultralytics")
    cap_gb = limit if limit else ram
    caps = {"sam": max(2, min(4, cpus // 4)), "cpu": 3}
    R = fp.Runner(work, env, max(16.0, 0.62 * cap_gb), caps, work / "front" / "logs", limit)

    class OnlyDancers:
        """build_tasks() adds ma2d / sync / prepare as well: this view of the runner drops them."""
        def add(self, t):
            return None if t.name.startswith(("ma2d:", "sync", "prepare")) else R.add(t)
    rf.build_tasks(OnlyDancers(), chunk)
    if resume:
        R.resume()
    R.meta = {"take": rf.take_name(chunk_id), "chunk": chunk_id, "cams": chunk["cams"], "purpose": "texture front (A1)",
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": {"cpus": cpus, "ram_gb": round(ram, 1), "mem_limit_gb": limit, "caps": caps}}
    R.save()
    log(f"{rf.take_name(chunk_id)}: {len(R.tasks)} tasks {sorted(R.tasks)}")
    R.loop()
    bad = [k for k, t in R.tasks.items() if t.state != "ok" and not t.optional]
    (work / "front").mkdir(parents=True, exist_ok=True)
    (work / "front" / "front_done.json").write_text(json.dumps({"rc": 0 if not bad else 1, "not_ok": bad, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                                "purpose": "texture front"}))
    log(f"done; not ok: {bad or 'none'}")
    return 0 if not bad else 1


def mask_camera_frames(work: Path, masks: list[tuple[str, float, float]]) -> dict:
    """cameras/<cam>.npz of a texture window: ok = False for the frames whose time lies in [t0, t1) of that camera, so the bake skips those
    views (a camera track that is known to be wrong there: R2 / the F3 audit; the IoU weight alone would still give such a view 5 %).
    -> {cam: frames switched off}."""
    out: dict = {}
    for cam, t0, t1 in masks:
        p = work / "cameras" / f"{cam}.npz"
        if not p.exists():
            out[cam] = out.get(cam, 0)
            continue
        with np.load(p, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        t = np.asarray(d["t"], float)
        hit = (t >= t0) & (t < t1)
        ok = np.asarray(d["ok"], bool).copy()
        n = int((ok & hit).sum())
        ok[hit] = False
        d["ok"] = ok
        np.savez(p, **d)
        out[cam] = out.get(cam, 0) + n
    return out


def pod_mask_cams(chunk_id: str, spec: str) -> int:
    from . import recap_front as rf
    work = rf.take_work(chunk_id)
    res = mask_camera_frames(work, rf.parse_masks(spec))
    (work / "avatar").mkdir(parents=True, exist_ok=True)
    (work / "avatar" / "mask_cams.json").write_text(json.dumps({"mask": spec, "frames_off": res}, indent=1))
    log(f"cameras masked {spec}: frames switched off {res}")
    return 0


def chunk_sync_file(t_mid: float) -> Path:
    """The sync offsets of the farm chunk whose CORE holds master time t_mid (work/larissa_kadu_recap_<id>/sync/offsets.json, pulled by the
    watcher); the stitched take's own sync/offsets.json when it exists."""
    own = RA.WORK / "sync" / "offsets.json"
    if own.exists():
        return own
    plan = json.loads((RA.WORK / "chunks.json").read_text())
    for c in plan.get("chunks", []):
        k0, k1 = c["core_k"]
        if k0 / 30.0 <= t_mid <= (k1 + 1) / 30.0:
            f = PACKAGE_ROOT / "work" / f"{RA.TAKE}_{c['id']}" / "sync" / "offsets.json"
            if f.exists():
                return f
    raise FileNotFoundError(f"no sync offsets for t = {t_mid:.1f} s: pass --sync (work/larissa_kadu_recap_<chunk>/sync/offsets.json of a pulled chunk)")


def texture_run(a) -> int:
    """`texture-run`: windows.json (texture-select) -> one pod job: the window's front + the bake on the sliced stitched poses."""
    doc = json.loads(a.windows.read_text())
    windows = doc["windows"]
    w = windows[a.which]
    if doc.get("smplx_dir") and a.smplx_dir == RA.WORK / "smplx":      # a pre-stitch pick: the raw fit of the chunk the window lies in
        a.smplx_dir = Path(doc["smplx_dir"])
    t0, t1 = float(w["t0"]), float(w["t1"])
    cams, why = window_cams(t0, t1)
    mask = getattr(a, "mask", None) or ",".join(w.get("mask") or []) or None   # cam:t0:t1 stretches where a camera track is wrong (windows.json "mask": [..])
    log(f"window {a.window_id}: {t0:.1f}-{t1:.1f} s, phones {cams} ({why}); masked {mask}")
    job = window_job(a.window_id, t0, t1, cams)
    jdir = RA.AV / "texture" / "jobs"
    jdir.mkdir(parents=True, exist_ok=True)
    jf = jdir / f"{a.window_id}.json"
    jf.write_text(json.dumps(job, indent=1))
    tk = texture_take(a.window_id)
    sm = PACKAGE_ROOT / "work" / tk / "smplx"
    sl = slice_smplx(a.smplx_dir, sm, t0 - 0.5, t1 + 0.5)
    log(f"stitched poses sliced: {sl} frames -> {sm}")
    sync = a.sync or chunk_sync_file(0.5 * (t0 + t1))
    jid = f"ra-texture-{a.window_id}-{time.strftime('%m%d%H%M')}"
    kind = f"{RA.KIND}{a.window_id}" if getattr(a, "own_kind", False) else RA.KIND
    spec = texture_spec(jid, a.window_id, a.step_ms, job_file=jf, smplx_dir=sm, sync_file=sync, hair_ref=a.with_hair_ref,
                        top_k=a.top_k, body_ref_q=a.body_ref_q, mask=mask, kind=kind)
    log(f"job {jid}: {len(spec['steps'])} steps, inputs {[i['path'] for i in spec['inputs']]}, sync {sync.name} of {sync.parent.parent.name}")
    if a.dry_run:
        print(json.dumps({"job": {k: spec[k] for k in ("id", "kind", "needs_gpu", "max_minutes")}, "cams": cams, "steps": [s["cmd"] for s in spec["steps"]]}, indent=1))
        return 0
    res = run_job(spec, cpu=False, minutes=a.minutes, hard_minutes=a.hard_minutes, max_cost=a.max_cost, kind=kind)
    print(json.dumps(res, default=str))
    return 0 if res.get("status") == "done" else 1


TEXTURE_FILES = ("lead_albedo.png", "follow_albedo.png", "lead_coverage.png", "follow_coverage.png", "lead_texels.npz", "follow_texels.npz",
                 "uv_mesh.npz", "head_fix.npz", "report.json", "follow_hair_region.png", "follow_hair.json", "follow_hair_strip.png", "follow_hair_cover.png",
                 "lead_albedo_face.png", "follow_albedo_face.png")      # the last two: face-texture's composite, when it ran on that window


def texture_install(window_id: str = "tx1", force: bool = False) -> dict:
    """Copy the pulled bake of a window (work/<take>_<window>/avatar/) into work/<take>/avatar/ - where demo_export, avatar_face and hair_groom
    read it. Refuses to overwrite a different file without force; the albedo is the bake of that ONE window (see AVATAR_PLAN.md)."""
    import shutil
    src = PACKAGE_ROOT / "work" / texture_take(window_id) / "avatar"
    rec: dict = {"from": str(src), "files": {}}
    for n in TEXTURE_FILES:
        f = src / n
        if not f.exists():
            rec["files"][n] = "missing"
            continue
        dst = RA.AV / n
        if dst.exists() and not force and dst.read_bytes() != f.read_bytes():
            rec["files"][n] = "kept (a different file is there; --force)"
            continue
        shutil.copy2(f, dst)
        rec["files"][n] = "installed"
    (RA.AV / "TEXTURE_FROM.json").write_text(json.dumps({**rec, "window": window_id, "installed": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    return rec
