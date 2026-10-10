"""Pod side of the Carlos & Aline avatars (agent C5): job specs of kind `avatarca`, the one-job-one-pod runner, the pod-side identity step.

Same design as recap_avatars_pod.py (READ-ONLY template; its run_job is bound to the recap's owner / state file, so this module has its own
copy parameterised for this take): ONE job spec (dancecap.farm.job/1) of the dedicated kind `avatarca`, id `ca-<stage>-<time>`; no farm worker
claims that kind (the fit pods run --kinds front,fit,refine,full,asr, the analysis pods --kinds analyse, the recap's avatar pods `avatar`), only
the pod THIS module starts (HM_KINDS=avatarca). Inputs = ONE deterministic tar of the small files (landmarks, frames, portraits, betas, identity
npz for a retarget); FLAME / SMIRK live on the owner's private volume (models/, uploaded once). A CPU pod (cpu5c x 8 vCPU, ~0.28 USD/h); the pod
is deleted in a `finally`, the on-pod kill switch (HM_KILL_AT / HM_HARD_CEIL) is the backstop. Other agents' fit pods on this take are never
touched: only ids this module's own calls created are deleted.

    python -m dancecap.ca_avatars face-identity [--max-cost 1.0] [--dry-run]
    python -m dancecap.ca_avatars face-retarget [--betas FILE]            (identity files already pulled; betas landed later)
    python -m dancecap.ca_avatars pod-identity [--retarget-only]          (the pod-side step)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import ca_avatars as CA
from .take import PACKAGE_ROOT

STATE = CA.AV / "pod_state.json"


def log(msg: str) -> None:
    print(f"[ca_avatars_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ inputs / steps / outputs

def identity_archive(betas: dict[str, np.ndarray] | None, retarget_only: bool) -> dict:
    """The small inputs as ONE deterministic tar (workspace-relative arcnames dancecap/work/carlos_aline/...): the landmarks, the chosen frames and
    the two portraits (under avatar/face_id/portrait_<role>.<ext> INSIDE the tar only) for a fit; the identity npz files for a retarget-only job;
    betas.json when the betas exist."""
    from .farm import chunks as C
    items: list[tuple[Path, str]] = []
    if not retarget_only:
        for role in CA.ROLES:
            lm = CA.FACE_ID / f"landmarks_{role}.json"
            items.append((lm, f"{CA.REL}/avatar/face_id/{lm.name}"))
            for r in json.loads(lm.read_text()):
                f = CA.portrait_path(role) if r["name"] == "portrait" else CA.WORK / r["file"]
                items.append((f, f"{CA.REL}/{r['file']}"))
    else:
        for role in CA.ROLES:
            f = CA.FACE_SHAPE / f"{role}_identity.npz"
            if not f.exists():
                raise FileNotFoundError(f"{f}: run face-identity first")
            items.append((f, f"{CA.REL}/face_shape/{f.name}"))
    if betas is not None:
        b = CA.AV / "face_id" / "betas.json"
        b.write_text(json.dumps({r: [float(x) for x in v] for r, v in betas.items()}, indent=1))
        items.append((b, f"{CA.REL}/avatar/betas.json"))
    items = sorted({a: (s, a) for s, a in items}.values(), key=lambda x: x[1])
    dest = CA.FACE_ID / "in" / ("retarget_in.tar" if retarget_only else "faceid_in.tar")
    dest.parent.mkdir(parents=True, exist_ok=True)
    C.pack_tar(dest, items)
    return {"path": f"{CA.REL}/avatar/face_id/{dest.name}", "archive": True, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "bytes": dest.stat().st_size, "local": str(dest), "files": len(items)}


def identity_steps(retarget_only: bool) -> list[dict]:
    links = ("set -e; cd ..; mkdir -p models external; ln -sfn $HM_ROOT/models/flame models/flame; "
             "ln -sfn $HM_ROOT/models/smirk external/smirk; "
             "ls models/flame/FLAME2020/generic_model.pkl external/smirk/pretrained_models/SMIRK_em1.pt > /dev/null; "
             "python -c 'import timm' 2>/dev/null || pip install -q 'timm==0.9.16'")
    cmd = "python -m dancecap.ca_avatars pod-identity" + (" --retarget-only" if retarget_only else "")
    return [{"name": "links", "cmd": links, "timeout_min": 8, "optional": False},
            {"name": "identity", "cmd": cmd, "timeout_min": 40, "optional": False}]


def identity_outputs(with_betas: bool, retarget_only: bool) -> list[dict]:
    rows: list[tuple[str, bool]] = []
    for role in CA.ROLES:
        if not retarget_only:
            rows += [(f"{CA.REL}/face_shape/{role}_identity.npz", False)]
            rows += [(f"{CA.REL}/face_shape/{role}_identity_{v}.npz", True) for v in ("portrait", "video")]
        if with_betas:
            rows += [(f"{CA.REL}/face_shape/{role}_identity_disp.npz", False)]
    rows += [(f"{CA.REL}/avatar/face_id/{'retarget_report' if retarget_only else 'fit_report'}.json", False)]
    return [{"path": p, **({"optional": True} if opt else {})} for p, opt in rows]


def identity_spec(job_id: str, betas: dict[str, np.ndarray] | None, retarget_only: bool = False) -> dict:
    from .farm import worker
    inp = identity_archive(betas, retarget_only)
    spec = worker.analyse_spec(job_id, CA.TAKE, identity_steps(retarget_only), identity_outputs(betas is not None, retarget_only), inputs=[inp],
                               max_minutes=45, priority=10, gpu=False, kind=CA.KIND)
    spec.update({"code": {}, "retries": 1, "needs_gpu": False, "env": {"DANCECAP_FARM_DANCE": CA.TAKE, "OMP_NUM_THREADS": "6"}})
    return spec


def estimate(retarget_only: bool = False, minutes_fit: float = 14.0) -> dict:
    mins = {"pod start + env + code": 2.0, "inputs (tar; models on the volume)": 1.0}
    if not retarget_only:
        mins["SMIRK + landmark fits (2 dancers, ~26 images)"] = minutes_fit
    mins["retarget (FLAME fit to the rest heads)"] = 1.5 if not retarget_only else 3.0
    tot = sum(mins.values())
    return {"minutes": mins, "total_min": round(tot, 1), "price_hr": CA.PRICE_CPU_8, "usd": round(tot / 60 * CA.PRICE_CPU_8, 3)}


# ------------------------------------------------------------------------------------------------ the pod-side step

def pod_identity(portrait_share: float = 0.5, work: Path | None = None, iters: int = 600, limit: int | None = None, retarget_only: bool = False) -> dict:
    """On the pod: per dancer the SMIRK + landmark fit of every image once (fit_images), three robust combinations (mixed: the portrait carries
    `portrait_share` of the total weight; portrait only; video only) -> face_shape/<role>_identity{,_portrait,_video}.npz; then, when betas.json is in
    the tar, the mixed identity retargeted to the REST head of those betas (face_identity.cached_displacement -> <role>_identity_disp.npz, the
    cache the export reads). retarget_only: skip the fit, read the identity files from the tar."""
    from . import export_unity as eu
    from . import face_identity as fi
    work = work or CA.WORK
    bp = work / "avatar" / "betas.json"
    betas = json.loads(bp.read_text()) if bp.exists() else None
    (work / "face_shape").mkdir(parents=True, exist_ok=True)
    report: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "portrait_share": portrait_share, "retarget_only": retarget_only, "roles": {}}
    for role in CA.ROLES:
        rep: dict = {}
        if not retarget_only:
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
                                           "take": CA.TAKE})
                files[v] = out.name
            b = {v: variants[v]["beta"] for v in variants}
            names = [p["name"] for p in per]
            rep = {"images": names, "inlier": dict(zip(names, variants["mixed"]["inlier"].tolist())),
                   "fit_err": dict(zip(names, [round(float(x), 4) for x in variants["mixed"]["fit_err"]])),
                   "dist_to_median": dict(zip(names, [round(float(x), 3) for x in variants["mixed"]["dist_to_median"]])),
                   "portrait_weight": pw, "beta_norm": {v: round(float(np.linalg.norm(x)), 3) for v, x in b.items()},
                   "beta_distance": {f"{a}-{c}": round(float(np.linalg.norm(b[a] - b[c])), 3) for a in b for c in b if a < c},
                   "files": files, "seconds_fit": round(time.time() - t0, 1)}
        if betas is not None:
            sk = eu.shaped_skin(eu.model_for({"model": "smplx_locked_head"}), np.asarray(betas[role], float))
            t1 = time.time()
            _, _, rrep = fi.cached_displacement(sk.vertices, work / "face_shape" / f"{role}_identity.npz", work, role, 1.0)
            rep["retarget"] = {k: rrep.get(k) for k in ("equivalent_rms_mm", "equivalent_max_mm", "stable_residual_mm", "max_move_mm",
                                                        "occ_mm_head_vs_identity", "cache_hit")}
            rep["retarget"]["seconds"] = round(time.time() - t1, 1)
        report["roles"][role] = rep
        print(json.dumps({role: rep}, default=float), flush=True)
    (work / "avatar" / "face_id" / ("retarget_report.json" if retarget_only else "fit_report.json")).write_text(json.dumps(report, indent=1, default=float))
    return report


# ------------------------------------------------------------------------------------------------ one job, one pod

def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {"pods": {}, "jobs": {}, "pulled": []}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    disk = load_state()
    merged = {"pods": {**disk.get("pods", {}), **st.get("pods", {})}, "jobs": {**disk.get("jobs", {}), **st.get("jobs", {})},
              "pulled": sorted(set(disk.get("pulled", [])) | set(st.get("pulled", [])))}
    tmp = STATE.with_name(f"{STATE.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(merged, indent=1, default=str))
    for k in range(40):          # Windows: another process of a parallel window may hold the target for a moment (WinError 5 / 32)
        try:
            tmp.replace(STATE)
            return
        except PermissionError:
            time.sleep(0.05 * (k + 1))
    try:
        tmp.unlink()
    except OSError:
        pass                     # the state file is bookkeeping only: a lost update must never abort a paid run


def run_job(spec: dict, *, cpu: bool = True, vcpu: int = 8, gpus: list[str] | None = None, minutes: float = 40, hard_minutes: float = 70,
            max_cost: float = 1.0, upload_models: bool = False, poll_s: float = 30.0, log_fn=log, kind: str | None = None) -> dict:
    """Queue `spec` on the volume, start ONE pod of kind `avatarca` (or `kind`: several jobs in parallel take a kind each, e.g. avatarcatx1), wait,
    pull, delete every pod this call created (also on an error)."""
    kind = kind or spec.get("kind") or CA.KIND
    from . import recap_avatars_pod as rap
    from . import recap_cameras as rc
    from .farm import pods, volume
    st = load_state()
    mine: list[str] = []
    gw = None
    try:
        gw, _ = pods.gateway(minutes=max(45, int(hard_minutes) + 20), vcpu=2, reuse=False, owner=CA.OWNER)    # must outlive the job (a 45-minute gateway died under a 55-minute bake: the poll failed, the finally deleted the GPU pod mid-bake)
        mine.append(gw.id)
        st["pods"][gw.id] = {"role": "gateway", "started": time.time(), "price_hr": gw.cost_per_hr}
        save_state(st)
        link = volume.Link(gw)
        info = volume.sync_code(link, make_current=False)
        spec["code"] = {"git_sha": info.get("git_sha"), "dirty": info.get("dirty"), "bundle_sha256": info.get("bundle_sha256")}
        if upload_models:
            t0 = time.time()
            r = rap.ensure_models(link)
            log_fn(f"models on the volume: {r} in {time.time() - t0:.0f} s")
        volume.submit_job(link, spec)
        st["jobs"][spec["id"]] = {"status": "queued", "queued": time.time(), "code": info.get("key")}
        save_state(st)
        log_fn(f"job {spec['id']} queued (code {info['key']}); starting one {'CPU' if cpu else 'GPU'} pod of kind {kind}")
        if cpu:
            launched = pods.launch_cpu(1, vcpu=vcpu, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds=kind, idle_min=4,
                                       prefix="hm-farm-caavatar", owner=CA.OWNER)
        else:
            launched = pods.launch(1, gpus=gpus, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds=kind, idle_min=4,
                                   prefix="hm-farm-caavatar", owner=CA.OWNER)
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
                pods.delete(pid, reason="carlos_aline avatars run finished")
                log_fn(f"pod {pid} deleted")
                st["pods"].get(pid, {})["deleted"] = time.time()
            except Exception as e:  # noqa: BLE001
                log_fn(f"WARNING pod {pid} NOT confirmed deleted: {e}")
        save_state(st)


def owner_spend() -> float:
    from . import remote
    return round(sum(float(r.get("cost") or 0.0) for r in remote.ledger() if r.get("owner") == CA.OWNER), 3)


def cli_job(a) -> int:
    """face-identity / face-retarget: build the spec, print the estimate, run (or --dry-run)."""
    retarget = a.cmd == "face-retarget"
    betas = CA.pinned_betas(a.betas)
    if retarget and betas is None:
        raise SystemExit("face-retarget needs the pinned betas (C3's shape pass: work/carlos_aline_cashape/mamma_pin_betas.json, or --betas FILE)")
    if retarget and getattr(a, "local", False):      # measured: FLAME's retarget takes ~10 s per dancer on the PC (the pod only pays off for the fit)
        (CA.AV).mkdir(parents=True, exist_ok=True)
        (CA.AV / "betas.json").write_text(json.dumps({r: [float(x) for x in v] for r, v in betas.items()}, indent=1))
        rep = pod_identity(retarget_only=True)
        print(json.dumps({r: v.get("retarget") for r, v in rep["roles"].items()}, default=float))
        return 0
    jid = a.job_id or f"ca-{'retarget' if retarget else 'identity'}-{time.strftime('%m%d%H%M')}"
    spec = identity_spec(jid, betas, retarget)
    est = estimate(retarget)
    log(f"estimate: {est['total_min']} min = {est['usd']} USD at {est['price_hr']}/h (ceiling {a.max_cost}); inputs {spec['inputs'][0]['bytes'] / 1e6:.1f} MB; "
        f"betas {'in the tar' if betas is not None else 'absent: no retarget'}")
    if a.dry_run:
        print(json.dumps({"job": {k: spec[k] for k in ("id", "kind", "needs_gpu", "max_minutes")}, "steps": [s["cmd"][:200] for s in spec["steps"]],
                          "outputs": [o["path"] for o in spec["outputs"]], "estimate": est}, indent=1))
        return 0
    res = run_job(spec, cpu=True, vcpu=a.vcpu, minutes=a.minutes, hard_minutes=a.hard_minutes, max_cost=a.max_cost, upload_models=a.upload_models)
    print(json.dumps(res, default=str))
    return 0 if res.get("status") == "done" else 1


# ------------------------------------------------------------------------------------------------ the pose-dependent texture job
# A parameterised copy of recap_avatars_pod's texture job for THIS take (that module is bound to the recap's constants and is read-only here):
# the window's farm job spec = a copy of the take's own shape job (farm/jobs/cashape.json: sources, ca_in.tar, env HM_FRONT_DANCE ...) with the
# window as its chunk; chunk-take writes the window's take TOML on the pod; `python -m dancecap.recap_avatars pod-texture-front <window>` is
# recap_front's front DAG minus its MAMMA tail (frames, cameras, people, SAM 2.1 masks + hair masks, merge); it only imports recap_avatars, the
# front itself reads HM_FRONT_DANCE=carlos_aline from the job env; then `dancecap texture carlos_aline_<window>` bakes. NOT yet exercised on a
# pod (no stitched poses existed when this was written): `texture-run --dry-run` builds and prints the spec.

BASE_JOB = "cashape"


def texture_take(window_id: str) -> str:
    return f"{CA.TAKE}_{window_id}"


def window_job(window_id: str, t0: float, t1: float, base_id: str = BASE_JOB) -> dict:
    import copy
    import math
    base = json.loads((CA.WORK / "farm" / "jobs" / f"{base_id}.json").read_text())
    job = copy.deepcopy(base)
    k0, k1 = int(math.ceil(t0 * 30.0 - 1e-6)), int(math.floor(t1 * 30.0 + 1e-6)) - 1
    take = texture_take(window_id)
    cams = list(base.get("cams") or ["01", "02"])
    job.update({"id": window_id, "take": take, "cams": cams, "kind": "front", "what": "texture window", "stages": ["front"], "after": []})
    job["chunk"].update({"id": window_id, "k0": k0, "k1": k1, "core_k0": k0, "core_k1": k1, "overlap_k": 0, "take": take, "cams": cams,
                         "T": k1 - k0 + 1, "core_s": round((k1 - k0 + 1) / 30.0, 2), "origin": "texture window"})
    return job


def chunk_sync_file(t_mid: float) -> Path:
    own = CA.WORK / "sync" / "offsets.json"
    if own.exists():
        return own
    plan = json.loads((CA.WORK / "chunks.json").read_text())
    for c in plan.get("chunks", []):
        k0, k1 = c["core_k"]
        if k0 / 30.0 <= t_mid <= (k1 + 1) / 30.0:
            f = PACKAGE_ROOT / "work" / f"{CA.TAKE}_{c['id']}" / "sync" / "offsets.json"
            if f.exists():
                return f
    raise FileNotFoundError(f"no sync offsets for t = {t_mid:.1f} s: pass --sync (work/carlos_aline_<chunk>/sync/offsets.json of a pulled chunk)")


def texture_archive(window_id: str, job_file: Path, smplx_dir: Path, sync_file: Path) -> dict:
    from .farm import chunks as C
    tk = texture_take(window_id)
    items = [(job_file, f"{CA.REL}/farm/jobs/{window_id}.json")]
    items += [(smplx_dir / f"{r}.npz", f"dancecap/work/{tk}/smplx/{r}.npz") for r in CA.ROLES]
    items += [(sync_file, f"dancecap/work/{tk}/sync/offsets.json")]
    miss = [str(src) for src, _ in items if not src.exists()]
    if miss:
        raise FileNotFoundError(f"texture job inputs missing: {miss}")
    dest = CA.AV / "texture" / "in" / f"texture_in_{window_id}.tar"
    C.pack_tar(dest, items)
    return {"path": f"dancecap/work/{tk}/texture_in.tar", "archive": True, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "bytes": dest.stat().st_size, "local": str(dest), "files": len(items)}


def shared_inputs(base_id: str = BASE_JOB) -> list[dict]:
    """The base job's rows that name SHARED volume files (ca_in.tar: the camera tracks, the people tables; already on the volume from the farm run,
    no `local` key: nothing is uploaded again). The videos are read from the volume's src/ by the front."""
    spec = json.loads((CA.WORK / "farm" / "jobs" / f"{base_id}.json").read_text())
    rows = [{k: v for k, v in r.items() if k != "local"} for r in spec["inputs"] if r["path"].endswith("ca_in.tar")]
    if not rows:
        raise RuntimeError(f"farm/jobs/{base_id}.json lists no ca_in.tar input")
    return rows


def texture_steps(window_id: str, step_ms: float = 100.0, top_k: int = 6, body_ref_q: float | None = 0.7, mask: str | None = None) -> list[dict]:
    tk = texture_take(window_id)
    w = f"dancecap/work/{tk}"
    steps = [
        {"name": "chunk_take", "cmd": f"python -m dancecap.farm chunk-take {window_id}", "timeout_min": 3, "optional": False},
        {"name": "weights", "cmd": "python -m dancecap.front_full fetch-weights", "timeout_min": 12, "optional": True},
        {"name": "front", "cmd": f"python -m dancecap.recap_avatars pod-texture-front {window_id}", "timeout_min": 70, "optional": False},
        {"name": "stage", "cmd": f"cd ..; touch {w}/smplx/*.npz; ls {w}/dancers/roles.json {w}/sync/offsets.json", "timeout_min": 3, "optional": False},
    ] + ([{"name": "maskcams", "cmd": f"python -m dancecap.recap_avatars pod-mask-cams {window_id} --mask {mask}", "timeout_min": 3, "optional": False}] if mask else []) + [
        {"name": "bake", "cmd": f"DANCECAP_POD=1 python -m dancecap texture {tk} --preset pod --step-ms {step_ms:g} --top-k {top_k} --align-passes 2 "
                               + (f"--body-ref-q {body_ref_q:g} " if body_ref_q else "") + "--hair-mode exclude --blend multiband --no-lock --allow-stale",
         "timeout_min": 60, "optional": False},
    ]
    return steps


def texture_outputs(window_id: str) -> list[dict]:
    from .avatar_pod import OUTPUTS
    w = f"dancecap/work/{texture_take(window_id)}"
    must = {"avatar/lead_albedo.png", "avatar/follow_albedo.png", "avatar/report.json"}
    rows = [{"path": f"{w}/{o}", **({} if o in must else {"optional": True})} for o in OUTPUTS]
    rows += [{"path": f"{w}/{o}", "optional": True} for o in ("dancers/roles.json", "dancers/report.json", "front/front_done.json", "front/front_pod.json",
                                                              "frames/01/frames.json", "frames/02/frames.json")]
    return rows


def texture_spec(job_id: str, window_id: str, step_ms: float, job_file: Path, smplx_dir: Path, sync_file: Path, top_k: int = 6,
                 body_ref_q: float | None = 0.7, mask: str | None = None, kind: str = CA.KIND) -> dict:
    from .farm import worker
    base = json.loads((CA.WORK / "farm" / "jobs" / f"{BASE_JOB}.json").read_text())
    inputs = [texture_archive(window_id, job_file, smplx_dir, sync_file)] + shared_inputs()
    spec = worker.analyse_spec(job_id, CA.TAKE, texture_steps(window_id, step_ms, top_k, body_ref_q, mask), texture_outputs(window_id), inputs=inputs,
                               max_minutes=95, priority=10, gpu=True, kind=kind)
    env = {k: v for k, v in (base.get("env") or {}).items() if k.startswith(("HM_", "DANCECAP_FARM"))}
    # HM_RED_TOP=1: her top is red (the bake's red-hair leak guards would erase it); HM_TEACHER_EVIDENCE=1: the demo's colour rule picks the label frame for another pair (C3: it failed cag4345); the teacher tracks already name the roles
    spec.update({"code": {}, "retries": 0, "needs_gpu": True, "min_vram_gb": 24,
                 "env": {**env, "DANCECAP_FARM_DANCE": CA.TAKE, "DANCECAP_POD": "1", "HM_TEACHER_EVIDENCE": "1", "HM_RED_TOP": "1"}})
    return spec


def texture_run(a) -> int:
    """`texture-run`: windows.json (texture-select) -> ONE pod job: the window's front + the bake on the sliced stitched poses."""
    from .recap_avatars_pod import slice_smplx
    doc = json.loads(a.windows.read_text())
    w = doc["windows"][a.which]
    t0, t1 = float(w["t0"]), float(w["t1"])
    mask = a.mask or ",".join(w.get("mask") or []) or None
    log(f"window {a.window_id}: {t0:.1f}-{t1:.1f} s, cameras 01 + 02; masked {mask}")
    job = window_job(a.window_id, t0, t1)
    jdir = CA.AV / "texture" / "jobs"
    jdir.mkdir(parents=True, exist_ok=True)
    jf = jdir / f"{a.window_id}.json"
    jf.write_text(json.dumps(job, indent=1))
    sm = PACKAGE_ROOT / "work" / texture_take(a.window_id) / "smplx"
    sl = slice_smplx(a.smplx_dir, sm, t0 - 0.5, t1 + 0.5)
    log(f"stitched poses sliced: {sl} frames -> {sm}")
    sync = a.sync or chunk_sync_file(0.5 * (t0 + t1))
    jid = f"ca-texture-{a.window_id}-{time.strftime('%m%d%H%M')}"
    kind = f"{CA.KIND}{a.window_id}" if a.own_kind else CA.KIND
    spec = texture_spec(jid, a.window_id, a.step_ms, jf, sm, sync, a.top_k, a.body_ref_q, mask, kind)
    log(f"job {jid}: {len(spec['steps'])} steps, inputs {[i['path'] for i in spec['inputs']]}")
    if a.dry_run:
        print(json.dumps({"job": {k: spec[k] for k in ("id", "kind", "needs_gpu", "max_minutes", "env")}, "steps": [s["cmd"] for s in spec["steps"]]}, indent=1))
        return 0
    res = run_job(spec, cpu=False, minutes=a.minutes, hard_minutes=a.hard_minutes, max_cost=a.max_cost, kind=kind)
    print(json.dumps(res, default=str))
    return 0 if res.get("status") == "done" else 1


TEXTURE_FILES = ("lead_albedo.png", "follow_albedo.png", "lead_coverage.png", "follow_coverage.png", "lead_texels.npz", "follow_texels.npz",
                 "uv_mesh.npz", "head_fix.npz", "report.json", "follow_hair_region.png", "follow_hair.json", "follow_hair_strip.png", "follow_hair_cover.png",
                 "lead_albedo_face.png", "follow_albedo_face.png")


def texture_install(window_id: str, force: bool = False) -> dict:
    import shutil
    src = PACKAGE_ROOT / "work" / texture_take(window_id) / "avatar"
    rec: dict = {"from": str(src), "files": {}}
    for n in TEXTURE_FILES:
        f = src / n
        if not f.exists():
            rec["files"][n] = "missing"
            continue
        dst = CA.AV / n
        if dst.exists() and not force and dst.read_bytes() != f.read_bytes():
            rec["files"][n] = "kept (a different file is there; --force)"
            continue
        shutil.copy2(f, dst)
        rec["files"][n] = "installed"
    (CA.AV / "TEXTURE_FROM.json").write_text(json.dumps({**rec, "window": window_id, "installed": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    return rec
