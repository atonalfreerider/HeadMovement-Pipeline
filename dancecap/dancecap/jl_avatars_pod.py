"""Pod side of the `john_lula` avatars (agent J5a): the face-identity job of kind `avatarjl`, the pod-side step, and the local trust analysis.

Same design as ca_avatars_pod.py (its run_job is bound to that take's owner / state file, so this module has its own copy parameterised for this one):
ONE job spec (dancecap.farm.job/1) of the dedicated kind `avatarjl`, id `jl-identity-<time>`; no farm worker claims that kind (the fit pods run
--kinds front,fit,refine,full,asr, the analysis pods analyse, the other avatar pods avatar / avatarca), only the pod THIS module starts (HM_KINDS=avatarjl).
Inputs = ONE deterministic tar of the small files (landmarks, the chosen frames, the lead's portrait); FLAME / SMIRK live on the owner's private volume
(models/flame, models/smirk, uploaded once). A CPU pod (cpu5c x 8 vCPU, ~0.28 USD/h); the pod is deleted in a `finally`, the on-pod kill switch
(HM_KILL_AT / HM_HARD_CEIL) is the backstop. Pods this module did not create are never touched. Owner label `J5a`.

    python -m dancecap.jl_avatars face-identity [--max-cost 1.0] [--dry-run]
    python -m dancecap.jl_avatars_pod pod-identity                       (the pod-side step)
    python -m dancecap.jl_avatars face-trust                             (local: how far to trust a video-only identity)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import ca_avatars as CA
from . import jl_avatars as J
from .take import PACKAGE_ROOT

STATE = J.AV / "pod_state.json"
PRICE_CPU_8 = CA.PRICE_CPU_8


def log(msg: str) -> None:
    print(f"[jl_avatars_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ inputs / steps / outputs

def identity_archive() -> dict:
    """The small inputs as ONE deterministic tar (workspace-relative arcnames dancecap/work/john_lula/...): per role the landmarks file and its frames
    (the lead's portrait under avatar/face_id/portrait_lead.<ext> INSIDE the tar only)."""
    from .farm import chunks as C
    items: list[tuple[Path, str]] = []
    for role in J.ROLES:
        lm = J.FACE_ID / f"landmarks_{role}.json"
        items.append((lm, f"{J.REL}/avatar/face_id/{lm.name}"))
        for r in json.loads(lm.read_text()):
            f = J.portrait_path(role) if r["name"] == "portrait" else J.WORK / r["file"]
            items.append((f, f"{J.REL}/{r['file']}"))
    items = sorted({a: (s, a) for s, a in items}.values(), key=lambda x: x[1])
    dest = J.FACE_ID / "in" / "faceid_in.tar"
    dest.parent.mkdir(parents=True, exist_ok=True)
    C.pack_tar(dest, items)
    return {"path": f"{J.REL}/avatar/face_id/{dest.name}", "archive": True, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "bytes": dest.stat().st_size, "local": str(dest), "files": len(items)}


def identity_steps() -> list[dict]:
    links = ("set -e; cd ..; mkdir -p models external; ln -sfn $HM_ROOT/models/flame models/flame; "
             "ln -sfn $HM_ROOT/models/smirk external/smirk; "
             "ls models/flame/FLAME2020/generic_model.pkl external/smirk/pretrained_models/SMIRK_em1.pt > /dev/null; "
             "python -c 'import timm' 2>/dev/null || pip install -q 'timm==0.9.16'")
    return [{"name": "links", "cmd": links, "timeout_min": 8, "optional": False},
            {"name": "identity", "cmd": "python -m dancecap.jl_avatars_pod pod-identity", "timeout_min": 40, "optional": False}]


def identity_outputs() -> list[dict]:
    rows: list[tuple[str, bool]] = [(f"{J.REL}/face_shape/{role}_identity.npz", False) for role in J.ROLES]
    rows += [(f"{J.REL}/face_shape/lead_identity_portrait.npz", True)]
    rows += [(f"{J.REL}/face_shape/{role}_identity_video.npz", True) for role in J.ROLES]
    rows += [(f"{J.REL}/avatar/face_id/fit_report.json", False)]
    return [{"path": p, **({"optional": True} if opt else {})} for p, opt in rows]


def identity_spec(job_id: str) -> dict:
    from .farm import worker
    inp = identity_archive()
    spec = worker.analyse_spec(job_id, J.TAKE, identity_steps(), identity_outputs(), inputs=[inp], max_minutes=45, priority=10, gpu=False, kind=J.KIND)
    spec.update({"code": {}, "retries": 1, "needs_gpu": False, "env": {"DANCECAP_FARM_DANCE": J.TAKE, "OMP_NUM_THREADS": "6"}})
    return spec


def estimate(minutes_fit: float = 12.0) -> dict:
    mins = {"pod start + env + code": 2.0, "inputs (tar; models on the volume)": 1.0, "SMIRK + landmark fits (2 dancers)": minutes_fit}
    tot = sum(mins.values())
    return {"minutes": mins, "total_min": round(tot, 1), "price_hr": PRICE_CPU_8, "usd": round(tot / 60 * PRICE_CPU_8, 3)}


# ------------------------------------------------------------------------------------------------ the pod-side step

def pod_identity(portrait_share: float = 0.5, work: Path | None = None, iters: int = 600, limit: int | None = None) -> dict:
    """On the pod: per dancer the SMIRK + landmark fit of every image once (fi.fit_images), then robust combinations: mixed (the lead: the portrait carries
    `portrait_share` of the total weight; the follow has no portrait: the video faces alone), portrait only (lead), video only -> face_shape/<role>_identity
    {,_portrait,_video}.npz (shape code + per-image codes: the trust analysis reads those). No retarget here: the fitted poses (and their betas) do not exist yet."""
    from . import face_identity as fi
    work = work or J.WORK
    (work / "face_shape").mkdir(parents=True, exist_ok=True)
    report: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "portrait_share": portrait_share, "roles": {}}
    for role in J.ROLES:
        lm = json.loads((work / "avatar" / "face_id" / f"landmarks_{role}.json").read_text())
        lm = lm[:limit] if limit else lm
        for r in lm:
            r["file"] = str(work / r["file"])
        t0 = time.time()
        res = fi.fit_images(lm, device="cpu", portrait_weight=1.0, iters=iters)
        per = res["per_image"]
        n_video = sum(1 for p in per if p["name"] != "portrait")
        has_portrait = any(p["name"] == "portrait" for p in per)
        pw = max(1.0, portrait_share / max(1e-6, 1 - portrait_share) * n_video) if has_portrait else 1.0
        variants = {"mixed": fi.combine(per, pw)}
        if has_portrait:
            variants["portrait"] = fi.combine([p for p in per if p["name"] == "portrait"], 1.0)
        if n_video:
            variants["video"] = fi.combine([p for p in per if p["name"] != "portrait"], 1.0)
        files = {}
        for v, rv in variants.items():
            out = work / "face_shape" / (f"{role}_identity.npz" if v == "mixed" else f"{role}_identity_{v}.npz")
            fi.save_identity(rv, out, {"landmarks": f"landmarks_{role}.json", "variant": v, "portrait_weight": pw if v == "mixed" else None, "take": J.TAKE})
            files[v] = out.name
        b = {v: variants[v]["beta"] for v in variants}
        names = [p["name"] for p in per]
        rep = {"images": names, "inlier": dict(zip(names, variants["mixed"]["inlier"].tolist())),
               "fit_err": dict(zip(names, [round(float(x), 4) for x in variants["mixed"]["fit_err"]])),
               "dist_to_median": dict(zip(names, [round(float(x), 3) for x in variants["mixed"]["dist_to_median"]])),
               "portrait_weight": pw, "has_portrait": has_portrait, "beta_norm": {v: round(float(np.linalg.norm(x)), 3) for v, x in b.items()},
               "beta_distance": {f"{a}-{c}": round(float(np.linalg.norm(b[a] - b[c])), 3) for a in b for c in b if a < c},
               "files": files, "seconds_fit": round(time.time() - t0, 1)}
        report["roles"][role] = rep
        print(json.dumps({role: rep}, default=float), flush=True)
    (work / "avatar" / "face_id" / "fit_report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


# ------------------------------------------------------------------------------------------------ pure: the trust analysis

def combine_betas(B: np.ndarray, w: np.ndarray, err: np.ndarray, max_err: float = 2.0, max_dist: float = 2.5) -> np.ndarray:
    """face_identity.combine on bare arrays (B (n, 300) per-image shape codes, w weights, err fit errors): residual inliers, shape-space inliers (distance to
    the weighted median), weighted mean. -> beta (300,)."""
    from .face_identity import _wmedian
    ok = err <= max_err * np.median(err)
    med = np.array([_wmedian(B[ok, j], w[ok]) for j in range(B.shape[1])])
    dist = np.linalg.norm(B - med, axis=1)
    ok &= dist <= max_dist * max(np.median(dist[ok]), 1e-6)
    return (w[ok, None] * B[ok]).sum(0) / w[ok].sum()


def split_half(B: np.ndarray, err: np.ndarray, rng: np.random.Generator, n_splits: int = 200) -> dict:
    """Split-half reliability of a combination of n >= 4 per-image codes: the distance between the combinations of two random halves (mean, p10..p90) and the
    implied error of the full-set mean against the truth that the images share (half the half-to-half distance: the random part only)."""
    n = len(B)
    if n < 4:
        return {"n": n, "half_distance_mean": None}
    d = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2: 2 * (n // 2)]
        d.append(float(np.linalg.norm(combine_betas(B[a], np.ones(len(a)), err[a]) - combine_betas(B[b], np.ones(len(b)), err[b]))))
    d = np.asarray(d)
    return {"n": n, "half_distance_mean": round(float(d.mean()), 3), "half_distance_p10_p90": [round(float(np.percentile(d, 10)), 3), round(float(np.percentile(d, 90)), 3)],
            "implied_full_set_error": round(float(d.mean() / 2.0), 3)}


def shrink_weight(signal_rms: float, error: float, floor: float = 0.25) -> float:
    """The Bayes shrinkage of an estimate beta_hat = beta + e towards the template when beta ~ N(0, s^2 I) and |e| = error: w = s^2 / (s^2 + error^2), s = the
    RMS norm of a person's identity code (signal_rms). Clipped to [floor, 1]."""
    s2, e2 = float(signal_rms) ** 2, float(error) ** 2
    return float(np.clip(s2 / (s2 + e2), floor, 1.0))


def projection_scale(est: np.ndarray, ref: np.ndarray) -> float:
    """The factor a that minimises |a est - ref| (a < 1: the estimate carries noise energy the reference does not): (est . ref) / (est . est)."""
    return float(est @ ref / max(float(est @ est), 1e-12))


def nearest_rank(est: np.ndarray, truth: np.ndarray, others: list[np.ndarray]) -> int:
    """How many of the `others` lie closer to the estimate than the truth does (0 = the estimate points at its own person among them)."""
    d0 = float(np.linalg.norm(est - truth))
    return int(sum(1 for o in others if float(np.linalg.norm(est - o)) < d0))


def load_id(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def stage_face_trust() -> dict:
    """How far to trust the video-only identity (the follow has no portrait). Evidence, all computed from the fitted per-image shape codes:
    1. the LEAD has both: |beta_video - beta_portrait| = what a video-only identity (this resolution) gets wrong against the 450-px portrait fit;
    2. split-half reliability of the video faces of each dancer (random part of the error);
    3. the between-person scale: the RMS norm of the portrait / video identities of the other takes' dancers and the distances between them;
    4. a shrinkage weight s^2 / (s^2 + e^2) for the follow's `identity_weight` (the template = weight 0)."""
    rng = np.random.default_rng(0)
    rep: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "roles": {}}
    ids = {r: {v: load_id(J.FACE_SHAPE / f"{r}_identity{s}.npz") for v, s in (("mixed", ""), ("portrait", "_portrait"), ("video", "_video"))} for r in J.ROLES}
    for r in J.ROLES:
        vid = ids[r]["video"] or ids[r]["mixed"]
        if vid is None:
            continue
        per = vid["per_image_beta"]
        keep = np.array([n != "portrait" for n in vid["images"]])
        rep["roles"][r] = {"video_images": int(keep.sum()), "video_inliers": int(vid["inlier"][keep].sum()), "beta_norm_video": round(float(np.linalg.norm(vid["beta"])), 3),
                           "split_half": split_half(per[keep], vid["fit_err"][keep], rng)}
    lead_p, lead_v = ids["lead"]["portrait"], ids["lead"]["video"]
    if lead_p is not None and lead_v is not None:
        rep["lead_video_vs_portrait"] = {"projection_scale": round(projection_scale(lead_v["beta"], lead_p["beta"]), 3), "distance": round(float(np.linalg.norm(lead_p["beta"] - lead_v["beta"])), 3),
                                         "cosine": round(float(lead_p["beta"] @ lead_v["beta"] / (np.linalg.norm(lead_p["beta"]) * np.linalg.norm(lead_v["beta"]))), 3),
                                         "beta_norm_portrait": round(float(np.linalg.norm(lead_p["beta"])), 3)}
    # the between-person scale from every other take's identity files (portrait-only where there is one, else the mixed one)
    others = {}
    for d in sorted((PACKAGE_ROOT / "work").glob("*/face_shape")):
        if d.parent.name == J.TAKE:
            continue
        for role in J.ROLES:
            f = next((d / n for n in (f"{role}_identity_portrait.npz", f"{role}_identity.npz") if (d / n).exists()), None)
            z = load_id(f) if f else None
            if z is not None and z["beta"].shape == (300,):
                others[f"{d.parent.name}/{role}"] = z["beta"]
    if lead_p is not None:
        others[f"{J.TAKE}/lead_portrait"] = lead_p["beta"]
    if len(others) >= 2:
        B = np.stack(list(others.values()))
        pair = [float(np.linalg.norm(B[i] - B[j])) for i in range(len(B)) for j in range(i + 1, len(B))]
        rep["between_people"] = {"identities": len(B), "rms_norm": round(float(np.sqrt((B ** 2).sum(1).mean())), 3),
                                 "pair_distance_median": round(float(np.median(pair)), 3), "pair_distance_p10_p90": [round(float(np.percentile(pair, 10)), 3), round(float(np.percentile(pair, 90)), 3)]}
        s_rms = rep["between_people"]["rms_norm"]
        errs = {}
        if "lead_video_vs_portrait" in rep:
            errs["lead_measured_video_vs_portrait"] = rep["lead_video_vs_portrait"]["distance"]
        for r in J.ROLES:
            sh = rep["roles"].get(r, {}).get("split_half", {})
            if sh.get("implied_full_set_error") is not None:
                errs[f"{r}_split_half_random_part"] = sh["implied_full_set_error"]
        rep["shrink_weights"] = {k: round(shrink_weight(s_rms, e), 2) for k, e in errs.items()}
        rep["errors"] = errs
        # the follow's face is as small as the lead's video faces: the lead's MEASURED video-vs-portrait error is the honest error bar for her
        e_f = max(errs.get("lead_measured_video_vs_portrait", 0.0), errs.get("follow_split_half_random_part", 0.0))
        proj = rep.get("lead_video_vs_portrait", {}).get("projection_scale")
        w_f = shrink_weight(s_rms, e_f) if e_f else None
        if w_f is not None and proj:
            w_f = min(w_f, float(np.clip(proj, 0.25, 1.0)))       # the lead's measured over-dispersion of a video-only code, as a second bound
        rep["suggested_follow_identity_weight"] = round(float(np.round(w_f * 20) / 20), 2) if w_f is not None else None
        if lead_p is not None and lead_v is not None:
            oth = [v for k, v in others.items() if not k.startswith(J.TAKE)]
            rep["lead_video_nearest_rank"] = {"others_closer_than_portrait": nearest_rank(lead_v["beta"], lead_p["beta"], oth), "of": len(oth)}
        rep["suggested_lead_identity_weight"] = 1.0 if lead_p is not None else None
    (J.FACE_ID / "trust.json").write_text(json.dumps(rep, indent=1))
    return rep


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
    for k in range(40):          # Windows: a parallel window's process may hold the target for a moment (WinError 5 / 32)
        try:
            tmp.replace(STATE)
            return
        except PermissionError:
            time.sleep(0.05 * (k + 1))
    try:
        tmp.unlink()
    except OSError:
        pass                     # bookkeeping only: a lost update never aborts a paid run


def run_job(spec: dict, *, vcpu: int = 8, minutes: float = 30, hard_minutes: float = 50, max_cost: float = 1.0, upload_models: bool = False,
            poll_s: float = 20.0, log_fn=log, cpu: bool = True, gpus: list[str] | None = None, kind: str | None = None) -> dict:
    """Queue `spec` on the volume, start ONE pod (CPU, or a GPU when cpu=False) of kind `avatarjl` (or `kind`: parallel windows take a kind each), wait, pull,
    delete every pod this call created (also on an error)."""
    kind = kind or spec.get("kind") or J.KIND
    from . import recap_avatars_pod as rap
    from . import recap_cameras as rc
    from .farm import pods, volume
    st = load_state()
    mine: list[str] = []
    gw = None
    try:
        gw, _ = pods.gateway(minutes=max(45, int(hard_minutes) + 20), vcpu=2, reuse=False, owner=J.OWNER)
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
        if spec.get("kind") == J.KIND and cpu:
            have = link.sh("ls /workspace/hm/models/flame/FLAME2020/generic_model.pkl /workspace/hm/models/smirk/pretrained_models/SMIRK_em1.pt 2>&1 | tail -3", 60)
            if "No such file" in have and not upload_models:
                raise RuntimeError(f"FLAME / SMIRK are not on the volume ({have.strip()}): rerun with --upload-models")
            log_fn("FLAME + SMIRK present on the volume")
        volume.submit_job(link, spec)
        st["jobs"][spec["id"]] = {"status": "queued", "queued": time.time(), "code": info.get("key")}
        save_state(st)
        log_fn(f"job {spec['id']} queued (code {info['key']}); starting one {'CPU' if cpu else 'GPU'} pod of kind {kind}")
        if cpu:
            launched = pods.launch_cpu(1, vcpu=vcpu, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds=kind, idle_min=4,
                                       prefix="hm-farm-jlavatar", owner=J.OWNER)
        else:
            launched = pods.launch(1, gpus=gpus, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds=kind, idle_min=4,
                                   prefix="hm-farm-jlavatar", owner=J.OWNER)
        for l in launched:
            if l.get("pod_id"):
                mine.append(l["pod_id"])
                st["pods"][l["pod_id"]] = {"role": "cpu" if cpu else "gpu", "gpu": l.get("gpu"), "price_hr": l.get("cost_per_hr"), "started": time.time(),
                                           "kill_at": l.get("kill_at")}
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
                pods.delete(pid, reason="john_lula avatars run finished")
                log_fn(f"pod {pid} deleted")
                st["pods"].get(pid, {})["deleted"] = time.time()
            except Exception as e:  # noqa: BLE001
                log_fn(f"WARNING pod {pid} NOT confirmed deleted: {e}")
        save_state(st)


# ------------------------------------------------------------------------------------------------ the pose-dependent texture job (agent J5b)
# ONE farm job per window (kind `avatarjl<window>` so parallel windows never claim each other's job): the window's farm job spec = a copy of the take's own
# fit job jlg0000 (sources, jl_in.tar, env HM_FRONT_DANCE ...) with the window as its chunk; chunk-take writes the window's take TOML on the pod;
# `python -m dancecap.jl_front texture-front <window>` is the front WITHOUT SAM / ma_2d / prepare (frames, cameras, J3a's masks, the follow's hair / body
# split); then `dancecap texture john_lula_<window>` bakes (ONE camera: head fix off, there is no held-out camera to validate it).

BASE_JOB = "jlg0000"


def texture_take(window_id: str) -> str:
    return f"{J.TAKE}_{window_id}"


def window_job(window_id: str, t0: float, t1: float, base_id: str = BASE_JOB) -> dict:
    import copy
    import math
    base = json.loads((J.WORK / "farm" / "jobs" / f"{base_id}.json").read_text())
    job = copy.deepcopy(base)
    k0, k1 = int(math.ceil(t0 * 30.0 - 1e-6)), int(math.floor(t1 * 30.0 + 1e-6)) - 1
    take = texture_take(window_id)
    job.update({"id": window_id, "take": take, "cams": list(base["cams"]), "kind": "front", "what": "texture window", "stages": ["front"], "after": []})
    job["chunk"].update({"id": window_id, "k0": k0, "k1": k1, "core_k0": k0, "core_k1": k1, "overlap_k": 0, "take": take, "cams": list(base["cams"]),
                         "T": k1 - k0 + 1, "core_s": round((k1 - k0 + 1) / 30.0, 2), "origin": "texture window"})
    return job


def texture_archive(window_id: str, job_file: Path, smplx_dir: Path) -> dict:
    from .farm import chunks as C
    tk = texture_take(window_id)
    items = [(job_file, f"{J.REL}/farm/jobs/{window_id}.json")]
    items += [(smplx_dir / f"{r}.npz", f"dancecap/work/{tk}/smplx/{r}.npz") for r in J.ROLES]
    miss = [str(src) for src, _ in items if not src.exists()]
    if miss:
        raise FileNotFoundError(f"texture job inputs missing: {miss}")
    dest = J.AV / "texture" / "in" / f"texture_in_{window_id}.tar"
    C.pack_tar(dest, items)
    return {"path": f"dancecap/work/{tk}/texture_in.tar", "archive": True, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "bytes": dest.stat().st_size, "local": str(dest), "files": len(items)}


def texture_steps(window_id: str, step_ms: float = 100.0, top_k: int = 6, body_ref_q: float | None = 0.7, mask: str | None = None) -> list[dict]:
    tk = texture_take(window_id)
    w = f"dancecap/work/{tk}"
    steps = [
        {"name": "chunk_take", "cmd": f"python -m dancecap.farm chunk-take {window_id}", "timeout_min": 3, "optional": False},
        {"name": "front", "cmd": f"python -m dancecap.jl_front texture-front {window_id}", "timeout_min": 25, "optional": False},
        {"name": "stage", "cmd": f"cd ..; touch {w}/smplx/*.npz; ls {w}/dancers/roles.json", "timeout_min": 3, "optional": False},
    ] + ([{"name": "maskcams", "cmd": f"python -m dancecap.recap_avatars pod-mask-cams {window_id} --mask {mask}", "timeout_min": 3, "optional": False}] if mask else []) + [
        {"name": "bake", "cmd": f"DANCECAP_POD=1 python -m dancecap texture {tk} --preset pod --step-ms {step_ms:g} --top-k {top_k} --align-passes 2 --head-fix off "
                               + (f"--body-ref-q {body_ref_q:g} " if body_ref_q else "") + "--hair-mode exclude --blend multiband --no-lock --allow-stale",
         "timeout_min": 60, "optional": False},
    ]
    return steps


def texture_outputs(window_id: str) -> list[dict]:
    from .avatar_pod import OUTPUTS
    w = f"dancecap/work/{texture_take(window_id)}"
    must = {"avatar/lead_albedo.png", "avatar/follow_albedo.png", "avatar/report.json"}
    rows = [{"path": f"{w}/{o}", **({} if o in must else {"optional": True})} for o in OUTPUTS]
    rows += [{"path": f"{w}/{o}", "optional": True} for o in ("dancers/roles.json", "dancers/hair_split.json", "front/front_done.json", "frames/01/frames.json")]
    return rows


def texture_spec(job_id: str, window_id: str, step_ms: float, job_file: Path, smplx_dir: Path, top_k: int = 6, body_ref_q: float | None = 0.7,
                 mask: str | None = None, kind: str = J.KIND) -> dict:
    from .farm import worker
    base = json.loads((J.WORK / "farm" / "jobs" / f"{BASE_JOB}.json").read_text())
    shared = [dict(r) for r in base["inputs"]]      # the video and jl_in.tar under src/: content-addressed, uploaded once
    for r in shared:
        if r.get("local") and not Path(r["local"]).is_absolute():
            r["local"] = str(PACKAGE_ROOT.parent / r["local"])
    inputs = [texture_archive(window_id, job_file, smplx_dir)] + shared
    spec = worker.analyse_spec(job_id, J.TAKE, texture_steps(window_id, step_ms, top_k, body_ref_q, mask), texture_outputs(window_id), inputs=inputs,
                               max_minutes=95, priority=10, gpu=True, kind=kind)
    env = {k: v for k, v in (base.get("env") or {}).items() if k in ("HM_FRONT_DANCE", "HM_FRONT_CAPTURE")}
    spec.update({"code": {}, "retries": 0, "needs_gpu": True, "min_vram_gb": 12,   # measured: the bake peaks at 1.6 GB of GPU memory (487 views, 2048 atlas)
                
                 # no "owner" key: a spec with one is claimed only by workers with a matching HM_OWNER (the farm queue isolation); the dedicated kind isolates these jobs
                 "env": {**env, "DANCECAP_FARM_DANCE": J.TAKE, "DANCECAP_POD": "1", "HM_TEACHER_EVIDENCE": "1",
                         "HM_RED_TOP": "1"}})   # her top is brown / maroon / coral (and the hair auburn): the bake's red-hair leak guards would erase the print; the hair masks are real here
    return spec


def wait_for_gpu(wait_min: float, min_gb: float = 12.0, poll_s: float = 20.0) -> list[str]:
    """The names of the GPUs in stock in the volume's data centre with at least min_gb of memory (the bake peaks at 1.6 GB), the cheapest first, waiting up to
    wait_min minutes for one to appear (the stock comes and goes). Raises when none shows up."""
    from .farm import pods
    t_end = time.time() + wait_min * 60.0
    while True:
        offers = sorted((o for o in pods.gpu_offers() if float(o.get("memory_gb") or 0) >= min_gb), key=lambda o: o["price"])
        if offers:
            return [o["gpu"] for o in offers]
        if time.time() > t_end:
            raise RuntimeError(f"no GPU with {min_gb:g} GB in stock after {wait_min:g} min")
        log("no GPU with enough memory in stock: waiting")
        time.sleep(poll_s)


def texture_run(a) -> int:
    """`texture-run`: windows.json (texture-select) -> ONE pod job: the window's front + the bake on the sliced fitted poses."""
    from .recap_avatars_pod import slice_smplx
    doc = json.loads(a.windows.read_text())
    w = doc["windows"][a.which]
    t0, t1 = float(w["t0"]), float(w["t1"])
    mask = a.mask or ",".join(w.get("mask") or []) or None
    log(f"window {a.window_id}: {t0:.1f}-{t1:.1f} s, camera 01; masked {mask}")
    job = window_job(a.window_id, t0, t1)
    jdir = J.AV / "texture" / "jobs"
    jdir.mkdir(parents=True, exist_ok=True)
    jf = jdir / f"{a.window_id}.json"
    jf.write_text(json.dumps(job, indent=1))
    sm = PACKAGE_ROOT / "work" / texture_take(a.window_id) / "smplx"
    sl = slice_smplx(a.smplx_dir, sm, t0 - 0.5, t1 + 0.5)
    log(f"poses sliced: {sl} frames -> {sm}")
    jid = f"jl-texture-{a.window_id}-{time.strftime('%m%d%H%M')}"
    kind = f"{J.KIND}{a.window_id}"
    spec = texture_spec(jid, a.window_id, a.step_ms, jf, sm, a.top_k, a.body_ref_q, mask, kind)
    log(f"job {jid}: {len(spec['steps'])} steps, inputs {[i['path'] for i in spec['inputs']]}")
    if a.dry_run:
        print(json.dumps({"job": {k: spec[k] for k in ("id", "kind", "needs_gpu", "max_minutes", "env")}, "steps": [s["cmd"] for s in spec["steps"]]}, indent=1))
        return 0
    gpus = wait_for_gpu(a.wait_stock_min)
    res = run_job(spec, cpu=False, gpus=gpus, minutes=a.minutes, hard_minutes=a.hard_minutes, max_cost=a.max_cost, kind=kind)
    print(json.dumps(res, default=str))
    log(f"ledger spend of owner {J.OWNER}: {owner_spend()} USD")
    return 0 if res.get("status") == "done" else 1


TEXTURE_FILES = ("lead_albedo.png", "follow_albedo.png", "lead_coverage.png", "follow_coverage.png", "lead_texels.npz", "follow_texels.npz",
                 "uv_mesh.npz", "head_fix.npz", "report.json", "follow_hair_region.png", "follow_hair.json", "follow_hair_strip.png", "follow_hair_cover.png")


def owner_spend() -> float:
    from . import remote
    return round(sum(float(r.get("cost") or 0.0) for r in remote.ledger() if r.get("owner") == J.OWNER), 3)


def cli_job(a) -> int:
    """face-identity: build the spec, print the estimate, run (or --dry-run)."""
    jid = a.job_id or f"jl-identity-{time.strftime('%m%d%H%M')}"
    spec = identity_spec(jid)
    est = estimate()
    log(f"estimate: {est['total_min']} min = {est['usd']} USD at {est['price_hr']}/h (ceiling {a.max_cost}); inputs {spec['inputs'][0]['bytes'] / 1e6:.1f} MB")
    if a.dry_run:
        print(json.dumps({"job": {k: spec[k] for k in ("id", "kind", "needs_gpu", "max_minutes")}, "steps": [s["cmd"][:200] for s in spec["steps"]],
                          "outputs": [o["path"] for o in spec["outputs"]], "estimate": est}, indent=1))
        return 0
    res = run_job(spec, vcpu=a.vcpu, minutes=a.minutes, hard_minutes=a.hard_minutes, max_cost=a.max_cost, upload_models=a.upload_models)
    print(json.dumps(res, default=str))
    log(f"ledger spend of owner {J.OWNER}: {owner_spend()} USD")
    return 0 if res.get("status") == "done" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_avatars_pod")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pod-identity")
    p.add_argument("--work", type=Path)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    pod_identity(work=a.work, iters=a.iters, limit=a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
