"""john_lula profile of the farm planner (agent J3b, 2026-10-11): the Jack & Jill heat of John and Lula, ONE handheld camera, as an equal-window chunk job graph.

A take opts in with `[chunks] profile = "john_lula"` (takes/john_lula.toml); `farm plan --dance john_lula` then calls cmd_plan() below. It is carlos_aline's profile
(farm/ca_profile.py, untouched) with what a single camera changes:

  cameras    ONE phone that rotates and zooms about a fixed centre (J2: cameras/01.npz, per-frame K and R in the metric world frame, floor y = 0). No second view:
             nothing is triangulated, so ma_3d runs in its single-camera mode (MAMMA_MONO=1: mamma-turing optimization/utils/mono.py, the start from the ankle rays,
             a heading search, floor and stature terms) and dancecap.sync is not needed (the identity clock).
  people     J3a: people/masks_01.npz (SAM 2.1 per dancer, bib identity) and people/teachers_01.npz (133 whole-body keypoints): no SAM step on the pod
             (dancecap.jl_front: frames, cameras, masks, ma_2d, fixed sync, prepare, mono-init).
  ids        every job id starts with "jl" (queue/ and chunks/ of the volume are shared with other farm runs): jlshape, jl0000 ..., jlfinal. Every spec carries
             "owner": "J3b": a worker whose HM_OWNER differs never claims it (farm/worker.py), so another session's pods leave these jobs alone.
  shape      one 30 s window where both dancers are fully in the picture (not cropped, not hidden behind the partner, identities sure), fitted with FREE betas held to
             the owner's statures (MAMMA_STATURE: a single camera cannot separate size from distance); every chunk freezes them (await_pin).
  final      one CPU job: dancecap.stitch of the windows (the demo's seam gates).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import analysis as A
from . import chunks as C
from . import farm_dir, jobs_dir
from . import recap_profile as RP
from .state import State, atomic_write_json, now_iso
from .timeline import HZ, DanceSpec
from ..take import PACKAGE_ROOT

PROFILE = "john_lula"
DANCE = "john_lula"
CAPTURE = "John-Lula"
ID_PREFIX = "jl"
OWNER = "J3b"
SHAPE_ID = "jlshape"
FINAL_ID = "jlfinal"
SHAPE_TAKE = f"{DANCE}_{SHAPE_ID}"
IN_TAR_REL = f"src/{CAPTURE}/jl_in.tar"
PIN_REL = f"dancecap/work/{SHAPE_TAKE}/mamma_pin_betas.json"
WORK = PACKAGE_ROOT / "work" / DANCE
PEOPLE = WORK / "people"
CAMS = ("01",)
JOB_ENV = {"DANCECAP_FARM_DANCE": DANCE, "HM_FRONT_DANCE": DANCE, "HM_FRONT_CAPTURE": CAPTURE, "HM_SHAPE_JOB": SHAPE_ID, "MAMMA_MONO": "1"}

# rates (4090 minutes): ma_2d 8 camera-frames/s as the recap's model; one camera's single-body fit ~0.6 s per frame (+ 74 s fixed); the front has no SAM
MA2D_FPS, DECODE_FPS = 8.0, 25.0
FIT_S_PER_FRAME_MONO = 0.6
FRONT_FIXED_MIN = 5.0

SYNC_FIXED = {"no_measure": True, "offsets_ms": {"01": 0.0}, "why": "one camera: the video is its own reference clock (identity alignment), nothing to measure"}


def load_teachers() -> dict:
    p = PEOPLE / "teachers_01.npz"
    z = np.load(p, allow_pickle=True)
    return {k: np.asarray(z[k]) for k in ("t", "ok", "conf", "merged", "role_conf", "edge", "skel_in_partner", "visible")}


def shape_window(tt: dict, frames: int = 900, step_s: float = 2.5) -> dict:
    """The 30 s window (900 grid frames) in which both dancers are best fit for a SHAPE: per grid instant both dancers have pose confidence >= 0.5, are not merged
    (one box for two bodies), not cropped by the frame edge, not hidden behind the partner (skeleton on the partner < 0.3) and have a sure identity (>= 0.8)."""
    L = frames / HZ
    grid = np.arange(0.0, float(tt["t"][-1]), 1.0 / HZ)
    j = np.clip(np.searchsorted(tt["t"], grid), 1, len(tt["t"]) - 1)
    j = np.where(np.abs(tt["t"][j - 1] - grid) < np.abs(tt["t"][j] - grid), j - 1, j)
    good = ((tt["conf"][j] >= 0.5).all(1) & ~tt["merged"][j].any(1) & ~tt["edge"][j].any(1) & (tt["skel_in_partner"][j] < 0.3).all(1)
            & (tt["role_conf"][j] >= 0.8).all(1) & (np.abs(tt["t"][j] - grid) <= 0.05))
    best, rows = None, []
    for t0 in np.arange(0.0, grid[-1] - L, step_s):
        m = (grid >= t0) & (grid < t0 + L)
        s = float(good[m].mean())
        rows.append((round(float(t0), 2), round(s, 3)))
        if best is None or s > best[1]:
            best = (float(t0), s)
    k0 = int(round(best[0] * HZ))
    return {"t0": best[0], "t1": best[0] + L, "k0": k0, "k1": k0 + frames - 1, "score": round(best[1], 3), "scores": rows}


# ------------------------------------------------------------------------------------------------ the shared input tar


def build_in(out: Path | None = None) -> dict:
    """ONE deterministic tar of what a job needs and the volume does not hold: J2's camera track (with pts / native_size), recap_cams.json, world_from_sfm.json,
    J3a's masks and teacher tracks, the identity clock. Workspace-relative members; uploaded once to src/<capture>/ (shared by every job)."""
    out = out or PEOPLE / "in" / "jl_in.tar"
    rel = f"dancecap/work/{DANCE}"
    items = [(WORK / "cameras" / "01.npz", f"{rel}/cameras/01.npz"), (WORK / "cameras" / "recap_cams.json", f"{rel}/cameras/recap_cams.json"),
             (WORK / "cameras" / "world_from_sfm.json", f"{rel}/cameras/world_from_sfm.json"), (PEOPLE / "teachers_01.npz", f"{rel}/people/teachers_01.npz"),
             (PEOPLE / "masks_01.npz", f"{rel}/people/masks_01.npz"), (WORK / "clock" / "alignment.json", f"{rel}/clock/alignment.json")]
    missing = [str(p) for p, _ in items if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"jl_in.tar: missing {missing[:3]}: run python -m dancecap.jl_inputs prepare")
    C.pack_tar(out, items)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return {"path": IN_TAR_REL, "sha256": h.hexdigest(), "bytes": out.stat().st_size, "local": C._ws(out), "archive": True, "n_files": len(items)}


# ------------------------------------------------------------------------------------------------ estimates and job specs


def estimate_chunk(chunk: C.Chunk, cam_frames: int | None = None, shape: bool = False) -> dict:
    e = RP.estimate_chunk(chunk, cam_frames=cam_frames, shape=shape)
    cf = e["cam_frames"]
    front = (cf / MA2D_FPS + cf / DECODE_FPS) / 60.0 + FRONT_FIXED_MIN
    fit = (RP.FIT_A + FIT_S_PER_FRAME_MONO * chunk.T) / 60.0
    tot = front + fit + e["refine_min"] + e["quality_min"] + RP.BOOT_MIN
    e.update({"front_min": round(front, 1), "fit_min": round(fit, 1), "total_min_4090": round(tot, 1), "usd_4090": round(tot / 60 * RP.PRICE_4090, 2),
              "total_min": round(tot / RP.GPU_SLOWDOWN, 1)})
    return e


def chunk_steps(chunk: C.Chunk, est: dict, refine_args: str, *, shape: bool, refined_dir: str, variant: bool = False) -> list[dict]:
    """chunk_take -> front -> qa_mask -> [await_pin] -> fit -> [pin | refine -> quality] -> evidence. variant: a focal-hypothesis re-solve of a window (pinned betas, raw fit only)."""
    take = chunk.take
    ctx = {"chunk": chunk.id, "take": take}
    w = f"dancecap/work/{take}"
    fit_env = {"MAMMA_MONO_INIT": "{WS}/" + w + "/mamma/mono_init.npz"}
    if shape:
        fit_env["MAMMA_STATURE"] = "{WS}/" + w + "/mamma/stature.json"
    else:
        fit_env.update({"MAMMA_FREEZE_BETAS": "1", "MAMMA_INIT_BETAS": "{WS}/" + PIN_REL})
    steps = [C._step("chunk_take", C.fmt("python -m dancecap.farm chunk-take {chunk}", **ctx), 3),
             C._step("front", C.fmt("python -m dancecap.jl_front run {chunk}", **ctx), max(45, math.ceil(2.0 * est["front_min"] + 15)),
                     save=[f"{w}/mamma/prepare.json", f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/ma_2d/", f"{w}/mamma/ma_cap/", f"{w}/mamma/mono_init.npz",
                           f"{w}/mamma/mono_init.json", f"{w}/mamma/stature.json", f"{w}/sync/offsets.json", f"{w}/cameras/world_from_sfm.json", f"{w}/dancers/roles.json"]),
             C._step("qa_mask", C.fmt("python -m dancecap.recap_front qa-mask {chunk}", **ctx), 10, save=[f"{w}/mamma/ma_2d/", f"{w}/mamma/qa_mask.json"])]
    if not shape:
        steps.append(C._step("await_pin", C.fmt("python -m dancecap.recap_front await-pin {chunk}", **ctx), 130))
    steps.append(C._step("fit", C.fmt("python -m dancecap.recap_front fit {chunk}" + ("" if shape else " --pin"), **ctx), max(30, math.ceil(1.6 * est["fit_min"] + 15)),
                         env=fit_env, live=[f"{w}/mamma/ma_3d/"], save=[f"{w}/smplx/", f"{w}/mamma/fits/", f"{w}/mamma/select.json", f"{w}/mamma/report.json"]))
    if shape:
        steps.append(C._step("pin", C.fmt("python -m dancecap.recap_front pin {chunk}", **ctx), 3, save=[f"dancecap/work/{take}/mamma_pin_betas.json"]))
    elif not variant:
        steps += [C._step("refine", C.fmt("python -m dancecap.pose_refine {take} --src smplx --dst {d} {a}", d=refined_dir, a=refine_args, **ctx),
                          max(10, math.ceil(2 * est["refine_min"] + 5)), save=[f"{w}/{refined_dir}/"]),
                  C._step("quality", C.fmt("python -m dancecap.pose_quality {take} --smplx-dir {d} --tag rf2_noslip", d=refined_dir, **ctx), 20, optional=True)]
    last = "fit" if (shape or variant) else "refined"
    steps.append({"name": "evidence", "cmd": f"python -m dancecap.farm chunk-evidence {chunk.id} --stage {last}", "timeout_min": 10.0, "optional": False, "stage": last})
    for s in steps[:-1]:
        s["stage"] = {"chunk_take": "front", "front": "front", "qa_mask": "front", "await_pin": "front", "fit": "fit", "pin": "fit", "refine": "refined", "quality": "refined"}[s["name"]]
    return steps


def chunk_outputs(chunk: C.Chunk, *, shape: bool, refined_dir: str) -> list[dict]:
    rows = RP.chunk_outputs(chunk, shape=shape, refined_dir=refined_dir)
    for rel in ("mamma/qa_mask.json", "front/front_done.json", "mamma/mono_init.json", "mamma/mono_init.npz", "mamma/stature.json"):
        rows.append({"path": f"dancecap/work/{chunk.take}/{rel}", "optional": True})
    return rows


def build_job(chunk: C.Chunk, spec: DanceSpec, *, hash_inputs: bool = True, code: dict | None = None, shape: bool = False, in_tar: dict | None = None,
              variant: dict | None = None) -> dict:
    est = chunk.estimate if chunk.estimate.get("front_min") is not None and "cam_frames" in chunk.estimate else estimate_chunk(chunk, shape=shape or bool(variant))
    steps = chunk_steps(chunk, est, RP.refine_args_recipe(), shape=shape, refined_dir=spec.refined_dir, variant=bool(variant))
    if hash_inputs:
        inputs, sources = C.source_inputs(spec, chunk.cams)
        inputs.append(in_tar or build_in())
    else:
        inputs, sources = [{"path": IN_TAR_REL, "archive": True}], {}
    total = est["total_min_4090"]
    return {"schema": C.JOB_SCHEMA, "id": chunk.id, "dance": spec.name, "kind": "full", "what": "shape" if shape else ("variant" if variant else "chunk"), "stages": ["front", "fit"] + ([] if (shape or variant) else ["refined"]),
            "take": chunk.take, "chunk": {**chunk.to_dict(), "sync_fixed": SYNC_FIXED}, "cams": chunk.cams, "after": [], "world_id": None, "world_sha256": None, "pose_id": None,
            "code": code if code is not None else C.code_info(), "sources": sources, "inputs": inputs, "steps": steps,
            "outputs": [o for o in chunk_outputs(chunk, shape=shape or bool(variant), refined_dir=spec.refined_dir) if not (variant and "mamma_pin_betas" in o["path"])], "env": {**JOB_ENV, **((variant or {}).get("env") or {})}, "cwd": "dancecap", "owner": OWNER,
            "max_minutes": round(max(60.0, 1.6 * total + 20.0), 1), "priority": 100.0 if shape else float(total), "min_vram_gb": 24, "needs_gpu": True, "estimate": est,
            "created": now_iso(), "profile": PROFILE, "retries": 1}


def build_final(chunks: list[C.Chunk], spec: DanceSpec, *, code: dict | None = None) -> dict:
    cl = [c.to_dict() for c in chunks]
    job = A.build_final_job(spec, cl, None, analyses=(), jid=FINAL_ID, hash_inputs=False, code=code)
    for s in job["steps"]:
        if s["name"] == "stitch":
            s["cmd"] = s["cmd"].replace(" --require-world", "")
    job["inputs"] = [{"path": f"chunks/{c.id}/out/result.tar", "archive": True} for c in chunks]
    job["profile"] = PROFILE
    job["owner"] = OWNER
    job["env"] = {**(job.get("env") or {}), **JOB_ENV}
    return job


# ------------------------------------------------------------------------------------------------ the plan


def make_plan(spec: DanceSpec, *, hash_inputs: bool = True, code: dict | None = None, claimed_ids: set | None = None) -> dict:
    tt = load_teachers()
    g = spec.grid
    chunks, rep = C.plan_windows(spec, kind="full", claimed_ids=claimed_ids, cams_fn=lambda k0, k1: list(CAMS))
    rows = []
    for ch in chunks:
        ch.id = ID_PREFIX + ch.id
        ch.take = f"{spec.name}_{ch.id}"
        ch.cams = list(CAMS)
        ch.estimate = estimate_chunk(ch, cam_frames=ch.T)
        ch.priority = ch.estimate["total_min_4090"]
        rows.append({"id": ch.id, "k": [ch.k0, ch.k1], "core_k": [ch.core_k0, ch.core_k1], "window_s": round(ch.T / HZ, 2), "core_s": round(ch.core_T / HZ, 2),
                     "t": [round(float(g.t(ch.k0)), 2), round(float(g.t(ch.k1 + 1)), 2)], "cams": ch.cams, "estimate": ch.estimate})
    sw = shape_window(tt)
    shape = C.Chunk(id=SHAPE_ID, k0=sw["k0"], k1=sw["k1"], core_k0=sw["k0"], core_k1=sw["k1"], overlap_k=0, kind="full", cams=list(CAMS), world_id=None, origin="shape", take=SHAPE_TAKE)
    shape.estimate = estimate_chunk(shape, cam_frames=shape.T, shape=True)
    jobs = [build_job(shape, spec, hash_inputs=hash_inputs, code=code, shape=True)]
    in_tar = jobs[0]["inputs"][-1] if hash_inputs else None
    jobs += [build_job(ch, spec, hash_inputs=hash_inputs, code=code, in_tar=in_tar) for ch in chunks]
    final = build_final(chunks, spec, code=code)
    costs = RP.cost_table(jobs)
    return {"created": now_iso(), "dance": spec.name, "profile": PROFILE, "params": C.chunk_params(spec), "grid": g.to_dict(), "arithmetic": rep["arithmetic"], "chunks": rows,
            "shape": {"id": SHAPE_ID, "window": [round(sw["t0"], 2), round(sw["t1"], 2)], "k": [sw["k0"], sw["k1"]], "cams": list(CAMS), "score": sw["score"],
                      "scores": sw["scores"], "estimate": shape.estimate},
            "costs": costs, "seams": [{"between": [chunks[i].id, chunks[i + 1].id], "overlap_s": round((chunks[i].k1 - chunks[i + 1].k0 + 1) / HZ, 3),
                                       "cut_t": round(float(g.t(chunks[i + 1].core_k0)), 2)} for i in range(len(chunks) - 1)],
            "_chunks": chunks, "_shape": shape, "_jobs": jobs + [final]}


def validate_plan(plan: dict, spec: DanceSpec) -> dict:
    """RP.validate_plan's offline checks with this take's ids (the recap's checks hard-code its shape id and require three cameras)."""
    saved = RP.SHAPE_ID
    RP.SHAPE_ID = SHAPE_ID
    p2 = dict(plan)
    p2["_jobs"] = [dict(j, id="final") if j["id"] == FINAL_ID else j for j in plan["_jobs"]]
    try:
        v = RP.validate_plan(p2, spec)
    finally:
        RP.SHAPE_ID = saved
    v["checks"].pop("every_chunk_has_at_least_3_cameras", None)
    v["checks"]["every_chunk_has_the_camera"] = {"ok": all(c.cams == list(CAMS) for c in plan["_chunks"])}
    v["checks"]["job_env_selects_the_take_and_mono"] = {"ok": all(all((j.get("env") or {}).get(k) == val for k, val in JOB_ENV.items()) for j in plan["_jobs"]), "env": JOB_ENV}
    v["checks"]["ids_and_owner"] = {"ok": all(j["id"].startswith(ID_PREFIX) and j.get("owner") == OWNER for j in plan["_jobs"]), "ids": [j["id"] for j in plan["_jobs"]]}
    v["ok"] = all(c["ok"] for c in v["checks"].values())
    return v


def print_plan(plan: dict) -> None:
    ar = plan["arithmetic"]
    print(f"john_lula plan: {len(plan['chunks'])} chunks of {min(ar['window_s'])}-{max(ar['window_s'])} s, overlap {ar['overlap_k'] / HZ:g} s")
    print(f"{'id':<9}{'window (s)':<18}{'core (s)':<9}{'front':>6}{'fit':>6}{'ref+q':>6}{'4090 min':>9}{'USD':>6}")
    for r in plan["chunks"]:
        e = r["estimate"]
        print(f"{r['id']:<9}{r['t'][0]:>7.1f}-{r['t'][1]:<9.1f}{r['core_s']:<9}{e['front_min']:>6}{e['fit_min']:>6}{e['refine_min'] + e['quality_min']:>6.1f}{e['total_min_4090']:>9}{e['usd_4090']:>6}")
    s = plan["shape"]
    e = s["estimate"]
    print(f"{s['id']:<9}{s['window'][0]:>7.1f}-{s['window'][1]:<9.1f}{30.0:<9}{e['front_min']:>6}{e['fit_min']:>6}{'':>6}{e['total_min_4090']:>9}{e['usd_4090']:>6}  (shape: free betas held to the statures; score {s['score']})")
    c = plan["costs"]
    print(f"total: {c['gpu_hours']} GPU h = {c['usd']} USD on RTX 4090 pods ({c['usd_with_30pct_margin']} with a 30 % margin); wall about {c['wall_min_with_one_pod_per_job']} min with one pod per job")


def cmd_plan(a, spec: DanceSpec) -> int:
    st = State(a.dance)
    cur = st.load()
    claimed = {cid for cid, c in cur["chunks"].items() if c.get("status") in ("queued", "running", "done", "pulled", "stitched")}
    claimed |= {cid[len(ID_PREFIX):] for cid in claimed if cid.startswith(ID_PREFIX)}
    plan = make_plan(spec, hash_inputs=not a.dry_run, code=C.code_info(), claimed_ids=claimed)
    public = {k: v for k, v in plan.items() if not k.startswith("_")}
    public["validation"] = validate_plan(plan, spec)
    if a.json:
        print(json.dumps(public, default=str))
    else:
        print_plan(plan)
        v = public["validation"]
        print("validation (offline): " + ("OK" if v["ok"] else "FAILED") + " - " + ", ".join(f"{k}{'' if c['ok'] else ' FAILED'}" for k, c in v["checks"].items()))
    if a.out:
        atomic_write_json(Path(a.out), public)
    if a.dry_run:
        return 0
    jobs = plan["_jobs"]
    for j in jobs:
        (A.write_job if j.get("what") == "stitch_analyse" else C.write_job)(j)
    st.put_chunks([c.to_dict() for c in plan["_chunks"]], replace_planned=True)
    for j in jobs:
        st.set_job(j["id"], chunk=j["id"] if j.get("what") != "stitch_analyse" else None, kind=j.get("what") or j["kind"], status="planned", after=j.get("after") or [],
                   written=now_iso(), spec=str(jobs_dir(a.dance) / f"{j['id']}.json"))
    atomic_write_json(farm_dir(a.dance) / "plan.json", public)
    atomic_write_json(WORK / "chunks.json", public)
    print(f"{len(jobs)} job spec(s) -> {jobs_dir(a.dance)}   chunk plan -> {WORK / 'chunks.json'}   next: python -m dancecap.farm --dance {a.dance} up --dry-run")
    return 0


# ------------------------------------------------------------------------------------------------ focal-hypothesis variants (the depth sanity check)


def variant_jobs(spec: DanceSpec, k0: int, k1: int, scales=(0.9, 1.1), code: dict | None = None) -> list[dict]:
    """One job per focal scale: the window k0..k1 re-solved with every frame's focal multiplied by the scale (JL_FOCAL_SCALE; rotation and camera centre unchanged), betas pinned
    to the shape pass, raw fit only. Ids jlfm090 / jlfp110 ...; takes john_lula_<id>."""
    in_tar = build_in()
    jobs = []
    for sc in scales:
        cid = f"jlf{'m' if sc < 1 else 'p'}{int(round(sc * 100)):03d}"
        ch = C.Chunk(id=cid, k0=k0, k1=k1, core_k0=k0, core_k1=k1, overlap_k=0, kind="full", cams=list(CAMS), world_id=None, origin="variant", take=f"{spec.name}_{cid}")
        ch.estimate = estimate_chunk(ch, cam_frames=ch.T, shape=True)
        jobs.append(build_job(ch, spec, hash_inputs=True, code=code, in_tar=in_tar, variant={"env": {"JL_FOCAL_SCALE": repr(float(sc))}}))
    return jobs


def cmd_variants(spec: DanceSpec, k0: int, k1: int, scales=(0.9, 1.1)) -> list[str]:
    jobs = variant_jobs(spec, k0, k1, scales, code=C.code_info())
    st = State(spec.name)
    for j in jobs:
        C.write_job(j)
        st.set_job(j["id"], chunk=j["id"], kind="variant", status="planned", after=[], written=now_iso(), spec=str(jobs_dir(spec.name) / f"{j['id']}.json"))
    return [j["id"] for j in jobs]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m dancecap.farm.jl_profile")
    ap.add_argument("cmd", choices=("variants",))
    ap.add_argument("--k0", type=int, required=True)
    ap.add_argument("--k1", type=int, required=True)
    ap.add_argument("--scales", default="0.9,1.1")
    a = ap.parse_args(argv)
    print(",".join(cmd_variants(DanceSpec.load(DANCE), a.k0, a.k1, tuple(float(x) for x in a.scales.split(",")))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------------------------------------------------ the synthetic ground-truth check (dancecap.mono_synth)


def synth_job(spec: DanceSpec, frames: int = 300, noise_px: float = 2.0, init_noise: float = 0.25, code: dict | None = None) -> dict:
    """One pod job: dancecap.mono_synth run (the synthetic couple seen by a john_lula-like camera, fitted at the true focal and at x0.9 / x1.1) -> work/synthetic_mono/report.json."""
    gt = PACKAGE_ROOT / "work" / "synthetic_mamma" / "gt120_anat" / "smplx"
    items = [(gt / f"{r}.npz", f"dancecap/work/synthetic_mamma/gt120_anat/smplx/{r}.npz") for r in ("lead", "follow")]
    tar = PEOPLE / "in" / "jl_synth_in.tar"
    C.pack_tar(tar, items)
    h = hashlib.sha256(tar.read_bytes()).hexdigest()
    cid = "jlsynth"
    ch = C.Chunk(id=cid, k0=0, k1=frames - 1, core_k0=0, core_k1=frames - 1, overlap_k=0, kind="full", cams=list(CAMS), world_id=None, origin="synthetic", take=f"{spec.name}_{cid}")
    ch.estimate = estimate_chunk(ch, cam_frames=ch.T, shape=True)
    job = build_job(ch, spec, hash_inputs=False, code=code, variant={"env": {}})
    cmd = f"python -m dancecap.mono_synth run --frames {frames} --noise {noise_px} --init-noise {init_noise} --scales 1.0,0.9,1.1"
    job.update({"what": "synth", "stages": ["fit"], "inputs": [{"path": f"src/{CAPTURE}/jl_synth_in.tar", "archive": True, "sha256": h, "bytes": tar.stat().st_size, "local": C._ws(tar)}],
                "steps": [{**C._step("synth", cmd, 90, save=["dancecap/work/synthetic_mono/"]), "stage": "fit"}, {"name": "evidence", "cmd": "true", "timeout_min": 1.0, "optional": True, "stage": "fit"}],
                "outputs": [{"path": "dancecap/work/synthetic_mono/report.json"}], "priority": 50.0, "max_minutes": 100.0})
    return job
