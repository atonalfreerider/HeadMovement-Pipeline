"""carlos_aline profile of the farm planner (agent C3, 2026-10-10): the Jack & Jill heat of Carlos and Aline as an equal-window chunk job graph.

A take opts in with `[chunks] profile = "carlos_aline"` (takes/carlos_aline.toml); `farm plan --dance carlos_aline` then calls cmd_plan() below. It is the class
recap's profile (farm/recap_profile.py, which stays untouched) made take-agnostic where the heat differs:

  cameras    TWO tripod heads that pan / tilt (01 also zooms): C2's per-frame tracks (cameras/<cam>.npz) are inputs; camera 02 starts 14.8 s after 01, so the
             first 11.2 s of the danced span are monocular ([stitch] mono in the take TOML flags them).
  people     NOT the owner's 2D JSON: YOLO11x-pose of every person (C2's pods) as pose tables, and carlos_aline_people (zone + static-furniture veto + colour +
             tracking) as the teacher tracks: people/teachers_<cam>.npz in the recap's contract, so dancecap.recap_front runs unchanged on them (job env
             HM_FRONT_DANCE / HM_FRONT_CAPTURE / HM_SHAPE_JOB select the take; defaults = the recap).
  ids        every job id starts with "ca" (queue/ and chunks/ of the volume are shared with the recap's g0000 ... and shape): cag0000 ..., cashape, cafinal.
  steps      the recap's chunk steps + the T1 `qa_mask` step (zero the landmarks that are not on the teacher) before the fit waits for the pinned betas.
  shape      a 30 s window with BOTH cameras on both dancers, the couple apart / not merged (so the betas see the whole body), fitted with free betas;
             every chunk freezes them (await_pin), the stitcher verifies they are identical; no pin to any other take.
  final      one CPU job: dancecap.stitch of the windows (the demo's seam gates), no analysis, no world check.
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
from ..take import PACKAGE_ROOT, WORKSPACE

PROFILE = "carlos_aline"
DANCE = "carlos_aline"
CAPTURE = "Carlos-Aline"
ID_PREFIX = "ca"
SHAPE_ID = "cashape"
FINAL_ID = "cafinal"
SHAPE_TAKE = f"{DANCE}_{SHAPE_ID}"
IN_TAR_REL = f"src/{CAPTURE}/ca_in.tar"
PIN_REL = f"dancecap/work/{SHAPE_TAKE}/mamma_pin_betas.json"
WORK = PACKAGE_ROOT / "work" / DANCE
PEOPLE = WORK / "people"
CAMS = ("01", "02")
ALIGNMENT = WORKSPACE / "prototype" / "larissa_kadu" / "alignment_carlos_aline.json"
JOB_ENV = {"DANCECAP_FARM_DANCE": DANCE, "HM_FRONT_DANCE": DANCE, "HM_FRONT_CAPTURE": CAPTURE, "HM_SHAPE_JOB": SHAPE_ID}
CONF = 0.5

# the fit of two cameras is cheaper per frame than the recap's six (1.62 s/frame measured with 6): scaled by this share of the 6-camera rate, never below the drill's model
FIT_S_PER_FRAME_2CAM = 0.9

# dancecap.sync needs >= 2 OTHER cameras per camera: a two-camera take cannot be measured by it (the first pods died at the sync step). The video offsets of the pair are
# the two-view reprojection minimum of the YOLO joints of both dancers against the shift of camera 02's frames (work/carlos_aline/people/video_offset.json: minimum at
# +62 ms, i.e. camera 02's frame labelled t + 0.062 s shows what camera 01's frame labelled t shows, so camera 02's delta is -62 ms; C2 found 60 ms the same way).
SYNC_FIXED = {"no_measure": True, "offsets_ms": {"01": 0.0, "02": -62.0},
              "why": "two cameras: dancecap.sync cannot measure; two-view joint reprojection minimum (C3, people/video_offset.json), C2 -60 ms"}


def load_teachers(cams=CAMS) -> dict:
    """{cam: {t, ok, conf (F, 2)}} and the per-frame 'merged' flags from people/teachers_<cam>.npz."""
    out = {}
    for c in cams:
        p = PEOPLE / f"teachers_{c}.npz"
        if p.exists():
            z = np.load(p)
            out[c] = {"t": np.asarray(z["t"], float), "ok": np.asarray(z["ok"], bool), "conf": np.asarray(z["conf"], float), "merged": np.asarray(z["merged"], bool),
                      "floor": np.asarray(z["floor"], float)}
    return out


def shape_window(tt: dict, frames: int = 900, step_s: float = 2.5, t_min: float = 14.9, t_max: float = 197.0) -> dict:
    """The 30 s window (900 grid frames) where BOTH cameras see BOTH dancers best and the couple is not in a clinch: per 30 Hz grid instant, the share of the two cameras
    that have both dancers with confidence >= CONF (weight 0.55), the share of instants whose two detections are apart (floor distance in camera 01 >= 0.9 m, weight 0.3) and
    not merged (weight 0.15). Only instants after camera 02 starts (monocular frames carry no second view)."""
    L = frames / HZ
    grid = np.arange(0.0, 200.0, 1.0 / HZ)
    both = np.zeros((len(grid), len(tt)), bool)
    apart = np.zeros(len(grid), bool)
    clear = np.ones(len(grid), bool)
    for n, (c, o) in enumerate(tt.items()):
        j = np.clip(np.searchsorted(o["t"], grid), 1, len(o["t"]) - 1)
        j = np.where(np.abs(o["t"][j - 1] - grid) < np.abs(o["t"][j] - grid), j - 1, j)
        good = (np.abs(o["t"][j] - grid) <= 0.05) & o["ok"][j]
        both[:, n] = good & (o["conf"][j] >= CONF).all(1)
        clear &= ~(good & o["merged"][j].any(1)) | ~good
        if n == 0:
            d = np.linalg.norm(o["floor"][j, 0] - o["floor"][j, 1], axis=-1)
            apart = good & (d >= 0.9)
    score = 0.55 * both.mean(1) + 0.30 * apart + 0.15 * clear
    best, rows = None, []
    for t0 in np.arange(t_min, t_max - L, step_s):
        m = (grid >= t0) & (grid < t0 + L)
        s = float(score[m].mean() * (both[m].all(1).mean() ** 0.5))          # a window where both cameras really carry both dancers
        rows.append((round(float(t0), 2), round(s, 3)))
        if best is None or s > best[1]:
            best = (float(t0), s, {"both_cameras_both_dancers": round(float(both[m].all(1).mean()), 3), "apart": round(float(apart[m].mean()), 3), "not_merged": round(float(clear[m].mean()), 3)})
    k0 = int(round(best[0] * HZ))
    return {"t0": best[0], "t1": best[0] + L, "k0": k0, "k1": k0 + frames - 1, "score": round(best[1], 3), "stats": best[2], "scores": rows}


def window_cams(tt: dict, t0: float, t1: float) -> tuple[list[str], dict]:
    """Cameras of a window: those that record >= 35 % of it with at least one dancer in 30 % of their frames (the recap's rule, RP.choose_cams, on the teacher tracks)."""
    stats = RP.window_stats({c: {"t": o["t"], "ok": o["ok"], "conf": o["conf"]} for c, o in tt.items()}, t0, t1)
    return RP.choose_cams(stats)[0], stats


# ------------------------------------------------------------------------------------------------ the shared input tar


def build_in(out: Path | None = None) -> dict:
    """ONE deterministic tar of what a chunk job needs and the volume does not hold: C2's camera tracks (with pts / native_size), recap_cams.json, world_from_sfm.json, the
    teacher tracks, the pose tables (kp, n only) and the alignment file. Workspace-relative members; uploaded once to src/<capture>/ (shared by every job)."""
    out = out or PEOPLE / "in" / "ca_in.tar"
    rel = f"dancecap/work/{DANCE}"
    slim = PEOPLE / "in" / "pose_tables"
    slim.mkdir(parents=True, exist_ok=True)
    items = []
    for c in CAMS:
        z = np.load(PEOPLE / "pose_tables" / f"{c}.npz")
        np.savez_compressed(slim / f"{c}.npz", kp=z["kp"], n=z["n"])
        items += [(WORK / "cameras" / f"{c}.npz", f"{rel}/cameras/{c}.npz"), (PEOPLE / f"teachers_{c}.npz", f"{rel}/people/teachers_{c}.npz"), (slim / f"{c}.npz", f"{rel}/people/pose_tables/{c}.npz")]
    items += [(WORK / "cameras" / "recap_cams.json", f"{rel}/cameras/recap_cams.json"), (WORK / "cameras" / "world_from_sfm.json", f"{rel}/cameras/world_from_sfm.json"),
              (ALIGNMENT, "prototype/larissa_kadu/alignment_carlos_aline.json")]
    missing = [str(p) for p, _ in items if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"ca_in.tar: missing {missing[:3]}")
    C.pack_tar(out, items)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    cams = {c: hashlib.sha256((WORK / "cameras" / f"{c}.npz").read_bytes()).hexdigest()[:16] for c in CAMS}
    return {"path": IN_TAR_REL, "sha256": h.hexdigest(), "bytes": out.stat().st_size, "local": C._ws(out), "archive": True, "n_files": len(items), "cameras": cams}


# ------------------------------------------------------------------------------------------------ estimates and job specs


def estimate_chunk(chunk: C.Chunk, cam_frames: int | None = None, shape: bool = False) -> dict:
    e = RP.estimate_chunk(chunk, cam_frames=cam_frames, shape=shape)
    fit = (RP.FIT_A + FIT_S_PER_FRAME_2CAM * chunk.T) / 60.0
    tot = e["front_min"] + fit + e["refine_min"] + e["quality_min"] + RP.BOOT_MIN
    e.update({"fit_min": round(fit, 1), "total_min_4090": round(tot, 1), "usd_4090": round(tot / 60 * RP.PRICE_4090, 2), "total_min": round(tot / RP.GPU_SLOWDOWN, 1)})
    return e


def chunk_steps(chunk: C.Chunk, est: dict, refine_args: str, *, shape: bool, refined_dir: str) -> list[dict]:
    """chunk_take -> front -> qa_mask -> [await_pin] -> fit -> [pin | refine -> quality] -> evidence (RP.chunk_steps with this take's pin and the qa_mask step)."""
    take = chunk.take
    ctx = {"chunk": chunk.id, "take": take}
    w = f"dancecap/work/{take}"
    steps = [C._step("chunk_take", C.fmt("python -m dancecap.farm chunk-take {chunk}", **ctx), 3),
             C._step("front", C.fmt("python -m dancecap.recap_front run {chunk}", **ctx), max(45, math.ceil(2.0 * est["front_min"] + 15)),
                     save=[f"{w}/mamma/prepare.json", f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/ma_2d/", f"{w}/mamma/ma_cap/", f"{w}/sync/offsets.json",
                           f"{w}/cameras/world_from_sfm.json", f"{w}/dancers/roles.json"]),
             C._step("qa_mask", C.fmt("python -m dancecap.recap_front qa-mask {chunk}", **ctx), 10, save=[f"{w}/mamma/ma_2d/", f"{w}/mamma/qa_mask.json"])]
    if not shape:
        steps.append(C._step("await_pin", C.fmt("python -m dancecap.recap_front await-pin {chunk}", **ctx), 130))
    steps.append(C._step("fit", C.fmt("python -m dancecap.recap_front fit {chunk}" + ("" if shape else " --pin"), **ctx), max(30, math.ceil(1.6 * est["fit_min"] + 15)),
                         env=({} if shape else {"MAMMA_FREEZE_BETAS": "1", "MAMMA_INIT_BETAS": "{WS}/" + PIN_REL}), live=[f"{w}/mamma/ma_3d/"],
                         save=[f"{w}/smplx/", f"{w}/mamma/fits/", f"{w}/mamma/select.json", f"{w}/mamma/report.json"]))
    if shape:
        steps.append(C._step("pin", C.fmt("python -m dancecap.recap_front pin {chunk}", **ctx), 3, save=[f"dancecap/work/{take}/mamma_pin_betas.json"]))
    else:
        steps += [C._step("refine", C.fmt("python -m dancecap.pose_refine {take} --src smplx --dst {d} {a}", d=refined_dir, a=refine_args, **ctx),
                          max(10, math.ceil(2 * est["refine_min"] + 5)), save=[f"{w}/{refined_dir}/"]),
                  C._step("quality", C.fmt("python -m dancecap.pose_quality {take} --smplx-dir {d} --tag rf2_noslip", d=refined_dir, **ctx), 20, optional=True)]
    last = "fit" if shape else "refined"
    steps.append({"name": "evidence", "cmd": f"python -m dancecap.farm chunk-evidence {chunk.id} --stage {last}", "timeout_min": 10.0, "optional": False, "stage": last})
    for s in steps[:-1]:
        s["stage"] = {"chunk_take": "front", "front": "front", "qa_mask": "front", "await_pin": "front", "fit": "fit", "pin": "fit", "refine": "refined", "quality": "refined"}[s["name"]]
    return steps


def chunk_outputs(chunk: C.Chunk, *, shape: bool, refined_dir: str) -> list[dict]:
    rows = RP.chunk_outputs(chunk, shape=shape, refined_dir=refined_dir)
    rows.append({"path": f"dancecap/work/{chunk.take}/mamma/qa_mask.json", "optional": True})
    rows.append({"path": f"dancecap/work/{chunk.take}/front/front_done.json", "optional": True})
    return rows


def build_job(chunk: C.Chunk, spec: DanceSpec, *, hash_inputs: bool = True, code: dict | None = None, shape: bool = False, in_tar: dict | None = None) -> dict:
    est = chunk.estimate if chunk.estimate.get("front_min") is not None and "cam_frames" in chunk.estimate else estimate_chunk(chunk, shape=shape)
    steps = chunk_steps(chunk, est, RP.refine_args_recipe(), shape=shape, refined_dir=spec.refined_dir)
    if hash_inputs:
        inputs, sources = C.source_inputs(spec, chunk.cams)
        inputs.append(in_tar or build_in())
    else:
        inputs, sources = [{"path": IN_TAR_REL, "archive": True}], {}
    total = est["total_min_4090"]
    return {"schema": C.JOB_SCHEMA, "id": chunk.id, "dance": spec.name, "kind": "full", "what": "shape" if shape else "chunk", "stages": ["front", "fit"] + ([] if shape else ["refined"]),
            "take": chunk.take, "chunk": {**chunk.to_dict(), "sync_fixed": SYNC_FIXED}, "cams": chunk.cams, "after": [], "world_id": None, "world_sha256": None, "pose_id": None,
            "code": code if code is not None else C.code_info(), "sources": sources, "inputs": inputs, "steps": steps,
            "outputs": chunk_outputs(chunk, shape=shape, refined_dir=spec.refined_dir), "env": dict(JOB_ENV), "cwd": "dancecap",
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
    job["env"] = {**(job.get("env") or {}), **JOB_ENV}
    return job


# ------------------------------------------------------------------------------------------------ the plan


def make_plan(spec: DanceSpec, *, hash_inputs: bool = True, code: dict | None = None, claimed_ids: set | None = None) -> dict:
    tt = load_teachers()
    if len(tt) < 2:
        raise FileNotFoundError("people/teachers_<cam>.npz missing: run dancecap.carlos_aline_people run first")
    g = spec.grid
    chunks, rep = C.plan_windows(spec, kind="full", claimed_ids=claimed_ids, cams_fn=lambda k0, k1: window_cams(tt, g.t(k0), g.t(k1 + 1))[0])
    rows = []
    for ch in chunks:
        ch.id = ID_PREFIX + ch.id                       # the volume's chunks/ and queue/ are shared with the class recap (g0000 ...)
        ch.take = f"{spec.name}_{ch.id}"
        cams, st = window_cams(tt, g.t(ch.k0), g.t(ch.k1 + 1))
        ch.cams = cams
        ch.estimate = estimate_chunk(ch, cam_frames=sum(st[c]["frames"] for c in cams))
        ch.priority = ch.estimate["total_min_4090"]
        rows.append({"id": ch.id, "k": [ch.k0, ch.k1], "core_k": [ch.core_k0, ch.core_k1], "window_s": round(ch.T / HZ, 2), "core_s": round(ch.core_T / HZ, 2),
                     "t": [round(float(g.t(ch.k0)), 2), round(float(g.t(ch.k1 + 1)), 2)], "cams": cams, "cams_stats": st, "estimate": ch.estimate,
                     "monocular_until_s": 14.82 if g.t(ch.k0) < 14.82 else None})
    sw = shape_window(tt)
    scams, _ = window_cams(tt, sw["t0"], sw["t1"])
    shape = C.Chunk(id=SHAPE_ID, k0=sw["k0"], k1=sw["k1"], core_k0=sw["k0"], core_k1=sw["k1"], overlap_k=0, kind="full", cams=scams, world_id=None, origin="shape", take=SHAPE_TAKE)
    shape.estimate = estimate_chunk(shape, cam_frames=sum(int(HZ * 30) * 1 for _ in scams), shape=True)
    jobs = [build_job(shape, spec, hash_inputs=hash_inputs, code=code, shape=True)]
    in_tar = jobs[0]["inputs"][-1] if hash_inputs else None
    jobs += [build_job(ch, spec, hash_inputs=hash_inputs, code=code, in_tar=in_tar) for ch in chunks]
    final = build_final(chunks, spec, code=code)
    costs = RP.cost_table(jobs)
    return {"created": now_iso(), "dance": spec.name, "profile": PROFILE, "params": C.chunk_params(spec), "grid": g.to_dict(), "arithmetic": rep["arithmetic"], "chunks": rows,
            "shape": {"id": SHAPE_ID, "window": [round(sw["t0"], 2), round(sw["t1"], 2)], "k": [sw["k0"], sw["k1"]], "cams": scams, "score": sw["score"], "stats": sw["stats"],
                      "scores": sw["scores"], "estimate": shape.estimate},
            "costs": costs, "seams": [{"between": [chunks[i].id, chunks[i + 1].id], "overlap_s": round((chunks[i].k1 - chunks[i + 1].k0 + 1) / HZ, 3),
                                       "cut_t": round(float(g.t(chunks[i + 1].core_k0)), 2)} for i in range(len(chunks) - 1)],
            "_chunks": chunks, "_shape": shape, "_jobs": jobs + [final]}


def validate_plan(plan: dict, spec: DanceSpec) -> dict:
    """RP.validate_plan's offline checks with this take's ids (the recap's checks hard-code its shape id and require three cameras)."""
    saved = RP.SHAPE_ID
    RP.SHAPE_ID = SHAPE_ID
    p2 = dict(plan)
    p2["_jobs"] = [dict(j, id="final") if j["id"] == FINAL_ID else j for j in plan["_jobs"]]      # the recap's check names the final job "final"
    try:
        v = RP.validate_plan(p2, spec)
    finally:
        RP.SHAPE_ID = saved
    v["checks"].pop("every_chunk_has_at_least_3_cameras", None)
    v["checks"]["every_chunk_has_both_cameras"] = {"ok": all(len(c.cams) == 2 for c in plan["_chunks"]), "cams": {c.id: c.cams for c in plan["_chunks"]}}
    envs_ok = all(all((j.get("env") or {}).get(k) == val for k, val in JOB_ENV.items()) for j in plan["_jobs"])
    v["checks"]["job_env_selects_the_take"] = {"ok": envs_ok, "env": JOB_ENV}
    ids_ok = all(j["id"].startswith(ID_PREFIX) for j in plan["_jobs"])
    v["checks"]["ids_do_not_meet_the_recap"] = {"ok": ids_ok, "ids": [j["id"] for j in plan["_jobs"]]}
    v["ok"] = all(c["ok"] for c in v["checks"].values())
    return v


def print_plan(plan: dict) -> None:
    ar = plan["arithmetic"]
    print(f"carlos_aline plan: {len(plan['chunks'])} chunks of {min(ar['window_s'])}-{max(ar['window_s'])} s, overlap {ar['overlap_k'] / HZ:g} s "
          f"(n = {ar['n']}: " + ", ".join(f"{t['n']} x {t['window_s']} s" for t in ar["tried"][-3:]) + ")")
    print(f"{'id':<9}{'window (s)':<18}{'core (s)':<9}{'cams':<8}{'front':>6}{'fit':>6}{'ref+q':>6}{'4090 min':>9}{'USD':>6}")
    for r in plan["chunks"]:
        e = r["estimate"]
        print(f"{r['id']:<9}{r['t'][0]:>7.1f}-{r['t'][1]:<9.1f}{r['core_s']:<9}{','.join(r['cams']):<8}{e['front_min']:>6}{e['fit_min']:>6}{e['refine_min'] + e['quality_min']:>6.1f}"
              f"{e['total_min_4090']:>9}{e['usd_4090']:>6}")
    s = plan["shape"]
    e = s["estimate"]
    print(f"{s['id']:<9}{s['window'][0]:>7.1f}-{s['window'][1]:<9.1f}{30.0:<9}{','.join(s['cams']):<8}{e['front_min']:>6}{e['fit_min']:>6}{'':>6}{e['total_min_4090']:>9}{e['usd_4090']:>6}  (shape: free betas -> pin; {s['stats']})")
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
