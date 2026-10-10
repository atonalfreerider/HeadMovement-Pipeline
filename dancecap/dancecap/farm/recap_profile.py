"""Recap profile of the farm planner: the class recap (larissa_kadu_recap) as a 60-second-chunk job graph (agent R3, 2026-10-09).

A take opts in with `[chunks] profile = "recap"` in its TOML (see takes/larissa_kadu_recap.toml); `farm plan --dance larissa_kadu_recap` then
calls cmd_plan() below instead of the demo's gap planner, and everything else of the farm (queue, up, watch, pull, coverage, status) works on the
specs it writes unchanged. Nothing of the demo's planning changes for a take without that key.

WHAT IS DIFFERENT FROM THE DEMO (and why):
  windows    7 EQUAL fit windows of at most 60 s that overlap their neighbour by 4 s (chunks.equal_windows: 54.9 s each, cores cut at the middle
             of the overlaps) instead of 30 s cores + 4 s into each neighbour. The arithmetic is in chunks.json.
  cameras    per chunk the phones that see BOTH teachers (people/teachers_<cam>.npz of recap_people): the coverage map decides, not the clock span.
  front      no localisation against the room map and no colour-rule dancer search: R2's per-frame camera tracks and R3's teacher tracks are INPUTS
             (one shared tar on the volume); dancecap.recap_front slices them to the chunk and runs the demo's SAM 2.1 / ma_2d / sync / prepare.
  betas      NOT the demo's. One SHAPE job (`shape`, a 30 s window with the best camera coverage) fits with free betas; its betas become the pin
             (MAMMA_INIT_BETAS + MAMMA_FREEZE_BETAS, the demo's mechanism) of every chunk job, which therefore carry one body shape per teacher and
             stitch without re-expression. The chunk jobs are claimable at once (F1, 2026-10-09): their front runs in parallel with the shape pass and
             the step `await_pin` (recap_front) waits on the volume for the shape job's result before the fit.
  clocks     per-phone VIDEO offsets are measured per chunk by dancecap.sync and re-gauged so that the master camera 01 has offset 0 (the same gauge in
             every chunk); the demo pinned its gauge to a reference take, which the recap does not have.
  final      one CPU job: dancecap.stitch of the 7 chunks (the demo's seam gates), NO analysis stage (no soundtrack: timing / physics / counterbalance
             / floorcraft are beat based) and no world check (the recap has no pose world: its cameras are R2's, its betas the shape job's).
All jobs are kind front / fit-class GPU jobs for the worker (`needs_gpu`), the final one CPU. world_id is None: the room is the demo's but the clocks, camera
types and betas are not, so the demo's world package must not be applied (R1: NOTES_R1_sync.md section 6).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from . import chunks as C
from . import analysis as A
from . import farm_dir, jobs_dir, work_dir
from .state import State, atomic_write_json, now_iso
from .timeline import HZ, DanceSpec
from ..take import PACKAGE_ROOT, WORKSPACE

PROFILE = "recap"
DANCE = "larissa_kadu_recap"
CAPTURE = "Larissa-Kadu-recap"
SHAPE_ID = "shape"
SHAPE_TAKE = f"{DANCE}_shape"
IN_TAR_REL = f"src/{CAPTURE}/recap_in.tar"                 # shared on the volume (uploaded once), extracted into every job's workspace
PIN_REL = f"dancecap/work/{SHAPE_TAKE}/mamma_pin_betas.json"
WORK = PACKAGE_ROOT / "work" / DANCE
PEOPLE = WORK / "people"
CAMS = ("00", "01", "02", "03", "05", "06")
ATTACHED = ("05", "06")
RECIPE_PATH = PACKAGE_ROOT / "work" / "worlds" / "larissa_kadu_full" / "612dfea3d607" / "shared" / "refine_recipe.json"

# --- camera rule: a phone joins a chunk when it records in most of the window and sees both teachers in a useful share of its frames
MIN_RECORD_SHARE = 0.35         # of the window's 30 Hz frames, the phone has a localised canonical frame (a phone that starts or stops inside the
                                # window still serves the rest of it: mamma prepare --coverage union)
MIN_BOTH_SHARE = 0.30           # of the phone's frames in the window, both teachers are associated with confidence >= CONF
MIN_ONE_SHARE = 0.60            # ... or one teacher in this share and the other in at least MIN_ONE_OTHER
MIN_ONE_OTHER = 0.15
CONF = 0.5

# --- time model on one RTX 4090 pod (EU-RO-1, 0.89 USD/h). Provenance: work/larissa_kadu_full/PLAN.md 4.1 (SAM 0.30-0.37 s/frame with 5 side by side,
#     ma_2d 0.13-0.27 s/frame with 2 side by side, A100), farm/drill (ma_3d fit s = 74 + 0.92 x frames on the 4090, 6.4 GiB per 900 frames, refine 19-32 s).
PRICE_4090 = 0.89
SAM_FPS, MA2D_FPS, DECODE_FPS = 12.0, 8.0, 25.0   # camera-frames per second of the whole pod (4 / 2 / 3 processes side by side)
FRONT_FIXED_MIN = 8.0                              # sync (4 min), prepare (2), chunk-take / env / pull (2)
FIT_A, FIT_B = 74.0, 0.92                          # fit seconds = A + B x frames
QUALITY_MIN, BOOT_MIN = 3.0, 3.0
GPU_SLOWDOWN = 0.45                                # farm.launch.GPU_SLOWDOWN: A100 minutes x this = RTX 4090 minutes


# ------------------------------------------------------------------------------------------------ coverage of the teachers


def load_teachers(cams=CAMS) -> dict:
    """{cam: {t, ok, conf (F,2), frame_index}} from people/teachers_<cam>.npz (recap_people.finalize)."""
    out = {}
    for c in cams:
        p = PEOPLE / f"teachers_{c}.npz"
        if p.exists():
            z = np.load(p)
            out[c] = {"t": np.asarray(z["t"], float), "ok": np.asarray(z["ok"], bool), "conf": np.asarray(z["conf"], float)}
    return out


def window_stats(tt: dict, t0: float, t1: float) -> dict:
    """Per camera: frames in [t0, t1], share of the window's 30 Hz frames, share of the camera's frames with both / lead / follow teacher."""
    out = {}
    n_win = max(1.0, (t1 - t0) * HZ)
    for c, o in tt.items():
        m = o["ok"] & (o["t"] >= t0) & (o["t"] < t1)
        n = int(m.sum())
        row = {"frames": n, "record_share": round(n / n_win, 3)}
        if n:
            L, F = o["conf"][m, 0] >= CONF, o["conf"][m, 1] >= CONF
            row.update(lead=round(float(L.mean()), 3), follow=round(float(F.mean()), 3), both=round(float((L & F).mean()), 3))
        else:
            row.update(lead=0.0, follow=0.0, both=0.0)
        out[c] = row
    return out


def choose_cams(stats: dict) -> tuple[list[str], dict]:
    """The camera rule (constants above). Returns (cams, {cam: reason})."""
    cams, why = [], {}
    for c, s in sorted(stats.items()):
        if s["record_share"] < MIN_RECORD_SHARE:
            why[c] = f"records {100 * s['record_share']:.0f} % of the window (< {100 * MIN_RECORD_SHARE:.0f} %)"
        elif s["both"] >= MIN_BOTH_SHARE or (max(s["lead"], s["follow"]) >= MIN_ONE_SHARE and min(s["lead"], s["follow"]) >= MIN_ONE_OTHER):
            cams.append(c)
            why[c] = f"both teachers in {100 * s['both']:.0f} % of its frames" + (" (attached: low weight)" if c in ATTACHED else "")
        else:
            why[c] = f"both teachers in only {100 * s['both']:.0f} % (lead {100 * s['lead']:.0f}, follow {100 * s['follow']:.0f})"
    return cams, why


def shape_window(tt: dict, frames: int = 900, step_s: float = 5.0, grid_t1: float = 360.48) -> dict:
    """The 30 s window (900 frames) with the best camera coverage of BOTH teachers: score = mean over the window's seconds of the weighted number of
    phones that see both (main phone 1, attached 0.5), a pose-diversity bonus from the teachers' 3D joint speeds is added when track3d.npz exists."""
    L = frames / HZ
    best = None
    rows = []
    for t0 in np.arange(0.0, grid_t1 - L + 1e-6, step_s):
        t1 = t0 + L
        st = window_stats(tt, t0, t1)
        w = sum((0.5 if c in ATTACHED else 1.0) * s["both"] * min(1.0, s["record_share"] / 0.9) for c, s in st.items())
        rows.append((float(t0), float(w)))
        if best is None or w > best[1]:
            best = (float(t0), float(w), st)
    return {"t0": best[0], "t1": best[0] + L, "k0": int(round(best[0] * HZ)), "k1": int(round(best[0] * HZ)) + frames - 1, "score": round(best[1], 3),
            "stats": best[2], "scores": [[round(a, 1), round(b, 3)] for a, b in rows]}


# ------------------------------------------------------------------------------------------------ the shared input tar


def camera_source(cam: str) -> tuple[Path, str]:
    """(file, version) of the camera track a recap job gets: R3's teacher-refined attach of a phone (people/attach/<cam>.npz, same schema, only R / tvec
    differ) when R3's own held-out gate adopted it (people/attach/<cam>.json "adopted": true), else R2's cameras/<cam>.npz. R2's files are never overwritten."""
    j = PEOPLE / "attach" / f"{cam}.json"
    f = PEOPLE / "attach" / f"{cam}.npz"
    if j.is_file() and f.is_file():
        try:
            if json.loads(j.read_text(encoding="utf-8")).get("adopted") is True:
                return f, "r3_attach"
        except (OSError, ValueError):
            pass
    return WORK / "cameras" / f"{cam}.npz", "r2"


def camera_versions() -> dict:
    """{cam: {"version": "r2" | "r3_attach", "sha256": ...}} of the camera tracks build_recap_in packs (recorded in the plan, so a later camera change
    tells which chunks must be fitted again)."""
    out = {}
    for c in CAMS:
        f, v = camera_source(c)
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        out[c] = {"version": v, "sha256": h.hexdigest()[:16]}
    return out


def build_recap_in(out: Path | None = None) -> dict:
    """ONE deterministic tar of everything a recap chunk job needs and the volume does not hold yet (workspace-relative members): R2's six camera
    tracks + recap_cams.json + world_from_sfm.json, R3's teacher tracks, the pose tables and the alignment file. Uploaded once to src/<capture>/ (shared)."""
    out = out or PEOPLE / "in" / "recap_in.tar"
    rel = "dancecap/work/larissa_kadu_recap"
    items = []
    for c in CAMS:
        items += [(camera_source(c)[0], f"{rel}/cameras/{c}.npz"), (PEOPLE / f"teachers_{c}.npz", f"{rel}/people/teachers_{c}.npz"),
                  (PEOPLE / "pose_tables" / f"{c}.npz", f"{rel}/people/pose_tables/{c}.npz")]
    items += [(WORK / "cameras" / "recap_cams.json", f"{rel}/cameras/recap_cams.json"), (WORK / "cameras" / "world_from_sfm.json", f"{rel}/cameras/world_from_sfm.json"),
              (WORKSPACE / "prototype" / "larissa_kadu" / "alignment_recap.json", "prototype/larissa_kadu/alignment_recap.json")]
    missing = [str(p) for p, _ in items if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"recap_in.tar: missing {missing[:3]} (+{max(0, len(missing) - 3)})")
    C.pack_tar(out, items)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return {"path": IN_TAR_REL, "sha256": h.hexdigest(), "bytes": out.stat().st_size, "local": C._ws(out), "archive": True, "n_files": len(items),
            "cameras": camera_versions()}


# ------------------------------------------------------------------------------------------------ estimates


def estimate_chunk(chunk: C.Chunk, cam_frames: int | None = None, shape: bool = False) -> dict:
    """Minutes on ONE RTX 4090 pod of the whole chunk job (front -> fit -> refine -> quality), the model in the constants above."""
    T = chunk.T
    cf = cam_frames if cam_frames is not None else T * len(chunk.cams)
    cf += int(2 * FRONT_MARGIN_FRAMES * len(chunk.cams))
    front = (cf / SAM_FPS + 0.35 * cf / MA2D_FPS + cf / DECODE_FPS) / 60.0 + FRONT_FIXED_MIN
    fit = (FIT_A + FIT_B * T) / 60.0
    refine = 0.0 if shape else (C.REFINE_S_PER_S * T / HZ * GPU_SLOWDOWN + 25.0) / 60.0
    quality = 0.0 if shape else QUALITY_MIN
    tot = front + fit + refine + quality + BOOT_MIN
    return {"front_min": round(front, 1), "fit_min": round(fit, 1), "refine_min": round(refine, 1), "quality_min": round(quality, 1), "boot_min": BOOT_MIN,
            "cam_frames": cf, "total_min_4090": round(tot, 1), "usd_4090": round(tot / 60 * PRICE_4090, 2),
            "total_min": round(tot / GPU_SLOWDOWN, 1),      # the farm's estimate field is in A100 minutes (launch.estimate_run multiplies by GPU_SLOWDOWN)
            "note": "RTX 4090 model of farm/recap_profile.py (PLAN.md 4.1 rates for SAM / ma_2d, drill fit model); total_min = A100-basis minutes for farm launch"}


FRONT_MARGIN_FRAMES = int(C.FRONT_MARGIN_S * HZ)


# ------------------------------------------------------------------------------------------------ job specs

W_ = "dancecap/work/{take}"


def refine_args_recipe() -> str:
    """pose_refine's arguments from the demo's recipe smplx_rf2_noslip (the world package's shared/refine_recipe.json of the demo, a parameter file)."""
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8")) if RECIPE_PATH.exists() else None
    return C.refine_args(recipe)


def chunk_steps(chunk: C.Chunk, est: dict, refine_args: str, *, shape: bool, refined_dir: str) -> list[dict]:
    """The step list of a chunk job: chunk_take -> front -> fit (pinned betas; the shape job's fit is free) -> [pin | refine -> quality] -> evidence."""
    take = chunk.take
    ctx = {"chunk": chunk.id, "take": take}
    w = W_.format(take=take)
    steps = [
        C._step("chunk_take", C.fmt("python -m dancecap.farm chunk-take {chunk}", **ctx), 3),
        C._step("front", C.fmt("python -m dancecap.recap_front run {chunk}", **ctx), max(45, math.ceil(2.0 * est["front_min"] + 15)),
                save=[f"{w}/mamma/prepare.json", f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/ma_2d/", f"{w}/mamma/ma_cap/", f"{w}/sync/offsets.json",
                      f"{w}/cameras/world_from_sfm.json", f"{w}/dancers/roles.json"]),
    ]
    if not shape:   # F1 2026-10-09: the chunk's front runs in parallel with the shape job; only the fit waits for the pin (recap_front.step_await_pin)
        steps.append(C._step("await_pin", C.fmt("python -m dancecap.recap_front await-pin {chunk}", **ctx), 130))
    steps += [
        C._step("fit", C.fmt("python -m dancecap.recap_front fit {chunk}" + ("" if shape else " --pin"), **ctx), max(30, math.ceil(1.6 * est["fit_min"] + 15)),
                env=({} if shape else {"MAMMA_FREEZE_BETAS": "1", "MAMMA_INIT_BETAS": "{WS}/" + PIN_REL}),
                live=[f"{w}/mamma/ma_3d/"], save=[f"{w}/smplx/", f"{w}/mamma/fits/", f"{w}/mamma/select.json", f"{w}/mamma/report.json"]),
    ]
    if shape:
        steps.append(C._step("pin", C.fmt("python -m dancecap.recap_front pin {chunk}", **ctx), 3, save=[f"dancecap/work/{take}/mamma_pin_betas.json"]))
    else:
        steps += [C._step("refine", C.fmt("python -m dancecap.pose_refine {take} --src smplx --dst {d} {a}", d=refined_dir, a=refine_args, **ctx),
                          max(10, math.ceil(2 * est["refine_min"] + 5)), save=[f"{w}/{refined_dir}/"]),
                  C._step("quality", C.fmt("python -m dancecap.pose_quality {take} --smplx-dir {d} --tag rf2_noslip", d=refined_dir, **ctx), 20, optional=True)]
    last = "fit" if shape else "refined"
    steps.append({"name": "evidence", "cmd": f"python -m dancecap.farm chunk-evidence {chunk.id} --stage {last}", "timeout_min": 10.0, "optional": False, "stage": last})
    for s in steps[:-1]:
        s["stage"] = {"chunk_take": "front", "front": "front", "await_pin": "front", "fit": "fit", "pin": "fit", "refine": "refined", "quality": "refined"}[s["name"]]
    return steps


def chunk_outputs(chunk: C.Chunk, *, shape: bool, refined_dir: str) -> list[dict]:
    w = f"dancecap/work/{chunk.take}"
    rows = [("mamma/prepare.json", False), ("mamma/ma_2d/", True), ("sync/offsets.json", True), ("dancers/roles.json", True), ("dancers/report.json", True),
            ("mamma/logs/", True), ("front/front_pod.json", True), ("smplx/", False), ("mamma/fits/", False), ("mamma/select.json", False),
            ("mamma/report.json", True), ("farm/evidence.npz", False)]
    rows += [("mamma_pin_betas.json", False), ("mamma_pin_betas.source.json", True)] if shape else [(f"{refined_dir}/", False), ("quality/", True)]
    return [{"path": f"{w}/{rel}", **({"optional": True} if opt else {})} for rel, opt in rows]


def build_job(chunk: C.Chunk, spec: DanceSpec, world=None, kind: str | None = None, *, hash_inputs: bool = True, code: dict | None = None,
              max_minutes: float | None = None, shape: bool = False, in_tar: dict | None = None) -> dict:
    """The farm job spec of one recap chunk (or of the shape job). `kind` is accepted for chunks.build_job's call and ignored: a recap chunk job always
    runs front + fit + refine + quality (the shape job: front + fit + pin)."""
    est = chunk.estimate if chunk.estimate.get("front_min") is not None and "cam_frames" in chunk.estimate else estimate_chunk(chunk, shape=shape)
    refined = spec.refined_dir
    steps = chunk_steps(chunk, est, refine_args_recipe(), shape=shape, refined_dir=refined)
    inputs, sources = [], {}
    if hash_inputs:
        inputs, sources = C.source_inputs(spec, chunk.cams)
        inputs.append(in_tar or build_recap_in())
    else:
        inputs = [{"path": IN_TAR_REL, "archive": True}]
    total = est["total_min_4090"]
    return {"schema": C.JOB_SCHEMA, "id": chunk.id, "dance": spec.name, "kind": "full", "what": "shape" if shape else "chunk", "stages": ["front", "fit"] + ([] if shape else ["refined"]),
            "take": chunk.take, "chunk": chunk.to_dict(), "cams": chunk.cams, "after": [], "world_id": None, "world_sha256": None, "pose_id": None,
            "code": code if code is not None else C.code_info(), "sources": sources, "inputs": inputs, "steps": steps, "outputs": chunk_outputs(chunk, shape=shape, refined_dir=refined),
            "env": {"DANCECAP_FARM_DANCE": spec.name}, "cwd": "dancecap", "max_minutes": round(max_minutes or max(60.0, 1.6 * total + 20.0), 1),
            "priority": 100.0 if shape else float(total), "min_vram_gb": 24, "needs_gpu": True, "estimate": est, "created": now_iso(),
            "profile": PROFILE, "retries": 1}


def build_final(chunks: list[C.Chunk], spec: DanceSpec, *, code: dict | None = None, hash_inputs: bool = True) -> dict:
    """The stitch job: analysis.build_final_job with NO analyses and without the world check (the recap has no pose world)."""
    cl = [c.to_dict() for c in chunks]
    job = A.build_final_job(spec, cl, None, analyses=(), hash_inputs=False, code=code)
    for s in job["steps"]:
        if s["name"] == "stitch":
            s["cmd"] = s["cmd"].replace(" --require-world", "")
    job["inputs"] = [{"path": f"chunks/{c.id}/out/result.tar", "archive": True} for c in chunks]
    job["profile"] = PROFILE
    return job


# ------------------------------------------------------------------------------------------------ the plan (farm plan)


def cost_table(jobs: list[dict]) -> dict:
    """Per job and in total: 4090 minutes and dollars (the model), plus the critical path with unlimited pods."""
    rows, total_min, total_usd = [], 0.0, 0.0
    for j in jobs:
        e = j.get("estimate") or {}
        m, u = e.get("total_min_4090"), e.get("usd_4090")
        if m is None:
            continue
        rows.append({"id": j["id"], "what": j.get("what"), "cams": len(j.get("cams") or []), "min_4090": m, "usd": u, "front": e.get("front_min"), "fit": e.get("fit_min"),
                     "refine": e.get("refine_min"), "quality": e.get("quality_min")})
        total_min += m
        total_usd += u
    shape = next((r["min_4090"] for r in rows if r["what"] == "shape"), 0.0)
    longest = max([r["min_4090"] for r in rows if r["what"] == "chunk"] or [0.0])
    # the chunk jobs run IN PARALLEL with the shape job (their fit waits for its betas, step await_pin): the shape pass only lengthens the longest
    # chunk job when it ends after that chunk's front
    chunk_front = max([(r["front"] or 0.0) + 3.0 for r in rows if r["what"] == "chunk"] or [0.0])
    wait = max(0.0, shape - chunk_front)
    return {"jobs": rows, "gpu_hours": round(total_min / 60, 2), "usd": round(total_usd, 2), "wall_min_with_one_pod_per_job": round(longest + wait + 4.0, 1),
            "usd_with_30pct_margin": round(1.3 * total_usd, 2), "price_hr": PRICE_4090}


def make_plan(spec: DanceSpec, *, hash_inputs: bool = True, code: dict | None = None, world_id=None, claimed_ids: set | None = None) -> dict:
    """Chunks (equal windows, cameras from the teacher coverage), the shape window, every job spec, the arithmetic and the costs. Writes nothing."""
    tt = load_teachers()
    if not tt:
        raise FileNotFoundError("people/teachers_<cam>.npz missing: run dancecap.recap_people (finalize) first")
    g = spec.grid
    cams_fn = lambda k0, k1: choose_cams(window_stats(tt, g.t(k0), g.t(k1 + 1)))[0]          # noqa: E731
    chunks, rep = C.plan_windows(spec, world_id=world_id, kind="full", claimed_ids=claimed_ids, cams_fn=cams_fn)
    rows = []
    for ch in chunks:
        st = window_stats(tt, g.t(ch.k0), g.t(ch.k1 + 1))
        cams, why = choose_cams(st)
        ch.cams = cams
        cf = sum(st[c]["frames"] for c in cams)
        ch.estimate = estimate_chunk(ch, cam_frames=cf)
        ch.priority = ch.estimate["total_min_4090"]
        rows.append({"id": ch.id, "k": [ch.k0, ch.k1], "core_k": [ch.core_k0, ch.core_k1], "window_s": round(ch.T / HZ, 2), "core_s": round(ch.core_T / HZ, 2),
                     "t": [round(float(g.t(ch.k0)), 2), round(float(g.t(ch.k1 + 1)), 2)], "cams": cams, "cams_stats": st, "cams_why": why, "estimate": ch.estimate})
    sw = shape_window(tt)
    st = sw["stats"]
    scams, swhy = choose_cams(st)
    shape = C.Chunk(id=SHAPE_ID, k0=sw["k0"], k1=sw["k1"], core_k0=sw["k0"], core_k1=sw["k1"], overlap_k=0, kind="full", cams=scams, world_id=None, origin="shape",
                    take=SHAPE_TAKE)
    shape.estimate = estimate_chunk(shape, cam_frames=sum(st[c]["frames"] for c in scams), shape=True)
    jobs = [build_job(shape, spec, hash_inputs=hash_inputs, code=code, shape=True)]
    in_tar = jobs[0]["inputs"][-1] if hash_inputs else None
    jobs += [build_job(ch, spec, hash_inputs=hash_inputs, code=code, in_tar=in_tar) for ch in chunks]
    final = build_final(chunks, spec, code=code, hash_inputs=hash_inputs)
    costs = cost_table(jobs)
    return {"created": now_iso(), "dance": spec.name, "profile": PROFILE, "params": C.chunk_params(spec), "grid": g.to_dict(), "arithmetic": rep["arithmetic"],
            "camera_rule": {"min_record_share": MIN_RECORD_SHARE, "min_both_share": MIN_BOTH_SHARE, "min_one_share": MIN_ONE_SHARE, "min_one_other": MIN_ONE_OTHER,
                            "conf": CONF, "attached": list(ATTACHED)},
            "chunks": rows, "shape": {"id": SHAPE_ID, "window": [sw["t0"], sw["t1"]], "k": [sw["k0"], sw["k1"]], "cams": scams, "why": swhy, "score": sw["score"],
                                      "estimate": shape.estimate},
            "costs": costs, "seams": [{"between": [chunks[i].id, chunks[i + 1].id], "overlap_s": round((chunks[i].k1 - chunks[i + 1].k0 + 1) / HZ, 3),
                                       "cut_t": round(float(g.t(chunks[i + 1].core_k0)), 2)} for i in range(len(chunks) - 1)],
            "_chunks": chunks, "_shape": shape, "_jobs": jobs + [final]}


def print_plan(plan: dict) -> None:
    ar = plan["arithmetic"]
    print(f"recap plan: {len(plan['chunks'])} chunks of {min(ar['window_s'])}-{max(ar['window_s'])} s, overlap {ar['overlap_k'] / HZ:g} s "
          f"(n = {ar['n']}: " + ", ".join(f"{t['n']} x {t['window_s']} s" for t in ar["tried"][-3:]) + ")")
    print(f"{'id':<7}{'window (s)':<18}{'core (s)':<8}{'cams':<22}{'front':>6}{'fit':>6}{'ref+q':>6}{'4090 min':>9}{'USD':>6}")
    for r in plan["chunks"]:
        e = r["estimate"]
        print(f"{r['id']:<7}{r['t'][0]:>7.1f}-{r['t'][1]:<9.1f}{r['core_s']:<8}{','.join(r['cams']):<22}{e['front_min']:>6}{e['fit_min']:>6}"
              f"{e['refine_min'] + e['quality_min']:>6.1f}{e['total_min_4090']:>9}{e['usd_4090']:>6}")
    s = plan["shape"]
    e = s["estimate"]
    print(f"{s['id']:<7}{s['window'][0]:>7.1f}-{s['window'][1]:<9.1f}{30.0:<8}{','.join(s['cams']):<22}{e['front_min']:>6}{e['fit_min']:>6}{'':>6}{e['total_min_4090']:>9}{e['usd_4090']:>6}  (shape: free betas -> pin)")
    c = plan["costs"]
    print(f"total: {c['gpu_hours']} GPU h = {c['usd']} USD on RTX 4090 pods ({c['usd_with_30pct_margin']} with a 30 % margin); wall about {c['wall_min_with_one_pod_per_job']} min with one pod per job")


def cmd_plan(a, spec: DanceSpec) -> int:
    """`farm plan` for a take with [chunks] profile = "recap". --dry-run writes nothing (the plan document with --out); without it the job
    specs go to farm/jobs, chunks + jobs into state.json (status planned) and chunks.json / farm/plan.json are written."""
    st = State(a.dance)
    cur = st.load()
    claimed = {cid for cid, c in cur["chunks"].items() if c.get("status") in ("queued", "running", "done", "pulled", "stitched")}
    plan = make_plan(spec, hash_inputs=not a.dry_run, code=C.code_info(), claimed_ids=claimed)
    public = {k: v for k, v in plan.items() if not k.startswith("_")}
    public["validation"] = validate_plan(plan, spec)
    if a.json:
        print(json.dumps(public, default=str))
    else:
        print_plan(plan)
        v = public["validation"]
        print("validation (offline, the farm's own queue reader and claim rules): " + ("OK" if v["ok"] else "FAILED") + " - "
              + ", ".join(f"{k}{'' if c['ok'] else ' FAILED'}" for k, c in v["checks"].items()))
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


# ------------------------------------------------------------------------------------------------ validation (the farm's own machinery, offline)


def _commands(job: dict) -> list[str]:
    return [s["cmd"] for s in job.get("steps", [])]


def validate_plan(plan: dict, spec: DanceSpec) -> dict:
    """Checks that need no pod: unique ids, `after` references and no cycle, every `python -m <module>` of every step importable and every farm helper a
    real sub-command, the specs load through the farm WORKER's own queue reader and the claim rules behave (the shape job AND the chunk jobs claimable at
    once: a chunk's front runs in parallel with the shape pass and its `await_pin` step waits for the shape betas; the final job only after all planned
    chunks), the chunk take TOML of every job parses and names the right cameras, every output is a workspace-relative path under the job's own work dir,
    the windows tile the timeline. The tiling is checked on the FULL window arithmetic (plan["arithmetic"]), because plan["_chunks"] leaves out the chunks
    already queued / running / done (cmd_plan passes them as claimed_ids, so it can even be empty); every planned chunk must be one of those windows."""
    import importlib.util
    import re
    import tempfile
    import tomllib
    jobs = plan["_jobs"]
    res = {}

    def check(name, ok, **info):
        res[name] = {"ok": bool(ok), **info}
    ids = [j["id"] for j in jobs]
    check("unique_ids", len(ids) == len(set(ids)), ids=ids)
    known = set(ids)
    bad = [(j["id"], a) for j in jobs for a in j.get("after") or [] if a not in known]
    order, seen = [], set()

    def visit(i, stack=()):
        if i in stack:
            raise ValueError(f"cycle {stack + (i,)}")
        if i not in seen:
            for a in next(j for j in jobs if j["id"] == i).get("after") or []:
                visit(a, stack + (i,))
            seen.add(i)
            order.append(i)
    try:
        for i in ids:
            visit(i)
        acyclic = True
    except ValueError:
        acyclic = False
    check("after_references", not bad and acyclic, missing=bad, order=order)
    mods, missing_mods = set(), []
    for j in jobs:
        for cmd in _commands(j):
            for m in re.findall(r"python -m ([\w.]+)", cmd):
                mods.add(m)
    for m in sorted(mods):
        try:
            if importlib.util.find_spec(m) is None:
                missing_mods.append(m)
        except (ImportError, ValueError):
            missing_mods.append(m)
    check("step_modules_importable", not missing_mods, modules=sorted(mods), missing=missing_mods)
    from . import cli as farm_cli
    parser = farm_cli.build_parser()
    sub = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    helpers = {m.split()[0] for j in jobs for c in _commands(j) for m in re.findall(r"dancecap\.farm ([\w-]+)", c)}
    check("farm_helpers_are_subcommands", all(h in sub.choices for h in helpers), helpers=sorted(helpers))
    from . import worker as W_
    with tempfile.TemporaryDirectory() as td:
        vol = Path(td) / "vol"
        (vol / "queue").mkdir(parents=True)
        for j in jobs:
            s = json.loads(json.dumps(j))
            for i in s.get("inputs", []):
                i.pop("local", None)
            (vol / "queue" / f"{j['id']}.json").write_text(json.dumps(s))
        w = W_.Worker(vol, pod_id="validate", ws=Path(td) / "ws", work=Path(td) / "work", gpu_required=False)
        loaded = [s["id"] for s in w.specs()]
        first = {s["id"] for s in w.specs() if w.claimable(s)}
        (vol / "chunks" / SHAPE_ID).mkdir(parents=True)
        (vol / "chunks" / SHAPE_ID / "done.json").write_text("{}")
        second = {s["id"] for s in w.specs() if w.claimable(s)}
        for c in plan["_chunks"]:
            (vol / "chunks" / c.id).mkdir(parents=True, exist_ok=True)
            (vol / "chunks" / c.id / "done.json").write_text("{}")
        third = {s["id"] for s in w.specs() if w.claimable(s)} - {c.id for c in plan["_chunks"]} - {SHAPE_ID}
    chunk_ids = {c.id for c in plan["_chunks"]}
    # F1 2026-10-09: the chunk jobs are claimable AT ONCE (their front runs while the shape job fits; the `await_pin` step waits for its betas)
    waits = all(any(s["name"] == "await_pin" for s in j["steps"]) for j in jobs if j.get("what") == "chunk")
    # With no chunk planned (every chunk id already claimed) the final job has `after = []`: nothing to wait for in THIS plan, so it is claimable at once.
    final_waits = bool(chunk_ids)
    expect_first = chunk_ids | {SHAPE_ID} | (set() if final_waits else {"final"})
    check("worker_queue_and_claim_rules", sorted(loaded) == sorted(ids) and first == expect_first and (not final_waits or "final" not in second) and third == {"final"} and waits,
          loaded=len(loaded), claimable_at_start=sorted(first), after_shape=len(second & chunk_ids), final_after_all=sorted(third), chunks_wait_for_pin_in_a_step=waits,
          chunks_planned=len(chunk_ids))
    takes = {}
    for j in jobs:
        if j.get("what") in ("chunk", "shape"):
            txt = C.chunk_take_toml(j, spec)
            d = tomllib.loads(txt)
            takes[j["id"]] = {"window": d["window"], "exclude": d["exclude"], "cams": d["segment"]["cams"], "fit_k": d["segment"]["fit_k"]}
    ok_takes = all(set(v["cams"]) == set(next(j for j in jobs if j["id"] == k)["cams"]) and not (set(v["cams"]) & set(v["exclude"])) and v["fit_k"][0] < v["fit_k"][1]
                   for k, v in takes.items())
    check("chunk_take_tomls_parse", ok_takes, takes=len(takes))
    bad_out = []
    for j in jobs:
        for o in j.get("outputs", []):
            p = o["path"]
            if p.startswith("/") or ".." in p.split("/") or not p.startswith("dancecap/work/"):
                bad_out.append((j["id"], p))
            elif j.get("what") in ("chunk", "shape") and not p.startswith(f"dancecap/work/{j['take']}/"):
                bad_out.append((j["id"], p))
    check("outputs_are_workspace_relative_under_the_job_take", not bad_out, bad=bad_out[:5])
    g = spec.grid
    ar = plan["arithmetic"]                                  # the FULL tiling (frame indices from the grid start), claimed chunks included
    wins, cores = [tuple(w) for w in ar["windows"]], [tuple(c) for c in ar["cores"]]
    tiles = cores[0][0] == 0 and cores[-1][1] == g.k1 - g.k0 and all(b[0] == a[1] + 1 for a, b in zip(cores, cores[1:]))
    over = [w0[1] - w1[0] + 1 for w0, w1 in zip(wins, wins[1:])]
    known_windows = {(w[0] + g.k0, w[1] + g.k0, c[0] + g.k0, c[1] + g.k0) for w, c in zip(wins, cores)}
    planned_ok = all((c.k0, c.k1, c.core_k0, c.core_k1) in known_windows for c in plan["_chunks"])      # vacuously true when every chunk is claimed
    max_s = float((plan.get("params") or {}).get("chunk_s", 60.0))
    check("windows_tile_the_timeline_with_the_overlap", tiles and planned_ok and all(abs(o - round(ar["overlap_k"])) <= 1 for o in over) and max(b - a + 1 for a, b in wins) <= max_s * g.hz,
          overlap_frames=over, longest_window_s=round(max(b - a + 1 for a, b in wins) / g.hz, 2), chunks_planned=len(plan["_chunks"]), windows=len(wins))
    check("every_chunk_has_at_least_3_cameras", all(len(c.cams) >= 3 for c in plan["_chunks"]), cams={c.id: len(c.cams) for c in plan["_chunks"]})
    return {"ok": all(v["ok"] for v in res.values()), "checks": res}
