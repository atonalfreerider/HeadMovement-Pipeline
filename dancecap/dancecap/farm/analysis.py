"""Pod jobs for everything that is not a chunk fit: prepare, stitch, analyse, stitch+analyse (FARM.md 0.10).

User directive of 2026-10-08, "let's give everything to runpod that pertains to analysis": the 4-CPU workstation runs
orchestration, Unity, image review and git only. Every analysis is a farm job whose inputs are already on the volume (the
chunk results the fit pods left there) and whose outputs are small. This module PLANS those jobs (the pod side is
worker.py: `after`, archive inputs of `chunks/<id>/out/result.tar`, kind `analyse` on CPU pods, `worker.ANALYSIS` commands):

    what (farm kind) job `kind` (worker)  steps                                                            pod   after
    prepare          analyse              mamma prepare of a chunk whose front outputs are on the volume    CPU   -
                                          (re-run after a review-gate veto) + the chunk evidence
    stitch           analyse              stitch-prep, dancecap.stitch, evidence                            CPU   chunk ids
    analyse          analyse              registered analyses (ANALYSES) on ONE take, + evidence            CPU*  -
    stitch_analyse   analyse              stitch + analyses + coverage evidence: ONE CPU pod, no transfers  CPU   chunk ids
                                          (the final job `farm plan` emits for the dance)

(* an analysis registered with gpu=True makes the job needs_gpu: the worker leaves it to GPU pods.) The worker-facing `kind`
is always `analyse` so a CPU pod launched with `--kinds analyse` (pods.launch_cpu) claims every one of them; the farm-level
kind travels in `what`.

`after` = ids of chunk jobs that must have a done.json first (the worker enforces it). Their results reach the job as archive
inputs `{"path": "chunks/<id>/out/result.tar", "archive": true}` (extracted into the pod workspace at workspace paths, i.e.
dancecap/work/<chunk take>/...). Results that exist only on the workstation (the S1 segments a-d fitted by an earlier
session) travel as one deterministic archive per chunk (`motion.tar`, chunks.motion_input) plus the reference span's motion and
the studio beat grid (`reference.tar`, about 20 MB, hash-diffed: uploaded once).

SMALL RESULTS. Every job that touches a chunk or the dance ends with `chunk-evidence`: work/<take>/farm/evidence.npz, a few KB
of per-frame evidence (level, cameras, flag, reprojection gate, seam failures; timeline.Frames.to_npz). The workstation computes
coverage from those files alone (timeline.collect_evidence), so the big motion files never have to be read locally. The dance's
own job also writes farm/coverage_pod.json (the coverage computed on the pod).

Registering more analyses (camera-gate evidence, ...): `register_analysis(Analysis(...))` from any module; they become
selectable with `farm analyse --analyses name,...` and `farm plan --analyses ...`. The base commands come from
worker.ANALYSIS (one place); an Analysis only adds this dance's arguments, its outputs and its scope.
"""
from __future__ import annotations

import json
import math
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

from . import DEFAULT_DANCE, VOLUME_ROOT, farm_dir, jobs_dir, work_dir
from . import world as W
from .state import atomic_write_json, now_iso, read_json
from .timeline import DanceSpec
from ..take import WORKSPACE

WORKER_KIND = "analyse"                                          # what worker.py / pods.launch_cpu(kinds="analyse") claim
WHAT = ("prepare", "stitch", "analyse", "stitch_analyse")        # the farm-level kinds (job["what"])
CPU_KINDS = WHAT                                                 # run on CPU pods (about $0.035 per vCPU-hour)
GPU_KINDS = ("front", "fit", "refine", "full")                   # need a GPU (MAMMA, SAM, pose_refine)
FINAL_ID = "final"
DONE_STATES = ("done", "pulled", "stitched")


# ------------------------------------------------------------------------------------------------ registry


@dataclass
class Analysis:
    name: str
    base: str                       # key of worker.ANALYSIS: the command + default timeout live there
    args: str = ""                  # appended; placeholders {take} {dance} {refined_dir} {beats}
    timeout_min: float | None = None
    optional: bool = True           # an optional analysis that fails does not fail the job
    gpu: bool = False
    scope: str = "dance"            # dance (the stitched take) | chunk (needs the chunk's mamma inputs) | any
    outputs: tuple = ()             # paths under work/<take>/ (a trailing / = directory)
    desc: str = ""


ANALYSES: dict[str, Analysis] = {}
DEFAULT_ANALYSES = ("timing", "physics", "counterbalance", "floorcraft")


def register_analysis(a: Analysis) -> Analysis:
    ANALYSES[a.name] = a
    return a


for _a in (
    Analysis("timing", "timing", "--beats {beats} --no-refine --workers 4", 40, optional=False, outputs=("timing/",),
             desc="footstep touchdowns + joint accents vs the studio beat grid (raw fit, as larissa_kadu_38)"),
    Analysis("physics", "physics", "--beats {beats}", 30, optional=False, outputs=("physics/",),
             desc="L0 centroidal dynamics: COM, F_net, couple GRF, XCoM, beat impulses (raw fit, as larissa_kadu_38)"),
    Analysis("counterbalance", "counterbalance", "--smplx-dir {refined_dir} --no-review", 20, outputs=("counterbalance/",),
             desc="counterbalance intervals on the refined motion (needs timing/)"),
    Analysis("floorcraft", "floorcraft", "--smplx-dir {refined_dir}", 25, outputs=("floorcraft/",),
             desc="floor-craft transitions / pivots / dials on the refined motion (needs counterbalance/)"),
    Analysis("pose_quality", "pose_quality", "--smplx-dir {refined_dir} --tag rf2_noslip", 25, scope="chunk",
             outputs=("quality/",),
             desc="reprojection per camera / 10-s block, zoom checks: needs the chunk's mamma inputs, so it runs per chunk take"),
    Analysis("analysis_review", "analysis_review", "", 30, outputs=("review/",), desc="analysis review sheets (images; small)"),
):
    register_analysis(_a)


def resolve_analyses(names) -> list[Analysis]:
    out = []
    for n in names:
        if n not in ANALYSES:
            raise ValueError(f"unknown analysis {n!r} (have {sorted(ANALYSES)})")
        out.append(ANALYSES[n])
    return out


class _Fmt(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _fmt(s, **ctx):
    return s.format_map(_Fmt(**ctx)) if isinstance(s, str) else s


def analysis_step(a: Analysis, ctx: dict) -> dict:
    """One step of a registered analysis: worker.analysis_steps(base command) + the analysis's arguments."""
    from . import worker
    st = worker.analysis_steps(a.base, ctx["take"], _fmt(a.args, **ctx), minutes=a.timeout_min, name=a.name)[0]
    st["optional"] = a.optional
    return st


def _ctx(spec: DanceSpec, take: str, jid: str, **extra) -> dict:
    return {"id": jid, "take": take, "dance": spec.name, "refined_dir": spec.refined_dir,
            "beats": f"work/{spec.name}/song/beats_studio.json", **extra}


def evidence_step(jid: str, stage: str) -> dict:
    return {"name": "evidence", "cmd": f"python -m dancecap.farm chunk-evidence {jid} --stage {stage}", "timeout_min": 10.0,
            "optional": False, "stage": "evidence"}


def stitch_step(spec: DanceSpec, partial: bool, minutes: float) -> dict:
    """worker.ANALYSIS['stitch'] on the dance with the chunk list written by stitch-prep. Exit 2 = a seam gate failed:
    stitch.json carries the verdict and coverage shows the cells as partial; that is not a job failure."""
    from . import worker
    base = worker.analysis_steps("stitch", spec.name, f"--chunks work/{spec.name}/farm/stitch_list.json --require-world"
                                 + (" --partial" if partial else ""), minutes=minutes, name="stitch")[0]
    base["cmd"] += ("; rc=$?; if [ $rc -eq 2 ]; then echo '[farm] seam gate failed (rc 2): see stitch.json'; exit 0; fi; exit $rc")
    base["stage"] = "stitch"
    return base


def stitch_outputs(spec: DanceSpec) -> list[tuple[str, bool]]:
    """(path under work/<dance>/, optional) of the stitch step."""
    return [("smplx/", False), (f"{spec.refined_dir}/", True), ("stitch.json", False), ("stitch_rejected/", True),
            ("review/pose_seams_full.png", True), ("review/stitch_overview.png", True)]


# ----------------------------------------------------------------------------------------------- the stitch list


def stitch_list(spec: DanceSpec, chunks: list, *, work_root: Path | None = None, partial: bool = True, worlds: dict | None = None,
                ref_world: dict | None = None) -> dict:
    """The `dancecap.stitch --chunks` JSON for the dance: the reference span (larissa_kadu_38, reused bit for bit) and every
    chunk (dict or Chunk) as {name, src, refined, k, margin_s, prepare, ma_2d, world_id, pose_id}. Paths are relative to dancecap/work. A
    chunk's ma_2d is added when its prepare.json exists under `work_root` (always the case on the pod after the archives are extracted).
    `worlds` ({chunk id: {world_id, pose_id}}) and `ref_world` (the reference span's) put the world each chunk was fitted in into the list,
    which is what makes `dancecap.stitch --require-world` a real check instead of a note that every chunk is 'unknown'."""
    from ..take import PACKAGE_ROOT
    work_root = Path(work_root) if work_root else PACKAGE_ROOT / "work"
    cfg = spec.cfg
    st = cfg.get("stitch") or {}
    items = []
    if spec.reference and spec.ref_k:
        reuse = cfg.get("reuse") or {}
        items.append({"name": spec.reference.rsplit("_", 1)[-1], "src": f"{spec.reference}/{reuse.get('motion_raw', 'smplx')}",
                      "refined": f"{spec.reference}/{reuse.get('motion_refined', spec.refined_dir)}",
                      "k": [int(spec.ref_k[0]), int(spec.ref_k[1])], "margin_s": float(st.get("reuse_margin_s", 0.5)),
                      "reference": True, "kind": "reuse", "prepare": f"{spec.reference}/mamma/prepare.json",
                      **{k: v for k, v in (ref_world or {}).items() if k in ("world_id", "pose_id") and v}})
    for c in chunks:
        c = c if isinstance(c, dict) else c.to_dict()
        take = c["take"]
        e = {"name": c["id"], "src": f"{take}/smplx", "refined": f"{take}/{spec.refined_dir}", "k": [int(c["k0"]), int(c["k1"])],
             "margin_s": float(st.get("edge_margin_s", 0.2)), "kind": "segment" if c.get("origin") == "s1" else "gap"}
        wm = (worlds or {}).get(c["id"]) or {}
        for key in ("world_id", "pose_id"):
            if wm.get(key) or c.get(key):
                e[key] = wm.get(key) or c.get(key)
        prep = work_root / take / "mamma" / "prepare.json"
        if prep.exists():
            e["prepare"] = f"{take}/mamma/prepare.json"
            try:
                d = json.loads(prep.read_text(encoding="utf-8"))
                tag = max(d["sets"], key=lambda t: len(d["sets"][t]["cams"]))
                e["ma_2d"] = f"{take}/mamma/ma_2d/{d['sets'][tag]['seq']}"
            except (OSError, ValueError, KeyError):
                pass
        items.append(e)
    items.sort(key=lambda e: e["k"][0])
    doc = {"take": spec.name, "out_raw": st.get("out_raw", "smplx"), "out_refined": spec.refined_dir, "anchor": spec.grid.anchor,
           "hz": spec.grid.hz, "blend_s": float(st.get("blend_s", 2.0)), "edge_margin_s": float(st.get("edge_margin_s", 0.2)),
           "gate_mm": dict(st.get("gate_mm", {})), "chunks": items, "partial": partial}
    for key in ("mono", "low_view", "lead_absent", "follow_absent"):
        if key in st:
            doc[key] = st[key]
    if spec.reference and (work_root / spec.reference / "song" / "beats_studio.json").exists():
        doc["beats"] = f"{spec.name}/song/beats_studio.json"
    return doc


# ----------------------------------------------------------------------------------------------- job builders


def _local_motion(spec: DanceSpec, chunk: dict) -> bool:
    wk = work_dir(chunk["take"])
    return all((wk / d / f"{r}.npz").exists() for d in ("smplx", spec.refined_dir) for r in ("lead", "follow"))


def _ref_inputs(spec: DanceSpec, jid: str) -> list[dict]:
    """The reference span's raw + refined motion and the studio beat grid as ONE archive input (about 20 MB, hashed, uploaded
    once): everything the pod cannot get from the volume."""
    from . import chunks as C
    if not spec.reference:
        return []
    ref = work_dir(spec.reference)
    items = []
    for d in ("smplx", spec.refined_dir):
        items += [(p, C._ws(p)) for p in sorted((ref / d).glob("*")) if p.is_file()]
    if (ref / "mamma" / "prepare.json").exists():
        items.append((ref / "mamma" / "prepare.json", C._ws(ref / "mamma" / "prepare.json")))
    beats = ref / "song" / "beats_studio.json"
    if beats.exists():
        items.append((beats, f"dancecap/work/{spec.name}/song/beats_studio.json"))
    if not items:
        return []

    class _Id:   # archive_input only needs .id (the folder under farm/in/)
        id = jid
    return [C.archive_input(spec, _Id, "reference", items, f"dancecap/work/{spec.reference}/reference.tar")]


def _job(spec: DanceSpec, jid: str, what: str, take: str, *, steps, outputs, inputs, after, analyses, world, code, gpu, max_minutes,
         estimate_min, chunk: dict, extra: dict | None = None) -> dict:
    from . import chunks as C
    return {"schema": "dancecap.farm.job/1", "id": jid, "dance": spec.name, "kind": WORKER_KIND, "what": what,
            "stages": [s["stage"] for s in steps if s.get("stage")], "take": take, "analyses": [a.name for a in analyses],
            "chunk": chunk, "cams": [], "after": list(after), "world_id": world.id if world else None,
            "world_sha256": world.sha256 if world else None, "pose_id": world.pose_id if world else None,
            "code": code if code is not None else C.code_info(), "inputs": inputs, "steps": steps, "outputs": outputs,
            "env": {"DANCECAP_FARM_DANCE": spec.name}, "cwd": "dancecap", "max_minutes": round(max_minutes, 1), "priority": -1.0,
            "needs_gpu": bool(gpu), "estimate": {"total_min": round(estimate_min, 1)}, "created": now_iso(), **(extra or {})}


def pose_of(dance: str, world_id: str | None, world: "W.World | None" = None) -> str | None:
    """pose id of a world id: the current world, else the tracked manifest (None when unknown)."""
    if not world_id:
        return None
    if world is not None and world.id == world_id:
        return world.pose_id
    try:
        return W.manifest_of(dance, world_id)["pose_id"]
    except Exception:  # noqa: BLE001
        return None


def chunk_worlds(spec: DanceSpec, cl: list[dict], world: "W.World | None", local_ids, *, verify: bool = True) -> dict:
    """{chunk id: {world_id, pose_id, how}}: the world each chunk was really fitted in, for the stitch guard. A chunk the farm produced
    is held to the world its job named (the pod's out/manifest.json overrides this at stitch time); a chunk fitted OUTSIDE the farm (the S1
    segments) is checked by CONTENT against the world (`world compare`: world frame, betas, clocks, camera types, avatar) and carries the
    world ids only when that check passes - a [world] pin in its TOML is a declaration, not evidence."""
    out = {}
    for c in cl:
        cid = c["id"]
        rec: dict = {"world_id": None, "pose_id": None, "how": None}
        if cid in local_ids:
            if world is not None and verify:
                try:
                    r = W.compare_take(c["take"], spec.name, world.id, deep=False)
                    bad = [x["name"] for x in r["checks"] if x["ok"] is False and x["scope"] == W.POSE_SCOPE]
                    if r["ok"]:
                        rec.update(world_id=world.id, pose_id=world.pose_id, how="compare_take: content equals the world")
                    else:
                        rec["how"] = f"compare_take FAILED: {', '.join(bad)}"
                except Exception as e:  # noqa: BLE001
                    rec["how"] = f"compare_take error: {type(e).__name__}: {str(e)[:120]}"
            elif world is not None:
                rec.update(world_id=world.id, pose_id=world.pose_id, how="not checked (dry run)")
        else:
            wid = c.get("world_id") or (world.id if world else None)
            if wid:
                rec.update(world_id=wid, pose_id=pose_of(spec.name, wid, world), how="job spec")
        out[cid] = rec
    return out


def build_final_job(spec: DanceSpec, chunks: list, world: "W.World | None" = None, *, analyses=DEFAULT_ANALYSES,
                    jid: str = FINAL_ID, partial: bool = True, hash_inputs: bool = True, code: dict | None = None,
                    max_minutes: float | None = None, with_stitch: bool = True) -> dict:
    """The job `farm plan` emits for the dance: extract the chunk results, stitch, run the analyses on the stitched motion, write
    the coverage evidence. ONE CPU pod; what = 'stitch_analyse' (with_stitch=False: 'analyse' on the existing result).
    `chunks`: chunk dicts / Chunks with a core. A chunk whose fit + refined motion already sits on the workstation travels as an
    input archive; every other chunk is in `after` and its result.tar on the volume is an archive input."""
    from . import chunks as C
    cl = [c if isinstance(c, dict) else c.to_dict() for c in chunks]
    cl = sorted((c for c in cl if c.get("status") not in ("failed", "cancelled")), key=lambda c: c["k0"])
    after, inputs, local = [], [], []
    for c in cl:
        if _local_motion(spec, c):
            local.append(c["id"])
            if hash_inputs:
                inputs.append(C.motion_input(spec, C.Chunk.from_dict(c)))
        else:
            after.append(c["id"])
            inputs.append({"path": f"chunks/{c['id']}/out/result.tar", "archive": True})
    if hash_inputs:
        inputs += _ref_inputs(spec, jid)
    sel = resolve_analyses(analyses)
    ctx = _ctx(spec, spec.name, jid)
    steps = []
    if with_stitch:
        steps += [{"name": "stitch_prep", "cmd": f"python -m dancecap.farm stitch-prep {jid}", "timeout_min": 15.0, "optional": False,
                   "stage": "stitch"},
                  stitch_step(spec, partial, max(20, math.ceil(0.4 * spec.grid.seconds / 10) * 5))]
    steps += [dict(analysis_step(a, ctx), stage="analyse") for a in sel if a.scope != "chunk"]
    steps.append(evidence_step(jid, "final"))
    outputs, seen = [], set()

    def out(rel, opt):
        p = f"dancecap/work/{spec.name}/{rel}"
        if p not in seen:
            seen.add(p)
            outputs.append({"path": p, **({"optional": True} if opt else {})})
    if with_stitch:
        for rel, opt in stitch_outputs(spec):
            out(rel, opt)
    for a in sel:
        if a.scope != "chunk":
            for rel in a.outputs:
                out(rel, a.optional)
    out("farm/evidence.npz", False)
    out("farm/coverage_pod.json", True)
    out("farm/stitch_list.json", True)
    est = 25 + 0.2 * spec.grid.seconds / 10 * len(sel)
    chunk = {"id": jid, "k0": spec.grid.k0, "k1": spec.grid.k1, "core_k0": spec.grid.k0, "core_k1": spec.grid.k1, "origin": "final",
             "kind": "stitch_analyse", "take": spec.name, "chunks": [c["id"] for c in cl]}
    cw = chunk_worlds(spec, cl, world, set(local), verify=hash_inputs) if with_stitch else {}
    return _job(spec, jid, "stitch_analyse" if with_stitch else "analyse", spec.name, steps=steps, outputs=outputs, inputs=inputs,
                after=after, analyses=sel, world=world, code=code, gpu=any(a.gpu for a in sel),
                max_minutes=max_minutes or max(60.0, 2.0 * est + 30), estimate_min=est, chunk=chunk,
                extra={"local_chunks": local, "chunks": [c["id"] for c in cl], "stitch_chunks": cl, "chunk_worlds": cw})


def build_analysis_job(spec: DanceSpec, take: str, analyses, world: "W.World | None" = None, *, jid: str | None = None,
                       hash_inputs: bool = True, code: dict | None = None, max_minutes: float | None = None,
                       after: list | None = None) -> dict:
    """`farm analyse`: registered analyses on ONE existing take (the stitched dance or a chunk take) as a pod job. The take's
    motion arrives from the volume (after=[the stitch / chunk job ids]: their result.tar archives) or, for a take that exists
    only here, as an archive input."""
    from . import chunks as C
    sel = resolve_analyses(analyses)
    jid = jid or f"an-{take}"
    chunk_like = {"id": jid, "take": take, "k0": spec.grid.k0, "k1": spec.grid.k1, "origin": "analysis"}
    inputs = [{"path": f"chunks/{a}/out/result.tar", "archive": True} for a in (after or [])]
    if hash_inputs and not after and _local_motion(spec, chunk_like):
        mi = C.motion_input(spec, C.Chunk(id=jid, k0=spec.grid.k0, k1=spec.grid.k1, core_k0=spec.grid.k0, core_k1=spec.grid.k1,
                                          take=take))
        inputs += [mi] if mi else []
        if take == spec.name:
            inputs += _ref_inputs(spec, jid)
    ctx = _ctx(spec, take, jid)
    steps = [dict(analysis_step(a, ctx), stage="analyse") for a in sel] + [evidence_step(jid, "analysed")]
    outputs = [{"path": f"dancecap/work/{take}/{rel}", **({"optional": True} if a.optional else {})} for a in sel for rel in a.outputs]
    outputs.append({"path": f"dancecap/work/{take}/farm/evidence.npz"})
    est = sum((a.timeout_min or 30) for a in sel) / 2
    return _job(spec, jid, "analyse", take, steps=steps, outputs=outputs, inputs=inputs, after=after or [], analyses=sel, world=world,
                code=code, gpu=any(a.gpu for a in sel), max_minutes=max_minutes or max(30.0, 2 * est + 15), estimate_min=est,
                chunk=chunk_like)


def build_prepare_job(chunk, spec: DanceSpec, world: "W.World | None" = None, *, code: dict | None = None) -> dict:
    """what 'prepare': re-run `mamma prepare` for a chunk whose front outputs are on the volume (after a review-gate veto): the
    chunk's front result.tar is an archive input."""
    c = chunk if isinstance(chunk, dict) else chunk.to_dict()
    jid = f"prep-{c['id']}"
    t = c["take"]
    steps = [{"name": "chunk_prepare", "cmd": f"python -m dancecap.farm chunk-prepare {jid}", "timeout_min": 30.0, "optional": False,
              "stage": "prepare"}, evidence_step(jid, "front")]
    outputs = [{"path": f"dancecap/work/{t}/mamma/prepare.json"}, {"path": f"dancecap/work/{t}/mamma/ma_cap/"},
               {"path": f"dancecap/work/{t}/mamma/ma_2d/"}, {"path": f"dancecap/work/{t}/farm/evidence.npz"}]
    return _job(spec, jid, "prepare", t, steps=steps, outputs=outputs,
                inputs=[{"path": f"chunks/{c['id']}/out/result.tar", "archive": True}], after=[], analyses=[], world=world, code=code,
                gpu=False, max_minutes=60.0, estimate_min=10, chunk=c, extra={"cams": c.get("cams", [])})


def write_job(job: dict) -> Path:
    p = jobs_dir(job["dance"]) / f"{job['id']}.json"
    atomic_write_json(p, job)
    return p


# -------------------------------------------------------------------------------- readiness (optional queue hold)


def deps(job: dict) -> list[str]:
    return list(job.get("after") or [])


def needs_met(job: dict, chunk_status: dict[str, str]) -> bool:
    """Every chunk the job runs after has finished on a pod (status done / pulled / stitched)."""
    return all(chunk_status.get(c) in DONE_STATES for c in deps(job))


def ready_jobs(jobs: list[dict], chunk_status: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """(ready, held) split of job specs by their `after`."""
    ready = [j for j in jobs if needs_met(j, chunk_status)]
    return ready, [j for j in jobs if j not in ready]


# ------------------------------------------------------------------------------------------------ pod-side helpers


def _vol(volume=None) -> Path:
    return Path(volume or os.environ.get("HM_ROOT") or VOLUME_ROOT)


def check_upstream(after: list[str], volume: Path, job_world_id: str | None, dance: str, job_pose_id: str | None = None) -> dict:
    """Before stitching: every chunk this job runs after must have a clean result on the volume (out/manifest.json status ok,
    nothing missing) and share the job's pose id (world.check_same_world)."""
    heads = []
    for cid in after:
        man = read_json(Path(volume) / "chunks" / cid / "out" / "manifest.json", None)
        if not man:
            raise FileNotFoundError(f"chunk {cid}: out/manifest.json missing on the volume: not finished")
        if man.get("status") not in ("ok", None) or man.get("missing"):
            raise RuntimeError(f"chunk {cid}: result status {man.get('status')!r}, missing {man.get('missing')}")
        if man.get("world_id"):
            pid = man.get("pose_id")
            if not pid:   # a result written before the worker recorded the pose id: the volume holds the world's manifest next to it
                for cand in (Path(volume) / "worlds" / dance / man["world_id"] / "manifest.json", Path(volume) / "worlds" / man["world_id"] / "manifest.json"):
                    pid = (read_json(cand, None) or {}).get("pose_id")
                    if pid:
                        break
            heads.append({"id": cid, "world_id": man["world_id"], **({"pose_id": pid} if pid else {})})
    if not job_world_id:
        return {"ok": True, "chunks": len(heads), "world_ids": sorted({h["world_id"] for h in heads}), "heads": heads}
    rep = W.check_same_world([{"world_id": job_world_id, "id": "job", **({"pose_id": job_pose_id} if job_pose_id else {})}] + heads, dance)
    rep["heads"] = heads
    return rep


def stitch_prep(job_id: str, dance: str = DEFAULT_DANCE, job_path=None, volume=None, ws: Path | None = None) -> dict:
    """Pod step of `stitch` / `stitch_analyse` (the chunk results are already extracted by the worker's archive inputs): check
    the upstream manifests, copy the world frame next to the dance (analyses read cameras/world_from_sfm.json) and write
    farm/stitch_list.json (dancecap.stitch --chunks) from what is on disk now."""
    from . import chunks as C
    job = C.load_job(job_id, dance, job_path)
    spec = DanceSpec.load(dance)
    ws = Path(ws) if ws else WORKSPACE
    rep = check_upstream(deps(job), _vol(volume), job.get("world_id"), dance, job.get("pose_id"))
    # the world each chunk was fitted in: farm chunks from the pod's own record of it, local ones from the planner's content check
    worlds = {k: dict(v) for k, v in (job.get("chunk_worlds") or {}).items()}
    for h in rep.get("heads", []):
        worlds[h["id"]] = {"world_id": h["world_id"], "pose_id": h.get("pose_id"), "how": "out/manifest.json of the chunk job"}
    unverified = [c["id"] for c in job.get("stitch_chunks") or [] if c["id"] in (job.get("local_chunks") or [])
                  and not (worlds.get(c["id"]) or {}).get("world_id")]
    if unverified:
        raise W.WorldMismatch(f"chunk(s) {unverified} were fitted outside the farm and `world compare` does not confirm they are in world "
                              f"{job.get('world_id')}: "
                              + "; ".join(f"{k}: {(worlds.get(k) or {}).get('how')}" for k in unverified))
    wdir = os.environ.get("HM_WORLD_DIR")
    wf = Path(wdir) / "ref" / "cameras" / "world_from_sfm.json" if wdir else None
    if wf and wf.exists():
        dst = work_dir(spec.name) / "cameras" / "world_from_sfm.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(wf.read_bytes())
    doc = stitch_list(spec, job.get("stitch_chunks") or [], work_root=ws / "dancecap" / "work", partial=True, worlds=worlds,
                      ref_world={"world_id": job.get("world_id"), "pose_id": job.get("pose_id")})
    atomic_write_json(farm_dir(spec.name) / "stitch_list.json", doc)
    missing = [c["name"] for c in doc["chunks"] if not (ws / "dancecap" / "work" / c["src"] / "lead.npz").exists()]
    if missing:
        raise FileNotFoundError(f"chunk motion not extracted into the workspace: {missing}")
    return {"world_check": rep, "stitch_list": len(doc["chunks"]), "world_frame": bool(wf and wf.exists())}


def chunk_prepare(chunk_id: str, dance: str = DEFAULT_DANCE, job_path=None) -> dict:
    """Pod step of `prepare`: mamma prepare --skip-2d --coverage union on the lattice of the dance for one chunk (the cameras that
    already have tracks and ma_2d landmarks), as front_full does at the end of its task graph. `chunk_id` = the prepare job id
    (prep-<chunk>) or the chunk id."""
    from . import chunks as C
    from .. import front_full as ff
    from .. import mamma_dyn as md
    job = C.load_job(chunk_id, dance, job_path)
    take = job["take"]
    wk = md.take_work(take)
    c = job["chunk"]
    ready = sorted(p.stem for p in (wk / "cameras").glob("*.npz") if (wk / "mamma" / "ma_2d_native" / f"{p.stem}.npz").exists())
    if len(ready) < 2:
        raise RuntimeError(f"{take}: only {ready} have camera tracks and ma_2d landmarks")
    prep = md.prepare(take, {f"c{len(ready)}": ready}, skip_2d=True, coverage="union", grid_anchor=ff.GRID_ANCHOR,
                      grid_k=(int(c["k0"]), int(c["k1"])))
    return {"take": take, "cams": ready, "T": prep["grid"]["T"]}


def chunk_evidence(job_id: str, stage: str = "fit", dance: str = DEFAULT_DANCE, job_path=None) -> dict:
    """Last step of every job: work/<take>/farm/evidence.npz (+ for the dance's own job farm/coverage_pod.json). On the dance
    (the stitched result + the reference span + the chunk evidence files already extracted next to it) or on a chunk take."""
    from . import chunks as C
    from . import timeline as T
    job = C.load_job(job_id, dance, job_path)
    spec = DanceSpec.load(dance)
    take = job["take"]
    stitched = take == spec.name
    fr = T.build_evidence(spec, work_dir(take), job["chunk"].get("id", job_id), stitched=stitched)
    p = work_dir(take) / T.EVIDENCE
    fr.to_npz(p)
    res = {"evidence": str(p), "frames": int((fr.level > 0).sum()), "stitched": stitched, "stage": stage}
    if stitched:
        cells = T.fold_cells(fr)
        cov = T.Coverage(dance=spec.name, grid=spec.grid, cells=cells, frames=fr, sources=list(fr.sources), updated=now_iso())
        atomic_write_json(farm_dir(spec.name) / "coverage_pod.json", cov.to_json(), indent=None)
        res["coverage"] = {"counts": cov.counts(), "percent": cov.percent()}
    return res
