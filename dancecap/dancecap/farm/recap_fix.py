"""Pose polish of the class recap (agent T1, 2026-10-09): the redo of chunk g0000 and the boundary refits of the four failing seams.

    python -m dancecap.farm --dance larissa_kadu_recap fix-plan [--dry-run] [--only h0000,x1384,...]
    python -m dancecap.farm --dance larissa_kadu_recap fix-final [--dry-run]        # the second stitch job (final2) after the fixes are pulled

WHAT THE DIAGNOSIS FOUND (work/larissa_kadu_recap/NOTES_T1.md has the numbers)
  g0000 (0-55 s): the follow was displaced 0.4-0.6 m in z (the axis between phones 00 and 05, which look at each other) in 9-30 s. Cameras 00, 02 and 05
  agree with each other and with the teacher's 2D joints there; phone 01's R2 track does not (its centre jumps by 2.7 m along the optical axis in
  8-30 s: a pose / zoom ambiguity). The first fit used all four phones (01 pulled), F1's refit left out phone 02, the only phone that sees z
  from the side, and used the wrong video offsets (the original -185.56 ms for 00 instead of the override +113.1 ms: the front checkpoint was saved
  before the pod override). So: h0000 = g0000 again with the override offsets, phone 02 IN, phone 01 OUT for 9-30 s.
  The four failing seams: the neighbouring chunks differ in their camera sets and SAM masks; a 12-s refit centred on the seam, with the cameras of
  both neighbours' good stretches, the neighbours' offsets and the new `qa-mask` step (a camera's landmarks of one dancer are zeroed where they are
  not on the teacher: recap_front.qa_mask_camera) is one continuous fit across the seam; the stitcher then blends it with each neighbour at the
  position where they agree best (a longer common window than the 4-s overlap).

Every job here is a full chunk job (front -> [mask] -> qa_mask -> await_pin -> fit --pin -> refine -> quality -> evidence) built from the spec of an
existing chunk (same inputs on the volume, same pinned betas), with its own id and take, so nothing of the earlier chunks is touched or moved.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from . import farm_dir, jobs_dir
from .state import State, atomic_write_json, now_iso

DANCE = "larissa_kadu_recap"
TEMPLATE_ID = "g1588"                      # the spec the new ones are copied from (inputs on the volume, steps of a working chunk job)
HZ = 30.0
# id -> (k0, k1, neighbours whose measured video offsets the job takes, reason)
WINDOWS = {
    "h0000": (0, 1647, ("g0000_override",), "g0000 again: F1's override offsets, phone 02 in, phone 01 out for 9-30 s"),
    "y1384": (1384, 1744, ("g0000_override", "g1588"), "seam g0000|g1588 (51.1-53.1 s): 46.13-58.10 s (x1384 without phone 06: its front found no clean frame with both dancers separated)"),
    "x4440": (4440, 4800, ("g3116", "g4644"), "seam g3116|g4644 (153.0-155.0 s): 148.00-159.97 s"),
    "x5966": (5966, 6326, ("g4644", "g6171"), "seam g4644|g6171 (203.9-205.9 s): 198.87-210.83 s"),
    "x9044": (9044, 9404, ("g7699", "g9227"), "seam g7699|g9227 (306.5-308.4 s): 301.47-313.43 s"),
    "z9044": (9044, 9404, ("g7699", "g9227"), "x9044 again with qa thr 0.12: phone 01's follow landmarks are on somebody else in 52 % of the window (median error 0.107)"),
}
QA_THR_BY_ID = {"z9044": 0.12}
CAMS_EXCLUDE = {"y1384": ["06"]}                  # SAM's auto labelling needs one clean frame with both dancers apart in each phone: phone 06 has none in 46-58 s
MANUAL_MASKS = {"h0000": [("01", 9.0, 30.0)]}     # camera 01's R2 track is wrong in 9-30 s (NOTES_T1.md section 2)
REPLACED = {"g0000": "h0000"}                      # chunk ids the stitch drops when the new one is there
# Edge margins (s) of the chunks in the second stitch: the first / last frames a blend window never touches. The stitcher takes a number (both ends) or a pair [first, last].
# Tried: 1.5 s at the ends that face a fixed seam (so that the blends run between interior frames); on this take it was not better than 0.2 s (neighbours) / 0.5 s
# (the short fix windows) at any of the four seams (NOTES_T1.md section 5), so the simple margins are used.
MARGINS = {"h0000": 0.2, "y1384": 0.5, "g1588": 0.2, "g3116": 0.2, "x4440": 0.5, "g4644": 0.2, "x5966": 0.5, "g6171": 0.2, "g7699": 0.2, "x9044": 0.5, "z9044": 0.5, "g9227": 0.2}
EDGE_MARGIN_S = MARGINS
SELECT_WEIGHTS = "arm=0.3,fingers=0.3"      # the blend window is CHOSEN with the elbow / wrist and finger gates counted 0.3 times, so the body and pelvis gates decide first (the brief: body gates
                                            # first, the raw MAMMA hands may stay exceeded); the gates themselves are unchanged and every exceeded one is reported
G0000_OVERRIDE = {"00": 113.1, "01": 0.0, "02": 4.5, "05": -25.7}   # F1's documented override of g0000's offsets (NOTES_F1.md section 4)


def take_of(cid: str) -> str:
    return f"{DANCE}_{cid}"


def work_root() -> Path:
    from ..take import PACKAGE_ROOT
    return PACKAGE_ROOT / "work"


# ------------------------------------------------------------------------------------------------ pure parts


def mean_offsets(docs: list[dict]) -> dict:
    """Per camera the mean (ms, 0.1 resolution) of the video offsets the neighbouring chunks used; a camera only one of them has keeps that value."""
    cams = sorted({c for d in docs for c in d})
    out = {}
    for c in cams:
        v = [float(d[c]) for d in docs if c in d and d[c] is not None]
        if v:
            out[c] = round(sum(v) / len(v), 1)
    out["01"] = 0.0
    return out


def window_cams(stats: dict, min_record: float = 0.35, min_teacher: float = 0.15) -> list[str]:
    """The phones of a boundary window: every phone that records in most of it and sees at least one teacher in a useful share of its frames
    (recap_profile.choose_cams asks for BOTH teachers in 30 % and drops phones the neighbouring 55-s chunks did use: for a window that must agree
    with those neighbours the camera set should be theirs)."""
    return [c for c, s in sorted(stats.items()) if s["record_share"] >= min_record and max(s.get("lead", 0.0), s.get("follow", 0.0)) >= min_teacher]


def chunk_offsets(cid: str) -> dict:
    """The offsets (ms) a chunk's fit used: its mamma/prepare.json sync.offsets_ms; g0000_override = F1's documented values."""
    if cid == "g0000_override":
        return dict(G0000_OVERRIDE)
    p = work_root() / take_of(cid) / "mamma" / "prepare.json"
    return {k: float(v) for k, v in ((json.loads(p.read_text(encoding="utf-8")).get("sync") or {}).get("offsets_ms") or {}).items()}


def rename_text(obj, old_id: str, old_take: str, new_id: str, new_take: str):
    """The template's strings with its chunk id / take replaced (the take first: it contains the id)."""
    s = json.dumps(obj).replace(old_take, new_take).replace(old_id, new_id)
    return json.loads(s)


def mask_step(cid: str, masks: list[tuple[str, float, float]]) -> dict:
    arg = ",".join(f"{c}:{a:g}:{b:g}" for c, a, b in masks)
    return {"name": "mask", "cmd": f"python -m dancecap.recap_front mask {cid} --mask {arg}", "timeout_min": 10, "optional": False, "stage": "front"}


def qa_step(cid: str, thr: float | None = None) -> dict:
    return {"name": "qa_mask", "cmd": f"python -m dancecap.recap_front qa-mask {cid}" + (f" --thr {thr:g}" if thr else ""), "timeout_min": 15, "optional": False, "stage": "front"}


def fix_steps(tpl_steps: list[dict], tpl_id: str, cid: str, T: int, est: dict, masks: list, qa_thr: float | None = None) -> list[dict]:
    """The template's steps for the new id, the (manual) mask and the qa_mask step after `front`, timeouts from the window length."""
    steps = rename_text(tpl_steps, tpl_id, take_of(tpl_id), cid, take_of(cid))
    out = []
    for s in steps:
        if s["name"] == "front":
            s["timeout_min"] = max(45, math.ceil(2.0 * float(est["front_min"]) + 15))
        if s["name"] == "fit":
            s["timeout_min"] = max(30, math.ceil(1.1 * (1.8 * T / 60.0) + 15))
        out.append(s)
        if s["name"] == "front":
            if masks:
                out.append(mask_step(cid, masks))
            out.append(qa_step(cid, qa_thr))
    return out


def fix_outputs(tpl_outputs: list[dict], tpl_id: str, cid: str) -> list[dict]:
    out = rename_text(tpl_outputs, tpl_id, take_of(tpl_id), cid, take_of(cid))
    w = f"dancecap/work/{take_of(cid)}/mamma/"
    return out + [{"path": w + "mask.json", "optional": True}, {"path": w + "qa_mask.json", "optional": True}]


def chunk_dict(cid: str, k0: int, k1: int, cams: list[str], est: dict, sync_fixed: dict, masks: list, why: str) -> dict:
    T = k1 - k0 + 1
    return {"id": cid, "k0": k0, "k1": k1, "core_k0": k0, "core_k1": k1, "overlap_k": 0, "kind": "full", "cams": list(cams), "world_id": None,
            "origin": "fix", "take": take_of(cid), "stages_done": [], "status": "planned", "priority": float(est.get("total_min_4090", 30.0)),
            "estimate": est, "T": T, "core_s": round(T / HZ, 2), "sync_fixed": sync_fixed, "manual_masks": [list(m) for m in masks], "why": why}


def build_fix_job(tpl: dict, cid: str, k0: int, k1: int, cams: list[str], est: dict, sync_fixed: dict, masks: list, why: str, code: dict | None = None,
                  qa_thr: float | None = None) -> dict:
    """The farm job spec of one fix chunk from the template chunk job's spec."""
    T = k1 - k0 + 1
    job = copy.deepcopy(tpl)
    job["id"] = cid
    job["take"] = take_of(cid)
    job["what"] = "chunk"
    job["chunk"] = chunk_dict(cid, k0, k1, cams, est, sync_fixed, masks, why)
    job["cams"] = list(cams)
    job["steps"] = fix_steps(tpl["steps"], tpl["id"], cid, T, est, masks, qa_thr)
    job["outputs"] = fix_outputs(tpl["outputs"], tpl["id"], cid)
    job["estimate"] = est
    job["priority"] = float(est.get("total_min_4090", 30.0)) + (100.0 if cid == "h0000" else 0.0)
    job["max_minutes"] = round(1.6 * float(est["total_min_4090"]) + 30.0, 1)
    job["created"] = now_iso()
    job["fix"] = {"agent": "T1", "why": why, "manual_masks": [list(m) for m in masks], "sync_fixed": sync_fixed}
    if code is not None:
        job["code"] = code
    return job


def final2_job(tpl_final: dict, chunk_specs: list[dict], out_take: str = "larissa_kadu_recap_t1", jid: str = "final2", margins: dict | None = None,
                select_weights: str = SELECT_WEIGHTS) -> dict:
    """The second stitch job: the template `final` job with the chunk list of the fixed set, writing into work/<out_take>/ (the first stitch in
    work/<dance>/ stays untouched until the result is verified and promoted by hand)."""
    job = copy.deepcopy(tpl_final)
    ids = [c["id"] for c in chunk_specs]
    job["id"] = jid
    job["chunk"] = {**job["chunk"], "id": jid, "chunks": ids}
    job["after"] = ids
    job["chunks"] = ids
    job["inputs"] = [{"path": f"chunks/{c}/out/result.tar", "archive": True} for c in ids]
    job["local_chunks"] = []
    sc = []
    for c in sorted(chunk_specs, key=lambda c: c["k0"]):
        d = dict(c)
        d["margin_s"] = (margins or MARGINS).get(c["id"], 0.2)
        sc.append(d)
    job["stitch_chunks"] = sc
    job["chunk_worlds"] = {c: {"world_id": None, "pose_id": None, "how": None} for c in ids}
    for s in job["steps"]:
        if s["name"] == "stitch_prep":
            s["cmd"] = f"python -m dancecap.farm stitch-prep {jid}"
        elif s["name"] == "stitch":
            s["cmd"] = s["cmd"].replace("--partial --force", f"--out {out_take} --partial --force" + (f" --select-weights {select_weights}" if select_weights else ""))
        elif s["name"] == "evidence":
            s["cmd"] = s["cmd"].replace(" final ", f" {jid} ")
    w = f"dancecap/work/{out_take}/"
    job["outputs"] = [{"path": w + "smplx/"}, {"path": w + "smplx_rf2_noslip/", "optional": True}, {"path": w + "stitch.json"},
                      {"path": w + "stitch_rejected/", "optional": True}, {"path": w + "review/pose_seams_full.png", "optional": True},
                      {"path": w + "review/stitch_overview.png", "optional": True}, {"path": f"dancecap/work/{DANCE}/farm/stitch_list.json", "optional": True}]
    job["created"] = now_iso()
    return job


ANALYSIS_IDS = {"an-timing-w": "an2-timing", "an-physics-m": "an2-physics", "an-floor": "an2-floor"}
PRELUDE = "mkdir -p work/{dance} && cp -r work/{out}/. work/{dance}/"


def analysis2_jobs(jobs: list[dict], after: str = "final2", out_take: str = "larissa_kadu_recap_t1") -> list[dict]:
    """The beat-free analyses (farm/recap_analysis.build_jobs) again, behind the second stitch: ids an2-*, the archive input is final2's result, a first
    step copies the stitched motion from work/<out_take>/ to work/<dance>/ where the analyses read it, and the step timing reads the REFINED fit
    this time (--smplx-dir smplx_rf2_noslip; the first run used the raw one)."""
    out = []
    for j in jobs:
        j = copy.deepcopy(j)
        j["id"] = ANALYSIS_IDS[j["id"]]
        j["chunk"] = {**j["chunk"], "id": j["id"]}
        j["after"] = [after]
        j["inputs"] = [{"path": f"chunks/{after}/out/result.tar", "archive": True}]
        for s in j["steps"]:
            if s["name"] == "timing_free" and "--smplx-dir" not in s["cmd"]:
                s["cmd"] += " --smplx-dir smplx_rf2_noslip"
        pre = {"name": "stage_motion", "cmd": PRELUDE.format(dance=DANCE, out=out_take), "timeout_min": 10, "optional": False, "stage": "analyse"}
        j["steps"] = [pre] + j["steps"]
        out.append(j)
    return out


# ------------------------------------------------------------------------------------------------ planning (needs the local data)


def plan_window(cid: str, tpl: dict) -> dict:
    """Cameras (the teacher coverage rule of recap_profile), offsets and estimate of one fix window."""
    from . import chunks as C
    from . import recap_profile as RP
    k0, k1, neigh, why = WINDOWS[cid]
    tt = RP.load_teachers()
    st = RP.window_stats(tt, k0 / HZ, (k1 + 1) / HZ)
    cams, reasons = RP.choose_cams(st)
    cams = list(tpl_cams_of("g0000")) if cid == "h0000" else [c for c in window_cams(st) if c not in CAMS_EXCLUDE.get(cid, [])]
    ch = C.Chunk(id=cid, k0=k0, k1=k1, core_k0=k0, core_k1=k1, overlap_k=0, kind="full", cams=cams, take=take_of(cid))
    est = RP.estimate_chunk(ch, cam_frames=sum(st[c]["frames"] for c in cams))
    sync_fixed = {"offsets_ms": {c: v for c, v in mean_offsets([chunk_offsets(n) for n in neigh]).items() if c in cams},
                  "from": list(neigh)}
    return {"id": cid, "k0": k0, "k1": k1, "cams": cams, "why": why, "reasons": reasons, "est": est, "sync_fixed": sync_fixed,
            "masks": MANUAL_MASKS.get(cid, []), "stats": st}


def tpl_cams_of(cid: str) -> list[str]:
    return list(json.loads((jobs_dir(DANCE) / f"{cid}.json").read_text(encoding="utf-8"))["cams"])


def cmd_fix_plan(a) -> int:
    from . import chunks as C
    tpl = json.loads((jobs_dir(a.dance) / f"{TEMPLATE_ID}.json").read_text(encoding="utf-8"))
    only = [x for x in (a.only or "").split(",") if x] or list(WINDOWS)
    plans = [plan_window(cid, tpl) for cid in only]
    print(f"{'id':<7}{'k':<14}{'window (s)':<16}{'cams':<22}{'4090 min':>9}{'USD':>6}  offsets (ms)")
    for p in plans:
        e = p["est"]
        print(f"{p['id']:<7}{p['k0']}..{p['k1']:<8}{p['k0'] / HZ:6.1f}-{(p['k1'] + 1) / HZ:<9.1f}{','.join(p['cams']):<22}{e['total_min_4090']:>9}{e['usd_4090']:>6}  "
              f"{p['sync_fixed']['offsets_ms']}  masks {p['masks'] or '-'}")
    tot = sum(p["est"]["usd_4090"] for p in plans)
    print(f"total {tot:.2f} USD on RTX 4090 pods (the model of recap_profile; the real h0000 will be about 1.1)")
    if a.dry_run:
        return 0
    st = State(a.dance)
    code = C.code_info()
    for p in plans:
        job = build_fix_job(tpl, p["id"], p["k0"], p["k1"], p["cams"], p["est"], p["sync_fixed"], p["masks"], p["why"], code=code, qa_thr=QA_THR_BY_ID.get(p["id"]))
        C.write_job(job)
        st.put_chunks([job["chunk"]], replace_planned=False)
        st.set_job(p["id"], chunk=p["id"], kind="chunk", status="planned", after=[], written=now_iso(), spec=str(jobs_dir(a.dance) / f"{p['id']}.json"))
    atomic_write_json(farm_dir(a.dance) / "fix_plan.json", {"created": now_iso(), "agent": "T1", "plans": [{k: v for k, v in p.items() if k != "stats"} for p in plans]})
    print(f"{len(plans)} job spec(s) -> {jobs_dir(a.dance)}; next: queue --only {','.join(p['id'] for p in plans)}")
    return 0


def cmd_fix_final(a) -> int:
    from . import chunks as C
    root = jobs_dir(a.dance)
    tpl = json.loads((root / "final.json").read_text(encoding="utf-8"))
    want = [x for x in (a.ids or "").split(",") if x]
    specs = []
    for c in tpl["stitch_chunks"]:
        if c["id"] in REPLACED or (want and c["id"] not in want):
            continue
        specs.append(c)
    for cid in WINDOWS:
        p = root / f"{cid}.json"
        if p.exists() and (not want or cid in want):
            specs.append(json.loads(p.read_text(encoding="utf-8"))["chunk"])
    job = final2_job(tpl, specs)
    print(f"final2: {len(specs)} chunks: " + ", ".join(f"{c['id']}[{c['k0']}..{c['k1']}]" for c in sorted(specs, key=lambda c: c['k0'])))
    if a.dry_run:
        return 0
    C.write_job(job)
    State(a.dance).set_job("final2", kind="stitch_analyse", status="planned", after=job["after"], written=now_iso(), spec=str(root / "final2.json"))
    print(f"written {root / 'final2.json'}")
    return 0


# ------------------------------------------------------------------------------------------------ QA refits of existing chunks (same id)

QA_REFITS = {          # id -> manual masks (master seconds) for CAMERA-TRACK faults only; wrong SAM masks are the qa_mask step's job
    "g4644": [],
    "g6171": [("02", 242.0, 258.7)],     # phone 02's R2 track moves 1 m along its axis while the phone is held still (NOTES_F1.md section 6)
}
QA_THR = 0.12


def qa_refit_spec(base: dict, masks: list, ckpt_rel: str, tag: str, thr: float = QA_THR) -> dict:
    """The refit job spec of recap_refit.refit_spec (front checkpoint of the first run as an archive input, no front) with the qa_mask step after the
    manual mask (no mask step when there is nothing to mask: `mask --mask ` with an empty list is an argparse error)."""
    from . import recap_refit as RR
    spec = RR.refit_spec(base, masks or [("00", 0.0, 0.001)], ckpt_rel, tag)
    steps = []
    for st in spec["steps"]:
        if st["name"] == "mask":
            if masks:
                steps.append(st)
            steps.append({**qa_step(spec["id"]), "cmd": f"python -m dancecap.recap_front qa-mask {spec['id']} --thr {thr:g}"})
            continue
        steps.append(st)
    spec["steps"] = steps
    spec["refit"]["masks"] = [{"cam": c, "t0": a, "t1": b} for c, a, b in masks]
    spec["refit"]["qa_thr"] = thr
    return spec


def cmd_qa_refit(a) -> int:
    import shutil
    from . import chunks as C
    from . import cli
    from . import launch
    from . import recap_refit as RR
    from .state import State
    ids = [x for x in (a.only or "").split(",") if x] or list(QA_REFITS)
    st = State(a.dance)
    root = work_root()
    plan = []
    for cid in ids:
        base = json.loads((jobs_dir(a.dance) / f"{cid}.v1.json").read_text(encoding="utf-8"))
        ckpt_rel = f"src/{RR.CAPTURE}/refit/{cid}.r1.front.tar"
        spec = qa_refit_spec(base, QA_REFITS[cid], ckpt_rel, a.tag, a.thr)
        plan.append((cid, spec, ckpt_rel))
        print(f"{cid}: steps {[s['name'] for s in spec['steps']]}  manual masks {QA_REFITS[cid] or '-'}  qa thr {a.thr:g}  max {spec['max_minutes']} min")
    if a.dry_run:
        return 0
    with launch.volume_link(a.dance, keep=True) as link:
        for cid, spec, ckpt_rel in plan:
            out = link.sh(f"ls -l /workspace/hm/{ckpt_rel} 2>&1 | head -1", 60)
            if cid not in out:
                print(f"{cid}: the front checkpoint {ckpt_rel} is not on the volume: {out[-200:]}")
                return 3
    for cid, spec, ckpt_rel in plan:
        keep = root / DANCE / "chunks" / a.tag_prev / cid
        take_dir = root / take_of(cid)
        if take_dir.exists() and not keep.exists():
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(take_dir, keep / "work")
            pulled = farm_dir(a.dance) / "chunks" / cid
            if pulled.exists():
                shutil.copytree(pulled, keep / "farm_chunk")
                shutil.rmtree(pulled, ignore_errors=True)
        spec_path = jobs_dir(a.dance) / f"{cid}.json"
        shutil.copyfile(spec_path, spec_path.with_name(f"{cid}.{a.tag_prev}.json"))
        atomic_write_json(spec_path, spec)
        st.set_chunk(cid, status="planned", pod=None, progress=0.0, step=None, eta_min=None, pull=None, done=None, last_error=None, requeues=0, refit=a.tag,
                     note=f"qa refit {a.tag}: qa_mask thr {a.thr:g}, manual {QA_REFITS[cid]}")
        st.set_job(cid, status="planned", refit=a.tag)
    ns = type("NS", (), {"only": ",".join(c for c, _, _ in plan), "exclude": None, "force_foreign": False, "keep_gateway": False, "hold": False})()
    res = cli.queue_jobs(a.dance, args=ns)
    print(json.dumps({k: res.get(k) for k in ("queued", "errors", "missing")}, default=str))
    return 0 if all(c in (res.get("queued") or []) for c, _, _ in plan) else 1


def cmd_fix_analysis(a) -> int:
    from . import analysis as A
    from . import chunks as C
    from . import recap_analysis as RA
    from .timeline import DanceSpec
    spec = DanceSpec.load(a.dance)
    jobs = analysis2_jobs(RA.build_jobs(spec, code=None if a.dry_run else C.code_info()))
    for j in jobs:
        print(f"job {j['id']}: " + " ; ".join(s["cmd"] for s in j["steps"]) + f"  (after {','.join(j['after'])}, kind {j['kind']})")
        if not a.dry_run:
            A.write_job(j)
            State(a.dance).set_job(j["id"], kind=j["kind"], status="planned", after=j["after"], written=now_iso(), spec=str(jobs_dir(a.dance) / f"{j['id']}.json"))
    return 0


def register(sub) -> None:
    q2 = sub.add_parser("qa-refit", help="T1: refit g4644 / g6171 from their first front checkpoints with the qa_mask step (same ids, first results kept in chunks/<tag_prev>/)")
    q2.add_argument("--only", default="")
    q2.add_argument("--tag", default="r2")
    q2.add_argument("--tag-prev", dest="tag_prev", default="r1")
    q2.add_argument("--thr", type=float, default=QA_THR)
    q2.add_argument("--dry-run", action="store_true")
    q2.set_defaults(func=cmd_qa_refit)
    r = sub.add_parser("fix-analysis", help="T1: the beat-free analyses again behind the second stitch (an2-timing, an2-physics, an2-floor)")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_fix_analysis)
    p = sub.add_parser("fix-plan", help="T1: plan the g0000 redo (h0000) and the four boundary refits (x<k>) as chunk jobs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", default="", help="comma list of ids (default all)")
    p.set_defaults(func=cmd_fix_plan)
    q = sub.add_parser("fix-final", help="T1: write the second stitch job (final2) over the fixed chunk set")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--ids", default="", help="comma list of the chunk ids to stitch (default: the first stitch's chunks without g0000 + every fix window with a spec)")
    q.set_defaults(func=cmd_fix_final)
