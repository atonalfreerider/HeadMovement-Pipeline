"""carlos_aline seam polish (agent C6, 2026-10-10): the boundary refit of seam 1 and the second stitch.

    python -m dancecap.farm --dance carlos_aline ca-fix-plan [--dry-run]       # job spec of the refit chunk cax1408 (12 s centred on the failing blend)
    python -m dancecap.farm --dance carlos_aline queue --only cax1408
    python -m dancecap.farm --dance carlos_aline ca-fix-final [--dry-run]      # the second stitch job (cafinal2) over the five chunks

The recap did the same (farm/recap_fix.py: a full chunk job from the spec of a working chunk, 12-s window centred on the failing blend, the same pinned betas, the
neighbours' camera set and offsets, the qa_mask step), generalised here to this take: the template is the spec of cag1488 (inputs on the volume, steps of a working job),
cameras are both (a two-camera take), the video offsets are C3's fixed ones, and `HM_TEACHER_EVIDENCE=1` is set because the demo's colour rule picks the label frame.
The stitch writes into work/carlos_aline_t6/ (the first stitch in work/carlos_aline/ is untouched until the result is verified and promoted by hand).
"""
from __future__ import annotations

import copy
import json

from . import farm_dir, jobs_dir
from .state import State, now_iso

DANCE = "carlos_aline"
TEMPLATE_ID = "cag1488"
HZ = 30.0
# id -> (k0, k1, why). Seam 1 (blend 51.93-53.90 s) failed; PROVISIONAL.md names the improvement path: refit 46.93-58.90 s (grid k 1408..1768).
WINDOWS = {
    "cax1408": (1408, 1768, "seam cag0000|cag1488 (51.9-53.9 s): 46.93-58.90 s, one continuous fit across the failing blend"),
}
MARGINS = {"cag0000": 0.2, "cax1408": 0.5, "cag1488": 0.2, "cag2916": 0.2, "cag4345": 0.2}
SELECT_WEIGHTS = "arm=0.3,fingers=0.3"          # the recap's choice (body gates decide where the blend sits; every exceeded gate is still reported)
OUT_TAKE = "carlos_aline_t6"
FINAL_ID = "cafinal2"
EXTRA_ENV = {"HM_TEACHER_EVIDENCE": "1"}


def take_of(cid: str) -> str:
    return f"{DANCE}_{cid}"


def build_fix_job(tpl: dict, cid: str, k0: int, k1: int, est: dict, why: str, code: dict | None = None) -> dict:
    """The farm job spec of one refit chunk from the template chunk job's spec (pure)."""
    from . import recap_fix as RF
    T = k1 - k0 + 1
    job = RF.build_fix_job(tpl, cid, k0, k1, list(tpl["cams"]), est, dict(tpl["chunk"]["sync_fixed"]), [], why, code=code)
    qa = [i for i, st in enumerate(job["steps"]) if st["name"] == "qa_mask"]
    if len(qa) > 1:            # recap_fix adds a qa_mask step after `front`; this take's template already has one (with its `save` list): keep that one
        job["steps"].pop(next(i for i in qa if not job["steps"][i].get("save")))
    job["take"] = take_of(cid)
    job["chunk"]["take"] = take_of(cid)
    job["chunk"]["origin"] = "fix"
    job["env"] = {**(job.get("env") or {}), **EXTRA_ENV}
    job["fix"] = {"agent": "C6", "why": why, "sync_fixed": job["chunk"]["sync_fixed"]}
    job["profile"] = tpl.get("profile", "carlos_aline")
    job["chunk"]["T"] = T
    return job


def final_job(tpl_final: dict, chunk_specs: list[dict], out_take: str = OUT_TAKE, jid: str = FINAL_ID) -> dict:
    """cafinal with the refit chunk added, writing into work/<out_take>/ (recap_fix.final2_job, outputs re-pointed to this dance)."""
    from . import recap_fix as RF
    job = RF.final2_job(tpl_final, chunk_specs, out_take=out_take, jid=jid, margins=MARGINS, select_weights=SELECT_WEIGHTS)
    s = json.dumps(job["outputs"]).replace("larissa_kadu_recap", DANCE)
    job["outputs"] = json.loads(s)
    job["env"] = {**(job.get("env") or {}), **EXTRA_ENV}
    job["chunk"]["take"] = out_take
    for st in job["steps"]:
        if st["name"] == "evidence":
            st["cmd"] = st["cmd"].replace("chunk-evidence cafinal ", f"chunk-evidence {jid} ")
    return job


def estimate(cid: str, k0: int, k1: int):
    from . import chunks as C
    from . import ca_profile as P
    ch = C.Chunk(id=cid, k0=k0, k1=k1, core_k0=k0, core_k1=k1, overlap_k=0, kind="full", cams=["01", "02"], take=take_of(cid))
    return P.estimate_chunk(ch, cam_frames=2 * ch.T)


def cmd_fix_plan(a) -> int:
    from . import chunks as C
    tpl = json.loads((jobs_dir(a.dance) / f"{TEMPLATE_ID}.json").read_text(encoding="utf-8"))
    code = None if a.dry_run else C.code_info()
    st = State(a.dance)
    for cid, (k0, k1, why) in WINDOWS.items():
        est = estimate(cid, k0, k1)
        job = build_fix_job(tpl, cid, k0, k1, est, why, code=code)
        print(f"{cid}: k {k0}..{k1} ({k0 / HZ:.2f}-{(k1 + 1) / HZ:.2f} s) cams {job['cams']}  est {est['total_min_4090']} min, {est['usd_4090']} USD (4090)  steps {[s['name'] for s in job['steps']]}")
        if a.dry_run:
            continue
        C.write_job(job)
        st.put_chunks([job["chunk"]], replace_planned=False)
        st.set_job(cid, chunk=cid, kind="chunk", status="planned", after=[], written=now_iso(), spec=str(jobs_dir(a.dance) / f"{cid}.json"))
    return 0


def cmd_fix_final(a) -> int:
    from . import chunks as C
    root = jobs_dir(a.dance)
    tpl = json.loads((root / "cafinal.json").read_text(encoding="utf-8"))
    specs = list(tpl["stitch_chunks"])
    for cid in WINDOWS:
        p = root / f"{cid}.json"
        if p.exists():
            specs.append(json.loads(p.read_text(encoding="utf-8"))["chunk"])
    job = final_job(tpl, specs)
    print(f"{FINAL_ID}: {len(specs)} chunks: " + ", ".join(f"{c['id']}[{c['k0']}..{c['k1']}]" for c in sorted(specs, key=lambda c: c['k0'])))
    if a.dry_run:
        return 0
    C.write_job(job)
    State(a.dance).set_job(FINAL_ID, kind="stitch_analyse", status="planned", after=job["after"], written=now_iso(), spec=str(root / f"{FINAL_ID}.json"))
    print(f"written {root / (FINAL_ID + '.json')}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("ca-fix-plan", help="C6: job spec of the carlos_aline seam-1 boundary refit (cax1408)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_fix_plan)
    q = sub.add_parser("ca-fix-final", help="C6: the second stitch job (cafinal2) over the original chunks + the refit")
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(func=cmd_fix_final)
