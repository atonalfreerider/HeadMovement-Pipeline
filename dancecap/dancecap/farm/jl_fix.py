"""john_lula seam polish (agent J3b, 2026-10-11): the boundary refit of the one seam and the second stitch (farm/ca_fix.py made for this take).

    python -m dancecap.farm --dance john_lula jl-fix-plan [--dry-run]      # job spec of the refit chunk jlx1158 (12 s centred on the failing blend, the stitcher's own window)
    python -m dancecap.farm --dance john_lula up --no-plan --only jlx1158 ...
    python -m dancecap.farm --dance john_lula jl-fix-final [--dry-run]     # the second stitch job (jlfinal2) over the three chunks, writing work/john_lula_t2/

The template is the spec of jlg0000 (steps of a working job: single camera, mono mode, pinned betas, qa_mask), the window the one the stitcher named for the failed seam
(stitch.json: refit window k 1158..1518 = 38.60-50.57 s).
"""
from __future__ import annotations

import json

from . import farm_dir, jobs_dir
from .state import State, now_iso

DANCE = "john_lula"
TEMPLATE_ID = "jlg0000"
HZ = 30.0
WINDOWS = {"jlx1158": (1158, 1518, "seam jlg0000|jlg1322 (43.6-45.6 s) failed the body / elbow gates: 38.60-50.57 s, one continuous fit across the blend (the stitcher's refit window)")}
MARGINS = {"jlg0000": 0.2, "jlx1158": 0.5, "jlg1322": 0.2}
SELECT_WEIGHTS = "arm=0.3,fingers=0.3"
OUT_TAKE = "john_lula_t2"
FINAL_ID = "jlfinal2"


def take_of(cid: str) -> str:
    return f"{DANCE}_{cid}"


def build_fix_job(tpl: dict, cid: str, k0: int, k1: int, est: dict, why: str, code: dict | None = None) -> dict:
    from . import recap_fix as RF
    T = k1 - k0 + 1
    job = RF.build_fix_job(tpl, cid, k0, k1, list(tpl["cams"]), est, dict(tpl["chunk"]["sync_fixed"]), [], why, code=code)
    qa = [i for i, st in enumerate(job["steps"]) if st["name"] == "qa_mask"]
    if len(qa) > 1:            # recap_fix adds a qa_mask step after `front`; the template already has one (with its `save` list): keep that one
        job["steps"].pop(next(i for i in qa if not job["steps"][i].get("save")))
    job["take"] = take_of(cid)
    job["chunk"]["take"] = take_of(cid)
    job["chunk"]["origin"] = "fix"
    job["outputs"] = json.loads(json.dumps(job["outputs"]).replace("larissa_kadu_recap_", f"{DANCE}_"))     # recap_fix names its optional mask / qa_mask outputs for the recap
    job["owner"] = tpl.get("owner")
    job["fix"] = {"agent": "J3b", "why": why, "sync_fixed": job["chunk"]["sync_fixed"]}
    job["profile"] = tpl.get("profile", "john_lula")
    job["chunk"]["T"] = T
    return job


def final_job(tpl_final: dict, chunk_specs: list[dict], out_take: str = OUT_TAKE, jid: str = FINAL_ID) -> dict:
    from . import recap_fix as RF
    job = RF.final2_job(tpl_final, chunk_specs, out_take=out_take, jid=jid, margins=MARGINS, select_weights=SELECT_WEIGHTS)
    s = json.dumps(job["outputs"]).replace("larissa_kadu_recap", DANCE)
    job["outputs"] = json.loads(s)
    job["chunk"]["take"] = out_take
    job["owner"] = tpl_final.get("owner")
    for st in job["steps"]:
        if st["name"] == "evidence":
            st["cmd"] = st["cmd"].replace("chunk-evidence jlfinal ", f"chunk-evidence {jid} ")
    return job


def estimate(cid: str, k0: int, k1: int):
    from . import chunks as C
    from . import jl_profile as P
    ch = C.Chunk(id=cid, k0=k0, k1=k1, core_k0=k0, core_k1=k1, overlap_k=0, kind="full", cams=["01"], take=take_of(cid))
    return P.estimate_chunk(ch, cam_frames=ch.T)


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
    tpl = json.loads((root / "jlfinal.json").read_text(encoding="utf-8"))
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
    p = sub.add_parser("jl-fix-plan", help="J3b: job spec of the john_lula seam boundary refit (jlx1158)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_fix_plan)
    q = sub.add_parser("jl-fix-final", help="J3b: the second stitch job (jlfinal2) over the original chunks + the refit")
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(func=cmd_fix_final)
