"""python -m dancecap.sides <verb> --take <t>   (DESIGN 4.1)

    evidence | decide | crops ... | prepare | plan | calibrate-gate | solve --part i/n | regate | merge | final | sheets | report | export
    | install
    job [--kind evidence | solve --parts n]  | obs | adapter

The install is the Unity stage's (the orchestrator's): `install --check` only verifies; `install` copies the motion files under
work/unity.lock with a once-only backup; `install --restore` puts the backup back.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERBS = ["evidence", "decide", "report", "job", "obs", "adapter", "crops", "prepare", "plan", "calibrate-gate", "solve", "regate", "merge",
         "gate", "final", "sheets", "export", "install"]


def _span(s: str) -> tuple[float, float]:
    a, b = s.split(":") if ":" in s else s.split("-")
    return float(a), float(b)


def _part(s: str) -> tuple[int, int]:
    a, b = s.split("/")
    return int(a), int(b)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "crops":
        from . import crops
        return crops.main(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m dancecap.sides")
    ap.add_argument("verb", choices=VERBS)
    ap.add_argument("--take", required=True)
    ap.add_argument("--workers", type=int, default=None, help="evidence: processes for the triangulation; solve: worker processes")
    ap.add_argument("--range", type=_span, default=None, help="evidence: only these dance seconds, e.g. 28:36 (a development run)")
    ap.add_argument("--out", type=Path, default=None, help="evidence / decide: another output folder (development runs)")
    ap.add_argument("--evidence-dir", type=Path, default=None, help="decide: read the evidence from this folder")
    ap.add_argument("--crops", type=Path, default=None, help="decide: crops.json (default work/<take>/sides/crops.json when it exists)")
    ap.add_argument("--cfg", default=None, help="JSON overrides of SidesCfg (after the take's [sides] table and sides/cfg.json), or @<json file>")
    ap.add_argument("--compare", action="append", default=[], help="report / sheets: name=<fit dir> scored next to the base (repeatable)")
    ap.add_argument("--first", action="append", default=[], type=_span, help="report: film-second spans listed first (repeatable)")
    ap.add_argument("--obs-from", type=Path, default=None, help="job: the compressed obs folder to upload")
    ap.add_argument("--max-minutes", type=int, default=60, help="job: the pod's time limit")
    ap.add_argument("--kind", default="evidence", choices=["evidence", "solve"], help="job: J1 (evidence + decide) or J2 (solve)")
    ap.add_argument("--parts", type=int, default=2, help="job --kind solve: pods")
    ap.add_argument("--tag", default="C", help="job --kind solve: job name tag (re-runs)")
    ap.add_argument("--part", type=_part, default=(0, 1), help="solve: i/n")
    ap.add_argument("--device", default="cuda", help="solve: cuda | cpu")
    ap.add_argument("--only", default=None, help="solve / job: comma-separated window ids")
    ap.add_argument("--solve-dir", type=Path, default=None, help="regate / merge: the solve results folder (default sides/solve)")
    ap.add_argument("--times", default=None, help="sheets: comma-separated dance seconds for the owner sheet")
    ap.add_argument("--capture", default=None, help="export / install: the Unity capture (default cfg.install_capture)")
    ap.add_argument("--capture-dir", type=Path, default=None, help="install: the capture folder (default StreamingAssets/<capture>)")
    ap.add_argument("--check", action="store_true", help="install: verify only")
    ap.add_argument("--restore", action="store_true", help="install: put the backed-up motion back")
    a = ap.parse_args(argv)
    from .config import load_cfg
    # --cfg '{...}' or --cfg @<json file> (a pod job ships the file: no JSON quoting in its command line)
    over = (json.loads(Path(a.cfg[1:]).read_text(encoding="utf-8")) if a.cfg.startswith("@") else json.loads(a.cfg)) if a.cfg else None
    cfg = load_cfg(a.take, {k: v for k, v in over.items() if not k.startswith("_")} if over else None)
    only = [int(x) for x in a.only.split(",")] if a.only else None
    if a.verb == "evidence":
        from . import evidence
        rep = evidence.run(a.take, cfg, a.out, a.range, a.workers)
        print(json.dumps({"seconds": rep["seconds"], "role_check": {c: v["stored_better"] for c, v in rep["role_check"].items()}}))
    elif a.verb == "decide":
        from . import decide
        print(json.dumps(decide.run(a.take, cfg, a.crops, a.out, a.evidence_dir)))
    elif a.verb == "report":
        from . import report
        comp = dict(x.split("=", 1) for x in a.compare)
        rep = report.run(a.take, cfg, comp or None, a.first or None)
        print(json.dumps({"counts": rep["counts"], "lanes": rep["lanes"]}))
    elif a.verb == "job":
        from . import job
        if a.kind == "evidence":
            j, p = job.job_evidence(a.take, cfg, workers=a.workers or 8, max_minutes=a.max_minutes, obs_from=a.obs_from)
            print(p)
            print(f"python -m dancecap remote run --jobs {p} --max-cost 0.5 --cpu --vcpu 8")
        else:
            res = job.job_solve(a.take, a.parts, cfg, max_minutes=a.max_minutes, tag=a.tag, only=only, workers=a.workers or 4)
            paths = ",".join(str(p) for _, p in res)
            print(paths)
            print(f"python -m dancecap remote run --jobs {paths} --parallel --max-cost <per pod>")
    elif a.verb == "obs":
        from . import adapters
        print(json.dumps(adapters.build_obs(a.take, cfg)))
    elif a.verb == "adapter":
        from . import adapters
        print(json.dumps(adapters.adapter_for(a.take, cfg).summary(), indent=1))
    elif a.verb == "prepare":
        from . import prepare
        rep = prepare.run(a.take, cfg)
        print(json.dumps({"splices": rep["splices"], "changed": rep["changed"], "checks": rep["checks"]}, default=str))
    elif a.verb == "plan":
        from . import plan
        print(json.dumps(plan.plan_windows(a.take, cfg)["summary"]))
    elif a.verb == "calibrate-gate":
        from . import gate
        rep = gate.calibrate_gate(a.take, cfg)
        print(json.dumps({"ok": rep["ok"], "cases": {k: v.get("ok") for k, v in rep["cases"].items()}}))
    elif a.verb == "solve":
        from . import solve
        i, n = a.part
        print(json.dumps(solve.solve_all(a.take, i, n, a.device, cfg, only, workers=a.workers)["windows"]))
    elif a.verb == "regate":
        from . import merge
        rg = merge.regate(a.take, cfg, workers=a.workers or 4, ids=only, solve_dir=a.solve_dir)
        print(json.dumps({i: v["chosen"] for i, v in rg["windows"].items()}))
    elif a.verb in ("merge", "gate"):
        from . import merge
        print(json.dumps(merge.merge(a.take, cfg, solve_dir=a.solve_dir)["summary"]))
    elif a.verb == "final":
        from . import report
        comp = dict(x.split("=", 1) for x in a.compare)
        rep = report.final(a.take, cfg, comp or None)
        print(json.dumps({"summary": rep["summary"], "windows": len(rep["windows"])}))
    elif a.verb == "sheets":
        from . import look
        comp = dict(x.split("=", 1) for x in a.compare)
        times = [float(x) for x in a.times.split(",")] if a.times else None
        print(json.dumps(look.sheets(a.take, comp or None, owner_times=times)))
    elif a.verb == "export":
        from . import install
        cap = a.capture or cfg.install_capture
        rep = install.side_export(a.take, cap)
        print(json.dumps({"sha256": rep["sha256"], "proof_ok": rep["proof_vs_input_export"].get("ok")}))
    elif a.verb == "install":
        from . import install
        from ..export_unity import UNITY_STREAMING
        from ..take import PACKAGE_ROOT
        cap = a.capture or cfg.install_capture
        cd = a.capture_dir or (UNITY_STREAMING / cap)
        rep = install.install(install.unity_motion_dir(a.take), cd, check=a.check, restore=a.restore, lock=PACKAGE_ROOT / "work" / "unity.lock")
        print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
