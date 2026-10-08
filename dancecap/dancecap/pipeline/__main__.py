"""CLI of dancecap.pipeline (see the package docstring)."""
from __future__ import annotations

import argparse
import json
import sys

PLANNED_CMDS = {
    "pair-calib": "two-view calibration from the dancers (stage pair_calib)",
    "refine-cameras": "human-keypoint camera refinement (stage cam_refine_body)",
}


def _fmt_s(s) -> str:
    if s is None:
        return "-"
    m = s / 60
    return f"{m / 60:.1f} h" if m >= 90 else f"{m:.0f} min"


def cmd_stages(a) -> int:
    from . import registry
    st = registry.variant_stages(a.variant)
    if a.json:
        print(json.dumps(st, indent=1, default=str))
        return 0
    for s in st:
        where = {"local": "[L]", "local_gpu": "[G]", "pod": "[P]", "pod_secure": "[S]", "human": "[H]",
                 "unity": "[U]"}.get(s.get("runs_on"), "[?]")
        impl = "" if s.get("implemented") else "  (planned)"
        print(f"{s['column']} {where} {s['id']:16s} <- {', '.join(s['deps']) or '-'}{impl}")
    return 0


def cmd_status(a) -> int:
    from . import status
    ctx = status.context(a.take)
    snap = status.snapshot(a.take, ctx)
    if a.json:
        print(json.dumps(snap, indent=1, default=str))
        return 0
    print(f"{a.take}: {ctx['variant']}-view, {len(ctx['used'])} camera(s); workload {ctx['wl']['V']:.2f} video-min, "
          f"{ctx['wl']['CM']:.2f} camera-min")
    for x in snap["stages"]:
        reason = x.get("status_reason") or "; ".join(x["outputs"]["stale_because"][:2])
        print(f"  {x['id']:16s} {x['status']:8s} {x['runs_on'] or '':10s} {reason[:100] if reason else ''}")
    r = snap["remaining"]
    print(f"remaining (critical path) p50 {_fmt_s(r['p50_s'])} / p90 {_fmt_s(r['p90_s'])}; pod cost p50 "
          f"${snap['pod_cost_remaining_usd']['p50']:.2f} / p90 ${snap['pod_cost_remaining_usd']['p90']:.2f}")
    return 0


def cmd_plan(a) -> int:
    from . import plan
    p = plan.make_plan(a.take, {"target": a.target, "scope": a.scope, "mode": a.mode,
                                "options": {"capture": a.capture, "assume_reviews": bool(a.what_if)}})
    if a.json:
        print(json.dumps(p, indent=1, default=str))
        return 0
    for st in p["steps"]:
        ct = st.get("camera_types")
        print(f"  {st['stage']:16s} {st['action']:12s} {st.get('reason') or ''}"
              + (f"  [cameras: {'; '.join(f'{k} {chr(44).join(v)}' for k, v in ct.items())}]" if ct else ""))
    for s in p["sessions"]:
        print(f"  session {s['id']}: {s['stages']} p50 {s['est_min']['p50']} min / p90 {s['est_min']['p90']} min, "
              f"${s['cost_usd']['p50']:.2f} / ${s['cost_usd']['p90']:.2f}")
    t = p["totals"]
    print(f"wall p50 {_fmt_s(t['wall_s']['p50'])} / p90 {_fmt_s(t['wall_s']['p90'])}; pods ${t['pod_cost_usd']['p50']:.2f}"
          f" / ${t['pod_cost_usd']['p90']:.2f}; budget gate {p['budget']['gate']}; plan {p['plan_hash']}")
    for w in p["warnings"]:
        print(f"  ! {w}")
    return 0


def cmd_create(a) -> int:
    from . import intake
    spec = json.loads(open(a.spec, encoding="utf-8-sig").read())
    try:
        res = intake.create(spec, dry_run=a.dry_run)
    except intake.IntakeError as e:
        for p in e.problems:
            print(f"  {p['field']}: {p['message']}", file=sys.stderr)
        return 1
    for w in res["warnings"]:
        print(f"  ! {w}")
    print(res["toml_preview"] if a.dry_run else f"[pipeline] wrote {res['toml_path']} ({res['variant']}-view)")
    return 0


def cmd_gate(a) -> int:
    """Confirm a human gate from the terminal (the monitor's POST /api/takes/<take>/gates/<stage>)."""
    import subprocess
    import time
    from . import gates, paths, registry
    from .util import read_json, write_json_atomic
    if a.stage == "labels_review":
        p = paths.take_work(a.take) / "intake.json"
        ij = read_json(p, {}) or {}
        ij["labels_reviewed_at"] = time.time()
        write_json_atomic(p, ij)
        print(f"[pipeline] {a.take}: labels review confirmed")
        return 0
    if a.stage not in gates.DECISIONS:
        print(f"[pipeline] gates: labels_review, {', '.join(gates.DECISIONS)}", file=sys.stderr)
        return 2
    allowed = list(gates.DECISIONS[a.stage]) + ["reject"]
    decision = a.decision or (gates.DECISIONS[a.stage][0] if len(gates.DECISIONS[a.stage]) == 1 else None)
    if decision not in allowed:
        print(f"[pipeline] --decision {' | '.join(allowed)}", file=sys.stderr)
        return 2
    if not gates.is_open(a.take, a.stage):
        print(f"[pipeline] no open review gate at {a.stage} for {a.take}", file=sys.stderr)
        return 1
    if decision == "apply":
        for argv in registry.gate_apply_commands(a.stage, {"take": a.take}):
            rc = subprocess.run([sys.executable, *argv], cwd=str(paths.root())).returncode
            if rc != 0:
                print(f"[pipeline] {' '.join(argv)} failed (rc {rc}): the gate stays open", file=sys.stderr)
                return rc
    gates.confirm(a.take, a.stage, decision)
    print(f"[pipeline] {a.take}: {a.stage} review confirmed ({decision})")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        from .orchestrator import main as run_main
        return run_main(argv[1:])
    if argv and argv[0] == "probe":
        from .probe import main as probe_main
        return probe_main(argv[1:])
    if argv and argv[0] == "triggers":
        from .triggers import main as trig_main
        return trig_main(argv[1:])
    if argv and argv[0] == "audio-sync":
        from .audio_sync import main as sync_main
        return sync_main(argv[1:])
    if argv and argv[0] == "export":
        from .export import main as export_main
        return export_main(argv[1:])
    if argv and argv[0] in PLANNED_CMDS:
        print(f"[pipeline] {argv[0]}: planned, not built yet - {PLANNED_CMDS[argv[0]]} (docs/PIPELINE.md)",
              file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("monitor", help="the Pipeline Monitor (local web UI)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--serve", action="store_true", help="run the server (normally started by --launch)")
    g.add_argument("--launch", action="store_true", help="open a window; start the server when none runs")
    p.add_argument("--port", type=int)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--take")
    p.add_argument("--view", choices=["videos", "overview", "runpod", "new"])
    p.add_argument("--capture", help="open the take that exported this Unity capture")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--foreground", action="store_true", help="--serve: keep stdout (debugging)")
    p = sub.add_parser("stages")
    p.add_argument("--variant", default="multi", choices=["multi", "two", "single"])
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("take")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("take")
    p.add_argument("--target")
    p.add_argument("--scope", choices=["only", "upto", "all"])
    p.add_argument("--mode", choices=["resume", "force"], default="resume")
    p.add_argument("--capture")
    p.add_argument("--what-if", action="store_true", help="assume your reviews are done (dry-run-only plan)")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("create", help="register a new capture from a JSON spec (TakeCreate with file paths)")
    p.add_argument("--spec", required=True)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("gate", help="confirm a review gate: fit_qa ok|reject, geoveto apply|keep|reject, labels_review")
    p.add_argument("take")
    p.add_argument("stage")
    p.add_argument("--decision")
    sub.add_parser("run", help="orchestrator (see python -m dancecap.pipeline run --help)")
    sub.add_parser("probe", help="ffprobe + camera-type suggestion (python -m dancecap.pipeline probe <take>)")
    sub.add_parser("triggers", help="viewer triggers (python -m dancecap.pipeline triggers <take>)")
    sub.add_parser("audio-sync", help="audio clocks of the take's videos (python -m dancecap.pipeline audio-sync <take>)")
    sub.add_parser("export", help="demo_export + the triggers export (python -m dancecap.pipeline export <take> "
                                  "--capture <Name>)")
    a = ap.parse_args(argv)
    if a.cmd == "monitor":
        from . import monitor
        if a.serve:
            return monitor.serve(port=a.port, replace=a.replace, redirect=not a.foreground)
        take = a.take
        if a.capture and not take:
            take = _take_for_capture(a.capture)
        return monitor.launch(take=take, view=a.view, replace=a.replace, no_window=a.no_window)
    return {"stages": cmd_stages, "status": cmd_status, "plan": cmd_plan, "create": cmd_create,
            "gate": cmd_gate}[a.cmd](a)


def _take_for_capture(capture: str) -> str | None:
    from . import paths
    from .util import CAPTURE_RE
    if not CAPTURE_RE.match(capture or ""):
        return None
    hits = sorted(paths.work().glob(f"*/export/{capture}_demo_export.json"), key=lambda p: p.stat().st_mtime)
    return hits[-1].parents[1].name if hits else None


if __name__ == "__main__":
    sys.exit(main())
