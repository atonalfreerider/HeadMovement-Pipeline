"""dancecap: multi-phone dance capture -> cameras, room, dancers, poses, physics, timing -> Unity.

    python -m dancecap ingest larissa_kadu            # window frames + room keyframes, native resolution
    python -m dancecap mamma larissa_kadu             # MAMMA with per-frame cameras -> smplx/{lead,follow}.npz
    python -m dancecap detect larissa_kadu            # all-person boxes, COCO-17 kpts, spectator ignore masks
    python -m dancecap dancers-labels larissa_kadu    # couple tracks + lead/follow labels + review sheet
    python -m dancecap dancers larissa_kadu           # SAM 2.1 lead/follow masks, identity repair, QA
    python -m dancecap cameras larissa_kadu           # static-room SfM + per-frame 6DoF/zoom -> cameras/<cam>.npz
    python -m dancecap timing synthetic --validate    # touchdowns/accents vs zouk beats -> timing/timing.json
    python -m dancecap physics synthetic --validate   # L0 centroidal dynamics -> physics/physics.{npz,json}
    python -m dancecap export synthetic --capture 01_SyntheticSMPLX  # -> head-movement StreamingAssets (v3)
"""
from __future__ import annotations

import argparse
import json

from .take import Take


def cmd_ingest(a):
    from .ingest import ingest_keyframes, ingest_window
    take = Take.load(a.take)
    cams = a.cams.split(",") if a.cams else None
    known = {s.id for s in take.sources}
    if cams and not set(cams) <= known:
        raise SystemExit(f"unknown camera ids {sorted(set(cams) - known)}; take has {sorted(known)}")
    if not a.keyframes_only:
        metas = ingest_window(take, cams)
        for cam, m in sorted(metas.items()):
            iv = m["frame_interval_ms"]
            print(f"[ingest] {cam}: {m['frame_count']} frames {m['width']}x{m['height']} "
                  f"~{m['fps_median']:.2f} fps (interval {iv.get('min', 0):.1f}-{iv.get('max', 0):.1f} ms) "
                  f"t={m['frames'][0]['t']:.3f}..{m['frames'][-1]['t']:.3f}")
    if not a.window_only:
        kfs = ingest_keyframes(take, cams)
        print(f"[ingest] keyframes: " + ", ".join(f"{c}:{len(v)}" for c, v in sorted(kfs.items())))


def cmd_detect(a):
    from .detect import detect_take
    take = Take.load(a.take)
    cams = a.cams.split(",") if a.cams else None
    known = {s.id for s in take.sources}
    if cams and not set(cams) <= known:
        raise SystemExit(f"unknown camera ids {sorted(set(cams) - known)}; take has {sorted(known)}")
    rep = detect_take(take, cams, keyframes=not a.no_keyframes)
    print(f"[detect] total {rep['runtime_s']:.0f}s -> {take.work / 'people' / 'report.json'}")


def _cams(take, cams):
    cams = cams.split(",") if cams else None
    known = {s.id for s in take.sources}
    if cams and not set(cams) <= known:
        raise SystemExit(f"unknown camera ids {sorted(set(cams) - known)}; take has {sorted(known)}")
    return cams


def cmd_dancers_labels(a):
    from .dancers import run_labels
    take = Take.load(a.take)
    run_labels(take, _cams(take, a.cams), typesafe=a.typesafe)
    print(f"[dancers] -> {take.work / 'dancers' / 'labels.json'}, review {take.work / 'dancers' / 'labels_review.jpg'}"
          f"; flip a camera with dancers/labels_override.json {{\"<cam>\": {{\"swap\": true}}}}")


def cmd_dancers(a):
    from .dancers import run, run_labels
    take = Take.load(a.take)
    cams = _cams(take, a.cams)
    if a.relabel:
        run_labels(take, cams)
    rep = run(take, cams, reuse_sam=a.reuse_sam)
    print(f"[dancers] -> {take.work / 'dancers'} ({len(rep['cams'])} cameras in report.json)")


def cmd_cameras(a):
    from .cameras.pipeline import STAGES, run
    take = Take.load(a.take)
    stages = tuple(a.stages.split(",")) if a.stages else STAGES
    if not set(stages) <= set(STAGES):
        raise SystemExit(f"unknown stages {sorted(set(stages) - set(STAGES))}; choose from {STAGES}")
    rep = run(take, _cams(take, a.cams), stages, sigma_s=a.sigma, max_reproj_px=a.max_reproj)
    for g, v in rep.get("gates_G2a", {}).items():
        print(f"[cameras] G2a {g}: {'PASS' if v['passed'] else 'FAIL'} {json.dumps(v.get('value'), default=str)[:200]}")
    print(f"[cameras] -> {take.work / 'cameras' / 'report.json'}")


def cmd_mamma(a):
    from . import mamma_dyn
    if a.validate_synthetic:
        rep = mamma_dyn.validate_synthetic(noise_px=a.noise, n_cams=a.n_cams, frames=a.frames,
                                           anatomical=a.anatomical)
        print(json.dumps({k: rep[k] for k in ("noise_px", "n_cams", "T", "mpjpe", "ma_3d", "smplx")}, indent=1,
                         default=str))
        return
    if not a.take:
        raise SystemExit("mamma: give a take name or --validate-synthetic")
    cams = a.cams.split(",") if a.cams else None
    rep = mamma_dyn.run(a.take, cams=cams, skip_2d=a.skip_2d)
    print(json.dumps({k: rep[k] for k in ("grid", "ma_3d", "smplx")}, indent=1, default=str))


def cmd_timing(a):
    from . import timing
    from .take import PACKAGE_ROOT
    cfg = timing.TimingConfig(
        allowed_types=tuple(int(x) for x in a.allowed_types.split(",")),
        allowed_positions=tuple(int(x) for x in a.positions.split(",")) if a.positions else None,
        hit_tol_s=a.tol_ms / 1000, mc_draws=a.mc, workers=a.workers, seed=a.seed, coarsen_grid=a.coarsen,
        refine_beats=not a.no_refine, av_offset_s=a.av_offset_ms / 1000, av_sync_event=a.av_sync_event,
        sync_sigma_s=a.sync_ms / 1000, n_cams=a.n_cams,
        noise_m=None if a.noise_mm is None else a.noise_mm / 1000,
        beat_jitter_s=None if a.beat_jitter_ms is None else a.beat_jitter_ms / 1000)
    work = PACKAGE_ROOT / "work" / a.take
    if a.validate:
        rep = timing.validate_synthetic(cfg, noise_levels=(0.0, a.noise_cm / 100), work=work)
        print(json.dumps(timing._clean(rep["gate"]), indent=1))
        return
    from pathlib import Path
    try:
        res = timing.run(work, cfg, Path(a.beats) if a.beats else None)
    except FileNotFoundError as e:
        raise SystemExit(f"timing: {e}")
    for role, d in res["asynchrony"].items():
        if role in timing.ROLES and d["footsteps"].get("n"):
            f = d["footsteps"]
            print(f"[timing] {role}: {f['n']} touchdowns, mean asynchrony {f['mean_ms']:+.1f} ms "
                  f"[{f['ci95_ms'][0]:+.1f}, {f['ci95_ms'][1]:+.1f}], R={f['R']:.2f}, hit {f['hit_rate']:.0%} "
                  f"(chance {f['surrogate']['hit_rate_mean']:.0%}, p={f['surrogate']['hit_rate_p']:.3f})")
    for w in res["beats"]["warnings"]:
        print(f"[timing] WARNING: {w}")
    print(f"[timing] -> {res['_out'] / 'timing.json'}")


def cmd_physics(a):
    from pathlib import Path

    import numpy as np

    from . import physics
    from .take import PACKAGE_ROOT
    cfg = physics.PhysicsConfig(cutoff_hz=a.cutoff, mu=a.mu, identifiable_width_n=a.identifiable_n, seed=a.seed,
                                n_boot=a.boot)
    if a.validate:
        if a.take != "synthetic":
            raise SystemExit("physics --validate runs on the synthetic take only")
        rep = physics.validate_synthetic(cfg, noise=((a.noise_mm / 1000, a.noise_deg),))
        print(json.dumps(physics._clean({"clean": rep["clean"], "G6_passed_all": rep["G6_passed_all"],
                                         "noisy_G6": [r["G6"] for k, v in rep.items() if k.startswith("noise")
                                                      for r in v]}), indent=1))
        return
    work = PACKAGE_ROOT / "work" / a.take
    if not (work / "smplx" / "lead.npz").exists():
        raise SystemExit(f"physics: {work / 'smplx'} has no lead/follow npz (run the mamma stage first)")
    pick = lambda x, y: {k: v for k, v in (("lead", x), ("follow", y)) if v is not None}
    res = physics.run(work, cfg, Path(a.beats) if a.beats else None, masses=pick(a.mass_lead, a.mass_follow),
                      statures=pick(a.height_lead, a.height_follow))
    for role in physics.ROLES:
        i = res[role]["info"]
        print(f"[physics] {role}: {i['mass_kg']:.1f} kg ({i['mass_source']}), stature {i['stature_m']:.2f} m, "
              f"contact {i['contact_fraction']}, F_net band95 median {i['f_net_band95_median_n']} N")
    if "cycles" in res and res["cycles"].get("n_cycles"):
        print(f"[physics] couple GRF_y over {res['cycles']['n_cycles']} measures: mean "
              f"{res['cycles']['mean_ratio']:.4f} x (mA+mB) g, worst {res['cycles']['max_abs_dev']:.2%}")
    if "partner" in res:
        st = res["partner"]["status"]
        print("[physics] partner force: " + ", ".join(f"{n} {np.mean(st == k):.0%}"
                                                    for k, n in physics.PARTNER_STATUS.items()))
    print(f"[physics] -> {res['_json']}")


def cmd_export(a):
    from pathlib import Path

    from . import export_unity
    try:
        rep = export_unity.export(a.take, a.capture, model=Path(a.model) if a.model else None,
                                  audio=Path(a.audio) if a.audio else None, audio_t0=a.audio_t0,
                                  beats=Path(a.beats) if a.beats else None, install=not a.no_install)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"export: {e}")
    for r, d in rep["dancers"].items():
        print(f"[export] {r}: {Path(d['model']).name} ({d['betas']} betas), FK vs npz joints {d['fk_max_err_mm']} mm")
    print(f"[export] {rep['frames']} frames @ {rep['fps']:.2f} fps: {', '.join(rep['files'])}")
    print(f"[export] -> {rep.get('installed', rep['out'])}")


UNITY_PROJECT = None  # resolved lazily: WORKSPACE/head-movement


def _unity(command: str, args: dict | None = None) -> str:
    import subprocess

    from .take import WORKSPACE
    cmd = ["unity", "command", command, "--result-only", "--no-banner", "--caller", "plugin", "--skill", "typesafe-ai",
           "--project-path", str(WORKSPACE / "head-movement")]
    for k, v in (args or {}).items():
        cmd += [f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def cmd_ask(a):
    from .judgments import route_lesson_command
    lesson = {}
    if a.run:
        raw = _unity("hm_state")
        try:
            lesson = json.loads(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            raise SystemExit(f"hm_state unavailable (is the editor in Play mode?): {raw[:200]}")
        lesson = {k: lesson.get(k) for k in ("captures", "capture", "measure", "measureCount", "speed", "playing", "loopMeasure", "layers")}
    routed = route_lesson_command(a.text, lesson)
    print(json.dumps({"commands": routed.commands, "confidence": round(routed.confidence, 3), "clarify": routed.clarify},
                     indent=1))
    if a.debug:
        print(json.dumps(routed.debug, indent=1, default=str))
    if a.run and routed.commands and routed.confidence < 0.4 and not a.force:
        print(f"weakest judgment {routed.confidence:.2f} < 0.4: not executing; rerun with --force to apply this plan")
        return
    if a.run and routed.commands and (routed.clarify is None or a.force):
        for name, args in routed.commands:
            print(f"> {name} {args}: {_unity(name, args)[:160]}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dancecap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("ingest")
    sp.add_argument("take")
    sp.add_argument("--cams")
    sp.add_argument("--window-only", action="store_true")
    sp.add_argument("--keyframes-only", action="store_true")
    sp.set_defaults(fn=cmd_ingest)
    sp = sub.add_parser("detect", help="all-person boxes + COCO-17 keypoints + spectator ignore masks (YOLO)")
    sp.add_argument("take")
    sp.add_argument("--cams")
    sp.add_argument("--no-keyframes", action="store_true")
    sp.set_defaults(fn=cmd_detect)
    sp = sub.add_parser("dancers-labels", help="couple candidate tracks + automatic lead/follow labels + review sheet")
    sp.add_argument("take")
    sp.add_argument("--cams")
    sp.add_argument("--typesafe", action="store_true", help="cross-check each camera's labels with TypeSafe")
    sp.set_defaults(fn=cmd_dancers_labels)
    sp = sub.add_parser("dancers", help="SAM 2.1 lead/follow masks (full/body/hair), identity repair, roles.json, QA")
    sp.add_argument("take")
    sp.add_argument("--cams")
    sp.add_argument("--relabel", action="store_true", help="recompute labels.json first (overrides still apply)")
    sp.add_argument("--reuse-sam", action="store_true",
                    help="reuse dancers/cache/<cam>_sam.npz when the label is unchanged (redo identity/post/QA only)")
    sp.set_defaults(fn=cmd_dancers)
    sp = sub.add_parser("synthetic", help="synthetic two-dancer SMPL-X take with ground truth (work/synthetic)")
    sp.set_defaults(fn=lambda a: __import__("dancecap.synthetic", fromlist=["main"]).main())
    sp = sub.add_parser("review", help="review sheets: all cameras at the same instants (re-ID, pose, spectator masks, SMPL-X)")
    sp.add_argument("take")
    sp.add_argument("--instants", type=int, default=6)
    sp.set_defaults(fn=lambda a: [print(f"[review] {p}") for p in __import__("dancecap.review", fromlist=["render"])
                                  .render(Take.load(a.take), a.instants)])
    sp = sub.add_parser("cameras", help="static-room SfM (ALIKED+LightGlue, COLMAP) + per-frame pose/zoom of every camera")
    sp.add_argument("take")
    sp.add_argument("--cams")
    sp.add_argument("--stages", help="comma-separated subset of map,loc,attach,world,qa (default: all; later stages reuse caches)")
    sp.add_argument("--sigma", type=float, default=0.04, help="temporal smoothing sigma (s) for pose and log-focal")
    sp.add_argument("--max-reproj", type=float, default=4.0, help="frames above this median reprojection (px) -> ok=False")
    sp.set_defaults(fn=cmd_cameras)
    sp = sub.add_parser("mamma", help="MAMMA ma_2d + ma_3d with moving/zooming cameras -> smplx/{lead,follow}.npz")
    sp.add_argument("take", nargs="?")
    sp.add_argument("--cams", help="comma-separated camera ids (default: every cameras/<cam>.npz)")
    sp.add_argument("--skip-2d", action="store_true", help="reuse mamma/ma_2d_native/<cam>.npz")
    sp.add_argument("--validate-synthetic", action="store_true",
                    help="ground-truth check: synthetic couple seen by moving/zooming cameras (work/synthetic_mamma)")
    sp.add_argument("--noise", type=float, default=0.0, help="pixel noise for --validate-synthetic")
    sp.add_argument("--anatomical", action="store_true",
                    help="--validate-synthetic with the follow's hyperextended left elbow flipped to flexion")
    sp.add_argument("--n-cams", type=int, default=8)
    sp.add_argument("--frames", type=int, default=180)
    sp.set_defaults(fn=cmd_mamma)
    sp = sub.add_parser("timing", help="footstep touchdowns + joint accents vs the zouk beat grid -> timing/timing.json")
    sp.add_argument("take", help="work/<take> with smplx/{lead,follow}.npz (e.g. synthetic, larissa_kadu)")
    sp.add_argument("--beats", help="zouk beats json [[t, type]] (default: timing/beats.json or beats.json in work)")
    sp.add_argument("--allowed-types", default="1,2,3", help="beat types an event may be assigned to")
    sp.add_argument("--positions", help="allowed eighth positions 0-7 inside the measure (default all)")
    sp.add_argument("--tol-ms", type=float, default=50.0, help="hit tolerance")
    sp.add_argument("--mc", type=int, default=200, help="Monte-Carlo draws")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--coarsen", type=int, default=1, help="2 = undo beat_this min_bpm=120 measure halving")
    sp.add_argument("--no-refine", action="store_true", help="keep the beat grid as is (no local tempo fit)")
    sp.add_argument("--av-offset-ms", type=float, default=0.0, help="added to video event times (audio clock)")
    sp.add_argument("--av-sync-event", help="describe the A/V sync event that pins the offset (enables absolute claims)")
    sp.add_argument("--sync-ms", type=float, default=5.0, help="per-camera residual sync uncertainty (SD)")
    sp.add_argument("--n-cams", type=int, default=8)
    sp.add_argument("--noise-mm", type=float, help="joint noise SD (default: estimated)")
    sp.add_argument("--beat-jitter-ms", type=float, help="beat time SD (default: max(refit residual, 10 ms))")
    sp.add_argument("--validate", action="store_true", help="score against truth.json (gate G5) clean + noisy")
    sp.add_argument("--noise-cm", type=float, default=1.0, help="Gaussian joint noise for the noisy validation run")
    sp.set_defaults(fn=cmd_timing)
    sp = sub.add_parser("physics", help="L0 centroidal dynamics: COM, F_net + band, couple GRF, XCoM, beat impulses")
    sp.add_argument("take", help="work/<take> with smplx/{lead,follow}.npz")
    sp.add_argument("--beats", help="zouk beats json (default: timing/beats.json or beats.json in work)")
    sp.add_argument("--mass-lead", type=float, help="kg (default: mesh volume x 1010 kg/m^3, +-10%%)")
    sp.add_argument("--mass-follow", type=float)
    sp.add_argument("--height-lead", type=float, help="m, cross-check against the shaped mesh stature")
    sp.add_argument("--height-follow", type=float)
    sp.add_argument("--cutoff", type=float, default=5.0, help="smoothing spline -6 dB cutoff (Hz)")
    sp.add_argument("--mu", type=float, default=0.6, help="floor friction for the partner-force LP")
    sp.add_argument("--identifiable-n", type=float, default=40.0, help="max partner interval width to call it identifiable")
    sp.add_argument("--boot", type=int, default=200, help="residual bootstrap draws")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--validate", action="store_true", help="synthetic checks + gate G6 (clean and noisy)")
    sp.add_argument("--noise-mm", type=float, default=10.0, help="root translation noise for the noisy validation")
    sp.add_argument("--noise-deg", type=float, default=1.0, help="per-joint rotation noise for the noisy validation")
    sp.set_defaults(fn=cmd_physics)
    sp = sub.add_parser("export", help="smplx + timing + physics -> head-movement StreamingAssets capture (v3)")
    sp.add_argument("take", help="work/<take> with smplx/{lead,follow}.npz")
    sp.add_argument("--capture", help="capture folder name (default: the take name), e.g. 01_SyntheticSMPLX")
    sp.add_argument("--model", help="SMPL-X model npz (default: MAMMA locked-head for MAMMA fits, else smplx/SMPLX_NEUTRAL.npz)")
    sp.add_argument("--audio", help="audio wav (default: work/<take>/audio.wav)")
    sp.add_argument("--audio-t0", type=float, default=0.0, help="reference time (s) of the audio's first sample")
    sp.add_argument("--beats", help="zouk beats json when there is no timing/timing.json")
    sp.add_argument("--no-install", action="store_true", help="only write work/<take>/export/<capture>/")
    sp.set_defaults(fn=cmd_export)
    sp = sub.add_parser("ask",help="natural-language lesson control -> hm_* Unity commands (TypeSafe)")
    sp.add_argument("text")
    sp.add_argument("--run", action="store_true", help="read hm_state and execute in the open Unity editor")
    sp.add_argument("--force", action="store_true", help="execute even when a clarification is suggested")
    sp.add_argument("--debug", action="store_true")
    sp.set_defaults(fn=cmd_ask)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
