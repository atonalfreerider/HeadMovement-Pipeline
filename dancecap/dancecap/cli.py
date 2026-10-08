"""dancecap: multi-phone dance capture -> cameras, room, dancers, poses, physics, timing -> Unity.

    python -m dancecap ingest larissa_kadu            # window frames + room keyframes, native resolution
    python -m dancecap mamma prepare larissa_kadu     # local ma_2d + grid inputs per camera set (+ pod_job.json)
    python -m dancecap mamma fit larissa_kadu --tag c8   # MAMMA ma_3d (Runpod) -> mamma/fits/c8/
    python -m dancecap mamma select larissa_kadu      # pick a fit -> smplx/{lead,follow}.npz
    python -m dancecap sync larissa_kadu              # per-camera video offsets from geometry -> sync/offsets.json
    python -m dancecap detect larissa_kadu            # all-person boxes, COCO-17 kpts, spectator ignore masks
    python -m dancecap dancers-labels larissa_kadu    # couple tracks + lead/follow labels + review sheet
    python -m dancecap dancers larissa_kadu           # SAM 2.1 lead/follow masks, identity repair, QA
    python -m dancecap cameras larissa_kadu           # static-room SfM + per-frame 6DoF/zoom -> cameras/<cam>.npz
    python -m dancecap beats larissa_kadu             # cleanest phone audio on the reference clock + zouk beat grid
    python -m dancecap timing synthetic --validate    # touchdowns/accents vs zouk beats -> timing/timing.json
    python -m dancecap physics synthetic --validate   # L0 centroidal dynamics -> physics/physics.{npz,json}
    python -m dancecap export synthetic --capture 01_SyntheticSMPLX  # -> head-movement StreamingAssets (v3)
    python -m dancecap room larissa_kadu              # masked 3DGS of the studio -> room/ splats + Quest mesh
    python -m dancecap texture larissa_kadu           # photoreal per-dancer SMPL-X albedo (2048^2) -> avatar/
"""
from __future__ import annotations

import argparse
import json
import os

from .take import Take


def cmd_remote(a):
    from . import remote
    if a.action == "status":
        print(json.dumps(remote.status(), indent=1))
    elif a.action == "gpus":
        for r in remote.gpu_table():
            print(f"{r['gpu']:<34} {r['gb']:>4} GB  community ${r['community']}  secure ${r['secure']}  "
                  f"stock {r['stock']}  in {len(r['datacenters'])} DCs")
    elif a.action == "cleanup":
        print("deleted:", remote.cleanup_orphans())
    elif a.action == "smoke":
        import tempfile
        import time as _t
        from pathlib import Path as _P
        with remote.pod_session("smoke", max_minutes=10) as pod:
            pod.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && "
                    "python -c \"import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,"
                    "torch.cuda.is_available(),torch.cuda.get_device_name(0))\" && nproc && free -g | head -2 && df -h / | tail -1")
            with tempfile.TemporaryDirectory() as d:
                blob = _P(d) / "probe.bin"
                blob.write_bytes(os.urandom(50 * 1024 * 1024))
                t0 = _t.time()
                pod.put(blob, "/tmp/probe.bin")
                up = _t.time() - t0
                t0 = _t.time()
                pod.get("/tmp/probe.bin", _P(d) / "back.bin")
                down = _t.time() - t0
                ok = (_P(d) / "back.bin").read_bytes() == blob.read_bytes()
            print(f"[smoke] 50 MB up {50 / up:.1f} MB/s, down {50 / down:.1f} MB/s, round-trip identical={ok}")
        print(json.dumps(remote.status(), indent=1))
    elif a.action in ("jobs", "bundle", "run", "validate"):
        _cmd_remote_jobs(a)


def _cmd_remote_jobs(a):
    from . import remote_jobs as rj
    if a.action == "validate":
        jobs = [rj.validation_job(a.frames)]
    else:
        jobs = rj.resolve_jobs(a.take, a.jobs.split(",") if a.jobs else None)
    if not jobs:
        raise SystemExit("no pod jobs: give --take <take> (work/<take>/*/pod_job.json) and/or --jobs")
    if a.action == "jobs":
        for j in jobs:
            print(f"{j.name:<10} {j.legacy or rj.SCHEMA:<20} max {j.max_minutes:>5.0f} min  needs {j.needs}  "
                  f"{len(j.inputs)} inputs  cmds {[c.name for c in j.commands]}  {j.source}")
        return
    if a.action == "bundle":
        files = rj.bundle_files(jobs)
        path, man = rj.build_bundle(files)
        print(json.dumps({k: man[k] for k in ("path", "bytes", "raw_bytes", "n_files", "links", "groups_raw_bytes",
                                              "build_s")}, indent=1))
        print(json.dumps(rj.estimate_minutes(jobs, man["bytes"]), indent=1))
        return
    res = rj.run_session(jobs, sid=a.session, label=a.action if a.action == "validate" else "batch",
                         max_cost=a.max_cost, gpus=a.gpus.split(",") if a.gpus else None, force=a.force,
                         dry_run=a.dry_run, max_minutes=a.max_minutes)
    print(json.dumps(res, indent=1, default=str))


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
    from .cameras.pipeline import STAGES, KeyframeCacheError, run
    take = Take.load(a.take)
    stages = tuple(a.stages.split(",")) if a.stages else STAGES
    if not set(stages) <= set(STAGES):
        raise SystemExit(f"unknown stages {sorted(set(stages) - set(STAGES))}; choose from {STAGES}")
    try:
        rep = run(take, _cams(take, a.cams), stages, sigma_s=a.sigma, max_reproj_px=a.max_reproj,
                  camera_types=getattr(a, "camera_types", None))
    except KeyframeCacheError as e:  # a locked tripod's keyframes-only localisation cannot carry the new type
        raise SystemExit(f"[cameras] REFUSED: {e}") from None
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
    words = list(a.words)
    action = words.pop(0) if words and words[0] in MAMMA_ACTIONS else "run"
    if not words:
        raise SystemExit("mamma: give [prepare|fit|select|run|bundle] <take>, or --validate-synthetic")
    take = words[0]
    cams = a.cams.split(",") if a.cams else None
    if action == "prepare":
        all_cams = sorted(p.stem for p in (Take.load(take).work / "cameras").glob("*.npz"))
        sets = mamma_dyn.parse_sets(a.set or ([f"run={','.join(cams)}"] if cams else None), all_cams)
        gk = tuple(int(x) for x in a.grid_k.split(":")) if getattr(a, "grid_k", None) else None
        prep = mamma_dyn.prepare(take, sets, force_2d=a.force_2d, skip_2d=a.skip_2d,
                                 allow_sfm_frame=a.allow_sfm_frame, use_sync=not a.no_sync,
                                 coverage=getattr(a, "coverage", None) or "all",
                                 grid_anchor=getattr(a, "grid_anchor", None), grid_k=gk)
        print(json.dumps({"sets": {k: v["cams"] for k, v in prep["sets"].items()}, "eval": prep["eval"],
                          "grid": {k: prep["grid"][k] for k in ("t0", "T", "hz")}, "sync": prep["sync"],
                          "ma_2d": prep["ma_2d"]}, indent=1))
        print(f"[mamma] -> {Take.load(take).work / 'mamma' / 'pod_job.json'}")
    elif action == "fit":
        if not a.tag:
            raise SystemExit("mamma fit: --tag <set> (see mamma/prepare.json)")
        fr = tuple(int(x) for x in a.fit_frames.split(":")) if a.fit_frames else None
        rep = mamma_dyn.fit(take, a.tag, frames=fr)
        print(json.dumps({k: rep[k] for k in ("run_tag", "cams", "ma_3d", "smplx")}, indent=1, default=str))
    elif action == "select":
        sel = mamma_dyn.select(take, tags=a.tags.split(",") if a.tags else None, force_tag=a.force_tag,
                               qa=not a.no_qa)
        print(json.dumps({k: sel.get(k) for k in ("chosen", "reasons", "promoted", "review")}, indent=1))
    elif action == "bundle":
        mamma_dyn.bundle(take)
    else:
        sel = mamma_dyn.run(take, cams=cams, skip_2d=a.skip_2d, sets=a.set, allow_sfm_frame=a.allow_sfm_frame)
        print(json.dumps({k: sel.get(k) for k in ("chosen", "reasons", "promoted")}, indent=1))


MAMMA_ACTIONS = ("prepare", "fit", "select", "run", "bundle")


def cmd_sync(a):
    from . import sync
    doc = sync.run(a.take, cams=a.cams.split(",") if a.cams else None, range_ms=a.range_ms, workers=a.workers,
                   gauge_ref=getattr(a, "gauge_ref", None))
    for c, o in doc["offsets"].items():
        sp = "n/a" if o["spread_ms"] is None else f"{o['spread_ms']:.1f}"
        print(f"[sync] {c}: {o['delta_ms']:+7.1f} ms  {'stable' if o['stable'] else 'UNSTABLE':8s} spread {sp} ms  "
              f"residual {o['residual_px']['at_zero']:.2f} -> {o['residual_px']['at_delta']:.2f} px")


def cmd_beats(a):
    from . import beats
    cfg = beats.BeatsConfig(source=a.source, checkpoint=a.checkpoint, device=a.device, min_bpm=a.min_bpm,
                            out_sr=a.sr)
    res = beats.run(a.take, cfg, plot=not a.no_plot)
    t = res["tempo"]
    print(f"[beats] tempo {t['bpm']:.2f} BPM (quarter notes; linear fit {t['bpm_linear_fit']:.2f}, residual "
          f"{t['linear_fit_resid_rms_ms']:.1f} ms rms), measure {t['measure_s']:.3f} s, {t['n_measures']} measures; "
          f"beat_this tracked {t['beat_level']['tracked_bpm']} BPM -> x{t['beat_level']['factor']:g}")
    s = t["analyzer_split_measures"]
    print(f"[beats] beat_this_analyzer.split_measures(min_bpm=120) would halve {s.get('split', 0)}/"
          f"{s.get('measures', 0)} measures (grid at {s.get('resulting_grid_bpm_median')} 'BPM') - not used")
    for w in t.get("warnings", []):
        print(f"[beats] WARNING: {w}")
    m = res["audio"]
    print(f"[beats] audio cam{m['source_cam']} {m['reference_start']:.4f}..{m['reference_end']:.4f} s @ "
          f"{m['sample_rate']} Hz -> {res['work'] / 'audio' / 'audio_overlap.wav'}")
    print(f"[beats] export with: --audio-t0 {m['reference_start']:.7f}")
    print(f"[beats] -> {res['work'] / 'beats'}" + (f", {', '.join(str(p) for p in res.get('plots', []))}"))


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
        beat_jitter_s=None if a.beat_jitter_ms is None else a.beat_jitter_ms / 1000,
        mc_corr_noise_m=a.mc_corr_noise_mm / 1000, mc_corr_time_s=a.mc_corr_time_s)
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
    try:  # stature check = mesh vs stature + the take's shoe sole (cameras/world_scale convention)
        from .cameras.world import sole_from_take
        from .take import Take
        soles = sole_from_take(Take.load(a.take).cfg)[0]
    except (FileNotFoundError, OSError, KeyError):
        soles = None
    res = physics.run(work, cfg, Path(a.beats) if a.beats else None, masses=pick(a.mass_lead, a.mass_follow),
                      statures=pick(a.height_lead, a.height_follow), soles=soles)
    for role in physics.ROLES:
        i = res[role]["info"]
        print(f"[physics] {role}: {i['mass_kg']:.1f} kg ({i['mass_source']}), stature {i['stature_m']:.2f} m, "
              f"contact {i['contact_fraction']}, F_net band95 median {i['f_net_band95_median_n']} N")
        if i.get("stature_warning"):  # mesh vs stature + sole beyond the world_scale tolerance (1.5 %)
            print(f"[physics] WARNING {role}: mesh {i['stature_rel_diff']:+.1%} vs stature + shoe sole (tol "
                  f"{i['stature_tol']:.1%}): python -m dancecap.cameras.world_scale check {a.take}")
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
                                  beats=Path(a.beats) if a.beats else None, install=not a.no_install,
                                  allow_no_audio=a.no_audio, textures=not a.no_textures, room=not a.no_room,
                                  smplx_dir=a.smplx_dir, shape_edits=not a.no_shape_edits)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"export: {e}")
    for r, d in rep["dancers"].items():
        print(f"[export] {r}: {Path(d['model']).name} ({d['betas']} betas), FK vs npz joints {d['fk_max_err_mm']} mm")
        if d.get("shape_edits"):
            from .shape_edits import summary
            print(f"[export] {r}: shape edits {summary(d['shape_edits'])}")
    print(f"[export] audio: {rep['audio']}")
    if rep.get("room"):
        print(f"[export] room: {rep['room']['triangles']} triangles, {rep['room']['vertices']} vertices")
    print(f"[export] {rep['frames']} frames @ {rep['fps']:.2f} fps, {rep['bytes'] / 1e6:.1f} MB: {', '.join(rep['files'])}")
    print(f"[export] -> {rep.get('installed', rep['out'])}")


def cmd_room(a):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from . import room
    from .env import IS_WINDOWS, cuda_wrapper
    from .take import PACKAGE_ROOT
    # the take's work dir only (no Take.load: on a pod there is no takes/ alignment or source video, just the bundle)
    work = Path(a.take) if Path(a.take).is_dir() else PACKAGE_ROOT / "work" / a.take
    if a.write_pod_job or a.pod:
        cfg = room.RoomConfig()
        room.write_bundle(work, work / "room" / "bundle", cfg)
        if a.pod:
            print(json.dumps(room.run_pod(work, a.preset or "pod", a.pod_minutes), indent=1))
        else:
            room.write_pod_job(work, a.preset or "pod", a.pod_minutes)
        return
    stages = tuple(a.stages.split(",")) if a.stages else room.STAGES
    if not set(stages) <= set(room.STAGES):
        raise SystemExit(f"unknown stages {sorted(set(stages) - set(room.STAGES))}; choose from {room.STAGES}")
    needs_gpu = bool(set(stages) - {"prep"})
    if needs_gpu and IS_WINDOWS and shutil.which("cl") is None and not os.environ.get("DANCECAP_CUDA_ENV"):
        # gsplat's cached JIT extension still needs ninja + MSVC on PATH: re-run inside the CUDA build env
        argv = sys.argv[1:] if a.argv is None else a.argv
        r = subprocess.run(cuda_wrapper(["-m", "dancecap", *argv]), cwd=PACKAGE_ROOT,
                           env={**os.environ, "DANCECAP_CUDA_ENV": "1", "PYTHONUNBUFFERED": "1"})
        raise SystemExit(r.returncode)
    cfg = room.preset(a.preset or "local", iters=a.iters, cap_max=a.cap_max, quest_max=a.quest_max,
                      mesh_max_tris=a.mesh_max_tris, mesh_voxel=a.mesh_voxel, max_side=a.max_side)
    if a.cams:
        cfg.cams = tuple(a.cams.split(","))
    out = Path(a.out).resolve() if a.out else None
    rep = room.run(work, cfg, stages, out=out)
    for g, v in rep.get("gates", {}).items():
        print(f"[room] gate {g}: {'PASS' if v['passed'] else 'FAIL'} {json.dumps(v.get('value'), default=str)}")
    print(f"[room] -> {(out or work / 'room') / 'report.json' if 'gates' in rep else work / 'room' / 'bundle'}")


def cmd_texture(a):
    from types import SimpleNamespace

    from . import avatar_texture
    from .take import PACKAGE_ROOT
    try:
        take = Take.load(a.take)
    except FileNotFoundError:  # on a pod: no takes/ alignment or source videos, only work/<take>
        take = SimpleNamespace(name=a.take, work=PACKAGE_ROOT / "work" / a.take, sources=[])
    if a.write_pod_job or a.pod:
        from . import avatar_pod
        if a.pod:
            avatar_pod.run_pod(take.work, a.preset or "pod", a.pod_minutes, step_ms=a.step_ms, max_cost=a.max_cost,
                               hair_mode=a.hair_mode, blend=a.blend)
        else:
            avatar_pod.write_pod_job(take.work, a.preset or "pod", a.pod_minutes, pack=not a.no_pack,
                                     step_ms=a.step_ms, hair_mode=a.hair_mode, blend=a.blend)
        return
    cfg = avatar_texture.preset_config(a.preset or "local", size=a.size,
                                       step_s=a.step_ms / 1000 if a.step_ms else None, top_k=a.top_k,
                                       device=a.device, head_fix=a.head_fix, align_passes=a.align_passes,
                                       allow_stale=True if a.allow_stale else None, hair_mode=a.hair_mode,
                                       blend=a.blend, body_ref_q=a.body_ref_q,
                                       allow_missing_hair_masks=True if a.allow_missing_hair_masks else None)
    if a.review_only:
        from .avatar_review import render_review
        out = avatar_texture.default_out(take.work, cfg)
        for p in render_review(take, cfg, lock=not a.no_lock, out=out, prefix=avatar_texture.review_prefix(cfg)):
            print(f"[texture] {p}")
        return
    cams = _cams(take, a.cams) if take.sources else (a.cams.split(",") if a.cams else None)
    if cams:
        cfg.cams = tuple(cams)
    if a.exclude_cams:
        known = cfg.cams or tuple(sorted(p.stem for p in (take.work / "cameras").glob("[0-9][0-9].npz")))
        cfg.cams = tuple(c for c in known if c not in a.exclude_cams.split(","))
    for kv in a.cam_weight or []:
        c, w = kv.split("=")
        cfg.cam_weight[c] = float(w)
    rep = avatar_texture.run(take, cfg, lock=not a.no_lock, qa=not a.no_review)
    for r, d in rep["roles"].items():
        print(f"[texture] {r}: measured {d['measured_fraction']:.1%} (face {d['face_measured_fraction']:.1%}), mirrored "
              f"{d['mirror_fraction']:.1%}, diffused {d['diffused_fraction']:.1%}, median inlier cams "
              f"{d['median_inlier_cams']}; weight share {d['camera_weight_share']}")
    for p in rep.get("review", []):
        print(f"[texture] {p}")
    print(f"[texture] -> {avatar_texture.default_out(take.work, cfg)} ({rep['runtime_s']['total']:.0f} s)")


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
    sp = sub.add_parser("remote", help="Runpod offload: status | gpus | cleanup | smoke | jobs | bundle | run | validate "
                                       "(budget-capped, auto-deleting pods; batched pod_job.json sessions)")
    sp.add_argument("action", choices=["status", "gpus", "cleanup", "smoke", "jobs", "bundle", "run", "validate"])
    sp.add_argument("--take", help="jobs/bundle/run: discover work/<take>/*/pod_job.json")
    sp.add_argument("--jobs", help="comma-separated stage names (mamma,room,avatar) or pod_job.json paths")
    sp.add_argument("--session", help="run/validate: resume this session id (work/runpod/sessions/<sid>)")
    sp.add_argument("--max-cost", type=float, help="run/validate: per-session worst-case $ ceiling (price x max minutes)")
    sp.add_argument("--max-minutes", type=float, help="override the pod lifetime (kill switch); default = estimate")
    sp.add_argument("--gpus", help="comma-separated Runpod GPU ids to try (default remote.GPU_PREFERENCE)")
    sp.add_argument("--force", action="store_true", help="re-run jobs already done in this session")
    sp.add_argument("--dry-run", action="store_true", help="build the bundle + estimate, no pod")
    sp.add_argument("--frames", type=int, default=180, help="validate: synthetic MAMMA grid frames")
    sp.set_defaults(fn=cmd_remote)
    sp = sub.add_parser("cameras", help="static-room SfM (ALIKED+LightGlue, COLMAP) + per-frame pose/zoom of every camera")
    sp.add_argument("take")
    sp.add_argument("--cams")
    sp.add_argument("--stages", help="comma-separated subset of map,loc,attach,world,qa (default: all; later stages reuse caches)")
    sp.add_argument("--sigma", type=float, default=0.04, help="temporal smoothing sigma (s) for pose and log-focal")
    sp.add_argument("--max-reproj", type=float, default=4.0, help="frames above this median reprojection (px) -> ok=False")
    sp.add_argument("--camera-types", choices=["user", "auto", "off"],
                    help="per-video camera types (takes/<take>.toml [cameras.\"NN\"], dancecap.camera_types): user = "
                         "confirmed answers only (default, or $DANCECAP_CAMERA_TYPES), auto = + suggestions with "
                         "confidence >= 0.7, off = handheld + zoom for every camera (the solver before camera types)")
    sp.set_defaults(fn=cmd_cameras)
    sp = sub.add_parser("mamma", help="MAMMA with moving/zooming cameras: prepare (local ma_2d + inputs) | fit (ma_3d, "
                                      "pod) | select (-> smplx/) | run (all) | bundle (pod upload)")
    sp.add_argument("words", nargs="*", help="[prepare|fit|select|run|bundle] <take>  (no action = run)")
    sp.add_argument("--cams", help="comma-separated camera ids (default: every cameras/<cam>.npz)")
    sp.add_argument("--set", action="append", help="prepare/run: camera set tag=01,02,.. | tag=all | tag=-05 "
                                                   "(repeatable; default one set 'all')")
    sp.add_argument("--tag", help="fit: which prepared camera set")
    sp.add_argument("--tags", help="select: restrict to these fit tags")
    sp.add_argument("--force-tag", help="select: take this fit regardless of the residual rule")
    sp.add_argument("--fit-frames", help="fit: only grid frames A:B (dry check; never selectable)")
    sp.add_argument("--no-qa", action="store_true", help="select: skip review images (pod has no frames)")
    sp.add_argument("--skip-2d", action="store_true", help="reuse mamma/ma_2d_native/<cam>.npz as is")
    sp.add_argument("--force-2d", action="store_true", help="recompute ma_2d even when its manifest matches")
    sp.add_argument("--no-sync", action="store_true", help="ignore sync/offsets.json")
    sp.add_argument("--coverage", choices=["all", "union"],
                    help="prepare: grid frames every camera covers (all, default) or any camera covers (union; each "
                         "camera contributes only where it has frames)")
    sp.add_argument("--grid-anchor", type=float, help="prepare: put the 30 Hz grid on t_k = anchor + k/30")
    sp.add_argument("--grid-k", help="prepare (with --grid-anchor): keep lattice indices k0:k1 (inclusive)")
    sp.add_argument("--allow-sfm-frame", action="store_true",
                    help="run without cameras/world_from_sfm.json (result never promoted to smplx/)")
    sp.add_argument("--validate-synthetic", action="store_true",
                    help="ground-truth check: synthetic couple seen by moving/zooming cameras (work/synthetic_mamma)")
    sp.add_argument("--noise", type=float, default=0.0, help="pixel noise for --validate-synthetic")
    sp.add_argument("--anatomical", action="store_true",
                    help="--validate-synthetic with the follow's hyperextended left elbow flipped to flexion")
    sp.add_argument("--n-cams", type=int, default=8)
    sp.add_argument("--frames", type=int, default=180)
    sp.set_defaults(fn=cmd_mamma)
    sp = sub.add_parser("sync", help="per-camera residual video offsets from multi-view geometry -> sync/offsets.json")
    sp.add_argument("take")
    sp.add_argument("--cams")
    sp.add_argument("--range-ms", type=float, default=300.0)
    sp.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: 1 off-pod, the container's CPU quota up to 16 on a pod)")
    sp.add_argument("--gauge-ref", help="pin the gauge to this take's sync/offsets.json (same recordings and audio "
                                        "clocks): median over shared cameras of (delta - delta_ref) = 0")
    sp.set_defaults(fn=cmd_sync)
    sp = sub.add_parser("beats", help="cleanest phone audio on the reference clock + beat_this zouk grid, tempo, clicks")
    sp.add_argument("take")
    sp.add_argument("--source", help="force the source camera id (default: best music quality score)")
    sp.add_argument("--checkpoint", default="final0", help="beat_this checkpoint (default final0)")
    sp.add_argument("--device", help="cuda|cpu (default: cuda when the GPU lock is free, else cpu)")
    sp.add_argument("--min-bpm", type=float, default=60.0, help="quarter-note tempo folded into [min, 2*min)")
    sp.add_argument("--sr", type=int, help="output wav sample rate (default: the source's native rate)")
    sp.add_argument("--no-plot", action="store_true")
    sp.set_defaults(fn=cmd_beats)
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
    sp.add_argument("--mc-corr-noise-mm", type=float, default=0.0, help="MC: low-frequency (correlated) pose error SD, "
                    "e.g. 10 for real fused fits (analysis review convention); 0 = white noise only")
    sp.add_argument("--mc-corr-time-s", type=float, default=0.1, help="MC: correlation time of that error")
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
    sp.add_argument("--audio", help="audio wav (default: work/<take>/audio/audio_overlap.wav, else a synthetic "
                                    "take's audio.wav)")
    sp.add_argument("--audio-t0", type=float, default=None,
                    help="reference time (s) of the audio's first sample (required with --audio; default for "
                         "audio/audio_overlap.wav: its json's reference_start)")
    sp.add_argument("--no-audio", action="store_true", help="allow a capture without audio (Unity plays a silent clock)")
    sp.add_argument("--no-textures", action="store_true", help="untextured skins (no UV split, no albedo)")
    sp.add_argument("--no-room", action="store_true", help="skip room_mesh.bin / splat_quest.ply")
    sp.add_argument("--beats", help="zouk beats json when there is no timing/timing.json")
    sp.add_argument("--no-install", action="store_true", help="only write work/<take>/export/<capture>/")
    sp.add_argument("--smplx-dir", default=None, help="contract folder under work/<take> to export (e.g. a "
                                                      "dancecap.pose_refine output); default: takes/<take>.toml "
                                                      "[export] smplx_dir, else smplx/ (--smplx-dir smplx = the raw fit)")
    sp.add_argument("--no-shape-edits", action="store_true", help="ignore takes/<take>.toml [dancers.<role>.shape_edits] "
                                                                  "(export the plain fitted rest shapes)")
    sp.set_defaults(fn=cmd_export)
    sp = sub.add_parser("room", help="static studio: masked 3DGS (gsplat) -> splat_full/splat_quest.ply + Quest mesh")
    sp.add_argument("take", help="take name (work/<take>) or a work directory")
    sp.add_argument("--cams", help="comma-separated cameras (default: the static-model cameras 01,02,03,04,07,09); "
                                   "a prep setting")
    sp.add_argument("--stages", help="comma-separated subset of prep,train,quest,mesh,eval (default all; prep is local "
                                     "only and skipped when the bundle is up to date; later stages reuse room/cache)")
    sp.add_argument("--preset", choices=["smoke", "local", "pod"],
                    help="quality preset: smoke (code path, minutes), local (2080 Ti, default), pod (80 GB, full)")
    sp.add_argument("--out", help="output dir (default work/<take>/room; review sheets go to <out>/review when set)")
    sp.add_argument("--pod", action="store_true", help="run train/quest/mesh/eval on a Runpod 80 GB GPU (prep + "
                                                       "upload here, download + extract into work/<take>)")
    sp.add_argument("--pod-minutes", dest="pod_minutes", type=float, default=110,
                    help="pod lifetime cap incl. setup (on-pod kill switch; budget check)")
    sp.add_argument("--write-pod-job", dest="write_pod_job", action="store_true",
                    help="only prep + pack the pod inputs and write room/pod_job.json")
    sp.add_argument("--iters", "--max-steps", dest="iters", type=int, help="training steps (preset: smoke 300, else 30000)")
    sp.add_argument("--cap-max", dest="cap_max", type=int, help="MCMC Gaussian cap (default 1M)")
    sp.add_argument("--quest-max", dest="quest_max", type=int, help="Quest splat budget (default 150k)")
    sp.add_argument("--mesh-max-tris", dest="mesh_max_tris", type=int, help="Quest mesh budget (default 80k)")
    sp.add_argument("--mesh-voxel", dest="mesh_voxel", type=float, help="TSDF voxel (m, default 0.03)")
    sp.add_argument("--max-side", dest="max_side", type=int, help="training image long side (default 1000)")
    sp.set_defaults(fn=cmd_room, argv=None)
    sp = sub.add_parser("texture", help="photoreal per-dancer albedo on the SMPL-X UV layout -> avatar/ + review sheets")
    sp.add_argument("take")
    sp.add_argument("--cams", help="comma-separated cameras (default: every cameras/<cam>.npz)")
    sp.add_argument("--exclude-cams", help="comma-separated cameras to leave out (e.g. a hold-out check)")
    sp.add_argument("--cam-weight", action="append", help="CAM=W trust multiplier (default 05=0.25), repeatable")
    sp.add_argument("--preset", choices=("smoke", "local", "pod"),
                    help="smoke (512^2, 1.5 s, -> avatar/smoke/) | local (2080 Ti, default) | pod (80 GB GPU)")
    sp.add_argument("--size", type=int, help="atlas size (default 2048, Quest budget)")
    sp.add_argument("--step-ms", type=float, help="sample each camera every N ms of reference time (local 66.7)")
    sp.add_argument("--top-k", type=int, help="samples kept per texel per camera (local 4)")
    sp.add_argument("--device", help="cuda|cpu (default cuda)")
    sp.add_argument("--head-fix", choices=("auto", "on", "off"),
                    help="neck correction to the COCO face keypoints (auto = adopt when leave-one-camera-out improves)")
    sp.add_argument("--align-passes", type=int, help="flow-registration passes after the first bake (local 1)")
    sp.add_argument("--hair-mode", choices=("exclude", "soft"),
                    help="follow hair: exclude (default: hair mask + colour hard-excluded, learned scalp, class fill; "
                         "avatar_hair.py) | soft (old colour down-weighting)")
    sp.add_argument("--blend", choices=("multiband", "mean"),
                    help="multiband (default: two-band per-camera fusion, no view seams) | mean (old)")
    sp.add_argument("--allow-stale", action="store_true",
                    help="bake even if smplx/ predates cameras/world_from_sfm.json (fit from an older world frame)")
    sp.add_argument("--write-pod-job", action="store_true",
                    help="write avatar/pod_job.json + upload tars in avatar/pod/ (no pod is started)")
    sp.add_argument("--no-pack", action="store_true", help="with --write-pod-job: plan only, do not build the tars")
    sp.add_argument("--pod", action="store_true", help="run the bake on a budget-capped Runpod 80 GB GPU")
    sp.add_argument("--pod-minutes", type=float, help="pod auto-delete after N minutes (default: the job's "
                    "max_minutes_recommended; a smaller value is refused)")
    sp.add_argument("--max-cost", type=float, default=10.0, help="with --pod: session cost ceiling in $ (default 10)")
    sp.add_argument("--allow-missing-hair-masks", action="store_true",
                    help="hair-mode exclude: bake even when < 90%% of the follow's views have a real hair mask")
    sp.add_argument("--body-ref-q", type=float, default=None,
                    help="hair roles' upper body / neck texels under her hair: lighter luminance quantile of the "
                         "fusion's inlier reference (0.5 = off, default; e.g. 0.7) - keeps clean samples over hair")
    sp.add_argument("--no-lock", action="store_true", help="do not take work/gpu.lock")
    sp.add_argument("--no-review", action="store_true", help="skip the review renders")
    sp.add_argument("--review-only", action="store_true", help="only re-render the review sheets from avatar/")
    sp.set_defaults(fn=cmd_texture)
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
