"""atlas-bridge: phone video -> sync -> Atlas virtual cameras -> MAMMA -> head-movement.

    python -m atlas_bridge init sessions/spin --videos a.mp4 b.mp4
    python -m atlas_bridge sync sessions/spin
    python -m atlas_bridge rig sessions/spin
    python -m atlas_bridge atlas-request sessions/spin     # phase 1: reconstruct; phase 2: render views
    python -m atlas_bridge align sessions/spin             # after environment.ply arrives
    python -m atlas_bridge atlas-fetch sessions/spin
    python -m atlas_bridge mamma sessions/spin [--prep-only]
    python -m atlas_bridge check-scale sessions/spin
    python -m atlas_bridge beats sessions/spin
    python -m atlas_bridge export sessions/spin
    python -m atlas_bridge status sessions/spin
    python -m atlas_bridge demo                             # synthetic capture into StreamingAssets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import align as align_mod
from . import beats as beats_mod
from . import cameras as cam_mod
from . import mamma_bridge
from . import sync as sync_mod
from .atlas import AtlasJob, make_provider
from .session import TEMPLATE, Session
from .unity_export import assign_roles, export_capture, smplx_to_smpl24, standing_height

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HeadMovement


def _rel(p: Path, base: Path) -> str:
    try:
        return Path(__import__("os").path.relpath(p, base)).as_posix()
    except ValueError:
        return p.as_posix()


# ---------------------------------------------------------------------------

def cmd_init(a):
    root = Path(a.session).resolve()
    root.mkdir(parents=True, exist_ok=True)
    toml = root / "session.toml"
    if toml.exists() and not a.force:
        sys.exit(f"{toml} exists (use --force)")
    name = root.name
    videos = ", ".join(f'"{_rel(Path(v).resolve(), root)}"' for v in a.videos)
    beat_py = Path.home() / "miniconda3" / "envs" / "beat_this" / "python.exe"
    toml.write_text(TEMPLATE.format(
        name=name, videos=videos,
        mamma_root=_rel(WORKSPACE / "mamma", root),
        beat_this_root=_rel(WORKSPACE / "beat_this", root),
        beat_this_python=beat_py.as_posix(),
        streaming_assets=_rel(WORKSPACE / "head-movement" / "Assets" / "StreamingAssets", root),
        capture_name=a.capture_name or "".join(w.capitalize() for w in name.replace("_", "-").split("-")),
    ))
    if a.music:
        text = toml.read_text().replace('music = ""', f'music = "{_rel(Path(a.music).resolve(), root)}"', 1)
        toml.write_text(text)
    print(f"wrote {toml}")


def cmd_sync(a):
    s = Session.load(Path(a.session))
    videos = s.videos()
    if not videos:
        sys.exit("no [input] videos")
    music = s.section("input").get("music") or None
    music = s.path(music) if music else None
    reference = s.section("input").get("reference") or None
    manual = s.section("sync").get("offsets", {})

    probes = {}
    for name, p in videos.items():
        info = probes[name] = sync_mod.probe(p)
        print(f"[sync] {name}: {info['width']}x{info['height']} {info['fps']:.3f} fps "
              f"{'VFR ' if info['vfr'] else ''}{info['duration']:.2f}s audio={info['has_audio']}")
    rates = {round(i["fps"], 3) for i in probes.values()}
    if len(rates) > 1:
        print(f"[sync] NOTE mixed frame rates {sorted(rates)}: sources get resampled to one rate")

    if len(videos) == 1 and music is None:
        name, p = next(iter(videos.items()))
        ref, syncs = name, [sync_mod.SourceSync(name, str(p), 0.0, sync_mod.probe(p)["duration"], float("inf"))]
    else:
        ref, syncs = sync_mod.sync_sources(videos, music, reference)
    for sy in syncs:
        if sy.name in manual:
            sy.offset, sy.confidence = float(manual[sy.name]), float("inf")
        flag = "" if sy.confidence > 5 else "  <-- LOW CONFIDENCE, check or set [sync.offsets]"
        print(f"[sync] {sy.name}: offset {sy.offset:+.4f}s  confidence {sy.confidence:.1f}{flag}")

    start, end = sync_mod.common_window(syncs, set(videos))
    ref_video = ref if ref in probes else next(iter(probes))
    fps = s.fps(probes[ref_video]["fps"])
    frames = int(np.floor((end - start) * fps))
    duration = frames / fps
    print(f"[sync] common window {start:.3f}-{end:.3f}s on '{ref}' timeline -> {frames} frames @ {fps:g} fps")

    out = s.work / "synced"
    by_name = {sy.name: sy for sy in syncs}
    for name, p in videos.items():
        sync_mod.cut_constant_rate(p, out / f"{name}.mp4", start - by_name[name].offset, duration, fps)
    ref_src = music if music is not None else videos[ref]
    ref_offset = 0.0 if music is not None else by_name[ref].offset
    sync_mod.extract_wav(ref_src, s.work / "audio.wav", start - ref_offset, duration)
    sync_mod.write_sync(s.work / "sync.json", ref, syncs, (start, end), fps, frames)
    print(f"[sync] wrote {out} and {s.work / 'audio.wav'}")


def _sync_info(s: Session) -> dict:
    p = s.work / "sync.json"
    if not p.exists():
        sys.exit("run `sync` first")
    return json.loads(p.read_text())


def cmd_rig(a):
    s = Session.load(Path(a.session))
    cams = cam_mod.build_rig(s.section("rig"))
    (s.work / "rig").mkdir(parents=True, exist_ok=True)
    cam_mod.save_cameras_npz(s.work / "rig" / "cameras.npz", cams)
    (s.work / "rig" / "calib_preview.yaml").write_text(cam_mod.to_mamma_yaml(cams))
    cam_mod.write_json(s.work / "rig" / "cameras_unity.json", cam_mod.to_unity_json(cams))
    for c in cams:
        print(f"[rig] {c.name}: C=({c.C[0]:+.2f},{c.C[1]:+.2f},{c.C[2]:.2f}) hfov {c.hfov_deg:.0f}")
    for w in cam_mod.check_coverage(cams, stage_radius=float(s.section("rig").get("dance_radius", 1.2))):
        print(f"[rig] WARNING {w}")
    print(f"[rig] {len(cams)} virtual cameras -> {s.work / 'rig'}")


def _stage_to_scene(s: Session) -> tuple[np.ndarray, str]:
    cfg = s.section("atlas")
    explicit = cfg.get("scene_to_stage", "auto")
    if explicit != "auto":
        return np.linalg.inv(np.array(explicit, dtype=float).reshape(4, 4)), "session.toml"
    p = s.work / "atlas" / "scene_to_stage.json"
    if p.exists():
        return np.linalg.inv(np.array(json.loads(p.read_text())["matrix"])), "align"
    scale = float(cfg.get("scale", 1.0))
    return np.diag([1 / scale] * 3 + [1.0]), "UNALIGNED (stage == scene)"


def _job(s: Session) -> AtlasJob:
    info = _sync_info(s)
    rig = s.work / "rig" / "cameras.npz"
    if not rig.exists():
        sys.exit("run `rig` first")
    stage_to_scene, _ = _stage_to_scene(s)
    return AtlasJob(
        session=s.name,
        source_videos={n: s.work / "synced" / f"{n}.mp4" for n in s.videos()},
        fps=info["fps"], frame_count=info["frame_count"],
        virtual_cameras=cam_mod.load_cameras_npz(rig),
        stage_to_scene=stage_to_scene,
    )


def cmd_atlas_request(a):
    s = Session.load(Path(a.session))
    cfg = s.section("atlas")
    provider = make_provider(cfg.get("provider", "manual"), s.work, cfg)
    _, how = _stage_to_scene(s)
    job_id = provider.submit(_job(s))
    print(f"[atlas] request written (job {job_id}); camera alignment: {how}")
    if how.startswith("UNALIGNED"):
        print("[atlas] PHASE 1: reconstruct the scene in Atlas from work/atlas/request/inputs, export\n"
              "        environment.ply (+ source_cameras.json if available) into work/atlas/renders/,\n"
              "        then run `align` and `atlas-request` again for the view renders.")


def cmd_align(a):
    s = Session.load(Path(a.session))
    cfg = s.section("atlas")
    rd = s.work / "atlas" / "renders"
    env = rd / "environment.ply"
    if not env.exists():
        sys.exit(f"need {env} (Atlas splat export) to align")
    src = {}
    if (rd / "source_cameras.json").exists():
        src = {k: np.array(v) for k, v in json.loads((rd / "source_cameras.json").read_text()).items()}
    ref = s.section("input").get("reference") or next(iter(s.videos()), None)
    M, notes = align_mod.auto_scene_to_stage(env, src, float(cfg.get("scale", 1.0)), ref,
                                             dancers_scene=cfg.get("dancers_scene"))
    for n_ in notes:
        print(f"[align] {n_}")
    cam_mod.write_json(s.work / "atlas" / "scene_to_stage.json", {"matrix": M.tolist(), "notes": notes})
    print("[align] wrote work/atlas/scene_to_stage.json - rerun `atlas-request` to regenerate camera poses")


def cmd_atlas_fetch(a):
    s = Session.load(Path(a.session))
    cfg = s.section("atlas")
    provider = make_provider(cfg.get("provider", "manual"), s.work, cfg)
    res = provider.fetch(_job(s), "manual")
    if res is None:
        sys.exit(2)
    print(f"[atlas] {len(res.views)} views OK; environment splat: {bool(res.environment_splat)}; "
          f"splat sequence: {bool(res.splat_sequence_dir)}")


def cmd_mamma(a):
    s = Session.load(Path(a.session))
    cfg = s.section("mamma")
    root = s.path(cfg.get("root", "../../mamma"))
    job = _job(s)
    provider = make_provider(s.section("atlas").get("provider", "manual"), s.work, s.section("atlas"))
    res = provider.fetch(job, "manual")
    if res is None:
        sys.exit("Atlas renders incomplete")
    capture = mamma_bridge.write_capture(root, s.name, res.views, job.virtual_cameras, job.fps)
    cmd = mamma_bridge.run_command(root, capture, cfg.get("preset"), cfg.get("out_tag", "atlas"),
                                   cfg.get("conda_env", "mamma"))
    print(f"[mamma] capture: {capture}")
    if a.prep_only:
        print("[mamma] run from the mamma repo:\n  " + " ".join(cmd))
        return
    mamma_bridge.run(root, capture, cfg.get("preset"), cfg.get("out_tag", "atlas"), cfg.get("conda_env", "mamma"))


def _people(s: Session) -> dict[int, dict]:
    cfg = s.section("mamma")
    return mamma_bridge.load_people(s.path(cfg.get("root", "../../mamma")), s.name, cfg.get("out_tag", "atlas"))


def cmd_check_scale(a):
    s = Session.load(Path(a.session))
    roles = s.section("roles")
    people = {pid: smplx_to_smpl24(p["joints"]) for pid, p in _people(s).items()}
    lead, follow, note = assign_roles(people, roles.get("lead", "auto"), roles.get("follow", "auto"))
    print(f"[scale] {note}")
    ratios = []
    for pid, key in ((lead, "lead_height"), (follow, "follow_height")):
        h = standing_height(people[pid])
        # SMPL head joint sits ~0.10-0.12 m below the crown
        est = h + 0.11
        msg = f"[scale] id {pid}: head joint {h:.2f} m -> stature ~{est:.2f} m"
        if key in roles:
            ratios.append(float(roles[key]) / est)
            msg += f" (expected {roles[key]:.2f})"
        print(msg)
    if ratios:
        cur = float(s.section("atlas").get("scale", 1.0))
        print(f"[scale] suggested [atlas] scale = {cur * float(np.mean(ratios)):.4f} (current {cur}); "
              "rerun align/atlas-request/mamma if it moves more than ~3%")


def cmd_beats(a):
    s = Session.load(Path(a.session))
    cfg = s.section("beats")
    audio = s.work / "audio.wav"
    out = s.work / "zouk-time-analysis.json"
    py = cfg.get("python", "")
    if py and Path(py).exists():
        beats_mod.run_beat_this(audio, s.path(cfg["beat_this_root"]), py, out)
        print(f"[beats] beat_this -> {out}")
    elif "bpm" in cfg:
        info = _sync_info(s)
        dur = info["frame_count"] / info["fps"]
        beats_mod.write_beats(out, beats_mod.zouk_grid(float(cfg["bpm"]), dur, float(cfg.get("first_downbeat", 0))))
        print(f"[beats] beat_this unavailable; steady {cfg['bpm']} BPM grid -> {out}")
    else:
        sys.exit(f"beat_this python not found at {py!r}; create the env (beat_this/environment.yml) or set [beats] bpm")


def cmd_export(a):
    s = Session.load(Path(a.session))
    info = _sync_info(s)
    roles = s.section("roles")
    raw = _people(s)
    people = {pid: smplx_to_smpl24(p["joints"]) for pid, p in raw.items()}
    lead, follow, note = assign_roles(people, roles.get("lead", "auto"), roles.get("follow", "auto"))
    print(f"[export] {note}")

    rd = s.work / "atlas" / "renders"
    env = rd / "environment.ply"
    seq = rd / "splats"
    beats = s.work / "zouk-time-analysis.json"
    cams_unity = s.work / "rig" / "cameras_unity.json"
    stage_to_scene, how = _stage_to_scene(s)
    smplx = None
    if raw[lead]["smplx"] is not None and raw[follow]["smplx"] is not None:
        smplx = {f"lead_{k}": v for k, v in raw[lead]["smplx"].items()}
        smplx.update({f"follow_{k}": v for k, v in raw[follow]["smplx"].items()})
    uni = s.section("unity")
    out = s.path(uni["streaming_assets"]) / uni["capture_name"]
    export_capture(
        out, people[lead], people[follow], info["fps"],
        audio=s.work / "audio.wav", audio_offset=0.0,
        beats=beats if beats.exists() else None,
        cameras_unity=json.loads(cams_unity.read_text()) if cams_unity.exists() else None,
        environment_splat=env if env.exists() else None,
        splat_sequence_dir=seq if seq.is_dir() and any(seq.glob("*.ply")) else None,
        scene_to_stage=np.linalg.inv(stage_to_scene),
        smplx=smplx,
        extra_meta={"session": s.name, "roles": note, "alignment": how, "mamma_ids": [lead, follow]},
    )
    if not beats.exists():
        print("[export] WARNING no beats - run `beats` then export again")
    print(f"[export] -> {out}")


def cmd_status(a):
    s = Session.load(Path(a.session))
    w = s.work
    rd = w / "atlas" / "renders"
    checks = [
        ("sync", (w / "sync.json").exists()),
        ("rig", (w / "rig" / "cameras.npz").exists()),
        ("atlas request", (w / "atlas" / "request" / "REQUEST.md").exists()),
        ("atlas environment.ply", (rd / "environment.ply").exists()),
        ("align", (w / "atlas" / "scene_to_stage.json").exists()),
        ("atlas renders", (w / "rig" / "cameras.npz").exists() and all(
            (rd / f"{c.name}.mp4").exists() for c in cam_mod.load_cameras_npz(w / "rig" / "cameras.npz"))),
        ("beats", (w / "zouk-time-analysis.json").exists()),
    ]
    try:
        _people(s)
        checks.append(("mamma output", True))
    except FileNotFoundError:
        checks.append(("mamma output", False))
    uni = s.section("unity")
    checks.append(("unity export", (s.path(uni["streaming_assets"]) / uni["capture_name"] / "capture.json").exists()))
    for name, ok in checks:
        print(f"  [{'x' if ok else ' '}] {name}")


def cmd_demo(a):
    from .demo import make_demo
    out = Path(a.out) if a.out else WORKSPACE / "head-movement" / "Assets" / "StreamingAssets" / "00_SyntheticDemo"
    model = Path(a.smplx) if a.smplx else WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"
    body_py = a.body_model_python or str(Path.home() / "miniconda3" / "envs" / "mamma" / "python.exe")
    make_demo(out, model, body_py, seconds=a.seconds)
    print(f"[demo] -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="atlas_bridge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        sp = sub.add_parser(name, **kw)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("init", cmd_init)
    sp.add_argument("session")
    sp.add_argument("--videos", nargs="+", required=True)
    sp.add_argument("--music")
    sp.add_argument("--capture-name")
    sp.add_argument("--force", action="store_true")
    for name, fn in (("sync", cmd_sync), ("rig", cmd_rig), ("atlas-request", cmd_atlas_request),
                     ("align", cmd_align), ("atlas-fetch", cmd_atlas_fetch), ("check-scale", cmd_check_scale),
                     ("beats", cmd_beats), ("export", cmd_export), ("status", cmd_status)):
        add(name, fn).add_argument("session")
    sp = add("mamma", cmd_mamma)
    sp.add_argument("session")
    sp.add_argument("--prep-only", action="store_true", help="write the MAMMA capture and print the command")
    sp = add("demo", cmd_demo)
    sp.add_argument("--out")
    sp.add_argument("--smplx", help="path to SMPLX_NEUTRAL.npz")
    sp.add_argument("--body-model-python", help="python with smplx+torch (default: the mamma conda env)")
    sp.add_argument("--seconds", type=float, default=24.0)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
