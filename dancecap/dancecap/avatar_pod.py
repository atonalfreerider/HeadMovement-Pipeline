"""Runpod offload of the avatar texture bake (avatar_texture.py, preset "pod").

The bake needs the posed SMPL-X fits (smplx/), so it runs AFTER the MAMMA refit. Two ways to run it:

  chain_after_mamma   in the SAME pod session as the MAMMA fit, right after `mamma select` promoted smplx/ there:
                      the pod already has the workspace layout, smplx/ and the SMPL-X body model, so only
                      texture_inputs.tar.gz (code + the frames/masks/cameras the bake reads) is uploaded.
  standalone          its own pod (python -m dancecap texture <take> --pod) once smplx/ is back locally: uploads
                      texture_inputs.tar.gz + smplx.tar.gz + the body model (SMPLX_NEUTRAL.npz, 109 MB).

Uploads are kept small (home upload ~2.5 MB/s): only the frames the bake samples (every step_s slot of absolute
reference time over the whole window: a superset of what the pod run picks for any smplx window) and the frames the
hair-colour model reads, the dancer masks of those frames, cameras/<cam>.npz (never the 3.4 GB cameras caches),
people/<cam>.npz (head fix), roles.json, frames.json. Everything the bake writes comes back in texture_out.tar.gz.

    python -m dancecap texture larissa_kadu --write-pod-job [--step-ms 66.7]   # avatar/pod_job.json + avatar/pod/*.tar.gz
    python -m dancecap texture larissa_kadu --pod [--step-ms 66.7] [--pod-minutes N] [--max-cost 10]
                                                                  # standalone pod session (budget-capped)

--pod-minutes defaults to the job's max_minutes_recommended (estimate x 1.3 + 3 min); an explicit value below it is
refused (the on-pod kill timer would delete the pod mid-bake: money spent, no output). --max-cost caps the session
(default $10) on top of the global $30 cap.

Likeness + SMPL-X data (non-commercial): the pod is deleted at the end of the session (remote.pod_session).
"""
from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path

import numpy as np

from .avatar_texture import PIDS, PRESETS, ROLES, TextureConfig, _view_list
from .take import PACKAGE_ROOT

POD_WS = "/workspace/HeadMovement"  # same layout as the MAMMA pod job (mamma_dyn.write_pod_job)
POD_PRICE_HINT = 1.39  # $/h A100 80 GB community (2026-10-06); pod_session checks the live price + budget
OUTPUTS = ["avatar/lead_albedo.png", "avatar/follow_albedo.png", "avatar/lead_coverage.png",
           "avatar/follow_coverage.png", "avatar/lead_texels.npz", "avatar/follow_texels.npz", "avatar/uv_mesh.npz",
           "avatar/head_fix.npz", "avatar/report.json", "avatar/pod_texture.log", "review/avatar_texture_lead.jpg",
           "review/avatar_texture_follow.jpg", "review/avatar_uv.jpg", "review/avatar_turntable.jpg",
           "review/avatar_face.jpg",
           # hair-aware bake (avatar_hair.py): learned scalp region + hair colour sample for the Unity hair
           "avatar/follow_hair_region.png", "avatar/follow_hair.json", "avatar/follow_hair_strip.png",
           "avatar/follow_hair_cover.png"]
# measured on the A100 pod (2026-10-06, larissa_kadu, 1399 views at 2048^2, 176 grid frames) with the OLD code path
# (hair_mode soft, blend mean): pass 0 0.031 s/view, each registration pass 0.089 s/view, head fix (leave-one-camera-
# out, scales with the frames) + hair models + fills + review ~4 min -> modelled as 2 min + 0.012 min per 30 Hz frame.
# The hair-aware path (exclude + multiband) was RE-MEASURED on the pod (2026-10-07, A100-SXM4-80GB, larissa_kadu 352
# views at 2048^2, texture-fix validation): 21.3 s pass 0 + 40.2 s aligned pass with 12 loader threads (the pod
# preset's load_threads) -> 0.061 / 0.114 s/view (per-pass fusion + fills included, so slightly high for long takes).
# With the old 4 threads it took 73.9 / 112.2 s on an A100 PCIe pod: the new path is CPU-bound in view loading.
POD_S_PER_VIEW_OLD = {"pass0": 0.031, "aligned": 0.089}
POD_S_PER_VIEW = {"pass0": 0.061, "aligned": 0.114}
POD_RATE_BASIS = ("A100-SXM4 2026-10-07 hair-aware path re-measured (352 views, 12 loader threads): 0.061 s/view "
                  "pass 0, 0.114 s/view per registration pass")
POD_FIXED_MIN = 2.0
POD_MIN_PER_FRAME = 0.012
DEFAULT_MAX_COST = 10.0  # $ per standalone texture session (the global cap still applies)


def per_view_rates(code_path: dict) -> dict:
    """s/view used by the estimate for this code path (the old rates for hair_mode soft + blend mean)."""
    old = code_path.get("hair_mode") == "soft" and code_path.get("blend") == "mean"
    return dict(POD_S_PER_VIEW_OLD) if old else dict(POD_S_PER_VIEW)


def needed_frames(work: Path, cfg: TextureConfig, cams: list[str]) -> dict[str, dict]:
    """Per camera: frame indices the bake samples (any smplx window) and those the hair-colour model reads."""
    out = {}
    for c in cams:
        meta = json.loads((work / "frames" / c / "frames.json").read_text())
        views = {i for cc, i, _ in _view_list(work, [c], -1e9, 1e9, cfg.step_s) if cc == c}
        hair = set(range(0, len(meta["frames"]), cfg.hair_frame_step)) if cfg.hair_roles else set()
        out[c] = {"views": sorted(views), "hair": sorted(hair)}
    return out


def input_items(work: Path, cfg: TextureConfig, cams: list[str]) -> list[tuple[Path, str]]:
    """(local file, archive path relative to the pod workspace) of everything the pod bake reads, minus smplx/ and
    the body model (those come from the MAMMA session or the standalone extras)."""
    take = work.name
    arc = lambda p: f"dancecap/work/{take}/{p.relative_to(work).as_posix()}"
    items: list[tuple[Path, str]] = []
    for f in (work / "cameras" / "world_from_sfm.json", work / "dancers" / "roles.json"):
        items.append((f, arc(f)))
    if (work / "sync" / "offsets.json").exists():  # views are posed at sync-corrected times (camera_views)
        items.append((work / "sync" / "offsets.json", arc(work / "sync" / "offsets.json")))
    frames = needed_frames(work, cfg, cams)
    for c in cams:
        for f in (work / "cameras" / f"{c}.npz", work / "frames" / c / "frames.json", work / "people" / f"{c}.npz"):
            if f.exists():
                items.append((f, arc(f)))
        dd = work / "dancers" / c
        for i in sorted(set(frames[c]["views"]) | set(frames[c]["hair"])):
            cand = [work / "frames" / c / f"{i:06d}.jpg"]
            cand += [dd / "masks" / f"mask_{i:04d}_{PIDS[r]:02d}.png" for r in ROLES]
            if i in frames[c]["views"]:
                cand += [dd / "body" / f"mask_{i:04d}_{PIDS[r]:02d}.png" for r in ROLES]
            if i in frames[c]["hair"] or (i in frames[c]["views"] and cfg.hair_mode == "exclude"):
                cand.append(dd / "hair" / f"mask_{i:04d}_{PIDS['follow']:02d}.png")  # hair_mode exclude: every view
            items += [(f, arc(f)) for f in cand if f.exists()]
    return items


def _code_items() -> list[tuple[Path, str]]:
    return [(PACKAGE_ROOT / "dancecap", "dancecap/dancecap")]


def _tar(out: Path, items: list[tuple[Path, str]], gz: bool = True) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti
    with tarfile.open(out, "w:gz" if gz else "w", **({"compresslevel": 1} if gz else {})) as tar:
        for p, a in items:
            tar.add(str(p), arcname=a, filter=skip)
    return out.stat().st_size


def pod_commands(take: str, preset: str = "pod", extra_args: str = "") -> dict:
    """extra_args: CLI flags appended to the bake (default: the explicit hair-aware code path of TextureConfig)."""
    wd = f"{POD_WS}/dancecap"
    rel = f"work/{take}"
    files = " ".join(OUTPUTS)
    if not extra_args:
        d = TextureConfig()
        extra_args = f"--hair-mode {d.hair_mode} --blend {d.blend}"
    return {
        "extract": f"set -e; mkdir -p {POD_WS} && tar xzf /workspace/texture_inputs.tar.gz -C {POD_WS} && "
                   f"rm -f /workspace/texture_inputs.tar.gz",
        "extract_standalone": f"set -e; tar xzf /workspace/smplx.tar.gz -C {POD_WS} && mkdir -p "
                              f"{POD_WS}/mamma/data/body_models/smplx_locked_head/smplx && mv /workspace/SMPLX_NEUTRAL.npz "
                              f"{POD_WS}/mamma/data/body_models/smplx_locked_head/smplx/",
        "setup": "set -e; pip install -q 'numpy<2' smplx scipy 'opencv-python-headless==4.10.0.84' && python -c "
                 "\"import torch, smplx, cv2, scipy; print('torch', torch.__version__, 'cv2', cv2.__version__, "
                 "torch.cuda.get_device_name(0))\"",
        "run": f"set -o pipefail; cd {wd} && mkdir -p {rel}/avatar && DANCECAP_POD=1 PYTHONUNBUFFERED=1 python -m "
               f"dancecap texture {take} --preset {preset} {extra_args} 2>&1 | tee -a {rel}/avatar/pod_texture.log",
        "pack": f"cd {wd}/{rel} && tar czf /workspace/texture_out.tar.gz $(ls {files} 2>/dev/null)",
    }


def write_pod_job(work: Path, preset: str = "pod", max_minutes: float | None = None, cams: list[str] | None = None,
                  pack: bool = True, log=print, step_ms: float | None = None, hair_mode: str | None = None,
                  blend: str | None = None) -> dict:
    """avatar/pod_job.json (+ the upload tars in avatar/pod/ when pack): inputs with sizes, commands, outputs.
    step_ms overrides the preset's view step (e.g. 66.7 for a long take: half the frames to upload and bake);
    hair_mode / blend override the code path (default TextureConfig's: exclude + multiband). max_minutes None = the
    estimate's max_minutes_recommended; an explicit value below it raises ValueError (the kill timer would end the
    bake)."""
    from .avatar_texture import smplx_freshness
    from .mamma_dyn import SMPLX_MODELS
    take = work.name
    over = {**({"step_s": step_ms / 1000} if step_ms else {}), **({"hair_mode": hair_mode} if hair_mode else {}),
            **({"blend": blend} if blend else {})}
    cfg = TextureConfig(**{**PRESETS[preset], **over}, preset=preset)
    cams = cams or sorted(p.stem for p in (work / "cameras").glob("[0-9][0-9].npz"))
    pod_dir = work / "avatar" / "pod"
    items = input_items(work, cfg, cams) + _code_items()
    frames = needed_frames(work, cfg, cams)
    raw = sum(p.stat().st_size for p, _ in items if p.is_file()) + sum(
        f.stat().st_size for p, _ in items if p.is_dir() for f in p.rglob("*.py"))
    model = SMPLX_MODELS / "smplx" / "SMPLX_NEUTRAL.npz"
    inputs = {"texture_inputs": {"local": str(pod_dir / "texture_inputs.tar.gz"),
                                 "pod": "/workspace/texture_inputs.tar.gz", "raw_bytes": raw,
                                 "contents": "code + frames/masks/cameras/people/roles the bake reads",
                                 "frames_per_cam": {c: len(set(v["views"]) | set(v["hair"])) for c, v in frames.items()}},
              "smplx (standalone only)": {"local": str(pod_dir / "smplx.tar.gz"), "pod": "/workspace/smplx.tar.gz",
                                          "from": str(work / "smplx")},
              "body_model (standalone only)": {"local": str(model), "pod": "/workspace/SMPLX_NEUTRAL.npz",
                                               "bytes": model.stat().st_size if model.exists() else None}}
    if pack:
        t0 = time.time()
        inputs["texture_inputs"]["bytes"] = _tar(pod_dir / "texture_inputs.tar.gz", items)
        if all((work / "smplx" / f"{r}.npz").exists() for r in ROLES):
            inputs["smplx (standalone only)"]["bytes"] = _tar(
                pod_dir / "smplx.tar.gz", [(work / "smplx" / f"{r}.npz", f"dancecap/work/{take}/smplx/{r}.npz")
                                           for r in ROLES])
        log(f"[texture] pod inputs packed in {time.time() - t0:.0f} s: texture_inputs "
            f"{inputs['texture_inputs']['bytes'] / 1e6:.1f} MB")
    for v in inputs.values():
        if v.get("bytes"):
            v["mb"] = round(v["bytes"] / 1e6, 1)
    chain_mb = inputs["texture_inputs"].get("mb") or round(raw / 1e6, 1)
    alone_mb = chain_mb + (inputs["smplx (standalone only)"].get("mb") or 1.4) + \
        (inputs["body_model (standalone only)"].get("mb") or 108.8)
    code_path = {"hair_mode": cfg.hair_mode, "blend": cfg.blend, "step_s": cfg.step_s,
                 "modules": ["dancecap/avatar_texture.py", "dancecap/avatar_hair.py"]}
    cmds = pod_commands(take, preset, f"--hair-mode {cfg.hair_mode} --blend {cfg.blend}"
                        + (f" --step-ms {step_ms:g}" if step_ms else ""))
    fresh = smplx_freshness(work)
    n_views = sum(len(v["views"]) for v in frames.values())
    passes = 1 + int(cfg.align_passes)
    ts = [float(f["t"]) for c in cams[:1] for f in json.loads((work / "frames" / c / "frames.json").read_text())["frames"]]
    grid_frames = int(round((max(ts) - min(ts)) * 30)) + 1 if ts else 0
    fixed_min = POD_FIXED_MIN + POD_MIN_PER_FRAME * grid_frames
    rates = per_view_rates(code_path)
    bake_min = (n_views * (rates["pass0"] + (passes - 1) * rates["aligned"])) / 60 + fixed_min
    recommended = int(np.ceil(bake_min * 1.3 + 3))
    if max_minutes is None:
        max_minutes = recommended
    elif max_minutes < recommended:
        raise ValueError(f"texture pod job: max_minutes={max_minutes:g} < recommended {recommended} (~{bake_min:.0f} "
                         f"min of bake for {n_views} views): the kill timer would delete the pod mid-bake. Raise "
                         "--pod-minutes or drop it (default = recommended); --step-ms 66.7 halves the views")
    job = {
        "what": "avatar texture bake (photoreal per-dancer 2048^2 albedo on the SMPL-X UV layout), preset " + preset,
        "take": take, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "preset": preset,
        "preset_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PRESETS[preset].items()},
        "code_path": code_path,
        "cams": cams, "max_minutes": max_minutes,
        "order": "AFTER the MAMMA refit: the bake refuses smplx/ older than cameras/world_from_sfm.json "
                 f"(local smplx/ now: {fresh['status']})",
        "pod_workspace": POD_WS,
        "gpu": "one 80 GB A100/H100 (dancecap.remote.pod_session: budget cap, on-pod kill switch, delete in finally)",
        "inputs": inputs,
        "upload_mb": {"chain_after_mamma": chain_mb, "standalone": round(alone_mb, 1)},
        "upload_s_at_2_5_MBps": {"chain_after_mamma": round(chain_mb / 2.5), "standalone": round(alone_mb / 2.5)},
        "pod_commands": cmds,
        "chain_after_mamma": {
            "when": "same pod session, after `python -m dancecap mamma select <take> --no-qa` promoted smplx/ on the pod",
            "upload": ["texture_inputs"],
            "commands": [cmds["extract"], cmds["setup"], cmds["run"], cmds["pack"]],
            "note": "texture_inputs carries the current dancecap code: extracting it over the MAMMA bundle's code "
                    "is intended (same files, newer avatar_*.py)",
        },
        "standalone": {
            "run": f"python -m dancecap texture {take} --pod --pod-minutes {max_minutes:g}"
                   + (f" --step-ms {step_ms:g}" if step_ms else "") + f" --hair-mode {cfg.hair_mode} --blend {cfg.blend}",
            "upload": ["texture_inputs", "smplx (standalone only)", "body_model (standalone only)"],
            "commands": [cmds["extract"], cmds["extract_standalone"], cmds["setup"], cmds["run"], cmds["pack"]],
        },
        "download": {"pod": "/workspace/texture_out.tar.gz", "local": str(pod_dir / "texture_out.tar.gz"),
                     "extract_to": str(work), "estimate_mb": 75},
        "outputs": OUTPUTS,
        "estimate": {"setup_min": "1-2 (pip smplx/opencv; no CUDA build: the rasteriser is pure torch)",
                     "views": n_views, "passes": passes,
                     "bake_min": round(bake_min, 1),
                     "s_per_view": rates,
                     "bake_min_basis": (POD_RATE_BASIS if rates == POD_S_PER_VIEW else "A100 2026-10-06 old path")
                                       + f"; +{fixed_min:.1f} min head fix ({grid_frames} grid frames) / hair models / "
                                       "fills / review",
                     "max_minutes_recommended": recommended,
                     "cost_usd_max": round(POD_PRICE_HINT * (max_minutes + 6) / 60, 2)},
        "local_alternative": f"python -m dancecap texture {take}   # preset local on the 2080 Ti (~5-8 min incl. "
                             "head fix + review): the bake is light enough to run locally if no pod session is open",
        "after_download_locally": "inspect review/avatar_texture_{lead,follow}.jpg, avatar_uv.jpg, avatar_face.jpg; "
                                  "the export stage copies avatar/*_albedo.png + uv_mesh.npz into the Unity capture",
    }
    (work / "avatar").mkdir(parents=True, exist_ok=True)
    (work / "avatar" / "pod_job.json").write_text(json.dumps(job, indent=1))
    log(f"[texture] pod job -> {work / 'avatar' / 'pod_job.json'} (upload chain {chain_mb} MB / standalone "
        f"{alone_mb:.0f} MB)")
    return job


def run_pod(work: Path, preset: str = "pod", max_minutes: float | None = None, log=print,
            step_ms: float | None = None, max_cost: float | None = DEFAULT_MAX_COST, hair_mode: str | None = None,
            blend: str | None = None) -> dict:
    """Standalone pod session: pack -> upload -> setup -> bake -> pack -> download -> extract. smplx/ must be fresh.
    Refuses to start when max_minutes is below the estimate's recommendation (write_pod_job raises); the session is
    capped at max_cost $ (default 10) besides the global cap."""
    from . import remote
    from .avatar_texture import smplx_freshness
    fresh = smplx_freshness(work)
    if fresh["stale"]:
        raise RuntimeError(f"texture --pod: smplx/ {fresh['status']}; refit MAMMA first (or chain the bake into the "
                           "MAMMA pod session, see avatar/pod_job.json chain_after_mamma)")
    job = write_pod_job(work, preset, max_minutes, log=log, step_ms=step_ms, hair_mode=hair_mode, blend=blend)
    max_minutes = float(job["max_minutes"])
    cmds = job["standalone"]["commands"]
    plog = work / "avatar" / "pod" / "pod.log"
    got = Path(job["download"]["local"])
    got.unlink(missing_ok=True)  # never extract a previous run's archive
    res: dict = {"plan": str(work / "avatar" / "pod_job.json")}
    with remote.pod_session("texture", max_minutes=max_minutes, max_cost=max_cost) as pod:
        res["pod"] = {"id": pod.id, "gpu": pod.gpu, "cost_per_hr": pod.cost_per_hr}
        t0 = time.time()
        for k in job["standalone"]["upload"]:
            pod.put(Path(job["inputs"][k]["local"]), job["inputs"][k]["pod"])
        res["upload_s"] = round(time.time() - t0)
        for c in cmds[:3]:
            pod.run(c, log=plog)
        t0 = time.time()
        res["bake_rc"] = pod.run(cmds[3], check=False, log=plog)
        res["bake_min"] = round((time.time() - t0) / 60, 1)
        pod.run(cmds[4], check=False, log=plog)
        try:
            pod.get(job["download"]["pod"], got)
        except Exception as e:  # noqa: BLE001 - the session deletes the pod either way
            res["download_error"] = str(e)
    if got.exists() and "download_error" not in res:
        with tarfile.open(got) as tar:
            tar.extractall(work, filter="data")
        res["extracted"] = True
    (work / "avatar" / "pod" / "result.json").write_text(json.dumps(res, indent=1))
    if res.get("bake_rc"):
        raise RuntimeError(f"texture: pod bake failed (rc {res['bake_rc']}); see {plog}")
    return res
