"""Pod jobs of the "sides" step (DESIGN 5), schema dancecap.pod_job/1 (the shape of the v4 contact jobs under work/<take>/v4/contacts/jobs/).

    job_evidence(take)  J1 sides_<take>_A: a CPU pod runs `evidence` + `decide` (no GPU, no phone video, no key file)
                        -> work/<take>/sides/jobs/sides_<take>_A.json; run it with
                           python -m dancecap remote run --jobs dancecap/work/<take>/sides/jobs/sides_<take>_A.json --max-cost 0.5 --cpu --vcpu 8

Inputs are found from the take's own layout (no take constants): the base fits, the teachers and cameras of every camera in the chunks'
select.json union, people/attach, stitch.json, every chunk's mamma/select.json + sync/offsets.json, the consolidated obs (the first
compressed copy among cfg.obs_upload_dirs, mapped to the pod path work/<take>/sides/obs/), the SMPL-X neutral model + MAMMA's 512-point
part split, the film direction (film-time labels only).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..take import PACKAGE_ROOT, WORKSPACE
from .adapters import chunk_select_cams, chunk_work, read_json, work_dir
from .config import SidesCfg, load_cfg

SCHEMA = "dancecap.pod_job/1"
MODEL_FILES = ("mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz",
               "mamma/data/body_models/downsampled_verts/smplx_512_body_parts.npz")


def ws(p: Path) -> str:
    """workspace-relative posix path"""
    return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()


def jobs_dir(take: str) -> Path:
    return work_dir(take) / "sides" / "jobs"


def obs_source(take: str, cfg: SidesCfg) -> Path:
    work = work_dir(take)
    cams = sorted({c for v in chunk_select_cams(work).values() for c in v})
    for d in cfg.obs_upload_dirs:
        p = work / d
        if all((p / f"cam{c}.npz").exists() for c in cams):
            return p
    raise FileNotFoundError(f"{take}: no consolidated obs in {list(cfg.obs_upload_dirs)} (python -m dancecap.sides obs --take {take})")


def evidence_inputs(take: str, cfg: SidesCfg | None = None, obs_from: Path | None = None) -> list:
    cfg = cfg or load_cfg(take)
    work = work_dir(take)
    st = read_json(work / "stitch.json")
    cams = sorted({c for v in chunk_select_cams(work).values() for c in v})
    items: list = [ws(work / cfg.base_fit) + "/", ws(work / "stitch.json"), ws(work / "cameras") + "/"]
    if (work / "people" / "attach").exists():
        items.append(ws(work / "people" / "attach") + "/")
    for c in cams:
        items.append(ws(work / "people" / f"teachers_{c}.npz"))
    for ch in st["chunks"]:
        cw = chunk_work(work, ch["name"])
        for rel in ("mamma/select.json", "sync/offsets.json"):
            if (cw / rel).exists():
                items.append(ws(cw / rel))
    src = Path(obs_from) if obs_from else obs_source(take, cfg)
    for c in cams:
        items.append({"local": ws(src / f"cam{c}.npz"), "pod": ws(work / "sides" / "obs" / f"cam{c}.npz")})
    for m in MODEL_FILES:
        if (WORKSPACE / m).exists():
            items.append(m)
    if (work / "film" / "direction.json").exists():
        items.append(ws(work / "film" / "direction.json"))
    return items


def job_evidence(take: str, cfg: SidesCfg | None = None, workers: int = 8, max_minutes: int = 60, obs_from: Path | None = None,
                 write: bool = True) -> tuple[dict, Path]:
    """J1 (DESIGN 5): evidence + decide on a CPU pod -> (job dict, path of the json)"""
    cfg = cfg or load_cfg(take)
    w = ws(work_dir(take))
    job = {
        "schema": SCHEMA,
        "name": f"sides_{take}_A",
        "take": take,
        "max_minutes": int(max_minutes),
        "needs": [],
        "cpu": True,
        "models": "none",
        "min_vcpu": 4,
        "inputs": evidence_inputs(take, cfg, obs_from),
        "cwd": "dancecap",
        "env": {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        "exclude": ["dancecap/work/runpod/pywheels"],
        "commands": [
            {"name": "evidence", "cmd": f"python -m dancecap.sides evidence --take {take} --workers {int(workers)}", "timeout_min": 40},
            {"name": "decide", "cmd": f"python -m dancecap.sides decide --take {take}", "timeout_min": 15},
        ],
        "outputs": [f"{w}/sides/evidence/", f"{w}/sides/relations.npz", f"{w}/sides/relations.json", f"{w}/sides/yaw.npz",
                    f"{w}/sides/spins.json", f"{w}/sides/violations.json"],
        "verify": [f"{w}/sides/evidence/evidence.json", f"{w}/sides/relations.json", f"{w}/sides/violations.json"],
        "notes": ["sides Phase A (DESIGN 6.1): camera evidence (17 COCO joints of both dancers triangulated at 30 Hz, visibility-consistent, "
                  "held-out phone variant, per-view face / L-R / part bits) + the relation decisions, yaw track and violation list; no motion "
                  "is changed. CPU pod; the phone videos and key files never leave the PC."],
    }
    p = jobs_dir(take) / f"{job['name']}.json"
    if write:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(job, indent=1), encoding="utf-8")
    return job, p


# ------------------------------------------------------------------------------------------------ J2: the windowed solve on GPU pods

POD_MODEL_FILES = ("mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz",
                   "mamma/data/body_models/downsampled_verts/verts_512.pkl",
                   "mamma/data/body_models/downsampled_verts/smplx_512_body_parts.npz")


def slim_inputs(take: str, cfg: SidesCfg | None = None) -> dict:
    """slim copies of the input and the prepared fits for the pods (poses, translation, betas, t, stance, owner; the joints are FK'd on
    the pod, the 44 MB of contact scores stay home) -> {role: {"input": path, "cpu": path}}"""
    import numpy as np
    from .prepare import SLIM_KEYS, load_fit, save_fit
    cfg = cfg or load_cfg(take)
    work = work_dir(take)
    out = {}
    for r in cfg.roles:
        d = {}
        for name, src in (("input", work / cfg.base_fit / f"{r}.npz"), ("cpu", work / "sides" / "cpu" / f"{r}.npz")):
            f = load_fit(src)
            dst = work / "sides" / "cpu" / "pod" / name / f"{r}.npz"
            save_fit(dst, {k: f[k] for k in SLIM_KEYS if k in f})
            d[name] = dst
        out[r] = d
    return out


def solve_inputs(take: str, cfg: SidesCfg | None = None, obs_from: Path | None = None) -> list:
    cfg = cfg or load_cfg(take)
    work = work_dir(take)
    st = read_json(work / "stitch.json")
    sd = work / "sides"
    sl = slim_inputs(take, cfg)
    items: list = []
    for r, d in sl.items():
        items.append({"local": ws(d["input"]), "pod": ws(work / cfg.base_fit / f"{r}.npz")})
        items.append({"local": ws(d["cpu"]), "pod": ws(sd / "cpu" / f"{r}.npz")})
    for rel in ("cpu/prepare.json", "plan.json", "relations.npz", "relations.json", "yaw.npz", "spins.json", "cfg.json"):
        if (sd / rel).exists():
            items.append(ws(sd / rel))
    for f in sorted((sd / "evidence").iterdir()):
        if f.suffix in (".npz", ".json"):
            items.append(ws(f))
    items.append(ws(work / "stitch.json"))
    for ch in st["chunks"]:
        cw = chunk_work(work, ch["name"])
        for rel in ("mamma/select.json", "sync/offsets.json"):
            if (cw / rel).exists():
                items.append(ws(cw / rel))
    cams = sorted({c for v in chunk_select_cams(work).values() for c in v})
    # the full 512-point obs of both dancers (sides/obs_pod: landmarks quantized to 1/8 px) as the pods' sides/obs; the older compressed
    # copy (v4/contacts/pod_inputs/obs) keeps only the leader's 160 arm / hand points and none of the follower's
    pod_obs = work / "sides" / "obs_pod"
    if obs_from is None and all((pod_obs / f"cam{c}.npz").exists() for c in cams):
        for c in cams:
            items.append({"local": ws(pod_obs / f"cam{c}.npz"), "pod": ws(work / "sides" / "obs" / f"cam{c}.npz")})
    else:
        src = Path(obs_from) if obs_from else obs_source(take, cfg)
        for c in cams:
            items.append({"local": ws(src / f"cam{c}.npz"), "pod": ws(work / "v3" / "hand_contact" / "obs" / f"cam{c}.npz")})
    for m in POD_MODEL_FILES:
        if (WORKSPACE / m).exists():
            items.append(m)
    if (work / "film" / "direction.json").exists():
        items.append(ws(work / "film" / "direction.json"))
    return items


def job_solve(take: str, n_parts: int = 2, cfg: SidesCfg | None = None, max_minutes: int = 60, write: bool = True,
              tag: str = "C", only: list | None = None, workers: int = 4) -> list:
    """J2 (DESIGN 5): one GPU job per part (`--parallel` runs them on separate pods) -> [(job dict, path)]"""
    cfg = cfg or load_cfg(take)
    w = ws(work_dir(take))
    inputs = solve_inputs(take, cfg)
    out = []
    for i in range(n_parts):
        extra = (f" --only {','.join(str(x) for x in only)}" if only else "") + f" --workers {int(workers)}"
        job = {
            "schema": SCHEMA,
            "name": f"sides_{take}_{tag}_p{i}",
            "take": take,
            "max_minutes": int(max_minutes),
            "needs": [],
            "min_vram_gb": 24,
            "inputs": inputs,
            "cwd": "dancecap",
            "env": {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
            "exclude": ["dancecap/work/runpod/pywheels"],
            "commands": [{"name": f"solve_p{i}", "cmd": f"python -m dancecap.sides solve --take {take} --part {i}/{n_parts} --device cuda{extra}",
                          "timeout_min": int(max_minutes) - 5}],
            "outputs": [f"{w}/sides/solve/part_{i}/"],
            "verify": [f"{w}/sides/solve/part_{i}/summary.json"],
            "notes": [f"sides Phase C (DESIGN 6.1): the windowed solve of plan.json, part {i}/{n_parts}; each window gated on the pod "
                      "(attempts: planned -> lambda_prior x 2 -> the worst chain only); merge + the whole-take gate run on the PC. No phone "
                      "video, no key file."],
        }
        p = jobs_dir(take) / f"{job['name']}.json"
        if write:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(job, indent=1), encoding="utf-8")
        out.append((job, p))
    return out
