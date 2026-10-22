"""Batched Runpod sessions: many pod_job.json specs -> ONE pod, ONE upload, one bootstrap, every job, downloads.

    python -m dancecap remote jobs --take larissa_kadu            # discovered pod_job.json files, parsed
    python -m dancecap remote bundle --take larissa_kadu          # build the upload bundle only (offline)
    python -m dancecap remote run --take larissa_kadu --jobs mamma,room --max-cost 6
    python -m dancecap remote validate --max-cost 2.5             # bootstrap + import checks + MAMMA synthetic

A session (work/runpod/sessions/<sid>/) is:
  1. bundle    one tar.gz: dancecap code + takes + remote/bootstrap.sh, mamma-turing (no .git/docs/gui/output),
               only the model files needed (MAMMA locked-head SMPLX_NEUTRAL.npz, verts_512.pkl, part split, the
               workspace smplx/SMPLX_NEUTRAL.npz - identical files are stored once, as tar hard links), cached CUDA
               wheels (work/runpod/wheelhouse/<key>/) and every job's inputs, laid out relative to the workspace so it
               extracts to /workspace/HeadMovement/... exactly like the local tree. Cached by content
               (work/runpod/bundles/bundle_<hash>.tar.gz) so re-running never rebuilds or re-uploads needlessly.
  2. pod       dancecap.remote.pod_session (budget cap, per-session max_cost, on-pod kill switch, always deleted); GPUs
               fastest first for the jobs' memory need (job_vram_gb), each GPU in every cloud before the next GPU; a GPU
               the session max kept out is a warning in state.json (choice / warnings), never silent
  3. upload    skipped when the pod already holds a file with the same sha256
  4. extract   marker per bundle hash
  5. bootstrap remote/bootstrap.sh <features of all jobs> check (idempotent per feature); newly built wheels are
               downloaded into the local wheelhouse so the next session installs instead of compiling. Features a job
               does not declare but needs are added (job_features: ffmpeg + ffprobe for video, open3d for physics)
  6. per job   setup + commands (each with a .done marker, timeout and log), then outputs packed on the pod,
               downloaded, sha256-checked, extracted with a path whitelist and verified; a failed job still
               returns its logs and partial outputs. While a command runs, its frame counter (FrameProgress) writes
               progress/frames.json: frames (or iterations) done / total and the rate, from the lines it prints
  7. result    result.json (timings of every step, pod, GPU, price, cost from the ledger, per-job status)

Jobs whose last successful run in this session dir had the same fingerprint (spec + input sizes/mtimes) are not
re-run (--force overrides): resume a failed batch with --session <sid>.
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import re
import shlex
import tarfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

from . import env
from .take import PACKAGE_ROOT, WORKSPACE

SCHEMA = "dancecap.pod_job/1"
POD_WS = "/workspace/HeadMovement"
POD_UPLOAD = "/workspace/upload"
POD_OUT = "/workspace/out"
RUNPOD_DIR = PACKAGE_ROOT / "work" / "runpod"
SESSIONS = RUNPOD_DIR / "sessions"
BUNDLES = RUNPOD_DIR / "bundles"
WHEEL_KEY = "cp311-torch2.5.1-cu124-sm80_90"  # must match remote/bootstrap.sh
WHEELHOUSE = RUNPOD_DIR / "wheelhouse" / WHEEL_KEY
# linux cp311 wheels of the bootstrap's PyPI packages, installed offline first (bootstrap.sh pipq); fill with
#   pip download --only-binary=:all: --platform manylinux2014_x86_64 --python-version 3.11 --implementation cp
#     --abi cp311 --abi abi3 --abi none -d <dir> <bootstrap base/mamma packages>  (+ --no-deps for torch users:
#     smplx kornia pyrender hydra-core omegaconf; --no-binary for PyOpenGL==3.1.0 antlr4-python3-runtime==4.9.3)
PYWHEELS = RUNPOD_DIR / "pywheels" / "cp311-linux"
FEATURES = ("base", "mamma", "room", "texture", "dancers", "cameras", "ma2d", "gsplat", "ffmpeg", "open3d",
            "check")  # check last
UPLOAD_MBPS, DOWNLOAD_MBPS = 2.5, 12.0  # measured home link (2026-10-06)
# bootstrap minutes per feature: cold (CUDA wheels compiled on the pod) / with cached wheels. ffmpeg: a static build fetched
# in the background while base installs; open3d: the 2026-10-21 trial installed it in 19 s
BOOTSTRAP_MIN = {"base": 5, "mamma": 8, "room": 15, "texture": 0, "dancers": 3, "cameras": 1, "ma2d": 3, "gsplat": 12,
                 "ffmpeg": 1, "open3d": 1, "check": 1}
BOOTSTRAP_MIN_CACHED = {"base": 5, "mamma": 2, "room": 3, "texture": 0, "dancers": 3, "cameras": 1, "ma2d": 3,
                        "gsplat": 1, "ffmpeg": 1, "open3d": 1, "check": 1}
# features that come from the network in seconds (a static ffmpeg build, one PyPI wheel): they never pull the 330 MB pywheels
# or the CUDA wheelhouse into the bundle (wheel_policy), so a slim bundle stays slim
LIGHT_FEATURES = {"ffmpeg", "open3d"}
# Implied features (job_features): the 2026-10-21 trial lost 4 sessions (~45 min, $1.33) to pods without ffprobe / ffmpeg or
# open3d. A job gets 'ffmpeg' when it decodes video - a pod stage that does (cameras/newcam, ma2d, dancers: front_pod and
# front_full decode the source MP4s), a video among its inputs, or a command that names ffmpeg / ffprobe / a video file / a
# decoding module - and 'open3d' when a command runs physics (physics_body.segment_masses) or the room mesh / review.
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".mpg", ".mpeg")
DECODING_FEATURES = {"cameras", "ma2d", "dancers"}
_FFMPEG_CMD = re.compile(r"\bff(?:mpeg|probe)\b|\.(?:mp4|mov|m4v|mkv|avi|webm|mts)\b|\bnewcam\b|"
                         r"dancecap[ .](?:front_pod|front_full|recap_front|ingest|front)\b", re.I)
# open3d: physics_body.segment_masses imports it; physics, counterbalance (body_com -> physics._body) and pose_quality
# --physics (physics.run) reach it, as do the room mesh / review
_OPEN3D_CMD = re.compile(r"-m\s+dancecap(?:\.cli)?\s+(?:physics|counterbalance)\b|\bdancecap\.(?:physics|counterbalance)\b|"
                         r"\broom_mesh\b|\broom_review\b|"
                         r"(?:-m\s+dancecap(?:\.cli)?\s+(?:pose_)?quality|\bdancecap\.pose_quality)\b[^;&|]*\s--physics\b")


def implied_features(job: "JobSpec") -> list[str]:
    """Bootstrap features a job needs without declaring them (see _FFMPEG_CMD / _OPEN3D_CMD above)."""
    cmds = " ".join([c.cmd for c in job.commands] + list(job.setup))
    out = []
    if (set(job.needs) & DECODING_FEATURES or _FFMPEG_CMD.search(cmds)
            or any(str(p).lower().endswith(VIDEO_EXT) for p, _ in job.inputs + job.inputs_if_alone)):
        out.append("ffmpeg")
    if _OPEN3D_CMD.search(cmds):
        out.append("open3d")
    return [f for f in out if f not in job.needs]


def job_features(job: "JobSpec") -> list[str]:
    """What the pod must bootstrap for this job: its declared needs + the implied ones."""
    return list(job.needs) + implied_features(job)


# GPU memory per job (remote.gpus_for picks the fastest GPUs that have it). Every runner job measured so far fits a 24 GB card:
# SAM 3D Body peaks at 3.5 GB, LHM++ at 8.2 GB (4090), MAMMA 12.1 GiB for 1648 frames x 6 cameras and 6.4 GiB for a 900-frame
# chunk, the texture bake 3.8 GB, the 2026-10-21 trial's front / fit / analyses ran on a 4090. Kinds measured or sized above:
DEFAULT_VRAM_GB = 24.0
VRAM_BY_KIND = (
    # CoTracker3 offline (TrackEverything stage 0) ran out of memory on a 44 GB A40 with 39 GB in use (20261020-130536-batch)
    (re.compile(r"cotracker|trackeverything|\bct3\b", re.I), 80.0),
    # sized for ONE 80 GB pod and never measured on a 24 GB card (2026-10-21 review): mamma_chunks fits 6 chunks + a holdout
    # side by side under CUDA MPS; dancers_pod runs --workers 5 SAM 2.1 large processes (3-5 GB each, ~32 GB host RAM each);
    # front_pod runs every heavy step at once (A100 80 GB, ~30 vCPU, >100 GB RAM). The 80 GB cards also come with the
    # host RAM / vCPUs these need, which the GPU ladder does not check. Until measured: 80 GB (H100 first).
    (re.compile(r"dancecap\.mamma_chunks\s+fit\b|dancecap\.dancers_pod\s+run\b|dancecap\.front_pod\s+(?:run|supervise)\b"),
     80.0),
)


def job_vram_gb(job: "JobSpec") -> float:
    """min_vram_gb when declared; 80 for room training (room.py's pod preset is sized for an 80 GB A100/H100: every camera
    at native resolution, 1M Gaussians); VRAM_BY_KIND; 48 for a single `mamma fit` without --fit-frames over more than
    MAMMA_LONG_FRAMES frames (6.4 GiB per 900 frames: a 24 GB card holds ~3000); else DEFAULT_VRAM_GB."""
    if job.min_vram_gb is not None:
        return float(job.min_vram_gb)
    if "room" in job.needs:
        return 80.0
    text = " ".join([job.name] + [c.cmd for c in job.commands])
    gb = next((gb for rx, gb in VRAM_BY_KIND if rx.search(text)), None)
    if gb is not None:
        return gb
    return 48.0 if _long_mamma_fit(job) else DEFAULT_VRAM_GB


MAMMA_LONG_FRAMES = 3000


def _long_mamma_fit(job: "JobSpec") -> bool:
    """A `mamma fit <take>` over the whole take (no --fit-frames window) whose take is longer than MAMMA_LONG_FRAMES frames at
    its timeline rate. Reads the take TOML only; unknown -> False (24 GB, as measured for the trial's fit)."""
    for c in job.commands:
        m = re.search(r"\bmamma\s+fit\s+([A-Za-z0-9_.-]+)", c.cmd)
        if not m or "--fit-frames" in c.cmd:
            continue
        try:
            import tomllib
            d = tomllib.loads((PACKAGE_ROOT / "takes" / f"{m.group(1)}.toml").read_text(encoding="utf-8"))
            w = d.get("window") or (d.get("stitch") or {}).get("mono", [[None, None]])[0]
            hz = float((d.get("timeline") or {}).get("hz") or 30)
            if w and w[0] is not None and (float(w[1]) - float(w[0])) * hz > MAMMA_LONG_FRAMES:
                return True
        except Exception:  # noqa: BLE001 - no take file / no window: the default
            continue
    return False
DEFAULT_EXCLUDE = ("__pycache__", "*.pyc", ".git", ".pytest_cache", "node_modules", ".ipynb_checkpoints")
ARCHIVE_CODE_EXCLUDE = ("dancecap/dancecap", "mamma-turing")  # code always comes from the bundle (current tree)
MAMMA_SKIP = ("docs", "gui", "docker", "data", "tests", "output", "outputs", "wandb")  # top-level mamma-turing dirs


# ---------------------------------------------------------------------------------------------- job spec


@dataclass
class Command:
    name: str
    cmd: str
    timeout_min: float = 60.0
    optional: bool = False  # failure is reported but the job's later commands still run


@dataclass
class JobSpec:
    name: str
    take: str
    max_minutes: float
    commands: list[Command]
    outputs: list[str]                       # workspace-relative posix paths; trailing "/" = directory
    inputs: list[tuple[str, str]] = field(default_factory=list)  # (local path, workspace-relative pod path)
    needs: list[str] = field(default_factory=list)  # bootstrap features
    exclude: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    secrets: dict = field(default_factory=dict)  # {ENV_NAME: local key file}: pushed over ssh stdin, exported per command
    cwd: str = "dancecap"
    setup: list[str] = field(default_factory=list)
    verify: list[str] = field(default_factory=list)
    extract: str = "workspace"               # "workspace" (in place) | "session" (session dir, for checks)
    after_download: list[str] = field(default_factory=list)  # local follow-ups (not run by the runner)
    notes: list[str] = field(default_factory=list)
    source: str | None = None
    legacy: str | None = None                # "mamma-v0" / "room-v0" / "avatar-v0" when adapted from an older shape
    archives: list[tuple[str, str]] = field(default_factory=list)  # (local .tar[.gz], ws-relative dir to extract into)
    after: list[str] = field(default_factory=list)  # jobs that run first when they are in the same batch
    inputs_if_alone: list[tuple[str, str]] = field(default_factory=list)  # inputs used when no `after` job is in the batch
    min_vram_gb: float | None = None         # optional: lets the runner try every GPU with this much memory (remote.gpus_for)
    bundle_pywheels: bool | None = None      # None = by its bootstrap features (bundle_files); True/False forces it
    # pod log files (workspace-relative globs) whose progress lines the runner relays while a command runs (watch_globs:
    # MAMMA's ma_3d log is added for every `mamma fit` command); they feed the frame counter (FrameProgress)
    watch: list[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        d = asdict(self)
        d.pop("source", None)
        if not d.get("watch"):   # added 2026-10-21: an unset field leaves earlier fingerprints (session resume) unchanged
            d.pop("watch", None)
        h = hashlib.sha256(json.dumps(d, sort_keys=True).encode())
        for local, _ in self.inputs + self.archives + self.inputs_if_alone:
            p = _local(local)
            for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
                if f.is_file():
                    st = f.stat()
                    h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}".encode())
        return h.hexdigest()[:16]


def ws_rel(p: str) -> str:
    """Normalise a pod path to workspace-relative posix; refuses absolute paths outside the pod workspace and '..'."""
    s = str(p).replace("\\", "/").strip()
    trailing = s.endswith("/")
    if s.startswith(POD_WS + "/"):
        s = s[len(POD_WS) + 1:]
    if s.startswith("/") or re.match(r"^[A-Za-z]:", s):
        raise ValueError(f"pod path must be relative to {POD_WS}: {p}")
    parts = [x for x in PurePosixPath(s).parts if x not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError(f"bad pod path: {p}")
    return "/".join(parts) + ("/" if trailing else "")


def _local(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else WORKSPACE / p


def _command(c, i: int, default_timeout: float) -> Command:
    if isinstance(c, str):
        return Command(f"cmd{i + 1}", c, default_timeout)
    return Command(c.get("name") or f"cmd{i + 1}", c["cmd"], float(c.get("timeout_min", default_timeout)),
                   bool(c.get("optional", False)))


def _parse_v1(d: dict, source: str | None) -> JobSpec:
    for k in ("name", "take", "max_minutes", "commands", "outputs"):
        if k not in d:
            raise ValueError(f"{source or 'job'}: missing '{k}' ({SCHEMA})")
    mm = float(d["max_minutes"])
    def items(key):
        out = []
        for x in d.get(key, []):
            if isinstance(x, str):
                out.append((x, ws_rel(x)))
            else:
                out.append((str(x["local"]), ws_rel(x.get("pod") or x["local"])))
        return out
    inputs = items("inputs")
    archives = [(str(a["local"]), ws_rel(a["extract_to"]).rstrip("/") if a.get("extract_to") not in (None, "", ".")
                 else "") for a in d.get("archives", [])]
    return JobSpec(archives=archives, after=list(d.get("after", [])), inputs_if_alone=items("inputs_if_alone"),
                   min_vram_gb=float(d["min_vram_gb"]) if d.get("min_vram_gb") is not None else None,
                   bundle_pywheels=d.get("bundle_pywheels"), watch=[ws_rel(w) for w in d.get("watch", [])],
                   name=d["name"], take=d["take"], max_minutes=mm,
                   commands=[_command(c, i, mm) for i, c in enumerate(d["commands"])],
                   outputs=[ws_rel(o) for o in d["outputs"]], inputs=inputs, needs=list(d.get("needs", [])),
                   exclude=list(d.get("exclude", [])), env=dict(d.get("env", {})), secrets=_secrets(d.get("secrets", {})),
                   cwd=ws_rel(d.get("cwd", "dancecap")).rstrip("/"), setup=list(d.get("setup", [])),
                   verify=[ws_rel(v) for v in d.get("verify", d["outputs"])], extract=d.get("extract", "workspace"),
                   after_download=list(d.get("after_download", [])), notes=list(d.get("notes", [])), source=source)


def _parse_legacy_mamma(d: dict, source: str | None) -> JobSpec:
    """pose-prep's first pod_job.json (upload.groups / setup / dry_check / run / download)."""
    inputs = [(it["local"], ws_rel(it["pod"])) for g in d["upload"]["groups"].values() for it in g["items"]]
    mm = float(d.get("estimate_minutes", {}).get("total_max", 60))
    cmds = []
    if d.get("dry_check"):
        cmds.append(Command("dry_check", d["dry_check"], 10))
    for c in d["run"]:
        verb = re.search(r"mamma (fit|select)\b", c)
        tag = re.search(r"--tag (\S+)", c)
        nm = (f"fit_{tag.group(1)}" if verb and verb.group(1) == "fit" and tag else
              ("select" if verb and verb.group(1) == "select" else f"cmd{len(cmds) + 1}"))
        cmds.append(Command(nm, c, mm))
    outs = [ws_rel(o) for o in d["download"]]
    return JobSpec(name="mamma", take=d["take"], max_minutes=mm, commands=cmds, outputs=outs, inputs=inputs,
                   needs=["mamma"], env=dict(d.get("env", {})), cwd=ws_rel(d.get("cwd", "dancecap")).rstrip("/"),
                   verify=[o for o in outs if not o.rstrip("/").endswith("logs")],
                   after_download=[d["after_download_locally"]] if d.get("after_download_locally") else [],
                   notes=list(d.get("notes", [])) + ["legacy setup lines replaced by bootstrap feature 'mamma'"],
                   exclude=list(DEFAULT_EXCLUDE), source=source, legacy="mamma-v0")


def _parse_legacy_room(d: dict, source: str | None) -> JobSpec:
    """room's first pod_job.json (inputs code/bundle tarballs, pod_commands with /workspace/dancecap paths)."""
    take = d["take"]
    work_rel = f"dancecap/work/{take}"
    b = d["inputs"]["bundle"]
    src = Path(b["from"])
    try:
        pod_b = ws_rel(src.relative_to(WORKSPACE).as_posix())
    except ValueError:
        pod_b = f"{work_rel}/room/bundle"
    fix = lambda c: re.sub(r"/workspace/dancecap(?=/|\b)", f"{POD_WS}/dancecap", c)
    pc = d["pod_commands"]
    mm = float(d.get("max_minutes", 110))
    cmds = [Command("train", fix(pc["train"]), mm)]
    if pc.get("post"):
        cmds.append(Command("post", fix(pc["post"]), 30, optional=True))  # fallback: mesh/eval locally
    outs = [ws_rel(f"{work_rel}/{o}") for o in d["outputs"]]
    return JobSpec(name="room", take=take, max_minutes=mm, commands=cmds, outputs=outs, inputs=[(str(src), pod_b)],
                   needs=["room"], cwd="dancecap",
                   verify=[o for o in outs if "/room/" in o and not any(x in o for x in ("mesh", "eval", "_tex"))
                           and not o.endswith(".log")],  # train/quest products; mesh/eval may fall back locally
                   after_download=[d["fallback"]] if d.get("fallback") else [],
                   notes=["legacy setup/pack lines replaced by bootstrap feature 'room' and the runner's packing",
                          "post (mesh, eval) is optional: the fallback runs it locally"],
                   exclude=list(DEFAULT_EXCLUDE), source=source, legacy="room-v0")


def _parse_legacy_avatar(d: dict, source: str | None) -> JobSpec:
    """texture's first pod_job.json (texture_inputs tarball + chain_after_mamma / standalone command lists)."""
    take = d["take"]
    work_rel = f"dancecap/work/{take}"
    ins = d["inputs"]
    archives = [(ins["texture_inputs"]["local"], "")]
    alone = []
    for k, v in ins.items():
        if "standalone" in k and v.get("from"):  # the local smplx/ (only when MAMMA is not refitted in the batch)
            try:
                alone.append((v["from"], ws_rel(Path(v["from"]).relative_to(WORKSPACE).as_posix())))
            except ValueError:
                pass
    mm = float(d.get("max_minutes", 35))
    outs = [ws_rel(f"{work_rel}/{o}") for o in d["outputs"]]
    return JobSpec(name="avatar", take=take, max_minutes=mm, commands=[Command("bake", d["pod_commands"]["run"], mm)],
                   outputs=outs, needs=["texture"], archives=archives, after=["mamma"], inputs_if_alone=alone,
                   verify=[o for o in outs if o.endswith(("_albedo.png", "uv_mesh.npz", "avatar/report.json"))],
                   after_download=[d["after_download_locally"]] if d.get("after_download_locally") else [],
                   notes=["legacy extract/setup/pack lines replaced by the runner (archive minus code, bootstrap "
                          "'texture', packing)"], exclude=list(DEFAULT_EXCLUDE), source=source, legacy="avatar-v0")


def parse_job(d: dict, source: str | None = None) -> JobSpec:
    if d.get("schema") == SCHEMA:
        job = _parse_v1(d, source)
    elif "pod_commands" in d and "texture_inputs" in d.get("inputs", {}):
        job = _parse_legacy_avatar(d, source)
    elif "upload" in d and "run" in d:
        job = _parse_legacy_mamma(d, source)
    elif "pod_commands" in d:
        job = _parse_legacy_room(d, source)
    else:
        raise ValueError(f"{source or 'job'}: not a pod job (expected schema {SCHEMA!r})")
    bad = set(job.needs) - set(FEATURES)
    if bad:
        raise ValueError(f"{job.name}: unknown bootstrap features {sorted(bad)}; known {FEATURES}")
    if job.extract not in ("workspace", "session"):
        raise ValueError(f"{job.name}: extract must be 'workspace' or 'session'")
    names = [c.name for c in job.commands]
    if len(set(names)) != len(names) or not names:
        raise ValueError(f"{job.name}: command names must be unique and non-empty: {names}")
    if not job.outputs:
        raise ValueError(f"{job.name}: no outputs")
    return job


def load_job(path: str | Path) -> JobSpec:
    path = Path(path)
    return parse_job(json.loads(path.read_text(encoding="utf-8")), str(path))


def discover_jobs(take: str) -> list[Path]:
    return sorted((PACKAGE_ROOT / "work" / take).glob("*/pod_job.json"))


def resolve_jobs(take: str | None, names: list[str] | None) -> list[JobSpec]:
    """--jobs entries are pod_job.json paths or stage names (the directory holding pod_job.json under the take)."""
    found = {p.parent.name: p for p in discover_jobs(take)} if take else {}
    out = []
    for n in names or sorted(found):
        p = Path(n)
        if p.suffix == ".json" and (p.exists() or _local(p).exists()):
            out.append(load_job(p if p.exists() else _local(p)))
        elif n in found:
            out.append(load_job(found[n]))
        else:
            raise SystemExit(f"no pod job {n!r} (found {sorted(found)} under work/{take})")
    seen = set()
    for j in out:
        if j.name in seen:
            raise SystemExit(f"two jobs named {j.name}")
        seen.add(j.name)
    return order_jobs(out)


def order_jobs(jobs: list[JobSpec]) -> list[JobSpec]:
    """Stable topological order on `after` (only edges between jobs in this batch count)."""
    names = {j.name for j in jobs}
    done: list[JobSpec] = []
    left = list(jobs)
    while left:
        ready = [j for j in left if not (set(j.after) & names - {d.name for d in done})]
        if not ready:
            raise ValueError(f"cyclic 'after' between {[j.name for j in left]}")
        done.append(ready[0])
        left.remove(ready[0])
    return done


def archive_arc(job: JobSpec, local: str) -> str:
    return f"dancecap/work/runpod/archives/{job.name}/{Path(local).name}"


def effective_inputs(job: JobSpec, batch: list[JobSpec]) -> list[tuple[str, str]]:
    """inputs (+ inputs_if_alone when none of the job's `after` jobs is in the batch) + archives at their bundle arc."""
    names = {j.name for j in batch}
    extra = job.inputs_if_alone if not (set(job.after) & names) else []
    return job.inputs + extra + [(a, archive_arc(job, a)) for a, _ in job.archives]


def validation_job(frames: int = 180) -> JobSpec:
    """Pod self-test: dancecap imports on Linux + MAMMA end-to-end on the synthetic couple (ground truth)."""
    gt = "dancecap/work/synthetic_mamma/gt120/smplx"
    out = "dancecap/work/synthetic_mamma/noise0"
    imports = ("python -c \"import dancecap.mamma_dyn, dancecap.room, dancecap.room_mesh, dancecap.avatar_texture, "
               "dancecap.env as e; print('dancecap imports ok; is_pod', e.is_pod(), 'python', e.python_exe(), "
               "'mamma', e.mamma_root(), e.mamma_root().exists(), 'data', e.mamma_data(), e.mamma_data().exists())\"")
    return parse_job({
        "schema": SCHEMA, "name": "validate", "take": "synthetic_mamma", "max_minutes": 25,
        "needs": ["mamma", "room", "texture"],
        "inputs": [f"{gt}/lead.npz", f"{gt}/follow.npz"],
        "commands": [{"name": "imports", "cmd": imports, "timeout_min": 3},
                     {"name": "mamma_synthetic", "timeout_min": 25,
                      "cmd": f"python -m dancecap mamma --validate-synthetic --frames {frames}"}],
        "outputs": [f"{out}/report.json", f"{out}/smplx/", f"{out}/mamma/logs/"],
        "verify": [f"{out}/report.json", f"{out}/smplx/lead.npz", f"{out}/smplx/follow.npz"],
        "extract": "session",
    }, "builtin:validate")


# ---------------------------------------------------------------------------------------------- bundle


@dataclass
class BundleFile:
    arc: str          # workspace-relative posix path on the pod
    local: str
    size: int
    mtime_ns: int
    link: str | None = None  # identical content already stored under this arc (tar hard link)


def _excluded(arc: str, patterns) -> bool:
    parts = arc.split("/")
    for pat in patterns:
        if "/" in pat:
            if fnmatch.fnmatch(arc, pat) or arc.startswith(pat.rstrip("/") + "/"):
                return True
        elif any(fnmatch.fnmatch(x, pat) for x in parts):
            return True
    return False


BASE_FEATURES = {"base", "check"}   # what every session bootstraps; tool jobs (needs: []) use only these


def wheel_policy(jobs: list[JobSpec] | None) -> dict:
    """Which cached wheel dirs a batch uploads (slim bundles, 2026-10-21). The 330 MB pywheels dir went into EVERY bundle
    and agents opted out job by job: at the measured home uplink (median 2.1 MB/s, p10 0.73 MB/s) that is 2.6-7.5 min of
    billed upload per session, while a pod pulls the same wheels from PyPI in seconds.
    - pywheels: only for batches with a bootstrap feature beyond base/check (their long installs keep the offline-first
      cache: one host once stalled 20 min on PyPI), or a job with "bundle_pywheels": true. DANCECAP_BUNDLE_PYWHEELS=1/0
      forces it on/off.
    - wheelhouse (19 MB of compiled CUDA wheels): only when a feature beyond base/check is needed.
    jobs=None (code_items() without a batch, the other modules' callers): everything, as before."""
    if jobs is None:
        return {"pywheels": True, "wheelhouse": True, "why": "no batch given (legacy default)"}
    feats = {n for j in jobs for n in j.needs} - BASE_FEATURES - LIGHT_FEATURES
    forced = os.environ.get("DANCECAP_BUNDLE_PYWHEELS")
    asked = [j.name for j in jobs if j.bundle_pywheels]
    refused = jobs and all(j.bundle_pywheels is False for j in jobs)
    if forced in ("0", "1"):
        pyw, why = forced == "1", f"DANCECAP_BUNDLE_PYWHEELS={forced}"
    elif asked:
        pyw, why = True, f"asked by {asked}"
    elif refused:
        pyw, why = False, "every job sets bundle_pywheels false"
    else:
        pyw, why = bool(feats), (f"features {sorted(feats)}" if feats else "base/check only: the pod uses PyPI")
    return {"pywheels": pyw, "wheelhouse": bool(feats), "why": why}


def code_items(jobs: list[JobSpec] | None = None) -> list[tuple[Path, str]]:
    items = [(PACKAGE_ROOT / "dancecap", "dancecap/dancecap"), (PACKAGE_ROOT / "takes", "dancecap/takes"),
             (PACKAGE_ROOT / "remote", "dancecap/remote"), (PACKAGE_ROOT / "CONTRACTS.md", "dancecap/CONTRACTS.md")]
    root = env.mamma_root()
    for p in sorted(root.iterdir()) if root.exists() else []:
        if p.name in MAMMA_SKIP or p.name.startswith("."):
            continue
        items.append((p, f"mamma-turing/{p.name}"))
    pol = wheel_policy(jobs)
    if pol["wheelhouse"] and WHEELHOUSE.exists():
        items.append((WHEELHOUSE, f"dancecap/work/runpod/wheelhouse/{WHEEL_KEY}"))
    if pol["pywheels"] and PYWHEELS.exists() and any(PYWHEELS.iterdir()):
        items.append((PYWHEELS, "dancecap/work/runpod/pywheels/cp311-linux"))
    return items


def model_items() -> list[tuple[Path, str]]:
    """MAMMA's ma_3d builds male + female + neutral locked-head models (utils_smplx.get_smplx_models) even though
    dancecap fits are neutral, so all three npz must exist on the pod (~109 MB each)."""
    bm = env.mamma_data() / "body_models"
    lh = [(bm / "smplx_locked_head" / "smplx" / f"SMPLX_{g}.npz",
           f"mamma/data/body_models/smplx_locked_head/smplx/SMPLX_{g}.npz") for g in ("FEMALE", "MALE", "NEUTRAL")]
    return lh + [
            (bm / "downsampled_verts" / "verts_512.pkl", "mamma/data/body_models/downsampled_verts/verts_512.pkl"),
            (bm / "downsampled_verts" / "smplx_512_body_parts.npz",
             "mamma/data/body_models/downsampled_verts/smplx_512_body_parts.npz"),
            (WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz", "smplx/SMPLX_NEUTRAL.npz")]


def expand(items: list[tuple[Path | str, str]], exclude=DEFAULT_EXCLUDE) -> list[BundleFile]:
    """(local file|dir, workspace-relative arc) -> sorted file list. Same arc twice: fine when it is the same local
    file (jobs often list the shared body models), an error when it is not."""
    files: dict[str, BundleFile] = {}
    for local, arc in items:
        local = _local(local)
        arc = ws_rel(arc).rstrip("/")
        if not local.exists():
            raise FileNotFoundError(f"bundle input missing: {local}")
        walk = [(local, arc)] if local.is_file() else [
            (f, f"{arc}/{f.relative_to(local).as_posix()}") for f in sorted(local.rglob("*")) if f.is_file()]
        for f, a in walk:
            if _excluded(a, exclude):
                continue
            st = f.stat()
            old = files.get(a)
            if old:
                if Path(old.local).resolve() != f.resolve():
                    raise ValueError(f"bundle conflict at {a}: {old.local} vs {f}")
                continue
            files[a] = BundleFile(a, str(f), st.st_size, st.st_mtime_ns)
    out = sorted(files.values(), key=lambda b: b.arc)
    _dedupe(out)
    return out


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _dedupe(files: list[BundleFile], min_bytes: int = 1 << 20) -> None:
    """Large files with identical content are stored once; later copies become hard links to the first."""
    by_size: dict[int, list[BundleFile]] = {}
    for f in files:
        if f.size >= min_bytes:
            by_size.setdefault(f.size, []).append(f)
    for group in by_size.values():
        if len(group) < 2:
            continue
        first: dict[str, str] = {}
        for f in group:  # files are sorted by arc, so the first arc of each content is the stored one
            h = _sha256(Path(f.local))
            if h in first:
                f.link = first[h]
            else:
                first[h] = f.arc


MODEL_FEATURES = {"mamma", "texture"}  # stages that read the SMPL-X / MAMMA body models


def bundle_files(jobs: list[JobSpec], code: bool = True, models: bool | None = None) -> list[BundleFile]:
    """models=None: the body models go in only when some job needs a feature in MODEL_FEATURES."""
    items: list[tuple[Path | str, str]] = []
    if code:
        items += code_items(jobs)
    if models or (models is None and any(set(j.needs) & MODEL_FEATURES for j in jobs)):
        items += model_items()
    excl = set(DEFAULT_EXCLUDE)
    for j in jobs:
        items += effective_inputs(j, jobs)
        excl |= set(j.exclude)
    return expand(items, tuple(sorted(excl)))


def bundle_key(files: list[BundleFile]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.arc}|{f.size}|{f.mtime_ns}|{f.link}\n".encode())
    return h.hexdigest()[:12]


def build_bundle(files: list[BundleFile], out_dir: Path = BUNDLES, compresslevel: int = 6) -> tuple[Path, dict]:
    """tar.gz of the files at their workspace-relative paths (extract with tar -xzf <b> -C <workspace>).

    Content-keyed: an existing bundle with the same key is reused. Shell scripts are written with LF line endings
    (core.autocrlf checkouts) and mode 755. Returns (path, manifest)."""
    key = bundle_key(files)
    out = out_dir / f"bundle_{key}.tar.gz"
    man_p = out.with_suffix("").with_suffix(".json")
    if out.exists() and man_p.exists():
        man = json.loads(man_p.read_text())
        if man.get("key") == key and man.get("bytes") == out.stat().st_size:
            return out, man
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    t0 = time.time()
    with tarfile.open(tmp, "w:gz", compresslevel=compresslevel, format=tarfile.GNU_FORMAT) as tar:
        for f in files:
            ti = tarfile.TarInfo(f.arc)
            ti.mtime = int(f.mtime_ns // 1_000_000_000)
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = "root"
            if f.link:
                ti.type, ti.linkname, ti.mode = tarfile.LNKTYPE, f.link, 0o644
                tar.addfile(ti)
                continue
            data = None
            if f.arc.endswith(".sh"):
                data = Path(f.local).read_bytes().replace(b"\r\n", b"\n")
                ti.mode, ti.size = 0o755, len(data)
                tar.addfile(ti, io.BytesIO(data))
                continue
            ti.mode, ti.size = 0o644, f.size
            with open(f.local, "rb") as fh:
                tar.addfile(ti, fh)
    os.replace(tmp, out)
    groups: dict[str, int] = {}
    for f in files:
        g = f.arc.split("/")[0] if not f.arc.startswith("dancecap/work/") else "/".join(f.arc.split("/")[:4])
        groups[g] = groups.get(g, 0) + (0 if f.link else f.size)
    man = {"key": key, "path": str(out), "bytes": out.stat().st_size, "sha256": _sha256(out),
           "raw_bytes": sum(f.size for f in files if not f.link), "n_files": len(files),
           "links": {f.arc: f.link for f in files if f.link}, "groups_raw_bytes": groups,
           "build_s": round(time.time() - t0, 1), "files": [asdict(f) for f in files]}
    man_p.write_text(json.dumps(man, indent=1))
    prune_bundles(out_dir, keep=3)
    return out, man


def prune_bundles(out_dir: Path = BUNDLES, keep: int = 3) -> list[str]:
    """Drop all but the `keep` newest bundles (each is 100-400 MB; they are rebuilt from the tree in seconds)."""
    olds = sorted(out_dir.glob("bundle_*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True)[keep:]
    for p in olds:
        p.unlink(missing_ok=True)
        p.with_suffix("").with_suffix(".json").unlink(missing_ok=True)
    return [p.name for p in olds]


# ---------------------------------------------------------------------------------------------- session


def estimate_minutes(jobs: list[JobSpec], upload_bytes: int, cached_wheels: bool | None = None,
                     rates: dict | None = None) -> dict:
    """Pod minutes for a batch. rates (optional, progress_eta.load_rates()): adds a history-based estimate
    {"history": {p50_min, p90_min}, "p90_total"}; without it the result is exactly as before."""
    feats = needed_features(jobs)
    cached = WHEELHOUSE.exists() and any(WHEELHOUSE.glob("*.whl")) if cached_wheels is None else cached_wheels
    table = BOOTSTRAP_MIN_CACHED if cached else BOOTSTRAP_MIN
    boot = sum(table[f] for f in feats)
    up = upload_bytes / 1e6 / UPLOAD_MBPS / 60
    run = sum(j.max_minutes for j in jobs)
    margin = 10.0
    out = {"bootstrap": boot, "upload": round(up, 1), "jobs": run, "download": 3.0, "margin": margin,
           "total": round(boot + up + run + 3.0 + margin, 1), "wheels_cached": bool(cached)}
    if rates is not None:
        try:
            from . import progress_eta
            h = progress_eta.estimate_session([asdict(j) for j in jobs], upload_bytes, feats, 0.0, rates=rates,
                                              wheels_cached=bool(cached))
            out["history"] = {"p50_min": h["p50_min"], "p90_min": h["p90_min"], "phases": h["phases"]}
            out["p90_total"] = round(max(h["p90_min"], 0.0) + margin, 1)
        except Exception as e:  # history is advisory only
            out["history_error"] = str(e)[:200]
    return out


def needed_features(jobs: list[JobSpec]) -> list[str]:
    want = {"base", "check"} | {n for j in jobs for n in job_features(j)}
    return [f for f in FEATURES if f in want]


def pack_command(job: JobSpec, log_rel: str, out_path: str) -> str:
    """Pod shell: tar the job's existing outputs + its logs, report missing ones, print the archive's sha256."""
    paths = " ".join(shlex.quote(o.rstrip("/")) for o in job.outputs)
    return (f"set -u; mkdir -p {POD_OUT}; cd {POD_WS}; have=(); "
            f"for p in {paths}; do if [ -e \"$p\" ]; then have+=(\"$p\"); else echo \"MISSING $p\"; fi; done; "
            f"[ -d {shlex.quote(log_rel)} ] && have+=({shlex.quote(log_rel)}); "
            f"if command -v pigz >/dev/null; then Z='pigz -p 8'; else Z=gzip; fi; "
            f"tar -cf - \"${{have[@]}}\" | $Z > {out_path}.part && mv {out_path}.part {out_path}; "
            f"echo SIZE $(stat -c %s {out_path}); echo SHA $(sha256sum {out_path} | cut -d' ' -f1)")


SECRETS_DIR = "/workspace/.secrets"


def _secrets(d: dict) -> dict:
    """{ENV_NAME: local key file} (names upper-case identifiers; files are read only when the job runs)."""
    out = {}
    for k, v in (d or {}).items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(k)):
            raise ValueError(f"secret name must be an upper-case identifier: {k!r}")
        out[str(k)] = str(v)
    return out


def secret_exports(job: JobSpec) -> str:
    """Shell prefix exporting the job's secrets from the pod's 0600 files (never their values)."""
    return "".join(f"export {k}=\"$(cat {SECRETS_DIR}/{k})\"; " for k in job.secrets)


def push_secrets(pod, job: JobSpec) -> list[str]:
    """Read each secret's local key file and write it to the pod through ssh stdin -> the names pushed."""
    names = []
    for k, f in job.secrets.items():
        p = Path(os.path.expanduser(f))
        if not p.exists():
            raise FileNotFoundError(f"secret {k}: key file {p} does not exist")
        pod.put_secret(k, p.read_text().strip())
        names.append(k)
    return names


def watch_globs(job: JobSpec, cmd: Command) -> list[str]:
    """Pod log globs (workspace-relative) relayed while `cmd` runs: the job's `watch` + MAMMA's ma_3d log of every take a
    `mamma fit <take>` names. ma_3d writes its 'Iter. n of N' lines to that log only, so the trial's 18-min fit
    (2026-10-21) showed no counter at all (the puller's tail looked under the job's take, not the chunk take it fitted)."""
    out = list(job.watch)
    for m in re.finditer(r"\bmamma\s+fit\s+([A-Za-z0-9_.-]+)", cmd.cmd):
        out.append(f"dancecap/work/{m.group(1)}/mamma/logs/ma_3d_*.log")
    return list(dict.fromkeys(out))


# the relayed lines: MAMMA runs / every 10th iteration / early stop / total, SAM 3D Body chunk and frame lines (mawk-safe)
_WATCH_AWK = (r'/^(RUN: |Stopping early|Optimization took|sequence length:)/ {print p $0; fflush(); next} '
              r'/^Iter\. [0-9]+ of [0-9]+/ {n = $2 + 0; t = $4 + 0; if (n == 1 || n % 10 == 0 || n == t) '
              r'{print p "Iter. " n " of " t; fflush()}; next} '
              r'/(chunk|frame) [0-9]+\/[0-9]+/ {print p substr($0, 1, 240); fflush()}')


def watch_script(globs: list[str], every_s: int = 3, root: str = POD_WS) -> str:
    """Background relay for command_script: each log matching a glob that is written AFTER the command starts is followed
    (tail -F) and its progress lines printed as '[watch <file>] <line>' into the command's ssh stream (not into its pod
    log), where FrameProgress counts them. It ends with the command's shell ($$); logs already on the pod are not replayed."""
    if not globs:
        return ""
    pats = " ".join(f"{root}/{g}" for g in globs)   # unquoted on purpose: the shell expands the globs
    # Ubuntu's awk is mawk, which reads a pipe in whole blocks: without -W interactive the 2026-10-21 live probe got every
    # relayed line in one burst when the command ended (gawk, e.g. Git Bash's, reads lines as they come)
    return (f"_wm=$(mktemp); ( A=awk; awk -W version 2>&1 | grep -qi mawk && A='awk -W interactive'; seen=' '; "
            f"while kill -0 $$ 2>/dev/null; do for f in {pats}; do "
            f"case \"$seen\" in *\" $f \"*) continue;; esac; [ -f \"$f\" ] && [ \"$f\" -nt \"$_wm\" ] || continue; "
            f"seen=\"$seen$f \"; ( tail -n +1 -F --pid=$$ \"$f\" 2>/dev/null | "
            f"$A -v p=\"[watch $(basename \"$f\")] \" {shlex.quote(_WATCH_AWK)} ) & done; sleep {every_s}; done; "
            f"rm -f \"$_wm\" ) & ")


def command_script(job: JobSpec, cmd: Command, log_rel: str) -> str:
    """Pod shell for one command: skip when its .done marker exists; log with tee; exit code to <name>.rc."""
    logd = f"{POD_WS}/{log_rel}"
    envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in {"DANCECAP_POD": "1", "PYTHONUNBUFFERED": "1",
                                                                **_progress_env(job, cmd, log_rel), **job.env}.items())
    secs = int(cmd.timeout_min * 60)
    return (f"mkdir -p {logd}; if [ -f {logd}/{cmd.name}.done ]; then echo '[pod-runner] {job.name}/{cmd.name} "
            f"already done'; exit 0; fi; cd {POD_WS}/{job.cwd} || exit 97; export {envs}; {secret_exports(job)}set -o pipefail; "
            f"{watch_script(watch_globs(job, cmd))}"
            f"echo \"[pod-runner] $(date -u +%FT%TZ) start {job.name}/{cmd.name}\" | tee -a {logd}/{cmd.name}.log; "
            f"timeout --kill-after=30 {secs} bash -c {shlex.quote(cmd.cmd)} 2>&1 | tee -a {logd}/{cmd.name}.log; "
            f"rc=${{PIPESTATUS[0]}}; echo $rc > {logd}/{cmd.name}.rc; "
            f"echo \"[pod-runner] $(date -u +%FT%TZ) end {job.name}/{cmd.name} rc=$rc\" | tee -a {logd}/{cmd.name}.log; "
            f"[ $rc -eq 0 ] && date +%s > {logd}/{cmd.name}.done; exit $rc")


def _progress_env(job: JobSpec, cmd: Command, log_rel: str) -> dict:
    """dancecap.progress/1 for pod commands (docs/PIPELINE_PROGRESS.md 10.2): every command of a session appends to
    <session>/progress.jsonl on the pod (derived from log_rel '<...>/sessions/<sid>/pod_logs/<job>'); jobs that ignore
    the variables behave exactly as before, and the file is never packed (pack_command tars outputs + log_rel only)."""
    parts = log_rel.rstrip("/").split("/")
    if len(parts) < 3 or parts[-2] != "pod_logs":
        return {}
    sess = "/".join(parts[:-2])
    return {"DANCECAP_PROGRESS": f"{POD_WS}/{sess}/progress.jsonl", "DANCECAP_PROGRESS_JOB": job.name,
            "DANCECAP_PROGRESS_CMD": cmd.name, "DANCECAP_SID": parts[-3], "DANCECAP_TAKE": job.take}


# ---------------------------------------------------------------------------------------------- frame counter


FRAMES_FILE = "frames.json"   # <session>/progress/frames.json (format: FrameProgress)
# progress.LineParser adapters for command output (runner / bootstrap lines are the L0 parser's, not a command's progress)
FRAME_ADAPTERS = ("texture", "room", "hands", "loc", "chunks", "mamma_iter", "tqdm", "detect", "splats")
_WATCH_PREFIX = re.compile(r"^\[watch ([^\]]+)\] ")
_S3DB_CHUNK = re.compile(r"^\[sam3db_pod [\d:]+\] ([\w-]+): chunk (\d+)/(\d+) \((\d+) frames, (\d+) prompts\) in (\d+) s = "
                         r"([\d.]+) s per person-frame")
_S3DB_FRAME = re.compile(r"^\[sam3db_pod [\d:]+\] ([\w-]+): frame (\d+)/(\d+)\s*$")
_SEQ_LEN = re.compile(r"^sequence length:\s+(\d+)")
# the rate of the fastest GPU measured, per source: a pod running far slower says so (SLOW_X_WARN). 2026-10-21 trial: an A100-SXM
# ran SAM 3D Body at 2.0 s per person-frame, 5.6x the 4090's 0.355 s, for 29 min ($0.87) before anyone noticed.
REF_RATE_S = {"sam3db": {"rate_s": 0.355, "rate_unit": "person-frame", "gpu": "NVIDIA GeForce RTX 4090",
                         "source": "session 20261021-151934-batch"}}
SLOW_X_WARN = 2.0


class FrameProgress:
    """The frame counter of one pod command, from the lines it prints (remote.Pod.run on_line) and the logs the runner relays
    (watch_script). No job code changes: it reads what long jobs already print - SAM 3D Body's chunk / frame lines, MAMMA's
    'Iter. n of N' (relayed from the ma_3d log), tqdm bars, '[cameras] cam: d/t frames', '[texture] d/t views', room /
    splat steps (progress.LineParser adapters).

    <session>/progress/frames.json (atomic, at most every 2 s; read it, never write it):
      {"v": 1, "sid", "gpu", "updated": <epoch s>,
       "jobs": {"<job>/<cmd>": {"job", "cmd", "src": sam3db|mamma_iter|tqdm|loc|texture|..., "step", "stage",
                "unit": frames|iters|views|steps, "done", "total" (null = unknown), "rate_s" (s per rate_unit, last
                interval), "rate_avg_s" (since the step's first report), "rate_unit" (default = unit), "eta_s" (this step),
                "t0", "ts", "state": running|done, "gpu", "ref": {rate_s, rate_unit, gpu, source} | null,
                "slow_x" (rate_s / ref rate_s) | null, "hist": [[ts, done], ...] (<= 60), "frames" (MAMMA: sequence length)}}}
    Each update is also a throttled 'tick' event (job, cmd, step, done, total, unit, extra {rate_s, src}) in
    progress/runner.jsonl (start / end always, ticks >= 15 s apart). A rate SLOW_X_WARN x the reference is a session warning
    (state.json "warnings", kind slow_gpu)."""

    WRITE_MIN_S = 2.0
    EVENT_MIN_S = 15.0

    def __init__(self, ses: "Session", job: JobSpec, cmd: Command, gpu: str | None = None):
        from . import progress
        self.ses, self.job, self.cmd, self.gpu = ses, job, cmd, gpu
        self.parsers: dict[str, object] = {}
        self._LineParser = progress.LineParser
        self.key = f"{job.name}/{cmd.name}"
        self.entry: dict | None = None
        self.aside: dict[tuple, dict] = {}   # counts of the same step interrupted by another one (nested tqdm bars)
        self.last_write = self.last_event = 0.0
        self.frames = None
        self.warned = False

    # input ----------------------------------------------------------------------------------------------------
    def feed(self, line: str, now: float | None = None) -> None:
        try:
            self._feed(line.rstrip("\r\n"), time.time() if now is None else now)
        except Exception:  # noqa: BLE001 - a counter never ends a command
            pass

    def _parser(self, stream: str):
        p = self.parsers.get(stream)
        if p is None:
            p = self.parsers[stream] = self._LineParser(FRAME_ADAPTERS, job=self.job.name)
            p.cmd = self.cmd.name
        return p

    def _feed(self, line: str, now: float) -> None:
        stream = "stdout"
        m = _WATCH_PREFIX.match(line)
        if m:
            stream, line = m.group(1), line[m.end():]
        m = _SEQ_LEN.match(line)
        if m:
            self.frames = int(m.group(1))
            return
        m = _S3DB_CHUNK.match(line)
        if m:
            var, c, n, f, prompts, dt, rate = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                                               int(m.group(5)), float(m.group(6)), float(m.group(7)))
            e = self._entry(f"sam3db/{var}", "frames", "sam3db", now)
            e.setdefault("chunk_frames", f)
            # chunks reused from an earlier run print no line: count the chunks BEFORE this one as done at the first
            # chunk's size (the review saw 199/199 'done' when 2 of 3 chunks were reused), capped by the last chunk's size
            first = e.get("frames_done") is None
            e["frames_done"] = int(e.get("frames_done") or (e["chunk_frames"] * (c - 1) if first else 0)) + f
            e["chunks"] = [c, n]
            done = e["frames_done"]
            total = done if c >= n else max(done, e["chunk_frames"] * n)
            self._update(e, done, total, now, rate_s=rate, rate_unit="person-frame",
                         s_per_unit=dt / max(1, f), stage=f"chunk {c}/{n}")
            return
        m = _S3DB_FRAME.match(line)
        if m:
            done, total = int(m.group(2)), int(m.group(3))
            e = self._entry(f"sam3db/{m.group(1)}", "frames", "sam3db", now, done, total)
            if e is not None:
                self._update(e, done, total, now)
            return
        for ev in self._parser(stream).feed(line):
            if ev.get("done") is None or ev.get("kind") not in ("start", "tick", "end"):
                continue
            src = ev.get("src") or "log"
            step = str(ev.get("step") or src)
            e = self._entry(step if not ev.get("lane") else f"{step}@{ev['lane']}", ev.get("unit") or "frames", src, now,
                            int(ev["done"]), ev.get("total"), end=ev.get("kind") == "end")
            if e is None:
                continue
            if src == "mamma_iter":
                e["stage"], e["stages_total"] = step, len(MAMMA_RUNS)
                if self.frames:
                    e["frames"] = self.frames
            self._update(e, int(ev["done"]), ev.get("total"), now, end=ev.get("kind") == "end")

    # state ----------------------------------------------------------------------------------------------------
    def _entry(self, step: str, unit: str, src: str, now: float, done: int | None = None, total=None,
               end: bool = False) -> dict | None:
        """The entry this line counts into. A new one starts when the step or source changes, and also when the same step
        finished, went backwards or changed its total (a second tqdm bar in one command: the 2026-10-21 review saw every
        line after the first bar become an unthrottled 'end' + a forced frames.json write, shown as 'done' while running).
        None = the finished count's last line again (tqdm reprints it when the bar closes). A running count of the same
        step that another one interrupts (nested bars) is kept aside and resumed by its next line. An end at a smaller total
        ends the running count (MAMMA's 'Stopping early at iteration 30' sets total = 30 of 50; the live probe of 19:51
        counted it as a second bar)."""
        tot = int(total) if total is not None else None
        e = self.entry
        if e is not None and e["step"] == step and e["src"] == src:
            same_total = tot is None or e["total"] is None or tot == e["total"]
            if e["state"] == "running":
                if done is None or (same_total or end) and done >= e["done"]:
                    return e
                if not same_total:   # another bar of the same step: resumed when its next line comes
                    self.aside[(src, step, e["total"])] = e
                    while len(self.aside) > 4:
                        self._end(self.aside.pop(next(iter(self.aside))), now)
                    e = None
            elif done is not None and done == e["done"] and same_total:
                return None
        a = self.aside.pop((src, step, tot), None) if done is not None else None
        if a is not None and done >= a["done"]:
            self.entry = a
            return a
        if a is not None:
            self._end(a, now)
        if e is not None and e["state"] == "running":
            self._end(e, now)
        e = self.entry = {"job": self.job.name, "cmd": self.cmd.name, "src": src, "step": step, "unit": unit,
                          "done": 0, "total": None, "rate_s": None, "rate_avg_s": None, "rate_unit": unit,
                          "eta_s": None, "t0": round(now, 1), "ts": round(now, 1), "state": "running",
                          "gpu": self.gpu, "ref": REF_RATE_S.get(src), "slow_x": None, "hist": []}
        self._event("start", e, now)
        return e

    def _end(self, e: dict, now: float) -> None:
        """A count another one replaced before it reached its total: ended, once."""
        if e.get("state") == "running":
            e["state"], e["eta_s"] = "done", None
            self._event("end", e, now)

    def _update(self, e: dict, done: int, total, now: float, rate_s: float | None = None, rate_unit: str | None = None,
                s_per_unit: float | None = None, stage: str | None = None, end: bool = False) -> None:
        was = e["state"]
        hist = e["hist"]
        prev = hist[-1] if hist else None
        if prev is None or done != prev[1]:
            hist.append([round(now, 1), done])
            if len(hist) > 60:
                del hist[1:len(hist) - 59]   # keep the first point (the average's start) and the latest ones
        if rate_s is None and prev is not None and done > prev[1]:
            rate_s = (now - prev[0]) / (done - prev[1])
        if rate_s is not None:
            e["rate_s"] = round(rate_s, 4)
        first = hist[0]
        if done > first[1] and now > first[0]:
            e["rate_avg_s"] = round((now - first[0]) / (done - first[1]), 4)
        if rate_unit:
            e["rate_unit"] = rate_unit
        if stage:
            e["stage"] = stage
        e.update(done=done, total=int(total) if total is not None else e.get("total"), ts=round(now, 1))
        spu = s_per_unit if s_per_unit is not None else (e.get("rate_avg_s") if e.get("rate_unit") == e["unit"] else None)
        if e["total"] is not None and spu:
            e["eta_s"] = round(max(0, e["total"] - done) * spu, 1)
        if end or (e["total"] is not None and done >= e["total"]):
            e["state"], e["eta_s"] = "done", 0.0
        ref = e.get("ref")
        if ref and e.get("rate_s") and e.get("rate_unit") == ref.get("rate_unit"):
            e["slow_x"] = round(e["rate_s"] / ref["rate_s"], 2)
            if e["slow_x"] >= SLOW_X_WARN and not self.warned:
                self.warned = True
                from .remote import short_gpu
                self.ses.warn("slow_gpu", f"{self.key}: {e['rate_s']:.3f} s per {ref['rate_unit']} on "
                                          f"{short_gpu(self.gpu)}, {e['slow_x']:.1f}x the {ref['rate_s']} s measured "
                                          f"on {short_gpu(ref['gpu'])} ({ref['source']})",
                              job=self.job.name, cmd=self.cmd.name, slow_x=e["slow_x"])
        finished = e["state"] == "done" and was == "running"   # 'end' + a forced write once per count, never per line
        if finished or e["state"] == "running":
            self._event("end" if finished else "tick", e, now)
        self._write(now, force=finished)

    # output ---------------------------------------------------------------------------------------------------
    def _event(self, kind: str, e: dict, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if kind == "tick" and now - self.last_event < self.EVENT_MIN_S:
            return
        self.last_event = now
        try:
            self.ses.writer().emit(kind, e["step"], job=e["job"], cmd=e["cmd"], src="frames", done=e.get("done"),
                                   total=e.get("total"),
                                   unit=e["unit"] if e["unit"] in ("frames", "iters", "views", "steps") else None,
                                   extra={"counter": e["src"], "rate_s": e.get("rate_s"),
                                          "rate_unit": e.get("rate_unit"), "eta_s": e.get("eta_s"),
                                          "slow_x": e.get("slow_x")})
        except Exception:  # noqa: BLE001
            pass

    def _write(self, now: float, force: bool = False) -> None:
        if self.entry is None or (not force and now - self.last_write < self.WRITE_MIN_S):
            return
        self.last_write = now
        try:
            from . import progress
            p = self.ses.dir / "progress" / FRAMES_FILE
            doc = self._doc()
            doc.update(v=1, sid=self.ses.sid, gpu=self.gpu, updated=round(now, 1))
            doc.setdefault("jobs", {})[self.key] = self.entry
            progress.write_json_atomic(p, doc)
        except Exception:  # noqa: BLE001
            pass

    def close(self, rc=None) -> None:
        """The command ended (rc as _run_job records it): a count that did not reach its total is written as 'ended' (rc 0)
        or 'failed', so a stale 'running' never outlives its command - also one a dead runner left in frames.json for this
        command when the rerun printed no counter line (resume)."""
        now = time.time()
        for a in self.aside.values():
            self._end(a, now)
        self.aside.clear()
        e = self.entry
        if e is None:
            old = (self._doc().get("jobs") or {}).get(self.key)
            if not (isinstance(old, dict) and old.get("state") == "running"):
                return
            e = self.entry = old
        if e.get("state") not in ("done", "ended", "failed"):
            e["state"] = "ended" if rc == 0 else "failed"
            e["eta_s"] = None
            self._event("end" if rc == 0 else "fail", e)
        e["rc"] = rc if isinstance(rc, int) else (str(rc)[:80] if rc is not None else None)
        self._write(now, force=True)

    def _doc(self) -> dict:
        from . import progress
        doc = getattr(self.ses, "_frames_doc", None)
        if doc is None:
            doc = self.ses._frames_doc = progress.read_json(self.ses.dir / "progress" / FRAMES_FILE) or {"v": 1, "jobs": {}}
        return doc


def end_stale_frames(ses: "Session", why: str = "the runner that counted it is gone") -> int:
    """A resumed session: every frames.json entry still 'running' belongs to a runner that died mid-command (its pod is gone
    too) -> 'ended' with no ETA. Returns how many. Never fatal."""
    try:
        from . import progress
        p = ses.dir / "progress" / FRAMES_FILE
        doc = progress.read_json(p)
        if not doc:
            return 0
        n = 0
        for e in (doc.get("jobs") or {}).values():
            if isinstance(e, dict) and e.get("state") == "running":
                e.update(state="ended", eta_s=None, note=why)
                n += 1
        if n:
            doc["updated"] = round(time.time(), 1)
            progress.write_json_atomic(p, doc)
        ses._frames_doc = doc
        return n
    except Exception:  # noqa: BLE001
        return 0


MAMMA_RUNS = (50, 500, 100, 25)   # = progress.MAMMA_RUN_TOTALS: ma_3d's four optimisation runs


def safe_extract(archive: Path, dest_outputs: Path, dest_logs: Path, allowed: list[str], log_prefix: str) -> list[str]:
    """Extract only regular files/dirs under the allowed workspace-relative prefixes (outputs) or the log dir."""
    got = []
    with tarfile.open(archive, "r:*") as tar:
        for m in tar.getmembers():
            name = m.name[2:] if m.name.startswith("./") else m.name
            if m.issym() or m.islnk() or m.isdev() or ".." in PurePosixPath(name).parts or name.startswith("/"):
                continue
            is_log = name == log_prefix or name.startswith(log_prefix + "/")
            ok = is_log or any(name == a.rstrip("/") or name.startswith(a.rstrip("/") + "/") for a in allowed)
            if not ok:
                continue
            root = dest_logs if is_log else dest_outputs
            target = root / name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                continue
            tmp = target.with_name(target.name + ".part")
            with open(tmp, "wb") as fh:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    fh.write(b)
            os.replace(tmp, target)
            os.utime(target, (m.mtime, m.mtime))
            got.append(name)
    return got


class Session:
    def __init__(self, sid: str | None = None, label: str = "batch"):
        self.sid = sid or f"{datetime.now():%Y%m%d-%H%M%S}-{label}"
        self.dir = SESSIONS / self.sid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_p = self.dir / "state.json"
        self.state = json.loads(self.state_p.read_text()) if self.state_p.exists() else {
            "sid": self.sid, "created": datetime.now().isoformat(timespec="seconds"), "timings": [], "jobs": {}}
        self.runner_log = self.dir / "runner.log"

    def save(self):
        self.state_p.write_text(json.dumps(self.state, indent=1, default=str))

    def log(self, msg: str):
        line = f"[pod-runner {datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.runner_log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def timed(self, step: str, fn, **extra):
        t0 = time.time()
        self.log(f"{step} ...")
        # progress events only: keys event() takes positionally / sets itself never come from a caller's extra
        ev_extra = {k: v for k, v in extra.items() if k not in ("seconds", "msg", "kind", "step")}
        self.event("start", step, **ev_extra)
        ok = False
        try:
            r = fn()
            ok = True
            return r
        finally:
            dt = round(time.time() - t0, 1)
            self.state["timings"].append({"step": step, "seconds": dt, "ok": ok, "start": round(t0, 1), **extra})
            self.save()
            self.log(f"{step} {'done' if ok else 'FAILED'} in {dt:.1f}s")
            self.event("end" if ok else "fail", step, seconds=dt, **ev_extra)

    # dancecap.progress/1 (L1, docs/PIPELINE_PROGRESS.md 10.2): machine-readable runner phases next to runner.log.
    # Additive and never fatal: a failing event write is swallowed.
    def event(self, kind: str, step: str, msg: str | None = None, **extra) -> None:
        try:
            w = self.writer()
            job, name = (step.split("/", 1) if "/" in step else ("_session", step))
            extra = {k: v for k, v in extra.items() if k in ("seconds", "mb", "job", "features", "bytes", "done",
                                                              "total", "level")}
            done, total = extra.pop("done", None), extra.pop("total", None)
            w.emit(kind, name, job=job, msg=msg, done=done, total=total,
                   unit="bytes" if total is not None else None, extra={"runner_step": step, **extra})
        except Exception:
            pass

    def writer(self):
        """The progress.Writer of progress/runner.jsonl (one per session object)."""
        w = getattr(self, "_pw", None)
        if w is None:
            from . import progress
            w = self._pw = progress.Writer(self.dir / "progress" / "runner.jsonl", job="_session", sid=self.sid)
        return w

    def warn(self, kind: str, msg: str, **extra) -> None:
        """A warning the dashboard shows: state.json "warnings" [{kind, level, msg, ts, ...}] (+ a runner.log line and a note
        event). One per (kind, msg)."""
        ws = self.state.setdefault("warnings", [])
        if any(w.get("kind") == kind and w.get("msg") == msg for w in ws):
            return
        ws.append({"kind": kind, "level": "warn", "msg": msg, "ts": round(time.time(), 1), **extra})
        self.save()
        self.log(f"WARNING ({kind}): {msg}")
        self.event("note", kind, msg=msg, level="warn")

    def write_runner_info(self, label: str, jobs: list[str]) -> None:
        """runner.json: lets a monitor check the runner process is alive (no argv, no environment)."""
        try:
            p = self.dir / "runner.json"
            prev = json.loads(p.read_text()) if p.exists() else {}
            info = {"sid": self.sid, "pid": os.getpid(), "started": round(time.time(), 1),
                    "attempt": int(prev.get("attempt") or 0) + 1, "label": label, "jobs": jobs}
            from . import progress
            progress.write_json_atomic(p, info)
        except Exception:
            pass

    def log_rel(self, job: str) -> str:
        return f"dancecap/work/runpod/sessions/{self.sid}/pod_logs/{job}"


def _retry(fn, tries: int = 3, wait_s: float = 10, what: str = "step"):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # transient ssh/scp failures: retry the idempotent step
            if i == tries - 1:
                raise
            print(f"[pod-runner] {what} failed ({e}); retry {i + 2}/{tries} in {wait_s:.0f}s", flush=True)
            time.sleep(wait_s)


def run_session(jobs: list[JobSpec], sid: str | None = None, label: str = "batch", max_cost: float | None = None,
                gpus: list[str] | None = None, force: bool = False, dry_run: bool = False,
                max_minutes: float | None = None, clouds: list[str] | None = None) -> dict:
    """clouds: e.g. ["SECURE"] when the bundle holds licence-restricted files (MANO/FLAME); default remote.CLOUDS."""
    from . import remote
    ses = Session(sid, label)
    for k in ("error", "bootstrap_rc", "gpu_choice", "warnings"):   # a resumed session starts clean
        ses.state.pop(k, None)
    end_stale_frames(ses)   # counts a dead runner left 'running' (the review: one outlived its command on resume)
    tag = analysis_tag()
    if tag:
        ses.state["analysis"] = tag   # the monitor groups this session under that analysis run (pipeline/analyze.py)
        if not dry_run:   # the run's pod limit (analyze --max-cost / analysis limit) caps every session tagged with it
            try:
                from .pipeline import analyze
                left = analyze.budget_left(tag["run"])
            except Exception:  # noqa: BLE001 - the dashboard's bookkeeping never blocks a session
                left = None
            if left is not None:
                if left <= 0:
                    ses.state["error"] = (f"BudgetError: analysis run {tag['run']} has no pod limit left "
                                          f"(raise it with `python -m dancecap.pipeline analysis limit {tag['run']} USD`)")
                    ses.log(ses.state["error"])
                    ses.save()
                    return _finish(ses, None)
                capped = max_cost is None or left < float(max_cost)
                max_cost = left if capped else float(max_cost)
                ses.state["run_limit_left"] = round(left, 4)
                # say which one set the max: the live check's --max-cost 1.2 read "what is left of the run's limit" at $3.00 left
                ses.log(f"session max ${max_cost:.2f} " + (f"(what is left of analysis run {tag['run']}'s pod limit)" if capped
                        else f"(--max-cost; analysis run {tag['run']} has ${left:.2f} of its pod limit left)"))
    ses.write_runner_info(label, [j.name for j in jobs])
    if not dry_run:
        _monitor_up(ses)
        remote.record_cap_in_use(ses.sid)
    if not gpus:   # fastest first among the cards with enough memory for every job (job_vram_gb: 24 GB unless measured)
        need = max([job_vram_gb(j) for j in jobs] or [DEFAULT_VRAM_GB])
        gpus = remote.gpus_for([need])   # None (nothing that big in the ladder) = remote.GPU_PREFERENCE
        ses.state["gpu_need_gb"] = need
        ses.log(f"GPUs for {need:.0f} GB, fastest first: "
                f"{', '.join(remote.short_gpu(g) for g in (gpus or remote.GPU_PREFERENCE))}")
    ses.state["gpus"] = list(gpus or remote.GPU_PREFERENCE)
    todo = []
    for j in jobs:
        fp = j.fingerprint()
        prev = ses.state["jobs"].get(j.name, {})
        if not force and prev.get("status") == "ok" and prev.get("fingerprint") == fp:
            ses.log(f"{j.name}: already done in this session (fingerprint {fp}); skipped (--force to redo)")
            continue
        ses.state["jobs"][j.name] = {"status": "pending", "fingerprint": fp, "spec": asdict(j)}
        todo.append(j)
    ses.save()
    if not todo:
        ses.log("nothing to do")
        return _finish(ses, None)

    files = ses.timed("bundle_scan", lambda: bundle_files(todo))
    bundle, man = ses.timed("bundle_build", lambda: build_bundle(files))
    ses.state["bundle"] = {k: man[k] for k in ("key", "path", "bytes", "raw_bytes", "n_files", "sha256",
                                                "groups_raw_bytes", "links", "build_s")}
    est = estimate_minutes(todo, man["bytes"])
    mm = float(max_minutes or est["total"])
    ses.state["estimate_min"] = est
    ses.state["max_minutes"] = mm
    feats = needed_features(todo)
    ses.log(f"bundle {bundle.name}: {man['bytes'] / 1e6:.1f} MB ({man['n_files']} files, raw "
            f"{man['raw_bytes'] / 1e6:.1f} MB, {len(man['links'])} hard links); upload ~{est['upload']:.1f} min; "
            f"features {feats}; pod max {mm:.0f} min")
    ses.save()
    if dry_run:
        ses.log("dry run: no pod")
        return _finish(ses, None)

    pod_info = {}
    try:
        with remote.pod_session(f"{label}", max_minutes=mm, gpus=gpus, max_cost=max_cost, clouds=clouds) as pod:
            pod_info.update({"id": pod.id, "gpu": pod.gpu, "cost_per_hr": pod.cost_per_hr, "host": pod.host,
                             "killswitch": pod.killswitch, "started": time.time()})
            # for the monitor (additive): ssh port (optional read-only legacy probe) and the kill-switch estimate
            # (the puller replaces it with the exact /workspace/.dancecap_kill_at)
            pod_info.update({"port": getattr(pod, "port", None), "kill_at_est": round(time.time() + mm * 60, 1)})
            choice = getattr(pod, "choice", None)
            if choice:
                pod_info["choice"] = choice
            ses.state["pod"] = pod_info
            ses.save()
            ses.event("note", "pod", msg=f"pod {pod.id} {pod.gpu} ${pod.cost_per_hr:.2f}/h ready")
            if choice and choice.get("warning"):   # never a slower card without a word (owner: speed over cost)
                ses.warn("gpu_choice", choice["warning"], excluded=choice.get("excluded_faster"))
            puller = _start_puller(ses, pod, todo, bundle, man)
            try:
                _pod_steps(ses, pod, todo, bundle, man, feats)
            finally:
                _stop_puller(puller)
    except Exception as e:
        ses.state["error"] = f"{type(e).__name__}: {e}"[:2000]
        if getattr(e, "choice", None) and not pod_info:   # no pod: what was tried, so the dashboard can say why
            ses.state["gpu_choice"] = e.choice
        ses.log(f"session error: {ses.state['error']}")
    finally:
        ses.save()
    return _finish(ses, pod_info.get("id"))


# ---------------------------------------------------------------------------------------------- monitor, snapshots, upload


ANALYSIS_RUN_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9_]{1,48}$")


def analysis_tag() -> dict | None:
    """DANCECAP_ANALYSIS=<run id> (+ DANCECAP_ANALYSIS_STEP=<step>): set by `pipeline analyze` for the sessions it starts,
    or by an agent driving a step of that analysis by hand."""
    run = os.environ.get("DANCECAP_ANALYSIS")
    if not run or not ANALYSIS_RUN_RE.match(run):
        return None
    step = os.environ.get("DANCECAP_ANALYSIS_STEP")
    return {"run": run, "step": step if step and re.fullmatch(r"[a-z0-9_]{1,40}", step) else None}


def _monitor_up(ses: Session) -> None:
    """The dashboard runs whenever a pod session does (owner, 2026-10-21): start it headless if needed, leased to this
    process. Never fatal."""
    try:
        from .pipeline import monitor
        run = os.environ.get("DANCECAP_ANALYSIS")   # a session of an analysis keeps the window on that analysis
        r = monitor.ensure_running(f"remote run {ses.sid}",
                                   route=f"#/analysis/{run}" if run else f"#/runpod/session/{ses.sid}")
        if r.get("state") in ("started", "starting"):
            ses.log("monitor: started the Pipeline Monitor in the background (python -m dancecap.pipeline monitor "
                    "--launch opens its window)")
    except Exception:  # noqa: BLE001
        pass


def snapshot(ses: Session, event: str, label: str, final: bool = False) -> None:
    """A dashboard snapshot at a progress point (work/pipeline/snapshots/<analysis run or sid>/): rendered by a detached
    background process, never in this one's way. Never fatal."""
    try:
        from .pipeline import snapshots
        tag = ses.state.get("analysis") or {}
        if tag.get("run"):   # several sessions share an analysis's snapshot folder: say which one
            label = f"{ses.sid[16:] or ses.sid}{' (' + tag['step'] + ')' if tag.get('step') else ''}: {label}"
            event = f"{tag.get('step') or 'pod'}_{event}"
        snapshots.request(tag.get("run") or ses.sid, event, label, sid=ses.sid, final=final)
    except Exception:  # noqa: BLE001
        pass


def upload_streams(nbytes: int) -> int:
    """Parallel ssh streams for a bundle upload: DANCECAP_UPLOAD_STREAMS (default 4) from 24 MB up, else 1 (plain scp).
    On the shared home uplink each stream gets about the same share when one stream is slow (FARM.md: one stream
    0.08-0.13 MB/s, four 0.42 MB/s; 2026-10-19..21: 5 single-stream uploads timed out)."""
    try:
        n = int(os.environ.get("DANCECAP_UPLOAD_STREAMS", "4"))
    except ValueError:
        n = 4
    return max(1, min(n, 8)) if nbytes >= 24 << 20 else 1


class UploadProgress:
    """<session>/progress/upload.json {total, sent, streams, t0, ts, mbps, done, history[[t, sent]...]}: written by the
    uploader itself (exact bytes; the monitor shows rate, ETA and a rate history), at most once a second."""

    def __init__(self, sdir: Path, total: int, streams: int):
        self.p = sdir / "progress" / "upload.json"
        self.total, self.streams = int(total), streams
        self.t0 = time.time()
        self.last = 0.0
        self.hist: list[list[float]] = [[round(self.t0, 1), 0]]
        self.sent = 0
        self._write(False)

    def __call__(self, rel: str, done: int, size: int) -> None:
        self.sent = done
        now = time.time()
        if now - self.last >= 1.0 or done >= size:
            self.last = now
            if not self.hist or now - self.hist[-1][0] >= 5 or done >= size:
                self.hist.append([round(now, 1), done])
                if len(self.hist) > 400:
                    self.hist = self.hist[::2]
            self._write(done >= size)

    def _write(self, done: bool) -> None:
        try:
            el = max(1e-3, time.time() - self.t0)
            obj = {"total": self.total, "sent": self.sent, "streams": self.streams, "t0": round(self.t0, 1),
                   "ts": round(time.time(), 1), "mbps": round(self.sent / 1e6 / el, 3), "done": done, "history": self.hist}
            self.p.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.p.with_name(f"upload.json.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(obj))
            os.replace(tmp, self.p)
        except OSError:
            pass


def striped_upload(pod, local: Path, remote_dir: str, streams: int, progress=None) -> dict:
    """Upload `local` to <remote_dir>/<name> over `streams` parallel ssh connections, resumable per stripe and sha256-checked
    before the atomic rename (farm.volume.Link.push_file: the volume uploader the farm uses since 2026-10-08)."""
    from .farm import volume
    return volume.Link(pod, root=remote_dir).push_file(Path(local), Path(local).name, progress=progress, streams=streams)


def components(jobs: list[JobSpec]) -> list[list[JobSpec]]:
    """Independent groups of a batch: jobs linked by `after` (either way) stay together (one pod); groups share nothing
    and can run on separate pods at the same time (`remote run --parallel`)."""
    names = [j.name for j in jobs]
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for j in jobs:
        for a in j.after:
            if a in parent:
                parent[find(a)] = find(j.name)
    groups: dict[str, list[JobSpec]] = {}
    for j in jobs:
        groups.setdefault(find(j.name), []).append(j)
    return [order_jobs(g) for g in groups.values()]


def run_parallel(jobs: list[JobSpec], label: str = "batch", max_cost: float | None = None, gpus: list[str] | None = None,
                 force: bool = False, dry_run: bool = False, max_minutes: float | None = None,
                 clouds: list[str] | None = None, max_pods: int = 6) -> dict:
    """Owner, 2026-10-21: speed over cost. Independent job groups (components) run as separate sessions on separate pods at
    the same time instead of one after another on one pod; each session keeps every guard of run_session (budget cap,
    max_cost per session, kill switch, always deleted). At most `max_pods` sessions at once."""
    from concurrent.futures import ThreadPoolExecutor
    groups = components(jobs)
    if len(groups) == 1:
        return {"sessions": [run_session(jobs, label=label, max_cost=max_cost, gpus=gpus, force=force, dry_run=dry_run,
                                         max_minutes=max_minutes, clouds=clouds)]}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def one(k_g):
        k, g = k_g
        time.sleep(3 * k)   # stagger the pod creates (runpodctl, ledger lock)
        return run_session(g, sid=f"{stamp}-{label}-p{k + 1}", label=f"{label}-p{k + 1}", max_cost=max_cost, gpus=gpus,
                           force=force, dry_run=dry_run, max_minutes=max_minutes, clouds=clouds)
    with ThreadPoolExecutor(max_workers=max(1, min(max_pods, len(groups)))) as ex:
        res = list(ex.map(one, enumerate(groups)))
    return {"sessions": res, "groups": [[j.name for j in g] for g in groups]}


def _start_puller(ses: Session, pod, todo: list[JobSpec], bundle: Path, man: dict):
    """L1 progress puller (dancecap.progress_pull): read-only probe over this pod's ssh every 15 s into
    <session>/progress/. Never fails the session: any import/start problem -> None. DANCECAP_PROGRESS_PULL=0 disables."""
    try:
        from . import progress_pull
        return progress_pull.start_for_session(ses, pod, todo, upload_part=f"{POD_UPLOAD}/{bundle.name}.part",
                                               upload_bytes=man.get("bytes"))
    except Exception:
        return None


def _stop_puller(puller) -> None:
    try:
        if puller is not None:
            puller.stop(timeout=10)
    except Exception:
        pass


def _pod_steps(ses: Session, pod, todo: list[JobSpec], bundle: Path, man: dict, feats: list[str]) -> None:
    rlog = ses.runner_log
    remote_b = f"{POD_UPLOAD}/{bundle.name}"

    streams = upload_streams(man["bytes"])

    def upload():
        have = pod.output(f"mkdir -p {POD_UPLOAD}; [ -f {remote_b} ] && sha256sum {remote_b} | cut -d' ' -f1 || true")
        if have.strip() == man["sha256"]:
            ses.log("upload: pod already has this bundle")
            return
        if streams > 1:
            striped_upload(pod, bundle, POD_UPLOAD, streams, UploadProgress(ses.dir, man["bytes"], streams))
            return
        pod.put(bundle, remote_b + ".part", timeout=max(600, man["bytes"] / 1e6 / 0.5))
        got = pod.output(f"sha256sum {remote_b}.part | cut -d' ' -f1").strip()
        if got != man["sha256"]:
            raise RuntimeError(f"upload sha256 mismatch {got} != {man['sha256']}")
        pod.output(f"mv {remote_b}.part {remote_b}")
    ses.timed("upload", lambda: _retry(upload, what="upload"), mb=round(man["bytes"] / 1e6, 1), streams=streams)
    snapshot(ses, "upload_done", f"upload of {man['bytes'] / 1e6:.0f} MB done")

    marker = f"{POD_WS}/.bundle_{man['key']}.ok"
    ses.timed("extract", lambda: pod.run(
        f"set -e; mkdir -p {POD_WS}; if [ -f {marker} ]; then echo extracted already; else "
        f"tar -xzf {remote_b} -C {POD_WS} && echo {man['n_files']} > {marker}; fi; "
        f"echo files=$(find {POD_WS} -type f | wc -l); du -sh {POD_WS} | cut -f1", log=rlog, timeout=900))

    boot_to = 60 * (sum(BOOTSTRAP_MIN[f] for f in feats) * 2 + 10)
    boot_cmd = (f"bash {POD_WS}/dancecap/remote/bootstrap.sh {' '.join(f for f in feats if f != 'base')} "
                f"2>&1 | tee -a /workspace/bootstrap.log; exit ${{PIPESTATUS[0]}}")
    rc = ses.timed("bootstrap", lambda: pod.run(boot_cmd, check=False, log=rlog, timeout=boot_to), features=feats)
    if rc != 0:  # transient pip/network failures (2026-10-07: PyPI read timeouts lost a whole session): one more try
        ses.log(f"bootstrap rc={rc}: retrying once on this pod (finished features are skipped)")
        rc = ses.timed("bootstrap_retry", lambda: pod.run(boot_cmd, check=False, log=rlog, timeout=boot_to),
                       features=feats)
    for name in ("check.json", "timings.tsv"):
        try:
            (ses.dir / f"bootstrap_{name}").write_text(pod.output(f"cat /workspace/.dancecap_bootstrap/{name}"),
                                                       encoding="utf-8")
        except RuntimeError:
            pass
    try:
        pod.get("/workspace/bootstrap.log", ses.dir / "bootstrap.log", timeout=300)
    except Exception:
        pass
    ses.state["bootstrap_rc"] = rc
    snapshot(ses, "bootstrap_done" if rc == 0 else "bootstrap_failed", "bootstrap done" if rc == 0 else
             f"bootstrap FAILED rc={rc}")
    ses.timed("wheels_back", lambda: _sync_wheels(pod, ses))
    if rc != 0:
        ses.log(f"bootstrap failed (rc={rc}); see {ses.dir / 'bootstrap.log'} - jobs will still be attempted "
                f"only if their features passed")
    failed_feats = _failed_features(ses)
    if rc != 0 and not (ses.dir / "bootstrap_timings.tsv").exists():
        failed_feats = set(feats)  # the script itself did not run

    for job in todo:
        st = ses.state["jobs"][job.name]
        dep_bad = [d for d in job.after if d in ses.state["jobs"] and d in {j.name for j in todo}
                   and ses.state["jobs"][d].get("status") != "ok"]
        if dep_bad:
            st["status"] = f"skipped: dependency failed {dep_bad}"
            ses.log(f"{job.name}: {st['status']}")
            ses.save()
            continue
        feats_j = set(job_features(job))
        if feats_j & failed_feats or "base" in failed_feats:
            st["status"] = "skipped: bootstrap feature failed " + str(sorted(feats_j & failed_feats or {"base"}))
            ses.log(f"{job.name}: {st['status']}")
            ses.save()
            continue
        _run_job(ses, pod, job, st)


def _failed_features(ses: Session) -> set[str]:
    p = ses.dir / "bootstrap_timings.tsv"
    last = {}
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                last[parts[0]] = parts[2]
    return {f for f, rc in last.items() if rc != "0" and f != "check"}


def _sync_wheels(pod, ses: Session) -> list[str]:
    wh = f"{POD_WS}/dancecap/work/runpod/wheelhouse/{WHEEL_KEY}"
    listing = pod.output(f"for f in {wh}/*.whl; do [ -f \"$f\" ] && stat -c '%s %n' \"$f\"; done; true")
    got = []
    for line in listing.splitlines():
        try:
            size, path = line.split()
        except ValueError:
            continue
        local = WHEELHOUSE / PurePosixPath(path).name
        if local.exists() and local.stat().st_size == int(size):
            continue
        tmp = local.with_name(local.name + ".part")
        pod.get(path, tmp, timeout=900)
        os.replace(tmp, local)
        got.append(local.name)
    if got:
        ses.log(f"wheels cached locally for next session: {got}")
    ses.state["wheels_downloaded"] = got
    return got


def _run_job(ses: Session, pod, job: JobSpec, st: dict) -> None:
    log_rel = ses.log_rel(job.name)
    st.update(status="running", commands={}, started=time.time())
    ses.save()
    failed = None
    for a, dest in job.archives:
        arc = archive_arc(job, a)
        mk = f"{POD_WS}/{arc}.extracted"
        excl = " ".join(f"--exclude={x}" for x in ARCHIVE_CODE_EXCLUDE)
        rc = ses.timed(f"{job.name}/archive", lambda arc=arc, mk=mk, dest=dest, excl=excl: pod.run(
            f"set -e; [ -f {mk} ] && {{ echo archive already extracted; exit 0; }}; mkdir -p {POD_WS}/{dest}; "
            f"tar -xzf {POD_WS}/{arc} -C {POD_WS}/{dest} {excl}; touch {mk}", check=False, log=ses.runner_log,
            timeout=1800))
        if rc != 0:
            failed = f"archive {Path(a).name} rc={rc}"
            break
    if job.secrets and not failed:
        try:
            st["secrets"] = ses.timed(f"{job.name}/secrets", lambda: push_secrets(pod, job))
        except Exception as e:  # never echo a value: the message names the secret / file only
            failed = f"secrets: {type(e).__name__}: {e}"
    for i, c in enumerate(job.setup if not failed else []):
        rc = ses.timed(f"{job.name}/setup{i + 1}", lambda c=c: pod.run(
            f"cd {POD_WS}/{job.cwd} && {secret_exports(job)}{c}", check=False, log=ses.runner_log, timeout=1800))
        if rc != 0:
            failed = f"setup{i + 1} rc={rc}"
            break
    for c in job.commands if not failed else []:
        st["current"] = c.name  # for the monitor (additive)
        fp = FrameProgress(ses, job, c, getattr(pod, "gpu", None))   # frames done / total + rate -> progress/frames.json
        rc = None
        try:
            rc = ses.timed(f"{job.name}/{c.name}", lambda c=c, fp=fp: pod.run(
                command_script(job, c, log_rel), check=False, log=ses.runner_log,
                timeout=c.timeout_min * 60 + 120, on_line=fp.feed), job=job.name)
        except TimeoutError as e:
            rc = f"timeout: {e}"
        finally:
            fp.close(rc)
        st["commands"][c.name] = rc
        ses.save()
        if rc != 0 and not c.optional:
            failed = f"{c.name} rc={rc}"
            break
    # always bring back what exists (logs + partial outputs), even after a failure
    out_p = f"{POD_OUT}/{ses.sid}_{job.name}.tar.gz"
    try:
        # pack is idempotent (.part + mv): retry through ssh blips - a lost pack used to discard a finished job
        # (2026-10-06: one ssh timeout after a 40 min room run -> pod deleted with the outputs on it)
        info = ses.timed(f"{job.name}/pack", lambda: _retry(
            lambda: pod.output(pack_command(job, log_rel, out_p), timeout=1800), tries=5, wait_s=30, what="pack"))
        missing = [l[8:] for l in info.splitlines() if l.startswith("MISSING ")]
        size = int(next(l.split()[1] for l in info.splitlines() if l.startswith("SIZE ")))
        sha = next(l.split()[1] for l in info.splitlines() if l.startswith("SHA "))
        local = ses.dir / "downloads" / f"{job.name}.tar.gz"
        st["current"] = None
        st["pack_bytes"] = size  # the monitor divides downloads/<job>.tar.gz.part by this (additive)
        ses.save()

        def download():
            if local.exists() and local.stat().st_size == size and _sha256(local) == sha:
                return
            tmp = local.with_name(local.name + ".part")
            pod.get(out_p, tmp, timeout=max(600, size / 1e6 / 1.0))
            if _sha256(tmp) != sha:
                raise RuntimeError("download sha256 mismatch")
            os.replace(tmp, local)
        ses.timed(f"{job.name}/download", lambda: _retry(download, tries=5, wait_s=30, what="download"), mb=round(size / 1e6, 1))
        dest = WORKSPACE if job.extract == "workspace" else ses.dir / "outputs"
        got = ses.timed(f"{job.name}/extract", lambda: safe_extract(local, dest, WORKSPACE, job.outputs, log_rel))
        missing_verify = [v for v in job.verify if not (dest / v.rstrip("/")).exists()]
        st.update(download_mb=round(size / 1e6, 2), files=len(got), missing_on_pod=missing,
                  missing_verify=missing_verify, outputs_root=str(dest), logs=str(WORKSPACE / log_rel))
        if not failed and missing_verify:
            failed = f"outputs missing: {missing_verify}"
    except Exception as e:
        failed = failed or f"pack/download failed: {e}"
        st["download_error"] = str(e)[:500]
    st["status"] = "ok" if not failed else f"failed: {failed}"
    st["seconds"] = round(time.time() - st["started"], 1)
    ses.log(f"{job.name}: {st['status']} ({st['seconds']:.0f}s)")
    ses.save()
    snapshot(ses, f"job_{job.name}", f"{job.name}: {st['status']}")


def _finish(ses: Session, pod_id: str | None) -> dict:
    from . import remote
    res = {"sid": ses.sid, "dir": str(ses.dir), "jobs": {k: {x: v.get(x) for x in (
        "status", "commands", "seconds", "download_mb", "files", "missing_on_pod", "missing_verify", "outputs_root",
        "logs")} for k, v in ses.state["jobs"].items()}, "timings": ses.state["timings"],
        "bundle": {k: ses.state.get("bundle", {}).get(k) for k in ("key", "bytes", "raw_bytes", "n_files")},
        "estimate_min": ses.state.get("estimate_min"), "max_minutes": ses.state.get("max_minutes"),
        "error": ses.state.get("error"), "bootstrap_rc": ses.state.get("bootstrap_rc")}
    if pod_id:
        row = next((r for r in remote.ledger() if r["pod_id"] == pod_id), {})
        res["pod"] = {**ses.state.get("pod", {}), "ledger_cost": row.get("cost"),
                      "lifetime_s": round(row["deleted_ts"] - row["created_ts"], 1) if row.get("deleted_ts") else None,
                      "deleted_reason": row.get("deleted_reason")}
        steps = {t["step"]: t["seconds"] for t in ses.state["timings"]}
        res["bootstrap_s"] = steps.get("bootstrap")
    res["spent_total"] = round(remote.spent(), 4)
    res["remaining"] = round(remote.remaining(), 4)
    (ses.dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    ses.event("fail" if res.get("error") else "end", "session", msg=f"session ended: {res.get('error') or 'ok'}")
    if pod_id:
        bad = [k for k, v in res["jobs"].items() if v.get("status") != "ok"]
        snapshot(ses, "session_end", "session ended: " + (res.get("error") or (f"not ok: {bad}" if bad else "ok"))[:120],
                 final=True)
    try:  # refresh the ETA history (work/progress/rates.json; derived, rebuildable, never fatal)
        if pod_id:
            from . import progress_eta
            progress_eta.rates_rebuild()
    except Exception:
        pass
    return res
