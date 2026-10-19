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
  2. pod       dancecap.remote.pod_session (budget cap, per-session max_cost, on-pod kill switch, always deleted)
  3. upload    skipped when the pod already holds a file with the same sha256
  4. extract   marker per bundle hash
  5. bootstrap remote/bootstrap.sh <features of all jobs> check (idempotent per feature); newly built wheels are
               downloaded into the local wheelhouse so the next session installs instead of compiling
  6. per job   setup + commands (each with a .done marker, timeout and log), then outputs packed on the pod,
               downloaded, sha256-checked, extracted with a path whitelist and verified; a failed job still
               returns its logs and partial outputs
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
FEATURES = ("base", "mamma", "room", "texture", "dancers", "cameras", "ma2d", "gsplat", "check")  # check last
UPLOAD_MBPS, DOWNLOAD_MBPS = 2.5, 12.0  # measured home link (2026-10-06)
# bootstrap minutes per feature: cold (CUDA wheels compiled on the pod) / with cached wheels
BOOTSTRAP_MIN = {"base": 5, "mamma": 8, "room": 15, "texture": 0, "dancers": 3, "cameras": 1, "ma2d": 3, "gsplat": 12,
                 "check": 1}
BOOTSTRAP_MIN_CACHED = {"base": 5, "mamma": 2, "room": 3, "texture": 0, "dancers": 3, "cameras": 1, "ma2d": 3,
                        "gsplat": 1, "check": 1}
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

    def fingerprint(self) -> str:
        d = asdict(self)
        d.pop("source", None)
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


def code_items() -> list[tuple[Path, str]]:
    items = [(PACKAGE_ROOT / "dancecap", "dancecap/dancecap"), (PACKAGE_ROOT / "takes", "dancecap/takes"),
             (PACKAGE_ROOT / "remote", "dancecap/remote"), (PACKAGE_ROOT / "CONTRACTS.md", "dancecap/CONTRACTS.md")]
    root = env.mamma_root()
    for p in sorted(root.iterdir()) if root.exists() else []:
        if p.name in MAMMA_SKIP or p.name.startswith("."):
            continue
        items.append((p, f"mamma-turing/{p.name}"))
    if WHEELHOUSE.exists():
        items.append((WHEELHOUSE, f"dancecap/work/runpod/wheelhouse/{WHEEL_KEY}"))
    if PYWHEELS.exists() and any(PYWHEELS.iterdir()):
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
        items += code_items()
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
    want = {"base", "check"} | {n for j in jobs for n in j.needs}
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


def command_script(job: JobSpec, cmd: Command, log_rel: str) -> str:
    """Pod shell for one command: skip when its .done marker exists; log with tee; exit code to <name>.rc."""
    logd = f"{POD_WS}/{log_rel}"
    envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in {"DANCECAP_POD": "1", "PYTHONUNBUFFERED": "1",
                                                                **_progress_env(job, cmd, log_rel), **job.env}.items())
    secs = int(cmd.timeout_min * 60)
    return (f"mkdir -p {logd}; if [ -f {logd}/{cmd.name}.done ]; then echo '[pod-runner] {job.name}/{cmd.name} "
            f"already done'; exit 0; fi; cd {POD_WS}/{job.cwd} || exit 97; export {envs}; {secret_exports(job)}set -o pipefail; "
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
            from . import progress
            w = getattr(self, "_pw", None)
            if w is None:
                w = self._pw = progress.Writer(self.dir / "progress" / "runner.jsonl", job="_session", sid=self.sid)
            job, name = (step.split("/", 1) if "/" in step else ("_session", step))
            extra = {k: v for k, v in extra.items() if k in ("seconds", "mb", "job", "features", "bytes", "done",
                                                              "total")}
            done, total = extra.pop("done", None), extra.pop("total", None)
            w.emit(kind, name, job=job, msg=msg, done=done, total=total,
                   unit="bytes" if total is not None else None, extra={"runner_step": step, **extra})
        except Exception:
            pass

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
    ses.state.pop("error", None)  # a resumed session starts clean
    ses.state.pop("bootstrap_rc", None)
    ses.write_runner_info(label, [j.name for j in jobs])
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
            ses.state["pod"] = pod_info
            ses.save()
            ses.event("note", "pod", msg=f"pod {pod.id} {pod.gpu} ${pod.cost_per_hr:.2f}/h ready")
            puller = _start_puller(ses, pod, todo, bundle, man)
            try:
                _pod_steps(ses, pod, todo, bundle, man, feats)
            finally:
                _stop_puller(puller)
    except Exception as e:
        ses.state["error"] = f"{type(e).__name__}: {e}"[:2000]
        ses.log(f"session error: {ses.state['error']}")
    finally:
        ses.save()
    return _finish(ses, pod_info.get("id"))


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

    def upload():
        have = pod.output(f"mkdir -p {POD_UPLOAD}; [ -f {remote_b} ] && sha256sum {remote_b} | cut -d' ' -f1 || true")
        if have.strip() == man["sha256"]:
            ses.log("upload: pod already has this bundle")
            return
        pod.put(bundle, remote_b + ".part", timeout=max(600, man["bytes"] / 1e6 / 0.5))
        got = pod.output(f"sha256sum {remote_b}.part | cut -d' ' -f1").strip()
        if got != man["sha256"]:
            raise RuntimeError(f"upload sha256 mismatch {got} != {man['sha256']}")
        pod.output(f"mv {remote_b}.part {remote_b}")
    ses.timed("upload", lambda: _retry(upload, what="upload"), mb=round(man["bytes"] / 1e6, 1))

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
        if set(job.needs) & failed_feats or "base" in failed_feats:
            st["status"] = "skipped: bootstrap feature failed " + str(sorted(set(job.needs) & failed_feats or {"base"}))
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
        try:
            rc = ses.timed(f"{job.name}/{c.name}", lambda c=c: pod.run(
                command_script(job, c, log_rel), check=False, log=ses.runner_log,
                timeout=c.timeout_min * 60 + 120), job=job.name)
        except TimeoutError as e:
            rc = f"timeout: {e}"
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
    try:  # refresh the ETA history (work/progress/rates.json; derived, rebuildable, never fatal)
        if pod_id:
            from . import progress_eta
            progress_eta.rates_rebuild()
    except Exception:
        pass
    return res
