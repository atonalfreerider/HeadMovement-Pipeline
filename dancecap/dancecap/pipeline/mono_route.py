"""The monocular route of `pipeline analyze`, step by step, with no agent: one phone video in, a finished analysis out (owner,
2026-10-21: "I want ... a monocular video and want to see a completed analysis on the other side").

The 2026-10-21 trial ran these steps through take-specific glue (work/trial_mono_jl20/agent/: camera and people reused from the
parent video, hand-written pod jobs, a copy instead of the stitch). Each step here writes its own pod jobs from the take (the TOML,
the facts of the video, the outputs of the steps before it) and runs them through `python -m dancecap remote run` tagged with the
analysis (DANCECAP_ANALYSIS / DANCECAP_ANALYSIS_STEP: the monitor shows its sessions under the step):

    beats       local  `dancecap beats` (the video's own sound -> audio/) + heat_beats (the beat grid -> song/)            26 s
    camera      pod    cameras.mono_cam pod: probe, PTZ skeleton, per-frame track, MoGe-2 (-> camera_work/)
    people      pod    people_mono run: everybody detected, the couple + lead / follow, SAM 2.1, RTMW (-> people/)
    world       local  cameras.mono_cam world: focal, gravity, camera height from the statures, origin (-> cameras/)
    sam3d_body  pod    mono4d.jl_sam3db for the take: frames decoded on the pod, SAM 3D Body, MHR -> SMPL-X; import, QA and
                       the prior on the fit lattice here (no first-pass fit needed: qa_nofit / prior_nofit)
    fits        pod    one session per chunk of <= CHUNK_MAX_S s (side by side, `remote run --parallel`): the chunk front
                       (frames, cameras, masks, ma_2d, sync, prepare, mono-init, qa-mask), the build (configs + prior slice),
                       MAMMA_MONO + SAM 3D Body prior with the betas PINNED from the statures (no shape pass), select, refine
    post        local  the stitcher over the chunks (per-frame flags the export needs; the trial's copy lacked them) + jl_repair
    analyses    pod    timing, physics, counterbalance, floorcraft on the stitched motion
    avatars     pod    LHM++ splat avatars (optional; nothing in the export reads them)
    export      local  jl_export --take --no-install (work/<take>/export/<capture>/; the Unity project is never touched)
    film        local  needs the Unity editor (another workflow owns it): skipped with that reason

    python -m dancecap.pipeline.mono_route STEP --take TAKE [--run RUN] [--dry-run]
    python -m dancecap.pipeline.mono_route fit-build --take TAKE --cid CID        (on the fit pod)

`--dry-run` writes the pod jobs and prints them, starts nothing. Every pod job maps the source video to dances/<take>/<file> on the pod
(where recap_people.video_path looks with HM_FRONT_CAPTURE = <take>), so the video is uploaded by each session that decodes it (MBs;
the trial uploaded 349 MB of JPEGs and a 1.6 GB checkpoint: here the 540 MB stripped ma_2d checkpoint, front_pod.MA2D_STRIPPED).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE

POD_WS = "/workspace/HeadMovement"
CAM = "01"
HZ = 30.0
CHUNK_MAX_S = 26.0          # fit windows of at most 26 s overlapping by 4 s: cores of ~22 s (the plan's chunk length)
OVERLAP_S = 4.0
BETAS_N = 16                # MAMMA's betas per body (the pinned shapes of john_lula / the recap: 16 values)
SHAPE_DIR = "shape"         # work/<take>/shape/mamma_pin_betas.json (as john_lula's pinned shape pass)
MA2D_POD = "mamma/data/weights/ma_2d/mamma_mask_full_cvpr.ckpt"
EXCLUDE = ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse"]
GENDER_EXCLUDE = ["mamma/data/body_models/smplx_locked_head/smplx/SMPLX_MALE.npz",
                  "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_FEMALE.npz"]
GENDER_LINKS = ("for g in MALE FEMALE; do ln -sf SMPLX_NEUTRAL.npz "
                f"{POD_WS}/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_$g.npz; done")
# pip pins under the bootstrap's numpy / torch (a newer numpy or torch from a dependency breaks the env: bootstrap.sh _constraints)
CONSTRAINTS = "printf '%s\\n' numpy==1.26.4 torch==2.5.1 torchvision==0.20.1 > /tmp/dc_constraints.txt"
PIP = "python -m pip install -q -c /tmp/dc_constraints.txt"
ULTRALYTICS = f"{PIP} ultralytics==8.4.138 opencv-python==4.11.0.86"
MOGE_COMMIT = "07444410f1e33f402353b99d6ccd26bd31e469e8"       # cameras/single_cam_pod.py: MoGe-2 before the V3 rewrite
UTILS3D_COMMIT = "3fab839f0be9931dac7c8488eb0e1600c236e183"
MOGE = (f"python -m pip install -q --no-deps git+https://github.com/microsoft/MoGe.git@{MOGE_COMMIT} "
        f"git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_COMMIT} && "
        f"{PIP} click trimesh matplotlib scipy huggingface_hub einops safetensors")   # the hub: MoGe-2's weights (first pod run)
# rtmlib pulls opencv-contrib 5 over the pinned 4.11 (people_mono / NOTES_J3a): put 4.11 back
RTMW = (f"{PIP} rtmlib onnxruntime-gpu==1.20.2 && (python -m pip uninstall -y -q opencv-contrib-python || true) && "
        "python -m pip install -q --force-reinstall --no-deps opencv-python==4.11.0.86")
# onnxruntime-gpu finds CUDA / cuDNN in torch's nvidia wheels (NOTES_J3a.md 7)
NV_LIBS = ("export LD_LIBRARY_PATH=$(python -c \"import glob, site; print(':'.join(glob.glob(site.getsitepackages()[0] + "
           "'/nvidia/*/lib')))\"):${LD_LIBRARY_PATH:-}")
SESSION_TRIES = 3           # a session that failed for pod reasons (no pod, bootstrap, upload, ssh) is started again
RETRY_JOB_CMDS = ("setup", "install")   # a job that failed in these commands (network installs) is retried too


def log(msg: str) -> None:
    print(f"[mono_route {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ the take


class Ctx:
    """One step of one take: paths, the TOML, the video and its pod path, the analysis tag, the job folder."""

    def __init__(self, take: str, run: str | None = None, step: str | None = None, dry_run: bool = False):
        self.take, self.run, self.step, self.dry_run = take, run, step, dry_run
        self.work = PACKAGE_ROOT / "work" / take
        self.toml = PACKAGE_ROOT / "takes" / f"{take}.toml"
        self.cfg = tomllib.loads(self.toml.read_text(encoding="utf-8"))
        from ..take import Take
        tk = Take.load(take)
        if len(tk.sources) != 1:
            raise SystemExit(f"{take}: the monocular route needs exactly one video ({len(tk.sources)} sources)")
        self.video = Path(tk.sources[0].path)
        if not self.video.exists():
            raise SystemExit(f"{take}: the video {self.video} is missing")
        self.video_pod = f"dances/{take}/{self.video.name}"          # workspace-relative on the pod
        self.video_arg = f"../{self.video_pod}"                       # from the pod commands' cwd (dancecap/)
        self.jobs_dir = self.work / "jobs" / "analyze"
        self.timings: dict = {}

    def ws(self, p: Path) -> str:
        return ws_rel(p)

    def video_input(self) -> dict:
        return {"local": str(self.video.resolve()), "pod": self.video_pod}

    def statures(self) -> dict:
        d = self.cfg.get("dancers") or {}
        return {r: float(d[r]["stature_m"]) for r in ("lead", "follow")}

    def soles(self) -> dict:
        d = self.cfg.get("dancers") or {}
        return {r: float(d[r].get("shoe_sole_m", 0.025)) for r in ("lead", "follow")}

    def write_job(self, d: dict) -> Path:
        d = {"schema": "dancecap.pod_job/1", "take": self.take, "cwd": "dancecap", **d}
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        p = self.jobs_dir / f"{d['name']}.json"
        p.write_text(json.dumps(d, indent=1), encoding="utf-8", newline="\n")
        return p

    def local(self, argv: list[str], timeout_min: float = 30.0, env: dict | None = None) -> None:
        """A local command (python -m ...), output to this process's stdout (the step's log)."""
        t0 = time.time()
        log(f"local: {' '.join(argv)}")
        if self.dry_run:
            return
        rc = subprocess.run([sys.executable] + argv, cwd=str(PACKAGE_ROOT), timeout=timeout_min * 60,
                            env={**os.environ, **(env or {})}).returncode
        if rc != 0:
            raise RuntimeError(f"{' '.join(argv[:4])} ... rc={rc}")
        log(f"local done in {time.time() - t0:.0f} s")

    def remote(self, jobs: list[Path], parallel: bool = False, max_minutes: float | None = None,
               max_cost: float | None = None) -> list[dict]:
        """`remote run` of these pod jobs, tagged with the analysis; restarted (a new session) when it failed for pod reasons.
        Returns the result.json of every session that ran. Raises with the failing command's log tail."""
        names = [json.loads(p.read_text(encoding="utf-8"))["name"] for p in jobs]
        mm = max_minutes or sum(float(json.loads(p.read_text(encoding="utf-8"))["max_minutes"]) for p in jobs) + 20
        cost = max_cost if max_cost is not None else session_max_cost(mm)
        argv = [sys.executable, "-m", "dancecap", "remote", "run", "--jobs", ",".join(str(p) for p in jobs),
                "--max-cost", f"{cost:.2f}", "--max-minutes", f"{mm:.0f}"] + (["--parallel"] if parallel else [])
        env = {**os.environ}
        if self.run:
            env.update(DANCECAP_ANALYSIS=self.run, DANCECAP_ANALYSIS_STEP=self.step or "")
        log(f"remote: {len(jobs)} job(s) {names}, max {mm:.0f} min, max ${cost:.2f} per session" + (" (parallel)" if parallel else ""))
        if self.dry_run:
            for p in jobs:
                log(f"  {p}")
            return []
        todo, out = list(jobs), []
        for attempt in range(1, SESSION_TRIES + 1):
            t0 = time.time()
            before = set(_sessions_since(t0 - 2))
            argv[argv.index("--jobs") + 1] = ",".join(str(p) for p in todo)
            rc = subprocess.run(argv, cwd=str(PACKAGE_ROOT), env=env).returncode
            # this call's sessions only: other workflows start sessions at the same time (the first pod check took a sibling
            # session's "no GPU pod" for its own and retried a real command failure)
            sids = [s for s in _sessions_since(t0 - 2) if s not in before and _session_jobs(s) & set(names)]
            res = [r for r in (_result(s) for s in sids) if r]
            out += res
            ok_jobs = {j for r in res for j, v in (r.get("jobs") or {}).items() if v.get("status") == "ok"}
            left = [p for p, n in zip(todo, [json.loads(q.read_text(encoding="utf-8"))["name"] for q in todo]) if n not in ok_jobs]
            if not left:
                log(f"remote: all jobs ok ({', '.join(sids)}) in {time.time() - t0:.0f} s")
                return out
            why = session_failure(res, rc)
            if not why["retry"] or attempt == SESSION_TRIES:
                raise RuntimeError(f"remote run failed (sessions {', '.join(sids) or 'none'}): {why['msg']}")
            log(f"remote: attempt {attempt} failed for pod reasons ({why['msg'][:200]}); a new session for {len(left)} job(s)")
            todo = left
        raise RuntimeError("unreachable")


def ws_rel(p: Path) -> str:
    """Workspace-relative posix path WITHOUT resolving links (a worktree reaches work/ through a junction: resolve() would leave
    the workspace)."""
    r = Path(os.path.relpath(os.path.abspath(p), os.path.abspath(WORKSPACE))).as_posix()
    if r.startswith(".."):
        raise ValueError(f"{p} is outside the workspace {WORKSPACE}")
    return r


def session_max_cost(max_minutes: float, per_hr: float = 1.8) -> float:
    """--max-cost of one session (the runner's worst case = price x max minutes): room for the fast cards (RTX 4090 $0.59-0.89/h,
    L40S, A100 up to ~$1.8/h) and never so tight that the runner falls to a slow card (the trial's A40)."""
    return round(max(0.6, per_hr * float(max_minutes) / 60.0), 2)


def _sessions_since(t0: float) -> list[str]:
    from . import paths
    d = paths.sessions_dir()
    if not d.is_dir():
        return []
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(t0))
    return sorted(x.name for x in d.iterdir() if x.is_dir() and x.name[:15] >= stamp)


def _session_jobs(sid: str) -> set:
    """The job names a session runs (its runner.json, written when it starts)."""
    from . import paths
    try:
        return set(json.loads((paths.sessions_dir() / sid / "runner.json").read_text(encoding="utf-8")).get("jobs") or [])
    except (OSError, ValueError):
        return set()


def _result(sid: str) -> dict | None:
    from . import paths
    p = paths.sessions_dir() / sid / "result.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def session_failure(results: list[dict], rc: int) -> dict:
    """{retry, msg}: retry when the sessions failed for pod reasons (no pod, upload, bootstrap, ssh: result `error` or a failed
    bootstrap) or in a network install command; a job command that failed on its own is a real failure (its log tail in msg)."""
    if not results:
        return {"retry": True, "msg": f"no session result (runner rc {rc})"}
    msgs, retry = [], False
    for r in results:
        if r.get("error") or (r.get("bootstrap_rc") not in (None, 0)):
            retry = True
            msgs.append(f"{r.get('sid')}: {r.get('error') or 'bootstrap rc ' + str(r.get('bootstrap_rc'))}")
            continue
        for j, v in (r.get("jobs") or {}).items():
            if v.get("status") == "ok":
                continue
            bad = [c for c, rc_ in (v.get("commands") or {}).items() if rc_ not in (0, None)]
            if any(c in RETRY_JOB_CMDS for c in bad):
                retry = True
            tail = _log_tail(v.get("logs"), bad[0]) if bad else ""
            msgs.append(f"{j}: {v.get('status')}" + (f" (command {bad[0]}: {tail})" if bad else ""))
    return {"retry": retry, "msg": "; ".join(msgs)[:1500] or f"runner rc {rc}"}


def _log_tail(logs_dir: str | None, cmd: str, n: int = 600) -> str:
    if not logs_dir:
        return ""
    for name in (f"{cmd}.log", f"{cmd}.txt"):
        p = Path(logs_dir) / name
        if p.exists():
            return " ".join(p.read_text(encoding="utf-8", errors="replace")[-n:].split())
    return ""


# ------------------------------------------------------------------------------------------------ pure helpers (unit-tested)


def grid_frames(duration_s: float, fps: float, hz: float = HZ) -> int:
    """Instants of the take's fit lattice t_k = k / hz inside the video: k = 0 .. floor(hz x t_last) (t_last = the last frame)."""
    n = max(2, int(round(float(duration_s) * float(fps))))
    t_last = (n - 1) / float(fps)
    return int(math.floor(t_last * hz + 1e-6)) + 1


def plan_chunks(take: str, n_grid: int, max_s: float = CHUNK_MAX_S, overlap_s: float = OVERLAP_S, k_first: int = 0) -> list[dict]:
    """Fit windows of at most max_s over the lattice k_first .. k_first + n_grid - 1, overlapping by overlap_s, cut at the overlaps'
    middles (farm.chunks.equal_windows). Ids g<core start, 4 digits> (farm.chunks.chunk_id); chunk takes <take>_<id>."""
    from ..farm.chunks import equal_windows
    w = equal_windows(n_grid, HZ, max_s, overlap_s)
    out = []
    for (k0, k1), (c0, c1) in zip(w["windows"], w["cores"]):
        k0, k1, c0, c1 = k0 + k_first, k1 + k_first, c0 + k_first, c1 + k_first
        cid = f"g{c0:04d}"
        out.append({"id": cid, "k0": int(k0), "k1": int(k1), "core_k0": int(c0), "core_k1": int(c1),
                    "overlap_k": int(w["overlap_k"]) if w["n"] > 1 else 0, "take": f"{take}_{cid}", "T": int(k1 - k0 + 1),
                    "core_s": round((c1 - c0 + 1) / HZ, 2)})
    return out


def rest_height(v_template: np.ndarray, shapedirs: np.ndarray, betas: np.ndarray, up: int = 1) -> float:
    """Crown-to-sole height of the shaped rest mesh (mamma-turing optimization/utils/mono.stature_loss)."""
    nb = len(betas)
    v = v_template + np.einsum("vcb,b->vc", shapedirs[:, :, :nb], betas)
    return float(v[:, up].max() - v[:, up].min())


def beta0_for_height(v_template: np.ndarray, shapedirs: np.ndarray, target_m: float, n: int = BETAS_N,
                     lo: float = -6.0, hi: float = 6.0) -> np.ndarray:
    """Betas (n) with only beta0 set so the rest height is target_m (bisection: the height is monotonic in beta0 over the
    range). The fit-route audit (2026-10-22): john_lula's pinned shape pass gave beta0 1.064 and |others| <= 0.15 for the lead,
    all ~0 for the follow, so a stature-only shape lands within millimetres of the shape pass's height without its 20 min."""
    def h(b0):
        b = np.zeros(n)
        b[0] = b0
        return rest_height(v_template, shapedirs, b)
    a, b = lo, hi
    inc = h(b) > h(a)
    for _ in range(60):
        m = 0.5 * (a + b)
        if (h(m) < target_m) == inc:
            a = m
        else:
            b = m
    out = np.zeros(n)
    out[0] = 0.5 * (a + b)
    return out


def stitch_list_doc(take: str, chunks: list[dict], duration_s: float, tag: str = "c1") -> dict:
    """The stitcher's chunk list (the trial / john_lula format): one camera = the whole window is monocular, partial."""
    return {"take": take, "out_raw": "smplx", "out_refined": "smplx_rf2_noslip", "anchor": 0.0, "hz": HZ, "blend_s": 2.0,
            "edge_margin_s": 0.2, "gate_mm": {},
            "chunks": [{"name": c["id"], "src": f"{c['take']}/smplx", "refined": f"{c['take']}/smplx_rf2_noslip",
                        "k": [c["k0"], c["k1"]], "margin_s": 0.2, "kind": "gap", "prepare": f"{c['take']}/mamma/prepare.json",
                        "ma_2d": f"{c['take']}/mamma/ma_2d/{c['take']}_{tag}"} for c in chunks],
            "partial": True, "mono": [[0.0, round(float(duration_s), 2)]],
            "note": "pipeline/mono_route post: the chunks of the fits step"}


def frames_json_doc(t: np.ndarray, pts: np.ndarray, size: tuple[int, int], source: str) -> dict:
    """frames/01/frames.json (ingest's layout) for the video's frames 0..F-1: what jl_sam3db.prep lists (the pod decodes them)."""
    t, pts = np.asarray(t, float), np.asarray(pts, float)
    d = np.diff(pts)
    return {"camera": CAM, "source": source, "width": int(size[0]), "height": int(size[1]), "rotation": 0, "slope": 1.0, "offset": 0.0,
            "frame_count": int(len(t)), "fps_median": float(1.0 / np.median(d)) if len(d) else 0.0,
            "frame_interval_ms": {"min": float(d.min() * 1000), "max": float(d.max() * 1000)} if len(d) else {},
            "frames": [{"file": f"{i:06d}.jpg", "pts": round(float(pts[i]), 6), "t": round(float(t[i]), 6)} for i in range(len(t))]}


# ------------------------------------------------------------------------------------------------ steps


def step_beats(ctx: Ctx) -> str:
    """The video's own sound (audio/audio_overlap.wav: the export's soundtrack) and the heat's beat grid (song/beats_heat.json,
    segments.json no_beat: the analyses' --beats / --no-beat). Light: 26 s on this PC (2026-10-21 trial)."""
    ctx.local(["-m", "dancecap", "beats", ctx.take, "--no-plot"], timeout_min=15)
    ctx.local(["-m", "dancecap.heat_beats", "run", "--take", ctx.take, "--no-plot"], timeout_min=15)
    seg = _json(ctx.work / "song" / "segments.json") or {}
    tempo = _json(ctx.work / "song" / "tempo.json") or {}
    bpm = tempo.get("bpm") or (tempo.get("pulse") or {}).get("bpm")
    return f"beat grid: {bpm} BPM, no-beat {seg.get('no_beat')}" if bpm else "beat grid written (song/)"


def step_camera(ctx: Ctx) -> str:
    """Pod: the phone's per-frame rotation and focal (PTZ skeleton + track, people masked) and MoGe-2 on wide frames."""
    cw = ctx.work / "camera_work"
    v = ctx.video_arg
    common = f"--take {ctx.take} --video {v}"
    job = ctx.write_job({
        "name": f"{ctx.take}_camera"[:60], "max_minutes": 40, "needs": ["cameras"], "min_vram_gb": 24,
        "inputs": [ctx.video_input()], "exclude": EXCLUDE + ["mamma-turing"],
        "env": {"YOLO_CONFIG_DIR": "/tmp/Ultralytics", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "HF_HOME": "/workspace/hf_home", "TORCH_HOME": "/workspace/torch_home", "PYTHONUNBUFFERED": "1"},
        "setup": [CONSTRAINTS, ULTRALYTICS, MOGE],
        "commands": [{"name": "probe", "timeout_min": 3, "cmd": f"python -m dancecap.cameras.mono_cam probe {common}"},
                     {"name": "skeleton", "timeout_min": 15, "cmd": f"python -m dancecap.cameras.mono_cam skeleton {common}"},
                     {"name": "track", "timeout_min": 25, "cmd": f"python -m dancecap.cameras.mono_cam track {common} --shards 2"},
                     {"name": "moge", "timeout_min": 10, "cmd": f"python -m dancecap.cameras.mono_cam moge {common}"}],
        "outputs": [ctx.ws(cw) + "/"],
        "verify": [ctx.ws(cw / "skel" / "skeleton.npz"), ctx.ws(cw / "probe.json")],
        "notes": [f"pipeline/mono_route camera: the PTZ track of {ctx.video.name} + MoGe-2 (generic: cameras.mono_cam)"]})
    ctx.remote([job])
    info = _json(cw / "mono_cam.json") or {}
    return f"camera track: {json.dumps({k: info[k] for k in list(info)[:6]}, default=str)[:300]}" if info else "camera track pulled"


def step_people(ctx: Ctx) -> str:
    """Pod: everybody detected, the couple picked and told apart (lead / follow), SAM 2.1 masks both ways, RTMW whole body."""
    pe = ctx.work / "people"
    common = f"--take {ctx.take} --video {ctx.video_arg}"
    job = ctx.write_job({
        "name": f"{ctx.take}_people"[:60], "max_minutes": 50, "needs": ["dancers"], "min_vram_gb": 24,
        "inputs": [ctx.video_input()] + ([ctx.ws(pe / "roles_override.json")] if (pe / "roles_override.json").exists() else []),
        "exclude": EXCLUDE + ["mamma-turing"],
        "env": {"YOLO_CONFIG_DIR": "/tmp/Ultralytics", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_HOME": "/workspace/torch_home", "PYTHONUNBUFFERED": "1"},
        "setup": [CONSTRAINTS, ULTRALYTICS, RTMW],
        "commands": [{"name": "fetch_sam2", "timeout_min": 5, "cmd": "python -m dancecap.front_pod fetch-sam2"},
                     {"name": "detect", "timeout_min": 15, "cmd": f"python -m dancecap.people_mono detect {common}"},
                     {"name": "anchors", "timeout_min": 10, "cmd": f"python -m dancecap.people_mono anchors {common}"},
                     {"name": "sam", "timeout_min": 20, "cmd": f"python -m dancecap.people_mono sam {common}"},
                     {"name": "masks", "timeout_min": 10, "cmd": f"python -m dancecap.people_mono masks {common}"},
                     {"name": "pose", "timeout_min": 15, "cmd": f"{NV_LIBS} && python -m dancecap.people_mono pose {common}"},
                     {"name": "refine", "timeout_min": 30, "cmd": f"{NV_LIBS} && python -m dancecap.people_mono refine {common}"},
                     {"name": "finalize", "timeout_min": 10, "cmd": f"python -m dancecap.people_mono finalize {common}"}],
        "outputs": [ctx.ws(pe) + "/teachers_01.npz", ctx.ws(pe) + "/masks_01.npz", ctx.ws(pe) + "/teachers.json",
                    ctx.ws(pe) + "/identity.json", ctx.ws(pe) + "/coverage.json", ctx.ws(pe) + "/verification.json",
                    ctx.ws(pe) + "/people_mono.json", ctx.ws(pe) + "/anchors0.json", ctx.ws(pe / "in") + "/ocr.json",
                    ctx.ws(pe / "pose_tables") + "/", ctx.ws(ctx.work / "review" / "people") + "/"],
        "verify": [ctx.ws(pe / "teachers_01.npz"), ctx.ws(pe / "masks_01.npz")],
        "notes": [f"pipeline/mono_route people: the couple of {ctx.video.name} (generic: people_mono, no bibs)"]})
    ctx.remote([job])
    tj = _json(pe / "teachers.json") or {}
    idt = (_json(pe / "identity.json") or {}).get("roles") or tj.get("identity") or {}
    return f"people: masks {tj.get('mask_share')}, both conf>=0.5 {tj.get('both_ge_0.5')}; roles {json.dumps(idt, default=str)[:200]}"


def step_world(ctx: Ctx) -> str:
    """Local: the metric world of the camera (focal from MoGe-2, gravity, camera height from the dancers' statures, origin at the
    couple) -> cameras/01.npz etc.; frames/01/frames.json for SAM 3D Body (the pod decodes the frames)."""
    ctx.local(["-m", "dancecap.cameras.mono_cam", "world", "--take", ctx.take, "--video", str(ctx.video)], timeout_min=20)
    if not ctx.dry_run:
        write_frames_json(ctx)
    rig = _json(ctx.work / "cameras" / "rig.json") or {}
    keys = ("focal_1x_px", "camera_height_m", "dancers_mean_distance_m")
    msg = "world: " + ", ".join(f"{k} {rig.get(k):.3g}" if isinstance(rig.get(k), (int, float)) else f"{k} {rig.get(k)}" for k in keys)
    return msg + camera_model_warning(_json(ctx.work / "camera_work" / "mono_cam.json") or {})


SKEL_MAX_PX = 2.0      # a fixed-centre (pan / tilt / zoom) camera fits the background to ~1 px (J2: 0.85-0.92; lk03_mono20: 0.85)
TRACK_P90_MAX_PX = 4.0


def camera_model_warning(mc: dict) -> str:
    """'' or a warning for the step's message: background residuals this large say the phone walked (translation: parallax the
    fixed-centre model cannot explain), so the world frame and every distance are doubtful. Not fatal: the analysis goes on."""
    st = mc.get("steps") or {}
    sk, tr = (st.get("skeleton") or {}).get("median_px"), (st.get("track") or {}).get("rms_p90_px")
    bad = [f"skeleton median {sk:.2f} px" for _ in [0] if isinstance(sk, (int, float)) and sk > SKEL_MAX_PX] + \
          [f"track p90 {tr:.2f} px" for _ in [0] if isinstance(tr, (int, float)) and tr > TRACK_P90_MAX_PX]
    return ("; WARNING: the camera may have moved (" + ", ".join(bad) + "): the fixed-centre model is doubtful") if bad else ""


def write_frames_json(ctx: Ctx) -> Path:
    z = np.load(ctx.work / "cameras" / f"{CAM}.npz", allow_pickle=True)
    size = tuple(int(v) for v in z["native_size"]) if "native_size" in z.files else (int(2 * z["K"][0, 0, 2]), int(2 * z["K"][0, 1, 2]))
    doc = frames_json_doc(np.asarray(z["t"], float), np.asarray(z["pts"] if "pts" in z.files else z["t"], float), size, ctx.video.name)
    p = ctx.work / "frames" / CAM / "frames.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return p


def step_sam3d_body(ctx: Ctx) -> str:
    """Pod: SAM 3D Body for both dancers on every frame (the frames decoded on the pod), then here: import, QA without a fit,
    and the prior on the take's 30 Hz lattice (realism/sam3db/prior_3db_full.npz)."""
    from ..mono4d import jl_sam3db as S
    S.configure(ctx.take, CAM)
    if not (ctx.work / "frames" / CAM / "frames.json").exists():
        write_frames_json(ctx)
    S.prep(decode_on_pod=True)
    F = len(json.loads((S.INPUT / "meta.json").read_text())["frames"])
    infer_min = 0.355 * 2 * F / 60 * 1.6 + 5         # 0.355 s per person-frame on a 4090 (trial), +60 %, + model load
    job = S.job(max_minutes=round(25 + infer_min), chunk=200, reserve_min=6.0, video=ctx.video, video_pod=ctx.video_pod)
    d = json.loads(job.read_text(encoding="utf-8"))
    for c in d["commands"]:
        if c["name"] == "infer":
            c["timeout_min"] = round(infer_min + 5)
    job.write_text(json.dumps(d, indent=1), encoding="utf-8", newline="\n")
    res = ctx.remote([job])
    if ctx.dry_run:
        return "dry run"
    sid = next((r["sid"] for r in reversed(res) if (r.get("jobs") or {}).get(S.JOB_NAME, {}).get("status") == "ok"), None)
    imp = S.do_import(sid)
    q = S.qa(sheets=False)
    pr = S.prior()
    roles = {r: {"valid": imp["roles"][r]["valid"], "qa_ok": q["roles"][r].get("qa_ok"),
                 "prior_frames": pr["roles"][r]["grid_frames_with_prior"]} for r in S.ROLES}
    return f"SAM 3D Body: {json.dumps(roles)}; prior on {pr['T']} lattice instants"


# ------------------------------------------------------------------------------------------------ fits


def usable_lattice(t: np.ndarray, window: tuple[float, float] | None, hz: float = HZ) -> tuple[int, int]:
    """(k_first, k_last) of the lattice instants t_k = k / hz that lie between the first and the last video frame AND inside the
    take's window: the instants mamma prepare keeps. A cut's first frame need not be at 0 (lk03_mono20: 0.033 s) and the window
    ends on the last frame rounded down: prepare kept k 1..597 where the plan said 0..598 (first fits check, 2026-10-22)."""
    lo, hi = float(np.min(t)), float(np.max(t))
    if window is not None:
        lo, hi = max(lo, float(window[0])), min(hi, float(window[1]))
    return int(math.ceil(lo * hz - 1e-6)), int(math.floor(hi * hz + 1e-6))


def fit_plan(ctx: Ctx) -> dict:
    """The chunks of the take on its fit lattice (from cameras/01.npz: the frames the fit can use, inside the take's window)."""
    z = np.load(ctx.work / "cameras" / f"{CAM}.npz", allow_pickle=True)
    t = np.asarray(z["t"], float)
    win = ctx.cfg.get("window")
    ka, kb = usable_lattice(t, tuple(win) if win else None)
    return {"n_grid": kb + 1, "k_first": ka, "k_last": kb, "duration_s": float(t.max()),
            "chunks": plan_chunks(ctx.take, kb - ka + 1, k_first=ka)}


def write_pinned_betas(ctx: Ctx) -> Path:
    """work/<take>/shape/mamma_pin_betas.json from the statures (+ soles): {"0": lead betas, "1": follow betas} (MAMMA body ids,
    recap_front.step_pin's format) with beta0 solved for the rest height and the rest 0 (+ .source.json)."""
    model = WORKSPACE / "mamma" / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"
    m = np.load(model, allow_pickle=True)
    vt, sd = np.asarray(m["v_template"], float), np.asarray(m["shapedirs"], float)
    st, so = ctx.statures(), ctx.soles()
    doc, src = {}, {}
    for body, role in enumerate(("lead", "follow")):
        target = st[role] + so[role]
        b = beta0_for_height(vt, sd, target)
        doc[str(body)] = [float(x) for x in b]
        src[str(body)] = {"role": role, "stature_m": st[role], "sole_m": so[role], "target_rest_height_m": round(target, 4),
                          "rest_height_m": round(rest_height(vt, sd, b), 4), "n": BETAS_N,
                          "how": "beta0 solved for the rest height (crown to sole), other betas 0: no shape pass (pipeline/mono_route)"}
    out = ctx.work / SHAPE_DIR / "mamma_pin_betas.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    out.with_name("mamma_pin_betas.source.json").write_text(json.dumps(src, indent=1), encoding="utf-8")
    return out


def write_chunk_inputs(ctx: Ctx, c: dict, n_grid: int) -> dict:
    """The farm job json (recap_front.load_chunk via FARM_JOB), the chunk take TOML (farm.chunks.chunk_take) and the prior slice."""
    from ..farm import chunks as FC
    z = np.load(ctx.work / "cameras" / f"{CAM}.npz", allow_pickle=True)
    t = np.asarray(z["t"], float)
    size = tuple(int(v) for v in z["native_size"]) if "native_size" in z.files else (int(2 * z["K"][0, 0, 2]), int(2 * z["K"][0, 1, 2]))
    fps = float(1.0 / np.median(np.diff(t))) if len(t) > 1 else 30.0
    job = {"schema": FC.JOB_SCHEMA, "kind": "full", "what": "chunk", "stages": ["front", "fit", "refined"], "after": [], "world_id": None,
           "world_sha256": None, "pose_id": None, "id": c["id"], "dance": ctx.take, "take": c["take"], "cams": [CAM],
           "chunk": {**{k: c[k] for k in ("id", "k0", "k1", "core_k0", "core_k1", "overlap_k", "take", "T", "core_s")},
                     "kind": "full", "cams": [CAM], "world_id": None, "origin": "pipeline/mono_route", "stages_done": [],
                     "status": "planned", "priority": 0.0,
                     "sync_fixed": {"no_measure": True, "offsets_ms": {CAM: 0.0},
                                    "why": "one camera: the video is its own reference clock (identity alignment), nothing to measure"}},
           "sources": {CAM: {"w": size[0], "h": size[1], "fps": round(fps, 3), "t_first": float(t[0]), "t_last": float(t[-1]),
                             "path": f"src/{ctx.take}/{ctx.video.name}"}},
           "steps": [], "notes": ["written by pipeline/mono_route fits; read by recap_front.load_chunk via FARM_JOB"]}
    jp = ctx.work / "farm" / "jobs" / f"{c['id']}.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
    FC.chunk_take(c["id"], ctx.take, job_path=str(jp))
    from ..mono4d import jl_sam3db as S
    S.configure(ctx.take, CAM)
    pr = S.prior_slice(c["k0"], c["k1"], ctx.work / "realism" / "sam3db" / "chunks" / f"{c['id']}.npz")
    return {"job": jp, "toml": PACKAGE_ROOT / "takes" / f"{c['take']}.toml", "prior": pr}


def chunk_jobs(ctx: Ctx, c: dict, inputs: dict, pin: Path) -> list[Path]:
    """Two linked pod jobs of one chunk (one session): the front, then build + fit + select + refine. The env of the fit is only
    set on its job (mamma prepare must not see MAMMA_POSE_PRIOR / the pins). MAMMA_MONO is set on both: prepare writes the mono
    config with it (the trial's front ran without it; its config was only right because build regenerated it)."""
    from ..farm.chunks import refine_args
    from ..front_pod import MA2D_STRIPPED
    ct, cid = c["take"], c["id"]
    cw = ctx.work.parent / ct
    pw = f"{POD_WS}/{ctx.ws(cw)}"
    farm_job = f"{POD_WS}/{ctx.ws(inputs['job'])}"
    base_env = {"DANCECAP_POD": "1", "PYOPENGL_PLATFORM": "egl", "MAMMA_MONO": "1", "HM_FRONT_DANCE": ctx.take,
                "HM_FRONT_CAPTURE": ctx.take, "FARM_JOB": farm_job, "PYTHONUNBUFFERED": "1"}
    common_in = [ctx.ws(ctx.work / "cameras") + "/", ctx.ws(ctx.work / "people" / "teachers_01.npz"),
                 ctx.ws(ctx.work / "people" / "masks_01.npz"), ctx.ws(ctx.work / "clock" / "alignment.json"),
                 ctx.ws(inputs["job"]), ctx.video_input()]
    T = int(c["T"])
    front = ctx.write_job({
        "name": f"{ct}_front"[:60], "max_minutes": 25, "needs": ["ma2d", "mamma"], "min_vram_gb": 24,
        "inputs": common_in + [{"local": str((WORKSPACE / MA2D_STRIPPED).resolve()), "pod": MA2D_POD}],
        "exclude": EXCLUDE + GENDER_EXCLUDE,
        "env": base_env, "setup": [GENDER_LINKS],
        "commands": [{"name": "frames", "timeout_min": 10, "cmd": f"python -m dancecap.recap_front frames {cid}"},
                     {"name": "cameras", "timeout_min": 5, "cmd": f"python -m dancecap.jl_front cameras {cid}"},
                     {"name": "masks", "timeout_min": 5, "cmd": f"python -m dancecap.jl_front masks {cid}"},
                     {"name": "ma2d", "timeout_min": 20, "cmd": f"python -m dancecap.front_pod ma2d {ct} {CAM}"},
                     {"name": "sync", "timeout_min": 5, "cmd": f"python -m dancecap.recap_front sync {cid}"},
                     {"name": "prepare", "timeout_min": 15,
                      "cmd": f"python -m dancecap mamma prepare {ct} --skip-2d --set c1={CAM} --coverage union --grid-anchor 0.0 "
                             f"--grid-k={c['k0']}:{c['k1']}"},
                     {"name": "mono_init", "timeout_min": 5, "cmd": f"python -m dancecap.jl_front mono-init {cid}"},
                     {"name": "qa_mask", "timeout_min": 5, "cmd": f"python -m dancecap.recap_front qa-mask {cid}"}],
        "outputs": [f"{ctx.ws(cw)}/mamma/prepare.json", f"{ctx.ws(cw)}/mamma/mono_init.json", f"{ctx.ws(cw)}/mamma/stature.json",
                    f"{ctx.ws(cw)}/mamma/qa_mask.json"],
        "verify": [f"{ctx.ws(cw)}/mamma/prepare.json"],
        "notes": [f"pipeline/mono_route fits: chunk {cid} of {ctx.take} (k {c['k0']}..{c['k1']}): the front"]})
    fit_min = round(0.045 * T + 10)
    fit = ctx.write_job({
        "name": f"{ct}_fit"[:60], "max_minutes": fit_min + 30, "needs": ["mamma"], "min_vram_gb": 24, "after": [f"{ct}_front"[:60]],
        "inputs": [ctx.ws(pin), ctx.ws(inputs["prior"])],
        "exclude": EXCLUDE + GENDER_EXCLUDE,
        "env": {**base_env, "MAMMA_MONO_INIT": f"{pw}/mamma/mono_init.npz", "MAMMA_FREEZE_BETAS": "1",
                "MAMMA_INIT_BETAS": f"{POD_WS}/{ctx.ws(pin)}", "MAMMA_POSE_PRIOR": f"{pw}/mamma/prior_3db.npz"},
        "setup": [GENDER_LINKS, "nvidia-smi --query-gpu=name,memory.total --format=csv"],
        "commands": [{"name": "build", "timeout_min": 5,
                      "cmd": f"python -m dancecap.pipeline.mono_route fit-build --take {ctx.take} --cid {cid}"},
                     {"name": "fit", "timeout_min": fit_min, "cmd": f"python -m dancecap mamma fit {ct} --tag c1"},
                     {"name": "select", "timeout_min": 5, "cmd": f"python -m dancecap mamma select {ct} --no-qa"},
                     {"name": "refine", "timeout_min": 15,
                      "cmd": f"python -m dancecap.pose_refine {ct} --src smplx --dst smplx_rf2_noslip {refine_args(None)}"}],
        "outputs": [f"{ctx.ws(cw)}/build.json", f"{ctx.ws(cw)}/mamma/prepare.json", f"{ctx.ws(cw)}/mamma/ma3d_config.yaml",
                    f"{ctx.ws(cw)}/mamma/ma3d_config_3db.yaml", f"{ctx.ws(cw)}/mamma/fits/", f"{ctx.ws(cw)}/mamma/logs/",
                    f"{ctx.ws(cw)}/mamma/select.json", f"{ctx.ws(cw)}/mamma/report.json", f"{ctx.ws(cw)}/mamma/ma_2d/",
                    f"{ctx.ws(cw)}/smplx/", f"{ctx.ws(cw)}/smplx_rf2_noslip/"],
        "verify": [f"{ctx.ws(cw)}/smplx/lead.npz", f"{ctx.ws(cw)}/smplx_rf2_noslip/follow.npz"],
        "notes": [f"pipeline/mono_route fits: chunk {cid} of {ctx.take}: MAMMA_MONO + SAM 3D Body prior (weight 0.5), betas pinned "
                  "from the statures; select; pose_refine (chunks.refine_args)"]})
    return [front, fit]


def step_fits(ctx: Ctx) -> str:
    """Pod: every chunk in its own session, side by side (independent job pairs -> remote run --parallel)."""
    plan = fit_plan(ctx)
    pin = write_pinned_betas(ctx)
    jobs = []
    for c in plan["chunks"]:
        ins = write_chunk_inputs(ctx, c, plan["n_grid"])
        jobs += chunk_jobs(ctx, c, ins, pin)
    (ctx.work / "farm").mkdir(parents=True, exist_ok=True)
    (ctx.work / "farm" / "mono_plan.json").write_text(json.dumps({**plan, "pin": str(pin)}, indent=1), encoding="utf-8")
    T_max = max(c["T"] for c in plan["chunks"])
    ctx.remote(jobs, parallel=len(plan["chunks"]) > 1, max_minutes=round(25 + 0.045 * T_max + 10 + 30 + 20))
    if ctx.dry_run:
        return "dry run"
    rep = []
    for c in plan["chunks"]:
        r = _json(ctx.work.parent / c["take"] / "mamma" / "report.json") or {}
        rep.append(f"{c['id']} k {c['k0']}..{c['k1']}" + (f" reproj {r.get('reproj_px') or r.get('median_px')}" if r else ""))
    return f"{len(plan['chunks'])} chunk(s) fitted: " + "; ".join(rep)


def fit_build(take: str, cid: str) -> dict:
    """(on the fit pod) The chunk's MAMMA_MONO + SAM 3D Body prior configs (mono4d.jl_refit._configs: ma3d_config.yaml with
    MAMMA_MONO, ma3d_config_3db.yaml with pose_prior_loss), prepare.json pointed at the prior config, and the prior slice
    checked against the prepared grid and copied to <chunk>/mamma/prior_3db.npz. The 2026-10-21 trial's trial_fit.build, generic."""
    import copy
    from ..mono4d import jl_refit as JR
    ct = f"{take}_{cid}"
    cw = PACKAGE_ROOT / "work" / ct
    dst = cw / "mamma"
    prep = json.loads((dst / "prepare.json").read_text(encoding="utf-8"))
    if "refit" in prep:   # rebuilt on a resumed session: start from the front's prepare.json
        prep = json.loads((dst / "prepare.front.json").read_text(encoding="utf-8"))
    gk = [int(v) for v in prep["grid"]["trim"]["grid_k"]]
    tg = np.asarray(prep["grid"]["t"], float)
    job = json.loads((PACKAGE_ROOT / "work" / take / "farm" / "jobs" / f"{cid}.json").read_text(encoding="utf-8"))
    k0, k1 = int(job["chunk"]["k0"]), int(job["chunk"]["k1"])
    # prepare may keep fewer instants than planned (no camera frame within its tolerance at an edge): the prior follows the
    # prepared grid, as long as it lies inside the planned one and on the lattice
    if not (k0 <= gk[0] <= gk[1] <= k1) or len(tg) != gk[1] - gk[0] + 1 or np.abs(tg - np.arange(gk[0], gk[1] + 1) / HZ).max() > 1e-9:
        raise SystemExit(f"prepared grid {gk} is not inside k {k0}..{k1} on the {HZ:g} Hz lattice")
    (dst / "prepare.front.json").write_text(json.dumps(prep, indent=1), encoding="utf-8")
    rep = {"configs": JR._configs(dst)}
    src = PACKAGE_ROOT / "work" / take / "realism" / "sam3db" / "chunks" / f"{cid}.npz"
    z = np.load(src, allow_pickle=True)
    kz = np.asarray(z["k"], int)
    sel = (kz >= gk[0]) & (kz <= gk[1])
    d = {n: (np.asarray(z[n])[sel] if np.ndim(z[n]) and len(z[n]) == len(kz) else z[n]) for n in z.files}
    if len(d["t"]) != len(tg) or np.abs(np.asarray(d["t"], float) - tg).max() > 1e-9:
        raise SystemExit(f"the prior slice {src.name} does not cover the prepared grid {gk}")
    d["source"] = np.array(f"{src.name} k {gk[0]}..{gk[1]}")
    np.savez_compressed(dst / "prior_3db.npz", **d)
    p2 = copy.deepcopy(prep)
    p2["sets"][JR.SET]["config"] = JR.CFG_3DB
    p2["sets"][JR.SET]["pose_prior"] = {"weight": JR.PRIOR_W, "npz": "prior_3db.npz", "source": str(src)}
    p2["config"] = "ma3d_config.yaml"
    p2["refit"] = {"by": "pipeline/mono_route fit-build", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (dst / "prepare.json").write_text(json.dumps(p2, indent=1), encoding="utf-8")
    rep["prior"] = {"frames_with_prior": [int((np.asarray(d["conf"])[:, b] > 0).sum()) for b in range(2)], "grid_k": gk,
                    "planned_k": [k0, k1]}
    rep["seq"] = prep["sets"][JR.SET].get("seq")
    (cw / "build.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    log(f"fit-build {ct}: {json.dumps(rep, default=str)[:400]}")
    return rep


# ------------------------------------------------------------------------------------------------ post, analyses, export


def step_post(ctx: Ctx) -> str:
    """Local: the stitcher over the chunks (blend in the overlaps, the per-frame flags: absent, ...) into smplx/ and
    smplx_rf2_noslip/, then jl_repair (monocular depth spikes). Seconds of CPU."""
    plan = _json(ctx.work / "farm" / "mono_plan.json") or fit_plan(ctx)
    doc = stitch_list_doc(ctx.take, plan["chunks"], plan["duration_s"])
    sl = ctx.work / "stitch_list.json"
    sl.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    # --gate-exit report: a seam gate that fails is written into stitch.json (the verdict, the refit advice), not a failed step
    ctx.local(["-m", "dancecap.stitch", ctx.take, "--chunks", str(sl), "--partial", "--force", "--select-weights",
               "arm=0.3,fingers=0.3", "--no-sheets", "--gate-exit", "report"], timeout_min=15)
    ctx.local(["-m", "dancecap.jl_repair", str(ctx.work / "smplx"), str(ctx.work / "smplx_rf2_noslip")], timeout_min=10)
    st = _json(ctx.work / "stitch.json") or {}
    gate = st.get("gate") or {}
    return f"stitched {len(plan['chunks'])} chunk(s) -> smplx/, smplx_rf2_noslip/; repaired; stitch gate " + (
        "passed" if gate.get("passed") else f"FAILED (reported, not fatal): {json.dumps(gate.get('why'), default=str)[:200]}")


def analyses_job(ctx: Ctx) -> Path:
    """One pod job, the four analyses in order (timing writes the grid the others read), on the stitched + repaired motion."""
    from ..farm.jl_analysis import no_beat_arg
    seg = _json(ctx.work / "song" / "segments.json") or {"no_beat": []}
    nb = no_beat_arg(seg)
    beats = f"work/{ctx.take}/song/beats_heat.json"
    t_args = (f"--beats {beats} --smplx-dir smplx_rf2_noslip --no-refine --mc 60 --workers 8 --mc-corr-noise-mm 10 --sync-ms 15 "
              f"--n-cams 1" + (f" --no-beat {nb}" if nb else ""))
    w = ctx.ws(ctx.work)
    return ctx.write_job({
        "name": f"{ctx.take}_analyses"[:60], "max_minutes": 40, "needs": [], "min_vram_gb": 24,
        # base + open3d (implied by physics) is the whole environment; physics / counterbalance read the MAMMA locked-head
        # neutral model (physics_body.MAMMA_LOCKED_HEAD), which a job without the mamma feature does not get by default
        "inputs": [f"{w}/smplx/", f"{w}/smplx_rf2_noslip/", f"{w}/song/", f"{w}/clock/alignment.json",
                   "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz", "smplx/SMPLX_NEUTRAL.npz"],
        "exclude": EXCLUDE + GENDER_EXCLUDE,
        "setup": [GENDER_LINKS],
        "env": {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
        "commands": [{"name": "timing", "timeout_min": 15, "cmd": f"python -m dancecap timing {ctx.take} {t_args}"},
                     {"name": "physics", "timeout_min": 15, "cmd": f"python -m dancecap physics {ctx.take} --beats {beats}"},
                     {"name": "counterbalance", "timeout_min": 10,
                      "cmd": f"python -m dancecap.counterbalance detect --take {ctx.take} --smplx-dir smplx_rf2_noslip --no-review"},
                     {"name": "floorcraft", "timeout_min": 10,
                      "cmd": f"python -m dancecap.floorcraft {ctx.take} --no-review --smplx-dir smplx_rf2_noslip"}],
        "outputs": [f"{w}/timing/", f"{w}/physics/", f"{w}/counterbalance/", f"{w}/floorcraft/"],
        "verify": [f"{w}/timing/timing.json", f"{w}/physics/physics.json", f"{w}/counterbalance/counterbalance.json",
                   f"{w}/floorcraft/floorcraft.json"],
        "notes": [f"pipeline/mono_route analyses of {ctx.take} (farm/jl_analysis's commands, generic)"]})


def step_analyses(ctx: Ctx) -> str:
    ctx.remote([analyses_job(ctx)])
    if ctx.dry_run:
        return "dry run"
    out = []
    for k in ("timing", "physics", "counterbalance", "floorcraft"):
        out.append(f"{k} {'ok' if (ctx.work / k / f'{k}.json').exists() else 'missing'}")
    return "analyses: " + ", ".join(out)


def step_avatars(ctx: Ctx) -> str:
    """LHM++ splat avatars (optional; nothing in the export reads them). The generic job needs reference crops chosen from the
    masks and the fit, which is not written yet: this step reports that and is skipped (never fails the run)."""
    raise SkipStep("LHM++ avatars for a new take need generic reference crops (masks + fit); not built yet - the export uses "
                   "plain bodies")


def step_export(ctx: Ctx) -> str:
    """Local: the Unity capture into work/<take>/export/<capture>/ (--no-install: the Unity project is never touched here)."""
    from .. import jl_export as JE
    cap = JE.default_capture(ctx.take)
    ctx.local(["-m", "dancecap.jl_export", "--take", ctx.take, "--capture", cap, "--no-install"], timeout_min=15)
    rep = _json(ctx.work / "export" / f"{cap}_jl_export.json") or {}
    v = rep.get("validation") or {}
    return f"export {cap}: {'OK' if v.get('ok') else 'see report'} ({len(v.get('passed') or [])} checks passed, " \
           f"{len(v.get('failed') or [])} failed) -> work/{ctx.take}/export/{cap}/"


def step_film(ctx: Ctx) -> str:
    raise SkipStep("the film needs the Unity editor, which another workflow owns: render it with the film tools when the editor "
                   "is free (the capture is in work/<take>/export/)")


class SkipStep(Exception):
    """The step cannot run here for a stated reason: the driver marks it skipped (optional steps only)."""


STEPS = {"beats": step_beats, "camera": step_camera, "people": step_people, "world": step_world, "sam3d_body": step_sam3d_body,
         "fits": step_fits, "post": step_post, "analyses": step_analyses, "avatars": step_avatars, "export": step_export,
         "film": step_film}
SKIP_RC = 3                 # the CLI's exit code for SkipStep (analyze marks the step skipped with its message)


def _json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_step(step: str, take: str, run: str | None = None, dry_run: bool = False) -> str:
    ctx = Ctx(take, run, step, dry_run)
    t0 = time.time()
    msg = STEPS[step](ctx)
    log(f"{step}: {msg} ({time.time() - t0:.0f} s)")
    return msg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline.mono_route", description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=sorted(STEPS) + ["fit-build"])
    ap.add_argument("--take", required=True)
    ap.add_argument("--run", help="the analysis run (tags the pod sessions)")
    ap.add_argument("--cid", help="fit-build: the chunk id")
    ap.add_argument("--dry-run", action="store_true", help="write the pod jobs, start nothing")
    a = ap.parse_args(argv)
    if a.step == "fit-build":
        fit_build(a.take, a.cid)
        return 0
    try:
        msg = run_step(a.step, a.take, a.run, a.dry_run)
    except SkipStep as e:
        print(f"SKIP: {e}", flush=True)
        return SKIP_RC
    print(f"RESULT: {msg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
