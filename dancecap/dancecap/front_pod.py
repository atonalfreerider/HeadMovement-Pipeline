"""Capture front-end of a take on ONE Runpod pod: everything between `detect` and the MAMMA fit, in parallel.

    python -m dancecap.front_pod write-job larissa_kadu_38      # local: work/<take>/front/pod_job.json (dancecap.pod_job/1)
    python -m dancecap.front_pod run larissa_kadu_38            # on the pod (the job's command)
    python -m dancecap remote run --jobs dancecap/work/larissa_kadu_38/front/pod_job.json --max-cost 5

Why: the workstation has 4 logical CPUs and a shared 11 GB Turing GPU; a 38 s take is ~10.4k canonical frames over 8
phones (the 6 s take was 1.4k). The pod (A100 80 GB, ~30 vCPU, >100 GB RAM) runs every heavy step at once:

  label:<cam>  dancers-labels per camera (dancers.labels_for_cam; candidate tracks + colour-rule role label), merged
               into dancers/labels.json + labels_review.jpg exactly like dancers.run_labels
  loc:<cam>    per-frame localisation (ALIKED + LightGlue + pycolmap P4Pf/P3P, zoom) of the main-model cameras:
               python -m dancecap cameras <take> --stages loc --cams <cam>   (one process per camera)
  sub:<cam>    the attached cameras' sub-model localisation (python -m dancecap.cameras.reuse presolve-sub), cached
               under the key stage_attach uses
  sam:<cam>    SAM 2.1 dancer masks, identity repair, body/hair split, QA (python -m dancecap.dancers_pod one) as soon
               as the camera's label exists; scheduled by a host-RAM budget (cam02 at 2302 x 1080p needs ~35 GB)
  ma2d:<cam>   MAMMA ma_2d on that camera's body masks as soon as its masks exist (own log mamma/logs/ma_2d_<cam>.log;
               writes ma_2d_native/<cam>.npz + _boxes.npy + manifest like mamma_dyn.run_ma2d)
  merge        dancers_pod merge -> dancers/roles.json, report.json, review/dancers_body_hair.jpg
  cameras      python -m dancecap cameras <take> --stages attach,world,qa (needs roles.json: attach + scale)
  sync         python -m dancecap sync <take> (video offsets on the take's own clocks; required by prepare)
  chain        world_scale chain after sync -> scale/chain_after_sync.json (coarse scale cross-check)
  prepare      python -m dancecap mamma prepare <take> --skip-2d --set c<n>=<cams> -> grid inputs for the fit

GPU tasks share the A100 under CUDA MPS (DANCECAP_GPU_SHARED=1: no work/gpu.lock); a GPU task starts only when >= 8 GB
are free. Every task logs to work/<take>/front/logs/<task>.log; work/<take>/front/front_pod.json is the status
(per task rc, seconds, start/end, host CPUs/RAM, what failed and what was skipped because of it). A failed task never
stops independent ones. Exit code 0 only when prepare succeeded.

clock_studio window --take needs the source videos (ffprobe): it runs locally after the download.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import progress as _progress  # dancecap.progress/1 (stdlib; no-op unless DANCECAP_PROGRESS)
from .take import PACKAGE_ROOT, WORKSPACE, Take

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"  # the local checkpoint (2026-10-06)
MA2D_STRIPPED = "dancecap/work/runpod/weights/ma_2d/mamma_mask_full_cvpr.ckpt"  # state_dict only (540 MB, not 1.6 GB)
TORCH_HOME_REL = "dancecap/work/runpod/torch_home"  # ALIKED + LightGlue weights (uploaded from the local hub cache)
GPU_FREE_MIN_MB = 8000
MPS_PIPE = Path("/tmp/dancecap_mps_front")


# ------------------------------------------------------------------------------------------------ host facts


def cgroup_cpus() -> int:
    """vCPU quota of the container (os.cpu_count() and the affinity mask show 64-256 host cores on a pod; the CFS
    quota is what we get: 8.5 CPUs on the 2026-10-06 A100 PCIe community pod). cgroup v2, then v1."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            return max(1, -(-int(q) // int(p)))
    except (OSError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return max(1, -(-q // p))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def mem_gb() -> float:
    """Usable host RAM (GB): the cgroup limit when set, else MemAvailable."""
    try:
        lim = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if lim != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            return (int(lim) - used) / 2**30
    except (OSError, ValueError):
        pass
    try:  # cgroup v1 (the host's MemAvailable can be ~0.9 TB while the container may use 167 GB)
        lim = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text())
        used = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())
        cache = 0
        for line in Path("/sys/fs/cgroup/memory/memory.stat").read_text().splitlines():
            if line.startswith("total_inactive_file "):
                cache = int(line.split()[1])
        if lim < 1 << 50:
            return (lim - used + cache) / 2**30
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 32.0


def gpu_free_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def mps_start(env: dict) -> dict | None:
    if not shutil.which("nvidia-cuda-mps-control"):
        return None
    logd = Path("/tmp/dancecap_mps_front_log")
    MPS_PIPE.mkdir(exist_ok=True)
    logd.mkdir(exist_ok=True)
    menv = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(MPS_PIPE), CUDA_MPS_LOG_DIRECTORY=str(logd))
    if subprocess.run(["nvidia-cuda-mps-control", "-d"], env=menv).returncode != 0:
        return None
    env.update(CUDA_MPS_PIPE_DIRECTORY=str(MPS_PIPE), CUDA_MPS_LOG_DIRECTORY=str(logd))
    return menv


# ------------------------------------------------------------------------------------------------ workers


def label_one(take_name: str, cam_id: str) -> None:
    """dancers-labels for one camera -> dancers/label_<cam>.json + candidates/<cam>.npz (run_labels' per-camera part)."""
    import numpy as np

    from .dancers import Cam, DancersConfig, labels_for_cam
    take = Take.load(take_name)
    out = take.work / "dancers"
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    cam = Cam.load(take.work, cam_id)
    tracks, ev, lab = labels_for_cam(cam, DancersConfig())
    np.savez_compressed(out / "candidates" / f"{cam_id}.npz", idx=tracks["idx"], boxes=tracks["boxes"],
                        llr=ev["llr"], a_is_follow=ev["a_is_follow"], swap_post=ev["swap_post"],
                        pair_iou=ev["pair_iou"])
    (out / f"label_{cam_id}.json").write_text(json.dumps(lab, indent=1))
    print(f"[dancers] {cam_id}: label frame {lab['frame']} conf {lab['confidence']:.2f} (llr {lab['frame_llr']:.1f}, "
          f"agreement {lab['evidence_agreement']:.2f}, track switches {lab['track_switches']}) cues {lab['cues']}",
          flush=True)


def dancer_one(take_name: str, cam_id: str) -> None:
    """label + SAM 2.1 + identity repair + body/hair + QA for one camera in ONE process: the frames are decoded once and
    the label's tracks/evidence are reused by dancers.run_camera (which otherwise re-reads every frame and recomputes
    them). Writes dancers/label_<cam>.json, candidates/<cam>.npz and pod_cam_<cam>.json (dancers_pod.merge input)."""
    import numpy as np

    from . import dancers as D
    from .dancers_sam import build_predictor
    take = Take.load(take_name)
    work = take.work
    out = work / "dancers"
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    cfg = D.DancersConfig()
    cam = D.Cam.load(work, cam_id)
    t0 = time.time()
    imgs = D.read_frames(cam.files)
    tracks, ev, lab = D.labels_for_cam(cam, cfg, imgs)
    np.savez_compressed(out / "candidates" / f"{cam_id}.npz", idx=tracks["idx"], boxes=tracks["boxes"],
                        llr=ev["llr"], a_is_follow=ev["a_is_follow"], swap_post=ev["swap_post"],
                        pair_iou=ev["pair_iou"])
    (out / f"label_{cam_id}.json").write_text(json.dumps(lab, indent=1))
    print(f"[dancers] {cam_id}: label frame {lab['frame']} conf {lab['confidence']:.2f} (llr {lab['frame_llr']:.1f}, "
          f"agreement {lab['evidence_agreement']:.2f}, track switches {lab['track_switches']}) cues {lab['cues']} "
          f"({time.time() - t0:.0f} s)", flush=True)
    ov = out / "labels_override.json"
    label = D.apply_override({cam_id: lab}, json.loads(ov.read_text()) if ov.exists() else {})[cam_id]
    files = cam.files
    orig_read = D.read_frames
    D.read_frames = lambda f, idx=None: imgs if (idx is None and list(f) == files) else orig_read(f, idx)
    D.labels_for_cam = lambda c, cfg_, imgs_=None: (tracks, ev, lab)
    held: dict = {}

    def predictor():
        if "p" not in held:
            held["p"] = build_predictor()
        return held["p"]
    r = D.run_camera(cam, label, predictor, cfg, reuse_sam=True)  # dancers/cache/<cam>_sam.npz when its label matches
    r["report"]["wall_seconds_pod"] = round(time.time() - t0, 1)
    (out / f"pod_cam_{cam_id}.json").write_text(json.dumps(
        {"frames": r["frames"], "report": r["report"],
         "sam_overrides": getattr(held.get("p"), "dancecap_overrides", None)}))
    rp = r["report"]
    print(f"[dancers] {cam_id}: {rp['frames']} frames, SAM {rp['sam_s_per_frame']:.2f} s/frame "
          f"(total {rp['total_s_per_frame']:.2f}), anchors {rp['anchors']}, embrace {rp['embrace_frames']}, "
          f"swaps repaired {rp['swaps_repaired']}, tight {100 * rp['tight_rate']:.1f}%"
          + (f", IDENTITY FLAGS {rp['identity_flags']}" if rp["identity_flags"] else ""), flush=True)


def loc_ref_cams(take, cam: str) -> list[str]:
    """Cameras whose map images loc-part localises `cam` against. A take with a [cameras] map table (its map settings
    are its own, the two-view takes) gets every camera, exactly as stage_loc: Localizer.candidates falls back to the
    other cameras' nearest images when a frame's own-camera inliers are few (an orbiting Steadicam). Other takes keep
    loc-part's own camera only, as they were localised."""
    if "map" in ((take.cfg or {}).get("cameras") or {}):
        ids = [s.id for s in take.sources]  # stage_loc's order (the references' order breaks ties)
        return ids if cam in ids else ids + [cam]
    return [cam]


def loc_part(take_name: str, cam: str, a: int, b: int) -> None:
    """Per-frame localisation of window frames a..b-1 of one camera (stage_loc's Localizer + localize_camera, against
    the map images of loc_ref_cams; the part's first frame starts without the previous frame's correspondences) ->
    cameras/cache/locpart_<cam>_<a>_<b>.npz (stage_loc's raw format). The long-pole camera (cam02, 2302 frames) is
    localised in two halves side by side.
    A part always localises EVERY frame: the keyframes-only decision of a locked tripod belongs to the whole camera
    (parts deciding on their own keyframes could disagree and join into a cache valid for no model), so build_tasks
    never splits a locked tripod - its keyframe localisation is ~10 % of the work."""
    from .cameras.features import Features, FeatureSettings, list_images
    from .cameras.localize import Localizer, localize_camera
    from .cameras.mapping import map_settings
    from .cameras.pipeline import _policy_name, _save_raw, load_model, loc_settings, map_images
    from .cameras.typed import loc_marker
    take = Take.load(take_name)
    work = take.work.resolve()
    recs = list_images(work, loc_ref_cams(take, cam))  # frames.json / keyframes.json only: light
    F = Features(work, work / "cameras" / "cache", FeatureSettings())
    model = load_model(work)
    images, _ = map_images(work, recs, map_settings(take.cfg))  # the map's own settings: references = its images
    L = Localizer(F, model, images, loc_settings(take))
    if cam not in L.cams_reg:
        raise SystemExit(f"[cameras] {cam}: not in the main model")
    frames = [r for r in recs[cam] if r.kind == "win"][a:b]
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    t0 = time.time()
    st = take.camera_types([cam]).get(cam)
    poses = localize_camera(L, frames, int(meta["width"]), int(meta["height"]))
    out = work / "cameras" / "cache" / f"locpart_{cam}_{a:05d}_{b:05d}.npz"
    tmp = out.with_name(out.stem + ".part.npz")
    _save_raw(tmp, frames, poses, loc_marker("all", st.cls if st is not None else "handheld_zoom", _policy_name(None),
                                             len(frames)))
    os.replace(tmp, out)
    print(f"[cameras] {cam} frames {a}-{b - 1}: localised {sum(p.ok for p in poses)}/{len(frames)} "
          f"({time.time() - t0:.0f}s) -> {out.name}", flush=True)


def join_raw(ds: list[dict]) -> dict:
    """Concatenate stage_loc raw dicts (pipeline._save_raw layout) in frame order; corr_off re-based."""
    import numpy as np
    out = {k: np.concatenate([d[k] for d in ds]) for k in ("t", "ok", "R", "tvec", "f", "k1", "n_inliers", "reproj_px",
                                                           "source", "corr_xy", "corr_xyz")}
    off, base = [np.zeros(1, np.int64)], 0
    for d in ds:
        off.append(np.asarray(d["corr_off"][1:], np.int64) + base)
        base += int(d["corr_off"][-1])
    out["corr_off"] = np.concatenate(off)
    return out


def loc_join(take_name: str, cam: str, parts: list[tuple[int, int]]) -> None:
    """locpart files (in frame order) -> cameras/cache/loc_<cam>.npz + cameras/<cam>.npz (stage_smooth), as stage_loc."""
    import numpy as np

    from .cameras.pipeline import stage_smooth
    take = Take.load(take_name)
    work = take.work.resolve()
    cache = work / "cameras" / "cache"
    ds = [dict(np.load(cache / f"locpart_{cam}_{a:05d}_{b:05d}.npz")) for a, b in parts]
    n = len(json.loads((work / "frames" / cam / "frames.json").read_text())["frames"])
    if sum(len(d["t"]) for d in ds) != n or [a for a, _ in parts][0] != 0:
        raise SystemExit(f"[cameras] {cam}: parts {parts} do not cover the {n} window frames")
    out = join_raw(ds)
    from .cameras.typed import LOC_MARKER_KEYS, loc_mode
    modes = {loc_mode(d) for d in ds}
    if "keyframes" in modes:  # parts written before loc_part always localised every frame: never join a mixed cache
        raise SystemExit(f"[cameras] {cam}: parts {parts} were localised on keyframes only (an older loc-part): "
                         f"re-localise them (delete cameras/cache/locpart_{cam}_*.npz and re-run the loc jobs)")
    for k in LOC_MARKER_KEYS:  # the marker of the parts (they agree: every frame, the same class / policy)
        if k in ds[0]:
            out[k] = ds[0][k] if k != "loc_localised" else np.int64(sum(int(d.get(k, len(d["t"]))) for d in ds))
    np.savez(cache / f"loc_{cam}.npz", **out)
    stage_smooth(work, [cam], 0.04, 4.0)  # cli defaults (--sigma 0.04, --max-reproj 4.0), as `cameras --stages loc`
    print(f"[cameras] {cam}: joined {len(parts)} parts, {int(out['ok'].sum())}/{n} frames ok", flush=True)


def merge_label(work: Path, cam_id: str) -> None:
    out = work / "dancers"
    path = out / "labels.json"
    labels = json.loads(path.read_text()) if path.exists() else {}
    labels[cam_id] = json.loads((out / f"label_{cam_id}.json").read_text())
    path.write_text(json.dumps(labels, indent=1))


def labels_review(take_name: str) -> None:
    import cv2

    from .dancers import Cam, load_labels, review_tile
    from .dancers_qa import grid
    take = Take.load(take_name)
    final = load_labels(take.work)
    tiles = [review_tile(Cam.load(take.work, c), final[c]) for c in sorted(final)]
    cv2.imwrite(str(take.work / "dancers" / "labels_review.jpg"), grid(tiles, cols=4), [cv2.IMWRITE_JPEG_QUALITY, 88])


def ma2d_one(take_name: str, cam: str) -> None:
    """mamma_dyn.run_ma2d for ONE camera with its own log (several cameras run side by side on the pod)."""
    import numpy as np

    from . import mamma_dyn as md
    work = md.take_work(take_name)
    native = work / "mamma" / "ma_2d_native"
    native.mkdir(parents=True, exist_ok=True)
    stage = work / "mamma" / "ma2d_masks"
    man = md.ma2d_manifest(work, cam)
    for f in (native / f"{cam}.npz", native / f"{cam}_boxes.npy", native / f"{cam}.manifest.json"):
        f.unlink(missing_ok=True)
    how = md.link_dir(stage / cam / "masks", md.mask_dir(work, cam))
    print(f"[mamma] ma_2d {cam}: masks {man['mask_dir']} ({man['n_masks']} files, {how})", flush=True)
    cmd = [md.python(), "run_ma_2d.py", "--images_root_dir", work / "frames", "--cam_names", cam,
           "--mask_path", stage, "--out_folder", native, "--config_path", md.MA2D_CONFIG,
           "--weights", md.MA2D_WEIGHTS, "--downsampled-verts", md.VERTS_512, "--disable-visualizations"]
    stats = md._run_logged(cmd, md.MAMMA_ROOT / "landmarks", work / "mamma" / "logs" / f"ma_2d_{cam}.log",
                           f"mamma ma_2d {cam}")
    n = np.load(native / f"{cam}.npz")["landmarks"].shape[0]
    np.save(native / f"{cam}_boxes.npy", md.mask_boxes(work, cam, n))
    (native / f"{cam}.manifest.json").write_text(json.dumps(dict(
        man, created=time.strftime("%Y-%m-%dT%H:%M:%S"), host="pod" if os.environ.get("DANCECAP_POD") else "local",
        weights_note="state_dict-only copy of the checkpoint on the pod (identical tensors, sha256 in "
                     "work/runpod/weights/ma_2d/*.json); masks_sig/mtimes are the pod's: run prepare with --skip-2d "
                     "locally", seconds=stats["seconds"]), indent=1))
    print(f"[mamma] ma_2d {cam}: {n} frames in {stats['seconds']:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ memory


def mem_state() -> tuple[float | None, float | None]:
    """(working set GB, limit GB) of the container: cgroup usage minus reclaimable file cache (v2, then v1)."""
    try:
        cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
        lim = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        inact = 0
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            if line.startswith("inactive_file "):
                inact = int(line.split()[1])
        return (cur - inact) / 2**30, (int(lim) / 2**30 if lim != "max" else None)
    except (OSError, ValueError):
        pass
    try:
        cur = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())
        lim = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text())
        inact = 0
        for line in Path("/sys/fs/cgroup/memory/memory.stat").read_text().splitlines():
            if line.startswith("total_inactive_file "):
                inact = int(line.split()[1])
        return (cur - inact) / 2**30, (lim / 2**30 if lim < 1 << 50 else None)
    except (OSError, ValueError):
        return None, None


def tree_rss_gb(pid: int) -> float:
    """RSS of a process and all its descendants (Linux /proc; 0 elsewhere)."""
    proc = Path("/proc")
    if not proc.exists():
        return 0.0
    kids: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            st = (d / "stat").read_text()
            ppid = int(st[st.rindex(")") + 2:].split()[1])
            kids.setdefault(ppid, []).append(int(d.name))
            for line in (d / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss[int(d.name)] = int(line.split()[1])
        except (OSError, ValueError):
            continue
    tot, todo = 0, [pid]
    while todo:
        q = todo.pop()
        tot += rss.get(q, 0)
        todo += kids.get(q, [])
    return tot / 2**20


# ------------------------------------------------------------------------------------------------ scheduler


def _alive(pid: int | None) -> bool:
    """pid exists and is not a zombie (Linux /proc; os.kill elsewhere)."""
    if not pid:
        return False
    st = Path(f"/proc/{pid}/stat")
    if st.exists():
        try:
            s = st.read_text()
            return s[s.rindex(")") + 2] != "Z"
        except (OSError, ValueError, IndexError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


@dataclass
class Task:
    name: str
    cmd: object                        # subprocess argv, a callable returning it at start, or None (`fn`)
    nice: int = 0                      # CPU priority (the CFS quota is small: the long-pole camera keeps nice 0)
    deps: list[str] = field(default_factory=list)  # must be ok
    after: list[str] = field(default_factory=list)  # must be finished (any state)
    kind: str = "cpu"
    mem_gb: float = 2.0                # estimate, used for admission until the task is measured
    gpu: bool = False
    optional: bool = False
    critical: bool = False             # long pole: never killed by the memory brake, its memory is reserved early
    fn: object = None                  # in-process action (quick bookkeeping)
    on_ok: object = None               # in-process callback after success
    done_if: object = None             # callable -> True when the task's outputs already exist (earlier session)
    timeout_min: float = 90.0
    retries: int = 1                   # re-runs after a failure (a killed driver leaves killed workers behind)
    state: str = "waiting"             # waiting | running | ok | failed | skipped
    rc: object = None
    proc: object = None
    pid: int | None = None
    adopted: bool = False
    t0: float = 0.0
    t1: float = 0.0
    log: Path | None = None
    rss_gb: float = 0.0
    peak_gb: float = 0.0
    restarts: int = 0
    note: str = ""


class Runner:
    """Dependency scheduler for a container with a small CPU quota and a hard memory limit.

    A task starts only when its kind's cap, the memory estimate budget AND the measured working set allow it; above
    BRAKE x limit the youngest non-critical task is killed and re-queued. Every worker runs in its own session through a
    wrapper that appends to its log and writes front/rc/<task>.rc on exit, and the state (pids, start times) is saved
    in front/front_pod.json: a driver started with resume=True ADOPTS the still-running workers of a driver that died
    (2026-10-06: the driver process was SIGKILLed twice mid-run while its workers lived on), reads the rc files of the
    ones that finished meanwhile and re-queues the rest."""
    ADMIT, BRAKE = 0.62, 0.86

    def __init__(self, work: Path, env: dict, mem_budget_gb: float, caps: dict[str, int], log_dir: Path,
                 mem_limit_gb: float | None = None):
        self.work, self.env, self.mem_budget, self.caps, self.log_dir = work, env, mem_budget_gb, caps, log_dir
        self.limit = mem_limit_gb
        self.tasks: dict[str, Task] = {}
        self.status_p = work / "front" / "front_pod.json"
        self.memlog = work / "front" / "memlog.tsv"
        self.rc_dir = work / "front" / "rc"
        self.rc_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.meta: dict = {}
        self.ws_peak = 0.0

    def add(self, t: Task) -> Task:
        self.tasks[t.name] = t
        return t

    def rc_file(self, t: Task) -> Path:
        return self.rc_dir / f"{t.name.replace(':', '_')}.rc"

    def _ready(self, t: Task) -> bool | None:
        """True = deps ok, False = still waiting, None = a dependency failed/skipped."""
        for d in t.deps:
            s = self.tasks[d].state
            if s in ("failed", "skipped"):
                return None
            if s != "ok":
                return False
        return all(self.tasks[d].state in ("ok", "failed", "skipped") for d in t.after if d in self.tasks)

    def _running(self, kind: str | None = None) -> list[Task]:
        return [t for t in self.tasks.values() if t.state == "running" and (kind is None or t.kind == kind)]

    def save(self):
        doc = {**self.meta, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "working_set_peak_gb": round(self.ws_peak, 1),
               "mem_limit_gb": self.limit, "driver_pid": os.getpid(), "tasks": {
                   k: {"state": t.state, "rc": t.rc, "kind": t.kind, "deps": t.deps, "after": t.after,
                       "optional": t.optional, "seconds": round((t.t1 or time.time()) - t.t0, 1) if t.t0 else None,
                       "start": time.strftime("%H:%M:%S", time.localtime(t.t0)) if t.t0 else None, "start_ts": t.t0,
                       "pid": t.pid, "adopted": t.adopted, "note": t.note,
                       "mem_est_gb": t.mem_gb, "rss_peak_gb": round(t.peak_gb, 2), "restarts": t.restarts,
                       "log": str(t.log.relative_to(self.work)) if t.log else None} for k, t in self.tasks.items()}}
        self.status_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_p.with_suffix(".part")
        tmp.write_text(json.dumps(doc, indent=1, default=str))
        os.replace(tmp, self.status_p)

    def resume(self) -> None:
        """Adopt / finish / pre-complete tasks from a previous driver's state and the rc files."""
        prev = {}
        if self.status_p.exists():
            try:
                prev = json.loads(self.status_p.read_text()).get("tasks", {})
            except (OSError, ValueError):
                prev = {}
        for name, t in self.tasks.items():
            p = prev.get(name) or {}
            t.restarts = int(p.get("restarts") or 0)
            t.log = self.log_dir / f"{name.replace(':', '_')}.log"
            rcf = self.rc_file(t)
            if p.get("state") == "running" and _alive(p.get("pid")) and "dancecap" in _cmdline(p["pid"]) \
                    and not rcf.exists():
                t.state, t.pid, t.t0, t.adopted = "running", int(p["pid"]), float(p.get("start_ts") or time.time()), True
                t.mem_gb = max(t.mem_gb, float(p.get("rss_peak_gb") or 0))
                print(f"[front] adopted running {name} (pid {t.pid})", flush=True)
                continue
            if rcf.exists() and p.get("state") in ("running", "ok", "failed"):
                rc = rcf.read_text().strip()
                rc = int(rc) if rc.lstrip("-").isdigit() else rc
                t.t0 = float(p.get("start_ts") or 0.0)
                self._finish(t, rc, quiet=False)
                if t.state == "failed" and t.restarts < t.retries:
                    t.restarts += 1
                    t.state, t.rc = "waiting", None
                    rcf.unlink(missing_ok=True)
                continue
            if p.get("state") == "ok" and t.cmd is None:
                t.state, t.rc = "ok", 0
                continue
            if p.get("state") in ("running", "failed") and t.cmd is not None:
                t.restarts += 1  # its worker died with the driver (no rc file): run it again
                t.note = "re-queued after a driver restart"
            if t.done_if is not None:
                try:
                    done = bool(t.done_if())
                except Exception:  # noqa: BLE001
                    done = False
                if done:
                    t.state, t.rc, t.note = "ok", 0, "outputs from an earlier session"
                    if t.on_ok is not None:
                        t.on_ok()
                    print(f"[front] {name}: done earlier (outputs present)", flush=True)

    def _start(self, t: Task) -> None:
        t.t0 = time.time()
        t.t1 = 0.0
        t.state = "running"
        t.adopted = False
        t.log = self.log_dir / f"{t.name.replace(':', '_')}.log"
        print(f"[front] {time.strftime('%H:%M:%S')} start {t.name}", flush=True)
        _progress.start("front", lane=t.name, unit="tasks")
        if t.cmd is None:
            try:
                t.fn()
                self._finish(t, 0)
            except Exception as e:  # noqa: BLE001 - recorded, dependants are skipped
                t.log.write_text(f"{type(e).__name__}: {e}\n")
                self._finish(t, f"error: {type(e).__name__}: {e}"[:300])
            return
        try:
            argv = t.cmd() if callable(t.cmd) else t.cmd
        except Exception as e:  # noqa: BLE001 - e.g. no camera has the inputs
            t.log.write_text(f"{type(e).__name__}: {e}\n")
            self._finish(t, f"error: {type(e).__name__}: {e}"[:300])
            return
        rcf = self.rc_file(t)
        rcf.unlink(missing_ok=True)
        with open(t.log, "a" if t.restarts else "w", encoding="utf-8") as fh:
            fh.write(f"$ {' '.join(str(c) for c in argv)}   # {time.strftime('%H:%M:%S')}\n")
        if t.nice and shutil.which("nice"):
            argv = ["nice", "-n", str(t.nice), *argv]
        if os.name != "nt" and shutil.which("bash"):  # the worker survives its driver; its rc file tells a later driver
            wrap = ('exec >>"$FRONT_LOG" 2>&1; "$@"; rc=$?; echo $rc > "$FRONT_RC.part" && mv "$FRONT_RC.part" '
                    '"$FRONT_RC"; exit $rc')
            argv = ["bash", "-c", wrap, "frontwrap", *[str(c) for c in argv]]
            out = subprocess.DEVNULL
        else:  # Windows (tests): no wrapper, the driver writes the rc file
            out = open(t.log, "a", encoding="utf-8")
        t.proc = subprocess.Popen([str(c) for c in argv], stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  env=dict(self.env, FRONT_LOG=str(t.log), FRONT_RC=str(rcf),
                                           DANCECAP_PROGRESS_LANE=t.name), cwd=str(PACKAGE_ROOT),
                                  start_new_session=True)
        t.proc._out = out  # type: ignore[attr-defined]
        t.pid = t.proc.pid
        self.save()

    def _finish(self, t: Task, rc, quiet: bool = False) -> None:
        t.t1 = time.time()
        t.rc = rc
        if rc == 0 and t.on_ok is not None:
            try:
                t.on_ok()
            except Exception as e:  # noqa: BLE001
                rc = t.rc = f"on_ok error: {type(e).__name__}: {e}"[:300]
        t.state = "ok" if rc == 0 else "failed"
        if t.state == "ok":
            _progress.end("front", lane=t.name, seconds=round(t.t1 - t.t0, 1), kind=t.kind)
        else:
            _progress.fail("front", rc=str(rc)[:80], lane=t.name)
        print(f"[front] {time.strftime('%H:%M:%S')} {t.name}: {t.state} rc={rc} in {t.t1 - t.t0:.0f}s "
              f"(rss peak {t.peak_gb:.1f} GB){' [adopted]' if t.adopted else ''}", flush=True)
        if t.state == "failed" and t.log and t.log.exists() and not quiet:
            tail = t.log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            print("\n".join(f"[front]   | {x[-300:]}" for x in tail), flush=True)
        self.save()

    def _kill(self, t: Task) -> None:
        import signal
        if t.pid:
            try:
                os.killpg(t.pid, signal.SIGKILL)
            except (OSError, AttributeError):
                try:
                    os.kill(t.pid, signal.SIGKILL)
                except OSError:
                    pass
        if t.proc is not None:
            t.proc.wait()

    def _exit_rc(self, t: Task):
        """None while running; the rc once the worker ended (rc file first, else the process status)."""
        rcf = self.rc_file(t)
        if t.proc is not None:
            prc = t.proc.poll()
            if prc is None:
                return None
            out = getattr(t.proc, "_out", None)
            if out not in (None, subprocess.DEVNULL):
                out.close()
                if not rcf.exists():
                    rcf.write_text(str(prc))
        elif _alive(t.pid):
            return None
        else:
            prc = "lost (worker ended without an rc file)"
        if rcf.exists():
            s = rcf.read_text().strip()
            return int(s) if s.lstrip("-").isdigit() else s
        return prc

    def _poll(self) -> None:
        for t in self._running():
            if t.cmd is None:
                continue
            rc = self._exit_rc(t)
            if rc is None and time.time() - t.t0 > t.timeout_min * 60:
                self._kill(t)
                rc = "timeout"
            if rc is None:
                continue
            self._finish(t, rc)
            if t.state == "failed" and t.restarts < t.retries and rc not in ("timeout",):
                t.restarts += 1
                t.state, t.rc, t.note = "waiting", None, f"retry after rc={rc}"
                print(f"[front] {t.name}: re-queued (retry {t.restarts}/{t.retries})", flush=True)

    def _measure(self) -> float | None:
        ws, lim = mem_state()
        self.limit = lim or self.limit
        row = [time.strftime("%H:%M:%S"), f"{ws:.1f}" if ws is not None else "", f"{lim:.0f}" if lim else ""]
        for t in self._running():
            if t.pid:
                t.rss_gb = tree_rss_gb(t.pid)
                t.peak_gb = max(t.peak_gb, t.rss_gb)
                row.append(f"{t.name}={t.rss_gb:.1f}")
        if ws is not None:
            self.ws_peak = max(self.ws_peak, ws)
        with open(self.memlog, "a") as fh:
            fh.write("\t".join(row) + "\n")
        return ws

    def _brake(self, ws: float | None) -> None:
        if ws is None or not self.limit or ws < self.BRAKE * self.limit:
            return
        cand = [t for t in self._running() if not t.critical and t.pid and t.kind != "cpu"]
        if not cand:
            return
        t = max(cand, key=lambda x: x.t0)  # youngest
        print(f"[front] MEMORY BRAKE: working set {ws:.0f} of {self.limit:.0f} GB -> killing {t.name} "
              f"(rss {t.rss_gb:.1f} GB)", flush=True)
        self._kill(t)
        t.restarts += 1
        if t.restarts <= t.retries + 1:
            t.mem_gb = max(t.mem_gb, t.peak_gb * 1.2)
            t.state, t.proc, t.pid, t.note = "waiting", None, None, "killed by the memory brake, re-queued"
        else:
            self._finish(t, "killed: memory brake")

    def loop(self, poll_s: float = 3.0) -> None:
        last_save = last_mem = 0.0
        ws = None
        while True:
            self._poll()
            if time.time() - last_mem > 10:
                ws = self._measure()
                self._brake(ws)
                last_mem = time.time()
            progressed = False
            for t in self.tasks.values():
                if t.state != "waiting":
                    continue
                r = self._ready(t)
                if r is None:
                    t.state = "skipped"
                    t.rc = "dependency failed: " + ",".join(d for d in t.deps
                                                            if self.tasks[d].state in ("failed", "skipped"))
                    print(f"[front] {t.name}: skipped ({t.rc})", flush=True)
                    progressed = True
                    continue
                if not r:
                    continue
                if len(self._running(t.kind)) >= self.caps.get(t.kind, 99):
                    continue
                run = self._running()
                # estimates of the running tasks (or their measured RSS when larger) + memory reserved for critical
                # tasks that have not started yet
                used = sum(max(x.mem_gb, x.rss_gb) for x in run) + sum(
                    x.mem_gb for x in self.tasks.values() if x.critical and x.state == "waiting" and x is not t)
                if run and used + t.mem_gb > self.mem_budget:
                    continue
                if run and ws is not None and self.limit and ws + t.mem_gb * 0.5 > self.ADMIT * self.limit:
                    continue
                if t.gpu and run:
                    free = gpu_free_mb()
                    if free is not None and free < GPU_FREE_MIN_MB:
                        continue
                self._start(t)
                progressed = True
                if t.gpu or t.mem_gb > 4:
                    time.sleep(3)  # let it allocate before the next admission check
                    ws = self._measure()
            if all(t.state in ("ok", "failed", "skipped") for t in self.tasks.values()):
                break
            if time.time() - last_save > 30:
                self.save()
                last_save = time.time()
            if not progressed:
                time.sleep(poll_s)
        self.save()


def _qa_frame_px(w: int, h: int) -> int:
    """Pixels of one dancers QA frame (dancers_qa.overlay: resized to qa_height 960): round(w * 960 / h) x 960 for a
    landscape video (1707 x 960 for 1280x720); 540 x 960 otherwise (the 9:16 phones)."""
    return round(w * 960 / h) * 960 if w > h else 960 * 540


def _sam_mem_gb(work: Path, cam: str) -> float:
    """Host RAM of one dancers worker: all frames (BGR) + 2 bool masks + SAM probs + QA frames, + 3 GB."""
    m = json.loads((work / "frames" / cam / "frames.json").read_text())
    w, h = int(m["width"]), int(m["height"])
    f, px = len(m["frames"]), w * h
    work_px = min(1.0, 1280 / max(w, h)) ** 2 * px
    return round((f * (3 * px + 2 * px + 2 * work_px + 3 * _qa_frame_px(w, h))) / 2**30 * 1.1 + 3.0, 1)


def _main_cams(work: Path, cams: list[str]) -> list[str]:
    mp = work / "cameras" / "sfm_work" / "map.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        main = set(m.get("main_cams") or [])
        if main:
            return [c for c in cams if c in main]
    import pycolmap

    from .cameras.mapping import model_cams
    return [c for c in cams if c in model_cams(pycolmap.Reconstruction(str(work / "cameras" / "sfm")))]


def _n_frames(work: Path, cam: str) -> int:
    return len(json.loads((work / "frames" / cam / "frames.json").read_text())["frames"])


def _dancers_done(work: Path, cam: str) -> bool:
    """dancers outputs of a camera are complete: its pod_cam json and every full / body mask of both dancers."""
    if not (work / "dancers" / f"pod_cam_{cam}.json").exists() or not (work / "dancers" / f"label_{cam}.json").exists():
        return False
    n = _n_frames(work, cam)
    return all(len(list((work / "dancers" / cam / sub).glob("mask_*.png"))) >= 2 * n for sub in ("masks", "body"))


def _locked_tripods(take_name: str, cams: list[str]) -> set[str]:
    """Cameras whose type (DANCECAP_CAMERA_TYPES policy, default: confirmed answers) is a locked tripod: stage_loc
    localises them on keyframes (+ one constant pose) when it fits, every frame otherwise."""
    from .cameras.typed import locked_tripod
    try:
        types = Take.load(take_name).camera_types(list(cams))
    except Exception as e:  # noqa: BLE001 - the types only change the split, never the result
        print(f"[cameras] camera types not read ({type(e).__name__}: {e}): every camera as handheld + zoom", flush=True)
        return set()
    return {c for c, st in types.items() if locked_tripod(st)}


def build_tasks(R: Runner, take_name: str, work: Path, cams: list[str], main: list[str], subs: list[str],
                skip: tuple[str, ...], loc_split: int = 2) -> str:
    py = sys.executable
    sizes = {c: _sam_mem_gb(work, c) for c in cams}
    by_size = sorted(cams, key=lambda c: -sizes[c])  # long poles first
    long_pole = by_size[0]  # cam02 (2302 x 1080p): its chains set the wall time
    nice = lambda c: 0 if c == long_pole else 5  # noqa: E731
    cache = work / "cameras" / "cache"
    # dancers: label + SAM 2.1 (reused from dancers/cache when its label matches) + identity + body/hair + QA, one
    # process per camera (frames decoded once)
    for c in by_size:
        R.add(Task(f"sam:{c}", [py, "-m", "dancecap.front_pod", "dancer", take_name, c], kind="sam",
                   mem_gb=sizes[c], gpu=True, timeout_min=120, nice=nice(c), critical=c == long_pole,
                   on_ok=lambda c=c: merge_label(work, c), done_if=lambda c=c: _dancers_done(work, c)))
    tripods = _locked_tripods(take_name, main)
    for c in by_size:
        if c not in main or "loc" in skip:
            continue
        n = _n_frames(work, c)
        # a locked tripod is never split: its keyframe localisation is one decision for the whole camera (loc_part)
        parts = loc_split if c == long_pole and loc_split > 1 and c not in tripods else 1
        done = lambda c=c: (cache / f"loc_{c}.npz").exists() and (work / "cameras" / f"{c}.npz").exists()  # noqa: E731
        if parts == 1:
            R.add(Task(f"loc:{c}", [py, "-m", "dancecap", "cameras", take_name, "--stages", "loc", "--cams", c],
                       kind="loc", mem_gb=4.0, gpu=True, timeout_min=120, nice=nice(c), critical=c == long_pole,
                       done_if=done))
            continue
        bounds = [round(k * n / parts) for k in range(parts + 1)]
        names = []
        for a, b in zip(bounds, bounds[1:]):
            nm = f"loc:{c}:{a}"
            names.append(nm)
            R.add(Task(nm, [py, "-m", "dancecap.front_pod", "loc-part", take_name, c, "--range", f"{a}:{b}"],
                       kind="loc", mem_gb=4.0, gpu=True, timeout_min=120, nice=nice(c), critical=True,
                       done_if=lambda c=c, a=a, b=b, done=done: (cache / f"locpart_{c}_{a:05d}_{b:05d}.npz").exists()
                       or done()))
        R.add(Task(f"loc:{c}", [py, "-m", "dancecap.front_pod", "loc-join", take_name, c, "--parts",
                                ",".join(f"{a}:{b}" for a, b in zip(bounds, bounds[1:]))], deps=names, kind="cpu",
                   mem_gb=3.0, timeout_min=20, done_if=done))
    for c in by_size:
        if c in subs and "loc" not in skip:
            R.add(Task(f"sub:{c}", [py, "-m", "dancecap.cameras.reuse", "presolve-sub", take_name, "--cams", c],
                       kind="loc", mem_gb=4.0, gpu=True, timeout_min=120, optional=True, nice=5,
                       done_if=lambda c=c: (cache / f"locsub_{c}.pkl").exists()))
    for c in by_size:
        R.add(Task(f"ma2d:{c}", [py, "-m", "dancecap.front_pod", "ma2d", take_name, c], deps=[f"sam:{c}"],
                   kind="ma2d", mem_gb=8.0, gpu=True, timeout_min=60, nice=nice(c),
                   done_if=lambda c=c: (work / "mamma" / "ma_2d_native" / f"{c}.manifest.json").exists()))
    R.add(Task("labels_review", None, after=[f"sam:{c}" for c in cams], kind="cpu", optional=True, retries=0,
               fn=lambda: labels_review(take_name)))
    # merge whatever cameras finished (dancers_pod.merge reports the missing ones); the cameras stage localised the
    # attached cameras through sub:<cam> caches when those finished, else it redoes that localisation itself
    R.add(Task("merge", [py, "-m", "dancecap.dancers_pod", "merge", take_name, "--cams", ",".join(cams)],
               after=[f"sam:{c}" for c in cams], kind="cpu", mem_gb=8.0, timeout_min=20, retries=0))
    loc_after = [f"loc:{c}" for c in main] + [f"sub:{c}" for c in subs] if "loc" not in skip else []
    R.add(Task("cameras", [py, "-m", "dancecap", "cameras", take_name, "--stages", "attach,world,qa"],
               deps=["merge"], after=loc_after, kind="cpu", mem_gb=16.0, timeout_min=60, retries=0))

    def ready_cams() -> list[str]:  # cameras with both a camera track and ma_2d landmarks
        ok = [c for c in cams if (work / "cameras" / f"{c}.npz").exists()
              and (work / "mamma" / "ma_2d_native" / f"{c}.npz").exists()]
        if len(ok) < 3:
            raise RuntimeError(f"only {ok} have camera tracks + ma_2d landmarks")
        return ok
    R.add(Task("sync", lambda: [py, "-m", "dancecap", "sync", take_name, "--cams", ",".join(ready_cams())],
               deps=["cameras"], after=[f"ma2d:{c}" for c in cams], kind="cpu", mem_gb=12.0, timeout_min=60,
               retries=0))
    R.add(Task("chain", [py, "-m", "dancecap.cameras.world_scale", "chain", take_name, "--out",
                         str(work / "scale" / "chain_after_sync.json")], deps=["sync"], kind="cpu", mem_gb=4.0,
               optional=True, timeout_min=20, retries=0))
    R.add(Task("prepare", lambda: [py, "-m", "dancecap", "mamma", "prepare", take_name, "--skip-2d", "--set",
                                   f"c{len(ready_cams())}={','.join(ready_cams())}"],
               deps=["sync"], kind="cpu", mem_gb=8.0, timeout_min=30, retries=0))
    return long_pole


def run(take_name: str, cams: list[str] | None = None, sam_workers: int | None = None,
        skip: tuple[str, ...] = ("mps",), resume: bool = False) -> int:
    take = Take.load(take_name)
    work = take.work
    cams = cams or [s.id for s in take.sources]
    cpus, ram = cgroup_cpus(), mem_gb()
    ws0, limit = mem_state()
    log_dir = work / "front" / "logs"
    threads = str(max(2, min(4, cpus // 8)))
    env = dict(os.environ, DANCECAP_GPU_SHARED="1", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               OPENBLAS_NUM_THREADS=threads, OPENCV_FOR_THREADS_NUM=threads, PYTHONUNBUFFERED="1",
               DANCECAP_POD=os.environ.get("DANCECAP_POD", "1"))
    torch_home = WORKSPACE / TORCH_HOME_REL
    if torch_home.exists():
        env.setdefault("TORCH_HOME", str(torch_home))
    mps = mps_start(env) if "mps" not in skip else None
    main = _main_cams(work, cams) if "loc" not in skip else list(cams)
    subs = [c for c in cams if c not in main]
    cap_gb = limit if limit else ram
    mem_budget = max(16.0, 0.6 * cap_gb)
    # ~1 busy CPU per SAM / localisation process: keep their sum near the quota (8.5 CPUs on the 2026-10-06 pods)
    n_sam = sam_workers or max(2, min(5, cpus // 3))
    caps = {"sam": n_sam, "loc": max(2, min(5, cpus - n_sam - 2)), "ma2d": 2, "cpu": 2}
    R = Runner(work, env, mem_budget, caps, log_dir, limit)
    long_pole = build_tasks(R, take_name, work, cams, main, subs, skip)
    if resume:
        R.resume()
    else:
        R.memlog.unlink(missing_ok=True)
        for f in R.rc_dir.glob("*.rc"):
            f.unlink()
    R.meta = {"take": take_name, "host": {"cpus": cpus, "ram_gb": round(ram, 1), "mem_limit_gb": limit,
                                          "working_set_at_start_gb": ws0, "mem_budget_gb": round(mem_budget, 1),
                                          "mps": bool(mps), "threads_per_proc": threads, "caps": caps,
                                          "gpu_free_mb_at_start": gpu_free_mb()},
              "cams": cams, "main_cams": main, "attached_cams": subs, "long_pole": long_pole, "resumed": resume,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"[front] {take_name}: {len(R.tasks)} tasks; cpus {cpus}, RAM limit {limit} GB (working set {ws0} GB, "
          f"estimate budget {mem_budget:.0f}), caps {caps}, MPS {'on' if mps else 'off'}, main {main}, attached {subs}, "
          f"long pole {long_pole}{', RESUMED' if resume else ''}", flush=True)
    R.save()
    try:
        R.loop()
    finally:
        if mps:
            subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", env=mps)
    R.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    R.save()
    bad = [k for k, t in R.tasks.items() if t.state != "ok" and not t.optional]
    ok = R.tasks["prepare"].state == "ok"
    print(f"[front] done; not ok: {bad or 'none'}; working set peak {R.ws_peak:.0f} GB -> {R.status_p}", flush=True)
    (work / "front" / "front_done.json").write_text(json.dumps({"rc": 0 if ok else 1, "not_ok": bad,
                                                                 "finished": R.meta["finished"]}))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ supervisor


def _cgroup_report() -> dict:
    """What can kill a process here: cgroup limits, OOM counters, rlimits, this process's cgroup + oom_score_adj."""
    out: dict = {}
    for f in ("/proc/self/cgroup", "/proc/self/oom_score_adj", "/proc/self/oom_score"):
        try:
            out[f] = Path(f).read_text().strip()
        except OSError:
            pass
    for root in (Path("/sys/fs/cgroup/memory"), Path("/sys/fs/cgroup")):
        for name in ("memory.limit_in_bytes", "memory.max_usage_in_bytes", "memory.usage_in_bytes", "memory.failcnt",
                     "memory.oom_control", "memory.max", "memory.current", "memory.events", "memory.peak",
                     "memory.memsw.limit_in_bytes", "memory.kmem.limit_in_bytes", "pids.max", "pids.current"):
            p = root / name
            if p.exists():
                try:
                    out[str(p)] = p.read_text().strip()
                except OSError:
                    pass
    try:
        import resource
        out["rlimits"] = {n: resource.getrlimit(getattr(resource, n)) for n in
                          ("RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS", "RLIMIT_NPROC", "RLIMIT_NOFILE")}
    except (ImportError, AttributeError, ValueError):
        pass
    try:
        out["dmesg_tail"] = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=10).stdout[-3000:]
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def supervise(take_name: str, attempts: int = 4) -> int:
    """The pod job's command: runs the driver (`run --resume`) in its own session and restarts it if it dies before
    front_done.json; echoes its log; keeps ps / cgroup snapshots and writes front/forensic_<n>.json when the driver
    dies (2026-10-06: drivers were SIGKILLed twice with memory at half the limit; cause unknown)."""
    work = PACKAGE_ROOT / "work" / take_name
    front = work / "front"
    front.mkdir(parents=True, exist_ok=True)
    done = front / "front_done.json"
    pidf = front / "driver.pid"
    log = front / "driver.log"
    (front / "host_start.json").write_text(json.dumps(_cgroup_report(), indent=1, default=str))
    pos = log.stat().st_size if log.exists() else 0
    for k in range(attempts):
        if done.exists():
            return int(json.loads(done.read_text())["rc"])
        pid = int(pidf.read_text()) if pidf.exists() and pidf.read_text().strip().isdigit() else None
        proc = None
        if pid and _alive(pid) and "front_pod run" in _cmdline(pid):
            print(f"[supervise] following the running driver (pid {pid})", flush=True)
        else:
            with open(log, "a", encoding="utf-8") as fh:
                proc = subprocess.Popen([sys.executable, "-m", "dancecap.front_pod", "run", take_name, "--resume"],
                                        stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        cwd=str(PACKAGE_ROOT), start_new_session=True)
            pid = proc.pid
            pidf.write_text(str(pid))
            print(f"[supervise] attempt {k + 1}/{attempts}: driver pid {pid}", flush=True)
        snaps: list[str] = []
        last_snap = 0.0
        while (proc.poll() is None) if proc is not None else _alive(pid):
            if log.exists() and log.stat().st_size > pos:
                with open(log, encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                for line in chunk.splitlines():
                    if line.startswith(("[front]", "[supervise]", "Traceback")) or "Error" in line:
                        print(line, flush=True)
            if time.time() - last_snap > 30:
                try:
                    ps = subprocess.run(["ps", "-eo", "pid,ppid,pgid,sid,stat,rss,etimes,args", "--sort=-rss"],
                                        capture_output=True, text=True, timeout=20).stdout
                except (OSError, subprocess.SubprocessError):
                    ps = ""
                snaps = (snaps + [f"{time.strftime('%H:%M:%S')} ws {mem_state()[0]}\n" + "\n".join(ps.splitlines()[:40])])[-3:]
                last_snap = time.time()
            time.sleep(5)
        status = proc.returncode if proc is not None else "unknown (not our child)"
        if done.exists():
            return int(json.loads(done.read_text())["rc"])
        rep = {"attempt": k + 1, "driver_pid": pid, "driver_status": status, "time": time.strftime("%H:%M:%S"),
               "last_ps": snaps, "cgroup": _cgroup_report()}
        (front / f"forensic_{k + 1}.json").write_text(json.dumps(rep, indent=1, default=str))
        print(f"[supervise] driver pid {pid} ended without front_done.json (status {status}); forensic_{k + 1}.json; "
              "restarting with --resume", flush=True)
        time.sleep(10)
    return 1


# ------------------------------------------------------------------------------------------------ pod checks


def fetch_sam2() -> None:
    """SAM 2.1 large from Meta's CDN (the URL of facebookresearch/sam2 checkpoints/download_ckpts.sh), verified against
    the local checkpoint's sha256: saves a 0.9 GB upload over the 2.5 MB/s home link."""
    import urllib.request

    from .dancers_sam import CHECKPOINT
    if CHECKPOINT.exists() and _sha256(CHECKPOINT) == SAM2_SHA256:
        print(f"[front] {CHECKPOINT.name} present (sha256 ok)")
        return
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix(".part")
    t0 = time.time()
    urllib.request.urlretrieve(SAM2_URL, tmp)
    h = _sha256(tmp)
    if h != SAM2_SHA256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"[front] SAM 2.1 checkpoint sha256 {h} != local {SAM2_SHA256}: refusing it")
    os.replace(tmp, CHECKPOINT)
    print(f"[front] {CHECKPOINT.name}: {CHECKPOINT.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f} s, sha256 ok")


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def check_env(take_name: str) -> None:
    """Fail fast (seconds) instead of after an hour: imports of every stage the pod runs + the inputs it reads."""
    import importlib
    errs = []
    for m in ("torch", "cv2", "numpy", "scipy", "pycolmap", "lightglue", "sam2", "smplx", "dancecap.dancers",
              "dancecap.cameras.pipeline", "dancecap.sync", "dancecap.mamma_dyn"):
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            errs.append(f"import {m}: {type(e).__name__}: {e}")
    from . import mamma_dyn as md
    r = subprocess.run([md.python(), "run_ma_2d.py", "--help"], cwd=md.MAMMA_ROOT / "landmarks", env=md._env(),
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        errs.append("run_ma_2d.py --help: " + (r.stderr or r.stdout)[-800:])
    take = Take.load(take_name)
    for p in (md.MA2D_WEIGHTS, md.VERTS_512, take.work / "cameras" / "sfm" / "images.bin"):
        if not Path(p).exists():
            errs.append(f"missing {p}")
    for s in take.sources:
        fj = take.work / "frames" / s.id / "frames.json"
        n = len(json.loads(fj.read_text())["frames"]) if fj.exists() else -1
        nj = len(list((take.work / "frames" / s.id).glob("*.jpg")))
        if n != nj:
            errs.append(f"frames/{s.id}: frames.json {n} vs {nj} jpg")
    import torch
    print(f"[front] torch {torch.__version__} cuda {torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}; cpus {cgroup_cpus()}, "
          f"RAM {mem_gb():.0f} GB; gpu free {gpu_free_mb()} MB", flush=True)
    if errs:
        raise SystemExit("[front] environment check FAILED:\n  " + "\n  ".join(errs))
    print("[front] environment check ok", flush=True)


# ------------------------------------------------------------------------------------------------ job spec


def _resume_inputs(work: Path, w: str, cams: list[str]) -> list[str]:
    """Outputs of an earlier (interrupted) session that let the next one skip work: finished dancers cameras (masks +
    pod_cam/label json + candidates), complete SAM caches of unfinished cameras (the worker reuses them when its label
    matches), localisation results, sub-model presolves and ma_2d landmarks."""
    import zipfile
    out = []
    for c in cams:
        if _dancers_done(work, c):
            out += [f"{w}/dancers/{c}/", f"{w}/dancers/pod_cam_{c}.json", f"{w}/dancers/label_{c}.json"]
            if (work / "dancers" / "candidates" / f"{c}.npz").exists():
                out.append(f"{w}/dancers/candidates/{c}.npz")
        else:
            sc = work / "dancers" / "cache" / f"{c}_sam.npz"
            try:
                ok = sc.exists() and zipfile.ZipFile(sc).testzip() is None
            except (zipfile.BadZipFile, OSError):
                ok = False
            if ok:
                out.append(f"{w}/dancers/cache/{c}_sam.npz")
        cc = work / "cameras" / "cache"
        if (cc / f"loc_{c}.npz").exists() and (work / "cameras" / f"{c}.npz").exists():
            out += [f"{w}/cameras/cache/loc_{c}.npz", f"{w}/cameras/{c}.npz"]
        out += [f"{w}/cameras/cache/{p.name}" for p in sorted(cc.glob(f"locpart_{c}_*.npz")) if ".part" not in p.name]
        if (cc / f"locsub_{c}.pkl").exists():
            out.append(f"{w}/cameras/cache/locsub_{c}.pkl")
        nat = work / "mamma" / "ma_2d_native"
        if all((nat / f"{c}{x}").exists() for x in (".npz", "_boxes.npy", ".manifest.json")):
            out += [f"{w}/mamma/ma_2d_native/{c}{x}" for x in (".npz", "_boxes.npy", ".manifest.json")]
    return out


def write_job(take_name: str, max_minutes: float = 120) -> Path:
    """work/<take>/front/pod_job.json: what the pod session uploads, runs and downloads (dancecap.pod_job/1)."""
    take = Take.load(take_name)
    work = take.work
    w = f"dancecap/work/{take_name}"
    ref = (take.cfg.get("cameras") or {}).get("scale_reference") or {}
    rel = lambda p: Path(p).resolve().relative_to(WORKSPACE).as_posix()  # noqa: E731
    align = rel(PACKAGE_ROOT / "takes" / take.cfg["alignment"])
    inputs = [f"{w}/frames/", f"{w}/keyframes/", f"{w}/people/", f"{w}/people_masks/", f"{w}/people_masks_kf/",
              f"{w}/cameras/sfm/", f"{w}/cameras/sfm_work/", f"{w}/cameras/cache/feats/", f"{w}/cameras/report.json",
              align, {"local": MA2D_STRIPPED, "pod": "mamma/data/weights/ma_2d/mamma_mask_full_cvpr.ckpt"},
              "mamma/data/body_models/downsampled_verts/verts_512.pkl",
              "mamma/data/body_models/downsampled_verts/smplx_512_body_parts.npz"]
    hub = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    for f in ("aliked-n16.pth", "aliked_lightglue_v0-1_arxiv.pth"):
        if (hub / f).exists():
            inputs.append({"local": str(hub / f), "pod": f"{TORCH_HOME_REL}/hub/checkpoints/{f}"})
    if ref:
        rt = Take.load(ref["take"])
        rw = f"dancecap/work/{ref['take']}"
        inputs += [ref["sim3"] if ref["sim3"].startswith("dancecap/") else f"dancecap/{ref['sim3']}",
                   rel(PACKAGE_ROOT / "takes" / rt.cfg["alignment"]),
                   *[f"{rw}/cameras/sfm/{b}.bin" for b in ("cameras", "images", "points3D")],
                   *[f"{rw}/cameras/{p.name}" for p in sorted((rt.work / "cameras").glob("[0-9][0-9].npz"))],
                   *[f"{rw}/frames/{p.parent.name}/frames.json"
                     for p in sorted((rt.work / "frames").glob("[0-9][0-9]/frames.json"))]]
    cams = [s.id for s in take.sources]
    resumed = _resume_inputs(work, w, cams)
    inputs += resumed
    inputs = list(dict.fromkeys(json.dumps(x) for x in inputs))
    inputs = [json.loads(x) for x in inputs]
    cache = [f"{w}/cameras/cache/{n}" for c in cams for n in (f"loc_{c}.npz", f"attach_{c}.npz", f"locsub_{c}.pkl",
                                                              f"vertical_sub_{c}.pkl")]
    outputs = [f"{w}/dancers/", f"{w}/front/", *[f"{w}/cameras/{c}.npz" for c in cams],
               f"{w}/cameras/world_from_sfm.json", f"{w}/cameras/world_from_sfm.prev.json", f"{w}/cameras/report.json",
               f"{w}/cameras/qa/", *cache, f"{w}/cameras/cache/vertical_main.pkl",
               f"{w}/cameras/cache/horizontal_main.pkl", f"{w}/cameras/cache/dancers_sfm.npz",
               f"{w}/mamma/ma_2d_native/", f"{w}/mamma/ma_2d/", f"{w}/mamma/ma_cap/", f"{w}/mamma/prepare.json",
               f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/pod_job.json", f"{w}/mamma/logs/", f"{w}/sync/",
               f"{w}/scale/chain_after_sync.json", f"{w}/review/"]
    job = {
        "schema": "dancecap.pod_job/1", "name": "front", "take": take_name, "max_minutes": max_minutes,
        "needs": ["dancers", "cameras", "ma2d"],
        "inputs": inputs,
        "exclude": [f"{w}/people/qa", "*.mp4", "*.db", "*.db-shm", "*.db-wal"],
        "env": {"TORCH_HOME": f"/workspace/HeadMovement/{TORCH_HOME_REL}", "PYOPENGL_PLATFORM": "egl"},
        # the supervisor is restarted (optional commands) if the runner's command itself is killed: each call follows
        # the running driver or resumes it (adopts live workers, reads their rc files)
        "commands": [
            {"name": "sam2_ckpt", "cmd": "python -m dancecap.front_pod fetch-sam2", "timeout_min": 10},
            {"name": "check", "cmd": f"python -m dancecap.front_pod check {take_name}", "timeout_min": 12},
            *[{"name": f"front{'' if k == 0 else f'_{k + 1}'}", "optional": True, "timeout_min": max_minutes - 15,
               "cmd": f"python -m dancecap.front_pod supervise {take_name}"} for k in range(3)],
            {"name": "status", "cmd": f"python -m dancecap.front_pod status {take_name}", "timeout_min": 2},
        ],
        "outputs": outputs,
        "verify": [f"{w}/dancers/roles.json", f"{w}/dancers/labels.json", f"{w}/cameras/world_from_sfm.json",
                   f"{w}/mamma/ma_2d_native/", f"{w}/sync/offsets.json", f"{w}/mamma/prepare.json",
                   f"{w}/front/front_pod.json"],
        "after_download": [f"python -m dancecap.clock_studio window --take {take_name}",
                           f"python -m dancecap review {take_name} --instants 8",
                           "local quality gates (frames/PTS, detection coverage, re-ID swaps, localisation, gravity, "
                           "scale, sync) on the downloaded outputs"],
        "notes": ["front stage 2026-10-06 (Runpod first): labels, SAM 2.1 masks, per-frame localisation, ma_2d, "
                  "attach/world/qa, sync and mamma prepare of ~10.4k frames in one pod session "
                  "(dancecap.front_pod); never on the 4-CPU workstation",
                  "SAM 2.1 large is fetched on the pod from Meta's CDN and checked against the local sha256; the "
                  "ma_2d checkpoint is uploaded as a state_dict-only copy (identical tensors)",
                  "not downloaded: cameras/cache/feats (~2 MB per frame), mamma/ma2d_masks (links)",
                  f"resumed from earlier sessions (uploaded, tasks pre-completed): {len(resumed)} inputs"],
    }
    p = work / "front" / "pod_job.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(job, indent=1))
    print(f"[front] -> {p}")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.front_pod", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "supervise", "label", "dancer", "ma2d", "loc-part", "loc-join",
                                       "labels-review", "write-job", "fetch-sam2", "check", "status"])
    ap.add_argument("take", nargs="?")
    ap.add_argument("cam", nargs="?")
    ap.add_argument("--cams")
    ap.add_argument("--sam-workers", type=int)
    ap.add_argument("--skip", default="mps", help="comma list of loc, mps (default mps: CUDA MPS off)")
    ap.add_argument("--max-minutes", type=float, default=120)
    ap.add_argument("--resume", action="store_true", help="run: adopt / continue a previous driver's tasks")
    ap.add_argument("--range", help="loc-part: a:b window frames")
    ap.add_argument("--parts", help="loc-join: a:b,b:c,...")
    a = ap.parse_args(argv)
    if a.action == "run":
        raise SystemExit(run(a.take, a.cams.split(",") if a.cams else None, a.sam_workers,
                             tuple(x for x in a.skip.split(",") if x), resume=a.resume))
    if a.action == "supervise":
        raise SystemExit(supervise(a.take))
    if a.action == "status":
        w = PACKAGE_ROOT / "work" / a.take
        d = json.loads((w / "front" / "front_pod.json").read_text()) if (w / "front" / "front_pod.json").exists() else {}
        for k, v in (d.get("tasks") or {}).items():
            print(f"{k:<18} {v['state']:<8} rc={v['rc']} {v['seconds']} s {v.get('note', '')}")
        raise SystemExit(0 if (w / "mamma" / "prepare.json").exists() else 1)
    if a.action == "label":
        label_one(a.take, a.cam)
    elif a.action == "dancer":
        dancer_one(a.take, a.cam)
    elif a.action == "ma2d":
        ma2d_one(a.take, a.cam)
    elif a.action == "loc-part":
        lo, hi = (int(x) for x in a.range.split(":"))
        loc_part(a.take, a.cam, lo, hi)
    elif a.action == "loc-join":
        loc_join(a.take, a.cam, [tuple(int(x) for x in r.split(":")) for r in a.parts.split(",")])
    elif a.action == "labels-review":
        labels_review(a.take)
    elif a.action == "write-job":
        write_job(a.take, a.max_minutes)
    elif a.action == "fetch-sam2":
        fetch_sam2()
    elif a.action == "check":
        check_env(a.take)


if __name__ == "__main__":
    main()
