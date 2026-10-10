"""Chunks: the gap planner, the S1 segments as chunks, job specs and the pod-side step helpers (FARM.md section 2).

A CHUNK is a fit window on the fused grid: `{id, k0, k1, core_k0, core_k1, overlap_k, kind, cams, world_id, ...}`.
`core` = the frames the chunk fills in the final result, `k0..k1` = the window actually fitted (the core plus an
overlap of 4 s into every neighbour, done or planned, so the stitcher can blend; no overlap beyond the timeline's ends).
The planner turns the not-complete cells of a coverage into gaps (frame runs) and splits each gap into
equal chunks whose cores are 24-36 s long (the length nearest to 30 s): 33.3 s -> one 33.3 s chunk, 116.6 s -> four
29.2 s chunks. It skips frames that already belong to a chunk with a job (queued .. stitched).

JOB SPECS (`farm/jobs/<chunk>.json` locally, `/workspace/hm/queue/<chunk>.json` on the volume) follow the worker's
protocol (farm/worker.py):

    {"schema": "dancecap.farm.job/1", "id", "dance", "kind", "stages", "take", "chunk": {...}, "world_id", "world_sha256",
     "pose_id", "code": {"git_sha", "dirty", "bundle_sha256"}, "cams", "sources": {cam: {w, h, fps, t_first, t_last, path, sha256}},
     "inputs": [{"path", "sha256", "bytes", "local", "archive"?}], "steps": [{"name", "cmd", "timeout_min", "optional", "env",
     "save", "live", "stage"}], "outputs": [{"path", "optional"?}], "env": {...}, "cwd": "dancecap", "max_minutes",
     "priority", "min_vram_gb", "estimate": {...}, "created"}

* `inputs[].path` is the workspace-relative DESTINATION (the worker's rule): `src/<capture>/<cam>.mp4` (shared on the
  volume, content-addressed; the take TOML a pod generates points `source_dir` at HM_ROOT/src/<capture>) or a per-chunk
  archive `dancecap/work/<take>/mamma/prepared.tar` (`archive: true`, deterministic tar: sorted, mtime 0) holding the prepared
  MAMMA inputs at workspace paths. `local` is the file on the workstation for the uploader (`farm queue`).
* `outputs[].path` are workspace-relative (`dancecap/work/<take>/smplx/`, `.../mamma/fits/`, `.../<refined>/`); the worker packs
  them into out/result.tar + manifest.json, the puller extracts them at the same paths locally, which is exactly where
  timeline.collect_* looks (the chunk's take is a work dir like the S1 segments').
* Steps run in `dancecap/` of the pod's workspace with the volume's venv on PATH (plain `python`); env values may use {WS}.
  The world is copied by the worker to dancecap/work/worlds/<dance>/<world_id> ($HM_WORLD_DIR). A step with a `save` list
  leaves a checkpoint on the volume so a restarted worker (any pod) resumes after it. STEP_TEMPLATES below is the one place
  that composes them (worker.py's fit_steps / refine_steps / front_steps are the same idea for single takes).

Pod-side helpers the steps call (`python -m dancecap.farm chunk-take | chunk-prep | chunk-fit ID`): the chunk's take TOML +
front plan, the map reuse from the WORLD package, the MAMMA fit with the world's betas pinned and frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import DEFAULT_DANCE, VOLUME_ROOT, farm_dir, jobs_dir, work_dir
from . import world as W
from .state import atomic_write_json, now_iso, read_json
from .timeline import HZ, Coverage, DanceSpec, camera_spans, runs
from ..take import PACKAGE_ROOT, WORKSPACE

JOB_SCHEMA = "dancecap.farm.job/1"
CORE_TARGET_S, CORE_MIN_S, CORE_MAX_S = 30.0, 24.0, 36.0
OVERLAP_S = 4.0
MIN_WINDOW_S = 12.0        # a tiny gap still gets a window this long (the overlaps grow into the neighbours)
FRONT_MARGIN_S = 2.0       # the front-end processes this much beyond the fit window (a boundary refit never needs a new front)
MIN_CAM_S = 2.0            # a phone joins a chunk if it records at least this long inside the window
KINDS = ("front", "prepare", "fit", "refine", "stitch", "analyse", "stitch_analyse", "full")   # CPU kinds: analysis.CPU_KINDS
# rough rates for ETA / LPT priority (measured on one A100: PLAN.md section 4 / front/front_pod.json); never used to gate anything
FRONT_S_PER_CAMFRAME = 0.39
REFINE_S_PER_S = 1.6


def fit_seconds(T: int) -> float:
    """MAMMA full fit on one A100 with 8 cameras: 3047 s for 1148 frames (model 0.76 x 675 x (0.4 s + 4.8 ms x T))."""
    return 0.76 * 675 * (0.4 + 0.0048 * T)


# --------------------------------------------------------------------------------------------------- the model


@dataclass
class Chunk:
    id: str
    k0: int                       # fit window (inclusive grid indices)
    k1: int
    core_k0: int                  # the frames this chunk fills
    core_k1: int
    overlap_k: int = int(OVERLAP_S * HZ)
    kind: str = "full"            # the job to run: front | fit | refine | analyse | full (= every stage not yet done)
    cams: list = field(default_factory=list)
    world_id: str | None = None
    origin: str = "planned"       # planned (by the gap planner) | s1 (an existing segment a-d, front already done)
    take: str = ""                # work/<take>: the chunk's own take / work dir
    stages_done: list = field(default_factory=list)
    status: str = "planned"
    priority: float = 0.0
    estimate: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.k1 - self.k0 + 1

    @property
    def core_T(self) -> int:
        return self.core_k1 - self.core_k0 + 1

    @property
    def ov_left(self) -> int:
        return self.core_k0 - self.k0

    @property
    def ov_right(self) -> int:
        return self.k1 - self.core_k1

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(T=self.T, core_s=round(self.core_T / HZ, 2))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


def chunk_id(spec: DanceSpec, core_k0: int) -> str:
    """g<frames from the start of the timeline to the core's first frame>: stable for the same gap and split."""
    return f"g{core_k0 - spec.grid.k0:04d}"


# --------------------------------------------------------------------------------------------------- planning


def split_gap(n_frames: int, hz: float = HZ, target_s: float = CORE_TARGET_S, lo_s: float = CORE_MIN_S,
              hi_s: float = CORE_MAX_S) -> list[int]:
    """Core lengths (frames) that cover a gap of n_frames with equal chunks of 24-36 s, nearest to 30 s (a gap
    up to 36 s is one chunk, however short). Ties go to more, shorter chunks (they run in parallel)."""
    if n_frames <= 0:
        return []
    L = n_frames / hz
    if L <= hi_s + 1e-9:
        return [n_frames]
    lo_n, hi_n = math.ceil(L / hi_s - 1e-9), math.floor(L / lo_s + 1e-9)
    cands = list(range(lo_n, hi_n + 1)) or sorted({max(1, math.floor(L / target_s)), math.ceil(L / target_s)})
    n = min(cands, key=lambda c: (abs(L / c - target_s), -c))
    edges = [int(round(j * n_frames / n)) for j in range(n + 1)]
    return [edges[j + 1] - edges[j] for j in range(n)]


# --------------------------------------------------------------------- equal windows (class recap, 2026-10-09: 60 s chunks)
#
# The demo's planner above cuts a gap into cores of 24-36 s and grows every core by OVERLAP_S into each neighbour, so a "30 s chunk" is a
# 38 s fit window and two neighbours share 8 s. A take can ask for the other convention in its TOML (`[chunks] chunk_s, overlap_s`):
# chunks are FIT WINDOWS of at most chunk_s seconds that overlap their neighbour by overlap_s seconds (the S1 segments a-d are of this
# kind: 4 s overlap regions, cores cut at the middle of each overlap). equal_windows() does the arithmetic, plan_windows() makes the chunks.


def chunk_params(spec: "DanceSpec") -> dict:
    """The take's `[chunks]` table with defaults: {chunk_s: max fit window (s), overlap_s: overlap region of two neighbours (s),
    mode: "windows" (this convention) | "cores" (the demo's), profile: step-template profile of the take ("" = the demo's)}."""
    c = dict((spec.cfg or {}).get("chunks") or {})
    return {"chunk_s": float(c.get("chunk_s", CORE_TARGET_S)), "overlap_s": float(c.get("overlap_s", OVERLAP_S)),
            "mode": str(c.get("mode", "windows" if "chunk_s" in c else "cores")), "profile": str(c.get("profile", ""))}


def equal_windows(n_frames: int, hz: float = HZ, max_window_s: float = 60.0, overlap_s: float = 4.0) -> dict:
    """Cover n_frames with the fewest EQUAL fit windows of at most max_window_s whose neighbours overlap by overlap_s.

    n windows of length L overlapping by o cover n L - (n - 1) o frames, so L = (N + (n - 1) o) / n; the smallest n with L <= max_window_s
    wins (a fixed 60 s window stepping by 56 s would leave a tiny last chunk). Returns {n, window_k (int frames per window, +-1),
    overlap_k, windows: [[k0, k1]] (0-based frame indices), cores: [[a, b]] (cut at the middle of each overlap), window_s, core_s,
    step_s, tried: [{n, window_s}]}."""
    o = int(round(overlap_s * hz))
    cap = max_window_s * hz
    if n_frames <= cap:
        return {"n": 1, "window_k": n_frames, "overlap_k": o, "windows": [[0, n_frames - 1]], "cores": [[0, n_frames - 1]],
                "window_s": [round(n_frames / hz, 3)], "core_s": [round(n_frames / hz, 3)], "step_s": None, "tried": []}
    tried, n = [], 1
    while True:
        n += 1
        L = (n_frames + (n - 1) * o) / n
        tried.append({"n": n, "window_s": round(L / hz, 3)})
        if L <= cap + 1e-9 or n > 200:
            break
    step = (n_frames - L) / (n - 1)
    wins = [[int(round(j * step)), int(round(j * step + L)) - 1] for j in range(n)]
    wins[-1][1] = n_frames - 1
    cuts = [(wins[j + 1][0] + wins[j][1] + 1) // 2 for j in range(n - 1)]
    cores = [[0 if j == 0 else cuts[j - 1], n_frames - 1 if j == n - 1 else cuts[j] - 1] for j in range(n)]
    return {"n": n, "window_k": int(round(L)), "overlap_k": o, "windows": wins, "cores": cores,
            "window_s": [round((b - a + 1) / hz, 3) for a, b in wins], "core_s": [round((b - a + 1) / hz, 3) for a, b in cores],
            "step_s": round(step / hz, 3), "tried": tried}


# --------------------------------------------------------------------------------------------------- reslice (refits)
#
# A seam refit (the stitcher's `refit` recommendation) or any gap inside / next to spans whose front end is done does NOT need the front end
# again: every array of a prepared MAMMA set (ma_2d/<seq>/cam*.npz, ma_cap/<seq>/gt/*.npz) has the frame axis first, so a window is an exact
# slice, and a window that straddles two takes (a segment's tail + the next take's head, the stitcher's "contains_window: false") is the
# concatenation of two slices on the same 30 Hz lattice, each taken where its own take is furthest from its edge. The front end (video
# upload, detection, SAM, localisation, ma_2d: about 800 MB of source video over the home uplink) is only for frames NO take has prepared.
RESLICE_MIN_CAMS = 3
RESLICE_MARK = "reslice.json"


class ResliceError(RuntimeError):
    pass


def prepared_sources(spec: DanceSpec, exclude=()) -> list[dict]:
    """Prepared MAMMA sets on this machine as [{take, k0, k1, tag, seq, cams, dir, prep}]: the reference take, the S1 segments and the
    farm's chunk takes whose mamma/prepare.json grid lies on the dance lattice and whose ma_2d / ma_cap inputs are still on disk."""
    g = spec.grid
    takes = [spec.reference] if spec.reference else []
    takes += [spec.segment_take(s) for s in spec.segments]
    base = work_dir(spec.name).parent
    for d in sorted(base.glob(f"{spec.name}_*")):
        if d.is_dir() and d.name not in takes:
            takes.append(d.name)
    out = []
    for t in takes:
        if t in exclude:
            continue
        w = work_dir(t)
        prep = read_json(w / "mamma" / "prepare.json", None)
        if not isinstance(prep, dict) or not prep.get("grid") or not prep.get("sets"):
            continue
        t0 = float(prep["grid"]["t0"])
        if abs((t0 - g.anchor) * g.hz - round((t0 - g.anchor) * g.hz)) > 1e-3:
            continue   # not on this dance's lattice
        k0 = g.k_of(t0)
        k1 = k0 + int(prep["grid"]["T"]) - 1
        tag = max(prep["sets"], key=lambda x: len(prep["sets"][x]["cams"]))
        seq = prep["sets"][tag]["seq"]
        if not (w / "mamma" / "ma_2d" / seq).is_dir() or not (w / "mamma" / "ma_cap" / seq / "gt").is_dir():
            continue
        out.append({"take": t, "k0": k0, "k1": k1, "tag": tag, "seq": seq, "cams": sorted(prep["sets"][tag]["cams"]), "dir": w, "prep": prep})
    return out


def plan_reslice(spec: DanceSpec, k0: int, k1: int, exclude=(), sources: list[dict] | None = None) -> dict | None:
    """Can the window k0..k1 be cut out of prepared sets that exist? -> {"parts": [{take, a, b}], "cams": [...]} (contiguous runs, a..b
    inclusive grid indices, each frame taken from the source furthest from its own grid edges: edge frames have the fewest cameras),
    or None when some frame has no prepared source or fewer than RESLICE_MIN_CAMS cameras are common to the sources involved."""
    srcs = [s for s in (sources if sources is not None else prepared_sources(spec, exclude)) if s["k0"] <= k1 and s["k1"] >= k0]
    if not srcs:
        return None
    owner = []
    for k in range(k0, k1 + 1):
        cands = [s for s in srcs if s["k0"] <= k <= s["k1"]]
        if not cands:
            return None
        owner.append(max(cands, key=lambda s: (min(k - s["k0"], s["k1"] - k), len(s["cams"]), s["take"]))["take"])
    parts, a = [], k0
    for i in range(1, len(owner) + 1):
        if i == len(owner) or owner[i] != owner[i - 1]:
            parts.append({"take": owner[i - 1], "a": a, "b": k0 + i - 1})
            a = k0 + i
    involved = {p["take"] for p in parts}
    cams = sorted(set.intersection(*(set(s["cams"]) for s in srcs if s["take"] in involved)))
    if len(cams) < RESLICE_MIN_CAMS:
        return None
    return {"parts": parts, "cams": cams, "window": [k0, k1]}


def _cut(path: Path, a: int, b: int, T: int) -> tuple[dict, set]:
    """(arrays with the frame axis cut to a..b, names of the frame-axis arrays) of one npz of a source take with T frames."""
    z = np.load(path, allow_pickle=False)
    arrs = {k: z[k] for k in z.files}
    frame = {k for k, v in arrs.items() if v.ndim >= 1 and v.shape[0] == T}
    return {k: (v[a:b] if k in frame else v) for k, v in arrs.items()}, frame


def _merge_npz(files: list[tuple[Path, int, int, int]], dst: Path, n: int, scalars: dict | None = None) -> None:
    """Concatenate the frame-axis arrays of several npz (path, a, b, T of the source) and keep the other arrays, which must agree."""
    cuts = [_cut(p, a, b, T) for p, a, b, T in files]
    keys = cuts[0][0].keys()
    out = {}
    for k in keys:
        vals = [c[0][k] for c in cuts]
        if all(k in c[1] for c in cuts):
            out[k] = np.concatenate(vals, axis=0) if len(vals) > 1 else vals[0]
            if out[k].shape[0] != n:
                raise ResliceError(f"{dst.name}:{k}: {out[k].shape[0]} frames, expected {n}")
        else:
            if k not in (scalars or {}) and any(not np.array_equal(vals[0], v) for v in vals[1:]):
                raise ResliceError(f"{dst.name}:{k} differs between the takes the window is cut from")
            out[k] = vals[0]
    for k, v in (scalars or {}).items():
        if k in out:
            out[k] = np.asarray(v) if isinstance(v, str) else np.asarray(v, dtype=out[k].dtype)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)


def _reslice_complete(dst: Path, plan: dict) -> bool:
    """Every file a reslice writes is there (a half-deleted or half-written directory is rebuilt, not reused)."""
    prep = read_json(dst / "prepare.json", None)
    if not isinstance(prep, dict) or not prep.get("sets"):
        return False
    tag = next(iter(prep["sets"]))
    seq = prep["sets"][tag]["seq"]
    want = [dst / prep["config"], dst / "ma_cap" / seq / "gt" / "global.npz"]
    for cam in plan["cams"]:
        want += [dst / "ma_2d" / seq / f"cam{cam}.npz", dst / "ma_cap" / seq / "gt" / f"cam{cam}.npz"]
    return all(p.is_file() for p in want)


def reslice_take(spec: DanceSpec, chunk: "Chunk", plan: dict, *, force: bool = False) -> dict:
    """Write work/<chunk.take>/mamma/{prepare.json, ma3d_config.yaml, ma_2d/<seq>, ma_cap/<seq>} = the chunk's window cut out of the prepared
    sets of existing takes (plan_reslice). The result is what `fit` reads, exactly as a front end would have prepared it. Refuses to
    overwrite a take dir that holds real outputs (a fit) or that this function did not write."""
    g = spec.grid
    k0, k1 = chunk.k0, chunk.k1
    n = k1 - k0 + 1
    srcs = {s["take"]: s for s in prepared_sources(spec, exclude=(chunk.take,))}
    parts = plan["parts"]
    if any(p["take"] not in srcs for p in parts):
        raise ResliceError(f"{chunk.id}: a source take of the window is no longer prepared on disk")
    dst = work_dir(chunk.take) / "mamma"
    if dst.exists():
        mark = read_json(dst / RESLICE_MARK, None)
        if not mark and not force:
            raise ResliceError(f"{dst} exists and was not written by reslice_take: not touched")
        if mark and mark.get("plan") == plan and not force and _reslice_complete(dst, plan):
            return {"take": chunk.take, "reused": True, "frames": n, "parts": parts}
        if (dst / "fits").exists() or (dst / "select.json").exists():
            raise ResliceError(f"{dst} already holds a fit: not overwritten")
        shutil.rmtree(dst)
    cams = list(plan["cams"])
    tag = f"c{len(cams)}"
    seq = f"{chunk.take}_{tag}"
    first = srcs[parts[0]["take"]]
    cfgs = {hashlib.sha256((srcs[p["take"]]["dir"] / "mamma" / srcs[p["take"]]["prep"]["config"]).read_bytes()).hexdigest() for p in parts}
    if len(cfgs) != 1:
        raise ResliceError(f"{chunk.id}: the takes the window is cut from used different ma_3d configs")
    ts, firsts, lasts = [], {c: [] for c in cams}, {c: [] for c in cams}
    for p in parts:
        s = srcs[p["take"]]
        a, b = p["a"] - s["k0"], p["b"] - s["k0"] + 1
        ts += list(s["prep"]["grid"]["t"][a:b])
        cpf = (s["prep"]["grid"].get("trim") or {}).get("cams_per_frame") or {}
        for c in cams:
            lo, hi = cpf.get(c, [s["prep"]["grid"]["t"][0], s["prep"]["grid"]["t"][-1]])
            firsts[c].append(float(lo))
            lasts[c].append(float(hi))
    if len(ts) != n:
        raise ResliceError(f"{chunk.id}: window of {n} frames but {len(ts)} grid times")
    tmp = dst.with_name("mamma.reslice.part")
    if tmp.exists():
        shutil.rmtree(tmp)
    for c in cams:
        _merge_npz([(srcs[p["take"]]["dir"] / "mamma" / "ma_2d" / srcs[p["take"]]["seq"] / f"cam{c}.npz", p["a"] - srcs[p["take"]]["k0"],
                     p["b"] - srcs[p["take"]]["k0"] + 1, srcs[p["take"]]["k1"] - srcs[p["take"]]["k0"] + 1) for p in parts],
                   tmp / "ma_2d" / seq / f"cam{c}.npz", n)
        _merge_npz([(srcs[p["take"]]["dir"] / "mamma" / "ma_cap" / srcs[p["take"]]["seq"] / "gt" / f"cam{c}.npz",
                     p["a"] - srcs[p["take"]]["k0"], p["b"] - srcs[p["take"]]["k0"] + 1,
                     srcs[p["take"]]["k1"] - srcs[p["take"]]["k0"] + 1) for p in parts],
                   tmp / "ma_cap" / seq / "gt" / f"cam{c}.npz", n, scalars={"frame_start": 0, "frame_end": n})
    _merge_npz([(srcs[p["take"]]["dir"] / "mamma" / "ma_cap" / srcs[p["take"]]["seq"] / "gt" / "global.npz", p["a"] - srcs[p["take"]]["k0"],
                 p["b"] - srcs[p["take"]]["k0"] + 1, srcs[p["take"]]["k1"] - srcs[p["take"]]["k0"] + 1) for p in parts],
               tmp / "ma_cap" / seq / "gt" / "global.npz", n, scalars={"frames_len": n, "frame_start": 0, "frame_end": n, "seq_name": seq})
    base = first["prep"]
    trim = {"coverage": "union", "grid_anchor": float(g.anchor), "grid_k": [int(k0), int(k1)],
            "cams_per_frame": {c: [min(firsts[c]), max(lasts[c])] for c in cams}, "window": [float(ts[0]), float(ts[-1])]}
    prep = {"take": chunk.take, "created": now_iso(), "world_frame": base.get("world_frame"), "camera_types": base.get("camera_types"),
            "grid": {"t0": float(ts[0]), "T": n, "hz": base["grid"].get("hz", g.hz), "t": ts, "trim": trim},
            "sync": {**(base.get("sync") or {}), "note": "per-take offsets differ by part; see reslice.parts"},
            "ma_2d": {}, "config": base["config"], "config_overrides": base.get("config_overrides"), "base_config": base.get("base_config"),
            "sets": {tag: {"seq": seq, "cams": cams, "inputs": base["sets"][first["tag"]].get("inputs")}},
            "eval": {"seq": seq, "cams": cams},
            "reslice": {"parts": [{"take": p["take"], "k": [p["a"], p["b"]]} for p in parts], "window": [k0, k1],
                        "note": "cut out of prepared sets that exist (farm.chunks.reslice_take): no front end"}}
    (tmp).mkdir(parents=True, exist_ok=True)
    (tmp / "prepare.json").write_text(json.dumps(prep, indent=1), encoding="utf-8")
    shutil.copy2(first["dir"] / "mamma" / base["config"], tmp / base["config"])
    (tmp / RESLICE_MARK).write_text(json.dumps({"plan": plan, "written": now_iso(), "chunk": chunk.id}, indent=1), encoding="utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, dst)
    return {"take": chunk.take, "reused": False, "frames": n, "parts": parts, "cams": cams, "seq": seq}


def window_for(spec: DanceSpec, core_k0: int, core_k1: int, overlap_k: int, min_window_k: int) -> tuple[int, int]:
    """The fit window of a core: overlap_k into each neighbour (none past the timeline's ends), grown symmetrically to
    min_window_k when the core is tiny."""
    K0, K1 = spec.grid.k0, spec.grid.k1
    k0 = max(K0, core_k0 - (overlap_k if core_k0 > K0 else 0))
    k1 = min(K1, core_k1 + (overlap_k if core_k1 < K1 else 0))
    L = k1 - k0 + 1
    if L < min_window_k:
        extra = min_window_k - L
        a, b = extra // 2, extra - extra // 2
        k0n, k1n = k0 - a, k1 + b
        if k0n < K0:
            k1n, k0n = k1n + (K0 - k0n), K0
        if k1n > K1:
            k0n, k1n = k0n - (k1n - K1), K1
        k0, k1 = max(K0, k0n), min(K1, k1n)
    return k0, k1


def chunk_cams(spec: DanceSpec, k0: int, k1: int, spans: dict | None = None) -> list[str]:
    """Phones that record at least MIN_CAM_S inside the window."""
    spans = spans if spans is not None else camera_spans(spec)
    a, b = float(spec.grid.t(k0)), float(spec.grid.t(k1 + 1))
    return sorted(c for c, (lo, hi) in spans.items() if min(b, hi) - max(a, lo) >= MIN_CAM_S)


def estimate(chunk: Chunk, spec: DanceSpec, spans: dict | None = None) -> dict:
    """Minutes per stage on one A100 (rough: measured rates, see the constants); drives the LPT priority and max_minutes."""
    spans = spans if spans is not None else camera_spans(spec)
    a, b = float(spec.grid.t(chunk.k0)), float(spec.grid.t(chunk.k1 + 1))
    cam_s = sum(max(0.0, min(b, spans[c][1]) - max(a, spans[c][0])) for c in chunk.cams if c in spans)
    front = FRONT_S_PER_CAMFRAME * cam_s * HZ / 60.0
    fit = fit_seconds(chunk.T) / 60.0
    refine = (REFINE_S_PER_S * chunk.T / HZ + 25.0) / 60.0
    est = {"front_min": round(front, 1), "fit_min": round(fit, 1), "refine_min": round(refine, 1)}
    todo = {"front": front, "fit": fit, "refined": refine}
    est["total_min"] = round(sum(v for k, v in todo.items() if k not in chunk.stages_done), 1)
    est["note"] = "one A100, rates from the larissa_kadu_38 / S1 runs; a guide, not a gate"
    return est


def plan_gaps(cov: Coverage, spec: DanceSpec, *, claimed: list[tuple[int, int, str]] | None = None, mode: str = "missing",
              overlap_s: float = OVERLAP_S, world_id: str | None = None, kind: str = "full", spans: dict | None = None,
              min_gap_s: float = 0.0, core_target_s: float | None = None) -> tuple[list[Chunk], dict]:
    """Coverage -> chunks. Frames in `claimed` ((core_k0, core_k1, id) of chunks that already have a job) are left alone.
    core_target_s: the core length the gaps are cut towards (default CORE_TARGET_S = 30 s, cores of 0.8-1.2 x that: 24-36 s).
    Returns (chunks, report) where the report lists the gaps found, what claimed frames were skipped and the numbers."""
    g = spec.grid
    ct = float(core_target_s) if core_target_s else CORE_TARGET_S
    mask = cov.gap_mask(mode).copy()
    raw = runs(mask, g.k0)
    for a, b, _cid in claimed or []:
        lo, hi = max(a, g.k0) - g.k0, min(b, g.k1) - g.k0
        if hi >= lo:
            mask[lo:hi + 1] = False
    gaps = [r for r in runs(mask, g.k0) if (r[1] - r[0] + 1) / g.hz >= min_gap_s]
    ov = int(round(overlap_s * g.hz))
    spans = spans if spans is not None else camera_spans(spec)
    out: list[Chunk] = []
    for a, b in gaps:
        pos = a
        for n in split_gap(b - a + 1, g.hz, ct, ct * CORE_MIN_S / CORE_TARGET_S, ct * CORE_MAX_S / CORE_TARGET_S):
            c0, c1 = pos, pos + n - 1
            pos = c1 + 1
            k0, k1 = window_for(spec, c0, c1, ov, int(MIN_WINDOW_S * g.hz))
            ch = Chunk(id=chunk_id(spec, c0), k0=k0, k1=k1, core_k0=c0, core_k1=c1, overlap_k=ov, kind=kind,
                       cams=chunk_cams(spec, k0, k1, spans), world_id=world_id, origin="planned",
                       take=f"{spec.name}_{chunk_id(spec, c0)}")
            ch.estimate = estimate(ch, spec, spans)
            ch.priority = ch.estimate["total_min"]
            out.append(ch)
    rep = {"mode": mode, "gaps_before_claims": [list(r) for r in raw], "gaps": [list(r) for r in gaps],
           "gap_s": [round((b - a + 1) / g.hz, 2) for a, b in gaps],
           "claimed_skipped": [c for c in (claimed or []) if any(max(c[0], a) <= min(c[1], b) for a, b in raw)],
           "chunks": len(out), "core_s": [round(c.core_T / g.hz, 2) for c in out], "overlap_s": overlap_s}
    return out, rep


def plan_windows(spec: DanceSpec, *, max_window_s: float | None = None, overlap_s: float | None = None, world_id: str | None = None,
                 kind: str = "full", spans: dict | None = None, claimed_ids: set | None = None, cams_fn=None) -> tuple[list[Chunk], dict]:
    """The WHOLE timeline of a dance as equal fit windows (equal_windows; parameters from the take's `[chunks]`): chunks of at most
    max_window_s overlapping by overlap_s, cores cut at the middle of each overlap. A chunk whose id is in `claimed_ids` is not planned
    again (ids are g<frames from the timeline start to the core's first frame>, stable for the same parameters). cams_fn(k0, k1) -> the
    phones of a window (default: every phone that records >= MIN_CAM_S inside it; the recap planner passes its coverage-map rule).
    Returns (chunks, report); the report has the keys cmd_plan prints (gaps, gap_s, claimed_skipped, ...) plus the arithmetic."""
    g = spec.grid
    p = chunk_params(spec)
    eq = equal_windows(g.T, g.hz, max_window_s or p["chunk_s"], overlap_s if overlap_s is not None else p["overlap_s"])
    spans = spans if spans is not None else camera_spans(spec)
    out: list[Chunk] = []
    skipped = []
    for (a, b), (ca, cb) in zip(eq["windows"], eq["cores"]):
        k0, k1, c0, c1 = g.k0 + a, g.k0 + b, g.k0 + ca, g.k0 + cb
        cid = chunk_id(spec, c0)
        if claimed_ids and cid in claimed_ids:
            skipped.append((c0, c1, cid))
            continue
        ch = Chunk(id=cid, k0=k0, k1=k1, core_k0=c0, core_k1=c1, overlap_k=eq["overlap_k"], kind=kind,
                   cams=cams_fn(k0, k1) if cams_fn else chunk_cams(spec, k0, k1, spans), world_id=world_id, origin="planned",
                   take=f"{spec.name}_{cid}")
        ch.estimate = estimate(ch, spec, spans)
        ch.priority = ch.estimate["total_min"]
        out.append(ch)
    rep = {"mode": "windows", "gaps_before_claims": [[g.k0, g.k1]], "gaps": [[g.k0, g.k1]], "gap_s": [round(g.T / g.hz, 2)],
           "claimed_skipped": skipped, "chunks": len(out), "core_s": [round(c.core_T / g.hz, 2) for c in out],
           "overlap_s": eq["overlap_k"] / g.hz, "window_s": [round(c.T / g.hz, 2) for c in out], "arithmetic": eq}
    return out, rep


# ----------------------------------------------------------------------------------------------- S1 segments


def register_s1(spec: DanceSpec, world_id: str | None = None) -> list[Chunk]:
    """The existing segment takes a-d (front-end done, prepared inputs on disk) as chunks, as they are: the window = their
    prepared grid, the cores split at the middle of each overlap, the reference span (larissa_kadu_38) untouched."""
    g = spec.grid
    segs = []
    for sid in spec.segments:
        w = spec.segment_work(sid)
        prep = read_json(w / "mamma" / "prepare.json", None)
        if not prep:
            continue
        k0 = g.k_of(float(prep["grid"]["t0"]))
        k1 = k0 + int(prep["grid"]["T"]) - 1
        cams = max((s["cams"] for s in prep["sets"].values()), key=len)
        done = ["front"]
        fits = [f for f in (w / "mamma" / "fits").glob("*/fit.json") if not (read_json(f, {}) or {}).get("partial")]
        if fits or (w / "smplx" / "lead.npz").exists():
            done.append("fit")
        if (w / spec.refined_dir / "lead.npz").exists():
            done.append("refined")
        segs.append({"id": sid, "k0": k0, "k1": k1, "cams": sorted(cams), "done": done, "take": spec.segment_take(sid)})
    segs.sort(key=lambda s: s["k0"])
    for s in segs:
        s["core"] = [s["k0"], s["k1"]]
    r0, r1 = spec.ref_k if spec.ref_k else (None, None)
    for s in segs:   # the reference span is complete: nobody owns a frame of it
        if r0 is not None and s["k0"] < r0 <= s["k1"]:
            s["core"][1] = min(s["core"][1], r0 - 1)
        if r1 is not None and s["k0"] <= r1 < s["k1"]:
            s["core"][0] = max(s["core"][0], r1 + 1)
    for a, b in zip(segs, segs[1:]):   # neighbours: cut at the middle of their overlap
        if b["k0"] <= a["k1"]:
            cut = (b["k0"] + a["k1"] + 1) // 2
            a["core"][1], b["core"][0] = min(a["core"][1], cut - 1), max(b["core"][0], cut)
    spans = None
    try:
        spans = camera_spans(spec)
    except Exception:  # noqa: BLE001
        pass
    out = []
    for s in segs:
        ch = Chunk(id=s["id"], k0=s["k0"], k1=s["k1"], core_k0=s["core"][0], core_k1=s["core"][1],
                   overlap_k=int(OVERLAP_S * g.hz), kind="full", cams=s["cams"], world_id=world_id, origin="s1",
                   take=s["take"], stages_done=s["done"])
        if spans:
            ch.estimate = estimate(ch, spec, spans)
            ch.priority = ch.estimate["total_min"]
        out.append(ch)
    return out


# ----------------------------------------------------------------------------------------------- job specs


class _Fmt(dict):
    def __missing__(self, key):   # leave any other {placeholder} as it is
        return "{" + key + "}"


def fmt(template: str, **ctx) -> str:
    return template.format_map(_Fmt(**ctx))


def _step(name: str, cmd: str, timeout_min, optional: bool = False, env: dict | None = None, save: list | None = None,
          live: list | None = None) -> dict:
    d = {"name": name, "cmd": cmd, "timeout_min": timeout_min, "optional": optional}
    if env:
        d["env"] = env
    if save:
        d["save"] = save
    if live:
        d["live"] = live
    return d


W_ = "dancecap/work/{take}"   # a chunk's work dir, workspace-relative

# THE STEP TEMPLATES (one place). Per stage the commands a worker runs, in order. {braces} are filled in by build_job()
# from the chunk and the world; `save` = workspace paths the worker checkpoints on the volume after the step succeeds.
STEP_TEMPLATES: dict[str, list[dict]] = {
    "front": [
        _step("chunk_take", "python -m dancecap.farm chunk-take {chunk}", 3),
        _step("chunk_prep", "python -m dancecap.farm chunk-prep {chunk}", 40),
        _step("front", "python -m dancecap.front_full run --segs {chunk}", "{front_min}",
              save=[f"{W_}/mamma/prepare.json", f"{W_}/mamma/ma3d_config.yaml", f"{W_}/mamma/ma_2d/", f"{W_}/mamma/ma_cap/",
                    f"{W_}/sync/offsets.json", f"{W_}/cameras/world_from_sfm.json", f"{W_}/dancers/roles.json"]),
    ],
    "fit": [
        _step("fit", "python -m dancecap.farm chunk-fit {chunk}", "{fit_min}",
              env={"MAMMA_FREEZE_BETAS": "1", "MAMMA_INIT_BETAS": "{WS}/dancecap/work/worlds/{dance}/{world_id}/shared/betas.json"},
              live=[f"{W_}/mamma/ma_3d/"],
              save=[f"{W_}/smplx/", f"{W_}/mamma/fits/", f"{W_}/mamma/select.json", f"{W_}/mamma/report.json"]),
    ],
    "refined": [
        _step("refine", "python -m dancecap.pose_refine {take} --src smplx --dst {refined_dir} {refine_args}", "{refine_min}",
              save=[f"{W_}/{{refined_dir}}/"]),
        # per-chunk analysis on the pod that already holds the chunk's inputs (FARM.md 0.10); optional, small outputs
        _step("quality", "python -m dancecap.pose_quality {take} --smplx-dir {refined_dir} --tag rf2_noslip", 20, optional=True),
    ],
    "analyse": [
        _step("timing", "python -m dancecap timing {take} --no-refine", 40, save=[f"{W_}/timing/"]),
        _step("physics", "python -m dancecap physics {take}", 20, save=[f"{W_}/physics/"]),
    ],
}
STAGE_OUTPUTS: dict[str, list[tuple[str, bool]]] = {   # (path under the chunk's work dir, optional)
    "front": [("mamma/prepare.json", False), ("sync/offsets.json", True), ("dancers/roles.json", True),
              ("cameras/report.json", True), ("mamma/logs/", True)],
    "fit": [("smplx/", False), ("mamma/fits/", False), ("mamma/select.json", False), ("mamma/report.json", True),
            ("mamma/logs/", True)],
    "refined": [("{refined_dir}/", False), ("quality/", True)],
    "analyse": [("timing/", False), ("physics/", False)],
}


def refine_args(recipe: dict | None) -> str:
    """pose_refine's command line from the world's refine recipe (smplx_rf2_noslip = these values)."""
    r = recipe or {}
    f, x = r.get("filter") or {}, r.get("refine") or {}
    a = [f"--mode {r.get('mode', 'refine')}", f"--root-hz {f.get('root_hz', 5.0)}", f"--rot-hz {f.get('rot_hz', 8.0)}",
         f"--w-floor {x.get('w_floor', 1.0)}", f"--floor-mm {x.get('floor_mm', 5.0)}", f"--w-feet {x.get('w_feet', 1.0)}",
         f"--w-slip {x.get('w_slip', 0.0)}", f"--slip-max-v {x.get('slip_max_v', 0.15)}",
         f"--w-anchor {x.get('w_anchor', 1.0)}", f"--anchor-mm {x.get('anchor_mm', 10.0)}",
         f"--iters {x.get('iters', 400)}", "--device cuda"]
    if x.get("root_xz"):
        a.append("--root-xz")
    return " ".join(a)


def stages_for(chunk: Chunk, kind: str) -> list[str]:
    """Stages a job of `kind` has to run for this chunk (what is already done is skipped)."""
    want = {"front": ["front"], "fit": ["fit"], "refine": ["refined"], "analyse": ["analyse"],
            "full": ["front", "fit", "refined"]}[kind]
    return [s for s in want if s not in chunk.stages_done]


def code_info(cwd: Path = WORKSPACE) -> dict:
    """{git_sha, dirty, branch}: the code the job was written for (bundle_sha256 is filled by whoever bundles it)."""
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"git_sha": git("rev-parse", "HEAD") or None, "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain")), "bundle_sha256": None}


def _ws(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:   # F1 2026-10-09: a source folder outside the workspace (the recap's junction to the owner's videos): absolute, launch.prepare_spec accepts it
        return Path(p).resolve().as_posix()


def pack_tar(dest: Path, items: list[tuple[Path, str]]) -> None:
    """Deterministic tar (sorted, mtime 0, uid/gid 0): the same files always give the same sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with tarfile.open(tmp, "w", format=tarfile.GNU_FORMAT) as tar:
        for src, arc in sorted(items, key=lambda x: x[1]):
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = src.stat().st_size, 0, 0o644, 0, 0, "", ""
            with open(src, "rb") as fh:
                tar.addfile(ti, fh)
    os.replace(tmp, dest)


def archive_input(spec: DanceSpec, chunk: Chunk, name: str, items: list[tuple[Path, str]], dest_rel: str) -> dict:
    """An archive input holding `items` ((local file, workspace-relative arcname)): built once per content (a sidecar
    records the file-tree hash it was built from), listed with the tar's sha256 for the worker to verify."""
    tree = W.tree_sha([(arc, *W.cache().sha(src)) for src, arc in items])
    tar = farm_dir(spec.name) / "in" / chunk.id / f"{name}.tar"
    side = tar.with_name(tar.name + ".json")
    meta = read_json(side, None)
    if not (meta and meta.get("tree_sha256") == tree and tar.exists() and tar.stat().st_size == meta.get("bytes")):
        pack_tar(tar, items)
        sha, n = W.cache().sha(tar, use_cache=False)
        meta = {"tree_sha256": tree, "sha256": sha, "bytes": n, "files": len(items)}
        atomic_write_json(side, meta)
    return {"path": dest_rel, "archive": True, "sha256": meta["sha256"], "bytes": meta["bytes"], "local": _ws(tar),
            "tree_sha256": tree, "n_files": meta["files"]}


def prepared_input(spec: DanceSpec, chunk: Chunk) -> dict:
    """The prepared MAMMA inputs of a chunk (what `fit` reads): prepare.json, the ma_3d config, ma_2d/<seq>, ma_cap/<seq>."""
    root = work_dir(chunk.take) / "mamma"
    prep = json.loads((root / "prepare.json").read_text(encoding="utf-8"))
    files = [root / "prepare.json", root / prep["config"]]
    seqs = [s["seq"] for s in prep["sets"].values()] + ([prep["eval"]["seq"]] if prep.get("eval") else [])
    for seq in dict.fromkeys(seqs):
        for sub in ("ma_2d", "ma_cap"):
            d = root / sub / seq
            files += [p for p in sorted(d.rglob("*")) if p.is_file()] if d.exists() else []
    items = [(p, _ws(p)) for p in files]
    return archive_input(spec, chunk, "prepared", items, f"dancecap/work/{chunk.take}/mamma/prepared.tar")


def motion_input(spec: DanceSpec, chunk: Chunk) -> dict | None:
    """The chunk's pulled fit (work/<take>/smplx + mamma/prepare.json + select.json): the input of a stand-alone refine."""
    wk = work_dir(chunk.take)
    files = [p for sub in ("smplx",) for p in sorted((wk / sub).rglob("*")) if p.is_file()]
    files += [p for p in (wk / "mamma" / "prepare.json", wk / "mamma" / "select.json") if p.exists()]
    if not files:
        return None
    return archive_input(spec, chunk, "motion", [(p, _ws(p)) for p in files], f"dancecap/work/{chunk.take}/motion.tar")


def source_inputs(spec: DanceSpec, cams: list[str]) -> tuple[list[dict], dict]:
    """The source videos as shared inputs (src/<capture>/<file> on the volume; hashed once and cached) and their facts."""
    from ..take import Take
    take = Take.load(spec.name)
    plan = read_json(spec.work / "front" / "plan.json", None) or {}
    facts = dict(plan.get("sources") or {})
    if not facts:   # no plan.json yet: ffprobe
        from ..front_full import source_facts
        facts = source_facts(take)
    capture = Path(spec.cfg["source_dir"]).name
    inputs, src = [], {}
    for s in take.sources:
        if s.id not in cams:
            continue
        sha, n = W.cache().sha(s.path)
        rel = f"src/{capture}/{s.path.name}"
        inputs.append({"path": rel, "sha256": sha, "bytes": n, "local": _ws(s.path)})
        src[s.id] = {**{k: facts[s.id][k] for k in ("w", "h", "fps", "t_first", "t_last")}, "path": rel, "sha256": sha, "bytes": n}
    return inputs, src


def build_job(chunk: Chunk, spec: DanceSpec, world: "W.World | None" = None, kind: str | None = None, *,
              hash_inputs: bool = True, code: dict | None = None, max_minutes: float | None = None) -> dict:
    """The job spec for a chunk (see the module docstring). kind: the job to run (default chunk.kind); 'full' runs every
    stage the chunk has not done yet, so an S1 segment (front done) gets fit + refine only."""
    kind = kind or chunk.kind
    if kind not in KINDS:
        raise ValueError(f"unknown job kind {kind!r} (have {KINDS})")
    if kind in ("prepare", "analyse", "stitch", "stitch_analyse"):   # pod jobs of analysis.py (FARM.md 0.10)
        from . import analysis as A
        if kind == "prepare":
            return A.build_prepare_job(chunk, spec, world, code=code)
        if kind == "analyse":   # the per-chunk analyses (scope chunk) on the chunk's own result, after its fit job
            names = [n for n, a in A.ANALYSES.items() if a.scope == "chunk"]
            return A.build_analysis_job(spec, chunk.take, names, world, hash_inputs=hash_inputs, code=code, after=[chunk.id])
        raise ValueError(f"a {kind!r} job covers a list of chunks: analysis.build_final_job(spec, chunks, world)")
    if chunk_params(spec)["profile"] == "recap":   # take-specific step templates (farm/recap_profile.py)
        from . import recap_profile as RPf
        return RPf.build_job(chunk, spec, world, kind, hash_inputs=hash_inputs, code=code, max_minutes=max_minutes, shape=chunk.id == RPf.SHAPE_ID)
    stages = stages_for(chunk, kind)
    est = chunk.estimate or estimate(chunk, spec)
    recipe = world.load_json("shared/refine_recipe.json") if world else None
    ctx = {"chunk": chunk.id, "take": chunk.take, "dance": spec.name, "world_id": world.id if world else chunk.world_id,
           "refined_dir": spec.refined_dir, "refine_args": refine_args(recipe),
           "front_min": max(30, math.ceil(2.0 * est["front_min"] + 20)), "fit_min": max(30, math.ceil(1.6 * est["fit_min"] + 15)),
           "refine_min": max(10, math.ceil(2 * est["refine_min"] + 5))}
    steps = []
    for st in stages:
        for t in STEP_TEMPLATES[st]:
            s = {k: v for k, v in t.items()}
            s["cmd"] = fmt(t["cmd"], **ctx)
            tm = fmt(t["timeout_min"], **ctx) if isinstance(t["timeout_min"], str) else t["timeout_min"]
            s["timeout_min"] = round(float(tm), 1)
            if t.get("env"):
                s["env"] = {k: fmt(v, **ctx) for k, v in t["env"].items()}
            for key in ("save", "live"):
                if t.get(key):
                    s[key] = [fmt(x, **{**ctx, "take": chunk.take}).replace("{{refined_dir}}", spec.refined_dir) for x in t[key]]
            s["stage"] = st
            steps.append(s)
    if steps:   # every job ends with the small per-frame evidence file coverage is computed from (analysis.chunk_evidence)
        steps.append({"name": "evidence", "cmd": fmt("python -m dancecap.farm chunk-evidence {chunk} --stage {last}", last=stages[-1], **ctx),
                      "timeout_min": 10.0, "optional": False, "stage": stages[-1]})
    outputs, seen = [], set()
    for st in stages:
        for rel, opt in STAGE_OUTPUTS[st]:
            path = f"dancecap/work/{chunk.take}/" + fmt(rel, **ctx)
            if path not in seen:
                seen.add(path)
                outputs.append({"path": path, **({"optional": True} if opt else {})})
    if steps:
        outputs.append({"path": f"dancecap/work/{chunk.take}/farm/evidence.npz"})
    inputs, sources = [], {}
    if hash_inputs:
        if "front" in stages:
            inputs, sources = source_inputs(spec, chunk.cams)
        elif "fit" in stages and (work_dir(chunk.take) / "mamma" / "prepare.json").exists():
            inputs = [prepared_input(spec, chunk)]
        elif stages == ["refined"] or stages == ["analyse"]:
            mi = motion_input(spec, chunk)
            inputs = [mi] if mi else []
    total = est.get("total_min") or 60.0
    return {"schema": JOB_SCHEMA, "id": chunk.id, "dance": spec.name, "kind": kind, "stages": stages, "take": chunk.take,
            "chunk": chunk.to_dict(), "cams": chunk.cams,
            "world_id": world.id if world else chunk.world_id, "world_sha256": world.sha256 if world else None,
            "pose_id": world.pose_id if world else None,
            "code": code if code is not None else code_info(),
            "sources": sources, "inputs": inputs, "steps": steps, "outputs": outputs,
            "env": {"DANCECAP_FARM_DANCE": spec.name}, "cwd": "dancecap",
            "max_minutes": round(max_minutes or max(45.0, 1.5 * total + 20.0), 1), "priority": chunk.priority or total,
            "min_vram_gb": 24 if "front" in stages else 12, "needs_gpu": stages != ["analyse"],
            "estimate": est, "created": now_iso()}


def write_job(job: dict, dance: str | None = None) -> Path:
    p = jobs_dir(dance or job["dance"]) / f"{job['id']}.json"
    atomic_write_json(p, job)
    return p


def load_job(chunk_id: str, dance: str = DEFAULT_DANCE, path: str | Path | None = None) -> dict:
    """The job spec of a chunk: an explicit path, $FARM_JOB, the local farm/jobs/, or the volume's queue/ (on a pod)."""
    cands = [Path(path)] if path else []
    if os.environ.get("FARM_JOB"):
        cands.append(Path(os.environ["FARM_JOB"]))
    cands += [jobs_dir(dance) / f"{chunk_id}.json", Path(os.environ.get("HM_ROOT", VOLUME_ROOT)) / "queue" / f"{chunk_id}.json"]
    for p in cands:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no job spec for chunk {chunk_id}: tried {[str(c) for c in cands]}")


# ------------------------------------------------------------------------------- chunk takes (pod-side helpers)


def chunk_take_toml(job: dict, spec: DanceSpec) -> str:
    """takes/<dance>_<id>.toml for a planned chunk, in the format front_full.write_segments writes for the S1 segments:
    window = the fit window + FRONT_MARGIN_S, every camera in the chunk, max_fps 30, loc kf_by_time, sync required,
    [segment] fit_k / grid_anchor, the dancers / camera types of the dance, and the [world] pin. On a pod ($HM_ROOT, the
    world copied to $HM_WORLD_DIR) the source videos are read from the volume's src/ and the clocks from the world."""
    from ..front_full import MAX_FPS, REF, _table, _tv
    c = job["chunk"]
    g = spec.grid
    cfg = spec.cfg
    t0 = max(float(g.t_start), float(g.t(c["k0"])) - FRONT_MARGIN_S)
    t1 = min(float(g.t_end), float(g.t(c["k1"] + 1)) + FRONT_MARGIN_S)
    cams = list(job["cams"])
    all_cams = spec.cams or sorted({x for s in spec.segments.values() for x in s.get("cams", [])})
    excl = sorted(set(cfg.get("exclude", [])) | (set(all_cams) - set(cams)))
    source_dir, alignment = cfg["source_dir"], cfg["alignment"]
    if os.environ.get("HM_ROOT"):
        source_dir = f"{os.environ['HM_ROOT']}/src/{Path(cfg['source_dir']).name}"
    if os.environ.get("HM_WORLD_DIR") and (Path(os.environ["HM_WORLD_DIR"]) / "shared" / "alignment_studio.json").exists():
        alignment = (Path(os.environ["HM_WORLD_DIR"]) / "shared" / "alignment_studio.json").as_posix()
    lines = [
        f"# GENERATED by `python -m dancecap.farm chunk-take {c['id']}` ({time.strftime('%Y-%m-%d %H:%M')}) from the job spec; do not edit.",
        f"# Chunk {c['id']} of {spec.name}: core grid k {c['core_k0']}..{c['core_k1']} ({c['core_s']} s), fit window k {c['k0']}..{c['k1']},",
        f"# front window = fit window +- {FRONT_MARGIN_S:g} s. Room map, world frame, scale, betas and avatars are the world package's.",
        f"name = {_tv(c['take'])}",
        f"source_dir = {_tv(source_dir)}",
        f"alignment = {_tv(alignment)}",
        f"exclude = {_tv(excl)}",
        f"window = {_tv([round(t0, 4), round(t1, 4)])}",
        "keyframe_span = [30.0, 68.0]   # the reused map's keyframes",
        "keyframe_step = 4.0", "",
        "[segment]", f"parent = {_tv(spec.name)}", f"id = {_tv(c['id'])}", f"fit_k = {_tv([int(c['k0']), int(c['k1'])])}",
        f"grid_anchor = {g.anchor!r}", f"cams = {_tv(cams)}", f"reuse = {_tv(spec.reference or REF)}", "",
        "[ingest]", f"max_fps = {MAX_FPS}", "", "[sync]", "required = true", "", "[cameras]", "loc = { kf_by_time = true }", "",
    ]
    if job.get("world_id"):
        lines += ["[world]", f"dance = {_tv(spec.name)}", f"id = {_tv(job['world_id'])}", f"pose_id = {_tv(job.get('pose_id'))}",
                  'role = "chunk"', ""]
    lines += _table("dance", cfg["dance"]) + [""]
    for role in ("lead", "follow"):
        d = cfg["dancers"][role]
        lines += _table(f"dancers.{role}", d) + [""]
        if d.get("shape_edits"):
            lines += _table(f"dancers.{role}.shape_edits", d["shape_edits"]) + [""]
    for cam in cams:
        ct = (cfg.get("cameras") or {}).get(cam)
        if isinstance(ct, dict):
            lines += [f'[cameras."{cam}"]'] + [f"{k} = {_tv(v)}" for k, v in ct.items()] + [""]
    return "\n".join(lines)


def chunk_take(chunk_id: str, dance: str = DEFAULT_DANCE, job_path: str | None = None, out_dir: Path | None = None) -> dict:
    """Write takes/<take>.toml and merge the chunk into work/<dance>/front/plan.json (what front_full.run reads)."""
    import tomllib
    job = load_job(chunk_id, dance, job_path)
    spec = DanceSpec.load(dance)
    text = chunk_take_toml(job, spec)
    tomllib.loads(text)   # parses
    p = (out_dir or PACKAGE_ROOT / "takes") / f"{job['take']}.toml"
    p.write_text(text, encoding="utf-8")
    plan_p = spec.work / "front" / "plan.json"
    plan = read_json(plan_p, None) or {"created": now_iso(), "sources": {}, "segments": {}}
    for cam, s in (job.get("sources") or {}).items():
        plan["sources"].setdefault(cam, {k: s[k] for k in ("w", "h", "fps", "t_first", "t_last")})
    t0, t1 = (float(x) for x in tomllib.loads(text)["window"])
    segcams, tot = {}, 0
    for cam in job["cams"]:
        f = plan["sources"].get(cam)
        if not f:
            continue
        lo, hi = max(t0, f["t_first"]), min(t1, f["t_last"])
        every = max(1, int(round(f["fps"] / 30.0))) if f["fps"] > 45 else 1
        n = max(0, int((hi - lo) * f["fps"] / every))
        segcams[cam] = {"frames": n, "w": f["w"], "h": f["h"], "every": every, "span": [round(lo, 3), round(hi, 3)]}
        tot += n
    plan["segments"][chunk_id] = {"take": job["take"], "window": [t0, t1], "cams": segcams, "frames": tot}
    plan["frames_total"] = sum(v["frames"] for v in plan["segments"].values())
    atomic_write_json(plan_p, plan)
    return {"take": p.name, "window": [t0, t1], "frames": tot}


def _world_root(world_root) -> Path | None:
    r = world_root or os.environ.get("HM_WORLD_DIR")
    return Path(r) if r else None


def chunk_prep(chunk_id: str, dance: str = DEFAULT_DANCE, world_root: str | Path | None = None,
               job_path: str | None = None) -> dict:
    """Map reuse + world copy for the chunk take from the WORLD package (front_full.prep_segment against world/ref)."""
    from .. import front_full
    job = load_job(chunk_id, dance, job_path)
    w = W.require(dance, job["world_id"], job.get("world_sha256"), root=_world_root(world_root), level="quick")
    info = front_full.prep_segment(chunk_id, base=w.ref())
    gauge = stage_gauge_reference(DanceSpec.load(dance), w)
    return {"take": job["take"], "world": w.id, "images": len(info.get("images", [])), "gauge_ref": gauge}


def stage_gauge_reference(spec: DanceSpec, w: "W.World") -> str:
    """front_full's sync task pins its gauge to the reference take (`sync --gauge-ref larissa_kadu_38`), which reads
    work/<reference>/sync/offsets.json - a file that exists on the PC but not on a pod. The world package carries it as ref/sync/offsets.json
    (member clock_gauge): put it where the sync step looks. Never overwrites a DIFFERENT file (on the PC that is the real one)."""
    from .. import front_full
    src = w.ref("sync/offsets.json")
    if not src.exists():
        return "world has no ref/sync/offsets.json"
    dst = work_dir(spec.reference or front_full.REF) / "sync" / "offsets.json"
    if dst.exists():
        return "present" if dst.read_bytes() == src.read_bytes() else "kept (a different file is there)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return "staged from the world package"


def chunk_fit(chunk_id: str, dance: str = DEFAULT_DANCE, world_root: str | Path | None = None,
              job_path: str | None = None) -> dict:
    """MAMMA full fit of the chunk's prepared camera set with the world's betas pinned and frozen, then select
    (promotes mamma/fits/<tag> to smplx/)."""
    job = load_job(chunk_id, dance, job_path)
    w = W.require(dance, job["world_id"], job.get("world_sha256"), root=_world_root(world_root), level="quick")
    os.environ["MAMMA_INIT_BETAS"] = str(w.betas_file())
    os.environ["MAMMA_FREEZE_BETAS"] = "1"
    from .. import mamma_dyn as md
    prep = json.loads((md.take_work(job["take"]) / "mamma" / "prepare.json").read_text(encoding="utf-8"))
    tag = max(prep["sets"], key=lambda t: len(prep["sets"][t]["cams"]))
    md.fit(job["take"], tag)
    md.select(job["take"], qa=False)
    return {"take": job["take"], "tag": tag, "world": w.id}
