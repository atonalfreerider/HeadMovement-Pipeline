# Farm: chunked, multi-pod, volume-backed 3D-pose pipeline (contract v1, 2026-10-08)

Written by the orchestrator from the user's request of 2026-10-08. Builders implement against this file; change it
only additively and append every change to the log at the bottom. Nothing here contains likeness data: world
manifests hold hashes and paths only.

## 0. What the user asked for (condensed, binding)

1. 3D poses for the whole timeline as fast as possible; the pipeline is REDONE around it.
2. Gaps are filled as **30-second chunks**; chunks are **stitched** together (plus any already-analysed span).
3. The **same room model** (solved once) is used by every analysis; the same **avatars** and **skeleton
   configurations**: all shared consistently across sessions and pods ("world package", section 3).
4. **Several pods** solve different parts of the timeline independently; results are merged.
5. Results are **downloaded the moment a chunk is done** (verified), not at the end of a session.
6. **Persistent storage on Runpod** (a network volume) so a session does not start from scratch: pods start from the
   volume (environment, weights, code, world, inputs already there) and run in minutes.
7. The Pipeline Monitor is cleared of old jobs and redesigned around a **defragmentation-style map** of the final
   result: per timeline cell fully completed / partially completed / not completed at all.
8. Both repositories are fully git-managed with proper `main` branches.
9. Standing: no hands/face pods; non-commercial; secrets only read at runtime; spend cap in `remote.py`
   (DEFAULT_CAP, ledger `dancecap/work/runpod/ledger.json`); never leave finished pods running; the run must survive
   a dropped connection (detached local watcher, resumable pod jobs, pods deleted only after verified pulls).

10. **(2026-10-08, user: "let's give everything to runpod that pertains to analysis")** ALL analysis compute runs on
    Runpod, never on the 4-CPU workstation (it sits at 100 % CPU and every local analysis starves the rest). That
    includes, besides front/fit/refine: `prepare` (mamma prepare), `stitch`, `pose_quality` (reprojection per
    camera/10-s block, zoom checks), timing, physics, counterbalance, floorcraft, analysis_review, coverage
    computation on big outputs, camera-gate evidence (video_zoom / video_motion / attach solves) and the narration
    ASR. They are farm jobs of kind `analyse` (CPU pods, about \$0.035/vCPU-h, or a GPU pod when a job needs one;
    `needs_gpu` per job) reading their inputs from the volume (chunk outputs already live there: no transfer) and
    writing small results that the watcher pulls. The PC keeps only: orchestration, Unity (viewer, film), review of
    images/video, git, and packaging (`demo_export` copy/validate). A local fallback exists only when the user asks.
    Heavy new analysis code must therefore be runnable headless on a pod from the volume's environment.

## 1. Timeline and cells

- Dance timeline = the reference clock (cam06) in seconds. The canonical fused grid of a dance is 30 Hz anchored on
  `t_k = anchor + k/30` (larissa_kadu_full: anchor 30.151176844227862, k = -998..4645, T = 5644, 188.1 s).
- A **cell** is 1 s of timeline (cell i = [t_start + i, t_start + i + 1)). Coverage is stored per cell:
  `state` in `none | partial | complete` plus `stage` (the highest completed level) and `reasons[]`.
- Levels (in order): `front` (cameras + 2D + sync + prepared inputs exist) -> `fit` (raw 3D fit) -> `refined`
  (refined pose) -> `analysed` (timing/physics/counterbalance/floorcraft present for the cell).
- `complete` = refined, >= 3 cameras (or the span's maximum), seam gates passed, no quality flag worse than `multi`.
  `partial` = anything between: front only, raw fit only, `two_view` / `mono` spans, a failed or unrepaired seam,
  reprojection above the gate. `none` = nothing usable. `absent` (the lead before his entry) counts as `partial`
  with reason `absent` and is drawn distinctly.
- Coverage is COMPUTED from the real outputs (stitched npz `quality_flag` / `n_cams` / `segment_owner`, stitch.json
  gates, pose_quality), never only from job state. File: `work/<dance>/farm/coverage.json`:
  `{"dance", "t0", "t1", "cell_s": 1, "cells": [{"i", "t0", "t1", "state", "stage", "n_cams", "reasons": [], "chunk"}],
    "counts": {"complete": n, "partial": n, "none": n}, "seconds": {...}, "world_id", "updated"}`.
  (Another workflow also writes a first coverage_cells.json in the same cell format: accept both.)

## 2. Chunks and jobs

- A **chunk** = a fit window on the fused grid: `{id, k0, k1, core_k0, core_k1, overlap_k, kind, cams, world_id}`.
  Default core length 30 s (900 frames) with 4 s (120 frames) overlap into each neighbour (and into already-done
  spans); cores may stretch to 24-36 s so a gap of any length is covered by ceil(gap/30) equal chunks. Existing
  S1 segments a-d (37-43 s) are accepted as chunks as they are.
- Job kinds: `front` (ingest, detect, labels/SAM, localisation vs the world's room map, ma_2d, sync, prepare),
  `fit` (MAMMA full fit, betas pinned from the world), `refine` (pose_refine, recipe `smplx_rf2_noslip`), `analyse`
  (timing/physics on the stitched result; local or pod), `full` = front + fit + refine.
- Job spec (JSON, `farm/jobs/<chunk>.json` locally and `/workspace/hm/queue/<chunk>.json` on the volume):
  `{"schema": "dancecap.farm.job/1", "id", "dance", "kind", "chunk": {...}, "world_id", "world_sha256",
    "code": {"git_sha", "dirty": false|true, "bundle_sha256"}, "inputs": [{"path", "sha256", "bytes"}],
    "steps": [{"name", "cmd", "timeout_min", "optional"}], "outputs": [{"path"}], "max_minutes", "priority"}`.
  Every output records `world_id`, the code sha, the GPU, seconds and cost.

## 3. World package (shared consistently across sessions)

`dancecap/worlds/<dance_id>/<world_id>/manifest.json` (tracked: hashes + relative paths only) with the data under
`work/worlds/<dance_id>/<world_id>/` (git-ignored) and the same tree on the volume (`/workspace/hm/worlds/...`).
It contains, each with a sha256: the **room model** (SfM sparse model, cached features, keyframes), `world_from_sfm.json`
(gravity, floor, scale), the camera clocks (`alignment_studio.json`) and per-camera **types** (mount, zoom),
the per-dancer **betas** (`mamma_pin_betas.json`), the **skeleton configuration** (SMPL-X joint set, the MAMMA
ma_3d config template, the refine recipe parameters), the **avatar configuration** (shape edits, texture/groom
file hashes, shoe config - references, never copies into tracked paths). `world_id` = first 12 hex of the sha256
of the sorted member hashes. v1 = larissa_kadu_38's solved room + world frame + scale + betas (the 38 s take is the
reference). A take/chunk TOML names its world (`[world] dance = ..., id = ...`). Rules: a job refuses to run if the
volume copy's sha256 differs from `world_sha256`; the stitcher and exporters refuse chunks from different worlds
unless `--allow-mixed-world`; publishing a changed world makes a NEW id (old ids stay readable); a world can be
`sealed` once its room model is solved ("once it's been solved for").

## 4. Persistent volume (Runpod network volume)

- One network volume in a Secure-Cloud data center that has the GPUs we need (check `list-data-centers` with
  GPU_AVAILABILITY and `get-capacity`; network volumes exist only in some DCs). Size 150 GB to start (storage is
  about \$0.07/GB-month); the builder states the price before creating it. Name `hm-farm`. Its id and DC are written
  to `dancecap/farm_volume.json` (tracked: no secrets).
- Layout (mount `/workspace`, root `/workspace/hm/`):
  `env/` (venvs / conda, built ONCE, all features: base, mamma, dancers, cameras, ma2d, room, texture, check),
  `wheels/`, `weights/` (MAMMA, ma_2d, SAM 2.1, YOLO, ALIKED/LightGlue; sha256 manifest), `models/` (gated body
  models: stay on the private volume, never in git), `code/<git_sha>/` (dancecap + mamma-turing as committed; a
  `current` symlink), `worlds/<world_id>/`, `src/` (source videos, when `front` jobs run), `queue/` (job specs),
  `chunks/<chunk_id>/{in/, work/, out/, log/, claim.lock/, heartbeat, done.json}`, `bin/` (worker entrypoints).
- A pod started with the volume must run its first job within **3 minutes of the pod being up** (no bootstrap, no
  upload except new inputs). The builder MEASURES this and reports it.
- Multi-pod: first verify whether one network volume can be mounted by several pods at once (Runpod docs + a real
  test with two tiny pods). If yes: N pods share the volume and claim chunks (atomic `mkdir claim.lock`). If not,
  use one multi-GPU pod per volume (one chunk per GPU) and/or per-pod volumes seeded from the golden volume; pick
  the design that is fastest and report it honestly. Never rely on an unverified assumption.
- Uploads from the home PC are slow (0.9-2.8 MB/s): upload ONLY new inputs (hash-diffed, resumable, compressed);
  everything reusable lives on the volume. Prefer fetching public weights/wheels from the pod itself.

## 5. Worker protocol (pod side)

`python -m dancecap.farm.worker --volume /workspace/hm [--kinds fit,refine] [--max-jobs N] [--idle-exit-min 5]`:
loop: pick the highest-priority unclaimed job in `queue/` (atomic `mkdir chunks/<id>/claim.lock`, owner file with pod
id), verify the world sha256 and input sha256s, run `steps` in order under setsid with per-step `.done` markers
(resumable: a restarted worker skips finished steps), write `heartbeat` every 30 s (step, progress, GPU util),
write `out/` + `out/manifest.json` (sha256 per file) + `done.json {status, steps, seconds, gpu, price, world_id,
code}`; a failed job writes `failed.json` and releases the claim after N retries. Exit when the queue is empty
(or `--idle-exit-min`). The worker never deletes its pod: the local watcher does after the verified pull (a
fallback on-pod timer deletes the pod after a generous grace if the watcher is gone, see 6).

## 6. Local control (launcher + watcher), survivability

`python -m dancecap.farm {status, plan, queue, up, launch, watch, pull, coverage, world, volume, down}`:
- `plan`: coverage cells -> gaps (runs of non-complete cells, or only `none`/`partial` as asked) -> chunk list (30 s)
  -> job specs (skips what exists). `queue`: upload job specs + new inputs to the volume.
- `launch --pods N --gpus <priority list>`: starts N pods from the volume (cap/balance gates, kill switch generous
  and extendable), each running a worker. `watch` = one DETACHED process per dance (Windows DETACHED_PROCESS |
  CREATE_NEW_PROCESS_GROUP, log + pid file, idempotent, refuses a second instance): polls pods + heartbeats, pulls
  every finished chunk's `out/` immediately (sha256, resumable, retry with backoff, survives SSH drops), updates
  `work/<dance>/farm/state.json` (pods, chunks, spend, events) and `coverage.json` atomically every minute, extends
  kill switches while healthy and under the hard ceilings, re-queues failed chunks, deletes a pod only after all
  its chunks are pulled and verified, relaunches a lost pod's chunks elsewhere, writes `result.json` at the end.
  `status` and `wait N` print one JSON line for monitors/agents.
- Budget: every launch passes the gate `max_cost <= balance - 2` and `<= remaining cap`; hard per-pod ceilings;
  the watcher refuses to extend past them; spend is read from the ledger and the Runpod billing API.
- Fast stitch: chunks are stitched as they land (`dancecap.stitch`, generic over chunk lists); coverage updates.

## 7. Pipeline Monitor (redesign)

- Remove all old jobs/runs/stages from the monitor's view (archive them to `work/pipeline/archive/<date>/`; do not
  delete evidence; the registry/runs history stays on disk but is not shown). One dance at a time.
- Main view = the **defrag map** of the dance's final result: a grid of 1-s cells (about 190 for this dance), 3 states
  (complete = solid green, partial = amber with a hatch or lighter fill and a tooltip with the reasons / stage,
  none = dark), chunk boundaries and overlaps outlined, the span currently being worked on animated, a legend and
  big numbers (xx % complete / yy % partial / zz % none, seconds). Cells flip live as `coverage.json` changes.
- Below it: pods (GPU, price, state, age, spend), the active chunk jobs with progress bars + ETA, the queue, the
  budget (spent / cap / balance), the volume (size, used, price), world id + seal state. A "Fill gaps" action runs
  `farm plan` + `queue` + `launch` (with a cost/time estimate and a confirmation), and "Pull/Restart watcher".
- Reads only `farm/state.json` and `farm/coverage.json` (single source of truth); also exports `coverage_map.png`
  headlessly so agents can see it (`python -m dancecap.farm coverage --png`).

## 8. Git

Both repositories (workspace `HeadMovement-Pipeline`, Unity `head-movement`) keep a protected-in-practice `main`:
work on short branches or directly on main with small commits, conventional messages, a pre-commit hook that
rejects likeness / audio / keys / gated models / large binaries (`tools/githooks/`, `core.hooksPath`), `.gitignore`
verified, no pushes without the user's go-ahead (head-movement is PUBLIC MIT).

## 9. Modules and ownership (package `dancecap/dancecap/farm/`)

| module | owner (workflow agent) | content |
|---|---|---|
| `timeline.py`, `chunks.py`, `analysis.py`, `world.py`, `state.py`, `cli.py` | core | grid/cell math, coverage from outputs, planner, job specs (+ stitch / analyse / prepare pod jobs, evidence), world package, atomic state, CLI front |
| `volume.py`, `pods.py`, `worker.py`, `bootstrap/` | infra | volume create/attach/sync, pod launch from the volume with GPU priority + kill-switch extension + attach/reattach/delete, pod-side worker, one-time environment build |
| `watch.py` | watch | the detached watcher, pulls, survivability, coverage refresh, stitch-as-they-land hook |
| `dancecap/pipeline/*`, `tools/PipelineMonitor/*` | monitor | the redesigned Monitor |
| `dancecap/stitch.py` | (existing, other workflow) | generic stitcher |
Existing `remote.py` / `remote_jobs.py` / `front_full.py` are reused and only changed additively.

## 10. Change log
- 2026-10-08 v1: first version (orchestrator).
- 2026-10-08 infra (volume/pods/worker builder), measured facts that refine sections 4-6:
  - Volume `hm-farm` = id `8ncxy247rb`, EU-RO-1, STANDARD, 150 GB, $0.07/GB-month ($10.50/month provisioned), see `dancecap/farm_volume.json`.
    EU-RO-1 was chosen because it has an RTX 4090 / L4 / A4500 (the sm80+ GPUs the CUDA wheels run on; Blackwell sm120 does not) AND
    HIGH CPU stock for all six CPU flavors (cpu3c/3g/3m/5c/5g/5m): analysis CPU pods mount the volume there. GPU stock is LOW everywhere, live
    (a create can still fail with "no longer any instances"; `pods.launch` walks the priority list).
  - The volume is a network filesystem (MooseFS, mounted at /workspace): 1.2 GB/s sequential write, 480 MB/s read, but ~24 ms per small file
    created and ~12 ms per small file read. So section 4's `env/` is NOT a venv tree: the environment is built once on a pod's local disk
    (`/opt/hm/env`, same path on every pod) and stored as `env/env-<id>.tar` + `env/CURRENT` ("name sha256"); `bin/pod_prepare.sh` unpacks it
    (8.3 GB in ~16 s), the code bundle (`code/<key>.tar.gz` + `code/CURRENT`, ~5 MB) into `/opt/hm/ws` and the cached system .debs; the worker
    runs on LOCAL disk (`/opt/hm/ws`, scratch `/opt/hm/work` = dancecap/work) and touches the volume with a few large files only
    (`chunks/<id>/out/result.tar` + `out/manifest.json`). Models/weights stay as plain big files (`models/`, `weights/`), linked into the workspace.
  - Multi-attach is SUPPORTED: three pods mounted the one volume at once; `mkdir claim.lock` races over 30 rounds had exactly one winner per
    round; a file written by one pod was visible to another within 0.07 s. Several pods share one volume and claim chunks atomically (section 4's first option).
  - Pod start from the volume (no bootstrap, no upload): container up -> first job step in 28 s (CPU pod and RTX 4090 pod; env unpack 16 s,
    code 1 s, libs 10 s), API create -> ssh reachable 29-41 s. Target was 3 min. (An image not cached on a new host adds its pull time.)
  - Job spec additions used by the worker: `after: [chunk ids]` (claimable once each has done.json), `needs_gpu` (default by kind: front/fit/refine/full/asr
    need one, analyse does not; a CPU pod never claims a GPU job), inputs may be `{"path": "chunks/<id>/out/result.tar", "archive": true}` (another
    job's output) or carry an explicit `"src"`; steps may carry `save` (checkpoint tar written to the volume when the step succeeds, restored by any
    pod that resumes), `live`/`live_min` (periodic snapshot of a running step) and `env` (`{WS}` = the pod's workspace).
  - Analysis on pods (user directive, item 10): step templates `worker.analysis_steps(what, take, args)` for prepare, stitch, pose_quality, timing, physics,
    counterbalance, floorcraft, analysis_review, video_zoom, video_motion, gate_cameras, gate_final, attach_global; `worker.asr_steps` for the narration
    ASR (faster-whisper 1.2.1 / ctranslate2 4.8.2 in the volume env, large-v3 + large-v3-turbo in `weights/hf`, GPU fp16 tested); `pods.launch_cpu`
    for CPU pods (about $0.035/vCPU-h). The volume env imports every analysis module headless (`bin/import_check.py`, 0 failures).
    `typesafe_sdk` is installed so imports work but NO key is ever put on a pod: LLM judgments stay on the PC.
- 2026-10-08 core (world / timeline / chunks / state / cli builder), additions and measured facts:
  - WORLD ids (section 3). `world_id` = first 12 hex of `world_sha256` as specified; the same manifest also carries `pose_id` (the members that
    determine the 3D poses: room model, world frame, scale, clocks, camera types, betas, skeleton) and `avatar_id` (the avatar configuration).
    Re-baking a face / groom changes `world_id` but NOT `pose_id`; poses of one `pose_id` are interchangeable, so the stitcher / exporters call
    `world.check_same_world(headers)`: different pose ids are refused unless `--allow-mixed-world`, an avatar-only difference passes with a
    warning. A member's sha256 is the file's own sha256 when it is one file (the worker re-hashes), else a tree hash (`world.tree_sha`: sha256 over sorted
    "path NUL sha NUL bytes LF" lines). `manifest.json` (tracked, `dancecap/worlds/<dance>/<world_id>/`) lists the members (`root`, sha256, scope, source)
    and EVERY data file `{path, sha256, bytes, member}`; there is no separate files.json. Pointers: `worlds/<dance>/CURRENT.json`, `seals.json`
    ({pose_id: ...}: sealing freezes a pose id; `world publish` refuses a changed room while CURRENT is sealed, unless `--new-pose-world`). The data tree
    (`work/worlds/<dance>/<world_id>/`, also what the worker copies to the pod) holds `ref/` = the reference take's work layout (cameras/sfm, sfm_work,
    cache/feats, keyframes, people_masks_kf, frames/<cam>/frames.json + the map's window images, sync/offsets.json, scale/, mamma/ma3d_config.yaml) so
    `front_full.prep_segment(sid, base=world.ref())` runs from it unchanged, and `shared/` (alignment_studio.json, camera_types.json, betas.json (MAMMA
    body ids "0" lead / "1" follow), skeleton.json, refine_recipe.json, avatar.json = references only). On the volume it is `worlds/<world_id>/` (the dance
    is in the manifest). Take TOMLs carry `[world] dance, id, pose_id, role` (`dancecap.farm world pin`); the earlier take `larissa_kadu` has `member = false`.
  - Verified 2026-10-08 (`python -m dancecap.farm world compare <take>`): segments a-d share the reference's room model (cameras.bin / points3D.bin
    byte-identical, every image pose identical, image NAMES differ only by prep's renaming), the attached sub-models, `world_from_sfm.json` (byte-identical),
    clocks, camera types (c, d: only their cameras) and the pinned betas (the pin file is the reference's betas to 0.0). `larissa_kadu_38` has no camera types
    (solved before they existed). `larissa_kadu` (capture 02) is NOT in this world: world frame rotated 2.00 deg, scale 1.0545 vs 1.0065 (4.8 %), origin 0.62 m off,
    older clocks (alignment.json), its own betas: never mix its poses into this dance. Fits' betas are re-checked by `compare` once they exist.
  - COVERAGE (section 1) as implemented in `timeline.py`: frame levels 0 none / 1 front / 2 fit / 3 refined / 4 analysed; a frame is COMPLETE when it is part of
    the stitched result or of the reused reference span (`final`), refined, flag multi (or unknown), reprojection under the gate (> max(3 x the span's median,
    25 px); frames the fit's own repair step interpolated are not failures) and outside a failed seam; a cell is `complete` when >= 99 % of its frames are
    refined and >= 90 % complete, `none` with no data at all, else `partial` (reasons: front only, raw fit, N/30 frames refined, refined awaiting stitch,
    absent / mono / two_view, seam gate failed, reprojection over the gate). Segment / chunk outputs that are not stitched yet are therefore partial
    ("refined, awaiting stitch") but are NOT planned again. mono / two_view / absent frames whose phones are all used (`max_cams`) are unfixable and never
    planned. Sources read: reference take dirs, work/<dance>/{smplx,<refined>}/ (+ quality_flag, n_cams, segment_owner) and work/<dance>/stitch.json (seams:
    a list or dict, window as blend_s / blend_k / window_s ..., verdict ok / passed / status / gates; `repaired: true` turns a failed seam ok), the segment
    and chunk takes work/<dance>_<id>/ (mamma/prepare.json, mamma/fits/*/smplx, smplx/, <refined>/), physics + timing for `analysed`, and optionally another
    workflow's coverage_cells.json which can only RAISE a cell. `farm/coverage_frames.npz` keeps the per-frame arrays for the planner.
  - CHUNKS (section 2): gap = frames of non-complete cells that have no refined 3D yet (or sit in a failed seam), minus claimed chunks and unfixable frames;
    a gap becomes equal cores of 24-36 s nearest to 30 s (<= 36 s: one chunk; 33.3 s -> 1, 116.6 s -> 4 x 29.2 s; a tie goes to more chunks), each fitted
    with a 4 s overlap into every neighbour (none past the timeline's ends, window grown to 12 s for tiny gaps). Ids: `g<frames from the timeline start>`; the
    S1 segments keep `a`-`d` with cores cut at the middle of their overlaps (reference span untouched: a -998..-1, b 1148..2252, c 2253..3418, d 3419..4645).
    Job specs follow the worker's protocol (workspace-relative `inputs[].path`, archives for prepared inputs, `outputs[].path` workspace-relative, `save` / `env`);
    step templates are `chunks.STEP_TEMPLATES`; a chunk take is generated on the pod by `farm chunk-take`, its map reuse from the world by `farm chunk-prep`
    (`source_dir` = $HM_ROOT/src/<capture>, clocks from $HM_WORLD_DIR), the pinned fit by `farm chunk-fit`.
  - CLI plug-ins: any non-core module of `dancecap.farm` that defines `register(sub)` adds sub-commands (and `cli.register_hook(name, fn)` hooks such as `queue_upload`,
    `coverage_updated`) without editing cli.py; `farm plugins` lists them and their import errors.
  - Additive edits to existing code: `front_full.prep_segment(sid, force, base=None)` (reference tree to copy from) and `write_segments` emits `[world]` from the
    full take's table.
- 2026-10-08 core (analysis on pods, item 10): `farm/analysis.py` plans every non-fit job; the pod side is worker.py (`after`, archive inputs of
  `chunks/<id>/out/result.tar`, CPU kind `analyse`, `worker.ANALYSIS` = the one table of base commands).
  - Farm-level kinds (job field `what`; worker `kind` is always `analyse` so `pods.launch_cpu(kinds="analyse")` claims them): `prepare` (re-run mamma prepare of a
    chunk), `stitch`, `analyse` (registered analyses on ONE take), `stitch_analyse` (the final job `farm plan` emits: stitch + analyses + coverage evidence on ONE CPU
    pod). GPU kinds stay `front`, `fit`, `refine`, `full`. `chunks.KINDS` lists both families.
  - `farm plan` (default) writes the chunk jobs AND one `final` stitch_analyse job: `after` = the chunk ids whose motion is not already on this machine, inputs =
    `chunks/<id>/out/result.tar` archives (no transfer: the results are on the volume) + one `motion.tar` per chunk that exists only here (S1 segments fitted
    by an earlier session) + `reference.tar` (larissa_kadu_38's raw + refined motion and the studio beat grid, ~19 MB, uploaded once). Steps: `stitch-prep`
    (checks every upstream manifest + the pose id, copies the world frame, writes farm/stitch_list.json), `python -m dancecap.stitch <dance> --chunks ... --partial`
    (rc 2 = a seam gate failed: the verdict is in stitch.json and coverage, not a job failure), the analyses (timing, physics on the raw fit with the studio beat
    grid; counterbalance, floorcraft on the refined motion, as larissa_kadu_38), then `chunk-evidence final`. Every chunk job also ends with `chunk-evidence`
    and runs the optional per-chunk `quality` (pose_quality) step on the pod that already holds the inputs.
  - SMALL RESULTS: `work/<take>/farm/evidence.npz` (a few KB: per-frame level / cameras / flag / reprojection gate / seam failures, `timeline.Frames.to_npz`) is
    written by every pod job and declared as an output; `timeline.collect_evidence` computes coverage from those files alone (a chunk's own evidence is NOT the
    stitched result: `final` comes only from the dance job's evidence), so the PC never reads big motion files. The dance job also writes `farm/coverage_pod.json`.
  - `farm queue` uploads every planned job at once (the worker enforces `after`); `--hold` keeps jobs with unmet `after` locally and `cli.release_ready(dance)` (for the
    watcher, once per cycle) uploads them as their chunks finish. `farm analyse [--analyses a,b] [--take T] [--list]` plans registered analyses on one take
    (`analysis.register_analysis(Analysis(name, base, args, ...))` adds more; scope `chunk` analyses run per chunk take, not on the stitched dance).
  - Asks of the infra / watch stages: (1) CPU pods need, in the volume env, everything `dancecap.stitch` (numpy, torch CPU + smplx FK, matplotlib for the two sheets) and
    timing / physics / counterbalance / floorcraft import (scipy, numpy, matplotlib; no pycolmap needed) and the locked-head SMPL-X model reachable as
    `mamma/data` (`DANCECAP_MAMMA_DATA`); (2) the watcher should extract each pulled `result.tar` at workspace paths (as it does) and then call
    `cli.release_ready` only when running with `--hold`; (3) a worker `group` hint ("small" first) on outputs would let evidence.npz / coverage_pod.json come home
    before the stitched motion (today all outputs are in one result.tar).
