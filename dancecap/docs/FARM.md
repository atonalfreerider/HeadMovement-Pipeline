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
- 2026-10-08 monitor (Pipeline Monitor redesign, section 7), additions and decisions:
  - The monitor reads `farm/coverage.json` and `farm/state.json` as written by core / watch (it also accepts `coverage_cells.json`
    when `coverage.json` is missing) and the live `dancecap.fit_full` sessions through the read-only adapter
    `pipeline/fit_sessions.py`; a segment whose session has a pod and a running job is drawn as a running chunk (animated cells).
  - When a fit / refine / quality result lands under `work/<chunk take>/` (newer than coverage.json) the monitor starts
    `farm coverage` itself (detached, one at a time, at most every 2 min); a watcher that already refreshes coverage makes this a no-op.
  - Fill gaps = `farm plan --dry-run` (estimate, GPU offers, budget gate) -> confirmation with `plan_hash` + `approve.max_cost_usd` ->
    `farm up --pods N --minutes M --hard-minutes H --budget B --only <planned ids> --exclude <live fit_full chunks>`; the chunks a fit_full
    session is fitting are first marked `running` (owner fit_full) in state.json so no planner or queue picks them up. Chunks whose
    estimate is 0 min (core: nothing left to run) are not planned; the stitch + analysis `final` job is planned (CPU pod) only when no
    fit_full session is still fitting. Stop pods = `farm down --pod ids --abandon` (verified pull first), only pods in this dance's
    state.json, never a fit_full pod.
  - Asks: (1) `farm_volume.json` / state.json["volume"] could carry `used_gb` + `measured` (the monitor shows a stored measurement
    work/monitor/farm_volume_usage.json = 24.6 GB from the infra report until then); (2) a `final` job that waits for chunks fitted
    outside the farm (a-d) needs those chunks' motion pulled locally first, otherwise its `after` list never clears.
  - Old runs / plans archived to work/pipeline/archive/2026-10-08/ (`python -m dancecap.pipeline archive-legacy`).
- 2026-10-08 watch (watcher + launch + pull + down; `farm/watch.py`, `farm/pull.py`, `farm/launch.py`), what exists and how it behaves:
  - COMMANDS (registered through the CLI plug-in mechanism; `farm queue` now loads plug-ins too via `cli.HOOK_COMMANDS`, so the volume uploader
    hook `queue_upload` is installed, and `queue` takes `--only/--exclude`):
    `farm up [--pods N --gpus 4090,l4 --minutes M --hard-minutes H --budget USD --only ids --exclude ids --hold --interim-stitch --dry-run]`
    = plan -> sync code + world to the volume -> queue (inputs hash-diffed) -> launch GPU pods -> detached watcher. `--dry-run` prints the plan,
    the queue, GPU offers and a cost/time estimate (A100 minutes x `--slowdown`, default 1.5: the 4090 route is not measured) and creates nothing.
    `farm launch` (the pod part alone; `--cpu N` adds CPU analysis pods), `farm watch [start|run|status|stop|wait]`, `farm wait N`,
    `farm pull [ids] [--redo]`, `farm down [--pod ids] [--abandon]`. Progress goes to stderr, `--json` output to stdout.
  - POD NAMES AND OWNERSHIP: pods are named `hm-farm-<dance tag>-...`; the watcher manages EXACTLY the pods recorded in state.json["pods"]
    (plus an orphan of this dance's prefix it finds on start-up and adopts). `farm down` deletes only recorded pods, never by prefix. Chunks
    with `owner` other than farm (the monitor marks live fit_full chunks `running`, owner fit_full) are ignored; `queue_upload` also skips a
    chunk with an open `dancecap-fit-<id>` ledger row (`--force-foreign` overrides). GPU pods run `--kinds front,fit,refine,full,asr` and never
    claim the CPU analysis jobs (kind analyse); the watcher launches one CPU pod when such a job becomes claimable.
  - THE WATCHER is one DETACHED process per dance (Windows DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, breakaway from the parent job object when
    allowed; every console child it spawns gets CREATE_NO_WINDOW so ssh / runpodctl never flash windows), single-instance by an OS file lock
    (`farm/watcher.lock`, freed by the OS when the process dies: no stale pids), `farm/watcher.pid` (pid, heartbeat), `farm/watcher.log`
    (rotated at 5 MB), `farm/watcher.stop` (graceful stop), `farm/result.json` at the end. Started again after a PC restart it rebuilds
    everything from the volume snapshot (ONE ssh call per cycle through any pod that answers), Runpod's pod list, `farm/chunks/<id>/pull.json`
    and state.json. Cycle = 30 s; state.json every cycle, coverage.json at once after a pull and at least every 60 s. Verified live on this
    PC: start, second start refused, hard kill of the process tree, restart, graceful stop.
  - PULLS: every `done` job at once, 2 at a time, through any pod (the volume is shared): `result.tar` resumable by byte offset (survives ssh resets),
    size + sha256 against `out/manifest.json`, then every member extracted at its workspace path and hash-checked while written (temp file +
    atomic replace, long-path safe), `pull.json` written LAST (the proof), `work/<take>/farm/pulled.json` records world_id / pod / gpu / code.
    A failed pull backs off 10 s .. 5 min per job and never marks anything pulled; a per-job pull lock (dead owner pid = taken over at once) lets
    the watcher, `farm pull` and `farm down` coexist. A pod is deleted only when it is idle/exited AND every job it finished is `pulled`
    (never the pod it reads through while a pull is in flight and no other pod exists).
  - FAILURES: a pod missing from Runpod's list on two consecutive polls AND without a fresh heartbeat is LOST (an unreadable pod list is never
    read as "no pods"): its claims are released at once (guarded: only if it still owns them), the ledger row closed without calling Runpod, a
    replacement is launched when work is claimable. `failed.json` -> re-queued up to `max_requeues` (default 2; evidence kept as failed.<ts>.json,
    attempts reset, the pod that failed it is recycled), then `failed` (a job downstream of it is marked `blocked`). A claim silent for 10 min on a
    pod Runpod still lists: claim released, pod deleted, job re-queued (3 strikes = failed). A pod without a worker heartbeat after 13 min is deleted;
    two such pods in a row halt the watcher's launching (the pod's `events.log` tail is in state.json as `last_event`). Nothing is deleted while the
    volume cannot be read (no fresh snapshot). A second `launch` / `up` tops up to the target and to the number of waiting jobs, never doubles.
  - MONEY: GPU pods start with a 100 min kill switch and a 240 min hard ceiling (`--minutes/--hard-minutes`); the watcher extends the on-pod
    deadline (`pods.extend_killswitch`) by 30 min when < 25 min are left, only for a busy pod with a fresh heartbeat, never past the ceiling
    (`hard_ceil` of the pod, ledger as fallback) and only if the next step fits `min(cap left, balance - $2)` and the farm budget. Launches go
    through `pods.launch` / `launch_cpu` gates (`max_cost <= balance - 2` and `<= remaining cap`, counting every open pod's worst case) plus the
    run's own `--budget`; at most 12 watcher launches per run, cool-down after a failed launch (120 s doubling, max 15 min), one pod per cycle.
  - STITCH: default = the planned `final` stitch+analyse job on a CPU pod (FARM.md 0.10). `--interim-stitch` also queues `stitch-N` jobs
    (reference span + the chunks the farm has finished, `--partial`, evidence only) as chunks land, until the final job can run. `--stitch local`
    runs `dancecap.stitch` here (single thread, normal priority, skipped with a note when the module is absent). The uploader never sends what the volume
    holds: a stitch/analysis job that lists a farm-produced chunk as a local `motion.tar` input is rewritten to take `chunks/<id>/out/result.tar`
    from the volume (`launch.prefer_volume_results`), and a finished `chunks/<id>/` of an earlier plan with the same id is moved aside
    (`chunks/<id>.old.<ts>`, never deleted; a claimed one is refused).
  - state.json written by the watcher: `pods[id]` {role gpu|cpu|gateway, gpu, price_hr, state starting|running|idle|exited|stuck|lost|deleted, age_min,
    spend_usd, kill_at, kill_in_min, hard_ceil, jobs, extensions, last_event, runpod_status, worker, hb_age_s, launched_by, cost_usd},
    `chunks[id]` {status, pod, progress 0..1, eta_min, step, step_i, steps_n, gpu_util, hb_age_s, attempt, requeues, lost_count, last_error,
    pull {state pulling|ok|error, bytes, total, attempts, next_at, error}, done {seconds, cost_usd, gpu, steps}, world_id}, `policy`, `spend`
    {farm_usd, farm_committed_usd, rate_hr, balance, cap_remaining, budget_usd}, `watcher` {pid, heartbeat, cycle, state, link_pod, stats}.
  - Not billable-tested: everything above is verified with fake pods (tests/test_farm_watch.py, test_farm_watch_ops.py: a local bash plays the pod,
    real `worker.Worker`s write real done.json / result.tar). First real run = the drill. Open for the drill: the GPU speed / VRAM of the RTX 4090
    route; whether the system `python3` of the pod image runs the snapshot script (stdlib only, 3.5 KB, sent through a heredoc as infra's scripts are).
- 2026-10-08 drill (end-to-end proof on the real volume; report: work/larissa_kadu_full/farm/DRILL.md, evidence: work/larissa_kadu_full/farm/drill/evidence/).
  `farm/drill.py` (`python -m dancecap.farm drill setup|make|verify|cleanup`) runs the farm on a SCRATCH dance `farmdrill` (takes/farmdrill.toml, the same 30 Hz
  lattice and the SAME world package as larissa_kadu_full, its own state / coverage / watcher / pods, so the live dance, its watcher and the fit_full sessions are
  untouched; `drill cleanup` MOVES the scratch dance (tracked-path files and work dirs) to work/pipeline/archive/<date>/farmdrill_drill/: the Monitor lists every dance with a
  farm directory, newest first, so a left-over drill would open as the default view). Its chunks are exact slices of larissa_kadu_38's prepared MAMMA inputs: dra / drb / drc 40 frames, drs
  300 frames, dr9 900 frames (a real 30-s chunk); outputs are compared with the existing 38 s fit / refine on the same frames (`drill verify`).
  MEASURED (RTX 4090 pods of the hm-farm volume, EU-RO-1, $0.89/h, one pod per job): container up -> first job step 30 s (api create -> first step 36 s); ma_3d fit
  of 40 frames 105-144 s, 300 frames 350 s (0.52 s/iter, 2.2 GiB), 900 frames 952 s (6.4 GiB, 94 % util); model ma_3d_s = 74 + 0.92 x frames (predicted 903 s for 900, measured
  952 s); refine 19-32 s; a whole 900-frame fit + refine job 1015 s = $0.25. Versus the A100 model (`chunks.fit_seconds`: 2362 s for 900 frames, 56 min per 1286): the 4090 needs
  0.41 x the time at $0.89/h against $1.39-1.79/h, i.e. about 2.4 x faster and 4-5 x cheaper per chunk. `launch.GPU_SLOWDOWN = 0.45` / `farm_actions.GPU_SLOWDOWN` replace the
  guessed 1.5 in every estimate. Correctness on the volume's environment: every window passes (joint error vs the 38 s result: 900 frames median 1.9 / 2.1 mm, p95 5.8 / 8.2 mm;
  40-300 frames median 6.6-11.7 mm, p95 22-37 mm; reprojection equal or better; betas identical to 0.0). Pulls: done -> verified at home 15-40 s (one 30-s watcher cycle + 8-10 s),
  1.3-22 MB. A pod is deleted only after its pulls verify (log lines "deleted pod ...: all its jobs are pulled"). Watcher hard-killed mid-run (taskkill /T): restarted 13 s later,
  first cycle 9 s after, nothing lost. ssh cut for 4 min (watcher's ssh children killed every 0.1 s): 8 snapshot failures logged, nothing deleted or launched, the pull
  started 20 s after the link came back, verified after 33 s, the pod deleted after 64 s. Kill switch: watcher extension verified on the pod (kill_at file = now + 40 min, capped at the hard ceiling), manual `pods.extend_killswitch`
  +10 min read back over ssh (1791504163 -> 1791504763; hm_killswitch.sh armed with the ceiling). A tampered world_sha256 is refused by the worker before any step (failed
  twice, `max_requeues` 1, then `failed`). Drill spend $1.19 for 18 pods (0.54 of it on pods that hit the three bugs below).
  BUGS FOUND AND FIXED (each only visible on a real pod / the real world): (1) the world package had NEVER been uploaded to the volume (infra never ran `sync_world`), and
  `Link.state` put 501 paths into one ssh command line (WinError 206 on Windows): batched (`volume.STATE_BATCH`); a failed first sync also left a manifest without its files,
  which `sync_world` then treats as complete (manifest-last rule): the partial world dir was repaired by hand, full sync 343.7 MB in 711 s; (2) `world.verify` counted the
  worker's `.installed` marker as a stray file, so EVERY job's `chunk-fit` died at the world check; (3) `worker._safe_extract` resolved targets through the pod's symlink
  `dancecap/work -> /opt/hm/work` and silently skipped every member there (prepared inputs / archive inputs never arrived); lexical containment now; (4) `worker._stale`
  treated a claim directory without owner.json (the winner between mkdir and write) as dead: two pods could claim one job (test_claim_is_exclusive_between_workers);
  (5) `farm --help` / `farm plugins` printed the status help / no plug-ins; (6) tests/test_front_full.py still expected pid-keyed pin betas; (7) the monitor flagged every farm pod
  as an "orphan" (runpod_view looked for fit_full sessions): farm pods tracked by an alive watcher are exempt.
  STILL OPEN (watch-owned): `launch.archive_stale` moves a failed chunk dir aside including its uploaded `in/` (a re-queue re-uploads it: 50 MB cost 2 min); a deterministic
  failure recycles a pod per re-queue (about $0.02 each); a worker never re-claims a job it failed, so idle pods are useless after a fix and wait for the watcher's idle
  deletion; the watcher counts a pod deleted outside it as committed budget until its next two polls (a launch was refused meanwhile); home uplink 0.5-1 MB/s (the 343 MB world
  took 12 min: upload it ONCE, it is on the volume now); the ledger rows `podidexample05` / `podidexample06` still over-count.
- 2026-10-09 speech (S1, class recap ASR + diarization; package `dancecap/speech/`, notes `work/larissa_kadu_recap/NOTES_speech1.md`): an additive job on the existing
  worker, no change to any farm module. One `asr` job spec (`speech.jobs.job_spec`, ids `speech-recap-N`) = 10 idempotent steps (`speech.pod.run <stage>`: vad, whisper large-v3 /
  large-v3-turbo, parakeet-tdt-0.6b-v3, canary-1b-v2, embed, events, sortformer, segembed, quality; `regions` on request) over seven 16 kHz wavs uploaded ONCE to
  `src/larissa_kadu_recap/speech_audio/` (66 MB, shared prefix: the worker reads them from the volume). The job brings its own code (`speech_code.tar` archive input, not the volume's
  code bundle) and its own environment: `speech/speech-env-<id>.tar` + `speech/CURRENT` on the volume (9.1 GB = the farm env + `nemo_toolkit[asr]==2.5.3`, speechbrain 1.1.1,
  transformers 4.53.3, built by `speech/bootstrap/speech_env.sh build`, pins in its constraints file: numpy 1.26.4, torch 2.5.1, pyarrow<20, datasets<4), unpacked by the job's first
  step to `/opt/hm/env_speech` (25 s) next to the untouched farm env. Measured on an L4 ($0.59/h): env unpack 25 s, whisper large-v3 3 passes + language id on 7 sources 559 s,
  turbo 162 s, parakeet 38 s, canary 50 s, embeddings 233 s, events 53 s, sortformer 16 s. Facts for other builders: NeMo 2.5.3's own word-level confidence aggregation raises on a
  few hypotheses (`RuntimeError ... len(words) 76 len(word_confidence) 77`), so `stage_parakeet.word_confidences` rebuilds it from token confidences; `pip install --target` ignores the
  installed packages (so the env is extended in place and packed whole); Git Bash rewrites `/tmp/...` arguments: set `MSYS_NO_PATHCONV=1` for `speech pod put/get/sh`.
- 2026-10-09 recap cameras (R2; `cameras/newcam.py`, `cameras/newcam_report.py`, `cameras/attach_people.py`, `recap_cameras.py`, notes `work/larissa_kadu_recap/NOTES_R2_cameras.md`): an
  additive use of the worker, no change to any farm module. The seven recap videos (1.2 GB) and the owner's 2D pose JSONs (gzip, 91 MB) are on the volume ONCE at
  `src/Larissa-Kadu-recap/<file>.mp4` and `src/Larissa-Kadu-recap/poses/<file>.mp4.json.gz` (+ `UPLOAD.json` with sha256; the capture folder name is `Path(source_dir).name` as
  `chunks.py` does; the home uplink gave 0.8 MB/s on 6 striped streams, 0.12 MB/s on one: the 817 MB HEVC file took 20 min). Jobs: kind `front` (GPU), world `612dfea3d607`, the code bundle
  pinned by `code.bundle_sha256` (`volume.sync_code(link, make_current=False)`: the pod's `code/CURRENT` is never moved by this run), ONE job per camera (`rc-cam-<cam>`: calibration, then its
  60 s chunks, 3 processes at a time on the pod's GPU). Measured on RTX 4090 pods ($0.89/h, EU-RO-1): one tracking process leaves the GPU at 20-40 %, 160 ms per localised frame; 3
  processes 40 ms per frame together; a camera job (calibration 3.5 min + 6 chunks) takes 5-9 min; 6 cameras on 2 pods took 15 min wall, about $0.47 of GPU-pod time; the whole camera job of the class, with the upload gateway and the checks, cost $0.73. The new cameras are NOT in
  the room model, so `Localizer` (same-camera map images) cannot be used: `newcam` registers anchor frames against the map images, estimates ONE focal + k1 per camera (the owner: handheld,
  no zoom) and tracks every 2nd 30 Hz frame with the intrinsics fixed; the pod needs ffmpeg, pycolmap and lightglue only (all in the volume env).
- 2026-10-09 recap people and chunk plan (R3; `recap_people.py`, `recap_people_run.py`, `recap_attach.py`, `recap_front.py`, `farm/recap_profile.py`, additive edits of `farm/chunks.py` and `farm/cli.py`,
  notes `work/larissa_kadu_recap/NOTES_R3_people.md`): a take opts in with `[chunks] mode = "windows", profile = "recap"` and `farm plan --dance larissa_kadu_recap` then calls `recap_profile.cmd_plan`
  (a take without the key plans exactly as before; `farm/chunks.py` got `chunk_params`, `equal_windows`, `plan_windows` and `plan_gaps(core_target_s=)`, nothing else changed). Chunk convention of this profile: the
  FEWEST EQUAL fit windows of at most `chunk_s` (60 s) whose neighbours overlap by `overlap_s` (4 s), cores cut at the middle of each overlap like the S1 segments (360.48 s = 7 x 54.93 s; six windows would need 63.4 s,
  a fixed 60 s window stepping by 56 s would leave a 24.5 s last chunk). Cameras per chunk come from the teacher coverage (`people/teachers_<cam>.npz`: a phone joins when it records >= 35 % of the window and sees
  both teachers in >= 30 % of its frames, or one in >= 60 % and the other in >= 15 %). Jobs: one `shape` job (30 s window with the best camera coverage: front + fit with FREE betas + `recap_front pin` ->
  mamma_pin_betas.json per MAMMA body id), seven chunk jobs (kind full: chunk-take, `recap_front run` = frames / cameras / people / SAM 2.1 with the teacher tracks / ma_2d / sync re-gauged to the master camera 01 /
  prepare, then the fit with the shape betas PINNED and FROZEN (the demo's mechanism), refine with the demo's recipe smplx_rf2_noslip, pose_quality), one CPU `final` job (stitch only: no soundtrack, so no timing / physics /
  counterbalance / floorcraft, and no world check: the recap has no pose world). world_id is None for every job: the room is the demo's, the clocks, camera types and betas are not. R2's camera tracks, R3's teacher
  tracks and the owner's 2D pose tables travel as ONE shared tar on the volume (`src/Larissa-Kadu-recap/recap_in.tar`, about 100 MB, uploaded once). Offline validation (`recap_profile.validate_plan`, printed by
  `farm plan`): unique ids, `after` references without a cycle, every `python -m` module importable and every farm helper a real sub-command, the specs load through the WORKER's own queue reader and its claim rules
  behave, the chunk take TOML of every job parses and names the right cameras, every output is a workspace-relative path under the job's own take, the windows tile the timeline with the overlap. Cost model on RTX 4090
  pods ($0.89/h): 8.8 GPU hours = 7.8 USD for shape + 7 chunks (10.2 with a 30 % margin), about 118 min wall with one pod per job; the front is the least certain part of it (SAM / ma_2d rates are the demo's A100
  rates, not measured on the 4090). The first real run of the people stage (`recap_people_run`) was an L4 / 4090 job of about 10 minutes and 0.1 USD.
- 2026-10-09 recap fits (F1, the launch and monitor of the class recap, notes `work/larissa_kadu_recap/NOTES_F1.md`): the recap profile (`farm/recap_profile.py`) run for real, 8 RTX 4090 pods of the
  volume in parallel, 7 chunks + the shape job fitted, refined and quality-checked in about 2 h of wall clock for about 13 USD of farm spend. What the real run taught (each fixed or documented):
  - The chunk jobs are claimable at once; the front (SAM / ma_2d / sync / prepare, 12-35 min) runs in parallel with the shape pass and a step `await_pin` (`recap_front await-pin`) waits on the volume for
    `chunks/shape/done.json` and extracts the pin from the shape job's `result.tar` (no `after` on the shape job: it would have serialised 40 minutes). A 1650-frame chunk fit took 26-29 min on a 4090 (900 frames,
    6 cameras: 24 min; the planner's 26.5 min for 1650 frames was right for 4 cameras, the shape pass was slower than modelled).
  - BUG, every recap job died at the front task `merge` after SAM and ma_2d had run: `dancers/labels.json` is written only by `front_pod.merge_label`, which the demo's DAG calls as `on_ok` of each `sam:<cam>` task and
    `recap_front.build_tasks` did not. Fixed (`on_ok=merge_label`). A running job cannot take new code (the pinned bundle is part of every step fingerprint), so the 7 live pods were hot-fixed over ssh.
  - SHARED QUEUE HAZARD: every worker pod claims ANY claimable job of `queue/`, whoever queued it. Two agents' farm jobs were swapped between two pods (one finished the other's job, the other's pod, deleted by its
    owner at the end of ITS job, took one of ours down after 7 minutes). Before `up` check `chunks/` for unclaimed foreign jobs and tell their owner; a stolen job whose pod is gone is released with
    `pull.release_claim` (the watcher does not release a claim of a pod it does not own).
  - `dancecap.sync` per chunk: the six windows from 100 s agree (video offsets 00 +99..+125 ms, others within +-25 ms, residual 2.4-8.8 px); the first window (0-55 s) does not (16-71 px, estimators 299 ms apart,
    one phone at the search edge). `recap_front.sync_policy` replaces an unreliable window estimate (residual > 10 px, estimators > 60 ms apart, search edge, `use=false`) by the median of the reliable windows
    (`REF_OFFSETS_MS`); the window's own estimate is kept as `sync/offsets.window_estimate.json`.
  - `farm refit <chunk> --mask cam:t0:t1[,...]` (`farm/recap_refit.py`): refits a finished recap chunk from its FRONT CHECKPOINT (no front) with a camera left out of a stretch (`recap_front mask` zeroes that
    camera's landmarks in `mamma/ma_2d`, as `mamma prepare` does for an unregistered frame). First results (`work/larissa_kadu_recap/chunks/v1/<id>/`) are kept, the volume dir is moved aside by the uploader,
    the new job has the same id. The reasons were camera-specific failures that drag the whole chunk: phone 06 on the follow in g4644 (follow reprojection 21 -> 5.4 px after dropping it), phone 02 at 242-278 s
    (its R2 track moves 1 m along the optical axis: 100-180 px; g6171, g7699), 02 at 350-355 s and 01 at 328-333 s in g9227. A refit costs about 1.1 USD and 35 min.
  - `python -m dancecap.recap_fitreport` summarises `quality/quality_rf2_noslip.json` of every chunk against gates taken from the demo's segment a (all-cameras reprojection <= 8 px median / 12 px p90, per camera
    <= 10 px (14 for the attached 05 / 06), rest height within 3 %, floor, jitter): it passes the demo segment, 6 of 7 recap chunks pass after the refits, g0000 (10-30 s: the follow is fitted badly in 00 and 01) does not.
- 2026-10-09 recap avatars (A1; `recap_avatars.py`, `recap_avatars_pod.py`, `recap_avatars_review.py`, `recap_avatars_select.py`, `recap_face_lm.py`, notes `work/larissa_kadu_recap/NOTES_A1.md`, plan `AVATAR_PLAN.md`): an additive use of the worker, no change to any farm module. The pods of this track
  run jobs of a DEDICATED kind `avatar` (id `ra-<stage>-<n>`, `HM_KINDS=avatar` on the one pod the run starts): the fit pods (`--kinds front,fit,refine,full,asr`) and the analysis pods (`--kinds analyse`) can never claim them, so no shared-claim-queue collision
  (tests: `tests/test_recap_avatars.py::test_pod_job_is_a_dedicated_kind_no_other_worker_claims`); ledger owner `A1-recap-avatars`; one job -> one pod (CPU 8 vCPU for the FLAME identity fit, a 4090 for the texture bake) -> pull -> pod deleted in a `finally`. The gated
  FLAME 2020 + SMPL-X <-> FLAME vertex map and SMIRK (MIT, code + 140 MB checkpoint) sit ONCE on the owner's private volume under `models/flame/`, `models/smirk/` (uploaded in 527 s). Measured: identity job on a CPU pod (cpu5c x 8, $0.28/h) 4 min of work for two teachers (SMIRK + 600 landmark iterations per
  image, 10-13 images each), pod up -> pulled 3 min; texture dry run on a 4090 ($0.89/h) over the shape pass's 30 s window with six phones: front 601 s (frames 21 s; SAM 2.1 per phone 114-267 s, two to three side by side; no ma_2d / sync / prepare: `recap_avatars pod-texture-front` runs
  `recap_front.build_tasks` without its MAMMA tail), bake 480 s (1799 views, 2048^2, top_k 4, two registration passes, peak GPU 3.8 GB), pod time 19 min = $0.28. The farm's chunk jobs return neither frames nor SAM masks, so the avatar texture needs that front again, on the chosen window only.
  Gotcha: `farm.chunks.pack_tar` writes mtime 0, so a smplx/ delivered in an archive input always looks older than `cameras/world_from_sfm.json` to `avatar_texture.smplx_freshness`: the job's `stage` step touches it.
- 2026-10-09 recap stitch, analyses and first capture (F2; `stitch_provisional.py`, `timing_free.py`, `farm/recap_analysis.py`, `recap_export.py`, notes `work/larissa_kadu_recap/NOTES_F2.md`): the `final` stitch job the planner had written
  was queued as it was (one step edited: `--force`, so a failed seam gate still writes `smplx/` and the verdict stays in stitch.json) and ran on ONE CPU pod (8 vCPU cpu5c, $0.28/h): up -> done 70 s, pulled 26 s, pod deleted by the
  watcher, about $0.01; 10815 frames, every owned frame bit-identical to its chunk (raw and refined), the pinned betas identical in all seven chunks (max |db| 0.0). `python -m dancecap.stitch_provisional <take> --decision ... --weak ... --write`
  checks the arrays against the pod's sha256 and writes `flags.provisional` + PROVISIONAL.md (every seam, what exceeded which gate, weak stretches, the 12-s refit windows). 4 of 6 seams failed their gates (wrists / fingers first).
  - Analyses without music (`farm recap-analysis`, plug-in `farm/recap_analysis.py`, one CPU job each behind `final`, inputs = `chunks/final/out/result.tar` from the volume): `dancecap.timing_free` (the demo's touchdown detector,
    accents and kinematics.npz, step intervals / bouts / cadence / lead-follow offsets, NO beat grid, tempo or asynchrony; `worker.ANALYSIS["timing_free"]`), `physics` (it skips its beat impulses itself), `counterbalance` and
    `floorcraft` (both fall back to a nominal 76.4 BPM beat LENGTH as a time scale when there is no grid: their `*_beats` fields mean nothing, the seconds fields are real). Moves labelling was NOT run (no classifier exists; features /
    proposals pool per eighth note and would invent a constant 76.4 BPM grid).
  - MEMORY: the first runs of timing and physics on 16 GB `cpu5c` pods were SIGKILLed (rc -9) after 11 minutes, twice each. `timing.py` builds dense smoother operators of (4 x samples) x samples (about 3.7 GB each for 360 s,
    cached per cutoff), so memory grows with the SQUARE of the take: `timing_free` analyses 30-s cores with 3 s of context (the demo's 188 s fitted). `physics` needs more than 16 GB on 10815 frames: job kind `analyse_mem`, claimed only
    by a pod started with `kinds=analyse_mem` on a cpu5m flavor (64 GB, $0.52/h; `recap_analysis.launch_mem_pod`). `farm.watch.RunpodBackend.launch_cpu` does NOT pass `flavors` on (a `flavors=["cpu5m"]` call through it starts a 16 GB cpu5c),
    so call `pods.launch_cpu` directly. A job that fails with rc -9 is re-queued and the next pod fails it again: remove `queue/<id>.json` and queue a new id (`farm chunks cancel` only marks the state). The wall times on the pods:
    timing 8 min (windowed, 8 workers, 60 draws), physics 23 min, counterbalance + floorcraft 14 min. Total pod spend of this stage about $0.5.
  - `demo_export.floor_layer` RE-ANALYSES on the workstation when `counterbalance.is_current` says the fits differ (it ran minutes of CPU at 4.8 GB and overwrote work/<take>/counterbalance/counterbalance.json with a run on `smplx/` when
    called without the smplx dir); `recap_export` checks `is_current` first and skips the layer instead. A pod result is restored with `farm pull <id> --redo` AFTER moving `farm/chunks/<id>/pull.json` aside (a pull whose tar sha256 matches is a no-op).
- 2026-10-09 recap pose polish (T1; `farm/recap_fix.py`, `recap_front.qa_mask_camera / step_qa_mask / apply_fixed_offsets`, per-end margins + `--select-weights` in `stitch.py`, `--prev / --improvement` in `stitch_provisional.py`,
  notes `work/larissa_kadu_recap/NOTES_T1.md`): plug-in commands `farm fix-plan` (chunk jobs built from a working chunk spec: own id / take, offsets fixed from the neighbours, a manual `mask` step, then the new `qa_mask` step),
  `farm qa-refit` (a refit from the first front checkpoint of an EXISTING chunk with the qa_mask step; same id, the first results kept in `chunks/<tag>/`), `farm fix-final` (the second stitch job `final2`, writing to
  `work/<dance>_t1/` so the promoted arrays stay until the result is verified) and `farm fix-analysis` (`an2-*`, behind `final2`, the step timing on the refined fit). `qa_mask` compares per camera, dancer and frame the extent of
  the ma_2d landmarks with the extent of the teacher's 2D joints and zeroes the (camera, dancer) stretches where the median error over 1 s exceeds 0.12 (healthy 0.01-0.07, a SAM mask on the wrong person 0.3-1.0); it replaces the
  blunt "leave the camera out for BOTH dancers" (g4644 phone 06: only the follow was wrong). Gotchas found: the front checkpoint of a chunk is saved BEFORE any hand override typed on the live pod (a refit from it silently
  restores the original offsets: put overrides in the job spec, `chunk.sync_fixed`); `dancers.auto_label` needs one frame with both dancers apart in every phone of the window (phone 06 has none in 40-108 s: a window with 06 fails
  `sam:06` after 3 minutes and the watcher re-queues it twice); `launch --budget` is the ceiling of the DANCE's whole farm spend (spent so far + committed + new pods), not of the new pods; a finished watcher does not restart for
  a job queued afterwards: `launch --cpu 1 --pods 0` starts a CPU pod and the watcher. Short windows are cheap: a 12-s boundary refit (front 6 min + fit 7 min, RTX 4090) costs about $0.2-0.3, the full 55-s g0000 about $0.5 (21 min of fit
  with four phones).
- 2026-10-10 carlos_aline fits (C3; `carlos_aline_inputs.py`, `carlos_aline_people.py`, `carlos_aline_fitcheck.py`, `farm/ca_profile.py`, additive env switches in `recap_front.py` / `recap_people.py` / `recap_fitreport.py`, notes `work/carlos_aline/NOTES_C3.md`): the recap profile generalised to a
  two-camera Jack & Jill heat (`[chunks] profile = "carlos_aline"`: 4 equal windows of 51.6 s with 4 s overlap, one `cashape` job, a CPU `cafinal` stitch; every job id starts with `ca` because `queue/` and `chunks/` of the volume are shared with the recap's `g0000` / `shape`).
  `HM_FRONT_DANCE`, `HM_FRONT_CAPTURE`, `HM_SHAPE_JOB` (job env) make `dancecap.recap_front` / `recap_people` / `recap_fitreport` serve another take that supplies the same inputs (defaults = the recap). Facts learned on real pods:
  - `dancecap.sync` triangulates every camera's landmarks from at least TWO OTHER cameras: a two-camera take dies in it (`arange: cannot compute length`). The job spec carries `chunk["sync_fixed"] = {"no_measure": true, "offsets_ms": {...}}` and
    `recap_front.fixed_sync_doc` writes the offsets.json (clocks recorded, master 0); the offset itself is measured on the PC from the two dancers' YOLO joints (two-view reprojection against the shift of camera 02: minimum at 62 ms).
  - `dancers.auto_label` wants colour evidence > 1 for the label frame and its colour rule belongs to the demo's pair: `HM_TEACHER_EVIDENCE=1` makes the evidence the teacher tracks' own (constant where both are present and apart). Without it one chunk (cag4345, the embrace-heavy end) found no label frame in camera 01.
  - `min_vram_gb 24` refuses an L4 (22 GB usable): `farm up/launch --gpus 4090` (the default list starts 4090, L4 and the watcher re-launched an L4 for the refused job; two jobs were failed for good by it, `chunks retry` + `queue --only` revived one).
  - the fused grid can only start where a camera has a frame and end where the (offset) cameras end: a fit holds k 109..5937 although the take said 108..5940; `dancecap.stitch` refuses a chunk list that disagrees with what a chunk holds (window in the take TOML corrected).
  - 4 chunk jobs + the shape job + stitch on RTX 4090 pods: front 7-10 min (SAM 2.1 4.5-7.5 min, ma_2d 1-2 min per camera), shape fit 25 min (900 frames, 2 cameras = 1.65 s/frame), 1548-frame chunk fit about 50 min, refine + quality 3-4 min; per chunk 0.76-1.07 USD; stitch on a CPU pod 12 s.
- 2026-10-10 carlos_aline polish (C6; `farm/ca_fix.py`, `farm/ca_analysis.py`, `carlos_aline_audio.py`, `carlos_aline_export.py`, additive `--smplx-dir` / `--no-beat` of `dancecap timing`, `HM_POD_OWNER` env of `pods.create`; notes `work/carlos_aline/NOTES_C6.md`):
  - `ca-fix-plan` / `ca-fix-final` = recap_fix generalised: a 12-s refit chunk `cax1408` (k 1408..1768) built from the spec of `cag1488` (same inputs on the volume, pinned betas, both cameras, `HM_TEACHER_EVIDENCE=1`) and the second stitch `cafinal2`
    (writes `work/carlos_aline_t6/`; the first stitch stays until the result is verified and promoted by hand; `--select-weights arm=0.3,fingers=0.3`, margins 0.5 s for the refit). The refit ran 15 min on one 4090 (front 4.3, fit 10.3, refine 0.3 min), 0.22 USD; the stitch 30 s on a CPU pod that the watcher started by itself.
  - `ca-analysis`: timing (beat grid, refined fit, steps in the no-beat intervals dropped) -> counterbalance -> floorcraft, and physics, as CPU-pod jobs behind `cafinal2`. The 194-s timing with 8 Monte-Carlo workers is SIGKILLed (rc -9) on a 16 GB pod after 19 min:
    it needs the 64 GB `analyse_mem` pod (`ca_analysis.launch_mem_pod()`; the job spec is `kind analyse_mem`).
  - Facts: a job id whose volume dir holds a stale `claim.lock` is NOT re-uploaded by `queue` ("a worker holds a claim"): after killing a pod move `chunks/<id>` aside on the volume, `chunks retry <id>`, then queue; the watcher exits ("all jobs terminal") and deletes its pods when
    the only job is failed, so queue first, start pods after. `HM_POD_OWNER=C6` in the environment of every launching command (also the watcher's) labels the ledger rows.
- 2026-10-11 J3b (john_lula: ONE handheld camera, single-camera 3D fit), additions:
  - `farm/jl_profile.py` (`[chunks] profile = "john_lula"`): ids `jl*` (jlshape, jlg0000, jlg1322, jlfinal), spec field `"owner": "J3b"`. **Queue isolation**: `worker.claimable` takes a spec only when its `owner` equals the worker's env `HM_OWNER` (both
    empty = as before); `pods.create` forwards `HM_QUEUE_OWNER` of the launching process as the pod's `HM_OWNER`. Run every farm command of a session with `HM_OWNER=<name> HM_QUEUE_OWNER=<name>` and put the same name in the specs.
  - `dancecap.jl_front` (front DAG without SAM: frames, cameras (JL_FOCAL_SCALE focal-hypothesis variant), masks from J3a's people/masks_01.npz, ma_2d, fixed sync, prepare, mono-init), `dancecap.jl_mono` (pelvis start from the ankle rays), `mamma-turing/optimization/utils/mono.py`
    (`MAMMA_MONO=1`: no triangulation / closest-point start, heading search + Viterbi, `floor_loss`, `stature_loss` via `MAMMA_STATURE`; `mamma_dyn.MA3D_MONO_OVERRIDES`), `dancecap.mono_synth` (ground-truth check of the single-camera optimiser), `jl_fitcheck`, `jl_depth`, `jl_repair`,
    `farm/jl_fix.py` (`jl-fix-plan` / `jl-fix-final`: the boundary refit + second stitch). Facts: the front of a 46 s window takes 1-2 min (no SAM); the mono fit 20 min per 900-1380 frames on a 4090 (fourth run 5 min of that); `stitch` with `--partial --force` writes the promoted arrays.
  - A variant job (`what: variant`, e.g. the focal re-solve) must not list `mamma_pin_betas.json` as a required output (fixed in `jl_profile.build_job`); `farm up --only <id>` with the watcher running queues the job but the watcher launches its pod only when one of its `--pods` slots frees.
- 2026-10-11 J4 (john_lula analyses; `farm/jl_analysis.py`, `jl_lowconf.py`, `jl_export.py`, notes `work/john_lula/NOTES_J4.md`):
  - `jl-analysis`: ca-analysis for a take whose stitched motion is on THIS machine: the inputs are the PROMOTED arrays (`smplx/`, `smplx_rf2_noslip/`, 43 MB, one archive `src/<capture>/jl_an_in.tar` uploaded by `queue`) + the beat grid, no `after`, spec `"owner": "J4"` (queue isolation of J3b).
    Jobs `jlan-timing` (timing on the refined fit with the heat's beat grid, steps in the no-beat intervals dropped, ONE camera: `--n-cams 1 --sync-ms 15`, then counterbalance, floorcraft) and `jlan-physics` (raw fit), both kind `analyse_mem`, claimed only by a pod started with
    `jl_analysis.launch_mem_pod()` (`HM_QUEUE_OWNER=J4`; the 64 GB `cpu5m`, $0.52/h). Two such pods (one per job), then `watch start --pods 0 --no-launch` (pulls, deletes the pods): physics 3 min, timing + counterbalance + floorcraft 9 min on an 88-s take; total pod spend under 0.30 USD.
  - Gotcha: a pull whose pods the watcher still holds in state from an earlier session (exited / deleted by hand) makes the watcher retry `delete-pod` four times a minute: harmless, ignore.
  - Gotcha: `counterbalance.is_current` / `floorcraft.is_current` hash `timing/timing.json` and `counterbalance.json`: annotating them (`python -m dancecap.jl_lowconf annotate`) re-stamps the recorded hashes (`source.restamped`), so annotate BEFORE the export.
