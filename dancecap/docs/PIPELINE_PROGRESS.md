# Pipeline progress, ETA and budget (`dancecap.progress/1`)

> **Hands stages retired (2026-10-07, user decision):** `hands_survey`, `hands_pod`, `hands_build` and the `S3_hands`
> session were removed from the pipeline. The hands rows below (job names, adapters, measured rates, rollout item 5)
> are historical. The `[hand_worker]/[mano_worker]` adapter stays because the same parser also reads face_worker lines.

> **Implementation status (2026-10-07, stage "progress instrumentation"):** L0, L1 and most of L2 are implemented:
> `dancecap/progress.py` (helper, adapter table, L0 `parse_runner_log`, readers, CLI `tail | eta [--at] | watch |
> budget | rates --rebuild`), `dancecap/progress_eta.py` (history `rates.json`, ETA sections 6.2-6.7, session
> `snapshot()` and live `Tracker`, `budget()`, `launch_gate()`, `estimate_session()`, camera-type workload
> `loc_units()/loc_estimate()/estimate_front()` and QA `camera_type_flags()`), `dancecap/progress_pull.py` (puller +
> read-only PROBE), additive `remote_jobs.py` hooks (section 10.2: all items), L2 hooks in `avatar_texture`,
> `mamma_chunks` (plan, `watch_log` per lane, gate note, fallback plan), `room` (train), `front_pod.Runner`,
> `dancers_sam` (frames), `cameras/localize` (frames), `cameras/pipeline` (`camera_motion` metric = probe B) and
> `detect`. Review fixes (same day, section 14): robust live rate (pauses dropped), stall-aware remaining time and
> warnings (heartbeats are not progress), camera types read through `dancecap.camera_types` (what the stages apply),
> the QA metric from probe B (unconstrained track), and the monitor's live path `live_snapshot()` (one
> background-polled `Tracker` per watched session).
> Not done: `bootstrap.sh` JSON lines (its `[bootstrap]` lines already stream into runner.log, so L0 covers it),
> hands workers (run as standalone scripts; their `[hand_worker]/[mano_worker]` lines are covered by the L0
> adapter), `tqdm_bridge` (ma_2d is covered by the puller's tqdm tail), the optional legacy probe of 4.4 and the
> billing reconciliation CLI of 8.4. Tests: `tests/test_progress.py`.
>
> Design for the pipeline monitor, stage "progress" (2026-10-07). This doc
> is the contract that the monitor, the Runpod runner (`remote_jobs.py`) and the stage code implement. The
> measured numbers come from the 32 sessions in `work/runpod/sessions/` (2026-10-06..07), from
> `work/runpod/ledger.json` (36 rows), from `work/larissa_kadu_38/front/front_pod.json` and from the MAMMA logs in
> `work/larissa_kadu_38/mamma/logs/`. Compute policy (user): heavy CV/neural work runs on Runpod. The local PC
> (4 logical CPUs, a shared 2080 Ti) handles orchestration, the UI and light checks.

## 0. Decisions on one screen

1. **One event format.** `dancecap.progress/1` is a stream of JSON lines `{v, ts, job, step, kind, done, total,
   unit, msg, ...}`. A stdlib-only helper, `dancecap.progress`, writes it. It runs on a pod before bootstrap, in
   `.venv-nopo4d`, and in a frozen exe, and it needs no new dependencies or pins.
2. **Producers write files and nothing serves them.** On the pod, all jobs of a session append to one file,
   `<pod ws>/dancecap/work/runpod/sessions/<sid>/progress.jsonl`. Locally, the runner writes its own phases to
   `work/runpod/sessions/<sid>/progress/runner.jsonl` and keeps a mirror of the pod file in
   `.../progress/pod.jsonl`. Local long stages write `work/<take>/progress/<stage>.jsonl`.
3. **The runner pulls, and the monitor does not.** A daemon thread in the runner runs a read-only probe every 15 s
   over the ssh the pod already exposes (mapped port 22, the existing key `~/.ssh/runpod_dancecap`). Each probe
   returns:
   - the new bytes of the pod progress file since the last offset,
   - the size of the upload's `.part` file,
   - the kill-switch deadline,
   - `nvidia-smi` and the load average,
   - the tails of a few third-party logs that we do not instrument (MAMMA `Iter. N of M`, tqdm).

   This adds no new ports, no pod daemon and no pod state.
4. **The monitor only reads.** It opens local files, reads them and closes them at once, and it never writes
   inside a session dir. It reads the ledger. It polls `runpodctl pod list` every 30 s and `runpodctl user` every
   5 min, keeping only the balance, spend and limit fields. It never creates, stops or deletes a pod unless the
   user clicks and confirms.
5. **Three levels, so that history and running sessions stay covered:**
   - **L0** parses what already exists (`runner.log`, `state.json`, `front_pod.json`, the ledger) with regex
     adapters and needs no code change. It covers every past session and any runner started on the current code.
   - **L1** adds the puller and the runner events through a small additive `remote_jobs.py` patch (section 10).
   - **L2** instruments the stage code with the helper.
6. **ETA** is the live rate blended with the historical rate in log space, weighted by the fraction done.
   - Non-linear steps (gsplat densification, MAMMA optimisation runs) scale a historical time profile.
   - Parallel lanes take the slowest lane. The front-end task DAG is simulated.
   - Branches are shown as two ETAs, for example the MAMMA merge gate: pass, or fall back to a full fit.
   - Every ETA carries a p10-p90 band and is checked against the pod's kill switch.
7. **Budget:**
   - **Spent** is `remote.spent()`: the ledger, with open rows accruing at their price.
   - **Projected** adds price x ETA (p50 and p90) for running pods, plus the estimates of queued sessions.
   - **Hard ceiling:** the kill switches bound every running pod.
   - **Warnings** fire at a reserve ($2 by default) and at the cap.
   - **The monitor's launch gate** is stricter than `pod_session`'s, because it also counts the future cost of
     pods that are already running.
8. **Camera types are workload inputs.** Mount is `handheld|tripod`, zoom is `yes|no`, with values from the review
   UI or an automatic suggestion.
   - Tripod: one pose per video, so localisation runs on keyframes only.
   - No zoom: one fixed focal per video.
   - While a type is only an unconfirmed suggestion, the estimator prices it for p50 and uses the conservative
     `handheld+zoom` for p90.
   - Live camera metrics are checked against the declared type (section 9).

## 1. What exists today (read-only inventory)

| signal | where | live? | content |
|---|---|---|---|
| `runner.log` | local session dir | **yes, line by line** | `Pod.run(log=...)` copies every stdout line the pod prints over the runner's ssh, plus `[pod-runner HH:MM:SS] <step> ...` / `<step> done in N s` / `FAILED`. Python's universal newlines turn tqdm `\r` updates into separate lines. |
| `state.json` | local session dir | after each step | `timings[] {step, seconds, ok, start (epoch), mb, features, job}`, `jobs{name: {status, commands{cmd: rc}, seconds, spec, fingerprint}}`, `bundle{bytes, raw_bytes, n_files}`, `estimate_min`, `max_minutes`, `pod{id, gpu, cost_per_hr, host, killswitch{armed, after_min}, started}`. Written with `write_text`, which truncates in place: a reader can see a partial file. |
| `result.json` | local session dir | at the end | Same summary, plus ledger cost. A resumed session (same sid) keeps the previous run's `result.json` until it ends: stale while running (seen on `20261007-065744-batch`). |
| `pod_logs/<job>/<cmd>.{log,rc,done}` | local, after download | no | Copies of the pod logs, which arrive with pack/download. |
| `bootstrap_timings.tsv`, `bootstrap_check.json` | local, after bootstrap | no | Per feature: seconds and rc. |
| MAMMA chunk stdout `mamma/logs/chunk_<tag>.out`, `ma_3d_<tag>.log` | **pod only** while running | no | `Iter. N of M` for 4 optimisation runs (50 / 500 / 100 / 25 iterations, with early stop), **no timestamps**; `Stopping early at iteration N`; `Optimization took S seconds`. |
| `mamma/logs/ma_2d_<cam>.log` | pod only | no | tqdm `k/N [mm:ss<mm:ss, x it/s]`. |
| `front/front_pod.json`, `front/logs/<task>.log` | pod only | no | Per task: state, rc, seconds, start (atomic `os.replace` on Linux). Loc logs: `[cameras] 02: 60/1151 frames, 60 ok (41s)`. SAM logs only report at the end. |
| `/workspace/.dancecap_kill_at` | pod | n/a | Epoch of the on-pod kill switch (`remote.arm_killswitch`). |
| `ledger.json` | local | yes | Per pod: `cost_per_hr`, `created_ts`, `deleted_ts`, `cost`, `max_minutes`. A row without `deleted_ts` is a running pod (or a crash). |
| `runpodctl pod list/get -o json` | API, key from the existing file | yes | `desiredStatus`, `runtimeStatus` (running / initializing / stopped / terminated / unknown), `costPerHr`, `lastStartedAt`, ports. No GPU utilisation in REST v1. |
| `runpodctl user` | API | yes | `clientBalance`, `currentSpendPerHr`, `spendLimit`. It also returns `email`, `id` and notify flags, which are **never** read out (`remote.status()` already filters to balance/spend/limit keys). |

**Gaps that the protocol closes:**
- (a) Output of parallel or background subprocesses (MAMMA chunks, front tasks) never reaches `runner.log`.
- (b) Several steps announce no total: texture pass count, gsplat total steps, SAM frames, MAMMA iterations after
  early stop.
- (c) Upload and download have no byte progress (`scp -q`).
- (d) A step's start epoch is only stored when the step ends.
- (e) Nothing machine-readable carries an ETA.
- (f) The ssh port is not stored in `state.json`. Only the optional legacy probe in section 4.4 needs it.

### 1.1 Measured history (seeds for the estimator)

| quantity | measured | n | note |
|---|---|---|---|
| pod create -> ssh ready + kill switch armed | 16-48 s, median ~34 s | 21 | `state.pod.started - ledger.created_ts` |
| upload | median ~2.5 MB/s, p90-slow **0.7 MB/s**, worst 0.19 MB/s | 28 | 4191 MB front bundle took **77 min** once (0.91 MB/s) |
| download | 1.5-16 MB/s; >300 MB archives 11-16 MB/s | 17 | small archives are latency-bound |
| bootstrap `base` | 68-122 s typical, 369-418 s on slow PyPI | 21 | one failure at 614 s |
| bootstrap features | mamma 23 s (cached wheels) / 218 s cold; dancers 90-311 s; cameras 10-230 s; ma2d 30-67 s; room 19-30 s with wheels | | whole bootstrap 74-1214 s (2026-10-07: 20 min PyPI stall, $0.40) |
| MAMMA chunk fit | 7 lanes (6 chunks of 232 frames + held-out) under CUDA MPS on A100 PCIe: 2261-2831 s per lane | 7 | iterations actually run: 16/50, 270/500, 100/100, 17/25 = 403 of 675 |
| MAMMA merge / gate fallback / select | 222 s / 3182 s (gate FAILED -> full fit of 1148 frames; it printed "~66 min", took 53 min) / 5 s | 1 | |
| texture bake (pod preset, 2292 views x 3 passes, A100-SXM4) | pass 0 145 s (0.063 s/view), aligned passes 315 / 303 s (0.137 / 0.132 s/view), rest 318 s; total 1081 s | 1 | the prior (`avatar_pod.POD_S_PER_VIEW` 0.061 / 0.114 + fixed 2 + 0.012 min/frame) predicted ≈ 27 min; actual 18.0 |
| room gsplat (pod, 30000 steps, 1 M cap) | 1944 s; first 1000 steps 38 s, last 1000 ~68 s | 1 | **non-linear**: rate falls as the Gaussians grow |
| front (38 s take, 8 cams) | sam 340-862 s/cam (0.32 s/frame on cam02, 2302 frames); loc ~0.56-0.8 s/frame; ma2d 147-614 s/cam (0.13-0.27 s/frame); cameras 1171 s; sync 1567 s (CPU); job 4374 s | 1 | per task in `front_pod.json` |
| hands | `[hand_worker] hands 500/5779` every ~42 s; hamer 5779 in ~140 s | several | A40 |
| detect (pod) | 1151 frames, 85 ms/frame | 1 | |

## 2. The protocol: `dancecap.progress/1`

### 2.1 Event (one JSON object per line, UTF-8, `\n`-terminated, at most 1024 bytes)

| field | type | required | meaning |
|---|---|---|---|
| `v` | int | yes | `1` |
| `ts` | float | yes | Writer's epoch seconds (UTC). The pod clock for pod events. The puller adds `rx` (local receive epoch) and `skew`. |
| `job` | str | yes | Pod job name (`mamma`, `avatar`, `front`, `room`, `hands_build`, ...) or local stage id (`ingest`, `cameras`, ...). On the pod it comes from `DANCECAP_PROGRESS_JOB`. |
| `step` | str | yes | Path within the job, `/`-separated, stable across runs (used as the history key): `fit_chunks/c8_f0-232/run2`, `bake/pass1`, `front/loc:02`, `upload`. |
| `kind` | str | yes | `plan` · `start` · `tick` · `end` · `fail` · `note` · `metric` · `heartbeat` |
| `done`, `total` | number / null | tick, end | Units completed / expected. `total: null` means indeterminate. A later event may revise `total` (early stop). |
| `unit` | str | with done | `frames` · `views` · `iters` · `steps` · `bytes` · `tasks` · `chunks` · `cams` · `features` |
| `msg` | str | no | At most 200 chars, human-readable. **Plain text: the monitor renders it as text, never as HTML.** |
| `rc` | int / str | end, fail | Exit code or error token. |
| `steps` | list | plan | `[{step, total, unit, weight?, after?, lane?}]`: what the job will do, so the ETA knows the remaining steps. |
| `lane` | str | no | Parallel lane id (MAMMA chunk tag, front task). Steps in different lanes run concurrently. |
| `eta_s` | float | no | The producer's own estimate (for example tqdm's remaining time, or mamma_chunks' "~66 min"). |
| `extra` | object | no | Small metrics (loss, PSNR, Gaussians, reprojection px, camera spreads). |
| `sid`, `take`, `pod`, `pid`, `host` | | no | Filled by the helper from the environment or by the puller. |

**Kinds:**
- `plan`: once per job, or again when the plan changes (the gate falls back to a full fit).
- `start` / `end` / `fail`: per step.
- `tick`: throttled progress.
- `note`: a warning or decision (`gate FAILED: follow seam p95 83 mm > 60`).
- `metric`: a QA number (section 9).
- `heartbeat`: sent every 30 s while a step is open and nothing else was written, so a silent but alive step can
  be told apart from a dead one.

**Rules:**
- **Throttle:** a tick goes out only if at least 2 s **and** at least 0.5 % of the total have passed since the
  previous one. `start`, `end`, `fail`, `plan` and `note` are always written. A 30 000-step gsplat run therefore
  writes about 200 ticks, not 30 000.
- **Atomic appends:** writers open the file `O_APPEND` and write one line in one `write()` call. On Linux local
  filesystems, appends under 4096 bytes (`PIPE_BUF`) from concurrent processes do not interleave, so parallel
  MAMMA chunks and front workers can share the single pod file. On Windows, where concurrent appends are not
  guaranteed, every local process writes its own file `work/<take>/progress/<stage>.<pid>.jsonl`.
- **Never fatal:** every helper call is wrapped. A full disk or a missing directory loses an event, never a stage.
- **No secrets:** events never carry environment variables, command lines with tokens, or anything read from
  `/proc/*/environ` (which holds the pod-scoped `RUNPOD_API_KEY`). The helper drops `msg` values that look like
  credentials (a run of 32 or more `[A-Za-z0-9_-]` characters) before writing.
- **Step ids are stable:** history is keyed by `job/step-kind`. Instance parts (chunk tag, camera id) go in `lane`
  or in the last path segment. The history key strips them with a per-job rule (section 7).
- **Pod clock:** pods run NTP. The puller measures `skew = rx - rtt/2 - pod_now` on every probe and stores it. The
  monitor uses `ts + skew` whenever it compares pod times with local time (staleness, kill time).

### 2.2 Examples

```json
{"v":1,"ts":1791392937.1,"job":"avatar","step":"bake","kind":"plan","steps":[{"step":"bake/load","total":null,"unit":"views"},{"step":"bake/pass0","total":2292,"unit":"views","weight":0.19},{"step":"bake/pass1","total":2292,"unit":"views","weight":0.41},{"step":"bake/pass2","total":2292,"unit":"views","weight":0.40},{"step":"bake/finish","total":null,"unit":null}]}
{"v":1,"ts":1791393600.4,"job":"avatar","step":"bake/pass2","kind":"tick","done":1450,"total":2292,"unit":"views"}
{"v":1,"ts":1791384480.0,"job":"mamma","step":"fit_chunks/run2","lane":"c8_f0-232","kind":"tick","done":187,"total":500,"unit":"iters"}
{"v":1,"ts":1791386200.0,"job":"mamma","step":"fit_chunks/run2","lane":"c8_f0-232","kind":"end","done":270,"total":270,"unit":"iters","msg":"Stopping early at iteration 270"}
{"v":1,"ts":1791387531.6,"job":"mamma","step":"gate","kind":"note","msg":"gate FAILED: follow: seam p95 83 mm > 60","extra":{"branch":"fallback","eta_s":3960}}
{"v":1,"ts":1791392300.0,"job":"_session","step":"upload","kind":"tick","done":402653184,"total":1521600000,"unit":"bytes"}
{"v":1,"ts":1791384100.0,"job":"front","step":"front/loc:02","lane":"loc:02:0","kind":"tick","done":180,"total":1151,"unit":"frames","extra":{"ok":180}}
```

### 2.3 Files

| path | writer | reader |
|---|---|---|
| pod: `/workspace/HeadMovement/dancecap/work/runpod/sessions/<sid>/progress.jsonl` | every pod job command (the helper, through `DANCECAP_PROGRESS`) | the probe (read-only) |
| `work/runpod/sessions/<sid>/progress/runner.jsonl` | runner (`Session.event`): bundle_scan/build, pod start, upload bytes, extract, bootstrap, per command start/end, pack, download bytes, extract, delete | monitor |
| `work/runpod/sessions/<sid>/progress/pod.jsonl` | puller: the pod file mirrored line by line, plus `pod`, `rx`, `skew` | monitor |
| `work/runpod/sessions/<sid>/progress/puller.json` | puller, atomically, every tick: `{pod, offset, last_ok, last_err, rtt_s, skew_s, kill_at, gpu[], load, upload_part_bytes, tails{}}` | monitor |
| `work/runpod/sessions/<sid>/runner.json` | runner at start: `{pid, started, attempt, label, jobs}` (no argv, no environment) | monitor (is the runner alive?) |
| `work/<take>/progress/<stage>[.<pid>].jsonl` | local stages (the helper) | monitor |
| `work/progress/rates.json` | `python -m dancecap.progress rates --rebuild` (runner at session end, or the monitor on demand): derived, rebuildable | estimator |

Pack and download never pick up the pod progress file: `pack_command` tars only job outputs and `log_rel`. A
resumed session (same sid, new pod) starts a fresh pod file. The puller keeps one offset per pod id, so the local
mirror simply continues.

## 3. Producers

### 3.1 Helper API (`dancecap/dancecap/progress.py`, stdlib only, a no-op when `DANCECAP_PROGRESS` is unset)

```python
from dancecap import progress as P

P.plan([("bake/pass0", n, "views"), ("bake/pass1", n, "views"), ("bake/pass2", n, "views")])
with P.step("bake/pass1", total=n_views, unit="views") as s:      # start ... end / fail (exception -> fail, re-raised)
    for i, v in enumerate(views):
        ...
        s.update(i + 1)                                              # throttled tick; s.update(i + 1, total=new) revises
P.note("gate FAILED: ...", level="warn", branch="fallback", eta_s=3960)
P.metric("camera_motion", lane="05", source="probe_b", centre_p95_m=0.004, focal_ratio=1.0, mount="tripod")
stop = P.watch_log(path, adapter="mamma_iter", step="fit_chunks", lane=tag)   # thread tails a 3rd-party log
P.tqdm_bridge(step="ma2d", lane=cam)                                 # optional: tqdm callback -> ticks
```

**Environment** (set by the runner for every pod command, by the pipeline launcher for local stages):
- `DANCECAP_PROGRESS` is the file path.
- `DANCECAP_PROGRESS_JOB` and `DANCECAP_PROGRESS_CMD` name the job and command.
- `DANCECAP_SID` is the session id.
- `DANCECAP_PROGRESS_ECHO=1` also prints a short human line to stdout.

Existing `print` lines stay as they are: humans and the L0 parser still read them.

### 3.2 Adapters: one regex table shared by three consumers

The same adapters are used by the L0 parser of `runner.log`, by `watch_log` on the pod, and by the probe's log
tails. Each one is tested against fixtures cut from the real logs (section 11). These logs hold no media.

| adapter | pattern (Python `re`) | event |
|---|---|---|
| `runner` | `^\[pod-runner (\d\d:\d\d:\d\d)\] (\S+) \.\.\.$` / `^\[pod-runner (\S+)\] (\S+) (done\|FAILED) in ([\d.]+)s` | start / end of a runner step (local clock; date from the sid or mtime) |
| `bootstrap` | `^\[bootstrap\] (\w+): start` / `^\[bootstrap\] (\w+): done in (\d+)s` / `finished rc=(\d+)` | `bootstrap/<feature>` start/end, unit `features` |
| `texture` | `^\[texture\] (\d+)/(\d+) views` / `^\[texture\] pass (\d+)(?: \(aligned\))?: (\d+) s` | tick on `bake/pass<k>`; k advances at each `pass k:` line; pass count = 1 + `preset_config.align_passes` from `avatar/pod_job.json` |
| `room` | `^\[room\] step\s+(\d+)\s+L1 ([\d.]+)\s+masked PSNR ([\d.]+)\s+N ([\d,]+)\s+(\d+)s` | tick `train`, total = `iters` of the preset (pod 30000), extra psnr / gaussians |
| `hands` | `^\[(hand_worker\|mano_worker)\] (\w+) (\d+)/(\d+) (\d+)s` | tick `<worker>/<stage>` frames |
| `front` | `^\[front\] (\d\d:\d\d:\d\d) start (\S+)` / `^\[front\] \S+ (\S+): ok rc=(\d+) in (\d+)s` / `^\[front\] (\S+): done earlier` | `front/<task>` start/end; tasks done / total from `front_pod.json` |
| `loc` | `^\[cameras\] (\w+): (\d+)/(\d+) frames, (\d+) ok \((\d+)s\)` | tick `front/loc:<cam>` frames |
| `chunks` | `^\[chunks\] (\d+) chunks of (\S+)(?: \+ held-out (\S+))?, (\d+) in parallel` / `^\[chunks\] (\S+): rc=(\d+) in (\d+)s` / `^\[chunks\] gate (PASSED\|FAILED)` / `fallback: one full fit of \S+ \((\d+) frames, ~(\d+) min\)` | plan (lanes), lane end, gate note, fallback plan with `eta_s` |
| `mamma_iter` | `^Iter\. (\d+) of (\d+)` / `^Stopping early at iteration (\d+)` / `^Optimization took ([\d.]+) seconds` | tick `fit_chunks/run<r>` (r advances when `Iter. 1 of` appears; totals are 50, 500, 100, 25 by config), early stop sets total := done |
| `tqdm` | `(\d+)/(\d+) \[(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?)` (last match in the line) | tick with `eta_s` from tqdm's remaining |
| `detect` | `^\[detect\] (\w+): (\d+) frames .* (\d+) ms/frame` | end per camera (in-progress ticks need L2) |

### 3.3 Mapping per job (L0 = today's output, L2 = instrumented)

| job (where) | steps and units | L0 signal | L2 instrumentation (stdlib helper) |
|---|---|---|---|
| session (local + pod) | `bundle_scan`, `bundle_build` (bytes), `pod_start`, `upload` (bytes), `extract`, `bootstrap/<feature>`, `<job>/<cmd>`, `<job>/pack`, `<job>/download` (bytes), `<job>/extract`, `delete` | `runner.log` lines plus `state.json` timings; upload is estimated from elapsed time x historical MB/s | runner events (L1); upload bytes from the `.part` size in the probe; download bytes from the local `downloads/<job>.tar.gz.part` size vs `pack_bytes` |
| `mamma` chunked (pod) | plan: K chunk lanes plus a held-out lane; per lane `run1..run4` iters (expected total from history, section 6.4); then `merge`, `gate` (branch), optional `fallback_fit` (4 runs, 1 lane), `select` | `[chunks]` lane ends only (no in-lane progress: chunk stdout goes to files on the pod) | `fit_chunks` starts `watch_log(ma_3d_<tag>.log, "mamma_iter", lane=tag)` per chunk (mamma-turing stays untouched); `gate_or_fallback` emits the branch note plus a new plan |
| `mamma` single fit (pod / local) | `run1..run4` iters | none | same `watch_log` in `mamma_dyn`'s ma_3d launcher |
| `avatar` texture (pod) | `bake/load`, `bake/head_fix`, `bake/hair_model`, `bake/pass0..k` views, `bake/finish` | `[texture] N/M views` every 50 views; pass count from the job's preset | plan at start; `Step` around each pass and the pre/post phases (today about 318 s of the 1081 s run is outside the passes and invisible) |
| `room` gsplat (pod) | `train` steps (30000), `quest` steps (5000), `post` (mesh, eval) | `[room] step N` every 1000 | `Step` in the training loop; profile-scaled ETA (section 6.3) |
| `dancer_splats` (pod) | `splats/train` steps over the S1-S3 schedule (25000; the total shrinks when the time budget / kill switch demands it: a new `plan` event), `splats/eval` views | `[dancer_splats] step N/M stage S psnr X iou Y N_gauss Z Ns` every 250 steps, `[dancer_splats] eval N/M views` (adapter `splats`) | `tick` every step (read back every 25 steps: no per-step GPU sync) with stage, psnr, iou, gaussians; linear ETA from the history key `dancer_splats/train/s_per_step` (seed 0.06 s/step, A100 estimate) |
| `front` (pod) | DAG of tasks `label:<c>`, `sam:<c>` frames, `loc:<c>[:<a>]` frames, `sub:<c>`, `ma2d:<c>` frames, `merge`, `cameras`, `sync`, `chain`, `prepare` | `[front]` start/ok in `runner.log`; per-task state in the pod's `front_pod.json` (probe tail) | `Runner._start/_finish` emit task start/end (workers inherit `Runner.env`, so `DANCECAP_PROGRESS` reaches them); `dancers_sam` propagation loop ticks frames; ma_2d via `tqdm_bridge` or `watch_log(..., "tqdm")`; loc's existing per-60-frame line becomes a `Step` |
| `hands`, `hands_build` (pod) | `<worker>/<stage>` frames | `[hand_worker] hands N/M Ns` | `Step` around the existing loops |
| `detect` (pod / local) | per camera frames | end line only | `Step` per camera |
| bootstrap (pod, bash) | `bootstrap/<feature>` | `[bootstrap] f: start / done in Ns` | the same lines, plus one `printf '{"v":1,...}\n' >> "$DANCECAP_PROGRESS"` per feature when the variable is set (an optional, additive bootstrap.sh change) |
| local stages (`ingest`, `cameras map`, `sync`, `beats`, `song`, `physics`, `export`, `demo_export`) | stage-specific frames / images / cams | stdout only | `Step` per camera / stage; the pipeline launcher sets `DANCECAP_PROGRESS=work/<take>/progress/<stage>.<pid>.jsonl` |

## 4. Transport: the runner pulls over the existing ssh

### 4.1 Puller (`dancecap/dancecap/progress_pull.py`, local, started by the runner)

- **Lifecycle.** One daemon thread per session. It starts as soon as `pod_session` yields the pod, before the
  upload, and stops in a `finally` after the last download.
- **Isolation.** Any exception inside the puller is caught and written to `puller.json.last_err`. It never
  propagates, so the session cannot fail because of progress.
- **Transport.** Every `interval` seconds (default 15; 60 after 3 consecutive failures; reset on success) the
  thread runs one ssh command with the pod's own options (`Pod._ssh_base()`: same key, same port, `LogLevel=ERROR`,
  plus `-o ConnectTimeout=10`):

  ```
  ssh <base> "python3 - --progress <pod progress file> --offset <n> --max-bytes 262144
              --upload-part /workspace/upload/<bundle>.part --tails '<json list>'" < PROBE
  ```

  `PROBE` is a stdlib Python script (under 8 KB) sent on **stdin**, so it works before the bundle is extracted.
  The image always has `python3`. If `python3` is missing, a bash one-liner fallback (`stat`, `tail -c +N`,
  `nvidia-smi`, `cat`) returns the same fields.
- **What the probe does.** It is read-only and prints one ASCII-only JSON line (`ensure_ascii`, so the Windows
  cp1252 console problem in `remote._echo` cannot happen):
  `{"now": pod_epoch, "size": n, "data": "<new bytes since offset, complete lines only>", "kill_at": int|null,
  "upload_part": bytes|null, "gpu": [{"util": %, "mem_used": MiB, "mem_total": MiB}], "load1": x,
  "tails": {"<path>": {"size": n, "mtime": t, "lines": ["..last matching lines.."]}}, "front": {...front_pod.json
  task states...}}`.
  - Allowed operations: `stat`, read, `nvidia-smi --query-gpu`, `/proc/loadavg`.
  - Forbidden: writes, process control, and `/proc/*/environ` (the pod-scoped key lives there).
- **What the puller writes.** It appends the received lines to `progress/pod.jsonl`, adding `pod`, `rx` and
  `skew`, and holding back an incomplete last line until the next tick. Tail matches go through the adapters
  (section 3.2) and are written as events with `"src": "tail"`. Finally it rewrites `puller.json` (tmp file, then
  `os.replace`, retried on `PermissionError`).
- **Tails list.** This is the L1 bridge for uninstrumented tools. It comes from an optional `progress` key in the
  pod job (`{"tails": [{"glob": "...", "adapter": "...", "lane_from": "regex"}]}`) or from per-job defaults:
  - `mamma`: `dancecap/work/<take>/mamma/logs/ma_3d_*.log` with `mamma_iter`, the lane from the file name;
  - `front`: `front/front_pod.json` (parsed) and `front/logs/{loc,ma2d}_*.log`.

  The probe reads only the last 16 KB of each matched file.
- **Why poll instead of keeping one long-lived ssh.** A fresh connection per tick needs no reconnect logic, cannot
  hang the runner, and costs about 1-2 s of ssh handshake on this PC every 15 s. That is negligible next to the
  runner's own streaming ssh, which it leaves alone. Windows OpenSSH has no usable `ControlMaster`, so a
  multiplexed design is not an option.
- **Ports.** Only the pod's mapped port 22, which `pod_session` already requests (`--ports 22/tcp`).

### 4.2 Sessions without a puller (L0: history, and runners started on today's code)

The monitor parses `runner.log` from a stored byte offset with the adapters in section 3.2, and reads
`state.json`, `front_pod.json` (once downloaded) and the ledger.

| what is visible for an L0 session | what is not |
|---|---|
| Runner phases with start times; texture views; room steps; hands frames; `[front]` task start/end; `[chunks]` lane ends; bootstrap features | In-lane MAMMA iterations; SAM/ma_2d frames; upload bytes (estimated from elapsed time x historical MB/s, flagged "estimated") |

### 4.3 What the runner records for the monitor (L1, additive)

- `state.pod.port`.
- `state.pod.kill_at`: the first probe's `kill_at`. Until then the estimate is `started + killswitch.after_min x 60`.
- `jobs[j].current` (running command).
- `jobs[j].pack_bytes` (set before the download, so the local `.part` file gives a fraction).
- `runner.json`, so the monitor can check that the runner PID is alive.

### 4.4 Optional legacy probe from the monitor (user toggle, read-only)

An L0 session that runs a long MAMMA fit is blind for an hour. For that case the monitor may run **the same read-only
probe** itself, every 30 s or more, but only while the user has switched on "Live probe" for that session.
- It needs the port. The monitor gets it from `runpodctl pod get <id>` (port mapping for 22), because L0 state does
  not store it.
- It uses a separate known-hosts file, `work/monitor/known_hosts`, so it never contends with the runner's.
- Anything beyond the probe is refused, and the probe script is fixed and allowlisted.
- It runs in its own ssh connection, so it cannot disturb the runner's.

## 5. How the monitor reads without disturbing a running session

**Entry point.** The monitor backend (`pipeline/runpod_view.py`) calls `progress_eta.live_snapshot(session_dir)` for
a running session: one `Tracker` per watched session, kept by a daemon thread that polls it every 3 s (open, read
from the stored offset, close) while somebody asked for that session in the last 30 min, dropped 2 min after the
session ended. New `runner.log` lines get their receive time: the file's mtime when the poll follows the previous one
within 10 s, else spread over (previous poll, mtime] by line number. Lines already in the file at the first poll
are timed as `snapshot()` does (line interpolation between clocked lines; an unfinished bake pass re-timed from the
command start, section 6.2). Ended sessions use plain `snapshot()`. `budget()` reuses a live tracker when one exists.

1. **Discovery.** List `work/runpod/sessions/*/` by mtime. A session is **active** when any of these holds:
   - the ledger has an open row for `state.pod.id`;
   - `runner.log` changed in the last 120 s and its last runner line is not terminal;
   - (L1) `runner.json.pid` is alive.

   `result.json` alone is not proof of an ended session (a resumed sid keeps the old one).
2. **Open, read, close.** Never keep a handle open. On Windows, Python opens files without `FILE_SHARE_DELETE`, so
   a held handle would make any writer's `os.replace` on that file fail with WinError 5/32. Today the runner writes
   `state.json` and the ledger with `write_text` (truncate in place), which a reader cannot break. **Keep it that
   way:** do not switch the runner to `os.replace` without a retry loop.
3. **Partial JSON** (a read during a truncate-write): retry 3 times at 100 ms, then keep the last good snapshot and
   mark it "stale by N s". For JSONL, read from the stored offset and drop an incomplete last line.
4. **Cadence:**
   - local files: every 2 s, `stat` first and read only on a size or mtime change;
   - `runpodctl pod list`: every 30 s, only while a session is active or the Pods panel is open;
   - `runpodctl user`: every 5 min;
   - GPU price list (`remote.gpu_table`): every 10 min, only in the launch dialog.

   API calls go through `remote._ctl`, so the key comes only from `RUNPOD_API_KEY` or the existing key file, in the
   child's environment. It is never shown, logged, put in a URL (the REST and GraphQL APIs take it as a Bearer
   header), or frozen into an exe. Error text from `runpodctl` is scrubbed of token-like runs before display.
5. **Connection state per pod:**

   | state | rule |
   |---|---|
   | `creating` | ledger row exists, no ssh yet |
   | `starting` | ssh not ready, or the kill switch is not armed yet |
   | `connected` | last probe OK within 2 intervals, or `runner.log` growing |
   | `degraded` | last OK 2-5 intervals ago |
   | `lost` | more than 5 intervals, or the pod is missing from `pod list` |
   | `ended` | ledger row closed |

   Extra flags:
   - `runtimeStatus` other than `running`.
   - `killswitch.armed == false`: red, because the local delete is the only guard.
   - **Pod alive but runner dead**: red "orphan: the pod self-deletes at HH:MM ($X)". The [Delete now] button
     asks for confirmation.
6. **Strictly read-only by default.** The only actions are buttons that need confirmation and call existing code:
   - delete one `dancecap-*` pod: `remote._delete`, which closes the ledger row;
   - `remote.cleanup_orphans`;
   - launch a session through `remote_jobs.run_session` in a new runner process.

   The monitor binds to 127.0.0.1 only.

## 6. ETA

### 6.1 Notation

- For a step i: total `N`, done `d`, fraction done `x = d / N`, and observations `(t_k, d_k)` corrected by
  `skew`.
- History for its key (section 7): median seconds per unit `s_h`, log spread `σ_h`, sample count `n_h`, warm-up
  `h_h` (from step start to the first tick), tail `τ_h` (after the last tick up to `end`). Optionally a time
  profile `F_h(x)`, the fraction of the step's duration spent by progress `x`, and the duration `T_h(N)`.

### 6.2 Linear steps: blended rate

```
advances      the observations reduced to the moments `done` increased (heartbeats / repeated values carry no
              rate; ticks read in one batch with the same receive time are one observation)
pauses        an interval slower than 5 x the median s/unit of the last 30 intervals AND longer than 30 s
              is a pause: left out of the mean AND the spread (the window reaches further back instead)
live rate     s_live = Σ dt / Σ dd over the newest kept intervals until >= W = 120 s of working time and
              >= 4 intervals; valid when >= 20 s and >= 1 unit
weight        w = x / (x + x0),  x0 = 0.05 if n_h >= 3, 0.02 if 1 <= n_h < 3 (0.25 when the tick times are
              estimated, cold L0 read);  w = 1 if no history;  w = 0 if no s_live
blend         ln s = w ln s_live + (1 - w) ln s_h
remaining     R0 = (N - d) * s + τ_h  at the last advance; then
              R = max(0.1 R0, R0 - since)              while since <= 3 x the median gap between advances
              R = max(0.1 R0, R0 - 3 x gap), R90 += since - 3 x gap      once stalled (stops counting down)
not ticking   R = max(0, h_h - (now - t_start)) + N * s_h + τ_h      (started, no tick yet: model loading)
pending       R = h_h + N * s_h + τ_h
band          σ = (1 - w) σ_h + w σ_live   (σ_live = MAD of ln of the kept rates / sqrt(n), floor 0.05; 0.1 after a
              dropped pause; 0.3 with < 3 intervals); history with < 10 samples keeps σ_h >= 0.15
              R10 = R e^(-1.28σ),  R90 = R e^(+1.28σ)
unsettled     fewer than 3 kept intervals, a pause just dropped (< 5 kept), one interval > 60 % of the window, or
              estimated tick times: never the basis of a red alarm (section 6.6)
```

**Texture bake specifics.** Every pass walks the views in the same order and the per-view cost varies ~4x along it
(2026-10-07: 0.04-0.18 s/view inside one pass; the passes' time profiles agree within 0.01-0.04). From pass 1 on,
the previous pass's time profile F(x) replaces the 2-minute window (section 6.3 with T_h = the previous pass's
seconds, ρ shrunk to the expected pass-to-pass ratio while x is small; a resumed pause's own time is subtracted).
A finished pass's speed is taken from its ticks without its pauses. An unfinished pass's live speed reaches the
later passes only bounded to 0.5x-2x and weighted by x; pass 0's speed carries at half weight (another kernel).
Cold L0 read (no receive times): the unfinished pass is re-timed from the exact command start
(`[pod-runner] <ISO>Z start`) + the load phase from history + the finished passes' measured seconds, instead of
spreading its ticks back over the ~5 min load phase.

Blending in log space is the right choice for rates: a 2x error in either direction costs the same.

### 6.3 Non-linear steps: profile scaling

gsplat densification slows down as Gaussians are added: 38 s for the first 1000 of 30 000 steps, about 68 s for
the last 1000. MAMMA's four runs also differ per iteration. For these steps the history stores `F_h` at
x = 0, 0.05, ..., 1, and the estimator computes:

```
ρ  = (now - t_start) / (T_h F_h(x))          how much slower or faster than the historical run, so far
ln ρ̂ = w ln ρ                                 (shrinks to 1 while little is done)
R  = ρ̂ T_h (1 - F_h(x)) + τ_h
```

With a linear profile (`F_h(x) = x`) this reduces to section 6.2.

### 6.4 Indeterminate totals (early stopping)

`E[N] = clip(q_h N_max, d + 1, N_max)`, where `q_h` is the historical fraction of `N_max` actually used:
- MAMMA on 2026-10-07: run 1 16/50 (0.32), run 2 270/500 (0.54), run 3 100/100, run 4 17/25 (0.68);
- seed q = (0.4, 0.6, 1.0, 0.8) until 3 samples exist.

A `Stopping early` event sets `N := d` and ends the run.

### 6.5 Parallel lanes, the front DAG, and branches

- **Lanes** (MAMMA chunks under MPS: 7 lanes on 2026-10-07). `R_job = max_l R_l + R_after`, where `R_after` covers
  merge, gate and select. Each lane's live rate already includes GPU contention. The estimator does not model the
  speed-up when other lanes finish, so it stays conservative. On 2026-10-07 the lanes ended between 2261 and
  2831 s.
- **Front DAG.** The remaining tasks are list-scheduled. Task durations are history x workload (frames of that
  camera). Running tasks use their live `R`. The caps per kind and the dependencies come from `front_pod.json` and
  `front_pod.build_tasks`. The makespan is the ETA. There are about 40 tasks, so this costs microseconds. On
  38 s takes the critical path is the long camera: `sam:02` (862 s), then `ma2d:02` (614 s); `loc:02` (two halves
  of about 1000 s); then `cameras` (1171 s), then `sync` (1567 s, CPU).
- **Branches.** When a job can take two paths, show both. The MAMMA gate is either pass (`select`) or fail
  (`fallback_fit`, `solo_minutes(T)` from `mamma_chunks`, or its history: "~66 min" printed, 53 min measured).
  The display reads "ETA 6 min if the gate passes / 59 min with the full-fit fallback". Budget p90 takes the
  fallback branch while P(fail) is at least 0.1. The prior is P = 0.5 until 3 gates exist; the record so far is
  1 gate, failed.

### 6.6 Session ETA, and the kill-switch check

```
R_session = R(bundle_build) + R(pod_start) + R(upload) + R(extract) + Σ_features R(bootstrap/f)
            + Σ_jobs [R(archives) + R(setup) + R(commands) + R(pack) + R(download) + R(extract)] + R(delete)
R(upload) = bytes_left / r_up   (r_up live from the .part size; else the historical p50, and p90 for the budget)
display band: R10/R90 of the sum = R50 ± sqrt(Σ (R90_i - R50_i)²)       (independent steps)
budget p90 : Σ R90_i                                                    (conservative)
```

**Kill switch.**
- `kill_at` is the probe's `/workspace/.dancecap_kill_at`. Under L0 it is `pod.started + killswitch.after_min x 60`.
- `R_pod` is the time until the last download ends, after which the pod is no longer needed.
- The margin is `m = kill_at - (now + R_pod_p90)`.
- If the p50 margin is negative, the state is **red**: "the kill switch fires ~N min before <job> is downloaded:
  outputs lost, ~$X wasted". `mamma_chunks.gate_or_fallback` already makes this decision for its fallback on the
  pod; the monitor makes it visible for every job. While a running phase's live rate is **unsettled** (section 6.2)
  the same condition is amber ("may fire ... if the current rate holds"): one resumed stall must not cry wolf.
- If the p90 margin `m < 5 min`, the state is amber.

### 6.7 Stalls and anomalies

| condition | level | message |
|---|---|---|
| a step WITH a progress signal (>= 2 advances of `done`) has not advanced for `max(300 s, 10 x its median gap between advances)` | amber | "no progress in <command> (<step>@<lane>) for N min at d/N" + "the process is alive (heartbeats) but not advancing" when heartbeats came after the last advance. Heartbeats never reset this clock |
| no non-heartbeat event and no log growth for `max(300 s, 5 x median inter-event gap)` during a runner phase (upload, bootstrap, ...); `max(that, 900 s)` during a command without in-step signal | amber | "no progress for N min (last: <step>)" / "no new log line ..." |
| GPU utilisation < 5 % for > 5 min during a command known to use the GPU (`progress_eta.GPU_COMMANDS`: texture, MAMMA fits / gate, room, avatar_face, detect, dancers, hand_worker, refine_hands detectors, pose_refine) | amber | "GPU idle"; CPU-bound commands (timing, physics, pose_quality, the MANO fits, front localisation / sync) never raise it |
| `abs(ln s_live - ln s_h) > max(3 σ_h, ln 1.5)` | amber | "running 2.1x slower than usual (A100 PCIe, 8 vCPU)" |
| probe `load1` > 1.5 x cgroup CPUs during CPU-bound steps | info | "CPU-bound: X vCPU" (front_pod measured an 8.5-vCPU quota on an A100 PCIe community pod) |

### 6.8 Worked example: the texture bake of 2026-10-07 (`20261007-065744-batch`, A100-SXM4 secure)

| | prior (`avatar_pod`) | live, at `bake/pass2` view 1450/2292 | actual |
|---|---|---|---|
| s/view, aligned pass | 0.114 | `s_live` 0.132; x = 0.63, w = 0.93 -> **s = 0.131** | 0.132 |
| remaining in pass 2 | | 842 x 0.131 = **110 s** | ~111 s |
| fixed / non-pass time | 2 + 0.012 x ~1150 frames = 15.8 min | `τ_h` (from history once L2 measures it) | 5.3 min |
| whole bake | ≈ 27 min for 2292 views (the job file's `max_minutes` is 39) | | 18.0 min |

The prior was 18 % optimistic on the aligned rate and about 3x pessimistic on the fixed part. The live blend fixes
the first within a minute of pass 1. L2's explicit pre/post steps give the second a measured history, so the next
bake of this length can be sized from data rather than from a constant.

## 7. History: `work/progress/rates.json` (derived, rebuildable)

```json
{"v": 1, "built": "2026-10-07T18:00:00", "sessions": 32,
 "keys": {
  "session/upload/MBps":            {"n": 28, "p10": 0.7, "p50": 2.5, "p90": 2.8},
  "session/pod_start/s":            {"n": 21, "p50": 34, "p90": 45},
  "session/bootstrap/base/s":       {"n": 21, "p50": 87, "p90": 370},
  "session/bootstrap/mamma/s":      {"n": 3, "p50": 23, "cold": 218},
  "avatar/bake.pass0/views@A100-SXM4":   {"n": 1, "s_p50": 0.063, "sigma": 0.10},
  "avatar/bake.aligned/views@A100-SXM4": {"n": 2, "s_p50": 0.134, "sigma": 0.10},
  "room/train/steps@A100-SXM4":     {"n": 1, "T_p50": 1944, "N": 30000, "profile": [0, 0.019, 0.04, "..."]},
  "mamma/run2/iters@A100-PCIe":     {"n": 7, "s_p50": "a + b*frames, x parallel factor", "q_used": 0.54},
  "front/loc/frames@A100-PCIe-8vcpu": {"n": 6, "s_p50": 0.78, "sigma": 0.2, "mount": "handheld", "zoom": true}
 }}
```

- **Key:** `<job>/<step-kind>/<unit>@<host-class>`. The host class is the short GPU name for GPU-bound kinds, plus
  `-<vCPU>vcpu` for CPU-bound kinds. Texture loading, loc, sync and cameras are CPU-bound: the texture path is
  "CPU-bound in view loading" per `avatar_pod`. Local stages use `@local`.
- **Instance stripping:** `fit_chunks/c8_f0-232/run2` becomes `mamma/run2`; `front/loc:02:0` becomes `front/loc`;
  `bake/pass2` becomes `avatar/bake.aligned`.
- **Workload normalisation:**
  - MAMMA per iteration: `a + b x frames` (seeded from `mamma_chunks.ITER_FIXED_S` 0.40 and `ITER_PER_FRAME_S`
    0.0048), x `PARALLEL_SLOWDOWN` 2.1 x (lanes + 1) / lanes.
  - Texture: per view.
  - gsplat: per step, with the profile.
  - Front tasks: per frame of that camera.
  - Upload and download: per byte.
- **Lookup fallback chain:**
  1. exact key;
  2. the same kind on another GPU, x GPU factor;
  3. the code constant seed: `UPLOAD_MBPS`, `BOOTSTRAP_MIN[_CACHED]`, `mamma_chunks` constants,
     `avatar_pod.POD_S_PER_VIEW`.

  **The GPU factors are placeholders until measured:** A100-SXM4 1.0, A100-PCIe 1.05, H100 0.7, L40S 1.1,
  A40 1.5. The estimator says "estimated from <other GPU>" whenever it uses one.
- **Statistics:**
  - median and MAD of `ln s` over the last 10 samples per key, so recent code paths dominate;
  - samples from a different `code_path` signature (for example `avatar_pod.per_view_rates`' old and new paths)
    form separate keys.
- **Rebuild:** `python -m dancecap.progress rates --rebuild` reads every session's progress files (L1/L2), plus
  `state.json` timings and L0 adapter parses of old `runner.log` files, and `front_pod.json`. Its writer uses a tmp
  file plus `os.replace`, retried on `PermissionError`. The runner calls it at the end of `_finish` inside
  `try/except`; the monitor may call it on demand. Both write only under `work/progress/`.

## 8. Budget

### 8.1 Formulas

```
C_closed    = Σ_{rows with deleted_ts} cost_r                          (ledger; reconciled values when present)
C_open(now) = Σ_{open rows} p_r (now - created_r) / 3600
S           = C_closed + C_open = remote.spent()                        (the same function: one source of truth)
remaining   = cap - S                                                   (cap = DANCECAP_RUNPOD_CAP, default 30)
per running pod a:  P_a = p_a R_pod_a / 3600 (p50),  P90_a = p_a R90_pod_a / 3600,
                    W_a = p_a max(0, kill_at_a - now) / 3600            (hard worst case: on-pod kill switch)
per queued session q: P_q = p_q (pod_start + upload(bytes) + extract + bootstrap(features) + Σ jobs + pack/download
                    + margin) / 60,   p_q = cheapest live price among the allowed GPUs/clouds (remote.gpu_price)
projected   T50 = S + Σ P_a + Σ P_q,      T90 = S + Σ P90_a + Σ P90_q
worst       Wc  = S + Σ W_a + Σ P90_q
```

Runpod bills pods per second, so the ledger's `price x lifetime` should match the bill to within cents.

### 8.2 Warnings (reserve default $2, env `DANCECAP_RUNPOD_RESERVE`)

| level | condition |
|---|---|
| green | `T90 <= cap - reserve` |
| amber | `T90 > cap - reserve`, or `remaining` < one hour of the cheapest 80 GB GPU (A100 PCIe community $1.19/h), or kill-switch margin < 5 min |
| red | `T50 > cap`; `Wc > cap`; an active pod with `killswitch.armed == false`; untracked spend (section 8.4); kill-switch margin < 0 |

### 8.3 Launch gate (monitor "Run")

The gate requires `S + Σ_active W_a + P90_new <= cap`. `pod_session`'s own check is
`price x (max_minutes + start) <= cap - spent()`, which counts running pods only by what they have spent so far. Two
concurrent sessions could therefore over-commit; the monitor's gate closes that hole for anything it launches. The
dialog shows p50 and p90 minutes and dollars per phase, the GPU and cloud, and the kill-switch minutes it will
request.

### 8.4 Cross-checks (read-only)

- **Untracked pod.** `runpodctl user.currentSpendPerHr` > Σ open-row `cost_per_hr` + $0.05/h means a pod the
  ledger does not know about (created from the console or MCP). Red, with the existing [Cleanup dancecap-* pods]
  action (confirm).
- **Phantom accrual.** An open ledger row whose pod has been absent from `pod list` for more than 2 min makes
  `spent()` keep accruing a pod that is gone (a runner killed hard, a pod that self-deleted at its kill time). The
  monitor displays the corrected spend `p (min(now, kill_at, last_seen) - created) / 3600` next to the ledger value
  and offers a reconcile command. Closing the row writes the ledger; it is a user action through a CLI, never
  automatic.
- **Billing reconciliation (optional CLI, not the monitor loop).** `GET https://rest.runpod.io/v1/billing/pods?
  podId=<id>&bucketSize=hour` with a Bearer header returns `amount` and `timeBilledMs`. Write the result to an
  additive `billed` field on the ledger row.
- **Account balance.** `clientBalance` is shown as "Runpod balance". If it is below `cap - S`, the effective cap is
  the balance: amber.

### 8.5 Where the money goes: phase breakdown (two pods of 2026-10-07)

The monitor shows the cost per phase for every pod, live, because the pod bills while it waits:

| pod | lifetime | upload | bootstrap | work | other |
|---|---|---|---|---|---|
| A100 PCIe community $1.19/h (MAMMA) | 137.8 min, $2.73 | 9.0 min, $0.18 | 20.2 min, **$0.40** (PyPI stall) | chunk fits 47.2 min $0.94; merge 3.7 min $0.07; **gate fallback full fit 53.0 min $1.05** | ~$0.09 |
| A100-SXM4 secure $1.59/h (texture re-bake) | 29.8 min, $0.79 | 9.1 min, **$0.24 (31 %)** | 1.5 min, $0.04 | bake 18.0 min, $0.48 | ~$0.03 |

**Levers the monitor can point at, and size from the same history:**
- The upload share. A 4.2 GB front bundle at the p90 rate (0.7-0.9 MB/s) means more than 75 min of A100 time
  (about $1.5) spent uploading.
- Chaining jobs on one pod instead of re-creating one: re-upload and re-bootstrap.
- The pywheels cache: the 20-min bootstrap.
- The probability and cost of the MAMMA gate fallback.

Choosing a lever is out of scope for this doc; the point is to make each lever's cost visible.

### 8.6 State as of this writing (ledger, 2026-10-07 ~17:30 UTC)

- **Spent:** $19.94 of the $30 cap; **remaining $10.06**.
- **Pods:** none running (`runpodctl pod list` is empty). The texture re-bake session `20261007-065744-batch` ended
  `ok` and its pod was deleted with the session ($0.79).
- **Projection:** a front + MAMMA + avatar session for a new 38 s take, chained on one A100 PCIe community pod at
  $1.19/h, from the history in section 1.1:
  - **p50 ≈ 181 min (≈ $3.6) if the gate passes, ≈ 234 min (≈ $4.6) with the fallback.** Upload 4.2 GB at
    2.5 MB/s is 28 min, bootstrap 7 min, front 73 min, MAMMA 51 or 104 min, bake 18 min, plus start and
    downloads.
  - **p90 ≈ 321 min (≈ $6.4)**, from a 0.7 MB/s upload (100 min), a 20-min bootstrap and the fallback.
  - One such session fits under the reserve rule ($10.06 - $2 = $8.06); two do not.

## 9. Camera types (mount, zoom) in the workload, ETA, budget and QA

Per source video, the take TOML holds the user's answer from the review UI, or an automatic suggestion awaiting
confirmation:

```toml
[cameras."01"]
mount = "handheld"      # handheld | tripod   (exclusive)
zoom = false            # independent: a tripod camera can zoom
source = "user"         # user | auto (suggested, not confirmed)
```

`[cameras]` already holds `scale_reference` and `loc`. Readers must take per-camera subtables only for keys that
are source ids.

### 9.1 Effect on work units, which drives the ETA and pod minutes

| stage | handheld (today) | tripod | zoom = no | zoom = yes | ETA / cost effect |
|---|---|---|---|---|---|
| `cameras` loc (pod; per-frame ALIKED + LightGlue + P4Pf / P3P) | every frame: units = frames (0.56-0.8 s/frame on the pod) | **one pose per video**: localise K keyframes (about 1 per 2 s, so about 20 on a 38 s take), take a rigid consensus, check a sample of frames; units = K + sample, about 3-5 % of the frames | fixed intrinsics, one focal per video: P3P (fewer unknowns, fewer RANSAC trials) | focal per frame; on a tripod the pose is still fixed, so 1 unknown per frame against the static map | a tripod camera's `loc` lane drops from about 650-1000 s to tens of seconds; the DAG simulation shows the real saving (on 38 s takes the long camera's SAM + ma_2d may stay critical) |
| `cameras` smooth / `depth_guard` | per-frame pose + log-focal smoothing | skipped (constant pose) | constant focal | log-focal smoothing, zoom events | small, local |
| `sync` | unchanged cost | a static background helps the image cue | | | none |
| MAMMA `prepare` / fit | cameras are inputs | same cost; tripod cameras are preferred for the held-out check | | | none |
| texture bake | per view | same | | | none |
| single / two-camera takes | fewer cameras means fewer units for loc, SAM and ma_2d | | | | the workload model counts cameras and frames |

**Estimator rule** (`progress_eta.camera_types()` -> `loc_units` / `loc_estimate` / `front_dag` / `estimate_front`):
- The types come from `dancecap.camera_types` (`load` + `stage_type` under the active policy: `DANCECAP_CAMERA_TYPES`,
  default `user`), i.e. exactly what `stage_loc` / `front_pod.loc_part` / `stage_smooth` apply. Under `user` only a
  confirmed answer (`source = "user"` with BOTH mount and zoom) counts; an `auto` suggestion, a table without
  `source`, or a missing zoom run today's general model, and are priced as such.
- **p50** prices the effective class: keyframe localisation only for a fixed, non-panning, non-zooming tripod
  (`typed.localize_typed`); tripod + zoom and a panning head localise every frame (both code paths).
- **p90** adds the fallback the stage can still take: a keyframe tripod whose constant pose is rejected is localised
  again on every frame (keyframes + all frames).
- `estimate_front` also returns `if_confirmed_p50` / `potential_saving_s`: what confirming the automatic suggestions
  would change, so the launch dialog can say "confirm 3 tripods: -17 min".
- `front_dag` mirrors `front_pod.build_tasks`: only the long pole (largest SAM size among ALL cameras, split only if
  it is a main-model camera) is localised in two parts named by their first frame (`loc:02:0`, `loc:02:1151`).

### 9.2 QA against the declared type (the same metrics feed the automatic suggestion)

The `cameras` stage's qa step emits one `metric` event per camera **from probe B** (`camera_type_probe`, after
`build_report`): `dancecap.camera_types.track_metrics` on today's UNCONSTRAINED per-frame track (a typed camera's
general model is re-derived; never the type-constrained export, whose centre spread / focal ratio are 0 / 1 by
construction) plus probe B's suggestion:
`{source: "probe_b", constrained: false, centre_p95_m, axis_p95_deg, angvel_p50_dps, hf_rot_deg, hf_centre_mm,
focal_ratio, focal_step_pct, frames, mount_judgeable, attached, mount, zoom, pans, confidence, mount_confidence,
zoom_confidence, at}`. The monitor (`camera_type_flags`) uses `camera_types`' own thresholds (`suggest_from_metrics`,
`MIN_CONFIDENCE` 0.7: tripod centre p95 < 2 cm and rotation jitter < 0.05 deg, handheld > 5 cm; zoom focal p95/p5 >
1.05 or a step > 3 %, no zoom < 1.02 and no step > 2 %), so it never disagrees with the camera_types gate:

| declared | probe B (confidence >= 0.7) | flag |
|---|---|---|
| confirmed (user) | contradicts mount or zoom | amber per field, e.g. "camera 03 marked tripod but measures handheld: centre 18.0 cm (p95) ...; the rigid pose will be rejected - check the type" (= camera_types' `conflict` state) |
| not confirmed | tripod / no zoom | info "camera 06 looks static: confirm 'tripod' ..." / "shows no zoom: confirm 'no zoom' ..." |

A measurement older than the user's latest answer (`at <= set_at`) was acknowledged and raises nothing. Events
without `source = "probe_b"` (the first version's metric on the constrained export) are ignored.

## 10. Integration points (minimal, additive)

### 10.1 New modules (local and pod, stdlib only, no new pins)

- `dancecap/dancecap/progress.py`:
  - helper API (section 3.1) and the adapter table (section 3.2);
  - `parse_runner_log(path, offset)` for L0;
  - CLI `python -m dancecap.progress tail <sid>|--take T` (terminal view),
    `python -m dancecap.progress eta <sid>` (the monitor's snapshot as JSON), and
    `python -m dancecap.progress rates --rebuild`.
- `dancecap/dancecap/progress_pull.py`: `Puller(pod, session, interval=15)`, the `PROBE` script string, and the
  mirror writer.
- `dancecap/dancecap/progress_eta.py`: pure functions for history lookup, the ETA (sections 6.2-6.6), the budget
  (section 8.1) and the workload model (frames per camera from `frames/<cam>/frames.json` or ffprobe before ingest;
  camera types from the take TOML). It returns `snapshot(session_dir) -> {phases[], steps[], eta {p10, p50, p90},
  kill_margin, cost {so_far, p50, p90, worst}, warnings[]}`. The monitor and the CLI share it.

### 10.2 `remote_jobs.py`: every edit additive and wrapped so progress can never fail a session

1. **`command_script`.** Add `DANCECAP_PROGRESS`, `DANCECAP_PROGRESS_JOB`, `DANCECAP_PROGRESS_CMD` and
   `DANCECAP_SID` to the `envs` dict. The pod progress path is derived from `log_rel` (`<log_rel before
   /pod_logs/>/progress.jsonl`), so the signature does not change. Jobs that ignore the variables behave exactly
   as before.
2. **`Session`.**
   - New `event(kind, step, **kw)` that appends to `self.dir / "progress" / "runner.jsonl"` inside `try/except`.
   - `timed()` calls `event("start")` before and `event("end"/"fail")` after. Timings and logs are unchanged.
   - New `write_runner_info()` writes `runner.json`: pid, started, attempt (count of previous runner.json writes +
     1), label, job names.
3. **`run_session`.** After `pod_session` yields, `pod_info` also stores `port` and `kill_at_est = time.time() +
   mm x 60`. The puller later stores the exact `kill_at` read on the pod.
4. **`_pod_steps`.** First line: `puller = _start_puller(pod, ses, upload_part=f"{remote_b}.part",
   upload_bytes=man["bytes"], tails=...)`. It imports `progress_pull` lazily and returns `None` if the import or
   the start fails. Stop it in a `try/finally` around the rest of the function: set the stop event, `join(10)`.
5. **`_run_job`.** Set `st["current"] = c.name` before each command and `st["pack_bytes"] = size` before the
   download. Write `download` events with the bytes of the local `.part` file every 5 s from a small timer while
   the scp runs (optional).
6. **`estimate_minutes(jobs, upload_bytes, cached_wheels=None, rates=None)`.** New optional `rates` argument. When
   given, upload and bootstrap come from history and the result gains `p90_total` (an additive key). Existing
   callers and `max_minutes` defaults stay unchanged.
7. **`_finish`.** Call `progress.rates_rebuild()` inside `try/except` (it writes only under `work/progress/`).

**Not touched:** `pod_session`, `arm_killswitch`, the ledger writes, `_delete`, the retry/pack/download/extract
logic, the bundle format. `remote.py` needs **no change**. `remote.status()["account"]` already returns only the
balance, spend and limit fields; the monitor uses `remote.ledger()`, `remote.spent()`, `remote.cap()` and
`remote._ctl("pod", "list")` read-only.

**Running sessions.** A runner process started before these edits keeps its old code in memory and is L0 for its
whole life. Editing the file on disk does not affect it. Sessions resumed with `--session <sid>` after the edit
become L1 from their next pod on.

### 10.3 Stage instrumentation (L2), in order of dollars at risk times missing signal

1. `mamma_chunks.fit_chunks` / `gate_or_fallback`: `watch_log` per chunk on `ma_3d_<tag>.log`, plan, and the branch
   note. This is the longest blind spot (47 + 53 min on 2026-10-07).
2. `front_pod.Runner._start/_finish`; `dancers_sam` propagation (frames); ma_2d via `tqdm_bridge`; the loc
   per-60-frame print as a `Step`.
3. `avatar_texture`: plan; `Step` per pass and for the pre/post phases.
4. `room` training loop: `Step("train", total=iters)`, plus the quest and post phases.
5. ~~`hands_pod` / workers~~ (hands stages retired 2026-10-07; nothing to do); `detect`.
6. `bootstrap.sh`: one JSON line per feature when `DANCECAP_PROGRESS` is set. **Coordinate with the in-flight
   uncommitted bootstrap.sh change.**
7. Local long stages, launched by the pipeline monitor with `DANCECAP_PROGRESS` set.

### 10.4 CONTRACTS.md additions (when implemented)

Add `work/runpod/sessions/<sid>/progress/{runner.jsonl, pod.jsonl, puller.json}`, `runner.json`,
`work/<take>/progress/<stage>[.<pid>].jsonl`, `work/progress/rates.json`, the event schema of section 2.1, and the
optional `progress.tails` key of the pod job.

## 11. Rollout and tests

| level | needs | monitor shows |
|---|---|---|
| L0 | nothing (works on the 32 existing sessions and on any runner on the current code) | phases with start times; texture, room, hands, loc and front-task progress; MAMMA lanes only at their ends; upload as an estimate; budget, connection, kill timer |
| L1 | §10.1 modules + §10.2 edits | everything in L0, plus live upload bytes, GPU utilisation, exact kill time, MAMMA iterations (probe tails), front task states, download bytes, runner liveness |
| L2 | §10.3 | plans with totals for every step, measured warm-up and tail times, SAM / ma_2d frames, QA metrics |

**Tests** (no pods; fixtures are log excerpts, which hold no media):
- **Adapters.** Every regex runs against excerpts of `runner.log` (texture views and passes, room steps, chunks,
  front, bootstrap), `ma_3d_*.log` (`Iter.`, early stop), `ma_2d_*.log` (tqdm with `\r`), and loc logs.
- **L0 replay.** Parsing `20261007-065744-batch/runner.log` must reproduce the step durations in its `state.json`
  timings within 2 s.
- **ETA.**
  - Synthetic linear and profiled sequences converge to the truth: within 10 % at x = 0.2.
  - Early-stop revision works.
  - Lanes use the max.
  - The front DAG makespan stays within 15 % of the measured 4374 s when replayed from its task seconds.
  - The texture example of section 6.8.
- **Budget.** With a fake ledger (open and closed rows), `S == remote.spent()`. Also check the phantom-accrual
  correction, the launch gate with two active pods, and the warning levels.
- **Puller.** A fake `Pod` whose `_ssh_base` runs a local `python - < PROBE` against a temp dir must give:
  - offsets that are byte-exact across ticks, including split lines;
  - backoff on failure;
  - no exception escaping the thread;
  - `puller.json` written atomically.
- **Safety.**
  - The probe source contains no `environ`, `kill` or write calls (a static check).
  - Event writer redaction.
  - The monitor renders `msg` as text.

## 12. Risks and open questions

- **Probe cost on the pod.** A `python3` start every 15 s (~50 ms of CPU) on a container with an 8.5-vCPU quota is
  negligible, but a `glob` over large log directories must stay bounded: at most 32 files, tails at most 16 KB.
- **ssh handshakes from this PC** (4 CPUs, Windows OpenSSH): about 1-2 s per tick. Use a 30 s interval when two
  sessions run at once.
- **Clock skew.** Pods are NTP-synced, but L0 `runner.log` times are local `HH:MM:SS` without a date. Take the date
  from the sid, or from the file mtime when a session crosses midnight.
- **History is thin.** For most keys n = 1-7. Bands stay wide (σ_h prior 0.3 when n < 3) until L1/L2 have collected
  data, and the UI says "estimated from 1 run". The GPU factors are placeholders.
- **`state.json` truncate-writes are safe for readers, `os.replace` is not** while the monitor reads on Windows.
  Any future change to atomic writes in the runner needs a `PermissionError` retry.
- **Upload variance dominates p90** (0.19-2.8 MB/s). A live `.part` rate after 30 s replaces the prior quickly, but
  before the pod exists only the history is available.
- **Camera-type thresholds** (section 9.2, owned by `dancecap.camera_types`) are uncalibrated. All 8 larissa_kadu
  phones are handheld, so the tripod path has no measured rate yet: its loc saving is an estimate from unit counts.

## 13. Validation on a live session (2026-10-07, read-only)

`20261007-110948-batch` (larissa_kadu_38 `avatar_fix` re-bake, A100-SXM4 community $1.39/h, runner started on the
pre-progress code, so L0 only) was followed without touching it: runner.log was polled every 2 s (open, read,
close) and each new line stamped with its receive time, exactly what `progress_eta.Tracker` does. Replaying that
recording through `snapshot()` with the history as it was before the session (`rates.json` built 11:23):

| when | phase | predicted end (p50) | error vs actual 11:46:51 | p10-p90 band |
|---|---|---|---|---|
| 11:13 | upload, 0.5 of 1.5 GB | 11:44:11 | -2.7 min | covered |
| 11:23 | bootstrap `base` (overdue: 6.4 min vs 1.4 typical) | 11:47:32 | +0.7 min | covered |
| 11:31 | bake pass 1/3 | 11:49:19 | +2.5 min | covered |
| 11:38 | bake pass 2/3, 2100 views | 11:48:42 | +1.8 min | covered |
| 11:46 | metrics | 11:48:00 | +1.1 min | covered |

Cost: projected p50 $0.79-0.98 throughout, actual ledger $0.86; worst case (kill switch) $1.68. The bake took 1085 s
(18.1 min; the plan estimate was 17.7). The remaining error came from the two optional commands without history
(face 122 s, metrics 6 s, priced at 0.3 x timeout) and the stalled `base` bootstrap; both are now in `rates.json`.

Measured on this run and now the seeds: the bake's fixed time splits into **pre 294 s** (load, head fix, hair
model, atlas, masks) and **post 28 s** (fusion, fills, review images), not the ~50/50 the docs assumed; pass 0 165 s,
aligned passes 299 / 295 s.

The same replay of the resumed front session `20261007-002359-batch` (front DAG from its `[front]` lines + the take's
frames) gave a remaining time within 4 % of the actual at +15 min and +30 min (`tests/test_progress.py`).

Sources (checked 2026-10-07):
- [Runpod pricing analysis, which states that pods are billed per second](https://www.thundercompute.com/blog/runpod-pricing-vs-thunder-compute)
- [Runpod REST: get a pod (fields, Bearer auth, no utilisation telemetry)](https://docs.runpod.io/api-reference/pods/GET/pods/podId)
- [Runpod REST: pod billing history](https://docs.runpod.io/api-reference/billing/GET/billing/pods)
- [Runpod GraphQL: manage pods (runtime GPU utilisation fields)](https://docs.runpod.io/sdks/graphql/manage-pods)
- `runpodctl 2.14.0`: `pod get --help` documents `runtimeStatus` and `runtimeStatusReason`.

## 14. Review fixes and re-validation (2026-10-07)

The adversarial review of the first version found five medium issues; all are fixed, re-tested on the reviewer's
own synthetic sessions and on the recorded live session `20261007-110948-batch`.

| issue | fix | before | after |
|---|---|---|---|
| a resumed pause inflated the rate (and cost, kill level) while the band stayed narrow | pauses dropped from mean and spread; a finished pass's speed without its pauses; later passes bounded 0.5x-2x; no red from an unsettled rate | 300 s pause in pass 2, at the resume: ETA error +74.7 min, band 77-89 min vs 7.9 true, cost p50 $2.41, kill RED | error +0.0 min, band 5.4-13.2 min (true 7.9), cost p50 $0.68, kill green; whole run 2/58 points outside the band, all before the pause |
| a stalled step kept counting down; heartbeats hid the stall | remaining counted from the last ADVANCE, frozen after 3 gaps (p90 grows); warning from the progress signal, heartbeats excluded | 500 s left read 100 s after a 400 s stall, no warning | p50 stays (3 gaps credited), p90 grows; amber "no progress in avatar/bake (bake/pass1) for 5 min at 1125/2292" (+ "alive (heartbeats)") |
| run 1.3x slower than history: bands overconfident | history sigma floor 0.15 for < 10 samples; pass 0 speed carried at half weight; previous-pass profile | 27/59 points outside the band, error -5.5 min | 1/59 outside (before the bake started), bake ETA error <= +0.2 min |
| estimator read camera types differently from the stages | `camera_types()` = `dancecap.camera_types.stage_type` under the active policy; tripod + zoom = every frame; p90 keyframe fallback | an unconfirmed all-tripod 'auto' take priced 3539 s | priced as handheld, 4547 s (what the stage runs); `if_confirmed_p50` says what confirming saves |
| QA metric computed on the constrained export (circular) | metric = probe B on the unconstrained track, camera_types' thresholds, one implementation | "marked tripod moves" could never fire on an accepted tripod model | fires on probe B's evidence exactly when camera_types opens its gate |
| the monitor's path (cold `snapshot()` without receive times) was not the validated one | `live_snapshot()`: one background-polled `Tracker` per watched session; cold reads re-time an unfinished bake pass from the command start; mtime anchors the file's last line | cold replay: error up to +13.5 min, 7/34 points outside | cold replay (worst case, monitor opened at every point): -2.7..+5.4 min, 3/34 outside; live tracker from the start: -2.7..+4.2 min, 2/34 outside (face / metrics with no history, at the 1-second edge); tracker opened at +10 / +18 / +25 min: <= +4.7 min |

Smaller fixes from the same review: `[front] HH:MM:SS` lines roll over midnight UTC (and never land before the
session's local midnight); the GPU-idle warning only for GPU commands; `budget()`'s cap default is `remote.cap()`;
the command history key keeps whole verbs (`front_pod.fetch-sam2`); `dancers_sam` ticks distinct frames;
`Session.timed` filters reserved keys before `event()`. Not changed (reported): `remote_jobs.py`'s bootstrap retry
and `PYWHEELS` bundle item come from another workstream; the retry is unbounded by the kill-switch time left.
