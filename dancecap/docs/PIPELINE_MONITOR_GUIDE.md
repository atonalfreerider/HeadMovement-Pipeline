# Pipeline Monitor: user guide

> **2026-10-08: the monitor was redesigned around the farm.** The window now opens on the **Farm** view: a defrag-style map of
> the final result (section 0 below). The takes / stages / runs / Runpod-sessions screens described from section 1 on are
> still there under **Older tools** (top bar) but are no longer the front page, and the old runs and plans were archived to
> `dancecap/work/pipeline/archive/2026-10-08/` (nothing was deleted).

## 0. The Farm view (the front page)

One dance at a time (dropdown, newest first). From the top:

1. **Header**: the dance, its world package (id, sealed), the watcher's state, when the map was computed, and the actions:
   **Fill gaps**, **Pull now**, **Open watcher log**, **Restart watcher**, **Stop pods**, **Export map PNG**.
2. **Big numbers**: how much of the final result is complete / partial / not done, in percent and seconds, and a stacked bar.
3. **The map**: one square per second of the dance, wrapped like a disk-defragmentation display (30 per row = 30 s per row).
   - green = complete (refined, stitched, gates passed);
   - amber with a hatch = partial; the denser the hatch the further along: front only (cameras, 2D, sync, no 3D yet) < raw 3D fit
     < refined and waiting for the stitch; a red corner = mono / two-view / absent;
   - dark = not done at all;
   - the digit = cameras that saw that second; the bar under a square = the chunk that owns it; a bar in the gap = where a chunk
     starts; a dotted outline = a second that only belongs to a chunk's overlap;
   - a pulsing blue outline with a travelling wave = being worked on right now.
   Hover (or click, or use the arrow keys) for the stage, cameras and reasons of one second.
   The map follows `work/<dance>/farm/coverage.json`: it flips live (2 s poll) when the file changes, and when a fit / refine
   result lands on disk the monitor recomputes the coverage itself (`farm coverage`, about 4 s, at most every 2 min).
4. **Panels**: **Chunk jobs** (progress bar, ETA, heartbeat, GPU, cost so far per chunk), **Pods** (GPU, price, state, age, spend),
   **Queue**, **Budget** (spent / cap / account balance / free for new pods), **Persistent volume** (size, used, price), **World
   package**, **Recent activity**.
5. The four MAMMA fits started with `python -m dancecap.fit_full` (segments a-d) appear as pods + chunk jobs (tagged *fit_full*)
   and their chunks are drawn as being worked on. The monitor only reads them: it never stops them (a fit_full pod has no Stop
   button here).

**Fill gaps** asks the planner (`farm plan --dry-run`) which spans still need 3D poses, shows the chunks, the pods, the GPU
price, the time, the expected cost and the worst case (every pod runs to its kill switch) and the budget gate, and starts
nothing until you press the confirm button and then confirm again. Chunks that a fit_full session is fitting are excluded. The
plan expires after 15 minutes. It then runs `farm up` (plan, queue, launch, watcher) detached; follow it under Recent activity.
**Stop pods** needs you to type STOP and deletes only this farm's pods (`farm down`: finished chunks are pulled and verified
first; a pod whose results are not pulled yet is kept).

Command line: `python -m dancecap.pipeline monitor --launch [--dance larissa_kadu_full]`;
synthetic demo state: `python -m dancecap.pipeline farm-demo [--scenario mostly_partial|midrun|done|empty]` (open it with
`#/farm/farm_demo`); clear the old jobs again: `python -m dancecap.pipeline archive-legacy`.

## 0.1 Analyses, pod sessions and snapshots (2026-10-21)

Owner, 2026-10-21: a dashboard with the status of uploads, analyses, cost and time ETAs, kept running whenever anything is
analysed, with snapshots at progress points. What was added:

- **Analyses and pod sessions** (the strip at the top of the Farm page, GET /api/activity, every 3 s):
  - every analysis run (`python -m dancecap.pipeline analyze`, below): its steps as small squares, done / total, the
    end-to-end ETA (p50, p90 and the finish time), the pod cost so far and expected, and which steps wait for an agent;
  - **every** pod session running now, from any entry point (`dancecap remote run`, an analysis, an agent): the phase,
    the **upload** (MB sent / total, MB/s, time left, the number of streams and a rate sparkline), the session ETA and cost
    (so far, expected, worst case at the kill switch). Before this the remote-runner sessions were only under Older tools;
  - the pod sessions of the last 24 h, with their real outcome (a job "skipped: bootstrap feature failed" is now a failed
    session, not a green ok; a session whose runner died while its pod is open is "runner gone");
  - the budget: spent / cap / left / account balance. The cap shown is the one the pod runners use: without
    DANCECAP_RUNPOD_CAP it follows the Runpod balance (ledger spend + balance - $2, since 2026-10-21); a runner started
    with an explicit DANCECAP_RUNPOD_CAP records that cap in `work/runpod/cap_in_use.json` and it is shown instead; the spent figure is `remote.spent()` exactly (ledger rows
    left open past their kill switch no longer accrue forever: the monitor showed $530 spent where the ledger said $101).
  - The Farm page opens on the dance being analysed (when it has a farm), not on the one last viewed in this browser.
- **One analysis** (`#/analysis/<run>`): KPIs (steps, elapsed, ETA with the finish time, pod cost, critical path), the
  **analysis map** (one row per step, one square per pod x 2 minutes of the plan: green done, blue running and filled as far
  as it got, amber hatched = needs an agent, grey pending, red failed), the run's pod sessions with their uploads and
  phases, the take's coverage map (the farm's defrag squares) when it has one, the events and the snapshots.
- **Snapshots** (`#/snapshots`, `#/snapshots/<run>`): PNG + static HTML in `work/pipeline/snapshots/<run>/` with an
  `index.json`. Taken by a detached background process (never in the way of the work) at: the analysis start, each step's
  end, the driver's end, each `analysis step` mark, a 15-min heartbeat while steps run; and in every pod session (tagged to
  an analysis or not): upload done, bootstrap done / failed, each job's end, the session end. A static page (no script)
  rendered by headless Edge in 2-3 s; a text card (PIL) when Edge fails. Also: `python -m dancecap.pipeline analysis snap RUN`.
- **Always running**: `dancecap remote run`, `dancecap.pipeline run`, `dancecap.pipeline analyze` and `dancecap.farm up /
  launch` start the monitor server themselves (headless, no window, no console; about 1.5 s when it is not running, 15 ms
  when it is) and leave a lease (`work/monitor/leases/<pid>.json`). The server no longer quits after 20 idle minutes while a
  lease's process lives, a pod is open in the ledger, or an analysis step runs. Open the window any time with the launcher.

### `python -m dancecap.pipeline analyze <folder|video>`

```
python -m dancecap.pipeline analyze C:\Users\johnb\Videos\Zouk\john-lula\john_lula_jnj.mp4 --take john_lula --dry-run
python -m dancecap.pipeline analyze <folder> --take NAME --yes [--room-known] [--no-realism] [--detach]
python -m dancecap.pipeline analysis list | show RUN | resume RUN | snap RUN | cancel RUN
python -m dancecap.pipeline analysis step RUN STEP start|done|fail|skip [--note TEXT] [--session SID]
```

- One video = the monocular route; a folder of 2+ videos = the multi-view route (an audio file in the folder is the
  soundtrack). It probes the videos, reuses the take when it exists with the same videos, or creates it (a new take needs
  `--lead NAME:STATURE --follow NAME:STATURE --level --category [--consent-cloud]` or `--spec FILE.json`).
- It prints the plan: every step with where it runs (every heavy step on Runpod; this PC only for the intake, the audio
  clocks, the Unity export and the film), how many pods side by side, p50 / p90 minutes and dollars from the measured 2026-10
  runs, the end-to-end time on the critical path and the total pod cost. `--dry-run` stops there.
- Without `--dry-run` it runs the steps that have a generic executor today: the intake, the **upload of the source videos to
  the Runpod volume at t = 0** (a CPU gateway pod this run creates and deletes; 6 resumable streams; needs `--yes`) and the
  audio clocks. The other steps are marked **needs an agent** (they need take-specific code today; PIPELINE_EFFICIENCY.md
  5 rank 1): an agent drives such a step with `DANCECAP_ANALYSIS=<run> DANCECAP_ANALYSIS_STEP=<step>` set on its
  `remote run` sessions (the dashboard shows them under the step, as running) and marks it with `analysis step ... done`,
  which restarts the driver for the next steps.

---

The Pipeline Monitor is one window for the whole capture pipeline:
- add a new capture (videos + soundtrack + dancers + camera types);
- see every stage of every take, in the order it runs, with where it runs (this PC or Runpod);
- plan and start runs, with the time and the Runpod cost shown **before** anything is paid for;
- watch Runpod sessions live: progress bar, ETA (p50 / p90), cost so far and projected, kill-switch margin, budget;
- open the narration review tool, and export a take to Unity.

The design and the API are in `docs/PIPELINE_MONITOR.md`; the stage order is in `docs/PIPELINE.md`. This page is
the short "how do I" version. Screenshots: `dancecap/work/review/pipeline_monitor_FINAL_*.png`.

---

## 1. Start it

Double-click **Pipeline Monitor** on the Desktop. The same launcher is also at:
- `tools\PipelineMonitor\PipelineMonitor.exe` (the shortcut points here);
- Start menu > Head Movement > Pipeline Monitor;
- in Unity: **Head Movement > Pipeline Monitor**;
- fallback without the exe: `tools\PipelineMonitor\PipelineMonitor.cmd`.

What happens:
- The first double-click starts a small local server (`127.0.0.1:8790`, never reachable from the network) and opens
  an Edge app window. It takes about 2-5 s.
- A second double-click only opens another window on the same server.
- **Closing the window never stops anything.** Runs, pod sessions and the narration tool are separate processes.
- The server quits by itself after 20 minutes without a window open, unless work is running (a leased runner process,
  an open pod, a running analysis step: section 0.1). Every pod runner and analysis starts it by itself (headless).
- Settings > Restart / Quit restarts or stops the server (runs keep going).
- If the window says "This window must reload", press **Reload** (the server was restarted).

Command-line equivalent (from `dancecap/`): `..\.venv-nopo4d\Scripts\python.exe -m dancecap.pipeline monitor --launch`.
Logs: `dancecap\work\monitor\logs\launcher.log` and `monitor.log`.

---

## 2. The screen

- **Top bar** (right): Runpod connection, the number of pods running, and the budget gauge (spent / cap).
- **Alerts** under the top bar: budget warnings, a pod without a runner (orphan), stalled sessions, camera types to
  confirm. Each has a button to the right place.
- **Takes** (left): every take with a strip of its stages (green done, orange striped stale, blue running, red failed,
  outlined waiting), "x/48 done", the remaining time, and "needs you: n" when a human step is waiting.
- **Status bar** (bottom): active runs, the GPU lock and the Unity lock (a held Unity lock means another workflow is
  using Unity: exports wait), and the progress level of the live session.
- Keyboard: `g t` takes, `g r` Runpod, `g l` logs, `?` help, `Esc` closes the top dialog.

---

## 3. Add a capture (New capture)

Click **+ New** (or `g n`). Six steps:

1. **Sources.**
   - **Browse videos…** or **Pick a folder…** opens a file dialog on your desktop (it may open behind the window).
     You can also paste a folder path and press **Scan**.
   - An audio file in the folder is taken as the **soundtrack** (studio track). It is optional: with it, the phone
     clocks are aligned to the studio track and the beat grid comes from it.
   - **Take name**: lowercase letters, digits and `_`.
   - **Capture type** follows the number of videos: 1 = single-view, 2 = two-view, 3+ = multi-view.
   - Files are used in place (never copied, never uploaded except to your own pod sessions).
2. **Videos & camera types** - one card per video, with a filmstrip and the probe facts (resolution, fps, duration,
   audio). For each video answer two questions:
   - **Mount: Handheld | Tripod.**
   - **Zoomed during the video: No | Yes** (independent of the mount: a tripod can zoom).
   The probe pre-fills a **suggestion** measured from the background motion (hatched, with a confidence). Confirm it,
   correct it, or press **Accept suggestions ≥ 70 %**. Unanswered videos use the general model (handheld + zoom).
   Untick a video to leave it out; edit the two-digit camera id if needed.
3. **Dancers** - name, stature (`5'8"`, `173 cm` or `1.73` all work), shoes (sole height), a short description that
   tells them apart, and optionally a portrait photo for the avatar's face.
4. **Dance** - level, category, song, artist, date, venue.
5. **Consent** - per dancer, click to cycle yes / no / ?. **cloud_processing = yes for both dancers is required for
   any Runpod stage**; without it the pipeline stops at the first pod stage.
6. **Review & create** - a checklist and the exact `takes/<take>.toml` that will be written (a dry run: nothing is
   written yet). **Create take** writes it, starts the probe, and opens the take's **Video Review**.

After Create, the pipeline runs the audio clocks (`audio_sync`) when you plan it; the **window** step then asks for
the analysis window (the maximum overlap of the videos is suggested).

---

## 4. Camera types (handheld / tripod, zoom)

Where you can set them (all three write the same thing):
- the wizard (step 2);
- **Videos & camera types** tab of a take (Video Review): `H` handheld, `T` tripod, `Z` toggles zoom, `A` (or
  Enter) confirms, `J`/`K` next / previous video, Space plays, `N` narrates with this camera;
- the narration tool: the camera panel under the video.

Stored in the take file, one block per video:
```toml
[cameras."01"]
mount = "tripod"     # handheld | tripod
zoom = false         # true | false
source = "user"      # user = confirmed by you | auto = a suggestion only
```

How the pipeline uses them (only **confirmed** answers change anything; a suggestion only pre-fills the form):

| Your answer | What the camera stages do |
|---|---|
| Tripod, no zoom | one pose for the whole video, localised on keyframes only (~10 % of the per-frame work), one lens |
| Tripod + zoom | fixed camera centre; per-frame rotation + focal |
| Handheld, no zoom | per-frame pose with one focal + lens distortion for the whole video |
| Handheld + zoom | per-frame pose and focal with smoothing (the general model, used when nothing is confirmed) |

- The plan dialog lists the model each video gets ("Camera types this plan applies") and the camera steps show
  their per-camera split.
- Unconfirmed videos are priced at p90 as handheld + zoom ("+n pod-min" in the plan).
- If you change a type after the camera stages ran, the screen says which stages must re-run before you save.
- The data can disagree with you (e.g. a "tripod" that moves). The card then says so, the camera stage keeps a
  safer model, and you decide. You always win; nothing is blocked.
- Single-view and two-view captures depend most on the type (a tripod single-view skips visual odometry entirely).

---

## 5. Run the pipeline

Open a take: the **Pipeline** tab shows every stage as a graph, columns = phases, in the order they run. Bold blue
is the critical path; tinted bands are pod sessions; letters say where a stage runs: **L** this PC, **G** this PC's
GPU, **P** Runpod, **H** you. Click a stage for details, its runs, its log and its actions.

**Plan & run…** (the only way a pod is ever created):
1. Pick **Run up to** (default: demo_export), **Scope** (it and everything before / only this / the whole
   pipeline), **Mode** (resume skips fresh outputs; force re-runs), and the **Unity capture name** for the export.
2. The plan shows, with no side effect: each step's action and reason, ETA p50 / p90, the pod sessions (GPU, price,
   upload size, time, cost p50 / p90, kill switch, max cost), the budget after the plan, the consent check, and the
   camera types it applies.
3. **Dry-run** logs what would run, in order, without starting anything.
4. **Run** needs the approval tick when the plan creates pods ("I approve creating n paid pods, at most $x"). The
   server re-checks the plan, the budget cap and the consent before it starts.

**What-if…** (dry-run only): the whole pipeline as if your reviews were done - the order, the wall time and the
total pod cost of a new capture before any review exists. Your human steps show as "pauses for you"; Run is
disabled.

Rules the plan follows for you:
- **One pod per job.** The front pod session (SAM, localisation, ma_2d, cameras, sync) runs all its stages together,
  so it waits until the labels review is done instead of starting a half-useful pod.
- **Soft dependencies are waited for** when they run in the same run (the export after the hair groom and the
  triggers, the narration after the move proposals).
- **The run pauses for you** at the human steps: labels review (look at the lead/follow sheet), the wrong-person veto
  (geoveto), the fit review (fit_qa) and the narration. Confirm a review gate in the stage drawer (**Looks good**,
  **Apply the veto**, **Keep as measured**, or **Reject**), then plan again.
- Stop a run from **Runs**: **Stop after this step**, or **Stop now** (cleanup still runs). A running pod session is
  never killed by a stop; delete its pod in the Runpod panel if you must.

**Resume…** plans every stage that is not done, to the end.

---

## 6. Monitor Runpod, ETA and budget

**Runpod** (top menu or `g r`):
- **Budget**: spent (from the ledger), expected and p90 of the running sessions, worst case (every running pod until
  its kill switch), against the cap ($80) and the reserve ($2). Green / amber / red.
- **Pods**: every pod on the account, read-only. A dancecap pod can be deleted only after typing the last 4
  characters of its id. **Clean up orphans** appears only when a dancecap pod has lost its runner.
- **Sessions**: every pod session (also the ones other workflows start). Click one for the live view:
  - **Now**: the current step with a progress bar (e.g. "upload 321/2166 MB at 2.66 MB/s"), and its ETA;
  - **Remaining** p50 (p90), **Cost** so far and p50 / p90 / worst, **Kill switch** time and its margin after the
    last download;
  - **Phases**: a timeline of every phase (bundle, upload, bootstrap, each job command, download, delete);
  - the runner log, live.
- The progress level says how the numbers are measured: L0 from the logs, L1 with live probes (upload, GPU).
- When a running pod job's tasks fail (e.g. ingest on the pod), an amber alert says how many failed and were skipped:
  the pod keeps billing until the job ends. Open the session and read the runner log.

The monitor never creates, stops or deletes a pod on its own.

---

## 7. Narrate the moves

**Narrate moves** (take header, or the narrate stage) starts the narration review tool
(`python -m dancecap.moves annotate --take <take>`) and opens it in its own window. Narrate by voice while the video
plays; the tool turns your words into move labels. The camera-type panel under the video sets the type of the
camera you are watching. The microphone permission is asked once.

The tool keeps running when you close its window. Stop it from **Runs** (the `tool` run, **Stop now**). Narration
needs only the clocks and the beat grid, so you can narrate while the pod work runs.

---

## 8. Export to Unity

**Export to Unity** (take header): enter the capture name (e.g. `07_LarissaKadu`), the motion and the audio.
- **Dry run** writes nothing: it lists the inputs found or missing, the folders it would replace
  (`work/<take>/export/<name>/` and the Unity `StreamingAssets/<name>/`), the Unity lock, and the classifier
  triggers the export would carry. Use it first.
- **Export** runs `demo_export` plus the triggers export. It waits while the Unity lock is held (another workflow is
  using Unity) unless you tick "export anyway".
- **Open Unity** / **Load capture** then show it in the viewer. Triggers (counterbalance, pivots, plants, moves)
  switch the matching viewer layers on while they play.

---

## 9. Where things run, and what costs money

- This PC (4 CPUs, a shared 2080 Ti): orchestration, the UI, clocks, ingest, detection, analyses, export.
- Runpod (A100 community pods, about $1.19-1.39/h): SAM masks, localisation, ma_2d, video sync, the body fit,
  texture, face and hair.
- Upload time is paid GPU time (the pod exists while the bundle uploads).
- Example: the what-if of the synthetic 4-camera, 34 s test capture says about 2.5 h wall (p90 5.5 h), 3 pod
  sessions, $3.01 p50 / $6.46 p90. Longer takes and more cameras scale the pod part roughly with camera-minutes.

---

## 10. Troubleshooting

| You see | Do |
|---|---|
| Nothing opens after a double-click | wait 20 s; then read `work\monitor\logs\launcher.log` |
| "This window must reload" | press **Reload** (the server restarted) |
| "The plan changed" | a file or a price moved: review the new plan and approve again |
| A stage says "camera types needed" | answer the camera types in Video Review |
| "cloud_processing consent missing" | **Consent…** in the take header |
| Export says the Unity lock is held | another workflow is in Unity: export later, or tick "export anyway" |
| A pod keeps running with no session | Runpod panel: **Clean up orphans** (typed confirmation) |

Command-line equivalents (from `dancecap/`, with `..\.venv-nopo4d\Scripts\python.exe -m`):
`dancecap.pipeline status <take>`, `dancecap.pipeline plan <take> --target demo_export [--what-if]`,
`dancecap.pipeline run <take> --what-if`, `dancecap.pipeline gate <take> fit_qa --decision ok`,
`dancecap.pipeline export <take> --capture <Name> --dry-run`, `dancecap.camera_types show <take>`.

---

## 11. Not built yet

- Single-view stages: `mono_camera` (camera track by camera type: tripod = static), `mono_hmr` (world-grounded
  couple pose), `couple_refine` (statures, floor, contact). A single-view take today stops before `mono_camera`.
- Two-view stages: `pair_calib` (calibration and sync from the dancers; two tripods = classic stereo) and
  `mono_init`. A two-view take today stops before the body fit.
- `scale_probe` (coarse fit to check the scale before the full fit), `cam_refine_body` (bodies refine weak cameras),
  and the move `classifier` (waits on approvals). `dancer_splats` is built (2026-10-08): an optional pod stage in
  its own session S_splats (include it in a plan; the session is priced at its approved $1.50 ceiling, the run
  halts at its review gate on review/splats/).
These show as "planned" in the graph and are never started.
