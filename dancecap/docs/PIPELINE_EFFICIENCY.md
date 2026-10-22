# Pipeline efficiency audit: from "here are the videos" to a completed analysis (2026-10-21)

Owner, 2026-10-21: "speed of analysis is more important than cost. any job that can be performed on runpod should be
placed there". This document maps what happens today for (a) a folder of multi-phone videos and (b) one handheld
monocular video, measures where the wall-clock time goes, and ranks the speedups. It is a read-and-measure audit: no
pod was created for it, and no code was changed.

Words used:
- **measured**: read from session records, farm state or notes (the source is named).
- **estimate**: derived from measured rates; not run.
- **remote runner**: `python -m dancecap remote run --jobs ...` (`remote.py`, `remote_jobs.py`). Each session builds a
  bundle, creates a pod in any data center, uploads, bootstraps, installs, runs and deletes the pod.
- **volume farm**: `python -m dancecap.farm ...` (`farm/`). Pods mount the network volume `hm-farm` (EU-RO-1, 150 GB)
  and unpack a prebuilt environment. The watcher pulls each result the moment it is done.

---

## 1. Summary

1. **Most of today's wall time is not compute.** The class recap (6 phones, 6 min) took about 9 h from the first
   step to the first Unity capture; its pod compute on the critical path was about 3.6 h. john_lula (1 phone, 88 s)
   took 8.3 h for the first pass, with about 3.2 h of compute on the critical path. The rest was agents writing
   take-specific code, handing work from one agent to the next, and waiting.
2. **There is no single command.** Every new take was driven by 5-10 agents in sequence (R1, R2, R3, F1, F2, T1 for
   the recap; J1-J5, R9 for john_lula), each with its own profile module (`recap_profile.py`, `ca_profile.py`,
   `jl_profile.py`). The generic orchestrator (`dancecap.pipeline`) still plans the older S1/S2 route, with ingest,
   detect, cam_map and the analyses on this PC.
3. **The 2026-10-19..21 round spent 55 % of its billed pod time on overhead.** Of 858 billed pod-minutes, 382 were
   compute. Upload took 226 (26 %), bootstrap 105 (12 %), tool installs 80 (9 %), pod start and delete 42 (5 %).
   All of it used the remote runner, which uploads and installs again in every session.
4. **Failures cost hours of wall time.** In that round, 5 uploads timed out (about 1.5 h of wall, $1.84). Six
   sessions landed on one broken host (60.249.37.148, "CUDA unknown error"): 0 of 6 worked, and each one cost 3-11 min
   before the next try. 20 runner starts got no pod (no GPU stock, or every GPU above the session's price ceiling).
   Each failure needed an agent to notice it and start again.
5. **The single-pod jobs are the longest compute steps.** People detection and masks for john_lula ran 82 min on one
   GPU. SAM 3D Body ran 30 min on one GPU. Physics on the recap ran 23 min on one CPU pod. All three split by time.
6. **Targets** (estimates, section 6): about **1 h 40 min** for a recap-sized folder of 6 phones x 6 min in a known
   room, and about **1 h 30 min** for an 88-s handheld video, including the realism steps and the film.
7. **The five largest speedups, in order:** one `analyze <folder|video>` command that runs the whole graph unattended;
   time-sharded parallel pods for the long single-pod jobs; prebuilt environments for every tool, run from the
   volume; every input uploaded once at t = 0 and never again; automatic avoidance of broken hosts and empty GPU
   types.

---

## 2. The two pod runners, measured

| | remote runner (`dancecap remote run`) | volume farm (`dancecap.farm`) |
|---|---|---|
| Where pods run | any data center, community or secure | EU-RO-1 only (the volume's data center), secure |
| Inputs | uploaded in every session as one bundle | uploaded once to the volume, hash-diffed |
| Environment | `bootstrap.sh` per session: median 100 s, p90 182 s, max 404 s (n = 49, 2026-10-19..21) | env tar unpacked in 16 s; container up to first job step in 28-36 s (FARM.md drill) |
| Tool install per job | 28-443 s (LHM++ 120-443 s, PromptHMR 58-400 s, SAM 3D Body 40-298 s) | none for tools in the env tar; new tools are not in it |
| Results | packed, downloaded and extracted at the end of the session | pulled per job within 15-40 s of done (FARM.md drill) |
| Used for | the whole 2026-10-19..21 tool and realism round (57 pods) | the recap, carlos_aline and john_lula v1 fits, stitches and analyses |

Measured overhead of the remote runner in the 2026-10-19..21 round (93 session dirs under
`work/runpod/sessions/2026101[9]*`..`20261021-*`, `result.json` timings and `pod.lifetime_s`):

| Phase | Billed pod-minutes | Share |
|---|---|---|
| Compute (job commands) | 382 | 45 % |
| Upload (bundle over the home uplink) | 226 | 26 % |
| Bootstrap (`bootstrap.sh`) | 105 | 12 % |
| Tool install (job `install` / `setup` steps) | 80 | 9 % |
| Pod create to ssh, delete | 42 | 5 % |
| Download and extract | 23 | 3 % |
| **Total** | **858** | (plus about 30 min on the three orphaned pods below, whose timings are incomplete) |

| Outcome (57 pods) | Pods | Cost |
|---|---|---|
| Every job ok | 28 | $6.71 |
| A job failed (tool or code bugs, mostly tool tests) | 13 | $3.64 |
| The runner stopped mid-session; the pod was removed later by orphan cleanup (ledger) | 3 | $0.37 |
| Bootstrap failed (6 of 8 on host 60.249.37.148) | 8 | $0.47 |
| Upload failed (scp timeout at 0.07-0.25 MB/s, or a dropped pod) | 5 | $1.84 |

Also: 20 runner starts got no pod (no stock, or every GPU above the session price ceiling), and 16 were dry runs.

Uploads in the round (successful, > 30 MB, n = 33): median 2.1 MB/s, p10 0.73 MB/s, worst 0.15 MB/s. Downloads over
100 MB: median 9 MB/s. One scp stream is used per upload. The recap upload measured 0.8 MB/s on 6 parallel streams
against 0.12 MB/s on one stream at a slow moment (FARM.md, R2 log entry).

What gets uploaded again and again (from the bundles' `groups_raw_bytes`):
- The MAMMA body models (130 MB compressed) in every MAMMA session: about 12 sessions in the round, about 1.6 GB.
- The 330 MB `pywheels` and 19 MB `wheelhouse` directories in every bundle by default (`remote_jobs.code_items`).
  Jobs opt out one by one with an `exclude` entry (for example `remote/cand_kimodo/pod_job.json`). The default is
  unchanged today: the card "Slim pod bundles" is not reflected in the code.
- 349 MB of JPEG frames for SAM 3D Body (`20261019-225141-batch`), where the source video is 13 MB.
- 2.5 GB of realism data for a PromptHMR run that failed after 39 s (`20261020-151313-batch`, 798 s upload).
- 2.3 GB of DiffLocks weights, twice (`20261019-165713-batch`, `20261019-173239-batch`: 992 s and 827 s).

---

## 3. (a) A folder of multi-phone videos

Reference run: the class recap `larissa_kadu_recap` (7 handheld phones, 6 used in the fits, 360 s, 1.23 GB of
source video, 2026-10-09), in the room already solved for larissa_kadu_38. Times are local clock (notes and file
times). The older route of larissa_kadu_38 (frames uploaded, S1/S2 sessions) is in PIPELINE.md 7.1; its S1 alone was
77 min of upload plus 73 min of front.

### 3.1 What happens today

| Step | Where | Measured wall | Cost | Serial / parallel | Who drives it | Source |
|---|---|---|---|---|---|---|
| Intake: owner facts (all handheld, no zoom), take TOML | agent + owner chat | part of R1 (09:15-10:27) | 0 | serial | manual | `NOTES_R1_owner_facts.md` |
| Ingest, audio sync (`recap_sync`) | this PC (ffmpeg + numpy) | "several minutes" of run; R1 done 10:27 | 0 | serial | agent | `NOTES_R1_sync.md` |
| Upload 7 videos to the volume | PC to volume | 1.23 GB at about 0.8 MB/s (6 streams); the 817 MB HEVC file took 20 min | gateway pod | serial, on the critical path | agent | FARM.md (R2), `NOTES_R2_cameras.md` |
| Cameras: one focal per phone, track every 2nd frame against the room map | 2 RTX 4090 pods | 5-9 min per camera job; 6 cameras in 15 min wall | $0.73 | parallel by camera | agent (new code `newcam.py`); R2 done 12:48 | `NOTES_R2_cameras.md` |
| People: the two teachers in every camera | pods + PC | 3 pod runs of 4-9 min; local fill / finalize | $0.35 | parallel by camera | agent; R3 done 14:35 | `NOTES_R3_people.md` |
| Chunk plan (7 windows of 54.9 s) and a 100 MB input tar | PC | minutes; tar upload at about 1 MB/s | 0 | serial | agent | `NOTES_F1.md` |
| Front per chunk: frames, SAM 2.1, ma_2d, sync, prepare | 4090 pods (inside each chunk job) | 20-38 min (1192-2254 s) | in the chunk cost | parallel by chunk; SAM 2-3 per GPU | farm | `work/larissa_kadu_recap/farm/state.json` |
| Shape pass (free betas, 30 s) | 4090 pod | about 20-25 min | $0.55 | in parallel with the fronts; every chunk then waits for it | farm | `NOTES_F1.md`, state.json `await_pin` 11-18 min |
| Chunk fits + refine + quality | 8 4090 pods | fit 27-43 min (1638-2570 s), refine 0.5-1.5 min; first fits pulled 1 h 40 min after `farm up` (14:18 to 15:41-16:00) | about $8.7 | parallel | farm; launch by agent | `NOTES_F1.md`, state.json |
| Refits of weak spans | 4090 pods | 35 min each; queued 16:03-16:15, last pull 17:01 | about $5 | after an agent read the reports | agent | `NOTES_F1.md` |
| Stitch | CPU pod | 70 s + 26 s pull | $0.01 | serial | farm | `NOTES_F2.md` |
| Analyses: timing 8 min, physics 23 min (64 GB pod), counterbalance + floorcraft 14 min | CPU pods | about 1 h (17:09-18:08), including two OOM kills of 11 min each on 16 GB pods | $0.46 | 3 jobs in parallel | agent | `NOTES_F2.md`, FARM.md (F2) |
| Export (first Unity capture) | this PC | minutes | 0 | serial | agent; F2 done 18:11 | `NOTES_F2.md` |
| Avatars (face identity, texture front + bake) | CPU pod + 4090 | identity about 4 min; texture front 601 s + bake 480 s | $1.22 | in parallel with the fits | agent; A1 done 18:09 | `NOTES_A1.md` |
| Speech (ASR, diarization) | L4 pod | 74 min (a clean re-run is about 25 min) | $0.73 | in parallel from 09:44 | agent | `NOTES_speech1.md` |
| Pose polish (boundary refits) | 4090 + CPU pods | 3 h (18:35-21:35) | $3.59 | serial review loops | agent | `NOTES_T1.md` |
| Film | this PC, Unity | about 28 frames/s: 6.5 min per 6-min render | 0 | serial | agent | `RECAP_FILM.md` |
| LHM++ splat avatars (2026-10-20) | remote runner, A6000 | install 199 s + build 124 s; session 8.5 min | $0.08 | | agent | `20261020-113940-batch` |

### 3.2 Critical path, idle gaps, manual steps

- Critical path today: R1 sync -> upload -> R2 cameras -> R3 people -> F1 fronts + shape + fits -> refits -> F2 stitch
  -> analyses -> export. About 9 h from 09:15 to 18:11.
- Compute on that path: upload about 25 min, cameras 15 min, people about 10 min, fronts + shape + fits 1 h 40 min,
  stitch 2 min, analyses about 1 h. About 3.6 h.
- The other 5.4 h: new code for this take (camera tracker, people module, profile), agent handoffs, and reviews.
- Parallel today: cameras by camera, fits by chunk, analyses by kind, speech and avatars beside the fits.
- Serial today: the agents, one after another. Every pod launch, refit and analysis was started by an agent.
- Billed idle: a chunk pod waits for the shape pass after its front (`await_pin`: 11 and 18 min on the two recorded
  first-pass recap chunks; 13-18 min on carlos_aline; 19 min on john_lula). This costs money, not wall time.
- Lost time: a shared-queue collision (two agents' pods claimed each other's jobs), the `merge_label` bug found on the
  first real pods, and two OOM kills of 11 min each (FARM.md F1 and F2 entries).
- Local work that the owner's rule moves to pods: audio sync and ingest (`recap_sync`), the people fill / finalize,
  the export packaging stays local (it writes the Unity folder), and the film stays local (Unity).

### 3.3 Other multi-view measurements

| Take | What | Measured | Source |
|---|---|---|---|
| larissa_kadu_38 (8 phones, 38 s), older route | local detect 39 min (2080 Ti), cam_map 17 min (local CPU), S1 upload 77 min of 4.2 GB frames, S1 front 73 min (resumed), S2 MAMMA 47 + 4 + 53 min (seam-gate fallback) | PIPELINE.md 5.2-5.4, 7.1 |
| larissa_kadu_full (8 phones, 188 s) | front_full: upload 2.2 GB in 18 min, front 2 h 55 min on one A100, download 4.9 GB, local extract 13.6 min; 4 segment fits in parallel, 30-60 min each | `20261007-175818-front-full`, `20261008-13*-fit-*` |
| carlos_aline (2 phones, 194 s) | front 7-10 min; shape fit 25 min; a 1548-frame chunk fit about 50 min (two cameras: 1.65 s per frame); timing SIGKILLed after 19 min on a 16 GB pod | FARM.md (C3, C6) |
| Farm drill (8 phones, 900 frames) | 4090 fit 952 s; container up to first step 30 s; pull verified 15-40 s after done | FARM.md (drill) |

---

## 4. (b) One monocular handheld video

Reference run: john_lula (one phone, 538x960, 88.1 s, 13 MB, two dancers among 3-5 couples). First pass on
2026-10-11; the realism route (SAM 3D Body prior, LHM++ avatars) on 2026-10-19..20.

### 4.1 First pass (2026-10-11): from the video to the film

| Step | Where | Measured wall | Cost | Serial / parallel | Who drives it | Source |
|---|---|---|---|---|---|---|
| Intake, ingest, beat grid (beat_this) | this PC (GPU 3 s; numpy / OpenCV) | 01:55 to 02:34 | 0 | serial | agent | `NOTES_J1.md` |
| Camera track + room | 7 small 4090 pods + this PC (6 focal solves of 30-100 s, track refinement 2 min) | single-camera pod about 11 min; J2 done 03:58 | $0.60 ($0.25 of it on failed MoGe installs) | 4 shards on one GPU | agent | `NOTES_J2.md` |
| People: detect everybody, read the bibs, SAM 2.1 both ways, RTMW whole body | one 4090 pod | 1 h 22 min | $1.21 | one pod, serial inside | agent; J3a done 04:17 | `NOTES_J3a.md` |
| Shape pass (900 frames, free betas, statures held) | volume farm, 4090 | 20 min (fit 1202 s) | $0.31 | in parallel with the chunk fronts | farm | `work/john_lula/farm/state.json` |
| Chunk fits (2 x 46 s, MAMMA_MONO) | 2 4090 pods | front 1 min, waits 19 min for the shape, fit 32-33 min | $0.79 each | parallel | farm | state.json, `NOTES_J3b.md` |
| A heading bug on the first launch | | pods deleted after 6 min, relaunch 15 min later | $0.4 | | agent | `NOTES_J3b.md` |
| Boundary refit (12 s) + stitch on a CPU pod | 4090 + CPU pod | refit 11-14 min; stitch 5-7 s | $0.2 | serial | agent | `NOTES_J3b.md` |
| Fit stage total | | 04:50 to 06:25 promoted (1 h 35 min); variants pulled 06:59 | about $4.3 | | | `NOTES_J3b.md` |
| Analyses: physics 3 min, timing + counterbalance + floorcraft 9 min | 2 CPU pods (64 GB) | 9 min wall | $0.11 | parallel | agent; J4 done 07:47 | `NOTES_J4.md` |
| Avatars (faces, ponytail, texture) | pods + this PC | J5a in parallel (done 04:55); J5b done 10:00 | about $0.5 | partly parallel | agent | `NOTES_J5a.md`, `NOTES_J5b.md` |
| Export 08_JohnLula, film render, library card | this PC, Unity | R9 done 10:13 | 0 | serial | agent | `NOTES_R9.md` |

Critical path: J1 -> J2 -> J3a -> J3b -> J4 -> J5b -> R9. **8 h 18 min** (01:55 to 10:13). Compute on that path:
camera 11 min, people 82 min, shape 20 min, fits 33 min, refit 14 min, analyses 9 min, avatar pods about 20 min, film
a few minutes: **about 3.2 h**. The rest was new code (single-camera tracker, people module, MAMMA_MONO), handoffs
and review.

### 4.2 Realism route (2026-10-19..20, "full run", $2.40)

| Step | Where | Measured | Cost | Source |
|---|---|---|---|---|
| SAM 3D Body on 2643 frames x 2 dancers | remote runner, 4090 | two tries on the broken host first (22:31, 22:39: 7.7 + 10.9 min); then upload of 394 MB of JPEG frames 146 s, bootstrap 68 s, infer 1785 s, convert 306 s (40 min session) | $0.59 + $0.24 lost | `20261019-223141/-223956/-225141-batch` |
| MAMMA_MONO + 3DB prior refit, 3 chunks | 3 remote-runner sessions in parallel (4090) | one try on the broken host (5 min) and one dropped pod (4.6 min) first; each session uploads 142-166 MB (the same 130 MB body models), bootstrap 94-242 s, fit 675-2398 s | $1.17 + $0.09 lost | `20261020-001443/-001614/-002053/-002204/-002437-batch` |
| Stitch, compare, audit, patch | this PC | stitch 01:09, compare 01:24, then **nothing until 08:19** (audit_patch) | 0 | file times under `work/john_lula/realism/refit/` |
| LHM++ avatars + renders | remote runner, 4090 | one try on the broken host first (3.5 min); install 380 s, render 143 s, download 2.4 GB 251 s, extract 53 s | $0.26 + $0.05 lost | `20261020-083919/-084328-batch` |
| Tutorial videos (encode with audio) | this PC (4 CPUs) | 388 + 433 + 66 s, about 15 min | 0 | `realism/videos/videos.json`, `videos_encode.log` |

Wall: 22:30 to 09:16, **10 h 45 min**, with an idle gap of about **7 h** (01:24 to 08:19): no file under
`work/john_lula/realism/` was written and no pod session started between 01:30 and 08:15. The notes do not record
why. Compute on the path: about 2 h. The v2 pass (facing evidence, foot locking, swinging hair) added $1.62 the same
day (`NOTES_REALISM_NEXT.md`).

### 4.3 What is serial, local or manual here

- Serial: the people job (82 min, one GPU), SAM 3D Body (30 min, one GPU), and every realism step as its own
  remote-runner session, started by an agent after the previous one was read.
- Local: ingest and beats; the focal hypotheses and the track refinement; the stitch / audit / patch of the realism
  refit; the video encodes (15 min of the 4-CPU PC, after a 2.4 GB download of frames).
- Manual: identity by bibs (checked by eye), the focal choice, promotion of the stitched fit, the choice of the
  render driver, each launch and relaunch.
- Re-uploads: frames instead of the 13 MB video; the body models in every MAMMA session.

---

## 5. Speedups, ranked by wall-clock saved

Effort: S = hours, M = 1-2 days, L = 3-5 days. Savings are estimates from the measured numbers above, per take.

| Rank | Speedup | What changes | Saved, folder (6 x 6 min) | Saved, mono (88 s) | Cost effect | Effort |
|---|---|---|---|---|---|---|
| 1 | **One `analyze <folder\|video>` command, unattended** | Classify the input (number of videos, probe camera types), write the take TOML from the probe and defaults, run the whole graph on the volume farm with a detached watcher, auto-pass gates above their confidence bars, queue boundary refits when a seam or camera gate fails, and leave the owner's checks as non-blocking review pages. One generic profile replaces `recap_profile` / `ca_profile` / `jl_profile`; the mono route (camera track, people, SAM 3D Body prior, MAMMA_MONO) becomes stages. | 4-6 h (measured non-compute 5.4 h) | 4-8 h (5.1 h first pass; about 8 h in the realism run, 7 h of it idle) | none | L |
| 2 | **Time-sharded parallel pods for the long single-pod jobs** | People / masks / whole-body pose by time shard (anchors already split the SAM runs); SAM 3D Body by frame range; physics by window with context (as `timing_free` does); chunk fronts with one GPU per camera group. | 20-40 min (fronts 20-38 min to about 12; physics 23 to about 6) | 70-90 min (people 82 to about 15; 3DB 30 to about 8) | about the same GPU-minutes, plus about 1 min of start per extra pod | M |
| 3 | **Prebuilt environments for every adopted tool, run from the volume** | Env tars on the volume for MAMMA_MONO + prior, SAM 3D Body, LHM++, gsplat, GVHMR (as the speech env: 9.1 GB, unpacked in 25 s), so no job bootstraps or installs. Or a custom image (its pull time on a fresh host is not measured). | 10-20 min (fewer sequential sessions on the path) | 15-40 min (4-6 sequential remote sessions x 3-10 min of bootstrap + install) | saves the 22 % of billed time spent in bootstrap and installs | M |
| 4 | **Upload once, at t = 0, never again** | `analyze` starts uploading the source videos to the volume before anything else: hash-diffed, resumable, parallel streams, audio tracks first (so clocks, beats and ASR can start in a minute), stream-copy trims to the window when one is known. Pod jobs decode on the pod. Body models, weights and earlier results are read from the volume. | 10-20 min (no re-uploads; the 25-min upload overlaps the first camera jobs) | 5-15 min (13 MB video instead of 349 MB of frames; no 130 MB models per session) | saves the 26 % of billed time spent uploading, and the timeouts ($1.84, about 1.5 h of wall in the round) | S-M |
| 5 | **Avoid broken hosts and empty GPU types automatically** | Before uploading: a 10-s CUDA check on the pod (torch init + one matmul); a failed check deletes the pod and adds the host to a deny-list file kept from history (60.249.37.148: 0 of 6). A wider default GPU list for jobs that fit in 24-48 GB (4090, A6000, L40S, A40, then A100). Retry in the runner, not by an agent. | 5-30 min on a bad day | 10-40 min on a bad day (the realism run lost about 30 min to 4 broken-host tries) | saves $0.4-0.5 per round | S |
| 6 | **Shorter chunks on more pods; shape pass first** | 30-s cores (or 20-s for mono) with one pod each; start the shape pass on the first clean window as soon as its people exist. Never run two MAMMA fits on one GPU (measured: 3x slower each). | 15-20 min (fits 27-43 min to about 17-20; drill: 900 frames in 16 min) | 15-20 min (mono fit with the prior 40 min for 46 s to about 23 min for 4 chunks) | more seams to check; about the same GPU-minutes | S |
| 7 | **Encode on the pod, download mp4 files** | Renders and tutorial videos are encoded on the render pod (with the audio), and only the mp4 files come home. | n/a | about 18 min (2.4 GB download + 53 s extract + 15 min local encode to about 2 min) | none | S |
| 8 | **Right-size the analysis pods** | Pick the 64 GB CPU pod from the frame count (timing and physics memory grows with the square of the take) before the first try. | 10-40 min (two OOM kills of 11 min on the recap; 19 min on carlos_aline) | 0 (an 88-s take fits) | none | S |
| 9 | **Faster GPUs** | Measured: the 4090 fits MAMMA 2.4x faster than an A100 (drill). Not measured: H100 / L40S against the 4090 on one 30-s chunk. MAMMA is bound by sequential iterations, so parallel pods (rank 6) are the stronger lever. | 0-10 min (unknown) | 0-10 min (unknown) | higher price per hour | S (one $1 test) |
| 10 | **The remaining local compute on pods** | Audio sync and ingest, the focal hypotheses, the stitch / audit of the realism refit, large extracts (13.6 min once for front_full). The export copy, Unity and the film stay local. | a few minutes; frees the PC | 5-10 min; frees the PC | cents | S each |
| 11 | **A second volume in another data center** (option) | GPU stock in EU-RO-1 is low; the recap got 8 4090s, but 20 creates failed in the latest round. A second volume (about $3.50/month for 50 GB) doubles the pool the farm can use. | only when stock is short | only when stock is short | $0.07 per GB-month | M |

What is not a speedup:
- Running several MAMMA fits side by side on one GPU (MONO4D_NOVEL_VIEWS.md 10.6: each ran 3x slower; the SDF stage
  6.5x).
- The 330 MB wheel directories in bundles: they only slow uploads (rank 4 removes uploads of them altogether).

---

## 6. Target end-to-end times (estimates)

Assumptions: speedups 1-8 are built; the volume farm runs every pod job; GPU stock gives at least 12 RTX 4090s at
once; uploads run at the measured median of 2.1 MB/s (at the p10 of 0.73 MB/s, add the difference shown); the room is
already solved (a new room adds a room-map job of about 20 min on a pod, partly during the upload).

### 6.1 (a) Folder: 6 phones x 6 min, 1.2 GB, known room

| t (min) | What finishes | Basis |
|---|---|---|
| 1 | Probe, classification, take TOML; upload starts; audio tracks uploaded | ffprobe seconds; 6 x 11 MB of 16 kHz audio |
| 3 | Clocks (audio sync) on a CPU pod; ASR starts on an L4 | `recap_sync` is minutes of CPU |
| 10 | Last video on the volume (28 at p10) | 1.23 GB / 2.1 MB/s |
| 20 | Cameras and people per camera, each started when its video landed | measured 5-9 min per camera job, 4-9 min people runs |
| 32 | Chunk fronts (SAM 2.1, ma_2d, sync, prepare), one pod per chunk, cameras split over GPUs | measured 20-38 min with 2-3 SAM per GPU |
| 50 | Shape pass (front 10 min + fit 20 min, started at t = 20) | measured 20-25 min |
| 70 | Chunk fits of 30-s cores (12 pods), refine, quality, pulled | measured 952 s for 900 frames on a 4090 |
| 74 | Stitch on a CPU pod | measured 70 s + pod start |
| 85 | Analyses in parallel (timing, windowed physics, counterbalance, floorcraft); LHM++ avatars done earlier (from t = 50) | measured 8 / 23 / 14 min; physics windowed (estimate 6 min) |
| 87 | Export of the Unity capture (local) | minutes |
| 95-100 | Film render (local Unity, 1080p) | measured about 28 frames/s |

**Target: about 1 h 40 min** (about 2 h with the p10 uplink). With one round of automatic boundary refits (35 min,
in parallel): about 2 h 20 min. Measured today: about 9 h first pass, 3.6 h of it compute. Cost: about the same GPU
work as today's $14-16 for the recap (state.json spend $16.36 including refits and polish).

### 6.2 (b) One handheld video: 88 s, 13 MB

| t (min) | What finishes | Basis |
|---|---|---|
| 1 | Probe, classification, upload (13 MB in about 7 s) | |
| 10 | Camera track and focal on pods (shards; the focal hypotheses in parallel on CPU) | measured single-camera pod about 11 min |
| 16 | People: detection, bibs, SAM 2.1, RTMW, in 6 time shards | measured 82 min on one pod |
| 26 | SAM 3D Body in 4 shards, conversion | measured 30 min infer + 5 min convert on one pod |
| 46 | Shape pass | measured 20 min |
| 70 | MAMMA_MONO + 3DB prior, 4 chunks of about 22 s, refine | measured 40 min for 46 s; scaled by frames |
| 74 | Stitch, facing repair, foot locking on CPU pods | stitch measured seconds; the others are minutes of CPU |
| 84 | In parallel: analyses (9 min), LHM++ avatars + renders + encode on the pod (about 10 min with a prebuilt env) | measured 9 min; LHM++ 143 s render + 380 s install today |
| 90 | Export and film (88 s of film in about 1.5 min) | |

**Target: about 1 h 30 min**; about 1 h 10 min without the realism steps (no SAM 3D Body prior, baked avatar).
Measured today: 8 h 18 min first pass (3.2 h compute); the realism run 10 h 45 min (about 2 h compute, 7 h idle).

### 6.3 How the targets scale

With chunk-parallel fits, the wall time is about: upload (size / 2.1 MB/s) + the slowest camera or people shard +
the shape pass (about 30 min) + one chunk fit (17-25 min) + analyses (about 10 min, windowed) + the film (about
real time, local). Longer takes add to the upload and the film; the rest stays about constant while enough GPUs are
in stock. The limit is GPU stock: 12-18 4090s at once in one data center is not guaranteed (rank 11).

---

## 7. What the dashboard must show for this (gaps found)

The owner wants uploads, analyses, cost and ETAs on one dashboard, the defrag view kept, running for every analysis,
with snapshots at progress points. Today:
- The Farm view (the defrag map) reads only `work/<dance>/farm/state.json` and `coverage.json`, plus the
  `*-fit-<seg>` sessions through `pipeline/fit_sessions.py`. The 57 remote-runner pods of the latest round
  (`*-batch` sessions) appear only under Older tools > Runpod sessions, and never on the map.
- Local steps (ingest, sync, encodes, export, film) report nowhere.
- Uploads to the volume (`farm queue`, `recap_cameras upload`) have no progress bar in the monitor (not verified
  further than reading `farm_view.py`); the remote runner's upload has one through the progress puller.
- The only image snapshot is `farm coverage --png` (the map alone). Nothing saves the whole dashboard at milestones.
- The monitor is started by hand and quits after 20 minutes without a window (`idle_exit_min` 20). It was running
  during this audit (pid 16392, started by another session); this audit did not touch it.

So `analyze` should start the monitor and the watcher itself, write every step (local and pod) into the one state
file the Farm view reads, and save a dashboard PNG at each milestone (upload done, cameras, fronts, each chunk
pulled, stitch, analyses, export, film).

---

## 7b. Built on 2026-10-21 (after this audit)

Code: `pipeline/analyze.py`, `pipeline/activity.py`, `pipeline/snapshots.py`, the monitor (`/api/activity`,
`/api/analyses/<run>`, `/api/snapshots/...`, views `analysis.js`, `snapshots.js`, the Farm page strip), `remote.py`,
`remote_jobs.py`, `remote/bootstrap.sh`. Tests: `tests/test_analyze_monitor.py` (17). User guide: PIPELINE_MONITOR_GUIDE.md
0.1.

| Audit item | What exists now | Status |
|---|---|---|
| Rank 1: one `analyze` command | `python -m dancecap.pipeline analyze <folder\|video>`: classifies, probes, creates or reuses the take, prints the route plan (every heavy step on Runpod, pods side by side, p50 / p90 minutes and $ per step, critical-path ETA, total cost), runs the generic steps and tracks the rest | partly: the steps with take-specific code today are marked *needs an agent*; an agent drives them tagged `DANCECAP_ANALYSIS` / `DANCECAP_ANALYSIS_STEP` and marks them with `analysis step`. Making them generic is still the L-effort item |
| Rank 2: independent jobs on parallel pods | `remote run --parallel`: jobs not linked by `after` run as separate sessions on separate pods at once (`remote_jobs.components` / `run_parallel`) | built, measured below |
| Rank 4: upload once at t = 0 | `analyze` uploads the source videos to the volume (`src/<take>/`, the farm's layout) through a CPU gateway pod it creates and deletes; the runner's bundle upload is striped over 4 resumable ssh streams (`farm.volume.Link.push_file`) with an exact byte counter | built, measured below |
| Rank 5: broken hosts and empty GPU types | a CUDA check (nvidia-smi + one torch matmul) on every new pod before the upload; a deny list (`work/runpod/bad_hosts.json`, seeded with 60.249.37.148: 0 of 6) that deletes a pod landing on a denied host at once; a job may declare `min_vram_gb` to get the wider GPU ladder (4090, L40S, RTX 6000 Ada, L40, A100, H100, A40, A6000, 3090; no Blackwell: torch 2.5.1 / cu124 has no sm_120) | built, measured below |
| "Slim pod bundles" | the 330 MB pywheels and 19 MB wheelhouse are bundled only when a job needs a bootstrap feature beyond base/check (or asks with `bundle_pywheels`); a tool job's bundle is code + its inputs | built (a probe bundle: 45.8 MB = 12.5 MB code + 42 MB input, no wheels) |
| apt stall in the bootstrap (found again live) | `remote/bootstrap.sh`: the image's deadsnakes PPA source is dropped before `apt-get update` (it answered 503 and stalled one bootstrap 860 s), apt gives up on a dead mirror after 20 s, and the update is skipped when the packages are present | built, not yet measured on a pod |
| Dashboard gaps (section 7) | budget = `remote.spent()` exactly ($100.95, was shown as $530.57); the cap the runners use; every pod session on the Farm page with upload MB / MB/s / time left / streams / rate sparkline, phase, ETA, cost so far / expected / worst; per-analysis step map, end-to-end ETA and cost; honest session states (skipped job = failed, runner gone = orphaned); stale farm pods hidden; runner pods labelled | built |
| Monitor always running | `remote run`, `pipeline run`, `analyze`, `farm up / launch` start it headless (1.4 s cold, 15 ms when up) and lease it; no idle exit while a lease's process lives, a pod is open or an analysis step runs | built, measured |
| Snapshots at progress points | `work/pipeline/snapshots/<run>/NN_<event>.png + .html + index.json`, rendered by a detached process from a static page (headless Edge 1.8-4.7 s each, measured 2026-10-21 14:26-14:34), at analysis start / step ends / driver end / step marks / 15-min heartbeat and at upload done / bootstrap / each job / session end of every pod session | built, measured |
| Prebuilt environments on the volume, a network-volume cache for the runner | not built: the remote runner places pods in any data center, the volume binds them to EU-RO-1 where GPU stock is the limit (rank 11). The farm already runs from the volume | open |

Measured in the live test of 2026-10-21 (run `20261021-142636-john_lula`, a test run on john_lula, cancelled after the
test; sessions `20261021-142911-batch-p1/-p2`; no take data on the pods, a 42 MB random input):
- `analyze ... --yes`: intake 2 s; the volume upload step 58 s in all (gateway CPU pod up in 37 s, 13 MB pushed in 18 s,
  gateway deleted).
- `remote run --parallel` with two independent jobs: two pods started at the same time. The first pod landed on the denied
  host 60.249.37.148 (an L40S) and was deleted before any upload; the session then got a 3090 elsewhere. Both pods passed
  the CUDA check.
- Striped upload, 4 streams: 45.8 MB in 28.2 s and 45.3 s (two uploads at the same time; 1.95 and 1.57 MB/s each, about
  3.5 MB/s together). The review probe of the same morning, one scp stream: 22.3 MB in 47.8 s (0.47 MB/s). Different hosts
  and times: one comparison, not a benchmark.
- Bootstrap base on the 4090 (secure): 154 s. On the 3090 (community): 557 s, after the same deadsnakes PPA 503 in
  `apt-get update` (the fix above was written during this test and was not in its bundle; how much of the 557 s it saves
  is not measured yet).
- Session lifetimes: 266 s (4090) and 700 s (3090); each probe job about 47 s. Pod cost of the whole test: $0.116 (gateway
  $0.001, the denied-host L40S $0.006 for 29 s, the 3090 $0.043, the 4090 $0.066); every pod deleted, every ledger row
  closed; `remote.spent()` went from $100.95 to $101.07.
- Snapshots: 17 for the run, all rendered by headless Edge in 1.2-4.7 s each, while the sessions ran.
- The monitor: auto-started in 1.4 s, reused in 15 ms; the Farm page showed both live sessions with their upload (MB, MB/s,
  4 streams, sparkline), phase, ETA and cost, a stall warning for the slow 3090 bootstrap ("no progress for 6 min"), and
  the analysis with the agent-driven step as running.

---

## 7c. Round 2 (2026-10-21): fixed and measured

Three groups (runner, plan, dashboard) fixed the review's gap list in parallel, each checked by an adversarial verifier;
the integrator applied the cross-group hand-offs and the verifiers' blocker / major findings. Code: `remote.py`,
`remote_jobs.py`, `remote/bootstrap.sh`, `cameras/newcam.py`, `pipeline/{analyze,activity,runpod_view,snapshots,estimate,
farm_view}.py`, `pipeline/monitor/`, `progress_eta.py`. Tests: `tests/test_runner_round2.py`, `test_analyze_round2.py`,
`test_dashboard_round2.py` (new), `test_analyze_monitor.py`. User guide: PIPELINE_MONITOR_GUIDE.md 0.1, 0.2.

**What changed**
- Runner pods get ffmpeg + ffprobe (a static 8.1 build fetched in the background while base installs, apt as the fallback)
  and open3d whenever a job decodes video or reaches physics, without declaring them and without fattening slim bundles.
- The static ffmpeg fetch has one 150-s deadline, drops a URL that stays under 1 MB/s for 15 s, never runs twice on one
  pod (a pid file for a retried bootstrap) and never repeats a failed fetch in the foreground.
- `cameras/newcam.FrameStream` ends with the error on every failure (the trial's 10-min hang), and a window that decodes
  short of the PTS table warns, or fails when ffmpeg logged decode errors and 10 % or more is missing.
- `remote run` without `--gpus` tries the fastest card with enough memory first (4090 for the 24-GB default; 80 GB for
  room, CoTracker3, `mamma_chunks fit`, `dancers_pod run` and `front_pod run`; 48 GB for a single MAMMA fit over 3000
  frames), each GPU in every cloud before the next GPU.
- A (GPU, cloud) pair that led to a broken host waits behind the next GPU type only, and that demotion lapses 3 days after
  the host was last met; the host itself stays denied.
- Every pod a session creates is in its GPU choice (`tried[*].pod_id`), and a GPU over the session max or the remaining
  budget is skipped with a NOTE naming the faster card and the `--max-cost` that would allow it.
- A frame counter per pod command (`progress/frames.json`) reads what jobs already print (SAM 3D Body chunks, MAMMA
  iterations relayed live from its log, tqdm); a second or nested bar, a MAMMA early stop, reused SAM 3D Body chunks and a
  runner that died mid-command are counted correctly, and a rate 2x the 4090 reference is a slow-GPU warning.
- `analyze` plans what is built: one pod per mono step plus the measured 6.3 / 15 min session start-up, side-by-side lanes
  sharing the home uplink for the 1.6 GB checkpoint, and at most 6 runner pods at once.
- A running step past its estimate pushes the ETA out, a session ETA only pushes out and one built from a timeout is
  ignored, and the older estimator (`estimate.live_blend`) follows the same rule.
- `--max-cost` is the run's pod limit: the driver starts no pod step past it, every session tagged with the run is capped
  at what is left (none starts at 0), and `analysis limit` / `analysis resume --yes` un-park the steps they held.
- ELAPSED stops at the run's end and `cancel` keeps an earlier end; a builtin failure beside a running agent step keeps the
  run open; the run's cost counts every pod its sessions created and is kept in analysis.json once the run is finished.
- One driver per run, snapshots only at progress points, and a heartbeat process that snapshots every 15 min while the run
  moves and no longer runs for ever on a step an agent left "running".
- The dashboard fills the step map only from measured progress (striped squares = time against the plan), draws reused
  steps apart, shows ended uploads as stopped, says why a session got no pod (and no longer says it of a session still
  starting), lists pods deleted inside sessions, and raises one alert per slow GPU.
- Volume-farm runs take snapshots at the farm's real chunk events; the Farm page shows a runner-only take's analysis; the
  GPU list shows the runner's ladder; ended sessions stop counting their duration.

**What was measured**
- Live probe through `remote run` (session `20261021-194737-batch`, $0.056, pod deleted): 4090 community had no instance,
  4090 secure was next, and the broken L40S community pair was announced as tried after the next GPU type.
- On that pod: base 74 s, the static ffmpeg fetched in 9 s behind base (0 s of feature time), open3d 25 s, import check
  ALL OK, bootstrap 106 s in all.
- On that pod: FrameStream decoded 60 of 60 frames and a missing file raised after 0.04 s.
- On that pod: the relayed MAMMA iterations and the SAM 3D Body chunk lines gave 6 frame events in all (not one per line),
  600/600 frames, and one slow-GPU warning (the probe's rate is fake on purpose).
- The runner group's two earlier probes ($0.164): the static fetch took 24 s and 101 s, open3d 36 s and 51 s, and the
  relayed MAMMA lines arrived live after the mawk fix (0.503 s per iteration, as written).
- Trial replay with the new plan and ETA rule: at 15:11 p50 57 / p90 132 min against 84 actually left (the page said 18);
  cost p90 $2.78 against the $2.26 spent (it said $1.73).
- The new plan for an 88-s mono video is 4.3 h p50 (p90 8.8 h, $9.95 p50) and for the 20-s trial 2.0 h ($2.20); the 88-s
  figure is dominated by 5 fit sessions uploading 5 x 1.6 GB through one home uplink.
- The trial run's cost is $2.2601, every ledger row of its sessions including 4 broken-host pods ($0.0318), and the page
  explains the $0.03 between the session rows and the KPI.
- The monitor restarted through `/api/restart` three times (new pid in 3.3-6.3 s) with no new Edge window; #/farm,
  #/analysis/20261021-145414-trial_mono_jl20, #/snapshots and #/runpod were checked in the browser pane.
- Tests: 291 passed in the main set from the workspace root and from `dancecap/` (test_remote, the three round-2 files,
  test_analyze_monitor, test_progress, test_pipeline_*, test_newcam, test_pod_detach), and the 14 farm files 207 passed
  with 1 skipped from both, after one order-dependent farm test was fixed (below). One run of the main set beside the farm
  run lost `test_pipeline_fixes::test_stop_now...` to load (its 90-s wait); it passes alone (8.6 s) and with its file.

**What is still open**
- The agent steps (camera, people, SAM 3D Body, fits, analyses, avatars; cameras and people for a folder) still need
  take-specific code, so `analyze` is a planner, tracker and uploader, not "video in, analysis out".
- Every runner session still uploads its own bundle, so the 1.6 GB ma_2d checkpoint goes up once per fit pod; putting it on
  the volume (or a prebuilt image) is the biggest single saving left in the plan.
- `front_full.py` still installs apt ffmpeg on top of the bootstrap's static build, `cli.py`'s `--gpus` help still names
  `GPU_PREFERENCE`, and `progress_pull.py` still tails the MAMMA log under the job's take; those files were outside this
  round's scope.
- The runner ladder checks GPU memory only; host RAM and vCPUs (dancers_pod, front_pod) are covered by giving those jobs
  the 80 GB cards, not by a check.
- MAMMA's frame counter gives progress within a run but no whole-fit ETA, and only SAM 3D Body has a reference rate for
  the slow-GPU warning.
- SAM 3D Body prints one line per 200-frame chunk, so its first rate (and a slow-GPU warning) comes one chunk late.
- Cost attribution of a pod a session deleted itself is exact only for sessions run with this round's runner (pod ids in
  the GPU choice); older sessions are matched by label, kill time and start window.
- The tagged-session cap, the shared-uplink plan and the un-parking were checked with unit tests only; no analysis with a
  pod limit has run live yet.
- The snapshots page alone shows "connecting..." in the top bar until another view has polled the server (cosmetic,
  app.js).
- `test_farm_watch_ops::test_down_abandon...` failed whenever `test_farm_watch.py` ran first because that file leaves
  `FlakyLink.cut` at 3000; the test now sets it itself, which is a test-only change.

---

## 7e. Generic round (2026-10-22): one video in, a finished analysis out

The round-2 verdict was "a good planner, tracker and uploader, not video in / analysis out": camera, people, avatars and every
mono job spec were take-specific. This round makes every mono step a generic executor and fixes the four dashboard leftovers of
the round-2 final review. Code: `pipeline/mono_route.py` (new), `people_mono.py` (new), `cameras/mono_cam.py` (new),
`pipeline/{analyze,activity,snapshots,farm_view}.py`, `monitor/static/components/stepmap.js`, `mono4d/jl_sam3db.py`,
`jl_front.py`, `jl_export.py`, `jl_lowconf.py`, `heat_beats.py`, `remote/cand_sam3d_body/decode_frames.py`. Tests:
`tests/test_analyze_generic.py` (15, new), `test_people_mono.py` (8, new), `test_mono_cam.py` (8, new), `test_heat_beats.py`
(+1), the plan tests of `test_analyze_round2.py` / `test_analyze_monitor.py` updated to the generic plan, and
`test_dashboard_round2.py`'s no-history ETA test, which failed on every day but 2026-10-21 (a hard-coded session id
dated the runner log's clock times), now builds its session ids from the clock (test-only).

**What each mono step runs now** (`python -m dancecap.pipeline.mono_route <step> --take T --run R`; the driver runs it as a
`cmd` step, its `RESULT:` line becomes the step's message):

| Step | Where | What | Replaces |
|---|---|---|---|
| beats | this PC | `dancecap beats` (the video's own sound) + `heat_beats` (the grid); a bass-less phone clip that is music all through now gets its grid from the clearest band | the trial's hand-added TOML + local run |
| camera | 1 pod | `cameras.mono_cam pod`: PTZ skeleton (every fps/2-th frame) + per-frame track in 2 shards + MoGe-2 on 14 wide frames | `room_single_cam` (john_lula constants) / the parent's solve |
| people | 1 pod | `people_mono run`: yolo11x-pose, the couple without bibs (tracklets, dancer-likeness, colour self-training), lead / follow from `people/roles_override.json` > the TOML descriptions > the statures, SAM 2.1 both ways, RTMW, a second round where the runs disagree | `jl_people` (bibs 124 / 116, zoom table) / the parent's tables |
| world | this PC | `cameras.mono_cam world`: focal from MoGe-2 (wide frames), gravity from MoGe-2 floor normals through the track, camera height from the dancers' head / feet rays and statures, origin at the couple, Z towards the camera -> `cameras/` | J2's `room_single_cam solve` |
| sam3d_body | 1 pod | `jl_sam3db` configured for the take: frames decoded ON the pod from the video (no JPEG upload), 4090 first, mask thresholds scaled to the frame; import, `qa_nofit`, `prior_nofit` here | `trial_sam3db.py` glue (constants overridden, T = 599 hard-coded) |
| fits | 1 pod per chunk of <= 26 s, side by side | per chunk: the farm job json + chunk TOML + prior slice written here; pod: the chunk front (frames, cameras, masks, ma_2d, sync, prepare, mono-init, qa-mask), `fit-build` (MAMMA_MONO configs + prior), the fit with the betas PINNED from the statures (beta0 solved for the rest height: no shape pass), select, refine | `trial_front.py` / `trial_fit.py`, the parent's pinned betas |
| post | this PC | the stitcher over the chunks (the per-frame flags the export needs; the trial's copy lacked them) + `jl_repair` | the trial's copy |
| analyses | 1 GPU pod | timing, physics, counterbalance, floorcraft (base env + open3d + the locked-head model: no mamma bootstrap) | `pod_job_analyses*.json` by hand |
| export | this PC | `jl_export --take T --no-install` -> `work/T/export/Mono_<Take>/` (the Unity project is never touched) | `trial_export.py` |
| avatars, film | - | skipped with their reason (LHM++ reference crops for a new take are not written; the film needs the Unity editor) | |

Every pod step needs `--yes` and runs as one tagged `remote run` (DANCECAP_ANALYSIS / DANCECAP_ANALYSIS_STEP): a session that
failed for pod reasons (no pod, upload, bootstrap, ssh, a network install) is started again, up to 3 times; a command that failed
on its own fails the step with its log tail. Only the sessions that run this call's jobs count (the first pod check took a
sibling workflow's "no GPU pod" session for its own).

**Dashboard leftovers (round-2 final review), fixed**
- **Run ETA vs step ETA**: a live session ETA from measured progress that covers the whole step (one session per lane, every lane
  live, no generic prior, no stall) IS the step's time left, in or out; anything else can only push out; a timeout-based one is
  ignored. The step row shows the time left the run ETA counts. The review's 20:59:02 case now reads 212 s / 350 s (it read
  396 s / 1188 s; 127 s were left; the row said 92 s).
- **Snapshots at their event**: `request()` captures the monitor's data (`gather`, about 0.1 s warm) into
  `<run>/pending/<ms>_<event>.json` and the detached renderer draws that file; the index has `ts` (data time), `rendered`, `frozen`.
- **Badge**: "running" while any step runs (driver end status, the analysis card, the Farm page line).
- **pod-cpu**: the mono plan has no pod-cpu step left: beats, world and post run here (seconds to a minute of CPU), the analyses
  on a GPU runner pod (`remote run` has no CPU route; a CPU route is the ready-pods session's work).
- The multi-view steps that still need an agent say exactly why (`MULTI_BLOCKERS`).

**Validated offline before any pod** (no Runpod cost)
- `people_mono` on john_lula's real detections, no bibs: torso joints inside the right truth mask 88.2 % (lead) / 79.9 % (follow)
  against 85.5 % / 79.5 % for the bib-based `jl_people.solve`; lead / follow by the descriptions, confidence 0.999; anchors 25.9 s
  wall for 2643 frames.
- `mono_cam world` from J2's john_lula intermediates: focal 806 vs 808 px, gravity within 0.015 deg, camera height 1.44 m vs 1.43
  (J2's convention), J2's origin reprojects 0.66 px median from ours; 188 s wall for 2643 frames.

**Pod checks on the trial clip, step by step** (lk03_mono20 = a fresh 20-s cut of `Larissa-Kadu-demo/03-Demo-Mao.mp4`,
source 34.6-54.6 s, 480x848, 30 fps, 599 frames, 11 MB; not john_lula):

| Step | Result | Wall | Pod cost |
|---|---|---|---|
| camera (3 tries) | try 1: a duplicate keyword in `mono_cam.record()` (fixed); try 2: MoGe-2 needs `huggingface_hub` (its weights come from the Hub; the `--no-deps` recipe skipped it; fixed); try 3 ok: skeleton 0.85 px median on 40 frames, track 0.96 px median / 1.17 p90 over all 599 frames, none bad, MoGe-2 on 14 frames | session 430 s: bundle 2 s, upload 13-147 s (the home uplink was shared), bootstrap 64-144 s, installs 30 s, probe 3 s, skeleton 22-38 s, track 75-198 s (2 shards), MoGe-2 9-12 s | $0.10 (+ $0.36 for the failed tries and one wrong retry, below) |
| people | ok on the first try: masks of both dancers on 599 / 599 frames, both pose-confident (>= 0.5) on 74 %, the right dancers in every review frame (seated and standing bystanders ignored), lead / follow by the TOML descriptions (confidence 0.998) | session 535 s: detect 30 s, couple + anchors 4.5 s, SAM 2.1 82 s, fuse 8 s, RTMW 38 s, second round 126 s, finalize 7.5 s | $0.13 |
| world | focal 780 px (MoGe-2 wide median 779), camera height 1.76 m, the couple 5.0 m away (p5-p95 4.25-6.13), track residual 0.97 px; the review grid lies on the floor, the stature poles end at the heads | 45 s on this PC | - |
| sam3d_body | ok on the first try: prompted 551 / 564 of 599 frames, QA-ok 547 / 560, prior on 547 / 560 of the 599 lattice instants | step 887 s: session 856 s (upload 28 s, bootstrap 102 s, venv install 88 s, decode ON the pod 2.8 s, infer 513 s = 0.46 s per person-frame, convert 64 s), import + QA + prior here 30 s | $0.21 |
| fits (2 tries) | try 1: the job excluded `mamma/data/weights/ma_2d/mamma_mask_full_cvpr.ckpt`, the very path the stripped checkpoint is mapped to, so ma_2d found no weights (fixed; a test guards it); try 2: the whole chunk front passed (frames 4 s, cameras 3 s, masks 3 s, ma_2d 28 s, sync, prepare 3.5 s, mono-init 3 s, qa-mask 3 s: 62 s), then `fit-build` refused the chunk: the cut's first frame is at 0.033 s and the window ends at 19.93 s, so prepare kept lattice instants 1..597, not the planned 0..598 (fixed: the chunks are planned on the instants that have frames inside the window, and fit-build follows prepare's grid; checked here on the pod's prepare.json) | try 2: upload of the 632 MB bundle 860 s (0.74 MB/s: the uplink was shared), bootstrap (base + mamma + ma2d) 262 s | $0.12 + $0.31 |
| analyses (scratch copy of the 2026-10-21 trial's motion) | ok on the first try with the base env + open3d + the locked-head model (no mamma bootstrap): timing 25 s, physics 22 s, counterbalance 8 s, floorcraft 8 s | session 668 s: upload of 127 MB 320 s (0.40 MB/s), bootstrap 216 s, commands 70 s | $0.16 |
| post, export (scratch copy) | the stitcher over one chunk + jl_repair: 6 s, gate passed; `jl_export --take --no-install`: 25 checks passed, 0 failed, 5 s | this PC | - |

The wrong retry: the step took a sibling workflow's concurrent session (the ready-pods session's CPU-route test, "no GPU pod") for its
own and started the camera session again although its command had failed on its own. Fixed: a step counts only the sessions whose
`runner.json` lists its jobs.

{FINAL_RUN}

**Folder trial** (run `20261022-020110-trial_mv_lk15`: the 8-phone, 15-s stream-copy cuts of larissa_kadu, 76 MB, a known
room, `--room-known --yes`). It does **not** reach done; the exact blocking step is `cameras` (and `people` beside it):
- unattended: intake 2.1 s (the take reused), the volume upload 95 s (a CPU gateway pod; the 8 files were already on the volume:
  hash check only), the audio clocks 8.4 s on this PC; the driver then parked the rest and the run read "waiting" 2 min 16 s
  after it started, pod cost $0.0007;
- `cameras`: "no generic multi-phone camera executor: a shared world frame needs every phone localised against a solved room map,
  and that code is take-specific (cameras/newcam.py WORK paths, recap_cameras for the recap); cameras.mono_cam solves one phone in
  its own frame only";
- `people`: "no generic multi-view people executor: the dancers are picked by triangulating across phones (recap_people,
  carlos_aline_people are take-specific); people_mono handles one phone";
- behind them, fronts / shape / fits / stitch / analyses wait on the farm's per-take profiles (recap_profile, ca_profile,
  jl_profile) and the export on recap_export / carlos_aline_export. Each waiting step now carries that reason in its message.

**Still open**
- **Multi-view folders**: no generic executor for cameras (a shared world frame against a solved room map), people (triangulated
  dancers), the farm's chunk profiles, the multi-view analyses and export: a folder runs intake, upload and clocks, then waits with
  the exact reason (above).
- **The fit session uploads ~630 MB** (the 540 MB stripped ma_2d checkpoint + the locked-head model) through the home uplink, in
  every chunk session: 860 s at 0.74 MB/s today, and the pod bills while it waits. K chunk lanes share the one uplink. The
  ready-pods session (the volume route, slim inputs) is the fix; until then the fits step is upload-bound on a slow day.
- **Bootstraps of 100-320 s** per session (base, mamma, ma2d features from PyPI / wheels); the same ready-pods work (uv installs
  measured at 11-14 s on 2026-10-22) is the fix.
- **LHM++ avatars for a new take**: the reference crops (chosen from the masks and the fit) are not written, so the step is skipped
  and the export carries plain bodies. The film needs the Unity editor and is skipped with that reason.
- **people_mono on a crowded heat** (john_lula, 3-5 couples, no bibs): both roles right on 63.5 % of the frames by box IoU (the
  bib tracker scores about the same on the torso-in-mask test); a follow hidden behind the lead is the main miss. On the demo
  clip (one couple, a seated audience) the review sheets are right in every sampled frame.
- **Camera models**: a fixed-centre phone only (pan / tilt / zoom); a walking phone now gets a warning in the world step
  (background residuals over 2 px / 4 px), not a solver. A video with a rotation tag is refused with the ffmpeg command for an
  upright copy. The focal is MoGe-2's (+-10 %): distances scale with it, the camera height does not.
- In a git worktree whose `dancecap/work` is a junction, intake writes `source_dir` through the junction (five `..`); harmless on
  the pods (the video is mapped), and the main checkout has no junction.

---

## 8. Sources

- Session records: `dancecap/work/runpod/sessions/*/result.json`, `state.json`, `runner.log`,
  `bootstrap_timings.tsv` (2026-10-06..21); ledger `dancecap/work/runpod/ledger.json`.
- Farm state: `dancecap/work/{larissa_kadu_recap,carlos_aline,john_lula}/farm/state.json` (per-chunk step seconds,
  `await_pin`, cost).
- Notes: `dancecap/work/larissa_kadu_recap/NOTES_{R1_owner_facts,R1_sync,R2_cameras,R3_people,F1,F2,A1,T1,speech1}.md`;
  `dancecap/work/john_lula/NOTES_{J1,J2,J3a,J3b,J4,J5a,J5b,R9,REALISM_NEXT}.md`; file times of
  `dancecap/work/john_lula/realism/`.
- Docs: `PIPELINE.md` (5, 7), `PIPELINE_PROGRESS.md` (1.1), `FARM.md` (change log: drill, R2, F1, F2, C3, C6, J3b, J4),
  `MONO4D_NOVEL_VIEWS.md` (10.6), `PIPELINE_MONITOR_GUIDE.md` (0, 1).
- Code read: `dancecap/dancecap/remote_jobs.py` (`code_items`, `bundle_files`, `run_session`, `_pod_steps`),
  `remote.py` (`pod_session`, `GPU_PREFERENCE`), `pipeline/stages_draft.json`, `pipeline/farm_view.py`,
  `pipeline/fit_sessions.py`.
- The aggregation scripts for section 2 ran on this PC over the session JSON files (seconds of CPU); they are not
  part of the repo.
