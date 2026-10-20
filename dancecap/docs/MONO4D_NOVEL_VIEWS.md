# One phone -> 4D provider -> virtual cameras -> MAMMA: findings and test plan

**Status 2026-10-15.** The owner asked for "a monocular video to the api, get a 4d gaussian, and create novel camera
angles so we can run MAMMA", with Gracia (gracia.ai) and World Labs Atlas (worldlabs.ai/blog/atlas) as the
candidate APIs. This file is the decision record: what the two vendors offer today, what was built
(`dancecap.mono4d`), what was tested, and the next steps. Numbers marked *measured* come from runs on 2026-10-15.
Numbers marked *stated* come from the vendors' own pages, read that day; *estimate* means reasoned, not measured.

> **Update 2026-10-19 (section 10).** The owner now has Atlas beta access and widened the question: "highly
> realistic 3D moving dancers with correct poses and natural appearance, clothes, hair ... from monocular video
> (handheld phone)", with Atlas, Unity Learned Motion Matching, NVIDIA Kimodo, Dense4D and Viggle JST-2 as
> candidates ("there are several other new tools that might do a better job"). Section 10 holds:
> - live measurements of Atlas and Viggle on the 38-s take;
> - a TypeSafe judge calibrated on real frames;
> - a verified research sweep, including stronger released tools than the five named;
> - the scoring harness every candidate now reports to;
> - the pod tests of the candidates.

---

## 1. Summary

1. **No vendor offers the requested chain today.** No API takes one phone video of moving people and returns a 4D
   Gaussian or renders at our cameras.
   - **Gracia** makes production 4DGS (volumetric video) from **genlocked, hardware-synchronised multi-camera rigs**.
     Their FAQ says a full 360-degree capture needs about 60 cameras.
     - It is a processing-only service through capture partners, with delivery in 3-5 days.
     - Processing costs $800 per minute (streaming/VR tier) or $2,500 per minute (VFX tier). Capture is separate,
       about $15-20k per day.
     - Their SDKs are good: the Python SDK renders a `.mint` 4DGS file to colour + depth at any camera.
     - A phone video is outside their stated workflow. They position themselves on reconstruction without
       hallucination, the opposite of monocular generation.
   - **World Labs Atlas** (announced 2026-09-01) is in **early access with select partners**, through a request
     form. There is no public API, model id or pricing.
     - Stated capabilities: camera-controlled generation at any camera pose you specify (video up to 1 minute at
       1440p), and reconstruction into point clouds / static 3D Gaussian splats.
     - Reframing of footage is stated "with footage from as few as three cameras", not from one.
     - It does not claim dynamic (4D) output or people in motion. It says plainly that it imagines the regions it
       did not see.
   - **The World Labs World API** (public, the Marble models) generates **static** worlds.
     - Video input is at most 30 s and 100 MB, and must be of a static space; the guide says to avoid moving people.
     - A world costs 1,500 credits ($1.20). Outputs: SPZ/PLY splats, a GLB mesh, and metric-scale metadata.
     - It is useful at most for a room model, never for the dancers.
2. **The views would not be measurements.** Every pixel a generated view shows that the phone did not see comes from
   the provider's prior. MAMMA triangulates whatever it is given. So the route can at best reproduce the provider's
   own 3D guess of the dancers, including invented limbs behind the partner.
   - Whether that guess beats our one-camera fit (`dancecap.jl_mono` + MAMMA_MONO) is an empirical question.
   - The question can be answered on a take we already filmed with 8 phones: give the provider one phone, compare
     with the 8-phone fit (section 6).
3. **Built and tested, vendor-independent (`dancecap.mono4d`):**
   - request packaging (input cut, virtual camera ring, held-out phone poses, trajectories in OpenCV + OpenGL axes);
   - a provider layer: manual drop (the only route to Atlas today), the World API client, Gracia `.mint` + renderer;
   - frame alignment;
   - a virtual take that MAMMA's own loader accepts;
   - hold-out metrics and a quality gate;
   - two TypeSafe (Jev) judgment modules: a vendor readiness watch and a visual check of generated frames.

   Unit tests: 23 pass. A live end-to-end smoke on `larissa_kadu_38` used a stand-in "oracle" delivery.
4. **Recommended next steps (section 9):**
   - The owner applies for Atlas early access, describing this exact use.
   - A $1-3 Runpod spike with a self-hostable monocular novel-view model (TrajectoryCrafter first) on the same 4 s
     window, scored with the protocol.
   - The one-camera baseline fit of that window.
   - The owner's go-ahead to calibrate the visual judge on real frames.

---

## 2. What each vendor states (2026-10-15)

| | Gracia | World Labs Atlas | World Labs World API (Marble) |
|---|---|---|---|
| Access | Inquiry form; processing via capture partners; free 1-2 min test for new partners | Early access for select partners (form) | Public, self-serve (platform.worldlabs.ai, key header `WLT-Api-Key`) |
| Input | Genlocked multi-camera rigs (iPhone 17 genlock, Blackmagic, Sony FR7; not GoPro/Insta360); ~60 cameras for 360 | Text, images, **camera poses**, depth maps; video as image sequences | Text, image(s), panorama, video <= 30 s / 100 MB of a static space |
| One phone of moving people | No | Not stated (reframing stated from >= 3 cameras) | No (keep the space static, avoid moving people) |
| Dynamic (4D) output | Yes: `.mint` (temporally stable 4DGS) or a PLY sequence; 100-125k splats/frame streaming, 200-300k VR, up to 1M VFX | Not stated (video out; static splats / point clouds) | No |
| Camera control | Free viewpoint at playback (SDKs) | Yes, any specified camera pose | No (we render the exported splat) |
| Export / SDK | `.mint`/PLY; Unity (Asset Store), Web, Python (`graciasdk`: render to NumPy colour + depth), C/C++ (Vulkan), Apple (Metal) | Not stated | SPZ (100k/500k/full), PLY export, GLB mesh; arbitrary units + `metric_scale_factor`, `ground_plane_offset` |
| Price | $800/min streaming/VR, $2,500/min VFX, minimum per segment; capture $15-20k/day | Not stated | $1.00 = 1,250 credits; marble-1.1 = 1,500 credits ($1.20) |

Sources: gracia.ai, gracia.ai/4dgs (FAQ: its JSON-LD block), gracia.ai/data, github.com/gracia-labs/python-sdk,
worldlabs.ai/blog/atlas, docs.worldlabs.ai (llms.txt, api/index, api/pricing, api/faq, marble video prompt guide,
OpenAPI spec). The Python SDK README lists `graciasdk` as CPython >= 3.12 with Vulkan on Windows/Linux, local files
only (streaming via the Apple/C SDKs with a Gracia token).

**Readiness watch (TypeSafe Jev, measured).** `python -m dancecap.mono4d vendors` asks the vendors' own pages, per
capability, three things: whether they state it, whether they rule it out, and which passage is the evidence. Code
then decides yes / no / unclear. Run of 2026-10-15 (jev-1.13.0, about 4 s, about 35k input tokens):

| vendor | monocular input | dynamic output | self-serve API | camera control | export | people in motion | ready |
|---|---|---|---|---|---|---|---|
| gracia | **no** 0.97 (genlock requirement) | yes 0.99 | unclear | unclear | yes 0.96 | yes 0.87 | no |
| worldlabs_atlas | unclear | unclear | **no** 0.95 (request access) | yes 0.98 | unclear | unclear | no |
| worldlabs_world_api | **no** 0.94 (static space) | **no** 0.97 | yes 0.89 | unclear | yes 0.89 | no 0.97 | no |

That matches a manual reading of the same pages. One pitfall was found and fixed: gracia.ai renders its text in the
browser. Its facts (the FAQ and descriptions) exist in the HTML only as schema.org JSON-LD, which the page reader now
extracts. Snapshots go to `work/mono4d/vendor_watch/<date>.json`, and each run lists what changed. Re-run it before
any decision, and after any World Labs or Gracia announcement.

---

## 3. Can generated views make MAMMA "accurate"?

- **What MAMMA gains:** exact, frame-locked calibration (the virtual cameras are defined, not estimated) and many
  viewpoints.
- **What it cannot gain:** information. The phone sees each dancer from one side.
  - A provider that renders the far side generates it from priors.
  - Several generated views agree with each other because one model made them all, not because they measured the
    same thing. MAMMA's triangulation then turns that agreement into confident 3D.
  - This is the failure the NoPo4D attempt showed in its worst form (`atlas_bridge/README.md`: duplicated dancers,
    misplaced limbs). A better generator makes the error smaller and harder to see, not zero.
- **Where it can still help:**
  - A strong generative prior of people (trained on large multi-view human data) may estimate depth along the
    phone's viewing ray better than our one-camera optimiser. Along-ray depth is the weak axis there: `mono_synth`
    reports the pelvis error split along and across the ray for that reason.
  - It may hold body shape better through partner occlusion.
- **Hence the rules in the code:**
  - The real phone always stays in the virtual take; it is the only measurement.
  - The manifest records `independent_views = 1`.
  - Views rendered at real phones' poses are compared with those phones and kept out of the fit set.
  - Nothing enters MAMMA without `gate.py`.
  - The route counts as better only if it beats the one-camera baseline on held-out ground truth (section 6).

---

## 4. What was built: `dancecap/dancecap/mono4d/`

```
request ──> provider ──> fetch ──> [render_gracia] ──> package ──> detect / dancers / mamma (unchanged stages)
   │           │            │                             │                      │
   │      manual drop   result.json                 virtual take            fit A (route)
   │      World API     (kind, dynamic,            work/<take>__m4d_<run>/        │
   │      Gracia .mint   world_from_scene)          real phone + v00.. + h<cam>     ▼
   ▼                                                      │              holdout.compare_fits vs the
 request/  inputs/<cam>.mp4 (+ .times.json),              ▼              8-phone fit, and vs fit B
           source_trajectory.json,             judge (Jev) + views2d ──> gate ──> quality_gate.json
           target_trajectory.json, REQUEST.md
```

| Module | What it does |
|---|---|
| `rig.py` | `ViewSpec` (per-frame K, R, t in the CONTRACTS world frame). Functions: `ring` (static cameras around the couple, portrait 1080x1920 by default; renders stay 1080p); `auto_radius` (the smallest radius at which the couple, a 0.9 m x 2.2 m cylinder, stays in every ring camera's frame over its whole floor path; bisection); `from_track` (a real phone's pose resampled to the grid); `umeyama`; `world_from_scene` (provider frame -> world: rotation from the camera orientations, scale and translation from the centres; refuses a phone that barely moved); `to_scene`; `trajectory` (c2w in OpenCV and OpenGL axes + intrinsics per frame, static cameras written once); `qvec_wxyz` (COLMAP quaternions for graciasdk). |
| `providers.py` | `Mono4DRequest` / `Mono4DResult`. `ManualProvider` validates deliveries in `drop/` (frame count = grid, exact size, scene alignment from `scene_cameras.json`). `GraciaProvider` writes the request but marks a one-phone input unsupported. `WorldLabsProvider` runs the World API flow (prepare_upload, PUT to the signed URL **without the API key**, worlds:generate with a video prompt, operations polling, PLY export). It refuses to upload without `--allow-upload` and to spend credits without `--yes-spend`, and returns kind `static_splat`. `cut_input_video` keeps every canonical frame and writes the exact times beside it. |
| `render_gracia.py` | A standalone script (graciasdk needs Python >= 3.12 + Vulkan; the project venv is 3.11). It renders a `.mint` at our cameras (already converted to the asset's frame) and writes JPEG views, alpha masks (free dancer masks) and depth. |
| `virtual_take.py` | Rendered views + the real phone -> `work/<take>__m4d_<run>/` in the CONTRACTS layout: frames, `cameras/<cam>.npz`, an identity `world_from_sfm.json`, and the `mono4d.json` provenance manifest. |
| `holdout.py` | `compare_fits`: MPJPE, PA-MPJPE, pelvis (horizontal / vertical) and ankle height against the reference fit. `compare_views`: people counted in the generated vs the real view (duplicates show as extra people), PCK@0.1, dancer-mask IoU. `verdict`: does the route beat the baseline by 10 %? |
| `judge.py` | TypeSafe Jev visual judgments, one request per sampled frame. Decisive: `duplicate_person`, `render_defects`, `cut_off`, plus `same_pose` against the real frame at the same pose. Advisory: `broken_anatomy`. People are counted in code (Jev's own notes say it does not count reliably). Images are a "dreamer-ready" TypeSafe feature (header `TypeSafe-Danger-Zone-Images`); sent as base64 assets through the installed SDK 0.7.2 (`extra_body`). |
| `vendor_watch.py` | The readiness watch of section 2. |
| `gate.py` | The admission decision (rules in its docstring): dynamic delivery only; alignment within 3 degrees p95 and 5 cm rms; Jev-flagged views dropped; held-out views must match the real phones (otherwise the whole run fails); >= 4 ring views spanning >= 180 degrees; on an evaluation take, the hold-out verdict. |
| `calibrate.py`, `oracle.py` | A labelled mannequin set for the judge, and a stand-in "perfect provider" (the take's own fit drawn as mannequins at every camera). Neither contains a real person. |

Commands (from `dancecap/`, `.venv-nopo4d`):

```
python -m dancecap.mono4d vendors
python -m dancecap.mono4d request larissa_kadu_38 --cam 06 --t0 58 --t1 62 --holdout all [--provider manual]
python -m dancecap.mono4d fetch    larissa_kadu_38 --run <run>
python -m dancecap.mono4d render-job larissa_kadu_38 --run <run>      # .mint deliveries, then render_gracia.py
python -m dancecap.mono4d package  larissa_kadu_38 --run <run>
python -m dancecap.mono4d judge    larissa_kadu_38 --run <run> --allow-upload   # owner's go-ahead: real frames
python -m dancecap.mono4d views2d  larissa_kadu_38 --run <run>      # after detect / dancers on the virtual take
python -m dancecap.mono4d holdout  larissa_kadu_38 --run <run> --test <fitA smplx> --baseline <fitB smplx>
python -m dancecap.mono4d gate     larissa_kadu_38 --run <run>
python -m dancecap.mono4d judge-calibrate
```

Layouts are in `CONTRACTS.md` (section "mono4d").

---

## 5. What was tested (measured, 2026-10-15)

- **Unit tests:** `tests/test_mono4d.py`, 23 tests, no network. They cover:
  - ring geometry and conventions, trajectory round trips, the auto radius;
  - Sim3 recovery, including a phone walking in a straight line. A first version solved rotation from camera
    centres alone and was off by 28 degrees in that case; it was fixed;
  - quaternions; metrics and the verdict; 2D view comparison;
  - manual delivery validation and the virtual take loaded by `mamma_dyn.load_tracks`;
  - scene alignment from provider poses; Gracia refusing a phone input;
  - the World API flow against a mock server, checking that the key never reaches the storage URLs;
  - the judge's request shape, image limits and policy; the vendor watch's parsing, decision and change log;
  - the gate's admit and fail cases; a variable-rate input cut.

  The README's NumPy-only suite plus `test_jl_mono` still pass (177 tests).
- **Real take, request only:** `larissa_kadu_38`, phone 06, 58-62 s (the window of the NoPo4D test).
  - 121 grid frames, 120 input frames cut losslessly. A first version using ffmpeg's concat demuxer dropped 18 of 120
    frames; it was fixed.
  - Ring: 8 cameras at 1.5 m, auto radius 3.87 m around the couple's mean floor position.
  - Projecting the 8-phone reference fit into every ring camera keeps both dancers in frame over the whole window.
  - The 7 held-out phone poses: 6 frame the couple; phone 09 cuts them at the edge, as the real phone 09 does.
- **Oracle end-to-end:** request -> oracle delivery (15 views x 121 frames, mannequins of the reference fit) -> fetch
  -> package -> `load_tracks` (16 cameras: the real 06 + 15 generated) -> live Jev judge (75 frames, about 151k
  input tokens, 4 s) -> gate. Result: admit, all 8 ring views, 315 degrees of azimuth, nothing dropped.
- **Visual judge calibration (live, jev-1.13.0):** 18 labelled mannequin frames (clean, ghost duplicate, missing arm +
  detached shin, floating blobs + a hole, same / different pose), in 4 prompt variants.
  - `same_pose`: accuracy 1.00; mean P = 0.98-0.99 for the same pose vs 0.09-0.14 for a pose 1.1 s later.
  - `duplicate_person`: ghosts scored 0.47 / 0.71 / 0.73; the 90 clean frames (calibration + oracle) peaked at 0.62.
    The threshold is 0.65: 2 of 3 ghosts caught, no false alarm.
  - `render_defects`: ranks every defect frame above every clean one, but clean mannequins already score 0.45-0.8,
    so the threshold is 0.9.
  - `broken_anatomy`: does not separate on mannequins, so it is advisory only.
  - Giving Jev the input phone's frame as context changed little.
  - Mannequins are not renders of real people. These thresholds are provisional until real labelled frames
    calibrate them: the NoPo4D views as known-bad, phone frames as known-good. That needs the owner's go-ahead,
    because those frames would go to TypeSafe.
- **Not run:**
  - any vendor or model delivery of real generated views;
  - `views2d` (needs the detect / dancers stages on the virtual take: YOLO / SAM work for a pod);
  - any MAMMA fit of a virtual take;
  - graciasdk itself (no `.mint` of ours, and installing its prebuilt wheel was not part of this session).

---

## 6. The hold-out protocol (how the route is judged)

On `larissa_kadu_38` (8 phones; reference = its 8-phone fit, `smplx/`):

1. `request --cam 06 --holdout all` over a window. Start with 58-62 s: fast head turns, used before. Then a
   30 s window that contains a counterbalance.
2. The provider sees only `inputs/06.mp4` (and, for Atlas, `source_trajectory.json`: Atlas takes camera poses as
   input). It renders the ring and the 7 held-out phone poses.
3. `package`, then on a pod: detect + dancers + ma_2d on the virtual take.
4. `views2d`: the generated `h01`...`h09` against the real phones. This is the cheapest and most direct test of
   invention: wrong limbs and duplicates show here first.
5. Fit A: MAMMA on {06 + ring} (the `h*` views excluded). Fit B: the one-camera fit of 06 over the same window
   (MAMMA_MONO path, `jl_mono` start).
6. `holdout --test A --baseline B` -> per role MPJPE, PA-MPJPE, pelvis, ankle height. Then `gate`.

The route earns more work only if A beats B by >= 10 % on the PA-MPJPE and pelvis medians for both dancers, and the
held-out views pass (PCK@0.1 median >= 0.8, mask IoU >= 0.6, extra people on <= 5 % of frames). Thresholds are in
`holdout.VERDICT` and `gate.GATE`.

---

## 7. Self-hostable stand-ins while Atlas is closed (to check before any download)

| Candidate | What it is | Status found 2026-10-15 |
|---|---|---|
| TrajectoryCrafter (ICCV 2025 oral) | Redirects the camera trajectory of a monocular video with diffusion; precise pose control | Weights on Hugging Face (`TrajectoryCrafter/TrajectoryCrafter`); inference only. Licence not checked |
| ReCamMaster (Kuaishou / Kling, ICCV 2025) | Camera-controlled re-rendering of a single video | Code + checkpoint released 2025-04 as a Wan2.1 port (repo licence MIT). The authors say the port may not match the paper's internal model |
| GEN3C (NVIDIA) | 3D-informed, camera-controlled video, including re-rendering a dynamic monocular video | Release status and licence not checked |
| FreeOrbit4D (2026) | Training-free camera redirection via foreground-complete 4D reconstruction | Code status not checked |
| NoPo4D | Feed-forward dynamic Gaussians from unposed multi-camera videos | Tested 2026-10-02 and rejected (atlas_bridge/README.md) |

All are generative, the same caveat as Atlas (section 3), and all run on Runpod (Linux CUDA, large VRAM). The
manual provider takes their output as-is: render the requested trajectories and drop `views/<camera>.mp4`.

---

## 8. Constraints the code enforces

- **Consent and likeness.** Footage leaves the PC only with `--allow-upload` (the owner's go-ahead, per request).
  This covers uploads to World Labs, frames sent to TypeSafe for judging, and a vendor hand-off. Generated views of
  real dancers are likeness data too. TypeSafe's zero data retention is offered to enterprise customers only.
- **Cost.** World API generation needs `--yes-spend`. Pods follow the farm rules (deadline, kill switch, ledger).
  Gracia pricing makes it a studio option, not a per-take one.
- **Compute.** Request, package, judge and gate are light local steps. Rendering generated views, detection, ma_2d
  and MAMMA run on pods.
- **Licences.** Provider outputs and MAMMA results stay in `work/` (git-ignored). Nothing goes to the public
  head-movement repository.

---

## 9. Next steps

1. **Owner:** apply for Atlas early access (form on worldlabs.ai/blog/atlas). Describe one handheld phone video of a
   partner dance, re-rendered at known camera poses for markerless motion capture, with ground-truth multi-phone
   footage available for evaluation. That evaluation offer is a genuine contribution for them.
2. **Owner go-ahead (likeness to TypeSafe):** calibrate the judge on real frames. About 40 NoPo4D novel-view frames
   (known-bad) + 40 phone frames (known-good) at about 3k tokens each: well under $0.01. Then re-set
   `judge.THRESHOLDS`.
3. **Runpod spike (needs approval of a cost cap; estimate $1-3):** TrajectoryCrafter on `smoke_06_58`. Render the 7
   held-out poses first, then score with `views2d`; render the ring only if the held-out views are plausible. Check
   its licence first.
4. **Baseline fit B** for the same window (the one-camera MAMMA path that exists for john_lula), so the verdict has
   a reference.
5. **Gracia:** only for a future studio capture (or their free 1-2 minute test). `render_gracia.py` turns a `.mint`
   into MAMMA views. Not a route for phone videos.
6. Re-run `python -m dancecap.mono4d vendors` monthly, or after announcements; the table and its change log say when
   a vendor becomes "ready".

---

## 10. Tool round 2026-10-19: realistic moving dancers from one phone

The owner's end goal, verbatim: "highly realistic 3D moving dancers with correct poses and natural appearance,
clothes, hair for 3d tutorial dance videos and playback in VR, from monocular video (handheld phone)". The tools
named were World Labs Atlas (now accessible), the Unity asset Learned Motion Matching, NVIDIA Kimodo, Dense4D and
Viggle JST-2. The owner consented to any images and likenesses being used for these tests, and set a Runpod cap of
$25 for the round.

Method:
- Six research tracks: one per tool, plus a sweep for other 2026 tools. Each was re-checked by an independent
  adversarial verifier against primary sources.
- Live tests where access allowed.
- One scoring harness: `mono4d/candidates.py`.

### 10.1 What each tool is, for this goal (verified)

| Tool | What it is | Role for us | Verdict |
|---|---|---|---|
| **World Labs Atlas beta** (Marble 2 API) | Hosted IMAGE tasks on posed RGBD views (images2PosedRGBD, atlasGenerate, atlasMasked, atlasChisel, text-to-image, splats2Mesh). 1280x720 generation grid; no video, temporal or human task. Its own best-practices page says people and dynamic objects are not yet a strength. Paid users own outputs (ToS 3.3) and can revoke the training licence. | Single-image camera + depth estimation (measured: very good); room views. Not the dancers. | **Use for cameras/depth only**: 10.2 |
| **Viggle JST-2** (API) | Commercial cloud service: one person's video -> skeletal animation (GLB/FBX on a 65-joint Mixamo rig with fingers, plus MetaHuman rigs), $0.05 per source second; image -> Gaussian-splat character (.vsplat). | A one-person articulation prior. | **Measured: 84-109 mm PA-MPJPE**: 10.3 |
| **NVIDIA Kimodo** | Motion DIFFUSION model (text + kinematic constraints -> motion) on SOMA / G1 / SMPL-X skeletons. Code Apache-2.0; SOMA weights NVIDIA Open Model License. The SMPL-X weights are gated, under an R&D licence that forbids "generating works for sale or distribution". No video input. | In-betweening / gap filling, lesson variations. Not pose estimation. | Pod test (10.6); open SOMA weights only (owner's choice) |
| **Dense4D** (arXiv 2610.22325, 2026-10-15) | Feed-forward, template-free dense 4D correspondence of body, clothing and hair from monocular frames (VGGT-Omega backbone, about 974M parameters). | The best match to "clothes, hair" from one phone: measuring clothed-surface and hair motion. | **Code not released** ("coming soon"); likely non-commercial; it does not take person masks as input, which matters in a close embrace |
| **Unity Learned Motion Matching** (AiKodex, $20) | Locomotion-only, single-character runtime controller; pinned to a 2023 pre-release Sentis; reviews report it broken on current Sentis; no custom-data training path. | None. Our viewer already replays the captured fits at 72 fps on Quest 3. | **Do not buy** |

The sweep found released tools that fit parts of the goal better than the five named ones (licence and release
status verified):
- **Pose:**
  - Meta SAM 3D Body + SAM-Body4D: gated on Hugging Face (SAM License); reported 3DPW MPJPE 54.8.
  - NAVER Multi-HMR 2: multi-person metric placement; non-commercial.
- **Partner contact:** CloseApp (CVPR 2025, MIT code, SMPL), the only released monocular method built for two
  people in close contact.
- **Appearance:** LHM++ (Apache-2.0), an animatable Gaussian human from frames, as a texture source and PC layer.
- **Hair:** DiffLocks (non-commercial, registration), about 100K strands from one frontal photo.
- **Novel views:** MV-Performer, human-specific monocular -> 360-degree multi-view video; its repository has no
  licence file.
- **Physics:** ProtoMotions3 (Apache-2.0 code; NVIDIA Open Model License weights).
- **Paid video-to-mocap services** (Move.ai, DeepMotion, Flow Studio, Rokoko) publish no close-contact accuracy.
  They rank last.

### 10.2 Atlas beta, measured (larissa_kadu_38, phone 06, 59.98 s)

`python -m dancecap.mono4d atlas-test larissa_kadu_38 --cam 06 --t 60 --targets 01,03,05,07 --allow-upload`, with a
report under `work/larissa_kadu_38/mono4d/atlas/06_60000/`.
- **images2PosedRGBD** (one portrait frame at 720x1280, 2.7-3.2 s):
  - the focal length came within **0.6 %** of our calibration (760 vs 765 px);
  - gravity agreed within about 1 degree;
  - depth was **x1.11 of metric** against 3,586 room SfM points with the people masked (IQR 1.04-1.25).

  That is a usable single-frame camera + depth prior for one-phone takes, for example to settle the focal
  ambiguity the john_lula one-camera path could not exclude.
- **atlasGenerate** at four other phones' real poses (cameras mapped into Atlas's frame by that Sim3):
  - The first request mixed an RUB context camera with RDF targets and came back upside down and facing away. All
    cameras are now sent in RUB.
  - Afterwards the couple sits in the right place, but the bodies are not turned correctly in 3D. YOLO11x-pose
    PCK@0.1 per phone and dancer was 0.31/0.06 (01), 0.67/0.63 (03), 0.0/0.31 (05) and 0.0/0.0 (07), **median
    0.19**, against the gate's 0.8.
  - Jev same_pose: 0.12-0.64.
  - Atlas does not give correct dancer novel views.
- **atlasMasked** clean plate:
  - Removing the people refilled the mask with melted human figures (Jev: any person 1.00, plausible empty studio
    0.06), and the first canvas attempt left blurred ghosts.
  - Not usable for clean plates of a crowded room.
- Five paid beta calls in all. The prompt enhancer wrote a detailed description of the dancers into the operation
  record. Library uploads are not purged automatically.

### 10.3 Viggle JST-2, measured (same take, 58-62 s)

`python -m dancecap.mono4d viggle-test larissa_kadu_38 --cam 06 --t0 58 --t1 62 --allow-upload`: 40 credits ($0.40).
- **Input:** one fixed crop per dancer, with the partner's SAM mask inpainted away. A flat grey fill would leave a
  body-shaped silhouette that a tracker can still follow.
- **Scoring:** the 65-joint Mixamo GLB, read by our own numpy GLB reader, on the 14 joints both rigs define.
- **Result:** PA-MPJPE median **109 mm (lead) / 84 mm (follow)** (p95 142 / 133; re-scored on the reference clock, 10.6); limb-direction error **21 / 17
  degrees**.
  - Time-aligned: the best shift is 0 to +1 frame.
  - Hands are worst (about 150 mm): the arms wrap the inpainted partner.
- **Terms:** ToS 5(b) gives Viggle a perpetual, irrevocable, sublicensable licence to uploaded content from EVERY
  user, API users included. The API section only forbids training, third-party use and marketing. The motions are
  deletable through the API.

### 10.4 The TypeSafe (Jev) visual judge, calibrated on real frames

With the owner's consent:
- **Data:** 60 known-bad NoPo4D renders against 60 real phone frames of the same takes, downscaled to the same
  resolution so that blur does not decide (jev-1.13.0, 122k input tokens).
- **render_defects:** AUC 1.000, threshold 0.80.
- **broken_anatomy:** AUC 1.000, threshold 0.60. It is decisive on real frames, though it was useless on
  mannequins.
- **duplicate_person:** AUC 0.999, threshold 0.45.
- **cut_off:** AUC 0.904 and advisory, because real phones that crop the dancers score up to 0.92.
- **Result:** 60/60 bad and 0/60 good frames flagged. Fitted on one take, the thresholds score 0.963 / 0.975 on the
  other.
- The Atlas views (clean-looking, wrongly posed) pass, as they should: their error is the pose, which same_pose and
  the YOLO PCK catch.
- The mannequin values remain as `PRESETS["mannequin"]` for oracle runs.

The Jev vendor watch now also reads the five tools' own pages (`python -m dancecap.mono4d vendors`: 8 vendors,
9 s, 102k tokens). Its page-based verdicts:
- Kimodo: released.
- Dense4D: clothing/hair and motion from video, release unclear.
- Viggle: video -> motion, usable today.
- Atlas beta: people ruled out as a strength.

### 10.5 One scoring harness for every one-camera candidate

`dancecap/mono4d/candidates.py`.
- **Contract:** a candidate is per-dancer joints in SMPL body order (world, camera or free frame) under
  `work/<take>/mono4d/cand/<name>/`.
- **Role assignment:** each predicted track is projected into the phone against our lead/follow boxes.
- **Metrics:** PA-MPJPE (22 joints, and the 14 that every rig defines), root-relative MPJPE and metric pelvis error
  for world/camera candidates, and limb-direction error.
- **Leaderboard:** `python -m dancecap.mono4d leaderboard larissa_kadu_38`. It includes the multi-view floor: the
  7-phone held-out chunk agrees with the 8-phone reference to 4-10 mm on these metrics. That is a consistency
  floor, not ground truth, since the reference is built from the same chunks.

### 10.6 Pod tests (Runpod, $25 cap for the round), measured

All runs used phone 06 of larissa_kadu_38, 58-62 s. Each pose candidate was scored by the harness (10.5), then checked
by an independent auditor (joint order, left/right, bone lengths, time shift, roles, re-score). Every pod was created
and deleted by the farm runner. Licensed weights stayed on the pods; outputs stay in git-ignored `work/`.

**Pose, one camera** (medians, mm; lead / follow; PA22 = PA-MPJPE on 22 joints; pelvis = world position):

| candidate | PA22 | root-rel | pelvis | limb (deg) | cost | note |
|---|---|---|---|---|---|---|
| multi-view floor (7 phones) | 4 / 8 | 5 / 10 | 7 / 8 | 1 / 2 | - | consistency floor |
| **fit B** `mamma_mono06` (our MAMMA_MONO) | 47 / 44 | 63 / 55 | 67 / 48 | 13 / 9 | $1.38 | the bar; free shape from the statures |
| CloseApp (refining fit B) | 48 / 44 | 64 / 54 | 62 / 56 | 12 / 9 | $0.23 | changes the pose by < 1 mm |
| Multi-HMR 2, our mask identities | 63 / 61 | 81 / 77 | 254 / 72 | 13 / 13 | $0.10 | follow missing in ~20 % of frames |
| Multi-HMR 2, its own tracker ids | 73 / 81 | 91 / 94 | 297 / 231 | 17 / 17 | (same run) | swaps the dancers on 28 % of frames |
| Viggle JST-2 (10.3) | 109 / 84 | - | - | 21 / 17 | $0.40 | free frame, 14 joints |

- **fit B** (`cand_fitb.py`): MAMMA ma_3d with MAMMA_MONO=1 on phone 06 alone, betas free and held to the owner's
  statures. Its world metrics are optimistic for a true one-phone take: it uses phone 06's camera track and sync from
  the 8-phone SfM. PA-MPJPE is not affected.
- **CloseApp** (`cand_closeapp.py`, commit 516ed6f, MIT): the release demo config. Its proxemics prior cannot run as
  released (an undefined variable; batch shape mismatch), and the penetration weight is 0. Initialised from fit B,
  because there is no HMR2 init. Penetration depth 42 mm vs fit B 45 mm (reference 31 mm); hand-contact recall 0.74
  vs 0.73. Its 3D-Gaussian avatars are coarse (hair a blob). **Not adopted**; its SDF penetration term may be worth
  porting into MAMMA_MONO.
- **Multi-HMR 2** (`cand_multihmr2.py`, NAVER non-commercial): our focal passed to the decoder (its own predicted
  focal is 19 % short, which puts the dancers 0.7-0.9 m too close). Front/back ordering of the couple agrees with the
  reference on 88 % of frames, against 99 % for fit B. **Not adopted**, not even as a placement prior.
- **SAM 3D Body** (`remote/cand_sam3d_body/`): **running**. The first pod attempt hit HTTP 403 ($0.04): the
  owner's Hugging Face access to `facebook/sam-3d-body-dinov3` was not yet approved. Meta approved the request at
  17:22 the same day. The job prompts the model with our SAM 2.1 masks and our K, and converts MHR -> SMPL-X with
  Meta's official converter. Note: listing a gated repo's files succeeds without access; only a download (HEAD)
  proves access.

**Gap filling** (`cand_kimodo.py`, Kimodo-SOMA-RP-v1.2, open weights, no text encoder; $0.67): one held-out second
(59-60 s) of the 8-phone fit, keyframes on both sides, 5 samples, medians.
- Follow (a 189-degree turn inside the gap): PA-MPJPE **43 mm** vs 69-72 mm for SLERP / linear / cubic; the turn goes the
  right way in 3 of 5 samples (SLERP turns the wrong way).
- Lead: ties SLERP on PA-MPJPE (43 vs 40) and is 10 mm worse than cubic on world MPJPE.
- **Head: 58-67 mm vs 22-26 mm for cubic**, and it lifts the follow's head during the turn (about 70 mm in the median sample, up to 98 mm). Head motion
  is this project's focus, so Kimodo is **not a drop-in in-betweener**. Next: head/hand end-effector constraints,
  heading constraints at the boundaries, keyframes from fit B instead of the reference.

**Appearance** (`cand_lhmpp.py`, LHM++ 700M; code Apache-2.0, weights CC BY-NC 4.0; $0.32): an avatar per dancer from 8
masked phone-06 crops, posed with the reference SMPL-X, rendered into held-out phones 01 / 03 / 05 at 10 times
(60 dancer-views, inside the real masks, native 480-px phones):

| variant | PSNR dB | SSIM | LPIPS | silhouette coverage |
|---|---|---|---|---|
| LHM++, 8 crops from 58-62 s, reference shape | **12.6** | **0.228** | 0.203 | **0.86** |
| LHM++, 16 crops from the whole take | 11.8-11.9 | 0.21-0.22 | 0.20-0.21 | 0.86-0.90 |
| current baked texture on naked SMPL-X (8 phones, saw 01/03/05) | 11.2 | 0.202 | 0.216 | 0.75 |

- From ONE phone it beats the 8-phone bake on every whole-mask metric, mostly through clothes and hair geometry
  (follow: 10.9 dB vs 8.7 dB). Where both renders cover the dancer, the bake is still sharper (LPIPS 0.126 vs 0.149).
- In the contact sheet (`work/larissa_kadu_38/mono4d/lhmpp/contact_sheet.jpg`) the jeans, crop top, t-shirt and red
  hair read correctly. Hair is skinned rigidly (it does not swing), and the partner's occluding arm leaves white
  patches that the avatar copies.
- Raw PSNR is dominated by exposure differences between phones: colour-corrected PSNR is about 4 dB higher for every
  method.
- **Most promising result of the round for "natural appearance, clothes, hair".** Quest path (Gaussians are ruled out
  on the Quest): render the canonical Gaussians from about 48 virtual cameras, bake onto the existing SMPL-X UV atlas,
  and push subdivided vertices out to the Gaussian surface for the clothes silhouette; long hair stays with the strand
  pipeline. About a day of glue code.

**Hair** (`cand_difflocks.py`, DiffLocks fcc7374, MPI non-commercial licence; $0.53, two A40 runs): strand hair
(80K-124K strands) for the follow from one face crop, run on 12 crops (phones 06 and 02, the owner's portrait;
raw, black and grey backgrounds). Sheet: `work/larissa_kadu_38/mono4d/difflocks/contact_sheet.jpg`.
- **Length is about half of hers.** Her tip drop below the crown is 0.52 m; DiffLocks gives p95 0.31-0.36 m at
  best (the portrait), 0.12-0.17 m on phone 02. Its crop is 2.8x the face, so it never sees hair below the shoulders.
- Colour and texture follow the crop: blurred 70-px phone faces give curls or frizz, black backgrounds give blond or
  white hair. Only the portrait and one grey-background phone-06 crop look plausible. 9 of 12 have a centre part
  (hers is on her left).
- Run 1 grew 3 cm hair on the "Dream" print of her T-shirt: DiffLocks keeps the largest MediaPipe face at
  confidence 0.1. Run 2 picks the detection nearest her fitted face.
- **Not adopted.** The current Unity v2 groom (160 guides, hair cards) is closer to her real hair. DiffLocks could at
  most give a shape prior near the scalp (1-2 days of work, little gain).

**Harness clock fix.** The reference fit and `load_tracks` use the sync-corrected clock (phone 06: frames.json t
-30.04 ms). A candidate saved on raw frame times is matched one frame late: a perfect pose then still scores PA22 +7.5 /
+10 mm and root-relative +15 / +18 mm. Multi-HMR 2 had this bug (fixed by its auditor; follow pelvis 86 -> 72 mm), and
so did the Viggle import (re-imported; PA22 108 -> 109 / 84 -> 84 mm, since its motion is smooth).
`candidates.frame_to_ref` now does the conversion, and `score()` records a clock check per role and flags raw-clock
candidates in the leaderboard.

**Cost.** $3.83 of the $25 so far (ledger $91.37 of the $112.54 cap, with the SAM 3D Body and comparison pods still running). $1.36 of it was lost to uploads timing out
on the home uplink: the runner bundles 330 MB of wheels even for jobs that need none.
