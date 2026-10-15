# One phone -> 4D provider -> virtual cameras -> MAMMA: findings and test plan

**Status 2026-10-15.** The owner asked for "a monocular video to the api, get a 4d gaussian, and create novel camera
angles so we can run MAMMA", with Gracia (gracia.ai) and World Labs Atlas (worldlabs.ai/blog/atlas) as the
candidate APIs. This file is the decision record: what the two vendors offer today, what was built
(`dancecap.mono4d`), what was tested, and the next steps. Numbers marked *measured* come from runs on 2026-10-15.
Numbers marked *stated* come from the vendors' own pages, read that day; *estimate* means reasoned, not measured.

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
