# Dancer Gaussian splats guided by the avatars: research and prototype plan

**Status 2026-10-08: approved by the user ("run the splats attached to body fits and track them as part of the
pipeline") and built** as `dancecap.dancer_splats` (pipeline stage `dancer_splats`, pod session S_splats). What
was built, how it deviates from this plan and what was measured locally: section 12. The pod run itself had not
started when section 12 was written.

Research (2026-10-07; sections 1-11 are unchanged since). This file answers three questions:
- What was the earlier NoPo4D splat attempt, and why did it look messy?
- What can we build now that we have calibrated cameras, masks, SMPL-X fits, textured avatars and a hair groom?
- How should a prototype of a few seconds be run?

User constraints this file follows:
- **Dancers only.** Splats show the two dancers. No floor, room or spectators (VIEWER_SPEC 3.2a). The splat
  layer starts off.
- **Licence.** The work is non-commercial research. Splats are likeness data: they stay in `work/` and the
  git-ignored StreamingAssets, never in `head-movement`.
- **Headset.** The 2026-10-05 decision is that the Quest standalone build uses textured skinned avatars, and
  splat avatars are out for the headset. This file treats splats as a PC review layer. Section 8 lists evidence
  that would let the user revisit that decision; it does not change it.
- **Compute.** Heavy work runs on Runpod. Locally we only prepare data and run short smoke checks. The budget left
  is about $10 of the $30 cap. **Any pod run in section 7 needs the user's approval first.**
- **Faces.** No further investment in faces for now (user, 2026-10-07). So the face criterion is only "no worse
  than the avatar".

Numbers marked *measured* come from files in `work/`, `prototype/` or `docs/PIPELINE_PROGRESS.md`. Numbers
marked *estimate* are reasoned from those measurements. Numbers marked *reported* come from papers or READMEs.

---

## 1. Summary

1. **The NoPo4D attempt did not fail because of its settings.** NoPo4D is a feed-forward model that guesses the
   cameras itself. We fed it 8 handheld, zooming phone videos at 126x224 pixels, with no masks and only 4
   timestamps in 4 s. Its guessed cameras did not agree with each other, so every view placed the dancers in a
   different spot. That produced duplicated dancers, holes, scale drift up to 2.2x between chunks and an
   in-sample PSNR of only 10.4-13.6 dB. Today every one of those causes has a measured fix in our data.
2. **Do not feed the avatars into NoPo4D.** Instead, fit the splat to the avatar. Each Gaussian is attached to a
   triangle of the dancer's fitted SMPL-X surface. Clothes and loose hair get small learned per-frame offsets on
   top. We supervise with the calibrated cameras and the per-dancer masks. This is the route taken by
   GaussianAvatars, SplattingAvatar and ExAvatar (mesh-bound Gaussians), Animatable Gaussians and SqueezeMe
   (multi-view, pose-driven), and STG-Avatar and DualGS (skeleton motion plus residual Gaussians).
3. **Recommended: build it ourselves on gsplat.** gsplat is Apache-2.0. Our room stage already uses it on pods,
   with cached wheels and a Windows patch. `room.py` already has the trainer pieces we need:
   - MCMC densification,
   - per-image colour transforms,
   - masked L1 + SSIM loss,
   - PLY export,
   - pod job packaging.

   We call the new method **AGS (avatar-guided splats)**. Published codebases each fit poorly here:
   - Most published avatar code is monocular, SMPL-only, non-commercial or needs 16-160 camera domes.
   - None handles two dancers in contact, phone zoom and auto-exposure together.
4. **Prototype (needs approval):**
   - **Data:** larissa_kadu, **58.0-61.5 s** (105 frames). It contains her three fastest head turns (59.18 s
     at 284 deg/s, 60.88 s at 299 deg/s, 61.41 s), so the hair moves.
   - **Training:** 8 cameras, camera 03 held out for evaluation.
   - **Where:** one pod session using the existing remote_jobs runner. *Estimate:* 17-37 min, $0.40-0.90 on an
     A100 at community price. Cap $1.50; at most $2 including one retry.
   - **Success:** no ghost dancers, better silhouettes and colour error than the current avatar on the held-out
     camera, visible hair motion in the turns, and playback in the existing Unity splat layer (key G).
5. **Large reconstruction models (LHM++) are a useful second tool, not the first.** LHM++ works on a Linux pod in
   about 2 s for 8 views (*reported*). Its weights are CC BY-NC 4.0, which is fine for non-commercial research.
   - Its best use is the **single-view and two-view** pipelines, where a learned prior must fill unseen sides.
   - For the 8-view take it is optional: a starting point and an A/B comparison.
6. **On the headset (information only):** Meta's SqueezeMe runs 3 Gaussian full-body avatars at 72 fps on a
   Quest 3. It uses about 60k Gaussians per avatar, skinning plus linear correctives (*reported*). Our 2 dancers
   at 170k Gaussians in total, played with skinning plus a small per-frame offset, are in that range. The standing
   decision stays: the headset ships skinned textured avatars.

---

## 2. What the earlier attempt was

**Code and outputs:**
- `prototype/run_nopo4d.py`, `reconstruct_clip.py`, `refine_nopo4d.py`, `render_saved.py`,
  `nopo4d-turing.patch`, `apply_gsplat_windows_patch.py`
- `prototype/nopo4d_{smoke,mono,quality,refined,square,full}/`
- `prototype/larissa_kadu/{nopo4d_test,nopo4d_four_second,quality_gate.json,test_manifest.json}`
- `atlas_bridge/README.md` ("Larissa/Kadu multiview test - 2026-10-02")

**Unity side:**
- `head-movement/Assets/Splats/{SplatCloud.cs,SplatPly.cs,NoPo4DPrototypePreview.cs}`
- `Assets/Editor/NoPo4DPrototypeScene.cs`
- `Resources/SplatPreview.shader`

**The model:**
- **NoPo4D**, "No Pose, No Problem in 4D" (Balice 2026, arXiv 2605.22190), MIT code, checkpoint
  `external/checkpoints/nopo4d`.
- Feed-forward, **pose-free** dynamic 4D Gaussian splatting from unposed multi-view videos.
- Built on a Depth Anything 3 backbone (DA3 is non-commercial: CONTRACTS licence row).
- Benchmarked on ExoRecon, Immersive Light Field, Kubric and N3DV, mostly fixed rigs and synthetic scenes. Only
  inference code is public; the training code is not.

**Runs (all local, 2080 Ti 11 GB, all *measured*):**

| run | input | result |
|---|---|---|
| `carlos-aline-spin` full | 2 cameras, 134 frames, 448x252, chunks of 4 frames, chunk-to-chunk similarity alignment | 34 chunks: scale vs the first chunk drifts 0.92-2.18x. Camera-1 rotation drift up to 12.6 deg. Centre drift up to 0.35 baseline. `quality_gate.json`: `accepted_for_mamma: false`, "Two-camera fusion duplicates dancers; monocular novel views reveal holes" |
| `nopo4d_quality` | 2 cameras x 4 frames, 504x280, 1.13 M Gaussians, 26 s, 9.6 GB | Predicted-camera Sampson error 42 px median (`geometry_check.json`: inconclusive) |
| `nopo4d_refined` | photometric refinement, 100 steps, cameras fixed | In-sample PSNR 18.4 -> 22.4 dB (fitting metric, not held out) |
| larissa_kadu `nopo4d_test` | 8 cameras x 2 consecutive frames, 168x294, 790k Gaussians, 5.9 GB | "distorted and duplicated limbs" |
| larissa_kadu `nopo4d_four_second` | 8 cameras x **4 timestamps** (58.0, 59.3, 60.7, 62.0 s), **126x224** (168x294 ran out of memory), 903k Gaussians, 7.1 GB | In-sample PSNR **10.4-13.6 dB** per view. Duplicated dancers, inconsistent room, floating dark regions, black occluder blobs in every view (`quality_gate.json`: `failed_visual_quality`) |

**Unity:**
- `SplatCloud` plays a per-frame PLY sequence (`capture.json` `splat_sequence`: `dir`, `pattern`, `fps`,
  `scene_to_unity`).
- It draws unsorted, isotropic, camera-facing discs with DC colour only (`SplatPreview.shader`). That is fine for
  placing splats in the room, but it is not a faithful 3DGS render.
- `HeadMovement.cs` already wires a `splats` layer (key G) to `SplatCloud.SetFrame`.
- `demo_export` currently writes no `splat_sequence` (its check requires "no room / splat").

### 2.1 Why it was messy, and what fixes each cause now

| symptom | cause | what we have now (measured) |
|---|---|---|
| Duplicated / ghost dancers | Pose-free: each view's guessed camera disagrees, so each view puts the dancers in a different place. The 2-camera fusion was blended, not reconciled. | `cameras/<cam>.npz`: per-frame K, R, t for every camera. Static reprojection 0.8-1.6 px median (larissa_kadu `cameras/report.json`). World frame is metric, Y-up, with the floor at 0. Attached cameras have their own gates. NoPo4D even accepts `input_extrinsics/intrinsics` (`src/model/nopo4d.py`), but our runs never used them. |
| Scale / rotation drift between chunks | Each 4-frame chunk is predicted on its own and aligned afterwards with a 2-camera similarity | One world frame for the whole take (`world_from_sfm.json`, scale checked against statures to +-1.5 %) |
| Room, spectators and mirror compete with the dancers; floating dark regions | No masks. Every pixel had to be explained, including the mirror wall, phone screens and letterboxing. | `dancers/<cam>/masks/mask_{i:04d}_{01,02}.png`: per-dancer FULL masks (body + hair), plus `body/` and `hair/` splits. `people_masks/` are all people + phone screens (as "ignore"). The room is a separate diffuse splat with planar reflectors (`room/`). |
| Holes in novel views | Per-pixel depth from 2 views; nothing behind the visible surface | A closed body surface (SMPL-X) per dancer and frame. Every Gaussian lives on it, so the back is always there. The texture bake already fills unseen texels by mirroring and diffusion. |
| Smeared, blobby limbs; PSNR 10.4-13.6 dB | VRAM forced 126x224 for 8 views (the dancer was about 60 px tall). Out-of-distribution input: 8 wide-baseline handheld phones on a circle, zoom. | Pods with 80 GB GPUs. Training on dancer **crops** at native resolution (480p phones, two 1080p cameras). Optimisation, not one feed-forward pass. |
| Temporal incoherence (motion inferred between 4 timestamps over 4 s) | Feed-forward over a handful of timestamps; no identity of Gaussians over time | SMPL-X on a 30 Hz grid (`smplx/`, or the take's `[export] smplx_dir`; `smplx_face/` expressions, `avatar/head_fix.npz` neck). One canonical set of Gaussians moved by the body, so identity is kept by construction. |
| Two people in contact merge | No person identity at all | Lead/follow identities, anchored and repaired (`dancers/roles.json`, swap probabilities). Posed meshes give an exact z-buffer per pixel (`avatar_raster.py`). Hand contact: `hands/contact.npz` exists only for larissa_kadu, from runs before the hands stages were retired (2026-10-07); elsewhere use posed-mesh proximity. |
| Video sync errors | Audio clocks only (+-16-67 ms residual) | `sync/offsets.json` per-camera video offsets (for example larissa_kadu_38 cam01 +90 ms, cam04 +107 ms). The texture bake already samples bodies at sync-corrected times. |
| Splats looked wrong in Unity | Disc preview: no anisotropy, no sorting, DC colour only | Section 8: Phase 1 keeps the disc preview for a quick look; Phase 2 adopts a real 3DGS renderer |

**Can the avatars help NoPo4D itself?** Only partly. Known cameras can condition its backbone, and masked,
dancer-only crops remove the room. That would probably fix the pose drift and much of the duplication.

But the core problems stay:
- **No body.** NoPo4D predicts Gaussians per pixel from depth. It cannot take a mesh or skeleton as input, so the
  avatar cannot hold views together, fill the back or keep identity over time.
- **Sparse in time.** It reasons over a few timestamps, while we need 105 frames with 300 deg/s spins.

So a "NoPo4D + known cameras + masks" run is at best a cheap baseline (seconds of inference on an A100, plus
installing its dependencies). It is not the way to dancer splats. **Recommendation: retire NoPo4D for dancers.**
Keep it in mind only as a room/scene tool for captures where SfM fails.

---

## 3. What the data allows (larissa_kadu, 56-62 s, measured)

### 3.1 Resolution and face size

The limiting factor for faces is resolution, not the method. Six of the eight cameras are 480p. Faces span 4-32
px between the eyes (inter-ocular distance, IOD), from `face/survey.npz`. Published avatar methods train on
1K-2K pixel frames with faces over 100 px IOD.

| cam | frames (W x H, before display rotation) | fps | face IOD lead / follow (median px) | camera notes (`cameras/report.json`) |
|---|---|---|---|---|
| 01 | 480x864 | 30 | 7 / 10 | main model, 1.1 px |
| 02 | 1080x1920 | 59.94 (decimated to 30) | **32 / 32** | main model, 0.8 px |
| 03 | 480x848 | 30 | 15 / 9 | main model, 1.0 px (proposed held-out camera) |
| 04 | 480x864 | 30 | 14 / 8 | main model, 1.0 px; zoom 1.71x in larissa_kadu_38 |
| 05 | 480x640 | 30 | 12 / 17 | attached, rotation-only, 1.6 px |
| 06 | 1080x1920 | 30 | **25** / 16 | attached, rotation-only, 1.3 px |
| 07 | 480x848 | 30 | 4 / 8 | main model, 1.1 px; zoom 1.37x in _38 |
| 09 | 480x848 | 30 | **31** / 17 | main model, 1.6 px; **zoom 1.49x** here, 2.57x in _38 |

What this means:
- **Face detail** can only come from cameras 02, 06 and zoomed frames of 09. Everywhere else the face Gaussians
  must stay close to the avatar's face texture (`avatar/<role>_albedo_face.png`).
- **Hair and clothes** are large on screen (the follow's hair reaches 0.38-0.61 m, `hair_groom.json`). All 8
  cameras see them, so they are where splats can clearly beat the naked textured SMPL-X.

### 3.2 Body fit accuracy

MAMMA's median reprojection error is 5-6 px (lead) and 5-8 px (follow). It rises to 8-11 px during her turns at
60.0-61.0 s (`smplx/<role>.npz` `reproj_px`). On a 480p frame, 8 px is several centimetres. This is the main
remaining source of cross-view disagreement. **The splat trainer must refine the pose per frame** (small
regularised corrections, written to its own folder; `smplx/` stays MAMMA's), as GaussianAvatar, 3DGS-Avatar and
AvatarPose do. The texture bake solved the same problem with optical-flow registration passes.

### 3.3 Phone artefacts

- **Exposure and white balance.** These differ strongly per camera: the bake's per-camera gains range from 0.6x
  to 2.5x per channel (`avatar/report.json` `roles.*.camera_gain_rgb_median`). They also change per frame
  (auto-exposure). This calls for per-image colour transforms, as `room.Appearance` already does.
- **Motion blur.** It is heavy in the spins at 30 fps; per-view sharpness is already measured in the bake report.
- **Rolling shutter.** It is unmodelled anywhere in the pipeline today. Phone sensors read out over roughly 20-30
  ms (*estimate*), so the head and feet are seen at different instants.

### 3.4 Baseline to beat (the current avatar)

From `avatar/report.json` `review_metrics`, best frame per camera, pod 2048-pixel bake:

| role | MAE (sRGB 0-255) | silhouette IoU |
|---|---|---|
| lead | 11.9-33.7 | 0.77-0.88 |
| follow | 37.0-47.0 | 0.66-0.83 (naked SMPL-X, no hair geometry) |

These numbers are in-sample for the avatar: every camera was used in the bake. Comparing on a camera held out
from the splat training therefore favours the avatar.

---

## 4. Method survey

Columns:
- **Inputs:** what the method needs.
- **Licence:** of the code; weights are noted where separate.
- **Platform:** Linux and GPU needs.
- **Time / cost:** for our prototype, about 3.5 s x 8 cameras = 840 images (*estimate* unless noted). A100 at
  $1.19-1.59/h (ledger prices: A100 PCIe community $1.19, SXM community $1.39, SXM secure $1.59; L40S community
  $0.79).
- **Face / long hair:** expected quality on *our* data.
- **Unity / Quest:** export and playback.

### 4.1 Mesh-bound animatable Gaussian avatars (mostly built for one camera)

| method | inputs | licence | platform | time / cost (ours) | face / long hair (ours) | Unity / Quest |
|---|---|---|---|---|---|---|
| **GaussianAvatars** (Qian, CVPR 2024): Gaussians bound to FLAME triangles, moving and scaling with them | multi-view **head** video + FLAME tracking | CC BY-NC-SA 4.0 (+ Inria 3DGS licence) | Linux, CUDA | head only | The **binding scheme** is what we adopt for the whole body (section 5.1) | Binding = triangle id + barycentric + local offset: easy to skin in a compute shader |
| **SplattingAvatar** (Shao, CVPR 2024): Gaussians embedded in the mesh (barycentric + displacement, can walk across triangles) | monocular video + mesh tracking | CC BY-NC-SA 4.0 | Linux | ~30-60 min | No hair dynamics | Same as above |
| **ExAvatar** (Moon, ECCV 2024): SMPL-X + 3DGS hybrid, each Gaussian a vertex with mesh connectivity; FLAME face offsets | **monocular** short video; its own SMPL-X/FLAME fitting | MIT (re-implementation) | Linux, CUDA | Not documented; *estimate* 1-3 h with its own fitting, less if fed our fits; multi-view needs code changes | Best face handling of the group (FLAME expressions). Long hair limited by surface connectivity. | Mesh-like, so skinnable |
| **GaussianAvatar** (Hu, CVPR 2024): pose-dependent Gaussians on a UV map, joint pose refinement | monocular video, SMPL or SMPL-X, camera + pose files | MIT | Linux | Not documented; *estimate* 1-2 h | Moderate. Hair moves with the head only. | Skinnable |
| **3DGS-Avatar** (Qian, CVPR 2024): deformable 3DGS + non-rigid MLP, pose correction | monocular (People-Snapshot) or ZJU multi-view; **SMPL only** | MIT core (+3DGS parts) | Linux, PyTorch 1.12, tinycudann | ~30-50 min (*reported*) = $0.6-1.2 | No SMPL-X hands or face | Needs MLP evaluation per frame |
| **GauHuman** (Hu, CVPR 2024) | monocular; SMPL; ZJU-style preprocessing | S-Lab (non-commercial) | Linux | 1-2 min (*reported*) | Low detail, SMPL only | Skinnable |
| **HUGS** (Apple, CVPR 2024): human + scene from one video | monocular; SMPL | Apple licence (check before use) | Linux, CUDA 11.7 | Not verified | Low; SMPL | Triplane MLP |
| **STG-Avatar** (2025): skinning + spacetime Gaussians for wrinkles and hair | monocular; SMPL | Not verified | Linux | 25 min on a 4090 (*reported*) | Better dynamic regions (+1.5 dB in dynamic areas, *reported*) | Time-dependent Gaussians: per-frame export |

**Takeaway.** The ideas are right: mesh binding, joint pose refinement and a residual for non-rigid motion. But no
codebase fits as-is. All are single-person, most use SMPL rather than SMPL-X, most are monocular, and none models
per-image exposure or zoom. Re-using one would mean rewriting its data loader, camera model, two-person rendering
and losses. That is about the same work as writing AGS on our own gsplat trainer, minus the licence and
dependency baggage (old PyTorch versions, tinycudann, the Inria rasteriser).

### 4.2 Multi-view, pose-conditioned avatars (best quality on camera domes)

| method | inputs | licence | time / cost | notes for us |
|---|---|---|---|---|
| **Animatable Gaussians** (Li, CVPR 2024): learned template + front/back Gaussian maps from a StyleGAN-style CNN conditioned on pose | multi-view video (16-24 views; 47 of ActorsHQ's 160), SMPL-X registrations, masks; 1500-3000 frames | Not stated in the README (check before use) | **About 2 days on an RTX 4090** for 500k iterations (*reported*); our 105 frames still need tens of thousands of iterations: *estimate* 6-12 h = $8-17 | Its sparse-view ablation (3/6/14 views) loses little quality (*reported*), which is encouraging for 8 phones. With 105 frames the pose-conditioned CNN has almost no pose variety to learn from. Long hair is a stated limit: "physical motion of components not driven by body joints". Too slow and costly for the budget. |
| **MMLPHuman** (2025): spatially distributed MLPs on anchor points | 14-39 views at 1-2K; SMPL-X; masks; template mesh | Not verified | 17.5 h on a 3090 (*reported*); *estimate* 8-10 h on A100 = $10-14 | Real-time 166 fps. Hair and skirt dynamics are a stated limit. Too costly. |
| **TaoAvatar** (Alibaba, CVPR 2025): clothed template + StyleUnet teacher distilled into a light MLP | multi-view studio recordings | Not verified | Teacher training takes hours | Runs at 90 fps on Apple Vision Pro (*reported*). Same idea as SqueezeMe. |
| **SqueezeMe** (Meta, SIGGRAPH 2025): distils Animatable-Gaussians-style avatars into skinning + linear correctives shared per 4x4 Gaussian block | 512-camera dome, 2-3k frames | No code | n/a | **3 avatars at 72 fps on Quest 3**, about 60k Gaussians each, Vulkan sort + quad rasterisation (*reported*). The template for any future headset path (section 8). |

### 4.3 Large reconstruction models (feed-forward, seconds)

| method | inputs | licence | platform | time / cost | face / long hair | Unity |
|---|---|---|---|---|---|---|
| **LHM** (ICCV 2025) | 1 image | Apache-2.0 code | Linux | seconds | Generic. Hair rigid with the head. | PLY |
| **LHM++** (2025, open-sourced March 2026) | 1-16 (or more) **pose-free** images, no cameras needed | Apache-2.0 code; **weights CC BY-NC 4.0** | Linux, CUDA 12.1, torch 2.3; compiled pointops, spconv, torch_scatter, pytorch3d, diff-gaussian-rasterization, simple-knn, gsplat; 7.3-8 GB VRAM | 0.8 s (1 view) to 2.1 s (16 views) on an A100 (*reported*). The first install (compiling extensions) is *estimated* at 15-40 min = $0.3-0.9 once; then cache the wheels like the other CUDA wheels. | About 160k Gaussians in canonical space bound to SMPL-X. Face identity is plausible but invented where not seen. Loose clothes need its neural renderer, which Unity cannot run; the raw splats are weaker. Long hair is rigid with the head. | `to_gs_ply.py` gives canonical or posed PLY. Driven by SMPL-X JSON (+ optional FLAME). |

**Fit for us.** LHM++ gives a complete avatar in seconds with no cameras, which is exactly what single-view and
two-view captures lack. For the 8-view take it can be:
- a starting point (transfer its canonical Gaussians onto our SMPL-X rest shape by closest point, since it fits
  its own betas);
- a baseline to compare against.

The memory note "no LHM++ (needs WSL2)" applies only to running it locally; on a Linux pod it works.

### 4.4 Volumetric video and generic 4DGS (per-frame Gaussians)

| method | inputs | licence | time / cost (ours) | notes |
|---|---|---|---|---|
| **HiFi4G** (CVPR 2024): per-frame NeuS2 geometry + embedded deformation graph + Gaussian graph, about 25x compression | dense multi-view dome (about 80 cameras) | Not verified | per frame: *estimate* many hours for 105 frames | Its per-frame geometry needs dense views. Our 8 sparse 480p phones give poor NeuS2 meshes. |
| **DualGS** (SIGGRAPH Asia 2024): "joint" Gaussians carry motion, "skin" Gaussians carry appearance; 120x compression; Unity plugin and mobile player in the paper | masked dense multi-view + calibration + initial points | MIT for the authors' code + Gaussian-Splatting licence for the rasteriser | *estimate* 9-17 h for 105 frames | The idea of separating motion from appearance is what AGS does with SMPL-X as the "joint" layer. Its own tracking needs dense views. The Unity player is described in the paper; the README shows only a SIBR viewer. |
| **4DGaussians / Deformable-3DGS / SpacetimeGaussians** (scene-level) | multi-view video + cameras | mostly non-commercial | 30-90 min | No body prior. With 8 sparse views and 300 deg/s spins they suffer the same under-constraint as NoPo4D. Useful only as a **residual** on top of the body (which is what AGS does). |
| **NoPo4D** | unposed multi-view frames | MIT (+ DA3 non-commercial) | seconds per chunk | Section 2 |

### 4.5 Two people in contact

| method | idea | use for us |
|---|---|---|
| **AvatarPose** (ECCV 2024, ETH): per-person avatars from sparse multi-view; poses refined by colour and silhouette rendering loss; collision loss between avatars | It solves our two-dancer pose refinement problem | Adopt: per-person colour and silhouette loss drives per-frame pose deltas; collision penalty |
| **GeoAvatar** (CVPR 2025): 2DGS multi-person avatars from sparse multi-view; multi-layer alpha blending; penetration penalised through segmentation mismatch | Rendering each person's own coverage map and comparing with their mask | Adopt: render a per-role alpha channel and compare with per-role masks. No public code found. |
| Huang et al. (CVPR 2025): close interaction from monocular video with Gaussians + proxemics | Gaussian depth order for occlusion between the two people | Background reading |
| **MultiPly** (CVPR 2024) | Multiple people from monocular video in the wild | Background reading for the single-view pipeline |

### 4.6 Long hair

| method | idea | use for us |
|---|---|---|
| **HADES** (ICCV 2025): explicit hair strands bound to Gaussians; SMPL-X body; hair motion from an MLP conditioned on current and past scalp states; multi-view | Dynamic strand hair + body in one avatar | Adopt the idea: hair Gaussians ride on strand polylines rooted on the scalp. Code not found. |
| **Strand-aligned Gaussians / "Gaussian Haircut"** (ECCV 2024) | Static strand geometry from multi-view | The idea of initialising from strands; licence not checked |
| **Our `avatar/hair_groom.json`** | 160 guide strands x 16 segments, rooted on SMPL-X scalp triangles; lengths 0.38-0.61 m (median 0.49); colliders; dynamics targets from `hair_reference.json` | **Initialisation for the hair Gaussians** (section 5.3) |

### 4.7 Phone video artefacts

| method | idea | use for us |
|---|---|---|
| **Motion-Aware Animatable Gaussian Avatars Deblurring** (arXiv 2411.16758, CVPR 2026) | Render the avatar at several instants within each exposure and average them to match the blurred photo | v2 option, for the spins only. Our SMPL-X is continuous in time (`avatar_uv.Body.at(t)` interpolates), so averaging sub-frame renders is cheap. |
| **RS-Avatar** (arXiv 2608.17314, 2026-08) | Rolling shutter: each image row sees a different instant; composite renders row band by row band | v2 option: the readout time per phone is unknown and has to be estimated |

---

## 5. Recommended method: AGS (avatar-guided splats)

### 5.1 Representation (per dancer, in the world frame, metres, Y-up)

1. **Body Gaussians bound to SMPL-X triangles** (GaussianAvatars / SplattingAvatar scheme).
   - Each Gaussian `i` stores: a parent triangle `f`, barycentric coordinates `(u, v)`, an offset `d` along the
     normal, a local rotation and log-scale, opacity, and colour (SH degree 1-2).
   - At frame `t` the posed triangle gives the position, the frame (`R_f(t)`) and the scale. The posed SMPL-X
     includes expression and the neck fix. Fingers come from the body fit (hand refinement was retired on
     2026-10-07).
   - Formulas:
     - `mu_i(t) = sum_k b_k v_{f,k}(t) + d_i n_f(t) + R_f(t) delta_i(t)`
     - `R_i(t) = R_f(t) R_i`
     - `s_i(t) = s_i * l_f(t) / l_f(rest)`
   - **Initialisation:**
     - Sample about 60k (follow body) and about 70k (lead) points over the UV atlas (`avatar_uv.UVAtlas`).
     - Colour from `avatar/<role>_albedo.png`, with the face from `<role>_albedo_face.png`.
     - Scale from the triangle size; opacity 0.5.
2. **Clothing and soft-tissue residual** `delta_i(t)`.
   - A low-rank trajectory per Gaussian: `delta_i(t) = sum_{k<K} c_ik phi_k(t)`, using a cosine (DCT) basis over
     the window.
   - K = 8 for skin and 12 for clothes; magnitude capped at about 3 cm for skin and about 6 cm for clothes, with
     L2 and smoothness penalties.
   - It is deterministic, smooth and tiny (K x 3 numbers per Gaussian). It is also exactly the form a compute
     shader can evaluate, like SqueezeMe's linear correctives.
   - A learned canonical offset along the normal handles loose jeans and the t-shirt.
3. **Hair Gaussians** (follow; section 5.3).
   - About 40k Gaussians on the 160 groom guides x 16 segments, plus children, oriented along the segments.
   - Node tracks are expressed in the head-joint frame.
4. **No free Gaussians.** Every Gaussian is bound to a triangle or a strand. MCMC relocation may move Gaussians
   between triangles, but never off the body. **So ghosts, room, spectators and mirror splats cannot exist.**
   This is the structural fix for the NoPo4D failure. Gaussians below the floor (`y < -1 cm`) are removed.

### 5.2 Supervision and losses

**Views.**
- Crops around the union of the two dancers' masks, padded 10 %, at native resolution.
- Each crop uses its frame's intrinsics (K) and pose (R, t). Sync-corrected time `t = frames.json t + offsets.json
  delta` is the time the bodies are posed at, as in the texture bake.
- Camera-type handling is in section 6.

**Joint rendering of both dancers.**
- One gsplat call with both dancers' Gaussians, so mutual occlusion is sorted correctly.
- Extra rendered channels: a one-hot role id. gsplat renders any number of colour channels, so one pass yields
  the RGB image, the alpha and each dancer's own alpha.

**Losses:**
- **Photometric:** masked L1 + 0.2 D-SSIM inside the union mask (`room.photo_loss`), after a per-image colour
  transform (`room.Appearance`: 3x3 + bias, the mean pulled to identity). Optional LPIPS in the last stage.
- **Silhouette:**
  - The union alpha is pulled towards 1 inside the union mask and towards 0 outside.
  - The background is random each step (`random_bg`), which forces opaque, closed dancers.
  - **Ignore pixels** are `people_masks` minus the dancers. That covers spectators and phone screens in front of
    the dancers, so they never carve holes in a dancer.
- **Role ownership:** each dancer's alpha vs their SAM mask, only where `roles.json` `swap_prob` is low. At pixels
  both dancers cover, the posed-mesh z-buffer (`avatar_raster.rasterize`) decides who should be in front
  (GeoAvatar / AvatarPose idea).
- **Hair:** the follow's hair-Gaussian alpha vs `dancers/<cam>/hair/` masks. Body Gaussians are kept out of
  hair-only pixels.
- **Interpenetration:** a hinge penalty when a body Gaussian of one dancer lies more than 1 cm inside the
  partner's posed SMPL-X. Use the signed distance (`external/pytorch_sdf`) or the closest-point normal. Frames
  and parts in contact are exempt. `hands/contact.npz` exists only for larissa_kadu (from runs before the hands
  stages were retired on 2026-10-07); for other takes, take contact from posed-mesh proximity.
- **Avatar prior:** face and hand Gaussians use a 10x smaller learning rate on colour and stay near the avatar
  albedo, except in views where the face is at least 15 px IOD (section 3.1). This keeps the portrait-quality face
  where the video is too small.
- **Per-frame pose refinement:** deltas of at most 3 deg per body joint and 2 cm translation, with an
  acceleration (second-difference) penalty, pulled towards MAMMA. They are written to
  `dancer_splats/pose_delta.npz`, never to `smplx/`.
  - Side benefit: the size of these deltas per frame is a **photometric check on the fit**. It shows where
    MAMMA was wrong, for example the follow at 60-61 s.
- **Regularisers:** the gsplat MCMC opacity and scale regularisers; a scale cap of 3 cm for body Gaussians.

**Schedule** (*estimate*, A100):

| stage | steps | what trains |
|---|---|---|
| S1 canonical | 6k | Body-bound Gaussians, colour transforms, pose deltas; residuals and hair off. SH degree 0. |
| S2 dynamic | 15k | Residual coefficients, hair node tracks, all of the above. SH degree 1. |
| S3 polish | 4k | SH degree 2, LPIPS, MCMC relocation off |

Each step renders 4 random views of one random instant. Rendering two dancers with about 170k Gaussians into
crops of about 0.3-1.3 MP takes a few ms per view forward and backward. Total *estimate*: **10-25 min**. For
calibration, the room's 30k steps at up to 1M Gaussians on full 1000-1920 px frames took 32 min on an A100
(*measured*, PIPELINE_PROGRESS).

### 5.3 Hair: groom-initialised, photo-refined

**Start from the groom:**
- Take each guide strand of `hair_groom.json`: its root triangle, segment length and rest position in head-bone
  space.
- Turn each into a strand of Gaussians, elongated along the segment, about 4 per segment, plus children at
  `child_spread_m`.
- Colour starts from the groom palette (`colour.strand_palette_linear`, root -> tip).

**Motion:**
- Per-frame node offsets in the head-joint frame, free per frame. They are kept physical by:
  - inextensibility (segment length +-5 %);
  - a fixed root;
  - an acceleration penalty;
  - collision with the head, neck and shoulder capsules (the groom's `colliders`).
- 160 x 16 x 105 x 3 is about 0.8 M parameters, which is cheap.
- Export can compress them to a cosine (DCT) basis afterwards (K = 24-32 keeps up to about 4-5 Hz over 3.5 s).

**Supervision:** hair masks, photometric loss, and her hair colour statistics (`hair_reference.json`) as a weak
prior.

**v2:** drive the nodes from the Unity strand simulation of the same take and only learn corrections. That would
make the splat hair and the viewer's card hair move the same way.

What to expect: in-sample the turns should show real trailing hair, which the textured avatar cannot show at all.
From novel angles the hair whip is constrained only by 8 blurred 30 fps views, so expect some softness. The
success test in section 7.4 measures this rather than assuming it.

### 5.4 Outputs (proposed contract, not built)

```
work/<take>/dancer_splats/
  bundle/                 prep: crops, masks (full/hair per role), views.json (K, R, t world frame per crop,
                          t_sync, camera type, sharpness), smplx inputs, avatar albedo/uv, groom, manifest (mtimes)
  model.pt                canonical Gaussians + binding + residual coefficients + hair tracks + pose deltas
                          + colour transforms (~40 MB)
  canonical_<role>.ply    rest pose (standard 3DGS PLY, SH1)
  frames/frame_{k:05d}.ply  per-frame posed splats for capture.json splat_sequence (generated LOCALLY from
                          model.pt, so only ~40 MB is downloaded)
  pose_delta.npz          per-frame pose corrections (QA signal only)
  eval.json               held-out camera + held-out instants: masked PSNR / SSIM / LPIPS (raw + colour-corrected),
                          per-role IoU, hair IoU, ghost fraction, avatar comparison on the same frames
  report.json             config, host / GPU, timings, Gaussian counts, gates
  review/dancer_splats_*.jpg  held-out real | splat | avatar sheets at 6 instants (incl. 59.18 s and 60.88 s),
                          orbit video from a fixed novel camera
```

**Monitor progress line** (fits `docs/PIPELINE_PROGRESS.md` section 5):

```
[dancer_splats] step N/M stage S1|S2|S3 psnr X iou Y N_gauss Z Ns
```

Steps have a known total per stage; the rate is close to constant because the Gaussian cap is reached early.

**Pipeline placement** (`docs/PIPELINE.md`, stage `dancer_splats`, already drawn as "future"):
- Depends on `fit_qa`, `texture`, `face_pod` (`smplx_face`) and `hair_groom`. The hands stages (`hands_build`,
  `smplx_detail`) were retired on 2026-10-07 and are no longer an input. The monitor's registry currently draws
  the stage with hard deps `fit_qa`, `texture`, `dancers_sam`, `cam_world` and a soft dep on `hair_reference`.
- Optional leaf.
- Runs on a pod with the `room` bootstrap feature (gsplat wheels).
- Local prep runs on the CPU; local export too.

---

## 6. Camera types in the splat stage

This ties into the user's 2026-10-07 request: handheld, tripod, has zoom.

The per-video camera type (`[cameras."<cam>"] mount = "handheld" | "tripod"`, `zoom = true | false`) changes
how the splat trainer treats each camera:

| type | what AGS does |
|---|---|
| **tripod** | One extrinsic for the whole video. Pose refinement is off, with at most one shared rigid correction. These are the best **held-out evaluation** cameras: stable view, background does not move. Motion blur comes from the dancers only. |
| **handheld** | Per-frame extrinsic refinement is on but small (rotation 0.3 deg or less, translation 1 cm or less, smoothed over about 0.2 s). Frames blurred by camera shake are down-weighted (bake sharpness score). Rolling-shutter wobble is more likely (v2 option). |
| **has zoom** | Intrinsics per frame (already in `cameras/<cam>.npz`). Optional small per-frame focal refinement for zoom cameras only. **Anti-aliased rasterisation** (`rasterize_mode="antialiased"`, Mip-Splatting filter, already used for the room) is mandatory, because the pixel footprint changes by up to 2.6x within one video. Zoomed-in frames are **up-weighted for face and hair** Gaussians (largest IOD). |
| **no zoom** | One fixed intrinsic per video: fewer unknowns, no focal refinement |

Data to pre-fill the type (larissa_kadu_38 `cameras/report.json`, *measured*):

| cam | camera path over 38 s | zoom range | suggestion |
|---|---|---|---|
| 01 | 1.19 m | 1.005 | handheld, no zoom |
| 02 | 0.98 m | 1.005 | handheld, no zoom |
| 03 | 0.85 m | 1.008 | handheld, no zoom |
| 04 | 1.18 m | **1.714** | handheld, zoom |
| 07 | 1.55 m | **1.372** | handheld, zoom |
| 09 | 4.27 m | **2.573** | handheld, zoom |

Cameras 05 and 06 show 0.0 m, but they were attached with a **rotation-only** model. Their fixed centre is an
assumption of that model, not a measurement, so the user has to confirm them.

---

## 7. Prototype plan (needs user approval before any pod)

### 7.1 Data

**Take: `larissa_kadu`, 58.0-61.5 s** (105 frames on the 30 Hz grid, all 8 cameras). Why this take and window:
- **All inputs exist and are fresh:**
  - `smplx/` (2026-10-06 12:19) is newer than `world_from_sfm.json` (05:10).
  - `smplx_face/`, `avatar/` (albedo, faces, `uv_mesh`, `head_fix`) and `hair_groom.json` (2026-10-07) are all
    present. The body motion is `smplx/` (the take has no `[export] smplx_dir`); `smplx_detail/` on disk is from
    the retired hands stages and is not used.
- **The window holds her three fastest head turns** (`hair_groom.json` `motion`: 59.18 s at 284 deg/s, 60.88 s at
  299 deg/s, 61.41 s at 169 deg/s), so the hair moves a lot.
- **It includes the worst body-fit stretch** (follow 8-11 px at 60.0-61.0 s), which tests pose refinement.
- **Camera 09 zooms** inside it (1.49x), which tests per-frame intrinsics.
- **It does not touch `larissa_kadu_38`,** whose textures are being re-baked right now. Moving to the 38 s take
  comes later, once the method passes.

**Held-out data:**
- **Camera 03** for the whole window: main model, 1.0 px, mid face size.
- Every 10th instant of the other cameras, for the temporal check.

**Inputs uploaded** (*estimate* about 90 MB):

| item | size |
|---|---|
| 840 dancer crops, JPEG q95 | ~50 MB |
| masks: per-role full + follow hair | ~13 MB |
| `views.json` | small |
| `smplx` (or the take's `[export] smplx_dir`) + `smplx_face` + `head_fix` | ~2 MB |
| albedos + face albedos | ~20 MB |
| `uv_mesh.npz`, `hair_groom.json`, `contact.npz` (larissa_kadu only, pre-retirement) | ~2 MB |

The SMPL-X model and the code travel in the runner's usual bundle.

### 7.2 Steps

1. **Prep (local CPU, about 5 min).** `python -m dancecap.dancer_splats prep larissa_kadu --window 58.0 61.5
   --holdout 03` (new module, not written yet) builds `dancer_splats/bundle/`:
   - converts the SfM-frame cameras to the world frame (`world_from_sfm`);
   - crops the frames and rescales K for each crop;
   - samples bodies at sync-corrected times;
   - attaches the camera types and per-view sharpness;
   - checks freshness (refuses `smplx/` older than `world_from_sfm`, like the bake).
2. **Smoke (local, shared 2080 Ti, at most 10 min, `gpu.exclusive()` lock).** 300 steps at half resolution on a
   0.5 s window. Checks that the losses stay finite, the binding is correct (render overlaid on the frame) and a
   frame PLY loads in Unity. This is the only local GPU use.
3. **Pod session (approval needed).**
   - Run `python -m dancecap remote run --take larissa_kadu --jobs dancer_splats` (a new
     `dancer_splats/pod_job.json`; bootstrap features `base,room`).
   - The pod trains S1-S3, evaluates the held-out data, renders review sheets, and packs `model.pt` + `eval.json`
     + `review/`. It is always deleted at the end.
   - The runner and its ledger are unchanged; this is only a new job description. No change to `remote.py` or
     `remote_jobs.py` is needed.
4. **Export and review (local CPU).**
   - Generate `frames/frame_{k:05d}.ply` from `model.pt`, capped at about 120k Gaussians per frame for the disc
     preview.
   - Add a `splat_sequence` entry to the capture (an additive `demo_export` hook).
   - Play it in Unity with layer G next to the avatars and skeletons.
5. **User A/B.** On the review sheets and in Unity, decide: extend to the 38 s take, build the proper renderer
   (section 8), and/or try LHM++ or simulation-driven hair.

### 7.3 Cost (*estimate*, from measured runner rates)

| item | typical | slow case |
|---|---|---|
| pod create -> ssh ready | 0.6 min | 0.8 min |
| upload ~90 MB (median 2.5 MB/s; p90-slow 0.7 MB/s) | 0.6 min | 2.2 min |
| bootstrap `base` + `room` (cached gsplat wheels) | 1.5-2.5 min | 7 min (slow PyPI) |
| training S1-S3 | 10-25 min | 30 min |
| held-out evaluation + review renders | 3-5 min | 6 min |
| pack + download ~60 MB | 0.5 min | 1 min |
| **session** | **17-37 min** | **~47 min** |

| GPU (ledger price) | typical | cap at `max_minutes` 60 |
|---|---|---|
| A100 PCIe community $1.19/h | $0.35-0.75 | $1.19 |
| A100 SXM community $1.39/h | $0.40-0.90 | $1.39 |
| L40S community $0.79/h (48 GB is plenty; training maybe 1.3-1.6x slower) | $0.30-0.60 | $0.79 |

- **Budget to request:** $1.50 per session cap, **at most $2 in total** including one retry. That is about 20 % of
  what is left of the $30 cap.
- **Optional add-on in the same session:** the LHM++ install plus 2 inference runs (canonical starting point and a
  baseline), +15-40 min = +$0.30-0.90. **Recommended to postpone** until the first run shows whether the
  remaining gap is in the starting point.

### 7.4 Success criteria (vs the current avatar, on the held-out data)

| # | criterion | threshold |
|---|---|---|
| 1 | **No ghosts or duplicates** (the NoPo4D failure) | Rendered alpha outside the 2 % dilated union dancer mask: 1 % of the crop or less, on every held-out frame |
| 2 | **Silhouette** (held-out camera 03, per role vs SAM) | Follow IoU 0.85 or more (avatar 0.66-0.83, no hair); lead IoU 0.88 or more (avatar 0.77-0.88) |
| 3 | **Colour** (held-out camera, inside the mask, after an affine colour fit like `room` eval) | Masked MAE (sRGB) 0.6x the avatar's on the same frames or better; masked PSNR 24 dB or more held out and 27 dB or more in-sample (NoPo4D: 10.4-13.6 dB in-sample) |
| 4 | **Hair in the turns** (59.18 s, 60.88 s, held-out camera) | Hair-mask IoU 0.5 or more, and trailing hair visible on the review sheet |
| 5 | **Temporal stability** (fixed novel camera halfway between cameras 03 and 04) | Mean frame-to-frame change on the body (hair excluded) no more than the avatar's + 10 %; no popping on the orbit video |
| 6 | **Face** | No worse than `avatar_face` at camera 02 (user's eye); no target beyond that (faces paused) |
| 7 | **Unity** | Plays all 105 frames in layer G on PC at 30 fps or more, with feet within 2 cm of the avatar's feet and the floor |
| 8 | **Cost** | Training 30 min or less; session $1.50 or less |

**Stop rule:**
- If criteria 1-3 fail, write up the failure (which loss or which camera) and do **not** start a second pod
  without the user's approval.
- If 1-3 pass and 4 fails, the next step is the hair (v2 simulation-driven tracks, deblur), not more body
  training.

---

## 8. Unity playback and the headset

**Phase 1 (prototype, PC, no Unity code change).**
- Uses the existing `CaptureManifest.splat_sequence` + `SplatCloud`: discs, unsorted, DC colour only, background
  loading, 12-frame cache.
- The world-to-Unity matrix is the one `export_unity` already uses for the avatars.
- Size, *estimate*: 105 frames x 120k Gaussians x 68 B (float PLY, SH0) = about 860 MB in `work/` and the
  git-ignored StreamingAssets. A quantised frame format (about 18 B per Gaussian) would cut that to about 230 MB.
- Good enough to judge placement, silhouette, hair motion and colour. Not good enough to judge sharpness, because
  the disc shader blurs everything.

**Phase 2 (real renderer).** Candidate: **ninjamode/Unity-VR-Gaussian-Splatting**.
- MIT, forked from aras-p's UnityGaussianSplatting. URP with Render Graph; Unity 6000.3 or later (we use
  6000.6.3f1, URP 17.6); D3D12, Vulkan or Metal.
- Quest Pro / Quest 3 support: about 72 fps up to around 400k Gaussians with tuned settings; AMD FFX sort
  (*reported* by search results; to verify).
- Alternatives: Prempi1364's Quest 3 multiview fork (MIT, but **Built-in pipeline only**, so it does not fit our
  URP project); SplatBridge (1.0.4, 2026-09, by Stray Cats; licence not checked; Quest 3 experimental).
- Add a **skinned-splat compute pass**:
  - canonical buffer + binding (triangle id, barycentric, local frame) + residual coefficients + hair node tracks;
  - each frame, apply the SMPL-X pose the capture already carries (55 joint rotations), then the cosine (DCT)
    residual, then the renderer's sort and rasterisation.
  - The project has no compute shaders yet; the renderer brings its own.
- Data, *estimate*: 170k Gaussians x (binding 16 B + attributes about 40 B at half precision with SH1 + residual
  12 x 3 x 2 B) is about 17 MB per take window, plus hair tracks of about 2.4 MB. That is far smaller than a
  per-frame sequence.

**Headset feasibility (information for the user, not a plan).**
- Two dancers at about 170k Gaussians is under the reported Quest budget of about 400k.
- SqueezeMe shows 3 skinned Gaussian avatars with correctives at 72 fps on Quest 3.
- So a Quest splat layer is technically plausible. The 2026-10-05 decision (headset = textured skinned avatars,
  hair cards with the tip bloom) stands until the user revisits it.
- Splat hair has no tip bloom. If splats ever ship, the glow would need an overlay.

---

## 9. Single-view and two-view captures

AGS is the same algorithm with fewer cameras. The data and priors change:

**Two views:**
- The canonical model still adds up information over time: a spinning dancer shows every side within a few
  seconds. Require at least one full turn in the window.
- Stronger residual regularisation: K = 4-6, larger smoothness.
- Pose refinement only with an AvatarPose-style collision term.
- Evaluate on held-out instants (no spare camera).
- Use LHM++ (multi-image, pose-free, built from frames of both cameras across the spin) as the starting point and
  as a weak prior for regions not seen at time `t`.

**Single view:**
- This is the setting most published avatar methods target (ExAvatar, GaussianAvatar, 3DGS-Avatar). Expect good
  in-sample results and weaker novel views.
- LHM++ starting point recommended; residuals limited to hair and clothes.
- Depth ambiguity: Gaussian offsets along the camera ray are penalised more strongly than sideways ones.

Camera types matter more here:
- A tripod single view gives a clean background and stable evaluation.
- A handheld single view needs per-frame camera refinement against the static room (the existing SfM / room
  stage), not against the dancers.

---

## 10. Risks and open questions

1. **Body fit error (5-11 px) and video sync.** If pose refinement cannot absorb them, cross-view disagreement
   shows up as blur rather than ghosts, because the binding prevents duplicates. Watch `pose_delta.npz` and the
   per-camera residuals.
2. **Hand holds and embraces.** Ownership mistakes can swap Gaussians between the dancers' hands. Mitigations:
   - contact frames exempt from the collision penalty, but role-ownership loss kept;
   - masks whose `swap_prob` is above 0.2 dropped.
3. **Motion blur in the spins.** It bakes a blurred appearance into the canonical model.
   - First run: down-weight the blurriest views.
   - v2: sub-frame rendering, only for frames flagged blurry (2-3x the cost there).
4. **Rolling shutter:** not modelled (v2: RS-Avatar). Visible as skew in the fastest arm and hair motions.
5. **Low resolution:** six cameras at 480p. Expect fine texture (prints, jeans seams) to match the avatar's bake
   at best. The win is in shape (hair, clothes, silhouette) and true per-frame appearance, not texel sharpness.
6. **Licences.** gsplat (Apache-2.0), our code and SMPL-X (non-commercial) are fine for research. These are
   **not verified**: Animatable Gaussians, MMLPHuman, STG-Avatar, HUGS (Apple licence) and Gaussian Haircut. All
   outputs are likeness data and stay local; no splat ever goes into `head-movement`.
7. **The disc preview hides quality.** Phase-1 Unity playback will look softer than the review renders. Judge
   sharpness on the gsplat review sheets, not in Unity, until Phase 2.
8. **Budget.** A failed bootstrap (a 20 min PyPI stall cost $0.40 on 2026-10-07) is the main way the cost could
   run over. Use cached wheels and set `max_minutes` to 60.

---

## 11. Sources

- NoPo4D: `external/NoPo4D/README.md`; arXiv 2605.22190 ([arxiv.org/abs/2605.22190](https://arxiv.org/abs/2605.22190))
- ExAvatar: [github.com/mks0601/ExAvatar_RELEASE](https://github.com/mks0601/ExAvatar_RELEASE),
  [arXiv 2407.21686](https://arxiv.org/abs/2407.21686)
- Animatable Gaussians: [github.com/lizhe00/AnimatableGaussians](https://github.com/lizhe00/AnimatableGaussians/blob/master/README.md),
  [arXiv 2311.16096](https://arxiv.org/html/2311.16096v3)
- MMLPHuman: [arXiv 2504.12909](https://arxiv.org/html/2504.12909v1), [gapszju.github.io/mmlphuman](https://gapszju.github.io/mmlphuman)
- LHM / LHM++: [github.com/aigc3d/LHM](https://github.com/aigc3d/LHM), [github.com/aigc3d/LHM-plusplus](https://github.com/aigc3d/LHM-plusplus),
  [arXiv 2506.13766](https://arxiv.org/html/2506.13766), weights licence [LICENSE_WEIGHT](https://raw.githubusercontent.com/aigc3d/LHM-plusplus/main/LICENSE_WEIGHT)
- SqueezeMe: [arXiv 2412.15171](https://arxiv.org/html/2412.15171v4), [project page](https://forresti.github.io/squeezeme/),
  [UploadVR article](https://www.uploadvr.com/meta-squeezeme-mobile-ready-distillation-of-gaussian-full-body-avatars/)
- TaoAvatar: [arXiv 2503.17032](https://arxiv.org/pdf/2503.17032), [project page](https://pixelai-team.github.io/TaoAvatar/)
- DualGS: [arXiv 2409.08353](https://arxiv.org/abs/2409.08353), [github.com/HiFi-Human/DualGS](https://github.com/HiFi-Human/DualGS)
- HiFi4G: [arXiv 2312.03461](https://arxiv.org/abs/2312.03461)
- 3DGS-Avatar: [github.com/mikeqzy/3dgs-avatar-release](https://github.com/mikeqzy/3dgs-avatar-release)
- GaussianAvatar: [github.com/aipixel/GaussianAvatar](https://github.com/aipixel/GaussianAvatar)
- GauHuman: [github.com/skhu101/GauHuman](https://github.com/skhu101/GauHuman)
- HUGS: [github.com/apple/ml-hugs](https://github.com/apple/ml-hugs)
- STG-Avatar: [arXiv 2510.22140](https://arxiv.org/html/2510.22140)
- GaussianAvatars: [github.com/ShenhanQian/GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars);
  SplattingAvatar: [github.com/initialneil/SplattingAvatar](https://github.com/initialneil/SplattingAvatar)
- AvatarPose: [arXiv 2408.02110](https://arxiv.org/abs/2408.02110), [eth-ait.github.io/AvatarPose](https://eth-ait.github.io/AvatarPose/)
- GeoAvatar: [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_GeoAvatar_Geometrically-Consistent_Multi-Person_Avatar_Reconstruction_from_Sparse_Multi-View_Videos_CVPR_2025_paper.html)
- Close interaction (Huang, CVPR 2025): [arXiv 2507.02565](https://arxiv.org/html/2507.02565); MultiPly: [arXiv 2406.01595](https://arxiv.org/pdf/2406.01595)
- HADES: [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liao_HADES_Human_Avatar_with_Dynamic_Explicit_Hair_Strands_ICCV_2025_paper.html);
  strand-aligned Gaussians: [arXiv 2409.14778](https://arxiv.org/pdf/2409.14778)
- Deblurring avatars: [arXiv 2411.16758](https://arxiv.org/abs/2411.16758); RS-Avatar: [arXiv 2608.17314](https://arxiv.org/abs/2608.17314)
- gsplat: [JMLR paper](http://jmlr.org/papers/volume26/24-1476/24-1476.pdf) (Apache-2.0)
- Unity renderers: [ninjamode/Unity-VR-Gaussian-Splatting](https://github.com/ninjamode/Unity-VR-Gaussian-Splatting),
  [aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting),
  [Prempi1364 Quest 3 multiview fork](https://github.com/Prempi1364/UnityGaussianSplatting-Quest-Multiview-BuiltIn),
  [SplatBridge](https://radiancefields.com/splatbridge-brings-streamed-gaussian-splats-and-splat-colliders-to-unity-6)
- Local evidence: `prototype/**/report.json`, `quality_gate.json`, `geometry_check.json`; `work/larissa_kadu/{cameras/report.json,
  face/survey.npz, smplx/*.npz, avatar/report.json, avatar/hair_groom.json}`; `work/larissa_kadu_38/cameras/report.json`;
  `work/runpod/ledger.json`; `docs/PIPELINE_PROGRESS.md` section 1 (measured rates)

---

## 12. Implementation status (2026-10-08)

**Built:** `dancecap/dancecap/dancer_splats/` (`python -m dancecap.dancer_splats <take> {prepare, write-pod-job, train,
eval, export, render, run-pod}`):
- `binding.py`: triangle frames, the body binding of 5.1 (barycentric anchor + tangent / normal offset in triangle
  units + local rotation and log-scale, scale cap 3 cm), the pose-correction chain, re-skinning, the cosine basis,
  SH, strand frames, the groom's rest strands.
- `data.py`: the prep (`bundle/`, CONTRACTS.md), labels, the avatar reference renders.
- `model.py`: the Gaussian groups (lead, follow, hair), posing at any time, batched gsplat rendering with one-hot role
  channels, MCMC relocation / growth on the anchors.
- `train.py` (S1 6k / S2 15k / S3 4k steps, 4 views per step), `evaluate.py` (metrics, gates, review sheets),
  `export.py` (PLYs, `splat_sequence.json`, `render`), `pod.py` (the pod job).
- Tests: `tests/test_dancer_splats.py`.
- Pipeline: the registry stage is implemented; deps fit_qa, texture, hair_groom; soft dep face_pod; never the
  retired hands outputs. It has its own pod session S_splats (approved ceiling $1.50, kill switch 55 min) and done
  markers `model.pt` + `metrics.json` + `pose_delta.npz`. Live progress uses L0 `[dancer_splats] step N/M stage S psnr
  X iou Y N_gauss Z Ns` and L2 `splats/train`, `splats/eval`. The ETA is linear on the measured s/step, and the run
  halts at a review gate on `review/splats/`.

**As designed (sections 5.1-5.4, 7):**
- Every Gaussian is bound to a SMPL-X triangle or a groom strand segment. There are no free Gaussians, and
  relocation never leaves the anchors.
- Per-frame pose corrections: 22 body joints at most 3 deg each, root translation at most 2 cm, L2 + acceleration
  penalty. They go to `pose_delta.npz`; `smplx/` is never written.
- Cosine residuals: K = 12, skin 8; capped at 3 cm for skin and 6 cm for clothes.
- Hair node tracks are free per frame in the head frame, with:
  - inextensibility +-5 %;
  - a fixed root;
  - an acceleration penalty;
  - the head sphere and the groom's capsules.
- Losses:
  - masked L1 + 0.2 D-SSIM on a random background, after the per-image colour transform (`room.Appearance`);
  - silhouette, role ownership (`swap_prob` at most 0.2) and hair channels;
  - an interpenetration hinge;
  - MCMC opacity / scale regularisers;
  - a floor hinge at y < -1 cm.
- Held out: camera 03 plus every 10th instant of the other cameras.
- Gates 1-8 of 7.4 are in `metrics.json`. 6 (face) and 7 (Unity) stay manual.

**Deviations / simplifications in the prototype:**
1. **Bootstrap feature `gsplat`.** A new lean feature (`remote/bootstrap.sh`, `remote_jobs.FEATURES`): the cached
   gsplat 1.5.3 wheel + plyfile, not `room`. This avoids open3d's ~450 MB PyPI download, which is the main stall
   risk of 10.8.
2. **Pose corrections.** They re-skin the fit's posed vertices with D_j = G'_j G_j^-1 from the stored global joint
   rotations. So no SMPL-X model goes to the pod. The pose correctives of the tiny change are ignored.
3. **Batched views.** The step's views (one instant, other cameras within 0.75 frame) are posed and rendered in ONE
   gsplat call (batch dim, crops padded to the largest, covariances passed directly). The step time does not depend
   on the Gaussian count; it is kernel-launch bound.
4. **Fixed cameras.** No per-frame extrinsic / focal refinement (section 6, handheld / zoom). Antialiased
   rasterisation is always on.
5. **LPIPS is evaluation-only** and optional (lpips 0.1.4 on the pod, skipped if unavailable). It is not an S3 loss.
6. **Avatar prior.** Face and hand colours get a x0.1 learning rate and a weak L1 to the albedo. Both are released
   for the face in steps that include a view where a face is at least 15 px IOD (from the posed eye joints).
7. **Interpenetration.** Nearest-vertex normal test on 2048 sampled Gaussians per step, no SDF. Gaussians whose
   anchor is less than 2 cm outside the partner (contact) are exempt.
8. **Rest strands.** Roots and lengths come from `hair_groom.json`. The shape is: over the skull along a great
   circle (front roots swept behind the ear, so none starts over the face), then down, settled by a short
   position-based drape in the T-pose (gravity, exact segment lengths, the head sphere and the groom's capsules WITH
   their offsets: head, neck, shoulder bar, chest, mid-back; arms from point 5). The Unity-side comb / style blocks
   are not reproduced; the photos refine the strands.
   - **Frames (fixed 2026-10-07 after the pre-flight review):** `head_sphere_centre_unity`, `guides[].pos_unity`
     and `colliders[].o?_unity` are Unity offsets FROM THE HEAD JOINT (`to_unity(c - J_head)`), so rest =
     `J_rest[15] + flip_z(offset)`. The first prep read the sphere centre as an absolute soles-at-0 position: the
     "head" sat 1.53 m below the head joint and the strands stuck out ~33 cm sideways (71.5 % of the strand nodes
     projected onto background). Now the prep checks `head_joint_rest` against this fit's head joint (1 cm) and the
     roots against `pos_unity` (median 0.05 mm), and 3.1 % of the nodes 4-16 project onto background, 48.8 % onto
     the hair label, 94.5 % onto the dancers (657 training views, fit poses, no corrections).
   - **Colliders in training** use the corrected bone rotation x the rest-axes offset (`AGSModel.collider_ends`);
     before, the offsets were dropped (the head capsule sat 7 cm low, the shoulder bar 14 cm low).
9. **Avatar reference.** `avatar/<role>_albedo.png` (the video bake), unlit, on the same posed SMPL-X. That pose
   includes the adopted head fix and the shipped `smplx_face` expression and jaw. The prep renders it on the CPU.
   It is not the portrait face that Unity shows.
10. **People masks.** The detector's 10 px dilation ring around the dancers is removed from the ignore pixels.
    Otherwise the silhouette loss could not see the outline.
11. **Camera 02 (59.94 fps).** The prep keeps the frame nearest each 30 Hz grid instant.
12. **Window.** `takes/<take>.toml [dancer_splats] window`, else the approved one (larissa_kadu 58.0-61.5), else
    3.5 s centred in the take window.
13. **Time budget.** Training shrinks its schedule to end within 24 min, and 15 min before the pod's kill switch
    (`/workspace/.dancecap_kill_at`). So the eval, pack and download always fit.
    - The shrink uses the step time of the last ~500 steps (not the run's average).
    - **Hard stop:** when the budget is spent (or on SIGTERM, e.g. the command's `timeout`), the current step
      finishes, the schedule is truncated there and `model.pt` + `pose_delta.npz` are still written, so eval runs.
14. **Retry across pods.** The pod is always deleted, so `dancer_splats/cache/ckpt.pt` (every 2000 steps) is packed
    with the outputs even after a failure. `run-pod --retry` uploads it (or `model.pt` + `pose_delta.npz` when only
    eval failed; train is then skipped) and caps the session at $2.00 minus the earlier splat sessions' ledger cost.
    A checkpoint from another host starts that session's time budget afresh.
15. **Evaluation extras.** Per dancer (each one's own mask pixels; the overlap is excluded) and face / hair crops of
    cameras 02 / 06 / 09 (32 cm face square around each skull, 60 cm hair square from the follow's crown down),
    splat vs avatar PSNR / SSIM / LPIPS raw and colour-corrected (`metrics.json` `tables`). SSIM runs on the GPU.
16. **Local review sheets** (`python -m dancecap.dancer_splats <take> review`, run by `run-pod` after the download
    with the export): video | textured avatar | splats at the 6 review moments for camera 03 (held out) and camera
    02 (training), a 12-view novel turntable at 59.18 s, hair and face close-ups (02 / 06 / 09 in the three turns),
    and 12-frame temporal strips (camera 03 and the fixed novel camera).

**Measured locally (2026-10-08):**
- **Prep, larissa_kadu 58.0-61.5 s:**
  - 832 views: 657 train, 105 of camera 03, 70 held-out instants. 8 frames were dropped by the cameras' ok flags.
  - The bundle is 101.8 MB, built in 185 s on 4 CPUs. The avatar renders took 49 s for 215 views.
  - The pod upload is 255 MB raw: 102 MB bundle, 135 MB base wheels, 13 MB gsplat wheel, 6 MB code.
- **Smoke run** (cameras 02 + 04 at 360 px or less, 59.0-59.2 s, 32k Gaussians, 300 steps x 2 views, shared 2080 Ti):
  - 42 s in total.
  - In-sample masked PSNR 14.9 -> 22.3 dB, IoU 0.82 -> 0.97.
  - Eval, export (PLYs on the floor, 1.73 m tall) and render all ran.
  - Review: `work/larissa_kadu/review/splats/smoke_*`.
- **Pod-scale step timing** on the real bundle (171k Gaussians: 70k lead, 60k follow, 41k hair; 4 native-resolution
  views per step; canvases up to 772 x 1075):
  - 0.15 s/step in S1 and 0.17 s/step in S2 on the shared 2080 Ti; peak VRAM 1.5 GB.
  - *Estimate* for an A100 on Linux: about 0.06 s/step, so 25k steps take about 25 min. The 24 min budget may trim
    the schedule a little.

**Run (approved budget: one session, $1.50 cap, at most $2 with one retry):**
- `python -m dancecap remote run --take larissa_kadu --jobs dancer_splats --max-cost 1.5 --max-minutes 55`.
- The bundle and `pod_job.json` are written.
- Start it only when no other session is uploading. `python -m dancecap.dancer_splats larissa_kadu run-pod` waits
  for that by itself.
- Then `python -m dancecap.dancer_splats larissa_kadu export`.
- The `demo_export` hook (`splat_sequence`) is not wired yet.
