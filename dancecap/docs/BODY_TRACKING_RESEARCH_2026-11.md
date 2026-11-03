# Body tracking research 2026-11: foot slip, awkward poses, partner dance on one to three phones

**Status 2026-11-03.** The owner's verdict this morning: "we are not quite there with two source video, and we are really
not there with monocular. there is too much foot slip, awkward body positions, etc." This report covers the research round
that followed. It asks which 2025-2026 work can fix those failures in our setting, ranks the options and sets out the first
experiments on our own data.

**How it was made.** A research workflow ran:
- five literature sweeps: foot contact and physics, close contact, sparse phones, dance, newest releases;
- one sweep of this repo's own results;
- three adversarial verifiers, which re-read the primary sources of the 18 strongest candidates;
- this synthesis, which re-checked the numbers it relies on;
- a second check the same day: 8 skeptics re-read 36 of the report's decision-driving claims against the repo, the
  work files, git and the primary pages. 24 were confirmed and 12 corrected in place; none was refuted.

Nothing was run on pods for this report. Every claim about a method links its primary source. Labels:
- **measured** = our runs, with the file or commit named;
- **stated** = the authors' own numbers;
- **reasoned** = our inference, not measured.

PA22 means PA-MPJPE over the 22 SMPL-X body joints, in mm, given as lead/follow.

**Coverage caveat.** The round used up its shared web-search allowance. The newest-releases sweep and the end of the dance
sweep worked from direct fetches of primary pages, and this synthesis could fetch known pages but not search. Work released
after about mid-October 2026 may be missing if none of the fetched papers cite it.

## In short

- **No released method reconstructs a couple in close embrace from phone video with grounded feet.** MAMMA stays the
  backbone. The fixes are parts around it:
  - per-person priors where the cameras are blind;
  - a soft, contact-weighted foot term;
  - better camera geometry: a well-placed third phone, and a correct camera height for one phone.
- **The cheapest wins are in our own code.**
  - A soft slip term we already have (`rf2`) cut eight-phone stance slip from 93/119 to 77/113 mm/s with unchanged feet
    reprojection (7.45/8.39 against 7.43/8.37 px). The shipped recipe has no slip term. The follower's 113 only
    matches the filter-only fit, so the real gain is the lead's.
  - The one-phone fit turns its SAM 3D Body prior down to 25-50 % of its weight exactly while the follower is hidden.
    Her heading is then right, but her hidden-frame error stays about 170 mm.
  - The two-view fit has no pose prior at all.
- **Released tools worth one test each:**
  - ExMR, a one-phone prior with better feet;
  - FlowHMR and DuoMo, alternatives for the prior and for the hidden span;
  - FECO and Sapiens2, for foot contact and foot keypoints;
  - d4sm or SAM 3.1, for masks in the embrace;
  - MAMMA upstream `dev/v1.1.0`.
- **No paper measures two versus three cameras for a couple.** We can measure it on larissa_kadu_38 by fitting subsets of
  its phones (E1, about $1-1.5). The third phone should go about 120 degrees from the other two, and raised.
- **One-phone placement is a camera-height problem, not a focal problem.** This was measured this round: on the lk03 cut the
  focal matches the eight-phone value within 0.15 %, but the couple is placed 9.5 % too far (section 2, finding 6).
- **First experiment: E1**, two against three phones on our eight-phone take. E2, the foot term, can run beside it.

## 1. The problem in one paragraph

Our fits fail in five ways, and how badly depends on the number of phones. **Feet slip**: with eight phones the stance feet
still move 93/119 mm/s (lead/follow, contact point on the reference's stance frames, `work/larissa_kadu_38/quality/variants.json`)
because the adopted refine recipe `smplx_rf2_noslip` has no slip term, and with one phone (john_lula) the fit slid 6.5-7.6 mm
per frame inside the stances the video shows, while feet near the floor still move a median 9.3/13.5 mm per frame (p90 about
64) after the john_lula-only foot lock. **Bodies take awkward positions**: limbs that no camera sees take odd angles, because
the multi-view and two-view fits have no pose prior, and hands pass through the partner (three contact-solver rounds made
the arms worse). **Partner relations break**: one phone misjudges the couple's depth gap by 64-132 mm at the recommended
prior weight (49-118 at weight 0.5, 90-147 without the prior), and a lucas_luiza seam where the two cameras saw the
couple only 22-28 degrees apart was 0.1-0.25 m off. **The hidden follower is the worst case**: when she spins behind the leader
(larissa_kadu_38, 44-48 s), every one-phone MAMMA fit is about 170 mm off on the 43 hidden frames, against 81 mm for SAM 3D
Body alone. **Spins and identity fail last**: without the prior the follower's torso heading is more than 90 degrees off on 19 %
of the 44-48 s spin window (30 % of its hidden frames; none with the prior at weight 2), every model under-rotates
between front and side, and on lucas_luiza's Steadicam SAM lost a dancer for 4.4-5 s, once in an embrace (the pipeline
now re-prompts a lost dancer from its box, c2cdc18). With one
phone the weak axis is depth along the camera ray; with two, any stretch that one phone loses becomes a one-phone stretch;
with three we do not know yet, because no three-phone take exists.

Sources: the repo baseline in
[MONO4D_NOVEL_VIEWS.md](MONO4D_NOVEL_VIEWS.md) s10.6, branch `claude/vigilant-moser-a1afa5` (85c9f78),
[PIPELINE_EFFICIENCY.md](PIPELINE_EFFICIENCY.md) s7e, [`../CONTRACTS.md`](../CONTRACTS.md) (refine recipes) and
[RECAP_FILM.md](RECAP_FILM.md) (hands).

## 2. What the field can do now (2026)

**Finding 1. No released method reconstructs a couple in close embrace from phone video with grounded feet.**
- Partner-dance work from 2024 to 2026 is almost all generation, for example
  [Duolando](https://arxiv.org/abs/2403.18811), [DuetGen](https://arxiv.org/abs/2506.18680) and
  [InterDance](https://arxiv.org/abs/2412.16982).
- The reconstruction work nearest to us had no code on its pages when the sweeps read them:
  - [SocialMirror](https://arxiv.org/abs/2604.13581) infills occluded bodies with VLM guidance;
  - [GNOCHI](https://arxiv.org/abs/2607.10408) generates a partner's pose from the other dancer's;
  - [DanceHMR](https://arxiv.org/abs/2605.18102) does dance HMR that refines the root from static feet.
- The released two-person priors were trained on hugs and handshakes, and on our data they did not help (measured):
  - CloseApp tied fit B, 48/44 against 47/44 mm ([MONO4D_NOVEL_VIEWS.md](MONO4D_NOVEL_VIEWS.md) s10.6);
  - PromptHMR's pair prompt added 4 front/back errors on john_lula (`docs/PROMPTHMR_JOHN_LULA.md` on branch
    `claude/nifty-maxwell-2d0320`).
- For us: keep MAMMA and add parts. The parts are per-person priors, a contact-weighted foot term and better camera geometry.

**Finding 2. Per-person priors keep getting better, and masks matter more than two-person models.**
- [SAM 3D Body](https://arxiv.org/abs/2602.15989) (stated):
  - Mask prompts cut MPJPE from 35.6 to 30.1 mm on Harmony4D and from 76.4 to 47.0 mm on Hi4D (Table 8).
  - On held-out Harmony4D it scores 54.0 mm, against 78.0 for PromptHMR (Table 3).
  - It works on single images, one person per crop.
- Our data agrees (measured): mask-prompted SAM 3D Body is our best per-frame prior (34/43 mm alone, 31/36 inside
  MAMMA_MONO).
- New in October 2026, all with released weights (stated):
  - [ExMR](https://arxiv.org/abs/2610.26213) ([code](https://github.com/twehrbein/ExMR)) adds a temporal model and
    foot/hand refiners on SAM 3D Body features, with SMPL-X output.
    - RICH PA-MPJPE: 27.3 mm, against 33.0 for the released SAM 3D Body in the authors' re-run. SAM 3D Body's own paper
      reports 30.9.
    - Acceleration error on 3DPW: 4.1 against 14.0 m/s^2 (RICH: 3.6 against 10.6). Against GVHMR the gain is small
      (5.0 to 4.1). MOYO ankle-angle error: 20.0 against 24.4 degrees.
    - Foot contacts have been in its output since v0.1.5.
  - [FlowHMR](https://arxiv.org/abs/2610.03691) ([code](https://github.com/flowhmr/flowhmr)) is post-trained with a physics
    tracking reward (PHC+) and outputs SMPL-H.
    - On Wild-4K, 82.47 % of its motions can be tracked in simulation, against 62.82 % for GVHMR.
    - Its ground penetration is 0.06 mm and its float 0.20 mm, but its skate is 2.17 against GVHMR's 1.59.
  - [DuoMo](https://arxiv.org/abs/2603.03265) (CVPR 2026, [code](https://github.com/facebookresearch/DuoMo)) keeps
    generating motion through frames with no visible keypoints. EMDB foot skating: 3.7, against 3.5 for GVHMR.
- All three take one person per crop and know nothing of the partner. ExMR (YOLO11 + ByteTrack boxes, per its README)
  and FlowHMR take a box, not a mask.
- For us: these are candidates for the prior inside MAMMA (one phone, and a new prior for two phones), not replacements for
  the fit.

**Finding 3. Every released system that reports low foot skating does the same thing: it predicts contact, then softly
penalises the speed of the contact point inside an optimisation.**
- Examples (stated):
  - [PromptHMR-Vid](https://arxiv.org/abs/2504.06397): per-joint contact plus zero velocity; EMDB-2 foot skating 3.5.
  - [JOSH](https://arxiv.org/abs/2501.02158): its contact-static term cuts SLOPER4D foot sliding from 68.2 to 28.2 mm.
  - [RoHM](https://arxiv.org/abs/2401.08570): skate guidance in diffusion; AMASS skating 0.078 against 0.230 for HuMoR.
  - [OpenCap Monocular](https://arxiv.org/abs/2603.24733): penalises heel and toe speed during contact and keeps one floor
    height across contact events.
  - [worldhmr-fastcam](https://arxiv.org/abs/2610.19729): contact-weighted foot speed in a final optimisation; K-Pop foot
    skating 17.94 against 31.57 for GVHMR.
- Image-only foot contact is new but weak in the wild. [FECO](https://arxiv.org/abs/2511.22184) (CVPR 2026) labels contact
  on 265 SMPL-X foot vertices and on heel, ball and toe regions.
  - Stated: joint-level F1 of 0.515 on COFE.
  - An [open issue](https://github.com/dqj5182/FECO_RELEASE/issues/3) reports poor results on a dance video with both feet
    visible.
- MAMMA's authors found that a floor-contact term did not improve multi-view fits ([v4](https://arxiv.org/pdf/2506.13040v4)
  s5.4).
  - Their network places foot landmarks at shoe height, so a strong floor term pulls bodies down.
  - They recommend the floor-contact signal for single views, or as a post-process near the floor (s22).
- Our own refine variants agree (measured, `quality/variants.json` and [`../CONTRACTS.md`](../CONTRACTS.md)):
  - A soft slip term on feet slower than 0.15 m/s (recipe `rf2`) cut slip from 93/119 to 77/113 mm/s. Feet reprojection
    stayed the same: 7.45/8.39 against 7.43/8.37 px. Two caveats: the filter-only fit already has 93/113, so for the
    follower `rf2` only removes the slip the no-slip refine adds; and all-points reprojection rose 0.09/0.10 px.
  - Pinning every stance frame hurt: feet +1.5/+1.0 px against the raw fit, and stance ankles moved 29.5/26.4 mm
    sideways. That run also freed the horizontal root and dropped the ankle anchor and the feet term, so pinning alone
    is not proven to be the cause.
- For us: switch a soft, contact-weighted term back on in every route. Pin the contact point (heel or ball), not the ankle, so
  pivots stay free.

**Finding 4. Physics removes floating, sinking and skate for one body, but no released tool handles a partner who carries
weight.**
- Stated:
  - [Physics restoration](https://arxiv.org/abs/2412.17377) ([ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_A_Plug-and-Play_Physical_Motion_Restoration_Approach_for_In-the-Wild_High-Difficulty_Motions_ICCV_2025_paper.html)),
    on AIST++ dance from GVHMR: foot skate 2.232 to 0.587, ground penetration 12.39 to 0.50, float 71.19 to 1.98, MPJPE
    93.5 to 94.0. Its arXiv page states no code release.
  - A [multi-view physics method for combat sports](https://arxiv.org/abs/2504.08175): foot speed error 2.2 to 0.3, foot
    height error 16.4 to 8.1. Its arXiv page states no code release.
- What is released:
  - [PHC/PHC+](https://github.com/ZhengyiLuo/PHC) simulates one body, so a partner's support looks like a fall.
  - [MultiPhys](https://arxiv.org/abs/2404.11987) simulates two bodies, but it has no licence and its custom-input path is
    broken ([issue 6](https://github.com/nicolasugrinovic/multiphys/issues/6)). Its accuracy got worse on Lindy-hop aerials:
    PA-MPJPE 159.1 to 164.3.
- The cheap part to borrow: the restoration paper's first stage projects each mesh into the image and flags frames whose mesh
  falls outside the person's mask. Masks beat 2D keypoints as that cue.
- For us: physics only as a screen for solo windows; mask-check flags for awkward frames.

**Finding 5. No paper measures two against three cameras for a couple. What exists says the angle between cameras matters
more than their number.**
- [MAMMA](https://arxiv.org/pdf/2506.13040v4) (stated, supplementary s12-13, Fig. 12):
  - It sampled 2/4/8/12/16 cameras, so there is no 3-camera point. It reports "strong performance with as few as 4
    cameras" and the best accuracy at around 12.
  - Cross-view identity matching was 100 % correct even with 2 cameras.
- [AHAP](https://arxiv.org/abs/2602.23951) (stated):
  - Going from one view to two: W-MPJPE 1.54 to 0.93 m, PA-MPJPE 0.16 to 0.09 m.
  - Its text says the gains "begin to plateau at 3-4 views", while its figure caption speaks of continued improvement.
  - The code is not released.
- [OpenCap](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462) used two iPhones at plus and
  minus 45 degrees, one person (stated). Joint-angle error was 4.5 degrees, and "using three cameras did not improve
  accuracy".
- Ours (measured, git 46b8552; two seams of one take, not a controlled test): on lucas_luiza one seam was 0.1-0.25 m
  off where the Steadicam and tripod saw the couple only 22-28 degrees apart. At another, 68-76 degrees apart, the
  bodies agreed within 16 mm, though the follow's fingers were still 120 mm off.
- For us: measure it on larissa_kadu_38 phone subsets (E1).

**Finding 6. Camera details turn into apparent foot slip.**
- **Sync.** [RocSync](https://arxiv.org/abs/2511.14948) measured audio sync of two GoPros at 30 fps (stated).
  - Audio sync gave an 11.34 px stereo reprojection error, against 3.07 px with its LED clock and 1.02 px with the clock
    plus sub-frame interpolation. The authors blame audio and video being encoded separately.
  - In a separate test (four GoPros, hand-annotated hand joints), interpolating 2D keypoints to sub-frame times halved
    triangulation error, from 8.80 to 4.40 px.
  - We sync by audio, then refine epipolarly. Whether the fit uses time-interpolated landmarks has not been checked.
- **Lens distortion.** [Kineo](https://arxiv.org/abs/2510.24464) fits Brown-Conrady distortion from people. The sparse
  sweep read its tables as W-MPJPE 410 to 170 mm on EgoHumans with distortion fitted (not re-checked here). Our camera files
  carry one radial coefficient, k1 (SIMPLE_RADIAL, [`../CONTRACTS.md`](../CONTRACTS.md)).
- **One phone (measured this round).** We compared lk03_cut20b (phone 03 of the demo, 40-60 s on the reference clock) with
  the eight-phone SfM.
  - The one-phone camera matches the SfM in focal (780.2 against 781.4 px over the same 40-60 s), pitch (-14.3 against -14.5 degrees) and roll
    (1.4 against 1.3 degrees).
  - Yet it puts the couple 9.5 % too far: 4.93 against 4.50 m, horizontal median, with the camera 1.77 against 1.72 m
    high.
  - Reasoned: the pelvis is seen about 10 degrees below the horizon and the feet about 21. At those angles every centimetre
    of camera-height error moves the couple 2.6-5.7 cm along the floor.
  - So the 0.90-0.91 "best scale" of the floor track is not the focal's +-10 %, which
    [PIPELINE_EFFICIENCY.md](PIPELINE_EFFICIENCY.md) s7e assumed.
  - Files: `work/lk03_cut20b/cameras/01_world.npz`, `work/lk03_cut20b/smplx_rf2_noslip/`, `work/larissa_kadu_38/cameras/03.npz`
    with `world_from_sfm.json`, and `clock_studio/window.json` for the time mapping.
- For us:
  - check sync and time interpolation for two and three phones;
  - get the one-phone camera height and pitch from the dancers' stance feet over the whole clip;
  - raise the phone.

**Finding 7. Masks and identity have new released tools aimed squarely at our failure, SAM dropping a dancer in the
embrace.**
- [SAM 3.1 Object Multiplex](https://github.com/facebookresearch/sam3) (released 2026-03-27) tracks several objects jointly
  with a shared memory. SAM License; the SAM 3 checkpoints are gated.
- [DAM4SAM](https://arxiv.org/abs/2509.13864) adds a distractor-aware memory to SAM 2.1 (Apache-2.0). Its multi-object
  version, [d4sm](https://github.com/alanlukezic/d4sm), gives each dancer its own distractor memory.
- [SAM2MOT](https://github.com/TripleJoy/SAM2MOT) (Apache-2.0, zero-shot, SAM 2.1 plus a detector) states DanceTrack test
  HOTA 75.8 and IDF1 83.9.
- MAMMA upstream `dev/v1.1.0` ([changelog](https://raw.githubusercontent.com/cuevhv/mamma/dev/v1.1.0/CHANGELOG.md)) has
  four changes our fork lacks:
  - it re-acquires people left without a mask;
  - it gives the SAM 3 tracker non-overlapping memory, so one id cannot slide onto a partner in contact;
  - it uses robust IRLS triangulation (opt-in: `--tri-irls`);
  - it adds an outlier gate against teleports (opt-in: `--tri-outlier-gate`).
- For us: one mask test (E5) and a fork update.

**Finding 8. Real partner-dance ground truth now exists. It is for testing, not for training on zouk.**
- [MammaEval-Dance](https://github.com/cuevhv/mamma/blob/main/docs/DATASETS.md): 18 West Coast Swing takes by one couple, 32
  synchronised cameras, Vicon and MoSh++ SMPL-X ground truth.
  - Cameras can be downloaded one by one.
  - [MPI non-commercial licence](https://raw.githubusercontent.com/cuevhv/mamma/main/LICENSE), with a MAMMA account. The
    owner enters the credentials.
- [CoMPAS3D](https://huggingface.co/datasets/Rosie-Lab/compas3d): LA-style salsa, 9 pairs, more than 3 hours, SMPL-X,
  CC BY-NC 4.0 (the audio is excluded).
- [Harmony4D](https://arxiv.org/abs/2410.20294): 208 sequences, more than 20 synchronised cameras, close contact including
  dancing. Fine-tuning HMR2.0 on it improved PVE by 54.8 % under severe occlusion and contact.
- For us: MammaEval-Dance can test two against three views on true ground truth, and can check our eight-phone reference
  itself.

## 3. Ranked recommendations

Rank 1 is what to do first in each table.
- Effort: low = hours with existing code; medium = about a day of code; high = several days.
- "Gain" is an expectation, followed by the experiment (section 4) that confirms it.

### 3a. One phone

| Rank | Method / technique | Fixes | Released? Licence | Inputs | Effort | Expected gain | Risk |
|---|---|---|---|---|---|---|---|
| 1 | Keep the [SAM 3D Body](https://arxiv.org/abs/2602.15989) prior at full strength while a dancer is hidden (mask area turns it down to 25-50 % today): confidence from 2D reliability, and her 2D sigmas left unhalved (MAMMA halves them for the whole short window); prior weight 2 | hidden follower, spin heading | ours (`cand_fitb`, mamma-turing `utils/mono.py`); 3DB is SAM License, gated | phone video, our masks, K | low | hidden-frame follow error from about 170 mm toward 3DB alone (81 mm) (E3) | while she is hidden the crop shows mostly him, but with our masks 3DB had 0 swaps on john_lula |
| 2 | [ExMR](https://github.com/twehrbein/ExMR) as the per-frame prior, A/B against 3DB | jitter, ankle and foot angles; contacts as stance votes | code and [weights](https://huggingface.co/twehrbein/ExMR) (not gated); research / non-profit licence; needs gated 3DB and SMPL-X/MANO | one video, one person per box, focal | low (`remote/cand_hmr/exmr_pod.py`) | smoother and better ankles (stated) (E3 arm B) | box-only crops in the embrace; its gain over 3DB's own published numbers is 0.8-3.6 mm PA |
| 3 | Generic stance foot lock: generalise `jl_footlock`, with stance by vote (RTMW stillness, MAMMA `floor_contact`, GVHMR or ExMR contacts) | foot slip | ours | any fit folder, RTMW | medium | inside detected stances 6.5-7.6 to 0.6-0.9 mm/frame (john_lula, measured) (E2) | covers only 33/20 % of foot frames today; false contacts in pivots |
| 4 | Generalise `jl_facing` (image facing evidence plus the `MAMMA_FACING_HINT` refit) | front/back in spins | ours | fit, images | medium | agreement with the image-facing posterior during spins 0.870 to 0.942 on john_lula (measured; not independent of the hint, held-out eye labels unchanged at 17/19) | front-to-side under-rotation remains in every model |
| 5 | Camera height and pitch from the dancers' stance feet over the whole clip; at capture, raise the phone | placement, floor travel | ours | fit, stances | medium | lk03 floor track 82/96 mm toward the 39/46 mm of the best scale (finding 6) | needs reliable stances |
| 6 | [FlowHMR](https://github.com/flowhmr/flowhmr) per dancer, as prior or stance cue | float, sinking, implausible single-body poses | code Apache-2.0; [weights](https://huggingface.co/fafsaf1/flowhmr-0.46B) non-commercial, not gated; SMPL-H (owner registers) | one video, one person; our boxes and K can be injected | medium | unknown; stated plausibility gains | skates more than GVHMR on Wild-4K; may straighten poses the partner supports |
| 7 | [DuoMo](https://github.com/facebookresearch/DuoMo) on the hidden span | hidden follower | XRCIA non-commercial; weights partly on Google Drive | one video, static camera; our K, boxes and keypoints pre-written | medium (A100) | keep only if hidden-frame error is 10 mm or more below 81 | follows the leader's keypoints; invents a plausible spin |

Also possible: one re-run of [SAM-Body4D](https://arxiv.org/abs/2512.08406) on the 44-48 s window. It added nothing at 58-62 s
(measured).

### 3b. Two phones

| Rank | Method / technique | Fixes | Released? Licence | Inputs | Effort | Expected gain | Risk |
|---|---|---|---|---|---|---|---|
| 1 | Place the phones 60-90 degrees apart, never under 45 | seams, partner depth | capture rule | none | none | one lucas_luiza seam at 22-28 degrees was 0.1-0.25 m off, one at 68-76 degrees 16 mm (measured, 46b8552; two seams) | venue layout |
| 2 | Turn a soft slip term back on in the farm recipe: `rf2` now, the contact-weighted term after E2 | foot slip | ours (`pose_refine --w-slip`) | fit folder | low | 8 phones: 93/119 to 77/113 mm/s at equal feet reprojection (measured) (E2) | the follower's slip fell only from 119 to 113 mm/s |
| 3 | Pose prior in the two-view fit: SAM 3D Body per phone and dancer, rotated to the world, taken from the better-seen view each frame, fed through `MAMMA_POSE_PRIOR` | awkward limbs where one phone is blind; a dancer hidden in one view | ours plus 3DB | two views, masks, cameras | low-medium | unknown (E4) | the two views' priors disagree; a crop of the wrong dancer |
| 4 | Distractor-aware masks: [d4sm](https://github.com/alanlukezic/d4sm) or [SAM 3.1](https://github.com/facebookresearch/sam3) | dancer dropped in the embrace (lucas_luiza seams 0.4-0.6 m) | Apache-2.0 / SAM License (gated) | video, first-frame masks | low | E5 | runs forward only and cannot be re-anchored mid-video |
| 5 | Update the fork to MAMMA `dev/v1.1.0`, and try the occlusion-aware gating and VPoser our fork already has (off; VPoser weights must first be set up on the pods) | teleports, lost masks in contact, odd hidden limbs | [MPI non-commercial](https://raw.githubusercontent.com/cuevhv/mamma/main/LICENSE) | as today | low-medium | fewer seam failures (E1 arm E) | VPoser may pull dips toward standing; merge work |
| 6 | Check sub-frame sync, and that the fit reads time-interpolated 2D | apparent slip in fast footwork | ours | none | low | interpolation halved triangulation error in [RocSync](https://arxiv.org/abs/2511.14948) (stated) | none |
| 7 | Prefer fixed cameras (tripod or gimbal in place) to walking ones | Steadicam registration: only 52 % pose-verified on lucas_luiza (measured) | capture rule | none | none | fewer unregistered stretches | venue |

### 3c. Three phones

| Rank | Method / technique | Fixes | Released? Licence | Inputs | Effort | Expected gain | Risk |
|---|---|---|---|---|---|---|---|
| 1 | Measure two against three phones on larissa_kadu_38 subsets before filming (E1) | decides the capture | ours (`mamma prepare --set` plus a small driver) | existing 8-phone inputs | low | a number instead of a guess | best case: eight-phone cameras and sync |
| 2 | Place the three about 120 degrees apart (no pair under 60), one raised 2-2.5 m and tilted 15-30 degrees down; fixed zoom; whole couple in frame | seams, hidden follower, placement, audience | capture rule | none | none | reasoned in section 5 | elevation untested on zouk |
| 3 | Use the third phone as a held-out check: fit all three for the product, leave one out for acceptance | trust | ours (`mamma_loo`) | none | low | a held-out reprojection per take | three extra fits |
| 4 | Calibrate from the dancers when the phones share little background ([HSfM](https://github.com/hongsukchoi/HSfM_RELEASE)-style, as our pair calibration does), and fit lens distortion beyond k1 | cameras | HSfM MIT | none | medium | unknown | none known |
| 5 | Validate on [MammaEval-Dance](https://github.com/cuevhv/mamma/blob/main/docs/DATASETS.md) camera subsets with true ground truth | trust in our reference and in E1 | MPI non-commercial; the owner's MAMMA account | per-camera downloads | medium | ground-truth numbers | West Coast Swing is not zouk; high, fixed studio cameras |

### 3d. Post-processing that works on any route

| Rank | Method / technique | Fixes | Released? Licence | Inputs | Effort | Expected gain | Risk |
|---|---|---|---|---|---|---|---|
| 1 | Contact-weighted, pivot-aware foot term in `pose_refine`. The contact point is the sole vertices within 1 cm of the lowest (as `pose_quality` defines it), weighted by a stance vote; foot yaw is left free | foot slip | ours | any fit folder | low-medium | E2 | false contacts freeze pivots |
| 2 | Mask-check flags: the share of each projected mesh (per limb) inside its own dancer's mask, per view | awkward poses found automatically | idea from the [restoration paper](https://arxiv.org/abs/2412.17377); ours | fit, masks, cameras | low | a list of frames to fix or review | masks are wrong in embraces |
| 3 | Better foot evidence as extra votes, each after an agreement test: [Sapiens2](https://github.com/facebookresearch/sapiens2) foot keypoints and [FECO](https://github.com/dqj5182/FECO_RELEASE) contact regions | stance detection; heel or ball for pivots | Sapiens2 licence ([pose-1b](https://huggingface.co/facebook/sapiens2-pose-1b) not gated); FECO CC BY-NC, not gated | crops from our masks | low | adopt only past the gates in section 4 | FECO's in-the-wild F1 is 0.515, plus the dance issue |
| 4 | [RoHM](https://github.com/sanweiliti/RoHM) per dancer on flagged windows: contact-guided denoising and infill | slip, jitter, short occlusions | CC BY-NC 4.0 | SMPL-X sequence plus a per-joint visibility mask | medium | stated AMASS skating 0.078 against 0.230 | knows nothing of the partner; biased toward AMASS; two users report wrong output on custom input ([27](https://github.com/sanweiliti/RoHM/issues/27), [29](https://github.com/sanweiliti/RoHM/issues/29)) |
| 5 | Fill hidden spans and label them "inferred" in tutorials: Kimodo with head, hand and heading constraints; DuoMo or RoHM as alternatives | hidden follower | Kimodo SMPL-X weights R&D only | fit around the gap | medium | Kimodo: 43 mm against 69-72 for interpolation on one held-out second (measured) | invented motion |
| 6 | Pair metrics on the acceptance sheet: couple vector error, Pair-PA-MPJPE ([PromptHMR](https://arxiv.org/abs/2504.06397)), and penetration measured against a band rather than zero (our reference itself shows about 31 mm) | partner relations | technique | fits | low | catches the wrong relations that per-person PA hides | none |
| 7 | Physics as a screen only: [PHC](https://github.com/ZhengyiLuo/PHC) SMPL-X humanoid on solo windows; our `physics.py` support analysis for couples | implausible single-body poses | BSD-3-Clause Clear | fit | medium | flags, no edits | partner-supported moves look like falls |

## 4. Test plan

The rules for every experiment:
- one change at a time;
- the eight-phone larissa_kadu_38 fit (`smplx_rf2_noslip`) as the reference unless stated, with views held out where possible;
- the result is a sheet plus about three numbers, then a 20-30 s preview of the changed spots once it passes;
- MAMMA fits run one per GPU.

Costs assume an RTX 4090 at about $0.9/h and an A100 80 GB at about $1.4-1.8/h, the rates seen in recent runs.

**Step 0 (free, before E3).** Copy the ExMR `hmr_test` result (`work/lucas_luiza/hmr_test/score/{sheet.md,score.json}`) from
the owner's Ubuntu laptop. If ExMR failed the owner's bar on clips E1 (closed hold) and H1 (embrace), E3 drops its ExMR arm.

| # | Question | Data | Main metric | Pass bar | Pod | Minutes | Dollars |
|---|---|---|---|---|---|---|---|
| E1 | Do three phones beat two, and where should the third go? | larissa_kadu_38, whole 38.4 s, phone subsets; phone 01 held out | follow PA22 at 44-48 s; pelvis and couple vector; held-out reprojection | three-phone arm 30 % or more better than the good pair | 1x 4090 | 70-90 | 1.0-1.5 |
| E2 | Does a contact-weighted foot term cut slip in every route? | the eight-phone fit, E1's good pair, the mono06 fits | stance contact-point slip (mm/s) | beats `rf2`; -30 % on the two-view and one-phone fits; feet +0.3 px at most | 1x 4090 or A100 | 30-40 | 0.3-0.6 |
| E3 | Does keeping the prior on fix the hidden follower? | larissa_kadu_38_mono06, 44-48 s (58-62 s as control) | follow PA22 on the 43 hidden frames | 90 mm or less (today about 170); control within 2 mm | 1x 4090 | 40-60 | 0.6-1.0 |
| E4 | Does a pose prior fix awkward limbs with two phones? | E1's good pair {06, 07}, 40-64 s | PA22, limb-angle error | follow PA22 -15 % overall or -30 % at 44-48 s; held-out reprojection no worse | 1x 4090 | 60-75 | 0.9-1.2 |
| E5 | Do distractor-aware masks stop drops and swaps in the embrace? | phone 06 at 40-50 and 56-64 s; lucas_luiza's Steadicam (camera 01) at 112.7-117.7 and 160.7-165.1 s | swaps plus drops | -50 %, with no IoU loss on visible frames | 1x 4090 | 20-30 | 0.2-0.4 |

The total is about $3-5. E1 and E2 can run side by side, because E2's first two inputs already exist.

**E1. Two against three phones on our own eight-phone take.**
- The phones (computed this round from `cameras/*.npz`, `world_from_sfm.json` and the reference pelvis, medians over
  30-68 s):

  | Phone | Azimuth around the couple (deg) | Height (m) | Distance (m) | Frame |
  |---|---|---|---|---|
  | 01 | 0 | 1.66 | 4.4 | 480x864 |
  | 02 | 27 | 0.73 | 3.2 | 1080x1920 |
  | 09 | 51 | 1.37 | 3.8 | 480x848 |
  | 03 | 64 | 1.71 | 4.4 | 480x848 |
  | 04 | 72 | 1.57 | 5.0 | 480x864 |
  | 05 | 165 | 0.99 | 3.8 | 480x640 |
  | 06 | 235 | 0.79 | 3.9 | 1080x1920 |
  | 07 | 312 | 1.54 | 6.8 | 480x848 |

  All eight phones are 0.7-1.7 m high, from just below the dancers' pelvis height to about head height. None looks down
  on the couple by more than about 10 degrees.
- Arms. Each is a MAMMA fit on the reference's own inputs (landmarks, masks, cameras, sync, pinned betas), refined with the
  same recipe:
  - A, narrow pair: {02, 09}, 24 degrees apart, like lucas_luiza.
  - B, good pair: {06, 07}, 78 degrees apart (70-84 in the two scoring windows).
  - C, third phone across: {04, 06, 07}. Phone 04 bisects the large gap: pairs at 78, 120 and 162 degrees.
  - D, third phone beside: {05, 06, 07}, with pairs at 70, 78 and 148 degrees.
  - E: arm B with `--occlusion-aware-weights --use-vposer`. Both options are already in mamma-turing, and are off today.
    VPoser needs its weights on the pod first: the bootstrap leaves the package out, and the run fails its preflight.
- Held out: phone 01 is in no arm. Report its reprojection, all points and feet, for every arm.
- Code: keep-only camera sets already exist (`mamma prepare --set tag=02,09`), and `mamma fit` reports held-out
  residuals. The registered `mamma_loo` stage cannot run E1 as it is. Its script (`work/larissa_kadu/mamma/loo/pod_loo.py`,
  git-ignored) is a one-off for the 5.8-s larissa_kadu take: it fits eight-minus-one sets and re-promotes `smplx/`. E1
  needs a small driver that fits the listed sets and promotes nothing.
- Metrics against the reference, per role, over the whole take plus 44-48 s and 58-62 s:
  - PA22;
  - pelvis error with no alignment;
  - couple vector error (follow pelvis minus lead pelvis);
  - stance contact-point slip on the reference's stance frames (`pose_quality`);
  - held-out reprojection in phone 01.
- Decision:
  - If C or D is 30 % or more better than B on the follow's 44-48 s PA22, or on the couple vector, film three phones. The
    better of C and D says where the third goes.
  - If neither is 10 % better, two well-separated phones are enough for takes like this one, and the effort goes into
    priors (E3, E4).
  - A against B checks the separation rule: A is expected to be at least twice as bad.
- Limits:
  - This is the best case. Real two- and three-phone captures add their own calibration and sync errors.
  - The arms mix 1080p and 480p phones, one 1080p phone per arm.
  - Elevation cannot be tested, because no phone here is raised.

**E2. One foot term for all routes.**
- Arms in `pose_refine`:
  - 0: today's `smplx_rf2_noslip`.
  - A: the existing slip term below 0.15 m/s (`rf2`).
  - B, new: a horizontal-speed penalty on the contact point, the sole vertices within 1 cm of the foot's lowest point
    (`pose_quality`'s definition).
    - It is weighted by a stance vote that needs 2 of 3 cues: MAMMA `floor_contact` on those vertices of 0.5 or more; foot
      keypoints still in at least one view (`jl_footlock`'s 2.5 px rule); sole lower than 3 cm.
    - It has hysteresis, no term on foot rotation (pivots stay free) and the existing 10 mm ankle anchor.
- Inputs:
  - the eight-phone `smplx/` fit;
  - E1 arm B, or the two-view fit of any take if E1 has not run;
  - larissa_kadu_38_mono06 at prior weight 2, both windows.
- Metrics, from `pose_quality`:
  - stance contact-point slip on the reference's stance frames (today 93/119 mm/s with eight phones);
  - feet reprojection, in the fitted views and in held-out phone 01;
  - sideways movement of the stance ankles against the input fit.
- Pass:
  - B beats A on slip for all three inputs, with feet reprojection up by 0.3 px at most and stance ankles moved 10 mm at
    most;
  - on the two-view and one-phone fits, slip falls by 30 % or more against arm 0;
  - then preview the 44-48 s spin for the owner.
- The October batch ran six refine variants on one A100 for $0.32.

**E3. One phone: the hidden follower.**
- Arms on larissa_kadu_38_mono06:
  - 0: prior weight 2 as on 85c9f78, which gives 32/78 at 44-48 s.
  - A: weight 2, with the prior's confidence floored at 0.5 while she is hidden (today's floor is 0.25, at half her
    median mask area). The confidence comes from 2D reliability instead of visible mask area, and her 2D sigmas are
    not halved. Today's turned-down prior already gets her heading right on the hidden frames, but her PA22 there is
    169 mm, against 172 without the prior.
  - B: arm A with ExMR as the prior. This arm runs only if Step 0 passed and ExMR's per-frame PA22 ties the smoothed 3DB
    control (33/40 at 58-62 s, 32/55 at 44-48 s).
- Metrics:
  - follow PA22 on the 43 hidden frames (today about 170 mm; 3DB alone 81);
  - whole-window PA22;
  - couple depth order;
  - jitter.
- Pass:
  - hidden-frame follow at 90 mm or less;
  - the 58-62 s control within 2 mm of 31/36;
  - jitter at most 1.2 times today's.

**E4. Two phones: a pose prior for the two-view fit.**
- Build the prior:
  - Run SAM 3D Body, with mask and K prompts as `cand_sam3d_body` does, on phones 06 and 07, both dancers, 40-64 s. That is
    about 2,900 person-frames, about 22 min at 0.46 s each.
  - Rotate each view's root-relative joints into the world with that phone's R.
  - Each frame, take the view where the dancer's visible-mask fraction is larger, or the average when both are above 0.8.
  - Write `prior_3db.npz` as (T, 2, 22, 3) with confidence.
- Add `pose_prior_loss` to the multi-view config stages. Today it sits only in the one-phone configs: the multi-view
  `ma3d_config.yaml` has a shape prior and temporal smoothness terms, but no pose prior. Run weights 0.5 and 2, and compare with E1 arm B.
- Metrics: PA22 (whole take and 44-48 s), limb-angle error, held-out reprojection in phone 01.
- Pass: follow PA22 -15 % overall or -30 % at 44-48 s, lead no worse, held-out reprojection no worse.

**E5. Masks in the embrace.**
- Arms:
  - [d4sm](https://github.com/alanlukezic/d4sm) (`DAM4SAMMOT` large);
  - stock SAM 2.1-L, forward from the same start;
  - our production masks, which are re-anchored.
- Every arm starts from our masks on a clean frame before each window.
- Data: phone 06 at 40-50 s and 56-64 s; lucas_luiza's Steadicam (camera 01) at 112.7-117.7 s and 160.7-165.1 s,
  where SAM dropped a dancer. The lucas_luiza work files are on the owner's Ubuntu laptop.
- Reference: each dancer's visible silhouette rendered from the eight-phone fit into phone 06 (z-buffered). For lucas_luiza,
  the two-view fit rendered into camera 01.
- Metrics per frame:
  - IoU with the right dancer;
  - swaps: higher IoU with the wrong dancer;
  - drops: an empty mask while 20 % or more of the dancer is visible;
  - frames needed to recover.
- Pass: swaps plus drops down by 50 % or more, with no IoU loss on visible frames.

**Next, only if needed.** Each has a written plan from the verifiers in this round.
- FECO agreement check on all eight phones at 58-62 and 44-48 s (about $1). Gate on phone 06: precision 0.85 or more at
  recall 0.6 or more, and F1 0.10 above both MAMMA `floor_contact` and GVHMR.
- Sapiens2 foot keypoints against RTMW and MAMMA landmarks (about $0.3). Gate: foot error -20 %, or stance coverage +5
  points.
- FlowHMR A/B (about $0.5, once the owner has put the SMPL-H neutral model on the volume).
- DuoMo on the hidden span (about $1.5 on an A100).
- RoHM on flagged windows (about $2).
- Camera-height oracle on lk03: re-fit with the eight-phone camera height and pitch (about $1).
- MammaEval-Dance at two and three views against true ground truth (about $2-3, plus a download with the owner's account).

## 5. What a third phone buys over two, and where to put it

**What the evidence says.**
- From the literature:
  - The large step is from one view to two ([AHAP](https://arxiv.org/abs/2602.23951): W-MPJPE 1.54 to 0.93 m).
  - Beyond that, single-person results flatten. AHAP's text says gains "begin to plateau at 3-4 views".
    [OpenCap](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462)'s third camera did not
    improve accuracy for one person in a lab.
  - [MAMMA](https://arxiv.org/pdf/2506.13040v4) is strong from 4 cameras and best near 12. It matched identities across
    views 100 % of the time with 2, so identity is not a reason for a third phone.
  - Nobody reports two against three cameras for a couple in contact.
- From our data (measured):
  - On lucas_luiza a seam where the two cameras were 22-28 degrees apart was 0.1-0.25 m off; one at 68-76 degrees
    agreed within 16 mm (two seams, not a controlled test).
  - In the 44-48 s window, phone 06 sees less than 0.6 of the follower's whole-take median mask area on about 65 % of
    frames. With
    two phones, every frame where the second phone also loses her becomes a one-phone frame, the class of error that is
    about 170 mm today.
  - The eight phones converge: a 7-phone fit with phone 03 held out agrees with the 8-phone fit to a median PA22 of
    4/8 mm at 58-62 s (one window; 4-10 mm across metrics). That is a consistency floor, not ground truth.
- Reasoned. A third phone mostly buys coverage: at every moment, at least one well-separated pair that sees each dancer.
  It also buys a held-out view to check the fit. It adds little where two good views already see everything. How often
  both of two phones lose a dancer at the same time decides the gain, and E1 measures that.

**Where to put it (reasoned, with what it rests on).**
- **Around the couple, about 120 degrees from the other two, no pair under 60 degrees.** If two phones are already placed,
  for example a front tripod and a side gimbal, put the third across the larger gap.
  - Triangulation depends on the angle between the rays.
  - A pair that is nearly opposite (about 180 degrees) cannot resolve depth along its shared axis, just like a pair at 0
    degrees. Opposite phones do see front and back, which helps with occlusion and with spins.
  - Even spacing keeps every pair useful.
- **Raised.** One phone, the third is easiest, at 2-2.5 m or higher, tilted 15-30 degrees down. Three reasons:
  - It sees over the leader's shoulder when she is behind him, and over audience members walking in front.
  - It sees the feet against the floor more steeply, which helps contact.
  - Any stretch the fit has to solve from one phone becomes less sensitive to camera-height errors. The couple's floor
    distance moves about 1/tan(angle) times a height error: 5.7x at 10 degrees down, 2.7x at 20, 1.7x at 30 (finding 6).

  This has not been measured on zouk: every larissa_kadu_38 phone is between 0.7 and 1.7 m high. The verifier read the
  MAMMA paper as placing its evaluation cameras on a 5-m truss, so MammaEval subsets may be able to test elevation (not
  confirmed).
- **Framing.**
  - Keep the whole couple in frame with margin for the whole dance, because feet near the frame edge are the most distorted
    and the most often cut.
  - Fix the zoom. Use landscape for the raised wide phone. Record 1080p.
- **Sync.** Start every phone before the music, and make one clap that every phone both sees and hears. Our epipolar
  refinement then finds the sub-frame offsets. RocSync found audio alone was not frame-accurate on consumer cameras.

**Sourcing a ready-made three-angle video.** Social-media "multi-angle" uploads are often one camera re-cropped or
re-edited. Before fitting one, check parallax at a few audio-synced instants:
- a re-crop of the same camera maps onto the other with one homography, down to compression noise;
- a second camera position leaves the dancers ghosting at every time offset.

This round's best sourcer candidate failed this check (measured, 2026-11-03). Allan & Vicky's BZWC 2026 Originals routine
appeared as three uploads: the BZWC 16:9 wide, the BZWC 9:16 Short, and Makana Media's 9:16 cut on the dancers' Instagram.
- The audio aligns them within 0.08 and 0.64 s.
- At five synced instants, one homography maps each vertical onto the wide with a constant scale (0.75 and 0.84), and the
  dancers, truss, screen and floor all overlay without ghosting.
- So it is one camera re-cropped three ways. The TypeSafe visual check had called them three positions, so the sourcer now
  also runs this geometric test.

## 6. Dropped candidates and why

Verified this round:
- [worldhmr-fastcam](https://arxiv.org/abs/2610.19729) ([code](https://github.com/chien90190/worldhmr-fastcam)): it fixes
  scale drift between SLAM and HMR under fast broadcast cameras, which our metric cameras do not have. Its bodies are
  PromptHMR-Vid or GVHMR, which both lost on john_lula. Its 0.2 m repulsion and its "planted" default for undetected frames
  fight embraces and hidden spins.
- [PromptHMR / PromptHMR-Vid](https://github.com/yufu-wang/PromptHMR): already measured on john_lula, spins window. Lula's
  facing agreement was 65-77 % against the fit's 79 %. The video heads put both dancers 57-87 cm too near the phone, a
  monocular scale ambiguity the couple refinement fixes, and the contact cue was weaker than GVHMR's. With pose from
  our keypoint pass the same heads reached 81-83 %, so they lost as a whole route, not in every part. Nothing new has been released since 3b566b7.
- [MultiPhys](https://github.com/nicolasugrinovic/multiphys): no licence, a broken custom-input path, SMPL-H. It pushes
  partners apart, and its accuracy and smoothness got worse on ExPI Lindy hop.
- [PHC/PHC+](https://github.com/ZhengyiLuo/PHC) as a fitter: one simulated body, so partner support reads as a fall. It
  moves joints about 35 mm on AIST++ ([Half-Physics](https://arxiv.org/abs/2507.23778), preprint). It is kept only as a solo
  screen.
- [Duolando + DD100](https://github.com/lisiyao21/Duolando): never looks at the video, has no zouk, and its follower touches
  the leader on 52 % of frames against 74 % for real dancers. CUDA 10.1 stack.
- [BUDDI](https://github.com/muelea/buddi): per image, trained on hugs and handshakes, CUDA 10.2-era stack. CloseApp, which
  is stronger, only tied on our window.
- [VolumetricSMPL](https://github.com/markomih/VolumetricSMPL): would duplicate MAMMA's existing person-to-person SDF term
  (weight 1.0 in the final, limbs-only stage of our contact config), and the owner's rule is against more collision pressure.
- [DPoser-X](https://github.com/moonbow721/DPoser-X): parked as a diagnostic only. Its gains over older priors are small, and
  its AMASS bias points toward upright poses, against zouk dips.

Read by the sweeps, not re-checked here:
- [CloseInt](https://arxiv.org/abs/2404.11291) and [InterMesh](https://arxiv.org/abs/2605.04554): one-phone close-interaction
  methods trained on Hi4D/CHI3D-style contact. Their sibling CloseApp tied on our data.
- [Human3R](https://arxiv.org/abs/2510.06219) and other feed-forward 4D human-scene models: initialisers at most. WildHSR's
  tables ([arXiv](https://arxiv.org/abs/2609.29106)) report Human3R at 22.6 cm/s foot sliding and 1682 identity switches on
  3DPW two-person sequences.
- [HSfM](https://arxiv.org/abs/2412.17806), [AHAP](https://arxiv.org/abs/2602.23951),
  [HAMSt3R](https://arxiv.org/abs/2508.16433) and [HARMONI](https://arxiv.org/abs/2603.12789) as body estimators: they
  report EgoHumans PA-MPJPE of 50-140 mm. That is a different benchmark, so only a rough comparison, but it is far from our
  31-36 mm one-phone numbers. They are camera tools, and only HSfM is released.
- [UnderPressure](https://arxiv.org/abs/2208.04598): a contact model trained on locomotion, whose licence forbids
  modification. [ContactVision](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.70334) is trained on gait.
- [PhysHMR](https://arxiv.org/abs/2510.02566): its own table shows it skating more than GVHMR on AIST++ (9.14 against
  4.64).
- [GEM / GENMO](https://arxiv.org/abs/2505.01425) and [GEM-X](https://huggingface.co/nvidia/GEM-X): one person each, and
  nothing new for foot slip.
- [CoMotion](https://github.com/apple/ml-comotion): no contact output.
- [Ponimator](https://github.com/stevenlsw/ponimator): generates a partner pose (SMPL-X, ICCV 2025), but its repo has no
  licence file and the training data is not dance.
- Commercial video-mocap services publish no accuracy on dance or two-person contact. Rokoko Vision 3.0 is one camera and one
  performer; Meshcapade's platforms are shut down.

Already rejected on our own data in earlier rounds ([MONO4D_NOVEL_VIEWS.md](MONO4D_NOVEL_VIEWS.md) s10):
- Multi-HMR 2: 63/61 mm, and it swaps the dancers on 28 % of frames.
- Viggle: 109/84 mm.
- NoPo4D: duplicated dancers.
- Atlas novel views for pose: PCK 0.19.
- DiffLocks: hair.
- CloseApp as a refiner: a tie.

Watch list, no usable code yet:
- [SocialMirror](https://arxiv.org/abs/2604.13581), [GNOCHI](https://arxiv.org/abs/2607.10408),
  [DanceHMR](https://arxiv.org/abs/2605.18102), [UniCon3R](https://arxiv.org/abs/2604.19923),
  [WildHSR](https://arxiv.org/abs/2609.29106), [combat-sports physics](https://arxiv.org/abs/2504.08175), the
  [physics restoration](https://arxiv.org/abs/2412.17377) paper and [InterDance](https://arxiv.org/abs/2412.16982) (full
  release).
- [Kineo](https://arxiv.org/abs/2510.24464): the code page stated in the paper did not load on 2026-11-03.
- [AvatarPose](https://arxiv.org/abs/2408.02110): code without a licence, evaluated with 8 views on Hi4D and 4 on CHI3D, not
  2-3.
- [Humans as a Calibration Pattern](https://arxiv.org/abs/2412.19089): sweep reading.

## 7. Sources

**This repo (read 2026-11-03).**
- [`../CONTRACTS.md`](../CONTRACTS.md): camera model, world frame, and the refine recipes (`smplx_refined`, `smplx_rf2*`).
- [MONO4D_NOVEL_VIEWS.md](MONO4D_NOVEL_VIEWS.md) s10.6: one-phone candidates and the prior.
- [PIPELINE_EFFICIENCY.md](PIPELINE_EFFICIENCY.md) s7e: the lk03 mono route and its floor track.
- [FARM.md](FARM.md): the two-view route and recipes. [RECAP_FILM.md](RECAP_FILM.md): hands and contacts.
- Code:
  - [`../dancecap/pose_refine.py`](../dancecap/pose_refine.py), [`../dancecap/pose_quality.py`](../dancecap/pose_quality.py)
    (the slip metric);
  - [`../dancecap/mono4d/cand_fitb.py`](../dancecap/mono4d/cand_fitb.py) (`MAMMA_POSE_PRIOR`),
    [`../dancecap/hmr_test.py`](../dancecap/hmr_test.py);
  - [`../dancecap/pipeline/registry.py`](../dancecap/pipeline/registry.py) (`mamma_loo`).
- mamma-turing: `optimization/utils/mono.py` and `optimization/utils/optimization.py` (the pose prior),
  `optimization/run_ma_3d.py` (`--occlusion-aware-weights`, `--use-vposer`).
- Work files, git-ignored:
  - `work/larissa_kadu_38/quality/variants.json`;
  - `work/larissa_kadu_38/cameras/*.npz` with `world_from_sfm.json`;
  - `work/larissa_kadu_38/smplx_rf2_noslip/`;
  - `work/larissa_kadu_38/clock_studio/window.json`;
  - `work/lk03_cut20b/cameras/01_world.npz`, `work/lk03_cut20b/smplx_rf2_noslip/`.
- Branches: `claude/vigilant-moser-a1afa5` (85c9f78, the 44-48 s window), `claude/nifty-maxwell-2d0320` (24ab40a,
  PromptHMR and GVHMR on john_lula), `claude/footlock-gvhmr-union` (GVHMR stance cue). Commit 46b8552 holds the two-view seam
  angles.

**MAMMA and its data.**
- [arXiv 2506.13040](https://arxiv.org/abs/2506.13040), [v4 PDF](https://arxiv.org/pdf/2506.13040v4): s5.4, s22, supplementary
  s12-13.
- [Code](https://github.com/cuevhv/mamma), [licence](https://raw.githubusercontent.com/cuevhv/mamma/main/LICENSE),
  [datasets](https://github.com/cuevhv/mamma/blob/main/docs/DATASETS.md),
  [`dev/v1.1.0` changelog](https://raw.githubusercontent.com/cuevhv/mamma/dev/v1.1.0/CHANGELOG.md),
  [project page](https://mamma.is.tue.mpg.de/).

**Per-person estimators and priors.**
- SAM 3D Body: [arXiv](https://arxiv.org/abs/2602.15989), [code](https://github.com/facebookresearch/sam-3d-body),
  [weights (gated)](https://huggingface.co/facebook/sam-3d-body-dinov3).
- SAM-Body4D: [arXiv](https://arxiv.org/abs/2512.08406), [code](https://github.com/gaomingqi/sam-body4d).
- ExMR: [arXiv](https://arxiv.org/abs/2610.26213), [code](https://github.com/twehrbein/ExMR),
  [licence](https://raw.githubusercontent.com/twehrbein/ExMR/main/LICENSE), [weights](https://huggingface.co/twehrbein/ExMR).
- FlowHMR: [arXiv](https://arxiv.org/abs/2610.03691), [code](https://github.com/flowhmr/flowhmr),
  [weights](https://huggingface.co/fafsaf1/flowhmr-0.46B), [issue 3](https://github.com/flowhmr/flowhmr/issues/3).
- DuoMo: [arXiv](https://arxiv.org/abs/2603.03265), [code](https://github.com/facebookresearch/DuoMo).
- DPoser-X: [arXiv](https://arxiv.org/abs/2508.00599), [code](https://github.com/moonbow721/DPoser-X),
  [weights](https://huggingface.co/Moon-bow/DPoser-X).
- PromptHMR: [arXiv](https://arxiv.org/abs/2504.06397), [code](https://github.com/yufu-wang/PromptHMR).
- Sapiens2: [arXiv](https://arxiv.org/abs/2604.21681), [code](https://github.com/facebookresearch/sapiens2),
  [pose-1b](https://huggingface.co/facebook/sapiens2-pose-1b).

**Foot contact, motion priors, physics.**
- FECO: [arXiv](https://arxiv.org/abs/2511.22184), [code](https://github.com/dqj5182/FECO_RELEASE),
  [issue 3](https://github.com/dqj5182/FECO_RELEASE/issues/3).
- RoHM: [arXiv](https://arxiv.org/abs/2401.08570), [code](https://github.com/sanweiliti/RoHM).
- [JOSH](https://arxiv.org/abs/2501.02158).
- [OpenCap Monocular](https://arxiv.org/abs/2603.24733) ([code](https://github.com/utahmobl/opencap-monocular)).
- worldhmr-fastcam: [arXiv](https://arxiv.org/abs/2610.19729), [code](https://github.com/chien90190/worldhmr-fastcam).
- Physics restoration: [arXiv](https://arxiv.org/abs/2412.17377),
  [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_A_Plug-and-Play_Physical_Motion_Restoration_Approach_for_In-the-Wild_High-Difficulty_Motions_ICCV_2025_paper.html).
- [Combat-sports physics](https://arxiv.org/abs/2504.08175).
- [PHC](https://github.com/ZhengyiLuo/PHC), [ProtoMotions](https://github.com/NVlabs/ProtoMotions),
  [Half-Physics](https://arxiv.org/abs/2507.23778).
- MultiPhys: [arXiv](https://arxiv.org/abs/2404.11987), [code](https://github.com/nicolasugrinovic/multiphys).
- [PhysHMR](https://arxiv.org/abs/2510.02566), [UnderPressure](https://arxiv.org/abs/2208.04598),
  [ContactVision](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.70334), [GENMO/GEM](https://arxiv.org/abs/2505.01425),
  [GEM-X](https://huggingface.co/nvidia/GEM-X).

**Close contact and partner dance.**
- BUDDI: [arXiv](https://arxiv.org/abs/2306.09337), [code](https://github.com/muelea/buddi).
- VolumetricSMPL: [arXiv](https://arxiv.org/abs/2506.23236), [code](https://github.com/markomih/VolumetricSMPL).
- [CloseApp](https://arxiv.org/abs/2507.02565), [CloseInt](https://arxiv.org/abs/2404.11291),
  [InterMesh](https://arxiv.org/abs/2605.04554), [SocialMirror](https://arxiv.org/abs/2604.13581),
  [GNOCHI](https://arxiv.org/abs/2607.10408), [DanceHMR](https://arxiv.org/abs/2605.18102).
- Duolando: [arXiv](https://arxiv.org/abs/2403.18811), [code](https://github.com/lisiyao21/Duolando).
- [DuetGen](https://arxiv.org/abs/2506.18680), [InterDance](https://arxiv.org/abs/2412.16982),
  [Ponimator](https://github.com/stevenlsw/ponimator).
- CoMPAS3D: [arXiv](https://arxiv.org/abs/2507.19684), [data](https://huggingface.co/datasets/Rosie-Lab/compas3d).
- Harmony4D: [arXiv](https://arxiv.org/abs/2410.20294), [project](https://jyuntins.github.io/harmony4d/).
- [AvatarPose](https://arxiv.org/abs/2408.02110).

**Sparse cameras, sync, calibration.**
- [AHAP](https://arxiv.org/abs/2602.23951).
- [OpenCap](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462).
- RocSync: [arXiv](https://arxiv.org/abs/2511.14948), [code](https://github.com/jaromeyer/RocSync).
- [Kineo](https://arxiv.org/abs/2510.24464).
- HSfM: [arXiv](https://arxiv.org/abs/2412.17806), [code](https://github.com/hongsukchoi/HSfM_RELEASE).
- [HAMSt3R](https://arxiv.org/abs/2508.16433), [HARMONI](https://arxiv.org/abs/2603.12789),
  [Human3R](https://arxiv.org/abs/2510.06219), [WildHSR](https://arxiv.org/abs/2609.29106),
  [UniCon3R](https://arxiv.org/abs/2604.19923), [Humans as a Calibration Pattern](https://arxiv.org/abs/2412.19089).

**Masks and identity.**
- [SAM 3 / SAM 3.1](https://github.com/facebookresearch/sam3).
- DAM4SAM: [arXiv](https://arxiv.org/abs/2509.13864), [code](https://github.com/jovanavidenovic/DAM4SAM).
- [d4sm](https://github.com/alanlukezic/d4sm), [SAM2MOT](https://github.com/TripleJoy/SAM2MOT),
  [CoMotion](https://github.com/apple/ml-comotion).
