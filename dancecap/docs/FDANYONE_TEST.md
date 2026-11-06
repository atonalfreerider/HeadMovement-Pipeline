# 4DAnyone on a couple: monocular and two-view test (lucas_luiza, 2026-11-05)

Owner, 2026-11-05: "https://4danyone.github.io/ did we test this approach for monocular or two camera videos? if not -
run a test on runpod for monocular and two view dances". It had not been tested. Code: `dancecap/fda_test.py`,
`remote/cand_fda/`; results (likeness-derived, git-ignored): `work/lucas_luiza/fda_test/score/` and
`work/lucas_luiza/hmr_test/fda_score/`.

## Verdict

**Not usable for our couples, from one camera or two.**

- **The partner is erased or merged.** 4DAnyone is a one-person method: GVHMR recovers one skeleton, and the video model
  generates that one person on a ring of cameras. The partner is not modelled. The contact sheets show what happens:
  - In the lead's runs the follow is erased, except her hair, which is pasted onto his shoulder.
  - In the follow's runs (Steadicam) she becomes a chimera, with his face and jeans on her body in several views.

  Only 2-29 % of the generated frames hold two people.
- **Pose is no gain.** Its only 3D is the GVHMR skeleton it generates from. On the same frames it does not pass the
  2026-10-27 bar against B1 / B2, and ExMR beats it on PA-MPJPE for both dancers.
- **Two cameras do not make it two-view.** 4DAnyone takes one video. Running each camera through it and averaging the
  two skeletons cuts PA-MPJPE about 10 % (49-55 mm to 46 mm). That is still worse than ExMR from the tripod alone
  (36 / 48 mm), and far from our two-view MAMMA fit, which is the reference.
- **It is slow.** About 16.5 min per 4-second clip, per dancer, on an L40S. Its README's 27 s figure covers the
  denoising only; here the per-view skeleton rendering (numpy, one core) dominated until we parallelised it.

## What ran

- **Clips:** lucas_luiza, the first 121 frames (4DAnyone's clip length) of hmr_test's clips E1 (closed hold) and E2
  (open spins).
- **Cameras:** tripod 02 (9:16 crop around the couple, scaled to 1080x1920) and Steadicam 01 (full frame, x1.5).
- **Dancers:** one run per dancer, with GVHMR's tracker fed OUR dancer boxes (a 4-line patch).
- **Settings:** 12 views on 4DAnyone's default 15-degree ring, Turbo, seed 42.
- **Coverage:** 8 of 16 planned runs finished inside the time budget: E1 for both dancers and both cameras, E2 for both
  dancers on the tripod and the lead on the Steadicam. H1 (embrace) and H3 (fast spins) were not reached. The
  embrace would only be harder for a one-person model.
- **Cost:** $3.40 in all; $2.30 for the run that worked, the rest on four failed starts (tool path, input timeout,
  tkinter, skeleton render time).

## Numbers

**Monocular (tripod in, Steadicam held out)**, hmr_test scorer (self-check passed), on the 139 / 141 good-reference
frames where every method scored:

| dancer | B0 two-view M1-B | B1 mono fit | B2 SAM 3D Body | ExMR | 4DAnyone (its GVHMR) |
|---|---|---|---|---|---|
| lead: M1-B % / PA mm | 2.12 / - | 4.66 / 80 | 3.51 / 48 | **3.00 / 36** | 3.07 / 51 |
| follow: M1-B % / PA mm | 2.63 / - | 5.32 / 86 | **3.35** / 55 | 3.59 / **48** | 3.53 / 62 |

**Per run** (`fda_test/score/sheet.md`; PA vs B0, mm):

| run | PA mm |
|---|---|
| tripod E1 lead | 52 |
| tripod E1 follow | 53 |
| tripod E2 lead | 60 |
| tripod E2 follow | 73 |
| Steadicam E1 lead | 37 |
| Steadicam E1 follow | 48 |
| Steadicam E2 lead | 58 |

**Generated views:**
- The generator follows its own skeleton exactly: PCK 100 % in every run.
- Frames with two people: 2-29 %. Frames with one person: 53-98 %.
- The "partner found" score of 41-71 % mostly matches the merged figure; the contact sheets show no real second dancer.
- GVHMR's global trajectory is off. A Sim3 from its skeleton to B0's dancer leaves 250-560 mm RMS (static-camera
  assumption; the tripod pans and the Steadicam moves), so the views' alignment numbers (self PCK 8-58 %) are rough.

**Two cameras, fused** (root-relative world joints averaged, PA mm):

| dancer | tripod alone | Steadicam alone | fused |
|---|---|---|---|
| lead | 55 | 49 | 46 |
| follow | 53 | 47 | 46 |

## Round 2: each dancer isolated with SAM, and our two-view pose as the skeleton (2026-11-05)

Owner, after round 1: "why aren't we using SAM to isolate the follow and the lead and then running 4DAnyone so there is no
overlap. we also have room data for triangulation" -> "yes do E1+E2". Code: `dancecap/fda_masked.py`, `dancecap/fda_lhmpp.py`,
`remote/cand_fda/` (`pipeline`, `inject`, `appearance`, `lhmpp_score.py`). Results: `work/lucas_luiza/fda_test/score_m/`
(`sheet.md`, `held_pairs_E1.jpg`, `appearance_E1.jpg`), `work/lucas_luiza/fda_test/lhmpp/score.json`,
`work/lucas_luiza/hmr_test/fdam_score/`.

### Verdict

**Isolation fixes the merging. Our pose fixes the skeleton. Neither makes 4DAnyone worth adopting.**

- **Isolation works.** With the partner masked out, every generated frame holds exactly one person (100 %, against 2-29 %
  two people in round 1), and it is the right dancer. In the merged figures of round 1, the partner had been the cause.
- **4DAnyone's own pose gets no better, and the follow's gets worse.** Variant A (masked input, its own GVHMR skeleton),
  hmr_test on the 139 / 141 common frames:
  - lead: PA-MPJPE 50 mm (round 1: 51);
  - follow: 93 mm (round 1: 62).

  Where the lead is in front of her, her mask has holes and GVHMR loses her. ExMR (36 / 48 mm) and SAM 3D Body stay ahead.
- **The room data is used best by giving 4DAnyone our triangulated pose.** Triangulating its generated views would only
  return its own one-camera skeleton, because the views follow that skeleton exactly. Variant B swaps in our two-view
  fit: 10 body-shape values reproduce the fit's joints to 5 mm, and the pose lands on 4DAnyone's own 2D keypoints at
  18-27 px. View 0 was aimed at the other, held-out real camera, and landed within 1-9 degrees of it (median per run).
- **Appearance, against the real held-out camera.** Scored inside the real dancer mask, 8 runs, about 27 frames each:

  | | LPIPS (lower is better) | colour-matched PSNR |
  |---|---|---|
  | 4DAnyone B, view 0 | 0.187 | 14.4 dB |
  | our LHM++ avatar posed by the fit, at the same camera | 0.262 | 15.4 dB |
  | our LHM++ avatar posed by the fit, at the real camera | **0.156** | **18.0 dB** |

  - 4DAnyone's generated texture beats the LHM++ avatar at the same camera on LPIPS in 6 of 8 runs; PSNR ties or goes
    the other way.
  - It cannot render at the real camera: it only renders its own ring of up to 48 fixed cameras around one dancer.
  - What the input camera never saw is invented. The back print of the lead's shirt is not his.
- **Cost of use.** About 6.5 min of L40S per 4-second window, per dancer, for 6 views. That works out to about 20 GPU
  hours ($22) for a 3-minute couple.
  - The output is still videos, not a 3D model. Rendering at any camera, or in VR, would need a 4DGS trained on each
    window (FreeTimeGS; not integrated in 4DAnyone's repo yet).
  - Each 4-second window is generated on its own, so there are seams between windows.

  The LHM++ avatars already render at any camera, for the whole dance, posed by the same two-view fit.

### What ran

- **Inputs:**
  - E1 and E2 (the same 121-frame windows), both cameras, both dancers.
  - Each dancer's SAM 2.1 mask comes from the single-camera takes (`lucas_luiza_h` / `lucas_luiza_v` `people/masks_01.npz`).
    The frame map is found by box IoU: tripod offset 0, Steadicam source index -55, IoU 0.83-0.90. The slot is chosen per
    frame against the two-view teacher box.
  - Everything outside the dilated mask is set to grey.
- **Variants:** A = 4DAnyone as published on the isolated clip. B = the same clip with our fit injected as its motion,
  ring rotated so view 0 faces the held-out camera. 6 views.
- **Runs:** 16 of 16 OK, on two L40S pods in parallel (one per camera), $2.63. The LHM++ baseline ran on a third pod
  ($0.24): the same avatars and the same cameras, the same score (`lhmpp_score.py`).

| run | A: PA mm (round 1) | B: LPIPS / PSNR dB | LHM++ same camera | LHM++ real camera |
|---|---|---|---|---|
| tripod E1 lead | 55 (52) | 0.207 / 11.8 | 0.310 / 13.6 | 0.189 / 17.5 |
| tripod E1 follow | 100 (53) | 0.178 / 10.7 | 0.291 / 10.6 | 0.159 / 14.8 |
| tripod E2 lead | 65 (60) | 0.228 / 11.8 | 0.266 / 15.1 | 0.209 / 17.4 |
| tripod E2 follow | 94 (73) | 0.187 / 11.6 | 0.265 / 11.6 | 0.178 / 14.8 |
| Steadicam E1 lead | 58 (37) | 0.188 / 18.6 | 0.178 / 19.1 | 0.153 / 19.6 |
| Steadicam E1 follow | 67 (48) | 0.179 / 17.2 | 0.259 / 16.8 | 0.127 / 19.1 |
| Steadicam E2 lead | 59 (58) | 0.202 / 18.8 | 0.178 / 20.5 | 0.143 / 22.2 |
| Steadicam E2 follow | 65 (-) | 0.174 / 17.0 | 0.224 / 15.7 | 0.127 / 18.6 |

**Caveats:**
- The view-0 camera is 4DAnyone's: a different focal length and distance from the real one, 1-9 degrees off in
  direction. Both crops are therefore letterboxed to a common canvas, and the silhouettes do not coincide (the grey and
  black areas in `appearance_E1.jpg`). That costs 4DAnyone B and LHM++-at-the-same-camera alike, which is why the
  real-camera column is the production comparison.
- Raw PSNR is low across the board because the real crops keep the partner's occlusion holes.

### What would be worth trying

4DAnyone B's texture (variant B: our pose, the dancer isolated) beat our LHM++ avatar at the same camera on LPIPS. Its
views could serve as extra reference images for the LHM++ avatar, covering sides the input camera saw least. Expect the
invented details too (the back print). That is a separate, small test, not a reason to put 4DAnyone in the pipeline.

## If revisited

Re-run only if 4DAnyone (or a successor) states multi-person support. Single-dancer solo footage would be the one fit.
The pod job, both patches (our tracker boxes, the parallel skeleton render) and the scorer are ready:

```
python -m dancecap.fda_test job
python -m dancecap remote run --jobs work/lucas_luiza/jobs/pod/lucas_luiza_fda.json --max-cost 6
```

Plan for about 17 min per run.
