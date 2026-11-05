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

## If revisited

Re-run only if 4DAnyone (or a successor) states multi-person support. Single-dancer solo footage would be the one fit.
The pod job, both patches (our tracker boxes, the parallel skeleton render) and the scorer are ready:

```
python -m dancecap.fda_test job
python -m dancecap remote run --jobs work/lucas_luiza/jobs/pod/lucas_luiza_fda.json --max-cost 6
```

Plan for about 17 min per run.
