# Uthana Video-to-Motion 3.0: one tool for feet, hands and body pose? (lucas_luiza, 2026-11-12)

Owner, 2026-11-12: "test this technology, especially version 3.0. we will need to use the same technique of dancer isolation
before processing. the end goal should evaluate unification of tools to solve foot slip, hand and arm placement, and body
pose". The test ran in the owner's Uthana account (web app, driven in Chrome; $5 of test credit). Code:
`dancecap/uthana_test.py`. Results (likeness-derived, git-ignored): `work/lucas_luiza/uthana/` (`inputs/`, `out/*.bvh`,
`score/score.json`, `score/skeletons_side.jpg`).

## Verdict

**No: Uthana cannot replace our set of tools. Its one strength is planted feet in slow figures.**

| Target | Uthana VTM 3.0 from the tripod | Best one-camera tool here | Ours today |
|---|---|---|---|
| Body pose (PA-MPJPE, mm) | 78-107 on visible dancers | ExMR 49-65 | two-view fit (the reference) |
| Foot slip, hold / embrace | **7-11 cm/s, at the reference's own level** | ExMR 12-28 | two-view + contact term (E2b) |
| Foot slip, fast spins | 53 / 78 cm/s | ExMR 41 / 57 | two-view 38 / 52 |
| Arm placement (elbow / wrist, mm) | 130-222 | ExMR 97-128 | two-view fit |
| Hand on the partner (lead's right hand, recall within 15 cm) | 0-2 % (22-36 cm off) | ExMR 29-76 % | two-view + hand magnets |
| Placement of the dancer (root path after one Sim3) | 15-63 cm | 2-21 cm | room map + two cameras |

- **Feet: genuinely good where the dancer stands still.** The 3.0 "foot IK" keeps planted feet planted in the hold (E1) and the
  lead's embrace (H1), as well as our two-view fit does. In fast spins it slides more than ExMR and has long tails
  (p90 187-212 cm/s).
- **Body pose: weaker than every learned one-camera method we have measured.** Part of the gap is structural: the motion comes
  retargeted onto Uthana's "Tar" mannequin (UE5 skeleton, its own bone proportions), not the dancer's body.
- **Arms and hands: the weakest of all methods.** Each dancer is reconstructed alone, so nothing places a hand on the partner.
  The lead's right hand on the follow's back, the clearest contact in these clips, lands within 15 cm in 0-2 % of frames.
- **No placement of its own.** Uthana takes no camera, so a panning tripod reads as body motion. After the best single
  alignment per clip, its pelvis path is still 15-63 cm off. A couple cannot be assembled from two of its outputs without our
  cameras and our fit.
- **Occlusion breaks it.** The follow in the embrace (H1: behind the lead, hidden for much of the clip, which is the normal
  view from one camera) comes out as noise: PA 173 mm, slip 155 cm/s, lying on the floor in places.

**The useful lesson is narrow:** its foot IK shows what planted feet should look like in slow figures. That is the same target
as our contact term (E2b, adopted for fits from two or more phones). Nothing in Uthana is needed for it.

## What ran

- **Clips:** tripod only (the one-phone case, and the "stable camera" Uthana asks for), the cores of hmr_test's E1 closed hold
  (7.8 s), H1 embrace (12.0 s) and H3 fast spins (15.3 s), both dancers: 6 uploads, 70.5 s of video, about $4.23 at
  $0.06/s.
- **Isolation, same technique as the 4DAnyone test, but re-segmented:**
  - The people stage's masks were not good enough here. Both dancers took the same mask on 6-38 % of frames, and in H3 the
    lead was lost for 2 s while his mask drifted onto a seated spectator. The stale masks this left behind were the owner's
    catch ("a mask over random space in the shape of lucas").
  - So each clip was re-segmented with SAM 2.1 (`jl_people_sam.py`, unchanged), anchored every 0.5 s where both
    existing masks are clean: the dancer's box, positive points on the own torso, negative points on the partner's. The
    partner's mask is subtracted.
  - The result is a mask on every frame. Grey outside the dilated mask, native 1280 x 720, 29.97 fps.
- **Uthana:**
  - Video-to-Motion 3.0 only (2.1 is preselected in the app), each about 80 s to process, BVH download.
  - The download is a UE5 mannequin skeleton (pelvis, thigh / calf / foot / ball, upperarm / lowerarm / hand, full fingers),
    Y-up in cm, 30 fps, root path kept ("In place" off).
- **Score:** the same metrics for every method, on the same reference frames (`uthana_test score`, CPU pod).
  - **Pose:** PA-MPJPE on 15 joints every rig has.
  - **Arms:** elbows and wrists after a similarity fitted on the torso only.
  - **Feet:** ankle horizontal speed on the two-view fit's stance frames, and the stance ankle's height above that method's
    floor.
  - **Hands:** wrist within 15 cm of the partner's pelvis / neck / shoulders where the two-view fit has it there.
  - Every method's dancer is placed by one Sim3 per clip onto the two-view dancer. That is generous to Uthana, which has no
    placement at all.
  - Baselines: our old one-camera fit (B1), SAM 3D Body (B2), ExMR, all on the same tripod frames.

## Per clip

PA / arm in mm, skate (median, p90) in cm/s, root in cm. Reference skate is the two-view fit's own.

| clip | dancer | Uthana PA / arm / skate / root | ExMR PA / arm / skate / root | SAM 3D Body PA / skate | B1 PA / skate | reference skate |
|---|---|---|---|---|---|---|
| E1 hold | lead | 107 / 222 / **10.8** (90) / 23 | 53 / 128 / 20.1 (62) / 2 | 70 / 32.8 | 99 / 31.9 | 11.1 |
| E1 hold | follow | 81 / 168 / **10.5** (80) / 22 | 53 / 106 / 28.1 (71) / 4 | 51 / 40.1 | 74 / 39.1 | 11.1 |
| H1 embrace | lead | 78 / 141 / **7.2** (65) / 15 | 49 / 101 / 12.4 (34) / 3 | 52 / 27.8 | 81 / 24.7 | 8.6 |
| H1 embrace | follow | 173 / 296 / 155 (856) / 35 | 75 / 164 / 27.5 (83) / 4 | 78 / 51.1 | 112 / 28.5 | 18.6 |
| H3 spins | lead | 99 / 135 / 52.6 (187) / 63 | 65 / 114 / 40.7 (98) / 19 | 73 / 60.7 | 98 / 50.1 | 38.1 |
| H3 spins | follow | 94 / 130 / 78.3 (212) / 46 | 52 / 97 / 56.9 (144) / 3 | 57 / 65.7 | 70 / 58.5 | 51.7 |

**Hand on the partner:** recall within 15 cm, and the median distance error.

| | contact frames | Uthana | ExMR | SAM 3D Body | B1 |
|---|---|---|---|---|---|
| E1, lead's right | 25 | 0 % / 35.5 cm | 36 % / 3.1 cm | 4 % / 8.8 cm | 16 % / 6.5 cm |
| H1, lead's right | 101 | 2 % / 22.4 cm | 76 % / 2.1 cm | 42 % / 5.0 cm | 32 % / 7.3 cm |
| H1, follow's right | 61 | 2 % / 26.4 cm | 30 % / 7.5 cm | 44 % / 5.6 cm | 0 % / 15.4 cm |
| H3, lead's right | 34 | 0 % / 21.8 cm | 29 % / 10.4 cm | 18 % / 8.7 cm | 6 % / 14.5 cm |
| H3, follow's left | 27 | 26 % / 42.5 cm | 41 % / 8.5 cm | 44 % / 8.9 cm | 48 % / 5.1 cm |

**Caveats:**
- Three clips, one take, one camera.
- Uthana's pose numbers include the mannequin's proportions. The per-frame similarity removes overall size, not limb ratios.
- The upload included Uthana's terms §6.2: inputs and outputs may be used for training unless an opt-out agreement is signed.
  The owner approved testing in their account.

## Pipeline fixes made on the way (useful beyond this test)

- **`fda_masked.role_masks`:**
  - assigns the two mask slots jointly per frame (they collided on 6-38 % of frames);
  - rejects spilled masks (area 0.35-1.8 x the dancer's median, box IoU >= 0.3);
  - the 4DAnyone round-2 doc carries a correction note.
- **`front_pod.fetch_sam2`:** falls back to the byte-identical Hugging Face copy of SAM 2.1 (sha256 checked). Meta's CDN
  served an expired certificate on 2026-11-12, which would break every SAM step on a pod.
- **Key files:** the RunPod key now lives in `~/Desktop/keys/`. These runs set `DANCECAP_RUNPOD_KEY_FILE`, and the runner's
  default path still points at `~/Desktop/runpod-key.txt`.
