# Appearance decision: current avatar vs moving 3D Gaussian splats vs hybrid

Decision record, 2026-10-19, revised the same day after two independent reviews (section 10).
Take `larissa_kadu_38`, 58.0-61.5 s (the same phone videos as `larissa_kadu`).
Nothing here is committed. All likeness data (splats, renders, videos) stays in git-ignored `work/`.

The owner's question (2026-10-19, verbatim): "remember that we are going for realism and we need to decide between
avatar and 3d gauss splat moving. the pose can still be MAMMA or better, but the rendering needs to be improved from
the avatar models we are currently using. I need to see picture side-by-sides".

Tags:
- **(m)** measured by us. The file is named.
- **(r)** reported by a paper, README, source file or vendor. We did not measure it.
- **(e)** our estimate.
- **by eye**: someone looked at the pictures. Not a number.

---

## 1. The answer

### 1.1 What "the current avatar" means in every number below

- Every avatar tile, avatar number and Jev judgement uses an **offline render** of the shipped avatar: the shipped
  skin and the shipped `face` albedo, unlit, 2 x 2 supersampled (m, `renders/avatar/manifest.json`).
- That render has **no hair cards** (the follow's hair is a short painted cap), **no procedural shoes**, **no eye
  blinks** and **no scene shading**. Unity adds these four.
- Unity also shades the albedo with the scene's main light and ambient at 35 % strength (`HM_AvatarLit.shader`,
  `_LightInfluence` 0.35, `SmplxAvatar.cs` line 312). The offline render is unlit, so the headset avatar has soft
  body shading that the judged render lacks (r, source read). The texture itself is the same.
- On the headset the avatars are also shown 65 % transparent by default (section 6.1); every render judged here is
  opaque.
- The shipped Unity avatar was **not captured at the same instants**. `sheet_unity_reference.jpg` shows another moment
  (scene 23.0 s, about 53.15 s, outside the compared clip), a near-T pose, a black background with HUD overlays, and the
  face shape from before the 2026-10-08 edit.
- So the headline numbers are for a weaker avatar than the one that ships. Hair cards, shoes and blinks change only
  the hair, feet and eyelids. The 35 % shading adds body shape but keeps the painted texture. So they are unlikely to
  reverse the ranking (e). This is not measured.
- Optional owner check **O1**: capture the Unity avatar from the VR seat at one instant (for example i2, 59.88 s), so it
  can sit next to LHM++ on one sheet.

### 1.2 Recommendation

| use | recommendation | key evidence | confidence |
|---|---|---|---|
| **(a) 3D tutorial videos** (rendered on a PC) | **Moving 3D Gaussian splats, the "skinned avatar" kind.** One LHM++ Gaussian avatar per dancer. The MAMMA SMPL-X fit poses it every frame. Rendered offline at 1080p. | In all 18 real views an LHM++ render has the highest Jev P(photo) (m). Mean P(photo) 0.82 (1-phone) and 0.85 (8-phone) vs 0.19 for the offline avatar, n = 18 views (m). It holds on phone 03, which LHM++ 1-phone never saw: 0.76 vs 0.30 (m). The sheets agree by eye. No pixel metric gives LHM++ a clear lead (section 3.2). | **High** that it looks more photographic than the offline avatar render. **Medium** against the avatar the headset shows (section 1.1). The look shown comes from LHM++'s own posing code; our own posing path is not built yet (section 4). |
| **(b) VR playback, Quest 3 standalone** | **The current avatar stays.** Skinned splats only if a device test shows the splat layer fits the measured GPU headroom. Do not move the headset to the hybrid mesh. | No splat has run on our Quest. Fill, not Gaussian count, sets the cost. Our Gaussians are large (m). Our fill estimate alone is 8-20 ms at the VR seat (e), against about 4-7 ms of room at 72 Hz (e, from device logs; GPU clock level not logged). | **Low** for 2 x 160k. Expect heavy reduction or distillation to smaller Gaussians (new work), or PC-tethered VR. |

Why splats, and why not the hybrid:
- The realism comes from the Gaussians themselves: soft edges, hair and clothes volume, soft shading (by eye).
- Baking the same LHM++ colours onto the SMPL-X mesh (the hybrid) lost it. Hybrid P(photo) is 0.14 (plain) and
  0.20 (shell), against 0.19 for the offline avatar. Neither hybrid won a single Jev choice (m). The hair becomes a
  painted red helmet or a smooth hood, and the faces look rubbery (by eye).
- AGS, our own 8-phone trained splats, is blurry with floaters (by eye). Jev never picked it (m). Stop it.
  Its PSNR and SSIM are the best of all methods, for reasons that are not realism (section 3.2).
- Per-frame ("volumetric") splats need a camera dome. Our NoPo4D try on these phones failed (10.4-13.6 dB in-sample,
  duplicated dancers; DANCER_SPLATS_RESEARCH section 1) (m). Skinned splats can also be re-posed later with a better
  pose, mirrored or slowed down. Per-frame splats cannot.

Limits of this evidence:
- One automated judge (Jev) plus the sheets. 18 independent views. Jev's mean confidence in its choice is 0.36 (m).
- All renders are in 58.0-61.5 s; the LHM++ inputs are from 58.1-61.9 s (section 3.4). Other takes are untested.
- LHM++ faces are not likenesses (section 5.3). LHM++ hair is rigid (section 5.1).
- LHM++ is not indistinguishable from a photo. It is soft at 2x zoom (by eye).

---

## 2. The side-by-sides

All files are in [`work/larissa_kadu_38/mono4d/compare/`](../work/larissa_kadu_38/mono4d/compare/).
- Instant and close-up sheets: the real photo first, then six renders in the same crop:
  current avatar | AGS splats | LHM++ 1-phone | LHM++ 8-phone | hybrid mesh | hybrid mesh + shell.
- Turntable and VR-seat sheets: renders only. No phone stands there, so there is no real photo.
- Unity reference sheet: two Unity screenshots only.

Start with `sheet_i2.jpg` (whole bodies) and `sheet_closeup_hair_follow.jpg` (her hair).

| file | what it shows | look for |
|---|---|---|
| [sheet_i0.jpg](../work/larissa_kadu_38/mono4d/compare/sheet_i0.jpg) ... [sheet_i5.jpg](../work/larissa_kadu_38/mono4d/compare/sheet_i5.jpg) | 6 instants. Rows: phone 03 (480 x 848), phones 02 and 06 (1080 x 1920). Each render shows Jev's P(photo) and picks. | Cloth folds, hair volume, skin. Tags: green = held out, amber = that phone was in training, red = that frame was in training. |
| [sheet_turntable.jpg](../work/larissa_kadu_38/mono4d/compare/sheet_turntable.jpg) | 12 cameras every 30 deg at 59.185 s, 3 m away. No real photo. | Sides no phone filmed |
| [sheet_vr_seat.jpg](../work/larissa_kadu_38/mono4d/compare/sheet_vr_seat.jpg) | The headset seat: 2.5 m away, 1.6 m high, 50 deg from the nearest phone, 6 instants. No real photo. | What VR would show |
| [sheet_closeup_face_follow.jpg](../work/larissa_kadu_38/mono4d/compare/sheet_closeup_face_follow.jpg) | Her face, 32 cm boxes, re-rendered at 512 px | Her face shows only at i0, i1 and i5. Rows i2, i3 and i4 show the back or side of her head. |
| [sheet_closeup_face_lead.jpg](../work/larissa_kadu_38/mono4d/compare/sheet_closeup_face_lead.jpg) | His face, same layout | Likeness, skin tone |
| [sheet_closeup_hair_follow.jpg](../work/larissa_kadu_38/mono4d/compare/sheet_closeup_hair_follow.jpg) | Her hair, 60 cm box from the crown down | Length, volume, colour |
| [sheet_unity_reference.jpg](../work/larissa_kadu_38/mono4d/compare/sheet_unity_reference.jpg) | Real Unity screenshots of today's avatars (2026-10-07). Another moment (about 53.15 s), near-T pose, black background, HUD, older face shape. | What the headset adds (hair cards, shoes, soft scene shading). Not the same instant. |
| [clip_vr_seat.mp4](../work/larissa_kadu_38/mono4d/compare/clip_vr_seat.mp4) | All six methods from the VR seat; 105 frames at real time, then at 0.5x | Motion, hair, smudges |
| [clip_phone03.mp4](../work/larissa_kadu_38/mono4d/compare/clip_phone03.mp4) | Real phone 03 \| avatar \| LHM++ 8-phone \| hybrid shell | The real panel's header says "held-out phone". That is wrong for two panels: the avatar bake and LHM++ 8-phone saw other phone-03 frames (their own sub-labels say so). |
| [clip_phone03_all.mp4](../work/larissa_kadu_38/mono4d/compare/clip_phone03_all.mp4) | Real phone 03 + all six methods | Same. Only AGS, LHM++ 1-phone and the hybrids never saw phone 03. |

What to look for that the numbers miss (by eye, review of 2026-10-19):
- LHM++ 1-phone makes the lead's skin noticeably lighter than the real man (`sheet_i2.jpg`, lead face sheet i0 and i2).
  LHM++ 8-phone is closer.
- LHM++ 8-phone colours are darker overall. But on phones 02 and 03 its burgundy hair and darker skin match the
  photos better. So its colour is not only a defect.
- Reddish-brown smudges on the follow's white top near the shoulder and back, in LHM++ (VR-seat clip frames 84-88,
  `sheet_i2.jpg`).
- At 2x zoom, LHM++ is soft overall.
- The two dancers hide each other correctly on all sheets.

Older quick-look sheets: `compare/lhmpp/` (LHM++ only), `compare/hybrid/check/` (hybrid only).

What ran:
- LHM++ pod session `20261019-182851-batch`, RTX 4090, 8.5 pod-min, $0.1264 (m).
- LPIPS for all six methods: pod session `20261019-213202-batch`, RTX 3090 (community), 3.7 pod-min, $0.0135 (m).
  One earlier try (`20261019-212746-batch`) got a pod with a broken GPU and stopped before scoring: $0.016 (m).
- Avatar, AGS and hybrid renders, sheets, videos and the other metrics ran locally ($0).
- Jev judge: 360 requests, $0.046 (m). Bookkeeping: `compose.json`.

---

## 3. Options compared

### 3.1 Jev realism judge (TypeSafe jev-1.13.0, `jev.json`)

How it was asked (m, `jev.json`):
- 18 real views = 6 instants x 3 phones. These are the independent samples: **n = 18**.
- Choice: "which render looks most like a real photograph", the six renders in random order. Each view was asked 3
  times with a new order: 54 asks, not 54 independent questions.
- P(photo): each render judged alone. P(same people): each render against the real photo. **Means** over the 18 views.

| method | choice: views won by majority (of 18) | choice: asks won (of 54) | asks won, phones 02/06 (of 36) | asks won, phone 03 (of 18) | mean P(photo), 18 views | mean P(photo), phone 03 (6 views) | mean P(same people), 18 views |
|---|---|---|---|---|---|---|---|
| current avatar (offline render) | 0 | 0 | 0 | 0 | 0.19 | 0.30 | 0.79 |
| hybrid mesh | 0 | 0 | 0 | 0 | 0.14 | 0.28 | 0.81 |
| hybrid mesh + shell | 0 | 0 | 0 | 0 | 0.20 | 0.38 | 0.85 |
| AGS splats | 0 | 0 | 0 | 0 | 0.20 | 0.24 | 0.77 |
| **LHM++ 1-phone** | 7 | 20 | 18 | 2 | 0.82 | 0.76 | 0.93 |
| **LHM++ 8-phone** | 11 | 33 | 18 | 15 | 0.85 | 0.81 | 0.94 |
| "none" | 0 | 1 | 0 | 1 | - | - | - |

What these numbers can and cannot say:
- Jev's mean confidence in its choice is 0.36 (m). Many picks are near-ties between the two LHM++ renders, or close
  to "none" (i0_c03: 8-phone 0.33 vs none 0.32; i1_c03, third ask: "none" won at 0.28).
- An LHM++ render has the highest P(photo) in all 18 views. The closest any other method gets is the avatar at
  i4_c03: 0.76 vs 0.87 (m).
- P(photo) saturates near the top. The real photo cut out on grey averages 0.91. The best LHM++ render equals or
  beats it in 2 views (i2_c02 0.94 vs 0.93; i5_c03 0.87 vs 0.87) and is within 0.01 in 4 more (i2_c03 0.90 vs 0.91,
  i4_c03 0.87 vs 0.88, i4_c06 0.84 vs 0.85, i5_c06 0.92 vs 0.93) (m). So in 6 of 18 views LHM++ is level with the
  real cut-out or within 0.01. P(photo) cannot say how far LHM++ is from a real photo.
- Dropped as evidence: "the real cut-out wins 18 of 18 when added". That option was the reference photo itself, cut
  out onto grey. Jev only had to spot a near-duplicate.
- LHM++ 1-phone vs 8-phone: they tie. On phones 02/06 they split 18 vs 18 asks (6 vs 6 views by majority). 8-phone's
  lead is all on phone 03 (15 of 18 asks, 5 of 6 views), and 8-phone had phone-03 crops in its input while 1-phone
  never saw phone 03. In pixels they are within 0.12 dB PSNR cc and 0.008 LPIPS (m).
- P(same people) does **not** measure facial likeness. The positive control, a real photo of the same couple from
  another phone, averages 0.87 and drops to 0.33 at i1_c03. That is below LHM++'s 0.93-0.94 (m). The question
  rewards a matching viewpoint and clothes.

### 3.2 Pixel metrics (`metrics.json`, `lpips_all.json`)

How: per dancer, inside the dancer's real mask; crop = the mask box + 10 %; render over black; "cc" = after a
least-squares colour fit. **Medians** over 36 dancer-views (18 views x 2 dancers). LPIPS: AlexNet,
`dancer_splats.evaluate._crop_scores`, all six methods in one pod run on the same crops and masks; lower is closer.
The job pinned `lpips` 0.1.4 (`pod_job_lpips.json`); the pod did not record the version (`lpips_version` null).
PSNR and SSIM ran locally (`metrics.json`).

All 18 views (m):

| method | PSNR dB | PSNR cc dB | SSIM cc | LPIPS | LPIPS cc | dancer IoU |
|---|---|---|---|---|---|---|
| current avatar (offline render) | 10.40 | 15.80 | 0.387 | 0.222 | 0.180 | 0.71 |
| hybrid mesh | 10.62 | 15.88 | 0.389 | 0.222 | 0.195 | 0.71 |
| hybrid mesh + shell | 11.79 | 16.06 | 0.400 | 0.215 | 0.191 | 0.78 |
| AGS splats | 14.80 | 16.35 | 0.420 | 0.197 | 0.187 | 0.73 |
| LHM++ 1-phone | 12.26 | 16.16 | 0.418 | 0.188 | 0.174 | 0.80 |
| LHM++ 8-phone | 12.80 | 16.28 | 0.409 | 0.180 | 0.166 | 0.79 |

Phone 03 at i0, i2, i4, i5 (8 dancer-views; no method trained on these frames; the avatar bake and LHM++ 8-phone saw
other phone-03 frames) (m):

| method | PSNR dB | PSNR cc dB | SSIM cc | LPIPS | LPIPS cc |
|---|---|---|---|---|---|
| current avatar (offline render) | 12.43 | 17.45 | 0.359 | 0.226 | 0.175 |
| hybrid mesh | 11.34 | 17.16 | 0.369 | 0.245 | 0.183 |
| hybrid mesh + shell | 11.53 | 17.45 | 0.376 | 0.242 | 0.180 |
| AGS splats | 14.67 | 18.23 | 0.414 | 0.194 | 0.172 |
| LHM++ 1-phone | 11.66 | 17.62 | 0.376 | 0.225 | 0.173 |
| LHM++ 8-phone | 14.39 | 17.67 | 0.378 | 0.186 | 0.169 |

What this says:
- **No pixel metric gives LHM++ a clear lead.** PSNR and SSIM favour AGS. PSNR cc spans 0.55 dB over all six
  methods, and AGS is highest (m). On the held-out phone-03 frames AGS is best on PSNR cc (18.23 vs 17.62 / 17.67)
  and SSIM cc (0.414 vs 0.376 / 0.378) (m).
- LPIPS (new, m): over all views LHM++ has the lowest raw LPIPS (0.180 / 0.188 vs 0.197-0.222). After the
  colour fit the gap to the avatar is small (0.166 / 0.174 vs 0.180). On the held-out phone-03 frames all six methods
  are within 0.014 LPIPS cc (0.169-0.183). The lowest LPIPS cc per dancer-view (36) goes to LHM++ 1-phone 12 times,
  AGS 11, LHM++ 8-phone 9, the avatar 4 and the hybrids 0.
- Checks (m): the LPIPS pod's PSNR equals `metrics.json` on all 216 rows (same crops), and its LHM++ LPIPS equals the
  first LHM++ pod's rows to within 0.0001.
- AGS's raw PSNR lead (14.80 vs 10.40-12.80 dB over all views) has three causes. AGS is drawn with its own trained
  per-photo colour transform on its training photos (`renders/ags_8phone/manifest.json`, "colour"). 10 of the 12
  phone-02/06 views are its training frames (red tags). And a blurry render averages well. On phone 03 (identity
  colour, held out) its raw lead over LHM++ 8-phone shrinks to 0.28 dB (14.67 vs 14.39) (m).
- So the ranking rests on the Jev judge plus the visual sheets, not on pixel agreement with the photos.
- Dancer IoU: LHM++ 0.79-0.80 vs 0.71 for the avatar: clothes and hair volume (m).
- No view is held out from every method. Per method, `metrics.md` lists "heldout_for_this_method"; LHM++ 8-phone and
  the avatar have no held-out views at all.

### 3.3 Popping: a weak proxy only

- Number: mean second difference of the pixels inside the dancers over the VR-seat clip (0-255): avatar 9.0, AGS 7.41,
  LHM++ 1-phone 7.84, LHM++ 8-phone 6.39, hybrids 7.47 / 7.46 (m, `compose.json`).
- It is not evidence about popping. Sharp texture moving dominates it, so softer renders score lower. It was computed
  on panels scaled by 0.504. It cannot see splat sort pops. It says nothing about the Quest renderer, which sorts
  differently.
- Watch `clip_vr_seat.mp4` at 0.5x instead.

### 3.4 How close the test frames are to the LHM++ inputs

- Both LHM++ variants were built from crops at 58.1-61.9 s. The test instants are 58.28-61.42 s (m, manifests).
- LHM++ 1-phone on phone 06: three views are 0.07-0.10 s (2-3 frames) from an input crop, and i4_c06 is an input
  frame (follow). LHM++ 8-phone: four views are 0.07-0.10 s from an input crop of the same phone (m, `metrics.json`
  trained_status).
- So rigid hair and frozen folds match this window by construction.
- No render outside 58.0-61.5 s exists. "One avatar per person and outfit, reused by every take" is untested.
  Step A5 tests it.

### 3.5 All options

| option | realism evidence | pose source | hair | clothes | face | Quest 3 standalone | PC / video | work to adopt | main risks | licence |
|---|---|---|---|---|---|---|---|---|---|---|
| **Current avatar** (textured skinned SMPL-X, hair cards) | Offline render: 0 of 18 views, mean P(photo) 0.19 (m). Painted look, smears on her nape and back (by eye). | MAMMA SMPL-X, 55-bone GPU skinning | Unity: 160 simulated guides as hair cards; swings, but game-like | Paint on a naked body; IoU 0.71 (m) | Painted portrait; lead reads right, follow doll-like (by eye); Unity adds blinks | Ships today. Recap at 72.0 fps, GPU 8.1-10 ms of 13.9 ms, hair on (m, NOTES_X3) | Looks CG | None | Does not meet "realism" | SMPL-X non-commercial |
| **Hybrid mesh** (shipped mesh + LHM++-baked texture) | 0 of 18 views, mean P(photo) 0.14 (m) | Same as avatar | Painted on the scalp (helmet); Unity cards would still work | Cleaner colours, no volume; IoU 0.71 | Rubbery, light nose patch | Drop-in: 2 PNGs (not tried in Unity) | No gain | 15 min to try | No realism gain | LHM++ CC BY-NC + SMPL-X |
| **Hybrid + shell** (subdivided mesh pushed out) | 0 of 18 views, mean P(photo) 0.20 (m) | Same, 43,554 vertices per dancer | Smooth hood | Volume; IoU 0.78 | Rubbery; blinks would tear (new vertices) | Not tested in Unity; face blendshapes break | No gain | 1-2 days of Unity fixes | No realism gain | same |
| **LHM++ skinned splats** (recommended for video) | Highest P(photo) in 18 of 18 views; mean 0.82-0.85 (m) | MAMMA SMPL-X. Today: LHM++'s own posing on the pod. Planned: our own LBS (section 4) | Long red hair with volume, but rigid: no swing | Real folds and volume; IoU 0.79-0.80; folds frozen | Soft, generic, fixed expression | Not measured. Fill-bound; low confidence (section 6) | Offline gsplat, 1080p, locally or on a pod | (a) about 1.5 weeks; (b) about 1.5 weeks of device tests | Quest GPU time; rigid hair; faces; loose skirts; reuse untested | LHM++ weights CC BY-NC 4.0 + SMPL-X |
| **AGS** (our multi-view splats) | 0 of 18 views, mean P(photo) 0.20; gates 1-4 failed (m). Best PSNR and SSIM (section 3.2). | Own fit + pose corrections | Red cloud | Blurry, colour casts | Unrecognisable | Not pursued | Blurry | Large: retrain, new method | Phones too few and too soft | ours + SMPL-X |
| **Per-frame 4D splats** (NoPo4D, 4DGS HexPlane) | NoPo4D: 10.4-13.6 dB in-sample, duplicated dancers (m) | None: motion is baked in | Moves if captured | Moves if captured | As captured | Renderer plays HexPlane 4DGS (r); about 860 MB per 3.5 s as frames (e) | Needs a dome | Not possible with our phones | Cannot re-pose, mirror or fix pose | varies |
| **SqueezeMe-style distilled avatar** (reference only) | Paper quality on a dome (r) | LBS + linear correctives | Not stated | Pose-dependent correctives | Dome quality | 3 avatars at 72 fps on Quest 3, custom Vulkan renderer (r) | - | Not available: 512-camera dome data, no public code found | - | - |

---

## 4. How the pose drives each option

The pose source stays the MAMMA SMPL-X fit (`work/larissa_kadu_38/smplx/{lead,follow}.npz`, 30 Hz grid, 55 joint
rotations + root translation per dancer). Unity plays the refined `smplx_rf2_noslip/` motion. For a fair test, the
comparison posed every method with the reference fit (AGS used its own take's fit, mapped by a 12.5 mm body Sim3).

**What exists today, and what is only planned (LHM++):**

| | today (every LHM++ render in this record) | planned |
|---|---|---|
| who poses the Gaussians | LHM++'s own `renderer.animate_gs_model`, on the pod (`remote/cand_lhmpp/lhmpp_compare_pod.py`, lines 211-229) | our own plain LBS: offline (A1, A2), then a Unity compute pass (B2) |
| what we saved | canonical PLYs with `x y z`, normal, `f_dc`, opacity, scale, rotation only. **No skinning weights, no bone ids** (m, PLY header) | A1 exports the weights and the neutral-pose transform |
| what is unknown | whether `animate_gs_model` does more than plain LBS (for example diffused skinning weights or pose correctives). Not checked. | A2 must reproduce the pod renders on the 18 views before we rely on plain LBS |

So the look in the sheets is LHM++'s pipeline. The proposed local and Unity path is not reproduced yet.

- **Avatar.** `SmplxAvatar.cs` builds a SkinnedMeshRenderer with 55 bones. Each frame sets the bone rotations from the
  fit. Normal GPU skinning.
- **Skinned splats (planned).** LHM++ makes 160,000 Gaussians per dancer (m) in its neutral (A-) pose. Each Gaussian
  sits on a SMPL-X query point. Each frame, linear blend skinning (LBS) moves every Gaussian's centre and turns its
  rotation with the same 55 joint rotations and translation. Scale, opacity and colour stay fixed.
  - LHM++ takes 10 shape betas; the fit has 16. With the fit's first 10, LHM++'s posed joints are 3.6 / 2.8 mm (lead /
    follow, median) from the reference; the pelvis is exact (m, `lhmpp/pod_out/lhmpp_compare_pod.json`).
  - Expression coefficients are 0: no smile. The fit's jaw, eye and hand poses are passed in (m, the pod script).
    The avatar uses the shipped face; Unity adds blinks.
- **Hybrids.** The avatar's own skinning.

Both paths read the same SMPL-X parameters. A better pose later (for example SAM 3D Body inside MAMMA_MONO,
MONO4D_NOVEL_VIEWS 10.6) plugs into either one unchanged. Mirroring, slow motion and free cameras for tutorials work
for both.

---

## 5. Topic by topic

### 5.1 Hair (the biggest open gap for splats)

| option | look | motion |
|---|---|---|
| Avatar in Unity | Hair cards with tip bloom (`HairStrands.cs`). Game-like (`sheet_unity_reference.jpg`). | 160 guides x 16 segments, Verlet + XPBD, collisions. Swings. |
| Avatar offline (the one judged here) | Short painted cap; no long hair from behind | Skinned |
| LHM++ | Long, red, full, photographic (`sheet_closeup_hair_follow.jpg`, by eye) | **Rigid**: skinned to head, neck and back. No swing. |
| Hybrids | Painted helmet, or a smooth hood (shell) | Skinned |

Her head turns peak at 285 deg/s (59.15 s) and 283 deg/s (60.89 s), and 185 deg/s at 61.29 s (m, `views.json`).
Rigid hair will show there.
Proposed fix (step A4): bind her hair Gaussians to the same 160 guide strands that Unity simulates. Offline renders
use recorded guide tracks; the Quest uses the live simulation. DiffLocks strands were tested and not adopted
(MONO4D 10.6).

### 5.2 Loose clothing

- LHM++ has clothes volume: dancer IoU 0.79-0.80 vs 0.71 for the naked-body avatar (m). Jeans and t-shirt folds look
  real (by eye).
- The folds are frozen. Clothes move with the skinning only. There is no cloth motion and no pose-dependent wrinkle.
  This window cannot show the problem: the inputs came from the same seconds (section 3.4).
- Fine for this take (jeans, t-shirt, crop top). A skirt or loose dress would stretch between the legs. Not tested.

### 5.3 Faces

- Avatar: a painted portrait (the shipped `face` albedo). The lead reads right (beard, skin tone). The follow looks
  doll-like (by eye). Unity adds blinks.
- LHM++: soft and generic, not likenesses (by eye). Expression is fixed, so her smile at i0 is missing.
- LHM++ 1-phone makes the lead's skin lighter than in the photos; 8-phone is closer (by eye).
- Her face is visible only at i0, i1 and i5 on the face sheet. That is the whole face evidence for her.
- Jev's "same people" score is not a likeness measure (section 3.1).
- At the VR seat (2.5 m) faces are small. In tutorial close-ups they show.
- Clean frontal crops are rare in a close embrace. The 1-phone follow has no input crop within 40 deg of frontal
  (nearest 42 deg) (m, `renders/lhmpp_1phone/manifest.json`).

### 5.4 Two dancers in contact

- The splat renderer draws both dancers in one pass and sorts them together. They hide each other correctly on all
  sheets (by eye). In Unity this needs one global sort across the two models (the renderer's `GaussiansGroup`, r).
- The partner's arm inside an input crop left white patches on the avatar. A new mask test dropped 13 of the 16
  earlier crops (m, `lhmpp/prep.log`). The lead's shoulder patch (black shirt) is gone (by eye,
  `lhmpp/avatars_cano_before_after.jpg`). That image is on a white background, so patches on her white top cannot be
  judged there. The reddish-brown smudges on her top (section 2) may be the same problem.
- Interpenetration comes from the fit, not the renderer (reference fit 31 mm penetration depth, MONO4D 10.6) (m).

### 5.5 Lighting and colour

- Every option has the studio light baked in. None relights. The Unity avatar keeps the baked studio light and adds
  35 % scene shading (section 1.1). The offline avatar render judged here has no scene shading.
- So lighting does not separate the options much. A lit VR room would need relighting for all of them (not planned).
- LHM++ colour is view-independent (SH degree 0; the PLY has only `f_dc`) (m).
- LHM++ 8-phone mixes the phones' different white balance: darker and more saturated (by eye). On phones 02 and 03
  its hair and skin match the photos better than 1-phone (by eye). Any colour fix (step A3) must be judged against
  the photos.

### 5.6 Licence

- LHM++ weights: CC BY-NC 4.0 (checked `LICENSE_WEIGHT`, 2026-10-19). Code Apache-2.0, with EG3D-derived parts under
  NVIDIA's non-commercial licence. Fine for this project's non-commercial research.
- The avatar path also uses SMPL-X (non-commercial). So both paths are non-commercial; licence does not separate them.
- Renderer for Unity (ninjamode): MIT. gsplat: Apache-2.0.
- Splats, renders and videos are likeness data: `work/` and git-ignored StreamingAssets only, never `head-movement`.

---

## 6. Quest 3 standalone: can skinned splats run?

### 6.1 What we checked (2026-10-19)

| claim | source | status |
|---|---|---|
| ninjamode/Unity-VR-Gaussian-Splatting "Works on desktop as well as Quest 3". No fps or Gaussian-count figures. | README, v0.4.0, last push 2026-10-07 | (r) |
| Needs Unity 6000.3+, D3D12 / Metal / Vulkan, URP with Render Graph; no OpenGL | README, `package.json` | (r). Ours: Unity 6000.6.3f1, newer than the declared minimum 6000.3, with no report of a run on it (r, review); Android graphics APIs Vulkan, then GLES3 as a fallback, not automatic; OpenXR single-pass instanced (m, ProjectSettings and OpenXR settings) |
| "Quest 3 gets 72 fps stable until around 400k Gaussians with proper settings." | The repo's `old` branch and tag `1.0`, `Readme.md` (read 2026-10-19) | (r). That is the retired codebase: AMD FFX sort, centre-eye sorting, adjustable sort frequency, static CT-scan data. It does not transfer to moving dancers sorted every frame, and v0.4.0 sorts differently. DANCER_SPLATS_RESEARCH section 8 said "AMD FFX sort" and "under the reported Quest budget of about 400k"; both lines now point here (2026-10-19). |
| v0.4.0 sort: DeviceRadixSort, 8-bit LSD, 4 passes for 32-bit keys; single-pass stereo shares one centre-eye sort | `Runtime/Core/GpuSorting.cs`, `Camera.cs` | (r, read by the review; not re-read) |
| Recommended XR settings: "Instanced Rendering", "SRP Foveation", "Symmetric Projection & Multiview Render Regions Optimization". Keep the subsampled layout off "if it causes black regions, particularly with Symmetric Projection on Quest 3". | `Documentation~/3DGS.md` (read 2026-10-19) | (r). **Ours differ**: every OpenXR target has `m_foveatedRenderingApi: 0` (legacy), `m_symmetricProjection: 0`, `m_optimizeMultiviewRenderRegions: 0` (m, `Assets/XR/Settings/Open XR Package Settings.asset`); `VrQuality.cs` sets the legacy `foveatedRenderingLevel` |
| Deformation hook `IGaussianSplat3DDeformation.TryGetFrame(canonical, out frame)`. The frame is an **FP32** buffer in the renderer's layout: 48 B per splat (position/opacity, scale, rotation) plus SH in FP32 (12 B per coefficient) or FP16 (8 B for SH0). Our provider allocates and owns it ("The creator owns both buffers"); the renderer borrows it until the provider prepares a new frame. "Position changes invalidate sorting." | `Runtime/3DGS/GaussianSplat3DData.cs` on main (read 2026-10-19) | (r) |
| A renderer needs an editor-created `GaussianSplat3DAsset` with a matching splat count, and culls with that asset's bounds. Public `InitializeFloat`, `SetAssetFiles`, `SetRenderBounds` exist. | `GaussianSplat3DAsset.cs`, `GaussianSplat3DRenderer.Source.cs` | (r, read by the review; not re-read) |
| Maturity: v0.4.0 rewritten 2026-09-17 to 2026-10-07 (22 commits); README: "research project ... No support"; `GaussiansGroup` added 2026-10-05; the splat shader carries a DXC stereo workaround specific to 6000.3. The only Quest note on deformation: "ONNX seems a lot faster on Quest 3", no numbers. | repo history, README, `Documentation~/4DGS.md` | (r, review) |
| Other public data: 500k splats run at about 12 fps in Unity (aras-p plugin) on Quest 3; 2M crash | World Labs Marble Unity export docs | (r, review; not re-read) |
| SqueezeMe: 3 Gaussian full-body avatars at 72 fps on Quest 3, own Vulkan pipeline (compute stages, per-primitive frustum culling, one sort for both eyes, hardware raster) | arXiv 2412.15171 v4 (SIGGRAPH 2025) | (r) |
| SqueezeMe: 60,381 Gaussians per avatar, given only in the training section. No frame-time breakdown, no per-eye resolution, no avatar screen size, no Gaussian sizes. The only Quest timing is the corrective decoder (5.0 ms, then 0.45 ms after sharing). Trained from a 512-camera dome; no code. | Paper, project page | (r). It cannot anchor a Unity budget for our Gaussians. |
| Our lesson scene on the Quest 3 | `work/larissa_kadu_recap/NOTES_X3.md`, `NOTES_X4.md` | (m, device logs): recap at 72.0 fps, GPU 8.1-10 ms of 13.9 ms, hair on; menu alone GPU 6.7 ms. A later recap session stepped the hair down ("47 percent of frames over budget"); NOTES_X4 blames load transients, unverified. Refresh rates 72 / 80 / 90 / 120 Hz are offered. |
| Our render settings on the Quest 3 | `VrQuality.cs`, URP asset | (m): MSAA 4x (hair alpha-to-coverage needs it), HDR off, eye scale 1.0, 72 Hz |
| Default display mode | VIEWER_SPEC 3.2 (owner, 2026-10-07) | Avatars **65 % transparent as displayed**, depth-correct. The splat layer (3.2a) shows the dancers only and is off by default. |

### 6.2 Budget: GPU time, not Gaussian count

Count does not predict the cost. Old ninjamode: 400k static Gaussians at 72 fps; World Labs: 500k at about 12 fps
(both r). What costs time is **fill**: how many screen pixels each Gaussian's quad covers, blended for two eyes.

Our Gaussians (m, `lhmpp/pod_out/gs/*_cano.ply`):
- 160,000 per dancer, 320,000 for the couple.
- Median sigma per axis (smallest / middle / largest): 2.7-3.9 / 4.1-5.8 / 7.2-9.4 mm over the four avatars.
- The largest axis is capped at 5 cm. 28-46 % of Gaussians have a largest sigma of 1 cm or more.
- 16-27 % have opacity below 0.05.

| item | GPU time per frame | tag |
|---|---|---|
| Frame at 72 Hz / 90 Hz | 13.9 / 11.1 ms | |
| Today's lesson scene (avatars + hair cards + overlays) | 8.1-10 ms | (m, NOTES_X3; GPU clock level not logged) |
| Menu alone | 6.7 ms | (m, NOTES_X4; GPU clock level not logged) |
| **Room for the splat layer**, if it replaces the avatars and hair | **about 4-7 ms at 72 Hz; about 1-4 ms at 90 Hz** | (e; at the unknown clock level of those runs) |
| LBS writing FP32 frames (about 90 B of traffic per Gaussian, about 29 MB per frame) | 0.5-1 ms | (e) |
| Stereo view data + a 4-pass radix sort of 320k keys, every frame (posed positions invalidate the sort, r) | 1-3 ms | (e: aras-p's desktop figures, 6.1M splats sort 1.1 ms + view 0.8 ms on a 3080 Ti, scaled 11-20x) |
| Translucent composite for the 65 % mode (offscreen per eye, then composite) | 1-2 ms | (e) |
| **Fill at the VR seat (2.5 m)**: about 80-100 M quad fragments per stereo frame | **8-20 ms** | (e, review estimate from our PLYs: real rotations, opacity-aware quad bounds, about 13 px/deg; assumes the Quest 3 blends 5-10 G fragments/s, no source found; 18 px/deg doubles it) |
| Fill at 1.5 m / 1.0 m | about 3x / 6x the 2.5 m figure | (e) |

How the room is estimated (e): 13.9 ms minus the menu scene's 6.7 ms leaves about 7 ms; the overlays and skeletons
stay and a safety margin is needed, so 4-7 ms. At 90 Hz: 11.1 - 6.7 = 4.4 ms, so 1-4 ms. B1 measures it.

The room is not a fixed number. The Quest picks its CPU and GPU clock levels itself. The project sets no
performance-level hint (`VrQuality.cs`, `VrPerf.cs`), and `HM-PERF` does not log the level. The only logged level,
CPU/GPU 2/2, is from a Quest 2 menu run (NOTES_X2), so it says nothing about the Quest 3 numbers above. A GPU time
taken at a lower clock is longer than at the top clock. B1 must log or pin the level (section 7).

What follows (e):
- The non-fill costs alone use most of the room: LBS plus sort and view data about 1.5-4 ms; with the 65 % composite
  (the default mode) about 2.5-6 ms.
- Fill at the seat is probably larger than the whole room. It grows as the viewer walks closer.
- MSAA 4x stays for the hair. If splats draw into the MSAA target, blending work grows.
- **A pass at the full 2 x 160k is unlikely.** Likely outcomes: heavy reduction (opacity cull, smaller maximum
  scale, LOD by distance), distillation to smaller Gaussians (new work), or PC-tethered VR (decision 10).
- Count pruning alone barely helps if the large Gaussians survive.

### 6.3 What our Unity project would need

1. The renderer package (MIT), first on a test branch. Splat data is created at run time from pushed lesson data
   (like `deploy_data.ps1` does today), never built into the APK (likeness data).
2. A **deformation provider** (`IGaussianSplat3DDeformation`): a compute pass that reads the 55 bone matrices
   `SmplxAvatar` already computes and writes an FP32 frame in the renderer's layout (48 B per Gaussian, plus SH0 at
   12 B in FP32 or 8 B in FP16) every frame. The provider allocates and owns this buffer; the renderer borrows it.
   The project has no compute shaders of its own yet.
3. **Runtime assets and bounds**: create each `GaussianSplat3DAsset` at run time (`InitializeFloat` / `SetAssetFiles`)
   and call `SetRenderBounds` every frame from the posed dancers. Otherwise moving dancers are culled with A-pose
   bounds.
4. Both dancers in one `GaussiansGroup`, so arms and hands in contact sort correctly.
5. **GLES3 fallback**: the renderer has no OpenGL path. If the device falls back to GLES3, splats vanish silently.
   Detect it and show the avatars.
6. **XR settings**: the renderer recommends settings we do not use (6.1). Changing them is a persistent
   project-configuration change. It can affect passthrough and the hair shaders. It needs the owner's approval
   (decision 5). B1 measures with our current settings first.
7. Viewer features: the 65 % display mode. Scaling every splat's alpha would show the inside and back of each body.
   Doing it right needs the renderer's composite path (accumulate offscreen per eye, then composite): extra render
   targets and a pass on a tiled GPU, untested and uncosted (1-2 ms, e). "65 % transparent as displayed" (VIEWER_SPEC
   3.2) also needs a per-pixel alpha from the shaded colour's luminance, like the avatars' `DisplayTransparency.cs`
   AlphaLut, applied to the composited splat image. All realism tests here were opaque. Also:
   skeleton overlays need splat depth; the layer toggle. The existing `SplatCloud` disc preview is not enough: isotropic
   discs, unsorted, blurry.
8. Hair Gaussians driven by the `HairStrands` guides (step A4). `HairStrands` simulates 160 guides x 16 segments on
   the CPU (Burst), so uploading guide positions is cheap.

No one has reported this combination on a Quest: group sort + external deformation + single-pass multiview +
MSAA 4x + passthrough compositing of splat alpha.

### 6.4 Data size per take

| item | size |
|---|---|
| LHM++ canonical PLY (float32: centre, normal, colour, opacity, scale, rotation) | 10.9 MB per dancer = 68 B x 160,000 (m) |
| + skinning, 4 bones (8 B per Gaussian) | 12.2 MB per dancer, 24 MB per couple (e) |
| Stored on the Quest, quantised (about 32 B: fp16 centre, rotation and scale, RGBA8, skin 8 B) | about 5 MB per dancer, 10 MB per couple (e). **Storage only.** |
| GPU frame buffer at run time (FP32 in the renderer's layout, owned by our provider: 48 B + SH0 12 B in FP32 or 8 B in FP16) | 9.6 MB per dancer with FP32 SH, 9.0 MB with FP16 SH; positions and rotations rewritten every frame (e, from the r layout) |
| Per-frame input | The SMPL-X pose stream the viewer already plays, under 1 KB per dancer per frame (e) |
| For comparison: per-frame splat sequence | about 860 MB for 3.5 s at 120k Gaussians (e, DANCER_SPLATS_RESEARCH 8) |
| Reuse | Planned: one avatar per person and outfit, reused by every take in that outfit. **Untested** (section 3.4). |

---

## 7. Recommendation and next steps

### (a) Tutorial videos: skinned LHM++ splats, posed by the MAMMA fit

| step | what | work (e) | cost (e) |
|---|---|---|---|
| A1 | On the pod, also export each Gaussian's neutral centre, SMPL-X skinning weights and the neutral-pose transform (extend `remote/cand_lhmpp/lhmpp_compare_pod.py`). Record what `animate_gs_model` does beyond plain LBS. Also export a few posed PLYs for B1, including contact. Needed by A2, B1 and B2. | 0.5-1 day | one pod run, about $0.15 (this round's LHM++ session: $0.13, m) |
| A2 | Local clip renderer: our LBS + gsplat on the shared 2080 Ti (under `work/gpu.lock`), any time window and camera path, 1080p. **Gate:** it reproduces the pod renders on the 18 views (side-by-side sheet + pixel difference). | 1-2 days | $0 |
| A3 | Inputs: 1-phone crops by default. Only for 8-phone inputs: per-phone colour normalisation, judged against the photos. | optional, 0.5 day | about $0.15 pod |
| A4 | Hair motion: bind her hair Gaussians to the 160 Unity guide strands; record guide tracks from the Unity simulation for offline renders. Check the three fast turns. | 2-4 days | $0 |
| A5 | A 20-30 s test clip, mostly **outside** 58-62 s (VR seat + one tutorial camera). It tests reuse, rigid hair and frozen folds. The owner looks at it. | 0.5 day | $0 |
| O1 | Optional, owner: capture the Unity avatar at the VR seat for one instant; we add it to a sheet next to LHM++. | 15 min | $0 |

Total: about 1.5 weeks of work and under $1 of pods (e).

### (b) Quest 3: a device test first, fill before features

| step | what | work (e) | cost |
|---|---|---|---|
| B1 | **GPU sweep in the real lesson scene** (test branch, not main). **Needs A1**: the canonical PLYs have no skinning data, so they cannot be posed. Data: `lhmpp/pod_out/gs/scene_lhmpp_{1,8}phone_tt_i1.ply` (both dancers posed at i1, 59.185 s; the only posed splats on disk) plus a few posed PLYs that A1 exports, including contact. (1) Measure the **room**: the lesson scene with avatars and hair cards off, same settings. (2) Measure the **splat layer's own GPU ms** (layer on minus layer off) at 1.0, 1.5 and 2.5 m. Log the CPU/GPU clock level with every number, or pin it with an XR performance-level hint; report the room and the splat ms at the same level. Variants: full 2 x 160k; opacity cull (< 0.05); maximum scale capped (2 cm, 1 cm); pruning / LOD to 2 x 80k and 2 x 40k; MSAA 4x vs 2x; opaque vs the 65 % composite. Current XR settings; the renderer's recommended settings only with the owner's approval. | 2 days | $0 (A1's pod run is counted under A1) |
| B2 | Only if B1 comes near the room: the LBS deformation provider, runtime assets, per-frame bounds, one `GaussiansGroup` (uses A1's data) | 3-5 days | $0 |
| B3 | Play the larissa_kadu_38 clip on the headset in passthrough. **Gate: the splat layer's own GPU time, with the real scene, fits the room B1 measured (both at the same clock level; estimate 4-7 ms at 72 Hz; plan for 4 ms), and 72 fps holds.** The owner looks at it opaque and at 65 % transparency. | 0.5 day | $0 |
| B4 | Pass: reopen the 2026-10-05 headset decision and build the splat layer (with A4 hair on device). Fail: keep the avatar. Options then: distil to smaller Gaussians (new work) or PC-tethered VR (decision 10). | - | - |

Total: about 1.5 weeks, $0 beyond A1. B1 needs A1 first. B1 alone answers most of the question; stop after B1 if
fill is far over.

### Not recommended

- No more AGS training.
- No Unity work on the hybrid mesh. Optional 15-minute look: copy `compare/hybrid/<role>_albedo.png` over the
  shipped albedos to see it with the hair cards in Unity.
- No per-frame 4D splats from phones.

---

## 8. What the owner must decide

1. **Tutorial videos:** switch from the avatar to skinned LHM++ splats? Recommended: yes, if A2 reproduces the look with
   our own LBS and the A5 clip holds up outside the input window.
2. **Fair baseline (optional):** capture the Unity avatar at the VR seat for one instant (O1)?
3. **Inputs:** 1-phone crops (recommended; matches the one-phone goal) or 8-phone crops? They tie on this evidence:
   18 vs 18 asks (6 vs 6 views) on phones 02/06, and 8-phone's phone-03 lead comes from phone-03 crops in its own
   input. 8-phone is optional where a take has it.
4. **Headset test:** approve B1 (2 days, after A1) first, and B2-B3 only if B1 comes near the room? The gate is the
   splat layer's own GPU time with the real scene (estimate 4-7 ms at 72 Hz), not 13.9 ms for the whole frame.
5. **XR settings:** may the device test try the renderer's recommended XR settings (SRP foveation, symmetric
   projection, multiview render regions)? It is a persistent project change and can affect passthrough and the hair
   shaders. Default without approval: measure with the current settings only.
6. **Display mode in VR:** must splats show at 65 % transparency like the avatars (composite path plus the per-pixel
   AlphaLut alpha, extra GPU time, untested), or is an opaque splat mode acceptable?
7. **Hair:** may the first tutorial clips have rigid splat hair, or must the hair swing first (A4)?
8. **Faces:** accept LHM++'s generic faces for now, or fund a face step? (2026-10-07: "no further investment in
   faces"; the 2026-10-08 handoff lists face shape as a priority.)
9. **Publishing:** will tutorial videos leave the research group? LHM++ (CC BY-NC) and SMPL-X allow only
   non-commercial use. Confirm the likeness consent covers published videos.
10. **If the Quest test fails:** keep the avatar on the headset, or show splats in PC-tethered VR (that would change
    the "Quest standalone" decision)?

---

## 9. Sources

Local (m):
- `work/larissa_kadu_38/mono4d/compare/`: `compose.json`, `metrics.json`, `metrics.md`, `lpips_all.json`,
  `lpips_all.md`, `jev.json`, `views.json`, `README.md`, `renders/<method>/manifest.json`, `lhmpp/report.json`,
  `lhmpp/prep.log`, `lhmpp/pod_out/lhmpp_compare_pod.json`, `lhmpp/pod_out/gs/*.ply`, `hybrid/manifest.json`,
  `hybrid/scores.json`.
- Code: `dancecap/mono4d/appearance_compare.py`, `appearance_compose.py`, `appearance_jev.py`, `appearance_hybrid.py`,
  `appearance_avatar_ags.py`, `appearance_lpips.py`, `cand_lhmpp.py`; `remote/cand_lhmpp/lhmpp_compare_pod.py`,
  `remote/cand_lhmpp/pod_job_lpips.json`.
- Device logs: `work/larissa_kadu_recap/NOTES_X3.md`, `NOTES_X4.md` (Quest 3); `NOTES_X2.md` (Quest 2, clock level).
- Unity (read only): `head-movement/Assets/Avatar/SmplxAvatar.cs`, `Assets/Resources/HM_AvatarLit.shader`,
  `Assets/Avatar/DisplayTransparency.cs`, `Assets/Hair/HairStrands.cs`, `Assets/Splats/SplatCloud.cs`,
  `Assets/VR/Runtime/VrQuality.cs`, `Assets/VR/Runtime/VrPerf.cs`, `Assets/XR/Settings/Open XR Package Settings.asset`,
  `Assets/Bloom.asset`, `ProjectSettings/ProjectSettings.asset`, `docs/VIEWER_SPEC.md` 3.2 and 3.2a.
- Docs: [DANCER_SPLATS_RESEARCH.md](DANCER_SPLATS_RESEARCH.md) sections 1, 8, 12;
  [MONO4D_NOVEL_VIEWS.md](MONO4D_NOVEL_VIEWS.md) sections 10.1 and 10.6.

Web (r, read 2026-10-19 unless marked "review"):
- ninjamode/Unity-VR-Gaussian-Splatting: [repository](https://github.com/ninjamode/Unity-VR-Gaussian-Splatting)
  (README, `CHANGELOG.md`, `package.json` v0.4.0, `Documentation~/Benchmark.md`, `Documentation~/3DGS.md`,
  `Runtime/3DGS/GaussianSplat3DData.cs` on main, re-read 2026-10-19 for buffer ownership; the `old` branch / tag `1.0`
  `Readme.md`). Review only: `GpuSorting.cs`,
  `Camera.cs`, `GaussianSplat3DAsset.cs`, commit history, `Documentation~/4DGS.md`.
- SqueezeMe: [arXiv 2412.15171](https://arxiv.org/abs/2412.15171), [full text v4](https://arxiv.org/html/2412.15171v4),
  [project page](https://forresti.github.io/squeezeme/).
- Review only: World Labs Marble Unity export docs; aras-p/UnityGaussianSplatting README (desktop timings).
- LHM++ weights licence: [LICENSE_WEIGHT](https://raw.githubusercontent.com/aigc3d/LHM-plusplus/main/LICENSE_WEIGHT).

---

## 10. Revision note (2026-10-19, after two reviews)

Both reviews agreed with the core direction: skinned LHM++ splats over the current avatar for PC video; drop AGS and
the hybrid mesh; Quest only after a device test. They found secondary claims wrong or overstated. Changed:
- The avatar baseline is now stated up front (1.1): offline render, not the shipped Unity look at the same instant.
- "8-phone crops (Jev's pick)" is gone: on phones 02/06 the two LHM++ variants split 18 vs 18; 8-phone's phone-03 lead
  comes from phone-03 crops in its input. Recommendation: 1-phone crops.
- Jev: n = 18 views (3 reshuffled asks each), means labelled as means, the near-trivial real-photo control dropped,
  "same people" no longer read as likeness, mean confidence reported.
- Pixel metrics: LPIPS added for all six methods (one pod run, $0.0135); the AGS explanation corrected; no pixel
  metric gives LHM++ a clear lead, so the ranking rests on Jev plus the sheets.
- Popping number demoted to a weak proxy. Near-training frames and untested reuse stated.
- Posing: the LHM++ look came from LHM++'s own code on the pod; our LBS path is planned, not reproduced.
- Quest: budget rewritten around fill and the splat layer's own GPU time, using measured headroom; the 400k claim cited
  from its primary source with why it does not transfer; XR settings and renderer integration specifics added;
  confidence lowered to low; B1 redesigned; the B3 gate is now the splat layer's own GPU time.

Second check (2026-10-19, two checkers). Changed:
- 1.1 and 5.5: the Unity avatar is **not** unlit. `HM_AvatarLit.shader` shades the albedo with the main light (wrap
  0.5) and SH ambient at 35 % (`_LightInfluence` 0.35). The earlier "no lighting on top of the albedo (m, manifest)"
  came from the manifest's own text and the Unity sheet caption, not from a measurement. Both texts are corrected in
  the code (`appearance_avatar_ags.py`, `appearance_compose.py`) and in `renders/avatar/manifest.json`; the Unity sheet
  was redrawn.
- 3.1: LHM++ is level with or within 0.01 of the real cut-out in 6 of 18 views, not 3.
- Time windows: renders 58.0-61.5 s, LHM++ inputs 58.1-61.9 s. LPIPS version: pinned in the job, not recorded.
- Section 6: the frame buffer is ours (the renderer borrows it), SH can be FP16; GPU times carry no clock level; B1
  now measures the room, logs or pins the clock level and needs A1's posed PLYs; the 65 % mode needs the per-pixel
  AlphaLut too; non-fill costs 1.5-4 ms, 2.5-6 ms with the composite. DANCER_SPLATS_RESEARCH section 8 corrected.
- One checker said the GPU times were taken at clock level 2 (NOTES_X2). Not used: that log is from a Quest 2 menu
  run; the Quest 3 runs did not log the level.
