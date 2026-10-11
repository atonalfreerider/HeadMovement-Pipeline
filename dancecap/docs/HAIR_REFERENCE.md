# Hair reference: the follow's hair, acceptance criteria for the Unity hair

> **Local only.** Sections 2-4 hold body and hair measurements of a consenting dancer (non-commercial demo). This
> file lives in the workspace repo (no remote); strip sections 2-4 before this doc goes anywhere public. The numbers
> are also in `work/<take>/avatar/hair_reference.json` (git-ignored), which is the source of truth.

Target for the hair review and rebuild of the follow avatar in the demo: match her real hair in **shape and colour**
and **keep the original glowing tip bloom** of the head-movement hair. **Update 2026-10-06 (user):** "make her hair
sleek and straight like the portrait ... realistic, performant hair but add a little stylization with the tips of the
hair strands": the texture and styling now follow the PORTRAIT (sleek, straight, glossy, deep side part swept to her
right), the length and motion still follow the video, and the tip bloom is kept as a small stylised accent. **Review
fixes 2026-10-07:** the default colour is now the portrait's crimson (the video's brick red one switch away), the
scalp cap reaches the hairline painted into her texture, the tip glow is a soft bloom on real strand ends, the hair
fades out with the body, and the right front no longer hooks under the chin in fast turns. Section 6 is the current
acceptance list; section 8 describes the Unity implementation (groom version 2). Every number comes from
`dancecap/dancecap/hair_reference.py` (version 2, 2026-10-06) on the take `larissa_kadu`: full data in
`work/larissa_kadu/avatar/hair_reference.json` (`acceptance`, `unity_notes`), reference sheet
`work/larissa_kadu/review/hair_reference.jpg`, 3D check `work/larissa_kadu/review/hair_reference_hull.jpg`.

Units: **world** = metres of the current fit's world (the larissa_kadu world is about 4 % too large; the follow's
SMPL-X avatar is 1.747 m tall in it). **true** = world x 0.959 (stature 1.651 m + 2.5 cm sole over the avatar's rest
height). **rest height** = height above the sole on the avatar's REST (T-pose) mesh, so it applies to any pose.
Prefer the fractions of avatar height: they carry over to a rescaled avatar (larissa_kadu_38).

## 1. Target look in one paragraph

Long, layered, red hair. Seen from behind, the tips fall **between the lower tips of the shoulder blades and the
mid-back**, about **0.52 m (true) below the crown** (31 % of her height); a back strand runs about **0.60 m (true)**
from the crown over the skull and down the back. **Styled like the portrait (user, 2026-10-06): sleek, straight and
glossy**, lying flat on the skull, a deep side part on her left, the hair swept over to her right **and pulled back**
(user, 2026-10-07: "her hair part needs to be pulled back more"): the part starts a little behind the hairline, the
right front section sweeps over the temple and ear to **behind** the right shoulder (until 2026-10-07 it fell over the
right temple and down in front of the right shoulder), the face clear, the left front tucked behind the ear;
tapered, slightly piecey ends; no fringe. (The demo video shows it fuller with soft loose waves; that texture is no
longer the target.) Front sections reach at least the collarbone (longer in the portrait). The silhouette is widest around the neck and shoulders (about 0.6 x her shoulder width) and tapers to a
narrow tip end. Colour: an even, saturated **brick / scarlet red** from root to tip (no darker regrowth), deep
red-brown in the shadows, a light coral-red sheen in the highlights. In calm moments the hair mass sits over her right
shoulder (about 32 deg off straight behind). In turns it swings out to about 0.24 m (true) from the neck axis (an upper
bound). **How far it trails the head is not resolved in this window**: leave that to the simulation. The strand tips
keep the original stylised **white bloom**: only the last quarter of every strand glows.

## 2. Shape (calm frames: 89 of 176, head yaw < 90 deg/s and head speed < 0.6 m/s)

| Quantity | Value | Notes |
|---|---|---|
| Back tip level, rest height | **1.200 m world** (0.687 of avatar height) | mean of 3D hull 1.215 and single-view ray-cast 1.186; longest quartile 1.170 |
| Tip drop below the crown | **0.547 m world = 0.525 m true** (31.3 % of height) | longest quartile 0.577 world / 0.554 true |
| Where the tips fall | between the lower tips of the shoulder blades (rest 1.259) and the mid-back (1.160), 59 % of the way down | longest quartile: just above the mid-back |
| Posed tip level in calm frames | 0.488 m world below the posed crown | about 6 cm higher than the rest-mapped level: posture (head tilt, curved back) lifts the posed hair; the review sheet draws both lines |
| Back strand length, crown to tip | **0.622 m world = 0.597 m true** (0.356 x avatar height) | posterior-envelope arc length over the skull and back, +1 cm hair layer; range 0.585-0.619 true |
| Front sections, tip level | rest 1.342 m world **or lower** | **lower bound** of the front length: the lead and her arms hide the front; the portrait shows the front reaching the armpit / bust line |
| Width at ear level | 0.239 m world (1.31 x the avatar's head width 0.182) | upper bound |
| Width at shoulder level | 0.280 m world (0.61 x the avatar's shoulder width 0.462) | upper bound |
| Widest | 0.284 m world, 0.30-0.35 m below the crown | upper bound |
| Width near the tips | 0.152 m at 0.50-0.55 m, 0.114 m at 0.55-0.60 m below the crown | upper bound |
| Hair layer stand-off from her back | 0.056 m world (p90), almost constant from the nape to the tips | upper bound (includes up to 1.5 cm of mask dilation) |

Width profile (world m, lateral p3-p97 extent per 5 cm band below the crown, calm median):
0.18, 0.22, 0.24, 0.26, 0.26, 0.28, 0.28, 0.27, 0.25, 0.22, 0.15, 0.11 (bands starting at 0.00, 0.05, ... 0.55 m).

Qualitative (by eye; `work/larissa_kadu/hair_ref/observations.json`): parting = deep side part on her left in the
portrait (sweeps over to her right), not resolvable in the video; no fringe; long layers, shorter at the front and
sides, feathered ends; texture = sleek and straight in the portrait, soft waves and more volume in the video. The
calm-frame hair mass sitting to her right (section 3) agrees with the hair swept to her right.

**Look differences, handled honestly:** the portrait and the demo differ. Portrait: straight, smoothed, deeper and
cooler crimson red (studio colour grade, magenta cast on everything). Video: wavy, fuller, warmer brick red. The
portrait date is unknown. **The demo hair follows the portrait for texture and styling** (sleek, straight, side-swept:
the user's request of 2026-10-06) and **the video for length, motion and colour** (the portrait's crimson is a studio
grade; `hm_hair --action set --colour portrait` switches the albedo to the portrait's for comparison).

## 3. Motion (turn frames: 26, head yaw rate > 180 deg/s; 12 one way, 14 the other)

| Quantity | Value |
|---|---|
| Spread radius r95 about the neck's vertical axis, calm | 0.195 m world (0.187 true) |
| Spread radius r95, turns | median 0.253 m world (0.243 true), p90 0.310 m (0.297 true), max 0.328 m: **upper bounds** (turn-frame hulls hold 1.43 x the calm voxels, partly motion blur) |
| Hair mass in calm frames | 32 deg toward her right of straight behind the head (37 deg vs the torso), IQR -39..+19 deg |
| Trailing angle in turns | **not resolved**: raw median 20 deg, but that is the calm offset; offset-corrected -4 deg vs the head (bootstrap 95 % CI -12..+44), +4 deg vs the torso (CI -12..+16); the two turn directions disagree (+58 vs -12 deg vs the head) |
| Lag time | **not resolved** (phase-alignment residual 43 deg RMS) |
| Head yaw rate | median 64 deg/s, p90 205 deg/s, max 292 deg/s |
| Correlation spread vs abs(yaw rate) | 0.49 |

Why the trailing angle is unresolved: the hair-mass centroid lies only about 6.5 cm from the neck axis, so 1 cm of
centroid noise moves its azimuth by about 9 deg; and the hair lies over one shoulder even when she is calm. The
measurement is kept in the JSON (`shape.motion.trailing`, with the rule it failed) and re-run on the 38 s take, which
has many more turns in both directions. It is not an acceptance criterion until it resolves.

## 4. Colour (OBSERVED, white-balanced; not an albedo)

Video: hair pixels = dancers-stage hair mask AND MediaPipe SelfieMulticlass hair (156 views over 8 cameras, at most
30 per camera), white-balanced per view on her white top scaled to an assumed top reflectance of 0.80; zones by
distance from the head centre (root <= 0.11 m, tips = outer 20 % of the visible length); median across cameras.
**These colours include the hair's own shading and self-shadowing.** They describe the image a render must reproduce
under the demo lighting. Used as the albedo of a lit shader, the hair would be darkened a second time.

| Zone / statistic | Hex (sRGB) | Linear sRGB | CIELAB L*, a*, b* |
|---|---|---|---|
| Image p10 (by luminance) | #4D160E | 0.075, 0.008, 0.004 | 16.5, 25.3, 19.2 |
| **Image p50** | **#732C24** | **0.171, 0.025, 0.018** | 28.3, 30.7, 20.9 |
| Image p90 | #B54D43 | 0.465, 0.074, 0.056 | 46.4, 41.7, 27.3 |
| Root | #772F27 | 0.185, 0.028, 0.021 | 29.6, 31.2, 20.7 |
| Mid | #712B22 | 0.166, 0.024, 0.016 | 27.8, 30.5, 21.3 |
| Tip | #732F26 | 0.171, 0.028, 0.019 | 28.9, 29.3, 20.7 |
| Highlight (p90-99 band) | #CC574E | 0.601, 0.095, 0.076 | 52.0, 45.7, 28.4 |
| Shadow (p5-15 band) | #4D170D | 0.075, 0.008, 0.004 | 16.6, 25.1, 19.8 |

- Root to tip is **uniform**: dE76 root-mid 2.1, tip-mid 1.8; roots read slightly brighter (top light on the crown),
  not darker: **no dark regrowth**.
- Across the 8 phones the hue is consistent (mid hue 343-10 deg, SD 8 deg; cameras 04 and 05 lean magenta even after
  white balance), the **lightness is not**: mid L* 16-44 (SD 9.1), each camera dE76 6.5-24 from the consensus. The
  tolerances below follow from that spread.
- Portrait (white-balanced on her white top the same way): root #570C15, mid #63131F, tip #480A11, highlight #832937,
  shadow #37070B: hue 351-353 deg (crimson), L* 13-21, darker and cooler than the video (dE76 11-17). Its darker root
  zone is shadowed by her raised arm, not a dye difference.
- **Render default (2026-10-07): the portrait.** The user asked for hair "like the portrait"; the groom's
  `render.albedo_default = "portrait"` (albedo = portrait mid x 1.85, TRT highlight = the portrait highlight's hue at
  the video highlight's brightness). The video's brick red stays available (`hm_hair --action set --colour video`;
  set `albedo_default` to `video` in hair_groom.py to make it the default again). Under the demo lighting the
  portrait default renders, white-balanced on her top, p10 / p50 / p90 #470D14 / #631920 / #94544F against the
  portrait's shadow / mid / highlight #37070B / #63131F / #832937 (mid within dE76 3; the p90 highlight is paler
  than the portrait's, its Kajiya-Kay white primary lobe). The video targets below apply to `--colour video`
  (#562019 / #743026 / #A35E51 on 2026-10-07).
- Compared with version 1 (497 views at every 2nd frame) the palette moved by dE76 0.7-1.5 with 156 views: the colour
  median had converged.

## 5. The original tip bloom (to keep)

Read from the head-movement repo (the user's original `Assets/HairSimulation.cs` at `a4e493b`, created in 961b058
"parametric hair and line renderers"; bloom set up in 91da08c "bloom") and verified live in the running editor
(Unity 6000.6.3f1, 2026-10-06). `tip_bloom_recipe()` re-parses it on every local run (a pod run uses the verified
fallback values, which are identical).

1. **Strands**: 180 per head (30 dome strands within 60 deg of the top + 5 ring layers x 30 over the next 60 deg),
   each one `LineRenderer` (world space, camera-facing ribbon) with **6 points**, width **10 mm at the root tapering
   to 1 mm at the tip**, length max(0.2, 0.16 (1 - cos alpha)) = 0.20-0.32 m (cardioid, longest at the back),
   skull radius 0.07 m. Material `BloomMat` (shader `Shader Graphs/bloom-shader`).
2. **Colour gradient** (authored, gamma): black at 0, dark grey 0.1 at 0.5, white at 0.95, HDR (4, 3.2, 1.5) at 1.0.
3. **What the GPU gets**: LineRenderer evaluates the gradient at its 6 vertices, converts gamma to linear (project in
   Linear colour space) and stores **UNorm8x4** vertex colours, so the HDR key is **clamped to white**: baked
   linear colours **0, 1/255, 2/255, 18/255, 115/255, 1.0** (measured with `LineRenderer.BakeMesh`; the
   gamma-to-linear 8-bit model gives 0, 1, 2, 19, 114, 255, within one step), interpolated linearly along each segment.
   The warm (4, 3.2, 1.5) key has never been visible.
4. **Shader** (`bloom-shader.shadergraph`): URP **Unlit**, opaque, ZWrite on, front faces, casts shadows;
   **Base Color = Vertex Color x 3.1** (a Float node; the HDR `_Color` property, 6.42 on BloomMat, is declared but not
   connected). Linear output along the strand: **0, 0.012, 0.024, 0.219, 1.398, 3.10** at s = 0, 0.2, ..., 1.0.
5. **Pipeline**: `Bloom.asset` (URP asset set in Graphics settings, no per-quality override): **HDR on**
   (R11G11B10), MSAA off, Forward renderer `Bloom_Renderer`; camera post-processing on, HDR on.
6. **Bloom volume**: the URP global default volume profile `DefaultVolumeProfile.asset` (no Volume in the scene):
   **Bloom threshold 1, intensity 1, scatter 0.7**, clamp 65472, tint white, high-quality filtering off, downscale
   half, max iterations 6; **Tonemapping None**.
7. **Result**: the strand is near-black at the root, the output crosses the bloom threshold at **s = 0.733** and
   reaches **3.1 (pure white)** at the tip: the **last ~27 % of every strand glows white**, with a soft halo from the
   bloom pyramid; on a 0.32 m strand about 8.5 cm glow. URP's bloom prefilter has a hard-coded **soft knee of
   0.5 x threshold**: anything above 0.5 linear (max channel) already blooms a little (0.6 -> 0.8 %, 0.7 -> 2.9 %,
   1.0 -> 12.5 % of the pixel), which is why the knee starts at s = 0.648 on the strand.

How to keep it on the photoreal hair (measured colour as the base, the tip emission added on top):

- Emission E(s) along each strand, s = normalised arc length from root (0) to tip (1), **white**, piecewise linear
  through **(0, 0), (0.2, 0.012), (0.4, 0.024), (0.6, 0.219), (0.8, 1.398), (1.0, 3.10)** linear HDR, added to the
  lit hair colour (the review sheet's "target" strip shows exactly this, continuous). With more points per strand
  (spec: 16), evaluate this 6-point curve, do NOT re-evaluate the authored gradient (that gives a different, sharper
  ramp).
- Do **not** feed the authored (4, 3.2, 1.5) key through a float vertex stream or a shader colour: it would turn the
  tips gold and 4x brighter than they have ever been. White x 3.1 is the look to keep.
- Keep the bloom settings (threshold 1, intensity 1, scatter 0.7, Tonemapping None) and HDR on; keep the **lit hair
  body <= 0.6 linear (max channel)** so the soft knee leaves it alone and only the tips bloom (the observed highlight
  #CC574E peaks at 0.60).
- Keep the 10 mm -> 1 mm taper at the tips (the glow is a thin bright end, not a blob).
- The glowing tips are a stylised trace of the hair's whip, not something in the footage. At the default avatar
  opacity the tips must still bloom (do not scale the emission by the hair opacity unless the user asks).
- Quest: URP HDR + bloom is the most expensive post effect on standalone; if the Quest tier cannot afford it, use the
  same emission curve with a cheaper bloom (fewer iterations / quarter resolution) and re-check this look.

## 6. Acceptance checklist

1. **Length**: back tips at rest height 1.200 +- 0.04 m on the current avatar (0.687 +- 0.023 of avatar height), i.e.
   between the lower tips of the shoulder blades and the mid-back; back strands 0.36 +- 0.02 x avatar height from
   the crown; front sections reach at least the collarbone (rest 1.34 m world, a lower bound).
2. **Silhouette** (back view, calm pose): widest around the neck-shoulder line, **0.21-0.24 m** world there (sleek: the
   lower end of the hull's upper bounds 0.21-0.28), tapering to 0.11-0.15 m near the tips; about 1.2 x head width at
   ear level; flat on the skull (about 1 cm hair layer on top); the hair layer stands 1.5-3.5 cm off her back.
3. **Texture and styling (the portrait)**: sleek, straight, glossy, **no waves**; no fringe; deep side part on her
   left, the top swept over to her right and **pulled back** (2026-10-07): the forehead / part hair combed back and
   flat over the crown, the part starting ~2 cm behind the hairline, the right front section over the temple and ear
   to **behind the right shoulder** (rest shape: tips at least 2 cm behind the right shoulder joint; no guide point
   beside or in front of the face; before: tips in front of the shoulder, 94 guide points beside the face); the left
   front behind the ear;
   one coherent sheet with a continuous highlight band (no crimped stripes); tapered, slightly piecey ends. **No dark
   band under the hairline**: the scalp cap reaches the hairline painted into the follow's texture
   (`hair_groom.painted_hairline`: el 28 deg at the front, 20-22 deg at the temples on 02_LarissaKadu; the root
   hairline is 45-50 deg) with a soft, fringed edge, and no card starts with a hard edge (card roots fade in).
   In fast turns the right front swings as a sheet: its tips never turn up into a hook under the chin.
4. **Motion**: in turns faster than 180 deg/s the hair spreads beyond its calm r95 (about 0.19 m world) but not
   beyond 0.25 m world (upper bound). **No trailing-angle or lag criterion** (not resolved in this window).
5. **Colour, as image statistics under the demo lighting**: render the avatar under the demo lights, white-balance
   the render on her white top the same way, and take the hair pixels' luminance p10 / p50 / p90. Default (portrait,
   section 4): close to the portrait's shadow / mid / highlight **#37070B / #63131F / #832937**. With `--colour
   video` they must match **#4D160E / #732C24 / #B54D43** with hue within 10 deg, HSV saturation within 0.16 and lightness within 14 L*
   (tolerances = 1.5 x the measured cross-camera spread, lightness clamped to 10-14). Root, mid and tip medians within
   dE76 6 of each other (no darker roots). Start the albedo lighter than the observed mid (same hue) and tune.
6. **Tip bloom, now "a little stylisation with the tips"** (user, 2026-10-06): white emission following the original
   curve E(s) (above 1.0 from s = 0.73, 3.1 at the tip), compressed to the last ~2.7 cm of the strand ends (window
   0.10 m instead of the original 0.32 m), on every second outer card and the end flyaways of the back and sides
   (|az| >= 80 deg: the front sections hang over her chest in every face-on view), no two tips within 2 cm (no
   crossing "X"). Each glow sits on a real strand of its card's atlas tile (the clump centre) and ends where that
   strand visibly ends (never past the hair). It is drawn as a **soft** glow: a Gaussian core >= 1.5 px, a halo below
   the bloom threshold (0.25 x the core, radius 4 mm clamped to 2-5 px) and a round end, so it reads as a bloom with
   or without URP's bloom (threshold 1 / intensity 1 / scatter 0.7, HDR on, tonemapping off); dimmed by
   sqrt(strand width / pixel size) (0.4..1) so it does not outshine the hair at a distance. Lit hair body <= 0.66
   linear (soft clamp 0.5-0.66). Nothing but the tips blooms: in an off-screen render of the back at 1 m with
   post-processing OFF (values above 1.0 kept), at least 20 pixels above 1.0 with the glow on and none with it off,
   and at least 20 halo pixels more than 2 px from a core. The glow fades out with the AVATAR's opacity
   (x smoothstep(0, 0.1, avatar opacity)), not with the hair's.
7. **Face always clear**: no guide point inside the head-attached face box (in front of the face from the chin to just
   above the brows, 0.8 x head radius to each side) in the rest shape and on any frame of the whole-take sweep
   (`hm_hair --action sweep`: faceIntrusionsMax = 0); rendered card vertices in the box are reported, not asserted:
   up to ~20 on every frame, the dropped front card roots on the forehead (their root rows fade in from coverage 0)
   and card edges at the box edge; none is visible over the face.
8. **Opacity is a real fade**: hair opacity = a + 0.15 x smoothstep(0, 0.2, a) for avatar opacity a (VIEWER_SPEC
   3.2: +0.15 at the normal opacities, 0.35 -> 0.50; it fades out WITH the body: 0.1155 -> 0.21, 0.035 -> 0.047,
   0 -> 0, no floating wig and no pop) or the `hm_hair --opacity` override; hair luminance over a dark background at opacity 1 / 0.65 / 0.45 / 0.3 is monotonic and within 0.12 of
   proportional (`hm_hair --action probe`); it does not vanish below 0.5 (the old shader's cutoff).
9. **Budget (Quest 3)**: at most 200 guides x 16 segments; at most 12k triangles at LOD 0 including the tip glow
   (LOD 1 / 2 by her head's height on screen); 3 draw calls (depth prepass, colour, glow), no shadow pass; seek, loop
   and restart are a lookup in the background bake (<= 1.5 ms per update in the editor; was a 14-39 ms re-hang); CPU
   per frame update <= 1.5 ms in the editor (mean; single updates spike to a few ms when the shared 4-core machine is
   busy).
10. **Right front in fast turns** (`hm_hair --action sweep`): the front-right section (|az| < 60, right of the
   part) is held to its rest shape further down (shape 0.8 / falloff 1.0 / deviation 0.25), the right side behind it
   (|az| 60-95) moderately (0.6 / 0.4 / 0.5; a strong hold folds those strands where a raised arm pushes them); no
   lower-half bend of the right-front guides (|az| < 120) above 90 deg on any frame.

## 7. Method, limits, and how to run it

- **3D hair hull** per 30 Hz frame: voxels (1.5 cm) around her head carved by the follow masks of all 8 cameras
  (sync-corrected nearest frames, max 16 ms apart), each view registered to the fit by its silhouette (18 of 1350
  registered views hit the +-12 px search limit: cam 01 x10, 07 x6, 09 x2); lead pixels only carve what is clearly in
  front of his fitted body; hair = voxels voted hair at least 2:1 against confident body pixels (or one hair vote and
  no body vote within 3 cm of her skin), linked to her head. Check: the re-projected 3D hair covers **96.6 %** of the
  2D hair-mask pixels (median view; 89-99 % per camera); precision 0.60 (the hull is a superset: dilation,
  concavities, voxel disks). Left out (8 of 176): frames 139-144 (about 60.7 s), where camera 01's silhouette did not
  register to the fit (IoU 0.25-0.44, carve-only) and the 3D hair covers too little of the masks, and frames 23 and 46,
  just under the 80 % recall threshold.
- **Tip level** is measured twice: lowest hull hair lying on her back mapped to the rest mesh, and hair pixels
  ray-cast onto the fitted body in every view. They differ by 2.9 cm; the reference is their mean. The adversarial
  review's independent 2D check (hair-mask bottom along the spine in 31 back views, as a fraction of shoulder to
  pelvis) gave 1.221 m.
- **Colour**: phone JPEGs are tone-mapped and auto-white-balanced; the white top fixes the cast and most of the
  exposure, not the tone curve. Hue/saturation are robust; lightness carries about +-9 L* (1 SD across phones).
- **Trailing**: measured against the calm-frame offset, for the head and the torso heading, per turn direction, with
  a bootstrap CI; "resolved" needs both references and both directions to agree in sign, |median| >= 15 deg and a CI
  that excludes 0.
- **Compute** (the workstation is shared): the hull is the only GPU step (about 1.5 s per grid frame on the 2080 Ti;
  about 23 s per frame on the CPU, so `--device cpu` is only for re-measuring). Locally it runs in a child process
  under `dancecap.gpu.exclusive` that returns its VRAM (`torch.cuda.empty_cache`, then the process exits); everything
  else is CPU at below-normal priority with `--threads` (default 4). The re-projection check and the colour samples are
  cached (`hair_ref/consistency.json`, `hair_ref/colour_samples.npz`): a re-render takes under a minute.
- **The 38 s take goes to Runpod**: once its cameras, sync, dancers masks and `smplx/` exist (or in the same batch
  as the `dancers` and `mamma` jobs), copy `hair_ref/observations.json` from larissa_kadu, then
  `python -m dancecap.hair_reference larissa_kadu_38 --write-pod-job` (writes `hair_ref/pod_job.json`, schema
  dancecap.pod_job/1: hull + check + colour + MediaPipe on the pod, only the sampled frames uploaded), run
  `python -m dancecap remote run --take larissa_kadu_38 --jobs hair_ref`, and re-render locally with
  `python -m dancecap.hair_reference larissa_kadu_38 --reuse-hull --reuse-colour --device cpu`.

## 8. Unity implementation (groom version 2, head-movement Assets/Hair)

- **Groom** (`python -m dancecap.hair_groom <take> --restyle` upgrades an existing groom without re-measuring; a full
  run writes version 2 directly): `style` (comb directions per root zone, skull hug, layered volume, the front-right
  drape, settle hold), `cards` (layer table, atlas seed, LOD thresholds), `face_box`, colliders (head = a front-back
  capsule fitted to the scalp within 2.7 mm p95; upper arms and forearms: points 5+, frictionless, capped push-out),
  `render` (albedo = observed mid x 1.85 at the same hue, Kajiya-Kay v2, transmission, soft clamp, tip glow),
  `dynamics` (damping 2.4/s, bend compliance 1e-6, shape 0.6 / 0.22, deviation 0.6 front / 1.0 back).
- **Rest shape** (HairStrands.Settle): skull-hugging combed polylines, then a static settle on the T-pose body with the
  face box; tuned offline with a numpy replica (hair builder's `settle_sim.py`).
- **Simulation**: 160 Burst XPBD guides; face box last in every substep; the whole take is baked in the background
  after load (chunks of 24 frames, 1.3-2.6 ms per frame of background job time = 0.23-0.46 s for the 6 s take, 2.7 MB
  in half precision); seek / loop / slow motion read the bake (0.6-0.8 ms per frame update in the editor, worst seek
  1.5 ms; the live path, used until a frame is baked, costs 1.4 ms per simulation step).
- **Render**: 136 guide-driven cards (24 inner, 40 mid, 48 outer, 8 part, 16 flyaway; inner to 97 % of the length)
  + a 546-triangle scalp cap down to the painted hairline (soft fringed edge, procedural strands across the comb,
  occlusion 0.85 at the hairline / 0.6 under the cards), 15 front card roots dropped toward it, card roots fading in
  over 1.5 cm; procedural 1024 x 1024 strand atlas with coverage-preserving mips and per-tile glow anchors, two-pass
  depth (alpha-to-coverage, MSAA 4x) + ZTest Equal premultiplied colour (queues 2991 / 3001), 16 additive soft tip
  glows (queue 3002); default albedo the portrait's crimson, wider TRT highlight (exponent 24, strength 0.5).
- **CLI**: `hm_hair` state | set (--opacity, --glow, --mode cards|ribbons, --lod, --layers, --colour, --faceguard,
  --arms, dynamics, --frshape / --frfalloff / --frdev / --fraz and --sdshape / --sdfalloff / --sddev / --sdaz for the
  right-front zones) | frame (--solo hides the lead) | sweep | bake | probe (post-processing off) | dump;
  `Tools/playtest_hair.ps1` asserts items 1-10 that can be measured in Unity.

Measured on 02_LarissaKadu (2026-10-07 review fixes, editor, `Tools/playtest_hair.ps1`): rest shape 0 guide points
in the face box, the front-right section's tips 100 % in front of the right shoulder; whole-take sweep 0 guide points
over the face on all 176 frames (up to 20 card vertices in the box, see item 7), stretch 1.03, no resets; turn spread
r95 0.222 m (calm 0.164 m, reference upper bound 0.253 m); trailing angle 14 deg (reported only); right-front guides:
17 frames with a lower-half bend over 45 deg (was 25), p90 45 deg (was 49), max 76 deg, the front-right tips never turn
up (were up to +0.04 on 6 frames); colour (portrait default) white-balanced p10 / p50 / p90 #470D14 / #631920 /
#94544F; fade hair share 1.00 / 0.65 / 0.44 / 0.29 at opacity 1 / 0.65 / 0.45 / 0.3; avatar 0.35 / 0.1155 / 0.035 ->
hair 0.50 / 0.21 / 0.047, glow x 1 / 1 / 0.28; lit body p99.5 0.41 linear; post-off probe 143 pixels above 1.0 with the
glow on, 0 off, 405 halo pixels; 9.5k / 6.6k / 5.3k triangles at LOD 0 / 1 / 2; bake 0.19-0.37 s, 2.7 MB; baked
update 0.55-0.69 ms mean. Review sheets: `work/larissa_kadu/review/realism/hair_fix_*.png` (before / after of the
2026-10-07 review) and `hair_*.png`.

## 9. Ponytail style (2026-10-11, `hair_style = "ponytail"`; `dancecap/hair_ponytail.py`, viewer `Assets/Hair/HairPonytail.cs`)

The groom system's loose sleek style stays the default for every take; a take whose follow block says `hair_style = "ponytail"` gets the ponytail restyle
when `hair_groom` seeds / restyles its groom (`ponytail_hook`). Same guide roots, colliders, render and solver; different style:

- **Tie**: `style.ponytail.tie_az_deg` / `tie_el_deg` (from the head-sphere centre, az 180 = straight back; default 52 deg = a high ponytail on the crown
  / back), `tie_lift_m` above the hair surface. The scalp hair is swept back smooth: every guide runs from its root along a great circle over the head
  collider to the tie (pinned: `skull_hold`), the scalp cap's comb runs the same way (strands converge on the tie), the clean hairline is the cap's
  hairline table (painted into the texture) as before.
- **Tail**: leaves the tie along the tie normal, drooping toward gravity (`tail_droop0`, `tail_down_slerp`), `tail_frac_of_height` x rest height long (0.22:
  about to the shoulder blades), +- `tail_length_jitter` per strand (layered cut), fanned out from the tie (`bundle_radius_m` lateral x depth at
  full offset, `flare` x at the ends, offsets ramped over `offset_ramp_m`): each strand sits in the bundle on the side it entered from. Soft:
  `tail_hold` (per-frame stiffness at the tie, e-fold falloff in tail fractions), deviation cone `deviation`; the existing XPBD solver, face box and body
  colliders do the rest (the tail lies on the back and swings with the head).
- **Cards**: a ponytail table (`hair_ponytail.CARD_LAYERS`: 18 inner / 34 mid / 40 outer / 10 flyaway, no side-part cards) that starts a few cm above
  the tie (the cap covers the scalp), with normals turned around the tail axis (`CardDesc.tubeS / tubePhi`) so the tail is a tube from every side;
  `neighbour_blend` 0.5 keeps the cards from collapsing onto the bundle axis.
- **Tie + barrette**: a bumpy ring round the bundle (`tie.band_radius_m`, `tie.colour`) and an optional small barrette on the scalp (`tie.clip.*`:
  direction, size, angle, colour), vertex-coloured meshes on the head bone with the avatar's translucent attachment materials.
- **Colour**: the take's `work/<take>/avatar/hair_palette.json` (sRGB `mid`, `highlight`, ...; likeness colours stay in work/): `render.albedo_linear` = mid x
  gain as for every groom; `cap_tint` (a shader multiplier on the scalp cap only: hair at the scalp reads darker than the tail in the reference photo),
  `copper_lift` (lighter, golden tips). `[dancers.follow] hair_tip_glow` / `hair_albedo_gain` flip the palette's options.
- **Budget**: 160 guides x 16 segments as before; 102 cards, 7.2k triangles at LOD 0 including glow; bake 0.5 s for 176 frames; sweep on the demo body: 0
  strand resets, max stretch 1.028, face intrusions 0, turn spread r95 median 0.23 m, p90 0.27 m (fast head turns 285-300 deg/s: the tail swings out
  and back as one sheet).
- Re-run: `python -m dancecap.hair_groom <take> --from-take larissa_kadu_full` (needs the take's `smplx/follow.npz`; roots re-based on its betas) then the
  export; or `python -m dancecap.hair_ponytail <take> --from-take larissa_kadu_full`.
