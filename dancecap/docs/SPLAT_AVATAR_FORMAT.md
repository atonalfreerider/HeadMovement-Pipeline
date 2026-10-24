# Splat avatar format (`.splatav` + `.json`), version 1

Written 2026-10-20 by the recap-avatars agent (LHM++ export). Shared contract between the dancecap exporter
(`dancecap/mono4d/recap_splatav.py`, pod side `remote/cand_lhmpp/recap_lhmpp_pod.py`) and the Unity splat layer
(`head-movement/Assets/Splats/`). A change here needs both sides: bump `version`.

One skinned 3D Gaussian avatar per dancer, posed every frame by the SAME 55 SMPL-X joint rotations the avatar plays
(the capture's `smplx_<role>.bin` motion), by plain linear blend skinning (LBS) with 4 joints per Gaussian.

**Likeness data.** These files are a photographic likeness of real people. They live only in
`head-movement/Assets/StreamingAssets/<capture>/splats/` (git-ignored by `Assets/StreamingAssets/*`, checked with
`git check-ignore`) and in `dancecap/work/<take>/` (git-ignored). Never commit them, never put them in an APK.

## 1. Files

```
StreamingAssets/<capture>/splats/<role>.splatav   binary, little-endian: 64-byte header + N fixed-size records
StreamingAssets/<capture>/splats/<role>.json      JSON header: everything else (skeleton, canonical pose, provenance, stats)
```
`<role>` is `lead` or `follow`. The `.json` names its `.splatav` (`"data"`) and carries its byte size and SHA-256.

**Named sets (Quest only, 2026-10-21).** A capture may also carry compact sets of the same avatars in sub-folders,
`splats/<set>/<role>.splatav + <role>.json` (e.g. `splats/quest_25k/`), written by `dancecap/mono4d/quest_distill.py`
(section 10). They are complete files of THIS format (same version, record, skeleton, canonical pose, hair binding), only
fewer Gaussians; the JSON adds a `quest_distill` block. The viewer reads them only when its Quest tier names the set
(`quest_settings.json` `"splat_set"`, `-hmsplatset`; `SplatAvatarData.JsonPath(capture, role, set, out fromSet)`, falling
back to `splats/<role>.json`); the PC and the full files never change.

## 2. Coordinates

- Units: metres.
- **Stored in Unity coordinates** (left-handed, Y up, floor y = 0 at the capture origin), like `smplx_<role>_skin.bin`
  and `smplx_<role>.bin`. The dancecap world is right-handed, Y up, metres; the conversion is exactly
  `export_unity.py`: point / vector `(x, y, z) -> (x, y, -z)` (`to_unity_points`), quaternion `(qx, qy, qz, qw) ->
  (-qx, -qy, qz, qw)` (`to_unity_quats`, the z-mirror conjugation S R S). Scales are unchanged by the mirror.
  The JSON says `"coords": "unity (x, y, -z) of the Y-up right-handed dancecap world"`.
- Quaternions are `xyzw`, unit length, any sign.
- 4x4 matrices in the JSON are row-major, column-vector convention: `p' = M p`, translation in column 3
  (`M[0][3], M[1][3], M[2][3]`). Unity's `Matrix4x4` uses the same convention (assign `m00..m33` row by row).

## 3. Binary `.splatav`

Header, 64 bytes:

| offset | type | field |
|---|---|---|
| 0 | char[4] | magic `SPAV` |
| 4 | u32 | version = 1 |
| 8 | u32 | N = number of Gaussians |
| 12 | u32 | record_bytes = 120 |
| 16 | u32 | sh_rest = number of higher-order SH coefficients per colour channel after the records (0 for LHM++: degree 0 only) |
| 20 | u32 | flags: bit 0 = coordinates are Unity (always 1 in v1) |
| 24 | u32 | header_bytes = 64 |
| 28 | u32[9] | reserved, 0 |

Then N records of 120 bytes (array of structs; a C# `[StructLayout(Sequential, Pack = 4)]` struct reads it as is):

| offset | type | field | meaning |
|---|---|---|---|
| 0 | f32[3] | mean | canonical centre (m), canonical pose = section 4 |
| 12 | f32[4] | rotation | canonical orientation, quaternion xyzw |
| 28 | f32[3] | log_scale | natural log of the three standard deviations (m) along the rotated axes |
| 40 | f32 | opacity_logit | opacity = 1 / (1 + exp(-opacity_logit)) |
| 44 | f32[3] | sh0 | DC spherical-harmonic coefficient per channel: colour = 0.5 + 0.28209479 * sh0 (clamp 0..1) |
| 56 | u16[4] | joints | SMPL-X joint indices 0..54 (unused slots: 0 with weight 0) |
| 64 | f32[4] | weights | LBS weights, >= 0, sum 1 (sorted by weight, largest first) |
| 80 | u8 | hair | 1 = hair Gaussian (section 6), 0 = body / clothes / skin |
| 81 | u8[3] | reserved | 0 |
| 84 | i32 | strand | hair binding: guide strand index (HairStrands guide order), -1 = unbound |
| 88 | i32 | segment | hair binding: segment index along the strand (0 = root segment), -1 = unbound |
| 92 | f32[3] | local_offset | hair binding: centre in the segment frame (m) |
| 104 | f32[4] | local_rotation | hair binding: orientation in the segment frame, xyzw |

If `sh_rest > 0`, a block of N x sh_rest x 3 f32 follows the records (Gaussian-major, coefficient, then RGB).

Colour: LHM++ predicts display colours (it is trained against sRGB photos and renders them by plain alpha
blending). `colour` is therefore an **sRGB-encoded** value. The Unity project is in Linear colour space: convert
with the sRGB transfer function before blending to match the offline look (or composite in gamma space).

## 4. Canonical pose, skeleton, posing

**Canonical pose = the SMPL-X zero pose (all 55 joint rotations identity, "T-pose") of the dancer's shaped body**
(`betas` in the JSON, SMPL-X neutral, 10 betas, `flat_hand_mean` true). Every Gaussian `mean` / `rotation` is in that
pose. So the canonical joint world transform of joint j is a pure translation to its rest joint:
`C_j = T(J_j)`, and the inverse bind is `C_j^-1 = T(-J_j)` - the same bindpose convention as `SmplxAvatar.cs`
(`bindpose_j = T(-J_j)`). The JSON carries:

- `rest_joints`: 55 x 3, Unity coordinates, the shaped zero-pose joints J_j of THIS skeleton (10 betas).
- `parents`: 55 ints, SMPL-X kinematic tree, -1 for the pelvis.
- `canonical_joint_world`: 55 x (4x4 row-major) = T(J_j). Inverse bind = its inverse.
- `canonical_pose`: the 55 x 3 axis-angle of the canonical pose (all zero) and its name.
- `lhmpp_neutral_pose`: the pose LHM++ itself predicts the Gaussians in (its "neutral" pose, `cano_pose_type` of
  the model config: legs spread about the hips' z axis, template shape, betas 0; the exact 55 x 3 axis-angles are read
  from the loaded model on the pod and written here). The exporter moved every Gaussian from that pose to the
  canonical zero pose of the shaped body with LHM++'s own transforms (section 7), so a loader never needs it.
  Provenance only. (LHM++'s saved `*_cano.ply` is yet another pose: `_prepare_smplx_data`'s A-pose, hips +-15 deg,
  shoulders -+30 deg, shaped. Not used here.)

Posing at frame t (exact LHM++ positions, section 7):

1. Local rotations `R_j(t)`: the 55 rotations of the motion file (`smplx_<role>.bin`, Unity coordinates, SMPL-X local
   joint rotations; the ones `SmplxAvatar` assigns to `bones[j].localRotation`).
2. World joint transforms of THIS skeleton: `G_0(t) = [R_0(t) | p_pelvis(t)]` with `p_pelvis(t)` = the dancer's posed
   pelvis position = the avatar's `bones[0]` world position (in `SmplxAvatar` terms `restRoot + Transl[t]`, where
   `restRoot` is the AVATAR skin's J_0; do not add the splat skeleton's own J_0 to `Transl`). For j > 0:
   `G_j(t) = G_parent(j)(t) * [R_j(t) | J_j - J_parent(j)]`.
3. Skin matrices `S_j(t) = G_j(t) * C_j^-1 = G_j(t) * T(-J_j)`.
4. Per Gaussian: `M = sum_k w_k S_{joint_k}(t)` (4 joints), posed centre `= M * mean`.
   Posed rotation = `q(M_3x3) * rotation`, where `q(M_3x3)` is the rotation of the blended 3x3 (LHM++ uses
   `normalize(matrix_to_quaternion(M_3x3))`; a shader may use the normalised, sign-aligned weighted sum of the 4 joint
   quaternions, `normalize(sum_k w_k * sign_k * quat(S_k))` - the difference is reported in the JSON `checks`).
   Scale, opacity and colour do not change with the pose.
5. Hair Gaussians with `strand >= 0` are posed by the hair binding instead (section 6).

Using the avatar's own bones (16-beta skin rest joints) instead of step 2 moves Gaussians by the rest-joint difference
of the two skeletons; the JSON `checks.rest_joints_vs_avatar_skin_mm` gives it per joint.

## 5. JSON header (`<role>.json`)

```
format            "dancecap splat avatar"          version 1
role, name        "follow", "Larissa"              capture "06_LarissaKaduRecap"   take "larissa_kadu_recap"
data              "<role>.splatav"  data_bytes  data_sha256
count             N                                 hair_count
record            [[name, type, count], ...]       the table of section 3
coords            "unity (x, y, -z) of the Y-up right-handed dancecap world"   units "m"
colour            "sh0 DC; colour = 0.5 + 0.28209479 * sh0; sRGB-encoded"
canonical_pose    {"name": "smplx_zero_pose", "axis_angle": 55 x 3 zeros}
lhmpp_neutral_pose {"name": "lhmpp_a_pose", "axis_angle": 55 x 3, "note": ...}
betas             10 floats (SMPL-X neutral; = the recap fit's betas[:10]); fit_betas 16 floats (provenance)
body_model        "SMPL-X neutral (SMPLX_NEUTRAL.npz of LHMPP-Prior), flat_hand_mean"
rest_joints       55 x 3      parents 55      canonical_joint_world 55 x 16
skinning          {"joints_per_gaussian": 4, "source": "...", "truncation": {...}}
posing            text of section 4
hair              {"selection": "...", "binding": "unbound" | "unity_hair_strands", "binding_info": {...} (section 6), ...}
provenance        {"lhmpp_commit", "lhmpp_model", "weights_licence": "CC BY-NC 4.0", "smplx_licence", "inputs": [...],
                   "pod_session", "gpu", "created"}
checks            A2 gate numbers (LHM++ own animation vs this file's LBS), truncation error, rest-joint offsets
stats             opacity / scale percentiles, bounds of the canonical means
```

## 6. Hair

- `hair = 1` marks Gaussians that belong to the hair (long hair, ponytail, loose strands). Body, face skin, clothes and
  headset are 0. The first selection is written by the exporter (`hair.selection` in the JSON says how); a later
  hair stage may replace it (it then rewrites the records' hair bytes and the JSON `hair` block).
- An unbound hair Gaussian (`strand = -1`) is skinned like any other Gaussian (rigid hair, LHM++'s own skinning).
- **Binding** (filled by the hair stage, `dancecap/mono4d/recap_splat_hair.py` for the recap follow, 2026-10-20): the
  Gaussian rides on segment `segment` (points `segment` -> `segment + 1`) of guide `strand` of the viewer's simulated
  guides (`HairStrands`, built from the capture's `hair_groom.json`: 160 guides x 17 points for the recap, guide-major).
  The JSON `hair` block then carries `"binding": "unity_hair_strands"` and `hair.binding_info`:
  `guides`, `points`, `rest_points` (guides x points x 3: the guides' settled REST shape the binding was made on, in
  HEAD-BONE space = Unity coordinates relative to the avatar skin's head joint 15, T-pose axes; `unity command hm_hair
  --action dump` "rest"), `rest_sha256`, `groom_sha256`, `attach {y_top_m, width_m}`, `lever {near_m, far_m}` and the
  binding statistics. A reader needs `guides`, `points`, `rest_points`; `attach` / `lever` default to
  `{0.0, 0.05}` / `{0.02, 0.07}`.
- Gaussian side, head-bone space: `c = mean - rest_joints[15]` (this file's own head joint; the canonical pose is the zero
  pose, so the head bone's rest axes are the Unity axes), canonical rotation `q = rotation`.
- **Segment frame** of rest segment s (points R_s, R_s+1 of `rest_points`), columns X, Y, Z: Y = unit tangent
  R_s -> R_s+1; Z = her forward, Unity `(0, 0, -1)` (= +Z of the SMPL-X head joint, mirrored), made orthogonal to Y
  (Unity `(0, 1, 0)` instead when |Y . forward| > 0.999); X = Y x Z (a rotation, det +1).
  `local_offset = F_s^T (c - R_s)`, `local_rotation = q(F_s)^-1 q` (xyzw). So `c = R_s + F_s local_offset` and
  `q = q(F_s) local_rotation` (the binder checks the float32 round trip: <= 1e-5 mm, 0.03 deg).
- **Posing** (Unity: `Assets/Splats/Avatar/Core/SplatHairBinding.cs` = the CPU reference, compute kernels
  `HairSegments` + `SkinHair` of `HM_SkinSplats.compute`, after the LBS pass; dancecap `recap_splat_hair.pose_reference`
  without the face box):
  1. Simulated points P (world) and the head pose the simulation used (hs, Hs) are re-anchored to the SHOWN head, the
     splat skeleton's head: h = S_15 J_15, H = rotation of S_15: `P' = h + H Hs^-1 (P - hs)` (the hair sits on the head
     the body splats show, also between capture frames: the film's sub-frame poses).
  2. Per segment: `D_s = minrot(H t_s, t'_s) H`, t_s = the simulation's own rest tangent, t'_s = the tangent of P'
     (minimal rotation; 180 deg about a perpendicular axis when opposite). At rest D_s = H.
  3. Per Gaussian, with the simulation's own rest R (live; equal to `rest_points` within 0.001 mm on the recap): offset
     o = c - R_s split along / across the rest tangent, `o = a + p`; `K_s = nlerp(H, D_s, k)`,
     `k = 1 - smoothstep(lever.near_m, lever.far_m, |p|)`; pose from segment s: `x_s = P'_s + D_s a + K_s p`. Hair on its
     strand turns with it; hair far from every guide (the recap's side hair beside the face is 5-12 cm from the nearest
     guide, which drapes behind the shoulder) follows the guide's displacement with the head's rotation instead of
     swinging on a lever arm through the head.
  4. Along the strand, blended with the neighbour segment N = s + 1 when `u = local_offset.y / |R_s+1 - R_s| >= 0.5`,
     else s - 1 (none at the ends): `x = lerp(x_s, x_N, w)`, `w = clamp(|u - 0.5|, 0, 0.5)` (at a segment joint both
     sides blend 50/50: no crease). Rotation `nlerp(K_s, K_N, w) q`.
  5. Face box: the simulation's `HairJobs.FaceBox` (the box in front of the face the guides are kept out of in every
     step; its centre / sizes from `hair_groom.json`) in the shown head's frame: a Gaussian inside it goes out sideways
     (to the side of the face it hangs on at rest, +-1 along head-bone x) or down below the chin, whichever is nearer.
  6. Attach: `lerp(S_15 mean, x, A)`, rotation `nlerp(H q, ..., A)`, `A = smoothstep` of the head-bone height from
     `attach.y_top_m` (A = 0: rigid with the head) down to `attach.y_top_m - attach.width_m` (A = 1), so the hanging hair
     grows out of the rigid scalp hair without a seam.
  Scale, opacity and colour are unchanged (the hair-mode opacity of the LBS pass applies).
- Without a hair source (no simulated hair in the viewer, or a guide / point count that does not match the binding) or
  with the viewer's `hairmotion rigid`, bound Gaussians keep their LBS skinning.

## 7. How the exporter derives the data from LHM++ (LHM++ @ 906b5d9, `LHMPP-700M-PixelShuffle`)

LHM++ (`core/models/rendering/base_gs_render.py` `animate_gs_model`, skinning
`core/models/rendering/skinnings/smplx_voxel_skinning.py` `transform_to_posed_verts_from_neutral_pose`, read
2026-10-20) poses a Gaussian with centre `x = query_point + offset_xyz` (neutral A-pose, template shape) as:

```
W(x)      = skinning weights: the diffused voxel field (FITE, cano_1_volume.npz) sampled at x;
            for hand and face query points the template SMPL-X weights instead
M_null    = sum_j W_j T_j(neutral -> zero pose)                 (55 per-joint transforms of the A-pose, inverted)
x0        = M_null x + shape_blendshape(betas)_i                 (per query point i: zero pose, shaped)
M_pose(t) = sum_j W_j A_j(t)                                      (A_j = SMPL-X FK of the pose about the shaped rest joints)
posed     = M_pose(t) x0 + trans(t)
rotation  = normalize(quat(M_pose(t) M_null)) * rotation_raw
```
Expression offsets are 0 (we pass expression 0), no pose-corrective blendshapes. So the exported `mean` = `x0`
(exact), the weights = `W(x)` (55 joints, truncated to the top 4 and renormalised: error in `checks`), and the
exported `rotation` = `normalize(quat(M_null)) * rotation_raw`. Plain LBS of section 4 then gives LHM++'s posed
centres exactly with all 55 weights; the 4-joint truncation and the rotation factorisation
`quat(M_pose M_null) ~ quat(M_pose) quat(M_null)` are the only differences, measured by the A2 check.

## 8. Measured on the recap export (2026-10-20, `work/larissa_kadu_recap/lhmpp/`)

Pod session `20261020-113940-batch` (RTX A6000, 500.8 s pod lifetime, $0.0821), `pod_out/recap_lhmpp_pod.json`,
`pod_out/a2.json`, local import `report.json`:

- LHM++'s renderer is `GSPlatBackFeatRenderer` with `SMPLXDiffusedVoxelSkinning` (`smplx_diffused_voxel`),
  `apply_pose_blendshape` false, `cano_pose_type` 1 (neutral pose = hips +-20 deg about z; written in
  `lhmpp_neutral_pose`). Expression is passed as 0.
- **LHM++ adds nothing beyond LBS.** Plain LBS of the exported zero-pose means with all 55 exported weights reproduces
  LHM++'s own posed centres to 0.001 mm (max over 12 views x 2 dancers x 160k). Nothing per-frame to bake.
- The format's 4-joint truncation is the only real difference: kept weight mass mean 0.995 / 0.994 (lead / follow),
  min 0.87 / 0.86; posed centres mean 0.43 / 0.50 mm, p99 <= 3.7 / 4.0 mm, max 15.4 / 15.5 mm from LHM++'s.
- Rotation: the blended-3x3 rule (section 4) is 0.12 / 0.14 deg mean from LHM++ (p99 <= 1.3 / 1.8 deg); the shader-
  friendly quaternion blend is 0.47 / 0.52 deg mean (p99 up to 8.6 / 11.4 deg) - the images barely differ (PSNR
  inside the dancers 39.46 vs 39.455 dB median).
- Images (gsplat, both dancers, the real phone 01 / 06 cameras at 1080 x 1920, 6 instants each): ours vs LHM++'s own
  animation PSNR 45.5-55.3 dB over the frame (median 51.4), 37.4-44.1 dB inside the dancers (median 39.5); alpha mean
  abs difference <= 0.0004. Sheet: `a2_sheet.jpg`.
- Placing `G_0` at the motion's pelvis (section 4 step 2) equals LHM++'s `J_0 + trans` exactly (0.000 mm).
- The file itself: re-read `.splatav` posed in Unity coordinates by section 4 matches the world-frame pose mirrored to
  0.0002 mm / 0.0 deg; SHA-256 matches the JSON.
- Rest joints of the splat skeleton (10 betas) vs the shipped avatar skin (`smplx_<role>_skin.bin`, 16 betas):
  max 6.0 / 5.3 mm (`checks.rest_joints_vs_avatar_skin_mm`).
- Sizes: 160,000 Gaussians per dancer, 19,200,064 bytes per `.splatav`. Follow hair flag: 33,492 Gaussians (first pass,
  `hair_follow.jpg`); lead: 0 (shaved head).

## 9. Measured: the recap follow's hair binding (2026-10-20, `work/larissa_kadu_recap/splat_hair_unity/`)

`python -m dancecap.mono4d.recap_splat_hair bind --install` (local CPU, 14 s, deterministic: the same `.splatav` SHA-256
on every run, `79527210...`), from the unbound export `lhmpp/follow.splatav` and the Unity guides' rest dump
(`unity_hair_dump_frame0.json`; the live rest after later domain reloads differed by 0.0008 mm). `report.json`:

- Selection: `splat_hair.select_hair` style `loose` (the other hair workflow's selection; it reproduces their saved
  `splat_hair/follow_hair.npz` exactly): 8,952 scalp Gaussians (RIGID, LHM++ LBS, head weight >= 0.91) and 13,801 FREE.
  Of the free ones, 1,303 white fringe Gaussians (CIE L* > 55, mean opacity 0.02) and 346 grey ones (a* < 15, L* > 15:
  the microphone pack clipped at her lower back under the hair tips, and edges of the top) keep their LBS; they flew out
  with the hair as grey spikes in the first test. **Bound: 12,152** on 749 segments of 150 guides. Hair byte = exporter
  flag OR rigid OR free = 35,899.
- Distance to the bound segment: median 10.0 mm, p90 53 mm, p99 92 mm, max 126 mm (the side hair beside her face and in
  front of the shoulders: the guides there drape behind the shoulders; hence the lever rule of section 6).
- Unity (6000.6.5f1, RTX 2080 Ti, D3D12): GPU vs CPU reference 0.0002 mm / 0.06 deg; hair pass CPU 0.17 ms per frame in
  the editor (reading 2,720 simulated points; the segment frames are a GPU kernel); EditMode tests 14/14.
- `hm_splatdancers --action haircheck --from --to` (every frame of 6 s spans): the bound hair's largest frame-to-frame
  step is at most the guides' (dance 347.8-353.8 s: max 165 mm vs the guides' 193 mm); bound centres inside her head
  (sphere 0.85 x head radius) in 6 of 181 frames (max 52; 708 before the lever rule), 0 at the 778 deg/s spin
  (196.5-202.5 s); inside the partner's capsules max 173 vs 357 for the rigid LHM++ hair (crude capsules; his arm
  under her hair); inside her own torso capsules (r 8 cm about the spine) max 1,108 vs 45 rigid: hair the guides press
  onto her back, hidden inside the opaque body.


## 10. Quest sets: distilled compact avatars (2026-10-21, `work/quest_distill/`)

`python -m dancecap.mono4d.quest_distill prepare` (local: the full avatars of the recap and John & Lula + 32 poses per dancer
sampled from `smplx_<role>.bin`, local rotations only) -> pod job `work/quest_distill/pod_job.json` (gsplat 1.5.3) -> `import`
(writes `head-movement/Assets/StreamingAssets/<capture>/splats/quest_<N/1000>k/<role>.{splatav,json}`, git-ignored; the full
`splats/<role>.*` are never touched). Why: on a Quest 3 the splat dancers are fill bound (head-movement `docs/SPLAT_DANCERS.md`
"Quest": 2 x 160k = 5 fps at the floor edge), and thinning by opacity keeps the large Gaussians and loses arms and faces.

- **Teacher** = the full file, canonical (zero) pose and posed by its own 4-joint LBS (section 4, pelvis at its rest joint,
  quaternion blend). Renders: 512 x 512, random backgrounds, 100 canonical views (fibonacci sphere, 1.2-3.4 m body views, head and
  upper-body close-ups) + 6 views for each of 24 training poses; 8 poses and 32 views held out.
- **Student** = N Gaussians drawn without replacement with p ~ opacity x sqrt(sigma_mid sigma_max) x (1 + 4 x colour detail),
  attributes copied, scales widened to the neighbour spacing; Adam on means / log scales / rotations / colours / opacity logits,
  4000 steps of 4 views: 0.8 L1 + 0.2 (1 - SSIM) + 0.5 alpha L1 + 0.02 x fill (the Quest renderer's fragment count relative to the
  teacher's: per Gaussian 8 min(ln(alpha / (1/255)), 4) sqrt(det Sigma_2D) px^2, net.kleinbeck.gaussians' opacity-aware quad) +
  0.05 x max-scale penalty (sigma above 12 / 16 / 20 mm for 40k / 25k / 15k); dead Gaussians (opacity < 0.02) relocated every 300
  steps for the first 70 %.
- **Skinning / hair transferred**: joints + weights = the top 4 of the inverse-distance blend of the 6 nearest teacher Gaussians'
  weights (re-done every 250 steps and at the end); `hair` = the nearest teacher Gaussian's; a hair Gaussian whose nearest teacher
  Gaussian is bound takes its `strand` / `segment` and gets `local_offset` / `local_rotation` recomputed from its own centre and
  rotation in the rest segment frame (section 6). The JSON is the teacher's with `count`, `data_*`, `hair_count`, `stats`,
  `hair.binding_info.bound` updated and a `quest_distill` block (teacher SHA-256, steps, caps, scores, the thinned baseline).
- **Scores** (held-out views vs the teacher, dancer crop over grey; LPIPS alex): quest_25k 41.6-44.9 dB, LPIPS 0.010-0.017, 5-7 %
  of the teacher's fragments; quest_15k 39.4-43.5 dB, 0.012-0.026; quest_40k 42.8-45.7 dB, 0.008-0.013. The viewer's load-time
  thinning at the same counts: 17.6-24.8 dB, LPIPS 0.24-0.38 (all numbers in head-movement `docs/SPLAT_DANCERS.md` "Quest";
  `pod_out/distill_report.json`). Bound hair Gaussians (recap follow): 1057 / 636 / 379 of 3070 / 1860 / 1110 hair.
- Pod session `20261021-022537-batch`: A40, 1002 s of fitting for 12 sets, pod lifetime 20 min, $0.20.

## 11. Mesh avatars archived (owner rule 2026-10-23)

"I don't ever want to see the basic avatars if gauss splat avatars are available for a dance. fully archive them when they are
replaced with gauss splats." The viewer draws a dancer with `splats/<role>.json` as splats ALWAYS (editor, Windows player, Quest;
skeleton lines where the device cannot draw splats, never the mesh avatar; head-movement `docs/SPLAT_DANCERS.md` "Splats always"), so
the capture files only the mesh avatar reads leave such a capture: `<role>_albedo.png` and `smplx_<role>_face.bin` of each dancer with
splats, and `shoes.json` when both have splats (proved by reference searches: `SmplxAvatar.Build` / `VrTextureBake`, `SmplxFace`,
`SneakerPair` read them; nothing else in the viewer does). `python -m dancecap.export_unity archive-avatars --capture <capture>` moves
them from `StreamingAssets/<capture>/` (with their `.meta`) and `work/<take>/export/<capture>/` into the git-ignored
`work/<take>/archive/mesh_avatars_2026-10-23/<capture>/{StreamingAssets,export}/` with `archive.json` (original path, sha256, size,
why, the restore command; `--restore` moves them back after checking the hashes and refuses to overwrite a different file). Kept:
`smplx_<role>.bin`, `smplx_<role>_skin.bin` (the rig's rest joints and parents the splats are posed from, the hair simulation's scalp),
`hair_groom.json` (the simulation that moves the bound splat hair), `figure*.json`, capture.json (its `smplx_albedo` entries stay) and
every layer. `export_unity.export` routes these kinds of a capture with `splats/` into the archive before the install (with the copy
the install would have made), `face_export` / `shoes` exports sweep them there after writing; tests `tests/test_export_mesh_archive.py`.
Archived 2026-10-23: the class recap, the Carlos & Aline and the John & Lula captures (6 files each: both albedos and `shoes.json`, from
StreamingAssets and the export mirror; no face bins). The source bakes in `work/<take>/avatar/` stay (the splat pipeline reads them).
Fix round 1 (same day): `demo_export.validate` (behind `recap_export` / `carlos_aline_export` / `jl_export --validate-only`) counts an
archived file as present - "archived (splat dancers)" when archive.json records it as moved from that folder and the archived copy
still has its sha256 (`export_unity.archived_from`; only while the capture has splats) - so the validators pass again; `splat_roles`
also counts a named set alone (`splats/<set>/<role>.json`, as the viewer's `CaptureManifest.HasSplatAvatar`). The film library's posters
of the three dances were stills of the mesh avatars: `python -m dancecap.export_unity archive-poster --capture <capture> --film <id>
--new <still>` replaced the library card (`StreamingAssets/library/thumbs/<id>.jpg`, its `.meta` kept) and the `takes/library/films.toml`
source by a still of the splat dancers and archived both mesh stills under `posters/{library,source}/` (archive.json "posters": sha256,
size, why, `replaced_by` with the new still's sha256 and how it was made); `--restore` swaps them back (the splat still kept as
`<name>.splats.jpg`). Head-movement `Tools/deploy_data.ps1` now mirrors each capture folder, so a deploy drops the archived files.
