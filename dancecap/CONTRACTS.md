# dancecap data contracts

Every stage reads and writes `dancecap/work/<take>/` (git-ignored). Stages never re-decode the source
videos; they read the canonical frame store. When a stage changes a contract, update this file in the
same commit.

## Conventions (all stages)

| What | Convention |
|---|---|
| Time | **reference seconds** `t` on camera 06's clock (affine audio fit per camera). Every per-frame record carries `t`. Never assume a constant frame rate. |
| Master grid | 30 Hz grid `t_k = t0 + k/30` over the analysis window for fused 3D results (MAMMA, physics, timing, Unity). cam02 (59.94) is decimated, never upsampled. |
| World frame | right-handed, **Y-up**, metres, floor at `y = 0`, origin at the dancers' mean floor position. Before scale/gravity are known, cameras live in the SfM frame; `cameras/world_from_sfm.json` (Sim3: `s, R(3x3), t`) maps SfM -> world. |
| Camera model | OpenCV: `x_cam = R @ X_world + t`, +X right, +Y down, +Z forward. Intrinsics `K = [[f,0,cx],[0,f,cy],[0,0,1]]` per frame (zoom). |
| Images | canonical frames are **already rotated** (display matrix applied), native resolution, JPEG q95. Pixel coordinates always refer to these images. |
| Identity | person id **01 = lead** (black t-shirt, shaved head), **02 = follow** (long red hair, white crop top). Same ids in every camera and every stage. Spectators are never given an id. |
| Licence | MAMMA, SMPL-X, VGGT-1B, DA3, LHM++ are non-commercial. No SMPL-X-derived meshes/FBX, dancer textures or splats are ever committed to `head-movement` (public MIT repo). |
| GPU | one 11 GB Turing (sm_75): never bf16 (emulated, 18x slower), never flash-attn; fp16 autocast + SDPA. Use `dancecap.gpu.amp_dtype()`. One heavy GPU job at a time; take `work/gpu.lock` (`dancecap.gpu.exclusive()`). |
| Python | `.venv-nopo4d\Scripts\python.exe` (numpy OK). CUDA extension builds via `prototype\run_cuda.cmd`. Never launch `envs\mamma\python.exe` directly (numpy crash); use `conda run -n mamma` if the conda env is needed. |

## Layout

```
work/<take>/
  frames/<cam>/000000.jpg ...           canonical frames for the analysis window       [ingest]
  frames/<cam>/frames.json              {camera, source, width, height, rotation, slope, offset,
                                          frame_count, fps_median, frames:[{file, pts, t}]}
  keyframes/<cam>/<ref_ms>.jpg          sharp snapshots across the take for the static room
  keyframes/<cam>/keyframes.json        [{file, pts, t}]

  people/<cam>.npz                      [detect] all-person YOLO detections per window frame:
                                          boxes (F, Nmax, 4) xyxy px, scores (F, Nmax), valid (F, Nmax) bool,
                                          kpts (F, Nmax, 17, 3) COCO x,y,conf (if pose model used)
  people_masks/<cam>/000000.png         [detect] uint8 union mask of ALL people (dilated ~10 px) + phone
                                          screens/captions; 255 = ignore for static geometry
  people_masks_kf/<cam>/<ref_ms>.png    same for keyframes
  static_masks/<cam>/mirror.png         [cameras] optional per-camera polygons: mirror wall, burned-in captions

  dancers/<cam>/masks/mask_{frame:04d}_{pid:02d}.png   [dancers] FULL mask (body+hair), MAMMA layout,
                                          frame index = canonical frame index, 255 = person
  dancers/<cam>/body/mask_{frame:04d}_{pid:02d}.png    body-only mask (hair trimmed) -> ma_2d crops. Lead: = full.
                                          Follow: full - hair; her face core, arms and hands (from her matched
                                          keypoints) always stay body, including arms raised above her head
  dancers/<cam>/hair/mask_{frame:04d}_02.png           follow hair mask -> hair analysis: hair-coloured pixels
                                          (per-camera Lab model) connected to her head; empty when her keypoints
                                          are unknown. Invariants: body & hair = 0, body | hair = full
  dancers/roles.json                    {cams:{<cam>:{frames:[{i, t, lead:{box,conf}, follow:{box,conf},
                                          swap_prob, sam_swap_prob, embrace, repaired, anchor}]}}, labels_source,
                                          swaps_repaired:[{cam,start,end}], identity_flags:{<cam>:[msg]}}
                                          swap_prob = posterior that the WRITTEN lead/follow masks are swapped
                                          (after repair); sam_swap_prob = same for SAM's raw objects. Identity
                                          repair is pinned to the verified anchor frames (label + re-anchors);
                                          identity_flags lists cameras whose evidence contradicts the anchors
  dancers/labels.json                   one-time role labels: {<cam>: {frame, t, lead_box, follow_box, by,
                                          confidence, frame_llr, evidence_agreement, cues, descriptors,
                                          colour_names, typesafe?}}; labels_override.json {<cam>:{swap:true}}
                                          (user) -> by="override"; labels_review.jpg shows every labelled cam
  dancers/report.json                   per camera: timings, anchors, embrace/evidence counts, galleries (+ source,
                                          majority diagnostic), identity_flags, hair_model, stray_px_removed,
                                          qa_picks; sam_overrides (when SAM was built this run)
  dancers/cache/<cam>_sam.npz           raw SAM probabilities (working res) keyed by the label; --reuse-sam
  dancers/qa/<cam>.mp4                  overlay video for review (+ <cam>_sheet.jpg 1 s, <cam>_dense.jpg 0.25 s)
  review/dancers_body_hair.jpg          [dancers] follow FULL | BODY | HAIR on raised-arm and hair-whip frames

  cameras/sfm/                          main COLMAP model (keyframes + every ~0.67 s window frame of the strongly
                                          linked cameras; one camera per image): cameras/images/points3D.bin
  cameras/<cam>.npz                     [cameras] per canonical window frame i:
                                          t (F,), K (F,3,3), R (F,3,3), tvec (F,3)  (SfM frame until world_from_sfm),
                                          dist (F,) k1 SIMPLE_RADIAL (0 if pinhole), ok (F,) bool,
                                          reproj_px (F,) median reprojection error, n_inliers (F,)
                                          additive keys (readers may ignore them): attached () bool - True = not in the
                                          static model (own sub-model + attach; reproj_px is then sub-model
                                          self-consistency, NOT main-frame accuracy: down-weight it), dancer_px (F,)
                                          per-frame median dancer-keypoint residual vs the dancers triangulated by the
                                          OTHER main-model cameras (NaN = no observation; written by the qa step).
                                          Attached cameras: ok=False where dancer_px > max(60 px, 3 x its median).
  cameras/world_from_sfm.json           {s, R, t, gravity_source, scale_source, floor_source, x_axis, diagnostics}
                                          gravity = the building's vertical (vertical vanishing points of the main
                                          cameras: diagnostics.vertical, adopted=true) when consistent, else the
                                          dancers' ankle plane; floor offset = sparse floor points within 3 m of the
                                          dancers with the normal fixed to gravity; diagnostics.room_metric_check =
                                          horizontal edges / walls vs world Y. Previous file -> world_from_sfm.prev.json
  cameras/report.json                   registration per camera ("how": main model | own sub-model + attach
                                          (<method>), in_main_model, attached), attach.<cam> (method, vertical
                                          constraint, candidates, rejected Sim3s, dancer fit + temporal hold-out),
                                          map/frame reprojection, focal/zoom events, dancer_loo_check (main cameras:
                                          triangulated from the other MAIN cameras = non-circular; attached: fit
                                          residual + hold-out), vertical_check (per camera building vertical vs world Y,
                                          roll/pitch/sigma), gates_G2a (main-model and attached cameras separately,
                                          gravity_vs_building, dancer_check)
  cameras/qa/                           layout.png (top-down + side, world), focal.png, reproj_<cam>.jpg (static
                                          residuals + 1 m floor grid), dancers_loo_<cam>.jpg
  review/cameras_vertical_check.jpg     [cameras] vertical edges used + new/previous world up per camera
  review/cameras_attach_check.jpg       [cameras] attached cameras: dancers from the main cameras vs detected
  cameras/sfm_work/, cameras/cache/     COLMAP databases, raw models, sub_<cam>/ sub-models, ALIKED features and
                                          LightGlue matches, raw per-frame localisation (loc_<cam>.npz), attach_<cam>.npz,
                                          vertical_*.pkl / horizontal_main.pkl / locsub_<cam>.pkl (keyed caches: inputs'
                                          mtimes + settings + code version)
                                        Cameras without strong static links to the main model (05, 06 on
                                          larissa_kadu: they face the wall behind the main cameras, which no main
                                          camera sees; LightGlue, LoFTR and RoMa all failed to link them) are
                                          localised in their own sub-model and attached: tilt/roll from the building
                                          vertical seen in their own frames, yaw + constant camera centre from the
                                          dancers' keypoints ("rotation-only+vertical").

  sync/offsets.json                     [sync] per-camera residual video offset delta_c (s) after geometric
                                          refinement; corrected t = t + delta_c

  mamma/                                [mamma] ma_cap-style inputs + run outputs (see mamma_dyn.py)
  smplx/{lead,follow}.npz               [mamma] fused result on the master grid, WORLD frame:
                                          t (T,), global_orient (T,3) axis-angle, body_pose (T,63),
                                          left_hand_pose (T,45), right_hand_pose (T,45), jaw_pose (T,3),
                                          transl (T,3), betas (B,), joints (T,55,3), contact (T,...),
                                          floor_contact (T,...), quality (T,) reprojection-based
                                          MAMMA output: model = MAMMA's locked-head SMPL-X neutral
                                          (mamma/data/body_models/smplx_locked_head), betas (16,),
                                          hand poses ABSOLUTE (flat_hand_mean=True); also leye_pose/reye_pose
                                          (T,3), reproj_px (T,) median px, quality = exp(-reproj_px/10),
                                          contact/floor_contact (T,512) on MAMMA's 512 downsampled vertices

  audio/audio_overlap.wav               [beats] mono 16-bit music of the cleanest phone, resampled onto the REFERENCE
                                          clock over the all-camera overlap; sample i at reference_start + i/sample_rate
  audio/audio_overlap.json              [beats] {source_cam, source_file, sample_rate, n_samples, reference_start,
                                          reference_end, time_to_audio (= -reference_start, pass reference_start as
                                          export --audio-t0), slope, offset, caveat}
  audio/quality.json, clicks_overlay.wav [beats] per-phone music quality (envelope SNR vs the other phones, loudness,
                                          clipping, codec bandwidth, beat confidence, beat offsets); wav + grid clicks
  beats/beats.json                      [beats] zouk grid [[t, type]] in REFERENCE seconds, 8 eighth slots per 4-beat
                                          measure (type 1 slots 0,4; 2 slots 3,6; 3 otherwise), starts on a downbeat,
                                          whole measures only (timing's i//8, i%8 measure/slot rule holds). Quarter
                                          level = true tempo (never the split_measures dense grid). Copied to
                                          work/<take>/beats.json for the timing/physics/export search paths.
  beats/beats_raw.json                  [beats] beat_this beats/downbeats (source cam alone + all-phone vote), per-cam
                                          shifts, grid [t, type, measure, slot]
  beats/tempo.json                      [beats] bpm (quarter notes), measure_s, linear-fit residuals, beat level
                                          decision, bar-phase decision + kick/backbeat check, split_measures diagnosis
  timing/timing.json                    [timing] version 2. beats (refined grid [t,type,measure,count] + density
                                          warnings; source: --beats, else timing/beats.json, beats.json or
                                          beats/beats.json - never zouk-time-analysis.json, which is audio-clip time),
                                          footsteps {lead,follow}: [{t, foot, method template|hinge+offset|hinge,
                                          sigma, detect_rate, t_hinge, t_ankle, t_toe, stance, clamped_to_stance?,
                                          beat_t, count, measure, async_ms, ...}], touchdown_method.<role>
                                          {template_feet.<foot>{offset_ms, anchor_ms, offset_sd_ms, at_search_edge},
                                          n_template, n_fallback, n_clamped, mc{draws, per_foot common_shift_sd_ms}},
                                          accents {role:{joint:{acc|jerk:[{t,sigma,snr,reliable,..}]}}},
                                          kinematics.<role>{noise_m, derivatives.{acc|jerk}.<joint>{cutoff_hz, snr,
                                          reliable}, cross_check}, asynchrony (mean/CI/circular incl. rayleigh_p,
                                          v_test_p / surrogate: hit_rate_p, V_p = jitter null (independent uniform
                                          phase per event, one-sided), shift_* = circular shift + p_floor_isochronous
                                          / per measure / measure_effects two-sided vs the OTHER measures /
                                          follow_minus_lead {overall_ms, overall_ci95_ms, per_measure}),
                                          absolute_timing (A/V offset caveat, sync_sigma_s, n_cams, sync_source),
                                          method notes
  timing/kinematics.npz                 [timing] <role>_t (T,), <role>_{pos,vel} (T,55,3) float32 from the quintic
                                          smoothing spline (s = N sigma^2), <role>_{acc,jerk} (T,55,3) from the
                                          penalised quintic smoother at a per-joint MSE-optimal cutoff (world frame,
                                          m, m/s, m/s^2, m/s^3); reliability: <role>_noise_m (), <role>_{acc,jerk}_
                                          {cutoff_hz (55,), snr (55,), reliable (55,) bool, noise_sd (T,55) per
                                          coordinate}. Consumers (Unity jerk overlay) should hide or grey joints
                                          with reliable = False: their jerk is mostly noise
  timing/qa_timing.png                  [timing] foot heights + touchdowns + per-measure asynchrony CIs
  timing/validation.json                [timing] synthetic only: gate G5 PER DANCER (24 s take clean / 1 cm +
                                          noise seeds, and 6 s windows clean / 1 cm x seeds with MC coverage,
                                          jitter-null power and the late-measure contrast)
  physics/physics.npz + physics.json    [physics] L0 centroidal dynamics on the smplx grid t (see physics.py):
                                          npz <role>_{com_raw,com,vcom,acom,f_net,f_net_band,L,dL,L_vertical,
                                          spin_rate,contact (T,2) L/R,support (T,16,2) xz NaN-padded,xcom (T,2) xz,
                                          margin_xcom,margin_com,flags,mass,elem_mass}, grf_couple(+_band),
                                          beats_{t0,t1,impulse_<role|couple>(+_band),mean_force_*},
                                          partner_{lo,hi,lo_robust,hi_robust,cap_bound,status,axis}; F_net = m(a - g)
                                          (N, floor + partner), band = 95 % half-width excl. the mass scale (+-10 %
                                          when mass = volume x 1010). Non-finite input frames are bridged (weight ~0
                                          in the smoother, flag interpolated=512); runs > max_gap_s are NaN in every
                                          output (flag gap=1024); partner bounds that sit on the LP's +-partner_cap_n
                                          box are +-inf in the npz / null in json (flag partner_cap_bound=2048).
                                          json (Unity): t, dancers.<role>.{com, vcom,
                                          xcom_xz, f_net, f_net_band, balance_margin, contact "L"/"R"/"LR"/"",
                                          support_xz, L_vertical, spin_rate, flags (flag_bits), info}, couple.{grf,
                                          grf_band, mg_n}, beats[{t0,t1,impulse,impulse_band,mean_force}], cycles,
                                          beats_source (file the beat grid came from: timing/timing.json's refined
                                          grid first, then timing/beats.json, beats.json, beats/beats.json),
                                          partner.{interval (only identifiable/no contact, else null), lp_range
                                          (null = unbounded or no data), cap_bound_fraction, status
                                          not_identifiable|identifiable|no_contact|inconsistent|no_data}. Partner
                                          force is NOT identifiable in general (see notes).
  physics/qa_physics.png, validation.json [physics] QA plot; synthetic checks + gate G6 per dancer / noise seed,
                                          incl. 6 s windows (windows_6s) clean and noisy with the partner LP

  avatar/{lead,follow}_albedo.png       [texture] photoreal albedo, 2048x2048 sRGB 8-bit RGB, on the SMPL-X UV layout (vt/ft of
                                          the model npz; identical for MAMMA's locked-head and the plain neutral model).
                                          UV convention = Unity's: PNG row 0 is v = 1, texel centre u = (x+0.5)/S,
                                          v = 1 - (y+0.5)/S. Islands are padded by a 16-texel gutter (mip-safe). Lighting
                                          is baked in (no de-lighting); hair is not part of the texture except the
                                          follow's scalp (hair colour). Likeness data: work/ or git-ignored StreamingAssets only
  avatar/{lead,follow}_coverage.png     [texture] per texel R = measured-data alpha (255 = fully from cameras), G = inlier
                                          cameras x 32, B = fill source (0 measured, 128 left/right mirror, 255 diffused
                                          over the mesh); black outside the UV islands
  avatar/{lead,follow}_texels.npz       [texture] per covered texel (atlas order of avatar_uv.build_atlas): rgb_srgb (N,3)
                                          f16 before gutter, alpha, conf, ncam, source, view_gain (n_views,3) linear gains
  avatar/uv_mesh.npz                    [texture] seam-split mesh for a textured skinned mesh: vertex_ids (11313,) i32 =
                                          SMPL-X vertex of each split vertex (copy its rest position / LBS weights from
                                          there), uv (11313,2) f32, faces (20908,3) i32 into the split vertices, WORLD
                                          right-handed winding (reverse for Unity, as for the skin). vertex_ids[faces] == f
  avatar/head_fix.npz                   [texture] neck-rotation correction fitted to the multi-view COCO face keypoints
                                          (avatar_head.py), per role: <role>_t (T,), <role>_delta_rotvec (T,3) world rotation
                                          about joint 12, <role>_neck_pose (T,3) corrected LOCAL axis-angle of joint 12
                                          (= body_pose[:, 33:36] replacement; exact SMPL-X FK), <role>_err_{before,after}_m
                                          (T,) median face-keypoint error (m at the head), <role>_n_obs, <role>_adopted ()
                                          bool (auto: leave-one-camera-out median ratio < 0.9 and >= 75 % cameras better).
                                          The texture is baked on the adopted pose; smplx/ is NOT modified (mamma owns it)
  avatar/report.json                    [texture] config, FK check, head_fix (incl. leave-one-camera-out), hair_model
                                          (per-camera hair-colour model recall / false positives), passes (flow
                                          registration px), per-role measured/mirror/diffused fractions, camera weight
                                          shares + exposure gains, per-view stats (silhouette IoU, fit quality, sharpness,
                                          samples, flow), review_metrics (render-vs-photo MAE per camera)
  review/avatar_texture_{lead,follow}.jpg, avatar_uv.jpg, avatar_turntable.jpg, avatar_face.jpg   [texture] QA:
                                          re-render into real frames (photo | avatar | overlay), atlases + coverage,
                                          virtual turntable, face close-ups (photo | avatar) where each face is seen best

  room/                                 [room] static studio, all in the SfM frame (python -m dancecap room <take>):
                                          splat_full.ply (standard 3DGS PLY: f_dc, channel-major f_rest SH3, logit
                                          opacity, log scales, wxyz rot; trained 'antialiased'), splat_quest.ply (<= 150k
                                          Gaussians, SH1), room_mesh.obj/.mtl (materials "room" -> room_mesh_tex.png
                                          xatlas atlas, "floor" -> room_floor_tex.png orthographic floor plane),
                                          room_mesh.ply (same triangles, vertex colours), eval.json (held-out masked
                                          PSNR/SSIM per camera, raw + colour-corrected), report.json (config, views,
                                          train history, quest/mesh stats, gates), cache/splat_full.pt (+ per-image
                                          colour transforms, learned reflectances)
  room/transform.json                   {frame:"sfm", world_from_sfm (copy), floor_plane_sfm {n, d, tilt vs world up},
                                          mirror {world/sfm plane n.X+d=0 (n towards the cameras), along axis + range,
                                          height range}, reflectance {floor {f0, fresnel_gain}, mirror {f0}},
                                          room_box_sfm}. The splats/mesh are the DIFFUSE room: the glossy floor and
                                          the mirror wall were modelled as planar reflectors during training (photo =
                                          diffuse + f x mirrored render), so Unity may add planar reflections there.
  review/room_*.jpg                     [room] room_renders (held-out real | model | diffuse | Quest splat | Quest
                                          mesh), room_overview (top-down + obliques + Gaussian height section),
                                          room_overview_quest, room_mesh
 head-movement StreamingAssets capture (see below)
```

## Unity capture (export) - capture.json version 3

Backwards compatible with v2 (figure1/2.json, video_meta.json still written). Adds:

```
"version": 3,
"times": "times.json",                 per-frame reference seconds (T,) - playback looks frames up by time
"world": "right-handed Y-up metres, floor y=0; Unity = (x, y, -z)",
"smplx": {"lead": "smplx_lead.bin", "follow": "smplx_follow.bin"},   little-endian float32 blocks:
                                         header json + per frame: transl(3) + 55 joint local rotations as
                                         unity quaternions (x,y,z,w) + 55 joint positions (unity)
"smplx_skin": {"lead": "smplx_lead_skin.bin", "follow": "smplx_follow_skin.bin"},
"time_to_audio": s,                    audio-clip time = reference t + time_to_audio (--audio-t0 = reference time of
                                         the wav's first sample); audio_offset (v2) = times[0] + time_to_audio
"timing": "timing.json", "physics": "physics.json",
"contacts": "contacts.json"            (reserved, not written yet)
```

Written by `python -m dancecap export <take> --capture <Name>` (dancecap/export_unity.py) to
`work/<take>/export/<Name>/` and mirrored to `head-movement/Assets/StreamingAssets/<Name>/` (git-ignored).
Binary files: 4-byte magic + uint32 header length + JSON header (space-padded to 4 bytes) + raw little-endian blocks.

- `smplx_<role>.bin` (magic `DCXM`): T x 388 float32 = transl(3) + 55 local rotations (x,y,z,w Unity) + 55 joint
  positions (Unity). Joint order = SMPL-X (0 pelvis .. 21 wrists, 22 jaw, 23/24 eyes, 25-39 left hand, 40-54 right
  hand); rotations are the npz axis-angles (hands absolute, flat_hand_mean=True), so FK with the skin's rest joints
  (root bone at rest_joint0 + transl) reproduces `joints` (exporter checks < 0.01 mm on synthetic).
- `smplx_<role>_skin.bin` (magic `DCXS`, SMPL-X-derived, never commit): rest_vertices (V,3) f32 for the dancer's
  betas (expression 0), faces (F,3) i32 with winding reversed for the left-handed frame, LBS weights (V,55) f32,
  rest_joints (55,3) f32, parents (55,) i32. Model = MAMMA locked-head SMPL-X for MAMMA fits, else
  `smplx/SMPLX_NEUTRAL.npz` (`--model` overrides). Plain LBS: pose-corrective blendshapes are not applied.
- `timing.json` (Unity subset of timing/timing.json, reference seconds): beats.grid [t,type,measure,count],
  footsteps.{lead,follow}[{t, foot, async_ms, beat_t, sigma, ...}], asynchrony.{role}.{n, mean_ms, ci95_ms,
  hit_rate, R}, and when timing/kinematics.npz exists jerk.{lead,follow} (T,24) |jerk| m/s^3 of the SMPL-24 joints
  (SMPL-X 0..21 + left/right_middle1) on the capture grid - Unity uses it instead of finite differences.
- `physics.json` is copied verbatim (world frame; the Unity overlay converts (x,y,z) -> (x,y,-z)).
- `zouk-time-analysis.json` holds the beat grid in audio-clip time ([[t + time_to_audio, type]]).

Unity (head-movement): `CaptureTimeline` maps audio time -> frame by binary search over times[] + time_to_audio
(nearest frame); `SmplxAvatar` builds a 55-bone SkinnedMeshRenderer per dancer (layer `avatars`); overlays
`TimingOverlay` (layer `timing`) and `PhysicsOverlay` (layer `physics`) load only when their files exist.

Unity conversion from the Y-up right-handed world: `unity = (x, y, -z)`; rotation quaternion
`(qx, qy, qz, qw)_world -> (-qx, -qy, qz, qw)_unity`.
