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
  dancers/<cam>/body/mask_{frame:04d}_{pid:02d}.png    body-only mask (hair trimmed) -> ma_2d crops
  dancers/<cam>/hair/mask_{frame:04d}_02.png           follow hair mask -> hair analysis
  dancers/roles.json                    {cams:{<cam>:{frames:[{i, t, lead:{box,conf}, follow:{box,conf},
                                          swap_prob, embrace}]}}, labels_source, swaps_repaired:[...]}
  dancers/labels.json                   one-time role labels: {<cam>: {frame, lead_box, follow_box, by}}
  dancers/qa/<cam>.mp4                  overlay video for review

  cameras/sfm/                          main COLMAP model (keyframes + every ~0.67 s window frame of the strongly
                                          linked cameras; one camera per image): cameras/images/points3D.bin
  cameras/<cam>.npz                     [cameras] per canonical window frame i:
                                          t (F,), K (F,3,3), R (F,3,3), tvec (F,3)  (SfM frame until world_from_sfm),
                                          dist (F,) k1 SIMPLE_RADIAL (0 if pinhole), ok (F,) bool,
                                          reproj_px (F,) median reprojection error, n_inliers (F,)
  cameras/world_from_sfm.json           {s, R, t, gravity_source, scale_source, floor_source, x_axis, diagnostics}
  cameras/report.json                   registration per camera ("how": main model | own sub-model + dancer
                                          attach), map/frame reprojection, focal/zoom events, dancer leave-one-out
                                          reprojection check, gates_G2a
  cameras/qa/                           layout.png (top-down + side, world), focal.png, reproj_<cam>.jpg (static
                                          residuals + 1 m floor grid), dancers_loo_<cam>.jpg
  cameras/sfm_work/, cameras/cache/     COLMAP databases, raw models, sub_<cam>/ sub-models, ALIKED features and
                                          LightGlue matches, raw per-frame localisation (loc_<cam>.npz)
                                        Cameras without strong static links to the main model (05, 06 on
                                          larissa_kadu) are localised in their own sub-model and attached with the
                                          dancers' keypoints; "rotation-only" attach = constant camera centre.

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

  timing/timing.json                    [timing] beats (refined grid [t,type,measure,count] + density warnings),
                                          footsteps {lead,follow}: [{t, foot, sigma, detect_rate, beat_t, count,
                                          measure, async_ms, ...}], accents {role:{joint:{acc|jerk:[{t,sigma,..}]}}},
                                          asynchrony (mean/CI/circular/hit rate vs surrogate/per measure/permutation,
                                          follow_minus_lead), absolute_timing (A/V offset caveat), method notes
  timing/kinematics.npz                 [timing] <role>_t (T,), <role>_{pos,vel,acc,jerk} (T,55,3) float32 from the
                                          quintic smoothing spline (world frame, m, m/s, m/s^2, m/s^3)
  timing/qa_timing.png                  [timing] foot heights + touchdowns + per-measure asynchrony CIs
  physics/physics.npz + physics.json    [physics] L0 centroidal dynamics on the smplx grid t (see physics.py):
                                          npz <role>_{com_raw,com,vcom,acom,f_net,f_net_band,L,dL,L_vertical,
                                          spin_rate,contact (T,2) L/R,support (T,16,2) xz NaN-padded,xcom (T,2) xz,
                                          margin_xcom,margin_com,flags,mass,elem_mass}, grf_couple(+_band),
                                          beats_{t0,t1,impulse_<role|couple>(+_band),mean_force_*},
                                          partner_{lo,hi,lo_robust,hi_robust,status,axis}; F_net = m(a - g) (N,
                                          floor + partner), band = 95 % half-width excl. the mass scale (+-10 %
                                          when mass = volume x 1010). json (Unity): t, dancers.<role>.{com, vcom,
                                          xcom_xz, f_net, f_net_band, balance_margin, contact "L"/"R"/"LR"/"",
                                          support_xz, L_vertical, spin_rate, flags (flag_bits), info}, couple.{grf,
                                          grf_band, mg_n}, beats[{t0,t1,impulse,impulse_band,mean_force}], cycles,
                                          partner.{interval (only identifiable/no contact, else null), lp_range,
                                          status}. Partner force is NOT identifiable in general (see notes).
  physics/qa_physics.png, validation.json [physics] QA plot; synthetic checks + gate G6

  export/<CaptureName>/                 [export] head-movement StreamingAssets capture (see below)
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
