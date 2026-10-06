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

  cameras/sfm/                          COLMAP model of keyframes (pycolmap): cameras.bin, images.bin, points3D.bin
  cameras/<cam>.npz                     [cameras] per canonical window frame i:
                                          t (F,), K (F,3,3), R (F,3,3), tvec (F,3)  (SfM frame until world_from_sfm),
                                          dist (F,) k1 SIMPLE_RADIAL (0 if pinhole), ok (F,) bool,
                                          reproj_px (F,) median reprojection error, n_inliers (F,)
  cameras/world_from_sfm.json           {s, R, t, gravity_source, scale_source}
  cameras/report.json                   registration per camera, reprojection stats, gate results

  sync/offsets.json                     [sync] per-camera residual video offset delta_c (s) after geometric
                                          refinement; corrected t = t + delta_c

  mamma/                                [mamma] ma_cap-style inputs + run outputs (see mamma_dyn.py)
  smplx/{lead,follow}.npz               [mamma] fused result on the master grid, WORLD frame:
                                          t (T,), global_orient (T,3) axis-angle, body_pose (T,63),
                                          left_hand_pose (T,45), right_hand_pose (T,45), jaw_pose (T,3),
                                          transl (T,3), betas (B,), joints (T,55,3), contact (T,...),
                                          floor_contact (T,...), quality (T,) reprojection-based

  timing/timing.json                    [timing] beats (refined), footsteps (touchdown t, foot, sigma),
                                          joint accents, asynchrony stats with CIs, method notes
  physics/physics.npz + physics.json    [physics] L0 centroidal: com, vcom, acom, F_net + band, couple GRF,
                                          XCoM, support polygon, balance margin, beat impulses, Lz

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
"timing": "timing.json", "physics": "physics.json",
"contacts": "contacts.json"
```

Unity conversion from the Y-up right-handed world: `unity = (x, y, -z)`; rotation quaternion
`(qx, qy, qz, qw)_world -> (-qx, -qy, qz, qw)_unity`.
