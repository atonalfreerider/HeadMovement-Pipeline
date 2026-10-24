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
| GPU | one 11 GB Turing (sm_75): never bf16 (emulated, 18x slower), never flash-attn; fp16 autocast + SDPA. Use `dancecap.gpu.amp_dtype()`. One heavy GPU job at a time; take `work/gpu.lock` (`dancecap.gpu.exclusive()`; `DANCECAP_GPU_SHARED=1` skips the lock - only for several small jobs side by side on an 80 GB pod GPU). |
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
  dancers/geo_veto.json                 [dancers_geoveto] wrong-PERSON check the colour identity cannot make: each role's
                                          COCO keypoints vs the same role (and the partner) triangulated from the OTHER main
                                          cameras; explained by neither dancer on >= 3 grid instants -> veto runs per
                                          camera/role (rule, thresholds, runs, native frame plan). --apply / --manual
                                          cam:role:i0-i1 (visually verified spans where the camera pose is not ok and
                                          geometry is blind): masks/body/hair PNGs of that role blanked (originals in
                                          dancers/geo_veto_backup/<cam>/), roles.json frames[i][role] box None + geo_veto,
                                          roles.json geo_veto.history [{applied, frames, source}], mamma/ma_2d_native/<cam>.npz
                                          visibilities of that person 0 (original <cam>.pre_veto.npz); then `mamma prepare
                                          --skip-2d`. review/reid_veto.png (per-role residual curves; vetoes APPLIED so far
                                          from roles.json solid - lead top band, follow bottom - and the automatic rule's
                                          current candidates hatched, not applied), review/reid_veto_cases.jpg (the removed
                                          masks). --check: re-measure only -> dancers/geo_veto_check.json (+ applied_so_far),
                                          geo_veto.json (the applied run's record) is left alone

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
                                          axis_sd (F,) f4 posterior SD of the camera centre along its optical axis (SfM
                                          units, from the frame's inlier correspondences, focal free; NaN when not
                                          localised), recentred (F,) bool: zoom-depth guard (cameras/depth_guard.py, run
                                          by stage_smooth on the raw localisation): frames with axis_sd > 0.003 x the
                                          camera's median static-point depth or < 60 inliers get their centre from an
                                          inverse-variance robust smooth (0.2 s) of all ok centres, rotation + focal
                                          re-solved with it fixed; ok=False if the re-solve fits its static points worse
                                          than 4 px. Attached cameras are skipped. report.json cams.<cam>.depth_guard
                                          Camera types (2026-10-07, additive; only for a camera whose take TOML gives
                                          it a confirmed type, see "Camera types" below): camera_class () str
                                          tripod | tripod_zoom | handheld | handheld_zoom, type_model () str
                                          per_frame | fixed_focal | constant | fixed_centre (the model applied by
                                          cameras/typed.py, after its residual / drift check), type_held (F,) bool
                                          (frames whose pose is the constant tripod pose; ok on frames without static
                                          points of their own). A typed camera skips the zoom-depth guard (axis_sd NaN,
                                          recentred False). Files without these keys = today's general model.
  cameras/world_from_sfm.json           {s, R, t, gravity_source, scale_source, floor_source, x_axis, diagnostics}
                                          gravity = the building's vertical (vertical vanishing points of the main
                                          cameras: diagnostics.vertical, adopted=true) when consistent, else the
                                          dancers' ankle plane; floor offset = sparse floor points within 3 m of the
                                          dancers with the normal fixed to gravity; diagnostics.room_metric_check =
                                          horizontal edges / walls vs world Y. Previous file -> world_from_sfm.prev.json
                                          Metric scale (2026-10-06, additive keys): scale_rule "chain" (default) |
                                          "eye_p90" (takes/<take>.toml [cameras] scale_rule; the larissa_kadu world of
                                          2026-10-06 = eye_p90, 4.8 % too large). chain = each dancer's unfolded COCO
                                          chain (median leg hip-knee-ankle + median mid-hip->mid-shoulder + median
                                          mid-shoulder->mid-eye; segment lengths, so bent knees / crouch bias it little;
                                          trunk and neck bend, so it is pose-robust, not invariant) = CHAIN_FRAC 0.900
                                          x stature (CHAIN_FRAC is CALIBRATED on larissa_kadu's c8 betas, not
                                          validated; independent anthropometric prior 0.895 +- 0.01 = -0.7 % there),
                                          both dancers combined by an inverse-variance log mean (variance = block-
                                          bootstrap SE^2 + SCALE_SYS_REL^2, 1 % systematic). diagnostics: scale_rule, scale_chain.<role>
                                          {chain, leg, trunk, shoulder_eye, n, rel_se, s, stature_m}, scale_eye_p90.<role>
                                          (old rule, always recorded), scale_per_dancer (= adopted rule), scale_weights,
                                          scale_disagreement, scale_agreement_ok (<= 1.5 %), scale_rule_fallback (chain
                                          keypoints missing -> eye_p90), rescaled [{factor, s_before, s_after, reason}]
                                          (world_scale.py rescale). cameras/cache/dancers_sfm.npz now also holds
                                          shoulders/hips/knees (<role>_<kp>, KP_SCALE)
                                          Scale review fixes (2026-10-06, additive): diagnostics scale_rule_requested,
                                          scale_sys_rel, scale_rel_se_combined, scale_s_equal_weight, chain_frac {value,
                                          kind, prior, s_with_prior}, scale_abs_uncertainty_rel (0.015), sole_m,
                                          sole_source, dancer_roles.<cam> {source "roles.json" | "fallback: two largest
                                          boxes", frames, lead, follow}; scale_chain.<role> + {halves, halves_rel_diff,
                                          pct {leg|trunk|shoulder_eye: [p10, p90]}, leg_lr, per_frame_median,
                                          shoulder_eye_eyes, shoulder_eye_both / shoulder_eye_one_eye {median, n}};
                                          scale_reference {ref_take, sim3_path, ref_record, s_bootstrap, s_inherited,
                                          factor, chain_vs_reference, gravity_vs_inherited_deg, n_cams, resid_rel,
                                          per_cam, n_frames, n_used, adopted, why} when takes/<take>.toml [cameras]
                                          scale_reference = {take, sim3} is set: after the bootstrap the world is
                                          rescaled (about the origin; R, floor, origin kept) to the scale inherited from
                                          that reference world (a file made by `world_scale reference <ref take>`)
                                          through the SfM similarity of the same camera frames, when >= 4 cameras have
                                          >= 30 matched frames, the centre residual is < 1 % of their spread, gravity is
                                          within 1 deg of the inherited one (compared, never changed) and the reference
                                          was built with the take's shoe soles; else the bootstrap scale is kept
                                          (adopted false, why). The chain is then a cross-check (chain_vs_reference).
                                          ABSOLUTE scale +-1.5 % (any rule): the statures are taken as barefoot and the
                                          fitted mesh sole as the shoe sole: takes/<take>.toml [dancers.<role>]
                                          shoe_sole_m (default 0.025, "assumed"; 0 if the statures were measured in the
                                          dance shoes; sole 0 vs 2.5 cm = 1.5 %). Also the floor's ankle height.
                                        Scale check / correction (cameras/world_scale.py, python -m
                                          dancecap.cameras.world_scale check|rescale|compare|reference <take>): check =
                                          each dancer's SMPL-X rest height from smplx/<role>.npz betas (crown to sole,
                                          the fit's model) vs stature + shoe_sole_m -> factor, disagreement,
                                          max_per_dancer_rel, verdict ok (mean AND each dancer within 1.5 %) | rescale |
                                          inconsistent (--strict: exit 2 unless ok; it cannot see the sole assumption,
                                          whose effect equals its tolerance); reference --out f = the take's world x its
                                          check factor + a "reference" record {take, factor, statures, sole_m, k,
                                          verdict, mtimes} (refuses an inconsistent check or a fit older than the
                                          world); rescale --factor f = world x f about the origin
                                          (s, t x f; R, floor, origin and image projections unchanged; cameras/<cam>.npz
                                          are SfM-frame and untouched; room/transform.json world_from_sfm copy + its
                                          world-metre fields synced, previous -> transform.prev.json; smplx/, mamma
                                          inputs, physics, timing, avatar, export become stale - re-run them);
                                          compare --ref <take> = SfM similarity between two takes of the same
                                          recordings from main-camera centres matched by frames.json pts -> the scale
                                          (and gravity) inherited from the reference world
                                        Scale review round 2 (2026-10-06 evening, additive):
                                          chain rule sole coupling: chain = CHAIN_FRAC x chain_stature, chain_stature
                                          = stature + shoe_sole_m - CHAIN_FRAC_SOLE_M (0.025, the calibration's sole;
                                          = stature at the default), so a changed shoe_sole_m moves the chain like the
                                          betas reference; bootstrap(sole_m=) takes {role: m} (floor = their mean).
                                          diagnostics: soles_m, scale_abs_verified false, chain_frac
                                          {calibration_sole_m, chain_stature_m, note}, scale_chain.<role> {sole_m,
                                          chain_stature_m}; scale_reference {statures_checked, ref_inputs_verified
                                          (true | false | null = record predates fingerprints), ref_inputs, warning,
                                          abs_verified false, chain_vs_reference_max}; gates_G2a world_scale.value
                                          abs_verified + note (passed = self-consistent, not absolutely verified).
                                          reference record + inputs {files {relpath: {sha1, size, mtime}}, sha1,
                                          patterns} (cameras/sfm/{cameras,images,points3D}.bin, cameras/<cam>.npz,
                                          frames/<cam>/frames.json of the reference take), chain_frac_calibration_sole_m,
                                          abs_verified false. adopt_reference(statures=) refuses a reference whose
                                          statures differ from the take's (> 0.5 mm) or whose inputs changed (sha1);
                                          the cameras stage passes the take's statures and prints a WARNING when the
                                          reference is not adopted. world_scale check + absolute {verified false,
                                          uncertainty_rel, note, verdict_means} and floor.<role> {n_frames, median_mm,
                                          p5_mm, p95_mm, min_mm, max_mm, n_below_15mm, frac_below, warning (> 10 % of
                                          frames), n_above_15mm, frac_above (floating: no body point within 15 mm of the
                                          floor)} = posed lowest mesh vertex over all frames (LBS, no pose correctives;
                                          --no-floor skips). New action `chain <take> [--no-sync]`: chain rule
                                          re-measured on the main cameras with sync/offsets.json applied to the frame
                                          times -> {s_world, s_chain, chain_vs_world, passed (|.| <= 2 %),
                                          sync_applied, offsets_ms, combined, per_dancer, s_chain_pre_sync,
                                          sync_effect_rel}. Coarse cross-check only, never a reason to rescale alone.
  cameras/report.json                   registration per camera ("how": main model | own sub-model + attach
                                          (<method>), in_main_model, attached), attach.<cam> (method, vertical
                                          constraint, candidates, rejected Sim3s, dancer fit + temporal hold-out),
                                          map/frame reprojection, focal/zoom events, dancer_loo_check (main cameras:
                                          triangulated from the other MAIN cameras = non-circular; attached: fit
                                          residual + hold-out), vertical_check (per camera building vertical vs world Y,
                                          roll/pitch/sigma), gates_G2a (main-model and attached cameras separately,
                                          gravity_vs_building, dancer_check, and from 2026-10-06 world_scale = the
                                          adopted scale is trustworthy: requested rule ran, every main camera's dancers
                                          role-labelled by dancers/roles.json, configured reference adopted with the
                                          chain within 2 % of it - or, without a reference, world_scale_chain - and
                                          world_scale_chain = dancers agree <= 1.5 %, per-dancer statistical SE
                                          <= 0.75 %, first vs second half of the window <= 2 %; both have `checks`)
  cameras/qa/                           layout.png (top-down + side, world), focal.png, reproj_<cam>.jpg (static
                                          residuals + 1 m floor grid), dancers_loo_<cam>.jpg
  review/cameras_vertical_check.jpg     [cameras] vertical edges used + new/previous world up per camera
  review/cameras_attach_check.jpg       [cameras] attached cameras: dancers from the main cameras vs detected
  cameras/sfm_work/, cameras/cache/     COLMAP databases, raw models, sub_<cam>/ sub-models, ALIKED features and
                                          LightGlue matches, raw per-frame localisation (loc_<cam>.npz; additive
                                          camera-type marker loc_mode () keyframes|all, loc_class (), loc_policy (),
                                          loc_localised () int: see "Camera types"), attach_<cam>.npz,
                                          vertical_*.pkl / horizontal_main.pkl / locsub_<cam>.pkl (keyed caches: inputs'
                                          mtimes + settings + code version)
                                        Cameras without strong static links to the main model (05, 06 on
                                          larissa_kadu: they face the wall behind the main cameras, which no main
                                          camera sees; LightGlue, LoFTR and RoMa all failed to link them) are
                                          localised in their own sub-model and attached: tilt/roll from the building
                                          vertical seen in their own frames, yaw + constant camera centre from the
                                          dancers' keypoints ("rotation-only+vertical").

  sync/offsets.json                     [sync] per-camera residual video offsets from multi-view geometry (sync.py):
                                          {version, gauge, method, audio_beat_offsets_ms, offsets:{<cam>:{delta (s),
                                          delta_ms, stable, use, spread_ms, sub_windows [{t_mid, delta_ms, min_px,
                                          n_frames}], rate (s/s), t_ref, use_rate, residual_px{at_delta, at_zero,
                                          improvement}, at_search_edge}}}. Corrected reference time = t + delta
                                          (+ rate (t - t_ref) only where use_rate); "use": false -> no correction.
                                          Gauge: median delta over cameras = 0 (geometry fixes relative offsets only).
                                          stable = 2 s sub-window estimates within 10 ms. sync/curves.json = the
                                          residual curves; review/sync_curves.png.
                                          clocks {alignment, sources {<cam>: [slope, offset]}} = the audio clocks of
                                          frames/<cam>/frames.json the deltas were measured on: readers
                                          (mamma_dyn.read_sync_offsets -> prepare, load_tracks, hair_reference) refuse
                                          a file whose clocks differ from the frames'/take's ("stale", e.g. copied
                                          between takes) and, for a take with [sync] required = true, a file without
                                          clocks ("unrecorded"). Never copy offsets.json between takes. + workers, host
                                          (sync --workers N: residual evaluations in a process pool, identical numbers;
                                          default 1 off-pod, the container CPU quota up to 16 on a pod)
                                          + camera_models {<cam>: per_frame | fixed_focal | constant | fixed_centre}
                                          (2026-10-07, additive: the camera-type model of each track measured on)
  mamma/                                [mamma] split in prepare (local) / fit (Runpod) / select (mamma_dyn.py):
                                          ma_2d_native/<cam>.npz (+ _boxes.npy, .manifest.json: recomputed when
                                          masks/frames/weights change), ma_2d/<take>_<tag>/cam<cam>.npz + ma_cap/
                                          <take>_<tag>/gt/ (30 Hz grid inputs per camera set, grid inside EVERY camera's sync-corrected coverage; gt npz carry t,
                                          native_index, img_rel_path relative to work/<take>), ma3d_config.yaml
                                          (MAMMA contact config + dancecap overrides: light acceleration prior in
                                          second_run), prepare.json (grid t + grid.trim {window, start/end_limited_by:
                                          cameras whose sync-corrected coverage trimmed the take window}, sets, eval
                                          set, sync offsets used + sync.required,
                                          world_frame, camera_types {<cam>: {class, model}} (additive 2026-10-07: from
                                          cameras/<cam>.npz; a 'constant' tripod track gives one cam_ext for every grid
                                          frame)), pod_job.json (what the pod uploads/runs/downloads),
                                          ma_3d/<tag>/<seq>/ raw output, fits/<tag>/{smplx/, fit.json} (export +
                                          per-camera residuals incl. held-out cameras), select.json + report.json
                                          (chosen set and why). smplx_v1/ = first fit in the OLD world frame.
                                          Leave-one-camera-out (pod-session 2026-10-06): loo/pod_loo.py + loo/pod_job.json
                                          (job mamma_loo, after mamma; fits x<cam> = c8 minus <cam> on the c8 inputs, 7 in
                                          parallel under CUDA MPS) -> fits_loo/x<cam>/ (never in fits/, so a plain select
                                          ignores them) + loo.json {keep, reasons, table[cam]{held_by, lead/follow
                                          {in_fit_c8_px, held_out_px, ratio}}, loo_outliers}; select.json reasons/loo
                                          carry the LOO decision (cam05 rejected only if its held-out/in-fit ratio > 2x the
                                          median of the other cameras' LOO ratios, or adding it worsens shared cameras)
  smplx/{lead,follow}.npz               [mamma] fused result on the master grid, WORLD frame (copied from the selected
                                          mamma/fits/<tag>/smplx; never written without cameras/world_from_sfm.json):
                                          t (T,), global_orient (T,3) axis-angle, body_pose (T,63),
                                          left_hand_pose (T,45), right_hand_pose (T,45), jaw_pose (T,3),
                                          transl (T,3), betas (B,), joints (T,55,3), contact (T,...),
                                          floor_contact (T,...), quality (T,) reprojection-based
                                          MAMMA output: model = MAMMA's locked-head SMPL-X neutral
                                          (mamma/data/body_models/smplx_locked_head), betas (16,),
                                          hand poses ABSOLUTE (flat_hand_mean=True); also leye_pose/reye_pose
                                          (T,3), reproj_px (T,) median px over all 512 points and cameras,
                                          reproj_px_raw (T,) before outlier repair, repaired (T,) bool = interpolated
                                          frame (quality <= 0.1), suspect (T,) bool = > 3x median reprojection but not
                                          repaired, parts (6,) str (body, left_hand, right_hand, left_foot, right_foot,
                                          head), reproj_parts_px (T,6) median px per part (NaN = unobserved),
                                          quality_parts (T,6) = exp(-reproj_parts_px/10); quality = exp(-w/10) with
                                          w = the worst of body/feet/head (and >= reproj_px): hands are only in
                                          quality_parts; contact/floor_contact (T,512) on MAMMA's 512 downsampled
                                          vertices; strings gender, model, source; flat_hand_mean, num_betas
  review/sync_curves.png, review/pose_mamma_<tag>.jpg   [sync]/[mamma] offset curves; fitted 512 points vs ma_2d
                                          landmarks per camera (held-out cameras labelled)

  (prototype/<take>/alignment_studio.json) [clock_studio build] alignment.json's schema (reference, convention, sources[]
                                          {id, name, duration, numbered, slope, offset, anchors, accepted, ...}, intervals,
                                          maximum_overlap) with every phone clock re-derived through the studio track
                                          (song.json cross_check_detail.<cam>.studio_derived_clock; cam06 identity) +
                                          clock_source, studio_fit, provenance, deltas_vs_alignment_json, validation.
                                          AUDIO clocks like alignment.json (no A/V latency, no sound travel): every
                                          phone's audio within about +-1-2 ms of cam06's (onset-envelope sense; the
                                          sub-ms fit figures are precision, not capture-time accuracy). They leave
                                          per-phone VIDEO offsets of tens of ms (predicted -45..+92 ms, cam04 worst,
                                          validation D) for sync.py to measure. Used by takes/larissa_kadu_38.toml
                                          ([sync] required = true: mamma prepare refuses to build fit inputs until
                                          that take's own sync/offsets.json exists). build / validate --write-back
                                          refuse the base take's alignment file and any file clock_studio did not
                                          build; alignment.json is never modified
  clock_studio/validation.json          [clock_studio validate] per window: (A) phone vs studio track through either clock
                                          set, (B) phone-pair discrepancies solved per phone (no studio track), (C)
                                          envelope peaks at strong studio onsets, (D) expected per-phone VIDEO offsets
                                          on the new clocks (base take's sync.py deltas minus the clock change);
                                          review/clock_studio_validation.png
  clock_studio/window.json              [clock_studio window --take] the take window vs every camera's video
                                          coverage {frame_times: reference | sync_corrected, margins_ms, current
                                          {inside_every_camera, short_cameras, trimmed}}; sync-corrected once
                                          sync/offsets.json exists (re-run after sync; trim the window if told)
                                          + decoded {per_camera first/last decoded frame, grid {t0, t1, T, on_lattice},
                                          frames_lost_vs_window, limited_by}: the grid mamma prepare builds (ingest
                                          decodes only frames inside the window, so T is 1-2 below the window's and
                                          t0 is off the 1/30 s lattice; larissa_kadu_38 before sync: 30.0636-68.3969 s,
                                          T 1151, not 1153; prepare.json grid.trim is authoritative)
                                          + time_base: beats/audio_overlap.wav are on cam06's AUDIO clock, poses on
                                          sync.py's median-gauge VIDEO time; pose - beat = cam06's sync delta (measured
                                          after sync; predicted -44 ms) assuming cam06's own A/V latency ~0. timing /
                                          export must record or remove it before reading beat offsets under ~50 ms
  keyframes/reused_from.json            [clock_studio reuse-keyframes] keyframes + people_masks_kf COPIED from another
                                          take of the same recordings, t re-mapped through this take's clocks (file names
                                          keep the source take's ms labels); then ingest --window-only and
                                          detect --no-keyframes (plain detect re-detects keyframes and overwrites them)

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
                                          suspect?, beat_t, count, measure, async_ms, ...}], touchdown_method.<role>
                                          {template_feet.<foot>{offset_ms, anchor_ms, offset_sd_ms, at_search_edge,
                                          touchdown_minus_stance_ms}, n_template, n_fallback, n_clamped,
                                          suspect_feet [{foot, reasons}] (median touchdown - first stance frame
                                          outside plausible_lag_ms: do not trust that foot's times), mc{draws,
                                          per_foot common_shift_sd_ms}},
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
  timing/{timing.json,kinematics.npz,qa_timing.png}  [timing_free] BEAT-FREE variant for a take without a soundtrack (larissa_kadu_recap):
                                          same file names and the keys that still mean something (footsteps {role}: [{t, foot,
                                          method, sigma, detect_rate, stance, ...}] with beat_t / count / measure / async_ms
                                          absent, accents, kinematics, touchdown_method), plus `beatless: true`, `beats` = {grid:
                                          null, tempo_bpm: null, skipped: true, reason}, `skipped` [what was NOT computed: beat grid,
                                          asynchrony / hit rate / circular statistics, surrogate and permutation tests, measure
                                          effects, absolute timing], `absolute_timing.valid` false, asynchrony {} and `step_timing`
                                          {role: {n_touchdowns, interval_all / per_foot {n, median_s, p10_s, p90_s, ...}, bouts [{t0,
                                          t1, n, cadence_hz}] (runs of touchdowns <= bout_gap_s apart), share_in_bouts, longest_still_s,
                                          steps_per_window [{t0, t1, n}]}, couple: {lead_follow_offset {n_pairs, median_ms, mad_ms,
                                          ...}}}. Analysed in 30-s windows with 3 s of context (touchdown_method.<role>.windowed);
                                          kinematics.npz is joined at the cores (cutoff_hz / snr = medians over the windows, reliable =
                                          in >= half of them). export_unity reads it like the beat version (no zouk-time-analysis.json)
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
                                          info.stature_rel_diff (with --height-*) = mesh crown-to-sole / (stature +
                                          shoe sole) - 1 since 2026-10-06 (same convention as world_scale check;
                                          info.stature_sole_m = the take's shoe_sole_m, default 0.025); warning > 5 %
                                          -> since the scale review round 2 (2026-10-06): warning > info.stature_tol
                                          = 1.5 % (world.SCALE_AGREE_REL; 5 % missed the follow's +4.25 %), also
                                          printed by `python -m dancecap physics`
  physics/qa_physics.png, validation.json [physics] QA plot; synthetic checks + gate G6 per dancer / noise seed,
                                          incl. 6 s windows (windows_6s) clean and noisy with the partner LP
  review/timing_asynchrony.png, timing_accents.png, physics_overview.png, analysis_summary.json
                                        [analysis review] `python -m dancecap.analysis_review <take>` (read-only on
                                          timing/, physics/, smplx/): step asynchrony vs zouk count; accents (whole-body
                                          coincident events ringed, 4.5-12 Hz root-jitter strip); vertical pulse =
                                          peaks of F_net,y/mg, phase on the eighth grid in the longest run of eighths
                                          where both dancers pulse once (summary.pulse.steady_bounce) + follow-minus-lead
                                          (paired, xcorr); COM top view, couple GRF, beat impulses, XCoM margin.
                                          Real takes: run timing with mc_corr_noise_m=0.01, mc_corr_time_s=0.1 (stated
                                          assumption; the 4th-difference noise of a fused fit is 0.5-1 mm, a lower bound)

  avatar/{lead,follow}_albedo.png       [texture] photoreal albedo, 2048x2048 sRGB 8-bit RGB, on the SMPL-X UV layout (vt/ft of
                                          the model npz; identical for MAMMA's locked-head and the plain neutral model).
                                          UV convention = Unity's: PNG row 0 is v = 1, texel centre u = (x+0.5)/S,
                                          v = 1 - (y+0.5)/S. Islands are padded by a 16-texel gutter (mip-safe). Lighting
                                          is baked in (no de-lighting); hair is not part of the texture except the
                                          follow's scalp (hair colour). Likeness data: work/ or git-ignored StreamingAssets only
                                          Hair (2026-10-06, avatar_hair.py; TextureConfig hair_mode "exclude" + blend
                                          "multiband" = default, "soft"/"mean" = the previous bake): her hair mask + strongly
                                          hair-coloured pixels next to it, dilated (r0 = 0.6 % of her mask height >= 2 px, ramp
                                          to full weight over another 0.6 %), get weight 0 on every non-scalp texel; the scalp
                                          under the hair = the ANATOMICAL scalp (avatar_hair.scalp_vertices: head minus face
                                          minus ears minus everything below the eye line in front of the ear; 98 % inside
                                          FLAME's scalp mask) with only its hairline learned: texels > 1.5 cm inside it are
                                          always scalp, the 1.5 cm band on either side of its boundary follows the samples'
                                          hair evidence (outside only above the eye line, never the face core / ears), so
                                          hanging hair over a cheek / ear / jaw / neck is never scalp (those texels are body:
                                          hair excluded, filled from skin). Before the fill (fill_guards) measured upper-body /
                                          neck / non-scalp head texels that are still DARK hair-coloured (red_cast, L* < 45)
                                          are dropped (leaked hair; counts in report roles.follow.fill_guards), hair-coloured
                                          or mostly-hair-seen (evidence > 0.5) texels never seed a fill (no mirror copy, no
                                          class seed, a red k-means class seeds nothing); scalp and body are filled
                                          separately, body holes per colour class (white top / skin / ...), neck / nape / face
                                          / ear holes from the skin class only. Registration passes copy her dilated hair
                                          pixels from the photo into the render and give them zero flow weight
                                          (avatar_texture.hair_flow_mask); per-camera layers fused in two bands (high: sharp
                                          weights, face ^4, smoothed over ~2 texels; low, sigma = 1.2 % of the atlas: the same
                                          weights low-passed inside each UV island x hair group) so exposure / white-balance
                                          (face texels keep the one-band sharp fusion, feathered ~4 texels: face_sharp_fusion)
                                          steps at view boundaries blend over centimetres. Her face: the
                                          dancers-stage hair split often covers warm-lit face skin, so inside her projected
                                          face (front-facing SMPL-X face triangles, FaceProjector) hair = mask AND a face-aware
                                          colour model (her face pixels as skin seeds) >= 0.8, other face pixels get a relative
                                          (1 - p)^2 weight, face-core texels where the previous pass saw hair over the face
                                          take a lighter weighted luminance quantile as the inlier reference (0.5 -> 0.75 as
                                          that evidence rises to 0.3; clean face texels keep the median) and a face texel still
                                          fused dark hair-coloured is refilled (mirror first). report hair_model.<cam>.face_aware.
                                          Off-body hair (hair_offbody, default on): in the bake, her full-mask pixels
                                          outside her posed model's silhouette (dilated 3 % of her mask height) that connect
                                          to her hair are hair whatever their colour (the motion-blurred outer part of a hair
                                          whip that the dancers split labels body); a view whose off-body hair is >= 5 % of
                                          her mask (a whip) multiplies both exclusion radii by 3 (hair_whip_mult), so smeared
                                          hair beside her neck / shoulders / arms is excluded too. Counts in report.json
                                          passes[].hair_offbody {views, whip_views, px_median}.
                                          View times (2026-10-07): every view poses the bodies at the frame's SYNC-CORRECTED
                                          time (frames.json t + sync delta, avatar_texture.camera_views / view_time_offsets;
                                          view t_frame = the frames.json time), like the fit's grid; before, a camera with a
                                          large delta (larissa_kadu_38 cam01 +90 ms, cam04 +107 ms) was sampled through a
                                          body ~3 frames off. avatar_head face observations and review.render use the same
                                          times; the pod inputs carry sync/offsets.json.
                                          hair_mode exclude refuses to bake when < 90 % of the follow's views (with her mask)
                                          have a real dancers/<cam>/hair mask (--allow-missing-hair-masks only warns); counts
                                          in report hair_masks.follow {mask, derived (full - body), none, real_fraction}
  avatar/follow_hair_region.png         [texture] S x S uint8 on the UV layout: 255 = scalp under her hair (feathered edge),
                                          the region the bake gave the hair colour = anatomical scalp + learned hairline (never
                                          cheek, ear, jaw, temple below the eye line, face core); where the Unity hair cards /
                                          strands root
  avatar/follow_hair_cover.png          [texture] S x S uint8 on the UV layout: share of the sample weight that saw her hair
                                          over the texel (255 = always hair in the photos, 0 = never / unseen) = where her
                                          HANGING hair covered body / face / neck in the capture. NOT a root region
  avatar/follow_hair.json               [texture] hair colour sample for Unity hair (exposure-normalised like the albedo):
                                          palette [{srgb8, linear, fraction}] (k-means, dark -> light), gradient [{t (0 root
                                          .. 1 tips), srgb8, n}], root/mid/tip/mean/median_srgb8,
                                          hair_extent_from_head_centre_m_p95 (image spread of the hair pixels from the head
                                          centre at head depth: NOT the strand length - hair_reference.json measures that),
                                          scalp_albedo_mean_srgb8, n_pixels, n_views, strip
  avatar/follow_hair_strip.png          [texture] 256 x 512 sRGB strand strip: row 0 = root, last row = tips, columns =
                                          strands (luminance quantiles of the measured hair pixels, fixed shuffled order)
  avatar/{lead,follow}_coverage.png     [texture] per texel R = measured-data alpha (255 = fully from cameras), G = inlier
                                          cameras x 32, B = fill source (0 measured, 128 left/right mirror, 255 diffused
                                          over the mesh); black outside the UV islands
  avatar/{lead,follow}_texels.npz       [texture] per covered texel (atlas order of avatar_uv.build_atlas): rgb_srgb (N,3)
                                          f16 before gutter, alpha, conf, ncam, source, view_gain (n_views,3) linear gains,
                                          hair_region (N,) u8 (follow only, 255 = scalp under hair), hair_cover (N,) u8
                                          (follow only, as follow_hair_cover.png)
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
                                          samples, flow), review_metrics (render-vs-photo MAE per camera); also host (pod,
                                          platform, GPU), smplx_freshness, t_window (exact bake window), peak_gpu_gb,
                                          per-role median_view_iou, texture_metrics (% hair-coloured texels per region:
                                          chest/shoulders/back, chest, back, shoulders, neck, face_core, forearms baseline;
                                          avatar_hair.red_cast = CIELAB a* > 9 and a* - b* > 3; hair_residue_pct =
                                          avatar_hair.hair_residue: also desaturated / darkened hair on a light top or
                                          skin, a* - b* > 1.5 and a* > 6.5 and L* < 72, or red_cast and L* < 45 - the pink
                                          print on her top's front counts too, compare the back), follow hair_sample;
                                          hair_region [{pass, role, head/prior/face/face_outside_core/ear hair fractions,
                                          outside_scalp_hair_fraction_of_head, evidence_above_thresh_outside_allowed_head_
                                          fraction, leak_guard_dropped(_pct_of_measured), bad_seed_texels}], hair_masks,
                                          roles.follow.fill_guards (leak guard counts, fill classes, skin class)
                                        config.body_ref_q (CLI texture --body-ref-q, default 0.5 = off): the hair roles'
                                          upper body / neck texels (not scalp) use a lighter luminance quantile as the
                                          fusion's inlier reference, ramped 0.5 -> body_ref_q with the previous pass's hair
                                          evidence (as skin_ref_q on the face core): hair over her white top / skin only
                                          darkens or reddens it (larissa_kadu_38 avatar_fix: 0.7)
                                        Freshness rule: the bake refuses smplx/ older than cameras/world_from_sfm.json (a fit
                                          from a previous world frame) unless allow_stale (--allow-stale; the smoke preset
                                          allows it and stamps "STALE smplx" on its review sheets).
                                        Presets (avatar_texture.PRESETS): smoke (512^2, step 0.25 s, 1.5 s window, -> avatar/
                                          smoke/ + review/avatar_smoke_*.jpg, same file names inside), local (2080 Ti default:
                                          2048^2, step 1/15 s, top-k 4, 1 registration pass), pod (step 1/30 s, top-k 6, 2
                                          passes). View frames = first frame of every step slot of ABSOLUTE reference time
                                          (window-independent, so a pod bundle is a superset of any run's frames).
                                          View loading: load_threads (local 4, pod preset 12) with a bounded prefetch; each
                                          view's data is released once sampled (it used to stay alive for the whole pass:
                                          ~13 MB per 1080p / ~66 MB per 4K view, OOM on long takes).
                                          Visibility/renders: avatar_raster.py exact pure-torch triangle z-buffer (CPU/CUDA,
                                          Windows/Linux; no nvdiffrast build)
  avatar/pod_job.json, avatar/pod/      [texture] Runpod job (python -m dancecap texture <take> --write-pod-job | --pod;
                                          avatar_pod.py): inputs (texture_inputs.tar.gz = code + only the frames/masks the
                                          bake and hair model read + cameras/<cam>.npz + people/ + roles.json, with sizes;
                                          smplx.tar.gz + SMPLX_NEUTRAL.npz for standalone), pod_commands (extract / setup /
                                          run / pack), chain_after_mamma (same pod session after `mamma select` promoted
                                          smplx/: uploads texture_inputs only), standalone, outputs, estimate (views, passes,
                                          bake_min from the measured A100 rates, max_minutes_recommended), code_path
                                          (hair_mode, blend, step_s: the run command passes --hair-mode/--blend explicitly;
                                          hair masks of EVERY view frame are bundled; --step-ms overrides the preset step;
                                          --hair-mode / --blend are passed through); max_minutes defaults to
                                          estimate.max_minutes_recommended (bake estimate x 1.3 + 3) and a smaller explicit
                                          --pod-minutes is REFUSED (write_pod_job / run_pod raise before any pod exists);
                                          estimate.s_per_view = the per-view rates used (hair-aware path re-measured on an
                                          A100-SXM4 pod 2026-10-07 with 12 loader threads: 0.061 / 0.114 s/view; old path
                                          0.031 / 0.089), see bake_min_basis;
                                          --pod sessions pass --max-cost (default $10) to remote.pod_session;
                                          pod/pod.log,
                                          pod/result.json, texture_out.tar.gz (extracted into work/<take>/); avatar/
                                          pod_texture.log written on the pod. Pod workspace = /workspace/HeadMovement (same
                                          layout as the MAMMA pod job)
  review/avatar_texture_{lead,follow}.jpg, avatar_uv.jpg, avatar_turntable.jpg, avatar_face.jpg   [texture] QA:
                                          re-render into real frames (photo | avatar | overlay), atlases + coverage,
                                          virtual turntable, face close-ups (photo | avatar) where each face is seen best
  review/avatar_texture_follow_before_after*.jpg, avatar_uv_before_after.jpg, avatar_turntable_before_after.jpg,
  avatar_face_before_after.jpg          [texture review] python -m dancecap.avatar_hair compare <take> --before DIR
                                          --after DIR --review DIR [--heldout-cam C] (avatar_hair_review.py, read-only on
                                          the bakes): held-out photos (frames the bakes never sampled / a left-out camera)
                                          photo | before | after with body-pixel MAE (gain-fitted) and false hair-coloured
                                          pixels; follow atlas crops; turntables; face close-ups; avatar_hair_region_head.jpg
                                          (her head from 5 angles: before / after / scalp region green + hair cover blue).
                                          --json: summary {before_/after_ mae, mae_gain, mae_upper, mae_gain_upper, mae_face,
                                          mae_gain_face, false_hair_pct (raw AND gain-fitted, pixel-weighted), per_cam {same},
                                          worse_views {metric: worse/better/n}, registration (per-view flow p90 before vs
                                          after), gate (leave-one-camera-out only: no upper-body regression on the held-out
                                          camera, raw and gain-fitted, tolerance 0.1 sRGB levels; passed_all_pixels and
                                          passed_hair_free_pixels = only body pixels the photo does not show hair-coloured,
                                          because the hair masks miss strands and a texture with hair baked on matches
                                          those; passed = both), *_nh / *_hc metrics (hair-free / hair-coloured photo
                                          pixels), upper_photo_haircol_pct, texture_metrics, hair_masks}.
                                          Leave-one-camera-out is the primary test (the frames-between-step-slots set sits one
                                          frame from baked frames)

  avatar/hair_reference.json            [hair_reference] python -m dancecap.hair_reference <take> [--step N] (likeness data,
                                          local only), version 2: scale (true_per_world = stature + sole over the follow's
                                          rest height), compute (device, GPU lock wait / peak / reserved-after-release,
                                          threads, step timings, colour views), shape {frames (+ left_out with reasons,
                                          registration: views at the shift limit / carve-only per camera),
                                          landmarks_rest_m_above_sole_world, length (back tip level on the REST mesh: hull /
                                          single_view_raycast / reference mean, tips_fall, strand_length_crown_to_tip_back
                                          world_m/true_m; tip_level_front_rest_m = LOWER bound of the front length;
                                          posed_vertical_extent_below_crown), volume (5 cm band profiles below the crown:
                                          width / front_back_extent / back_layer_standoff, upper bounds), motion (spread r95,
                                          an upper bound in turns; trailing {status resolved|not_resolved, rule, checks,
                                          head/torso: calm_offset_deg, raw/corrected medians, CI, by_turn_direction}; v1 keys
                                          trailing_angle_deg_turns.median and lag_time_s.phase_alignment are null unless
                                          resolved; yaw)}, hull_consistency (re-projected 3D hair vs 2D hair masks),
                                          colour {what (OBSERVED, white-balanced, includes shading: not an albedo), video /
                                          portrait palettes: root, mid, tip, highlight, shadow as linear_srgb, srgb8, hex,
                                          hsv, lab; video.image_stats p10/p50/p90; video.camera_spread (L*, hue, saturation,
                                          dE to consensus); per camera; video_vs_portrait}, observations
                                          (hair_ref/observations.json, by eye), tip_bloom (the original head-movement recipe
                                          parsed from the repo; vertex_colours_8bit model vs measured; soft_knee),
                                          acceptance (colour = image statistics + tolerances; no trailing/lag criterion
                                          unless resolved: not_checked), unity_notes; docs/HAIR_REFERENCE.md
  hair_ref/                             [hair_reference] hull.npz (k, P = world hair voxels, per-frame view diagnostics),
                                          consistency.json (re-projection check, keyed to hull.npz size+mtime),
                                          colour_samples.npz (per colour view: cam, i, white, hair pixels as 8-bit BGR, zone,
                                          d; keyed to the views + settings; --reuse-colour), hull_compute.json (GPU child),
                                          mp/<cam>/<frame>.png (MediaPipe SelfieMulticlass crops, B = hair), mp_portrait.png,
                                          motion.png, observations.json (written by a person), pod_job.json (--write-pod-job:
                                          dancecap.pod_job/1 job "hair_ref", hull + check + colour + MediaPipe on a pod, only
                                          the sampled frames uploaded; after_download = local --reuse-hull --reuse-colour
                                          re-render)
  avatar/hair_groom.json                [hair_groom] python -m dancecap.hair_groom <take> (CPU ~3 min; likeness data, work/ and
                                          git-ignored StreamingAssets only), version 1: model {path, fit_model, rest_height_m,
                                          head_joint_rest, head_sphere_centre_unity, head_radius_m, crown_rest_y, scalp_area_m2,
                                          hairline_el_deg_vs_abs_az}, strands {guides (<= 200), segments 16, children,
                                          child_spread_m, tip_clump, width_root_m, width_tip_m}, guides [{vid (3 SMPL-X vertex
                                          ids of the root triangle), bary, len (m, fit world), pos_unity / normal_unity (rest,
                                          relative to the head joint, Unity axes = head-bone space), az, el}], shape
                                          {length_source (hair_reference.json when its rest height matches, else the 2D
                                          masks), tip_level_back/front_rest_m, crown_strand_m, length_m, standoff_m, part_x_unity,
                                          measured_2d (hair masks vs the PROJECTED SMPL-X head: drop / reach / width / area
                                          per calm / turn view)}, colour {root, mid, tip, shadow, highlight (colour_entry:
                                          linear_srgb, srgb8, hex, hsv, lab), strand_palette_linear, vs_hair_reference_dE76},
                                          render {tip_emission (the original bloom curve), kajiya_kay}, colliders [{name, a, b
                                          (SMPL-X joints), oa_unity, ob_unity (rest offsets in the bone frame), r}] (head
                                          sphere, neck, shoulders, chest, midback; no arms), dynamics {substeps, iterations,
                                          damping_per_s, bend_compliance, shape_stiffness_root (fraction back to the rest
                                          shape per 1/30 s), shape_falloff, friction, particle_radius_m, targets (turn spread
                                          r95, trailing angle, lag from hair_reference)}, motion {head_speed_dps,
                                          fastest_head_moments_reference_t}. Copied by the same command (or --export-only
                                          [--capture]) into StreamingAssets/<capture>/hair_groom.json and work/<take>/export/
                                          <capture>/ for every capture whose provenance.take is the take - re-run after every
                                          `python -m dancecap export` (that rewrites the capture folder)
  review/hair_groom.jpg                 [hair_groom] scalp + roots + combed-down strand lengths on the rest head (front /
                                          side / back), colour swatches, 2D measurement overlays (hair mask, projected head)
  review/hair_peak<k>_<view>[_noglow].png, hair_shots.json, hair_sheet.jpg   [unity hair] head-movement
                                          Tools/playtest_hair.ps1 shots of the fastest head moments (back / side / front34,
                                          tip bloom on / off) + sweep numbers; `python -m dancecap.hair_groom <take> --sheet`
                                          composes the labelled hair_sheet.jpg
  review/hair_reference.jpg, hair_reference_hull.jpg   [hair_reference] reference sheet (portrait + video crops with
                                          the tip level, calm-frame hair frequency back / side with REST landmark lines and
                                          the dashed POSED calm tip level, turn top view with the trailing verdict, motion
                                          plot (raw vs calm-offset-corrected trailing), observed palettes + image p10/p50/p90,
                                          original vs target tip bloom (continuous E(s)), numbers with world / TRUE units);
                                          3D hair hull re-projected into every camera (visible / hidden voxels, recall /
                                          precision; drawn on the CPU)

  face/survey.npz, survey.json, jobs.json   [face_survey] per camera frame and dancer, from the fitted SMPL-X face
                                          landmarks (iBUG 52 = chin + 51 static, face_common): <role>_<cam>_{i,t,iod (outer
                                          eye corners, px), face_px, angle/yaw/pitch (deg, 0 = frontal), vis, partner, hair,
                                          own, ok, inimg, base (usable), lmk (F,52,2)}; <role>_grid_nviews_iod{10,20,30} on
                                          the 30 Hz grid; jobs.json = crops for the detector worker (iod >= 5 px, < 80 deg)
  face/detections.npz                   [face_worker, .venv-face] aligned with jobs: MediaPipe Face Landmarker (mp_ok,
                                          mp_lmk (N,478,3) frame px, mp_blend (N,52) + mp_blend_names, mp_M) and RTMW-x
                                          (rtmw_face (N,68,2) iBUG order, rtmw_score); portrait_<role>.npz likewise
  face/eval.json                        [face_eval] detection rate / NME / jitter per IOD bin, cross-camera agreement
  face/smirk.npz, degrade.json, smirk_agreement.json   [face_smirk, gated FLAME 2020] per job SMIRK expression (50),
                                          jaw (3), eyelids (2), mouth metrics (mm), smplx_expression (20) via the
                                          SMPL-X<->FLAME vertex correspondence (+ relative transfer residual)
  face/expr_<role>.npz, expr.json       [face_expr --research] per 30 Hz grid frame: expression (T, n_expr) SMPL-X,
                                          expression_rest, jaw_pose (T,3), neck_delta_rotvec, neck_pose (= body_pose[:, 33:36]
                                          corrected), n_obs, confidence (0 = no observation); expr.json = in-sample +
                                          leave-one-camera-out. Research only: never adopt its neck_pose
  face/detections_shard<i>of<n>.npz     [face_worker jobs --shard i/n] partial detections (+ job_index); `face_worker merge
                                          <take> <n>` -> face/detections.npz (dancecap.face_pod runs n shards on a pod)
  smplx_face/<role>.npz                 [face_expr] THE face block for the avatar, on the smplx/<role>.npz time grid t (T):
                                          jaw_pose (T,3) axis-angle of SMPL-X joint 22 (x = opening only), expression (T,
                                          n_expr=10) SMPL-X expression coefficients, confidence (T,) 0..1, expression_rest
                                          (adopted; 0 = neutral), shipped = the adopted hold-out candidate (neutral | rest |
                                          rest_closed | final_expr | final) refitted on all views - exactly the recipe the
                                          hold-out scored (2026-10-07; rest_closed = a resting expression that keeps the
                                          neutral lip gap), expression_rest_fitted (+ rest_fitted_kind,
                                          expression_rest_free / _closed, expression_anchor_perframe), jaw_rest[_fitted]_deg,
                                          adopted_rest / adopted_expression / adopted_jaw (hold-out model selection: a layer
                                          that does not beat the simpler one on held-out cameras / time blocks, or raises the
                                          held-out mouth-opening error beyond max(1.5 x, + 0.005), is replaced by it; the
                                          per-frame layer must also not lose on co-visible held-out views when there are
                                          >= 5), expression_captured / jaw_captured_deg (before gating, diagnostics), eyelid
                                          (T,2) [left, right] weights 0..1 = PROCEDURAL blinks (stylisation, seeded per take
                                          + dancer), blink_starts, leye_pose / reye_pose (the body fit's, unchanged), n_views
  smplx_face/face_blendshapes.npz       [face_expr] vertex_ids (Vh,) SMPL-X vertices that move, names (S,) = expr00..09 +
                                          eyelid_left/right (eyelids only when the gated FLAME files are present), deltas
                                          (S,Vh,3) f32 metres in the model's rest axes (y up, +z forward; Unity (x, y, -z)),
                                          i.e. vertex v of the dancer's rest mesh moves by sum_s w_s deltas[s] before LBS.
                                          SMPL-X / FLAME-derived: local + StreamingAssets only, never committed.
                                          Unity (IMPLEMENTED 2026-10-07, dancecap.face_export; also run by export_unity
                                          export when smplx_face/ exists): smplx_<role>.bin joints 22-24 = jaw_pose /
                                          leye / reye of smplx_face/<role>.npz (same grid, refused otherwise);
                                          smplx_<role>_face.bin (magic DCXF, blob layout as DCXS): blocks vertex_ids (Vh,)
                                          i4 (SMPL-X ids; map onto the split skin through its vertex_ids block), deltas
                                          (S,Vh,3) f4 Unity axes (x, y, -z), weights (T,S) f4 = [expression (T,n_expr),
                                          eyelid (T,2)], header names / stylised / face (shipped, adopted_*);
                                          smplx_<role>_skin.bin + the mouth cap (30 triangles appended, header
                                          "mouth_cap": closes SMPL-X's one open boundary, the mouth bag, so the lip gap shows
                                          the painted interior); <role>_albedo.png = the chosen variant (face = portrait look
                                          default, videolook, bake; a face texture older than the bake falls back to the
                                          bake); capture.json + smplx_face {role: file}, smplx_albedo_variant, provenance.faces.
                                          head-movement Assets/Avatar/SmplxFace.cs (HeadMovement.AvatarsLoaded) adds the S
                                          blendshape frames and sets SetBlendShapeWeight(s, 100 w) per frame (Strength,
                                          Blinks toggles; expression weights are signed and may exceed 1: keep
                                          legacyClampBlendShapeWeights off)
  smplx_face/report.json                [face_expr] limitations, per role: views, shipped, shipped_heldout (the shipped
                                          candidate's own hold-out numbers), rest_fits, adoption + reasons, cross-camera
                                          (+ covisible) and time-block hold-out scores (aligned mouth NME % IOD, mouth
                                          opening / width correlation) per candidate, in_sample (neutral / rest /
                                          rest_closed / shipped)
  avatar/<role>_albedo_face.png, avatar/<role>_albedo_face_videolook.png   [avatar_face] next to the bake it was built
                                          on (`--bake ROLE=PNG` for any 2048 SMPL-X UV albedo: <stem>_face.png and
                                          <stem>_face_videolook.png beside it; the input is never overwritten): portrait look
                                          and video-matched look (face/look.json targets relative to the cheeks, measured
                                          on the texture: follow lips = the portrait look's lip contrast x the makeup-free
                                          ratios, lead chin dL = the video's scaled to full resolution); the user picks.
                                          Steps (postprocess): composite, follow: refill_face_base (portrait over its own
                                          skin extrapolation, not the hair-covered bake) + skin repair, head L/R evening,
                                          gradient-domain clean-up (follow repair / portrait border; lead scalp band with
                                          the bake seam), follow lip colour (mesh lips take the portrait's lip-vs-cheek
                                          contrast), mouth interior + slit, lip / beard streak removal, eyes, UV island
                                          padding. avatar/face/<role>_texel_masks.npz: labels (lips 1 / beard 2 / cheek 8 /
                                          mesh lips 16), repaired, alpha per atlas texel; report.json: seams (vertical seam /
                                          portrait / repair border luma steps per stage and for the bake and a previous
                                          face texture avatar/face/prev_*/), likeness (gain-fitted face MAE / NCC vs held-out
                                          video crops, posed and landmark-aligned)
  avatar/face/<role>_albedo_face.png    [avatar_face] legacy copy of avatar/<role>_albedo_face.png (default bake only)
                                          = the bake with the portrait face composited (colour-transferred to the
                                          bake's lighting, multi-band blended) + procedural eyeballs; <role>_face_layer.png
                                          RGBA (alpha = portrait weight) to re-composite onto a re-baked albedo
                                          (python -m dancecap.avatar_face <take> --composite); <role>_portrait_uv.npz
                                          (head texels, rgb, w), <role>_portrait_fit.npz, <role>_eyes.json, report.json.
                                          Portraits: <source_dir>/<name lower>.jpg or [dancers.<role>].portrait (likeness
                                          data: never in a repo)
  review/face_*.jpg|png, avatar_face_portrait.jpg, avatar_face_uv.jpg   [face stages] survey, detectors, degrade test,
                                          portrait fit, expression fits, SMIRK vs landmark fit; face_expr.jpg (video | avatar
                                          today | with the face block, close-ups, face-block parts) + face_expr_confidence.png
                                          [face_expr]; avatar_face_portrait.jpg (reference | old | new | video look at 3 head
                                          angles), avatar_face_turntable.jpg, avatar_face_video.jpg, avatar_face_likeness.jpg
                                          [avatar_face; `--review-only` redraws them from the written textures]

  RETIRED 2026-10-07 by the user: hands refinement (hands_survey / hands_pod / hands_build, S3_hands) - the pipeline no longer runs it and nothing reads the hands/, smplx_hands/, smplx_detail/ files below (historical).
  hands/wrist_gate.npz + .json          [refine_hands build = hands_detail] per hand <role>_<side>_wrist (T,3) local
                                          wrist axis-angle used as the finger IK's start / prior, _weight (T,) share of
                                          the MANO regressors' (WiLoR + HaMeR) fused wrist (0 = MAMMA's), _disagree_deg,
                                          _det_views; json: per hand adopted + why + the independent keypoint judge,
                                          frames_bridged / bridged_share (gated frames whose regressors disagree > 30 deg:
                                          slerped, never across > 90 deg), contact_check (weight cut where the gated hand
                                          would sink > max(MAMMA's sink, 1 cm) + 1 cm into the partner)
  hands/smplx_fingers_build/, fingers_build*.{json,npz}, contact.{json,npz}, smplx/   [build] fingers (hybrid IK, gated
                                          wrists) and the contact refinement from them; contact.json refined[] carries
                                          the guard (guard, adopted, attempts[]: arm + hand landmarks all cameras and per
                                          camera, shoulder / elbow turn, wrist turn beyond its envelope (only back towards
                                          MAMMA's wrist), wrist / elbow jitter, palm displacement vs its cap, finger
                                          crossings, arm penetration); an episode whose every attempt fails keeps its
                                          input pose (refined 'no (guard failed ...)'); merged hand-hand holds carry
                                          attract_runs (both records + bridged gaps), bridged_gaps, union, partner_side,
                                          and the twin record merged_into. Gaps: exact point-to-triangle distance, inside
                                          = generalized winding number of the partner's closed body (mesh_contact)
  hands/metrics.json + metrics.npz      [build] MAMMA (smplx/) vs fingers vs final: keypoints[<role>_<side>][judge]
                                          [stage][cam|all] {n, px, len, shape_len, hand_px} (judges fused keypoints /
                                          hamer / wilor 2D joints), ma2d (MAMMA's 63 hand landmarks, px), contact[]
                                          per hold/touch episode {<stage>: gap cm p5..p95, touching_within_1cm,
                                          float_gt_1cm, apart_gt_2cm, sunk_gt_1cm, penetration_cm_max / _p95,
                                          float_cm_max}, holds[] (hand-to-hand distance on attracted and bridged frames),
                                          penetration, arm_penetration, finger_crossings (frames, max_pairs, worst_t,
                                          top_pairs), palm_vs_mamma_cm, jitter (mm residual from a local cubic + shoulder
                                          / elbow / wrist angular acceleration, wrist and fingertip acceleration); npz
                                          '<stage>|<role>|<side>|gap|pen|arm_gap|arm_pen|cross' (T,), '<stage>|hold<n>|
                                          hand_dist' (T,) m
  smplx_hands/{lead,follow}.npz         [build] the smplx/ contract (same keys, FK joints) with refined fingers, gated
                                          wrists and contact-refined arms + additive keys: hands_refined (T,) bool,
                                          wrist_gate_weight (T,2) left/right, contact_state (T,2) int8 0 free 1 touch
                                          2 hold (both hands of one merged hold share it frame by frame), contact_gap_m
                                          (T,2) signed hand-to-partner gap (negative = sunk in), contact_episodes (json
                                          str), hands_source (json str), quality_source. quality, quality_parts,
                                          reproj_parts_px, reproj_px, parts are RECOMPUTED for this fit (MAMMA's
                                          definitions, ma_2d landmarks; dropped when there are none); repaired, suspect,
                                          reproj_px_raw, contact (ma_3d's 512-point contact evidence), floor_contact are
                                          MAMMA's unchanged
  smplx_detail/{lead,follow}.npz        [build] smplx_hands + face/expr_<role>.npz: jaw_pose, expression (T,n_expr)
                                          f4, expression_rest, n_expr, face_confidence; neck only when the face npz
                                          says adopt_neck (face_neck_adopted); face_source. Same schema as smplx/ (the
                                          Unity export reads jaw_pose; expression is carried for a blendshape path)
  review/hands_metrics.png, hands_render.jpg, hands_finger_crossing.jpg   [build, pod: no frames needed] the numbers as
                                          charts; rendered close-ups of the contacts (surface within 20 cm, sunk hand
                                          vertices yellow, sunk arm vertices orange); per hand the worst finger-crossing
                                          frame in each stage (crossing faces red)
  review/hands_contact.jpg              [refine_hands review <take> --video-only, local: needs frames/] video crops at
                                          contact moments (deepest sink, bridged hold gaps, largest moves; best 2-3
                                          visible cameras) before / after with hand skeletons + detections
  review/hands_contact_timeline.png     [contact] per hand: gap of the contact step's input fit, 2D evidence, score,
                                          episodes, gap after refinement

  room/bundle/                          [room prep, local] compact platform-neutral training bundle (~100 MB): manifest.json
                                          (version, prep_config = cams/bundle_max_side/win_step_s/holdout_*/mirror, inputs =
                                          mtimes of cameras/world_from_sfm.json, sfm/*.bin, <cam>.npz -> stale check),
                                          views.json [View: name images/<i>.jpg, mask masks/<i>.png, src = original frame,
                                          K/R/tvec in the SfM frame at the bundle resolution, k1 = 0 (undistorted), split
                                          train|test], geo.json (floor plane, mirror wall, reflectors, room box, world_from_sfm),
                                          points.npz (P, C, E, L of cameras/sfm). Train/quest/mesh/eval read ONLY the bundle
                                          (no pycolmap / frame store) so they run on Windows or a Linux pod
  room/pod_job.json, room/pod/          [room] Runpod job (python -m dancecap room <take> --pod | --write-pod-job): inputs
                                          (code.tar.gz, bundle.tar with sizes), exact pod commands (setup / train,quest /
                                          mesh,eval / pack), outputs, estimate; pod/pod.log, pod/result.json, room_out.tar.gz
                                          (extracted into work/<take>/); room/pod_{train,post}.log written on the pod
  room/smoke/                           [room] --preset smoke --out room/smoke: code-path check (own report, review/)
  room/                                 [room] static studio, all in the SfM frame (python -m dancecap room <take>;
                                          presets smoke | local | pod; report.json runs[] records preset, host, pod, GPU):
                                          splat_full.ply (standard 3DGS PLY: f_dc, channel-major f_rest SH3, logit
                                          opacity, log scales, wxyz rot; trained 'antialiased'), splat_quest.ply (<= 150k
                                          Gaussians, SH1), room_mesh.obj/.mtl (materials "room" -> room_mesh_tex.png
                                          xatlas atlas, "floor" -> room_floor_tex.png orthographic floor plane),
                                          room_mesh.ply (same triangles, vertex colours), eval.json (held-out masked
                                          PSNR/SSIM per camera, raw + colour-corrected; "view" = the original frame name),
                                          report.json (config, views,
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
  dancer_splats/bundle/                 [dancer_splats prepare, local CPU] one short window (takes/<take>.toml
                                          [dancer_splats] window, else the approved prototype window): manifest.json
                                          (version, prep_config, input mtimes + prep code sha1 -> stale check),
                                          views.json (crop K/R/t_w world frame, sync-corrected t, split train |
                                          test_cam | test_time, swap_prob, sharpness weight, camera type, IOD),
                                          images/<i>.jpg (undistorted crops), labels/<i>.png (0 bg, 1 lead, 2 follow,
                                          3 follow hair, 4 both, 255 ignore), bodies.npz (posed fit + global joint
                                          rotations + posed joints + rest mesh + top-4 LBS weights), init_<role>.npz,
                                          hair.npz (groom guides as rest polylines in the head frame, draped over the
                                          groom's capsules; head_centre_local = flip_z(head_sphere_centre_unity), the
                                          groom's offsets being from the head joint; colliders a, b, r, first_point +
                                          collider_oa / collider_ob in rest axes), eval/ (textured
                                          avatar renders of the held-out views + novel camera), novel.json
  dancer_splats/pod_job.json            [dancer_splats write-pod-job] dancecap.pod_job/1 (needs gsplat; train, eval)
  dancer_splats/                        [dancer_splats] avatar-guided dancer splats, world frame (right-handed, Y up,
                                          metres, floor y = 0): model.pt (Gaussian groups lead / follow / hair with
                                          their bindings - SMPL-X triangle + barycentrics / groom guide + segment + u -,
                                          cosine residual coefficients, hair node tracks, pose corrections, per-image
                                          colour transforms), pose_delta.npz (per grid frame and dancer: r (22 body
                                          joints, axis-angle rad, child side), dT (m), max_deg, dT_cm, hair node
                                          offsets; a QA signal - smplx/ is NEVER written), metrics.json (held-out
                                          metrics vs the avatar, gates 1-8; tables.splat_vs_avatar per group / camera /
                                          dancer and tables.crops = face / hair crops of cameras 02, 06, 09),
                                          report.json, progress.json, train.log, cache/ckpt.pt (deleted after a
                                          finished train; packed with the outputs after a failure = a retry's input);
                                          review_extra.json (after `review`); after `export`:
                                          canonical_<role>.ply (rest pose), frames/frame_<n>.ply (standard 3DGS PLY per
                                          window grid frame), splat_sequence.json (CaptureManifest.splat_sequence)
  review/splats/                        [dancer_splats eval] dancer_splats_t<t>_cam<c>.jpg (held-out real | splat |
                                          avatar | error), dancer_splats_heldout.jpg, dancer_splats_novel.jpg,
                                          dancer_splats_pose_delta.png (smoke runs: <out name>_*); [dancer_splats
                                          review, local] sheet_cam03_heldout.jpg / sheet_cam02_train.jpg (video |
                                          textured avatar | splats, 6 moments), turntable_t<t>.jpg, hair_closeups.jpg,
                                          face_closeups.jpg, temporal_strip_cam03.jpg, temporal_strip_novel.jpg
 head-movement StreamingAssets capture (see below)
```

## Camera types - `takes/<take>.toml [cameras."NN"]` (dancecap.camera_types, 2026-10-07)

The user's answer per source video: **mount** `handheld | tripod` (exclusive) and **zoom** `true | false` (independent:
a tripod can zoom), entered in the narration tool (camera panel under the video), the Pipeline Monitor wizard and its
Video Review. One table per two-digit source id, beside the reserved `[cameras]` keys `scale_rule`, `scale_reference`,
`loc`:

```toml
[cameras."05"]              # camera type (Pipeline Monitor / narration tool; block written by dancecap.camera_types)
mount = "tripod"            # handheld | tripod                 (user)
zoom = false                # true | false, independent of mount (user)
source = "user"             # user = confirmed | auto = suggestion, not confirmed yet
set_at = "2026-10-07T18:02:11Z"
# written by the probe, kept for the record; never overrides the user:
suggested = { mount = "tripod", zoom = false, pans = false, confidence = 0.64, by = "probe_a", at = "...", evidence = { parallax_px = 0.3, jitter_hf = 0.05 } }
pans = false                # tripod only, automatic
zoom_events = []            # automatic: [t0, t1, focal ratio] in reference seconds
```

| Rule | |
|---|---|
| Missing / unconfirmed | handheld + zoom = the general model = the solver before camera types; old takes are unchanged |
| Precedence | the user always wins; probe-B evidence measured AFTER the answer (confidence >= 0.7) that contradicts it -> CameraType `state: conflict` (camera_types gate), never a changed value; answering again acknowledges it. Evidence the track cannot judge (an attached camera: centre and focal fixed by the attach model) is never a conflict |
| Pre-fill | a measured 'tripod' below confidence 0.7 pre-fills handheld (the prior), evidence `mount_lean: tripod` |
| Effective type (camera stages) | policy `user` (default of the CLI, the pod jobs and the orchestrator): confirmed answers only; `auto`: + suggestions with confidence >= 0.7; `off`: none. `python -m dancecap cameras --camera-types`, or `DANCECAP_CAMERA_TYPES`. CameraType.effective is the `user` policy; each cameras run records its policy (report.json `camera_types.policy`, loc_<cam>.npz `loc_policy`) and the monitor's stale check compares under that policy |
| Applied (CameraType.model_check) | from report.json `cams.<cam>.camera_type`: `{declared, model, passed, why, at, current, refused}`; `refused` = the stage could not use the current declared type and kept a safer model: a warning next to the answer, summary count `refused`, a monitor alert - never a block |
| Writes | owned blocks only (header to next header, rewritten whole; comments typed inside are not kept), etag = `sha1:` + hex of the bytes (If-Match / 412), tomllib re-parse + everything outside the camera tables compared, backup `work/<take>/monitor/toml_backups/<YYYYmmdd-HHMMSS-mmm>-<take>.toml` (never overwritten), atomic replace. No tomlkit. Only `[cameras."NN"]` table headers are supported: a hand-written inline (`cameras = { "01" = {...} }`) or dotted (`cameras."01".mount = ...`) form makes every save refuse (nothing written) |
| Readers | `dancecap.camera_types` (`load`, `get`, `stage_types`, `effective`, `set_type`, `accept`, `clear`, `write_suggestion`); `Take.camera_type(cam)` / `Take.camera_types()` for stage code |

Effect on the camera stages (`cameras/typed.py`, applied by `stage_smooth` to the raw localisation; `stage_loc`
localises a LOCKED tripod (no zoom, no pans) on keyframes only - one frame pair every ~2 s + the map's frames - when one
constant pose fits them, else every frame; `front_pod` never splits a locked tripod into `loc_part`s, and parts always
localise every frame):

| class | model tried first | when the residual / drift check fails |
|---|---|---|
| handheld_zoom | per-frame pose + focal (unchanged code path) | - |
| handheld | per-frame 6-dof pose, ONE focal + k1 per video (joint robust fit) | per-frame focal (today) + check failed |
| tripod | ONE pose + focal + k1 for the whole video, every frame held | shared centre + per-frame rotation (pans), then the handheld model + check failed |
| tripod_zoom | shared centre, per-frame rotation + focal | today's model + check failed |

Fits: the joint (shared-parameter) fits use <= 300 correspondences per frame over <= 40-80 frames; the FINAL
per-frame solves (per-frame pose with a fixed focal, per-frame rotation [+ focal] with a fixed centre) use every
correspondence of each frame. Check: median over frames of the per-frame median static reprojection within
max(1.5 x, +1 px) of today's per-frame solution and p90 <= max(4 px, 1.5 x today's); fixed-centre models also need the
0.25 s low-pass of today's per-frame centres to drift <= max(4 x its noise, 0.0015 (locked) / 0.002 (pans) x the
median static-point depth; lateral only with zoom; on keyframes the pair noise is taken out of the keyframes' spread);
a fixed focal also needs today's per-frame focal within x1.05 of it (p90); every per-frame model needs its
high-frequency jitter <= 1.25 x today's + 0.000125 x depth (centre, ~1 mm at 8 m) / + 0.003 deg (rotation) (a wrong
"no zoom" otherwise becomes a dolly-zoom slide that still fits the static points: larissa_kadu cam09 1.85 px, centre
jitter 17 -> 412 mm). A keyframes-only `loc_<cam>.npz` (`loc_mode` keyframes) is valid for the constant pose only:
`stage_smooth` does not export such a camera under any other model and `cameras` runs without `loc` refuse it before
any work (`KeyframeCacheError`: re-run `--stages loc --cams <cam>`). `report.json` (additive):
`camera_types {policy, applied {<cam>: {cam, mount, zoom, pans, source, confidence, class}}, localised {<cam>: {mode,
class, policy, localised, frames}}}`, `cams.<cam>.camera_type {class, source, model, tried [...], check, intrinsics,
centre_drift, pose, localised, at}`, `cams.<cam>.camera_type_probe_b` (the suggestion below).

Automatic suggestion files (pre-fill the UI; `python -m dancecap.camera_types suggest <take> [--probe a|b|both]`):

```
work/<take>/probe/camera_types_suggested.json   probe A (before calibration, local CPU, below-normal priority): {version,
                                                take, by, at, cameras {<cam>: {mount, zoom, pans, confidence,
                                                mount_confidence, zoom_confidence, by: probe_a, at, evidence {parallax_px,
                                                jitter_hf, zoom_scale, pan_px, kf_*}, windows [...], keyframe_scan {...},
                                                file {name, size, mtime_ns}, version}}} (a result is reused for the same
                                                video file in another take)
work/<take>/probe/camera_types_probe_b.json     probe B (after calibration; also written by the cameras qa step): {version,
                                                take, by, at, metric, cameras {<cam>: {mount (null when the attach model
                                                fixes the centre), zoom (null when it fixes the focal: every attached
                                                camera), pans, confidence, mount_confidence, zoom_confidence, by:
                                                probe_b, at, evidence {centre_p95_m, axis_p95_deg, angvel_p50_dps,
                                                hf_rot_deg, hf_centre_mm, focal_ratio, focal_step_pct, frames,
                                                mount_judgeable, zoom_judgeable, mount_lean?, attached,
                                                keyframes_only?}, zoom_events}}} - always from today's general per-frame
                                                track (never the typed export: circular)
```

Narration tool API (take mode; the same CameraType JSON as the monitor, PIPELINE_MONITOR.md 7.3):
`GET /api/cameras/types`, `PUT /api/cameras/types/<cam>` `{mount, zoom}` (If-Match; `?preview=1` -> `{rerun_needed}`),
`DELETE /api/cameras/types/<cam>`, `POST /api/cameras/types/accept` `{cams?, min_confidence?}` (JSON only, 415
otherwise). Writes from another origin (`Origin` not the tool's own, `Origin: null`, `Sec-Fetch-Site: cross-site`) are
refused (403) on every endpoint of the tool. `rerun_needed` compares with the class the export was made with
(`cameras/<cam>.npz camera_class`; none = the general model).

## Runpod pod jobs - `work/<take>/<stage>/pod_job.json` (schema `dancecap.pod_job/1`)

GPU-heavy stages describe their pod work in a pod job; `python -m dancecap remote run --take <take> [--jobs a,b]`
(dancecap/remote_jobs.py) runs ALL of them in ONE pod session: one compressed upload, one bootstrap, jobs in
`after` order, downloads, pod always deleted. Stages must NOT pack their own code/model tarballs or pip lines:
the runner's bundle carries the current dancecap code, mamma-turing, the body models and cached CUDA wheels, and
`dancecap/remote/bootstrap.sh <features>` installs the environment. The pod workspace is `/workspace/HeadMovement`
laid out exactly like the local one (`dancecap/`, `mamma-turing/`, `mamma/data/body_models/...`, `smplx/`), so
stage code using `dancecap.env` paths runs unchanged; commands get `DANCECAP_POD=1 PYTHONUNBUFFERED=1`.

```
{"schema": "dancecap.pod_job/1",
 "name": "mamma",                       unique in a batch (= stage dir name by convention)
 "take": "larissa_kadu",
 "max_minutes": 50,                     this job's wall-time budget (session kill switch = bootstrap + upload +
                                          sum of jobs + 13 min unless --max-minutes)
 "needs": ["mamma"],                    bootstrap features: base (always) | mamma (MAMMA ma_3d deps + pytorch_sdf)
                                          | room (gsplat 1.5.3, open3d, xatlas, plyfile) | texture (base only)
                                          | gsplat (gsplat 1.5.3 + plyfile only: dancer_splats)
 "inputs": ["dancecap/work/<take>/mamma/prepare.json",               workspace-relative: same path locally and
            {"local": "<abs or ws-relative>", "pod": "<ws-relative>"}],  on the pod; files or directories
 "inputs_if_alone": [...],              added only when none of the `after` jobs is in the batch
 "archives": [{"local": "x.tar.gz", "extract_to": "."}],  pre-built ws-relative tarballs, extracted on the pod
                                          before the job (dancecap/dancecap and mamma-turing members are skipped: code
                                          always comes from the bundle). Prefer plain inputs.
 "after": ["mamma"],                    run after these jobs when they are in the batch; skipped if they failed
 "exclude": ["*.mp4"],                  extra glob patterns for the bundle (component or ws-relative path)
 "env": {"PYOPENGL_PLATFORM": "egl"},   "cwd": "dancecap" (ws-relative, default),  "setup": ["shell", ...]
 "commands": ["python -m dancecap mamma fit larissa_kadu --tag c7",            string -> name cmd<i>,
              {"name": "select", "cmd": "...", "timeout_min": 10, "optional": false}],  timeout = max_minutes
 "outputs": ["dancecap/work/<take>/mamma/fits/", ...],  ws-relative; trailing / = directory; downloaded as one tar.gz
 "verify": [...],                       must exist after extraction (default = outputs)
 "extract": "workspace",                "workspace" (in place) | "session" (work/runpod/sessions/<sid>/outputs/)
 "after_download": ["python -m dancecap mamma select <take>"],   local follow-ups (reported, not run)
 "notes": [...]}
```

The first-generation shapes written by mamma (upload.groups / dry_check / run / download), room (inputs.bundle /
pod_commands) and texture (inputs.texture_inputs / pod_commands.run) are still accepted through adapters
(`legacy` = mamma-v0 / room-v0 / avatar-v0: their setup/pip/pack lines are ignored, room paths `/workspace/dancecap`
are rewritten to `/workspace/HeadMovement/dancecap`, room `post` is optional, avatar runs after mamma). New writers
should emit schema 1.

Bootstrap features added 2026-10-06 (front stage): `cameras` (pycolmap 4.2.1 CPU wheel + LightGlue/ALIKED at the local
commit eb42fee + kornia) and `ma2d` (the lightning / albumentations / omegaconf stack MAMMA's landmarks/run_ma_2d.py
imports, local versions); every pip line of both runs under a constraints file holding numpy 1.26.4 / torch 2.5.1.

```
work/<take>/front/pod_job.json          [front_pod] job "front" (python -m dancecap.front_pod write-job <take>): the whole
                                          capture front-end on one pod - per camera label + SAM 2.1 masks (dancer worker:
                                          frames decoded once), per-frame localisation (loc) / sub-model presolve (sub),
                                          ma_2d per camera as its masks land, then dancers merge, cameras attach/world/qa,
                                          sync, world_scale chain, mamma prepare --skip-2d. Inputs: frames, keyframes,
                                          people, people masks, the reused SfM map + its cached features, the scale
                                          reference (+ the reference take's sfm bins / camera npz / frames.json), ALIKED +
                                          LightGlue weights (TORCH_HOME), the ma_2d checkpoint as a state_dict-only copy
                                          (work/runpod/weights/ma_2d/, identical tensors, sha256 in its .json); SAM 2.1
                                          large is fetched on the pod from Meta's CDN and checked against the local sha256
work/<take>/front/front_pod.json        per task state, rc, seconds, measured RSS peak, restarts; host cpus / memory
                                          limit / caps; working_set_peak_gb. front/memlog.tsv: working set + per-task RSS
                                          every 10 s; front/logs/<task>.log. The scheduler admits a task only within its
                                          kind's cap, an estimate budget (0.6 x the container limit) and the MEASURED
                                          working set (< 0.62 x limit); above 0.86 x limit the youngest non-critical task is
                                          killed and re-queued. Resumable: the job's command is `front_pod supervise`
                                          (3 optional copies) which runs the driver (`run --resume`) in its own session;
                                          every worker runs in its own session through a wrapper that writes front/rc/
                                          <task>.rc on exit, so a restarted driver ADOPTS live workers, reads finished
                                          ones' rc files and re-queues the rest (2026-10-06: the driver process was
                                          SIGKILLed twice with memory at half the limit while its workers lived on; cause
                                          not found: front/forensic_<n>.json + host_start.json record cgroup limits, OOM
                                          counters, rlimits and ps snapshots). Tasks whose outputs exist (uploaded from an
                                          interrupted session: done dancers cameras, valid dancers/cache/<cam>_sam.npz, loc,
                                          locsub, ma_2d) are pre-completed. The long-pole camera's localisation runs in two
                                          halves (front_pod loc-part -> cameras/cache/locpart_<cam>_<a>_<b>.npz, loc-join ->
                                          loc_<cam>.npz + cameras/<cam>.npz)
work/<take>/front/gates.json            [front gates, local] frames/PTS, labels + per-dancer coverage, geometric re-ID
                                          swap test (review/reid_geometry.png), localisation / gravity / scale / sync /
                                          prepare numbers
dancers/label_<cam>.json, pod_cam_<cam>.json   per-camera label and roles/report entries written by the pod workers
                                          (merged into labels.json / roles.json / report.json by dancers_pod merge)
mamma/ma_2d_native/<cam>.manifest.json  + host, seconds, weights_note: pod-computed landmarks carry the POD's mask
                                          signature, so a local `mamma prepare` must use --skip-2d (else it recomputes)

work/<take>/mamma/chunks.json           [mamma_chunks plan] tag, seq, cams, T, overlap, chunks [{a, b, run_tag, t}],
                                          estimate_min {chunk_parallel, single_fit_solo, basis}; optional holdout {tag
                                          h<cam>, cam, chunk, a, b, run_tag, t, seq} (plan --holdout-cam C: that chunk is also
                                          fitted from the prepared set h<C> = the fit set without C, side by side; merge
                                          reports chunks_merge.json holdout {cam, frames, baseline, lead/follow
                                          {held_out_median_px, in_fit_median_px (= the same-frames chunk WITH C when it
                                          exists, else the merged fit), in_fit_same_chunk_median_px,
                                          in_fit_merged_median_px, ratio}}: informational, never gates)
work/<take>/mamma_chunks/pod_job.json   job "mamma" (so the avatar bake runs after it): fit_chunks (every chunk = `mamma
                                          fit --tag <tag> --fit-frames a:b`, side by side under CUDA MPS) -> merge -> select
                                          --no-qa; outputs fits/, logs/, select.json, report.json, chunks_fit.json,
                                          chunks_merge.json, smplx/
work/<take>/mamma/fits/<tag>/           merged chunk fit (partial false, chunked true): smplx/<role>.npz = the smplx
                                          contract + chunk_owner (T,) i2, chunk_ranges (n,2); one betas per dancer (frame-
                                          weighted mean of the chunks'), each chunk re-expressed with it keeping the pelvis
                                          trajectory, cosine cross-fade over the central half of every overlap (per-joint
                                          slerp, linear translation / contacts), FK + reprojection + outlier repair as
                                          export_smplx; fit.json residuals on the eval set
work/<take>/mamma/chunks_merge.json     per role: betas_chunks / betas_sd / betas, betas_swap_mm (joint displacement from
                                          the shape unification, per chunk median/p95), seam_mm (mamma_chunks.seam_stats:
                                          the two chunks' disagreement inside each overlap: frames [a, b), pooled median /
                                          p95 / p95_body (joints 0-21) / p95_fingers / pelvis_median, and PER JOINT for
                                          blend (the cross-faded central half; gated) and overlap (informational):
                                          {joint_p95_max + joint (worst non-arm body joint 0-17), arm_p95_max + arm_joint
                                          (worst elbow / wrist), fingers_rel_wrist_p95_max (finger joints relative to
                                          their wrist), per_joint_p95 {22 names}, frames}), chunks_had_repairs; merged vs
                                          per-chunk median reprojection; chunk_frames; gate {passed, why, checks,
                                          thresholds}: per role and seam, over the blend window, every body joint 0-17
                                          p95 <= 60 mm, every elbow / wrist p95 <= 75 mm, every finger joint relative to
                                          its wrist p95 <= 80 mm (a pooled p95 over 22 joints cannot see ONE wrong limb:
                                          larissa_kadu_38 follow seam 2 pooled 44 mm, her left wrist 107 mm; reports
                                          without per-joint stats fall back to the pooled all-joint p95 <= 60 mm), pelvis
                                          median <= 40 mm, betas swap p95 <= 40 mm, merged median reprojection <= 1.25 x
                                          the chunks' + 0.5 px, no chunk above 2 x the chunks' median reprojection
work/<take>/mamma/chunks_recheck.json   [mamma_chunks recheck, local] the merge's checks re-run from the chunk fits with
                                          the current code (seam_mm per joint, holdout vs the same-frames chunk, gate;
                                          gate_at_merge = the pod's verdict); chunks_merge.json is left as the pod wrote it.
                                          larissa_kadu_38: FAILED (follow seam 2 l_wrist 107 mm > 75, seam 3 l_foot 74 mm
                                          > 60; both next to chunk c8_f368-600) -> the full fit in smplx/ stands
work/<take>/mamma/chunks_gate.json      [mamma_chunks gate, pod, between merge and select] {gate, fallback {mode, need_min,
                                          kill_minutes_left, result, seconds}}: gate failed -> --fallback full re-fits the
                                          whole grid in one run (`mamma fit --tag <tag>`, the rejected merge moved to
                                          fits/<tag>_merged_rejected/, its fit.json re-tagged <tag>_merged_rejected +
                                          partial/rejected true so `mamma select` never reads it as <tag>) when the
                                          pod's kill switch (/workspace/
                                          .dancecap_kill_at, written by remote.arm_killswitch) leaves time, else exit 3:
                                          select and the avatar bake (after: mamma) do not run. write_job budgets the
                                          fallback fit in the job's max_minutes (--fallback none: not)
work/<take>/mamma/chunks_seams.json     [mamma_seams, local after download] continuity of the merged fit at each blend
                                          window (central half of an overlap), per role and seam: disagree_mm (the two
                                          chunks over the overlap: median / p95 / pelvis median / pelvis max),
                                          pelvis_acc / body_acc (pelvis-relative body joints) / speed_step {merged_max,
                                          prev_max, next_max, ratio_vs_chunks} over the window, merged_pelvis_acc_vs_take_p99
                                          (context), smooth (all three ratios <= 1.25: the cross-fade added no motion of
                                          its own); take_outside_windows p95/p99; top_pelvis_acc [5] {frame, t, acc,
                                          in_blend_window}; merged_is_chunked false = the gate's full-fit fallback won
work/<take>/review/pose_seams.jpg       [mamma_seams] whole-take pelvis speed (blend windows shaded) + per seam: pelvis path
                                          from above and pelvis speed (merged vs both chunks) per dancer, chunk-vs-chunk
                                          joint distance over the overlap

Stitch of a long take's fits (2026-10-08, dancecap.stitch + dancecap.stitch_job; new, additive; larissa_kadu_full = a,
38, b, c, d). RUNS ON RUNPOD (FARM.md 0.10: all analysis): numpy + scipy only, headless (no torch / smplx / mamma_dyn;
matplotlib only for the two sheets), so a CPU pod runs it from the volume's env; locally only its unit tests run:
python -m dancecap.stitch <dance> [--chunks chunks.json] [--work-root ROOT] [--out DIR] [--check] [--seam b|c] [--partial]
        [--no-refined] [--no-sheets] [--force] [--gate-exit strict|report] [--allow-mixed-world] [--require-world]
python -m dancecap.stitch <take> --write-chunks FILE   the take's chunk list as JSON (paths relative to the work root)
work root                                 $DANCECAP_WORK_ROOT / --work-root, default dancecap/work (pod: the workspace's
                                          dancecap/work or any /workspace/hm/... folder): every relative path (the chunk list,
                                          --out) resolves against it. --out DIR = the output folder (default <root>/<dance>);
                                          --check writes no motion (with --out: stitch_check.json + review sheets there);
                                          --gate-exit report exits 0 on a failed gate (the verdict is in stitch.json, the motion
                                          in stitch_rejected/) so a pod job's results are pulled; exit 3 = world mismatch
                                          (chunks with different pose_id refused unless --allow-mixed-world; ids from the chunk
                                          entry, its `header` json or the list's world_id / pose_id), 1 = missing fit
python -m dancecap.stitch_job build <dance> [--mode stitch|check] [--chunks F] [--tar CHUNK=PATH] [--world-id ...]   the
                                          farm job (dancecap.farm.job/1, kind analyse, needs_gpu false, cwd dancecap/, env
                                          DANCECAP_WORK_ROOT={WS}/dancecap/work): [stage = extract chunks/<id>/out/result.tar
                                          of the volume, sha256-checked, only when the list names tars] + stitch (STEP_TEMPLATES
                                          "stitch" / "stitch_check", register() adds them to farm.chunks.STEP_TEMPLATES without
                                          touching it); outputs dancecap/work/<dance>/{stitch.json, smplx/, <refined>/, review/,
                                          stitch_rejected/} (check: <dance>_check/{stitch_check.json, review/}); the chunk list
                                          is the job's one input file (farm/stitch/<id>.chunks.json). `python -m
                                          dancecap.stitch_job stage --chunks F` is the pod-side staging step
takes/<take>.toml [segments.*] [reuse] [stitch]   the chunk list: segment fit windows (fit_k; the fit's own t wins), the
                                          reused analysed take (motion_span, motion_raw, motion_refined; its first and last
                                          [stitch] reuse_margin_s = 0.5 s are never used), blend_s = 2.0, gate_mm {body_p95,
                                          elbow_wrist_p95, fingers_rel_wrist_p95, pelvis_median, betas_swap_p95},
                                          mono / low_view / <role>_absent spans (reference s). chunks.json (any chunk list,
                                          e.g. 30 s gap-fill chunks): {take, work, anchor, hz, blend_s, edge_margin_s, gate_mm,
                                          beats, mono, low_view, lead_absent, chunks: [{name, src, refined, k [k0, k1],
                                          margin_s, reference, prepare, ma_2d, world_id, pose_id, header, tar}], world_id,
                                          pose_id}; relative paths resolve against the work root (default dancecap/work)
fused grid                                t_k = 30.151176844227862 + k / 30 s (larissa_kadu_38's lattice; its frame i is k = i,
                                          full-grid frame i + 998 for k from -998 to 4645, T = 5644). A chunk's window is the
                                          lattice index of its own `t` (checked on the lattice to 1 us and contiguous)
work/<take>/smplx/{lead,follow}.npz       [stitch] the smplx contract on the fused grid (t, pose keys, transl, betas =
                                          the reference chunk's, joints, contact, floor_contact, reproj_px, quality, ...) +
                                          chunk_names (n,), segment_owner (T,) str (the owning chunk's name, "" = uncovered),
                                          segment_index (T,) i2 (its index into chunk_names, -1 = uncovered),
                                          blend_seam (T,) i2 (-1 outside a window), blend_weight (T,) f4 (weight of the later
                                          chunk), covered (T,), n_cams (T,) i2 (cameras seeing this dancer: >= 50 of 512
                                          ma_2d landmarks visible, else the prepare.json camera spans), quality_flag (T,) i1
                                          0 absent / 1 mono / 2 two_view / 3 multi (>= 3 cameras) + quality_flag_names, absent
                                          (T,) bool (quality and quality_parts are 0 there: the lead before he enters is never
                                          tracked motion; a zero-camera run >= 1 s is absent too, a shorter one is mono).
                                          Frames a chunk owns outside every blend window are BIT-IDENTICAL copies of its
                                          source (stitch.json bit_identity: max abs diff 0.0 for pose, transl, joints, ...;
                                          t is the clean lattice). Inside a window: per-joint slerp (axis-angle branch
                                          nearest the linear blend), linear transl / contacts, joints = FK with the common
                                          betas, reprojection fields = the worse chunk, quality = the lower. Written
                                          byte-deterministically (fixed zip timestamps): running twice gives the same files
work/<take>/smplx_rf2_noslip/             [stitch] the chunks' refined folders stitched with the SAME windows (same extra
                                          keys; smoothing = json of each chunk's pose_refine record; stance carried). Only
                                          written when every chunk has a refined folder
work/<take>/stitch.json                   [stitch] dancecap.stitch/1: grid {anchor, hz, k, T, frame_offset per chunk}, chunks
                                          {k, owned_k, margin_frames, sha256, betas {pinned, max_abs_diff, swap_mm}, n_cams
                                          source}, world {world_ids, pose_ids, unknown, warnings}, seams [{name, ok,
                                          window_k [k0, k1] inclusive (what farm.timeline.parse_stitch reads), overlap,
                                          allowed, window k / t, centre_t, utilisation,
                                          tie_break {method beat_phase | motion, n_tied}, scan, roles {lead, follow}
                                          {blend per-joint p95 stats, checks, continuity {pelvis / body acc ratio, max joint
                                          step}}, gate {passed, why}, refit}], gate, bit_identity, flags (runs, counts,
                                          first tracked frame), refined, coverage {frames complete / partial / none, tiles:
                                          1 char per second C / P / N}, outputs {promoted, sha256}, verified_on_disk
                                          (the written files re-read and compared with the sources)
                                          Window: ONE per seam, 2 s (60 frames) inside the overlap and inside both chunks'
                                          trusted frames (reuse margin 0.5 s, fits 0.2 s), after the previous window; centre =
                                          the position with the lowest worst-gate utilisation over both dancers (per-joint p95
                                          over the window / limit); positions within 5 % (+0.02) of the best are ties, broken
                                          toward the beat phase of lowest motion (beats.json), else the lowest local motion.
                                          Gate per role and seam over the window: body joints 0-17 p95 <= 60 mm, elbows /
                                          wrists <= 75, fingers relative to the wrist <= 80, pelvis median <= 40, betas swap
                                          p95 <= 40 (betas not within 2e-5 of the reference's are re-expressed keeping the
                                          pelvis trajectory, as mamma_chunks.merge). A failing seam names the gate and
                                          `refit`: the centred 12 s window (360 frames, k and t, which chunk's prepared grid
                                          contains it)
work/<take>/stitch_rejected/              [stitch] the outputs of a run whose gate failed (smplx/ is not touched; --force
work/<take>/stitch_partial/               promotes); stitch_partial/ when chunks are still missing (--partial; gaps are
                                          flagged absent and counted "none" in coverage)
work/<take>/review/pose_seams_full.png    [stitch] per seam: pelvis paths (chunks vs stitch), chunk-vs-chunk distance with
                                          the window, per-joint p95 vs the gate lines, window scan
work/<take>/review/stitch_overview.png    [stitch] ownership bars + blend windows, coverage strip, cameras per dancer,
                                          reprojection, pelvis speed, quality flags

larissa_kadu_38 pod-stage review fix (2026-10-07; scratch scripts, take-specific outputs):
work/larissa_kadu_38/review/pose_seam_joints.png   per-joint p95 per seam (chunks_recheck.json) with the gate limits
work/larissa_kadu_38/scale/floor_diagnosis.json    final fit's lowest body point per frame, BOTH directions (below /
                                          above +-15 mm, both feet up), fast (< 0.5 s) vs slow part, correlation of the
                                          fast part with pelvis height, planted-foot height (0.5 s running minimum) by 0.5 m
                                          floor cell + plane fit, lead/follow time correlation, float / sink runs >= 5
                                          frames -> input for physics; review/floor_contact.png
work/larissa_kadu_38/avatar_fix/pod_job.json        pod job: re-bake (pod preset, hair exclude, multiband, 133.3 ms,
                                          --body-ref-q 0.7) + avatar_face portrait composite + hair metrics; the previous
                                          bake is kept in avatar/prev_pod1_20261007/ (+ review/prev_pod1_20261007/);
                                          avatar/{fix_metrics,compare_fix,hair_metrics_fix}.json, pod_{texture,face}_fix.log;
                                          review/avatar_back_residue_before_after.jpg (torso atlas, residue in magenta),
                                          review/avatar_face_portrait_video.jpg (photo | avatar with the portrait-face
                                          albedo where each face is seen best), review/avatar_fix/*_before_after.jpg

Post-fit motion clean-up + quality (2026-10-07, dancecap.pose_refine / dancecap.pose_quality; new, additive):
work/<take>/<dst>/{lead,follow}.npz     [pose_refine] the smplx/ contract (same keys; joints re-computed by FK, plain
                                          LBS) + smoothing (json str: mode, filter {root_hz, rot_hz, transl_change_mm_rms,
                                          stance_frac}, refine {iters, w_floor, data_term_before_after,
                                          data_per_cam_before_after, root_offset_mm, leg_rot_offset_deg_p95, history}) +
                                          stance (T,2) bool left/right. smplx/ is never written. Modes: filter (zero-phase
                                          low-pass: root 5 Hz, joint rotations 8 Hz via sign-continuous quaternions) |
                                          root / ik / root+ik (scripted floor fixes, image-blind) | refine (default: filter
                                          + VERTICAL root and hip/knee/ankle offsets per frame minimising the 2D
                                          Geman-McClure error of MAMMA's 512 points vs the same ma_2d landmarks + w_feet x
                                          each foot's own points (--w-feet, default 1) + stance lowest-sole height /
                                          --floor-mm + penetration + slip only on near-static stance (--slip-max-v 0.15
                                          m/s) + horizontal ankle/foot anchor to the filtered input (--w-anchor 1,
                                          --anchor-mm 10) + second differences + offset prior; torch, --device cuda on a
                                          pod). The first run's energy = --w-feet 0 --slip-max-v 0 --w-anchor 0 --root-xz.
                                          refine metadata adds floor_mm, w_feet, w_slip, slip_max_v, root_xz, w_anchor,
                                          anchor_mm, slip_pairs, feet_data_per_cam_before_after. Every mode RECOMPUTES
                                          reproj_px, reproj_parts_px, quality, quality_parts for the new motion
                                          (pose_refine.requality = export_smplx's definition; --no-quality drops them);
                                          contact, floor_contact, repaired, suspect, reproj_px_raw are the source's
                                          (smoothing.carried_from_source). Refuses a fit whose t differs from the grid's t
                                          (full-span fits only).
  pose_refine.json                      per role the same metadata as the npz 'smoothing'
work/<take>/quality/quality_<tag>.json  [pose_quality] reproj_px {whole, window, all_cams_per_frame_median(_window)}
  + quality_<tag>_frames.npz              per camera / dancer (median + p90 of the per-frame median px of the 512 points,
                                          smplx FK with correctives, chosen set's landmarks); compare {take, reproj_px,
                                          all_cams_per_frame_median, jitter_4p5_12hz_rms_mm} (--compare-take);
                                          jitter_4p5_12hz_mm per dancer {whole_rms, turn_rms, no_turn_rms, turn_fraction,
                                          window_rms, worst_0p5s_windows, p95_0p5s_window} (turn = pelvis yaw rate > 200
                                          deg/s +-3 frames); floor per dancer {lowest_sole_mm, frac_below_15mm,
                                          frac_above_15mm, frac_above_30mm, stance_sole_mm, stance_abs_gt_15mm_frac,
                                          stance_slip_mm (max deviation from the run mean: grows with run length, not a
                                          per-frame slip), lowest_foot_joint_minus_rest_mm}; rest_height_m vs stature +
                                          sole; physics {dir, per dancer mass, contact_fraction, band} (--physics).
                                          frames npz: t, px_<cam>_<role>, jit_<role>, turn_<role>, low_<role>,
                                          stance_l/r_<role> (+ cmp_* for the compare take).
                                          Added 2026-10-07 (post review): reproj_feet_px (same layout, feet only = 88 of
                                          the 512 points, + per_foot_all_cams); slip per dancer on FIXED frames = the
                                          reference fit's own-stance runs (--ref-dir, default smplx/): contact-point speed
                                          (median horizontal speed of the sole vertices within 1 cm of the lowest point,
                                          same vertices in both frames: heel-toe roll is not slip), ref_run_drift_mm, and
                                          for pose_refine folders pinned_in_sample (the frames the slip term minimised) /
                                          ref_stance_not_pinned; displacement_vs_ref_mm per joint (pelvis, ankles, feet:
                                          horiz / vert, stance / swing; median, p95, p99); notes (jitter band is removed by
                                          a root low-pass by construction). frames npz adds allpx_<role>, fpx_<cam>_<role>,
                                          fpx_<role>, fpx_l/r_<role>, slipv_l/r_<role> (T-1), rst_l/r_<role>,
                                          sole_l/r_<role>, refsole_l/r_<role>, pin_l/r_<role>, dh_/dv_<joint>_<role>
work/<take>/quality/variants.json       [pose_quality --variants <tags> --ref-tag final] per variant and dancer:
                                          reprojection all / feet (+ per camera and delta vs ref), feet px change on ref
                                          stance frames with the ref sole > 15 mm above / within 15 mm of the floor,
                                          stance ankle displacement, floor fractions, slip on fixed frames (+ ref vs
                                          variant on the variant's pinned frames), jitter; review/pose_refine_variants.png
work/<take>/physics_<smplx dir>/        [pose_quality --physics] physics.run on that folder (beats from timing/timing.json).
                                          Its bands are OPTIMISTIC for a pre-filtered input (the residual bootstrap sees
                                          little above 5 Hz); physics/ (always on smplx/) is the one with honest bands
takes/<take>.toml [export] smplx_dir    the motion folder of EVERY exporter when --smplx-dir is not given
                                          (export_unity.resolve_smplx_dir: `python -m dancecap export` and demo_export;
                                          default smplx/); recorded in capture.json provenance smplx_dir +
                                          smplx_dir_source

larissa_kadu_38 final stage (2026-10-07; two pod sessions, $0.75):
work/larissa_kadu_38/final/pod_job.json, pod_job_b.json   pose_refine variants + pose_quality + timing + physics on
                                          A100 pods (the local PC's CPU was starved: timing ran > 25 min locally without
                                          finishing; 133 s on the pod)
work/larissa_kadu_38/smplx_refined/     FIRST refinement (superseded 2026-10-07 by smplx_rf2_noslip, kept for comparison):
                                          refine with every stance frame pinned against slip, free horizontal root. vs
                                          smplx/: floats 22 / 29 % -> 0.2 / 1.5 %, sinks 13 / 9 % -> 0, all-points
                                          reprojection 5.71 / 7.32 -> 6.05 / 7.70 px BUT feet-only 7.57 / 8.56 -> 9.04 /
                                          9.58 px (cam02 feet +6.6 / +7.0 px) and stance ankles moved 29.5 / 26.4 mm
                                          sideways (median, p95 94 / 75 mm); per camera all points lead 02 +1.21, 03
                                          +0.47, 04 +0.27, 05 +0.35, 09 +0.31; follow 02 +1.38, 03 +1.03, 04 +0.75, 06
                                          +0.20, 09 +1.54 (cost on 6 of 8 cameras for the follow)
work/larissa_kadu_38/smplx_rf2_noslip/  ADOPTED for Unity ('post' review fix; takes/larissa_kadu_38.toml [export]; =
                                          pose_refine defaults): vertical root + leg offsets, each foot's own 2D term,
                                          ankles/feet anchored horizontally, no slip term. vs smplx/ (lead / follow):
                                          floats 22 / 29 % -> 0.6 / 1.5 %, sinks 13 / 9 % -> 0; stance ankles 2.5 / 2.8 mm
                                          sideways (p95 6.5 / 7.2), 7 / 8 mm vertically; all points 5.71 / 7.32 -> 6.05 /
                                          7.42 px (lead 02 +0.62, 03 +0.74, 04 +0.45, 09 +0.87, 06 -0.28, 07 -0.36; follow
                                          02 +0.58, 03 +0.77, 04 +0.50, 09 +1.29, 01 -0.30, 05 -0.26, 06 -0.53, 07 -0.39);
                                          feet 7.57 / 8.56 -> 7.43 / 8.37 px (in-sample); contact-point slip on smplx/'s
                                          stance frames 99 / 127 -> 93 / 119 mm/s (filter alone 93 / 113); jitter 0.58 /
                                          0.98 mm (the 5 Hz low-pass, by construction)
work/larissa_kadu_38/smplx_vert/, smplx_rf2/, smplx_rf2_noanchor/, smplx_rf2_feet10/, smplx_rf2_feet30/   the other
                                          'post'-fix variants (quality/variants.json, review/pose_refine_variants.png):
                                          vert = mode root (whole-body vertical shift: +1.15 / +0.96 px all points, sinks
                                          2.4 / 6 %); rf2 = + slip term below 0.15 m/s (slip 77 / 113 mm/s, +0.09 / +0.10 px
                                          more than noslip); noanchor = no horizontal anchor (ankles 12 / 13 mm sideways);
                                          feet10 / feet30 = --w-feet 10 / 30 (leg offsets p95 11-16 deg, ankles 8-12 mm
                                          sideways). physics_smplx_{vert,rf2,rf2_feet10}/ = physics on those (rf2's COM
                                          is 4.6 / 5.9 mm (median) from physics/'s; noslip's joints differ from rf2's by
                                          0.5 / 0.8 mm median)
work/larissa_kadu_38/final/pod_job_postfix.json + postfix_pod.sh   the 'post'-fix pod batch ($0.32): 6 refine
                                          variants in parallel on one A100, timing re-run (qa_timing.png now on the take's
                                          span), pose_quality of 9 folders (+ physics for 3), variants table;
                                          logs/postfix/*.log, logs/postfix_pod_session.log. The re-run timing's MC CIs
                                          moved by a few ms (workers 256 -> 31 changes the draw pairing; point estimates,
                                          hit rates and p-values identical); the previous timing/ is in
                                          backup_postfix_20261007/
work/larissa_kadu_38/smplx_filtered/    filter only (image-neutral: 5.70 / 7.34 px; jitter 0.58 / 0.98 mm; floor
                                          unchanged). quality_smplx_refined_w{1,03,01}.json = first-run variants with a
                                          biased soft-min floor term (soles ~7 mm above the floor; folders deleted)
work/larissa_kadu_38/quality/quality_final.json   the MAMMA fit (smplx/) + the 6 s c8 fit of larissa_kadu on 56-62 s
work/larissa_kadu_38/timing/            timing on the raw fit with the studio grid (--no-refine) and the correlated MC
                                          convention (--mc-corr-noise-mm 10 --mc-corr-time-s 0.1)
work/larissa_kadu_38/physics/           physics on the raw fit (statures 1.727 / 1.651 m + 2.5 cm sole): 83.2 / 57.2 kg
work/larissa_kadu_38/final/room_from_larissa_kadu.json   similarity larissa_kadu world -> this world (x0.9545, 2.0
                                          deg, t); the room is not exported (dancers-only viewer)
review/pose_smoothing_before_after.png (adopted smplx_rf2_noslip, + feet per camera), pose_refine_variants.png,
review/timing_physics_38.png, review/unity_*.png (04 playtest; quality/unity_playtest_postfix.json)
```

```
work/runpod/ledger.json                 every pod created (price, created/deleted, cost) -> budget cap (remote.py)
work/runpod/bundles/bundle_<key>.tar.gz content-keyed upload bundle + bundle_<key>.json manifest (files, hard links
                                          for identical content, raw/compressed bytes)
work/runpod/wheelhouse/<key>/*.whl      CUDA wheels built on a pod (pytorch_sdf, gsplat; cp311 torch2.5.1 cu124
                                          sm_80/90), uploaded with later bundles so bootstrap installs instead of
                                          compiling. Licensed/third-party binaries: never committed
work/runpod/pywheels/cp311-linux/       linux cp311 wheels (+ 2 sdists) of the bootstrap's PyPI packages (base + mamma
                                          features), filled locally with `pip download` (command in remote_jobs.PYWHEELS);
                                          bundled when present (~200 MB) and installed first with --no-index by
                                          bootstrap.sh pipq, PyPI only as fallback (a host that stalls on large PyPI
                                          pages cost a 20 min bootstrap on 2026-10-07). bootstrap.sh also sets
                                          PIP_TIMEOUT 60 s / PIP_RETRIES 10 and the runner retries a failed bootstrap
                                          once on the same pod. Third-party binaries: never committed
work/runpod/sessions/<sid>/             state.json (per job fingerprint/status, step timings), result.json (pod, GPU,
                                          price, ledger cost, bootstrap seconds, per-job status/missing files),
                                          runner.log, bootstrap.log, bootstrap_check.json (import checks),
                                          bootstrap_timings.tsv, downloads/<job>.tar.gz, pod_logs/<job>/<cmd>.{log,rc},
                                          outputs/ (jobs with extract=session)
                                          + (progress, 2026-10-07; docs/PIPELINE_PROGRESS.md) runner.json {sid, pid,
                                          started, attempt, label, jobs} (no argv/env), progress/runner.jsonl (runner
                                          phases, dancecap.progress/1), progress/pod.jsonl (pod events mirrored by the
                                          puller + log-tail events src "tail", each with pod/rx/skew), progress/
                                          puller.json (offset, last_ok/last_err, rtt, skew, kill_at, gpu[], load1, cpus,
                                          upload_part_bytes, tails{}, front{}). state.pod gains port + kill_at_est,
                                          state.jobs[j] gains current + pack_bytes (all additive)
pod: <ws>/dancecap/work/runpod/sessions/<sid>/progress.jsonl   every pod command appends (DANCECAP_PROGRESS set by
                                          remote_jobs.command_script); never packed; read only by the puller's probe
work/<take>/progress/<stage>[.<pid>].jsonl   local stages (DANCECAP_PROGRESS set by the launcher; one file per process
                                          on Windows)
work/progress/rates.json                 ETA history (python -m dancecap.progress rates --rebuild; also at each
                                          session end): {v, built, sessions, keys{"<job>/<kind>/<unit>@<gpu>": {n, p10,
                                          p50, p90, sigma, samples[]}}}. Derived, rebuildable, never hand-edited
```

Progress events (`dancecap.progress/1`, one JSON object per line, <= 1024 bytes, plan <= 4000): `{v: 1, ts, job, step,
kind: plan|start|tick|end|fail|note|metric|heartbeat, done?, total?, unit?, msg? (plain text, redacted), lane?, cmd?,
rc?, eta_s?, steps? (plan), extra?, sid?, take?, pid}`; ticks are throttled (>= 2 s and >= 0.5 % apart). Metric
`camera_motion` (cameras stage qa step = camera_types probe B, lane = camera): `extra {source: "probe_b",
constrained: false, metric_scale, at, centre_p95_m, axis_p95_deg, angvel_p50_dps, hf_rot_deg, hf_centre_mm,
focal_ratio, focal_step_pct, frames, mount_judgeable, attached, mount, zoom, pans, confidence, mount_confidence,
zoom_confidence}` measured on today's UNCONSTRAINED per-frame track (never the type-constrained export) and checked
against the user's camera types with `dancecap.camera_types`' thresholds; events without `source: "probe_b"` (the
first version's `{centre_spread_m, rot_spread_deg, focal_ratio}`) are ignored. Writers: `dancecap.progress` (stdlib,
a no-op unless DANCECAP_PROGRESS is set). Readers: `dancecap.progress_eta.snapshot()/live_snapshot()/budget()`
(read-only; `live_snapshot` keeps one background-polled Tracker per watched session in the monitor process).

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
"smplx_albedo": {"lead": "lead_albedo.png", "follow": "follow_albedo.png"},   photoreal albedo (copy of
                                         avatar/<role>_albedo.png) when the texture stage ran; needs skin version 2;
                                         with smplx_face/ the face variant (avatar_face portrait look by default)
"smplx_albedo_variant": {"lead": "face"|"videolook"|"bake", ...}   which texture <role>_albedo.png is (face_export)
"smplx_face": {"lead": "smplx_lead_face.bin", ...}   DCXF expression / blink blendshapes + weights (face_export;
                                         read by Assets/Avatar/SmplxFace.cs; see smplx_face/ above)
"room": {"mesh": "room_mesh.bin", "shading", "triangles", "vertices", "cleared_triangles", "textures", "bounds_min",
         "bounds_max"},                  Quest room mesh (from room/room_mesh.obj, else room_mesh.ply vertex colours)
"environment_splat": {"path": "splat_quest.ply", "scene_to_unity": [16 row-major], "frame"}   room/splat_quest.ply
                                         (SfM frame) + SfM -> Unity = diag(1,1,-1,1) @ world_from_sfm
```

Written by `python -m dancecap export <take> --capture <Name>` (dancecap/export_unity.py) to
`work/<take>/export/<Name>/` and mirrored to `head-movement/Assets/StreamingAssets/<Name>/` (git-ignored).
`<Name>` must match `^[A-Za-z0-9_][A-Za-z0-9_.-]*$` (not `.`/`..`). Re-run `python -m dancecap.moves export` (and the
counterbalance export) afterwards: the capture folder is rewritten.
Binary files: 4-byte magic + uint32 header length + JSON header (space-padded to 4 bytes) + raw little-endian blocks
listed in the header's `blocks` [[name, dtype, shape], ...], each padded to 4 bytes; readers locate blocks by name.

Audio (required): `--audio X --audio-t0 T` (both or neither), else `audio/audio_overlap.wav` with
`audio/audio_overlap.json` reference_start as t0, else a synthetic take's `audio.wav` (truth.json present, t0 = 0);
otherwise the export fails unless `--no-audio` (Unity then plays against a silent clock). time_to_audio is never -0.0.
Freshness: physics.json is exported only when its `t` is the smplx grid (same length, max |dt| < 1 ms); a role's jerk
only when kinematics `<role>_t` covers the capture within half a frame; otherwise skipped with a warning
(report["warnings"]). Unreliable jerk joints (`<role>_jerk_reliable` False) are written as 0 and listed in
timing.json `jerk_reliable`.

- `smplx_<role>.bin` (magic `DCXM`): T x 388 float32 = transl(3) + 55 local rotations (x,y,z,w Unity) + 55 joint
  positions (Unity). Joint order = SMPL-X (0 pelvis .. 21 wrists, 22 jaw, 23/24 eyes, 25-39 left hand, 40-54 right
  hand); rotations are the npz axis-angles (hands absolute, flat_hand_mean=True), so FK with the skin's rest joints
  (root bone at rest_joint0 + transl) reproduces `joints` (exporter checks < 0.01 mm on synthetic).
- `smplx_<role>_skin.bin` (magic `DCXS`, SMPL-X-derived, never commit): rest_vertices (V,3) f32 for the dancer's
  betas (expression 0), faces (F,3) i32 with winding reversed for the left-handed frame, LBS weights (V,55) f32,
  rest_joints (55,3) f32, parents (55,) i32. Model = MAMMA locked-head SMPL-X for MAMMA fits, else
  `smplx/SMPLX_NEUTRAL.npz` (`--model` overrides). Plain LBS: pose-corrective blendshapes are not applied.
  Version 2 (default; `--no-textures` writes version 1): split at the model's UV seams = one vertex per unique
  (v, vt) pair of (f, ft) -> V = 11313, rest positions and weights duplicated, plus blocks uv (V,2) f4 (Unity UV,
  PNG row 0 = v 1), normals (V,3) f4 (area-weighted normals of the UNSPLIT rest mesh, so seams do not crease) and
  vertex_ids (V,) i4 (SMPL-X vertex; equals avatar/uv_mesh.npz vertex_ids).
  Rest-shape edits: takes/<take>.toml `[dancers.<role>.shape_edits]` (dancecap.shape_edits, every export unless
  `--no-shape-edits`) move the rest_vertices after the betas - midsection_scale (waist / abdomen width and depth, lower
  ribs .. hips, cosine ramps into chest and hips; midsection_reference = another take's unedited body as the base,
  midsection_depth_pivot), shoe_length_m / foot_height_scale / foot_scale (feet scaled about the ankle's vertical line
  on the sole plane, so the sneaker built round the foot comes out at the length measured in the video). Joints,
  weights, faces, UVs and the motion are unchanged; the header records `shape_edits` (shape_edits.summary) and
  capture.json `provenance.shape_edits`.
  The optional `[dancers.<role>.shape_edits.face]` table (lower face / jaw / chin / under-chin / neck top; eyes, brows,
  nose, forehead, scalp = the hair roots, ears and the neck-shoulder seam stay exactly in place so the face texture and
  the hair keep lining up) has three layers, applied in this order: (1) `identity` (+ `identity_weight` 0..1.5,
  `identity_mouth` 0..1, `identity_neck` 0..1): the dancer's own FLAME 2020 identity (dancecap.face_identity: SMIRK from
  her portrait + video faces, work/<take>/face_shape/*.npz, likeness data, git-ignored) transplanted onto the rest head
  through SMPL-X__FLAME_vertex_ids, re-targeted per take (different betas) and cached in
  work/<take>/face_shape/<role>_identity_disp.npz (keyed on the rest head, the identity file and the mask parameters; FLAME
  is only needed on a cache miss); the neck is a biharmonic fade-out of the chin's move, never FLAME's own neck;
  (2) the height / width / depth factors (`lower_face_height_scale`, `chin_height_scale`, `chin_width_scale`,
  `jaw_width_scale`, `cheek_scale`, `lower_face_depth_scale`, `submental_height_scale`); (3) the petite taper (`jaw_taper`,
  `cheekbone_keep`, `chin_point`, `neck_under_jaw`) and `chin_forward_mm` (-6..12 mm: the chin proper moves forward,
  fading out at the lower-lip fold, under the chin and sideways: the identity fit sets the chin back, a front photo does
  not constrain its depth). Every field is C1 (cosine ramps / smoothstep / thin-plate), zero
  below ~5.5 cm under the menton and on the arms; shape_edits.summary records the non-default factors and the identity
  file / weight / max move. Candidates and numbers: work/larissa_kadu_38/review/face_shape_v4/.
- `room_mesh.bin` (magic `DCXR`, likeness-free but capture data: git-ignored): blocks positions (N,3) f4 in Unity
  capture coordinates, uv (N,2) f4 or colors (N,4) u1, indices (M,) i4; header submeshes [{name, start, count,
  texture, vertex_start, vertices}] ("room" -> room_tex.png, "floor" -> floor_tex.png, PNGs resized to
  power-of-two sides <= 2048). "room" triangles lying over the eroded (0.4 m) floor footprint and entirely below
  2.3 m are dropped at export (`cleared_dance_floor_triangles`): TSDF ghosts of the dancers along their path.
  Rendered unlit, double-sided (Resources/HM_RoomUnlit).
- `timing.json` (Unity subset of timing/timing.json, reference seconds): beats.grid [t,type,measure,count],
  footsteps.{lead,follow}[{t, foot, async_ms, beat_t, sigma, ...}], asynchrony.{role}.{n, mean_ms, ci95_ms,
  hit_rate, R}, and when timing/kinematics.npz exists jerk.{lead,follow} (T,24) |jerk| m/s^3 of the SMPL-24 joints
  (SMPL-X 0..21 + left/right_middle1) on the capture grid - Unity uses it instead of finite differences.
  `absolute_timing` {valid, av_sync_event, note} when timing.json has it (2026-10-07): valid false (no audio-video sync
  event) makes the viewer print step offsets "vs beat (unsynced)" with no on beat / late / early verdict and no hit rate.
- `physics.json` is copied verbatim (world frame; the Unity overlay converts (x,y,z) -> (x,y,-z)).
- `zouk-time-analysis.json` holds the beat grid in audio-clip time ([[t + time_to_audio, type]]).

`hair_groom.json` (optional, dancecap.hair_groom): when present the follow avatar gets `HairStrands` (head-movement
Assets/Hair): guide roots placed on the exported skin by SMPL-X vertex id (skin vertex_ids), lengths scaled by the skin's
rest height over model.rest_height_m, rest shape settled under gravity on the rest pose, Burst XPBD simulation stepped
per CAPTURE frame (times.json intervals, substeps interpolate the head and capsules; seek / loop back / jumps > 8 frames
re-hang + 0.5 s static settle + 20-frame pre-roll), one dynamic ribbon mesh (Resources/HM_HairStrands, Kajiya-Kay,
alpha-to-coverage, tip emission). v3 captures never create the old LineRenderer HairSimulation.

`shoes.json` (optional, dancecap.shoes; likeness data - work/ and git-ignored StreamingAssets only): per dancer
{style court_platform|runner|neutral, upper, sole, sole_heel, outsole, accent, lace, lining, heel_tab (#rrggbb sRGB,
graded like the albedo bake; white parts balanced to a neutral white), sole_heel_m, sole_fore_m, toe_spring_m,
heel_bevel_m, source, confidence, measured}. takes toml `[dancers.<role>] shoe_geometry = {sole_heel_m, sole_fore_m,
toe_spring_m, heel_bevel_m}` (measured sole stack) overrides the style's numbers, also on `--export-only`.
Real sizes: `python -m dancecap.shoe_size <take> [--detect]` -> avatar/shoe_size.json (outer length / width / toe-box
height from SAM2 silhouettes along the RTMW-triangulated foot axis, planted feet; review/demo/shoe_size_<take>.jpg).
Every avatar gets procedural sneakers (head-movement Assets/Shoes/SneakerPair; neutral without the file). They are built
on a worker thread at load from the dancer's fitted rest feet and fitted to the capture's motion:
- The outsole bottom is the fitted sole plane (the fit's sole = the shoe sole, see the world scale above).
- Floor contact is baked per frame: never below the floor, stances planted by at most 2 cm.
- The bare feet are cut along each shoe's top edge (SmplxAvatar.SetFootHeights / HideFeet).
- The rear upper and the collar are sized to the visible leg over the whole capture.
- The shoes share the avatar's opacity.
`python -m dancecap.shoes <take> --export-only` re-copies it after every export.
`python -m dancecap.demo_export <take> --capture <Name>` does the export (no room, studio soundtrack when present)
and every add-on (hair groom + its colliders re-sized for the shape edits, shoes, faces, counterbalance, moves) and
validates the folder (work/<take>/export/<Name>_demo_export.json); `--hooks-only` re-runs the add-ons.

Unity (head-movement): `CaptureTimeline` maps audio time -> frame by binary search over times[] + time_to_audio
(nearest frame); `SmplxAvatar` builds a 55-bone SkinnedMeshRenderer per dancer (layer `avatars`; textured with
Resources/HM_AvatarLit = opaque, one texture, main light + SH, block-compressed on load when smplx_albedo is
present, else translucent role tint); `RoomMesh` (layer `room`, key M) loads room_mesh.bin; `SplatCloud` (layer
`splats`) loads environment_splat - off by default when a room mesh exists until the user toggles it; overlays
`TimingOverlay` (layer `timing`) and `PhysicsOverlay` (layer `physics`, every visual hidden on frames with a
non-finite COM / F_net / band) load only when their files exist. All capture-space objects are placed with
`DanceOrigin.Place`.

Unity conversion from the Y-up right-handed world: `unity = (x, y, -z)`; rotation quaternion
`(qx, qy, qz, qw)_world -> (-qx, -qy, qz, qw)_unity`.

## moves/ and counterbalance/ - see docs/MOVES.md

```
taxonomy/zouk_graph.json              [moves graph] the zouk dance graph (tracked): Zouk1.json + taxonomy/additions.toml
                                        (+ taxonomy/vocabulary_research.json); version 1, contract in dancecap/moves/graph.py
work/<take>/moves/labels.json         [moves] couple-tier move timeline (reference | video time), provenance
                                        human|narration|auto|placeholder; contract in dancecap/moves/labels.py
work/<take>/moves/path.json           [moves path] Viterbi node instances + links, new links flagged (moves/path.py)
work/<take>/moves/fingerprint.json    [moves fingerprint] dwell per node, link counts, energy bands, counterbalance
                                        share (moves/fingerprint.py); review: work/<take>/review/fingerprint.png
work/<take>/counterbalance/counterbalance.json   [counterbalance] intervals (pivot, leader circle, shared COM track)
StreamingAssets/<capture>/moves/{graph,labels,path,fingerprint}.json   capture time (= reference + time_to_audio),
                                        Unity frame; written by `python -m dancecap.moves export <take>` - re-run it
                                        after every `python -m dancecap export` (that wipes the capture folder)
```

## counterbalance/ and floorcraft/ - the floor record (2026-10-07)

```
work/<take>/counterbalance/truth_user.json   [human label] {"counterbalance": [{t0, t1 (reference s), hud, capture_time}],
                                        statement, time_base_check}: the user's interval read in the viewer HUD
                                        (HUD = reference - times.json[0]); `counterbalance detect` scores against it
                                        (else work/<take>/truth.json for synthetic takes)
work/<take>/counterbalance/counterbalance.json   [counterbalance] version 1, additive fields: intervals[].overhang_m
                                        {lead, follow} (median COM beyond the own support edge, m), intervals[].hanger
                                        (role that hangs), track.overhang_{follow,lead}_m; params.lean_mode
                                        ('counterweight' | 'both'), overhang_m, counter_lean_m, support_soft_m,
                                        supported_w, hand_body_extension; source.code_sha1 (sha1 of
                                        counterbalance.py: counterbalance.is_current = same lead npz, beat grid AND
                                        detector code -> a changed criterion / threshold re-detects)
work/<take>/floorcraft/floorcraft.json  [floorcraft] python -m dancecap.floorcraft <take> [--export-capture <name>]
                                        [--review-dir <dir>]
                                        version 3 (2026-10-07; ADDITIVE over version 2: every v2 key below keeps its
                                        meaning; new: t_axes, transitions, axis_pivots, dials, balance, evaluations -
                                        renderer-facing schema with drawing notes: work/larissa_kadu_38/review/
                                        floorcraft2/SCHEMA_V2.md), kind "floorcraft", frame "world", time_base
                                        "reference". The track is CUT at every counterbalance pivot: no plant, move or
                                        rotation overlaps a pivot (the pivot owns the leader's circling: pivots[].
                                        leader_axis); a renderer draws every record as it is, nothing is skipped:
  axis       {long_m 2.0, short_m 0.8, origin, source}: the live leader-axis size (VIEWER_SPEC 3.6) and how it is built
             (spine3 joint on the floor; long = spine3 global +x projected = the leader's LEFT; forward = its floor
             perpendicular on the chest's front)
  track      {t, origin [[x,0,z]], yaw_deg (unwrapped), speed_m_s, step_s}: the smoothed axis at 15 Hz
  plants[]   {id, t0, t1, duration_s, duration_beats, origin [x,0,z] (centroid), radius_m, radius_p90_m, drift_m_s,
             yaw_deg / forward / long_axis (unit [x,0,z]) = the axis he ESTABLISHED: circular-mean yaw over `held`
             [t0, t1], the plant's longest stretch without a rotation in progress (yaw_source 'held'; 'mean' /
             'mean (turning throughout)' = the whole-plant mean), yaw_mean_deg (whole-plant circular mean),
             yaw_start_deg, yaw_end_deg, turn_deg}. One stand = one plant: neighbours whose union passes the plant
             test, or that touch with centroids <= 0.20 m apart, are merged; plants never overlap
  moves[]    {id, kind step|travel, t0, t1, duration_s, from_plant, to_plant (null at the take edges and next to a
             pivot), from_pivot, to_pivot (the pivot it leaves / reaches, else null), from, to (plant origins, or the
             axis origin at a take edge / pivot edge), path {t, pos} = from -> axis origin -> to (draw `path` alone: it
             joins the plant crosses), length_m (of path), origin_path_m (the origin's own path between the plants),
             displacement_m, heading_deg, from_yaw_deg, to_yaw_deg, yaw_change_deg (unwrapped, full turns kept)}.
             kind 'step' = one step between two plants (origin_path_m < 0.5 m within 1 beat; when two plants touch,
             the step is cut out of their edges at the origin's speed peak, else it sits at their junction, 1-3
             frames); 'travel' = the rest
  rotations[] {id, t0, t1, duration_s, from_yaw_deg, to_yaw_deg, signed_deg, direction ccw|cw, rate_deg_s,
             context plant|move, plant (owner plant | null), plants (every plant it touches), move (owner step /
             travel move | null), centre [x,0,z] (owner plant origin, else the axis origin mid-rotation),
             swing {index, deg, t, parts}}. Swings split only where they cross a travel move: plants chained by
             steps are one context, so a turn while stepping to the next stand stays one rotation
  pivots[]   the counterbalance.json pivots {id, counterbalance (interval index), t0, t1, counterbalance_t, pivot,
             pivot_foot, leader_radius_m, swept_deg, swept_signed_deg, direction, arc_start_deg, confidence,
             leader_path, beat_ticks} + leader_axis {t0, t1, path {t, pos} (his chest on the floor), length_m,
             from_yaw_deg, to_yaw_deg, turn_deg (net, unwrapped), direction} (null when the pivot lies outside the
             fits); counterbalances[] {id, kind, t0, t1, confidence, com_floor, connection, pivots}
  events[]   {type plant|move|rotation|pivot|counterbalance|t_axis|transition|axis_pivot|dial, id, t0, t1 (+ kind
             for moves and dials)} by t0 - the viewer's accumulating record
  --- version 3: the leader's STABLE T, its reorientations, dials, the balance axis (user 2026-10-07) ---
  --- revision 3.1 (review fixes 2026-10-07, additive: top-level "revision"; from_t / to_t may be null) ---
  t_axes[]   {id, t0, t1 (episode: forming start .. the first frame of the sustained exit / rotation that ended it),
             t_formed (end of the forming window), origin [x,0,z] (MEDIAN chest-on-floor over ALL the T's active
             frames inside its region with the heading within 30 deg; 3.0: over the forming window), yaw_deg
             (wrapped) / yaw_unwrapped_deg (MEAN heading over the same frames), formed_origin / formed_yaw_deg (the
             forming window's values), dropped_before (brief Ts dropped since the previous T), forward (the stem) /
             parallel (the crossbar = the leader's LEFT) unit [x,0,z], start_reason / end_reason start|exit|rotation|
             exit+rotation|counterbalance|gap|end, same_place_as (id of the previous T when this one is that T
             rotated in place: origin copied), prev, next, pivot_in, pivot_out (axis_pivots ids), dial, excursions,
             occupancy {forward_m [min,max], side_m [min,max], in_region_share, yaw_dev_p90_deg, centre_offset_m
             (median chest over all its active frames -> origin)}}. Origin and heading are averaged and the end
             re-checked against them (t_refine_iter), then FIXED; the T holds while his chest stays in the ALLOWED
             REGION = stem corridor (|side|
             <= 0.30 m, |forward| <= 1.2 m) U crossbar corridor (|forward| <= 0.30 m, |side| <= 1.2 m) with his
             heading within 30 deg; >= 1 beat outside ('exit') or turned > 30 deg ('rotation') starts a new T. A
             forming window (1 beat) is SETTLED: all its frames in its own region, headings within 20 deg of its mean,
             net drift <= 0.20 m/s along the stem OR the crossbar (a diagonal arrival forms no T). A new T within
             0.25 m of the previous one and within 30 deg of its heading CONTINUES it (excursions + 1). No T forms
             inside a counterbalance pivot (its dial owns that motion). A T holding < 1 beat after forming is DROPPED
             (it would only flash; the take's last T is kept) - its heading stays a WAYPOINT of the axis pivots.
             Between Ts there may be no current T (travelling / turning)
  transitions[] {id, from_t, to_t, t0 (from_t's t1), t1 (to_t's t0), from, to (T origins), distance_m, same_place,
             reason (= to_t's start_reason), axis_pivot (the first between them, null across a counterbalance),
             axis_pivots [ids on the way], dropped_ts, dial, counterbalance_pivot, path {t, pos} (his chest between
             them, informative)} - consecutive T origins = the dotted line
  axis_pivots[] {id, t0, t1 (heading leaves the old T heading / reaches the new one, within min(15 deg, 25 % of the
             turn)), at [x,0,z] (the old origin when stayed, else his chest mid-turn), from_t, to_t, from_yaw_deg,
             to_yaw_deg, signed_deg (unwrapped, + = ccw, may exceed 360), abs_deg, direction, stayed, dial,
             context between_ts | lead_in | into_counterbalance | out_of_counterbalance | lead_out, from_kind / to_kind
             t_axis | brief_t | track_start | track_end | counterbalance (from_t / to_t null at a non-T end),
             via_brief, counterbalance_pivot, evaluation} - every reorientation of his axis > 30 deg: T -> T through
             the waypoints (a heading change <= 30 deg to a waypoint is passed over), track start -> first T, T -> his
             heading where a counterbalance pivot begins, where it ends -> T, last T -> track end (the counterbalance
             dial owns only the circling inside the pivot)
  dials[]    {id, kind counterbalance|leader_spin|couple_turn|leader_around_follower, t0, t1, centre [x,0,z],
             degrees (signed cumulative, + = ccw, can exceed 360), abs_deg, turns, direction, who (text), subject
             lead|couple|both, radius_m, arc_start_deg (needle at t0, from +x towards -z), heading_deg, orbit_deg,
             centre_spread_m, counterbalance_pivot, axis_pivots [ids], t_axis (the associated T: the one it starts
             from), evaluation}. counterbalance = every counterbalance pivot (centre = her anchored foot, degrees =
             swept_signed_deg); the others = consecutive same-sign axis pivots (each <= 2 beats after the previous)
             summing to >= 180 deg while one point stays put (p90 distance to its median <= 0.30 m): the leader's
             chest (leader_spin), the couple's combined COM (couple_turn) or her chest (leader_around_follower) -
             tried from the least moving; the last two only when the leader's orbit about the centre has the turn's
             sign and >= 0.6 x its degrees (3.1; else it is his own spin, not a turn about that point); a chain that
             fails is tried in its contiguous sub-chains (3.1); pivots chain through a T or a waypoint; turning into /
             out of a counterbalance is never a dial; a turn while travelling is no dial (params.turn_groups lists
             every chain tried with spreads, orbits and rejections)
  balance    {t (every analysed frame), frame_step_s, stance_codes, com_source, lead|follow {stance [0 none|1 left|2
             right], double [0|1], support [[x,0,z] | null], chest [[x,y,z]] (spine3), com [[x,y,z]], com_offset_m,
             tilt_deg, score}, couple {com, support, com_offset_m, tilt_deg, score, pivot} (null outside the
             counterbalance pivots)}: the BALANCE AXIS chest -> support of the stance foot. Contact per foot = joint
             height above the dancer's floor (4..9 cm) x foot speed (0.4..1.0 m/s); stance = the foot in contact
             whose sole line (ankle + s (ball - ankle), s -0.35..1.35 = heel..toe tip) is nearest the COM's floor
             point; support = that nearest sole point; score = min(offset ramp 4..12 cm, tilt ramp 3..9 deg) in 0..1
             (line brightness). couple: the combined COM (masses: takes/<take>.toml mass_kg, else the body-model
             mass) over the follower's ANCHORED foot inside each counterbalance pivot. COM = shaped SMPL-X mass
             elements (counterbalance.body_com = physics_body) on the analysed fits, 5 Hz spline
  evaluation {subject, measure, frames, aligned_share (score >= 0.5 = COM <= 8 cm off the sole line and tilt <= 6
             deg), mean / median / p90_offset_m, mean_tilt_deg, mean_score, verdict good (>= 60 % aligned) | ok (>= 35
             %) | needs work | unknown, text, window [t0, t1] (the turn, widened to >= 0.5 beat), per_dancer {lead,
             follow}, worse (subject both), note, graded} on axis_pivots[], dials[], pivots[] (= its dial's) and
             rotations[] (subject lead). 3.1: reorientations (axis_pivots, rotations) also carry turn_kind one_foot |
             both_feet | stepped | airborne, turn_kind_label, feet {double_share, contact_share, ball_travel_m,
             stance_share, stance_changes, pivot_foot}: only a ONE-FOOT pivot (one foot in contact >= 80 % of the
             turn, the weight on it >= 70 % of the stance frames, its ball within 10 cm) is graded - over the frames
             with the weight on that foot; both feet (>= 70 % double support) / stepped -> verdict "not graded",
             graded false, balance_verdict (the COM test alone). Dials are always graded. ESTIMATES from the fitted
             bodies
  params     + t_axis {half_width_m, forward_m, side_m, pivot_deg, sustain_beats, form_beats, form_yaw_deg,
             form_drift_m_s, same_place_m, pivot_edge_deg, draw_forward_m, draw_side_m, min_hold_beats, refine_iter},
             dial {min_deg, place_m, max_hold_beats, orbit_ratio}, turn {double_share, pivot_contact, pivot_stance,
             pivot_travel_m}, balance {...}, t_axis_share, turn_groups, com_source, counts + dropped_short_ts,
             axis_pivots_by_context, axis_pivots_by_turn_kind, axis_pivots_graded (also flat: t_*, dial_*, bal_*,
             turn_*)
  source     {smplx_dir, lead_sha1, follow_sha1, lead_probe, t_range, counterbalance, counterbalance_sha1,
             masses_kg, code_sha1, ...} (floorcraft.is_current: version 3, same lead AND follow npz, beat grid
             (beats_sha1), counterbalance.json and floorcraft.py (code_sha1) -> no re-analysis)
  File layout: one key per line, arrays of numbers / points on one line (floorcraft.dumps; plain JSON).
  Dashing: a counterbalance is drawn dashed when its INTERVAL confidence (counterbalances[].confidence =
  counterbalance.json intervals[].confidence) is < 0.6; pivots[].confidence is the pivot-geometry score (informative,
  not used for dashing).
  Angles: yaw = angle of the forward axis on the floor from +x towards -z = ccw seen from above (right-handed about
  +Y); signed angles + = ccw. Thresholds in params (plant radius 0.20 m, drift 0.15 m/s, >= 1 beat; touch
  0.10 m / 0.15 s; step < 0.5 m within 1 beat; rotation 25 deg).
work/<take>/review/floorcraft_map.png, floorcraft_timeline.png   [floorcraft review] overhead map + segmentation
work/<take>/review/taxis_map.png, taxis_timeline.png   [floorcraft review, version 3] the Ts (current bright, old
                                        faded teal), dotted transitions, axis pivots / dials with verdicts, the
                                        viewer's floor at six instants; lanes + heading vs held T heading + offsets
                                        from the current T + balance scores (--review-dir moves all four)
StreamingAssets/<capture>/floorcraft.json   Unity copy (frame "unity" = (x, y, -z) before the dance-origin offset,
                                        time_base "capture" = reference + time_to_audio; *_reference keep the
                                        reference seconds; yaw / signed angle numbers unchanged (ccw from above) plus
                                        *_euler_y_deg = 90 - yaw (Quaternion.Euler(0, y, 0) * forward) and
                                        *_unity_euler_y = -signed; plants also euler_y_mean_deg, held_reference;
                                        pivots[].leader_axis converted like a move; version 3: t_axes euler_y_deg /
                                        euler_y_unwrapped_deg, t_formed_reference, axis_pivots *_euler_y_deg +
                                        signed_deg_unity_euler_y, dials degrees_unity_euler_y + arc_start_deg_unity
                                        (= arc_start_deg: from +x towards Unity +z), evaluation.window(_reference),
                                        balance t (capture) + t_reference, support / chest / com mirrored, nulls
                                        kept); capture.json gets "floorcraft": "floorcraft.json".
                                        Written by `python -m dancecap.floorcraft <take> --export-capture <name>` and by
                                        every `python -m dancecap.demo_export` (with counterbalance.json; both
                                        re-analysed only when missing or stale, both validated)
```

## Pipeline Monitor and orchestrator - `dancecap.pipeline` (2026-10-07)

What the orchestrator, the run wrapper and the monitor write (docs/PIPELINE.md, docs/PIPELINE_MONITOR.md, user guide
docs/PIPELINE_MONITOR_GUIDE.md). All under work/ (git-ignored); nothing here is read by a stage as an input except
the clock file of a wizard take.

```
work/<take>/intake.json                 [intake] the registered videos (path, size, probe, exclusions), warnings
work/<take>/probe/probe.json            [probe] ffprobe facts per camera
work/<take>/probe/camera_types_suggested.json   [probe A] suggestions per camera (+ `file` = {name, size, mtime_ns}:
                                        the wizard's staged result is written here at intake and reused by the
                                        probe stage instead of decoding the video again)
work/<take>/clock/alignment.json        [audio_sync] the audio clocks of a WIZARD take (identity for a single view);
                                        older takes keep theirs outside work/ (never overwritten)
work/<take>/pipeline/state.json         [orchestrator] per-stage state of the last run (dry runs say dry_run: true and
                                        never mark a stage running / failed)
work/<take>/pipeline/events.jsonl       [orchestrator] append-only: plan, start, would_run, would_pause (what-if),
                                        end, fail, gate, note, stop, finish
work/<take>/pipeline/run.lock           one orchestrator per take (CLI or monitor)
work/<take>/pipeline/gates.json         review gates the orchestrator opened (fit_qa, geoveto) and their decisions
work/<take>/pipeline/camera_types_applied.json  {stage: {cam: class}, policy} the type-reading stages applied
work/<take>/pipeline/accepted_inputs.json       stale stages kept on purpose ("Keep outputs")
work/<take>/monitor/probe_cache.json, toml_backups/   the monitor's ffprobe cache; a backup before every TOML write
work/pipeline/plans/<sha1>.json         plans (no side effects); an approval is bound to one run id and used once;
                                        a what-if plan (options.assume_reviews) is dry-run only
work/pipeline/runs/<run id>/            run.json, pid.json, launcher.json, output.log, current.json, stop.json, rc.json
work/pipeline/runs_archive/             finished runs older than 14 days
work/pipeline/slots/, work/pipeline/pod_session.lock   the local CPU / GPU slots and the one-pod-session lock
work/monitor/                           monitor.lock (single instance), server.json {pid, port, instance} (never the
                                        token), settings.json, staging/h-*.json (picked files), logs/{monitor,launcher}.log
```

## Recap cameras (2026-10-09, additive; dancecap.cameras.newcam, dancecap.recap_cameras)

Take `larissa_kadu_recap`: seven NEW handheld phones (not in the room model) localised against the world package's room model. Outputs in
`work/larissa_kadu_recap/cameras/`, same schema as `cameras/<cam>.npz` above, with these additive keys: `pts` (video time of the frame, s), `frame_index` (index of the frame in the
video = index in the owner's 2D pose JSON), `source` (`map` | `track` | `interp`), `interpolated`, `localised` (the frame itself was solved, not interpolated), `frame` (`sfm` | `sub_05` ...),
`work_size`, `native_size`. `K` is at the camera's DISPLAY pixel size (the size of the 2D pose JSONs; ffmpeg autorotate), `dist` = k1 of ONE focal per video (`type_model` fixed_focal).
`t` is the reference clock of `alignment_recap.json` (audio clocks; the video offsets are dancecap.sync's job): the track is per video frame, so a later offset correction only shifts `t`.
`<cam>_world.npz`: the same pose in the metric world frame (R world -> camera, tvec, C, K, dist, ok). `<cam>_subframe.npz`: a phone localised in a SUB-MODEL frame (05, 06): NOT in the room
frame, never read as `<cam>.npz`; `cameras/<cam>.npz` of such a phone exists only when `dancecap.cameras.attach_people` wrote it (attached = True, rotation-only, one centre).
`calib/<cam>.json` (focal, k1, anchors, reference images), `chunks/<cam>_<k>.npz` (the 60 s pod outputs with 4 s overlaps), `report.json` (quality, verdict usable | partial | unusable with the
reason, coverage, epipolar / people checks), `attach_<cam>.json` (the evidence of an attach), `review/cameras/` (contact sheets, coverage map).

## Recap people and chunk plan (2026-10-09, additive; dancecap.recap_people, recap_attach, recap_front, farm/recap_profile)

Who the two teachers are in every camera of `larissa_kadu_recap`; identity-free inputs are the owner's 2D pose JSONs and R2's camera tracks. Files in `work/larissa_kadu_recap/people/` (git-ignored):
```
pose_tables/<cam>.npz        kp (F, P, 17, 3) float32 [X, Y, conf] in DISPLAY pixels, NaN padded; n (F,) persons per frame. Row r = video frame r = `frame_index` of cameras/<cam>.npz
world_people.npz             3D people of every 30 Hz instant (the four localised phones 00-03, matched by the epipolar cost of their joints, triangulated, metric room frame Y up)
track3d.npz                  the two anonymous teacher tracks (slot A / B): k (grid index), obs, X (J, 2, 17, 3), nv, res
assoc_<cam>.npz              per canonical frame of cameras/<cam>.npz and slot: row (-1 none), cost, kind (0 none, 1 member of the 3D candidate, 2 projected observed 3D, 3 projected interpolated 3D, 4 followed in 2D),
                             nj, gap, inside
appearance_<cam>.npz         colour descriptors (dancers_appearance) of the associated persons per canonical frame
roles.npz                    state (B,) 0 = slot A is the lead, per bin of bin_s (one 30 Hz frame), p_a_lead, e, n_obs
teachers_<cam>.npz           THE OUTPUT, per canonical frame i of cameras/<cam>.npz: t, frame_index, ok, row (F, 2) int16 [lead = Kadu (pid 1), follow = Larissa (pid 2)] = the row of the person in pose_tables
                             (-1 none), box (F, 2, 4) xyxy (from the joints: the JSON has no boxes), kp (F, 2, 17, 3), conf (F, 2) 0..1 (use >= 0.5; >= 0.3 is a guess), cost, kind, nj, gap, merged (one detection for
                             two bodies: the role is a guess), role_conf (F,), lead_slot, size, attached
teachers.json, coverage.json summary and the per-second coverage ({lead, follow, both: [[cams]] per master second, per_camera shares, cameras-per-second histograms})
verification.json, detcompare*.json   zone / height / colour checks; the owner's JSON versus the pipeline's own detector (all frames and the frames the JSON misses)
attach/<cam>.json|npz|_world.npz|assoc_<cam>.npz   R3's teacher-refined attach of the rotation-only phones 05 / 06 (R2's schema, only R / tvec differ; `adopted` = held-out gate passed)
```
`work/larissa_kadu_recap/chunks.json`: the chunk plan (arithmetic, windows, cores, seams, cameras and the reason per camera, shape window, cost table, offline validation); the farm jobs are in `farm/jobs/`.
Chunk takes (`work/larissa_kadu_recap_<id>/`) additionally hold `dancers/teacher_tracks/<cam>.npz` (idx (F, 2) = person row of lead / follow in `people/<cam>.npz`, -1 = MISS; boxes (F, 2, 4); conf) which `recap_front`
feeds to the demo's SAM stage as its candidate tracks (slot A = lead, slot B = follow, the colour rule's keep / swap decision forced to keep).


## john_lula 2D people (2026-10-11, additive; dancecap.jl_people, jl_people_detect / _sam / _pose / _pod; agent J3a)

One handheld camera (cam `01`), a floor heat with other couples: `work/john_lula/people/teachers_01.npz` follows the recap people contract above (row = slot in `pose_tables/01.npz`, box = SAM mask box, kp 17 joints, conf >= 0.5, role_conf,
merged = the two mask boxes overlap > 0.5, floor = NaN until the room calibration) and ADDS: `kp_wb` (F, 2, 133, 3) COCO-WholeBody from RTMW-x (17 body, 6 feet, 68 face, 42 hands; score = min(1, RTMW peak / 6)), `wb_variant`,
`mask_area`, `mask_agree`, `mask_dist`, `visible` (zoom-normalised), `skel_in_partner` (> 0.5: the dancer is hidden, the skeleton is a guess), `edge`, `box_iou`, `bib_dist`. `masks_01.npz`: `m_<frame>_<0 John | 1 Lula>` PNG bytes + box, area.
Identity = competition bibs (John 124, Lula 116; 126 is another couple) read by EasyOCR on a pod, anchors from a pair Viterbi, SAM 2.1 masks between anchors (forward + backward runs, their IoU = `mask_agree`). Method, checks, hard stretches: `work/john_lula/NOTES_J3a.md`.


## mono4d - one phone -> 4D provider -> virtual cameras (2026-10-15, additive; dancecap.mono4d, docs/MONO4D_NOVEL_VIEWS.md)

`work/<take>/mono4d/<run>/` (run = `<provider>_<cam>_<t0 ms>_<t1 ms>` unless named):
`request/request.json` (Mono4DRequest: take, input_cam, provider, t0, t1, grid = the 30 Hz master grid of the window, video, video_fps,
video_frames, targets, allow_upload, notes); `request/inputs/<cam>.mp4` (every canonical frame of the window, upright, at the window's mean
rate) + `<cam>.times.json` {t, source_files} (frame i = t[i]); `request/source_trajectory.json` and `request/target_trajectory.json`
(`dancecap.mono4d.trajectory/1`: time_s, frame "world" | "scene", world_from_scene, cameras {name: {kind ring | holdout | source, width,
height, static, frames [{t, ok, fx, fy, cx, cy, c2w_opencv 4x4, c2w_opengl 4x4}]}}; a static camera has one frame entry; source = per INPUT
frame, targets = per grid time; ring cameras v00.., held-out real-phone poses h<cam>); `request/REQUEST.md`; `drop/` (the delivery:
`views/<name>.mp4` or `views/<name>/NNNNNN.jpg` with exactly one frame per grid time at the requested size, or `asset.mint` /
`splats/frame_NNNNN.ply` (+ `scene_cameras.json` {input: [T x 4x4 c2w, OpenCV axes, scene frame]} or `world_from_scene.json` Sim3);
`static_world.ply` from the World API); `provider_job.json` (ids only, never a key); `result.json` (Mono4DResult: provider, kind views |
mint | ply_sequence | static_splat, dynamic, views {name: path}, asset, world_from_scene {s, R, t, rms m, rot_err_deg {median, p95}, n,
spread_m}, scene_meta, checks, notes); `render_job.json` (render_gracia.py input: COLMAP qvec/tvec in the asset frame, times in asset
seconds); `judge.json` (judge.judge_views: views {name: {frames [{t, path, ok, why, answers, model}], judged, errors, bad, bad_frac, ok}},
input_tokens, model, thresholds); `views2d.json` ({h<cam>: holdout.compare_views}); `holdout.json` ({reference, test, baseline, window,
route, baseline_fit, verdict}); `quality_gate.json` (gate.decide: admit, views_admitted, views_dropped, why, notes, checks, thresholds).

Virtual take `work/<take>__m4d_<run>/` (virtual_take.package): the CONTRACTS layout for frames/ and cameras/ (generated views: one frame
per grid time, t = grid time; the real input phone: its own frames and track over the window), `cameras/world_from_sfm.json` = identity,
`mono4d.json` (`dancecap.mono4d.virtual_take/1`: source_take, run, provider, input_cam, real_cams, synthetic_cams, independent_views = 1,
window, grid [t0, t1, n], world_from_scene, excluded, warning). Rule: generated views are the provider's prior, never measurements;
h<cam> views are for comparison with the real phone and stay out of the fit set; nothing enters MAMMA without quality_gate.json admit.

`work/mono4d/vendor_watch/<date_time>.json` (`dancecap.mono4d.vendor_watch/1`: vendors {v: {pages [{url, passages, chars, model,
input_tokens, capabilities {cap: {stated, ruled_out, evidence_id, evidence_conf, evidence}}}], decision {cap: {status yes | no | unclear,
p, url, evidence}, _ready}}, changes); `work/mono4d/calibration/` (labelled mannequin frames, labels.json, calibration_<variant>.json).

## Two-view takes (2026-10-23, additive; dancecap.cameras.twoview, dancers_seed, farm/ca_profile.TwoView; lucas_luiza)

A take filmed by TWO cameras (one tripod + one moving camera). Module docstrings hold the full rules; the keys and files:

```toml
[cameras]
x_cam = "02"                                 # the camera whose frames the map's cross-camera pairs start from (default: the first)
map = { win_step_s = 2.0, dense_step_s = 6.0, shared_intrinsics = ["01"] }   # mapping.map_settings: map image steps; one focal for a no-zoom video
[sync]
required = true                              # mamma prepare / the farm refuse without this take's own sync/offsets.json
```

```
cameras/sfm_work/map.json            + intrinsics_mode {<cam>: single | shared | per_image}, verification.per_campair.<a>-<b>.link_pairs
                                       (pairs >= link_min_inliers), link_min_pairs
cameras/qa/map_gate.json             twoview map-gate: registered / total per camera, link per --need pair, refined reprojection,
                                       doppelgangers {flagged, per_cam, kept_majority} of cameras/sfm (G1 no_doppelgangers wants 0);
                                       mean_reproj_px = the written model's (after any clean)
cameras/sfm_orig/                    cameras.plausible clean: the model before its doppelganger images were deregistered (belongs to
                                       the current map: stage_map / cameras.reuse remove it when they write a new cameras/sfm)
map.json doppelgangers               {flagged {image: why}, kept_majority, per_cam, passes, bands {up, room, floor, box {centre, h_lo,
                                       h_hi, radius, points, n}, exempt, cams {<cam>: h_med, h_tol, xz_med, xz_tol, n}}, rule,
                                       removed {images, points_before, points_after}, at?, how?, history? [earlier cleans]}: stage_map
                                       (before the final refine) or `cameras.plausible clean` (in place, no BA: frame kept). Implausible =
                                       outside the camera's band AND outside the room (points seen by >= 2 cameras)
cameras/cache/loc_<cam>.npz source   + "implausible": the best pose was implausible (ok False, no corr). Main-model localisation and
                                       map_extend only: single-camera sub-models (attach, reuse, map_extend) are never gated
cameras/cache/loc_<cam>_raw.npz      cameras.anchored: the moving camera's raw localisation (written once; every run starts from it)
cameras/cache/loc_<cam>.npz source   + "anchor" | "cont" | "prior" (ok) and "dancers" | "unanchored" (not ok) after cameras.anchored run:
                                       the moving camera judged against the fixed one by the dancers (epipolar + ankle transfer)
cameras/qa/anchored_<cam>.json       anchored run: offset_ms (+ from), raw judged / residual_pct / transfer_pct / anchors, map_images
                                       {judged, total, bad}, map_images_excluded, params, counts {anchor, cont, prior, refused,
                                       unanchored, ok, frames, share}, coverage_8s, residual_after_pct, transfer_after_pct;
                                       anchored_score_<cam>.json = the `score` (CPU) half with every map image's verdict
cameras/ptz/<cam>.npz                twoview ptz --mode attach: the tripod's fixed-centre PTZ track in the cameras/<cam>.npz schema
                                       (camera_class tripod_zoom); --adopt installs it, cameras/orig/<cam>_loc.npz + cameras/adopted.json
cameras/qa/ptz_check.json            check / attach sections; cameras/qa/atlas_check.json + review/atlas_check.png (Atlas = drift check only,
                                       never a scale or focal prior)
sync/offsets.json                    method "two_view_epipolar", gauge = the tripod (offset 0), dancecap.sync's layout + clocks block;
                                       sync/curves_2v.json (+ piecewise {windows [{t, delta_ms, even_ms, odd_ms, n, counted}], curve,
                                       share, repro_median_ms, repro_max_ms, range_ms, cost_px, cost_constant_px, reproducible}),
                                       review/sync_2v.png. Never written with use: false. When no constant holds but the even- and
                                       odd-frame curves reproduce delta(t) (a moving camera's audio clock follows its distance to the
                                       speakers, ~3 ms per metre): offsets.<cam>.piecewise [[t, delta_s], ...] + piecewise_detail,
                                       stable_on "piecewise: ...", use_rate false; delta stays the whole-window value.
                                       mamma_dyn.read_sync_offsets passes piecewise through and apply_offset / offset_label use it
                                       (every reader); farm TwoView.offset_ms takes it at the chunk's centre
cameras/qa/twoview_check[_<tag>].json  sync-corrected two-view DLT of COCO-17 joints: reprojection, coverage, bone CV, statures, ankle heights
cameras/qa/twoview_gates.json        G1-G8 {gate: {value, threshold, pass, status, checks}}, _meta (adopt_ptz rule)
people/teachers_<cam>.npz            dancers_seed: the recap teachers contract on this take's own detections (reviewed slots, src_frame, src)
dancers/seed_report.json             G0 of the seed (scan offsets, matched share, skeleton vetoes, role jumps); review/seed_roles.jpg
dancers/roles.json                   ONE entry per frames.json frame (positional), dancers_pod.merge's schema
song/segments.json                   + shifted_from {take, dt_s} (heat_beats shift: a verified grid moved onto another take's clock)
```

Farm: `farm/ca_profile.PROFILES` (`carlos_aline`, `lucas_luiza`): `TwoView.sync_for(chunk)` = the measured offsets at the chunk's centre
(delta + rate (t - t_ref) where use_rate), master = `HM_FRONT_MASTER` (lucas_luiza: 02); job ids `ll*`, owner `LL2V`; the two-view share
gate (recap_fitreport `TWO_VIEW_MIN_SHARE`, lucas_luiza 0.7) fails a chunk fitted mostly from one view.
