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
                                          p5_mm, p95_mm, min_mm, n_below_15mm, frac_below, warning (> 10 % of frames)}
                                          = posed lowest mesh vertex over all frames (LBS, no pose correctives;
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
                                          LightGlue matches, raw per-frame localisation (loc_<cam>.npz), attach_<cam>.npz,
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
  mamma/                                [mamma] split in prepare (local) / fit (Runpod) / select (mamma_dyn.py):
                                          ma_2d_native/<cam>.npz (+ _boxes.npy, .manifest.json: recomputed when
                                          masks/frames/weights change), ma_2d/<take>_<tag>/cam<cam>.npz + ma_cap/
                                          <take>_<tag>/gt/ (30 Hz grid inputs per camera set, grid inside EVERY camera's sync-corrected coverage; gt npz carry t,
                                          native_index, img_rel_path relative to work/<take>), ma3d_config.yaml
                                          (MAMMA contact config + dancecap overrides: light acceleration prior in
                                          second_run), prepare.json (grid t + grid.trim {window, start/end_limited_by:
                                          cameras whose sync-corrected coverage trimmed the take window}, sets, eval
                                          set, sync offsets used + sync.required,
                                          world_frame), pod_job.json (what the pod uploads/runs/downloads),
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
                                          avatar_hair.red_cast = CIELAB a* > 9 and a* - b* > 3), follow hair_sample;
                                          hair_region [{pass, role, head/prior/face/face_outside_core/ear hair fractions,
                                          outside_scalp_hair_fraction_of_head, evidence_above_thresh_outside_allowed_head_
                                          fraction, leak_guard_dropped(_pct_of_measured), bad_seed_texels}], hair_masks,
                                          roles.follow.fill_guards (leak guard counts, fill classes, skin class)
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
 head-movement StreamingAssets capture (see below)
```

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
                                          reports chunks_merge.json holdout {cam, frames, lead/follow {held_out_median_px,
                                          in_fit_median_px, ratio}}: informational, never gates)
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
                                          the shape unification, per chunk median/p95), seam_mm (the two chunks'
                                          disagreement inside each overlap, median/p95/pelvis), chunks_had_repairs; merged
                                          vs per-chunk median reprojection; chunk_frames; gate {passed, why, checks,
                                          thresholds}: per role every seam p95 <= 60 mm and pelvis median <= 40 mm, betas
                                          swap p95 <= 40 mm, merged median reprojection <= 1.25 x the chunks' + 0.5 px, no
                                          chunk above 2 x the chunks' median reprojection
work/<take>/mamma/chunks_gate.json      [mamma_chunks gate, pod, between merge and select] {gate, fallback {mode, need_min,
                                          kill_minutes_left, result, seconds}}: gate failed -> --fallback full re-fits the
                                          whole grid in one run (`mamma fit --tag <tag>`, the rejected merge moved to
                                          fits/<tag>_merged_rejected/) when the pod's kill switch (/workspace/
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
```

```
work/runpod/ledger.json                 every pod created (price, created/deleted, cost) -> budget cap (remote.py)
work/runpod/bundles/bundle_<key>.tar.gz content-keyed upload bundle + bundle_<key>.json manifest (files, hard links
                                          for identical content, raw/compressed bytes)
work/runpod/wheelhouse/<key>/*.whl      CUDA wheels built on a pod (pytorch_sdf, gsplat; cp311 torch2.5.1 cu124
                                          sm_80/90), uploaded with later bundles so bootstrap installs instead of
                                          compiling. Licensed/third-party binaries: never committed
work/runpod/sessions/<sid>/             state.json (per job fingerprint/status, step timings), result.json (pod, GPU,
                                          price, ledger cost, bootstrap seconds, per-job status/missing files),
                                          runner.log, bootstrap.log, bootstrap_check.json (import checks),
                                          bootstrap_timings.tsv, downloads/<job>.tar.gz, pod_logs/<job>/<cmd>.{log,rc},
                                          outputs/ (jobs with extract=session)
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
heel_bevel_m, source, confidence, measured}.
Every avatar gets procedural sneakers (head-movement Assets/Shoes/SneakerPair; neutral without the file). They are built
on a worker thread at load from the dancer's fitted rest feet and fitted to the capture's motion:
- The outsole bottom is the fitted sole plane (the fit's sole = the shoe sole, see the world scale above).
- Floor contact is baked per frame: never below the floor, stances planted by at most 2 cm.
- The bare feet are cut along each shoe's top edge (SmplxAvatar.SetFootHeights / HideFeet).
- The rear upper and the collar are sized to the visible leg over the whole capture.
- The shoes share the avatar's opacity.
`python -m dancecap.shoes <take> --export-only` re-copies it after every export.

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
