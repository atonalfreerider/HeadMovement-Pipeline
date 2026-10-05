# atlas_bridge

## Larissa/Kadu multiview test — 2026-10-02

NoPo4D was tested on `dances/Larissa-Kadu-demo`, including handheld motion,
zoom and the mirrored back wall. Audio alignment uses nine independent anchors
per recording and an affine clock fit with outlier rejection. Accepted fits have
8–9 anchors and maximum residuals of approximately 16–67 ms. This is not proof of
frame-exact visual synchronization; camera audio/video offsets and sound travel
remain uncertain. Contact sheets at reference times 38, 48 and 58 seconds were
visually inspected. A synthetic known-lag test recovered the correct 5.0 seconds.

All nine recordings overlap from **29.971 to 68.509 seconds in 06-Demo-John.mp4**.
The two unnumbered recordings appear to show the same camera; the clean WhatsApp
copy (09) was retained and the overlaid SaveInsta copy (08) excluded. The selected
test is **58–62 seconds**, eight distinct views, 96 frames per view at 24 fps.
Numbered views retain their supplied counterclockwise order; camera 09 is unplaced.
Rotation metadata was applied and full views letterboxed, with no person masks.
All eight output clips were decoded and checked for dimensions, rate and count.

Artifacts under `prototype/larissa_kadu/`:

- `alignment.json`: fitted clocks, anchor evidence, maximum-coverage intervals.
- `test_manifest.json`: source filenames, exact source start/end times, exclusions.
- `clips/cam*.mp4` and `synchronized_mosaic.mp4`: synchronized four-second inputs.
  Mosaic order is 01,02,03,04 / 05,06,07,09.
- `nopo4d_test/`: eight-camera, two-consecutive-frame inference at 168×294,
  790,272 Gaussians, 5.94 GB peak allocated VRAM. Source and novel views retain
  distorted/duplicated anatomy even over this short interval.
- `nopo4d_four_second/`: single 4D Gaussian prediction using source frames
  0,32,64,95 from every view at 126×224. 903,168 Gaussians, 7.10 GB peak allocated
  VRAM. A 168×294 attempt was stopped after exceeding practical GPU memory;
  the lower-resolution run completed. `dynamic_gaussians.npz` and `cameras.npz`
  preserve the learned model and per-frame camera estimates.
- `nopo4d_four_second/novel_0.1.mp4` and `novel_0.5.mp4`: 96 rendered frames at
  fixed novel cameras placed 10% and 50% between predicted cameras 03 and 04 at
  the first timestamp. Intermediate times are inferred from four observations,
  not dense reconstruction. The four-second source/render comparison is
  `original_vs_novel.mp4` (original left, novel right).

Camera averaging was disabled (`average_poses=False`) to permit per-frame poses
and intrinsics. This is an experimental adaptation, not validation of NoPo4D for
handheld/zoom capture. Both runs **failed visual quality checks**: duplicated
dancers, misplaced limbs, inconsistent room geometry and floating dark regions
(letterboxing may contribute). Mirrors were retained, and no causal attribution
to mirrors versus calibration, synchronization or model priors has been established.
The sparse temporal test also cannot validate fast dance motion between samples.
Do not feed these generated views to MAMMA or treat them as validated Unity
reconstruction assets. No skeletons, dancer identities or metric geometry were
inferred from this test.

Reproduce from the workspace root:

```powershell
.\.venv-nopo4d\Scripts\python.exe prototype/sync_larissa_kadu.py
.\.venv-nopo4d\Scripts\python.exe prototype/prepare_larissa_kadu.py
cmd /c prototype\run_cuda.cmd prototype/test_larissa_kadu.py
cmd /c prototype\run_cuda.cmd prototype/test_larissa_kadu.py --frames 0 32 64 95 --width 126 --height 224 --out prototype/larissa_kadu/nopo4d_four_second
cmd /c prototype\run_cuda.cmd prototype/render_larissa_kadu.py
```

## Local Atlas substitute experiment — 2026-10-02

**Selected prototype tool: NoPo4D.** Continue development on this backend; the
candidate comparison below is historical context, not a plan to switch methods.

Follow-up experiments: `prototype/refine_nopo4d.py` implements an experimental
100-step photometric refinement of the four-frame test. This is not the official
post-optimization implementation or a reproduction of its benchmark: it uses MSE
and local SSIM, without LPIPS, with fixed predicted cameras. Mean source-view PSNR
improved from 18.40 to 22.37 dB, but SSIM fell from 0.782 to 0.759 and visual
inspection showed worse novel-view coherence. The variant was rejected; the
134-frame baseline was not overwritten. Results are in `prototype/nopo4d_refined/`.
A separate 448×448, two-frame-per-camera test also retained duplicate bodies
(`prototype/nopo4d_square/`), so changing input dimensions alone does not fix the
cross-camera geometry. Continue with NoPo4D camera/geometry consistency work
before applying photometric refinement across the whole clip.

**NoPo4D is now runnable on this machine, but its reconstruction of Carlos/Aline
failed the visual quality gate. Do not run MAMMA on these experimental views.**
The existing Atlas provider remains unchanged; the experiment lives in `../prototype/`.

The two original clips have 134 frames each, 1920×1080, 24000/1001 fps. We used their
existing synchronization without audio realignment or dropped frames. The RTX 2080 Ti
has 11 GB VRAM. A two-camera, four-frame test at 504×280 used 9.60 GB and about 26 seconds
for encoding/export. The full 134-frame experiment uses 448×252, four-frame chunks.
It exports real learned dynamic Gaussian parameters, a per-frame Gaussian PLY sequence,
and three fixed-camera rendered videos. The middle camera is novel. This is a chunked
approximation, not a globally optimized, temporally coherent scene.

Observed failure: the two cameras' dancers appear as displaced duplicate couples in
the reconstruction. The monocular control avoids that duplication but exposes holes
even at a small sideways offset. Chunk predictions disagree by up to 12.59 degrees
in camera orientation and 0.347 of the inferred baseline in camera centre, after
approximate alignment. These are consistency diagnostics, not ground-truth errors.
A SIFT wall-feature check was inconclusive because the repeating panels generated
ambiguous matches. Metric scale, floor orientation, identity and lead/follow roles
have not been established. No MAMMA skeletons or contact/physics results were generated.

Candidate assessment (primary sources checked on the date above):

| Candidate | Fit for this project | Decision |
|---|---|---|
| [NoPo4D](https://github.com/bralani/NoPo4D) | Pretrained dynamic Gaussians from unposed multi-camera videos; explicit renderable geometry. Public checkpoint available. | Best first **local feasibility baseline** among the candidates checked. Actually tested; fails quality on this pair without further refinement. |
| [MonoFusion](https://github.com/Z1hanW/MonoFusion) | Scene optimization for sparse synchronized, calibrated video; demonstrated on human activities with four static cameras. | Considered but not selected. Two-camera performance here is untested. |
| [RiGS](https://github.com/ladvu/RiGS) | Monocular dynamic Gaussian reconstruction with motion/depth preprocessing. | Relevant monocular comparison; not evaluated locally in this experiment. |
| [Diffuman4D](https://github.com/zju3dv/Diffuman4D) | Sparse-view human novel-view synthesis using diffusion priors. | Relevant appearance candidate; generated hidden anatomy still needs independent validation. Not evaluated locally. |
| [Forge4D](https://github.com/zhenliuZJU/Forge4D) | Four-camera human reconstruction; repository recommends ≥24 GB and lists checkpoints as coming soon. | Not an immediately runnable substitute on this setup. |

[Atlas's announcement](https://www.worldlabs.ai/blog/atlas) is a capability description,
not a verified API/export contract for this bridge. None of the evaluated evidence
establishes an off-the-shelf replacement for asynchronous multi-camera capture. Keep
synchronization, calibration, foreground tracking and quality validation as separate
pipeline stages. Generated views must not be counted as independent measurements.

### Inspect the outputs

- `../prototype/nopo4d_full/view1.mp4`: all 134 frames from the novel middle camera.
- `../prototype/nopo4d_full/view0.mp4`, `view2.mp4`: reconstructed endpoint views.
- `../prototype/nopo4d_full/quality_comparison.jpg`: real inputs and failure examples.
- `../prototype/nopo4d_full/quality_gate.json`: explicitly rejects this result for MAMMA.
- `../prototype/nopo4d_full/chunks/`: original learned 4D Gaussian attributes and camera estimates.
- `../prototype/nopo4d_full/splats/frame_00000.ply` … `frame_00133.ply`: aligned preview sequence.
- `../head-movement/Assets/Scenes/NoPo4DPrototype.unity`: standalone Unity inspection scene.
  Open it and press Play; Space plays/pauses, arrows step, A/D orbit, W/S move.
  Its data is read from the adjacent prototype directory, not imported as Unity assets.
  Unity uses subsampled, DC-only, unsorted isotropic splat discs; this is not the full
  anisotropic, view-dependent NoPo4D renderer. The preview intentionally has no skeletons.

### Reproduce locally

Commands below run from the workspace root, one directory above this README:

```powershell
# Environment already created. It inherits mamma's torch but overrides NumPy locally.
.\.venv-nopo4d\Scripts\python.exe prototype/run_nopo4d.py --frames 0 1 2 3 --width 504 --height 280 --out prototype/nopo4d_quality
.\prototype\run_cuda.cmd prototype/render_saved.py prototype/nopo4d_quality
.\prototype\run_cuda.cmd prototype/run_nopo4d.py --cameras 1 --render --out prototype/nopo4d_mono
.\prototype\run_cuda.cmd prototype/render_saved.py prototype/nopo4d_mono
.\.venv-nopo4d\Scripts\python.exe prototype/reconstruct_clip.py --no-render
.\prototype\run_cuda.cmd prototype/render_clip.py prototype/nopo4d_full
.\.venv-nopo4d\Scripts\python.exe prototype/package_preview.py
.\.venv-nopo4d\Scripts\python.exe prototype/test_contract.py
unity run ./head-movement --timeout 300 -- -executeMethod NoPo4DPrototypeScene.Verify
```

For recreation, create `.venv-nopo4d` with mamma's Python and `--system-site-packages`,
install `prototype/requirements-local.txt`, then install `external/NoPo4D` editable
with `--no-deps`. NoPo4D revision: `cb54c9349792d474aa541274842e0fadf1d807c7` (clone
recursively). Apply `prototype/nopo4d-turing.patch` in that clone and run
`prototype/apply_gsplat_windows_patch.py` in the venv. The Turing patch selects FP32
instead of unsupported BF16; the gsplat patch replaces a GCC-only flag with MSVC flags.
`run_cuda.cmd` selects the installed MSVC/CUDA 12.2 toolchain; its compiler compatibility
override is specific to this machine. The pretrained checkpoint is saved under
`external/checkpoints/nopo4d` from `bralani01/nopo4d`; the DA3 backbone is cached by
Hugging Face. Original conda packages and input videos were not changed.

Next quality milestone: establish dependable source-camera calibration and foreground
tracks, then evaluate scene-specific refinement against the real images. Only after
the duplicate-body and temporal-drift failures are resolved should the provider's
views enter MAMMA. A height-based automatic role assignment in the older bridge is a
heuristic, not evidence of leader/follower identity.

Phone dance video → **World Labs Atlas** (virtual camera ring) → **MAMMA** (multi-view SMPL-X) → **beat_this** (zouk time) → **head-movement** (Unity/VR).

```
phone videos ──sync──▶ synced CFR clips ──Atlas──▶ 12 locked-off virtual views ──MAMMA──▶ SMPL-X per dancer
   (+ music)            + audio.wav         │         (known K, R, t)                        │
                                            └──▶ environment.ply (static splat)              ▼
                                                                         StreamingAssets/<Capture>/ ◀── beat_this
```

Why the virtual ring: MAMMA needs calibrated, frame-locked multi-view video. Handheld phones give neither.
Atlas re-renders the performance from cameras we choose, so the calibration is exact by construction
and every view shares one timeline.

## Commands

Run from this folder with any Python ≥3.11 that has numpy + scipy + pyyaml, plus ffmpeg on PATH.
MAMMA, beat_this and SMPL-X forward kinematics each run as subprocesses in their own conda envs.

```bash
python -m atlas_bridge init sessions/spin --videos path/a.mp4 path/b.mp4 [--music track.wav]
python -m atlas_bridge sync sessions/spin           # audio GCC-PHAT offsets, CFR re-encode, audio.wav
python -m atlas_bridge rig sessions/spin            # virtual ring + coverage check
python -m atlas_bridge atlas-request sessions/spin  # writes work/atlas/request/REQUEST.md
python -m atlas_bridge align sessions/spin          # after environment.ply (+source_cameras.json) arrive
python -m atlas_bridge atlas-request sessions/spin  # again: camera poses now in Atlas' scene frame
python -m atlas_bridge atlas-fetch sessions/spin    # validates renders (fps, frames, resolution)
python -m atlas_bridge mamma sessions/spin          # writes mamma/data/atlas/<session>* and runs MAMMA
python -m atlas_bridge check-scale sessions/spin    # dancer heights → Atlas metric scale correction
python -m atlas_bridge beats sessions/spin          # beat_this zouk time (or [beats] bpm fallback)
python -m atlas_bridge export sessions/spin         # → head-movement/Assets/StreamingAssets/<capture_name>
python -m atlas_bridge status sessions/spin
python -m atlas_bridge demo                         # synthetic capture 00_SyntheticDemo, no Atlas/MAMMA needed
```

Tests: `python -m unittest discover -s tests` (includes a contract test against MAMMA's own calibration loader).

## Atlas access day checklist

The Atlas API and file formats are not public yet (2026-09). Everything Atlas-specific sits behind
`atlas/base.py` (`AtlasJob` → `AtlasResult`). Provider `manual` is a file drop and works with the web UI.

1. **Phase 1, reconstruction:** upload `work/atlas/request/inputs/*.mp4`, export the static splat as
   `work/atlas/renders/environment.ply`, plus Atlas' poses for the input videos as `source_cameras.json`
   (`{name: 4x4 c2w, OpenCV axes}`) if it exposes them. Then run `align`.
2. **Phase 2, views:** render every camera in `virtual_cameras.json` for the full clip. The file has both
   `c2w_opencv` and `c2w_opengl`, so use whichever convention Atlas takes. Drop `v00.mp4 … v11.mp4` into `renders/`.
3. **Verify assumptions, then encode them** (each is a one-line change):
   - output frame *i* == input frame *i* (else set a frame offset in `ManualAtlasProvider.fetch`)
   - Atlas units vs metres: run `check-scale` with `[roles] lead_height/follow_height`, update `[atlas] scale`
   - rendered resolution == requested (intrinsics depend on it; `atlas-fetch` checks)
   - scene handedness (if poses come out mirrored, flip one axis in `align.scene_to_stage`)
4. Once there's an API, implement `atlas/http.py` (`WORLDLABS_API_KEY` env var) and switch
   `[atlas] provider = "http"`.

**Risk to watch:** Atlas views are *generated*. Occluded limbs in the novel views can be plausible but
wrong, and MAMMA will happily triangulate them. Keep the real phone view(s) in the loop: compare MAMMA's
reprojection against the real footage before trusting a take, and prefer the low ring (closest to the
real phone height) when views disagree.

## Coordinate frames

| frame | axes | used by |
|---|---|---|
| stage | right-handed, Z-up, metres, floor z=0, origin at the dancers | rig, MAMMA calibration + output |
| OpenCV camera | +X right, +Y down, +Z forward | MAMMA YAML (`cam2world`, wxyz quaternions) |
| Atlas scene | whatever Atlas reconstructs; `scene_to_stage.json` maps it | splats, render requests |
| Unity | left-handed, Y-up; `unity = (x, z, y)` of stage | figure*.json, cameras.json, splats at load |

Joints: MAMMA SMPL-X joints 0–21 map 1:1 to SMPL. SMPL `L_Hand`/`R_Hand` come from `left/right_middle1`
(SMPL-X 28/43). Full SMPL-X params are kept in `smplx_params.npz` for future mesh/physics work.

## Known environment issue

In the `mamma` conda env on this machine, **any numpy BLAS/LAPACK call crashes** (`0xC06D007F`:
`libblas 3.11 *_mkl` paired with `mkl 2026.1.0`). MAMMA itself will hit this. Likely fix:
`conda install -n mamma "mkl<2026"` (or switch to OpenBLAS: `conda install -n mamma "libblas=*=*openblas"`).
