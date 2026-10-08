# Classifying our own zouk moves: research synthesis

2026-10-06. Research only: nothing here was installed, trained or run on a pod.

This document answers the user's question: are there open-source libraries, from NVIDIA or anyone else,
that we can use to classify our own Brazilian zouk moves? It also fixes the vocabulary, the counterbalance
criteria and the staged plan that the build stages implement.

Companion files:
- `dancecap/taxonomy/vocabulary_research.json`: machine-readable vocabulary rows, merged by `python -m dancecap.moves graph`.
- `head-movement/docs/VIEWER_SPEC.md` v3, sections 3.5, 3.7, 3.9, 3.10, 3.11, 10 and 11: what the viewer shows.
- `dancecap/docs/MOVES.md` (written by the build stages): the labelling workflow and commands.

It is a synthesis of five research reports, covering:
1. the NVIDIA motion and robotics stack;
2. skeleton action recognition and temporal segmentation;
3. dance-specific prior work;
4. narration labelling and speech-to-text on Windows;
5. zouk vocabulary and counterbalance biomechanics.

Claims marked *unverified* rest on search snippets or secondary sources. Section 12 records an adversarial
fact-check against primary sources (2026-10-06); its corrections are already applied inline.

---

## 1. The short answer

**No library classifies zouk moves, and none classifies partner-dance moves in general.** That is true for
NVIDIA and everyone else. The closest existing work is **CoMPAS3D**:
- improvised salsa duets stored as SMPL-X, the same body model as our fits;
- about 2,800 expert-labelled 8-beat segments;
- an 11-class move benchmark (plus "Other") on which the best model reaches 69.9 % accuracy / 53.6 F1
  (a fine-tuned vision-language model, InternVL3, on rendered follower-only videos; not a skeleton
  classifier).

CoMPAS3D is a dataset and a benchmark, not a classifier we can reuse. Its salsa vocabulary does not carry
over to zouk. Its structure does, and we copy it:
- one non-overlapping move track for the couple;
- segments on the beat;
- labels made in ELAN;
- a kappa check on double-annotated data;
- at least about 20 examples per move before a move enters the classifier.

**NVIDIA has no dance-move or two-person classifier.**
- GR00T N1.7 is a robot-manipulation vision-language-action model and does nothing for dance recognition.
- HOVER, SONIC/MotionBricks, Cosmos 3, ActionRecognitionNet and BodyPose3DNet do not help either.

Four NVIDIA pieces are worth knowing:

| NVIDIA piece | What it gives us | When |
|---|---|---|
| **Parakeet-TDT-0.6B-v3** (speech-to-text, CC-BY-4.0) | English and Portuguese transcription with word timestamps, phrase boosting | Good model, but NeMo is Linux-only, so locally we use faster-whisper (section 3.2). Parakeet is the Runpod or CPU (sherpa-onnx) alternative. |
| **TMR-SOMA-RP-v1** (Kimodo release; NVIDIA Open Model License) | 256-d motion-text embedding trained on 700 h of mocap including dance | Stage C feature extractor. Needs our SMPL-X converted to the SOMA skeleton (via SOMA-X). Single-person. Turing GPU support unverified. |
| **SEED-Timeline-Annotations** (CC-BY-4.0) | A labelling precedent: timed text events, a "human vs propagated" flag, labels spread to similar motions by DTW | Now: the design of our provenance field and of label propagation |
| **TAO PoseClassificationNet** | A plain ST-GCN skeleton classifier (6 generic pretrained classes) | Not needed: the same model in plain PyTorch runs on Windows. TAO needs Linux and Docker. |

Two further NVIDIA options are physics only, not classification:
- ProtoMotions / MaskedMimic (physics trackers) could check on Runpod whether the follower alone is
  physically plausible during counterbalance. If she is not, that is evidence she leans on the leader.
- Newton is a simulator that runs on Windows.

**So we build our own classifier from our own labels.** The libraries that matter are generic:
- change-point detection (ruptures, ClaSPy);
- small-data time-series classifiers (aeon MiniRocket);
- temporal-action-segmentation heads (MS-TCN++);
- skeleton encoders (TMR, MotionBERT, SkateFormer, USDRL);
- local speech-to-text (faster-whisper).

The bottleneck is labelled data, not models. A 3-minute zouk dance gives roughly 30-60 couple segments.
Expect:
- an *assistive* first classifier (pre-fill and rank suggestions) after about 10 narrated dances;
- a classifier worth reporting after about 25-30 dances;
- pretrained-encoder fine-tuning to pay off at about 60-100 dances.

The plan has three stages:
- **Stage A (now): no training.** Narration labelling infrastructure, interpretable motion features,
  unlabelled boundary proposals and counterbalance detection.
- **Stage B:** a first classifier once enough dances are labelled.
- **Stage C:** pretrained motion encoders, fine-tuned on our data.

Section 8 gives the details; section 9 covers licences and Windows.

---

## 2. Decisions for the build stages

These are the decisions the other stages of this workflow need. IDs are for cross-reference.

### Graph (graph stage)
- **G1. Corredor = Lateral.** Two sources equate them (a salsaforums post: "what our teacher call lateral is
  called corredor"; Helen Avelar's tutorial titled "LATERAL/CORREDOR"), and the user decided on 2026-10-06
  that "Lateral is renamed Corredor". (goandance's "El Corredor" and Bath Zouk's Lateral page do not name
  the other term, so they are not independent confirmations.)
  - Keep the slug `lateral`, so labels stay stable.
  - Show "Corredor" as its display name and add `corredor` as an alias.
  - Delete the separate proposed move `corredor` from `taxonomy/additions.toml`. Until then the alias
    collides and `graph` warns three times. The dry run below confirmed this is the only conflict.
- **G2. Portuguese display names.**
  - Add the extra keys `display_name` and `display_confidence` to moves and nodes, taken from
    `vocabulary_research.json`.
  - The viewer shows `display_name` when the confidence is `confirmed` or `likely`. For `translation` it
    shows the user's own Zouk1 name.
  - `name` and `slug` keep their current meaning, so the contract does not change.
- **G3. The user's `additions.toml` must win over `vocabulary_research.json`.**
  - Today `build_graph` creates proposed moves from the research *before* it applies the additions aliases,
    so the user cannot re-point an alias (for example `chicote` onto `howl-up`).
  - Fix: apply additions aliases first, and support `ignore_research = ["<slug>", ...]` in `additions.toml`.
- **G4. Unvisited proposed nodes stay out of the way.** The research adds 8 new proposed nodes (pião, chicote,
  bate cabelo, balão, frango assado, boneca, toalha, balanço) and only aliases to the existing contrapeso proposal.
  - Draw proposed nodes that have no labelled time dim, or hide them behind a toggle.
  - The fingerprint ignores them until they are visited.
- **G5. Hub states.** Mark the grounded hubs (Stand, Hug, Side Lunge, Forward/Back Basic) with an extra key
  `hub: true`. The fingerprint can then report "time in the base" separately from "time in moves", as in
  Snap-Together Motion's hub nodes.

### Labels (labels / annotate / proposals stages)
- **L1. Couple tier only in v1** (`TIERS = ("couple",)`).
  - Head movements are couple moves, as in Zouk1, where they are nodes and are led.
  - Per-dancer styling tiers (`follower`, `leader`, overlapping, as in CoMPAS3D) are a later contract
    extension.
- **L2. Unidentified automatic boundaries are not labels.**
  - Change-point proposals go to a new file, `work/<take>/moves/proposals.json`, owned by the proposals
    module. It holds candidate boundaries with score, energy band, cluster id and repeat-of.
  - `labels.json` gets `provenance:"auto"` segments only when a move identity is guessed: counterbalance
    intervals become `counterbalance-pivot` now, and classifier output later.
  - Unlabelled time stays a gap. The fingerprint already reports `labelled_seconds`.
- **L3. Narration segments keep their anchor.** Every narrated label is a *timestamp label*: a moment
  inside the move, not its boundaries (Li, Abu Farha and Gall, CVPR 2021). Store these extra per-segment
  keys:
  - `anchor_t`: the word time converted to the label time base;
  - `anchor_word`;
  - `lag_s`;
  - `b0` and `b1`: eighth-note indices from `beats.json`;
  - `asr: {engine, model, word_prob}`.

  The raw words go in `transcript`. A later classifier can then train directly from the timestamps.
- **L4. Time base.** Store labels in `reference` time.
  - The annotation tool plays one source video. HTML5 `currentTime` / `requestVideoFrameCallback`
    `mediaTime` are presentation-time seconds, the same clock as `Source.to_reference`, whose affine fit
    came from audio.
  - Check that each source's video stream `start_time` is 0, or subtract it.
  - Never use frame index divided by fps: the phone videos are variable frame rate (cam 04: 29.75 vs 29.96
    fps; cam 06: 29.83 vs 29.95), so frame-based times drift by up to about 0.4 s per minute.
- **L5. Alias learning.** When the user confirms that an ASR string means a move (for example "veraginha"
  means viradinha), append it to `additions.toml` `[aliases]`. This is an additive edit to the user's file;
  show it in the UI first.
- **L6. Interchange.** labels.json maps one-to-one onto BABEL frame labels:
  - `transcript` = raw_label
  - `move` = proc_label
  - family/band = act_cat
  - t0/t1 = start_t/end_t
  - `annotator` = anntr_id

  Add `.eaf` export and import (ELAN, via `pympi-ling`) with tiers `couple-move` (controlled vocabulary =
  move slugs), `narration-words`, `auto-proposals` and `beats`, so the user can review four camera angles
  at once in ELAN.

### Speech-to-text (annotate stage)
- **A1. Engine.**
  - *Status 2026-10-07 (implemented, measured on the user's two narrations: work/larissa_kadu/moves/narration/<id>/
    transcript.json vs the kept transcript.sapi.json):* `.venv-asr` (Python 3.11, faster-whisper 1.2.1, ctranslate2 4.8.2, nvidia-cublas-cu12
    12.9.2.10; no cuDNN needed) with **large-v3** as the default (turbo is the faster option), two passes
    (`language="en"` and `"pt"`) fused per utterance by vocabulary match x word probability, Silero VAD regions
    joined with 1 s silences (faster-whisper's own `vad_filter` moved words across the joins), hotwords =
    Portuguese name words first. `auto` never falls back to SAPI. Code: `dancecap/moves/annotate/asr.py`.
    Review fixes (same day): turbo is clearly weaker on Portuguese names (24 IPA-synthesised Brazilian names:
    large-v3 24/24, turbo 17/24 - Sahada, Shikachi, Solchinho, Piel, Torcel, Avial), so a busy GPU now QUEUES a
    narration instead of silently running CPU turbo (English pass only); 'Transcribe on CPU now' or
    `DANCECAP_ASR_GPU_BUSY=cpu` skips the queue, and such results are marked 're-transcribe recommended'.
    Re-transcription keeps every earlier transcript (`transcript.<engine-model-mode>.json`, timestamped when the
    name is taken). A word split by a hyphen ("Yo -Yo" = Ioiô) is no longer merged as a decoder repetition, and
    short Portuguese names match their Brazilian respelling ("Piau" -> Pião).
  - Model: **faster-whisper 1.2.x** with `large-v3-turbo`, run with `device="cuda"`,
    `compute_type="float16"`. Never bf16: Turing cannot run it, and CTranslate2 would fall back to fp32.
  - Settings: `language="en"`, `word_timestamps=True`, `vad_filter=True` (min_silence_duration_ms 300-500),
    `beam_size=5`, `condition_on_previous_text=False`.
  - Vocabulary: pass the graph's display names and aliases as `hotwords`. faster-whisper prepends hotwords to
    *every* 30 s window and truncates them at 223 tokens. Do not rely on `initial_prompt`: with
    `condition_on_previous_text=False` it conditions only the first window (verified in
    `faster_whisper/transcribe.py`, `prompt_reset_since`).
  - Expected cost: a few GB of VRAM and seconds per narration.
  - Wrap it in `dancecap.gpu.exclusive("asr")`. After one download, run with `local_files_only=True`, so it
    is fully offline.
- **A2. Isolation.** Run ASR in a subprocess from a dedicated `.venv-asr` (Python 3.11, no torch) that holds
  `faster-whisper` and `nvidia-cublas-cu12`. Add `nvidia-cudnn-cu12==9.*` only if the smoke test fails.
  - How the Windows wheel finds CUDA (fact-checked from CTranslate2 source, 2026-10-06): the ctranslate2
    4.8.2 `cp311-win_amd64` wheel is built with `WITH_CUDNN=OFF` and `CUDA_DYNAMIC_LOADING=ON` against CUDA
    12.8.1 (`python/tools/prepare_build_environment_windows.sh`). At first GPU use, `src/cuda/cublas_stub.cc`
    calls `SetDllDirectoryA(%CUDA_PATH%\bin)` and then `LoadLibraryA("cublas64_12.dll")`. A
    `SetDllDirectory` folder is searched *before* `PATH`.
  - On this machine `CUDA_PATH` points to CUDA v12.2 and `PATH` also holds its `cublas64_12.dll`. Prepending
    to `PATH` alone would therefore still load the old 12.2 cuBLAS (and mix in its `cublasLt64_12.dll`).
  - So in the subprocess environment set `CUDA_PATH=<.venv-asr>/Lib/site-packages/nvidia/cublas` (its `bin`
    holds `cublas64_12.dll` and `cublasLt64_12.dll`) **and** prepend that `bin` to `PATH`.
  - cuDNN is probably not needed on Windows (built `WITH_CUDNN=OFF`; the wheel ships only the 266 KB
    `cudnn64_9.dll` loader). This is *unverified* until the smoke test passes without it.
  - faster-whisper's README documents the pip-wheel route for Linux and points Windows users to Purfview's
    whisper-standalone-win archive instead. That archive is the fallback if the pip route fails.
  - Smoke test: `ctranslate2.get_cuda_device_count() > 0`, the model loads on cuda, and a 10 s clip
    transcribes. This catches a silent fall-back to CPU. Also check the cuBLAS version that `cublas_stub.cc`
    logs at info level (`ctranslate2.set_log_level(logging.INFO)`): it must be 12.9, not 12.2.
  - Do not install into the shared `.venv-nopo4d`: the other workflow uses it.
- **A3. Fall-backs, in order:**
  1. faster-whisper on CPU (`small.en` or `medium.en`, int8);
  2. the `transformers` ASR pipeline with `return_timestamps="word"` on the existing torch 2.5.1;
  3. the whisper.cpp prebuilt Windows CUDA zip as a subprocess (`-ml 1`, `-oj`, `--vad`). The asset name
     changes between releases: currently `whisper-bin-win-cuda-12.4.0-x64.zip` (about 684 MB), earlier
     `whisper-cublas-12.4.0-bin-x64.zip`. `--vad` also needs a separate Silero VAD model file (`-vm`).

  Not used:
  - NeMo / Parakeet locally: no official Windows support, and WSL2 is ruled out.
  - Vosk: grammar words missing from its lexicon are dropped with only a log warning ("Ignoring word
    missing in vocabulary"), so Portuguese move names are lost.
  - WhisperX: it pins torch 2.8 and numpy 2.
- **A4. Graceful degradation.** The annotation tool must work without ASR installed: typed labels, a
  space-bar tap at transitions, and a "transcript JSON" import in a common word-timestamp format
  (`[{word, start, end, prob}]`) that any engine can produce.
- **A5. Approval and privacy.**
  - Downloads need the user's OK:
    - about 0.55 GB of CUDA wheels: today's `nvidia-cublas-cu12` 12.9.2.10 `win_amd64` wheel is 553 MB, and it
      is larger once unpacked. If cuDNN turns out to be needed, add 743 MB for `nvidia-cudnn-cu12` 9.27.0.42;
    - 1.62 GB of model: `mobiuslabsgmbh/faster-whisper-large-v3-turbo` `model.bin`, which faster-whisper's
      `large-v3-turbo` alias downloads.
  - Audio never leaves the machine.
  - The optional TypeSafe disambiguation sends only transcript text and candidate move names. It sits
    behind a flag and is off by default.

### Features and proposals (features / proposals stages)
- **F1. Inputs.**
  - Read `work/<take>/smplx/` (the Phase 2b refit) when it exists, else `smplx_v1/`. With `smplx_v1/`, warn
    that the world frame is about 4.8° off gravity, which corrupts lean features by about 0.08 m at hip
    height.
  - Reuse `physics.analyze_dancer` (COM and mass), `physics.partner_connections` and the `timing.py`
    touchdowns.
- **F2. Pooling.** Pool per eighth note from `beats.json` (8 eighths per measure; type 1 at eighths 0 and 4).
  Keep mean, min, max and delta, and robust-z-score them per take.
- **F3. Change points.**
  - Use `ruptures` KernelCPD (rbf) or PELT if it is installed. Otherwise use a numpy-only Foote novelty
    curve (a checkerboard kernel on the self-similarity matrix), so Stage A needs no new packages.
  - Tune the penalty on `work/synthetic` `truth.json`.
  - ClaSPy is an optional second opinion, run only from its own venv. claspy 0.2.8 pins numba<0.62,
    scipy<1.16 and scikit-learn<1.8, but `.venv-nopo4d` has numba 0.68.0, scipy 1.17.1 and scikit-learn
    1.9.0. `ruptures` and `stumpy` do fit `.venv-nopo4d`.
- **F4. Snapping.** Snap boundaries to type-1 eighths, with a minimum segment of 4 eighths.
- **F5. Energy band.** Propose an energy band from couple kinetic energy, ranked within the take until
  labels calibrate it.
- **F6. Clusters.** Cluster segment means into recurring states (GMM or k-means; CLaP or an HMM later) and
  flag repeats with STUMPY motifs or plain DTW.
- **F7. Compute.** CPU only. No training. No pods.

### Counterbalance (counterbalance stage)
The criteria are in section 6.
- **C1.** Detect two kinds of interval. Write the kind as an extra key: `kind:"static"|"pivot"`.
  - `static`: lean-away plus connection. This is enough for the yellow axis.
  - `pivot`: static plus an anchored follower foot plus the leader circling. This drives the pivot ring and
    the leader arc.
  - For `static` intervals, `pivot` and `pivot_foot` are null.
- **C2.** `direction` is the *physical* rotation sense seen from above. Compute it in the world frame. Do
  not flip it when positions are converted to Unity: the conversion flips z but preserves the physical
  scene. The viewer draws the arc from the converted positions, never from the sign of the label.

---

## 3. Library comparison

The project is non-commercial research. "NC" means non-commercial licence, which is fine for us. Nothing
under NC, AGPL or a gated licence may enter the public MIT Unity repo: no weights, no data, no derived
embeddings, no code. "Win" means it runs natively on Windows 10 on this machine (RTX 2080 Ti 11 GB, Turing:
fp16 yes, bf16 no).

### 3.1 NVIDIA human-motion and robotics stack

| Name / link | Licence | Inputs | What we would use it for | Win | Verdict |
|---|---|---|---|---|---|
| [Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) + NeMo GPU-PB boosting; [sherpa-onnx int8 port](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html) | CC-BY-4.0 (model), Apache-2.0 (NeMo, sherpa-onnx) | wav, 25 languages incl. en/pt (European Portuguese) | Narration transcript with word timestamps; boost move names | NeMo no (Linux / WSL2 only); sherpa-onnx yes on CPU, boosting for NeMo transducers unconfirmed | **Alternative**: Runpod or CPU fallback; faster-whisper is primary locally |
| [Canary-1B-v2](https://huggingface.co/nvidia/canary-1b-v2) | CC-BY-4.0 | wav | Same as Parakeet | No | Fallback only |
| [TMR-SOMA-RP-v1](https://huggingface.co/nvidia/TMR-SOMA-RP-v1) ([Kimodo](https://github.com/nv-tlabs/kimodo)) | NVIDIA Open Model License (weights, per model card; not re-checked); Apache-2.0 code | SOMA 30-joint xyz, ≤10 s at 30 fps; text | Per-dancer 256-d window embeddings for kNN/prototypes; zero-shot text queries ("man walks in a circle around partner") as weak proposals | Model card lists Windows; Turing unverified | **Stage C experiment**; blocked on the SMPL-X to SOMA conversion |
| [SOMA-X](https://github.com/NVlabs/SOMA-X) | Apache-2.0 (SMPL-X files under their own licence) | SMPL-X params plus our licensed model files | Bridge SMPL-X to SOMA for TMR-SOMA (pose inversion) | OS not stated | Only if TMR-SOMA is tried; API unverified |
| [SEED-Timeline-Annotations](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations) | CC-BY-4.0 | n/a (dataset) | Precedent: events `{start_time, end_time, description}`, a `propagated_from_filename` field (null = human-labelled), DTW propagation of a few human labels | n/a | **Adopt the design now** (provenance, propagation) |
| [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | Custom, gated; reportedly free for academic and startups under $1M revenue (eligibility to check) | n/a | Solo-dance pretraining data or "non-zouk" negatives (11k dance clips, SOMA) | n/a | Stage C maybe; solo only; check eligibility |
| [Kimodo](https://github.com/nv-tlabs/kimodo) generator | Apache-2.0 code; Kimodo-SMPLX weights under NVIDIA's research-only R&D licence | Text, constraints | Not a classifier; only hosts TMR-SOMA | Linux ("Windows should work") | No |
| [TAO PoseClassificationNet](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/pose_classification/pose_classification.html) (ST-GCN) | NVIDIA TAO terms | (N, 3, T≤300, V fixed layout, M persons) | Two-person clip classifier with M=2 (default layout: NVIDIA 34 joints) | No (Ubuntu + Docker + NGC; 32 GB recommended) | **No**: write a plain PyTorch ST-GCN instead (Stage B/C) |
| [ProtoMotions3](https://github.com/NVlabs/ProtoMotions) (MaskedMimic, ADD, AMP/ASE, FSQ tracker) | Apache-2.0 | SMPL/SMPL-X motion (AMASS-style conversion) | Physics check: track the follower alone; failure inside counterbalance intervals is independent evidence of partner-supported lean | Linux-documented. It now also has Newton and MuJoCo backends, both cross-platform, so native Windows may be possible (*unverified*) | **Optional Runpod experiment** (or local via Newton, if that works), untested idea; single-character |
| [MaskedMimic](https://research.nvidia.com/labs/par/project/maskedmimic.html) | in ProtoMotions; checkpoint licence unchecked | Partial pose constraints | Repair implausible frames | No | No |
| [GEAR-SONIC / GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | Apache-2.0; NVIDIA Open Model License | SMPL joints at 50 Hz | Token histograms as features (experiment) | No (C++/TensorRT, Isaac Lab) | No: tokens are about 0.2 s of robot control intent, not move semantics |
| [GEM / GENMO](https://github.com/NVlabs/GENMO), [GEM-X](https://huggingface.co/nvidia/GEM-X) | NVIDIA OneWay Noncommercial (GEM); Open Model License (GEM-X) | Monocular video | Cross-check our MAMMA fits (capture side, not classification) | No (Linux; GEM-X tested on A100) | No (for classification) |
| [Cosmos Reason 2](https://huggingface.co/nvidia/Cosmos-Reason2-8B) | NVIDIA Open Model License | mp4 at 4 fps + prompt | Ask "when does she turn / dip / throw her head" for weak proposals | No (model card: BF16 only tested, minimum 32 GB, Linux, Hopper/Blackwell) | Optional Runpod experiment only; zouk names unknown to it |
| [Cosmos-Embed1](https://huggingface.co/nvidia/Cosmos-Embed1-336p) | NVIDIA Open Model License | 8 frames at 1-2 fps + text | Appearance-based second feature stream | FP32 on Turing maybe (unverified) | No: too coarse for head movement; confounded by camera and room |
| [GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) | Apache-2.0; Open Model License | Images + language | Nothing for dance | No | **No** (the user's hope that GR00T helps does not hold) |
| [HOVER](https://github.com/NVlabs/HOVER), [PULSE](https://github.com/ZhengyiLuo/PULSE), [PHC](https://github.com/ZhengyiLuo/PHC), [MimicKit](https://github.com/xbpeng/MimicKit)/[CALM](https://github.com/NVlabs/CALM) | Apache-2.0 / none (PULSE) / BSD-3-Clear / Apache-2.0 / NOASSERTION (CALM) | Robot or SMPL motion in simulation | Controllers and skill latents | No | No (avoid PULSE and CALM: licence unclear) |
| [Newton](https://github.com/newton-physics/newton) | Apache-2.0 | URDF/MJCF | A later custom two-body hand-tension model for counterbalance | **Yes** (Maxwell+ GPU) | Later, optional; the analytic COM checks in `physics.py` come first |
| [TAO ActionRecognitionNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet), [BodyPose3DNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/bodypose3dnet) | NVIDIA terms | RGB video | Video classifier / single-person 3D pose | No | No: our multi-view SMPL-X fits are better input |

### 3.2 Speech-to-text for narration

| Name / link | Licence | Inputs | Use for us | Win | Verdict |
|---|---|---|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.1 + [CTranslate2](https://pypi.org/project/ctranslate2/) 4.8.2 | MIT | Any audio (PyAV) | Word timestamps (start, end, probability), `hotwords`, `initial_prompt`, Silero VAD; large-v3-turbo fp16 | **Yes**: separate venv with the `nvidia-cublas-cu12` wheel, plus `CUDA_PATH` and `PATH` pointing at it (A2) | **Primary** |
| [transformers ASR pipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines) | Apache-2.0 / MIT weights | Audio | `return_timestamps="word"` (cross-attention DTW) on the existing torch 2.5.1+cu124 | Yes (install needs a dry run first) | Fallback 2 |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | MIT | 16 kHz wav | Standalone exe with CUDA; `-ml 1`, `--vad` (needs a separate VAD model, `-vm`), `-oj` JSON output | Yes (prebuilt `whisper-bin-win-cuda-12.4.0-x64.zip`, about 684 MB; the asset name changes between releases; download needs approval) | Fallback 3 |
| [openai-whisper](https://github.com/openai/whisper) | MIT | Audio | Reference implementation | Yes, but its DTW Triton kernels fall back to a slow CPU path | Emergency only |
| [WhisperX](https://github.com/m-bain/whisperX) | BSD-2 | Audio | wav2vec2 forced alignment (tens of ms) | Needs torch 2.8 and numpy 2 in its own env | No: precision is not the bottleneck (narration lag is 0.3-2 s) |
| [Vosk](https://alphacephei.com/vosk/adaptation) | Apache-2.0 | 16 kHz PCM | Grammar-restricted decoding | Yes (CPU) | No: grammar words missing from its lexicon are dropped; pt WER 54-69 % |
| [torchaudio forced_align / MMS_FA](https://docs.pytorch.org/audio/2.5.0/tutorials/ctc_forced_alignment_api_tutorial.html) | BSD-2 | Waveform + transcript | Refine word onsets if ever needed | Yes (already installed) | Spare |
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | Audio | Skip silences: sparse narration otherwise invites hallucination | Yes (inside faster-whisper) | Use (via `vad_filter`) |

### 3.3 Proposals without training (Stage A)

| Name / link | Licence | Inputs | Use for us | Win | Verdict |
|---|---|---|---|---|---|
| [ruptures](https://github.com/deepcharles/ruptures) | BSD-2 | [T, F] array | PELT / KernelCPD change points on per-eighth couple features | Yes (CPU) | **Use** (optional install; numpy fallback F3) |
| [ClaSPy](https://github.com/ermshaua/claspy) (ClaSP + CLaP) | BSD-3 | Multivariate series | Parameter-free second opinion; CLaP gives recurring states plus transitions, an unsupervised analogue of the Zouk1 state machine | Yes, but **not in `.venv-nopo4d`**: claspy 0.2.8 pins numba<0.62, scipy<1.16, scikit-learn<1.8 and depends on aeon | Optional, own venv |
| [aeon](https://www.aeon-toolkit.org/en/stable/api_reference/segmentation.html) | BSD-3 | Equal-length series collections | HMMSegmenter / ClaSP / BinSeg now; MiniRocket classifier in Stage B | Yes, but **not in `.venv-nopo4d`**. Every aeon 1.x pins numba below 0.62-0.64 and caps scipy and scikit-learn below the venv's numba 0.68.0, scipy 1.17.1 and scikit-learn 1.9.0; aeon 1.6.0 also needs numpy ≥ 2, and the venv has 1.26.4 | **Use in Stage B**, from its own venv, or port MiniRocket (a small numpy/numba algorithm) |
| [STUMPY](https://github.com/stumpy-dev/stumpy) | BSD-3 | 1-D/N-D series | Motifs: "this looks like the segment you labelled at t = ..." | Yes (stumpy 1.14.1 needs numba ≥ 0.61.2; fits `.venv-nopo4d`) | Optional (label propagation inside a dance) |
| [keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq) + [dynamax](https://github.com/probml/dynamax) | Harvard academic licence; MIT | Keypoints | AR-HMM "syllables" plus transition graphs; its usage-and-transition-graph analysis is the closest published analogue of our fingerprint | JAX is CPU-only on Windows | Borrow the **analysis ideas** (difference graphs), not the model |
| [SMQ](https://github.com/bachlab/SMQ) (ICCV 2025) | **AGPL-3.0** | Skeleton .npy | Learned unsupervised skeleton segmentation (tested on BABEL) | Probably (PyTorch) | Later; run as a separate tool only, never in distributed code |

### 3.4 Classifiers and segmentation heads (Stage B)

| Name / link | Licence | Inputs | Use for us | Win | Verdict |
|---|---|---|---|---|---|
| aeon [MiniRocket](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.classification.convolution_based.MiniRocketClassifier.html) | BSD-3 | Fixed-length multivariate windows | First classifier on beat-resampled windows, both dancers as channels; trains in seconds | Yes (CPU), in its own venv (see the aeon row in 3.3) | **Stage B1** |
| kNN / nearest prototype (scikit-learn) | BSD-3 | Feature vectors | Few-shot baseline; ranks suggestions for the narrator | Yes | **Stage B1** |
| HMM / Viterbi with Zouk1 links as the transition prior | n/a (our code) | Per-segment class scores | Graph-constrained decoding; picks the node instance of duplicated names. A ballroom study saw 85.95 → 92.31 % from such a prior. | Yes | **Stage B1** (`path.py` already does Viterbi over instances) |
| [MS-TCN++](https://github.com/sj-li/MS-TCN2) / [ASFormer](https://github.com/ChinaYi/ASFormer) | MIT | Per-step feature sequences | Dense segmentation head over per-eighth features (a few hundred steps per dance), with a timestamp-supervision loss | Yes (port the model files; their environments are pinned to torch 1.x) | **Stage B2** |
| [Timestamp supervision](https://github.com/ZheLi2020/TimestampActionSeg) (CVPR 2021) | no licence (reimplement) | Features + one timestamp per segment | Train from narration anchors without exact boundaries | Yes | **Reimplement the loss** in Stage B2 |
| [DiffAct](https://github.com/Finspire13/DiffAct) / [FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT) | MIT | Per-frame features | Upgrades over MS-TCN++ | Yes | Stage C, only once MS-TCN++ works |
| [MS-GCN](https://github.com/BenjaminFiltjens/MS-GCN) / [LaSA](https://github.com/HaoyuJi/LaSA) / [DeST](https://github.com/lyhisme/DeST) | no licence file (repo archived) / MIT / assume CC BY-NC (README vs API conflict) | 3-D joints | Segmenters straight on joints; LaSA uses text embeddings of class names (our aliases) | Yes | Stage C candidates |
| [ActionFormer](https://github.com/happyharrycn/actionformer_release) / [OpenTAD](https://github.com/sming256/OpenTAD) | MIT / Apache-2.0 | Features | Sparse, overlapping detection | Tight on 11 GB | No: couple segments are dense and back-to-back |
| [PySKL](https://github.com/kennymckormick/pyskl) / [MMAction2](https://github.com/open-mmlab/mmaction2) | Apache-2.0 | Skeleton pickles, M=2 | Source of ST-GCN++ / CTR-GCN code and NTU weights | mmcv is painful on Windows; PySKL unmaintained | Copy architectures, do not depend on them |

### 3.5 Pretrained motion encoders and backbones (Stage C)

| Name / link | Licence | Inputs | Use for us | Win | Verdict |
|---|---|---|---|---|---|
| [TMR](https://github.com/Mathux/TMR) / [TMR++](https://github.com/leorebensabath/TMRPlusPlus) | MIT | HumanML3D 263-d (22 SMPL joints, 20 fps) | Frozen per-dancer embeddings plus zero-shot text features. SMPL-X body joints 0-21 are the SMPL-22 layout, so the conversion is direct. | Yes | **First encoder to try** (B1 extra features, C) |
| [MotionBERT](https://github.com/Walter0807/MotionBERT) | Apache-2.0 | H36M-17 2-D keypoints, ≤243 frames | Per-dancer embeddings (project our 3-D joints to virtual views); one-shot protocol | Yes | Stage C; never use its action head as-is: it **averages over persons** and loses lead/follow roles |
| [USDRL / FoundSkelModel](https://github.com/wengwanjiang/FoundSkelModel) | Apache-2.0 | NTU-25 3-D | Dense (per-frame) self-supervised encoder; reports segmentation results | Yes | Stage C candidate |
| [MAMP](https://github.com/maoyunyao/MAMP) | no licence (reimplement) | NTU-25 3-D | Masked-motion pretraining recipe on our unlabelled captures | Yes (short clips) | Stage C recipe |
| [SkateFormer](https://github.com/KAIST-VICLab/SkateFormer) | MIT (+ commercial note) | NTU-25, M=2 | Two-person backbone; NTU-interaction weights | Yes | **Stage C two-person backbone** |
| [2P-GCN](https://github.com/mgiant/2P-GCN) / [ISTA-Net](https://github.com/Necolizer/ISTA-Net) | MIT | Two NTU-25 skeletons | One graph over both bodies with hand/hip edges | Yes | Use 2P-GCN's idea with a **fixed role order**; disable ISTA-Net's permutation invariance |
| [ProtoGCN](https://github.com/firework8/ProtoGCN), [HD-GCN](https://github.com/Jho-Yonsei/HD-GCN) | MIT | NTU pickles | Single-clip recognisers | Yes (via PySKL code) | Alternatives |
| [MotionPatches](https://github.com/line/MotionPatches) | CC BY-NC 4.0 | HumanML3D/KIT joints | Image-pretrained ViT motion-text model; helps tiny datasets | Yes | Alternative to TMR (NC) |
| [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [MoMask](https://github.com/EricGuo5513/momask-codes), [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), [MotionLLM](https://github.com/IDEA-Research/MotionLLM) | MIT / MIT / MIT / IDEA | Single-person SMPL / HumanML3D | Captions, tokens | Partly | No: stale, weak or too heavy; generic vocabulary |
| [Action Motifs (A4Mer)](https://arxiv.org/abs/2604.28173) | no licence | SMPL | Atoms combining into motifs, conceptually like "isolations grow into moves" | ? | Watch, do not depend on it |
| [Bailando](https://github.com/lisiyao21/Bailando) / [DuetGen](https://github.com/snap-research/duet-gen) | S-Lab NC / Snap NC | Solo or duet SMPL(-X) | VQ "dance units"; DuetGen encodes the couple as one unit (relative root + both bodies) | Yes | Ideas for couple tokens; retraining needed |

### 3.6 Datasets (all research-only; none may enter the public repo)

| Name / link | Licence | Content | Use for us | Verdict |
|---|---|---|---|---|
| [CoMPAS3D](https://huggingface.co/datasets/Rosie-Lab/compas3d) ([paper](https://arxiv.org/abs/2507.19684)) | Conflicting (HF card CC-BY-NC-4.0, "except for audio in .mp4 files where rights retained by owners"; paper CC BY 4.0; project page CC BY-SA 4.0), so treat as NC. Not gated on Hugging Face (API check, 2026-10-06) | 3 h salsa duets, SMPL-X leader+follower, 8-beat segments, couple + per-dancer tiers, 30 move classes | Template for our schema; partner-dance pretraining (salsa to zouk); external test of our tooling | **Template now; data in Stage C** |
| [DD100 / Duolando](https://github.com/lisiyao21/Duolando) | NTU S-Lab 1.0 (NC) | 117 min of duets, 10 genres, SMPL-X, no move labels | Two-person self-supervised pretraining | Stage C |
| [MDD](https://gprerit96.github.io/mdd-page) | Not released | 10 h of duets incl. sensual bachata (closest to zouk), 10k segment descriptions | Precedent for free-text segment narration | Watch |
| [InterDance](https://inter-dance.github.io/), [ReMoCap](https://github.com/anindita127/ReMoS), [ExPI](https://arxiv.org/abs/2105.08825), [HDM12 tango](https://cg.cs.uni-bonn.de/en/publications/paper-details/voegele-2016-HDM12/) | Unreleased / NC / INRIA contract / unknown | Duets (ExPI: 16 named Lindy aerials; HDM12: 11 tango couples) | ExPI shows "one move per clip" capture; HDM12 is the most zouk-like (improvised lead-follow) | Low priority |
| [BABEL](https://babel.is.tue.mpg.de/data.html) | Non-commercial research (+ AMASS) | 43 h AMASS, dense frame labels | Label schema compatibility (L6); BABEL-trained tools can run on our data | **Schema compatibility now** |
| [Inter-X](https://github.com/liangxuy/Inter-X), [InterHuman](https://github.com/tr3e/InterGen) | Research form / CC BY-NC-SA | Two-person everyday interactions (actor-reactor) | Two-person pretraining | Stage C maybe |
| [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [Motion-X](https://github.com/IDEA-Research/Motion-X), [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html), [FineDance](https://github.com/li-ronghui/FineDance), [BRACE](https://github.com/dmoltisanti/brace) | Various NC / CC BY 4.0 annotations (AIST++) | Solo motion | HumanML3D: its feature format feeds TMR. AIST++: "basic moves at several tempos" is the right capture pattern. BRACE: a coarse family tier is learnable. | Patterns only |

### 3.7 Annotation tools

| Name / link | Licence | Use for us | Win | Verdict |
|---|---|---|---|---|
| **Custom local tool** (`dancecap.moves.annotate`, Flask + waitress, both already installed) | ours | Narration capture synced through pause, seek and slow motion; beat grid; graph vocabulary; reference clock | Yes | **Build** (Stage A) |
| [ELAN 7.1](https://archive.mpi.nl/tla/elan) + [pympi-ling](https://pypi.org/project/pympi-ling/) | GPL-3 tool / MIT library | Multi-tier, controlled vocabularies, up to 4 synced videos, a "delayed mode" that compensates keystroke lag; CoMPAS3D used it | Yes (bundled Java) | **Round-trip `.eaf`**, not the primary tool. EAF has offsets only: export each camera in its own time base. |
| [Label Studio](https://labelstud.io/tags/timelinelabels) TimelineLabels | Apache-2.0 | Frame-indexed ranges | Yes | No: frame indices drift on our variable-frame-rate videos; heavy; no narration |
| [CVAT](https://docs.cvat.ai/docs/administration/community/basics/installation/) | MIT | Frame tags only | Needs WSL2 | No |
| [VIA 3](https://www.robots.ox.ac.uk/~vgg/software/via/), [ANVIL](https://www.anvil-software.de/) | BSD-2 / research-only | Simple temporal segments / 2017-era tool | Yes | No (UI inspiration only) |

### 3.8 Helper libraries for phrase-to-move mapping

| Name / link | Licence | Use | Verdict |
|---|---|---|---|
| [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) | MIT | WRatio on 1-4-word n-grams of the transcript against names, aliases and learned ASR variants | **Use** (small install into `.venv-nopo4d`; needs approval) |
| [jellyfish](https://jamesturk.github.io/jellyfish/functions/) | MIT | Metaphone bonus, e.g. 'veraginha' ≈ 'viradinha' | Optional |
| TypeSafe (`dancecap/judgments.py`) | n/a | `Choice` over the top-k candidates with context (previous move, graph neighbours) for ambiguous phrases; refuse when the weakest judgment is < 0.4 | Optional, behind a flag; write it in a new module |

---

## 4. What prior dance research tells us (design rules)

| Evidence | Rule for us |
|---|---|
| CoMPAS3D (salsa, SMPL-X): one couple move track plus per-dancer styling tracks, 8-beat segments, ELAN, kappa 0.752 on 5 % double-annotated, ≥20 instances per class for the benchmark, about 120 h for one expert to label half of 3 h of dance (about 80× real time) | A non-overlapping couple tier (we have it); beat-aligned segments; double-annotate 5-10 % of beats later; a move enters the classifier at about 20 instances; narration plus proposals exists because manual annotation is too slow |
| Ballroom smartwatch study: a Markov chain of allowed figure transitions raised accuracy from 85.95 % to 92.31 % | Decode with the Zouk1 links as the transition prior; transitions missing from the graph become `new_link` proposals, not errors |
| Endo et al., MOCO '24: dance segment boundaries sit on beats or half-beats; scored at ±0.5 beat | Snap to the eighth grid; score boundaries at ±0.5 beat |
| Kovar 2002 motion graphs; Snap-Together Motion hub nodes; Kim et al. 2003 beat-based movement transition graph | A dance is a walk on the move graph; grounded states are hubs (G5); Zouk1 keeps moves on nodes and transitions on links; store a transition duration on links for the Unity camera |
| Li, Abu Farha and Gall 2021 (timestamp supervision); point-supervised skeleton segmentation (arXiv 2603.06201); EPIC-KITCHENS pause-and-talk narration | A spoken label is a timestamp inside the move; allow pause-and-talk; keep the anchors (L3) |
| Naming latency in psycholinguistics: 0.47-2 s, about 0.6-1.1 s typical; ELAN's delayed mode | Model a narration lag (default 0.8 s wall-clock, learned per annotator); at 0.5× playback the lag in video time halves |
| BABEL: free-text labels first, then mapped to categories by clustering | Keep the raw transcript; map through aliases; cluster unmatched phrases for the user to accept as alias or new move |
| FineGym: three-level hierarchy | Evaluate at band / family, move, and node level; per-move counts stay small for a long time |
| Carlson et al. 2020: individual dancers are easier to identify than genre | A classifier trained on one couple learns the couple: hold out whole takes now and whole couples later |
| keypoint-MoSeq: usage per state plus transition graphs, with difference graphs between groups | Fingerprint comparison: Jensen-Shannon divergence on node share and band share, plus a difference transition graph |
| Partner-dance generation (DuetGen, SalsaAgent): the couple as one unit, relational tokens | Couple features must include relative pose and connection, not just two bodies side by side |

---

## 5. Zouk vocabulary

### 5.1 Decisions
- **Corredor = Lateral** (G1). In the corredor the follower walks a line back and forth past the leader,
  turning with a compass step at each end, hips along the line and chest to the leader. That twist of the
  torso is a measurable signature (chest yaw minus pelvis yaw).
- **Display in Brazilian Portuguese where a term is attested** (G2); the user's English name stays as an
  alias. Where nothing is attested, the user's name stays.
- **Slugs never change.** They come from the Zouk1 names.
- **Counterbalance is two things.**
  - A *technique* (contrapeso) that can sit on many moves. It comes from `counterbalance.json` and the
    fingerprint's `counterbalance_share`.
  - A *move*: the Larissa/Kadu pivot, proposed as `counterbalance-pivot` and displayed "Contrapeso". In
    zouk terms it is most likely a **planada** danced in contrapeso.
    - AmoZouk defines the planada as "a single-footed turn where the leader rotates the follower on a
      stationary axis".
    - That the leader *walks around her* comes from a separate zouk description of one-foot spins
      (theperfectfollow.com), which does not use the word planada. Treat "planada = leader circles her" as
      *likely*, not confirmed.
  - On the word itself: English sources call the technique "counter-balance" (Wikipedia) or "counterbalance"
    (dZouk Zouk 3/4). The Portuguese "contrapesos" is attested in one Brazilian academic article on zouk
    (SciELO 2023, as a performance element). "Contrapeso" as the display name is therefore *likely*, not
    confirmed.

### 5.2 Existing Zouk1 moves with attested or likely Portuguese names

"Energy" is the graph's normalised height (0 = lowest node, 1 = highest).

| Display | Zouk1 name (slug) | Aliases added | Portuguese meaning | Zouk1 node(s) | Energy | Confidence |
|---|---|---|---|---|---|---|
| **Corredor** | Lateral (`lateral`) | corredor, el corredor, corridor | 'corridor'; side-to-side travelling basic | 8c3386a3, 1c1bdf22 | 0.27 | confirmed (2 sources + user decision) |
| **Viradinha** | Viradinha (`viradinha`) | veraginha, viradjinha | 'little turn': 180° backward pivot on the ball of the foot, entry into the corredor | af78b12c (Little Turn 63f848a0 possible duplicate, H2) | 0.19 | confirmed |
| **Soltinho** | Soltinho | – | 'a little loose': alternating turns orbiting a circle | d844c4c7 | 0.65 | confirmed |
| **Gostosinho** | Gostosinho | – | 'cosy': a hug at the end of a back-and-forth basic (Zouk1 has it after Back Basic) | 0c46f359 | 0.23 | confirmed |
| **Bônus** | Bonus | bônus, bonus turn | travelling turn, leader steps behind her; she turns about 1.5×; the left bonus ends in a *patinha* | 3b407072 | 0.31 | confirmed |
| Boomerang | Boomerang | bumerangue | no-hands relative of the bônus (AmoZouk); Zouk1's Boomerang → Bonus matches | fc67c615 | 0.32 | likely (Portuguese spelling unverified) |
| **Ioiô** | Yoyo | ioiô, yo-yo | travelling turn in towards the leader, then out | e286518b (isolated: add links) | 0.33 | likely |
| **Cambré** | Dip Back (`dip-back`) | cambré, backbend | controlled whole-spine backbend ("the dips") | 79a56109 (Fall Back possible) | 0.26 | likely |
| **Abraço** | Hug | abraço, close embrace | closed embrace; the grounded hub | e25c66b0 | 0.14 | likely |
| **Abertura** | Side Lunge | abertura, lunge | 'opening': preparatory side step into turns (Side Lunge is Zouk1's main hub into turns) | 3ee2c3ee | 0.31 | likely |
| Forward / Back Basic | Forward Basic, Back Basic | passo básico frente / trás | the slow-quick-quick basic, forward / back | 33fd1aaf, 09e3ca68 | 0.19 / 0.20 | translation |
| Simple Turn, Double Turn, Pirouette | – | giro, giro simples, giro duplo, pirueta | generic Portuguese turn words | 4cd989d6, 415e57f3, a573c6f1 | 0.34-0.92 | translation |
| Airplane | Airplane | avião | the dZouk curriculum names an avião; no description found | b7c93212 | 0.90 | translation |

### 5.3 Proposed new moves (status `proposed`, dashed until the user accepts)

| Display (slug) | Meaning | Family / tone | Energy (band) | Near | Source quality |
|---|---|---|---|---|---|
| **Contrapeso** (`counterbalance-pivot`, already proposed) | The Larissa/Kadu pivot: her foot is the axis, he circles at arm's length, both lean away; aliases contrapeso, planada | turn / yellow | already 0.50 in additions; research suggests 0.62 (band 3), e.g. [0.6, 0.55, -0.45] | Walk Around, Hug, Pirouette; links in from Lateral / Open Break / Hug | "contrapesos": SciELO 2023 only; "counterbalance" (English): Wikipedia, dZouk Zouk 3/4; planada: AmoZouk |
| **Pião** (`piao`) | Closed frame, both rotate around a central axis between them; advanced form: reverse pião | turn / yellow | 0.58 (band 2) | Hug, Hug Simple Turn, Double Turn | Jettence pt 2, dZouk, Oaktown |
| **Chicote** (`chicote`) | Hair or head whip back while looking up (Jettence: "throw their hair back as an accent by looking up"); "on count 1 or 5" is *unverified* | head / green | 0.41 (band 2) | Howl Up / Down, Behind Back Howl | Jettence pt 2, dZouk Zouk 3, Oaktown (a level guide on a personal `chatgpt.site` subdomain: secondary) |
| **Bate cabelo** (`bate-cabelo`) | Figure-8 of the upper body with a relaxed neck so the hair whips | head / green | 0.48 | Howl Up, Body Roll | Jettence, Oaktown |
| **Balão** (`balao`) | Upper body circles around a vertical axis (variant: balão apagado) | head / green | 0.24 | Body Roll, howl Down, Hug | Jettence |
| **Frango assado** (`frango-assado`) | Upper body rotates around a horizontal axis | head / green | 0.34 | howl Down, Body Roll | Jettence, Oaktown |
| **Boneca** (`boneca`) | Lambazouk head/upper-body move, often led visually | head / green | 0.44 | Howl Up, Behind Back Howl | Jettence pt 2, Oaktown |
| **Toalha** (`toalha`) | Head movement while turning with her back to the leader | head / green | 0.54 | Behind Back Howl, Intimate Howl | Jettence pt 2 |
| **Balanço** (`balanco`) | Lambada-derived angled basic, three steps forward or back (base paraense) | step / green | 0.17 (band 0) | Forward/Back Basic, Stand | Jettence pt 2, The Brazilian Ways |

A dry run of `build_graph` with this file (no file written) produced:
- 153 nodes, of which 10 are proposed: the 9 rows above plus the existing proposed `corredor`;
- 187 links;
- no validation errors;
- three warnings, all from the `corredor` collision (G1).

Re-run during the fact-check (2026-10-06, in memory, after the graph stage removed the separate `corredor`
and added `[display_names]`): 152 nodes, 9 proposed, 188 links, no validation errors, no warnings.

The head-movement nodes have no links yet: links are added when narrated dances show where they occur.

### 5.4 Hypotheses for the user (not merged automatically; `hypotheses[]` in the JSON)

| # | Question | Why it matters |
|---|---|---|
| H1 | **Howl Up / howl Down**: a head throw (head back and up, like a howl, close to chicote), or the *elástico* / *Raul* (Brazilian "R" sounds like "H": Raul ≈ "ha-OOL" ≈ "howl")? The user-confirmed legend (Tetra = head movements, howls) favours the head throw. | Decides whether chicote is a new node or an alias of Howl Up, and whether Elástico/Raul must be added (it is a confirmed zouk move: the leader stretches her forward like an elastic and she springs back) |
| H2 | **Little Turn vs Viradinha**: preparation, or the same move? (viradinha *means* little turn) | Duplicate node |
| H3 | Is either **Walk Around** node the leader circling her, i.e. the planada? | Would make Contrapeso an alias rather than a new node |
| H4 | Is **Bent Spin** the *tilted turn* (upper body tilted to the rotation axis)? | Otherwise propose Tilted Turn (Star, energy about 0.85) |
| H5 | Are the three **Dip** nodes, Dip Back, Fall Back and the two Side Dips different shapes (cambré, side dip, drop)? | Separate slugs instead of one `dip` |
| H6 | Is **Dive** the *mergulho*? | Alias vs a separate head movement |
| H7 | Is **Back And Forth** the *passo básico frente e trás*? | Alias |
| H8 | Body wave (*onda*/*ondulação*, unverified): **Body Roll** or **Do The Wave**? | Alias |
| H9 | **Lateral From Behind** = "Reverse Lateral"? (low confidence) | Alias |
| H10 | **Volta ao mundo**: head roll side to side, or a turn variant? Sources disagree. | Not proposed until answered |
| H11 | Separate nodes for *basic in place* and *travelling turn*? | Possible additions |
| H12 | Are the dancers Kadu Pires and Larissa Thayane? (Kadu Pires co-founded the Brazilian Zouk Dance Council in 2014; the pair also teach samba de gafieira) | If so their own move names are the natural standard |

### 5.5 The "cicada" question

No zouk source uses "cicada", "cigarra" or anything close. Ranked candidates:

1. **Sacada** (sa-KA-da). Closest sound. It is a tango and forró term for displacing the partner's free
   leg with your own, but it appears in no zouk syllabus we found (re-checked 2026-10-06). It is also a
   staple of samba de gafieira, the Brazilian ballroom partner dance, with many tutorials on "sacada de
   perna" and "sacada com giro em volta da dama". Brazilian zouk teachers who also teach gafieira may use
   the word, and Kadu and Larissa released zouk and gafieira DVDs (H12). That makes sacada more
   plausible than "no zouk syllabus" suggests.
2. **Chicote / chicotada** (shi-KO-chee / shi-ko-TA-da). Core zouk vocabulary: a whip-like head or hair
   accent. "Chicotada" shares the "-ada" ending.
3. **Cabeçada** (ka-be-SA-da). A generic word for a head movement.
4. **Sarrada**. Reportedly listed by one zouk school; no description found, and the fact-check could not
   re-find it (*unverified*).
5. **Calesita**. The tango "carousel": the leader walks around the follower while she pivots. It matches
   the counterbalance pivot in *meaning*, not in sound.

Resolution:
- Keep "cicada" as an unresolved alias (`unresolved_aliases[]`). Do not create a node.
- When the user narrates it, the tool shows the segment and asks: leg displacing her leg (sacada), or
  head/hair whip on an accent (chicote)?
- The answer becomes an alias through L5.

The user's "veraginha" for viradinha shows the pattern: names are written by ear. The alias-learning step
is the robust fix, not guessing spellings in advance.

### 5.6 Do not add, and graph clean-up
- Do not add **Tesoura** (in no zouk glossary checked) or **Sacada** (only a "cicada" candidate).
- Do not add **Patinha** as a node: it is the closing step inside the bônus.
- **Display clean-up**:
  - title-case "howl Down", "right Arm Out", "wrap Hug", "open Lunge In Front", "Behind Back hand Switch",
    "Cross wrap" and "Two Hand howl Down";
  - fix "Double Unwined Turn" (alias exists);
  - show "180" as "180° turn".
- **Repeated names** (Stand ×3, Dip ×3, Lateral ×2, ...) share one slug; `path.py` picks the instance from
  the previous node (G1/H5 may split Dip).
- **Blank and isolated nodes** are questions for the user. The narration step will show which ones are
  ever used.

---

## 6. Counterbalance

### 6.1 Definition
Sources: zouk contrapeso; west coast swing; contact improvisation; acro base/flyer; tango colgada and
calesita; the figure-skating death spiral, which is the same geometry with the roles swapped.

**Both partners lean away from each other, so that each alone would fall, while the pair is stable as one
system through a tensioned connection.** The connection is usually extended arms joined at the hands, or
the leader's hand on her shoulder blade. The two centres of mass offset each other and the **shared centre
of mass** sits between them.

**Pivot variant (Larissa/Kadu):**
- the follower's support foot is the fixed axis;
- the leader walks a circle around her at arm's length;
- she pivots to keep facing him;
- in zouk terms, a counterbalanced *planada*; in closed frame, a counterbalanced *pião*.

The leader may lift her slightly at the start, which stops her travelling sideways (*unverified* teaching
note). That lift is a possible onset cue.

### 6.2 Measurable criteria (per frame, world frame: right-handed, Y-up, metres, floor y = 0)

Smooth joints over about 0.2 s first. Foot points are SMPL-X joints 10/11 (balls of the feet)
or the lowest sole vertices; prefer the `timing.py` contact state where it exists.

| Criterion | Measurement | Threshold (candidate / strong) |
|---|---|---|
| **Anchor** (pivot only) | Follower foot in contact (`timing.py`, or ball of the foot < 0.05 m above the floor); horizontal drift from the interval median point; median speed | p95 drift ≤ 0.06 m (loose 0.10 m); median speed < 0.15 m/s. Pivot = that median point; `pivot_foot` = that foot |
| **Circling** (pivot only) | Leader pelvis angle around the pivot, θ = atan2(−(z_L − z_p), x_L − x_p), unwrapped (positive = counter-clockwise seen from above); radius r = horizontal distance | swept ≥ 90° to be a candidate, ≥ 270° to call it a full circle; ≥ 80 % of frames turn the same way; mean abs(dθ/dt) ≥ 30°/s; 0.4 ≤ r ≤ 1.4 m with std/mean < 0.25; follower pelvis yaw tracks θ within ±60° |
| **Lean-away** (both kinds) | u_i = horizontal unit vector from the partner's COM to dancer i's own COM; b_i = centroid of i's floor contacts (for the pivot follower, the anchored foot); d_i = (COM_i − b_i)·u_i | each d_i ≥ 0.04 m (about 2.5° at COM height ≈ 0.95 m lead / 0.92 m follow); strong ≥ 0.08 m (about 5°) |
| Lean-away, preferred physics test | Extrapolated COM (Hof 2005): XcoM_i = COM_i + v_i/ω0 with ω0 = √(g/l), l = COM height; margin of stability = distance from XcoM to the edge of the base of support | each dancer's margin is **negative on the away side**, while the couple's combined XcoM lies **inside the convex hull of all contacts** |
| **Connection** (both kinds) | `physics.partner_connections`: hand-to-hand / hand-to-wrist (or hand-to-upper-back in closed frame) distance; arm extension = shoulder-to-wrist ÷ (upper arm + forearm) | contact ≤ 0.12 m for ≥ 80 % of frames; at least one connecting arm with extension ≥ 0.90 (elbow about ≥ 130°). COM-COM ≥ 0.6 m in open hold supports it, but never decides alone: normal partnered stepping is already 0.39-0.72 m (Chen et al. 2015, human-robot partnered stepping) |
| **Duration** | Interval length | ≥ 1.5 s (about 2 beats at 76.4 BPM) |

### 6.3 Interval logic, scores and confidence
- **Scores.** Map each criterion to a 0..1 score with a logistic curve centred on its threshold. This fills
  `scores{anchor, circling, lean, connection}`.
  - Circling is scored on a sliding window of about 1.5 s; `swept_deg` is the interval total.
  - For `kind:"static"`, the anchor and circling scores are reported but do not gate the interval.
- **Gating.**
  - Static intervals are gated by min(lean, connection).
  - Pivot intervals are gated by the minimum of all four scores.
  - `confidence` is that minimum. Intervals whose weakest score is below 0.4 are kept but flagged, and
    drawn dashed (spec §3.9).
- **Hysteresis.**
  - Start an interval when the gate score stays ≥ 0.6 for 0.5 s.
  - End it when the score stays < 0.4 for 0.3 s.
  - Merge gaps shorter than 0.4 s.
- **Labels.** Pivot intervals with confidence ≥ 0.6 may be written to labels.json as
  `move:"counterbalance-pivot", provenance:"auto"`, but only into gaps: never over narration or human
  segments.

### 6.4 What the viewer draws from it
- **Yellow dot and axis.**
  - Combined COM: c = (m_L·COM_L + m_F·COM_F) / (m_L + m_F).
  - Masses come from `physics.py` (body volume × 1010 kg/m³, or user input). If those are missing, use mass
    ∝ stature², which gives a lead share of 0.522 for 1.727 m vs 1.651 m.
  - The dot is at (c_x, 0, c_z); the axis rises from it to c_y.
  - Export `com_track` at the capture frame rate. The dot fades in and out over about 0.2 s at the interval
    edges.
- **Pivot ring and leader arc** (floor craft, spec §3.5): the yellow ring at `pivot`; the leader's pelvis
  path projected to the floor as an arc labelled with `swept_deg` (for example "360°"); a tick per beat.
- **Follower spirals emphasised** (spec §3.7) during intervals: brighter, with a longer window for the free
  foot and free hand.
- **Implied tension** (Physics mode only, as an interval; our own quasi-static estimate, not from a source):

  T_i ≈ m_i · (g·d_i + ω²·r_i·h_com,i) / h_hand,i

  - Example: 65 kg, d = 0.08 m, hands at 1.1 m gives about 46 N.
  - A full circle in 4 s at r = 0.9 m adds roughly 110-150 N for the circling leader. For the anchored
    follower that term is near zero, so the two estimates should be the same order of magnitude but need
    not match.

### 6.5 Pitfalls
- **Gravity tilt.** `smplx_v1` is about 4.8° off gravity, which is about 0.08 m at COM height: larger than
  the lean threshold.
  - Use the refit (`smplx/`) when it exists.
  - Otherwise gate on the mean (d_L + d_F)/2, which cancels a global tilt to first order because u_L ≈ −u_F.
  - Also remove the 360°-periodic sinusoid that a tilt adds to d_i during a full circle.
- **Direction sign** (C2). Compute cw/ccw in the world frame as seen from above. It is a physical label and
  is not flipped for Unity.
- **Single-person physics tools** (ProtoMotions, PHC, MaskedMimic) cannot model the partner. They can only
  give *indirect* evidence: the follower tracked alone fails inside the intervals. That is an optional
  Runpod experiment, and untested.

### 6.6 Validation
1. A synthetic counterbalance segment with known pivot, radius, sweep and lean:
   `dancecap/dancecap/synthetic_counterbalance.py` already exists in the working tree.
2. The Larissa/Kadu window [56, 62] s.
3. Later, the user's narrated "contrapeso" labels: interval IoU and the share of narrated contrapeso
   seconds covered.

---

## 7. Labelling design: narration to timeline

1. **Proposals.** `auto` boundary candidates go to `proposals.json`. They come from:
   - per-eighth change points (F3/F4);
   - counterbalance intervals;
   - the beat grid.
2. **Narrate pass.**
   - The video plays through **headphones**, so music and lyrics stay out of the microphone.
   - Speed is selectable; 0.5× is recommended.
   - Pause-and-talk is allowed (EPIC-KITCHENS precedent).
   - An optional space-bar tap marks transitions. That costs only a reaction time of about 0.2-0.3 s
     (general knowledge) while speech supplies the label.
   - The browser records microphone PCM against `AudioContext.currentTime` and logs, every frame:
     audio clock, `mediaTime` from `requestVideoFrameCallback`, playback rate, paused flag and seeks.
   - `http://localhost` counts as a secure context, so microphone capture works without HTTPS.
3. **Transcribe** locally (A1-A3).
   - Word audio time → video time, by piecewise-linear interpolation through the log. Words spoken while
     paused map to the paused position.
   - Video time → reference time via `Source.to_reference`.
4. **Map phrases to moves.**
   - Normalise the text and take 1-4-word n-grams.
   - Score them with RapidFuzz WRatio against display names, Zouk1 names, aliases and learned ASR variants.
     Add a metaphone bonus and a prior for graph links leaving the previous move.
   - Accept at ≥ 88.
   - For 60-88, or when the top two are within about 5 points, optionally ask a TypeSafe `Choice`. Refuse
     when the weakest judgment is < 0.4: the move stays unresolved and is flagged for review.
   - Control words: "again" / "same" repeats the last move; "no" / "scratch that" / "undo" removes the
     last label. Fillers are dropped.
5. **Build segments.**
   - t_event = t_word − lag × rate. The lag prior is 0.8 s wall-clock (plausible range 0.3-2 s); it is
     learned per annotator as the median gap between t_word and the corrected t0.
   - Start = the nearest candidate boundary or type-1 eighth at or before t_event, within tolerance; else
     the nearest eighth.
   - End = the next segment's start.
   - Provenance `narration`; confidence = match score × timing score. Keep `transcript` and the extra keys
     from L3.
6. **Correct.**
   - Drag boundaries with snapping to the eighth grid (a modifier key disables snapping).
   - Relabel through a fuzzy search box; split, merge and loop.
   - Any edit sets provenance to `human`. Save labels.json v1, then run `path`, `fingerprint` and `export`.
7. **Learn.** Confirmed ASR strings become aliases (L5); the lag estimate is updated.
8. **Interchange.** `.eaf` export and import (L6).

**Evaluation**, once a classifier exists:
- Per-beat accuracy, segmental Edit score, F1@{10, 25, 50} (MS-TCN protocol), and macro-F1 per move with
  support counts.
- Boundary precision / recall / F at ±0.5 beat.
- Graph-level: the share of predicted transitions that lie on existing links; edge precision/recall against
  the human path; fingerprint JSD against the human fingerprint.
- An ablation with and without graph-constrained decoding.
- Ceiling: beat-wise Cohen's kappa on a double-annotated 5-10 %, and narration-pass vs careful-pass
  agreement.
- Splits are never random frames: whole takes now, whole couples later. `work/synthetic` `truth.json` is
  for pipeline tests only; it has no move labels.

---

## 8. Staged plan

### Stage A: now (infrastructure, no training, no pods)

| Piece | What | Runs | Installs (need the user's OK) |
|---|---|---|---|
| Graph | Merge `vocabulary_research.json`; Portuguese display names; Corredor = Lateral; hubs | Local CPU | none |
| Annotation tool | `python -m dancecap.moves annotate` with sub-commands serve, transcribe, map, export-eaf, import-eaf (Flask + waitress, already installed); works without ASR (A4) | Local | `rapidfuzz`, `pympi-ling` (small); optional `jellyfish` |
| ASR | faster-whisper large-v3-turbo fp16 in `.venv-asr` subprocess | Local 2080 Ti (`gpu.exclusive("asr")`) | `.venv-asr`: faster-whisper + the nvidia-cublas wheel (about 0.55 GB; cuDNN, 0.74 GB more, only if the smoke test needs it) + model (1.62 GB) |
| Features | Interpretable per-frame features pooled per eighth (list below) | Local CPU | none (numpy, scipy) |
| Proposals | Change points + snapping + clusters + band + repeats → `proposals.json` | Local CPU | optional `ruptures`, `stumpy` (both fit `.venv-nopo4d`); `claspy` only in its own venv (F3) |
| Counterbalance | Section 6 → `counterbalance.json` + Unity copy | Local CPU | none |
| Dataset | `python -m dancecap.moves dataset`: per-beat feature table + labels per take (Parquet/NPZ), the input for Stage B | Local | none |

**Interpretable features** (per frame, then pooled per eighth):
- **Per dancer:**
  - COM height relative to standing (dips, drops);
  - horizontal COM speed;
  - trunk lean (angle and direction relative to the partner);
  - pelvis yaw rate and accumulated yaw per beat (turns);
  - chest-minus-pelvis yaw (torso torsion: the corredor signature);
  - head orientation relative to the chest, and head angular speed (head movements);
  - foot contacts and step rate (`timing.py`);
  - mass-weighted kinetic energy (`physics.py` segment masses);
  - arm extension per arm.
- **Couple:**
  - couple-centre speed (the travel that the graph view removes);
  - COM-COM distance;
  - relative facing (face-to-face, side by side, behind);
  - the follower's bearing in the leader's chest frame (in front / side / behind: matters for Lateral From
    Behind, Hold Behind);
  - yaw rate of the leader-to-follower axis (circling);
  - hand-connection state (`physics.partner_connections`: which hands, one / both / none, hand-to-back);
  - counterbalance scores;
  - total couple kinetic energy (energy band).

### Stage B: first classifier (gated by labelled data)

| Gate | Model | Runs |
|---|---|---|
| **B1**: about 10 narrated dances; moves with ≥ 5 instances | Nearest prototype / kNN plus MiniRocket (aeon in its own venv, or a local MiniRocket port: aeon's pins conflict with `.venv-nopo4d`, see 3.3) on beat-resampled windows (e.g. 16 eighths). Inputs: interpretable features plus optional frozen TMR embeddings per dancer, concatenated in a **fixed role order (leader, follower)**, plus relational features. Then HMM/Viterbi over Zouk1 links (+ proposed links) with beat-duration priors. Leave-one-take-out validation. Used only to **pre-fill and rank** segments for the narrator (active learning: ask about the most uncertain boundaries first, as in B-ACT 2026). | Local CPU (MiniRocket trains in seconds); TMR on the 2080 Ti |
| **B2**: about 25-30 dances; ≥ 20 instances for the 10-15 most common moves | Promote to a reported classifier. Add an MS-TCN++ / ASFormer-style head on per-eighth features with the timestamp-supervision loss. Report the metrics in section 7. | Local 2080 Ti, fp16, minutes per run |

### Stage C: pretrained encoders and fine-tuning (about 60-100 dances)

| Step | What | Runs |
|---|---|---|
| Self-supervised pretraining | MAMP-style masked motion modelling (reimplemented) or USDRL (Apache-2.0) on all our unlabelled zouk captures, plus partner-dance corpora where the licence allows (CoMPAS3D NC, DD100 NC, Inter-X, the BONES-SEED dance subset if eligible) | **Runpod** for corpora above a few hours (A100/L40S; within the `dancecap.remote` $30 cap); small runs local |
| Two-person backbone | SkateFormer (M=2) or a 2P-GCN-style joint graph with hand/hip edges, fixed roles; or MotionBERT per dancer with LoRA / partial unfreezing (never its person-averaging action head) | Fine-tuning local on the 2080 Ti (fp16, short beat windows) |
| Segmentation head | MS-TCN++ first, then FACT or DiffAct | Local |
| Optional NVIDIA experiments | TMR-SOMA embeddings (after an SMPL-X to SOMA conversion; check fp32 on Turing, else Runpod); ProtoMotions follower-alone physics check; Cosmos Reason 2 video proposals | TMR-SOMA local if Turing works; the others **Runpod only** (Linux / Isaac / BF16) |

Throughout: NVIDIA weights, NC datasets and anything derived from them (embeddings, checkpoints) stay in
`dancecap/work/` or `weights/` (git-ignored), never in the public Unity repo.

---

## 9. Licences and Windows feasibility at a glance

| Licence class | Items | OK for us? | Public MIT Unity repo? |
|---|---|---|---|
| Permissive (MIT / BSD / Apache-2.0) | faster-whisper, CTranslate2, ruptures, ClaSPy, aeon, STUMPY, RapidFuzz, pympi-ling, TMR/TMR++, MotionBERT, USDRL, SkateFormer, 2P-GCN, MS-TCN++, ASFormer, DiffAct, FACT, ProtoMotions, SOMA-X, Kimodo code, Newton | Yes | Code ideas yes; weights still only where the weights' own licence allows |
| CC-BY-4.0 | Parakeet, Canary, SEED-Timeline-Annotations, AIST++ annotations | Yes (attribution) | Attribution needed; we have no reason to ship them |
| NVIDIA Open Model License | TMR-SOMA (per model card), GEM-X, Cosmos | Yes | No (keep weights and embeddings out) |
| Non-commercial / research-only | GEM (NVIDIA OneWay NC), Kimodo-SMPLX (NVIDIA R&D), CoMPAS3D (assume NC), DD100 / Bailando (S-Lab), DuetGen (Snap NC), BABEL, Motion-X, MotionPatches, DeST, keypoint-MoSeq (Harvard academic), FineDance, BRACE, InterGen | Yes (non-commercial research) | **No** |
| Gated / custom | BONES-SEED (eligibility for a company to check), Inter-X, ExPI | Check first | No |
| Copyleft | SMQ (AGPL-3.0), ELAN (GPL-3, used as a tool only) | As a separate tool | No |
| No licence | PULSE, InfoGCN, MAMP, Timestamp-supervision code, Action Motifs, MS-GCN (archived, no licence file), CALM (NOASSERTION) | Reimplement ideas; do not copy code | No |

**Windows (native, no WSL2):**
- *Yes:* faster-whisper (with A2), whisper.cpp, transformers, all CPU time-series tools (aeon and ClaSPy
  only in their own venv: their numba/scipy/scikit-learn pins conflict with `.venv-nopo4d`), PyTorch
  skeleton models on the 2080 Ti (fp16), ELAN, Newton, sherpa-onnx (CPU).
- *CPU only:* JAX (keypoint-MoSeq, dynamax).
- *Painful:* PySKL / MMAction2 (mmcv).
- *No:* NeMo, TAO, Isaac / MaskedMimic / PHC / SONIC, Cosmos Reason 2, GEM / GEM-X, CVAT. These run on
  Runpod only.
- *Unverified:* ProtoMotions through its Newton or MuJoCo backend (its docs are Linux-only).

---

## 10. Open questions for the user
1. **"cicada"**: sacada, chicote / chicotada, cabeçada, sarrada, or something else? (5.5)
2. **Howl** moves: head throw (≈ chicote) or Raul / elástico? (H1)
3. Little Turn vs Viradinha (H2); Walk Around = planada? (H3); Bent Spin = tilted turn? (H4)
4. Are the Dip variants different moves? (H5) Dive = mergulho? (H6) Back And Forth = frente e trás? (H7)
   Body wave = Body Roll or Do The Wave? (H8)
5. Accept the 8 new proposed Portuguese moves (5.3), or keep only those that appear in your dances?
6. Is the pivot "Contrapeso" (display), with planada as an alias, the name you want? Or would you rather
   name the move "Planada" and keep contrapeso only as the technique?
7. Approve the downloads for Stage A ASR (about 2.2 GB: the cuBLAS wheel plus the model; about 2.9 GB if
   cuDNN turns out to be needed) and the small Python packages (rapidfuzz, pympi-ling; optional ruptures,
   stumpy)?
8. Is the take Kadu Pires and Larissa Thayane? (H12)
9. BONES-SEED eligibility for Stage C (company size), if we ever want it.

---

## 11. Sources

**NVIDIA:**
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/nvidia/canary-1b-v2
- https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_customization/word_boosting.html
- https://github.com/NVIDIA-NeMo/NeMo/discussions/6515
- https://huggingface.co/nvidia/TMR-SOMA-RP-v1
- https://github.com/nv-tlabs/kimodo
- https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations
- https://huggingface.co/datasets/bones-studio/seed
- https://github.com/NVlabs/SOMA-X
- https://github.com/NVlabs/ProtoMotions
- https://research.nvidia.com/labs/par/project/maskedmimic.html
- https://github.com/NVlabs/GR00T-WholeBodyControl
- https://github.com/NVlabs/GENMO
- https://huggingface.co/nvidia/GEM-X
- https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/pose_classification/pose_classification.html
- https://huggingface.co/nvidia/Cosmos-Reason2-8B
- https://huggingface.co/nvidia/Cosmos-Embed1-336p
- https://github.com/NVIDIA/Isaac-GR00T
- https://github.com/NVlabs/HOVER
- https://github.com/newton-physics/newton
- https://github.com/isaac-sim/IsaacLab

**Segmentation, recognition and encoders:**
- https://github.com/deepcharles/ruptures
- https://github.com/ermshaua/claspy
- https://www.aeon-toolkit.org/
- https://github.com/stumpy-dev/stumpy
- https://github.com/dattalab/keypoint-moseq
- https://github.com/bachlab/SMQ
- https://arxiv.org/abs/2103.06669
- https://arxiv.org/abs/2603.06201
- https://github.com/sj-li/MS-TCN2
- https://github.com/ChinaYi/ASFormer
- https://github.com/Finspire13/DiffAct
- https://github.com/ZijiaLewisLu/CVPR2024-FACT
- https://github.com/HaoyuJi/LaSA
- https://github.com/kennymckormick/pyskl
- https://github.com/firework8/ProtoGCN
- https://github.com/KAIST-VICLab/SkateFormer
- https://github.com/Walter0807/MotionBERT
- https://github.com/maoyunyao/MAMP
- https://github.com/wengwanjiang/FoundSkelModel
- https://github.com/Mathux/TMR
- https://github.com/leorebensabath/TMRPlusPlus
- https://github.com/line/MotionPatches
- https://github.com/mgiant/2P-GCN
- https://arxiv.org/abs/2604.15173
- https://arxiv.org/abs/2210.10352

**Dance:**
- https://arxiv.org/html/2507.19684
- https://huggingface.co/datasets/Rosie-Lab/compas3d
- https://ar5iv.labs.arxiv.org/html/2008.10122
- https://arxiv.org/html/2405.19727
- https://graphics.cs.wisc.edu/Papers/2002/KGP02/
- https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/SnapTogetherMotion
- https://github.com/lisiyao21/Duolando
- https://github.com/snap-research/duet-gen
- https://arxiv.org/html/2508.16911v1
- https://ar5iv.labs.arxiv.org/html/2105.08825
- https://keypoint-moseq.readthedocs.io/en/latest/analysis.html
- https://ar5iv.labs.arxiv.org/html/2106.09696
- https://sdolivia.github.io/FineGym/
- https://jyx.jyu.fi/jyx/Record/jyx_123456789_93755
- https://pjyazdian.github.io/Salsa-Agent

**Narration and annotation:**
- https://github.com/SYSTRAN/faster-whisper
- https://github.com/SYSTRAN/faster-whisper/issues/1080
- https://raw.githubusercontent.com/OpenNMT/CTranslate2/master/python/tools/prepare_build_environment_windows.sh
- https://pypi.org/project/nvidia-cudnn-cu12/
- https://github.com/ggml-org/whisper.cpp
- https://developers.openai.com/cookbook/examples/whisper_prompting_guide
- https://alphacephei.com/vosk/adaptation
- https://archive.mpi.nl/tla/elan
- https://pypi.org/project/pympi-ling/
- https://labelstud.io/tags/timelinelabels
- https://github.com/rapidfuzz/RapidFuzz
- https://arxiv.org/html/2006.13256
- https://www.frontiersin.org/articles/10.3389/fpsyg.2011.00255/full
- https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback

**Zouk vocabulary:**
- https://bathzouk.co.uk/library/
- https://www.jettence.com/blog/names-of-brazilian-zouk-moves-in-portuguese-with-gifs/
- https://www.jettence.com/blog/names-of-brazilian-zouk-moves-in-portuguese-with-gifs-part-2/
- https://amozouk.com/zouk-move-dictionary
- https://amozouk.com/blog/the-3-fundamental-types-of-turns-in-zouk
- https://www.dzouk.com/zouk-university-program-toronto/
- https://oaktown-zouk-level-guide.taylorkinney.chatgpt.site/
- https://thebrazilianways.com/essential-portuguese-terms-every-zouk-dancer-should-know/
- https://www.salsaforums.com/threads/zouk-my-new-obsession.21650/page-3
- https://www.youtube.com/watch?v=jjAZJ5Far6g
- https://www.goandance.com/en/blog/post/123/5-basic-steps-of-zouk-for-beginners
- https://en.wikipedia.org/wiki/Brazilian_Zouk
- https://www.scielo.br/j/rbep/a/rNhzqXVKxSQHzYbJVB66Q6h/?lang=pt
- https://www.brazilianzoukcouncil.com/educational/history/history-details

**Counterbalance:**
- https://zoukology.com/tag/counter-balance/
- https://zouksidedown.wordpress.com/2014/12/12/out-of-balance-when-you-dance-try-these-tips/
- https://westiefied.com/west-coast-swing-definitions/
- https://myriadicity.net/contact-improvisation/learning-contact-improvisation/slight-counterbalance
- https://yogaslackers.com/blog/yogaslackers-principles-of-acrobatics/
- https://www.elizabethwartlufttango.com/blog/2018/7/26/colgadas-more-tips-for-off-axis-tango-moves
- https://en.wikipedia.org/wiki/Figures_of_Argentine_tango
- https://en.wikipedia.org/wiki/Death_spiral_(figure_skating)
- https://research.rug.nl/en/publications/the-condition-for-dynamic-stability/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4438977/

**Added by the fact-check (2026-10-06):**
- https://github.com/OpenNMT/CTranslate2/blob/master/src/cuda/cublas_stub.cc
- https://opennmt.net/CTranslate2/quantization.html
- https://pypi.org/project/ctranslate2/ (4.8.2, 2026-08-31, `cp311-win_amd64` wheel)
- https://pypi.org/project/faster-whisper/ (1.2.1, 2025-10-31)
- https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo
- https://github.com/m-bain/whisperX/blob/main/pyproject.toml
- https://github.com/ggml-org/whisper.cpp/releases
- https://github.com/alphacep/vosk-api/blob/master/src/recognizer.cc ("Ignoring word missing in vocabulary")
- https://www.mpi.nl/corpus/html/elan/ch06s04.html (segmentation mode, delayed mode)
- https://arxiv.org/abs/2008.10122 (ballroom smartwatch study: 85.95 % to 92.31 % with a first-order Markov chain)
- https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/installation.rst
- https://bones.studio/info/seed-license
- https://theperfectfollow.com/category/zouk/ (one-foot spin with the leader walking around her)
- https://www.brazilianzoukcouncil.com/profile/kadu-pires
- https://www.youtube.com/watch?v=R_IyzNzB3bM (samba de gafieira: "Sacada com Giro em Volta da Dama")

---

## 12. Fact-check log (2026-10-06)

An adversarial check of the twelve claims that drive decisions, each tested against primary sources (model
cards, Hugging Face / PyPI / GitHub APIs, source code, the papers). Corrections are applied inline above.

| # | Claim | Result | Effect on decisions |
|---|---|---|---|
| 1 | NVIDIA has no dance or partner-dance classifier; GR00T N1.7 is a robot VLA | Holds. GR00T N1.7 (April 2026) is a vision-language-action model for humanoid manipulation | None |
| 2 | Parakeet-TDT-0.6B-v3: CC-BY-4.0, 25 languages incl. (European) Portuguese, word timestamps; NeMo not for Windows | Holds. The model card prefers Linux and lists Ampere/Hopper/Blackwell/Volta GPUs; the NeMo Speech README documents only Linux, uv and Docker; the sherpa-onnx int8 port exists | None |
| 3 | TMR-SOMA-RP-v1: NVIDIA Open Model License, 256-d, 700 h incl. dancing, SOMA 30 joints, ≤ 300 frames at 30 fps, lists Windows | Holds. Listed GPUs start at Ampere, so Turing is still *unverified* | None |
| 4 | SEED-Timeline-Annotations: CC-BY-4.0, human vs propagated flag, DTW propagation | Holds. Field names corrected to `start_time`, `end_time`, `description`, `propagated_from_filename` | None |
| 5 | CoMPAS3D: SMPL-X, 11 classes + Other, 69.9 % / 53.6 F1, ELAN, kappa 0.752 on 5 %, ≥ 20 instances, 2,803 segments | Numbers hold (the best model is a fine-tuned InternVL3 on videos). **Wrong:** "gated". The HF dataset is not gated; CC-BY-NC-4.0, audio rights excepted | None (still NC; still a template) |
| 6 | faster-whisper large-v3-turbo fp16 on the 2080 Ti in a separate venv, CUDA wheels plus a PATH prepend | Engine choice holds: faster-whisper 1.2.1 (MIT) has `hotwords`, the turbo alias is 1.62 GB, ctranslate2 4.8.2 has a cp311 Windows wheel, and on compute capability 7.x float16 is native while bfloat16 falls back to float32. **Wrong mechanism:** CTranslate2 calls `SetDllDirectory(%CUDA_PATH%\bin)` before loading cuBLAS, and `CUDA_PATH` here points to CUDA 12.2, so a PATH prepend alone would load the old cuBLAS. **Over-sized:** Windows wheels are built `WITH_CUDNN=OFF`, so the 743 MB cuDNN wheel is probably unnecessary. **Ineffective:** `initial_prompt` with `condition_on_previous_text=False` only primes the first window | A2 now sets `CUDA_PATH` too and drops cuDNN unless the smoke test needs it; A1 uses `hotwords`; downloads about 2.2 GB, not 2.6 GB |
| 7 | WhisperX pins torch 2.8 and numpy 2 | Holds (`torch~=2.8.0`, `numpy>=2.1.0`) | None |
| 8 | whisper.cpp prebuilt `whisper-cublas-12.x-bin-x64.zip` with `--vad` | Partly wrong. The current asset is `whisper-bin-win-cuda-12.4.0-x64.zip` (684 MB), and `--vad` needs a separate VAD model (`-vm`) | Fallback 3 wording |
| 9 | Vosk drops grammar words missing from its lexicon | Holds (Kaldi warning in `recognizer.cc`) | None |
| 10 | ELAN: up to 4 synced videos, delayed mode, offsets only | Holds | None |
| 11 | "cicada" is unresolved; sacada (closest sound) is not zouk vocabulary | No zouk source for "cicada" or "sacada" was found. Sacada is, however, a core samba de gafieira move, a Brazilian partner dance that many zouk teachers (possibly this couple) also teach. "Sarrada" could not be re-found | Decision unchanged: no node, resolve at narration. Sacada stays first candidate, with a stronger rationale |
| 12 | "Contrapeso" is *the* zouk term (Wikipedia, SciELO 2023, dZouk) and the figure is a planada with the leader walking around her | Overclaimed. Only SciELO 2023 uses "contrapesos"; Wikipedia and dZouk say "counterbalance" in English. AmoZouk's planada is "a single-footed turn where the leader rotates the follower on a stationary axis" and does not mention him walking around her | Display name "Contrapeso" downgraded to *likely*; open question 6 stands |

Further corrections from the same pass:
- Corredor = Lateral rests on two sources, not three, plus the user's decision.
- aeon and claspy cannot be installed into `.venv-nopo4d` because of their numba/scipy/scikit-learn (and
  numpy) pins, which affects Stage B MiniRocket and the optional ClaSPy.
- ProtoMotions now has Newton and MuJoCo backends (native Windows *unverified*).
- Cosmos Reason 2 needs at least 32 GB.
- MS-GCN has no licence file.
- STUMPY moved to stumpy-dev.
- The Chen et al. 2015 distances come from human-robot stepping.
- Kadu Pires (not both dancers) is a BZDC co-founder.
- "Chicote on count 1 or 5" is *unverified*.

Not re-checked: the per-camera frame-rate figures in L4 (local data), Hof 2005, Snap-Together Motion, the
naming-latency range, and the licences in the 3.5 and 3.6 rows not named above.
