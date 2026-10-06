# TMR-SOMA implementation plan: move captions with calibrated confidence

2026-10-06, revised the same day after three reviews (feasibility, governance, method). The Review log at the
end lists what changed.

**This is a plan only.** Nothing in it has been installed, downloaded, trained, or run on a dance. One step
predates it: the research stage fetched some Kimodo files and tried a conversion on larissa_kadu before these
gates existed (section 3.2, open question 16).

**If you read only two parts:** "At a glance" just below, and the open questions in section 10.2.

These steps wait for the user's explicit OK, given in chat (section 2.1):
- recording each dancer's consent scope;
- downloading weights;
- spending pod money, and what gets uploaded to a pod;
- running a model on a dance;
- moving a model from one level to another;
- pushing anything that names real dancers to the public repo.

**Read with:**
- `dancecap/docs/MOVES_RESEARCH.md`, sections 1, 3.1, 3.5, 7, 8 and 12: why TMR-SOMA was chosen, the staged
  plan (Stages A, B1, B2, C) and its data gates.
- `head-movement/docs/VIEWER_SPEC.md`:
  - §2.2 silos and categories;
  - §3.10-3.13 graph, fingerprint, caption and reaction;
  - §10 confidence per provenance (§10 names this file);
  - §12 capture.json v4;
  - §14 acceptance tests.
- `dancecap/CONTRACTS.md`: world frame, master grid, GPU policy, pod jobs.
- The code it changes:
  - `dancecap/dancecap/moves/{labels,dataset,features,model,path,fingerprint,export}.py`;
  - `dancecap/dancecap/moves/annotate/`;
  - `dancecap/dancecap/remote_jobs.py`, `dancecap/remote/bootstrap.sh`;
  - `dancecap/dancecap/gpu.py`.

**Sources.** Technical claims come from the research stage and the three reviews of 2026-10-06, which read the
upstream code at the pinned versions below. Links are in the Sources section at the end. Claims nobody could
check are marked *unverified*. Numbers marked *proposal* are starting values, to be revised once data exists.

**Pinned upstream versions.** Record these in every output, and re-check them before any upgrade. Expected file
hashes are in section 2.5.

| What | Pin |
|---|---|
| Kimodo code (Apache-2.0) | `nv-tlabs/kimodo` commit `58e781898b3d7e328a676a75d3e338c45dce3ad9` (2026-09-22) |
| TMR-SOMA-RP-v1 weights (NVIDIA Open Model License, not gated) | `nvidia/TMR-SOMA-RP-v1` revision `e427752ae3446dedba49e928c93ddc9f0e413401` (2026-04-10) |
| SOMA-X (Apache-2.0); only the reference conversion uses it | `NVlabs/SOMA-X` commit `d6aa640f` = py-soma-x 0.3.3 (2026-09-29) |

**Terms used below:**
- **Eighth / half-measure.** The zouk grid of `features.py`: 8 eighths per measure, and a half-measure is
  4 eighths, i.e. 2 beats. At the Larissa/Kadu tempo (76.47 BPM, `work/larissa_kadu/beats/tempo.json`) an
  eighth lasts 0.39 s and a measure 3.14 s.
- **Window.** A stretch of one dancer's motion, cut on that grid; at most 300 frames at 30 fps.
- **Silo.** One level: professional, intermediate or novice.
- **LODO / LOCO.** Leave-one-dance-out / leave-one-couple-out evaluation.
- **Song group.** A dance plus every other dance to the same song. LODO always holds out a whole song group, so a
  rehearsed choreography cannot leak.
- **Blind dance.** A dance the user narrates and confirms end to end without seeing any model output. Only blind
  dances calibrate and score the captions (section 6.2).
- **Cross-fitted.** Every statistic used to score a held-out dance was fitted without that dance.

---

## At a glance

- **What TMR-SOMA becomes:** a *frozen motion encoder*. It turns each dancer's window into one 256-d vector.
- **What classifies the moves:** a small classifier of our own, trained **per silo** on the user's narrated
  labels. It turns `[leader vector, follower vector, couple features]` into move probabilities.
- **What fixes the sequence:** a decoder walks those probabilities along the zouk graph.
- **What makes the percentage honest:** calibration on blind dances, then an honesty gate (section 6.4). The
  caption shows a number only for the level, category and couple type that passed the gate, and the number is
  a cautious lower bound.

**What the viewer shows during directed playback:**
- the move's Portuguese name, with the English alias beneath;
- a confidence such as "70 %" (right at least about this often), or "uncertain" with two candidate names,
  "A or B", "unknown move", or "uncalibrated";
- the graph inset with the current node lit and the likely next moves glowing (a ranking, not a probability);
- the title as a reaction: **Kadu + Larissa × Ficar Sem Você**, with "Professional · Demo" above the arrow, and
  the dance fingerprint as the product.

```
smplx/{lead,follow}.npz --soma--> soma/<route>/{lead,follow}.npz   (uniform SOMA 30-joint skeleton, world frame)
features.npz (beat grid) ----+--> embed --> moves/embed/tmr_soma.npz   (W windows x 2 roles x 256)
                             |
      silo dataset (one level) --> classifier (per silo, cross-fitted) --> window probabilities
                                                               |
                         graph HMM decoder (zouk graph links) --> calibrated top-label p + ranked alternatives
                                                               |
labels.json (auto segments) --> path.json / fingerprint.json --> captions.json --export--> Unity caption + inset
```

**Where things stand today:**
- There is one Professional dance. Its 3D fit covers 5.8 s: 176 frames, i.e. 15 eighths. That is shorter
  than one 16-eighth window.
- Its labels are placeholders.
- So no real embedding, training or calibration can happen yet.
- Runpod: $5.89 of the $30 cap spent; no pod open (ledger, 2026-10-06).

**What can happen now:**
- Work that needs no approval: metadata, consent fields and silos, the approvals ledger, the export allow-list,
  the decoder, the caption rules and contracts, and the new signed head features.
- A small compatibility spike, once the about 19 MB download is approved.

**The real bottleneck is full-dance 3D, not this model.**
- A full dance means re-running the **whole capture pipeline** over the song, not just MAMMA: cameras drop in and
  out, so each stretch of the song needs its own camera set (section 3.8).
- Today's measured pod run puts MAMMA at about **$8-10 per camera set** for a 3-minute dance, or about **$18** with
  today's two-set workflow. The measurements also show a large fixed cost, so it could be as low as about $4.5
  per set. Ten dances: about **$45-100** with one set, **$110-180** with two. The cap is $30.
- Locally: about 9 h per set on the 2080 Ti. Whether a full-length fit fits in memory is *unverified*.
- First step proposed: a **pilot** that refits larissa_kadu over its whole 9-camera stretch (38.5 s) for about
  $2-2.6, to measure time and memory before deciding the budget (section 4.6, open questions 4 and 5).

**When will you see a number on an automatic caption?** Not before about 10 fitted, narrated and fully
confirmed Professional dances. Every dance narrated in blind mode before the classifier exists is blind by
default, so those 10 can be the blind set: at 30-60 segments per dance, that is 300-600 blind segments. In simulation a
well-calibrated model passes the honesty gate about half the time at 300 segments and about 9 times in 10 at 600
(section 6.4). Demo comes first. Lessons and Jack and Jill each need their own blind dances. A couple the model has
never seen needs at least 3 couples in the silo. A calibrated number on *narration* captions can come much sooner,
because it needs no 3D (section 6.3, step 7).

---

## 1. Goal: what TMR-SOMA will and will not do for us

### 1.1 What it is

- **Architecture.** A TMR dual encoder from NVIDIA's Kimodo release:
  - a motion encoder of 4.8 M parameters and a text encoder of 5.8 M;
  - 6 layers, 4 heads, width 256;
  - output: one L2-normalised 256-d vector per clip (the VAE mean).
  - Sources: model card; `config.yaml`; `kimodo/model/tmr.py` `_encode`.
- **Input.** *Global* joint positions of the 30-joint SOMA skeleton (`somaskel30`) at 30 fps, at most 300
  frames (10 s).
  - The code does not enforce the 300-frame limit. Longer clips run silently but are out of distribution.
  - Per window the model rotates and translates so that frame 0 faces +Z at x = z = 0.
  - It keeps the trajectory inside the window, the root height, root-relative joints, velocities and 4 foot
    contacts (`tmr_motionrep.py`, `reps/base.py` canonicalize).
- **Training data.** About 700 h of proprietary Bones Rigplay optical mocap, 170 subjects, single-person.
  - It is retargeted to **one** uniform body proportion.
  - The card says it covers "dancing" among many actions. There is no breakdown of dance hours or styles.
  - It was trained on the full dataset (train + test). Its published retrieval numbers (R@3 86-94 %) are
    therefore optimistic.

### 1.2 What we use it for

1. **Classifier features.** Per-dancer window embeddings, in a fixed role order (leader, follower), combined
   with our relational and interpretable features (section 5).
2. **Retrieval for the narrator.** Within a silo: "this window looks like 01:23 in Kadu + Larissa × Ficar Sem
   Você" (section 8.5). Never shown during blind narration.
3. **Optional, later: weak zero-shot hints.** English text prompts per move, shown in the annotation tool only
   (section 9, M10).

### 1.3 What it will not do

- **Name zouk moves.**
  - No zouk or partner-dance material is documented in its training data.
  - Our 126-move vocabulary (`taxonomy/zouk_graph.json`) is new to it.
- **See the couple.**
  - Each dancer is embedded alone, and the model removes each dancer's own heading and floor position.
  - Relative position, facing, hand holds and counterbalance are therefore invisible to it.
  - These features must keep coming from `features.py` (section 5.1).
- **Tell left from right reliably, or catch small details.**
  - The model card says it is "not always sensitive to small details in motion".
  - The card's example: "waving right hand" is encoded close to left-hand waving.
  - Many zouk moves differ only by side or by the head, so signed features of ours carry that information
    (section 5.1). Two of them are new and get added in M1.
- **Give probabilities.**
  - Its similarity score `cos/2 + 0.5` is a ranking score (`kimodo/metrics/tmr.py`).
  - It is never shown as a caption confidence.
- **Classify zero-shot from text.**
  - On generic actions, zero-shot BABEL-60 top-1 is 32.75 % for TMR and 40.11 % for TMR++ (arXiv 2405.16909).
  - TMR's zero-shot moment retrieval on BABEL reached about 20 % at IoU 0.4 (arXiv 2305.00976).
  - The text side needs LLM2Vec on the gated Meta-Llama-3-8B-Instruct: about 16 GB in bf16 (14.2 GiB measured
    in Kimodo issue #52). It does not fit the 11 GB GPU.
  - Text is therefore optional, off the critical path, and never a caption source.

### 1.4 How we decide whether it stays

TMR earns its place only through one pre-registered test (section 8.4, criterion A1):
- **Test:** cross-fitted LODO macro-F1 with versus without the TMR features, on blind dances.
- **Pass:** a one-sided paired sign-flip test over the per-dance differences, at the 5 % level.
- Before TMR feeds captions for couples the model has not seen, the same test must also pass under LOCO.

If it fails, nothing else in this plan changes. Silos, decoder, calibration and captions do not depend on
the encoder, and they work on the interpretable features alone.

If the SOMA conversion itself fails (section 3), the fallback encoder is the original **TMR / TMR++** (MIT). It
reads HumanML3D 263-d features built from SMPL-X body joints 0-21 (MOVES_RESEARCH §3.5). That needs no SOMA
conversion, but it is not a renaming either: HumanML3D retargets to its own uniform skeleton and runs at 20 fps,
so we would resample from 30 fps. The fallback therefore gets the same V-style checks (section 3.5).

---

## 2. Prerequisites and gates

### 2.1 What needs the user's OK

| # | Gate | Exactly what | Size / cost |
|---|---|---|---|
| G0 | **Consent per dance** | The user states, per dancer, what they agreed to: capture, 3D fit, ML analysis, cloud processing, name shown, level shown, publication in builds or renders, public repo. Plus who owns the footage, and bystanders. Recorded in the take's `[consent]` table (section 7.1). Nothing below runs on a dance whose required consent is missing. | none |
| G1 | **Licence** | The user accepts the NVIDIA Open Model License (version of 24 Oct 2025, shipped in the HF repo) for the weights. Key terms are listed below the table. Kimodo and SOMA-X code: Apache-2.0. | none |
| G2 | **Weights download, motion side only** | From the pinned revision via `huggingface_hub.snapshot_download(allow_patterns=...)` (huggingface-hub 1.29.0 is already in `.venv-nopo4d`):<br>- `config.yaml` (449 B)<br>- `last_weights/motion_encoder.pt` (19,172,062 B)<br>- `stats/motion/**` (8 files, each under 2 KB)<br>- `LICENSE`, `README.md`<br>Every file is checked against the hashes in section 2.5.<br>**Not** `motion_decoder.pt` (25.5 MB; inference does not use it). **Not** `text_encoder.pt` (23.2 MB) unless G7. | about 19.2 MB (whole repo 67.9 MB) |
| G3 | **Kimodo code and skeleton assets** | From the pinned commit:<br>- the 17 Apache-2.0 source files of section 2.4, plus `LICENSE` and `ATTRIBUTIONS.MD`;<br>- `kimodo/assets/skeletons/somaskel30/joints.p` (1,879 B), a pickle;<br>- only if the 77-joint check needs them: `somaskel77/joints.p`, `bvh_joints.p`, `standard_t_pose_global_offsets_rots.p` (11 KB together; also pickles);<br>- for the spike, the demo motions `kimodo/assets/demo/examples/kimodo-soma-rp/*/motion.npz`. | about 125 KB of code and licence files, plus the demo motions; the fetch command lists sizes before downloading |
| G4 | **Spike runs** | TMR on the Kimodo demo motions and on `work/synthetic`. Neither is dancer data. | local, minutes |
| G5 | **Conversion and embedding of real dances** | `moves soma` / `moves embed` on any non-synthetic take. Approval is per batch of takes. | local CPU, seconds per dance |
| G6 | **Pod spend and pod uploads** | Each spend request lists, per batch:<br>- money: the job's maximum minutes × rate, and the cap left;<br>- dancer data uploaded (SMPL-X fits are dancer motion and body shape);<br>- licensed files uploaded: SMPL-X model files (MPI licence) and, only if TMR ever runs on a pod, the NVIDIA weights;<br>- packages installed and assets downloaded on the pod;<br>- the cloud tier: COMMUNITY (third-party hosts) or SECURE;<br>- every gated action inside the batch (G5, G9).<br>The R1 reference conversion is one session, about $0.5. Full-dance 3D fits are budgeted in section 4.6. | ledger, $30 cap |
| G7 | **Text side (optional)** | The user requests Meta Llama 3 access on Hugging Face (gated, manual approval, Llama 3 licence). Then either download LLM2Vec (about 16 GB) or run it once on a pod; plus `text_encoder.pt` (23 MB). | about 16 GB, or about 1 pod-hour |
| G8 | **R1 conversion locally (optional)** | A new `.venv-soma` with py-soma-x 0.3.3, warp-lang, rtree, cholespy, usd-core and smplx (Windows wheels exist; native Windows is *untested*).<br>SOMA-X assets for the SMPL-X backend only: `SOMA_neutral.npz` (26.6 MB), `SOMA_template_rig.usda` (28 MB), `SMPLX/base_body.obj` + `SMPLX/SOMA_wrap.obj` (3.3 MB), `SOMA_procedural_transforms.json`; optionally `correctives_model.pt` (70 MB).<br>**Licence check first:** the SOMA-X README credits SMPL-Body (MPI) for the SMPL-to-SOMA correspondence and MHR for the correctives, so these assets may not be purely Apache. The same check applies before R1 runs on a pod.<br>**Never** MHR (696 MB). | about 60 MB plus packages |
| G9 | **Classifier training and inference on dances** | `model fit` / `model predict` stop refusing. Lifting the refusal is a separate code change made only after this approval. | needs B1 data too (section 8.1) |
| G10 | **Cross-level transfer** | A model or prototypes from one level used as a prior for another level (section 7.3, rule 5). | none |
| G11 | **Public repo** | Installing the guard hooks in `head-movement/.git/hooks`, and any push to `head-movement` that names real dancers, songs or levels (section 2.2). | none |

**Licence key terms (G1).** This is a reading of the licence, not legal advice.
- The grant is perpetual, royalty-free, non-exclusive and revocable.
- **Outputs** such as embeddings are not Derivative Models, and NVIDIA claims no ownership of them (1.1, 2.4).
  The user is responsible for the outputs and how they are used (2.4).
- **Trustworthy AI terms** are part of the licence (2.3). Among other things they forbid processing biometric
  information without the subject's consent where the law requires it, and intentionally misleading or deceiving
  others. That is one reason for G0 and for the honesty gate on caption numbers.
- **Derivative Model.** The frozen encoder plus our classifier head is arguably a "work based on the Model". That
  is harmless while nothing is distributed. Model files never leave `work/`.
- Redistributing the weights requires the Agreement plus a Notice: "Licensed by NVIDIA Corporation under the
  NVIDIA Open Model License" (3.1).
- The rights end on litigation against the model or on bypassing its guardrails (2.1).
- **Updates.** NVIDIA may update the Agreement. `SOURCE.json` records the `LICENSE` file's hash (section 2.5), and
  every fetch compares it. A change goes back to the user before anything else runs.
- **Kimodo code (Apache-2.0).** Every file we modify carries a prominent "modified by" notice (§4(b)), and
  Kimodo's `ATTRIBUTIONS.MD` travels with the vendored files.
- **Llama 3 (only with G7).** If any text-derived artefact were ever distributed, the Llama 3 naming and
  attribution duties would apply.

**How the gates are enforced in code:**
- **Approvals ledger.** `dancecap/approvals.toml`, tracked in the local workspace repo. One entry per approval:
  gate, takes, scope, expiry, date, and the user's chat message quoted verbatim. An agent writes an entry only
  right after the user's OK in chat. `python -m dancecap approvals list` shows them all, so the user can audit
  them.
  - Honest limit: a file an agent writes cannot prove consent. It makes every approval visible and reviewable,
    and the git history shows when each entry appeared.
- **Checked in the library, not only in the command line.** These functions refuse a non-synthetic take that the
  ledger does not cover: `soma.convert`, `embed.embed_take`, `decode.decode_take`, `MoveClassifier.fit` and
  `predict_proba`, the annotate pre-fill, and `captions.build` for renders.
- **Synthetic takes** are marked `synthetic = true` by their generator (in the take's TOML or the generated
  manifest), never recognised by name.
- **Pod jobs.** `remote run` refuses a job whose commands contain a gated sub-command the ledger does not cover.
- **Hard refusal stays.** `model fit|predict` keep their `return 2` (`model.py`) until a separate change, made
  after G9, that the user approves.
- **The spend cap.** No agent sets `DANCECAP_RUNPOD_CAP`. Raising the cap is the user's act.
- **Provenance.** Outputs record the approval's ledger id. They do not copy the free text, which could carry
  names.

### 2.2 Where things live

| Item | Location | Tracked |
|---|---|---|
| Weights subset, `LICENSE`, `NOTICE.txt`, `SOURCE.json` (`{repo, revision, files: {path: {bytes, sha256}}, license_blob, fetched, approval}`) | `dancecap/weights/tmr_soma_rp_v1/` | no (`/dancecap/weights/` in the workspace `.gitignore`) |
| somaskel30 rest skeleton. The pickle is checked against its pinned git blob hash, loaded once with `torch.load(weights_only=True)` (the research showed that this works), and saved as `somaskel30.npz` (names, parents, rest joints). No `weights_only=False` path exists. | `dancecap/weights/kimodo_assets/` | no |
| Vendored motion-only TMR code (Apache-2.0): upstream originals, one patch, the patched copies, Kimodo's `LICENSE` and `ATTRIBUTIONS.MD`, and a `NOTICE` naming the commit | `dancecap/dancecap/moves/tmr/_kimodo/` | local workspace repo only; never `head-movement` |
| Approvals ledger | `dancecap/approvals.toml` | local workspace repo |
| Consent, per dance | `[consent]` in `dancecap/takes/<take>.toml` or `takes/meta/<slug>.toml` | local workspace repo |
| SOMA joints, embeddings, kNN indices, prototypes, models, calibration records | `dancecap/work/...` | no |
| Text prototypes (optional; shared by all silos) | `dancecap/work/moves_shared/` | no |
| Unity copies: `labels`, `path`, `fingerprint`, `captions`, `graph` JSON. **Never** embeddings, indices, prototypes or model files | `head-movement/Assets/StreamingAssets/<capture>/moves/` | git-ignored, but **packaged into desktop and Quest builds** |

**Guards for what ships in builds:**
- `moves export` copies only a whitelist of JSON names and refuses anything from `moves/embed/` or
  `work/moves_silos/`.
- **Key-level allow-list.** `export.convert_labels` today copies every unknown key through. It, the fingerprint
  export and the new captions export keep only the keys DanceHud and DanceGraphLayer read. They drop
  `models.*.trained_on`, the calibration record path, approval ids, `transcript`, `notes`, an `annotator` that
  names a person, and every filesystem path.
- A test asserts the file whitelist on every capture folder, and asserts that no exported JSON contains another
  take's name or a filesystem path.

**Guards for the public repo.** `head-movement` is a separate git repo with a public GitHub origin. The
workspace `.gitignore` excludes it, so a workspace hook never sees its commits, and git hooks are neither
versioned nor run on `--no-verify`. Therefore:
- **Hooks (G11).** A pre-commit and a pre-push hook installed in `head-movement/.git/hooks`.
- **Test.** `tests/test_public_repo_guard.py` in dancecap runs `git -C head-movement ls-files` and
  `git diff origin/<branch>...HEAD`, and fails on any of these:
  - a tracked file under `Assets/StreamingAssets/`;
  - extensions `.pt .pth .p .pkl .npy .npz .parquet .safetensors .bin .ckpt .obj .usda`;
  - any blob over 1 MB added since `origin`;
  - a deny-list of dancer names and slugs taken from `takes/*.toml`, unless that dancer's consent allows the
    public repo.
- Public docs, tests and fixtures use synthetic dancer and song names.
- **Already pending:** `head-movement` branch `atlas-prototype` is 5 commits ahead of `origin`. Two of them
  (622c80f, 6429a96) put "Kadu + Larissa × Ficar Sem Você" and "Professional · Demo" into
  `docs/VIEWER_SPEC.md`. Nobody pushes them until the user decides (open question 12).

**Why embeddings stay private.** NVIDIA's terms would allow sharing them, because they are outputs. Keeping
them private is a **project** rule, for two reasons: dancer consent, and the licence row of
`CONTRACTS.md` (SMPL-X-derived data never enters the public MIT repo).

### 2.3 Turing / Windows compatibility spike (milestone M2)

**Why.**
- NVIDIA lists Ampere, Lovelace and Blackwell as tested hardware. Turing (our RTX 2080 Ti, sm_75) is not
  listed.
- The model card names Windows, but Kimodo was "developed on Linux".

**What makes success likely.** The motion path is plain PyTorch: `nn.TransformerEncoder`, GELU, LayerNorm.
- It imports only torch, einops, numpy and the standard library.
- It has no bf16 casts, no autocast, and no flash-attn, xformers, Triton or Warp kernels.
- The checkpoint size implies fp32 weights (about 4.79 M parameters).

So it should run, *unverified*. The spike proves it on data that is not a dance.

**Steps:**
1. **Fetch** (after G1-G3). `python -m dancecap.moves tmr fetch` lists files and sizes, needs `--yes`, and
   checks every file against section 2.5 before writing `SOURCE.json`.
2. **Load.** Use our own loader, `dancecap.moves.tmr.load_motion_encoder(device)`:
   - Parse `config.yaml` with OmegaConf (2.3.1, already installed). Build `ACTORStyleEncoder` and
     `TMRMotionRep` from the vendored code, with `text_encoder=None`.
   - **Never import `kimodo` or call its `load_model`.** Importing `kimodo` or `kimodo.model` pulls in hydra,
     safetensors and LLM2Vec (transformers, peft), and `_resolve_hf_model_path` snapshot-downloads the whole
     repo, decoder and text weights included. (`load_model` itself can skip the LLM when it is handed a pre-built
     `text_encoder`; issue #11 is an open PR adding a dummy mode. Vendoring avoids both problems.)
   - Build the skeleton as `SOMASkeleton30(load=False)`. Encoding needs only joint names and indices, so nothing
     is unpickled at encode time. R2 reads the rest joints only from the converted `somaskel30.npz`.
   - Load `motion_encoder.pt` after its SHA-256 matches section 2.5, with `torch.load(weights_only=True)`. If
     that fails, stop and report. There is no automatic fallback to `weights_only=False` (Kimodo itself uses it,
     which unpickles the file); using it would need the user's explicit OK.
   - Load the stats with `np.load(allow_pickle=False)`.
   - Call `.eval()`. Without it, dropout 0.1 makes every embedding random.
   - Run in fp32 under `torch.inference_mode()`, with keyword arguments:
     `encode_motion(posed_joints, original_skeleton=skel30, lengths=lengths, unit_vector=True)`. (Passed
     positionally, `lengths` would land in `original_skeleton` and fail.)
   - On CUDA, turn TF32 off (`torch.set_float32_matmul_precision("highest")`, `allow_tf32 = False`) and record the
     device. This changes nothing on Turing, but Kimodo issue #52 shows TF32 changes results on A100 pods.
3. **Check** on the demo motions (77 joints). Once section 3 exists, check the synthetic take too.

   | Check | Pass |
   |---|---|
   | shape and norm | output (B, 256), every norm 1 ± 1e-5 |
   | determinism | two CPU runs: max abs difference ≤ 1e-6 |
   | 77 vs 30 joints | cosine ≥ 0.99999 between the 77-joint input (`SOMASkeleton77(load=False)`, or the somaskel77 assets if slicing needs them) and the same input pre-sliced to 30 joints |
   | padding mask | one window alone vs the same window inside a padded batch with `lengths`: cosine ≥ 0.9999 |
   | CPU vs CUDA (2080 Ti, fp32) | cosine ≥ 0.9999 for every window |
   | standing Hips height (for V4) | distribution of Hips y on standing frames of the demo motions |
   | time-shift stability (report only) | cosine after shifts of 1, 5 and 15 frames |
   | mirror (report only) | cosine of a motion vs its left/right mirror; documents the handedness weakness |
   | speed (report only) | windows per second on CPU and GPU, peak memory |

4. **Report.** Write `work/moves_tmr_spike/report.json` plus a console summary.

**Fallbacks, in order:**
1. CUDA fails or disagrees on Turing → use **CPU fp32**. CPU is the default anyway: the encoder has 4.8 M
   parameters, and a dance has about 230 windows.
2. The vendored code fails on native Windows → use a `.venv-tmr` subprocess worker.
   - The interpreter is `$DANCECAP_TMR_PYTHON`, else `<workspace>/.venv-tmr`, the same pattern as
     `moves/annotate/asr_worker.py`.
   - Kimodo is installed there with `SKIP_MOTION_CORRECTION_IN_SETUP=1`, which needs approval.
   - Never install it into `.venv-nopo4d`: Kimodo pins `transformers==5.1.0` and `gradio>=6.8`, and pulls in
     peft.
3. Both fail → run on Runpod with a `tmr` bootstrap feature, batched with other jobs (G6 lists the weights
   upload).

**No fp16.** fp16 and autocast are neither documented nor tested by NVIDIA, and at this model size there is
nothing to gain. This differs from the `CONTRACTS.md` GPU default (fp16 autocast). It also differs from
`model.py` `PLAN`, which says "TMR on the RTX 2080 Ti (fp16, ...)"; M2 corrects that line.

### 2.4 Environment: vendor the motion path

**Recommendation:** vendor the motion-only inference path into `dancecap/dancecap/moves/tmr/_kimodo/`, so that
it runs inside `.venv-nopo4d` without new packages.

**The real import closure** (17 Apache-2.0 files at the pinned commit; read by the feasibility review):
- `kimodo/model/tmr.py`;
- `kimodo/{sanitize,tools,geometry,assets}.py`;
- `kimodo/motion_rep/{feature_utils,feet,stats,conditioning}.py`;
- `kimodo/motion_rep/reps/{base,tmr_motionrep}.py`;
- `kimodo/skeleton/{__init__,base,definitions,kinematics,transforms,registry}.py`.

Four package `__init__` files are replaced by minimal ones: `kimodo/__init__.py` (upstream imports
`model.load_model`, which pulls in hydra, safetensors and the Hugging Face downloader), `kimodo/model/__init__.py`
(upstream imports LLM2Vec, i.e. transformers and peft), `motion_rep/__init__.py` and `motion_rep/reps/__init__.py`
(upstream import `kimodo_motionrep`).

**How the copy stays honest:**
- `_kimodo/upstream/` holds the 17 originals byte for byte.
- `_kimodo/kimodo.patch` is one reviewed patch:
  - rewrites the absolute `kimodo.` imports (at least 4 files) to `dancecap.moves.tmr._kimodo.`;
  - replaces `load_checkpoint_state_dict` with a ten-line state-dict loader;
  - stubs `sanitize` for the unused text path;
  - adds the four minimal `__init__` files;
  - puts an Apache-2.0 "modified by dancecap, <date>" header on every file it changes.
- `_kimodo/` (the importable package) is the result of applying the patch.
- A test pins the SHA-256 of every original and of the patch. It re-applies the patch to the originals and
  asserts that the result equals the importable files.

**What the vendored files need:** only torch, einops (0.8.2, installed) and numpy, with Python ≥ 3.10; the venv
has 3.11.16.

### 2.5 Pinned hashes

Read from the Hugging Face API (`?blobs=true` at the pinned revision) and the GitHub tree of the pinned commit
on 2026-10-06. The fetch command refuses any file that does not match.

**Weights (SHA-256 of the LFS objects):**

| File | Bytes | SHA-256 |
|---|---|---|
| `last_weights/motion_encoder.pt` | 19,172,062 | `1fe9861da891b83f29bb1600255c9e2b9d4fabb2a6f1c34a0c609adc18345605` |
| `stats/motion/mean.npy` | 1,648 | `9ac9d47414dc8e777a2c0212c350b5dbbea5f43a31564bcc4b748ea9ae218f07` |
| `stats/motion/std.npy` | 1,648 | `8f662b94d580ad5e522235a07a237a354c2b66ca508902f4377a6d45a265147f` |
| `stats/motion/body/mean.npy` | 1,576 | `c7910fb7f6c594153989996cac20a958c41466d2abc8f3c3eef9e97a6d23053e` |
| `stats/motion/body/std.npy` | 1,576 | `60d36004772cbaa71e35bb45acc12e156b2dad00e960b13709d2aa5d0f652a3a` |
| `stats/motion/global_root/mean.npy` | 168 | `04572543751d7c5fb45b6cde6472be4f495d94ad2ca5e29c36c804cb9684a123` |
| `stats/motion/global_root/std.npy` | 168 | `7000e8d876f7b9b22a17c09a22e3a9517145bff8af1721ad468259ddda1217c5` |
| `stats/motion/local_root/mean.npy` | 160 | `3764a583e8cabd16b5132b666ef108b30843c302689ce0119de1478ed167bf41` |
| `stats/motion/local_root/std.npy` | 160 | `f579043030fcba03bd83f06a0add461aef09939cf347624edf986524364e1b20` |

**Small files of the weights repo (git blob SHA-1, as `git hash-object` computes it):** `config.yaml`
`100e526d58cd322dc8f21ae308f36883e21cbbcc`; `LICENSE` `8b5083f79b8856267b2fb16581ebc72e967b2e82`; `README.md`
`79cf7f6e4fa9aa19d407a2b6f32e70c039773148`.

**Kimodo pickles (git blob SHA-1 at commit 58e7818):**

| File | Bytes | Blob SHA-1 |
|---|---|---|
| `kimodo/assets/skeletons/somaskel30/joints.p` | 1,879 | `9e319f6fa3cb4959ed3221040b0ae945ec9646c7` |
| `kimodo/assets/skeletons/somaskel77/joints.p` | 2,967 | `8c5eb2e67e1314d8765c821ae8dd4e0ff1351a3b` |
| `kimodo/assets/skeletons/somaskel77/bvh_joints.p` | 3,390 | `3e3f2195fe6e094af5943cd98007344defa0c426` |
| `kimodo/assets/skeletons/somaskel77/standard_t_pose_global_offsets_rots.p` | 4,653 | `b6d68af1e73df0fc69c2cd9a59f2f6ed6aebb425` |

**Kimodo code:** `LICENSE` `e048a61796b130714fac0dffe9a3c2a1faf9521d`, `ATTRIBUTIONS.MD`
`ca10ff22ca65d441acfbceea3638894c69f7f909`. The 17 source files are fetched by commit, so git's own object
hashes pin them; their blob hashes are recorded in `SOURCE.json`, and the SHA-256 values in the vendoring test.

---

## 3. SMPL-X → SOMA conversion

### 3.1 What TMR needs, and why renaming joints is not enough

**Input format.**
- `posed_joints [T, 30, 3]` (a 77-joint input is sliced down to 30), float32, as **global** positions.
- Metres, right-handed, Y up, +Z forward, floor at y = 0.
- Foot contact means height under 0.10 m and speed under 0.15 m/s (`motion_rep/feet.py`).

**Frame compatibility.** Our world frame (`CONTRACTS.md`: right-handed, Y-up, metres, floor at y = 0) is the
same convention, so **no axis swap** is needed. Never feed the Unity copy: the export flips z (`unity =
(x, y, -z)`), which would mirror left and right.

**Body proportions.** The model knows **one** proportion, the rest skeleton `somaskel30`:

| Measure | somaskel30 rest |
|---|---|
| hip width | 0.201 m |
| thigh | 0.432 m |
| shin | 0.423 m |
| ankle → ToeBase | 0.142 m |
| upper arm | 0.287 m |
| forearm | 0.271 m |
| Hips → Head | 0.598 m |
| Hips above ToeBase | 0.989 m |

These numbers come from unpinned scratch copies (section 3.2) and are re-checked in M2 against the hash-pinned
files. In Kimodo's own SOMA outputs, ToeBase sits about 0.03 m above the floor and standing Hips at about
0.96 m (a bent-knee stance).

**Why renaming is not enough.** The SMPL-X neutral body (betas = 0) has different proportions:
- hip-joint width 0.122 m, thigh 0.383, shin 0.407, upper arm 0.268, forearm 0.253;
- its rest arm droops 16-18°;
- TMR's features are in absolute metres (root height, the 0.10 m contact threshold) and are normalised with
  statistics from that single proportion.

**No ready-made converter exists.** Kimodo converts only amass, kimodo, soma-bvh and g1-csv. SOMA-X's
`smpl2soma` and `convert_amass_to_soma` are SMPL-only demos (identity `smpl`, 22 joints).

**Our fits.** `smplx/{lead,follow}.npz` carry neck and head rotation, checked on the current fit:
- mean neck rotation 18-20° and mean head rotation 14-18° per dancer;
- `jaw_pose` is 0.

So TMR's head joints do move. MAMMA's "locked-head" model files are geometrically identical to the standard
`SMPLX_NEUTRAL.npz`: every array matches with zero difference.

### 3.2 Chosen route: R2 in production if it matches R1; R1 as reference and fallback

**R2: rotation-copy retarget onto the uniform skeleton.** Local numpy, no installs.
1. **SMPL-X rotations.** Run SMPL-X forward kinematics (FK) from the stored parameters: `global_orient`,
   `body_pose`, `jaw_pose`, the eye poses, `betas` (16) and `transl`. This gives global joint rotations.
   Check against the stored `joints` (V1).
2. **Rotation copy.** For each SOMA joint, take the SMPL-X global rotation of the mapped joint (3.3) and
   multiply it by a constant per-bone alignment `A_j`.
   - `A_j` turns the SOMA rest bone direction onto the SMPL-X rest bone direction.
   - Hips and Chest are aligned on both the torso axis and the left-right axis.
3. **Uniform skeleton.** Run FK on the `somaskel30` rest offsets. Bone lengths are then uniform by construction.
4. **Root scale.** Use one constant scale per dancer:
   `s = (SOMA Hips height above ToeBase + 0.03 m) / (SMPL-X pelvis height above the shaped rest sole)`.
   Apply it to the pelvis trajectory.
5. **Floor offset.** Add one constant y offset per take, so that the 5th-percentile ToeBase height is
   0.03 m. Never use a per-window offset: it would erase dips and jumps.
6. **Hands.** Ignore the fingers. ThumbEnd and MiddleEnd stay rigid to the hand (0.136 m and 0.190 m), as in
   Kimodo's outputs; MAMMA's hands are its noisiest part.

**Scratch prototype results.** The research stage tried R2 on larissa_kadu and on the synthetic take. The
prototype lives in that stage's scratchpad and is *not* in the repo; implementers re-implement it.

| Measure | Result |
|---|---|
| bone lengths | exact |
| limb and neck directions | 0.0° difference |
| torso direction (Hips → Neck1) | 2-5° median, up to 7.3° (the spines are segmented differently) |
| hip-vector heading | 0.15-0.4° median, < 1° max |
| root scale `s` | 1.006 (lead), 1.050 (follow) |
| Kimodo foot-contact heuristic vs the same heuristic on the SMPL-X joints | 81-97 % agreement |

**Disclosure: steps already taken before these gates existed.** To get those numbers, the research stage:
- fetched Kimodo's skeleton assets into its session scratch folder (`somaskel30`, `somaskel77` and `smplx22`
  `joints.p`, `standard_t_pose_global_offsets_rots.p`, `soma_base_fit_mhr_params.npz`), unpinned and unhashed;
- loaded `soma_base_fit_mhr_params.npz` with `allow_pickle=True`;
- ran the R2 prototype on larissa_kadu, a real dance.

That is G3 and G5 territory. Nothing left the machine. The user either approves it retroactively or asks for
the scratch copies and their dancer-derived outputs to be deleted (open question 16).

**R1: the official mesh-based route (reference).** Each arrow below is one step.
1. SMPL-X parameters → vertices with the `smplx` package (`flat_hand_mean=True`, `num_betas=16`).
2. Build the SOMA-X layer: `SOMALayer(identity_model_type='smplx', identity_model_kwargs={'model_path':
   SMPLX_NEUTRAL.npz, 'num_betas': 16}, mode='warp')`.
   - Pass `num_betas` explicitly: SOMA-X defaults to 10 and then fails or mis-shapes our 16-beta bodies.
3. Fit: `PoseInversion(soma, low_lod=True).prepare_identity(betas).fit(verts, body_iters=2, full_iters=1,
   lie_iters=3, batch_size=256)` gives 77 rotations.
4. `remove_joint_orient_local` converts them to T-pose-relative rotations, used as Kimodo `somaskel77` local
   rotations.
   - Assert the name mapping of SOMA-X's 78 public joints (77 plus a virtual Root) to `somaskel77`. Never map
     by index.
5. `SOMASkeleton30.from_SOMASkeleton77`, then Kimodo FK on the uniform skeleton with the same root scale,
   gives `[T, 30, 3]`.

About 60 lines of our own driver. Facts and caveats:
- **Published error** (arXiv 2603.16858, Table 2, AMASS with SMPL-X): 5.3 mm mean per vertex with the
  analytical solver, 4.1 mm with autograd refinement.
- **Not covered by that number:** our large betas (up to |3.46|) and partner occlusion.
- **Where it runs:** Windows, Warp and Turing are *untested*, so R1 runs on Runpod (G6) unless G8 is
  approved.
- **Pod environment.** SOMA-X 0.3.3's PyPI dependencies are unpinned, but its CI pins torch 2.10.0 and
  numpy 1.23.5. The pod base has torch 2.5.1 and numpy 1.26.4 (`bootstrap.sh feature_base`); `smplx` 0.1.28 is
  already in base. So `feature_check` for `soma` runs a tiny PoseInversion round trip (V9) on a synthetic SMPL-X
  body, and a version break shows up before the paid conversion starts.

**Why R2 is the production candidate:**
- it runs in `.venv-nopo4d` in seconds with no installs;
- it is deterministic and auditable;
- it is already prototyped.

**Why R1 stays.** It is the official route. R2's 2-5° torso-tilt bias falls exactly where lean and
counterbalance moves live.

**How the tie is broken.** The SOMA research stage recommended R1 as the default. This plan instead makes the
choice **data-driven**:
- R2 goes to production only if it passes V7 against R1 on `work/synthetic` and larissa_kadu.
- Otherwise R1 becomes production: one Runpod batch per set of dances, or `.venv-soma` after G8.

**Rules for both routes:**
- Read only `smplx/`, the new world frame.
- Refuse `smplx_v1/`: it is in the old frame, and its README says "do not mix".
- Refuse anything flagged by `features.world_frame_warning`.
- Mark frames as invalid when their SMPL-X quality is ≤ 0.1 (repaired) or they are flagged suspect.

### 3.3 Joint mapping (R2)

Each SOMA bone follows the global rotation of one SMPL-X joint. SMPL-X indices follow the standard 55-joint
order used in `features.py`.

| SOMA joint (index) | Driven by SMPL-X joint (index) | Note |
|---|---|---|
| Hips (0) | pelvis (0) | root; torso and left-right axes aligned |
| Spine1 (1), Spine2 (2) | spine1 (3), spine2 (6) | |
| Chest (3) | spine3 (9) | torso and left-right axes aligned |
| Neck1 (4), Neck2 (5) | neck (12) | SMPL-X has one neck joint; Neck2's position comes from the SOMA rest offsets |
| Head (6) | head (15) | |
| Jaw (7) | jaw (22) | 0 in our fits |
| LeftEye (8), RightEye (9) | left_eye (23), right_eye (24) | |
| LeftShoulder (10), LeftArm (11), LeftForeArm (12), LeftHand (13) | left_collar (13), left_shoulder (16), left_elbow (18), left_wrist (20) | |
| LeftHandThumbEnd (14), LeftHandMiddleEnd (15) | rigid to LeftHand | fingers ignored |
| Right arm (16-21) | 14, 17, 19, 21; ends rigid to RightHand | |
| LeftLeg (22), LeftShin (23), LeftFoot (24), LeftToeBase (25) | left_hip (1), left_knee (4), left_ankle (7), left_foot (10) | |
| Right leg (26-29) | 2, 5, 8, 11 | |

This mapping is checked by V3 and V6; it is not taken on trust.

### 3.4 Module, command and outputs

**Module:** `dancecap/dancecap/moves/soma.py` (plus `soma_review.py`). It is registered as an optional `moves`
sub-command: add `"soma"` to `OPTIONAL` in `moves/__main__.py`.

```
python -m dancecap.moves soma <take> [--route r2|r1] [--smplx-dir smplx] [--no-review]
python -m dancecap.moves soma <take> --compare      # R1 vs R2 report (needs both outputs)
```

**Outputs, one folder per route** (so a downloaded R1 result can never overwrite R2, or the reverse):
- `work/<take>/soma/<route>/{lead,follow}.npz` (`allow_pickle=False`). Keys:
  - `version`, `t` (T,) in reference seconds (the same as the SMPL-X `t`);
  - `joints` (T, 30, 3) float32 in the **world frame**;
  - `joint_names` (30,), `parents` (30,), `valid` (T,) bool;
  - `route` (`r1` / `r2`), `scale`, `y_offset`;
  - `fit_err_m` (T,) for R1, NaN for R2.
- `work/<take>/soma/<route>/report.json`:
  - `sources` (SMPL-X lead and follow: path and `sha256_16`);
  - `route`, `versions` (Kimodo commit, somaskel30 hash, SOMA-X commit for R1);
  - per dancer: `scale`, `y_offset`, checks V1-V9 with values and pass/fail;
  - `warnings`, `approval` (ledger id), `dance` (metadata snapshot, section 7.1).
- `work/<take>/soma/compare.json`: V7, written by `--compare`.
- `work/<take>/soma/<route>/review/soma_overlay.png` and `soma_angles.png` (section 3.6).
- `work/<take>/soma/pod_job.json` for R1 (section 4.6).

`embed` reads the production route (a constant in `soma.py`, set once V7 has decided), or the route given with
`--route`.

### 3.5 Validation tests and thresholds

| ID | Check | Pass threshold | Routes |
|---|---|---|---|
| V1 | Our SMPL-X FK reproduces the stored `joints` | max error < 0.1 mm (the research FK reached < 0.001 mm) | both |
| V2 | Uniform skeleton | every somaskel30 bone length equals the rest skeleton within 1e-5 m, every frame | both |
| V3 | Angles vs SMPL-X | knee, elbow, hip and shoulder flexion and limb bone directions: median ≤ 5°.<br>Torso tilt (Hips→Neck1 vs pelvis→neck): median ≤ 5°, 95th percentile ≤ 8° (*proposal*).<br>Hip-vector heading: median ≤ 2°. | both |
| V4 | Floor and height | Hips y on standing frames (both feet in contact, COM speed < 0.2 m/s) within [0.94, 1.03] m. This is provisional: M2 re-derives the bounds from the Kimodo demo motions and the rest skeleton. (R2's scale puts a straight-legged body at about 1.02 m, while Kimodo's outputs stand bent-kneed at about 0.96 m.)<br>5th-percentile ToeBase height 0.03 ± 0.01 m.<br>No frame with ToeBase < −0.02 m. | both |
| V5 | Contacts | Kimodo's contact heuristic on Foot/ToeBase agrees with `timing.py` contacts in ≥ 85 % of frames | both |
| V6 | Handedness | LeftHand and LeftFoot lie on the anatomical left of the pelvis forward axis in ≥ 99 % of frames, matching the SMPL-X side | both |
| V7 | **R1 vs R2 agreement (decides the route)** | Mean MPJPE on somaskel30 after root alignment < 3 cm.<br>Once G5 allows TMR runs: embedding cosine median ≥ 0.95 and 5th percentile ≥ 0.90 on the same windows (*proposal*; revisit after the first look). | R1 + R2 |
| V8 | Inversion fit | `per_vertex_error` take mean < 10 mm.<br>Flag frames whose max is > 50 mm or > 3 × the take median.<br>Hands, feet and head reported separately. | R1 |
| V9 | Round trip | `SOMALayer.pose` with the fitted rotations on the dancer's identity reproduces the SMPL-X vertices: mean < 1 cm, max < 5 cm (SOMA-X's own test thresholds) | R1 |

### 3.6 Review image

**`soma_overlay.png`** shows four frames, chosen automatically from `features.npz`:
- the lowest COM (a dip);
- the highest pelvis yaw rate (a turn);
- the highest head angular speed (a head throw);
- one neutral frame.

Each frame has a front view and a top view. It draws the SMPL-X 55 joints in grey under the SOMA-30 stick
figure (lead red, follow white, the viewer colours). The uniform skeleton stands at the dancer's root, and
L and R labels are drawn on the hands and feet.

**`soma_angles.png`** plots, over time, the V3 differences (torso tilt, heading, limb directions) and the
contact agreement.

### 3.7 Unit tests (`tests/test_moves_soma.py`)

The tests use synthetic data only: no dancer data, no weights. A small rest-skeleton fixture stands in for
`somaskel30.npz` until G3.
- **Rest pose.** An SMPL-X rest pose maps to the somaskel30 rest pose after scaling (error < 1e-6 m).
- **Random poses.** Bone lengths are exact (V2). Left and right are preserved (V6), and a mirrored input flips
  them.
- **Constants.** `scale` and `y_offset` are constant per dancer and per take.
- **Refusals.** `smplx_v1/` and any Unity-frame document are refused.
- **Names.** The joint-name assertion fails loudly when a name is missing.
- **Masks.** Repaired and suspect frames get `valid = False`.
- **Routes.** R1 and R2 write to separate folders.
- **No pickles.** No test loads a pickle.

### 3.8 Data prerequisite: full-dance 3D (P0)

Today `smplx/` covers 176 frames of larissa_kadu (t = 56.08-61.91 s, the take's `window = [56, 62]`). That is
15 eighths, so **no 16-eighth window exists**.

**Widening the take window is not enough.** A full dance needs the whole capture pipeline re-run over the song:
1. frames and people detection/tracking over the song;
2. camera localisation: `cameras/<cam>.npz` is solved per canonical window frame (`CONTRACTS.md`), so every new
   frame needs its own camera solve;
3. sync, timing and beats;
4. MAMMA `ma_2d` preparation;
5. MAMMA fits, with **a camera set per availability span**;
6. selection, then stitching the spans together;
7. then features and counterbalance.

**Why camera sets change along the song (larissa_kadu, from `alignment.json`):**

| Reference time | Cameras available (08 is excluded as a duplicate of 09) |
|---|---|
| about 10-30 s | 01, 02, 03, 04, 05, 06, 09 |
| 29.97-68.51 s | 01-07, 09 (all; the take's comment "all 9 overlap" refers to this stretch) |
| 68.51-70.2 s | 04 has ended |
| 70.2-about 159 s | 01, 02, 03, 05, 06, 07 (09 has ended) |
| about 159-183 s | 01, 02, 03, 05, 06 (07 has ended) |

Both current MAMMA sets (c7 = 01, 02, 03, 04, 06, 07, 09; c8 = 01-07, 09) contain 04 and 09, so neither works
after about 70 s.

**What P0 costs** is in section 4.6. The runtime of steps 1-4 and 7 over a whole song is not measured yet; the
pilot measures it.

**Dances that exist only as one video** (no multi-camera capture) get no 3D fit and therefore no embeddings. They
still feed the label statistics (section 7.1), and their narration can be calibrated (section 6.3, step 7).

---

## 4. Embedding extraction

### 4.1 Windows on the beat grid

**Grid.** Windows use the eighth table of `features.npz` (`eighth_t0`, `eighth_t1`, `eighth_slot`,
`eighth_frames`). That is the same grid as the features and the dataset:
- `timing/timing.json`, else `beats.json`;
- the studio-track grid from `song.py` (`song/beats_studio.json`) once it is adopted upstream.

**Primary windows:**
- **Length:** 16 eighths (two measures, eight beats).
- **Starts:** every half-measure, given by `dataset.window_starts(fs, window=16, hop=4)`. Windows therefore
  align one to one with the dataset windows (`start_eighth`).
- **Example:** at 76.47 BPM, 16 eighths last 6.28 s, i.e. 188 frames.
- The length is revisited once B1 data gives the real distribution of move durations (section 8.3).

**Frames.**
- A window takes the master-grid frames with `t ∈ [t0, t1)`.
- Our fits are on an exact 30 Hz grid, so nothing is resampled.
- A future source that is not on 30 Hz is resampled by time `t`, never by frame index.

**Limit.**
- At most 300 frames per window. 16 eighths stay under that whenever the tempo is ≥ 48 BPM.
- Otherwise the window drops to 8 eighths, and `win_window_eighths` records the change for that window.
- No time warping: speed carries information.

**Secondary windows (ablation, 8.3):** 8 eighths (one measure, 94 frames) at the same starts.

**Validity**, per window and role:
- Invalid runs of ≤ 0.2 s are linearly interpolated.
- The window is valid when at least 90 % of its frames are valid and no invalid run is longer than 0.2 s.
- An invalid window is stored as zeros with `valid = False`, and is masked everywhere downstream.

**Stretches with no grid.** Some lesson stretches have no music, e.g. talk-throughs.
- They use fixed 6.0 s windows with a 1.5 s hop, marked per window: `grid[w] = "time"` and
  `win_start_eighth[w] = -1`.
- They serve only to train and score the `_none` class (section 5.4).
- The decoder never steps through them: a no-grid stretch is emitted as `_none` and breaks the chain.

**What gets embedded.** Everything, including `exclude` spans (7.5). The dataset decides what to use.

**Size.** A 3-minute dance gives about 115 windows per dancer (180 s / (4 × 0.39 s)).

### 4.2 Per-dancer embeddings in a fixed role order

**Role order.** Role 0 is the **leader** and role 1 the **follower**, as in `features.ROLES = ("lead",
"follow")`. The order never changes, and both dancers go through the same frozen encoder.

**Per-eighth embeddings.** `eighth_emb` is the mean of all windows covering an eighth.
- Each window is weighted by a Gaussian centred on the window (sd = window / 4), exactly as
  `model.window_votes`.
- The mean is then re-normalised to unit length.
- It is used for retrieval and for optional eighth-level decoding.

**Optional experiment (ablation only; an untested idea).** A third channel holds the follower's joints
expressed in the leader's frame-0 root frame, which keeps her path around him.

### 4.3 File contract: `work/<take>/moves/embed/tmr_soma.npz` + `tmr_soma.json`

**`tmr_soma.npz`** (`allow_pickle=False`):

| Key | Shape / type | Meaning |
|---|---|---|
| `version` | int | contract version (1); bump on any key change |
| `model`, `hf_revision`, `kimodo_commit` | str | `nvidia/TMR-SOMA-RP-v1`, the pinned revision and commit |
| `weights_sha256_16`, `stats_sha256_16`, `code_sha256_16` | str | motion encoder, stats, vendored code |
| `route` | str | `r1` or `r2` (section 3) |
| `roles` | (2,) str | `["lead", "follow"]` |
| `fps`, `window_eighths`, `hop_eighths` | int | 30, 16, 4 |
| `grid` | (W,) str | `beat` or `time`, per window |
| `win_start_eighth` | (W,) int | the join key with `dataset.window_starts` and `features.npz`; −1 for time windows |
| `win_window_eighths` | (W,) int | 16, or 8 when the 300-frame limit forced it; 0 for time windows |
| `win_t0`, `win_t1` | (W,) float64 | reference seconds |
| `n_frames` | (W,) int | ≤ 300 |
| `emb` | (W, 2, 256) float32 | unit vectors; zeros when invalid |
| `valid` | (W, 2) bool | per role |
| `quality` | (W, 2) float32 | mean SMPL-X `quality` of the window's frames |
| `eighth_emb` | (N, 2, 256) float32 | row-aligned with the `features.npz` eighths |
| `eighth_valid` | (N, 2) bool | |

**`tmr_soma.json`:**
- `version`, `take`, `created`, `approval` (ledger id);
- `sources`: `smplx_lead`, `smplx_follow`, `soma_lead`, `soma_follow`, `features`, `features_schema`, each as
  `{path, sha256_16}`;
- `config`: window, hop, minimum valid share, interpolation limit;
- `device`, `dtype: "fp32"`, `tf32: false`, `runtime_s`;
- `dance`: level, category, couple_id (section 7.1);
- `cache_key`, `warnings`.

### 4.4 Command

```
python -m dancecap.moves embed <take> [--model tmr-soma] [--route r1|r2] [--device cpu|cuda] [--window 16]
                                      [--hop 4] [--force]
python -m dancecap.moves embed --check        # weights, hashes, vendored code, imports; runs nothing
```

**Module:** `dancecap/dancecap/moves/embed.py`, with the encoder in `dancecap/dancecap/moves/tmr/`. Add
`"embed"` and `"tmr"` to `OPTIONAL`. The approval check sits inside `embed.embed_take` (section 2.1).

**Encoder interface:** `Encoder.encode(joints [B, T, 30, 3], lengths [B]) -> [B, 256]`.
- A `FakeEncoder` (a fixed random projection plus normalisation) implements the same interface.
- It lets every plumbing test run without NVIDIA weights.

### 4.5 Caching and staleness

- **Cache key:** the SHA of these inputs:
  - the SOMA files and `features.npz`;
  - the config;
  - the weights SHA-256 and the vendored-code hash;
  - `VERSION`.
- **Re-runs.** An unchanged key skips the work unless `--force` is given.
- **`embed` refuses** in two cases (the same pattern as `fingerprint.stale_sources`):
  - the SMPL-X `sha256_16` in `soma/<route>/report.json` differs from the one in `features_schema.json`;
  - the SOMA files are older than the SMPL-X files.
- **Who else checks.** `moves validate <take>` reports stale embeddings, and `moves dataset` refuses them.

### 4.6 Where it runs, and what it costs

**Local by default.**
- CPU fp32, in process, through the vendored code.
- GPU only with `--device cuda`, in fp32 with TF32 off, inside `dancecap.gpu.exclusive("tmr")`.
- The `.venv-tmr` worker is the fallback (2.3).
- Features, R2, `--compare` and `embed` take seconds on the local CPU, so they never run on paid pod time.

**What was measured** (pod session `20261006-113945-batch`, A100 80 GB, COMMUNITY, $1.39/h, 176 frames of
larissa_kadu):

| MAMMA run | Wall time |
|---|---|
| dry check, 3 frames | 303 s (280 s of it optimisation) |
| fit c7 (7 cameras) | 665 s |
| fit c8 (8 cameras) | 865 s |

The 3-frame dry check alone took 280 s of optimisation, so a large part of each run may be a fixed cost. That
gives two extrapolations: *linear* in frames (upper, used for planning) and *fixed cost plus per frame* (lower).

**Costs for one 3-minute dance (5,400 frames):**

| Step | Local | Runpod (A100, $1.39/h) |
|---|---|---|
| R2 conversion (2 × 5,400 frames) | seconds on CPU (*estimate*) | not needed |
| TMR embedding (about 230 windows of ≤ 188 frames) | seconds to about a minute on CPU fp32 (*estimate*; measured by the spike) | not needed |
| R1 reference conversion | only after G8 (untested on Windows) | standalone session about 20 min including bootstrap and upload ≈ $0.46. The paper reports 882 fps on an A100, i.e. under a minute of fitting. |
| MAMMA, one camera set per span | about 8.7 h on the 2080 Ti (1,042 s per 180 frames, linear) | linear: about 340 min (7-camera) to 440 min (8-camera) ≈ **$8-10**. Lower bound: about 190-295 min ≈ $4.5-7. |
| MAMMA, today's two-set workflow (both sets, then select) | about 17 h | linear: about 780 min ≈ **$18**. Lower bound: about 490 min ≈ $11. |
| Session overhead (bootstrap, 170 MB upload) | none | about 15-20 min ≈ $0.4 per session |
| Capture steps 1-4 and 7 of section 3.8 over the whole song | not measured yet | not planned on pods |

**For the B1 gate (about 10 dances):** about **$45-100** with one camera set per span, about **$110-180** with
today's two-set workflow. Locally: about 9 h per dance and set.

**Memory is unverified.** MAMMA peaked at 3.2 GB for 180 frames on the 2080 Ti, and `mamma_dyn.py` has no
chunking. How memory grows with sequence length is unknown, so a full-length fit may need chunked fits with
overlap (for example 30-60 s chunks overlapping by 2 s), even on an 80 GB pod (section 10.3).

**Proposed first step: a pilot (P0a).** Refit larissa_kadu over its whole 9-camera stretch, 29.97-68.51 s
(38.5 s, about 1,156 frames), with set c8 only. Every camera of c8 is present there, so no new camera set is
needed; the capture steps 1-4 still run locally over that stretch.
- Pod cost: about 67-95 A100-minutes plus setup, ≈ **$2-2.6**, inside the $24.11 left under the cap.
- It measures: time per frame at 6.6 × today's length, peak memory, the local capture-step runtimes, and whether
  chunking is needed.
- It also yields the first real windows: 38.5 s gives about 22 windows per dancer.

**Budget.**
- $5.89 of the $30 cap is spent, and no pod is open (ledger, 2026-10-06).
- Ten full dances exceed the cap in every variant above.
- The user chooses (open question 5): raise the cap, run MAMMA locally overnight, or mix the two; one or two camera
  sets per span; COMMUNITY or SECURE cloud for jobs that carry SMPL-X files and dancer data (open question 6).

**R1 pod job.** `work/<take>/soma/pod_job.json`, schema `dancecap.pod_job/1`. It does **only** the R1 conversion.
It cannot ride in the same session as that dance's MAMMA fit: `smplx/` is promoted locally after the MAMMA
download (`mamma select`), so it does not exist on the pod. It batches instead with the next pod session.

```json
{"schema": "dancecap.pod_job/1",
 "name": "soma-r1-<take>",
 "take": "<take>",
 "max_minutes": 15,
 "needs": ["soma"],
 "inputs": ["dancecap/work/<take>/smplx/lead.npz", "dancecap/work/<take>/smplx/follow.npz",
            "dancecap/weights/kimodo_assets/somaskel30.npz"],
 "commands": ["python -m dancecap.moves soma <take> --route r1"],
 "outputs": ["dancecap/work/<take>/soma/r1/"],
 "extract": "workspace",
 "after_download": ["python -m dancecap.moves soma <take> --route r2",
                    "python -m dancecap.moves soma <take> --compare",
                    "python -m dancecap.moves embed <take> --model tmr-soma"]}
```

- The NVIDIA weights are not uploaded: TMR never runs on the pod in this plan.
- The pod downloads only `soma/r1/`, so nothing local (such as `features.npz`) can be overwritten.
- `soma --route r1` on a real take is a G5 action, so `remote run` checks the approvals ledger before the pod
  starts, and the G6 request lists it.

**Runner changes this needs (M3):**
- `remote_jobs.FEATURES` gains `soma`, with entries in `BOOTSTRAP_MIN` and `BOOTSTRAP_MIN_CACHED`.
- `soma` joins `MODEL_FEATURES`, so the SMPL-X model files are bundled. **This sends MPI-licensed files to a
  third-party host**, as today's MAMMA jobs already do. The SMPL-X licence allows one archive copy and forbids
  making the model available to third parties (a reading, not legal advice), so the user decides whether such
  jobs may use COMMUNITY cloud (open question 6).
- `bootstrap.sh` gains `feature_soma`: py-soma-x at the pinned commit, warp-lang, rtree, cholespy, usd-core, the
  SOMA-X assets fetched to an explicit `data_root` (after the G8-style licence check), and a `feature_check` that
  runs the tiny V9 round trip.
- **Budget rules.** The ledger, the $30 cap and the on-pod kill switch apply unchanged.
- **Batching.** Several takes share one session through
  `python -m dancecap remote run --jobs <a.json>,<b.json>`.

---

## 5. Couple representation and classifier

### 5.1 Couple vector per window

```
x = [ CL2N(e_lead), CL2N(e_follow), w_r · r / sqrt(d_r), h ]
```

| Part | Meaning |
|---|---|
| `CL2N(e)` | SimpleShot centring (arXiv 1911.04623): subtract the silo's mean embedding for that role (fitted inside each training fold only), then re-normalise to unit length |
| `r` | **Relational features** from `features.npz` `pooled_z`, pooled over the window's eighths as mean / min / max / delta (robust z). The columns are listed below this table. |
| `h` | **Signed handedness features**, because TMR blurs left and right. Per dancer, over the window: accumulated pelvis yaw (from the signed `{r}_yaw_rate`: turn direction and size), mean `{r}_torsion` (signed), and two **new** signed features, `{r}_head_roll_side` and `{r}_head_yaw_rel` (below). |
| `w_r` | weight of the relational block; 1 until B2, then chosen from {0.5, 1, 2} by nested LODO; `d_r` = dim(r) |

**The relational columns (`r`):** `couple_speed`, `couple_yaw_rate`, `axis_yaw_rate`, `com_distance`,
`follow_lateral`, `follow_forward`, `facing_cos`, `facing_sin`, `couple_ke`, `hand_hand_min`, `n_hand_holds`,
`lead_hand_on_follow`, `follow_hand_on_lead`, `cb_score`, `cb_lean`, `cb_connection`, `cb_anchor`,
`cb_circling`, `cb_interval`.

They must come from `features.py`, because TMR removes each dancer's heading and floor position.

**The two new signed head features (milestone M1).** Today's head features cannot carry side:
`{r}_head_off_axis` is an unsigned angle and `{r}_head_ang_speed` an unsigned norm.
- `{r}_head_roll_side`: the head's tilt toward the dancer's own left (+) or right (−), measured about the chest's
  forward axis. From SMPL-X joints 12 (neck) and 15 (head) against the chest frame from 9 (spine3) and 16/17
  (shoulders).
- `{r}_head_yaw_rel`: head yaw relative to the chest, signed (+ = toward the dancer's left).
- `features.py` bumps its `VERSION` and schema, `features` is re-run on larissa_kadu (no model involved), and
  tests check the signs on mirrored synthetic poses.

**The baseline arm (no TMR)** uses the interpretable per-dancer features plus `r` and `h`, classified with
prototypes or MiniRocket as in MOVES_RESEARCH Stage B1.

### 5.2 Classifier ladder (per silo, per move, by labelled instance count)

| Instances of the move in the silo | Model | Allowed use |
|---|---|---|
| 0 | Zero-shot text prior only (if G7) | Annotator suggestions only, never a viewer caption |
| 1-4 | Nearest centroid (cosine). Its softmax temperature is fitted inside the folds. | Annotator suggestions only |
| ≥ 5, from ≥ 2 dances | Multinomial logistic regression: L2 penalty, balanced class weights (scikit-learn 1.9.0, already installed). `C` = 0.1 until B2, then from `{0.01, 0.03, 0.1, 0.3, 1}` by nested LODO.<br>If it overfits: PCA to 32-64 dimensions per role, fitted inside each fold. | Annotator pre-fill and the review queue. In the viewer: family level or "uncalibrated", never the move's name with a number. |
| ≥ 20 instances, and ≥ 10 blind calibration segments in which the move was predicted, with no clear per-move overconfidence (its 90 % interval of accuracy minus mean p does not lie wholly below −0.10) | the same model | **Captionable by name with a number** in the viewer (section 6.5) |
| Below the pre-fill bar | Pooled into an `other` class with **family back-off**. The taxonomy families, with their move counts: hold 33, open 27, step 23, turn 20, head 16, wave 4, rest 1, none 2. | Caption at family level, or "uncertain" |

The ≥ 20 bar follows MOVES_RESEARCH (about 20 examples per move before it enters a reported classifier, as in
CoMPAS3D). Class sets and the `other` pooling are decided inside each fold, from that fold's training dances.

**Two further tools:**
- **kNN (k = 3, cosine, distance-weighted)** is for retrieval only ("looks like 01:23 in dance X"). It never
  supplies probabilities, and it is hidden during blind narration.
- **Text-prior blend (optional, *proposal*):** `logit_c = fewshot_c + a_c · zeroshot_c`, with
  `a_c = 2 / (2 + n_c)`. This is in the spirit of Tip-Adapter and LP++.

**Window purity.** `dataset.py` already records each window's `purity` (the share of its labelled eighths that
carry the majority move). Windows with purity below 0.6 are dropped from training and from temperature fitting,
and the purity distribution is reported. Long windows straddle moves, and fitting a temperature on mixed windows
would make the probabilities too soft.

### 5.3 Interface changes in `moves/model.py` and `moves/dataset.py`

**Classifier input.**
- `MoveClassifier.predict_proba(batch)` takes the same dict that `fit` already takes:
  `{"X": (n,16,F,4), "E": (n,2,256), "E_valid": (n,2), "R": (n,d_r), "H": (n,d_h)}`.
- A bare ndarray still means `{"X": X}`, so existing tests keep working.
- Both check the approvals ledger for non-synthetic inputs (section 2.1).

**`ModelConfig` gains:**
- `embeddings: "none" | "tmr_soma"`;
- `level`;
- `decode: "argmax" | "graph_hmm"` (`graph_hmm` from B1 on);
- `caption_threshold = 0.5`.

**`dataset.build_dataset(level=..., embeddings="tmr_soma")`:**
- It refuses takes of mixed levels (section 7.3).
- It looks up each window's embedding by `start_eighth`.
- It adds `E`, `E_valid`, `level`, `category`, `couple`, `song`, `song_group`, `blind`, `exposed_to_model` and
  `label_confidence` (n,) to `windows.npz`.
- The manifest records the embedding SHA per take.
- Auto moves the user later confirms count as training data, but they carry `exposed_to_model = true` and never
  serve as calibration or evaluation targets (section 6.2).

**Model files (no pickles, no joblib)** live in `work/moves_silos/<level>/models/<name>@<version>/`:
- `model.npz`: per fold and for the final refit: coefficients, intercepts, classes, CL2N means, PCA;
- `classes.json`, `calibration.json`;
- `manifest.json`: `{level, trained_on: [{take, sha}], folds, embeddings: {model, revision, route}, config,
  metrics, approval}`. It stays in `work/`; nothing in it is exported.

**`propose_labels`:** windows → `predict_proba` → `window_votes` → **decode (5.4)** → segments with the
section 6.6 keys → `merge_into_gaps`. `merge_into_gaps` is unchanged: auto output never overwrites human,
narration or earlier auto labels.
- **Clipped pieces.** `merge_into_gaps` copies a segment's whole dict into each piece it clips into a gap. A new
  post-step marks every such piece `clipped: true`, and a piece shorter than 75 % of its decoded segment gets
  `confidence: null`, because the segment's calibrated number does not describe the shorter piece.
- **Cross-fit.** A dance that is in the silo's training data is decoded only by the fold model that excluded it
  (section 6.3).

**`model.PLAN`** is updated to name TMR-SOMA, CPU fp32 and this file.

### 5.4 Graph-constrained decoding: `dancecap/dancecap/moves/decode.py`

`path.py` today only resolves node instances of moves that are *already labelled*. It has no emissions and no
posteriors. The new decoder reuses its transition costs.

**Time step:** one half-measure (4 eighths). That matches the window hop and the half-measure snapping of
`segments_from_votes`.

**Emissions:**
1. Average the per-eighth `window_votes` over each half-measure.
2. Apply the emission temperature `T` (6.3).
3. Convert to a scaled likelihood `(p(c|x) / p_train(c))^λ`: the hybrid-HMM rule of Bourlard and Morgan.
4. `λ` = 0.5 until B2; then chosen from `{0.3, 0.5, 0.7, 1.0}` by nested LODO.

**States:**
- one per graph node instance of each move above the pre-fill bar (`graph.instances(move)`);
- one `other:<family>` state per family;
- one `_none` state for non-move time (lesson talk, pause, solo demonstration, walking, intros and bows). It has
  an explicit `_none` class, trained on `exclude` spans and non-dance stretches from **every** category, so it is
  not simply a "this is a lesson" detector.

**Transitions:**
1. Start from `path._transition` costs, with the same `PathParams`:

   | Parameter | Value |
   |---|---|
   | no link | 10 |
   | reverse link | 0.5 |
   | via, per blank junction | 0.1 |
   | distance | 0.05 per metre |

   Weight = `exp(−cost / τ)`, with τ = 1 until B2, then by nested LODO.
2. Smooth with the observed link counts (Dirichlet, β = 2). The counts come only from the fold's training
   dances (narration-only dances included), never from the held-out song group.
3. Reserve a total mass ε for unlinked moves: 0.02, or 0.05 for Jack and Jill. These transitions become
   `new_link` candidates, drawn dashed in the viewer.
4. Self-loops: geometric, from the fold's median move duration in half-measures, capped at 16. Upgrade to an
   explicit-duration HSMM only if boundary F1 is poor.
5. Gaps over `max_gap_s` (1 s), invalid stretches and no-grid stretches break the chain, as in `path.py`.

**Algorithms:**
- **Viterbi** gives the segment sequence and node instances. Each segment's `node` is written into
  `labels.json`, so `path.json` pins it unchanged.
- **Forward-backward** gives per-half-measure marginals in `work/<take>/moves/posteriors.npz`:
  `t0`, `t1`, `classes`, `P` (H, C), plus state marginals.
- **Forward-filtering backward-sampling** draws 200 paths, used for the fingerprint spread (7.1).

**Boundaries.** Optionally, each boundary may move by up to ±1 beat to the strongest Foote-novelty peak from
`proposals.py`. Narration also snaps to those peaks, so boundary scores are reported with this snapping both on
and off (section 8.2).

**Precedent.** In a ballroom study a figure-transition prior raised accuracy from 85.95 % to 92.31 %
(MOVES_RESEARCH §3.4, not re-checked). Criterion A2 tests whether the decoder pays off here.

### 5.5 What comes out per segment

| Output | Definition |
|---|---|
| move, node | from the Viterbi path |
| `p_raw` | the mean forward-backward marginal of the chosen move over the segment's half-measures. If `p_raw` saturates (more than half of the segments above 0.99), the map is fitted on a less saturated score instead: the mean emission posterior. |
| `viterbi_agrees` | false when the Viterbi move is not the move with the highest mean marginal over the segment; the caption is then "uncertain" |
| `confidence` | the **calibrated** top-label probability (6.3), a point estimate; null unless the calibration status is `calibrated` for this dance's category and couple type |
| `p_shown` | the cautious number the viewer shows (6.5); null when `confidence` is null |
| `alternatives` | the top 3 moves by mean marginal, the chosen move first. Only the first carries a calibrated p. The others carry raw marginals rescaled to fill at most `1 − confidence`, marked `"calibrated": false`, and are shown as ranked names without numbers. |
| family | the chosen move's family. Its p is present only once a separate family-level map is fitted and passes the gate (6.3, step 6). |
| conformal set | when calibrated (6.3) |
| `uncertain` | true when `p_shown` < 0.5 (`caption_policy.threshold`), or when `viterbi_agrees` is false |

**Spans decoded as `other:<family>` or `_none`** are not written as segments, because `labels.json` requires a
graph move. They become **caption events** in `captions.json` (6.8) carrying their top two candidate names.

---

## 6. Confidence for captions

### 6.1 What the number means

**Definition.** The percentage on an `auto` caption answers: *if you look at the screen at a random moment
while this caption is showing, how likely is it to name the move the user would label there?* This is top-label
calibration (Gupta and Ramdas, arXiv 2107.08353) weighted by time on screen: among all caption-time shown at
80 %, about 80 % is correct.

**"Correct" means:** at that moment, the human-confirmed label of a blind dance names the same move. The node
instance does not have to match. A segment that is right for half its length counts as half right, not as fully
right. The segment-majority score (right if the label covering most of the segment matches) is still reported,
as a secondary number.

**Only one number is ever calibrated:** the shown move's. Second candidates, family sums and the next-move glow
are ranks, not probabilities, until they get their own maps.

### 6.2 Calibration data: blind dances

Labels shaped by the model cannot check the model. Pre-filled segments invite keeping them; a model-ranked review
queue chooses what gets confirmed; model-driven dance selection chooses what gets narrated. So calibration uses
**blind dances** only.
- **Choice.** Every dance narrated before the classifier exists (before B1 and G9) is blind by default: there is
  no model output to see. After that, blind dances are picked **at random**, stratified by category, never by
  active learning (proposal: every third new dance).
- **Narration.** No pre-fill, no review queue, no similar-windows panel, no model text hints. Boundaries snap only
  to beats and taps, not to `proposals.py` peaks.
- **Confirmation.** The user confirms the whole dance end to end. Only fully confirmed blind dances count toward
  E1.
- **Marking.** Every segment records `exposed_to_model` (whether the annotator saw model output for it), and every
  boundary records `boundary_source` (`tap`, `drag`, `beat_snap`, `proposal_snap`). Exposed segments never serve as
  calibration targets.
- Blind dances still serve as training data for the other folds; their labels are simply clean.

This answers the old question of whether calibration targets should include narration: no. Narration carries lag
and speech-to-text noise, so it is confirmed first (open question 15 asks whether the blind workload is
acceptable).

### 6.3 Calibration method (per silo, nested and cross-fitted)

1. **Outer loop.** Each blind dance is held out together with its song group. Everything below is fitted on the
   remaining dances only: PCA, CL2N means, class set and `other` pooling, duration and transition priors,
   hyperparameters (from B2), the temperature and the maps. Fold-level statistics live under `eval/`, the final
   ones under `stats/` (7.2).
2. **Emission temperature.** One parameter `T`, fitted by negative log-likelihood on the inner folds' out-of-fold
   window logits (Guo et al. 2017, arXiv 1706.04599), windows with purity ≥ 0.6 only. Small numpy code. If
   scikit-learn's `CalibratedClassifierCV` is ever used instead, it gets explicit `LeaveOneGroupOut` splits with
   groups = song group: its default splits windows at random, which leaks overlapping windows.
3. **Segment map.** A top-label Platt map with 2 parameters (`a`, `b` on `logit(p_raw)`), fitted on decoded
   segments of the inner folds' blind dances. The target is each segment's correct fraction of time, and each
   segment is weighted by its duration (section 6.1). Each dance counts equally: at most 5 instances of a move
   per dance enter the fit, or weights of 1/√count.
4. **Deployment.** The fold models are kept.
   - A dance in the silo's training data is captioned only by the fold model that excluded it, so its calibration
     is exactly what was measured.
   - A new dance is decoded by every fold model. The caption shows the move most fold models chose, with the mean
     of their calibrated p for it, counting 0 for fold models that chose another move. That should err low: a
     linear pool of calibrated forecasts tends to be under-confident (Ranjan and Gneiting 2010).
   - The outer loop scores this same pooling, so the gate checks what the viewer will actually show rather than
     relying on that argument.
5. **Isotonic regression or histogram binning** only from 1,000 blind segments. scikit-learn's guidance: isotonic
   regression beats sigmoid only above about 1,000 samples.
6. **Family map.** A separate top-label Platt map on family correctness. Until it is fitted and passes the gate,
   a family carries no p.
7. **Narration map.** The narration confidence (`match × timing score`, `annotate/pipeline.py`) is calibrated
   against the user's own keep and relabel decisions (logged by the annotation tool), per silo, with Platt or
   isotonic and the same gate. It needs no 3D, so it can be the first honest number the user sees: proposal, at
   least 200 reviewed narration segments from at least 5 dances.
8. **Conformal sets.**
   - Split conformal at α = 0.2 with the APS score, from the cross-fitted non-conformity scores.
   - Rare moves are grouped by family (clustered conformal, Ding et al. 2023).
   - Coverage is approximate twice over: windows within one dance are not exchangeable (Dunn, Wasserman and
     Ramdas, JASA 2023), and scores come from fold models, which makes this cross-conformal. Its worst-case
     guarantee is only about 1 − 2α.

**How it is reported** (on cross-fitted predictions only):
- the debiased calibration error (Kumar et al. 2019, arXiv 1909.10155; a root-mean-square measure, stricter than
  the usual ECE) with a 90 % dance-cluster bootstrap interval; adaptive ECE as well;
- class-wise ECE, negative log-likelihood, Brier score;
- per category and per couple type, each with its interval;
- reliability diagram, risk-coverage curve.

Binned ECE underestimates the true error (Kumar et al. 2019; Nixon et al. 2019), so it is never the only number.

### 6.4 The honesty gate and the calibration status

**The gate (criterion A3).** All of these, on cross-fitted predictions from blind dances:
- **Bins.** Equal-mass bins with at least 50 segments each, between 2 and 10 bins. The bin count follows the
  data; there is no jump at a fixed segment count.
- **Overall.** The upper end of the 90 % dance-cluster bootstrap interval of the debiased calibration error is at
  most 0.10.
- **No bin clearly overconfident.** In every bin, the upper end of the 90 % interval of (accuracy − mean p) is at
  least −0.10.

*Note on a review suggestion:* the method review proposed a two-sided per-bin test (the whole interval within
±0.10). With 50 segments per bin the 90 % interval alone is about ±0.11 wide, so even a perfectly calibrated model
would rarely pass before about 150 segments per bin. The one-sided test above catches clear overconfidence, and the
cautious display (6.5) covers the rest.

**Why not a point estimate.** At the old minimum sizes, the old gate (point ECE ≤ 0.08, no bin off by more than
0.15) was mostly noise. Synthetic simulation (`a3_gate2.py` in the review scratchpad, re-created as an M7 test;
shown p averages about 0.71;
pass rates out of 150 runs each, so about ±4 points; "dances differ" means each dance's accuracy is shifted by a
random logit offset with sd 0.5, while the model stays calibrated on average):

| Blind segments / dances | Perfectly calibrated, old → new | Calibrated on average, dances differ, old → new | Overconfident by 0.10, old → new | Overconfident by 0.15, old → new |
|---|---|---|---|---|
| 100 / 5 | 44 % → 39 % | 37 % → 21 % | 9 % → 9 % | 1 % → 3 % |
| 200 / 5 | 87 % → 61 % | 75 % → 22 % | 13 % → 1 % | 1 % → 0 % |
| 300 / 6 | 65 % → 63 % | 35 % → 19 % | 2 % → 1 % | 0 % → 0 % |
| 300 / 10 | 54 % → 53 % | 51 % → 26 % | 1 % → 0 % | 1 % → 0 % |
| 600 / 15 | 94 % → 91 % | 81 % → 56 % | 5 % → 0 % | 0 % → 0 % |
| 1,000 / 20 | 99 % → 100 % | 93 % → 91 % | 3 % → 1 % | 0 % → 0 % |

(The old gate's dip at 300 comes from its switch from 5 to 10 bins.)

**How to read it:**
- The new gate almost never lets an overconfident model through, from 200 segments on.
- It passes a calibrated model less often at small sizes, especially when dances differ from each other. That is
  the honest outcome: a few dances cannot show that the numbers will hold for the next dance.
- At the E1 minimum (300 blind segments) even a well-calibrated model passes only about half the time. About
  600 blind segments make passing likely.

**Data needed (per silo):**

| Step | Minimum |
|---|---|
| temperature `T` | ≥ 100 segments inside the folds |
| segment map, and the gate | ≥ 300 fully confirmed blind segments from ≥ 5 blind dances, in a silo of ≥ 10 labelled dances (E1) |
| a category's own status | ≥ 3 blind dances of that category (*proposal*) |
| a couple not in training | leave-one-couple-out calibration, so ≥ 3 couples (E2) |
| family map | ≥ 300 blind segments |
| narration map | ≥ 200 reviewed narration segments from ≥ 5 dances (*proposal*) |
| isotonic / histogram binning | ≥ 1,000 blind segments |
| conformal sets at α = 0.2 | ≥ 100 segments; aim for 300 |
| conformal sets at α = 0.1 | ≥ 300 segments |

**Why these conformal minimums.** Split-conformal coverage given the calibration set follows a Beta distribution.
At α = 0.2, its 5-95 % range is 0.71-0.89 at n = 50, 0.73-0.86 at n = 100, and 0.76-0.84 at n = 300.

**The status** is kept per **(silo, category, couple seen or unseen)**:
- `calibrated` only when the segment map is fitted on the minimums above, the gate passes, the category has its
  own blind dances, and, for a couple not in training, the LOCO version also passes;
- otherwise `uncalibrated`, with a `reason`: `insufficient_data`, `gate_failed`, `new_category` or
  `new_couple`.

The gate's result is written into `calibration.json` (6.7). `validate_labels` and `captions` refuse a number
unless the status for that dance's category and couple type is `calibrated` and the recorded gate passed.

### 6.5 Caption rules per provenance

The rules implement VIEWER_SPEC §3.12 and §10, with a default threshold of 0.5.

**The shown number is cautious by construction.** `p_shown` is the lower end of the one-sided 90 %
dance-bootstrap interval of accuracy in the reliability bin of the segment's calibrated p, never above that p,
rounded **down** to 10 % steps. The highest number possible is the top bin's lower bound. The viewer's legend
reads it as "right at least about this often". (The method review also suggested three verbal bands; a number is
kept because the user asked for a confidence score, open question 14.)

| Provenance / state | Name line | Confidence shown | Below threshold |
|---|---|---|---|
| `human` | Portuguese name, English alias beneath | "labelled" (no bar) | never |
| `narration`, narration map not yet calibrated | move | "narrated", no number and no bar | below 0.5 match: "uncertain" plus the **names** of the two best `match.candidates` |
| `narration`, narration map calibrated | move | "narrated · 80 %", from the narration map, cautious as above, outlined bar | `p_shown` < 0.5: "uncertain" plus the two candidate names |
| `auto`, `calibrated`, move captionable by name | move | `p_shown`, e.g. "70 %", solid bar; never 100 % | see the order below |
| `auto`, `calibrated`, move not captionable by name | family name if the family map passed, otherwise the move with "uncalibrated" | family `p_shown`, or none | "uncertain" plus two names |
| `auto`, `uncalibrated` | move | "uncalibrated" plus the reason ("new couple", "lesson", ...), no number, no bar; alternatives as ranked names | no uncertain state; whether to show these at all is open question 10 |
| `placeholder` | move | "placeholder" | never |
| `other:<family>` / `_none` span (no segment) | family name or nothing | "uncertain" plus the top two names (auto), or the caption is hidden (`_none`) | |

**Order of checks for a calibrated auto caption**, first match wins:
1. Nearest-prototype cosine below the training 5th percentile (computed inside the folds) → "unknown move", no
   number.
2. `viterbi_agrees` is false → "uncertain" plus the top two names.
3. `p_shown` < 0.5 → "uncertain" plus the top two names. If the family map passed and the family's `p_shown` is
   ≥ 0.8, show the family name (e.g. "head movement") plus the two names.
4. Conformal set of size 2-3 → "A or B" (or "A, B or C"), no number. Size > 3 → "uncertain".
5. Otherwise: the move with `p_shown`.

**Renders for sharing** (VIEWER_SPEC §8 MP4s): auto captions appear only when `calibrated`, always with the
"automatic - unreviewed" badge. Dancer names and levels appear only as their consent allows (7.1).

**The next-move glow** in the graph inset is a **ranking**, not a probability; the legend says so.

Captions change only at event boundaries, with a short cross-fade. The next move may be previewed in the last
beat (VIEWER_SPEC §3.12).

### 6.6 Label contract extension (`labels.json` v1, additive, no version bump)

`Segment.extra` and `Labels.extra` already preserve unknown keys (`labels.py`, `to_dict` / `from_dict`). The
export no longer copies them blindly: it applies the key-level allow-list of section 2.2. The version stays 1.

**An `auto` segment** (in `work/`):
```json
{"id": "seg-...", "t0": 61.2, "t1": 64.3, "move": "<slug>", "node": "<node uuid>", "tier": "couple",
 "provenance": "auto",
 "confidence": 0.83,
 "p_shown": 0.7,
 "p_raw": 0.91,
 "viterbi_agrees": true,
 "uncertain": false,
 "clipped": false,
 "alternatives": [{"move": "<slug>", "p": 0.83, "calibrated": true, "node": "<uuid>"},
                  {"move": "<slug2>", "p": 0.11, "calibrated": false},
                  {"move": "<slug3>", "p": 0.04, "calibrated": false}],
 "family": {"name": "head", "p": null},
 "conformal_set": ["<slug>"],
 "annotator": "model:tmr-lr@1",
 "level": "professional",
 "calibration": {"id": "professional/tmr-lr@1/cal-2026-11-02", "method": "temperature+platt",
                 "status": "calibrated", "category": "demo", "couple": "seen",
                 "gate": {"passed": true, "ce_upper": 0.08}, "n_blind_segments": 320, "n_blind_dances": 6,
                 "targets": "blind_human"},
 "decode": {"method": "graph_hmm", "posterior": "forward_backward", "lambda": 0.5}}
```

**What the keys mean:**
- **`confidence`** is the calibrated top-label p, or `null` unless the status for this dance's category and
  couple type is `calibrated`. It replaces today's mean max-vote in `segments_from_votes`, which is uncalibrated.
- **`alternatives`**: the first entry is the shown move with its calibrated p. The others are raw marginals
  rescaled so that all entries sum to at most 1: `p_i = p_raw_i × (1 − confidence) / (1 − p_raw_top)`.
- **`conformal_set`** is present only when calibrated.

**New keys on human and narration segments:** `exposed_to_model` (bool) and, per boundary, `boundary_source`.
Confirming an auto segment sets provenance `human` and keeps `exposed_to_model: true`.

**New top-level keys:**
- `dance`: a metadata snapshot (7.1);
- `models`: `{id: {name, version, level}}`. The training list stays in the model manifest under `work/`;
- `exclude`: `[{t0, t1, reason: talk|pause|solo_demo|walk|other}]` (7.5).

**`validate_labels` gains these rules:**
- For `auto` segments:
  - `alternatives` holds 1 to 3 entries whose moves exist in the graph;
  - the first entry is `move`, and only the first may have `calibrated: true`;
  - the p values sum to at most 1 + 1e-6;
  - `calibration.status` is `calibrated` or `uncalibrated`;
  - `confidence` and `p_shown` are null unless the status is `calibrated` and `gate.passed` is true;
  - `family.p` is null unless a family map is recorded;
  - `level` equals `dance.level`.
- `narration` segments must not carry `alternatives` with p. Their candidates stay under `match`.
- No `auto` segment overlaps an `exclude` span.

### 6.7 Calibration record

`work/moves_silos/<level>/models/<id>/calibration.json`:
```
{version, id, level, model, created, method, targets: "blind_human", cross_fitted: true,
 status: {"<category>/<seen|unseen>": {status, reason}},
 params: {T, a, b, family: {a, b} | null, narration: {...} | null},
 fitted_on: {takes, blind_takes, n_blind_segments, n_dances, categories: {demo, lesson, jack_and_jill}, couples},
 gate: {passed, ce_debiased, ce_upper, bins, bin_checks, simulation_ref},
 metrics: {ece_adaptive, ece_ci, classwise_ece, nll, brier, confident_wrong_share, risk_coverage,
           segment_majority_accuracy},
 bins: [{lo, hi, n, mean_p, accuracy, accuracy_lower90}],
 conformal: {alpha, qhat, coverage_per_dance, mean_set_size, note: "cross-conformal, approximate"},
 per_category: {...}, transfer_from: null}
```

### 6.8 Caption track: `captions.json` (new, derived)

**Where it lives:**
- `python -m dancecap.moves captions <take>` writes `work/<take>/moves/captions.json` in reference time.
- `moves export` converts it to capture time in `StreamingAssets/<capture>/moves/captions.json`, through the
  key-level allow-list.

**Why a separate track.** Python builds every string. The rules in 6.5 are then a single tested function, and
the viewer only renders them.

```
{version: 1, take, time_base, provenance,
 caption_policy: {threshold: 0.5, round_down: 0.1, shown: "lower90", version: 2},
 dance: {title, conditions},
 events: [{t0, t1, segment: <id or null>, kind: labelled|narration|auto|placeholder|uncertain|unknown|none,
           move, node, name_pt, name_en,
           shown: "labelled" | "70 %" | "narrated" | "narrated · 80 %" | "uncalibrated: new couple" | "A or B"
                  | "unknown move" | "placeholder" | "uncertain",
           p_shown: <0..1 or null>, bar: <0..1 or null>, bar_style: solid|outline|null,
           uncertain: bool,
           candidates: [{move, name_pt, name_en, rank}],     names only, top two when uncertain
           next: [{move, node, rank}]}]}                      next moves for the inset glow, ranked
```

- `name_pt` and `name_en` come from the graph's `display_name` and `name_en`.
- `next` lists the outgoing graph links of the current node, ranked by the posterior of the following
  half-measure when posteriors exist; otherwise unranked.

### 6.9 How Unity consumes it (`head-movement/Assets/DanceGraph/`)

Today `DanceGraphData.ReadLabels` parses `Confidence` but nothing shows it, and the badge depends only on
provenance.

**`DanceGraphData`:**
- adds a `captions.json` parser (Newtonsoft `JObject`; unknown keys are tolerated);
- falls back to the labels-based caption when the file is missing.

**`DanceHud`:**
- draws the Portuguese name, with the English alias smaller beneath;
- shows the `shown` text, plus a bar when `bar` is not null (solid for auto, outlined for narration);
- in the uncertain state, shows "uncertain" and the two candidate names;
- keeps the provenance badge ("AUTOMATIC - unreviewed", "PLACEHOLDER");
- the legend explains that a number means "right at least about this often".

**`DanceGraphLayer` (the graph inset):**
- highlights the current node: path steps are joined to events by `Step.Segment` == `event.segment`;
- keeps the fading trail;
- makes `next` candidates glow by rank (brightest first), or uniformly when unranked;
- draws new links dashed.

**Never** loads embeddings, prototypes or model files.

**The reaction title** (opening card, closing fingerprint shot, library card) comes from the capture.json v4
`dance` and `title` keys (VIEWER_SPEC §12; written by `export_unity`, section 7.1).

**Acceptance** is checked with the `hm_*` CLI playtests of VIEWER_SPEC §14:
- the caption changes within 1 frame of each boundary;
- the shown confidence follows the provenance rule;
- below the threshold the caption reads "uncertain" with two candidate names;
- the reaction title and the library shelf match the dance's level.

These run on placeholder labels and on a synthetic auto timeline with known p, so they need no dancer
inference. Their fixtures use synthetic dancer and song names.

---

## 7. Data silos and categories

### 7.1 Take metadata, consent and the reaction title

**Single source:** `dancecap/takes/<take>.toml`. The `[dance]` table already exists in
`takes/larissa_kadu.toml`. Today only `song.py` reads it, and only `song_file`.

```toml
[dance]
level = "professional"            # set directly, or derived from the dancers' levels when both are given (7.3)
category = "demo"                 # demo | lesson | jack_and_jill          (required for any analysis)
jj_division = ""                  # NEW, J&J only: the event's division name, mapped to a level (7.3)
song = "Ficar Sem Você (feat. Urias)"
artist = "Davi Sabbag"
song_file = "../../dances/Larissa-Kadu-demo/Ficar Sem Você.mp3"
song_display = "Ficar Sem Você"   # NEW, optional: the song as shown in the reaction title
date = "YYYY-MM-DD"               # NEW, optional
venue = "..."                     # NEW, optional

[dancers.lead]
name = "Kadu"                     # exists today
id = "kadu"                       # NEW, optional: stable dancer id when two dancers share a name
level = "professional"            # NEW: level belongs to the person

[consent]                         # NEW (G0): what each dancer agreed to, as the user reported it
lead   = {capture = true, fit_3d = true, ml_analysis = "?", cloud_processing = "?", name_shown = "?",
          level_shown = "?", publication = "?", public_repo = false, recorded = "YYYY-MM-DD", approval = "<ledger id>"}
follow = {...}
source_rights = "?"               # who owns the footage (matters for third-party single videos)
bystanders = "?"                  # other people in view: camera operators, audience, students, J&J floor
```

What is on record today: Larissa and Kadu consented to photoreal digital doubles. The other scopes are
unknown (open question 11).

**Loader:** `dancecap/dancecap/dance_meta.py`, with `DanceMeta.load(take)`.
- It reads only `[dance]`, `[dancers.*]` and `[consent]`, and never requires `source_dir` or `alignment`.
  (`Take.load` does require them.)
- It validates the enums. A missing `level` or `category` gives `None` plus a warning, and the dance is left
  out of every analysis (7.3).
- **Consent checks.** `DanceMeta.require(scope)` refuses when the scope is missing or false. G5 conversion and
  embedding need `ml_analysis`; G6 pod jobs need `cloud_processing`; `moves export` and `captions` need
  `name_shown` and `level_shown` for the title, and `publication` for renders.
- **Anonymised title.** When `name_shown` is false, the title reads "Leader A + Follower B × Song", and
  `couple_id` stays internal.

**What it derives:**

| Field | Rule | Larissa/Kadu |
|---|---|---|
| `title` | `"{leader} + {follower} × {song_display}"` | "Kadu + Larissa × Ficar Sem Você" |
| `title_ascii` (file names) | ASCII transliteration, × → x | "Kadu + Larissa x Ficar Sem Voce" |
| `conditions` (above the arrow) | `"{Level} · {Category}"`; categories display as Demo, Lesson, Jack and Jill | "Professional · Demo" |
| `couple_id` | `slug(leader id or name) + "__" + slug(follower id or name)` | `kadu__larissa` |
| `song_group` | slug of the song title without "(feat. ...)" | `ficar_sem_voce` |
| song credit | `song_display` when given; otherwise the song with a trailing "(feat. ...)" removed and the featured artist moved into the credit | "Davi Sabbag feat. Urias" |

The default song rule is open question 8.

**The reaction.**
- *Reactants:* the leader and the follower.
- *Driver:* the song.
- *Conditions:* level and category.
- *Product:* the **dance fingerprint** (`fingerprint.json`, VIEWER_SPEC §3.11).
- The text reads: **Kadu + Larissa × Ficar Sem Você → [fingerprint]**.

`fingerprint.json` gains a `reaction` block:
`{leader, follower, song: {title, display, artist}, level, category, title, conditions}`. The SHA of the
take's TOML is added to `sources`, so a metadata change marks the fingerprint stale.

**The fingerprint from automatic labels:**
- With a human path, the fingerprint is computed as today.
- With auto labels, it is the **expected** fingerprint: dwell = sum of posterior marginals × duration.
- **Prior bias.** The decoder's learned priors pull every automatic fingerprint toward the typical dance of the
  silo. So fingerprints used for comparison are decoded with priors that exclude the dance and its song group,
  and the report shows how far the fingerprint moves when decoded with the graph-only prior instead.
- **Spread, not intervals.** The 200 sampled paths give each node and link a "model spread". It is called an
  80 % interval only after its coverage has been checked on blind human-labelled dances.
- `fingerprint.json` records the shares of human, narration and auto time.
- Fingerprint comparisons (JSD) are made within one silo by default, show the auto share next to every
  distance, and by default compare only dances with similar provenance mixes.

**Optional motion signature.** A dance-level vector (the mean of the CL2N embeddings) for "dances like this
one" retrieval inside a silo. It stays in `work/` and is never exported.

**Where the metadata snapshot goes:** a `dance` snapshot of the metadata (`level`, `category`, `couple_id`,
`title`) is copied into:
- `features_schema.json`, `soma/<route>/report.json` and `tmr_soma.json`;
- `labels.json` (top level), `captions.json` and `fingerprint.json` (as `reaction`);
- the dataset `windows.npz` and manifest, per window and per take;
- capture.json v4, as `dance` + `title` written by `export_unity`, as VIEWER_SPEC §12 specifies.

Consent is never exported; only its effect (an anonymised title, hidden level) is.

**Dances with no multi-camera capture** (narrated from one video) cannot use `takes/<slug>.toml`, because
`Take.load` needs sources and alignment.
- Their metadata goes in a new tracked folder: `dancecap/takes/meta/<slug>.toml`, holding only `[dance]`,
  `[dancers]` and `[consent]`, including `source_rights` (third-party footage).
- They get no embeddings.
- They feed the silo's label statistics: move durations and transition counts for the decoder priors, always
  per fold.
- Their narration can be calibrated (6.3, step 7).

### 7.2 Storage per silo

```
work/moves_silos/<level>/
  dataset/      windows.npz, eighths.parquet, manifest.json (level, takes, categories, couples, songs, song groups,
                blind takes, embedding SHA per take, missing_metadata, missing_consent, excluded spans)
  index/        tmr_soma_index.npz: every window of the silo (emb, take, t0, t1, label or -1), for kNN retrieval
  stats/        final refit: cl2n_mean.npy, pca.npz, duration_prior.json, transition_counts.json
  models/<name>@<version>/   model.npz (fold models + final), classes.json, calibration.json, manifest.json
  eval/<model id>/           report.json, reliability.png, per_category.json, folds/ (fold-level stats)
work/moves_silos/_mixed/     dances whose dancers' levels differ, held until the user assigns them (7.3)
work/moves_silos/_cross/     only with --cross-level (experiments; never merged into labels.json)
work/moves_shared/           text prompts' embeddings (optional): keyed by graph SHA and prompt-file SHA
```

`work/moves_dataset/` (today's default) is retired.
- `moves dataset` writes into the silo folder.
- It requires `--level` once a second level exists.
- Until then it infers the level from the takes and refuses a mix.

### 7.3 No cross-silo leakage

1. **One level per artefact.** Every artefact is built from **one level** and records `level` in its
   manifest: datasets, normalisation stats, CL2N means, PCA, prototypes, the kNN index, models, temperature
   and Platt parameters, conformal quantiles, duration and transition priors, evaluation reports.
2. **Shared across silos:** only the frozen TMR weights, the zouk graph and taxonomy, the text prompts and their
   embeddings, and code.
3. **Level checks.**
   - `discover_takes` filters by level.
   - A take without a level is excluded and listed under `manifest.missing_metadata`.
   - Applying a model to a take of another level is refused (`propose`, `merge`, `decode`, `captions`).
4. **Cross-level work** needs `--cross-level`. It writes to `_cross/` and is never merged into a take's labels.
5. **Transfer is a gate (G10).** Professional prototypes may act as a prior for another silo while that silo has
   fewer than 5 dances, but only after the user approves it. Such a model records `transfer_from: professional`
   in its manifest and calibration, is always recalibrated and evaluated on the target silo's own blind dances,
   and its labels enter a take's `labels.json` only after a further approval. Lower-level data is never pooled
   into Professional models.
6. **Level belongs to people.** Each dancer has a level. The dance's level is theirs when both agree. A mixed-level
   dance (a professional teacher with a student, a mixed J&J draw) is held in `_mixed/` and used by no silo until
   the user assigns it (open question 13).
7. **Jack and Jill divisions** map to levels by an explicit table the user confirms. *Proposal:* Newcomer and
   Novice → novice; Intermediate → intermediate; Advanced, All-Star, Champion and Invitational → professional.
8. The same dancer may appear in several silos. That is allowed; LOCO groups by dancer id within a silo.

**Why so strict:**
- In CoMPAS3D (arXiv 2507.19684), proficiency is highly recognisable: 84.63 % from both dancers.
- Professionals there perform 54.5 styling moves per performance against 5.1 for beginners.
- The class priors and the move vocabulary therefore differ across levels, and calibration would not transfer.
- The user's decision: never mix levels unless explicitly asked.

### 7.4 Evaluation per silo

- Every report (section 8) is per silo.
- `model status` evaluates the readiness gates per silo, and the calibration status per category and couple
  type. Intermediate and Novice report "no data" today.
- Category metrics (7.5) are reported inside each silo, with intervals.

### 7.5 How category is used

**General rules:**
- Category is metadata on every window, segment and model card.
- Blind dances are chosen stratified by category. (LODO folds hold out one song group at a time, so stratifying
  the folds themselves would mean nothing.)
- Every metric is reported per category, with its interval.
- The calibration status is per category (6.4). There are **no separate models per category** at this data size.
- Each dance counts equally in calibration and evaluation: at most 5 instances of a move per dance, or weights of
  1/√count, so a long drill lesson cannot dominate.

| Category | Caveat | Handling |
|---|---|---|
| **Lesson** | Talking, pauses and demonstrations. A teacher dancing solo breaks the two-role assumption. Drills repeat one move many times. Students appear. | - The narration tool marks `exclude` spans with `reason: talk / pause / solo_demo / walk / other`.<br>- The dataset drops them from move training; the `_none` class learns from them and from non-dance stretches of all categories.<br>- The drill cap above.<br>- Time-based windows where there is no music (4.1).<br>- Students need consent (G0), and a teacher-plus-student dance is usually mixed-level (7.3). |
| **Jack and Jill** | Improvised, with a random partner, on a crowded floor. Usually filmed as one competition video. | - If captured multi-camera: a **reserved external test set**, never used for training or calibration until J&J has its own blind dances; J&J playback reads "uncalibrated: Jack and Jill" until then.<br>- If single-video (the usual case): narration and decoder priors only (open question 9).<br>- New-link mass ε = 0.05.<br>- Expect more new links and noisier labels.<br>- Random partners and bystanders need consent (G0). |
| **Demo** | Usually rehearsed; the same choreography may be danced again to the same song. | LODO holds out whole song groups, so memorised sequences cannot leak through the transition priors. |

---

## 8. Evaluation protocol and acceptance criteria

### 8.1 Data gates (per silo)

These follow MOVES_RESEARCH §8 and `model.PLAN`.

| Gate | Needs | Unlocks |
|---|---|---|
| E0 | ≥ 3 labelled dances | LODO evaluation, reported as "seen couple only" until E2 |
| **B1** | about 10 narrated dances; moves with ≥ 5 instances | the classifier pre-fills gaps as `auto` labels for the narrator (with G9) |
| E1 | ≥ 10 labelled dances, including ≥ 5 fully confirmed blind dances with ≥ 300 segments | calibrated numbers in viewer captions, for categories with their own blind dances and for seen couples, once the gate passes |
| E1-N | ≥ 200 reviewed narration segments from ≥ 5 dances | calibrated numbers on narration captions |
| E2 | ≥ 3 couples | leave-one-couple-out evaluation and calibration; numbers for couples not in training |
| **B2** | about 25-30 dances; ≥ 20 instances for 10-15 moves | a reported classifier; hyperparameter tuning by nested LODO; the MS-TCN++ head (MOVES_RESEARCH §8) |
| C | about 60-100 dances | encoder fine-tuning and pretraining (outside this plan) |

### 8.2 Folds and metrics

**Folds:**
- Nested LODO first: the outer loop holds out one song group, and nothing fitted for it sees that group
  (section 6.3). LOCO once there are 3 couples.
- Never random windows: windows of one dance overlap.
- *Note on a review suggestion:* the method review proposed also excluding every dance that shares the held-out
  dance's couple. With one couple in the silo that would leave nothing to train on, so couple exclusion waits for
  LOCO, and LODO results are labelled "seen couple" until then.

**Window level:** top-1 and top-3 accuracy, and the window purity distribution.

**Decoded:**
- per-half-measure accuracy;
- macro-F1 over moves with support ≥ 5, with support reported, plus family-level macro-F1;
- segmental Edit score and F1@{10, 25, 50} (MS-TCN protocol);
- boundary F1 at ±0.5 beat (primary) and ±1 beat, scored **only on boundaries the user placed or moved** (`tap`,
  `drag`, or snapping off). Also reported: the share of label boundaries that coincide with `proposals.py` peaks,
  and the decoder's scores with Foote snapping off. Narration and the decoder both snap to those peaks, so
  agreement there proves little;
- graph-level: the share of predicted transitions on existing links, edge precision and recall against the
  human path, and fingerprint JSD against the human fingerprint.

**Calibration:** as listed in 6.3, plus conformal coverage and mean set size per held-out dance.

**Caption usefulness** (reported, not a gate):
- a risk-coverage curve;
- the **confident-wrong time share**: of the held-out blind dances' time that shows an automatic caption with a
  number, the share where the move is wrong.

**Uncertainty and ceiling:**
- Bootstrap resamples over dances, never over windows.
- The human ceiling: beat-wise Cohen's kappa on 5-10 % double-annotated time, plus narration-pass versus
  careful-pass agreement.

### 8.3 Ablations

These are **exploratory**: reported, never used to adopt anything except through A1 and A2.
- interpretable features only, TMR only, and both;
- with and without the graph decoder;
- with and without the text prior (only if G7);
- conversion route R1 versus R2;
- 16- versus 8-eighth windows, and the primary length re-chosen from the measured move durations once B1 data
  exists;
- optional: the follower-in-leader-frame channel.

Until B2, hyperparameters stay at frozen defaults (λ = 0.5, τ = 1, w_r = 1, C = 0.1, ε = 0.02), so that ten
dances are not spent on tuning.

### 8.4 Acceptance criteria

All numbers here are *proposals*, to be revised once data exists.

| ID | Criterion | Pass |
|---|---|---|
| A1 | TMR adoption (the one pre-registered confirmatory test) | Cross-fitted LODO macro-F1 on blind dances (moves with support ≥ 5), interpretable + TMR minus interpretable only: one-sided paired sign-flip test over per-dance differences, p < 0.05. Otherwise TMR leaves the classifier; it may stay for retrieval only. Before TMR feeds captions for couples not in training, the same test must pass under LOCO. |
| A2 | Graph decoder adoption | Decoded per-half-measure accuracy and Edit score improve, by the same kind of paired test |
| A3 | Numbers in captions (the honesty gate) | Section 6.4, per silo, category and couple type. It checks honesty only; usefulness is reported in 8.2. |
| A4 | "A or B" captions | Median conformal coverage per held-out dance ≥ 0.75 at α = 0.2 |
| A5 | Pre-fill usefulness (B1) | The narrator keeps the move of ≥ 50 % of pre-filled segments (logged by the annotation tool). This keep rate is anchored by the pre-fill itself, so it measures time saved, not accuracy; it is always reported next to blind accuracy. |
| A6 | Plumbing and guards | All unit tests of M1-M9 pass. `moves validate` is clean on the synthetic take. The public-repo guard test passes. No embedding, index, prototype or model file exists under `head-movement/`. |

### 8.5 Active-learning loop with the narration tool

This loop runs only after G9, and never on blind dances.

```
narrate (annotate) -> confirm / correct (provenance human) -> dataset --level L -> fit + calibrate (approved)
      ^                                                                                     |
      +--- review queue + pre-filled gaps <- propose (decode, captions) <-------------------+
```

**Which dance to narrate next (per silo, non-blind dances only):**
- Below 10 labelled dances: TypiClust-style (arXiv 2202.02794). Run k-means over the silo's window vectors
  (TMR plus interpretable features). Pick the unlabelled dance that covers the most *typical* clusters not
  yet labelled. When the budget is low, typical examples beat uncertain ones.
- From 10 dances: rank dances by mean decoded entropy plus the number of predicted rare moves (fewer than 5
  instances, p ≥ 0.3).
- Both stages: keep the category and couple mix balanced.
- Blind dances are set aside first, at random (6.2); every dance narrated before B1 is blind anyway.

**Inside a narrated (non-blind) dance:**
1. **Pre-fill.** Auto segments go into gaps only (`merge_into_gaps`), badged "automatic - unreviewed".
2. **Review queue** of 10-20 items, in this order:
   1. narration-model disagreement: p_model(narrated move) < 0.1, a likely speech-to-text or mapping error;
   2. low-margin boundaries (B-ACT, arXiv 2604.15173);
   3. conformal sets of size ≥ 3.
3. **Confirm** sets provenance `human` with `exposed_to_model: true`. Every keep, relabel and boundary move is
   logged, which gives A5, the narration map (6.3, step 7) and the narration-lag estimate.
4. **Similar windows** panel: kNN within the silo, with time links into other dances, to speed narration.

**Blind mode** in the annotation tool hides items 1, 2 and 4, turns off proposal snapping, and records
`exposed_to_model: false`.

**Retraining and recalibration** are manual commands, run after each batch of narrated dances and each
approved by the user.

---

## 9. Work breakdown

Effort is in focused agent-days of implementation. Runs on real dances happen later, when their gates open.

| M | Milestone | Deliverables | Tests | Effort | Approval needed | Depends on |
|---|---|---|---|---|---|---|
| P0a | 3D pilot (prerequisite, outside this plan) | larissa_kadu refit over 29.97-68.51 s with set c8; capture steps re-run over that stretch; measured time per frame, peak memory, local step runtimes | MAMMA's own QA | about $2-2.6 of pod time, plus local runs | G0 (cloud processing), G6 | — |
| P0b | Full-dance 3D (prerequisite, outside this plan) | the whole capture pipeline over each song, camera sets per availability span, chunked fits if P0a shows the need, `smplx/` for whole dances | MAMMA's own QA | per section 4.6 | G0, G6 or a local-overnight decision | P0a |
| M0 | Decisions | answers to section 10.2; G0-G3 | — | user, about 45 min | — | — |
| M1 | Metadata, consent, silos, signed head features | `dance_meta.py` with consent checks and anonymised titles; `takes/meta/`; per-dancer level, `_mixed` silo, J&J division table; `dataset --level` and the silo layout; level filter in `discover_takes`; `dance` + `exclude` in `labels.json`; `reaction` in `fingerprint.json`; capture.json v4 `dance` / `title`; `{r}_head_roll_side`, `{r}_head_yaw_rel` with a features `VERSION` bump and a re-run of `features` on larissa_kadu; `CONTRACTS.md` entries | title / conditions / couple_id / song_group derivation; enum validation; consent refusals; anonymised title; mixed-level dance held; dataset refuses mixed levels; take without level excluded; fingerprint stale after a TOML change; exclude spans dropped; head-feature signs flip on mirrored synthetic poses | 2 | none (the features re-run involves no model) | — |
| M1b | Governance plumbing and early blind narration | approvals ledger and `approvals list`; ledger checks inside the library functions; `synthetic = true` marker; key-level export allow-list; public-repo guard test; hook scripts for `head-movement`; annotate: `boundary_source` per boundary, `exposed_to_model`, and a `--blind` switch that turns off proposal snapping, so dances narrated from now on can count as blind | a gated function refuses an uncovered take; export drops the listed keys and paths; guard test fails on each listed extension, a large blob and a deny-listed name; `remote run` refuses an uncovered gated command; `--blind` snaps only to beats and taps | 2 | installing the hooks: G11 | — |
| M2 | Turing / Windows spike | vendored code (originals + patch) in `moves/tmr/`; loader with `load=False` skeletons and keyword calls; `moves tmr fetch / check / spike`; `SOURCE.json`; spike report; V4 bounds; corrected `model.PLAN` | shape and determinism on a randomly initialised encoder (no download); patch re-application test; hash checks against section 2.5; the spike table (2.3) | 1.5 | G1, G2, G3, G4 | M0 |
| M3 | SMPL-X → SOMA | `soma.py`, `soma_review.py`, per-route folders and `report.json`; R1 driver; `soma` bootstrap feature with the V9 check; R1 pod job | 3.7 unit tests; V1-V9 on synthetic, then larissa_kadu | 2.5 + one R1 pod session | G5, plus G6 or G8 | M2 (somaskel30 asset), P0a for real windows |
| M4 | Embedding extraction | `embed.py`; file contract with per-window `grid`; cache; `moves validate` checks | FakeEncoder plumbing tests (listed after this table) | 1.5 | G5 for real takes | M3 |
| M5 | Graph decoder | `decode.py`; explicit `_none` class; `posteriors.npz`; sampled paths; `propose_labels(decode="graph_hmm")` with the clipped-piece rule | decoder tests (listed after this table) | 2 | none | M1 |
| M6 | Classifier | ladder, CL2N, LR, prototypes, interpretable baseline, nested cross-fitted LODO harness with song groups, fold models, purity filter, model files, ablation report | separable synthetic classes recovered; level refusal; npz round trip with no pickle; no statistic of a held-out song group leaks into its fold (asserted on synthetic data) | 3 code; runs gated | G9 and B1 data | M1, M4, M5 |
| M7 | Calibration and captions | `calibrate.py` (temperature, duration-weighted Platt, family and narration maps, pooling for new dances, the gate, cautious `p_shown`); `calibration.json`; label contract + `validate_labels`; `captions.py` + `captions.json`; export allow-list for captions | calibration and caption tests (listed after this table) | 3 | none for code; real calibration needs E1 or E1-N | M5, M1b |
| M8 | Viewer integration | `DanceGraphData` captions parser; `DanceHud` number + bar + uncertain state + legend; inset glow by rank; reaction title from capture.json v4 | VIEWER_SPEC §14 caption and reaction playtests via `hm_*` on placeholder and synthetic auto captions, with synthetic names | 2 | none | M7; coordinate with the agent editing `Assets/DanceGraph` |
| M9 | Active learning | annotate: pre-fill in gaps, review queue, confirm action, keep/relabel log, similar-windows panel; blind mode extended to hide all of these | queue order on synthetic posteriors; confirm sets `human` without changing the move and keeps `exposed_to_model`; blind mode hides every model output; log written | 2 | G9 for real proposals | M6, M7 |
| M10 | Text side (optional) | `taxonomy/move_prompts_en.json` (3-5 English body-movement paraphrases per move per role, from `name_en` and `description`; the display stays Portuguese); one-off LLM2Vec job (pod or CPU fp32, 64 GB RAM); cached text prototypes | every captionable move covered; prompt-file SHA recorded | 1, plus the user's account step | G7 | M2 |

**M4 plumbing tests** (with the FakeEncoder):
- the `start_eighth` join with the dataset, and −1 for time windows;
- ≤ 300 frames per window;
- validity masks;
- the role order;
- refusal of stale inputs;
- the cache key;
- export never copies `embed/`.

**M5 decoder tests:**
- a known path is recovered from noisy synthetic emissions;
- marginals sum to 1;
- pinned nodes are respected;
- chain breaks at gaps and at no-grid stretches;
- ε creates new-link candidates;
- with τ = 1, β = 0, ε = 0, unnormalised `exp(−cost)` weights and no self-loop prior, one-hot emissions reproduce
  `path.resolve_path` at segment level (the smoothed, normalised decoder legitimately differs, so it is tested
  separately on synthetic paths);
- clipped pieces shorter than 75 % lose their number.

**M7 calibration and caption tests:**
- the full caption rule table and the order of checks (provenance × status × threshold × conformal set);
- temperature and Platt recover a known synthetic miscalibration;
- the gate passes a calibrated synthetic model and fails one that is overconfident by 0.15 at 1,000 segments;
- `p_shown` never exceeds `confidence` and is rounded down;
- alternatives sum to at most 1, and only the first is calibrated;
- `validate` rejects narration `alternatives` with p, and numbers without a passed gate;
- the export allow-list.

**Total:** about 23 agent-days of code (P0 and M0 excluded). Training, calibration and real captions follow the data gates.

**Can start now, with no approvals:**
- M1, M1b (except installing the hooks), M5 and M7 (code and tests);
- the M4 plumbing with the FakeEncoder;
- the M3 code and tests with a synthetic rest-skeleton fixture.

These touch `dataset.py`, `labels.py`, `model.py`, `export.py`, `features.py` and `fingerprint.py`. Some of these
files are being edited by other workflows today, so implementers must re-read the current state first.

**Steps for each milestone.** These are the same whoever implements it.
1. Re-read the files the milestone touches and the latest `CONTRACTS.md`.
2. Write the contract change in `CONTRACTS.md` and in the module docstring first.
3. Implement and test it on synthetic data.
4. Run `python -m dancecap.moves validate synthetic`.
5. Report what still waits for approval.

---

## 10. Risks and open questions

### 10.1 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| **3D coverage and budget.** Only 5.8 s is fitted. A full dance needs the whole capture pipeline with changing camera sets, and MAMMA alone costs about $8-10 per set per dance (measured run, extrapolated), $45-180 for ten dances, against a $30 cap. Memory at full length is unknown. | No real training data for TMR, the features or the dataset. | The P0a pilot first; budget decision (Q5); one set per span instead of two; local overnight MAMMA; chunked fits; narration-only dances still feed priors and the narration map. |
| **TMR does not separate zouk moves.** Solo commercial mocap; insensitive to small details; weak on handedness; never saw partner dance. | No accuracy gain. | Criterion A1; signed handedness (two new head features) and relational features; the pipeline does not depend on the encoder; MIT TMR as an alternative. |
| **Conversion error changes embeddings more than the moves do.** R2 torso bias of 2-5°; R1 untested on our high betas; the uniform proportions distort reach. | Systematic shift of the embeddings. | V1-V9; V7 decides the route; R1 vs R2 ablation; root scale and floor offset constant per dancer and take. |
| **Turing or Windows failure** | Spike fails. | CPU fp32 (the default anyway), then `.venv-tmr`, then Runpod. |
| **Dependency contamination.** Kimodo pins transformers 5.1.0 and gradio ≥ 6.8; SOMA-X brings Warp and usd-core and is tested on other torch and numpy versions. The mamma environment is already fragile. | Breaks `.venv-nopo4d`, the pod base or the other stages. | Vendored motion path with a reviewed patch; separate `.venv-tmr` / `.venv-soma`; the V9 check in `feature_check`. Every install needs approval. |
| **Pickles.** `joints.p` and Kimodo's own `weights_only=False` loading unpickle code. | Code execution if a file is swapped. | Hashes pinned in this plan (2.5); `torch.load(weights_only=True)` only, no fallback; skeletons built with `load=False`; `joints.p` converted to npz once. |
| **Captions that sound surer than they are** | A shown "87 %" misleads the audience; the licence's Trustworthy AI terms forbid deceiving others. | Blind calibration data; the gate on an upper confidence bound; cautious lower-bound display; status per category and couple type; only the top label carries a number; "uncalibrated" without a number. |
| **Labels shaped by the model** (pre-fill, model-ranked queue, model-picked dances) | Calibration and accuracy look better than they are. | Blind dances, `exposed_to_model`, boundary scoring only on user-placed boundaries. |
| **Leakage inside a silo** (PCA, priors, tuning and calibration seeing the held-out dance) | Optimistic numbers behind the shown percentage. | Nested, cross-fitted LODO by song group; frozen defaults until B2; one pre-registered test. |
| **Silent silo leakage across levels.** `discover_takes` globs every take today; mixed-level dances. | Mixed levels in the first model. | M1 lands before any training; per-dancer levels and the `_mixed` silo; transfer only through G10. |
| **Correlated windows; one couple** | LODO overstates accuracy on new couples; the conformal guarantee is only approximate. | Status per couple type; LOCO from 3 couples; "seen couple only" labels. |
| **Consent and privacy.** Real dancers, students, J&J partners, bystanders, third-party footage. | Processing or publishing without consent. | G0 consent table; refusals in `DanceMeta`; anonymised titles; consent never exported. |
| **Public repo leaks.** `head-movement` is public; 5 unpushed commits already name the dancers. | Dancer names, levels or data published. | Hooks plus the guard test; synthetic names in public fixtures; the user decides the pending push (Q12). |
| **Build leaks.** StreamingAssets is packaged into Quest and desktop builds, and the export copies unknown keys today. | Other dances' names, paths, transcripts or approvals shipped. | Key-level allow-list and its test. |
| **Cloud uploads.** SMPL-X files and dancer motion go to third-party community hosts. | Breach of the SMPL-X licence terms or of consent. | Itemised G6 requests; the user picks the cloud tier (Q6); NVIDIA weights never uploaded. |
| **Approvals by agents.** Free-text approvals and command-line-only checks. | A gated step runs without the user's OK. | Approvals ledger with verbatim quotes, checks inside the library and `remote run`, hard refusal kept in `model fit` and `predict`. |
| **Lessons and J&J break assumptions** | Moves forced onto talk; teacher solos; random partners. | `exclude` spans; `_none` class; drill cap; J&J as a reserved test. |
| **Choreography leaks** (rehearsed demos) | Inflated decoder gains. | LODO by song group. |
| **Upstream drift.** Kimodo's 2026-03-19 breaking change moved I/O to `somaskel77`; SOMA-X went from 0.3.1 to 0.3.3 in 11 days. | Silent changes. | Pins and hashes recorded in every output; re-check before upgrading. |
| **Licence revocable or updated** (NVIDIA may update the Agreement; Llama 3 terms if text is used) | Loss of rights. | Keep weights and embeddings local; `LICENSE` hash compared on every fetch; text optional. |
| **Concurrent edits.** `takes/*.toml`, `Assets/DanceGraph/*.cs`, VIEWER_SPEC and `remote_jobs.py` are being edited today. | Field names may move. | Re-read before implementing; contracts first. |

### 10.2 Open questions for the user

**Licence and downloads**
1. **Licence and download.** Do you accept the NVIDIA Open Model License, and approve the motion-only download
   (about 19 MB) plus the Kimodo code and the small skeleton file? (G1-G3)
2. **Vendoring.** May we copy the 17 Apache-2.0 Kimodo files, with one reviewed patch, into `dancecap` (the local
   workspace repo only), or do you prefer a separate `.venv-tmr`?
3. **Text hints.** Do you want zero-shot text hints in the annotation tool? They need your own Hugging Face
   request for Meta Llama 3 access, and its licence.

**Money and cloud**

4. **Pilot.** May we run the P0a pilot (about $2-2.6) to measure full-length MAMMA time and memory?
5. **Budget for full-dance 3D.** With the corrected numbers (about $8-10 per camera set per dance, $45-100 for ten
   dances with one set per span, $110-180 with two; or about 9 h per set locally), do you raise the $30 cap, run
   MAMMA locally overnight, or mix the two? One camera set per span, or two?
6. **Cloud tier.** SMPL-X model files and dancer motion already go to COMMUNITY pods (third-party hosts) for MAMMA.
   The SMPL-X licence allows one archive copy and forbids making the model available to third parties. May jobs
   that carry them keep using COMMUNITY, or must they use SECURE (more expensive), or run locally?

**Dances, consent and publication**

7. **More metadata.** Will you supply a date and a venue per dance? Do any dancers share a name, so that we need
   dancer ids?
8. **Song title in the reaction.** "Ficar Sem Você" (with "feat. Urias" moved into the credit) or the full
   "Ficar Sem Você (feat. Urias)"?
9. **Capture type.** Will future Professional dances be multi-camera captures like Larissa/Kadu, or single
   videos? Single videos get narration labels, decoder priors and narration calibration, but no 3D and no TMR.
   Will any Jack and Jill dances be multi-camera?
10. **Uncalibrated captions.** In the viewer, before a level, category or couple type is calibrated, should auto
    captions show the move with "uncalibrated" and no number, or be hidden? (Renders for sharing hide them either
    way.)
11. **Consent scope.** For Larissa and Kadu, beyond photoreal digital doubles: did they agree to ML analysis, cloud
    processing, their names and level shown in captions and renders, and publication? For future dances: how will
    J&J partners, lesson students, bystanders and owners of third-party footage be covered?
12. **Pending public push.** `head-movement` has 5 unpushed commits; two put "Kadu + Larissa × Ficar Sem Você" and
    "Professional · Demo" into `docs/VIEWER_SPEC.md`. Push as they are (with the dancers' consent), or anonymise
    first? May we install the guard hooks in `head-movement` (G11)?
13. **Levels.** For a mixed-level dance (teacher with student, mixed J&J draw): hold it out, or assign it to a
    level? Is the proposed J&J division mapping right?

**Labelling and display**

14. **The shown number.** A cautious lower bound shown as "70 %" (right at least about this often), or three
    words (likely / probably / possibly)?
15. **Blind dances.** Will you narrate dances in blind mode (no proposal snapping) and confirm them end to end:
    every dance before the classifier exists, then about 1 in 3? Without them, captions never get a number.
16. **Earlier research steps.** The research stage already fetched Kimodo's skeleton files into a scratch folder
    and ran the R2 prototype on larissa_kadu (section 3.2). Approve that retroactively, or have the scratch copies
    and their outputs deleted?
17. **Comparisons across levels.** Will you want them later (for example Professional vs Novice fingerprints)?
    They would use the `--cross-level` path, and a model transfer would need G10.
18. **Volume.** About how many Professional dances per category (Demo, Lesson, Jack and Jill) do you expect?
    That sets when E0, B1 and E1 open.

### 10.3 Unknowns (to resolve by measurement, not assumption)

**Hardware and software:**
- Whether TMR fp32 runs and matches CPU on the 2080 Ti under native Windows (spike M2).
- Whether `motion_encoder.pt` loads with `weights_only=True`.
- Whether slicing 77 to 30 joints works with `SOMASkeleton77(load=False)`.
- Whether SOMA-X with Warp runs on Windows and Turing, and on the pod's torch 2.5.1.
- Real throughput on this machine.

**3D capture at full length:**
- MAMMA time per frame at full length: linear, or a large fixed cost plus a smaller per-frame cost (P0a).
- MAMMA memory versus sequence length, and whether chunked fits with overlap are needed, locally and on 80 GB.
- The runtime of the other capture steps (cameras, detection, sync, `ma_2d`) over a whole song.
- Fit quality where only 5-6 cameras remain (after about 70 s and 159 s for larissa_kadu).

**The model and its training data:**
- TMR's training window-length distribution, minimum length, cropping and augmentations (including whether
  clips were mirrored).
- The share and styles of dance in Bones Rigplay.
- Whether Rigplay's joints came from somaskel30 FK (rigid fingertips) or from somaskel77 then sliced.
- The language of TMR's training text (presumably English).
- The exact LLM2Vec prompt form used in training (Llama-3 chat wrapping or not, which depends on the load
  path).

**The conversion:**
- The size of the "very slightly different" gap between the SOMA rig and the SOMA layer.
- The R1 error on our high-beta, partly occluded dancers.
- How sensitive the embeddings are to R1 vs R2, to the root scale and to the floor offset.
- The right V4 bounds (M2).

**Our data:**
- The real move-duration distribution, whether zouk moves start on half-measures, and the window purity.
- How saturated the decoded `p_raw` is.
- How many segments a 3-minute dance yields (30-60 expected), which sets how many blind dances E1 needs.
- Whether TMR helps at all (A1).

---

## Sources

Upstream (pinned):
- Kimodo code at commit 58e7818: https://github.com/nv-tlabs/kimodo
  - `kimodo/model/tmr.py`, `kimodo/model/load_model.py`, `kimodo/model/loading.py`, `kimodo/model/__init__.py`;
  - `kimodo/motion_rep/reps/{tmr_motionrep,base}.py`, `kimodo/motion_rep/{feature_utils,feet,stats}.py`;
  - `kimodo/skeleton/{base,definitions,kinematics,registry}.py`, `kimodo/metrics/tmr.py`;
  - `ATTRIBUTIONS.MD`, `pyproject.toml`, `setup.py`;
  - `docs/source/key_concepts/{skeleton,motion_representation}.md`,
    `docs/source/benchmark/{results,metrics}.md`;
  - file tree and blob hashes: https://api.github.com/repos/nv-tlabs/kimodo/git/trees/58e781898b3d7e328a676a75d3e338c45dce3ad9?recursive=1
- Kimodo issues: #11 (dummy text-encoder mode, open PR) and #52 (fp32 vs TF32 reproduction, LLM2Vec memory).
- TMR-SOMA-RP-v1 at revision e427752: https://huggingface.co/nvidia/TMR-SOMA-RP-v1 (model card, `config.yaml`,
  `LICENSE`), and its file list with hashes:
  https://huggingface.co/api/models/nvidia/TMR-SOMA-RP-v1/revision/e427752ae3446dedba49e928c93ddc9f0e413401?blobs=true
- NVIDIA Trustworthy AI terms: https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/
- SOMA-X at commit d6aa640: https://github.com/NVlabs/SOMA-X
  - `soma/body/soma.py`, `soma/body/identity_model.py`, `soma/fitting/pose_inversion.py`;
  - `tools/{smpl2soma,convert_amass_to_soma,pose_converter}.py`, `tests/test_pose_inversion.py`;
  - `docs/{tools,installation,model_card}.md`, `README.md` (acknowledgements), CI requirements.
- SOMA-X assets: https://huggingface.co/nvidia/SOMA-X
- SMPL-X model licence: https://smpl-x.is.tue.mpg.de/modellicense.html

Papers:
- Kimodo: arXiv 2603.15546.
- SOMA-X: arXiv 2603.16858.
- TMR: arXiv 2305.00976.
- TMR++ zero-shot study: arXiv 2405.16909.
- Action descriptions for zero-shot: arXiv 2404.12192.
- SimpleShot: arXiv 1911.04623.
- Linear probe on embeddings (Tian et al.): arXiv 2003.11539.
- Tip-Adapter: arXiv 2111.03930.
- LP++: CVPR 2024.
- Temperature scaling: arXiv 1706.04599.
- Verified calibration and the debiased calibration-error estimator (Kumar et al. 2019): arXiv 1909.10155.
- Measuring calibration: arXiv 1904.01685.
- Top-label calibration: arXiv 2107.08353.
- Combining probability forecasts (Ranjan and Gneiting): J. R. Stat. Soc. B 72(1), 2010.
- Conformal prediction introduction: arXiv 2107.07511.
- Hierarchical conformal: arXiv 1809.07441.
- Clustered conformal: arXiv 2306.09335.
- TypiClust: arXiv 2202.02794.
- B-ACT: arXiv 2604.15173.
- CoMPAS3D: arXiv 2507.19684.
- Hybrid HMM/ANN (Bourlard and Morgan): https://publications.idiap.ch/downloads/papers/2013/Bourlard_KLUWER_1994.pdf
- scikit-learn calibration guide: https://scikit-learn.org/stable/modules/calibration.html

Local:
- Documents:
  - `dancecap/docs/MOVES_RESEARCH.md`;
  - `head-movement/docs/VIEWER_SPEC.md`;
  - `dancecap/CONTRACTS.md`.
- Takes and taxonomy:
  - `dancecap/takes/larissa_kadu.toml`;
  - `prototype/larissa_kadu/alignment.json` (camera spans);
  - `dancecap/taxonomy/zouk_graph.json`.
- Code:
  - `dancecap/dancecap/moves/{labels,dataset,features,model,path,fingerprint,export}.py`;
  - `dancecap/dancecap/moves/annotate/{pipeline,timemap,asr,asr_worker}.py`;
  - `dancecap/dancecap/remote_jobs.py`, `dancecap/dancecap/remote.py`, `dancecap/remote/bootstrap.sh`;
  - `dancecap/dancecap/gpu.py`, `dancecap/dancecap/song.py`, `dancecap/dancecap/mamma_dyn.py`;
  - `head-movement/Assets/DanceGraph/{DanceGraphData,DanceGraphLayer,DanceHud}.cs`.
- Data:
  - `dancecap/work/larissa_kadu/{smplx,moves,beats,mamma}/`;
  - `dancecap/work/runpod/ledger.json`;
  - `dancecap/work/runpod/sessions/20261006-113945-batch/` (measured MAMMA times).

---

## Review log

Three reviews (feasibility, governance, method) all returned "accept with fixes". Every high and medium issue
is addressed; the cheap low ones too. Net effect: the code estimate grows from about 18 to about 23 agent-days
(new M1b, larger M1, M2, M6 and M7), three gates are added (G0 consent, G10 transfer, G11 public repo), and the
first calibrated number on an automatic caption moves later, to about 10 blind dances.

**Feasibility review**
- *High, full-dance 3D scope and cost.* P0 now covers the whole capture pipeline with camera sets per availability
  span (3.8, with the larissa_kadu span table). Costs are re-derived from the measured pod run: about $8-10 per set
  per dance, $18 for two sets, $45-180 for ten dances (4.6, At a glance, Q5). Added the P0a pilot, the memory and
  chunking unknown (10.3) and the local runtime per set. One refinement: the 3-frame dry check took 280 s, so
  linear scaling is an upper bound and a lower bound is given too.
- *Medium, vendoring closure.* The exact 17-file closure and the 4 replaced `__init__` files are listed;
  originals are kept unchanged plus one reviewed patch, with every original and the patch hash-pinned (2.4). M2
  effort 1 → 1.5.
- *Medium, loader.* Skeletons are built with `load=False`; `encode_motion` takes keywords; rest joints come only
  from `somaskel30.npz`; the somaskel77 assets join G3 only if the 77-joint check needs them (2.1, 2.3).
- *Medium, `h` features.* Two new signed head features in M1, with a features version bump, a re-run and tests
  (5.1).
- *Medium, pod job.* The pod job does R1 only, downloads only `soma/r1/`, and leaves features, R2, compare and
  embed to `after_download`. R1 and R2 write to separate folders. TF32 is off on CUDA (2.3, 3.4, 4.6).
- *Medium, calibrated status.* `calibrated` now requires the segment map and a passed gate (6.4).
- *Medium, alternatives and family p.* Only the top label carries a calibrated p; alternatives are rescaled raw
  marginals shown as names; the family needs its own map (5.5, 6.5, 6.6).
- *Low, all fixed:* clipped pieces (5.3); the `load_model` wording (2.3); the decoder test conditions (9, M5);
  J&J's test role tied to multi-camera capture, and per-window `grid` with −1 starts (4.1, 4.3, 7.5); the TMR++
  fallback wording (1.4); V4 bounds (3.5); PCA inside folds and the SOMA-X version check (3.2, 5.2).

**Governance review**
- *High, consent.* New G0 and a per-dancer `[consent]` table with source rights and bystanders; `DanceMeta`
  refuses missing scopes; anonymised titles; OML §2.3/§2.4 and the Trustworthy AI terms written out; renders show
  auto captions only when calibrated (2.1, 6.5, 7.1, Q11).
- *High, public-repo guard.* Hooks inside `head-movement/.git/hooks` (new G11) plus a dancecap guard test with the
  full extension list, a large-blob rule and a dancer-name deny-list; synthetic names in public fixtures; the 5
  unpushed commits are flagged and wait for the user (2.2, Q12).
- *Medium, approvals.* Approvals ledger with verbatim quotes, checks inside library functions and `remote run`,
  `synthetic = true` markers, the hard refusal kept in `model fit|predict`, and no agent touches the cap
  (2.1). Its honest limit is stated.
- *Medium, mixed levels.* Per-dancer levels, a `_mixed` holding area, a J&J division table, and transfer as G10
  (7.3, Q13).
- *Medium, leakage inside a silo.* Nested, cross-fitted LODO for every statistic; fold-level stats under `eval/`
  (6.3, 7.2).
- *Medium, export leaks.* Key-level allow-list and a test that no exported JSON carries other takes' names or
  paths (2.2, 6.6).
- *Medium, cloud uploads.* G6 itemises data, licensed files, packages, assets and cloud tier; the SOMA-X asset
  licence check joins G8 and the R1 pod gate; NVIDIA weights are no longer uploaded (2.1, 4.6, Q6).
- *Low, all fixed:* the research stage's earlier fetch and R2 run are disclosed, with retroactive approval or
  deletion asked (3.2, Q16); expected hashes are now in the plan (2.5); `weights_only=True` only, no fallback;
  the Apache change notice, `ATTRIBUTIONS.MD`, Derivative Model status, Llama duties and licence-change detection
  are added (2.1, 2.4). The ledger figure is updated to $5.89 spent, no pod open.

**Method review**
- *High, gate noise.* The gate now uses the upper confidence bound of the debiased calibration error, bins of at
  least 50 segments, a one-sided per-bin overconfidence check, and E1 at ≥ 10 dances and ≥ 300 blind segments; a
  simulation table shows why (6.4, 8.1). Partly declined: the two-sided per-bin test, which even a calibrated model
  would rarely pass below about 150 segments per bin (note in 6.4).
- *High, model-shaped labels.* Blind dances chosen at random, narrated without model help, confirmed end to end;
  `exposed_to_model` and `boundary_source` recorded (6.2, 8.5, M9).
- *High, uncalibrated numbers.* Only the top label shows a number; second candidates and the next-move glow are
  ranks; the family needs its own map; the order of conformal and p rules is defined (6.5).
- *High, calibration per silo only.* Status per (silo, category, couple seen or unseen); LOCO required for new
  couples; A1 must also pass under LOCO before TMR feeds those captions (1.4, 6.4, 8.4).
- *Medium, all fixed:* time-weighted correctness (6.1); `p_raw` saturation and Viterbi-marginal disagreement
  (5.5); nested tuning with frozen defaults until B2 and one pre-registered test (8.3, 8.4); deployed model equals
  scored model, with explicit group splits if scikit-learn is used and cross-conformal wording (6.3);
  boundary scores only on user-placed boundaries, snapping on and off (8.2); A3 is an honesty gate only, with the
  confident-wrong share defined and reported, and a "when will you see a number" line (At a glance, 8.2, 8.4);
  cautious lower-bound display rounded down (6.5); fingerprint prior bias and "model spread" (7.1); the ≥ 20
  instance bar for named captions (5.2); narration calibrated against the user's corrections, shown as "narrated"
  until then (6.3, 6.5).
- *Low, all fixed:* window purity (5.2); explicit `_none` class, per-dance weighting, J&J as a reserved test, and
  stratification moved from folds to blind-dance choice (5.4, 7.5).
- Partly declined: excluding every dance that shares the held-out couple. With one couple that leaves no training
  data, so couple exclusion waits for LOCO (note in 8.2).
