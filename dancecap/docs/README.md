# dancecap documentation index

One line per document. The project overview is the [README at the repository root](../../README.md); the data contracts
sit one level up in [`../CONTRACTS.md`](../CONTRACTS.md).

## Start here

| Document | What it is |
|---|---|
| [`../CONTRACTS.md`](../CONTRACTS.md) | Data contracts of every stage: conventions (time base, world frame, camera model, identity ids), file formats under `work/<take>/`, GPU and pod rules. Change it in the same commit as the stage it describes. |
| [PIPELINE.md](PIPELINE.md) | Canonical spec of the end-to-end pipeline: stages, dependency graph, optimal order, where each stage runs, measured numbers and what is built versus planned. |
| [FARM.md](FARM.md) | The Runpod farm contract: 30-second chunks, several pods, a persistent network volume, the world package, the watcher and the defrag-style coverage map; includes the change log. |
| [GIT.md](GIT.md) | Git policy of this repository: what is tracked, what must never be committed, the commit hooks and their rules, commit conventions, push policy. |

## Pipeline operation

| Document | What it is |
|---|---|
| [PIPELINE_PROGRESS.md](PIPELINE_PROGRESS.md) | Progress events, ETA estimation and the budget gate (`dancecap.progress/1`); the hands rows are historical (hands stages were retired). |
| [PIPELINE_MONITOR.md](PIPELINE_MONITOR.md) | Design and API contract of the Pipeline Monitor, the local web UI (`dancecap.monitor/1`); section 11 is the redesign around the farm. |
| [PIPELINE_MONITOR_GUIDE.md](PIPELINE_MONITOR_GUIDE.md) | User guide of the Pipeline Monitor: the Farm view (defrag map), buttons, and the older takes / stages / runs screens. |

## Analyses and films

| Document | What it is |
|---|---|
| [RHYTHM.md](RHYTHM.md) | Rhythm entrainment: how phase-locked the dancers are to the beat grid, the live score, its controls and its limits (an estimate, not a judge's score). |
| [SPEECH.md](SPEECH.md) | Class speech: voice activity, multi-model ASR and diarization of the phones' audio for lesson recordings (`dancecap/speech/`). |
| [RECAP_FILM.md](RECAP_FILM.md) | The directed tutorial film built from a class recap and its transcript: build commands, the directing rules and checks. |
| [MOVES_RESEARCH.md](MOVES_RESEARCH.md) | Research synthesis on classifying zouk moves (no library does it), the vocabulary, the counterbalance criteria and the staged plan. |
| [TMR_SOMA_PLAN.md](TMR_SOMA_PLAN.md) | Plan (not built) for move captions with calibrated confidence, with its governance gates and open questions. |
| [ZOUK_TERMINOLOGY.md](ZOUK_TERMINOLOGY.md) | Brazilian Portuguese names for the zouk moves, with sources and status; data in `../taxonomy/pt_br_terminology.json`. |

## Avatars and appearance

| Document | What it is |
|---|---|
| [HAIR_REFERENCE.md](HAIR_REFERENCE.md) | Acceptance criteria for the follow's hair in the viewer. Sections 2-4 hold a dancer's measurements: strip them before sharing the file. |
| [DANCER_SPLATS_RESEARCH.md](DANCER_SPLATS_RESEARCH.md) | Research and build notes for Gaussian splats of the dancers guided by the avatars (a PC review layer, not for the headset). |
| [MONO4D_NOVEL_VIEWS.md](MONO4D_NOVEL_VIEWS.md) | One phone video -> a 4D provider (Gracia, World Labs Atlas, the World API) -> virtual cameras -> MAMMA: what the vendors offer (2026-10-15), why generated views are not measurements, `dancecap.mono4d`, the hold-out protocol and next steps. |

## Other documentation in the repository

| Document | What it is |
|---|---|
| [`../../atlas_bridge/README.md`](../../atlas_bridge/README.md) | The early prototype: audio sync of phone videos and the NoPo4D experiment, with the quality gate that rejected it. Historical; the Atlas route continues in [MONO4D_NOVEL_VIEWS.md](MONO4D_NOVEL_VIEWS.md). |
| `../takes/*.toml` | Per-take configuration (footage folder, clocks, window, dancers, camera types); each file carries its own comments. |
| `../taxonomy/` | The move graph, trigger rules and Portuguese terminology as data. |
| `../worlds/` | World package manifests (hashes and paths only) that tie a take to its room model. |
| [head-movement](https://github.com/atonalfreerider/head-movement) | The Unity viewer: a separate public repository (MIT). |
