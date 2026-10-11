# Class speech: ASR + diarization of the recap (package `dancecap/speech/`)

Written 2026-10-09 (agent S1). Run outputs live in `work/larissa_kadu_recap/speech/` (git-ignored: they hold the voices of real people).

## What it does

Seven phones film one 6-minute zouk class (take `larissa_kadu_recap`, master clock = `01-Recap-Tina` video time). Every phone hears the same
room, so the audio of all seven is one multi-channel recording. The package, per phone:

| stage (pod step) | engine | output `raw/<cam>/` |
|---|---|---|
| `vad` | Silero VAD (faster-whisper's ONNX) | `vad.json` regions + frame probabilities |
| `whisper_large-v3`, `whisper_large-v3-turbo` | faster-whisper, word times + word probabilities, glossary prompt, passes auto / en / pt, language id per region | `whisper-<model>.json` |
| `parakeet` | NeMo parakeet-tdt-0.6b-v3 (word confidences = mean max-softmax of the word's tokens) | `parakeet-tdt-0.6b-v3.json` |
| `canary` | NeMo canary-1b-v2 (third family; language per segment from the Whisper region id) | `canary-1b-v2.json` |
| `embed` | speechbrain ECAPA + NeMo TitaNet embeddings, F0, level, 1.5 s windows every 0.75 s | `embed.npz` |
| `sortformer`, `segembed` | NeMo Sortformer 4-speaker diarization (overlap aware), every segment embedded | `sortformer.json`, `segembed.npz` |
| `events` | AST AudioSet tagging (laughter, applause, crowd, speech gender ...) + 3-8 Hz modulation | `events.json` |
| `quality` | SNR, level, clipping, band, rolloff | `quality.json` |
| `regions` (on request) | the decoders above with several settings on listed hard passages | `raw_regions/<id>/<cam>.json` |

All times in `raw/` are the phone's OWN seconds; `speech.sources.Clock.to_master` (slope * t + offset from `work/<take>/sync/sync.json`) rebases them.
Locally (numpy / sklearn, seconds): `cluster.py` (joint clustering of all phones on the master clock), `identity.py` (names from addressing + acoustic priors, jokes),
`events.py`, `report.py`, `diarize.py` (-> `diarization_global.json`), `review.py` (30 clips), `plots.py`.

## Contracts

- Never name a voice from a self-introduction, a name mention or a remark about appearance (the class has two jokes of exactly that kind); `identity.JOKE_NOTE`.
- `diarization_global.json` (schema `dancecap.speech/1/diarization_global`): `voices[]` {voice, name, naming evidence, seconds, f0, gender tags, loudest phones},
  `naming` {assignment, evidence, naming_confidence}, `turns[]` {start, end (master s), voice, name, confidence, margin, overlap_s, loudest_cam},
  `interjections[]` (route B: a second voice inside a turn or in a pause), `overlaps_sortformer[]`, `utterances[]` {start, end, voice, name, confidence, text (parakeet preview), alt},
  `mentions[]`, `reactions[]`, `jokes[]`, `emoji_tokens[]` (U+1F602 candidates with the time of the laugh or of the end of the joke), `introduction_readings[]`,
  `unplaced_sources` (phones without a clock, in their own time), `limitations[]`.
- A phone without a clock (04-Recap-Tony) takes part in no cross-phone step.
- Pod jobs: `speech.jobs.job_spec` (kind `asr`), environment `speech/speech-env-<id>.tar` on the volume (`speech/bootstrap/speech_env.sh build|unpack`).

## Re-run

    python -m dancecap.speech audio --source <recap folder>     # once: 16 kHz / 48 kHz wav + m4a, ffmpeg
    python -m dancecap.speech glossary
    python -m dancecap.speech pod up --gpu "NVIDIA L4"          # budget gate, kill switch, ledger row
    python -m dancecap.speech push-audio                        # hash-diffed upload to the volume
    python -m dancecap.speech submit; python -m dancecap.speech start-worker; python -m dancecap.speech job status
    python -m dancecap.speech job pull; python -m dancecap.speech pod down
    python -m dancecap.speech events; report; diarize; review; plots     # local post-processing

Tests: `python -m pytest tests/test_speech.py -q` (offline, no GPU).

## Fusion: one transcript from 24 (S2, `fuse*.py`, local, no pod)

The six phones with a clock x four engines (whisper large-v3 `:en`, large-v3-turbo `:en`, parakeet, canary) are 24 hypotheses of the same speech. `fuse.py` aligns them word by word into a ROVER-style confusion
network (dynamic programming with a sound / spelling substitution cost, a gap cost and a time penalty, leave-one-out refinement), votes per slot (a hypothesis that recorded but wrote nothing votes "nothing"; a phone
that was not recording does not vote; fillers are judged only among the engines that write fillers) and keeps for every word the vote tally and the alternatives. Vote weights are the probability that a token agrees with
the vote of all the others, from engine, own word confidence, local SNR and distance proxy (`fuse_levels.py`); they change the result by < 0.5 %, the gain is the voting. `fuse_post.py` adds words with times (parakeet-anchored
weighted median), speaker per word (diarization turns snapped to the silences between words), turns, sentences, events and a joke scan; `fuse_batches.py` writes the places where the vote did not settle.

    python -m dancecap.speech fuse            # speech/consensus_v0.json, transcript_v0.txt, disagreements/batch_NN.json (about 20 s)
    python -m dancecap.speech fuse-eval       # speech/fusion_metrics.json (about 13 min); --only clock refreshes just the clock check
    python -m pytest tests/test_speech_fuse.py -q

- `consensus_v0.json` (schema `dancecap.speech/1/consensus_v0`): `turns[]` {id, voice, speaker, start, end, speaker_conf, voice_conf, interjection, text, text_clean, sentences[], events[], words[]}, each word {text, norm,
  start, end, punct, share (winner's share of the voting weight), margin, null_share, tally[] {w, share, n, by, text}, hyps, phones, engines, eligible_phones, t_spread, filler, flags, speaker_conf, speaker_edge, ...};
  `events[]` (laughter tokens U+1F602 at S1's times, flagged `needs_owner_confirmation`), `wordless_sounds[]`, `glossary_hits[]`, `suspects[]`, `jokes` {diarization_candidates, emoji_tokens, owner_facts, scan},
  `unplaced_sources["04"]` (Tony fused in his own seconds), `stats`, `limitations[]`. It is a vote, NOT adjudicated: `consensus_text` of a span can be a stitched reading that no engine wrote.
- Disagreement batches (`dancecap.speech/1/disagreement_batch`): a slot is disputed when the winner holds < 0.90 of the weight (chosen from the held-out reliability curve: below it a held-out transcript disagrees in
  15.6 % of the slots, above in 2.3 %), when "nothing" holds >= 0.25 against a winning word or a single word holds >= 0.15 against "nothing", in overlap / second-voice windows, for known mishearings and
  glossary near-misses, and in the owner's joke windows (`speech/owner_facts.json`, git-ignored). 177 spans, 9 batches of 19-20; each span has the speaker, times, consensus text, the sentence and the 2 sentences before / after,
  every candidate reading with `phone/engine confidence` sources, the non-voting targeted decodes, glossary hits, reasons, a phone to listen to.
- Measured (`fusion_metrics.json`, pseudo-WER against held-out transcripts, 61-359 s): fused 0.199 vs 0.266 for the best single engine when the phone is held out (24/24 references better, 20/24 better than the best single
  transcript), 0.207 vs 0.285 when the reference's engine is held out too; weights, compound merging and refinement rounds change it by < 0.5 %; the words show no residual in any phone clock (all within +-14 ms).
- Limits: errors shared by every engine survive ("lights engage" for "legs engaged"), a quiet second voice is dropped, names are not heard ("Kadu" is never written; "Larissa" at 117 s only by one of 24 hypotheses), the
  naming of the voices therefore rests on the acoustic prior alone. Details: `work/larissa_kadu_recap/NOTES_speech2.md`.

## Final transcript and what is derived from it (S4, `final_*.py`, local, seconds)

`consensus_v0.json` + the language-model editors' decisions (`disagreements/batch_NN.decisions.json`, one per span, every span must have one) + the owner facts become the FINAL transcript. `final_merge.py` replaces
the words of each span by the decision text (words the decision keeps stay with their consensus time and vote share; new words get a time from the raw word times of the engines that wrote them, else a share of the
span's window; punctuation and case come from the decision), assigns speakers, places the laugh tokens (U+1F602) after the words of the two jokes, builds turns and sentences (`utterances`), gives every word a confidence
(`conf`: the editor's confidence on a decided span, the vote share elsewhere) and checks the result (monotone times, nothing lost or duplicated, no repeat at a turn edge, the recurring terms of the editors). What is
specific to THIS recording (laugh times, speaker corrections, chapters, teaching-event curation, joke findings, recurring terms) is read from `speech/final_overrides.json` (git-ignored with the other real-person data).

    python -m dancecap.speech final [--no-clips]     # everything below, about 4 s (the clips need ffmpeg)
    python -m pytest tests/test_speech_final.py -q

Outputs under `work/<take>/speech/` (all master clock = 01-Recap-Tina video time, seconds; the film starts at 0, words before 0 exist only for phone 00):

- `transcript_final.json` (`dancecap.speech/1/transcript_final`): `turns[]`, `utterances[]` {id, turn, speaker, voice, start, end, text, text_clean, conf_mean, conf_min, flags, events, words[]}, a word is {text (lower case,
  capitals are applied when rendering), norm, start, end, punct, kind word | filler | event, conf, conf_src vote | adjudicator | owner, edit vote | kept | changed | inserted | event | owner, share, span, flags, v0}, `events[]`, `adjudication`
  {spans[] with every decision, changes[], counts}, `proofread_first[]`, `checks`, `systematic[]`, `stats`, `unplaced["04"]` (Tony, own seconds, no clock), `speakers` (Larissa, Kadu, other: nobody else speaks), `naming`.
- `transcript_final.srt / .vtt / .txt`: speaker-prefixed cues (the prefix is not counted in the 42 characters), a readable text with `*word*` for confidence < 0.6 and the proofread-first list.
- `captions.json`: the film's word-highlight captions, <= 2 lines of <= 42 characters (an emoji counts 2), per word start / end, the laugh as its own word (kind `emoji`), `speaker_key` (speaker_larissa / speaker_kadu /
  speaker_other, with suggested colours), `check` = the caption contains a word the owner should proofread (dim or skip it), `show_start / show_end`. Fillers and words before master 0 are not captions.
- `chapters.json`: 13 chapters, contiguous from 0 to the end of the master, title <= 5 words, summary, key terms checked against the text, speakers' seconds, mean confidence.
- `teaching_events.json`: the transcript-to-direction bridge. Every moment a teacher refers to something a 3D film can show (body part, axis / exit, weight, step, direction, path, count / enumeration, move name, "again",
  "slow", "like this"), found by rules (`final_direct.RULES`) and curated, with PROPOSED tags from camera_focus, overlay_T_axis, overlay_weight_change, overlay_footsteps/physics, overlay_timing,
  overlay_path/floorcraft, slow_motion, replay, graph_moves, caption_only, `focus` {who leader | follower | couple, where, speaker} (Kadu = lead, Larissa = follow), chapter, `needs_check`. Proposals, not decisions.
- `jokes.json`: the two joke tokens in the text, `laugh_candidates[]` (flags laugh_candidate, `approve_emoji` null = the owner decides), stretches explained by speech, what the pipeline found on its own evidence.
- `review.html` (+ `review/final_clips/`): a self-contained page for the owner: transcript by speaker with timestamps, low-confidence words highlighted, the spans to proofread first with an audio clip each, the
  jokes, S1's 30 who-is-who clips, the evaluation. Voices of real people: local only, never publish.
- `evaluation.json`: pseudo-WER of the vote (v0) and of the final against each of the 24 sources (in-sample: the final is on average farther from single sources than the vote, because the editors restored words the
  vote had dropped and overruled the majority on some spans), the span-by-span effect, the held-out numbers of the vote (S2), a band-based estimate of the expected accuracy.
- Limits: the adjudication has no held-out measurement and no ground truth; the opening jokes (first 8 s) are the owner's words, no engine writes "Kadu"; the naming of the voices is a prior. Details:
  `work/larissa_kadu_recap/NOTES_speech4.md`.

## Independent audit of the final transcript (S5, `audit.py`, `audit_report.py`, `audit_struct.py`, `pod/stage_audit.py`)

A re-check that does not trust the pipeline: its own seeded stratified sample (25 utterances after the opening: speaker group x third of the class x loud / quiet best phone), engines nobody used before (faster-whisper
large-v2 without prompt, parakeet-tdt-1.1b), teacher-forced scoring of competing readings (whisper large-v2 and large-v3), CTC forced alignment (torchaudio wav2vec2) for the word times, WavLM x-vectors (not ECAPA /
TitaNet) for the speaker labels, CLAP (not AST) for laughter, and consistency checks of SRT / VTT / captions / chapters. Everything lands in the git-ignored `work/<take>/speech/audit_s5/`; one L4 pod job (about 0.33 USD).

    python -m dancecap.speech.audit sample | inputs            # sample.json, audit_inputs.json (needs audit_s5/score_candidates.json: git-ignored, it quotes the class)
    python -m dancecap.speech pod up | sync | put ...          # then run `python -m dancecap.speech.pod.stage_audit all --inputs ... --out ...` on the pod (see the module docstring), get the outputs into audit_s5/pod_out/, `pod down`
    python -m dancecap.speech.audit_struct                     # exports consistency, no pod
    python -m dancecap.speech.audit_report all                 # audit_report.json from the pod outputs and audit_s5/sample_judgments.json (my manual judgments)
    python -m pytest tests/test_audit_s5.py -q

Notes of the findings: `work/larissa_kadu_recap/NOTES_speech5.md`. The pod needs `transformers` to load the two .bin checkpoints (WavLM, CLAP) with the torch-2.6 load check switched off (`allow_trusted_bin`), and
parakeet-tdt-1.1b drops words in segments longer than about 10 s and on the far phones: read it as a second opinion only.

## Corrections after the audit, and the owner's opening (S6, `final_merge.py` / `final_direct.py` / `final_run.py`, local, seconds)

The audit found a few concrete defects; they are fixed through `speech/final_overrides.json` (git-ignored) and `python -m dancecap.speech final` (about 5 s), never by editing a generated file. Every correction
carries a one-line `why`; the list of what changed in this recording is `work/<take>/speech/CHANGES_S6.md` (local). The mechanisms are generic:

- `span_overrides` = `{span id: {why, text?, confidence?, decision?, flags?, add_flags?, drop_flags?, needs_owner?, alternatives?, note?}}`: a corrected decision for a disputed span (the editors' text was
  unsupported). `why` is mandatory, an unknown span or field is an error, a changed text without `decision` becomes `rewrite`; the editor's own reason stays in `editor_reason`, the old values in
  `override.was`, the log in `adjudication.span_overrides[]`. Confidence rules are those of any decision (a span flagged `unclear` caps its words).
- `speaker_windows` = `[{t0, t1 | span, voice?, name?, interjection?, conf?, flags?, why}]`. Without `voice` the label is kept and only the confidence is lowered (`voice_conf` / `speaker_conf` become the smaller value) and
  `flags` are added; the audit uses the flag `speaker_unresolved` for a short utterance whose speaker the independent evidence does not settle (shown as "speaker not settled" in the text export and a badge in
  `review.html`; a caption carries it in `flags`).
- `utterance_breaks` = `[{t, why}]`: split the sentence that contains time `t`, structure only (no word is changed). The first half keeps its id, the second gets the id plus a letter (`u003`, `u003b`) and
  `continues`, so every other utterance id (and every reference to it in chapters, captions, teaching events and notes) stays what it was.
- `owner_clips` = `[{id, start, end, listen_to?: {phone}, pad?}]`: audio clips (exact range, `pad` default 0) that `final` cuts into `review/final_clips/` for the owner's questions.
- `speech/OWNER_QUESTIONS.md`, when it exists, is shown at the top of `review.html` ("Questions for you") with an audio player for every `review/clips/*.wav` / `review/final_clips/*.wav` path it names.

**The owner's opening answer, applied in one step.** The first seconds of the recording (overlapping voices and laughter) cannot be settled from the audio. When the owner has listened to the clips, the answer goes
into `work/<take>/speech/owner_opening.json` and the next `python -m dancecap.speech final` merges it over the opening (nothing else to edit):

    {"segments": [{"start": 1.2, "end": 1.7, "speaker": "Larissa", "text": "Hello!"},
                  {"start": 1.9, "end": 3.0, "speaker": "Kadu", "text": "What is said, with the punctuation wanted."}],
     "emoji": [{"t": 2.1, "speaker": "Larissa", "joke": "optional label"}],
     "from": 1.2, "until": 8.5,
     "chapter_summary": "optional: replaces the summary of chapter 0", "chapter_title": "optional", "conf": 0.95, "note": "optional"}

- Every word and token that starts inside `[from, until)` is replaced (default window: the segments and emoji; give `until` = the start of the recap, a span only partly inside the window stays open). A segment's
  `text` is read like a decision text (case and names as usual, inline emoji allowed); its words are timed from the engines that wrote them inside the segment, else spread over it by length.
- The owner's words get `conf` 0.95 (`conf_src` owner), the flags `owner_opening` / `owner_fact`, never `unclear` / `needs_owner`; a segment is a second voice (`interjection`) when the speakers before and after it
  are the same other voice (or `"interjection": true|false` in the segment). `emoji` lists every laugh token wanted (none listed = none in the text); each is a caption word, its `speaker` is kept as `laugher`.
- The decision spans the window covers become `decision: owner`, `owner_resolved`, no longer `needs_owner` / `unclear` and drop out of `proofread_first`; `adjudication` keeps the editors' text as `editor_text`.
  A laugh candidate in `jokes.json` that holds an owner token gets `approve_emoji: true`. `utterance_breaks` inside the window are ignored. The removed words are counted in `owner_opening.removed_v0`, so the
  `lost_words` check still means "nothing disappeared without a record".
- An empty answer (`segments` and `emoji` empty) changes nothing. Tests: `tests/test_speech_final.py` (`test_owner_opening_*`, `test_span_overrides_*`, `test_utterance_break_*`).

## Voice-to-voice class audio (V1, 2026-10-10: `speech/v2v*.py`, `pod/stage_v2v.py`)

The owner asked for the teachers' speech to be converted into their own clean cloned voices (ElevenLabs speech-to-speech, the owner's key and the two voices
"Larissa" and "Kadu" of his account; permission recorded 2026-10-10). Only the teachers' own speech was sent (no student speech exists in the class); the key is read
from the Desktop key file at run time and is never printed or stored (`v2v.ElevenKey`; the ledger `v2v/spend_ledger.jsonl` holds credits and request ids only).

    python -m dancecap.speech v2v plan                       # pieces (<= 40 s, whole stretches of one teacher), phone per piece, light cleanup (high-pass, spectral gate <= 9 dB, level)
    python -m dancecap.speech v2v refs                       # the voices' own training samples (reference of the voice check), into work/<take>/speech/v2v/ref/ (never committed)
    python -m dancecap.speech v2v convert --variant D        # one request per piece, cached by a hash of audio + voice + model + settings; stops on a quota error
    python -m dancecap.speech v2v align --variant D          # envelope cross-correlation back onto the master clock; time-stretch only above 80 ms of drift
    (pod)   v2v_gate.build_items / build_run / job_spec / run_pod    # ASR (faster-whisper large-v3, no prompt) + ECAPA voice check on ONE GPU pod of kind v2vgate, owner V1-v2v
    python -m dancecap.speech v2v decide --runs g2,g3 --variants D,D2   # accept / reject per piece -> decisions.json, gate_table.json
    python -m dancecap.speech v2v assemble                   # film/audio/class_mix_v2v.wav + stems/ (larissa, kadu, original, bed)

Gate (per piece, vs transcript_final.json): WER of the converted piece clearly better than the film's current audio (gain >= 0.10) or <= 0.10 while that is >= 0.20,
and not worse than the cleaned original it would replace, not above 0.6, at most 2 inserted words, own teacher's reference nearer than the other's by 0.10, no clipping.
A piece that fails keeps the CLEANED ORIGINAL (best phone by SNR + the S1 engines' own WER on that phone, light denoise). The opening (0 to the first pause after 9 s) stays the
class mix until `speech/owner_opening.json` exists; then re-run `plan` (unchanged pieces keep their ids and conversions), `convert --only`, the gate, `decide`, `assemble`.
`film_recap_audio.soundtrack_class_mix` picks `class_mix_regen.wav` (V3), else `class_mix_v2v_full.wav` (V2), else `class_mix_v2v.wav` (V1), else `class_mix.wav` (switch `film/audio/soundtrack.json`
{"class_mix": "auto"|"regen"|"v2v_full"|"v2v"|"original"}); `class_mix.wav` and V1's `class_mix_v2v.wav` are never changed. `recap_export --audio` reads the same switch.
Findings and numbers: work/larissa_kadu_recap/NOTES_V1.md.

## Full cloned-voice re-recording (V2, 2026-10-10: `speech/v2v_full.py`, `v2v_splice.py`, `v2v_warp.py`, `v2v_check.py`, `v2v_pod.py`, `pod/enh_v2v.py`, `pod/asr_v2v.py`)

V1 sent whole turns from one lightly cleaned phone and 36 of 38 came back garbled. V2 (owner: "finish the re-recording", credits sufficient) measured that the input is NOT the lever
(six enhancers change the ASR WER of the input by 0.02 and the converted WER not at all) but the speech-to-speech model itself: the WER of one sentence over 15 conversions of the same input
runs from 0.0 to 1.0. So V2 converts at sentence level (81 pieces of 2-8 s), tries each piece up to 16 times (other input / settings / seed), measures every try with the independent ASR
(faster-whisper large-v3, no prompt, on a pod) against transcript_final.json, splices the correctly recognised words of several tries where both have a pause (`v2v_splice`), and keeps the best.

    python -m dancecap.speech.v2v_full plan | inputs             # sentence pieces, phones ranked, candidate inputs (r1, r2, 3-phone delay-and-sum bf, pre1, mix)
    python -m dancecap.speech.v2v_pod up | sh | put | get | down  # ONE GPU pod, owner V2, own state file (v2v2/pod.json); `pod/enh_v2v.py` = the enhancers, `pod/asr_v2v.py` = the ASR + ECAPA
    v2v_full.campaign_round(k, pieces)   # try k: convert (RECIPES[k-1]), align, ASR, tries.json; v2v_full.splice_all; decide_policy; assemble_full -> film/audio/class_mix_v2v_full.wav

Policy (`decide_policy`, the owner's): use the converted piece unless its WER is worse than the enhanced original's by more than 0.10 AND above 0.25 (absolute); then the enhanced original
(3-phone beamform + MossFormer2_SE_48K) is used and the piece is flagged. The final decision uses WERs measured INSIDE the whole-class ASR of three candidate mixes (`in_context_piece_wer`),
because the best-of-16 isolated WER is optimistic. Credits: ~10.3 per second of audio; the session ledger `v2v2/spend_ledger.jsonl` holds credits and request ids only (never the key).
Numbers and what to listen to: work/larissa_kadu_recap/NOTES_V2.md. Never commit `speech/v2v2/` (voices of real people; work/ is git-ignored).

## Only regenerated lines (V3, 2026-10-11: `speech/v2v_regen.py` pure parts, `speech/v2v_regen_run.py` orchestration)

Owner: only regenerated ElevenLabs lines. Per sentence piece (V2's 81, plus `o3` = the opening's last words and `g1` = Kadu's 0.26 s "good"): (1) a speech-to-speech try whose isolated ASR WER is <= 0.15 and whose speech-band envelope
never lags the best phone by more than 80 ms in a 1 s window (V2's tries re-scored, up to 32 new ones `sts_recipes`, a lagging try re-timed with `v2v_warp.retime`, a try's stretches after a pause moved onto the transcript
onsets with `realign_runs`); (2) else text-to-speech in the same cloned voice from the transcript sentence (`tts_request`: with-timestamps, multilingual v2, neighbouring sentences of the same teacher as context, up to 6 takes, the
best by `tts_key`), fitted to the transcript clock by `fit_take` (runs of continuous words time-warped with WSOLA onto the transcript word times, pauses = silence, the audible onset replaces the alignment's character start).
The opening 0 - 8.45 s (owner-pending words, the laugh) stays the original recording; everything else is regenerated. `v2v_mix.assemble(..., bed_speech_spans=)` builds the room-tone bed from non-speech frames only.

    use v2v_pod.use_label("V3", "pod_v3.json")      # own pod record and owner label
    v2v_regen_run.sts_batch / tts_batch / asr_tags   # tries -> aligned/<tag>/<id>.wav (v2v2), candidate table v2v2/regen.json, credits v2v2/spend_ledger_v3.jsonl (never the key)
    v2v_regen.choose_final -> v2v_regen_run.realign_final -> assemble_regen -> film/audio/class_mix_regen.wav, stems_regen/; verify_mix (whole-class ASR, onsets, identity, joins)

Numbers and what to listen to: work/larissa_kadu_recap/NOTES_V3.md. Never commit `speech/v2v2/` (voices of real people).
