# Rhythm entrainment and the live rhythm score (`dancecap/rhythm.py`, `rhythm_entrain.py`, `rhythm_controls.py`)

**Rhythm entrainment (estimated from motion, not an official score).** For a Jack & Jill heat (or any take with a beat grid) it measures how strongly the dancers'
body sway and foot touchdowns are *phase-locked* to the song's beat grid, at whatever metrical level they lock to, and replays the estimate *live* along the dance. It does not
judge technique, partnering or musicality as a whole; it says how sure it is and why. **There is no validated mapping from this number to judges' scores** (see "Limits").

```
python -m dancecap.rhythm run <take> [--timing P] [--song-dir P] [--audio WAV] [--out DIR] [--no-plots] [--no-sensitivity]
python -m dancecap.rhythm install <take> --capture 07_Name       # put the viewer's copy into the capture folder(s): no full re-export
python -m dancecap.rhythm selftest                               # synthetic: locked near the top, unrelated / wrong tempo near the bottom
python -m dancecap.rhythm controls                               # the positive / negative / no-dance controls -> work/<heat>/rhythm/controls.json + controls.png
                                                                 # (the takes of the roles are named in the git-ignored work/rhythm_controls_takes.json: demo, demo_audio, ref38, heat, recap)
python -m dancecap.rhythm_controls --calibrate                   # the numbers the reference levels are read from (the demo's first half)
python -m pytest dancecap/tests/test_rhythm.py -q
```

It runs **locally** (numpy + scipy, about 70 s for a 200 s heat with the sensitivity analysis, about 12 to 20 s without; no pod: the pods rule is for CV / neural / heavy CPU work).
Re-run it whenever `timing/timing.json` (the beat grid, the touchdowns) or the physics layer changes.

## Why version 2: what the controls showed about the first score

Version 1 (agent S6) asked "does a foot touchdown land on a pulse / 'and' slot of the grid (60 ms kernel)?" and scored 5.3 / 100 on the Carlos + Aline heat. The brief of version 2 was
to find out whether that reads the dancing or the metric. The same pipeline on the **positive controls** (takes with a trusted, careful audio alignment):

| take | version 1 score |
|---|---|
| demo (a professional couple on a studio song, careful alignment) | **6.5** (range 3.8 to 7.2) |
| 38 s reference span of the demo (a second pose fit) | 4.9 |
| the heat | 5.3 (range 2.2 to 12.1) |
| the demo against a grid at 1.2 x the tempo | 1.0 |

Every professional take reads about 5 to 7, a wrong grid reads 1: the metric sits on its floor for all of them, so the heat's "5" says nothing about the heat. What was broken:

1. **The construct.** Zouk is flowing weight transfer, not a footfall on every slot. On the demo the touchdowns are at chance against the slot grid (310 touchdowns, 34 % within 60 ms against 31 % chance, phase concentration R 0.03 against
   0.06 for random, Rayleigh p 0.78): the same as on the heat. Widening the kernel to 100 / 130 ms only moves the demo from 6.5 to 8.3 / 9.5 and the heat from 5.3 to 6.6 / 7.4.
2. **The measurement is not the problem for slow sway.** The centre-of-mass sway of both couples is **phase-locked to the grid at the half-bar level (one cycle per two pulses, the "slow")**:
   whole-dance phase-locking value of the band-passed (+-12 %) horizontal CoM speed 0.39 (demo lead) / 0.57 (demo follow) / 0.36 (heat lead) / 0.54 (heat follow) against 0.11 to 0.18 for the same signal and a metronome at
   0.88 .. 1.12 x the tempo (6 wrong tempi); the CoM height gives 0.42 to 0.48 for three of the four roles. The locked phase of the leads' CoM height agrees between the two couples and two songs to 0.04 pulses (30 ms) (a weight transfer lands at the same place in the
   grid: another sign that the grid and the clocks are right). The foot touchdowns lock at the pulse / 3-in-2 level only weakly (R 0.14 to 0.29 at 4 or 6 cycles per 4 pulses, against 0.08 to 0.09 chance), and they carry +-0.05 to 0.2 s of detection noise.
3. **A constant clock offset matters little to a sway at 0.6 Hz** (+-30 ms is +-11 degrees of a 1.6 s cycle), but a 60 ms hard kernel on a 371 ms eighth is slot-exact.

So the controls say: *the metric and the measurement of foot touchdowns were mis-specified*, the heat's low number was not about the heat, and the body does keep time. Version 2 scores the **entrainment** of the motion.

## What is measured (version 2)

**Levels.** Cycles per 4-pulse measure `h` in {2, 4, 6, 8}: **half bar** (one weight transfer per two pulses), the **pulse**, **three in two pulses**, the **eighth**. The level a role locks to is chosen *causally* (the level with the highest lock so far) and the baseline pays for choosing (below).
The bar level (h = 1) is not used: its resolution against the wrong-tempo baseline is too coarse for a 45 s window.

**Streams.** *Body*: the centre of mass' horizontal speed (weight transfer) and its height (bounce) from `physics/physics.npz` (else the pelvis of `timing/kinematics.npz`), band-passed (+-25 % around the level's frequency, causal 2nd-order). *Feet*: the touchdown times
(`timing.json`) as an event train.

**Phase-locking value.** `u = x * exp(-2 pi i h phi / 4)` with `phi` the grid phase in pulses (interpolated between the grid rows, so the song's tempo changes and beat jitter are followed); `rho = |avg(u)| / avg(|x|) * 4/pi` (1 = a locked oscillation, ~0.3 = unrelated narrow-band noise at these window lengths). The average is a causal
exponential window: **45 s** for the body / feet / couple components (about everything since the start of the song, forgetting slowly so that a drifting phase can be followed), **15 s** for the recent form. For the feet `rho` is the length of the mean unit phasor of the touchdowns in the window.

**The permutation baseline.** Every statistic is also computed against 6 **metronomes at the wrong tempo** (the grid stretched by 0.84, 0.88, 0.92, 1.08, 1.13, 1.19 about the start of the song: the same regularity, no relation to the dancers), choosing their own level by the same rule, so the price of choosing among four levels,
the noise floor of a narrow-band estimator and the start-up bias are all in the baseline. `skill = clip((M - b) / (ref - b), 0, 1)` with `b` = the baseline's mean + 0.5 sd (trailing 20 s smoothing: causal), and the skill the baseline runs themselves reach under the same scaling (the positive bias of a quantity clipped at 0) taken off again:
**0 = what unrelated motion gives, 1 = the reference level `ref`**. The accents use the same recipe with the music moved by +-2.3 .. 5.9 s against the motion as the baseline (the onsets are not periodic). The lead-follow coherence uses the follow replayed at 0.72 .. 0.93 x the rate.
Resolution caveat: with these windows a tempo 8 % or more away from the song's is told from it; 5 % off is not (it still locks).

| component | weight | what | reference level |
|---|---|---|---|
| **Body sway** | 45 | phase-locking of the body streams to the grid at the chosen level, 45 s window, roles averaged | 0.48 |
| **Footwork** | 15 | phase concentration of the foot touchdowns at the chosen level (needs 3 steps in the window) | 0.31 |
| **Recent form** | 20 | the body lock over the last ~15 s (follows the last bars) | 0.48 |
| **Couple sync** | 10 | coherence of the lead's and the follow's locked oscillations at the lead's level (the grid phase cancels in the product: it is a property of the couple, not of the music) | 0.37 |
| **Accents** | 10 | stops (stillness after movement), **hits** (sudden decelerations of the CoM speed above half the dancer's trailing 90th percentile) and steps on the music's strong onsets (kernel 75 ms; credit above the local mean credit) | 0.038 credit above chance |

`score = 100 * sum_k weight_k * component_k` (weights in `Config.weights`; `final.weights`). A component that cannot be measured (no body signals, no steps) is left out and the weights renormalised.
The accents are the only **absolute-alignment** component (everything else is phase-free); the stops and hit thresholds follow the dancer's own trailing 60 s, so nothing in a value at t depends on the dancers' motion after t.

### Reference levels (the calibration, and what it does not claim)

`ref` = the **90th percentile, over the first half of the demo (a professional couple, careful alignment) past the first 45 s, pooled over the two roles, of the statistic itself** (`python -m dancecap.rhythm_controls --calibrate`): body 0.476, feet 0.307, couple 0.373,
accents 0.038 (rounded in `EntrainConfig` / `Config`). "100" therefore means *as tightly locked as the demo's best stretches*, not a ceiling of what is possible; a dance can reach it and the number is clipped at 1 per component. The reference is **one professional couple's first half**; the demo's second half, the 38 s span and the heat were not used to set it
(train / test: the heat is out of sample, the demo's second half is held out). The reference levels are the main reason the scale could be moved by another choice (a 75th percentile would raise everything by about 20 %): the **relative** statements (against the baseline and the controls) are the supported ones, the 0..100 scale is a convention.

### The live score

At time `t` the score uses only the dancers' motion up to `t` (a pure function of time: the viewer seeks; `tests/test_rhythm.py::test_the_live_score_is_causal` truncates the take and compares every component to 1e-9; the song's tempo and the grid are the music, known in advance). Each component's live value is `(1 - w) * start + w * s`
with `s` the causally smoothed (8 s) estimate above, `w = 1 - exp(-elapsed / 45 s)` of the song and `start` = a neutral 0.5 at the start of the dance and the number on screen after a DJ seam (the entrainment of the new song starts again). So the **number starts at 50 and settles as the evidence accumulates**.
In the **no-beat stretches** (no song, the DJ seam) nothing is added and the score is frozen; the confidence is 0 (halved in the seam).

**Final score** (`final.score`): the mean of each running component over the beat-covered time past the first 15 s of each song. **Interval** (`final.score_interval_90`): the 5th to 95th percentile of the estimators' *noise floor* (the baseline runs' standard deviation per component, divided by the square root of the number of independent 45 s windows, propagated through the weights)
**united with the sensitivity range** (`final.range_under_uncertainty`): the audio-video clock moved by +-30 ms and, for a two-song heat, the DJ seam handled the other ways (the first / second song's pulses through the mix); the whole analysis is re-run for each. It does not include the unknown validity of the reference levels.

**Confidence** (`final.confidence`, `live.confidence`): grid confidence (beats_merged: the song's audibility; 0.8 where unknown) x the **lock evidence** (the body lock's mean excess over the wrong-tempo baseline, per sample in baseline standard deviations, times the square root of the number of independent windows, divided by 5, kept in [0.25, 1])
x the time seen (sqrt of seen / 45 s) x the A/V term (0.95 without an audio-video sync event: the body locks are a few degrees of phase at +-30 ms). `high >= 0.7`, `medium >= 0.4`, else `low`. **`insufficient_evidence`** (`final`, and `live.evidence < 0.3` along the dance) is set when the lock evidence is under 2 standard errors:
the viewer then greys the number and says "insufficient beat-lock evidence" instead of reading a number into noise.

## Controls (`python -m dancecap.rhythm controls`; `work/carlos_aline/rhythm/controls.json`, `controls.png`)

Final score; the components in the order body, feet, recent, couple, accents; confidence and lock evidence z. "restarted" = the grid cropped to that half, so the averages restart.

| control | score | body | feet | recent | couple | accents | confidence |
|---|---|---|---|---|---|---|---|
| **positive**: demo, whole take | 39.1 | 0.39 | 0.25 | 0.37 | 0.55 | 0.46 | medium, z 2.9 |
| demo, first half (restarted; the reference levels come from here) | 19.1 | 0.20 | 0.13 | 0.20 | 0.07 | 0.37 | low, insufficient |
| demo, second half (restarted; held out) | 32.0 | 0.26 | 0.18 | 0.31 | 0.46 | 0.66 | low, insufficient |
| 38 s reference span (a second pose fit of the same performance) | 14.2 | 0.16 | 0.08 | 0.13 | 0.00 | 0.34 | low, insufficient (38 s: too short for the window) |
| **the heat** (out of sample), whole | **41.4** | 0.48 | 0.33 | 0.44 | 0.58 | 0.02 | medium, z 3.6 |
| heat, first half (restarted) / second half (restarted) | 42.2 / 35.6 | 0.51 / 0.38 | 0.16 / 0.43 | 0.49 / 0.33 | 0.71 / 0.29 | 0.00 / 0.27 | medium / low |
| **negative**: demo against a grid at 0.78 / 0.82 / 1.15 / 1.22 / 1.30 x the tempo (and the music's accents moved 3.3 s) | 18 / 23 / 6 / 9 / 3 | 0.11 / 0.15 / 0.01 / 0.07 / 0.00 | 0.02 / 0.09 / 0.10 / 0.11 / 0.03 | 0.17 / 0.21 / 0.03 / 0.10 / 0.04 | 0.91 / 0.99 / 0.34 / 0.20 / 0.13 | 0.08 / 0.08 / 0.02 / 0.00 / 0.01 | low, insufficient |
| demo against the heat's song grid; against wandering-tempo grids (2 seeds) | 10; 16, 16 | 0.08; 0.05, 0.10 | 0.07; 0.07, 0.07 | 0.10; 0.15, 0.20 | 0.25; 0.93, 0.59 | 0.08; 0.05, 0.06 | low, insufficient |
| heat against grids at 0.78 .. 1.30 x the tempo | 11 / 15 / 5 / 3 / 8 | 0.04 / 0.17 / 0.02 / 0.00 / 0.03 | 0.05 / 0.07 / 0.01 / 0.03 / 0.01 | 0.15 / 0.19 / 0.03 / 0.03 / 0.04 | 0.40 / 0.11 / 0.10 / 0.02 / 0.34 | 0.16 / 0.15 / 0.23 / 0.19 / 0.25 | low, insufficient |
| heat against the demo's song grid; wandering-tempo grids (2 seeds) | 12; 8, 12 | 0.01; 0.02, 0.05 | 0.03; 0.00, 0.04 | 0.08; 0.10, 0.09 | 0.77; 0.35, 0.60 | 0.17; 0.17, 0.15 | low, insufficient |
| **invariance**: demo / heat with the grid shifted by 0.31 s (the score must not change: it is phase-free) | 39.5 / 41.6 | 0.39 / 0.49 | 0.25 / 0.33 | 0.37 / 0.44 | 0.60 / 0.57 | 0.46 / 0.02 | unchanged |
| demo with the motion played backwards (a sinusoidal lock is invariant to time reversal) | 41.9 | 0.45 | 0.31 | 0.39 | 0.76 | 0.15 | medium, z 3.7 |
| heat played backwards (the second song's motion lands on the first song's tempo: a wrong-tempo negative by construction) | 9.4 | 0.01 | 0.10 | 0.07 | 0.47 | 0.13 | low, insufficient |
| **no dance**: the recap (two teachers talking) against a 76 bpm grid (the recap itself has no beat grid: the pipeline refuses it, `no beat grid ... a take without a soundtrack has an undefined rhythm score`) | 18.1 | 0.05 | 0.22 | 0.10 | 0.77 | 0.28 | low, insufficient |

How to read it. The three **music-dependent** components (body, feet, recent) are 0.00 to 0.2 for every wrong-tempo or wrong-song negative (means 0.06, 0.05, 0.11), 0.25 to 0.5 for the professional takes, and the evidence z of the negatives is negative or near 0 (insufficient) while the whole professional takes sit at z 2.9 to 3.7.
**The couple coherence and the accents are not music-referenced in the same way**: the couple coherence is a property of the two bodies (a wrong grid leaves it where it is, and two people talking and shifting their weight together score 0.77), and the accents read the music's onsets, not the grid, so only the moved-music baseline calls them. That is why the composite score of a wrong-grid
negative is 3 to 23, not 0, and why the couple sync carries only 10 %. Random synthetic dancers average 6 (8 seeds, max 13); the wrong-tempo synthetic sway 7 to 11.
The separation (professional whole takes 39 to 42; the 16 negatives 3 to 23, mean 11; no-dance 18 with insufficient evidence) is real but not wide, and a 90 s half of a take is too short to be called evidence (insufficient) in most cases. A professional score of about 40 means "locked about 40 % of the way from the wrong-tempo baseline to the demo's best stretches".

**What the heat's number is.** The heat scores **41.4** (interval 22.7 to 60.1, medium confidence, lock evidence 3.6 standard errors; first half 42.2, second 35.6) against 39.1 for the demo with the same recipe and the same reference levels: **the heat's body entrainment is in the same range as a professional demo's**. Version 1's "5" was the metric.
Both roles lock to the half-bar level (98 % and 100 % of the time), the second song (the DJ mix: 10.8 % faster) as well as the first (the grid restarts at the seam). The footwork reads 0.33 (the follow's touchdowns lock best at 3-in-2: rho 0.32 against 0.20 for the other levels), and the **accents read about 0**: the hits and steps sit on the music's strong onsets no more than chance (hits +0.016 credit, steps -0.024, stops -0.117 against the local mean credit; the demo's hits +0.025, steps +0.017, stops +0.054).

## Inputs (all optional except the first)

| file | what | used for |
|---|---|---|
| `work/<take>/timing/timing.json` | `beats.grid` rows `[t, type, measure, slot]` (8 eighth slots per 4-pulse measure, even slots = pulses), `footsteps.lead / follow` (touchdown `t`, `foot`, `sigma`), `dropped_no_beat.intervals`, `absolute_timing.valid` | the grid, the touchdowns, the no-beat intervals, whether an A/V sync event pins the clocks. **No grid: refused (`ValueError`), the score is undefined** |
| `work/<take>/physics/physics.npz` (else `timing/kinematics.npz`) | the centre of mass and its velocity (else the pelvis) | the body streams |
| `work/<take>/timing/kinematics.npz` | `*_vel`, `*_t` | stops and holds |
| `work/<take>/song/` (two-song heats) | `beats_merged.json` (per-slot confidence, song), `segments.json` (`no_beat`, timeline), `beats_studio_raw.json` (each song's pulses) | the grid's confidence term, no-song stretches, the DJ-seam variants of the sensitivity analysis |
| a soundtrack WAV on the reference clock: `--audio`, else `work/<take>/audio/heat_studio.wav`, `audio/audio_overlap.wav` or `song/studio_reference.wav` (+ its `.json` `reference_start`) | the music | accent events (onset-flux peaks, strength against the local 90th percentile) and breaks. Without it the grid's downbeats stand in for accents (weak: the bar phase may be off) |

A take without a `song/` folder works: the grid's own gaps (more than 1.6 slots) are the DJ-seam / no-song stretches, the grid's ends are "no song", confidence is 0.8 per slot, and there are no seam variants.

## Outputs (`work/<take>/rhythm/`)

* `rhythm.json` (times are reference seconds): `final` (score, components, levels, confidence, `score_interval_90`, `range_under_uncertainty`, `insufficient_evidence`), `final_card`, `describe` (`entrainment`: per role and level the mean phase-locking of body and feet, the share of time each level was chosen, the baseline, the excess in baseline sd, the locked phase;
  `binary_grid`: the old slot-exact view as a *description*; `accent_events`), `steps` (columns + rows: `t, role, foot, valid, measure, slot, kind` (beat / and), `offset_ms` to the nearest eighth slot, `level`, `level_offset_ms` (from the dancer's own locked step phase), `class` (on / near / off / neutral), `lock`, `accent`, `conf`, `state`, `sigma_ms`),
  `measures`, `windows` (the running components), `live` (the 30 Hz causal timeline: `score`, the five components, `confidence`, `evidence`, `state`, the chosen level per role), `no_beat`, `music`, `sensitivity`, `claims` (the plain-language list of what is and is not supported), `config`, `formula`.
* `rhythm_review.png` (body lock vs baseline per role, the chosen level, live score and components, confidence and evidence), `rhythm_measures.png`, `rhythm_sensitivity.png`, `controls.png`.
* `for_unity(doc)` (the viewer's copy, written into the capture as `rhythm.json` and `capture.json "rhythm"`): the live timeline, the valid steps with their class, the final card, the notes.

**Step classes** (the HUD lamps and floor rings): a touchdown is compared with the dancer's *own* locked step phase at the level their steps lock to (the circular mean of their previous steps, 15 s memory; causal): **on** within 100 ms, **near** within 200 ms (both capped at 16 % / 32 % of the level's period), **off** beyond; **neutral** (pale) while the footwork shows no lock against the baseline
(skill under 0.15) or fewer than 3 steps have been seen. So the lamps colour only where a lock exists, and say "the step is where this dancer's steps usually are in the beat", not "the step is wrong".

## The viewer (Unity, `Assets/Overlays/RhythmScore.cs`; no names, no media)

* The capture's `rhythm.json` is loaded by `DanceLayers` (capture.json `"rhythm"`); the layer is **off by default**. Free viewer: `hm_layer rhythm on|off` (CLI), `DanceLayers.SetLayer("rhythm", true)`.
  Directed playback (library films, the recorder): the film direction's segment layer **`"rhythm_score": true`** turns it on and **`"rhythm_rings": true`** colours the touchdown rings on the floor by the step class; without them the demo and the recap are unchanged.
* The panel (top left, level with the zouk beat counter and under the platform's top band; free viewer: top right): "RHYTHM ENTRAINMENT / estimated from motion, not an official score", the big live number (greyed, with the subtitle "insufficient beat-lock evidence (provisional)", while `live.evidence` is under 0.3), "confidence low / medium / high",
  five component bars (the labels and weights come from the file: Body sway 45, Footwork 15, Recent form 20, Couple sync 10, Accents 10), the sparkline of the score so far (grey = paused), a beat pulse, and one lamp per dancer that flashes at each footstep (green / amber / red / pale as above).
  It appears 1.5 s before the first beat, dims to 30 % under a title or chapter card, and **at the end of the dance** (the last 4 s) the **final card** shows the score counting up (greyed with "insufficient beat-lock evidence: the number is not meaningful" when the whole-dance evidence is thin), the components, the confidence, the range and "estimated from motion, not an official score".
* A pure function of time: seeks, scrubbing and the recorder (`hm_film record`) show the same value at the same time. Hidden in VR.

## Film direction (`film_heat.py`, `takes/library/films.toml [<preset>.heat]`)

`avatar_opacity`, `room`, `skeleton_lines`, `rhythm_score` (HUD + final card + rings): unchanged from version 1 (the Carlos + Aline preset reads "avatars at 75 % transparent" literally, `avatar_opacity = 0.25`).

## Reuse for another take (a next Jack & Jill heat)

1. Run the usual stages up to `timing` with the beat grid and the `physics` stage.
2. `python -m dancecap.rhythm run <take>` (add `--audio` if the soundtrack is elsewhere); read `claims`, the plots, `final.insufficient_evidence`.
3. `python -m dancecap.rhythm install <take> --capture <NN_Name>` (or the capture export, which copies it).
4. Set `rhythm_score = true` in the take's `[<preset>.heat]` block and rebuild the film.
5. Compare couples only with the same recipe on takes of comparable length (the evidence and the interval grow with the time seen), and never quote a number without its interval and confidence.

## A second heat: one handheld phone, one song (`john_lula`, 2026-10-11)

Same recipe, same reference levels (out of sample), `python -m dancecap.rhythm run john_lula` (17 s; the inputs: `timing/timing.json` on the heat's own beat grid `song/beats_heat.json`, constant 75.0 BPM, 24 whole bars 3.40 to 80.20 s,
`physics/physics.npz`, the video's own mono audio `audio/audio_overlap.wav` for the accents; `beats_merged.json` has no `song` column, so the grid confidence stays at the default 0.8). The motion is a MONOCULAR fit (the distance from the camera is +-10 %; the body sway is read from the horizontal speed of the centre of mass, whose band-passed lock does not depend on that scale).

| | |
|---|---|
| final score | **57.1**, confidence **medium** (0.60, lock evidence 4.0 baseline sd, 76.4 s of beat-covered time), 90 % noise interval 24.8 to 89.3 |
| components | body sway 0.68, footwork 0.06, recent form 0.64, couple sync 0.96, accents 0.34 |
| levels | both roles lock at the half-bar level (lead 0.48 against 0.34 for the wrong-tempo metronomes, 2.4 baseline sd; follow 0.63 against 0.36, 4.4 sd) |
| the +-30 ms offset | 54.7 to 57.2 (only the accents move; body, feet and couple are phase-free) |
| music-referenced components only (body, footwork, recent form, weights 45 / 15 / 20) | 55 |
| live number | starts at 50, reads "insufficient beat-lock evidence" until **44.7 s** (live evidence below 0.3 for the first 41 s of the music: 49 % of the beat-covered time is out of evidence), 63 at 52 s, 66 at 60 s, then 51 to 54 |
| footwork | 98 touchdowns (lead 72, follow only 26: her steps are small and half hidden), 40 % within 60 ms of a slot against 30 % at random, R 0.08 against 0.10: no step lock (zouk steps are not on every slot), as for the other heat |
| no-beat stretches | the 3.4 s lead-in and the 7.9 s of applause (13 % of the video): the score is frozen there, the final card shows at 84 to 88 s over the applause |

Reading: the sway locks to the song about as strongly as in the professional demo and the other heat (the body component 0.68 against 0.39 and 0.48, the couple coherence saturated at 0.96 as for any two bodies moving together), but the number is a statement about 76 s of one phone's monocular fit:
it settles only in the last 30 s, the interval (25 to 89) is wide, the follow's half-hidden stretches (27 to 32 s, 43 to 46 s, 83 to 88 s) are the pose prior's guesses (flagged low-confidence in the layers, `dancecap.jl_lowconf`), and nothing validates the scale. Not tuned: the first run is the one reported.

## Limits (read before quoting a number)

* **No validated mapping to judges' scores.** There is no ground truth: no judged heat with scores was available, so nothing says that 41 is "good" or that a higher number wins. It is an estimate of motion-music phase-locking against a reference taken from one professional demonstration.
* The 0..100 scale is a convention (the reference level is the 90th percentile of the demo's first half); the interval is the estimators' noise floor plus the clock / seam sensitivity, not the uncertainty of that convention.
* Phase offsets are invisible by design (a steady sway half a beat late scores the same as one on the beat); the accents component is the only absolute-alignment part and is near chance for both couples. The A/V clocks are unpinned (+-30 ms).
* The metric sees what the grid says: a grid that is wrong (a mixed song, a tempo that wanders, a missing seam) lowers the number, and the confidence's grid term only covers the grid's own self-assessment.
* Tempo specificity is +-8 %: a tempo 5 % off still reads as locked (the windows are 45 s).
* The couple coherence is not music-referenced and is saturated by any two bodies moving together (the no-dance control reads 0.77 there); the footwork lock is noisy (three to eight steps per window).
* The touchdown times carry +-0.05 to 0.2 s of detection noise; the body streams do not depend on them.
* A professional score of about 40 and a wrong-tempo baseline of about 10 are close enough that a take shorter than about 90 s cannot be called either (insufficient evidence); the viewer says so.
