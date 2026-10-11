# Rhythm analysis and the live rhythm score (`dancecap/rhythm.py`)

**Rhythm score (estimated from motion, not an official score).** For a Jack & Jill heat (or any take with a beat grid and step events) it
measures how the dancers' footsteps, stops and holds sit in the song's beat grid, scores five things a judge listens for, and replays the
score *live* along the dance. It does not judge technique, partnering or musicality as a whole; it says how sure it is and why.

```
python -m dancecap.rhythm run <take> [--timing P] [--song-dir P] [--audio WAV] [--out DIR] [--no-plots] [--no-sensitivity]
python -m dancecap.rhythm install <take> --capture 07_Name       # put the viewer's copy into the capture folder(s): no full re-export
python -m dancecap.rhythm selftest                               # perfect steps near the top, random steps near the bottom
python -m pytest dancecap/tests/test_rhythm.py -q
```

It runs **locally** (numpy + scipy, about 8 s for a 200 s heat, of which about 4 s are the soundtrack's STFT: the pods rule is for CV / neural /
heavy CPU work; this is neither). Re-run it whenever `timing/timing.json` (or the beat grid) changes. The viewer gets the result through the
capture export (`export_unity.export` copies it; `rhythm install` does only that) and a film direction turns the HUD on (below).

## Inputs (all optional except the first)

| file | what | used for |
|---|---|---|
| `work/<take>/timing/timing.json` | `beats.grid` rows `[t, type, measure, slot]` (8 eighth slots per 4-pulse measure, even slots = pulses), `footsteps.lead / follow` (touchdown `t`, `foot`, `sigma`), `dropped_no_beat.intervals`, `absolute_timing.valid` | the step events, the grid, the no-beat intervals, whether an A/V sync event pins the clocks |
| `work/<take>/timing/kinematics.npz` | `lead_vel / follow_vel`, `*_t` | stops (stillness after motion) and holds on the music's breaks |
| `work/<take>/song/` (two-song heats) | `beats_merged.json` (per-slot confidence, song), `segments.json` (`no_beat`, timeline), `beats_studio_raw.json` (each song's pulses) | the grid's confidence term, no-song stretches, the DJ-seam variants of the sensitivity analysis |
| a soundtrack WAV on the reference clock: `--audio`, else `work/<take>/audio/heat_studio.wav` or `audio/audio_overlap.wav` (+ its `.json` `reference_start`) | the music | accent events (onset-flux peaks, strength against the local 90th percentile) and breaks (level dips). Without it the grid's downbeats stand in for accents (weak: the bar phase may be off) |

A take without a `song/` folder works: the grid's own gaps (more than 1.6 slots) are the DJ-seam / no-song stretches, the grid's ends are "no song", confidence is
0.8 per slot, and there are no seam variants. A take whose timing ran with `--no-beat` already has the intervals in `dropped_no_beat`.

## Outputs (`work/<take>/rhythm/`)

* `rhythm.json` (read it with `json`; times are reference seconds): `final` (score, five components, flow R, chosen metrical levels, confidence, range under uncertainty),
  `final_card`, `describe` (distribution numbers per role, per song, the evidence per metrical level), `steps` (columns + rows: `t, role, foot, valid, measure, slot,
  kind` (beat / and), `offset_ms` to the nearest slot, `level`, `level_offset_ms`, `class` (on / near / off), `lock`, `accent`, `accent_chance`, `conf`, `state`, `sigma_ms`),
  `measures` (a score and the components per measure, steps per pulse, on-share), `windows` (12 s windows every 2 s), `live` (the 30 Hz causal timeline: `score`, `lock`, `accents`, `tempo`,
  `flow`, `sync`, `confidence`, `state`, the chosen level per role), `no_beat`, `music` (accent count, breaks), `sensitivity`, `claims` (the plain-language list of what is and is not supported), `config`, `formula`.
* `rhythm_review.png`, `rhythm_measures.png`, `rhythm_sensitivity.png`: the distribution plots (step offsets, offset over time, live score and components, confidence; per measure and per window scores; the sensitivity bars).
* `for_unity(doc)` (the viewer's copy, written into the capture as `rhythm.json` and `capture.json "rhythm"`): the live timeline, the valid steps with their class, the final card, the notes.

## What is measured

**Step events.** Every touchdown of the timing analysis. A step is *valid* when the grid exists there (not in a no-song stretch or the DJ seam) and a slot is within 0.75 of a slot spacing.
For each valid step: the nearest grid slot (`kind` = **beat** on a pulse, **and** on the eighth between), the signed `offset_ms` (step minus slot), and a class **on** (within 60 ms, or 16 % of the lattice period if that is shorter), **near**
(within 120 ms, or 32 % of the period), **off**. A 60 ms window is 32 % of a 371 ms eighth: random steps are *on* 32 % of the time (the table prints that chance next to the measured share).

**Metrical levels.** Zouk dancers step on the pulse or the "and", but a step pattern in 3-against-2 is as much in time with the music, and the plain grid cannot see it. So each role's steps are also compared with
the lattices of the **pulse**, the **eighth** (pulses and ands), **3 steps in 2 pulses** (two phases: which pulses the lattice starts on) and **3 per pulse**, all anchored on the grid's pulses. A role is scored against the
level its own steps fit best, *with a price for choosing*: the skill of every level is reduced by `1.35 * sd / sqrt(n_eff)` (1.35 = the expected maximum of five standard normals), so a role that locks to nothing
scores nothing and the chosen level is only ever a lock that stands out of the noise. The class and the HUD flash of a step are against the level its role had chosen *before* the step (causal); with no level standing out they fall back to the plain beat / "and" grid.

**Credits.** All kernels are Gaussian credits `exp(-d^2 / (2 sigma^2))`, `sigma = 60 ms` (the A/V uncertainty and the touchdown detection sit inside it). `chance` of a credit is what a uniformly random phase scores
(`lock_chance`, analytic: 0.405 for the eighth, 0.20 for the pulse, 0.30 for 3-in-2, 0.61 for 3 per pulse); `ideal` is what a perfect dancer with 30 ms rms of timing noise scores (0.894). Every component is
`skill = clip((mean credit - mean chance) / (ideal - mean chance), 0, 1)`: **0 = no better than random, 1 = as good as perfect stepping can be measured**. Random steps score about 2 of 100; perfect steps about 99 (`selftest`).

| component | weight | what | notes |
|---|---|---|---|
| **Beat lock** | 40 | credit of each step against its role's chosen level (above) | the two roles pool, each with its own level and its own price of choosing |
| **Accents** | 20 | accent hits (steps and stops landing on the music's strong onsets, kernel 50 ms, strength = onset peak / local 90th percentile, only strength >= 0.5) plus **holds** on musical breaks (the body still, 25 % of the component once a break has been seen) | chance = the local mean of the credit curve (+-4 s), so a dense stretch of accents does not flatter. Ideal 0.75 (no dancer hits only the strongest accents). Stops = the body speed under 0.4 x the dancer's median for 0.3 s after movement. Breaks = the level 6 dB under its local median for 0.25 s |
| **Tempo adherence** | 15 | consecutive steps of a role being a whole number of the chosen level's periods apart (steps per beat per measure are in the measure table) | offset-invariant: a steady lag does not hurt it |
| **Flow** | 15 | the steadiness of the step phase about the chosen lattice: the bias-corrected mean resultant length R of the phase (circular statistics; random phases give 0 after the correction), times a penalty `exp(-0.5 (mean offset / 90 ms)^2)` | steady but half a slot off is not flow; a small constant offset (inside the A/V uncertainty) is |
| **Couple synchrony** | 10 | each touchdown against the partner's nearest touchdown (kernel 60 ms), against the chance of a Poisson partner at the local step rate | ideal 0.8; known 0.15 s after the later step |

`score = 100 * sum_k weight_k * component_k` (weights in `Config.weights`; `final.weights`). A role or component with no events is neutral (0.5).

**The live score** (30 Hz, `live.score`). The score at time t uses only events that have *happened* by t (a step is known 0.05 s after its touchdown; synchrony 0.15 s after the later step): a pure function of time that a viewer can scrub back and forth.
Each component = 0.5 x the whole dance so far + 0.5 x an exponential memory (half-life 10 s), each starting from a neutral prior of 0.5 worth 4 events (the memory's prior fades with the memory), so the number **starts at 50**, falls or rises as steps come, and responds to the last
bars of the dance as well as to its history. The memory clock stands still in the **no-beat stretches** (no song, the DJ seam): **the score is frozen there**, the confidence is 0, and the HUD says "score paused". Per-measure and per-window scores are the descriptive, non-causal versions of the same formula (no prior memory).
The **final score** is the whole-dance aggregate (equal weight on all events, no prior); the live number at the end is within a few points of it by construction (it is half of the same aggregate).

**Confidence** (`final.confidence`, `live.confidence`): the product of the beat grid's own per-slot confidence (beats_merged: the song's audibility; 0.8 where unknown), the share of steps seen (about 8 steps for the first third; 32 for full), the **lock evidence** (the z of the best level's excess over chance, after the price of choosing, divided by 3, kept in [0.25, 1]:
"is there any detectable lock at all?"), and the A/V term (0.85 without an audio-video sync event: absolute on-the-beat claims carry +-30 ms). It is 0 in a no-song stretch and halved in the DJ seam. `high >= 0.7`, `medium >= 0.4`, else `low`. When the evidence factor is at its floor the lock components measure noise and the claims list says so.

**Sensitivity** (`sensitivity`): the final score with the motion moved +-30 ms against the music (the unpinned A/V offset) and, for a two-song heat, with the DJ seam handled the other ways (the first song's pulses through the whole mix, or the second song's from the start of the mix) instead of frozen; the range is printed on the HUD's final card.

## The viewer (Unity, `Assets/Overlays/RhythmScore.cs`; no names, no media)

* The capture's `rhythm.json` is loaded by `DanceLayers` (capture.json `"rhythm"`); the layer is **off by default**. Free viewer: `hm_layer rhythm on|off` (CLI), `DanceLayers.SetLayer("rhythm", true)`.
  Directed playback (library films, the recorder): the film direction's segment layer **`"rhythm_score": true`** turns it on and **`"rhythm_rings": true`** colours the touchdown rings on the floor by the step class; without them the demo and the recap are unchanged.
* The panel: the big live number (a smooth count: the score smoothed over the preceding 0.2 s, still a pure function of time), "confidence low / medium / high", five component bars, the sparkline of the score so far over the whole dance (grey = paused), a beat pulse (bright on each pulse of the grid), and one lamp per dancer that flashes at each footstep: **green on the beat, amber near, red off** (and the same colours on the floor rings at the foot contact).
  Film placement: top left, level with the zouk beat counter, under the platform's top band (vertical 1080 x 1920: x 26 px, below 14 % of the height; landscape 1920 x 1080: below 5 %); the bottom bands stay free. Free viewer: top right.
  It appears 1.5 s before the first beat, dims to 30 % under a title or chapter card, and **at the end of the dance** (the last 4 s) the **final score card** shows the score counting up, the five components, the confidence, the range under the uncertainty and "estimated from motion, not an official score".
* A pure function of time: seeks, scrubbing and the recorder (`hm_film record`) show the same value at the same time. Hidden in VR (`FilmDirector.HeadIsCamera`: the head is the camera, a flat overlay has no place there; a world-space panel is future work).

## Film direction (`film_heat.py`, `takes/library/films.toml [<preset>.heat]`)

`avatar_opacity` (the dancers' OPACITY, one number at the top of the direction), `room` (the room mesh), `skeleton_lines` (stroke factor), `rhythm_score` (HUD + final card + rings). The Carlos + Aline preset reads the owner's "avatars at 75 % transparent" literally
(`avatar_opacity = 0.25`); the other reading, 75 % opaque, is `0.75`: change the number and rebuild (`python -m dancecap.film_library build --carlos-aline`), or edit `"avatar_opacity"` in `direction.json`.

## Reuse for another take (a next Jack & Jill heat)

1. Run the usual stages up to `timing` with the beat grid (`python -m dancecap timing <take> --beats ... [--no-beat ...]`).
2. `python -m dancecap.rhythm run <take>` (add `--audio` if the soundtrack is elsewhere); read `claims` and the plots.
3. Export / install: `export_unity.export` already copies `rhythm/rhythm.json`; or `python -m dancecap.rhythm install <take> --capture <NN_Name>`.
4. Set `rhythm_score = true` in the take's `[<preset>.heat]` block (and the looks you want) and rebuild the film.

## What the score does not support (read before quoting a number)

* It is an estimate from detected touchdowns (about 15 ms measurement noise) against a grid with +-30 ms absolute uncertainty, **not** an official score and not a comparison between couples: the same recipe on two heats is comparable, a number alone is not.
* When the steps do not sit detectably on the grid the lock components are near 0 and the claims say so: that is a statement about the data (many zouk steps are syncopated, the music may be danced in a swung / 3-against-2 feel, the grid's bar phase is untrusted), not a verdict on the dancing.
* The bar phase (which pulse is "1") is not used by any component: accent hits come from the audio's onsets, not from downbeats, unless there is no soundtrack at all.
* The DJ-seam stretch of a two-song heat has no common grid: it is frozen, and the sensitivity table shows what handling it the other ways would do.
