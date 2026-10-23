# The class-recap film: a directed tutorial built from the teachers' own transcript

Written 2026-10-09 (agent D1). Code: `dancecap/film_recap.py` (build, validation, review table, CLI), `film_recap_plan.py` (the directing rules),
`film_recap_captions.py` (captions, emoji image), `film_recap_audio.py` (class mix, film soundtrack). Tests: `tests/test_film_recap.py`,
`tests/test_film_recap_audio.py`. Everything the build writes is under `work/larissa_kadu_recap/film/` (git-ignored: it holds the teachers' words and voices).

The film is the recap of one six-minute class: the teachers talk and now and then demonstrate. There is no soundtrack and no narrator: the SOUND is the
class audio, the SPEECH is the director. Vertical 1080x1920 is the primary composition (16:9 comes from the same direction).

## Run it

    python -m dancecap.film_recap build [--layers auto|assume|none|all] [--capture 06_LarissaKaduRecap] [--refresh-speech auto|yes|no] [--no-mix]
    python -m dancecap.film_recap mix class        # the class audio mix, heavy (about 3 minutes, local CPU), once; `build` then derives the film soundtrack
    python -m dancecap.film_recap check            # validate the written direction against the transcript
    python -m dancecap.film_recap emoji            # film/emoji/1f602.png from the machine's colour-emoji font
    python -m dancecap.film_recap chapters         # the [[recap.chapters]] block of takes/library/films.toml
    python -m dancecap.film_recap_audio report     # before / after levels, spectrograms, word timing

`build` is the ONE command to repeat after anything changed. After the owner's opening answer (`speech/owner_opening.json`) it needs nothing else: with
`--refresh-speech auto` (default) it runs `python -m dancecap.speech final --no-clips` first when the answer is newer than `transcript_final.json`, then
rebuilds direction, captions, chapters, review table and film soundtrack. After the recap capture and its analysis files exist, the same command probes
them (see Layers) and re-plans. Then run the exporter of the phones' video for the direction (`python -m dancecap.camera_video larissa_kadu_recap
--capture 06_LarissaKaduRecap`) if the opening should fly into phone 01's own video (see Opener), and `python -m dancecap.film_library build --recap`.

## Inputs and outputs

| input | what for |
|---|---|
| `speech/transcript_final.json` | utterances (speaker, sentence times, words): the sentence is the unit of the shots |
| `speech/teaching_events.json` | what a teacher refers to (body part, weight, axis, path, steps, timing, enumerations, move names), proposed tags, who (leader / follower / couple); roles Kadu = leader |
| `speech/chapters.json` | the 13 chapters (title <= 5 words), contiguous 0 .. 360.48 s |
| `speech/captions.json` | the words as the film shows them (text, capitals, punctuation, speaker key and colour, the laugh as `kind: emoji`) |
| `people/track3d.npz` (+ `roles.npz`) | the two teachers triangulated in 3D on the 30 Hz grid: how far apart they stand and how fast they move (the activity signal; replaced by nothing else until the capture's own analysis is a better source) |
| the capture folder `Assets/StreamingAssets/<capture>/` | which analysis layers exist: `physics.json`, `counterbalance.json`, `floorcraft.json`, `timing.json`, `moves/graph.json`, `cameras/*.track.json` |
| `audio/class_mix.wav` | the soundtrack (master clock) |

| output | |
|---|---|
| `direction.json` | name `larissa_kadu_recap`, capture `06_LarissaKaduRecap`; segments with film times, dance (= master) times, camera, layers, call-outs, chapter cards; schema = the demo film's, additive (VIEWER_SPEC 3.15) |
| `narration/timeline.json` | the captions, vertical (<= 2 lines of 24 characters) and horizontal (1 line of 62), speaker colours, the emoji word; `direction` = the direction's name, which is how the director finds it |
| `chapters.json`, `direction_report.json`, `direction_review.md` | chapter marks (film seconds); what was used / dropped / decided; what the teachers say and what the film shows, per shot and per second |
| `audio/<name>_mix.wav`, `audio/make_film_mix.py` | the film-clock soundtrack; the recorder's hook that rebuilds it from the direction |
| `emoji/1f602.png` | the laugh emoji for the captions |

Time: DANCE time = master seconds (the clock of the master phone = the capture's time base, anchor 0, 30 Hz); FILM time = seconds into the video. They are
equal except after an insert (below), when the film is later by the inserted seconds.

## What the camera and the overlays do

The unit of direction is the **sentence**. A shot starts 0.3 s before the sentence it illustrates and holds until the sentence ends; the cut is a camera
blend of 0.7 s that has settled when the words begin. There is no shot change inside a sentence unless the sentence is longer than 11 s and carries two
different teaching points (it then moves at the clause break between them, marked `demanded` and validated). A treatment other than talking holds at
most 12.5 s, then the camera changes angle at the next sentence start (alternating direction / height). Shots shorter than 3.2 s join a neighbour.

Treatments, chosen per sentence from the teaching events inside it (score = sum of tag weight x confidence; below 1.8 the sentence is just talk):

| the teacher talks about | camera | layers / call-outs |
|---|---|---|
| nothing special; standing apart | `orbit` 4.2 m (4.8 apart), slow, leaning 35 % toward who is talking, `fit_span` | avatars 0.55, no overlays: calm |
| a body part (frame, shoulder blades, arm, hip, chest, connection ...) | closer `orbit` 3.1 m, focus on the dancer named | spine beads; a 3D arrow with the teacher's word on the joint |
| weight, counterbalance, heavy, balance, suspension | `physics_3q` | physics skeleton (estimated load), counterbalance, centre-of-mass markers, a call-out on the couple's centre of mass |
| axis, exit, rotation, pivot | `overhead_floorcraft` (couple / whole floor alternating) | the leader's T, floor record |
| path, direction, space, side to side | `overhead_floorcraft` floor | dotted path, dials |
| feet, legs, steps | `feet_closeup` (both dancers, waist down) | footprints; "right foot" arrow when the foot is named |
| momentum, speed, stop, at the same time | `rhythm` (medium close) | touchdown rings, neutral (no beat colours: no soundtrack) |
| a move name, an enumeration | whatever the sentence has | a screen readout in the teachers' words ("Second exit"), pt-BR names from ZOUK_TERMINOLOGY.md ("Corredor" for "lateral") |

Rules that keep it from being busy:
- **Motion families need motion.** Feet, axis, path and timing are only used when the teachers actually move in the window (activity: faster than 0.25 m/s for at
  least 20 % of it); weight needs a little movement; a body-part call-out may illustrate a still posture. Teachers standing more than 0.95 m apart get the calm shot.
- **One call-out at a time** (at most 3 arrows per 10 s), a word is labelled once per 40 s, only events with confidence >= 0.6 that no editor flagged.
- **Dancers always in frame.** Every orbit carries `fit_span` (Unity: the lens widens, then the eye backs off, until both pelvises plus an arm's margin sit inside
  90 % of the width; it never narrows). The teachers stand apart 1.2-2.1 m at 0-20 s, 255-262 s, 333-340 s and 354-360 s of this class.
- **Chapter cards** at every chapter start (a pill with the number and the title, 2.6 s; the first is the title card "Class recap" with the teachers' names).
- **Captions** are the teachers' words, one speaker's phrase per chunk, the speaker's colour (coral / blue from `captions.json`) and a small name tag; the laugh
  emoji is a caption word of its own at the two jokes. Unsure words (the transcript marks them) are NOT dimmed; `direction_review.md` lists the captions to proofread first.

Layers (`--layers auto`): the capture folder is probed. While it does not exist the PLANNED set (physics, counterbalance, floorcraft, timing) is assumed so the
direction can be reviewed now (`data_status` says PLAN and `layers_assumed` is true). With a capture, a missing file drops what needs it: a treatment falls back
to a calmer one, a layer key or a call-out is removed, and every drop is listed in `direction_report.json` and at the end of `direction_review.md`. There is no
dance graph for this class (no soundtrack, no move recognition): move names appear as readouts only.

Opener: when the exporter wrote `cameras/01.track.json` into the capture the first shot flies into phone 01's own video (the real people for the opening jokes)
and fades back to 3D; otherwise it is the calm orbit. Not needed for the film to work.

## Slow motion, replays and the sound (design, implemented, tested)

The teachers' voices are only ever heard at 1.0x, locked to their picture. So a slowed passage is allowed in exactly two places and never plays class sound:
1. **`gap_slow`**: inside a speech-free gap of at least 1.8 s (0.35 s kept clear of the neighbouring words), the picture runs at 0.5x while the room-tone bed plays;
2. **`replay`**: at a sentence end the speech is PARKED: a replay of the demonstration (at most 3.6 s of dance at 0.5x = 7.2 s of film, a white-flash cut, the
   "replay - slow motion" pill) runs over the bed, then the film returns to the master time where the speech stopped. The words after it are later by the inserted seconds.

Only when a teacher ASKS to look ("see", "watch", "look at it / this", "slowly", "one more time") and a demonstration burst (the teachers move faster than
0.45 m/s for at least 1.6 s) follows within 6.5 s; at most 4 inserts, 24 s in total, 25 s apart. A plain "again" ("starting here again") describes the dance, it is not a
request, and neither is "slowing down" (dance advice): both off by default (`Cfg.allow_again_cue`, `allow_slowing_cue`). In this class one passage qualifies
("See so he start pivoting and then went", 240.5 s: the strongest burst of the class, 1.0 m/s): one replay, +7.2 s. The film is 367.7 s (6:07.7) for 360.5 s of class.

The film soundtrack (`film_mix`): a LOCKED segment (speed 1, no replay) copies the class mix at its dance time sample for sample; any other segment plays the room-tone
bed, junctions cross-faded over 0.15 s; fades follow the picture's (0.12 s in, 0.9 s out). The bed (S8 fix: until then the "quietest 6 s" fallback looped a stretch
that had a teacher's words in it, so the replay played speech at -24 dB) is, in this order, the speech-free `bed` stem of the chosen class mix (`audio/stems_regen/bed.wav`
for the regenerated mix; the steadiest 6 s, looped with a raised-cosine self cross-fade, unity gain), else a loop of a stretch of the class mix that has no word within 0.5 s
(-3 dB), else silence; never a stretch that has speech. `python -m dancecap.film_recap_audio slow-check` lists what the rendered mix plays in every slowed / held / replayed
segment and exits 1 when one is louder than -40 dB RMS (the replay now plays -49 dB). The viewer's slow motion (the bar's 0.5x, the VR panel's 0.25x ...) is a separate
rule: the recap is a `dialogue` lesson, so the viewer plays no sound at all below 1x (head-movement/docs/FILM_LIBRARY.md section 7). Checked by `validate`: a locked segment's film-minus-dance offset equals the inserts before it (so speech and picture cannot drift), a
non-locked segment must use the bed, a slowed segment must not contain a word, every caption word stands at its master time plus the same offset.

## The class audio mix (`audio/class_mix.wav`)

48 kHz stereo, sample 0 = master 0, length = the master video (17 302 886 samples = 360.477 s). Method and what was measured on the real class (`audio/class_mix.json`,
`class_mix_report.json`, `class_mix_spectrograms.png`):
1. **Clocks**: every phone is put on the master clock with the sync stage's clock (`reference = slope x source + offset`; 00 and 03 are affine) by a block-wise
   fractional-delay warp (Hann windows, FFT phase ramp): error under 1e-3 of full scale at the worst drift (tested against an analytic signal).
2. **Base = the master phone** (the picture's own sound). Another phone is considered only where it is CLEARLY better for at least 10 s: speech-band SNR proxy (90th
   percentile of 50 ms frame levels minus the 10th percentile over +-12 s) plus a bonus for resembling the room's envelope; it must beat the master by 3 dB. 4 stretches
   qualified (05: 88-108, 132-144, 228-252 s; 02: 204-216 s; margins 2.6-3.3 dB: the six phones are all similar, SNR proxy 16.7-19.7 dB).
3. **Phase-aligned splices, only when confirmed**: the lag of the incoming phone against the master is measured on the stretch with the sync stage's own estimator
   (`recap_sync.estimate_delay`: GCC-PHAT in the speech band, Hann ramps, accepted at z >= 6) in 8 s windows; the splice is made only when at least two windows agree within
   1 ms and the lag is under 2 ms. Result: **one stretch used** (02, 204-216 s: lag -0.44 ms, MAD 0.13 ms, z 7.1, 2/2 windows); the three stretches of phone 05 were NOT used (z 3.7-4.9,
   the alignment cannot be confirmed; its clock's reported uncertainty is +-16 ms, and a word measured in one of them started 60 ms early against the master). The cut sits at the quietest instant of the
   master near the planned boundary, level-matched per stretch, raised-cosine cross-fade 0.12 s. Findings worth keeping (also in the docstring of `xcorr_lag`): a plain cross-correlation of voiced
   speech picked lags of +-60 ms (equal peaks at every pitch period); an UNTAPERED PHAT shows a spurious peak at lag 0 (z 20) for ANY two signals, which looked like "the clocks are exact to
   0.1 ms" and is not; R1's tapered estimator gives the honest numbers (the phones agree to +-1-2 ms where z >= 6).
4. **Room noise**: ffmpeg `afftdn` (tracking, 10 dB) after a 70 Hz high-pass; no compressor. **Finding**: afftdn delays the audio by 24.94 ms (1197 samples). The first version of the mix was
   therefore 25 ms late against the picture; the word-onset check (below) showed it (mix 20-30 ms later than the master phone's own audio, global activity lag 68 ms against 43 ms), `remove_latency`
   now measures it by cross-correlation and removes it.
5. **Loudness** -16.0 LUFS integrated, **true peak -1.4 dBTP** (target <= -1), sample peak -1.4 dBFS, loudness range 8.8 -> 9.1 LU (laughter and dynamics kept): a static gain of +9.8 dB (the phones are quiet:
   -25.2 LUFS) and a look-ahead peak limiter (+-20 ms Hann dip, never overshoots by construction). The limiter works on 2.6 % of the samples, at most 10.8 dB, in 17 of the 360 seconds
   (the loud first minute - laughter, the headset voice - and a few later peaks); everything else is untouched. The noise floor in the speech band moves from -46.1 to -38.0 dB while the speech
   peaks move from -26.8 to -17.2 dB (SNR proxy 19.3 -> 20.8 dB): gentle by design; above 3 kHz the gaps are cleaner (spectrogram).
6. **Timing check** (`word_timing_check`, 10 words that start after a pause of >= 0.4 s, spread over the class): the time the speech-band level rises 8 dB above the pause, in ms after the
   transcript's word start, in the mix and in the master phone's own audio: the two agree within 10 ms for all 10 words (differences 0, 0, 0, 0, 0, 0, -10, 0, -10, -10 ms; the 10 ms is the frame);
   the median onset is 80 ms after the transcript's start (ASR word starts lead the acoustic rise); the global lag of the whole class's speech activity is 42.9 ms in the mix and 42.7 ms in the master
   phone. The film soundtrack (`word_timing_film_mix`): the same words at their film times (+7.2 s after the replay) show the same onsets (30 / 80 / 70 ms).
No third-party voice or speech service is involved at any step; nothing is uploaded; the voices stay on this machine.

## Emoji captions in Unity

`FilmEmoji` (Assets/Film): a caption word `{kind: "emoji", emoji: "1f602"}` is an image: `<film folder>/emoji/1f602.png` (made by `film_recap emoji` from the machine's own
colour-emoji font, Segoe UI Emoji here; git-ignored, never committed), else `<StreamingAssets>/emoji/<hex>.png`, else the built-in drawing of U+1F602 in the C# file (own
artwork under the project's MIT licence). So the public repository carries no third-party emoji art and no binary asset; a Noto Color Emoji (OFL / Apache) or Twemoji
(CC-BY 4.0, attribution required) PNG dropped into the same folder would be used the same way. Verified with the viewer (`hm_film_show --action shots`) AND the frame-locked
recorder (`hm_film record --aspect vertical --start 1 --duration 3`, 90 frames, the picture and the layout are identical): `review/` in the film folder.

## Chapters and the library

`chapters.json` has the 13 chapters (titles from `speech/chapters.json`, <= 5 words). `takes/library/films.toml` carries the `[[recap.chapters]]` block anchored at the first
segment of each chapter (`chNN_01_open`, stable across rebuilds). The recap card stays "Processing" until its capture folder, its direction and a soundtrack exist.

## Limits and open questions

- (T3, 2026-10-09) The capture now exists and `build` probes it (physics, counterbalance, floorcraft, timing present; only the dance graph is dropped, 3 sentences). The stills of the whole film were
  reviewed on the real capture and the camera rules below were tuned from them (`work/larissa_kadu_recap/film/PREVIEW_REVIEW.md`).
- The first ~8 s (opening jokes) are the owner's words: the transcript carries the hint-based reading; the film shows it, and `build --refresh-speech auto` applies the owner's answer.
- Which phones are better is judged by a level-contour proxy, not by intelligibility; the replaced stretches are few and small. 05's clock is the weakest.
- The 16:9 aspect uses the same shots; since R10 (2026-10-11) the camera blocks carry opt-in `horizontal` overrides (`plan.with_horizontal`: physics band / distance, a parallax-aware
  closer overhead, a narrower talk / body lens, the finale band, the feet shot) and the horizontal preset the phone-clip switches (`phone_side_fill`, `portrait_zoom`, `safe.side_frac`,
  `legend_scale`; `presets_for`); the 9:16 render ignores all of them (`build --no-horizontal-overrides` omits them). See the viewer spec 3.15.

## Camera rules fixed after looking at the real stills (T3)

- **Feet shot**: low and wide (fov 60, 3.6 m, look at 0.6-0.7 m): both teachers whole, the feet in the lower half, never a frame that cuts the heads or makes the hands the biggest thing in the picture.
- **Overhead**: always the `couple` framing with `across_m` (2.2 / 2.8 m; the replay at least 2.8 m); the viewer widens it by the actual spread of the four feet. The `floor` framing fits the foot area of the whole take
  (25 m up in this class: a speck) and `full_record` drew every T so far: neither is used. The path view uses `current_t` (the faint record stays under it).
- **Calm orbit**: 3.6 m when the teachers stand together (`Cfg.talk_radius_m`), 4.8 m and `fit_span` when they stand apart.
- **Physics legend**: enlarged 1.5x in 9:16 and moved below a chapter / title card while it shows (Unity: `SkeletonLegend.Scale`, `FilmOverlay`).
- The capture is exported with `python -m dancecap.recap_export larissa_kadu_recap --capture 06_LarissaKaduRecap --avatars --audio` (`--audio` = `film/audio/class_mix.wav` as the capture's `audio.wav`, sample 0 = capture 0 s).

## Director mode (D2, owner 2026-10-10): the phones' own video, feature zooms, closer overhead

Code: `dancecap/film_recap_director.py` (the shot scheduler), `dancecap/recap_pov.py` (which phone can be shown when, the crop, the clips, the video
offsets), `dancecap/recap_overhead_check.py` (how much of the frame the couple fills); tests `tests/test_film_recap_director.py`, `tests/test_recap_pov.py`.
`python -m dancecap.film_recap build` runs it when the phones' people / camera files exist (`--director auto|yes|no`); then export the clips the direction
needs: `python -m dancecap.recap_pov clips` (about 4 minutes, a few hundred MB in the git-ignored capture folder), under the Unity lock.

The owner's words: "in director mode use a lot more camera footage pov in the lesson, and zoom to body features in 3d model or floor features or physics when
attention is called. keep circling through camera POVs as the shots cycle" and "for overhead views as a general rule, get closer with the camera". The phone
videos show students; the owner wants them anyway, so the plan frames the TEACHERS (a crop into the phone's frame) and cuts away from wide views quickly.

How the shots are made:
1. **Attention calls** (a teaching event that names a body part, the weight / balance, the axis / path / steps, the timing: 107 in this class) each get a 3D
   window (0.6 s before the word to 1.9 s after; neighbours of one treatment join, up to 6.5 s). The treatment comes from the call itself (`plan.decide_unit`):
   a tight `feature_zoom` on the named joints of BOTH dancers (frame, shoulders, arm, hip, chest, upper body, connection, legs; hands are shown as the frame:
   no hand close-ups), the closer physics view with the centre-of-mass markers, the overhead with the leader's T or the path, the feet close-up, the touchdown
   rings. A body part called again within 8 s of its zoom is shown by the phone with the arrow instead (a phone with an accurate pose: 00-03).
2. **Free time** between the windows is cut at the speech's own pauses into shots of about 3 to 6.4 s that alternate between a phone's point of view and a calm 3D
   orbit (never more than two phone shots in a row); a teacher asking to look ("do this", "like this") prefers the phone. The film opens in 3D.
3. If the phones carry less than 38 % of the film, long zooms hand their tail to a phone and calm shots become phone shots (a run of more than two phone shots gets a 3D shot in its middle: 39 % in this class).
4. **Circling**: each phone shot takes the phone used longest ago among those that see both teachers well (teachers' boxes from `people/teachers_<cam>.npz`:
   both detected or the embrace's merged box, inside the frame, the 9:16 strip of a landscape phone must hold both), never the phone of the previous shot,
   never inside a bad stretch (`recap_pov.BAD`: 01 at 9-30 and 327.5-334.5 s, 02 at 242-278 and 350-356 s, 05 at 203.5-207.5 s, 06 at 152-208 s, plus
   `chunks/bad_camera_stretches.json`, each padded 1 s; the "02 wrong in 0-35 s" claim is not supported by NOTES_T1 and is not used), never in a slow-motion insert.
5. **Crop**: per shot the centre and the zoom that put the two teachers' box at about half the view height, 40 % from the top (captions own the lower third),
   smoothed, inside the frame (2 % rim kept: the undistorted video has black corner wedges), zoom at least 1.07 and capped by the source resolution (3.4x up to the
   1920 px output). The viewer keeps the phone's position and turns / narrows the lens (VIEWER_SPEC 3.15): the skeleton stays registered.
6. A phone shot cuts in (`camera_tour`) or, every other long one after a 3D shot, flies in (`fly_to_camera`: 0.8 s flight, the dancers fade out as the video fades in).

Video offsets: the phones' own video clocks were measured per chunk by `dancecap.sync` (`sync_chunks/table.md`); `recap_pov offsets` turns each chunk's value
(at the centre of the part of the class it owns, from `stitch.json`) into a piecewise delta per phone (`camera_video_offsets.json`: 00 about +108 ms, 02 +19,
03 +25, 05 +7, 06 +14, master 01 = 0) and `camera_video --offsets` applies it to the poses, the clip frames and the teachers' boxes (the same
`corrected t = t + delta` the poses of the fit used). `camera_video --clip CAM T0 T1` may now repeat for one phone.

Overhead: `fill` 0.68 / 0.62 (the share of the frame width the box of both dancers' joints plus a body's radius fills at the tightest), `min_fill` 0.46; never cropped
(VIEWER_SPEC 3.15). Audit fixes of the first render: avatars 0.95 / 0.92 / 0.85 instead of 0.55 / 0.42 / 0.30 (they read as grey ghosts); the physics legend has a fixed
place, 1.85x; an orbit that runs along the line between the dancers lifts the eye (`avoid_inline`); same-speaker caption chunks closer than 1 s stay up (no flicker);
the chapter marks start at the first sentence of the chapter (`offset_s` in `films.toml`), not at the shot's 0.3 s lead-in.

## D4 (owner 2026-10-11): more physics, more arrows, more phone video, avatars 75 % transparent

The owner's words: "the lesson needs to have more physics, more arrows, more POV from phone video, only use regenerated elevenlabs lines, output to
vertical recording. when avatars are in view, we need to set to 75% transparency." Code: `film_recap_arrows.py` (new), `film_recap_director.py`,
`film_recap_plan.py` (`Cfg`, `treatment_layers`), `film_recap.py`; tests `tests/test_film_recap_arrows.py`. The audio is not part of this (V3).

* **Avatar opacity: ONE value.** `plan.Cfg.avatar_opacity = 0.25` (75 % transparent, read literally; 0.75 would be the other reading, 25 % transparent) replaces
  every positive `avatars` value of the presets (3D orbit, zooms, feet, physics, overhead, the flight into a phone, the replay) and is written to the
  direction's top-level `avatar_opacity` key, which the viewer applies to every segment that shows avatars. Flip it without code:
  `python -m dancecap.film_recap build --avatar-opacity 0.75`. The skeleton lines, the physics colours and the overlays keep their own strength (only the
  bodies fade). On a phone's own video there are no avatars (only the skeleton lines sit on the people); `--avatar-on-video` keeps them there too (layer
  `avatars_over_video`).
* **Physics view.** Every teaching event about weight / balance / counterbalance / heavy / suspension / shift (tag `overlay_weight_change`) and about
  momentum / speed / energy (`overlay_timing`), and every spoken word of `director.PHYSICS_WORDS` (weight, balance, momentum, force, pressure, engaged,
  counterbalance, fall, push, pull, heavy, suspension, energy, speed, gravity) that no event covers, gets the physics view (`physics_calls`: the events are
  flagged `physics_call`, the other words become synthetic events that are never counted as the teachers' attention calls). Where the teachers stand
  apart (more than 0.95 m) there is nothing to show and it is skipped. While the teachers MOVE a physics view shows at least every `phys_gap_s` (20 s) of moving
  film time (`physics_cadence`); stretches where they stand and talk do not count. A physics view holds about 4 s (`trim_physics`), the rest of its sentence goes to
  the phones. The legend keeps D2's fixed place and says "estimated from motion".
* **Arrows.** `film_recap_arrows.make_arrows`: one labelled arrow for every attention call that has something to point at, in the teacher's own word, from
  0.35 s before the word for 1.3-2.4 s (shorter when the next one needs the room), at most two call-outs at once (the validator checks it). It points at:
  the joint of a body part; the named foot (else `stance_foot`); the centre of mass of the dancer or of the couple for weight / balance / push; the leader's T
  alive at that time for axis / exit / rotation; the direction of travel (`vector_arrow` velocity) for path / direction / timing; a floor ring for "here".
  Calls with nothing to point at ("look at it", "like this") and the four calls the transcript flags as unsure get none and are listed with the reason
  (`direction_review.md`, `direction_report.json` director.arrows). The physics view adds the velocity of each centre of mass the whole view long (unlabelled,
  explained once by a readout), the net push for push / pull / force words ("est.") and the weight moving from one foot to the other where the physics contact
  changes (`weight_transfers`, "his weight to the right foot"). Arrows lie in the phone shots too (the viewer draws them through the phone's camera, so they
  are registered on the video exactly as the skeleton is).
* **Phone video.** `DirCfg.pov_target = (0.56, 0.62)`, up to three phone shots in a row (`max_pov_run`, `chain_pov`), a phone is never repeated back to back; when
  the free time is not enough, body / feet / path / timing zooms (then the axis ones, then the first body zooms) hand their call to a phone with the arrow
  (`zoom_to_phone`); the physics views never do. Shots stay 3-6.8 s and end at the speech's pauses; the bad stretches and the teachers-in-frame crops of D2 are
  unchanged. Export the clips after a plan change: `python -m dancecap.recap_pov clips` (then `camera_video --install-only` when the editor owns the capture folder).

## The film with the LHM++ splat dancers (2026-10-20)

Owner: "I need the larissa kadu recap to look LHM++ quality with swinging hair". The same direction, captions, soundtrack and recorder; the
dancers are the skinned LHM++ Gaussian avatars (`Assets/StreamingAssets/06_LarissaKaduRecap/splats/`, SPLAT_AVATAR_FORMAT.md) with the
follower's hair on the Unity hair simulation (head-movement `docs/SPLAT_DANCERS.md`). Record it with the viewer toggle on for the editor
session (Play mode: `unity command hm_splatdancers --action on`, leave Play mode, then `unity command hm_film --action record --aspect
vertical --capture 06_LarissaKaduRecap`; the editor must run on D3D12: `-force-d3d12`). The splats are opaque (the direction's
`avatar_opacity` 0.25 is not applied to them); what showed THROUGH the 75 %-transparent avatars is drawn over the splats instead
(`hm_splatdancers --action skeleton --mode physics`, the default): in the physics view the load-coloured skeleton, the spine beads, the
balance axes and the COM spheres; in every view the hand-contact lights; seen from above the floor craft (the leader's T). Elsewhere the
skeleton lines are off while the splats show (clean photoreal bodies); on the phones' video nothing changed. Renders (git-ignored, likeness
data): `head-movement/Recordings/06_LarissaKaduRecap_vertical_2026-10-20.mp4` and `..._horizontal_2026-10-20.mp4` (both PASS, A/V 0.0 ms);
the avatar film `06_LarissaKaduRecap_vertical_2026-10-11b.mp4` is unchanged. Review material: `work/larissa_kadu_recap/splat_film/`.

Fixes after the review of those two films (2026-10-21, `work/larissa_kadu_recap/film/splats_fix/`; the 10-20 films are kept):
- the splats fade into and out of the phone's video at the five flights with the avatars' factor (before: both bodies for 9 frames,
  then gone on one frame; back at full opacity on the way out); the caption backing (0.85, it also covers the speaker tag) and the
  chapter card backing (0.9) are darker while splats show; frames stay in film order in the recording (two neighbouring frames
  were swapped 6-7 times per splat film - the Movie Recorder's readback callbacks on D3D12, not the phone video);
- the recorder's AAC track has a burst at film 56.876 s (NOTES_R10): with the splat dancers on, `hm_film record` now replaces it
  after its checks with `python -m dancecap.film_audiofix <name>.mp4 --mix film/audio/<direction>_mix.wav --start <s>` (video
  stream copied, the mix as AAC 192k padded to the video length, verified: video md5, A/V sync in up to 30 windows, loudness vs
  the mix, true peak, no burst; the recorder's file kept as `<name>_rawaudio.mp4`; a section in the .report.md, "audio_fix" in
  the .report.json). It can be run by hand on any recording (rerunning reads the kept raw file). On a copy of the 10-20 vertical
  film: -16.0 LUFS, true peak -1.3 dBTP, 0.0 ms in 30 windows, the raw track's burst found at 56.860-56.900 s (peak 199.6).
- re-record both films the same way (splat dancers on in Play mode, then `unity command hm_film --action record --aspect vertical
  --capture 06_LarissaKaduRecap` and the same with `--aspect horizontal`; defaults `--readback auto --audiofix auto`); a full film
  now renders at about 16 frames/s (about 12 min).

Final splat films (re-recorded 2026-10-20 22:00-22:15 local with these fixes, splat dancers + swinging hair on, D3D12, nothing else
running): `head-movement/Recordings/06_LarissaKaduRecap_vertical_2026-10-20_final.mp4` and `..._horizontal_2026-10-20_final.mp4`
(renamed from the recorder's `_2026-10-20_2`; reports, audiofix json/log and the recorder's raw file `..._final_rawaudio.mp4` beside
them; the first 10-20 versions and the 10-11 avatar films are kept). Both: 11032 frames = 367.733 s x 30, 1080x1920 / 1920x1080,
ordered readback (at most 1 frame pending, 0 timeouts), about 28 frames/s (6.5 min each). Checked independently
(`work/larissa_kadu_recap/film/splats_final/`, `tools/verify_film.py`): recorded track and final track 0.0 ms against the mix at 14
points; final track -16.0 LUFS / -1.3 dBTP like the mix, no 10 ms block departs from the mix by more than 0.12 (the raw track's
burst at 56.88 s is gone); every frame of the 45 phone shots (5964 frames) in film order against the avatar film (0 flagged) while
the same test finds the 10-20 splat films' swaps (vertical 7 pairs, horizontal 9 pairs); the five flights cross-dissolve and the
captions have the dark backing (sheets in `splats_final/sheets/`).

## Face swap of the recorded film (v3, owner 2026-10-21)

Owner: "if we keep track of Larissa and Kadu's face poses (literally the face points detected in the unified 3d body poses) we could
feed those to facefusion and use the inswapper_128 model individually on each person with their included profile headshots. this
would be for the video output. we could also track face pose to warp the splats". Sources: `C:/Users/johnb/Videos/Zouk/
larissa-kadu-counter-recap/kadu.jpg` (lead) and `larissa.jpg` (follow), small studio portraits (FaceFusion finds one face each: inter-ocular 20.9 px for Kadu, 36.5 px for Larissa; arcface similarity between the two -0.055).
Code: `dancecap/film_faceswap.py`. Outputs and review material: `work/larissa_kadu_recap/v3/faceswap/` (git-ignored: likeness data).

**FaceFusion v4** (`C:/Users/johnb/Desktop/facefusion`, branch `v4` at 57c01ed, conda env `facefusion`). v4 needs Python >= 3.13
(`core.pre_check`; `queue.ShutDown`), so the env went from 3.12.13 to 3.13.13 in place (`conda install -n facefusion python=3.13 pip`).
FaceFusion 3's gradio stack was uninstalled first (v4 has no gradio UI; its cp312 wheels would not load under 3.13), and so was
`opencv-python` (it and v4's `opencv-python-headless` both own `cv2`). Then `python install.py cuda@13 --force-reinstall` installed
onnxruntime-gpu 1.30.0, numpy 2.4.6, opencv-python-headless 5.0.0.93 and the rest of `requirements.txt`. CUDA 13 is right for this PC:
driver 616.56 (CUDA 13.4) and an RTX 2080 Ti (sm_75, still supported by CUDA 13). The provider DLL imports only `cublas64_13` /
`cublasLt64_13` (cudart is static) and finds them, with `cudnn64_9`, in the env's `Library/bin` (the env's own conda cudnn 9.22 /
libcublas 13.5 for cuda-version 13.3). The pip `nvidia-*-cu12` packages left over from the 3.x install are unused; pytest 9.1.1 was added (to run the unit test there too). Always start it with
`conda run -n facefusion ...` (activation puts `Library/bin` on PATH). v4 has no `headless-run`: `run` is the headless command (the UI
is now the `api` server). Verified: `python facefusion.py run -s kadu.jpg -t <film frame>.png -o out.png --processors face_swapper
--face-swapper-model inswapper_128 --execution-providers cuda`, with every model on the CUDA provider (`v3/faceswap/cli_test/`).
No other env and no system setting was touched.

### Run it

    cd dancecap
    conda run -n facefusion --no-capture-output python -m dancecap.film_faceswap swap --film <rec>.mp4 --out <rec>_faceswap.mp4 \
        --sidecar <rec>.faces.json [--start S --end S] [--side-by-side <sbs>.mp4] [--lossless]
        # v4.2: the sidecar is required (it says which faces are splat-dancer renders); --heuristic for a prototype run without one;
        # --phone-shots swap is retired (the swap applies to the splat dancers only)
    ... -m dancecap.film_faceswap verify --report <out>.report.json     # bit-exact / PSNR outside the faces + flicker ratio
    ... -m dancecap.film_faceswap sheet --report <out>.report.json --out sheet.jpg [--frames N ...] [--dets]
    ... -m dancecap.film_faceswap canon-points                          # face_points_canonical.json (below)
    ... -m dancecap.film_faceswap mock-sidecar --report <heuristic run>.report.json --out mock.faces.json   # tests the sidecar path
    ... -m dancecap.film_faceswap restore-start --report <out>.report.json   # a whole-film output made before the fix: video start as recorded

The GPU passes hold `work/gpu.lock`. FaceFusion's own content analyser runs on every 15th analysed frame (it is not bypassed). How
`swap` works:
1. **Analysis.** Each frame is decoded once (ffmpeg yuv420p). FaceFusion runs yolo_face on the whole frame and on two overlapping
   square tiles of a 9:16 frame (a 1920-px frame is at 1/3 scale in the 640 detector), then 2dfan4 68 points, the arcface embedding
   and fairface gender. With a sidecar it only analyses frames where a dancer is in view and the splats show. Without one it only
   analyses 3D frames: phone shots (`camera_tour`, `fly_to_camera`) are taken from `direction.json`, plus 0.7 s after a phone shot,
   the flights in between, and - for a v3 direction - every window in which a phone's frustum video surface shows (`flow.frustums`
   [a, d] +- 2 frames: without a sidecar nothing says where on the screen the surface is).
2. **Who and where.** With a sidecar, each dancer's projected 5 points ARE the face. A detection counts only when it lies on them
   (mean point error < 0.45 of the inter-ocular distance) and then refines them. Frames with no such detection use the projected
   points plus the running detector-minus-projection offset (median over +-15 frames). A student or a poster is never swapped, and
   neither is a dancer whose face lies behind a phone's frustum video surface (sidecar `surfaces`, below): the picture there is the
   phone's video - real faces, often the teachers themselves - so the weight is 0 and no detection on it refines the points. Without a sidecar (the prototype), identity is 6 x (arcface similarity to Kadu's headshot minus to
   Larissa's), +-0.5 for fairface's gender, and minus a capped crimson-hair share in a ring around the face. It is averaged over a
   whole track (greedy box tracker, gaps <= 4 frames); |mean| < 0.35 means unknown and the track is not swapped; two tracks of one
   dancer at the same time leave the weaker one unknown.
3. **Plan.** Landmarks: gaps filled, Savitzky-Golay 7 / 2 (zero phase). Gate: inter-ocular 10 -> 16 px, |yaw| 45 -> 65 deg,
   |pitch| 30 -> 45 deg, sidecar visibility 0.55 -> 0.8, 0 behind a frustum surface. A hysteresis switch (on at 0.5, off below 0.25)
   makes the weight 0 or 1. No flicker (v3.1): a dip of at most 0.4 s (`bridge_frames` 12) where the gate stays >= 0.05 (the face is
   still a face: up to ~64 deg of yaw, a detector-score dip; never the back of the head, never a phone frame) does not switch it off;
   then a swap shows for at least 1 s (`min_on_frames` 30; v3.0 dropped only stretches under 10 frames) or not at all - counted per
   DANCER over all of its tracks (a face the tracker split in two counts as one run) - and the fades are made on that per-dancer
   timeline: erosion + box blur give 1/3-s fades that reach 0 on the last qualifying frame. The ends of a processed range stay open,
   so chunks of a film join without a fade. The report's `swap_spans` lists per dancer the film seconds its swap shows, the shortest
   span and the shortest gap (the flicker check).
4. **Swap.** inswapper_128 with the dancer's headshot embedding. Pixel boost is per track by face size: 128, 256 for inter-ocular
   > 44 px, 512 for > 88 px. The mask is FaceFusion's box mask (blur 0.3) x xseg_1 occlusion, times the weight. The face is pasted in
   YUV only where mask x weight > 1/255, so every other pixel of the decoded frame reaches the encoder unchanged (counted per frame).
   Text glyphs of UI rects (sidecar `ui`; without it the 9:16 caption band and the chapter card while it shows) are restored where the
   swap changed them, and so are the grey / blue skeleton strokes of the 3D overlays that cross a face (`overlay_mask`: bluer than
   skin and hair, connected strokes >= 0.4 % of the box, + 1 px rim; xseg alone let the swap paint them partly away). Audio: an
   excerpt re-cuts the same instant as AAC 192k; a full film copies the track and keeps the video's start offset (the recorder's
   0.033 s) by a stream-copy remux after the encode (`restore_start`: ffmpeg 8 drops the `-itsoffset` of the encoded raw pipe; video
   and audio bits unchanged by the remux). Frames are read with `-fps_mode passthrough` (without it ffmpeg duplicated the first frame
   of every range: see "The whole v3 films" below). Video: x264 CRF 14 slow (`--lossless`: qp 0).

### The face sidecar (`<recording>.faces.json`, format "hm film faces" v1): the Unity recorder writes it

One JSON file per recording, next to the MP4. One entry per recorded frame, in recording order:

    {"format": "hm film faces", "version": 1, "recording": "<name>.mp4", "direction": "larissa_kadu_recap", "capture": "06_LarissaKaduRecap",
     "aspect": "vertical", "width": 1080, "height": 1920, "fps": 30, "frames": 11032,
     "landmarks": ["eye_r", "eye_l", "nose_tip", "mouth_r", "mouth_l"],
     "face_points": "<path + sha256 of face_points_canonical.json used>",
     "dancers": [{"key": "lead", "name": "Kadu"}, {"key": "follow", "name": "Larissa"}],
     "f": [ {"t": 12.3667, "dt": 12.3667, "view": "3d" | "video" | "mix", "video_alpha": 0.0, "splats": 1.0, "phone": null | "02",
             "cam": [px, py, pz, qx, qy, qz, qw, vfov_deg],
             "ui": [[x, y, w, h], ...],
             "surfaces": [ {"phone": "02", "alpha": 0..1, "quad": [[x, y] x 4], "z": [m x 4], "role": "enter" | "leave", "full": 0 | 1}, ... ],
             "d": [ {"k": "lead", "in": 1, "lm": [[x, y] x 5], "z": [m x 5], "yaw": deg, "pitch": deg, "roll": deg, "iod": px, "size": px,
                     "vis": 0..1, "by": "partner" | "self" | "", "layer": "splat" | "avatar" | "skeleton" | "none", "la": 0..1},
                    {"k": "follow", ...} ] }, ... ] }

- Pixels: the recorded MP4's pixels, origin top-left, y down; a pixel's centre is at +0.5. `t` = film seconds of the frame (frame index / 30),
  `dt` = the dance (master) time the dancers were posed at (sub-frame). `video_alpha` = the phone video's opacity in that frame
  (0 = pure 3D, 1 = the phone's own video), `splats` = the splat composite opacity, `view` = "3d" (video_alpha 0), "video" (1),
  "mix" (in between). `cam` = the recording camera in Unity world (position, rotation, vertical FOV). `ui` = pixel rects of every visible
  screen-space overlay element: caption backing including the speaker tag, chapter / title card, readouts, legend, the label pills of the
  3D arrows. Elements below alpha 0.05 are left out.
- `surfaces` (v3.1; required with a v3 direction, may be absent or empty on frames without one): every phone frustum video surface
  drawn in this frame with alpha > 0 (the `flow.frustums` envelopes of the v3 direction: the phone's live video on its image rectangle at
  `depth_m` along its optical axis, CameraVideoRig / FilmCameraSource). `alpha` = the surface's opacity in this frame (the envelope
  value, as drawn); `quad` = its 4 corners in recorded pixels, in the order top-left, top-right, bottom-right, bottom-left of the
  phone's image, projected by the recording camera into the output size (y down); `z` = the camera depth (m) of each corner. A
  rectangle that crosses the near plane is clipped against it first and written as the clipped polygon (3-6 points, a `z` per point);
  `full: 1` when it covers the whole frame (the last frames of a way in, the first of a way out). `role` as in `flow.frustums`. The
  phone's FULL-FRAME video is not a surface: it stays `video_alpha`. **Reader rule** (`film_faceswap.surface_hit`): a dancer's face is
  not swapped in a frame where a surface with alpha > 0.05 covers one of its 5 points or a corner of its face box (the points' centre
  +- 1 inter-ocular distance: the paste region) and lies in front of the face there - the plane's depth at those pixels (1 / z is affine
  in screen coordinates, fitted to the corners) below the face's depth (max of its `z`); without `z` (or with `full`) a covering
  surface counts as in front. A surface BEHIND the dancer (the camera looks past him at the phone's frustum) hides nothing. The weight
  is then 0 on those frames, and the fades put it at 0 on the last frame before the surface covers the face: no pop.
- `lm`: the 5 points are `roles.<role>.points5` of `work/larissa_kadu_recap/v3/faceswap/face_points_canonical.json`, in the splat avatar's
  canonical pose (Unity coordinates). They are posed rigidly with the head: `p = SkinnedSplatAvatar.CurrentSkin[15].MultiplyPoint3x4(points5[i])`
  (S_15 = G_15 T(-J_15), the matrix the splat skinning uses; checked: the SMPL-X rest joints of those betas equal the splat files'
  `rest_joints` to 0.000 mm). They are projected by the recording camera into the output size. `samples9` (the 5 + chin, forehead, two
  temples) are for `vis`. `z` = camera depth (m). `in` = 1 when all 5 are in front of the near plane and the head centre is in the frame.
  The order is the subject's right eye first: the InsightFace / FaceFusion order of a face seen from the front.
- Orientation, relative to the line of sight (not the camera axis): z' = -normalize(head centre - camera position), x' = camera right
  made orthogonal to z', y' = camera up made orthogonal to both (no cross products: Unity is left-handed). With the face forward
  `f = S_15.MultiplyVector((0, 0, -1))` and up `u = S_15.MultiplyVector((0, 1, 0))`: yaw = atan2(f.x', f.z') (0 = into the lens,
  +90 = turned to the image's right, +-180 = back of the head), pitch = asin(f.y') (+ = face up), roll = atan2(u.x', u.y').
  `film_faceswap.pnp_pose` uses the same conventions on detections.
- `iod` = |lm[0] - lm[1]| px; `size` = 0.2 m head diameter x focal px / depth. `vis` = the share of the 9 samples that a ray from the camera
  reaches without first hitting the partner's body or the dancer's own forearms / hands (capsules from the bones: head r 0.11 m, torso
  0.15, upper arm / forearm / hand 0.05; hit 1 cm before the sample counts). `by` = the occluder of most hidden samples. A depth test
  against the splat depth would also catch hair; it is not needed for v1 because FaceFusion's xseg occlusion mask handles hair and hands
  in the picture.

- `layer` / `la` (v4.2, owner 2026-10-22: "apply the face fusion to the gauss splats only"): the layer this dancer's face is drawn on in
  this frame and that layer's opacity - "splat" (the dancer's splats are drawn: `SplatDancers.ShownFor`, la = the splats' composite
  opacity, i.e. the phone-video fade), "avatar" (the mesh avatar instead, la = its opacity), "skeleton" (neither: lines and overlays only,
  or the body hidden under a phone's video; la 0), "none" (no head pose; `in` 0). **Reader rule** (`film_faceswap.splat_share`): a face
  is swapped only on layer "splat", with weight x la x (1 - video_alpha), and never behind a frustum surface (above). A sidecar recorded
  before v4.2 has no `layer`: the frame's `splats` share decides (0 = the splat dancers are off: no swap); a sidecar with neither swaps
  nothing. `swap` refuses to run without a sidecar unless `--heuristic` asks for a prototype.

`film_faceswap.validate_sidecar` checks format, landmark order, dancers, size, frame count, the `surfaces` entries (alpha 0..1, a
polygon of >= 3 points, one `z` per point) and each dancer's `layer` / `la` (one of the four layers, 0..1). A mock sidecar made from a heuristic run (`mock-sidecar`) exercises the sidecar path until the
recorder writes real ones.

**Unity stage (not done in this workflow; Unity was in use by another one):**
1. `Assets/Film/FilmFaceSidecar.cs` plus a hook in the frame-locked recorder (`Assets/Film/Editor/FilmRecorder.cs`, after the director
   posed the frame that the Movie Recorder captures). For every written frame and each dancer, compute the entry above from
   `SkinnedSplatAvatar.CurrentSkin[15]` (with splats off: the avatar's head bone x its bindpose, the same S_15 within the 1.3-5.3 mm
   skeleton difference). Use the recording camera and the recorder's output size, not the Game view's. Take `video_alpha` / `phone`
   from `FilmCamera.PovVideoOpacity` / `PovCamera`, `splats` from `SplatDancers.FilmVideoFade`, and `ui` from the FilmUi / caption /
   chapter-card / arrow-label RectTransforms (`GetWorldCorners` -> screen -> px, y flipped).
2. Load `face_points_canonical.json` from the take's work folder (or copy it into the capture's `splats/` folder; it holds face geometry
   of the two dancers, so it is git-ignored like the splats).
3. Write `<name>.faces.json` next to the MP4 when the recording ends (frames in film order: the ordered readback already guarantees it).
   Add `--faces on|off` to `hm_film record` (default on with the splat dancers) and mention the file in the `.report.md`.
4. The phone frustum surfaces of the v3 transitions (v3.1; the review found v3.0's claim "nothing extra" wrong: the projected points of a
   dancer behind a surface land on the phone's video of real faces, which would be swapped and pop when the surface fades): for every
   surface drawn this frame with alpha > 0, write a `surfaces` entry - the alpha it is drawn with, the 4 corners of its video rectangle
   (`CameraVideoRig` / `FilmCameraSource`: apex at the phone pose, rectangle at `depth_m` along the optical axis) through
   `recordingCamera.WorldToScreenPoint` scaled to the recorder's output size with y flipped, and each corner's camera-space depth
   (`-recordingCamera.worldToCameraMatrix.MultiplyPoint(c).z`); clip against the near plane when a corner is behind it; `full: 1` from the
   frame the camera reaches the phone pose (`video_full_at_film`) on.
5. Then, outside Unity: `film_faceswap swap --sidecar` on each aspect (the 16:9 film needs the sidecar: the heuristic UI bands are 9:16 only),
   then `verify`, and look at a `sheet`.

### Prototype without the sidecar (vertical final film, 2026-10-21)

Excerpts in `v3/faceswap/`: `<X>_swap.mp4` (with audio), `<X>_before_after.mp4` (source | swapped, 2160x1920, audio), `<X>_sheet.jpg`
(face crops before / after), `<X>_swap.report.json` (tracks, identity scores, weights, per-frame faces, speed, verify).

| excerpt | what is in it | swapped (face-frames) | looked at |
|---|---|---|---|
| A 21-27 s | feature zoom "frame", then the flight into phone 02 (25.9 s) | Larissa 67 | she turns 3/4 to the camera: crisper eyes, brows and lips, clearly her; Kadu is in profile the whole zoom (yaw 60-85 deg): left alone, rightly; phone 02's video untouched |
| B 350-360 s | feature zoom "frame", physics view, rhythm with turns up to 393 deg/s | v3.0: Kadu 73, Larissa 59; **v3.1: Kadu 18** (below) | v3.0: short spans between the turns; Kadu's swapped face gets his real beard and eyes; identity right on every swapped frame; no back of a head is ever detected or swapped; the slash of skin across her cheek is in the splat render (LHM++ side hair), not from the swap |
| C 105-111 s | phone shot, then the close zoom "arm" (Kadu inter-ocular ~70 px, boost 256), then the flight into phone 03 | Kadu 78 | the biggest gain: a sharp, recognisable Kadu instead of the soft splat face; the chapter card's "and space" over his forehead stays crisp (the xseg mask leaves the text out); phone frames untouched |
| D 209-214 s | zoom "connection", Kadu 3/4 with her forearm across his chin | Kadu 113 | one continuous span (after widening the yaw gate from 35-55 to 45-65 deg the swap no longer switched off and on at ~46 deg); her forearm stays in front of his face |

Identity per track (prototype): Kadu tracks arcface 0.14-0.25 to his headshot vs 0.02-0.06 to hers, 100 % male; Larissa tracks 0.09-0.20
vs -0.06-0.02, red-hair share 0.18-0.49. Flicker: within a face box swapped in two neighbouring frames, the frame-to-frame change of the
output is 0.99-1.00 times the source's (A 1.001, B 0.996, C 1.002, D 0.992). Outside the faces: 0 pixels changed before encoding in every
frame. After CRF 14 the luma PSNR outside the faces is 50.1-52.7 dB (encoding only). Lossless test (`v3/faceswap/tests/C_107-109_lossless.*`,
`--lossless`, 60 frames, 56 with Kadu's face swapped): decoded output = decoded source in all 119.6 M luma pixels outside the face boxes
(0 differing). Sidecar path (`tests/B_352-358_mocksidecar.*`, a mock sidecar made from excerpt B's heuristic tracks + 1 px noise): one track
per dancer, 95 of 97 (lead) and 93 of 94 (follow) in-view frames refined by a matching detection, the rest from the offset-corrected
projection; the right person on every swapped frame, flicker 0.989.

**Fast turns (v3.1).** In B the v3.0 plan swapped each dancer in 7 separate 0.37-0.8 s spans (Kadu 352.9-353.5, 355.4-356.0, 357.2-357.9,
359.4-360.0; Larissa 353.3-354.1, 355.9-356.3, 357.7-358.5): between the turns a face qualifies for 0.4-0.8 s, so the swap came and went
every ~2 s, mostly as a triangle that peaked at weight 0.09-1.0 - a flicker of a sharper face at the turn rate. v3.1 (Plan, above): a swap
shows for at least 1 s per dancer or not at all, and dips under 0.4 s where the face is still a face are bridged. Re-run with the same
analysis (`B_350-360_swap_v31.*`, `B_350-360_v31_before_after.mp4`, `B_350-360_v31_sheet.jpg`, `B_350-360_v31_weights.png` = the weights
of both plans over the excerpt): none of the 7 blips is swapped any more (the frames where v3.0 swapped are now source = output: Kadu
frontal at yaw -10 at 353.1 s and Larissa at yaw 8 at 353.6 s keep their splat faces); the only swap left is Kadu's 359.4-360.0 s span,
kept because it touches the end of the excerpt (it continues past 360 s; in a whole-film run it is judged with its continuation). 18
face-frames (was 132), 0 px changed outside them, flicker ratio 0.999, PSNR outside the faces 52.5 dB. Looked at: no on / off anywhere in
the turning passage; the price is that the fast-turn section shows the softer splat faces. A, C and D are unchanged by the new rule
(their spans are 2.2-3.8 s: the same 67 / 78 / 113 face-frames, computed from their cached analyses).

**Frustum surface (v3.1), end to end** (`tests/D_209-214_surface.*`): excerpt D with a mock sidecar plus a synthetic phone-02 surface over
Kadu's face for 211.0-212.0 s (alpha ramps 0.2 s in / out; no depths, so in front): his swap fades 1.00 (210.0 s) -> 0.55 -> 0.09 (210.77 s)
-> none from the frame the surface passes alpha 0.05 (41 frames `behind_surface`), and back 212.23 s (0.27 -> 0.55 -> 1.00 at 212.83 s):
two spans (209.57-210.80, 212.23-213.33), no pop, 0 px changed outside the faces.
Unit tests: `tests/test_film_faceswap.py` (weights and fades, hysteresis, open range ends, pose proxies, the bit-exact YUV paste, the
sidecar validator incl. surfaces, a face behind / in front of / beside a surface, the plane depth of a tilted surface, the fast-turn
blips dropped and the dips bridged, the minimum-on over a dancer's tracks, the heuristic's frustum windows; no FaceFusion needed).

**Speed** (RTX 2080 Ti, measured while the other workflow's Unity editor used about 1.5 of the 4 CPU threads and the total CPU load
was about 100 %): analysis 3.5-5.1 frames/s; the swap pass 2.2-3.5 frames/s including decode and two encodes (a frame without a face
passes at about 40 frames/s); 114-310 ms per face (boost 128-256) over the runs. Component times: yolo_face about 60 ms per window, 2dfan4 40-60 ms,
arcface 13 ms, fairface 10 ms, xseg 150 ms. FaceFusion's numpy pre- and post-processing on this CPU-starved PC dominates, not the GPU.
The film is 46 % 3D (169 s = 5070 frames), so a whole film is about 1-1.5 h per aspect at this load. With the sidecar, detection can be
cut down to the projected face boxes (not done yet).

### Warping the splat faces by the tracked face pose: recommendation, not built

- The head pose is already in the splats: they are skinned from the fit's head joint, which is what the sidecar projects. The fit has no
  face motion: `jaw_pose` = 0 in all 10815 frames of both dancers (`smplx_rf2_noslip`), eyes under 1.3 deg. inswapper keeps the TARGET's
  pose and expression, so the swapped faces are exactly as still as the splat faces (a closed mouth while the teachers talk).
- A per-Gaussian face warp from expression parameters is NOT worth it now. There is no expression track (the phones see the faces at
  10-70 px; SMIRK / EMOCA on such crops would be noisy). It needs a GPU blendshape pass and an inner mouth the splats do not have. Faces
  are big enough to read a mouth only in the feature zooms (C: inter-ocular ~70 px). The risk is an uncanny mouth.
- If talking faces are wanted, the cheap credible step is jaw-only: open the jaw joint (22) 0-12 deg from the speaking teacher's speech
  envelope (transcript speaker x class-mix level), smoothed. 3484 of each avatar's 160000 Gaussians carry a jaw weight > 0.3 (LHM++'s
  template weights on the face), so the splat mouth moves with plain LBS, and the swap then follows for free (inswapper keeps the
  target's mouth). Check first in Unity how an opened LBS mouth looks without inner-mouth Gaussians.
- Use the sidecar's detector-minus-projection offset as a measurement. If it shows systematic head-rotation errors, refine the head
  rotation of the fit (all views and the Quest benefit). That is a pose fix, not a splat warp. On the Quest there is no swap, so a
  better-looking face there needs a better LHM++ face, not this tool.

## v3 direction (owner 2026-10-21): the counterbalance pivot, physics on every connected shot, no jump cut anywhere

The owner's words: (3) "every time kadu say's 'back to the axis' he's referring to the counterbalance pivot point, not the T axis. correct
this"; (4) "even more physics activations of foot center of gravity projections (the dot for each individual dancer on the floor) and tension
compression estimations rendered in arms and legs"; (5) "every time we switch to a new camera view we fly in and out of it smoothly with a
transition from splat to camera POV ... entering and exiting phone camera POV by seeing the 2D surface projected in the phone frustrum in the 3D
world like we did in the explainer video ... stay at ground level for these transitions ... we just never jumpcut".

Code: `dancecap/film_recap_flow.py` (post-passes over `film_recap.build_direction`; no `film_recap*` module is changed, so `film_recap build` and
`film/direction.json` are exactly what they were), tests `tests/test_film_recap_flow.py`. Outputs (git-ignored) in `work/larissa_kadu_recap/v3/direction/`:
`direction.json`, `narration/timeline.json` (identical to film/'s), `chapters.json`, `direction_report.json`, `review_v3.md` (the review table: per
sentence the old and the new shots and how the camera gets there; the pivots; physics before / after; every transition), `clips.json`, `flow_sheet.png`
(eye azimuth and height over the film), `audio/larissa_kadu_recap_mix.wav` (sample-identical to film/'s: the time map is unchanged) + `make_film_mix.py`.

    python -m dancecap.film_recap_flow build [--mix]   # the v3 direction (+ its soundtrack); exits 1 when a check fails
    python -m dancecap.film_recap_flow check           # film_recap.validate + the v3 checks on v3/direction/direction.json
    python -m dancecap.film_recap_flow sheet           # redraw flow_sheet.png
    python -m dancecap.film_recap_flow clips           # UNITY STAGE ONLY (writes the capture's cameras/): the phones' video incl. the frustum ways in / out

**(3) "back to the axis".** `back_to_axis` finds the phrase and its close variants ("go back to axis", "return to the axis", "bring her TO the
axis" - flagged as a variant). Six sentences (u026 is Larissa's in the transcript; the same phrase, treated the same): 161.4, 181.2, 188.8, 196.6,
292.9 and 323.3 s (master). The current film shows them as the overhead with the leader's T (u026, u032) or an arrow "axis" on the leader's T
(the others). In v3 their "axis" events are taken out of the attention set (no T window) and each phrase gets a ground-level PIVOT shot from just
before its verb ("bring") to 1.6 s after "axis" (never over a neighbouring call that does mean his T - "exit", "rotate", "rotation", "side to side",
"in front of her": those keep the T; a squeezed window grows backwards to 3.2 s): `physics_3q` framing at 22 deg elevation (eye about 2.25 m),
the look point drawn 35 % toward the pivot, layers counterbalance (the yellow couple-COM dot and ring), `floorcraft: pivot_dials` (the
counterbalance dials with the yellow ring at her anchored foot; + the current T where a T call sits inside the shot), physics (the couple's
balance axis from her anchored foot to the combined COM lights up), com_marker (both dancers and the couple), and an arrow "axis (pivot point)"
on the counterbalance pivot the phrase means: `pivot_foot` of the pivot within 6 s (u026 -> the 164.0-166.5 s pivot, u032 -> 176.6-178.6 s,
u061 -> 287.1-291.7 s, u069 -> 319.8-323.3 s), or - for the two counterbalances without an anchored foot (u033, u034) - `couple_com` (their shared
balance point on the floor). The replay (240.1-243.7 s, a 371 deg pivot) is that pivot view too (was an overhead). Table: `review_v3.md` section (3).

**(4) physics.** `apply_physics`: every segment where the teachers demonstrate (moving >= 20 % of it) or are connected (a hand on the partner -
`v3/hand_contact/events.json` - or the counterbalance connection >= 0.5, for >= 50 % of it) and do not stand apart (> 0.95 m) gets
`skeleton_mode: physics` (tension orange / compression blue in arms and legs, drawn over the splats and over the phones' video), `balance`,
the legend, a com_marker with each dancer's COM (sphere + its floor dot; + the couple's in a counterbalance) and `physics_limbs` (arm / leg lines
1.5x wide). On screen, film seconds before -> after: load colours 85.4 s (23 %) -> 320.1 s (87 %); the dancers' COM floor dots 69.2 s (19 %) ->
292.3 s (79 %); balance axes 85.4 -> 320.1 s; counterbalance layer 113.1 -> 145.9 s; pivot shots 0 -> 36.0 s. In phone shots 0 -> 122.1 s of load
colours. Off only where they stand apart (8 segments, 47.6 s of film: 0-20.8, 261.4-271.1, 339.4-350.0, 361.2-367.7 s). The legend keeps its
fixed place in the top band (under a card while one shows); captions are screen-space on top of the 3D picture, so nothing hides them.

**(5) flights.** A shot = a run of segments with one camera block. Every shot but the first carries `camera.transition_in`; the types follow from
the two views: `flight` (3D -> 3D), `crane_up` / `crane_down` (into / out of an overhead), `phone_in`, `phone_out`, `phone_to_phone`, `replay_in` /
`replay_out`. ONE easing for all: smootherstep. ONE duration rule: T = base + per180 x |d azimuth| / 180 + per_m_r x |d radius| + per_m_h x
|d height|, clipped (v3.1: flight 0.9-2.4 s, phone in / out 0.8-2.2 s, phone to phone 1.0-2.4 s, crane 1.6-2.2 s, +0.6 s when a phone way passes a
crane and never quicker than the crane; `flow.defaults.rule`), never shorter than the arc floor (no arc turns faster than 200 deg/s at its
smootherstep peak) or the frustum's readable floor (below), squeezed by at most 0.35 s. Where it sits: into a 3D view the new view has arrived 0.35 s
after its segment starts (the words begin), into a phone at its segment start (0.3 s before the words); a view that answers a call has arrived
0.1 s before the word; out of a phone the way out starts 0.1 s before the next segment (the phone keeps its video); phone to phone is centred on the
boundary; a replay's starts at it. Each view holds at least 1.0 s (3D), 1.4 s of full-frame video (phone), 2.2 s (overhead) between its way in and
out; a shot that cannot joined a neighbour (10 joins: an ordinary shot joins the more important view beside it, a pivot / physics view grows
instead; while the phones are short of the director's share a phone outranks a body / feet / timing / path zoom; never past 9 s for a phone). Eye
azimuths are planned (`world_azimuth_deg`): a 3D shot starts where the previous view was when that is within its cone of side-on (else at the
nearest edge) and drifts toward the next phone. Ground level: every eye of every way between 0.5 and 2.4 m; only the overheads are entered and left
by a crane. Result (v3.1): 75 shots (88 before), 74 flights (6 flight, 23 phone_in, 22 phone_out, 16 phone_to_phone, 2 crane_up + 3 crane_down + 3
phone ways passing an overhead, 2 replay; median 1.16 s, 96.7 s = 26.3 % of the film in motion), 77 frustum envelopes; the cut film had 62 hard cuts,
20 0.7-s blends and 5 fly-ins. The phones' own video: 196.6 s of phone shots (53.5 %; the cut film 54.1 %), of which 163.8 s (44.6 %) full frame
(the cut film 52.4 %, v3.0 30.4 %) - the rest is the way in and out with the picture standing in its frustum. Checks (`check`): the film checks, no
shot change without a transition, one easing, the rule, the holds, ground level, the crane only at the overheads, the arcs <= 200 deg/s, phones
entered / left through their own frustums and each surface readable >= 0.6 s, the pivots, the legend and dots on every physics segment, the
'connected' rule up to date with the hand-contact events, captions and the film / master time map identical to film/'s (3677 points): PASS.

### v3.1 (owner decisions 2026-10-21): the overheads keep their crane; quicker phone ways, the phones' video back toward half the film

The review of v3.0 found the full-frame phone video down to 30 % and 8 crane transitions to the 4 overheads. The owner chose **"Keep overheads,
crane up"** (the 4 true top-down views and the smooth crane up / down for those 8 ways only) and **"Shorter flights, ~50% phone"** (shorter flights into
and out of the phones and a shorter minimum phone hold; the flights stay smooth but quicker). Earlier rules hold: no jump cut anywhere, ground level
otherwise, the frustum surface on every phone entry / exit, "back to the axis" = the counterbalance pivot, the physics layer.

What changed (`FlowCfg` / `V3_DIRCFG`; v3.0 values in brackets; tested in `tests/test_film_recap_flow.py`):
- phone ways: phone_in / phone_out 0.8 + 1.2 x |daz| / 180 + 0.1 x |dr| + 0.1 x |dh|, 0.8-2.2 s, min 0.75 [1.3 + 1.8 x ..., 1.3-3.2 s, min 1.2];
  phone_to_phone 1.0 + 1.1 x ..., 1.0-2.4 s, min 0.95 [1.9 + 1.8 x ..., 1.9-3.6 s, min 1.7]; the optical-axis dolly takes 30 % of a phone way [40 %],
  so the arc gets more of the time; `MAX_ARC_DEG_S` 200 [160] and the rule never goes below the arc floor 1.875 x |daz| / (200 x arc share).
- the phone's minimum full-frame hold 1.4 s [2.0]; the phone's video has arrived at its segment start [0.35 s after]; the way out of a phone starts
  0.1 s before the next segment [0.3].
- the frustum surface: on every phone way it must READ as a picture standing in the 3D world for >= 0.6 s (`frustum_readable_s`: alpha >= 0.5 while
  the eye is not yet / no longer on the phone's optical axis, where the surface grows to fill the frame). The rule has a readable floor (0.61 s phone
  in, 0.68 s phone out, 0.92 s phone to phone); a leaving phone's surface stays opaque for 70 % of the way out [55 %] and fades over the rest + 0.45 s;
  the phone-to-phone envelopes are enter [0.05 T, 0.3 T, T, T] and leave [0, 0, 0.65 T, 0.95 T] [0.2/0.5 and 0.5/0.8]. Measured: min 0.68 s, median
  0.89 s over the 77 surfaces.
- the crane: the 4 overheads (173.5-179.4, 237.8-243.3, 279.0-284.3, 333.2-339.4 film s) are entered and left by the crane (2 crane_up, 3 crane_down,
  3 phone ways with `crane: true`, 1.6-2.6 s); a phone way that passes an overhead is never quicker than the crane itself (1.6 s). `check` fails a crane
  between two ground-level views.
- the director (`V3_DIRCFG`): up to 3 phone shots in a row [2] (D4's run), long free stretches cut into ~7.5-s phone parts, phone shots up to 10 s
  [6.4 / 8.6], and the share it aims at 62-68 % [D4's 56-62 %]; its balance step stops at 59.3 % on this take (no calm shot or second zoom is left to
  hand to a phone).

Result (v3.0 -> v3.1; `review_v3.md` section (5) has the full table): full-frame phone video 111.7 s (30.4 %) -> **163.8 s (44.6 %)**; phone shots
29 (156.7 s, 42.6 %) -> 39 (196.6 s, 53.5 %); camera in motion 128.8 s (35.0 %) -> **96.7 s (26.3 %)**; phone ways (61) median 1.83 -> **1.12 s**, p90
3.18 -> 2.09 s, max 3.8 -> 2.6 s; phone_in median 1.83 -> 1.02 s, phone_out 1.65 -> 0.96 s, phone_to_phone (16, was 8) 3.0 -> 1.57 s; all 74
transitions: 25 under 1 s, 19 at 1-1.25 s, 10 at 1.25-1.5 s, 9 at 1.5-1.75 s, 4 at 1.75-2 s, 6 at 2-2.5 s, 1 at 2.5-3 s, none longer; the fastest arc
199.9 deg/s (9 big phone changes of 106-175 deg turn at 166-200 deg/s at their peak; v3.0's fastest was 158 deg/s); phone holds median 3.76 s, 5 of
38 under 2 s (1.4-1.93 s: short phone shots between the pivots). The soundtrack is unchanged (same time map, captions identical), so
`audio/larissa_kadu_recap_mix.wav` was not re-rendered. v3.0's files are kept in `v3/direction_prev_v3.0/` (the review compares against them).

**Not 50 %.** The phone shots are back at the cut film's share (53.5 % vs 54.1 %); the gap to full frame is the ways (a phone_to_phone, centred on its
boundary, takes all of its 27.1 s from the two phones). More phone time would have to come out of the 3D views: the 6 pivots, the 4 overheads, the
physics views (physics_3q 89 s) and the replay. A what-if with the director's physics-call views off (`DirCfg.phys_calls = False`, physics_3q
89 -> 44 s) reached only 47.3 % and changed the overheads (5, 10 crane ways), so it was not taken. The next levers, for the owner: fewer / shorter
physics views, or letting a protected view take only the time it needs from a phone neighbour instead of joining it whole (the 315.7 s physics view
takes a 4.3-s phone shot now).

**The 'connected' rule and the hand-contact events.** `apply_physics` reads `v3/hand_contact/events.json` (holds and touches -> "a hand on the
partner") at build time; the build records its sha256 in `recap_film.v3.connected_events`, and `check` prints "STALE: ..." when the file has changed
since. After the hand-contact events change (e.g. the review's event #118 on her upper back, not her front waist), re-run

    python -m dancecap.film_recap_flow build [--hand-events <events.json>]   # re-applies the physics layer; the time map and captions do not change
    python -m dancecap.film_recap_flow check

and look at `review_v3.md` section (4) (the physics before / after table and the segments the rule leaves off). Only `layers` (skeleton_mode,
physics_limbs, com_marker) of the affected segments can change: the shots, the transitions and the soundtrack do not depend on the events.
Sensitivity, measured on this build: the rule needs only WHEN a hand is on the partner, not where - and with event #118 removed, or all 21
events the review flagged (#9, 29, 31, 36, 41, 47, 60, 64, 97, 104, 115, 118, 123, 124, 138, 140, 141, 143, 145, 146, 147) removed, no
segment's physics layer changes (load colours stay 320.1 s: those segments are also demonstrations, or other holds cover them).

### v3.2 (2026-10-21, toward the owner's "~50% phone"): the 3D views take only the time they need

v3.1 stopped at 44.6 % full-frame phone video. v3.2 reaches **178.6 s = 48.6 %** of the 367.7-s film without touching the rules the owner set
(no jump cut, ground level, the crane only for the 4 overheads at the same film times, arcs <= 200 deg/s, every frustum surface readable
>= 0.6 s, one easing). `film_recap_flow` changes only (the director, `film/direction.json`, the captions and the soundtrack's time map are
unchanged: `build --mix` re-renders `audio/larissa_kadu_recap_mix.wav` sample-identical, sha256 cb0cd023...); v3.1's files are kept in
`v3/direction_prev_v3.1/`, and `review_v3.md` section (5) compares the cut film, v3.1 and v3.2.

- **`give_phones_time`** (after `plan_flow`; every move is tried on a copy and kept only when `schedule` finds no view that misses its hold, every
  chapter still opens a shot, and the full-frame video grows by >= 0.25 s): (1) a calm orbit or a zoom with no call of its own next to a phone joins
  that phone (max 3 phone shots in a row, a phone shot at most 10 s = the director's `pov_max_s`), else becomes a phone shot of its own with the
  phone that shows the window best, never one more than 90 deg round from a phone neighbour (`own_phone_max_turn_deg`: no whip for a second of
  video), else hands the part the phone beside it can show (keeping the director's 3.0 s for a calm view); (2) the opening orbit keeps the title card on a still 3D view and hands the rest to a phone; (3) a physics view / zoom (never a pivot, the
  replay or an overhead) ends 0.3 s after its last call's arrow + its way into the next phone, and starts 0.15 s before its first call's arrow
  (the director's `phys_pre_s`), at the speech's own pauses, never shorter than the director's minimum (3.0 s physics, 2.6 s zoom). Taken: the
  call-less 'frame' zoom 43.1-45.7 s -> its own phone shot 02 (+1.56 s), the calm orbit 271.1-276.0 s hands its last 1.9 s to phone 03 (+1.86 s;
  the orbit keeps 3.0 s under the ch09 card), the 'frame' zoom 20.8 s starts at 22.5 s (+1.73 s to phone 00), the physics views at 79.6 s and
  253.7 s end 0.9 s earlier (+0.92 / +0.86 s). Not taken (listed in the review): the opening (only phone 00 sees both teachers at 5-8 s, 174 deg
  round from phone 05: a whip; phone 05 cannot show it - it records from 7.41 s; ch01 opens at 8.2 s), the rest of that orbit (no phone sees both
  teachers at 263.9-266.9 s master), most physics tails / leads (their minimum length or their way out leaves no room).
- **`nudge`** (in `plan_flow`, before any join): a shot of >= 2 s that misses its hold by <= 0.8 s takes that little from the shot after it (at a
  pause, or by the deficit itself when <= 0.4 s) instead of swallowing it. The physics view at 315.7 s missed by 0.03 s and in v3.1 swallowed the
  4.3-s phone shot 05 318.4-322.7; it now takes 0.08 s of it.
- **`phone_first`** (in `fix_boundary`): a shot that cannot hold and has nothing of its own to show (no call arrow) joins the phone beside it before a
  pivot / physics view: the orbit sliver 158.1-160.0 s goes to phone 00 (was pivot01's), the fly-in sliver 183.8-184.6 s to phone 02 (was pivot02's;
  phone 02's 1.6-s shot grows to 2.4 s). The 6 pivot windows themselves (verb - 0.45 s .. 'axis' + 1.6 s) are all still pivot shots.
- **The phone ways rest on their floors**: phone in / out 0.7 + 1.2 x |daz| / 180 (+0.1 x |dr| + 0.1 x |dh|), 0.7-2.2 s, min 0.7 [v3.1 0.8 / 0.75];
  phone to phone 0.93 + 0.8 x .. (+0.05 x ..), 0.93-2.4 s, min 0.93 [1.0 + 1.1 x, min 0.95] - so the shortest way is set by the frustum's
  readable floor (0.61 in / 0.68 out / 0.92 phone to phone) and every big turn by the 200 deg/s arc floor; the way out of a phone starts AT the
  next segment (`depart_before_s` 0.0 [0.1]: the phone keeps its full frame to its end). `call_arrive_before_s` = 0.1 (unchanged; a parameter now).
- **A phone is entered only while it records** (`schedule`, from its track's first / last sample; `check` fails a phone window outside it): the
  way into phone 05 at the film's start (ch01_01_open) began at 6.3 s in v3.1, before phone 05 records (7.41 s) - its surface would have had no
  video and the flight no live pose. It is now 0.35 s shorter than its rule and starts 1.07 s later (the orbit holds the title card longer; phone
  05's full frame starts 1.4 s after its segment). Each phone shot's `cameras[0].dance_t` is now its way in .. way out (schema below).

Result (v3.1 -> v3.2): full-frame phone video 163.8 s (44.6 %) -> **178.6 s (48.6 %)**; phone shots 39 (196.6 s) -> 41 (211.6 s, 57.5 %);
camera in motion 96.7 s (26.3 %) -> 91.7 s (24.9 %); 75 shots / 74 transitions -> 76 / 75 (5 flight, 2 crane_up + 3 crane_down + 3 phone ways passing
an overhead = the same 8 crane ways at 173.5-179.4, 237.8-243.3, 279.0-284.3, 333.2-339.4 s, 23 phone_in, 22 phone_out, 18 phone_to_phone, 2
replay); phone ways median 1.12 -> 1.02 s (phone_in 1.0, phone_out 0.86, phone_to_phone 1.32), max 2.5 s; fastest arc 199.9 deg/s (unchanged);
frustum surfaces readable min 0.68 -> 0.62 s, median 0.79 s (81 surfaces); shortest holds phone 1.4 s, 3D 1.0 s (v3.1 1.15), overhead 2.55 s.
The price, physics on screen (film s): load colours 320.1 -> 315.8, the dancers' COM floor dots 292.5 -> 288.8, the counterbalance layer
140.6 -> 131.9, pivot shots 27.3 -> 24.5 (the slivers they had swallowed; the pivot windows are intact) - the phone shots carry the load colours
and the dots too (171.3 / 160.3 s of their 211.6 s). The phones' clips the direction needs: 340.4 s (clips.json). **Not 50 %**: what is left in 3D is the 6 pivots, the 4 overheads, the physics views (from 0.5 s before the first called word to the last
call's arrow + the way out), the replay, the opening under the title card and one calm orbit no phone can show; going further needs fewer of
those (the owner's call).

### The schema (additive; "hm film flow v1")

- `camera.transition_in` (on every segment of a shot but the first): `type`; `start_s` (film s relative to the shot's first segment start;
  negative = it begins in the previous shot); `duration_s`; `easing` ("smootherstep"); `path` ("arc" | "crane" | "arc_then_axis" | "axis_then_arc" |
  "axis_arc_axis"); `from` / `to` {kind, mode, phone, azimuth_deg, radius_m, height_m} (the PLAN's estimate; the viewer uses the live poses);
  `arc` {delta_deg, dir}; `constraints` {height_m [lo, hi] (hi null for a crane), clear_radius_m 1.4, keep_couple_in_view}; `axis_run_m` (1.0) and
  `axis_share` (0.3; v3.0 0.4) for phone ways; `crane: true` when a phone way passes an overhead; `frustum_in` / `frustum_out` {camera, film [a, b, c, d]};
  `video` {full_frame_at_s, splats_hidden_at_s, overlays_on_top_in_s} (phone_in, phone_to_phone); `video_out` {surface_from_s, splats_back_in_s,
  overlays_on_top_off_before_s}; `layers_switch_s` (when this shot's layers take over, relative to its start); `transit_layers` {film, layers}
  (phone_to_phone: the 3D world between the two phones); `from_pose: "frozen"` + `flash_s` (replays); `rule_s`, `squeezed` (information).
- 3D shot keys: `world_azimuth_deg` (the eye's azimuth about the couple's centre at the shot start: atan2(z, x) of eye - centre, Unity degrees),
  `drift_deg_s`, `ground_level: true`, `inline_lift_max_m` (orbit: avoid_inline's lift capped at 0.4 m); physics_3q `elevation_deg`, `look_target`
  (a FilmTargets target) + `look_target_weight`; `shot_key` (keeps two equal pivot views apart: no behaviour).
- overhead: `up_azimuth_deg` + `up_policy: "from_entry"`. Phone shots are all `camera_tour` with `cut: "flow"` (the 5 `fly_to_camera` became
  camera_tour + a phone_in). v3.2: a phone shot's `cameras[0].dance_t` is the master window from the start of its way in to the end of its way
  out (where FilmCamera.CameraTour must return the phone's live pose for the transition paths; the POV check of `clips` asks for pose and
  video there), not the director's original window (a carve or a nudge had left stale windows: ch06_06_pov said 192.86-198.08 s for a shot
  that ends at 193.82 s, and `clips` failed its POV check on it).
- layers: `floorcraft: "pivot_dials"`, `floorcraft_current_t`, `physics_limbs` {width_scale, parts}, `physics_from_above` (information).
- top level `flow` {version, schema, no_jump_cuts, defaults {easing, ground_m, clear_radius_m, frustum_depth_m, axis_run_m, axis_share, rule,
  squeeze_max_s, hold_s, flash_s, crane_extra_s, inline_lift_max_m}, frustums [{camera, film [a, b, c, d], role enter | leave, depth_m,
  video_full_at_film | video_full_until_film, shot}]}.

### What the Unity director must implement (the next stage; not done here - the editor was in use)

1. **FilmCamera: transitions instead of blends and cuts.** For a shot with `transition_in`, the window is [F0 + start_s, F0 + start_s + duration_s]
   (F0 = the shot's first segment); inside it the pose is the path from the PREVIOUS shot's live pose at that film time (`from_pose: "frozen"`: its
   pose at F0) to THIS shot's live pose at that film time, parameter smootherstep(u). Outside it, the shot's own pose. `blend_s`, BlendIn and the
   CutIn of a replay are ignored when `transition_in` is there (the camera never jumps, also over a replay's dance-time jump).
   Paths: **arc** = cylindrical about the couple's centre: azimuth the short way (the live difference, wrapped), horizontal radius and height
   lerped (radius never below clear_radius_m, height clamped to `constraints.height_m`), look point / FOV / LensShift lerped, no roll, the aim
   kept on the couple (FilmCamera.Fly's bump); **crane** = a vertical arc about the look point at the entry (exit) azimuth: elevation from the
   ground pose's to 89.9 deg, distance and FOV lerped, no roll (overhead up = `up_azimuth_deg`, below); an azimuth change is made at ground level
   (crane_down: after the descent); **arc_then_axis** (phone_in) = one C1 curve: the arc to the portal (the phone's position minus axis_run_m along
   its optical axis, oriented as the phone's CROPPED view), then a straight dolly along the optical axis into the phone's position, the last
   `axis_share` of the time; orientation reaches the cropped view's rotation at the portal, FOV the crop's at the phone (a cubic Bezier on the
   eye - P0 from-eye, P3 phone position, P2 = P3 - fwd x 2 x axis_run_m, P1 = P0 + 0.3 x (P2 - P0) horizontally - sampled by arc length is
   enough); **axis_then_arc** (phone_out) = the reverse; **axis_arc_axis** (phone_to_phone) = out along A's axis, arc, in along B's axis (each
   axis part axis_share / 2). Every path keeps the eye in `constraints.height_m` (except a crane), >= clear_radius_m from the couple's centre.
2. **3D shots honour the plan:** orbit, physics_3q, feature_zoom, feet_closeup, rhythm use `world_azimuth_deg` + `drift_deg_s` x (film - F0) instead
   of their own side choice (SideAzimuth / line + 90 + azimuth_deg / azimuth_offset_deg) when the key is there; orbit caps the avoid_inline lift at
   `inline_lift_max_m`; `ground_level` shots lower an eye above 2.4 m to 2.4 m keeping the look point (never below 0.5 m). physics_3q reads
   `elevation_deg` (default 18) and `look_target` / `look_target_weight` (the look point moves that share toward the target, horizontally).
   overhead_floorcraft with `up_policy: "from_entry"`: Up = -(cos a, 0, sin a), a = `up_azimuth_deg` (the horizontal view direction of the shot it
   rose from), instead of Vector3.forward.
3. **The phone frustum surface (CameraVideoRig / FilmCameraSource):** for every `flow.frustums` entry draw THAT phone's frustum (apex at its pose;
   lines to the four corners of its full image rectangle at `depth_m` = 0.9 m along its optical axis, and the rim; the phone's colour,
   CameraVideoRig.ColourOf) with its live video on the rectangle (the FULL strip, frame nearest the dance time, the phone's own vfov / aspect - not
   the crop), alpha 0 -> 1 over [a, b], 1 over [b, c], 1 -> 0 over [c, d]; at alpha 1 opaque and depth-tested (ZWrite on: the splats, skeletons and
   floor marks behind it are hidden by it), alpha-blended while fading; several at once (phone_to_phone). At `video_full_at_film` the camera is at
   the phone pose and the surface projects exactly onto the full-frame video: switch to the existing full-frame quad (Background, ZTest Always,
   3d_over_video) in that frame - the picture must not change; at `video_full_until_film` (role leave) the reverse. FilmCamera.PovVideoOpacity = 1
   from the arrival to the next transition's start, 0 otherwise (the surface carries the picture during the ways).
4. **Layers follow the view:** a shot's layers apply from F0 + `layers_switch_s` (phone_in: at the arrival; phone_out: at the departure, before F0);
   `transit_layers` over their film window; segment layers inside a shot blend as now. `video.splats_hidden_at_s`: SplatDancers.FilmVideoFade = 1
   from then (they are behind the surface already); `video.overlays_on_top_in_s`: the skeleton-on-top lines and the arrows / dots registered on the
   video fade in over 0.35 s after the arrival; `video_out.overlays_on_top_off_before_s`: off 0.25 s before the departure; `video_out.splats_back_in_s`:
   the splats fade back in over 0.35 s from the departure. `avatars_fade_with_video` follows the full-frame video's alpha, not the frustum's.
5. **Replays:** FilmOverlay flashes (0.55 alpha, 0.3 s) at a replay's first frame AND at the first frame after it (the dance jumps forward there);
   the camera flies from the frozen pose (1.2 s).
6. **FloorCraftOverlay:** `floorcraft: "pivot_dials"` = only the counterbalance dials (yellow pivot ring at her anchored foot, the running degrees,
   a tick per beat) and their record - no T, old Ts or transitions; `floorcraft_current_t: true` adds the current T. Physics layer on any view:
   `skeleton_mode: physics` on camera_tour / feature_zoom / feet_closeup / orbit / overhead segments switches the load colours there (SplatDancers'
   physicsView draws them over the splats; on a phone's video the skeleton-on-top lines are coloured), `physics_limbs.width_scale` widens the arm and
   leg segments in physics mode (SkeletonStyle), com_marker in phone shots is drawn through the phone's camera like the arrows.
7. **Run it:** `python -m dancecap.film_recap_flow clips` under the Unity lock (v3.2: the phone shots and frustum surfaces need 340.4 s of phone video:
   clips.json; it keeps the current clips; run 2026-10-21, see "Unity stage v3.2" below), then load `work/larissa_kadu_recap/v3/direction/direction.json` (or copy the folder over film/ after
   backing it up; its audio/ mix is identical), look at the ways in and out with `hm_film_show`, then record both aspects as before (splat dancers on).
   The face sidecar must also carry the frustum surfaces (`surfaces`, v3.1, section "The face sidecar" above): a dancer whose face lies
   behind a phone's frustum video surface is not swapped (the picture there is the phone's video of the real people).

Limits: the azimuths, radii and heights of the plan are estimates from the physics COMs and the phone tracks (the viewer flies between its own live
poses, so continuity holds by construction, but a duration may be off the rule for the live geometry by a little); nothing of v3 has been rendered;
v3.2: 91.7 s of the film is camera motion (24.9 %; v3.1 26.3 %, v3.0 35 %), and the phones' full-frame share (48.6 %; v3.1 44.6 %) is still below the cut film's 52.4 % (above); "rotation" right after "bring her to the
axis" (u034) and "rotate lateral" (u033) are kept as the leader's T (his T rotates there: floorcraft axis pivot 189.0-189.8 s); the "connected"
rule reads the hand-contact events of the v3 hand-contact work - rebuild after they change (`check` says STALE; see v3.1 above).

## Hand contact v3, safety pass (2026-10-21): what the Unity stage installs

An independent review of the first v3 hand-contact result found a whip, a hand on the wrong side of the partner (#118), hands further from
MAMMA's hand points, new penetration in hand-hand holds, fanned-out open hands and hovering palms. `dancecap/hand_contact.py` now has: a
loss on the motion it adds (free to 0.3 m/s), blend ramps that are a pure blend of the core's correction, the flat open hand with the fingers
closed (`finger_template`), the palm itself on the surface for flat holds, an exact push-out, and guards against the INPUT fit on every event
(MAMMA hand points over all core frames <= +3 px, independent COCO wrist / elbow <= +4 px, no new exact penetration, no added jump > 0.55 m/s,
no speed above the input's p99 + 0.25 m/s where the input was slower), then a post-check against the final fit (`pass4`, `revert`). Decisions
taken by looking at the phone crops are in `v3/hand_contact/curation.json`, each with its reason: #118 re-targeted to her left upper back
(anchored hold, core 223.2-239.6 s), 15 of the review's 16 flagged holds and #17 / #99 / #104 left as the input fit. Numbers:
`v3/hand_contact/validate.json` (81 refined holds; speed audit over the whole class 0 incidents, was 287; new penetration 0; MAMMA > 3 px
worse only #118, curated). The motion is exported to a side capture only (`work/larissa_kadu_recap/export/06_LarissaKaduRecap_v3hands/`);
the Unity stage installs the four motion files with `bash dancecap/work/larissa_kadu_recap/v3/unity_motion/install_motion.sh` (checks
the sha256 recorded by the export and the frame layout, backs up the installed files once, `--check`, `--restore`). `events.json` is
unchanged, so the direction's 'connected' rule is not stale.

## Unity stage v3.2 (2026-10-21, under work/unity.lock 15:28-16:00 local): motion installed, the phones' clips, the capture audited

- `install_motion.sh --check` (all four sources match the export's sha256, the .bin sizes match), then the install: the pre-v3 motion is backed
  up once in `v3/unity_motion/backup_installed_motion/` (+ capture.json), and the four files in `StreamingAssets/06_LarissaKaduRecap/` are now the
  v3 hand-contact motion (`--check` afterwards: all identical).
- `python -m dancecap.film_recap_flow clips` for the v3.2 direction: phone strips 00 3376 / 01 1611 / 02 2490 / 03 2665 / 05 1606 / 06 1674 frames
  (clips per phone 112.7 / 53.7 / 83.0 / 88.9 / 55.4 / 55.9 s, the current clips kept), 49 POV windows OK, **pov_ok: True**, installed into the
  capture's cameras/ (the editor's asset refresh made the .meta files again). The first two runs failed the POV check and were the reason for two
  v3.2 fixes above: a phone shot's camera window was the director's stale one (ch06_06_pov), and the way into phone 05 began before it records.
- `film_recap_flow check`: OK (time map = film/'s at 3677 points, captions identical, 'connected' rule up to date). Capture audit (every file
  capture.json references present and parseable; the motion = the v3 export; splats/lead + follow sha256 as their manifests, the follower's 35899
  hair splats + hair binding block, hair_groom.json; the physics / counterbalance / floorcraft / timing layers; every phone's HMJ strip readable
  and covering every range clips.json needs within the phone's recording): complete. The director items 1-6 were implemented next (below).

## Unity director v3 (2026-10-21, head-movement, under work/unity.lock): items 1-6, the face sidecar, the first v3 excerpts

**Which direction the recap plays.** `FilmRecorder.FindDirection(capture)` (used by `hm_film record`, `hm_film_show`, the Record / Film Preview
menus and the recorder window) walks `StreamingAssets/<capture>/film/`, then for every `dancecap/work/<take>/` first `v3/direction/` and then `film/`, and takes
the first `direction*.json` whose `"capture"` is the capture folder. So `06_LarissaKaduRecap` now records and previews
`work/larissa_kadu_recap/v3/direction/direction.json` (its narration timeline and its `audio/larissa_kadu_recap_mix.wav` are found beside it, as for
film/); the other films have no `v3/` folder and are unchanged. The old recap stays reachable: `hm_film record --direction <film/direction.json>`,
`hm_film_show --action start --direction <...>`. A recording of a direction with a top-level `flow` gets the file-name tag `_v3`
(`06_LarissaKaduRecap_vertical_<date>_t<a>-<b>_v3.mp4`; `--tag` overrides). The viewer's film library (desktop playback bar, the Quest) keeps
its own direction paths (`film_library build`): it still plays film/direction.json until the library is rebuilt from v3 (not done here).

**Gating.** Everything below runs only when the direction has `flow` (`FilmDirection.Flow`, `HmFilm.FlowTimeline.Build`); a direction without it
plays exactly as before (blends, cuts, segment layers, the old flash, the old overhead height).

Code (head-movement): `Assets/Film/Flow/` (new assembly `HmFilmFlow`: `FlowMath.cs` - smootherstep, envelopes, cylindrical coordinates, the
composite path `FlowPath`; `FlowTimeline.cs` - the schema, shots, phone windows, surfaces, covers, splat visibility, layer starts;
`FaceSidecarMath.cs`), `FilmCamera.cs` (transitions, the 3D keys), `FilmDirector.cs` (layer entries, surfaces, covers, limbs, dials),
`CameraVideoRig.cs` / `FilmCameraVideo.cs` / `FilmCameraSource.cs` (surfaces, covers), `Resources/HM_CameraVideo.shader` (`_ZWrite`),
`FilmOverlay.cs` (flashes, the POV pill), `Floor/FloorCraftOverlay.cs` (`DialsOnly`), `Splats/Avatar/SplatDancers.cs` (`FilmVideoFade` follows the
flow; `SplatOf`), `Splats/Avatar/Core/SkinnedSplatAvatar.cs` (`TrySkinNow`), `FilmFaceSidecar.cs`, `Editor/FilmRecorder.cs` / `FilmRecorderCli.cs`
(`--faces`, `--direction`, `--tag`), `Editor/FilmRenderCheck.cs` (the sidecar in the report), `Editor/FilmFlowCheck.cs` (`hm_film_flow`).

What it does (the numbers of RECAP_FILM.md "What the Unity director must implement"):
1. **Transitions.** The camera state is a pure function of film time (`FlowTimeline.CameraAt`): a shot's own pose, or the transition of the shot
   whose window [F0 + start_s, + duration_s] contains the time (start_s < 0: it starts in the previous shot; > 0: the previous view holds into the
   shot). Inside it the pose is the path from the previous shot's LIVE pose (`from_pose: frozen`: its pose 1 ms before F0) to this shot's live pose,
   eased by one smootherstep. `blend_s`, BlendIn and the replay's CutIn are ignored. Paths:
   - `arc` (3D to 3D, the replays): cylindrical about the couple's centre, the azimuth the plan's way round (`arc.delta_deg`: a planned turn over
     120 deg keeps its direction when the live difference wraps), radius >= clear_radius, height inside the band, look point / FOV / composition
     lerped (the aim stays on the couple).
   - the phone ways and the cranes are ONE polyline of (eye, rotation) keys sampled by cost length (metres + 0.25 m per radian turned), so the
     speed is continuous across the parts: [out along phone A's full optical axis to its portal, axis_run_m behind it] + a cubic Bezier in
     cylindrical coordinates about the couple (tangent to the axes: G1 with the dollies) + [in along phone B's axis]. The rotation turns inside
     the Bezier only (eased, so it joins the dollies without a jolt), pulled toward the couple mid-way, and reaches the phone's CROPPED view at
     the portal; the FOV lerps to the crop's.
   - **Deviations the live geometry forced** (measured with `hm_film_flow`, the stills and the excerpts): in 28 of the 45 phone ways the 3D
     end stands near the phone - within 1.5 m to its side, 0.6 m behind to 1.3 m in front of it, mostly 0.2-1.4 m above (`flowcheck_*.json` "geometry"). A straight "back along the
     axis, then arc" then doubled back on itself or cut through the picture (the first 22.5-s way flew into phone 00's picture and out the other
     side). So: (a) a U-turn handle grows with the turn it has to make (no hairpin at the axis); (b) each way gets a plan, decided once from the
     live poses at its start, middle and end (cached: the path never changes shape mid-way; the 3D end of the 22.5-s way moves 0.5 m during it),
     of the least summed "badness" over lifts 0-1.5 m and sideways swings 0-100 deg: crossing a picture 1000, a hairpin over 30 deg between
     samples 5 per degree, a stall 20, plus the effort (`hm_film_flow --action plan` shows the choice); (c) when the 3D end stands inside
     a phone's frustum before its picture, or the straight way to it passes through the picture, the picture stands closer to its phone (0.75 /
     0.6 / 0.45 / 0.3 m instead of 0.9; `FilmCamera.SurfaceDepth`; the projection from the phone is the same at any depth, so the hand-off is
     unchanged); (d) the cranes are one vertical arc straight between the overhead and the ground end (the portal or the 3D eye), the elevation
     leaving / reaching the ground with zero slope, the azimuth turning near the ground, the aim locked on the look point (with the
     slerp-plus-bump of the phone ways the couple left the top of the frame for 0.3 s on the way up); (e) a picture fades (0.25-0.9 m) while
     the eye passes close beside or in front of it - not on the phone's side inside its outline, where the dolly and the hand-off are - so a way
     that swings round it never wipes it across the lens in 3 frames.
2. **3D shots.** orbit, physics_3q, feature_zoom, feet_closeup and rhythm use `world_azimuth_deg + drift_deg_s x (film - F0)` (not clamped before
   F0: the transition into the shot sees the same drift); the orbit's avoid_inline lift is capped at `inline_lift_max_m`; `ground_level` keeps the
   eye in [0.5, 2.4] m; physics_3q reads `elevation_deg`, `look_target` / `look_target_weight` (damped like the couple follow). The overhead with
   `up_policy: from_entry` puts the top of the picture along the view it rose from - the LIVE azimuth of the previous shot's eye at the
   transition's start (`up_azimuth_deg` only as the fallback) - with the eye at 89.9 deg on that azimuth. Also (flow only): the damped follows
   reach back across shot boundaries and take new samples in with a weight ramp (a few-cm step at every shot start, hidden by the old cuts,
   showed in a flight); the overhead's height is a weighted mean over +-0.24 s (its 'never crop' need jittered 20 cm per frame in a pivot).
3. **Frustum surfaces.** For every `flow.frustums` entry with alpha > 0: the frustum lines (phone colour; the four edges fade out within 2 m of
   the phone and keep about a constant width on screen, so they never sweep across the lens as bars; the rim fades with the cover) and the
   phone's live FULL strip frame (the same texture and frame index as the full-frame video) on the image rectangle at its depth, queue 3100,
   ZTest LEqual, ZWrite at alpha 1. The envelope [a, b, c, d] ends (enter) / starts (leave) exactly at the camera's arrival / departure, so in
   every frame either the surface or the full-frame video carries the picture. **The hand-off cover** (instead of fading each overlay
   separately): the picture is drawn a second time over every 3D overlay (queue 4500, ZTest Always) - the surface's copy ramps to 1 over the
   last overlays_on_top_in_s before the arrival, the full-frame video's copy starts at 1 and fades over overlays_on_top_in_s after it (and back
   over overlays_on_top_off_before_s before the departure), each held 0.04 s at the switch. The frame before the switch and the frame of it both
   show the phone's video alone; the skeleton lines, arrows, COM dots, the T and the POV pill then fade in together. `PovVideoOpacity` = 1 from
   the arrival to the next way's start, 0 otherwise.
4. **Layers.** A layer timeline: every segment at its F0, the first segment of a shot at F0 + `layers_switch_s`, the `transit_layers` of a
   phone-to-phone way over their window; continuous values blend over 0.8 s from each entry as before. The splat dancers follow
   `FlowTimeline.SplatVisibility` (hidden from `splats_hidden_at_s`, back in over `splats_back_in_s` from the departure; kept when the phone's
   video is missing).
5. **Replays.** The flash also marks the first frame after the replay (a `replay_out` with `flash_s`), and in a flow film both flashes fade as
   alpha x rest^2.2 (a linear alpha left a grey of about 70 on the last flash frame and dropped to the picture in one step). The camera flies
   from the frozen pose (1.2 s).
6. **Floor craft and physics.** `floorcraft: "pivot_dials"` = the record with only the counterbalance dials (no old Ts, T-to-T path, axis
   pivots or spin dials); `floorcraft_current_t` adds the current T. `physics_limbs.width_scale` widens the arm and leg lines in physics mode
   (blended like the other layers). `skeleton_mode: physics` on phone / zoom / feet / orbit / overhead segments already switched the load
   colours (`SegLayers.Physics`); com_marker annotations are world-space, so they are drawn through the phone's camera on the video.

**Face sidecar** (`Assets/Film/FilmFaceSidecar.cs`): `hm_film record ... --faces auto|on|off` (auto = with the splat dancers) writes
`<recording>.faces.json` ("hm film faces" v1 as specified above): sampled in the director's late pass of every recorded frame (frame count =
the MP4's), S_15 = `SkinnedSplatAvatar.TrySkinNow(15)` (the same matrices the splat skinning uploads, computed from the posed bones; with splats
off the avatar's head bone x its bindpose), the canonical points from the capture's `splats/` or `work/<take>/v3/faceswap/face_points_canonical.json`
(path + sha256 in the file), the recording camera, the recorder's output size, `ui` = the visible uGUI graphics of the film canvas merged into
blocks + the legend, `surfaces` = every frustum surface drawn (polygon clipped against the near plane, a depth per point, `role`, `full`). The
`.report.md` names it.

**Checks.**
- Edit-mode tests `Assets/Film/Flow/Tests/Editor/FlowTests.cs` (assembly HmFilmFlow.Tests): smootherstep, the envelopes and their hand-off,
  the plan's turn direction, exact ends / continuity / G1 into the axis / clear radius of a phone way, a crane without roll, the timeline's
  camera states, windows, covers and splat visibility on a synthetic direction, the face geometry, and - when the workspace has it - the v3
  direction itself (a transition on every shot but the first, one easing, the 8 crane ways at the 4 overheads, holds >= 1.4 s, every frustum
  tied to its shot, an opaque surface the frame before every arrival and at every departure), and a plan that goes round a picture the straight
  way would cross (15 tests). All 84 edit-mode tests of the project pass.
- `unity command hm_film_flow --action check [--from --to --out]` (Play mode, the film started): sweeps the film at 30 fps through the live
  director and reports per-frame camera jumps, kinks, the ground band, the couple out of frame during a way, the hand-off frames, the surfaces'
  on-screen readable time, the holds, the fastest turn of each way and each phone way's geometry; `--action series` dumps a window frame by frame.
- `work/larissa_kadu_recap/v3/unity_film/jumpcuts.py <mp4> --start S --sheet` (frame-to-frame difference spikes: > 3 x the median of +-7 frames
  and > 4/255) and `sidecar_check.py <mp4>` (validates the sidecar with `film_faceswap.validate_sidecar`, draws the points, surfaces and UI rects).

**Results (vertical 1080x1920, the splat dancers + swinging hair on, D3D12, `--faces on`; renders git-ignored).**
- `hm_film_flow --action check` over the whole film (11031 frames; `work/larissa_kadu_recap/v3/unity_film/flowcheck_vertical_final.json`):
  no frame where a phone shot's own pose is not the phone's; the eye inside 0.5-2.4 m in every frame outside the 8 crane ways and the 4 overheads;
  shortest full-frame phone hold 1.40 s; the fastest turn about the couple of any way 168 deg/s (phone to phone at 322 s), cranes up to 143 deg/s;
  77 of 81 hand-off frames show the phone's picture alone over the whole frame at alpha 1 with its cover at 1 - the other 4 (phone 06 at 339.4 /
  344.6 s, phone 00 at 127.2 s, phone 02 at 327.4 s) have the crop reaching 1-2.5 % past the phone's picture at an edge, the same black edge
  the full-frame video shows there (the plan's crop, not the hand-off). The detector's 16 "jump" and 18 "kink" flags are turn-rate peaks of
  1.4-3.5 deg per frame in the middle of fast ways and the overhead at 240 s (looked at: continuous); the couple's centre leaves the frame for 4
  + 3 frames in the long ways into phone 03 (73.9 s) and phone 05 (317.6 s).
  **Not met: the pictures' on-screen readable time.** Measured as alpha >= 0.5 while the picture covers 1-97 % of the frame with every corner in
  front of the eye, 40 of the 81 surfaces read for >= 0.6 s (median 0.57 s, shortest 0.23 s; v3.2's own floor counts the envelope from its fade-in,
  0.62 s, but for the first part of most ways the picture is outside the 3D view's frame). Pulling the aim toward the pictures (tried: +1
  surface, the couple out of frame 11 instead of 7 frames) does not change it; longer ways in `film_recap_flow` would.
- Pictures stand closer than 0.9 m in 8 ways (0.30-0.75 m: the 3D views at 22.5, 34.7, 79.2, 90.4, 106.4, 127.2, 160.0 and 315.2 s stand just in
  front of or above their phone); 10 ways carry a small detour (lift 0.25-0.5 m; the 22.5-s way swings 30 deg round its picture);
  `hm_film_flow --action plan` lists them.
- 13 excerpts in `head-movement/Recordings/06_LarissaKaduRecap_vertical_2026-10-21_t<a>-<b>_v3[_nosplats].mp4` (each with `.faces.json`,
  `.report.json/.md`, `.audiofix.json/.log`, `_rawaudio.mp4`; all reports PASS, A/V 0.0 ms, the audio replaced from the mix; the earlier passes in
  `Recordings/v3_excerpts_superseded/`), every frame run through `jumpcuts.py` (sheets and `.jumps.json` in `work/.../v3/unity_film/`):

  | excerpt (film s) | what | frame-difference spikes |
  |---|---|---|
  | 12.3-15.7 | phone 05 -> phone 00 (both pictures in the world) | none |
  | 22.0-26.4 | out of phone 00 round its picture into the 'frame' zoom, into phone 02 | none (the first pass flew through the picture: 3 spikes, d 51-90) |
  | 41.6-46.5 | into phone 02, phone 02 -> phone 01 | none |
  | 147.3-148.7 | the fastest phone-to-phone turn (3 deg per frame) | none |
  | 159.5-164.3 | out of phone 00 into pivot 1 "back to the axis" (dials, "axis (pivot point)") | none |
  | 172.8-180.6 | out of phone 01 + crane up, the overhead, crane down into pivot 2 | none |
  | 228-244.5 (+ `_nosplats`) | physics view (momentum, weight heading, load colours, COM dots), #118, crane up / down | none |
  | 245.8-255.6 | the replay: flash + flight from the frozen pose in, the same out | the 2 replay cuts (under their flashes); a caption word at 255.2 s (d 4.6) |
  | 297.5-301.6 | out of phone 02 into pivot 5 and back into phone 02 | none |
  | 331-337 (+ `_nosplats`) | pivot 6, crane up, the overhead with #155 | none |
  | 354.4-361.6 | physics view, into phone 00 | none |
- Looked at (sheets every 0.2 s and every frame around each switch): the pictures stand in the world with their frustum lines and are hidden
  by / hide the splat bodies correctly; the hand-off frames are the phone's video on both sides; the overlays then fade in; no apex blob or bars
  near the lens (the first pass had both); the cranes keep the couple in the frame (the first pass lost it at the top for 0.3 s); the pivot views
  show the counterbalance dial and the "axis (pivot point)" arrow at her anchored foot, the widened arm / leg load lines and the balance axes.
- Hands: #118 - his right arm passes her left side and the hand-contact light sits at her left upper back through 228-237 s, with the splats and
  with the 25 % avatars; the 239.6-240.8 s return (the overhead) moves smoothly. #155 (film 332.2-336.5 s: pivot 6, the crane, the overhead) - no
  elbow wobble or finger flicker visible frame by frame from above or in the pivot view, so it was NOT reverted.
- Face sidecars: all 13 validate (`film_faceswap.validate_sidecar`: entries = the MP4's frames); `sidecar_check.py` sheets
  (`*_v3.sidecar.jpg`): the 5 points sit on both faces in 3D frames (yaw / iod plausible) and on the real heads in the phone frames, the
  `surfaces` polygons lie on the pictures (phone 05 leaving + phone 00 entering at 13.4 s), the UI rects on the captions, pills and legend;
  the reader would leave 0-85 face-frames per excerpt (both dancers) alone because a picture stands in front of them. The splats-off sidecars use the avatar's
  head bone (same layout).

**Not done / limits.** No full film and no 16:9 render (the side fill of a portrait clip now fades with the cover, untested in a recording);
the readable-time shortfall above; the crop of phones 00 / 06 reaching past the picture's edge at four hand-offs is the plan's; the splat dancers
were left OFF in the editor afterwards (SessionState), the lock released.

## The whole v3 films, recorded and face-swapped (2026-10-21 18:56 - 10-22 00:13 local; Unity under work/unity.lock 18:56-19:21)

**Recording.** Nothing heavy running (the pipeline monitor and a review server idle). Play mode `hm_splatdancers --action on` (hair
Skinned on the simulated strands, skeleton over splats Physics, D3D12), out of Play mode, then `unity command hm_film --action record
--aspect vertical --capture 06_LarissaKaduRecap --faces on` and the same with `--aspect horizontal`; the v3 direction was found by
`FindDirection` and the `_v3` tag set itself. Both reports PASS: 11032 frames = 367.733 s x 30 (film 0-367.677 s), 1080x1920 / 1920x1080,
splatDancers true, ordered readback (at most 1 frame pending, 0 timeouts), 444 / 445 s real (24.8 frames/s). The audio fix ran by itself
on both: PASS, video md5 unchanged, -16.0 LUFS, true peak -1.3 dBTP, sync 0.0 ms in 30 windows (r >= 0.99968), 0 burst blocks; the
recorder's own track again had the burst at 56.860-56.900 s (peak 199.6), kept in `_rawaudio.mp4`. Face sidecars: 11032 entries each,
`validate_sidecar` clean (vertical: 5675 3D frames, 5357 phone-video frames, 2694 frames with a frustum surface). Looked at 10 stills
per aspect (`scratchpad`, not kept): splat bodies with hair, captions, the frustum pictures standing in the world, the overheads, the
pivot dials, the replay pill. The splat dancers were switched OFF again (Play mode, `--action off`) and the lock released at 19:21.

**Face swap** (`conda run -n facefusion ... film_faceswap swap --film <rec>_v3.mp4 --sidecar <rec>_v3.faces.json --out <rec>_v3_faces.mp4`,
then `verify`; logs `v3/faceswap/run_full_<aspect>_v3[_final].log`, `verify_full_<aspect>_v3_final.log`):

| | vertical | horizontal |
|---|---|---|
| analysed frames (3D, a dancer in view) / speed | 5672 of 11032, 9.1 film frames/s | 5675 of 11032, 10.0 film frames/s |
| detections confirming the projected points (Kadu / Larissa) | 2327 / 1565 | 2356 / 1568 |
| swapped face-frames (Kadu / Larissa frames) | 1780 (1255 / 525) | 1615 (1209 / 406) |
| swap spans (Kadu / Larissa) | 16 / 9, shortest 0.87 / 0.97 s | 14 / 7, shortest 1.0 / 0.97 s |
| face-frames flagged behind a frustum picture, never swapped (Kadu / Larissa) | 1124 / 1143 | 1109 / 1127 |
| overlay strokes kept over a swapped face | 348 frames | 321 frames |
| pixels changed outside the faces before encoding | 0 | 0 |
| luma PSNR outside the faces (CRF 14) / flicker ratio | 48.2 dB / 0.991 | 49.1 dB / 0.991 |
| swap pass | 8.3 frames/s, 116 ms per face (boost 128) | 8.6 frames/s, 149 ms per face (boost 128) |

Checked on each final against its recording (`v3/faceswap/final_check_<aspect>_v3.json`, made by `v3/faceswap/tools/verify_final.py`; sheets by `look_frames.py`): 11032 frames in both files; video starts at
0.033 s and audio at 0 in both (as recorded); the audio stream is bit-identical to the recording's (packet md5); in sync with the mix
at 14 points (0.0 ms, r >= 0.99978); 0 burst blocks; -16.0 LUFS / -1.3 dBTP like the mix; at 16 places across the film picture k of the
final matches picture k of the recording (not k - 1 or k + 1). The phones' own video and every frame where a picture stands in front of a
face are untouched.

**Three faults the first full run showed, fixed in `film_faceswap.py` before the final run** (tests 14 pass, two new):
1. **The reader repeated the first frame.** Without `-fps_mode passthrough` ffmpeg's constant-rate rawvideo output filled the half-frame
   gap after the seek by duplicating the first frame: buffer k held frame first + k - 1. The swap therefore applied sidecar entry i to the
   picture of frame i - 1, and the output repeated frame 0 and lost the last frame (`verify` did not see it: it read both files the same
   way). Found by comparing decoded frames directly. **Every output made before this fix carries it** (the prototype excerpts A-D,
   `tests/`, `cli_test/`: picture one frame late against its audio, metadata one frame early). With the frames aligned, the detections
   that confirm the projected points rose from 1841 / 1151 to 2327 / 1565 (vertical).
2. **The video start.** The Writer's `-itsoffset` is dropped for an encoded raw pipe (the swapped video started at 0, one frame early
   against the copied audio once the reader was right). `restore_start` remuxes the swapped video with the recording's audio and
   timescale (stream copy, both md5-checked).
3. **Overlay strokes over a face.** In the physics views the grey / blue skeleton lines cross the faces (Larissa 315.6-316.8 s, Kadu
   286-288 s, 299-300 s); xseg did not always see them as occluders and the swap painted them partly away (a broken, flickering line).
   `overlay_mask` / `restore_overlay` put the recorded pixels back on those strokes (+ 1 px rim), so the line stays in front of the face.

**Looked at** (`v3/faceswap/look_full/*_frames.jpg` = 20 frames of each final, `*_faces.jpg` = before | after of every swapped face in
them): the right identity on every swapped face (Kadu's beard and eyes, Larissa's eyes and lips sharper than the splat faces), the
skeleton strokes intact over the faces, captions and pills untouched, phone video and the frustum pictures never swapped; the 16:9 hand-off
at 25.9 s (phone 02) shows the same picture on both frames.

**Limits.** The faces are small in most of the film (median inter-ocular 19 px vertical, 14 px horizontal; boost 128 throughout), so the
gain is modest and fast turns keep the splat faces (v3.1 rule). The light slash on Larissa's left cheek is in the splat render, not the
swap. The film starts with Kadu swapped for 0.87 s (fade-in from black, an open range end where the 1-s minimum does not apply), then
unswapped 0.87-1.97 s. At 190.97-191.33 s Kadu's swap fades out and back in while Larissa's head passes in front of his face (0.37 s gap).
In the two grey flash frames at 253.67-253.70 s (measured on the vertical) the stroke colour test also matched the washed-out faces, so the swap is mostly undone on
those 2 frames (a flatness test - strokes have luma std <= 3, the washed faces 7-19 - would separate them; not built). Red / orange
tension strokes are left to xseg. Superseded outputs (the shifted first run, the run without the stroke restore) are in
`head-movement/Recordings/v3_faces_superseded/`.

## v4.2 (owner 2026-10-22): the physics meter only at each switch-on, the orbits 20 % closer, the face swap on the splats only

Owner, after v4.1: "just show the physics meter at the very top of the frame for a few seconds every time we switch physics mode on and then
fade it out. I like the camera motions, but in gauss splat mode let's get 20% closer to the dancers when we orbit. on the very final pass
let's apply the face fusion to the gauss splats only ... just plan the motion and render once and I'll review". Nothing was recorded for
v4.2 (editor stills and play-mode sweeps only); the render is one later pass.

    python -m dancecap.film_recap_flow build --version v4      # = v4.2 -> work/larissa_kadu_recap/v4/direction/ (review_v4.2.md, UNITY_CONTRACT_v4.md)
    python -m dancecap.film_recap_flow build --version v4.1    # v4.1 as it was -> v4/direction/v4.1/
    python -m dancecap.film_recap_flow check --version v4      # the v4.1 targets + v4.2's (meter, closer orbits, 16:9 ways)

**How it is built.** `build_v42` rebuilds the v4.1 plan (meaning-first: every move, its timing and every orbit rate stay v4.1's - 17 moves,
ways 4.0-6.35 s, orbits 2.6 / 3.0 / 3.4 / 4.5 deg/s, 4 phone holds, mean 25.36 s) and changes only:

1. **The meter** (`flow.meter`, schema "hm film meter v1"). From the layer timeline (each segment's layers from its layer start, as Unity's
   FlowTimeline.LayerStart; transit layers count) one window per switch-on of the physics view: fade in 0.3 s, shown to 4 s after its start,
   fade out 1 s (smootherstep). A window that would meet a chapter card or the title starts 0.2 s after the card; a physics view too short
   for a window (under fade-in + 1 s + half the fade-out) gets none. This film: 3 switch-ons (20.78, 271.10, 349.99 s) -> windows 20.78-25.78,
   274.25-279.25 and 353.14-358.14 s (the last two after chapter 9's and 12's cards); 14.8 s of meter in 320.1 s of physics colours; 0 frames
   without physics, 0 over a card. Place: the very top of the safe area, top right, in both aspects (9:16 276.8 px down, 1.65 x - the place
   v4.1 had; 16:9 62 px down at the title-safe side margin, 1.5 x - new). Unity: FlowTimeline.MeterWindows / MeterAlpha, FilmOverlay sets
   SkeletonLegend.Alpha = meter alpha x (1 - card / title alpha); its rectangle is a label obstacle in both aspects (9:16 also lowers the
   label ceiling); the face sidecar lists the legend in `ui` only while it shows.
2. **The closer orbits.** Every slow-orbit block (the orbits; the pivots, the physics views and the replay ride on it) at 0.8 x v4.1's eye
   offset about its look point: radius 4.3 -> 3.44 m (9:16), 3.6 -> 2.88 m (16:9), eye height 2.15 -> 1.91 m (the same look-down angle), the
   inline lift's cap 0.25 -> 0.2 m; the eye distance to the look point 4.46 -> 3.57 m / 3.80 -> 3.04 m. `lens_distance_scale` 1.25: Unity
   chooses the lens for the v4.1 distance - with v4.1's lens rule (CoupleFov of the distance) the lens would widen with the closeness and the
   16:9 couple would stay the same size. No radius reaches the 1.4-m clear radius.
   - **The phone ways keep v4.1's paths.** With the orbit at 0.8 x everywhere the phone ways (same durations) had to cover more distance and
     their eye speed rose over the 1.2 m/s cap (python, first build: 9:16 1.27, 16:9 1.56 m/s). So the orbit is at v4.1's distance at each
     phone way and eases to 0.8 x over 6 s inside the orbit (`closer_in_s` / `closer_out_s`, smootherstep, the lens fixed: a slow dolly,
     about 0.28 m/s at its peak); 7 eases, 141.7 of 172.9 s of settled orbit at the full 0.8 x. Next to a crane or the replay the orbit
     stays at 0.8 x.
   - **The framing bands.** With the dancers larger, v4.1's 9:16 band (0.31-0.77 of the height, made to keep them under the always-on legend)
     put their feet onto the caption block, and so did the 16:9 framing (editor stills at 60 s). v4.2: 9:16 `vertical_band` 0.34-0.84, and a
     new 16:9 `horizontal_band` 0.24-0.99 (floor .. 1.9 m; the near foot sits about 6 % below the couple centre's floor point the band fits).
3. **The face swap** (`film_faceswap`): with a sidecar a face is swapped only where the recorder says it is a splat-dancer render (per dancer
   `layer` "splat", weight x `la` x (1 - video_alpha)); never the phones' own video, a frustum picture, a mesh-avatar or skeleton-only view.
   FilmFaceSidecar writes `layer` / `la` (SplatDancers.ShownFor); an older sidecar falls back to the frame's `splats` share. `swap` refuses
   to run without a sidecar unless `--heuristic`; `--phone-shots swap` is retired. Run it once, on the final recording.

**Measured** (python PoseModel on the plan; Unity `hm_film_flow --action check` on the live director in Play mode, whole film, both aspects,
v4.1 and v4.2; the sweeps are in `v4/direction/stills_v4.2/flowcheck_*.json`):

|  | v4.1 9:16 | v4.2 9:16 | v4.1 16:9 | v4.2 16:9 |
|---|---|---|---|---|
| block radius about the couple (m) | 4.30 | 3.44 | 3.60 | 2.88 |
| eye distance to the couple, settled orbits, median (Unity) (m) | 4.47 | 3.59 | 3.80 | 3.06 |
| orbit lens, median (Unity) (deg) | 60.0 | 60.0 | 39.7 | 43.3 |
| couple's height in the frame at 60 s (editor still, read off by eye, heads to feet) | about 645 px (34 %) | about 800 px (42 %) | about 680 px (63 %) | about 775 px (72 %) |
| closest the eye comes to a dancer (Unity, pelvis, horizontal) (m) | 3.25 | 2.83 | 2.71 | 2.27 |
| phone ways: peak turn / view rotation / eye speed (Unity) | 7.41 / 8.60 / 0.886 | 7.41 / 9.66 / 0.886 | 7.43 / 9.13 / 1.207 | 7.43 / 10.08 / 1.207 |
| cranes: peak pitch (Unity) (deg/s) | 19.75 | 19.10 | 19.27 | 18.45 |
| jumps / stutters / eye outside 0.5-2.4 m / heads out of frame (3D) (Unity) | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| heads under the meter while it shows (3D / phone frames) (Unity) | - | 0 / 0 | - | 0 / 0 |

(the 16:9 v4.1 phone ways already peak at 1.207 m/s; the 16:9 band widens the lens a little, so the 16:9 dancers are about 1.14 x v4.1's
size, the 9:16 about 1.24 x. The 16:9 hand-off lines read "CHECK fills 0.5" in both versions: the portrait phone clip is a centred panel
there, a property of the check, not a change.)

**Looked at** (editor stills, `v4/direction/stills_v4.2/`): 22.0 s 9:16 and 16:9 (phone 00 hold just after the first switch-on: the meter
at the top, clear of the heads and the POV pill), 26.0 s 9:16 (the same hold 4 s later: load colours on, meter gone), 60.0 s 9:16 and 16:9
(a settled orbit at 0.8 x, with v4.1's frame at 60.0 s next to it), 274.6 s both aspects (the second window over the orbit as the crane
starts: the heads under the meter's bottom edge, the feet above the captions). The editor's own "Enter VR" button shows at the top right of
the Play-mode stills.

Tests: `dancecap/tests/test_film_recap_flow.py` (meter windows, layer starts and transit layers, closer blocks, the eases), `test_film_faceswap.py`
(only splat-dancer faces swapped; no swap without a sidecar), `test_film_jev.py`; Unity HmFilmFlow edit-mode 26/26 (new: the meter's fade,
the closer factor, and on the v4.2 direction itself: one window per switch-on, 0.3 / 4 / 1 s, none over a card, 0.8 x radii, the eases at
every phone way).

## The v4.2 films, rendered once (2026-10-22: recorded 19:23-19:50, face-swapped 19:50-21:45 local; work/unity.lock held 19:09-21:53)

**Vertical final** - `head-movement/Recordings/06_LarissaKaduRecap_vertical_2026-10-22_v4.2_faces.mp4` (423.7 MB, 1080x1920, 30 fps,
11032 frames = 6:07.7, AAC 48 kHz; phone copy `..._v4.2_faces_phone576x1024.mp4`, 63.1 MB, the same frames and the same AAC track
bit for bit). The whole v4.2 film: the physics meter only in its three switch-on windows (sidecar: 20.87-25.57, 274.30-279.03,
353.20-357.93 s, 0 frames outside), the slow orbits at 0.8 x, the splat dancers with the swinging hair (D3D12), the faces of the splat
dancers swapped. Made: `hm_film record --aspect vertical --capture 06_LarissaKaduRecap --readback ordered --audiofix on --faces on --tag v4.2`
(v4/direction found by FindDirection; installed lead motion sha256 8bd1d1ad...; the recorder first rebuilt the soundtrack mix, older
than the direction) -> the raw recording `..._v4.2.mp4` (542.7 MB; report PASS 11/11, 20.9 frames/s, ordered readback at most 1
pending, 0 timeouts; audio fix PASS: video md5 unchanged, first picture at 0.000 s with the sound from 0.000 s, 0.0 ms in 30 windows,
-16.0 LUFS / -1.3 dBTP, 0 burst blocks; the recorder's track with its 56.86-s burst kept as `_rawaudio.mp4`), then `film_faceswap swap
--sidecar <rec>.faces.json` + `verify` (FaceFusion v4, inswapper_128, the two headshots): 2102 face-frames swapped (Kadu 827 in 8
spans, Larissa 1275 in 14; shortest 1.23 / 0.97 s), every one on the splat layer, none on a phone's video (max video_alpha 0) or behind
a frustum picture, 0 px changed outside the faces before encoding, luma PSNR outside the faces 49.6 dB (CRF 14), flicker ratio 0.991;
audio stream identical to the recording's, A/V start 0 frames apart. Logs, checks and the contact sheet (stills every 15 s + one in
each meter window, looked at once: no black or broken frame, dancers in every 3D still, the meter only in its windows) are in
`work/larissa_kadu_recap/v4/final_v4.2/`.

**Horizontal final** - `head-movement/Recordings/06_LarissaKaduRecap_horizontal_2026-10-22_v4.2_faces.mp4` (384.6 MB, 1920x1080,
30 fps, 11032 frames = 6:07.7; phone copy `..._v4.2_faces_phone1024x576.mp4`, 57.4 MB, same frames, same AAC track). The same film in
16:9 (meter 62 px down at the title-safe side margin, the same three windows and 0 frames outside them). Made the same way right after
the vertical (`--aspect horizontal`): raw recording `..._v4.2.mp4` (538.0 MB; report PASS 11/11, 21.3 frames/s, ordered readback, 0
timeouts; audio fix PASS with the same start shift, 0.0 ms in 30 windows, 0 burst blocks), then the face swap + verify: 2232
face-frames swapped (Kadu 908 in 8 spans, Larissa 1324 in 15; shortest 1.23 / 0.97 s), all on the splat layer, none on phone video
or behind a frustum picture, 0 px changed outside the faces before encoding, PSNR outside the faces 50.4 dB, flicker ratio 0.992,
A/V start 0 frames apart. Contact sheet and checks next to the vertical's; nothing blocking was seen (both films open on the title's
fade from black at 0 s, as designed), so neither aspect was re-rendered.
