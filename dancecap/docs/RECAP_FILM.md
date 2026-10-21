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
        [--sidecar <rec>.faces.json] [--start S --end S] [--side-by-side <sbs>.mp4] [--lossless] [--phone-shots off|swap]
    ... -m dancecap.film_faceswap verify --report <out>.report.json     # bit-exact / PSNR outside the faces + flicker ratio
    ... -m dancecap.film_faceswap sheet --report <out>.report.json --out sheet.jpg [--frames N ...] [--dets]
    ... -m dancecap.film_faceswap canon-points                          # face_points_canonical.json (below)
    ... -m dancecap.film_faceswap mock-sidecar --report <heuristic run>.report.json --out mock.faces.json   # tests the sidecar path

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
   swap changed them. Audio: an excerpt re-cuts the same instant as AAC 192k; a full film copies the track and keeps the video's
   start offset (the recorder's 0.033 s). Video: x264 CRF 14 slow (`--lossless`: qp 0).

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
                     "vis": 0..1, "by": "partner" | "self" | ""}, {"k": "follow", ...} ] }, ... ] }

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

`film_faceswap.validate_sidecar` checks format, landmark order, dancers, size, frame count and the `surfaces` entries (alpha 0..1, a
polygon of >= 3 points, one `z` per point). A mock sidecar made from a heuristic run (`mock-sidecar`) exercises the sidecar path until the
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
  camera_tour + a phone_in).
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
7. **Run it:** `python -m dancecap.film_recap_flow clips` under the Unity lock (v3.1: the phone shots and frustum surfaces need 326 s of phone video,
   192.5 s of it not in the capture: clips.json; it keeps the current clips), then load `work/larissa_kadu_recap/v3/direction/direction.json` (or copy the folder over film/ after
   backing it up; its audio/ mix is identical), look at the ways in and out with `hm_film_show`, then record both aspects as before (splat dancers on).
   The face sidecar must also carry the frustum surfaces (`surfaces`, v3.1, section "The face sidecar" above): a dancer whose face lies
   behind a phone's frustum video surface is not swapped (the picture there is the phone's video of the real people).

Limits: the azimuths, radii and heights of the plan are estimates from the physics COMs and the phone tracks (the viewer flies between its own live
poses, so continuity holds by construction, but a duration may be off the rule for the live geometry by a little); nothing of v3 has been rendered;
v3.1: 96.7 s of the film is camera motion (26.3 %; v3.0 35 %), and the phones' full-frame share (44.6 %) is still below the cut film's 52.4 % (above); "rotation" right after "bring her to the
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
