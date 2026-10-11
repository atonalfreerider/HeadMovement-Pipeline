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
bed (the quietest speech-free 6 s of the class, looped with a raised-cosine self cross-fade, -3 dB), junctions cross-faded over 0.15 s; fades follow the picture's
(0.12 s in, 0.9 s out). Checked by `validate`: a locked segment's film-minus-dance offset equals the inserts before it (so speech and picture cannot drift), a
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
- The 16:9 aspect uses the same shots; the orbit is not tightened for it (the couple is small in the frame, as in the demo film).

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

The owner's words: "the kadu larissa lesson needs to have more physics, more arrows, more POV from phone video, only use regenerated elevenlabs lines, output to
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
  apart (more than 0.95 m) there is nothing to show and it is skipped. While the teachers MOVE a physics view shows at least every `phys_gap_s` (24 s) of moving
  film time (`physics_cadence`); stretches where they stand and talk do not count. A physics view holds about 5 s (`trim_physics`), the rest of its sentence goes to
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
