"""heat_beats: the beat grid and rhythm inputs of a heat recorded by ONE camera, with NO studio file (take john_lula).

carlos_aline_beats works from two studio mp3s and checks the grid on the heat's audio. Here the only sound is the phone's own
audio (the song is not known), so the heat's recording IS the evidence and the grid is fitted on it directly:

  1. beat_this (final0, the demo's checkpoint) on the video's own audio (mono 22.05 kHz, whole video).
  2. A straight pulse lattice t_k = t0 + k P (carlos_aline_beats.fit_pulses: the tracked interval folded into [60, 120) BPM,
     the beats that sit on one lattice found by walking outwards, a line fitted with a 3-MAD pass); drift checked per 20 s.
  3. Where the music is: the lattice contrast of the onset flux in the bass band (30-150 Hz: the kick) = how much stronger the
     flux is ON the pulses than at the three off phases (z over 6 s windows), per band; the music's level step edges (the bass
     envelope) cut the take into intro / music / outro. beat_this places beats on any periodic noise (applause), so a beat
     counts only where the evidence says music (segments.json "no_beat", confidence per beat in beats_merged.json).
  4. Zouk grid: 8 eighth slots per 4-pulse measure, types 1 3 3 2 1 3 2 3, eighths = pulse midpoints; whole measures
     inside the music span only. Bar phase = k mod 4 of beat_this's downbeats; when those vote 2-pulse bars (a tie) the
     kick / backbeat rotation test and the section structure (the music's edges sit on whole bars) decide, and the
     confidence is stated.
  5. Validation against the audio itself (the grid is NOT checked against a studio file: there is none): folded onset flux per
     band (where the onsets sit relative to the pulses), per-20 s drift, per-beat onset support, beat_this vs the lattice.

    python -m dancecap.heat_beats run --take john_lula                 # everything; writes work/<take>/song/
    python -m dancecap.heat_beats run --take john_lula --phase 2       # force a bar phase (pulse index mod 4)
    python -m dancecap.heat_beats follow --take john_lula --steps <timing.json | steps.json>   # step phase lock

A heat with SEVERAL songs (lucas_luiza_h, 2026-10-23: two songs at 90 and ~87 BPM, mixed by the DJ) has no single lattice: the straight-line fit
locks onto one song and calls the other "no beat". Fit each song inside its own span, then merge:

    python -m dancecap.heat_beats run --take T --span 0 121 --label song1      # -> song/seg_song1/ (the files below, for that span only)
    python -m dancecap.heat_beats run --take T --span 118 200 --label song2    # -> song/seg_song2/
    python -m dancecap.heat_beats merge --take T --labels song1,song2           # -> song/ (one governing grid, one timeline per song)

A span run silences the audio outside the span and keeps beat_this's beats inside it (the tracker runs once on the whole heat, cached).
merge concatenates the governing grids in time (a later song's measures that start before the earlier song's music ends replace the
earlier song's from there: the DJ's hand-over), timeline states = the labels (plus "intro" / "gap" / "outro" = no beat), tempo.json the
duration-weighted BPM with every song's own fit under "songs", beats_merged.json with a "song" column (dancecap.rhythm reads it).

Outputs (work/<take>/song/, git-ignored; times on the take's reference clock = the single video's own time; the demo's layout
with the studio file left out; NO audio is written except a clicks overlay of the heat's own sound):
  beats_heat.json           GOVERNING grid [[t, type]], whole measures inside the music span (bare list, measure = index // 8:
                            a drop-in --beats for timing / physics / export). The role of beats_studio.json, which needs a studio file
  beats_merged.json         every pulse and eighth of the audible lattice (also the weak intro / outro): t, type, measure, slot,
                            confidence, p_tracker, onset support, presence z, governing, pulse_position
  downbeats.json            bars (t, measure, governing, confidence)
  tempo.json                pulse period / BPM, fit, drift, bar-phase decision, beat placement, tempo curve (constant)
  segments.json             timeline intro / music / outro, no_beat intervals, the danced span
  presence.json             lattice contrast z per band over time, bass level
  validation.json           onset agreement per band and window, beat_this vs lattice, per-beat onset support
  song.json                 what is known of the song (nothing but the tempo), the files
  heat_beats_beat_this.json beat_this's own beats / downbeats on the heat audio
  beats_heat_raw.json       provenance + everything the above are cut from
  heat_clicks_overlay.wav   the heat's own sound + clicks on the governing grid (listening check, work/ only)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

from . import beats as B
from . import carlos_aline_beats as C
from . import song as S
from .take import PACKAGE_ROOT, Take

VERSION = 1
SR = B.ANALYSIS_SR
BANDS = C.BANDS  # bass_30_150, mid_1_4k, hi_4_9k
CONTRAST_WIN_S = 6.0
Z_LO, Z_HI = 1.5, 4.5  # music presence ramp on the best lattice-contrast z (0 at <= Z_LO, 1 at >= Z_HI)
MUSIC_Z = 3.0  # a stretch is "music" where the bass lattice contrast exceeds this
# A phone in a studio often hears little bass (lk03_mono20, 2026-10-22: bass z <= 1.9 over a clip that is music all through, the
# 1-4 kHz band 2-3): when the bass band shows no music at all, the band with the clearest lattice decides at this lower bar, if
# it holds for at least FALLBACK_SHARE of the clip and the pulse lattice itself is well supported (FALLBACK_MIN_SUPPORT beats)
FALLBACK_Z = 2.0
FALLBACK_SHARE = 0.5
FALLBACK_MIN_SUPPORT = 12
DROP_DB = 8.0  # the music's bass envelope (peak over 0.6 s) stays above (its median - DROP_DB) inside the music span
ENV_FRAMES = 7  # 0.7 s of 0.1 s frames: the peak envelope


# ------------------------------------------------------------------------------------------------ pure analysis
def flux_at(F: np.ndarray, dt: float, t: float, half_s: float = 0.02) -> float | None:
    """Max of the onset flux F (frame step dt, first frame at 0) within +-half_s of t; None outside the signal."""
    i0, i1 = int(math.floor((t - half_s) / dt)), int(math.floor((t + half_s) / dt)) + 1
    if i0 < 0 or i1 > len(F) or i1 <= i0:
        return None
    return float(F[i0:i1].max())


def lattice_contrast(F: np.ndarray, dt: float, t0: float, period: float, centres, win_s: float = CONTRAST_WIN_S,
                     half_s: float = 0.02, phases: tuple = (0.25, 0.5, 0.75)) -> np.ndarray:
    """z of (flux ON the pulses) - (flux at the off phases `phases` of the period) in windows of win_s around `centres`: the
    two-sample z of the window's means. ~0 for noise, 5-8 for a kick on the pulse; nan when a window holds < 3 pulses."""
    out = []
    for c in np.asarray(centres, float):
        ks = np.arange(math.ceil((c - win_s / 2 - t0) / period), math.floor((c + win_s / 2 - t0) / period) + 1)
        on, off = [], []
        for k in ks:
            tb = t0 + period * k
            v = flux_at(F, dt, tb, half_s)
            if v is None:
                continue
            on.append(v)
            off += [w for ph in phases if (w := flux_at(F, dt, tb + ph * period, half_s)) is not None]
        if len(on) < 3 or len(off) < 3:
            out.append(np.nan)
            continue
        on, off = np.asarray(on), np.asarray(off)
        out.append((on.mean() - off.mean()) / math.sqrt(on.var() / len(on) + off.var() / len(off) + 1e-12))
    return np.asarray(out)


def level_runs(t: np.ndarray, level_db: np.ndarray, floor_db: float, min_s: float = 4.0, gap_s: float = 1.0) -> list[list[float]]:
    """[t_from, t_to] runs where level_db >= floor_db, runs closer than gap_s merged, shorter than min_s dropped. Edges are
    the frame times of the first / last frame above the floor (frame step = median dt)."""
    t, lv = np.asarray(t, float), np.asarray(level_db, float)
    above = lv >= floor_db
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    runs, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            runs.append([float(t[i]), float(t[j] + dt)])
            i = j + 1
        else:
            i += 1
    merged: list[list[float]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_s]


def rolling_max(a: np.ndarray, n: int) -> np.ndarray:
    """Centred running maximum over n frames (edges clipped)."""
    a = np.asarray(a, float)
    h = n // 2
    return np.asarray([a[max(0, i - h):i + h + 1].max() for i in range(len(a))])


def snap_to_pulses(t: float, t0: float, period: float, bar: int = 4, phase: int = 0) -> tuple[float, int]:
    """(time, k) of the nearest pulse whose index satisfies k mod bar == phase."""
    k = int(round((t - t0) / period))
    best = min(range(k - bar, k + bar + 1), key=lambda j: (abs(j - (t - t0) / period) if (j - phase) % bar == 0 else 1e9))
    return t0 + period * best, best


def music_segments(span: tuple[float, float] | None, duration: float) -> list[dict]:
    """intro / music / outro timeline from the music span (None: no music at all)."""
    if span is None:
        return [{"from_s": 0.0, "to_s": duration, "state": "none"}]
    a, b = span
    out = []
    if a > 0:
        out.append({"from_s": 0.0, "to_s": a, "state": "intro"})
    out.append({"from_s": a, "to_s": min(b, duration), "state": "music"})
    if b < duration:
        out.append({"from_s": b, "to_s": duration, "state": "outro"})
    return out


def presence_ramp(z: np.ndarray, lo: float = Z_LO, hi: float = Z_HI) -> np.ndarray:
    """0..1 audibility from the best lattice-contrast z (nan = 0)."""
    return np.clip((np.nan_to_num(np.asarray(z, float), nan=0.0) - lo) / (hi - lo), 0.0, 1.0)


def bar_structure_phase(edges_k: list[int], bar: int = 4) -> dict:
    """Bar phase suggested by where the music's edges fall: sections start and end on whole bars, so the edge pulse indices
    share k mod bar. {phase, agreement (share of edges on it), n}."""
    if not edges_k:
        return {"phase": None, "agreement": 0.0, "n": 0}
    v = np.bincount(np.asarray(edges_k) % bar, minlength=bar)
    return {"phase": int(v.argmax()), "agreement": round(float(v.max() / v.sum()), 3), "n": int(v.sum())}


def fold_lag(F: np.ndarray, dt: float, pulses: np.ndarray, max_s: float = 0.08, step_s: float = 0.001) -> float | None:
    """Lag (s) of the peak of the onset flux folded on the pulses (mean over pulses of F at pulse + lag, lag in +-max_s):
    > 0 = the onsets come after the lattice. Parabolic sub-step refinement."""
    lags = np.arange(-max_s, max_s + step_s / 2, step_s)
    t = np.arange(len(F)) * dt
    ok = (pulses - max_s > 0) & (pulses + max_s < t[-1])
    if ok.sum() < 5:
        return None
    prof = np.mean([np.interp(p + lags, t, F) for p in pulses[ok]], axis=0)
    k = int(np.argmax(prof))
    if 0 < k < len(prof) - 1:
        a, b, c = prof[k - 1], prof[k], prof[k + 1]
        den = a - 2 * b + c
        return float(lags[k] + (0.5 * (a - c) / den * step_s if abs(den) > 1e-12 else 0.0))
    return float(lags[k])


# ------------------------------------------------------------------------------------------------ the analysis
def analyse(x: np.ndarray, sr: int, tr: dict, duration: float, phase: int | None = None,
            music: tuple[float, float] | None = None) -> dict:
    """The whole grid analysis on the heat audio x and beat_this's outputs `tr` (beats_refined, downbeats_refined, beat_prob).
    `music` = the music span given from outside (a quiet song in a loud room shows no lattice contrast although its pulses are
    tracked: lucas_luiza_h's first song, z <= 1 in every band, its tempo confirmed by the studio file's clock). Pure: no files, no model."""
    hop = int(round(S.HOP_S * sr))
    dt = hop / sr
    model = C.fit_pulses(tr["beats_refined"])
    P, t0 = model["period"], model["t0"]
    flux = {"broadband": B.band_onset(x, sr, 30.0, 10000.0, hop=hop)[0]}
    for n, (lo, hi) in BANDS.items():
        flux[n] = B.band_onset(x, sr, lo, hi, hop=hop)[0]
    centres = np.arange(0.0, duration + 1e-9, 1.0)
    z = {n: lattice_contrast(F, dt, t0, P, centres) for n, F in flux.items()}
    z_best = np.nanmax(np.stack([z[n] for n in BANDS]), axis=0) if len(centres) else np.zeros(0)
    # the music's bass envelope (energy of 30-150 Hz in 0.1 s frames, dB): the level step edges
    import librosa
    Sx = np.abs(librosa.stft(x.astype(np.float32), n_fft=2048, hop_length=int(0.1 * sr))) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bass_db = 10 * np.log10(Sx[(f >= 30) & (f < 150)].sum(0) + 1e-12)
    t_bass = np.arange(len(bass_db)) * 0.1
    # music = bass lattice contrast above MUSIC_Z ... then the span edges are refined on the bass level step
    zb = z["bass_30_150"]
    music_c = centres[np.nan_to_num(zb, nan=0.0) >= MUSIC_Z]
    music_band = "bass_30_150"
    if not len(music_c) and len(centres) and int(model.get("n_support") or 0) >= FALLBACK_MIN_SUPPORT:
        best = max(BANDS, key=lambda n: float(np.nanmedian(z[n])) if np.isfinite(z[n]).any() else -9.0)
        zf = np.nan_to_num(z[best], nan=0.0)
        if (zf >= FALLBACK_Z).mean() >= FALLBACK_SHARE:
            music_c, music_band = centres[zf >= FALLBACK_Z], best
    span = None
    level_floor = None
    if len(music_c):
        core = (t_bass >= music_c[0]) & (t_bass <= music_c[-1])
        env = rolling_max(bass_db, ENV_FRAMES)  # the beat-resolved dips of the kick are not a gap in the music
        level_floor = float(np.median(env[core]) - DROP_DB)
        runs = level_runs(t_bass, env, level_floor, min_s=8.0, gap_s=1.5)
        if runs:
            # the run holding most of the contrast-music seconds is the music span; its edges are refined on the raw bass level
            best = max(runs, key=lambda r: ((music_c >= r[0] - 1) & (music_c <= r[1] + 1)).sum())
            raw_ok = np.where(bass_db >= level_floor)[0]
            near = raw_ok[(t_bass[raw_ok] >= best[0] - 0.5) & (t_bass[raw_ok] <= best[1] + 0.5)]
            span = (float(t_bass[near[0]]), float(t_bass[near[-1]])) if len(near) else (best[0], best[1])
    if music is not None:
        span = (max(0.0, float(music[0])), min(duration, float(music[1])))
        music_band = "given"
    k_all_lo = int(math.ceil((0.0 - t0) / P))
    k_all_hi = int(math.floor((duration - t0) / P))
    edges_k = []
    if span is not None:
        # a pulse owns the time up to the next one: the span runs from the first loud pulse to the END of the last loud pulse
        k_a, k_b = int(round((span[0] - t0) / P)), int(round((span[1] - t0) / P)) + 1
        span = (float(t0 + P * k_a), float(t0 + P * k_b))
        edges_k = [k_a, k_b]
    votes = C.downbeat_votes(tr["downbeats_refined"], model)
    structure = bar_structure_phase(edges_k)
    # kick / backbeat rotation test over the music span
    scores = {}
    if span is not None:
        lo_o, lo_t = B.band_onset(x, sr)
        sn_o, sn_t = B.band_onset(x, sr, 1000.0, 4000.0)
        k_lo = int(round((span[0] - t0) / P))
        k_hi = int(round((span[1] - t0) / P))
        for p in range(4):
            g = C.zouk_rows(model, k_lo, k_hi, p)
            if len(g) >= 16:
                rot = B.rotation_scores(B.folded_profile(g[:, :4], lo_t, lo_o), B.folded_profile(g[:, :4], sn_t, sn_o))
                scores[p] = rot["scores"][0]
    ranked = sorted(scores, key=lambda p: -scores[p])
    decision = {"by": "beat_this downbeats", "votes": votes, "structure": structure,
                "kick_backbeat": {"scores": {int(p): round(float(v), 3) for p, v in scores.items()},
                                  "best": int(ranked[0]) if ranked else None,
                                  "margin": round(float(scores[ranked[0]] - scores[ranked[1]]), 3) if len(ranked) > 1 else None}}
    if phase is not None:
        chosen, conf, by = int(phase) % 4, "given", f"override --phase ({int(phase) % 4})"
    elif not votes["ambiguous"]:
        chosen, conf, by = votes["phase"], "high", "beat_this downbeats"
    else:
        # downbeats vote two-pulse bars: the section structure (the music starts and ends on a whole bar) and the kick /
        # backbeat test arbitrate between the two phases the votes hold
        cands = {votes["phase"], (votes["phase"] + 2) % 4}
        s_ph = structure["phase"]
        kb = decision["kick_backbeat"]["best"]
        if s_ph in cands and structure["n"] >= 2 and structure["agreement"] == 1.0:
            chosen, by = s_ph, "section structure (the music's edges sit on whole bars) within the downbeat tie"
            conf = "medium" if kb in (None, s_ph) else "medium: the kick / backbeat test prefers another pulse"
        elif kb in cands:
            chosen, by, conf = kb, "kick on 1 + snare backbeat within the downbeat tie", "low"
        else:
            chosen, by, conf = votes["phase"], "beat_this downbeats (a tie)", "low"
    decision.update(by=by, confidence=conf, phase=chosen, ambiguous=votes["ambiguous"])
    # per-pulse numbers
    ks = np.arange(k_all_lo, k_all_hi + 1)
    tk = C.pulse_times(model, ks)
    p_track = C.tracker_support(tr["beat_prob"], tk)
    tracked = np.array([np.abs(tr["beats_refined"] - t).min() < C.BEAT_TOL * P for t in tk])
    z_at = np.interp(tk, centres, np.nan_to_num(z_best, nan=0.0))
    aud = presence_ramp(z_at)
    if music is not None and span is not None:
        # a GIVEN music span was verified another way (the studio file's clock and downbeats): the heat audio's lattice contrast
        # is no evidence against the pulses inside it (lucas_luiza_h song 1: z <= 1, so every beat had ~0 confidence)
        aud = np.where((tk >= span[0]) & (tk <= span[1]), 1.0, aud)
    conf_pulse = np.round(np.clip(p_track, 0, 1) * aud * np.where(tracked, 1.0, 0.7), 4)
    support = {n: np.array([_support(flux[n], dt, t) for t in tk]) for n in BANDS}
    rows_all = C.zouk_rows(model, k_all_lo, k_all_hi, chosen)  # every whole measure of the audible lattice
    gov = C.whole_measures(rows_all, span[0], span[1]) if span is not None else np.zeros((0, 5))
    return {"model": model, "dt": dt, "flux": flux, "centres": centres, "z": z, "z_best": z_best, "t_bass": t_bass, "bass_db": bass_db,
            "level_floor_db": level_floor, "span": span, "edges_k": edges_k, "decision": decision, "k": ks, "t_pulse": tk,
            "p_track": p_track, "tracked": tracked, "aud": aud, "conf": conf_pulse, "support": support, "rows_all": rows_all,
            "gov": gov, "phase": chosen, "duration": duration, "music_band": music_band if span is not None else None}


def _support(F: np.ndarray, dt: float, t: float, half_s: float = 0.02, neigh_s: float = 1.0) -> float:
    """1 when the flux at t is the strongest around (+-neigh_s), 0 at the median level or below."""
    i0, i1 = int(max(0, (t - half_s) / dt)), int((t + half_s) / dt) + 1
    j0, j1 = int(max(0, (t - neigh_s) / dt)), int((t + neigh_s) / dt) + 1
    if i1 <= i0 or j1 - j0 < 20 or i1 > len(F):
        return 0.0
    v = float(F[i0:i1].max())
    return float(np.clip((np.mean(F[j0:j1] < v) - 0.5) * 2.0, 0.0, 1.0))


def merged_rows(a: dict) -> dict:
    """The beats_merged table: every pulse and eighth of the audible lattice (whole measures of the phase), the weak intro and
    outro included, with the confidence parts."""
    model = a["model"]
    t_by_k = dict(zip(a["k"].tolist(), range(len(a["k"]))))
    gov_t = set(np.round(a["gov"][:, 0], 6).tolist()) if len(a["gov"]) else set()
    span = a["span"]
    rows = []
    for r in a["rows_all"]:
        t, typ, m, slot, kpos = r
        k0 = int(math.floor(kpos))
        i = t_by_k.get(k0)
        if i is None:
            continue
        i1 = t_by_k.get(k0 + 1, i)
        j = i if slot % 2 == 0 else i1
        conf = float(a["conf"][i]) if slot % 2 == 0 else float(min(a["conf"][i], a["conf"][i1]))
        rows.append([round(float(t), 6), int(typ), int(m), int(slot), round(float(t), 6) in gov_t, round(conf, 4),
                     round(float(a["p_track"][j]), 4), round(float(a["support"]["bass_30_150"][j]), 3),
                     round(float(np.interp(t, a["centres"], np.nan_to_num(a["z_best"], nan=0.0))), 2),
                     bool(span is not None and span[0] <= t <= span[1]), float(kpos)])
    return {"columns": ["t", "type", "measure_in_take", "slot", "governing", "confidence", "p_tracker", "bass_support", "presence_z",
                        "in_music", "pulse_position"], "rows": rows,
            "note": "measure_in_take counts whole measures of the phase over the whole audible lattice from the first one; "
                    "beats_heat.json renumbers the governing ones from 0"}


# ------------------------------------------------------------------------------------------------ validation
def validate(x: np.ndarray, sr: int, a: dict, tr: dict) -> dict:
    """Onset agreement per band against the lattice, per-20 s windows; beat_this beats against the lattice."""
    P = a["model"]["period"]
    pulses = a["t_pulse"]
    span = a["span"] or (0.0, a["duration"])
    inside = pulses[(pulses >= span[0] + 1.0) & (pulses <= span[1] - 1.0)]
    out = {"method": "onset flux of each band folded on the lattice pulses (mean over pulses of the flux at pulse + lag, +-80 ms); "
                     "lag > 0 = the audio's onsets come after the lattice. The lattice sits on beat_this's beat times, which mark "
                     "the kick's energy peak (see tempo.json beat_placement).",
           "span_s": [round(span[0], 3), round(span[1], 3)], "bands": {}}
    for n, F in {"broadband": a["flux"]["broadband"], **{k: a["flux"][k] for k in BANDS}}.items():
        wins = []
        for w0 in np.arange(span[0], span[1] - 20.0 + 1e-9, 10.0):
            sel = pulses[(pulses >= w0) & (pulses < w0 + 20.0)]
            lag = fold_lag(F, a["dt"], sel)
            if lag is not None:
                wins.append({"t": round(float(w0 + 10.0), 1), "lag_ms": round(1000 * lag, 2), "n": int(len(sel))})
        lag_all = fold_lag(F, a["dt"], inside)
        d = C.drift_fit(np.array([w["t"] for w in wins]), np.array([w["lag_ms"] for w in wins])) if wins else {}
        out["bands"][n] = {"lag_ms": None if lag_all is None else round(1000 * lag_all, 2), "windows": wins, "summary": d}
    # beat_this's own beats vs the lattice: signed offsets within a third of a period
    b = tr["beats_refined"]
    k = np.round((b - a["model"]["t0"]) / P)
    off = 1000 * (b - (a["model"]["t0"] + P * k))
    ok = np.abs(off) < 1000 * P / 3
    inm = (b >= span[0]) & (b <= span[1])
    out["beat_this_vs_lattice"] = {"n_beats": int(len(b)), "n_within_third_period": int(ok.sum()), "median_ms": round(float(np.median(off[ok])), 2),
                                   "rms_ms": round(float(np.sqrt(np.mean(off[ok] ** 2))), 2),
                                   "rms_ms_in_music": round(float(np.sqrt(np.mean(off[ok & inm] ** 2))), 2) if (ok & inm).any() else None,
                                   "tempo_bpm_median_interval": round(60.0 / float(np.median(np.diff(b))), 3)}
    sup = a["support"]["bass_30_150"]
    inm_p = (pulses >= span[0]) & (pulses <= span[1])
    out["per_beat_onset_support"] = {
        "bass_in_music": round(float(np.mean(sup[inm_p] > 0.5)), 3) if inm_p.any() else None,
        "bass_outside_music": round(float(np.mean(sup[~inm_p] > 0.5)), 3) if (~inm_p).any() else None,
        "note": "share of pulses whose bass flux is the strongest of +-1 s (above the 75th percentile of its neighbourhood, support > 0.5)"}
    return out


def heat_tempo_curve(a: dict, win_s: float = 20.0, step_s: float = 10.0) -> list[dict]:
    """Tempo from the lattice-supporting beat_this beats in windows (constant tempo: the numbers agree)."""
    m = a["model"]
    out = []
    for w0 in np.arange(0.0, a["duration"] - win_s + 1e-9, step_s):
        sel = (m["t"] >= w0) & (m["t"] < w0 + win_s)
        if sel.sum() >= 5:
            pk = np.polyfit(m["k"][sel], m["t"][sel], 1)[0]
            out.append({"t": round(float(w0 + win_s / 2), 1), "bpm": round(60.0 / float(pk), 3), "n": int(sel.sum())})
    return out


# ------------------------------------------------------------------------------------------------ io
def work_dir(name: str) -> Path:
    return PACKAGE_ROOT / "work" / name


def song_dir(name: str) -> Path:
    return work_dir(name) / "song"


def _r(a, nd=4):
    return np.round(np.asarray(a, float), nd).tolist()


def device_ctx(device: str | None):
    import torch

    from . import gpu
    if device == "cpu" or not torch.cuda.is_available():
        return "cpu", contextlib.nullcontext()
    try:
        ctx = gpu.exclusive("heat-beats", timeout_s=60.0)
        ctx.__enter__()
    except TimeoutError:
        return "cpu", contextlib.nullcontext()
    return "cuda", B._Entered(ctx)


def track(x: np.ndarray, sr: int, cache: Path, device: str | None, fresh: bool = False) -> tuple[dict, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    dev = "cache"
    if fresh or not cache.exists():
        dev, lock = device_ctx(device)
        with lock:
            r = B.BeatTracker("final0", dev)(x, sr)
        np.savez_compressed(cache, **{n: np.asarray(v) for n, v in r.items()})
        print(f"[heat_beats] beat_this: {len(r['beats'])} beats, {len(r['downbeats'])} downbeats ({dev})", flush=True)
    z = np.load(cache)
    return {n: z[n] for n in z.files}, dev


def write_outputs(take: Take, a: dict, val: dict, tr: dict, dev: str, extra: dict, sd: Path | None = None) -> dict:
    sd = sd or song_dir(take.name)
    sd.mkdir(parents=True, exist_ok=True)
    model, P = a["model"], a["model"]["period"]
    w = lambda name, obj, indent=1: S.write_json(sd / name, obj, indent=indent)  # noqa: E731
    gov = a["gov"]
    w("beats_heat.json", [[round(float(t), 6), int(k)] for t, k in gov[:, :2]], indent=None)
    merged = merged_rows(a)
    w("beats_merged.json", merged, indent=None)
    w("downbeats.json", [{"t": r[0], "measure_in_take": r[2], "governing": r[4], "confidence": r[5]} for r in merged["rows"] if r[3] == 0],
      indent=None)
    span = a["span"]
    seg = music_segments(span, a["duration"])
    no_beat = [[s["from_s"], s["to_s"]] for s in seg if s["state"] != "music"]
    tempo = {"version": VERSION, "pulse_s": round(P, 6), "bpm": round(60.0 / P, 4), "measure_s": round(4 * P, 5),
             "t0_s": round(model["t0"], 6), "fit": {"rms_ms": model["rms_ms"], "n_support": model["n_support"], "n_tracked": model["n_fit"],
                                                   "level": model["level"], "max_window_drift_ms": model["max_window_drift_ms"],
                                                   "constant_tempo": model["constant"], "windows": model["windows"]},
             "bar_phase": a["decision"], "tempo_curve": heat_tempo_curve(a),
             "beat_placement": extra["placement"],
             "note": "the heat plays at one constant tempo; the lattice is a straight line through beat_this's beats on the heat audio"}
    w("tempo.json", tempo)
    segments = {"version": VERSION, "timeline": seg, "no_beat": no_beat,
                "no_beat_note": ("intro: the music's quiet, bass-less lead-in before its first full bar; outro: the bass is gone and the level falls "
                                 "away: the lattice continues in the mid / high bands (applause / filtered fade) but is not evidence of the "
                                 "music's beat. No beat is written in beats_heat.json there; beats_merged.json keeps them with their confidence."),
                "danced_span": [float(v) for v in take.window], "music_span": None if span is None else [round(span[0], 3), round(span[1], 3)],
                "music_band": a.get("music_band"), "speech": extra.get("speech", [])}
    w("segments.json", segments)
    w("presence.json", {"t": _r(a["centres"], 2), "win_s": CONTRAST_WIN_S, "z": {n: [None if not np.isfinite(v) else round(float(v), 2) for v in zz]
                                                                                for n, zz in a["z"].items()},
                        "bass_db": {"t": _r(a["t_bass"][::5], 2), "db": _r(a["bass_db"][::5], 1), "floor_db": a["level_floor_db"]},
                        "ramp": {"lo": Z_LO, "hi": Z_HI}}, indent=None)
    w("validation.json", val)
    w("heat_beats_beat_this.json", {"source": take.sources[0].path.name, "beats_s": _r(tr["beats_refined"]), "downbeats_s": _r(tr["downbeats_refined"]),
                                    "note": "beat_this on the heat audio, video seconds; beats outside segments.json music are not evidence"}, indent=None)
    pulses = [[round(float(t), 6), int(k), int((k - a["phase"]) % 4), round(float(c), 3), round(float(p), 3), bool(tk)]
              for t, k, c, p, tk in zip(a["t_pulse"], a["k"], a["conf"], a["p_track"], a["tracked"])]
    w("beats_heat_raw.json", {"version": VERSION, "take": take.name, "module": "dancecap.heat_beats",
                              "provenance": {"source": take.sources[0].path.name, "beat_this": {"checkpoint": "final0", "device": dev, "fps": B.BEAT_THIS_FPS,
                                                                                               "input": f"mono {SR} Hz, the video's own audio"},
                                             "grid": "fit_pulses (straight lattice), downbeat_votes + section structure + kick/backbeat for the bar "
                                                     "phase, zouk_rows (eighths = pulse midpoints), whole measures inside the music span",
                                             "studio_file": None},
                              "pulse_fit": {k: v for k, v in model.items() if k not in ("k", "t", "resid_ms")},
                              "bar_phase": a["decision"], "music_span": segments["music_span"],
                              "pulse_columns": ["t", "k", "bar_pos", "confidence", "p_tracker", "tracked"], "pulses": pulses}, indent=None)
    w("song.json", {"version": VERSION, "take": take.name, "reference": "the single video's own time (identity clock)",
                    "song": {"title": "unknown", "artist": "unknown",
                             "how": "no studio file and no on-screen text: the song is not identified; only the tempo is known",
                             "tempo_bpm": round(60.0 / P, 3), "pulse_s": round(P, 6), "beats_per_bar": 4},
                    "timeline": seg, "files": sorted(p.name for p in sd.glob("*.json")),
                    "grid_file": "beats_heat.json (the role of beats_studio.json: there is no studio file)",
                    "copyright": "The heat's own audio is a third-party recording of a competition: keep everything under work/ (git-ignored); never commit or publish."})
    w("beats_heat.provenance.json", {"what": "beat grid of the heat fitted on the heat's own audio (beat_this + a straight pulse lattice): there is no studio file",
                                     "measure numbers": "beats_heat.json is time-ordered whole measures, index // 8 is the measure",
                                     "module": "dancecap.heat_beats", "reference_clock": "the single video's own time"})
    return {"governing_measures": int(len(gov) // 8), "music_span": segments["music_span"], "files": sorted(p.name for p in sd.glob("*.json"))}


def clicks_overlay(take: Take, x: np.ndarray, gov: np.ndarray, sd: Path | None = None) -> None:
    clicks = B.click_track(len(x), SR, gov[:, :2], 0.0)
    mix = 0.7 * x / max(1e-9, float(np.abs(x).max())) + clicks
    S.write_private_wav((sd or song_dir(take.name)) / "heat_clicks_overlay.wav", mix / max(1.0, float(np.abs(mix).max()) / 0.98), SR)


def review_plot(take: Take, a: dict, val: dict, tr: dict, x: np.ndarray, label: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(17, 11), sharex=True)
    hop = 256
    on, ot = B.onset_strength(x, SR, hop)
    ax[0].plot(ot, on, lw=0.5, color="0.3")
    gov = a["gov"]
    for t, k in gov[:, :2]:
        ax[0].axvline(t, color={1: "tab:red", 2: "tab:orange", 3: "tab:blue"}[int(k)], lw=0.4, alpha=0.6)
    ax[0].set_ylabel("onset strength + governing grid (1 red, 2 orange, 3 blue)")
    for n, z in a["z"].items():
        if n != "broadband":
            ax[1].plot(a["centres"], z, label=n)
    ax[1].axhline(MUSIC_Z, color="k", ls=":", lw=0.8)
    ax[1].set_ylabel("lattice contrast z (6 s)")
    ax[1].legend(loc="upper right")
    ax[2].plot(a["t_bass"], a["bass_db"], lw=0.6)
    if a["level_floor_db"] is not None:
        ax[2].axhline(a["level_floor_db"], color="r", ls=":", lw=0.8)
    ax[2].set_ylabel("bass 30-150 Hz (dB)")
    ax[3].plot(a["t_pulse"], a["conf"], ".", ms=3)
    ax[3].set_ylabel("beat confidence")
    for axx in ax:
        for s in (take.window[0], take.window[1]):
            axx.axvline(s, color="g", lw=0.6)
        if a["span"] is not None:
            axx.axvspan(a["span"][0], a["span"][1], color="tab:green", alpha=0.06)
    ax[3].set_xlabel("video time (s)")
    fig.suptitle(f"{take.name}: {60 / a['model']['period']:.2f} BPM, phase {a['phase']}, music "
                 f"{'-' if a['span'] is None else '%.1f-%.1f s' % a['span']}", fontsize=11)
    out = take.work / "review" / f"{take.name}_beats{'_' + label if label else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=75)
    plt.close(fig)
    return out


def span_input(x: np.ndarray, tr: dict, span: tuple[float, float], sr: int = SR) -> tuple[np.ndarray, dict]:
    """The heat audio silenced outside [t0, t1] and beat_this's beats / downbeats kept inside it: what analyse() sees for ONE song
    of a multi-song heat (the frame-wise beat probability is kept: it is only read at the lattice's pulse times)."""
    t0, t1 = float(span[0]), float(span[1])
    y = np.zeros_like(x)
    a, b = max(0, int(round(t0 * sr))), min(len(x), int(round(t1 * sr)))
    y[a:b] = x[a:b]
    out = dict(tr)
    for n in ("beats_refined", "downbeats_refined", "beats", "downbeats"):
        if n in out:
            v = np.asarray(out[n], float)
            out[n] = v[(v >= t0) & (v <= t1)]
    return y, out


def run(take_name: str, device: str | None = None, phase: int | None = None, fresh: bool = False, plot: bool = True,
        span: tuple[float, float] | None = None, label: str | None = None, music: tuple[float, float] | None = None) -> dict:
    warnings.filterwarnings("ignore")
    t_start = time.time()
    take = Take.load(take_name)
    src = take.sources[0]
    x, _ = B.decode_audio(src.path, SR)
    duration = float(len(x) / SR)
    cache = work_dir(take.name) / "song" / "_cache" / "beat_this_heat.npz"
    tr, dev = track(x, SR, cache, device, fresh)
    sd = None
    if span is not None:
        label = label or f"{span[0]:g}_{span[1]:g}"
        sd = song_dir(take.name) / f"seg_{label}"
        x, tr = span_input(x, tr, span)
    a = analyse(x, SR, tr, duration, phase, music)
    val = validate(x, SR, a, tr)
    extra = {"placement": S.beat_placement(x, SR, a["t_pulse"][(a["t_pulse"] > 0.2) & (a["t_pulse"] < duration - 0.2)])}
    if span is not None:
        extra["placement"] = S.beat_placement(x, SR, a["t_pulse"][(a["t_pulse"] > span[0] + 0.2) & (a["t_pulse"] < span[1] - 0.2)])
    res = write_outputs(take, a, val, tr, dev, extra, sd)
    if span is not None:
        S.write_json(sd / "span.json", {"span": [float(span[0]), float(span[1])], "label": label,
                                        "music_given": None if music is None else [float(music[0]), float(music[1])]})
        res["span"], res["label"] = [float(span[0]), float(span[1])], label
    if len(a["gov"]):
        clicks_overlay(take, x, a["gov"], sd)
    if plot:
        res["plot"] = review_plot(take, a, val, tr, x, label if span is not None else None).as_posix()
    res["seconds"] = round(time.time() - t_start, 1)
    res.update(bpm=round(60.0 / a["model"]["period"], 4), phase=a["phase"], phase_confidence=a["decision"]["confidence"])
    return res


def follow(take_name: str, steps: Path) -> dict:
    """Step phase lock R (mean resultant length of the step phase against the pulse / eighth) over the whole music span and in
    10 s sliding windows: song/follow.json. Steps = timing.json footsteps or a list / {"steps": [...]}."""
    take = Take.load(take_name)
    raw = json.loads((song_dir(take.name) / "beats_heat_raw.json").read_text(encoding="utf-8"))
    pulses = np.asarray([r[0] for r in raw["pulses"]], float)
    st = C.load_steps(steps)
    lo, hi = (raw["music_span"] or [pulses[0], pulses[-1]])
    out = {"steps_file": Path(steps).as_posix(), "n_steps": int(len(st)), "music": C.phase_lock(st, pulses, lo, hi),
           "windows": [{"t0": float(t0), "t1": float(t0 + 10.0), **C.phase_lock(st, pulses[(pulses >= t0 - 1) & (pulses <= t0 + 11)], t0, t0 + 10.0)}
                       for t0 in np.arange(lo, hi - 10.0 + 1e-9, 2.0)]}
    S.write_json(song_dir(take.name) / "follow.json", out)
    return out


def merge(take_name: str, labels: list[str]) -> dict:
    """One governing grid, timeline and tempo for a multi-song heat from its span runs (song/seg_<label>/, in time order)."""
    take = Take.load(take_name)
    sd = song_dir(take.name)
    segs = []
    for lab in labels:
        d = sd / f"seg_{lab}"
        seg = json.loads((d / "segments.json").read_text(encoding="utf-8"))
        if not seg.get("music_span"):
            raise SystemExit(f"[heat_beats] seg_{lab}: no music found in its span")
        segs.append({"label": lab, "dir": d, "music": [float(v) for v in seg["music_span"]],
                     "gov": json.loads((d / "beats_heat.json").read_text(encoding="utf-8")),
                     "merged": json.loads((d / "beats_merged.json").read_text(encoding="utf-8")),
                     "tempo": json.loads((d / "tempo.json").read_text(encoding="utf-8")),
                     "downbeats": json.loads((d / "downbeats.json").read_text(encoding="utf-8"))})
    segs.sort(key=lambda g: g["music"][0])
    duration = max(float(json.loads((g["dir"] / "segments.json").read_text(encoding="utf-8"))["timeline"][-1]["to_s"]) for g in segs)
    # governing measures in time order; where a later song's music starts inside an earlier one's, the later song governs from its first
    # whole measure (8 eighths per measure: whole measures are kept whole)
    gov_rows, timeline, merged_rows_, downbeats = [], [], [], []
    cut_end = [g["music"][1] for g in segs]
    for i in range(len(segs) - 1):
        nxt = segs[i + 1]["gov"]
        if nxt and nxt[0][0] < cut_end[i]:
            cut_end[i] = float(nxt[0][0])
    m_off = 0
    for g, end in zip(segs, cut_end):
        rows = g["gov"]
        meas = [rows[j:j + 8] for j in range(0, len(rows) - len(rows) % 8, 8)]
        keep = [m for m in meas if m[-1][0] < end]
        g["governing"] = [keep[0][0][0], keep[-1][-1][0]] if keep else None
        gov_rows += [r for m in keep for r in m]
        cols = g["merged"]["columns"]
        ci = {c: i for i, c in enumerate(cols)}
        kept_t = {round(r[0], 6) for m in keep for r in m}
        for r in g["merged"]["rows"]:
            r = list(r)
            r[ci["governing"]] = round(r[ci["t"]], 6) in kept_t
            r[ci["measure_in_take"]] = int(r[ci["measure_in_take"]]) + m_off
            merged_rows_.append(r + [g["label"]])
        m_off = max((int(r[ci["measure_in_take"]]) for r in merged_rows_), default=-1) + 1
        downbeats += [dict(d, song=g["label"]) for d in g["downbeats"]]
    merged_rows_.sort(key=lambda r: r[0])
    # timeline: intro / song / gap / song ... / outro (song = its music span, cut at the next song's governing start)
    t = 0.0
    for g, end in zip(segs, cut_end):
        a, b = g["music"][0], min(g["music"][1], end) if end > g["music"][0] else g["music"][1]
        a = max(a, t)
        if a > t + 1e-6:
            timeline.append({"from_s": round(t, 3), "to_s": round(a, 3), "state": "intro" if not timeline else "gap"})
        timeline.append({"from_s": round(a, 3), "to_s": round(b, 3), "state": g["label"], "songs": [g["label"]]})
        t = b
    if t < duration:
        timeline.append({"from_s": round(t, 3), "to_s": round(duration, 3), "state": "outro"})
    no_beat = [[x["from_s"], x["to_s"]] for x in timeline if x["state"] in ("intro", "gap", "outro")]
    w = lambda name, obj, indent=1: S.write_json(sd / name, obj, indent=indent)  # noqa: E731
    w("beats_heat.json", sorted(gov_rows, key=lambda r: r[0]), indent=None)
    cols = segs[0]["merged"]["columns"] + ["song"]
    w("beats_merged.json", {"columns": cols, "rows": merged_rows_,
                            "note": "the songs' span runs concatenated (heat_beats merge); governing = the merged grid's measures; "
                                    "measure_in_take continues from song to song"}, indent=None)
    w("downbeats.json", sorted(downbeats, key=lambda d: d["t"]), indent=None)
    spans = [x for x in timeline if x["state"] not in ("intro", "gap", "outro")]
    total = sum(x["to_s"] - x["from_s"] for x in spans) or 1.0
    bpm = sum(g["tempo"]["bpm"] * (x["to_s"] - x["from_s"]) for g, x in zip(segs, spans)) / total
    first = segs[0]["tempo"]
    w("tempo.json", {"version": VERSION, "bpm": round(bpm, 4), "pulse_s": round(60.0 / bpm, 6), "measure_s": round(240.0 / bpm, 5),
                     "bar_phase": first.get("bar_phase"), "note": "several songs: bpm is the duration-weighted mean; every song's own fit is under songs",
                     "songs": {g["label"]: dict(g["tempo"], governing=g["governing"], music_span=g["music"]) for g in segs}})
    w("segments.json", {"version": VERSION, "timeline": timeline, "no_beat": no_beat,
                        "governing": {g["label"]: g["governing"] for g in segs},
                        "danced_span": [float(v) for v in take.window],
                        "music_span": [spans[0]["from_s"], spans[-1]["to_s"]] if spans else None,
                        "songs": [g["label"] for g in segs],
                        "no_beat_note": "intro / gap / outro: no song's music (heat_beats merge of the span runs)"})
    return {"songs": {g["label"]: {"bpm": g["tempo"]["bpm"], "music": g["music"], "governing": g["governing"]} for g in segs},
            "timeline": timeline, "governing_measures": len(gov_rows) // 8, "bpm_weighted": round(bpm, 3)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="dancecap.heat_beats", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="beat_this on the heat audio -> pulse lattice, segments, grid, validation -> work/<take>/song/")
    r.add_argument("--take", required=True)
    r.add_argument("--device", choices=["cpu", "cuda"], default=None)
    r.add_argument("--phase", type=int, default=None, help="force the bar phase (pulse index mod 4)")
    r.add_argument("--fresh", action="store_true", help="re-run beat_this")
    r.add_argument("--no-plot", action="store_true")
    r.add_argument("--span", type=float, nargs=2, metavar=("T0", "T1"), help="fit ONE song inside [T0, T1] (seconds) -> song/seg_<label>/")
    r.add_argument("--label", help="the span's name (the song's state in the merged timeline)")
    r.add_argument("--music", type=float, nargs=2, metavar=("T0", "T1"),
                   help="the music span, given (when the bass lattice contrast cannot see a quiet song; check its tempo another way)")
    m = sub.add_parser("merge", help="merge span runs (song/seg_<label>/) into one grid / timeline / tempo in song/")
    m.add_argument("--take", required=True)
    m.add_argument("--labels", required=True, help="comma list of span labels")
    f = sub.add_parser("follow", help="step phase lock against the pulse lattice")
    f.add_argument("--take", required=True)
    f.add_argument("--steps", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        print(json.dumps(run(args.take, args.device, args.phase, args.fresh, not args.no_plot,
                             tuple(args.span) if args.span else None, args.label, tuple(args.music) if args.music else None),
                         indent=1, default=S._np))
    elif args.cmd == "merge":
        print(json.dumps(merge(args.take, [x for x in args.labels.split(",") if x]), indent=1, default=S._np))
    else:
        print(json.dumps(follow(args.take, args.steps), indent=1, default=S._np)[:3000])


if __name__ == "__main__":
    main()
