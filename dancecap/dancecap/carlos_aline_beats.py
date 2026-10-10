"""carlos_aline: beat grid and rhythm analysis of a two-song heat (a DJ mix) on the reference clock.

The Jack & Jill heat of Carlos and Aline runs over TWO studio songs (takes/carlos_aline.toml [dance] song_file /
song_file_2): the DJ plays each one SLOWER than the studio file with the pitch kept (Kojo Funds x0.9615, Guetta x0.9425
of the studio tempo; carlos_aline_sync / sync.json call the factors 1.040 and 1.061, the duration stretch), starts
Kojo Funds at reference 13.31 s, mixes Guetta in under its end (113.47-122.25 s) and stops at about 194.3 s. This
module is dancecap.song / dancecap.beats for that heat: beat_this on the studio mp3s, the beat level made explicit, the
grid mapped through C1's song clocks, then CHECKED against the heat's own audio (both videos) and written in the
demo's layout. It never writes audio except a clicks overlay of the heat's own sound (work/, git-ignored).

    python -m dancecap.carlos_aline_beats run                       # everything below
    python -m dancecap.carlos_aline_beats run --device cpu          # beat_this on the CPU
    python -m dancecap.carlos_aline_beats run --phase kojo=1        # force a bar phase (pulse index mod 4)
    python -m dancecap.carlos_aline_beats run --steps work/carlos_aline/timing/timing.json
                                                                    # let the dancers' steps pick the governing song
    python -m dancecap.carlos_aline_beats follow --steps <timing.json | steps.json>    # step phase lock per song

Clock convention (alignment_carlos_aline.json "songs", sync.json): song_time = slope * reference_time + offset, song
time = the gapless ffmpeg decode of the mp3 from its first sample, reference = camera 01's video time. slope < 1:
a reference second holds slope song seconds, so the heat plays slower than the studio (tempo ratio = slope).

The beat level (the demo's zouk grid, dancecap.beats): the DANCE pulse is the tracked beat folded into [60, 120) BPM of
studio time; beat_this switches level inside Kojo Funds (84 and 168 BPM alternate), so the pulse is a straight line
t_k = t0 + k P fitted on the beats that sit on integer k (the half-integer ones are the eighths), checked for drift
per 20 s. Eighths are the midpoints of consecutive pulses, 8 slots per 4-pulse measure, types 1 3 3 2 1 3 2 3. The bar
phase comes from beat_this downbeats (vote of k mod 4); when that is ambiguous (Kojo Funds: 2-pulse bars) the kick /
backbeat rotation test decides, and the result is flagged low confidence.

Outputs (work/carlos_aline/song/, git-ignored; times on the reference clock; the demo's layout, see NOTES_C4.md):
  song.json                 songs, clocks, audible spans, timeline, governing switch, tempo ratio, files
  beats_studio.json         GOVERNING grid [[t, type]]: whole measures, time-ordered, Kojo then Guetta (a drop-in
                            --beats for timing / physics; measure = index // 8). Nothing before 13.31 s or after 194.25 s
  beats_studio_kojo.json, beats_studio_guetta.json    each song's whole-measure grid over its own audible span
                            (they overlap in the mix; their union is NOT time-ordered)
  beats_merged.json         every pulse and eighth of BOTH grids with song, governing, overlap, confidence parts
  downbeats.json            bars of both grids (t, song, measure, governing, confidence)
  tempo.json                per-song pulse tempo (studio, heat, ratio), bar length, level decision, tempo curve
  segments.json             timeline none / kojo / mix / guetta / none, governing intervals, no-beat intervals
  presence.json             how audible each song is in the heat's audio (z over time), the governing decision
  validation.json           beat times against the heat's own onsets (folded flux per window, both cameras) and
                            against beat_this run on the heat audio itself
  beats_studio_raw.json     provenance + everything the above are cut from
  review/carlos_aline_beats.png   beats over the heat's onset envelope, overlap, offsets, tempo
  heat_clicks_overlay.wav   camera 01's sound + clicks on the governing grid (listening check, work/ only)
Hook for the dancers (poses exist later): `run --steps` / `follow` decide which grid the dancers follow in the mix
from their step times; the audio-dominant default is used until then.
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
from . import song as S
from .take import PACKAGE_ROOT, Take

VERSION = 1
TAKE = "carlos_aline"
SR = B.ANALYSIS_SR  # 22050: beat_this input and the onset features
ZOUK_TYPES = B.ZOUK_TYPES
SONGS = ("kojo", "guetta")
OUT_DIR = "song"
CAMS = ("01", "02")
MIN_BPM = 60.0  # the dance pulse is folded into [60, 120) BPM of studio time (beats.py)
BEAT_TOL = 0.12  # a tracked beat supports pulse k when within this fraction of the period
PRESENT_Z = 3.5  # a song is audible in a 3 s window above this z
BANDS = {"bass_30_150": (30.0, 150.0), "mid_1_4k": (1000.0, 4000.0), "hi_4_9k": (4000.0, 9000.0)}  # onset flux bands (Hz)
FEATURES = ("broadband",) + tuple(BANDS)
HEADLINE_BANDS = ("mid_1_4k", "hi_4_9k")  # the bands in which the two cameras' onsets agree (see validate)


# ------------------------------------------------------------------------------------------------ clocks
class Clock:
    """song_time = slope * reference_time + offset (alignment_carlos_aline.json "songs")."""

    def __init__(self, slope: float, offset: float):
        self.slope, self.offset = float(slope), float(offset)

    def to_ref(self, song_t):
        return (np.asarray(song_t, float) - self.offset) / self.slope

    def to_song(self, ref_t):
        return self.slope * np.asarray(ref_t, float) + self.offset

    @property
    def tempo_ratio(self) -> float:
        """heat tempo / studio tempo (< 1: the heat plays slower)."""
        return self.slope

    @property
    def duration_stretch(self) -> float:
        """heat seconds per studio second (the 1.040 / 1.061 of sync.json)."""
        return 1.0 / self.slope

    def as_dict(self) -> dict:
        return {"slope": self.slope, "offset": self.offset, "convention": "song_time = slope * reference_time + offset",
                "tempo_ratio_heat_over_studio": self.tempo_ratio, "duration_stretch": self.duration_stretch}


# ------------------------------------------------------------------------------------------------ pulse model
def pulse_period(beats: np.ndarray, lo_bpm: float = MIN_BPM, tol: float = 0.12) -> tuple[float, dict]:
    """Dance-pulse period (s): the median tracked interval folded by powers of two into [lo_bpm, 2 lo_bpm), then
    refined to the median of the tracked intervals within +-tol of it (the dominant interval at that level)."""
    ibi = np.diff(np.asarray(beats, float))
    ibi = ibi[ibi > 0.05]
    if len(ibi) < 3:
        raise ValueError("fewer than 4 tracked beats")
    m = float(np.median(ibi))
    p = next(m * 2.0 ** j for j in range(-3, 4) if lo_bpm <= 60.0 / (m * 2.0 ** j) < 2 * lo_bpm)
    near = ibi[np.abs(ibi / p - 1) < tol]
    return (float(np.median(near)) if len(near) >= 3 else p), \
        {"tracked_bpm_median": round(60.0 / m, 3), "folded_to_bpm": round(60.0 / p, 3), "n_at_level": int(len(near)),
         "factor": round(p / m, 3)}


def _track_lattice(b: np.ndarray, P: float, tol: float, win: int = 12, max_miss: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Pulse indices and times of the beats that sit on one pulse lattice, found by walking outwards from the longest run of
    beats one period apart: the next pulse is predicted by a line through the nearest `win` supported pulses and takes the
    beat within tol * P of the prediction (none: the walk steps on by prediction, at most max_miss in a row). Beats half-way
    between pulses (the tracker's eighths) are never within tol * P of a prediction, so they are skipped by construction,
    and a tempo that changes slowly is followed."""
    ibi = np.diff(b)
    good = np.abs(ibi / P - 1) < 0.08
    best, run, i0 = (0, 0), 0, 0
    for i, g in enumerate(good):
        run = run + 1 if g else 0
        if run > best[0]:
            best, i0 = (run, i), i
    if best[0] < 3:
        raise ValueError("pulse fit: no run of 4 beats one period apart")
    first = i0 - best[0] + 1
    ks, ts = list(range(best[0] + 1)), [float(x) for x in b[first:first + best[0] + 1]]

    def predict(k_next: int, front: bool) -> float:
        idx = np.argsort(np.abs(np.asarray(ks) - k_next))[:win]
        kk, tt = np.asarray(ks)[idx], np.asarray(ts)[idx]
        if len(kk) >= 3:
            A = np.polyfit(kk, tt, 1)
            return float(np.polyval(A, k_next))
        return float(ts[-1] + P) if front else float(ts[0] - P)

    for front in (True, False):
        k = ks[-1] if front else ks[0]
        miss = 0
        while True:
            k += 1 if front else -1
            tp = predict(k, front)
            if (front and tp > b[-1] + tol * P) or (not front and tp < b[0] - tol * P):
                break
            near = np.where(np.abs(b - tp) < tol * P)[0]
            if len(near):
                j = int(near[np.argmin(np.abs(b[near] - tp))])
                ks.append(k) if front else ks.insert(0, k)
                ts.append(float(b[j])) if front else ts.insert(0, float(b[j]))
                miss = 0
            else:
                miss += 1
                if miss > max_miss:
                    break
                # carry the prediction forward as a (virtual) support so the walk can continue across a gap
                ks.append(k) if front else ks.insert(0, k)
                ts.append(tp) if front else ts.insert(0, tp)
    ks, ts = np.asarray(ks, float), np.asarray(ts, float)
    real = np.asarray([bool(np.abs(b - t).min() < 1e-9) for t in ts])
    return ks[real], ts[real]


def fit_pulses(beats: np.ndarray, span: tuple[float, float] | None = None, lo_bpm: float = MIN_BPM,
               tol: float = BEAT_TOL) -> dict:
    """Pulse lattice t_k = t0 + k * period from tracked beats of mixed level.

    The dance pulse period is the tracked beat interval folded into [lo_bpm, 2 lo_bpm) (pulse_period); the beats that sit
    on one lattice are found by walking outwards from the longest run of beats one period apart (_track_lattice) and a
    least-squares line is fitted to them (after a 3-MAD outlier pass). `span` limits the fit to beats inside [lo, hi]
    (song time). The straight line is the model while no 20 s window of the supporting beats sits more than 8 ms off it
    ('constant'); otherwise pulse_times follows the supporting beats locally.
    Returns {period, t0, k, t, resid_ms, rms_ms, n_support, n_fit, windows [{t, mean_ms, n}], max_window_drift_ms, constant, level}.
    """
    b = np.asarray(beats, float)
    if span is not None:
        b = b[(b >= span[0]) & (b <= span[1])]
    P, level = pulse_period(b, lo_bpm)
    k, t = _track_lattice(b, P, tol)
    for _ in range(3):
        P, t0 = (float(v) for v in np.polyfit(k, t, 1))
        res = t - (t0 + P * k)
        mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826
        keep = np.abs(res - np.median(res)) < max(3 * mad, 0.004)
        if keep.all():
            break
        k, t = k[keep], t[keep]
    P, t0 = (float(v) for v in np.polyfit(k, t, 1))
    res = t - (t0 + P * k)
    wins = []
    edges = np.arange(t[0], t[-1] + 20.0, 20.0)
    for lo_t, hi_t in zip(edges[:-1], edges[1:]):
        m = (t >= lo_t) & (t < hi_t)
        if m.sum() >= 3:
            wins.append({"t": round(float(0.5 * (lo_t + hi_t)), 2), "mean_ms": round(1000 * float(res[m].mean()), 2),
                         "n": int(m.sum())})
    drift = max((abs(w["mean_ms"]) for w in wins), default=0.0)
    return {"period": P, "t0": t0, "k": k, "t": t, "resid_ms": 1000 * res,
            "rms_ms": round(1000 * float(np.sqrt(np.mean(res ** 2))), 2), "n_support": int(len(k)),
            "n_fit": int(len(b)), "windows": wins, "max_window_drift_ms": round(drift, 2),
            "constant": bool(drift <= 8.0), "level": level}


def pulse_times(model: dict, k) -> np.ndarray:
    """Pulse time (song s) of integer pulse indices k: the straight line, or (non-constant tempo) a tricube-weighted
    local line over +-12 pulses of the supporting beats."""
    k = np.asarray(k, float)
    if model.get("constant", True):
        return model["t0"] + model["period"] * k
    ks, ts = model["k"], model["t"]
    out = np.empty(len(k))
    for i, kk in enumerate(k):
        m = np.abs(ks - kk) <= 12
        if m.sum() < 3:
            out[i] = model["t0"] + model["period"] * kk
            continue
        w = (1 - (np.abs(ks[m] - kk) / 13.0) ** 3) ** 3
        A = np.stack([np.ones(m.sum()), ks[m] - kk], 1) * np.sqrt(w)[:, None]
        out[i] = np.linalg.lstsq(A, ts[m] * np.sqrt(w), rcond=None)[0][0]
    return out


def tracker_support(prob: np.ndarray, t: np.ndarray, fps: float = B.BEAT_THIS_FPS, half: int = 2) -> np.ndarray:
    """beat_this beat probability at each time: the max within +-half frames (+-40 ms at 50 fps); 0 outside."""
    f = np.round(np.asarray(t, float) * fps).astype(int)
    out = np.zeros(len(f))
    for i, fi in enumerate(f):
        lo, hi = fi - half, fi + half + 1
        if lo >= 0 and hi <= len(prob):
            out[i] = float(prob[lo:hi].max())
    return out


def downbeat_votes(downbeats: np.ndarray, model: dict, bar: int = 4, tol: float = 0.2) -> dict:
    """Bar phase of the pulse lattice from beat_this downbeats: votes[j] = downbeats on pulses with k mod bar == j."""
    d = np.asarray(downbeats, float)
    kd = np.round((d - model["t0"]) / model["period"])
    ok = np.abs(d - (model["t0"] + model["period"] * kd)) < tol * model["period"]
    votes = np.bincount((kd[ok] % bar).astype(int), minlength=bar)
    p = int(votes.argmax()) if votes.sum() else 0
    agree = float(votes[p] / votes.sum()) if votes.sum() else 0.0
    half = float(votes[(p + bar // 2) % bar] / max(1, votes[p]))
    return {"phase": p, "votes": votes.tolist(), "agreement": round(agree, 4), "n": int(ok.sum()),
            "half_bar_ratio": round(half, 3), "ambiguous": bool(agree < 0.8 or half > 0.5)}


def zouk_rows(model: dict, k_first: int, k_last: int, phase: int) -> np.ndarray:
    """(N, 5) [t_song, type, measure, slot, k_position] for every WHOLE measure whose
    downbeat pulse k satisfies k mod 4 == phase and whose closing pulse (k + 4) is <= k_last."""
    ks = np.arange(k_first, k_last + 1)
    t = dict(zip(ks.tolist(), pulse_times(model, ks)))
    first = k_first + ((phase - k_first) % 4)
    rows, m = [], 0
    for k0 in range(first, k_last - 3, 4):
        for j in range(8):
            a = k0 + j // 2
            tt = t[a] if j % 2 == 0 else 0.5 * (t[a] + t[a + 1])
            rows.append((tt, ZOUK_TYPES[j], m, j, k0 + j / 2))
        m += 1
    return np.asarray(rows, float).reshape(-1, 5)


def whole_measures(rows: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    """Rows [t_ref, type, measure, slot, ...] of the whole measures with all 8 slots inside [t_lo, t_hi], renumbered."""
    if not len(rows):
        return rows
    keep = [m for m in np.unique(rows[:, 2]).astype(int)
            if (r := rows[rows[:, 2] == m])[0, 0] >= t_lo and r[-1, 0] <= t_hi and len(r) == 8]
    out = rows[np.isin(rows[:, 2].astype(int), keep)].copy()
    out[:, 2] = np.searchsorted(np.asarray(keep), out[:, 2].astype(int)) if keep else out[:, 2]
    return out


# ------------------------------------------------------------------------------------------------ the mix
def timeline(kojo: tuple[float, float], guetta: tuple[float, float], t_end: float) -> list[dict]:
    """none / kojo / mix / guetta / none intervals (reference s) from the two songs' audible spans (sync.json order)."""
    pts = sorted({0.0, kojo[0], kojo[1], guetta[0], guetta[1], t_end})
    out = []
    for lo, hi in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (lo + hi)
        on = [s for s, (a, b) in zip(SONGS, (kojo, guetta)) if a <= mid < b]
        state = "none" if not on else "mix" if len(on) == 2 else on[0]
        if out and out[-1]["state"] == state:
            out[-1]["to_s"] = hi
        else:
            out.append({"from_s": lo, "to_s": hi, "state": state, "songs": on})
    return out


def state_at(segments: list[dict], t: float) -> str:
    for s in segments:
        if s["from_s"] <= t < s["to_s"]:
            return s["state"]
    return "none"


def audio_switch(t: np.ndarray, z_kojo: np.ndarray, z_guetta: np.ndarray, overlap: tuple[float, float],
                 min_run_s: float = 2.0) -> dict:
    """Where the audio hands over from Kojo to Guetta inside the mix [lo, hi]: the first time Guetta's presence is
    >= Kojo's for at least min_run_s (both z smoothed by the caller); Kojo's end (hi) when it never is, lo when
    Guetta leads from the start. Returns {switch_s, by}."""
    lo, hi = overlap
    t = np.asarray(t, float)
    m = (t >= lo) & (t <= hi)
    if not m.any():
        return {"switch_s": 0.5 * (lo + hi), "by": "no presence samples in the mix: midpoint"}
    lead = np.asarray(z_guetta)[m] >= np.asarray(z_kojo)[m]
    tt = t[m]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    need = max(1, int(round(min_run_s / dt)))
    for i in range(len(tt) - need + 1):
        if lead[i:i + need].all():
            return {"switch_s": float(tt[i]), "by": f"Guetta's presence >= Kojo's for {min_run_s:g} s"}
    return {"switch_s": float(hi), "by": "Guetta never leads inside the mix: Kojo until its end"}


def govern(rows_a: np.ndarray, rows_b: np.ndarray, switch: float) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered governing grids from song A (earlier) and song B (later), rows [t, type, measure, slot, ...] in
    reference time, whole measures. A keeps the measures whose midpoint is before `switch`; B starts at its first
    downbeat after A's last kept measure (and not before `switch` minus half a B bar): at most one B bar of the mix is
    left without a governing grid, and the two kept lists never overlap in time."""
    def mids(r):
        return {int(m): float(r[r[:, 2] == m][:, 0].mean()) for m in np.unique(r[:, 2])} if len(r) else {}
    ma = mids(rows_a)
    keep_a = [m for m, c in ma.items() if c < switch]
    a = rows_a[np.isin(rows_a[:, 2].astype(int), keep_a)] if keep_a else rows_a[:0]
    t_end_a = (float(a[-1, 0]) + float(np.median(np.diff(a[:, 0])))) if len(a) > 1 else -np.inf
    keep_b = [int(m) for m in np.unique(rows_b[:, 2])
              if rows_b[(rows_b[:, 2] == m) & (rows_b[:, 3] == 0)][0, 0] >= t_end_a - 1e-9]
    b = rows_b[np.isin(rows_b[:, 2].astype(int), keep_b)] if len(rows_b) else rows_b
    if len(b):
        bar = float(np.median(np.diff(b[b[:, 3] == 0][:, 0]))) if (b[:, 3] == 0).sum() > 1 else 0.0
        b = b[b[:, 0] >= switch - 0.5 * bar]
        keep = np.unique(b[:, 2])
        full = [m for m in keep if (b[:, 2] == m).sum() == 8]
        b = b[np.isin(b[:, 2], full)]
    return a, b


def phase_lock(steps: np.ndarray, pulses: np.ndarray, t_lo: float, t_hi: float) -> dict:
    """How tightly the dancers' step times (reference s) sit on one grid's pulses inside [t_lo, t_hi]: the mean resultant
    length R of the step phase against the pulse (harmonic 1) and the eighth (harmonic 2), R in [0, 1] (random ~ 1/sqrt(n)),
    and the mean offset (ms) of the best harmonic. Steps outside the pulse range are ignored."""
    s = np.sort(np.asarray(steps, float))
    s = s[(s >= t_lo) & (s <= t_hi)]
    p = np.sort(np.asarray(pulses, float))
    j = np.searchsorted(p, s, side="right") - 1
    ok = (j >= 0) & (j < len(p) - 1)
    s, j = s[ok], j[ok]
    if len(s) < 4:
        return {"n": int(len(s)), "R": None, "R1": None, "R2": None, "mean_offset_ms": None}
    ph = (s - p[j]) / (p[j + 1] - p[j])
    out = {"n": int(len(s))}
    best = (-1.0, 0.0)
    for h in (1, 2):
        z = np.exp(2j * np.pi * h * ph).mean()
        out[f"R{h}"] = round(float(abs(z)), 4)
        if abs(z) > best[0]:
            best = (float(abs(z)), float(np.angle(z) / (2 * np.pi * h)))
    per = float(np.median(np.diff(p)))
    out["R"] = round(best[0], 4)
    out["mean_offset_ms"] = round(1000 * best[1] * per, 1)
    out["chance_R"] = round(float(1 / math.sqrt(len(s))), 3)
    return out


def follow_choice(steps: np.ndarray, pulses: dict[str, np.ndarray], overlap: tuple[float, float],
                  margin: float = 0.12) -> dict:
    """Which song's pulses the dancers' steps follow inside the mix. `pulses`: song -> pulse times (reference s). A
    song wins when its R beats the other's by more than `margin` and is itself above 2.5 / sqrt(n) (chance level for n
    steps); else 'undecided' (the audio default stays)."""
    lo, hi = overlap
    lock = {s: phase_lock(steps, pulses[s], lo, hi) for s in SONGS}
    r = {s: (lock[s]["R"] if lock[s]["R"] is not None else -1.0) for s in SONGS}
    if min(lock[s]["n"] for s in SONGS) < 6:
        return {"follows": "undecided", "why": "fewer than 6 steps inside the mix", "lock": lock}
    d = r["guetta"] - r["kojo"]
    win = "guetta" if d > 0 else "kojo"
    need = 2.5 / math.sqrt(lock[win]["n"])  # the winner must also lock clearly above chance (R ~ 1 / sqrt(n) for random steps)
    who = win if abs(d) > margin and r[win] >= need else "undecided"
    return {"follows": who, "R_guetta_minus_R_kojo": round(d, 4), "margin": margin, "needs_R": round(need, 3), "lock": lock}


def follow_curve(steps: np.ndarray, pulses: dict[str, np.ndarray], t_lo: float, t_hi: float, win_s: float = 10.0,
                 hop_s: float = 2.0) -> list[dict]:
    """Step phase lock against each song's pulses in sliding windows: the check over the whole heat."""
    out = []
    for t0 in np.arange(t_lo, t_hi - win_s + 1e-9, hop_s):
        row = {"t0": float(t0), "t1": float(t0 + win_s)}
        for s in SONGS:
            p = pulses[s]
            row[s] = phase_lock(steps, p[(p >= t0 - 1) & (p <= t0 + win_s + 1)], t0, t0 + win_s) if len(p) > 3 else None
        out.append(row)
    return out


def load_steps(path: Path) -> np.ndarray:
    """Step times (reference s) from timing/timing.json (footsteps.{lead,follow}[{t}]) or a plain list / {"steps": [..]}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, list):
        return np.sort(np.asarray([x["t"] if isinstance(x, dict) else x for x in d], float))
    if "steps" in d:
        return np.sort(np.asarray(d["steps"], float))
    fs = d.get("footsteps") or {}
    t = [float(r["t"]) for role in fs.values() if isinstance(role, list) for r in role if "t" in r]
    return np.sort(np.asarray(t, float))


# ------------------------------------------------------------------------------------------------ onset evidence
def fold_profile(G: np.ndarray, dt: float, t0: float, pulses: np.ndarray, lags: np.ndarray) -> np.ndarray | None:
    """Mean onset flux G (frame dt, frame 0 at time t0) at lags around each pulse (None: fewer than 6 pulses fit)."""
    ts = t0 + np.arange(len(G)) * dt
    ok = (pulses + lags[0] > ts[0]) & (pulses + lags[-1] < ts[-1])
    if ok.sum() < 6:
        return None
    return np.mean([np.interp(p + lags, ts, G) for p in pulses[ok]], 0)


def profile_peak(prof: np.ndarray, lags: np.ndarray, search_s: float, smooth_bins: int = 3) -> float:
    """Lag (s) of the highest point of the folded profile within +-search_s of the pulse (parabolic sub-bin)."""
    y = np.convolve(prof, np.ones(smooth_bins) / smooth_bins, "same")
    step = float(lags[1] - lags[0])
    m = np.where(np.abs(lags) <= search_s)[0]
    m = m[(m > 0) & (m < len(y) - 1)]
    k = int(m[np.argmax(y[m])])
    return float(lags[k] + S._parabola(y, k) * step)


def _xcorr_shift(a: np.ndarray, b: np.ndarray, step_s: float, max_shift_s: float) -> tuple[float, float]:
    """Lag (s) and normalised value of the best cross-correlation of a(l) ~ b(l - shift) within +-max_shift_s."""
    n = int(round(max_shift_s / step_s))
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    L = len(a)
    r = np.asarray([float(np.mean(a[max(0, k):min(L, L + k)] * b[max(0, k) - k:min(L, L + k) - k]))
                    for k in range(-n, n + 1)])
    k = int(np.argmax(r))
    return float((k - n + (S._parabola(r, k) if 0 < k < len(r) - 1 else 0.0)) * step_s), float(r[k])


def profile_shift(hp: np.ndarray, sp: np.ndarray, step_s: float, max_shift_s: float = 0.08) -> tuple[float, float]:
    """Shift (s) of the heat's folded profile against the studio's: heat(l) ~ studio(l - shift), by normalised
    cross-correlation over +-max_shift_s with a parabolic peak. > 0: the heat's onsets come later. Returns (shift, ncc).
    Correlating the whole profile (kick, snare, hats, the syncopated offbeats) does not flip between the several
    peaks of one pulse the way a single peak pick does."""
    return _xcorr_shift(hp, sp, step_s, max_shift_s)


def window_flux_shift(Gh: np.ndarray, Gs: np.ndarray, dt: float, max_shift_s: float = 0.08) -> tuple[float, float]:
    """The same shift for two flux series on one frame grid (heat, studio warped onto the reference clock): the lag of
    the best normalised cross-correlation over the whole window, not only at the beats. Returns (shift_s, ncc)."""
    return _xcorr_shift(Gh, Gs, dt, max_shift_s)


def drift_fit(t: np.ndarray, y_ms: np.ndarray) -> dict:
    """Median and drift (ms per minute) of window offsets (robust line)."""
    t, y = np.asarray(t, float), np.asarray(y_ms, float)
    if len(t) < 3:
        return {"n": int(len(t)), "median_ms": round(float(np.median(y)), 2) if len(y) else None, "drift_ms_per_min": None}
    A = np.polyfit(t, y, 1)
    r = y - np.polyval(A, t)
    keep = np.abs(r) < max(3 * 1.4826 * np.median(np.abs(r - np.median(r))), 3.0)
    if keep.sum() >= 3:
        A = np.polyfit(t[keep], y[keep], 1)
    return {"n": int(len(t)), "median_ms": round(float(np.median(y)), 2),
            "mad_ms": round(float(1.4826 * np.median(np.abs(y - np.median(y)))), 2),
            "drift_ms_per_min": round(float(A[0] * 60), 2)}


def nearest_offsets(x: np.ndarray, ref: np.ndarray, max_abs: float) -> np.ndarray:
    """For each x the signed distance to the nearest ref (sorted) when within max_abs, else dropped."""
    ref = np.sort(np.asarray(ref, float))
    if not len(ref):
        return np.zeros(0)
    j = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1)
    d = np.where(np.abs(ref[j - 1] - x) < np.abs(ref[j] - x), ref[j - 1] - x, ref[j] - x)
    return d[np.abs(d) <= max_abs]


def audibility(z: np.ndarray, lo: float = 1.5, hi: float = 6.0) -> np.ndarray:
    """Ramp 0..1 of the song's presence z in the heat's audio (0 at z <= lo, 1 at z >= hi)."""
    return np.clip((np.asarray(z, float) - lo) / (hi - lo), 0.0, 1.0)


def beat_confidence(p_track: np.ndarray, z: np.ndarray, interpolated: np.ndarray) -> np.ndarray:
    """Per-beat confidence in [0, 1] that this beat is on the song's lattice AND audible in the heat: beat_this's probability
    at the pulse x the audibility ramp of the song's presence z there x 0.7 when no tracked beat supports the pulse
    (interpolated). Eighths take the lower of their two pulses' probabilities."""
    c = np.clip(np.asarray(p_track, float), 0, 1) * audibility(z)
    return np.round(np.where(interpolated, 0.7 * c, c), 4)


def heat_support(G: np.ndarray, dt: float, t0: float, beats: np.ndarray, bias_s: float, half_s: float = 0.01,
                 neigh_s: float = 1.0) -> np.ndarray:
    """Onset evidence of the heat's audio at each beat: the mean flux within +-half_s of the (bias-corrected) beat, as the
    fraction of the frames within +-neigh_s that are weaker, mapped so that chance (0.5) is 0 and 'the strongest frame
    around' is 1."""
    ts = t0 + np.arange(len(G)) * dt
    out = np.zeros(len(beats))
    for i, b in enumerate(np.asarray(beats, float) + bias_s):
        i0, i1 = np.searchsorted(ts, [b - half_s, b + half_s])
        j0, j1 = np.searchsorted(ts, [b - neigh_s, b + neigh_s])
        if i1 <= i0 or j1 - j0 < 20:
            continue
        v = float(G[i0:i1].mean())
        out[i] = float(np.clip((np.mean(G[j0:j1] < v) - 0.5) * 2.0, 0.0, 1.0))
    return out


# ------------------------------------------------------------------------------------------------ io / setup
def work_dir() -> Path:
    return PACKAGE_ROOT / "work" / TAKE


def song_dir() -> Path:
    return work_dir() / OUT_DIR


def load_config(take: Take | None = None) -> dict:
    """Songs, clocks and audible spans from takes/carlos_aline.toml + alignment_carlos_aline.json "songs" + sync.json."""
    take = take or Take.load(TAKE)
    tp = PACKAGE_ROOT / "takes" / f"{TAKE}.toml"
    dance = S.read_toml(tp).get("dance") or {}
    al = json.loads((PACKAGE_ROOT / "takes" / take.cfg["alignment"]).resolve().read_text(encoding="utf-8"))["songs"]
    sync_p = work_dir() / "sync" / "sync.json"
    sync = json.loads(sync_p.read_text(encoding="utf-8")) if sync_p.exists() else {}
    files = {"kojo": dance["song_file"], "guetta": dance["song_file_2"]}
    cfg = {"songs": {}, "files": {s: S.resolve_path((tp.parent / f).resolve()) for s, f in files.items()},
           "take": take, "danced_span": [float(x) for x in take.window]}
    for s in SONGS:
        a = al[s]
        cfg["songs"][s] = {"clock": Clock(a["slope"], a["offset"]), "audible": [float(x) for x in a["audible_reference_s"]],
                           "title": a.get("title", ""), "artist": a.get("artist", ""),
                           "song_zero_at_reference_s": a.get("song_zero_at_reference_s"), "residual_ms": a.get("residual_ms")}
    cfg["duration_s"] = float((sync.get("cameras") or [{}])[0].get("duration_s", 202.083))
    cfg["sync_fit"] = {s: (sync.get("songs", {}).get(s) or {}).get("fit") for s in SONGS}
    return cfg


def camera_clock(take: Take, cam: str) -> tuple[float, float]:
    src = take.source(cam)
    return src.slope, src.offset


def device_ctx(device: str | None):
    """('cuda', lock) when CUDA is free for a few seconds, else ('cpu', null)."""
    import torch

    from . import gpu
    if device == "cpu" or not torch.cuda.is_available():
        return "cpu", contextlib.nullcontext()
    try:
        ctx = gpu.exclusive("carlos-aline-beats", timeout_s=60.0)
        ctx.__enter__()
    except TimeoutError:
        return "cpu", contextlib.nullcontext()
    return "cuda", B._Entered(ctx)


def _jsonable(o):
    return S._np(o)


def write_json(path: Path, obj, indent: int | None = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, default=_jsonable), encoding="utf-8")
    return path


def _r(a, nd: int = 4):
    return np.round(np.asarray(a, float), nd).tolist()


# ------------------------------------------------------------------------------------------------ run
def track_all(cfg: dict, device: str | None, cache: Path) -> tuple[dict, dict, str]:
    """beat_this on the two studio mp3s and on the heat audio of both videos. Returns (audio, tracker outputs, device)."""
    take: Take = cfg["take"]
    audio = {}
    for s in SONGS:
        audio[s] = B.decode_audio(cfg["files"][s], SR)[0]
    for cam in CAMS:
        audio[cam] = B.decode_audio(take.source(cam).path, SR)[0]
    cache.mkdir(parents=True, exist_ok=True)
    trk, dev = {}, "cache"
    todo = [k for k in audio if not (cache / f"beat_this_{k}.npz").exists()]
    if todo:
        dev, lock = device_ctx(device)
        with lock:
            tracker = B.BeatTracker("final0", dev)
            for k in todo:
                r = tracker(audio[k], SR)
                np.savez_compressed(cache / f"beat_this_{k}.npz", **{n: np.asarray(v) for n, v in r.items()})
                print(f"[beats] beat_this on {k}: {len(r['beats'])} beats, {len(r['downbeats'])} downbeats ({dev})", flush=True)
    for k in audio:
        z = np.load(cache / f"beat_this_{k}.npz")
        trk[k] = {n: z[n] for n in z.files}
    return audio, trk, dev


def song_grid(s: str, cfg: dict, x: np.ndarray, tr: dict, phase_override: int | None) -> dict:
    """The song's pulse model, bar phase and zouk grid in song time and reference time (audible span only)."""
    info = cfg["songs"][s]
    clk: Clock = info["clock"]
    a_lo, a_hi = info["audible"]
    s_lo, s_hi = (float(clk.to_song(a_lo)), float(clk.to_song(a_hi)))
    span = (max(0.0, s_lo) - 2.0, s_hi + 12.0)
    model = fit_pulses(tr["beats_refined"], span=span)
    votes = downbeat_votes(tr["downbeats_refined"][(tr["downbeats_refined"] >= span[0]) & (tr["downbeats_refined"] <= span[1])], model)
    k_lo = int(math.ceil((max(s_lo, 0.0) - model["t0"]) / model["period"]))
    k_hi = int(math.floor((s_hi - model["t0"]) / model["period"]))
    phase, decision = votes["phase"], {"by": "beat_this downbeats", "ambiguous": votes["ambiguous"], "votes": votes}
    # kick / backbeat rotation test on the pulse grid (even rotations only: the pulse is anchored by the tracker)
    rows0 = zouk_rows(model, k_lo, k_hi, phase)
    lo_o, lo_t = B.band_onset(x, SR)
    sn_o, sn_t = B.band_onset(x, SR, 1000.0, 4000.0)
    scores = {}
    for p in range(4):
        g = zouk_rows(model, k_lo, k_hi, p)
        rot = B.rotation_scores(B.folded_profile(g[:, :4], lo_t, lo_o), B.folded_profile(g[:, :4], sn_t, sn_o))
        scores[p] = {"score": rot["scores"][0], "kick_z": rot["kick_z"], "snare_z": rot["snare_z"]}
    ranked = sorted(scores, key=lambda p: -scores[p]["score"])
    evidence = {"phase_scores": {p: round(scores[p]["score"], 3) for p in scores}, "best": ranked[0],
                "margin": round(scores[ranked[0]]["score"] - scores[ranked[1]]["score"], 3),
                "bands": "kick/bass < 150 Hz on beat 1 minus beat 3, snare 1-4 kHz on beats 2 and 4"}
    if phase_override is not None:
        phase = int(phase_override) % 4
        decision.update(by=f"override --phase ({phase})", confidence="given")
    elif votes["ambiguous"] and ranked[0] != phase and evidence["margin"] > 1.0:
        phase = ranked[0]
        decision.update(by="kick on 1 + snare backbeat (beat_this downbeats ambiguous)", confidence="low")
    else:
        decision["confidence"] = ("low" if votes["ambiguous"] else "high" if ranked[0] == phase else
                                  "medium: the kick / backbeat test prefers another pulse as the bar start")
    decision["evidence"] = evidence
    decision["phase"] = phase
    rows = zouk_rows(model, k_lo, k_hi, phase)
    # support per pulse (tracker) and interpolation flag
    ks = np.arange(k_lo, k_hi + 1)
    tk = pulse_times(model, ks)
    p_track = tracker_support(tr["beat_prob"], tk)
    near = np.array([np.abs(tr["beats_refined"] - t).min() < BEAT_TOL * model["period"] for t in tk])
    # the song-time placement of the beats against kick / hf attacks (the demo's beat_placement): a diagnostic
    place = S.beat_placement(x, SR, tk[(tk > 0.2) & (tk < len(x) / SR - 0.2)])
    return {"model": model, "votes": votes, "phase": phase, "decision": decision, "rows_song": rows, "k": ks, "t_song": tk,
            "p_track": p_track, "tracked": near, "placement": place, "span_song": [s_lo, s_hi]}


def run(device: str | None = None, phase: dict | None = None, steps: Path | None = None, plot: bool = True,
        fresh: bool = False) -> dict:
    t_start = time.time()
    cfg = load_config()
    take: Take = cfg["take"]
    sd = song_dir()
    sd.mkdir(parents=True, exist_ok=True)
    audio, trk, dev = track_all(cfg, device, sd / "_cache")

    # ---- heat onset features on the reference clock (both videos)
    dt = S.frame_dt(SR)
    heat = {}
    for cam in CAMS:
        sl, off = camera_clock(take, cam)
        F = S.onset_features(audio[cam], SR)
        lo_o, lo_t = B.band_onset(audio[cam], SR, 30.0, 200.0)  # kick / bass flux: the readable one for the review plot
        heat[cam] = {"F": F, "G": F.sum(0), "t0": off, "dt": dt * sl, "slope": sl, "offset": off,
                     "low": lo_o, "low_t": sl * lo_t + off, "bands": band_flux(audio[cam], SR)}
    studio = {s: {"F": S.onset_features(audio[s], SR)} for s in SONGS}
    for s in SONGS:
        studio[s]["G"] = studio[s]["F"].sum(0)
        studio[s]["bands"] = band_flux(audio[s], SR)

    # ---- grids per song (song time -> reference time)
    grids = {}
    for s in SONGS:
        g = song_grid(s, cfg, audio[s], trk[s], (phase or {}).get(s))
        clk: Clock = cfg["songs"][s]["clock"]
        rows = g["rows_song"].copy()
        rows[:, 0] = clk.to_ref(rows[:, 0])
        lo, hi = cfg["songs"][s]["audible"]
        g["rows_ref"] = whole_measures(rows, lo - 0.05, hi + 0.05)
        g["pulse_ref"] = clk.to_ref(g["t_song"])
        grids[s] = g
        print(f"[beats] {s}: pulse {g['model']['period']:.5f} s = {60 / g['model']['period']:.3f} BPM studio -> "
              f"{60 / g['model']['period'] * clk.tempo_ratio:.3f} BPM in the heat (ratio {clk.tempo_ratio:.5f}); "
              f"fit rms {g['model']['rms_ms']} ms, drift {g['model']['max_window_drift_ms']} ms, bar phase {g['phase']} "
              f"({g['decision']['by']}); {int(g['rows_ref'][:, 2].max()) + 1 if len(g['rows_ref']) else 0} whole measures", flush=True)

    # ---- how audible is each song in the heat (3 s windows, both cameras)
    pc = sd / "_cache" / "presence.json"
    if pc.exists() and not fresh:
        pres = json.loads(pc.read_text(encoding="utf-8"))
    else:
        pres = presence(cfg, heat, studio)
        write_json(pc, pres, indent=None)

    # ---- the mix: which grid governs
    seg = timeline(tuple(cfg["songs"]["kojo"]["audible"]), tuple(cfg["songs"]["guetta"]["audible"]), cfg["duration_s"])
    overlap = (cfg["songs"]["guetta"]["audible"][0], cfg["songs"]["kojo"]["audible"][1])
    zs = {s: np.asarray(pres["z"][s], float) for s in SONGS}
    sw = audio_switch(np.asarray(pres["t"]), smooth(zs["kojo"]), smooth(zs["guetta"]), overlap)
    follow = None
    if steps is not None:
        st = load_steps(steps)
        follow = follow_choice(st, {s: grids[s]["pulse_ref"] for s in SONGS}, overlap)
        if follow["follows"] == "guetta":
            sw = {"switch_s": overlap[0], "by": "the dancers' steps lock to Guetta's pulses in the mix (--steps)"}
        elif follow["follows"] == "kojo":
            sw = {"switch_s": overlap[1], "by": "the dancers' steps lock to Kojo's pulses in the mix (--steps)"}
    switch = float(np.clip(sw["switch_s"], *overlap))
    gov_k, gov_g = govern(grids["kojo"]["rows_ref"], grids["guetta"]["rows_ref"], switch)
    gov = np.vstack([gov_k, gov_g]) if len(gov_g) else gov_k
    gov_song = np.concatenate([np.zeros(len(gov_k), int), np.ones(len(gov_g), int)])
    gov_meas = np.concatenate([gov_k[:, 2], gov_g[:, 2] + (gov_k[:, 2].max() + 1 if len(gov_k) else 0)])
    print(f"[beats] mix {overlap[0]:.2f}-{overlap[1]:.2f} s; governing switch {switch:.2f} s ({sw['by']}); governing grid "
          f"{len(gov_k) // 8} Kojo + {len(gov_g) // 8} Guetta whole measures", flush=True)

    # ---- validation against the heat's own audio
    val = validate(cfg, grids, heat, studio, trk, overlap)

    # ---- per-beat confidence and merged table
    merged = merged_rows(cfg, grids, heat, studio, val, pres, gov, gov_song, overlap)
    val["onset_support"] = onset_support(merged)

    # ---- outputs
    out = write_outputs(cfg, grids, gov, gov_song, gov_meas, merged, seg, overlap, sw, switch, follow, pres, val, trk, dev, dt)
    if plot:
        out["review"] = [str(p) for p in review_plot(cfg, grids, heat, {k: studio[k]['G'] for k in SONGS}, gov, merged, seg, overlap, switch, pres, val)]
    clicks_overlay(cfg, audio, gov)
    out["seconds"] = round(time.time() - t_start, 1)
    return out


# ------------------------------------------------------------------------------------------------ presence
def smooth(z: np.ndarray, n: int = 3) -> np.ndarray:
    z = np.nan_to_num(np.asarray(z, float), nan=0.0)
    return np.convolve(np.pad(z, n // 2, mode="edge"), np.ones(n) / n, "valid")


def band_flux(x: np.ndarray, sr: int) -> dict:
    """Half-wave rectified log-energy flux of each BANDS band on the 5 ms frame grid of S.onset_features (hop 110 at 22.05 kHz)."""
    hop = int(round(S.HOP_S * sr))
    return {n: B.band_onset(x, sr, lo, hi, hop=hop)[0] for n, (lo, hi) in BANDS.items()}


def feature(d: dict, name: str) -> np.ndarray:
    """One onset feature of a heat / studio entry: the broadband flux or a band's flux (same frame grid)."""
    return d["G"] if name == "broadband" else d["bands"][name]


def _zc(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    return a / (a.std(1, keepdims=True) + 1e-9)


def _resample(F: np.ndarray, t0: float, dt: float, tt: np.ndarray) -> np.ndarray:
    ts = t0 + np.arange(F.shape[1]) * dt
    return np.stack([np.interp(tt, ts, f, left=0.0, right=0.0) for f in F])


def presence(cfg: dict, heat: dict, studio: dict, win_s: float = 3.0, hop_s: float = 1.0) -> dict:
    """z of the heat/studio onset correlation in 3 s windows: best of lags +-30 ms against the same correlation at
    lags 0.12-0.6 s (non-matching alignments): a song is audible where z >= PRESENT_Z. Mean over the two cameras."""
    dt = 0.005
    t_end = cfg["duration_s"]
    starts = np.arange(0.0, t_end - win_s, hop_s)
    z = {s: np.full((len(CAMS), len(starts)), np.nan) for s in SONGS}
    for ci, cam in enumerate(CAMS):
        h = heat[cam]
        for i, w0 in enumerate(starts):
            tt = np.arange(w0, w0 + win_s, dt)
            if tt[0] < h["t0"] or tt[-1] > h["t0"] + len(h["G"]) * h["dt"]:
                continue
            hz = _zc(_resample(h["F"], h["t0"], h["dt"], tt))
            for s in SONGS:
                clk: Clock = cfg["songs"][s]["clock"]
                F = studio[s]["F"]
                dts = S.frame_dt(SR)

                def corr(lag):
                    u = clk.to_song(tt + lag)
                    if u[0] < 0 or u[-1] > F.shape[1] * dts:
                        return np.nan
                    st = _resample(F, 0.0, dts, u)
                    return float((hz * _zc(st)).mean()) if st.std() > 1e-9 else np.nan
                fit = [corr(l) for l in np.arange(-0.03, 0.0301, 0.005)]
                r0 = np.nanmax(fit) if np.isfinite(fit).any() else np.nan
                null = np.array([corr(l) for l in np.concatenate([np.arange(-0.6, -0.11, 0.013), np.arange(0.12, 0.61, 0.013)])])
                if not np.isnan(r0) and np.isfinite(null).sum() > 10:
                    z[s][ci, i] = (r0 - np.nanmean(null)) / (np.nanstd(null) + 1e-9)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zm = {s: np.nanmean(z[s], 0) for s in SONGS}  # NaN: the window is not fully inside the song / the recording
    nn = lambda a: [None if not np.isfinite(v) else round(float(v), 3) for v in a]  # noqa: E731
    return {"t": (starts + 0.5 * win_s).tolist(), "win_s": win_s, "z": {s: nn(zm[s]) for s in SONGS},
            "z_cam": {s: {c: nn(z[s][i]) for i, c in enumerate(CAMS)} for s in SONGS}, "present_z": PRESENT_Z,
            "note": "null: no full window of the song inside the recording there; consumers treat it as 0 (audio_switch, "
                    "plot) or take the nearest valid value (the beats at the song's edges)"}


# ------------------------------------------------------------------------------------------------ validation
def validate(cfg: dict, grids: dict, heat: dict, studio: dict, trk: dict, overlap: tuple[float, float],
             win_s: float = 24.0, hop_s: float = 8.0) -> dict:
    """Beat times against the heat's audible onsets (both videos' audio tracks), three ways.

    agreement: the heat's onset flux folded on the pulses (24 s windows and all solo pulses) and the STUDIO flux folded
    on the same pulses (song time, lags scaled by the song slope) are cross-correlated per band (broadband, bass 30-150 Hz,
    mid 1-4 kHz, hi 4-9 kHz): agreement_ms = how much later the heat's onsets come than the studio's at the pulses = the
    song clock's error plus the room (sound travel, phone, codec). Median over windows, MAD, drift per minute.
    camera_pair: the same band flux of camera 01 against camera 02 (12 s windows) - the two videos carry one sound
    (waveform GCC-PHAT 0.5 ms) but their band onsets differ by band: this separates room / phone from the clock.
    heat_beat_this: beat_this on each heat audio itself, each beat to the nearest pulse (within a third of the period): the
    median signed offset carries the tracker's bias on that recording, a wrong clock would show as a drift.
    Only the solo parts count (the mix and 1.5 s inside the audible span are left out)."""
    out = {"method": "folded onset-flux profile cross-correlation per band; camera 01 vs 02 band flux; beat_this on the heat audio",
           "features": list(FEATURES), "headline_bands": list(HEADLINE_BANDS)}
    dts = S.frame_dt(SR)
    agree, trk_cmp, head = {}, {}, {}
    for s in SONGS:
        clk: Clock = cfg["songs"][s]["clock"]
        g = grids[s]
        P_ref = g["model"]["period"] / clk.slope
        half = min(0.30, 0.45 * P_ref)
        lags = np.arange(-half, half + 1e-9, 0.0025)
        lo, hi = cfg["songs"][s]["audible"]
        other = cfg["songs"]["guetta" if s == "kojo" else "kojo"]["audible"]
        a, b = (max(lo, 14.0) + 1.5, other[0] - 0.5) if s == "kojo" else (other[1] + 0.5, hi - 2.0)
        pr = g["pulse_ref"]
        agree[s] = {"span_s": [round(a, 2), round(b, 2)]}
        for fn in FEATURES:
            agree[s][fn] = {c: {"windows": [], "all_pulses": None, "summary": None} for c in CAMS}
            for w0 in list(np.arange(a, b - win_s + 1e-9, hop_s)) + [None]:
                p = pr[(pr >= a) & (pr <= b)] if w0 is None else pr[(pr >= w0) & (pr < w0 + win_s)]
                if len(p) < 10:
                    continue
                sp = fold_profile(feature(studio[s], fn), dts, 0.0, clk.to_song(p), lags * clk.slope)
                if sp is None:
                    continue
                for c in CAMS:
                    h = heat[c]
                    hp = fold_profile(feature(h, fn), h["dt"], h["t0"], p, lags)
                    if hp is None:
                        continue
                    sh, ncc = profile_shift(hp, sp, 0.0025)
                    if w0 is None:
                        agree[s][fn][c]["all_pulses"] = {
                            "agreement_ms": round(1000 * sh, 2), "ncc": round(ncc, 3), "n_pulses": int(len(p)),
                            "onset_lag_ms": round(1000 * profile_peak(hp, lags, 0.05), 1),
                            "studio_onset_lag_ms": round(1000 * profile_peak(sp, lags, 0.05 * clk.slope) / clk.slope, 1)}
                    else:
                        agree[s][fn][c]["windows"].append({"t": round(w0 + 0.5 * win_s, 2), "agreement_ms": round(1000 * sh, 2),
                                                           "ncc": round(ncc, 3), "n": int(len(p))})
            for c in CAMS:
                w = [r for r in agree[s][fn][c]["windows"] if r["ncc"] > 0.3]
                agree[s][fn][c]["summary"] = drift_fit([r["t"] for r in w], [r["agreement_ms"] for r in w])
                am = np.asarray([r["agreement_ms"] for r in w])
                med = agree[s][fn][c]["summary"].get("median_ms")
                agree[s][fn][c]["summary"]["share_within_10ms"] = (round(float(np.mean(np.abs(am - med) <= 10.0)), 3)
                                                                   if len(am) else None)
        head[s] = {c: {"agreement_ms_median": round(float(np.median(
            [agree[s][fn][c]["summary"]["median_ms"] for fn in HEADLINE_BANDS
             if agree[s][fn][c]["summary"]["median_ms"] is not None])), 1),
            "worst_band_drift_ms_per_min": round(float(max(abs(agree[s][fn][c]["summary"]["drift_ms_per_min"] or 0.0)
                                                           for fn in HEADLINE_BANDS)), 1)} for c in CAMS}
        # beat_this on the heat audio against the pulse lattice
        for c in CAMS:
            bt = clk_cam(cfg, c, trk[c]["beats_refined"])
            bt = bt[(bt >= a) & (bt <= b)]
            dd = -nearest_offsets_signed(bt, pr)  # beat minus nearest pulse
            m = np.abs(dd) <= 0.33 * P_ref
            if m.sum() >= 8:
                d = dd[m]
                trk_cmp.setdefault(s, {})[c] = {
                    "n_heat_beats": int(len(bt)), "n_within_third_period": int(m.sum()),
                    "median_ms": round(1000 * float(np.median(d)), 1),
                    "mad_ms": round(1000 * float(1.4826 * np.median(np.abs(d - np.median(d)))), 1),
                    "p10_p90_ms": [round(1000 * float(np.percentile(d, 10)), 1), round(1000 * float(np.percentile(d, 90)), 1)],
                    "drift_ms_per_min": drift_fit(bt[m], 1000 * d)["drift_ms_per_min"]}
    # camera 01 vs camera 02, band by band (12 s windows over the solo parts of both songs)
    pair = {}
    for fn in FEATURES:
        sh = []
        for s in SONGS:
            a, b = agree[s]["span_s"]
            for t0 in np.arange(a, b - 12.0 + 1e-9, 6.0):
                tt = np.arange(t0, t0 + 12.0, dts)
                f = [np.interp(tt, heat[c]["t0"] + np.arange(len(feature(heat[c], fn))) * heat[c]["dt"], feature(heat[c], fn))
                     for c in CAMS]
                d, r = _xcorr_shift(f[0], f[1], dts, 0.1)
                if r > 0.15:
                    sh.append(-1000 * d)  # cam 02 minus cam 01 (positive: cam 02's onsets come later)
        pair[fn] = {"n": len(sh), "cam02_minus_cam01_ms_median": round(float(np.median(sh)), 1) if sh else None,
                    "mad_ms": round(float(1.4826 * np.median(np.abs(np.asarray(sh) - np.median(sh)))), 1) if sh else None}
    out.update(agreement=agree, headline=head, camera_pair=pair, heat_beat_this=trk_cmp)
    return out


def nearest_offsets_signed(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """For each x the signed distance to the nearest ref (sorted): ref - x."""
    ref = np.sort(np.asarray(ref, float))
    j = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1)
    return np.where(np.abs(ref[j - 1] - x) < np.abs(ref[j] - x), ref[j - 1] - x, ref[j] - x)


def clk_cam(cfg: dict, cam: str, t_src) -> np.ndarray:
    sl, off = camera_clock(cfg["take"], cam)
    return sl * np.asarray(t_src, float) + off


# ------------------------------------------------------------------------------------------------ merged table
def merged_rows(cfg: dict, grids: dict, heat: dict, studio: dict, val: dict, pres: dict, gov: np.ndarray, gov_song: np.ndarray,
                overlap: tuple[float, float]) -> dict:
    """One row per pulse and eighth of both grids, with the confidence parts."""
    zt = np.asarray(pres["t"])
    rows = []
    for si, s in enumerate(SONGS):
        g = grids[s]
        clk: Clock = cfg["songs"][s]["clock"]
        R = g["rows_ref"]
        if not len(R):
            continue
        # pulses (even slots): tracker probability from the pulse list; eighths inherit the lower neighbour
        kpos = R[:, 4]
        k_idx = {int(k): i for i, k in enumerate(g["k"])}
        kk = np.floor(kpos).astype(int)
        p0 = np.array([g["p_track"][k_idx[k]] for k in kk])
        p1 = np.array([g["p_track"][k_idx.get(k + 1, k_idx[k])] for k in kk])
        trk_p = np.where(R[:, 3] % 2 == 0, p0, np.minimum(p0, p1))
        interp = np.array([not g["tracked"][k_idx[k]] for k in kk])
        # onset evidence of the heat: the mean of the three bands at each beat, each corrected by the lag at which this
        # song's onsets sit in this band of this camera (folded profile peak within +-50 ms), mean over the two cameras
        sup = []
        for c in CAMS:
            h = heat[c]
            per_band = []
            for fn in BANDS:
                ap = val["agreement"][s][fn][c]["all_pulses"] or {}
                per_band.append(heat_support(feature(h, fn), h["dt"], h["t0"], R[:, 0], (ap.get("onset_lag_ms") or 0.0) / 1000.0))
            sup.append(np.mean(per_band, 0))
        sup = np.mean(sup, 0)
        # the same evidence in the clean studio recording (song time): does the studio itself have an onset at this beat?
        ssup = np.mean([heat_support(feature(studio[s], fn), S.frame_dt(SR), 0.0, clk.to_song(R[:, 0]),
                                     ((val["agreement"][s][fn]["01"]["all_pulses"] or {}).get("studio_onset_lag_ms") or 0.0)
                                     * clk.slope / 1000.0) for fn in BANDS], 0)
        zv = np.asarray(pres["z"][s], float)
        ok = np.isfinite(zv)
        z = np.interp(R[:, 0], zt[ok], zv[ok]) if ok.any() else np.zeros(len(R))  # clamps to the nearest valid window at the edges
        conf = beat_confidence(trk_p, z, interp)
        lo, hi = overlap
        in_ov = (R[:, 0] >= lo) & (R[:, 0] <= hi)
        gk = {int(round(float(t) * 1e6)) for t in gov[gov_song == si][:, 0]}
        for i in range(len(R)):
            rows.append([round(float(R[i, 0]), 6), int(R[i, 1]), int(R[i, 2]), int(R[i, 3]), s,
                         int(round(float(R[i, 0]) * 1e6)) in gk, bool(in_ov[i]),
                         float(conf[i]), round(float(trk_p[i]), 3), round(float(sup[i]), 3), round(float(ssup[i]), 3), round(float(z[i]), 2),
                         bool(interp[i]), round(float(R[i, 4]), 1)])
    rows.sort(key=lambda r: (r[0], r[4]))
    return {"columns": ["t", "type", "measure_in_song", "slot", "song", "governing", "overlap", "confidence",
                        "p_tracker", "heat_support", "studio_support", "presence_z", "interpolated", "pulse_position"],
            "rows": rows}


def onset_support(merged: dict) -> dict:
    """Per-beat onset evidence of the heat audio at the audible pulses (even slots, presence z >= PRESENT_Z, not
    interpolated). The music is syncopated, so most pulses carry no onset even in the clean studio recording: the fair
    test is the pulses WHERE THE STUDIO HAS AN ONSET (studio_support >= 0.5): the share of those at which the heat's
    audio also shows one (heat_support >= 0.5), against the share at the pulses where the studio has none (false hits)."""
    c = merged["columns"]
    ih, ist, iz, iint, isl, isg = (c.index(n) for n in ("heat_support", "studio_support", "presence_z", "interpolated", "slot", "song"))
    out = {}
    for s in SONGS:
        rows = [r for r in merged["rows"] if r[isg] == s and r[isl] % 2 == 0 and r[iz] >= PRESENT_Z and not r[iint]]
        on = np.asarray([r[ih] for r in rows if r[ist] >= 0.5])
        off = np.asarray([r[ih] for r in rows if r[ist] < 0.5])
        out[s] = {"n_pulses": len(rows), "n_studio_onsets": int(len(on)),
                  "heat_onset_at_studio_onsets": round(float(np.mean(on >= 0.5)), 3) if len(on) else None,
                  "heat_onset_at_pulses_without_studio_onset": round(float(np.mean(off >= 0.5)), 3) if len(off) else None}
    return out


# ------------------------------------------------------------------------------------------------ outputs
def heat_tempo(cfg: dict, grids: dict, trk: dict, win_s: float = 20.0) -> dict:
    """Tempo measured on the heat audio itself (beat_this on each video, intervals within 15 % of the lattice period, so
    the tracker's level switches are ignored): median BPM over the solo part of each song and in 20 s windows, per camera.
    The check that the DJ's playback tempo is the one the clock says (lattice BPM = studio BPM x slope) and is steady."""
    out = {}
    for s in SONGS:
        lo, hi = cfg["songs"][s]["audible"]
        other = cfg["songs"]["guetta" if s == "kojo" else "kojo"]["audible"]
        a, b = (max(lo, 14.0) + 1.0, other[0] - 0.5) if s == "kojo" else (other[1] + 0.5, hi - 1.0)
        P = grids[s]["model"]["period"] / cfg["songs"][s]["clock"].slope
        out[s] = {"lattice_bpm": round(60.0 / P, 3)}
        for c in CAMS:
            bt = clk_cam(cfg, c, trk[c]["beats_refined"])
            bt = bt[(bt >= a) & (bt <= b)]
            d = np.diff(bt)
            tm = 0.5 * (bt[1:] + bt[:-1])
            ok = np.abs(d / P - 1) < 0.15
            wins = []
            for w0 in np.arange(a, b - win_s + 1e-9, win_s):
                m = ok & (tm >= w0) & (tm < w0 + win_s)
                if m.sum() >= 6:
                    wins.append({"t": round(float(w0 + 0.5 * win_s), 1), "bpm": round(60.0 / float(np.median(d[m])), 2), "n": int(m.sum())})
            out[s][c] = {"bpm_median": round(60.0 / float(np.median(d[ok])), 3) if ok.sum() >= 6 else None,
                         "mean_interval_bpm": round(60.0 / float(np.mean(d[ok])), 3) if ok.sum() >= 6 else None,
                         "n": int(ok.sum()), "windows": wins}
    return out


def tempo_curve(cfg: dict, grids: dict, seg: list[dict], step_s: float = 1.0) -> list[dict]:
    out = []
    for t in np.arange(0.0, cfg["duration_s"], step_s):
        row = {"t": float(t), "state": state_at(seg, float(t))}
        for s in SONGS:
            lo, hi = cfg["songs"][s]["audible"]
            clk: Clock = cfg["songs"][s]["clock"]
            if lo <= t <= hi:
                row[f"bpm_{s}"] = round(60.0 / grids[s]["model"]["period"] * clk.tempo_ratio, 3)
        out.append(row)
    return out


def write_outputs(cfg, grids, gov, gov_song, gov_meas, merged, seg, overlap, sw, switch, follow, pres, val, trk, dev, dt) -> dict:
    sd = song_dir()
    typ = lambda R: [[round(float(t), 6), int(k)] for t, k in R[:, :2]]  # noqa: E731
    write_json(sd / "beats_studio.json", typ(gov), indent=None)
    for s in SONGS:
        write_json(sd / f"beats_studio_{s}.json", typ(grids[s]["rows_ref"]), indent=None)
    write_json(sd / "beats_merged.json", merged, indent=None)
    down = [{"t": r[0], "song": r[4], "measure_in_song": r[2], "governing": r[5], "overlap": r[6], "confidence": r[7]}
            for r in merged["rows"] if r[3] == 0]
    write_json(sd / "downbeats.json", down, indent=None)
    tempo = {"version": VERSION, "direction": "ratio = heat tempo / studio tempo; < 1: the heat plays the studio file SLOWER "
             "(pitch kept). sync.json's 1.040 / 1.061 are the duration stretch 1 / ratio.", "songs": {}}
    for s in SONGS:
        g, clk = grids[s], cfg["songs"][s]["clock"]
        P = g["model"]["period"]
        tempo["songs"][s] = {
            "title": cfg["songs"][s]["title"], "artist": cfg["songs"][s]["artist"],
            "studio": {"pulse_s": round(P, 6), "bpm": round(60 / P, 4), "measure_s": round(4 * P, 5),
                       "fit_rms_ms": g["model"]["rms_ms"], "fit_n": g["model"]["n_support"], "level": g["model"]["level"],
                       "max_window_drift_ms": g["model"]["max_window_drift_ms"], "constant_tempo": g["model"]["constant"]},
            "heat": {"pulse_s": round(P / clk.slope, 6), "bpm": round(60 / P * clk.tempo_ratio, 4),
                     "measure_s": round(4 * P / clk.slope, 5)},
            "clock": clk.as_dict(), "audible_reference_s": cfg["songs"][s]["audible"],
            "bar_phase": g["decision"], "beat_placement_studio": g["placement"]}
    tempo["measured_on_heat_audio"] = heat_tempo(cfg, grids, trk)
    tempo["curve"] = tempo_curve(cfg, grids, seg)
    write_json(sd / "tempo.json", tempo)
    segments = {"version": VERSION, "timeline": seg, "mix": list(overlap),
                "no_beat": [[0.0, cfg["songs"]["kojo"]["audible"][0]], [cfg["songs"]["guetta"]["audible"][1], cfg["duration_s"]]],
                "no_beat_note": "0-13.31 s: announcement / entrance sound, no song; after 194.25 s: applause. No beat is "
                                "written there. The danced span (3.6-198.0 s) is no-beat for 9.7 s at the start and 3.75 s at the end.",
                "governing": {"switch_s": switch, "decided_by": sw["by"], "audio_default": sw,
                              "dancers": follow if follow is not None else "not run: no step times yet (run --steps / follow)",
                              "kojo_until_s": float(gov[gov_song == 0][-1, 0]) if (gov_song == 0).any() else None,
                              "guetta_from_s": float(gov[gov_song == 1][0, 0]) if (gov_song == 1).any() else None},
                "danced_span": cfg["danced_span"]}
    write_json(sd / "segments.json", segments)
    write_json(sd / "presence.json", pres, indent=None)
    write_json(sd / "validation.json", val)
    heat_tr = {c: {"source": cfg["take"].source(c).path.name, "beats_reference_s": _r(clk_cam(cfg, c, trk[c]["beats_refined"]), 4),
                   "downbeats_reference_s": _r(clk_cam(cfg, c, trk[c]["downbeats_refined"]), 4)} for c in CAMS}
    write_json(sd / "heat_beats_beat_this.json", heat_tr, indent=None)
    raw = {"version": VERSION, "take": TAKE, "module": "dancecap.carlos_aline_beats",
           "provenance": {"songs": {s: {"file": S.rel_to_workspace(cfg["files"][s]), "sha256_16": S.sha256_16(cfg["files"][s]),
                                         "clock": cfg["songs"][s]["clock"].as_dict()} for s in SONGS},
                          "beat_this": {"checkpoint": "final0", "device": dev, "fps": B.BEAT_THIS_FPS,
                                        "input": f"mono {SR} Hz, whole song and both heat recordings"},
                          "grid": "fit_pulses (straight-line pulse lattice from mixed-level tracked beats), downbeat_votes, "
                                  "kick/backbeat rotation test, zouk_rows (eighths = pulse midpoints); song time mapped "
                                  "through the affine song clock; whole measures inside the song's audible span"},
          "songs": {s: {"pulse_fit": {k: v for k, v in grids[s]["model"].items() if k not in ("k", "t", "resid_ms")},
                        "bar_phase": grids[s]["decision"], "beat_placement": grids[s]["placement"],
                        "rows_reference": [[round(float(a), 6), int(b), int(c), int(d), float(e)]
                                           for a, b, c, d, e in grids[s]["rows_ref"]],
                        "pulses_reference": pulses_table(cfg, grids[s], s),
                        "beat_this_song_time": {"beats": _r(trk[s]["beats_refined"]), "downbeats": _r(trk[s]["downbeats_refined"])}}
                    for s in SONGS},
          "grid_columns": ["t", "type", "measure", "slot", "pulse_position"]}
    write_json(sd / "beats_studio_raw.json", raw, indent=None)
    song = {"version": VERSION, "take": TAKE, "reference": "camera 01 video time (alignment_carlos_aline.json)",
            "songs": {s: {"title": cfg["songs"][s]["title"], "artist": cfg["songs"][s]["artist"],
                          "clock": cfg["songs"][s]["clock"].as_dict(), "audible_reference_s": cfg["songs"][s]["audible"],
                          "song_zero_at_reference_s": cfg["songs"][s]["song_zero_at_reference_s"],
                          "clock_residual_ms": cfg["songs"][s]["residual_ms"]} for s in SONGS},
            "timeline": seg, "governing_switch_s": switch, "files": sorted(p.name for p in sd.glob("*.json")),
            "copyright": S.COPYRIGHT}
    write_json(sd / "song.json", song)
    prov = {"what": "beat grid of the two-song heat, derived from the studio mp3s (beat times only, no audio) and checked on the heat audio",
            "measure numbers": "beats_studio.json is time-ordered whole measures: Kojo Funds measures first, Guetta's after; "
                               "index // 8 is the measure",
            "module": "dancecap.carlos_aline_beats", "reference_clock": "camera 01 video time"}
    write_json(sd / "beats_studio.provenance.json", prov)
    return {"governing_measures": int(len(gov) // 8), "switch_s": switch, "files": song["files"]}


def pulses_table(cfg: dict, g: dict, s: str) -> dict:
    """Every pulse of the song inside its audible span (not only those of whole measures): the dancers' step analysis uses
    these (a trailing partial measure, the pulses around the mix). columns: t (reference s), k (pulse index of the
    lattice), bar_pos (0 = bar start under the chosen phase), p_tracker, tracked (a beat_this beat supports it)."""
    clk: Clock = cfg["songs"][s]["clock"]
    lo, hi = cfg["songs"][s]["audible"]
    t = g["pulse_ref"]
    m = (t >= lo) & (t <= hi)
    return {"columns": ["t", "k", "bar_pos", "p_tracker", "tracked"],
            "rows": [[round(float(t[i]), 6), int(g["k"][i]), int((g["k"][i] - g["phase"]) % 4), round(float(g["p_track"][i]), 3),
                      bool(g["tracked"][i])] for i in np.where(m)[0]]}


def clicks_overlay(cfg: dict, audio: dict, gov: np.ndarray) -> None:
    """Camera 01's own sound with clicks on the governing grid (types 1/2/3), 16-bit wav under work/ (git-ignored)."""
    x = audio["01"]
    clicks = B.click_track(len(x), SR, gov[:, :2], 0.0)
    mix = 0.7 * x / max(1e-9, float(np.abs(x).max())) + clicks
    S.write_private_wav(song_dir() / "heat_clicks_overlay.wav", mix / max(1.0, float(np.abs(mix).max()) / 0.98), SR)


# ------------------------------------------------------------------------------------------------ plot
def review_plot(cfg, grids, heat, studio_G, gov, merged, seg, overlap, switch, pres, val) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"kojo": "tab:blue", "guetta": "tab:red"}
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(6, 3, height_ratios=[1.3, 1.0, 1.0, 1.0, 1.0, 1.1], hspace=0.55, wspace=0.18)
    t01 = heat["01"]["low_t"]
    Gs = np.convolve(heat["01"]["low"], np.ones(5) / 5, "same")
    rows = merged["rows"]

    def ticks(ax, lo, hi, ymax, lw=0.6):
        """Grid ticks in a strip above the signal: downbeat full height, pulse 55 %, eighth 25 %; solid: governing."""
        for r in rows:
            if lo <= r[0] <= hi:
                big, half = r[3] == 0, r[3] % 2 == 1
                top = ymax * (1.32 if big else 1.19 if not half else 1.09)
                ax.vlines(r[0], ymax * 1.02, top, color=col[r[4]], lw=2.0 if big else 1.0 if not half else 0.6,
                          alpha=0.95 if r[5] else 0.5, ls="-" if r[5] else ":")
                if not half:
                    ax.vlines(r[0], 0, ymax, color=col[r[4]], lw=0.5, alpha=0.35 if r[5] else 0.15)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(t01, Gs, color="0.25", lw=0.5)
    ymax = float(np.percentile(Gs, 99.5))
    ax.set_ylim(0, ymax * 1.35)
    ticks(ax, 0, cfg["duration_s"], ymax)
    for s in seg:
        c = {"none": "0.8", "kojo": "none", "guetta": "none", "mix": "gold"}[s["state"]]
        if s["state"] in ("none", "mix"):
            ax.axvspan(s["from_s"], s["to_s"], color=c, alpha=0.3, lw=0)
    ax.axvline(switch, color="k", lw=1.2, ls="--")
    ax.axvspan(*cfg["danced_span"], ymin=0.97, ymax=1.0, color="green")
    ax.set_xlim(0, cfg["duration_s"])
    ax.set_title("carlos_aline: governing grid (solid) and the other song's grid (dotted) over camera 01's bass onset flux; blue Kojo Funds, red Guetta; "
                 "grey: no song; gold: the mix; dashed: governing switch; tall = downbeat; green bar: danced span", fontsize=9)
    ax.set_xlabel("reference time (s)")
    # three zooms
    zooms = [(30.0, 42.0, "Kojo Funds only"), (112.0, 126.0, "the DJ mix (Kojo -> Guetta)"), (150.0, 162.0, "Guetta only")]
    for zi, (lo, hi, name) in enumerate(zooms):
        for ri, cam in enumerate(CAMS):
            ax = fig.add_subplot(gs[1 + ri, zi])
            h = heat[cam]
            tt = h["low_t"]
            m = (tt >= lo) & (tt <= hi)
            ax.plot(tt[m], h["low"][m], color="0.25", lw=0.8)
            yl = float(np.percentile(h["low"][m], 99.5)) if m.any() else 1.0
            ax.set_ylim(0, yl * 1.35)
            ticks(ax, lo, hi, yl, lw=1.0)
            ax.set_xlim(lo, hi)
            if ri == 0:
                ax.set_title(f"{name} {lo:g}-{hi:g} s", fontsize=9)
            ax.set_ylabel(f"cam {cam} bass flux", fontsize=8)
    # presence + switch
    ax = fig.add_subplot(gs[3, :2])
    zt = np.asarray(pres["t"])
    for s in SONGS:
        ax.plot(zt, np.asarray(pres["z"][s], float), color=col[s], label=f"{s} z")
    ax.axhline(PRESENT_Z, color="0.5", lw=0.7, ls=":")
    ax.axvspan(*overlap, color="gold", alpha=0.3, lw=0)
    ax.axvline(switch, color="k", ls="--", lw=1.0)
    ax.set_title("how audible each song is in the heat (onset correlation z, 3 s windows, mean of the 2 cameras)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(0, cfg["duration_s"])
    ax = fig.add_subplot(gs[3, 2])
    for s in SONGS:
        for fn, mk in zip(HEADLINE_BANDS, "os"):
            for c in CAMS:
                w = val["agreement"][s][fn][c]["windows"]
                ax.plot([x["t"] for x in w], [x["agreement_ms"] for x in w], mk, color=col[s], ms=4,
                        mfc="none" if c == "02" else col[s])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-60, 30)
    ax.set_title("heat - studio onset lag at the pulses (ms), 24 s windows; mid (o) and hi (s) bands;\nfilled: cam 01, open: cam 02", fontsize=8)
    ax.set_xlabel("reference time (s)")
    # folded profiles
    for si, s in enumerate(SONGS):
        ax = fig.add_subplot(gs[4, si])
        clk: Clock = cfg["songs"][s]["clock"]
        P_ref = grids[s]["model"]["period"] / clk.slope
        half = min(0.34, 0.5 * P_ref)
        pr = grids[s]["pulse_ref"]
        lo, hi = ((15.5, 111.0) if s == "kojo" else (123.0, 192.0))
        p = pr[(pr >= lo) & (pr <= hi)]
        lags = np.arange(-half, half, 0.0025)
        for cam, ls in (("01", "-"), ("02", "--")):
            h = heat[cam]
            tt = h["t0"] + np.arange(len(h["G"])) * h["dt"]
            prof = np.mean([np.interp(q + lags, tt, h["G"]) for q in p if q - half > tt[0] and q + half < tt[-1]], 0)
            ax.plot(1000 * lags, prof / prof.max(), color=col[s], ls=ls, label=f"heat cam {cam}")
        ps = clk.to_song(p)
        dts = S.frame_dt(SR)
        tts = np.arange(len(studio_G[s])) * dts
        sl = lags * clk.slope
        prof = np.mean([np.interp(q + sl, tts, studio_G[s]) for q in ps if q + sl[0] > 0 and q + sl[-1] < tts[-1]], 0)
        ax.plot(1000 * lags, prof / prof.max(), color="k", lw=1.0, label="studio (clean)")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{s}: onset flux folded on all solo pulses (peak = where the audible onsets sit)", fontsize=8)
        ax.set_xlabel("lag from the pulse (ms)")
        ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[4, 2])
    tc = tempo_curve(cfg, grids, seg)
    for s in SONGS:
        xs = [r["t"] for r in tc if f"bpm_{s}" in r]
        ax.plot(xs, [r[f"bpm_{s}"] for r in tc if f"bpm_{s}" in r], color=col[s], lw=2, label=f"{s} (heat)")
        ax.axhline(60.0 / grids[s]["model"]["period"], color=col[s], lw=0.6, ls=":")
    # heat tracker local tempo, folded to the pulse level
    for cam, mk in zip(CAMS, "+x"):
        b = clk_cam(cfg, cam, np.load(song_dir() / "_cache" / f"beat_this_{cam}.npz")["beats_refined"])
        for s in SONGS:
            lo, hi = cfg["songs"][s]["audible"]
            P = grids[s]["model"]["period"] / cfg["songs"][s]["clock"].slope
            bb = b[(b >= lo + 1) & (b <= hi - 1)]
            d = np.diff(bb)
            t = 0.5 * (bb[1:] + bb[:-1])
            m = (np.abs(d / P - 1) < 0.2)
            ax.plot(t[m], 60 / d[m], mk, color=col[s], ms=3, alpha=0.4)
    ax.set_ylim(70, 110)
    ax.set_title("tempo (BPM) in the heat; dotted: studio; marks: beat_this on the heat audio", fontsize=8)
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[5, :])
    cs = np.array([r[7] for r in rows if r[3] % 2 == 0])
    ct = np.array([r[0] for r in rows if r[3] % 2 == 0])
    cc = np.array([col[r[4]] for r in rows if r[3] % 2 == 0])
    ax.scatter(ct, cs, c=cc, s=8, alpha=0.7)
    ax.axvspan(*overlap, color="gold", alpha=0.3, lw=0)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, cfg["duration_s"])
    ax.set_title("confidence of each pulse (beat_this probability x audibility of the song in the heat x 0.7 if interpolated)", fontsize=9)
    ax.set_xlabel("reference time (s)")
    rev = work_dir() / "review"
    rev.mkdir(exist_ok=True)
    p = rev / "carlos_aline_beats.png"
    fig.savefig(p, dpi=85, bbox_inches="tight")
    plt.close(fig)
    return [p]


# ------------------------------------------------------------------------------------------------ cli
def _phase_args(v: list[str] | None) -> dict:
    out = {}
    for a in v or []:
        k, _, n = a.partition("=")
        if k not in SONGS or not n.isdigit():
            raise SystemExit(f"--phase wants kojo=N or guetta=N, got {a!r}")
        out[k] = int(n)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="beat grids, mix, validation, review plot")
    r.add_argument("--device", choices=["cpu", "cuda"], default=None)
    r.add_argument("--phase", action="append", help="force a bar phase: kojo=N / guetta=N (N = pulse index mod 4)")
    r.add_argument("--steps", type=Path, default=None, help="timing.json or a step list: the dancers pick the governing song in the mix")
    r.add_argument("--no-plot", action="store_true")
    r.add_argument("--fresh", action="store_true", help="recompute the audibility curves (cached in song/_cache/presence.json)")
    f = sub.add_parser("follow", help="step phase lock against each song's pulses (needs step times)")
    f.add_argument("--steps", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        res = run(a.device, _phase_args(a.phase), a.steps, not a.no_plot, a.fresh)
        print(json.dumps({k: v for k, v in res.items() if k != "files"}, default=_jsonable))
    else:
        cfg = load_config()
        sd = song_dir()
        raw = json.loads((sd / "beats_studio_raw.json").read_text(encoding="utf-8"))
        pulses = {}
        for s in SONGS:
            pulses[s] = np.asarray([r[0] for r in raw["songs"][s]["pulses_reference"]["rows"]], float)
        steps = load_steps(a.steps)
        ov = (cfg["songs"]["guetta"]["audible"][0], cfg["songs"]["kojo"]["audible"][1])
        res = {"mix": follow_choice(steps, pulses, ov),
               "curve": follow_curve(steps, pulses, cfg["songs"]["kojo"]["audible"][0], cfg["songs"]["guetta"]["audible"][1])}
        write_json(sd / "follow.json", res)
        print(json.dumps(res["mix"], default=_jsonable))


if __name__ == "__main__":
    main()
