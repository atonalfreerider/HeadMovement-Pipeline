"""Rhythm analysis of a heat or a take with a beat grid and step events, and a causal LIVE SCORE (agent S6, 2026-10-11).

    python -m dancecap.rhythm run <take> [--timing P] [--song-dir P] [--audio WAV] [--out DIR] [--no-plots] [--no-sensitivity]
    python -m dancecap.rhythm install <take> --capture <NN_Name>   # the viewer's copy into the capture folder(s), nothing else touched
    python -m dancecap.rhythm selftest                 # the synthetic checks (perfect steps -> near the top, random steps -> near the bottom)

What it is. For every footstep (touchdown) of the lead and the follow it finds the phase inside the beat grid (on a pulse, on the "and", between),
the signed offset in ms to the nearest grid slot, and scores five things that a Jack & Jill judge listens for: BEAT LOCK (steps on the beat), ACCENT
HITS (steps and stops on the music's accents; holds on its breaks), TEMPO ADHERENCE (steps a whole number of eighths apart), FLOW (a steady phase
against the grid) and COUPLE SYNCHRONY (lead and follow touch down together). Each component is scored against what CHANCE would give (random times
score 0), 0..1, and the live score is 100 x their weighted sum (docs/RHYTHM.md has the formula, the weights and the constants):

    score = 100 * (0.40 lock + 0.20 accents + 0.15 tempo + 0.15 flow + 0.10 synchrony)

The LIVE score at time t uses only data up to t (events become known when they have happened): a running score = 0.5 x the whole dance so far +
0.5 x the last ~10 s, starting from a neutral 50 (a prior worth a few steps) and FROZEN in the no-beat stretches (no song, the DJ seam). The FINAL score is
the whole-dance aggregate. It is an ESTIMATE FROM MOTION, not an official score, and it says how sure it is: confidence combines the beat grid's
own confidence, the no-beat and seam stretches, the +-30 ms audio-video uncertainty of a take without a sync event, the number of steps seen and
the evidence that the steps lock to the grid at all (a Rayleigh test: when they do not, the lock components measure noise and the confidence drops).

Compute: numpy + scipy, under a second for a 200 s heat (the STFT of the soundtrack is the longest part, about 3 s): run locally (the pods rule is for
CV / neural / heavy CPU work).

Outputs (work/<take>/rhythm/): rhythm.json (steps, measures, windows, 30 Hz live timeline, final card, sensitivity, claims), rhythm_review.png (+ more
plots), and the Unity capture gets a copy (export_unity: capture.json "rhythm") for the viewer's rhythm layer. Inputs: timing/timing.json (footsteps + beat grid), optionally
timing/kinematics.npz (stops, holds), song/ (confidence, no-beat intervals, DJ seam variants) and a soundtrack WAV on the reference clock (accents, breaks).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

VERSION = 1
TITLE = "Rhythm score (estimated from motion, not an official score)"
ROLES = ("lead", "follow")
COMPONENTS = ("lock", "accents", "tempo", "flow", "sync")


# --------------------------------------------------------------------------------------------------------------- configuration
@dataclass
class Config:
    # step classes against the nearest slot (ms): on <= on_ms, near <= near_ms, else off
    on_ms: float = 60.0
    near_ms: float = 120.0
    # the lock kernel: credit = exp(-d^2 / (2 sigma^2)); ideal = what perfect stepping scores with the timing noise (A/V +-30 ms and ~15 ms measurement)
    sigma_ms: float = 60.0
    ideal_jitter_ms: float = 30.0
    ideal_flow_R: float = 0.88          # mean resultant length of a perfect dancer (30 ms rms about the slot on a 371 ms eighth)
    flow_offset_sigma_ms: float = 90.0  # a steady phase counts fully when its mean offset from the grid is within the audio-video uncertainty, less beyond
    accent_ideal: float = 0.75          # mean accent credit of a very good dancer (no dancer hits only the strongest accents)
    sync_ideal: float = 0.80            # share of touchdowns with the partner's touchdown within the kernel for a very synchronised couple
    # component weights (sum 1)
    weights: dict = field(default_factory=lambda: {"lock": 0.40, "accents": 0.20, "tempo": 0.15, "flow": 0.15, "sync": 0.10})
    # accents: onset events of at least this strength (1 = the local 90th percentile of onset peaks) count; credit kernel sigma (ms)
    accent_min_strength: float = 0.5
    accent_sigma_ms: float = 50.0
    accent_chance_window_s: float = 4.0
    hold_weight: float = 0.25           # share of the accents component that the holds on musical breaks take once a break has been seen
    # breaks in the music (dB below the local median level, minimum length) and the motion that counts as still
    break_db: float = 6.0
    break_min_s: float = 0.25
    stop_ratio: float = 0.4             # a stop = the body speed under stop_ratio x the dancer's median speed for stop_min_s
    stop_min_s: float = 0.3
    # memory of the live score
    half_life_s: float = 10.0           # the "recent" part
    recent_share: float = 0.5           # live = (1 - share) x whole dance so far + share x recent
    prior_events: float = 4.0           # the neutral prior (0.5 per component) is worth this many events
    neutral: float = 0.5
    sync_latency_s: float = 0.15        # a synchrony event is known this long after the later step
    sync_sigma_ms: float = 60.0
    # confidence
    av_pinned: bool = False             # an audio-video sync event pins the camera clocks to the soundtrack (timing.json absolute_timing.valid)
    av_uncertainty_ms: float = 30.0
    av_cap_unpinned: float = 0.85
    grid_confidence_default: float = 0.8
    min_events_for_confidence: float = 8.0
    # outputs
    live_hz: float = 30.0
    window_s: float = 12.0
    window_step_s: float = 2.0
    final_card_s: float = 4.0

    def lock_ideal(self) -> float:
        return self.sigma_ms / math.sqrt(self.sigma_ms ** 2 + self.ideal_jitter_ms ** 2)


def lock_chance(sigma_ms: float, p8_ms: float) -> float:
    """expected credit exp(-d^2/2s^2) for d uniform on one slot spacing (what random steps score)"""
    a = p8_ms / 2.0
    return float(sigma_ms * math.sqrt(2.0 * math.pi) * math.erf(a / (sigma_ms * math.sqrt(2.0))) / p8_ms)


# --------------------------------------------------------------------------------------------------------------- data classes
@dataclass
class Grid:
    """the beat grid on the reference clock: one row per slot (eighth), slot 0..7 in a measure, even slots = pulses (the quarter notes)"""
    t: np.ndarray
    slot: np.ndarray
    type: np.ndarray
    measure: np.ndarray
    conf: np.ndarray
    song: list
    no_beat: list                      # [(a, b, kind)] kind "none" (no song) or "seam" (a gap inside the grid: the DJ mix)
    seam_note: str = ""

    def __len__(self):
        return len(self.t)

    @property
    def p8(self) -> float:
        d = np.diff(self.t)
        d = d[(d > 0.05) & (d < 1.0)]
        return float(np.median(d)) if len(d) else 0.37

    def state_at(self, t: float) -> int:
        """0 beat, 1 no song, 2 DJ seam (frozen)"""
        for a, b, kind in self.no_beat:
            if a <= t < b:
                return 1 if kind == "none" else 2
        return 0

    def shifted(self, dt: float) -> "Grid":
        g = Grid(self.t + dt, self.slot, self.type, self.measure, self.conf, self.song, [(a + dt, b + dt, k) for a, b, k in self.no_beat], self.seam_note)
        return g


@dataclass
class Music:
    """what the soundtrack says: accent events (t, strength 0..1), breaks [(a, b)], and a 10 Hz level curve"""
    acc_t: np.ndarray
    acc_s: np.ndarray
    breaks: list
    level_t: np.ndarray
    level_db: np.ndarray
    source: str                         # "audio" | "grid" (downbeats only)


@dataclass
class Motion:
    """body speed per role at the capture frame times (m/s, mean of pelvis / head / wrists / feet), for stops and holds"""
    t: np.ndarray
    speed: dict


@dataclass
class Inputs:
    grid: Grid
    steps: dict                         # role -> list of {"t", "foot", "sigma"(s)}
    music: Music | None
    motion: Motion | None
    t0: float
    t1: float
    meta: dict = field(default_factory=dict)
    variants: dict = field(default_factory=dict)      # name -> Grid (DJ seam handled another way), for the sensitivity analysis


# --------------------------------------------------------------------------------------------------------------- loaders
def _merge_intervals(iv):
    out = []
    for a, b, k in sorted(iv, key=lambda x: x[0]):
        if out and a <= out[-1][1] + 1e-9 and k == out[-1][2]:
            out[-1] = (out[-1][0], max(out[-1][1], b), k)
        else:
            out.append((a, b, k))
    return out


def build_grid(rows, no_beat_none=(), explicit_gaps=(), conf=None, song=None, default_conf: float = 0.8) -> Grid:
    """rows [[t, type, measure, slot], ...] (a bare [[t, type]] list is numbered 8 per measure). Gaps in the grid larger than 1.6 slots, and
    `explicit_gaps` (the DJ seam), become "seam" intervals; `no_beat_none` (no song) "none". The grid's own ends are "none" too."""
    rows = np.asarray(rows, dtype=float)
    order = np.argsort(rows[:, 0])
    rows = rows[order]
    t = rows[:, 0]
    typ = rows[:, 1].astype(int)
    if rows.shape[1] >= 4:
        measure, slot = rows[:, 2].astype(int), rows[:, 3].astype(int)
    else:
        idx = np.arange(len(t))
        measure, slot = idx // 8, idx % 8
    c = np.full(len(t), default_conf) if conf is None else np.asarray(conf, float)
    sg = list(song) if song is not None else [""] * len(t)
    p8 = float(np.median(np.diff(t))) if len(t) > 2 else 0.37
    iv = [(float(a), float(b), "none") for a, b in no_beat_none]
    for a, b in explicit_gaps:
        iv.append((float(a), float(b), "seam"))
    big = np.where(np.diff(t) > 1.6 * p8)[0]
    for i in big:
        a, b = float(t[i] + p8 / 2), float(t[i + 1] - p8 / 2)
        if not any(x[0] <= a + 0.05 and x[1] >= b - 0.05 for x in iv):
            iv.append((a, b, "seam"))
    iv.append((-1e9, float(t[0] - p8 / 2), "none"))
    iv.append((float(t[-1] + p8 / 2), 1e9, "none"))
    return Grid(t, slot, typ, measure, c, sg, _merge_intervals(iv))


def load_timing(path: Path, cfg: Config) -> tuple[dict, list, list, dict]:
    """timing.json -> ({role: steps}, grid rows, [(a, b)] dropped no-beat intervals, meta)"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = {r: [{"t": float(f["t"]), "foot": f.get("foot", ""), "sigma": float(f.get("sigma") or 0.02)} for f in d.get("footsteps", {}).get(r, [])] for r in ROLES}
    rows = d["beats"]["grid"]
    iv = []
    for r in ROLES:
        iv += [tuple(x) for x in (d.get("dropped_no_beat", {}).get(r, {}) or {}).get("intervals", [])]
    iv = sorted({(round(a, 3), round(b, 3)) for a, b in iv})
    ab = d.get("absolute_timing") or {}
    meta = {"timing": str(path), "av_pinned": bool(ab.get("valid")), "tempo_bpm": d["beats"].get("tempo_bpm"), "grid_rows": len(rows),
            "beats_provenance": d.get("inputs", {}).get("beats")}
    return steps, rows, iv, meta


def load_song_dir(song_dir: Path | None, rows) -> dict:
    """optional song/ files: per-slot confidence (beats_merged.json), the no-song intervals (segments.json), seam variants (beats_studio_raw.json)"""
    out: dict = {"conf": None, "none": [], "variants": {}, "song": None, "timeline": None}
    if song_dir is None or not Path(song_dir).exists():
        return out
    song_dir = Path(song_dir)
    t = np.asarray([r[0] for r in rows], float)
    try:
        m = json.loads((song_dir / "beats_merged.json").read_text(encoding="utf-8"))
        ci = {c: i for i, c in enumerate(m["columns"])}
        gov = [r for r in m["rows"] if r[ci["governing"]]]
        gt = np.asarray([r[ci["t"]] for r in gov], float)
        gc = np.asarray([r[ci["confidence"]] for r in gov], float)
        gs = [r[ci["song"]] for r in gov]
        j = np.searchsorted(gt, t)
        conf = np.zeros(len(t))
        song = [""] * len(t)
        for i, ti in enumerate(t):
            best, bd = -1, 1e9
            for k in (j[i] - 1, j[i]):
                if 0 <= k < len(gt) and abs(gt[k] - ti) < bd:
                    best, bd = k, abs(gt[k] - ti)
            if best >= 0 and bd < 0.02:
                conf[i], song[i] = gc[best], gs[best]
            else:
                conf[i] = np.nan
        out["conf"], out["song"] = conf, song
    except (OSError, ValueError, KeyError):
        pass
    try:
        seg = json.loads((song_dir / "segments.json").read_text(encoding="utf-8"))
        out["none"] = [(float(a), float(b)) for a, b in seg.get("no_beat", [])]
        out["timeline"] = seg.get("timeline")
        out["governing"] = seg.get("governing")
    except (OSError, ValueError):
        pass
    return out


def seam_variants(song_dir: Path | None, base: Grid, cfg: Config) -> dict:
    """The DJ seam handled the other ways (sensitivity): "seam_first": the first song governs through the whole mix to the second song's
    start; "seam_second": the second song governs from the mix's start. Built from song/beats_studio_raw.json (every pulse of each song's
    audible span; eighths = midpoints) and song/segments.json (the mix interval). Empty when the take has no such structure."""
    if song_dir is None:
        return {}
    try:
        raw = json.loads((Path(song_dir) / "beats_studio_raw.json").read_text(encoding="utf-8"))
        seg = json.loads((Path(song_dir) / "segments.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    mix = next((x for x in seg.get("timeline", []) if len(x.get("songs", [])) == 2), None)
    if mix is None:
        return {}
    first, second = mix["songs"]
    pf = np.asarray([r[0] for r in raw["songs"][first]["pulses_reference"]["rows"]], float)
    ps = np.asarray([r[0] for r in raw["songs"][second]["pulses_reference"]["rows"]], float)
    mix_a, mix_b = float(mix["from_s"]), float(mix["to_s"])

    def eighths(p):
        return np.sort(np.r_[p, 0.5 * (p[1:] + p[:-1])])

    out = {}
    p8 = base.p8
    seam = [(a, b) for a, b, k in base.no_beat if k == "seam"]
    if not seam:
        return {}
    a0, b0 = seam[0]
    for name, src, lo, hi in (("seam_first", pf, a0, b0), ("seam_second", ps, mix_a, b0)):
        tt = eighths(src)
        tt = tt[(tt >= lo - 1e-6) & (tt < hi + p8 * 0.5)]
        if name == "seam_second":
            keep = base.t < lo - p8 * 0.5
            t_old = base.t[keep]
        else:
            keep = base.t < lo
            t_old = base.t[keep]
        tt = tt[tt > (t_old[-1] + 0.5 * p8 if len(t_old) else -1e9)]
        t_new = np.r_[t_old, tt]
        after = base.t > hi - 1e-6
        t_new = np.r_[t_new, base.t[after]]
        # re-number slots / types by continuing the measure grid of the base rows around the seam (types are not used by the score)
        n = len(t_new)
        slot = np.arange(n) % 8
        out[name] = Grid(t_new, slot, np.full(n, 3), np.arange(n) // 8, np.full(n, base.conf.mean()), [""] * n,
                         [(a, b, k) for a, b, k in base.no_beat if k == "none"], f"{name}: the {first if name == 'seam_first' else second} song's pulses through the DJ mix")
        # confidence for the filled slots: the song's audibility is unknown here, use the base mean of the song it comes from (the mix is the least certain part)
        c = np.empty(n)
        c[:] = np.nan
        out[name].conf = c
    for name, g in out.items():
        c = np.interp(g.t, base.t, np.nan_to_num(base.conf, nan=cfg.grid_confidence_default))
        in_fill = (g.t > a0 - 1e-6) & (g.t < b0 + 1e-6) | (name == "seam_second") & (g.t >= mix_a) & (g.t < b0)
        c = np.where(in_fill, 0.5 * c, c)
        g.conf = c
    return out


def load_motion(npz: Path | None) -> Motion | None:
    """body speed (mean over pelvis, head, both wrists, both feet of the smoothed velocity norm) per role"""
    if npz is None or not Path(npz).exists():
        return None
    k = np.load(npz)
    sp, tt = {}, None
    for r in ROLES:
        if f"{r}_vel" not in k.files:
            return None
        v = k[f"{r}_vel"]
        j = [0, 15, 20, 21, 10, 11]
        sp[r] = np.linalg.norm(v[:, j], axis=2).mean(1)
        tt = k[f"{r}_t"]
    return Motion(np.asarray(tt, float), sp)


def _read_wav_mono(path: Path):
    from scipy.io import wavfile
    sr, x = wavfile.read(str(path))
    x = x.astype(np.float32)
    if x.dtype.kind == "f":
        pass
    if x.ndim > 1:
        x = x.mean(1)
    peak = float(np.abs(x).max()) or 1.0
    if peak > 4.0:
        x = x / 32768.0 if peak <= 32768.0 else x / peak
    return sr, x


def music_from_audio(path: Path, t0: float, cfg: Config, exclude=()) -> Music:
    """accent events (onset-flux peaks, strength 0..1 against the local 90th percentile) and breaks (level dips) of a soundtrack whose sample 0 is
    reference time t0. `exclude` [(a, b)]: no-song stretches (no accents or breaks there)."""
    from scipy.ndimage import percentile_filter, uniform_filter1d
    from scipy.signal import find_peaks, resample_poly, stft
    sr, x = _read_wav_mono(path)
    if sr >= 32000:
        x = resample_poly(x, 1, 2)
        sr //= 2
    hop = max(1, int(round(sr * 0.01)))
    f, tt, Z = stft(x, sr, nperseg=1024, noverlap=1024 - hop, boundary=None, padded=False)
    S = np.abs(Z)
    edges = np.geomspace(40.0, min(10000.0, sr / 2 - 100), 41)
    B = np.stack([S[(f >= edges[i]) & (f < edges[i + 1])].sum(0) for i in range(40)])
    L = np.log1p(1000.0 * B)
    flux = np.r_[0.0, np.maximum(0.0, np.diff(L, axis=1)).sum(0)]
    fl = np.maximum(0.0, flux - uniform_filter1d(flux, 101))
    fl = uniform_filter1d(fl, 3)
    dt = float(tt[1] - tt[0])
    tref = t0 + tt + 512.0 / sr
    pk, _ = find_peaks(fl, distance=max(1, int(0.12 / dt)), height=float(np.percentile(fl, 75)))
    h = fl[pk]
    # local 90th percentile of the peak heights (+-12 s) as the strength reference
    pt_ = tref[pk]
    lo_i, hi_i = np.searchsorted(pt_, pt_ - 12.0), np.searchsorted(pt_, pt_ + 12.0, side="right")
    ref = np.array([np.percentile(h[a:b], 90) for a, b in zip(lo_i, hi_i)]) if len(pk) else np.array([])
    s = np.clip(h / np.maximum(ref, 1e-9), 0.0, 1.0)
    at = tref[pk]
    keep = np.ones(len(at), bool)
    for a, b in exclude:
        keep &= ~((at >= a) & (at < b))
    # level curve (RMS 50 ms, 10 ms hop) and breaks
    rms = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(hop * 5) / (hop * 5), "same"))[::hop]
    db = 20.0 * np.log10(rms + 1e-6)
    lt = t0 + np.arange(len(db)) * hop / sr
    loc = percentile_filter(db, 50, size=2001)
    dbs = uniform_filter1d(db, 5)
    dip = dbs < loc - cfg.break_db
    breaks, i = [], 0
    while i < len(dip):
        if dip[i]:
            j = i
            while j < len(dip) and dip[j]:
                j += 1
            a, b = float(lt[i]), float(lt[min(j, len(lt) - 1)])
            if b - a >= cfg.break_min_s and not any(e0 <= 0.5 * (a + b) < e1 for e0, e1 in exclude):
                breaks.append((a, b))
            i = j
        else:
            i += 1
    step = 10
    return Music(at[keep], s[keep], breaks, lt[::step], db[::step], "audio")


def music_from_grid(grid: Grid) -> Music:
    """no soundtrack: the grid's accent types stand in (type 1 = 1.0, 2 = 0.7, 3 = 0.35; the bar phase may be off, so downbeat claims are weak)"""
    w = np.where(grid.type == 1, 1.0, np.where(grid.type == 2, 0.7, 0.35))
    return Music(grid.t.copy(), w, [], np.array([]), np.array([]), "grid")


def load_inputs(take: str, timing: Path | None = None, song_dir: Path | None = None, audio: Path | None = None, cfg: Config | None = None, work: Path | None = None) -> Inputs:
    cfg = cfg or Config()
    if work is None:
        from .take import Take
        work = Take.load(take).work
    work = Path(work)
    timing = Path(timing) if timing else work / "timing" / "timing.json"
    steps, rows, dropped, meta = load_timing(timing, cfg)
    song_dir = Path(song_dir) if song_dir else (work / "song" if (work / "song").exists() else None)
    sd = load_song_dir(song_dir, rows)
    conf = sd["conf"]
    none = list(sd["none"])
    seam = [x for x in dropped if not any(abs(x[0] - n[0]) < 0.5 and abs(x[1] - n[1]) < 0.5 for n in none)]
    if not none and dropped:
        # no segments.json: the intervals the timing dropped that touch the grid's ends are "no song", the others seams
        t_lo, t_hi = rows[0][0], rows[-1][0]
        none = [x for x in dropped if x[1] <= t_lo + 1.0 or x[0] >= t_hi - 1.0]
        seam = [x for x in dropped if x not in none]
    grid = build_grid(rows, no_beat_none=none, explicit_gaps=seam, conf=None, song=sd["song"], default_conf=cfg.grid_confidence_default)
    if conf is not None:
        c = np.where(np.isnan(conf), cfg.grid_confidence_default, conf)
        grid.conf = c
        grid.song = sd["song"] or grid.song
    # audio: explicit, else the capture / take soundtrack on the reference clock
    t0a = 0.0
    if audio is None:
        for cand in (work / "audio" / "heat_studio.wav", work / "audio" / "audio_overlap.wav"):
            if cand.exists():
                audio = cand
                break
    if audio is not None:
        j = Path(audio).with_suffix(".json")
        if j.exists():
            try:
                t0a = float(json.loads(j.read_text(encoding="utf-8")).get("reference_start", 0.0))
            except (OSError, ValueError):
                pass
    motion = load_motion(work / "timing" / "kinematics.npz")
    none_all = [(a, b) for a, b, k in grid.no_beat if k == "none"]
    music = music_from_audio(Path(audio), t0a, cfg, exclude=none_all) if audio is not None and Path(audio).exists() else None
    if music is None:
        music = music_from_grid(grid)
    t0 = float(motion.t[0]) if motion is not None else min(min((s["t"] for s in v), default=1e9) for v in steps.values()) - 1.0
    t1 = float(motion.t[-1]) if motion is not None else max(max((s["t"] for s in v), default=0) for v in steps.values()) + 1.0
    cfg.av_pinned = bool(meta.get("av_pinned"))
    meta.update({"take": take, "audio": str(audio) if audio else None, "song_dir": str(song_dir) if song_dir else None, "music_source": music.source})
    inp = Inputs(grid, steps, music, motion, t0, t1, meta)
    inp.variants = seam_variants(song_dir, grid, cfg)
    return inp


# --------------------------------------------------------------------------------------------------------------- metrical levels
# The lattices a step can sit on, all anchored on the grid's pulses: the pulse (quarter), the eighth (pulses and "ands"), and the triplet levels
# (three steps in the time of two pulses, a swung / hemiola feel, two phases = which pulses it starts on; and three per pulse). Zouk dancers step
# on the pulse or the "and", but a step pattern in 3-against-2 is as much "in time" with the music: each role is scored against the level its own
# steps fit best (the choice is penalised, see level_skills), so a role that locks to nothing scores nothing.
LEVELS = (
    {"name": "pulse", "label": "beat", "factor": 1.0, "parity": None},
    {"name": "eighth", "label": "beat / and", "factor": 0.5, "parity": None},
    {"name": "triplet_a", "label": "3 in 2 (A)", "factor": 2.0 / 3.0, "parity": 0},
    {"name": "triplet_b", "label": "3 in 2 (B)", "factor": 2.0 / 3.0, "parity": 1},
    {"name": "triplet_eighth", "label": "triplet", "factor": 1.0 / 3.0, "parity": None},
)
K = len(LEVELS)
PRIMARY = 1                                  # the plain binary grid (every slot) when no level fits better
SELECT_Z = 1.35                              # expected maximum of K=5 standard normals: the price of picking the best level


class PulseTable:
    """the grid's pulses (even slots) with the pulse index inside their contiguous segment (parity of the 3-in-2 lattices) and the local period"""

    def __init__(self, grid: Grid):
        m = grid.slot % 2 == 0
        self.t = grid.t[m]
        med = float(np.median(np.diff(self.t))) if len(self.t) > 2 else 0.74
        gap = np.r_[False, np.diff(self.t) > 1.6 * med]
        seg = np.cumsum(gap)
        idx = np.zeros(len(self.t), int)
        for s_id in np.unique(seg):
            w = np.where(seg == s_id)[0]
            idx[w] = np.arange(len(w))
        self.par = idx % 2
        self.seg = seg
        per = np.diff(self.t)
        self.per = np.r_[per, per[-1] if len(per) else med]

    def offsets(self, x: float):
        """signed offset (ms) of time x to the nearest lattice point of every level, and the level periods (ms)"""
        j = int(np.searchsorted(self.t, x, side="right")) - 1
        j = int(np.clip(j, 0, len(self.t) - 1))
        P = float(self.per[j])
        d = np.zeros(K)
        per = np.zeros(K)
        for k, L in enumerate(LEVELS):
            a = j
            if L["parity"] is not None:
                for cand in (j, j - 1):
                    if 0 <= cand < len(self.t) and self.seg[cand] == self.seg[j] and self.par[cand] == L["parity"]:
                        a = cand
                        break
            p = P * L["factor"]
            off = x - self.t[a]
            d[k] = ((off + p / 2) % p) - p / 2
            per[k] = p
        return d * 1000.0, per * 1000.0


# --------------------------------------------------------------------------------------------------------------- events
@dataclass
class Events:
    """every scored event, with the time it becomes known (t_known) and the values the components are made of"""
    lock: list = field(default_factory=list)      # t, role, row (index into steps), d[K], per[K], cred[K], chance[K], chance2[K], z[K], conf
    accent: list = field(default_factory=list)    # t, role, kind (step|stop), credit, chance, conf
    tempo: list = field(default_factory=list)     # t, role, cred[K], chance[K], chance2[K], interval_ms
    sync: list = field(default_factory=list)
    hold: list = field(default_factory=list)
    steps: list = field(default_factory=list)     # the per-step table rows


def accent_credit_curve(music: Music, cfg: Config, t_lo: float, t_hi: float, hz: float = 100.0):
    """credit(t) = max over strong accents of strength x gaussian(t - accent), and its local mean (the chance level), on a regular grid"""
    tt = np.arange(t_lo - 1.0, t_hi + 1.0, 1.0 / hz)
    cr = np.zeros(len(tt))
    sig = cfg.accent_sigma_ms / 1000.0
    m = music.acc_s >= cfg.accent_min_strength
    for a, s in zip(music.acc_t[m], music.acc_s[m]):
        i0, i1 = np.searchsorted(tt, [a - 4 * sig, a + 4 * sig])
        if i1 > i0:
            cr[i0:i1] = np.maximum(cr[i0:i1], s * np.exp(-0.5 * ((tt[i0:i1] - a) / sig) ** 2))
    w = max(1, int(cfg.accent_chance_window_s * hz))
    c = np.cumsum(np.r_[0.0, cr])
    idx = np.arange(len(tt))
    lo, hi = np.clip(idx - w, 0, len(tt)), np.clip(idx + w + 1, 0, len(tt))
    mean = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return tt, cr, mean


def _nearest_slot(grid: Grid, t: float):
    i = int(np.searchsorted(grid.t, t))
    cands = [k for k in (i - 1, i) if 0 <= k < len(grid)]
    return min(cands, key=lambda c: abs(grid.t[c] - t))


def _crosses_gap(grid: Grid, a: float, b: float) -> bool:
    return any(x < b and y > a for x, y, _ in grid.no_beat)


def _level_stats(d, per, sigma):
    cred = np.exp(-0.5 * (d / sigma) ** 2)
    ch = np.array([lock_chance(sigma, p) for p in per])
    ch2 = np.array([lock_chance(sigma / math.sqrt(2.0), p) for p in per])
    z = np.exp(2j * np.pi * d / per)
    return cred, ch, ch2, z


def build_events(inp: Inputs, cfg: Config, grid: Grid | None = None, av_shift_s: float = 0.0) -> Events:
    """Events of one analysis. `av_shift_s` moves the music against the motion (the +-30 ms audio-video uncertainty): a step at motion time t is compared
    with the grid at t + av_shift_s."""
    grid = grid or inp.grid
    ev = Events()
    music = inp.music
    cr_t = cr = cr_mean = None
    if music is not None and len(music.acc_t):
        cr_t, cr, cr_mean = accent_credit_curve(music, cfg, inp.t0, inp.t1)
    pt = PulseTable(grid)
    p8_ms = grid.p8 * 1000.0

    def credit_at(t):
        if cr_t is None:
            return 0.0, 0.0
        i = int(np.clip(round((t - cr_t[0]) * 100.0), 0, len(cr_t) - 1))
        return float(cr[i]), float(cr_mean[i])

    steps_by_role = {r: [] for r in ROLES}
    for r in ROLES:
        prev = None
        for s in sorted(inp.steps.get(r, []), key=lambda x: x["t"]):
            tm = s["t"]                              # motion time
            tg = tm + av_shift_s                     # the same instant on the music's clock
            st = grid.state_at(tg)
            row = {"t": round(tm, 4), "role": r, "foot": s.get("foot", ""), "state": st, "sigma_ms": round(s.get("sigma", 0.02) * 1000.0, 1)}
            k = _nearest_slot(grid, tg) if len(grid) else 0
            if st != 0 or abs((tg - grid.t[k]) * 1000.0) > 0.75 * p8_ms:
                row["valid"] = False
                ev.steps.append(row)
                prev = None
                continue
            d_ms = (tg - grid.t[k]) * 1000.0
            slot = int(grid.slot[k])
            kind = "beat" if slot % 2 == 0 else "and"
            local_p8 = float(np.median(np.diff(grid.t[max(0, k - 3):k + 4]))) * 1000.0 if len(grid) > 3 else p8_ms
            conf = float(grid.conf[k])
            a_cr, a_ch = credit_at(tg)
            d_lv, per_lv = pt.offsets(tg)
            cred, ch, ch2, z = _level_stats(d_lv, per_lv, cfg.sigma_ms)
            row.update({"valid": True, "measure": int(grid.measure[k]), "slot": slot, "kind": kind, "offset_ms": round(d_ms, 1),
                        "accent": round(a_cr, 3), "accent_chance": round(a_ch, 3), "conf": round(conf, 3), "p8_ms": round(local_p8, 1)})
            ev.steps.append(row)
            ev.lock.append({"t": tm, "role": r, "row": len(ev.steps) - 1, "d": d_lv, "per": per_lv, "cred": cred, "chance": ch, "chance2": ch2, "z": z, "conf": conf})
            ev.accent.append({"t": tm, "role": r, "kind": "step", "credit": a_cr, "chance": a_ch, "conf": conf})
            steps_by_role[r].append((tm, tg, k, local_p8))
            if prev is not None:
                dt_ms = (tg - prev[1]) * 1000.0
                if 0.5 * local_p8 <= dt_ms <= 8.0 * local_p8 and not _crosses_gap(grid, prev[1], tg):
                    q = ((dt_ms + per_lv / 2.0) % per_lv) - per_lv / 2.0           # distance of the interval to a whole number of the level's periods
                    tc, tch, tch2, _ = _level_stats(q, per_lv, cfg.sigma_ms)
                    ev.tempo.append({"t": tm, "role": r, "cred": tc, "chance": tch, "chance2": tch2, "conf": conf, "interval_ms": dt_ms})
            prev = (tm, tg, k)
    # couple synchrony: each step against the other role's nearest step
    sig = cfg.sync_sigma_ms / 1000.0
    for r, o in (("lead", "follow"), ("follow", "lead")):
        ot = np.asarray([x[0] for x in steps_by_role[o]], float)
        if len(ot) == 0:
            continue
        for (tm, tg, k, p8l) in steps_by_role[r]:
            near = np.abs(ot - tm)
            j = int(np.argmin(near))
            dd = float(near[j])
            credit = math.exp(-0.5 * (dd / sig) ** 2)
            lam = float(((ot > tm - 6.0) & (ot < tm + 6.0)).sum()) / 12.0
            chance = 1.0 - math.exp(-lam * sig * math.sqrt(2.0 * math.pi))
            known = max(tm, float(ot[j]) if dd < 3 * sig else tm) + cfg.sync_latency_s
            ev.sync.append({"t": tm, "t_known": known, "role": r, "credit": credit, "chance": chance, "delta_ms": (float(ot[j]) - tm) * 1000.0,
                            "conf": float(grid.conf[k])})
    # stops (accent credit) and holds on the music's breaks (need the motion)
    if inp.motion is not None and len(inp.motion.t):
        mt = inp.motion.t
        fps = 1.0 / float(np.median(np.diff(mt)))
        for r in ROLES:
            sp = np.convolve(inp.motion.speed[r], np.ones(5) / 5.0, "same")
            thr = cfg.stop_ratio * float(np.median(sp))
            low = sp < thr
            i = 0
            while i < len(low):
                if low[i]:
                    j = i
                    while j < len(low) and low[j]:
                        j += 1
                    if (j - i) / fps >= cfg.stop_min_s and i > int(0.3 * fps) and sp[max(0, i - int(0.3 * fps)):i].max() > 1.5 * thr:
                        tm = float(mt[i])
                        tg = tm + av_shift_s
                        if grid.state_at(tg) == 0 and grid.t[0] <= tg <= grid.t[-1]:
                            k = _nearest_slot(grid, tg)
                            a_cr, a_ch = credit_at(tg)
                            ev.accent.append({"t": tm, "role": r, "kind": "stop", "credit": a_cr, "chance": a_ch, "conf": float(grid.conf[k])})
                    i = j
                else:
                    i += 1
        if music is not None and music.breaks:
            for (b0, b1) in music.breaks:
                bt0, bt1 = b0 - av_shift_s, b1 - av_shift_s
                if grid.state_at(b0) != 0 or bt1 > mt[-1] or bt0 < mt[0]:
                    continue
                a, b = bt0 + 0.1, max(bt0 + 0.2, bt1)
                m = (mt >= a) & (mt <= b)
                if m.sum() < 3:
                    continue
                for r in ROLES:
                    sp = inp.motion.speed[r]
                    ref = 0.6 * float(np.median(sp))
                    q = 1.0 - float(np.clip(sp[m].mean() / max(ref, 1e-6), 0.0, 1.0))
                    L = int(m.sum())
                    cs = np.cumsum(np.r_[0.0, sp])
                    w = np.clip(1.0 - (cs[L:] - cs[:-L]) / L / max(ref, 1e-6), 0.0, 1.0)
                    ti = np.searchsorted(mt, bt0)
                    lo, hi = max(0, ti - int(10 * fps)), min(len(w), ti + int(10 * fps))
                    chance = float(w[lo:hi].mean()) if hi > lo else 0.5
                    k = _nearest_slot(grid, b0)
                    ev.hold.append({"t": bt0, "t_known": bt1, "role": r, "credit": q, "chance": chance, "conf": float(grid.conf[k]), "break": [round(b0, 3), round(b1, 3)]})
    for lst in (ev.lock, ev.accent, ev.tempo):
        lst.sort(key=lambda e: e["t"])
    for e in ev.accent + ev.lock + ev.tempo:
        e["t_known"] = e["t"] + 0.05
    return ev


# --------------------------------------------------------------------------------------------------------------- accumulators
class Acc:
    """weighted sums of a metric's per-event credits and chances (for the whole dance, a window, or an exponential memory)"""
    __slots__ = ("w", "ww", "wv", "wc")

    def __init__(self):
        self.w = self.ww = self.wv = self.wc = 0.0

    def decay(self, f: float):
        self.w *= f
        self.ww *= f * f
        self.wv *= f
        self.wc *= f

    def add(self, v: float, c: float, w: float = 1.0):
        self.w += w
        self.ww += w * w
        self.wv += w * v
        self.wc += w * c


class AccK:
    """the same per metrical level (arrays of length K), plus the chance variance term, the phasor sum (flow) and the level periods"""
    __slots__ = ("w", "ww", "wv", "wc", "wc2", "wz", "wp")

    def __init__(self):
        self.w = self.ww = 0.0
        self.wv, self.wc, self.wc2, self.wp = (np.zeros(K) for _ in range(4))
        self.wz = np.zeros(K, complex)

    def decay(self, f: float):
        self.w *= f
        self.ww *= f * f
        self.wv *= f
        self.wc *= f
        self.wc2 *= f
        self.wp *= f
        self.wz *= f

    def add(self, cred, chance, chance2, z=None, per=None, w: float = 1.0):
        self.w += w
        self.ww += w * w
        self.wv += w * cred
        self.wc += w * chance
        self.wc2 += w * chance2
        if z is not None:
            self.wz += w * z
        if per is not None:
            self.wp += w * per


@dataclass
class State:
    lock: dict = field(default_factory=lambda: {r: AccK() for r in ROLES})
    tempo: dict = field(default_factory=lambda: {r: AccK() for r in ROLES})
    accent: Acc = field(default_factory=Acc)
    hold: Acc = field(default_factory=Acc)
    sync: Acc = field(default_factory=Acc)

    def decay(self, f):
        for r in ROLES:
            self.lock[r].decay(f)
            self.tempo[r].decay(f)
        self.accent.decay(f)
        self.hold.decay(f)
        self.sync.decay(f)


def add_event(st: State, kind: str, e: dict, w: float = 1.0):
    if kind == "lock":
        st.lock[e["role"]].add(e["cred"], e["chance"], e["chance2"], e["z"], e["per"], w)
    elif kind == "tempo":
        st.tempo[e["role"]].add(e["cred"], e["chance"], e["chance2"], None, None, w)
    elif kind == "accent":
        st.accent.add(e["credit"], e["chance"], w)
    elif kind == "sync":
        st.sync.add(e["credit"], e["chance"], w)
    elif kind == "hold":
        st.hold.add(e["credit"], e["chance"], w)


def skill(acc: Acc, ideal: float, n0: float, prior: float) -> tuple[float, float]:
    """(0..1 above chance, number of events) with `n0` prior events at `prior`"""
    if acc.w <= 1e-9:
        return (prior if n0 > 0 else float("nan")), 0.0                       # nothing seen and no prior: unknown (a descriptive window with no events)
    v, c = acc.wv / acc.w, acc.wc / acc.w
    s = float(np.clip((v - c) / max(ideal - c, 1e-6), 0.0, 1.0))
    return (n0 * prior + acc.w * s) / (n0 + acc.w), acc.w


def level_skills(a: AccK, ideal: float) -> dict:
    """per metrical level: raw skill above chance, the z of its excess over chance, and the skill after the price of choosing among K levels"""
    n_eff = a.w * a.w / max(a.ww, 1e-12) if a.w > 1e-9 else 0.0
    if n_eff < 1.0:
        return {"skill": np.zeros(K), "z": np.zeros(K), "pskill": np.zeros(K), "n_eff": n_eff}
    v, c, c2 = a.wv / a.w, a.wc / a.w, a.wc2 / a.w
    sd = np.sqrt(np.maximum(c2 - c * c, 1e-9))
    sk = (v - c) / np.maximum(ideal - c, 1e-6)
    z = (v - c) / (sd / math.sqrt(n_eff))
    pen = SELECT_Z * (sd / math.sqrt(n_eff)) / np.maximum(ideal - c, 1e-6)
    return {"skill": sk, "z": z, "pskill": np.clip(sk - pen, 0.0, 1.0), "n_eff": n_eff}


def components(st: State, cfg: Config, prior_scale: float = 1.0) -> dict:
    n0 = cfg.prior_events * prior_scale
    p = cfg.neutral
    ideal = cfg.lock_ideal()
    lock_num = flow_num = tempo_num = 0.0
    lock_w = flow_w = tempo_w = 0.0
    sel, zmax, R_all, levels = {}, [], [], {}
    for r in ROLES:
        a = st.lock[r]
        ls = level_skills(a, ideal)
        k = int(np.argmax(ls["pskill"]))
        if ls["pskill"][k] < 0.03:
            k = PRIMARY
        sel[r] = k
        levels[r] = ls
        w = a.w
        lock_num += n0 * p + w * ls["pskill"][k]
        lock_w += n0 + w
        if w > 1e-9:
            zmax.append(float(ls["z"].max()))
            # flow: the steadiness of the phase about the chosen lattice (bias corrected), only if the steps sit near the lattice points at all
            R = abs(a.wz[k]) / w
            n_eff = ls["n_eff"]
            r2 = (n_eff * R * R - 1.0) / (n_eff - 1.0) if n_eff > 1.0 else 0.0
            s = float(np.clip(math.sqrt(max(r2, 0.0)) / cfg.ideal_flow_R, 0.0, 1.0))
            per_ms = a.wp[k] / w
            mu_ms = math.atan2(a.wz[k].imag, a.wz[k].real) / (2.0 * math.pi) * per_ms
            s *= math.exp(-0.5 * (mu_ms / cfg.flow_offset_sigma_ms) ** 2)
            flow_num += n0 * p + w * s
            R_all.append(R)
        else:
            flow_num += n0 * p
        flow_w += n0 + w
        t = st.tempo[r]
        if t.w > 1e-9:
            v, c = t.wv[k] / t.w, t.wc[k] / t.w
            s = float(np.clip((v - c) / max(ideal - c, 1e-6), 0.0, 1.0))
            tempo_num += n0 * p + t.w * s
        else:
            tempo_num += n0 * p
        tempo_w += n0 + t.w
    nan = float("nan")
    lock = lock_num / lock_w if lock_w > 1e-9 else nan
    flow = flow_num / flow_w if flow_w > 1e-9 else nan
    tempo = tempo_num / tempo_w if tempo_w > 1e-9 else nan
    acc_steps, n_acc = skill(st.accent, cfg.accent_ideal, n0, p)
    hold, n_hold = skill(st.hold, cfg.accent_ideal + 0.15, n0, p)
    hshare = cfg.hold_weight * (n_hold / (n_hold + 2.0)) if st.hold.w > 1e-9 else 0.0
    accents = (1.0 - hshare) * acc_steps + hshare * hold
    sync, n_s = skill(st.sync, cfg.sync_ideal, n0, p)
    z_ev = (sum(zmax) / math.sqrt(len(zmax)) - SELECT_Z) if zmax else 0.0
    return {"lock": lock, "accents": accents, "tempo": tempo, "flow": flow, "sync": sync, "sel": sel, "levels": levels, "z_evidence": z_ev,
            "n": {"lock": st.lock["lead"].w + st.lock["follow"].w, "accents": n_acc, "holds": n_hold, "tempo": st.tempo["lead"].w + st.tempo["follow"].w, "sync": n_s},
            "R": float(np.mean(R_all)) if R_all else 0.0}


def _r(x, p: int = 3):
    """round, with an unknown (NaN) value as None (valid JSON)"""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), p)


def score_of(comp: dict, cfg: Config) -> float:
    """100 x the weighted components; a component with nothing seen (NaN, descriptive windows only) is left out and the weights renormalised"""
    ok = [k for k in COMPONENTS if not math.isnan(comp[k])]
    if not ok:
        return float("nan")
    wsum = sum(cfg.weights[k] for k in ok)
    return float(100.0 * sum(cfg.weights[k] * comp[k] for k in ok) / wsum)


def all_events(ev: Events):
    """[(t_known, kind, event)] sorted"""
    out = []
    for kind in ("lock", "accent", "tempo", "sync", "hold"):
        for e in getattr(ev, kind):
            out.append((e.get("t_known", e["t"]), kind, e))
    out.sort(key=lambda x: x[0])
    return out


# --------------------------------------------------------------------------------------------------------------- live score
def live_timeline(inp: Inputs, ev: Events, cfg: Config, grid: Grid | None = None) -> dict:
    """The causal live score at cfg.live_hz from inp.t0 to inp.t1: at sample time t only events with t_known <= t are used. The memory clock
    stands still in the no-beat stretches (no song, the DJ seam), so the score is frozen there."""
    grid = grid or inp.grid
    hz = cfg.live_hz
    n = int(math.floor((inp.t1 - inp.t0) * hz + 1e-6)) + 1
    ts = inp.t0 + np.arange(n) / hz
    events = all_events(ev)
    cum, rec = State(), State()
    ei = 0
    out = {k: np.zeros(n) for k in ("score", "lock", "accents", "tempo", "flow", "sync", "confidence", "recent_score", "cum_score")}
    out["sel_lead"] = np.full(n, PRIMARY, int)
    out["sel_follow"] = np.full(n, PRIMARY, int)
    state = np.zeros(n, dtype=int)
    active_s = 0.0
    share = cfg.recent_share
    gconf = np.interp(ts, grid.t, grid.conf)
    for j in range(n):
        t = ts[j]
        st = grid.state_at(t)
        state[j] = st
        if j > 0 and st == 0:
            rec.decay(0.5 ** ((1.0 / hz) / cfg.half_life_s))
            active_s += 1.0 / hz
        added = False
        while ei < len(events) and events[ei][0] <= t + 1e-9:
            tk, kind, e = events[ei]
            if grid.state_at(tk) == 0 or st == 0:
                add_event(cum, kind, e)
                add_event(rec, kind, e)
                ei += 1
                added = True
            else:
                break
        if added or j == 0:
            # a pure decay scales every sum of the memory AND its prior by the same factor, so the memory's components change only when events arrive
            cc = components(cum, cfg)
            rc = components(rec, cfg, prior_scale=0.5 ** (active_s / cfg.half_life_s))
            comp = {k: (1 - share) * cc[k] + share * rc[k] for k in COMPONENTS}
            sc, cs, rs = score_of(comp, cfg), score_of(cc, cfg), score_of(rc, cfg)
            samp = min(1.0, max(0.0, cc["n"]["lock"] / cfg.min_events_for_confidence)) ** 0.5
            evidence = min(1.0, max(0.25, cc["z_evidence"] / 3.0))
            av = 1.0 if cfg.av_pinned else cfg.av_cap_unpinned
            conf_k = samp * evidence * av
            sel = (cc["sel"]["lead"], cc["sel"]["follow"])
        out["cum_score"][j], out["recent_score"][j] = cs, rs
        for k in COMPONENTS:
            out[k][j] = comp[k]
        out["score"][j] = sc
        out["sel_lead"][j], out["sel_follow"][j] = sel
        c = min(1.0, max(0.0, float(gconf[j]))) * conf_k
        out["confidence"][j] = 0.0 if st == 1 else c * 0.5 if st == 2 else c
    out["t"] = ts
    out["state"] = state
    return out


def annotate_steps(ev: Events, live: dict, cfg: Config):
    """each step's level (what its role's steps fitted best BEFORE this step: causal) and its offset / class against that lattice"""
    t0, hz = live["t"][0], cfg.live_hz
    for e in ev.lock:
        row = ev.steps[e["row"]]
        j = int(np.clip(math.floor((e["t"] - t0) * hz), 0, len(live["t"]) - 1))
        k = int(live["sel_" + e["role"]][j])
        d = float(e["d"][k])
        per = float(e["per"][k])
        on, near = min(cfg.on_ms, 0.16 * per), min(cfg.near_ms, 0.32 * per)
        row["level"] = LEVELS[k]["name"]
        row["level_offset_ms"] = round(d, 1)
        row["class"] = "on" if abs(d) <= on else "near" if abs(d) <= near else "off"
        row["lock"] = round(float(e["cred"][k]), 3)
        e["sel"] = k


# --------------------------------------------------------------------------------------------------------------- aggregates (whole dance / windows / measures)
def aggregate(ev: Events, cfg: Config, t_lo: float = -1e9, t_hi: float = 1e9, n0_scale: float = 0.0) -> dict:
    st = State()
    for kind in ("lock", "accent", "tempo", "sync", "hold"):
        for e in getattr(ev, kind):
            if t_lo <= e["t"] < t_hi:
                add_event(st, kind, e)
    comp = components(st, cfg, prior_scale=n0_scale)
    comp["score"] = score_of(comp, cfg)
    return comp


def confidence_summary(ev: Events, grid: Grid, inp: Inputs, cfg: Config, comp: dict) -> dict:
    n = comp["n"]["lock"]
    z = comp["z_evidence"]
    gc = float(np.mean([e["conf"] for e in ev.lock])) if ev.lock else 0.0
    evidence = float(np.clip(z / 3.0, 0.25, 1.0))
    av = 1.0 if cfg.av_pinned else cfg.av_cap_unpinned
    beat_frac = float(np.mean([grid.state_at(t) == 0 for t in np.arange(inp.t0, inp.t1, 0.5)]))
    conf = gc * evidence * av * float(np.clip(n / (4 * cfg.min_events_for_confidence), 0.0, 1.0)) ** 0.5
    label = "high" if conf >= 0.7 else "medium" if conf >= 0.4 else "low"
    return {"value": round(conf, 3), "label": label, "grid_confidence_mean": round(gc, 3), "lock_evidence_z": round(z, 2),
            "evidence_factor": round(evidence, 3), "av_factor": av, "beat_covered_share": round(beat_frac, 3), "n_steps_scored": int(n)}


def measure_table(inp: Inputs, ev: Events, cfg: Config, grid: Grid | None = None) -> list:
    grid = grid or inp.grid
    out = []
    for m in np.unique(grid.measure):
        idx = np.where(grid.measure == m)[0]
        if len(idx) < 8:
            continue
        a, b = float(grid.t[idx[0]] - grid.p8 / 2), float(grid.t[idx[-1]] + grid.p8 / 2)
        sub = [s for s in ev.steps if s.get("valid") and a <= s["t"] < b]
        comp = aggregate(ev, cfg, a, b)
        out.append({"measure": int(m), "t0": round(a, 3), "t1": round(b, 3), "song": grid.song[idx[0]] if grid.song else "",
                    "n_lead": sum(1 for s in sub if s["role"] == "lead"), "n_follow": sum(1 for s in sub if s["role"] == "follow"),
                    "on_share": round(sum(1 for s in sub if s.get("class") == "on") / len(sub), 3) if sub else None,
                    "steps_per_pulse": {r: round(sum(1 for s in sub if s["role"] == r) / 4.0, 3) for r in ROLES},
                    "mean_offset_ms": round(float(np.mean([s["offset_ms"] for s in sub])), 1) if sub else None,
                    "score": _r(comp["score"], 1), "components": {k: _r(comp[k]) for k in COMPONENTS},
                    "levels": {r: LEVELS[comp["sel"][r]]["name"] for r in ROLES}, "conf": round(float(np.mean(grid.conf[idx])), 3)})
    return out


def window_table(inp: Inputs, ev: Events, cfg: Config) -> list:
    out = []
    half = cfg.window_s / 2
    t = inp.t0 + half
    while t <= inp.t1 - half + 1e-6:
        a, b = t - half, t + half
        comp = aggregate(ev, cfg, a, b)
        beat = float(np.mean([inp.grid.state_at(x) == 0 for x in np.arange(a, b, 0.25)]))
        out.append({"t": round(t, 2), "score": _r(comp["score"], 1) if beat > 0.3 else None, "beat_share": round(beat, 2),
                    "components": {k: _r(comp[k]) for k in COMPONENTS}, "n_lock": round(comp["n"]["lock"], 1)})
        t += cfg.window_step_s
    return out


def level_table(ev: Events, cfg: Config, t_lo: float = -1e9, t_hi: float = 1e9) -> dict:
    """per role: for every metrical level the raw skill above chance, its z and the skill after the price of choosing (the evidence behind the lock)"""
    st = State()
    for e in ev.lock:
        if t_lo <= e["t"] < t_hi:
            add_event(st, "lock", e)
    out = {}
    for r in ROLES:
        ls = level_skills(st.lock[r], cfg.lock_ideal())
        out[r] = {"n": round(st.lock[r].w), "levels": {L["name"]: {"skill": round(float(ls["skill"][k]), 3), "z": round(float(ls["z"][k]), 2),
                                                                 "skill_after_choice_price": round(float(ls["pskill"][k]), 3)} for k, L in enumerate(LEVELS)}}
    return out


def descriptive(ev: Events, inp: Inputs, cfg: Config) -> dict:
    """distribution numbers of the per-step table (valid steps)"""
    out: dict = {}
    p8 = inp.grid.p8 * 1000.0
    for r in ROLES + ("both",):
        s = [x for x in ev.steps if x.get("valid") and (r == "both" or x["role"] == r)]
        if not s:
            continue
        d = np.array([x["offset_ms"] for x in s])
        n = len(s)
        R = float(abs(np.exp(2j * np.pi * d / p8).mean()))
        within = lambda ms: float(np.mean(np.abs(d) <= ms))                # noqa: E731
        out[r] = {"n": n, "binary_grid": {"within_on_ms": round(within(cfg.on_ms), 3), "within_near_ms": round(within(cfg.near_ms), 3),
                                          "on_chance": round(2 * cfg.on_ms / p8, 3), "on_pulse_slot": round(float(np.mean([x["kind"] == "beat" for x in s])), 3),
                                          "mean_offset_ms": round(float(d.mean()), 1), "median_offset_ms": round(float(np.median(d)), 1), "sd_offset_ms": round(float(d.std()), 1),
                                          "mean_abs_offset_ms": round(float(np.abs(d).mean()), 1), "R_eighth": round(R, 3), "R_chance": round(1.0 / math.sqrt(n), 3),
                                          "rayleigh_p": round(float(math.exp(-n * R * R)), 4)}}
        cl = [x.get("class") for x in s]
        out[r]["against_chosen_level"] = {"on": round(cl.count("on") / n, 3), "near": round(cl.count("near") / n, 3), "off": round(cl.count("off") / n, 3),
                                          "levels_used": {str(nm): sum(1 for x in s if x.get("level") == nm) for nm in {x.get("level") for x in s}}}
    sy = [e["delta_ms"] for e in ev.sync if e["role"] == "lead" and abs(e["delta_ms"]) < 150]
    out["couple"] = {"n_pairs": len(sy), "follow_minus_lead_ms_median": round(float(np.median(sy)), 1) if sy else None,
                     "within_60ms": round(float(np.mean([abs(x) <= 60 for x in sy])), 3) if sy else None}
    per_song = {}
    for sg in sorted({x for x in inp.grid.song if x}):
        idx = np.where(np.array(inp.grid.song) == sg)[0]
        a, b = float(inp.grid.t[idx[0]]), float(inp.grid.t[idx[-1]])
        ent = {"span_s": [round(a, 2), round(b, 2)], "levels": level_table(ev, cfg, a, b)}
        for r in ROLES:
            n = sum(1 for x in ev.steps if x.get("valid") and x["role"] == r and a <= x["t"] <= b)
            ent[r] = {"steps": n, "steps_per_pulse": round(n / max((b - a) / (2 * inp.grid.p8), 1e-9), 3)}
        per_song[sg] = ent
    if per_song:
        out["per_song"] = per_song
    out["levels_whole_dance"] = level_table(ev, cfg)
    return out


def analyse_one(inp: Inputs, cfg: Config, grid: Grid | None = None, av_shift_s: float = 0.0, full: bool = True) -> dict:
    grid = grid or inp.grid
    ev = build_events(inp, cfg, grid, av_shift_s)
    final = aggregate(ev, cfg, n0_scale=0.0)
    res = {"events": ev, "final": final}
    if full:
        live = live_timeline(inp, ev, cfg, grid)
        annotate_steps(ev, live, cfg)
        res["live"] = live
    return res


def sensitivity(inp: Inputs, cfg: Config) -> dict:
    """The final score under the +-30 ms audio-video uncertainty and with the DJ seam handled the other ways"""
    rows = []
    grids = {"default (seam frozen)": inp.grid}
    grids.update(inp.variants)
    for gname, g in grids.items():
        for sh in (-cfg.av_uncertainty_ms, 0.0, cfg.av_uncertainty_ms):
            f = analyse_one(inp, cfg, g, av_shift_s=sh / 1000.0, full=False)["final"]
            rows.append({"grid": gname, "shift_ms": sh, "score": _r(f["score"], 1), "components": {k: _r(f[k]) for k in COMPONENTS},
                         "levels": {r: LEVELS[f["sel"][r]]["name"] for r in ROLES}, "n_lock": round(f["n"]["lock"], 1)})
    sc = [r["score"] for r in rows if r["score"] is not None] or [None]
    shift_only = [r["score"] for r in rows if r["grid"].startswith("default") and r["score"] is not None] or [None]
    return {"rows": rows, "range": [min(sc), max(sc)], "range_shift_only": [min(shift_only), max(shift_only)],
            "note": "av shift: the motion is moved against the music by +-%g ms (the unpinned audio-video offset); seam variants: the first / second song's pulses through the DJ mix" % cfg.av_uncertainty_ms}


def claims(desc: dict, conf: dict, final: dict, inp: Inputs, cfg: Config) -> list:
    out = [f"{TITLE}. It measures how the detected foot touchdowns, stops and holds sit in the song's beat grid; it does not judge technique, musicality as a whole or anything an official judge scores."]
    b = desc.get("both")
    if b:
        g = b["binary_grid"]
        out.append(f"{b['n']} scored touchdowns: {g['within_on_ms']*100:.0f}% within {cfg.on_ms:.0f} ms of a beat or 'and' slot (random steps: {g['on_chance']*100:.0f}%), mean offset {g['mean_offset_ms']:+.0f} ms, "
                   f"phase concentration R {g['R_eighth']:.2f} against {g['R_chance']:.2f} for random (Rayleigh p {g['rayleigh_p']:.2f}).")
        if g["rayleigh_p"] > 0.05:
            out.append("The steps do NOT sit on the plain beat / 'and' grid detectably: that lock is at chance.")
    used = {r: LEVELS[final["sel"][r]]["label"] for r in ROLES}
    out.append(f"Best-fitting metrical level over the whole dance (chosen with a price for choosing): lead {used['lead']}, follow {used['follow']}; evidence z {final['z_evidence']:.1f} after the price (about 3 or more = a real lock).")
    if not inp.meta.get("av_pinned"):
        out.append(f"Absolute on-the-beat claims carry the unpinned audio-video offset (+-{cfg.av_uncertainty_ms:.0f} ms): no sync event in the video. Relative timing (lead vs follow, measure vs measure) does not.")
    if any(k == "seam" for _, _, k in inp.grid.no_beat):
        out.append("The DJ mix hands one song over to the other without a shared grid: the score is frozen there (no steps are scored in the seam).")
    if inp.music is not None and inp.music.source != "audio":
        out.append("No soundtrack was given: accents are the grid's downbeats only and the bar phase may be off, so the accent component is weak.")
    out.append(f"Confidence {conf['label']} ({conf['value']:.2f}): grid confidence {conf['grid_confidence_mean']:.2f}, lock evidence factor {conf['evidence_factor']:.2f}, A/V factor {conf['av_factor']:.2f}.")
    return out


def pack_live(live: dict, cfg: Config) -> dict:
    r = lambda a, p: [round(float(x), p) for x in a]                      # noqa: E731
    return {"hz": cfg.live_hz, "t0": round(float(live["t"][0]), 4), "n": int(len(live["t"])),
            "score": r(live["score"], 2), "lock": r(live["lock"], 3), "accents": r(live["accents"], 3), "tempo": r(live["tempo"], 3), "flow": r(live["flow"], 3),
            "sync": r(live["sync"], 3), "confidence": r(live["confidence"], 3), "state": [int(x) for x in live["state"]],
            "level_lead": [int(x) for x in live["sel_lead"]], "level_follow": [int(x) for x in live["sel_follow"]],
            "levels": [L["label"] for L in LEVELS], "state_codes": {"0": "beat", "1": "no song (frozen)", "2": "DJ seam (frozen)"}}


def run_analysis(inp: Inputs, cfg: Config | None = None, with_sensitivity: bool = True) -> dict:
    """the full analysis -> the rhythm.json dict"""
    cfg = cfg or Config()
    t_start = time.time()
    res = analyse_one(inp, cfg)
    ev, final, live = res["events"], res["final"], res["live"]
    desc = descriptive(ev, inp, cfg)
    conf = confidence_summary(ev, inp.grid, inp, cfg, final)
    sens = sensitivity(inp, cfg) if with_sensitivity else None
    cols = ["t", "role", "foot", "valid", "measure", "slot", "kind", "offset_ms", "level", "level_offset_ms", "class", "lock", "accent", "accent_chance", "conf", "state", "sigma_ms"]
    t_end = float(inp.t1)
    doc = {
        "version": VERSION, "kind": "rhythm", "title": TITLE, "take": inp.meta.get("take"), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "time_base": "reference seconds (the capture's time_to_audio maps them to the audio clock)",
        "config": asdict(cfg), "inputs": {k: v for k, v in inp.meta.items()},
        "formula": {"score": "100 * sum_k weight_k * component_k", "weights": cfg.weights,
                    "components": "each 0..1, 0 = what random steps would give, 1 = perfect; live = (1 - share) * whole dance so far + share * exponential memory (half-life s)",
                    "live_share": cfg.recent_share, "half_life_s": cfg.half_life_s, "prior": f"neutral 0.5 worth {cfg.prior_events:g} events"},
        "final": {"score": _r(final["score"], 1), "components": {k: _r(final[k]) for k in COMPONENTS}, "weights": cfg.weights,
                  "n": {k: round(v, 1) for k, v in final["n"].items()}, "flow_R": round(final["R"], 3), "levels": {r: LEVELS[final["sel"][r]]["name"] for r in ROLES},
                  "confidence": conf, "range_under_uncertainty": sens["range"] if sens else None},
        "final_card": {"from_t": round(t_end - cfg.final_card_s, 3), "to_t": round(t_end, 3), "title": TITLE},
        "describe": desc,
        "steps": {"columns": cols, "rows": [[s.get(c) for c in cols] for s in ev.steps]},
        "measures": measure_table(inp, ev, cfg),
        "windows": window_table(inp, ev, cfg),
        "live": pack_live(live, cfg),
        "no_beat": [[round(max(a, inp.t0), 3), round(min(b, inp.t1), 3), k] for a, b, k in inp.grid.no_beat if b > inp.t0 and a < inp.t1],
        "music": {"source": inp.music.source if inp.music else None, "accents": int(len(inp.music.acc_t)) if inp.music else 0,
                  "breaks": [[round(a, 2), round(b, 2)] for a, b in (inp.music.breaks if inp.music else [])]},
        "sensitivity": sens,
        "claims": [],
        "seconds": round(time.time() - t_start, 2),
    }
    doc["claims"] = claims(desc, conf, final, inp, cfg)
    return doc


# --------------------------------------------------------------------------------------------------------------- the viewer's copy
COMPONENT_LABELS = {"lock": "Beat lock", "accents": "Accents", "tempo": "Tempo", "flow": "Flow", "sync": "Couple sync"}


def for_unity(doc: dict) -> dict:
    """The viewer's copy of rhythm.json (the capture's "rhythm" file, read by Assets/Overlays/RhythmScore.cs): the 30 Hz live timeline, the valid
    steps with the class the HUD flashes, the final card and the texts. Times are reference seconds (the viewer adds the capture's time_to_audio)."""
    cols = doc["steps"]["columns"]
    ci = {c: i for i, c in enumerate(cols)}
    rows = []
    for r in doc["steps"]["rows"]:
        if not r[ci["valid"]] or r[ci.get("class")] is None:
            continue
        rows.append([r[ci["t"]], 0 if r[ci["role"]] == "lead" else 1, {"on": 0, "near": 1, "off": 2}[r[ci["class"]]], r[ci["level_offset_ms"]]])
    w = doc["formula"]["weights"]
    f = doc["final"]
    return {
        "version": VERSION, "kind": "rhythm_view", "title": doc["title"], "take": doc.get("take"),
        "components": [{"key": k, "label": COMPONENT_LABELS[k], "weight": round(100 * w[k] / sum(w.values()))} for k in COMPONENTS],
        "final": {"score": f["score"], "components": f["components"], "confidence": f["confidence"]["label"], "confidence_value": f["confidence"]["value"],
                  "range": f.get("range_under_uncertainty")},
        "final_card": doc["final_card"],
        "live": doc["live"],
        "no_beat": doc["no_beat"],
        "steps": {"columns": ["t", "role", "class", "offset_ms"], "classes": ["on", "near", "off"], "roles": ["lead", "follow"], "rows": rows},
        "notes": doc["claims"][1:4],
    }


# --------------------------------------------------------------------------------------------------------------- synthetic takes (tests, selftest)
def synthetic_inputs(kind: str = "on_beat", seed: int = 0, jitter_ms: float = 20.0, bpm: float = 84.0, measures: int = 40, rate: str = "pulse",
                     offset_ms: float = 0.0) -> Inputs:
    """a perfect (or random, or half-bar-off) dance on a regular grid; both roles step together. kind: on_beat | random | off_beat | offset_consistent"""
    rng = np.random.default_rng(seed)
    P = 60.0 / bpm
    p8 = P / 2
    n = measures * 8
    t = 10.0 + np.arange(n) * p8
    rows = [[t[i], (1, 3, 3, 2, 1, 3, 2, 3)[i % 8], i // 8, i % 8] for i in range(n)]
    grid = build_grid(rows, no_beat_none=[(0.0, 10.0 - p8 / 2)], conf=np.ones(n), default_conf=1.0)
    steps = {"lead": [], "follow": []}
    base = t[:-1][::2] if rate == "pulse" else t[:-1]
    for r in ROLES:
        for i, b in enumerate(base):
            if kind == "on_beat":
                tm = b + rng.normal(0, jitter_ms / 1000.0)
            elif kind == "off_beat":
                tm = b + p8 / 2 + rng.normal(0, jitter_ms / 1000.0)
            elif kind == "offset_consistent":
                tm = b + offset_ms / 1000.0 + rng.normal(0, jitter_ms / 1000.0)
            else:
                tm = b + rng.uniform(-p8 / 2, p8 / 2)
            steps[r].append({"t": float(tm), "foot": "left" if i % 2 == 0 else "right", "sigma": 0.015})
    if kind == "random":
        steps = {r: [{"t": float(rng.uniform(t[0], t[-1])), "foot": "left", "sigma": 0.015} for _ in range(len(base))] for r in ROLES}
    music = Music(t[::2], np.where(np.arange(len(t[::2])) % 2 == 0, 1.0, 0.6), [], np.array([]), np.array([]), "grid")
    return Inputs(grid, steps, music, None, 8.0, float(t[-1]) + 2.0, {"take": f"synthetic_{kind}", "av_pinned": True})


def selftest() -> int:
    cfg = Config()
    bad = 0
    for kind in ("on_beat", "random", "off_beat"):
        inp = synthetic_inputs(kind)
        doc = run_analysis(inp, cfg, with_sensitivity=False)
        sc = doc["final"]["score"]
        print(f"{kind:10s} final {sc:5.1f}  components {doc['final']['components']}")
        if kind == "on_beat" and sc < 85:
            bad += 1
        if kind == "random" and sc > 10:
            bad += 1
        if kind == "off_beat" and sc > 35:
            bad += 1
    print("selftest", "OK" if not bad else "FAILED")
    return 1 if bad else 0


# --------------------------------------------------------------------------------------------------------------- plots
def make_plots(doc: dict, inp: Inputs, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = doc["steps"]["columns"]
    S = [dict(zip(cols, r)) for r in doc["steps"]["rows"]]
    V = [s for s in S if s["valid"]]
    live = doc["live"]
    t = live["t0"] + np.arange(live["n"]) / live["hz"]
    paths = []

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    p8 = inp.grid.p8 * 1000
    for r, c in (("lead", "#d9453d"), ("follow", "#3a6fd9")):
        d = [s["offset_ms"] for s in V if s["role"] == r]
        ax[0, 0].hist(d, bins=np.linspace(-p8 / 2, p8 / 2, 21), alpha=0.55, label=f"{r} (n={len(d)})", color=c)
    ax[0, 0].axhline(len(V) / 2 / 20, color="k", ls=":", lw=1, label="uniform (random steps)")
    for x in (-doc["config"]["on_ms"], doc["config"]["on_ms"]):
        ax[0, 0].axvline(x, color="g", lw=0.8)
    ax[0, 0].set_title("step offset to the nearest grid slot (ms)")
    ax[0, 0].legend(fontsize=8)
    for sg, m in (("beat", "o"), ("and", "s")):
        ax[0, 1].scatter([s["t"] for s in V if s["kind"] == sg], [s["offset_ms"] for s in V if s["kind"] == sg], s=9, marker=m, label=f"slot kind: {sg}")
    ax[0, 1].axhline(0, color="k", lw=0.6)
    ax[0, 1].set_title("offset over time")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(t, live["score"], "k", lw=1.6, label="live score")
    for k, c in zip(COMPONENTS, ("#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e")):
        ax[1, 0].plot(t, 100 * np.array(live[k]), lw=0.8, color=c, label=f"{k} x100", alpha=0.8)
    for a, b, k in doc["no_beat"]:
        ax[1, 0].axvspan(a, b, color="0.85")
    ax[1, 0].set_ylim(0, 100)
    ax[1, 0].set_title(f"live score and components (final {doc['final']['score']}, confidence {doc['final']['confidence']['label']})")
    ax[1, 0].legend(fontsize=7, ncol=3)
    ax[1, 1].plot(t, live["confidence"], color="#555")
    ax[1, 1].set_title("confidence")
    ax[1, 1].set_ylim(0, 1)
    fig.tight_layout()
    p = out_dir / "rhythm_review.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ms = doc["measures"]
    ax[0].bar([m["measure"] for m in ms], [m["score"] or 0 for m in ms], color="#6a8caf")
    ax[0].set_title("score per measure (descriptive, non-causal)")
    w = [x for x in doc["windows"] if x["score"] is not None]
    ax[1].plot([x["t"] for x in w], [x["score"] for x in w], color="#6a8caf")
    ax[1].set_title("sliding %g s window score" % doc["config"]["window_s"])
    fig.tight_layout()
    p = out_dir / "rhythm_measures.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    if doc.get("sensitivity"):
        rows = doc["sensitivity"]["rows"]
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = [f"{r['grid'].split(' ')[0]}\n{r['shift_ms']:+g} ms" for r in rows]
        ax.bar(range(len(rows)), [r["score"] for r in rows], color="#8aa")
        ax.set_xticks(range(len(rows)), labels, fontsize=7)
        ax.set_title("final score: audio-video shift x DJ seam handling")
        ax.set_ylim(0, max(60, max(r["score"] for r in rows) * 1.2))
        fig.tight_layout()
        p = out_dir / "rhythm_sensitivity.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------------------------------------------- CLI
def write_json(path: Path, doc: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
    tmp.replace(path)


def install_into(doc: dict, capture_dirs: list) -> list:
    """write the viewer's copy (for_unity) into capture folders and name it in their capture.json ("rhythm"); nothing else in the capture is touched.
    Returns the folders written."""
    done = []
    view = json.dumps(for_unity(doc), separators=(",", ":"))
    for d in capture_dirs:
        d = Path(d)
        mp = d / "capture.json"
        if not mp.exists():
            continue
        (d / "rhythm.json").write_text(view, encoding="utf-8", newline="\n")
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("rhythm") != "rhythm.json":
            m["rhythm"] = "rhythm.json"
            mp.write_text(json.dumps(m, indent=2), encoding="utf-8")
        done.append(str(d))
    return done


def run(take: str, timing: str | None = None, song_dir: str | None = None, audio: str | None = None, out: str | None = None, plots: bool = True,
        sensitivity_on: bool = True, work: Path | None = None) -> dict:
    from .take import Take
    w = Path(work) if work else Take.load(take).work
    cfg = Config()
    inp = load_inputs(take, Path(timing) if timing else None, Path(song_dir) if song_dir else None, Path(audio) if audio else None, cfg, w)
    doc = run_analysis(inp, cfg, with_sensitivity=sensitivity_on)
    out_dir = Path(out) if out else w / "rhythm"
    write_json(out_dir / "rhythm.json", doc)
    if plots:
        doc["plots"] = make_plots(doc, inp, out_dir)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.rhythm", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse a take: work/<take>/rhythm/rhythm.json")
    r.add_argument("take")
    r.add_argument("--timing", help="timing.json (default work/<take>/timing/timing.json)")
    r.add_argument("--song-dir", help="song/ folder with beats_merged.json, segments.json, beats_studio_raw.json (default work/<take>/song)")
    r.add_argument("--audio", help="soundtrack WAV on the reference clock (default work/<take>/audio/heat_studio.wav or audio_overlap.wav)")
    r.add_argument("--out", help="output folder (default work/<take>/rhythm)")
    r.add_argument("--no-plots", action="store_true")
    r.add_argument("--no-sensitivity", action="store_true")
    sub.add_parser("selftest", help="synthetic checks")
    i = sub.add_parser("install", help="copy work/<take>/rhythm/rhythm.json (the viewer's subset) into the capture folder(s) and name it in capture.json")
    i.add_argument("take")
    i.add_argument("--capture", required=True, help="capture folder name in the viewer's StreamingAssets (and work/<take>/export/)")
    i.add_argument("--no-viewer", action="store_true", help="only work/<take>/export/<capture>")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "install":
        from .take import Take
        from . import export_unity as eu
        w = Take.load(a.take).work
        path = w / "rhythm" / "rhythm.json"
        if not path.exists():
            print(f"error: {path} missing (python -m dancecap.rhythm run {a.take})", file=sys.stderr)
            return 2
        dirs = [w / "export" / a.capture] + ([] if a.no_viewer else [eu.UNITY_STREAMING / a.capture])
        done = install_into(json.loads(path.read_text(encoding="utf-8")), dirs)
        print("[rhythm] installed into", *(done or ["nothing (no capture.json in " + ", ".join(map(str, dirs)) + ")"]))
        return 0 if done else 1
    doc = run(a.take, a.timing, a.song_dir, a.audio, a.out, not a.no_plots, not a.no_sensitivity)
    f = doc["final"]
    print(f"[rhythm] {a.take}: final {f['score']} (confidence {f['confidence']['label']} {f['confidence']['value']}); components {f['components']}")
    if doc.get("sensitivity"):
        print(f"[rhythm] range under +-{doc['config']['av_uncertainty_ms']:g} ms and the DJ seam variants: {doc['sensitivity']['range']}")
    for c in doc["claims"]:
        print("  -", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
