"""Rhythm analysis of a heat or a take with a beat grid, and a causal LIVE rhythm score (agents S6 + S7, 2026-10-11).

    python -m dancecap.rhythm run <take> [--timing P] [--song-dir P] [--audio WAV] [--out DIR] [--no-plots] [--no-sensitivity]
    python -m dancecap.rhythm install <take> --capture <NN_Name>   # the viewer's copy into the capture folder(s), nothing else touched
    python -m dancecap.rhythm selftest                 # the synthetic checks
    python -m dancecap.rhythm controls                 # positive / negative / no-dance controls of the metric (work/<take>/rhythm/controls.json)

What it is (version 2). A "rhythm entrainment" estimate from MOTION: how strongly the dancers' movement is phase-locked to the song's beat grid, at whatever metrical level
it locks to (the half bar = one weight transfer per two pulses, the pulse, three in two pulses, the eighth). Five components, each 0 = what a metronome at the wrong tempo
would give, 1 = the reference level (the 90th percentile of the professional demo's first half; docs/RHYTHM.md, "Reference levels"):

    score = 100 * (0.45 body sway + 0.15 footwork + 0.20 recent form + 0.10 couple sync + 0.10 accents)

  * BODY SWAY: phase-locking of the centre-of-mass sway (horizontal speed = weight transfer; height = bounce) to the grid at the best level, over a 45 s causal window,
    against a permutation baseline of metronomes at the wrong tempo (dancecap/rhythm_entrain.py).      * FOOTWORK: the same for the foot touchdowns.
  * RECENT FORM: the body lock over the last ~15 s (the number follows the last bars).        * COUPLE SYNC: the lead's and the follow's locked oscillations stay in step.
  * ACCENTS: stops, hits (sudden decelerations) and steps on the music's strong onsets (kernel 75 ms), against the same statistic with the music moved by seconds.
The LIVE score at time t uses only data up to t (a pure function of time: the viewer's HUD seeks), starts at a neutral 50, moves to the estimate as the evidence accumulates, is
FROZEN in the no-beat stretches (no song, the DJ seam) and says how sure it is (lock evidence against the baseline). The FINAL score is the whole-dance aggregate with a noise-floor
interval. It is an ESTIMATE FROM MOTION, not an official score, and nothing validates a mapping to judges' scores.

Compute: numpy + scipy, about 15 s for a 200 s heat (of which about 4 s are the soundtrack's STFT): local (the pods rule is for CV / neural / heavy CPU work).
Outputs (work/<take>/rhythm/): rhythm.json, plots, and the Unity capture gets a copy (export_unity: capture.json "rhythm") for the viewer's rhythm layer.
Inputs: timing/timing.json (footsteps + beat grid), physics/physics.npz (centre of mass; else the pelvis of timing/kinematics.npz), optionally song/ (confidence, no-beat
stretches, DJ seam variants) and a soundtrack WAV on the reference clock (accents, breaks).
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

from . import rhythm_entrain as ent

VERSION = 2
TITLE = "Rhythm entrainment (estimated from motion, not an official score)"
ROLES = ("lead", "follow")
COMPONENTS = ("body", "feet", "recent", "couple", "accents")
COMPONENT_LABELS = {"body": "Body sway", "feet": "Footwork", "recent": "Recent form", "couple": "Couple sync", "accents": "Accents"}
LEVEL_LABELS = ent.LEVEL_LABELS


# --------------------------------------------------------------------------------------------------------------- configuration
@dataclass
class Config:
    # step flash classes (HUD lamps, floor rings): a step is "on" when it sits within on_ms of the dancer's own locked step phase at the chosen level, "near" within near_ms
    on_ms: float = 100.0
    near_ms: float = 200.0
    flash_memory_s: float = 15.0
    flash_min_skill: float = 0.15       # the step stream must show a lock at least this strong before a step is coloured (else neutral)
    # component weights (sum 1)
    weights: dict = field(default_factory=lambda: {"body": 0.45, "feet": 0.15, "recent": 0.20, "couple": 0.10, "accents": 0.10})
    # the entrainment engine (docs/RHYTHM.md "Reference levels")
    ent: ent.EntrainConfig = field(default_factory=ent.EntrainConfig)
    # accents: onset events of at least this strength (1 = the local 90th percentile of onset peaks) count; credit kernel sigma (ms)
    accent_min_strength: float = 0.5
    accent_sigma_ms: float = 75.0
    accent_chance_window_s: float = 4.0
    accent_memory_s: float = 25.0
    accent_prior_events: float = 5.0
    ref_accent: float = 0.038           # mean accent credit above chance that reads 1.0 (the 90th percentile of the demo's first half)
    accent_min_span: float = 0.02       # the smallest ref - baseline span (credit units) the accents are scaled by
    hit_share: float = 0.5              # a hit = a deceleration peak above this share of the dancer's 90th percentile deceleration
    # breaks in the music (dB below the local median level, minimum length) and the motion that counts as still
    break_db: float = 6.0
    break_min_s: float = 0.25
    stop_ratio: float = 0.4             # a stop = the body speed under stop_ratio x the dancer's median speed for stop_min_s
    stop_min_s: float = 0.3
    norm_window_s: float = 60.0         # the stop / hit thresholds follow the dancer's own trailing 60 s (causal)
    live_smooth_s: float = 8.0          # the live number follows the estimate through a causal exponential smoothing of this time constant
    neutral: float = 0.5                # the live number starts here and moves to the estimate as the evidence accumulates
    # confidence
    av_pinned: bool = False             # an audio-video sync event pins the camera clocks to the soundtrack (timing.json absolute_timing.valid)
    av_uncertainty_ms: float = 30.0
    av_cap_unpinned: float = 0.95       # the body measures are phase-locks at 0.3-1.3 Hz: a +-30 ms clock offset is a few degrees; absolute touchdown claims would need 0.85
    grid_confidence_default: float = 0.8
    evidence_z_full: float = 5.0        # the lock's whole-dance excess over the wrong-tempo baseline in standard errors (independent windows) that counts as full evidence
    evidence_z_min: float = 2.0         # under this the HUD says "insufficient beat lock evidence"
    min_valid_s: float = 45.0
    final_skip_s: float = 15.0          # the first seconds of a grid segment (the averages are still filling) are left out of the final aggregate
    # outputs
    live_hz: float = 30.0
    window_s: float = 12.0
    final_card_s: float = 4.0


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
        return Grid(self.t + dt, self.slot, self.type, self.measure, self.conf, self.song, [(a + dt, b + dt, k) for a, b, k in self.no_beat], self.seam_note)


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
    body: ent.Body | None = None


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
    rows = d.get("beats", {}).get("grid")
    if not rows:
        raise ValueError("no beat grid in " + str(path) + ": nothing to score the dancers' rhythm against (a take without a soundtrack has an undefined rhythm score)")
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
        n = len(t_new)
        slot = np.arange(n) % 8
        out[name] = Grid(t_new, slot, np.full(n, 3), np.arange(n) // 8, np.full(n, base.conf.mean()), [""] * n,
                         [(a, b, k) for a, b, k in base.no_beat if k == "none"], f"{name}: the {first if name == 'seam_first' else second} song's pulses through the DJ mix")
        out[name].conf = np.full(n, np.nan)
    for name, g in out.items():
        c = np.interp(g.t, base.t, np.nan_to_num(base.conf, nan=cfg.grid_confidence_default))
        in_fill = (g.t > a0 - 1e-6) & (g.t < b0 + 1e-6) | (name == "seam_second") & (g.t >= mix_a) & (g.t < b0)
        g.conf = np.where(in_fill, 0.5 * c, c)
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
    pt_ = tref[pk]
    lo_i, hi_i = np.searchsorted(pt_, pt_ - 12.0), np.searchsorted(pt_, pt_ + 12.0, side="right")
    ref = np.array([np.percentile(h[a:b], 90) for a, b in zip(lo_i, hi_i)]) if len(pk) else np.array([])
    s = np.clip(h / np.maximum(ref, 1e-9), 0.0, 1.0)
    at = tref[pk]
    keep = np.ones(len(at), bool)
    for a, b in exclude:
        keep &= ~((at >= a) & (at < b))
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
        t_lo, t_hi = rows[0][0], rows[-1][0]
        none = [x for x in dropped if x[1] <= t_lo + 1.0 or x[0] >= t_hi - 1.0]
        seam = [x for x in dropped if x not in none]
    grid = build_grid(rows, no_beat_none=none, explicit_gaps=seam, conf=None, song=sd["song"], default_conf=cfg.grid_confidence_default)
    if conf is not None:
        grid.conf = np.where(np.isnan(conf), cfg.grid_confidence_default, conf)
        grid.song = sd["song"] or grid.song
    t0a = 0.0
    if audio is None:
        for cand in (work / "audio" / "heat_studio.wav", work / "audio" / "audio_overlap.wav", work / "song" / "studio_reference.wav"):
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
    body = ent.load_body(work)
    none_all = [(a, b) for a, b, k in grid.no_beat if k == "none"]
    music = music_from_audio(Path(audio), t0a, cfg, exclude=none_all) if audio is not None and Path(audio).exists() else None
    if music is None:
        music = music_from_grid(grid)
    if body is not None:
        t0, t1 = float(body.t[0]), float(body.t[-1])
    elif motion is not None:
        t0, t1 = float(motion.t[0]), float(motion.t[-1])
    else:
        t0 = min(min((s["t"] for s in v), default=1e9) for v in steps.values()) - 1.0
        t1 = max(max((s["t"] for s in v), default=0) for v in steps.values()) + 1.0
    cfg.av_pinned = bool(meta.get("av_pinned"))
    meta.update({"take": take, "audio": str(audio) if audio else None, "song_dir": str(song_dir) if song_dir else None, "music_source": music.source,
                 "body_source": body.source if body is not None else None})
    inp = Inputs(grid, steps, music, motion, t0, t1, meta, body=body)
    inp.variants = seam_variants(song_dir, grid, cfg)
    return inp


# --------------------------------------------------------------------------------------------------------------- events: the steps table and the accents
@dataclass
class Events:
    accent: list = field(default_factory=list)     # t, role, kind (step|stop|hit), credit, chance, conf
    hold: list = field(default_factory=list)
    steps: list = field(default_factory=list)      # the per-step table rows


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


def _trailing_percentile(x: np.ndarray, fs: float, win_s: float, q: float, step_s: float = 0.5) -> np.ndarray:
    """the q-th percentile of x over the trailing win_s seconds (at least 5 s are needed; before that the samples so far), evaluated every step_s and held: a causal normalisation"""
    n = len(x)
    out = np.zeros(n)
    w = int(win_s * fs)
    step = max(1, int(step_s * fs))
    last = float(np.percentile(x[:max(2, int(5 * fs))], q))
    for i in range(0, n, step):
        seg = x[max(0, i - w + 1):i + 1]
        if len(seg) >= int(5 * fs):
            last = float(np.percentile(seg, q))
        out[i:i + step] = last
    return out


def _runs(mask: np.ndarray):
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def build_events(inp: Inputs, cfg: Config, grid: Grid | None = None, av_shift_s: float = 0.0) -> Events:
    """The per-step table (nearest eighth slot and its signed offset: a description, not a score) and the accent events: steps, stops, hits (sudden decelerations) on the
    music's accents, holds on its breaks. `av_shift_s` moves the music against the motion: a step at motion time t is compared with the music at t + av_shift."""
    grid = grid or inp.grid
    ev = Events()
    music = inp.music
    cr_t = cr = cr_mean = None
    if music is not None and len(music.acc_t):
        cr_t, cr, cr_mean = accent_credit_curve(music, cfg, inp.t0, inp.t1)
    p8_ms = grid.p8 * 1000.0

    def credit_at(t):
        if cr_t is None:
            return 0.0, 0.0
        i = int(np.clip(round((t - cr_t[0]) * 100.0), 0, len(cr_t) - 1))
        return float(cr[i]), float(cr_mean[i])

    for r in ROLES:
        for s in sorted(inp.steps.get(r, []), key=lambda x: x["t"]):
            tm = s["t"]
            tg = tm + av_shift_s
            st = grid.state_at(tg)
            row = {"t": round(tm, 4), "role": r, "foot": s.get("foot", ""), "state": st, "sigma_ms": round(s.get("sigma", 0.02) * 1000.0, 1)}
            k = _nearest_slot(grid, tg) if len(grid) else 0
            if st != 0 or abs((tg - grid.t[k]) * 1000.0) > 0.75 * p8_ms:
                row["valid"] = False
                ev.steps.append(row)
                continue
            d_ms = (tg - grid.t[k]) * 1000.0
            slot = int(grid.slot[k])
            conf = float(grid.conf[k])
            a_cr, a_ch = credit_at(tg)
            row.update({"valid": True, "measure": int(grid.measure[k]), "slot": slot, "kind": "beat" if slot % 2 == 0 else "and", "offset_ms": round(d_ms, 1),
                        "accent": round(a_cr, 3), "accent_chance": round(a_ch, 3), "conf": round(conf, 3)})
            ev.steps.append(row)
            ev.accent.append({"t": tm, "role": r, "kind": "step", "credit": a_cr, "chance": a_ch, "conf": conf})
    # stops and hits (need the motion), holds on the music's breaks
    if inp.motion is not None and len(inp.motion.t):
        mt = inp.motion.t
        fps = 1.0 / float(np.median(np.diff(mt)))
        for r in ROLES:
            sp = np.convolve(inp.motion.speed[r], np.ones(5) / 5.0, "same")
            thr = cfg.stop_ratio * _trailing_percentile(sp, fps, cfg.norm_window_s, 50.0)
            for i, j in _runs(sp < thr):
                if (j - i) / fps >= cfg.stop_min_s and i > int(0.3 * fps) and sp[max(0, i - int(0.3 * fps)):i].max() > 1.5 * thr[i]:
                    tm = float(mt[i])
                    tg = tm + av_shift_s
                    if grid.state_at(tg) == 0 and grid.t[0] <= tg <= grid.t[-1]:
                        a_cr, a_ch = credit_at(tg)
                        ev.accent.append({"t": tm, "role": r, "kind": "stop", "credit": a_cr, "chance": a_ch, "conf": float(grid.conf[_nearest_slot(grid, tg)])})
        body = inp.body
        if body is not None:
            from scipy.signal import butter, find_peaks, sosfilt
            fs = 1.0 / float(np.median(np.diff(body.t)))
            lag = 0.045                                                    # the causal 5 Hz low-pass' delay, taken off the event time
            for r in ROLES:
                v = sosfilt(butter(2, 5.0 / (fs / 2.0), "low", output="sos"), body.sig[r]["move"])
                dec = np.maximum(0.0, -np.r_[0.0, np.diff(v)] * fs)
                ref = _trailing_percentile(dec, fs, cfg.norm_window_s, 90.0)
                pk, _ = find_peaks(dec, distance=max(1, int(0.25 * fs)))
                pk = [i for i in pk if dec[i] >= cfg.hit_share * ref[i] and ref[i] > 0]
                for i in pk:
                    tm = float(body.t[i]) - lag
                    tg = tm + av_shift_s
                    if grid.state_at(tg) == 0 and grid.t[0] <= tg <= grid.t[-1]:
                        a_cr, a_ch = credit_at(tg)
                        ev.accent.append({"t": tm, "role": r, "kind": "hit", "credit": a_cr, "chance": a_ch, "conf": float(grid.conf[_nearest_slot(grid, tg)])})
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
                    ev.hold.append({"t": bt0, "t_known": bt1, "role": r, "credit": q, "chance": chance, "conf": float(grid.conf[_nearest_slot(grid, b0)]), "break": [round(b0, 3), round(b1, 3)]})
    ev.accent.sort(key=lambda e: e["t"])
    for e in ev.accent:
        e["t_known"] = e["t"] + 0.05
    return ev


def accent_series(ev: Events, t: np.ndarray, valid: np.ndarray, cfg: Config) -> tuple[np.ndarray, dict]:
    """The raw accents statistic on the sample grid t: the exponentially weighted mean of (credit - chance) over the stops, hits and steps known by t (memory cfg.accent_memory_s of
    beat time), with a prior of cfg.accent_prior_events events at chance. NaN where not beat-covered. Positive = the body's accents sit on the music's strong onsets more than chance."""
    dt = float(np.median(np.diff(t)))
    dec = math.exp(-dt / cfg.accent_memory_s)
    raw = np.full(len(t), np.nan)
    num = den = 0.0
    ia = 0
    acc = ev.accent
    n_ev = 0
    for j, tj in enumerate(t):
        if not valid[j]:
            continue
        num *= dec
        den *= dec
        while ia < len(acc) and acc[ia]["t_known"] <= tj:
            e = acc[ia]
            num += e["credit"] - e["chance"]
            den += 1.0
            n_ev += 1
            ia += 1
        raw[j] = num / (den + cfg.accent_prior_events)
    return raw, {"n_events": n_ev}


# --------------------------------------------------------------------------------------------------------------- one analysis: series, live, final
@dataclass
class Analysis:
    S: ent.Series
    K: ent.Skills
    comps: dict               # component -> (n,) skill series (cumulative since the song's start for body / feet / couple), NaN where not beat-covered
    wt: np.ndarray            # (n,) weight of the evidence against the prior in the live number (0 at the start of a grid segment, ->1 over ~tau seconds)
    noise: dict               # component -> (n,) 1-sigma noise of the skill under the wrong-tempo baseline
    acc_info: dict
    events: Events
    t: np.ndarray
    valid: np.ndarray
    elapsed: np.ndarray
    z: np.ndarray             # (n,) mean over roles of the body lock's excess over the baseline in baseline standard deviations
    ends: np.ndarray          # sample indices where a beat-covered grid segment ends
    cfg_hash: dict = field(default_factory=dict)


def segment_ends(valid: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
    n = len(valid)
    out = [i for i in range(n) if valid[i] and (i == n - 1 or not valid[i + 1] or elapsed[i + 1] < elapsed[i])]
    return np.array(out, int)


def _time_base(inp: Inputs, cfg: Config) -> ent.Body:
    """the body (or, without one, an empty time base) the entrainment is evaluated on"""
    if inp.body is not None:
        return inp.body
    n = int(math.floor((inp.t1 - inp.t0) * cfg.live_hz + 1e-6)) + 1
    return ent.Body(inp.t0 + np.arange(n) / cfg.live_hz, None, "none")


def run_series(inp: Inputs, cfg: Config, grid: Grid | None = None, av_shift_s: float = 0.0) -> Analysis:
    grid = (grid or inp.grid).shifted(-av_shift_s) if av_shift_s else (grid or inp.grid)
    body = _time_base(inp, cfg)
    S = ent.analyse(grid, body, inp.steps, cfg.ent)
    K = ent.normalise(S, ent.baseline_set(grid, body, inp.steps, cfg.ent), cfg.ent)
    ev = build_events(inp, cfg, inp.grid, av_shift_s)
    raw, info = accent_series(ev, body.t, S.valid, cfg)
    # the accents' permutation baseline: the same statistic with the music moved against the motion by seconds (the onsets are not periodic, so the shifted music is unrelated to the motion)
    nulls = [accent_series(build_events(inp, cfg, inp.grid, av_shift_s + sh), body.t, S.valid, cfg)[0] for sh in ACCENT_SHIFTS_S]
    fs = 1.0 / float(np.median(np.diff(body.t)))
    acc, ab, asd, anz = ent.calibrated(raw, nulls, cfg.ref_accent, S.valid, fs, cfg.ent.base_smooth_s, min_span=cfg.accent_min_span, margin_sd=cfg.ent.base_margin_sd)
    wt = 1.0 - np.exp(-S.elapsed / cfg.ent.tau_long_s)
    comps = {"body": K.body, "feet": K.feet, "recent": K.recent, "couple": K.couple, "accents": acc}
    noise = {**K.noise_sd, "accents": anz}
    z = ent._nan(np.nanmean, np.vstack([K.z_body[r] for r in ROLES]), 0)
    return Analysis(S, K, comps, wt, noise, info, ev, body.t, S.valid, S.elapsed, z, segment_ends(S.valid, S.elapsed))


CUMULATIVE = ("body", "feet", "couple")
ACCENT_SHIFTS_S = (-5.3, -3.7, -2.3, 2.9, 4.1, 5.9)


def aggregate(A: Analysis, cfg: Config) -> dict:
    """The whole-dance components: the mean of each running component over the beat-covered time past the first cfg.final_skip_s of every song (NaN where unknown)."""
    out = {}
    m = A.valid & (A.elapsed >= cfg.final_skip_s)
    for k in COMPONENTS:
        v = A.comps[k][m]
        v = v[np.isfinite(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
    out["score"] = score_of(out, cfg)
    out["n_samples"] = int(np.sum(A.valid))
    return out


def score_of(comp: dict, cfg: Config) -> float:
    """100 x the weighted components; a component with nothing seen (NaN) is left out and the weights renormalised"""
    ok = [k for k in COMPONENTS if k in comp and not math.isnan(comp[k])]
    if not ok:
        return float("nan")
    wsum = sum(cfg.weights[k] for k in ok)
    return float(100.0 * sum(cfg.weights[k] * comp[k] for k in ok) / wsum)


def noise_interval(A: Analysis, cfg: Config, comp: dict) -> tuple[float, float, float]:
    """The 90 % interval of the score's NOISE FLOOR: the estimators' standard deviations under the wrong-tempo baseline (what chance alone does to each component, averaged over the
    time seen and divided by the square root of the number of independent windows), propagated through the weights. It does not include the systematic uncertainties (clock offset,
    DJ seam: `sensitivity`) nor the unknown validity of the reference levels."""
    m = A.valid & (A.elapsed >= cfg.final_skip_s)
    ok = [k for k in COMPONENTS if k in comp and not math.isnan(comp[k])]
    wsum = sum(cfg.weights[k] for k in ok) or 1.0
    n_win = max(1.0, float(m.sum()) / (cfg.live_hz * cfg.ent.tau_long_s))
    var = 0.0
    for k in ok:
        v = A.noise[k][m]
        v = v[np.isfinite(v)]
        s = float(v.mean()) / math.sqrt(n_win) if len(v) else 0.0
        var += (cfg.weights[k] / wsum * s) ** 2
    sd = 100.0 * math.sqrt(var)
    score = comp["score"]
    return max(0.0, score - 1.645 * sd), min(100.0, score + 1.645 * sd), sd


def live_timeline(inp: Inputs, A: Analysis, cfg: Config, grid: Grid | None = None) -> dict:
    """The causal live score at the body time base's rate: component(t) = (1 - wt) x the starting value + wt x the skill estimated from everything since the song's start (and, for the
    recent form, the last ~tau seconds). The starting value is a neutral 0.5 at the start of the dance and the previous song's last value after a DJ seam. wt grows 0 -> 1 over ~tau
    seconds of beat-covered motion, so the number starts neutral and settles as the evidence comes. In the no-beat stretches nothing is added and the number is frozen.
    Everything is a function of the samples up to t (the series are causal), so the HUD can seek."""
    grid = grid or inp.grid
    t = A.t
    n = len(t)
    out = {k: np.zeros(n) for k in COMPONENTS}
    out.update({k: np.zeros(n) for k in ("score", "confidence", "evidence")})
    state = np.zeros(n, int)
    gconf = np.interp(t, grid.t, grid.conf)
    cur = {k: cfg.neutral for k in COMPONENTS}
    start = dict(cur)
    prev_el = -1.0
    conf_k = evid = 0.0
    dt = float(np.median(np.diff(t)))
    valid_s = 0.0
    zsum = zn = 0.0
    sm: dict = {k: None for k in COMPONENTS}
    alpha = 1.0 - math.exp(-dt / cfg.live_smooth_s)
    last_score = score_of(cur, cfg)
    for j in range(n):
        st = grid.state_at(float(t[j]))
        state[j] = st
        if st == 0 and A.valid[j]:
            if A.elapsed[j] < prev_el or prev_el < 0:
                start = dict(cur)                                       # a new song: continue from the number on screen
                sm = {k: None for k in COMPONENTS}
            prev_el = A.elapsed[j]
            valid_s += dt
            for k in COMPONENTS:
                v = A.comps[k][j]
                if np.isfinite(v):
                    sm[k] = float(v) if sm[k] is None else sm[k] + (float(v) - sm[k]) * alpha            # a causal smoothing of the estimate (cfg.live_smooth_s)
                    wj = A.wt[j] if k != "accents" else min(1.0, A.wt[j] * 1.5)
                    cur[k] = (1.0 - wj) * start[k] + wj * sm[k]
            last_score = score_of(cur, cfg)
            zj = A.z[j]
            if A.elapsed[j] >= cfg.final_skip_s and np.isfinite(zj):
                zsum += float(zj)
                zn += 1.0
            z_tot = (zsum / zn) * math.sqrt(max(1.0, zn / (cfg.live_hz * cfg.ent.tau_long_s))) if zn > 0 else 0.0
            evid = float(np.clip(z_tot / cfg.evidence_z_full, 0.0, 1.0))
            samp = min(1.0, valid_s / cfg.min_valid_s) ** 0.5
            av = 1.0 if cfg.av_pinned else cfg.av_cap_unpinned
            conf_k = samp * max(0.25, evid) * av
        for k in COMPONENTS:
            out[k][j] = cur[k]
        out["score"][j] = last_score
        c = min(1.0, max(0.0, float(gconf[j]))) * conf_k
        out["confidence"][j] = 0.0 if st == 1 else c * 0.5 if st == 2 else c
        out["evidence"][j] = evid
    out["t"] = t
    out["state"] = state
    out["level_lead"] = A.S.best_body["lead"]
    out["level_follow"] = A.S.best_body["follow"]
    return out


# --------------------------------------------------------------------------------------------------------------- the step table: flashes against the dancer's own locked step rhythm
def annotate_steps(inp: Inputs, A: Analysis, ev: Events, cfg: Config, grid: Grid | None = None):
    """Each valid step's class for the HUD lamps and the floor rings: its offset (ms) from the dancer's OWN step phase at the level their steps lock to best (the circular mean of the
    previous steps at that level, exponential memory cfg.flash_memory_s of step time; causal), against the music's metrical level. on <= cfg.on_ms, near <= cfg.near_ms, else off;
    neutral while the footwork shows no lock against the baseline (skill under cfg.flash_min_skill) or fewer than 3 steps have been seen."""
    grid = grid or inp.grid
    S = A.S
    t = A.t
    P = ent.pulse_period(grid, grid.t[0], grid.t[-1])
    for r in ROLES:
        rows = [(i, s) for i, s in enumerate(ev.steps) if s["role"] == r and s.get("valid")]
        prev: list = []
        for i, s in rows:
            j = int(np.clip(np.searchsorted(t, s["t"]) - 1, 0, len(t) - 1))
            hi = int(S.best_steps[r][j])
            h = ent.LEVEL_H[hi]
            ph = float(ent.grid_phase(grid, np.array([s["t"]]))[0])
            th = 2.0 * math.pi * h * ph / 4.0
            z = 0j
            for (tp, thp) in prev:
                z += math.exp(-(s["t"] - tp) / cfg.flash_memory_s) * complex(math.cos(thp), math.sin(thp))
            sk = A.K.feet_role[r][j]
            skill = float(sk) if np.isfinite(sk) else 0.0
            per_ms = 4.0 * P / h * 1000.0
            if len(prev) >= 3 and abs(z) > 1e-9 and skill >= cfg.flash_min_skill:
                res = ((th - math.atan2(z.imag, z.real) + math.pi) % (2 * math.pi) - math.pi) / (2 * math.pi) * per_ms
                on, near = min(cfg.on_ms, 0.16 * per_ms), min(cfg.near_ms, 0.32 * per_ms)
                s["class"] = "on" if abs(res) <= on else "near" if abs(res) <= near else "off"
                s["level_offset_ms"] = round(res, 1)
                s["lock"] = round(math.exp(-0.5 * (res / cfg.on_ms) ** 2), 3)
            else:
                s["class"] = "neutral"
                s["level_offset_ms"] = None
                s["lock"] = None
            s["level"] = ent.LEVEL_NAMES[hi]
            prev.append((s["t"], th))
            if len(prev) > 60:
                prev.pop(0)
    for s in ev.steps:
        if not s.get("valid"):
            s.setdefault("class", None)


# --------------------------------------------------------------------------------------------------------------- tables
def _running(A: Analysis, cfg: Config, a: float, b: float) -> dict:
    """the mean of the running component values over [a, b) (descriptive tables only)"""
    m = (A.t >= a) & (A.t < b) & A.valid
    out = {}
    for k in COMPONENTS:
        v = A.comps[k][m]
        v = v[np.isfinite(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
    out["score"] = score_of(out, cfg)
    return out


def measure_table(inp: Inputs, A: Analysis, ev: Events, cfg: Config, grid: Grid | None = None) -> list:
    grid = grid or inp.grid
    out = []
    for m in np.unique(grid.measure):
        idx = np.where(grid.measure == m)[0]
        if len(idx) < 8:
            continue
        a, b = float(grid.t[idx[0]] - grid.p8 / 2), float(grid.t[idx[-1]] + grid.p8 / 2)
        sub = [s for s in ev.steps if s.get("valid") and a <= s["t"] < b]
        comp = _running(A, cfg, a, b)
        out.append({"measure": int(m), "t0": round(a, 3), "t1": round(b, 3), "song": grid.song[idx[0]] if grid.song else "",
                    "n_lead": sum(1 for s in sub if s["role"] == "lead"), "n_follow": sum(1 for s in sub if s["role"] == "follow"),
                    "steps_per_pulse": {r: round(sum(1 for s in sub if s["role"] == r) / 4.0, 3) for r in ROLES},
                    "score": _r(comp["score"], 1), "components": {k: _r(comp[k]) for k in COMPONENTS}, "conf": round(float(np.mean(grid.conf[idx])), 3)})
    return out


def window_table(inp: Inputs, A: Analysis, cfg: Config) -> list:
    out = []
    half = cfg.window_s / 2
    t = inp.t0 + half
    while t <= inp.t1 - half + 1e-6:
        a, b = t - half, t + half
        m = (A.t >= a) & (A.t < b)
        beat = float(np.mean(A.valid[m])) if m.any() else 0.0
        comp = _running(A, cfg, a, b)
        out.append({"t": round(t, 2), "score": _r(comp["score"], 1) if beat > 0.3 else None, "beat_share": round(beat, 2), "components": {k: _r(comp[k]) for k in COMPONENTS}})
        t += 2.0
    return out


def _r(x, p: int = 3):
    """round, with an unknown (NaN) value as None (valid JSON)"""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), p)


def _tmean(x: np.ndarray, m: np.ndarray):
    v = x[m]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else None


def entrainment_table(A: Analysis, cfg: Config) -> dict:
    """what the entrainment looks like over the whole dance: per role and metrical level the mean phase-locking of the body streams and of the foot touchdowns (running values past the
    first seconds of each song), the share of time each level was the chosen one, the wrong-tempo baseline of the chosen-level statistic and its excess in baseline standard deviations"""
    S, K = A.S, A.K
    m = A.valid & (A.elapsed >= cfg.final_skip_s)
    out: dict = {"levels": list(ent.LEVEL_LABELS), "baseline_scales": list(ent.NULL_SCALES)}
    for r in ROLES:
        ent_r = {}
        for g, arr, best, key in (("body", S.cum_body[r], S.best_body[r], "body"), ("steps", S.cum_steps[r], S.best_steps[r], "steps")):
            lv = {nm: {"mean_rho": _r(_tmean(arr[hi], m)), "chosen_share": _r(float(np.mean(best[m] == hi)), 2)} for hi, nm in enumerate(ent.LEVEL_NAMES)}
            b, sd = K.baseline[(key, r)]
            ent_r[g] = {"levels": lv, "best_level_rho_mean": _r(_tmean(K.raw[(key, r)], m)), "baseline_mean": _r(_tmean(b, m)),
                        "excess_sd": _r(_tmean((K.raw[(key, r)] - b) / np.maximum(sd, 0.02), m), 2)}
        ph = {}
        for hi, nm in enumerate(ent.LEVEL_NAMES):
            pb = S.phase_body[r][hi][m]
            pb = pb[np.isfinite(pb)]
            if len(pb):
                ph[nm] = _r(float(np.angle(np.exp(2j * np.pi * pb).mean()) / (2 * np.pi)), 3)
        ent_r["body_phase_cycles"] = ph
        out[r] = ent_r
    out["couple_coherence_mean"] = _r(_tmean(S.couple, m))
    return out


def descriptive(ev: Events, inp: Inputs, cfg: Config) -> dict:
    """distribution numbers of the per-step table (valid steps): the old slot-exact view, kept as a DESCRIPTION (it is not scored)"""
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
        out[r] = {"n": n, "binary_grid": {"within_60ms": round(within(60.0), 3), "within_100ms": round(within(100.0), 3), "chance_60ms": round(2 * 60.0 / p8, 3),
                                          "chance_100ms": round(2 * 100.0 / p8, 3), "on_pulse_slot": round(float(np.mean([x["kind"] == "beat" for x in s])), 3),
                                          "mean_offset_ms": round(float(d.mean()), 1), "sd_offset_ms": round(float(d.std()), 1), "R_eighth": round(R, 3),
                                          "R_chance": round(1.0 / math.sqrt(n), 3), "rayleigh_p": round(float(math.exp(-n * R * R)), 4)}}
    kinds: dict = {}
    for e in ev.accent:
        d = kinds.setdefault(e["kind"], [0, 0.0])
        d[0] += 1
        d[1] += e["credit"] - e["chance"]
    out["accent_events"] = {k: {"n": v[0], "mean_credit_above_chance": round(v[1] / max(v[0], 1), 4)} for k, v in kinds.items()}
    return out


def confidence_summary(A: Analysis, grid: Grid, inp: Inputs, cfg: Config, final: dict) -> dict:
    m = A.valid & (A.elapsed >= cfg.final_skip_s)
    valid_s = float(np.sum(A.valid)) / cfg.live_hz
    zv = A.z[m]
    zv = zv[np.isfinite(zv)]
    n_win = max(1.0, float(m.sum()) / (cfg.live_hz * cfg.ent.tau_long_s))
    z = float(zv.mean()) * math.sqrt(n_win) if len(zv) else 0.0                  # the mean excess over the wrong-tempo baseline in units of the standard error of independent windows
    gc = float(np.mean(grid.conf)) if len(grid) else 0.0
    evidence = float(np.clip(z / cfg.evidence_z_full, 0.0, 1.0))
    av = 1.0 if cfg.av_pinned else cfg.av_cap_unpinned
    samp = min(1.0, valid_s / cfg.min_valid_s) ** 0.5
    conf = gc * max(0.25, evidence) * av * samp
    label = "high" if conf >= 0.7 else "medium" if conf >= 0.4 else "low"
    return {"value": round(conf, 3), "label": label, "grid_confidence_mean": round(gc, 3), "lock_evidence_z": round(z, 2), "evidence_factor": round(evidence, 3), "av_factor": av,
            "beat_covered_s": round(valid_s, 1), "insufficient_evidence": bool(z < cfg.evidence_z_min)}


def sensitivity(inp: Inputs, cfg: Config) -> dict:
    """The final score under the +-30 ms audio-video uncertainty and with the DJ seam handled the other ways (the whole analysis re-run)"""
    rows = []
    grids = {"default (seam frozen)": inp.grid}
    grids.update(inp.variants)
    for gname, g in grids.items():
        for sh in (-cfg.av_uncertainty_ms, 0.0, cfg.av_uncertainty_ms):
            A = run_series(inp, cfg, g, sh / 1000.0)
            f = aggregate(A, cfg)
            rows.append({"grid": gname, "shift_ms": sh, "score": _r(f["score"], 1), "components": {k: _r(f[k]) for k in COMPONENTS}})
    sc = [r["score"] for r in rows if r["score"] is not None] or [None]
    shift_only = [r["score"] for r in rows if r["grid"].startswith("default") and r["score"] is not None] or [None]
    return {"rows": rows, "range": [min(sc), max(sc)], "range_shift_only": [min(shift_only), max(shift_only)],
            "note": "av shift: the music is moved against the motion by +-%g ms (the unpinned audio-video offset); seam variants: the first / second song's pulses through the DJ mix" % cfg.av_uncertainty_ms}


def best_level(ent_t: dict, role: str) -> str:
    lv = ent_t[role]["body"]["levels"]
    return max(lv, key=lambda k: (lv[k]["chosen_share"] or 0.0))


def claims(desc: dict, ent_t: dict, conf: dict, final: dict, inp: Inputs, cfg: Config) -> list:
    out = [f"{TITLE}. It measures how strongly the dancers' centre-of-mass sway and foot touchdowns are phase-locked to the song's beat grid; it does not judge technique, partnering, "
           "musicality as a whole or anything an official judge scores, and nothing validates a mapping from it to judges' scores."]
    for r in ROLES:
        bl = best_level(ent_t, r)
        b = ent_t[r]["body"]
        out.append(f"{r}: the body locks most often at the '{ent.LEVEL_LABELS[ent.LEVEL_NAMES.index(bl)]}' level (phase-locking {b['best_level_rho_mean']} on average against {b['baseline_mean']} for "
                   f"metronomes at the wrong tempo, {b['excess_sd']} baseline sd above).")
    b = desc.get("both")
    if b:
        g = b["binary_grid"]
        out.append(f"{b['n']} foot touchdowns: {g['within_60ms']*100:.0f}% within 60 ms of a beat or 'and' slot (random: {g['chance_60ms']*100:.0f}%), phase concentration R {g['R_eighth']:.2f} at the eighth "
                   f"(random {g['R_chance']:.2f}): zouk steps are not placed slot by slot, so this is a description and is not scored.")
    if conf["insufficient_evidence"]:
        out.append("The body shows no detectable phase-lock to this grid (insufficient beat lock evidence): the score is not meaningful for this take.")
    if not inp.meta.get("av_pinned"):
        out.append(f"The audio-video offset is unpinned (+-{cfg.av_uncertainty_ms:.0f} ms): a few degrees of phase at the half-bar level, so the locks are barely affected; absolute alignment claims would be.")
    if any(k == "seam" for _, _, k in inp.grid.no_beat):
        out.append("The DJ mix hands one song over to the other without a shared grid: the score is frozen there and the entrainment starts again with the next song.")
    out.append("A constant phase offset between the sway and the beat is not penalised (the convention for where in the beat a weight transfer lands is not established); a sway at the wrong tempo is.")
    if inp.music is not None and inp.music.source != "audio":
        out.append("No soundtrack was given: accents are the grid's downbeats only and the bar phase may be off, so the accent component is weak.")
    if inp.body is None:
        out.append("No body signals (physics / kinematics) were found: the score uses the foot touchdowns only.")
    out.append(f"Confidence {conf['label']} ({conf['value']:.2f}): grid confidence {conf['grid_confidence_mean']:.2f}, lock evidence {conf['lock_evidence_z']:.1f} baseline sd, A/V factor {conf['av_factor']:.2f}.")
    return out


def pack_live(live: dict, cfg: Config) -> dict:
    r = lambda a, p: [round(float(x), p) for x in a]                      # noqa: E731
    d = {"hz": cfg.live_hz, "t0": round(float(live["t"][0]), 4), "n": int(len(live["t"])), "score": r(live["score"], 2)}
    for k in COMPONENTS:
        d[k] = r(live[k], 3)
    d.update({"confidence": r(live["confidence"], 3), "evidence": r(live["evidence"], 3), "state": [int(x) for x in live["state"]],
              "level_lead": [int(x) for x in live["level_lead"]], "level_follow": [int(x) for x in live["level_follow"]],
              "levels": list(ent.LEVEL_LABELS), "state_codes": {"0": "beat", "1": "no song (frozen)", "2": "DJ seam (frozen)"}})
    return d


def analyse_one(inp: Inputs, cfg: Config, grid: Grid | None = None, av_shift_s: float = 0.0, full: bool = True) -> dict:
    A = run_series(inp, cfg, grid, av_shift_s)
    final = aggregate(A, cfg)
    res = {"A": A, "final": final}
    if full:
        res["live"] = live_timeline(inp, A, cfg, grid)
        annotate_steps(inp, A, A.events, cfg, grid)
    return res


def run_analysis(inp: Inputs, cfg: Config | None = None, with_sensitivity: bool = True) -> dict:
    """the full analysis -> the rhythm.json dict"""
    cfg = cfg or Config()
    t_start = time.time()
    res = analyse_one(inp, cfg)
    A, final, live = res["A"], res["final"], res["live"]
    ev = A.events
    desc = descriptive(ev, inp, cfg)
    ent_t = entrainment_table(A, cfg)
    desc["entrainment"] = ent_t
    conf = confidence_summary(A, inp.grid, inp, cfg, final)
    lo, hi, sd = noise_interval(A, cfg, final)
    sens = sensitivity(inp, cfg) if with_sensitivity else None
    rng = [x for x in ([lo, hi] + (sens["range"] if sens else [])) if x is not None and not math.isnan(x)]
    rng = [round(min(rng), 1), round(max(rng), 1)] if rng else None
    cols = ["t", "role", "foot", "valid", "measure", "slot", "kind", "offset_ms", "level", "level_offset_ms", "class", "lock", "accent", "accent_chance", "conf", "state", "sigma_ms"]
    t_end = float(inp.t1)
    doc = {
        "version": VERSION, "kind": "rhythm", "title": TITLE, "take": inp.meta.get("take"), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "time_base": "reference seconds (the capture's time_to_audio maps them to the audio clock)",
        "config": asdict(cfg), "inputs": {k: v for k, v in inp.meta.items()},
        "formula": {"score": "100 * sum_k weight_k * component_k", "weights": cfg.weights,
                    "components": "each 0..1, 0 = what a metronome at the wrong tempo gives, 1 = the reference level (docs/RHYTHM.md); body / feet / couple = estimated from everything since the song's start, "
                                  "recent = the last ~20 s; the live number starts at a neutral 50 and moves to the estimate as the evidence accumulates",
                    "evidence_time_s": cfg.ent.tau_long_s, "levels": list(ent.LEVEL_LABELS)},
        "final": {"score": _r(final["score"], 1), "components": {k: _r(final[k]) for k in COMPONENTS}, "weights": cfg.weights,
                  "n_samples": final["n_samples"], "levels": {r: best_level(ent_t, r) for r in ROLES},
                  "confidence": conf, "noise_floor_sd": _r(sd, 1), "score_interval_90": [_r(lo, 1), _r(hi, 1)], "range_under_uncertainty": rng,
                  "insufficient_evidence": conf["insufficient_evidence"]},
        "final_card": {"from_t": round(t_end - cfg.final_card_s, 3), "to_t": round(t_end, 3), "title": TITLE},
        "describe": desc,
        "steps": {"columns": cols, "rows": [[s.get(c) for c in cols] for s in ev.steps]},
        "measures": measure_table(inp, A, ev, cfg),
        "windows": window_table(inp, A, cfg),
        "live": pack_live(live, cfg),
        "no_beat": [[round(max(a, inp.t0), 3), round(min(b, inp.t1), 3), k] for a, b, k in inp.grid.no_beat if b > inp.t0 and a < inp.t1],
        "music": {"source": inp.music.source if inp.music else None, "accents": int(len(inp.music.acc_t)) if inp.music else 0,
                  "breaks": [[round(a, 2), round(b, 2)] for a, b in (inp.music.breaks if inp.music else [])]},
        "sensitivity": sens,
        "claims": [],
        "seconds": round(time.time() - t_start, 2),
    }
    doc["claims"] = claims(desc, ent_t, conf, final, inp, cfg)
    return doc


# --------------------------------------------------------------------------------------------------------------- the viewer's copy
CLASS_CODE = {"on": 0, "near": 1, "off": 2, "neutral": 3}


def for_unity(doc: dict) -> dict:
    """The viewer's copy of rhythm.json (the capture's "rhythm" file, read by Assets/Overlays/RhythmScore.cs): the 30 Hz live timeline, the valid steps with the class the HUD
    flashes, the final card and the texts. Times are reference seconds (the viewer adds the capture's time_to_audio)."""
    cols = doc["steps"]["columns"]
    ci = {c: i for i, c in enumerate(cols)}
    rows = []
    for r in doc["steps"]["rows"]:
        if not r[ci["valid"]] or r[ci["class"]] is None:
            continue
        rows.append([r[ci["t"]], 0 if r[ci["role"]] == "lead" else 1, CLASS_CODE[r[ci["class"]]], r[ci["level_offset_ms"]]])
    w = doc["formula"]["weights"]
    f = doc["final"]
    return {
        "version": VERSION, "kind": "rhythm_view", "title": doc["title"], "take": doc.get("take"),
        "components": [{"key": k, "label": COMPONENT_LABELS[k], "weight": round(100 * w[k] / sum(w.values()))} for k in COMPONENTS],
        "final": {"score": f["score"], "components": f["components"], "confidence": f["confidence"]["label"], "confidence_value": f["confidence"]["value"],
                  "range": f.get("range_under_uncertainty"), "insufficient_evidence": bool(f.get("insufficient_evidence")), "interval_90": f.get("score_interval_90"),
                  "lock_evidence_z": f["confidence"].get("lock_evidence_z")},
        "final_card": doc["final_card"],
        "live": doc["live"],
        "no_beat": doc["no_beat"],
        "steps": {"columns": ["t", "role", "class", "offset_ms"], "classes": ["on", "near", "off", "neutral"], "roles": ["lead", "follow"], "rows": rows},
        "notes": doc["claims"][1:4],
    }


# --------------------------------------------------------------------------------------------------------------- synthetic takes (tests, selftest)
def synthetic_inputs(kind: str = "locked", seed: int = 0, jitter_ms: float = 20.0, bpm: float = 84.0, measures: int = 40, rate: str = "pulse", offset_ms: float = 0.0,
                     body_tempo: float = 1.0, with_body: bool = True) -> Inputs:
    """A synthetic dance on a regular grid, with the dancers' body sway (a 2-pulse oscillation of the centre of mass speed / height) and steps.
    kind: locked (sway and steps on the grid) | random (steps and sway unrelated to the grid) | wrong_tempo (the same sway at `body_tempo` x the music's tempo)
          | offset (the sway a constant `offset_ms` late) | on_beat / off_beat / offset_consistent (steps only variants)"""
    rng = np.random.default_rng(seed)
    P = 60.0 / bpm
    p8 = P / 2
    n = measures * 8
    t = 10.0 + np.arange(n) * p8
    rows = [[t[i], (1, 3, 3, 2, 1, 3, 2, 3)[i % 8], i // 8, i % 8] for i in range(n)]
    grid = build_grid(rows, no_beat_none=[(0.0, 10.0 - p8 / 2)], conf=np.ones(n), default_conf=1.0)
    steps = {"lead": [], "follow": []}
    base = t[:-1][::2] if rate == "pulse" else t[:-1]
    kk = {"locked": "on_beat", "offset": "offset_consistent", "wrong_tempo": "on_beat"}.get(kind, kind)
    for r in ROLES:
        for i, b in enumerate(base):
            if kk == "on_beat":
                tm = b + rng.normal(0, jitter_ms / 1000.0)
            elif kk == "off_beat":
                tm = b + p8 / 2 + rng.normal(0, jitter_ms / 1000.0)
            elif kk == "offset_consistent":
                tm = b + offset_ms / 1000.0 + rng.normal(0, jitter_ms / 1000.0)
            else:
                tm = b + rng.uniform(-p8 / 2, p8 / 2)
            steps[r].append({"t": float(tm), "foot": "left" if i % 2 == 0 else "right", "sigma": 0.015})
    if kind == "random":
        steps = {r: [{"t": float(rng.uniform(t[0], t[-1])), "foot": "left", "sigma": 0.015} for _ in range(len(base))] for r in ROLES}
    elif kind == "wrong_tempo":
        steps = {r: [{"t": float(t[0] + (s["t"] - t[0]) / body_tempo), "foot": s["foot"], "sigma": s["sigma"]} for s in steps[r]] for r in ROLES}
    music = Music(t[::2], np.where(np.arange(len(t[::2])) % 2 == 0, 1.0, 0.6), [], np.array([]), np.array([]), "grid")
    t0, t1 = 8.0, float(t[-1]) + 2.0
    body = None
    if with_body:
        fs = 30.0
        tb = np.arange(t0, t1, 1.0 / fs)
        f2 = 1.0 / (2.0 * P)                                      # the half-bar sway: one cycle per two pulses
        sig = {}
        for r in ROLES:
            if kind == "random":
                ph = 2 * np.pi * 1.31 * f2 * tb + np.cumsum(rng.normal(0.0, 0.25, len(tb)))               # a sway at an unrelated rate with a wandering phase
            elif kind == "wrong_tempo":
                ph = 2 * np.pi * f2 * body_tempo * tb
            else:
                ph = 2 * np.pi * f2 * (tb - offset_ms / 1000.0 * (kind == "offset")) + rng.normal(0, 0.05)
            sig[r] = {"move": 0.35 + 0.2 * np.cos(ph) + 0.05 * rng.normal(size=len(tb)), "bounce": 0.95 + 0.02 * np.cos(ph + 1.0) + 0.004 * rng.normal(size=len(tb))}
        body = ent.Body(tb, sig, "synthetic")
    return Inputs(grid, steps, music, None, t0, t1, {"take": f"synthetic_{kind}", "av_pinned": True}, body=body)


def selftest() -> int:
    cfg = Config()
    bad = 0
    for kind, ok in (("locked", lambda s: s >= 70), ("random", lambda s: s <= 20), ("wrong_tempo", lambda s: s <= 20), ("offset", lambda s: s >= 70)):
        inp = synthetic_inputs(kind, body_tempo=1.12)
        doc = run_analysis(inp, cfg, with_sensitivity=False)
        sc = doc["final"]["score"]
        print(f"{kind:12s} final {sc:5.1f}  components {doc['final']['components']}")
        if not ok(sc):
            bad += 1
    print("selftest", "OK" if not bad else "FAILED")
    return 1 if bad else 0


# --------------------------------------------------------------------------------------------------------------- plots
def make_plots(doc: dict, inp: Inputs, out_dir: Path, A: Analysis | None = None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    live = doc["live"]
    t = live["t0"] + np.arange(live["n"]) / live["hz"]
    paths = []
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    if A is not None:
        K = A.K
        for r, c in (("lead", "#d9453d"), ("follow", "#3a6fd9")):
            ax[0, 0].plot(A.t, K.raw[("body", r)], color=c, lw=1.0, label=f"{r} body lock (cumulative, best level)")
            ax[0, 0].plot(A.t, K.baseline[("body", r)][0], color=c, lw=1.0, ls=":", label=f"{r} wrong-tempo baseline")
        ax[0, 0].set_ylim(0, 1)
        ax[0, 0].set_title("phase-locking of the body to the grid vs the wrong-tempo baseline")
        ax[0, 0].legend(fontsize=7)
        for r, c in (("lead", "#d9453d"), ("follow", "#3a6fd9")):
            ax[0, 1].plot(A.t, A.S.best_body[r] + (0.04 if r == "follow" else 0), color=c, lw=1.0, label=f"{r}: chosen level")
        ax[0, 1].set_yticks(range(len(ent.LEVEL_NAMES)), ent.LEVEL_NAMES)
        ax[0, 1].set_title("the metrical level the body locks to (causal choice)")
        ax[0, 1].legend(fontsize=7)
    ax[1, 0].plot(t, live["score"], "k", lw=1.6, label="live score")
    for k, c in zip(COMPONENTS, ("#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e")):
        ax[1, 0].plot(t, 100 * np.array(live[k]), lw=0.8, color=c, label=f"{k} x100", alpha=0.8)
    for a, b, k in doc["no_beat"]:
        ax[1, 0].axvspan(a, b, color="0.85")
    ax[1, 0].set_ylim(0, 100)
    ax[1, 0].set_title(f"live score and components (final {doc['final']['score']}, confidence {doc['final']['confidence']['label']})")
    ax[1, 0].legend(fontsize=7, ncol=3)
    ax[1, 1].plot(t, live["confidence"], color="#555", label="confidence")
    ax[1, 1].plot(t, live["evidence"], color="#a33", label="lock evidence")
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_title("confidence and lock evidence")
    fig.tight_layout()
    p = out_dir / "rhythm_review.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ms = doc["measures"]
    ax[0].bar([m["measure"] for m in ms], [m["score"] or 0 for m in ms], color="#6a8caf")
    ax[0].set_title("running score per measure (descriptive)")
    w = [x for x in doc["windows"] if x["score"] is not None]
    ax[1].plot([x["t"] for x in w], [x["score"] for x in w], color="#6a8caf")
    ax[1].set_title("running score, %g s windows" % doc["config"]["window_s"])
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
        ax.set_ylim(0, 100)
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
    tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8", newline="\n")
    tmp.replace(path)


def install_into(doc: dict, capture_dirs: list) -> list:
    """write the viewer's copy (for_unity) into capture folders and name it in their capture.json ("rhythm"); nothing else in the capture is touched.
    Returns the folders written."""
    done = []
    view = json.dumps(for_unity(doc), separators=(",", ":"), allow_nan=False)
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
        A = run_series(inp, cfg)
        doc["plots"] = make_plots(doc, inp, out_dir, A)
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
    sub.add_parser("controls", help="positive / negative / no-dance controls of the metric (see dancecap.rhythm_controls)")
    i = sub.add_parser("install", help="copy work/<take>/rhythm/rhythm.json (the viewer's subset) into the capture folder(s) and name it in capture.json")
    i.add_argument("take")
    i.add_argument("--capture", required=True, help="capture folder name in the viewer's StreamingAssets (and work/<take>/export/)")
    i.add_argument("--no-viewer", action="store_true", help="only work/<take>/export/<capture>")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "controls":
        from . import rhythm_controls
        return rhythm_controls.main([])
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
    print(f"[rhythm] 90% sampling interval {f['score_interval_90']}; range under the sampling, the +-{doc['config']['av_uncertainty_ms']:g} ms offset and the DJ seam variants: {f['range_under_uncertainty']}")
    for c in doc["claims"]:
        print("  -", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
