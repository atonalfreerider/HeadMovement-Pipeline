"""Beats: the take's music on the reference clock, the zouk beat grid and the true tempo.

Inputs: the take's source videos + the per-camera affine audio clock (takes/<take>.toml -> alignment.json,
reference_time = slope * source_time + offset, reference = cam06). Outputs (work/<take>/):

  audio/audio_overlap.wav       mono music from the cleanest phone, resampled onto the REFERENCE clock, covering
                                the full all-camera overlap; sample 0 = audio_overlap.json reference_start
  audio/audio_overlap.json      {source_cam, source_file, sample_rate, reference_start, reference_end, ...,
                                time_to_audio = -reference_start (capture.json convention), caveats}
  audio/quality.json            per-camera music quality (loudness, clipping, codec bandwidth, envelope SNR vs
                                the leave-one-out consensus of the other phones, beat_this confidence)
  audio/clicks_overlay.wav      audio_overlap.wav + clicks on the zouk grid (type 1 high, 2 mid, 3 soft)
  beats/beats.json              zouk grid [[t, type]] in reference seconds: 8 eighth slots per 4-beat measure,
                                type 1 at slots 0,4 / 2 at slots 3,6 / 3 otherwise (beat_this_analyzer
                                calculate_zouk_time); starts on a downbeat and holds whole measures only
  beats/beats_raw.json          beat_this beats/downbeats of the source camera alone (source + reference
                                seconds, frame-quantised and sub-frame refined), the all-phone vote that the grid
                                uses, the per-camera shifts, every other camera's beats, the grid with
                                [t, type, measure, slot]
  beats/tempo.json              true tempo (quarter-note BPM), measure length, stability, the level decision,
                                the bar-phase decision + kick/backbeat check, what split_measures would do
  review/beats_waveform.png     waveform + grid + beats/downbeats + local BPM + phase check + phone quality
  beats.json                    copy of beats/beats.json at the take root, where timing/physics/export look

How the grid is built (and the analyzer bug it avoids): beat_this runs on every phone's whole recording; the
phones' beat/downbeat probabilities are mapped to the reference clock, shifted onto the source phone's timing
(median beat offset) and averaged, then peak-picked (a single phone's tracker can slip by an eighth for a few
bars; on larissa_kadu cam02 alone was 150-180 ms off at 30.8-33.7 s, the vote fixes it). The analyzer's
split_measures(min_bpm=120) halves every measure whose 240/duration < 120, i.e. every measure of a 77 BPM track,
which doubles the grid (16 eighths per real measure). Here the beat level is decided explicitly: quarter-note
tempo is brought into [60, 120) BPM by halving/doubling the tracked beat rate (zouk is played 70-100 BPM). The
bar phase comes from beat_this's downbeats (majority vote of beat index mod 4); it is checked against the music
itself (onsets folded on the grid: kick/bass on beat 1 rather than 3, snare backbeat on 2 and 4, best of all 8
eighth shifts) and replaced by that evidence only when beat_this's downbeats are ambiguous. Eighths are the
midpoints of consecutive quarter beats, so the grid follows the music's micro-timing.

Caveat: phone A/V offsets and sound travel (~3 ms/m) are unknown, so absolute 'on the beat' claims need an
audio-visual event (clap, accented stomp); relative timing (follow vs lead, measure vs measure) is robust.

    python -m dancecap beats larissa_kadu                 # choose the cleanest phone automatically
    python -m dancecap beats larissa_kadu --source 06     # force the source camera
"""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, Take

ZOUK_TYPES = (1, 3, 3, 2, 1, 3, 2, 3)  # eighth slots 0..7 (beat_this_analyzer.calculate_zouk_time)
TRESILLO = (0, 3, 6)
BEAT_THIS_FPS = 50
ANALYSIS_SR = 22050
CAVEAT = ("Phone audio/video offsets and sound travel (~2.9 ms per metre speaker->phone) are unknown: beat times are "
          "exact on the reference AUDIO clock, but absolute 'on the beat' claims about the video need an audio-visual "
          "sync event (clap, accented stomp). Relative timing (follow vs lead, measure vs measure) is robust.")


@dataclass
class BeatsConfig:
    source: str | None = None  # force the source camera, else the best quality score
    checkpoint: str = "final0"
    device: str | None = None  # None: cuda when the GPU lock is free within gpu_wait_s, else cpu
    gpu_wait_s: float = 30.0
    min_bpm: float = 60.0  # quarter-note tempo is folded into [min_bpm, 2 * min_bpm)
    out_sr: int | None = None  # output wav rate (None = the source's native rate)
    margin_s: float = 0.0  # shrink the overlap by this much at both ends
    ensemble: bool = True  # average all phones' beat_this probabilities (False: source camera only)
    rephase_margin: float = 1.0  # kick + backbeat score margin needed to override AMBIGUOUS beat_this downbeats


# ------------------------------------------------------------------------------------------------ audio io
def decode_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Mono float32 samples of a media file via ffmpeg (container time 0 = sample 0, like the sync fit)."""
    if sr is None:
        sr = probe_sample_rate(path)
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                                   "-f", "f32le", "-"])
    return np.frombuffer(raw, np.float32).copy(), sr


def probe_sample_rate(path: Path) -> int:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                   "stream=sample_rate", "-of", "csv=p=0", str(path)], text=True)
    return int(out.strip().splitlines()[0])


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    """16-bit PCM wav (mono or (n, ch)); peak-safe (clips at +-1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, np.float64)
    ch = 1 if x.ndim == 1 else x.shape[1]
    pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), "<i2").astype(np.float32) / 32767
    return (x if ch == 1 else x.reshape(-1, ch)), sr


def to_reference(x: np.ndarray, sr: float, slope: float, offset: float, t0: float, t1: float,
                 sr_out: int) -> tuple[np.ndarray, float]:
    """Resample a source-clock signal onto the reference clock over [t0, t1).

    Source sample k sits at reference time offset + slope * k / sr, i.e. the source runs at sr / slope samples per
    reference second; soxr converts that rate to sr_out (delay-compensated, so output sample m sits at
    offset + m / sr_out). Returns (y, reference time of y[0]); the start is snapped to the nearest output sample
    (< 0.5 / sr_out error) and reported exactly. Samples outside the recording are zero.
    """
    import soxr
    rate = sr / slope
    y = np.asarray(x, np.float32) if abs(rate - sr_out) < 1e-9 * sr_out else \
        soxr.resample(np.asarray(x, np.float32), rate, sr_out, quality="VHQ")
    m0 = int(round((t0 - offset) * sr_out))
    n = int(round((t1 - t0) * sr_out))
    out = np.zeros(n, np.float32)
    lo, hi = max(m0, 0), min(m0 + n, len(y))
    if hi > lo:
        out[lo - m0:hi - m0] = y[lo:hi]
    return out, offset + m0 / sr_out


# ------------------------------------------------------------------------------------------------ quality
def logmel_envelope(x: np.ndarray, sr: int, hop_s: float = 0.02, n_mels: int = 40) -> np.ndarray:
    """(n_mels, T) log-mel energies, 20 ms hop, 60 Hz - 7 kHz (the band every phone codec keeps)."""
    import librosa
    hop = int(round(hop_s * sr))
    m = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=sr, n_fft=2048 if sr > 24000 else 1024,
                                       hop_length=hop, n_mels=n_mels, fmin=60, fmax=min(7000, sr / 2 - 100))
    return np.log10(np.maximum(m, 1e-6 * m.max() + 1e-20))  # 60 dB floor: digital silence must not dominate


def _zscore(e: np.ndarray) -> np.ndarray:
    return (e - e.mean(1, keepdims=True)) / (e.std(1, keepdims=True) + 1e-9)


def envelope_snr(envs: dict[str, np.ndarray], max_lag: int = 5) -> dict[str, dict]:
    """Music SNR proxy per camera: correlation r of its z-scored log-mel envelope with the leave-one-out median of
    the other cameras (lag-searched +-max_lag frames for residual sync / sound travel); for x = s + n with
    corr(x, s) = r, SNR = r^2 / (1 - r^2). Speech, handling noise or a weak phone lower r."""
    z = {c: _zscore(e) for c, e in envs.items()}
    T = min(v.shape[1] for v in z.values())
    z = {c: v[:, :T] for c, v in z.items()}
    out = {}
    for c in z:
        others = [z[o] for o in z if o != c]
        if not others:
            # a single-camera take has nothing to compare with: SNR is undefined, scored 0 (an infinite score broke the
            # review plot and the quality json); the lone phone is the source
            out[c] = {"r": 1.0, "snr_db": 0.0, "lag_frames": 0, "single_camera": True}
            continue
        cons = np.median(np.stack(others), 0)
        best = (-2.0, 0)
        for L in range(-max_lag, max_lag + 1):
            a = z[c][:, max(0, L):T + min(0, L)]
            b = cons[:, max(0, -L):T - max(0, L)]
            a = _zscore(a)
            b = _zscore(b)
            r = float(np.mean(np.mean(a * b, 1)))
            if r > best[0]:
                best = (r, L)
        r = min(max(best[0], 1e-6), 1 - 1e-6)
        out[c] = {"r": round(r, 4), "snr_db": round(10 * np.log10(r * r / (1 - r * r)), 2), "lag_frames": best[1]}
    return out


def level_stats(x: np.ndarray, sr: int) -> dict:
    """Loudness, clipping and codec bandwidth (highest frequency within 60 dB of the spectral peak)."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
    peak = float(np.max(np.abs(x)) + 1e-12)
    n = 8192
    segs = len(x) // n
    if segs:
        spec = np.mean(np.abs(np.fft.rfft(x[:segs * n].reshape(segs, n) * np.hanning(n), axis=1)) ** 2, 0)
        db = 10 * np.log10(spec + 1e-20)
        db = np.convolve(db, np.ones(9) / 9, "same")
        f = np.fft.rfftfreq(n, 1 / sr)
        above = np.nonzero(db > db[f > 50].max() - 60)[0]
        bw = float(f[above.max()]) if len(above) else 0.0
    else:
        bw = 0.0
    return {"rms_dbfs": round(20 * np.log10(rms), 2), "peak_dbfs": round(20 * np.log10(peak), 2),
            "clip_frac": float(np.mean(np.abs(x) >= 0.985)), "bandwidth_hz": round(bw)}


def quality_score(q: dict) -> float:
    """Higher is better: envelope SNR, minus clipping and narrow-codec penalties, plus a small beat-confidence term."""
    s = q["snr"]["snr_db"]
    s -= 3.0 * min(1.0, q["level"]["clip_frac"] / 1e-3)
    s -= 2.0 * max(0.0, (11000 - q["level"]["bandwidth_hz"]) / 5000)
    s += 2.0 * q.get("beat_confidence", 0.0)
    return float(s)


# ------------------------------------------------------------------------------------------------ beat_this
def pick_device(cfg: BeatsConfig):
    """('cuda', lock context) when CUDA is available and the GPU lock frees up within cfg.gpu_wait_s, else cpu."""
    import contextlib

    import torch

    from . import gpu
    if cfg.device:
        return cfg.device, (gpu.exclusive("beats") if cfg.device.startswith("cuda") else contextlib.nullcontext())
    if not torch.cuda.is_available():
        return "cpu", contextlib.nullcontext()
    try:
        ctx = gpu.exclusive("beats", timeout_s=cfg.gpu_wait_s)
        ctx.__enter__()
    except TimeoutError:
        print(f"[beats] GPU busy ({gpu.LOCK.read_text() if gpu.LOCK.exists() else '?'}); beat_this on CPU")
        return "cpu", contextlib.nullcontext()
    return "cuda", _Entered(ctx)


class _Entered:
    """Wrap an already-entered context manager so `with` only exits it."""

    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self.ctx.__exit__(*exc)


class BeatTracker:
    """beat_this (CPJKU, zouk branch checkout at HeadMovement/beat_this) framewise logits + minimal peak picking,
    plus sub-frame refinement: a parabola through the beat logits at the picked frame and its neighbours
    (the minimal postprocessor quantises to 20 ms frames)."""

    def __init__(self, checkpoint: str = "final0", device: str = "cpu"):
        from beat_this.inference import Audio2Frames
        from beat_this.model.postprocessor import Postprocessor
        self.a2f = Audio2Frames(checkpoint, device=device, float16=False)
        self.pp = Postprocessor(type="minimal", fps=BEAT_THIS_FPS)
        self.checkpoint, self.device = checkpoint, device

    def __call__(self, x: np.ndarray, sr: int) -> dict:
        bl, dl = self.a2f(np.asarray(x, np.float64), sr)
        beats, downs = self.pp(bl, dl)
        bl = bl.detach().cpu().numpy().astype(np.float64)
        dl = dl.detach().cpu().numpy().astype(np.float64)
        beats, downs = np.asarray(beats, float), np.asarray(downs, float)
        rb = refine_peaks(beats, bl, BEAT_THIS_FPS)
        # downbeats are snapped to beats by beat_this; carry the same refinement
        idx = np.abs(beats[None, :] - downs[:, None]).argmin(1) if len(beats) and len(downs) else np.zeros(0, int)
        rd = rb[idx] if len(idx) else downs
        prob = 1 / (1 + np.exp(-bl))
        frames = np.clip(np.round(beats * BEAT_THIS_FPS).astype(int), 0, len(bl) - 1)
        return {"beats": beats, "downbeats": downs, "beats_refined": rb, "downbeats_refined": rd,
                "beat_prob": prob, "downbeat_prob": 1 / (1 + np.exp(-dl)),
                "beat_peak_prob": prob[frames] if len(frames) else np.zeros(0)}


def refine_peaks(times: np.ndarray, logits: np.ndarray, fps: float) -> np.ndarray:
    """Parabolic sub-frame interpolation of each picked peak (shift limited to +-0.5 frame)."""
    out = np.asarray(times, float).copy()
    for k, t in enumerate(out):
        f = int(round(t * fps))
        if 0 < f < len(logits) - 1:
            a, b, c = logits[f - 1], logits[f], logits[f + 1]
            den = a - 2 * b + c
            if den < 0:
                out[k] = (f + float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))) / fps
    return out


# ------------------------------------------------------------------------------------------------ grid
def fix_beat_level(beats: np.ndarray, downbeats: np.ndarray, min_bpm: float = 60.0) -> tuple[np.ndarray, dict]:
    """Bring the tracked beat rate to the quarter-note level in [min_bpm, 2 * min_bpm).

    Tracked eighths (> 2 * min_bpm): keep every other beat, choosing the parity that holds most downbeats.
    Tracked half notes (< min_bpm): insert midpoints. Returns (quarter beats, info)."""
    beats = np.asarray(beats, float)
    info = {"tracked_bpm": None, "factor": 1.0, "parity": None}
    if len(beats) < 3:
        return beats, info
    bpm = 60.0 / float(np.median(np.diff(beats)))
    info["tracked_bpm"] = round(bpm, 3)
    while bpm >= 2 * min_bpm:
        near = np.abs(beats[:, None] - np.asarray(downbeats, float)[None, :]).min(1) < 0.035 if len(downbeats) \
            else np.zeros(len(beats), bool)
        votes = [int(near[p::2].sum()) for p in (0, 1)]
        p = int(np.argmax(votes))
        beats = beats[p::2]
        info["factor"] /= 2
        info["parity"] = p
        info["parity_votes"] = votes
        bpm /= 2
    while bpm < min_bpm:
        beats = np.sort(np.concatenate([beats, 0.5 * (beats[1:] + beats[:-1])]))
        info["factor"] *= 2
        bpm *= 2
    return beats, info


def repair_gaps(beats: np.ndarray, win: int = 8, tol: float = 0.3) -> tuple[np.ndarray, dict]:
    """Insert beats into ~2x / ~3x gaps and drop extra beats (< 0.5x) against the local median interval."""
    b = list(np.asarray(beats, float))
    ins = drop = 0
    i = 1
    while i < len(b):
        lo = max(1, i - win)
        ibis = np.diff(b[lo - 1:min(len(b), i + win)])
        med = float(np.median(ibis)) if len(ibis) else b[i] - b[i - 1]
        d = b[i] - b[i - 1]
        k = int(round(d / med))
        if k >= 2 and abs(d / k - med) < tol * med:
            for j in range(1, k):
                b.insert(i - 1 + j, b[i - 1] + d * j / k)
            ins += k - 1
            i += k
            continue
        if d < (1 - tol) * med * 0.93 and i + 1 < len(b):
            # an extra beat between two real ones: drop whichever leaves the more regular interval
            keep_next = abs((b[i + 1] - b[i - 1]) - med) < abs((b[i + 1] - b[i]) - med)
            del b[i if keep_next else i - 1]
            drop += 1
            continue
        i += 1
    return np.asarray(b), {"inserted": ins, "dropped": drop}


def bar_phase(beats: np.ndarray, downbeats: np.ndarray, beats_per_bar: int = 4, tol_s: float = 0.07) -> dict:
    """Phase p (beat index mod beats_per_bar of the downbeats) by majority vote; agreement = fraction of matched
    downbeats with that phase."""
    if not len(downbeats) or not len(beats):
        return {"phase": 0, "agreement": 0.0, "votes": [0] * beats_per_bar, "matched": 0}
    d = np.abs(beats[:, None] - downbeats[None, :])
    j = d.argmin(0)
    ok = d[j, np.arange(len(downbeats))] < tol_s
    votes = np.bincount(j[ok] % beats_per_bar, minlength=beats_per_bar)
    p = int(votes.argmax())
    return {"phase": p, "agreement": round(float(votes[p] / max(1, votes.sum())), 4),
            "votes": votes.tolist(), "matched": int(ok.sum())}


def zouk_grid(quarters: np.ndarray, phase: int) -> np.ndarray:
    """(N, 4) [t, type, measure, slot] for every WHOLE measure: slots 2j = quarter beat j, slots 2j+1 = midpoint of
    quarter beats j and j+1 (needs the next measure's downbeat to close the last eighth)."""
    q = np.asarray(quarters, float)
    rows = []
    m = 0
    for s in range(phase, len(q) - 4, 4):
        b = q[s:s + 5]
        for j in range(8):
            t = b[j // 2] if j % 2 == 0 else 0.5 * (b[j // 2] + b[j // 2 + 1])
            rows.append((t, ZOUK_TYPES[j], m, j))
        m += 1
    return np.asarray(rows, float).reshape(-1, 4)


def analyzer_split_check(downbeats: np.ndarray, min_bpm: float = 120.0, max_bpm: float = 190.0) -> dict:
    """What beat_this_analyzer.split_measures(min_bpm=120) would do with these downbeats: a measure whose
    240 / duration < min_bpm is halved when the half still is <= max_bpm."""
    d = np.diff(np.asarray(downbeats, float))
    if not len(d):
        return {"measures": 0, "split": 0}
    mbpm = 240.0 / d
    split = (mbpm < min_bpm) & (2 * mbpm <= max_bpm)
    return {"measures": int(len(d)), "split": int(split.sum()), "split_fraction": round(float(split.mean()), 4),
            "measure_bpm_median": round(float(np.median(mbpm)), 3),
            "resulting_grid_bpm_median": round(float(np.median(np.where(split, 2 * mbpm, mbpm))), 3),
            "note": "split_measures halves every measure below 120 'measure BPM'; for a < 120 BPM track this yields "
                    "16 eighth entries per real measure (a dense grid at twice the true tempo). Not used here."}


def onset_strength(x: np.ndarray, sr: int, hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    o = librosa.onset.onset_strength(y=x.astype(np.float32), sr=sr, hop_length=hop)
    return o, librosa.frames_to_time(np.arange(len(o)), sr=sr, hop_length=hop)


def folded_profile(grid: np.ndarray, ot: np.ndarray, o: np.ndarray, bins: int = 16) -> np.ndarray:
    """Mean onset strength per 1/bins of a measure (measure-relative phase from the grid's downbeats); bin i is
    CENTRED on phase i / bins, so with 16 bins the even bins are the 8 eighth slots."""
    down = grid[grid[:, 3] == 0, 0]
    if len(down) < 2:
        return np.zeros(bins)
    k = np.searchsorted(down, ot, side="right") - 1
    ok = (k >= 0) & (k < len(down) - 1)
    ph = (ot[ok] - down[k[ok]]) / (down[k[ok] + 1] - down[k[ok]])
    b = np.floor(ph * bins + 0.5).astype(int) % bins
    s = np.bincount(b, weights=o[ok], minlength=bins)
    n = np.bincount(b, minlength=bins)
    return s / np.maximum(n, 1)


def band_onset(x: np.ndarray, sr: int, fmin: float = 20.0, fmax: float = 150.0,
               hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Half-wave rectified log-energy flux of one band (default kick/bass < 150 Hz); (onset, frame times)."""
    import librosa
    S = np.abs(librosa.stft(x.astype(np.float32), n_fft=1024 if sr <= 24000 else 2048, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=1024 if sr <= 24000 else 2048)
    e = np.log1p(1e3 * S[(f >= fmin) & (f < fmax)].sum(0))
    return np.maximum(0.0, np.diff(e, prepend=e[0])), librosa.frames_to_time(np.arange(len(e)), sr=sr, hop_length=hop)


def _eighth_z(profile: np.ndarray) -> np.ndarray:
    prof = np.asarray(profile, float)
    z = prof[::max(1, len(prof) // 8)][:8]
    return (z - z.mean()) / (z.std() + 1e-9)


def rotation_scores(kick_profile: np.ndarray, snare_profile: np.ndarray) -> dict:
    """Bar-phase evidence for every shift of the bar start by r eighths (0..7), from onset profiles folded on the
    grid (16 bins, even bins = eighth slots): S(r) = [kick on the would-be beat 1 minus kick on beat 3]
    + [snare backbeat on the would-be beats 2 and 4]. The backbeat fixes which eighths are quarter beats, the kick
    on 1 vs 3 fixes the half bar. Odd r = the quarter beats are really the off-eighths. Also reports the tresillo
    (kick on eighths 0, 3, 6) score per rotation as information."""
    zk, zs = _eighth_z(kick_profile), _eighth_z(snare_profile)
    sc = [float(zk[r] - zk[(r + 4) % 8] + zs[(r + 2) % 8] + zs[(r + 6) % 8]) for r in range(8)]
    best = int(np.argmax(sc))
    return {"kick_z": [round(float(v), 3) for v in zk], "snare_z": [round(float(v), 3) for v in zs],
            "scores": [round(v, 3) for v in sc], "best": best, "margin": round(sc[best] - sc[0], 3),
            "tresillo_scores": [round(float(np.mean([zk[(r + e) % 8] for e in TRESILLO])), 3) for r in range(8)]}


def phase_check(kick_profile: np.ndarray, snare_profile: np.ndarray) -> dict:
    """Is the grid's bar phase the best of the 8 eighth rotations (kick on 1 + snare on 2 and 4)?"""
    r = rotation_scores(kick_profile, snare_profile)
    others = [v for k, v in enumerate(r["scores"]) if k != 0]
    return {"bands": "kick/bass < 150 Hz and snare 1-4 kHz onset flux", **r, "consistent": r["best"] == 0,
            "margin": round(r["scores"][0] - max(others), 3)}


def rephase(quarters: np.ndarray, phase: int, shift_eighths: int) -> tuple[np.ndarray, int]:
    """Move the bar start by shift_eighths eighths: odd shifts swap quarter beats and off-eighths (midpoints).
    Returns (new quarter beats, new bar phase as an index into them)."""
    q = np.asarray(quarters, float)
    e = np.empty(2 * len(q) - 1)
    e[0::2] = q
    e[1::2] = 0.5 * (q[1:] + q[:-1])
    start = 2 * phase + shift_eighths
    par = start % 2
    return e[par::2], ((start - par) // 2) % 4


def local_bpm(quarters: np.ndarray, win: int = 8) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(quarters, float)
    if len(q) < win + 1:
        return np.zeros(0), np.zeros(0)
    t = 0.5 * (q[win:] + q[:-win])
    return t, 60.0 * win / (q[win:] - q[:-win])


# ------------------------------------------------------------------------------------------------ cross-camera
def cross_camera_offsets(ref_beats: np.ndarray, cam_beats: dict[str, np.ndarray], t0: float, t1: float,
                         tol_s: float = 0.1) -> dict:
    """Median offset (ms) of each camera's beats vs the source camera's, inside [t0, t1] on the reference clock.
    Mixes residual audio-clock error and acoustic path differences; spread = 1.4826 * MAD."""
    r = ref_beats[(ref_beats >= t0) & (ref_beats <= t1)]
    out = {}
    for c, b in cam_beats.items():
        if not len(b) or not len(r):
            continue
        j = np.abs(r[:, None] - b[None, :]).argmin(1)
        d = b[j] - r
        d = d[np.abs(d) < tol_s]
        if len(d):
            med = float(np.median(d))
            out[c] = {"median_ms": round(1000 * med, 2), "mad_ms": round(1482.6 * float(np.median(np.abs(d - med))), 2),
                      "matched": int(len(d)), "of": int(len(r))}
    return out


def ensemble_activation(tracks: dict, take: Take, shifts: dict[str, float], t0: float, t1: float,
                        fps: float = 100.0, edge_s: float = 0.5) -> dict:
    """Mean beat / downbeat probability of all phones on the reference clock (every phone hears the same music).

    Each camera's 50 fps beat_this probabilities are mapped through its affine clock, shifted by -shifts[cam]
    (its median beat offset vs the source camera, so the result stays on the SOURCE audio's timing), linearly
    interpolated onto a common `fps` grid and averaged where the camera has audio (minus edge_s at its ends)."""
    t = np.arange(t0, t1, 1.0 / fps)
    sb, sd, n = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    for c, tr in tracks.items():
        s = take.source(c)
        ft = s.to_reference(np.arange(len(tr["beat_prob"])) / BEAT_THIS_FPS) - shifts.get(c, 0.0)
        m = (t >= ft[0] + edge_s) & (t <= ft[-1] - edge_s)
        sb[m] += np.interp(t[m], ft, tr["beat_prob"])
        sd[m] += np.interp(t[m], ft, tr["downbeat_prob"])
        n[m] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"t": t, "beat": np.where(n > 0, sb / np.maximum(n, 1), 0.0),
                "downbeat": np.where(n > 0, sd / np.maximum(n, 1), 0.0), "n_cams": n.astype(int), "fps": fps}


def pick_peaks(t: np.ndarray, p: np.ndarray, thr: float = 0.4, half_window_s: float = 0.07) -> np.ndarray:
    """Local maxima within +-half_window_s above thr (beat_this 'minimal' rule), parabolic sub-sample times."""
    from scipy.ndimage import maximum_filter1d
    dt = float(t[1] - t[0])
    w = 2 * int(round(half_window_s / dt)) + 1
    mx = maximum_filter1d(p, w, mode="nearest")
    idx = np.nonzero((p == mx) & (p > thr))[0]
    keep, last = [], -10**9
    for i in idx:  # flat tops: keep the first sample of each plateau
        if i - last > w // 2:
            keep.append(i)
            last = i
    out = []
    for i in keep:
        if 0 < i < len(p) - 1:
            a, b, c = p[i - 1], p[i], p[i + 1]
            den = a - 2 * b + c
            out.append(t[i] + (float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) * dt if den < 0 else 0.0))
        else:
            out.append(t[i])
    return np.asarray(out)


def snap(times: np.ndarray, to: np.ndarray, tol_s: float = 0.07) -> np.ndarray:
    """Each time replaced by the nearest of `to` (dropped when farther than tol_s); unique, sorted."""
    if not len(times) or not len(to):
        return np.zeros(0)
    j = np.abs(np.asarray(times)[:, None] - np.asarray(to)[None, :]).argmin(1)
    ok = np.abs(to[j] - times) < tol_s
    return np.unique(to[j[ok]])


def disagreement(a: np.ndarray, b: np.ndarray, tol_s: float = 0.04) -> np.ndarray:
    """Elements of a with no element of b within tol_s."""
    if not len(a):
        return np.zeros(0)
    if not len(b):
        return np.asarray(a)
    d = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :]).min(1)
    return np.asarray(a)[d > tol_s]


# ------------------------------------------------------------------------------------------------ click track
def click_track(n: int, sr: int, grid: np.ndarray, t_audio0: float) -> np.ndarray:
    """Clicks at grid times (reference t -> audio sample (t - t_audio0) * sr): type 1 1.6 kHz loud, type 2 1.1 kHz,
    type 3 750 Hz soft; 30 ms decaying sine bursts."""
    out = np.zeros(n, np.float32)
    L = int(0.03 * sr)
    env = np.exp(-np.arange(L) / (0.006 * sr)).astype(np.float32)
    spec = {1: (1600.0, 0.55), 2: (1100.0, 0.38), 3: (750.0, 0.18)}
    for t, k in grid[:, :2]:
        f, a = spec[int(k)]
        s = int(round((t - t_audio0) * sr))
        if s < 0 or s >= n:
            continue
        e = min(n, s + L)
        out[s:e] += (a * env[:e - s] * np.sin(2 * np.pi * f * np.arange(e - s) / sr)).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------------ run
def overlap_interval(take: Take, durations: dict[str, float] | None = None) -> tuple[float, float]:
    """Reference interval every source covers: alignment.json maximum_overlap, else from the clocks + durations."""
    al = json.loads((PACKAGE_ROOT / "takes" / take.cfg["alignment"]).resolve().read_text())
    ov = al.get("maximum_overlap") or []
    if ov:
        return float(ov[0]["start"]), float(ov[0]["end"])
    dur = durations or {r["id"]: float(r["duration"]) for r in al["sources"]}
    return max(s.to_reference(0.0) for s in take.sources), min(s.to_reference(dur[s.id]) for s in take.sources)


def run(take: Take | str, cfg: BeatsConfig | None = None, plot: bool = True) -> dict:
    cfg = cfg or BeatsConfig()
    take = Take.load(take) if isinstance(take, str) else take
    work = take.work
    (work / "audio").mkdir(parents=True, exist_ok=True)
    (work / "beats").mkdir(parents=True, exist_ok=True)
    ov0, ov1 = overlap_interval(take)
    ov0, ov1 = ov0 + cfg.margin_s, ov1 - cfg.margin_s
    print(f"[beats] all-camera overlap {ov0:.3f}..{ov1:.3f} s (reference = cam06 audio clock)")

    # ---- decode every camera once at the analysis rate; quality on the overlap
    sigs, durations = {}, {}
    for s in take.sources:
        x, _ = decode_audio(s.path, ANALYSIS_SR)
        sigs[s.id] = x
        durations[s.id] = len(x) / ANALYSIS_SR
    envs, levels = {}, {}
    for s in take.sources:
        y, _ = to_reference(sigs[s.id], ANALYSIS_SR, s.slope, s.offset, ov0, ov1, ANALYSIS_SR)
        levels[s.id] = level_stats(y, ANALYSIS_SR)
        envs[s.id] = logmel_envelope(y, ANALYSIS_SR)
    snr = envelope_snr(envs)
    # native bandwidth needs the native rate (the analysis rate caps it at 11 kHz)
    for s in take.sources:
        xn, srn = decode_audio(s.path)
        a = int(max(0, s.to_source(ov0)) * srn)
        b = int(min(len(xn) / srn, s.to_source(ov1)) * srn)
        levels[s.id]["bandwidth_hz"] = level_stats(xn[a:b], srn)["bandwidth_hz"]
        levels[s.id]["native_sr"] = srn

    # ---- beat_this on every camera (whole recording, source clock), mapped to the reference clock
    device, lock = pick_device(cfg)
    tracks = {}
    with lock:
        tracker = BeatTracker(cfg.checkpoint, device)
        for s in take.sources:
            tr = tracker(sigs[s.id], ANALYSIS_SR)
            tr["ref_beats"] = s.to_reference(tr["beats_refined"])
            tr["ref_downbeats"] = s.to_reference(tr["downbeats_refined"])
            tracks[s.id] = tr
        del tracker
    print(f"[beats] beat_this {cfg.checkpoint} on {device}: " +
          ", ".join(f"{c}:{len(t['beats'])}b/{len(t['downbeats'])}d" for c, t in tracks.items()))

    quality = {}
    for c in tracks:
        tr = tracks[c]
        inside = (tr["ref_beats"] >= ov0) & (tr["ref_beats"] <= ov1)
        conf = float(np.mean(tr["beat_peak_prob"][inside])) if inside.any() else 0.0
        quality[c] = {"level": levels[c], "snr": snr[c], "beat_confidence": round(conf, 4),
                      "duration_s": round(durations[c], 3)}
        quality[c]["score"] = round(quality_score(quality[c]), 3)
    ranking = sorted(quality, key=lambda c: -quality[c]["score"])
    src_id = cfg.source or ranking[0]
    if src_id not in tracks:
        raise SystemExit(f"beats: unknown source camera {src_id}; take has {sorted(tracks)}")
    src = take.source(src_id)
    print("[beats] music quality (score = envelope SNR - clipping/bandwidth penalties + beat confidence): " +
          ", ".join(f"{c} {quality[c]['score']:.1f}" for c in ranking) + f" -> source cam{src_id}")

    # ---- beats: the source's whole recording (most context), with every phone voting where they overlap
    tr = tracks[src_id]
    offsets = cross_camera_offsets(tr["ref_beats"], {c: t["ref_beats"] for c, t in tracks.items() if c != src_id},
                                   ov0, ov1)
    shifts = {c: v["median_ms"] / 1000 for c, v in offsets.items()}
    span = (src.to_reference(0.0), src.to_reference(durations[src_id]))
    ens = ensemble_activation(tracks, take, shifts, *span) if cfg.ensemble else None
    if ens is not None:
        b_ref = pick_peaks(ens["t"], ens["beat"])
        d_ref = snap(pick_peaks(ens["t"], ens["downbeat"]), b_ref)
    else:
        b_ref, d_ref = tr["ref_beats"], tr["ref_downbeats"]
    q_ref, level = fix_beat_level(b_ref, d_ref, cfg.min_bpm)
    q_ref, gaps = repair_gaps(q_ref)
    ph = bar_phase(q_ref, d_ref)
    grid = zouk_grid(q_ref, ph["phase"])
    lo_o, lo_t = band_onset(sigs[src_id], ANALYSIS_SR)
    lo_t = src.to_reference(lo_t)
    sn_o, sn_t = band_onset(sigs[src_id], ANALYSIS_SR, 1000.0, 4000.0)
    sn_t = src.to_reference(sn_t)
    pv = level.get("parity_votes")
    ambiguous = ph["agreement"] < 0.8 or (pv is not None and min(pv) > 0.5 * max(pv))
    rot = rotation_scores(folded_profile(grid, lo_t, lo_o), folded_profile(grid, sn_t, sn_o))
    decision = {"by": "beat_this downbeats", "beat_this_ambiguous": bool(ambiguous), "eighth_rotation": rot}
    if rot["best"] != 0 and ambiguous and rot["margin"] > cfg.rephase_margin:
        q_ref, phase = rephase(q_ref, ph["phase"], rot["best"])
        grid = zouk_grid(q_ref, phase)
        decision.update(by="kick on 1 + snare backbeat (beat_this downbeats ambiguous)", shifted_eighths=rot["best"])
    ibi = np.diff(q_ref)
    bpm = 60.0 / float(np.median(ibi))
    in_ov = (q_ref >= ov0) & (q_ref <= ov1)

    def linfit(q):
        i = np.arange(len(q), dtype=float)
        A = np.stack([np.ones_like(i), i], 1)
        coef = np.linalg.lstsq(A, q, rcond=None)[0]
        return coef, q - A @ coef

    coef, resid = linfit(q_ref)
    coef_ov, resid_ov = linfit(q_ref[in_ov]) if in_ov.sum() > 3 else (coef, resid)
    o, ot = onset_strength(sigs[src_id], ANALYSIS_SR)
    ot = src.to_reference(ot)
    prof_low = folded_profile(grid, lo_t, lo_o)
    prof_sn = folded_profile(grid, sn_t, sn_o)
    pcheck = phase_check(prof_low, prof_sn)
    split = analyzer_split_check(d_ref)
    lt, lb = local_bpm(q_ref)
    src_only_bad = disagreement(tr["ref_beats"], b_ref)
    tempo = {
        "bpm": round(bpm, 3), "bpm_definition": "quarter notes (4 per measure); median inter-beat interval",
        "bpm_linear_fit": round(60.0 / float(coef[1]), 3), "bpm_overlap_linear_fit": round(60.0 / float(coef_ov[1]), 3),
        "beat_period_s": round(float(coef_ov[1]), 5), "measure_s": round(4 * float(coef_ov[1]), 5),
        "bpm_iqr": [round(60 / float(np.percentile(ibi, 75)), 3), round(60 / float(np.percentile(ibi, 25)), 3)],
        "linear_fit_resid_rms_ms": round(1000 * float(np.sqrt(np.mean(resid ** 2))), 2),
        "linear_fit_resid_max_ms": round(1000 * float(np.abs(resid).max()), 2),
        "overlap_linear_fit_resid_rms_ms": round(1000 * float(np.sqrt(np.mean(resid_ov ** 2))), 2),
        "overlap_linear_fit_resid_max_ms": round(1000 * float(np.abs(resid_ov).max()), 2),
        "local_bpm_range": [round(float(lb.min()), 3), round(float(lb.max()), 3)] if len(lb) else None,
        "beat_level": {**level, "min_bpm": cfg.min_bpm, "gaps": gaps,
                       "note": "tracked beat rate folded into [min_bpm, 2*min_bpm) quarter-note BPM"},
        "bar_phase": ph, "phase_decision": decision, "phase_check": pcheck, "analyzer_split_measures": split,
        "tracking": {"method": "mean beat_this probability over all phones (offset-corrected to the source), "
                               "peak picking +-70 ms, p > 0.4, parabolic sub-frame" if ens is not None else
                     "source camera only",
                     "source_only_beats_disagreeing": int(len(src_only_bad)),
                     "source_only_beats_disagreeing_in_overlap": int(np.sum((src_only_bad >= ov0) & (src_only_bad <= ov1))),
                     "source_only_disagreeing_times": _r(src_only_bad, 3),
                     "cams_voting_in_overlap": int(np.median(ens["n_cams"][(ens["t"] >= ov0) & (ens["t"] <= ov1)]))
                     if ens is not None else 1},
        "source_cam": src_id, "n_quarter_beats": int(len(q_ref)), "n_measures": int(grid[:, 2].max() + 1) if len(grid) else 0,
        "grid_span": [round(float(grid[0, 0]), 4), round(float(grid[-1, 0]), 4)] if len(grid) else None,
        "caveat": CAVEAT,
    }
    warns = []
    if not pcheck["consistent"]:
        warns.append(f"kick/backbeat accents suggest the bar starts {pcheck['best']} eighth(s) later than the grid "
                     f"(margin {pcheck['margin']}); beat_this downbeats kept - check by ear with audio/clicks_overlay.wav")
    if ph["agreement"] < 0.8:
        warns.append(f"beat_this downbeats agree with one bar phase only {ph['agreement']:.0%} of the time")
    if tempo["overlap_linear_fit_resid_max_ms"] > 40:
        warns.append(f"beats in the overlap deviate up to {tempo['overlap_linear_fit_resid_max_ms']:.0f} ms from a "
                     f"constant tempo")
    if warns:
        tempo["warnings"] = warns

    # ---- audio on the reference clock (source's native rate), clicks
    xn, srn = decode_audio(src.path)
    sr_out = cfg.out_sr or srn
    y, t_audio0 = to_reference(xn, srn, src.slope, src.offset, ov0, ov1, sr_out)
    peak = float(np.max(np.abs(y)) + 1e-9)
    gain = min(1.0, 0.95 / peak)
    write_wav(work / "audio" / "audio_overlap.wav", y * gain, sr_out)
    clicks = click_track(len(y), sr_out, grid, t_audio0)
    mix = 0.6 * y * gain + clicks
    write_wav(work / "audio" / "clicks_overlay.wav", mix / max(1.0, float(np.abs(mix).max()) / 0.98), sr_out)
    meta = {"file": "audio_overlap.wav", "source_cam": src_id, "source_file": src.path.name,
            "sample_rate": sr_out, "channels": 1, "n_samples": int(len(y)),
            "reference_start": round(t_audio0, 7), "reference_end": round(t_audio0 + len(y) / sr_out, 7),
            "time_to_audio": round(-t_audio0, 7),
            "source_start": round(src.to_source(t_audio0), 7), "slope": src.slope, "offset": src.offset,
            "resampled_to_reference_clock": abs(src.slope - 1) > 1e-12 or sr_out != srn,
            "gain": round(gain, 5), "overlap": [ov0, ov1],
            "clock": "audio sample i is at reference time reference_start + i / sample_rate",
            "export_hint": f"python -m dancecap export {take.name} --capture <Name> --audio "
                           f"work/{take.name}/audio/audio_overlap.wav --audio-t0 {t_audio0:.7f}",
            "caveat": CAVEAT}
    (work / "audio" / "audio_overlap.json").write_text(json.dumps(meta, indent=1))
    (work / "audio" / "quality.json").write_text(json.dumps(
        {"overlap": [ov0, ov1], "ranking": ranking, "source_cam": src_id, "chosen_by": "forced" if cfg.source else "score",
         "cams": quality, "beat_offsets_vs_source": offsets,
         "snr_method": "corr of z-scored 40-band log-mel envelope (20 ms) with the leave-one-out median of the other "
                       "phones, lag-searched +-100 ms; SNR = r^2/(1-r^2)"}, indent=1))

    # ---- beats outputs
    out = [[round(float(t), 6), int(k)] for t, k in grid[:, :2]]
    (work / "beats" / "beats.json").write_text(json.dumps(out))
    shutil.copy2(work / "beats" / "beats.json", work / "beats.json")
    raw = {"source_cam": src_id, "checkpoint": cfg.checkpoint, "device": device, "fps": BEAT_THIS_FPS,
           "postprocessor": "minimal + parabolic sub-frame refinement",
           "clock": "reference = slope * source + offset (alignment.json)", "slope": src.slope, "offset": src.offset,
           "source": {"beats": _r(tr["beats"]), "downbeats": _r(tr["downbeats"]),
                      "beats_refined": _r(tr["beats_refined"]), "downbeats_refined": _r(tr["downbeats_refined"])},
           "reference": {"beats": _r(tr["ref_beats"]), "downbeats": _r(tr["ref_downbeats"])},
           "ensemble": {"beats": _r(b_ref), "downbeats": _r(d_ref), "quarter_beats": _r(q_ref),
                        "shifts_s": shifts, "enabled": ens is not None},
           "grid_columns": ["t", "type", "measure", "slot"], "grid": [[round(float(a), 6), int(b), int(c), int(d)]
                                                                     for a, b, c, d in grid],
           "other_cams": {c: {"beats": _r(t["ref_beats"]), "downbeats": _r(t["ref_downbeats"])}
                          for c, t in tracks.items() if c != src_id}}
    (work / "beats" / "beats_raw.json").write_text(json.dumps(raw))
    (work / "beats" / "tempo.json").write_text(json.dumps(tempo, indent=1))
    res = {"tempo": tempo, "quality": quality, "ranking": ranking, "source": src_id, "audio": meta,
           "offsets": offsets, "grid": grid, "work": work, "beats_ref": b_ref, "downbeats_ref": d_ref,
           "src_only_bad": src_only_bad}
    if plot:
        res["plots"] = plot_review(take, res, tracks, sigs, (ot, o), (lo_t, lo_o), (prof_sn, prof_low), (lt, lb),
                                   (ov0, ov1), ens)
    return res


def _r(a, nd: int = 5) -> list:
    return [round(float(v), nd) for v in np.asarray(a).ravel()]


# ------------------------------------------------------------------------------------------------ review
def plot_review(take: Take, res: dict, tracks: dict, sigs: dict, onset, low, profs, lbpm, ov, ens) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    review = take.work / "review"
    review.mkdir(parents=True, exist_ok=True)
    src_id, grid, tempo = res["source"], res["grid"], res["tempo"]
    src = take.source(src_id)
    x = sigs[src_id]
    t_ref = src.to_reference(np.arange(len(x)) / ANALYSIS_SR)
    ot, o = onset
    lt, lo = low
    tr = tracks[src_id]
    win = take.window
    bad = res["src_only_bad"]
    fig = plt.figure(figsize=(26, 17), dpi=110)
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.2, 1.4, 1.05], hspace=0.62, wspace=0.3)

    def envelope(ax, a, b, color="0.6"):
        m = (t_ref >= a) & (t_ref <= b)
        tt, xx = t_ref[m], x[m]
        step = max(1, len(xx) // 6000)
        n = len(xx) // step * step
        if n:
            sc = 1.0 / (np.abs(xx).max() + 1e-9)
            ax.fill_between(tt[:n:step], sc * xx[:n].reshape(-1, step).min(1), sc * xx[:n].reshape(-1, step).max(1),
                            color=color, lw=0)

    ax = fig.add_subplot(gs[0, :])
    envelope(ax, t_ref[0], t_ref[-1])
    ax.axvspan(*ov, color="tab:blue", alpha=0.08, label=f"all-camera overlap {ov[0]:.2f}-{ov[1]:.2f} s")
    ax.axvspan(*win, color="tab:orange", alpha=0.3, label=f"analysis window {win[0]:g}-{win[1]:g} s")
    if ens is not None:
        ax.plot(ens["t"], -1 + 0.3 * ens["n_cams"] / max(1, ens["n_cams"].max()), color="tab:green", lw=1.2,
                label=f"phones voting (0..{ens['n_cams'].max()})")
    ax2 = ax.twinx()
    if len(lbpm[0]):
        ax2.plot(lbpm[0], lbpm[1], color="tab:red", lw=1.4, label="local BPM (8-beat window)")
        ax2.axhline(tempo["bpm"], color="tab:red", ls=":", lw=1)
        ax2.set_ylim(tempo["bpm"] - 6, tempo["bpm"] + 6)
    ax2.set_ylabel("BPM", color="tab:red")
    ax.set_xlim(t_ref[0], t_ref[-1])
    ax.set_ylim(-1.05, 1.05)
    bl = tempo["beat_level"]
    ax.set_title(f"{take.name}: cam{src_id} audio (whole recording, reference clock). Tempo {tempo['bpm']:.2f} BPM "
                 f"quarter notes (overlap fit {tempo['bpm_overlap_linear_fit']:.2f}), measure {tempo['measure_s']:.3f} s,"
                 f" {tempo['n_measures']} measures. beat_this level {bl['tracked_bpm']} BPM x{bl['factor']:g}; analyzer"
                 f" split_measures would give {tempo['analyzer_split_measures'].get('resulting_grid_bpm_median')}"
                 f" (bug, not used)", fontsize=11.5, loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("reference time (s)")
    ax.set_yticks([])

    def beats_panel(ax, a, b, label_slots=False):
        envelope(ax, a, b)
        m = (ot >= a) & (ot <= b)
        if m.any():
            ax.plot(ot[m], -1 + 0.7 * o[m] / (o[m].max() + 1e-9), color="tab:purple", lw=0.8, alpha=0.8,
                    label="onset strength")
        m = (lt >= a) & (lt <= b)
        if m.any() and label_slots:
            ax.plot(lt[m], -1 + 0.7 * lo[m] / (lo[m].max() + 1e-9), color="tab:brown", lw=1.0, alpha=0.8,
                    label="kick/bass < 150 Hz onset")
        g = grid[(grid[:, 0] >= a) & (grid[:, 0] <= b)]
        sty = {1: ("tab:red", 2.0, 0.9), 2: ("tab:green", 1.6, 0.85), 3: ("0.3", 0.8, 0.6)}
        for k in (1, 2, 3):
            gg = g[g[:, 1] == k]
            c, lw, al = sty[k]
            ax.vlines(gg[:, 0], -1, 1, color=c, lw=lw, alpha=al, label=f"grid type {k}" + (
                " (slots 0,4)" if k == 1 else " (slots 3,6)" if k == 2 else " (other eighths)"))
        for t, _, mm, _ in g[g[:, 3] == 0]:
            ax.text(t, 1.03, f"m{int(mm)}", fontsize=8, ha="center", va="bottom", color="tab:red")
        rb = tr["ref_beats"][(tr["ref_beats"] >= a) & (tr["ref_beats"] <= b)]
        rd = tr["ref_downbeats"][(tr["ref_downbeats"] >= a) & (tr["ref_downbeats"] <= b)]
        ax.plot(rb, np.full(len(rb), 0.9), "v", color="k", ms=6, label=f"cam{src_id}-only beat_this beat")
        ax.plot(rd, np.full(len(rd), 0.78), "D", color="tab:orange", ms=6, label=f"cam{src_id}-only downbeat")
        bb = bad[(bad >= a) & (bad <= b)]
        if len(bb):
            ax.plot(bb, np.full(len(bb), 0.66), "x", color="red", ms=10, mew=2.5,
                    label="cam-only beat rejected by the all-phone vote")
        if label_slots:
            for t, k, mm, sl in g:
                ax.text(t, -1.06, f"{int(sl)}", fontsize=9, ha="center", va="top")
        ax.set_xlim(a, b)
        ax.set_ylim(-1.2, 1.15)
        ax.set_yticks([])

    ax = fig.add_subplot(gs[1, :])
    beats_panel(ax, *ov)
    ax.axvspan(*win, color="tab:orange", alpha=0.12)
    ax.set_title("all-camera overlap: zouk grid from the all-phone beat vote (8 eighths per measure, m = measure index)",
                 fontsize=12, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, ncol=8, bbox_to_anchor=(0, -0.1))
    ax = fig.add_subplot(gs[2, :])
    beats_panel(ax, win[0] - 0.5, win[1] + 0.5, label_slots=True)
    ax.axvspan(*win, color="tab:orange", alpha=0.08)
    ax.set_title(f"analysis window {win[0]:g}-{win[1]:g} s: eighth slot 0-7 under each line; red = type 1 (slots 0,4), "
                 f"green = type 2 (slots 3,6), grey = type 3", fontsize=12, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, ncol=8, bbox_to_anchor=(0, -0.13))
    ax.set_xlabel("reference time (s)")

    ax = fig.add_subplot(gs[3, 0])
    prof, prof_low = profs
    zb = (prof[::2] - prof[::2].mean()) / (prof[::2].std() + 1e-9)
    zl = (prof_low[::2] - prof_low[::2].mean()) / (prof_low[::2].std() + 1e-9)
    ax.bar(np.arange(8) - 0.2, zb, width=0.38, color="tab:purple", label="snare 1-4 kHz")
    ax.bar(np.arange(8) + 0.2, zl, width=0.38, color="tab:brown", label="kick/bass < 150 Hz")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(8), [f"{i}\nT{ZOUK_TYPES[i]}" for i in range(8)])
    ax.set_xlabel("eighth slot in the measure / zouk type")
    ax.set_ylabel("z-score")
    pc = tempo["phase_check"]
    ax.set_title(f"onsets folded on the grid: bar phase {'CONFIRMED' if pc['consistent'] else 'CHECK'} "
                 f"(set by {tempo['phase_decision']['by']})\nkick on 1 vs 3 + snare on 2 & 4, best of 8 eighth "
                 f"shifts, margin {pc['margin']:.2f}", fontsize=10,
                 loc="left")
    ax.legend(fontsize=8, loc="lower left")
    ax = fig.add_subplot(gs[3, 1])
    q = res["quality"]
    cams = res["ranking"]
    sc = [q[c]["score"] for c in cams]
    ax.barh(range(len(cams)), sc, color=["tab:red" if c == src_id else "tab:blue" for c in cams], alpha=0.85)
    for k, c in enumerate(cams):
        ax.text(0.15, k, f"cam{c}: SNR {q[c]['snr']['snr_db']:.1f} dB, {q[c]['level']['rms_dbfs']:.0f} dBFS, clip "
                         f"{100 * q[c]['level']['clip_frac']:.3f}%, bw {q[c]['level']['bandwidth_hz'] / 1000:.1f} kHz, "
                         f"conf {q[c]['beat_confidence']:.2f}", va="center", fontsize=8.5, color="black",
                bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=1))
    ax.set_xlim(0, max(sc) * 1.05)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("score = envelope SNR - clipping/bandwidth penalty + beat confidence")
    ax.set_title(f"phone music quality over the overlap: source = cam{src_id} (red)", fontsize=10, loc="left")
    ax = fig.add_subplot(gs[3, 2])
    off = res["offsets"]
    cs = sorted(off)
    ax.errorbar([off[c]["median_ms"] for c in cs], range(len(cs)), xerr=[off[c]["mad_ms"] for c in cs], fmt="o",
                color="k", capsize=3)
    ax.axvline(0, color="tab:red", lw=1)
    ax.set_yticks(range(len(cs)), [f"cam{c} ({off[c]['matched']}/{off[c]['of']})" for c in cs], fontsize=9)
    ax.set_xlabel(f"beat time minus cam{src_id} beat time (ms), median +- robust sd")
    ax.set_title("per-phone beat offsets on the reference clock\n(= audio-clock residual + sound path)", fontsize=10,
                 loc="left")
    p = review / "beats_waveform.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return [p]
