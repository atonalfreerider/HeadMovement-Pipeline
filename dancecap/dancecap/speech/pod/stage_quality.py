"""Stage `quality`: per-source speech-quality statistics (no reference transcript needed).

    python -m dancecap.speech.pod.run quality [--cams ...] [--force]

From the 16 kHz audio and the Silero frame probabilities (stage vad), overall and per 30-s block:
  speech_frac     share of 32-ms frames with speech probability >= 0.5
  speech_db       median frame RMS (dBFS) of speech frames;  noise_db  10th percentile of frame RMS over frames with probability < 0.2
  snr_db          speech_db - noise_db (a crude but comparable figure)
  clip_frac       share of samples at |x| >= 0.99;  peak_db  peak level;  dc  mean sample value
  rolloff_95_hz   frequency below which 95 % of the speech-frame spectral energy lies (low values = a narrow-band / heavily compressed
                  codec: the phones record AAC at 57-256 kbps)
  hf_ratio        energy above 4 kHz / total in speech frames
  band_snr_db     SNR in 300-3400 Hz (the band that carries intelligibility)
The ASR-based statistics (word confidence, agreement between engines) are computed on the PC from the engines' outputs
(dancecap.speech.report).

Output raw/<cam>/quality.json.
"""
from __future__ import annotations

import time

import numpy as np

from . import common as C
from . import stage_vad


def block_stats(x: np.ndarray, probs: np.ndarray, step: float, t0: float, t1: float) -> dict:
    a, b = int(t0 * C.SR), min(len(x), int(t1 * C.SR))
    seg = x[a:b]
    if seg.size < 1600:
        return {"t0": round(t0, 2), "t1": round(t1, 2), "speech_frac": 0.0}
    fr = C.frame_rms_db(seg, 512, 512)                 # one value per VAD frame
    p = probs[int(t0 / step):int(t0 / step) + len(fr)]
    fr = fr[:len(p)]
    sp = fr[p >= 0.5]
    nz = fr[p < 0.2]
    speech_db = float(np.median(sp)) if sp.size >= 5 else None
    noise_db = float(np.percentile(nz, 10)) if nz.size >= 5 else (float(np.percentile(fr, 5)) if fr.size else None)
    out = {"t0": round(t0, 2), "t1": round(t1, 2), "speech_frac": round(float(np.mean(p >= 0.5)) if len(p) else 0.0, 3),
           "speech_db": round(speech_db, 2) if speech_db is not None else None, "noise_db": round(noise_db, 2) if noise_db is not None else None,
           "snr_db": round(speech_db - noise_db, 2) if speech_db is not None and noise_db is not None else None,
           "clip_frac": round(float(np.mean(np.abs(seg) >= 0.99)), 6), "peak_db": round(C.rms_db(np.array([np.abs(seg).max()])), 2)}
    return out


def spectral(x: np.ndarray, probs: np.ndarray, step: float) -> dict:
    """Rolloff / high-frequency ratio / band SNR from the average power spectrum of speech and non-speech frames."""
    n = 512
    nfr = min(len(x) // n, len(probs))
    if nfr < 10:
        return {}
    frames = x[:nfr * n].reshape(nfr, n) * np.hanning(n)[None, :].astype(np.float32)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, d=1 / C.SR)
    p = probs[:nfr]
    sp, nz = spec[p >= 0.5], spec[p < 0.2]
    if len(sp) < 5:
        return {}
    avg = sp.mean(axis=0)
    cum = np.cumsum(avg) / avg.sum()
    out = {"rolloff_95_hz": float(freqs[int(np.searchsorted(cum, 0.95))]), "rolloff_99_9_hz": float(freqs[min(int(np.searchsorted(cum, 0.999)), len(freqs) - 1)]),
           "hf_ratio": round(float(avg[freqs >= 4000].sum() / avg.sum()), 5)}
    band = (freqs >= 300) & (freqs <= 3400)
    if len(nz) >= 5:
        out["band_snr_db"] = round(float(10 * np.log10(avg[band].sum() / (np.percentile(nz[:, band].sum(axis=1), 25) + 1e-12))), 2)
    return out


def run(cams: dict, force: bool = False, block: float = 30.0) -> dict:
    done = {}
    for i, (cam, wav) in enumerate(cams.items()):
        out = C.cam_dir(cam) / "quality.json"
        if out.exists() and not force:
            continue
        t0 = time.time()
        x = C.load_audio(wav)
        dur = len(x) / C.SR
        vad = stage_vad.load(cam)
        if vad is None:
            probs, step = C.vad_probabilities(x)
        else:
            probs, step = np.asarray(vad["probs"], np.float32), vad["frame_step"]
        blocks = [block_stats(x, probs, step, t, min(dur, t + block)) for t in np.arange(0, dur, block)]
        whole = block_stats(x, probs, step, 0.0, dur)
        whole.update(spectral(x, probs, step))
        whole["dc"] = round(float(x.mean()), 6)
        whole["duration"] = round(dur, 3)
        C.write_json(out, {"stage": "quality", "version": C.STAGE_VERSION, "cam": cam, "file": wav.name, "overall": whole, "blocks": blocks,
                           "seconds": round(time.time() - t0, 2)})
        C.log(f"quality {cam}: snr {whole.get('snr_db')} dB, speech {whole.get('speech_frac')}, rolloff {whole.get('rolloff_95_hz')} Hz")
        done[cam] = whole.get("snr_db")
    return done
