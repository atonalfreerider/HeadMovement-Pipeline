"""Checks of the cloned-voice class mix (agent V2): word-onset alignment, level continuity at piece joins, clipping, spectrogram figure.

Pure numpy / scipy on the local machine (no network); the whole-class ASR and the voice-identity check run on the pod (`pod/asr_v2v.py`).
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy import signal

SR = 48000


def speech_env_db(x: np.ndarray, sr: int = SR, frame_s: float = 0.01, band: tuple[float, float] = (250.0, 3500.0)) -> np.ndarray:
    sos = signal.butter(4, [band[0], min(band[1], sr / 2 - 100)], btype="band", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, x.astype(np.float64))
    n = int(round(frame_s * sr))
    m = len(y) // n
    return 10.0 * np.log10((y[:m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12)


def onset_near(env: np.ndarray, t: float, frame_s: float = 0.01, before: float = 0.25, after: float = 0.5, ctx: float = 3.0) -> float | None:
    """time (s) of the first frame in [t - before, t + after] where the envelope crosses the midpoint (dB) between the local noise floor (10th percentile
    of +-ctx) and the peak of the window; None when the window has no clear rise (peak less than 8 dB above the floor)"""
    i0, i1 = max(0, int((t - before) / frame_s)), min(len(env), int((t + after) / frame_s))
    if i1 - i0 < 5:
        return None
    floor = float(np.percentile(env[max(0, int((t - ctx) / frame_s)):min(len(env), int((t + ctx) / frame_s))], 10))
    seg = env[i0:i1]
    peak = float(seg.max())
    if peak - floor < 8.0:
        return None
    thr = floor + 0.5 * (peak - floor)
    j = int(np.argmax(seg > thr))
    if j > 0:                                                       # linear interpolation of the crossing
        a, b = seg[j - 1], seg[j]
        frac = (thr - a) / (b - a) if b > a else 0.0
        return (i0 + j - 1 + frac + 0.5) * frame_s
    return (i0 + 0.5) * frame_s


def pick_clean_words(words: list[dict], n: int = 20, min_gap: float = 0.25, t_min: float = 12.0, ref_env: np.ndarray | None = None, tol_s: float = 0.06) -> list[dict]:
    """n words spread over the class that start after a pause of >= min_gap and are >= 0.15 s long. With `ref_env` (the speech-band envelope of a phone on the
    master clock) only words whose onset in that phone is within tol_s of the transcript time are kept: the reference measurement is then trustworthy
    (a noisy onset in the phone would make the comparison meaningless)."""
    cands = []
    for a, b in zip(words, words[1:]):
        if b["t0"] >= t_min and b["t0"] - a["t1"] >= min_gap and b["t1"] - b["t0"] >= 0.15 and b["kind"] == "word":
            if ref_env is not None:
                o = onset_near(ref_env, b["t0"])
                if o is None or abs(o - b["t0"]) > tol_s:
                    continue
            cands.append(b)
    if len(cands) <= n:
        return cands
    idx = np.linspace(0, len(cands) - 1, n).round().astype(int)
    return [cands[i] for i in idx]


def onset_report(mix: np.ndarray, ref: np.ndarray, words: list[dict], sr: int = SR) -> dict:
    """for each word: onset in `mix` minus the onset in `ref` (a phone on the master clock, the same words as spoken) and minus the transcript time"""
    em, er = speech_env_db(mix, sr), speech_env_db(ref, sr)
    rows = []
    for w in words:
        om, orf = onset_near(em, w["t0"]), onset_near(er, w["t0"])
        rows.append({"t0": round(w["t0"], 3), "text": w["text"], "onset_mix": None if om is None else round(om, 3), "onset_ref": None if orf is None else round(orf, 3),
                     "d_mix_vs_ref_ms": None if om is None or orf is None else round((om - orf) * 1000, 1),
                     "d_mix_vs_transcript_ms": None if om is None else round((om - w["t0"]) * 1000, 1)})
    d = [abs(r["d_mix_vs_ref_ms"]) for r in rows if r["d_mix_vs_ref_ms"] is not None]
    return {"rows": rows, "measured": len(d), "max_abs_ms": max(d) if d else None, "median_abs_ms": float(np.median(d)) if d else None,
            "within_80ms": sum(1 for x in d if x <= 80)}


def join_steps(pieces: list[dict], mix: np.ndarray, sr: int = SR) -> dict:
    """level continuity at the joins of consecutive pieces (less than 1.5 s apart): the speech level (90th percentile of the 50 ms frames above -55 dB) of each
    piece in the final mix and the step between consecutive pieces, and the step of the room level (the quietest 100 ms of each piece in the mix). pieces = [{id, t0, t1}]"""
    m = mix.mean(axis=1) if mix.ndim > 1 else mix
    n = int(0.05 * sr)

    def frames(a: float, b: float):
        x = m[max(0, int(a * sr)):int(b * sr)]
        k = len(x) // n
        if k < 2:
            return np.array([])
        return 10 * np.log10((x[:k * n].reshape(k, n).astype(np.float64) ** 2).mean(axis=1) + 1e-12)

    def speech(e):
        s = e[e > -55]
        return float(np.percentile(s, 90)) if len(s) else None

    def active(e, span=25.0):
        """power-mean level of the frames within `span` dB of the loudest (the speech of the piece's words)"""
        if len(e) == 0 or e.max() < -90:
            return None
        sel = e[e >= e.max() - span]
        return float(10 * np.log10(np.mean(10 ** (sel / 10))))

    def floor(e):
        """the room level of a piece: its quietest 100 ms (two consecutive 50 ms frames)"""
        return float(np.min((e[:-1] + e[1:]) / 2)) if len(e) > 1 else None

    steps = []
    for a, b in zip(pieces, pieces[1:]):
        if b["t0"] - a["t1"] > 1.5:
            continue
        ea, eb = frames(a["t0"], a["t1"]), frames(b["t0"], b["t1"])
        sa = active(frames(a.get("w0", a["t0"]), a.get("w1", a["t1"])))
        sb = active(frames(b.get("w0", b["t0"]), b.get("w1", b["t1"])))
        if sa is None or sb is None:
            continue
        t = 0.5 * (a["t1"] + b["t0"])
        fa, fb = floor(ea), floor(eb)
        steps.append({"a": a["id"], "b": b["id"], "t": round(t, 2), "speech_step_db": round(sb - sa, 2),
                      "floor_step_db": None if fa is None or fb is None else round(fb - fa, 2)})
    big = [s for s in steps if abs(s["speech_step_db"]) > 3.0]
    fl = [abs(s["floor_step_db"]) for s in steps if s["floor_step_db"] is not None]
    return {"joins": len(steps), "max_abs_speech_step_db": max((abs(s["speech_step_db"]) for s in steps), default=0.0), "over_3db": big,
            "max_abs_floor_step_db": round(max(fl, default=0.0), 2), "steps": steps}


def clipping(mix: np.ndarray) -> dict:
    a = np.abs(mix)
    return {"peak": float(a.max()), "samples_ge_0.999": int((a >= 0.999).sum())}


def spectrogram_png(wavs: dict[str, np.ndarray], t0: float, t1: float, out_png, sr: int = SR) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(wavs), 1, figsize=(14, 3 * len(wavs)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (name, x) in zip(axs, wavs.items()):
        seg = (x.mean(axis=1) if x.ndim > 1 else x)[int(t0 * sr):int(t1 * sr)]
        f, t, S = signal.spectrogram(seg, sr, nperseg=1024, noverlap=768)
        ax.pcolormesh(t + t0, f / 1000, 10 * np.log10(S + 1e-12), shading="auto", vmin=-110, vmax=-40, cmap="magma")
        ax.set_ylim(0, 8)
        ax.set_ylabel(f"{name}\nkHz")
    axs[-1].set_xlabel("master seconds")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def piece_onset_errors(audio: np.ndarray, ref_env: np.ndarray, word_t0: list[float], sr: int = SR, tol_s: float = 0.06, min_gap_before: list[float] | None = None) -> list[dict]:
    """Onset errors of the words of one PIECE: `audio` (the piece on its own time base, 48 kHz), `ref_env` = speech_env_db of the phone cut of the same piece,
    `word_t0` = the transcript word starts in piece time. A word counts only when the phone's own onset is within tol_s of the transcript (a trustworthy
    reference) and the audio has a clear onset near it. Returns [{t0, err_ms (audio onset - phone onset), vs_transcript_ms}]."""
    ea = speech_env_db(audio, sr)
    out = []
    for i, t0 in enumerate(word_t0):
        if min_gap_before is not None and min_gap_before[i] < 0.12:
            continue
        orf = onset_near(ref_env, t0)
        if orf is None or abs(orf - t0) > tol_s:
            continue
        oa = onset_near(ea, t0)
        if oa is None:
            continue
        out.append({"t0": round(t0, 3), "err_ms": round((oa - orf) * 1000, 1), "vs_transcript_ms": round((oa - t0) * 1000, 1)})
    return out


def local_lag_ms(env_mix: np.ndarray, env_ref: np.ndarray, t0: float, t1: float, frame_s: float = 0.01, max_lag_s: float = 0.25) -> float | None:
    """lag (ms, positive = the mix is LATER) of the speech-band envelope of the mix against the reference over the window [t0, t1] (cross-correlation of the
    z-scored 10 ms envelopes, parabolic peak); None when the window has no speech structure (reference envelope range < 8 dB)"""
    i0, i1 = max(0, int(t0 / frame_s)), min(len(env_ref), int(t1 / frame_s))
    k = int(max_lag_s / frame_s)
    if i1 - i0 < 10 or i0 - k < 0 or i1 + k > len(env_mix):
        return None
    r = env_ref[i0:i1]
    if r.max() - r.min() < 8.0:
        return None
    r = (r - r.mean()) / (r.std() + 1e-9)
    cc = []
    for d in range(-k, k + 1):
        m = env_mix[i0 + d:i1 + d]
        m = (m - m.mean()) / (m.std() + 1e-9)
        cc.append(float(np.mean(r * m)))
    cc = np.array(cc)
    j = int(np.argmax(cc))
    frac = 0.0
    if 0 < j < len(cc) - 1:
        y0, y1, y2 = cc[j - 1], cc[j], cc[j + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return float((j - k + frac) * frame_s * 1000)


def word_lag_report(mix: np.ndarray, ref: np.ndarray, words: list[dict], sr: int = SR, lead: float = 0.15, tail: float = 0.30) -> dict:
    """for each word: the local envelope lag of the mix against the reference phone over [t0 - lead, t1 + tail]; the words need not be isolated"""
    em, er = speech_env_db(mix, sr), speech_env_db(ref, sr)
    rows = []
    for w in words:
        lag = local_lag_ms(em, er, w["t0"] - lead, min(w["t1"] + tail, w["t0"] + 1.2))
        rows.append({"t0": round(w["t0"], 3), "text": w["text"], "lag_ms": None if lag is None else round(lag, 1)})
    d = [abs(r["lag_ms"]) for r in rows if r["lag_ms"] is not None]
    return {"rows": rows, "measured": len(d), "max_abs_ms": max(d) if d else None, "median_abs_ms": float(np.median(d)) if d else None, "within_80ms": sum(1 for x in d if x <= 80)}
