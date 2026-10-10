"""Assemble the voice-to-voice class soundtrack (agent V1): film/audio/class_mix_v2v.wav + per-speaker stems.

    converted speech of the two teachers (accepted pieces, master clock)
  + the ORIGINAL where nothing was converted (the opening 0-~9.7 s = the owner's two jokes, overlaps of the two voices, stretches too short to convert,
    pieces the quality gate rejected: those use the cleaned best-phone cut that was sent to the voice changer)
  + a low-level ROOM-TONE bed synthesised from the original mix's own stationary noise spectrum (a low percentile of every frequency bin over time, random
    phase: it has no speech in it by construction, so nothing can be heard twice), ducked under speech
  -> loudness -16 LUFS integrated, true peak <= -1 dBTP (static gain, look-ahead limiter only if a peak needs it).

48 kHz stereo, sample 0 = master time 0, the length of class_mix.wav (360.477 s). The stems (larissa, kadu = every piece of that teacher, converted or cleaned original; original = the class mix where it was kept as it was; bed; 24-bit) are scaled with the
SAME gain as the mix so that their sum is the mix before the limiter (a limiter touches the mix only if a peak needs it; the report says how much).
The original class_mix.wav is never changed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

from . import SPEECH
from . import v2v as V

SR = V.SR
TARGET_LUFS = -16.0
TARGET_TP = -1.0
BED_BELOW_SPEECH_DB = 26.0     # the room tone sits this far under the speech level in pauses
BED_DUCK_DB = 6.0              # and this much lower again while a teacher speaks
EDGE_FADE_S = 0.03             # converted pieces fade in/out over this
ORIG_FADE_S = 0.12             # original <-> converted cross-fade
SPEECH_TARGET_DB = -24.0       # speech level (speech_level_db) every piece is brought to before the final loudness solve


# ------------------------------------------------------------------------------------------------ small pure helpers


def frame_rms_db(x: np.ndarray, frame_s: float = 0.05, sr: int = SR) -> np.ndarray:
    n = int(frame_s * sr)
    m = len(x) // n
    if m == 0:
        return np.array([-120.0])
    e = (x[:m * n].reshape(m, n).astype(np.float64) ** 2).mean(axis=1)
    return 10.0 * np.log10(e + 1e-12)


def speech_level_db(x: np.ndarray, sr: int = SR, floor_db: float = -55.0) -> float:
    """level of the speech in `x`: the 90th percentile of the 50 ms frame levels that are above `floor_db`"""
    e = frame_rms_db(x, 0.05, sr)
    e = e[e > floor_db]
    return float(np.percentile(e, 90)) if len(e) else -120.0


def active_level_db(x: np.ndarray, sr: int = SR, span_db: float = 25.0) -> float:
    """power-mean level (dB) of the 50 ms frames within `span_db` of the loudest frame: the level of the speech itself, stable also for a piece of two words
    (the 90th percentile of v2v's `speech_level_db` jumps with the number of frames)"""
    e = frame_rms_db(x, 0.05, sr)
    if len(e) == 0 or e.max() < -90:
        return -120.0
    sel = e[e >= e.max() - span_db]
    return float(10.0 * np.log10(np.mean(10.0 ** (sel / 10.0))))


def cos_fade(n: int, k: int) -> np.ndarray:
    """a window of n samples that rises over k samples, stays 1, falls over k samples (raised cosine)"""
    k = min(k, n // 2)
    w = np.ones(n, dtype=np.float32)
    if k > 0:
        r = (0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)).astype(np.float32)
        w[:k] = r
        w[n - k:] = r[::-1]
    return w


def place(track: np.ndarray, x: np.ndarray, i0: int, fade: int) -> None:
    """add x into track at sample i0 with raised-cosine edges (clipped to the track)"""
    a, b = max(i0, 0), min(i0 + len(x), len(track))
    if b <= a:
        return
    seg = (x * cos_fade(len(x), fade))[a - i0:b - i0]
    track[a:b] += seg


def smooth_mask(n: int, spans: list[tuple[float, float]], fade_s: float, sr: int = SR) -> np.ndarray:
    """1 inside the spans (seconds), 0 outside, with raised-cosine ramps of fade_s centred on the span edges"""
    m = np.zeros(n, dtype=np.float32)
    h = int(fade_s * sr / 2)
    for a, b in spans:
        i0, i1 = int(round(a * sr)), int(round(b * sr))
        m[max(i0, 0):min(i1, n)] = 1.0
    if h > 0:
        k = 2 * h
        ker = np.hanning(k + 2)[1:-1].astype(np.float32)
        ker /= ker.sum()
        m = np.convolve(m, ker, mode="same").astype(np.float32)
    return np.clip(m, 0.0, 1.0)


def duck_gain(activity: np.ndarray, depth_db: float, attack_s: float = 0.08, release_s: float = 0.4, sr: int = SR) -> np.ndarray:
    """gain curve for the bed: 1 in pauses, 10**(-depth_db/20) while `activity` (0..1) is up; fast attack, slow release (one-pole on a 10 ms grid, linear-interpolated back)"""
    step = int(0.01 * sr)
    a = activity[:len(activity) // step * step].reshape(-1, step).max(axis=1)
    ka, kr = math.exp(-0.01 / attack_s), math.exp(-0.01 / release_s)
    env = np.zeros_like(a)
    y = 0.0
    for i, v in enumerate(a):
        y = ka * y + (1 - ka) * v if v > y else kr * y + (1 - kr) * v
        env[i] = y
    t = np.arange(len(activity)) / step - 0.5
    up = np.interp(t, np.arange(len(env)), env)
    return (10 ** (-depth_db / 20.0)) ** up


def room_tone(orig: np.ndarray, n: int, sr: int = SR, *, block_s: float = 10.0, pct: float = 15.0, nfft: int = 2048, seed: int = 7,
              mod_db: float = 2.0) -> np.ndarray:
    """Stationary room tone [n, ch] in the colour of `orig` (stereo): per channel the `pct`-th percentile of every frequency bin's power over each
    `block_s` block (speech never lasts in every frame of a bin, so what remains is the room), random phase, inverse STFT; the blocks are interpolated
    over time, and the level follows the original's noise floor by at most +-mod_db. Unit-free (the caller sets the level)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, orig.shape[1]), dtype=np.float32)
    hop = nfft // 4
    for ch in range(orig.shape[1]):
        f, t, Z = signal.stft(orig[:, ch].astype(np.float64), sr, nperseg=nfft, noverlap=nfft - hop, window="hann")
        P = np.abs(Z) ** 2
        nb = max(1, int(round(len(t) * (t[1] - t[0]) / block_s))) if len(t) > 1 else 1
        edges = np.linspace(0, P.shape[1], nb + 1).astype(int)
        prof = np.stack([np.percentile(P[:, a:max(b, a + 1)], pct, axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)   # [bins, blocks]
        centres = (edges[:-1] + edges[1:]) / 2.0
        mag = np.empty((P.shape[0], n // hop + 3))
        cols = np.arange(mag.shape[1]) * (P.shape[1] / mag.shape[1])
        for k in range(P.shape[0]):
            mag[k] = np.sqrt(np.interp(cols, centres, prof[k]))
        ph = rng.uniform(0, 2 * np.pi, mag.shape)
        _, y = signal.istft(mag * np.exp(1j * ph), sr, nperseg=nfft, noverlap=nfft - hop, window="hann")
        y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
        # slow level follows the original's floor (block percentile of the broadband level)
        lev = np.array([10 * np.log10(prof[:, b].sum() + 1e-18) for b in range(prof.shape[1])])
        if len(lev) > 1:
            rel = np.clip(lev - lev.mean(), -mod_db, mod_db)
            g = 10 ** (np.interp(np.arange(n) / sr, centres * hop / sr, rel) / 20.0)
            y = y * g
        out[:, ch] = y.astype(np.float32)
    return out


def solve_loudness(mix: np.ndarray, tmp: Path, target_lufs: float = TARGET_LUFS, target_tp: float = TARGET_TP, sr: int = SR) -> tuple[float, np.ndarray, dict]:
    """static gain (dB) such that the mix after a look-ahead peak limiter measures target_lufs (ffmpeg ebur128), true peak <= target_tp; returns (gain_db, limited mix, report)"""
    from .. import film_recap_audio as fa
    tmp.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp), mix, sr, subtype="FLOAT")
    m0 = fa.measure_loudness(tmp)
    ceiling = target_tp - 0.4
    log = []
    final = None
    gain = target_lufs - m0["lufs"]
    for outer in range(4):
        for it in range(6):
            y = fa.lookahead_limit(mix * (10 ** (gain / 20.0)), ceiling)
            sf.write(str(tmp), y, sr, subtype="FLOAT")
            m = fa.measure_loudness(tmp)
            err = target_lufs - m["lufs"]
            log.append({"ceiling_db": round(ceiling, 2), "gain_db": round(gain, 3), "lufs": m["lufs"], "true_peak_dbtp": m["true_peak_dbtp"]})
            if abs(err) <= 0.05:
                break
            gain += err
        final = (gain, y, m)
        if m["true_peak_dbtp"] is not None and m["true_peak_dbtp"] <= target_tp - 0.02:
            break
        ceiling -= max(0.2, (m["true_peak_dbtp"] or 0.0) - target_tp + 0.1)
    gain, y, m = final
    pre = np.abs(mix * (10 ** (gain / 20.0))).max(axis=1)
    post = np.abs(y).max(axis=1)
    red = 20 * np.log10(np.maximum(post, 1e-9) / np.maximum(pre, 1e-9))
    rep = {"before": m0, "gain_db": round(gain, 3), "solve": log, "limiter_samples_touched_pct": round(float((red < -0.05).mean() * 100), 3),
           "limiter_max_reduction_db": round(float(-red.min()), 2)}
    tmp.unlink(missing_ok=True)
    return gain, y, rep


# ------------------------------------------------------------------------------------------------ the assembly


def load_decisions(path: Path | None = None) -> dict:
    p = path or V.V2V / "decisions.json"
    return json.loads(p.read_text(encoding="utf-8"))


def assemble(pieces: list[V.Piece], kept: list[dict], decisions: dict, *, orig_path: Path, out_wav: Path, stems_dir: Path, report_path: Path,
             bed_below_db: float = BED_BELOW_SPEECH_DB, bed_duck_db: float = BED_DUCK_DB, pre_dir: str = "pre", log=print, loader=None) -> dict:
    """decisions {piece id: {"variant": name | None, "cam": phone}}: None = the cleaned original (best phone, light denoise) for that piece. Both kinds go into the
    speaking teacher's stem; the `original` stem holds only the class mix kept as it was (opening, overlaps of the two voices, too-short stretches)."""
    orig, sr = sf.read(str(orig_path), dtype="float32", always_2d=True)
    assert sr == SR and orig.shape[1] == 2, (sr, orig.shape)
    n = len(orig)
    om = orig.mean(axis=1)
    stems = {"larissa": np.zeros(n, np.float32), "kadu": np.zeros(n, np.float32)}
    fb = int(EDGE_FADE_S * SR)
    rows = []
    # 1) one speech level for every piece (converted or the cleaned original), so the teachers and the phones sit at the same loudness
    target_db = SPEECH_TARGET_DB
    orig_spans: list[tuple[float, float]] = []
    for p in pieces:
        d = decisions.get(p.id) or {}
        v = d.get("variant")
        i0 = int(round(p.t0 * SR))
        if loader is not None:                                     # V2: the caller supplies each piece's audio: (x, label, converted) or None = the class mix as it was
            got = loader(p, d)
            if got is None:
                orig_spans.append((p.t0, p.t1))
                rows.append({"id": p.id, "speaker": p.speaker, "t0": p.t0, "t1": p.t1, "source": "original (class mix)", "why": d.get("why")})
                continue
            x, label, is_conv = got
            a0, a1 = max(0, int(round((p.w0 - p.t0) * SR))), int(round((p.w1 - p.t0) * SR))
            lvl = active_level_db(x[a0:a1] if a1 - a0 > SR // 4 else x)     # the level of the WORDS of the piece (not of its padding)
            g = float(np.clip(target_db - 3.2 - lvl, -20.0, 20.0))     # 3.2 dB = the measured difference P90 level - active level of the words
            y = x * (10 ** (g / 20.0))
            for a, b in p.keep:
                j0, j1 = int(round((a - p.t0) * SR)), int(round((b - p.t0) * SR))
                y[max(j0, 0):max(j1, 0)] = 0.0
                orig_spans.append((a, b))
            place(stems["larissa" if p.speaker == "Larissa" else "kadu"], y, i0, fb if is_conv else int(ORIG_FADE_S * SR / 2))
            rows.append({"id": p.id, "speaker": p.speaker, "t0": p.t0, "t1": p.t1, "w0": p.w0, "w1": p.w1, "source": label, "gain_db": round(g, 2), "keep": p.keep, "why": d.get("why")})
            continue
        if v:
            x, _ = sf.read(str(V.V2V / "aligned" / v / f"{p.id}.wav"), dtype="float32")
            g = float(np.clip(target_db - speech_level_db(x), -9.0, 9.0))
            y = x * (10 ** (g / 20.0))
            for a, b in p.keep:                                    # the other voice speaks too: original there
                j0, j1 = int(round((a - p.t0) * SR)), int(round((b - p.t0) * SR))
                y[max(j0, 0):max(j1, 0)] = 0.0
                orig_spans.append((a, b))
            place(stems["larissa" if p.speaker == "Larissa" else "kadu"], y, i0, fb)
            rows.append({"id": p.id, "speaker": p.speaker, "t0": p.t0, "t1": p.t1, "source": f"converted ({v}, phone {d.get('cam') or p.cam})", "gain_db": round(g, 2),
                         "keep": p.keep})
        else:
            f = V.V2V / pre_dir / f"{p.id}.{p.cam}.wav"
            if f.exists():                                          # not converted: the cleaned best-phone audio, level matched
                x, _ = sf.read(str(f), dtype="float32")
                g = float(np.clip(target_db - speech_level_db(x), -9.0, 9.0))
                place(stems["larissa" if p.speaker == "Larissa" else "kadu"], x * (10 ** (g / 20.0)), i0, int(ORIG_FADE_S * SR / 2))
                rows.append({"id": p.id, "speaker": p.speaker, "t0": p.t0, "t1": p.t1, "source": f"original (cleaned phone {p.cam})", "gain_db": round(g, 2),
                             "why": d.get("why")})
            else:
                orig_spans.append((p.t0, p.t1))
                rows.append({"id": p.id, "speaker": p.speaker, "t0": p.t0, "t1": p.t1, "source": "original (class mix)", "why": d.get("why")})
    overall = target_db
    target = {"Larissa": target_db, "Kadu": target_db}
    for k in kept:
        pad = 0.12 if k["t1"] - k["t0"] < 5 else 0.0
        orig_spans.append((max(0.0, k["t0"] - pad), k["t1"] + pad))
    # 2) the original class mix where nothing was converted: level-matched to the converted speech
    mask = smooth_mask(n, orig_spans, ORIG_FADE_S)
    mask_speech = mask > 0.99
    lv_o = speech_level_db(om[mask_speech]) if mask_speech.any() else None
    g_o = float(np.clip(overall - lv_o, -12.0, 12.0)) if lv_o is not None else 0.0
    orig_stem = orig * mask[:, None] * (10 ** (g_o / 20.0))
    # 3) the bed: room tone in the colour of the original, ducked under speech, absent where the original itself plays
    bed = room_tone(orig, n)
    bed_rms = float(np.sqrt((bed.astype(np.float64) ** 2).mean()))
    speech_ref = overall
    bed *= 10 ** ((speech_ref - bed_below_db) / 20.0) / (bed_rms + 1e-12)
    activity = smooth_mask(n, [(p.w0, p.w1) for p in pieces], 0.05)         # a teacher speaks (converted or the cleaned original)
    activity = np.maximum(activity, mask)
    bed *= duck_gain(activity, bed_duck_db)[:, None] * (1.0 - mask)[:, None]
    # 4) sum, loudness
    lar = np.stack([stems["larissa"]] * 2, axis=1)
    kad = np.stack([stems["kadu"]] * 2, axis=1)
    orig_total = orig_stem
    mix = lar + kad + orig_total + bed
    gain, y, lrep = solve_loudness(mix.astype(np.float32), out_wav.with_name("_v2v_probe.wav"))
    g_lin = 10 ** (gain / 20.0)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), np.clip(y, -1, 1), SR, subtype="PCM_16")
    stems_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in (("larissa", lar), ("kadu", kad), ("original", orig_total), ("bed", bed)):
        sf.write(str(stems_dir / f"{name}.wav"), np.clip(arr * g_lin, -1, 1), SR, subtype="PCM_24")
    from .. import film_recap_audio as fa
    final = fa.measure_loudness(out_wav)
    rep = {"schema": "dancecap.speech/1/v2v_mix", "length_s": round(n / SR, 3), "rate": SR, "channels": 2, "orig": str(orig_path.name),
           "pieces": rows, "converted": sum(1 for r in rows if str(r["source"]).startswith("converted")), "original_pieces": sum(1 for r in rows if str(r["source"]).startswith("original")),
           "kept_original": kept, "speech_target_db": {k: (None if v is None else round(v, 2)) for k, v in target.items()}, "original_gain_db": round(g_o, 2),
           "bed": {"below_speech_db": bed_below_db, "duck_db": bed_duck_db, "level_db": round(speech_ref - bed_below_db, 2)},
           "loudness": lrep, "final": final, "sample_peak_db": round(fa.sample_peak_db(y), 2)}
    report_path.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"mix: {rep['converted']} converted pieces, {rep['original_pieces']} original, LUFS {final['lufs']}, TP {final['true_peak_dbtp']}")
    return rep
