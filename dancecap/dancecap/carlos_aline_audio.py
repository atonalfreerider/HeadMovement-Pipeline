"""carlos_aline: the soundtrack of the Unity capture (agent C6, 2026-10-10): both studio songs on the reference clock, the heat's own sound where no song plays.

    python -m dancecap.carlos_aline_audio build [--out work/carlos_aline/audio/heat_studio.wav]
    python -m dancecap.carlos_aline_audio verify          # the sync figures against both videos' audio (also written by build)

What it builds (reference clock = camera 01's video time, sample i at reference_start + i / sr, reference_start = 0):
  0 - 13.31 s        camera 01's own sound (the announcement / entrance sound), faded out into the first song
  13.31 - 120.1 s    Kojo Funds - Check (With Raye), the studio mp3 time-STRETCHED to the DJ's tempo (x0.9615 of the studio tempo, pitch kept: song.pitch_check says which)
  120.1 - 122.17 s   equal-power crossfade Kojo -> Guetta (the governing grid's seam: song/segments.json governing.kojo_until_s / guetta_from_s)
  122.17 - 194.25 s  David Guetta - Say My Name, stretched to x0.9425
  194.25 - end       camera 01's own sound (the applause), faded in as the song is cut
Each song is loudness-matched (EBU R128 integrated, ffmpeg ebur128) to camera 01's own recording in the solo stretch of that song, so that the hand-over to the
heat's ambience does not jump. The stretch uses dancecap.song.render_soundtrack (ffmpeg rubberband; a constant latency is measured against the studio's features and
removed). The result is PRIVATE (work/, git-ignored): the songs are copyrighted studio recordings, research builds only (song.DISTRIBUTION).

`verify`: the onset features of the built track (heat-ambience parts zeroed) against the audio of BOTH videos, windows of 8 s (song.align_check): the median and p95 of
the lag in ms (positive = the video's onsets are later than the track's). Camera 02's lag is read through its fixed clock (reference = source + 14.8158 s).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import song as S
from .take import PACKAGE_ROOT

WORK = PACKAGE_ROOT / "work" / "carlos_aline"
AUDIO = WORK / "audio"
SONG = WORK / "song"
ALIGNMENT = PACKAGE_ROOT.parent / "prototype" / "larissa_kadu" / "alignment_carlos_aline.json"
SR = 48000
CAM02_OFFSET = 14.8158          # reference = camera 02 source time + this (NOTES_C1.md, sync.json: constant, 0.45 ms rms)
SOLO = {"kojo": (30.0, 100.0), "guetta": (130.0, 190.0)}      # reference-clock stretches where one song plays alone (loudness matching, pitch check)


# ------------------------------------------------------------------------------------------------ pure parts


def equal_power(t: np.ndarray, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """(fade_out, fade_in) gains of an equal-power crossfade running from t0 to t1 (cos / sin quarter waves, out^2 + in^2 = 1)."""
    x = np.clip((np.asarray(t, float) - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
    return np.cos(0.5 * np.pi * x), np.sin(0.5 * np.pi * x)


def gain_curves(t: np.ndarray, plan: dict) -> dict:
    """Gain of each source over the time axis `t` (seconds) for the plan {song_start: {kojo, guetta}, handover: [t0, t1], end_cut, heat_into_kojo: [a, b]}:
    heat ambience, kojo, guetta. The curves sum to 1 in power at every crossfade (heat->kojo, kojo->guetta, guetta->heat)."""
    t = np.asarray(t, float)
    h0, h1 = plan["handover"]
    a, b = plan["heat_into_kojo"]                 # heat fades out, Kojo fades in (equal power) inside [a, b]
    c, d = plan["guetta_into_heat"]               # Guetta fades out, the heat fades in inside [c, d]
    ho, ki = equal_power(t, a, b)
    heat = np.where(t < b, ho, 0.0)
    kojo = np.where(t < a, 0.0, ki)
    go, hi = equal_power(t, c, d)
    guetta = np.where(t < h0, 0.0, 1.0)
    ko, gi = equal_power(t, h0, h1)
    kojo = kojo * np.where(t < h0, 1.0, ko)
    guetta = guetta * np.where(t < h0, 0.0, gi) * np.where(t < c, 1.0, go)
    heat = np.where(t < c, heat, hi)
    kojo = np.where(t >= h1, 0.0, kojo)
    return {"heat": heat, "kojo": kojo, "guetta": guetta}


def plan_from_segments(seg: dict, songs: dict, xfade_heat_s: float = 0.3) -> dict:
    """The mixing plan from song/segments.json (governing hand-over, no-beat intervals) and the song clocks."""
    gov = seg["governing"]
    k0 = float(songs["kojo"]["song_zero_at_reference_s"])
    end = float(songs["guetta"]["audible_reference_s"][1])
    h0 = float(gov.get("kojo_until_s", gov["switch_s"] - 1.0))
    h1 = float(gov.get("guetta_from_s", gov["switch_s"]))
    if h1 <= h0:
        h0, h1 = float(gov["switch_s"]) - 1.0, float(gov["switch_s"])
    return {"song_start": {"kojo": k0, "guetta": float(songs["guetta"]["song_zero_at_reference_s"])}, "handover": [h0, h1], "end_cut": end,
            "heat_into_kojo": [k0 - 0.5 * xfade_heat_s, k0 + 0.5 * xfade_heat_s], "guetta_into_heat": [end - xfade_heat_s, end + 0.2]}


def lufs_gain_db(heat_lufs: float, song_lufs: float, limit_db: float = 18.0) -> float:
    """dB to apply to the song so that its integrated loudness equals the heat's; clamped (a silent measurement must not blow up the level)."""
    return float(np.clip(heat_lufs - song_lufs, -limit_db, limit_db))


# ------------------------------------------------------------------------------------------------ io


def songs_doc() -> dict:
    return json.loads(ALIGNMENT.read_text(encoding="utf-8"))["songs"]


def song_path(key: str) -> Path:
    return S.resolve_path(PACKAGE_ROOT.parent / "dances" / "Carlos-Aline" / songs_doc()[key]["file"])


def video(cam: str) -> Path:
    toml = (PACKAGE_ROOT / "takes" / "carlos_aline.toml")
    src = (toml.parent / S.read_toml(toml)["source_dir"]).resolve()
    names = {"01": "Carlos Aline Zouk Artist Jack Jill at Zouk SEA 2019.mp4", "02": "Zouk Sea 2019 Carlos and Aline Borges Jack Jill.mp4"}
    return src / names[cam]


def decode(path: Path, sr: int, ch: int = 2) -> np.ndarray:
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-"])
    y = np.frombuffer(raw, np.float32)
    return y.reshape(-1, ch) if ch > 1 else y


def lufs(y: np.ndarray, sr: int) -> float:
    """Integrated loudness (EBU R128, ffmpeg ebur128) of an (n, 2) float array."""
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-", "-af", "ebur128", "-f", "null", "-"],
                       input=np.ascontiguousarray(y, np.float32).tobytes(), capture_output=True)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", p.stderr.decode("utf-8", "replace"))
    return float(m[-1]) if m else float("nan")


# ------------------------------------------------------------------------------------------------ build


def build(out: Path | None = None, switch: tuple[float, float] | None = None) -> dict:
    cfg = S.SongConfig()
    out = out or AUDIO / "heat_studio.wav"
    songs = songs_doc()
    seg = json.loads((SONG / "segments.json").read_text(encoding="utf-8"))
    if switch:
        seg = json.loads(json.dumps(seg))
        seg["governing"]["kojo_until_s"], seg["governing"]["guetta_from_s"] = switch
    plan = plan_from_segments(seg, songs)
    heat = decode(video("01"), SR)
    dur = len(heat) / SR
    meta: dict = {"plan": plan, "duration_s": dur, "songs": {}}
    # pitch check: key-locked (pitch kept) or varispeed
    heat_mono24 = heat.mean(1)
    import soxr
    heat24 = soxr.resample(heat_mono24, SR, S.ALIGN_SR, quality="HQ")
    rendered = {}
    for key in ("kojo", "guetta"):
        sd = songs[key]
        slope_ref = 1.0 / float(sd["slope"])             # song.py convention: reference = slope * song + offset
        offset_ref = -float(sd["offset"]) / float(sd["slope"])
        sp = song_path(key)
        xs = S.load_mono(sp, S.ALIGN_SR)
        a0, a1 = [(r - offset_ref) / slope_ref for r in SOLO[key]]
        pc = S.pitch_check(xs, heat24, S.ALIGN_SR, slope_ref, offset_ref, (a0, a1))
        Fs = S.onset_features(xs, S.ALIGN_SR)
        y, ref_start, m = S.render_soundtrack(sp, slope_ref, offset_ref, 0.0, dur, cfg, Fs)
        rendered[key] = (y, ref_start)
        meta["songs"][key] = {"file_sha256_16": S.sha256_16(sp), "slope_ref": slope_ref, "offset_ref": offset_ref, "pitch_check": pc,
                              "stretch": {k: v for k, v in m.items() if k != "stretch_timing_after"}, "stretch_timing_after": m.get("stretch_timing_after")}
        print(f"[audio] {key}: {pc['interpretation']} (pitch {pc['semitones']:+.3f} st, varispeed would be {pc['varispeed_semitones']:+.3f}); {m['method'][:70]}", flush=True)
    t = np.arange(len(heat)) / SR
    g = gain_curves(t, plan)
    mix = heat * g["heat"][:, None]
    for key in ("kojo", "guetta"):
        y, rs = rendered[key]
        assert abs(rs) < 0.5 / SR and len(y) == len(heat), (rs, len(y), len(heat))
        a, b = [int(x * SR) for x in SOLO[key]]
        l_heat, l_song = lufs(heat[a:b], SR), lufs(y[a:b], SR)
        gdb = lufs_gain_db(l_heat, l_song)
        meta["songs"][key].update({"lufs_heat_solo": round(l_heat, 2), "lufs_song_solo": round(l_song, 2), "gain_db": round(gdb, 2), "solo_ref_s": SOLO[key]})
        print(f"[audio] {key}: heat {l_heat:.1f} LUFS, studio stretched {l_song:.1f} LUFS -> {gdb:+.1f} dB", flush=True)
        mix += y * (10 ** (gdb / 20)) * g[key][:, None]
    peak = float(np.abs(mix).max())
    gain = min(1.0, 0.98 / max(peak, 1e-9))
    meta.update({"peak_before": round(peak, 4), "gain": round(gain, 5)})
    S.write_private_wav(out, mix * gain, SR)
    mask = (g["heat"] < 0.999)          # samples that hold song material
    js = {"file": out.name, "sample_rate": SR, "channels": 2, "n_samples": int(len(mix)), "reference_start": 0.0, "reference_end": round(dur, 6),
          "time_to_audio": 0.0, "clock": "audio sample i is at reference time i / sample_rate (camera 01's video time)",
          "distribution": S.DISTRIBUTION, "copyright": S.COPYRIGHT, **meta}
    S.write_json(out.with_suffix(".json"), js)
    print(f"[audio] wrote {out} ({dur:.2f} s, gain {gain:.4f}); song material in {mask.mean() * 100:.0f} % of the samples", flush=True)
    try:
        js["verify"] = verify(out)
        S.write_json(out.with_suffix(".json"), js)
    except Exception as e:  # noqa: BLE001
        print(f"[audio] verify failed: {type(e).__name__}: {e}", flush=True)
    return js


def verify(wav: Path | None = None) -> dict:
    """Onset-feature alignment of the built track against both videos' audio (song.align_check, 8-s windows), heat-ambience parts zeroed."""
    import soxr
    wav = wav or AUDIO / "heat_studio.wav"
    y, sr = S.B.read_wav(wav)
    songs = songs_doc()
    seg = json.loads((SONG / "segments.json").read_text(encoding="utf-8"))
    plan = plan_from_segments(seg, songs)
    t = np.arange(len(y)) / sr
    g = gain_curves(t, plan)
    only = y * (1.0 - g["heat"])[:, None] ** 0.5 * (g["heat"] < 0.01)[:, None]       # keep only the stretches that are pure studio song
    cfg = S.SongConfig()
    dt = S.frame_dt(S.ALIGN_SR)
    res = {}
    Fy = S.onset_features(soxr.resample(only.mean(1), sr, S.ALIGN_SR, quality="HQ"), S.ALIGN_SR)
    for cam, shift in (("01", 0.0), ("02", CAM02_OFFSET)):
        x = decode(video(cam), S.ALIGN_SR, ch=1)
        Fr = S.onset_features(x, S.ALIGN_SR)
        r = {"whole": S.align_check(Fy, Fr, dt, -shift, cfg)}
        # per song: zero the other one
        for key, (a, b) in (("kojo", (13.4, 119.5)), ("guetta", (122.6, 193.9))):
            z = np.zeros_like(only)
            i0, i1 = int(a * sr), int(b * sr)
            z[i0:i1] = only[i0:i1]
            Fz = S.onset_features(soxr.resample(z.mean(1), sr, S.ALIGN_SR, quality="HQ"), S.ALIGN_SR)
            r[key] = S.align_check(Fz, Fr, dt, -shift, cfg)
        res[cam] = r
        print(f"[audio] vs camera {cam}: " + "; ".join(f"{k}: median {v.get('offset_error_ms_median')} ms p95 {v.get('abs_error_ms_p95')} ms ({v.get('n_inliers')}/{v.get('n_anchors')})"
                                                     for k, v in r.items()), flush=True)
    return {"convention": "ms the video's onsets are LATER than the built track's (positive = late); camera 02 through reference = source + 14.8158 s", "cams": res}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path)
    b.add_argument("--switch", nargs=2, type=float, metavar=("KOJO_UNTIL", "GUETTA_FROM"), help="override the crossfade interval (reference s)")
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.out, tuple(a.switch) if a.switch else None)
    else:
        print(json.dumps(verify(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
