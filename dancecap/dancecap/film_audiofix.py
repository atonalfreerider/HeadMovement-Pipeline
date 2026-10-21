"""Replace a film recording's audio track with the film soundtrack mix it was recorded under (the recorder's AAC burst).

The Unity Recorder's own AAC track of the recap films has a 13-16 ms burst at film 56.876 s (decoded peak ~199 x full scale,
NOTES_R10 for the avatar films, `film/splats_review` for the splat films of 2026-10-20); the recorder's own checks do not see it.
The soundtrack the recorder PLAYS is the film mix `<film dir>/audio/<direction>_mix.wav` (its correlation with the recorded track
is 1.000 at 0.0 ms), so the repair is exact: keep the video stream untouched and encode the audio again from the mix.

    python -m dancecap.film_audiofix <recording.mp4> --mix <mix.wav> [--start <film s>] [--report <recording>.report.json]
                                     [--ffmpeg <ffmpeg.exe>] [--ffprobe <ffprobe.exe>]

1. `<stem>.mp4` is renamed to `<stem>_rawaudio.mp4` (the recorder's file, kept alongside; when the raw file already exists, e.g.
   on a rerun, it is the input and is never overwritten);
2. `<stem>.mp4` is written again: video stream copied, audio = the mix from film time `--start` for the video's length, AAC-LC
   192 kb/s 48 kHz stereo, padded with silence to the video's length (the NOTES_R10 method; the mix is 56 ms shorter than the
   full film's last frame);
3. the new file is verified against the mix and the raw file is scanned (`verify`): the video stream is bit-identical to the raw
   one (stream md5), the audio lasts as long as the video, it is in sync with the mix in up to 30 windows (lag within 1 ms,
   correlation >= 0.99), its integrated loudness equals the mix's (ebur128, within 0.5 LU), no sample is over full scale, the
   true peak stays below 0 dBTP and no 10 ms block departs from the mix by more than the AAC coding noise (a burst); the raw
   file's bursts are listed (where the recorder's track went wrong);
4. `<stem>.audiofix.json` is written; the recorder report (`--report`) gets an "audio_fix" entry and the .report.md a section.

On a failed verification the raw file is put back as `<stem>.mp4`, the attempt is kept as `<stem>_audiofix_failed.mp4` and the
exit code is 1. The head-movement film recorder runs this after its own checks whenever the splat dancers were on (hm_film
`--audiofix auto`, the default; `on` for every director recording, `off` never).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SR = 48000
AAC_BITRATE = "192k"
RAW_SUFFIX = "_rawaudio"
FAILED_SUFFIX = "_audiofix_failed"
SYNC_WINDOWS = 30
SYNC_WIN_S = 2.0
SYNC_MAX_LAG_S = 0.02
SYNC_TOL_MS = 1.0
SYNC_MIN_R = 0.99
LOUDNESS_TOL_LU = 0.5
BLOCK_S = 0.01
# a burst = a 10 ms block where the track departs from the mix by more than this AND by more than the mix's own peak there. AAC
# 192 kb/s coding noise stays below it (the full recap film: largest departure 0.099 at a transient); the recorder's burst is ~200x
# full scale against a mix peak of 0.085
BURST_RESIDUAL = 0.25

_LUFS = re.compile(r"I:\s+(-?[\d.]+|-inf)\s+LUFS")
_TP = re.compile(r"Peak:\s+(-?[\d.]+|-inf)\s+dBFS")


class AudioFixError(RuntimeError):
    """a problem the message explains (missing file, ffmpeg failure)"""


def _run(exe: str, args: list[str], *, binary: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run([exe, *args], capture_output=True, text=not binary,
                       **({} if binary else {"encoding": "utf-8", "errors": "replace"}))
    return r


def _ff(ffmpeg: str, *args: str, binary: bool = False) -> subprocess.CompletedProcess:
    r = _run(ffmpeg, ["-hide_banner", "-nostdin", "-v", "error", *args], binary=binary)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace") if binary else r.stderr
        raise AudioFixError(f"ffmpeg {' '.join(args[:8])} ... failed: {err[-600:]}")
    return r


def probe(ffprobe: str, path: Path) -> dict:
    r = _run(ffprobe, ["-v", "error", "-show_entries",
                       "stream=index,codec_type,codec_name,profile,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,"
                       "sample_rate,channels,bit_rate:format=duration", "-of", "json", str(path)])
    if r.returncode != 0:
        raise AudioFixError(f"ffprobe {path.name} failed: {r.stderr[-400:]}")
    return json.loads(r.stdout)


def _stream(p: dict, kind: str) -> dict | None:
    return next((s for s in p.get("streams", []) if s.get("codec_type") == kind), None)


def _ratio(s: str | None) -> float:
    if not s:
        return float("nan")
    a, _, b = s.partition("/")
    return float(a) / float(b or 1)


def video_seconds(p: dict) -> tuple[float, int, float]:
    """(duration, frames, fps) of the video stream: frames / fps when the frame count is known (the exact film length)"""
    v = _stream(p, "video")
    if v is None:
        raise AudioFixError("no video stream")
    fps = _ratio(v.get("avg_frame_rate")) or _ratio(v.get("r_frame_rate"))
    n = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else -1
    dur = n / fps if n > 0 and fps > 0 else float(v.get("duration") or p["format"]["duration"])
    return dur, n, fps


def video_md5(ffmpeg: str, path: Path) -> str:
    """md5 of the video stream's packets (stream copy): equal = the picture is untouched"""
    r = _ff(ffmpeg, "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "md5", "-")
    return r.stdout.strip().removeprefix("MD5=")


def decode(ffmpeg: str, inputs: list[str]) -> np.ndarray:
    """(n, 2) float32 at 48 kHz"""
    r = _ff(ffmpeg, *inputs, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-", binary=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2)


def loudness(ffmpeg: str, inputs: list[str]) -> dict:
    r = _run(ffmpeg, ["-hide_banner", "-nostdin", *inputs, "-vn", "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"])
    txt = r.stderr
    tail = txt[txt.rfind("Summary:"):] if "Summary:" in txt else txt
    i, tp = _LUFS.search(tail), _TP.search(tail)
    if not i:
        raise AudioFixError(f"no ebur128 summary: {txt[-300:]}")
    f = lambda m: float(m.group(1)) if m and m.group(1) != "-inf" else float("-inf")  # noqa: E731
    return {"lufs": f(i), "true_peak_dbtp": f(tp)}


def _db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 0 else -200.0


def sync_windows(rec: np.ndarray, ref: np.ndarray, *, n_windows: int = SYNC_WINDOWS, win_s: float = SYNC_WIN_S,
                 max_lag_s: float = SYNC_MAX_LAG_S) -> list[dict]:
    """lag (ms, + = the recording is LATE) and normalised correlation of rec vs ref in up to n windows spread over the clip
    (windows where the reference is silent are skipped)"""
    a, b = rec.mean(axis=1).astype(np.float64), ref.mean(axis=1).astype(np.float64)
    n = min(len(a), len(b))
    lag = int(max_lag_s * SR)
    # a short excerpt gets shorter windows (at least 3 of them, each >= 0.5 s)
    win = int(min(win_s * SR, max(0.5 * SR, (n - 2 * lag) / 3)))
    if n < win + 2 * lag:
        win = max(0, n - 2 * lag - 1)
    if win < SR // 4:
        return []
    count = max(1, min(n_windows, int((n - 2 * lag) / win)))
    out = []
    for k in range(count):
        s0 = lag + int(round((n - 2 * lag - win) * (k + 0.5) / count))
        r0 = b[s0:s0 + win]
        er = float(r0 @ r0)
        if er < win * 1e-7:  # about -70 dBFS: silence in the mix here
            continue
        seg = a[s0 - lag:s0 + win + lag]
        m = len(seg) + win
        nfft = 1 << (m - 1).bit_length()
        cc = np.fft.irfft(np.fft.rfft(seg, nfft) * np.conj(np.fft.rfft(r0, nfft)), nfft)[:2 * lag + 1]
        # energy of every shifted window of seg
        c2 = np.concatenate([[0.0], np.cumsum(seg * seg)])
        es = c2[win:win + 2 * lag + 1] - c2[:2 * lag + 1]
        r = cc / np.sqrt(es * er + 1e-12)
        i = int(np.argmax(r))
        out.append({"t": round(s0 / SR, 2), "lag_ms": round((i - lag) * 1000.0 / SR, 3), "r": round(float(r[i]), 5)})
    return out


def bursts(rec: np.ndarray, ref: np.ndarray, *, threshold: float = BURST_RESIDUAL) -> tuple[float, list[dict]]:
    """largest |rec - ref| and the 10 ms blocks where it exceeds the threshold (merged runs): a burst the mix does not have"""
    n = min(len(rec), len(ref))
    d = np.abs(rec[:n].astype(np.float64) - ref[:n]).max(axis=1)
    blk = int(BLOCK_S * SR)
    nb = n // blk
    if nb == 0:
        return float(d.max()) if n else 0.0, []
    bd = d[:nb * blk].reshape(nb, blk).max(axis=1)
    bm = np.abs(ref[:nb * blk]).max(axis=1).reshape(nb, blk).max(axis=1)
    hits = np.nonzero((bd > threshold) & (bd > bm))[0]
    out: list[dict] = []
    for i in hits:
        if out and i - out[-1]["_last"] <= 1:
            o = out[-1]
            o["_last"] = int(i)
            o["t1"] = round((i + 1) * BLOCK_S, 3)
            o["residual_peak"] = round(max(o["residual_peak"], float(bd[i])), 3)
            o["mix_peak"] = round(max(o["mix_peak"], float(bm[i])), 3)
        else:
            out.append({"_last": int(i), "t0": round(i * BLOCK_S, 3), "t1": round((i + 1) * BLOCK_S, 3),
                        "residual_peak": round(float(bd[i]), 3), "mix_peak": round(float(bm[i]), 3)})
    for o in out:
        o.pop("_last")
    return float(d.max()), out


def raw_path(video: Path) -> Path:
    return video.with_name(video.stem + RAW_SUFFIX + video.suffix)


def fix(video: Path, mix: Path, *, start: float = 0.0, report: Path | None = None, ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe", log=print) -> dict:
    """replace the audio of `video` (see the module doc); returns the result written to <stem>.audiofix.json"""
    t_begin = time.time()
    video = Path(video)
    mix = Path(mix)
    if not mix.exists():
        raise AudioFixError(f"mix missing: {mix}")
    raw = raw_path(video)
    if raw.exists():
        log(f"[audiofix] raw recording already kept: {raw.name} (input)")
    else:
        if not video.exists():
            raise AudioFixError(f"recording missing: {video}")
        video.rename(raw)
        log(f"[audiofix] {video.name} -> {raw.name} (the recorder's file, kept)")

    p_raw = probe(ffprobe, raw)
    dur, frames, fps = video_seconds(p_raw)
    tmp = video.with_name(video.stem + "_audiofix_tmp" + video.suffix)
    log(f"[audiofix] video {frames} frames @ {fps:g} fps = {dur:.3f} s; audio = {mix.name} from {start:.3f} s, AAC {AAC_BITRATE}")
    _ff(ffmpeg, "-y", "-i", str(raw), "-ss", f"{start:.6f}", "-t", f"{dur:.6f}", "-i", str(mix),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af", f"apad=whole_dur={dur:.6f}",
        "-c:a", "aac", "-b:a", AAC_BITRATE, "-ar", str(SR), "-ac", "2", "-movflags", "+faststart", str(tmp))

    res = verify(tmp, raw, mix, start=start, ffmpeg=ffmpeg, ffprobe=ffprobe)
    res.update({"kind": "film_audio_fix", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "video": str(video), "raw": str(raw),
                "mix": str(mix), "start_s": round(start, 6), "duration_s": round(dur, 6), "frames": frames, "fps": fps,
                "method": f"video stream copied; audio = AAC-LC {AAC_BITRATE} 48 kHz stereo of the mix from film {start:.3f} s, "
                          f"padded to the video length (NOTES_R10)"})
    if res["pass"]:
        tmp.replace(video)
        res["result"] = "audio replaced"
    else:
        failed = video.with_name(video.stem + FAILED_SUFFIX + video.suffix)
        tmp.replace(failed)
        if not video.exists():
            raw.rename(video)  # the recorder's file back under its own name (a rerun keeps an earlier fixed file)
            res["raw"] = str(video)
        res["result"] = f"verification FAILED: the recorder's file is at {Path(res['raw']).name}; the attempt is {failed.name}"
    res["seconds"] = round(time.time() - t_begin, 1)
    out = video.with_name(video.stem + ".audiofix.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if report is not None:
        add_to_report(Path(report), res)
    log(f"[audiofix] {'PASS' if res['pass'] else 'FAIL'}: {summary(res)}")
    return res


def verify(new: Path, raw: Path, mix: Path, *, start: float, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    """the checks of the module doc, on `new` (the fixed file) against the mix; the raw file's bursts for the record"""
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    p_new, p_raw = probe(ffprobe, new), probe(ffprobe, raw)
    dur, frames, fps = video_seconds(p_raw)
    dur_n, frames_n, _ = video_seconds(p_new)
    md5_raw, md5_new = video_md5(ffmpeg, raw), video_md5(ffmpeg, new)
    add("video untouched", md5_raw == md5_new and frames_n == frames,
        f"video stream md5 {'identical' if md5_raw == md5_new else 'DIFFERENT'} ({md5_new[:12]}), {frames_n} frames (raw {frames})")
    a = _stream(p_new, "audio") or {}
    adur = float(a.get("duration") or "nan")
    add("audio stream", a.get("codec_name") == "aac" and int(a.get("channels") or 0) == 2 and str(a.get("sample_rate")) == str(SR),
        f"{a.get('codec_name')} {a.get('sample_rate')} Hz x{a.get('channels')}, {int(a.get('bit_rate') or 0) / 1000:.0f} kb/s")
    add("audio duration", abs(adur - dur) <= 0.03, f"audio {adur:.3f} s, video {dur:.3f} s")

    rec = decode(ffmpeg, ["-i", str(new)])
    rawa = decode(ffmpeg, ["-i", str(raw)])
    ref = decode(ffmpeg, ["-ss", f"{start:.6f}", "-t", f"{dur:.6f}", "-i", str(mix)])
    if len(ref) < len(rec):
        ref = np.concatenate([ref, np.zeros((len(rec) - len(ref), 2), np.float32)])

    wins = sync_windows(rec, ref)
    lags = [w["lag_ms"] for w in wins]
    rs = [w["r"] for w in wins]
    sync = {"windows": wins, "n": len(wins), "lag_ms_min": min(lags) if lags else None, "lag_ms_max": max(lags) if lags else None,
            "lag_ms_median": float(np.median(lags)) if lags else None, "r_min": min(rs) if rs else None}
    add("A/V sync vs. mix", bool(wins) and max(abs(x) for x in lags) <= SYNC_TOL_MS and min(rs) >= SYNC_MIN_R,
        f"{len(wins)} windows: lag {sync['lag_ms_min']}..{sync['lag_ms_max']} ms, r >= {sync['r_min']}" if wins else "no usable window")

    l_new = loudness(ffmpeg, ["-i", str(new)])
    l_mix = loudness(ffmpeg, ["-ss", f"{start:.6f}", "-t", f"{dur:.6f}", "-i", str(mix)])
    l_raw = loudness(ffmpeg, ["-i", str(raw)])
    add("loudness", abs(l_new["lufs"] - l_mix["lufs"]) <= LOUDNESS_TOL_LU,
        f"{l_new['lufs']:.1f} LUFS (mix {l_mix['lufs']:.1f}, recorder's track {l_raw['lufs']:.1f})")
    peak = float(np.abs(rec).max()) if rec.size else 0.0
    over = int((np.abs(rec) > 1.0).sum())
    add("peak", over == 0 and l_new["true_peak_dbtp"] < 0.0,
        f"sample peak {_db(peak):.1f} dBFS, true peak {l_new['true_peak_dbtp']:.1f} dBTP (mix {l_mix['true_peak_dbtp']:.1f}), "
        f"{over} samples over full scale")
    resid, b_new = bursts(rec, ref)
    add("no burst", not b_new, f"largest departure from the mix {resid:.4f} ({_db(resid):.1f} dBFS); "
                                f"{len(b_new)} burst block(s) (over {BURST_RESIDUAL} and over the mix's own peak)")
    resid_raw, b_raw = bursts(rawa, ref)
    for b in b_raw:
        b["film_t0"] = round(start + b["t0"], 3)
        b["film_t1"] = round(start + b["t1"], 3)
    return {
        "pass": all(c["pass"] for c in checks), "checks": checks, "sync": sync,
        "loudness": {"new": l_new, "mix": l_mix, "raw": l_raw},
        "peak": {"new_sample_peak_dbfs": round(_db(peak), 2), "new_over_full_scale": over,
                 "raw_sample_peak": round(float(np.abs(rawa).max()) if rawa.size else 0.0, 3),
                 "raw_over_full_scale": int((np.abs(rawa) > 1.0).sum())},
        "residual": {"new_max": round(resid, 5), "new_bursts": b_new, "raw_max": round(resid_raw, 3), "raw_bursts": b_raw},
        "video_md5": {"raw": md5_raw, "new": md5_new},
    }


def summary(res: dict) -> str:
    rb = res.get("residual", {}).get("raw_bursts") or []
    raw = ("recorder's track: " + ", ".join(f"burst {b['film_t0']:.3f}-{b['film_t1']:.3f} s (peak {b['residual_peak']:g})" for b in rb)
           if rb else "recorder's track: no burst")
    lo = res.get("loudness", {}).get("new", {})
    s = res.get("sync", {})
    return (f"{res.get('result', '')}; {lo.get('lufs', float('nan')):.1f} LUFS, true peak {lo.get('true_peak_dbtp', float('nan')):.1f} dBTP, "
            f"sync {s.get('lag_ms_min')}..{s.get('lag_ms_max')} ms in {s.get('n')} windows (r >= {s.get('r_min')}); {raw}")


def add_to_report(report: Path, res: dict) -> None:
    """the recorder's report: an "audio_fix" entry in the .json, a section in the .md (both describe the recorder's file otherwise)"""
    if report.exists():
        try:
            o = json.loads(report.read_text(encoding="utf-8"))
            o["audio_fix"] = {k: res[k] for k in ("pass", "result", "raw", "method", "checks", "sync", "loudness", "peak", "residual")
                              if k in res}
            report.write_text(json.dumps(o, indent=2), encoding="utf-8")
        except (OSError, ValueError) as e:
            print(f"[audiofix] could not update {report.name}: {e}", file=sys.stderr)
    md = report.with_suffix("").with_suffix(".report.md") if report.name.endswith(".report.json") else report.with_suffix(".md")
    if md.exists():
        lines = ["", "## Audio track replaced (dancecap.film_audiofix)", "",
                 f"**{'PASS' if res['pass'] else 'FAIL'}** - {res.get('result')}. {res.get('method')}. The checks above describe the "
                 f"recorder's own file, kept as `{Path(res['raw']).name}`.", "", "| Check | Result | Detail |", "|---|---|---|"]
        lines += [f"| {c['name']} | {'pass' if c['pass'] else '**FAIL**'} | {c['detail'].replace('|', '/')} |" for c in res["checks"]]
        rb = res["residual"]["raw_bursts"]
        lines += ["", "Recorder's track: " + ("; ".join(f"burst at film {b['film_t0']:.3f}-{b['film_t1']:.3f} s, peak {b['residual_peak']:g} "
                                                       f"x full scale (mix {b['mix_peak']:g})" for b in rb) if rb else "no burst found") + "."]
        text = md.read_text(encoding="utf-8")
        cut = text.find("\n## Audio track replaced")
        md.write_text((text if cut < 0 else text[:cut]).rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.film_audiofix", description=__doc__.split("\n\n")[0])
    ap.add_argument("video", type=Path, help="the recording (<stem>.mp4; <stem>_rawaudio.mp4 is used as input when it exists)")
    ap.add_argument("--mix", type=Path, required=True, help="the film soundtrack the recorder played (<film dir>/audio/<name>_mix.wav)")
    ap.add_argument("--start", type=float, default=0.0, help="film time of the recording's first frame (hm_film --start)")
    ap.add_argument("--report", type=Path, default=None, help="the recorder's <stem>.report.json to annotate")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    a = ap.parse_args(argv)
    try:
        res = fix(a.video, a.mix, start=a.start, report=a.report, ffmpeg=a.ffmpeg, ffprobe=a.ffprobe)
    except AudioFixError as e:
        print(f"[audiofix] FAILED: {e}", file=sys.stderr)
        return 2
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
