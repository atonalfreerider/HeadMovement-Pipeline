"""Stage `audio_sync` (local CPU, about a second per minute of audio): every source video's affine audio clock on the
reference video's timeline, reference_time = slope * source_time + offset. A port of prototype/sync_larissa_kadu.py
(same matcher, same thresholds, same output shape) for any take.

    python -m dancecap.pipeline audio-sync <take> [--reference 06] [--force]

Writes the file named by the take TOML's `alignment` (a wizard take: work/<take>/clock/alignment.json):
    {reference: <file>, convention, sources: [{id, name, duration, slope, offset, accepted, anchors,
     max_residual_seconds, anchor_inliers}], intervals, max_count, maximum_overlap, warning, by, created}
`name` is relative to the take's source_dir (Take.load joins them). A single-view take gets the identity clock
(identity()). An existing clock file outside work/<take>/ (the prototype / studio clocks of the older takes) is never
overwritten.

Matcher (unchanged): 8 kHz mono, 250-3000 Hz band-pass; nine 10 s anchors spread over each source; fractional PHAT
cross-correlation (|X|^0.7 whitening, 300-2800 Hz) against the reference; confidence = peak / 99.9th percentile of
the sidelobes (> 2 counts); the dominant agreement cluster (+-0.15 s); >= 3 inliers get a linear fit; a clock is
accepted with >= 5 inliers and |slope - 1| < 0.005. The reference is --reference, else the longest video with audio
(the most likely to contain the others). Audio alignment does not prove frame-exact video sync: the `sync` stage
measures the video offsets later.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import paths, takes
from .util import read_json, write_json_atomic

SR = 8000
ANCHORS = 9
ANCHOR_S = 10.0
CONVENTION = "reference_time = slope * source_time + offset"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _decode(path: Path, cache: Path):
    import numpy as np
    st = path.stat()
    sig = f"{st.st_size}|{st.st_mtime_ns}|{SR}"
    meta = cache.with_suffix(".json")
    if cache.exists() and (read_json(meta, {}) or {}).get("sig") == sig:
        return np.load(cache)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le",
                          "-"], capture_output=True, check=True, creationflags=NO_WINDOW,
                         stdin=subprocess.DEVNULL).stdout
    a = np.frombuffer(raw, dtype=np.float32).copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, a)
    write_json_atomic(meta, {"sig": sig, "file": path.name})
    return a


def match(sig, ref) -> tuple[float, float]:
    """(location in the reference [s], confidence) of an anchor (the prototype's matcher)."""
    import numpy as np
    from scipy import fft
    n = fft.next_fast_len(len(sig) + len(ref) - 1)
    cross = fft.rfft(ref, n) * fft.rfft(sig, n).conj()
    freq = fft.rfftfreq(n, 1 / SR)
    cross /= np.maximum(np.abs(cross), 1e-5) ** .7
    cross[(freq < 300) | (freq > 2800)] = 0
    cc = fft.irfft(cross, n)
    cc = np.abs(cc[:len(ref) - len(sig) + 1])  # only where the whole anchor fits inside the reference
    peak = int(cc.argmax())
    side = cc.copy()
    side[max(0, peak - int(.2 * SR)):peak + int(.2 * SR)] = 0
    return peak / SR, float(cc[peak] / (np.percentile(side, 99.9) + 1e-12))


def fit_source(audio, ref) -> dict:
    """Anchors + the affine clock of one source against the reference signal."""
    import numpy as np
    duration = len(audio) / SR
    win = min(ANCHOR_S, max(2.0, duration / 3))
    anchors = []
    for start in np.linspace(2, max(2, duration - win - 2), ANCHORS):
        sig = audio[int(start * SR):int((start + win) * SR)]
        if len(sig) < int(win * SR * 0.9) or len(sig) >= len(ref):
            continue
        loc, conf = match(sig, ref)
        anchors.append({"source_start": float(start), "reference_start": loc, "offset": loc - float(start),
                        "confidence": conf})
    if not anchors:
        return {"accepted": False, "anchors": [], "reason": "too short to place an anchor in the reference"}
    good = np.array([a["confidence"] > 2 for a in anchors])
    offsets = np.array([a["offset"] for a in anchors])
    if not good.any():
        return {"accepted": False, "anchors": anchors, "reason": "no confident anchor"}
    best = max(np.flatnonzero(good), key=lambda i: int(np.sum(good & (np.abs(offsets - offsets[i]) < .15))))
    keep = good & (np.abs(offsets - offsets[best]) < .15)
    x = np.array([a["source_start"] for a in anchors])
    y = np.array([a["reference_start"] for a in anchors])
    slope, offset = np.polyfit(x[keep], y[keep], 1) if keep.sum() >= 3 else (1.0, float(np.median(offsets[keep])))
    resid = y - (slope * x + offset)
    for a, k, r in zip(anchors, keep, resid):
        a.update(inlier=bool(k), residual_seconds=float(r))
    return {"slope": float(slope), "offset": float(offset), "anchors": anchors,
            "accepted": bool(keep.sum() >= 5 and abs(slope - 1) < .005),
            "max_residual_seconds": float(np.max(np.abs(resid[keep]))), "anchor_inliers": int(keep.sum())}


def overlap(entries: list[dict]) -> dict:
    """intervals / max_count / maximum_overlap of the accepted sources (the prototype's coverage table)."""
    events = []
    for r in entries:
        if r.get("accepted"):
            events += [r["offset"], r["offset"] + r["slope"] * r["duration"]]
    bounds = sorted(set(events))
    intervals = []
    for a, b in zip(bounds, bounds[1:]):
        mid = (a + b) / 2
        act = [r["id"] for r in entries if r.get("accepted") and r["offset"] <= mid < r["offset"] + r["slope"] * r["duration"]]
        intervals.append({"start": a, "end": b, "cameras": act, "count": len(act)})
    mx = max((x["count"] for x in intervals), default=0)
    return {"intervals": intervals, "max_count": mx, "maximum_overlap": [x for x in intervals if x["count"] == mx]}


def identity(take: str, cam: str, name: str, duration: float | None) -> dict:
    """The clock of a single-view take: its only video IS the reference (slope 1, offset 0)."""
    d = float(duration or 0.0)
    row = {"id": cam, "name": name, "duration": d, "slope": 1.0, "offset": 0.0, "anchors": [], "accepted": True,
           "max_residual_seconds": 0.0, "anchor_inliers": 0}
    out = {"reference": name, "convention": CONVENTION, "sources": [row], "by": "dancecap.pipeline.audio_sync (identity)",
           "created": time.time(), "warning": "single view: the video is its own reference clock"}
    out.update(overlap([row]) if d > 0 else {"intervals": [], "max_count": 1, "maximum_overlap": []})
    return out


def _target(take: str, cfg: dict, force: bool) -> Path:
    p = takes.alignment_path(cfg)
    if p is None:
        raise SystemExit(f"[audio_sync] takes/{take}.toml has no `alignment` path")
    inside = str(p.resolve()).lower().startswith(str(paths.take_work(take).resolve()).lower() + os.sep)
    if p.exists() and not inside and not force:
        raise SystemExit(f"[audio_sync] {paths.rel_display(p)} is outside work/{take}/ (a prototype or studio clock): "
                         "not overwritten (--force to replace it)")
    return p


def run(take: str, reference: str | None = None, force: bool = False, log=print) -> dict:
    cfg, _ = takes.load_cfg(take)
    out_p = _target(take, cfg, force)
    vids = [v for v in takes.videos(take, cfg) if not v["excluded"] and v.get("path")]
    if not vids:
        raise SystemExit(f"[audio_sync] {take}: no registered video")
    probe = (read_json(paths.take_work(take) / "probe" / "probe.json", {}) or {}).get("cameras", {})
    if len(vids) == 1:
        v = vids[0]
        doc = identity(take, v["cam"], v["file"], (probe.get(v["cam"]) or {}).get("duration_s"))
        write_json_atomic(out_p, doc)
        log(f"[audio_sync] {take}: single view, identity clock -> {paths.rel_display(out_p)}")
        return doc
    from scipy import signal
    filt = signal.butter(4, [250, 3000], btype="bandpass", fs=SR, output="sos")
    cache = paths.take_work(take) / "clock" / "audio"
    audios, entries = {}, []
    for i, v in enumerate(vids):
        log(f"[audio_sync] decode {v['cam']} ({i + 1}/{len(vids)})")
        try:
            a = _decode(Path(v["path"]), cache / f"audio_{v['cam']}.npy")
        except (OSError, subprocess.CalledProcessError) as e:
            entries.append({"id": v["cam"], "name": v["file"], "duration": None, "accepted": False, "anchors": [],
                            "reason": f"no audio ({str(e)[:120]})"})
            continue
        if len(a) < SR:
            entries.append({"id": v["cam"], "name": v["file"], "duration": len(a) / SR, "accepted": False,
                            "anchors": [], "reason": "no audio track"})
            continue
        audios[v["cam"]] = signal.sosfiltfilt(filt, a).astype("float32")
        dur = (probe.get(v["cam"]) or {}).get("duration_s") or len(a) / SR
        entries.append({"id": v["cam"], "name": v["file"], "duration": float(dur)})
    if not audios:
        raise SystemExit(f"[audio_sync] {take}: no video has audio")
    ref = reference if reference in audios else max(audios, key=lambda c: len(audios[c]))
    for k, row in enumerate(entries):
        if row["id"] not in audios:
            continue
        if row["id"] == ref:
            row.update(slope=1.0, offset=0.0, anchors=[], accepted=True, max_residual_seconds=0.0, anchor_inliers=0)
            continue
        row.update(fit_source(audios[row["id"]], audios[ref]))
        log(f"[audio_sync] {k + 1}/{len(entries)} {row['id']}: offset {row.get('offset', float('nan')):+.3f} s, "
            f"slope {row.get('slope', float('nan')):.6f}, inliers {row.get('anchor_inliers', 0)}, "
            f"{'accepted' if row['accepted'] else 'NOT accepted: ' + str(row.get('reason') or 'too few inliers')}")
    ref_name = next(r["name"] for r in entries if r["id"] == ref)
    doc = {"reference": ref_name, "convention": CONVENTION, "sources": entries, **overlap(entries),
           "warning": "Audio propagation and A/V offset still require visual verification (the sync stage).",
           "by": "dancecap.pipeline.audio_sync (port of prototype/sync_larissa_kadu.py)", "created": time.time()}
    write_json_atomic(out_p, doc)
    mo = doc["maximum_overlap"][0] if doc["maximum_overlap"] else None
    log(f"[audio_sync] reference {ref} ({ref_name}); {sum(1 for r in entries if r.get('accepted'))}/{len(entries)} "
        f"clocks accepted -> {paths.rel_display(out_p)}"
        + (f"; maximum overlap {mo['start']:.2f}-{mo['end']:.2f} s ({mo['count']} cameras): a window candidate"
           if mo else ""))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline audio-sync")
    ap.add_argument("take")
    ap.add_argument("--reference", help="camera id of the reference clock (default: the longest video with audio)")
    ap.add_argument("--force", action="store_true", help="replace a clock file outside work/<take>/")
    a = ap.parse_args(argv)
    doc = run(a.take, a.reference, a.force)
    n_ok = sum(1 for r in doc["sources"] if r.get("accepted"))
    if n_ok < min(2, len(doc["sources"])):
        print("[audio_sync] fewer than two clocks accepted: the take cannot be analysed as multi-view (check the "
              "videos' audio, or exclude cameras)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
