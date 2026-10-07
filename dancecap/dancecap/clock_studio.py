"""Studio-derived phone clocks: alignment_studio.json, its validation, the all-camera window and keyframe reuse.

dancecap.song aligned the studio track to every phone's OWN audio (song.json alignment.cross_check_detail.<cam>):
composing that studio -> phone fit with the studio -> reference fit gives each phone's audio clock
(reference_time = slope * source_time + offset, reference = cam06's audio clock, as in alignment.json), where
alignment.json's phone-to-phone clocks scatter 10-60 ms per anchor and are off by -36..+29 ms over 30-68.5 s (01/05/07
also drift 10-13 ms/min). This module writes those clocks in alignment.json's schema, checks them, and derives a take
from them.

Accuracy: the new clocks put every phone's AUDIO within about +-1-2 ms of cam06's audio, in the onset-envelope sense
(an independent envelope check agrees to 1-2 ms). The 0.1-0.7 ms studio fit residuals and the song cross-check are
precision figures from the same onset-feature machinery, not capture-time accuracy: raw-waveform correlation has
competing peaks (reverb, two speakers) and puts some phones ~10 ms elsewhere (cam03).

Caveat (both clock sets): these are AUDIO clocks. Phone audio-video latency and sound travel from the speakers
(~2.9 ms per metre) are not in them. larissa_kadu's video-geometry offsets (sync.py, 56-62 s) moved by the clock change
predict per-phone VIDEO offsets of about -45..+92 ms under the new clocks (cam04 worst, spread ~40 ms SD, as before):
validate (D). Video-frame sync is NOT improved by these clocks; sync.py must measure it on the new take before any
MAMMA fit (mamma prepare refuses a take with [sync] required = true until its own sync/offsets.json exists).

    python -m dancecap.clock_studio build --base larissa_kadu --out ../prototype/larissa_kadu/alignment_studio.json
    python -m dancecap.clock_studio validate --base larissa_kadu --alignment ../prototype/larissa_kadu/alignment_studio.json --review-take larissa_kadu_38 [--write-back]
    python -m dancecap.clock_studio window --alignment ../prototype/larissa_kadu/alignment_studio.json --exclude 08 --within 30 68.5
    python -m dancecap.clock_studio window --take larissa_kadu_38      # coverage of the take's window (+ sync offsets)
    python -m dancecap.clock_studio reuse-keyframes --from larissa_kadu --to larissa_kadu_38

build      sources[] in alignment.json's schema (cam06 identity; the other phones from song.json's studio_derived_clock;
           cameras the base take excludes are aligned to the studio track here with song.camera_check), intervals +
           maximum_overlap from the new clocks, provenance, the studio fit residuals and per-camera deltas vs
           alignment.json. Refuses to write the base take's alignment file or any file clock_studio did not build.
validate   (A) studio track: every phone's audio, mapped through either clock set, against the studio track on the
           reference clock (4 s windows every 1 s over the span): residual distribution + drift per clock set.
           (B) phones only, no studio track: all phone pairs (8 s windows every 1.5 s) mapped through either clock
           set; the pairwise discrepancies are solved for each phone's clock error (cam06 = 0).
           (C) envelope spot-check: each phone's onset envelope around strong studio onsets, both clock sets.
           (D) expected VIDEO offsets under the new clocks: the base take's sync.py deltas minus the clock change.
           -> work/<review-take>/clock_studio/validation.json (per window), work/<review-take>/review/
              clock_studio_validation.png; --write-back also stores the summary in the alignment file ("validation";
              only into a file clock_studio built, never the base take's alignment file).
window     the largest span inside [within] where every used camera has VIDEO frames (first/last packet PTS via
           ffprobe, mapped through the clocks), rounded inward to 1/30 s, + each camera's margin. --take uses the
           take's alignment/exclude/window and, once work/<take>/sync/offsets.json exists, the SYNC-CORRECTED frame
           times (t + delta (+ rate)): it says whether the take's window still lies inside every camera's coverage
           (mamma prepare otherwise trims the grid) -> work/<take>/clock_studio/window.json. Also: 'decoded', the
           grid mamma prepare builds from the DECODED frames (ingest decodes only frames inside the window, so the
           grid is a frame or two shorter than the window and starts off the 1/30 s lattice), and 'time_base'
           (beats/audio on cam06's audio clock vs poses on sync.py's median-gauge video time).
reuse-keyframes  the cameras code reads work/<take>/keyframes/<cam>/keyframes.json (+ people_masks_kf/) and has no
           option to point at another take, so the base take's static-room keyframes are COPIED (never linked: a
           re-ingest would write through a hard link into the base take) and their t re-mapped through the new
           clocks (pts are source timestamps; file names keep the base take's millisecond labels). Keep them with
           'ingest <take> --window-only' and 'detect <take> --no-keyframes' (plain detect re-detects keyframes).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import song as S
from .take import PACKAGE_ROOT, Source, Take

VERSION = 1
FPS_GRID = 30.0


# ------------------------------------------------------------------------------------------------ helpers
def _alignment_path(take: Take) -> Path:
    return (PACKAGE_ROOT / "takes" / take.cfg["alignment"]).resolve()


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PACKAGE_ROOT / p).resolve()


def _rel_ws(p: Path) -> str:
    return S.rel_to_workspace(Path(p).resolve())


def _clock(src: dict) -> tuple[float, float]:
    return float(src["slope"]), float(src["offset"])


def _check_target(path: Path, base: Take, what: str) -> None:
    """clock_studio only writes its own alignment files: never the base take's alignment file (another workflow's
    take uses it) and never an existing file that clock_studio build did not write."""
    path = Path(path).resolve()
    if path == _alignment_path(base):
        raise SystemExit(f"[clock] {what}: refusing to write {path}: it is take {base.name}'s alignment file")
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = None
        mod = str(((doc or {}).get("provenance") or {}).get("module", "")) if isinstance(doc, dict) else ""
        if not mod.startswith("dancecap.clock_studio"):
            raise SystemExit(f"[clock] {what}: refusing to overwrite {path}: not written by dancecap.clock_studio")


def song_to_source(s: float, o: float, clock: dict) -> tuple[float, float]:
    """(a, b): phone source time = a * song time + b, for the song -> reference line (s, o) and a phone clock
    reference = slope * source + offset (song.camera_check's prediction)."""
    sl, of = _clock(clock)
    return s / sl, (o - of) / sl


def compose_clock(s: float, o: float, a: float, b: float) -> tuple[float, float]:
    """Phone clock (slope, offset) from a measured studio -> phone fit source = a * song + b and the studio ->
    reference line reference = s * song + o (song.camera_check's studio_derived_clock): inverse of song_to_source."""
    return s / a, o - s * b / a


def clock_change_ms(old: dict, new: dict, t_ref_old: float) -> float:
    """new - old reference time (ms) of the source instant that the OLD clock puts at t_ref_old."""
    so, oo = _clock(old)
    sn, on = _clock(new)
    u = (t_ref_old - oo) / so
    return 1000.0 * ((sn * u + on) - t_ref_old)


def video_span(path: Path) -> tuple[float, float, int]:
    """(first, last) video packet PTS in source seconds + packet count (ffprobe, no decoding)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    pts = np.array([float(x.strip().rstrip(",")) for x in out.split() if x.strip().rstrip(",") not in ("", "N/A")])
    return float(pts.min()), float(pts.max()), int(len(pts))


def intervals_from(sources: list[dict]) -> tuple[list[dict], int, list[dict]]:
    """alignment.json's intervals / max_count / maximum_overlap from the clocks + durations (same rule as the
    original: [offset, slope * duration + offset] per accepted source)."""
    acc = [s for s in sources if s.get("accepted", True)]
    span = {s["id"]: (s["offset"], s["slope"] * s["duration"] + s["offset"]) for s in acc}
    num = {s["id"]: bool(s.get("numbered", True)) for s in acc}
    edges = sorted({v for ab in span.values() for v in ab})
    ivs = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = 0.5 * (a + b)
        cams = sorted(c for c, (lo, hi) in span.items() if lo <= m < hi)
        if cams:
            ivs.append({"start": a, "end": b, "cameras": cams, "count": len(cams),
                        "numbered_count": sum(num[c] for c in cams)})
    mc = max(iv["count"] for iv in ivs)
    return ivs, mc, [iv for iv in ivs if iv["count"] == mc]


# ------------------------------------------------------------------------------------------------ build
def build(base: str, out: Path) -> dict:
    take = Take.load(base)
    _check_target(out, take, "build")
    al_path = _alignment_path(take)
    al = json.loads(al_path.read_text(encoding="utf-8"))
    sj = S.read_json(take.work / "song" / "song.json")
    det = S.read_json(take.work / "song" / "align_detail.json")
    s, o = float(sj["alignment"]["slope"]), float(sj["alignment"]["offset"])
    ref_name = al["reference"]
    cc = sj["alignment"]["cross_check_detail"]
    anchors_all = det["cross_check_anchors"]
    root = take.sources[0].path.parent
    extra = {}  # cameras the base take excluded: aligned to the studio track here
    missing = [r["id"] for r in al["sources"] if r["name"] != ref_name and r["id"] not in cc]
    if missing:
        cfg = S.SongConfig()
        info = S.song_info(base)
        xs = S.load_mono(info.path, S.ALIGN_SR)
        Fs = S.onset_features(xs, S.ALIGN_SR)
        dt = S.frame_dt(S.ALIGN_SR)
        for cid in missing:
            r = next(x for x in al["sources"] if x["id"] == cid)
            src = Source(cid, root / r["name"], float(r["slope"]), float(r["offset"]))
            xc = S.load_mono(src.path, S.ALIGN_SR)
            chk = S.camera_check(Fs, dt, len(xs) / S.ALIGN_SR, src, S.onset_features(xc, S.ALIGN_SR),
                                 len(xc) / S.ALIGN_SR, s, o, cfg)
            print(f"[clock] cam{cid} (not in {base}'s song cross-checks): studio fit "
                  f"{chk.get('n_inliers')}/{chk.get('n_anchors')} anchors, residual {chk.get('fit_residual_ms_median')} "
                  f"ms, alignment.json clock error {chk.get('residual_ms')} ms")
            extra[cid] = chk
            anchors_all[cid] = chk.get("anchors", [])

    sources, deltas = [], {}
    span = (30.0, 68.5)
    for r in al["sources"]:
        cid = r["id"]
        new = {k: r[k] for k in ("id", "name", "duration", "numbered")}
        if r["name"] == ref_name:
            new.update(slope=1.0, offset=0.0, anchors=[], accepted=True, max_residual_seconds=0.0,
                       clock_source="reference (identity)")
            sources.append(new)
            continue
        d = cc.get(cid) or extra.get(cid)
        if not d or "studio_derived_clock" not in d:
            raise SystemExit(f"cam{cid}: no studio-derived clock (song cross-check missing)")
        sc, oc = float(d["studio_derived_clock"]["slope"]), float(d["studio_derived_clock"]["offset"])
        anc = []
        for song_t, src_t, rr, _disc, inl in anchors_all.get(cid, []):
            ref_t = s * song_t + o
            anc.append({"source_start": round(src_t, 6), "reference_start": round(ref_t, 6),
                        "offset": ref_t - src_t, "confidence": rr, "inlier": bool(inl),
                        "residual_seconds": ref_t - (sc * src_t + oc)})
        inl_res = [abs(a["residual_seconds"]) for a in anc if a["inlier"]]
        new.update(slope=sc, offset=oc, anchors=anc, accepted=bool(r.get("accepted", True)),
                   max_residual_seconds=max(inl_res) if inl_res else None,
                   anchor_inliers=sum(a["inlier"] for a in anc),
                   clock_source="studio track (song.json cross_check_detail.studio_derived_clock)" if cid in cc else
                   "studio track (song.camera_check run by clock_studio: camera excluded from the base take)",
                   studio_fit={"fit_residual_ms_median": d.get("fit_residual_ms_median"),
                               "fit_residual_ms_p95": d.get("fit_residual_ms_p95"),
                               "n_anchors": d.get("n_anchors"), "n_inliers": d.get("n_inliers"),
                               "song_span": d.get("song_span")})
        sources.append(new)
        so, oo = float(r["slope"]), float(r["offset"])
        # shift a frame gets: new clock - old clock at the same source time, at reference times on the OLD clock
        at = {}
        for tr in (span[0], 0.5 * (span[0] + span[1]), span[1]):
            u = (tr - oo) / so
            at[f"{tr:g}"] = round(1000 * ((sc * u + oc) - tr), 3)
        deltas[cid] = {"delta_ms_at_reference": at,
                       "slope_delta_ppm": round(1e6 * (sc - so), 3),
                       "old_minus_new_drift_ms_per_min": round(60e3 * (so - sc), 3),
                       "old_clock_error_ms_song_cross_check": d.get("residual_ms"),
                       "note": "new - old reference time of the same source instant (> 0: the studio clock puts "
                               "this camera's frames later than alignment.json did)"}
    ivs, mc, mo = intervals_from(sources)
    doc = {
        "reference": ref_name,
        "convention": al["convention"],
        "sources": sources,
        "intervals": ivs,
        "max_count": mc,
        "maximum_overlap": mo,
        "warning": al.get("warning"),
        "clock_kind": "AUDIO clocks (as alignment.json): phone audio-video latency and speaker->phone sound travel "
                      "(~2.9 ms/m) are not included; sync.py's video residual offsets measure what remains",
        "provenance": {
            "version": VERSION, "module": "dancecap.clock_studio build",
            "derived_from": {"song_json": f"dancecap/work/{base}/song/song.json",
                             "align_detail": f"dancecap/work/{base}/song/align_detail.json",
                             "replaces_clocks_of": _rel_ws(al_path)},
            "song_to_reference": {"convention": S.CONVENTION, "slope": s, "offset": o,
                                  "accuracy_ms": sj["alignment"].get("accuracy_ms"),
                                  "residual_ms_median": sj["alignment"].get("residual_ms_median"),
                                  "residual_ms_p95": sj["alignment"].get("residual_ms_p95")},
            "method": "per phone: studio -> phone-source affine fit y = a x + b (8 s onset-feature windows every 8 s, "
                      "robust LS; song.camera_check) composed with studio -> reference: reference = (s / a) * source "
                      "+ (o - s b / a). cam06 = identity (the reference clock is cam06's audio).",
            "anchors_note": "anchors[]: source_start / reference_start are the 8 s studio window CENTRES on the "
                            "phone's source clock and on the reference clock (through the song -> reference line); "
                            "residual_seconds = reference_start - (slope * source_start + offset)",
            "alignment_json_unchanged": True,
        },
        "deltas_vs_alignment_json": deltas,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")  # ASCII (take.py reads it with the locale codec)
    print(f"[clock] wrote {out}")
    for cid, dd in deltas.items():
        print(f"[clock] cam{cid}: new - old {dd['delta_ms_at_reference']} ms, old drifts "
              f"{dd['old_minus_new_drift_ms_per_min']:+.2f} ms/min vs new")
    return doc


# ------------------------------------------------------------------------------------------------ window
def inward_window(lo: float, hi: float, fps: float = FPS_GRID) -> tuple[float, float]:
    """[lo, hi] rounded INWARD to multiples of 1/fps (a value already on the grid stays)."""
    return math.ceil(lo * fps - 1e-9) / fps, math.floor(hi * fps + 1e-9) / fps


def window(alignment: Path, exclude: set[str], within: tuple[float, float], source_dir: Path,
           sync_offsets: dict | None = None, current: tuple[float, float] | None = None,
           spans: dict[str, tuple[float, float, int]] | None = None) -> dict:
    """All-camera video span on the clocks of `alignment` (+ sync_offsets {cam: {delta, rate, t_ref}} when given:
    the sync-corrected frame times mamma prepare uses). current = a take window to check against that coverage.
    spans = {file name: (first_pts, last_pts, packets)} instead of ffprobe (tests)."""
    from .mamma_dyn import apply_offset
    al = json.loads(alignment.read_text(encoding="utf-8"))
    key = "reference" if sync_offsets is None else "sync_corrected"
    rows = {}
    for r in al["sources"]:
        if r["id"] in exclude or not r.get("accepted", True):
            continue
        a, b, n = spans[r["name"]] if spans is not None else video_span(source_dir / r["name"])
        sl, of = _clock(r)
        row = {"first_pts": a, "last_pts": b, "packets": n, "reference": [sl * a + of, sl * b + of]}
        if sync_offsets is not None:
            off = sync_offsets.get(r["id"])
            row["sync"] = off
            row["sync_corrected"] = [float(apply_offset(np.array([x]), off)[0]) for x in row["reference"]]
        rows[r["id"]] = row
    lo = max(within[0], max(v[key][0] for v in rows.values()))
    hi = min(within[1], min(v[key][1] for v in rows.values()))
    t0, t1 = inward_window(lo, hi)
    lim0 = [c for c, v in rows.items() if abs(v[key][0] - lo) < 1e-9]
    lim1 = [c for c, v in rows.items() if abs(v[key][1] - hi) < 1e-9]
    out = {"window": [t0, t1], "exact": [lo, hi], "duration_s": t1 - t0, "frame_times": key,
           "limited_by": {"start": lim0 or ["within"], "end": lim1 or ["within"]},
           "margins_ms": {c: {"start": round(1000 * (t0 - v[key][0]), 2), "end": round(1000 * (v[key][1] - t1), 2)}
                          for c, v in rows.items()},
           "cameras": rows}
    if current is not None:
        cur = [float(current[0]), float(current[1])]
        m = {c: {"start": round(1000 * (cur[0] - v[key][0]), 2), "end": round(1000 * (v[key][1] - cur[1]), 2)}
             for c, v in rows.items()}
        bad = {c: v for c, v in m.items() if v["start"] < 0 or v["end"] < 0}
        all_lo = max(v[key][0] for v in rows.values())
        all_hi = min(v[key][1] for v in rows.values())
        out["current"] = {"window": cur, "margins_ms": m, "inside_every_camera": not bad, "short_cameras": bad,
                          "trimmed": list(inward_window(max(cur[0], all_lo), min(cur[1], all_hi))) if bad else cur}
    return out


def decoded_coverage(work: Path, cams: list[str], win: tuple[float, float],
                     sync_offsets: dict | None = None) -> dict | None:
    """The grid mamma prepare will build from the DECODED frames. Ingest decodes only frames inside the take window,
    so a camera's coverage is its first/last decoded frame time (frames/<cam>/frames.json t, + sync delta when
    given), not its video's first/last packet: window() can say 'inside every camera' while prepare still trims the
    edges (frame spacing ~33 ms > the 13-26 ms video margins). Same rule as mamma_dyn.prepare: master_grid(max(window
    start, latest first frame), min(window end, earliest last frame)) at 30 Hz, starting off the 1/30 s lattice when
    trimmed. Camera-pose gaps at the edges (cameras/<cam>.npz ok) can trim more: mamma/prepare.json grid.trim is
    authoritative. None before ingest."""
    from .mamma_dyn import GRID_HZ, apply_offset, master_grid
    rows, missing = {}, []
    for c in cams:
        p = work / "frames" / c / "frames.json"
        t = np.array([float(r["t"]) for r in json.loads(p.read_text())["frames"]]) if p.exists() else np.zeros(0)
        if not len(t):
            missing.append(c)
            continue
        if sync_offsets is not None:
            t = apply_offset(t, sync_offsets.get(c))
        rows[c] = {"first": round(float(t.min()), 6), "last": round(float(t.max()), 6), "frames": int(len(t)),
                   "start_gap_ms": round(1000 * (float(t.min()) - win[0]), 2),
                   "end_gap_ms": round(1000 * (win[1] - float(t.max())), 2)}
    if not rows:
        return None
    lo = max(win[0], max(v["first"] for v in rows.values()))
    hi = min(win[1], min(v["last"] for v in rows.values()))
    grid = master_grid(lo, hi)
    full = master_grid(win[0], win[1])
    return {"frame_times": "sync_corrected" if sync_offsets is not None else "reference",
            "per_camera": rows, "missing_cameras": missing,
            "grid": {"t0": round(float(grid[0]), 6), "t1": round(float(grid[-1]), 6), "T": int(len(grid)),
                     "hz": GRID_HZ, "on_lattice": bool(abs(lo * GRID_HZ - round(lo * GRID_HZ)) < 1e-6)},
            "window_grid_T": int(len(full)), "frames_lost_vs_window": int(len(full) - len(grid)),
            "limited_by": {"start": [c for c, v in rows.items() if v["first"] > win[0] and v["first"] >= lo - 1e-9],
                           "end": [c for c, v in rows.items() if v["last"] < win[1] and v["last"] <= hi + 1e-9]},
            "note": "mamma prepare's grid from the decoded frames (before camera-pose gaps); prepare.json grid.trim "
                    "is authoritative"}


def time_base(take: Take, sync_offsets: dict | None) -> dict:
    """Which clock beats/audio and poses are on. beats.json, song/beats_studio.json and audio/audio_overlap.wav are
    on the reference camera's AUDIO clock; pose time is sync-corrected VIDEO time t + delta with sync.py's gauge
    (median delta over cameras = 0), which geometry cannot pin. For the same physical instant, pose time - beat time
    = the reference camera's own delta, ASSUMING its video and audio are simultaneous. That is an assumption, not a
    measurement: with the phones' audio clocks aligned to ~1-2 ms (alignment_studio.json), their sync deltas still
    spread over tens to >100 ms (larissa_kadu_38: -32..+107 ms), i.e. phone A/V latencies are tens of ms, so the true
    pose-to-music offset is uncertain by about half that spread ('pose_minus_beat_uncertainty_ms'). timing/export must
    record this gauge (or re-gauge poses to the beat clock by subtracting the reference camera's delta from every sync
    delta) and check it by eye on a visible beat cue (a foot strike) before reading beat-vs-motion offsets < ~50 ms."""
    al = json.loads(_alignment_path(take).read_text(encoding="utf-8"))
    ref = next((r["id"] for r in al["sources"] if r["name"] == al["reference"]), None)
    pred = None
    v = take.work / "clock_studio" / "validation.json"
    if v.exists():
        D = (json.loads(v.read_text()).get("summary") or {}).get("D_video_offsets_expected") or {}
        pred = ((D.get("per_camera") or {}).get(ref) or {}).get("expected_new_ms")
    meas = (round(1000 * float(sync_offsets[ref]["delta"]), 2) if sync_offsets is not None and ref in sync_offsets
            else None)
    ds = [1000 * float(v["delta"]) for v in (sync_offsets or {}).values()]
    unc = round((max(ds) - min(ds)) / 2, 1) if len(ds) >= 2 else None
    return {"reference_camera": ref,
            "beats_and_audio_clock": f"cam{ref} AUDIO (beats.json, song/beats_studio.json, audio/audio_overlap.wav)",
            "pose_clock": "sync-corrected VIDEO time t + delta, sync.py gauge: median delta over cameras = 0",
            # key kept for readers; it is cam<ref>'s sync delta, NOT a measured music-to-motion offset (see 'assumes')
            "pose_minus_beat_ms_measured": meas, "reference_camera_delta_ms": meas,
            "pose_minus_beat_ms_predicted": pred, "pose_minus_beat_uncertainty_ms": unc,
            "assumes": f"cam{ref}'s own audio-video latency ~0 (unknown: the phones' sync deltas spread over "
                       f"{None if unc is None else 2 * unc} ms with their audio aligned to ~1-2 ms, so phone A/V "
                       "latencies are tens of ms); check on a visible beat cue (foot strike) before export",
            "to_beat_clock": f"subtract cam{ref}'s sync delta from every camera's delta (pose time - delta_{ref})",
            "note": "pose time - beat time of the same physical instant = the reference camera's sync delta; record "
                    "it in timing/export (or re-gauge) before reading beat-vs-motion offsets under ~50 ms"}


def take_window(take_name: str) -> dict:
    """window() for a take (its alignment, excluded cameras, sources), checking take.window against every camera's
    coverage; on the SYNC-CORRECTED frame times once work/<take>/sync/offsets.json exists (read through
    mamma_dyn.read_sync_offsets, which refuses a file measured on other clocks) -> work/<take>/clock_studio/window.json.
    mamma prepare trims its grid to the same coverage; a trimmed window is the cue to update the take's window.
    + decoded: the grid mamma prepare builds from the decoded frames (decoded_coverage), + time_base: the clock of
    beats/audio vs poses (time_base)."""
    from .mamma_dyn import read_sync_offsets
    take = Take.load(take_name)
    p = take.work / "sync" / "offsets.json"
    offs = read_sync_offsets(take.work) if p.exists() else None
    r = window(_alignment_path(take), set(take.cfg.get("exclude", [])), tuple(take.window), take.sources[0].path.parent,
               sync_offsets=offs, current=tuple(take.window))
    cur = r["current"]
    dec = decoded_coverage(take.work, [s.id for s in take.sources], tuple(take.window), offs)  # = mamma prepare
    r = {"take": take_name, "alignment": take.cfg["alignment"],
         "sync_file": f"work/{take_name}/sync/offsets.json" if offs is not None else None,
         "note": ("frame times = audio clock + sync.py delta (+ rate)" if offs is not None else
                  "AUDIO clocks only: sync/offsets.json not there yet; sync deltas of tens of ms can move the "
                  "edges (re-run after 'python -m dancecap sync')"), **r,
         "decoded": dec, "time_base": time_base(take, offs)}
    S.write_json(take.work / "clock_studio" / "window.json", r)
    edges = sorted([(v["start"], f"cam{c} start") for c, v in cur["margins_ms"].items()] +
                   [(v["end"], f"cam{c} end") for c, v in cur["margins_ms"].items()])
    print(f"[clock] {take_name} window {cur['window'][0]:.4f}-{cur['window'][1]:.4f} s on {r['frame_times']} frame "
          f"times: inside every camera = {cur['inside_every_camera']}; smallest margins "
          + ", ".join(f"{n} {m:+.1f} ms" for m, n in edges[:3]))
    if not cur["inside_every_camera"]:
        print(f"[clock] -> trim the take window to {cur['trimmed']} (short: {cur['short_cameras']})")
    if dec:
        g = dec["grid"]
        print(f"[clock] decoded frames ({dec['frame_times']}): mamma prepare's grid {g['t0']:.4f}-{g['t1']:.4f} s, "
              f"T = {g['T']} ({dec['frames_lost_vs_window']} fewer than the window's {dec['window_grid_T']}; start "
              f"limited by {dec['limited_by']['start'] or '-'}, end by {dec['limited_by']['end'] or '-'})"
              + (f"; no frames.json for {dec['missing_cameras']}" if dec["missing_cameras"] else ""))
    tb = r["time_base"]
    print(f"[clock] time base: beats/audio on cam{tb['reference_camera']} audio, poses on sync-corrected video "
          f"(median gauge): pose - beat = cam{tb['reference_camera']} delta {tb['reference_camera_delta_ms']} ms "
          f"IF its A/V latency is 0 (+-{tb['pose_minus_beat_uncertainty_ms']} ms unknown), "
          f"{tb['pose_minus_beat_ms_predicted']} ms predicted (validation D)")
    return r


# ------------------------------------------------------------------------------------------------ validate
def _stats(res_ms: np.ndarray, t: np.ndarray) -> dict:
    """Residual distribution (all windows) + drift from a robust line (song.robust_affine: Theil-Sen start, LS on
    windows within max(4 robust sd, 3 ms); a wrong correlation peak in a quiet passage must not set the drift)."""
    res_ms, t = np.asarray(res_ms, float), np.asarray(t, float)
    if len(res_ms) < 3:
        return {"n": int(len(res_ms))}
    a = np.abs(res_ms)
    fit = S.robust_affine(t, res_ms / 1000.0, k=4.0, floor_s=0.003)
    sl, of = 1000.0 * fit["slope"], 1000.0 * fit["offset"]
    out_t = [round(float(x), 2) for x in t[~fit["inlier"]]]
    return {"n": int(len(res_ms)), "median_ms": round(float(np.median(res_ms)), 2),
            "mean_ms": round(float(np.mean(res_ms)), 2),
            "abs_median_ms": round(float(np.median(a)), 2), "abs_p95_ms": round(float(np.percentile(a, 95)), 2),
            "abs_max_ms": round(float(a.max()), 2),
            "robust_sd_ms": round(1.4826 * float(np.median(np.abs(res_ms - np.median(res_ms)))), 2),
            "drift_ms_per_min": round(60.0 * sl, 2),
            "drift_se_ms_per_min": round(60.0 * 1000.0 * fit["slope_se"], 2),
            "line_at_first_ms": round(sl * float(t.min()) + of, 2), "line_at_last_ms": round(sl * float(t.max()) + of, 2),
            "outlier_windows_t": out_t,
            "outlier_rule": "excluded from the drift line only: |res - line| > max(4 robust sd, 3 ms)"}


def video_offsets_expected(base: Take, old: dict, new: dict) -> dict | None:
    """(D) per-phone VIDEO offsets to expect on the new clocks, from the base take's sync.py result: sync measured
    delta_old on the old clocks (true video time = t_old + delta_old); a frame's new time is t_old + change, so
    delta_new = delta_old - change (at the base window's middle), re-gauged to median 0 like sync.py. A prediction
    for sync.py on the new take, not a measurement (most base-take deltas are 'unstable')."""
    p = base.work / "sync" / "offsets.json"
    if not p.exists():
        return None
    offs = json.loads(p.read_text())["offsets"]
    tm = 0.5 * (base.window[0] + base.window[1])
    rows = {}
    for c, v in sorted(offs.items()):
        if c not in old or c not in new or v.get("delta_ms") is None:
            continue
        ch = clock_change_ms(old[c], new[c], tm)
        rows[c] = {"delta_old_ms": v["delta_ms"], "clock_change_ms": round(ch, 2), "raw_ms": v["delta_ms"] - ch,
                   "stable": bool(v.get("stable")), "spread_ms": v.get("spread_ms")}
    g = float(np.median([r["raw_ms"] for r in rows.values()]))
    for r in rows.values():
        r["expected_new_ms"] = round(r.pop("raw_ms") - g, 2)
    old_d = np.array([r["delta_old_ms"] for r in rows.values()])
    new_d = np.array([r["expected_new_ms"] for r in rows.values()])
    chg = np.array([r["clock_change_ms"] for r in rows.values()])
    return {"from": f"dancecap/work/{base.name}/sync/offsets.json (window {base.window[0]:g}-{base.window[1]:g} s, "
                    f"old clocks)", "at_reference_s": tm, "per_camera": rows,
            "sd_ms": {"old_clocks_measured": round(float(old_d.std()), 1), "new_clocks_expected": round(float(new_d.std()), 1)},
            "range_ms": {"old_clocks_measured": [round(float(old_d.min()), 1), round(float(old_d.max()), 1)],
                         "new_clocks_expected": [round(float(new_d.min()), 1), round(float(new_d.max()), 1)]},
            "corr_delta_old_vs_clock_change": round(float(np.corrcoef(old_d, chg)[0, 1]), 2),
            "note": "delta_new = delta_old - clock change, median 0 (sync.py's gauge). The audio clocks do not remove "
                    "the per-phone video offsets (A/V latency, sound travel): sync.py must measure them on the new "
                    "take before any MAMMA fit."}


def validate(base: str, alignment: Path, review_take: str, span: tuple[float, float] = (30.0, 68.5),
             plot: bool = True, write_back: bool = False) -> dict:
    take = Take.load(base)
    if write_back:
        _check_target(alignment, take, "validate --write-back")
    old = {r["id"]: r for r in json.loads(_alignment_path(take).read_text(encoding="utf-8"))["sources"]}
    new_doc = json.loads(alignment.read_text(encoding="utf-8"))
    new = {r["id"]: r for r in new_doc["sources"]}
    sets = {"alignment_json": old, "studio": new}
    sj = S.read_json(take.work / "song" / "song.json")
    s, o = float(sj["alignment"]["slope"]), float(sj["alignment"]["offset"])
    cams = [src.id for src in take.sources]  # the base take's used cameras (08 excluded there)
    ref = next(r["id"] for r in new.values() if r["name"] == new_doc["reference"])
    cfg = S.SongConfig()
    dt = S.frame_dt(S.ALIGN_SR)
    info = S.song_info(base)
    print(f"[clock] decoding the studio track + {len(cams)} phones at {S.ALIGN_SR} Hz")
    xs = S.load_mono(info.path, S.ALIGN_SR)
    Fs = S.onset_features(xs, S.ALIGN_SR)
    F = {c: S.onset_features(S.load_mono(take.source(c).path, S.ALIGN_SR), S.ALIGN_SR) for c in cams}

    # ---- (A) studio track vs each phone through either clock set
    centres_ref = np.arange(span[0] + 2.5, span[1] - 2.5 + 1e-9, 1.0)
    A, A_rows = {}, {}
    for c in cams:
        A[c], A_rows[c] = {}, {}
        for name, clk in sets.items():
            sl, of = _clock(clk[c])
            a, b = song_to_source(s, o, clk[c])  # predicted song -> phone source time
            anc = S.local_anchors(Fs, F[c], dt, a, b, (centres_ref - o) / s, 4.0, cfg.anchor_search_s)
            ok = [x for x in anc if x["r"] >= cfg.min_r and not x["edge"]]
            x = np.array([q["song_t"] for q in ok])
            y = np.array([q["ref_t"] for q in ok])  # measured phone source seconds
            t = s * x + o
            res = 1000 * ((sl * y + of) - t)
            A[c][name] = _stats(res, t)
            A_rows[c][name] = [[round(float(tt), 3), round(float(rr), 3), round(float(q["r"]), 3)]
                               for tt, rr, q in zip(t, res, ok)]
        print(f"[clock] (A) cam{c} vs studio over {span[0]:g}-{span[1]:g} s: alignment.json median "
              f"{A[c]['alignment_json'].get('median_ms')} ms (drift {A[c]['alignment_json'].get('drift_ms_per_min')} "
              f"ms/min), studio clock median {A[c]['studio'].get('median_ms')} ms |p95| "
              f"{A[c]['studio'].get('abs_p95_ms')} (drift {A[c]['studio'].get('drift_ms_per_min')} ms/min)")

    # ---- (B) phone pairs only (no studio track), solved for per-phone clock errors with cam06 = 0
    centres_pair = np.arange(span[0] + 4.5, span[1] - 4.5 + 1e-9, 1.5)
    B_pairs, B = {}, {}
    pairs = [(a_, b_) for i, a_ in enumerate(cams) for b_ in cams[i + 1:]]
    for name, clk in sets.items():
        rows = []
        for ca, cb in pairs:
            sa, oa = _clock(clk[ca])
            sb, ob = _clock(clk[cb])
            anc = S.local_anchors(F[ca], F[cb], dt, sa / sb, (oa - ob) / sb, (centres_pair - oa) / sa, 8.0,
                                  cfg.anchor_search_s)
            ok = [q for q in anc if q["r"] >= cfg.min_r and not q["edge"] and q["psr"] >= 1.05]
            if len(ok) < 3:
                B_pairs.setdefault(name, {})[f"{ca}-{cb}"] = {"n": len(ok)}
                continue
            d = np.array([1000 * ((sb * q["ref_t"] + ob) - (sa * q["song_t"] + oa)) for q in ok])  # e_b - e_a
            med = float(np.median(d))
            sd = 1.4826 * float(np.median(np.abs(d - med)))
            rows.append((ca, cb, med, len(ok), sd))
            B_pairs.setdefault(name, {})[f"{ca}-{cb}"] = {"median_ms": round(med, 2), "n": len(ok),
                                                          "robust_sd_ms": round(sd, 2),
                                                          "r_median": round(float(np.median([q['r'] for q in ok])), 3)}
        unk = [c for c in cams if c != ref]
        M = np.zeros((len(rows), len(unk)))
        yv = np.zeros(len(rows))
        w = np.zeros(len(rows))
        for k, (ca, cb, med, n, sd) in enumerate(rows):
            if cb != ref:
                M[k, unk.index(cb)] += 1
            if ca != ref:
                M[k, unk.index(ca)] -= 1
            yv[k] = med
            w[k] = math.sqrt(n) / max(sd, 1.0)
        e = np.linalg.lstsq(M * w[:, None], yv * w, rcond=None)[0]
        resid = yv - M @ e
        B[name] = {"clock_error_ms": {c: round(float(v), 2) for c, v in zip(unk, e)} | {ref: 0.0},
                   "pair_closure_residual_ms_median": round(float(np.median(np.abs(resid))), 2),
                   "pair_closure_residual_ms_max": round(float(np.abs(resid).max()), 2),
                   "pairs_used": len(rows), "pairs_total": len(pairs)}
        print(f"[clock] (B) phones only, {name}: per-phone clock error (ms, cam{ref} = 0) " +
              ", ".join(f"{c} {v:+.1f}" for c, v in B[name]["clock_error_ms"].items()) +
              f"; pair closure |res| median {B[name]['pair_closure_residual_ms_median']} ms, "
              f"max {B[name]['pair_closure_residual_ms_max']} ms ({len(rows)}/{len(pairs)} pairs)")
    tmid = 0.5 * (span[0] + span[1])
    pred = {}
    for c in cams:
        so, oo = _clock(old[c])
        sn, on = _clock(new[c])
        u = (tmid - on) / sn
        pred[c] = round(1000 * ((so * u + oo) - tmid), 2)  # old clock error at mid-span if the studio clocks are true
    B["studio_predicted_alignment_json_error_ms_at_mid_span"] = pred

    # ---- (C) envelope spot-check around strong studio onsets
    spot = _spot_check(Fs, F, sets, cams, s, o, dt, span)
    # ---- (D) per-phone VIDEO offsets to expect on the new clocks (the base take's sync.py result, moved)
    D = video_offsets_expected(take, old, new)
    if D:
        print("[clock] (D) expected video offsets on the new clocks (ms, median 0): " +
              ", ".join(f"{c} {r['expected_new_ms']:+.1f}" for c, r in D["per_camera"].items()) +
              f"; SD {D['sd_ms']['new_clocks_expected']} ms (old clocks, measured: {D['sd_ms']['old_clocks_measured']})"
              f"; corr(delta_old, clock change) {D['corr_delta_old_vs_clock_change']}")
    summary = {
        "accuracy_note": "audio clocks: every phone within about +-1-2 ms of cam06's audio in the onset-envelope "
                         "sense; the sub-ms fit residuals / medians below are precision from the same onset-feature "
                         "machinery, not capture-time accuracy (raw-waveform correlation puts some phones ~10 ms "
                         "elsewhere, cam03). VIDEO frame sync is not covered: see D and sync.py.",
        "span": list(span),
        "A_studio_track": {"method": "each phone's audio mapped through the clock set vs the studio track mapped "
                                     "through song.json's song->reference fit; 4 s onset-feature windows every 1 s "
                                     f"(centres {centres_ref[0]:g}-{centres_ref[-1]:g} s), r >= {cfg.min_r}; residual = "
                                     "clock(phone instant) - studio-line(same instant), ms; drift = robust line (outlier windows listed)",
                           "per_camera": A},
        "B_phones_only": {"method": "every pair of phones (no studio track) mapped through the clock set, 8 s windows "
                                    f"every 1.5 s (centres {centres_pair[0]:g}-{centres_pair[-1]:g} s), r >= {cfg.min_r}, "
                                    "psr >= 1.05; pair medians solved (weighted LS) for each phone's clock error with "
                                    f"cam{ref} = 0. Independent of the studio track.", **B},
        "C_envelope_spot_check": {k: v for k, v in spot.items() if k != "_plot"},
        "D_video_offsets_expected": D,
    }
    out_dir = PACKAGE_ROOT / "work" / review_take
    S.write_json(out_dir / "clock_studio" / "validation.json",
                 {"summary": summary, "A_rows_columns": ["t_ref", "residual_ms", "r"], "A_rows": A_rows,
                  "B_pairs": B_pairs})
    if write_back:
        new_doc["validation"] = {**summary, "detail": f"dancecap/work/{review_take}/clock_studio/validation.json",
                                 "review_image": f"dancecap/work/{review_take}/review/clock_studio_validation.png"}
        alignment.write_text(json.dumps(new_doc, indent=2), encoding="utf-8")
        print(f"[clock] validation summary written back into {alignment}")
    if plot:
        _plot(out_dir / "review" / "clock_studio_validation.png", A_rows, B, spot, cams, ref, span, D)
    return summary


def _broadband(F: np.ndarray) -> np.ndarray:
    return F.mean(0)


def _spot_check(Fs, F, sets, cams, s, o, dt, span, n_onsets: int = 3, half_s: float = 0.12) -> dict:
    """Peak time of each phone's broadband onset envelope near the strongest studio onsets (reference clock)."""
    es = _broadband(Fs)
    t_song = np.arange(len(es)) * dt
    t_ref = s * t_song + o
    sel = (t_ref > span[0] + 3) & (t_ref < span[1] - 3)
    idx = np.nonzero(sel)[0]
    order = idx[np.argsort(es[idx])[::-1]]
    picks = []
    for i in order:
        if all(abs(t_ref[i] - t_ref[j]) > 6.0 for j in picks):
            picks.append(i)
        if len(picks) == n_onsets:
            break
    picks = sorted(picks)
    grid = np.arange(-half_s, half_s + 1e-9, 0.001)
    res, plot = {"onsets_ref_s": [round(float(t_ref[i]), 4) for i in picks], "peak_lag_ms": {}}, {}
    for name, clk in sets.items():
        res["peak_lag_ms"][name] = {}
        for c in cams:
            sl, of = _clock(clk[c])
            e = _broadband(F[c])
            lags, curves = [], []
            for i in picks:
                tt = t_ref[i] + grid
                u = (tt - of) / sl / dt
                v = np.interp(u, np.arange(len(e)), e, left=0, right=0)
                near = np.abs(grid) <= 0.06
                k = int(np.argmax(np.where(near, v, -np.inf)))
                lags.append(round(1000 * float(grid[k]), 1))
                curves.append(v)
            res["peak_lag_ms"][name][c] = lags
            plot[(name, c)] = curves
        lag_all = np.array([res["peak_lag_ms"][name][c] for c in cams], float)
        res.setdefault("spread_ms", {})[name] = [round(float(np.ptp(lag_all[:, j])), 1) for j in range(len(picks))]
    # studio envelope at the same instants
    ss = []
    for i in picks:
        u = ((t_ref[i] + grid) - o) / s / dt
        ss.append(np.interp(u, np.arange(len(es)), es, left=0, right=0))
    res["note"] = ("peak of each phone's broadband onset envelope within +-60 ms of the studio onset (1 ms grid on 5 ms "
                   "features); spread = max - min over phones at each onset")
    res["_plot"] = {"grid": grid, "curves": plot, "studio": ss}
    return res


def _plot(path: Path, A_rows, B, spot, cams, ref, span, D: dict | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(cams)}
    n_on = len(spot["onsets_ref_s"])
    fig = plt.figure(figsize=(16, 18.5 if D else 15))
    hr = [1.2, 1.2, 0.9] + ([0.9] if D else []) + [1.6]
    gs = fig.add_gridspec(len(hr), max(2, n_on), height_ratios=hr, hspace=0.5, wspace=0.25)
    for k, (name, title) in enumerate((("alignment_json", "alignment.json clocks (phone-to-phone audio)"),
                                       ("studio", "studio-derived clocks (alignment_studio.json)"))):
        ax = fig.add_subplot(gs[k, :])
        for c in cams:
            rows = np.array(A_rows[c][name]) if A_rows[c][name] else np.zeros((0, 3))
            if len(rows):
                ax.plot(rows[:, 0], rows[:, 1], "-o", ms=2.5, lw=1.2, color=cols[c], label=f"cam{c}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlim(*span)
        ax.set_ylim(-40, 40)
        ax.set_ylabel("clock - studio (ms)")
        ax.grid(alpha=0.3)
        ax.set_title(f"(A) {title}: each phone's audio vs the studio track on the reference clock "
                     f"(4 s windows every 1 s)", loc="left", fontsize=10)
        if k == 0:
            ax.legend(ncol=len(cams), fontsize=8, loc="upper left")
    ax = fig.add_subplot(gs[2, :])
    x = np.arange(len(cams))
    pred = B["studio_predicted_alignment_json_error_ms_at_mid_span"]
    ax.bar(x - 0.27, [B["alignment_json"]["clock_error_ms"].get(c, np.nan) for c in cams], 0.27,
           label="alignment.json clocks, measured from phone pairs only", color="tab:red", alpha=0.75)
    ax.bar(x, [pred.get(c, np.nan) for c in cams], 0.27, label="alignment.json error predicted by the studio clocks",
           color="tab:orange", alpha=0.75)
    ax.bar(x + 0.27, [B["studio"]["clock_error_ms"].get(c, np.nan) for c in cams], 0.27,
           label="studio-derived clocks, measured from phone pairs only", color="tab:green", alpha=0.85)
    ax.set_xticks(x, [f"cam{c}" for c in cams])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("clock error (ms)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0))  # outside: never hides a bar
    ax.set_title(f"(B) phones only (no studio track): every phone pair cross-correlated, solved per phone "
                 f"(cam{ref} = 0), {span[0]:g}-{span[1]:g} s", loc="left", fontsize=10)
    if D:
        ax = fig.add_subplot(gs[3, :])
        dc = [c for c in cams if c in D["per_camera"]]
        x = np.arange(len(dc))
        rows = [D["per_camera"][c] for c in dc]
        ax.bar(x - 0.2, [r["delta_old_ms"] for r in rows], 0.4, color="tab:red", alpha=0.75,
               label=f"measured by sync.py on alignment.json clocks (SD {D['sd_ms']['old_clocks_measured']} ms)")
        ax.bar(x + 0.2, [r["expected_new_ms"] for r in rows], 0.4, color="tab:purple", alpha=0.85,
               label=f"expected on the studio clocks = measured - clock change (SD "
                     f"{D['sd_ms']['new_clocks_expected']} ms)")
        for i, r in enumerate(rows):
            ax.text(i + 0.2, r["expected_new_ms"] + (3 if r["expected_new_ms"] >= 0 else -3),
                    f"{r['expected_new_ms']:+.0f}", ha="center", va="bottom" if r["expected_new_ms"] >= 0 else "top",
                    fontsize=8)
        ax.set_xticks(x, [f"cam{c}\n{'stable' if r['stable'] else 'unstable'} ({r['spread_ms']:.0f} ms)"
                          for c, r in zip(dc, rows)], fontsize=8)
        vals = [v for r in rows for v in (r["delta_old_ms"], r["expected_new_ms"])] + [-33.3, 33.3]
        ax.set_ylim(min(vals) - 18, max(vals) + 18)  # room for the value labels
        ax.axhline(0, color="k", lw=0.8)
        for yy in (-33.3, 33.3):
            ax.axhline(yy, color="0.5", lw=0.6, ls=":")
        ax.set_ylabel("video offset (ms, median 0)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0))
        ax.set_title(f"(D) per-phone VIDEO offsets are NOT fixed by audio clocks: {D['from']}; dotted = +-1 frame. "
                     "A prediction - run sync.py on the new take before MAMMA", loc="left", fontsize=10)
    P = spot["_plot"]
    g = 1000 * P["grid"]
    inner = gs[len(hr) - 1, :].subgridspec(1, 2 * n_on, wspace=0.15)
    for j in range(n_on):
        for k, name in enumerate(("alignment_json", "studio")):
            ax = fig.add_subplot(inner[0, 2 * j + k])
            st = P["studio"][j]
            ax.plot(g, -1.2 + st / (st.max() + 1e-9), color="k", lw=1.2)
            for i, c in enumerate(cams):
                v = P["curves"][(name, c)][j]
                ax.plot(g, i + v / (v.max() + 1e-9), color=cols[c], lw=1.1)
                lag = spot["peak_lag_ms"][name][c][j]
                ax.plot([lag], [i + 1.0], "v", color=cols[c], ms=4)
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_yticks([-1.2] + list(range(len(cams))), ["studio"] + [f"{c}" for c in cams], fontsize=7)
            ax.set_xlim(g[0], g[-1])
            ax.set_xlabel("ms from studio onset", fontsize=8)
            ax.tick_params(labelsize=7)
            lab = "alignment.json" if name == "alignment_json" else "studio clocks"
            ax.set_title(f"t={spot['onsets_ref_s'][j]:.2f} s, {lab}\nspread {spot['spread_ms'][name][j]:.0f} ms",
                         fontsize=8)
    fig.suptitle("(C, bottom) envelope spot-check: each phone's own audio onset envelope mapped to the reference "
                 "clock around strong studio onsets (black = studio track)", fontsize=10, y=0.07 if D else 0.08)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"[clock] review image {path}")


# ------------------------------------------------------------------------------------------------ keyframes
def reuse_keyframes(src_take: str, dst_take: str) -> dict:
    src, dst = Take.load(src_take), Take.load(dst_take)
    out = {}
    for s_new in dst.sources:
        cam = s_new.id
        kf_src = src.work / "keyframes" / cam
        if not (kf_src / "keyframes.json").exists():
            out[cam] = "no keyframes in the base take"
            continue
        recs = json.loads((kf_src / "keyframes.json").read_text())
        kf_dst = dst.work / "keyframes" / cam
        kf_dst.mkdir(parents=True, exist_ok=True)
        m_src, m_dst = src.work / "people_masks_kf" / cam, dst.work / "people_masks_kf" / cam
        m_dst.mkdir(parents=True, exist_ok=True)
        new_recs, dts = [], []
        for r in recs:
            shutil.copy2(kf_src / r["file"], kf_dst / r["file"])  # copy, never link
            m = m_src / (Path(r["file"]).stem + ".png")
            if m.exists():
                shutil.copy2(m, m_dst / m.name)
            t = s_new.to_reference(float(r["pts"]))
            dts.append(1000 * (t - float(r["t"])))
            new_recs.append({"file": r["file"], "pts": r["pts"], "t": t})
        (kf_dst / "keyframes.json").write_text(json.dumps(new_recs, indent=1))
        out[cam] = {"n": len(new_recs), "t_shift_ms": [round(min(dts), 2), round(max(dts), 2)]}
    meta = {"from": f"work/{src_take}/keyframes + people_masks_kf", "how": "copied (not linked); t = new clock(pts); "
            "file names keep the base take's ms labels", "cameras": out}
    S.write_json(dst.work / "keyframes" / "reused_from.json", meta)
    print(f"[clock] keyframes reused from {src_take}: " + json.dumps(out))
    return meta


# ------------------------------------------------------------------------------------------------ cli
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.clock_studio", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--base", default="larissa_kadu")
    p.add_argument("--out", required=True, help="path (relative to dancecap/ or absolute)")
    p = sub.add_parser("validate")
    p.add_argument("--base", default="larissa_kadu")
    p.add_argument("--alignment", required=True)
    p.add_argument("--review-take", required=True)
    p.add_argument("--span", type=float, nargs=2, default=(30.0, 68.5))
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--write-back", action="store_true",
                   help="also store the summary in the alignment file (only a file clock_studio built)")
    p = sub.add_parser("window")
    p.add_argument("--take", help="check this take's window: its alignment/exclude/sources (+ sync/offsets.json "
                                  "when present) -> work/<take>/clock_studio/window.json")
    p.add_argument("--alignment")
    p.add_argument("--source-dir", default="../dances/Larissa-Kadu-demo")
    p.add_argument("--exclude", default="08")
    p.add_argument("--within", type=float, nargs=2, default=(30.0, 68.5))
    p = sub.add_parser("reuse-keyframes")
    p.add_argument("--from", dest="src", default="larissa_kadu")
    p.add_argument("--to", dest="dst", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.base, _resolve(a.out))
    elif a.cmd == "validate":
        validate(a.base, _resolve(a.alignment), a.review_take, tuple(a.span), plot=not a.no_plot,
                 write_back=a.write_back)
    elif a.cmd == "window" and a.take:
        r = take_window(a.take)
        print(json.dumps({k: r[k] for k in ("take", "frame_times", "note", "window", "exact", "limited_by",
                                            "current")}, indent=1))
    elif a.cmd == "window":
        if not a.alignment:
            raise SystemExit("window: --alignment <file> or --take <take>")
        r = window(_resolve(a.alignment), set(x for x in a.exclude.split(",") if x), tuple(a.within),
                   _resolve(a.source_dir))
        print(json.dumps(r, indent=1))
    elif a.cmd == "reuse-keyframes":
        reuse_keyframes(a.src, a.dst)


if __name__ == "__main__":
    main()
