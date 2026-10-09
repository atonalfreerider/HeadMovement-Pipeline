"""larissa_kadu_full front-end: the four segment takes of the whole dance on ONE Runpod pod (work/larissa_kadu_full/PLAN.md
3.1-3.2: C1-C5, P1). larissa_kadu_38 (30.15-68.38 s) is reused as is; its room map, world frame and scale carry over.

    python -m dancecap.front_full segments            # local: takes/larissa_kadu_full_{a,b,c,d}.toml from [segments]
    python -m dancecap.front_full prep-local          # local: map reuse (+ foreign map images as keyframes), world copy
    python -m dancecap.front_full write-job           # local: work/larissa_kadu_full/front/pod_job.json
    python -m dancecap.front_full session --max-cost 9 --max-minutes 330   # local: ONE pod (host checked), all segments
    python -m dancecap.front_full run [--resume]      # pod: the multi-take driver (the job runs `supervise`)

Segment takes (takes/larissa_kadu_full_<id>.toml, generated): window = the segment's FRONT span (PLAN.md 2.3), every
camera that records inside it, [ingest] max_fps = 30 (cam02's 60 fps -> every 2nd frame, C1), [cameras] loc =
{kf_by_time = true}, [sync] required, the user's camera types, [segment] fit_k / grid_anchor for prepare (C6).

Map reuse (C3, prep-local): larissa_kadu_38's bundle-adjusted map (main model + the attached cameras' sub-models, the
same SfM frame as larissa_kadu and larissa_kadu_38) is copied with its images renamed:
  keyframes/<cam>/<ms>.jpg (30-68 s, every 4 s)   copied files, same names
  frames/<cam>/<38 file> (56-62 s window images)  FOREIGN to every segment window -> copied in as extra keyframes
                                                  keyframes/<cam>/<ref ms>.jpg (+ masks, cached ALIKED features)
so every map image is in the segment's frame store, the cached features index the model's 2D points exactly, and the
SfM frame is IDENTICAL: larissa_kadu_38's cameras/world_from_sfm.json (gravity, floor, betas-corrected scale) is
COPIED (C4; no world re-solve). Keyframes of cameras that do not record in a segment (04, 09 in c/d) are copied too:
they are cross-camera references only (features.list_images reads keyframes without frames.json).

Pod driver tasks per segment s (names "<s>.<task>"), one scheduler and one RAM / CPU / GPU budget over all segments:
  ingest:<cam>  python -m dancecap ingest <take> --cams <cam> --window-only   (source MP4s uploaded, ffmpeg on the pod: C2)
  detect:<cam>  YOLO det / seg / pose of the window frames (python -m dancecap.front_full detect-one)
  sam:<cam>     labels + SAM 2.1 masks + identity + body/hair (front_pod dancer)
  loc:<cam>     per-frame localisation against the reused map (front_pod loc-part / loc-join; main-model cameras)
  sub:<cam>     attached cameras' sub-model localisation (cameras.reuse presolve-sub)
  ma2d:<cam>    MAMMA ma_2d (front_pod ma2d)
  p2:<cam>      P2 localisation gate on EVERY frame (>= 90 % ok, no failed stretch > 8 s) -> cameras/loc_gate_<cam>.json;
                a failing camera runs the conditional map extension C7 (dancecap.map_extend, when present on the pod)
  merge         dancers_pod merge + people/report.json
  cameras       python -m dancecap cameras <take> --stages attach,qa   (world COPIED; 05/06 attach re-solved per segment)
  sync          python -m dancecap sync <take> --gauge-ref larissa_kadu_38   (C5)
  chain         world_scale chain (cross-check), geocheck  dancers_geoveto --check
  prepare       mamma prepare --skip-2d --coverage union --grid-anchor 30.151176844227862 --grid-k=<fit_k>   (C6)
plus decode_check (pod-decoded frames vs larissa_kadu_38's local frames of the same PTS). No hands, no faces.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from . import front_pod as fp
from .take import PACKAGE_ROOT, WORKSPACE, Take

FULL = "larissa_kadu_full"
REF = "larissa_kadu_38"
SEG_IDS = ("a", "b", "c", "d")
GRID_ANCHOR = 30.151176844227862
GRID_HZ = 30.0
MAX_FPS = 30
YOLO = {  # ultralytics GitHub release assets v8.3.0, byte-identical to the local copies (sha256 checked 2026-10-07)
    "yolo12x.pt": ("mamma/data/weights/yolo/yolo12x.pt",
                   "682ce8dadee004dbe964950f1bf3eda451671815a6ed62db80b398916b9b7c6f"),
    "yolo11x-seg.pt": ("dancecap/weights/yolo11x-seg.pt",
                       "4e53a5f5fd3ee2ae3361c62169c6bb3ed4ae251dd0e57606e230955aa52d919c"),
    "yolo11x-pose.pt": ("dancecap/weights/yolo11x-pose.pt",
                        "013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8"),
}
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{}"
ULTRALYTICS = "ultralytics==8.4.138"  # as locally and in the detect09 pod session (2026-10-06)


def seg_name(sid: str) -> str:
    return f"{FULL}_{sid}"


def full_work() -> Path:
    return PACKAGE_ROOT / "work" / FULL


def ref_work() -> Path:
    return PACKAGE_ROOT / "work" / REF


def full_cfg() -> dict:
    return tomllib.loads((PACKAGE_ROOT / "takes" / f"{FULL}.toml").read_text(encoding="utf-8"))


def seg_cfg(sid: str) -> dict:
    return tomllib.loads((PACKAGE_ROOT / "takes" / f"{seg_name(sid)}.toml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ segment takes


def _tv(v) -> str:
    """TOML value (scalars, lists, inline tables)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_tv(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_tv(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v))


def _table(name: str, d: dict) -> list[str]:
    out = [f"[{name}]"]
    out += [f"{k} = {_tv(v)}" for k, v in d.items() if not (isinstance(v, dict) and k == "shape_edits")]
    return out


def write_segments() -> list[Path]:
    """takes/larissa_kadu_full_<id>.toml from takes/larissa_kadu_full.toml [segments] (generated, never edited)."""
    cfg = full_cfg()
    full = Take.load(FULL)
    all_cams = [s.id for s in full.sources]
    out = []
    for sid, seg in cfg["segments"].items():
        cams = [c for c in seg["cams"] if c in all_cams]
        excl = sorted(set(cfg.get("exclude", [])) | (set(all_cams) - set(cams)))
        lines = [
            f"# GENERATED by `python -m dancecap.front_full segments` from takes/{FULL}.toml [segments.{sid}] "
            f"({time.strftime('%Y-%m-%d %H:%M')}); do not edit.",
            f"# Segment {sid} of the whole dance (work/{FULL}/PLAN.md 2.3): the front-end (ingest .. mamma prepare) of",
            f"# every camera recording inside window = the segment's front span; the MAMMA fit uses grid indices fit_k on",
            f"# larissa_kadu_38's lattice t_k = {GRID_ANCHOR!r} + k/30. Room map, world frame and scale are",
            f"# larissa_kadu_38's (dancecap.front_full prep-local). Hands / faces are not run.",
            f"name = {_tv(seg_name(sid))}",
            f"source_dir = {_tv(cfg['source_dir'])}",
            f"alignment = {_tv(cfg['alignment'])}",
            f"exclude = {_tv(excl)}",
            f"window = {_tv([float(x) for x in seg['front']])}",
            "keyframe_span = [30.0, 68.0]   # the reused larissa_kadu_38 keyframes (+ its 56-62 s map images as keyframes)",
            "keyframe_step = 4.0",
            "",
            "[segment]",
            f"parent = {_tv(FULL)}",
            f"id = {_tv(sid)}",
            f"fit_k = {_tv([int(x) for x in seg['fit_k']])}",
            f"grid_anchor = {GRID_ANCHOR!r}",
            f"cams = {_tv(cams)}",
            f"reuse = {_tv(REF)}",
            "",
            "[ingest]",
            f"max_fps = {MAX_FPS}   # cam02 (59.94 fps) keeps every 2nd frame: the fused grid is 30 Hz (PLAN.md C1)",
            "",
            "[sync]",
            "required = true",
            "",
            "[cameras]",
            "loc = { kf_by_time = true }",
            "",
        ]
        if cfg.get("world"):   # the farm's world package pin (dancecap.farm.world.pin_tomls)
            lines += _table("world", {**cfg["world"], "role": "segment"}) + [""]
        lines += _table("dance", cfg["dance"]) + [""]
        for role in ("lead", "follow"):
            d = cfg["dancers"][role]
            lines += _table(f"dancers.{role}", d) + [""]
            if d.get("shape_edits"):
                lines += _table(f"dancers.{role}.shape_edits", d["shape_edits"]) + [""]
        for c in cams:
            ct = (cfg.get("cameras") or {}).get(c)
            if isinstance(ct, dict):
                lines += [f'[cameras."{c}"]'] + [f"{k} = {_tv(v)}" for k, v in ct.items()] + [""]
        p = PACKAGE_ROOT / "takes" / f"{seg_name(sid)}.toml"
        p.write_text("\n".join(lines), encoding="utf-8")
        t = Take.load(seg_name(sid))  # parses
        assert [s.id for s in t.sources] == [c for c in all_cams if c in cams], (sid, [s.id for s in t.sources])
        out.append(p)
        print(f"[front_full] {p.name}: window {t.window}, cams {[s.id for s in t.sources]}, fit_k {seg['fit_k']}")
    return out


# ------------------------------------------------------------------------------------------------ local preparation


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)


def _sha1(p: Path) -> str:
    import hashlib
    return hashlib.sha1(p.read_bytes()).hexdigest()


def prep_segment(sid: str, force: bool = False, base: Path | None = None) -> dict:
    """Map reuse + world copy for one segment take (local, before any pod; no frames needed). base: the reference tree
    to copy from (default larissa_kadu_38's work dir; the farm passes the world package's ref/, same layout)."""
    import numpy as np
    import pycolmap

    from .cameras.reuse import MAP_IMAGES, rename_model
    take = Take.load(seg_name(sid))
    work = take.work.resolve()
    base = (Path(base) if base else ref_work()).resolve()
    cams_dir = work / "cameras"
    sw = cams_dir / "sfm_work"
    if (cams_dir / "sfm").exists() and not force:
        info = json.loads((sw / MAP_IMAGES).read_text())
        print(f"[front_full] {take.name}: map already reused ({len(info['images'])} images)")
        return info
    info_38 = json.loads((base / "cameras" / "sfm_work" / MAP_IMAGES).read_text())
    map_cams = sorted({n.split("/")[1] for n in info_38["images"]})
    # 1. keyframes (+ their people masks) of every map camera
    for c in map_cams:
        for sub, pat in (("keyframes", "*.jpg"), ("people_masks_kf", "*.png")):
            for f in sorted((base / sub / c).glob(pat)):
                _copy(f, work / sub / c / f.name)
    # 2. the 38 s window images of the map are foreign to every segment window: copy them in as keyframes
    rename: dict[str, str] = {}
    added: dict[str, list] = {}
    in_window: list[str] = []
    for c in map_cams:
        kj = json.loads((base / "keyframes" / c / "keyframes.json").read_text())
        names = {r["file"] for r in kj}
        meta = json.loads((base / "frames" / c / "frames.json").read_text())
        by_file = {r["file"]: r for r in meta["frames"]}
        for name in sorted(n for n in info_38["images"] if n.startswith(f"frames/{c}/")):
            r = by_file[Path(name).name]
            # a map image that lies INSIDE this take's window (a farm chunk next to / over the reference span, e.g. the 38|b seam refit)
            # is copied in as a keyframe all the same: it is the model's own image, the take's frames are extracted separately
            if take.window[0] <= r["t"] <= take.window[1]:
                in_window.append(name)
            ms = int(round(r["t"] * 1000))
            src = base / name
            while f"{ms:07d}.jpg" in names:  # never two model images with one name (even of the same frame)
                ms += 1
            new = f"{ms:07d}.jpg"
            _copy(src, work / "keyframes" / c / new)
            m = base / "people_masks" / c / f"{Path(name).stem}.png"
            if m.exists():
                _copy(m, work / "people_masks_kf" / c / f"{ms:07d}.png")
            kj.append({"file": new, "pts": r["pts"], "t": r["t"], "from": f"{REF}/{name}"})
            names.add(new)
            rename[name] = f"keyframes/{c}/{new}"
            added.setdefault(c, []).append([name, new])
        kj.sort(key=lambda x: x["t"])
        (work / "keyframes" / c).mkdir(parents=True, exist_ok=True)
        (work / "keyframes" / c / "keyframes.json").write_text(json.dumps(kj, indent=1))
    # 3. the models, renamed (features copied under the new keys; rename_model checks they index the 2D points)
    info = {"reused_from": REF, "base_sfm": str(base / "cameras" / "sfm"), "subs": {},
            "foreign_as_keyframes": added, "map_images_inside_window": in_window,
            "note": "dancecap.front_full prep-local (map reuse with foreign images)"}
    main = pycolmap.Reconstruction(str(base / "cameras" / "sfm"))
    info["main"] = rename_model(main, rename, base, work)
    if info["main"]["missing"] or info["main"]["different_bytes"]:
        raise RuntimeError(f"{take.name}: map images missing / different: {info['main']['missing'][:5]} "
                           f"{info['main']['different_bytes'][:5]}")
    (cams_dir / "sfm").mkdir(parents=True, exist_ok=True)
    main.write_binary(str(cams_dir / "sfm"))
    names = sorted(im.name for im in main.images.values())
    for d in sorted((base / "cameras" / "sfm_work").glob("sub_*/best")):
        if not (d / "images.bin").exists():
            continue
        sub = pycolmap.Reconstruction(str(d))
        st = rename_model(sub, rename, base, work)
        if st["missing"] or st["different_bytes"]:
            raise RuntimeError(f"{d.parent.name}: {st['missing'][:5]} {st['different_bytes'][:5]}")
        dst = sw / d.parent.name / "best"
        if dst.parent.exists():
            shutil.rmtree(dst.parent)
        dst.mkdir(parents=True)
        sub.write_binary(str(dst))
        info["subs"][d.parent.name] = st
        names += sorted(im.name for im in sub.images.values())
    info["images"] = sorted(set(names))
    info["renamed"] = {k: v for k, v in sorted(rename.items())}
    m = json.loads((base / "cameras" / "sfm_work" / "map.json").read_text())
    m["reused_from"] = REF
    m["note"] = (f"static map copied from {REF} (dancecap.front_full prep-local): its 56-62 s window images became "
                 "keyframes of this segment take; match_s/mapping_s are larissa_kadu's")
    sw.mkdir(parents=True, exist_ok=True)
    (sw / "map.json").write_text(json.dumps(m, indent=1, default=str))
    (sw / MAP_IMAGES).write_text(json.dumps(info, indent=1))
    # 4. world frame + scale: COPIED (same SfM frame by construction)
    shutil.copyfile(base / "cameras" / "world_from_sfm.json", cams_dir / "world_from_sfm.json")
    (cams_dir / "world_copied_from.json").write_text(json.dumps({
        "from": f"{REF}/cameras/world_from_sfm.json", "sha1": _sha1(base / "cameras" / "world_from_sfm.json"),
        "why": "the SfM frame is identical (reused map, renamed images only): gravity, floor and the betas-corrected "
               "scale of larissa_kadu_38 carry over exactly; the cameras stage runs attach,qa only (no world re-solve)",
        "copied": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    rep = {"map": m, "world": {"copied_from": REF}}
    (cams_dir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    # model check: identical poses / points, just renamed
    a, b = pycolmap.Reconstruction(str(base / "cameras" / "sfm")), pycolmap.Reconstruction(str(cams_dir / "sfm"))
    pa = np.array(sorted(tuple(p.xyz) for p in a.points3D.values()))
    pb = np.array(sorted(tuple(p.xyz) for p in b.points3D.values()))
    assert pa.shape == pb.shape and float(np.abs(pa - pb).max()) == 0.0
    print(f"[front_full] {take.name}: map reused ({info['main']['images']} main images, "
          f"{sum(len(v) for v in added.values())} foreign window images -> keyframes, subs "
          f"{ {k: v['images'] for k, v in info['subs'].items()} }), world copied")
    return info


def source_facts(take: Take) -> dict:
    """{cam: {w, h, fps, t_first, t_last}} of every source (ffprobe; reference clock)."""
    from .ingest import probe
    out = {}
    for s in take.sources:
        info = probe(s.path)
        d = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=duration,start_time", "-of", "json", str(s.path)], capture_output=True, text=True,
                           check=True).stdout
        st = json.loads(d)["streams"][0]
        t0, dur = float(st.get("start_time") or 0), float(st["duration"])
        out[s.id] = {"w": info["width"], "h": info["height"], "fps": info["avg_fps"],
                     "t_first": s.to_reference(t0), "t_last": s.to_reference(t0 + dur)}
    return out


def plan_frames() -> dict:
    """Expected window frames per segment and camera (memory estimates of the driver; written to front/plan.json)."""
    facts = source_facts(Take.load(FULL))
    plan = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": facts, "segments": {}}
    for sid in SEG_IDS:
        t = Take.load(seg_name(sid))
        seg = {}
        for s in t.sources:
            f = facts[s.id]
            lo, hi = max(t.window[0], f["t_first"]), min(t.window[1], f["t_last"])
            every = max(1, int(round(f["fps"] / MAX_FPS))) if f["fps"] > 1.5 * MAX_FPS else 1
            n = max(0, int((hi - lo) * f["fps"] / every))
            seg[s.id] = {"frames": n, "w": f["w"], "h": f["h"], "every": every, "span": [round(lo, 3), round(hi, 3)]}
        plan["segments"][sid] = {"take": t.name, "window": list(t.window), "cams": seg,
                                 "frames": sum(v["frames"] for v in seg.values())}
    plan["frames_total"] = sum(v["frames"] for v in plan["segments"].values())
    p = full_work() / "front" / "plan.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan, indent=1))
    print(f"[front_full] plan: {plan['frames_total']} frames " +
          ", ".join(f"{k} {v['frames']}" for k, v in plan["segments"].items()) + f" -> {p}")
    return plan


def decode_refs() -> list[str]:
    """larissa_kadu_38 frames used by decode_check: 3 per camera inside segment a's front span (6 for 60 fps cam02),
    workspace-relative paths (uploaded with the job)."""
    wa = Take.load(seg_name("a")).window
    out = []
    for s in Take.load(REF).sources:
        meta = json.loads((ref_work() / "frames" / s.id / "frames.json").read_text())
        idx = [i for i, r in enumerate(meta["frames"]) if wa[0] + 0.2 <= r["t"] <= wa[1] - 0.2]
        if not idx:
            continue
        k = 6 if meta.get("fps_median", 30) > 45 else 3
        pick = [idx[int(round(j * (len(idx) - 1) / (k - 1)))] for j in range(k)]
        out += [f"dancecap/work/{REF}/frames/{s.id}/{meta['frames'][i]['file']}" for i in pick]
        out.append(f"dancecap/work/{REF}/frames/{s.id}/frames.json")
    return out


# ------------------------------------------------------------------------------------------------ pod workers


def detect_one(take_name: str, cam: str) -> None:
    """YOLO detection of one camera's window frames (no keyframes: their masks are larissa_kadu_38's) ->
    people/<cam>.npz, people_masks/<cam>/, people/qa/<cam>.jpg, people/report_<cam>.json (merged later: several
    cameras run side by side, so detect_take's shared report.json would race)."""
    from .detect import DET_WEIGHTS, POSE_WEIGHTS, SEG_WEIGHTS, Detector, Settings, detect_camera
    take = Take.load(take_name)
    s = Settings()
    t0 = time.time()
    model = Detector(s)
    r = detect_camera(take, cam, model, keyframes=False)
    rep = {"settings": s.__dict__, "models": {"det": DET_WEIGHTS.name, "seg": SEG_WEIGHTS.name, "pose": POSE_WEIGHTS.name,
                                              "licence": "AGPL-3.0 (Ultralytics)"}, "cams": {cam: r},
           "runtime_s": round(time.time() - t0, 1), "host": "pod" if os.environ.get("DANCECAP_POD") else "local"}
    (take.work / "people" / f"report_{cam}.json").write_text(json.dumps(rep, indent=1))
    print(f"[detect] {cam}: {r['frames']} frames, {r['people_per_frame']['mean']:.1f} people/frame, pair "
          f"{100 * r['dancer_pair_rate']:.1f}%, {r['runtime_s']['per_frame_ms']:.0f} ms/frame", flush=True)


def people_report(work: Path, cams: list[str]) -> None:
    rep: dict = {"cams": {}}
    for c in cams:
        p = work / "people" / f"report_{c}.json"
        if p.exists():
            d = json.loads(p.read_text())
            rep.update({k: v for k, v in d.items() if k not in ("cams", "runtime_s")})
            rep["cams"][c] = d["cams"][c]
    (work / "people" / "report.json").write_text(json.dumps(rep, indent=1))


def decode_check() -> dict:
    """Pod-decoded segment-a frames vs larissa_kadu_38's locally decoded frames of the same native PTS."""
    import cv2
    import numpy as np
    wa = Take.load(seg_name("a")).work
    out = {}
    for fj in sorted((ref_work() / "frames").glob("*/frames.json")):
        c = fj.parent.name
        mine_p = wa / "frames" / c / "frames.json"
        if not mine_p.exists():
            continue
        ref = json.loads(fj.read_text())["frames"]
        mine = json.loads(mine_p.read_text())
        mine_pts = np.array([r["pts"] for r in mine["frames"]])
        rows = []
        for r in ref:
            p_ref = fj.parent / r["file"]
            if not p_ref.exists():
                continue
            # nearest PTS within 2 ms: the pod's apt ffmpeg 4.4 prints showinfo pts_time to 6 significant digits
            j = int(np.argmin(np.abs(mine_pts - r["pts"])))
            if abs(mine_pts[j] - r["pts"]) > 0.002:
                continue
            m = mine["frames"][j]
            a = cv2.imread(str(p_ref)).astype(np.int16)
            b = cv2.imread(str(wa / "frames" / c / m["file"])).astype(np.int16)
            if a.shape != b.shape:
                rows.append({"pts": r["pts"], "shape_ref": a.shape, "shape_pod": b.shape, "ok": False})
                continue
            d = np.abs(a - b)
            rows.append({"pts": r["pts"], "pts_pod": m["pts"], "t_ref": r["t"], "t_pod": m["t"],
                         "mean_abs": round(float(d.mean()), 3), "p99_abs": float(np.percentile(d, 99)),
                         "ok": bool(d.mean() < 3.0 and abs(r["t"] - m["t"]) < 0.002)})
        out[c] = {"compared": len(rows), "ok": bool(rows) and all(x["ok"] for x in rows), "frames": rows,
                  "decimate_every": mine.get("decimate_every", 1), "size": [mine["width"], mine["height"]]}
    v = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True).stdout.splitlines()
    doc = {"ffmpeg": v[0] if v else None, "cams": out, "all_ok": all(x["ok"] for x in out.values()) if out else False}
    p = full_work() / "front" / "decode_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))
    print(f"[front_full] decode check: {'ok' if doc['all_ok'] else 'DIFFERENCES'} " +
          ", ".join(f"{c} {x['compared']}x {'ok' if x['ok'] else 'BAD'}" for c, x in out.items()), flush=True)
    return doc


def fetch_weights() -> None:
    """SAM 2.1 (Meta CDN) + the three YOLO weights (ultralytics GitHub release v8.3.0), sha256-checked."""
    import urllib.request
    fp.fetch_sam2()
    for name, (rel, sha) in YOLO.items():
        dst = WORKSPACE / rel
        if dst.exists() and fp._sha256(dst) == sha:
            print(f"[front_full] {name} present (sha256 ok)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        t0 = time.time()
        urllib.request.urlretrieve(YOLO_URL.format(name), tmp)
        h = fp._sha256(tmp)
        if h != sha:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"[front_full] {name}: sha256 {h} != {sha}: refusing it")
        os.replace(tmp, dst)
        print(f"[front_full] {name}: {dst.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f} s, sha256 ok")


def check_env() -> None:
    import importlib
    errs = []
    for m in ("torch", "cv2", "numpy", "scipy", "pycolmap", "lightglue", "sam2", "ultralytics", "dancecap.dancers",
              "dancecap.detect", "dancecap.cameras.pipeline", "dancecap.sync", "dancecap.mamma_dyn"):
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            errs.append(f"import {m}: {type(e).__name__}: {e}")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            errs.append(f"{tool} not on PATH")
    from . import mamma_dyn as md
    r = subprocess.run([md.python(), "run_ma_2d.py", "--help"], cwd=md.MAMMA_ROOT / "landmarks", env=md._env(),
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        errs.append("run_ma_2d.py --help: " + (r.stderr or r.stdout)[-800:])
    from .dancers_sam import CHECKPOINT
    need = [md.MA2D_WEIGHTS, md.VERTS_512, CHECKPOINT, *[WORKSPACE / rel for rel, _ in YOLO.values()]]
    for sid in SEG_IDS:
        t = Take.load(seg_name(sid))
        need += [t.work / "cameras" / "sfm" / "images.bin", t.work / "cameras" / "world_from_sfm.json"]
        need += [s.path for s in t.sources]
    need.append(ref_work() / "sync" / "offsets.json")
    for p in need:
        if not Path(p).exists():
            errs.append(f"missing {p}")
    import torch
    v = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True).stdout.splitlines()
    host = {"cpus": fp.cgroup_cpus(), "ram_gb": round(fp.mem_gb(), 1), "mem_state": fp.mem_state(),
            "gpu_free_mb": fp.gpu_free_mb(), "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ffmpeg": v[0] if v else None, "disk_free_gb": round(shutil.disk_usage("/workspace").free / 1e9, 1)
            if Path("/workspace").exists() else None}
    p = full_work() / "front" / "host.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**host, "errors": errs}, indent=1, default=str))
    print(f"[front_full] {json.dumps(host, default=str)}", flush=True)
    if errs:
        raise SystemExit("[front_full] environment check FAILED:\n  " + "\n  ".join(errs))
    print("[front_full] environment check ok", flush=True)


# ------------------------------------------------------------------------------------------------ P2 / C7 on the pod

# P2 (PLAN.md 3.1) moved into S1 (user, 2026-10-08 "Prioritize the runpod launch"): instead of a sparse local probe, the
# gate runs on the pod over the FULL per-frame localisation of every camera and segment (a superset of the probe:
# sequential tracking + people masks included). Pass = PLAN.md P2's rule: >= 90 % of frames ok and no failed stretch
# over 8 s. A failing camera triggers the conditional map extension (C7) when `dancecap.map_extend` is present on the
# pod; otherwise the task waits up to C7_WAIT_MIN for it (pushed to the pod during the session) or for
# work/larissa_kadu_full/front/c7_skip, then lets the segment continue: failed frames keep ok=False, so prepare drops
# that camera's landmarks there (mamma_dyn resample: landmarks/visibilities zeroed where the camera is not ok).
P2_MIN_OK, P2_MAX_GAP_S, P2_BLOCK_S = 0.90, 8.0, 10.0
C7_WAIT_MIN = 40.0


def loc_gate_stats(t: "np.ndarray", ok: "np.ndarray", n_inl: "np.ndarray | None" = None,
                   rp: "np.ndarray | None" = None, block_s: float = P2_BLOCK_S) -> dict:
    """ok fraction, longest failed stretch (s) and per-block ok fractions of one camera's localisation."""
    import numpy as np
    t, ok = np.asarray(t, float), np.asarray(ok, bool)
    if len(t) == 0:
        return {"frames": 0, "ok_frac": 0.0, "max_gap_s": float("inf"), "blocks": [], "pass": False}
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    gap, run_start = 0.0, None
    for i in range(len(t)):
        if not ok[i]:
            run_start = t[i] if run_start is None else run_start
            gap = max(gap, t[i] - run_start + dt)
        else:
            run_start = None
    blocks = []
    b0 = np.floor(t[0] / block_s) * block_s
    while b0 < t[-1] + 1e-9:
        m = (t >= b0) & (t < b0 + block_s)
        if m.any():
            row = {"t0": round(float(b0), 1), "t1": round(float(b0 + block_s), 1), "frames": int(m.sum()),
                   "ok_frac": round(float(ok[m].mean()), 3)}
            if n_inl is not None and (m & ok).any():
                row["inliers_med"] = int(np.median(np.asarray(n_inl)[m & ok]))
            if rp is not None and (m & ok).any():
                row["reproj_med_px"] = round(float(np.nanmedian(np.asarray(rp, float)[m & ok])), 2)
            blocks.append(row)
        b0 += block_s
    out = {"frames": int(len(t)), "ok_frac": round(float(ok.mean()), 4), "max_gap_s": round(float(gap), 2),
           "blocks": blocks}
    if n_inl is not None:
        n = np.asarray(n_inl)
        strong = ok & (n >= 120) & (np.nan_to_num(np.asarray(rp, float), nan=99) < 2.0 if rp is not None else True)
        out["strong_frac"] = round(float(strong.mean()), 4)
    out["pass"] = bool(out["ok_frac"] >= P2_MIN_OK and out["max_gap_s"] <= P2_MAX_GAP_S)
    return out


def _loc_arrays(work: Path, cam: str):
    """(t, ok, n_inliers, reproj_px, source_kind) of a camera's raw localisation (main model or sub-model)."""
    import pickle

    import numpy as np
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    t = np.array([r["t"] for r in meta["frames"]], float)
    p = work / "cameras" / "cache" / f"loc_{cam}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True)
        n = len(d["ok"])
        return (np.asarray(d["t"], float) if "t" in d.files else t[:n], np.asarray(d["ok"], bool),
                np.asarray(d["n_inliers"]) if "n_inliers" in d.files else None,
                np.asarray(d["reproj_px"], float) if "reproj_px" in d.files else None, "main")
    p = work / "cameras" / "cache" / f"locsub_{cam}.pkl"
    if p.exists():
        poses = pickle.loads(p.read_bytes())["value"]
        return (t[:len(poses)], np.array([q.ok for q in poses], bool), np.array([q.n_inliers for q in poses]),
                np.array([q.reproj_px for q in poses], float), "sub")
    return None


def loc_gate(take_name: str, cam: str, wait_min: float | None = None) -> dict:
    """P2 gate of one camera of one segment take (+ the C7 hook). Writes cameras/loc_gate_<cam>.json."""
    take = Take.load(take_name)
    work = take.work
    arr = _loc_arrays(work, cam)
    if arr is None:
        g = {"cam": cam, "take": take_name, "pass": False, "why": "no localisation output"}
    else:
        t, ok, n, rp, kind = arr
        g = {"cam": cam, "take": take_name, "model": kind, **loc_gate_stats(t, ok, n, rp)}
    g["rule"] = f"ok_frac >= {P2_MIN_OK} and max_gap_s <= {P2_MAX_GAP_S} (PLAN.md P2, on every frame)"
    if not g["pass"] and arr is not None:
        wait_min = C7_WAIT_MIN if wait_min is None else wait_min
        ext = PACKAGE_ROOT / "dancecap" / "map_extend.py"
        skip = full_work() / "front" / "c7_skip"
        deadline = time.time() + 60 * wait_min
        print(f"[p2] {take_name} {cam}: FAILS ({g.get('ok_frac')} ok, gap {g.get('max_gap_s')} s); C7 "
              f"{'present' if ext.exists() else f'not present: waiting up to {wait_min:.0f} min'}", flush=True)
        while not ext.exists() and not skip.exists() and time.time() < deadline:
            time.sleep(30)
        if ext.exists():
            r = subprocess.run([sys.executable, "-m", "dancecap.map_extend", take_name, cam], cwd=str(PACKAGE_ROOT))
            g["c7"] = {"rc": r.returncode}
            arr2 = _loc_arrays(work, cam)
            if r.returncode == 0 and arr2 is not None:
                g["c7"]["after"] = loc_gate_stats(*arr2[:4])
                g["pass_after_c7"] = g["c7"]["after"]["pass"]
        else:
            g["c7"] = {"rc": None, "why": "c7_skip" if skip.exists() else f"map_extend not delivered in {wait_min} min"}
    p = work / "cameras" / f"loc_gate_{cam}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(g, indent=1))
    print(f"[p2] {take_name} {cam}: {'PASS' if g['pass'] else 'FAIL'} ok {g.get('ok_frac')} gap {g.get('max_gap_s')} s"
          + (f" strong {g.get('strong_frac')}" if "strong_frac" in g else "") + " blocks " +
          " ".join(f"{b['t0']:.0f}:{b['ok_frac']:.2f}" for b in g.get("blocks", [])), flush=True)
    return g


def p2_summary(segs: list[str]) -> dict:
    out = {}
    for sid in segs:
        work = Take.load(seg_name(sid)).work
        for p in sorted((work / "cameras").glob("loc_gate_*.json")):
            g = json.loads(p.read_text())
            out[f"{sid}:{g['cam']}"] = {k: g.get(k) for k in ("model", "frames", "ok_frac", "strong_frac", "max_gap_s",
                                                               "pass", "pass_after_c7", "c7")}
    doc = {"rule": f"ok_frac >= {P2_MIN_OK}, max failed stretch <= {P2_MAX_GAP_S} s", "cams": out,
           "failing": sorted(k for k, v in out.items() if not v["pass"] and not v.get("pass_after_c7"))}
    p = full_work() / "front" / "p2_loc_gate.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))
    print(f"[p2] summary: {len(out)} camera-segments, failing {doc['failing'] or 'none'}", flush=True)
    return doc


# ------------------------------------------------------------------------------------------------ the driver


def _frames_done(work: Path, cam: str) -> bool:
    fj = work / "frames" / cam / "frames.json"
    if not fj.exists():
        return False
    n = len(json.loads(fj.read_text())["frames"])
    return n > 0 and len(list((work / "frames" / cam).glob("*.jpg"))) == n


def _sam_mem(n: int, w: int, h: int) -> float:
    """front_pod._sam_mem_gb from the planned frame count (frames.json does not exist before the pod ingests)."""
    px = w * h
    work_px = min(1.0, 1280 / max(w, h)) ** 2 * px
    return round((n * (3 * px + 2 * px + 2 * work_px + 960 * 540 * 3)) / 2**30 * 1.1 + 3.0, 1)


def build_tasks(R: fp.Runner, plan: dict, segs: list[str], cpus: int) -> dict:
    py = sys.executable
    T = fp.Task
    per = {}
    map_main = set(json.loads((Take.load(seg_name(segs[0])).work / "cameras" / "sfm_work" / "map.json")
                              .read_text()).get("main_cams") or [])
    sync_workers = str(max(4, min(16, cpus // 2)))
    for sid in segs:
        take = seg_name(sid)
        work = Take.load(take).work
        cams = sorted(plan["segments"][sid]["cams"], key=lambda c: -plan["segments"][sid]["cams"][c]["frames"]
                      * plan["segments"][sid]["cams"][c]["w"] * plan["segments"][sid]["cams"][c]["h"])
        per[sid] = {"take": take, "work": work, "cams": cams, "main": [c for c in cams if c in map_main],
                    "subs": [c for c in cams if c not in map_main]}
    order = [(sid, c) for sid in segs for c in per[sid]["cams"]]
    size = lambda sid, c: plan["segments"][sid]["cams"][c]  # noqa: E731
    by_size = sorted(order, key=lambda x: -size(*x)["frames"] * size(*x)["w"] * size(*x)["h"])
    # 1. ingest + detect (everything else waits for them; ingest is ffmpeg + JPEG encoding on the CPU)
    for sid, c in by_size:
        P = per[sid]
        R.add(T(f"{sid}.ingest:{c}", [py, "-m", "dancecap", "ingest", P["take"], "--cams", c, "--window-only"],
                kind="ingest", mem_gb=2.0, timeout_min=30, done_if=lambda w=P["work"], c=c: _frames_done(w, c)))
    for sid, c in by_size:
        P = per[sid]
        R.add(T(f"{sid}.detect:{c}", [py, "-m", "dancecap.front_full", "detect-one", P["take"], c],
                deps=[f"{sid}.ingest:{c}"], kind="detect", mem_gb=6.0, gpu=True, timeout_min=45,
                done_if=lambda w=P["work"], c=c: (w / "people" / f"report_{c}.json").exists()
                and (w / "people" / f"{c}.npz").exists()))
    R.add(T("decode_check", None, after=[f"a.ingest:{c}" for c in per["a"]["cams"]] if "a" in per else [],
            kind="cpu", optional=True, retries=0, fn=decode_check))
    # 2. SAM (RAM-bound) and localisation (CPU-bound) side by side, largest first across segments
    for sid, c in by_size:
        P = per[sid]
        s = size(sid, c)
        R.add(T(f"{sid}.sam:{c}", [py, "-m", "dancecap.front_pod", "dancer", P["take"], c],
                deps=[f"{sid}.detect:{c}"], kind="sam", mem_gb=_sam_mem(s["frames"], s["w"], s["h"]), gpu=True,
                timeout_min=150, on_ok=lambda w=P["work"], c=c: fp.merge_label(w, c),
                done_if=lambda w=P["work"], c=c: fp._dancers_done(w, c)))
    cache = lambda sid: per[sid]["work"] / "cameras" / "cache"  # noqa: E731
    for sid, c in sorted(order, key=lambda x: -size(*x)["frames"]):
        P = per[sid]
        if c in P["main"]:
            n = size(sid, c)["frames"]
            done = lambda w=P["work"], c=c: (w / "cameras" / "cache" / f"loc_{c}.npz").exists() \
                and (w / "cameras" / f"{c}.npz").exists()  # noqa: E731
            parts = 2 if n > 500 else 1
            if parts == 1:
                R.add(T(f"{sid}.loc:{c}", [py, "-m", "dancecap", "cameras", P["take"], "--stages", "loc", "--cams", c],
                        deps=[f"{sid}.detect:{c}"], kind="loc", mem_gb=4.0, gpu=True, timeout_min=150, done_if=done))
                continue
            # bounds from frames.json at start time (the planned count is approximate): a callable command
            names = []
            for k in range(parts):
                nm = f"{sid}.loc:{c}:{k}"
                names.append(nm)

                def cmd(take=P["take"], w=P["work"], c=c, k=k, parts=parts):
                    nf = fp._n_frames(w, c)
                    b = [round(j * nf / parts) for j in range(parts + 1)]
                    return [py, "-m", "dancecap.front_pod", "loc-part", take, c, "--range", f"{b[k]}:{b[k + 1]}"]

                def pdone(w=P["work"], c=c, k=k, parts=parts, done=done):
                    if done():
                        return True
                    if not (w / "frames" / c / "frames.json").exists():
                        return False
                    nf = fp._n_frames(w, c)
                    b = [round(j * nf / parts) for j in range(parts + 1)]
                    return (w / "cameras" / "cache" / f"locpart_{c}_{b[k]:05d}_{b[k + 1]:05d}.npz").exists()
                R.add(T(nm, cmd, deps=[f"{sid}.detect:{c}"], kind="loc", mem_gb=4.0, gpu=True, timeout_min=150,
                        done_if=pdone))

            def jcmd(take=P["take"], w=P["work"], c=c, parts=parts):
                nf = fp._n_frames(w, c)
                b = [round(j * nf / parts) for j in range(parts + 1)]
                return [py, "-m", "dancecap.front_pod", "loc-join", take, c, "--parts",
                        ",".join(f"{x}:{y}" for x, y in zip(b, b[1:]))]
            R.add(T(f"{sid}.loc:{c}", jcmd, deps=names, kind="cpu", mem_gb=3.0, timeout_min=20, done_if=done))
        else:
            R.add(T(f"{sid}.sub:{c}", [py, "-m", "dancecap.cameras.reuse", "presolve-sub", P["take"], "--cams", c],
                    deps=[f"{sid}.detect:{c}"], kind="loc", mem_gb=4.0, gpu=True, timeout_min=150, optional=True,
                    done_if=lambda sid=sid, c=c: (cache(sid) / f"locsub_{c}.pkl").exists()))
    # P2 gate (+ conditional C7 map extension) of every camera of every segment, as soon as its localisation lands
    for sid, c in order:
        P = per[sid]
        src = f"{sid}.loc:{c}" if c in P["main"] else f"{sid}.sub:{c}"
        R.add(T(f"{sid}.p2:{c}", [py, "-m", "dancecap.front_full", "locgate", P["take"], c], after=[src], kind="gate",
                mem_gb=0.5, optional=True, retries=0, timeout_min=C7_WAIT_MIN + 60,
                done_if=lambda w=P["work"], c=c: (w / "cameras" / f"loc_gate_{c}.json").exists()))
    R.add(T("p2_summary", None, after=[f"{sid}.p2:{c}" for sid, c in order], kind="cpu", optional=True, retries=0,
            fn=lambda segs=segs: p2_summary(segs)))
    for sid, c in by_size:
        P = per[sid]
        R.add(T(f"{sid}.ma2d:{c}", [py, "-m", "dancecap.front_pod", "ma2d", P["take"], c], deps=[f"{sid}.sam:{c}"],
                kind="ma2d", mem_gb=8.0, gpu=True, timeout_min=60,
                done_if=lambda w=P["work"], c=c: (w / "mamma" / "ma_2d_native" / f"{c}.manifest.json").exists()))
    # 3. per segment: merge -> cameras (attach, qa; world copied) -> sync (gauge pinned to 38) -> checks -> prepare
    for sid in segs:
        P = per[sid]
        take, work, cams = P["take"], P["work"], P["cams"]
        R.add(T(f"{sid}.labels_review", None, after=[f"{sid}.sam:{c}" for c in cams], kind="cpu", optional=True,
                retries=0, fn=lambda take=take: fp.labels_review(take)))
        R.add(T(f"{sid}.people_report", None, after=[f"{sid}.detect:{c}" for c in cams], kind="cpu", optional=True,
                retries=0, fn=lambda work=work, cams=cams: people_report(work, cams)))
        R.add(T(f"{sid}.merge", [py, "-m", "dancecap.dancers_pod", "merge", take, "--cams", ",".join(cams)],
                after=[f"{sid}.sam:{c}" for c in cams], kind="cpu", mem_gb=8.0, timeout_min=20, retries=0))
        loc_after = [f"{sid}.loc:{c}" for c in P["main"]] + [f"{sid}.sub:{c}" for c in P["subs"]] +             [f"{sid}.p2:{c}" for c in cams]
        R.add(T(f"{sid}.cameras", [py, "-m", "dancecap", "cameras", take, "--stages", "attach,qa"],
                deps=[f"{sid}.merge"], after=loc_after, kind="cpu", mem_gb=16.0, timeout_min=90, retries=1))

        def ready(work=work, cams=cams) -> list[str]:
            ok = [c for c in cams if (work / "cameras" / f"{c}.npz").exists()
                  and (work / "mamma" / "ma_2d_native" / f"{c}.npz").exists()]
            if len(ok) < 2:
                raise RuntimeError(f"only {ok} have camera tracks + ma_2d landmarks")
            return ok
        R.add(T(f"{sid}.sync", lambda take=take, ready=ready: [
            py, "-m", "dancecap", "sync", take, "--cams", ",".join(ready()), "--gauge-ref", REF,
            "--workers", sync_workers], deps=[f"{sid}.cameras"], after=[f"{sid}.ma2d:{c}" for c in cams], kind="cpu",
            mem_gb=12.0, timeout_min=90, retries=0))
        R.add(T(f"{sid}.chain", [py, "-m", "dancecap.cameras.world_scale", "chain", take, "--out",
                                 str(work / "scale" / "chain_after_sync.json")], deps=[f"{sid}.sync"], kind="cpu",
                mem_gb=4.0, optional=True, timeout_min=20, retries=0))
        R.add(T(f"{sid}.geocheck", [py, "-m", "dancecap.dancers_geoveto", take, "--check"], deps=[f"{sid}.sync"],
                kind="cpu", mem_gb=6.0, optional=True, timeout_min=30, retries=0))
        k0, k1 = (int(x) for x in seg_cfg(sid)["segment"]["fit_k"])
        R.add(T(f"{sid}.prepare", lambda take=take, ready=ready, k0=k0, k1=k1: [
            py, "-m", "dancecap", "mamma", "prepare", take, "--skip-2d", "--set", f"c{len(ready())}={','.join(ready())}",
            "--coverage", "union", "--grid-anchor", repr(GRID_ANCHOR), f"--grid-k={k0}:{k1}"],
            deps=[f"{sid}.sync"], kind="cpu", mem_gb=8.0, timeout_min=30, retries=0))
    return per


def run(resume: bool = False, segs: list[str] | None = None, sam_workers: int | None = None) -> int:
    segs = segs or list(SEG_IDS)
    work = full_work()
    plan = json.loads((work / "front" / "plan.json").read_text())
    cpus, ram = fp.cgroup_cpus(), fp.mem_gb()
    ws0, limit = fp.mem_state()
    threads = str(max(2, min(4, cpus // 8)))
    env = dict(os.environ, DANCECAP_GPU_SHARED="1", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               OPENBLAS_NUM_THREADS=threads, OPENCV_FOR_THREADS_NUM=threads, PYTHONUNBUFFERED="1",
               DANCECAP_POD=os.environ.get("DANCECAP_POD", "1"), YOLO_CONFIG_DIR="/tmp/Ultralytics")
    torch_home = WORKSPACE / fp.TORCH_HOME_REL
    if torch_home.exists():
        env.setdefault("TORCH_HOME", str(torch_home))
    cap_gb = limit if limit else ram
    mem_budget = max(16.0, 0.68 * cap_gb)
    n_sam = sam_workers or max(2, min(5, cpus // 6))
    caps = {"ingest": max(2, min(4, cpus // 7)), "detect": 3 if cpus >= 20 else 2, "sam": n_sam,
            "loc": max(2, min(9, cpus - n_sam - 4)), "ma2d": 2, "cpu": 3, "gate": 64}
    R = fp.Runner(work, env, mem_budget, caps, work / "front" / "logs", limit)
    per = build_tasks(R, plan, segs, cpus)
    if resume:
        R.resume()
    else:
        R.memlog.unlink(missing_ok=True)
        for f in R.rc_dir.glob("*.rc"):
            f.unlink()
    R.meta = {"take": FULL, "segments": {s: {k: v for k, v in per[s].items() if k != "work"} for s in segs},
              "host": {"cpus": cpus, "ram_gb": round(ram, 1), "mem_limit_gb": limit, "working_set_at_start_gb": ws0,
                       "mem_budget_gb": round(mem_budget, 1), "threads_per_proc": threads, "caps": caps,
                       "gpu_free_mb_at_start": fp.gpu_free_mb()},
              "resumed": resume, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"[front] {FULL} segments {segs}: {len(R.tasks)} tasks; cpus {cpus}, RAM limit {limit} GB (budget "
          f"{mem_budget:.0f}), caps {caps}{', RESUMED' if resume else ''}", flush=True)
    R.save()
    R.loop()
    R.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    R.save()
    bad = [k for k, t in R.tasks.items() if t.state != "ok" and not t.optional]
    prepared = [s for s in segs if R.tasks[f"{s}.prepare"].state == "ok"]
    print(f"[front] done; prepared {prepared}; not ok: {bad or 'none'}; working set peak {R.ws_peak:.0f} GB", flush=True)
    (work / "front" / "front_done.json").write_text(json.dumps({"rc": 0 if len(prepared) == len(segs) else 1,
                                                                 "prepared": prepared, "not_ok": bad,
                                                                 "finished": R.meta["finished"]}))
    return 0 if len(prepared) == len(segs) else 1


def _kill_minutes_left() -> float | None:
    try:
        return (float(Path("/workspace/.dancecap_kill_at").read_text().split()[0]) - time.time()) / 60.0
    except (OSError, ValueError, IndexError):
        return None


def _stop_driver(pid: int | None) -> None:
    import signal
    if pid and fp._alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, AttributeError):
            pass


def supervise(attempts: int = 4, reserve_min: float = 20.0) -> int:
    """The pod job's command (front_pod.supervise for the multi-take driver): restarts a dead driver with --resume.
    reserve_min: stop (driver and its workers) when the on-pod kill switch is that close, so pack + download of the
    partial outputs still fit (rc 2)."""
    front = full_work() / "front"
    front.mkdir(parents=True, exist_ok=True)
    done, pidf, log = front / "front_done.json", front / "driver.pid", front / "driver.log"
    (front / "host_start.json").write_text(json.dumps(fp._cgroup_report(), indent=1, default=str))
    pos = log.stat().st_size if log.exists() else 0
    for k in range(attempts):
        if done.exists():
            return int(json.loads(done.read_text())["rc"])
        pid = int(pidf.read_text()) if pidf.exists() and pidf.read_text().strip().isdigit() else None
        proc = None
        if pid and fp._alive(pid) and "front_full run" in fp._cmdline(pid):
            print(f"[supervise] following the running driver (pid {pid})", flush=True)
        else:
            with open(log, "a", encoding="utf-8") as fh:
                proc = subprocess.Popen([sys.executable, "-m", "dancecap.front_full", "run", "--resume"], stdout=fh,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(PACKAGE_ROOT),
                                        start_new_session=True)
            pid = proc.pid
            pidf.write_text(str(pid))
            print(f"[supervise] attempt {k + 1}/{attempts}: driver pid {pid}", flush=True)
        snaps: list[str] = []
        last_snap = 0.0
        while (proc.poll() is None) if proc is not None else fp._alive(pid):
            left = _kill_minutes_left()
            if left is not None and left < reserve_min:
                print(f"[supervise] kill switch in {left:.0f} min < reserve {reserve_min:.0f} min: stopping the driver "
                      "so the partial outputs can be packed", flush=True)
                _stop_driver(pid)
                return 2
            if log.exists() and log.stat().st_size > pos:
                with open(log, encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                for line in chunk.splitlines():
                    if line.startswith(("[front]", "[supervise]", "Traceback")) or "Error" in line:
                        print(line, flush=True)
            if time.time() - last_snap > 30:
                try:
                    ps = subprocess.run(["ps", "-eo", "pid,ppid,pgid,sid,stat,rss,etimes,args", "--sort=-rss"],
                                        capture_output=True, text=True, timeout=20).stdout
                except (OSError, subprocess.SubprocessError):
                    ps = ""
                snaps = (snaps + [f"{time.strftime('%H:%M:%S')} ws {fp.mem_state()[0]}\n"
                                  + "\n".join(ps.splitlines()[:40])])[-3:]
                last_snap = time.time()
            time.sleep(5)
        status = proc.returncode if proc is not None else "unknown (not our child)"
        if done.exists():
            return int(json.loads(done.read_text())["rc"])
        (front / f"forensic_{k + 1}.json").write_text(json.dumps(
            {"attempt": k + 1, "driver_pid": pid, "driver_status": status, "time": time.strftime("%H:%M:%S"),
             "last_ps": snaps, "cgroup": fp._cgroup_report()}, indent=1, default=str))
        print(f"[supervise] driver pid {pid} ended without front_done.json (status {status}); restarting", flush=True)
        time.sleep(10)
    return 1


def hold(max_minutes: float = 50.0, reserve_min: float = 30.0) -> int:
    """Pod job command after `front`: when the driver ended with failures (front_done.json rc != 0, no driver alive),
    keep the pod for up to max_minutes so a fix can be pushed and the driver restarted by hand
    (`nohup python -m dancecap.front_full supervise &` after removing front_done.json); returns as soon as
    front/release exists, a restarted driver is running (the next command, `front_2`, follows it), front_done.json
    says rc 0, or the kill switch is within reserve_min. Nothing to wait for when the front succeeded."""
    front = full_work() / "front"
    done, release, pidf = front / "front_done.json", front / "release", front / "driver.pid"
    t_end = time.time() + 60 * max_minutes
    said = False
    while time.time() < t_end:
        left = _kill_minutes_left()
        if release.exists() or (left is not None and left < reserve_min):
            return 0
        pid = int(pidf.read_text()) if pidf.exists() and pidf.read_text().strip().isdigit() else None
        running = bool(pid and fp._alive(pid) and "front_full run" in fp._cmdline(pid))
        rc = int(json.loads(done.read_text())["rc"]) if done.exists() else None
        if rc == 0 or (running and not done.exists()):
            print(f"[hold] {'front done' if rc == 0 else f'driver {pid} running'}: no hold", flush=True)
            return 0
        if not said:
            print(f"[hold] front rc={rc}, no driver running: holding the pod up to {max_minutes:.0f} min for a fix "
                  f"(touch {release} to release)", flush=True)
            said = True
        time.sleep(15)
    return 0


def status() -> int:
    p = full_work() / "front" / "front_pod.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    for k, v in (d.get("tasks") or {}).items():
        print(f"{k:<22} {v['state']:<8} rc={v['rc']} {v['seconds']} s {v.get('note', '')}")
    done = full_work() / "front" / "front_done.json"
    return int(json.loads(done.read_text())["rc"]) if done.exists() else 1


# ------------------------------------------------------------------------------------------------ fit-stage inputs (C6)


def pin_betas(out: Path | None = None) -> Path:
    """work/larissa_kadu_full/mamma_pin_betas.json = larissa_kadu_38's betas per MAMMA body id, for mamma-turing's
    env-gated pin (MAMMA_INIT_BETAS=<this file> MAMMA_FREEZE_BETAS=1): every segment fit then carries
    larissa_kadu_38's exact body shape, so its avatars (textures, faces, hair, shoes, shape edits) apply unchanged and
    the stitch needs no betas re-expression (PLAN.md C6).

    MAMMA's body ids are the 0-based person index of the fit (body_id-00 = lead = mamma_dyn.PIDS[0], body_id-01 =
    follow), NOT the dancer pids 1 / 2: the first version of this file keyed by pid, which would have pinned the
    FOLLOW body to the LEAD's betas (found 2026-10-08 in the fit prep; fit_full.preflight also checks the result)."""
    import numpy as np

    from .mamma_dyn import PIDS, ROLES
    doc = {}
    src = {}
    for body_id, pid in enumerate(PIDS):
        role = ROLES[pid]
        f = ref_work() / "smplx" / f"{role}.npz"
        doc[str(body_id)] = [float(x) for x in np.asarray(np.load(f)["betas"], float).reshape(-1)]
        src[str(body_id)] = {"role": role, "pid": pid, "file": f"{REF}/smplx/{role}.npz", "sha1": _sha1(f),
                             "n": len(doc[str(body_id)])}
    out = out or full_work() / "mamma_pin_betas.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    out.with_name(out.stem + ".source.json").write_text(json.dumps(src, indent=1))
    print(f"[front_full] pinned betas -> {out} ({ {k: v['role'] for k, v in src.items()} })")
    return out


# ------------------------------------------------------------------------------------------------ job + session


def _seg_outputs(sid: str, cams: list[str]) -> list[str]:
    w = f"dancecap/work/{seg_name(sid)}"
    out = [f"{w}/people/", f"{w}/people_masks/", f"{w}/dancers/", f"{w}/cameras/report.json", f"{w}/cameras/qa/",
           f"{w}/cameras/world_from_sfm.json", f"{w}/cameras/cache/vertical_main.pkl",
           f"{w}/cameras/cache/horizontal_main.pkl", f"{w}/cameras/cache/dancers_sfm.npz",
           f"{w}/mamma/ma_2d_native/", f"{w}/mamma/ma_2d/", f"{w}/mamma/ma_cap/", f"{w}/mamma/prepare.json",
           f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/pod_job.json", f"{w}/mamma/logs/", f"{w}/sync/",
           f"{w}/scale/", f"{w}/review/", f"{w}/probe/"]
    for c in cams:
        out += [f"{w}/frames/{c}/frames.json", f"{w}/cameras/{c}.npz", f"{w}/cameras/loc_gate_{c}.json",
                f"{w}/cameras/map_extend_{c}.json", f"{w}/cameras/cache/loc_{c}.pre_c7.npz",
                f"{w}/cameras/cache/locsub_{c}.pre_c7.pkl"]  # C7 report + the originals it replaced (revertible)
        out += [f"{w}/cameras/cache/{n}" for n in (f"loc_{c}.npz", f"attach_{c}.npz", f"locsub_{c}.pkl",
                                                    f"vertical_sub_{c}.pkl", f"vertical_check_{c}.pkl")]
    return out


def write_job(max_minutes: float = 330.0, front_minutes: float = 240.0) -> Path:
    from . import mamma_dyn as md
    plan = json.loads((full_work() / "front" / "plan.json").read_text())
    rel = lambda p: Path(p).resolve().relative_to(WORKSPACE).as_posix()  # noqa: E731
    full = Take.load(FULL)
    inputs: list = [rel(s.path) for s in full.sources]
    inputs.append(rel(PACKAGE_ROOT / "takes" / full.cfg["alignment"]))
    inputs += [f"dancecap/work/{FULL}/front/plan.json", f"dancecap/work/{REF}/sync/offsets.json", *decode_refs()]
    inputs += [{"local": fp.MA2D_STRIPPED, "pod": "mamma/data/weights/ma_2d/mamma_mask_full_cvpr.ckpt"},
               "mamma/data/body_models/downsampled_verts/verts_512.pkl",
               "mamma/data/body_models/downsampled_verts/smplx_512_body_parts.npz"]
    hub = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    for f in ("aliked-n16.pth", "aliked_lightglue_v0-1_arxiv.pth"):
        if (hub / f).exists():
            inputs.append({"local": str(hub / f), "pod": f"{fp.TORCH_HOME_REL}/hub/checkpoints/{f}"})
    outputs, verify, resumed = [f"dancecap/work/{FULL}/front/"], [f"dancecap/work/{FULL}/front/front_pod.json"], 0
    for sid in SEG_IDS:
        t = Take.load(seg_name(sid))
        w = f"dancecap/work/{t.name}"
        inputs += [f"{w}/keyframes/", f"{w}/people_masks_kf/", f"{w}/cameras/sfm/", f"{w}/cameras/sfm_work/",
                   f"{w}/cameras/cache/feats/", f"{w}/cameras/world_from_sfm.json", f"{w}/cameras/world_copied_from.json",
                   f"{w}/cameras/report.json"]
        cams = list(plan["segments"][sid]["cams"])
        # an earlier (interrupted) session's results let the driver pre-complete tasks (done_if); frames are re-decoded
        for c in cams:
            if (t.work / "people" / f"report_{c}.json").exists():
                inputs += [f"{w}/people/{c}.npz", f"{w}/people/report_{c}.json", f"{w}/people_masks/{c}/"]
                resumed += 1
        r = fp._resume_inputs(t.work, w, [c for c in cams if (t.work / "frames" / c / "frames.json").exists()])
        inputs += r
        resumed += len(r)
        outputs += _seg_outputs(sid, cams)
        verify += [f"{w}/dancers/roles.json", f"{w}/sync/offsets.json", f"{w}/mamma/prepare.json"]
    seen, uniq = set(), []
    for x in inputs:
        k = json.dumps(x)
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    cons = "/workspace/.dancecap_bootstrap/constraints.txt"
    whl = "/workspace/HeadMovement/dancecap/work/runpod/pywheels/cp311-linux"
    job = {
        "schema": "dancecap.pod_job/1", "name": "front_full", "take": FULL, "max_minutes": max_minutes,
        "needs": ["dancers", "cameras", "ma2d"],
        "inputs": uniq,
        "exclude": ["*.db", "*.db-shm", "*.db-wal"],
        "env": {"TORCH_HOME": f"/workspace/HeadMovement/{fp.TORCH_HOME_REL}", "PYOPENGL_PLATFORM": "egl",
                "YOLO_CONFIG_DIR": "/tmp/Ultralytics"},
        "setup": [
            "(apt-get install -y -qq --no-install-recommends ffmpeg > /dev/null 2>&1 || (apt-get update -qq && "
            "apt-get install -y -qq --no-install-recommends ffmpeg > /dev/null)) && ffmpeg -hide_banner -version | head -1",
            # bundled wheels first (pywheels/cp311-linux: ultralytics, opencv-python, polars, httpx closure; some hosts
            # stall on large PyPI pages - memory runpod-first-compute), PyPI only as the fallback
            f"(python -m pip install -q --no-index --find-links {whl} {ULTRALYTICS} opencv-python==4.11.0.86 numpy==1.26.4"
            f" || python -m pip install -q --find-links {whl} $( [ -f {cons} ] && echo -c {cons} ) {ULTRALYTICS} "
            "opencv-python==4.11.0.86 numpy==1.26.4) && python -c \"import ultralytics, cv2, torch; print('ultralytics', "
            "ultralytics.__version__, 'cv2', cv2.__version__, 'torch', torch.__version__)\"",
        ],
        "commands": [
            {"name": "weights", "cmd": "python -m dancecap.front_full fetch-weights", "timeout_min": 12},
            {"name": "check", "cmd": "python -m dancecap.front_full check", "timeout_min": 12},
            {"name": "front", "optional": True, "timeout_min": front_minutes,
             "cmd": "python -m dancecap.front_full supervise"},
            # an early failure keeps the pod for a pushed fix (hold), then front_2 follows the restarted driver;
            # supervise stops itself 20 min before the kill switch so pack + download always fit
            {"name": "hold", "optional": True, "timeout_min": 55, "cmd": "python -m dancecap.front_full hold"},
            {"name": "front_2", "optional": True, "timeout_min": front_minutes,
             "cmd": "python -m dancecap.front_full supervise"},
            {"name": "status", "cmd": "python -m dancecap.front_full status", "timeout_min": 2},
        ],
        "outputs": outputs,
        "verify": verify,
        "after_download": ["python -m dancecap.front_full_review all   (local: per-camera / per-10 s gates, identity vs "
                           "larissa_kadu_38 and across segments, review images)"],
        "notes": [f"larissa_kadu_full front (PLAN.md 3.2): {plan['frames_total']} planned frames over 4 segment takes, "
                  "one driver (dancecap.front_full); source MP4s uploaded and decoded on the pod (ffmpeg); YOLO and "
                  "SAM 2.1 weights fetched on the pod and sha256-checked; no hands, no faces",
                  "not downloaded: frames (JPEGs; frames.json only), cameras/cache/feats",
                  f"resumed from earlier sessions (uploaded, tasks pre-completed): {resumed} inputs"],
    }
    p = full_work() / "front" / "pod_job.json"
    p.write_text(json.dumps(job, indent=1))
    print(f"[front_full] -> {p} ({len(uniq)} inputs, {len(outputs)} outputs)")
    return p


def host_facts(pod, probe_mb: int = 16) -> dict:
    """cgroup vCPU quota, memory limit, disk and the upload rate from here (an 8 MB probe), before the bundle goes up."""
    out = pod.output(
        "q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo \"$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) "
        "$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)\"); echo CPU $q; "
        "m=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes); echo MEM $m; "
        "echo AVAIL $(awk '/MemAvailable/ {print $2}' /proc/meminfo); echo DISK $(df -B1 --output=avail /workspace | tail -1); "
        "echo NPROC $(nproc)", timeout=60)
    f: dict = {}
    for line in out.splitlines():
        k, _, v = line.partition(" ")
        f[k] = v.strip()
    try:
        q, per = f.get("CPU", "max 100000").split()[:2]
        cpus = (int(q) / int(per)) if q not in ("max", "-1") else float(f.get("NPROC", 0))
    except (ValueError, ZeroDivisionError):
        cpus = float(f.get("NPROC", 0) or 0)
    try:
        mem = int(f.get("MEM", "max")) / 2**30
        if mem > 4096:
            raise ValueError
    except ValueError:
        mem = int(f.get("AVAIL", 0) or 0) / 2**20
    disk = int(f.get("DISK", 0) or 0) / 1e9
    tmp = full_work() / "front" / "_upload_probe.bin"
    tmp.write_bytes(os.urandom(probe_mb * 2**20))
    t0 = time.time()
    try:
        pod.put(tmp, "/workspace/_upload_probe.bin", timeout=120)
        up = probe_mb * 2**20 / 1e6 / max(time.time() - t0, 1e-3)
    except Exception:  # noqa: BLE001
        up = 0.0
    tmp.unlink(missing_ok=True)
    return {"cpus": round(cpus, 1), "mem_gb": round(mem, 1), "disk_gb": round(disk, 1), "upload_mb_s": round(up, 2),
            "host": pod.host, "gpu": pod.gpu, "cost_per_hr": pod.cost_per_hr}


def balance_gate(max_cost: float) -> dict:
    from . import remote
    st = remote.status()
    bal = float((st.get("account") or {}).get("clientBalance") or 0.0)
    ok = max_cost <= bal - 2.0 and max_cost <= st["remaining"]
    return {"balance": round(bal, 2), "cap": st["cap"], "spent": st["spent"], "remaining": st["remaining"],
            "max_cost": max_cost, "ok": ok}


def session(max_minutes: float = 330.0, max_cost: float = 9.0, sid: str | None = None, attempts: int = 3,
            min_cpus: float = 20.0, min_mem_gb: float = 90.0, min_upload: float = 1.3, disk_gb: int = 160,
            gpus: list[str] | None = None) -> dict:
    """remote_jobs.run_session for the one front_full job, plus a host check BEFORE the upload: a pod whose cgroup quota
    is under min_cpus vCPUs or min_mem_gb GB, or whose upload rate from here is under min_upload MB/s, is deleted and
    another is started (up to `attempts`; the last one is kept whatever it is). 160 GB container disk: ~30k cached
    ALIKED features (2 MB each) + frames + masks."""
    from dataclasses import asdict

    from . import remote
    from . import remote_jobs as rj
    gate = balance_gate(max_cost)
    print(f"[front_full] balance gate: {gate}")
    if not gate["ok"]:
        raise SystemExit(f"[front_full] balance gate failed: {gate}")
    job = rj.load_job(full_work() / "front" / "pod_job.json")
    label = "front-full"
    ses = rj.Session(sid, label)
    ses.state.pop("error", None)
    ses.state.pop("bootstrap_rc", None)
    ses.state["balance_gate"] = gate
    ses.write_runner_info(label, [job.name])
    fpr = job.fingerprint()
    prev = ses.state["jobs"].get(job.name, {})
    if prev.get("status") == "ok" and prev.get("fingerprint") == fpr:
        ses.log(f"{job.name}: already done in this session")
        return rj._finish(ses, None)
    ses.state["jobs"][job.name] = {"status": "pending", "fingerprint": fpr, "spec": asdict(job)}
    ses.save()
    files = ses.timed("bundle_scan", lambda: rj.bundle_files([job]))
    bundle, man = ses.timed("bundle_build", lambda: rj.build_bundle(files, compresslevel=1))
    ses.state["bundle"] = {k: man[k] for k in ("key", "path", "bytes", "raw_bytes", "n_files", "sha256",
                                                "groups_raw_bytes", "links", "build_s")}
    ses.state["estimate_min"] = rj.estimate_minutes([job], man["bytes"])
    ses.state["max_minutes"] = max_minutes
    feats = rj.needed_features([job])
    ses.log(f"bundle {bundle.name}: {man['bytes'] / 1e6:.1f} MB ({man['n_files']} files); features {feats}; "
            f"pod max {max_minutes:.0f} min, session max ${max_cost:.2f}")
    ses.save()
    pod_id = None
    spent0 = remote.spent()
    try:
        for k in range(attempts):
            with remote.pod_session(label, max_minutes=max_minutes, gpus=gpus, max_cost=max_cost,
                                    disk_gb=disk_gb) as pod:
                pod_id = pod.id
                facts = host_facts(pod)
                ses.state.setdefault("host_checks", []).append({"attempt": k + 1, "pod": pod.id, **facts})
                bad = [x for x, v, lim in (("cpus", facts["cpus"], min_cpus), ("mem_gb", facts["mem_gb"], min_mem_gb),
                                           ("upload_mb_s", facts["upload_mb_s"], min_upload)) if v < lim]
                ses.log(f"host check {k + 1}/{attempts}: {facts}" + (f" -> REJECTED ({bad})" if bad else " -> ok"))
                ses.save()
                if bad and k < attempts - 1:
                    continue  # leaving the with-block deletes this pod
                pod_info = {"id": pod.id, "gpu": pod.gpu, "cost_per_hr": pod.cost_per_hr, "host": pod.host,
                            "killswitch": pod.killswitch, "started": time.time(), "port": getattr(pod, "port", None),
                            "kill_at_est": round(time.time() + max_minutes * 60, 1), "host_facts": facts}
                ses.state["pod"] = pod_info
                ses.save()
                ses.event("note", "pod", msg=f"pod {pod.id} {pod.gpu} ${pod.cost_per_hr:.2f}/h ready")
                puller = rj._start_puller(ses, pod, [job], bundle, man)
                try:
                    rj._pod_steps(ses, pod, [job], bundle, man, feats)
                finally:
                    rj._stop_puller(puller)
                break
    except Exception as e:  # noqa: BLE001
        ses.state["error"] = f"{type(e).__name__}: {e}"[:2000]
        ses.log(f"session error: {ses.state['error']}")
    finally:
        ses.state["session_cost"] = round(remote.spent() - spent0, 4)
        ses.save()
    res = rj._finish(ses, pod_id)
    res["session_cost"] = ses.state["session_cost"]
    (ses.dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------------------------------------ CLI


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.front_full", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["segments", "prep-local", "plan", "write-job", "session", "run", "supervise",
                                       "status", "detect-one", "decode-check", "fetch-weights", "check", "locgate",
                                       "p2-summary", "pin-betas", "hold"])
    ap.add_argument("take", nargs="?")
    ap.add_argument("cam", nargs="?")
    ap.add_argument("--segs", help="comma list of segment ids (default a,b,c,d)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sam-workers", type=int)
    ap.add_argument("--max-minutes", type=float, default=330.0)
    ap.add_argument("--front-minutes", type=float, default=240.0)
    ap.add_argument("--max-cost", type=float, default=9.0)
    ap.add_argument("--session")
    ap.add_argument("--gpus")
    a = ap.parse_args(argv)
    segs = a.segs.split(",") if a.segs else list(SEG_IDS)
    if a.action == "segments":
        write_segments()
    elif a.action == "prep-local":
        for s in segs:
            prep_segment(s, force=a.force)
    elif a.action == "plan":
        plan_frames()
    elif a.action == "write-job":
        write_job(a.max_minutes, a.front_minutes)
    elif a.action == "session":
        res = session(a.max_minutes, a.max_cost, a.session, gpus=a.gpus.split(",") if a.gpus else None)
        print(json.dumps(res, indent=1, default=str))
    elif a.action == "run":
        raise SystemExit(run(a.resume, segs, a.sam_workers))
    elif a.action == "supervise":
        raise SystemExit(supervise())
    elif a.action == "hold":
        raise SystemExit(hold())
    elif a.action == "status":
        raise SystemExit(status())
    elif a.action == "detect-one":
        detect_one(a.take, a.cam)
    elif a.action == "locgate":
        loc_gate(a.take, a.cam)
    elif a.action == "pin-betas":
        pin_betas()
    elif a.action == "p2-summary":
        p2_summary(segs)
    elif a.action == "decode-check":
        decode_check()
    elif a.action == "fetch-weights":
        fetch_weights()
    elif a.action == "check":
        check_env()


if __name__ == "__main__":
    main()
