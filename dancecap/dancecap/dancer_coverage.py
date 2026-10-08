"""Cheap dancer-coverage probe: which cameras SEE the couple (not just record) over each camera's whole recording.

    python -m dancecap.dancer_coverage probe <take> [--cams 01,02] [--every 0.2] [--long-edge 640]   # local GPU, short
    python -m dancecap.dancer_coverage table <take> [--block 2] [--validate larissa_kadu_38]            # local CPU
    python -m dancecap.dancer_coverage sheet <take> [--step 4] [--cams 06] [--t0 0 --t1 20]           # review images

Why: a take over the whole dance needs, per short block of reference time, the set of cameras whose frames show the
dancers (a camera can record the floor without them: zoomed on one dancer, blocked by spectators, pointed away).
The capture front-end only exists inside larissa_kadu_38's 30-68 s window, so this probe decodes every source video
at low resolution and a few frames per second (ffmpeg at below-normal priority, 2 threads, autorotated like ingest,
source PTS -> reference time through the take's audio clocks) and runs ONE model (yolo11x-pose, person class, 640 px,
fp16, under work/gpu.lock) on the samples.

Per sample, each person box gets the colour descriptors of dancers_appearance (torso / legs brightness, red hair above
the shoulders; the clothing the user described) and the size / centrality score of dancers_select. A box is
  lead-like    dark top (torso V <= 110) and no red hair (red < 0.05)
  follow-like  red hair (red >= 0.06) or a light top (torso V >= 150) without a dark top
and must be dancer-sized (height >= max(0.10, 0.35 x the camera's median dancer height in the validation take) of the
frame) and not a reflection (nested inside a much larger box). The PAIR is visible when the best lead-like / follow-like
pair has colour log-odds (dancers_appearance.lead_llr) >= 1.5 and the two boxes are within 3 box heights.
Embraces merge the two people into one box, so "at least one dancer" is reported too.

Outputs (work/<take>/probe/coverage/):
  <cam>.npz        t (S,) reference s, pts (S,), wh (2,) probe px, scale (native px per probe px), boxes (S, N, 4),
                   scores (S, N), kpts (S, N, 17, 3), desc (S, N, 7) [torso_v, torso_s, leg_v, red, torso_n, head_n,
                   legs_n], valid (S, N)
  samples.json     per camera per sample: pair (bool), lead_box, follow_box, n_dancers (0-2), llr
  coverage.json    blocks [{t0, t1, cams {<cam>: {rec, n, pair, any}}}], per camera recording span, validation vs a
                   take whose dancers/roles.json covers part of the recording (precision / recall of "pair visible")
The table / sheets are review aids for planning, not inputs of any fit (the front-end's SAM masks and re-ID decide).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .take import PACKAGE_ROOT, Take

SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[\d.]+)")
POSE_WEIGHTS = PACKAGE_ROOT / "weights" / "yolo11x-pose.pt"
NMAX = 24
BELOW_NORMAL = 0x00004000 if os.name == "nt" else 0
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
DESC_KEYS = ("torso_v", "torso_s", "leg_v", "red", "torso_n", "head_n", "legs_n")


def _below_normal_self() -> None:
    try:
        if os.name == "nt":
            import ctypes
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(h, BELOW_NORMAL)
        else:
            os.nice(5)
    except Exception:
        pass


def out_dir(take: Take) -> Path:
    d = take.work / "probe" / "coverage"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _probe_wh(path: Path) -> tuple[int, int]:
    from .ingest import probe
    info = probe(path)
    return info["width"], info["height"]


def sample_frames(path: Path, every_s: float, long_edge: int, threads: int = 2):
    """Yield (pts_seconds, bgr) for frames at least every_s apart (source PTS, autorotated, long edge = long_edge)."""
    w, h = _probe_wh(path)
    s = min(1.0, long_edge / max(w, h))
    ow, oh = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    vf = (f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{every_s - 0.004:.4f})',"
          f"scale={ow}:{oh}:flags=area,showinfo")
    cmd = ["ffmpeg", "-v", "info", "-nostats", "-threads", str(threads), "-copyts", "-i", str(path), "-map", "0:v:0",
           "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            creationflags=BELOW_NORMAL | NO_WINDOW)
    pts: list[float] = []
    cond = threading.Condition()

    def read_err():
        for line in iter(proc.stderr.readline, b""):
            m = SHOWINFO.search(line.decode("utf-8", "replace"))
            if m:
                with cond:
                    pts.append(float(m.group(3)))
                    cond.notify_all()
        with cond:
            pts.append(float("nan"))  # sentinel: stderr closed
            cond.notify_all()

    th = threading.Thread(target=read_err, daemon=True)
    th.start()
    nbytes = ow * oh * 3
    k = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            with cond:
                while len(pts) <= k:
                    cond.wait(timeout=30)
                p = pts[k]
            if not np.isfinite(p):
                break
            yield p, np.frombuffer(buf, np.uint8).reshape(oh, ow, 3), s
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()
        th.join(timeout=5)


def _descriptors(img: np.ndarray, boxes: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    from .dancers_appearance import descriptors
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = np.full((len(boxes), len(DESC_KEYS)), np.nan, np.float32)
    for j, (b, k) in enumerate(zip(boxes, kpts)):
        d = descriptors(hsv, b, k)
        out[j] = [d.get(key, np.nan) for key in DESC_KEYS]
    return out


def probe(take: Take, cams: list[str] | None = None, every_s: float = 0.2, long_edge: int = 640,
          batch: int = 16, log=print) -> dict:
    """Decode + detect every camera's whole recording -> probe/coverage/<cam>.npz."""
    from ultralytics import YOLO
    from .gpu import exclusive
    _below_normal_self()
    od = out_dir(take)
    model = YOLO(str(POSE_WEIGHTS))
    res = {}
    with exclusive("dancer_coverage probe"):
        for s in take.sources:
            if cams and s.id not in cams:
                continue
            t0 = time.time()
            T, P, B, S, K, D, V = [], [], [], [], [], [], []
            pend: list[tuple[float, np.ndarray]] = []
            wh, scale = None, 1.0

            def flush():
                if not pend:
                    return
                imgs = [im for _, im in pend]
                rs = model.predict(imgs, imgsz=long_edge, conf=0.2, classes=[0], half=True, verbose=False, device=0)
                for (p, im), r in zip(pend, rs):
                    bx = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                    sc = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
                    kp = (r.keypoints.data.cpu().numpy() if r.keypoints is not None and len(bx)
                          else np.zeros((len(bx), 17, 3)))
                    order = np.argsort(-sc)[:NMAX]
                    bx, sc, kp = bx[order], sc[order], kp[order]
                    n = len(bx)
                    b_ = np.zeros((NMAX, 4), np.float32); b_[:n] = bx
                    s_ = np.zeros(NMAX, np.float32); s_[:n] = sc
                    k_ = np.zeros((NMAX, 17, 3), np.float32); k_[:n] = kp
                    d_ = np.full((NMAX, len(DESC_KEYS)), np.nan, np.float32); d_[:n] = _descriptors(im, bx, kp)
                    v_ = np.zeros(NMAX, bool); v_[:n] = True
                    T.append(s.to_reference(p)); P.append(p); B.append(b_); S.append(s_); K.append(k_)
                    D.append(d_); V.append(v_)
                pend.clear()

            for p, im, sc_ in sample_frames(s.path, every_s, long_edge):
                wh, scale = (im.shape[1], im.shape[0]), 1.0 / sc_
                pend.append((p, im))
                if len(pend) >= batch:
                    flush()
            flush()
            np.savez_compressed(od / f"{s.id}.npz", t=np.asarray(T), pts=np.asarray(P), wh=np.asarray(wh or (0, 0)),
                                scale=np.float32(scale), boxes=np.asarray(B), scores=np.asarray(S),
                                kpts=np.asarray(K), desc=np.asarray(D), valid=np.asarray(V))
            res[s.id] = {"samples": len(T), "seconds": round(time.time() - t0, 1),
                         "t": [round(T[0], 3), round(T[-1], 3)] if T else None}
            log(f"[coverage] {s.id}: {len(T)} samples {res[s.id]['t']} in {res[s.id]['seconds']} s")
    return res


# ------------------------------------------------------------------------------------------------ classification

def _load(take: Take, cam: str) -> dict | None:
    p = out_dir(take) / f"{cam}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def _reference_heights(ref_work: Path, cam: str, scale_to_probe: float) -> float | None:
    """Median dancer box height (probe px) of a take whose roles.json covers part of this camera."""
    p = ref_work / "dancers" / "roles.json"
    if not p.exists():
        return None
    fr = json.loads(p.read_text()).get("cams", {}).get(cam, {}).get("frames", [])
    hs = [(r[role]["box"][3] - r[role]["box"][1]) for r in fr for role in ("lead", "follow")
          if r.get(role, {}).get("box") is not None]
    return float(np.median(hs)) * scale_to_probe if hs else None


def classify(z: dict, ref_h: float | None) -> list[dict]:
    from .dancers_appearance import lead_llr
    W, H = (int(z["wh"][0]), int(z["wh"][1]))
    min_h = max(0.10 * H, 0.35 * ref_h) if ref_h else 0.10 * H
    out = []
    for i in range(len(z["t"])):
        v = z["valid"][i]
        b, sc, d = z["boxes"][i][v], z["scores"][i][v], z["desc"][i][v]
        h = b[:, 3] - b[:, 1] if len(b) else np.zeros(0)
        keep = (h >= min_h) & (sc >= 0.3)
        # reflections / screen copies: nested inside a much larger box
        for a in range(len(b)):
            for c in range(len(b)):
                if a == c or not keep[a]:
                    continue
                ia = max(0, min(b[a, 2], b[c, 2]) - max(b[a, 0], b[c, 0])) * max(0, min(b[a, 3], b[c, 3]) - max(b[a, 1], b[c, 1]))
                area_a = (b[a, 2] - b[a, 0]) * (b[a, 3] - b[a, 1])
                area_c = (b[c, 2] - b[c, 0]) * (b[c, 3] - b[c, 1])
                if area_a > 0 and ia / area_a > 0.92 and area_a < 0.25 * area_c:
                    keep[a] = False
        idx = np.flatnonzero(keep)
        desc = {j: dict(zip(DESC_KEYS, map(float, d[j]))) for j in idx}

        def lead_like(j):
            x = desc[j]
            return np.isfinite(x["torso_v"]) and x["torso_v"] <= 110 and not (np.isfinite(x["red"]) and x["red"] >= 0.05)

        def follow_like(j):
            x = desc[j]
            red = np.isfinite(x["red"]) and x["red"] >= 0.06
            light = np.isfinite(x["torso_v"]) and x["torso_v"] >= 150
            return red or light

        best = None
        for a in idx:
            for c in idx:
                if a == c or not (lead_like(a) and follow_like(c)):
                    continue
                llr, _ = lead_llr(desc[a], desc[c])
                ca, cc = (b[a, :2] + b[a, 2:]) / 2, (b[c, :2] + b[c, 2:]) / 2
                hh = 0.5 * (h[a] + h[c])
                if np.linalg.norm(ca - cc) > 3.0 * hh:
                    continue
                score = llr + 0.5 * np.log(hh / H + 1e-6) - abs(0.5 * (ca[0] + cc[0]) / W - 0.5)
                if best is None or score > best[0]:
                    best = (score, a, c, llr)
        pair = best is not None and best[3] >= 1.5
        n_lead = int(any(lead_like(j) for j in idx))
        n_fol = int(any(follow_like(j) for j in idx))
        rec = {"t": float(z["t"][i]), "pair": bool(pair), "n_dancers": int(2 if pair else min(1, n_lead + n_fol)),
               "llr": round(float(best[3]), 2) if best else None}
        if pair:
            rec["lead_box"] = [round(float(x), 1) for x in b[best[1]]]
            rec["follow_box"] = [round(float(x), 1) for x in b[best[2]]]
        out.append(rec)
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def validate(samples: list[dict], ref_work: Path, cam: str, scale_to_probe: float) -> dict | None:
    """Precision / recall of 'pair visible' vs a take's roles.json (both role boxes present, by nearest frame)."""
    p = ref_work / "dancers" / "roles.json"
    if not p.exists():
        return None
    fr = json.loads(p.read_text()).get("cams", {}).get(cam, {}).get("frames", [])
    if not fr:
        return None
    rt = np.array([r["t"] for r in fr])
    tp = fp = fn = tn = 0
    role_ok = role_swapped = n_pair = 0
    for s in samples:
        if s["t"] < rt[0] or s["t"] > rt[-1]:
            continue
        r = fr[int(np.argmin(np.abs(rt - s["t"])))]
        both = r["lead"].get("box") is not None and r["follow"].get("box") is not None
        if s["pair"] and both:
            tp += 1
            lb = np.asarray(r["lead"]["box"]) * scale_to_probe
            fb = np.asarray(r["follow"]["box"]) * scale_to_probe
            n_pair += 1
            a, b = _iou(s["lead_box"], lb), _iou(s["follow_box"], fb)
            c, d = _iou(s["lead_box"], fb), _iou(s["follow_box"], lb)
            role_ok += int(max(a, b) >= 0.3 and not (c > a and d > b))  # >= one role on its mask box, not swapped
            role_swapped += int(c >= 0.3 and d >= 0.3 and c > a and d > b)
        elif s["pair"]:
            fp += 1
        elif both:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    if not n:
        return None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "roles_match": round(role_ok / n_pair, 3) if n_pair else None,
            "roles_swapped": round(role_swapped / n_pair, 3) if n_pair else None}


def table(take: Take, block_s: float = 2.0, ref_take: str | None = None, log=print) -> dict:
    """Coverage per block of reference time -> probe/coverage/coverage.json + samples.json."""
    od = out_dir(take)
    ref_work = (PACKAGE_ROOT / "work" / ref_take) if ref_take else None
    samples, spans, val = {}, {}, {}
    for s in take.sources:
        z = _load(take, s.id)
        if z is None:
            continue
        native = None
        fj = (ref_work / "frames" / s.id / "frames.json") if ref_work else None
        if fj is not None and fj.exists():
            native = json.loads(fj.read_text())
        to_probe = 1.0 / float(z["scale"])
        ref_h = _reference_heights(ref_work, s.id, to_probe) if ref_work else None
        samples[s.id] = classify(z, ref_h)
        # recording span on the reference clock: the sampled range (first/last decoded frame within one sample)
        spans[s.id] = [float(z["t"][0]), float(z["t"][-1])]
        if ref_work and native is not None:
            val[s.id] = validate(samples[s.id], ref_work, s.id, to_probe)
            val[s.id] = {**(val[s.id] or {}), "ref_dancer_h_probe_px": round(ref_h, 1) if ref_h else None}
    lo = min(v[0] for v in spans.values())
    hi = max(v[1] for v in spans.values())
    b0 = np.floor(lo / block_s) * block_s
    edges = np.arange(b0, hi + block_s, block_s)
    blocks = []
    for a, b in zip(edges[:-1], edges[1:]):
        cams = {}
        for c, ss in samples.items():
            sel = [x for x in ss if a <= x["t"] < b]
            rec = spans[c][0] <= b and spans[c][1] >= a
            if not sel:
                cams[c] = {"rec": bool(rec), "n": 0, "pair": 0.0, "any": 0.0}
                continue
            cams[c] = {"rec": True, "n": len(sel), "pair": round(float(np.mean([x["pair"] for x in sel])), 2),
                       "any": round(float(np.mean([x["n_dancers"] > 0 for x in sel])), 2)}
        blocks.append({"t0": round(float(a), 3), "t1": round(float(b), 3), "cams": cams})
    out = {"take": take.name, "block_s": block_s, "spans": spans, "validation_take": ref_take, "validation": val,
           "blocks": blocks, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (od / "samples.json").write_text(json.dumps(samples))
    (od / "coverage.json").write_text(json.dumps(out, indent=1))
    log(f"[coverage] {len(blocks)} blocks of {block_s} s -> {od / 'coverage.json'}")
    return out


def sheet(take: Take, cams: list[str] | None, t0: float | None, t1: float | None, step: float, out: Path,
          thumb_h: int = 200, cols: int = 10) -> Path:
    """Contact sheet: thumbnails every `step` s per camera with the probe's pair boxes (lead blue, follow red)."""
    od = out_dir(take)
    samples = json.loads((od / "samples.json").read_text())
    rows = []
    for s in take.sources:
        if cams and s.id not in cams:
            continue
        ss = samples.get(s.id, [])
        if not ss:
            continue
        ts = np.array([x["t"] for x in ss])
        lo = max(t0 if t0 is not None else ts[0], ts[0])
        hi = min(t1 if t1 is not None else ts[-1], ts[-1])
        want = np.arange(np.ceil(lo / step) * step, hi + 1e-6, step)
        tiles = []
        for tw in want:
            k = int(np.argmin(np.abs(ts - tw)))
            x = ss[k]
            img = _grab(s, x["t"])
            if img is None:
                continue
            sc = thumb_h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * sc), thumb_h), interpolation=cv2.INTER_AREA)
            z = _load(take, s.id)
            ps = thumb_h / float(z["wh"][1])  # probe px -> thumbnail px (same aspect)
            for key, col in (("lead_box", (255, 120, 0)), ("follow_box", (0, 0, 255))):
                if key in x:
                    bb = (np.asarray(x[key]) * ps).astype(int)
                    cv2.rectangle(img, tuple(bb[:2]), tuple(bb[2:]), col, 2)
            lab = f"{s.id} {x['t']:.1f}s {'PAIR' if x['pair'] else ('one' if x['n_dancers'] else '-')}"
            cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
            cv2.putText(img, lab, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        if tiles:
            rows.append(tiles)
    flat = [t for r in rows for t in r]
    if not flat:
        raise SystemExit("[coverage] nothing to draw")
    tw = max(t.shape[1] for t in flat)
    grid_rows = []
    for r in rows:
        for i in range(0, len(r), cols):
            chunk = [cv2.copyMakeBorder(t, 0, 0, 0, tw - t.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
                     for t in r[i:i + cols]]
            while len(chunk) < cols:
                chunk.append(np.full((thumb_h, tw, 3), 40, np.uint8))
            grid_rows.append(np.hstack(chunk))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(grid_rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def _grab(src, t_ref: float) -> np.ndarray | None:
    """One autorotated frame at reference time t (low-res, for sheets)."""
    st = max(0.0, src.to_source(t_ref))
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, st - 0.5):.3f}", "-copyts", "-i", str(src.path), "-map", "0:v:0",
           "-vf", f"select='gte(t,{st - 0.02:.3f})',scale=-2:480", "-frames:v", "1", "-f", "image2pipe",
           "-vcodec", "png", "-"]
    p = subprocess.run(cmd, capture_output=True, creationflags=BELOW_NORMAL | NO_WINDOW, stdin=subprocess.DEVNULL)
    if p.returncode != 0 or not p.stdout:
        return None
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.dancer_coverage", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["probe", "table", "sheet"])
    ap.add_argument("take")
    ap.add_argument("--cams")
    ap.add_argument("--every", type=float, default=0.2)
    ap.add_argument("--long-edge", type=int, default=640)
    ap.add_argument("--block", type=float, default=2.0)
    ap.add_argument("--validate", help="take whose dancers/roles.json covers part of the recordings")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--step", type=float, default=4.0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    take = Take.load(a.take)
    cams = a.cams.split(",") if a.cams else None
    if a.action == "probe":
        probe(take, cams, a.every, a.long_edge)
    elif a.action == "table":
        table(take, a.block, a.validate)
    else:
        out = Path(a.out) if a.out else take.work / "review" / f"coverage_sheet{'_' + '-'.join(cams) if cams else ''}.jpg"
        print(sheet(take, cams, a.t0, a.t1, a.step, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
