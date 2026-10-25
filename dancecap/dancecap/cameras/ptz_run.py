"""Pan / tilt / zoom calibration of one video: the 1 Hz skeleton and the per-frame track (dancecap.cameras.ptz).

    python -m dancecap.cameras.ptz_run skeleton <video> --out <dir> [--step 24] [--rect x0,y0,x1,y1 ...]
    python -m dancecap.cameras.ptz_run track    <video> --skeleton <dir> --out <dir> [--start n --count m]

skeleton  every `step`-th frame: person masks, ALIKED features, LightGlue matches between frames up to ~1.5 min apart,
          homography-verified correspondences, init chain, robust sparse LM (rotation + focal per frame, shared k1 k2) ->
          <out>/skeleton.npz {frame, rv, f, k1, k2, ox, oy, rms_px, ...}, <out>/skeleton.json (numbers), <out>/pairs.npz
track     every frame of [start, start+count): features, matches to the two nearest skeleton frames and the previous
          solved frame, refine_frame() -> <out>/track_<start>.npz {frame, rv, f, rms_px, n_matches}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import ptz as P
from . import ptz_feats as F

OFFSETS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)


def probe(video: str) -> dict:
    import subprocess
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", video],
                       capture_output=True, text=True, check=True).stdout
    s = json.loads(o)["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": float(a) / float(b),
            "frames": int(s["nb_read_packets"])}


def _frames_and_masks(video, meta, step, start, count, rects, yolo, batch=8):
    frames = list(F.read_video_frames(video, meta["width"], meta["height"], step, start, count))
    idx = [k for k, _ in frames]
    imgs = [f for _, f in frames]
    masks = []
    for b in range(0, len(imgs), batch):
        pm = F.person_masks_yolo(imgs[b:b + batch], yolo)
        masks += [F.static_ignore(im.shape[:2], m, rects) for im, m in zip(imgs[b:b + batch], pm)]
    return idx, imgs, masks


def skeleton(video: str, out: Path, step: int | None = None, rects=(), yolo: str = "yolo11x-seg.pt",
             device: str = "cuda", per_pair: int = 40, log=print,
             f0s: tuple[float, ...] = (1200, 1800, 2600, 3600, 4600)) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    meta = probe(video)
    step = step or int(round(meta["fps"]))
    t0 = time.time()
    idx, imgs, masks = _frames_and_masks(video, meta, step, 0, None, list(rects), yolo)
    n = len(idx)
    log(f"{n} frames decoded + masked ({time.time() - t0:.0f} s)")
    mt = F.Matcher(device)
    feats = [mt.extract(im, mk) for im, mk in zip(imgs, masks)]
    del imgs
    log(f"features: mean {np.mean([len(f['kpts']) for f in feats]):.0f} keypoints ({time.time() - t0:.0f} s)")
    ii, jj, pp, qq, edges, nin = [], [], [], [], {}, {}
    for i in range(n):
        for d in OFFSETS:
            j = i + d
            if j >= n:
                continue
            r = F.pair_inliers(feats[i], feats[j], mt.match(feats[i], feats[j]))
            if r is None:
                continue
            p, q, H = r
            ii.append(np.full(len(p), i)); jj.append(np.full(len(p), j)); pp.append(p); qq.append(q)
            edges[(i, j)] = H
            nin[(i, j)] = len(p)
    log(f"{len(edges)} verified pairs ({time.time() - t0:.0f} s)")
    ii, jj = np.concatenate(ii), np.concatenate(jj)
    pp, qq = np.concatenate(pp), np.concatenate(qq)
    raw = (ii, jj, pp, qq)
    pairs = P.balance_pairs(ii, jj, pp, qq, (meta["width"], meta["height"]), per_pair=per_pair)
    log(f"{len(pairs)} correspondences kept")
    np.savez_compressed(out / "pairs.npz", i=pairs.i, j=pairs.j, p=pairs.p, q=pairs.q)
    lens0 = P.Lens(meta["width"], meta["height"])
    init_edges = {k: v for k, v in edges.items() if k[1] - k[0] <= 2}
    best = None
    root = P.best_root(n, init_edges)
    if root != 0:
        log(f"  init chain rooted at skeleton frame {root} (video frame {idx[root]}): frame 0 is badly linked")
    for f0 in f0s:
        rv, f, order = P.init_chain(n, init_edges, lens0, f0, root=root)
        if len(order) < n:
            log(f"  f0 {f0}: only {len(order)}/{n} frames reachable")
        sub = np.isin(pairs.i, order) & np.isin(pairs.j, order)
        s = P.Pairs(pairs.i[sub], pairs.j[sub], pairs.p[sub], pairs.q[sub])
        sol = P.solve_ptz(n, s, rv, f, lens0, free=("k1",), max_nfev=30)
        log(f"  f0 {f0}: median {sol.median_px:.2f} px, p90 {sol.p90_px:.2f}, f {sol.f.min():.0f}..{sol.f.max():.0f}")
        if best is None or sol.median_px < best[0].median_px:
            best = (sol, f0, order)
    sol, f0, order = best
    sub = np.isin(pairs.i, order) & np.isin(pairs.j, order)
    pairs = P.Pairs(pairs.i[sub], pairs.j[sub], pairs.p[sub], pairs.q[sub])
    sol = P.solve_ptz(n, pairs, sol.rv, sol.f, sol.lens, free=("k1",), max_nfev=60)
    for rounds, free, thr in ((1, ("k1",), 8.0), (2, ("k1", "k2"), 4.0), (3, ("k1", "k2"), 2.5)):
        keep = P.reject_outliers(sol, pairs, thr)
        pairs_k = P.Pairs(pairs.i[keep], pairs.j[keep], pairs.p[keep], pairs.q[keep])
        sol = P.solve_ptz(n, pairs_k, sol.rv, sol.f, sol.lens, free=free, max_nfev=40)
        log(f"  round {rounds}: kept {keep.sum()}/{len(keep)}, median {sol.median_px:.2f} p90 {sol.p90_px:.2f} rms {sol.rms_px:.2f} "
            f"k1 {sol.lens.k1:+.4f} k2 {sol.lens.k2:+.4f}")
    np.savez_compressed(out / "skeleton.npz", frame=np.array(idx), rv=sol.rv, f=sol.f,
                        lens=np.array([sol.lens.k1, sol.lens.k2, sol.lens.ox, sol.lens.oy]),
                        size=np.array([meta["width"], meta["height"]]), reached=np.isin(np.arange(n), order))
    info = {"video_frames": meta["frames"], "fps": meta["fps"], "step": step, "n_frames": n, "n_reached": len(order),
            "n_pairs": len(edges), "n_corr": len(pairs), "f_start": f0, "median_px": sol.median_px, "p90_px": sol.p90_px,
            "rms_px": sol.rms_px, "k1": sol.lens.k1, "k2": sol.lens.k2, "pp_offset": [sol.lens.ox, sol.lens.oy],
            "f_min": float(sol.f.min()), "f_max": float(sol.f.max()), "seconds": time.time() - t0}
    (out / "skeleton.json").write_text(json.dumps(info, indent=1))
    return info


def _interp_skeleton(sk: dict, k: float) -> tuple[np.ndarray, float]:
    """Rotation vector and focal at the (fractional) skeleton index k: slerp + log-linear."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    n = len(sk["f"])
    k = float(np.clip(k, 0, n - 1))
    k0 = int(np.floor(k))
    k1 = min(k0 + 1, n - 1)
    u = k - k0
    R = Rot.from_rotvec(sk["rv"][[k0, k1]])
    r = Slerp([0, 1], R)([u]).as_rotvec()[0] if k1 > k0 else sk["rv"][k0]
    f = float(np.exp((1 - u) * np.log(sk["f"][k0]) + u * np.log(sk["f"][k1])))
    return r, f


def track(video: str, skel_dir: Path, frames: list[int], out: Path, rects=(), yolo: str = "yolo11x-seg.pt",
          device: str = "cuda", fix_f: float | None = None, n_ref: int = 3, keep_matches: int = 250, log=print) -> dict:
    """Per-frame (rotation, focal) of the requested video frames against the skeleton frames (lens fixed)."""
    from scipy.spatial.transform import Rotation as Rot
    sk = dict(np.load(skel_dir / "skeleton.npz"))
    meta = probe(video)
    sk_frames = sk["frame"]
    step = int(sk_frames[1] - sk_frames[0]) if len(sk_frames) > 1 else int(round(meta["fps"]))     # the skeleton's own step (J2: 15)
    lens = P.Lens(int(sk["size"][0]), int(sk["size"][1]), float(sk["lens"][0]), float(sk["lens"][1]),
                  float(sk["lens"][2]), float(sk["lens"][3]))
    sol = P.PtzSolution(sk["rv"], sk["f"], lens)
    # only skeleton frames the init chain reached are references (an unreached frame keeps a meaningless pose)
    usable = [s for s in range(len(sk["f"])) if "reached" not in sk or bool(sk["reached"][s])]
    t0 = time.time()
    mt = F.Matcher(device)
    # skeleton features (the frames the skeleton was solved on)
    idx, imgs, masks = _frames_and_masks(video, meta, step, 0, None, list(rects), yolo)
    skf = [mt.extract(im, mk) for im, mk in zip(imgs, masks)]
    del imgs, masks
    log(f"skeleton features {len(skf)} ({time.time() - t0:.0f} s)")
    rows = []
    mrec: dict[str, list] = {"frame": [], "ref": [], "p": [], "q": []}
    rng = np.random.default_rng(0)
    batch = 16
    done = 0

    def process(got):
        nonlocal done
        pm = F.person_masks_yolo([g for _, g in got], yolo)
        for (fi, im), m in zip(got, pm):
            ft = mt.extract(im, F.static_ignore(im.shape[:2], m, list(rects)))
            k = fi / step
            r0, f0 = _interp_skeleton(sk, k)
            near = sorted(usable, key=lambda s: abs(s - k))[:n_ref]
            p_new, q_ref, used = [], [], []
            for s in near:
                r = F.pair_inliers(ft, skf[s], mt.match(ft, skf[s]))
                if r is None:
                    continue
                p_new.append(r[0]); q_ref.append(r[1]); used.append(s)
                keep = rng.permutation(len(r[0]))[:keep_matches]
                mrec["frame"].append(np.full(len(keep), fi)); mrec["ref"].append(np.full(len(keep), s))
                mrec["p"].append(r[0][keep]); mrec["q"].append(r[1][keep])
            if not used:
                rows.append((fi, *r0, f0, np.nan, 0, 0))
                continue
            fg = fix_f or f0
            rvn, fn, rms = P.refine_frame(sol, used, p_new, q_ref, r0, fg, fix_f=fix_f)
            rows.append((fi, *rvn, fn, rms, int(sum(len(x) for x in p_new)), len(used)))
        done += len(got)
        if (done // batch) % 20 == 0:
            log(f"{done}/{len(frames)} frames ({time.time() - t0:.0f} s)")

    buf = []
    for fi, im in F.read_video_frames(video, meta["width"], meta["height"], frames=frames):
        buf.append((fi, im))
        if len(buf) >= batch:
            process(buf)
            buf = []
    if buf:
        process(buf)
    a = np.array(rows, float)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frame=a[:, 0].astype(int), rv=a[:, 1:4], f=a[:, 4], rms_px=a[:, 5], n_inl=a[:, 6].astype(int),
                        n_ref=a[:, 7].astype(int), lens=sk["lens"], size=sk["size"])
    if mrec["p"]:
        np.savez_compressed(out.with_name(out.stem + "_matches.npz"), frame=np.concatenate(mrec["frame"]),
                            ref=np.concatenate(mrec["ref"]), p=np.concatenate(mrec["p"]), q=np.concatenate(mrec["q"]))
    return {"frames": len(rows), "median_rms": float(np.nanmedian(a[:, 5])), "seconds": time.time() - t0}


def refine_tracks(matches: dict, rv_sk: np.ndarray, f_sk: np.ndarray, lens: P.Lens, step: int,
                  fix_f: float | None = None, frames: list[int] | None = None) -> dict:
    """Per-frame (rotation, focal) from the stored track matches (track()'s *_matches.npz) against ANY skeleton solution
    (rv_sk, f_sk at skeleton indices k = frame / step) - so a changed focal hypothesis re-solves in seconds, no GPU.
    Returns {frame, rv, f, rms_px, n_inl}."""
    sk = {"rv": rv_sk, "f": f_sk}
    sol = P.PtzSolution(rv_sk, f_sk, lens)
    fr_all = matches["frame"]
    out = {"frame": [], "rv": [], "f": [], "rms_px": [], "n_inl": []}
    order = np.argsort(fr_all, kind="stable")
    fr_s = fr_all[order]
    ref_s, p_s, q_s = matches["ref"][order], matches["p"][order], matches["q"][order]
    uf, start = np.unique(fr_s, return_index=True)
    end = np.r_[start[1:], len(fr_s)]
    want = set(frames) if frames is not None else None
    for fi, s0, s1 in zip(uf, start, end):
        if want is not None and int(fi) not in want:
            continue
        refs = np.unique(ref_s[s0:s1])
        pn = [p_s[s0:s1][ref_s[s0:s1] == r] for r in refs]
        qr = [q_s[s0:s1][ref_s[s0:s1] == r] for r in refs]
        r0, f0 = _interp_skeleton(sk, fi / step)
        rvn, fn, rms, _ = P.refine_frame_robust(sol, list(refs), pn, qr, r0, fix_f or f0, fix_f=fix_f)
        out["frame"].append(int(fi)); out["rv"].append(rvn); out["f"].append(fn); out["rms_px"].append(rms)
        out["n_inl"].append(int(s1 - s0))
    return {k: np.array(v) for k, v in out.items()}


def load_solution(path: Path) -> tuple[np.ndarray, np.ndarray, P.Lens]:
    """(rv, f, lens) of a skeleton.npz ({frame, rv, f, lens[k1 k2 ox oy], size}) or a skel_f*.npz (one tied focal: {rv, f, k1, size})."""
    d = np.load(path)
    size = d["size"]
    if "lens" in d:
        k = d["lens"]
        lens = P.Lens(int(size[0]), int(size[1]), float(k[0]), float(k[1]), float(k[2]), float(k[3]))
    else:
        lens = P.Lens(int(size[0]), int(size[1]), float(d["k1"]), 0.0, 0.0, 0.0)
    return d["rv"], d["f"], lens


def load_matches(paths: list[Path], sk_len: int | None = None) -> dict:
    parts = [dict(np.load(p)) for p in paths]
    m = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    if sk_len is not None:
        ok = m["ref"] < sk_len                 # references inside the solved skeleton subset
        m = {k: v[ok] for k, v in m.items()}
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("skeleton")
    s.add_argument("video")
    s.add_argument("--out", required=True)
    s.add_argument("--step", type=int)
    s.add_argument("--rect", action="append", default=[], help="x0,y0,x1,y1 screen-fixed overlay to ignore")
    s.add_argument("--yolo", default="yolo11x-seg.pt")
    s.add_argument("--f0", default="", help="comma list of initial focals (px) to try, default 1200..4600 (a 1920 px wide frame)")
    s2 = sub.add_parser("track")
    s2.add_argument("video")
    s2.add_argument("--skeleton", required=True)
    s2.add_argument("--out", required=True)
    s2.add_argument("--frames", required=True, help="json list of video frame indices")
    s2.add_argument("--rect", action="append", default=[])
    s2.add_argument("--yolo", default="yolo11x-seg.pt")
    s2.add_argument("--fix-f", type=float)
    s3 = sub.add_parser("refine")
    s3.add_argument("--matches", nargs="+", required=True)
    s3.add_argument("--solution", required=True)
    s3.add_argument("--step", type=int, required=True)
    s3.add_argument("--fix-f", type=float)
    s3.add_argument("--out", required=True)
    s3.add_argument("--frames", help="json list of frames (default: all)")
    a = ap.parse_args(argv)
    if a.cmd == "refine":
        rv, f, lens = load_solution(Path(a.solution))
        m = load_matches([Path(x) for x in a.matches], len(rv))
        fr = [int(k) for k in json.loads(Path(a.frames).read_text())] if a.frames else None
        t0 = time.time()
        res = refine_tracks(m, rv, f, lens, a.step, fix_f=a.fix_f, frames=fr)
        np.savez_compressed(a.out, **res, lens=np.array([lens.k1, lens.k2, lens.ox, lens.oy]), size=np.array([lens.width, lens.height]))
        print(json.dumps({"frames": int(len(res["frame"])), "median_rms": float(np.median(res["rms_px"])), "seconds": time.time() - t0}))
        return
    if a.cmd == "track":
        rects = [tuple(int(v) for v in r.split(",")) for r in a.rect]
        frames = [int(k) for k in json.loads(Path(a.frames).read_text())]
        print(json.dumps(track(a.video, Path(a.skeleton), frames, Path(a.out), rects, a.yolo, fix_f=a.fix_f), indent=1))
        return
    if a.cmd == "skeleton":
        rects = [tuple(int(v) for v in r.split(",")) for r in a.rect]
        info = skeleton(a.video, Path(a.out), a.step, rects, a.yolo,
                        f0s=tuple(float(v) for v in a.f0.split(",")) if a.f0 else (1200, 1800, 2600, 3600, 4600))
        print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
