"""Dancers: which two people are the couple, who is lead / follow, and their per-frame pixel masks.

    python -m dancecap dancers-labels larissa_kadu [--typesafe]   # steps 1-2: tracks + labels + review sheet
    python -m dancecap dancers larissa_kadu [--cams 01,02]       # steps 1-5: SAM 2.1 masks, identity repair, QA

Steps per camera (2D only - no cameras exist yet):
 1. candidate tracks (dancers_select): two continuous person-box tracks for the central couple; spectators,
    mirror reflections and in-hand phone screens are scored out.
 2. role labels (dancers_appearance): colour rule on measured pixels - the follow's white top and light jeans
    are brighter, the follow has red hair above the shoulders. Frame-wise log-odds are smoothed along the
    two tracks with a keep/swap Viterbi; the label frame is the clean (separated) frame with the strongest
    agreeing evidence -> dancers/labels.json (by="auto", per-camera confidence) and labels_review.jpg.
    dancers/labels_override.json {cam: {"swap": true}} flips a camera after human review (by="override").
 3. SAM 2.1 video propagation (dancers_sam), obj 1 = lead, obj 2 = follow, prompted with the label boxes,
    forward and backward; re-anchored only at clean frames (track boxes IoU < 0.1, confident colour
    evidence) whose propagated mask drifted from the role-matched detection (IoU < 0.5). The raw SAM
    output is cached in dancers/cache/<cam>_sam.npz (--reuse-sam redoes steps 4-5 only).
 4. identity verification: per-frame colour log-odds of the two MASKS against per-role galleries taken
    around the VERIFIED anchor frames (label + re-anchors; medians over non-embrace frames within 1-4 s, else
    the label's descriptors - never SAM's majority labelling), abstaining during embraces (mask IoU > 0.3 or
    area collapse); keep/swap Viterbi with a switching penalty worth ~5 frames of consistent evidence, pinned
    to 'keep' at the anchors; swapped segments relabelled; cameras whose evidence contradicts the anchors
    are flagged (report/roles identity_flags). Pixels claimed by both objects go to the more confident one;
    small detached blobs (relative to the running-median area, not near a wrist/elbow) are dropped.
 5. outputs (CONTRACTS.md): masks/ (full), body/ (follow hair trimmed; dancers_hair: keypoint-anchored, raised
    arms / face / hands stay body), hair/ (follow), roles.json, report.json, qa/<cam>.mp4 overlay videos,
    qa/<cam>_sheet.jpg (one couple-cropped tile per reference second), qa/<cam>_dense.jpg (every 0.25 s) and
    review/dancers_body_hair.jpg (full | body | hair on raised-arm and hair-whip frames).
 With --reuse-sam and valid caches, neither the GPU lock nor SAM is touched.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .dancers_appearance import RuleConfig, colour_names, descriptors, lead_llr
from .dancers_hair import HairColour, HairConfig, head_geom, split_hair  # noqa: F401  (split_hair re-exported)
from .dancers_select import MISS, SelectConfig, candidate_tracks
from .detect import iou_matrix
from .take import Take

ROLES = ("lead", "follow")
PID = {"lead": 1, "follow": 2}
CUES = ("torso_v", "leg_v", "red")


@dataclass
class DancersConfig:
    select: SelectConfig = field(default_factory=SelectConfig)
    rule: RuleConfig = field(default_factory=RuleConfig)
    hair: HairConfig = field(default_factory=HairConfig)
    clean_iou: float = 0.1  # pair separated
    evidence_iou: float = 0.3  # boxes overlapping more than this: colour evidence abstains
    llr_clip: float = 2.0  # per-frame evidence cap for the keep/swap Viterbi
    switch_frames: float = 5.0  # switching costs ~this many frames of capped, consistent evidence
    anchor_llr: float = 2.5  # per-frame colour log-odds needed to re-anchor from a detection
    drift_iou: float = 0.5
    reanchor_gap: int = 8
    embrace_iou: float = 0.3
    area_collapse: float = 0.4  # area below this x the running median = collapsed (occluded / merged)
    gallery_windows: tuple[float, ...] = (1.0, 2.0, 4.0)  # s around the verified anchors (grown until filled)
    gallery_min: int = 5  # frames per cue a gallery window must provide, else the label's own descriptors
    stray_frac: float = 0.02  # clean_mask: detached blobs < this x the running-median area may be dropped
    kp_fill: int = 3  # frames: a follow without matched keypoints borrows the nearest frame's (hair split)
    tight_margin: float = 0.15  # QA: mask pixels allowed outside the pair's boxes grown by this fraction
    tight_max_out: float = 0.10
    qa_height: int = 960


# ---------------------------------------------------------------------------
# camera data
# ---------------------------------------------------------------------------

@dataclass
class Cam:
    id: str
    work: Path
    meta: dict
    people: dict

    @property
    def files(self) -> list[Path]:
        return [self.work / "frames" / self.id / r["file"] for r in self.meta["frames"]]

    @property
    def t(self) -> np.ndarray:
        return np.array([r["t"] for r in self.meta["frames"]])

    @property
    def wh(self) -> tuple[int, int]:
        return int(self.meta["width"]), int(self.meta["height"])

    @property
    def fps(self) -> float:
        return float(self.meta["fps_median"])

    @classmethod
    def load(cls, work: Path, cam: str) -> "Cam":
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        p = np.load(work / "people" / f"{cam}.npz")
        people = {k: p[k] for k in ("boxes", "scores", "valid", "kpts")}
        if len(people["valid"]) != len(meta["frames"]):
            raise ValueError(f"people/{cam}.npz has {len(people['valid'])} frames, frames.json {len(meta['frames'])}")
        return cls(cam, work, meta, people)


def read_frames(files: list[Path], idx=None) -> list[np.ndarray]:
    idx = range(len(files)) if idx is None else idx
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(lambda i: cv2.imread(str(files[i]), cv2.IMREAD_COLOR), idx))


class LazyImages:
    """The canonical JPEGs as a sequence (len + [i]) decoded on demand, nothing kept: track_evidence reads each frame once,
    so the labels hold a few frames instead of the whole camera (5.7k x 720p = 15.8 GB, the 2026-10-23 crash). With
    `order` (the frames that will be read, in reading order) the next `ahead` of them are decoded by a small thread pool
    while the caller works on the current one (read_frames' parallel decode, bounded to `ahead` frames)."""

    def __init__(self, files: list[Path], order=None, ahead: int = 0, workers: int = 4):
        self.files = files
        self.order = [int(i) for i in order] if order is not None and ahead > 0 else []
        self.pos = {i: k for k, i in enumerate(self.order)}
        self.ahead = ahead
        self.pending: dict = {}  # frame -> future, the frames after the last one read (at most `ahead`)
        self.next = 0  # position in order of the next frame to submit
        self.pool = ThreadPoolExecutor(workers) if self.order else None

    def __len__(self) -> int:
        return len(self.files)

    def read(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.files[i])
        return img

    def __getitem__(self, i: int) -> np.ndarray:
        i = int(i)
        k = self.pos.get(i)
        if self.pool is None or k is None:
            return self.read(i)
        fut = self.pending.pop(i, None)
        for f in [f for f in self.pending if self.pos[f] < k]:  # frames skipped by the caller: never kept
            self.pending.pop(f).cancel()
        self.next = max(self.next, k + 1)
        while self.next < len(self.order) and self.next <= k + self.ahead:
            f = self.order[self.next]
            self.pending[f] = self.pool.submit(self.read, f)
            self.next += 1
        return fut.result() if fut is not None else self.read(i)

    def close(self) -> None:
        if self.pool is not None:
            for f in self.pending.values():
                f.cancel()
            self.pending.clear()
            self.pool.shutdown(wait=True)
            self.pool = None


def box_iou(a, b) -> float:
    if a is None or b is None or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    return float(iou_matrix(np.asarray(a)[None], np.asarray(b)[None])[0, 0])


def mask_box(m: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


# ---------------------------------------------------------------------------
# keep / swap smoothing (used for the candidate tracks and for the SAM masks)
# ---------------------------------------------------------------------------

def _state_costs(llr: np.ndarray, clip: float) -> np.ndarray:
    """(F, 2) negative log-likelihoods of {keep, swap} from per-frame log-odds that 'keep' is right."""
    e = np.clip(np.nan_to_num(np.asarray(llr, float)), -clip, clip)
    return np.stack([np.logaddexp(0, -e), np.logaddexp(0, e)], 1)


def keep_swap_viterbi(llr: np.ndarray, clip: float = 2.0, switch_frames: float = 5.0,
                      pinned=None) -> tuple[np.ndarray, np.ndarray]:
    """Two-state Viterbi {keep=0, swap=1} with a switching penalty worth `switch_frames` frames of capped
    evidence; returns (states (F,) int, swap posterior (F,) from forward-backward with the same model).
    `pinned` frames (verified anchors) are forced to 'keep'."""
    c = _state_costs(llr, clip)
    f = len(c)
    if f == 0:
        return np.zeros(0, int), np.zeros(0)
    for i in (pinned or ()):
        if 0 <= int(i) < f:
            c[int(i), 1] = 1e6
    pen = switch_frames * clip  # one frame of capped evidence moves the cost difference by `clip`
    trans = np.array([[0.0, pen], [pen, 0.0]])
    cost = c[0].copy()
    back = np.zeros((f, 2), int)
    for i in range(1, f):
        tot = cost[:, None] + trans
        back[i] = tot.argmin(0)
        cost = tot.min(0) + c[i]
    path = np.zeros(f, int)
    path[-1] = int(cost.argmin())
    for i in range(f - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    # posterior (log domain)
    lt = -trans
    la = np.zeros((f, 2))
    la[0] = -c[0] + np.log(0.5)
    for i in range(1, f):
        la[i] = np.logaddexp(la[i - 1][0] + lt[0], la[i - 1][1] + lt[1]) - c[i]
    lb = np.zeros((f, 2))
    for i in range(f - 2, -1, -1):
        nxt = lb[i + 1] - c[i + 1]
        lb[i] = np.logaddexp(lt[:, 0] + nxt[0], lt[:, 1] + nxt[1])
    post = la + lb
    post -= np.logaddexp(post[:, 0], post[:, 1])[:, None]
    return path, np.exp(post[:, 1])


# ---------------------------------------------------------------------------
# steps 1-2: candidate tracks and role labels
# ---------------------------------------------------------------------------

def track_evidence(cam: Cam, tracks: dict, imgs: list[np.ndarray], cfg: DancersConfig) -> dict:
    """Per-frame colour descriptors of track slots A/B and log-odds that slot A is the lead."""
    f = len(imgs)
    llr = np.zeros(f)
    pair_iou = np.zeros(f)
    desc: list[tuple[dict | None, dict | None]] = []
    cues: list[dict] = []
    k = cam.people["kpts"]
    for i in range(f):
        ia, ib = tracks["idx"][i]
        if ia == MISS or ib == MISS:
            desc.append((None, None))
            cues.append({})
            continue
        ba, bb = tracks["boxes"][i]
        pair_iou[i] = box_iou(ba, bb)
        hsv = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2HSV)
        da, db = descriptors(hsv, ba, k[i, ia]), descriptors(hsv, bb, k[i, ib])
        desc.append((da, db))
        if pair_iou[i] <= cfg.evidence_iou:
            llr[i], c = lead_llr(da, db, cfg.rule)
            cues.append(c)
        else:
            cues.append({})
    state, post = keep_swap_viterbi(llr, cfg.llr_clip, cfg.switch_frames)
    return {"llr": llr, "pair_iou": pair_iou, "desc": desc, "cues": cues, "a_is_follow": state.astype(bool),
            "swap_post": post}


def role_boxes(tracks: dict, ev: dict, i: int) -> dict[str, np.ndarray]:
    """{lead, follow} track boxes at frame i under the smoothed track labelling."""
    a, b = tracks["boxes"][i]
    return {"lead": b, "follow": a} if ev["a_is_follow"][i] else {"lead": a, "follow": b}


def choose_label_frame(cam: Cam, tracks: dict, ev: dict, cfg: DancersConfig) -> int | None:
    """Clean frame (separated, untruncated, both with keypoints) with the strongest evidence agreeing with
    the smoothed labelling; mildly prefers the middle of the window. When the couple never separates to
    IoU < clean_iou, the separation requirement is relaxed step by step (up to evidence_iou)."""
    w, h = cam.wh
    f = len(ev["llr"])
    signed = np.where(ev["a_is_follow"], -ev["llr"], ev["llr"])
    for max_iou, min_kp in ((cfg.clean_iou, 10), (0.2, 8), (cfg.evidence_iou, 6)):
        best, best_score = None, -np.inf
        for i in range(f):
            ia, ib = tracks["idx"][i]
            if ia == MISS or ib == MISS or ev["pair_iou"][i] > max_iou:
                continue
            bx = tracks["boxes"][i]
            if (bx[:, 3] > 0.985 * h).any() or (bx[:, 0] < 2).any() or (bx[:, 2] > w - 2).any():
                continue
            kp = cam.people["kpts"][i, [ia, ib], :, 2]
            if ((kp > 0.3).sum(1) < min_kp).any():
                continue
            score = signed[i] - 0.5 * abs(i / max(f - 1, 1) - 0.5)
            if score > best_score:
                best, best_score = i, score
        if best is not None and signed[best] > 1.0:
            return best
    return None


def auto_label(cam: Cam, tracks: dict, ev: dict, cfg: DancersConfig) -> dict:
    i = choose_label_frame(cam, tracks, ev, cfg)
    if i is None:
        raise RuntimeError(f"cam {cam.id}: no clean frame with both dancers separated - label it by hand")
    rb = role_boxes(tracks, ev, i)
    da, db = ev["desc"][i]
    lead_d, fol_d = (db, da) if ev["a_is_follow"][i] else (da, db)
    margin = float(abs(ev["llr"][i]))
    signed = np.where(ev["a_is_follow"], -ev["llr"], ev["llr"])
    used = ev["llr"] != 0
    agree = float(np.clip(signed[used], 0, None).sum() / max(np.abs(signed[used]).sum(), 1e-9)) if used.any() else 0.0
    conf = float(min(1 / (1 + np.exp(-margin)), agree))
    return {"frame": int(i), "t": float(cam.t[i]), "lead_box": [round(float(v), 1) for v in rb["lead"]],
            "follow_box": [round(float(v), 1) for v in rb["follow"]], "by": "auto", "confidence": round(conf, 3),
            "frame_llr": round(margin, 2), "evidence_agreement": round(agree, 3),
            "track_switches": int(np.abs(np.diff(ev["a_is_follow"].astype(int))).sum()),
            "cues": {k: round(v, 2) for k, v in ev["cues"][i].items()},
            "descriptors": {"lead": _round(lead_d), "follow": _round(fol_d)},
            "colour_names": {"lead": colour_names(lead_d), "follow": colour_names(fol_d)}}


def _round(d: dict) -> dict:
    return {k: (round(v, 3) if np.isfinite(v) else None) for k, v in d.items()}


def apply_override(labels: dict, override: dict) -> dict:
    """labels_override.json {cam: {"swap": true}} -> swapped boxes, by="override"."""
    out = json.loads(json.dumps(labels))
    for cam, o in (override or {}).items():
        if cam in out and o.get("swap"):
            e = out[cam]
            e["lead_box"], e["follow_box"] = e["follow_box"], e["lead_box"]
            e["descriptors"] = {"lead": e["descriptors"]["follow"], "follow": e["descriptors"]["lead"]}
            e["colour_names"] = {"lead": e["colour_names"]["follow"], "follow": e["colour_names"]["lead"]}
            e["by"] = "override"
            e["auto_swapped_by_override"] = True
    return out


def load_labels(work: Path) -> dict:
    d = work / "dancers"
    labels = json.loads((d / "labels.json").read_text())
    ov = d / "labels_override.json"
    return apply_override(labels, json.loads(ov.read_text()) if ov.exists() else {})


def labels_for_cam(cam: Cam, cfg: DancersConfig, imgs: list[np.ndarray] | None = None) -> tuple[dict, dict, dict]:
    """Steps 1-2 of one camera. Without `imgs` (decoded frames a caller already holds) the frames stream from disk:
    only the frames with both candidate tracks (the ones track_evidence reads), 8 decoded ahead."""
    p = cam.people
    tracks = candidate_tracks(p["boxes"], p["scores"], p["valid"], p["kpts"], *cam.wh, cam.fps, cfg.select)
    lazy = None
    if imgs is None:
        imgs = lazy = LazyImages(cam.files, np.flatnonzero((np.asarray(tracks["idx"]) != MISS).all(1)), ahead=8)
    try:
        ev = track_evidence(cam, tracks, imgs, cfg)
    finally:
        if lazy is not None:
            lazy.close()
    return tracks, ev, auto_label(cam, tracks, ev, cfg)


def run_labels(take: Take, cams: list[str] | None = None, cfg: DancersConfig | None = None,
               typesafe: bool = False) -> dict:
    """Steps 1-2 for every camera -> dancers/labels.json + labels_review.jpg (+ candidates/<cam>.npz)."""
    cfg = cfg or DancersConfig()
    work = take.work
    out = work / "dancers"
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    cams = cams or [s.id for s in take.sources]
    path = out / "labels.json"
    existed = path.exists()
    labels = json.loads(path.read_text()) if existed else {}
    tiles = []
    for c in cams:
        cam = Cam.load(work, c)
        tracks, ev, lab = labels_for_cam(cam, cfg)
        old = labels.get(c, {})
        if "typesafe" in old and not typesafe:  # keep the stored cross-check while it still describes this label
            if (old.get("frame") == lab["frame"] and old.get("lead_box") == lab["lead_box"]
                    and old.get("follow_box") == lab["follow_box"]):
                lab["typesafe"] = old["typesafe"]
            else:
                print(f"[dancers] {c}: label changed - stored TypeSafe cross-check dropped (rerun with --typesafe)")
        labels[c] = lab
        np.savez_compressed(out / "candidates" / f"{c}.npz", idx=tracks["idx"], boxes=tracks["boxes"],
                            llr=ev["llr"], a_is_follow=ev["a_is_follow"], swap_post=ev["swap_post"],
                            pair_iou=ev["pair_iou"])
        # per camera, so that a later camera's failure keeps this label: label_<cam>.json (front_pod.label_one's file) always,
        # labels.json only when it existed (it is the stage's done marker: a first run writes it once, complete)
        (out / f"label_{c}.json").write_text(json.dumps(lab, indent=1))
        if existed:
            path.write_text(json.dumps(labels, indent=1))
        print(f"[dancers] {c}: label frame {lab['frame']} conf {lab['confidence']:.2f} (llr {lab['frame_llr']:.1f}, "
              f"agreement {lab['evidence_agreement']:.2f}, track switches {lab['track_switches']}) "
              f"cues {lab['cues']}", flush=True)
    if typesafe:
        typesafe_crosscheck(labels, cams)
    path.write_text(json.dumps(labels, indent=1))
    final = load_labels(work)
    for c in sorted(final):  # the review sheet always shows every labelled camera, not just the rerun ones
        cam = Cam.load(work, c)
        tiles.append(review_tile(cam, final[c]))
    from .dancers_qa import grid
    cv2.imwrite(str(out / "labels_review.jpg"), grid(tiles, cols=4), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return final


def typesafe_crosscheck(labels: dict, cams: list[str]) -> None:
    """Optional: ask TypeSafe to SELECT which candidate (A = left, B = right box) matches the lead's
    clothing, given only code-measured colour descriptors; record agreement with the colour rule."""
    from .judgments import judge_roles
    query, lead_letter = {}, {}
    for c in cams:
        lab = labels[c]
        left_is_lead = (lab["lead_box"][0] + lab["lead_box"][2]) <= (lab["follow_box"][0] + lab["follow_box"][2])
        order = ("lead", "follow") if left_is_lead else ("follow", "lead")
        query[c] = {letter: {**lab["colour_names"][r], **{k: lab["descriptors"][r].get(k) for k in
                                                          ("torso_v", "leg_v", "red")}}
                    for letter, r in zip("AB", order)}
        lead_letter[c] = "A" if left_is_lead else "B"
    try:
        js = judge_roles(query)
    except Exception as e:  # optional cross-check only
        print(f"[dancers] TypeSafe cross-check unavailable: {e}")
        return
    for c, (choice, p) in js.items():
        labels[c]["typesafe"] = {"choice": choice, "confidence": round(p, 3),
                                 "agrees_with_colour_rule": choice == lead_letter[c]}
        print(f"[dancers] {c}: TypeSafe lead = {choice} (colour rule {lead_letter[c]}), p={p:.2f}")


def review_tile(cam: Cam, lab: dict, height: int = 720) -> np.ndarray:
    from .dancers_qa import LEAD, FOLLOW, label_text
    img = cv2.imread(str(cam.files[lab["frame"]]))
    s = height / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * s), height), interpolation=cv2.INTER_AREA)
    for role, col in (("lead", LEAD), ("follow", FOLLOW)):
        b = (np.array(lab[f"{role}_box"]) * s).astype(int)
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), col, 3)
        n = lab["colour_names"][role]
        org = (int(b[0]), max(26, int(b[1]) - 8)) if role == "follow" else (int(b[0]), min(height - 60, int(b[3]) + 26))
        label_text(img, role.upper(), org, col, 0.8)  # follow above her box, lead below his: no overlap
        label_text(img, f"top {n['top']}, legs {n['trousers']}, {n['red_above_shoulders']}",
                   (6, height - (52 if role == "lead" else 26)), col, 0.45)
    by = lab["by"] + (" (SWAPPED)" if lab.get("auto_swapped_by_override") else "")
    label_text(img, f"cam {cam.id}  #{lab['frame']}  conf {lab['confidence']:.2f}  {by}", (6, 26), (255, 255, 255), 0.6)
    ts = lab.get("typesafe")
    if ts:
        label_text(img, f"TypeSafe: {'agrees' if ts['agrees_with_colour_rule'] else 'DISAGREES'} ({ts['confidence']:.2f})",
                   (6, 52), (255, 255, 255), 0.5)
    return img


# ---------------------------------------------------------------------------
# step 3-4: SAM propagation, identity verification
# ---------------------------------------------------------------------------

def frame_role_boxes(cam: Cam, tracks: dict, ev: dict, cfg: DancersConfig, label_swapped: bool) -> dict[int, dict]:
    """Detections usable as re-anchor prompts: clean frames where the per-frame colour log-odds is
    confident AND agrees with the smoothed track labelling. {frame: {'lead': box, 'follow': box}}"""
    w, h = cam.wh
    out = {}
    for i in range(len(ev["llr"])):
        ia, ib = tracks["idx"][i]
        if ia == MISS or ib == MISS or ev["pair_iou"][i] >= cfg.clean_iou:
            continue
        bx = tracks["boxes"][i]
        if (bx[:, 3] > 0.985 * h).any():
            continue
        signed = -ev["llr"][i] if ev["a_is_follow"][i] else ev["llr"][i]
        if signed < cfg.anchor_llr:
            continue
        rb = role_boxes(tracks, ev, i)
        if label_swapped:
            rb = {"lead": rb["follow"], "follow": rb["lead"]}
        out[i] = rb
    return out


def make_check(native_wh, work_wh, role_dets: dict[int, dict], cfg: DancersConfig):
    s = work_wh[0] / native_wh[0]

    def check(frame: int, probs: np.ndarray):
        det = role_dets.get(frame)
        if det is None:
            return None
        drift = False
        check.stats["checked"] += 1
        for k, role in enumerate(ROLES):
            mb = mask_box(probs[k] > 127)
            iou = 0.0 if mb is None else box_iou(mb / s, det[role])
            check.stats["min_iou"] = min(check.stats["min_iou"], round(iou, 3))
            check.stats["ious"].append(round(iou, 3))
            if iou < cfg.drift_iou:
                drift = True
        if not drift:
            return None
        check.stats["drifted"] += 1
        return {PID[r]: det[r] for r in ROLES}
    check.stats = {"candidates": len(role_dets), "checked": 0, "drifted": 0, "min_iou": 1.0, "ious": []}
    return check


def save_sam_cache(path: Path, key: np.ndarray, res: dict) -> None:
    """SAM output (working-res probabilities) so identity repair / post-processing can be re-run cheaply."""
    path.parent.mkdir(parents=True, exist_ok=True)
    anchors = np.array([[k, *v[1], *v[2]] for k, v in sorted(res["anchors"].items())], float)
    np.savez_compressed(path, key=key, probs=res["probs"], anchors=anchors, seconds=res["seconds"],
                        frame_evals=res["frame_evals"], work_wh=np.array(res["work_wh"]),
                        check_stats=json.dumps(res.get("check_stats", {})))


def sam_cache_valid(path: Path, key: np.ndarray) -> bool:
    """True when dancers/cache/<cam>_sam.npz exists and was made for this label (reads only the key)."""
    if not path.exists():
        return False
    with np.load(path) as d:
        k = d["key"]
        return k.shape == key.shape and bool(np.allclose(k, key))


def sam_cache_key(label: dict) -> np.ndarray:
    return np.array([int(label["frame"]), *label["lead_box"], *label["follow_box"]], float)


def load_sam_cache(path: Path, key: np.ndarray) -> dict | None:
    if not sam_cache_valid(path, key):
        return None
    d = np.load(path)
    anchors = {int(r[0]): {1: r[1:5], 2: r[5:9]} for r in d["anchors"]}
    return {"probs": d["probs"], "anchors": anchors, "seconds": float(d["seconds"]),
            "frame_evals": int(d["frame_evals"]), "work_wh": tuple(int(v) for v in d["work_wh"]),
            "check_stats": json.loads(str(d["check_stats"]))}


def match_kpts(cam: Cam, i: int, box: np.ndarray | None, mask: np.ndarray | None = None,
               other: np.ndarray | None = None, min_iou: float = 0.3, kp_conf: float = 0.3) -> np.ndarray | None:
    """Keypoints of the detection that best explains a dancer's mask (None if nothing fits).

    Without a mask: the detection whose box best overlaps `box` (IoU > min_iou). With the mask, a candidate
    is scored by box IoU + fraction of its confident keypoints inside the (slightly dilated) mask - fraction
    inside the partner's mask (`other`), and rejected when fewer than half of them land on the mask - in an
    embrace the partner's detection overlaps the box just as well but its keypoints sit on the partner."""
    if box is None:
        return None
    v = cam.people["valid"][i]
    if not v.any():
        return None
    boxes, kps = cam.people["boxes"][i][v], cam.people["kpts"][i][v]
    iou = iou_matrix(np.asarray(box, np.float32)[None], boxes)[0]
    if mask is None:
        j = int(iou.argmax())
        return kps[j] if iou[j] > min_iou else None
    h, w = mask.shape
    r = max(3, int(0.02 * (box[3] - box[1])))
    x0, y0 = max(0, int(box[0]) - r), max(0, int(box[1]) - r)  # dilate only around the mask (full-res cams)
    x1, y1 = min(w, int(np.ceil(box[2])) + r), min(h, int(np.ceil(box[3])) + r)
    own_roi = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    best, best_s = None, -np.inf
    for j in np.flatnonzero(iou > min_iou):
        ok = kps[j, :, 2] > kp_conf
        if ok.sum() < 3:
            continue
        x = np.clip(np.round(kps[j, ok, 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(kps[j, ok, 1]).astype(int), 0, h - 1)
        inr = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        own = np.zeros(len(x), bool)
        own[inr] = own_roi[y[inr] - y0, x[inr] - x0]
        inside = float(own.mean())
        if inside < 0.5:
            continue
        s = iou[j] + inside - (float((other[y, x] & ~own).mean()) if other is not None else 0.0)
        if s > best_s:
            best, best_s = j, s
    return None if best is None else kps[best]


def role_kpts(cam: Cam, masks: np.ndarray, fill: int = 0) -> list[list[np.ndarray | None]]:
    """[frame][role] keypoints matched to the final lead/follow masks (F, 2, H, W). With fill > 0 a frame
    without a match borrows the nearest matched frame within `fill` frames, shifted by the mask-box motion."""
    f = len(masks)
    out = [[match_kpts(cam, i, mask_box(masks[i, k]), masks[i, k], masks[i, 1 - k]) for k in range(2)]
           for i in range(f)]
    if fill <= 0:
        return out
    filled = [list(r) for r in out]
    for i in range(f):
        for k in range(2):
            if out[i][k] is not None or not masks[i, k].any():
                continue
            for d in sorted(range(-fill, fill + 1), key=abs):
                j = i + d
                if d == 0 or not 0 <= j < f or out[j][k] is None:
                    continue
                bi, bj = mask_box(masks[i, k]), mask_box(masks[j, k])
                kp = out[j][k].copy()
                kp[:, 0] += (bi[0] + bi[2] - bj[0] - bj[2]) / 2  # top-centre shift of the mask box
                kp[:, 1] += bi[1] - bj[1]
                filled[i][k] = kp
                break
    return filled


def identity_evidence(cam: Cam, masks: np.ndarray, imgs: list[np.ndarray], cfg: DancersConfig,
                      anchors=None, label_desc: dict | None = None) -> dict:
    """masks (F, 2, H, W) bool native (obj1, obj2). Per-frame descriptors, embrace flags and log-odds that
    obj1 is the lead, scored against per-role galleries anchored on the VERIFIED frames: medians over
    non-embrace frames within +-1 s (grown to 2, 4 s until gallery_min frames) of the label / re-anchor
    frames, where SAM's objects were just prompted with the verified role boxes. A cue with too few such
    frames uses the label frame's own descriptors (labels.json). Never SAM's majority labelling: a camera
    whose SAM identities are swapped for most of the window must still be oriented by its anchor.
    gallery_majority (all non-embrace frames, SAM's labelling) is returned only as a diagnostic."""
    f = len(masks)
    area = masks.reshape(f, 2, -1).sum(-1).astype(float)
    inter = (masks[:, 0] & masks[:, 1]).reshape(f, -1).sum(-1)
    union = (masks[:, 0] | masks[:, 1]).reshape(f, -1).sum(-1)
    iou = inter / np.maximum(union, 1)
    win = max(3, int(round(cam.fps)))
    med = np.array([[np.median(area[max(0, i - win):i + win + 1, k]) for k in range(2)] for i in range(f)])
    collapse = (area < cfg.area_collapse * np.maximum(med, 1)).any(1)
    embrace = (iou > cfg.embrace_iou) | collapse
    desc = []
    for i in range(f):
        hsv = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2HSV)
        row = []
        for k in range(2):
            b = mask_box(masks[i, k])
            if b is None:
                row.append(None)
                continue
            row.append(descriptors(hsv, b, match_kpts(cam, i, b, masks[i, k], masks[i, 1 - k]),
                                   mask=masks[i, k], exclude=masks[i, 1 - k]))
        desc.append(row)
    anchors = sorted(int(a) for a in (anchors or ()))
    dist = (np.min(np.abs(np.arange(f)[:, None] - np.array(anchors)[None]), 1) if anchors
            else np.full(f, np.inf))

    def cue_vals(k, cue, sel):
        return [d[k][cue] for i, d in enumerate(desc) if sel[i] and not embrace[i] and d[k] is not None
                and np.isfinite(d[k][cue]) and d[k][_n(cue)] >= cfg.rule.min_px]

    gal, gal_src, gal_major = {}, {}, {}
    for k, role in enumerate(ROLES):
        g, src, gm = {}, {}, {}
        for cue in CUES:
            g[cue], src[cue] = float("nan"), "none"
            for w in cfg.gallery_windows:
                vals = cue_vals(k, cue, dist <= w * cam.fps)
                if len(vals) >= cfg.gallery_min:
                    g[cue], src[cue] = float(np.median(vals)), f"anchors+-{w:g}s"
                    break
            else:
                lv = (label_desc or {}).get(role, {}).get(cue)
                if lv is not None and np.isfinite(lv):
                    g[cue], src[cue] = float(lv), "label"
            allv = cue_vals(k, cue, np.ones(f, bool))
            gm[cue] = float(np.median(allv)) if allv else float("nan")
        gal[role], gal_src[role], gal_major[role] = g, src, gm
    scale = {"torso_v": cfg.rule.torso_v_scale, "leg_v": cfg.rule.leg_v_scale, "red": cfg.rule.red_scale}
    llr = np.zeros(f)
    for i, d in enumerate(desc):
        if embrace[i] or d[0] is None or d[1] is None:
            continue
        tot = 0.0
        for cue in CUES:
            gl, gf = gal["lead"][cue], gal["follow"][cue]
            x1, x2 = d[0][cue], d[1][cue]
            if not all(np.isfinite([gl, gf, x1, x2])) or min(d[0][_n(cue)], d[1][_n(cue)]) < cfg.rule.min_px:
                continue
            sep = abs(gl - gf) / scale[cue]
            if sep < 0.5:  # this cue does not separate the two dancers in this camera
                continue
            keep = abs(x1 - gl) + abs(x2 - gf)
            swap = abs(x1 - gf) + abs(x2 - gl)
            tot += np.clip((swap - keep) / scale[cue], -cfg.rule.clip, cfg.rule.clip)
        llr[i] = tot
    return {"llr": llr, "embrace": embrace, "iou": iou, "area": area, "desc": desc, "gallery": gal,
            "gallery_source": gal_src, "gallery_majority": gal_major}


def identity_flags(llr: np.ndarray, anchors, clip: float, switch_frames: float) -> list[str]:
    """Warnings when SAM's majority labelling contradicts the verified anchors (the repair is pinned to the
    anchors, so these frames are relabelled to agree with them - but a human should look)."""
    flags = []
    ev = llr != 0
    if ev.any() and (llr[ev] < 0).sum() > (llr[ev] > 0).sum():
        flags.append(f"most identity evidence ({int((llr[ev] < 0).sum())}/{int(ev.sum())} frames) says SAM's "
                     "objects are swapped relative to the verified anchor(s)")
    free, _ = keep_swap_viterbi(llr, clip, switch_frames)
    bad = [int(a) for a in anchors if 0 <= a < len(free) and free[a]]
    if bad:
        flags.append(f"unpinned repair would have swapped the verified anchor frame(s) {bad}")
    return flags


def _n(cue: str) -> str:
    return {"torso_v": "torso_n", "leg_v": "legs_n", "red": "head_n"}[cue]


def segments(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] (inclusive) runs of True."""
    out, start = [], None
    for i, v in enumerate(list(flags) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    return out


TYPICAL_BINS = (8, 16, 16)


def _colour_typical(lab: np.ndarray, own: np.ndarray, blob: np.ndarray, thr: float = 3e-4) -> float:
    """Fraction of the blob's pixels whose Lab colour is common in the dancer's own major parts this frame."""
    from scipy.ndimage import gaussian_filter
    b = TYPICAL_BINS

    def idx(px):
        return tuple((px[..., c].astype(np.int32) * b[c]) >> 8 for c in range(3))
    hist = np.zeros(b)
    np.add.at(hist, idx(lab[own]), 1)
    dens = gaussian_filter(hist, 1.0) / max(hist.sum(), 1)
    return float((dens[idx(lab[blob])] > thr).mean())


def clean_mask(m: np.ndarray, max_frac: float = 0.02, gap: float = 0.12, ref_area: float | None = None,
               keep_pts: np.ndarray | None = None, keep_radius: float = 0.06, partner: np.ndarray | None = None,
               lab: np.ndarray | None = None, typical: float = 0.8) -> tuple[np.ndarray, int]:
    """Drop small detached blobs (a lit sign behind a head, a speck on a spectator): components smaller than
    max_frac x max(object area, ref_area) that lie farther than gap x object height from its major parts
    (>= 5 % of the current area). ref_area = the object's running-median area, so a mostly occluded dancer
    (small current area) does not make a background blob look big. Kept anyway, because a hand or foot cut
    off by the partner's occlusion is not a speck:
      * blobs within keep_radius x object height of a keep_pts point (the dancer's wrists/elbows/knees/ankles);
      * blobs touching the `partner` mask (occlusion boundary) - unless `lab` (the frame in Lab) is given and
        less than `typical` of the blob's pixels have colours common in the dancer's own major parts this
        frame (the cam05 exit sign touching the follow's hair is light green; his hands are skin-coloured).
    Returns (mask, removed pixel count)."""
    n, comp, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m, 0
    area = st[1:, cv2.CC_STAT_AREA]
    total = area.sum()
    small = np.flatnonzero(area < max_frac * max(total, ref_area or 0)) + 1
    if not len(small):
        return m, 0
    major = np.isin(comp, np.flatnonzero(area >= 0.05 * total) + 1)
    if not major.any():
        return m, 0
    ys = np.nonzero(major.any(1))[0]
    height = ys[-1] - ys[0] + 1
    dist = cv2.distanceTransform((~major).astype(np.uint8), cv2.DIST_L2, 3)
    limit = gap * height
    near_kp = np.zeros_like(m, dtype=np.uint8)
    for x, y in (keep_pts if keep_pts is not None else ()):
        cv2.circle(near_kp, (int(round(x)), int(round(y))), max(2, int(keep_radius * height)), 1, -1)
    touch = (cv2.dilate(partner.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) if partner is not None else None
    drop = []
    for j in small:
        blob = comp == j
        if dist[blob].min() <= limit or near_kp[blob].any():
            continue
        if touch is not None and touch[blob].any() and (lab is None or _colour_typical(lab, major, blob) >= typical):
            continue
        drop.append(j)
    if not drop:
        return m, 0
    rm = np.isin(comp, drop)
    return m & ~rm, int(rm.sum())


# step 5 hair / body split: dancers_hair.split_hair (keypoint-anchored; raised arms stay body)

ARM_KP = (7, 8, 9, 10)  # COCO elbows + wrists
LIMB_KP = (7, 8, 9, 10, 13, 14, 15, 16)  # + knees, ankles


def running_median(x: np.ndarray, win: int) -> np.ndarray:
    """(F, K) -> per-column median over [i - win, i + win]."""
    return np.array([np.median(x[max(0, i - win):i + win + 1], 0) for i in range(len(x))])


def arm_points(kp: np.ndarray | None, conf: float = 0.3, joints=LIMB_KP) -> np.ndarray:
    if kp is None:
        return np.zeros((0, 2))
    a = kp[list(joints)]
    return a[a[:, 2] > conf, :2]


def qa_scores(follow: np.ndarray, hair: np.ndarray, kp: np.ndarray | None, cfg: HairConfig) -> dict:
    """Per-frame picks for the body/hair review sheet: 'raised' = how far her highest wrist is above her
    head centre (head sizes), 'whip' = horizontal reach of her hair from the head centre (head sizes)."""
    out = {"raised": float("-inf"), "whip": 0.0}
    if not follow.any():
        return out
    ys = np.nonzero(follow.any(1))[0]
    g = head_geom(kp, float(ys[-1] - ys[0] + 1), cfg)
    if g is None:
        return out
    h, w = follow.shape
    for j in (9, 10):
        x, y, c = kp[j]
        if c > cfg.kp_conf and 0 <= int(y) < h and 0 <= int(x) < w:
            out["raised"] = max(out["raised"], float((g.centre[1] - y) / g.size))
    if hair.any():
        out["whip"] = float(np.abs(np.nonzero(hair)[1] - g.centre[0]).max() / g.size)
    return out


# ---------------------------------------------------------------------------
# per-camera driver
# ---------------------------------------------------------------------------

def run_camera(cam: Cam, label: dict, predictor, cfg: DancersConfig, log=print, reuse_sam: bool = False) -> dict:
    """Steps 3-5 for one camera. `predictor` is a SAM2 video predictor, a zero-argument callable returning
    one (built lazily, only when the cache misses), or None (cache required)."""
    from .dancers_qa import contact_sheet, crop_tile, overlay, write_video
    from .dancers_sam import propagate, work_size
    out = cam.work / "dancers" / cam.id
    for sub in ("masks", "body", "hair"):
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*.png"):
            p.unlink()
    files, (w, h) = cam.files, cam.wh
    f = len(files)
    t_start = time.time()
    imgs = read_frames(files)
    tracks, ev, auto = labels_for_cam(cam, cfg, imgs)
    swapped = bool(label.get("auto_swapped_by_override"))
    role_dets = frame_role_boxes(cam, tracks, ev, cfg, swapped)
    a = int(label["frame"])
    role_dets[a] = {"lead": np.array(label["lead_box"]), "follow": np.array(label["follow_box"])}
    ww = work_size(w, h)
    check = make_check((w, h), ww, role_dets, cfg)
    cache = cam.work / "dancers" / "cache" / f"{cam.id}_sam.npz"
    key = sam_cache_key(label)
    res = load_sam_cache(cache, key) if reuse_sam else None
    if res is None:
        if predictor is None:
            raise RuntimeError(f"cam {cam.id}: no reusable SAM cache for this label")
        if callable(predictor) and not hasattr(predictor, "propagate_in_video"):
            predictor = predictor()
        res = propagate(predictor, files, (w, h), a, {1: np.array(label["lead_box"]), 2: np.array(label["follow_box"])},
                        check=check, min_gap=cfg.reanchor_gap, log=log)
        res["check_stats"] = dict(check.stats)
        save_sam_cache(cache, key, res)
    else:
        log(f"[dancers]   {cam.id}: reusing SAM propagation from {cache.name}")
    t_sam = res["seconds"]
    # native-resolution probabilities -> raw masks (overlap kept for the embrace test)
    masks = np.zeros((f, 2, h, w), bool)
    for i in range(f):
        for k in range(2):
            p = res["probs"][i, k]
            if p.shape != (h, w):
                p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
            masks[i, k] = p > 127
    conf = np.array([[float(res["probs"][i, k][res["probs"][i, k] > 127].mean() / 255)
                      if (res["probs"][i, k] > 127).any() else 0.0 for k in range(2)] for i in range(f)])
    # step 4: identity verification + repair, anchored on the verified frames (label + re-anchors)
    anchors = sorted(int(k) for k in res["anchors"])
    ide = identity_evidence(cam, masks, imgs, cfg, anchors, label.get("descriptors"))
    state, post = keep_swap_viterbi(ide["llr"], cfg.llr_clip, cfg.switch_frames, pinned=anchors)
    flags = identity_flags(ide["llr"], anchors, cfg.llr_clip, cfg.switch_frames)
    for msg in flags:
        log(f"[dancers]   ERROR cam {cam.id}: {msg} - repair pinned to the anchors; check qa/{cam.id}_dense.jpg")
    swaps = segments(state.astype(bool))
    for s0, s1 in swaps:
        masks[s0:s1 + 1] = masks[s0:s1 + 1, ::-1]
        conf[s0:s1 + 1] = conf[s0:s1 + 1, ::-1]
    # overlap resolution: a pixel claimed by both goes to the more confident object
    both = masks[:, 0] & masks[:, 1]
    if both.any():
        for i in np.flatnonzero(both.reshape(f, -1).any(1)):
            p = res["probs"][i].astype(np.int16)
            if state[i]:
                p = p[::-1]
            if p.shape[1:] != (h, w):
                p = np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR) for x in p]).astype(np.int16)
            lead_wins = p[0] >= p[1]
            masks[i, 0] &= ~(both[i] & ~lead_wins)
            masks[i, 1] &= ~(both[i] & lead_wins)
    # keypoints of each final role mask (embrace-safe matching); detached blobs relative to the running median
    kps = role_kpts(cam, masks, fill=cfg.kp_fill)
    med_area = running_median(masks.reshape(f, 2, -1).sum(-1).astype(float), max(3, int(round(cam.fps))))
    stray_px = np.zeros(2, int)
    for i in range(f):
        lab_img = None
        for k in range(2):
            if lab_img is None:
                lab_img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2LAB)
            masks[i, k], n = clean_mask(masks[i, k], cfg.stray_frac, ref_area=med_area[i, k],
                                        keep_pts=arm_points(kps[i][k]), partner=masks[i, 1 - k], lab=lab_img)
            stray_px[k] += n
    # follow hair colour model for this camera (pooled over frames)
    colour = HairColour(cfg.hair)
    for i in range(0, f, cfg.hair.seed_stride):
        fol = masks[i, 1]
        if fol.any():
            ys = np.nonzero(fol.any(1))[0]
            colour.add_frame(imgs[i], fol, head_geom(kps[i][1], float(ys[-1] - ys[0] + 1), cfg.hair), kps[i][1])
    colour_ok = colour.fit()
    log(f"[dancers]   {cam.id}: hair colour model {'fitted' if colour_ok else 'NOT fitted (fixed red rule)'} "
        f"seeds {colour.seeds}")
    # step 5: write masks, hair/body, QA
    frames_out, tight = [], []
    qa_frames = []
    sheet_idx, dense_idx = sheet_frames(cam.t), sheet_frames(cam.t, 0.25)
    sheet_tiles, dense_tiles = [], []
    picks: dict[str, list] = {"raised": [], "whip": []}
    for i in range(f):
        lead, fol = masks[i, 0], masks[i, 1]
        hair, body_f = split_hair(imgs[i], fol, kps[i][1], colour, cfg.hair)
        qa = qa_scores(fol, hair, kps[i][1], cfg.hair)
        picks["raised"].append((qa["raised"], i))
        picks["whip"].append((qa["whip"], i))
        for role, m in (("lead", lead), ("follow", fol)):
            cv2.imwrite(str(out / "masks" / f"mask_{i:04d}_{PID[role]:02d}.png"), m.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "body" / f"mask_{i:04d}_01.png"), lead.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "body" / f"mask_{i:04d}_02.png"), body_f.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "hair" / f"mask_{i:04d}_02.png"), hair.astype(np.uint8) * 255)
        near = tracks["boxes"][max(0, i - 3):i + 4].reshape(-1, 4)  # YOLO misses a dancer now and then
        tight.append(tightness(lead, fol, near, cam, i, cfg))
        boxes = {r: mask_box(m) for r, m in (("lead", lead), ("follow", fol))}
        frames_out.append({"i": i, "t": round(float(cam.t[i]), 4),
                           **{r: {"box": None if boxes[r] is None else [round(float(v), 1) for v in boxes[r]],
                                  "conf": round(float(conf[i, k]), 3)} for k, r in enumerate(ROLES)},
                           # swap_prob: posterior that the WRITTEN masks are swapped (after repair);
                           # sam_swap_prob: that SAM's raw objects were
                           "swap_prob": round(float(1 - post[i] if state[i] else post[i]), 4),
                           "sam_swap_prob": round(float(post[i]), 4), "embrace": bool(ide["embrace"][i]),
                           "repaired": bool(state[i]), "anchor": i in res["anchors"]})
        vis = overlay(imgs[i], lead, fol, hair, cfg.qa_height, f"cam {cam.id} #{i} t={cam.t[i]:.2f}"
                      + ("  EMBRACE" if ide["embrace"][i] else "") + ("  ANCHOR" if i in res["anchors"] else ""))
        qa_frames.append(vis)
        if i in sheet_idx or i in dense_idx:
            tile = crop_tile(imgs[i], lead, fol, hair, f"#{i} t={cam.t[i]:.2f}" + (" E" if ide["embrace"][i] else "")
                             + (" R" if state[i] else ""))
            if i in sheet_idx:
                sheet_tiles.append(tile)
            if i in dense_idx:
                dense_tiles.append(tile)
    (cam.work / "dancers" / "qa").mkdir(parents=True, exist_ok=True)
    write_video(cam.work / "dancers" / "qa" / f"{cam.id}.mp4", qa_frames, cam.fps)
    head = f"cam {cam.id}: lead = red, follow = blue, follow hair = yellow outline (E = embrace, R = repaired)"
    cv2.imwrite(str(cam.work / "dancers" / "qa" / f"{cam.id}_sheet.jpg"), contact_sheet(sheet_tiles, head),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(cam.work / "dancers" / "qa" / f"{cam.id}_dense.jpg"), contact_sheet(dense_tiles, head, cols=8),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    tight_ok = np.array([t["ok"] for t in tight])
    total = time.time() - t_start
    rep = {"frames": f, "work_wh": list(ww), "anchor": a, "anchors": sorted(int(k) for k in res["anchors"]),
           "sam_seconds": round(t_sam, 1), "sam_s_per_frame": round(t_sam / f, 3),
           "frame_evals": res["frame_evals"], "total_seconds": round(total, 1),
           "total_s_per_frame": round(total / f, 3), "embrace_frames": int(ide["embrace"].sum()),
           "swaps_repaired": [[int(s0), int(s1)] for s0, s1 in swaps],
           "evidence_frames": int((ide["llr"] != 0).sum()),
           "evidence_agreement": round(float((ide["llr"][ide["llr"] != 0] > 0).mean()) if (ide["llr"] != 0).any() else 0.0, 3),
           "tight_rate": round(float(tight_ok.mean()), 4), "untight_frames": [int(i) for i in np.flatnonzero(~tight_ok)],
           "empty_frames": {r: int(sum(1 for fr in frames_out if fr[r]["box"] is None)) for r in ROLES},
           "gallery": {r: _round(g) for r, g in ide["gallery"].items()},
           "gallery_source": ide["gallery_source"],
           "gallery_majority": {r: _round(g) for r, g in ide["gallery_majority"].items()},
           "identity_flags": flags,
           "hair_model": {"fitted": colour_ok, "seeds": colour.seeds},
           "qa_picks": {k: top_spaced(lst) for k, lst in picks.items()},
           "reanchor_checks": {k: v for k, v in res.get("check_stats", {}).items() if k != "ious"},
           "stray_px_removed": {r: int(stray_px[k]) for k, r in enumerate(ROLES)}}
    return {"frames": frames_out, "report": rep}


def top_spaced(scored: list[tuple[float, int]], n: int = 4, gap: int = 10) -> list[list]:
    """Best-scoring frames at least `gap` frames apart: [[score, frame], ...]."""
    out: list[list] = []
    for v, i in sorted(scored, reverse=True):
        if not np.isfinite(v):
            continue
        if all(abs(i - j) >= gap for _, j in out):
            out.append([round(float(v), 3), int(i)])
        if len(out) == n:
            break
    return out


def tightness(lead: np.ndarray, fol: np.ndarray, pair_boxes: np.ndarray, cam: Cam, i: int,
              cfg: DancersConfig) -> dict:
    """QA proxy for 'masks cover the dancers tightly, no spectators merged': the fraction of mask pixels
    outside the dancers' detection boxes (grown by tight_margin) must stay small. Reference boxes = the
    candidate track boxes plus every detection overlapping a mask's own box by IoU > 0.3."""
    h, w = lead.shape
    allowed = np.zeros((h, w), bool)
    refs = [b for b in pair_boxes if np.isfinite(b).all()]
    v = cam.people["valid"][i]
    for m in (lead, fol):
        mb = mask_box(m)
        if mb is not None and v.any():
            iou = iou_matrix(mb[None], cam.people["boxes"][i][v])[0]
            refs += list(cam.people["boxes"][i][v][iou > 0.3])
    for b in refs:
        bw, bh = b[2] - b[0], b[3] - b[1]
        x0, y0 = int(max(0, b[0] - cfg.tight_margin * bw)), int(max(0, b[1] - cfg.tight_margin * bh))
        x1, y1 = int(min(w, b[2] + cfg.tight_margin * bw)), int(min(h, b[3] + cfg.tight_margin * bh))
        allowed[y0:y1, x0:x1] = True
    out = {}
    for r, m in (("lead", lead), ("follow", fol)):
        n = m.sum()
        out[r] = float((m & ~allowed).sum() / n) if n else 0.0
    empty = not lead.any() or not fol.any()
    out["ok"] = (not empty) and max(out["lead"], out["follow"]) <= cfg.tight_max_out
    return out


def sheet_frames(t: np.ndarray, step: float = 1.0) -> set[int]:
    """Frame nearest each multiple of `step` reference seconds within the window (+-0.1 s tolerance at the
    ends): step 1.0 = the '1-second contact sheet'."""
    secs = np.arange(np.ceil((t[0] - 0.1) / step), np.floor((t[-1] + 0.1) / step) + 1) * step
    return {int(np.argmin(np.abs(t - s))) for s in secs}


def run(take: Take, cams: list[str] | None = None, cfg: DancersConfig | None = None, reuse_sam: bool = False) -> dict:
    """Steps 3-5 for every camera. The GPU lock is taken and SAM 2.1 built only on the first camera whose
    cache cannot be reused (with --reuse-sam and valid caches, nothing touches the GPU)."""
    import contextlib

    from . import gpu
    from .dancers_sam import build_predictor
    cfg = cfg or DancersConfig()
    work = take.work
    out = work / "dancers"
    cams = cams or [s.id for s in take.sources]
    if not (out / "labels.json").exists() or not set(cams) <= set(json.loads((out / "labels.json").read_text())):
        run_labels(take, cams, cfg)
    labels = load_labels(work)
    roles_path = out / "roles.json"
    roles = json.loads(roles_path.read_text()) if roles_path.exists() else {"cams": {}}
    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"cams": {}}
    with contextlib.ExitStack() as stack:
        held: dict = {}

        def predictor():
            if "p" not in held:
                stack.enter_context(gpu.exclusive(f"dancers {take.name}"))
                held["p"] = build_predictor()
                report["sam_overrides"] = getattr(held["p"], "dancecap_overrides", None)
            return held["p"]

        for c in cams:
            cam = Cam.load(work, c)
            r = run_camera(cam, labels[c], predictor, cfg, reuse_sam=reuse_sam)
            roles["cams"][c] = {"frames": r["frames"]}
            report["cams"][c] = r["report"]
            rp = r["report"]
            print(f"[dancers] {c}: {rp['frames']} frames, SAM {rp['sam_s_per_frame']:.2f} s/frame "
                  f"(total {rp['total_s_per_frame']:.2f}), anchors {rp['anchors']}, embrace {rp['embrace_frames']}, "
                  f"swaps repaired {rp['swaps_repaired']}, tight {100 * rp['tight_rate']:.1f}%"
                  + (f", IDENTITY FLAGS {rp['identity_flags']}" if rp["identity_flags"] else ""), flush=True)
            roles["labels_source"] = {k: labels[k]["by"] for k in labels}
            roles["swaps_repaired"] = [{"cam": k, "start": s0, "end": s1}
                                       for k, v in sorted(report["cams"].items()) for s0, s1 in v["swaps_repaired"]]
            roles["identity_flags"] = {k: v.get("identity_flags", []) for k, v in sorted(report["cams"].items())
                                       if v.get("identity_flags")}
            roles_path.write_text(json.dumps(roles))
            report_path.write_text(json.dumps(report, indent=1))
        if not held:
            print("[dancers] all SAM caches reused - GPU not used")
    body_hair_review(take, report, cfg)
    return report


def body_hair_review(take: Take, report: dict, cfg: DancersConfig, per_kind: int = 6) -> Path | None:
    """review/dancers_body_hair.jpg: full | body | hair of the follow on the frames where her wrists are
    highest above her head ('raised') and her hair reaches farthest sideways ('hair whip'), best frame per
    camera, best cameras first."""
    from .dancers_qa import body_hair_row, contact_sheet
    rows = []
    for kind, name in (("raised", "raised arm"), ("whip", "hair whip")):
        best = []
        for c, rp in report.get("cams", {}).items():
            lst = rp.get("qa_picks", {}).get(kind) or []
            if lst and (kind != "raised" or lst[0][0] > -1.0):  # wrist above about shoulder height
                best.append((lst[0][0], c, lst[0][1]))
        for v, c, i in sorted(best, reverse=True)[:per_kind]:
            cam = Cam.load(take.work, c)
            d = take.work / "dancers" / c
            rd = lambda sub, pid: cv2.imread(str(d / sub / f"mask_{i:04d}_{pid:02d}.png"), cv2.IMREAD_GRAYSCALE) > 0
            img = cv2.imread(str(cam.files[i]))
            full = rd("masks", 2)
            kp = match_kpts(cam, i, mask_box(full), full, rd("masks", 1))
            rows.append(body_hair_row(img, rd("masks", 1), full, rd("body", 2), rd("hair", 2), kp,
                                      f"{name}  cam {c} #{i}  score {v:.2f}"))
    if not rows:
        return None
    path = take.work / "review" / "dancers_body_hair.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ("follow: FULL mask | BODY (full - hair) | HAIR   (green = her keypoints; score: raised = wrist height above "
            "head centre, whip = hair reach sideways, in head sizes)")
    cv2.imwrite(str(path), contact_sheet(rows, head, cols=2, max_width=3600), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[dancers] -> {path}")
    return path
