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
 4. identity verification: per-frame colour log-odds of the two MASKS against per-role galleries (medians
    over non-embrace frames), abstaining during embraces (mask IoU > 0.3 or area collapse); keep/swap
    Viterbi with a switching penalty worth ~5 frames of consistent evidence; swapped segments relabelled.
    Pixels claimed by both objects go to the more confident one; tiny detached blobs are dropped.
 5. outputs (CONTRACTS.md): masks/ (full), body/ (follow hair trimmed), hair/ (follow), roles.json,
    report.json, qa/<cam>.mp4 overlay videos, qa/<cam>_sheet.jpg (one couple-cropped tile per reference
    second) and qa/<cam>_dense.jpg (every 0.25 s).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .dancers_appearance import RuleConfig, colour_names, descriptors, lead_llr, red_hair_mask
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
    clean_iou: float = 0.1  # pair separated
    evidence_iou: float = 0.3  # boxes overlapping more than this: colour evidence abstains
    llr_clip: float = 2.0  # per-frame evidence cap for the keep/swap Viterbi
    switch_frames: float = 5.0  # switching costs ~this many frames of capped, consistent evidence
    anchor_llr: float = 2.5  # per-frame colour log-odds needed to re-anchor from a detection
    drift_iou: float = 0.5
    reanchor_gap: int = 8
    embrace_iou: float = 0.3
    area_collapse: float = 0.4  # area below this x the running median = collapsed (occluded / merged)
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


def keep_swap_viterbi(llr: np.ndarray, clip: float = 2.0, switch_frames: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Two-state Viterbi {keep=0, swap=1} with a switching penalty worth `switch_frames` frames of capped
    evidence; returns (states (F,) int, swap posterior (F,) from forward-backward with the same model)."""
    c = _state_costs(llr, clip)
    f = len(c)
    if f == 0:
        return np.zeros(0, int), np.zeros(0)
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
    p = cam.people
    tracks = candidate_tracks(p["boxes"], p["scores"], p["valid"], p["kpts"], *cam.wh, cam.fps, cfg.select)
    imgs = imgs if imgs is not None else read_frames(cam.files)
    ev = track_evidence(cam, tracks, imgs, cfg)
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
    labels = json.loads(path.read_text()) if path.exists() else {}
    tiles = []
    for c in cams:
        cam = Cam.load(work, c)
        tracks, ev, lab = labels_for_cam(cam, cfg)
        labels[c] = lab
        np.savez_compressed(out / "candidates" / f"{c}.npz", idx=tracks["idx"], boxes=tracks["boxes"],
                            llr=ev["llr"], a_is_follow=ev["a_is_follow"], swap_post=ev["swap_post"],
                            pair_iou=ev["pair_iou"])
        print(f"[dancers] {c}: label frame {lab['frame']} conf {lab['confidence']:.2f} (llr {lab['frame_llr']:.1f}, "
              f"agreement {lab['evidence_agreement']:.2f}, track switches {lab['track_switches']}) "
              f"cues {lab['cues']}", flush=True)
    if typesafe:
        typesafe_crosscheck(labels, cams)
    path.write_text(json.dumps(labels, indent=1))
    final = load_labels(work)
    for c in cams:
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


def load_sam_cache(path: Path, key: np.ndarray) -> dict | None:
    if not path.exists():
        return None
    d = np.load(path)
    if d["key"].shape != key.shape or not np.allclose(d["key"], key):
        return None
    anchors = {int(r[0]): {1: r[1:5], 2: r[5:9]} for r in d["anchors"]}
    return {"probs": d["probs"], "anchors": anchors, "seconds": float(d["seconds"]),
            "frame_evals": int(d["frame_evals"]), "work_wh": tuple(int(v) for v in d["work_wh"]),
            "check_stats": json.loads(str(d["check_stats"]))}


def match_kpts(cam: Cam, i: int, box: np.ndarray | None) -> np.ndarray | None:
    """Keypoints of the detection that best overlaps a mask box (None if nothing overlaps)."""
    if box is None:
        return None
    v = cam.people["valid"][i]
    if not v.any():
        return None
    iou = iou_matrix(box[None], cam.people["boxes"][i][v])[0]
    j = int(iou.argmax())
    return cam.people["kpts"][i][v][j] if iou[j] > 0.3 else None


def identity_evidence(cam: Cam, masks: np.ndarray, imgs: list[np.ndarray], cfg: DancersConfig) -> dict:
    """masks (F, 2, H, W) bool native (obj1, obj2). Per-frame descriptors, embrace flags and log-odds that
    obj1 is the lead, scored against per-role galleries built from clean frames."""
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
            row.append(descriptors(hsv, b, match_kpts(cam, i, b), mask=masks[i, k], exclude=masks[i, 1 - k]))
        desc.append(row)
    # galleries: per-role medians at non-embrace frames, assuming SAM's identities (obj1 = lead) hold mostly
    gal = {}
    for k, role in enumerate(ROLES):
        g = {}
        for cue in CUES:
            vals = [d[k][cue] for i, d in enumerate(desc) if not embrace[i] and d[k] is not None
                    and np.isfinite(d[k][cue]) and d[k][_n(cue)] >= cfg.rule.min_px]
            g[cue] = float(np.median(vals)) if vals else float("nan")
        gal[role] = g
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
    return {"llr": llr, "embrace": embrace, "iou": iou, "area": area, "desc": desc, "gallery": gal}


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


def clean_mask(m: np.ndarray, max_frac: float = 0.01, gap: float = 0.12) -> tuple[np.ndarray, int]:
    """Drop tiny detached blobs (a lit sign behind a head, a speck on a spectator): components smaller than
    max_frac of the object that lie farther than gap x object height from its major parts (>= 5 %).
    Returns (mask, removed pixel count). Limbs split off by the partner's occlusion are large or near."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m, 0
    area = st[1:, cv2.CC_STAT_AREA]
    total = area.sum()
    small = np.flatnonzero(area < max_frac * total) + 1
    if not len(small):
        return m, 0
    major = np.isin(lab, np.flatnonzero(area >= 0.05 * total) + 1)
    if not major.any():
        return m, 0
    ys = np.nonzero(major.any(1))[0]
    dist = cv2.distanceTransform((~major).astype(np.uint8), cv2.DIST_L2, 3)
    limit = gap * (ys[-1] - ys[0] + 1)
    drop = [j for j in small if dist[lab == j].min() > limit]
    if not drop:
        return m, 0
    rm = np.isin(lab, drop)
    return m & ~rm, int(rm.sum())


# ---------------------------------------------------------------------------
# step 5: hair / body split
# ---------------------------------------------------------------------------

def head_top(kpts: np.ndarray | None) -> float | None:
    """Image y of the top of the skull from COCO face + shoulder keypoints (None if not visible)."""
    if kpts is None:
        return None
    face = kpts[:5]
    ok = face[:, 2] > 0.3
    if ok.sum() < 2:
        return None
    sh = kpts[5:7]
    if (sh[:, 2] > 0.3).all():
        size = max(np.linalg.norm(sh[0, :2] - sh[1, :2]) * 0.6, 1.0)
    else:
        pts = face[ok, :2]
        size = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 4.0) * 1.5
    return float(face[ok, 1].min() - 0.75 * size)


def split_hair(img: np.ndarray, follow: np.ndarray, kpts: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(hair, body) for the follow: hair = red-hair pixels inside her mask, cleaned and kept only where
    connected to the head region; body = mask - hair - everything above the estimated head top."""
    if not follow.any():
        return follow.copy(), follow.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = (red_hair_mask(hsv) & follow).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    red &= follow.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    keep = np.zeros(n, bool)
    if n > 1:
        min_area = max(12, 0.002 * follow.sum())
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    hair = keep[lab]
    body = follow & ~hair
    top = head_top(kpts)
    if top is not None:
        y = int(max(0, np.floor(top)))
        above = np.zeros_like(body)
        above[:y] = True
        hair |= follow & above  # anything of hers above the skull is hair (flying)
        body &= ~above
    return hair, body


# ---------------------------------------------------------------------------
# per-camera driver
# ---------------------------------------------------------------------------

def run_camera(cam: Cam, label: dict, predictor, cfg: DancersConfig, log=print, reuse_sam: bool = False) -> dict:
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
    key = np.array([a, *label["lead_box"], *label["follow_box"]], float)
    res = load_sam_cache(cache, key) if reuse_sam else None
    if res is None:
        if predictor is None:
            raise RuntimeError(f"cam {cam.id}: no reusable SAM cache for this label")
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
    # step 4: identity verification + repair
    ide = identity_evidence(cam, masks, imgs, cfg)
    state, post = keep_swap_viterbi(ide["llr"], cfg.llr_clip, cfg.switch_frames)
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
    stray_px = np.zeros(2, int)
    for i in range(f):
        for k in range(2):
            masks[i, k], n = clean_mask(masks[i, k])
            stray_px[k] += n
    # step 5: write masks, hair/body, QA
    frames_out, tight = [], []
    qa_frames = []
    sheet_idx, dense_idx = sheet_frames(cam.t), sheet_frames(cam.t, 0.25)
    sheet_tiles, dense_tiles = [], []
    for i in range(f):
        lead, fol = masks[i, 0], masks[i, 1]
        kp_f = match_kpts(cam, i, mask_box(fol))
        hair, body_f = split_hair(imgs[i], fol, kp_f)
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
                           "swap_prob": round(float(post[i]), 4), "embrace": bool(ide["embrace"][i]),
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
           "reanchor_checks": {k: v for k, v in res.get("check_stats", {}).items() if k != "ious"},
           "stray_px_removed": {r: int(stray_px[k]) for k, r in enumerate(ROLES)}}
    return {"frames": frames_out, "report": rep}


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
    with gpu.exclusive(f"dancers {take.name}"):
        predictor = build_predictor()
        for c in cams:
            cam = Cam.load(work, c)
            r = run_camera(cam, labels[c], predictor, cfg, reuse_sam=reuse_sam)
            roles["cams"][c] = {"frames": r["frames"]}
            report["cams"][c] = r["report"]
            rp = r["report"]
            print(f"[dancers] {c}: {rp['frames']} frames, SAM {rp['sam_s_per_frame']:.2f} s/frame "
                  f"(total {rp['total_s_per_frame']:.2f}), anchors {rp['anchors']}, embrace {rp['embrace_frames']}, "
                  f"swaps repaired {rp['swaps_repaired']}, tight {100 * rp['tight_rate']:.1f}%", flush=True)
            roles["labels_source"] = {k: labels[k]["by"] for k in labels}
            roles["swaps_repaired"] = [{"cam": k, "start": s0, "end": s1}
                                       for k, v in sorted(report["cams"].items()) for s0, s1 in v["swaps_repaired"]]
            roles_path.write_text(json.dumps(roles))
            report_path.write_text(json.dumps(report, indent=1))
    return report
