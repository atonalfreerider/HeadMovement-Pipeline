"""Lead / follow identity seeded onto a take's OWN frames from another take's eye-checked people_mono teacher tracks: no colour rule, and
nothing decoded beyond one label frame and the review sheet frames per camera.

    python -m dancecap.dancers_seed run lucas_luiza --map 01=lucas_luiza_v:01:+81 --map 02=lucas_luiza_h:01:clock-14.000 --scan 3 --min-conf 0.3
    python -m dancecap.dancers_seed scan lucas_luiza --map 01=lucas_luiza_v:01:+81 [--scan 3]      # the frame-map check only; writes nothing
    ... run ... --drop 02:lead:5181-5221     # after review: frames where the source handed a role to somebody else stay unmatched
    ... run ... --keep 01:follow:900-960     # after review: the role IS right there (its across-gap flags count as reviewed)

Why: dancers-labels' colour rule (dancers_appearance) is larissa_kadu's (bright top, light jeans, red hair) and weak or of the wrong sign for
other couples. A monocular take of the same footage (lucas_luiza_h = the tripod, lucas_luiza_v = the Steadicam) already has people_mono teacher
tracks whose roles the TOML descriptions decided and the owner checked by eye (people/teachers_<cam>.npz, slot 0 = lead, 1 = follow), each on
its own cut clock and frame grid. The boxes and joints kept are this take's own detections (people/<cam>.npz) on its exact frames.

--map CAM=SRC_TAKE:SRC_CAM:RULE   camera CAM takes work/SRC_TAKE/people/teachers_SRC_CAM.npz (the same frame size); frame i of CAM is source frame j:
    +N | -N     j = i + N (the same video at the same rate, trimmed differently)
    clock-S     j = the source frame nearest t_i - S on the source's own t (a 30 fps cut, t = n/30: round((t_i - S) * 30)); clock+S: t_i + S
    j = -1 outside the source.

Per camera (numpy; npz members read one at a time, the source's kp_wb and masks never: < 300 MB). Slots are the REVIEWED roles throughout
(the source's, flipped where dancers/labels_override.json swaps the camera), so seed_report's role keys are the names --drop / --keep take:
    scan    median IoU of the source's teacher boxes (conf >= 0.5) against the best detection of people/<cam>.npz, every 3rd frame, at source
            offsets -scan..+scan around the rule (all reported; G0 wants the rule's own offset to be the best)
    match   per frame the two teacher boxes (pose conf >= --min-conf) against the valid detections: the best ONE-TO-ONE assignment, IoU > 0.5
            (exact over every pair: the Hungarian solution of the 2 x N problem). Only where the source flags `merged` (one box for the
            embraced pair) may a role that assignment left unmatched share the other role's detection; --drop stretches stay unmatched
    world   roles.json (cameras.world.dancer_keypoints triangulates the detection's COCO skeleton) keeps a role's box only when that skeleton is
            not closer to the OTHER role's source skeleton by 2x (median joint distance / box height; undecided where the two source skeletons
            lie on one body). A shared detection goes to the one role whose source skeleton it clearly is, else to neither
    label   the clean frame (both roles in roles.json, pair IoU 0 / >= 10 joints each, relaxed to 0.1 / 8 and 0.3 / 6; dancers.choose_label_frame
            starts at its clean_iou 0.1; no box at an image edge) with the highest min(conf); its descriptors, colour names and colour-rule cues
            are measured on that ONE frame (dancers_appearance), so front_pod labels-review, dancers.apply_override and the SAM route read it like a
            dancers-labels label

Writes (work/<take>/; npz and json through a .part file and os.replace):
    people/teachers_<cam>.npz         the recap teachers contract (CONTRACTS.md "Recap people"), reviewed slots: t (frames.json), frame_index
                                      (-1 until carlos_aline_inputs fills it), ok (the source covers the frame), row (F, 2) int16 = the detection of
                                      people/<cam>.npz (-1 none; both slots the same detection only in merged frames), box / kp = that detection's
                                      (NaN none), conf = the source's where matched (else 0), cost = 1 - IoU (NaN none), kind (1 matched, 0 none),
                                      nj (joints >= 0.5), gap 0, merged, role_conf (F, 2, the source's), lead_slot (F,) 0, floor NaN, size, attached,
                                      src_frame = j, src = the --map. A re-run on the same frames keeps the previous file's frame_index and each
                                      role-frame's floor whose detection is unchanged (carlos_aline_inputs --teachers-floor, runbook step 11)
    dancers/teacher_tracks/<cam>.npz  {idx = row, boxes, conf} in the SOURCE's slot order, like candidates/<cam>.npz: a recap_front.dancer_one-style
                                      consumer (slot A taken as the lead, then labels_override applied to its label) flips them once. Nothing on
                                      the planned two-view route reads them (the chunk takes get theirs from teachers_<cam>.npz, recap_front
                                      .step_people)
    dancers/roles.json                ONE entry per frames.json frame, in order (cameras.world.dancer_keypoints indexes it by position): lead /
                                      follow {box = the matched detection's after the skeleton check (None otherwise), conf}, swap_prob 0,
                                      sam_swap_prob 0, embrace = merged, repaired / anchor false; labels_source, swaps_repaired [] and identity_flags
                                      {} in dancers_pod.merge's schema (the other cameras' entries are kept). Note: the registry's dancers_sam
                                      (done_when roles.json) then reads done although no masks exist
    dancers/label_<cam>.json, labels.json   dancers-labels' schema in the SOURCE's slot order (load_labels applies labels_override): by
                                      "seed:<src take>", confidence = the source's role decision (identity.json), frame_llr = |colour-rule llr| on the
                                      label frame as dancers.auto_label (the signed one, positive when the rule agrees, is seed.colour_rule_llr;
                                      informational only), cues, evidence_agreement = the mean source role_conf of the matched role-frames,
                                      track_switches 0, seed provenance
    review/seed_roles.jpg             per camera 6 frames spread over the window, then up to 6 flagged frames (across-gap, jump), lead (red) /
                                      follow (blue) boxes as in roles.json
    dancers/seed_report.json          scan, match shares, skeleton vetoes, role jumps, label frame, gate G0 per camera

Gate G0 (seed_report.json, per camera; `run` and `scan` exit 2 when it fails, after writing): the rule's scan offset is the best one with
median IoU >= 0.85 (+-N rules: >= 0.02 above both neighbours); both roles in roles.json on >= 85 % of the frames (80 % for portrait framing,
whose edge crops lose a dancer more often); no detection taking both roles without `merged` (an invariant of match, checked on the written
rows); roles.json box-centre jumps > 25 % of the box height on <= 0.5 % of the consecutive frames with a box; no unreviewed across-gap flag
(a role box landing more than a body height away after an unmatched gap of up to 30 frames: another person may have taken the role in the
source; --drop the stretch or --keep it after looking). G0 cannot see a role handed over between consecutive frames inside merged frames or
a wrong person followed from the window start: the two-view check (runbook step 5) has to catch those.

dancers/labels_override.json {cam: {"swap": true}} (the human check of dancers/labels_review.jpg and review/seed_roles.jpg) is applied exactly
ONCE: re-run `run` and that camera's slots are flipped in everything that never passes dancers.load_labels (teachers_<cam>.npz, roles.json,
seed_roles.jpg, seed_report), while label_<cam>.json / labels.json and teacher_tracks keep the source's slot order, exactly as for
dancers-labels' own labels and candidate tracks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .dancers import ROLES, _round, load_labels
from .dancers_appearance import RuleConfig, colour_names, descriptors, lead_llr
from .take import Take

MAP_RE = re.compile(r"^(?P<cam>[^=]+)=(?P<take>[^:]+):(?P<src_cam>[^:]+):(?P<rule>[^:]+)$")
RULE_RE = re.compile(r"^(?:(?P<frames>[+-]\d+)|clock(?P<clock>[+-]\d+(?:\.\d*)?))$")


@dataclass
class SeedConfig:
    min_conf: float = 0.3            # teacher roles whose POSE confidence (people_mono pconf: joint scores x role_conf x visibility) is below
                                     # this are left out, as recap_front.teacher_slots does; it is not the identity confidence (role_conf)
    match_iou: float = 0.5           # a teacher box takes a detection above this IoU
    scan: int = 3                    # source offsets -scan..+scan around the rule
    scan_step: int = 3
    scan_conf: float = 0.5           # teacher boxes the scan trusts
    skel_joints: int = 4             # skeleton check (world_keep): joints seen in both skeletons, else it cannot decide
    skel_sep: float = 0.05           # ... nor when the two source skeletons are closer than this x the box height (both on one body)
    skel_ratio: float = 0.5          # a detection is the partner's when its skeleton is closer to the partner's by this factor
    sheet_frames: int = 6            # tiles per sheet row (spread frames; flagged frames)
    jump_frac: float = 0.25          # a role's box centre moving more than this x its box height between consecutive frames
    max_gap: int = 30                # across-gap flags: unmatched gaps up to this many frames
    g0_iou: float = 0.85
    g0_margin: float = 0.02          # +-N rules: the rule's median IoU above both neighbours by at least this
    g0_both: float = 0.85
    g0_both_portrait: float = 0.80   # portrait (Steadicam) framing crops feet and sides: lucas_luiza_v edge_share 0.53
    g0_jumps: float = 0.005


def log(msg: str) -> None:
    print(f"[dancers_seed] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ inputs


def parse_map(s: str) -> dict:
    """'01=lucas_luiza_v:01:+81' -> {cam, take, src_cam, rule, kind 'frames' | 'clock', shift (frames | seconds added to t)}."""
    m = MAP_RE.match(s.strip())
    r = RULE_RE.match(m["rule"]) if m else None
    if r is None:
        raise ValueError(f"--map {s!r}: want CAM=SRC_TAKE:SRC_CAM:RULE with RULE +N | -N | clock-S | clock+S")
    kind = "frames" if r["frames"] else "clock"
    return {"cam": m["cam"], "take": m["take"], "src_cam": m["src_cam"], "rule": m["rule"], "kind": kind,
            "shift": int(r["frames"]) if kind == "frames" else float(r["clock"])}


def parse_drop(s: str, flag: str = "--drop") -> tuple[str, str, int, int]:
    """'02:lead:5181-5221' -> ('02', 'lead', 5181, 5221): a stretch of target frames (inclusive) of a reviewed role (--drop, --keep)."""
    m = re.match(r"^([^:]+):(lead|follow):(\d+)-(\d+)$", s.strip())
    if m is None or int(m[3]) > int(m[4]):
        raise ValueError(f"{flag} {s!r}: want CAM:lead|follow:I0-I1 (target frames, inclusive)")
    return m[1], m[2], int(m[3]), int(m[4])


def load_cam(work: Path, cam: str) -> dict:
    """frames.json (t, size, files) + people/<cam>.npz boxes / valid / kpts of the target camera."""
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    with np.load(work / "people" / f"{cam}.npz") as z:
        out = {k: z[k] for k in ("boxes", "valid", "kpts")}
    t = np.array([r["t"] for r in meta["frames"]], float)
    if len(out["valid"]) != len(t):
        raise ValueError(f"people/{cam}.npz has {len(out['valid'])} frames, frames.json {len(t)}")
    return {"id": cam, "t": t, "wh": (int(meta["width"]), int(meta["height"])),
            "files": [work / "frames" / cam / r["file"] for r in meta["frames"]], **out}


def load_src(src_work: Path, cam: str) -> dict:
    """The source's teacher track (t, box (F,2,4), conf (F,2), ok, merged, role_conf, COCO kp (F,2,17,3) and size when present; slot 0 = lead,
    people_mono.py stores the decided lead first) and its role decision. src_work = work/<src take>, which is Take.load(src).work."""
    with np.load(src_work / "people" / f"teachers_{cam}.npz") as z:
        out = {k: z[k] for k in ("t", "box", "conf", "ok", "merged", "role_conf")}
        out["kp"] = z["kp"] if "kp" in z.files else None
        out["size"] = tuple(int(v) for v in z["size"]) if "size" in z.files else None
    if out["role_conf"].ndim == 1:  # recap_people writes one role confidence per frame
        out["role_conf"] = np.repeat(out["role_conf"][:, None], 2, 1)
    ident = {}
    for name, key in (("teachers.json", "identity"), ("identity.json", "roles")):
        p = src_work / "people" / name
        if not ident and p.exists():
            ident = json.loads(p.read_text()).get(key) or {}
    out["identity"] = {k: ident.get(k) for k in ("decided_by", "confidence")}
    return out


def frame_map(rule: dict, t: np.ndarray, src_t: np.ndarray) -> np.ndarray:
    """Source frame j (F,) of every target frame under a parsed --map rule; -1 outside the source."""
    n = len(src_t)
    if n < 2:
        return np.full(len(t), -1, np.int64)
    if rule["kind"] == "frames":
        j = np.arange(len(t)) + int(rule["shift"])
    else:
        x = np.asarray(t, float) + float(rule["shift"])
        k = np.clip(np.searchsorted(src_t, x), 1, n - 1)
        j = np.where(x - src_t[k - 1] <= src_t[k] - x, k - 1, k)
        half = 0.5 * float(np.median(np.diff(src_t)))
        j = np.where((x >= src_t[0] - half) & (x <= src_t[-1] + half), j, -1)
    return np.where((j >= 0) & (j < n), j, -1).astype(np.int64)


# ------------------------------------------------------------------------------------------------ geometry


def box_iou(a, b) -> np.ndarray:
    """IoU of xyxy boxes broadcast over the leading axes; a NaN box has IoU 0 with everything."""
    a, b = np.nan_to_num(np.asarray(a, float)), np.nan_to_num(np.asarray(b, float))
    lt, rb = np.maximum(a[..., :2], b[..., :2]), np.minimum(a[..., 2:], b[..., 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), -1)
    area_a = np.prod(np.clip(a[..., 2:] - a[..., :2], 0, None), -1)
    area_b = np.prod(np.clip(b[..., 2:] - b[..., :2], 0, None), -1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def det_iou(tbox: np.ndarray, boxes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(n, 2, N) IoU of the two teacher boxes (n, 2, 4) against the detections (n, N, 4); invalid detections 0."""
    return box_iou(tbox[:, :, None], boxes[:, None]) * valid[:, None]


def skeleton_dist(kp: np.ndarray, ref: np.ndarray, h: np.ndarray, min_joints: int = 4) -> np.ndarray:
    """Median image distance of the joints seen (score > 0.3) in both COCO skeletons (..., 17, 3), over h (the box height); NaN with fewer than
    min_joints common joints."""
    seen = (np.nan_to_num(kp[..., 2]) > 0.3) & (np.nan_to_num(ref[..., 2]) > 0.3)
    d = np.where(seen, np.linalg.norm(np.nan_to_num(kp[..., :2]) - np.nan_to_num(ref[..., :2]), axis=-1), np.nan)
    with warnings.catch_warnings():  # all-NaN rows: no common joint
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(d, -1)
    ok = (seen.sum(-1) >= min_joints) & (np.nan_to_num(h) > 0)
    return np.where(ok, med / np.where(ok, h, 1.0), np.nan)


def scan_offset(j: np.ndarray, src: dict, cam: dict, cfg: SeedConfig) -> dict:
    """Median best-detection IoU of the trusted teacher boxes at source offsets -scan..+scan around the rule's j, every scan_step-th frame."""
    ii = np.arange(0, len(j), cfg.scan_step)
    ii = ii[(j[ii] >= 0) & cam["valid"][ii].any(1)]
    rows = []
    for d in range(-cfg.scan, cfg.scan + 1):
        jj = j[ii] + d
        k = (jj >= 0) & (jj < len(src["t"]))
        i_, j_ = ii[k], jj[k]
        tb = src["box"][j_]
        use = (src["conf"][j_] >= cfg.scan_conf) & np.isfinite(tb).all(-1)
        v = det_iou(tb, cam["boxes"][i_], cam["valid"][i_]).max(-1)[use]
        rows.append({"offset": d, "median_iou": round(float(np.median(v)), 4) if len(v) else None,
                     "share_iou_gt_0.8": round(float((v > 0.8).mean()), 4) if len(v) else None, "n": int(len(v))})
    med = {r["offset"]: (r["median_iou"] if r["median_iou"] is not None else -1.0) for r in rows}
    best = max(med, key=med.get)
    nb = [med[d] for d in (-1, 1) if d in med]
    return {"offsets": rows, "best": int(best), "rule_median_iou": med[0],
            "neighbour_margin": round(med[0] - max(nb), 4) if nb else None}


def match(j: np.ndarray, src: dict, cam: dict, cfg: SeedConfig, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """row (F, 2) int16 detection of each role (-1 = MISS) and its IoU (F, 2): the best ONE-TO-ONE assignment of the two usable teacher boxes
    to the valid detections with IoU > match_iou, every pair tried (column 0 = no detection). Then, only where the source flags merged, a role
    that assignment left unmatched shares the other role's detection when it overlaps it > match_iou (one detection for the embraced pair)."""
    F = len(j)
    has, jj = j >= 0, np.clip(j, 0, None)
    tb = np.where(has[:, None, None], src["box"][jj], np.nan)
    usable = has[:, None] & (src["conf"][jj] >= cfg.min_conf) & np.isfinite(tb).all(-1)
    merged = has & src["merged"][jj].any(1)
    row = np.full((F, 2), -1, np.int16)
    iou = np.zeros((F, 2), np.float32)
    for s in range(0, F, block):
        sl = slice(s, min(F, s + block))
        x = det_iou(tb[sl], cam["boxes"][sl], cam["valid"][sl])
        w = np.where((x > cfg.match_iou) & usable[sl][:, :, None], x, -np.inf)
        w = np.concatenate([np.zeros(w.shape[:2] + (1,)), w], 2)
        n1 = w.shape[2]
        same = np.eye(n1, dtype=bool)
        same[0, 0] = False
        tot = np.where(same[None], -np.inf, w[:, 0, :, None] + w[:, 1, None, :])
        k = tot.reshape(len(tot), -1).argmax(1)
        ab = np.stack([k // n1, k % n1], 1)
        for r in (0, 1):  # merged: the unmatched role takes its partner's detection (r = 0 first; then role 1 is matched or stays so)
            o = 1 - r
            fits = np.take_along_axis(w[:, r], ab[:, o:o + 1], 1)[:, 0] > -np.inf
            ab[:, r] = np.where(merged[sl] & (ab[:, r] == 0) & (ab[:, o] > 0) & fits, ab[:, o], ab[:, r])
        row[sl] = ab - 1
        got = np.take_along_axis(w, ab[:, :, None], 2)[:, :, 0]
        iou[sl] = np.where(ab > 0, got, 0.0)
    return row, iou


def gather(cam: dict, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The matched detections' boxes (F, 2, 4) and COCO keypoints (F, 2, 17, 3); NaN where a role has none."""
    fi, r = np.arange(len(row))[:, None], np.clip(row, 0, None)
    hit = row >= 0
    box = np.where(hit[..., None], cam["boxes"][fi, r], np.nan).astype(np.float32)
    kp = np.where(hit[..., None, None], cam["kpts"][fi, r], np.nan).astype(np.float32)
    return box, kp


def world_keep(row: np.ndarray, box: np.ndarray, kp: np.ndarray, src_kp: np.ndarray | None, cfg: SeedConfig | None = None) -> tuple[np.ndarray, dict]:
    """(F, 2) bool: the matched role-frames roles.json keeps. A role's detection whose COCO skeleton is closer to the OTHER role's source skeleton
    by skel_ratio (src_kp (F, 2, 17, 3) on the mapped frames, None: no check) is dropped; a detection both roles share goes to the one role whose
    source skeleton it is that clearly closer to, to neither otherwise (dancer_keypoints would triangulate one body as both). Undecided (kept,
    shared: neither): too few common joints, or the two source skeletons on one body (a doubted source role often sits on its partner)."""
    cfg = cfg or SeedConfig()
    hit = row >= 0
    shared = hit.all(1) & (row[:, 0] == row[:, 1])
    if src_kp is None:
        decided = np.zeros(row.shape, bool)
        own = other = np.full(row.shape, np.nan)
    else:
        h = box[..., 3] - box[..., 1]
        own, other = skeleton_dist(kp, src_kp, h, cfg.skel_joints), skeleton_dist(kp, src_kp[:, ::-1], h, cfg.skel_joints)
        sep = skeleton_dist(src_kp[:, :1], src_kp[:, 1:], h, cfg.skel_joints)
        with np.errstate(invalid="ignore"):
            decided = np.isfinite(own) & np.isfinite(other) & (sep >= cfg.skel_sep)
    with np.errstate(invalid="ignore"):
        partner = hit & decided & (other < cfg.skel_ratio * own)
        keep = hit & ~partner & ~shared[:, None]
        keep |= shared[:, None] & decided & (own < cfg.skel_ratio * other)
    return keep, {"partner_skeleton": {r: int(partner[~shared, k].sum()) for k, r in enumerate(ROLES)},
                  "shared_frames": int(shared.sum()), "shared_kept": int((keep & shared[:, None]).sum()),
                  "undecided": {r: int((hit[:, k] & ~decided[:, k]).sum()) for k, r in enumerate(ROLES)}}


def role_jumps(box: np.ndarray, frac: float, max_gap: int = 30, kept: np.ndarray | None = None) -> dict:
    """Per role (the slot order of box): consecutive frames with a box whose centre moved more than frac x the box height (a swap or a wrong
    match), and across_gaps (gap_jumps; kept (F, 2): frames checked by eye, --keep)."""
    c = 0.5 * (box[..., :2] + box[..., 2:])
    h = box[..., 3] - box[..., 1]
    d = np.linalg.norm(np.diff(c, axis=0), axis=-1)
    hh = 0.5 * (h[1:] + h[:-1])
    with np.errstate(invalid="ignore"):
        pair = np.isfinite(d) & (hh > 0)
        jump = pair & (d > frac * hh)
    out = {r: {"pairs": int(pair[:, k].sum()), "jumps": int(jump[:, k].sum()),
               "share": round(float(jump[:, k].sum() / max(pair[:, k].sum(), 1)), 5),
               "frames": [int(i) + 1 for i in np.flatnonzero(jump[:, k])[:20]]} for k, r in enumerate(ROLES)}
    for k, r in enumerate(ROLES):  # the source can hand a role to another person across unmatched frames
        out[r]["across_gaps"] = gap_jumps(c[:, k], h[:, k], max_gap, None if kept is None else kept[:, k])
    return out


def gap_jumps(c: np.ndarray, h: np.ndarray, max_gap: int = 30, kept: np.ndarray | None = None, frac: float = 1.0) -> dict:
    """Frames with a box that follow a gap of 1..max_gap-1 frames without one, the box centre more than frac x the box height away from the
    last one before the gap (a dancer covers well under a body height in about a second: another person took the role). Flags on a kept
    frame (checked by eye) count as reviewed."""
    with np.errstate(invalid="ignore"):
        m = np.flatnonzero(np.isfinite(c).all(1) & (h > 0))
    a, b = m[:-1], m[1:]
    g = (b - a > 1) & (b - a <= max_gap)
    d = np.linalg.norm(c[b] - c[a], axis=1) / (0.5 * (h[a] + h[b]))
    hit = b[g & (d > frac)]
    todo = hit if kept is None else hit[~kept[hit]]
    return {"count": int(len(hit)), "frames": [int(i) for i in hit[:20]], "unreviewed": int(len(todo)),
            "unreviewed_frames": [int(i) for i in todo[:20]]}


def choose_label(row: np.ndarray, box: np.ndarray, kp: np.ndarray, conf: np.ndarray, wh: tuple[int, int]) -> int | None:
    """The clean label frame: both roles with a box (NaN: none), separated (pair IoU 0 relaxed to 0.1 and 0.3), no box touching an image edge,
    enough joints (10, 8, 6); the highest min(conf), the middle of the window on ties."""
    w, h = wh
    F = len(row)
    both = (row >= 0).all(1) & np.isfinite(box).all((1, 2))
    with np.errstate(invalid="ignore"):
        edge = ((box[..., 0] < 2) | (box[..., 1] < 2) | (box[..., 2] > w - 2) | (box[..., 3] > 0.985 * h)).any(1)
    pair = box_iou(box[:, 0], box[:, 1])
    nkp = (np.nan_to_num(kp[..., 2]) > 0.3).sum(-1).min(1)
    score = conf.min(1) - 1e-3 * np.abs(np.arange(F) / max(F - 1, 1) - 0.5)
    for max_iou, min_kp in ((0.0, 10), (0.1, 8), (0.3, 6)):
        ok = both & ~edge & (pair <= max_iou) & (nkp >= min_kp)
        if ok.any():
            return int(np.flatnonzero(ok)[np.argmax(score[ok])])
    return None


def stretch_mask(n: int, stretches: list[tuple], what: str) -> np.ndarray:
    """(n, 2) bool of (role, i0, i1) stretches (inclusive, roles as reviewed)."""
    out = np.zeros((n, 2), bool)
    for role, i0, i1 in stretches:
        if i0 >= n:
            log(f"{what} {role}:{i0}-{i1} starts after the camera's last frame {n - 1}: ignored")
        out[max(0, i0):min(n, i1 + 1), ROLES.index(role)] = True
    return out


# ------------------------------------------------------------------------------------------------ per camera


def seed_camera(work: Path, m: dict, cfg: SeedConfig, swap: bool = False, drops: list[tuple] = (), keeps: list[tuple] = ()) -> dict:
    """Scan, match, skeleton check and label of one camera in the REVIEWED slot order (slot 0 = the source's lead, its follow when the override
    swaps the camera). Returns the arrays and the report; the label entry alone is in the source's order (label_<cam>.json, which load_labels
    flips). drops / keeps: (role, i0, i1) stretches left unmatched / checked by eye, roles as reviewed."""
    cam = load_cam(work, m["cam"])
    src = load_src(work.parent / m["take"], m["src_cam"])
    if src["size"] is not None and src["size"] != cam["wh"]:
        raise ValueError(f"cam {m['cam']}: {m['take']} teachers_{m['src_cam']}.npz is on {src['size'][0]}x{src['size'][1]} frames, this camera on "
                         f"{cam['wh'][0]}x{cam['wh'][1]}: the boxes cannot be compared")
    perm = [1, 0] if swap else [0, 1]
    for k in ("box", "conf", "role_conf", "kp"):
        if src[k] is not None:
            src[k] = src[k][:, perm]
    j = frame_map(m, cam["t"], src["t"])
    rep = {"map": {k: m[k] for k in ("take", "src_cam", "rule")} | {"size": list(cam["wh"])}, "frames": int(len(j)),
           "src_frames": int(len(src["t"])), "mapped_share": round(float((j >= 0).mean()), 4),
           "src_identity": src["identity"] | {"npz_slots": "lead, follow"}, "portrait": cam["wh"][1] > cam["wh"][0]}
    if m["kind"] == "clock":  # how far the nearest source frame is from the exact instant (a 30 fps re-timed cut of a 59.94 fps original)
        dt = np.abs(src["t"][j[j >= 0]] - (cam["t"][j >= 0] + m["shift"])) * 1000
        rep["clock_dt_ms"] = {"median": round(float(np.median(dt)), 2), "max": round(float(dt.max()), 2),
                              "share_gt_8ms": round(float((dt > 8).mean()), 4)} if len(dt) else None
    rep["scan"] = scan_offset(j, src, cam, cfg)
    for r in rep["scan"]["offsets"]:
        r["source_map"] = f"{m['shift'] + r['offset']:+d}" if m["kind"] == "frames" else f"clock{m['shift']:+.3f} {r['offset']:+d} frames"
    row, iou = match(j, src, cam, cfg)
    both_any_conf = float((match(j, src, cam, replace(cfg, min_conf=0.0))[0] >= 0).all(1).mean())
    rep["dropped"] = []
    for role, i0, i1 in drops:  # reviewed stretches where the source gave the role to somebody else
        k = ROLES.index(role)
        sl = slice(max(0, i0), min(len(row), i1 + 1))
        rep["dropped"].append({"role": role, "start": i0, "end": i1, "matched_before": int((row[sl, k] >= 0).sum())})
        row[sl, k], iou[sl, k] = -1, 0.0
    box, kp = gather(cam, row)
    hit = row >= 0
    has, jj = j >= 0, np.clip(j, 0, None)
    conf = np.where(hit, src["conf"][jj], 0.0).astype(np.float32)
    rconf = np.where(has[:, None], src["role_conf"][jj], 0.0).astype(np.float32)
    merged = has & src["merged"][jj].any(1)
    shared = hit.all(1) & (row[:, 0] == row[:, 1])
    assert not (shared & ~merged).any(), "match() gave one detection both roles outside a merged frame"
    keep, world = world_keep(row, box, kp, None if src["kp"] is None else np.where(has[:, None, None, None], src["kp"][jj], np.nan), cfg)
    rbox = np.where(keep[..., None], box, np.nan).astype(np.float32)
    usable = has[:, None] & (src["conf"][jj] >= cfg.min_conf) & np.isfinite(src["box"][jj]).all(-1)
    rep["match"] = {"lead": round(float(hit[:, 0].mean()), 4), "follow": round(float(hit[:, 1].mean()), 4),
                    "both": round(float(hit.all(1).mean()), 4), "both_in_roles": round(float(keep.all(1).mean()), 4),
                    "both_at_min_conf_0": round(both_any_conf, 4),
                    "usable_teacher": {r: round(float(usable[:, k].mean()), 4) for k, r in enumerate(ROLES)},
                    "usable_teacher_both": round(float(usable.all(1).mean()), 4),
                    "both_of_usable": round(float(hit.all(1).sum() / max(usable.all(1).sum(), 1)), 4),
                    "iou_median": {r: round(float(np.median(iou[hit[:, k], k])), 4) if hit[:, k].any() else None for k, r in enumerate(ROLES)},
                    "merged_frames": int(merged.sum()), "same_detection_unmerged": int((shared & ~merged).sum()), "world": world}
    rep["jumps"] = role_jumps(rbox, cfg.jump_frac, cfg.max_gap, stretch_mask(len(row), list(keeps), "--keep"))
    rep["kept"] = [{"role": r, "start": a, "end": b} for r, a, b in keeps]
    i = choose_label(row, rbox, kp, conf, cam["wh"])
    if i is None:
        raise RuntimeError(f"cam {m['cam']}: no clean frame with both seeded roles matched and separated - label it by hand")
    used = rconf[hit]
    agree = float(used.mean()) if used.size else 0.0
    lab = label_entry(cam, i, rbox[i, perm], kp[i, perm], iou[i, perm], conf[i, perm], rconf[i, perm], int(j[i]), m, src["identity"], agree)
    names = lab["colour_names"]
    rep["label"] = {"frame": i, "t": lab["t"], "src_frame": int(j[i]), "confidence": lab["confidence"],
                    "colour_rule_llr": lab["seed"]["colour_rule_llr"] * (-1 if swap else 1),
                    "colour_names": {"lead": names["follow"], "follow": names["lead"]} if swap else names}
    flags = flag_frames(rep["jumps"])
    rep["flagged_frames"] = [i for i, _ in flags[:20]]  # review/seed_roles.jpg shows the first sheet_frames of them
    return {"cam": cam, "j": j, "row": row, "iou": iou, "box": box, "kp": kp, "rbox": rbox, "conf": conf, "role_conf": rconf, "merged": merged,
            "src_ok": has & src["ok"][jj], "perm": perm, "label": lab, "report": rep, "flags": flags}


def flag_frames(jumps: dict) -> list[tuple[int, str]]:
    """(frame, note) for the sheet: the unreviewed across-gap flags first, then the jumps, in frame order."""
    notes: dict[int, list[str]] = {}
    for kind, key in (("across gap", "unreviewed_frames"), ("jump", "frames")):
        for r, v in jumps.items():
            for i in (v["across_gaps"] if kind == "across gap" else v)[key]:
                notes.setdefault(i, []).append(f"{r} {kind}")
    gap = [i for i in sorted(notes) if any("gap" in n for n in notes[i])]
    return [(i, ", ".join(notes[i])) for i in gap + [i for i in sorted(notes) if i not in gap]]


def label_entry(cam: dict, i: int, box: np.ndarray, kp: np.ndarray, iou: np.ndarray, conf: np.ndarray, rconf: np.ndarray, src_frame: int,
                m: dict, ident: dict, agree: float) -> dict:
    """dancers-labels' label schema for frame i (dancers.auto_label's keys; review_tile and apply_override read them) from the ONE decoded frame;
    box (2, 4), kp (2, 17, 3), iou / conf / rconf (2,) of that frame in the slot order the label is stored in (the source's)."""
    img = cv2.imread(str(cam["files"][i]), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(cam["files"][i])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    d = {r: descriptors(hsv, box[k], kp[k]) for k, r in enumerate(ROLES)}
    llr, cues = lead_llr(d["lead"], d["follow"], RuleConfig())
    c = ident.get("confidence")
    return {"frame": int(i), "t": float(cam["t"][i]), "lead_box": [round(float(v), 1) for v in box[0]],
            "follow_box": [round(float(v), 1) for v in box[1]], "by": f"seed:{m['take']}",
            "confidence": round(float(c if c is not None else rconf.min()), 3), "frame_llr": round(abs(float(llr)), 2),
            "evidence_agreement": round(agree, 3), "track_switches": 0, "cues": {k: round(v, 2) for k, v in cues.items()},
            "descriptors": {r: _round(d[r]) for r in ROLES}, "colour_names": {r: colour_names(d[r]) for r in ROLES},
            "seed": {"take": m["take"], "cam": m["src_cam"], "map": m["rule"], "src_frame": src_frame, "colour_rule_llr": round(float(llr), 2),
                     "iou": [round(float(v), 3) for v in iou], "conf": [round(float(v), 3) for v in conf],
                     "role_conf": [round(float(v), 3) for v in rconf], "decided_by": ident.get("decided_by")}}


def gate_g0(rep: dict, m: dict, cfg: SeedConfig) -> dict:
    sc, mt = rep["scan"], rep["match"]
    both = cfg.g0_both_portrait if rep["portrait"] else cfg.g0_both
    jumps = max(v["share"] for v in rep["jumps"].values())
    gaps = sum(v["across_gaps"]["unreviewed"] for v in rep["jumps"].values())
    g = {"scan_best_is_rule": {"value": sc["best"], "threshold": 0, "pass": sc["best"] == 0},
         "scan_median_iou": {"value": sc["rule_median_iou"], "threshold": cfg.g0_iou, "pass": sc["rule_median_iou"] >= cfg.g0_iou}}
    if m["kind"] == "frames" and sc["neighbour_margin"] is not None:
        g["scan_neighbour_margin"] = {"value": sc["neighbour_margin"], "threshold": cfg.g0_margin, "pass": sc["neighbour_margin"] >= cfg.g0_margin}
    g["both_matched"] = {"value": mt["both_in_roles"], "threshold": both, "pass": mt["both_in_roles"] >= both}
    g["same_detection_unmerged"] = {"value": mt["same_detection_unmerged"], "threshold": 0, "pass": mt["same_detection_unmerged"] == 0}
    g["role_jumps"] = {"value": jumps, "threshold": cfg.g0_jumps, "pass": jumps <= cfg.g0_jumps}
    g["across_gaps_unreviewed"] = {"value": gaps, "threshold": 0, "pass": gaps == 0}
    return {"checks": g, "pass": all(v["pass"] for v in g.values())}


# ------------------------------------------------------------------------------------------------ outputs


def teachers_arrays(s: dict, m: dict) -> dict:
    """people/teachers_<cam>.npz (recap contract) in the reviewed slot order."""
    row, hit = s["row"], s["row"] >= 0
    F = len(row)
    return {"t": s["cam"]["t"].astype(np.float64), "frame_index": np.full(F, -1, np.int32), "ok": s["src_ok"], "row": row, "box": s["box"],
            "kp": s["kp"], "conf": s["conf"], "cost": np.where(hit, 1.0 - s["iou"], np.nan).astype(np.float32), "kind": hit.astype(np.int8),
            "nj": (np.nan_to_num(s["kp"][..., 2]) >= 0.5).sum(-1).astype(np.int8), "gap": np.zeros((F, 2), np.int16),
            "merged": np.repeat(s["merged"][:, None], 2, 1), "role_conf": s["role_conf"], "lead_slot": np.zeros(F, np.int8),
            "floor": np.full((F, 2, 2), np.nan, np.float32), "size": np.array(s["cam"]["wh"], np.int32), "attached": np.array(False),
            "src_frame": s["j"].astype(np.int32), "src": np.array(f"{m['take']}:{m['src_cam']}:{m['rule']}")}


def carry_over(path: Path, new: dict) -> dict:
    """A re-run keeps what carlos_aline_inputs --teachers-floor (runbook step 11) filled into the previous people/teachers_<cam>.npz on the same
    frames: frame_index, and each role-frame's floor whose detection is unchanged (it follows a detection across an override swap); a role-frame
    whose detection changed gets NaN (re-run step 11). Fills `new` in place; returns the counts for seed_report."""
    out = {"frame_index": False, "floor_kept": 0, "floor_lost": 0}
    if not path.exists():
        return out
    with np.load(path) as z:
        t = z["t"]
        if len(t) != len(new["t"]) or not np.allclose(t, new["t"], rtol=0, atol=1e-6):
            return out
        fi, row = z["frame_index"], z["row"]
        floor = z["floor"] if "floor" in z.files else None
    if (fi >= 0).any():
        new["frame_index"] = fi.astype(np.int32)
        out["frame_index"] = True
    if floor is not None:
        had = np.isfinite(floor).all(-1)
        moved = np.zeros_like(had)
        for k in (0, 1):
            for k2 in (0, 1):
                same = (row[:, k2] == new["row"][:, k]) & (new["row"][:, k] >= 0) & had[:, k2]
                new["floor"][same, k] = floor[same, k2]
                moved[:, k2] |= same
        out.update(floor_kept=int(np.isfinite(new["floor"]).all(-1).sum()), floor_lost=int((had & ~moved).sum()))
    return out


def save_npz(path: Path, **arrays) -> None:
    """np.savez_compressed into <name>.part.npz, then os.replace: a killed run (the 1 GB scope) never leaves a truncated file."""
    tmp = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def write_json(path: Path, obj, **kw) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, **kw))
    os.replace(tmp, path)


def roles_frames(t: np.ndarray, box: np.ndarray, conf: np.ndarray, merged: np.ndarray) -> list[dict]:
    """roles.json frames: exactly one entry per frames.json frame, in order; box None (conf 0) where a role has none."""
    out = []
    for i in range(len(t)):
        fr = {"i": i, "t": round(float(t[i]), 4)}
        for k, r in enumerate(ROLES):
            b = box[i, k]
            ok = bool(np.isfinite(b).all())
            fr[r] = {"box": [round(float(v), 1) for v in b] if ok else None, "conf": round(float(conf[i, k]), 3) if ok else 0.0}
        fr.update(swap_prob=0.0, sam_swap_prob=0.0, embrace=bool(merged[i]), repaired=False, anchor=False)
        out.append(fr)
    return out


def sheet_tiles(s: dict, picks: list[tuple[int, str]], height: int = 480) -> list[np.ndarray]:
    """The picked (frame, note) frames (decoded one at a time) with the roles.json lead / follow boxes, cropped to them."""
    from .dancers_qa import FOLLOW, LEAD, label_text
    cam, box = s["cam"], s["rbox"]
    tiles = []
    for i, note in picks:
        fin = np.isfinite(box[i]).all(1)
        img = cv2.imread(str(cam["files"][i]), cv2.IMREAD_COLOR) if fin.any() else None
        if img is None:
            continue
        hh, ww = img.shape[:2]
        for k, (r, col) in enumerate((("lead", LEAD), ("follow", FOLLOW))):
            if not fin[k]:
                continue
            b = box[i, k].astype(int)
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), col, 3)
            org = (int(b[0]), max(26, int(b[1]) - 8)) if r == "follow" else (int(b[0]), min(hh - 8, int(b[3]) + 28))
            label_text(img, r.upper(), org, col, 0.9)  # follow above her box, lead below his (as dancers.review_tile)
        x0, y0 = box[i, fin, :2].min(0)
        x1, y1 = box[i, fin, 2:].max(0)
        ch = max((y1 - y0) * 1.4, 0.35 * hh)
        cw = max((x1 - x0) * 1.4, 0.6 * ch)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        crop = img[int(max(0, cy - ch / 2)):int(min(hh, cy + ch / 2)), int(max(0, cx - cw / 2)):int(min(ww, cx + cw / 2))]
        tile = cv2.resize(crop, (max(1, int(round(crop.shape[1] * height / crop.shape[0]))), height), interpolation=cv2.INTER_AREA)
        label_text(tile, f"cam {cam['id']} #{i} t={cam['t'][i]:.2f}  src #{int(s['j'][i])}", (6, 24), (255, 255, 255), 0.55)
        if note:
            label_text(tile, note, (6, 48), (0, 255, 255), 0.55)
        tiles.append(tile)
    return tiles


def spread_frames(box: np.ndarray, n: int) -> list[tuple[int, str]]:
    """n frames spread over the frames where both roles have a box."""
    both = np.flatnonzero(np.isfinite(box).all((1, 2)))
    return [(int(i), "") for i in both[np.unique(np.round(np.linspace(0, len(both) - 1, n + 2)[1:-1]).astype(int))]] if len(both) else []


def write_sheet(path: Path, rows: list[tuple[str, list[np.ndarray]]], n: int) -> None:
    from .dancers_qa import contact_sheet, label_text
    tiles = []
    for name, ts in rows:
        ts = list(ts) or [np.zeros((480, 288, 3), np.uint8)]
        label_text(ts[0], name, (6, 470), (255, 255, 255), 0.5)
        tiles += ts + [np.zeros_like(ts[0])] * (n - len(ts))
    head = "seeded roles (roles.json): lead = red, follow = blue; per camera a row spread over its window, then its flagged frames: check every tile"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), contact_sheet(tiles, head, cols=n), [cv2.IMWRITE_JPEG_QUALITY, 88])


def write_outputs(work: Path, maps: list[dict], seeded: dict, cfg: SeedConfig) -> dict:
    """Every output file (module docstring); the arrays are in the reviewed order, label and teacher_tracks go back to the source's."""
    out = work / "dancers"
    (out / "teacher_tracks").mkdir(parents=True, exist_ok=True)
    (work / "people").mkdir(parents=True, exist_ok=True)
    lp = out / "labels.json"
    labels = json.loads(lp.read_text()) if lp.exists() else {}
    for m in maps:
        c, s = m["cam"], seeded[m["cam"]]
        tp = work / "people" / f"teachers_{c}.npz"
        ta = teachers_arrays(s, m)
        s["report"]["teachers_carry_over"] = co = carry_over(tp, ta)
        if co["floor_lost"]:
            log(f"{c}: {co['floor_lost']} role-frames lost their floor (another detection now): re-run carlos_aline_inputs --teachers-floor")
        save_npz(tp, **ta)
        p = s["perm"]
        save_npz(out / "teacher_tracks" / f"{c}.npz", idx=s["row"][:, p].astype(int), boxes=s["box"][:, p], conf=s["conf"][:, p])
        write_json(out / f"label_{c}.json", s["label"], indent=1)
        labels[c] = s["label"]
    write_json(lp, labels, indent=1)
    final_labels = load_labels(work)
    rp = out / "roles.json"
    roles = json.loads(rp.read_text()) if rp.exists() else {}
    roles.setdefault("cams", {})
    for m in maps:
        c, s = m["cam"], seeded[m["cam"]]
        roles["cams"][c] = {"frames": roles_frames(s["cam"]["t"], s["rbox"], s["conf"], s["merged"]),
                            "seed": {k: m[k] for k in ("take", "src_cam", "rule")}}
    mine = {m["cam"] for m in maps}
    sw, fl = roles.get("swaps_repaired"), roles.get("identity_flags")
    roles["labels_source"] = {k: final_labels[k]["by"] for k in final_labels}
    roles["swaps_repaired"] = [x for x in sw if x.get("cam") not in mine] if isinstance(sw, list) else []
    roles["identity_flags"] = {k: v for k, v in fl.items() if k not in mine} if isinstance(fl, dict) else {}
    write_json(rp, roles)
    rows = []
    for m in maps:
        s = seeded[m["cam"]]
        name = f"cam {m['cam']} <- {m['take']}:{m['src_cam']} {m['rule']}" + (" (override: swapped)" if s["report"]["override_swap"] else "")
        rows.append((name, sheet_tiles(s, spread_frames(s["rbox"], cfg.sheet_frames))))
        if s["flags"]:
            rows.append((f"cam {m['cam']} flagged: {len(s['flags'])} frames (first {min(cfg.sheet_frames, len(s['flags']))})",
                         sheet_tiles(s, s["flags"][:cfg.sheet_frames])))
    write_sheet(work / "review" / "seed_roles.jpg", rows, cfg.sheet_frames)
    return {"teachers": [f"people/teachers_{m['cam']}.npz" for m in maps], "teacher_tracks": [f"dancers/teacher_tracks/{m['cam']}.npz" for m in maps],
            "roles": "dancers/roles.json", "labels": ["dancers/labels.json"] + [f"dancers/label_{m['cam']}.json" for m in maps],
            "sheet": "review/seed_roles.jpg"}


# ------------------------------------------------------------------------------------------------ driver


def run(work: Path, maps: list[dict], cfg: SeedConfig | None = None, write: bool = True, drops: list[tuple] = (), keeps: list[tuple] = ()) -> dict:
    """Seed every --map camera; with write=False only the scan / match / G0 numbers (nothing written). drops / keeps: parse_drop tuples."""
    cfg = cfg or SeedConfig()
    t0 = time.time()
    stray = sorted({d[0] for d in (*drops, *keeps)} - {m["cam"] for m in maps})
    if stray:
        raise ValueError(f"--drop / --keep name cameras without a --map: {', '.join(stray)}")
    ovp = work / "dancers" / "labels_override.json"
    ov = json.loads(ovp.read_text()) if ovp.exists() else {}
    swaps = {m["cam"]: bool((ov.get(m["cam"]) or {}).get("swap")) for m in maps}
    seeded = {}
    for m in maps:
        mine = [d[1:] for d in drops if d[0] == m["cam"]]
        kept = [d[1:] for d in keeps if d[0] == m["cam"]]
        seeded[m["cam"]] = s = seed_camera(work, m, cfg, swaps[m["cam"]], mine, kept)
        s["report"]["override_swap"] = swaps[m["cam"]]
        s["report"]["g0"] = gate_g0(s["report"], m, cfg)
        print_camera(m, s["report"])
    settings = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    settings["min_conf_on"] = "the source teachers' pose confidence `conf` (people_mono pconf), as recap_front.teacher_slots"
    report = {"take": work.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": settings,
              "cams": {c: s["report"] for c, s in seeded.items()}, "g0_pass": all(s["report"]["g0"]["pass"] for s in seeded.values())}
    if write:
        report["outputs"] = write_outputs(work, maps, seeded, cfg)
        report["seconds"] = round(time.time() - t0, 1)
        write_json(work / "dancers" / "seed_report.json", report, indent=1)
        log(f"-> {work / 'dancers'} (teacher_tracks/, roles.json, labels.json, seed_report.json), {work / 'review' / 'seed_roles.jpg'}")
    log(f"gate G0: {'PASS' if report['g0_pass'] else 'FAIL'}")
    return report


def print_camera(m: dict, rep: dict) -> None:
    sc, mt, wd = rep["scan"], rep["match"], rep["match"]["world"]
    log(f"{m['cam']} <- {m['take']}:{m['src_cam']} {m['rule']}: scan " + ", ".join(
        f"{r['source_map']}: {r['median_iou']}" for r in sc["offsets"]) + f" (best offset {sc['best']:+d})")
    jumps = ", ".join(f"{r} {v['jumps']}/{v['pairs']} (across gaps {v['across_gaps']['count']}, unreviewed "
                      f"{v['across_gaps']['unreviewed_frames'][:6]})" for r, v in rep["jumps"].items())
    log(f"{m['cam']}: matched lead {mt['lead']:.3f} follow {mt['follow']:.3f} both {mt['both']:.3f}, in roles.json {mt['both_in_roles']:.3f} "
        f"(teacher conf >= min-conf for both: {mt['usable_teacher_both']:.3f}, of which matched {mt['both_of_usable']:.3f}; both at min-conf 0: "
        f"{mt['both_at_min_conf_0']:.3f}), merged {mt['merged_frames']}, shared {wd['shared_frames']} (kept for one role {wd['shared_kept']}), "
        f"partner skeleton {wd['partner_skeleton']}; jumps {jumps}; label frame {rep['label']['frame']} (src #{rep['label']['src_frame']})")
    for k, v in rep["g0"]["checks"].items():
        log(f"{m['cam']}:   G0 {k:<24} {v['value']!s:<10} threshold {v['threshold']!s:<6} {'pass' if v['pass'] else 'FAIL'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m dancecap.dancers_seed", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "scan"])
    p.add_argument("take")
    p.add_argument("--map", action="append", required=True, help="CAM=SRC_TAKE:SRC_CAM:RULE (RULE +N | -N | clock-S | clock+S); repeat per camera")
    p.add_argument("--scan", type=int, default=3, help="source offsets -scan..+scan scanned around each rule")
    p.add_argument("--min-conf", type=float, default=0.3, help="teacher roles below this pose confidence are left unmatched")
    p.add_argument("--drop", action="append", default=[], help="CAM:lead|follow:I0-I1 frames where the source's role is somebody else (seen on "
                                                                "review/seed_roles.jpg / seed_report across_gaps; roles as reviewed): left unmatched")
    p.add_argument("--keep", action="append", default=[], help="CAM:lead|follow:I0-I1 frames checked by eye where the role IS right: their "
                                                                "across-gap flags count as reviewed (gate G0)")
    a = p.parse_args(argv)
    cfg = SeedConfig(min_conf=a.min_conf, scan=a.scan)
    rep = run(Take.load(a.take).work, [parse_map(s) for s in a.map], cfg, write=a.action == "run", drops=[parse_drop(s) for s in a.drop],
              keeps=[parse_drop(s, "--keep") for s in a.keep])
    if not rep["g0_pass"]:
        raise SystemExit(2)  # after writing: the sheets and seed_report are there to review, a scripted chain stops at the gate


if __name__ == "__main__":
    main()
