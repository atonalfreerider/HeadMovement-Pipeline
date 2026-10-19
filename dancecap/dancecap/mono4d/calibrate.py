"""Labelled mannequin frames for the visual judge (judge.py): known defects, no real person in any image.

The synthetic couple (dancecap.synthetic: two SMPL-X bodies, work/synthetic_mamma/gt120_anat/smplx) is drawn as shaded
capsule mannequins in a plain studio from one ring camera, at a few times, in these variants:

    clean       the couple as it is                                   all defects no
    duplicate   the follow drawn a second time, shifted and ghosted   duplicate_person yes
    broken      the lead's left arm missing, his right shin detached   broken_anatomy yes
    defects     dark floating blobs and a hole through the follow     render_defects yes
    pose pairs  reference = clean at t; view = clean at t (same_pose yes) or at t + 1.1 s (same_pose no)

    python -m dancecap.mono4d judge-calibrate [--times 3] [--out work/mono4d/calibration]

run() judges every image (the judge's real questions, one request per frame) and reports, per question, the mean
P(yes) on positives and negatives and the accuracy at judge.PRESETS["mannequin"]. Mannequins are not dancers: this tests that
the questions discriminate the defects, not how well Jev reads a real render. That needs labelled real frames, which
need the owner's go-ahead (likeness data).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT
from . import judge, rig
from .. import mamma_dyn as md

GT = PACKAGE_ROOT / "work" / "synthetic_mamma" / "gt120_anat" / "smplx"
OUT = PACKAGE_ROOT / "work" / "mono4d" / "calibration"
SIZE = (540, 960)  # portrait; judge.prepare_image keeps it (>= 280 px a side)
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]  # SMPL-X body joints 0-21
RADIUS = {1: .075, 2: .075, 4: .06, 5: .06, 7: .045, 8: .045, 10: .04, 11: .04, 3: .12, 6: .13, 9: .14,
          12: .055, 13: .08, 14: .08, 16: .05, 17: .05, 18: .04, 19: .04, 20: .035, 21: .035, 15: .05}  # child joint
COLOURS = {"lead": (52, 62, 84), "follow": (214, 210, 204)}
LABELS = ("duplicate_person", "broken_anatomy", "render_defects", "same_pose")


def _load() -> tuple[dict, np.ndarray]:
    return {r: np.load(GT / f"{r}.npz")["joints"][:, :22].astype(float) for r in ("lead", "follow")}, \
        np.load(GT / "lead.npz")["t"].astype(float)


def _camera(az0_deg: float = 30.0) -> rig.ViewSpec:
    return rig.ring(1, n=8, radius=2.7, heights=(1.45,), target_height=0.95, hfov_deg=48.0, size=SIZE,
                    az0_deg=az0_deg)[0]


def draw(bodies: list[tuple], cam: rig.ViewSpec, blobs: int = 0, hole: np.ndarray | None = None, seed: int = 0):
    """bodies: (joints (22,3), colour, alpha, skipped bones {child ids}, head) -> PIL image of the studio with the
    mannequins (a bone is drawn from PARENTS[c] to c unless c is skipped)."""
    from PIL import Image, ImageDraw
    W, H = cam.width, cam.height
    K, R, t = cam.K[0], cam.R[0], cam.t[0]
    horizon = int(np.clip(md.project(np.array([[0.0, 0.0, -30.0]]), K, R, t)[0][0, 1], 0, H))
    bg = np.empty((H, W, 3), np.uint8)
    bg[:horizon] = (178, 174, 168)  # wall
    k = (np.arange(horizon, H) - horizon) / max(1, H - horizon)  # floor: a wooden gradient below the horizon
    bg[horizon:] = np.stack([120 + 40 * k, 92 + 30 * k, 66 + 20 * k], 1).astype(np.uint8)[:, None, :]
    img = Image.fromarray(bg)
    d = ImageDraw.Draw(img)
    layers = []
    for b_i, (J, colour, alpha, skip, has_head) in enumerate(bodies):
        px, z = md.project(J, K, R, t)
        f = K[0, 0]
        for c, p in enumerate(PARENTS):
            if p < 0 or c in skip:
                continue
            layers.append((float((z[c] + z[p]) / 2), "bone", px[p], px[c], max(2, int(2 * RADIUS[c] * f / z[c])),
                           colour, alpha, b_i))
        if has_head:
            head = px[15] + np.array([0.0, -0.07 * f / z[15]])
            layers.append((float(z[15]), "head", head, None, int(0.11 * f / z[15]), colour, alpha, b_i))
    ghosts = {}  # translucent bodies: drawn opaque on their own overlay, composited once at their alpha
    for depth, kind, a, b, w, colour, alpha, b_i in sorted(layers, key=lambda x: -x[0]):  # far first
        if alpha < 1.0:
            if b_i not in ghosts:
                ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
                ghosts[b_i] = (ov, ImageDraw.Draw(ov), alpha)
            o = ghosts[b_i][1]
        else:
            o = d
        k = float(np.clip(1.05 - 0.04 * (depth - 2.5), 0.9, 1.05))  # a little depth cue; distance never greys a body
        shade = tuple(min(255, max(0, int(c * k))) for c in colour)
        fill = (*shade, 255) if alpha < 1.0 else shade
        if kind == "bone":
            o.line([tuple(a), tuple(b)], fill=fill, width=w)
            for q in (a, b):
                o.ellipse([q[0] - w / 2, q[1] - w / 2, q[0] + w / 2, q[1] + w / 2], fill=fill)
        else:
            o.ellipse([a[0] - w, a[1] - 1.2 * w, a[0] + w, a[1] + 1.2 * w], fill=fill)
    for ov, _, alpha in ghosts.values():
        ov.putalpha(ov.getchannel("A").point(lambda v: int(v * alpha)))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    rng = np.random.default_rng(seed)
    if blobs:
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        o = ImageDraw.Draw(over)
        for _ in range(blobs):
            cx, cy = rng.uniform(0.2 * W, 0.8 * W), rng.uniform(0.15 * H, 0.75 * H)
            rx, ry = rng.uniform(15, 45), rng.uniform(10, 60)
            o.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(18, 16, 20, int(rng.uniform(150, 230))))
        img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    if hole is not None:
        p, z = md.project(hole[None], K, R, t)
        r = 0.09 * K[0, 0] / z[0]
        ImageDraw.Draw(img).ellipse([p[0, 0] - r, p[0, 1] - r, p[0, 0] + r, p[0, 1] + r], fill=(178, 174, 168))
    return img


def build(times: int = 3, out: Path = OUT) -> list[dict]:
    """Write the labelled set -> [{"path", "reference", "labels": {question: bool}}] (also out/labels.json)."""
    J, t = _load()
    cam, src_cam = _camera(), _camera(130.0)  # src_cam stands in for the input phone: same moment, another side
    out.mkdir(parents=True, exist_ok=True)
    picks = np.linspace(int(1.0 * 120), len(t) - int(1.6 * 120), times).astype(int)
    shift = int(round(1.1 * 120))  # GT is at 120 Hz
    rows = []
    for k, i in enumerate(picks):
        L, F = J["lead"][i], J["follow"][i]
        clean = [(L, COLOURS["lead"], 1.0, set(), True), (F, COLOURS["follow"], 1.0, set(), True)]
        src = out / f"t{k}_source.jpg"
        draw(clean, src_cam).save(src, "JPEG", quality=92)
        sideways = cam.R[0].T @ np.array([1.0, 0.0, 0.0])  # camera right in the world
        ghost = F + 0.55 * sideways
        variants = {
            "clean": (clean, {}, {}),
            "duplicate": (clean + [(ghost, COLOURS["follow"], 0.55, set(), True)], {"duplicate_person": True}, {}),
            "broken": (None, {"broken_anatomy": True}, {}),
            "defects": (clean, {"render_defects": True}, {"blobs": 7, "hole": F[6]}),
        }
        for name, (bodies, pos, extra) in variants.items():
            if bodies is None:  # the lead without his left arm; his right shin + foot drawn detached, off the knee
                shin = L.copy()
                shin[[5, 8, 11]] += 0.35 * sideways + np.array([0.0, 0.25, 0.0])
                bodies = [(L, COLOURS["lead"], 1.0, {16, 18, 20, 8, 11}, True), clean[1],
                          (shin, COLOURS["lead"], 1.0, set(range(22)) - {8, 11}, False)]
            img = draw(bodies, cam, blobs=extra.get("blobs", 0), hole=extra.get("hole"), seed=k)
            p = out / f"t{k}_{name}.jpg"
            img.save(p, "JPEG", quality=92)
            labels = {q: bool(pos.get(q, False)) for q in LABELS if q != "same_pose"}
            rows.append({"path": str(p), "reference": None, "source": str(src), "variant": name, "labels": labels})
        j = min(i + shift, len(t) - 1)
        later = draw([(J["lead"][j], COLOURS["lead"], 1.0, set(), True),
                      (J["follow"][j], COLOURS["follow"], 1.0, set(), True)], cam)
        p_later = out / f"t{k}_later.jpg"
        later.save(p_later, "JPEG", quality=92)
        clean_p = out / f"t{k}_clean.jpg"
        none = {q: False for q in LABELS if q != "same_pose"}
        rows.append({"path": str(clean_p), "reference": str(clean_p), "source": None, "variant": "pose_same",
                     "labels": {**none, "same_pose": True}})
        rows.append({"path": str(p_later), "reference": str(clean_p), "source": None, "variant": "pose_later",
                     "labels": {**none, "same_pose": False}})
    (out / "labels.json").write_text(json.dumps(rows, indent=1))
    return rows


def score(rows: list[dict], answers: list[dict], th: dict | None = None) -> dict:
    """Per question: mean P(yes) on positives / negatives, accuracy at the judge threshold (same_pose: yes = p >= th)."""
    th = {**judge.PRESETS["mannequin"], **(th or {})}
    rep = {}
    for q in LABELS:
        pos = [a[q] for r, a in zip(rows, answers) if a and q in a and r["labels"].get(q) is True]
        neg = [a[q] for r, a in zip(rows, answers) if a and q in a and r["labels"].get(q) is False]
        if not pos and not neg:
            continue
        acc = [p >= th[q] for p in pos] + [p < th[q] for p in neg]
        rep[q] = {"n_pos": len(pos), "n_neg": len(neg), "mean_p_pos": float(np.mean(pos)) if pos else None,
                  "mean_p_neg": float(np.mean(neg)) if neg else None, "accuracy": float(np.mean(acc)),
                  "separates": bool(pos and neg and min(pos) > max(neg))}
    return rep


MANNEQUIN_ABOUT = ("`view` is one frame of a test scene in which two partner dancers are drawn as simple shaded "
                   "mannequins (rounded capsule limbs, oval heads, no faces or hands) in a plain studio. Judge the "
                   "mannequins as the people.")


def run(times: int = 3, out: Path = OUT, client=None, workers: int = 4, with_source: bool = True,
        mannequin_about: bool = False) -> dict:
    """Judge the labelled set. with_source: give each defect frame the clean frame from another camera as `source`
    (production gives the input phone's frame); mannequin_about: tell Jev the people are drawn as mannequins."""
    rows = build(times, out)
    frames = {"calibration": [{"t": i, "path": r["path"], "reference": r["reference"],
                               "source": r["source"] if with_source else None} for i, r in enumerate(rows)]}
    # mannequins drawn here, no real person: the upload needs no go-ahead
    rep = judge.judge_views(frames, allow_upload=True, workers=workers, client=client, th=judge.PRESETS["mannequin"],
                            about=MANNEQUIN_ABOUT if mannequin_about else judge.ABOUT)
    answers = [f.get("answers") for f in rep["views"]["calibration"]["frames"]]
    doc = {"images": len(rows), "input_tokens": rep["input_tokens"], "model": rep["model"],
           "with_source": with_source, "mannequin_about": mannequin_about, "per_question": score(rows, answers),
           "frames": [{**r, "answers": a} for r, a in zip(rows, answers)]}
    tag = ("src" if with_source else "nosrc") + ("_mq" if mannequin_about else "")
    (out / f"calibration_{tag}.json").write_text(json.dumps(doc, indent=1))
    return doc
