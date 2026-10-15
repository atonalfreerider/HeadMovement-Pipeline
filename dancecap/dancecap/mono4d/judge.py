"""TypeSafe (Jev) visual judgments of generated views: would a person reject this frame? Code owns the policy.

The NoPo4D attempt (atlas_bridge/README.md) failed on things a glance catches and our numbers did not: duplicated
dancers, misplaced limbs, floating dark regions. These are semantic judgments, so each sampled frame gets ONE
request with several Noul questions (asked together, answered independently):

    duplicate_person   a second, ghosted or overlapping copy of a dancer (a mirror reflection is not a duplicate)
    broken_anatomy     a missing, extra, detached, bent-backwards or melted limb or head
    render_defects     floating blobs, smears, holes through a body, body parts in the air
    cut_off            a dancer cut by the image edge (a framing problem of the rig, not of the provider)
    same_pose          only with a reference: the real phone frame at the same camera pose and time - do the dancers
                       hold the same poses and partner hold?

Counting people is left to code (YOLO boxes in holdout.compare_views): TypeSafe's own notes say Jev does not count
reliably. Thresholds are PROVISIONAL. The synthetic mannequin set (calibrate.py, jev-1.13.0, 2026-10-15, 18 frames x 4
prompt variants) gave: same_pose 1.00 accuracy (0.98-0.99 vs 0.09-0.14), duplicate_person ghosts 0.47 / 0.71 / 0.73
while the 90 clean frames (15 calibration + the oracle run: the 38 s take's own fit drawn at 15 cameras, 75 frames, with
overlapping partners) peak at 0.62, hence 0.65 (2 of 3 ghosts caught, no false alarm); render_defects ranks every
defect frame above every clean one, but clean mannequins already score 0.45-0.8 (so 0.9); broken_anatomy does not
separate on mannequins (0.70-0.82 vs 0.29-0.69): it is ADVISORY (reported, never decisive) until labelled real renders
(known-bad: the NoPo4D views; known-good: phone frames) calibrate it.

Images are a "dreamer-ready" TypeSafe feature (header TypeSafe-Danger-Zone-Images: true; jev-1.13 and 1.14 accept
them): advisory evidence for gate.py, never the only reason a run passes. Frames of real dancers are likeness data:
judge_views() refuses unless the caller passes allow_upload (the owner's go-ahead to send them to TypeSafe).
"""
from __future__ import annotations

import base64
import io
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from typesafe_sdk import Noul

MODEL = "jev-latest"
IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
MAX_SIDE = 1024  # TypeSafe accepts 256-1568 px per side; 1024 keeps tokens down and limbs visible
THRESHOLDS = {"duplicate_person": 0.65, "broken_anatomy": 0.85, "render_defects": 0.9, "cut_off": 0.6,
              "same_pose": 0.5, "bad_frame_frac": 0.15, "min_bad_frames": 2}
DECISIVE = ("duplicate_person", "render_defects", "cut_off")  # + same_pose when a reference is given
ADVISORY = ("broken_anatomy",)  # reported in every row, never drops a frame (see the calibration note above)

ABOUT = ("`view` is one frame rendered by a 3D reconstruction model from a video of two people partner dancing in a "
         "dance studio (one leads, one follows; they often hold each other closely or turn). Walls may have mirrors.")
ABOUT_REF = (" `reference` is a real photo taken from the same camera position at the same moment, so the same two "
             "people should appear in the same poses.")
ABOUT_SRC = (" `source` is a real photo of the same moment from a different camera position (the input video): the people "
             "in `view` should look like the people in `source`, seen from another side.")

QUESTIONS = {
    "duplicate_person": Noul(
        instructions="Does `view` show one of the dancers more than once, as a second copy of the same person?",
        criteria={"true": "The same person appears twice: a duplicate, a ghosted or semi-transparent double, or two "
                          "overlapping copies of one body, side by side or merged.",
                  "false": "Each person appears once. A reflection that is clearly inside a wall mirror, or a "
                           "spectator in the background, is not a duplicate."}),
    "broken_anatomy": Noul(
        instructions="Does a dancer in `view` have a broken body?",
        criteria={"true": "A missing, extra, detached, bent-backwards or melted arm, leg or head, or limbs attached in "
                          "the wrong place.",
                  "false": "Both bodies look like normal people; limbs hidden behind the partner or blurred by fast "
                           "motion are normal."}),
    "render_defects": Noul(
        instructions="Are there rendering defects on or around the dancers in `view`?",
        criteria={"true": "Floating dark or coloured blobs, smeared streaks, holes through a body, or pieces of a body "
                          "hanging in the air.",
                  "false": "The dancers look solid and clean; ordinary motion blur or soft edges are not defects."}),
    "cut_off": Noul(instructions="Is part of either dancer cut off by the edge of `view` (head, feet, hands or a side "
                                 "of the body outside the image)?"),
}
SAME_POSE = Noul(
    instructions="Do the dancers in `view` hold the same body poses as the dancers in `reference`?",
    criteria={"true": "The same arm and leg positions and the same partner hold, allowing for a slightly different "
                      "look, lighting or sharpness.",
              "false": "At least one dancer's arms, legs, facing direction or hold clearly differ from `reference`."})


def prepare_image(path: Path | str, max_side: int = MAX_SIDE) -> bytes:
    """Upright baseline JPEG with the longest side <= max_side and both sides >= 280 px (TypeSafe's limits)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    img.thumbnail((max_side, max_side))
    if min(img.size) < 280:
        s = 280 / min(img.size)
        img = img.resize((max(280, round(img.width * s)), max(280, round(img.height * s))))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _asset(data: bytes) -> tuple[str, dict]:
    return f"img-{uuid.uuid4()}", {"type": "image", "format": "jpg", "data_base64": base64.b64encode(data).decode()}


def build_request(view: Path | str, reference: Path | str | None = None, source: Path | str | None = None,
                  about: str = ABOUT) -> dict:
    """state, questions and assets of one frame's request (kept separate from the call for tests and logging).

    reference: the real phone at the view's own pose and time (held-out views) -> adds same_pose.
    source: the input phone's frame at the same time (any view): shows Jev what the people normally look like."""
    vk, va = _asset(prepare_image(view))
    state, assets, qs = {"about": about, "view": vk}, {vk: va}, dict(QUESTIONS)
    if reference is not None:
        rk, ra = _asset(prepare_image(reference))
        state["about"] += ABOUT_REF
        state["reference"] = rk
        assets[rk] = ra
        qs["same_pose"] = SAME_POSE
    if source is not None:
        sk, sa = _asset(prepare_image(source))
        state["about"] += ABOUT_SRC
        state["source"] = sk
        assets[sk] = sa
    return {"state": state, "questions": qs, "assets": assets}


def judge_frame(view: Path | str, reference: Path | str | None = None, source: Path | str | None = None, *,
                client=None, model: str = MODEL, about: str = ABOUT) -> dict:
    """-> {question: P(yes)} for one frame, plus "_model" and "_tokens"."""
    if client is None:
        from ..judgments import client as _client
        client = _client()
    req = build_request(view, reference, source, about)
    r = client.system_one(state=req["state"], questions=req["questions"], model=model,
                          extra_headers=IMAGE_HEADERS, extra_body={"assets": req["assets"]})
    out = {k: float(v.noul) for k, v in r.answers.items()}
    out["_model"] = r.model
    out["_tokens"] = int(getattr(r.usage, "input_tokens", 0) or 0)
    return out


def frame_verdict(ans: dict, th: dict | None = None) -> tuple[bool, list[str]]:
    """A frame is bad when a decisive defect is likely (p >= its threshold) or, with a reference, the pose is unlikely
    to match. Advisory questions never make a frame bad (frame rows still carry their answers)."""
    th = {**THRESHOLDS, **(th or {})}
    why = [f"{k} {ans[k]:.2f}" for k in DECISIVE if k in ans and ans[k] >= th[k]]
    if "same_pose" in ans and ans["same_pose"] < th["same_pose"]:
        why.append(f"same_pose {ans['same_pose']:.2f}")
    return not why, why


def judge_views(frames: dict[str, list[dict]], allow_upload: bool = False, workers: int = 4, client=None,
                model: str = MODEL, th: dict | None = None, about: str = ABOUT) -> dict:
    """frames {view: [{"t": s, "path": jpg, "reference": real jpg or None, "source": input jpg or None}]} -> per
    view: per-frame answers, the bad frames and ok (False when >= min_bad_frames AND > bad_frame_frac of the judged
    frames are bad, or any request failed); token total."""
    if not allow_upload:
        raise PermissionError("these frames show real people: pass allow_upload (the owner's go-ahead to send them "
                              "to TypeSafe) or judge synthetic test images only")
    th = {**THRESHOLDS, **(th or {})}
    jobs = [(v, f) for v, fs in frames.items() for f in fs]

    def one(job):
        v, f = job
        try:
            return v, f, judge_frame(f["path"], f.get("reference"), f.get("source"), client=client, model=model,
                                     about=about), None
        except Exception as e:  # a failed request is reported, never silently counted as a pass
            return v, f, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        done = list(ex.map(one, jobs))
    out, tokens = {}, 0
    for v in frames:
        rows = []
        for vv, f, ans, err in done:
            if vv != v:
                continue
            if ans is None:
                rows.append({"t": f.get("t"), "path": str(f["path"]), "error": err, "ok": None})
                continue
            ok, why = frame_verdict(ans, th)
            tokens += ans.get("_tokens", 0)
            rows.append({"t": f.get("t"), "path": str(f["path"]), "ok": ok, "why": why,
                         "answers": {k: round(x, 3) for k, x in ans.items() if not k.startswith("_")},
                         "model": ans.get("_model")})
        judged = [r for r in rows if r["ok"] is not None]
        bad = sum(not r["ok"] for r in judged)
        frac = bad / len(judged) if judged else None
        ok = (frac is not None and len(judged) == len(rows)
              and not (bad >= th["min_bad_frames"] and frac > th["bad_frame_frac"]))
        out[v] = {"frames": rows, "judged": len(judged), "errors": len(rows) - len(judged), "bad": bad,
                  "bad_frac": frac, "ok": ok}
    return {"views": out, "input_tokens": tokens, "model": model, "thresholds": th}


def sample_frames(work: Path, cam: str, every_s: float = 1.0, reference_work: Path | None = None,
                  reference_cam: str | None = None) -> list[dict]:
    """About one frame per every_s seconds of frames/<cam>/, each with the reference camera's nearest frame if given."""
    import json

    import numpy as np
    meta = json.loads((Path(work) / "frames" / cam / "frames.json").read_text())
    t = np.array([f["t"] for f in meta["frames"]])
    picks, last = [], -1e9
    for i, ti in enumerate(t):
        if ti - last >= every_s - 1e-6:
            picks.append(i)
            last = ti
    ref = None
    if reference_work is not None:
        rm = json.loads((Path(reference_work) / "frames" / (reference_cam or cam) / "frames.json").read_text())
        ref = (np.array([f["t"] for f in rm["frames"]]), rm["frames"],
               Path(reference_work) / "frames" / (reference_cam or cam))
    out = []
    for i in picks:
        row = {"t": float(t[i]), "path": str(Path(work) / "frames" / cam / meta["frames"][i]["file"]), "reference": None}
        if ref is not None:
            j = int(np.argmin(np.abs(ref[0] - t[i])))
            if abs(ref[0][j] - t[i]) <= 1 / 50:
                row["reference"] = str(ref[2] / ref[1][j]["file"])
        out.append(row)
    return out
