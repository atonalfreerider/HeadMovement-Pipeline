"""Jev (TypeSafe System One) judgments for the recap film (owner 2026-10-22: "still too many fast camera transitions. we should
hold longer on POVs and slowly orbit gaussian splats. also the physics meter is occluding the figures in the vertical video").

    python -m dancecap.film_jev need3d [--take larissa_kadu_recap] [--version v4]
        every utterance of the class transcript -> how much its moment needs the slow 3D splat orbit (with physics) rather
        than the phone video (Score 0-3) + whether it is about the floor seen from above (Noul)
        -> work/<take>/<version>/jev/need3d.json (per utterance, and per direction segment through its `say` ids)
    python -m dancecap.film_jev overlays <video.mp4> [--every 2] [--threshold 0.35]
        frames every N s -> does the skeleton-load legend (the "physics meter") or any overlay hide part of a dancer?
        -> <video>.jev_overlays.json + <video>.jev_overlays.jpg (the flagged frames); exit 1 when any frame is flagged
    python -m dancecap.film_jev povview --take T --cam 01 [--video V] [--every 1]
        a recording camera's video every N s -> is either dancer blocked (spectator, raised phone, frame edge) (Noul) + how well
        the frame shows the couple dancing (Score 0-3) -> work/<take>/film/jev_pov_<cam>.json on the take's reference clock;
        film_heat pov-table uses it as one more gate on the point-of-view shots (lucas_luiza 2026-10-23: the tripod films over
        a front row of heads and raised phones, which no detector box tells apart from a clear view). Validated 2026-10-23 on 9
        frames (6 tripod, 3 Steadicam) looked at by eye: a spectator's head over the follower 0.79, the leader half out of the
        vertical frame 0.84, clear views and one dancer overlapping the other 0.17-0.46 (the first wording, without "one dancer
        hiding the other does not count", put those at 0.5-0.67); view 1.6-2.1 on the far tripod, 1.9-2.6 on the Steadicam.
        The gate film_heat uses: blocked < 0.6 and view >= 1.5 - re-check on more frames before trusting it elsewhere.

Code owns the plan: film_recap_flow reads need3d.json as one more weight (phone holds over the low-need stretches, slow orbits
over the high-need ones); the overlay check is a gate on recordings, and every flagged frame is looked at by a person or agent.
Validated 2026-10-22 on 6 frames of the v3 vertical film: the legend over the leader's face (0:30) 0.80, over the top of the
follower's head (3:20) 0.46, the 4 clean 3D frames <= 0.07; the transcript scores put greetings / jokes at ~0 and the
counterbalance / weight / connection explanations at ~2.9 (77 utterances, 72k input tokens, about $0.003).
Credentials: dancecap.judgments.client() (TYPESAFE_API_KEY or ~/Desktop/jev-key.txt; never printed). Frames go to TypeSafe as
images (owner consent 2026-10-19: "it's fine to use any images and likenesses - this is an open collaboration").
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from typesafe_sdk import Noul, Score

from .take import PACKAGE_ROOT

MODEL = "jev-latest"
OVERLAY_THRESHOLD = 0.35   # 6-frame validation: positives 0.46 / 0.80, negatives <= 0.07 - re-check on more frames before trusting
CONTEXT = 2                 # utterances on each side given as context

ABOUT_TRANSCRIPT = ("Utterances of a Brazilian zouk class taught by a couple (Larissa and Kadu), in order. `target` names the "
                    "one to judge; the others are context.")
NEED3D_LEVELS = [
    "Talk that does not describe body mechanics: greetings, jokes, logistics, praise, asking the class to try or to change "
    "partners, or talking about the music or a feeling. The class video of the teachers is all a viewer needs.",
    "A quick mention or the name of a move or position, or counting the music, without explaining how the body does it. "
    "The class video shows it well enough.",
    "An explanation of a step, a direction of travel, timing, or where the partner goes, which is easier to follow from a "
    "chosen angle around the couple but can still be followed on the class video.",
    "An explanation of body mechanics the class video hides: the counterbalance pivot point (the teacher's 'back to the "
    "axis'), the axis, weight transfer or the share of weight on each foot, pushing or pulling through the connection, "
    "tension or compression in the arms, or the exact placement of the feet. A 3D view with physics overlays shows it best.",
]

ABOUT_FRAME = ("`frame` is one frame of a dance-lesson film: two dancers (a man leading, a woman following) shown either in "
               "the real class video (a studio with a seated audience) or as 3D figures with coloured skeleton lines on a "
               "dark background. The film draws overlays on top: captions at the bottom, small labels, and a small "
               "horizontal colour-bar legend titled 'Skeleton load - ESTIMATED from motion' (a red-to-blue bar with the "
               "words tension, neutral, compression).")
Q_LEGEND = Noul(
    instructions="Does the colour-bar legend titled 'Skeleton load - ESTIMATED from motion' cover any part of either of the "
                 "two dancers in `frame` (the couple dancing, not the seated audience)?",
    criteria={"true": "Part of the legend panel or its text is drawn on top of a dancer's head, hair, face, shoulders, arms "
                      "or body, so that part of the dancer is hidden behind it.",
              "false": "The legend is only over background (walls, floor, audience, dark empty space), or there is no such "
                       "legend in the frame; no part of either dancer is hidden behind it."})
Q_HEAD = Noul(
    instructions="Does any text, caption, label, legend or panel drawn over `frame` cover part of either dancer's head or "
                 "face (the couple dancing, not the seated audience)?",
    criteria={"true": "Some overlay hides part of a dancer's head, hair or face.",
              "false": "No overlay is drawn over either dancer's head or face."})


ABOUT_POV = ("`frame` is one frame of a video recorded at a Brazilian zouk Jack & Jill competition heat: one couple (a man "
             "leading, a woman following) dances on a ballroom floor; judges and spectators sit or stand around the floor, and "
             "spectators near the camera may appear in the foreground (heads, raised phones).")
Q_BLOCKED = Noul(
    instructions="Is either of the two dancers of the couple in `frame` partly hidden by a SPECTATOR or an object between the "
                 "camera and the couple (a spectator's head or body, a raised phone, an arm), or cut off by the edge of the "
                 "picture? One dancer hiding part of the other is normal in this dance and does not count.",
    criteria={"true": "A spectator, a phone or another object in front of the couple hides part of a dancer (head, torso, arms, "
                      "legs or feet), or part of a dancer falls outside the picture.",
              "false": "Nothing but the dancers themselves is in front of the couple and both are inside the picture; one dancer "
                       "may overlap the other."})
POV_VIEW_LEVELS = [
    "The couple cannot be followed: one or both dancers are missing, mostly hidden, badly blurred or tiny.",
    "The couple is visible but the view is poor: much of a dancer is hidden or cut off, or the picture is very blurred or dark.",
    "A usable view: both dancers are visible and the dance can be followed, with small parts hidden or slightly cut off.",
    "A clear view of the whole couple: both dancers fully in the picture, sharp, unobstructed, large enough to see the movement.",
]
Q_VIEW = Score(instructions="How well does `frame` show the two dancers of the couple dancing?", criteria=POV_VIEW_LEVELS)


def pov_request(frame: Path) -> dict:
    from .mono4d.judge import _asset, prepare_image
    k, asset = _asset(prepare_image(frame))
    return {"state": {"about": ABOUT_POV, "frame": k}, "questions": {"blocked": Q_BLOCKED, "view": Q_VIEW}, "assets": {k: asset}}


def judge_pov_frame(frame: Path, client=None) -> dict:
    from .mono4d.judge import IMAGE_HEADERS
    req = pov_request(frame)
    r = (client or _client()).system_one(state=req["state"], questions=req["questions"], model=MODEL,
                                         extra_headers=IMAGE_HEADERS, extra_body={"assets": req["assets"]})
    v = r.answers["view"]
    return {"blocked": round(float(r.answers["blocked"].noul), 3), "view": round(float(v.score), 3),
            "view_conf": round(float(v.confidence), 3), "tokens": int(getattr(r.usage, "input_tokens", 0) or 0)}


def _client():
    from .judgments import client
    return client()


# ------------------------------------------------------------------------------------------------ need3d (text)

def need3d_questions(uid: str) -> dict:
    """The two questions about utterance `uid` (its id is spelled into the path so the question is self-contained)."""
    return {
        "need_3d": Score(
            instructions=f"How much would a viewer of a recap film of this zouk class gain from seeing the moment of "
                         f"`utterances.{uid}.text` (said by `utterances.{uid}.speaker`) as a slowly orbiting 3D view of the "
                         f"two teachers with physics overlays, instead of the class phone video of them? Use the surrounding "
                         f"utterances only as context.",
            criteria=NEED3D_LEVELS),
        "overhead": Noul(
            instructions=f"Does `utterances.{uid}.text` describe something best seen from directly above the dancers: the "
                         f"path they travel on the floor, a turn on the spot, the foot pattern, or where each dancer stands "
                         f"on the floor?",
            criteria={"true": "The words are about floor paths, rotation on the floor, footwork pattern or floor positions.",
                      "false": "The words are about anything else (body mechanics seen from the side, talk, music, "
                               "feelings)."}),
    }


def utterances(take: str) -> list[dict]:
    t = json.loads((PACKAGE_ROOT / "work" / take / "speech" / "transcript_final.json").read_text(encoding="utf-8"))
    return [{"id": u["id"], "speaker": u["speaker"], "text": u.get("text_clean") or u["text"],
             "start": u.get("start"), "end": u.get("end")} for u in t["utterances"]]


def judge_utterance(us: list[dict], i: int, client=None) -> dict:
    u = us[i]
    ctx = us[max(0, i - CONTEXT): i + CONTEXT + 1]
    state = {"about": ABOUT_TRANSCRIPT, "target": u["id"],
             "utterances": {x["id"]: {"speaker": x["speaker"], "text": x["text"]} for x in ctx}}
    r = (client or _client()).system_one(state=state, questions=need3d_questions(u["id"]), model=MODEL)
    a, o = r.answers["need_3d"], r.answers["overhead"]
    return {"id": u["id"], "speaker": u["speaker"], "start": u["start"], "end": u["end"], "text": u["text"],
            "need_3d": round(float(a.score), 3), "need_3d_conf": round(float(a.confidence), 3),
            "need_3d_p": {int(k): round(float(v), 3) for k, v in dict(a.probabilities).items()},
            "overhead": round(float(o.noul), 3), "tokens": int(getattr(r.usage, "input_tokens", 0) or 0), "model": r.model}


SAY_ID = re.compile(r"\bu\d{3}\b")


def per_segment(direction: dict, by_id: dict) -> list[dict]:
    """Each direction segment's need: the max and the word-weighted mean over the utterances its `say` names (None = silent)."""
    out = []
    for s in direction.get("segments") or []:
        ids = SAY_ID.findall(str(s.get("say") or ""))
        rows = [by_id[i] for i in ids if i in by_id]
        if not rows:
            out.append({"id": s["id"], "film": s.get("film"), "need_3d_max": None, "need_3d_mean": None, "overhead": None})
            continue
        w = [max(1, len(r["text"].split())) for r in rows]
        out.append({"id": s["id"], "film": s.get("film"), "utterances": ids,
                    "need_3d_max": max(r["need_3d"] for r in rows),
                    "need_3d_mean": round(sum(r["need_3d"] * k for r, k in zip(rows, w)) / sum(w), 3),
                    "overhead": max(r["overhead"] for r in rows)})
    return out


def cmd_need3d(a) -> int:
    us = utterances(a.take)
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda i: judge_utterance(us, i), range(len(us))))
    by_id = {r["id"]: r for r in rows}
    ddir = PACKAGE_ROOT / "work" / a.take / a.version / "direction" / "direction.json"
    segs = per_segment(json.loads(ddir.read_text(encoding="utf-8")), by_id) if ddir.exists() else []
    out = PACKAGE_ROOT / "work" / a.take / a.version / "jev" / "need3d.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "hm film jev need3d v1", "model": MODEL, "levels": NEED3D_LEVELS, "utterances": rows,
           "segments": segs, "direction": str(ddir) if segs else None,
           "input_tokens": sum(r["tokens"] for r in rows), "usd": round(sum(r["tokens"] for r in rows) * 0.042e-6, 5)}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[film_jev] {len(rows)} utterances, {len(segs)} segments, {doc['input_tokens']} tokens (${doc['usd']}) -> {out}")
    return 0


# ------------------------------------------------------------------------------------------------ overlays (images)

def overlay_request(frame: Path) -> dict:
    from .mono4d.judge import _asset, prepare_image
    k, asset = _asset(prepare_image(frame))
    return {"state": {"about": ABOUT_FRAME, "frame": k}, "questions": {"legend_over_dancer": Q_LEGEND,
                                                                        "overlay_over_head": Q_HEAD}, "assets": {k: asset}}


def judge_frame(frame: Path, client=None) -> dict:
    from .mono4d.judge import IMAGE_HEADERS
    req = overlay_request(frame)
    r = (client or _client()).system_one(state=req["state"], questions=req["questions"], model=MODEL,
                                         extra_headers=IMAGE_HEADERS, extra_body={"assets": req["assets"]})
    return {q: round(float(x.noul), 3) for q, x in r.answers.items()} | {
        "tokens": int(getattr(r.usage, "input_tokens", 0) or 0)}


def sample_frames(video: Path, every_s: float, out_dir: Path, width: int = 720) -> list[tuple[float, Path]]:
    """One JPEG every `every_s` seconds (ffmpeg fps filter), scaled to `width` px wide."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps=1/{every_s},scale={width}:-2",
                    "-q:v", "3", str(out_dir / "f%05d.jpg")], check=True)
    return [((i) * every_s, p) for i, p in enumerate(sorted(out_dir.glob("f*.jpg")))]


def flagged(row: dict, threshold: float) -> bool:
    return max(row.get("legend_over_dancer", 0.0), row.get("overlay_over_head", 0.0)) >= threshold


def contact_sheet(rows: list[dict], out: Path, cols: int = 6, tile_w: int = 320) -> Path | None:
    from PIL import Image, ImageDraw
    rows = [r for r in rows if r.get("flagged")]
    if not rows:
        return None
    ims = [Image.open(r["path"]) for r in rows[:48]]
    th = int(ims[0].height * tile_w / ims[0].width)
    sheet = Image.new("RGB", (cols * tile_w, ((len(ims) + cols - 1) // cols) * (th + 22)), "white")
    for i, (im, r) in enumerate(zip(ims, rows)):
        x, y = (i % cols) * tile_w, (i // cols) * (th + 22)
        sheet.paste(im.resize((tile_w, th)), (x, y + 22))
        ImageDraw.Draw(sheet).text((x + 4, y + 4), f"{r['t']:.0f}s legend {r['legend_over_dancer']:.2f} head "
                                                   f"{r['overlay_over_head']:.2f}", fill="black")
    sheet.save(out, quality=85)
    return out


def cmd_overlays(a) -> int:
    video = Path(a.video)
    with tempfile.TemporaryDirectory() as td:
        frames = sample_frames(video, a.every, Path(td))
        keep = video.with_suffix(".jev_frames")
        keep.mkdir(exist_ok=True)
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda tp: judge_frame(tp[1]), frames))
        rows = []
        for (t, p), r in zip(frames, res):
            row = {"t": t, **r}
            row["flagged"] = flagged(row, a.threshold)
            if row["flagged"]:
                dst = keep / f"t{t:07.1f}.jpg"
                dst.write_bytes(p.read_bytes())
                row["path"] = str(dst)
            rows.append(row)
    n = sum(r["flagged"] for r in rows)
    sheet = contact_sheet(rows, video.with_suffix(".jev_overlays.jpg"))
    doc = {"schema": "hm film jev overlays v1", "video": str(video), "every_s": a.every, "threshold": a.threshold,
           "model": MODEL, "frames": len(rows), "flagged": n, "sheet": str(sheet) if sheet else None,
           "input_tokens": sum(r["tokens"] for r in rows), "rows": rows}
    video.with_suffix(".jev_overlays.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"[film_jev] {video.name}: {n} of {len(rows)} frames flagged (>= {a.threshold}); sheet {sheet}")
    return 1 if n else 0


def cmd_povview(a) -> int:
    import numpy as np
    w = PACKAGE_ROOT / "work" / a.take
    z = np.load(w / "cameras" / f"{a.cam}.npz", allow_pickle=True)
    t_ref = np.asarray(z["t"], float)
    pts = np.asarray(z["pts"], float) if "pts" in z.files else t_ref
    video = Path(a.video) if a.video else None
    if video is None:
        from .take import Take
        video = Take.load(a.take).source(a.cam).path
    with tempfile.TemporaryDirectory() as td:
        frames = sample_frames(video, a.every, Path(td))
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda tp: judge_pov_frame(tp[1]), frames))
    rows = [{"t_video": round(t, 3), "t": round(float(np.interp(t, pts, t_ref)), 3), **r} for (t, _), r in zip(frames, res)]
    doc = {"schema": "hm film jev pov v1", "take": a.take, "cam": a.cam, "video": str(video), "every_s": a.every, "model": MODEL,
           "questions": {"blocked": Q_BLOCKED.instructions, "view": Q_VIEW.instructions, "view_levels": POV_VIEW_LEVELS},
           "input_tokens": sum(r["tokens"] for r in rows), "rows": rows}
    out = w / "film" / f"jev_pov_{a.cam}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    good = sum(1 for r in rows if r["blocked"] < 0.6 and r["view"] >= 1.5)
    print(f"[film_jev] {a.take}/{a.cam}: {good} of {len(rows)} frames clear (blocked < 0.6, view >= 1.5); "
          f"{doc['input_tokens']} input tokens -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.film_jev", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("need3d")
    sp.add_argument("--take", default="larissa_kadu_recap")
    sp.add_argument("--version", default="v4")
    sp.add_argument("--workers", type=int, default=6)
    sp.set_defaults(fn=cmd_need3d)
    sp = sub.add_parser("overlays")
    sp.add_argument("video")
    sp.add_argument("--every", type=float, default=2.0)
    sp.add_argument("--threshold", type=float, default=OVERLAY_THRESHOLD)
    sp.add_argument("--workers", type=int, default=6)
    sp.set_defaults(fn=cmd_overlays)
    sp = sub.add_parser("povview")
    sp.add_argument("--take", required=True)
    sp.add_argument("--cam", required=True)
    sp.add_argument("--video", help="default: the take's video of that camera")
    sp.add_argument("--every", type=float, default=1.0)
    sp.add_argument("--workers", type=int, default=8)
    sp.set_defaults(fn=cmd_povview)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
