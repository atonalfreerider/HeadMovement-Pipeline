"""The Atlas sceptical test, reproducible: one phone frame -> Atlas -> the take's OTHER phones' poses, scored against them.

    python -m dancecap.mono4d atlas-test larissa_kadu_38 --cam 06 --t 60 --targets 01,03,05,07 [--clean-plate] [--no-jev]

Steps (work/<take>/mono4d/atlas/<cam>_<t ms>/):
    1. images2PosedRGBD on the input phone's frame at its own aspect (portrait 720x1280)       -> pose.json, depth .npy
    2. world_from_atlas: rotation from the two cameras, scale from the take's room SfM points (people masked)
    3. atlasGenerate at each target phone's real pose (1280x720 grid keeping the phone's field of view) -> gen_<cam>.png
    4. score each generated band against the real phone frame of the same instant:
       YOLO11x-pose people count, the two largest people's box IoU and PCK@0.1 (holdout.pck), Jev same_pose /
       duplicate_person / render_defects (judge.py; the input frame goes along as `source`)
    5. optional --clean-plate: atlasMasked on a landscape band around the dancers, people masked, Jev "any person left"
-> report.json + sheet.jpg (input | real / generated per target). Every Atlas call is a paid beta task; the report
lists them. Uploads happen only with --allow-upload (owner 2026-10-19: any images and likenesses are fine for tests).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import mamma_dyn as md
from ..take import PACKAGE_ROOT
from . import atlas as A
from . import holdout, judge


def _yolo():
    from ultralytics import YOLO
    return YOLO(str(PACKAGE_ROOT / "weights" / "yolo11x-pose.pt"))


def _detect(model, img: np.ndarray):
    r = model(img[..., ::-1].copy(), verbose=False, conf=0.4)[0]  # ultralytics takes BGR arrays
    if r.keypoints is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0, 17, 3))
    return r.boxes.xyxy.cpu().numpy(), np.concatenate([r.keypoints.xy.cpu().numpy(),
                                                        r.keypoints.conf.cpu().numpy()[..., None]], -1)


def pair_scores(model, real: np.ndarray, gen: np.ndarray) -> dict:
    """The two largest people of the real frame matched to the generated one by box IoU -> IoU, PCK@0.1, error."""
    br, kr = _detect(model, real)
    bg, kg = _detect(model, gen)
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
    rows = []
    for k in sorted(range(len(br)), key=lambda i: -area(br[i]))[:2]:
        if not len(bg):
            rows.append({"iou": 0.0, "pck": 0.0, "err": None})
            continue
        g = max(range(len(bg)), key=lambda m: holdout.box_iou(br[k], bg[m]))
        diag = float(np.hypot(br[k][2] - br[k][0], br[k][3] - br[k][1]))
        p, e = holdout.pck(kg[g, :, :2], kr[k, :, :2], kr[k, :, 2], diag)
        rows.append({"iou": round(holdout.box_iou(br[k], bg[g]), 3), "pck": round(p, 3), "err": round(e, 3)})
    return {"people_real": int(len(br)), "people_gen": int(len(bg)), "dancers": rows}


def run(take: str, cam: str, t: float, targets: list[str], clean_plate: bool = False, use_jev: bool = True,
        allow_upload: bool = False, client: A.AtlasClient | None = None) -> dict:
    if not allow_upload:
        raise PermissionError("atlas-test uploads the take's frames to World Labs: pass --allow-upload")
    from PIL import Image, ImageDraw, ImageFilter
    work = PACKAGE_ROOT / "work" / take
    out = work / "mono4d" / "atlas" / f"{cam}_{int(round(t * 1000))}"
    out.mkdir(parents=True, exist_ok=True)
    c = client or A.AtlasClient()
    calls = []
    tracks = md.load_tracks(work)
    tr = tracks[cam]
    i = int(np.argmin(np.abs(tr.t - t)))
    t_in = float(tr.t[i])
    aspect = (720, 1280) if tr.height > tr.width else (1280, 720)
    t0 = time.time()
    views, _ = A.pose_images(c, [tr.files[i]], target_resolution=aspect)
    calls.append({"task": "images2PosedRGBD", "s": round(time.time() - t0, 1)})
    v = views[0]
    A.save_view(v, out, "pose")
    sim = json.loads((work / "cameras" / "world_from_sfm.json").read_text())
    P = A.read_colmap_points(work / "cameras" / "sfm" / "points3D.bin")
    Pw = float(sim["s"]) * P @ np.asarray(sim["R"], float).T + np.asarray(sim["t"], float)
    pm_path = work / "people_masks" / cam / f"{i:06d}.png"
    keep = (np.asarray(Image.open(pm_path).convert("L").resize(aspect)) < 128) if pm_path.exists() else None
    w = A.world_from_atlas(v, tr.K[i], tr.R[i], tr.tvec[i], tr.width, tr.height, Pw, keep)
    rep = {"take": take, "cam": cam, "t": t_in, "frame": i, "atlas_camera": v.camera, "world_from_atlas": w,
           "atlas_focal_vs_ours": w["focal_ratio"], "targets": {}, "calls": calls}
    tgt_cams, boxes, reals = [], {}, {}
    for tc in targets:
        trt = tracks[tc]
        j = int(np.argmin(np.abs(trt.t - t_in)))
        ca, box = A.camera_in_atlas(trt.K[j], trt.R[j], trt.tvec[j], trt.width, trt.height, w)
        tgt_cams.append(ca)
        boxes[tc], reals[tc] = box, (trt.files[j], float(trt.t[j]))
    t0 = time.time()
    gen, meta = A.generate(c, [v], tgt_cams)
    calls.append({"task": "atlasGenerate", "targets": len(tgt_cams), "s": round(time.time() - t0, 1)})
    rep["prompt_used"] = meta.get("promptUsed")
    model = _yolo()
    H = 640
    tiles = [("input " + cam, Image.fromarray(v.rgb))]
    for tc, gv in zip(targets, gen):
        x0, y0, x1, y1 = boxes[tc]
        band = Image.fromarray(gv.rgb).crop((x0, y0, x1, y1))
        real = Image.open(reals[tc][0]).convert("RGB")
        band = band.resize(real.size, Image.LANCZOS)  # compare at the real phone's own resolution
        band.save(out / f"gen_{tc}.png")
        real.save(out / f"real_{tc}.png")
        row = {"real_t": reals[tc][1], "box": boxes[tc], **pair_scores(model, np.asarray(real), np.asarray(band))}
        if use_jev:
            row["jev"] = {k: round(x, 3) for k, x in judge.judge_frame(out / f"gen_{tc}.png", out / f"real_{tc}.png",
                                                                      out / "pose.png").items() if not k.startswith("_")}
        rep["targets"][tc] = row
        tiles += [(f"real {tc}", real), (f"Atlas {tc}", band)]
    pcks = [d["pck"] for r in rep["targets"].values() for d in r["dancers"]]
    rep["pck_median"] = float(np.median(pcks)) if pcks else None
    if clean_plate:
        rep["clean_plate"] = _clean_plate(c, work, cam, i, v, out, use_jev, calls)
    sheet = Image.new("RGB", (sum(int(im.width * H / im.height) + 10 for _, im in tiles), H + 24), "white")
    d, x = ImageDraw.Draw(sheet), 0
    for name, im in tiles:
        im = im.resize((int(im.width * H / im.height), H))
        sheet.paste(im, (x, 24))
        d.text((x + 4, 6), name, fill="black")
        x += im.width + 10
    sheet.save(out / "sheet.jpg", quality=88)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    del ImageFilter
    return rep


def _clean_plate(c: A.AtlasClient, work: Path, cam: str, i: int, v: A.PosedView, out: Path, use_jev: bool,
                 calls: list) -> dict:
    """atlasMasked on a 1280x720 band around the dancers (people masked) -> clean_plate.png (+ Jev checks)."""
    from PIL import Image, ImageFilter
    from typesafe_sdk import Noul
    src = Image.open(work / "frames" / cam / f"{i:06d}.jpg").convert("RGB")
    pm = Image.open(work / "people_masks" / cam / f"{i:06d}.png").convert("L").filter(ImageFilter.MaxFilter(21))
    s = 1280 / src.width
    big, bigm = src.resize((1280, round(src.height * s)), Image.LANCZOS), pm.resize((1280, round(src.height * s)))
    rows = np.nonzero(np.asarray(bigm).max(1) > 127)[0]
    y0 = int(np.clip((rows.min() + rows.max()) // 2 - 360, 0, big.height - 720)) if len(rows) else (big.height - 720) // 2
    rgb, keep = np.asarray(big)[y0:y0 + 720], np.asarray(bigm)[y0:y0 + 720] < 128
    it, sa = v.camera["intrinsics"], 1280 / v.camera["intrinsics"]["width"]
    cam_band = {"extrinsics": v.camera["extrinsics"], "intrinsics": {
        "fx": it["fx"] * sa, "fy": it["fy"] * sa, "cx": it["cx"] * sa, "cy": it["cy"] * sa - y0, "width": 1280, "height": 720}}
    t0 = time.time()
    cv, _ = A.masked(c, [(rgb, keep, cam_band)], enhance=False,
                     prompt="An empty indoor dance studio with a glossy black dance floor, walls and ceiling lights. "
                            "Nobody is in the room.")
    calls.append({"task": "atlasMasked", "s": round(time.time() - t0, 1)})
    Image.fromarray(cv[0].rgb).save(out / "clean_plate.png")
    res = {"fill_fraction": round(1 - float(keep.mean()), 3)}
    if use_jev:
        from ..judgments import client as jev
        req = judge.build_request(out / "clean_plate.png")
        q = {"any_person": Noul(instructions="Does `view` show any person or any part of a person (body, limbs, head, or "
                                             "a human-shaped silhouette)?"),
             "plausible_room": Noul(instructions="Does `view` look like a plausible, coherent photo of an empty indoor "
                                                 "dance studio?")}
        r = jev().system_one(state={"view": req["state"]["view"]}, questions=q, extra_headers=judge.IMAGE_HEADERS,
                             extra_body={"assets": req["assets"]})
        res["jev"] = {k: round(float(x.noul), 3) for k, x in r.answers.items()}
    return res
