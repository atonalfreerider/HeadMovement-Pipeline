r"""Jev (TypeSafe System One) realism judge of the appearance comparison (2026-10-19): which renderer looks most like a
real photograph of the two dancers? Step 'jev' of `python -m dancecap.mono4d.appearance_compare compose`.

For every real view of views.json (18: phone 03 x 6 instants = the held-out phone, + phones 02 / 06), on the spec's crop:
    (a) choice      ONE Choice over the six methods' renders (on neutral grey), labelled A-F in a fresh random order per
                    question (--shuffles questions per view), the real crop in the state as reference context (never an
                    option): "Which render looks most like a real photograph of these two people: skin, face, hair,
                    clothing fabric and folds?" + "none" (none of them looks like a photograph)
    (a') choice_ctl the same with the REAL photo cut out onto the same grey (the dancers' SAM masks) added as a seventh
                    option: a sanity check that Jev can tell a photo from the renders at all
    (b) photo       a Noul per image, no reference: "Does this look like a photo or video of real people rather than a 3D
                    game or CG character?" (six renders + controls: the real crop with its background, the real cut out
                    onto grey)
    (c) same        a Noul per render vs the real crop: "Is this the same two people, recognisable by face, hair and
                    clothes?" (+ controls: the same instant from the other high-res phone cut out onto grey = same
                    people, and a frame of another couple (carlos_aline) = different people)
Images: judge.prepare_image (upright baseline JPEG, longest side <= 1024, both sides >= 280), sent as assets in
extra_body with the header TypeSafe-Danger-Zone-Images (judge.py's convention). Raw answers -> compare/jev.json, the
images -> compare/jev/<view>/ (likeness data stays in work/). Owner consent 2026-10-19: "it's fine to use any images and
likenesses - this is an open collaboration"; the caller still has to pass allow_upload.
Price: jev-1.13 / 1.14 bill input tokens only, $0.042 per million (docs.typesafe.ai/models, read 2026-10-19).
"""
from __future__ import annotations

import base64
import json
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from typesafe_sdk import Choice, Noul

from ..take import PACKAGE_ROOT
from .appearance_compare import ROLES, _rel, read_mask
from .appearance_compose import ORDER, Ctx, load_rgba, log, over
from .judge import IMAGE_HEADERS, MODEL, prepare_image

USD_PER_TOKEN = 0.042e-6
LETTERS = "ABCDEFGHIJ"
NEG_CONTROL = ("work/carlos_aline/frames/A/f005.jpg", (180, 30, 420, 335))   # another couple (x0, y0, x1, y1)

ABOUT_CHOICE = ("Images of two people partner dancing in a dance studio (one leads, one follows; they hold each other "
                "closely and turn). `reference` is a real photo of them. Each image in `renders` shows the same two "
                "people at the same moment from the same camera, produced by a different 3D method and cut out onto "
                "a plain grey background (no studio). Judge the people only, not the grey background.")
ABOUT_PHOTO = ("`view` shows two people partner dancing. The background may be a real dance studio or a plain grey "
               "fill where the people were cut out. Judge the people only, not the background.")
ABOUT_SAME = ("`reference` is a real photo of two people partner dancing in a studio. `view` shows two people seen "
              "from about the same camera position on a plain grey or studio background; it may be a photo or a 3D "
              "render. Judge the people only, not the background.")

Q_CHOICE = "Which render looks most like a real photograph of these two people: skin, face, hair, clothing fabric and folds?"
Q_PHOTO = Noul(
    instructions="Does `view` look like a photo or video of real people rather than a 3D game or CG character?",
    criteria={"true": "The people look like real humans captured by a camera: natural skin, faces, hair and cloth.",
              "false": "The people look computer-generated: like a 3D game or CG character, a mannequin, a doll, or a "
                       "smeared or blobby reconstruction."})
Q_SAME = Noul(
    instructions="Is `view` the same two people as `reference`, recognisable by face, hair and clothes?",
    criteria={"true": "Each person in `view` matches one in `reference` in face, hair (colour, length, style) and "
                      "clothing (garments and colours), allowing for a different sharpness, pose detail or rendering "
                      "style.",
              "false": "At least one person in `view` has a clearly different face, hair or clothing, or cannot be "
                       "recognised as one of the people in `reference`."})


def _asset(data: bytes) -> tuple[str, dict]:
    return f"img-{uuid.uuid4()}", {"type": "image", "format": "jpg", "data_base64": base64.b64encode(data).decode()}


def _save(img_rgb: np.ndarray, p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return p


def prepare_images(ctx: Ctx) -> dict:
    """compare/jev/<view>/{real,real_grey,<method>}.jpg on the spec's crop + the negative control."""
    root = ctx.cmp / "jev"
    out = {}
    for v in ctx.spec["real_views"]:
        x0, y0, x1, y1 = v["crop"]
        ph = ctx.photo(v["file"])
        H, W = ph.shape[:2]
        d = {"real": _save(ph[y0:y1, x0:x1], root / v["id"] / "real.jpg")}
        mu = np.zeros((H, W), np.float32)
        for r in ROLES:
            mk = read_mask(ctx.work, v["cam"], v["frame"], r, (H, W))
            if mk is not None:
                mu = np.maximum(mu, mk.astype(np.float32))
        mu = cv2.GaussianBlur(mu, (0, 0), 0.8)
        d["real_grey"] = _save(over((ph.astype(np.float32) / 255.0, mu))[y0:y1, x0:x1], root / v["id"] / "real_grey.jpg")
        for m in ORDER:
            ra = load_rgba(ctx.path(m, "real", v["id"]))
            d[m] = None if ra is None or ra[0].shape[:2] != (H, W) else \
                _save(over(ra)[y0:y1, x0:x1], root / v["id"] / f"{m}.jpg")
        out[v["id"]] = d
    neg = PACKAGE_ROOT / NEG_CONTROL[0]
    im = cv2.imread(str(neg))
    if im is not None:
        a, b, c, e = NEG_CONTROL[1]
        out["_negative"] = _save(im[b:e, a:c, ::-1], root / "_negative_other_couple.jpg")
    return out


def _call(client, state: dict, questions: dict, assets: dict) -> tuple[dict, str, int]:
    r = client.system_one(state=state, questions=questions, model=MODEL, extra_headers=IMAGE_HEADERS,
                          extra_body={"assets": assets})
    ans = {}
    for k, a in r.answers.items():
        if hasattr(a, "noul"):
            ans[k] = {"noul": float(a.noul)}
        else:
            ans[k] = {"choice": a.choice, "confidence": float(a.confidence),
                      "probabilities": {o: float(p) for o, p in dict(a.probabilities).items()}}
    return ans, r.model, int(getattr(r.usage, "input_tokens", 0) or 0)


def build_jobs(ctx: Ctx, imgs: dict, shuffles: int, seed: int = 20261019) -> list[dict]:
    rng = random.Random(seed)
    jobs = []
    other = {"03": "02", "02": "06", "06": "02"}
    for v in ctx.spec["real_views"]:
        d = imgs[v["id"]]
        meths = [m for m in ORDER if d.get(m) is not None]
        for k in range(shuffles):
            order = meths[:]
            rng.shuffle(order)
            jobs.append({"kind": "choice", "view": v["id"], "rep": k, "options": order})
        order = meths + ["real_grey"]
        rng.shuffle(order)
        jobs.append({"kind": "choice_ctl", "view": v["id"], "rep": 0, "options": order})
        for m in meths + ["real_grey", "real"]:
            jobs.append({"kind": "photo", "view": v["id"], "image": m})
        ov = f"{v['instant']}_c{other[v['cam']]}"
        for m in meths + [f"ctl_same:{ov}", "ctl_other_couple"]:
            jobs.append({"kind": "same", "view": v["id"], "image": m})
    return jobs


def _img_bytes(imgs: dict, view: str, name: str) -> tuple[bytes, str]:
    if name == "ctl_other_couple":
        p = imgs["_negative"]
    elif name.startswith("ctl_same:"):
        p = imgs[name.split(":", 1)[1]]["real_grey"]
    else:
        p = imgs[view][name]
    return prepare_image(p), _rel(p)


def run_job(client, imgs: dict, job: dict) -> dict:
    view = job["view"]
    rec = dict(job)
    try:
        if job["kind"] in ("choice", "choice_ctl"):
            rb, rp = _img_bytes(imgs, view, "real")
            rk, ra = _asset(rb)
            assets, renders, crit, files = {rk: ra}, {}, {}, {"reference": rp}
            for n, m in enumerate(job["options"]):
                L = LETTERS[n]
                b, p = _img_bytes(imgs, view, m)
                k, a = _asset(b)
                assets[k] = a
                renders[L] = k
                crit[L] = f"render {L}, the image `renders.{L}`"
                files[L] = p
            crit["none"] = "none of them looks like a photograph"
            state = {"about": ABOUT_CHOICE, "reference": rk, "renders": renders}
            ans, model, tok = _call(client, state, {"most_photographic": Choice(instructions=Q_CHOICE, criteria=crit)},
                                    assets)
            a = ans["most_photographic"]
            letter_to = {LETTERS[n]: m for n, m in enumerate(job["options"])}
            rec.update({"letters": letter_to, "files": files, "answer": a,
                        "picked": letter_to.get(a["choice"], "none"),
                        "p_method": {letter_to.get(o, o): p for o, p in a["probabilities"].items()}})
        elif job["kind"] == "photo":
            b, p = _img_bytes(imgs, view, job["image"])
            k, a = _asset(b)
            ans, model, tok = _call(client, {"about": ABOUT_PHOTO, "view": k}, {"photo_real": Q_PHOTO}, {k: a})
            rec.update({"file": p, "p": ans["photo_real"]["noul"]})
        else:
            rb, rp = _img_bytes(imgs, view, "real")
            rk, ra = _asset(rb)
            b, p = _img_bytes(imgs, view, job["image"])
            k, a = _asset(b)
            ans, model, tok = _call(client, {"about": ABOUT_SAME, "reference": rk, "view": k}, {"same_people": Q_SAME},
                                    {rk: ra, k: a})
            rec.update({"file": p, "reference": rp, "p": ans["same_people"]["noul"]})
        rec.update({"model": model, "input_tokens": tok, "error": None})
    except Exception as e:  # a failed request is recorded, never counted
        rec.update({"error": f"{type(e).__name__}: {e}"[:500], "input_tokens": 0})
    return rec


def summarise(rows: list[dict], ctx: Ctx) -> dict:
    ok = [r for r in rows if not r.get("error")]
    out = {}
    groups = {"all": lambda r: True, "cam03": lambda r: r["view"].endswith("_c03"),
              "cam02_06": lambda r: not r["view"].endswith("_c03")}
    # (a) choice
    ch = {}
    for g, f in groups.items():
        rr = [r for r in ok if r["kind"] == "choice" and f(r)]
        n = len(rr)
        per = {}
        for m in list(ORDER) + ["none"]:
            wins = sum(r["picked"] == m for r in rr)
            ps = [r["p_method"].get(m, 0.0) for r in rr]
            per[m] = {"wins": wins, "rate": round(wins / n, 3) if n else None,
                      "mean_p": round(float(np.mean(ps)), 3) if ps else None}
        ch[g] = {"questions": n, "per_method": per}
    pos = {}
    rr = [r for r in ok if r["kind"] == "choice"]
    for L in LETTERS[:len(ORDER)]:
        pos[L] = round(sum(r["answer"]["choice"] == L for r in rr) / max(len(rr), 1), 3)
    ch["position_rate"] = pos
    ch["mean_confidence"] = round(float(np.mean([r["answer"]["confidence"] for r in rr])), 3) if rr else None
    out["choice"] = {**ch["all"], "by_group": {g: ch[g] for g in groups}, "position_rate": pos,
                     "mean_confidence": ch["mean_confidence"]}
    rc = [r for r in ok if r["kind"] == "choice_ctl"]
    out["choice_control"] = {"questions": len(rc), "real_grey_wins": sum(r["picked"] == "real_grey" for r in rc),
                             "real_grey_mean_p": round(float(np.mean([r["p_method"].get("real_grey", 0) for r in rc])), 3)
                             if rc else None,
                             "wins": {m: sum(r["picked"] == m for r in rc) for m in list(ORDER) + ["real_grey", "none"]}}
    for kind in ("photo", "same"):
        res = {}
        for g, f in groups.items():
            rr = [r for r in ok if r["kind"] == kind and f(r)]
            names = sorted({r["image"].split(":")[0] for r in rr})
            res[g] = {nm: {"n": len([r for r in rr if r["image"].split(":")[0] == nm]),
                           "mean_p": round(float(np.mean([r["p"] for r in rr if r["image"].split(":")[0] == nm])), 3),
                           "min_p": round(float(np.min([r["p"] for r in rr if r["image"].split(":")[0] == nm])), 3),
                           "max_p": round(float(np.max([r["p"] for r in rr if r["image"].split(":")[0] == nm])), 3),
                           "share_ge_0.5": round(float(np.mean([r["p"] >= 0.5 for r in rr
                                                               if r["image"].split(":")[0] == nm])), 3)}
                      for nm in names}
        out[kind] = res
    tok = sum(r.get("input_tokens", 0) for r in rows)
    out["requests"] = len(rows)
    out["errors"] = len(rows) - len(ok)
    out["input_tokens"] = tok
    out["cost_usd"] = round(tok * USD_PER_TOKEN, 5)
    out["models"] = sorted({r.get("model") for r in ok if r.get("model")})
    return out


def run_jev(ctx: Ctx, allow_upload: bool = False, shuffles: int = 3, workers: int = 4, limit: int | None = None) -> dict:
    if not allow_upload:
        raise PermissionError("the jev step sends real photos and renders of the dancers to TypeSafe: pass "
                              "--allow-upload (owner consent 2026-10-19)")
    from ..judgments import client as _client
    client = _client()
    imgs = prepare_images(ctx)
    jobs = build_jobs(ctx, imgs, shuffles)
    if limit:
        jobs = jobs[:limit]
    log(f"jev: {len(jobs)} requests ({MODEL}), {workers} workers")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = list(ex.map(lambda j: run_job(client, imgs, j), jobs))
    errs = [r for r in rows if r.get("error")]
    if errs:   # one retry of the failed requests
        log(f"jev: {len(errs)} failed, retrying once: {errs[0]['error'][:200]}")
        time.sleep(5)
        again = {id(r): run_job(client, imgs, {k: r[k] for k in ("kind", "view", "rep", "options", "image") if k in r})
                 for r in errs}
        rows = [again.get(id(r), r) for r in rows]
    summ = summarise(rows, ctx)
    out = {"schema": "dancecap.mono4d.appearance_jev/1", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model_alias": MODEL, "seconds": round(time.time() - t0, 1), "methods": list(ORDER),
           "questions": {"choice": Q_CHOICE, "choice_about": ABOUT_CHOICE,
                         "photo": {"instructions": Q_PHOTO.instructions, "criteria": dict(Q_PHOTO.criteria)
                                   if isinstance(Q_PHOTO.criteria, dict) else str(Q_PHOTO.criteria),
                                   "about": ABOUT_PHOTO},
                         "same": {"instructions": Q_SAME.instructions, "criteria": dict(Q_SAME.criteria)
                                  if isinstance(Q_SAME.criteria, dict) else str(Q_SAME.criteria), "about": ABOUT_SAME}},
           "images": "compare/jev/<view>/{real,real_grey,<method>}.jpg (spec crop; renders and real_grey on grey "
                     f"{128}); judge.prepare_image before upload; negative control {NEG_CONTROL[0]} crop "
                     f"{NEG_CONTROL[1]}",
           "price": "input tokens only, $0.042 per million (jev-1.13 / 1.14)",
           "summary": summ, "rows": rows}
    p = ctx.cmp / "jev.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    log(f"jev: {summ['requests']} requests, {summ['errors']} errors, {summ['input_tokens']} input tokens = "
        f"${summ['cost_usd']:.4f}; wrote {p}")
    return {"file": _rel(p), "requests": summ["requests"], "errors": summ["errors"],
            "input_tokens": summ["input_tokens"], "cost_usd": summ["cost_usd"]}
