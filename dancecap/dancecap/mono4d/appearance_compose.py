r"""Compose step of the appearance comparison (2026-10-19): the picture side-by-sides, videos and photo metrics the owner
asked for ("I need to see picture side-by-sides"), built from what every render track wrote to
compare/renders/<method>/ (views.json = the spec, appearance_compare.py spec).

    python -m dancecap.mono4d.appearance_compare compose [--take larissa_kadu_38]
        [--steps sheets,metrics,jev,videos,check] [--allow-upload] [--best-splat M] [--best-hybrid M]

Methods (columns, in this order; a missing render is a labelled grey tile, never dropped):
    avatar             the CURRENT shipped avatar (offline render of the shipped texture; Unity adds hair cards)
    ags_8phone         our avatar-guided splats, 8 phones (03 held out), take larissa_kadu (same videos)
    lhmpp_1phone       LHM++ Gaussian avatars from 8 crops of phone 06
    lhmpp_8phone       LHM++ Gaussian avatars from 16 crops of all 8 phones
    hybrid_mesh        the shipped skinned mesh with a texture baked from lhmpp_1phone
    hybrid_mesh_shell  the same with the mesh pushed out by LHM++'s clothes / hair offsets

Outputs (compare/): sheet_<instant>.jpg (rows = real phones 03 / 02 / 06, columns = real | methods, the spec's crop
for every column, renders on neutral grey, a tag per tile saying whether that camera / frame was in the method's
training data), sheet_closeup_<face_follow|face_lead|hair_follow>.jpg, sheet_turntable.jpg (rows = methods, 12
azimuths, real-phone azimuths marked), sheet_vr_seat.jpg (the headset seat, rows = instants), sheet_unity_reference.jpg,
clip_vr_seat.mp4, clip_phone03.mp4 (H.264, 30 fps, <= 1920 x 1080; real time, then 0.5x), metrics.json / metrics.md
(dancer_splats.evaluate._crop_scores on the 18 real views, LPIPS where the LHM++ pod scored it), compose.json.
Jev realism judge: appearance_jev.py (step jev). All sheets <= 1920 px wide, JPEG q92. Likeness data stays in work/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT
from .appearance_compare import AVATAR_CAPTION, ROLES, _rel, compare_dir, read_mask

TAKE = "larissa_kadu_38"
GREY = (128, 128, 128)
SHEET_W = 1920
JPEG_Q = 92
HYBRID_CAPTION = "offline render of a skinned mesh with an LHM++-baked texture (Unity-loadable; no hair cards drawn)"
METHODS = {
    "avatar": {"title": "CURRENT avatar", "built": "8 phones: texture baked from the footage",
               "what": "SMPL-X mesh + baked albedo", "kind": "mesh", "note": AVATAR_CAPTION},
    "ags_8phone": {"title": "AGS splats", "built": "8 phones (03 held out)",
                   "what": "3D Gaussians, multi-view trained", "kind": "splat",
                   "note": "own take's fit + trained pose corrections"},
    "lhmpp_1phone": {"title": "LHM++ splats", "built": "1 phone (06), 8 crops",
                     "what": "3D Gaussians, feed-forward", "kind": "splat", "note": "posed with the reference fit"},
    "lhmpp_8phone": {"title": "LHM++ splats", "built": "8 phones, 16 crops",
                     "what": "3D Gaussians, feed-forward", "kind": "splat", "note": "posed with the reference fit"},
    "hybrid_mesh": {"title": "Hybrid mesh", "built": "1 phone (06) via LHM++",
                    "what": "shipped mesh + LHM++-baked texture", "kind": "mesh", "note": HYBRID_CAPTION},
    "hybrid_mesh_shell": {"title": "Hybrid mesh + shell", "built": "1 phone (06) via LHM++",
                          "what": "mesh pushed out by LHM++ clothes/hair", "kind": "mesh", "note": HYBRID_CAPTION},
}
ORDER = tuple(METHODS)
SHORT = {"avatar": "CURRENT avatar", "ags_8phone": "AGS 8-phone", "lhmpp_1phone": "LHM++ 1-phone",
         "lhmpp_8phone": "LHM++ 8-phone", "hybrid_mesh": "Hybrid mesh", "hybrid_mesh_shell": "Hybrid mesh + shell"}
SPLATS = ("ags_8phone", "lhmpp_1phone", "lhmpp_8phone")
HYBRIDS = ("hybrid_mesh", "hybrid_mesh_shell")
LEVEL_COL = {0: (30, 140, 55), 1: (205, 125, 0), 2: (185, 35, 35)}
LEGEND = ("tags: green = held out for that method; amber = that phone was in its training data (other frames); "
          "red = this very frame was in its training data")
UNSEEN_CAM03 = ("i0_c03", "i2_c03", "i4_c03", "i5_c03")   # phone 03 frames no method trained on (checked in metrics)


def log(msg: str) -> None:
    print(f"[compose] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ inputs
class Ctx:
    """The spec, the render manifests and file lookups of one take's comparison."""

    def __init__(self, take: str = TAKE):
        self.take = take
        self.cmp = compare_dir(take)
        self.spec = json.loads((self.cmp / "views.json").read_text())
        self.work = PACKAGE_ROOT / "work" / take
        self.root = self.cmp / "renders"
        self.man = {}
        for m in ORDER:
            p = self.root / m / "manifest.json"
            self.man[m] = json.loads(p.read_text()) if p.exists() else None
        self.real = {v["id"]: v for v in self.spec["real_views"]}
        self.cu = {c["instant"]: c for c in self.spec["closeups"]}
        self.inst = {x["id"]: x for x in self.spec["instants"]}
        self.k0 = int(self.spec["clip"]["k"][0])
        self.lh_seen = {m: {r["id"]: r["seen_by_this"] for r in ((self.man[m] or {}).get("files") or {}).get("real", [])}
                        for m in ("lhmpp_1phone", "lhmpp_8phone")}
        self._photos: dict = {}
        self.jev_photo, self.jev_wins, self.jev_q = {}, {}, {}
        jp = self.cmp / "jev.json"
        if jp.exists():
            js = json.loads(jp.read_text())
            if js.get("summary", {}).get("requests", 0) >= 100:
                for r in js.get("rows", []):
                    if r.get("error"):
                        continue
                    if r["kind"] == "photo":
                        self.jev_photo[(r["view"], r["image"])] = r["p"]
                    elif r["kind"] == "choice":
                        self.jev_q[r["view"]] = self.jev_q.get(r["view"], 0) + 1
                        self.jev_wins[(r["view"], r["picked"])] = self.jev_wins.get((r["view"], r["picked"]), 0) + 1

    def jev_note(self, vid: str, m: str) -> str | None:
        p = self.jev_photo.get((vid, m))
        if p is None:
            return None
        w = self.jev_wins.get((vid, m), 0)
        return f"Jev: P(photo) {p:.2f}; most photo-like {w}/{self.jev_q.get(vid, 0)}"

    def path(self, m: str, group: str, key) -> Path:
        d = self.root / m
        lh = m.startswith("lhmpp")
        if group in ("real", "tt"):
            return d / f"{key}.png"
        if group == "closeup":
            inst, cam, box = key
            return d / "closeup_zoom" / f"{inst}_c{cam}_{box}.png" if lh else d / f"cu_{inst}_c{cam}_{box}.png"
        if group in ("clip_vr_seat", "clip_cam03"):
            return d / group / f"{int(key):04d}.png"
        if group == "role":
            return d / ("role" if lh else "roles") / f"{key}.png"
        raise KeyError(group)

    def photo(self, rel: str) -> np.ndarray:
        """Canonical frame (RGB uint8)."""
        if rel not in self._photos:
            im = cv2.imread(str(PACKAGE_ROOT / rel), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(rel)
            if len(self._photos) > 40:
                self._photos.clear()
            self._photos[rel] = im[:, :, ::-1].copy()
        return self._photos[rel]

    def role_masks(self, m: str, vid: str, shape) -> dict | None:
        im = cv2.imread(str(self.path(m, "role", vid)), cv2.IMREAD_UNCHANGED)
        if im is None or im.shape[:2] != tuple(shape):
            return None
        if m.startswith("lhmpp"):   # RGB role weights 0-255 (R lead, G follow), as cand_lhmpp.compare_score reads them
            w = im[:, :, 2::-1].astype(np.float32) / 255.0
            return {"lead": w[..., 0] > 0.5, "follow": w[..., 1] > 0.5}
        return {"lead": im == 1, "follow": im == 2}

    def status(self, m: str, vid: str) -> tuple[int, str]:
        """Was this real view's camera / frame in method m's training data? (level 0 held out, 1 phone, 2 frame)."""
        sb = self.real[vid]["seen_by"]
        if m == "avatar":
            if sb["avatar"]["frame_in_bake"]:
                return 2, "frame in bake"
            return (1, "phone in bake") if sb["avatar"]["phone_in_bake"] else (0, "held out")
        if m == "ags_8phone":
            a = sb["ags"]
            if a.startswith("train"):
                return 2, "frame trained"
            if a.startswith("test_time"):
                return 1, "phone trained"
            if a.startswith("test_cam"):
                return 0, "held out"
            return 1, a[:24]
        src = "lhmpp_1phone" if m in HYBRIDS else m
        s = self.lh_seen.get(src, {}).get(vid)
        if s is None:
            return 1, "unknown"
        if s["exact_frame_input_for"]:
            return 2, f"input frame ({'/'.join(s['exact_frame_input_for'])})"
        if s["phone_used"]:
            return 1, f"phone used ({s['nearest_input_frame_same_phone_s']:.2f} s)"
        return 0, "held out"

    def clip_index(self, inst: str) -> int:
        return int(self.inst[inst]["k"]) - self.k0


def load_rgba(p: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """RGBA PNG (straight alpha) -> (RGB float [0, 1], alpha float [0, 1])."""
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    return im[:, :, 2::-1].astype(np.float32) / 255.0, im[:, :, 3].astype(np.float32) / 255.0


def over(ra: tuple[np.ndarray, np.ndarray], bg=GREY) -> np.ndarray:
    rgb, a = ra
    out = rgb * a[..., None] + (np.asarray(bg, np.float32) / 255.0) * (1 - a[..., None])
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(img, (int(w), int(h)), interpolation=interp)


# ------------------------------------------------------------------------------------------------ drawing
_FONTS: dict = {}


def font(size: int, bold: bool = False):
    from PIL import ImageFont
    k = (size, bold)
    if k not in _FONTS:
        for name in (("arialbd.ttf", "calibrib.ttf") if bold else ("arial.ttf", "calibri.ttf")):
            try:
                _FONTS[k] = ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)
                break
            except OSError:
                continue
        else:
            _FONTS[k] = ImageFont.load_default(size)
    return _FONTS[k]


def wrap(draw, text: str, f, width: int) -> list[str]:
    out = []
    for para in text.split("\n"):
        line = ""
        for w in para.split(" "):
            cand = w if not line else line + " " + w
            if draw.textlength(cand, font=f) <= width or not line:
                line = cand
            else:
                out.append(line)
                line = w
        out.append(line)
    return out


def tag(draw, xy, text: str, size: int = 13, fg=(255, 255, 255), bg=(0, 0, 0), bold: bool = False, pad: int = 3,
        max_w: int | None = None):
    f = font(size, bold)
    if max_w is not None:
        while text and draw.textlength(text, font=f) > max_w - 2 * pad:
            text = text[:-2] + "~" if len(text) > 3 else text[:-1]
    x, y = xy
    _, _, r, b = draw.textbbox((x + pad, y + pad), text, font=f)
    draw.rectangle([x, y, r + pad, b + pad], fill=bg)
    draw.text((x + pad, y + pad), text, font=f, fill=fg)
    return r + pad, b + pad


def grid_sheet(title: str, sub: list[str], headers: list[dict], rows: list[dict], footer: list[str], tile_w: int,
               gap: int = 4, width: int = SHEET_W):
    """A dark sheet: title + wrapped sub lines, one header block per column, rows of equal-height tiles (None = a
    labelled grey 'missing' tile), optional full-width band text above a row, footer lines. -> PIL image."""
    from PIL import Image, ImageDraw
    ncol = len(headers)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    ft, fs, fh, fh2, fb = font(24, True), font(15), font(16, True), font(13), font(14, True)
    sub_l = [ln for s in sub for ln in wrap(probe, s, fs, width - 2 * gap - 8)]
    foot_l = [ln for s in footer for ln in wrap(probe, s, fs, width - 2 * gap - 8)]
    hdr = []
    for h in headers:
        lines = [(ln, fh, h.get("fg", (255, 255, 255))) for ln in wrap(probe, h["title"], fh, tile_w - 6)]
        for s in h.get("lines", []):
            lines += [(ln, fh2, h.get("fg2", (210, 210, 210))) for ln in wrap(probe, s, fh2, tile_w - 6)]
        hdr.append(lines)
    hdr_h = max(sum(18 if f is fh else 16 for _, f, _ in ls) for ls in hdr) + 10
    top = 12 + 30 + 19 * len(sub_l) + 8
    H = top + hdr_h + sum(r["h"] + gap + (22 if r.get("band") else 0) for r in rows) + gap + 19 * len(foot_l) + 14
    W = width
    img = Image.new("RGB", (W, H), (22, 22, 24))
    d = ImageDraw.Draw(img)
    d.text((gap + 4, 10), title, font=ft, fill=(255, 255, 255))
    y = 44
    for ln in sub_l:
        d.text((gap + 4, y), ln, font=fs, fill=(220, 220, 220))
        y += 19
    y = top
    x0s = [gap + c * (tile_w + gap) for c in range(ncol)]
    for c, ls in enumerate(hdr):
        bgc = headers[c].get("bg")
        if bgc is not None:
            d.rectangle([x0s[c], y, x0s[c] + tile_w - 1, y + hdr_h - 4], fill=bgc)
        yy = y + 4
        for ln, f, colr in ls:
            d.text((x0s[c] + 3, yy), ln, font=f, fill=colr)
            yy += 18 if f is fh else 16
    y += hdr_h
    for r in rows:
        if r.get("band"):
            d.rectangle([gap, y, W - gap - 1, y + 19], fill=(48, 48, 56))
            d.text((gap + 4, y + 2), r["band"], font=fb, fill=(255, 255, 255))
            y += 22
        for c in range(ncol):
            tl = r["tiles"][c]
            x = x0s[c]
            if tl is None:
                d.rectangle([x, y, x + tile_w - 1, y + r["h"] - 1], fill=(92, 92, 92))
                d.line([x, y, x + tile_w - 1, y + r["h"] - 1], fill=(120, 120, 120), width=2)
                d.line([x, y + r["h"] - 1, x + tile_w - 1, y], fill=(120, 120, 120), width=2)
                msg = (r.get("missing") or [None] * ncol)[c] or "missing"
                yy = y + r["h"] // 2 - 20
                for ln in wrap(d, "MISSING: " + msg, font(13, True), tile_w - 10):
                    tag(d, (x + 5, yy), ln, 13, bg=(60, 20, 20), bold=True)
                    yy += 19
            else:
                img.paste(Image.fromarray(tl), (x, y))
            tg = (r.get("tags") or [None] * ncol)[c]
            if tg is not None:
                tag(d, (x + 2, y + 2), tg[1], 12, bg=LEVEL_COL[tg[0]], bold=True, max_w=tile_w - 4)
            nt = (r.get("notes") or [None] * ncol)[c]
            if nt:
                lines = wrap(d, nt, font(12), tile_w - 10)
                yy = y + r["h"] - 4 - 17 * len(lines)
                for ln in lines:
                    tag(d, (x + 2, yy), ln, 12, bg=(0, 0, 0))
                    yy += 17
        y += r["h"] + gap
    y += 4
    for ln in foot_l:
        d.text((gap + 4, y), ln, font=fs, fill=(225, 225, 160))
        y += 19
    return img


def save_jpg(img, p: Path) -> Path:
    assert img.size[0] <= SHEET_W, img.size
    img.save(p, "JPEG", quality=JPEG_Q)
    return p


def method_header(m: str, extra: str | None = None) -> dict:
    mt = METHODS[m]
    lines = [mt["built"], mt["what"]] + ([extra] if extra else [])
    return {"title": f"{mt['title']}" + (" (" + m + ")" if m not in ("avatar",) else ""), "lines": lines,
            "bg": (60, 40, 40) if m == "avatar" else ((35, 45, 70) if mt["kind"] == "splat" else (40, 58, 40))}


def real_header(extra: str = "phone frame, real background kept") -> dict:
    return {"title": "REAL photo", "lines": [extra], "bg": (70, 70, 70)}


# ------------------------------------------------------------------------------------------------ sheets
def occlusion_note(v: dict) -> str | None:
    if v["both_well"]:
        return None
    out = []
    for r in ROLES:
        st = v["mask_stats"][r]
        if st["cut"]:
            out.append(f"{r} cut by the frame edge")
        elif st["vis"] < 0.6 or st["area_frac"] < 0.002:
            out.append(f"{r} mostly hidden behind the partner")
    return "real occlusion: " + "; ".join(out) if out else None


def sheet_instant(ctx: Ctx, inst: str, out: Path) -> dict:
    x = ctx.inst[inst]
    views = [v for v in ctx.spec["real_views"] if v["instant"] == inst]
    ncol = 1 + len(ORDER)
    gap = 4
    tile_w = (SHEET_W - gap * (ncol + 1)) // ncol
    rows, missing = [], []
    for v in views:
        x0, y0, x1, y1 = v["crop"]
        th = int(round(tile_w * (y1 - y0) / (x1 - x0)))
        ph = ctx.photo(v["file"])
        tiles, tags, notes, miss = [fit(ph[y0:y1, x0:x1], tile_w, th)], [None], [], [None]
        occ = occlusion_note(v)
        notes.append(f"phone {v['cam']} frame {v['frame']} ({v['dt_ms']:+.1f} ms)" + (f"; {occ}" if occ else ""))
        for m in ORDER:
            p = ctx.path(m, "real", v["id"])
            ra = load_rgba(p)
            if ra is None or ra[0].shape[:2] != (v["H"], v["W"]):
                tiles.append(None)
                miss.append(f"{_rel(p)}" if ra is None else f"{p.name} has size {ra[0].shape[:2]}")
                missing.append(_rel(p))
                tags.append(None)
            else:
                tiles.append(fit(over(ra)[y0:y1, x0:x1], tile_w, th))
                tags.append(ctx.status(m, v["id"]))
                miss.append(None)
            notes.append(ctx.jev_note(v["id"], m))
        rows.append({"h": th, "tiles": tiles, "tags": tags, "notes": notes, "missing": miss})
    turn = " - the follow's fast head turn" if x["head_turn"] else ""
    title = f"Instant {inst}: t = {x['t']:.3f} s{turn}   (larissa_kadu_38; same crop in every column)"
    sub = ["Rows: phone 03 (480 x 848; held out by AGS, LHM++ 1-phone and the hybrids; the avatar's bake and LHM++ "
           "8-phone saw other phone-03 frames), phone 02 and phone 06 (1080 x 1920; AGS training phones; 06 = LHM++ "
           "1-phone's input phone).",
           "Renders on neutral grey; the REAL column keeps the photo's background. Pose: the reference MAMMA SMPL-X fit "
           "for every method except AGS (its own take's fit + its trained pose corrections). Pinhole renders (the phone "
           "lens k1 is ignored: <= 7 px at phone 02's corners). " + LEGEND + "." + (
               " Bottom of each render: Jev's P(looks like a photo of real people) for that render alone, and how often "
               "Jev picked it as the most photograph-like of the six (shuffled A-F, real photo as reference; "
               "jev.json)." if ctx.jev_photo else "")]
    footer = [f"avatar = {AVATAR_CAPTION}. hybrids = {HYBRID_CAPTION}.",
              "Sources: compare/renders/<method>/<view>.png (manifest.json per method), views.json real_views[*].crop."]
    headers = [real_header()] + [method_header(m) for m in ORDER]
    img = grid_sheet(title, sub, headers, rows, footer, tile_w, gap)
    p = save_jpg(img, out / f"sheet_{inst}.jpg")
    return {"file": _rel(p), "size": list(img.size), "missing": missing, "views": [v["id"] for v in views]}


BOX_TITLE = {"face_follow": "the follow's face (32 cm square)", "face_lead": "the lead's face (32 cm square)",
             "hair_follow": "the follow's hair (60 cm square from the crown down)"}


def sheet_closeup(ctx: Ctx, box: str, out: Path) -> dict:
    ncol = 1 + len(ORDER)
    gap = 4
    tile_w = (SHEET_W - gap * (ncol + 1)) // ncol
    rows, missing = [], []
    for inst in ctx.inst:
        c = ctx.cu[inst]
        cam = best_cam_for(ctx, inst, box)
        b = c["cams"][cam]["boxes"][box]
        vid = f"{inst}_c{cam}"
        v = ctx.real[vid]
        if not b["box"]:
            rows.append({"h": tile_w, "tiles": [None] * ncol, "missing": ["box outside the photo"] * ncol})
            continue
        x0, y0, x1, y1 = b["box"]
        th = int(round(tile_w * (y1 - y0) / (x1 - x0)))
        ph = ctx.photo(v["file"])
        facing = b["facing_cos"]
        side = "faces the phone" if facing > 0.3 else ("seen from the side" if facing > -0.3 else "faces away")
        tiles, tags, miss = [fit(ph[y0:y1, x0:x1], tile_w, th)], [None], [None]
        notes = [f"{inst} {x_t(ctx, inst)} phone {cam}: {x1 - x0} px in the photo; head {side}"]
        for m in ORDER:
            p = ctx.path(m, "closeup", (inst, cam, box))
            ra = load_rgba(p)
            if ra is None:
                tiles.append(None)
                tags.append(None)
                miss.append(_rel(p))
                missing.append(_rel(p))
            else:
                tiles.append(fit(over(ra), tile_w, th))
                tags.append(ctx.status(m, vid))
                miss.append(None)
            notes.append(None)
        rows.append({"h": th, "tiles": tiles, "tags": tags, "notes": notes, "missing": miss})
    title = f"Close-ups: {BOX_TITLE[box]}, each instant on its best high-res phone"
    sub = ["REAL = the photo's own pixels in that box (upscaled; the photo has only the 'px in the photo' shown). "
           "Renders = each method re-rendered through a zoomed camera on exactly that box (512 px; they can show more "
           "detail than the photo has). Same pose / time as the photo; renders on neutral grey.",
           ("Phone per row = the high-res phone (02 / 06) that sees THIS face bigger, more frontally and less hidden by "
            "the partner (faces) / sees most of her hair unoccluded (hair; views.json closeups best_cam_hair). ") +
           LEGEND + "."]
    footer = [f"avatar = {AVATAR_CAPTION} (the Unity build also draws a portrait face / hair cards; see "
              f"sheet_unity_reference.jpg). hybrids = {HYBRID_CAPTION}.",
              "Sources: renders/<method>/cu_<instant>_c<phone>_<box>.png (LHM++: closeup_zoom/<view>_<box>.png)."]
    headers = [real_header("photo pixels in the box")] + [method_header(m) for m in ORDER]
    img = grid_sheet(title, sub, headers, rows, footer, tile_w, gap)
    p = save_jpg(img, out / f"sheet_closeup_{box}.jpg")
    return {"file": _rel(p), "size": list(img.size), "missing": missing}


def best_cam_for(ctx: Ctx, inst: str, box: str) -> str:
    """Hair: the spec's best_cam_hair. A face: the spec's best_cam rule applied to that ONE face (side_px x
    max(facing, 0.05) x (own mask at the centre ? 1 : 0.3) x (1 - partner_mask_frac) x min(1, own_mask_frac / 0.2),
    where a face fully hidden behind the partner scores 0), so a face hidden on the instant's best_cam is shown from the
    other phone."""
    c = ctx.cu[inst]
    if box.startswith("hair"):
        return c["best_cam_hair"]
    sc = {}
    for cam, d in c["cams"].items():
        b = d["boxes"][box]
        if b["box"]:
            sc[cam] = b["side_px"] * max(b["facing_cos"], 0.05) * (1.0 if b["own_mask_at_centre"] else 0.3) * (
                1 - b["partner_mask_frac"]) * min(1.0, b["own_mask_frac"] / 0.2)
    return max(sc, key=sc.get) if sc else c["best_cam"]


def x_t(ctx: Ctx, inst: str) -> str:
    return f"{ctx.inst[inst]['t']:.2f} s"


def union_bbox(alphas: list[np.ndarray], pad: float, W: int, H: int, thr: float = 0.05):
    box = None
    for a in alphas:
        ys, xs = np.nonzero(a > thr)
        if len(xs) == 0:
            continue
        b = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        box = b if box is None else [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
    if box is None:
        return [0, 0, W, H]
    p = pad * max(box[2] - box[0], box[3] - box[1])
    return [int(max(0, box[0] - p)), int(max(0, box[1] - p)), int(min(W, box[2] + p)), int(min(H, box[3] + p))]


def sheet_turntable(ctx: Ctx, out: Path) -> dict:
    tt = ctx.spec["novel"]["turntable"]
    vr_az = ctx.spec["novel"]["vr_seat"]["azimuth_deg"]
    views = tt["views"]
    ncol = len(views)
    gap = 3
    tile_w = (SHEET_W - gap * (ncol + 1)) // ncol
    imgs, alphas = {}, []
    for m in ORDER:
        for v in views:
            ra = load_rgba(ctx.path(m, "tt", v["id"]))
            imgs[(m, v["id"])] = ra
            if ra is not None:
                alphas.append(ra[1])
    S = views[0]["W"]
    x0, y0, x1, y1 = union_bbox(alphas, 0.02, S, S)
    th = int(round(tile_w * (y1 - y0) / (x1 - x0)))
    headers = []
    for v in views:
        dd = {c: abs((p["azimuth_deg"] - v["azimuth_deg"] + 180) % 360 - 180) for c, p in ctx.spec["phones"].items()}
        near = min(dd, key=dd.get)
        dvr = abs((vr_az - v["azimuth_deg"] + 180) % 360 - 180)
        lines = [f"nearest phone {near} ({dd[near]:.0f} deg)"]
        h = {"title": f"{v['id']} +{v['azimuth_from_cam03_deg']:.0f} deg", "lines": lines, "bg": (45, 45, 45)}
        if dd[near] <= 12:
            h["lines"] = [f"= REAL PHONE {near} ({dd[near]:.0f} deg)"]
            h["bg"] = (120, 90, 10)
        if dvr <= 12:
            h["lines"].append(f"~ VR seat ({dvr:.0f} deg)")
            h["bg"] = (110, 40, 110) if dd[near] > 12 else h["bg"]
        headers.append(h)
    rows, missing = [], []
    for m in ORDER:
        tiles, miss = [], []
        for v in views:
            ra = imgs[(m, v["id"])]
            if ra is None:
                tiles.append(None)
                miss.append(_rel(ctx.path(m, "tt", v["id"])))
                missing.append(miss[-1])
            else:
                tiles.append(fit(over(ra)[y0:y1, x0:x1], tile_w, th))
                miss.append(None)
        mt = METHODS[m]
        rows.append({"h": th, "tiles": tiles, "missing": miss,
                     "band": f"{mt['title']} ({m}) - {mt['built']}; {mt['what']}" + (
                         f" - {AVATAR_CAPTION}" if m == "avatar" else "")})
    title = (f"Turntable at {tt['instant']} (t = {tt['t']:.3f} s): 12 novel cameras every 30 deg, 3.0 m away, 1.6 m "
             f"high; +0 deg = phone 03's direction")
    sub = ["Columns marked REAL PHONE are within 12 deg of a real phone (that side was filmed); the others are "
           "in-between or unfilmed directions. Phones around the couple: " + ", ".join(
               f"{c} {p['azimuth_deg']:.0f}" for c, p in sorted(ctx.spec["phones"].items(),
                                                              key=lambda q: q[1]["azimuth_deg"])) +
           f" deg; VR seat {vr_az:.0f} deg (50 deg from the nearest phone). Renders on neutral grey, the same crop "
           f"for every tile ({x1 - x0} x {y1 - y0} px of the 1080 x 1080 frame)."]
    footer = [f"avatar = {AVATAR_CAPTION}. hybrids = {HYBRID_CAPTION}. AGS is posed with its own take's fit mapped by "
              "the body Sim3 (12.5 mm).", "Sources: renders/<method>/tt00..tt11.png."]
    img = grid_sheet(title, sub, headers, rows, footer, tile_w, gap)
    p = save_jpg(img, out / "sheet_turntable.jpg")
    return {"file": _rel(p), "size": list(img.size), "missing": missing, "crop": [x0, y0, x1, y1]}


def vr_crop(ctx: Ctx, every: int = 3, frames: list[int] | None = None) -> list[int]:
    n = int(ctx.spec["clip"]["n"])
    alphas = []
    for m in ORDER:
        for i in (frames if frames is not None else range(0, n, every)):
            ra = load_rgba(ctx.path(m, "clip_vr_seat", i))
            if ra is not None:
                alphas.append(ra[1])
    S = ctx.spec["novel"]["vr_seat"]["W"]
    return union_bbox(alphas, 0.025, S, S)


def sheet_vr_seat(ctx: Ctx, out: Path, crop: list[int]) -> dict:
    vr = ctx.spec["novel"]["vr_seat"]
    ncol = len(ORDER)
    gap = 4
    tile_w = (SHEET_W - gap * (ncol + 1)) // ncol
    x0, y0, x1, y1 = crop
    th = int(round(tile_w * (y1 - y0) / (x1 - x0)))
    rows, missing = [], []
    for inst in ctx.inst:
        n = ctx.clip_index(inst)
        tiles, miss, notes = [], [], []
        for c, m in enumerate(ORDER):
            p = ctx.path(m, "clip_vr_seat", n)
            ra = load_rgba(p)
            if ra is None:
                tiles.append(None)
                miss.append(_rel(p))
                missing.append(_rel(p))
            else:
                tiles.append(fit(over(ra)[y0:y1, x0:x1], tile_w, th))
                miss.append(None)
            notes.append(f"{inst} t {ctx.inst[inst]['t']:.2f} s (clip frame {n})" if c == 0 else None)
        rows.append({"h": th, "tiles": tiles, "missing": miss, "notes": notes})
    title = (f"VR seat (the headset's view): novel camera at azimuth {vr['azimuth_deg']:.0f} deg, "
             f"{vr['gap']['nearest_phone_deg']:.0f} deg from the nearest phone, 2.5 m away, 1.6 m high, 55 deg FOV")
    sub = [f"No real photo exists from here (it is the middle of the widest gap between phones, "
           f"{vr['gap']['between'][0]}-{vr['gap']['between'][1]}). Rows = the six instants (clip frames of "
           f"renders/<method>/clip_vr_seat/); same crop for every tile ({x1 - x0} x {y1 - y0} px of 1080 x 1080); "
           "renders on neutral grey."]
    footer = [f"avatar = {AVATAR_CAPTION}. hybrids = {HYBRID_CAPTION}."]
    img = grid_sheet(title, sub, [method_header(m) for m in ORDER], rows, footer, tile_w, gap)
    p = save_jpg(img, out / "sheet_vr_seat.jpg")
    return {"file": _rel(p), "size": list(img.size), "missing": missing, "crop": crop}


def sheet_unity(ctx: Ctx, out: Path) -> dict | None:
    from PIL import Image, ImageDraw
    refs = [PACKAGE_ROOT / u["file"] for u in ctx.spec.get("unity_reference", [])]
    refs = [p for p in refs if p.exists()]
    if not refs:
        return None
    tiles = []
    for p in refs:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)[:, :, ::-1]
        H, W = im.shape[:2]
        c = im[int(0.13 * H):int(0.87 * H), int(0.32 * W):int(0.68 * W)]   # the avatar, HUD text left out
        tiles.append(c)
    th = 900
    tiles = [fit(t, int(round(t.shape[1] * th / t.shape[0])), th) for t in tiles]
    W = min(SHEET_W, sum(t.shape[1] for t in tiles) + 4 * (len(tiles) + 1))
    img = Image.new("RGB", (W, th + 120), (22, 22, 24))
    d = ImageDraw.Draw(img)
    d.text((8, 8), "Unity reference: real screenshots of the CURRENT avatars in the Quest build scene", font=font(22, True),
           fill=(255, 255, 255))
    lines = wrap(d, "04_LarissaKaduDemo, captured 2026-10-07 at scene 23.0 s (reference ~53.15 s, outside the compared "
                    "clip), before the 2026-10-08 face-shape edit; HUD overlays visible. What Unity adds to the offline "
                    "render: hair cards with tip bloom, procedural shoes, eye blinks and 35 % scene shading of the "
                    "albedo (HM_AvatarLit _LightInfluence 0.35; the offline render is unlit).",
                 font(15), W - 16)
    y = 38
    for ln in lines:
        d.text((8, y), ln, font=font(15), fill=(220, 220, 220))
        y += 19
    x = 4
    for t, p in zip(tiles, refs):
        img.paste(Image.fromarray(np.ascontiguousarray(t)), (x, 116))
        tag(d, (x + 2, 118), p.name, 13)
        x += t.shape[1] + 4
    p = save_jpg(img, out / "sheet_unity_reference.jpg")
    return {"file": _rel(p), "size": list(img.size), "sources": [_rel(r) for r in refs]}


def build_sheets(ctx: Ctx) -> dict:
    out = ctx.cmp
    res = {"instants": {}, "closeups": {}}
    for inst in ctx.inst:
        res["instants"][inst] = sheet_instant(ctx, inst, out)
        log(f"sheet {res['instants'][inst]['file']} {res['instants'][inst]['size']} missing "
            f"{len(res['instants'][inst]['missing'])}")
    for box in ("face_follow", "face_lead", "hair_follow"):
        res["closeups"][box] = sheet_closeup(ctx, box, out)
        log(f"sheet {res['closeups'][box]['file']} {res['closeups'][box]['size']}")
    res["turntable"] = sheet_turntable(ctx, out)
    log(f"sheet {res['turntable']['file']} {res['turntable']['size']}")
    crop = vr_crop(ctx, frames=[ctx.clip_index(i) for i in ctx.inst])   # the six instants only (the video: whole clip)
    res["vr_seat"] = sheet_vr_seat(ctx, out, crop)
    log(f"sheet {res['vr_seat']['file']} {res['vr_seat']['size']}")
    res["unity_reference"] = sheet_unity(ctx, out)
    return res


# ------------------------------------------------------------------------------------------------ metrics
def compute_metrics(ctx: Ctx) -> dict:
    """Photo agreement on the 18 real views per dancer, cand_lhmpp.compare_score's definitions (crop = the dancer's
    real-mask bbox + 10 %, inside the real mask, the render composited over black, *_cc after a least-squares colour
    fit, iou / coverage of the rendered role vs the real mask, *_both = real mask AND rendered dancer) with
    dancer_splats.evaluate._crop_scores. LPIPS is not installed locally: the LHM++ pod's LPIPS rows (same code, same
    crops) are merged for its two variants, after checking that the local PSNR reproduces the pod's."""
    from ..dancer_splats.evaluate import _crop_scores
    from .cand_lhmpp import _crop_box
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        device = "cpu"
    if device == "cuda":   # the SSIM convolutions only (~1 GB): hold work/gpu.lock like every local GPU job
        from ..gpu import exclusive
        with exclusive("appearance_compose metrics"):
            return _compute_metrics(ctx, _crop_scores, _crop_box, device)
    return _compute_metrics(ctx, _crop_scores, _crop_box, device)


def _compute_metrics(ctx: Ctx, _crop_scores, _crop_box, device: str) -> dict:
    rows = []
    for v in ctx.spec["real_views"]:
        ph = ctx.photo(v["file"]).astype(np.float32) / 255.0
        H, W = ph.shape[:2]
        masks = {r: read_mask(ctx.work, v["cam"], v["frame"], r, (H, W)) for r in ROLES}
        union = None
        for r in ROLES:
            if masks[r] is not None:
                union = masks[r] if union is None else (union | masks[r])
        for m in ORDER:
            ra = load_rgba(ctx.path(m, "real", v["id"]))
            if ra is None or ra[0].shape[:2] != (H, W):
                rows.append({"view": v["id"], "cam": v["cam"], "method": m, "missing": True})
                continue
            rgb, a = ra
            pred = rgb * a[..., None]
            own = ctx.role_masks(m, v["id"], (H, W))
            lvl, why = ctx.status(m, v["id"])
            ui = None
            if union is not None:
                al = a > 0.5
                ui = float((al & union).sum() / max((al | union).sum(), 1))
            for r in ROLES:
                mk = masks[r]
                if mk is None or mk.sum() < 50:
                    continue
                x0, y0, x1, y1 = _crop_box(mk)
                s = _crop_scores(pred[y0:y1, x0:x1], ph[y0:y1, x0:x1], mk[y0:y1, x0:x1], None, device)
                row = {"view": v["id"], "cam": v["cam"], "instant": v["instant"], "role": r, "method": m,
                       "crop": [int(x0), int(y0), int(x1), int(y1)], "status": lvl, "status_text": why,
                       "both_well": v["both_well"], "union_iou": ui,
                       **{k: (None if s.get(k) is None else float(s[k])) for k in s}}
                if own is not None:
                    o = own[r]
                    row["iou"] = float((o & mk).sum() / max((o | mk).sum(), 1))
                    row["coverage"] = float((o & mk).sum() / mk.sum())
                    s2 = _crop_scores(pred[y0:y1, x0:x1], ph[y0:y1, x0:x1], (o & mk)[y0:y1, x0:x1], None, device)
                    row.update({f"{k}_both": float(s2[k]) for k in ("psnr", "ssim", "psnr_cc") if s2.get(k) is not None})
                rows.append(row)
        log(f"metrics {v['id']} done")
    # LPIPS of the LHM++ variants from the pod (cuda, lpips alex) - same function, same crops
    pod = ctx.cmp / "lhmpp" / "pod_out" / "metrics.json"
    lp_check = {}
    if pod.exists():
        pm = json.loads(pod.read_text())
        idx = {(r["view"], r["role"], r["variant"]): r for r in pm.get("rows", [])}
        d = []
        for row in rows:
            q = idx.get((row.get("view"), row.get("role"), row.get("method")))
            if q is None or row.get("missing"):
                continue
            d.append(abs(q["psnr"] - row["psnr"]))
            if q.get("crop") != row["crop"]:
                continue
            for k in ("lpips", "lpips_cc", "lpips_both"):
                if q.get(k) is not None:
                    row[k] = float(q[k])
            row["lpips_source"] = "LHM++ pod 20261019-182851-batch (compare_score, cuda)"
        lp_check = {"rows_matched": len(d), "psnr_abs_diff_max": round(max(d), 4) if d else None,
                    "psnr_abs_diff_median": round(float(np.median(d)), 5) if d else None}
    sets = {
        "cam03_unseen_frames": lambda r: r["view"] in UNSEEN_CAM03,
        "cam03": lambda r: r["cam"] == "03",
        "all_18_views": lambda r: True,
        "cam02": lambda r: r["cam"] == "02",
        "cam06": lambda r: r["cam"] == "06",
        "both_well_views": lambda r: r["both_well"],
        "lead_all": lambda r: r["role"] == "lead",
        "follow_all": lambda r: r["role"] == "follow",
        "heldout_for_this_method": lambda r: r["status"] == 0,
    }
    keys = ("psnr", "ssim", "psnr_cc", "ssim_cc", "lpips", "lpips_cc", "iou", "coverage", "union_iou", "psnr_both",
            "ssim_both", "psnr_cc_both", "lpips_both")
    summ = {}
    ok = [r for r in rows if not r.get("missing")]
    for m in ORDER:
        summ[m] = {}
        for sname, f in sets.items():
            rr = [r for r in ok if r["method"] == m and f(r)]
            summ[m][sname] = {"n": len(rr), "views": sorted({r["view"] for r in rr}).__len__(),
                              **{k: (round(float(np.median([r[k] for r in rr if r.get(k) is not None])), 4)
                                     if any(r.get(k) is not None for r in rr) else None) for k in keys}}
    # who trained on what (the caveat of every number)
    trained = {m: {v["id"]: ctx.status(m, v["id"]) for v in ctx.spec["real_views"]} for m in ORDER}
    unseen_check = {vid: {m: trained[m][vid] for m in ORDER} for vid in UNSEEN_CAM03}
    out = {"how": compute_metrics.__doc__, "device": device, "lpips_merge_check": lp_check, "summary_median": summ,
           "sets": {"cam03_unseen_frames": "phone-03 frames at i0, i2, i4, i5: no method trained on these frames (the "
                                           "avatar bake and LHM++ 8-phone saw OTHER phone-03 frames; AGS, LHM++ "
                                           "1-phone and the hybrids never saw phone 03)",
                    "cam03": "all six phone-03 views (i1_c03 / i3_c03 frames are in the avatar's bake)",
                    "heldout_for_this_method": "rows whose camera is held out for that method (differs per method)"},
           "unseen_check": unseen_check, "trained_status": trained, "missing": [r for r in rows if r.get("missing")],
           "rows": rows}
    (ctx.cmp / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    (ctx.cmp / "metrics.md").write_text(metrics_markdown(out), encoding="utf-8")
    log(f"metrics.json: {len(ok)} rows, LPIPS merge check {lp_check}")
    return out


def _f(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def metrics_markdown(mt: dict) -> str:
    lines = ["| view set | method | dancer-views | PSNR | PSNR cc | SSIM | SSIM cc | LPIPS | LPIPS cc | IoU |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for sname in ("cam03_unseen_frames", "cam03", "all_18_views", "heldout_for_this_method"):
        for m in ORDER:
            s = mt["summary_median"][m][sname]
            lines.append(f"| {sname} | {m} | {s['n']} | {_f(s['psnr'])} | {_f(s['psnr_cc'])} | {_f(s['ssim'], 3)} | "
                         f"{_f(s['ssim_cc'], 3)} | {_f(s['lpips'], 3)} | {_f(s['lpips_cc'], 3)} | {_f(s['iou'], 3)} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ videos
def ffmpeg_exe() -> str:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class Mp4:
    """H.264 (libx264, yuv420p, CRF 18, faststart) from RGB frames piped to ffmpeg."""

    def __init__(self, path: Path, W: int, H: int, fps: float = 30.0):
        assert W % 2 == 0 and H % 2 == 0 and W <= 1920 and H <= 1080, (W, H)
        self.path, self.W, self.H, self.n = path, W, H, 0
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert frame.shape == (self.H, self.W, 3), frame.shape
        self.proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        self.n += 1

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {self.path}")


def _panel_header(d, x, w, title, sub, sub2=None, bg=(40, 40, 40)):
    d.rectangle([x, 0, x + w - 1, 49], fill=bg)
    tag(d, (x + 2, 1), title, 15, bg=bg, bold=True, max_w=w - 4)
    tag(d, (x + 2, 19), sub, 12, bg=bg, max_w=w - 4)
    if sub2:
        tag(d, (x + 2, 33), sub2, 12, bg=bg, fg=(255, 220, 120), max_w=w - 4)


def build_video(ctx: Ctx, kind: str, panels: list[str], out: Path, crop: list[int]) -> dict:
    """kind 'vr_seat' | 'cam03'; panels = 'real' and / or methods. Real time once, then 0.5x (each frame twice)."""
    from PIL import Image, ImageDraw
    clip = ctx.spec["clip"]
    n = int(clip["n"])
    group = "clip_vr_seat" if kind == "vr_seat" else "clip_cam03"
    x0, y0, x1, y1 = crop
    cw, ch = x1 - x0, y1 - y0
    top, bot = 50, 24
    npn = len(panels)
    s = min((1920 - (npn - 1) * 4) / npn / cw, (1080 - top - bot) / ch)
    pw, phh = int(cw * s) // 2 * 2, int(ch * s) // 2 * 2
    W = npn * pw + (npn - 1) * 4
    W += W % 2
    H = top + phh + bot
    H += H % 2
    seen = {}
    for m in panels:
        if m == "real":
            seen[m] = None
        elif kind == "cam03":
            st = sorted({ctx.status(m, vid) for vid in ctx.real if vid.endswith("_c03")})
            seen[m] = st[-1]
    frames = []
    temporal = {m: [] for m in panels if m != "real"}
    flicker = {m: [] for m in panels if m != "real"}
    prev, prev2 = {}, {}
    missing = []
    for i in range(n):
        tiles = []
        for m in panels:
            if m == "real":
                fr = clip["cam03"]["frames"][i]
                tiles.append(fit(ctx.photo(fr["file"])[y0:y1, x0:x1], pw, phh))
                continue
            p = ctx.path(m, group, i)
            ra = load_rgba(p)
            if ra is None:
                missing.append(_rel(p))
                t = np.full((phh, pw, 3), 92, np.uint8)
                cv2.putText(t, "MISSING", (10, phh // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 80), 2)
                tiles.append(t)
                continue
            rgb, a = ra[0][y0:y1, x0:x1], ra[1][y0:y1, x0:x1]
            tiles.append(fit(over((rgb, a)), pw, phh))
            if m in prev:   # temporal change inside the dancers (both frames' alpha > 0.5), 0-255
                pr, pa = prev[m]
                sel = (a > 0.5) & (pa > 0.5)
                if sel.sum() > 100:
                    temporal[m].append(float(np.abs(rgb - pr)[sel].mean() * 255))
                if m in prev2:   # second difference |I_t - (I_t-1 + I_t+1) / 2| at t-1: popping / flicker proxy
                    p2, pa2 = prev2[m]
                    sel2 = sel & (pa2 > 0.5)
                    if sel2.sum() > 100:
                        flicker[m].append(float(np.abs(pr - 0.5 * (p2 + rgb))[sel2].mean() * 255))
                prev2[m] = prev[m]
            prev[m] = (rgb, a)
        frames.append(tiles)
    mp4 = Mp4(out, W, H)
    t_list = clip["t"]
    for speed, rep in (("real time", 1), ("0.5x", 2)):
        for i, tiles in enumerate(frames):
            img = Image.new("RGB", (W, H), (22, 22, 24))
            d = ImageDraw.Draw(img)
            for c, (m, t) in enumerate(zip(panels, tiles)):
                x = c * (pw + 4)
                img.paste(Image.fromarray(t), (x, top))
                if m == "real":
                    _panel_header(d, x, pw, "REAL phone 03", "the photo (held-out phone)", None, (70, 70, 70))
                else:
                    mt = METHODS[m]
                    hb = method_header(m)["bg"]
                    sub2 = None
                    if kind == "cam03" and seen[m] is not None:
                        sub2 = {0: "phone 03 held out", 1: "phone 03 seen (other frames)",
                                2: "phone 03: some of these frames trained"}[seen[m][0]]
                    ttl = SHORT[m] if pw < 300 else f"{mt['title']}" + ("" if m == "avatar" else f" ({m})")
                    _panel_header(d, x, pw, ttl, mt["built"], sub2, hb)
            fr = clip["cam03"]["frames"][i] if kind == "cam03" else None
            txt = (f"t {t_list[i]:.3f} s  clip frame {i}/{n - 1}  {speed}" +
                   (f"  phone 03 frame {fr['frame']}" if fr else "  VR seat (novel camera, 50 deg from any phone)"))
            d.text((6, H - bot + 4), txt, font=font(14), fill=(255, 255, 255))
            cap = "avatar: " + AVATAR_CAPTION + " | renders on grey"
            d.text((W - 6 - d.textlength(cap, font=font(13)), H - bot + 5), cap, font=font(13), fill=(225, 225, 160))
            arr = np.asarray(img)
            for _ in range(rep):
                mp4.write(arr)
    mp4.close()
    tstat = {m: {"mean_abs_change_0_255": round(float(np.mean(v)), 2), "frames": len(v),
                 "mean_abs_second_difference_0_255": round(float(np.mean(flicker[m])), 2) if flicker[m] else None,
                 "p95_second_difference_0_255": round(float(np.percentile(flicker[m], 95)), 2) if flicker[m] else None}
             for m, v in temporal.items() if v}
    log(f"video {out.name}: {W}x{H}, {mp4.n} frames, panels {panels}")
    return {"file": _rel(out), "size": [W, H], "frames": mp4.n, "fps": 30, "panels": panels, "crop": crop,
            "panel_px": [pw, phh], "scale": round(s, 3), "missing": sorted(set(missing)),
            "temporal_change_inside_dancers": tstat,
            "layout": f"{n} frames at 30 fps (the 30 Hz grid = real time), then the same {n} frames at 0.5x"}


def cam03_crop(ctx: Ctx, panels: list[str], pad: float = 0.06) -> list[int]:
    c3 = ctx.spec["clip"]["cam03"]
    W, H = c3["W"], c3["H"]
    al = []
    for i, fr in enumerate(c3["frames"]):
        for r in ROLES:
            mk = read_mask(ctx.work, "03", fr["frame"], r, (H, W))
            if mk is not None:
                al.append(mk.astype(np.float32))
        if i % 4 == 0:
            for m in panels:
                if m != "real":
                    ra = load_rgba(ctx.path(m, "clip_cam03", i))
                    if ra is not None:
                        al.append(ra[1])
    return union_bbox(al, pad, W, H)


def pick_best(ctx: Ctx, cands: tuple[str, ...]) -> tuple[str, str]:
    """The phone-03 video's splat / hybrid: Jev's main realism choice (win rate, then mean probability, then mean
    P(photo)) when a full jev.json exists, else the held-out phone-03 median PSNR after colour correction."""
    jp = ctx.cmp / "jev.json"
    if jp.exists():
        jsum = json.loads(jp.read_text()).get("summary", {})
        js = jsum.get("choice", {}).get("per_method", {})
        ph = jsum.get("photo", {}).get("all", {})
        sc = {m: (js[m]["rate"], js[m]["mean_p"], ph.get(m, {}).get("mean_p", 0.0)) for m in cands if m in js}
        if sc and jsum.get("requests", 0) >= 100:
            b = max(sc, key=sc.get)
            return b, (f"Jev among {list(cands)}: (choice win rate, mean choice p, mean P(photo)) = " +
                       ", ".join(f"{m} {v[0]:.2f}/{v[1]:.3f}/{v[2]:.2f}" for m, v in sc.items()))
    mp = ctx.cmp / "metrics.json"
    if mp.exists():
        sm = json.loads(mp.read_text())["summary_median"]
        sc = {m: sm[m]["cam03"]["psnr_cc"] for m in cands if sm.get(m, {}).get("cam03", {}).get("psnr_cc") is not None}
        if sc:
            b = max(sc, key=sc.get)
            return b, "best held-out phone-03 median PSNR cc: " + ", ".join(f"{m} {v:.2f}" for m, v in sc.items())
    return cands[0], "default (no metrics yet)"


def build_videos(ctx: Ctx, best_splat: str | None = None, best_hybrid: str | None = None) -> dict:
    res = {}
    why_s = why_h = "given on the command line"
    if best_splat is None:
        best_splat, why_s = pick_best(ctx, SPLATS)
    if best_hybrid is None:
        best_hybrid, why_h = pick_best(ctx, HYBRIDS)
    vr_panels = list(ORDER)
    res["vr_seat"] = build_video(ctx, "vr_seat", vr_panels, ctx.cmp / "clip_vr_seat.mp4", vr_crop(ctx))
    p3 = ["real", "avatar", best_splat, best_hybrid]
    res["phone03"] = build_video(ctx, "cam03", p3, ctx.cmp / "clip_phone03.mp4", cam03_crop(ctx, p3))
    res["phone03"]["best_splat"] = {"method": best_splat, "why": why_s}
    res["phone03"]["best_hybrid"] = {"method": best_hybrid, "why": why_h}
    pa = ["real"] + list(ORDER)   # every method, smaller panels (extra)
    res["phone03_all"] = build_video(ctx, "cam03", pa, ctx.cmp / "clip_phone03_all.mp4", cam03_crop(ctx, pa))
    return res


# ------------------------------------------------------------------------------------------------ check
def check_outputs(ctx: Ctx, comp: dict) -> dict:
    """Every sheet decodes and is <= 1920 wide; every mp4 decodes with imageio (frame count, size, fps)."""
    import imageio.v2 as iio
    res = {"sheets": {}, "videos": {}}
    for p in sorted(ctx.cmp.glob("sheet_*.jpg")):
        im = cv2.imread(str(p))
        res["sheets"][p.name] = {"bytes": p.stat().st_size, "decoded": im is not None,
                                 "size": None if im is None else [im.shape[1], im.shape[0]],
                                 "ok": im is not None and im.shape[1] <= SHEET_W}
    for name in ("clip_vr_seat.mp4", "clip_phone03.mp4", "clip_phone03_all.mp4"):
        p = ctx.cmp / name
        if not p.exists():
            res["videos"][name] = {"exists": False, "ok": False}
            continue
        r = iio.get_reader(str(p), "ffmpeg")
        meta = r.get_meta_data()
        nf, first, last = 0, None, None
        for fr in r:
            if first is None:
                first = fr.shape
            last = fr
            nf += 1
        r.close()
        res["videos"][name] = {"bytes": p.stat().st_size, "frames_decoded": nf, "fps": meta.get("fps"),
                               "size": meta.get("size"), "codec": meta.get("codec"), "first_shape": list(first),
                               "last_mean": round(float(last.mean()), 1),
                               "ok": nf > 0 and meta.get("size", (9999, 9999))[0] <= 1920
                               and meta.get("size", (9999, 9999))[1] <= 1080}
    log("check: " + json.dumps({k: {n: v["ok"] for n, v in d.items()} for k, d in res.items()}))
    return res


# ------------------------------------------------------------------------------------------------ driver
def run_compose(take: str = TAKE, steps=("sheets", "metrics", "videos", "check"), allow_upload: bool = False,
                best_splat: str | None = None, best_hybrid: str | None = None, shuffles: int = 3,
                workers: int = 4) -> dict:
    ctx = Ctx(take)
    mp = ctx.cmp / "compose.json"
    comp = json.loads(mp.read_text()) if mp.exists() else {}
    comp.update({"schema": "dancecap.mono4d.appearance_compare.compose/1", "take": take,
                 "code": "dancecap/dancecap/mono4d/appearance_compose.py (+ appearance_jev.py)",
                 "methods": {m: {**METHODS[m], "manifest": _rel(ctx.root / m / "manifest.json"),
                                 "present": ctx.man[m] is not None} for m in ORDER}})
    t0 = time.time()
    if "sheets" in steps:
        comp["sheets"] = build_sheets(ctx)
    if "metrics" in steps:
        mt = compute_metrics(ctx)
        comp["metrics"] = {"file": _rel(ctx.cmp / "metrics.json"), "table": _rel(ctx.cmp / "metrics.md"),
                           "lpips_merge_check": mt["lpips_merge_check"]}
    if "jev" in steps:
        from .appearance_jev import run_jev
        comp["jev"] = run_jev(ctx, allow_upload=allow_upload, shuffles=shuffles, workers=workers)
    jp = ctx.cmp / "jev.json"
    if "jev" not in steps and jp.exists():
        js = json.loads(jp.read_text())
        sm = js.get("summary", {})
        comp["jev"] = {"file": _rel(jp), "created": js.get("created"), "requests": sm.get("requests"),
                       "errors": sm.get("errors"), "input_tokens": sm.get("input_tokens"), "cost_usd": sm.get("cost_usd")}
    if "videos" in steps:
        comp["videos"] = build_videos(ctx, best_splat, best_hybrid)
    if "check" in steps:
        comp["check"] = check_outputs(ctx, comp)
    comp["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    comp["seconds_last_run"] = round(time.time() - t0, 1)
    mp.write_text(json.dumps(comp, indent=1, default=float))
    log(f"wrote {mp} ({time.time() - t0:.0f} s)")
    return comp

