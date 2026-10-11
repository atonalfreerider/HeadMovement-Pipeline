"""The directed class-recap film: direction.json + captions timeline + chapters + review table, built from the teachers' transcript.

    python -m dancecap.film_recap build [--layers auto|assume|none|all] [--capture 06_LarissaKaduRecap] [--refresh-speech auto|yes|no] [--no-mix]
    python -m dancecap.film_recap check                # validate work/<take>/film/direction.json
    python -m dancecap.film_recap emoji                # the laugh emoji image(s) for the caption words (system colour-emoji font)
    python -m dancecap.film_recap chapters             # the [[recap.chapters]] block for takes/library/films.toml
    python -m dancecap.film_recap mix [class|film]    # the class audio mix (heavy, once) / the film soundtrack from the direction

`build` is the one command to re-run after anything changed: the owner's opening answer (speech/owner_opening.json: `build --refresh-speech auto` runs
`python -m dancecap.speech final` first when the answer is newer than the transcript), the recap capture and its analysis files, the teachers' transcript.
It writes, under work/<take>/film/ (git-ignored: it holds the teachers' words and voices):

    direction.json                     the film (segments with camera, layers, call-outs, chapter cards; film seconds; schema = the demo film's, additive)
    narration/timeline.json            the captions (word-highlight chunks, vertical + horizontal, speaker colours, the emoji as a word)
    chapters.json                      the chapter marks (title <= 5 words, film seconds)
    direction_report.json              what was used, what was dropped (overlays whose data is absent), the slow-motion decisions
    direction_review.md                what the teachers say / what the film shows, second by second
    audio/larissa_kadu_recap_mix.wav   the film soundtrack (the class mix on the film clock; needs audio/class_mix.wav: `mix class`)
    audio/make_film_mix.py             the recorder's soundtrack hook (rebuilds the mix from the direction)

Rules, inputs, the sound policy of slow motion and how to re-run: docs/RECAP_FILM.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import film_recap_arrows as arrows
from . import film_recap_captions as caps
from . import film_recap_director as director
from . import film_recap_plan as plan
from . import recap_pov
from .take import PACKAGE_ROOT, WORKSPACE

TAKE = "larissa_kadu_recap"
CAPTURE = "06_LarissaKaduRecap"
NAME = "larissa_kadu_recap"
VIEWER = WORKSPACE / "head-movement"
FPS = 30
DEFAULT_TITLE = "Class recap"
DEFAULT_SUBTITLE = "Counterbalance, taught by the two teachers"


@dataclass
class Paths:
    work: Path = PACKAGE_ROOT / "work" / TAKE

    @property
    def speech(self) -> Path:
        return self.work / "speech"

    @property
    def film(self) -> Path:
        return self.work / "film"

    @property
    def audio(self) -> Path:
        return self.film / "audio"

    @property
    def people(self) -> Path:
        return self.work / "people"


class RecapError(RuntimeError):
    """a missing input or an invalid plan; the message says what to do"""


# ------------------------------------------------------------------------------------------------ inputs


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RecapError(f"missing {p}: run `python -m dancecap.speech final` (docs/SPEECH.md)") from None


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else "absent"


def load_speech(paths: Paths) -> dict:
    tr = _read(paths.speech / "transcript_final.json")
    cap = _read(paths.speech / "captions.json")
    chap = _read(paths.speech / "chapters.json")
    ev = _read(paths.speech / "teaching_events.json")
    duration = float(chap.get("master_duration_s") or max(c["end"] for c in chap["chapters"]))
    utts = []
    for u in tr["utterances"]:
        if u["end"] <= 0:
            continue
        words = [w for w in u["words"] if w.get("kind") != "event"]
        utts.append({"id": u["id"], "start": float(u["start"]), "end": float(u["end"]), "speaker": u.get("speaker"), "text": u["text"],
                     "words": words, "conf_mean": u.get("conf_mean"), "flags": u.get("flags", [])})
    roles = {k: v for k, v in (ev.get("roles") or {}).items() if k != "note"}
    return {"transcript": tr, "captions": cap, "chapters": chap["chapters"], "events": ev["events"], "roles": roles, "duration": duration,
            "utterances": utts, "jokes": _read(paths.speech / "jokes.json") if (paths.speech / "jokes.json").exists() else {}}


def owner_opening_status(paths: Paths) -> dict:
    """whether the owner's opening answer exists and whether the transcript already includes it"""
    oo = paths.speech / "owner_opening.json"
    tj = paths.speech / "transcript_final.json"
    if not oo.exists():
        return {"answer": "not yet", "applied": False, "note": "the first ~8 s are the owner's words: the transcript carries the hint-based reading, flagged for the owner"}
    stale = tj.exists() and oo.stat().st_mtime > tj.stat().st_mtime
    return {"answer": "present", "applied": not stale, "note": "speech/owner_opening.json is newer than transcript_final.json: run `python -m dancecap.speech final`" if stale else "applied"}


def refresh_speech(paths: Paths, mode: str) -> str | None:
    st = owner_opening_status(paths)
    if mode == "no" or (mode == "auto" and (st["answer"] != "present" or st["applied"])):
        return None
    r = subprocess.run([sys.executable, "-m", "dancecap.speech", "final", "--no-clips"], cwd=PACKAGE_ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RecapError(f"`python -m dancecap.speech final` failed: {r.stderr[-500:]}")
    return "ran `python -m dancecap.speech final --no-clips`"


def load_activity(paths: Paths, capture_dir: Path | None) -> plan.Activity:
    t3 = paths.people / "track3d.npz"
    if t3.exists():
        try:
            return plan.Activity.from_track3d(t3, paths.people / "roles.npz")
        except Exception as e:                       # pragma: no cover - corrupt file
            print(f"warning: {t3.name} unreadable ({e})")
    if capture_dir is not None and (capture_dir / "floorcraft.json").exists():
        return plan.Activity.from_floorcraft(json.loads((capture_dir / "floorcraft.json").read_text(encoding="utf-8")))
    return plan.Activity()


def film_title(presets: Path | None = None) -> tuple[str, str]:
    """(title, subtitle) of the recap card in takes/library/films.toml, else the defaults"""
    try:
        import tomllib
        p = presets or (PACKAGE_ROOT / "takes" / "library" / "films.toml")
        d = tomllib.loads(p.read_text(encoding="utf-8")).get("recap", {})
        return str(d.get("title") or DEFAULT_TITLE), str(d.get("subtitle") or DEFAULT_SUBTITLE)
    except Exception:
        return DEFAULT_TITLE, DEFAULT_SUBTITLE


def chapter_title(raw: str) -> str:
    """plain English, at most 5 words, no trailing punctuation"""
    t = re.sub(r"\s+", " ", str(raw)).strip().strip(".:;,")
    w = t.split(" ")
    return " ".join(w[:5])


# ------------------------------------------------------------------------------------------------ the film's static blocks

PRESETS = {
    "vertical": {"menu": "Head Movement/Record Vertical", "width": 1080, "height": 1920, "fps": FPS, "safe": {"top_frac": 0.14, "bottom_frac": 0.2},
                 "framing": "both teachers always in frame (fit_span), key action between 14 % and 66 % of the height (captions own the lower section)"},
    "horizontal": {"menu": "Head Movement/Record Horizontal", "width": 1920, "height": 1080, "fps": FPS, "safe": {"top_frac": 0.05, "bottom_frac": 0.1},
                   "framing": "both teachers centred; key action above 85 % of the height"},
    "output": "head-movement/Recordings/<name>_<preset>_<date>.mp4 (git-ignored), H.264 + AAC, Unity Recorder Movie Recorder, frame-locked",
}
def presets_for(cfg: "plan.Cfg") -> dict:
    """PRESETS plus the opt-in 16:9 switches (the viewer's FilmAspect reads `phone_side_fill`; the 9:16 render never does)"""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PRESETS.items()}
    if cfg.horizontal_overrides and cfg.phone_side_fill > 0:
        out["horizontal"]["phone_side_fill"] = round(float(cfg.phone_side_fill), 3)
        out["horizontal"]["safe"] = {**out["horizontal"]["safe"], "side_frac": 0.05}      # title-safe 5 % at the sides too (labels, chapter card, legend)
        out["horizontal"]["portrait_zoom"] = round(float(cfg.h_portrait_zoom), 3)   # portrait clips: the 9:16 plan's zoom x this (the panel has the width to spare)
        out["horizontal"]["legend_scale"] = 1.5                                         # the physics legend's 11 px text is small on a 1080-high frame
        out["horizontal"]["framing"] += ("; a portrait phone clip is a centred panel with its own blurred, dimmed copy filling the sides "
                                         "(phone_side_fill = brightness); camera blocks carry `horizontal` overrides for the wide frame")
    return out


STYLE = {"annotation_colours": {"arrow": "#FFFFFF", "label_text": "#FFFFFF", "label_pill": "#000000B0", "step": "#33D6FF", "wedge": "#33D6FF33",
                                "lead": "#FF3B30", "follow": "#FFFFFF", "couple_com": "counterbalance yellow (existing)"},
         "rules": ["call-outs use the teacher's own word; one call-out at a time; a word is labelled once per 40 s",
                   "every force number carries 'est.' / 'estimated from motion' (none are shown in this film)",
                   "no beat counter: the class has no soundtrack, the rhythm of the film follows the speech",
                   "labels inside the title-safe area and above the caption block"]}
CAPTIONS = {
    "source": "speech/captions.json (the final transcript's words and times); chunks re-built per output shape by dancecap.film_recap_captions",
    "current_word": {"highlight": "#FFD60A", "text": "#111111", "box_padding_px": [10, 3], "corner_px": 8, "transition_ms": 60},
    "other_words": {"text": "#FFFFFF", "outline_px": 3, "outline": "#000000C0", "shadow": True},
    "speaker_colors": "each chunk carries `color` (its speaker's key colour): the other words of the chunk use it; a small name tag above the block",
    "chunking": "one speaker's phrase; breaks at speaker change, pause, sentence end, comma when 60 % full, and when the next word would not fit; a chunk shows from 0.15 s before its first word to 0.4 s after its last",
    "vertical_9x16": {"position": "lower section, centred, above the bottom title-safe band (bottom 20 %)", "block_centre_y_frac": 0.735,
                      "block_bottom_y_frac": 0.79, "max_lines": 2, "max_chars_per_line": 24, "font_px_at_1080w": 58, "width_frac": 0.88,
                      "move_caption": "moved to the top, under the top safe band"},
    "horizontal_16x9": {"position": "bottom centre, one long line", "baseline_y_frac": 0.925, "max_lines": 1, "max_chars_per_line": 62,
                        "font_px_at_1080h": 44, "width_frac": 0.84},
}


# ------------------------------------------------------------------------------------------------ assemble the segments


def _beat_camera(beat: plan.Beat, ci: dict, layers: plan.Layers, cfg: plan.Cfg, first: bool, film_t: float) -> tuple[dict, str]:
    """(camera block with its opt-in 16:9 overrides, family actually used)"""
    cam, fam = _beat_camera_raw(beat, ci, layers, cfg, first, film_t)
    return plan.with_horizontal(cam, cfg), fam


def _beat_camera_raw(beat: plan.Beat, ci: dict, layers: plan.Layers, cfg: plan.Cfg, first: bool, film_t: float) -> tuple[dict, str]:
    """(camera block, family actually used)"""
    if beat.camera is not None:                       # director mode: the shot brings its own camera block
        return dict(beat.camera), beat.family
    apart = bool(beat.activity.get("apart")) if beat.activity.get("known") else False
    if first and layers.phones and beat.dur >= 6.0:
        hold = max(1.5, beat.dur - 3.4)
        return ({"mode": "fly_to_camera", "camera": "01", "fly_s": 1.0, "fade_in_s": 0.6, "hold_s": round(hold, 2), "fade_out_s": 0.8,
                 "pull_back_s": 1.0, "composite": "3d_over_video", "skeletons_on_top": True, "scene_geometry": "all"}, "phone")
    fw = 0.35 if beat.family == "talk" and beat.focus in ("lead", "follow") else 0.0
    return plan.camera_for(beat.family, variant=beat.variant, focus=beat.focus, focus_weight=fw, apart=apart, cfg=cfg), beat.family


def _dance_to_film(seg: dict, t: float) -> float:
    """film offset into a segment of dance time t"""
    d0, d1 = seg["dance"]
    t = min(max(t, d0), d1)
    sp = seg["speed"]
    return (t - d0) / (sp if not isinstance(sp, list) else sum(sp) / 2)


def assemble(beats: list[plan.Beat], inserts: list[dict], layers: plan.Layers, events: list[dict], chapters: list[dict], act: plan.Activity,
             cfg: plan.Cfg, title: tuple[str, str], dropped: list[dict], arrow_ctx: dict | None = None, arrow_skips: list[dict] | None = None) -> list[dict]:
    """beats + inserts -> segments with film times, layers, call-outs and chapter cards. `arrow_ctx` ({layers, t_axes, transfers}) switches the D4 arrows
    on (film_recap_arrows: one per attention call, the physics view's own); the calls without one are appended to `arrow_skips`."""
    segs: list[dict] = []
    f = 0.0
    state: dict = {}
    astate: dict = {}
    titles = [chapter_title(c["title"]) for c in chapters]
    for bi, b in enumerate(beats):
        cam, fam = _beat_camera(b, {}, layers, cfg, bi == 0, f)
        lay = plan.treatment_layers("talk" if fam == "phone" else fam, layers, dropped, b.id, b.variant, cfg.avatar_opacity, cfg.avatar_on_video)
        if fam == "phone":
            lay = dict(lay, source_video="01")
        view = plan.VIEW_STATE.get("talk" if fam == "phone" else fam, "orbit")
        view = "camera_tour" if fam == "phone" else view
        annos = plan.make_annotations(b if fam != "pov" else replace(b, family="talk"), events, layers, cfg, dropped, state) if fam != "phone" else []
        if cfg.arrows_every_call and arrow_ctx is not None and fam != "phone":
            ar, skp = arrows.make_arrows(b, events, cfg, arrow_ctx, act, astate)
            if b.family == "weight" and layers.any("physics"):
                ar += arrows.physics_extras(b, arrow_ctx.get("transfers", {}), cfg, astate, ar, arrow_ctx)
            annos += ar
            if arrow_skips is not None:
                arrow_skips += skp
            if fam == "pov" and layers.has("floorcraft") and any((a.get("target") or {}).get("type") == "leader_t" for a in ar):
                # an arrow on the leader's T needs the T under it: drawn on the phone's own floor (registered like the skeleton)
                lay.update({"floorcraft": "current_t", "t_width_m": 0.05, "t_brightness": 1.0})
        items = []                                   # (position, kind, insert)
        for ins in inserts:
            if ins["kind"] == "gap_slow" and b.t0 - 1e-6 <= ins["core"][0] and ins["core"][1] <= b.t1 + 1e-6:
                items.append((ins["core"][0], "slow", ins))
            elif ins["kind"] == "replay" and b.t0 + 1e-6 < ins["park"] <= b.t1 + 1e-6:
                items.append((ins["park"], "replay", ins))
        items.sort(key=lambda x: x[0])
        pieces = []
        cur = b.t0
        for pos, kind, ins in items:
            if kind == "slow":
                if pos > cur + 1e-6:
                    pieces.append(("locked", cur, pos, None))
                pieces.append(("slow", ins["core"][0], ins["core"][1], ins))
                cur = ins["core"][1]
            else:
                if pos > cur + 1e-6:
                    pieces.append(("locked", cur, pos, None))
                    cur = pos
                pieces.append(("replay", ins["core"][0], ins["core"][1], ins))
        if b.t1 > cur + 1e-6:
            pieces.append(("locked", cur, b.t1, None))
        k = 0
        for kind, d0, d1, ins in pieces:
            k += 1
            if kind == "locked":
                sid = b.id if k == 1 else f"{b.id}_p{k}"
                dur = d1 - d0
                seg = {"id": sid, "dance": [round(d0, 3), round(d1, 3)], "speed": 1.0, "camera": dict(cam), "view_state": view, "layers": dict(lay),
                       "annotations": [], "audio": "locked", "status": "ok"}
            elif kind == "slow":
                sid = f"{b.id}_slow{k}"
                dur = (d1 - d0) / ins["speed"]
                seg = {"id": sid, "dance": [round(d0, 3), round(d1, 3)], "speed": ins["speed"], "camera": dict(cam), "view_state": view,
                       "layers": dict(lay), "annotations": [], "audio": "bed", "status": "ok",
                       "notes": f"slow motion in a speech-free gap ({ins['why']}); the room-tone bed plays, the speech is not slowed"}
            else:
                sid = f"{b.id}_replay{k}"
                dur = (d1 - d0) / ins["speed"]
                rfam = "axis" if layers.any("floorcraft") else "body"
                rcam = plan.camera_for(rfam, variant=b.variant, focus="couple", cfg=cfg)
                if rcam.get("mode") == "overhead_floorcraft":
                    rcam["across_m"] = max(float(rcam.get("across_m", 0)), 2.8)    # a pivot throws arms and legs wide: never crop them
                rcam = plan.with_horizontal(rcam, cfg)
                seg = {"id": sid, "dance": [round(d0, 3), round(d1, 3)], "speed": ins["speed"], "camera": rcam,
                       "view_state": plan.VIEW_STATE[rfam], "layers": plan.treatment_layers(rfam, layers, dropped, sid, 0, cfg.avatar_opacity, cfg.avatar_on_video),
                       "annotations": [],
                       "audio": "bed", "replay": True, "status": "ok",
                       "notes": f"replay inserted at a sentence end, the speech is parked ({ins['why']}); the speech resumes at {ins['park']:.2f} s"}
            seg["film"] = [round(f, 3), round(f + dur, 3)]
            f += dur
            seg["narration"] = []
            if k == 1:
                seg["notes_beat"] = "; ".join(b.why[:4]) if b.why else "no teaching event: the teachers talk"
                if not seg.get("notes"):
                    seg["notes"] = f"{b.family}: " + (seg["notes_beat"])
                seg["say"] = _say(b)
                if b.treatment:
                    seg["treatment"] = dict(b.treatment)
                    seg["director"] = True
                if fam == "pov":
                    seg["layers"]["source_video"] = (b.treatment or {}).get("phone", "")
                if b.starts_inner:
                    seg["demanded"] = True
                    seg["notes"] += " | a shot change inside a long sentence: its two halves carry different teaching points"
                if b.chapter_start:
                    idx = b.chapter
                    seg["chapter"] = titles[idx]
                    seg["chapter_card"] = _card(idx, titles[idx], title, seg["film"][1] - seg["film"][0], cfg)
            segs.append(seg)
        # annotations onto the pieces that carry them
        for a in annos:
            for seg in segs:
                if seg["id"] == b.id or seg["id"].startswith(b.id + "_"):
                    if seg.get("replay"):
                        continue
                    a0, a1 = max(a["dance_t"][0], seg["dance"][0]), min(a["dance_t"][1], seg["dance"][1])
                    if a1 - a0 < 0.3:
                        continue
                    na = {k2: v for k2, v in a.items() if k2 != "dance_t"}
                    na["dance_t"] = [round(a0, 3), round(a1, 3)]
                    na["film_in_segment"] = [round(_dance_to_film(seg, a0), 3), round(_dance_to_film(seg, a1), 3)]
                    seg["annotations"].append(na)
    return segs


def _say(b: plan.Beat) -> str:
    return f"{b.speaker or '-'}: " + ", ".join(b.utterances[:3])


def _card(idx: int, title: str, film: tuple[str, str], seg_dur: float, cfg: plan.Cfg) -> dict:
    if idx == 0:
        return {"kind": "title", "title": film[0], "subtitle": film[1], "film_in": 0.5, "film_out": round(min(seg_dur - 0.2, 0.5 + cfg.card_s + 0.6), 2)}
    return {"kind": "chapter", "index": idx, "title": title, "film_in": 0.35, "film_out": round(min(seg_dur - 0.2, 0.35 + cfg.card_s), 2)}


# ------------------------------------------------------------------------------------------------ the direction document


def load_arrow_ctx(cap_dir: Path | None, layers: plan.Layers) -> dict:
    """what the arrows need from the capture: the floorcraft T axes (the leader's T alive at a time) and each dancer's weight transfers (physics contact)"""
    ctx: dict = {"layers": layers, "t_axes": [], "transfers": {}}
    if cap_dir is None or not Path(cap_dir).is_dir():
        return ctx
    fc = Path(cap_dir) / "floorcraft.json"
    if fc.exists():
        try:
            ctx["t_axes"] = [{"id": a["id"], "t0": a["t0"], "t1": a["t1"]} for a in json.loads(fc.read_text(encoding="utf-8")).get("t_axes", [])]
        except (OSError, ValueError, KeyError):
            pass
    ph = Path(cap_dir) / "physics.json"
    if ph.exists():
        try:
            d = json.loads(ph.read_text(encoding="utf-8"))
            t = d.get("t") or []
            hz = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] else 30.0
            ctx["transfers"] = {r: arrows.weight_transfers(d["dancers"][r].get("contact"), hz) for r in ("lead", "follow") if r in d.get("dancers", {})}
            if all(r in d.get("dancers", {}) and d["dancers"][r].get("vcom") for r in ("lead", "follow")):
                ctx["vcom"] = {"t": np.asarray(t, float), "lead": np.asarray(d["dancers"]["lead"]["vcom"], float), "follow": np.asarray(d["dancers"]["follow"]["vcom"], float),
                               "mass_lead": float(d["dancers"]["lead"].get("mass_kg") or 75.0), "mass_follow": float(d["dancers"]["follow"].get("mass_kg") or 55.0)}
        except (OSError, ValueError, KeyError):
            pass
    return ctx


def build_direction(sp: dict, act: plan.Activity, layers: plan.Layers, cfg: plan.Cfg, paths: Paths, capture: str, title: tuple[str, str],
                    pov: director.PovSource | None = None, dcfg: director.DirCfg | None = None, arrow_ctx: dict | None = None) -> tuple[dict, dict, dict]:
    """(direction doc, timeline doc, report). Pure given its inputs. `pov` switches the director mode on (phone POV shots, feature zooms)."""
    duration = sp["duration"]
    beats, brep = plan.plan_beats(sp["utterances"], sp["events"], sp["chapters"], act, layers, sp["roles"], duration, cfg)
    inserts, rejected = plan.plan_inserts(sp["utterances"], act, duration, cfg)
    dropped = list(brep["dropped"])
    drep: dict = {}
    events = sp["events"]
    if pov is not None:
        dc = dcfg or director.DirCfg()
        events = director.physics_calls(sp["utterances"], sp["events"], dc)      # D4: the teachers' events + the physics words nothing covered
        ctx = director.Ctx(sp["utterances"], events, sp["chapters"], act, layers, sp["roles"], duration, cfg, dc, pov, inserts,
                           director.cut_points(sp["utterances"], cfg.lead_s), dropped, brep["notes"])
        beats, drep = director.direct(beats, ctx)
        drep["attention"] = director.attention_table(beats, [e for e in events if not e.get("synthetic")], dc)
    arrow_skips: list[dict] = []
    segs = assemble(beats, inserts, layers, events, sp["chapters"], act, cfg, title, dropped, arrow_ctx if arrow_ctx is not None else ({"layers": layers, "t_axes": [], "transfers": {}} if pov is not None else None), arrow_skips)
    if drep:
        drep["arrows"] = arrow_report(segs, drep.get("attention", []), arrow_skips)
    tm = plan.TimeMap(inserts)
    film_duration = segs[-1]["film"][1]
    # captions
    words = caps.flatten_words(sp["captions"])
    words = [w for w in words if w["end"] > 0 and w["start"] >= 0]
    speakers = caps.speaker_table(sp["captions"])
    breaks = [i["park"] for i in inserts if i["kind"] == "replay"] + [i["core"][1] for i in inserts if i["kind"] == "gap_slow"]
    chunks = {asp: caps.build_chunks(words, caps.SHAPES[asp], tm.film, speakers, breaks, tag=asp[0]) for asp in ("vertical", "horizontal")}
    tl = caps.timeline_doc(NAME, film_duration, chunks, speakers)
    # a chapter starts where the teacher's first sentence of it does, not at the shot's lead-in (the camera starts 0.3 s before the words): the
    # library's chapter marks and the pill agree with the speech chapters (the audit saw the marks 0.1-0.3 s early)
    chapters_out = []
    for s in segs:
        if "chapter" not in s:
            continue
        idx = len(chapters_out)
        lead = max(0.0, min(0.5, float(sp["chapters"][idx]["start"]) - s["dance"][0])) if 0 < idx < len(sp["chapters"]) else 0.0
        chapters_out.append({"title": s["chapter"], "startS": round(s["film"][0] + lead, 3), "segment": s["id"], "master_s": s["dance"][0] + lead,
                             "offset_s": round(lead, 3)})
    for i, c in enumerate(chapters_out):
        c["endS"] = chapters_out[i + 1]["startS"] if i + 1 < len(chapters_out) else round(film_duration, 3)
    slow_ext = sum(i["added_s"] for i in inserts if i["kind"] == "gap_slow")
    replay_s = sum(i["added_s"] for i in inserts if i["kind"] == "replay")
    families: dict[str, float] = {}
    for b in beats:
        families[b.family] = round(families.get(b.family, 0.0) + b.dur, 2)
    doc = {
        "version": 1, "kind": "direction", "name": NAME, "title": f"{title[0]}", "capture": capture, "take": TAKE,
        "avatar_opacity": round(float(cfg.avatar_opacity), 3),
        "time_base": "DANCE time = master seconds (the recap's master phone clock = the capture's time base, anchor 0); FILM time = seconds into the rendered "
                     "video. Locked segments run at 1.0x with the class audio; slow motion and replays are inserts (see `audio`).",
        "dance_range": [0.0, round(duration, 3)],
        "data_status": ("PLAN: " if layers.assumed else "") + layers.note + ("; the owner's opening is not settled" if owner_opening_status(paths)["answer"] == "not yet" else ""),
        "pending": [], "created": "generated by dancecap.film_recap build (python -m dancecap.film_recap build): edit the rules there, not this file",
        "presets": presets_for(cfg), "style": STYLE, "captions": CAPTIONS,
        "audio": {
            "music": "none: the recap has no soundtrack (no rhythm analysis); the film's soundtrack is the class audio",
            "class_mix": "audio/class_mix.wav (master clock, 48 kHz stereo, -16 LUFS, true peak <= -1 dBTP)",
            "film_mix": f"audio/{NAME}_mix.wav (film clock: locked segments copy the class mix, inserts play the room-tone bed)",
            "slow_motion_rule": "the teachers' speech is only heard at 1.0x locked to its picture: a slowed passage lies in a speech-free gap (gap_slow) or "
                                "is a replay inserted at a sentence end with the speech parked; both play the room-tone bed, no pitch change ever",
            "segment_audio_values": {"locked": "the class mix at the dance (master) time", "bed": "room tone (slow motion, replays)"},
            "render_notice": "PRIVATE RENDER: contains the voices of real people (the class audio). Local only; no upload, no third-party voice or speech service. "
                             "Renders are for the teachers' own non-commercial use.",
        },
        "segments": segs, "narration": [],
        "chapters": [{"title": c["title"], "segment": c["segment"], "film_s": c["startS"]} for c in chapters_out],
        "timeline": {"film_duration_s": round(film_duration, 3), "film_duration_mmss": f"{int(film_duration // 60)}:{film_duration % 60:04.1f}",
                     "dance_span_s": round(duration, 3), "slow_motion_and_hold_extension_s": round(slow_ext, 2), "replay_s": round(replay_s, 2),
                     "inserts": [{"id": i["id"], "kind": i["kind"], "core": i["core"], "speed": i["speed"], "park": i.get("park"), "added_s": i["added_s"]} for i in inserts],
                     "narration_lines": 0, "t_zero_dance_s": 0.0, "t_zero_shift_s": 0.0,
                     "caption_chunks": {k: len(v) for k, v in chunks.items()}},
        "recap_film": {"director_mode": pov is not None, "pov_share": drep.get("pov_share"), "pov_seconds_per_phone": drep.get("pov_seconds_per_phone"),
                       "layers": layers.available, "layers_assumed": layers.assumed, "layers_note": layers.note, "phone_videos": layers.phones,
                       "activity_source": act.source, "families_seconds": families},
    }
    report = {"schema": "dancecap.film_recap/1/report", "capture": capture, "film_duration_s": round(film_duration, 3), "master_duration_s": round(duration, 3),
              "beats": len(beats), "segments": len(segs), "families_seconds": families, "inserts": doc["timeline"]["inserts"],
              "inserts_rejected": rejected, "activity_source": act.source, "layers": layers.available, "layers_assumed": layers.assumed,
              "layers_note": layers.note, "phone_videos": layers.phones, "dropped": _squash(dropped), "plan_notes": brep["notes"],
              "chapters": chapters_out, "caption_chunks": {k: len(v) for k, v in chunks.items()}, "director": drep,
              "emoji_words": sum(1 for w in words if w["kind"] == "emoji"),
              "needs_check_captions": [c["id"] for c in sp["captions"].get("captions", []) if c.get("check")],
              "owner_opening": owner_opening_status(paths)}
    return doc, tl, report


def arrow_report(segs: list[dict], attention: list[dict], skips: list[dict]) -> dict:
    """the arrows of the film: counts by kind, which attention calls got one (and which did not, with the reason)"""
    arrows_: list[tuple[dict, dict]] = [(s, a) for s in segs for a in s["annotations"] if a["kind"] in ("arrow3d", "vector_arrow", "floor_ring") and a.get("calls")]
    by_call: dict[str, list[str]] = {}
    for s, a in arrows_:
        for c in a.get("calls", []):
            by_call.setdefault(c, []).append(a["id"])
    kinds: dict[str, int] = {}
    for s in segs:
        for a in s["annotations"]:
            kinds[a["kind"]] = kinds.get(a["kind"], 0) + 1
    why = {x["id"]: x["why"] for x in skips}
    text_of = {a["id"]: f"{a['text']} ({a['target'].get('type') if isinstance(a.get('target'), dict) else a.get('vector', '')})" for s in segs for a in s["annotations"] if a.get("calls")}
    rows = [{"id": a["id"], "t": a["t"], "words": a["words"], "arrows": by_call.get(a["id"], []), "labels": [text_of[i] for i in by_call.get(a["id"], []) if i in text_of],
             "no_arrow_because": None if by_call.get(a["id"]) else why.get(a["id"], "no arrow planned")} for a in attention]
    n = len(rows)
    got = sum(1 for r in rows if r["arrows"])
    pov_ar = sum(1 for s, a in arrows_ if s.get("treatment", {}).get("kind") == "pov")
    return {"annotation_kinds": kinds, "attention_calls": n, "calls_with_arrow": got, "calls_without": n - got, "arrows_in_phone_shots": pov_ar,
            "event_arrows": len(arrows_), "rows": rows}


def _squash(items: list[dict]) -> list[dict]:
    """the dropped list without duplicates: {what, needs, n, first}"""
    out: dict[tuple, dict] = {}
    for d in items:
        k = (d["what"].split("=")[0] if d["what"].startswith("layer") else d["what"].split(" ")[0] + " " + d["what"].split(" ")[1] if " " in d["what"] else d["what"], d["needs"])
        e = out.setdefault(k, {"what": d["what"], "needs": d["needs"], "n": 0, "where": []})
        e["n"] += 1
        if len(e["where"]) < 6:
            e["where"].append(d["where"])
    return list(out.values())


# ------------------------------------------------------------------------------------------------ validation


def validate(doc: dict, tl: dict, sp: dict) -> list[str]:
    """problems that must stop a build (empty = fine)"""
    bad: list[str] = []
    segs = doc["segments"]
    if not segs:
        return ["no segments"]
    if abs(segs[0]["film"][0]) > 1e-6:
        bad.append("the film does not start at 0")
    ins = {i["id"]: i for i in doc["timeline"]["inserts"]}
    tm = plan.TimeMap([{"kind": i["kind"], "park": i.get("park"), "core": i["core"], "added_s": i["added_s"]} for i in ins.values()])
    prev_f, prev_d = 0.0, 0.0
    ids = set()
    for s in segs:
        if s["id"] in ids:
            bad.append(f"duplicate segment id {s['id']}")
        ids.add(s["id"])
        if abs(s["film"][0] - prev_f) > 2e-3:
            bad.append(f"{s['id']}: film starts at {s['film'][0]} but the previous segment ends at {prev_f}")
        d0, d1 = s["dance"]
        sp_ = s["speed"]
        want = (d1 - d0) / sp_
        if abs((s["film"][1] - s["film"][0]) - want) > 3e-3:
            bad.append(f"{s['id']}: film length {s['film'][1] - s['film'][0]:.3f} != dance {d1 - d0:.3f} / speed {sp_}")
        prev_f = s["film"][1]
        if not s.get("replay"):
            if abs(d0 - prev_d) > 2e-3:
                bad.append(f"{s['id']}: dance starts at {d0}, expected {prev_d} (a gap or a jump without replay)")
            prev_d = d1
        locked = s.get("audio") == "locked" and sp_ == 1.0 and not s.get("replay")
        if locked and abs((s["film"][0] - d0) - tm.extra(d0 + 1e-6)) > 5e-3:
            bad.append(f"{s['id']}: a locked segment's film-dance offset {s['film'][0] - d0:.3f} != the inserts before it {tm.extra(d0 + 1e-6):.3f} (the speech would drift)")
        if not locked and s.get("audio") != "bed":
            bad.append(f"{s['id']}: a slowed / replayed segment must use audio 'bed'")
        if not locked:
            # sound policy: no word of the transcript is heard in a non-locked segment; a gap_slow window must be speech-free
            if not s.get("replay") and any(w["start"] < d1 and w["end"] > d0 for u in sp["utterances"] for w in u["words"] if w["start"] >= 0):
                bad.append(f"{s['id']}: slow motion over speech (the speech would be slowed or lost)")
        if s.get("audio") == "locked" and sp_ != 1.0:
            bad.append(f"{s['id']}: locked audio at speed {sp_}")
        for a in s["annotations"]:
            fa = a["film_in_segment"]
            if not (-1e-6 <= fa[0] < fa[1] <= (s["film"][1] - s["film"][0]) + 1e-3):
                bad.append(f"{s['id']}/{a['id']}: film_in_segment {fa} outside the segment")
    if abs(prev_d - sp["duration"]) > 5e-3:
        bad.append(f"the dance does not reach the end: {prev_d} vs {sp['duration']}")
    if abs(doc["timeline"]["film_duration_s"] - prev_f) > 5e-3:
        bad.append("timeline.film_duration_s != the last segment's end")
    # chapters
    ch = [s for s in segs if "chapter" in s]
    if len(ch) != len(sp["chapters"]):
        bad.append(f"{len(ch)} chapter segments for {len(sp['chapters'])} chapters")
    for s in ch:
        if len(s["chapter"].split()) > 5:
            bad.append(f"chapter title longer than 5 words: {s['chapter']!r}")
    # no cut inside a sentence (unless demanded): shot starts are segment starts whose camera differs from the previous one
    cuts = []
    for a, b in zip(segs, segs[1:]):
        if a["camera"] != b["camera"] or (b.get("replay") or a.get("replay")):
            cuts.append(b)
    for u in sp["utterances"]:
        if u["end"] - u["start"] < 1.5 or u["start"] < 0:
            continue
        for c in cuts:
            if c.get("replay"):
                continue
            t = c["dance"][0] - tm.extra(c["dance"][0] + 1e-6) * 0 if not c.get("replay") else c["dance"][0]
            if u["start"] + 0.4 < t < u["end"] - 0.4 and not c.get("demanded", False) and not _demanded(doc, c["id"]) and not c.get("director"):
                bad.append(f"a shot change at {t:.2f} s inside the sentence {u['id']} ({u['start']:.1f}-{u['end']:.1f} s) without a demanding event ({c['id']})")
    # placeholders and bad text
    for path, text in _strings(doc["segments"], "segments"):
        if re.search(r"\bTBD\b|\{[A-Za-z_]+\}", text):
            bad.append(f"placeholder in {path}: {text[:60]!r}")
    # captions
    for asp, chunks in tl["captions"].items():
        bad += [f"captions {asp}: {p}" for p in caps.check_chunks(chunks, caps.SHAPES[asp])]
        n_emoji = sum(1 for c in chunks for w in c["words"] if w.get("kind") == "emoji")
        want = sum(1 for u in sp["transcript"]["utterances"] for w in u["words"] if w.get("kind") == "event" and w["start"] >= 0)
        if want and n_emoji < want and n_emoji == 0:
            bad.append(f"captions {asp}: the transcript has {want} laugh tokens, the caption timeline none")
    # caption words stand at the film time of their master time
    tmf = tm.film
    for c in tl["captions"]["vertical"]:
        for w in c["words"]:
            pass
    # the arrows: at most plan.Cfg.max_arrows_at_once labelled call-outs at a time (D4; the physics view's continuous velocity arrows are not call-outs)
    spans = sorted((a["film_in_segment"][0] + s["film"][0], a["film_in_segment"][1] + s["film"][0], a["id"]) for s in segs for a in s["annotations"]
                   if a["kind"] in ("arrow3d", "vector_arrow", "floor_ring") and not a.get("continuous"))
    limit = plan.Cfg().max_arrows_at_once
    for k, (x0, x1, aid) in enumerate(spans):
        live = [b for b in spans[:k] if b[1] > x0 + 0.05]
        if len(live) + 1 > limit:
            bad.append(f"{len(live) + 1} arrows at once at {x0:.2f} s: {aid} with {', '.join(b[2] for b in live)}")
    return bad


def _demanded(doc: dict, seg_id: str) -> bool:
    for s in doc["segments"]:
        if s["id"] == seg_id:
            return bool(s.get("notes", "")) and "split at" in s.get("notes", "")
    return False


def _strings(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(o, str):
        yield path, o


# ------------------------------------------------------------------------------------------------ review table


def review_md(doc: dict, report: dict, sp: dict, tl: dict) -> str:
    segs = doc["segments"]
    tm = plan.TimeMap([{"kind": i["kind"], "park": i.get("park"), "core": i["core"], "added_s": i["added_s"]} for i in doc["timeline"]["inserts"]])
    out = [f"# Direction review: {doc['title']}", "",
           f"Film {doc['timeline']['film_duration_mmss']} ({doc['timeline']['film_duration_s']} s) from {report['master_duration_s']} s of class; "
           f"{len(segs)} segments; activity source: {report['activity_source']}; {report['layers_note']}.", "",
           "## What the teachers say / what the film shows, per shot", "",
           "| film | master | ch | speaker | camera | layers / call-outs | the teachers say |", "|---|---|---|---|---|---|---|"]
    spk_at = lambda t: next((u["speaker"] for u in sp["utterances"] if u["start"] - 0.2 <= t <= u["end"] + 0.2), "")
    for s in segs:
        txt = " ".join(w["text"] for u in sp["utterances"] for w in u["words"] if s["dance"][0] <= w["start"] < s["dance"][1] and w.get("kind") != "event") if not s.get("replay") else "(speech parked)"
        ann = ", ".join(f"{a['kind']}:{a.get('text', '-')}" for a in s["annotations"]) or "-"
        lay = ",".join(k for k in ("floorcraft", "counterbalance", "skeleton_mode", "footprints", "graph") if s["layers"].get(k) not in (None, False, "off", "rhythm"))
        cam = s["camera"]["mode"] + (f" focus={s['camera'].get('focus')}" if s["camera"].get("focus") else "") + (f" x{s['speed']}" if s["speed"] != 1.0 else "") + (" REPLAY" if s.get("replay") else "")
        out.append(f"| {s['film'][0]:.1f}-{s['film'][1]:.1f} | {s['dance'][0]:.1f}-{s['dance'][1]:.1f} | {s.get('chapter', '')} | {spk_at(0.5 * (s['dance'][0] + s['dance'][1]))} | "
                   f"{cam} `{s['id']}` | {lay}; {ann} | {txt[:140]} |")
    out += ["", "## Second by second (film second: master second, speaker, what is said, shot, call-outs)", "",
            "| film s | master s | speaker | the teachers say | shot | call-out / card |", "|---|---|---|---|---|---|"]
    words = sorted((w["start"], w["end"], w["text"], u["speaker"]) for u in sp["utterances"] for w in u["words"] if w.get("kind") != "event" and w["start"] >= 0)
    seg_at = lambda f: next((s for s in segs if s["film"][0] - 1e-9 <= f < s["film"][1] - 1e-9), segs[-1])
    n = int(doc["timeline"]["film_duration_s"])
    for fs in range(n + 1):
        s = seg_at(fs + 0.5)
        if s.get("replay"):
            out.append(f"| {fs} | {s['dance'][0]:.0f}.. (replay) | - | (speech parked, room tone) | {s['camera']['mode']} `{s['id']}` | slow motion {s['speed']}x |")
            continue
        m = s["dance"][0] + (fs + 0.5 - s["film"][0]) * s["speed"]
        say = " ".join(w[2] for w in words if m - 0.5 <= w[0] < m + 0.5)
        sk = next((w[3] for w in words if m - 0.5 <= w[0] < m + 0.5), "")
        an = ", ".join(f"{a['text']}" for a in s["annotations"] if a["film_in_segment"][0] + s["film"][0] <= fs + 0.5 < a["film_in_segment"][1] + s["film"][0] and a.get("text"))
        card = ""
        if "chapter_card" in s and s["film"][0] + s["chapter_card"]["film_in"] <= fs + 0.5 < s["film"][0] + s["chapter_card"]["film_out"]:
            card = f"CARD: {s['chapter_card']['title']}"
        out.append(f"| {fs} | {m:.0f} | {sk} | {say[:110]} | {s['camera']['mode']} `{s['id']}` | {an}{' ' if an and card else ''}{card} |")
    dr = report.get("director") or {}
    if dr:
        out += ["", "## Director mode: the phones' own video", "",
                f"{dr['pov_shots_n']} phone shots, {dr['pov_seconds']} s = {dr['pov_share'] * 100:.1f} % of the film's screen time; per phone (s / share): " +
                ", ".join(f"{c} {v} s / {dr['pov_share_per_phone'][c] * 100:.1f} %" for c, v in dr["pov_seconds_per_phone"].items()), "",
                "| film | master | phone | how | teachers seen | height | crop zoom | inside | the teachers say |", "|---|---|---|---|---|---|---|---|---|"]
        for t in dr.get("pov_shots", []):
            sg = next((x for x in segs if x.get("treatment") is t or (x.get("treatment", {}).get("phone") == t.get("phone") and x.get("camera", {}).get("mode") in ("camera_tour", "fly_to_camera")
                                                                           and x.get("treatment", {}).get("zoom_median") == t.get("zoom_median"))), None)
            if sg is None:
                continue
            txt = " ".join(w["text"] for u in sp["utterances"] for w in u["words"] if sg["dance"][0] <= w["start"] < sg["dance"][1] and w.get("kind") != "event")
            out.append(f"| {sg['film'][0]:.1f}-{sg['film'][1]:.1f} | {sg['dance'][0]:.1f}-{sg['dance'][1]:.1f} | {t['phone']} | {'fly in' if t.get('fly') else 'cut'} | "
                       f"{t['cover'] * 100:.0f} % | {t['height'] * 100:.0f} % | {t['zoom_median']} | {t['inside'] * 100:.0f} % | {txt[:90]} |")
        arow = {r["id"]: r for r in (dr.get("arrows") or {}).get("rows", [])}
        out += ["", "## Attention calls: what the film shows, per call", "",
                "| call | master s | who | the teacher's word(s) | shot | what is shown | arrow |", "|---|---|---|---|---|---|---|"]
        for a in dr.get("attention", []):
            r = arow.get(a["id"], {})
            arrow = "; ".join(r.get("labels", [])) if r.get("arrows") else f"none: {r.get('no_arrow_because', '-')}"
            out.append(f"| {a['id']}{' (flagged)' if a['needs_check'] else ''} | {a['t']:.1f} | {a['speaker']} | {', '.join(a['words'])} | `{a['shot']}` | {a['shown']} | {arrow} |")
        ar_ = dr.get("arrows")
        if ar_:
            out += ["", f"Arrows: {ar_['calls_with_arrow']} of {ar_['attention_calls']} attention calls have one ({ar_['calls_without']} without, reasons in the table), "
                        f"{ar_['event_arrows']} call-out arrows in all, {ar_['arrows_in_phone_shots']} of them in phone shots; annotation kinds {ar_['annotation_kinds']}."]
    out += ["", "## Dropped (analysis data absent)", ""]
    out += [f"- {d['what']} needs {d['needs']} ({d['n']}x; e.g. {', '.join(d['where'][:3])})" for d in report["dropped"]] or ["- nothing dropped"]
    out += ["", "## Slow motion and replays", ""]
    out += [f"- {i['id']}: {i['kind']} {i['core']} at {i['speed']}x" + (f", speech parked at {i['park']}" if i.get("park") else "") + f", +{i['added_s']} s" for i in report["inserts"]] or ["- none planned"]
    out += [f"- not used: cue '{r.get('cue', '')}' at {r.get('t', '')}: {r['why']}" for r in report["inserts_rejected"]]
    out += ["", "## Plan notes", ""] + [f"- {x}" for x in report["plan_notes"]]
    out += ["", "## Proofread first (captions the transcript marks as unsure)", "", ", ".join(report["needs_check_captions"]) or "none"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ build


MIX_SCRIPT = '''"""the film recorder's soundtrack hook: rebuild <direction.name>_mix.wav from the direction (generated by dancecap.film_recap)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))          # .../dancecap
from dancecap import film_recap_audio as fa  # noqa: E402

direction = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "direction.json"
print(fa.film_mix(direction))
'''


def write_all(paths: Paths, doc: dict, tl: dict, report: dict, review: str) -> None:
    film = paths.film
    (film / "narration").mkdir(parents=True, exist_ok=True)
    paths.audio.mkdir(parents=True, exist_ok=True)
    (film / "direction.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    caps.write_timeline(film / "narration" / "timeline.json", tl)
    ch = [{"title": c["title"], "startS": c["startS"], "endS": c["endS"], "segment": c["segment"], "offset_s": c.get("offset_s", 0.0)} for c in report["chapters"]]
    (film / "chapters.json").write_text(json.dumps({"schema": "dancecap.film_recap/1/chapters", "film_duration_s": report["film_duration_s"], "chapters": ch},
                                                   indent=1, ensure_ascii=False), encoding="utf-8")
    (film / "direction_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (film / "direction_review.md").write_text(review, encoding="utf-8")
    (paths.audio / "make_film_mix.py").write_text(MIX_SCRIPT, encoding="utf-8")


def load_pov(paths: Paths, mode: str) -> director.PovSource | None:
    """the phones' data for the director mode: teacher boxes + poses of every phone (people/teachers_<cam>.npz, cameras/<cam>.npz) with the chunks' own
    video offsets; None when `mode` is 'no' or (auto) when the files do not exist"""
    if mode == "no":
        return None
    try:
        phones = recap_pov.build_phones(paths.work)
    except (FileNotFoundError, OSError, KeyError) as e:
        if mode == "yes":
            raise RecapError(f"director mode needs the phones' people / camera files ({e})") from e
        return None
    if not phones:
        if mode == "yes":
            raise RecapError("director mode: no phone has people/teachers_<cam>.npz + cameras/<cam>.npz")
        return None
    return director.PovSource(phones, recap_pov.bad_stretches())


def build(paths: Paths | None = None, *, layers_mode: str = "auto", capture: str = CAPTURE, refresh: str = "auto", no_mix: bool = False,
          cfg: plan.Cfg | None = None, log=print, director_mode: str = "auto") -> dict:
    paths = paths or Paths()
    cfg = cfg or plan.Cfg()
    msg = refresh_speech(paths, refresh)
    if msg:
        log(msg)
    sp = load_speech(paths)
    cap_dir = VIEWER / "Assets" / "StreamingAssets" / capture
    layers = {"auto": lambda: plan.Layers.probe(cap_dir, True), "assume": lambda: plan.Layers.probe(None, True), "none": lambda: plan.Layers.probe(None, False),
              "all": plan.Layers.all_on}[layers_mode]()
    act = load_activity(paths, cap_dir if cap_dir.is_dir() else None)
    pov = load_pov(paths, director_mode)
    doc, tl, report = build_direction(sp, act, layers, cfg, paths, capture, film_title(), pov, arrow_ctx=load_arrow_ctx(cap_dir if cap_dir.is_dir() else None, layers))
    bad = validate(doc, tl, sp)
    if bad:
        raise RecapError("direction.json NOT written - fix these first:\n  " + "\n  ".join(bad[:40]))
    review = review_md(doc, report, sp, tl)
    report["inputs"] = {n: _sha(paths.speech / n) for n in ("transcript_final.json", "captions.json", "chapters.json", "teaching_events.json", "jokes.json")}
    write_all(paths, doc, tl, report, review)
    log(f"direction.json: {doc['timeline']['film_duration_mmss']} ({doc['timeline']['film_duration_s']} s), {len(doc['segments'])} segments, "
        f"{len(report['chapters'])} chapters, families {report['families_seconds']}, inserts {len(report['inserts'])}, "
        f"captions {report['caption_chunks']}, dropped {len(report['dropped'])} kinds; {layers.note}")
    if not no_mix:
        from . import film_recap_audio as fa
        cm = fa.soundtrack_class_mix(paths.audio)       # class_mix_v2v.wav when present (switch: audio/soundtrack.json), else class_mix.wav
        if cm.exists():
            mixed = fa.film_mix(paths.film / "direction.json", cm)
            log(f"film mix from {cm.name}: " + json.dumps({k: v for k, v in mixed.items() if k != "slow"}))
            for row in mixed["slow"]:       # S8: the slowed / held / replayed parts play room tone only, never speech
                log(f"  slowed {row['id']} film {row['film'][0]:.1f}-{row['film'][1]:.1f} s: {row['rms_db']} dB " + ("room tone only" if row["room_tone_only"] else "NOT ROOM TONE: speech may play in slow motion"))
        else:
            log("no audio/class_mix.wav yet: run `python -m dancecap.film_recap mix class` (a few minutes), then `build` again")
    return report


# ------------------------------------------------------------------------------------------------ CLI


def cmd_build(a: argparse.Namespace) -> int:
    try:
        cfg = plan.Cfg()
        if a.no_horizontal_overrides:
            cfg = replace(cfg, horizontal_overrides=False)
        if a.avatar_opacity is not None:
            cfg = replace(cfg, avatar_opacity=min(1.0, max(0.0, float(a.avatar_opacity))))
        if a.avatar_on_video:
            cfg = replace(cfg, avatar_on_video=True)
        build(layers_mode=a.layers, capture=a.capture, refresh=a.refresh_speech, no_mix=a.no_mix, director_mode=a.director, cfg=cfg)
    except RecapError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    paths = Paths()
    sp = load_speech(paths)
    doc = json.loads((paths.film / "direction.json").read_text(encoding="utf-8"))
    tl = json.loads((paths.film / "narration" / "timeline.json").read_text(encoding="utf-8"))
    bad = validate(doc, tl, sp)
    print("OK" if not bad else "\n".join(bad))
    return 1 if bad else 0


def cmd_emoji(a: argparse.Namespace) -> int:
    paths = Paths()
    out = paths.film / "emoji"
    made = []
    for ch, code in caps.EMOJI_CODE.items():
        p = caps.render_emoji_png(ch, out / f"{code}.png")
        made.append(f"{code}.png: {'written' if p else 'no colour emoji font here: the viewer draws its built-in version'}")
    print("\n".join(made))
    return 0


def cmd_chapters(a: argparse.Namespace) -> int:
    paths = Paths()
    ch = json.loads((paths.film / "chapters.json").read_text(encoding="utf-8"))["chapters"]
    print("\n".join(f'[[recap.chapters]]\ntitle = "{c["title"]}"\nsegment = "{c["segment"]}"\n' + (f'offset_s = {c["offset_s"]}\n' if c.get("offset_s") else "") for c in ch))
    return 0


def cmd_mix(a: argparse.Namespace) -> int:
    from . import film_recap_audio as fa
    paths = Paths()
    if a.which == "class":
        rep = fa.build_class_mix()
        print(json.dumps(rep["final"]["after"]))
    else:
        print(json.dumps(fa.film_mix(paths.film / "direction.json")))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.film_recap", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build", help="direction + captions + chapters + review + film mix (the one command)")
    b.add_argument("--layers", choices=["auto", "assume", "none", "all"], default="auto",
                   help="auto: probe the capture's analysis files (the planned set is assumed while the capture does not exist); assume | none | all: force")
    b.add_argument("--capture", default=CAPTURE)
    b.add_argument("--refresh-speech", choices=["auto", "yes", "no"], default="auto")
    b.add_argument("--no-mix", action="store_true")
    b.add_argument("--avatar-opacity", type=float, default=None,
                   help="the ONE opacity of every avatar the film draws (default plan.Cfg.avatar_opacity = 0.25, i.e. 75 %% transparent; 0.75 = 25 %% transparent)")
    b.add_argument("--no-horizontal-overrides", action="store_true",
                   help="omit the opt-in 16:9 camera overrides / preset switches (the 9:16 direction is the same with or without them)")
    b.add_argument("--avatar-on-video", action="store_true", help="keep the 3D avatars (at that opacity) over a phone's own video too (default: only the skeleton lines)")
    b.add_argument("--director", choices=["auto", "yes", "no"], default="auto",
                   help="director mode (phone POV shots circling through the phones, tight feature zooms, closer overhead): auto = when the phones' people / camera files exist")
    b.set_defaults(fn=cmd_build)
    sub.add_parser("check", help="validate the written direction").set_defaults(fn=cmd_check)
    sub.add_parser("emoji", help="write the laugh emoji image from the system's colour-emoji font").set_defaults(fn=cmd_emoji)
    sub.add_parser("chapters", help="print the [[recap.chapters]] block for takes/library/films.toml").set_defaults(fn=cmd_chapters)
    m = sub.add_parser("mix", help="class mix (heavy) or film soundtrack")
    m.add_argument("which", nargs="?", choices=["class", "film"], default="film")
    m.set_defaults(fn=cmd_mix)
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        a = ap.parse_args(["build"])
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
