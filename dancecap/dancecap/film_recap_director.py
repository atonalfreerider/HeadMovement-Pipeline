"""Director mode of the class-recap film (owner 2026-10-10): much more of the lesson as the phones' own video, circling through the phones;
a tight 3D zoom on the body feature / floor feature / physics the teachers call attention to; overhead views much closer.

Pure functions over plain data (like film_recap_plan, which they extend): the sentence-based beats of `plan.plan_beats` go in, a list of SHOTS
comes out (the same `Beat` objects, now carrying their own camera block):

  1. every attention call (a teaching event that names a body part, the weight / balance, the axis / path / steps, the timing) gets its own
     3D window: it starts `zoom_pre_s` before the word and holds `zoom_post_s` after it (merged when events follow each other, never longer than
     `zoom_max_s`); the window's treatment follows the events in it (plan.decide_unit): a tight feature zoom on the named joints of BOTH dancers,
     the closer physics view, the closer overhead with the T / the path, the feet, the touchdown rings
  2. what is left between the windows is free time: it is cut at the speech's own pauses into shots of about 3 to 6.4 s that ALTERNATE between a
     phone's point of view and a calm 3D orbit (a teacher asking to look - 'watch him', 'do this', 'like this' - prefers the phone);
     the phone circles: each POV shot takes the phone used longest ago among those that see both teachers well (recap_pov.choose_phones),
     never one inside its bad stretches (recap_pov.BAD), and never twice in a row
  3. if the phones carry less than `pov_target[0]` of the film, long 3D windows hand their tail to a phone (the zoom keeps the event, the phone shows
     the demonstration with the 3D skeleton on top) and calm shots turn into phone shots, until the share is in range
  4. each POV shot is framed on the teachers (recap_pov.crop_track: a zoom into the phone's frame, the view kept inside it) and either cuts in
     (`camera_tour`, a hard cut) or flies in from the previous 3D shot (`fly_to_camera`, for the longer ones)

Every decision is recorded (`treatment` of the shot, the report's `attention` table) so that direction_review.md can say, per attention call, what the
film shows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import film_recap_plan as plan
from . import recap_pov as rp

ATTENTION_TAGS = ("camera_focus", "overlay_T_axis", "overlay_weight_change", "overlay_footsteps/physics", "overlay_path/floorcraft", "overlay_timing")
WATCH_WORDS = ("look at it", "this way", "that way", "do this", "like this", "from here", "here you can", "do here", "very square", "starting here",
               "you lose it", "see so")


@dataclass
class DirCfg:
    pov_target: tuple[float, float] = (0.38, 0.48)     # the share of the film's screen time that is a phone's own video
    pov_min_s: float = 3.0
    pov_max_s: float = 6.4
    pov_part_s: float = 5.2                           # a long stretch of free time is cut into parts of about this length
    zoom_pre_s: float = 0.6
    zoom_post_s: float = 1.9
    zoom_min_s: float = 2.4
    zoom_max_s: float = 6.5
    merge_gap_s: float = 0.6
    repeat_s: float = 8.0                              # a body part called again this soon after its zoom is shown by the phone with the arrow
    calm_min_s: float = 2.6
    fly_min_s: float = 4.4
    snap_tol_s: float = 0.8
    min_event_conf: float = 0.5
    insert_pad_s: float = 0.4                          # no phone shot this close to a slow-motion insert
    max_pov_run: int = 2                               # phone shots in a row (the phones circle; a 3D shot breathes between)
    first_shot_3d_s: float = 2.6                       # the film opens on the 3D couple (the title card), then the phones


@dataclass
class PovSource:
    phones: dict[str, rp.PhoneData]
    bad: dict[str, list]
    cfg: rp.PovCfg = field(default_factory=rp.PovCfg)

    def any_phone(self, a: float, b: float) -> bool:
        return any(rp.window_quality(ph, a, b, self.bad.get(c, []), self.cfg) for c, ph in self.phones.items())


@dataclass
class Piece:
    t0: float
    t1: float
    kind: str                         # zoom | free | pov | calm
    beat: plan.Beat
    family: str = "talk"              # zoom: the treatment family; free / calm: the beat's, for a tail that ends up in 3D
    events: list[str] = field(default_factory=list)
    prefer_pov: bool = False
    no_pov: bool = False
    first_of_beat: bool = False
    why: list[str] = field(default_factory=list)
    score: float = 0.0
    tail_of: str | None = None
    feature: tuple | None = None      # a body-part zoom: (feature, dancer)

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


# ------------------------------------------------------------------------------------------------ the speech's own pauses


def cut_points(utterances: list[dict], lead_s: float, gap_s: float = 0.16) -> list[float]:
    """where a shot may change without cutting a word: the sentence starts (less the lead), the sentence ends and every pause between words"""
    cuts: set[float] = set()
    for u in utterances:
        if u["end"] <= 0:
            continue
        cuts.add(round(max(0.0, u["start"] - lead_s), 3))
        cuts.add(round(u["end"] + 0.05, 3))
        ws = [w for w in u.get("words", []) if w.get("kind") != "event" and w.get("end", 0) > 0]
        for a, b in zip(ws, ws[1:]):
            gap = b["start"] - a["end"]
            if gap >= gap_s or str(a.get("punct", "")).strip() in (",", ".", "?", "!", ";", ":"):
                cuts.add(round(0.5 * (a["end"] + b["start"]), 3))
    return sorted(cuts)


def snap(t: float, cuts: list[float], tol: float, lo: float = -1e9, hi: float = 1e9) -> float:
    """the nearest cut point within `tol` of t (and inside [lo, hi]); t itself when there is none"""
    best, bd = t, tol + 1e-9
    for c in cuts:
        if c < lo or c > hi:
            continue
        d = abs(c - t)
        if d < bd:
            best, bd = c, d
    return best


def speaker_at(utterances: list[dict], t: float, fallback: str | None = None) -> str | None:
    best = None
    for u in utterances:
        if u["start"] - 0.3 <= t <= u["end"] + 0.3:
            if best is None or abs(t - 0.5 * (u["start"] + u["end"])) < abs(t - 0.5 * (best["start"] + best["end"])):
                best = u
    return best["speaker"] if best else fallback


# ------------------------------------------------------------------------------------------------ attention calls


def is_attention(e: dict, dc: DirCfg) -> bool:
    """a teaching event the camera should answer: it carries a family tag (a body part, weight, axis, path, steps, timing)"""
    if float(e.get("conf") or 0.0) < dc.min_event_conf:
        return False
    return bool(set(e.get("tags", [])) & set(ATTENTION_TAGS)) and bool(plan.event_families(e))


def is_watch(e: dict) -> bool:
    """the teacher asks to LOOK at the demonstration ('look at it', 'do this', 'like this'): the phone's picture is the better evidence"""
    if "demonstration" not in e.get("kinds", []):
        return False
    trig = " ".join(e.get("triggers", [])).lower()
    return not (e.get("focus") or {}).get("where") or any(w in trig for w in WATCH_WORDS)


FEATURES: dict[str, dict] = {
    # feature -> the joints that frame it, the dancer they belong to by default, the partner's matching joints (kept in the frame so that both
    # dancers stay understandable), the padding round them (m) and the closest the eye may come (m). No hand close-ups: hands are raw MAMMA output.
    # A 9:16 frame is narrow: a feature is "tight" when what must stay in it is NARROW (the torso of one dancer and the partner's matching joints),
    # so the paddings are small and the span comes from the joints alone.
    "frame": {"dancer": "both", "joints": ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow"], "pad": 0.22, "min_dist": 1.8},
    "shoulders": {"dancer": "follow", "joints": ["l_shoulder", "r_shoulder", "l_collar", "r_collar"], "partner": ["l_shoulder", "r_shoulder"], "pad": 0.2, "min_dist": 1.5},
    "arm": {"dancer": "lead", "joints": ["r_shoulder", "r_elbow", "r_wrist"], "partner": ["l_shoulder", "spine3"], "pad": 0.2, "min_dist": 1.6},
    "hip": {"dancer": "follow", "joints": ["pelvis", "l_hip", "r_hip"], "partner": ["pelvis", "l_hip", "r_hip"], "pad": 0.22, "min_dist": 1.6},
    "chest": {"dancer": "follow", "joints": ["spine3", "l_shoulder", "r_shoulder", "neck"], "partner": ["spine3", "neck"], "pad": 0.2, "min_dist": 1.5},
    "upper body": {"dancer": "follow", "joints": ["spine2", "spine3", "l_shoulder", "r_shoulder"], "partner": ["spine2", "spine3", "neck"], "pad": 0.2, "min_dist": 1.5},
    "connection": {"dancer": "both", "joints": ["spine3", "l_elbow", "r_elbow"], "pad": 0.25, "min_dist": 1.8},
    "legs": {"dancer": "both", "joints": ["l_hip", "r_hip", "l_knee", "r_knee"], "pad": 0.25, "min_dist": 1.9},
    "head": {"dancer": "follow", "joints": ["head", "neck"], "partner": ["head", "neck"], "pad": 0.2, "min_dist": 1.5},
}
WHERE_FEATURE = {"frame": "frame", "shoulders": "shoulders", "shoulder": "shoulders", "arm": "arm", "hands": "frame", "hand": "frame", "hip": "hip", "chest": "chest",
                 "upper body": "upper body", "connection": "connection", "legs": "legs", "leg": "legs", "thighs": "legs", "head": "head", "neck": "head"}


def feature_of(events: list[dict]) -> tuple[str, str]:
    """(feature, dancer) a body-part window frames: the strongest event's `where` (hands -> the frame), the dancer its `who` names"""
    best, best_c = None, -1.0
    for e in events:
        where = (e.get("focus") or {}).get("where")
        f = WHERE_FEATURE.get(str(where or "").lower())
        if f and float(e.get("conf") or 0) > best_c:
            best, best_c, who = f, float(e.get("conf") or 0), (e.get("focus") or {}).get("who", "couple")
    if best is None:
        return "frame", "both"
    spec = FEATURES[best]
    dancer = {"leader": "lead", "follower": "follow"}.get(who, spec["dancer"])
    if best in ("frame", "connection", "legs"):
        dancer = "both"
    return best, dancer


def feature_camera(feature: str, dancer: str, variant: int, blend: float) -> dict:
    """the camera block of a feature zoom (Unity: FilmCamera.FeatureZoom): the feature's joints on `dancer` (or both) plus the partner's matching
    joints, framed tight in the key band of the 9:16 frame, side-on to the couple's line (alternating side), eye at the feature's height"""
    spec = FEATURES[feature]
    sign = 1 if variant % 2 == 0 else -1
    cam = {"mode": "feature_zoom", "feature": feature, "dancer": dancer, "joints": list(spec["joints"]), "pad_m": spec["pad"], "min_distance_m": spec["min_dist"],
           "azimuth_offset_deg": 28 * sign, "elevation_deg": 5, "fov_deg": 36, "blend_s": round(blend, 2)}
    if dancer != "both" and spec.get("partner"):
        cam["partner_joints"] = list(spec["partner"])
    return cam


def physics_camera(variant: int, blend: float) -> dict:
    sign = 1 if variant % 2 == 0 else -1
    return {"mode": "physics_3q", "azimuth_deg": sign * 35, "drift_deg_s": sign * 4, "distance_m": 3.6, "fit_band": True, "top_m": 1.95, "band_lo": 0.34, "band_hi": 0.74, "blend_s": round(blend, 2)}


# ------------------------------------------------------------------------------------------------ pieces


def split_beat(b: plan.Beat, ctx: "Ctx") -> list[Piece]:
    """one sentence-level beat -> zoom windows (one per attention call, its treatment from the call itself; neighbours of one treatment join) and
    the free time between them"""
    dc = ctx.dc
    evs = sorted((e for e in ctx.events if b.t0 - 1e-6 <= e["start"] < b.t1 and is_attention(e, dc)), key=lambda e: e["start"])
    wins: list[dict] = []                              # {w0, w1, fam, feature, es, why, sc, ids}
    for e in evs:
        w0 = max(b.t0, e["start"] - dc.zoom_pre_s)
        w1 = min(b.t1, e["start"] + dc.zoom_post_s)
        fam, sc, why, ids = plan.decide_unit({"id": f"{b.id or 'b'}_{e['id']}", "t0": w0, "t1": w1}, [e], ctx.act, ctx.layers, ctx.cfg, ctx.dropped, ctx.notes)
        if fam == "graph":
            fam = "talk"
        feature = feature_of([e]) if fam == "body" else None
        w = {"w0": w0, "w1": w1, "fam": fam, "feature": feature, "es": [e], "why": why, "sc": sc}
        prev = wins[-1] if wins else None
        if prev is not None and w0 - prev["w1"] < dc.merge_gap_s:
            same = prev["fam"] == fam and prev["feature"] == feature
            if same and w1 - prev["w0"] <= dc.zoom_max_s:
                prev["w1"] = max(prev["w1"], w1)
                prev["es"].append(e)
                prev["why"] += why
                prev["sc"] = max(prev["sc"], sc)
                continue
            # different treatments: the earlier one ends where the later one starts (never shorter than a zoom)
            cutp = min(prev["w1"], max(w0, prev["w0"] + dc.zoom_min_s))
            prev["w1"] = cutp
            w["w0"] = cutp
            if w["w1"] - w["w0"] < dc.zoom_min_s:
                w["w1"] = min(b.t1, w["w0"] + dc.zoom_min_s)
        wins.append(w)
    pieces: list[Piece] = []
    cur = b.t0
    for w in wins:
        w0, w1, fam, es = w["w0"], w["w1"], w["fam"], w["es"]
        if w0 - cur < 1.2:
            w0 = cur                               # no sliver of free time before the window
        if b.t1 - w1 < 1.2:
            w1 = b.t1
        if w0 > cur:
            w0 = snap(w0, ctx.cuts, dc.snap_tol_s, cur, w1 - dc.zoom_min_s)
        if w1 < b.t1:
            w1 = snap(w1, ctx.cuts, dc.snap_tol_s, w0 + dc.zoom_min_s, b.t1)
        if w1 - w0 < 0.3:
            continue
        watch = any(is_watch(e) for e in es)
        why = list(w["why"])
        if fam == "body":
            feature, dancer = feature_of(es)
            last = ctx.last_zoom.get((feature, dancer))
            if last is not None and w0 - last < dc.repeat_s and ctx.pov is not None and ctx.pov.any_phone(w0, w1):
                why.append(f"the {feature} was zoomed on {w0 - last:.1f} s ago: the phone shows it, with the arrow")
                fam = "talk"
                watch = True
            else:
                ctx.last_zoom[(feature, dancer)] = w1
        if w0 > cur + 1e-6:
            pieces.append(Piece(cur, w0, "free", b, family=b.family, why=["free time before an attention call"]))
        if fam == "talk":
            # the call has no motion family here (the teachers stand, or apart): the phone shows it when it can (prefer_pov), else a calm orbit
            pieces.append(Piece(w0, w1, "free", b, family="talk", events=[e["id"] for e in es], prefer_pov=True, why=why or ["attention call without a motion family"]))
        else:
            pieces.append(Piece(w0, w1, "zoom", b, family=fam, events=[e["id"] for e in es], score=w["sc"], why=why,
                                feature=feature_of(es) if fam == "body" else None))
        cur = w1
    if b.t1 > cur + 1e-6:
        pieces.append(Piece(cur, b.t1, "free", b, family="talk", prefer_pov=False, why=["the teachers talk"]))
    out: list[Piece] = []
    for p in pieces:
        if out and out[-1].kind == "free" and p.kind == "free":
            out[-1].t1 = p.t1
            out[-1].events += p.events
            out[-1].prefer_pov = out[-1].prefer_pov or p.prefer_pov
        elif out and out[-1].kind == "zoom" and p.kind == "zoom" and out[-1].family == p.family and out[-1].feature == p.feature and abs(out[-1].t1 - p.t0) < 1e-6 and p.t1 - out[-1].t0 <= dc.zoom_max_s:
            out[-1].t1 = p.t1
            out[-1].events += p.events
        else:
            out.append(p)
    if out:
        out[0].first_of_beat = True
    return out


def absorb_short(pieces: list[Piece], ctx: "Ctx") -> list[Piece]:
    """free time shorter than a phone shot joins the zoom before it (a longer hold) or the piece after it"""
    dc = ctx.dc
    changed = True
    while changed:
        changed = False
        for i, p in enumerate(pieces):
            if p.kind != "free" or p.dur >= dc.pov_min_s or len(pieces) == 1:
                continue
            prev = pieces[i - 1] if i > 0 and not p.first_of_beat else None
            nxt = pieces[i + 1] if i + 1 < len(pieces) else None
            if prev is not None and prev.kind == "zoom" and prev.dur + p.dur <= dc.zoom_max_s + 1.5:
                prev.t1 = p.t1
                prev.events += p.events
            elif nxt is not None and not nxt.first_of_beat:
                nxt.t0 = p.t0
                nxt.first_of_beat = nxt.first_of_beat or p.first_of_beat
                nxt.events = p.events + nxt.events
                nxt.prefer_pov = nxt.prefer_pov or p.prefer_pov
            elif prev is not None:
                prev.t1 = p.t1
                prev.events += p.events
            elif nxt is not None:
                nxt.t0 = p.t0
                nxt.first_of_beat = True
            else:
                continue
            del pieces[i]
            changed = True
            break
    return pieces


def overlaps_insert(t0: float, t1: float, inserts: list[dict], pad: float) -> bool:
    for ins in inserts:
        a = ins["core"][0] if ins["kind"] == "gap_slow" else min(ins["core"][0], ins.get("park", ins["core"][0]))
        b = ins["core"][1] if ins["kind"] == "gap_slow" else max(ins["core"][1], ins.get("park", ins["core"][1]))
        if t0 < b + pad and t1 > a - pad:
            return True
    return False


def subdivide(p: Piece, ctx: "Ctx") -> list[Piece]:
    """free time longer than a phone shot is cut at the speech's pauses into parts of about pov_part_s"""
    dc = ctx.dc
    if p.dur <= dc.pov_max_s:
        return [p]
    n = int(math.ceil(p.dur / dc.pov_part_s))
    edges = [p.t0]
    for k in range(1, n):
        target = p.t0 + k * p.dur / n
        edges.append(snap(target, ctx.cuts, dc.snap_tol_s + 0.4, edges[-1] + dc.calm_min_s, p.t1 - dc.calm_min_s))
    edges.append(p.t1)
    out = []
    for i in range(n):
        if edges[i + 1] - edges[i] < 0.5:
            continue
        start = {e["id"]: e["start"] for e in ctx.events}
        mine = [e for e in p.events if edges[i] - 0.6 <= start.get(e, edges[i]) < edges[i + 1]]
        q = Piece(edges[i], edges[i + 1], "free", p.beat, family=p.family, events=mine, prefer_pov=p.prefer_pov and bool(mine) or (p.prefer_pov and not p.events),
                  why=list(p.why), first_of_beat=p.first_of_beat and i == 0)
        out.append(q)
    return out


def pov_run(pieces: list[Piece], i: int) -> int:
    """how many phone shots in a row the piece i would be part of (counting itself as one)"""
    n = 1
    j = i - 1
    while j >= 0 and pieces[j].kind == "pov":
        n += 1
        j -= 1
    j = i + 1
    while j < len(pieces) and pieces[j].kind == "pov":
        n += 1
        j += 1
    return n


def assign(pieces: list[Piece], ctx: "Ctx") -> None:
    """free pieces -> pov or calm, alternating with whatever came before; a teacher asking to look prefers the phone; the film opens in 3D;
    never more than dc.max_pov_run phone shots in a row (the phones circle, the 3D picture comes back between them)"""
    dc = ctx.dc
    prev = "3d"
    for i, p in enumerate(pieces):
        if p.kind != "free":
            prev = "3d"
            continue
        p.no_pov = overlaps_insert(p.t0, p.t1, ctx.inserts, dc.insert_pad_s)
        can = ctx.pov is not None and not p.no_pov and p.dur >= dc.pov_min_s - 0.2 and ctx.pov.any_phone(p.t0, p.t1)
        first = p.t0 < dc.first_shot_3d_s
        if can and not first and (prev != "pov" or p.prefer_pov):
            p.kind = "pov"
            if pov_run(pieces, i) > dc.max_pov_run:
                p.kind = "calm"
        else:
            p.kind = "calm"
        prev = "pov" if p.kind == "pov" else "3d"


def fix_short(pieces: list[Piece], ctx: "Ctx") -> list[Piece]:
    """a shot shorter than a zoom's minimum joins its neighbour (the previous one's hold, or the next one's lead-in at a chapter start)"""
    dc = ctx.dc
    limit = dc.zoom_min_s - 0.2
    changed = True
    while changed:
        changed = False
        for i, p in enumerate(pieces):
            if p.dur >= limit or len(pieces) == 1:
                continue
            prev = pieces[i - 1] if i > 0 and not p.first_of_beat else None
            nxt = pieces[i + 1] if i + 1 < len(pieces) else None
            if prev is not None and prev.kind == "pov" and nxt is not None and not nxt.first_of_beat and nxt.kind != "pov":
                prev = None                                  # do not lengthen a phone shot (and join two phone shots round it): the 3D neighbour takes it
            if prev is not None and (nxt is None or prev.kind == p.kind or nxt.first_of_beat or prev.dur <= nxt.dur):
                prev.t1 = p.t1
                prev.events += p.events
            elif nxt is not None and not nxt.first_of_beat:
                nxt.t0 = p.t0
                nxt.events = p.events + nxt.events
                if p.first_of_beat:
                    nxt.first_of_beat = True
                    nxt.beat = p.beat
            else:
                continue
            del pieces[i]
            changed = True
            break
    return pieces


def limit_runs(pieces: list[Piece], ctx: "Ctx") -> None:
    """a run of phone shots longer than max_pov_run (the balance step's tails and merges can make one) gets a 3D shot in its middle"""
    n = ctx.dc.max_pov_run
    i = 0
    while i < len(pieces):
        if pieces[i].kind != "pov":
            i += 1
            continue
        j = i
        while j < len(pieces) and pieces[j].kind == "pov":
            j += 1
        k = i + n
        while k < j:
            pieces[k].kind = "calm"
            pieces[k].no_pov = True
            k += n + 1
        i = j


def answer_calls_in_calm_shots(pieces: list[Piece], ctx: "Ctx") -> None:
    """a call that ended up in a calm 3D shot (no motion family, and the phones' turn was not there) still gets its own treatment: the body part
    named is zoomed on, the weight / balance shown in the physics view, steps / legs as the legs (the teachers stand: the shape of the posture is the point)"""
    byid = {e["id"]: e for e in ctx.events}
    for p in pieces:
        if p.kind != "calm":
            continue
        evs = [byid[i] for i in p.events if i in byid and is_attention(byid[i], ctx.dc)]
        if not evs:
            continue
        fams = []
        for e in evs:
            fs = [f for f in plan.event_families(e) if f[0] in ("weight", "body", "feet")]
            if fs:
                fams.append(max(fs, key=lambda x: x[1])[0])
        if not fams:
            continue
        fam = max(sorted(set(fams)), key=fams.count)
        if fam == "weight":
            p.kind, p.family = "zoom", "weight"
        elif fam in ("body", "feet"):
            feat = feature_of(evs)
            if fam == "feet" and not any((e.get("focus") or {}).get("where") in WHERE_FEATURE for e in evs):
                feat = ("legs", "both")
            p.kind, p.family, p.feature = "zoom", "body", feat
        else:
            continue
        p.why.append("a call with no motion family and no phone turn: shown as a posture zoom")


def pov_share(pieces: list[Piece]) -> float:
    tot = sum(p.dur for p in pieces)
    return sum(p.dur for p in pieces if p.kind == "pov") / tot if tot > 0 else 0.0


def balance(pieces: list[Piece], ctx: "Ctx") -> list[Piece]:
    """raise the phones' share to the target: tails of long zooms, then calm shots between two non-phone shots, then any calm shot"""
    dc = ctx.dc
    lo, hi = dc.pov_target
    if ctx.pov is None:
        return pieces
    guard = 0
    while pov_share(pieces) < lo and guard < 200:
        guard += 1
        acted = False
        # 1. a zoom that is long enough hands its tail to a phone (the zoom keeps the call; the phone shows what follows)
        for i, p in enumerate(pieces):
            if p.kind != "zoom" or p.dur < 6.0:
                continue
            ev_t = [e["start"] for e in ctx.events if e["id"] in p.events]
            last_ev = max(ev_t) if ev_t else p.t0
            cut = snap(max(last_ev + 2.4, p.t0 + 3.2), ctx.cuts, dc.snap_tol_s, p.t0 + 3.0, p.t1 - dc.pov_min_s)
            if p.t1 - cut < dc.pov_min_s or cut - p.t0 < 2.8:
                continue
            if overlaps_insert(cut, p.t1, ctx.inserts, dc.insert_pad_s) or not ctx.pov.any_phone(cut, p.t1):
                continue
            nxt = pieces[i + 1] if i + 1 < len(pieces) else None
            if nxt is not None and nxt.kind == "pov":
                continue
            tail = Piece(cut, p.t1, "pov", p.beat, family=p.family, events=[], why=[f"tail of the {p.family} zoom {p.t0:.1f}-{p.t1:.1f}: the phone shows the demonstration"],
                         tail_of=p.family)
            p.t1 = cut
            pieces.insert(i + 1, tail)
            acted = True
            break
        if acted:
            continue
        # 2. a calm 3D shot whose neighbours are not phone shots
        for i, p in enumerate(pieces):
            if p.kind != "calm" or p.no_pov or p.t0 < dc.first_shot_3d_s or p.dur < dc.pov_min_s - 0.2:
                continue
            a = pieces[i - 1].kind if i > 0 else "3d"
            b = pieces[i + 1].kind if i + 1 < len(pieces) else "3d"
            if "pov" in (a, b) or not ctx.pov.any_phone(p.t0, p.t1):
                continue
            p.kind = "pov"
            acted = True
            break
        if acted:
            continue
        # 3. any calm shot
        for i, p in enumerate(pieces):
            if p.kind == "calm" and not p.no_pov and p.t0 >= dc.first_shot_3d_s and p.dur >= dc.pov_min_s - 0.2 and ctx.pov.any_phone(p.t0, p.t1):
                p.kind = "pov"
                if pov_run(pieces, i) > dc.max_pov_run:
                    p.kind = "calm"
                    continue
                acted = True
                break
        if not acted:
            break
    return pieces


@dataclass
class Ctx:
    utterances: list[dict]
    events: list[dict]
    chapters: list[dict]
    act: plan.Activity
    layers: plan.Layers
    roles: dict
    duration: float
    cfg: plan.Cfg
    dc: DirCfg
    pov: PovSource | None
    inserts: list[dict]
    cuts: list[float]
    dropped: list[dict]
    notes: list[str]
    last_zoom: dict = field(default_factory=dict)         # (feature, dancer) -> end of its last zoom


# ------------------------------------------------------------------------------------------------ shots


def to_beats(pieces: list[Piece], ctx: "Ctx") -> list[plan.Beat]:
    out: list[plan.Beat] = []
    for p in pieces:
        b = p.beat
        fam = "pov" if p.kind == "pov" else (p.family if p.kind == "zoom" else ("talk" if p.kind == "calm" else p.family))
        if p.kind == "calm":
            fam = "talk"
        spk = speaker_at(ctx.utterances, 0.5 * (p.t0 + p.t1), b.speaker)
        evs = [e["id"] for e in ctx.events if p.t0 - 1e-6 <= e["start"] < p.t1 and (e["id"] in p.events or p.kind == "pov" or is_attention(e, ctx.dc))]
        nb = plan.Beat(id="", t0=p.t0, t1=p.t1, family=fam, chapter=b.chapter, speaker=spk, score=p.score, why=list(p.why), events=sorted(set(evs)),
                       chapter_start=bool(b.chapter_start and p.first_of_beat), utterances=list(b.utterances), activity=ctx.act.window(p.t0, p.t1, ctx.cfg),
                       treatment={"kind": p.kind, "family": p.family, "tail_of": p.tail_of, "feature": p.feature[0] if p.feature else None,
                                  "dancer": p.feature[1] if p.feature else None})
        out.append(nb)
    return out


def pov_camera(phone: dict, crop: dict, b: plan.Beat, fly: bool, prev_3d: bool) -> dict:
    cam = phone["camera"]
    if fly:
        fly_s, fin, fout, pull = 0.8, 0.3, 0.3, 0.1
        hold = max(1.0, b.dur - fly_s - fin - fout - pull)
        return {"mode": "fly_to_camera", "camera": cam, "fly_s": fly_s, "fade_in_s": fin, "hold_s": round(hold, 2), "fade_out_s": fout, "pull_back_s": pull,
                "composite": "3d_over_video", "skeletons_on_top": True, "scene_geometry": "all", "crop": crop}
    return {"mode": "camera_tour", "cameras": [{"camera": cam, "dance_t": [round(b.t0, 3), round(b.t1, 3)]}], "cut": "hard", "blend_s": 0.0,
            "composite": "3d_over_video", "skeletons_on_top": True, "scene_geometry": "all", "crop": crop}


def build_cameras(beats: list[plan.Beat], ctx: "Ctx", report: dict) -> None:
    """a camera block for every shot (after plan.assign_variation gave the families their variants); the phone shots get their phone and crop"""
    cfg, dc = ctx.cfg, ctx.dc
    blend = round(cfg.blend_s, 2)
    pov_idx = [i for i, b in enumerate(beats) if b.family == "pov"]
    wins = []
    for i in pov_idx:
        b = beats[i]
        # a 3D arrow on the video (a body part called inside the shot) needs a phone whose pose is accurate: the directly localised ones
        arrow = any(e["start"] >= b.t0 and e["start"] < b.t1 and "camera_focus" in e.get("tags", []) and (e.get("focus") or {}).get("where") in plan.BODY_ARROWS
                    and float(e.get("conf") or 0) >= ctx.cfg.arrow_min_conf for e in ctx.events)
        wins.append((b.t0, b.t1, rp.MAIN_PHONES if arrow else None))
    picks = rp.choose_phones(wins, ctx.pov.phones, ctx.pov.bad, ctx.pov.cfg) if pov_idx and ctx.pov else []
    eligible = 0
    shots_pov = []
    for n, i in enumerate(pov_idx):
        b = beats[i]
        pick = picks[n]
        if pick is None:
            b.family = "talk"
            b.treatment = {"kind": "calm", "family": "talk", "note": "no phone could show this stretch"}
            continue
        ph = ctx.pov.phones[pick["camera"]]
        crop = rp.crop_track(ph, b.t0, b.t1, ctx.pov.cfg)
        prev_3d = i == 0 or beats[i - 1].family != "pov"
        can_fly = prev_3d and b.dur >= dc.fly_min_s
        fly = can_fly and eligible % 2 == 0           # every other long shot after a 3D shot flies in; the rest cut
        eligible += 1 if can_fly else 0
        b.camera = pov_camera(pick, crop, b, fly, prev_3d)
        b.treatment = {"kind": "pov", "phone": pick["camera"], "cover": pick["cover"], "height": pick["height"], "fly": fly, "alternatives": pick["alternatives"],
                       "zoom_median": crop["zoom_median"], "inside": crop["inside"], "family": b.treatment.get("family"), "tail_of": b.treatment.get("tail_of")}
        shots_pov.append(b.treatment)
    report["pov_shots"] = shots_pov
    for i, b in enumerate(beats):
        if b.family == "pov":
            continue
        after_pov = i > 0 and beats[i - 1].family == "pov"
        bl = 0.0 if after_pov else blend
        fam = b.family
        apart = bool(b.activity.get("apart")) if b.activity.get("known") else False
        if fam == "body":
            feature, dancer = b.treatment.get("feature"), b.treatment.get("dancer")
            if not feature:
                feature, dancer = feature_of([e for e in ctx.events if e["id"] in set(b.events)])
            b.camera = feature_camera(feature, dancer, b.variant, bl)
            b.treatment.update({"feature": feature, "dancer": dancer})
        elif fam == "weight":
            b.camera = physics_camera(b.variant, bl)
        else:
            fw = 0.35 if fam == "talk" and b.focus in ("lead", "follow") else 0.0
            b.camera = plan.camera_for(fam, variant=b.variant, focus=b.focus, focus_weight=fw, apart=apart, cfg=cfg)
            b.camera["blend_s"] = bl


def direct(beats: list[plan.Beat], ctx: "Ctx") -> tuple[list[plan.Beat], dict]:
    """sentence-level beats -> director-mode shots (see the module docstring) and a report {pov_share, shots, ...}"""
    pieces: list[Piece] = []
    for b in beats:
        pieces += split_beat(b, ctx)
    pieces = absorb_short(pieces, ctx)
    sub: list[Piece] = []
    for p in pieces:
        sub += subdivide(p, ctx) if p.kind == "free" else [p]
    pieces = sub
    assign(pieces, ctx)
    pieces = balance(pieces, ctx)
    pieces = fix_short(pieces, ctx)
    limit_runs(pieces, ctx)
    answer_calls_in_calm_shots(pieces, ctx)
    # chapter starts stay shot starts
    shots = to_beats(pieces, ctx)
    plan.assign_variation(shots, ctx.chapters, ctx.act, ctx.roles, ctx.cfg)
    report: dict = {}
    build_cameras(shots, ctx, report)
    tot = sum(b.dur for b in shots)
    pov_s = sum(b.dur for b in shots if b.family == "pov")
    per_phone: dict[str, float] = {}
    for b in shots:
        if b.family == "pov":
            c = b.treatment["phone"]
            per_phone[c] = per_phone.get(c, 0.0) + b.dur
    report.update({"pov_share": round(pov_s / tot, 4) if tot else 0.0, "pov_seconds": round(pov_s, 2), "pov_shots_n": sum(1 for b in shots if b.family == "pov"),
                   "pov_seconds_per_phone": {c: round(v, 2) for c, v in sorted(per_phone.items())},
                   "pov_share_per_phone": {c: round(v / tot, 4) for c, v in sorted(per_phone.items())}, "shots": len(shots)})
    return shots, report


def attention_table(shots: list[plan.Beat], events: list[dict], dc: DirCfg) -> list[dict]:
    """per attention call: what the film shows (the shot, its treatment, the feature / phone); the review table prints it"""
    rows = []
    for e in events:
        tags = set(e.get("tags", [])) & set(ATTENTION_TAGS)
        if not tags:
            continue
        s = next((b for b in shots if b.t0 - 1e-6 <= e["start"] < b.t1 + 1e-6), shots[-1])
        t = s.treatment
        if s.family == "pov":
            shown = f"phone {t.get('phone')} point of view, 3D skeleton on top" + (" (flown into)" if t.get("fly") else "")
        elif s.family == "body" and s.camera and s.camera.get("mode") == "feature_zoom":
            shown = f"3D zoom on {t.get('feature')} ({t.get('dancer')})"
        elif s.family == "weight":
            shown = "physics view, closer, centre-of-mass markers"
        elif s.family in ("axis", "path"):
            shown = "overhead, close, " + ("the leader's T" if s.family == "axis" else "the floor path")
        elif s.family == "feet":
            shown = "feet close-up, footprints"
        elif s.family == "timing":
            shown = "medium close, touchdown rings"
        else:
            shown = "calm 3D orbit"
        rows.append({"id": e["id"], "t": round(e["start"], 2), "speaker": e.get("speaker"), "words": e.get("triggers", []), "tags": sorted(tags),
                     "conf": e.get("conf"), "needs_check": bool(e.get("needs_check")), "shot": s.id, "film_family": s.family, "shown": shown,
                     "deliberate": bool(is_attention(e, dc) and (s.family != "talk" or s.family == "pov"))})
    return rows
