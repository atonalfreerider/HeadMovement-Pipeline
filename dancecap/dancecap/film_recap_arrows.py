"""Call-out arrows of the class-recap film (owner 2026-10-11: "more arrows"): one labelled arrow for every attention call that has something to point at.

Pure functions over plain data (the events, the shot, the activity, the physics contacts), so every rule is testable. The film recap's older
`plan.make_annotations` drew an arrow only for the few body-part words that had a short label and only one at a time; the owner wants an arrow for
every attention call (a body part, a foot, the weight / balance, the axis, the path, the timing), anchored on the body feature or the floor feature
the teacher is talking about, in the phone's point of view as well as in the 3D shots.

An attention call becomes (first match wins):
  body part (where = frame, shoulders, arm, hip, chest, ...)      arrow3d on the joint (plan.BODY_ARROWS), labelled with the teacher's word
  feet / footwork / steps                                         arrow3d on the named foot ("right foot"), else on the foot that carries the weight (stance_foot)
  weight, balance, counterbalance, heavy, push, fall, engaged ...  arrow3d on the dancer's centre of mass, on the couple's for 'couple'
  axis, exit, rotation, pivot                                     arrow3d on the leader's T axis (the floorcraft T alive at that time)
  path, direction, space, forward, to the right                   vector_arrow: the direction the couple travels (velocity of the centre of mass)
  timing, momentum, speed, energy, at the same time               vector_arrow: velocity of the dancer's (or the couple's) centre of mass
  'here' (starting here, from here, do here)                       floor_ring at the couple
  anything else ('look at it', 'like this')                       nothing to point at: no arrow (listed)
and in the physics view, on top: a vector_arrow of each dancer's velocity (momentum), the net push (force, "est."), and the weight transfer from one
foot to the other where the physics contact changes.

Viewer side (head-movement, Assets/Film/Annotation3D.cs): `vector_arrow` = {target, vector: velocity | force, scale_px?} or {from, to, vector: between}.
Everything here is estimated from motion; the arrows carry the teacher's own word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from . import film_recap_plan as plan

# ------------------------------------------------------------------------------------------------ labels

TIDY = {"shoulder blades": "shoulder blades", "open chest": "open chest", "upper body": "upper body", "at the same time": "at the same time",
        "side to side": "side to side", "to the right": "to the right", "too far": "too far", "in front of her": "in front of her"}
WORD_LABEL = {
    "weights": "weight", "counter": "counterbalance", "fall": "falling", "falls": "falling", "pivoting": "pivot", "rotate": "rotation",
    "walk around": "walk around", "hands": "hands", "hand": "hands", "thighs": "thighs", "leg": "leg", "legs": "legs", "foot": "foot",
    "slowing": "slow down", "stop": "stop", "shift": "weight shift", "connected": "connection",
}
HERE_WORDS = ("here", "from here", "starting here", "do here")


AXIS_WORDS = ("axis", "second exit", "first exit", "exit", "rotation", "rotate", "pivoting", "pivot")
FEET_WORDS = ("right foot", "left foot", "walk around", "footwork", "step", "lunge", "foot")
WEIGHT_WORDS = ("counterbalance", "weights", "weight", "balance", "heavy", "suspension", "shift", "push", "fall", "engaged", "energy")
GENERIC = {"again", "here", "this", "that"}


def label_of(e: dict, prefer: tuple[str, ...] = ()) -> str | None:
    """the call-out's text: the teacher's own word(s) (docs/RECAP_FILM.md): the event's trigger that names what the arrow points at (`prefer`, in order),
    else the triggers, tidied; None when the event has none worth showing"""
    trig = [str(t).lower().strip() for t in e.get("triggers", []) if str(t).strip()]
    trig = [t for t in trig if t not in GENERIC] or trig
    if not trig:
        return None
    # one trigger inside another ('exit' in 'second exit'): the longer one
    trig = [t for t in trig if not any(t != o and t in o for o in trig)]
    for w in prefer:
        if w in trig:
            return WORD_LABEL.get(w, TIDY.get(w, w))
    key = tuple(trig)
    if key in plan.LABELS:
        return plan.LABELS[key]
    for t in trig:
        if (t,) in plan.LABELS:
            return plan.LABELS[(t,)]
    if "right foot" in trig or "left foot" in trig:
        return "right foot" if "right foot" in trig else "left foot"
    for t in trig:
        if t in plan.WEIGHT_LABELS:
            return plan.WEIGHT_LABELS[t]
    for t in trig:
        if t in WORD_LABEL:
            return WORD_LABEL[t]
        if t in TIDY:
            return TIDY[t]
    t = " ".join(trig[:2]) if len(trig) > 1 and len(" ".join(trig[:2])) <= 18 else trig[0]
    t = re.sub(r"\s+", " ", t).strip()
    return t[:22] if t else None


# ------------------------------------------------------------------------------------------------ what an event points at


@dataclass
class Spec:
    kind: str                       # arrow3d | vector_arrow | floor_ring
    target: dict
    text: str
    why: str
    vector: str | None = None
    needs_motion: bool = False      # a velocity arrow: only drawn while the couple moves
    dancer: str | None = None


def _who(e: dict) -> str:
    return (e.get("focus") or {}).get("who", "couple")


def _dancer(e: dict, default: str = "lead") -> str:
    return "follow" if _who(e) == "follower" else "lead" if _who(e) in ("leader",) else default


def t_axis_at(t_axes: list[dict] | None, t: float) -> int | None:
    """the id of the leader's T alive at master second t (else the nearest within 3 s), None without floorcraft data"""
    if not t_axes:
        return None
    best, bd = None, 1e9
    for a in t_axes:
        a0, a1 = float(a.get("t0", 0)), float(a.get("t1", 0))
        d = 0.0 if a0 <= t <= a1 else min(abs(t - a0), abs(t - a1))
        if d < bd:
            best, bd = a.get("id"), d
    return int(best) if best is not None and bd <= 3.0 else None


def spec_for_event(e: dict, ctx: dict) -> Spec | None:
    """what the arrow of an attention call points at, or None (nothing to point at / data absent). ctx: {t_axes, layers (plan.Layers)}"""
    tags = set(e.get("tags", []))
    kinds = set(e.get("kinds", []))
    where = (e.get("focus") or {}).get("where")
    who = _who(e)
    trig = [str(t).lower() for t in e.get("triggers", [])]
    layers = ctx.get("layers")
    focus_words = AXIS_WORDS if (tags & {"overlay_T_axis"} and where not in plan.BODY_ARROWS and where not in plan.FEET_WORDS_SET) else         FEET_WORDS if (where in plan.FEET_WORDS_SET) else WEIGHT_WORDS if ("overlay_weight_change" in tags or e.get("physics_call")) else ()
    text = label_of(e, focus_words)
    has_phys = layers is None or layers.any("physics")
    if not text:
        return None
    # 1. a body part
    if where in plan.BODY_ARROWS and where not in plan.FEET_WHERE:
        dancer, joint = plan.BODY_ARROWS[where].get(who, plan.BODY_ARROWS[where]["*"])
        return Spec("arrow3d", {"type": "joint", "dancer": dancer, "joint": joint}, text, f"body part '{where}'", dancer=dancer)
    # 2. the feet
    if where in plan.FEET_WHERE or ("footwork" in kinds and "overlay_footsteps/physics" in tags and not (tags & {"overlay_weight_change"})):
        dancer = "follow" if who == "follower" else "lead"
        side = "right" if "right foot" in trig or "right" in trig else "left" if "left foot" in trig or "left" in trig else None
        if side:
            return Spec("arrow3d", {"type": "foot", "dancer": dancer, "side": side}, text, f"the {side} foot", dancer=dancer)
        return Spec("arrow3d", {"type": "stance_foot", "dancer": dancer}, text, "the foot that carries the weight", dancer=dancer)
    # 3. the weight / balance / counterbalance / push / fall / engaged
    if "overlay_weight_change" in tags or e.get("physics_call"):
        if not has_phys:
            return None
        if set(trig) & {"push", "pushes", "pushing", "pull", "pulls", "pulling", "force", "pressure"}:
            d = _dancer(e, "lead")
            return Spec("vector_arrow", {"type": "com", "dancer": d}, f"{text} (est.)", f"the net push on the {d} (physics stage)", vector="force", dancer=d)
        if who in ("leader", "follower"):
            d = "lead" if who == "leader" else "follow"
            return Spec("arrow3d", {"type": "com", "dancer": d}, text, f"the {d}'s centre of mass", dancer=d)
        return Spec("arrow3d", {"type": "couple_com"}, text, "the couple's centre of mass")
    # 4. the axis / exit / rotation
    if "overlay_T_axis" in tags:
        tid = t_axis_at(ctx.get("t_axes"), float(e["start"]))
        if tid is None:
            return None
        return Spec("arrow3d", {"type": "leader_t", "t_axis": tid}, text, "the leader's T axis")
    # 5. the path / the direction of travel
    if tags & {"overlay_path/floorcraft"} or "direction" in kinds or "path" in kinds:
        return Spec("vector_arrow", {"type": "couple_com"}, text, "the direction the couple travels", vector="velocity", needs_motion=True)
    # 6. timing / momentum / speed
    if "overlay_timing" in tags:
        if who in ("leader", "follower"):
            d = "lead" if who == "leader" else "follow"
            return Spec("vector_arrow", {"type": "com", "dancer": d}, text, f"the {d}'s speed", vector="velocity", needs_motion=True, dancer=d)
        return Spec("vector_arrow", {"type": "couple_com"}, text, "the couple's speed and direction", vector="velocity", needs_motion=True)
    # 7. 'here'
    if any(w in " ".join(trig) for w in HERE_WORDS):
        return Spec("floor_ring", {"type": "couple_com"}, text or "here", "the place on the floor")
    return None


def scan_reason(e: dict, ctx: dict) -> str:
    """why an attention call has no arrow (for the report)"""
    if e.get("needs_check"):
        return "the transcript is unsure of this word (flagged)"
    if not label_of(e):
        return "no word to label the arrow with"
    tags = set(e.get("tags", []))
    if "overlay_T_axis" in tags and not (tags & {"camera_focus", "overlay_weight_change"}) and t_axis_at(ctx.get("t_axes"), float(e["start"])) is None:
        return "no leader's T axis alive at that time (floorcraft)"
    return "nothing to point at (a demonstration cue without a body part, a weight or a place)"


def com_speed(ctx: dict, who: str | None, t0: float, t1: float) -> float | None:
    """median horizontal speed (m/s) of a dancer's centre of mass ('lead' | 'follow') or of the couple's (None / 'couple': mass-weighted) over [t0, t1] from
    the physics stage (ctx['vcom'] = {t: array, lead / follow: (T, 3) velocities, mass_lead / mass_follow}); None when the capture has no physics data"""
    v = ctx.get("vcom")
    if not v:
        return None
    t = v["t"]
    i0, i1 = int(np.searchsorted(t, t0)), int(np.searchsorted(t, t1))
    if i1 <= i0:
        return None
    if who in ("lead", "follow"):
        vel = v[who][i0:i1]
    else:
        ml, mf = v["mass_lead"], v["mass_follow"]
        vel = (ml * v["lead"][i0:i1] + mf * v["follow"][i0:i1]) / (ml + mf)
    return float(np.median(np.hypot(vel[:, 0], vel[:, 2])))


# ------------------------------------------------------------------------------------------------ windows


def window_for(e: dict, t0: float, t1: float, cfg: plan.Cfg, text: str) -> tuple[float, float] | None:
    """[a, b] master seconds inside the shot [t0, t1]: from 0.35 s before the word, as long as the label needs (1.4 - 2.4 s), clipped to the shot"""
    a = max(t0 + 0.05, float(e["start"]) - 0.35)
    ln = min(cfg.arrow_max_s, max(cfg.arrow_min_s + 0.1, 1.1 + 0.055 * len(text)))
    b = min(t1 - 0.05, a + ln)
    if b - a < 1.1:
        # the call comes just before the cut: the arrow appears a little BEFORE the word instead (never before the shot)
        a = max(t0 + 0.05, b - 1.3)
    if b - a < 0.8:
        return None
    return round(a, 3), round(b, 3)


def limit_overlap(items: list[dict], cfg: plan.Cfg) -> tuple[list[dict], list[dict]]:
    """at most cfg.max_arrows_at_once arrows at a time: a later arrow shortens the earlier ones that would make a third (never below
    arrow_min_s); where it cannot, the later arrow is dropped. Returns (kept, dropped), both sorted by start."""
    kept: list[dict] = []
    dropped: list[dict] = []
    for it in sorted(items, key=lambda x: x["dance_t"][0]):
        a = it["dance_t"][0]
        active = [k for k in kept if k["dance_t"][1] > a + 1e-6]
        while len(active) >= cfg.max_arrows_at_once:
            victim = min(active, key=lambda k: k["dance_t"][0])
            if a - victim["dance_t"][0] >= cfg.arrow_min_s:
                victim["dance_t"][1] = round(a, 3)
                active.remove(victim)
            else:
                break
        if len(active) >= cfg.max_arrows_at_once:
            dropped.append(it)
            continue
        kept.append(it)
    kept.sort(key=lambda x: x["dance_t"][0])
    return kept, dropped


# ------------------------------------------------------------------------------------------------ the physics view's own arrows


def weight_transfers(contact: list[str] | None, hz: float = 30.0, hold_s: float = 0.35) -> list[dict]:
    """foot changes of one dancer from the physics `contact` series ("L" "R" "LR" ""): [{t, frm, to}] where the weight stood on one foot for at
    least hold_s, then on the other for at least hold_s (a double support between is allowed)"""
    if not contact:
        return []
    n = len(contact)
    runs: list[tuple[str, int, int]] = []
    i = 0
    while i < n:
        c = contact[i]
        j = i
        while j < n and contact[j] == c:
            j += 1
        runs.append((c, i, j))
        i = j
    solid = [(c, a, b) for c, a, b in runs if c in ("L", "R") and (b - a) / hz >= hold_s]
    out = []
    for (c0, a0, b0), (c1, a1, b1) in zip(solid, solid[1:]):
        if c0 != c1 and (a1 - b0) / hz <= 1.0:
            out.append({"t": round(0.5 * (b0 + a1) / hz, 3), "frm": c0, "to": c1})
    return out


def physics_extras(beat: plan.Beat, transfers: dict[str, list[dict]], cfg: plan.Cfg, state: dict, existing: list[dict] | None = None,
                   ctx: dict | None = None) -> list[dict]:
    """the physics view's own arrows: the velocity of each centre of mass the whole view long (unlabelled, the first view's leader arrow says
    "weight heading (est.)"), and at most two weight transfers (foot -> foot) in it"""
    out: list[dict] = []
    a0, a1 = round(beat.t0 + 0.5, 3), round(beat.t1 - 0.2, 3)
    if a1 - a0 >= 1.5:
        for k, d in enumerate(("lead", "follow")):
            it = {"id": f"{beat.id}_v{k}", "kind": "vector_arrow", "target": {"type": "com", "dancer": d}, "vector": "velocity", "dance_t": [a0, a1],
                  "needs_motion": True, "continuous": True}
            sp_ = com_speed(ctx or {}, d, beat.t0, beat.t1)
            if k == 0 and beat.t0 - state.get("velocity_label_t", -1e9) > 90.0 and (sp_ is None or sp_ >= 0.15):
                # the first physics view (and one every 90 s) says what the unlabelled arrows are, on the leader's: it sits where the label has room
                it["text"] = "weight heading (est.)"
                state["velocity_label_t"] = beat.t0
            out.append(it)
    n = 0
    for d, ts in transfers.items():
        for tr in ts:
            if beat.t0 + 0.4 <= tr["t"] <= beat.t1 - 0.5 and n < 2:
                a = max(beat.t0 + 0.05, tr["t"] - 0.6)
                b = min(beat.t1 - 0.05, tr["t"] + 1.2)
                if b - a < 0.9:
                    continue
                live = [x for x in (existing or []) + [o for o in out if not o.get("continuous")] if x["dance_t"][0] < b and x["dance_t"][1] > a]
                if any(sum(1 for y in live if y["dance_t"][0] <= t < y["dance_t"][1]) >= cfg.max_arrows_at_once for t in np.linspace(a, b, 8)):
                    continue                                             # the call-outs of the teachers' words have the room
                side = {"L": "left", "R": "right"}
                who = "his" if d == "lead" else "her"
                out.append({"id": f"{beat.id}_t{n}", "kind": "vector_arrow", "vector": "between",
                            "from": {"type": "foot", "dancer": d, "side": side[tr["frm"]]}, "to": {"type": "foot", "dancer": d, "side": side[tr["to"]]},
                            "text": f"{who} weight to the {side[tr['to']]} foot", "dance_t": [round(a, 3), round(b, 3)]})
                n += 1
    return out


# ------------------------------------------------------------------------------------------------ one shot


def make_arrows(beat: plan.Beat, events: list[dict], cfg: plan.Cfg, ctx: dict, act: plan.Activity | None, state: dict) -> tuple[list[dict], list[dict]]:
    """(annotations, skipped) of one shot: one arrow per attention call with something to point at (+ the physics view's extras). Annotation dicts carry
    `dance_t` (master seconds), as plan.make_annotations's do; skipped = [{id, t, words, why}]."""
    items: list[dict] = []
    skipped: list[dict] = []
    ids = set(beat.events)
    last: dict[tuple, dict] = state.setdefault("last_arrow", {})
    n = 0
    for e in sorted((x for x in events if x["id"] in ids or (beat.t0 <= x["start"] < beat.t1 and x.get("physics_call"))), key=lambda x: x["start"]):
        if not (beat.t0 <= float(e["start"]) < beat.t1):
            continue
        tags = set(e.get("tags", []))
        if not (tags & {"camera_focus", "overlay_T_axis", "overlay_weight_change", "overlay_footsteps/physics", "overlay_path/floorcraft", "overlay_timing"}):
            continue
        if float(e.get("conf") or 0.0) < cfg.arrow_min_conf:
            continue
        if e.get("needs_check"):
            skipped.append({"id": e["id"], "t": round(float(e["start"]), 2), "words": e.get("triggers", []), "why": scan_reason(e, ctx)})
            continue
        sp = spec_for_event(e, ctx)
        if sp is None:
            skipped.append({"id": e["id"], "t": round(float(e["start"]), 2), "words": e.get("triggers", []), "why": scan_reason(e, ctx)})
            continue
        w = window_for(e, beat.t0, beat.t1, cfg, sp.text)
        pk = last.get((sp.kind, sp.text, repr(sp.target)))
        if pk is not None and pk.get("_beat") == beat.id and float(e["start"]) - pk["dance_t"][1] < cfg.arrow_same_label_s and pk["dance_t"][1] >= float(e["start"]) - 0.4:
            pk.setdefault("calls", []).append(e["id"])                          # the same word on the same thing, still shown: one arrow answers both calls
            continue
        if w is None:
            skipped.append({"id": e["id"], "t": round(float(e["start"]), 2), "words": e.get("triggers", []), "why": "the call falls in the last moment of the shot"})
            continue
        if sp.needs_motion:
            cs = com_speed(ctx, sp.dancer, w[0], w[1])
            if cs is None and act is not None and act.known:
                wa = act.window(w[0], w[1] + 0.5, cfg)
                cs = wa["speed_med"] if wa.get("known") else None
            if cs is not None and cs < cfg.vector_min_speed:
                # standing: a velocity arrow would point nowhere; the body's centre of mass is still a thing to point at
                tgt = sp.target if sp.target.get("type") in ("com", "couple_com") else {"type": "couple_com"}
                sp = Spec("arrow3d", tgt, sp.text, "the centre of mass (standing: no direction of travel)")
        key = (sp.kind, sp.text, repr(sp.target))
        prev = last.get(key)
        if prev is not None and w[0] - prev["dance_t"][1] < cfg.arrow_same_label_s and prev.get("_beat") == beat.id:
            prev["dance_t"][1] = round(max(prev["dance_t"][1], w[1]), 3)                 # the same word on the same thing again: one longer arrow
            prev.setdefault("calls", []).append(e["id"])
            continue
        it = {"id": f"{beat.id}_ar{n}", "kind": sp.kind, "dance_t": [w[0], w[1]], "text": sp.text, "calls": [e["id"]], "why": sp.why, "_beat": beat.id}
        if sp.kind == "arrow3d":
            it["target"] = sp.target
        elif sp.kind == "vector_arrow":
            it["target"] = sp.target
            it["vector"] = sp.vector
            it["needs_motion"] = sp.needs_motion
        else:
            it["target"] = sp.target
        items.append(it)
        last[key] = it
        n += 1
    kept, dropped = limit_overlap(items, cfg)
    for d in dropped:
        skipped.append({"id": d["calls"][0], "t": d["dance_t"][0], "words": [d["text"]], "why": f"more than {cfg.max_arrows_at_once} arrows at once"})
    for k in kept:
        k.pop("_beat", None)
    return kept, skipped
