"""v3 of the class-recap direction (owner 2026-10-21): the counterbalance pivot for "back to the axis", the physics layer on every
demonstration and every connected shot, and NO jump cut anywhere (every change of view is a continuous ground-level flight; a phone's point
of view is entered and left through the phone's frustum with its 2D video surface standing in the 3D world).

    python -m dancecap.film_recap_flow build      # -> work/larissa_kadu_recap/v3/direction/ (direction.json + review; film/direction.json is NOT touched)
    python -m dancecap.film_recap_flow check [--direction PATH]
    python -m dancecap.film_recap_flow sheet      # the camera-flow sheet (azimuth / height / transitions over the film) for review
    python -m dancecap.film_recap_flow build --mix  # ... and the film soundtrack into v3/direction/audio/ (sample-identical to film/'s: the same time map)
    python -m dancecap.film_recap_flow clips      # UNITY STAGE ONLY: the phones' video for the frustum surfaces (writes the capture's cameras/)

v4.0 (owner 2026-10-22: camera moves / 3, phone holds x 3, slow orbits, smooth glides + zooms + fades, faint frustums of every phone):
    python -m dancecap.film_recap_flow build --version v4.0 [--out DIR]   # -> work/larissa_kadu_recap/v4/direction/v4.0/ (as built 2026-10-22)
v4.1 = v4 (owner 2026-10-22 after the v3 films: "hold longer on POVs and slowly orbit gaussian splats", the physics legend to the top, Jev):
    python -m dancecap.film_recap_flow build --version v4 [--plan meaning|share] [--mix]   # -> v4/direction/ (+ review_v4.1.md, before_after.*, UNITY_CONTRACT_v4.md)
    python -m dancecap.film_recap_flow build --version v4 --plan share --out <work>/v4/direction/v4.1_share   # the share-first alternative
    python -m dancecap.film_recap_flow check --version v4           # the flow checks + the owner's targets (exits 1 when one is missed)
    python -m dancecap.film_recap_flow clips --version v4           # UNITY STAGE ONLY
    (v4.1 is now built with --version v4.1 into v4/direction/v4.1/)
v4.2 = v4 (owner 2026-10-22 after v4.1: the physics meter only a few seconds at each switch-on, the slow orbits 20 % closer, the face swap on the
splats only): the v4.1 plan + flow.meter + closer orbit blocks:
    python -m dancecap.film_recap_flow build --version v4.2         # -> v4/direction/v4.2/ (reproducible: direction.json = the v4.2 the films were recorded from)
v4.3 = v4 (owner 2026-10-23: "when we show initial counter balances we need to orbit but maintain a two shot in profile"): the v4.2 direction + a
planned azimuth track for the slow orbit of chapters 1-3 (flow.azimuth_tracks, camera.azimuth_track; both phone ways stay v4.2's frame for frame):
    python -m dancecap.film_recap_flow build --version v4 --mix     # -> v4/direction/ (+ review_v4.3.md, track_v4.3.png, UNITY_CONTRACT_v4.md 'v4.3')
    python -m dancecap.film_recap_flow check --version v4           # v4.1's + v4.2's + v4.3's targets (the frame-exact one reads v4/direction/v4.2/)

The owner's words (2026-10-21):
  (3) "every time kadu say's 'back to the axis' he's referring to the counterbalance pivot point, not the T axis. correct this"
  (4) "since this is a physics based demo, I would like to see even more physics activations of foot center of gravity projections (the dot for
      each individual dancer on the floor) and tension compression estimations rendered in arms and legs"
  (5) "ideally every time we switch to a new camera view we fly in and out of it smoothly with a transition from splat to camera POV ... entering
      and exiting phone camera POV by seeing the 2D surface projected in the phone frustrum in the 3D world like we did in the explainer video ...
      we should stay at ground level for these transitions. all animation transitions should be consistent (we just never jumpcut ...)"

How it is built (pure post-passes over the D2/D4 director's plan, so the speech-driven shot list, captions, chapters, arrows and the soundtrack
time map stay exactly what film_recap builds; nothing of film_recap* is changed):
  1. `film_recap.build_direction` with the director mode, the "axis" events of the back-to-the-axis sentences taken out of the attention set
     (so no overhead / leader's-T window is planned for them) and a director config with longer phone shots (`V3_DIRCFG`: a phone shot now
     carries its own way in and out);
  2. `carve_pivots`: each back-to-the-axis phrase gets a ground-level counterbalance-pivot shot (physics_3q framing, counterbalance layer, the
     pivot dial, the couple's balance axis, an arrow "axis (pivot point)" on the pivot foot of that counterbalance - or, when the interval has
     no pivot, on the couple's centre of mass on the floor);
  3. `plan_flow`: shots (runs of segments with one camera block) -> per-shot eye azimuth plan (side-on, continuing from where the previous
     view was) -> one transition per shot boundary (type, duration from ONE rule, smootherstep easing, ground-level path constraints, the
     phones' frustum envelopes); a shot too short to hold its view between its way in and its way out first takes the little it misses from its
     neighbour (v3.2 `nudge`), else is merged into a neighbour (v3.2: a shot with nothing of its own to show into the phone beside it first);
  3b. v3.2 `give_phones_time` (owner "~50% phone"): the 3D views keep only the time they need (a calm / call-less view next to a phone becomes
     phone time; a physics view or zoom ends after its last call's arrow + its way out and starts just before its first call's arrow), every move
     kept only when every view still holds, every chapter still opens a shot and the phones' full-frame video grows;
  4. `apply_physics`: the physics layer (load colours = tension / compression in arms and legs, balance axes, each dancer's centre of mass with
     its floor dot) on every shot where the teachers demonstrate or are connected (hand contact or counterbalance connection) and not apart.
The Unity side reads the new keys (VIEWER_SPEC 3.15 + `flow` block, docs/RECAP_FILM.md "v3"); a viewer without them still plays the film
(blend at the shot start, hard cuts into phones) - the keys are additive.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import film_recap as fr
from . import film_recap_captions as caps
from . import film_recap_director as director
from . import film_recap_plan as plan
from . import recap_pov as rp
from .take import PACKAGE_ROOT

TAKE = fr.TAKE
CAPTURE = fr.CAPTURE
WORK = PACKAGE_ROOT / "work" / TAKE
OUT = WORK / "v3" / "direction"
PREV_V30 = WORK / "v3" / "direction_prev_v3.0" / "direction.json"     # v3.0 (2026-10-21, before the owner's v3.1 decisions)
PREV_V31 = WORK / "v3" / "direction_prev_v3.1" / "direction.json"     # v3.1 (2026-10-21, before v3.2 "the 3D views take only the time they need"), for the review
CAP_DIR = fr.VIEWER / "Assets" / "StreamingAssets" / CAPTURE

LEGEND = "Skeleton load - ESTIMATED from motion"
PIVOT_LABEL = "axis (pivot point)"

# the director with longer phone shots: a phone shot carries its own way in and out, so the 3-4 s phone shots of D4 would be mostly flight.
# v3.1 (owner 2026-10-21 "Shorter flights, ~50% phone"): the ways are quicker (0.8-2.6 s), so up to three phone shots in a row again (D4's run;
# v3.0 had two), long free stretches cut into ~7.5-s phone parts (v3.0 6.4, D4 5.2), and the share the director aims at raised from D4's 56-62 %
# to 62-68 % because the pivots, the joins and every phone way take some of it back (the balance step stops at about 59 % on this take: no calm
# shot or second zoom is left to hand to a phone)
V3_DIRCFG = director.DirCfg(pov_max_s=10.0, pov_part_s=7.5, calm_min_s=3.0, max_pov_run=3, zoom_min_s=2.6, pov_target=(0.62, 0.68))


# ------------------------------------------------------------------------------------------------ tunables


@dataclass
class FlowCfg:
    # (3) the pivot shots
    pivot_pre_s: float = 0.45          # before the phrase's first word ("bring")
    pivot_post_s: float = 1.6          # after the word "axis"
    pivot_min_s: float = 3.2
    pivot_search_s: float = 6.0        # a counterbalance interval within this many seconds of the word is the one meant
    pivot_distance_m: float = 3.6
    pivot_elevation_deg: float = 22.0  # eye about 2.25 m: the floor ring and the dial read, still ground level
    # (4) the physics layer
    phys_connected_min: float = 0.5    # share of a shot with a hand on the partner (or the counterbalance connection) that makes it "connected"
    phys_moving_frac: float = 0.2      # = plan.Cfg.motion_frac: the teachers demonstrate
    # (5) the flights. v3.1 (owner 2026-10-21, "Shorter flights, ~50% phone"): the ways into / out of the phones and the phones' minimum hold are
    # shorter, so the phones' full-frame video is back to about half the film; the overheads keep their crane (owner: "Keep overheads, crane up")
    easing: str = "smootherstep"       # every transition: x^3 (10 - 15 x + 6 x^2): zero speed AND zero acceleration at both ends
    arrive_after_s: float = 0.35       # a 3D view has arrived this long after its segment start (the segment starts 0.3 s before the words)
    phone_arrive_after_s: float = 0.0  # ... a phone's full-frame video (phone_in): at its segment start, 0.3 s before the words (v3.0: 0.35)
    arrive_late_max_s: float = 0.9     # ... at the latest (a squeezed boundary)
    call_arrive_before_s: float = 0.1  # a view that answers a call has arrived this long before the teacher says the word
    depart_before_s: float = 0.0       # the way out of a phone starts this long before the next segment (the phone keeps its video until then; v3.0: 0.3, v3.1: 0.1)
    hold_s: dict = field(default_factory=lambda: {"3d": 1.0, "phone": 1.4, "overhead": 2.2})       # phone v3.0: 2.0
    # duration rule T = base + per180 * |d azimuth| / 180 + per_m_r * |d radius| + per_m_h * |d height|, clipped to [lo, hi], never shorter than the
    # arc floor (the arc's smootherstep peak <= MAX_ARC_DEG_S); squeezable to `min` (and never below the arc floor)
    rule: dict = field(default_factory=lambda: {
        "flight": {"base": 0.9, "per180": 1.4, "per_m_r": 0.25, "per_m_h": 0.3, "lo": 0.9, "hi": 2.4, "min": 0.8},
        "crane_up": {"base": 1.6, "per180": 0.6, "per_m_r": 0.0, "per_m_h": 0.0, "lo": 1.6, "hi": 2.2, "min": 1.4},
        "crane_down": {"base": 1.6, "per180": 0.6, "per_m_r": 0.0, "per_m_h": 0.0, "lo": 1.6, "hi": 2.2, "min": 1.4},
        # v3.0: phone in / out 1.3 + 1.8 x |daz| / 180 (1.3-3.2 s, min 1.2), phone to phone 1.9 + 1.8 x (1.9-3.6, min 1.7)
        # v3.1: phone in / out 0.8 + 1.2 x (0.8-2.2, min 0.75), phone to phone 1.0 + 1.1 x (1.0-2.4, min 0.95)
        # v3.2: the floors are the frustum's readable floor (0.61 in / 0.68 out / 0.92 phone to phone) and the arc floor (200 deg/s), not the base
        "phone_in": {"base": 0.7, "per180": 1.2, "per_m_r": 0.1, "per_m_h": 0.1, "lo": 0.7, "hi": 2.2, "min": 0.7},
        "phone_out": {"base": 0.7, "per180": 1.2, "per_m_r": 0.1, "per_m_h": 0.1, "lo": 0.7, "hi": 2.2, "min": 0.7},
        "phone_to_phone": {"base": 0.93, "per180": 0.8, "per_m_r": 0.05, "per_m_h": 0.05, "lo": 0.93, "hi": 2.4, "min": 0.93},
        "replay_in": {"base": 1.2, "per180": 0.0, "per_m_r": 0.0, "per_m_h": 0.0, "lo": 1.2, "hi": 1.2, "min": 1.0},
        "replay_out": {"base": 1.2, "per180": 0.0, "per_m_r": 0.0, "per_m_h": 0.0, "lo": 1.2, "hi": 1.2, "min": 1.0},
    })
    squeeze_max_s: float = 0.35        # a transition is at most this much shorter than its rule (else the too-short shot is merged)
    crane_extra_s: float = 0.6         # a phone entered from / left to an overhead: the crane part on top of the phone rule
    axis_run_m: float = 1.0            # the last (first) metre of a phone entry (exit) runs along the phone's optical axis
    axis_share: float = 0.3            # ... and takes this share of the transition (v3.0: 0.4; the arc gets the rest, so it turns slower)
    frustum_depth_m: float = 0.9       # the 2D video surface stands this far in front of the phone, inside its frustum
    frustum_fade_s: float = 0.45
    frustum_leave_full: float = 0.7    # a leaving phone's surface stays opaque this share of the way out (then fades over the rest + frustum_fade_s)
    frustum_readable_s: float = 0.6    # every phone entry / exit shows the surface standing in the world (alpha >= 0.5, outside the axis dolly) this long
    ground_m: tuple = (0.5, 2.4)       # eye height of every flight (and of every 3D shot but the overhead)
    clear_radius_m: float = 1.4        # a flight never comes closer than this to the couple's centre
    inline_lift_max_m: float = 0.4     # the orbits' avoid_inline lift is capped (was up to 1.9 m: a swoop)
    flash_s: float = 0.3               # the white flash that hides the dancers' jump at a replay's start AND end
    max_merge_rounds: int = 80
    join_max_s: dict = field(default_factory=lambda: {"phone": 9.0, "3d": 10.0, "overhead": 8.0})   # a join never makes a longer shot
    # v3.2 (owner "~50% phone", 2026-10-21): the 3D views keep only the time they need, the rest goes to the phone beside them (give_phones_time)
    give_phone_cap_s: float = 10.0     # a phone shot that receives time is at most this long (= the director's pov_max_s)
    give_min_s: float = 0.6            # a smaller gift is not worth a boundary
    give_after_arrow_s: float = 0.3    # a view holds still this long after its last call's arrow before its way into the phone starts
    give_lead_s: float = 0.15          # a view starts this long before its first call's arrow (the director's phys_pre_s: 0.5 s before the word)
    nudge_margin_s: float = 0.05       # a view that misses its hold takes its deficit + this from the next shot (plan_flow's nudge) ...
    nudge_max_s: float = 0.8           # ... when that is at most this much (else the old join)
    nudge_min_shot_s: float = 2.0      # ... and only a shot at least this long (a sliver joins its neighbour as before)
    phone_first: bool = True           # a shot that cannot hold and has nothing of its own to show joins the phone beside it before a pivot / physics view
    own_phone_max_turn_deg: float = 90.0   # a new phone shot of its own (give_phones_time) never turns more than this from a phone neighbour
    # the eye of each 3D mode as the viewer frames it (9:16; estimates for the plan: radius about the couple's centre, height)
    eye_rh: dict = field(default_factory=lambda: {"orbit": (4.5, 1.75), "orbit_apart": (6.0, 1.75), "physics_3q": (3.4, 2.0), "pivot": (3.34, 2.25),
                                                  "feature_zoom": (2.0, 1.3), "feet_closeup": (3.4, 0.75), "rhythm": (3.9, 1.1), "overhead": (0.0, 7.0)})
    # half-width of the cone round side-on (perpendicular to the lead->follow line) a 3D shot may start in, and its drift (deg/s)
    cone_deg: dict = field(default_factory=lambda: {"orbit": 50.0, "physics_3q": 45.0, "pivot": 40.0, "feature_zoom": 40.0, "feet_closeup": 50.0, "rhythm": 60.0})
    drift_deg_s: dict = field(default_factory=lambda: {"orbit": 4.5, "physics_3q": 4.0, "pivot": 3.0, "feature_zoom": 0.0, "feet_closeup": 0.0, "rhythm": 12.0})


MAX_ARC_DEG_S = 200.0   # the fastest an arc may turn about the couple (smootherstep peaks at 1.875 x its mean rate); v3.0 had 160


def smootherstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------------------------------------ (3) "back to the axis"

AXIS_RX = re.compile(r"^axis$", re.I)
LEAD_WORDS = {"bring", "brings", "bringing", "go", "goes", "come", "comes", "coming", "return", "returns", "get", "take", "back"}


def back_to_axis(utterances: list[dict]) -> list[dict]:
    """every '(bring her / the follower) (back) to the axis' in the transcript: {utt, speaker, text, t0 (the phrase's first word), axis_t, axis_end,
    t1 (the sentence's end), variant ('back to the axis' | 'to the axis' (close variant, no 'back')), words}. Close variants: 'back to axis', 'go back to
    the axis', 'return to the axis', 'bring her to the axis'."""
    out = []
    for u in utterances:
        ws = [w for w in u.get("words", []) if w.get("kind") != "event" and w.get("start", -1) >= 0]
        norm = [re.sub(r"[^a-z']", "", str(w.get("text", "")).lower()) for w in ws]
        for i, n in enumerate(norm):
            if n != "axis":
                continue
            j = i - 1
            if j >= 0 and norm[j] == "the":
                j -= 1
            if j < 0 or norm[j] != "to":
                continue
            k = j - 1
            back = k >= 0 and norm[k] == "back"
            if not back and not (k >= 0 and norm[k] in LEAD_WORDS | {"her", "him", "follower", "leader", "partner"}):
                continue
            # the phrase starts at its verb (bring / go / come / return ...) within 4 words before 'to'
            s = j
            for m in range(j - 1, max(-1, j - 5), -1):
                if norm[m] in LEAD_WORDS - {"back"}:
                    s = m
                    break
                if norm[m] in (".", "and") or norm[m] == "axis":
                    break
            if s == j:
                s = k if back else j
            out.append({"utt": u["id"], "speaker": u.get("speaker"), "text": u.get("text", ""), "t0": float(ws[s]["start"]), "axis_t": float(ws[i]["start"]),
                        "axis_end": float(ws[i].get("end", ws[i]["start"] + 0.3)), "t1": float(u["end"]), "u_start": float(u["start"]),
                        "variant": "back to the axis" if back else "to the axis", "words": " ".join(w["text"] for w in ws[s:i + 1])})
    return out


def drop_axis_events(events: list[dict], phrases: list[dict]) -> tuple[list[dict], list[dict]]:
    """the teaching events minus the 'axis' calls of the back-to-the-axis phrases (they get the pivot shot instead of the leader's T)"""
    keep, dropped = [], []
    for e in events:
        trig = [str(t).lower() for t in e.get("triggers", [])]
        hit = any(abs(float(e["start"]) - p["axis_t"]) < 0.35 for p in phrases) and "axis" in trig and "overlay_T_axis" in e.get("tags", [])
        (dropped if hit else keep).append(e)
    return keep, dropped


def load_counterbalance(cap_dir: Path = CAP_DIR) -> dict:
    p = Path(cap_dir) / "counterbalance.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"intervals": []}


def pivot_target(cb: dict, t: float, search_s: float = 6.0) -> dict:
    """what the 'axis' of a back-to-the-axis phrase at master second t points at: the counterbalance pivot that contains t or is nearest within
    search_s (its anchored foot: the pivot point), else the counterbalance interval nearest (the couple's centre of mass on the floor = the
    balance point they share). Returns {target, interval {kind, t0, t1, pivot, pivot_foot}, why}."""
    t_off = float(cb.get("time_to_audio", 0.0) or 0.0)
    ivs = cb.get("intervals", [])

    def dist(iv):
        a, b = float(iv["t0"]) - t_off, float(iv["t1"]) - t_off
        return 0.0 if a <= t <= b else min(abs(t - a), abs(t - b))

    pivots = [iv for iv in ivs if iv.get("pivot") is not None and dist(iv) <= search_s]
    if pivots:
        # the one that contains the word, else the nearest; on a tie the one that comes (the phrase announces the return to it)
        iv = min(pivots, key=lambda x: (dist(x), 0 if float(x["t0"]) - t_off >= t else 1))
        mid = 0.5 * (float(iv.get("pivot_t0") or iv["t0"]) + float(iv.get("pivot_t1") or iv["t1"])) - t_off
        return {"target": {"type": "pivot_foot", "interval_t": round(mid, 3)},
                "interval": {"kind": "pivot", "t0": round(float(iv["t0"]) - t_off, 2), "t1": round(float(iv["t1"]) - t_off, 2), "pivot": iv["pivot"],
                             "pivot_foot": iv.get("pivot_foot"), "swept_deg": iv.get("swept_deg"), "gap_s": round(dist(iv), 2)},
                "why": f"the counterbalance pivot {float(iv['t0']) - t_off:.1f}-{float(iv['t1']) - t_off:.1f} s ({iv.get('pivot_foot')} foot, {iv.get('swept_deg')} deg): her anchored foot"}
    near = [iv for iv in ivs if dist(iv) <= search_s]
    if near:
        iv = min(near, key=dist)
        return {"target": {"type": "couple_com"},
                "interval": {"kind": iv.get("kind", "counterbalance"), "t0": round(float(iv["t0"]) - t_off, 2), "t1": round(float(iv["t1"]) - t_off, 2), "pivot": None,
                             "pivot_foot": None, "gap_s": round(dist(iv), 2)},
                "why": f"the counterbalance {float(iv['t0']) - t_off:.1f}-{float(iv['t1']) - t_off:.1f} s has no anchored foot: the couple's centre of mass on the floor (their shared balance point)"}
    return {"target": {"type": "couple_com"}, "interval": None, "why": "no counterbalance within reach: the couple's centre of mass on the floor"}


# ------------------------------------------------------------------------------------------------ segments: split / carve


def _locked(s: dict) -> bool:
    return not s.get("replay") and float(s.get("speed", 1.0)) == 1.0


def split_segment(s: dict, t: float) -> tuple[dict, dict]:
    """a locked segment cut at master second t into two (film times follow the constant offset; annotations go where they overlap >= 0.3 s)"""
    assert _locked(s) and s["dance"][0] < t < s["dance"][1]
    off = s["film"][0] - s["dance"][0]
    a, b = copy.deepcopy(s), copy.deepcopy(s)
    a["dance"] = [s["dance"][0], round(t, 3)]
    a["film"] = [s["film"][0], round(t + off, 3)]
    b["dance"] = [round(t, 3), s["dance"][1]]
    b["film"] = [round(t + off, 3), s["film"][1]]
    for k in ("chapter", "chapter_card"):
        b.pop(k, None)
    b["id"] = s["id"] + "_r"
    a["annotations"], b["annotations"] = _clip_annos(s["annotations"], a), _clip_annos(s["annotations"], b)
    return a, b


def _clip_annos(annos: list[dict], seg: dict) -> list[dict]:
    out = []
    d0, d1 = seg["dance"]
    for a in annos:
        x0, x1 = max(a["dance_t"][0], d0), min(a["dance_t"][1], d1)
        if x1 - x0 < 0.3:
            continue
        na = copy.deepcopy(a)
        na["dance_t"] = [round(x0, 3), round(x1, 3)]
        na["film_in_segment"] = [round(x0 - d0, 3), round(x1 - d0, 3)]
        out.append(na)
    return out


def refilm_annos(seg: dict) -> None:
    for a in seg["annotations"]:
        a["film_in_segment"] = [round((a["dance_t"][0] - seg["dance"][0]) / float(seg["speed"]), 3), round((a["dance_t"][1] - seg["dance"][0]) / float(seg["speed"]), 3)]


def carve(segs: list[dict], a: float, b: float, new: dict) -> list[dict]:
    """replace the master window [a, b] (inside locked segments) by the segment `new` (camera, layers, annotations, id given; dance / film set here).
    The pieces of the segments it cuts keep their ids (the later piece '_r'), chapters and cards; annotations of the window are kept on `new`."""
    out: list[dict] = []
    inside: list[dict] = []
    for s in segs:
        d0, d1 = s["dance"]
        if s.get("replay") or d1 <= a + 1e-6 or d0 >= b - 1e-6:
            out.append(s)
            continue
        if not _locked(s):
            raise ValueError(f"carve {a:.2f}-{b:.2f} crosses a slowed segment {s['id']}")
        cur = s
        if d0 < a - 1e-6:
            x, cur = split_segment(cur, a)
            out.append(x)
        if cur["dance"][1] > b + 1e-6:
            y, z = split_segment(cur, b)
            inside.append(y)
            out.append(("NEW",))
            out.append(z)
        else:
            inside.append(cur)
            out.append(("NEW",))
    # one NEW marker per carved window: keep the first, the chapter of the first inside piece moves to the new segment
    res, placed = [], False
    for x in out:
        if isinstance(x, tuple):
            if placed:
                continue
            placed = True
            off = inside[0]["film"][0] - inside[0]["dance"][0]
            n = copy.deepcopy(new)
            n["dance"] = [round(a, 3), round(b, 3)]
            n["film"] = [round(a + off, 3), round(b + off, 3)]
            for k in ("chapter", "chapter_card"):
                if k in inside[0] and abs(inside[0]["dance"][0] - a) < 1e-6:
                    n[k] = inside[0][k]
            keep = [an for s in inside for an in s["annotations"] if an.get("kind") in ("readout",)]
            n["annotations"] = _clip_annos(keep, n) + n.get("annotations", [])
            res.append(n)
            continue
        res.append(x)
    return res


# ------------------------------------------------------------------------------------------------ geometry (Unity capture coordinates)


class Geo:
    """the couple and the phones on the capture's clock, Unity capture coordinates (x, y, -z of the take's world; DanceOrigin.Offset not added:
    every position here is relative, so it cancels). Azimuth = atan2(z - centre z, x - centre x) in degrees, the viewer's convention
    (FilmCamera: eye = centre + (cos a r, h, sin a r))."""

    def __init__(self, cap_dir: Path = CAP_DIR, physics: Path | None = None):
        ph = json.loads((physics or (WORK / "physics" / "physics.json")).read_text(encoding="utf-8"))
        self.t = np.asarray(ph["t"], float)

        def unity(a):
            a = np.asarray(a, float).copy()
            a[:, 2] *= -1.0
            return a

        self.lead = unity(ph["dancers"]["lead"]["com"])
        self.follow = unity(ph["dancers"]["follow"]["com"])
        for a in (self.lead, self.follow):
            for c in range(3):
                ok = np.isfinite(a[:, c])
                if ok.sum() > 2 and not ok.all():
                    a[:, c] = np.interp(self.t, self.t[ok], a[ok, c])
        self.phones: dict[str, dict] = {}
        cams = Path(cap_dir) / "cameras"
        man = json.loads((cams / "cameras.json").read_text(encoding="utf-8")) if (cams / "cameras.json").exists() else {"cameras": []}
        for c in man.get("cameras", []):
            tp = cams / c["track"]["file"]
            if not tp.exists():
                continue
            tr = json.loads(tp.read_text(encoding="utf-8"))
            self.phones[c["id"]] = {"t": np.asarray(tr["t"], float), "pos": np.asarray(tr["pos"], float), "fwd": np.asarray(tr["fwd"], float),
                                    "up": np.asarray(tr["up"], float), "vfov": np.asarray(tr["vfov"], float), "aspect": float(c.get("aspect") or tr["size"][0] / tr["size"][1]),
                                    "clips": [list(map(float, x)) for x in c.get("clips", [])]}

    def _at(self, arr: np.ndarray, t: float) -> np.ndarray:
        return np.array([np.interp(t, self.t, arr[:, k]) for k in range(arr.shape[1])])

    def centre(self, t: float) -> np.ndarray:
        c = 0.5 * (self._at(self.lead, t) + self._at(self.follow, t))
        c[1] = 0.0
        return c

    def line_deg(self, t: float) -> float:
        d = self._at(self.follow, t) - self._at(self.lead, t)
        return math.degrees(math.atan2(d[2], d[0])) if math.hypot(d[0], d[2]) > 1e-3 else 0.0

    def sep(self, t: float) -> float:
        d = self._at(self.follow, t) - self._at(self.lead, t)
        return float(math.hypot(d[0], d[2]))

    def phone(self, cam: str, t: float) -> dict:
        p = self.phones[cam]
        i = int(np.clip(np.searchsorted(p["t"], t), 1, len(p["t"]) - 1))
        k = float(np.clip((t - p["t"][i - 1]) / max(1e-6, p["t"][i] - p["t"][i - 1]), 0.0, 1.0))
        pos = (1 - k) * p["pos"][i - 1] + k * p["pos"][i]
        fwd = (1 - k) * p["fwd"][i - 1] + k * p["fwd"][i]
        fwd /= max(1e-9, np.linalg.norm(fwd))
        c = self.centre(t)
        az = math.degrees(math.atan2(pos[2] - c[2], pos[0] - c[0]))
        return {"pos": pos, "fwd": fwd, "vfov": float((1 - k) * p["vfov"][i - 1] + k * p["vfov"][i]), "aspect": p["aspect"], "az": az,
                "r": float(math.hypot(pos[0] - c[0], pos[2] - c[2])), "h": float(pos[1])}

    def recording(self, cam: str) -> tuple[float, float]:
        """master seconds the phone records (its track's first / last sample: the cameras stage tracks every frame it has)"""
        t = self.phones[cam]["t"]
        return float(t[0]), float(t[-1])

    def clip_covers(self, cam: str, a: float, b: float) -> bool:
        return any(x0 <= a + 1e-3 and b <= x1 + 1e-3 for x0, x1 in self.phones.get(cam, {}).get("clips", []))


# ------------------------------------------------------------------------------------------------ shots


def shot_kind(cam: dict) -> str:
    m = cam.get("mode")
    if m in ("camera_tour", "fly_to_camera"):
        return "phone"
    if m == "overhead_floorcraft":
        return "overhead"
    return "3d"


def phone_of(cam: dict) -> str | None:
    if cam.get("mode") == "camera_tour":
        return str(cam["cameras"][0]["camera"]) if cam.get("cameras") else None
    if cam.get("mode") == "fly_to_camera":
        return str(cam.get("camera"))
    return None


def eye_key(seg: dict) -> str:
    cam = seg["camera"]
    if seg.get("pivot"):
        return "pivot"
    m = cam.get("mode", "orbit")
    if m == "orbit" and (cam.get("radius_m", 0) or 0) >= 4.7:
        return "orbit_apart"
    return m


def importance(seg: dict) -> int:
    if seg.get("replay"):
        return 9
    if seg.get("pivot"):
        return 6
    m = seg["camera"].get("mode")
    calls = sum(1 for a in seg["annotations"] if a.get("calls"))
    if m == "physics_3q":
        return 4
    if m in ("feature_zoom", "feet_closeup", "overhead_floorcraft", "rhythm"):
        return 3
    if shot_kind(seg["camera"]) == "phone":
        return 2 if calls else 1
    return 0


@dataclass
class Shot:
    segs: list[int]
    kind: str
    mode: str
    f0: float
    f1: float
    d0: float
    d1: float
    replay: bool
    phone: str | None
    imp: int
    key: str
    # the plan
    a0: float = 0.0
    a1: float = 0.0
    r0: float = 0.0
    h0: float = 0.0
    r1: float = 0.0
    h1: float = 0.0
    drift: float = 0.0
    arrive: float = 0.0
    trans: dict | None = None
    first_call: float | None = None

    @property
    def dur(self) -> float:
        return self.f1 - self.f0


def shots_of(segs: list[dict]) -> list[Shot]:
    out: list[Shot] = []
    for i, s in enumerate(segs):
        jump = i > 0 and abs(segs[i - 1]["dance"][1] - s["dance"][0]) > 0.05
        if out and not jump and segs[out[-1].segs[-1]]["camera"] == s["camera"]:
            sh = out[-1]
            sh.segs.append(i)
            sh.f1, sh.d1 = s["film"][1], s["dance"][1]
            sh.imp = max(sh.imp, importance(s))
            continue
        out.append(Shot([i], shot_kind(s["camera"]), s["camera"].get("mode"), s["film"][0], s["film"][1], s["dance"][0], s["dance"][1], bool(s.get("replay")),
                        phone_of(s["camera"]), importance(s), eye_key(s)))
    return out


def first_calls(segs: list[dict], shots: list[Shot]) -> None:
    """each shot's first attention word (film s): an arrow answering a call appears 0.35 s before the word"""
    for sh in shots:
        best = None
        for i in sh.segs:
            s = segs[i]
            for a in s["annotations"]:
                if not a.get("calls") or a.get("kind") not in ("arrow3d", "vector_arrow", "floor_ring"):
                    continue
                f = s["film"][0] + (a["dance_t"][0] + 0.35 - s["dance"][0]) / float(s["speed"])
                if best is None or f < best:
                    best = f
        sh.first_call = best


def dance_at(segs: list[dict], sh: Shot, f: float) -> float:
    """the master second shown at film second f inside the shot"""
    for i in sh.segs:
        s = segs[i]
        if s["film"][0] - 1e-6 <= f <= s["film"][1] + 1e-6:
            return s["dance"][0] + (f - s["film"][0]) * float(s["speed"])
    s = segs[sh.segs[-1]] if f > sh.f1 else segs[sh.segs[0]]
    return s["dance"][1] if f > sh.f1 else s["dance"][0]


# ------------------------------------------------------------------------------------------------ (5) the azimuth plan and the transitions


def transition_type(prev: Shot, nxt: Shot) -> str:
    if nxt.replay:
        return "replay_in"
    if prev.replay:
        return "replay_out"
    a, b = prev.kind, nxt.kind
    if a == "phone" and b == "phone":
        return "phone_to_phone"
    if b == "phone":
        return "phone_in"
    if a == "phone":
        return "phone_out"
    if b == "overhead" and a != "overhead":
        return "crane_up"
    if a == "overhead" and b != "overhead":
        return "crane_down"
    return "flight"


AXIS_TYPES = ("phone_in", "phone_out", "phone_to_phone")


def arc_share(kind: str, cfg: FlowCfg) -> float:
    """the share of a transition's time its arc takes (a phone way spends `axis_share` of it on the phone's optical axis)"""
    return 1.0 - (cfg.axis_share if kind in AXIS_TYPES else 0.0)


def arc_floor(kind: str, daz: float, cfg: FlowCfg) -> float:
    """the shortest a transition with an arc of daz degrees may be: smootherstep peaks at 1.875 x the mean rate of its arc part"""
    return 1.875 * (abs(daz) + 0.06) / (MAX_ARC_DEG_S * arc_share(kind, cfg))        # + the 0.1-deg rounding of the written arc


def crane_floor(cfg: FlowCfg) -> float:
    """a phone way that passes an overhead is never quicker than the crane itself (the owner's 'crane up' stays as smooth as the plain cranes)"""
    return float(cfg.rule["crane_up"]["lo"])


def readable_floor(kind: str, cfg: FlowCfg) -> float:
    """the shortest phone way whose frustum surface(s) still read for cfg.frustum_readable_s (0 for the other transitions)"""
    if kind not in AXIS_TYPES:
        return 0.0
    roles = {"phone_in": ("enter",), "phone_out": ("leave",), "phone_to_phone": ("enter", "leave")}[kind]

    def ok(T):
        return min(frustum_readable(r, kind, frustum_envelope(r, kind, 0.0, T, cfg), 0.0, T, cfg) for r in roles) >= cfg.frustum_readable_s
    lo, hi = 0.05, 10.0
    if ok(lo):
        return lo
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def rule_duration(kind: str, daz: float, dr: float, dh: float, cfg: FlowCfg, crane: bool = False) -> float:
    r = cfg.rule[kind]
    t = r["base"] + r["per180"] * abs(daz) / 180.0 + r["per_m_r"] * abs(dr) + r["per_m_h"] * abs(dh)
    t = max(min(r["hi"], max(r["lo"], t)), arc_floor(kind, daz, cfg), readable_floor(kind, cfg))
    if crane:
        t = max(t + cfg.crane_extra_s, crane_floor(cfg))
    return math.ceil(t * 1000.0 - 1e-6) / 1000.0


def squeeze_floor(kind: str, daz: float, cfg: FlowCfg, crane: bool = False) -> float:
    """the shortest a squeezed transition may get: its rule's `min` (+ the crane part), never below the arc floor, the frustum's readable floor or
    the crane floor"""
    m = cfg.rule[kind]["min"] + (cfg.crane_extra_s if crane else 0.0)
    m = max(m, arc_floor(kind, daz, cfg), readable_floor(kind, cfg) + 1e-3)
    if crane:
        m = max(m, crane_floor(cfg))
    return m


def side_start(line: float, prev_az: float, cone: float) -> float:
    """the start azimuth of a side-on 3D shot: the previous view's azimuth when it lies within `cone` of side-on (either side of the lead->follow line),
    else the nearest edge of the nearer cone (5 deg inside it)"""
    best = None
    for side in (line + 90.0, line - 90.0):
        d = wrap180(prev_az - side)
        if abs(d) <= cone:
            return wrap180(prev_az)
        edge = wrap180(side + math.copysign(cone - 5.0, d))
        dd = abs(wrap180(prev_az - edge))
        if best is None or dd < best[0]:
            best = (dd, edge)
    return best[1]


def plan_azimuths(shots: list[Shot], segs: list[dict], geo: Geo, cfg: FlowCfg) -> None:
    """eye azimuth / radius / height at each shot's start and end: phones are fixed (their own pose); a 3D shot starts where the previous view
    was when that is side-on enough, and drifts toward the next fixed view; an overhead keeps the azimuth it was entered from (its screen-up)"""
    prev_az = None
    sign = 1.0
    first_calls(segs, shots)
    for k, sh in enumerate(shots):
        if sh.kind == "phone":
            p0, p1 = geo.phone(sh.phone, sh.d0), geo.phone(sh.phone, sh.d1)
            sh.a0, sh.r0, sh.h0 = p0["az"], p0["r"], p0["h"]
            sh.a1, sh.r1, sh.h1 = p1["az"], p1["r"], p1["h"]
            prev_az = sh.a1
            continue
        if sh.kind == "overhead":
            sh.a0 = sh.a1 = prev_az if prev_az is not None else geo.line_deg(sh.d0) + 90.0
            sh.r0 = sh.r1 = cfg.eye_rh["overhead"][0]
            sh.h0 = sh.h1 = cfg.eye_rh["overhead"][1]
            prev_az = sh.a1
            continue
        key = sh.key
        cone = cfg.cone_deg.get("orbit" if key.startswith("orbit") else key, 45.0)
        line = geo.line_deg(sh.d0)
        start_from = prev_az if prev_az is not None else line + 90.0
        sh.a0 = side_start(line, start_from, cone)
        # the next fixed view (a phone within the next two shots): drift toward it
        nxt = next((x for x in shots[k + 1:k + 3] if x.kind == "phone"), None)
        if nxt is not None:
            tgt = geo.phone(nxt.phone, nxt.d0)["az"]
            d = wrap180(tgt - sh.a0)
            if abs(d) > 10.0:
                sign = math.copysign(1.0, d)
        rate = cfg.drift_deg_s.get("orbit" if key.startswith("orbit") else key, 0.0)
        sh.drift = sign * rate
        span = sh.d1 - sh.d0 if not sh.replay else (sh.f1 - sh.f0)
        sh.a1 = wrap180(sh.a0 + sh.drift * (sh.f1 - sh.f0))
        if nxt is not None and rate > 0:
            # never drift past the phone it is heading for
            tgt = geo.phone(nxt.phone, nxt.d0)["az"]
            if abs(wrap180(tgt - sh.a0)) < abs(sh.drift * (sh.f1 - sh.f0)):
                sh.a1 = wrap180(tgt)
        sh.r0, sh.h0 = cfg.eye_rh.get(key, (3.5, 1.6))
        sh.r1, sh.h1 = sh.r0, sh.h0
        prev_az = sh.a1
        _ = span


def place(typ: str, f0: float, T: float, cfg: FlowCfg, word: float | None = None) -> tuple[float, float]:
    """(start, latest start) of a transition into a shot starting at film f0. Where it sits follows one rule: the phone's own video keeps the screen as
    long as it can. Into a 3D view or a phone (flight, crane, phone_in) the new view has ARRIVED `arrive_after_s` after f0 (the words begin);
    out of a phone (phone_out) the way out starts `depart_before_s` before f0 (the 3D layers of the new shot show through the flight); from one phone to
    the next (phone_to_phone) the flight is centred on f0; a replay's flight starts at f0 (the dance time jumps there, under the flash)."""
    if typ in ("replay_in", "replay_out"):
        return f0, f0
    if typ == "phone_out":
        st, late = f0 - cfg.depart_before_s, f0 + 0.5
    elif typ == "phone_to_phone":
        st, late = f0 - 0.5 * T, f0 - 0.5 * T + 0.5
    else:
        arr = cfg.phone_arrive_after_s if typ == "phone_in" else cfg.arrive_after_s
        st, late = f0 + arr - T, f0 + max(arr, cfg.arrive_late_max_s) - T
    if word is not None and word - f0 < 3.0:
        # the view that illustrates a call has arrived 0.1 s before the teacher says the word
        st = min(st, word - cfg.call_arrive_before_s - T)
        late = min(late, max(st, word - cfg.call_arrive_before_s - T + 0.3))
    return st, late


def schedule(shots: list[Shot], segs: list[dict], geo: Geo, cfg: FlowCfg) -> list[dict]:
    """a transition into every shot but the first: type, duration (the rule, squeezed by at most `squeeze_max_s` if it must), start / arrival (film s).
    Returns the problems [{k (index of the shot that does not hold its view), deficit_s, boundary}] - empty = every view holds at least its minimum"""
    problems = []
    if not shots:
        return problems
    shots[0].arrive = shots[0].f0
    shots[0].trans = None
    for k in range(1, len(shots)):
        p, n = shots[k - 1], shots[k]
        typ = transition_type(p, n)
        daz = wrap180(n.a0 - p.a1)
        crane = (typ == "phone_in" and p.kind == "overhead") or (typ == "phone_out" and n.kind == "overhead")
        T = rule_duration(typ, daz, n.r0 - p.r1, 0.0 if crane else n.h0 - p.h1, cfg, crane)
        rule_T = T
        start, latest = place(typ, n.f0, T, cfg, n.first_call)
        need = p.arrive + (1.5 if p.replay else cfg.hold_s.get(p.kind, 1.0))
        squeezed = []
        if start < need - 1e-6 and typ not in ("replay_in", "replay_out"):
            late = min(need - start, latest - start)
            if late > 0:
                start += late
                squeezed.append(f"starts {late:.2f} s later")
            if start < need - 1e-6:
                cut = min(need - start, T - squeeze_floor(typ, daz, cfg, crane), cfg.squeeze_max_s)
                if cut > 0:
                    T -= cut
                    start += cut
                    squeezed.append(f"{cut:.2f} s shorter than the rule")
        # v3.2: a phone is entered only while it records (its frustum surface shows its video and the way flies to its live pose: phone 05 starts
        # at 7.41 s), and left before it stops: squeezed first (at most squeeze_max_s, never below the floors), then later / earlier
        if geo is not None and typ in ("phone_in", "phone_to_phone") and n.phone in geo.phones:
            rec0 = geo.recording(n.phone)[0] + (n.f0 - n.d0)
            earliest = rec0 + (cfg.frustum_fade_s if typ == "phone_in" else 0.0)
            if start < earliest - 1e-6:
                cut = min(earliest - start, T - squeeze_floor(typ, daz, cfg, crane), cfg.squeeze_max_s - (rule_T - T))
                if cut > 1e-6:
                    T -= cut
                    start += cut
                    squeezed.append(f"{cut:.2f} s shorter than the rule (phone {n.phone} records from {rec0:.2f} s)")
                if start < earliest - 1e-6:
                    squeezed.append(f"starts {earliest - start:.2f} s later: phone {n.phone} records from {rec0:.2f} s")
                    start = earliest
        if geo is not None and typ in ("phone_out", "phone_to_phone") and p.phone in geo.phones:
            rec1 = geo.recording(p.phone)[1] + (p.f1 - p.d1)
            last_end = rec1 - (cfg.frustum_fade_s if typ == "phone_out" else 0.0)
            if start + T > last_end + 1e-6:
                squeezed.append(f"starts {start + T - last_end:.2f} s earlier: phone {p.phone} records until {rec1:.2f} s")
                start = last_end - T
        if start < need - 1e-6:
            problems.append({"k": k - 1, "deficit_s": round(need - start, 3), "boundary": k})
        arrive = start + T
        if arrive > n.f1 - 0.2 and k + 1 < len(shots):
            problems.append({"k": k, "deficit_s": round(arrive - (n.f1 - 0.2), 3), "boundary": k + 1})
        n.arrive = arrive
        n.trans = {"type": typ, "start_film": round(start, 3), "arrive_film": round(arrive, 3), "duration_s": round(T, 3), "rule_s": rule_T,
                   "daz": round(daz, 1), "squeezed": squeezed}
    return problems


JOIN_CAP = FlowCfg().join_max_s
PROTECT = 4      # pivot (6), replay (9), physics view (4): a too-short one keeps its view and grows instead of vanishing


def join(segs: list[dict], shots: list[Shot], surv: Shot, gone: Shot, pov: director.PovSource | None) -> bool:
    """`gone`'s segments take `surv`'s camera, layers and view (the two shots become one). A phone survivor gets the crop of the joined window and must
    still see both teachers there (recap_pov.window_quality). False = not possible."""
    if surv.replay or gone.replay:
        return False
    return join_idx(segs, surv.segs, gone.segs, surv.kind, surv.phone, pov)


def phone_can_show(pov: director.PovSource | None, phone: str | None, lo: float, hi: float) -> bool:
    """the phone sees both teachers well enough over the master window [lo, hi] (recap_pov.window_quality; never inside its bad stretches)"""
    if pov is None or phone is None:
        return False
    ph = pov.phones.get(phone)
    return ph is not None and rp.window_quality(ph, lo, hi, pov.bad.get(phone, []), pov.cfg) is not None


def join_idx(segs: list[dict], surv: list[int], gone: list[int], kind: str, phone: str | None, pov: director.PovSource | None) -> bool:
    """the segments `gone` take the camera, layers and view of the segments `surv` (one shot from then on); a phone survivor's crop is re-made for the
    joined window, which the phone must still show (phone_can_show). False = not possible (nothing changed)."""
    cam = copy.deepcopy(segs[surv[0]]["camera"])
    lay = copy.deepcopy(segs[surv[0]]["layers"])
    view = segs[surv[0]]["view_state"]
    treat = copy.deepcopy(segs[surv[0]].get("treatment") or {})
    lo, hi = min(segs[i]["dance"][0] for i in surv + gone), max(segs[i]["dance"][1] for i in surv + gone)
    if kind == "phone":
        if not phone_can_show(pov, phone, lo, hi):
            return False
        cam["crop"] = rp.crop_track(pov.phones[phone], lo, hi, pov.cfg)
        if cam.get("mode") == "camera_tour":
            cam["cameras"] = [{"camera": phone, "dance_t": [round(lo, 3), round(hi, 3)]}]
    for i in sorted(set(surv + gone)):
        sg = segs[i]
        if i in gone:
            sg["v3_merged_from"] = sg["camera"].get("mode")
            sg["treatment"] = dict(treat, merged=True)
            if segs[surv[0]].get("pivot"):
                sg["pivot"] = segs[surv[0]]["pivot"]
            else:
                sg.pop("pivot", None)
        sg["camera"] = copy.deepcopy(cam)
        sg["layers"] = copy.deepcopy(lay)
        if kind != "phone":
            sg["layers"].pop("source_video", None)
        sg["view_state"] = view
    return True


def phone_share(shots: list[Shot]) -> float:
    tot = sum(x.dur for x in shots)
    return sum(x.dur for x in shots if x.kind == "phone") / tot if tot else 0.0


def rank(sh: Shot, share: float, target: float) -> int:
    """a shot's weight in a join: its importance, a phone shot lifted above the zooms (3) while the phones are short of their share"""
    if sh.kind == "phone" and share < target and sh.imp < PROTECT:
        return 3 + sh.imp          # 4 / 5: above a zoom (3), still below a physics view (4 + ... see PROTECT) on ties the longer survives
    return sh.imp


def fix_boundary(segs: list[dict], shots: list[Shot], k: int, pov: director.PovSource | None, log: list[str], phone_first: bool = False) -> bool:
    """shot k does not hold its view between its way in and its way out (its departure, the transition into shot k + 1, would start before it has held
    its minimum). One boundary next to it goes: an ordinary shot joins the neighbour with the more important view (a calm 3D or phone moment becomes part
    of the zoom / physics / pivot view beside it; on a tie the longer neighbour survives); a protected shot (pivot, physics view) keeps its view and takes
    its less important neighbour's time, or joins a protected neighbour (the longer view survives). Never across a replay. Each call removes one
    boundary, so the loop ends."""
    sh = shots[k]
    nbs = [j for j in (k + 1, k - 1) if 0 <= j < len(shots) and not shots[j].replay]
    share = phone_share(shots)
    target = V3_DIRCFG.pov_target[0]
    me = rank(sh, share, target)
    if phone_first and share < target and not sh.replay and sh.imp < PROTECT and not segs[sh.segs[0]].get("pivot") and not call_spans(segs, sh):
        # v3.2: a shot with nothing of its own to show (a calm sliver, a call-less piece, a phone sliver) goes to the phone beside it when that phone can
        # show it: the pivot / physics view beside it keeps only the time it needs (the longer phone first)
        for j in sorted((j for j in nbs if shots[j].kind == "phone"), key=lambda j: -shots[j].dur):
            nb = shots[j]
            if max(nb.f1, sh.f1) - min(nb.f0, sh.f0) > JOIN_CAP.get("phone", 9.0) + 1e-6:
                continue
            if join(segs, shots, nb, sh, pov):
                log.append(f"{sh.mode} {sh.f0:.1f}-{sh.f1:.1f} ({sh.dur:.1f} s, nothing of its own to show) joined the phone shot {nb.phone} "
                           f"{nb.f0:.1f}-{nb.f1:.1f}: it could not hold its view between its way in and out")
                return True
    pairs = []
    for j in nbs:
        nb = shots[j]
        if sh.replay:
            continue
        rn = rank(nb, share, target)
        protected_me = sh.imp >= PROTECT
        if protected_me and rn < me and not (nb.kind == "phone" and rn >= PROTECT):
            pairs.append((0, rn, nb.dur, sh, nb))
        elif rn > me and not (sh.imp >= PROTECT and nb.imp < PROTECT):
            pairs.append((1, -rn, nb.dur, nb, sh))
        elif sh.imp >= PROTECT and nb.imp < PROTECT:
            pairs.append((0, rn, nb.dur, sh, nb))
        else:
            surv, gone = (sh, nb) if sh.dur >= nb.dur else (nb, sh)
            pairs.append((2, 0, gone.dur, surv, gone))
    for *_, surv, gone in sorted(pairs, key=lambda x: x[:3]):
        cap = JOIN_CAP.get(surv.kind, 10.0)
        if max(surv.f1, gone.f1) - min(surv.f0, gone.f0) > cap + 1e-6:
            continue
        if join(segs, shots, surv, gone, pov):
            log.append(f"{gone.mode} {gone.f0:.1f}-{gone.f1:.1f} ({gone.dur:.1f} s) joined {surv.mode} {surv.f0:.1f}-{surv.f1:.1f}: "
                       f"the {sh.mode} shot at {sh.f0:.1f} s could not hold {('its view' if sh is gone else 'its view, it grew')} between its way in and out")
            return True
    return False


def nudge(segs: list[dict], shots: list[Shot], pr: dict, probs: list[dict], geo: Geo, cfg: FlowCfg, pov: director.PovSource | None,
          cuts: list[float], chapter_ids: set[str]) -> tuple[list[dict], float] | None:
    """v3.2: a view that misses its hold by a little takes that little from the shot after it (the boundary moves later, to a pause of the speech
    when one is near, else by the deficit itself when it is at most 0.4 s) instead of swallowing the whole next shot. Never into a replay, an
    overhead, a pivot or a chapter's first shot. Returns (the new segment list, the seconds taken) when the schedule's problems shrink, else None."""
    b = pr["boundary"]
    if b <= 0 or b >= len(shots):
        return None
    grow, shrink = shots[b - 1], shots[b]
    if grow.replay or shrink.replay or shrink.kind == "overhead" or segs[shrink.segs[0]].get("pivot") or grow.kind == "overhead" or grow.dur < cfg.nudge_min_shot_s:
        return None
    d = pr["deficit_s"] + cfg.nudge_margin_s
    if d > cfg.nudge_max_s:
        return None
    t0 = shrink.f0 + d
    cands = [c for c in film_cuts(segs, shrink, cuts) if t0 <= c <= t0 + 0.8] or ([round(t0, 3)] if d <= 0.4 else [])
    before = (len(probs), sum(p["deficit_s"] for p in probs))
    for c in cands:
        if c > shrink.f1 - 0.5:
            continue
        trial = copy.deepcopy(segs)
        if not split_at_film(trial, c):
            continue
        gone = [i for i, s in enumerate(trial) if s["film"][0] >= shrink.f0 - 1e-3 and s["film"][1] <= c + 1e-3]
        surv = [i for i, s in enumerate(trial) if s["film"][0] >= grow.f0 - 1e-3 and s["film"][1] <= grow.f1 + 1e-3]
        if not gone or not surv or any(not _locked(trial[i]) for i in gone):
            continue
        if not join_idx(trial, surv, gone, grow.kind, grow.phone, pov):
            continue
        for i in gone:
            trial[i]["v3_nudged_into"] = grow.mode
        renumber(trial)
        if not chapters_start_shots(trial, chapter_ids):
            continue
        sh2 = shots_of(trial)
        plan_azimuths(sh2, trial, geo, cfg)
        pr2 = schedule(sh2, trial, geo, cfg)
        if (len(pr2), sum(p["deficit_s"] for p in pr2)) < (before[0], before[1] - 0.01) and len(pr2) <= before[0]:
            return trial, c - shrink.f0
    return None


def plan_flow(segs: list[dict], geo: Geo, cfg: FlowCfg, pov: director.PovSource | None, cuts: list[float] | None = None,
              chapter_ids: set[str] | None = None) -> tuple[list[Shot], dict]:
    """join too-short shots until every view holds, then the azimuth plan and the transitions. v3.2: a shot that misses its hold by a little first
    tries to take that little from its neighbour (`nudge`, when `cuts` are given); only when that does not help does a shot join its neighbour.
    The segment list is changed IN PLACE."""
    log: list[str] = []
    for _ in range(cfg.max_merge_rounds):
        shots = shots_of(segs)
        plan_azimuths(shots, segs, geo, cfg)
        probs = schedule(shots, segs, geo, cfg)
        if not probs:
            break
        done = False
        if cuts is not None:
            for pr in sorted(probs, key=lambda x: -x["deficit_s"]):
                res = nudge(segs, shots, pr, probs, geo, cfg, pov, cuts, chapter_ids or set())
                if res is not None:
                    b = pr["boundary"]
                    log.append(f"nudge: {shots[b - 1].mode} {shots[b - 1].f0:.1f}-{shots[b - 1].f1:.1f} held {pr['deficit_s']:.2f} s too short: it takes "
                               f"{res[1]:.2f} s from {shots[b].mode}{(' ' + shots[b].phone) if shots[b].phone else ''} "
                               f"{shots[b].f0:.1f}-{shots[b].f1:.1f} instead of joining it")
                    segs[:] = res[0]
                    done = True
                    break
            if done:
                continue
        for pr in sorted(probs, key=lambda x: -x["deficit_s"]):
            if fix_boundary(segs, shots, pr["k"], pov, log, cfg.phone_first) or fix_boundary(segs, shots, pr["boundary"], pov, log, cfg.phone_first):
                done = True
                break
        if not done:
            log.append(f"could not fix {len(probs)} squeezed boundaries: {probs}")
            break
    shots = shots_of(segs)
    plan_azimuths(shots, segs, geo, cfg)
    probs = schedule(shots, segs, geo, cfg)
    return shots, {"merges": log, "problems": probs}


# ------------------------------------------------------------------------------------------------ v3.2: the 3D views take only the time they need


def full_frame_s(shots: list[Shot]) -> float:
    """seconds of the phones' own full-frame video in a scheduled shot list (a phone shot's arrival .. the start of the way out of it)"""
    tot = 0.0
    for k, sh in enumerate(shots):
        if sh.kind != "phone":
            continue
        a = sh.trans["arrive_film"] if sh.trans else sh.f0
        b = shots[k + 1].trans["start_film"] if k + 1 < len(shots) and shots[k + 1].trans else sh.f1
        tot += max(0.0, b - a)
    return tot


def call_spans(segs: list[dict], sh: Shot) -> list[tuple[float, float]]:
    """film [start, end] of every arrow / ring of the shot that answers a call (the arrow appears 0.35 s before the word)"""
    out = []
    for i in sh.segs:
        s = segs[i]
        sp = float(s["speed"])
        for a in s["annotations"]:
            if a.get("calls") and a.get("kind") in ("arrow3d", "vector_arrow", "floor_ring"):
                out.append((s["film"][0] + (a["dance_t"][0] - s["dance"][0]) / sp, s["film"][0] + (a["dance_t"][1] - s["dance"][0]) / sp))
    return sorted(out)


def film_cuts(segs: list[dict], sh: Shot, cuts: list[float]) -> list[float]:
    """the speech's cut points (master s) inside the shot's locked segments, as film seconds"""
    out = []
    for i in sh.segs:
        s = segs[i]
        if not _locked(s):
            continue
        off = s["film"][0] - s["dance"][0]
        out += [c + off for c in cuts if s["dance"][0] + 1e-3 < c < s["dance"][1] - 1e-3]
    return sorted(out)


def split_at_film(segs: list[dict], f: float) -> bool:
    """cut the locked segment that contains film second f into two (nothing to do at an existing boundary). False = f lies in a replay / slowed one."""
    for i, s in enumerate(segs):
        if s["film"][0] + 1e-3 < f < s["film"][1] - 1e-3:
            if not _locked(s):
                return False
            a, b = split_segment(s, s["dance"][0] + (f - s["film"][0]))
            segs[i:i + 1] = [a, b]
            return True
    return True


def chapters_start_shots(segs: list[dict], chapter_ids: set[str]) -> bool:
    """every chapter's anchor segment still starts a shot (its card and its first words open a new view)"""
    first = {segs[sh.segs[0]]["id"] for sh in shots_of(segs)}
    return all(c in first for c in chapter_ids if any(s["id"] == c for s in segs))


def pov_run_at(shots: list[Shot], k: int) -> int:
    """the number of phone shots in a row round shot k (itself included)"""
    a = b = k
    while a - 1 >= 0 and shots[a - 1].kind == "phone":
        a -= 1
    while b + 1 < len(shots) and shots[b + 1].kind == "phone":
        b += 1
    return b - a + 1


def min_view_s(sh: Shot, segs: list[dict]) -> float:
    """the shortest a 3D view may get when it gives time to a phone: the director's own minimum for its kind"""
    dc = V3_DIRCFG
    if sh.mode == "physics_3q":
        return dc.phys_min_s
    if sh.mode in ("feature_zoom", "feet_closeup", "rhythm"):
        return dc.zoom_min_s
    return dc.calm_min_s


def give_phones_time(segs: list[dict], geo: Geo, cfg: FlowCfg, pov: director.PovSource | None, cuts: list[float], chapter_ids: set[str]) -> tuple[list[dict], list[str]]:
    """v3.2 (owner 2026-10-21 "~50% phone"): the 3D views keep only the time they need; what they give goes to the phone shot beside them.
      1. a calm orbit or a zoom with no call of its own (its call's arrow is in the shot before) next to a phone becomes part of that phone shot (at most
         `max_pov_run` phone shots in a row, the phone shot at most `give_phone_cap_s`); where that phone cannot take it (a chapter opens there, it
         cannot show the window) it becomes a phone shot of its own: the phone that shows the window best and is nearest the next view (never the
         phone of a neighbour);
      2. the opening orbit keeps the title card on a still 3D view (the way into the first phone starts after the card is out) and hands the rest to
         the first phone (or, where that phone cannot show it, to a phone shot of its own as in 1);
      3. a view that answers calls (physics view, zoom; never a pivot, the replay or an overhead) ends `give_after_arrow_s` after its last call's
         arrow has gone (+ its way into the next phone; never shorter than the director's minimum for its kind) and starts `give_lead_s` before its
         first call's arrow (the way out of the phone then still arrives 0.1 s before the word: `place`), at the speech's own pauses.
    Each move is tried on a copy and kept only when every view still holds its minimum (`schedule` finds no problem), every chapter still opens a
    shot, and the phones' full-frame video grows. Returns (segments, log)."""
    log: list[str] = []
    rejected: list[str] = []
    if pov is None:
        return segs, log

    def state(ss):
        sh = shots_of(ss)
        plan_azimuths(sh, ss, geo, cfg)
        pr = schedule(sh, ss, geo, cfg)
        return sh, pr, full_frame_s(sh)

    shots, probs, ff = state(segs)
    if probs:
        log.append(f"not tried: the schedule has {len(probs)} problems")
        return segs, log

    def accept(trial, what):
        nonlocal segs, shots, probs, ff
        renumber(trial)
        sh2, pr2, ff2 = state(trial)
        if pr2 or ff2 < ff + 0.25 or not chapters_start_shots(trial, chapter_ids):
            rejected.append(f"{what}: " + (f"{len(pr2)} hold problems {pr2[:2]}" if pr2 else f"full frame {ff2 - ff:+.2f} s" if ff2 < ff + 0.25 else "a chapter would not open a shot"))
            return False
        log.append(f"{what}: full-frame phone video +{ff2 - ff:.2f} s")
        segs, shots, probs, ff = trial, sh2, pr2, ff2
        return True

    def give(sh_gone_idx: int, phone_k: int, f_cut: float | None, what: str) -> bool:
        """the part of shot `sh_gone_idx` on the phone's side of film f_cut (the whole shot when None) joins phone shot `phone_k`"""
        trial = copy.deepcopy(segs)
        g, p = shots[sh_gone_idx], shots[phone_k]
        lo, hi = (g.f0, g.f1) if f_cut is None else ((f_cut, g.f1) if phone_k > sh_gone_idx else (g.f0, f_cut))
        if max(hi, p.f1) - min(lo, p.f0) > cfg.give_phone_cap_s + 1e-6:
            rejected.append(f"{what}: the phone shot would be {max(hi, p.f1) - min(lo, p.f0):.1f} s")
            return False
        if f_cut is not None and not split_at_film(trial, f_cut):
            return False
        gone = [i for i, s in enumerate(trial) if s["film"][0] >= lo - 1e-3 and s["film"][1] <= hi + 1e-3 and not s.get("replay")]
        surv = [i for i, s in enumerate(trial) if s["film"][0] >= p.f0 - 1e-3 and s["film"][1] <= p.f1 + 1e-3]
        if not gone or not surv or any(not _locked(trial[i]) for i in gone):
            return False
        if not join_idx(trial, surv, gone, "phone", p.phone, pov):
            rejected.append(f"{what}: phone {p.phone} cannot show {min(lo, p.f0):.1f}-{max(hi, p.f1):.1f} (film)")
            return False
        for i in gone:
            trial[i]["v3_gave_to_phone"] = what
        return accept(trial, what)

    def own_phone(k: int, f_cut: float | None, what: str, check=None) -> bool:
        """the part of 3D shot k after film f_cut (the whole shot when None) becomes a phone shot of its own: among the phones that can show the window
        (none of the neighbours' phones), the best score less the turn to the next view's azimuth (a short way on)"""
        g = shots[k]
        lo = g.f0 if f_cut is None else f_cut
        d_lo, d_hi = dance_at(segs, g, lo), g.d1
        nbp = {shots[j].phone for j in (k - 1, k + 1) if 0 <= j < len(shots) and shots[j].kind == "phone"}
        nxt_az = shots[k + 1].a0 if k + 1 < len(shots) else None
        prv_az = shots[k - 1].a1 if k > 0 and shots[k - 1].kind == "phone" else None
        cands = []
        for cam, ph in pov.phones.items():
            if cam in nbp or cam not in geo.phones:
                continue
            q = rp.window_quality(ph, d_lo, d_hi, pov.bad.get(cam, []), pov.cfg)
            if q is None:
                continue
            turn = abs(wrap180(geo.phone(cam, d_hi)["az"] - nxt_az)) if nxt_az is not None else 0.0
            turn_in = abs(wrap180(geo.phone(cam, d_lo)["az"] - prv_az)) if prv_az is not None else 0.0
            if max(turn, turn_in) > cfg.own_phone_max_turn_deg:
                rejected.append(f"{what}: phone {cam} is {max(turn, turn_in):.0f} deg round from its phone neighbour (a whip, not worth it)")
                continue
            cands.append((float(q["score"]) - 0.3 * turn / 180.0, cam, q))
        for _, cam, q in sorted(cands, key=lambda x: -x[0]):
            trial = copy.deepcopy(segs)
            if f_cut is not None and not split_at_film(trial, f_cut):
                return False
            idx = [i for i, s in enumerate(trial) if s["film"][0] >= lo - 1e-3 and s["film"][1] <= g.f1 + 1e-3 and not s.get("replay")]
            if not idx or any(not _locked(trial[i]) for i in idx):
                return False
            tpl = next((segs[shots[j].segs[0]] for j in (k + 1, k - 1) if 0 <= j < len(shots) and shots[j].kind == "phone"), None)
            if tpl is None:
                return False
            cam_block = {"mode": "camera_tour", "cameras": [{"camera": cam, "dance_t": [round(d_lo, 3), round(d_hi, 3)]}], "cut": "flow", "blend_s": 0.0,
                         "composite": "3d_over_video", "skeletons_on_top": True, "scene_geometry": "all", "crop": rp.crop_track(pov.phones[cam], d_lo, d_hi, pov.cfg)}
            lay = copy.deepcopy(tpl["layers"])
            lay["source_video"] = cam
            for i in idx:
                sg = trial[i]
                sg["v3_merged_from"] = sg["camera"].get("mode")
                sg["camera"] = copy.deepcopy(cam_block)
                sg["layers"] = copy.deepcopy(lay)
                sg["view_state"] = "camera_tour"
                sg["treatment"] = {"kind": "pov", "phone": cam, "cover": q.get("cover"), "height": q.get("height"), "family": (sg.get("treatment") or {}).get("family"),
                                   "tail_of": None, "v3": "own phone shot (v3.2)"}
                sg["v3_gave_to_phone"] = f"{what} -> phone {cam} (score {q['score']:.2f})"
            if check is not None and not check(trial):
                rejected.append(f"{what} -> phone {cam}: {check.__doc__}")
                continue
            if accept(trial, f"{what} -> its own phone shot {cam} (score {q['score']:.2f})"):
                return True
        if not cands and not any(x.startswith(what) for x in rejected):
            rejected.append(f"{what}: no other phone can show {d_lo:.1f}-{d_hi:.1f} (master)")
        return False

    def run_ok(k: int, joined: bool) -> bool:
        """at most max_pov_run phone shots in a row once shot k is a phone shot (joined into a neighbour: the two are one shot)"""
        left = next((n for n in range(k) if shots[k - 1 - n].kind != "phone"), k)
        right = next((n for n in range(len(shots) - k - 1) if shots[k + 1 + n].kind != "phone"), len(shots) - k - 1)
        return left + right + (0 if joined else 1) <= V3_DIRCFG.max_pov_run

    # 1. whole calm / call-less 3D shots next to a phone
    k = 1
    while k < len(shots):
        sh = shots[k]
        if sh.kind == "3d" and not sh.replay and sh.imp < PROTECT and not segs[sh.segs[0]].get("pivot") and not call_spans(segs, sh):
            done = False
            nb_phone = any(0 <= j < len(shots) and shots[j].kind == "phone" for j in (k - 1, k + 1))
            for j in (k + 1, k - 1):
                if 0 <= j < len(shots) and shots[j].kind == "phone" and not shots[j].replay and run_ok(k, True):
                    if give(k, j, None, f"{sh.mode} {sh.f0:.1f}-{sh.f1:.1f} ({sh.dur:.1f} s, no call of its own) joined phone {shots[j].phone} "
                                        f"{shots[j].f0:.1f}-{shots[j].f1:.1f}"):
                        done = True
                        break
            if not done and nb_phone and run_ok(k, False) and sh.dur >= V3_DIRCFG.zoom_min_s:
                done = own_phone(k, None, f"{sh.mode} {sh.f0:.1f}-{sh.f1:.1f} ({sh.dur:.1f} s, no call of its own)")
            if not done:
                # the part of it the phone beside it can show (the 3D view keeps the director's minimum for its kind)
                mn = min_view_s(sh, segs)
                for j in (k + 1, k - 1):
                    if done or not (0 <= j < len(shots) and shots[j].kind == "phone" and not shots[j].replay):
                        continue
                    cs = film_cuts(segs, sh, cuts)
                    cs = [c for c in cs if sh.f0 + mn <= c <= sh.f1 - cfg.give_min_s] if j == k + 1 else                         [c for c in reversed(cs) if sh.f0 + cfg.give_min_s <= c <= sh.f1 - mn]
                    for c in cs:
                        g = (sh.f1 - c) if j == k + 1 else (c - sh.f0)
                        if give(k, j, c, f"{sh.mode} {sh.f0:.1f}-{sh.f1:.1f} (no call of its own) hands {g:.1f} s to phone {shots[j].phone} "
                                         f"{shots[j].f0:.1f}-{shots[j].f1:.1f}"):
                            done = True
                            break
            if done:
                continue          # the shot list changed: look at the same place again
        k += 1
    # 2. the opening: the title card on a still 3D view, then the first phone
    if len(shots) > 1 and shots[0].kind == "3d" and shots[1].kind == "phone":
        s0 = segs[shots[0].segs[0]]
        card = (s0.get("chapter_card") or {}).get("film_out", 0.0) + s0["film"][0]
        T = shots[1].trans["duration_s"]
        lo = max(card, V3_DIRCFG.first_shot_3d_s) + T + cfg.give_after_arrow_s
        def card_still(trial):
            """the way into the first phone would start before the title card is out"""
            sh2 = shots_of(trial)
            plan_azimuths(sh2, trial, geo, cfg)
            schedule(sh2, trial, geo, cfg)
            return len(sh2) > 1 and sh2[1].trans is not None and sh2[1].trans["start_film"] >= card - 1e-3

        done = False
        for c in film_cuts(segs, shots[0], cuts):
            if lo <= c <= shots[0].f1 - cfg.give_min_s and give(0, 1, c, f"the opening {shots[0].mode} keeps the title card (out at {card:.1f} s) "
                                                                     f"and hands {shots[0].f1 - c:.1f} s to phone {shots[1].phone}"):
                done = True
                break
        if not done and run_ok(0, False):
            for c in film_cuts(segs, shots[0], cuts):
                if max(card, V3_DIRCFG.first_shot_3d_s) + 0.8 <= c <= shots[0].f1 - V3_DIRCFG.zoom_min_s and \
                        own_phone(0, c, f"the opening {shots[0].mode} keeps the title card (out at {card:.1f} s) and hands {shots[0].f1 - c:.1f} s", card_still):
                    break
    # 3. the tails and the lead-ins of the views that answer calls
    k = 0
    while k < len(shots):
        sh = shots[k]
        calls = call_spans(segs, sh)
        if sh.kind != "3d" or sh.replay or segs[sh.segs[0]].get("pivot") or not calls:
            k += 1
            continue
        mn = min_view_s(sh, segs)
        if k + 1 < len(shots) and shots[k + 1].kind == "phone" and shots[k + 1].trans:
            T = shots[k + 1].trans["duration_s"]
            dep = max(max(c[1] for c in calls) + cfg.give_after_arrow_s, sh.arrive + cfg.hold_s["3d"] + 0.05)
            lo = max(dep + T, sh.f0 + mn)
            for c in film_cuts(segs, sh, cuts):
                if lo <= c <= sh.f1 - cfg.give_min_s and give(k, k + 1, c, f"{sh.mode} {sh.f0:.1f}-{sh.f1:.1f} ends at {c:.1f} s ({sh.f1 - c:.1f} s after its "
                                                                         f"last call's arrow + way out) -> phone {shots[k + 1].phone}"):
                    break
            sh = shots[k]
            calls = call_spans(segs, sh)
        if k > 0 and shots[k - 1].kind == "phone" and not any(segs[i].get("chapter") or segs[i].get("chapter_card") for i in sh.segs[:1]):
            first = calls[0][0]
            hi = first - cfg.give_lead_s
            for c in reversed(film_cuts(segs, sh, cuts)):
                if sh.f0 + cfg.give_min_s <= c <= hi and sh.f1 - c >= mn and give(k, k - 1, c, f"{sh.mode} {sh.f0:.1f}-{sh.f1:.1f} starts at {c:.1f} s "
                                                                                     f"({first - c:.2f} s before its first call's arrow) <- phone {shots[k - 1].phone}"):
                    break
        k += 1
    log += [f"(not taken) {x}" for x in rejected]
    return segs, log


def frustum_envelope(role: str, typ: str, a0: float, T: float, cfg: FlowCfg) -> list[float]:
    """the film seconds [a, b, c, d] of a phone's frustum surface on a way in (role enter) or out (leave): alpha 0 -> 1 over [a, b], 1 over [b, c],
    1 -> 0 over [c, d]. Into a phone it stands in the world before the flight starts and fills the frame at the arrival; out of a phone it is opaque
    through the way out (`frustum_leave_full` of it) and fades after; between two phones both stand while the camera crosses."""
    a1 = a0 + T
    if role == "enter":
        f = [a0 + 0.05 * T, a0 + 0.3 * T, a1, a1] if typ == "phone_to_phone" else [a0 - cfg.frustum_fade_s, a0 + 0.1, a1, a1]
    else:
        f = [a0, a0, a0 + 0.65 * T, a0 + 0.95 * T] if typ == "phone_to_phone" else [a0, a0, a0 + cfg.frustum_leave_full * T, a1 + cfg.frustum_fade_s]
    return [round(x, 3) for x in f]


def axis_part(typ: str, cfg: FlowCfg) -> float:
    """the share of a phone way spent on ONE phone's optical axis (a phone_to_phone has two, half each)"""
    return cfg.axis_share / 2.0 if typ == "phone_to_phone" else cfg.axis_share


def frustum_readable(role: str, typ: str, film: list[float], start: float, T: float, cfg: FlowCfg) -> float:
    """seconds the surface reads as a picture standing in the 3D world: alpha >= 0.5 and the eye not yet on (enter) / no longer on (leave) the
    phone's optical axis, where the surface grows to fill the frame"""
    a, b, c, d = film
    on, off = 0.5 * (a + b), 0.5 * (c + d)
    if role == "enter":
        return max(0.0, min(off, start + T * (1.0 - axis_part(typ, cfg))) - on)
    return max(0.0, off - max(on, start + T * axis_part(typ, cfg)))


def locked_dance(segs: list[dict], f: float, fallback: float) -> float:
    """the master second a locked (speed 1, not replayed) segment shows at film second f; `fallback` where no such segment is there"""
    for s in segs:
        if _locked(s) and s["film"][0] - 1e-6 <= f <= s["film"][1] + 1e-6:
            return s["dance"][0] + (f - s["film"][0])
    return fallback


def write_flow(segs: list[dict], shots: list[Shot], geo: Geo, cfg: FlowCfg) -> dict:
    """the flow keys into the camera blocks (identical on every segment of a shot): world_azimuth_deg / drift_deg_s / inline_lift_max_m for 3D shots,
    up_azimuth_deg for an overhead, transition_in for every shot but the first; the frustum envelopes (top-level list). Returns the flow block."""
    frustums: list[dict] = []
    for k, sh in enumerate(shots):
        cam = copy.deepcopy(segs[sh.segs[0]]["camera"])
        cam.pop("transition_in", None)
        if sh.kind == "3d":
            cam["world_azimuth_deg"] = round(sh.a0, 1)
            cam["drift_deg_s"] = round(sh.drift, 2)
            if cam.get("mode") == "orbit":
                cam["inline_lift_max_m"] = cfg.inline_lift_max_m
            cam["ground_level"] = True
        elif sh.kind == "overhead":
            cam["up_azimuth_deg"] = round(sh.a0, 1)
            cam["up_policy"] = "from_entry"
        elif sh.kind == "phone" and cam.get("mode") == "camera_tour":
            cam["cut"] = "flow"
        elif sh.kind == "phone" and cam.get("mode") == "fly_to_camera":
            # v3: the way in is the transition; the shot itself is the phone's view (the old fly / fade / pull-back keys go)
            ph = sh.phone
            crop = cam.get("crop")
            cam = {"mode": "camera_tour", "cameras": [{"camera": ph, "dance_t": [round(sh.d0, 3), round(sh.d1, 3)]}], "cut": "flow", "blend_s": 0.0,
                   "composite": "3d_over_video", "skeletons_on_top": True, "scene_geometry": "all", "crop": crop}
            for i in sh.segs:
                segs[i]["v3_was"] = "fly_to_camera"
        if sh.kind == "phone" and cam.get("mode") == "camera_tour" and len(cam.get("cameras") or []) == 1:
            # v3.2: the phone's camera is live from the start of its way in to the end of its way out (the transition paths fly from / to its
            # live pose; the POV check asks for its pose and video there - clips_needed covers it). Before, the director's window stayed on
            # the block after a carve / nudge had shortened the shot (ch06_06_pov: 192.86-198.08 s for a shot that ends at 193.82 s).
            a = sh.trans["start_film"] if sh.trans else sh.f0
            nt = shots[k + 1].trans if k + 1 < len(shots) else None
            b = nt["start_film"] + nt["duration_s"] if nt else sh.f1
            da, db = locked_dance(segs, a, sh.d0), locked_dance(segs, b, sh.d1)
            cam["cameras"] = [{"camera": cam["cameras"][0]["camera"], "dance_t": [round(min(da, sh.d0), 3), round(max(db, sh.d1), 3)]}]
        if sh.trans is not None:
            p = shots[k - 1]
            t = sh.trans
            typ = t["type"]
            T = t["duration_s"]
            block = {"type": typ, "start_s": round(t["start_film"] - sh.f0, 3), "duration_s": T, "easing": cfg.easing,
                     "from": {"kind": p.kind, "mode": p.mode, "phone": p.phone, "azimuth_deg": round(p.a1, 1), "radius_m": round(p.r1, 2), "height_m": round(p.h1, 2)},
                     "to": {"kind": sh.kind, "mode": sh.mode, "phone": sh.phone, "azimuth_deg": round(sh.a0, 1), "radius_m": round(sh.r0, 2), "height_m": round(sh.h0, 2)},
                     "arc": {"delta_deg": t["daz"], "dir": "ccw" if t["daz"] > 0 else "cw" if t["daz"] < 0 else "none"},
                     "constraints": {"height_m": list(cfg.ground_m), "clear_radius_m": cfg.clear_radius_m, "keep_couple_in_view": True,
                                     "look": "the couple (both ends look at it; the aim never leaves it mid-flight)"},
                     "rule_s": t["rule_s"]}
            if t["squeezed"]:
                block["squeezed"] = t["squeezed"]
            if typ == "flight":
                block["path"] = "arc"                                  # cylindrical about the couple: azimuth the short way, radius / height / look / FOV lerp
            elif typ in ("crane_up", "crane_down"):
                block["path"] = "crane"                                # a vertical arc about the look point at the entry (exit) azimuth: no roll at the top
                block["constraints"]["height_m"] = [cfg.ground_m[0], None]
            elif typ in ("phone_in", "phone_out", "phone_to_phone"):
                block["path"] = {"phone_in": "arc_then_axis", "phone_out": "axis_then_arc", "phone_to_phone": "axis_arc_axis"}[typ]
                block["axis_run_m"] = cfg.axis_run_m
                block["axis_share"] = cfg.axis_share
                if typ == "phone_in" and p.kind == "overhead" or typ == "phone_out" and sh.kind == "overhead":
                    block["crane"] = True
                    block["constraints"]["height_m"] = [cfg.ground_m[0], None]
            elif typ in ("replay_in", "replay_out"):
                block["path"] = "arc"
                block["from_pose"] = "frozen"                          # the previous shot's last pose (the dance time jumps at the segment start)
                block["flash_s"] = cfg.flash_s
            # the phones' frustums: shown through the way in / out
            fin = fout = None
            a0, a1 = t["start_film"], t["start_film"] + T
            if typ in ("phone_in", "phone_to_phone"):
                ph = sh.phone
                # appears before the flight starts (it is part of the world the camera flies through), fills the frame at the arrival
                fin = {"camera": ph, "film": frustum_envelope("enter", typ, a0, T, cfg), "role": "enter",
                       "depth_m": cfg.frustum_depth_m, "video_full_at_film": round(a1, 3), "shot": segs[sh.segs[0]]["id"]}
            if typ in ("phone_out", "phone_to_phone"):
                ph = p.phone
                fout = {"camera": ph, "film": frustum_envelope("leave", typ, a0, T, cfg),
                        "role": "leave", "depth_m": cfg.frustum_depth_m, "video_full_until_film": round(a0, 3), "shot": segs[p.segs[0]]["id"]}
            if fin:
                frustums.append(fin)
                block["frustum_in"] = {"camera": fin["camera"], "film": fin["film"]}
            if fout:
                frustums.append(fout)
                block["frustum_out"] = {"camera": fout["camera"], "film": fout["film"]}
            if typ == "phone_in" or typ == "phone_to_phone":
                block["video"] = {"full_frame_at_s": round(t["arrive_film"] - sh.f0, 3), "splats_hidden_at_s": round(t["arrive_film"] - sh.f0, 3),
                                  "overlays_on_top_in_s": 0.35}
            if typ == "phone_out" or typ == "phone_to_phone":
                block["video_out"] = {"surface_from_s": round(t["start_film"] - sh.f0, 3), "splats_back_in_s": 0.35, "overlays_on_top_off_before_s": 0.25}
            cam["transition_in"] = block
            cam["blend_s"] = 0.0 if sh.kind == "phone" else round(T, 2)
        for i in sh.segs:
            segs[i]["camera"] = copy.deepcopy(cam)
    return {"version": 1, "frustums": frustums}


# ------------------------------------------------------------------------------------------------ (4) the physics layer


HAND_EVENTS = WORK / "v3" / "hand_contact" / "events.json"


def file_sha(p: Path | None) -> str | None:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] if p is not None and Path(p).exists() else None


def connected_series(n: int, hand_events: Path | None, cb: dict) -> tuple[np.ndarray, str]:
    """per 30 Hz frame: a hand on the partner (v3/hand_contact/events.json, holds and touches) or the counterbalance connection score >= 0.5.
    The build records the events file's sha256 (recap_film.v3.connected_events); `check` says when the file has changed since (re-run `build`)."""
    con = np.zeros(n, bool)
    src = []
    if hand_events is not None and Path(hand_events).exists():
        d = json.loads(Path(hand_events).read_text(encoding="utf-8"))
        for e in d.get("events", []):
            con[int(e["i0"]):int(e["i1"]) + 1] = True
        src.append(f"{Path(hand_events).relative_to(WORK) if WORK in Path(hand_events).parents else hand_events} ({len(d.get('events', []))} events)")
    tr = (cb.get("track") or {}).get("connection")
    if tr:
        c = np.asarray(tr, float)[:n] >= 0.5
        con[:len(c)] |= c
        src.append("counterbalance.json track.connection >= 0.5")
    return con, " + ".join(src) or "none"


PHYS_KEYS = {"skeleton_mode": "physics", "physics": True, "balance": True, "legend": LEGEND}


def apply_physics(segs: list[dict], act: plan.Activity, con: np.ndarray, cfg: FlowCfg, pcfg: plan.Cfg) -> list[dict]:
    """the physics layer on every segment where the teachers demonstrate (move) or are connected, and do not stand apart: the load colours of the
    skeleton (tension orange / compression blue in arms and legs), the balance axes, and each dancer's centre of mass with its floor dot (com_marker:
    lead + follow; the couple's too where a counterbalance is). Segments that already are a physics view keep theirs. Returns the per-segment rows."""
    rows = []
    for s in segs:
        d0, d1 = s["dance"]
        w = act.window(d0, d1, pcfg)
        i0, i1 = int(round(d0 * 30)), max(int(round(d0 * 30)) + 1, int(round(d1 * 30)))
        cshare = float(con[i0:i1].mean()) if i1 > i0 and i0 < len(con) else 0.0
        moving = bool(w.get("known")) and w["moving_frac"] >= cfg.phys_moving_frac
        apart = bool(w.get("apart"))
        was = s["layers"].get("skeleton_mode") == "physics"
        had_dots = any(a["kind"] == "com_marker" for a in s["annotations"])
        on = was or (not apart and (moving or cshare >= cfg.phys_connected_min))
        why = "already a physics view" if was else ("apart" if apart else ("demonstration" if moving else "") + (" + " if moving and cshare >= cfg.phys_connected_min else "") +
                                                     ("connected" if cshare >= cfg.phys_connected_min else "")) or "standing, not connected"
        if on:
            for k, v in PHYS_KEYS.items():
                s["layers"][k] = v
            if s["camera"].get("mode") == "overhead_floorcraft":
                s["layers"]["physics_from_above"] = True          # the floor dots carry it from above; the coloured lines are thin there
            if not had_dots and d1 - d0 >= 1.0:
                tg = [{"type": "com", "dancer": "lead"}, {"type": "com", "dancer": "follow"}]
                if s["layers"].get("counterbalance"):
                    tg.append({"type": "couple_com"})
                a0, a1 = round(d0 + 0.15, 3), round(d1 - 0.1, 3)
                s["annotations"].append({"id": f"{s['id']}_coms", "kind": "com_marker", "dance_t": [a0, a1], "targets": tg, "v3": "physics",
                                         "film_in_segment": [round((a0 - d0) / float(s["speed"]), 3), round((a1 - d0) / float(s["speed"]), 3)]})
            s["layers"]["physics_limbs"] = {"width_scale": 1.5, "parts": ["arms", "legs"]}
        rows.append({"id": s["id"], "film": s["film"], "mode": s["camera"].get("mode"), "on": on, "was": was, "dots_before": had_dots,
                     "moving": moving, "connected": round(cshare, 2), "apart": apart, "why": why})
    return rows


# ------------------------------------------------------------------------------------------------ the pivot shots


def pivot_segment(ph: dict, tgt: dict, idx: int, cfg: FlowCfg, pcfg: plan.Cfg) -> dict:
    cam = {"mode": "physics_3q", "azimuth_deg": 35, "drift_deg_s": 3, "distance_m": cfg.pivot_distance_m, "elevation_deg": cfg.pivot_elevation_deg, "fit_band": True,
           "top_m": 1.95, "band_lo": 0.34, "band_hi": 0.74, "look_target": tgt["target"], "look_target_weight": 0.35, "blend_s": round(pcfg.blend_s, 2),
           "shot_key": f"pivot{idx:02d}"}           # two pivot views next to each other stay two shots (the viewer joins equal camera blocks)
    lay = plan.treatment_layers("weight", plan.Layers.all_on(), None, "", 0, pcfg.avatar_opacity)
    lay.update({"counterbalance": True, "floorcraft": "pivot_dials", "floorcraft_labels": "current", "balance": True, "skeleton_mode": "physics", "physics": True,
                "legend": LEGEND})
    a0 = max(ph["t0"] - 0.1, ph["axis_t"] - 0.35)
    annos = [{"id": f"pivot{idx}_ar", "kind": "arrow3d", "text": PIVOT_LABEL, "target": tgt["target"], "dance_t": [round(a0, 3), round(a0 + 2.4, 3)],
              "calls": [f"axis@{ph['axis_t']:.2f}"], "why": tgt["why"]},
             {"id": f"pivot{idx}_coms", "kind": "com_marker", "dance_t": None, "targets": [{"type": "com", "dancer": "lead"}, {"type": "com", "dancer": "follow"}, {"type": "couple_com"}]}]
    return {"id": None, "speed": 1.0, "camera": cam, "view_state": "physics", "layers": lay, "annotations": annos, "audio": "locked", "status": "ok", "narration": [],
            "director": True, "pivot": {"phrase": ph["words"], "utt": ph["utt"], "axis_t": ph["axis_t"], "target": tgt["target"], "interval": tgt["interval"], "why": tgt["why"]},
            "treatment": {"kind": "zoom", "family": "pivot"},
            "notes": f"v3 pivot: '{ph['words']}' ({ph['speaker']}, {ph['axis_t']:.1f} s) = the counterbalance pivot point, not the leader's T ({tgt['why']})"}


FLOOR_TAGS = ("overlay_T_axis", "overlay_path/floorcraft")


def pivot_window(ph: dict, events: list[dict], cuts: list[float], cfg: FlowCfg) -> tuple[float, float, list[str]]:
    """[a, b] master seconds of the pivot shot of one phrase: from just before its verb to `pivot_post_s` after 'axis' (snapped to a pause), but
    never over a neighbouring call that really means the leader's T or the floor path ('rotate', 'side to side', 'exit': they keep the T);
    a window squeezed by them below `pivot_min_s` grows backwards into its own sentence (still clear of the previous T call)"""
    why = []
    a = ph["t0"] - cfg.pivot_pre_s
    b = director.snap(ph["axis_end"] + cfg.pivot_post_s, cuts, 0.6, ph["axis_end"] + 0.9, ph["axis_end"] + 2.4)
    floor = [e for e in events if set(e.get("tags", [])) & set(FLOOR_TAGS) and float(e.get("conf") or 0) >= 0.5]
    nxt = sorted((float(e["start"]), e) for e in floor if float(e["start"]) > ph["axis_t"] + 0.05)
    prv = [float(e["start"]) for e in floor if float(e["start"]) < ph["t0"] - 0.05]
    hi = nxt[0][0] - 0.2 if nxt else 1e9
    lo = max(prv) + 0.3 if prv else -1e9
    if b > hi:
        why.append(f"ends at {hi:.2f} s: the next floor call ('{nxt[0][1]['triggers'][0]}', {nxt[0][0]:.2f} s) keeps the leader's T")
        b = hi
    if a < lo:
        why.append(f"starts at {lo:.2f} s: clear of the previous floor call")
        a = lo
    if b - a < cfg.pivot_min_s:
        a2 = max(lo, b - cfg.pivot_min_s)
        if a2 < a:
            why.append(f"starts {a - a2:.2f} s earlier (inside its sentence) to hold {b - a2:.1f} s")
            a = a2
    return round(a, 3), round(b, 3), why


def carve_pivots(segs: list[dict], phrases: list[dict], events: list[dict], cb: dict, cuts: list[float], cfg: FlowCfg, pcfg: plan.Cfg) -> tuple[list[dict], list[dict]]:
    """one ground-level counterbalance-pivot shot per back-to-the-axis phrase; the shots it replaces are recorded (old). Arrows of the replaced pieces
    that do not need the leader's T are kept (at most two call-outs at a time, the pivot arrow first)."""
    rows = []
    for n, ph in enumerate(phrases, 1):
        a, b, why = pivot_window(ph, events, cuts, cfg)
        tgt = pivot_target(cb, ph["axis_t"], cfg.pivot_search_s)
        inside = [s for s in segs if s["dance"][0] < b and s["dance"][1] > a and not s.get("replay")]
        old = [{"id": s["id"], "mode": s["camera"].get("mode"), "phone": phone_of(s["camera"]),
                "arrows": [(an.get("text"), (an.get("target") or {}).get("type")) for an in s["annotations"] if an.get("kind") in ("arrow3d", "vector_arrow")]}
               for s in inside]
        carried = [copy.deepcopy(an) for s in inside for an in s["annotations"]
                   if an.get("kind") in ("arrow3d", "vector_arrow", "floor_ring") and (an.get("target") or {}).get("type") not in ("leader_t", "dial")
                   and not (str(an.get("text", "")).lower() == "axis") and not an.get("continuous")]
        new = pivot_segment(ph, tgt, n, cfg, pcfg)
        segs = carve(segs, a, b, new)
        for s in segs:
            if s.get("pivot") and s["id"] is None:
                s["id"] = f"pivot{n:02d}_{ph['utt']}"
                for an in s["annotations"]:
                    if an["kind"] == "com_marker" and an["dance_t"] is None:
                        an["dance_t"] = [round(a + 0.2, 3), round(b - 0.1, 3)]
                    an["dance_t"] = [max(an["dance_t"][0], round(a + 0.05, 3)), min(an["dance_t"][1], round(b - 0.05, 3))]
                piv = next(an for an in s["annotations"] if an["id"].endswith("_ar"))
                kept = []
                for an in sorted(_clip_annos(carried, s), key=lambda x: x["dance_t"][0]):
                    live = [x for x in [piv] + kept if x["dance_t"][0] < an["dance_t"][1] - 0.05 and x["dance_t"][1] > an["dance_t"][0] + 0.05]
                    if len(live) + 1 <= pcfg.max_arrows_at_once:
                        an["id"] = f"{s['id']}_{an['id']}"
                        kept.append(an)
                s["annotations"] += kept
                s["pivot"]["kept_arrows"] = [x.get("text") for x in kept]
                refilm_annos(s)
        # a T arrow labelled 'axis' left in a neighbour within the phrase's sentence: it meant the pivot
        for s in segs:
            s["annotations"] = [an for an in s["annotations"] if not ((an.get("target") or {}).get("type") == "leader_t" and str(an.get("text", "")).lower() == "axis"
                                                                       and an["dance_t"][0] < ph["t1"] + 0.5 and an["dance_t"][1] > ph["u_start"] - 0.5)]
        rows.append({**{k: ph[k] for k in ("utt", "speaker", "words", "variant", "t0", "axis_t", "t1", "text")}, "window": [a, b], "window_why": why,
                     "target": tgt, "old": old})
    return segs, rows


def replay_to_ground(segs: list[dict], cb: dict, cfg: FlowCfg, pcfg: plan.Cfg) -> list[str]:
    """the replay was an overhead (the base plan's axis family): at ground level now (no crane in and out round a flash), the counterbalance
    pivot view - the replayed demonstration is a pivot (counterbalance.json) - with the physics layer"""
    out = []
    for s in segs:
        if not s.get("replay"):
            continue
        mid = 0.5 * (s["dance"][0] + s["dance"][1])
        tgt = pivot_target(cb, mid, cfg.pivot_search_s)
        cam = {"mode": "physics_3q", "azimuth_deg": 35, "drift_deg_s": 3, "distance_m": cfg.pivot_distance_m + 0.2, "elevation_deg": cfg.pivot_elevation_deg, "fit_band": True,
               "top_m": 1.95, "band_lo": 0.34, "band_hi": 0.74, "look_target": tgt["target"], "look_target_weight": 0.3, "blend_s": round(pcfg.blend_s, 2)}
        lay = plan.treatment_layers("weight", plan.Layers.all_on(), None, "", 0, pcfg.avatar_opacity)
        lay.update({"counterbalance": True, "floorcraft": "pivot_dials", "floorcraft_labels": "current", "balance": True, "legend": LEGEND})
        out.append(f"replay {s['id']}: {s['camera'].get('mode')} -> ground-level pivot view ({tgt['why']})")
        s["camera"], s["layers"], s["view_state"] = cam, lay, "physics"
        s["v3_replay_was"] = "overhead_floorcraft"
        if not any(a["kind"] == "com_marker" for a in s["annotations"]):
            d0, d1 = s["dance"]
            s["annotations"].append({"id": f"{s['id']}_coms", "kind": "com_marker", "dance_t": [round(d0 + 0.1, 3), round(d1 - 0.1, 3)],
                                     "targets": [{"type": "com", "dancer": "lead"}, {"type": "com", "dancer": "follow"}, {"type": "couple_com"}]})
        refilm_annos(s)
    return out


# ------------------------------------------------------------------------------------------------ build


def renumber(segs: list[dict]) -> None:
    seen: dict[str, int] = {}
    for s in segs:
        if s["id"] in seen:
            seen[s["id"]] += 1
            s["id"] = f"{s['id']}{seen[s['id']]}"
        else:
            seen[s["id"]] = 0


def t_under_t_arrows(segs: list[dict]) -> list[str]:
    out = []
    for s in segs:
        if not any((a.get("target") or {}).get("type") == "leader_t" for a in s["annotations"]):
            continue
        fc = s["layers"].get("floorcraft", "off")
        if fc == "pivot_dials":
            if not s["layers"].get("floorcraft_current_t"):
                s["layers"]["floorcraft_current_t"] = True
                out.append(f"{s['id']}: pivot dials + the current T (a T call inside the pivot shot)")
        elif fc in (None, "off"):
            s["layers"].update({"floorcraft": "current_t", "t_width_m": 0.05, "t_brightness": 1.0})
            out.append(f"{s['id']}: the current T on (its arrow points at it)")
    return out


def fill_layer_timing(segs: list[dict], shots: list[Shot], pcfg: plan.Cfg) -> None:
    """layers follow the VIEW, not the segment boundary: into a phone the 3D layers stay until the picture is the phone's full frame; out of a phone the
    next 3D shot's layers apply from the first frame of the way out; between two phones the flight shows the 3D world with `transit_layers`"""
    for k, sh in enumerate(shots):
        if k == 0:
            continue
        s0 = segs[sh.segs[0]]
        tr = s0["camera"].get("transition_in")
        if not tr:
            continue
        typ = tr["type"]
        arrive_s = round(tr["start_s"] + tr["duration_s"], 3)
        if typ == "phone_in":
            tr["layers_switch_s"] = arrive_s
        elif typ == "phone_out":
            tr["layers_switch_s"] = tr["start_s"]
        elif typ == "phone_to_phone":
            tr["layers_switch_s"] = arrive_s
            transit = copy.deepcopy(plan.L_TALK)
            transit["avatars"] = round(pcfg.avatar_opacity, 3)
            if s0["layers"].get("skeleton_mode") == "physics" or segs[shots[k - 1].segs[-1]]["layers"].get("skeleton_mode") == "physics":
                transit.update(PHYS_KEYS)
            tr["transit_layers"] = {"film": [round(sh.f0 + tr["start_s"], 3), round(sh.f0 + arrive_s, 3)], "layers": transit}
        else:
            tr["layers_switch_s"] = 0.0
        for i in sh.segs:
            segs[i]["camera"]["transition_in"] = copy.deepcopy(tr)


def build(log=print, cfg: FlowCfg | None = None, write: bool = True, hand_events: Path | None = None) -> dict:
    cfg = cfg or FlowCfg()
    hand_events = Path(hand_events) if hand_events else HAND_EVENTS
    paths = fr.Paths()
    pcfg = plan.Cfg()
    sp = fr.load_speech(paths)
    layers = plan.Layers.probe(CAP_DIR, True)
    act = fr.load_activity(paths, CAP_DIR)
    pov = fr.load_pov(paths, "yes")
    phrases = back_to_axis(sp["utterances"])
    events, dropped_ev = drop_axis_events(sp["events"], phrases)
    sp3 = dict(sp, events=events)
    doc, tl, report = fr.build_direction(sp3, act, layers, pcfg, paths, CAPTURE, fr.film_title(), pov, V3_DIRCFG, arrow_ctx=fr.load_arrow_ctx(CAP_DIR, layers))
    base_bad = fr.validate(doc, tl, sp)
    if base_bad:
        raise fr.RecapError("the base direction does not validate:\n  " + "\n  ".join(base_bad[:20]))
    segs = doc["segments"]
    cb = load_counterbalance()
    cuts = director.cut_points(sp["utterances"], pcfg.lead_s)
    segs, pivot_rows = carve_pivots(segs, phrases, events, cb, cuts, cfg, pcfg)
    replay_notes = replay_to_ground(segs, cb, cfg, pcfg)
    renumber(segs)
    geo = Geo()
    chapter_ids = {c["segment"] for c in doc["chapters"]}
    shots, flow_rep = plan_flow(segs, geo, cfg, pov, cuts if cfg.nudge_max_s > 0 else None, chapter_ids)
    if cfg.give_phone_cap_s > 0:
        segs, gave = give_phones_time(segs, geo, cfg, pov, cuts, chapter_ids)
        shots, rep2 = plan_flow(segs, geo, cfg, pov, cuts, chapter_ids)      # (every move kept the schedule clean: no further join)
        flow_rep = {"merges": flow_rep["merges"] + rep2["merges"], "problems": rep2["problems"], "gave": gave}
    flow = write_flow(segs, shots, geo, cfg)
    con, con_src = connected_series(int(round(sp["duration"] * 30)) + 2, hand_events, cb)
    phys_rows = apply_physics(segs, act, con, cfg, pcfg)
    t_notes = t_under_t_arrows(segs)
    shots = shots_of(segs)                        # (the camera blocks are unchanged by the physics pass: the same shots)
    plan_azimuths(shots, segs, geo, cfg)
    schedule(shots, segs, geo, cfg)
    fill_layer_timing(segs, shots, pcfg)
    for s in segs:
        refilm_annos(s)
    doc["segments"] = segs
    doc["flow"] = {**flow, "schema": FLOW_SCHEMA, "no_jump_cuts": True,
                   "defaults": {"easing": cfg.easing, "ground_m": list(cfg.ground_m), "clear_radius_m": cfg.clear_radius_m, "frustum_depth_m": cfg.frustum_depth_m,
                                "axis_run_m": cfg.axis_run_m, "axis_share": cfg.axis_share, "rule": cfg.rule, "squeeze_max_s": cfg.squeeze_max_s, "hold_s": cfg.hold_s,
                                "flash_s": cfg.flash_s, "crane_extra_s": cfg.crane_extra_s, "inline_lift_max_m": cfg.inline_lift_max_m,
                                "arrive_after_s": cfg.arrive_after_s, "phone_arrive_after_s": cfg.phone_arrive_after_s, "depart_before_s": cfg.depart_before_s,
                                "frustum_fade_s": cfg.frustum_fade_s, "frustum_leave_full": cfg.frustum_leave_full, "frustum_readable_s": cfg.frustum_readable_s,
                                "max_arc_deg_s": MAX_ARC_DEG_S, "call_arrive_before_s": cfg.call_arrive_before_s,
                                "give_phone_cap_s": cfg.give_phone_cap_s, "give_min_s": cfg.give_min_s, "give_after_arrow_s": cfg.give_after_arrow_s,
                                "give_lead_s": cfg.give_lead_s, "nudge_margin_s": cfg.nudge_margin_s, "nudge_max_s": cfg.nudge_max_s,
                                "nudge_min_shot_s": cfg.nudge_min_shot_s, "phone_first": cfg.phone_first, "own_phone_max_turn_deg": cfg.own_phone_max_turn_deg}}
    doc["created"] = "generated by dancecap.film_recap_flow build (v3: pivot, physics, flights) on top of dancecap.film_recap; edit the rules there, not this file"
    doc["recap_film"]["v3"] = {"flow_version": "3.2", "gave_to_phones": sum(1 for g in flow_rep.get("gave", []) if not g.startswith("(not taken)")),
                               "back_to_the_axis": len(phrases), "axis_events_dropped": [e["id"] for e in dropped_ev], "joins": len(flow_rep["merges"]),
                               "connected_source": con_src, "connected_events": {"path": str(hand_events), "sha256_16": file_sha(hand_events)}, "dircfg": {k: v for k, v in V3_DIRCFG.__dict__.items() if getattr(director.DirCfg(), k) != v},
                               "shots": len(shots)}
    ids = {s["id"] for s in segs}
    missing = [c for c in doc["chapters"] if c["segment"] not in ids]
    if missing:
        raise fr.RecapError(f"chapter anchors lost by the v3 passes: {missing}")
    v3rep = {"pivots": pivot_rows, "replay": replay_notes, "flow": flow_rep, "physics": phys_rows, "connected_source": con_src, "t_under_arrows": t_notes,
             "axis_events_dropped": dropped_ev, "base_director": report.get("director", {}).get("pov_share")}
    res = {"doc": doc, "tl": tl, "report": report, "v3": v3rep, "shots": shots, "sp": sp, "geo": geo, "cfg": cfg}
    res["checks"] = check(doc, tl, sp, geo, cfg)
    res["physics_quant"] = quantify_physics(doc, json.loads((paths.film / "direction.json").read_text(encoding="utf-8")))
    res["clips"] = clips_needed(doc, geo)
    res["flow_stats"] = flow_stats(doc, cfg)
    if write:
        write_all(res, paths)
    return res


FLOW_SCHEMA = ("hm film flow v1 (docs/RECAP_FILM.md 'v3: flights'): camera.transition_in {type, start_s, duration_s, easing, path, from, to, arc, constraints, "
               "frustum_in / frustum_out, video / video_out, layers_switch_s, transit_layers}, camera.world_azimuth_deg / drift_deg_s / ground_level / "
               "inline_lift_max_m, physics_3q elevation_deg / look_target / look_target_weight, overhead up_azimuth_deg / up_policy, flow.frustums")


# ------------------------------------------------------------------------------------------------ checks


def film_of(doc: dict, dance: float) -> float:
    """the film second a locked master second plays at"""
    for s in doc["segments"]:
        if not s.get("replay") and s["dance"][0] - 1e-6 <= dance <= s["dance"][1] + 1e-6:
            return s["film"][0] + (dance - s["dance"][0]) / float(s["speed"])
    return dance


def check(doc: dict, tl: dict, sp: dict, geo: Geo | None, cfg: FlowCfg) -> dict:
    """the film checks (film_recap.validate) + the v3 checks: no jump cut, every view holds, one easing, the duration rule, ground level, the phones
    entered and left through their frustums, the pivots, the physics legend, captions and the soundtrack's time map unchanged"""
    bad = list(fr.validate(doc, tl, sp))
    notes: list[str] = []
    segs = doc["segments"]
    shots = shots_of(segs)
    n_bound = len(shots) - 1
    no_tr = [segs[sh.segs[0]]["id"] for sh in shots[1:] if not segs[sh.segs[0]]["camera"].get("transition_in")]
    bad += [f"jump cut: no transition_in at {x}" for x in no_tr]
    zero = [segs[sh.segs[0]]["id"] for sh in shots[1:] if (segs[sh.segs[0]]["camera"].get("transition_in") or {}).get("duration_s", 0) <= 0]
    bad += [f"jump cut: a zero-length transition at {x}" for x in zero]
    eas = {(segs[sh.segs[0]]["camera"].get("transition_in") or {}).get("easing") for sh in shots[1:]}
    if eas != {cfg.easing}:
        bad.append(f"the easings are not one: {eas}")
    prev_arrive = 0.0
    holds = []
    for k, sh in enumerate(shots):
        cam = segs[sh.segs[0]]["camera"]
        tr = cam.get("transition_in")
        if k > 0 and tr:
            start = sh.f0 + tr["start_s"]
            arrive = start + tr["duration_s"]
            p = shots[k - 1]
            hold_min = 1.5 if p.replay else cfg.hold_s.get(p.kind, 1.0)
            held = start - prev_arrive
            holds.append({"shot": segs[p.segs[0]]["id"], "kind": p.kind, "hold_s": round(held, 3)})
            if held < hold_min - 2e-3:
                bad.append(f"{segs[p.segs[0]]['id']} ({p.kind}) holds {held:.2f} s < {hold_min} s between its way in and out")
            if start < p.f0 - 1e-3 and not (p.replay or sh.replay):
                bad.append(f"the way into {segs[sh.segs[0]]['id']} starts before the previous shot does")
            typ = tr["type"]
            want = transition_type(p, sh)
            if typ != want:
                bad.append(f"{segs[sh.segs[0]]['id']}: transition {typ}, expected {want}")
            crane = bool(tr.get("crane"))
            rule = rule_duration(typ, tr["arc"]["delta_deg"], tr["to"]["radius_m"] - tr["from"]["radius_m"], 0.0 if crane else tr["to"]["height_m"] - tr["from"]["height_m"], cfg, crane)
            if not (rule - cfg.squeeze_max_s - 2e-3 <= tr["duration_s"] <= rule + 2e-3):
                bad.append(f"{segs[sh.segs[0]]['id']}: {tr['duration_s']} s is off the rule ({rule} s)")
            # ground level: only a crane (into / out of an overhead) leaves the band
            lo, hi = cfg.ground_m
            for end in ("from", "to"):
                h = tr[end]["height_m"]
                if tr[end]["kind"] != "overhead" and not (lo - 1e-6 <= h <= hi + 1e-6):
                    bad.append(f"{segs[sh.segs[0]]['id']}: the {end} eye at {h} m is not at ground level")
            if (p.kind == "overhead" or sh.kind == "overhead") and typ not in ("crane_up", "crane_down") and not crane:
                bad.append(f"{segs[sh.segs[0]]['id']}: an overhead entered / left without the crane")
            peak = 1.875 * abs(tr["arc"]["delta_deg"]) / max(1e-3, tr["duration_s"] * (1 - (cfg.axis_share if "axis" in str(tr.get("path")) else 0)))
            if peak > MAX_ARC_DEG_S:
                bad.append(f"{segs[sh.segs[0]]['id']}: the arc turns at up to {peak:.0f} deg/s")
            # phones through their frustums
            if sh.kind == "phone" and typ in ("phone_in", "phone_to_phone"):
                fi = tr.get("frustum_in") or {}
                if fi.get("camera") != sh.phone:
                    bad.append(f"{segs[sh.segs[0]]['id']}: entered without phone {sh.phone}'s frustum")
            if p.kind == "phone" and typ in ("phone_out", "phone_to_phone"):
                fo = tr.get("frustum_out") or {}
                if fo.get("camera") != p.phone:
                    bad.append(f"{segs[sh.segs[0]]['id']}: phone {p.phone} left without its frustum")
            # the surface must read as a picture in the world on every phone way (owner: it may be shorter, but readable)
            for role, key in (("enter", "frustum_in"), ("leave", "frustum_out")):
                fz = tr.get(key)
                if fz:
                    rd = frustum_readable(role, typ, fz["film"], start, tr["duration_s"], cfg)
                    if rd < cfg.frustum_readable_s - 2e-3:
                        bad.append(f"{segs[sh.segs[0]]['id']}: phone {fz['camera']}'s surface reads only {rd:.2f} s on the way {'in' if role == 'enter' else 'out'} "
                                   f"(< {cfg.frustum_readable_s} s)")
            # the crane is for the overheads only: every other way stays at ground level
            if (typ in ("crane_up", "crane_down") or crane) and not (p.kind == "overhead" or sh.kind == "overhead"):
                bad.append(f"{segs[sh.segs[0]]['id']}: a crane between two ground-level views")
            prev_arrive = arrive
        elif k == 0:
            prev_arrive = sh.f0
        if sh.kind == "phone" and cam.get("cut") != "flow":
            bad.append(f"{segs[sh.segs[0]]['id']}: a phone shot still marked cut '{cam.get('cut')}'")
        if sh.kind != "overhead" and sh.kind != "phone" and not cam.get("ground_level"):
            bad.append(f"{segs[sh.segs[0]]['id']}: a 3D shot without ground_level")
    # the phones' video and the phone shots
    for sh in shots:
        if sh.kind == "phone" and segs[sh.segs[0]]["camera"].get("mode") != "camera_tour":
            bad.append(f"{segs[sh.segs[0]]['id']}: a phone shot that is not a camera_tour (v3: the way in is the transition)")
        # v3.2: the phone's camera window (way in .. way out) covers its shot and lies inside what the phone recorded
        cams_ = segs[sh.segs[0]]["camera"].get("cameras") or []
        if sh.kind == "phone" and cams_:
            w = cams_[0].get("dance_t") or [sh.d0, sh.d1]
            if w[0] > sh.d0 + 1e-3 or w[1] < sh.d1 - 1e-3:
                bad.append(f"{segs[sh.segs[0]]['id']}: the phone's window {w} does not cover its shot {sh.d0:.2f}-{sh.d1:.2f}")
            if geo is not None and sh.phone in geo.phones:
                r0, r1 = geo.recording(sh.phone)
                if w[0] < r0 - 0.05 or w[1] > r1 + 0.05:
                    bad.append(f"{segs[sh.segs[0]]['id']}: phone {sh.phone}'s window {w} is outside its recording {r0:.2f}-{r1:.2f} s")
    # the pivots
    phrases = back_to_axis(sp["utterances"])
    for ph in phrases:
        f = film_of(doc, ph["axis_t"])
        s = next((x for x in segs if x["film"][0] - 1e-6 <= f < x["film"][1]), None)
        if s is None or not s.get("pivot"):
            bad.append(f"'{ph['words']}' ({ph['utt']}, {ph['axis_t']:.2f} s): the word 'axis' is not in a pivot shot ({s and s['id']})")
            continue
        if not any(a.get("text") == PIVOT_LABEL and (a.get("target") or {}).get("type") in ("pivot_foot", "couple_com") for a in s["annotations"]):
            bad.append(f"{s['id']}: no '{PIVOT_LABEL}' arrow on the pivot")
        for x in segs:
            for a in x["annotations"]:
                if (a.get("target") or {}).get("type") == "leader_t" and str(a.get("text", "")).lower() == "axis" and a["dance_t"][0] < ph["t1"] and a["dance_t"][1] > ph["u_start"]:
                    bad.append(f"{x['id']}: a leader's-T arrow 'axis' is still in '{ph['words']}'")
    # physics: the legend wherever the load colours are, a floor dot per dancer on every physics segment of a second or more
    for s in segs:
        L = s["layers"]
        if L.get("skeleton_mode") == "physics":
            if not L.get("legend"):
                bad.append(f"{s['id']}: load colours without the 'ESTIMATED' legend")
            if s["dance"][1] - s["dance"][0] >= 1.0 and not any(a["kind"] == "com_marker" and any(t.get("type") == "com" for t in a.get("targets", [])) for a in s["annotations"]):
                bad.append(f"{s['id']}: a physics segment without the dancers' centre-of-mass floor dots")
    # the 'connected' rule read the hand-contact events of its build: say when they have changed since
    ce = ((doc.get("recap_film") or {}).get("v3") or {}).get("connected_events") or {}
    if ce.get("path"):
        now = file_sha(Path(ce["path"]))
        if now != ce.get("sha256_16"):
            notes.append(f"STALE: {ce['path']} changed since this build (sha {ce.get('sha256_16')} -> {now}): the physics layer's 'connected' rule must be "
                         f"re-applied - run `python -m dancecap.film_recap_flow build` (then `check`)")
        else:
            notes.append(f"'connected' rule up to date with {Path(ce['path']).name} (sha {now})")
    # captions and the soundtrack's time map: what the class film has
    cur_dir = WORK / "film" / "direction.json"
    if cur_dir.exists():
        cur = json.loads(cur_dir.read_text(encoding="utf-8"))
        a = [(round(t, 3), round(_dance_at_film(doc, t), 3), _locked_at(doc, t)) for t in np.arange(0.05, doc["timeline"]["film_duration_s"], 0.1)]
        b = [(round(t, 3), round(_dance_at_film(cur, t), 3), _locked_at(cur, t)) for t in np.arange(0.05, cur["timeline"]["film_duration_s"], 0.1)]
        if a != b:
            diff = next((x, y) for x, y in zip(a, b) if x != y) if len(a) == len(b) else (len(a), len(b))
            bad.append(f"the film / master time map differs from film/direction.json (first {diff}): the soundtrack would change")
        else:
            notes.append(f"time map = film/direction.json's at {len(a)} points every 0.1 s (film {doc['timeline']['film_duration_s']} s): the same soundtrack")
        cur_tl = WORK / "film" / "narration" / "timeline.json"
        if cur_tl.exists():
            ct = json.loads(cur_tl.read_text(encoding="utf-8"))
            same = json.dumps(ct.get("captions"), sort_keys=True) == json.dumps(tl.get("captions"), sort_keys=True)
            (notes if same else bad).append("captions identical to film/narration/timeline.json" if same else "the captions differ from film/narration/timeline.json")
    return {"bad": bad, "notes": notes, "boundaries": n_bound, "shots": len(shots), "holds_min": min((h["hold_s"] for h in holds), default=None),
            "transitions": {t: sum(1 for sh in shots[1:] if segs[sh.segs[0]]["camera"]["transition_in"]["type"] == t) for t in cfg.rule}}


def _dance_at_film(doc: dict, f: float) -> float:
    for s in doc["segments"]:
        if s["film"][0] - 1e-9 <= f < s["film"][1]:
            return s["dance"][0] + (f - s["film"][0]) * float(s["speed"])
    return float("nan")


def _locked_at(doc: dict, f: float) -> bool:
    for s in doc["segments"]:
        if s["film"][0] - 1e-9 <= f < s["film"][1]:
            return s.get("audio") == "locked" and float(s["speed"]) == 1.0 and not s.get("replay")
    return False


# ------------------------------------------------------------------------------------------------ (4) before / after


def phone_time(doc: dict) -> dict:
    """seconds of the film in phone shots, and of those the phone's own full-frame video (v3: arrival .. departure; before v3 a camera_tour was full
    frame from its first frame, a fly_to_camera from the end of its flight + fade)"""
    segs = doc["segments"]
    shots = shots_of(segs)
    tot = doc["timeline"]["film_duration_s"]
    in_shots = sum(sh.dur for sh in shots if sh.kind == "phone")
    full = 0.0
    for k, sh in enumerate(shots):
        if sh.kind != "phone":
            continue
        cam = segs[sh.segs[0]]["camera"]
        tr = cam.get("transition_in")
        a = sh.f0 + (tr["start_s"] + tr["duration_s"] if tr else (float(cam.get("fly_s", 0)) + 0.5 * float(cam.get("fade_in_s", 0)) if cam.get("mode") == "fly_to_camera" else 0.0))
        nxt = shots[k + 1] if k + 1 < len(shots) else None
        ntr = segs[nxt.segs[0]]["camera"].get("transition_in") if nxt else None
        b = nxt.f0 + ntr["start_s"] if ntr else (sh.f1 - (0.5 * float(cam.get("fade_out_s", 0)) + float(cam.get("pull_back_s", 0)) if cam.get("mode") == "fly_to_camera" else 0.0))
        full += max(0.0, b - a)
    return {"phone_shots_s": round(in_shots, 1), "phone_shots_share": round(in_shots / tot, 3), "full_frame_video_s": round(full, 1), "full_frame_share": round(full / tot, 3),
            "phone_shots": sum(1 for sh in shots if sh.kind == "phone")}


def cfg_from_doc(doc: dict) -> FlowCfg:
    """the flow rules a direction was built with (its flow.defaults), so `check` judges a direction by its own rules"""
    cfg = FlowCfg()
    d = (doc.get("flow") or {}).get("defaults") or {}
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, tuple(v) if k == "ground_m" else v)
    return cfg


def flow_stats(doc: dict, cfg: FlowCfg) -> dict:
    """the camera's motion over the film: per transition type the count and durations, a histogram, the share of the film in flight, the phones'
    full-frame share, the holds per view kind, the fastest arc and the frustum surfaces' readable seconds"""
    segs = doc["segments"]
    shots = shots_of(segs)
    dur = doc["timeline"]["film_duration_s"]
    rows = []
    for k, sh in enumerate(shots[1:], 1):
        tr = segs[sh.segs[0]]["camera"].get("transition_in")
        if not tr:
            continue
        st = sh.f0 + tr["start_s"]
        T = tr["duration_s"]
        peak = 1.875 * abs(tr["arc"]["delta_deg"]) / max(1e-3, T * (1 - (cfg.axis_share if "axis" in str(tr.get("path")) else 0)))
        rd = {role: round(frustum_readable(role, tr["type"], tr[key]["film"], st, T, cfg), 3) for role, key in (("enter", "frustum_in"), ("leave", "frustum_out")) if tr.get(key)}
        rows.append({"k": k, "type": tr["type"], "T": T, "start": st, "crane": bool(tr.get("crane")), "peak": peak, "readable": rd, "prev": shots[k - 1].kind, "kind": sh.kind})
    q = lambda xs, p: round(float(np.percentile(xs, p)), 2) if xs else None
    by = {}
    for typ in cfg.rule:
        xs = [r["T"] for r in rows if r["type"] == typ]
        if xs:
            by[typ] = {"n": len(xs), "min": round(min(xs), 2), "median": q(xs, 50), "max": round(max(xs), 2), "sum_s": round(sum(xs), 1)}
    phone_ways = [r["T"] for r in rows if r["type"] in AXIS_TYPES]
    edges = [0.0, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 99.0]
    hist = {f"{a:g}-{b:g} s" if b < 99 else f">= {a:g} s": sum(1 for r in rows if a <= r["T"] < b) for a, b in zip(edges, edges[1:])}
    motion = sum(r["T"] for r in rows)
    holds = {}
    for k, sh in enumerate(shots[:-1]):
        tr = segs[sh.segs[0]]["camera"].get("transition_in")
        arrive = sh.f0 + (tr["start_s"] + tr["duration_s"] if tr else 0.0)
        ntr = segs[shots[k + 1].segs[0]]["camera"].get("transition_in")
        if ntr:
            holds.setdefault("replay" if sh.replay else sh.kind, []).append(shots[k + 1].f0 + ntr["start_s"] - arrive)
    rds = [(v, r["k"], role) for r in rows for role, v in r["readable"].items()]
    return {"film_s": dur, "transitions": len(rows), "by_type": by, "histogram": hist, "all": {"min": round(min(r["T"] for r in rows), 2), "median": q([r["T"] for r in rows], 50),
            "max": round(max(r["T"] for r in rows), 2)},
            "phone_ways": {"n": len(phone_ways), "min": round(min(phone_ways), 2) if phone_ways else None, "median": q(phone_ways, 50), "p90": q(phone_ways, 90),
                           "max": round(max(phone_ways), 2) if phone_ways else None, "sum_s": round(sum(phone_ways), 1)},
            "motion_s": round(motion, 1), "motion_share": round(motion / dur, 3), "cranes": sum(1 for r in rows if r["type"] in ("crane_up", "crane_down") or r["crane"]),
            "peak_arc_deg_s": round(max(r["peak"] for r in rows), 1), "holds": {k: {"n": len(v), "min": round(min(v), 2), "median": q(v, 50)} for k, v in holds.items()},
            "frustum_readable": {"n": len(rds), "min_s": round(min(rds)[0], 2) if rds else None, "median_s": q([x[0] for x in rds], 50)},
            "phones": phone_time(doc)}


def quantify_physics(new: dict, old: dict) -> dict:
    """film seconds of: the load colours (skeleton_mode physics), each dancer's centre-of-mass floor dot (com_marker with the dancers' COM), the balance
    axes, the counterbalance layer, the pivot shots; per shot kind. The dots count the marker's own window."""
    def tally(doc):
        out = {"load_colours_s": 0.0, "floor_dots_s": 0.0, "balance_axes_s": 0.0, "counterbalance_s": 0.0, "pivot_s": 0.0, "by_kind": {}}
        for s in doc["segments"]:
            d = s["film"][1] - s["film"][0]
            k = shot_kind(s["camera"])
            bk = out["by_kind"].setdefault(k, {"film_s": 0.0, "load_colours_s": 0.0, "floor_dots_s": 0.0})
            bk["film_s"] += d
            if s["layers"].get("skeleton_mode") == "physics":
                out["load_colours_s"] += d
                bk["load_colours_s"] += d
                if s["layers"].get("balance", True):
                    out["balance_axes_s"] += d
            if s["layers"].get("counterbalance"):
                out["counterbalance_s"] += d
            if s.get("pivot"):
                out["pivot_s"] += d
            dots = [a for a in s["annotations"] if a["kind"] == "com_marker" and any(t.get("type") == "com" for t in a.get("targets", []))]
            if dots:
                span = max(a["film_in_segment"][1] for a in dots) - min(a["film_in_segment"][0] for a in dots)
                out["floor_dots_s"] += span
                bk["floor_dots_s"] += span
        r = lambda x: round(x, 1)
        out = {k: (r(v) if isinstance(v, float) else {kk: {k3: r(v3) for k3, v3 in vv.items()} for kk, vv in v.items()}) for k, v in out.items()}
        out["film_s"] = round(doc["timeline"]["film_duration_s"], 1)
        return out
    return {"before": tally(old), "after": tally(new), "phones_before": phone_time(old), "phones_after": phone_time(new)}


# ------------------------------------------------------------------------------------------------ the phones' clips


def clips_needed(doc: dict, geo: Geo, pad: float = 0.5) -> dict:
    """per phone, the master-second ranges its video is needed for: the phone shot itself AND the frustum surface's way in and out (the full-resolution
    strip: the surface fills the frame at the arrival). Compared with the clips the capture has now (cameras.json)."""
    need: dict[str, list[list[float]]] = {}
    for sh in shots_of(doc["segments"]):
        if sh.kind == "phone":
            need.setdefault(sh.phone, []).append([sh.d0 - pad, sh.d1 + pad])
    for fz in doc.get("flow", {}).get("frustums", []):
        a, b = fz["film"][0], fz["film"][3]
        da, db = _dance_at_film(doc, max(0.0, a)), _dance_at_film(doc, min(doc["timeline"]["film_duration_s"] - 1e-3, b))
        if math.isnan(da) or math.isnan(db):
            continue
        need.setdefault(fz["camera"], []).append([min(da, db) - pad, max(da, db) + pad])
    out = {}
    for cam, rs in need.items():
        rs.sort()
        merged = []
        for a, b in rs:
            if merged and a <= merged[-1][1] + 0.05:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        have = geo.phones.get(cam, {}).get("clips", [])
        missing = []
        for a, b in merged:
            cover = sorted([max(a, x0), min(b, x1)] for x0, x1 in have if x1 > a and x0 < b)
            t = a
            for x0, x1 in cover:
                if x0 > t + 0.05:
                    missing.append([round(t, 2), round(x0, 2)])
                t = max(t, x1)
            if b > t + 0.05:
                missing.append([round(t, 2), round(b, 2)])
        out[cam] = {"need": [[round(a, 2), round(b, 2)] for a, b in merged], "need_s": round(sum(b - a for a, b in merged), 1),
                    "missing": missing, "missing_s": round(sum(b - a for a, b in missing), 1)}
    return out


# ------------------------------------------------------------------------------------------------ the review table


def _sentence_shots(doc: dict, u: dict) -> list[str]:
    out = []
    for s in doc["segments"]:
        if s.get("replay") or s["dance"][1] <= u["start"] + 0.05 or s["dance"][0] >= u["end"] - 0.05:
            continue
        c = s["camera"]
        m = c.get("mode")
        ph = phone_of(c)
        name = {"camera_tour": f"phone {ph}", "fly_to_camera": f"fly into phone {ph}", "physics_3q": "physics view", "feature_zoom": f"zoom {c.get('feature')}",
                "overhead_floorcraft": "overhead", "orbit": "calm orbit", "feet_closeup": "feet close-up", "rhythm": "rhythm"}.get(m, m)
        if s.get("pivot"):
            name = "PIVOT (counterbalance)"
        if s["layers"].get("skeleton_mode") == "physics" and m != "physics_3q" and not s.get("pivot"):
            name += " +physics"
        tr = c.get("transition_in")
        tag = ""
        if tr and abs(s["film"][0] - (s["film"][0])) < 1e-9 and _is_shot_start(doc, s):
            tag = f" <- {tr['type']} {tr['duration_s']:.1f} s"
        if not out or out[-1].split(" <-")[0] != name or tag:
            out.append(name + tag)
    return out


def _is_shot_start(doc: dict, s: dict) -> bool:
    i = doc["segments"].index(s)
    return i == 0 or doc["segments"][i - 1]["camera"] != s["camera"]


def review_md(res: dict) -> str:
    doc, sp, v3 = res["doc"], res["sp"], res["v3"]
    cur = json.loads((WORK / "film" / "direction.json").read_text(encoding="utf-8"))
    ch, q, clips = res["checks"], res["physics_quant"], res["clips"]
    segs = doc["segments"]
    shots = shots_of(segs)
    L = [f"# v3 direction review: {doc['title']} ({doc['name']})", "",
         f"Film {doc['timeline']['film_duration_mmss']} ({doc['timeline']['film_duration_s']} s, unchanged), {len(segs)} segments, {len(shots)} shots "
         f"(film/direction.json: {len(cur['segments'])} segments, {len(shots_of(cur['segments']))} shots). Checks: "
         + ("PASS" if not ch["bad"] else f"{len(ch['bad'])} problems") + f"; {ch['boundaries']} shot changes, every one a flight (no jump cut).", ""]
    L += ["## (3) 'back to the axis' = the counterbalance pivot point", "",
          "| master s | film s | speaker | sentence | old shot(s) | new shot | the pivot it points at |", "|---|---|---|---|---|---|---|"]
    for p in v3["pivots"]:
        # what the film shows now (film/direction.json) from the phrase's first word to the end of 'axis'
        olds = []
        for x in cur["segments"]:
            if x.get("replay") or x["dance"][1] <= p["t0"] or x["dance"][0] >= p["axis_t"] + 0.4:
                continue
            c = x["camera"]
            ph_ = phone_of(c)
            arr = [f"'{an.get('text')}' on {(an.get('target') or {}).get('type')}" for an in x["annotations"]
                   if an.get("kind") == "arrow3d" and an["dance_t"][0] < p["axis_t"] + 0.4 and an["dance_t"][1] > p["t0"]]
            olds.append(f"{c.get('mode')}{(' ' + ph_) if ph_ else ''} `{x['id']}`" + (f" (arrows: {', '.join(arr)})" if arr else ""))
        old = "; ".join(olds)
        f0 = film_of(doc, p["window"][0])
        sh = next(s for s in segs if s.get("pivot") and s["pivot"]["utt"] == p["utt"])
        tr = sh["camera"].get("transition_in") or {}
        L.append(f"| {p['axis_t']:.2f} | {film_of(doc, p['axis_t']):.2f} | {p['speaker']} | {p['text']} | {old} | ground-level pivot view {p['window'][0]:.2f}-{p['window'][1]:.2f} s "
                 f"(film {f0:.1f}), in by {tr.get('type', '-')} {tr.get('duration_s', 0):.1f} s; arrow '{PIVOT_LABEL}'"
                 + (f"; {'; '.join(p['window_why'])}" if p["window_why"] else "") + f" | {p['target']['why']} |")
    L += ["", f"Kept as the leader's T (they mean his T: an axis pivot / the crossbar): the 'exit', 'rotate', 'rotation', 'side to side' calls next to "
          f"these phrases (the pivot shots end 0.2 s before them). Dropped 'axis' events: {', '.join(e['id'] for e in v3['axis_events_dropped'])}. "
          f"The transcript gives u026 to Larissa; the phrase is the same and it is treated the same. The replay (240.1-243.7 s, 'See so he start pivoting') "
          f"is the pivot view too (was an overhead): {'; '.join(v3['replay'])}.", ""]
    b, a = q["before"], q["after"]
    L += ["## (4) physics on screen, before / after (film seconds)", "", "| | before (film/direction.json) | after (v3) |", "|---|---|---|"]
    for k, lab in (("load_colours_s", "load colours (tension / compression in arms and legs)"), ("floor_dots_s", "each dancer's centre-of-mass floor dot"),
                   ("balance_axes_s", "balance axes (chest to stance foot)"), ("counterbalance_s", "counterbalance layer (couple COM dot, pivot rings)"),
                   ("pivot_s", "counterbalance pivot shots")):
        L.append(f"| {lab} | {b[k]} s ({100 * b[k] / b['film_s']:.0f} %) | {a[k]} s ({100 * a[k] / a['film_s']:.0f} %) |")
    for kind in ("3d", "phone", "overhead"):
        bb, aa = b["by_kind"].get(kind, {}), a["by_kind"].get(kind, {})
        L.append(f"| ... in {kind} shots: load colours / floor dots | {bb.get('load_colours_s', 0)} / {bb.get('floor_dots_s', 0)} s of {bb.get('film_s', 0)} s | "
                 f"{aa.get('load_colours_s', 0)} / {aa.get('floor_dots_s', 0)} s of {aa.get('film_s', 0)} s |")
    off = [r for r in v3["physics"] if not r["on"]]
    L += ["", f"Rule: a segment gets the physics layer when the teachers demonstrate (moving >= 20 % of it) or are connected (a hand on the partner or the "
          f"counterbalance connection >= 50 % of it; source: {v3['connected_source']}) and do not stand apart (> 0.95 m). Off ({len(off)} segments, "
          f"{sum(r['film'][1] - r['film'][0] for r in off):.1f} s): " + ", ".join(f"{r['id']} ({r['why']})" for r in off) + ". The legend 'Skeleton load - ESTIMATED from motion' "
          "keeps its fixed place in the top band (under a chapter card while one shows): the captions are never covered; the dots and colours are in the 3D picture, "
          "under the screen-space captions.", ""]
    pb, pa = q["phones_before"], q["phones_after"]
    fs = res.get("flow_stats") or flow_stats(doc, res["cfg"])
    prev = prev_q = None
    if PREV_V31.exists():
        pd = json.loads(PREV_V31.read_text(encoding="utf-8"))
        prev = flow_stats(pd, cfg_from_doc(pd))
        prev_q = quantify_physics(pd, cur)["after"]
    L += ["## (5) the transitions", "", "Owner decisions 2026-10-21 (v3.1): **overheads** - \"Keep overheads, crane up\": the 4 true top-down views stay, each entered and "
          "left by the smooth crane (the only ways that leave ground level); **phone share** - \"Shorter flights, ~50% phone\": quicker ways into and out of the "
          "phones (one easing, smootherstep; no arc faster than " + f"{MAX_ARC_DEG_S:.0f} deg/s at its peak) and a shorter minimum phone hold, so the phones' full-frame "
          "video comes back toward half the film. Every phone entry / exit still shows the frustum surface standing in the world, at least "
          f"{res['cfg'].frustum_readable_s} s readable (alpha >= 0.5, the eye not yet / no longer on the phone's optical axis). v3.2 (same day, toward the owner's "
          "~50 %): the 3D views take only the time they need and give the rest to the phone beside them (`give_phones_time`), a view that misses its hold by "
          "a little takes that little instead of swallowing a phone shot (`nudge`), a shot with nothing of its own to show joins the phone beside it before a "
          "pivot / physics view (`phone_first`), and the phone ways are as short as their floors allow (the frustum's readable floor, the 200 deg/s arc floor; "
          "the way out of a phone starts at the next segment).", "",
          "| | film/direction.json (cuts) | v3.1 | v3.2 (this) |", "|---|---|---|---|"]
    pv = (prev or {}).get("phones") or {}
    L.append(f"| phone shots | {pb['phone_shots']}, {pb['phone_shots_s']} s ({100 * pb['phone_shots_share']:.1f} %) | "
             + (f"{pv['phone_shots']}, {pv['phone_shots_s']} s ({100 * pv['phone_shots_share']:.1f} %)" if pv else "-")
             + f" | {pa['phone_shots']}, {pa['phone_shots_s']} s ({100 * pa['phone_shots_share']:.1f} %) |")
    L.append(f"| **full-frame phone video** | {pb['full_frame_video_s']} s ({100 * pb['full_frame_share']:.1f} %) | "
             + (f"{pv['full_frame_video_s']} s ({100 * pv['full_frame_share']:.1f} %)" if pv else "-")
             + f" | **{pa['full_frame_video_s']} s ({100 * pa['full_frame_share']:.1f} %)** |")
    def _row(lab, f):
        return f"| {lab} | - | " + (f(prev) if prev else "-") + f" | {f(fs)} |"
    L.append(_row("shots / transitions", lambda x: f"{x['transitions'] + 1} / {x['transitions']}"))
    L.append(_row("camera in motion (sum of the transitions)", lambda x: f"{x['motion_s']} s ({100 * x['motion_share']:.1f} %)"))
    L.append(_row("phone ways (in / out / phone to phone): n, median, p90, max", lambda x: f"{x['phone_ways']['n']}, {x['phone_ways']['median']} / {x['phone_ways']['p90']} / {x['phone_ways']['max']} s"))
    for typ in ("phone_in", "phone_out", "phone_to_phone", "flight", "crane_up", "crane_down", "replay_in", "replay_out"):
        L.append(_row(f"{typ}: n, min / median / max", lambda x, t=typ: (f"{x['by_type'][t]['n']}, {x['by_type'][t]['min']} / {x['by_type'][t]['median']} / {x['by_type'][t]['max']} s"
                                                                        if t in x["by_type"] else "0")))
    L.append(_row("crane ways (the overheads'; incl. a phone way passing one)", lambda x: str(x["cranes"])))
    L.append(_row("fastest arc (smootherstep peak)", lambda x: f"{x['peak_arc_deg_s']} deg/s"))
    L.append(_row("frustum surface readable on a phone way: min / median", lambda x: f"{x['frustum_readable']['min_s']} / {x['frustum_readable']['median_s']} s"))
    L.append(_row("shortest hold: phone / 3D / overhead", lambda x: " / ".join(f"{x['holds'].get(k, {}).get('min')} s" for k in ("phone", "3d", "overhead"))))
    if prev_q:
        L.append(f"| physics on screen: load colours / COM floor dots / counterbalance layer / pivot shots | - | {prev_q['load_colours_s']} / {prev_q['floor_dots_s']} / "
                 f"{prev_q['counterbalance_s']} / {prev_q['pivot_s']} s | {q['after']['load_colours_s']} / {q['after']['floor_dots_s']} / {q['after']['counterbalance_s']} / "
                 f"{q['after']['pivot_s']} s |")
    gave = v3["flow"].get("gave") or []
    L += ["", "Durations of all transitions (v3.2): " + ", ".join(f"{k}: {v}" for k, v in fs["histogram"].items()) + ".", "",
          f"v3.2, what the 3D views gave the phones ({sum(1 for g in gave if not g.startswith('(not taken)'))} moves, each kept only when every view still holds "
          "its minimum, every chapter still opens a shot and the phones' full-frame video grows):", ""]
    L += [f"- {g}" for g in gave if not g.startswith("(not taken)")]
    L += ["", "Tried and not taken:", ""] + [f"- {g[len('(not taken) '):]}" for g in gave if g.startswith("(not taken)")]
    L += ["", f"Why not 50 %: the phone shots are {100 * pa['phone_shots_share']:.1f} % of the film (the cut film {100 * pb['phone_shots_share']:.1f} %), but every "
          f"phone way overlaps the phone's own time a little (a phone_to_phone, centred on its boundary, takes all of its {fs['by_type'].get('phone_to_phone', {}).get('sum_s', 0)} s "
          f"from the two phones; a way out that must arrive before a called word starts before the boundary). What is left in 3D: the {len(v3['pivots'])} pivots "
          f"(their windows, from the verb to 1.6 s after 'axis'), the {sum(1 for x in shots if x.kind == 'overhead')} overheads, the physics views (each from 0.5 s "
          "before its first called word to its last call's arrow + its way out, and never shorter than the director's 3.0 s), the replay, the opening orbit under "
          "the title card (phone 05 records only from 7.41 s; only phone 00 sees both teachers before, 174 deg round from phone 05: a whip, not taken) and "
          "3.0 s of the calm orbit at 271 s (no phone sees both teachers at 263.9-266.9 s master). Going further would need fewer pivots / overheads / "
          "physics views - the owner's call, not done.", "",
          f"Counts: {ch['transitions']}; shortest hold {ch['holds_min']} s. Joins (a shot too short to hold between its way in "
          f"and out joined a neighbour, {len(v3['flow']['merges'])}):", ""] + [f"- {m}" for m in v3["flow"]["merges"]]
    sq = [(sh, segs[sh.segs[0]]["camera"]["transition_in"]) for sh in shots[1:] if segs[sh.segs[0]]["camera"]["transition_in"].get("squeezed")]
    L += ["", f"Squeezed ({len(sq)}; the rule allows {res['cfg'].squeeze_max_s} s shorter and an arrival up to {res['cfg'].arrive_late_max_s} s after the "
          f"segment start - later only where a phone does not record yet: it is entered once it does, its surface fading in on its own video): " +"; ".join(f"{segs[sh.segs[0]]['id']} {t['type']} ({', '.join(t['squeezed'])})" for sh, t in sq), ""]
    L += ["| # | film s | shot | kind | phone | way in | starts | duration | arc | from -> to |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, sh in enumerate(shots):
        s0 = segs[sh.segs[0]]
        t = s0["camera"].get("transition_in")
        if not t:
            L.append(f"| {k} | {sh.f0:.1f}-{sh.f1:.1f} | {s0['id']} | {sh.mode} | {sh.phone or ''} | (first shot) | | | | |")
            continue
        L.append(f"| {k} | {sh.f0:.1f}-{sh.f1:.1f} | {s0['id']}{' PIVOT' if s0.get('pivot') else ''} | {sh.mode} | {sh.phone or ''} | {t['type']} | {t['start_s']:+.2f} s | "
                 f"{t['duration_s']:.2f} s | {t['arc']['delta_deg']:+.0f} deg | {t['from']['kind']} {t['from']['height_m']} m -> {t['to']['kind']} {t['to']['height_m']} m |")
    L += ["", "## Per sentence: what the teachers say, the old shots, the new shots (and how the camera gets there)", "",
          "| film s | master s | speaker | sentence | old (film/direction.json) | new (v3) |", "|---|---|---|---|---|---|"]
    for u in sp["utterances"]:
        if u["end"] <= 0:
            continue
        old = " / ".join(_sentence_shots(cur, u))
        new = " / ".join(_sentence_shots(doc, u))
        L.append(f"| {film_of(doc, max(0.0, u['start'])):.1f} | {max(0.0, u['start']):.1f}-{u['end']:.1f} | {u.get('speaker') or ''} | {u['text'][:110]} | {old} | {new} |")
    L += ["", "## The phones' clips (the frustum surface needs the video through the way in and out)", "",
          "| phone | needed (s) | missing in the capture now (s) | missing ranges (master s) |", "|---|---|---|---|"]
    for cam, c in sorted(clips.items()):
        L.append(f"| {cam} | {c['need_s']} | {c['missing_s']} | {', '.join(f'{x:.1f}-{y:.1f}' for x, y in c['missing'][:12])}{' ...' if len(c['missing']) > 12 else ''} |")
    L += ["", "## Checks", ""] + ([f"- PROBLEM: {x}" for x in ch["bad"]] or ["- film_recap.validate + v3 checks: no problems"]) + [f"- {x}" for x in ch["notes"]]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ the camera-flow sheet


def sheet(res: dict, out: Path) -> Path:
    """film time vs the eye's azimuth about the couple and its height: shots as bars, the flights as smootherstep curves between them (phone_in / out:
    the arc then the axis run), the phones' azimuths as dotted lines; physics / pivot on a strip. For review: no jump anywhere, heights in the band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    doc, cfg = res["doc"], res["cfg"]
    segs = doc["segments"]
    shots = shots_of(segs)
    geo = res["geo"]
    plan_azimuths(shots, segs, geo, cfg)
    schedule(shots, segs, geo, cfg)
    col = {"3d": "#2f6fb0", "phone": "#c2571a", "overhead": "#6a3d9a"}
    rows = 4
    dur = doc["timeline"]["film_duration_s"]
    span = math.ceil(dur / rows)
    fig, axes = plt.subplots(rows * 2, 1, figsize=(22, 4.2 * rows), gridspec_kw={"height_ratios": [3, 1.4] * rows})
    # a continuous azimuth curve: hold (with drift) inside each shot, smootherstep across each transition
    pts_t, pts_a, pts_h = [], [], []
    for k, sh in enumerate(shots):
        tr = segs[sh.segs[0]]["camera"].get("transition_in")
        arrive = sh.f0 + (tr["start_s"] + tr["duration_s"] if tr else 0.0)
        nxt = shots[k + 1] if k + 1 < len(shots) else None
        ntr = segs[nxt.segs[0]]["camera"]["transition_in"] if nxt else None
        depart = nxt.f0 + ntr["start_s"] if nxt else sh.f1
        for t in np.linspace(arrive, depart, max(2, int((depart - arrive) * 10))):
            u = (t - sh.f0) / max(1e-3, sh.f1 - sh.f0)
            pts_t.append(t)
            pts_a.append(sh.a0 + wrap180(sh.a1 - sh.a0) * u)
            pts_h.append(sh.h0 + (sh.h1 - sh.h0) * u)
        if nxt:
            a_from, a_to = pts_a[-1], pts_a[-1] + wrap180(nxt.a0 - pts_a[-1])
            h_from, h_to = pts_h[-1], nxt.h0
            T = ntr["duration_s"]
            for t in np.linspace(depart, depart + T, 24):
                x = smootherstep((t - depart) / T)
                pts_t.append(t)
                pts_a.append(a_from + (a_to - a_from) * x)
                pts_h.append(h_from + (h_to - h_from) * x)
    pts_t, pts_a, pts_h = np.array(pts_t), np.unwrap(np.radians(pts_a)) * 180 / math.pi, np.array(pts_h)
    for r in range(rows):
        ax, axh = axes[2 * r], axes[2 * r + 1]
        t0, t1 = r * span, min(dur, (r + 1) * span)
        m = (pts_t >= t0 - 1) & (pts_t <= t1 + 1)
        base = np.median(pts_a[m]) if m.any() else 0
        off = 360 * round(base / 360)
        ylo, yhi = float(np.min(pts_a[m] - off)) - 40, float(np.max(pts_a[m] - off)) + 25
        ylo, yhi = min(ylo, -200), max(yhi, 200)
        ax.plot(pts_t[m], pts_a[m] - off, color="#222", lw=1.6)
        for cam in sorted(geo.phones):
            azs = [geo.phone(cam, x)["az"] for x in np.linspace(t0, t1, 5)]
            a = np.median(azs)
            for o in (-720, -360, 0, 360, 720):
                if ylo < a + o < yhi:
                    ax.axhline(a + o, color="#c2571a", ls=":", lw=0.8, alpha=0.6)
                    ax.text(t0 + 0.3, a + o + 3, f"phone {cam}", color="#c2571a", fontsize=8)
        for k, sh in enumerate(shots):
            if sh.f1 < t0 or sh.f0 > t1:
                continue
            s0 = segs[sh.segs[0]]
            lab = ("PIVOT" if s0.get("pivot") else sh.mode.replace("_floorcraft", "").replace("camera_tour", f"ph {sh.phone}"))
            ax.axvspan(sh.f0, sh.f1, ymin=0.0, ymax=0.04, color=col[sh.kind], alpha=0.9)
            ax.text(0.5 * (sh.f0 + sh.f1), ylo + 8, lab, fontsize=7, ha="center", rotation=90, va="bottom", color=col[sh.kind])
            tr = s0["camera"].get("transition_in")
            if tr:
                ax.axvspan(sh.f0 + tr["start_s"], sh.f0 + tr["start_s"] + tr["duration_s"], color="#999", alpha=0.18)
            phys = sum(segs[i]["film"][1] - segs[i]["film"][0] for i in sh.segs if segs[i]["layers"].get("skeleton_mode") == "physics")
            if phys > 0:
                axh.axvspan(sh.f0, sh.f1, ymin=0.92, ymax=1.0, color="#e07b00", alpha=0.85)
        for fz in doc["flow"]["frustums"]:
            if fz["film"][3] >= t0 and fz["film"][0] <= t1:
                axh.plot(fz["film"], [2.9, 3.3, 3.3, 2.9], color="#c2571a", lw=1)
        ax.set_xlim(t0, t1)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel("eye azimuth (deg)")
        axh.plot(pts_t[m], np.minimum(pts_h[m], 3.4), color="#222", lw=1.4)
        axh.axhspan(cfg.ground_m[0], cfg.ground_m[1], color="#4caf50", alpha=0.12)
        axh.set_xlim(t0, t1)
        axh.set_ylim(0, 3.5)
        axh.set_ylabel("eye h (m)\n(>3.4 = overhead)")
        axh.set_xlabel("film s   (grey = a flight; orange bar = physics layer on; orange trapezoid = a phone frustum shown)")
    fig.suptitle(f"v3.2 camera flow: {len(shots)} shots, {len(shots) - 1} flights, no cut (blue 3D, orange phone, purple overhead)", fontsize=14)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=72)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ write


MIX_SCRIPT = '''"""the film recorder's soundtrack hook for the v3 direction (generated by dancecap.film_recap_flow): <direction.name>_mix.wav from the direction"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))          # .../dancecap
from dancecap import film_recap_audio as fa  # noqa: E402

direction = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "direction.json"
print(fa.film_mix(direction, fa.soundtrack_class_mix(HERE.parents[2] / "film" / "audio")))
'''


def write_all(res: dict, paths: fr.Paths) -> None:
    doc, tl = res["doc"], res["tl"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "narration").mkdir(exist_ok=True)
    (OUT / "audio").mkdir(exist_ok=True)
    (OUT / "direction.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    caps.write_timeline(OUT / "narration" / "timeline.json", tl)
    ch = []
    for i, c in enumerate(doc["chapters"]):
        ch.append({"title": c["title"], "startS": c["film_s"], "endS": doc["chapters"][i + 1]["film_s"] if i + 1 < len(doc["chapters"]) else doc["timeline"]["film_duration_s"],
                   "segment": c["segment"]})
    (OUT / "chapters.json").write_text(json.dumps({"schema": "dancecap.film_recap/1/chapters", "film_duration_s": doc["timeline"]["film_duration_s"], "chapters": ch},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")
    rep = {"schema": "dancecap.film_recap_flow/1/report", "checks": res["checks"], "flow_stats": res["flow_stats"], "physics_on_screen": res["physics_quant"],
           "clips": res["clips"], **res["v3"],
           "base": {k: v for k, v in res["report"].items() if k in ("families_seconds", "inserts", "chapters", "caption_chunks", "dropped")},
           "base_director": {k: v for k, v in (res["report"].get("director") or {}).items() if k in ("pov_share", "pov_seconds", "pov_shots_n", "pov_seconds_per_phone")}}
    (OUT / "direction_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT / "clips.json").write_text(json.dumps({"what": "per phone: the master-second ranges of video the v3 direction needs (phone shots + frustum ways in / out, padded 0.5 s)",
                                                "phones": res["clips"]}, indent=1), encoding="utf-8")
    (OUT / "review_v3.md").write_text(review_md(res), encoding="utf-8")
    (OUT / "audio" / "make_film_mix.py").write_text(MIX_SCRIPT, encoding="utf-8")
    sheet(res, OUT / "flow_sheet.png")


def cmd_check(path: Path) -> int:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    tl = json.loads((Path(path).parent / "narration" / "timeline.json").read_text(encoding="utf-8"))
    sp = fr.load_speech(fr.Paths())
    try:
        geo = Geo()
    except (OSError, ValueError, KeyError):
        geo = None
    if int((doc.get("flow") or {}).get("version", 1)) >= 2:
        return cmd_check_v4(doc, tl, sp, geo)
    r = check(doc, tl, sp, geo, cfg_from_doc(doc))
    print("OK" if not r["bad"] else "\n".join(r["bad"]))
    for n in r["notes"]:
        print(n)
    print(f"{r['shots']} shots, {r['boundaries']} flights {r['transitions']}, shortest hold {r['holds_min']} s")
    return 1 if r["bad"] else 0


def cmd_check_v4(doc: dict, tl: dict, sp: dict, geo: Geo | None) -> int:
    """the v4 checks + the owner's targets (against the v3.2 direction when it is there, else the targets the direction carries)"""
    rev = (doc.get("flow") or {}).get("revision")
    # the revision gate: every revision has its own targets; one this code does not know is an error, never v4.0's targets by accident
    cfgs = {None: V4Cfg, "4.0": V4Cfg, "4.1": v41_cfg, "4.2": v42_cfg, "4.3": v43_cfg}
    if rev not in cfgs:
        print(f"flow.revision {rev!r}: no targets for it in this code (known: {', '.join(str(k) for k in cfgs if k)}); rebuild or update dancecap.film_recap_flow")
        return 1
    v41 = rev in V41_STYLE
    cfg4 = cfgs[rev]()
    geo4 = Geo4(geo) if geo is not None else None
    base = v4_baseline(geo4, cfg4) if (geo4 is not None and V32_DIRECTION.exists() and not v41) else None
    pov = fr.load_pov(fr.Paths(), "auto")
    phq = (PhoneQ41 if v41 else PhoneQ)(pov, geo4, cfg4) if (pov is not None and geo4 is not None) else None
    met = None
    if v41 and geo4 is not None:
        met = film_metrics(doc, geo4, cfg4)
        need_p = Path(((doc.get("recap_film") or {}).get("v4") or {}).get("need3d", {}).get("path") or NEED3D)
        v41_extra_metrics(met, doc, geo4, cfg4, NeedSeries(need_p, doc) if need_p.exists() else None, pov)
        if rev in ("4.2", "4.3"):
            met_h = film_metrics(doc, geo4, cfg4, aspect="horizontal")
            v42_extra_metrics(met, doc, geo4, cfg4, pov, met_h)
            if rev == "4.3":
                # the frame-exact comparison needs the v4.2 direction it was built on (build --version v4.2 reproduces it into v4/direction/v4.2/)
                d42 = json.loads(V42_DIRECTION.read_text(encoding="utf-8")) if V42_DIRECTION.exists() else None
                v43_extra_metrics(met, doc, geo4, cfg4, met_h, d42)
    r = check_v4(doc, tl, sp, geo, geo4, cfg4, base, met, phq)
    print("OK" if not r["bad"] else "\n".join(r["bad"]))
    for row in r["targets"]:
        print(f"  {'OK    ' if row['ok'] else 'MISSED'} {row['target']}: want {row['want']}, measured {row['measured']}")
    for n in r["notes"]:
        print(n)
    print(f"{r['shots']} shots, {r['boundaries']} ways {r['transitions']}")
    return 1 if r["bad"] else 0


def cmd_clips(no_install: bool, priority: str, out_dir: Path = OUT) -> int:
    """UNITY STAGE ONLY (it writes the capture's cameras/ folder, under the Unity lock): decode the phones' video the v3 direction needs (clips.json: the
    phone shots and the frustum surfaces' ways in and out) together with the clips the capture has now (so film/direction.json still plays)"""
    from . import camera_video as cv
    need = json.loads((out_dir / "clips.json").read_text(encoding="utf-8"))["phones"]
    geo = Geo()
    clips = {}
    for cam in sorted(set(need) | set(geo.phones)):
        rs = [tuple(x) for x in need.get(cam, {}).get("need", [])] + [tuple(x) for x in geo.phones.get(cam, {}).get("clips", [])]
        if rs:
            clips[cam] = sorted(rs)
    cv.OPTIONS["offsets_file"] = rp.write_offsets()
    cv.set_priority(priority)
    print("clips (s per phone):", {c: round(sum(y - x for x, y in cv.merge_ranges(v)), 1) for c, v in clips.items()})
    man = cv.run([TAKE], CAPTURE, None, [str(out_dir / "direction.json")], False, list(rp.PHONES), None, None, not no_install, False, False, 1280, 85, 256, 78, clips=clips)
    print("pov_ok:", man.get("pov_ok"))
    return 0 if man.get("pov_ok") else 1


# ======================================================================================================== v4 (owner 2026-10-22)
# The owner's words (2026-10-22): "we are moving the camera way too much, and we are not holding on video footage long enough ... reduce the
# camera motions actions by a factor of 3 ... hold on a video segment from a phone, increase that by a factor of 3 ... much smoother translations
# and zoom in and outs and fade in and outs from every camera point of view ... spend more time in the 3D orbiting Gaussian splat view, slow orbit
# ... the camera is in smooth motion at all times, but we should also lock into video footage for longer periods ... very, very faint camera
# frustums from the other phones at all times ... maybe 15% opacity".
#
# v4 is a second planner over the same base (the D2/D4 director's segments, the pivots, the replay at ground level: the time map, captions,
# chapters, arrows and soundtrack are exactly v3's / film/'s). Instead of v3's shot list it chooses, by dynamic programming over the speech's
# pauses, a few LONG phone holds (every 2-s piece of each hold shows both teachers: recap_pov.window_quality) and lets everything else be ONE
# slow orbit per stretch (4-10 deg/s, one rate per stretch) that carries the pivots, the physics views, the overheads (crane up / down inside the
# orbit, the azimuth keeps turning) and the replay (the orbit continues; only its centre glides over the dance-time jump). Every way is a C2 live
# blend (smootherstep ramps of LIVE endpoint states: the camera leaves an orbit at the orbit's speed and lands on a phone at the phone's), 2.5-5 s,
# capped in azimuth / rotation / speed, with its FOV zoom and its picture / outline fades spelled out per way. Every phone's frustum is drawn at
# 0.15 opacity in every 3D view (flow.ghost_frustums). v3 stays reproducible: `build` (no --version) still writes v3/direction/.

OUT_V4 = WORK / "v4" / "direction"
V32_DIRECTION = OUT / "direction.json"            # the 'before': the v3.2 direction the v3 finals were recorded from

FLOW_SCHEMA_V2 = ("hm film flow v2 (work/larissa_kadu_recap/v4/direction/UNITY_CONTRACT_v4.md; additive over v1): flow.version 2, flow.ghost_frustums, "
                  "flow.v4 {targets, baseline, measured}, flow.defaults.v4 {caps, profiles, orbit}; camera.transition_in.v4 {path, ease, profile, fov, fade, "
                  "caps, peaks, centre}; camera.v4 {role, run, rate_deg_s}; overhead up_policy 'orbit'; flow.frustums[].v4 {picture, outline}")

# the ways' profiles: u = 0 at the departure, 1 at the arrival; each entry is the window [a, b] of u over which that weight ramps 0 -> 1
# (smootherstep: zero 1st AND 2nd derivative at both ends of its window, so a sum of ramps applied to live endpoint states is C2)
V4_PROFILE = {
    # into a phone: the eye glides (cylindrical about the couple: azimuth the planned way round, radius, height) from the live orbit to the
    # phone's optical axis 1 m behind the phone (the portal), then dollies along the axis into the phone; the lens zooms to the crop's FOV
    # (log-tan: an even zoom speed in the picture); the picture fades in on the frustum while the camera glides; the outline rises from the
    # ghost's 0.15 to 1
    "phone_in": {"glide": [0.0, 0.75], "dolly": [0.5, 1.0], "rot": [0.0, 0.8], "zoom": [0.25, 1.0], "picture_in": [0.0, 0.35], "outline_in": [0.0, 0.25]},
    # out of a phone: the exact time reverse of phone_in
    "phone_out": {"dolly_out": [0.0, 0.5], "glide": [0.25, 1.0], "rot": [0.2, 1.0], "zoom": [0.0, 0.75], "picture_out": [0.65, 1.0], "outline_out": [0.75, 1.0]},
    # phone to phone (ONE move): back out of A along its axis, glide to B's portal, dolly into B; the two pictures cross-fade mid-way
    "phone_to_phone": {"dolly_out": [0.0, 0.3], "glide": [0.15, 0.85], "dolly_in": [0.7, 1.0], "rot": [0.15, 0.85], "zoom": [0.15, 0.85],
                       "picture_out": [0.35, 0.65], "picture_in": [0.35, 0.65], "outline_out": [0.45, 0.8], "outline_in": [0.2, 0.55]},
    # the overheads (owner: "Keep overheads, crane up"): a vertical arc about the look point; the azimuth keeps the orbit's live turn throughout
    "crane_up": {"elevation": [0.0, 1.0], "distance": [0.0, 1.0], "zoom": [0.0, 1.0]},
    "crane_down": {"elevation": [0.0, 1.0], "distance": [0.0, 1.0], "zoom": [0.0, 1.0]},
    # the replay: the orbit goes on (same azimuth function, radius, height); only its centre glides from the frozen pre-flash couple to the live one
    "replay_in": {"centre": [0.0, 1.0]},
    "replay_out": {"centre": [0.0, 1.0]},
    "flight": {"glide": [0.0, 1.0], "rot": [0.0, 1.0], "zoom": [0.0, 1.0]},
}
V4_PATH = {"phone_in": "glide_axis", "phone_out": "axis_glide", "phone_to_phone": "axis_glide_axis", "crane_up": "crane", "crane_down": "crane",
           "replay_in": "orbit_recentre", "replay_out": "orbit_recentre", "flight": "glide"}
# v4.1: the same ways, but the cranes follow the cruise law (the pitch rate eases in over the first quarter, holds, eases out over the last
# quarter: peak 1.33 x the mean instead of smootherstep's 1.875 x), so a crane under the 20 deg/s pitch cap takes 5.1 s instead of 7 s
V41_CRANE = {"window": [0.0, 1.0], "law": "cruise", "ease": 0.25}
V41_PROFILE = {k: copy.deepcopy(v) for k, v in V4_PROFILE.items()}
for _t in ("crane_up", "crane_down"):
    V41_PROFILE[_t] = {"elevation": dict(V41_CRANE), "distance": dict(V41_CRANE), "zoom": dict(V41_CRANE)}
# the revisions built on the v4.1 plan (its profiles, targets and checks); every one of them is listed here and in cmd_check_v4's gate, so a new
# revision is never checked against v4.0's targets by accident
V41_STYLE = ("4.1", "4.2", "4.3")


def ramp(u: float, a: float, b: float) -> float:
    """smootherstep of u over the window [a, b] (0 before a, 1 after b)"""
    if b <= a:
        return 1.0 if u >= b else 0.0
    return smootherstep((u - a) / (b - a))


def _int_smootherstep(y: float) -> float:
    """the integral of smootherstep from 0 to y (y in [0, 1]; 0.5 at 1)"""
    y = min(1.0, max(0.0, y))
    return y ** 4 * (2.5 - 3.0 * y + y * y)


def cruise(x: float, ease: float = 0.25) -> float:
    """v4.1's crane law (0 -> 1 over x in [0, 1]): the RATE rises by a smootherstep over the first `ease` share, holds, and falls by a smootherstep
    over the last `ease` share. Its peak rate is 1 / (1 - ease) x the mean (smootherstep's is 1.875 x), with zero 1st and 2nd derivative at both
    ends (C2 against the live orbit), so a crane under a pitch cap can be shorter than a smootherstep crane"""
    x = min(1.0, max(0.0, x))
    e = min(0.5, max(1e-3, ease))
    if x <= e:
        s = e * _int_smootherstep(x / e)
    elif x <= 1.0 - e:
        s = 0.5 * e + (x - e)
    else:
        y = (x - (1.0 - e)) / e
        s = 0.5 * e + (1.0 - 2.0 * e) + e * (y - _int_smootherstep(y))
    return s / (1.0 - e)


def ramp_spec(u: float, spec) -> float:
    """a profile entry at u: [a, b] = smootherstep over that window (v4); {"window": [a, b], "law": "cruise", "ease": e} = the cruise law (v4.1 cranes)"""
    if isinstance(spec, dict):
        a, b = spec.get("window", [0.0, 1.0])
        if spec.get("law") == "cruise":
            return cruise((u - a) / (b - a) if b > a else (1.0 if u >= b else 0.0), float(spec.get("ease", 0.25)))
        return ramp(u, a, b)
    return ramp(u, *spec)


def profile_weights(typ: str, u: float, ghost: float = 0.15, prof: dict | None = None) -> dict:
    """the weights of a v4 way at u (0 = departure, 1 = arrival): glide, axis distances (fractions of axis_run_m behind the phone: `d_out` for the
    phone left, `d_in` for the phone entered), rot, zoom, the pictures' alpha and the outlines' alpha (the ghost's `ghost` at rest). `prof` = the
    way's own profile (transition_in.v4.profile; default V4_PROFILE[typ])"""
    p = prof if prof is not None else V4_PROFILE[typ]
    w = {k: ramp_spec(u, v) for k, v in p.items()}
    out = {"glide": w.get("glide", 0.0), "rot": w.get("rot", w.get("glide", 0.0)), "zoom": w.get("zoom", 0.0)}
    if typ == "phone_in":
        out["d_in"] = 1.0 - w["dolly"]
        out["picture_in"] = w["picture_in"]
        out["outline_in"] = ghost + (1.0 - ghost) * w["outline_in"]
    elif typ == "phone_out":
        out["d_out"] = w["dolly_out"]
        out["picture_out"] = 1.0 - w["picture_out"]
        out["outline_out"] = ghost + (1.0 - ghost) * (1.0 - w["outline_out"])
    elif typ == "phone_to_phone":
        out["d_out"] = w["dolly_out"]
        out["d_in"] = 1.0 - w["dolly_in"]
        out["picture_out"] = 1.0 - w["picture_out"]
        out["picture_in"] = w["picture_in"]
        out["outline_out"] = ghost + (1.0 - ghost) * (1.0 - w["outline_out"])
        out["outline_in"] = ghost + (1.0 - ghost) * w["outline_in"]
    elif typ in ("crane_up", "crane_down"):
        out["elevation"] = w["elevation"]
        out["distance"] = w["distance"]
    elif typ in ("replay_in", "replay_out"):
        out["centre"] = w["centre"]
    return out


def picture_readable_u(typ: str, table: dict | None = None) -> dict:
    """the share of a v4 way in which a phone's picture READS as a picture standing in the world: alpha >= 0.5 and the eye not yet (enter) / no
    longer (leave) at the phone (until the dolly has covered half the axis run the picture does not fill the frame). `table` = the profiles
    (V4_PROFILE / V41_PROFILE)"""
    p = (table or V4_PROFILE)[typ]
    half = lambda w: 0.5 * (w[0] + w[1])
    out = {}
    if typ == "phone_in":
        out["enter"] = max(0.0, half(p["dolly"]) - half(p["picture_in"]))
    elif typ == "phone_out":
        out["leave"] = max(0.0, half(p["picture_out"]) - half(p["dolly_out"]))
    elif typ == "phone_to_phone":
        out["leave"] = max(0.0, half(p["picture_out"]) - half(p["dolly_out"]))
        out["enter"] = max(0.0, half(p["dolly_in"]) - half(p["picture_in"]))
    return out


@dataclass
class V4Cfg:
    # the owner's factors, measured against the v3.2 direction (V32_DIRECTION)
    move_factor: float = 3.0             # camera moves <= v3.2's / 3
    hold_factor: float = 3.0             # phone holds: mean, median and shortest >= 3 x v3.2's
    hold_min_floor_s: float = 4.2        # (the owner's example: 1.4 s -> >= 4.2 s); the build takes max(this, 3 x the baseline's shortest)
    hold_dp_min_s: float = 9.0           # the planner's own floor for a phone hold (the median / mean targets need long holds; 3x the minimum is the check)
    hold_max_s: float = 34.0             # a phone hold at most (variety; the phones' crops are re-made for the whole hold)
    hold_sub_s: float = 2.0              # every piece of this length inside a hold must show both teachers (recap_pov.window_quality)
    pure_orbit_min_s: float = 5.0        # an orbit between two phones holds at least this long between its way out and its way in
    way_margin_s: float = 4.5            # the planner keeps this much orbit for each way in / out (a way is 2.5-5 s; the schedule sets it)
    orbit_max_s: float = 160.0
    # the slow orbit (one rate per stretch between two phones; never static, never faster than the top of the band)
    orbit_rate_deg_s: tuple = (4.0, 10.0)
    orbit_rate_pref_deg_s: float = 6.0
    orbit_block: dict = field(default_factory=lambda: {"mode": "orbit", "radius_m": 3.6, "radius_vertical_m": 4.3, "height_m": 2.15, "look_height_m": 0.95,
                                                       "period_s": 60, "fit_span": True, "avoid_inline": True, "inline_lift_max_m": 0.25,
                                                       "focus": "couple", "focus_weight": 0.0, "blend_s": 0.0, "horizontal": {"fov_scale": 0.85}})
    orbit_r_h: tuple = (4.3, 2.15)       # the plan's estimate of the orbit's eye (9:16 framing): radius about the couple, height
    orbit_fov_deg: float = 42.0          # ... and of its vertical FOV (FitSpan widens it while the dancers stand apart)
    orbit_look_h: float = 0.95
    overhead_h: float = 7.0
    overhead_min_s: float = 3.5          # a base overhead at least this long is one of the owner's 4 (crane up / down); shorter ones ride in the orbit
    # the ways: [lo, hi] and the base duration per type (the caps and the picture's readable time may lengthen a way up to hi)
    dur: dict = field(default_factory=lambda: {"phone_in": [2.5, 5.0, 3.5], "phone_out": [2.5, 5.0, 3.5], "phone_to_phone": [3.5, 5.0, 4.0],
                                               "crane_up": [3.0, 5.0, 3.6], "crane_down": [3.0, 5.0, 3.6], "replay_in": [2.5, 2.5, 2.5],
                                               "replay_out": [2.5, 2.5, 2.5], "flight": [2.5, 5.0, 3.0]})
    cap_az_deg_s: float = 30.0           # the eye's turn about the couple (v3.2's ways: up to 168 deg/s measured in Unity, 199.9 planned)
    cap_rot_deg_s: float = 30.0          # the view direction's turn rate (the pan the viewer sees)
    cap_speed_m_s: float = 2.4           # the eye's speed (= the turn cap at the orbit's radius: 30 deg/s x 4.5 m)
    cap_crane_deg_s: float = 30.0        # a crane's pitch about the couple
    readable_s: float = 1.2              # every phone picture reads as a picture in the world this long on its way (v3: 0.6)
    axis_run_m: float = 1.0
    ghost_opacity: float = 0.15
    ghost_depth_m: float = 0.6
    crop_smooth_s: float = 1.8           # the phone's crop (the virtual camera inside its video) is smoothed over this (v3: recap_pov 0.6 s)
    call_arrive_before_s: float = 0.1
    arrive_after_s: float = 0.35         # a crane has arrived this long after its overhead's segment starts (the words begin)
    hold_s: dict = field(default_factory=lambda: {"3d": 1.0, "overhead": 2.2, "replay": 1.5})
    # the planner's value of a film second as phone video / as orbit, by what the base director chose there (its calls)
    weights: dict = field(default_factory=lambda: {"phone": [1.0, 0.6], "physics_3q": [0.45, 1.0], "zoom": [0.65, 0.9], "orbit": [0.8, 0.85],
                                                   "protected": [0.0, 1.0]})
    max_excess_deg: float = 50.0         # an orbit stretch must link its two phones with at most this much turn beyond the ways' natural ease
    excess_penalty: float = 3.0          # ... and pays this (at max_excess_deg) for what it needs
    same_phone_penalty: float = 2.0      # the same phone again after an orbit (variety of points of view)
    move_penalty: float = 0.5            # (the move budget is the real limit; this only breaks ties toward fewer ways)
    chapter_bonus: float = 1.5
    sample_hz: float = 30.0
    # ---- v4.1 (owner 2026-10-22 after the v3 films again: "still too many fast camera transitions. we should hold longer on POVs and slowly orbit
    # gaussian splats. also the physics meter is occluding the figures in the vertical video. it should move to the top of the frame"); every
    # default below is v4.0's behaviour, v41_cfg() sets the v4.1 values
    version: str = "4.0"
    crane_ease: float | None = None      # None = smootherstep cranes (v4.0); else the cruise law with this ease share (v41: 0.25)
    crane_span_deg: float = 74.0         # the crane's elevation change (the orbit's eye at about 15.6 deg to 89.9 deg)
    moves_cap: int | None = None         # an absolute move budget (v4.1: 22 = v4.0's count; v4.0: v3.2's / move_factor)
    share_min: float = 0.0               # the phones' full-frame share of the film at least (v4.1: 0.45) - the PLANNER's target
    share_check_min: float = -1.0        # >= 0: the CHECK's share target instead of share_min (v4.1: 0.25, the owner's decision of 2026-10-22:
                                         #   record the meaning-first plan, 27.6 % phone video; the 45 % target is dropped). The planner keeps
                                         #   share_min, so a rebuild still delivers the same meaning-first plan
    hold_floor_s: float = 0.0            # every phone hold at least (absolute; v4.1: 20 s)
    hold_mean_floor_s: float = 0.0       # the holds' mean at least (absolute; v4.1: 25 s)
    need_alpha: float = 0.0              # v4.1: the weight of Jev's need_3d in the plan's value (phone (3 - n) / 3, orbit n / 3 per second)
    phys_call_need: float = 2.6          # ... a second the base director framed as a physics view with a called arrow counts at least this need
    lam_max: float = 4.0                 # ... the share's Lagrange multiplier (value per phone second) is searched in [0, lam_max]
    overhead_leave_early_s: float = 0.0  # a crane down may leave its overhead this much before the overhead's end (v4.1: 2 s, when a phone hold needs it)
    overhead_gap_s: float = 0.0          # the orbit holds this long between a phone's way and a crane (v4.1: 1 s)
    live_rate_frac: float = 1e9          # a phone way only where the phone's own live turn about the couple and its own rotation stay under this share of the caps
    way_speed_frac: float = 0.8          # ... the way's planned length keeps its glide + dolly under this share of the speed cap
    final_phone: bool = False            # the film may end on a phone hold (no way out; v4.1)
    quality_ladder: tuple = ()           # after the strict rule (whole hold strong, every 2-s piece usable): ((whole cover, usable share of the 2-s pieces, longest unusable s), ...)
    crop: dict | None = None             # v4.1: the 9:16 crop of a phone hold {top, fill, max_fill} (the teachers' box under the physics legend)
    legend: dict | None = None           # v4.1: where the physics legend sits in 9:16 (presets.vertical.legend)
    need_clear_margin: float = 0.25      # v4.1 target: the holds' mean need_3d at least this much under the 3D views'
    schedule_step_s: float = 0.2
    schedule_iters: int = 5
    link_aim_share: float = 0.0          # v4.1: the share of the phone's aim offset the link's rotation estimate adds (0: the measured ways follow the turn)
    link_frac: float = 0.8               # v4.1: an orbit links two phones only when its ways' estimated peaks stay under this share of the caps
    orbit_lead: float = 0.0              # v4.1: a phone way turns >= (0.5 + this) x the orbit's rate x its time the orbit's way (0.25: never against it)
    legend_hold_min: float = 0.0         # v4.1: a phone hold only where its crop keeps the teachers clear of the 9:16 legend in this share of the frames
    plan: str = "share"                  # v4.1: "share" = the least multiplier that fills share_min (meaning within it); "meaning" = the most phone time
                                         # whose holds stay need_clear_margin under the 3D views' need

    def prof(self) -> dict:
        """the ways' profiles of this version"""
        return V41_PROFILE if self.version in V41_STYLE else V4_PROFILE


def v4_hold_targets(base: dict, cfg4: V4Cfg) -> dict:
    """the owner's targets as numbers, from the baseline's measured metrics"""
    h = base["phone_holds"]
    return {"moves_max": int(math.floor(base["moves"]["total"] / cfg4.move_factor + 1e-9)),
            "hold_mean_min_s": round(cfg4.hold_factor * h["mean_s"], 2), "hold_median_min_s": round(cfg4.hold_factor * h["median_s"], 2),
            "hold_min_s": round(max(cfg4.hold_min_floor_s, cfg4.hold_factor * h["min_s"]), 2),
            "orbit_rate_deg_s": list(cfg4.orbit_rate_deg_s), "way_s": [min(v[0] for k, v in cfg4.dur.items() if k.startswith("phone")),
                                                                       max(v[1] for k, v in cfg4.dur.items() if k.startswith("phone"))],
            "cap_az_deg_s": cfg4.cap_az_deg_s, "cap_rot_deg_s": cfg4.cap_rot_deg_s, "cap_speed_m_s": cfg4.cap_speed_m_s,
            "cap_crane_deg_s": cfg4.cap_crane_deg_s, "readable_s": cfg4.readable_s, "ghost_opacity": cfg4.ghost_opacity}


# ------------------------------------------------------------------------------------------------ v4: geometry with smoothed tracks


def _gauss_rows(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.3 or len(a) < 3:
        return a
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(a.shape[1])], axis=1)


class Geo4:
    """Geo with the couple's centre and the phones' poses smoothed (the viewer follows damped centres; the estimates of speed are taken on
    these). Master seconds in, Unity capture coordinates out."""

    def __init__(self, geo: Geo, centre_sigma_s: float = 0.5, phone_sigma_s: float = 0.25):
        self.g = geo
        hz = 1.0 / max(1e-3, float(np.median(np.diff(geo.t))))
        c = 0.5 * (geo.lead + geo.follow)
        c[:, 1] = 0.0
        self.t = geo.t
        self.c = _gauss_rows(c, centre_sigma_s * hz)
        self.ph = {}
        for cam, p in geo.phones.items():
            phz = 1.0 / max(1e-3, float(np.median(np.diff(p["t"])))) if len(p["t"]) > 2 else 30.0
            fwd = _gauss_rows(p["fwd"], phone_sigma_s * phz)
            fwd /= np.maximum(1e-9, np.linalg.norm(fwd, axis=1))[:, None]
            self.ph[cam] = {"t": p["t"], "pos": _gauss_rows(p["pos"], phone_sigma_s * phz), "fwd": fwd, "vfov": p["vfov"], "aspect": p["aspect"]}

    def centre(self, d: float) -> np.ndarray:
        return np.array([np.interp(d, self.t, self.c[:, k]) for k in range(3)])

    def phone(self, cam: str, d: float) -> dict:
        p = self.ph[cam]
        pos = np.array([np.interp(d, p["t"], p["pos"][:, k]) for k in range(3)])
        fwd = np.array([np.interp(d, p["t"], p["fwd"][:, k]) for k in range(3)])
        fwd /= max(1e-9, np.linalg.norm(fwd))
        return {"pos": pos, "fwd": fwd, "vfov": float(np.interp(d, p["t"], p["vfov"])), "aspect": p["aspect"]}

    def recording(self, cam: str) -> tuple[float, float]:
        return self.g.recording(cam)


def cyl(eye: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    """(azimuth deg, horizontal radius, height) of a point about the couple's centre c (the viewer's convention: atan2(z, x))"""
    d = eye - c
    return math.degrees(math.atan2(d[2], d[0])), float(math.hypot(d[0], d[2])), float(eye[1])


def uncyl(az: float, r: float, h: float, c: np.ndarray) -> np.ndarray:
    a = math.radians(az)
    return np.array([c[0] + r * math.cos(a), h, c[2] + r * math.sin(a)])


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _nlerp(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """the view direction between a and b (the slerp's path; the rate estimate does not need the exact slerp parametrisation)"""
    a, b = _unit(a), _unit(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9995:
        return _unit((1 - w) * a + w * b)
    om = math.acos(dot)
    return _unit((math.sin((1 - w) * om) * a + math.sin(w * om) * b) / math.sin(om))


def _ang(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


# ------------------------------------------------------------------------------------------------ v4: the poses of a direction (both versions)


class PoseModel:
    """The camera of a flow direction as a function of film time, from the plan (estimates; the viewer flies between its own live poses):
    a 3D shot's eye = the couple's smoothed centre + (world_azimuth_deg + drift x (film - F0), radius, height); an overhead 7 m above; a phone shot
    = the phone's smoothed pose. Inside a way the LIVE endpoint states are blended by the way's profile (v4: V4_PROFILE; v1 / v3: the v3 paths -
    the arc in its arc share, the dollies on the axis). Used for the before / after numbers of both v3.2 and v4."""

    def __init__(self, doc: dict, geo4: Geo4, cfg: FlowCfg | None = None, cfg4: V4Cfg | None = None, aspect: str = "vertical"):
        # aspect: "vertical" (9:16, the orbit's radius_vertical_m: every v4 number of the plan) or "horizontal" (16:9, radius_m; v4.2 reports both)
        self.doc, self.g = doc, geo4
        self.aspect = aspect
        self.segs = doc["segments"]
        self.shots = shots_of(self.segs)
        self.v4 = int((doc.get("flow") or {}).get("version", 1)) >= 2
        self.cfg = cfg or cfg_from_doc(doc)
        self.cfg4 = cfg4 or V4Cfg()
        self.F = float(doc["timeline"]["film_duration_s"])
        self.trs = []
        for k, sh in enumerate(self.shots):
            tr = self.segs[sh.segs[0]]["camera"].get("transition_in")
            if k > 0 and tr:
                self.trs.append((k, sh.f0 + tr["start_s"], tr["duration_s"], tr["type"], tr))
        self._f0 = np.array([sh.f0 for sh in self.shots])
        # v4.3: the planned azimuth tracks (flow.azimuth_tracks, read by a camera block's `azimuth_track`): inside a track's film window the eye's
        # world azimuth and its lift come from the table (linear between the 30-Hz samples), outside it the block's law (UNITY_CONTRACT_v4 K)
        self.tracks = {}
        for tid, tb in ((doc.get("flow") or {}).get("azimuth_tracks") or {}).items():
            n = int(tb["n"])
            tt = float(tb["t0"]) + float(tb["dt"]) * np.arange(n)
            lift = np.asarray(tb.get("lift_m") or [0.0] * n, float)
            self.tracks[tid] = (float(tb["film"][0]), float(tb["film"][1]), tt, np.asarray(tb["deg"], float), lift)

    def track_at(self, cam: dict, f: float) -> tuple[float, float] | None:
        """v4.3: (world azimuth deg, eye lift m) of a tracked block at film f strictly inside its track's window, else None (the block's law)"""
        tid = cam.get("azimuth_track")
        tr = self.tracks.get(tid) if tid is not None else None
        if tr is None or not (tr[0] < f < tr[1]):
            return None
        return float(np.interp(f, tr[2], tr[3])), float(np.interp(f, tr[2], tr[4]))

    # --- film -> master
    def dance(self, f: float, k: int | None = None) -> float:
        if k is not None:
            return dance_at(self.segs, self.shots[k], f)
        d = _dance_at_film(self.doc, min(max(f, 0.0), self.F - 1e-6))
        return d

    def shot_at(self, f: float) -> int:
        return int(max(0, np.searchsorted(self._f0, f, side="right") - 1))

    # --- a shot's own (live) state at film f (also before / after it: the ways blend live states)
    def state(self, k: int, f: float, centre_from: float | None = None) -> dict:
        sh = self.shots[k]
        cam = self.segs[sh.segs[0]]["camera"]
        if sh.f0 <= f <= sh.f1:
            d = self.dance(f, k)
        else:
            # outside the shot (a way into / out of it): the master second runs on at speed 1 from the shot's edge (locked) or holds (a replay)
            s_edge = self.segs[sh.segs[0]] if f < sh.f0 else self.segs[sh.segs[-1]]
            edge_d = sh.d0 if f < sh.f0 else sh.d1
            d = edge_d + (f - (sh.f0 if f < sh.f0 else sh.f1)) * (1.0 if _locked(s_edge) else 0.0)
        c = self.g.centre(d if centre_from is None else centre_from)
        if sh.kind == "phone":
            p = self.g.phone(sh.phone, d)
            zoom = 1.0
            crop = cam.get("crop") or {}
            if crop.get("zoom"):
                i = int(np.clip(round((d - float(crop["t0"])) / float(crop["dt"])), 0, len(crop["zoom"]) - 1))
                zoom = float(crop["zoom"][i])
            fov = 2 * math.degrees(math.atan(math.tan(math.radians(p["vfov"]) / 2) / max(1.0, zoom)))
            return {"kind": "phone", "eye": p["pos"], "dir": p["fwd"], "fov": fov, "c": c, "phone": sh.phone, "pos": p["pos"], "fwd": p["fwd"], "d": d}
        if sh.kind == "overhead":
            if self.v4 and cam.get("world_azimuth_deg") is not None:
                az = float(cam["world_azimuth_deg"]) + float(cam.get("drift_deg_s", 0.0)) * (f - sh.f0)
                tk = self.track_at(cam, f)
                if tk is not None:
                    az = tk[0]
            else:
                az = float(cam.get("up_azimuth_deg", sh.a0 if sh.a0 else 0.0))
            look = c + np.array([0.0, 0.9, 0.0])
            eye = look + np.array([0.0, self.cfg4.overhead_h - 0.9, 0.0]) + 0.05 * np.array([math.cos(math.radians(az)), 0.0, math.sin(math.radians(az))])
            return {"kind": "overhead", "eye": eye, "dir": look - eye, "fov": 40.0, "c": c, "az": az, "look": look, "d": d}
        az = float(cam.get("world_azimuth_deg", 0.0)) + float(cam.get("drift_deg_s", 0.0)) * (f - sh.f0)
        tk = self.track_at(cam, f)
        lift = 0.0
        if tk is not None:
            az, lift = tk
        if self.v4 and (cam.get("v4") or {}).get("role"):
            rkey = "radius_m" if self.aspect == "horizontal" else "radius_vertical_m"
            r, h = float(cam.get(rkey, self.cfg4.orbit_r_h[0])), float(cam.get("height_m", self.cfg4.orbit_r_h[1]))
            lh = float(cam.get("look_height_m", self.cfg4.orbit_look_h))
            fac = closer_factor(cam, f)                      # v4.2: 1 = the block's (closer) offset, 1 / k = v4.1's at a phone way
            r, h = r * fac, lh + (h - lh) * fac
        else:
            key = "pivot" if self.segs[sh.segs[0]].get("pivot") else eye_key(self.segs[sh.segs[0]])
            r, h = self.cfg.eye_rh.get(key, (3.5, 1.6))
            lh = 0.95
        h += lift                                            # v4.3: the track's planned inline lift (0 without a track)
        eye = uncyl(az, r, h, c)
        look = c + np.array([0.0, lh, 0.0])
        return {"kind": "3d", "eye": eye, "dir": look - eye, "fov": self.cfg4.orbit_fov_deg, "c": c, "az": az, "r": r, "h": h, "look": look, "d": d}

    def _axis_pt(self, st: dict, frac: float) -> np.ndarray:
        return st["pos"] - frac * self.cfg4.axis_run_m * st["fwd"]

    def pose(self, f: float) -> dict:
        """the camera at film f: {eye, dir, fov, c, in_way (k or None), typ, u, kind}. The view on screen is the shot whose way in has ARRIVED
        last (a way may start after its shot's first frame - the previous view holds into it - or arrive before it)"""
        cur = 0
        for k, t0, T, typ, tr in self.trs:
            if f < t0 - 1e-9:
                break
            if f <= t0 + T + 1e-9:
                return self._way(k, t0, T, typ, tr, f)
            cur = k
        st = self.state(cur, f)
        st.update({"in_way": None, "typ": None, "k": cur})
        return st

    def _way(self, k: int, t0: float, T: float, typ: str, tr: dict, f: float) -> dict:
        u = min(1.0, max(0.0, (f - t0) / max(1e-6, T)))
        p, n = k - 1, k
        if typ in ("replay_in", "replay_out"):
            # the dance time jumps at the replay's edge: v3 flew from the frozen pose; v4 orbits on about a centre that eases from the frozen one
            edge = self.shots[n].f0
            # v3: the previous shot's pose frozen at the edge; v4: its orbit going on at f about the couple's centre frozen at the edge
            A = self.state(p, f, centre_from=self.dance(edge - 1e-3, p)) if self.v4 else self.state(p, edge - 1e-3)
            B = self.state(n, f)
            w = ramp(u, 0.0, 1.0)
            if self.v4:
                c = (1 - w) * A["c"] + w * B["c"]
                az = B["az"]
                eye = uncyl(az, B["r"], B["h"], c)
                look = c + (B["look"] - B["c"])
                return {"kind": "3d", "eye": eye, "dir": look - eye, "fov": B["fov"], "c": c, "in_way": k, "typ": typ, "u": u, "k": k}
            return self._blend(A, B, {"glide": w, "rot": w, "zoom": w}, k, typ, u, B["c"])
        A, B = self.state(p, f), self.state(n, f)
        if self.v4:
            # the way's own profile as the direction wrote it (v4.1's cranes follow the cruise law), else v4.0's
            w = profile_weights(typ, u, self.cfg4.ghost_opacity, (tr.get("v4") or {}).get("profile"))
        else:
            w = self._v3_weights(typ, u)
        return self._blend(A, B, w, k, typ, u, (1 - w.get("glide", u)) * A["c"] + w.get("glide", u) * B["c"])

    def _v3_weights(self, typ: str, u: float) -> dict:
        s = self.cfg.axis_share
        if typ == "phone_in":
            return {"glide": ramp(u, 0.0, 1 - s), "rot": ramp(u, 0.0, 1 - s), "zoom": ramp(u, 0.0, 1.0), "d_in": 1.0 - ramp(u, 1 - s, 1.0)}
        if typ == "phone_out":
            return {"glide": ramp(u, s, 1.0), "rot": ramp(u, s, 1.0), "zoom": ramp(u, 0.0, 1.0), "d_out": ramp(u, 0.0, s)}
        if typ == "phone_to_phone":
            return {"glide": ramp(u, s / 2, 1 - s / 2), "rot": ramp(u, s / 2, 1 - s / 2), "zoom": ramp(u, 0, 1), "d_out": ramp(u, 0.0, s / 2), "d_in": 1.0 - ramp(u, 1 - s / 2, 1.0)}
        if typ in ("crane_up", "crane_down"):
            return {"elevation": ramp(u, 0, 1), "distance": ramp(u, 0, 1), "zoom": ramp(u, 0, 1)}
        return {"glide": ramp(u, 0, 1), "rot": ramp(u, 0, 1), "zoom": ramp(u, 0, 1)}

    def _blend(self, A: dict, B: dict, w: dict, k: int, typ: str, u: float, c: np.ndarray) -> dict:
        if typ in ("crane_up", "crane_down"):
            lo, hi = (A, B) if typ == "crane_up" else (B, A)          # lo = the ground view, hi = the overhead
            e = w["elevation"] if typ == "crane_up" else 1.0 - w["elevation"]
            look = lo.get("look", lo["c"] + np.array([0, 0.95, 0]))
            g_el = math.degrees(math.atan2(lo["eye"][1] - look[1], math.hypot(lo["eye"][0] - look[0], lo["eye"][2] - look[2])))
            g_d = float(np.linalg.norm(lo["eye"] - look))
            el = g_el + (89.9 - g_el) * e
            dist = g_d + (self.cfg4.overhead_h - 0.9 - g_d) * e
            if self.v4:
                # v4: both ends are the same orbit function a(t) (the overhead turns with it): continuous by construction
                az = A.get("az", 0.0) + u * wrap180(B.get("az", 0.0) - A.get("az", 0.0))
            else:
                # v3: the overhead's up azimuth (from its entry) and the ground view's differ; the azimuth turns near the ground
                a_hi, a_lo = hi.get("az", 0.0), lo.get("az", hi.get("az", 0.0))
                wa = ramp(u, 0.5, 1.0) if typ == "crane_down" else 1.0 - ramp(u, 0.0, 0.5)
                az = a_hi + wa * wrap180(a_lo - a_hi)
            a = math.radians(az)
            eye = look + dist * np.array([math.cos(math.radians(el)) * math.cos(a), math.sin(math.radians(el)), math.cos(math.radians(el)) * math.sin(a)])
            return {"kind": "crane", "eye": eye, "dir": look - eye, "fov": A["fov"], "c": lo["c"], "in_way": k, "typ": typ, "u": u, "k": k, "el": el, "az": az}
        # the two ends' anchor points: a phone end is its axis point (d x axis_run behind the phone), a 3D end its eye
        ea = self._axis_pt(A, w.get("d_out", 0.0)) if A["kind"] == "phone" else A["eye"]
        eb = self._axis_pt(B, w.get("d_in", 0.0)) if B["kind"] == "phone" else B["eye"]
        ca, cb = cyl(ea, c), cyl(eb, c)
        g = w.get("glide", 0.0)
        az = ca[0] + g * wrap180(cb[0] - ca[0])
        eye = uncyl(az, ca[1] + g * (cb[1] - ca[1]), ca[2] + g * (cb[2] - ca[2]), c)
        dr = _nlerp(A["dir"], B["dir"], w.get("rot", g))
        za = math.tan(math.radians(A["fov"]) / 2)
        zb = math.tan(math.radians(B["fov"]) / 2)
        z = w.get("zoom", g)
        fov = 2 * math.degrees(math.atan(za ** (1 - z) * zb ** z))
        return {"kind": "way", "eye": eye, "dir": dr, "fov": fov, "c": c, "in_way": k, "typ": typ, "u": u, "k": k}

    def sample(self, hz: float | None = None) -> dict:
        hz = hz or self.cfg4.sample_hz
        ts = np.arange(0.0, self.F, 1.0 / hz)
        eye = np.zeros((len(ts), 3))
        dr = np.zeros((len(ts), 3))
        cc = np.zeros((len(ts), 3))
        fov = np.zeros(len(ts))
        way = np.full(len(ts), -1)
        kind = []
        el = np.full(len(ts), np.nan)
        az_fix = np.full(len(ts), np.nan)
        for i, f in enumerate(ts):
            p = self.pose(float(f))
            eye[i], dr[i], cc[i], fov[i] = p["eye"], _unit(p["dir"]), p["c"], p["fov"]
            way[i] = p["in_way"] if p.get("in_way") is not None else -1
            kind.append(p["kind"] if p.get("in_way") is None else "way")
            if p["kind"] in ("overhead", "crane") and p.get("az") is not None:
                az_fix[i] = p["az"]
            if "el" in p:
                el[i] = p["el"]
        rel = eye - cc
        a = np.arctan2(rel[:, 2], rel[:, 0])
        a = np.where(np.isfinite(az_fix), np.radians(az_fix), a)
        az = np.degrees(np.unwrap(a))
        return {"t": ts, "eye": eye, "dir": dr, "c": cc, "fov": fov, "way": way, "kind": np.array(kind), "az": az, "el": el, "hz": hz}


def rates(sm: dict) -> dict:
    """per sample: the eye's turn about the couple (deg/s), the view direction's turn (deg/s), the eye's speed (m/s) - central differences;
    samples next to a dance-time jump (the replay's edges: the couple's centre jumps) are NaN"""
    t, hz = sm["t"], sm["hz"]
    n = len(t)
    az = sm["az"]
    w = np.full(n, np.nan)
    v = np.full(n, np.nan)
    rr = np.full(n, np.nan)
    w[1:-1] = (az[2:] - az[:-2]) * hz / 2
    v[1:-1] = np.linalg.norm(sm["eye"][2:] - sm["eye"][:-2], axis=1) * hz / 2
    d = sm["dir"]
    dot = np.clip(np.sum(d[2:] * d[:-2], axis=1), -1, 1)
    rr[1:-1] = np.degrees(np.arccos(dot)) * hz / 2
    jump = np.zeros(n, bool)
    cj = np.linalg.norm(np.diff(sm["c"], axis=0), axis=1) * hz > 3.0          # the centre moving > 3 m/s: a dance-time jump
    for i in np.nonzero(cj)[0]:
        jump[max(0, i - 2):i + 4] = True
    for a in (w, v, rr):
        a[jump] = np.nan
    return {"az_deg_s": w, "rot_deg_s": rr, "speed_m_s": v}


# ------------------------------------------------------------------------------------------------ v4: the numbers of a direction (before / after)


def film_metrics(doc: dict, geo4: Geo4 | None, cfg4: V4Cfg | None = None, sample: bool = True, aspect: str = "vertical") -> dict:
    """What the owner asked about, measured the same way on any flow direction (v3.2 and v4):
      moves - every camera move: each way between two views (flight, crane, phone in / out, phone to phone, replay) + every separate 3D
              re-framing inside a shot (none: a shot is one camera block; its drift is the orbit's continuous motion, not a move);
      phone holds - the phone's own full-frame video between its arrival and the start of the way out (the last one: to the film's end);
      3D - the film minus the full-frame phone video (the orbit time + the ways); the orbit's planned rates; the ways' durations;
      peaks (sampled at 30 Hz from PoseModel) - per way the fastest turn about the couple, view turn, eye speed, crane pitch."""
    cfg4 = cfg4 or V4Cfg()
    segs = doc["segments"]
    shots = shots_of(segs)
    F = float(doc["timeline"]["film_duration_s"])
    ways = []
    for k, sh in enumerate(shots[1:], 1):
        tr = segs[sh.segs[0]]["camera"].get("transition_in")
        if tr:
            ways.append({"k": k, "type": tr["type"], "start": sh.f0 + tr["start_s"], "T": float(tr["duration_s"]), "crane": bool(tr.get("crane")),
                         "planned_peak": round(1.875 * abs(tr["arc"]["delta_deg"]) / max(1e-3, tr["duration_s"] * (1 - (0.3 if "axis" in str(tr.get("path")) else 0.0))), 1)
                         if tr.get("arc") else None})
    by_type: dict[str, int] = {}
    for w in ways:
        by_type[w["type"]] = by_type.get(w["type"], 0) + 1
    reframes = 0                                       # a camera block change inside a shot cannot happen (a shot IS one camera block)
    holds = []
    for k, sh in enumerate(shots):
        if sh.kind != "phone":
            continue
        tr = segs[sh.segs[0]]["camera"].get("transition_in")
        a = sh.f0 + (tr["start_s"] + tr["duration_s"] if tr else 0.0)
        ntr = segs[shots[k + 1].segs[0]]["camera"].get("transition_in") if k + 1 < len(shots) else None
        b = shots[k + 1].f0 + ntr["start_s"] if ntr else sh.f1
        holds.append({"shot": segs[sh.segs[0]]["id"], "phone": sh.phone, "film": [round(a, 2), round(b, 2)], "hold_s": round(b - a, 2)})
    hs = [h["hold_s"] for h in holds]
    full = sum(hs)
    pans, zooms, wts = [], [], []
    for k, sh in enumerate(shots):
        crop = segs[sh.segs[0]]["camera"].get("crop") if sh.kind == "phone" else None
        if not crop or len(crop.get("cx") or []) < 3:
            continue
        dt = float(crop["dt"])
        i0 = int(max(0, round((sh.d0 - float(crop["t0"])) / dt)))
        i1 = int(min(len(crop["cx"]) - 1, round((sh.d1 - float(crop["t0"])) / dt)))
        if i1 - i0 < 2:
            continue
        cx, cy, z = (np.asarray(crop[key][i0:i1 + 1], float) for key in ("cx", "cy", "zoom"))
        # the view's centre moves (in view widths per second: the crop's -1..1 frame units / 2 x zoom), and its zoom changes (relative per second)
        pans.append(float(np.mean(np.hypot(np.diff(cx), np.diff(cy)) / 2.0 * z[1:] / dt)))
        zooms.append(float(np.mean(np.abs(np.diff(np.log(z))) / dt)))
        wts.append(sh.f1 - sh.f0)
    way_s = sum(w["T"] for w in ways)
    # 3D views: time between ways (not phone), and the planned drift rates (time-weighted)
    d3, d3_rates = 0.0, []
    for k, sh in enumerate(shots):
        if sh.kind == "phone":
            continue
        tr = segs[sh.segs[0]]["camera"].get("transition_in")
        a = sh.f0 + (tr["start_s"] + tr["duration_s"] if tr else 0.0)
        ntr = segs[shots[k + 1].segs[0]]["camera"].get("transition_in") if k + 1 < len(shots) else None
        b = shots[k + 1].f0 + ntr["start_s"] if ntr else sh.f1
        d3 += max(0.0, b - a)
        cam = segs[sh.segs[0]]["camera"]
        d3_rates.append((abs(float(cam.get("drift_deg_s", 0.0) or 0.0)), max(0.0, b - a), sh.kind))
    q = lambda xs, p: round(float(np.percentile(xs, p)), 2) if len(xs) else None
    out = {"film_s": round(F, 3), "shots": len(shots),
           "moves": {"total": len(ways) + reframes, "ways": len(ways), "in_shot_reframes": reframes, "by_type": by_type,
                     "cranes": sum(1 for w in ways if w["type"] in ("crane_up", "crane_down") or w["crane"])},
           "phone_holds": {"n": len(hs), "mean_s": round(float(np.mean(hs)), 2) if hs else 0.0, "median_s": q(hs, 50) or 0.0,
                           "min_s": round(min(hs), 2) if hs else 0.0, "max_s": round(max(hs), 2) if hs else 0.0, "sum_s": round(full, 1),
                           "share": round(full / F, 3), "list": holds,
                           "crop_pan_view_per_s": round(float(np.average(pans, weights=wts)), 4) if pans else None,
                           "crop_zoom_rel_per_s": round(float(np.average(zooms, weights=wts)), 4) if zooms else None},
           "ways": {"n": len(ways), "sum_s": round(way_s, 1), "share": round(way_s / F, 3), "min_s": round(min((w["T"] for w in ways), default=0), 2),
                    "median_s": q([w["T"] for w in ways], 50), "max_s": round(max((w["T"] for w in ways), default=0), 2),
                    "phone_ways_median_s": q([w["T"] for w in ways if w["type"] in AXIS_TYPES], 50),
                    "planned_peak_max_deg_s": max((w["planned_peak"] for w in ways if w["planned_peak"] is not None), default=None)},
           "three_d": {"settled_s": round(d3, 1), "settled_share": round(d3 / F, 3), "with_ways_s": round(F - full, 1), "with_ways_share": round((F - full) / F, 3),
                       "static_s": round(sum(t for r, t, kd in d3_rates if r < 0.5 and kd != "overhead"), 1),
                       "drift_rates_deg_s": sorted({round(r, 2) for r, t, kd in d3_rates}),
                       "drift_time_weighted_deg_s": round(sum(r * t for r, t, _ in d3_rates) / max(1e-6, sum(t for _, t, _ in d3_rates)), 2)}}
    if sample and geo4 is not None:
        pm = PoseModel(doc, geo4, cfg4=cfg4, aspect=aspect)
        sm = pm.sample()
        rt = rates(sm)
        in_way = sm["way"] >= 0
        ph_hold = np.zeros(len(sm["t"]), bool)
        for h in holds:
            ph_hold |= (sm["t"] >= h["film"][0]) & (sm["t"] < h["film"][1])
        settled3d = ~in_way & ~ph_hold
        az = np.abs(rt["az_deg_s"])
        ok = np.isfinite(az)
        s3 = az[settled3d & ok]
        out["three_d"]["sampled_az_deg_s"] = {"median": q(s3, 50), "p90": q(s3, 90), "max": round(float(np.nanmax(s3)), 1) if len(s3) else None,
                                              "static_share": round(float(np.mean(s3 < 1.0)), 3) if len(s3) else None}
        # every 3D second incl. the ways (what the viewer sees as 3D motion)
        all3 = az[~ph_hold & ok]
        out["three_d"]["sampled_az_incl_ways_deg_s"] = {"median": q(all3, 50), "p90": q(all3, 90), "p99": q(all3, 99), "max": round(float(np.nanmax(all3)), 1) if len(all3) else None}
        per = []
        for w in ways:
            m = (sm["way"] == w["k"])
            if not m.any():
                continue
            el = sm["el"][m]
            pitch = np.abs(np.gradient(el) * sm["hz"]) if np.isfinite(el).sum() > 3 else np.array([0.0])
            per.append({"k": w["k"], "type": w["type"], "start": round(w["start"], 2), "T": w["T"],
                        "peak_az_deg_s": round(float(np.nanmax(az[m])), 1) if np.isfinite(az[m]).any() else None,
                        "peak_rot_deg_s": round(float(np.nanmax(rt["rot_deg_s"][m])), 1) if np.isfinite(rt["rot_deg_s"][m]).any() else None,
                        "peak_speed_m_s": round(float(np.nanmax(rt["speed_m_s"][m])), 2) if np.isfinite(rt["speed_m_s"][m]).any() else None,
                        "peak_pitch_deg_s": round(float(np.nanmax(pitch)), 1) if np.isfinite(pitch).any() else None})
        out["ways"]["peaks"] = per
        for key in ("peak_az_deg_s", "peak_rot_deg_s", "peak_speed_m_s"):
            # the ground-level ways (a crane is a rotation about the couple: its pitch has its own cap; the replay's ways move only the centre)
            xs = [p[key] for p in per if p[key] is not None and p["type"] not in ("crane_up", "crane_down", "replay_in", "replay_out")]
            out["ways"][key] = {"max": max(xs) if xs else None, "median": q(xs, 50)}
        cr = [p["peak_pitch_deg_s"] for p in per if p["type"] in ("crane_up", "crane_down") and p["peak_pitch_deg_s"] is not None]
        out["ways"]["crane_pitch_deg_s"] = {"max": max(cr) if cr else None, "median": q(cr, 50)}
        out["_sample"] = {"t": sm["t"], "az": sm["az"], "eye_h": sm["eye"][:, 1], "az_rate": rt["az_deg_s"], "rot_rate": rt["rot_deg_s"], "speed": rt["speed_m_s"],
                          "way": sm["way"], "ph_hold": ph_hold, "kind": sm["kind"], "eye": sm["eye"], "c": sm["c"], "dir": sm["dir"], "fov": sm["fov"]}
    return out


# ------------------------------------------------------------------------------------------------ v4: the plan (long phone holds, slow orbits)


def v4_protected(segs: list[dict], cfg4: V4Cfg) -> list[dict]:
    """the film windows that must be 3D (the orbit) with room for their ways: the opening under the title card, the 6 pivots, the 4 overheads
    (+ their cranes, which run in the orbit round them), the replay (+ its recentring ways)"""
    out = []
    s0 = segs[0]
    card = s0.get("chapter_card") or {}
    out.append({"kind": "opening", "f": [0.0, round(s0["film"][0] + float(card.get("film_out", 0.0)) + 0.6, 3)], "id": s0["id"]})
    crane = cfg4.dur["crane_up"][1]
    rp_ = cfg4.dur["replay_in"][1]
    i = 0
    while i < len(segs):
        s = segs[i]
        m = s["camera"].get("mode")
        if s.get("pivot"):
            out.append({"kind": "pivot", "f": [s["film"][0], s["film"][1]], "id": s["id"]})
        elif m == "overhead_floorcraft":
            j = i
            while j + 1 < len(segs) and segs[j + 1]["camera"].get("mode") == "overhead_floorcraft":
                j += 1
            # the owner's 4 overheads (v3.2: 5.3-6.2 s each) keep their crane; a base overhead shorter than overhead_min_s (a 1.6-s / 2.7-s
            # remnant the v3 joins folded into a pivot / a phone) is not one: its floor craft rides in the orbit or on a phone
            keep = segs[j]["film"][1] - s["film"][0] >= cfg4.overhead_min_s
            for x in segs[i:j + 1]:
                x["v4_overhead"] = keep
            if keep:
                c0, c1 = s["film"][0], segs[j]["film"][1]
                if cfg4.version == "4.1":
                    # v4.1: the crane up arrives arrive_after_s into the overhead, the crane down leaves at its end - or up to overhead_leave_early_s
                    # before it (never before the overhead's 2.2-s hold) when a phone hold after it needs the room; the orbit holds overhead_gap_s
                    # between a phone's way and a crane
                    up = c0 + cfg4.arrive_after_s
                    down = max(up + cfg4.hold_s["overhead"], c1 - cfg4.overhead_leave_early_s)
                    f = [round(up - crane - cfg4.overhead_gap_s, 3), round(down + crane + cfg4.overhead_gap_s, 3)]
                else:
                    f = [round(c0 - crane, 3), round(c1 + crane, 3)]
                out.append({"kind": "overhead", "f": f, "core": [c0, c1], "id": s["id"]})
            i = j
        elif not _locked(s):
            out.append({"kind": "replay", "f": [round(s["film"][0] - 1.0, 3), round(s["film"][1] + rp_ + 0.5, 3)], "core": [s["film"][0], s["film"][1]], "id": s["id"]})
        i += 1
    return out


def v4_weight_kind(s: dict) -> str:
    if s.get("pivot") or s.get("replay") or not _locked(s) or s.get("v4_overhead"):
        return "protected"
    m = s["camera"].get("mode")
    if m == "overhead_floorcraft":
        return "zoom"
    if shot_kind(s["camera"]) == "phone":
        return "phone"
    if m == "physics_3q":
        return "physics_3q"
    if m in ("feature_zoom", "feet_closeup", "rhythm"):
        return "zoom"
    return "orbit"


class FilmGrid:
    """per 0.1 film s: the planner's value of showing a phone / the orbit there, as prefix sums (integrals over any window)"""

    def __init__(self, segs: list[dict], F: float, cfg4: V4Cfg, prot: list[dict]):
        self.dt = 0.1
        self.t = np.arange(0.0, F + self.dt, self.dt)
        wp = np.zeros(len(self.t))
        wo = np.zeros(len(self.t))
        for s in segs:
            a, b = int(round(s["film"][0] / self.dt)), int(round(s["film"][1] / self.dt))
            k = v4_weight_kind(s)
            wp[a:b], wo[a:b] = cfg4.weights[k]
        for p in prot:
            a, b = int(round(max(0.0, p["f"][0]) / self.dt)), int(round(p["f"][1] / self.dt))
            wp[a:b] = 0.0
        self.cp = np.concatenate([[0.0], np.cumsum(wp) * self.dt])
        self.co = np.concatenate([[0.0], np.cumsum(wo) * self.dt])

    def _i(self, f: float) -> int:
        return int(np.clip(round(f / self.dt), 0, len(self.cp) - 1))

    def phone(self, a: float, b: float) -> float:
        return float(self.cp[self._i(b)] - self.cp[self._i(a)])

    def orbit(self, a: float, b: float) -> float:
        return float(self.co[self._i(b)] - self.co[self._i(a)])


class PhoneQ:
    """can phone `cam` hold the full frame over the master window [d0, d1]? It must record there (+ the ways' margin), see both teachers well over the
    whole window (recap_pov.window_quality 'strong': cover >= min_cover) and in every hold_sub_s piece of it (window_quality not None)"""

    def __init__(self, pov: director.PovSource, geo4: Geo4, cfg4: V4Cfg, cams: list[str] | None = None):
        self.pov, self.g, self.cfg4 = pov, geo4, cfg4
        self.step = 0.5
        self.grid = np.arange(0.0, 366.0, self.step)
        self.cams = [c for c in (cams or sorted(pov.phones)) if c in geo4.ph]
        self.ok2 = {}
        for cam in self.cams:
            ph = pov.phones[cam]
            ok = np.array([rp.window_quality(ph, float(t), float(t) + cfg4.hold_sub_s, pov.bad.get(cam, []), pov.cfg) is not None for t in self.grid])
            self.ok2[cam] = np.concatenate([[0], np.cumsum(~ok)])
        self.cache: dict = {}

    def hold(self, cam: str, d0: float, d1: float, margin: float = 0.0) -> dict | None:
        key = (cam, round(d0, 2), round(d1, 2), round(margin, 2))
        if key in self.cache:
            return self.cache[key]
        r0, r1 = self.g.recording(cam)
        res = None
        if d0 - margin >= r0 and d1 + margin <= r1:
            i0 = int(math.ceil(d0 / self.step - 1e-9))
            i1 = int(math.floor((d1 - self.cfg4.hold_sub_s) / self.step + 1e-9))
            bad = self.ok2[cam]
            if i1 < i0 or bad[min(len(bad) - 1, i1 + 1)] - bad[i0] == 0:
                q = rp.window_quality(self.pov.phones[cam], d0, d1, self.pov.bad.get(cam, []), self.pov.cfg)
                if q is not None and q.get("strong"):
                    res = q
        self.cache[key] = res
        return res


def orbit_excess(az_from: float, az_to: float, L: float, T_way: float, cfg4: V4Cfg) -> float:
    """the least turn (deg) the two ways of an orbit stretch must make beyond their natural ease into / out of the orbit, over every rate in
    the band (either direction, up to two extra turns round): orbit_rate_plan's `excess` without its preferences - what the DP needs to know
    whether the stretch can link its two phones under the caps"""
    lo, hi = cfg4.orbit_rate_deg_s
    span = max(1e-6, L - T_way)                  # Lp + 0.5 (T_out + T_in) with T_out = T_in = T_way
    D0 = wrap180(az_to - az_from)
    best = 1e9
    lead = cfg4.orbit_lead
    for kk in (-2, -1, 0, 1, 2):
        D = D0 + 360.0 * kk
        for sgn in (1.0, -1.0):
            if lead <= 0.0:
                w = sgn * min(hi, max(lo, abs(D) / span)) if D * sgn > 0 else sgn * lo
                best = min(best, abs(D - w * span))
                continue
            # v4.1 (never reversing): the ways must turn the orbit's way by at least `lead` x |w| x their time beyond the natural ease, so the
            # phone entered is ahead of the orbit (and the one left behind it) and no frame turns against the orbit; the excess is that turn
            need = 2.0 * lead * T_way
            if D * sgn <= 0:
                continue
            wabs = min(hi, max(lo, abs(D) / (span + need)))
            ex = abs(D) - wabs * span
            if ex < wabs * need - 1e-6:
                continue
            best = min(best, ex)
    return best


def v4_dp(B: list[float], cams: list[str], hold_q, p2p_T, free, S_or, S_ph, cfg4: V4Cfg, max_moves: int, hold_min: float,
          open_end: float, chapter_starts: list[float] = (), az_of=None) -> tuple[list[dict], dict]:
    """Dynamic programming over the boundary candidates B (film s, the speech's pauses; B[0] = 0, B[-1] = the film's end): a sequence of runs
    (orbit | phone cam) covering the film, the first an orbit (the opening), maximising the value of what is shown (S_or / S_ph) less
    `move_penalty` per way, with at most `max_moves` ways in and out of the phones. Constraints: a phone run lasts hold_min..hold_max (after the
    halves of its phone-to-phone ways) and is `free` of protected windows; an orbit run between two phones keeps a way_margin_s at each end and
    pure_orbit_min_s between them, with its protected windows outside the margins, and must be able to carry the camera from the phone it leaves
    to the phone it enters with ways under the caps (`az_of(cam, f)`: the phone's azimuth about the couple; orbit_excess <= max_excess_deg); a
    phone to phone way (ONE move) only where `p2p_T` gives a duration. hold_q(cam, a, b) -> quality dict | None; p2p_T(cam_a, cam_b, f) -> s |
    None; free(a, b) -> bool; S_or(a, b), S_ph(cam, a, b, q). Modes: ('O', next cam | None) for an orbit, (cam, p2p start, may end in p2p) for a
    phone. Returns (runs [{kind, cam, f0, f1, q}], info)."""
    n = len(B)
    M = max_moves
    NEG = -1e18
    mg = cfg4.way_margin_s
    pure = cfg4.pure_orbit_min_s
    half_max = cfg4.dur["phone_to_phone"][1] / 2.0
    best: list[dict] = [dict() for _ in range(n)]
    back: list[dict] = [dict() for _ in range(n)]

    def bonus(f):
        return cfg4.chapter_bonus if any(abs(f - c) <= 1.0 for c in chapter_starts) else 0.0

    def put(j, mode, cand, i, pmode):
        """cand: the values per number of ways (already shifted by this run's way); keep the better ones with their back pointers"""
        if mode not in best[j]:
            best[j][mode] = np.full(M + 1, NEG)
            back[j][mode] = [None] * (M + 1)
        cur = best[j][mode]
        for m in np.nonzero(cand > cur)[0]:
            cur[m] = cand[m]
            back[j][mode][m] = (i, pmode, int(m) - 1 if pmode is not None else 0)

    def shift(arr, val):
        out = np.full(M + 1, NEG)
        out[1:] = arr[:-1] + val
        out[out < NEG / 2] = NEG
        return out

    # the opening orbit [0, B[j]] (free start azimuth), toward any first phone (or the whole film)
    for j in range(1, n):
        b = B[j]
        final = j == n - 1
        tail = 0.0 if final else mg
        if b - tail < max(open_end, pure) - 1e-6 or b > cfg4.orbit_max_s:
            continue
        if not final and not free(b - tail, b):
            continue
        v = np.full(M + 1, NEG)
        v[0] = S_or(0.0, b)
        for nc in ([None] if final else cams):
            put(j, ("O", nc), v, 0, None)
    excess_cache: dict = {}
    for j in range(1, n):
        final = j == n - 1
        for i in range(j - 1, 0, -1):
            L = B[j] - B[i]
            if L > max(cfg4.orbit_max_s, cfg4.hold_max_s):
                break
            ph_modes = [md for md in best[i] if md[0] != "O"]
            # an orbit run after a phone, toward the next phone (or to the film's end)
            if ph_modes and L <= cfg4.orbit_max_s:
                tail = 0.0 if final else mg
                if L - mg - tail >= pure - 1e-6 and free(B[i], B[i] + mg) and (final or free(B[j] - tail, B[j])):
                    val = S_or(B[i], B[j]) + bonus(B[i]) - cfg4.move_penalty
                    by_cam: dict = {}
                    for md in ph_modes:
                        by_cam.setdefault(md[0], []).append(md)
                    for pc, mds in by_cam.items():
                        for nc in ([None] if final else list(cams)):
                            pen = 0.0
                            if nc is not None and az_of is not None:
                                key = (pc, i, nc, j)
                                ex = excess_cache.get(key)
                                if ex is None:
                                    ex = orbit_excess(az_of(pc, B[i]), az_of(nc, B[j]), L, mg, cfg4)
                                    excess_cache[key] = ex
                                if ex > cfg4.max_excess_deg:
                                    continue
                                pen = cfg4.excess_penalty * ex / max(1.0, cfg4.max_excess_deg)
                            if nc is not None and nc == pc:
                                pen += cfg4.same_phone_penalty
                            for md in mds:
                                put(j, ("O", nc), shift(best[i][md], val - pen), i, md)
            # a phone run [B[i], B[j]]
            if hold_min - 1e-6 <= L <= cfg4.hold_max_s and free(B[i], B[j]):
                for cam in cams:
                    q = hold_q(cam, B[i], B[j])
                    if q is None:
                        continue
                    val = S_ph(cam, B[i], B[j], q) + bonus(B[i]) - cfg4.move_penalty
                    om = ("O", cam)
                    if om in best[i]:
                        put(j, (cam, 0, L - half_max >= hold_min), shift(best[i][om], val), i, om)
                    for md in ph_modes:
                        if md[0] == cam or not md[2]:
                            continue
                        T = p2p_T(md[0], cam, B[i])
                        if T is None or L - T / 2 < hold_min:
                            continue
                        put(j, (cam, 1, L - T / 2 - half_max >= hold_min), shift(best[i][md], val), i, md)
    # the best end
    endv, endk = NEG, None
    for md, arr in best[n - 1].items():
        if md[0] == "O" and md[1] is not None:
            continue
        m = int(np.argmax(arr))
        if arr[m] > endv:
            endv, endk = float(arr[m]), (md, m)
    if endk is None or endv < NEG / 2:
        return [], {"ok": False, "why": "no plan satisfies the constraints"}
    runs = []
    j, (md, m) = n - 1, endk
    while True:
        i, pmd, pm = back[j][md][m]
        runs.append({"kind": "orbit" if md[0] == "O" else "phone", "cam": None if md[0] == "O" else md[0], "f0": B[i], "f1": B[j], "q": None})
        if pmd is None:
            break
        j, md, m = i, pmd, pm
    runs.reverse()
    for r in runs:
        if r["kind"] == "phone":
            r["q"] = hold_q(r["cam"], r["f0"], r["f1"])
    return runs, {"ok": True, "value": round(endv, 2), "moves_phone": endk[1]}


def orbit_rate_plan(az_from: float | None, az_to: float | None, L: float, T_out: float, T_in: float, cfg4: V4Cfg, prefer_sign: float = 1.0,
                    A_out: float = 0.0, A_in: float = 0.0) -> dict:
    """one orbit stretch of L film s between the phone it leaves (azimuth az_from) and the phone it enters (az_to): its rate w (deg/s, |w| in the
    band) and its azimuth `a_out` when the way out of the phone ends (T_out after the stretch starts). A way blends LIVE states, so it turns by
    itself about half the orbit's own rate x its duration while it eases into (out of) the orbit's motion: r_out = a_out - az_from and
    r_in = az_to - (a_out + w x Lp) are each that natural turn + an excess; the excess (what makes a way faster than the orbit) is minimised and
    shared by the two ways in proportion to their durations; a way that would turn against the orbit's direction (the camera would reverse) is
    penalised; |w| near the preferred rate. az_from / az_to None = the film's start / end (free)."""
    lo, hi = cfg4.orbit_rate_deg_s
    Lp = max(0.0, L - T_out - T_in)                 # the stretch's own orbit between the two ways
    pref = cfg4.orbit_rate_pref_deg_s
    nat = 0.5 + cfg4.orbit_lead                     # the share of w x T a way turns by itself (v4.1: + the lead, so it never turns against the orbit)
    if az_from is None and az_to is None:
        return {"w": prefer_sign * pref, "a_out": 0.0, "r_out": 0.0, "r_in": 0.0, "excess_deg": 0.0}
    if az_from is None:
        w = prefer_sign * pref
        r_in = nat * w * T_in
        return {"w": w, "a_out": az_to - r_in - w * Lp, "r_out": 0.0, "r_in": round(r_in, 2), "excess_deg": 0.0}
    if az_to is None:
        w = prefer_sign * pref
        r_out = nat * w * T_out
        return {"w": w, "a_out": az_from + r_out, "r_out": round(r_out, 2), "r_in": 0.0, "excess_deg": 0.0}
    if cfg4.version == "4.1":
        lk = orbit_link(az_from, az_to, L, T_out, T_in, cfg4, prefer_sign, A_out, A_in)
        if lk is not None:
            return lk
    best = None
    D0 = wrap180(az_to - az_from)
    for strict in ((True, False) if cfg4.orbit_lead > 0 else (False,)):
        for sgn in (1.0, -1.0):
            for wabs in np.arange(lo, hi + 1e-9, 0.1):
                w = sgn * float(wabs)
                for kk in (-2, -1, 0, 1, 2):
                    D = D0 + 360.0 * kk
                    ex = D - w * (Lp + 0.5 * (T_out + T_in))            # what the two ways must turn beyond their natural ease into the orbit
                    if strict and ex * sgn < cfg4.orbit_lead * abs(w) * (T_out + T_in) - 1e-6:
                        continue                                          # v4.1: a way would turn against the orbit
                    e_out = ex * T_out / max(1e-6, T_out + T_in)
                    r_out, r_in = 0.5 * w * T_out + e_out, 0.5 * w * T_in + (ex - e_out)
                    against = sum(abs(r) for r in (r_out, r_in) if abs(r) > 2.0 and math.copysign(1.0, r) != sgn)
                    cost = 1.875 * abs(ex) / max(1e-6, T_out + T_in) + 0.5 * against + 0.4 * abs(abs(w) - pref) + (0.15 if sgn != prefer_sign else 0.0)
                    if best is None or cost < best[0]:
                        best = (cost, w, r_out, r_in, ex)
        if best is not None:
            break
    _, w, r_out, r_in, ex = best
    return {"w": round(w, 2), "a_out": az_from + r_out, "r_out": round(r_out, 2), "r_in": round(r_in, 2), "excess_deg": round(ex, 2)}


# ------------------------------------------------------------------------------------------------ v4: segments -> the plan's shots


V4_PHYS_KEYS = ("skeleton_mode", "balance", "counterbalance", "physics", "legend", "physics_limbs", "floorcraft", "floorcraft_labels", "floorcraft_current_t",
                "t_width_m", "t_brightness", "footprints", "spine_beads")


def v4_layers(s: dict, kind: str, cam: str | None, phone_tpl: dict, orbit_tpl: dict) -> dict:
    """a segment's layers for its v4 view: its own when the view kind is the base director's, else the template of the new kind with the
    segment's own physics / floor-craft keys carried over (apply_physics re-applies the physics layer afterwards)"""
    lay = s["layers"]
    was_phone = shot_kind(s["camera"]) == "phone"
    if kind == "phone":
        if was_phone:
            out = copy.deepcopy(lay)
        else:
            out = copy.deepcopy(phone_tpl)
            for k in V4_PHYS_KEYS:
                if k in lay:
                    out[k] = copy.deepcopy(lay[k])
        out["source_video"] = cam
        out["avatars_fade_with_video"] = True
        return out
    if not was_phone:
        out = copy.deepcopy(lay)
    else:
        out = copy.deepcopy(orbit_tpl)
        for k in V4_PHYS_KEYS:
            if k in lay:
                out[k] = copy.deepcopy(lay[k])
    for k in ("source_video", "avatars_fade_with_video"):
        out.pop(k, None)
    out["floor"] = True
    return out


def v4_assign(segs: list[dict], runs: list[dict], pov: director.PovSource, cfg4: V4Cfg) -> list[dict]:
    """split the segments at the runs' boundaries and give every segment its run's view: a phone run -> camera_tour of its phone (the crop
    re-made for the whole hold, smoothed over crop_smooth_s); an orbit run -> the slow orbit (provisional azimuths: v4_finish_orbits), an
    overhead keeps its framing inside the orbit, the replay is the orbit. Every shot boundary of v4 is a director's cut at a pause."""
    phone_tpl = next((copy.deepcopy(s["layers"]) for s in segs if shot_kind(s["camera"]) == "phone"), copy.deepcopy(plan.L_POV))
    phone_tpl.pop("source_video", None)
    orbit_tpl = next((copy.deepcopy(s["layers"]) for s in segs if s["camera"].get("mode") == "orbit"), copy.deepcopy(plan.L_TALK))
    for r in runs[1:]:
        if not split_at_film(segs, r["f0"]):
            raise fr.RecapError(f"v4: the run boundary {r['f0']:.2f} s is not in a locked segment")
    renumber(segs)
    starts = np.array([r["f0"] for r in runs])
    crops: dict[int, dict] = {}
    for ri, r in enumerate(runs):
        r["idx"] = ri
        if r["kind"] == "phone":
            d0 = locked_dance(segs, r["f0"] + 1e-3, r["f0"])
            d1 = locked_dance(segs, r["f1"] - 1e-3, r["f1"])
            ph = pov.phones[r["cam"]]
            # v4.1: the 9:16 crop puts the teachers lower (their box from about 23 % of the height: under the physics legend in the top band)
            pcfg = dataclasses_replace(pov.cfg, **cfg4.crop) if cfg4.crop else pov.cfg
            c_def = rp.crop_track(ph, d0, d1, pcfg)
            c_smo = rp.crop_track(ph, d0, d1, dataclasses_replace(pcfg, smooth_s=cfg4.crop_smooth_s))
            use = c_smo if c_smo["inside"] >= min(0.92, c_def["inside"] - 0.02) else c_def
            crops[ri] = {"crop": use, "smooth_s": cfg4.crop_smooth_s if use is c_smo else pov.cfg.smooth_s, "inside_default": c_def["inside"],
                         "inside": use["inside"], "d": [round(d0, 3), round(d1, 3)]} | ({"framing": dict(cfg4.crop)} if cfg4.crop else {})
            r["crop_info"] = {k: v for k, v in crops[ri].items() if k != "crop"}
    part = 0
    prev_run, prev_tag = None, None
    for s in segs:
        mid = 0.5 * (s["film"][0] + s["film"][1])
        ri = int(max(0, np.searchsorted(starts, mid, side="right") - 1))
        r = runs[ri]
        s["director"] = True
        s["v4_run"] = ri
        s.setdefault("v4_base_camera", s["camera"].get("mode"))
        if r["kind"] == "phone":
            cam = r["cam"]
            s["layers"] = v4_layers(s, "phone", cam, phone_tpl, orbit_tpl)
            s["camera"] = {"mode": "camera_tour", "cameras": [{"camera": cam, "dance_t": crops[ri]["d"]}], "cut": "flow", "blend_s": 0.0,
                           "composite": "3d_over_video", "skeletons_on_top": True, "scene_geometry": "all", "crop": crops[ri]["crop"],
                           "v4": {"role": "phone_hold", "run": ri, "crop_smooth_s": crops[ri]["smooth_s"]} | ({"crop_framing": dict(cfg4.crop)} if cfg4.crop else {})}
            s["view_state"] = "camera_tour"
            q = r.get("q") or {}
            s["treatment"] = {"kind": "pov", "phone": cam, "cover": q.get("cover"), "height": q.get("height"), "family": (s.get("treatment") or {}).get("family"),
                              "tail_of": None, "v4": "phone hold"}
            continue
        is_over = s["camera"].get("mode") == "overhead_floorcraft" and bool(s.get("v4_overhead"))
        tag = "overhead" if is_over else ("replay" if s.get("replay") else "orbit")
        if ri != prev_run or tag != prev_tag or s.get("replay") or (prev_tag == "replay"):
            part += 1
        prev_run, prev_tag = ri, tag
        s["layers"] = v4_layers(s, "3d", None, phone_tpl, orbit_tpl)
        if is_over:
            cam = copy.deepcopy(s["camera"])
            cam.pop("transition_in", None)
            cam["v4"] = {"role": "overhead", "run": ri, "part": part}
            s["camera"] = cam
            s["view_state"] = "overhead"
        else:
            cam = copy.deepcopy(cfg4.orbit_block)
            cam["v4"] = {"role": "slow_orbit", "run": ri, "part": part} | ({"replay": True} if s.get("replay") else {})
            s["camera"] = cam
            if s.get("view_state") == "camera_tour" or not s.get("view_state"):
                s["view_state"] = "orbit"
    return segs


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def v4_place(typ: str, prev: Shot, nxt: Shot, T: float, cfg4: V4Cfg) -> tuple[float, list[str]]:
    """where a v4 way starts (film s): into a phone it has ARRIVED at the phone's first frame (the hold starts at a pause); out of a phone it
    starts at the hold's last frame; phone to phone it is centred on the boundary; a crane up has arrived arrive_after_s into its overhead (or
    0.1 s before the overhead's first called word); a crane down starts at the overhead's end unless the view after it must have arrived 0.1 s
    before a called word; a replay's way starts at its edge (under the flash)"""
    notes = []
    f0 = nxt.f0
    if typ == "phone_in":
        return f0 - T, notes
    if typ == "phone_out":
        return f0, notes
    if typ == "phone_to_phone":
        return f0 - 0.5 * T, notes
    if typ == "crane_up":
        arr = f0 + cfg4.arrive_after_s
        if nxt.first_call is not None and nxt.first_call - f0 < 3.0:
            arr = min(arr, nxt.first_call - cfg4.call_arrive_before_s)
        return arr - T, notes
    if typ == "crane_down":
        st = f0
        if nxt.first_call is not None and nxt.first_call - f0 < T + 3.0:
            lat = nxt.first_call - cfg4.call_arrive_before_s - T
            if lat < st:
                notes.append(f"leaves the overhead {st - lat:.2f} s early: the view after it has arrived 0.1 s before its called word")
                st = lat
        return st, notes
    return f0, notes


def v4_fit_cranes(ways: list[dict], shots: list[Shot], cfg4: V4Cfg) -> None:
    """a crane down leaves its overhead no earlier than the overhead's hold (2.2 s after the crane up has arrived) and early enough that the
    orbit after it holds 1 s before the next way (e.g. the replay's flash); the called word after it comes first only when the overhead's hold
    leaves no other choice (noted)"""
    for idx, w in enumerate(ways):
        if w["type"] != "crane_down" or idx == 0:
            continue
        pw = ways[idx - 1]
        earliest = pw["start"] + pw["T"] + cfg4.hold_s["overhead"]
        nw = ways[idx + 1] if idx + 1 < len(ways) else None
        latest = nw["start"] - cfg4.hold_s["3d"] - w["T"] if nw else 1e9
        st = w["start"]
        if st > latest + 1e-6:
            w["notes"].append(f"leaves the overhead {st - latest:.2f} s earlier: the orbit after it holds {cfg4.hold_s['3d']} s before the next way")
            st = latest
        if st < earliest - 1e-6:
            w["notes"].append(f"arrives {earliest - st:.2f} s later than the called word asks: the overhead holds its {cfg4.hold_s['overhead']} s")
            st = earliest
        w["start"] = st


def v4_way_T(typ: str, cfg4: V4Cfg) -> float:
    """the base duration of a v4 way, never below the picture's readable floor"""
    lo, hi, base = cfg4.dur[typ]
    rd = picture_readable_u(typ, cfg4.prof())
    floor = max([cfg4.readable_s / u for u in rd.values() if u > 0] or [0.0])
    if typ in ("crane_up", "crane_down"):
        # the pitch from about 16 deg to 89.9 deg within the cap: smootherstep peaks at 1.875 x its mean rate, the cruise law at 1 / (1 - ease) x
        peak = 1.875 if cfg4.crane_ease is None else 1.0 / (1.0 - cfg4.crane_ease)
        floor = max(floor, peak * cfg4.crane_span_deg / cfg4.cap_crane_deg_s)
    return min(hi, math.ceil(max(lo, base, floor) * 100.0 - 1e-6) / 100.0)          # (rounded up: never under a floor)


def phone_aim(geo4: Geo4, cam: str, d: float, cfg4: V4Cfg) -> float:
    """the angle (deg) between phone `cam`'s aim at master second d and the slow orbit's view from the phone's azimuth (WayEst's `aim`)"""
    p = geo4.phone(cam, d)
    c = geo4.centre(d)
    portal = p["pos"] - cfg4.axis_run_m * p["fwd"]
    a = math.atan2(portal[2] - c[2], portal[0] - c[0])
    eye = np.array([c[0] + cfg4.orbit_r_h[0] * math.cos(a), cfg4.orbit_r_h[1], c[2] + cfg4.orbit_r_h[0] * math.sin(a)])
    return _ang(c + np.array([0.0, cfg4.orbit_look_h, 0.0]) - eye, p["fwd"])


def v4_set_orbits(segs: list[dict], shots: list[Shot], runs: list[dict], Ts: dict, geo4: Geo4, cfg4: V4Cfg, prefer: dict | None = None) -> dict:
    """the orbit plan of every orbit run (orbit_rate_plan with the ways' current durations) written into its shots' blocks: world_azimuth_deg
    = a(F0), drift_deg_s = w; an overhead turns with the orbit (up_policy 'orbit'); the replay orbits on. Returns {run: plan}."""
    plans = {}
    sign = 1.0
    for ri, r in enumerate(runs):
        if r["kind"] != "orbit":
            continue
        ks = [k for k, sh in enumerate(shots) if segs[sh.segs[0]].get("v4_run") == ri]
        if not ks:
            continue
        k0, k1 = ks[0], ks[-1]
        T_out = Ts.get(k0, 0.0) if k0 > 0 else 0.0
        T_in = Ts.get(k1 + 1, 0.0) if k1 + 1 < len(shots) else 0.0
        prev = runs[ri - 1] if ri > 0 else None
        nxt = runs[ri + 1] if ri + 1 < len(runs) else None
        az_from = cyl(geo4.phone(prev["cam"], locked_dance(segs, r["f0"] - 1e-3, r["f0"]))["pos"], geo4.centre(locked_dance(segs, r["f0"] - 1e-3, r["f0"])))[0] if prev else None
        az_to = cyl(geo4.phone(nxt["cam"], locked_dance(segs, r["f1"] + 1e-3, r["f1"]))["pos"], geo4.centre(locked_dance(segs, r["f1"] + 1e-3, r["f1"])))[0] if nxt else None
        if prev is None and nxt is not None:
            # the opening: turn the way that starts it side-on to the couple (the title card's framing)
            c0 = geo4.g.line_deg(0.5)
            best = None
            for sgn in (1.0, -1.0):
                pl = orbit_rate_plan(None, az_to, r["f1"] - r["f0"], 0.0, T_in, cfg4, sgn)
                d = min(abs(wrap180(pl["a_out"] - (c0 + 90))), abs(wrap180(pl["a_out"] - (c0 - 90))))
                if best is None or d < best[0]:
                    best = (d, pl)
            pl = best[1]
        else:
            aims = {}
            if cfg4.version == "4.1":
                aims = {"A_out": phone_aim(geo4, prev["cam"], locked_dance(segs, r["f0"] - 1e-3, r["f0"]), cfg4),
                        "A_in": phone_aim(geo4, nxt["cam"], locked_dance(segs, r["f1"] + 1e-3, r["f1"]), cfg4)}
            pl = orbit_rate_plan(az_from, az_to, r["f1"] - r["f0"], T_out, T_in, cfg4, (prefer or {}).get(ri, sign), **aims)
        t_ref = r["f0"] + T_out
        pl.update({"t_ref": round(t_ref, 3), "az_from": None if az_from is None else round(az_from, 1), "az_to": None if az_to is None else round(az_to, 1),
                   "T_out": T_out, "T_in": T_in, "L": round(r["f1"] - r["f0"], 2)})
        plans[ri] = pl
        sign = math.copysign(1.0, pl["w"])
        for k in ks:
            sh = shots[k]
            a = pl["a_out"] + pl["w"] * (sh.f0 - t_ref)
            for i in sh.segs:
                cam = segs[i]["camera"]
                cam["world_azimuth_deg"] = round(wrap180(a), 2)
                cam["drift_deg_s"] = pl["w"]
                if cam.get("mode") == "overhead_floorcraft":
                    cam["up_policy"] = "orbit"
                    cam["up_azimuth_deg"] = round(wrap180(a), 2)
                else:
                    cam["ground_level"] = True
    return plans


def v4_schedule(segs: list[dict], runs: list[dict], geo4: Geo4, cfg4: V4Cfg, cfg: FlowCfg, log: list[str], iters: int | None = None,
                T0=None) -> dict:
    """the ways of the v4 plan: type, duration, start; durations grow (within their type's [lo, hi]) until the PoseModel's sampled peaks stay
    under the caps; the orbits are re-planned with the new durations; repeated until nothing changes. Writes transition_in blocks (provisional:
    write_flow_v4 completes them) and the orbits' azimuths. `T0(prev, nxt, typ)` -> a first duration (v4.1: the plan's estimate) or None.
    Returns {Ts, plans, ways, problems}."""
    iters = iters or cfg4.schedule_iters
    shots = shots_of(segs)
    first_calls(segs, shots)
    Ts = {}
    for k in range(1, len(shots)):
        typ = transition_type(shots[k - 1], shots[k])
        Ts[k] = v4_way_T(typ, cfg4)
        if T0 is not None:
            t0 = T0(shots[k - 1], shots[k], typ)
            if t0 is not None:
                Ts[k] = round(min(cfg4.dur[typ][1], max(Ts[k], t0)), 2)
    plans, ways, problems = {}, [], []
    for it in range(iters):
        plans = v4_set_orbits(segs, shots, runs, Ts, geo4, cfg4)
        ways = []
        for k in range(1, len(shots)):
            p, n = shots[k - 1], shots[k]
            typ = transition_type(p, n)
            T = Ts[k]
            st, notes = v4_place(typ, p, n, T, cfg4)
            ways.append({"k": k, "type": typ, "start": st, "T": T, "notes": notes})
        v4_fit_cranes(ways, shots, cfg4)
        for w in ways:
            n = shots[w["k"]]
            tr = {"type": w["type"], "start_s": round(w["start"] - n.f0, 3), "duration_s": round(w["T"], 3), "easing": "smootherstep",
                  "arc": {"delta_deg": 0.0, "dir": "none"}, "path": V4_PATH[w["type"]]}
            if cfg4.version != "4.0":
                tr["v4"] = {"profile": copy.deepcopy(cfg4.prof()[w["type"]])}      # the PoseModel samples the way with its own profile
            for i in n.segs:
                segs[i]["camera"]["transition_in"] = copy.deepcopy(tr)
        doc_tmp = {"segments": segs, "timeline": {"film_duration_s": segs[-1]["film"][1]}, "flow": {"version": 2}}
        pm = PoseModel(doc_tmp, geo4, cfg, cfg4)
        changed = False
        for w in ways:
            k, typ, st, T = w["k"], w["type"], w["start"], w["T"]
            if typ in ("replay_in", "replay_out"):
                continue
            ts = np.arange(st, st + T + 1e-9, 1.0 / 30.0)
            ps = [pm.pose(float(t)) for t in ts]
            eye = np.array([x["eye"] for x in ps])
            cc = np.array([x["c"] for x in ps])
            dr = np.array([_unit(x["dir"]) for x in ps])
            rel = eye - cc
            az = np.degrees(np.unwrap(np.arctan2(rel[:, 2], rel[:, 0])))
            g = lambda a: np.abs(np.gradient(a) * 30.0)
            peak_az = float(np.max(g(az))) if len(az) > 2 else 0.0
            peak_v = float(np.max(np.linalg.norm(np.gradient(eye, axis=0), axis=1) * 30.0)) if len(eye) > 2 else 0.0
            rot = np.degrees(np.arccos(np.clip(np.sum(dr[1:] * dr[:-1], axis=1), -1, 1))) * 30.0
            peak_rot = float(np.max(rot)) if len(rot) else 0.0
            w.update({"peak_az_deg_s": round(peak_az, 1), "peak_rot_deg_s": round(peak_rot, 1), "peak_speed_m_s": round(peak_v, 2)})
            if typ in ("crane_up", "crane_down"):
                el = np.array([x.get("el", np.nan) for x in ps])
                w["peak_pitch_deg_s"] = round(float(np.nanmax(g(el))), 1) if np.isfinite(el).sum() > 2 else None
                over = (w["peak_pitch_deg_s"] or 0.0) > cfg4.cap_crane_deg_s
            else:
                over = peak_az > cfg4.cap_az_deg_s or peak_rot > cfg4.cap_rot_deg_s or peak_v > cfg4.cap_speed_m_s
            w["over_cap"] = bool(over)
            if over and T < cfg4.dur[typ][1] - 1e-6:
                Ts[k] = round(min(cfg4.dur[typ][1], T + cfg4.schedule_step_s), 2)
                changed = True
        if not changed:
            break
        log.append(f"v4 schedule pass {it + 1}: lengthened {sum(1 for w in ways if w.get('over_cap'))} ways to keep the caps")
    # holds between the ways
    for k, sh in enumerate(shots):
        arr = sh.f0 if k == 0 else ways[k - 1]["start"] + ways[k - 1]["T"]
        dep = ways[k]["start"] if k < len(ways) else sh.f1
        kind = "replay" if sh.replay else sh.kind
        need = cfg4.hold_s.get(kind, 0.0) if kind != "phone" else None
        sh.arrive = arr
        if need is not None and dep - arr < need - 1e-6:
            problems.append(f"{segs[sh.segs[0]]['id']} ({kind}) holds {dep - arr:.2f} s < {need} s between its ways")
        if k > 0 and arr > sh.f1 + 1e-6 and k + 1 < len(shots):
            problems.append(f"{segs[sh.segs[0]]['id']}: the way into it ends after it")
    return {"Ts": Ts, "plans": plans, "ways": ways, "problems": problems, "shots": shots}


def write_flow_v4(segs: list[dict], shots: list[Shot], sched: dict, runs: list[dict], geo4: Geo4, cfg: FlowCfg, cfg4: V4Cfg) -> dict:
    """the v4 transition_in blocks (the v1 keys a v1 viewer understands + a `v4` block with the exact profile, zoom, fades, caps and the plan's
    peaks), the phones' camera windows (way in .. way out), the frustum envelopes of the phones entered / left. Returns the flow block's lists."""
    frustums = []
    ways = {w["k"]: w for w in sched["ways"]}
    for k, sh in enumerate(shots):
        cam = segs[sh.segs[0]]["camera"]
        if sh.kind == "phone":
            w_in, w_out = ways.get(k), ways.get(k + 1)
            a = w_in["start"] if w_in else sh.f0
            b = (w_out["start"] + w_out["T"]) if w_out else sh.f1
            da, db = locked_dance(segs, max(0.0, a), sh.d0), locked_dance(segs, b, sh.d1)
            if w_in is None or a < sh.f0:
                da = min(da, sh.d0 - (sh.f0 - a))
            if w_out is not None and b > sh.f1:
                db = max(db, sh.d1 + (b - sh.f1))
            cam["cameras"] = [{"camera": sh.phone, "dance_t": [round(min(da, sh.d0), 3), round(max(db, sh.d1), 3)]}]
        if k == 0:
            cam.pop("transition_in", None)
            for i in sh.segs:
                segs[i]["camera"] = copy.deepcopy(cam)
            continue
        w = ways[k]
        p = shots[k - 1]
        typ, T, st = w["type"], w["T"], w["start"]
        arrive = st + T
        prof = cfg4.prof()[typ]
        A = {"kind": p.kind, "mode": p.mode, "phone": p.phone}
        Bk = {"kind": sh.kind, "mode": sh.mode, "phone": sh.phone}
        pa = geo4.phone(p.phone, locked_dance(segs, st, p.d1)) if p.kind == "phone" else None
        pb = geo4.phone(sh.phone, locked_dance(segs, arrive, sh.d0)) if sh.kind == "phone" else None
        ca, cb = geo4.centre(locked_dance(segs, st, p.d1)), geo4.centre(locked_dance(segs, arrive, sh.d0))
        a_cam = segs[p.segs[0]]["camera"]
        b_cam = cam
        def end_est(kind, phn, c, camb, f, shot):
            if kind == "phone":
                ay, r, h = cyl(phn["pos"], c)
                return {"azimuth_deg": round(ay, 1), "radius_m": round(r, 2), "height_m": round(h, 2)}
            if kind == "overhead":
                return {"azimuth_deg": round(float(camb.get("world_azimuth_deg", 0.0)) + float(camb.get("drift_deg_s", 0.0)) * (f - shot.f0), 1), "radius_m": 0.0, "height_m": cfg4.overhead_h}
            return {"azimuth_deg": round(wrap180(float(camb.get("world_azimuth_deg", 0.0)) + float(camb.get("drift_deg_s", 0.0)) * (f - shot.f0)), 1),
                    "radius_m": cfg4.orbit_r_h[0], "height_m": cfg4.orbit_r_h[1]}
        A.update(end_est(p.kind, pa, ca, a_cam, st, p))
        Bk.update(end_est(sh.kind, pb, cb, b_cam, arrive, sh))
        daz = wrap180(Bk["azimuth_deg"] - A["azimuth_deg"]) if typ not in ("crane_up", "crane_down", "replay_in", "replay_out") else 0.0
        v1_path = {"phone_in": "arc_then_axis", "phone_out": "axis_then_arc", "phone_to_phone": "axis_arc_axis", "crane_up": "crane", "crane_down": "crane",
                   "replay_in": "arc", "replay_out": "arc", "flight": "arc"}[typ]
        block = {"type": typ, "start_s": round(st - sh.f0, 3), "duration_s": round(T, 3), "easing": "smootherstep", "path": v1_path,
                 "from": A, "to": Bk, "arc": {"delta_deg": round(daz, 1), "dir": "ccw" if daz > 0 else "cw" if daz < 0 else "none"},
                 "constraints": {"height_m": [cfg.ground_m[0], None] if typ in ("crane_up", "crane_down") else list(cfg.ground_m), "clear_radius_m": cfg.clear_radius_m,
                                 "keep_couple_in_view": True, "look": "the couple (both ends look at it; the aim never leaves it mid-way)"},
                 "rule_s": v4_way_T(typ, cfg4)}
        if w.get("notes"):
            block["squeezed"] = w["notes"]
        if typ in AXIS_TYPES:
            block["axis_run_m"] = cfg4.axis_run_m
            block["axis_share"] = round(1.0 - prof.get("dolly", prof.get("dolly_out", [0, 0.5]))[0], 2) if typ != "phone_to_phone" else 0.3
        if typ in ("replay_in", "replay_out"):
            block["from_pose"] = "orbit_frozen_centre"
            block["flash_s"] = cfg.flash_s
        fov_from = cfg4.orbit_fov_deg if p.kind != "phone" else None
        fov_to = cfg4.orbit_fov_deg if sh.kind != "phone" else None
        v4b = {"path": V4_PATH[typ], "ease": "smootherstep ramps of LIVE endpoint states (C2: position, velocity and acceleration of the departing view at u=0 "
                                           "and of the arriving view at u=1)", "profile": copy.deepcopy(prof),
               "fov": {"law": "log_tan", "u": prof.get("zoom", [0.0, 1.0]), "from": "live" if fov_from is None else "orbit", "to": "phone crop" if sh.kind == "phone" else "live",
                       "from_deg_est": fov_from, "to_deg_est": fov_to},
               "caps": {"az_deg_s": cfg4.cap_az_deg_s, "rot_deg_s": cfg4.cap_rot_deg_s, "speed_m_s": cfg4.cap_speed_m_s} | ({"pitch_deg_s": cfg4.cap_crane_deg_s} if typ in ("crane_up", "crane_down") else {}),
               "peaks_planned": {k2: w[k2] for k2 in ("peak_az_deg_s", "peak_rot_deg_s", "peak_speed_m_s", "peak_pitch_deg_s") if w.get(k2) is not None}}
        rd = picture_readable_u(typ, cfg4.prof())
        if rd:
            v4b["picture_readable_s"] = {r: round(u * T, 2) for r, u in rd.items()}
        if typ in ("replay_in", "replay_out"):
            v4b["centre"] = {"from": "the couple's centre frozen at the edge (pre-jump)", "to": "the live centre", "u": prof["centre"]}
        if typ in ("crane_up", "crane_down"):
            v4b["azimuth"] = "the orbit's live a(t) = world_azimuth_deg + drift_deg_s x (film - F0) of the ground-level shot, throughout"
        block["v4"] = v4b
        # the frustums of the phones entered / left: the picture's alpha follows the profile; the outline rises from the ghost's opacity
        fz = {}
        if typ in ("phone_in", "phone_to_phone"):
            pi = prof["picture_in"]
            env = [round(st + pi[0] * T, 3), round(st + pi[1] * T, 3), round(arrive, 3), round(arrive, 3)]
            oi = prof["outline_in"]
            fz["in"] = {"camera": sh.phone, "film": env, "role": "enter", "depth_m": cfg.frustum_depth_m, "video_full_at_film": round(arrive, 3),
                        "shot": segs[sh.segs[0]]["id"], "v4": {"picture": {"curve": "smootherstep", "film": env[:2]},
                                                              "outline": {"from": cfg4.ghost_opacity, "to": 1.0, "curve": "smootherstep",
                                                                          "film": [round(st + oi[0] * T, 3), round(st + oi[1] * T, 3)]}}}
            block["video"] = {"full_frame_at_s": round(arrive - sh.f0, 3), "splats_hidden_at_s": round(arrive - sh.f0, 3), "overlays_on_top_in_s": 0.5}
        if typ in ("phone_out", "phone_to_phone"):
            po = prof["picture_out"]
            env = [round(st, 3), round(st, 3), round(st + po[0] * T, 3), round(st + po[1] * T, 3)]
            oo = prof["outline_out"]
            fz["out"] = {"camera": p.phone, "film": env, "role": "leave", "depth_m": cfg.frustum_depth_m, "video_full_until_film": round(st, 3),
                         "shot": segs[p.segs[0]]["id"], "v4": {"picture": {"curve": "smootherstep", "film": env[2:]},
                                                              "outline": {"from": 1.0, "to": cfg4.ghost_opacity, "curve": "smootherstep",
                                                                          "film": [round(st + oo[0] * T, 3), round(st + oo[1] * T, 3)]}}}
            block["video_out"] = {"surface_from_s": round(st - sh.f0, 3), "splats_back_in_s": 0.5, "overlays_on_top_off_before_s": 0.35}
        if "in" in fz:
            frustums.append(fz["in"])
            block["frustum_in"] = {"camera": fz["in"]["camera"], "film": fz["in"]["film"]}
        if "out" in fz:
            frustums.append(fz["out"])
            block["frustum_out"] = {"camera": fz["out"]["camera"], "film": fz["out"]["film"]}
        cam["transition_in"] = block
        cam["blend_s"] = 0.0 if sh.kind == "phone" else round(T, 2)
        for i in sh.segs:
            segs[i]["camera"] = copy.deepcopy(cam)
    return {"frustums": frustums}


def v4_ghost_block(geo4: Geo4, cfg4: V4Cfg) -> dict:
    """the faint frustums of every phone (owner 2026-10-22: 'very, very faint camera frustums from the other phones at all times ... maybe 15%
    opacity'): drawn in every 3D view and way, OUTLINE ONLY (apex, four edges, the rim at ghost depth; no video), the phone's colour at 0.15,
    depth-tested, kept off the dancers (a screen-space keep-out round each dancer, hardest round the heads), never at the lens"""
    return {"enabled": True, "opacity": cfg4.ghost_opacity, "draw": "outline", "video": False, "depth_m": cfg4.ghost_depth_m, "line_px": 1.5,
            "colour": "phone (CameraVideoRig.ColourOf)", "phones": sorted(geo4.ph), "while_recording_only": True,
            "views": {"3d": True, "ways": True, "overhead": True, "phone_full_frame": False},
            "rise": "the phone being entered / left: outline alpha = max(opacity, its way's outline curve) and its picture as flow.frustums says",
            "depth_test": "ZTest LEqual against the scene depth (floor, room); ZWrite off; drawn before the splats' transparent pass",
            "keep_out": {"faces": {"radius_m": 0.28, "feather_m": 0.12, "what": "a sphere round each dancer's head (SMPL head joint + 0.08 m up): alpha 0 inside"},
                         "bodies": {"radius_m": 0.35, "height_m": [0.0, 1.95], "feather_m": 0.15,
                                    "what": "a vertical capsule round each dancer's pelvis: alpha 0 where the line lies between the eye and the capsule (screen-space)"}},
            "near_fade_m": [1.0, 2.0], "label": False}


# ------------------------------------------------------------------------------------------------ v4: build


def v4_boundaries(segs: list[dict], cuts: list[float], prot: list[dict]) -> list[float]:
    """the film seconds a v4 run may start / end at: the speech's pauses (director.cut_points) inside locked segments, outside the protected
    windows' interiors, at least 0.3 s apart; 0 and the film's end"""
    F = float(segs[-1]["film"][1])
    out = []
    for c in cuts:
        s = next((x for x in segs if _locked(x) and x["dance"][0] + 0.05 < c < x["dance"][1] - 0.05), None)
        if s is None:
            continue
        f = s["film"][0] + (c - s["dance"][0])
        if any(p["f"][0] + 1e-3 < f < p["f"][1] - 1e-3 for p in prot):
            continue
        out.append(round(f, 3))
    out = sorted(set(out))
    res = [0.0]
    for f in out:
        if f - res[-1] >= 0.3 and F - f >= 0.3:
            res.append(f)
    res.append(round(F, 3))
    return res


def v4_base(hand_events: Path | None = None) -> dict:
    """the base every version shares: the D2/D4 director's direction (V3_DIRCFG), the back-to-the-axis pivots carved in, the replay at ground level"""
    paths = fr.Paths()
    pcfg = plan.Cfg()
    cfg = FlowCfg()
    sp = fr.load_speech(paths)
    layers = plan.Layers.probe(CAP_DIR, True)
    act = fr.load_activity(paths, CAP_DIR)
    pov = fr.load_pov(paths, "yes")
    phrases = back_to_axis(sp["utterances"])
    events, dropped_ev = drop_axis_events(sp["events"], phrases)
    doc, tl, report = fr.build_direction(dict(sp, events=events), act, layers, pcfg, paths, CAPTURE, fr.film_title(), pov, V3_DIRCFG,
                                         arrow_ctx=fr.load_arrow_ctx(CAP_DIR, layers))
    base_bad = fr.validate(doc, tl, sp)
    if base_bad:
        raise fr.RecapError("the base direction does not validate:\n  " + "\n  ".join(base_bad[:20]))
    cb = load_counterbalance()
    cuts = director.cut_points(sp["utterances"], pcfg.lead_s)
    segs, pivot_rows = carve_pivots(doc["segments"], phrases, events, cb, cuts, cfg, pcfg)
    replay_notes = replay_to_ground(segs, cb, cfg, pcfg)
    renumber(segs)
    return {"paths": paths, "pcfg": pcfg, "cfg": cfg, "sp": sp, "act": act, "pov": pov, "phrases": phrases, "dropped_ev": dropped_ev, "doc": doc, "tl": tl,
            "report": report, "cb": cb, "cuts": cuts, "segs": segs, "pivot_rows": pivot_rows, "replay_notes": replay_notes,
            "hand_events": Path(hand_events) if hand_events else HAND_EVENTS}


def v4_baseline(geo4: Geo4, cfg4: V4Cfg, path: Path = V32_DIRECTION) -> dict:
    """the 'before': the v3.2 direction measured with film_metrics (the same estimator as v4)"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    m = film_metrics(d, geo4, cfg4)
    m["path"] = str(path)
    m["sha256_16"] = file_sha(path)
    return m


def build_v4(log=print, cfg4: V4Cfg | None = None, write: bool = True, hand_events: Path | None = None, baseline_path: Path = V32_DIRECTION,
             out: Path | None = None) -> dict:
    cfg4 = cfg4 or V4Cfg()
    b = v4_base(hand_events)
    segs, sp, pov, cfg, pcfg = b["segs"], b["sp"], b["pov"], b["cfg"], b["pcfg"]
    geo = Geo()
    geo4 = Geo4(geo)
    base = v4_baseline(geo4, cfg4, baseline_path)
    tg = v4_hold_targets(base, cfg4)
    prot = v4_protected(segs, cfg4)
    B = v4_boundaries(segs, b["cuts"], prot)
    F = float(segs[-1]["film"][1])
    grid = FilmGrid(segs, F, cfg4, prot)
    phq = PhoneQ(pov, geo4, cfg4)
    n_over = sum(1 for p in prot if p["kind"] == "overhead")
    n_replay = sum(1 for p in prot if p["kind"] == "replay")
    fixed = 2 * n_over + 2 * n_replay
    max_phone_moves = tg["moves_max"] - fixed
    hold_min = max(tg["hold_min_s"], cfg4.hold_dp_min_s, tg["hold_median_min_s"] + 0.1)     # every hold >= the median target: mean / median / min all met
    open_end = next(p["f"][1] for p in prot if p["kind"] == "opening")
    chapters = [float(c["film_s"]) for c in b["doc"]["chapters"][1:]]

    _mc: dict = {}

    def m_of(f):
        k = round(f, 4)
        if k not in _mc:
            _mc[k] = locked_dance(segs, f, f)
        return _mc[k]

    def free(a, bb):
        return not any(p["f"][0] < bb - 1e-3 and p["f"][1] > a + 1e-3 for p in prot)

    def hold_q(cam, a, bb):
        return phq.hold(cam, m_of(a + 1e-3), m_of(bb - 1e-3), margin=cfg4.dur["phone_in"][1])

    _pc: dict = {}

    def p2p_T(ca, cb_, f):
        k = (ca, cb_, round(f, 4))
        if k not in _pc:
            _pc[k] = _p2p_T(ca, cb_, f)
        return _pc[k]

    def _p2p_T(ca, cb_, f):
        d = m_of(f)
        A, Bp = geo4.phone(ca, d), geo4.phone(cb_, d)
        c = geo4.centre(d)
        if not (geo4.recording(ca)[0] <= d - 3 and d + 3 <= geo4.recording(ca)[1] and geo4.recording(cb_)[0] <= d - 3 and d + 3 <= geo4.recording(cb_)[1]):
            return None
        daz = abs(wrap180(cyl(Bp["pos"], c)[0] - cyl(A["pos"], c)[0]))
        dist = float(np.linalg.norm((Bp["pos"] - cfg4.axis_run_m * Bp["fwd"]) - (A["pos"] - cfg4.axis_run_m * A["fwd"])))
        g = V4_PROFILE["phone_to_phone"]["glide"]
        gs = g[1] - g[0]
        T = max(cfg4.dur["phone_to_phone"][2], 1.875 * daz / (cfg4.cap_az_deg_s * gs), 1.875 * dist / (cfg4.cap_speed_m_s * gs),
                1.875 * _ang(A["fwd"], Bp["fwd"]) / (cfg4.cap_rot_deg_s * gs), v4_way_T("phone_to_phone", cfg4))
        return T if T <= cfg4.dur["phone_to_phone"][1] - 0.3 else None       # (0.3 s of room for the live geometry)

    def S_or(a, bb):
        return grid.orbit(a, bb)

    def S_ph(cam, a, bb, q):
        return grid.phone(a, bb) * (0.85 + 0.15 * min(1.0, float(q["score"]) / 0.9))

    _ac: dict = {}

    def az_of(cam, f):
        k = (cam, round(f, 4))
        if k not in _ac:
            d = m_of(f)
            _ac[k] = cyl(geo4.phone(cam, d)["pos"], geo4.centre(d))[0]
        return _ac[k]

    runs, dpi = v4_dp(B, phq.cams, hold_q, p2p_T, free, S_or, S_ph, cfg4, max_phone_moves, hold_min, open_end, chapters, az_of)
    if not dpi["ok"]:
        raise fr.RecapError(f"v4: {dpi['why']}")
    log(f"v4 plan: {len(runs)} runs ({sum(1 for r in runs if r['kind'] == 'phone')} phone holds), {dpi['moves_phone']} phone ways + {fixed} crane / replay ways "
        f"(max {tg['moves_max']}), boundaries {len(B)}")
    return _v4_finish(b, segs, runs, dpi, geo, geo4, base, tg, prot, B, fixed, cfg4, phq, write, out=out)


def _v4_finish(b: dict, segs: list[dict], runs: list[dict], dpi: dict, geo: Geo, geo4: Geo4, base: dict, tg: dict, prot: list[dict], B: list[float],
               fixed: int, cfg4: V4Cfg, phq, write: bool, T0=None, extra: dict | None = None, out: Path | None = None) -> dict:
    """the part of a v4 build after the plan (v4.0 and v4.1): the runs' views, the ways' schedule, the flow blocks, the physics layer, the
    metrics and the checks"""
    sp, pov, cfg, pcfg = b["sp"], b["pov"], b["cfg"], b["pcfg"]
    segs = v4_assign(segs, runs, pov, cfg4)
    plog: list[str] = []
    sched = v4_schedule(segs, runs, geo4, cfg4, cfg, plog, T0=T0)
    shots = sched["shots"]
    flow = write_flow_v4(segs, shots, sched, runs, geo4, cfg, cfg4)
    con, con_src = connected_series(int(round(sp["duration"] * 30)) + 2, b["hand_events"], b["cb"])
    phys_rows = apply_physics(segs, b["act"], con, cfg, pcfg)
    t_notes = t_under_t_arrows(segs)
    shots = shots_of(segs)
    fill_layer_timing(segs, shots, pcfg)
    for s in segs:
        refilm_annos(s)
    doc = b["doc"]
    doc["segments"] = segs
    doc["flow"] = {"version": 2, "frustums": flow["frustums"], "schema": FLOW_SCHEMA_V2, "no_jump_cuts": True,
                   "ghost_frustums": v4_ghost_block(geo4, cfg4),
                   "defaults": {"easing": "smootherstep", "ground_m": list(cfg.ground_m), "clear_radius_m": cfg.clear_radius_m, "frustum_depth_m": cfg.frustum_depth_m,
                                "axis_run_m": cfg4.axis_run_m, "axis_share": 0.5, "flash_s": cfg.flash_s, "inline_lift_max_m": cfg4.orbit_block["inline_lift_max_m"],
                                "hold_s": {"3d": cfg4.hold_s["3d"], "phone": tg["hold_min_s"], "overhead": cfg4.hold_s["overhead"]},
                                "v4": {"caps": {"az_deg_s": cfg4.cap_az_deg_s, "rot_deg_s": cfg4.cap_rot_deg_s, "speed_m_s": cfg4.cap_speed_m_s,
                                                "crane_pitch_deg_s": cfg4.cap_crane_deg_s},
                                       "durations": cfg4.dur, "profiles": cfg4.prof(), "paths": V4_PATH, "readable_s": cfg4.readable_s,
                                       "orbit": {"rate_deg_s": list(cfg4.orbit_rate_deg_s), "block": cfg4.orbit_block},
                                       "crop_smooth_s": cfg4.crop_smooth_s, "ghost_opacity": cfg4.ghost_opacity}}}
    v41 = cfg4.version == "4.1"
    if v41:
        doc["flow"]["revision"] = "4.1"
        doc["flow"]["schema"] = FLOW_SCHEMA_V2 + FLOW_SCHEMA_V41
        doc["flow"]["defaults"]["v4"]["crane_law"] = dict(V41_CRANE)
        doc["flow"]["defaults"]["v4"]["crop_framing_9x16"] = dict(cfg4.crop or {})
        doc["presets"]["vertical"]["legend"] = copy.deepcopy(cfg4.legend)
        doc["created"] = ("generated by dancecap.film_recap_flow build --version v4.1 (long phone holds placed by Jev's need_3d, slow orbits <= 6 deg/s, "
                          "ways >= 4 s, cruise cranes, the physics legend in the 9:16 top band) on top of dancecap.film_recap; edit the rules there")
    else:
        doc["created"] = "generated by dancecap.film_recap_flow build --version v4 (fewer, longer shots; slow orbits; C2 glides) on top of dancecap.film_recap; edit the rules there"
    rf = doc.setdefault("recap_film", {})
    rf["v3"] = {"flow_version": cfg4.version, "back_to_the_axis": len(b["phrases"]), "axis_events_dropped": [e["id"] for e in b["dropped_ev"]],
                "connected_source": con_src, "connected_events": {"path": str(b["hand_events"]), "sha256_16": file_sha(b["hand_events"])}, "shots": len(shots)}
    rf["v4"] = {"runs": [{k: v for k, v in r.items() if k not in ("q",)} | {"quality": (r.get("q") or {}).get("score")} for r in runs],
                "orbits": {str(k): v for k, v in sched["plans"].items()}, "dp": dpi, "targets": tg, "fixed_moves": fixed}
    if v41:
        rf["v4"]["runs"] = [x | {"level": (r.get("q") or {}).get("level"), "usable": (r.get("q") or {}).get("usable"),
                                 "worst_gap_s": (r.get("q") or {}).get("worst_gap_s")} for x, r in zip(rf["v4"]["runs"], runs)]
        rf["v4"]["need3d"] = {"path": str((extra or {}).get("need_path")), "sha256_16": file_sha((extra or {}).get("need_path"))}
    ids = {s["id"] for s in segs}
    missing = [c for c in doc["chapters"] if c["segment"] not in ids]
    if missing:
        raise fr.RecapError(f"chapter anchors lost by the v4 passes: {missing}")
    met = film_metrics(doc, geo4, cfg4)
    if v41:
        v41_extra_metrics(met, doc, geo4, cfg4, (extra or {}).get("need"), pov)
    checks = check_v4(doc, b["tl"], sp, geo, geo4, cfg4, base, met, phq)
    doc["flow"]["v4"] = {"targets": tg, "baseline": {k: v for k, v in base.items() if not k.startswith("_")} | {"phone_holds": {k: v for k, v in base["phone_holds"].items() if k != "list"}},
                         "measured": _metrics_public(met)}
    res = {"doc": doc, "tl": b["tl"], "report": b["report"], "sp": sp, "geo": geo, "geo4": geo4, "cfg": cfg, "cfg4": cfg4, "shots": shots, "runs": runs,
           "sched": sched, "plog": plog, "phys_rows": phys_rows, "t_notes": t_notes, "pivots": b["pivot_rows"], "replay": b["replay_notes"],
           "baseline": base, "metrics": met, "checks": checks, "targets": tg, "dp": dpi, "prot": prot, "boundaries": len(B), "con_src": con_src,
           "physics_quant": quantify_physics(doc, json.loads((b["paths"].film / "direction.json").read_text(encoding="utf-8"))), "clips": clips_needed(doc, geo)}
    res.update(extra or {})
    if v41:
        need = (extra or {}).get("need")
        res["why_high"] = why_high_on_phone(res, need, cfg4)
    if write:
        write_all_v4(res, out)
    return res


def _metrics_public(m: dict) -> dict:
    out = {k: v for k, v in m.items() if not k.startswith("_")}
    out["phone_holds"] = {k: v for k, v in m["phone_holds"].items() if k != "list"}
    out["ways"] = {k: v for k, v in m["ways"].items() if k != "peaks"}
    return out


# ------------------------------------------------------------------------------------------------ v4: checks (the owner's targets fail the check)


def check_v4(doc: dict, tl: dict, sp: dict, geo: Geo | None, geo4: Geo4 | None, cfg4: V4Cfg, base: dict | None, met: dict | None = None,
             phq: PhoneQ | None = None) -> dict:
    """the film checks (film_recap.validate), the flow checks that hold for every version (no jump cut, one easing, ground level except the cranes, the
    crane only at the overheads, the phones entered / left through their frustums and only while they record, the pivots, the physics legend and
    dots, the time map and captions = film/'s) and the v4 TARGETS (each a row {target, measured, ok}; a missed target is a problem):
    moves <= v3.2 / 3, phone holds mean / median / shortest >= 3 x v3.2's, every hold seen well by its phone, every orbit 4-10 deg/s and no static
    3D, every way inside its duration band and under the caps, every phone picture readable, the faint frustums set at 0.15"""
    bad = list(fr.validate(doc, tl, sp))
    notes: list[str] = []
    segs = doc["segments"]
    shots = shots_of(segs)
    flow = doc.get("flow") or {}
    met = met or film_metrics(doc, geo4, cfg4, sample=geo4 is not None)
    tg = (flow.get("v4") or {}).get("targets") or (v41_targets(cfg4) if cfg4.version in V41_STYLE else (v4_hold_targets(base, cfg4) if base else None))
    if tg is not None and cfg4.version in V41_STYLE and cfg4.share_check_min >= 0 and tg.get("share_min") != cfg4.share_check_min:
        # the owner's decision of 2026-10-22 (OWNER_SHARE_DECISION): the check holds the delivered plan to >= 25 % phone video; a direction
        # built before it carries the planner's 45 % in flow.v4.targets
        tg = {**tg, "share_plan_min": tg.get("share_min"), "share_min": cfg4.share_check_min, "share_decision": OWNER_SHARE_DECISION}
    if tg is None:
        bad.append("no v4 targets (the baseline v3.2 direction is missing and the direction does not carry them)")
        tg = {}
    rows = []

    def target(name, want, got, ok):
        rows.append({"target": name, "want": want, "measured": got, "ok": bool(ok)})
        if not ok:
            bad.append(f"TARGET MISSED: {name}: want {want}, measured {got}")

    # --- the flow rules every version keeps
    for sh in shots[1:]:
        tr = segs[sh.segs[0]]["camera"].get("transition_in")
        if not tr or float(tr.get("duration_s", 0)) <= 0:
            bad.append(f"jump cut: no transition into {segs[sh.segs[0]]['id']}")
    eas = {(segs[sh.segs[0]]["camera"].get("transition_in") or {}).get("easing") for sh in shots[1:]}
    if eas != {"smootherstep"}:
        bad.append(f"the easings are not one: {eas}")
    for k, sh in enumerate(shots[1:], 1):
        tr = segs[sh.segs[0]]["camera"]["transition_in"]
        p = shots[k - 1]
        typ = tr["type"]
        if typ != transition_type(p, sh):
            bad.append(f"{segs[sh.segs[0]]['id']}: transition {typ}, expected {transition_type(p, sh)}")
        if (p.kind == "overhead" or sh.kind == "overhead") != (typ in ("crane_up", "crane_down") or bool(tr.get("crane"))):
            bad.append(f"{segs[sh.segs[0]]['id']}: a crane must be exactly the way into / out of an overhead ({typ})")
        if typ in ("crane_up", "crane_down") and not ((tr.get("v4") or {}).get("path") == "crane"):
            bad.append(f"{segs[sh.segs[0]]['id']}: a crane without the crane path")
        if sh.kind == "phone" and typ in ("phone_in", "phone_to_phone") and (tr.get("frustum_in") or {}).get("camera") != sh.phone:
            bad.append(f"{segs[sh.segs[0]]['id']}: entered without phone {sh.phone}'s frustum")
        if p.kind == "phone" and typ in ("phone_out", "phone_to_phone") and (tr.get("frustum_out") or {}).get("camera") != p.phone:
            bad.append(f"{segs[sh.segs[0]]['id']}: phone {p.phone} left without its frustum")
        lo, hi, _ = cfg4.dur.get(typ, [0, 99, 0])
        if not (lo - 2e-3 <= tr["duration_s"] <= hi + 2e-3):
            bad.append(f"{segs[sh.segs[0]]['id']}: a {typ} of {tr['duration_s']} s is outside {lo}-{hi} s")
    for sh in shots:
        cam = segs[sh.segs[0]]["camera"]
        if sh.kind == "phone":
            if cam.get("mode") != "camera_tour" or cam.get("cut") != "flow":
                bad.append(f"{segs[sh.segs[0]]['id']}: a phone shot must be a camera_tour with cut 'flow'")
            w = (cam.get("cameras") or [{}])[0].get("dance_t") or [sh.d0, sh.d1]
            if w[0] > sh.d0 + 1e-3 or w[1] < sh.d1 - 1e-3:
                bad.append(f"{segs[sh.segs[0]]['id']}: the phone's window {w} does not cover its shot {sh.d0:.2f}-{sh.d1:.2f}")
            if geo is not None and sh.phone in geo.phones:
                r0, r1 = geo.recording(sh.phone)
                if w[0] < r0 - 0.05 or w[1] > r1 + 0.05:
                    bad.append(f"{segs[sh.segs[0]]['id']}: phone {sh.phone}'s window {w} is outside its recording {r0:.2f}-{r1:.2f} s (entered only while it records)")
        elif sh.kind == "3d" and not cam.get("ground_level"):
            bad.append(f"{segs[sh.segs[0]]['id']}: a 3D shot without ground_level")
    # ground level of the ways (sampled), except the cranes
    smp = met.get("_sample")
    if smp is not None:
        trs = {sh_k: segs[shots[sh_k].segs[0]]["camera"]["transition_in"]["type"] for sh_k in range(1, len(shots))}
        hs = smp["eye_h"]
        bad_h = []
        for i, k in enumerate(smp["way"]):
            if k >= 1 and trs.get(int(k)) not in ("crane_up", "crane_down") and not (cfg4 and 0.5 - 0.02 <= hs[i] <= 2.4 + 0.02):
                bad_h.append((round(float(smp["t"][i]), 2), round(float(hs[i]), 2)))
        if bad_h:
            bad.append(f"{len(bad_h)} way frames off ground level (0.5-2.4 m) outside the cranes, e.g. {bad_h[:3]}")
        else:
            notes.append("every way frame at ground level (0.5-2.4 m, sampled at 30 Hz) except the cranes")
    # the pivots
    for ph in back_to_axis(sp["utterances"]):
        f = film_of(doc, ph["axis_t"])
        s = next((x for x in segs if x["film"][0] - 1e-6 <= f < x["film"][1]), None)
        if s is None or not s.get("pivot"):
            bad.append(f"'{ph['words']}' ({ph['utt']}): the word 'axis' is not in a pivot segment")
            continue
        if shot_kind(s["camera"]) != "3d":
            bad.append(f"{s['id']}: the pivot is not a 3D view")
        if not any(a.get("text") == PIVOT_LABEL for a in s["annotations"]):
            bad.append(f"{s['id']}: no '{PIVOT_LABEL}' arrow on the pivot")
    over_now = sum(1 for sh in shots if sh.kind == "overhead")
    if over_now != 4:
        bad.append(f"{over_now} overheads (the owner's 4 keep their crane)")
    for s in segs:
        L = s["layers"]
        if L.get("skeleton_mode") == "physics":
            if not L.get("legend"):
                bad.append(f"{s['id']}: load colours without the 'ESTIMATED' legend")
            if s["dance"][1] - s["dance"][0] >= 1.0 and not any(a["kind"] == "com_marker" and any(t.get("type") == "com" for t in a.get("targets", [])) for a in s["annotations"]):
                bad.append(f"{s['id']}: a physics segment without the dancers' centre-of-mass floor dots")
    ce = ((doc.get("recap_film") or {}).get("v3") or {}).get("connected_events") or {}
    if ce.get("path"):
        now = file_sha(Path(ce["path"]))
        notes.append(f"STALE: {ce['path']} changed since this build: re-run build --version v4" if now != ce.get("sha256_16") else
                     f"'connected' rule up to date with {Path(ce['path']).name} (sha {now})")
    cur_dir = WORK / "film" / "direction.json"
    if cur_dir.exists():
        cur = json.loads(cur_dir.read_text(encoding="utf-8"))
        a = [(round(t, 3), round(_dance_at_film(doc, t), 3), _locked_at(doc, t)) for t in np.arange(0.05, doc["timeline"]["film_duration_s"], 0.1)]
        b2 = [(round(t, 3), round(_dance_at_film(cur, t), 3), _locked_at(cur, t)) for t in np.arange(0.05, cur["timeline"]["film_duration_s"], 0.1)]
        (notes.append(f"time map = film/direction.json's at {len(a)} points every 0.1 s: the same soundtrack") if a == b2 else
         bad.append("the film / master time map differs from film/direction.json: the soundtrack would change"))
        cur_tl = WORK / "film" / "narration" / "timeline.json"
        if cur_tl.exists():
            ct = json.loads(cur_tl.read_text(encoding="utf-8"))
            same = json.dumps(ct.get("captions"), sort_keys=True) == json.dumps(tl.get("captions"), sort_keys=True)
            (notes if same else bad).append("captions identical to film/narration/timeline.json" if same else "the captions differ from film/narration/timeline.json")
    if cfg4.version in V41_STYLE:
        _targets_v41(target, doc, met, cfg4, tg, phq, shots, notes)
        if cfg4.version in ("4.2", "4.3"):
            _targets_v42(target, doc, met, cfg4, notes)
        if cfg4.version == "4.3":
            _targets_v43(target, doc, met, cfg4, notes, bad)
        pw = [segs[shots[k].segs[0]]["camera"]["transition_in"] for k in range(1, len(shots))]
        return {"bad": bad, "notes": notes, "targets": rows, "shots": len(shots), "boundaries": len(shots) - 1,
                "transitions": {t: sum(1 for x in pw if x["type"] == t) for t in V4_PROFILE}}
    # --- the owner's targets (v4)
    mv = met["moves"]["total"]
    target("camera moves <= v3.2's / 3", f"<= {tg.get('moves_max')}", mv, mv <= tg.get("moves_max", -1))
    h = met["phone_holds"]
    target("phone hold mean >= 3 x v3.2's", f">= {tg.get('hold_mean_min_s')} s", h["mean_s"], h["mean_s"] >= tg.get("hold_mean_min_s", 1e9) - 1e-6)
    target("phone hold median >= 3 x v3.2's", f">= {tg.get('hold_median_min_s')} s", h["median_s"], h["median_s"] >= tg.get("hold_median_min_s", 1e9) - 1e-6)
    target("shortest phone hold >= 3 x v3.2's", f">= {tg.get('hold_min_s')} s", h["min_s"], h["min_s"] >= tg.get("hold_min_s", 1e9) - 1e-6)
    if phq is not None:
        poor = []
        for hh in h["list"]:
            d0, d1 = locked_dance(segs, hh["film"][0] + 1e-3, hh["film"][0]), locked_dance(segs, hh["film"][1] - 1e-3, hh["film"][1])
            if phq.hold(hh["phone"], d0, d1) is None:
                poor.append(f"{hh['shot']} phone {hh['phone']} {d0:.1f}-{d1:.1f}")
        target("every phone hold shows both teachers (whole hold strong, every 2-s piece usable)", "all", f"{len(h['list']) - len(poor)}/{len(h['list'])}" + (f" ({poor})" if poor else ""), not poor)
    lo, hi = cfg4.orbit_rate_deg_s
    rates3 = []
    for sh in shots:
        cam = segs[sh.segs[0]]["camera"]
        if sh.kind in ("3d", "overhead"):
            rates3.append(abs(float(cam.get("drift_deg_s", 0.0) or 0.0)))
    target("every 3D view orbits at 4-10 deg/s (never static)", f"{lo}-{hi} deg/s", f"{min(rates3):.2f}-{max(rates3):.2f} deg/s" if rates3 else "-",
           bool(rates3) and min(rates3) >= lo - 1e-6 and max(rates3) <= hi + 1e-6)
    pk = met["ways"].get("peak_az_deg_s", {}).get("max")
    pr = met["ways"].get("peak_rot_deg_s", {}).get("max")
    ps = met["ways"].get("peak_speed_m_s", {}).get("max")
    pc = met["ways"].get("crane_pitch_deg_s", {}).get("max")
    if pk is not None:
        target("ways: turn about the couple <= cap", f"<= {cfg4.cap_az_deg_s} deg/s", pk, pk <= cfg4.cap_az_deg_s + 0.5)
        target("ways: view rotation <= cap", f"<= {cfg4.cap_rot_deg_s} deg/s", pr, (pr or 0) <= cfg4.cap_rot_deg_s + 0.5)
        target("ways: eye speed <= cap", f"<= {cfg4.cap_speed_m_s} m/s", ps, (ps or 0) <= cfg4.cap_speed_m_s + 0.05)
        target("cranes: pitch <= cap", f"<= {cfg4.cap_crane_deg_s} deg/s", pc, (pc or 0) <= cfg4.cap_crane_deg_s + 0.5)
    pw = [segs[shots[k].segs[0]]["camera"]["transition_in"] for k in range(1, len(shots))]
    phw = [t["duration_s"] for t in pw if t["type"] in AXIS_TYPES]
    target("every phone way 2.5-5 s", f"{tg.get('way_s')} s", f"{min(phw):.2f}-{max(phw):.2f} s" if phw else "-",
           bool(phw) and min(phw) >= tg.get("way_s", [9, 9])[0] - 2e-3 and max(phw) <= tg.get("way_s", [0, 0])[1] + 2e-3)
    rd = [v for t in pw for v in ((t.get("v4") or {}).get("picture_readable_s") or {}).values()]
    target("every phone picture reads in the world", f">= {cfg4.readable_s} s", f"min {min(rd):.2f} s" if rd else "-", bool(rd) and min(rd) >= cfg4.readable_s - 2e-3)
    gh = flow.get("ghost_frustums") or {}
    target("faint frustums of every phone in every 3D view", f"opacity {cfg4.ghost_opacity}, all phones",
           f"opacity {gh.get('opacity')}, phones {gh.get('phones')}", bool(gh.get("enabled")) and abs(float(gh.get("opacity", -1)) - cfg4.ghost_opacity) < 1e-9
           and len(gh.get("phones") or []) >= 6 and (gh.get("views") or {}).get("3d"))
    # chapters: a chapter opening coincides with a move (information)
    starts = [segs[sh.segs[0]]["film"][0] for sh in shots]
    arrivals = [shots[k].f0 + segs[shots[k].segs[0]]["camera"]["transition_in"]["start_s"] + segs[shots[k].segs[0]]["camera"]["transition_in"]["duration_s"] for k in range(1, len(shots))]
    chs = [c["film_s"] for c in doc["chapters"][1:]]
    near = sum(1 for c in chs if any(abs(c - x) <= 1.5 for x in starts + arrivals))
    notes.append(f"chapter openings with a move within 1.5 s: {near} of {len(chs)} (the chapter cards are unchanged)")
    return {"bad": bad, "notes": notes, "targets": rows, "shots": len(shots), "boundaries": len(shots) - 1,
            "transitions": {t: sum(1 for x in pw if x["type"] == t) for t in V4_PROFILE}}


def _targets_v41(target, doc: dict, met: dict, cfg4: V4Cfg, tg: dict, phq, shots: list[Shot], notes: list[str]) -> None:
    """the owner's v4.1 targets (2026-10-22) as check rows; a missed one fails the check"""
    segs = doc["segments"]
    F = float(doc["timeline"]["film_duration_s"])
    mv, bt = met["moves"]["total"], met["moves"]["by_type"]
    # 1. moves
    target("1. camera moves <= 22 (never more than v4.0)", f"<= {tg['moves_max']}", f"{mv} {json.dumps(bt, sort_keys=True)}", mv <= tg["moves_max"])
    target("1. no phone-to-phone way, no 3D flight", "0 / 0", f"{bt.get('phone_to_phone', 0)} / {bt.get('flight', 0)}", not bt.get("phone_to_phone") and not bt.get("flight"))
    # 2. the phone holds
    h = met["phone_holds"]
    share = h["sum_s"] / F if F else 0.0
    exact = sum(x["hold_s"] for x in h["list"]) / F if F else 0.0
    target("2. phone full-frame share of the film", f">= {100 * tg['share_min']:.0f} %", f"{100 * exact:.1f} % ({h['sum_s']} s in {h['n']} holds)", exact >= tg["share_min"] - 1e-6)
    target("2. every phone hold", f">= {tg['hold_min_s']} s", f"shortest {h['min_s']} s (longest {h['max_s']} s)", h["n"] > 0 and h["min_s"] >= tg["hold_min_s"] - 1e-6)
    target("2. phone hold mean", f">= {tg['hold_mean_min_s']} s", f"{h['mean_s']} s (median {h['median_s']} s)", h["n"] > 0 and h["mean_s"] >= tg["hold_mean_min_s"] - 1e-6)
    if phq is not None:
        poor, lv = [], []
        for hh in h["list"]:
            f1 = min(float(hh["film"][1]), F) - 1e-3                      # (the holds' film times are rounded to 0.01 s; the last one ends with the film)
            d0, d1 = locked_dance(segs, hh["film"][0] + 1e-3, hh["film"][0]), locked_dance(segs, f1, f1)
            q = phq.hold(hh["phone"], d0, d1)
            if q is None:
                poor.append(f"{hh['shot']} phone {hh['phone']} {d0:.1f}-{d1:.1f}")
            else:
                lv.append(f"{hh['phone']}:L{q.get('level', 0)}")
        target("2. every phone hold shows both teachers clearly (L0 = v4's rule; L1 / L2 = the relaxed ladder)", "all", f"{len(h['list']) - len(poor)}/{len(h['list'])} "
               f"({', '.join(lv)})" + (f" poor {poor}" if poor else ""), not poor)
    # 3. meaning (Jev's need_3d)
    nd = met.get("need")
    if nd:
        ok = nd["holds_mean"] is not None and nd["three_d_mean"] is not None and nd["holds_mean"] <= nd["three_d_mean"] - tg["need_clear_margin"] + 1e-9
        target("3. Jev need_3d: phone holds clearly under the 3D views (speech-weighted mean)", f"holds <= 3D - {tg['need_clear_margin']}",
               f"holds {nd['holds_mean']} vs 3D {nd['three_d_mean']} (orbits between the ways {nd['orbit_mean']}); need >= 2.5 on the phone "
               f"{nd['high_on_phone_s']} of {nd['high_total_s']} s", ok)
    else:
        target("3. Jev need_3d: phone holds clearly under the 3D views", "measured", "no need3d.json", False)
    # 4. slow orbits and the ways
    lo, hi = cfg4.orbit_rate_deg_s
    by_run: dict = {}
    for sh in shots:
        cam = segs[sh.segs[0]]["camera"]
        if sh.kind in ("3d", "overhead"):
            by_run.setdefault((cam.get("v4") or {}).get("run"), set()).add(round(float(cam.get("drift_deg_s", 0.0) or 0.0), 3))
    rates = [abs(r) for v in by_run.values() for r in v]
    one = all(len(v) == 1 for v in by_run.values())
    tracked = sorted({(segs[sh.segs[0]]["camera"].get("v4") or {}).get("run") for sh in shots if segs[sh.segs[0]]["camera"].get("azimuth_track")}, key=str)
    target(f"4. every slow orbit: one constant rate, {lo:g}-{hi:g} deg/s" + (f" (the law; run {', '.join(str(x) for x in tracked)} plays its azimuth track: "
           "v4.3's rule, target 10)" if tracked else ""), f"{lo}-{hi} deg/s, one per orbit", f"{', '.join(f'{min(v):g}' for v in by_run.values())} deg/s"
           + ("" if one else " (a run with two rates)"), bool(rates) and one and min(rates) >= lo - 1e-6 and max(rates) <= hi + 1e-6)
    mo = met.get("motion") or {}
    if mo:
        target("4. never static, never reversing (the orbit's half of every way never turns against the orbit; settled orbit frames turn >= 1 deg/s its way)",
               f"<= {tg['against_max_deg_s']} deg/s against; 0 % static", f"{mo['ways_against_max_deg_s']} deg/s (whole ways incl. the phone's own motion "
               f"{mo['ways_against_whole_max_deg_s']}); {100 * mo['orbit_static_or_reverse_share']:.2f} %",
               mo["ways_against_max_deg_s"] <= tg["against_max_deg_s"] + 1e-9 and mo["orbit_static_or_reverse_share"] <= 0.001)
    ws = way_spans(doc)
    wmin = min((b - a for _, _, a, b in ws), default=0.0)
    target("4. every way (phone in / out, crane, replay)", f">= {tg['way_min_s']} s", f"shortest {wmin:.2f} s, longest {max((b - a for _, _, a, b in ws), default=0.0):.2f} s",
           bool(ws) and wmin >= tg["way_min_s"] - 2e-3)
    pk = met["ways"].get("peak_az_deg_s", {}).get("max")
    pr = met["ways"].get("peak_rot_deg_s", {}).get("max")
    ps = met["ways"].get("peak_speed_m_s", {}).get("max")
    pc = met["ways"].get("crane_pitch_deg_s", {}).get("max")
    target("4. ways: peak turn about the couple", f"<= {tg['cap_az_deg_s']} deg/s", pk, pk is not None and pk <= tg["cap_az_deg_s"] + 0.05)
    target("4. ways: peak view rotation", f"<= {tg['cap_rot_deg_s']} deg/s", pr, pr is not None and pr <= tg["cap_rot_deg_s"] + 0.05)
    target("4. ways: peak eye speed", f"<= {tg['cap_speed_m_s']} m/s", ps, ps is not None and ps <= tg["cap_speed_m_s"] + 0.01)
    # 5. the overheads
    target("5. cranes: peak pitch", f"<= {tg['cap_crane_deg_s']} deg/s", pc, pc is not None and pc <= tg["cap_crane_deg_s"] + 0.05)
    target("5. the 4 overheads keep their crane up (and down)", "4 / 4", f"{bt.get('crane_up', 0)} / {bt.get('crane_down', 0)}", bt.get("crane_up") == 4 and bt.get("crane_down") == 4)
    # 6. kept from v4
    pw = [segs[shots[k].segs[0]]["camera"]["transition_in"] for k in range(1, len(shots))]
    rd = [v for t in pw for v in ((t.get("v4") or {}).get("picture_readable_s") or {}).values()]
    target("6. every phone picture reads in the world", f">= {cfg4.readable_s} s", f"min {min(rd):.2f} s" if rd else "-", bool(rd) and min(rd) >= cfg4.readable_s - 2e-3)
    gh = (doc.get("flow") or {}).get("ghost_frustums") or {}
    target("6. faint frustums of every phone in every 3D view", f"opacity {cfg4.ghost_opacity}, all phones", f"opacity {gh.get('opacity')}, phones {gh.get('phones')}",
           bool(gh.get("enabled")) and abs(float(gh.get("opacity", -1)) - cfg4.ghost_opacity) < 1e-9 and len(gh.get("phones") or []) >= 6 and (gh.get("views") or {}).get("3d"))
    target("6. the replay (in / out ways)", "1 / 1", f"{bt.get('replay_in', 0)} / {bt.get('replay_out', 0)}", bt.get("replay_in") == 1 and bt.get("replay_out") == 1)
    # 7. the physics legend (v4.2: the meter windows, _targets_v42)
    lg = (met.get("legend") or {}).get("v41_place") or {}
    if cfg4.version in ("4.2", "4.3"):
        lg = {}
    pl = ((doc.get("presets") or {}).get("vertical") or {}).get("legend") or {}
    if cfg4.version not in ("4.2", "4.3"):
        target("7. physics legend in the 9:16 top band (presets.vertical.legend), the teachers' box clear of it in the phone holds",
           f"top {cfg4.legend['top_frac'] if cfg4.legend else '-'}; >= {100 * tg['legend_clear_min']:.0f} % of hold frames clear",
           f"top {pl.get('top_frac')}; {100 * (1 - lg.get('under_legend_share', 1.0)):.1f} % clear ({lg.get('frames')} frames; box top median {lg.get('box_top_median')})",
           bool(pl) and abs(float(pl.get("top_frac", -1)) - float(cfg4.legend["top_frac"])) < 1e-9 and lg.get("frames", 0) > 0
           and 1 - lg["under_legend_share"] >= tg["legend_clear_min"] - 1e-9)
    starts = [segs[sh.segs[0]]["film"][0] for sh in shots]
    arrivals = [shots[k].f0 + pw[k - 1]["start_s"] + pw[k - 1]["duration_s"] for k in range(1, len(shots))]
    chs = [c["film_s"] for c in doc["chapters"][1:]]
    near = sum(1 for c in chs if any(abs(c - x) <= 1.5 for x in starts + arrivals))
    notes.append(f"chapter openings with a move within 1.5 s: {near} of {len(chs)} (the chapter cards are unchanged)")
    over = [(segs[shots[k].segs[0]]["id"], t["type"]) for k, t in enumerate(pw, 1) if t["type"] in AXIS_TYPES
            for s in segs if s.get("pivot") and s["film"][0] < shots[k].f0 + t["start_s"] + t["duration_s"] and s["film"][1] > shots[k].f0 + t["start_s"]]
    if over:
        notes.append(f"phone ways over a pivot: {over}")


# ------------------------------------------------------------------------------------------------ v4: the faint frustums against the dancers


def ghost_overlap(doc: dict, met: dict, geo4: Geo4, cfg4: V4Cfg, step: int = 3) -> dict:
    """How often a phone's faint frustum (apex + its rim at ghost depth) would stand between the 3D camera and the dancers, or near the lens,
    if drawn without the keep-out (an estimate on the plan's poses, 10 Hz, 9:16, the plan's FOV): per phone the seconds it overlaps the couple's
    screen box in front of them and the seconds its apex is within 1 m of the eye. The keep-out in flow.ghost_frustums exists for these."""
    sm = met.get("_sample")
    if sm is None:
        return {}
    pm = PoseModel(doc, geo4, cfg4=cfg4)
    t = sm["t"]
    idx = [i for i in range(0, len(t), step) if not sm["ph_hold"][i]]
    dt = step / float(cfg4.sample_hz)
    per = {cam: {"over_couple_s": 0.0, "near_lens_s": 0.0} for cam in sorted(geo4.ph)}
    seen3d = 0.0
    aspect = 9.0 / 16.0
    for i in idx:
        f = float(t[i])
        p = pm.pose(f)
        d = pm.dance(f)
        eye, fwd = p["eye"], _unit(p["dir"])
        right = _unit(np.cross(fwd, np.array([0.0, 1.0, 0.0])))
        up = np.cross(right, fwd)
        th = math.tan(math.radians(max(10.0, p["fov"])) / 2)

        def proj(x):
            v = x - eye
            z = float(np.dot(v, fwd))
            if z <= 0.05:
                return None
            return float(np.dot(v, right)) / (z * th * aspect), float(np.dot(v, up)) / (z * th), z
        lead = np.array([np.interp(d, geo4.g.t, geo4.g.lead[:, k]) for k in range(3)])
        foll = np.array([np.interp(d, geo4.g.t, geo4.g.follow[:, k]) for k in range(3)])
        pts = []
        for c in (lead, foll):
            for dx in (-0.35, 0.35):
                for y in (0.0, 1.9):
                    pts.append(proj(np.array([c[0], y, c[2]]) + dx * right))
        pts = [x for x in pts if x is not None]
        if not pts:
            continue
        seen3d += dt
        bx = (min(x[0] for x in pts), max(x[0] for x in pts), min(x[1] for x in pts), max(x[1] for x in pts))
        zc = float(np.dot(geo4.centre(d) - eye, fwd))
        for cam in per:
            r0, r1 = geo4.recording(cam)
            if not (r0 <= d <= r1):
                continue
            ph = geo4.phone(cam, d)
            if np.linalg.norm(ph["pos"] - eye) < 1.0:
                per[cam]["near_lens_s"] += dt
            pr = _unit(np.cross(ph["fwd"], np.array([0.0, 1.0, 0.0])))
            pu = np.cross(pr, ph["fwd"])
            hh = cfg4.ghost_depth_m * math.tan(math.radians(ph["vfov"]) / 2)
            hw = hh * ph["aspect"]
            ctr = ph["pos"] + cfg4.ghost_depth_m * ph["fwd"]
            fp = [proj(ph["pos"])] + [proj(ctr + sx * hw * pr + sy * hh * pu) for sx in (-1, 1) for sy in (-1, 1)]
            fp = [x for x in fp if x is not None]
            if not fp or min(x[2] for x in fp) > zc:
                continue
            fb = (min(x[0] for x in fp), max(x[0] for x in fp), min(x[1] for x in fp), max(x[1] for x in fp))
            if fb[0] < bx[1] and fb[1] > bx[0] and fb[2] < bx[3] and fb[3] > bx[2] and fb[1] > -1 and fb[0] < 1 and fb[3] > -1 and fb[2] < 1:
                per[cam]["over_couple_s"] += dt
    out = {"3d_seconds_checked": round(seen3d, 1), "per_phone": {c: {k: round(v, 1) for k, v in x.items()} for c, x in per.items()}}
    out["over_couple_s_any"] = round(sum(x["over_couple_s"] for x in per.values()), 1)
    return out


# ------------------------------------------------------------------------------------------------ v4: sheets, review, contract


V4_COL = {"3d": "#2a78d6", "phone": "#eb6834", "overhead": "#6250d6", "way": "#52514e"}


def _kind_series(met: dict) -> np.ndarray:
    sm = met["_sample"]
    k = np.where(sm["way"] >= 0, "way", np.where(sm["ph_hold"], "phone", sm["kind"]))
    return np.where(k == "crane", "way", k)


def sheet_v4(met: dict, doc: dict, geo4: Geo4, out: Path, title: str, rows: int = 4, band: tuple = (4.0, 10.0), cap: float = 30.0, label: str = "v4") -> Path:
    """the camera-flow sheet (as v3's flow_sheet.png) from the sampled poses: per row ~92 film s - the eye's azimuth about the couple (the
    phones' azimuths dotted), what is on screen (blue orbit, orange phone hold, violet overhead, grey a way), and the turn rate about the couple
    on a log scale with the orbit band (4-10 deg/s) and the ways' cap (30 deg/s)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sm = met["_sample"]
    t, az, rate = sm["t"], sm["az"], np.abs(sm["az_rate"])
    kinds = _kind_series(met)
    F = float(doc["timeline"]["film_duration_s"])
    span = math.ceil(F / rows)
    fig, axes = plt.subplots(rows * 2, 1, figsize=(22, 4.6 * rows), gridspec_kw={"height_ratios": [3, 1.3] * rows})
    for r in range(rows):
        ax, axr = axes[2 * r], axes[2 * r + 1]
        t0, t1 = r * span, min(F, (r + 1) * span)
        m = (t >= t0) & (t <= t1)
        base = np.median(az[m]) if m.any() else 0.0
        off = 360 * round(base / 360)
        a = az[m] - off
        ylo, yhi = min(-200.0, float(a.min()) - 30), max(200.0, float(a.max()) + 30)
        # what is on screen: a band along the bottom
        tt, kk = t[m], kinds[m]
        i = 0
        while i < len(tt):
            j = i
            while j + 1 < len(tt) and kk[j + 1] == kk[i]:
                j += 1
            ax.axvspan(tt[i], tt[j] + 1 / 30, ymin=0.0, ymax=0.05, color=V4_COL.get(kk[i], "#999"), lw=0)
            if kk[i] == "way":
                ax.axvspan(tt[i], tt[j] + 1 / 30, color="#52514e", alpha=0.10, lw=0)
            i = j + 1
        for cam in sorted(geo4.ph):
            r0, r1 = geo4.recording(cam)
            ts = np.linspace(t0, t1, 40)
            pa = []
            for f in ts:
                d = _dance_at_film(doc, min(f, F - 1e-3))
                pa.append(cyl(geo4.phone(cam, d)["pos"], geo4.centre(d))[0] if r0 <= d <= r1 else np.nan)
            pa = np.array(pa)
            for o in (-720, -360, 0, 360, 720):
                y = pa + o - off
                y = np.where((y > ylo) & (y < yhi), y, np.nan)
                if np.isfinite(y).any():
                    ax.plot(ts, y, ls=":", lw=0.9, color="#eb6834", alpha=0.7)
                    k0 = int(np.nanargmin(np.abs(ts - t0)) if np.isfinite(y[0]) else np.nanargmax(np.isfinite(y)))
                    ax.text(ts[k0] + 0.3, y[k0] + 4, f"phone {cam}", color="#52514e", fontsize=8)
        ax.plot(t[m], a, color="#0b0b0b", lw=1.6)
        ax.set_xlim(t0, t1)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel("eye azimuth about the couple (deg)")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        axr.axhspan(band[0], band[1], color="#2a78d6", alpha=0.12, lw=0)
        axr.axhline(cap, color="#52514e", ls="--", lw=1)
        axr.plot(t[m], np.clip(rate[m], 0.3, 1000), color="#0b0b0b", lw=1.0)
        axr.set_yscale("log")
        axr.set_ylim(0.3, 1000)
        axr.set_xlim(t0, t1)
        axr.set_ylabel("turn about the couple\n(deg/s, log)")
        axr.set_xlabel(f"film s   (bottom band: blue orbit / 3D, orange phone hold, violet overhead, dark grey a way; shaded band {band[0]:g}-{band[1]:g} deg/s = "
                       f"{label} orbit, dashed {cap:g} deg/s = {label} cap)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=72)
    plt.close(fig)
    return out


def compare_plot(base: dict, met: dict, out: Path, tg: dict) -> Path:
    """v3.2 against v4 on one page: what is on screen over the film (two timelines), the turn rate about the couple over the film (log scale,
    one axis, two series) and every phone hold's length against the targets"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.8, 0.8, 2.4, 2.0], hspace=0.55)
    for row, (lab, m) in enumerate((("v3.2", base), ("v4", met))):
        ax = fig.add_subplot(gs[row])
        sm = m["_sample"]
        t = sm["t"]
        kk = _kind_series(m)
        i = 0
        while i < len(t):
            j = i
            while j + 1 < len(t) and kk[j + 1] == kk[i]:
                j += 1
            ax.axvspan(t[i], t[j] + 1 / 30, color=V4_COL.get(kk[i], "#999"), lw=0)
            i = j + 1
        ax.set_xlim(0, float(t[-1]))
        ax.set_yticks([])
        h = m["phone_holds"]
        ax.set_title(f"{lab}: {m['moves']['total']} camera moves (dark grey), {h['n']} phone holds (orange) of {h['sum_s']:.0f} s, "
                     f"3D (blue) + overheads (violet) {m['three_d']['settled_s']:.0f} s, ways {m['ways']['sum_s']:.0f} s", loc="left", fontsize=11)
    ax = fig.add_subplot(gs[2])
    for lab, m, col in (("v3.2", base, "#e34948"), ("v4", met, "#2a78d6")):
        sm = m["_sample"]
        ax.plot(sm["t"], np.clip(np.abs(sm["az_rate"]), 0.3, 1000), color=col, lw=0.9, label=lab)
    ax.axhspan(4, 10, color="#2a78d6", alpha=0.10, lw=0)
    ax.axhline(30, color="#52514e", ls="--", lw=1)
    ax.text(2, 32, "v4 cap 30 deg/s", fontsize=9, color="#52514e")
    ax.text(2, 4.3, "v4 orbit band 4-10 deg/s", fontsize=9, color="#52514e")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 1000)
    ax.set_xlim(0, float(met["_sample"]["t"][-1]))
    ax.set_ylabel("eye's turn about the couple (deg/s, log)")
    ax.set_xlabel("film s")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("how fast the camera turns about the couple, every frame (the plan's poses at 30 Hz)", loc="left", fontsize=11)
    ax = fig.add_subplot(gs[3])
    for y, (lab, m, col) in enumerate((("v3.2", base, "#e34948"), ("v4", met, "#2a78d6"))):
        hs = [x["hold_s"] for x in m["phone_holds"]["list"]]
        ax.scatter(hs, np.full(len(hs), y) + np.linspace(-0.12, 0.12, len(hs)), s=60, color=col, edgecolor="white", lw=1.5, zorder=3)
        ax.text(max(hs) + 0.6, y, f"{lab}: n {len(hs)}, mean {np.mean(hs):.1f} s, median {np.median(hs):.1f} s, min {min(hs):.1f} s", va="center", fontsize=10)
    for n_, (x, lab) in enumerate(((tg["hold_min_s"], "3 x v3.2 shortest"), (tg["hold_median_min_s"], "3 x v3.2 median"), (tg["hold_mean_min_s"], "3 x v3.2 mean"))):
        ax.axvline(x, color="#52514e", ls=":", lw=1)
        ax.text(x + 0.2, 1.55 - 0.13 * n_, lab, fontsize=9, color="#52514e", rotation=0)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["v3.2", "v4"])
    ax.set_ylim(-0.5, 1.7)
    ax.set_xlim(0, max(x["hold_s"] for x in met["phone_holds"]["list"]) + 22)
    ax.set_xlabel("phone hold: the phone's own full-frame video between its way in and its way out (s)")
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.set_title("every phone hold", loc="left", fontsize=11)
    fig.suptitle("class-recap camera: v3.2 (the 2026-10-21 finals) vs v4 (owner 2026-10-22)", fontsize=14, x=0.01, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=72, bbox_inches="tight")
    plt.close(fig)
    return out


def before_after_rows(base: dict, met: dict, tg: dict) -> list[tuple[str, str, str, str]]:
    """the table rows: (what, v3.2, v4, the owner's target)"""
    b, a = base, met
    bh, ah = b["phone_holds"], a["phone_holds"]
    bw, aw = b["ways"], a["ways"]

    def by(m, *ts):
        return sum(m["moves"]["by_type"].get(t, 0) for t in ts)
    r = [("camera moves (every way; no 3D re-framing happens inside a shot)", str(b["moves"]["total"]), str(a["moves"]["total"]), f"<= {tg['moves_max']} (v3.2 / 3)"),
         ("... phone in / out / phone to phone", f"{by(b, 'phone_in')} / {by(b, 'phone_out')} / {by(b, 'phone_to_phone')}", f"{by(a, 'phone_in')} / {by(a, 'phone_out')} / {by(a, 'phone_to_phone')}", ""),
         ("... crane up / down (the 4 overheads; v3.2: 3 of them phone ways passing an overhead)", f"{by(b, 'crane_up')} / {by(b, 'crane_down')} (+3)", f"{by(a, 'crane_up')} / {by(a, 'crane_down')}", "keep the 4 overheads' crane"),
         ("... 3D flights / replay ways", f"{by(b, 'flight')} / {by(b, 'replay_in', 'replay_out')}", f"{by(a, 'flight')} / {by(a, 'replay_in', 'replay_out')}", ""),
         ("shots", str(b["shots"]), str(a["shots"]), "fewer, longer"),
         ("phone holds (full-frame video): n", str(bh["n"]), str(ah["n"]), ""),
         ("phone hold mean", f"{bh['mean_s']} s", f"{ah['mean_s']} s", f">= {tg['hold_mean_min_s']} s (3 x)"),
         ("phone hold median", f"{bh['median_s']} s", f"{ah['median_s']} s", f">= {tg['hold_median_min_s']} s (3 x)"),
         ("phone hold shortest / longest", f"{bh['min_s']} / {bh['max_s']} s", f"{ah['min_s']} / {ah['max_s']} s", f"shortest >= {tg['hold_min_s']} s"),
         ("the crop's own motion inside a phone hold: pan (view widths / s) / zoom (relative / s), time-weighted",
          f"{bh.get('crop_pan_view_per_s')} / {bh.get('crop_zoom_rel_per_s')}", f"{ah.get('crop_pan_view_per_s')} / {ah.get('crop_zoom_rel_per_s')}", "lock into the video"),
         ("phone full-frame video, total (share of the film)", f"{bh['sum_s']} s ({100 * bh['share']:.1f} %)", f"{ah['sum_s']} s ({100 * ah['share']:.1f} %)", "may grow above 50 % (not required)"),
         ("3D views between the ways (orbit + overheads)", f"{b['three_d']['settled_s']} s ({100 * b['three_d']['settled_share']:.1f} %)",
          f"{a['three_d']['settled_s']} s ({100 * a['three_d']['settled_share']:.1f} %)", "more slow-orbit time"),
         ("3D incl. the ways (film - phone full frame)", f"{b['three_d']['with_ways_s']} s", f"{a['three_d']['with_ways_s']} s", ""),
         ("3D drift rates (planned, deg/s)", ", ".join(f"{x:g}" for x in b["three_d"]["drift_rates_deg_s"]), ", ".join(f"{x:g}" for x in a["three_d"]["drift_rates_deg_s"]), "4-10, never static"),
         ("3D time with a static camera (drift < 0.5 deg/s)", f"{b['three_d']['static_s']} s", f"{a['three_d']['static_s']} s", "0"),
         ("turn about the couple in the 3D views (sampled): median / p90 / max",
          " / ".join(str(b["three_d"]["sampled_az_deg_s"][k]) for k in ("median", "p90", "max")) + " deg/s",
          " / ".join(str(a["three_d"]["sampled_az_deg_s"][k]) for k in ("median", "p90", "max")) + " deg/s", "4-10"),
         ("turn about the couple over ALL 3D time incl. ways: median / p90 / p99 / max",
          " / ".join(str(b["three_d"]["sampled_az_incl_ways_deg_s"][k]) for k in ("median", "p90", "p99", "max")) + " deg/s",
          " / ".join(str(a["three_d"]["sampled_az_incl_ways_deg_s"][k]) for k in ("median", "p90", "p99", "max")) + " deg/s", ""),
         ("ways: total time in motion (share)", f"{bw['sum_s']} s ({100 * bw['share']:.1f} %)", f"{aw['sum_s']} s ({100 * aw['share']:.1f} %)", ""),
         ("ways: duration min / median / max", f"{bw['min_s']} / {bw['median_s']} / {bw['max_s']} s", f"{aw['min_s']} / {aw['median_s']} / {aw['max_s']} s", "phone ways 2.5-5 s"),
         ("phone ways: median duration", f"{bw['phone_ways_median_s']} s", f"{aw['phone_ways_median_s']} s", ""),
         ("ground-level ways, peak turn about the couple (sampled): max / median", f"{bw['peak_az_deg_s']['max']} / {bw['peak_az_deg_s']['median']} deg/s",
          f"{aw['peak_az_deg_s']['max']} / {aw['peak_az_deg_s']['median']} deg/s", f"<= {tg['cap_az_deg_s']} deg/s"),
         ("... planned by the direction itself (v3 formula) / Unity-measured", f"{bw['planned_peak_max_deg_s']} / 168 deg/s", "- (sampled above)", ""),
         ("ground-level ways, peak view rotation: max / median", f"{bw['peak_rot_deg_s']['max']} / {bw['peak_rot_deg_s']['median']} deg/s",
          f"{aw['peak_rot_deg_s']['max']} / {aw['peak_rot_deg_s']['median']} deg/s", f"<= {tg['cap_rot_deg_s']} deg/s"),
         ("ground-level ways, peak eye speed: max / median", f"{bw['peak_speed_m_s']['max']} / {bw['peak_speed_m_s']['median']} m/s",
          f"{aw['peak_speed_m_s']['max']} / {aw['peak_speed_m_s']['median']} m/s", f"<= {tg['cap_speed_m_s']} m/s"),
         ("cranes, peak pitch: max / median", f"{bw['crane_pitch_deg_s']['max']} / {bw['crane_pitch_deg_s']['median']} deg/s",
          f"{aw['crane_pitch_deg_s']['max']} / {aw['crane_pitch_deg_s']['median']} deg/s", f"<= {tg['cap_crane_deg_s']} deg/s")]
    return r


def review_v4_md(res: dict) -> str:
    doc, base, met, tg, ch = res["doc"], res["baseline"], res["metrics"], res["targets"], res["checks"]
    segs = doc["segments"]
    shots = shots_of(segs)
    rows = before_after_rows(base, met, tg)
    L = [f"# v4 direction review: {doc['title']} ({doc['name']})", "",
         "Owner 2026-10-22: \"reduce the camera motions actions by a factor of 3 ... hold on a video segment from a phone, increase that by a factor of 3 ... "
         "much smoother translations and zoom in and outs and fade in and outs ... more time in the 3D orbiting Gaussian splat view, slow orbit ... very, very "
         "faint camera frustums from the other phones at all times ... maybe 15% opacity\".", "",
         f"Film {doc['timeline']['film_duration_s']} s (time map, captions, chapters, soundtrack = film/'s and v3.2's). Checks: "
         + ("PASS" if not ch["bad"] else f"{len(ch['bad'])} problems") + f"; {ch['boundaries']} shot changes, every one a way (no jump cut).", "",
         "## Before / after (v3.2 = v3/direction/direction.json, the 2026-10-21 finals; both measured by `film_metrics` the same way)", "",
         "| | v3.2 | v4 | target |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    L += ["", "The sampled numbers come from the plan's poses (`PoseModel`: the couple's smoothed centre, the phones' smoothed tracks, every way a blend of "
          "LIVE end states by its profile; v3.2's ways with v3's arc share and axis dollies), 30 Hz. They are estimates of what the viewer does; v3.2's own "
          "planned peak (the v3 check's formula) was 199.9 deg/s and Unity measured 168 deg/s - the sampled 3.2 figures above are of that order. v4's live "
          "geometry is checked again in Unity (`hm_film_flow --action check`, UNITY_CONTRACT_v4.md section 6).", "",
          "## The owner's targets (the `check` command fails when one is missed)", "", "| target | want | measured | |", "|---|---|---|---|"]
    L += [f"| {r['target']} | {r['want']} | {r['measured']} | {'OK' if r['ok'] else 'MISSED'} |" for r in ch["targets"]]
    L += ["", "## The plan: long phone holds, one slow orbit per stretch between them", "",
          f"Dynamic programming over the speech's pauses ({res['boundaries']} candidates): at most {tg['moves_max']} moves in all, of which "
          f"{res['doc']['recap_film']['v4']['fixed_moves']} are fixed (the 4 overheads' crane up + down, the replay's two recentring ways); every phone hold at least "
          f"{max(tg['hold_median_min_s'] + 0.1, V4Cfg().hold_dp_min_s):.1f} s (so mean / median / shortest all clear 3 x v3.2's), every 2-s piece of it showing both teachers "
          "(recap_pov.window_quality, the whole hold 'strong'), the phone recording through its ways; the protected 3D windows (the opening under the title card, "
          "the 6 pivots, the 4 overheads with their cranes, the replay) inside orbits; an orbit between two phones at least 5 s besides its ways and able to "
          "carry the camera from one phone to the other at 4-10 deg/s (its ways' extra turn <= 50 deg). The value of a film second: as a phone where the base "
          "director chose a phone (1.0, else 0.45-0.8), as the orbit where it chose a physics view / zoom / pivot (1.0, else 0.6-0.85); the same phone "
          "twice in a row pays a little (variety).", "",
          "| run | film s | length | view | phone quality | orbit rate |", "|---|---|---|---|---|---|"]
    orbits = res["sched"]["plans"]
    for ri, r in enumerate(res["runs"]):
        pl = orbits.get(ri)
        L.append(f"| {ri} | {r['f0']:.2f}-{r['f1']:.2f} | {r['f1'] - r['f0']:.1f} s | {'phone ' + r['cam'] if r['kind'] == 'phone' else 'slow orbit'} | "
                 f"{(r.get('q') or {}).get('score', '')} | {'' if pl is None else str(pl['w']) + ' deg/s (ways turn ' + str(pl['r_out']) + ' / ' + str(pl['r_in']) + ' deg)'} |")
    L += ["", "## Every shot and the way into it", "", "| # | film s | shot | view | phone | way in | starts | duration | planned peaks: turn / rotation / speed / pitch |",
          "|---|---|---|---|---|---|---|---|---|"]
    ways = {w["k"]: w for w in res["sched"]["ways"]}
    for k, sh in enumerate(shots):
        s0 = segs[sh.segs[0]]
        tr = s0["camera"].get("transition_in")
        role = (s0["camera"].get("v4") or {}).get("role", sh.mode)
        if not tr:
            L.append(f"| {k} | {sh.f0:.1f}-{sh.f1:.1f} | {s0['id']} | {role} | {sh.phone or ''} | (first shot) | | | |")
            continue
        w = ways.get(k, {})
        L.append(f"| {k} | {sh.f0:.1f}-{sh.f1:.1f} | {s0['id']} | {role}{' (replay)' if sh.replay else ''} | {sh.phone or ''} | {tr['type']} | {tr['start_s']:+.2f} s | "
                 f"{tr['duration_s']:.2f} s | {w.get('peak_az_deg_s', '-')} / {w.get('peak_rot_deg_s', '-')} / {w.get('peak_speed_m_s', '-')} / {w.get('peak_pitch_deg_s', '-')} |")
    notes = [f"- {segs[shots[w['k']].segs[0]]['id']} ({w['type']}): {'; '.join(w['notes'])}" for w in res["sched"]["ways"] if w.get("notes")]
    L += ["", "Placement notes:", ""] + (notes or ["- none"])
    go = res.get("ghost") or {}
    if go:
        L += ["", "## The faint frustums against the dancers (estimate, without the keep-out)", "",
              f"Over {go['3d_seconds_checked']} s of 3D (10 Hz), the seconds in which a phone's ghost frustum would overlap the couple's screen box IN FRONT of "
              f"them, and in which its apex comes within 1 m of the eye: " + ", ".join(f"phone {c} {v['over_couple_s']} / {v['near_lens_s']} s" for c, v in go["per_phone"].items())
              + f". In all {go['over_couple_s_any']} phone-seconds: the keep-out (flow.ghost_frustums.keep_out) is needed, not decorative."]
    q = res["physics_quant"]
    L += ["", "## Physics and pivots (unchanged rules, new shots)", "",
          f"Load colours on screen {q['after']['load_colours_s']} s (v3.2 315.8 s), the dancers' COM floor dots {q['after']['floor_dots_s']} s (v3.2 288.8 s), counterbalance "
          f"layer {q['after']['counterbalance_s']} s, pivot segments {q['after']['pivot_s']} s. The 6 pivots ride inside the slow orbit (no separate framing, no move): "
          "their layers (the counterbalance dial, the 'axis (pivot point)' arrow, the balance axis, the COM dots) are the segments' own; the orbit's eye is at "
          "2.15 m, about 16-18 deg above the couple (v3.2's pivot framing 22 deg, physics view 18 deg).", ""]
    L += ["## Checks", ""] + ([f"- PROBLEM: {x}" for x in ch["bad"]] or ["- film_recap.validate + the flow checks + the v4 targets: no problems"]) + [f"- {x}" for x in ch["notes"]]
    L += ["", "## The phones' clips the v4 direction needs (the Unity stage decodes them: `clips --version v4`)", "",
          "| phone | needed (s) | missing in the capture now (s) |", "|---|---|---|"]
    for cam, c in sorted(res["clips"].items()):
        L.append(f"| {cam} | {c['need_s']} | {c['missing_s']} |")
    return "\n".join(L) + "\n"


def contract_md(res: dict) -> str:
    """UNITY_CONTRACT_v4.md: everything the Unity director must add for flow v2, with the exact keys and curves"""
    doc, cfg4, tg = res["doc"], res["cfg4"], res["targets"]
    gh = doc["flow"]["ghost_frustums"]
    go = res.get("ghost") or {}
    shots = shots_of(doc["segments"])
    segs = doc["segments"]
    P = cfg4.prof()
    v41 = cfg4.version == "4.1"
    band = f"{cfg4.orbit_rate_deg_s[0]:g}-{cfg4.orbit_rate_deg_s[1]:g} deg/s"
    L = ["# Unity contract: the v4 class-recap direction (`hm film flow v2`)", "",
         "Source: `dancecap/work/larissa_kadu_recap/v4/direction/direction.json` (+ `narration/timeline.json`, `chapters.json`, `audio/` - the soundtrack is "
         "film/'s: same time map, captions identical). Built by `python -m dancecap.film_recap_flow build --version v4`, checked by "
         "`python -m dancecap.film_recap_flow check --version v4` (fails when one of the owner's targets is missed). The before is "
         "`v3/direction/direction.json` (v3.2), still what `FilmRecorder.FindDirection` finds first.", "",
         "Write no names into head-movement: say lead / follow / \"the recap capture\".", "",
         "## 0. Gating and what is new", "",
         "Everything below applies only when `flow.version >= 2`. Every v1 key is still written (transition_in type / start_s / duration_s / easing / path / from / "
         "to / arc / constraints / frustum_in / frustum_out / video / video_out / layers_switch_s / transit_layers, flow.frustums, world_azimuth_deg / drift_deg_s), "
         "so the v3 director plays the file (with its own paths and linear fades); the `v4` blocks are additive.", "",
         "| # | what | where | v1 behaviour it replaces |", "|---|---|---|---|",
         "| 1 | ways are C2 blends of LIVE end states, per a profile of smootherstep ramps | `transition_in.v4.{path, profile, ease}` | v3 composite path cached from the poses at its start / middle / end |",
         "| 2 | the FOV zooms by a log-tan law over its own window | `transition_in.v4.fov` | lerp |",
         "| 3 | the pictures and outlines fade by smootherstep curves | `transition_in.v4.profile`, `flow.frustums[].v4` | linear alpha over [a, b] / [c, d] |",
         "| 4 | every phone's frustum, faint (0.15), outline only, in every 3D view | `flow.ghost_frustums` | (none) |",
         "| 5 | an overhead turns with the orbit | overhead `up_policy: \"orbit\"` + world_azimuth_deg / drift_deg_s | `from_entry` (static up) |",
         "| 6 | the replay's ways keep orbiting; only the centre glides | `transition_in.v4.path = orbit_recentre`, `from_pose: orbit_frozen_centre` | a 1.2-s arc from the frozen pose |",
         "| 7 | the slow orbit is the 3D view (one block per stretch) | camera `mode: orbit` + `v4.role: slow_orbit` | many 3D modes / re-framings |",
         "| 8 | the caps the live camera must keep | `flow.defaults.v4.caps` | v3: arcs <= 200 deg/s |",
         "| 9 | recording / preview finds v4 | FilmRecorder.FindDirection, the `_v4` tag | v3/direction first |", "",
         "## 1. The shots' own poses", "",
         f"- **Slow orbit** (`camera.v4.role == \"slow_orbit\"`, mode `orbit`): {json.dumps({k: v for k, v in cfg4.orbit_block.items()})} + `world_azimuth_deg`, "
         f"`drift_deg_s` (the rate, |w| in {list(cfg4.orbit_rate_deg_s)} deg/s; v3's WorldAzimuth already does azimuth = world_azimuth_deg + drift_deg_s x (film - F0), "
         "not clamped before F0) and `ground_level`. Nothing new in FilmCamera.Orbit; the orbits of one stretch (split by an overhead or the replay) share ONE "
         "azimuth function a(t) (each part's world_azimuth_deg is a(F0) of that part), so the turn never stops or jumps across them. FitSpan may widen the lens "
         "(never narrows) - keep it a lens change, damped (no eye jump).",
         "- **Overhead** (`camera.v4.role == \"overhead\"`): its own framing keys as before + `up_policy: \"orbit\"`: Up = -(cos a(t), 0, sin a(t)) with "
         "a(t) = world_azimuth_deg + drift_deg_s x (film - F0) (the orbit's function continued), the eye at 89.9 deg on that azimuth. The picture turns at the "
         f"orbit's rate ({band}): never static. `up_azimuth_deg` (= a(F0)) is the fallback.",
         "- **Phone hold** (`camera.v4.role == \"phone_hold\"`): camera_tour as v3; the crop is smoothed over `camera.v4.crop_smooth_s` (1.8 s; v3 0.6 s) - "
         "the virtual camera inside the video moves less; `cameras[0].dance_t` covers the way in .. the way out (the live pose and the video are needed there).",
         "- **Replay**: the replay segment carries the orbit block of its stretch (`v4.replay: true`); the dance time jumps at its first frame and after its last.", "",
         "## 2. The ways (`transition_in` of the first segment of every shot but the first)", "",
         "Notation: F0 = the shot's first frame (film s); the way runs over [t0, t0 + T], t0 = F0 + `start_s`, T = `duration_s`; u = (t - t0) / T; "
         "ramp(u; a, b) = smootherstep((u - a) / (b - a)) clipped to [0, 1], smootherstep(x) = x^3 (10 - 15 x + 6 x^2). S_A(t) = the PREVIOUS shot's own "
         "pose at film t (it keeps running: the orbit keeps turning, the phone keeps its live pose and crop at that dance time), S_B(t) = THIS shot's own pose "
         "at film t. Never freeze an end (v3 cached its path from the poses at start / middle / end: the camera then left a turning orbit at zero speed).", "",
         "**The weights per type** (`transition_in.v4.profile`, windows of u):", "",
         "| type | v4.path | glide | dolly (axis) | rot | zoom | picture | outline |", "|---|---|---|---|---|---|---|---|"]
    for typ in ("phone_in", "phone_out", "phone_to_phone"):
        p = P[typ]
        L.append(f"| {typ} | {V4_PATH[typ]} | {p['glide']} | " + ", ".join(f"{k} {v}" for k, v in p.items() if k.startswith("dolly")) + f" | {p['rot']} | {p['zoom']} | "
                 + ", ".join(f"{k} {v}" for k, v in p.items() if k.startswith("picture")) + " | " + ", ".join(f"{k} {v}" for k, v in p.items() if k.startswith("outline")) + " |")
    L += [("| crane_up / crane_down | crane | elevation: cruise law (v4.1 A) | - | (aim locked on the look point) | cruise law | - | - |" if v41 else
           "| crane_up / crane_down | crane | elevation [0, 1] | - | (aim locked on the look point) | [0, 1] | - | - |"),
          "| replay_in / replay_out | orbit_recentre | centre [0, 1] | - | (the orbit's) | - | - | - |", "",
          "**Eye.** Anchor of a phone end: P(t, d) = phone_pos(t) - d x axis_run_m x phone_fwd(t), axis_run_m = "
          f"{cfg4.axis_run_m}, with d = 1 - ramp(dolly) for the phone entered (it is 1 m behind the phone, on its optical axis - the portal - until the dolly "
          "starts, on the phone at u = 1) and d = ramp(dolly_out) for the phone left. Anchor of a 3D end: that state's eye. With C(t) the couple's damped "
          "centre and (az, r, h) cylindrical coordinates about it: az = az_A + glide x wrap180(az_B - az_A) (the planned way round: `arc.dir`), "
          "r = r_A + glide (r_B - r_A), h = h_A + glide (h_B - h_A); eye = C(t) + (r cos az, h, r sin az). Into a phone the eye reaches the phone's optical "
          "axis behind it at u = 0.75 while the dolly (from u = 0.5) already carries it forward: the picture is always in front of the camera and grows to "
          "fill the frame exactly at u = 1. Keep r >= clear_radius_m (1.4) and h in [0.5, 2.4] m.",
          "**Detours** (v3's plans of least badness: lift, sideways swing, the picture closer) stay allowed, but only as an offset that vanishes with its "
          "first two derivatives at both ends: eye += 64 u^3 (1 - u)^3 x detour (a C2 bump), never as a re-shaped cached path.",
          "**Rotation.** slerp(R_A(t), R_B(t), ramp(rot)); a phone's R is its CROPPED view (as v3). **FOV.** tan(fov / 2) = tan(fov_A / 2)^(1 - z) x "
          "tan(fov_B / 2)^z, z = ramp(zoom) (an even zoom speed in the picture); a phone's fov is its crop's.",
          "**C2 by construction.** Every weight is a smootherstep ramp (value, 1st and 2nd derivative 0 / 1 at its window's edges) of live, smooth end "
          "states: at u = 0 the camera's position, velocity and acceleration are S_A's, at u = 1 S_B's. Between a way and an orbit the camera never stops "
          "dead; out of a phone it leaves at the phone's own (handheld) motion. `easing` stays \"smootherstep\".",
          f"**Crane.** About the look point of the ground-level end: elevation = el_ground + (89.9 - el_ground) x {'cruise(u) (v4.1 A)' if v41 else 'ramp(u)'} (crane_up; mirrored for down), "
          "distance lerped by the same ramp, the aim locked on the look point, the azimuth = the orbit's live a(t) throughout (both ends are the same "
          "function: no 'turn near the ground'), Up continuous (at the top it is the overhead's orbit Up).",
          "**Replay (orbit_recentre).** The camera keeps the orbit's a(t), r, h; only its centre moves: C(t) = (1 - w) C_frozen + w C_live(t), w = ramp(u; 0, 1), "
          "C_frozen = the damped couple centre 1 ms before the dance-time jump (replay_in: before the replay's first frame; replay_out: the replay's last "
          "frame). The flash at both jumps as v3 (flash_s 0.3, alpha x rest^2.2).",
          "**Placement** (already in start_s / duration_s; do not re-place): phone_in arrives at the phone shot's F0 (a pause of the speech); phone_out "
          "starts at the next shot's F0; phone_to_phone is centred on the boundary; crane_up arrives 0.35 s into the overhead; crane_down leaves the "
          "overhead after >= 2.2 s and lets the orbit after it hold >= 1 s before the next way; a replay way starts at the jump.", "",
          "## 3. Fades", "",
          "- The entered phone's picture on its frustum: alpha = ramp(u; picture_in) (smootherstep; `flow.frustums[].film` [a, b, c, d] gives the same windows "
          "in film seconds, `v4.picture.curve: smootherstep`). The left phone's: alpha = 1 - ramp(u; picture_out). Opaque (ZWrite on) at alpha 1 as v3.",
          f"- The outline of the entered / left phone rises from the ghost's {cfg4.ghost_opacity} to 1 over outline_in, falls back over outline_out "
          "(`flow.frustums[].v4.outline` {from, to, film, curve}).",
          "- The hand-off at the arrival / departure is v3's (the cover, the picture alone in the switch frame). Splats hidden at the arrival "
          "(`video.splats_hidden_at_s`), back over `video_out.splats_back_in_s` = 0.5 s (v3 0.35); the overlays on top in over `video.overlays_on_top_in_s` "
          "= 0.5 s after the arrival, off `video_out.overlays_on_top_off_before_s` = 0.35 s before the departure (v3 0.35 / 0.25).",
          f"- Readable time (alpha >= 0.5 and the eye not yet / no longer on the phone): planned >= {cfg4.readable_s} s on every way "
          "(`transition_in.v4.picture_readable_s`; v3.2 planned 0.62 s and Unity measured a median 0.57 s on screen).", "",
          "## 4. The faint frustums (`flow.ghost_frustums`)", "", "```json", json.dumps(gh, indent=1), "```", "",
          "- Draw, for every phone in `phones` while it records (its live pose at the dance time), its frustum OUTLINE: apex, the four edges and the rim at "
          f"`depth_m` = {gh['depth_m']} m, the phone's colour, alpha {gh['opacity']}, {gh['line_px']} px constant on screen, no label, no video. In every 3D shot, "
          "every way and the overheads; not over a phone's full-frame video (`views.phone_full_frame: false`).",
          "- Outline only (no faint video rectangle), because: the owner asked for \"very, very faint camera frustums\" to keep track of where we are among "
          "the phones - the outline carries the position and the direction; a 15 % moving picture per phone would be 6 extra live video textures through all "
          "3D time (decoding every phone over the whole film) and visible motion behind the dancers; the phone entered / left still gets its full picture "
          "(the existing surface). A video rectangle can be tried later with `video: true` (not planned).",
          "- Rise: for the phone being entered / left, outline alpha = max(opacity, the way's outline curve); its picture follows flow.frustums.",
          "- Depth: ZTest LEqual against the scene (floor, room), ZWrite off, drawn before the splats' transparent pass.",
          "- Keep-out (the splats write no depth, so the depth test alone cannot hide a line in front of a dancer): per fragment, alpha x 0 inside a sphere of "
          f"{gh['keep_out']['faces']['radius_m']} m round each dancer's head (SMPL head joint + 0.08 m up), feathered {gh['keep_out']['faces']['feather_m']} m - never over a face; "
          f"alpha x 0 where the line lies between the eye and a vertical capsule of {gh['keep_out']['bodies']['radius_m']} m round each dancer's pelvis "
          f"(height {gh['keep_out']['bodies']['height_m']} m), feathered {gh['keep_out']['bodies']['feather_m']} m (screen-space test against the capsule's projection + "
          f"its depth). Near the lens: alpha fades to 0 between {gh['near_fade_m'][1]} and {gh['near_fade_m'][0]} m from the eye (the orbit passes phones).",
          ("- How often it matters (python estimate on the plan's poses, 10 Hz, without the keep-out): " + ", ".join(
              f"phone {c}: {v['over_couple_s']} s over the couple in front of them, {v['near_lens_s']} s within 1 m of the eye" for c, v in go.get("per_phone", {}).items())
           + f" (of {go.get('3d_seconds_checked')} s of 3D).") if go else "- (no estimate)",
          "- The face sidecar: the ghosts are NOT `surfaces` (with the keep-out they never cover a face; the swap ignores them). Optionally list them as "
          "`ghosts` (polylines per frame) for sidecar_check's sheet.", "",
          "## 5. The numbers this file promises (plan) and the caps to keep live", "",
          "| | value |", "|---|---|",
          f"| camera moves | {res['metrics']['moves']['total']} (v3.2 {res['baseline']['moves']['total']}; target <= {tg['moves_max']}) - {json.dumps(res['metrics']['moves']['by_type'])} |",
          f"| phone holds | {res['metrics']['phone_holds']['n']}: mean {res['metrics']['phone_holds']['mean_s']} s, median {res['metrics']['phone_holds']['median_s']} s, "
          f"shortest {res['metrics']['phone_holds']['min_s']} s (targets {(str(tg['hold_mean_min_s']) + ' / ' + str(tg['hold_median_min_s'])) if 'hold_median_min_s' in tg else 'mean ' + str(tg['hold_mean_min_s'])} / {tg['hold_min_s']} s) |",
          f"| slow orbit rates | {', '.join(str(x) for x in res['metrics']['three_d']['drift_rates_deg_s'])} deg/s |",
          f"| ways | {res['metrics']['ways']['min_s']}-{res['metrics']['ways']['max_s']} s (phone ways {', '.join(sorted({str(segs[sh.segs[0]]['camera']['transition_in']['duration_s']) for sh in shots[1:] if segs[sh.segs[0]]['camera']['transition_in']['type'] in AXIS_TYPES}))} s) |",
          f"| caps (live, every frame of a way) | turn about the couple <= {cfg4.cap_az_deg_s} deg/s, view rotation <= {cfg4.cap_rot_deg_s} deg/s, eye speed "
          f"<= {cfg4.cap_speed_m_s} m/s (ground-level ways), crane pitch <= {cfg4.cap_crane_deg_s} deg/s |",
          f"| planned peaks | turn {res['metrics']['ways']['peak_az_deg_s']['max']} deg/s, rotation {res['metrics']['ways']['peak_rot_deg_s']['max']} deg/s, speed "
          f"{res['metrics']['ways']['peak_speed_m_s']['max']} m/s, crane pitch {res['metrics']['ways']['crane_pitch_deg_s']['max']} deg/s |", "",
          ("Why these caps (v4.1, owner: 'still too many fast camera transitions ... slowly orbit'): the orbit turns at " + band + ", about one 9:16 frame "
           "width (about 24 deg) per 4-10 s; a way may peak at 2 x the orbit's top rate, 12 deg/s = 0.4 deg per frame, at the middle of a >= 4-s way, and "
           "never turns back against the orbit; the eye's speed is held to 1.2 m/s (v4.0 2.4 m/s); the crane's pitch to 20 deg/s with the cruise law (5.1-s "
           "cranes; v4.0 4.83-s smootherstep cranes at 28.8 deg/s, v3.2 1.6-1.7 s at 85 deg/s planned)." if v41 else
           "Why these caps: the orbit (the film's steady state) turns at 4-10 deg/s, about one frame width per 3-7 s in the 9:16 frame (about 30 deg wide), the "
           "rate at which a 30-fps render without motion blur pans without judder. A way may peak at 3 x the orbit's top rate, 30 deg/s = 1 deg per frame "
           "(a background point moves at most about 3 % of the frame width per frame), and only at the middle of a 2.5-5-s smootherstep; v3.2's big phone "
           "changes turned 5.6-6.7 deg per frame (168-200 deg/s). The eye-speed cap is the same turn at the orbit's radius (30 deg/s x 4.5 m = 2.4 m/s), so a "
           "radial or vertical glide is held to the same feel; the crane's pitch to the same 30 deg/s (4.8-s cranes; v3.2 1.6-1.7 s, 85 deg/s planned, 143 "
           "measured)."), "",
          "## 6. Checks on the Unity side", "",
          "`hm_film_flow --action check` over the whole film (both aspects) must report: no jump (no camera discontinuity outside the replay's two flashes); "
          "no kink - the acceleration continuous at both ends of every way (compare the one-sided 2nd differences over 3 frames: the jump below the orbit's "
          "own centripetal acceleration noise); the caps above in every frame; the eye in 0.5-2.4 m outside the 8 cranes and the 4 overheads; every orbit "
          f"frame turning at {band} about the couple (never static); each phone picture on screen and readable (alpha >= 0.5, 1-97 % of the frame) "
          f">= {cfg4.readable_s} s; the {sum(1 for sh in shots[1:] for t in [segs[sh.segs[0]]['camera']['transition_in']['type']] for _ in range(2 if t == 'phone_to_phone' else 1 if t in ('phone_in', 'phone_out') else 0))} "
          "hand-offs (every arrival on / departure from a phone) showing the phone's picture alone; the ghost lines never inside a head's keep-out and never brighter than "
          f"{cfg4.ghost_opacity} away from the entered / left phone. Then look at frames: every way in / out, the 4 cranes, the replay, 3 stretches of orbit with the "
          "ghosts.", "",
          "## 7. Run order (the Unity stage, under work/unity.lock)", "",
          "1. `python -m dancecap.film_recap_flow clips --version v4` (decodes the phones' video the v4 direction needs - `v4/direction/clips.json` - with "
          "the clips the capture has, into the capture's cameras/; POV check).",
          "2. FindDirection: look in `v4/direction/` before `v3/direction/` (or pass `--direction`); a recording of a flow-v2 direction gets the tag `_v4`.",
          "3. Implement 1-6 above behind `flow.version >= 2`, edit-mode tests (the ramps' C2 edges, a live orbit -> phone way keeps the orbit's velocity "
          "at u = 0, the overhead's orbit Up, the recentre, the ghost keep-out), `hm_film_show` on a few ways, then record both aspects (1080p) with the "
          "splat dancers and `--faces on`, the face swap as for v3.", "",
          "## 8. Unchanged", "",
          "The film's time map, soundtrack and captions (check: identical to film/'s), chapters and their cards, the layers' meaning (physics, pivots, "
          "COM dots, counterbalance dials, the legend), the 6 pivots' arrows, the hand-off, the face sidecar format, 1080p only.", "",
          "## Appendix: every way", "", "| # | film s | type | duration | from -> to | planned peaks (turn / rotation / speed / pitch) |", "|---|---|---|---|---|---|"]
    ways = {w["k"]: w for w in res["sched"]["ways"]}
    for k, sh in enumerate(shots[1:], 1):
        tr = segs[sh.segs[0]]["camera"]["transition_in"]
        w = ways.get(k, {})
        t0 = sh.f0 + tr["start_s"]
        L.append(f"| {k} | {t0:.2f}-{t0 + tr['duration_s']:.2f} | {tr['type']} ({tr['v4']['path']}) | {tr['duration_s']} s | "
                 f"{tr['from']['kind']}{' ' + tr['from']['phone'] if tr['from'].get('phone') else ''} -> {tr['to']['kind']}{' ' + tr['to']['phone'] if tr['to'].get('phone') else ''} | "
                 f"{w.get('peak_az_deg_s', '-')} / {w.get('peak_rot_deg_s', '-')} / {w.get('peak_speed_m_s', '-')} / {w.get('peak_pitch_deg_s', '-')} |")
    return "\n".join(L) + "\n"


def write_all_v4(res: dict, out: Path | None = None) -> None:
    if res["cfg4"].version == "4.1":
        return write_all_v41(res, Path(out or OUT_V41))
    doc, tl = res["doc"], res["tl"]
    out = Path(out or OUT_V40)
    (out / "narration").mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    (out / "direction.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    caps.write_timeline(out / "narration" / "timeline.json", tl)
    ch = []
    for i, c in enumerate(doc["chapters"]):
        ch.append({"title": c["title"], "startS": c["film_s"], "endS": doc["chapters"][i + 1]["film_s"] if i + 1 < len(doc["chapters"]) else doc["timeline"]["film_duration_s"],
                   "segment": c["segment"]})
    (out / "chapters.json").write_text(json.dumps({"schema": "dancecap.film_recap/1/chapters", "film_duration_s": doc["timeline"]["film_duration_s"], "chapters": ch},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")
    res["ghost"] = ghost_overlap(doc, res["metrics"], res["geo4"], res["cfg4"])
    rows = before_after_rows(res["baseline"], res["metrics"], res["targets"])
    ba = {"schema": "dancecap.film_recap_flow/v4/before_after", "before": V32_DIRECTION.name, "rows": [{"what": a, "v3.2": b, "v4": c, "target": d} for a, b, c, d in rows],
          "v3.2": _metrics_public(res["baseline"]), "v4": _metrics_public(res["metrics"]), "targets": res["targets"], "check_targets": res["checks"]["targets"]}
    (out / "before_after.json").write_text(json.dumps(ba, indent=1, default=str), encoding="utf-8")
    rep = {"schema": "dancecap.film_recap_flow/2/report", "checks": res["checks"], "targets": res["targets"], "metrics": _metrics_public(res["metrics"]),
           "baseline": _metrics_public(res["baseline"]), "runs": [{k: v for k, v in r.items() if k != "q"} | {"quality": r.get("q")} for r in res["runs"]],
           "ways": res["sched"]["ways"], "orbits": res["sched"]["plans"], "dp": res["dp"], "schedule_log": res["plog"], "ghost_overlap": res["ghost"],
           "physics_on_screen": res["physics_quant"], "clips": res["clips"], "pivots": res["pivots"], "replay": res["replay"], "t_under_arrows": res["t_notes"],
           "connected_source": res["con_src"], "protected": res["prot"],
           "physics": [{k: r[k] for k in ("id", "film", "mode", "on", "why")} for r in res["phys_rows"]]}
    (out / "direction_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "clips.json").write_text(json.dumps({"what": "per phone: the master-second ranges of video the v4 direction needs (phone holds + frustum ways in / out, padded 0.5 s)",
                                                "phones": res["clips"]}, indent=1), encoding="utf-8")
    (out / "review_v4.md").write_text(review_v4_md(res), encoding="utf-8")
    (out / "UNITY_CONTRACT_v4.md").write_text(contract_md(res), encoding="utf-8")
    (out / "audio" / "make_film_mix.py").write_text(MIX_SCRIPT.replace("the v3 direction", "the v4 direction"), encoding="utf-8")
    m = res["metrics"]
    sheet_v4(m, doc, res["geo4"], out / "flow_sheet.png",
             f"v4 camera flow: {m['moves']['total']} moves, {m['phone_holds']['n']} phone holds (mean {m['phone_holds']['mean_s']} s), slow orbits "
             f"{', '.join(str(x) for x in m['three_d']['drift_rates_deg_s'])} deg/s, no cut")
    b = res["baseline"]
    d32 = json.loads(V32_DIRECTION.read_text(encoding="utf-8"))
    sheet_v4(b, d32, res["geo4"], out / "flow_sheet_v32.png",
             f"v3.2 camera flow (the before, same tool): {b['moves']['total']} moves, {b['phone_holds']['n']} phone holds (mean {b['phone_holds']['mean_s']} s)")
    compare_plot(b, m, out / "before_after.png", res["targets"])




# ======================================================================================================== v4.1 (owner 2026-10-22, after the v3 films)
# The owner's newest words (2026-10-22, after watching the v3 films again): "still too many fast camera transitions. we should hold longer on
# POVs and slowly orbit gaussian splats. also the physics meter is occluding the figures in the vertical video. it should move to the top of the
# frame" (and: use Jev for this work).
#
# v4.1 is the v4 planner with new numbers and one new input: Jev's need_3d per utterance (dancecap.film_jev need3d -> v4/jev/need3d.json: how much
# the moment needs the slow 3D orbit with physics rather than the phone video, 0..3). The plan's value of a film second is (3 - n) / 3 as phone
# video and n / 3 as orbit, and a Lagrange multiplier on phone time is raised until the phones fill >= 45 % of the film: the smallest one that
# does, so the phone time goes where the need is lowest. Holds >= 20 s (the film may end on one), no phone-to-phone way, at most 22 moves; the
# orbit turns at one rate of 3-6 deg/s per stretch; every way >= 4 s under 12 deg/s (turn, rotation) and 1.2 m/s; the cranes follow the cruise law
# under a 20 deg/s pitch; the physics legend moves into the 9:16 top band and the phone crops keep the teachers under it. v4.0 stays
# reproducible (`build --version v4.0` -> v4/direction/v4.0/).

OUT_V40 = OUT_V4 / "v4.0"
V40_DIRECTION = OUT_V40 / "direction.json"
NEED3D = WORK / "v4" / "jev" / "need3d.json"
FLOW_SCHEMA_V41 = ("; v4.1 (flow.revision '4.1', UNITY_CONTRACT_v4.md 'v4.1'): a profile entry may be {window, law: 'cruise', ease} (the cranes), "
                   "presets.vertical.legend (the physics legend's 9:16 place), camera.v4.crop_framing, the slow orbit's vertical_band")

# the physics legend ("Skeleton load - ESTIMATED from motion", SkeletonLegend: a 300 x 74 px box at scale 1) in the 1080 x 1920 frame.
# Now (Unity, FilmOverlay.Tick): under the chapter-card slot, top = 0.14 x 1920 + (14 + 96 + 10) px = 388.8 px, scale 1.85 -> 20.3-27.4 % of the
# height: head height in the phone shots (the crop puts the teachers' box from 15 % of the height). v4.1: the top band, 8 px under the safe top.
LEGEND_UNITY_V3 = {"anchor": "top_right", "top_px_at_1080w": 388.8, "right_px_at_1080w": 14.0, "box_px": [300, 74], "scale": 1.85}
V41_LEGEND = {"what": "the physics legend ('Skeleton load - ESTIMATED from motion', the tension / neutral / compression bar) in the 9:16 frame",
              "anchor": "top_right", "top_px_at_1080w": 276.8, "right_px_at_1080w": 14.0, "box_px": [300, 74], "scale": 1.85,
              "top_frac": 0.1442, "bottom_frac": 0.2155, "x_frac": [0.4731, 0.987],
              "rule": "top = presets.vertical.safe.top_frac x H + 8 px (x W / 1080); a constant place: it never moves",
              "under_cards": "alpha x (1 - max(chapter card alpha, title alpha)): it fades under a card or the title and never steps below them",
              "labels": "the call-out labels and pills keep out of its rectangle (an obstacle; LabelCeiling under its bottom + 10 px while it shows)",
              "framing": "phone holds: the 9:16 crop puts the teachers' box top at >= 23 % of the height (camera.v4.crop_framing); 3D views: the slow orbit "
                         "frames the couple below it (vertical_band); physics_3q / pivots keep their fit_band (couple top at <= 26 %)",
              "horizontal": "unchanged (top right, title-safe margins, legend_scale 1.5)"}
V41_CROP = {"top": 0.46, "fill": 0.45, "max_fill": 0.46}            # recap_pov.PovCfg: the teachers' box centred 46 % down, 45 % of the view high
V41_ORBIT_BAND = {"fit_band": True, "top_m": 1.95, "band_lo": 0.31, "band_hi": 0.77}   # 9:16 (0 = bottom): the couple under the legend, over the captions


def legend_rect(spec: dict, W: float = 1080.0, H: float = 1920.0) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of the legend in fractions of the 9:16 frame (y down)"""
    s = float(spec.get("scale", 1.0))
    w, h = (float(v) * s for v in spec.get("box_px", [300, 74]))
    x1 = W - float(spec.get("right_px_at_1080w", 14.0)) * W / 1080.0
    y0 = float(spec["top_px_at_1080w"]) * W / 1080.0
    return (x1 - w) / W, x1 / W, y0 / H, (y0 + h) / H


OWNER_SHARE_DECISION = ("owner 2026-10-22, after seeing both v4.1 plans: record the MEANING-FIRST plan (v4/direction/, 27.6 % phone video, 4 "
                        "holds averaging 25 s, Jev puts the mechanics explanations into the slow 3D orbits), not the share-first alternative; "
                        "the 45 % phone-share target is dropped: the check holds the delivered plan to >= 25 %")


def v41_cfg() -> V4Cfg:
    ob = copy.deepcopy(V4Cfg().orbit_block)
    ob["vertical_band"] = dict(V41_ORBIT_BAND)
    return dataclasses_replace(
        V4Cfg(), version="4.1", moves_cap=22, share_min=0.45, share_check_min=0.25, hold_floor_s=20.0, hold_mean_floor_s=25.0, hold_dp_min_s=20.0, hold_max_s=80.0,
        orbit_max_s=240.0, orbit_rate_deg_s=(2.5, 6.0), orbit_rate_pref_deg_s=4.5, orbit_block=ob, link_frac=0.85,
        dur={"phone_in": [4.0, 7.0, 4.5], "phone_out": [4.0, 7.0, 4.5], "phone_to_phone": [4.0, 7.0, 5.0], "crane_up": [5.1, 5.1, 5.1],
             "crane_down": [5.1, 5.1, 5.1], "replay_in": [4.0, 4.0, 4.0], "replay_out": [4.0, 4.0, 4.0], "flight": [4.0, 7.0, 5.0]},
        cap_az_deg_s=12.0, cap_rot_deg_s=12.0, cap_speed_m_s=1.2, cap_crane_deg_s=20.0, crane_ease=0.25, crane_span_deg=74.4,
        max_excess_deg=30.0, way_margin_s=4.5, need_alpha=1.0, overhead_leave_early_s=3.0, overhead_gap_s=1.0, live_rate_frac=0.6,
        final_phone=True, quality_ladder=((0.8, 0.9, 3.0), (0.72, 0.85, 4.0)), crop=dict(V41_CROP), legend=copy.deepcopy(V41_LEGEND), orbit_lead=0.15, legend_hold_min=0.95,
        schedule_step_s=0.25, schedule_iters=14, plan="meaning", crop_smooth_s=3.0)


def v41_targets(cfg4: V4Cfg) -> dict:
    ways = [k for k in cfg4.dur if k not in ("flight", "phone_to_phone")]
    return {"revision": "4.1", "moves_max": cfg4.moves_cap,
            "share_min": cfg4.share_check_min if cfg4.share_check_min >= 0 else cfg4.share_min, "share_plan_min": cfg4.share_min,
            "share_decision": OWNER_SHARE_DECISION if cfg4.share_check_min >= 0 else None, "hold_min_s": cfg4.hold_floor_s, "hold_mean_min_s": cfg4.hold_mean_floor_s,
            "orbit_rate_deg_s": list(cfg4.orbit_rate_deg_s), "way_min_s": min(cfg4.dur[k][0] for k in ways), "cap_az_deg_s": cfg4.cap_az_deg_s,
            "cap_rot_deg_s": cfg4.cap_rot_deg_s, "cap_speed_m_s": cfg4.cap_speed_m_s, "cap_crane_deg_s": cfg4.cap_crane_deg_s, "readable_s": cfg4.readable_s,
            "ghost_opacity": cfg4.ghost_opacity, "need_clear_margin": cfg4.need_clear_margin, "legend_clear_min": 0.95, "against_max_deg_s": 1.0}


class NeedSeries:
    """Jev's need_3d (dancecap.film_jev need3d: 0 = talk, the phone video is all a viewer needs .. 3 = body mechanics the video hides) and its
    overhead Noul over FILM time (0.1-s bins), from the utterances' master times through the direction's time map. `need` is NaN outside speech;
    `filled` gives a silence the lower of its two neighbouring utterances (a demonstration without words is what the phone video shows well)"""

    def __init__(self, path: Path, doc: dict, dt: float = 0.1):
        nd = json.loads(Path(path).read_text(encoding="utf-8"))
        F = float(doc["timeline"]["film_duration_s"])
        self.dt = dt
        self.t = np.arange(0.0, F, dt) + 0.5 * dt
        n = len(self.t)
        self.need = np.full(n, np.nan)
        self.ov = np.full(n, np.nan)
        self.utts = []
        for u in nd["utterances"]:
            a, b = film_of(doc, max(0.0, float(u["start"]))), film_of(doc, float(u["end"]))
            self.utts.append({"id": u["id"], "film": [round(a, 2), round(b, 2)], "need_3d": float(u["need_3d"]), "overhead": float(u.get("overhead", 0.0)),
                              "conf": u.get("need_3d_conf"), "text": u.get("text", "")})
            m = (self.t >= a) & (self.t < b)
            self.need[m] = float(u["need_3d"])
            self.ov[m] = float(u.get("overhead", 0.0))
        self.speech = np.isfinite(self.need)
        idx = np.arange(n)
        fwd = np.maximum.accumulate(np.where(self.speech, idx, -1))
        bwd = np.minimum.accumulate(np.where(self.speech, idx, n)[::-1])[::-1]
        prev = np.where(fwd >= 0, self.need[np.clip(fwd, 0, n - 1)], np.nan)
        nxt = np.where(bwd < n, self.need[np.clip(bwd, 0, n - 1)], np.nan)
        self.filled = np.where(self.speech, self.need, np.fmin(prev, nxt))
        self.filled = np.where(np.isfinite(self.filled), self.filled, 1.0)

    def mask(self, spans) -> np.ndarray:
        m = np.zeros(len(self.t), bool)
        for a, b in spans:
            m |= (self.t >= a) & (self.t < b)
        return m

    def mean(self, m: np.ndarray) -> tuple[float | None, float]:
        """the speech-weighted mean need_3d over the bins of m, and the speech seconds it is taken over"""
        k = m & self.speech
        return (round(float(np.mean(self.need[k])), 3) if k.any() else None), round(float(k.sum() * self.dt), 1)


def legend_clear_frames(ph, cfg, rect: tuple[float, float, float, float]) -> np.ndarray:
    """per frame of a phone: can the 9:16 crop (recap_pov.crop_track's framing with `cfg`, before its smoothing) keep the teachers' union box out
    of the legend's rectangle (x0, x1, y0, y1 of the view, y down)? False where the teachers are not both detected"""
    out = np.zeros(len(ph.t), bool)
    ok = ph.ok & np.isfinite(ph.box).all(axis=1)
    if not ok.any():
        return out
    bx = ph.box[ok]
    fw, fh = ph.size
    fa = fw / fh
    cx_f, cy_f = 0.5 * (bx[:, 0] + bx[:, 2]), 0.5 * (bx[:, 1] + bx[:, 3])
    hh = bx[:, 3] - bx[:, 1]
    zmax = max(1.0, min(cfg.max_zoom, cfg.max_upscale * fh / cfg.out_h))
    z0 = np.clip(cfg.fill / np.maximum(hh, 0.05), 1.0, zmax)
    room = np.minimum(cy_f / cfg.top, (1.0 - cy_f) / (1.0 - cfg.top))
    z_room = np.minimum(1.0 / np.maximum(room, 0.2), cfg.max_fill / np.maximum(hh, 0.05))
    z = np.clip(np.maximum(z0, z_room), cfg.min_zoom, max(zmax, cfg.min_zoom))
    vh = 1.0 / z
    vw = vh * cfg.view_aspect / fa
    i = cfg.inset
    cyv = np.where(vh >= 1.0 - 2 * i, 0.5, np.clip(cy_f + (0.5 - cfg.top) * vh, vh / 2 + i, 1 - vh / 2 - i))
    cxv = np.where(vw >= 1.0 - 2 * i, 0.5, np.clip(cx_f, vw / 2 + i, 1 - vw / 2 - i))
    x0, x1 = (bx[:, 0] - (cxv - vw / 2)) / vw, (bx[:, 2] - (cxv - vw / 2)) / vw
    y0, y1 = (bx[:, 1] - (cyv - vh / 2)) / vh, (bx[:, 3] - (cyv - vh / 2)) / vh
    under = (x1 > rect[0]) & (x0 < rect[1]) & (y0 < rect[3]) & (y1 > rect[2])
    out[np.nonzero(ok)[0]] = ~under
    return out


class PhoneQ41:
    """v4.1's phone hold quality: the phone records over the hold and its ways, every 2-s piece passes the HARD rules (no known-bad pose stretch,
    the pose usable: the 3D overlays are drawn through it) and the whole hold sees both teachers (recap_pov.window_quality, cover >= 0.62);
    then the best level: 0 = v4's (whole hold strong, every 2-s piece usable), 1.. = cfg4.quality_ladder (whole cover, usable share of the 2-s
    pieces, longest run of unusable pieces) - the relaxation the owner allowed for long holds ("the best available quality that still shows both
    teachers clearly")"""

    def __init__(self, pov: director.PovSource, geo4: Geo4, cfg4: V4Cfg, cams: list[str] | None = None):
        self.pov, self.g, self.cfg4 = pov, geo4, cfg4
        self.step = 0.5
        self.grid = np.arange(0.0, 366.0, self.step)
        self.cams = [c for c in (cams or sorted(pov.phones)) if c in geo4.ph]
        self.ok, self.hard, self.clear = {}, {}, {}
        rect = legend_rect(cfg4.legend) if cfg4.legend else None
        pcfg = dataclasses_replace(pov.cfg, **cfg4.crop) if cfg4.crop else pov.cfg
        for cam in self.cams:
            ph, bad = pov.phones[cam], pov.bad.get(cam, [])
            self.ok[cam] = np.array([rp.window_quality(ph, float(t), float(t) + cfg4.hold_sub_s, bad, pov.cfg) is not None for t in self.grid])
            self.hard[cam] = np.array([rp.overlaps_bad(bad, float(t), float(t) + cfg4.hold_sub_s) is None
                                       and ph.pose_cover(float(t), float(t) + cfg4.hold_sub_s) >= pov.cfg.pose_cover for t in self.grid])
            if rect is not None:
                clr = legend_clear_frames(ph, pcfg, rect)
                self.clear[cam] = (np.concatenate([[0], np.cumsum(ph.ok)]), np.concatenate([[0], np.cumsum(clr)]))
        self.cache: dict = {}

    def clear_share(self, cam: str, d0: float, d1: float) -> float | None:
        """the share of the frames with both teachers in [d0, d1] whose box the 9:16 crop can keep out of the legend's rectangle"""
        if cam not in self.clear:
            return None
        ph = self.pov.phones[cam]
        i0, i1 = int(np.searchsorted(ph.t, d0)), int(np.searchsorted(ph.t, d1))
        n_ok, n_cl = self.clear[cam]
        n = n_ok[i1] - n_ok[i0]
        return float((n_cl[i1] - n_cl[i0]) / n) if n > 0 else 0.0

    def hold(self, cam: str, d0: float, d1: float, ml: float = 0.0, mr: float = 0.0) -> dict | None:
        key = (cam, round(d0, 2), round(d1, 2), round(ml, 2), round(mr, 2))
        if key in self.cache:
            return self.cache[key]
        res = None
        r0, r1 = self.g.recording(cam)
        if cam in self.ok and d0 - ml >= r0 - 1e-6 and d1 + mr <= r1 + 1e-6:
            i0 = int(math.ceil(d0 / self.step - 1e-9))
            i1 = int(math.floor((d1 - self.cfg4.hold_sub_s) / self.step + 1e-9))
            if i1 >= i0:
                ok, hard = self.ok[cam][i0:i1 + 1], self.hard[cam][i0:i1 + 1]
                if hard.all():
                    q = rp.window_quality(self.pov.phones[cam], d0, d1, self.pov.bad.get(cam, []), self.pov.cfg)
                    if q is not None:
                        u = float(ok.mean())
                        gap, run_ = 0, 0
                        for x in ok:
                            run_ = 0 if x else run_ + 1
                            gap = max(gap, run_)
                        gap_s = gap * self.step
                        level = 0 if (q["strong"] and ok.all()) else None
                        if level is None:
                            for L, (cmin, umin, gmax) in enumerate(self.cfg4.quality_ladder, 1):
                                if q["cover"] >= cmin - 1e-9 and u >= umin - 1e-9 and gap_s <= gmax + 1e-9:
                                    level = L
                                    break
                        clr = self.clear_share(cam, d0, d1)
                        if level is not None and (clr is None or clr >= self.cfg4.legend_hold_min - 1e-9):
                            res = q | {"level": level, "usable": round(u, 3), "worst_gap_s": round(gap_s, 1)} | ({"legend_clear": round(clr, 3)} if clr is not None else {})
        self.cache[key] = res
        return res


class WayEst:
    """per phone, how long a v4.1 way into / out of it at film f must be - the glide's radial + height distance between the slow orbit and the
    phone's portal (1 m behind it on its axis) and the 1-m axis dolly under way_speed_frac x the speed cap, the picture's readable floor - and
    whether the phone's own live motion there (its turn about the couple's damped centre, its own rotation) leaves room under the caps
    (live_rate_frac of them over the way); None = no way there"""

    def __init__(self, geo4: Geo4, cfg4: V4Cfg, m_of):
        self.g, self.cfg4, self.m_of = geo4, cfg4, m_of
        self.cache: dict = {}
        self.tr: dict = {}
        for cam, p in geo4.ph.items():
            r0, r1 = geo4.recording(cam)
            ds = np.arange(max(0.0, r0), r1, 0.1)
            if len(ds) < 5:
                continue
            pos = np.stack([np.interp(ds, p["t"], p["pos"][:, k]) for k in range(3)], axis=1)
            fwd = np.stack([np.interp(ds, p["t"], p["fwd"][:, k]) for k in range(3)], axis=1)
            fwd /= np.maximum(1e-9, np.linalg.norm(fwd, axis=1))[:, None]
            c = np.stack([np.interp(ds, geo4.t, geo4.c[:, k]) for k in range(3)], axis=1)
            rel = pos - c
            az = np.degrees(np.unwrap(np.arctan2(rel[:, 2], rel[:, 0])))
            rot = np.zeros(len(ds))
            rot[1:-1] = np.degrees(np.arccos(np.clip(np.sum(fwd[2:] * fwd[:-2], axis=1), -1.0, 1.0))) / 0.2
            rot[0], rot[-1] = rot[1], rot[-2]
            portal = pos - cfg4.axis_run_m * fwd
            relp = portal - c
            # the angle between the phone's aim and the slow orbit's view at the phone's azimuth (eye at the orbit's radius / height, looking at the
            # couple's look point): what the way's rotation must turn besides following the couple
            a = np.arctan2(relp[:, 2], relp[:, 0])
            eye = c + np.stack([cfg4.orbit_r_h[0] * np.cos(a), np.zeros(len(a)), cfg4.orbit_r_h[0] * np.sin(a)], axis=1)
            eye[:, 1] = cfg4.orbit_r_h[1]
            look = c + np.array([0.0, cfg4.orbit_look_h, 0.0])
            vd = look - eye
            vd /= np.maximum(1e-9, np.linalg.norm(vd, axis=1))[:, None]
            aim = np.degrees(np.arccos(np.clip(np.sum(vd * fwd, axis=1), -1.0, 1.0)))
            self.tr[cam] = {"d": ds, "azr": np.abs(np.gradient(az) / 0.1), "rot": rot, "r": np.hypot(relp[:, 0], relp[:, 2]), "h": portal[:, 1], "aim": aim}

    def aim(self, cam: str, f: float) -> float:
        tr = self.tr.get(cam)
        if tr is None:
            return 0.0
        i = int(np.clip(np.searchsorted(tr["d"], self.m_of(f)), 0, len(tr["d"]) - 1))
        return float(tr["aim"][i])

    def T(self, cam: str, f: float, side: str) -> float | None:
        key = (cam, round(f, 3), side)
        if key in self.cache:
            return self.cache[key]
        out = None
        tr = self.tr.get(cam)
        if tr is not None:
            c4 = self.cfg4
            typ = "phone_in" if side == "in" else "phone_out"
            lo, hi, _ = c4.dur[typ]
            d = self.m_of(f)
            i = int(np.clip(np.searchsorted(tr["d"], d), 0, len(tr["d"]) - 1))
            D = math.hypot(float(tr["r"][i]) - c4.orbit_r_h[0], float(tr["h"][i]) - c4.orbit_r_h[1])
            p = c4.prof()[typ]
            gw = p["glide"][1] - p["glide"][0]
            dw = (p.get("dolly") or p.get("dolly_out"))[1] - (p.get("dolly") or p.get("dolly_out"))[0]
            rw = p["rot"][1] - p["rot"][0]
            # the view's rotation: the phone's own aim eased over the rot window, on top of following the couple (about half the cap left)
            T_rot = 1.875 * float(tr["aim"][i]) / (rw * 0.5 * c4.cap_rot_deg_s)
            T = max(v4_way_T(typ, c4), 1.875 * max(D / gw, c4.axis_run_m / dw) / (c4.way_speed_frac * c4.cap_speed_m_s), T_rot)
            T = min(hi, math.ceil(T * 20.0 - 1e-6) / 20.0)
            a, b = (d - T, d) if side == "in" else (d, d + T)
            m = (tr["d"] >= a - 0.05) & (tr["d"] <= b + 0.05)
            if m.any() and float(tr["azr"][m].max()) <= c4.live_rate_frac * c4.cap_az_deg_s and float(tr["rot"][m].max()) <= c4.live_rate_frac * c4.cap_rot_deg_s:
                out = T
        self.cache[key] = out
        return out


def _way_peak_table(glide: tuple[float, float] = (0.0, 0.75)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """a phone way from a live orbit turning at w into a still phone that lies rho x w x T ahead (the way's own turn about the couple): the
    camera's turn rate over the way is w x ((1 - g) + g' (rho - u)), g = the glide's smootherstep. Per rho: the peak and the least of that
    factor (< 0 = the camera turns against the orbit for a moment). phone_out is the time reverse (the same numbers)."""
    u = np.linspace(0.0, 1.0, 4001)
    x = np.clip((u - glide[0]) / (glide[1] - glide[0]), 0.0, 1.0)
    g = x ** 3 * (10 - 15 * x + 6 * x * x)
    gp = np.where((x > 0) & (x < 1), 30 * x * x * (1 - x) ** 2, 0.0) / (glide[1] - glide[0])
    rho = np.linspace(0.0, 4.0, 401)
    v = (1 - g)[None, :] + gp[None, :] * (rho[:, None] - u[None, :])
    return rho, v.max(axis=1), v.min(axis=1)


_WAY_RHO, _WAY_PEAK, _WAY_MIN = _way_peak_table()


def way_peak_factor(rho: float) -> float:
    return float(np.interp(rho, _WAY_RHO, _WAY_PEAK))


def orbit_link(az_from: float, az_to: float, L: float, T_out: float, T_in: float, cfg4: V4Cfg, prefer_sign: float = 1.0,
               A_out: float = 0.0, A_in: float = 0.0) -> dict | None:
    """v4.1: can one slow orbit of L film s carry the camera from the phone it leaves (azimuth az_from about the couple) to the phone it enters
    (az_to) with its two ways under the caps and never against its own direction? The two ways turn R = r_out + r_in the orbit's way (R = D - w x
    (L - T_out - T_in) for the stretch's total turn D, either direction, up to two extra turns round), shared so both turn rho x |w| x T; rho
    >= 0.5 + orbit_lead (the phone entered is ahead of the orbit: no frame turns back), the peak turn |w| x way_peak_factor(rho) and the view's
    rotation (that turn + the phone's own aim A, eased over the way's rot window) under link_frac x the caps. The best (the lowest peaks, the
    rate nearest the preferred, the preferred direction) or None."""
    lo, hi = cfg4.orbit_rate_deg_s
    Lp = L - T_out - T_in
    if Lp < 0.0:
        return None
    rho_min = 0.5 + cfg4.orbit_lead
    cap_a, cap_r = cfg4.link_frac * cfg4.cap_az_deg_s, cfg4.link_frac * cfg4.cap_rot_deg_s
    rw = cfg4.prof()["phone_in"]["rot"]
    rot_w = rw[1] - rw[0]
    D0 = wrap180(az_to - az_from)
    best = None
    for sgn in (1.0, -1.0):
        for wabs in np.arange(lo, hi + 1e-9, 0.1):
            for kk in (-2, -1, 0, 1, 2):
                D = D0 + 360.0 * kk
                R = sgn * (D - sgn * wabs * Lp)
                if R <= 0.0:
                    continue
                rho = R / (wabs * (T_out + T_in))
                if rho < rho_min - 1e-9:
                    continue
                pk = wabs * way_peak_factor(rho)
                # the view's rotation follows the turn about the couple (it looks at the couple) plus a share of the phone's own aim (measured on
                # the v4.1 ways: the rotation peaks within about 0.5 deg/s of the turn's; WayEst lengthens a way whose phone aims far off)
                rot = pk + cfg4.link_aim_share * 1.875 * max(A_out / max(1e-3, T_out), A_in / max(1e-3, T_in)) / rot_w
                if pk > cap_a or rot > cap_r:
                    continue
                cost = max(pk / cap_a, rot / cap_r) + 0.4 * abs(wabs - cfg4.orbit_rate_pref_deg_s) / max(1e-6, hi - lo) + (0.15 if sgn != prefer_sign else 0.0)
                if best is None or cost < best[0]:
                    best = (cost, sgn * float(wabs), rho, pk, rot, R)
    if best is None:
        return None
    cost, w, rho, pk, rot, R = best
    sg = math.copysign(1.0, w)
    return {"w": round(w, 2), "rho": round(rho, 3), "r_out": round(sg * rho * abs(w) * T_out, 2), "r_in": round(sg * rho * abs(w) * T_in, 2),
            "a_out": az_from + sg * rho * abs(w) * T_out, "excess_deg": round(sg * (R - 0.5 * abs(w) * (T_out + T_in)), 2),
            "peak_az_est": round(pk, 2), "peak_rot_est": round(rot, 2), "cost": round(cost, 3)}


def v41_dp(B: list[float], cams: list[str], hold_q, free, S_or, S_ph, cfg4: V4Cfg, max_moves: int, hold_min: float, open_end: float,
           m_in, m_out, az_of, chapter_starts: list[float] = (), excess_cache: dict | None = None, aim_of=None) -> tuple[list[dict], dict]:
    """v4.1's plan: dynamic programming over the boundary candidates B (as v4_dp) for a sequence of runs (orbit | phone) maximising S_or / S_ph
    less move_penalty per way, with at most `max_moves` phone ways. No phone-to-phone way. Each way's own length: m_in(cam, f) / m_out(cam, f)
    (None = no way there); an orbit run between two phones keeps its two ways and pure_orbit_min_s of its own, its ways outside the protected
    windows, and links the two phones' azimuths at a rate in the band (orbit_excess <= max_excess_deg); a phone run lasts hold_min..hold_max_s
    and is `free`; the film may end on a phone run (cfg4.final_phone; no way out). hold_q(cam, a, b, final) -> quality | None."""
    n = len(B)
    M = max_moves
    NEG = -1e18
    pure = cfg4.pure_orbit_min_s
    best: list[dict] = [dict() for _ in range(n)]
    back: list[dict] = [dict() for _ in range(n)]
    exc = excess_cache if excess_cache is not None else {}
    aim_of = aim_of or (lambda cam, f: 0.0)

    def bonus(f):
        return cfg4.chapter_bonus if any(abs(f - c) <= 1.0 for c in chapter_starts) else 0.0

    def put(j, mode, cand, i, pmode):
        if mode not in best[j]:
            best[j][mode] = np.full(M + 1, NEG)
            back[j][mode] = [None] * (M + 1)
        cur = best[j][mode]
        for m in np.nonzero(cand > cur)[0]:
            cur[m] = cand[m]
            back[j][mode][m] = (i, pmode, int(m) - 1 if pmode is not None else 0)

    def shift(arr, val):
        out = np.full(M + 1, NEG)
        out[1:] = arr[:-1] + val
        out[out < NEG / 2] = NEG
        return out

    for j in range(1, n):
        b = B[j]
        if b > cfg4.orbit_max_s:
            break
        v = np.full(M + 1, NEG)
        v[0] = S_or(0.0, b)
        if j == n - 1:
            if b >= max(open_end, pure) - 1e-6:
                put(j, ("O", None), v, 0, None)
            continue
        for nc in cams:
            mi = m_in(nc, b)
            if mi is None or b - mi < max(open_end, pure) - 1e-6 or not free(b - mi, b):
                continue
            put(j, ("O", nc), v, 0, None)
    for j in range(1, n):
        final = j == n - 1
        for i in range(j - 1, 0, -1):
            L = B[j] - B[i]
            if L > max(cfg4.orbit_max_s, cfg4.hold_max_s):
                break
            ph_modes = [md for md in best[i] if md[0] != "O"]
            if ph_modes and L <= cfg4.orbit_max_s:
                val = S_or(B[i], B[j]) + bonus(B[i]) - cfg4.move_penalty
                for md in ph_modes:
                    pc = md[0]
                    mo = m_out(pc, B[i])
                    if mo is None or not free(B[i], B[i] + mo):
                        continue
                    for nc in ([None] if final else cams):
                        mi = 0.0 if nc is None else m_in(nc, B[j])
                        if mi is None or L - mo - mi < pure - 1e-6 or (nc is not None and not free(B[j] - mi, B[j])):
                            continue
                        pen = 0.0
                        if nc is not None:
                            key = (pc, i, nc, j, mo, mi)
                            if key not in exc:
                                exc[key] = orbit_link(az_of(pc, B[i]), az_of(nc, B[j]), L, mo, mi, cfg4, A_out=aim_of(pc, B[i]), A_in=aim_of(nc, B[j]))
                            lk = exc[key]
                            if lk is None:
                                continue
                            pen = cfg4.excess_penalty * lk["cost"] + (cfg4.same_phone_penalty if nc == pc else 0.0)
                        put(j, ("O", nc), shift(best[i][md], val - pen), i, md)
            if hold_min - 1e-6 <= L <= cfg4.hold_max_s and free(B[i], B[j]) and (not final or cfg4.final_phone):
                for cam in cams:
                    om = ("O", cam)
                    if om not in best[i]:
                        continue
                    q = hold_q(cam, B[i], B[j], final)
                    if q is None:
                        continue
                    put(j, (cam,), shift(best[i][om], S_ph(cam, B[i], B[j], q) + bonus(B[i]) - cfg4.move_penalty), i, om)
    endv, endk = NEG, None
    for md, arr in best[n - 1].items():
        if md[0] == "O" and md[1] is not None:
            continue
        m = int(np.argmax(arr))
        if arr[m] > endv:
            endv, endk = float(arr[m]), (md, m)
    if endk is None or endv < NEG / 2:
        return [], {"ok": False, "why": "no plan satisfies the constraints"}
    runs = []
    j, (md, m) = n - 1, endk
    while True:
        i, pmd, pm = back[j][md][m]
        runs.append({"kind": "orbit" if md[0] == "O" else "phone", "cam": None if md[0] == "O" else md[0], "f0": B[i], "f1": B[j], "q": None})
        if pmd is None:
            break
        j, md, m = i, pmd, pm
    runs.reverse()
    for r in runs:
        if r["kind"] == "phone":
            r["q"] = hold_q(r["cam"], r["f0"], r["f1"], r["f1"] >= B[-1] - 1e-6)
    return runs, {"ok": True, "value": round(endv, 2), "moves_phone": endk[1]}


def physics_call_mask(segs: list[dict], need: NeedSeries) -> np.ndarray:
    """the bins of the film the base director framed as a physics view with a called arrow (the owner's 'physics calls')"""
    m = np.zeros(len(need.t), bool)
    for s in segs:
        if s["camera"].get("mode") == "physics_3q" and any(a.get("calls") for a in s["annotations"]):
            m |= (need.t >= s["film"][0]) & (need.t < s["film"][1])
    return m


def build_v41(log=print, cfg4: V4Cfg | None = None, write: bool = True, hand_events: Path | None = None, need_path: Path = NEED3D,
              v40_path: Path = V40_DIRECTION, plan_only: bool = False, out: Path | None = None, alt_dir: Path | None = None) -> dict:
    cfg4 = cfg4 or v41_cfg()
    b = v4_base(hand_events)
    segs, pov = b["segs"], b["pov"]
    geo = Geo()
    geo4 = Geo4(geo)
    tg = v41_targets(cfg4)
    prot = v4_protected(segs, cfg4)
    B = v4_boundaries(segs, b["cuts"], prot)
    F = float(segs[-1]["film"][1])
    need = NeedSeries(need_path, {"segments": segs, "timeline": {"film_duration_s": F}})
    phq = PhoneQ41(pov, geo4, cfg4)
    n_over = sum(1 for p in prot if p["kind"] == "overhead")
    n_replay = sum(1 for p in prot if p["kind"] == "replay")
    fixed = 2 * n_over + 2 * n_replay
    max_phone_moves = int(cfg4.moves_cap) - fixed
    open_end = next(p["f"][1] for p in prot if p["kind"] == "opening")
    chapters = [float(c["film_s"]) for c in b["doc"]["chapters"][1:]]
    _mc: dict = {}

    def m_of(f):
        k = round(f, 4)
        if k not in _mc:
            _mc[k] = locked_dance(segs, f, f)
        return _mc[k]

    west = WayEst(geo4, cfg4, m_of)

    def m_in(cam, f):
        return west.T(cam, f, "in")

    def m_out(cam, f):
        return west.T(cam, f, "out")

    def free(a, bb):
        return not any(p["f"][0] < bb - 1e-3 and p["f"][1] > a + 1e-3 for p in prot)

    def hold_q(cam, a, bb, final):
        ml = m_in(cam, a)
        mr = 0.0 if final else m_out(cam, bb)
        if ml is None or mr is None:
            return None
        return phq.hold(cam, m_of(a + 1e-3), m_of(bb - 1e-3), ml, mr)

    _ac: dict = {}

    def az_of(cam, f):
        k = (cam, round(f, 4))
        if k not in _ac:
            d = m_of(f)
            _ac[k] = cyl(geo4.phone(cam, d)["pos"], geo4.centre(d))[0]
        return _ac[k]

    # the value of a film second: (3 - n) / 3 as phone video, n / 3 as orbit (n = Jev's need_3d; a physics call counts >= phys_call_need), + lam
    # per phone second (the share's multiplier)
    pc_mask = physics_call_mask(segs, need)
    n_eff = np.where(pc_mask, np.maximum(need.filled, cfg4.phys_call_need), need.filled)
    dt = need.dt
    c_ph = np.concatenate([[0.0], np.cumsum(cfg4.need_alpha * (1.0 - n_eff / 3.0)) * dt])
    c_or = np.concatenate([[0.0], np.cumsum(cfg4.need_alpha * n_eff / 3.0) * dt])

    def integ(c, a, bb):
        i0 = int(np.clip(round(a / dt), 0, len(c) - 1))
        i1 = int(np.clip(round(bb / dt), 0, len(c) - 1))
        return float(c[i1] - c[i0])

    qfac = {0: 1.0, 1: 0.95, 2: 0.9}
    lam = {"v": 0.0}

    def S_or(a, bb):
        return integ(c_or, a, bb)

    def S_ph(cam, a, bb, q):
        return (integ(c_ph, a, bb) + lam["v"] * (bb - a)) * (0.9 + 0.1 * min(1.0, float(q["score"]) / 0.9)) * qfac.get(q.get("level", 2), 0.85)

    exc_cache: dict = {}
    tried = []

    def plan_at(lv):
        lam["v"] = lv
        runs, info = v41_dp(B, phq.cams, hold_q, free, S_or, S_ph, cfg4, max_phone_moves, cfg4.hold_floor_s, open_end, m_in, m_out, az_of,
                            chapters, exc_cache, west.aim)
        if not info["ok"]:
            tried.append({"lam": round(lv, 4), "ok": False})
            return runs, info, -1.0
        ph = [(r["f0"], r["f1"]) for r in runs if r["kind"] == "phone"]
        share = sum(y - x for x, y in ph) / F
        hm = need.mask(ph)
        tried.append({"lam": round(lv, 4), "ok": True, "share": round(share, 4), "holds": len(ph), "need_holds": need.mean(hm)[0], "need_3d": need.mean(~hm)[0]})
        return runs, info, share

    # the smallest multiplier whose plan fills the share: the phone time goes where the need is lowest
    target = cfg4.share_min + 0.002
    runs, dpi, share = plan_at(0.0)
    lam_used = 0.0
    if cfg4.plan == "meaning":
        # the most phone time whose holds stay need_clear_margin (Jev's need_3d, speech-weighted) under the 3D views': a coarse sweep of the
        # multiplier up to the first plan that is not clear, then a bisection between the last clear one and it
        def clear(t_):
            return t_.get("ok") and t_["need_holds"] is not None and t_["need_3d"] is not None and t_["need_holds"] <= t_["need_3d"] - cfg4.need_clear_margin + 1e-9
        best_ = (runs, dpi, share, 0.0)
        good, bad_ = 0.0, None
        for lv in np.arange(0.1, cfg4.lam_max + 1e-9, 0.1):
            rm, im, sm_ = plan_at(float(lv))
            if clear(tried[-1]):
                good = float(lv)
                if sm_ > best_[2] + 1e-9:
                    best_ = (rm, im, sm_, float(lv))
                if sm_ >= target:
                    break
            else:
                bad_ = float(lv)
                break
        if bad_ is not None:
            for _ in range(6):
                mid = 0.5 * (good + bad_)
                rm, im, sm_ = plan_at(mid)
                if clear(tried[-1]):
                    good = mid
                    if sm_ > best_[2] + 1e-9:
                        best_ = (rm, im, sm_, mid)
                else:
                    bad_ = mid
        runs, dpi, share, lam_used = best_
    elif share < target:
        lo, hi = 0.0, cfg4.lam_max
        runs, dpi, share = plan_at(hi)
        lam_used = hi
        if share < target:
            log(f"v4.1: even lam {hi} gives a phone share of {share:.3f} < {cfg4.share_min}: the plan with the most phone time is kept")
        else:
            for _ in range(9):
                mid = 0.5 * (lo + hi)
                rm, im, sm_ = plan_at(mid)
                if sm_ >= target:
                    hi, runs, dpi, share, lam_used = mid, rm, im, sm_, mid
                else:
                    lo = mid
    if not dpi["ok"]:
        raise fr.RecapError(f"v4.1: {dpi['why']}")
    dpi = dict(dpi, lam=round(lam_used, 4), share=round(share, 4), search=tried)
    log(f"v4.1 plan: {len(runs)} runs ({sum(1 for r in runs if r['kind'] == 'phone')} phone holds, share {share:.3f}), lam {dpi['lam']}, "
        f"{dpi['moves_phone']} phone ways + {fixed} crane / replay ways (max {cfg4.moves_cap}), boundaries {len(B)}")

    def T0(prev, nxt, typ):
        if typ == "phone_in":
            return west.T(nxt.phone, nxt.f0, "in")
        if typ == "phone_out":
            return west.T(prev.phone, nxt.f0, "out")
        return None

    v40 = None
    if Path(v40_path).exists() and not plan_only:
        d40 = json.loads(Path(v40_path).read_text(encoding="utf-8"))
        v40 = film_metrics(d40, geo4, V4Cfg())
        v41_extra_metrics(v40, d40, geo4, V4Cfg(), need, pov)
        v40["path"] = str(v40_path)
    extra = {"need": need, "need_path": Path(need_path), "v40": v40, "pc_mask_s": round(float(pc_mask.sum() * dt), 1),
             "overheads_noul": overhead_noul(prot, need, b["doc"]["chapters"], segs), "free_stretches": free_stretches(prot, F, need)}
    if plan_only:
        return {"runs": runs, "dp": dpi, "prot": prot, "B": B, "need": need, "segs": segs, "west": west, "phq": phq, "m_of": m_of, "F": F} | extra
    base = v4_baseline(geo4, cfg4, V32_DIRECTION)
    v41_extra_metrics(base, json.loads(V32_DIRECTION.read_text(encoding="utf-8")), geo4, V4Cfg(), need, pov)
    alt_dir = Path(alt_dir) if alt_dir else None
    if alt_dir is not None and (alt_dir / "direction.json").exists():
        da = json.loads((alt_dir / "direction.json").read_text(encoding="utf-8"))
        am = film_metrics(da, geo4, cfg4)
        v41_extra_metrics(am, da, geo4, cfg4, need, pov)
        rep_ = json.loads((alt_dir / "direction_report.json").read_text(encoding="utf-8")) if (alt_dir / "direction_report.json").exists() else {}
        miss = [r["target"] for r in (rep_.get("checks") or {}).get("targets", []) if not r["ok"]]
        plan_a = (rep_.get("dp") or {}).get("plan") or ("share" if cfg4.plan == "meaning" else "meaning")
        nd_ = am.get("need") or {}
        extra["alt"] = am
        extra["alt_info"] = {"label": f"v4.1 alt ({plan_a}-first)", "dir": str(alt_dir.relative_to(PACKAGE_ROOT)) if alt_dir.is_relative_to(PACKAGE_ROOT) else str(alt_dir),
                             "plan": plan_a, "summary": f"{am['phone_holds']['n']} holds, {100 * am['phone_holds']['share']:.1f} % phone (mean {am['phone_holds']['mean_s']} s), "
                             f"need_3d holds {nd_.get('holds_mean')} vs 3D {nd_.get('three_d_mean')}, {am['moves']['total']} moves; targets missed: "
                             + ("; ".join(miss) if miss else "none")}
    dpi["plan"] = cfg4.plan
    return _v4_finish(b, segs, runs, dpi, geo, geo4, base, tg, prot, B, fixed, cfg4, phq, write, T0=T0, extra=extra, out=out)


def why_high_on_phone(res: dict, need: NeedSeries | None, cfg4: V4Cfg) -> dict:
    """for every high-need utterance a phone shows: why (the plan's own words)"""
    out = {}
    nd = (res["metrics"].get("need") or {})
    holds = res["metrics"]["phone_holds"]["list"]
    for u in nd.get("high_on_phone", []):
        h = next((x for x in holds if x["film"][1] > u["film"][0] and x["film"][0] < u["film"][1]), None)
        if h is None:
            continue
        hm = need.mean(need.mask([h["film"]]))[0] if need is not None else None
        if hm is not None and hm >= 2.0:
            out[u["id"]] = (f"inside phone {h['phone']}'s {h['hold_s']:.0f}-s hold {h['film'][0]:.0f}-{h['film'][1]:.0f} s (mean need {hm}), the one long hold in the "
                            "first free stretch: it buys the phone time and the 25-s mean (without it 17.6-18.7 % phone, mean 21.6 s)" if cfg4.plan == "meaning" else
                            f"inside phone {h['phone']}'s {h['hold_s']:.0f}-s hold (mean need {hm}): the phone share - past about 30 % the only room for >= 20-s "
                            "holds is the 9-155 s stretch")
        else:
            out[u["id"]] = (f"a short high-need moment inside phone {h['phone']}'s low-need hold {h['film'][0]:.0f}-{h['film'][1]:.0f} s (mean need {hm}): a hold lasts "
                            f">= {cfg4.hold_floor_s:g} s between pauses of the speech, so it carries its neighbours")
    return out


# ------------------------------------------------------------------------------------------------ v4.1: what the owner asked about, measured


def way_spans(doc: dict) -> list[tuple[int, str, float, float]]:
    segs = doc["segments"]
    out = []
    for k, sh in enumerate(shots_of(segs)[1:], 1):
        tr = segs[sh.segs[0]]["camera"].get("transition_in")
        if tr:
            a = sh.f0 + float(tr["start_s"])
            out.append((k, tr["type"], a, a + float(tr["duration_s"])))
    return out


def need_metrics(doc: dict, met: dict, need: NeedSeries, hi: float = 2.5, lo: float = 1.5) -> dict:
    """Jev's need_3d inside the phone holds against the 3D views (speech-weighted: only the seconds someone speaks count), and the high-need
    utterances (>= hi) the phones show / the low-need ones (< lo) the orbit shows"""
    holds = met["phone_holds"]["list"]
    hm = need.mask([h["film"] for h in holds])
    wm = need.mask([(a, b) for _, _, a, b in way_spans(doc)])
    h_mean, h_s = need.mean(hm)
    d_mean, d_s = need.mean(~hm)
    o_mean, o_s = need.mean(~hm & ~wm)
    hi_on, lo_off = [], []
    for u in need.utts:
        a, b = u["film"]
        on = sum(max(0.0, min(b, h["film"][1]) - max(a, h["film"][0])) for h in holds)
        ph = sorted({h["phone"] for h in holds if min(b, h["film"][1]) > max(a, h["film"][0])})
        if u["need_3d"] >= hi and on > 0.3:
            hi_on.append({"id": u["id"], "film": u["film"], "need_3d": u["need_3d"], "on_phone_s": round(on, 1), "of_s": round(b - a, 1), "phones": ph, "text": u["text"][:90]})
        if u["need_3d"] < lo and (b - a) - on > 0.3:
            lo_off.append({"id": u["id"], "film": u["film"], "need_3d": u["need_3d"], "in_3d_s": round((b - a) - on, 1)})
    hi_all = sum(u["film"][1] - u["film"][0] for u in need.utts if u["need_3d"] >= hi)
    lo_all = sum(u["film"][1] - u["film"][0] for u in need.utts if u["need_3d"] < lo)
    return {"holds_mean": h_mean, "holds_speech_s": h_s, "three_d_mean": d_mean, "three_d_speech_s": d_s, "orbit_mean": o_mean, "orbit_speech_s": o_s,
            "high_on_phone": hi_on, "high_on_phone_s": round(sum(x["on_phone_s"] for x in hi_on), 1), "high_total_s": round(hi_all, 1),
            "low_in_3d": lo_off, "low_in_3d_s": round(sum(x["in_3d_s"] for x in lo_off), 1), "low_total_s": round(lo_all, 1)}


def legend_overlap(doc: dict, pov: director.PovSource, rect: tuple[float, float, float, float], step: float = 0.1,
                   windows: list[tuple[float, float]] | None = None) -> dict:
    """in the 9:16 phone holds: the share of frames (with both teachers detected) in which the teachers' union box, through the hold's crop, reaches
    into the legend's rectangle (x0, x1, y0, y1 of the frame, y down); and the box top's median (fraction of the view height). windows (film s,
    v4.2): only the frames inside them count (the meter shows only there)"""
    segs = doc["segments"]
    shots = shots_of(segs)
    per, n_all, hit_all, tops = [], 0, 0, []
    for k, sh in enumerate(shots):
        if sh.kind != "phone" or sh.phone not in pov.phones:
            continue
        crop = segs[sh.segs[0]]["camera"].get("crop") or {}
        if not crop.get("zoom"):
            continue
        ph = pov.phones[sh.phone]
        fa = ph.size[0] / ph.size[1]
        n, hit = 0, 0
        for d in np.arange(sh.d0, sh.d1, step):
            if windows is not None and not any(a <= sh.f0 + (d - sh.d0) < b for a, b in windows):
                continue
            j = int(np.searchsorted(ph.t, d))
            if j <= 0 or j >= len(ph.t) or not ph.ok[j]:
                continue
            i = int(np.clip(round((d - float(crop["t0"])) / float(crop["dt"])), 0, len(crop["zoom"]) - 1))
            z = float(crop["zoom"][i])
            cxv, cyv = 0.5 + 0.5 * float(crop["cx"][i]), 0.5 - 0.5 * float(crop["cy"][i])
            vh = 1.0 / z
            vw = vh * (9.0 / 16.0) / fa
            bx = ph.box[j]
            x0, x1 = (bx[0] - (cxv - vw / 2)) / vw, (bx[2] - (cxv - vw / 2)) / vw
            y0, y1 = (bx[1] - (cyv - vh / 2)) / vh, (bx[3] - (cyv - vh / 2)) / vh
            n += 1
            tops.append(y0)
            if x1 > rect[0] and x0 < rect[1] and y0 < rect[3] and y1 > rect[2]:
                hit += 1
        per.append({"shot": segs[sh.segs[0]]["id"], "phone": sh.phone, "frames": n, "under_legend": round(hit / max(1, n), 3)})
        n_all += n
        hit_all += hit
    return {"rect": [round(x, 4) for x in rect], "frames": n_all, "under_legend_share": round(hit_all / max(1, n_all), 3),
            "box_top_median": round(float(np.median(tops)), 3) if tops else None, "per_hold": per}


def way_motion(doc: dict, met: dict) -> dict:
    """does the camera ever stop or turn back? Per way: the most it turns AGAINST the orbit's direction (deg/s about the couple; the orbit = the 3D
    end's drift sign); over the settled orbit frames: the share turning < 1 deg/s the orbit's way (static or reversing)"""
    sm = met.get("_sample")
    if sm is None:
        return {}
    segs = doc["segments"]
    shots = shots_of(segs)
    sign_of = []
    for sh in shots:
        dr = float(segs[sh.segs[0]]["camera"].get("drift_deg_s", 0.0) or 0.0) if sh.kind != "phone" else 0.0
        sign_of.append(math.copysign(1.0, dr) if abs(dr) > 1e-6 else 0.0)
    rate = sm["az_rate"]
    per = []
    trs = {k: segs[shots[k].segs[0]]["camera"]["transition_in"] for k in range(1, len(shots))}
    for k, typ, a, b in way_spans(doc):
        sgn = sign_of[k] if sign_of[k] != 0.0 else sign_of[k - 1]
        m = (sm["way"] == k) & np.isfinite(rate)
        if sgn == 0.0 or not m.any():
            continue
        # the orbit's half of a phone way (its glide weight on the orbit's side of 0.5): there the camera moves as the orbit; on the phone's side
        # it follows the phone's own live (handheld, couple-relative) motion, as in the hold
        u = (sm["t"] - a) / max(1e-6, b - a)
        prof = ((trs[k].get("v4") or {}).get("profile") or V4_PROFILE.get(typ) or {})
        gl = prof.get("glide") if isinstance(prof.get("glide"), list) else None
        mo = m.copy()
        if typ == "phone_in" and gl:
            mo &= u <= 0.5 * (gl[0] + gl[1])
        elif typ == "phone_out" and gl:
            mo &= u >= 0.5 * (gl[0] + gl[1])
        per.append({"k": k, "type": typ, "against_deg_s": round(float(max(0.0, np.max(-sgn * rate[mo]))), 2) if mo.any() else 0.0,
                    "against_whole_way_deg_s": round(float(max(0.0, np.max(-sgn * rate[m]))), 2)})
    settled = (sm["way"] < 0) & ~sm["ph_hold"] & np.isfinite(rate)
    sh_idx = np.array([max(0, int(np.searchsorted([s.f0 for s in shots], t, side="right")) - 1) for t in sm["t"]])
    sg = np.array([sign_of[i] for i in sh_idx])
    good = settled & (sg != 0.0)
    slow_or_back = good & (sg * rate < 1.0)
    return {"ways_against_max_deg_s": max((p["against_deg_s"] for p in per), default=0.0),
            "ways_against_whole_max_deg_s": max((p["against_whole_way_deg_s"] for p in per), default=0.0), "per_way": per,
            "orbit_static_or_reverse_share": round(float(slow_or_back.sum() / max(1, good.sum())), 4)}


def v41_extra_metrics(met: dict, doc: dict, geo4: Geo4, cfg4: V4Cfg, need: NeedSeries | None, pov) -> None:
    if need is not None:
        met["need"] = need_metrics(doc, met, need)
    if pov is not None:
        met["legend"] = {"v41_place": legend_overlap(doc, pov, legend_rect(V41_LEGEND)), "unity_v3_place": legend_overlap(doc, pov, legend_rect(LEGEND_UNITY_V3))}
    met["motion"] = way_motion(doc, met)
    ws = way_spans(doc)
    met["ways"]["all_min_s"] = round(min((b - a for _, _, a, b in ws), default=0.0), 2)


# ------------------------------------------------------------------------------------------------ v4.1: the overheads and Jev's overhead Noul


def overhead_noul(prot: list[dict], need: NeedSeries, chapters: list[dict], segs: list[dict], gain: float = 0.2) -> list[dict]:
    """target 5: for each of the 4 overheads, Jev's overhead Noul (the words are about floor paths, turns, foot patterns) at the overhead and the
    best utterance of its chapter; an overhead moves only to an utterance gain higher whose span is free of the pivots (a pivot is a ground-level
    3D view) - else it keeps v4.0's time (the reason is written)"""
    rows = []
    piv = [(s["film"][0], s["film"][1]) for s in segs if s.get("pivot")]
    for p in prot:
        if p["kind"] != "overhead":
            continue
        c0, c1 = p["core"]
        ch = next((c for c in reversed(chapters) if c["film_s"] <= c0 + 1e-6), chapters[0])
        k = chapters.index(ch)
        ch1 = chapters[k + 1]["film_s"] if k + 1 < len(chapters) else 1e9
        at = [u for u in need.utts if u["film"][1] > c0 and u["film"][0] < c1]
        here = max((u["overhead"] for u in at), default=0.0)
        cand = sorted((u for u in need.utts if u["film"][0] >= ch["film_s"] - 1e-6 and u["film"][1] <= ch1 + 1e-6), key=lambda u: -u["overhead"])
        best = cand[0] if cand else None
        row = {"overhead": p["id"], "core": [round(c0, 2), round(c1, 2)], "chapter": ch["title"], "noul_here": round(here, 2),
               "best": None if best is None else {"id": best["id"], "film": best["film"], "noul": best["overhead"], "text": best["text"][:70]}}
        if best is None or best["overhead"] < here + gain or any(a < best["film"][1] and b > best["film"][0] for a, b in [(c0, c1)]):
            row["decision"] = "kept (it already sits on its chapter's highest overhead Noul, or none is clearly higher)"
        elif any(a < best["film"][1] + 6.1 and b > best["film"][0] - 6.1 for a, b in piv):
            row["decision"] = (f"kept: the chapter's higher Noul ({best['id']}, {best['overhead']}) is a back-to-the-axis phrase / next to a pivot - "
                               "the pivot is a ground-level 3D view and the crane up / down would run through it")
        else:
            row["decision"] = "a move would help (not done: no free span with its cranes)"
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ v4.1: review, contract, plots


def before_after_rows_n(cols: list[tuple[str, dict]], tg: dict) -> list[list[str]]:
    """the before / after table: [what, value per column..., v4.1 target]"""
    def by(m, *ts):
        return sum(m["moves"]["by_type"].get(t, 0) for t in ts)

    def g(m, *keys, fmt="{}"):
        x = m
        for k in keys:
            if not isinstance(x, dict) or k not in x or x[k] is None:
                return "-"
            x = x[k]
        return fmt.format(x)

    def nd(m):
        n = m.get("need") or {}
        return n
    rows = [
        ["camera moves (every way)", [str(m["moves"]["total"]) for _, m in cols], f"<= {tg['moves_max']}"],
        ["... phone in / out / phone to phone", [f"{by(m, 'phone_in')} / {by(m, 'phone_out')} / {by(m, 'phone_to_phone')}" for _, m in cols], "no phone to phone"],
        ["... crane up / down", [f"{by(m, 'crane_up')} / {by(m, 'crane_down')}" for _, m in cols], "4 / 4"],
        ["... 3D flights / replay ways", [f"{by(m, 'flight')} / {by(m, 'replay_in', 'replay_out')}" for _, m in cols], "0 flights"],
        ["shots", [str(m["shots"]) for _, m in cols], ""],
        ["phone holds: n", [str(m["phone_holds"]["n"]) for _, m in cols], ""],
        ["phone hold mean / median", [f"{m['phone_holds']['mean_s']} / {m['phone_holds']['median_s']} s" for _, m in cols], f"mean >= {tg['hold_mean_min_s']} s"],
        ["phone hold shortest / longest", [f"{m['phone_holds']['min_s']} / {m['phone_holds']['max_s']} s" for _, m in cols], f"every hold >= {tg['hold_min_s']} s"],
        ["phone full-frame video, total (share)", [f"{m['phone_holds']['sum_s']} s ({100 * m['phone_holds']['share']:.1f} %)" for _, m in cols], f">= {100 * tg['share_min']:.0f} %"],
        ["the phones held", [", ".join(f"{h['phone']} {h['hold_s']:.0f}s" for h in m["phone_holds"]["list"]) if m["phone_holds"]["n"] <= 8 else f"{m['phone_holds']['n']} holds" for _, m in cols], ""],
        ["3D between the ways (orbit + overheads)", [f"{m['three_d']['settled_s']} s ({100 * m['three_d']['settled_share']:.1f} %)" for _, m in cols], "more slow orbit"],
        ["3D incl. the ways", [f"{m['three_d']['with_ways_s']} s" for _, m in cols], ""],
        ["orbit rates (planned, deg/s)", [", ".join(f"{x:g}" for x in m["three_d"]["drift_rates_deg_s"]) for _, m in cols], f"{tg['orbit_rate_deg_s'][0]:g}-{tg['orbit_rate_deg_s'][1]:g}, one per orbit"],
        ["3D time with a static camera", [f"{m['three_d']['static_s']} s" for _, m in cols], "0"],
        ["turn about the couple in the 3D views (sampled): median / p90 / max", [" / ".join(g(m, "three_d", "sampled_az_deg_s", k) for k in ("median", "p90", "max")) + " deg/s" for _, m in cols], ""],
        ["turn over ALL 3D time incl. ways: p90 / p99 / max", [" / ".join(g(m, "three_d", "sampled_az_incl_ways_deg_s", k) for k in ("p90", "p99", "max")) + " deg/s" for _, m in cols], ""],
        ["ways: n, total time", [f"{m['ways']['n']}, {m['ways']['sum_s']} s" for _, m in cols], ""],
        ["ways: duration min / median / max", [f"{m['ways']['min_s']} / {m['ways']['median_s']} / {m['ways']['max_s']} s" for _, m in cols], f">= {tg['way_min_s']} s"],
        ["ground-level ways: peak turn about the couple max / median", [f"{g(m, 'ways', 'peak_az_deg_s', 'max')} / {g(m, 'ways', 'peak_az_deg_s', 'median')} deg/s" for _, m in cols], f"<= {tg['cap_az_deg_s']} deg/s"],
        ["ground-level ways: peak view rotation max / median", [f"{g(m, 'ways', 'peak_rot_deg_s', 'max')} / {g(m, 'ways', 'peak_rot_deg_s', 'median')} deg/s" for _, m in cols], f"<= {tg['cap_rot_deg_s']} deg/s"],
        ["ground-level ways: peak eye speed max / median", [f"{g(m, 'ways', 'peak_speed_m_s', 'max')} / {g(m, 'ways', 'peak_speed_m_s', 'median')} m/s" for _, m in cols], f"<= {tg['cap_speed_m_s']} m/s"],
        ["cranes: peak pitch max", [f"{g(m, 'ways', 'crane_pitch_deg_s', 'max')} deg/s" for _, m in cols], f"<= {tg['cap_crane_deg_s']} deg/s"],
        ["turning against the orbit: the orbit's half of a way / whole ways (incl. the phone's own motion)",
         [f"{g(m, 'motion', 'ways_against_max_deg_s')} / {g(m, 'motion', 'ways_against_whole_max_deg_s')} deg/s" for _, m in cols], f"<= {tg['against_max_deg_s']} deg/s"],
        ["Jev need_3d (speech-weighted): phone holds / 3D views / orbits between the ways",
         [f"{nd(m).get('holds_mean')} / {nd(m).get('three_d_mean')} / {nd(m).get('orbit_mean')}" for _, m in cols], f"holds <= 3D - {tg['need_clear_margin']}"],
        ["speech with need_3d >= 2.5 shown on a phone", [f"{nd(m).get('high_on_phone_s')} of {nd(m).get('high_total_s')} s" for _, m in cols], "as little as possible"],
        ["speech with need_3d < 1.5 shown in 3D", [f"{nd(m).get('low_in_3d_s')} of {nd(m).get('low_total_s')} s" for _, m in cols], "as little as possible"],
        ["9:16 phone holds: teachers' box under the legend, Unity's present place (20.3-27.4 % down)",
         ["" for _ in cols], ""],
        ["9:16 phone holds: teachers' box under the legend in its v4.1 place (14.4-21.6 % down)", ["" for _ in cols], f"<= {100 * (1 - tg['legend_clear_min']):.0f} %"],
        ["the crop's own motion in a hold: pan (view widths/s) / zoom (rel/s)", [f"{m['phone_holds'].get('crop_pan_view_per_s')} / {m['phone_holds'].get('crop_zoom_rel_per_s')}" for _, m in cols], ""],
    ]
    for r in rows:
        if r[0].startswith("9:16 phone holds: teachers' box under the legend, Unity"):
            r[1] = [(f"{100 * m['legend']['unity_v3_place']['under_legend_share']:.1f} % of {m['legend']['unity_v3_place']['frames']} frames"
                     if (m.get("legend") or {}).get("unity_v3_place") else "-") for _, m in cols]
        if r[0].startswith("9:16 phone holds: teachers' box under the legend in its v4.1"):
            r[1] = [(f"{100 * m['legend']['v41_place']['under_legend_share']:.1f} %" if (m.get("legend") or {}).get("v41_place") else "-") for _, m in cols]
    return [[a] + list(vals) + [t] for a, vals, t in rows]


def compare_plot_n(cols: list[tuple[str, dict]], out: Path, need: NeedSeries | None, tg: dict, title: str) -> Path:
    """every version on one page: what is on screen over the film (one timeline each), Jev's need_3d over the film with each version's phone
    holds under it, the turn rate about the couple (log) and every phone hold's length against the targets"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(cols)
    fig = plt.figure(figsize=(20, 4.0 + 1.0 * n + 7.0))
    gs = fig.add_gridspec(n + 3, 1, height_ratios=[0.7] * n + [1.8, 2.2, 1.8], hspace=0.65)
    F = max(float(m["_sample"]["t"][-1]) for _, m in cols)
    pal = ["#e34948", "#8a8984", "#2a78d6", "#eb6834", "#6250d6"]
    for row, (lab, m) in enumerate(cols):
        ax = fig.add_subplot(gs[row])
        sm = m["_sample"]
        t = sm["t"]
        kk = _kind_series(m)
        i = 0
        while i < len(t):
            j = i
            while j + 1 < len(t) and kk[j + 1] == kk[i]:
                j += 1
            ax.axvspan(t[i], t[j] + 1 / 30, color=V4_COL.get(kk[i], "#999"), lw=0)
            i = j + 1
        ax.set_xlim(0, F)
        ax.set_yticks([])
        h = m["phone_holds"]
        nd_ = m.get("need") or {}
        ax.set_title(f"{lab}: {m['moves']['total']} moves (dark grey), {h['n']} phone holds (orange) {h['sum_s']:.0f} s = {100 * h['share']:.1f} %, "
                     f"3D (blue) + overheads (violet) {m['three_d']['settled_s']:.0f} s"
                     + (f"; Jev need_3d in holds {nd_.get('holds_mean')} vs 3D {nd_.get('three_d_mean')}" if nd_ else ""), loc="left", fontsize=10.5)
    ax = fig.add_subplot(gs[n])
    if need is not None:
        ax.fill_between(need.t, 0, np.where(need.speech, need.need, np.nan), step="mid", color="#52514e", alpha=0.35, lw=0, label="need_3d (speech)")
        for k, (lab, m) in enumerate(cols):
            for hh in m["phone_holds"]["list"]:
                ax.plot(hh["film"], [3.2 + 0.25 * k] * 2, color=pal[k % len(pal)], lw=4, solid_capstyle="butt")
            ax.text(F + 1, 3.2 + 0.25 * k, lab, fontsize=8, va="center", color=pal[k % len(pal)])
        ax.axhline(1.5, color="#2a78d6", ls=":", lw=1)
        ax.axhline(2.5, color="#e34948", ls=":", lw=1)
    ax.set_xlim(0, F)
    ax.set_ylim(0, 3.2 + 0.25 * n)
    ax.set_ylabel("Jev need_3d")
    ax.set_title("Jev need_3d per utterance (0 talk .. 3 mechanics the video hides; dotted 1.5 / 2.5) and each version's phone holds above it", loc="left", fontsize=10.5)
    ax = fig.add_subplot(gs[n + 1])
    for k, (lab, m) in enumerate(cols):
        sm = m["_sample"]
        ax.plot(sm["t"], np.clip(np.abs(sm["az_rate"]), 0.3, 1000), color=pal[k % len(pal)], lw=0.8, label=lab)
    ax.axhspan(tg["orbit_rate_deg_s"][0], tg["orbit_rate_deg_s"][1], color="#2a78d6", alpha=0.10, lw=0)
    ax.axhline(tg["cap_az_deg_s"], color="#52514e", ls="--", lw=1)
    ax.text(2, tg["cap_az_deg_s"] * 1.08, f"v4.1 cap {tg['cap_az_deg_s']:g} deg/s", fontsize=9, color="#52514e")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 1000)
    ax.set_xlim(0, F)
    ax.set_ylabel("turn about the couple\n(deg/s, log)")
    ax.legend(loc="upper right", frameon=False, ncol=n)
    ax.set_title(f"how fast the camera turns about the couple, every frame (the plan's poses at 30 Hz; band = the v4.1 orbit {tg['orbit_rate_deg_s'][0]:g}-"
                 f"{tg['orbit_rate_deg_s'][1]:g} deg/s)", loc="left", fontsize=10.5)
    ax = fig.add_subplot(gs[n + 2])
    for k, (lab, m) in enumerate(cols):
        hs = [x["hold_s"] for x in m["phone_holds"]["list"]]
        ax.scatter(hs, np.full(len(hs), k) + np.linspace(-0.12, 0.12, len(hs)), s=55, color=pal[k % len(pal)], edgecolor="white", lw=1.2, zorder=3)
        ax.text(max(hs) + 1.0, k, f"n {len(hs)}, mean {np.mean(hs):.1f} s, min {min(hs):.1f} s", va="center", fontsize=9)
    ax.axvline(tg["hold_min_s"], color="#52514e", ls=":", lw=1)
    ax.axvline(tg["hold_mean_min_s"], color="#52514e", ls="--", lw=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for lab, _ in cols])
    ax.set_xlim(0, max(x["hold_s"] for _, m in cols for x in m["phone_holds"]["list"]) + 30)
    ax.set_xlabel(f"phone hold (s): dotted = every hold >= {tg['hold_min_s']:g} s, dashed = mean >= {tg['hold_mean_min_s']:g} s")
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    fig.suptitle(title, fontsize=14, x=0.01, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=72, bbox_inches="tight")
    plt.close(fig)
    return out


def _md_table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]


def review_v41_md(res: dict) -> str:
    doc, met, tg, ch, cfg4 = res["doc"], res["metrics"], res["targets"], res["checks"], res["cfg4"]
    segs = doc["segments"]
    shots = shots_of(segs)
    cols = res["columns"]
    nd = met.get("need") or {}
    missed = [r for r in ch["targets"] if not r["ok"]]
    alt = res.get("alt_info")
    L = [f"# v4.1 direction review: {doc['title']} ({doc['name']})", "",
         "Owner 2026-10-22, after watching the v3 films again: \"still too many fast camera transitions. we should hold longer on POVs and slowly orbit "
         "gaussian splats. also the physics meter is occluding the figures in the vertical video. it should move to the top of the frame\" (and: use Jev).", "",
         f"Plan: **{cfg4.plan}-first** (`build --version v4.1 --plan {cfg4.plan}`). Film {doc['timeline']['film_duration_s']} s; time map, captions, chapters "
         "and soundtrack = film/'s, v3.2's and v4.0's. " + ("Checks: PASS." if not ch["bad"] else f"Checks: {len(missed)} of {len(ch['targets'])} targets missed "
         f"({'; '.join(r['target'] for r in missed)}) - the check exits 1."), ""]
    L += ["## What this version decides, in short", ""]
    sh = [t for t in res["dp"].get("search", []) if t.get("ok")]
    am = res.get("alt") or {}
    smax = max([t["share"] for t in sh] + ([am["phone_holds"]["share"]] if am else []), default=0.0)
    an = am.get("need") or {}
    L += [f"- The owner's 'hold longer on POVs': every phone hold is >= {tg['hold_min_s']:g} s (v4.0: 12.0-30.1 s), the mean {met['phone_holds']['mean_s']} s "
          f"(v4.0 19.8 s), {met['phone_holds']['n']} holds, at most {tg['moves_max']} moves ({met['moves']['total']} used, no phone-to-phone way, no 3D flight).",
          f"- 'Slowly orbit': one rate per orbit, {tg['orbit_rate_deg_s'][0]:g}-{tg['orbit_rate_deg_s'][1]:g} deg/s; every way >= {tg['way_min_s']:g} s with its peak "
          f"turn {met['ways']['peak_az_deg_s']['max']} deg/s, rotation {met['ways']['peak_rot_deg_s']['max']} deg/s, speed {met['ways']['peak_speed_m_s']['max']} m/s "
          f"(caps {tg['cap_az_deg_s']:g} / {tg['cap_rot_deg_s']:g} / {tg['cap_speed_m_s']:g}); cranes {met['ways']['crane_pitch_deg_s']['max']} deg/s pitch (cap {tg['cap_crane_deg_s']:g}).",
          (f"- Jev: phone holds where the need is low. Speech-weighted need_3d {nd.get('holds_mean')} in the holds vs {nd.get('three_d_mean')} in the 3D views."
           if (nd.get('holds_mean') is not None and nd.get('three_d_mean') is not None and nd['holds_mean'] <= nd['three_d_mean'] - tg['need_clear_margin']) else
           f"- Jev: NOT met - the speech-weighted need_3d is {nd.get('holds_mean')} in the holds against {nd.get('three_d_mean')} in the 3D views (the phone time "
           "this plan buys lies where Jev says the 3D view is needed most)."),
          f"- The conflict: the >= {100 * tg['share_min']:.0f} % phone share cannot be had together with the other targets on this film. The most phone time "
          f"any plan reached is {100 * smax:.1f} %" + (f" (the share-first plan, {alt['label']}: its holds' need {an.get('holds_mean')} is ABOVE the 3D "
          f"views' {an.get('three_d_mean')} - it shows the counterbalance, frame and arm-length chapters on the phones)" if alt else "")
          + (". This build is meaning-first because it misses one target (the share) where the share-first plan misses two (the share and Jev's "
             "placement); the alternative is built next to it." if cfg4.plan == "meaning" else
             ". This build is the share-first ALTERNATIVE (the most phone time the rules allow); the delivered v4.1 in v4/direction/ is meaning-first.")
          + " See 'Share against meaning' below.",
          "- The physics legend: its place is Unity-only (FilmOverlay.Tick sets SkeletonLegend.TopOverride: 20.3-27.4 % down in 9:16, head height in "
          "the phone shots). The direction now carries presets.vertical.legend (14.4-21.6 % down, the top band under the safe top) and every phone hold's "
          f"9:16 crop puts the teachers lower (box top median {(met.get('legend') or {}).get('v41_place', {}).get('box_top_median')} of the height); "
          "UNITY_CONTRACT_v4.md 'v4.1' has the exact requirement.", ""]
    L += ["## Before / after (every column measured by `film_metrics` the same way; v4.0 = v4/direction/v4.0/)", ""]
    L += _md_table(["", *[lab for lab, _ in cols], "v4.1 target"], res["ba_rows"])
    L += ["", "Sampled numbers come from the plan's poses (`PoseModel`, 30 Hz): estimates of what the viewer sees; Unity re-checks the live camera "
              "(`hm_film_flow --action check`, UNITY_CONTRACT_v4.md section 6). need_3d means are over the seconds someone speaks (77 utterances, "
              "work/larissa_kadu_recap/v4/jev/need3d.json).", ""]
    L += ["## The owner's targets (the `check` command exits 1 when one is missed)", ""]
    L += _md_table(["target", "want", "measured", ""], [[r["target"], r["want"], r["measured"], "OK" if r["ok"] else "MISSED"] for r in ch["targets"]])
    L += ["", "## Share against meaning (why 45 % is not reached)", "",
          "The plan's value of a film second is (3 - n) / 3 as phone video and n / 3 as orbit (n = Jev's need_3d; a base-director physics call counts "
          f">= {cfg4.phys_call_need}), plus a multiplier lam per phone second. lam = 0 is pure meaning; raising it buys phone time. Every plan the planner "
          "made (dynamic programming over the speech's pauses with all the hard rules: holds >= 20 s, <= 22 moves, the protected 3D windows, the ways' "
          "caps and no reversal, the legend clearance):", ""]
    L += _md_table(["lam", "phone share", "holds", "need_3d holds", "need_3d 3D", "clear?"],
                   [[t["lam"], f"{100 * t['share']:.1f} %", t["holds"], t["need_holds"], t["need_3d"],
                     "yes" if (t["need_holds"] is not None and t["need_3d"] is not None and t["need_holds"] <= t["need_3d"] - tg["need_clear_margin"]) else "no"]
                    for t in sorted(sh, key=lambda x: x["lam"])])
    fr_ = res.get("free_stretches") or []
    if am:
        L += ["", f"The share-first search (the alternative's own build) went up to lam {cfg4.lam_max:g}: its most phone time is {100 * am['phone_holds']['share']:.1f} % "
                  f"with holds' need {an.get('holds_mean')} against the 3D views' {an.get('three_d_mean')}."]
    L += ["", "Why: the 3D windows the standing decisions fix (the opening under the title card, the 6 pivots, the 4 overheads with their 5.1-s cranes and "
              "1-s settles, the replay with its two 4-s ways) cut the film into free stretches; a phone hold also needs its two ways (4-7 s) inside one, "
              "and an orbit between two holds must carry the camera from one phone's azimuth to the next at 2.5-6 deg/s with ways under 12 deg/s that "
              "never turn back, which only some pairs of phones and pauses allow. The free stretches (> 3 s):", ""]
    L += _md_table(["free stretch (film s)", "length", "fits a 20-s hold", "need_3d there (speech mean)"],
                   [[f"{a:.1f}-{b:.1f}", f"{b - a:.1f} s", "yes" if ok else "no", nm] for a, b, ok, nm in fr_])
    L += ["", "The low-need talk outside those stretches (the opening jokes, 162-172 s between pivot 1 and the first overhead, 243-246 s, the "
              "momentum talk at 256-273 s between the replay and the third overhead, 306-310 s) sits in pieces too short for a 20-s hold. So past about "
              "30 % the only room left is the long first stretch (the counterbalance, frame and arm-length chapters, need 2.4-3.0): more phone time there "
              "is exactly what Jev says needs the 3D view.", ""]
    if alt:
        L += [f"The alternative, {alt['label']}, is built in `{alt['dir']}` (same code: `build --version v4.1 --plan {alt['plan']} --out <dir>`): "
              f"{alt['summary']}", ""]
    L += ["## The plan: runs", ""]
    orbits = res["sched"]["plans"]
    need = res.get("need")
    rr = []
    for ri, r in enumerate(res["runs"]):
        pl = orbits.get(ri)
        q = r.get("q") or {}
        nm = need.mean(need.mask([(r["f0"], r["f1"])]))[0] if need is not None else None
        rr.append([ri, f"{r['f0']:.2f}-{r['f1']:.2f}", f"{r['f1'] - r['f0']:.1f} s", "phone " + r["cam"] if r["kind"] == "phone" else "slow orbit", nm,
                   (f"L{q.get('level')} cover {q.get('cover')} usable {q.get('usable')} legend-clear {q.get('legend_clear')}" if q else ""),
                   "" if pl is None else f"{pl['w']} deg/s (ways turn {pl['r_out']} / {pl['r_in']} deg)"])
    L += _md_table(["run", "film s", "length", "view", "need_3d", "phone quality", "orbit"], rr)
    lv = [r for r in res["runs"] if r["kind"] == "phone" and (r.get("q") or {}).get("level", 0) > 0]
    L += ["", "Phone quality: L0 = v4's rule (the whole hold 'strong': both teachers in >= 80 % of the frames, and every 2-s piece usable); L1 = whole "
              "hold >= 0.8, >= 90 % of the 2-s pieces usable, no unusable run longer than 3 s; L2 = 0.72 / 85 % / 4 s. Never relaxed: the pose (the "
              "3D overlays are drawn through it), the known-bad stretches, the phone recording through its ways, the legend clearance (>= 95 % of the "
              "hold's frames). Relaxed holds: " + ("; ".join(f"phone {r['cam']} {r['f0']:.1f}-{r['f1']:.1f} s (L{r['q']['level']}: {100 * r['q']['usable']:.1f} % of the "
              f"2-s pieces usable, longest gap {r['q']['worst_gap_s']} s, cover {r['q']['cover']})" for r in lv) or "none") + ".", ""]
    L += ["## Jev: need_3d in the holds and in the 3D views", "",
          f"Speech-weighted need_3d: phone holds {nd.get('holds_mean')} over {nd.get('holds_speech_s')} s of speech, all 3D (orbits, overheads, ways) "
          f"{nd.get('three_d_mean')} over {nd.get('three_d_speech_s')} s, the orbits between the ways {nd.get('orbit_mean')}.", "",
          "High-need speech (need_3d >= 2.5) the phones show, and why:", ""]
    hi = nd.get("high_on_phone") or []
    L += [f"The holds' mean stays clearly under the 3D views', but a {tg['hold_mean_min_s']:g}-s mean needs one long hold, and the only long free stretch is "
          "the first one (need 2.4-3.0): the plans without it have 17.6-18.7 % phone and a 21.6-s mean (the search above).", ""]
    L += (_md_table(["utterance", "film s", "need_3d", "on the phone", "phone", "words", "why"],
                    [[u["id"], f"{u['film'][0]}-{u['film'][1]}", u["need_3d"], f"{u['on_phone_s']} of {u['of_s']} s", ", ".join(u["phones"]), u["text"][:60],
                      res["why_high"].get(u["id"], "")] for u in hi]) if hi else ["- none"])
    L += ["", f"Low-need speech (need_3d < 1.5) the 3D views show: {nd.get('low_in_3d_s')} of {nd.get('low_total_s')} s (the opening under the title card, "
              "the pieces between the pivots, overheads and the replay that no 20-s hold fits).", ""]
    L += ["## The overheads and Jev's overhead Noul (target 5)", ""]
    L += _md_table(["overhead", "core film s", "chapter", "Noul at it", "chapter's highest", "decision"],
                   [[o["overhead"], f"{o['core'][0]}-{o['core'][1]}", o["chapter"], o["noul_here"],
                     "-" if not o["best"] else f"{o['best']['id']} {o['best']['noul']} ({o['best']['film'][0]}-{o['best']['film'][1]})", o["decision"]] for o in res["overheads_noul"]])
    L += ["", "The cranes follow the cruise law (the pitch rate eases in over the first quarter, holds, eases out over the last quarter: peak 1.33 x the "
              "mean) and take 5.1 s: pitch 19.4 deg/s (v4.0: smootherstep, 4.83 s, 28.8 deg/s). Where a phone hold follows, a crane down leaves its "
              f"overhead up to {cfg4.overhead_leave_early_s:g} s early, never before the overhead's 2.2-s hold.", ""]
    L += ["## Every shot and the way into it", ""]
    ways = {w["k"]: w for w in res["sched"]["ways"]}
    rows = []
    for k, sh_ in enumerate(shots):
        s0 = segs[sh_.segs[0]]
        tr = s0["camera"].get("transition_in")
        role = (s0["camera"].get("v4") or {}).get("role", sh_.mode)
        if not tr:
            rows.append([k, f"{sh_.f0:.1f}-{sh_.f1:.1f}", s0["id"], role, sh_.phone or "", "(first shot)", "", "", ""])
            continue
        w = ways.get(k, {})
        rows.append([k, f"{sh_.f0:.1f}-{sh_.f1:.1f}", s0["id"], role + (" (replay)" if sh_.replay else ""), sh_.phone or "", tr["type"], f"{tr['start_s']:+.2f} s",
                     f"{tr['duration_s']:.2f} s", f"{w.get('peak_az_deg_s', '-')} / {w.get('peak_rot_deg_s', '-')} / {w.get('peak_speed_m_s', '-')} / {w.get('peak_pitch_deg_s', '-')}"])
    L += _md_table(["#", "film s", "shot", "view", "phone", "way in", "starts", "duration", "planned peaks: turn / rotation / speed / pitch"], rows)
    notes = [f"- {segs[shots[w['k']].segs[0]]['id']} ({w['type']}): {'; '.join(w['notes'])}" for w in res["sched"]["ways"] if w.get("notes")]
    L += ["", "Placement notes:", ""] + (notes or ["- none"])
    lg = met.get("legend") or {}
    lg40 = ((res.get("v40") or {}).get("legend") or {})
    L += ["", "## The physics legend (target 7)", "",
          "Where it is defined: Unity only. head-movement Assets/Film/FilmOverlay.cs Tick() sets SkeletonLegend.TopOverride in 9:16 to "
          "max(HUD bottom + 10 u, safe top + 8 u) and then below the chapter-card slot (safe top + 14 u + 96 u + 10 u = 388.8 px at 1080 x 1920), "
          "scale 1.85 (a 555 x 137 px box, right margin 14 px): 20.3-27.4 % of the height. The direction only carries presets.horizontal.legend_scale "
          "and the safe bands; nothing in caption_style / vertical_9x16 places it.",
          "What v4.1 does: (1) presets.vertical.legend in the direction = the new place (top 276.8 px = 14.42 % = the safe top + 8 px, bottom 21.55 %, "
          "right-aligned, the same box and scale), fading under a chapter card or the title instead of stepping below them, the labels kept out of it; "
          f"(2) every phone hold's 9:16 crop is made with the teachers' box centred {cfg4.crop['top']} down and {cfg4.crop['fill']} of the view high "
          f"(v4.0: 0.40 / 0.50), so their heads start under the legend; (3) a hold is only planned where that crop keeps the teachers clear of the legend "
          f"in >= {100 * cfg4.legend_hold_min:.0f} % of its frames; (4) the slow orbit carries a 9:16 vertical_band (FitBand {V41_ORBIT_BAND['band_lo']}-"
          f"{V41_ORBIT_BAND['band_hi']} from the bottom) for Unity to keep the couple below the legend in 3D.", "",
          "Measured on the plan (every frame with both teachers detected, through the hold's own crop; the legend's rectangle in the 1080 x 1920 frame):", ""]
    L += _md_table(["", "teachers' box under the legend, Unity's present place", "... in the v4.1 place", "box top median (of the height)"],
                   [[lab, f"{100 * (m.get('legend') or {}).get('unity_v3_place', {}).get('under_legend_share', float('nan')):.1f} %",
                     f"{100 * (m.get('legend') or {}).get('v41_place', {}).get('under_legend_share', float('nan')):.1f} %",
                     (m.get("legend") or {}).get("v41_place", {}).get("box_top_median")] for lab, m in cols if m.get("legend")])
    L += ["", "Per v4.1 hold: " + ", ".join(f"{x['shot']} (phone {x['phone']}) {100 * x['under_legend']:.1f} % of {x['frames']} frames under it"
                                            for x in lg.get("v41_place", {}).get("per_hold", [])) + ".",
          "The Jev overlay gate (python -m dancecap.film_jev overlays) runs on the v4.1 recordings: the v3 vertical film had 87 of 184 frames flagged, 80 "
          "for the legend. The 3D views' framing under the legend is Unity's (vertical_band); python cannot measure it.", ""]
    go = res.get("ghost") or {}
    if go:
        L += ["## The faint frustums against the dancers (estimate, without the keep-out)", "",
              f"Over {go['3d_seconds_checked']} s of 3D (10 Hz): " + ", ".join(f"phone {c} {v['over_couple_s']} s over the couple / {v['near_lens_s']} s near the lens"
                                                                          for c, v in go["per_phone"].items()) + f"; {go['over_couple_s_any']} phone-seconds in all. Unchanged block "
              "(opacity 0.15, outline, keep-out, near-lens fade).", ""]
    q = res["physics_quant"]
    L += ["## Physics and pivots", "",
          f"Load colours on screen {q['after']['load_colours_s']} s, the dancers' COM floor dots {q['after']['floor_dots_s']} s, counterbalance layer "
          f"{q['after']['counterbalance_s']} s, pivot segments {q['after']['pivot_s']} s. The 6 pivots ride inside the slow orbits; pivot 6 lies inside "
          "the crane up to the fourth overhead (as in v4.0: the overhead's core starts 0.09 s after the pivot ends) and pivot 2 under the end of the "
          "first crane down.", ""]
    L += ["## Checks", ""] + ([f"- PROBLEM: {x}" for x in ch["bad"]] or ["- film_recap.validate + the flow checks + the v4.1 targets: no problems"]) + [f"- {x}" for x in ch["notes"]]
    L += ["", "## The phones' clips the v4.1 direction needs (the Unity stage decodes them: `clips --version v4`)", ""]
    L += _md_table(["phone", "needed (s)", "missing in the capture now (s)"], [[cam, c["need_s"], c["missing_s"]] for cam, c in sorted(res["clips"].items())])
    return "\n".join(L) + "\n"


def contract_v41_md(res: dict) -> str:
    """the v4.1 section that heads UNITY_CONTRACT_v4.md"""
    doc, cfg4, met = res["doc"], res["cfg4"], res["metrics"]
    lg = cfg4.legend
    x0, x1, y0, y1 = legend_rect(lg)
    h = met["phone_holds"]
    L = ["# Unity contract: the v4 class-recap direction (`hm film flow v2`, revision 4.1)", "",
         "Source: `dancecap/work/larissa_kadu_recap/v4/direction/direction.json` (v4.1: `flow.revision` \"4.1\"; v4.0 stays in `v4/direction/v4.0/`). Built by "
         f"`python -m dancecap.film_recap_flow build --version v4.1 --plan {cfg4.plan}`, checked by `python -m dancecap.film_recap_flow check --version v4` "
         "(exits 1 when one of the owner's targets is missed - see review_v4.1.md for which and why). Write no names into head-movement: say lead / "
         "follow / \"the recap capture\".", "",
         "## v4.1: what changes for the Unity director (on top of v4.0's items 1-9 below)", "",
         "| # | what | where | v4.0 |", "|---|---|---|---|",
         "| A | cranes follow the cruise law: elevation = el_ground + (89.9 - el_ground) x cruise(u; ease 0.25); distance and the lens by the same law | "
         "`transition_in.v4.profile.elevation = {window, law: \"cruise\", ease}`, `flow.defaults.v4.crane_law` | smootherstep ramp |",
         "| B | the physics legend in the 9:16 top band | `presets.vertical.legend` (below) | Unity's own place, 20.3-27.4 % down |",
         "| C | in 9:16 the slow orbit frames the couple below the legend | slow-orbit block `vertical_band` {fit_band, top_m, band_lo, band_hi} | none (CoupleFov + ScreenYFor) |",
         "| D | the film ends on a phone hold (its last way is a phone_in; no way out) | the last shot | ended on an orbit |",
         "| E | the caps the live camera keeps: turn about the couple 12 deg/s, view rotation 12 deg/s, eye speed 1.2 m/s, crane pitch 20 deg/s; orbits "
         f"{cfg4.orbit_rate_deg_s[0]:g}-{cfg4.orbit_rate_deg_s[1]:g} deg/s, never turning back | `flow.defaults.v4.caps`, the orbit blocks' drift | 30 / 30 / 2.4 / 30; 4-10 deg/s |",
         "| F | the phone holds' crops are framed for the legend (teachers' box centred 46 % down, 45 % high) | `camera.crop` (already made), `camera.v4.crop_framing` | 40 % / 50 % |",
         "| G | longer ways: phone in / out 4-7 s, cranes 5.1 s, the replay's recentring 4 s; a crane down may leave its overhead up to 3 s early (never before its 2.2-s hold) "
         "when a phone hold follows | `transition_in.start_s / duration_s` (placed: do not re-place) | 2.5-5 s, 4.83 s, 2.5 s |", "",
         "**A. The cruise law** (python `dancecap.film_recap_flow.cruise`): x in [0, 1], e = ease = 0.25, S = smootherstep, I(y) = y^4 (2.5 - 3 y + y^2) "
         "(the integral of S from 0 to y):", "",
         "    x <= e        : s = e I(x / e)", "    e < x <= 1 - e: s = e / 2 + (x - e)", "    x > 1 - e     : y = (x - 1 + e) / e;  s = e / 2 + (1 - 2 e) + e (y - I(y))",
         "    cruise(x) = s / (1 - e)", "",
         "Its rate rises by a smootherstep, holds at 1 / (1 - e) = 1.33 x the mean, and falls by a smootherstep: value, 1st and 2nd derivative are 0 / 1 / 0 "
         "at both ends (C2 against the live orbit, as every v4 ramp). With 5.1-s cranes the pitch peaks at 74.3 / 0.75 / 5.1 = 19.4 deg/s. Implement it "
         "where the profile entry is an object with `law: \"cruise\"`; a list [a, b] stays a smootherstep ramp.", "",
         "**B. The physics legend in 9:16** (`presets.vertical.legend`; FilmOverlay.Tick, SkeletonLegend):", "",
         f"- Place: top = `presets.vertical.safe.top_frac` x H + 8 x W / 1080 = {lg['top_px_at_1080w']} px at 1080 x 1920 ({100 * y0:.2f} % of the height); "
         f"right edge {lg['right_px_at_1080w']:g} px from the right; scale {lg['scale']} (the 300 x 74 px box -> {300 * lg['scale']:.0f} x {74 * lg['scale']:.0f} px): the rectangle "
         f"x {100 * x0:.1f}-{100 * x1:.1f} %, y {100 * y0:.2f}-{100 * y1:.2f} % of the frame. Set SkeletonLegend.TopOverride to that in 9:16 and do NOT push it "
         "below the HUD row or the chapter-card slot any more (FilmOverlay.cs lines with `cardSlotBottom`, and `Mathf.Max(hud.BottomFromTop + 10f * unit, ...)`): "
         "it is a constant place, it never moves.",
         "- Cards: while a chapter card or the title shows, the legend's alpha = 1 - max(card alpha, title alpha) (it fades under them; the card's pill sits "
         "at the same height, centred, up to 92 % wide). The HUD row (move caption, graph inset) is off in this film (`moves` / `graph_inset` false on "
         "every segment); if it ever shows in 9:16 it goes under the legend.",
         "- Labels: the legend's rectangle is an obstacle for the call-out labels (`director.Annotations.Obstacles`) and, while it shows, the 9:16 "
         "LabelCeiling is at most its bottom + 10 x unit; the phone POV pill keeps its place and the legend never covers it (if they meet, the pill "
         "moves under the legend).",
         "- The face sidecar: the legend's rectangle (`SkeletonLegend.Area`) stays in the sidecar's overlay list as now.",
         f"- Phone holds: the python crops already put the teachers' box under it (measured on the plan: {100 * (1 - met['legend']['v41_place']['under_legend_share']):.1f} % of the "
         f"hold frames clear, box top median {met['legend']['v41_place']['box_top_median']} of the height; v4.0's crops with Unity's present place: "
         f"{100 * (res.get('v40') or {}).get('legend', {}).get('unity_v3_place', {}).get('under_legend_share', float('nan')):.1f} % of the frames under it).",
         "- 16:9: unchanged (top right inside the title-safe margins, legend_scale 1.5).", "",
         f"**C. The slow orbit in 9:16**: the block carries `vertical_band` = {json.dumps(V41_ORBIT_BAND)}. In 9:16 (only), FilmCamera.Orbit applies "
         "FitBand(ref p, couple floor point, top_m, band_lo, band_hi, 24, 66) after FitSpan, exactly as physics_3q does with `fit_band` (band in frame "
         "heights from the bottom: the couple's 0..1.95 m between the caption block and the legend). A lens / look-point change, damped like FitSpan: "
         "never an eye jump.", "",
         f"**D. The last shot** is phone {h['list'][-1]['phone'] if h['list'] else '-'}'s hold to the film's end (no way out); the recorder stops on its video.", "",
         "**E. The checks** (`hm_film_flow --action check`) use the v4.1 caps: turn about the couple <= 12 deg/s and view rotation <= 12 deg/s and eye "
         "speed <= 1.2 m/s in every frame of a ground-level way, crane pitch <= 20 deg/s, every settled orbit frame turning "
         f"{cfg4.orbit_rate_deg_s[0]:g}-{cfg4.orbit_rate_deg_s[1]:g} deg/s about the couple in its own direction, and no frame of the orbit's half of a way turning "
         "against it by more than 1 deg/s.", "",
         "Everything below (v4.0's contract, with v4.1's numbers) still holds.", ""]
    return "\n".join(L) + "\n"


def free_stretches(prot: list[dict], F: float, need: NeedSeries | None, way_s: float = 4.0, hold_min: float = 20.0) -> list[tuple]:
    """the film stretches outside every protected window, and whether a hold of hold_min with a way_s way at each end fits"""
    iv = sorted((max(0.0, p["f"][0]), min(F, p["f"][1])) for p in prot)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, t = [], 0.0
    for a, b in merged + [[F, F]]:
        if a - t > 3.0:
            end_ = a >= F - 1e-6
            fits = (a - t) - way_s * (1 if end_ else 2) >= hold_min
            nm = need.mean(need.mask([(t, a)]))[0] if need is not None else None
            out.append((t, a, fits, nm))
        t = max(t, b)
    return out


def write_all_v41(res: dict, out: Path) -> None:
    doc, tl = res["doc"], res["tl"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "narration").mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    (out / "direction.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    caps.write_timeline(out / "narration" / "timeline.json", tl)
    ch = []
    for i, c in enumerate(doc["chapters"]):
        ch.append({"title": c["title"], "startS": c["film_s"], "endS": doc["chapters"][i + 1]["film_s"] if i + 1 < len(doc["chapters"]) else doc["timeline"]["film_duration_s"],
                   "segment": c["segment"]})
    (out / "chapters.json").write_text(json.dumps({"schema": "dancecap.film_recap/1/chapters", "film_duration_s": doc["timeline"]["film_duration_s"], "chapters": ch},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")
    res["ghost"] = ghost_overlap(doc, res["metrics"], res["geo4"], res["cfg4"])
    cols = [("v3.2", res["baseline"])] + ([("v4.0", res["v40"])] if res.get("v40") else []) + [(f"v4.1 ({res['cfg4'].plan}-first)", res["metrics"])]
    if res.get("alt"):
        cols.append((res["alt_info"]["label"], res["alt"]))
    res["columns"] = cols
    res["ba_rows"] = before_after_rows_n(cols, res["targets"])
    ba = {"schema": "dancecap.film_recap_flow/v4.1/before_after", "columns": [lab for lab, _ in cols], "rows": res["ba_rows"],
          "metrics": {lab: _metrics_public(m) for lab, m in cols}, "targets": res["targets"], "check_targets": res["checks"]["targets"],
          "plan_search": res["dp"].get("search"), "free_stretches": res.get("free_stretches"), "overheads_noul": res.get("overheads_noul")}
    (out / "before_after.json").write_text(json.dumps(ba, indent=1, default=str), encoding="utf-8")
    rep = {"schema": "dancecap.film_recap_flow/2.1/report", "checks": res["checks"], "targets": res["targets"], "metrics": _metrics_public(res["metrics"]),
           "baseline": _metrics_public(res["baseline"]), "runs": [{k: v for k, v in r.items() if k != "q"} | {"quality": r.get("q")} for r in res["runs"]],
           "ways": res["sched"]["ways"], "orbits": res["sched"]["plans"], "dp": res["dp"], "schedule_log": res["plog"], "ghost_overlap": res["ghost"],
           "physics_on_screen": res["physics_quant"], "clips": res["clips"], "pivots": res["pivots"], "replay": res["replay"], "t_under_arrows": res["t_notes"],
           "connected_source": res["con_src"], "protected": res["prot"], "overheads_noul": res.get("overheads_noul"), "free_stretches": res.get("free_stretches"),
           "physics": [{k: r[k] for k in ("id", "film", "mode", "on", "why")} for r in res["phys_rows"]]}
    (out / "direction_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "clips.json").write_text(json.dumps({"what": "per phone: the master-second ranges of video the v4.1 direction needs (phone holds + frustum ways in / out, padded 0.5 s)",
                                                "phones": res["clips"]}, indent=1), encoding="utf-8")
    (out / "review_v4.1.md").write_text(review_v41_md(res), encoding="utf-8")
    (out / "UNITY_CONTRACT_v4.md").write_text(contract_v41_md(res) + "\n" + contract_md(res).split("\n", 1)[1], encoding="utf-8")
    (out / "audio" / "make_film_mix.py").write_text(MIX_SCRIPT.replace("the v3 direction", "the v4.1 direction"), encoding="utf-8")
    m, c4 = res["metrics"], res["cfg4"]
    sheet_v4(m, doc, res["geo4"], out / "flow_sheet.png",
             f"v4.1 camera flow ({c4.plan}-first): {m['moves']['total']} moves, {m['phone_holds']['n']} phone holds (mean {m['phone_holds']['mean_s']} s, "
             f"{100 * m['phone_holds']['share']:.1f} % of the film), slow orbits {', '.join(str(x) for x in m['three_d']['drift_rates_deg_s'])} deg/s, no cut",
             band=c4.orbit_rate_deg_s, cap=c4.cap_az_deg_s, label="v4.1")
    compare_plot_n(cols, out / "before_after.png", res.get("need"), res["targets"],
                   "class-recap camera: v3.2 (the 2026-10-21 finals) / v4.0 / v4.1 (owner 2026-10-22: hold longer on POVs, slowly orbit, Jev)")


# ------------------------------------------------------------------------------------------------ v4.2: the physics meter's windows, closer orbits


OUT_V41 = OUT_V4 / "v4.1"
V41_DIRECTION = OUT_V41 / "direction.json"
V42_OWNER = ("owner 2026-10-22, after v4.1: \"for kadu larissa recap video output: just show the physics meter at the very top of the frame for a few "
             "seconds every time we switch physics mode on and then fade it out. I like the camera motions, but in gauss splat mode let's get 20% closer "
             "to the dancers when we orbit. on the very final pass let's apply the face fusion to the gauss splats only. no need to re-review videos "
             "dozens of times. just plan the motion and render once and I'll review\"")
V42_CLOSER = 0.8          # every slow orbit (and the pivots / physics views riding on it) at 0.8 x its v4.1 eye offset about the couple
# the physics meter ("Skeleton load - ESTIMATED from motion", tension / neutral / compression): per switch-on of the physics view a window that fades
# in over fade_in_s, shows until hold_s after its start and fades out over fade_out_s (smootherstep both ways); a window that would meet a chapter
# card or the title starts card_gap_s after it instead (the card's pill sits in the same top band)
V42_METER = {"schema": "hm film meter v1", "revision": "4.2", "fade_in_s": 0.3, "hold_s": 4.0, "fade_out_s": 1.0, "curve": "smootherstep", "card_gap_s": 0.2}
# where it shows (Unity FilmOverlay.Tick / SkeletonLegend: a 300 x 74 px box scaled from its top-right corner)
V42_METER_PLACE = {
    "vertical": {"anchor": "top_right", "top_px_at_1080w": 276.8, "right_px_at_1080w": 14.0, "box_px": [300, 74], "scale": 1.65,
                 "rule": "top = presets.vertical.safe.top_frac x H + 8 x W / 1080 (the very top of the 9:16 safe area, above the heads), right edge 14 px in, 1.65 x"},
    "horizontal": {"anchor": "top_right", "top_px_at_1080h": 62.0, "right_px_at_1920w": 96.0, "box_px": [300, 74], "scale": 1.5,
                   "rule": "top = presets.horizontal.safe.top_frac x H + 8 x H / 1080, right edge at the title-safe side margin (safe.side_frac x W), legend_scale 1.5"}}
FLOW_SCHEMA_V42 = ("; v4.2 (flow.revision '4.2', UNITY_CONTRACT_v4.md 'v4.2'): flow.meter {windows: [{on_s, start_s, full_s, fade_s, end_s}], place} - the physics "
                   "meter shows only in its windows; the slow-orbit block's lens_distance_scale (the lens chosen for eye distance x this) and camera.v4.closer")


def legend_rect_h(spec: dict, W: float = 1920.0, H: float = 1080.0) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of the meter in fractions of the 16:9 frame (y down)"""
    s = float(spec.get("scale", 1.0))
    w, h = (float(v) * s for v in spec.get("box_px", [300, 74]))
    x1 = W - float(spec.get("right_px_at_1920w", 14.0)) * W / 1920.0
    y0 = float(spec["top_px_at_1080h"]) * H / 1080.0
    return (x1 - w) / W, x1 / W, y0 / H, (y0 + h) / H


def physics_on(seg: dict, layers: dict | None = None) -> bool:
    """the director's physics view (Unity FilmDirector.SegLayers.Physics): skeleton_mode "physics", or the layer flag `physics` (default: the
    segment's view_state is "physics")"""
    L = (seg.get("layers") or {}) if layers is None else layers
    return L.get("skeleton_mode") == "physics" or bool(L.get("physics", seg.get("view_state") == "physics"))


def layer_starts(doc: dict) -> list[float]:
    """film time from which each segment's layers apply (Unity FlowTimeline.LayerStart): its F0, or for the first segment of a shot with a way in
    F0 + transition_in.layers_switch_s"""
    segs = doc["segments"]
    out = [float(s["film"][0]) for s in segs]
    for sh in shots_of(segs):
        tr = segs[sh.segs[0]]["camera"].get("transition_in")
        if tr:
            out[sh.segs[0]] = float(sh.f0) + float(tr.get("layers_switch_s", 0.0) or 0.0)
    return out


def physics_timeline(doc: dict) -> list[dict]:
    """the physics view over the film as the director switches it: pieces {a, b, on, seg} covering [0, F] - each segment's layers from its layer
    start (the entry in force is the latest start), a phone-to-phone way's transit_layers over their own window - one piece per state run"""
    segs = doc["segments"]
    F = float(doc["timeline"]["film_duration_s"])
    st = layer_starts(doc)
    ev = sorted((st[i], i) for i in range(len(segs)))
    cuts = {0.0, F}
    for a, _ in ev:
        cuts.add(min(F, max(0.0, a)))
    transits = []
    for s in segs:
        tl = (s["camera"].get("transition_in") or {}).get("transit_layers")
        if tl and tl.get("film"):
            a, b = (float(x) for x in tl["film"])
            transits.append((a, b, physics_on({}, tl.get("layers") or {}), s["id"]))
            cuts.update({a, b})
    ts = sorted(t for t in cuts if 0.0 <= t <= F)
    pieces: list[dict] = []
    for a, b in zip(ts[:-1], ts[1:]):
        if b - a < 1e-6:
            continue
        m = 0.5 * (a + b)
        cur = [i for t0, i in ev if t0 <= m]
        i = cur[-1] if cur else 0
        on, sid = physics_on(segs[i]), segs[i]["id"]
        for ta, tb, ton, tid in transits:
            if ta <= m < tb:
                on, sid = ton, tid + " (transit)"
        if pieces and pieces[-1]["on"] == on and abs(pieces[-1]["b"] - a) < 1e-6:
            pieces[-1]["b"] = b
        else:
            pieces.append({"a": a, "b": b, "on": on, "seg": sid})
    return pieces


def card_windows(doc: dict) -> list[tuple[float, float, str]]:
    """the film windows of the chapter cards (Unity ChapterCards: F0 + film_in .. min(F1, F0 + film_out)) and of a segment's title card"""
    out = []
    for s in doc["segments"]:
        f0, f1 = (float(x) for x in s["film"])
        cc = s.get("chapter_card")
        if cc:
            a, b = f0 + float(cc.get("film_in", 0.3)), min(f1, f0 + float(cc.get("film_out", 2.9)))
            if b - a >= 0.3:
                out.append((a, b, f"{cc.get('kind', 'chapter')} card '{cc.get('title', '')}' ({s['id']})"))
        t = s.get("title")
        if isinstance(t, dict):
            a = f0 + float(t.get("film_in", 0.0))
            b = min(f1, f0 + float(t["film_out"])) if t.get("film_out") is not None else f1
            out.append((a, b, f"title ({s['id']})"))
    return sorted(out)


def meter_windows(doc: dict, spec: dict | None = None) -> dict:
    """flow.meter: one window per switch-on (off -> on) of the physics view, from the layer timeline. A window starts at the switch-on (or, when it
    would meet a chapter card or the title, card_gap_s after that card), fades in over fade_in_s, holds until hold_s after its start and fades out
    over fade_out_s; it never outlasts the physics view (a view that ends earlier pulls the fade-out in) and is dropped when the physics view
    leaves it less than fade_in_s + 1 s + fade_out_s / 2 (a meter flashing up for a moment)"""
    sp = dict(V42_METER, **(spec or {}))
    fi, hold, fo, gap = (float(sp[k]) for k in ("fade_in_s", "hold_s", "fade_out_s", "card_gap_s"))
    pieces = physics_timeline(doc)
    cards = card_windows(doc)
    wins, notes = [], []
    prev = False
    for p in pieces:
        if p["on"] and not prev:
            t_on, t_off = p["a"], p["b"]
            s, after = t_on, []
            moved = True
            while moved:
                moved = False
                for a, b, what in cards:
                    if s < b + gap - 1e-9 and s + hold + fo > a - gap + 1e-9:
                        s, moved = b + gap, True
                        after.append(what)
            e = min(s + hold + fo, t_off)
            if e - s < fi + 1.0 + 0.5 * fo - 1e-9:
                notes.append(f"switch-on at {t_on:.2f} s ({p['seg']}): no window - the physics view ends at {t_off:.2f} s, before the meter could show"
                             + (f" after the {after[-1]}" if after else ""))
            else:
                fade = s + hold if e >= s + hold + fo - 1e-9 else max(s + fi, e - fo)
                wins.append({"on_s": round(t_on, 3), "start_s": round(s, 3), "full_s": round(s + fi, 3), "fade_s": round(fade, 3), "end_s": round(e, 3),
                             "delay_s": round(s - t_on, 3), "segment": p["seg"], "physics_until_s": round(t_off, 3), "after": after})
        prev = p["on"]
    return {"schema": sp["schema"], "revision": sp["revision"],
            "what": "the physics meter: the skeleton-load legend ('Skeleton load - ESTIMATED from motion', tension / neutral / compression)",
            "rule": "it shows only inside these windows (and only while the physics view is on): alpha = smootherstep up over [start_s, full_s], 1 until "
                    "fade_s, smootherstep down to 0 at end_s; one window per switch-on of the physics view; a window that would meet a chapter card or "
                    "the title starts card_gap_s after it",
            "fade_in_s": fi, "hold_s": hold, "fade_out_s": fo, "curve": sp["curve"], "card_gap_s": gap, "place": copy.deepcopy(V42_METER_PLACE),
            "windows": wins, "switch_ons": sum(1 for i, p in enumerate(pieces) if p["on"] and (i == 0 or not pieces[i - 1]["on"])), "dropped": len(notes),
            "notes": notes}


def meter_alpha(meter: dict, t: float) -> float:
    """the meter's opacity at film t (the python twin of Unity FlowTimeline.MeterAlpha)"""
    a = 0.0
    for w in (meter or {}).get("windows", []):
        if not (w["start_s"] <= t <= w["end_s"]):
            continue
        up = smootherstep((t - w["start_s"]) / max(1e-6, w["full_s"] - w["start_s"])) if t < w["full_s"] else 1.0
        dn = 1.0 - smootherstep((t - w["fade_s"]) / max(1e-6, w["end_s"] - w["fade_s"])) if t > w["fade_s"] else 1.0
        a = max(a, min(up, dn))
    return a


V42_CLOSER_EASE_S = 6.0   # next to a phone way the closer orbit eases from v4.1's distance (at the way) to 0.8 x over this long (smootherstep)
# the closer orbit's framing bands (fractions of the frame height from the bottom; FitBand: the lens only widens when the couple's floor .. top_m
# would not fit, else the composition is centred in the band). v4.1's 9:16 band (0.31-0.77) kept the couple under the always-on legend; with the
# dancers 1.25 x larger it put their feet onto the caption block (editor stills at 60 s and 274.6 s), and the legend now shows only in the meter's
# windows - so 9:16 moves up to 0.34-0.84, and 16:9 gets the same rule (it had none: the feet reached into the caption line at 60 s; the near foot sits about 6 % of the height below the couple centre's floor point the band fits, hence 0.24)
V42_BAND_V = {"fit_band": True, "top_m": 1.95, "band_lo": 0.34, "band_hi": 0.84}
V42_BAND_H = {"fit_band": True, "top_m": 1.9, "band_lo": 0.24, "band_hi": 0.99}


def closer_weight(cam: dict, f: float) -> float:
    """v4.2: how close the slow orbit is at film f, 0 (v4.1's distance) .. 1 (the block's, 0.8 x): a smootherstep up over camera.closer_in_s after a
    phone way out, down over camera.closer_out_s before a phone way in; 1 without them"""
    w = 1.0
    a = cam.get("closer_in_s")
    if a:
        w *= ramp(f, float(a[0]), float(a[1])) if float(a[1]) > float(a[0]) else (1.0 if f >= float(a[0]) else 0.0)
    b = cam.get("closer_out_s")
    if b:
        w *= 1.0 - (ramp(f, float(b[0]), float(b[1])) if float(b[1]) > float(b[0]) else (1.0 if f >= float(b[1]) else 0.0))
    return w


def closer_factor(cam: dict, f: float) -> float:
    """v4.2: the factor on the block's eye offset about the look point at film f: 1 at full closeness, 1 / k (v4.1's distance) at a phone way"""
    k = float(((cam.get("v4") or {}).get("closer") or {}).get("scale") or 1.0)
    if k >= 1.0 - 1e-9:
        return 1.0
    return 1.0 + (1.0 / k - 1.0) * (1.0 - closer_weight(cam, f))


def closer_eases(doc: dict, ease_s: float = V42_CLOSER_EASE_S) -> list[dict]:
    """v4.2: the closer orbit's ease windows. A phone way's 3D end is the orbit at v4.1's distance (the way is v4.1's way: its peaks under the
    caps); inside the orbit the distance eases to 0.8 x over ease_s after a phone way out (closer_in_s) and back before a phone way in
    (closer_out_s). The window may reach into a crane / replay way next to it (the crane reads the live orbit); between two phone ways of one
    orbit part each takes at most half the gap. Writes the keys into every segment of the part; returns one row per window"""
    segs = doc["segments"]
    shots = shots_of(segs)
    rows = []
    for k, sh in enumerate(shots):
        cam = segs[sh.segs[0]]["camera"]
        if (cam.get("v4") or {}).get("role") != "slow_orbit":
            continue
        tr = cam.get("transition_in") or {}
        ntr = (segs[shots[k + 1].segs[0]]["camera"].get("transition_in") or {}) if k + 1 < len(shots) else {}
        w_end = sh.f0 + float(tr["start_s"]) + float(tr["duration_s"]) if tr.get("type") == "phone_out" else None
        w_start = shots[k + 1].f0 + float(ntr["start_s"]) if ntr.get("type") == "phone_in" else None
        lo = sh.f0 + float(tr["start_s"]) if tr and tr.get("type") != "phone_out" else sh.f0
        hi = shots[k + 1].f0 + float(ntr["start_s"]) + float(ntr["duration_s"]) if ntr and ntr.get("type") != "phone_in" else sh.f1
        keys = {}
        if w_end is not None:
            avail = (w_start - w_end) / 2.0 if w_start is not None else hi - w_end
            T = max(0.0, min(ease_s, avail))
            keys["closer_in_s"] = [round(w_end, 3), round(w_end + T, 3)]
            rows.append({"segment": segs[sh.segs[0]]["id"], "side": "after the phone way out", "film": keys["closer_in_s"], "T": round(T, 2)})
        if w_start is not None:
            avail = (w_start - w_end) / 2.0 if w_end is not None else w_start - max(0.0, lo)
            T = max(0.0, min(ease_s, avail))
            keys["closer_out_s"] = [round(w_start - T, 3), round(w_start, 3)]
            rows.append({"segment": segs[sh.segs[0]]["id"], "side": "before the phone way in", "film": keys["closer_out_s"], "T": round(T, 2)})
        for i in sh.segs:
            segs[i]["camera"].update(copy.deepcopy(keys))
    return rows


def closer_block(cam: dict, k: float = V42_CLOSER, clear_m: float = 1.4, ground: tuple = (0.5, 2.4)) -> tuple[dict, dict]:
    """a slow-orbit camera block k x closer about the couple: both radii (9:16 radius_vertical_m, 16:9 radius_m) and the eye's height above the
    look point by k (the same look-down angle: the eye moves straight in along its line of sight), the inline lift's cap by k, never inside the
    couple's clear radius (a radius that k would put inside it stops at the clear radius - reported). `lens_distance_scale` = 1 / k: Unity chooses
    the lens (CoupleFov) for the v4.1 distance, so the dancers fill 1 / k more of the frame instead of the lens widening the closeness away.
    Returns (the block, what changed)"""
    c = copy.deepcopy(cam)
    lh = float(c.get("look_height_m", 0.95))
    h = float(c["height_m"])
    lift = float(c.get("inline_lift_max_m", 0.25))
    info = {"scale": k, "v41": {key: c.get(key) for key in ("radius_m", "radius_vertical_m", "height_m", "inline_lift_max_m")}, "clamped": []}
    for key in ("radius_m", "radius_vertical_m"):
        if c.get(key) is None:
            continue
        r = float(c[key])
        rn = k * r
        if rn < clear_m:
            info["clamped"].append(f"{key} {r:g} m x {k:g} = {rn:.2f} m is inside the clear radius: {clear_m:g} m")
            rn = clear_m
        c[key] = round(rn, 3)
    c["height_m"] = round(lh + k * (h - lh), 3)
    c["inline_lift_max_m"] = round(k * lift, 3)
    c["lens_distance_scale"] = round(1.0 / k, 4)
    if c.get("vertical_band"):
        info["v41"]["vertical_band"] = copy.deepcopy(c["vertical_band"])
        c["vertical_band"] = dict(V42_BAND_V)
    c["horizontal_band"] = dict(V42_BAND_H)
    if not (ground[0] - 1e-9 <= c["height_m"] and c["height_m"] + c["inline_lift_max_m"] <= ground[1] + 1e-9):
        info["clamped"].append(f"eye {c['height_m']} + lift {c['inline_lift_max_m']} m outside the ground band {ground}")
    for key, rk in (("vertical", "radius_vertical_m"), ("horizontal", "radius_m")):
        if c.get(rk) is not None and info["v41"].get(rk) is not None:
            info[f"eye_distance_{key}_m"] = {"v41": round(math.hypot(float(info["v41"][rk]), h - lh), 3),
                                             "v42": round(math.hypot(float(c[rk]), c["height_m"] - lh), 3)}
    c.setdefault("v4", {})["closer"] = {"scale": k, "v41": info["v41"], "lens": "CoupleFov(eye distance x lens_distance_scale / closer factor): the v4.1 lens, "
                                                                                 "the dancers 1/k larger at full closeness"}
    return c, info


def apply_closer(doc: dict, k: float = V42_CLOSER) -> dict:
    """every slow-orbit block of the direction (camera.v4.role slow_orbit: the orbits, the pivots, the physics views and the replay riding on them)
    k x closer (closer_block), the v1 from / to estimates of the ways' 3D ends with them, flow.defaults' orbit block; the overheads and their cranes'
    top end stay (the crane's ground end is the live orbit). Returns {scale, runs: per run v4.1 / v4.2, clamped: [...]}"""
    flow = doc.get("flow") or {}
    dflt = flow.get("defaults") or {}
    clear = float(dflt.get("clear_radius_m", 1.4))
    gm = tuple(dflt.get("ground_m", (0.5, 2.4)))
    runs: dict = {}
    clamped = []

    def scale_end(e: dict, lh: float) -> None:
        if e and e.get("kind") == "3d" and e.get("radius_m") is not None and not e.get("closer"):
            e["radius_m"] = round(max(clear, k * float(e["radius_m"])), 2)
            e["height_m"] = round(lh + k * (float(e["height_m"]) - lh), 2)
            e["closer"] = k

    for s in doc["segments"]:
        cam = s["camera"]
        if (cam.get("v4") or {}).get("role") != "slow_orbit":
            continue
        nc, info = closer_block(cam, k, clear, gm)
        if info["clamped"]:
            clamped.append({"segment": s["id"], "why": info["clamped"]})
        run = (cam.get("v4") or {}).get("run")
        if run not in runs:
            runs[run] = {"run": run, "first_segment": s["id"], "film": [s["film"][0], s["film"][1]], "drift_deg_s": cam.get("drift_deg_s"),
                         "v41": {"radius_vertical_m": cam.get("radius_vertical_m"), "radius_m": cam.get("radius_m"), "height_m": cam.get("height_m"),
                                 "inline_lift_max_m": cam.get("inline_lift_max_m")},
                         "v42": {"radius_vertical_m": nc.get("radius_vertical_m"), "radius_m": nc.get("radius_m"), "height_m": nc.get("height_m"),
                                 "inline_lift_max_m": nc.get("inline_lift_max_m"), "lens_distance_scale": nc.get("lens_distance_scale")},
                         "eye_distance_m": {"vertical": info.get("eye_distance_vertical_m"), "horizontal": info.get("eye_distance_horizontal_m")}, "segments": 0}
        runs[run]["film"][1] = s["film"][1]
        runs[run]["segments"] += 1
        s["camera"] = nc
    # the ways' v1 estimates of a 3D end (the v4 director blends the live states; these only document the plan): a crane's or the replay's 3D end
    # is the closer orbit; a phone way's is v4.1's (the orbit eases back to v4.1's distance at it: closer_eases)
    for s in doc["segments"]:
        tr = s["camera"].get("transition_in")
        if tr and tr.get("type") not in AXIS_TYPES:
            for end in ("from", "to"):
                scale_end(tr.get(end), 0.95)
    ob = ((dflt.get("v4") or {}).get("orbit") or {}).get("block")
    if ob:
        nb, _ = closer_block(ob, k, clear, gm)
        nb.pop("v4", None)
        dflt["v4"]["orbit"]["block"] = nb
        dflt["inline_lift_max_m"] = nb["inline_lift_max_m"]
    eases = closer_eases(doc)
    dflt.setdefault("v4", {})["closer"] = {"scale": k, "ease_s": V42_CLOSER_EASE_S, "rule": "the slow orbit at 0.8 x v4.1's eye offset about the look point; "
                                           "next to a phone way it eases (smootherstep, camera.closer_in_s / closer_out_s) to v4.1's distance at the way, "
                                           "so the phone ways keep v4.1's paths and peaks; the lens is CoupleFov(v4.1's distance) throughout"}
    return {"scale": k, "runs": [runs[r] for r in sorted(runs, key=lambda x: (x is None, x))], "clamped": clamped, "eases": eases}


def eye_stats(met: dict, doc: dict, geo4: Geo4) -> dict:
    """from a film_metrics sample: the eye's distance to the couple (its centre 1 m up) and its horizontal radius in the settled slow orbits, and
    the closest the eye comes to either dancer (centre of mass, horizontally) over all ground-level 3D time (orbits + their ways, not the cranes /
    overheads)"""
    sm = met.get("_sample")
    if sm is None or "eye" not in sm:
        return {}
    t, eye, c, kind, way = sm["t"], sm["eye"], sm["c"], sm["kind"], sm["way"]
    shots = shots_of(doc["segments"])
    trs = {k: doc["segments"][shots[k].segs[0]]["camera"]["transition_in"]["type"] for k in range(1, len(shots))}
    settled = (kind == "3d") & (way < 0) & ~sm["ph_hold"]
    ground_way = np.array([w >= 1 and trs.get(int(w)) not in ("crane_up", "crane_down") for w in way])
    ground3d = settled | ground_way
    dist = np.linalg.norm(eye - (c + np.array([0.0, 1.0, 0.0])), axis=1)
    rad = np.hypot(eye[:, 0] - c[:, 0], eye[:, 2] - c[:, 2])
    g = geo4.g
    dmin = np.full(len(t), np.inf)
    for i in np.nonzero(ground3d)[0][::3]:
        d = _dance_at_film(doc, float(t[i]))
        for arr in (g.lead, g.follow):
            px, pz = np.interp(d, g.t, arr[:, 0]), np.interp(d, g.t, arr[:, 2])
            dmin[i] = min(dmin[i], float(math.hypot(eye[i, 0] - px, eye[i, 2] - pz)))

    def q(xs, p):
        return round(float(np.percentile(xs, p)), 3) if len(xs) else None
    fin = np.isfinite(dmin) & ground3d
    return {"orbit_eye_distance_m": {"median": q(dist[settled], 50), "min": q(dist[settled], 0), "max": q(dist[settled], 100)},
            "orbit_radius_m": {"median": q(rad[settled], 50), "min": q(rad[settled], 0)},
            "ground_3d_radius_min_m": q(rad[ground3d], 0), "nearest_dancer_m": {"min": q(dmin[fin], 0), "p5": q(dmin[fin], 5), "median": q(dmin[fin], 50)},
            "frames": int(settled.sum())}


def v42_cfg() -> V4Cfg:
    c41 = v41_cfg()
    ob, _ = closer_block(c41.orbit_block)
    ob.pop("v4", None)
    r, h = c41.orbit_r_h
    return dataclasses_replace(c41, version="4.2", orbit_block=ob, orbit_r_h=(round(V42_CLOSER * r, 3), round(c41.orbit_look_h + V42_CLOSER * (h - c41.orbit_look_h), 3)))


def _peak_summary(m: dict) -> dict:
    w = m.get("ways") or {}
    return {"az": (w.get("peak_az_deg_s") or {}).get("max"), "rot": (w.get("peak_rot_deg_s") or {}).get("max"), "speed": (w.get("peak_speed_m_s") or {}).get("max"),
            "pitch": (w.get("crane_pitch_deg_s") or {}).get("max")}


def v42_extra_metrics(met: dict, doc: dict, geo4: Geo4 | None, cfg4: V4Cfg, pov, met_h: dict | None = None) -> None:
    """v4.2's numbers: the meter (windows against the physics view and the cards, the teachers' clearance in the holds it shows over), the eye's
    distance / clearance in 9:16 (met) and 16:9 (met_h), the 16:9 way peaks"""
    meter = (doc.get("flow") or {}).get("meter") or {}
    F = float(doc["timeline"]["film_duration_s"])
    ts = np.arange(0.0, F, 1.0 / 30.0)
    al = np.array([meter_alpha(meter, float(x)) for x in ts])
    on = np.zeros(len(ts), bool)
    for p in physics_timeline(doc):
        if p["on"]:
            on |= (ts >= p["a"]) & (ts < p["b"])
    card = np.zeros(len(ts), bool)
    for a, b, _ in card_windows(doc):
        card |= (ts >= a) & (ts <= b)
    wins = meter.get("windows") or []
    shapes = [{"start_s": w["start_s"], "fade_in_s": round(w["full_s"] - w["start_s"], 3), "full_until_s": round(w["fade_s"] - w["start_s"], 3),
               "fade_out_s": round(w["end_s"] - w["fade_s"], 3), "delay_s": w["delay_s"]} for w in wins]
    mm = {"switch_ons": meter.get("switch_ons"), "windows": len(wins), "shapes": shapes, "shown_s": round(float((al > 0.002).sum()) / 30.0, 2),
          "frames_shown_without_physics": int(((al > 0.002) & ~on).sum()), "frames_shown_over_a_card": int(((al > 0.002) & card).sum()),
          "physics_on_s": round(float(on.sum()) / 30.0, 1)}
    if pov is not None:
        mm["holds_clear"] = legend_overlap(doc, pov, legend_rect(V42_METER_PLACE["vertical"]), windows=[(w["start_s"], w["end_s"]) for w in wins])
    met["meter"] = mm
    sm = met.get("_sample")
    if sm is not None:
        segs = doc["segments"]
        f0s = np.array([float(x["film"][0]) for x in segs])
        settled = (sm["kind"] == "3d") & (sm["way"] < 0) & ~sm["ph_hold"]
        fac = np.array([closer_factor(segs[max(0, int(np.searchsorted(f0s, t, side="right")) - 1)]["camera"], float(t)) for t in sm["t"][settled]])
        dt = 1.0 / cfg4.sample_hz
        met["closer_ease"] = {"windows": len({(k, tuple(x["camera"][k])) for x in segs for k in ("closer_in_s", "closer_out_s") if x["camera"].get(k)}),
                              "settled_orbit_s": round(float(settled.sum()) * dt, 1), "full_closeness_s": round(float((fac <= 1.0 + 1e-6).sum()) * dt, 1),
                              "full_closeness_share": round(float((fac <= 1.0 + 1e-6).mean()), 3) if len(fac) else None,
                              "mean_factor": round(float(fac.mean()), 4) if len(fac) else None}
    if geo4 is not None:
        met["eye"] = eye_stats(met, doc, geo4)
        if met_h is not None:
            met["eye_16x9"] = eye_stats(met_h, doc, geo4)
    if met_h is not None:
        met["ways_16x9"] = _peak_summary(met_h)


def _targets_v42(target, doc: dict, met: dict, cfg4: V4Cfg, notes: list[str]) -> None:
    """the owner's v4.2 targets: the physics meter only for a few seconds at each switch-on, clear of the cards; the slow orbits 0.8 x closer in both
    aspects, never inside the clear radius, the eye in the ground band; the 16:9 ways under the v4.1 caps too"""
    mm = met.get("meter") or {}
    meter = (doc.get("flow") or {}).get("meter") or {}
    sp = V42_METER
    good = [s for s in mm.get("shapes", []) if abs(s["fade_in_s"] - sp["fade_in_s"]) < 2e-3 and abs(s["full_until_s"] - sp["hold_s"]) < 2e-3
            and abs(s["fade_out_s"] - sp["fade_out_s"]) < 2e-3]
    target("8. the physics meter: one window per switch-on of the physics view (fade in 0.3 s, shown 4 s, fade out 1 s), never while the physics view is off, "
           "never over a chapter card", f"{mm.get('switch_ons')} windows (or dropped: a physics view too short), 0 / 0 frames",
           f"{mm.get('windows')} windows ({len(good)} full-length) from {', '.join(str(w['start_s']) for w in meter.get('windows', []))} s, shown {mm.get('shown_s')} of "
           f"{mm.get('physics_on_s')} s of physics; frames without physics {mm.get('frames_shown_without_physics')}, over a card {mm.get('frames_shown_over_a_card')}",
           mm.get("windows", 0) >= 1 and mm.get("windows", 0) + int(meter.get("dropped", 0)) == mm.get("switch_ons") and mm.get("frames_shown_without_physics", 1) == 0
           and mm.get("frames_shown_over_a_card", 1) == 0)
    hc = mm.get("holds_clear") or {}
    if hc.get("frames"):
        target("8. the meter at the very top of the 9:16 frame: the teachers' box clear of it in the phone-hold frames it shows over",
               ">= 95 %", f"{100 * (1 - hc['under_legend_share']):.1f} % clear ({hc['frames']} frames)", 1 - hc["under_legend_share"] >= 0.95 - 1e-9)
    for w in meter.get("windows", []):
        if w["delay_s"] > 0:
            notes.append(f"the meter of the switch-on at {w['on_s']} s ({w['segment']}) starts {w['delay_s']} s later, after the {w['after'][-1]}")
    for n in meter.get("notes", []):
        notes.append("meter: " + n)
    # 9. closer orbits
    ratios, bad_r = [], []
    clear = float(((doc.get("flow") or {}).get("defaults") or {}).get("clear_radius_m", 1.4))
    for s in doc["segments"]:
        cam = s["camera"]
        if (cam.get("v4") or {}).get("role") != "slow_orbit":
            continue
        v41 = ((cam.get("v4") or {}).get("closer") or {}).get("v41") or {}
        for key in ("radius_vertical_m", "radius_m"):
            r41, r42 = v41.get(key), cam.get(key)
            if r41 is None or r42 is None:
                bad_r.append(f"{s['id']} {key}: no v4.1 record")
                continue
            want = max(clear, V42_CLOSER * float(r41))
            ratios.append(float(r42) / float(r41))
            if abs(float(r42) - want) > 2e-3 or float(r42) < clear - 1e-9:
                bad_r.append(f"{s['id']} {key} {r42} (want {want:.3f})")
        if not (0.5 <= float(cam["height_m"]) and float(cam["height_m"]) + float(cam.get("inline_lift_max_m", 0.0)) <= 2.4 + 1e-9):
            bad_r.append(f"{s['id']}: eye {cam['height_m']} + lift {cam.get('inline_lift_max_m')} m outside 0.5-2.4 m")
    ey, eh = met.get("eye") or {}, met.get("eye_16x9") or {}
    gv, gh = ey.get("ground_3d_radius_min_m"), eh.get("ground_3d_radius_min_m")
    target("9. every slow orbit (pivots, physics views, the replay on it) 0.8 x its v4.1 radius about the couple, both aspects; never inside the clear "
           "radius; eye in the ground band", f"0.8 x; >= {clear:g} m; 0.5-2.4 m",
           (f"{len(ratios)} radii, ratio {min(ratios):.3f}-{max(ratios):.3f}" if ratios else "no slow orbit") + f"; settled orbit radius median 9:16 "
           f"{(ey.get('orbit_radius_m') or {}).get('median')} m, 16:9 {(eh.get('orbit_radius_m') or {}).get('median')} m; closest ground-level 3D radius "
           f"{gv} / {gh} m" + (f"; problems {bad_r[:4]}" if bad_r else ""),
           bool(ratios) and not bad_r and (gv is None or gv >= clear - 1e-6) and (gh is None or gh >= clear - 1e-6))
    w16 = met.get("ways_16x9") or {}
    if w16.get("az") is not None:
        tg = v41_targets(cfg4)
        target("9. 16:9 ways under the v4.1 caps (turn about the couple / view rotation / eye speed / crane pitch)",
               f"<= {tg['cap_az_deg_s']} / {tg['cap_rot_deg_s']} deg/s / {tg['cap_speed_m_s']} m/s / {tg['cap_crane_deg_s']} deg/s",
               f"{w16.get('az')} / {w16.get('rot')} / {w16.get('speed')} / {w16.get('pitch')}",
               w16["az"] <= tg["cap_az_deg_s"] + 0.05 and (w16.get("rot") or 0) <= tg["cap_rot_deg_s"] + 0.05
               and (w16.get("speed") or 0) <= tg["cap_speed_m_s"] + 0.01 and (w16.get("pitch") or 0) <= tg["cap_crane_deg_s"] + 0.05)


def v42_doc(doc41: dict) -> tuple[dict, dict]:
    """the v4.2 direction from the v4.1 one: closer orbits, the meter's windows, the presets' meter places; returns (doc, closer report)"""
    doc = copy.deepcopy(doc41)
    closer = apply_closer(doc, V42_CLOSER)
    doc["flow"]["meter"] = meter_windows(doc)
    doc["flow"]["revision"] = "4.2"
    doc["flow"]["schema"] = doc["flow"]["schema"] + FLOW_SCHEMA_V42
    pv = doc["presets"]["vertical"]
    x0, x1, y0, y1 = legend_rect(V42_METER_PLACE["vertical"])
    pv["legend"] = dict(V42_METER_PLACE["vertical"], what=V41_LEGEND["what"], top_frac=round(y0, 4), bottom_frac=round(y1, 4), x_frac=[round(x0, 4), round(x1, 4)],
                        shows="only inside flow.meter.windows (v4.2): about 4 s at each switch-on of the physics view, then a 1-s fade out",
                        under_cards=V41_LEGEND["under_cards"], labels=V41_LEGEND["labels"], framing=V41_LEGEND["framing"])
    hx0, hx1, hy0, hy1 = legend_rect_h(V42_METER_PLACE["horizontal"])
    doc["presets"]["horizontal"]["legend"] = dict(V42_METER_PLACE["horizontal"], top_frac=round(hy0, 4), bottom_frac=round(hy1, 4), x_frac=[round(hx0, 4), round(hx1, 4)],
                                                  shows="only inside flow.meter.windows (v4.2)")
    doc["created"] = ("generated by dancecap.film_recap_flow build --version v4.2 (the v4.1 plan + the physics meter only at each switch-on + the slow "
                      "orbits 0.8 x closer) on top of dancecap.film_recap; edit the rules there")
    return doc, closer


def build_v42(log=print, write: bool = True, hand_events: Path | None = None, out: Path | None = None, need_path: Path = NEED3D) -> dict:
    """v4.2 = the v4.1 plan (every move, timing and rate: build_v41 meaning-first, rebuilt) + the physics meter's windows (flow.meter) + every slow
    orbit 0.8 x closer about the couple (apply_closer). The ways' durations and placement stay v4.1's; their peaks are re-measured on the closer
    orbits (PoseModel, 9:16 and 16:9) and must stay under v4.1's caps"""
    r = build_v41(log=log, cfg4=v41_cfg(), write=False, hand_events=hand_events, need_path=need_path)
    cfg4 = v42_cfg()
    geo4, need = r["geo4"], r.get("need")
    pov = fr.load_pov(fr.Paths(), "auto")
    doc41, met41 = r["doc"], r["metrics"]
    met41_h = film_metrics(doc41, geo4, r["cfg4"], aspect="horizontal")
    met41["eye"] = eye_stats(met41, doc41, geo4)
    met41["eye_16x9"] = eye_stats(met41_h, doc41, geo4)
    met41["ways_16x9"] = _peak_summary(met41_h)
    doc, closer = v42_doc(doc41)
    met = film_metrics(doc, geo4, cfg4)
    v41_extra_metrics(met, doc, geo4, cfg4, need, pov)
    met_h = film_metrics(doc, geo4, cfg4, aspect="horizontal")
    v42_extra_metrics(met, doc, geo4, cfg4, pov, met_h)
    # the ways' planned peaks on the closer orbits (9:16 as every v4 number; 16:9 next to it); v4.1's kept for the record
    shots = shots_of(doc["segments"])
    per = {p["k"]: p for p in met["ways"].get("peaks", [])}
    per_h = {p["k"]: p for p in met_h["ways"].get("peaks", [])}
    keys = ("peak_az_deg_s", "peak_rot_deg_s", "peak_speed_m_s", "peak_pitch_deg_s")
    for k, sh in enumerate(shots):
        if k == 0:
            continue
        for i in sh.segs:
            v4b = (doc["segments"][i]["camera"].get("transition_in") or {}).get("v4")
            if v4b is None:
                continue
            v4b["peaks_planned_v41"] = v4b.get("peaks_planned")
            ks = keys if (per.get(k) or {}).get("type") in ("crane_up", "crane_down") else keys[:3]
            v4b["peaks_planned"] = {k2: (per.get(k) or {})[k2] for k2 in ks if (per.get(k) or {}).get(k2) is not None}
            v4b["peaks_planned_16x9"] = {k2: (per_h.get(k) or {})[k2] for k2 in ks if (per_h.get(k) or {}).get(k2) is not None}
    phq = PhoneQ41(pov, geo4, r["cfg4"]) if pov is not None else None
    checks = check_v4(doc, r["tl"], r["sp"], r["geo"], geo4, cfg4, None, met, phq)
    doc["flow"]["v4"]["measured"] = _metrics_public(met)
    doc.setdefault("recap_film", {}).setdefault("v4", {})["v42"] = {"owner": V42_OWNER, "from": "the v4.1 plan (build_v41, meaning-first), rebuilt", "closer": closer,
                                                                     "meter": {k2: v for k2, v in doc["flow"]["meter"].items() if k2 in ("windows", "switch_ons", "notes")}}
    res = dict(r)
    res.update({"doc": doc, "cfg4": cfg4, "metrics": met, "metrics_16x9": met_h, "checks": checks, "closer": closer, "doc41": doc41, "metrics41": met41,
                "metrics41_16x9": met41_h})
    if write:
        write_all_v42(res, Path(out or OUT_V4))
    return res


def v42_rows(res: dict) -> list[list]:
    """the v4.1 / v4.2 table: radii, eye distances, way peaks (both aspects), the meter"""
    m41, m42 = res["metrics41"], res["metrics"]
    cl = res["closer"]["runs"]

    def rng(xs):
        xs = sorted({x for x in xs if x is not None}, key=str)
        return " / ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in xs) if xs else "-"

    def g(m, *keys):
        x = m
        for k in keys:
            if not isinstance(x, dict) or x.get(k) is None:
                return "-"
            x = x[k]
        return x

    def pk(m, key):
        return g(m, "ways", key, "max")
    caps_ = v41_targets(res["cfg4"])
    mm = m42.get("meter") or {}
    ed = cl[0]["eye_distance_m"] if cl else {"vertical": {}, "horizontal": {}}
    return [
        ["slow-orbit radius about the couple, 9:16 (radius_vertical_m, m)", rng(x["v41"]["radius_vertical_m"] for x in cl), rng(x["v42"]["radius_vertical_m"] for x in cl), "0.8 x"],
        ["slow-orbit radius about the couple, 16:9 (radius_m, m)", rng(x["v41"]["radius_m"] for x in cl), rng(x["v42"]["radius_m"] for x in cl), "0.8 x"],
        ["eye height + inline-lift cap (m)", rng(f"{x['v41']['height_m']} + {x['v41']['inline_lift_max_m']}" for x in cl),
         rng(f"{x['v42']['height_m']} + {x['v42']['inline_lift_max_m']}" for x in cl), "0.5-2.4 m"],
        ["eye distance to the look point (the block), 9:16 / 16:9 (m)", f"{ed['vertical'].get('v41')} / {ed['horizontal'].get('v41')}",
         f"{ed['vertical'].get('v42')} / {ed['horizontal'].get('v42')}", "0.8 x"],
        ["eye distance to the couple (centre 1 m up), settled orbits, sampled median 9:16 / 16:9 (m)",
         f"{g(m41, 'eye', 'orbit_eye_distance_m', 'median')} / {g(m41, 'eye_16x9', 'orbit_eye_distance_m', 'median')}",
         f"{g(m42, 'eye', 'orbit_eye_distance_m', 'median')} / {g(m42, 'eye_16x9', 'orbit_eye_distance_m', 'median')}", ""],
        ["settled slow-orbit time at the full 0.8 x (the rest: the 6-s eases next to the phone ways)", "-",
         f"{g(m42, 'closer_ease', 'full_closeness_s')} of {g(m42, 'closer_ease', 'settled_orbit_s')} s ({g(m42, 'closer_ease', 'windows')} eases)", ""],
        ["closest ground-level 3D eye radius about the couple, 9:16 / 16:9 (m)", f"{g(m41, 'eye', 'ground_3d_radius_min_m')} / {g(m41, 'eye_16x9', 'ground_3d_radius_min_m')}",
         f"{g(m42, 'eye', 'ground_3d_radius_min_m')} / {g(m42, 'eye_16x9', 'ground_3d_radius_min_m')}", ">= 1.4 m (the clear radius)"],
        ["closest the eye comes to a dancer (centre of mass, horizontal), 9:16 / 16:9 (m)", f"{g(m41, 'eye', 'nearest_dancer_m', 'min')} / {g(m41, 'eye_16x9', 'nearest_dancer_m', 'min')}",
         f"{g(m42, 'eye', 'nearest_dancer_m', 'min')} / {g(m42, 'eye_16x9', 'nearest_dancer_m', 'min')}", ""],
        ["ways 9:16: peak turn about the couple / view rotation (deg/s) / eye speed (m/s)", f"{pk(m41, 'peak_az_deg_s')} / {pk(m41, 'peak_rot_deg_s')} / {pk(m41, 'peak_speed_m_s')}",
         f"{pk(m42, 'peak_az_deg_s')} / {pk(m42, 'peak_rot_deg_s')} / {pk(m42, 'peak_speed_m_s')}", f"<= {caps_['cap_az_deg_s']} / {caps_['cap_rot_deg_s']} / {caps_['cap_speed_m_s']}"],
        ["ways 16:9: peak turn / rotation (deg/s) / eye speed (m/s)", f"{g(m41, 'ways_16x9', 'az')} / {g(m41, 'ways_16x9', 'rot')} / {g(m41, 'ways_16x9', 'speed')}",
         f"{g(m42, 'ways_16x9', 'az')} / {g(m42, 'ways_16x9', 'rot')} / {g(m42, 'ways_16x9', 'speed')}", f"<= {caps_['cap_az_deg_s']} / {caps_['cap_rot_deg_s']} / {caps_['cap_speed_m_s']}"],
        ["cranes: peak pitch 9:16 / 16:9 (deg/s)", f"{pk(m41, 'crane_pitch_deg_s')} / {g(m41, 'ways_16x9', 'pitch')}", f"{pk(m42, 'crane_pitch_deg_s')} / {g(m42, 'ways_16x9', 'pitch')}",
         f"<= {caps_['cap_crane_deg_s']}"],
        ["camera moves / ways' durations", f"{m41['moves']['total']} / {m41['ways']['min_s']}-{m41['ways']['max_s']} s", f"{m42['moves']['total']} / {m42['ways']['min_s']}-{m42['ways']['max_s']} s", "unchanged"],
        ["orbit rates (deg/s)", ", ".join(str(x) for x in m41["three_d"]["drift_rates_deg_s"]), ", ".join(str(x) for x in m42["three_d"]["drift_rates_deg_s"]), "unchanged"],
        ["phone holds (n, mean s)", f"{m41['phone_holds']['n']}, {m41['phone_holds']['mean_s']}", f"{m42['phone_holds']['n']}, {m42['phone_holds']['mean_s']}", "unchanged"],
        ["physics meter on screen", f"whenever the load colours are on ({mm.get('physics_on_s')} s)", f"{mm.get('windows')} windows, {mm.get('shown_s')} s",
         "a few seconds at each switch-on"],
    ]


def review_v42_md(res: dict) -> str:
    doc, met, ch = res["doc"], res["metrics"], res["checks"]
    meter = doc["flow"]["meter"]
    cl = res["closer"]
    L = ["# v4.2 direction review: Class recap (larissa_kadu_recap)", "", V42_OWNER, "",
         "v4.2 = the v4.1 plan (every move, its timing and every orbit rate: `build_v41` meaning-first, rebuilt) with three output changes: the physics "
         "meter only for a few seconds at each switch-on, the slow orbits 20 % closer, the face swap on the splat dancers only. v4.1 stays reproducible: "
         "`python -m dancecap.film_recap_flow build --version v4.1` -> `v4/direction/v4.1/` (its review there). Nothing was recorded.", "",
         "## v4.1 against v4.2", ""]
    L += _md_table(["", "v4.1", "v4.2", "rule"], v42_rows(res))
    L += ["", "Sampled numbers: the plan's poses (`PoseModel`, 30 Hz; 16:9 uses radius_m). Unity re-measures the live camera (`hm_film_flow --action check`), "
          "which also applies the inline lift, FitSpan and (9:16) the vertical band.", "",
          "## The physics meter (flow.meter)", "",
          f"One window per switch-on of the physics view in the layer timeline ({meter['switch_ons']} switch-ons): fade in {meter['fade_in_s']} s, shown until "
          f"{meter['hold_s']} s after its start, fade out {meter['fade_out_s']} s (smootherstep). Place: the very top of the safe area, top right "
          f"(9:16: {V42_METER_PLACE['vertical']['top_px_at_1080w']} px down at 1080 x 1920, scale {V42_METER_PLACE['vertical']['scale']}; 16:9: "
          f"{V42_METER_PLACE['horizontal']['top_px_at_1080h']:g} px down, right edge at the title-safe margin, scale {V42_METER_PLACE['horizontal']['scale']}).", ""]
    L += _md_table(["switch-on (film s)", "segment", "window", "delay", "why"],
                   [[w["on_s"], w["segment"], f"{w['start_s']}-{w['end_s']} (full {w['full_s']}-{w['fade_s']})", f"{w['delay_s']} s",
                     ("after the " + w["after"][-1]) if w["after"] else "at the switch-on"] for w in meter["windows"]])
    L += [f"- {n}" for n in meter.get("notes", [])]
    mm = met.get("meter") or {}
    hc = mm.get("holds_clear") or {}
    L += ["", f"Shown {mm.get('shown_s')} s of the {mm.get('physics_on_s')} s of physics; frames shown without physics: {mm.get('frames_shown_without_physics')}, "
          f"over a chapter card: {mm.get('frames_shown_over_a_card')}. In the 9:16 phone-hold frames it shows over, the teachers' box is clear of it in "
          f"{100 * (1 - hc.get('under_legend_share', 1.0)):.1f} % of {hc.get('frames')} frames.", "",
          "## The closer orbits", "",
          f"Every slow-orbit block (the orbits; the 6 pivots, the physics views and the replay ride on it) at {V42_CLOSER} x its v4.1 eye offset about the "
          "couple: both radii and the eye's height above the look point (the same look-down angle: the eye moves in along its line of sight), and the "
          "inline lift's cap. `lens_distance_scale` 1.25 makes Unity choose the lens of the v4.1 distance (CoupleFov), so the dancers fill 1.25 x more of the "
          "frame: with the v4.1 lens rule alone the lens widens with the closeness and the 16:9 picture would keep the couple at the same size. The ways "
          "blend the live orbit states (C2 as before); the cranes' ground end is the closer orbit, their top end and the overheads stay.", "",
          "The phone ways: with the orbit at 0.8 x everywhere, the ways into and out of the phones had to cover more distance in their v4.1 time and "
          "their eye speed rose over the 1.2 m/s cap (first build: 9:16 1.27 m/s, 16:9 1.56 m/s; the 16:9 v4.1 ways already peak at 1.2). So the orbit "
          f"is at v4.1's distance at each phone way and eases to 0.8 x over {V42_CLOSER_EASE_S:g} s inside the orbit (smootherstep, `closer_in_s` / "
          "`closer_out_s`; the lens stays v4.1's: a slow dolly in with a fixed lens, peak about 0.28 m/s); the phone ways are then exactly v4.1's. "
          "Next to a crane or the replay the orbit stays at 0.8 x (a crane's eye speed has no cap, its pitch is unchanged).", "",
          f"The framing bands: with the dancers 1.25 x larger, v4.1's 9:16 band (0.31-0.77 of the height from the bottom, made to keep them under the "
          f"always-on legend) put their feet onto the caption block, and so did the 16:9 framing (editor stills at 60 s). v4.2: 9:16 "
          f"{V42_BAND_V['band_lo']}-{V42_BAND_V['band_hi']} (the meter now shows over a 3D view only at its 274-279 s window), 16:9 "
          f"{V42_BAND_H['band_lo']}-{V42_BAND_H['band_hi']} (new: the couple's floor .. {V42_BAND_H['top_m']} m between the captions and the frame top; the "
          "lens widens only where they would not fit).", ""]
    L += _md_table(["orbit part", "ease", "film s", "length"], [[e["segment"], e["side"], f"{e['film'][0]}-{e['film'][1]}", f"{e['T']} s"] for e in cl.get("eases", [])])
    L += [""]
    L += _md_table(["run", "film s", "drift", "9:16 radius", "16:9 radius", "eye height + lift", "eye distance 9:16 / 16:9"],
                   [[x["run"], f"{x['film'][0]:.1f}-{x['film'][1]:.1f}", x["drift_deg_s"], f"{x['v41']['radius_vertical_m']} -> {x['v42']['radius_vertical_m']}",
                     f"{x['v41']['radius_m']} -> {x['v42']['radius_m']}", f"{x['v41']['height_m']} + {x['v41']['inline_lift_max_m']} -> {x['v42']['height_m']} + "
                     f"{x['v42']['inline_lift_max_m']}", f"{(x['eye_distance_m']['vertical'] or {}).get('v41')} -> {(x['eye_distance_m']['vertical'] or {}).get('v42')} / "
                     f"{(x['eye_distance_m']['horizontal'] or {}).get('v41')} -> {(x['eye_distance_m']['horizontal'] or {}).get('v42')}"] for x in cl["runs"]])
    L += ["", ("Clear radius: " + "; ".join(f"{c['segment']}: {c['why']}" for c in cl["clamped"])) if cl["clamped"] else
          "Clear radius: 0.8 x keeps every orbit outside the couple's 1.4-m clear radius; no radius had to stop at it.", "",
          "Per way (planned peaks: turn about the couple deg/s / view rotation deg/s / eye speed m/s / crane pitch deg/s):", ""]
    p41 = {p["k"]: p for p in res["metrics41"]["ways"].get("peaks", [])}
    p41h = {p["k"]: p for p in res["metrics41_16x9"]["ways"].get("peaks", [])}
    p42h = {p["k"]: p for p in res["metrics_16x9"]["ways"].get("peaks", [])}

    def f4(p):
        return " / ".join("-" if p.get(k) is None else f"{p[k]:g}" for k in ("peak_az_deg_s", "peak_rot_deg_s", "peak_speed_m_s", "peak_pitch_deg_s")) if p else "-"
    L += _md_table(["#", "type", "film s", "T", "v4.1 9:16", "v4.2 9:16", "v4.1 16:9", "v4.2 16:9"],
                   [[p["k"], p["type"], p["start"], p["T"], f4(p41.get(p["k"])), f4(p), f4(p41h.get(p["k"])), f4(p42h.get(p["k"]))] for p in met["ways"].get("peaks", [])])
    L += ["", "## Face swap on the splat dancers only", "",
          "`dancecap.film_faceswap` swaps a face only where the face sidecar says that dancer's face is a splat-dancer render (`d[].layer == \"splat\"`; weight "
          "x its opacity `d[].la` x (1 - the phone video's opacity)): never on a phone's own full-frame video, a frustum picture (`surfaces`), a skeleton-only "
          "or mesh-avatar view or any other layer. The recorder writes the layer per dancer and frame (FilmFaceSidecar); a sidecar without it falls back to "
          "the frame's `splats` share (0 = no swap). It runs once, on the final recording, after the render.", "",
          "## Checks", ""]
    L += [f"- {'OK' if t['ok'] else 'MISSED'}: {t['target']}: want {t['want']}, measured {t['measured']}" for t in ch["targets"]]
    L += [f"- PROBLEM: {b}" for b in ch["bad"]] + [f"- {n}" for n in ch["notes"]]
    return "\n".join(L) + "\n"


def contract_v42_md(res: dict) -> str:
    """the v4.2 section that heads UNITY_CONTRACT_v4.md (generic words only: lead / follow / the recap capture)"""
    meter = res["doc"]["flow"]["meter"]
    pv, ph = V42_METER_PLACE["vertical"], V42_METER_PLACE["horizontal"]
    x0, x1, y0, y1 = legend_rect(pv)
    hx0, hx1, hy0, hy1 = legend_rect_h(ph)
    cl = res["closer"]["runs"][0]
    L = ["# Unity contract: the v4 class-recap direction (`hm film flow v2`, revision 4.2)", "",
         "Source: `dancecap/work/larissa_kadu_recap/v4/direction/direction.json` (`flow.revision` \"4.2\"; v4.1 in `v4/direction/v4.1/`, v4.0 in "
         "`v4/direction/v4.0/`). Built by `python -m dancecap.film_recap_flow build --version v4.2`, checked by `... check --version v4`. Write no names "
         "into head-movement: say lead / follow / \"the recap capture\".", "",
         "## v4.2: what changes for the Unity director (on top of v4.1 A-G and v4.0 1-9 below)", "",
         "| # | what | where | v4.1 |", "|---|---|---|---|",
         "| H | the physics meter shows only in its windows (about 4 s at each switch-on of the physics view, 0.3-s fade in, 1-s fade out), at the very top of the "
         "safe area in both aspects | `flow.meter.windows`, `flow.meter.place`, FlowTimeline.MeterAlpha, FilmOverlay.Tick | always while the load colours are on |",
         "| I | the slow orbits 0.8 x closer about the couple; the lens chosen for the v4.1 distance | the slow-orbit blocks (radius_m, radius_vertical_m, height_m, "
         "inline_lift_max_m, `lens_distance_scale`), `camera.v4.closer` | 3.6 / 4.3 m, CoupleFov(distance) |",
         "| J | the face sidecar says which layer each dancer's face is on | FilmFaceSidecar `d[].layer` / `d[].la` | (not written) |", "",
         "**H. The meter.** alpha(t) = max over the windows of: smootherstep((t - start_s) / (full_s - start_s)) until full_s, 1 until fade_s, "
         "1 - smootherstep((t - fade_s) / (end_s - fade_s)) until end_s, 0 outside (FlowMath.Smootherstep, the quintic). SkeletonLegend.Alpha = alpha(t) x "
         "(1 - max(chapter card alpha, title alpha)); the legend's own Show (the physics layer) still gates it. Place: 9:16 top = safe.top_frac x H + 8 x "
         f"W / 1080 ({pv['top_px_at_1080w']} px), right edge {pv['right_px_at_1080w']:g} px in, scale {pv['scale']} (x {100 * x0:.1f}-{100 * x1:.1f} %, y "
         f"{100 * y0:.2f}-{100 * y1:.2f} %); 16:9 top = safe.top_frac x H + 8 x H / 1080 ({ph['top_px_at_1080h']:g} px), right edge at the title-safe margin "
         f"({ph['right_px_at_1920w']:g} px), scale {ph['scale']} (x {100 * hx0:.1f}-{100 * hx1:.1f} %, y {100 * hy0:.2f}-{100 * hy1:.2f} %). It never moves. "
         "While it shows (alpha > 0.05) its rectangle is an obstacle for the call-out labels and LabelCeiling stays under its bottom + 10 x unit, in both "
         "aspects; the face sidecar lists it in `ui` only while it shows. The windows (film s):", ""]
    L += _md_table(["switch-on", "start", "full", "fade", "end"], [[w["on_s"], w["start_s"], w["full_s"], w["fade_s"], w["end_s"]] for w in meter["windows"]])
    L += ["", "A window that met a chapter card starts 0.2 s after the card (the card's pill sits in the same top band): the meter never shows over a card.", "",
          f"**I. Closer orbits.** Every block with `v4.role` slow_orbit: radius_m {cl['v41']['radius_m']} -> {cl['v42']['radius_m']}, radius_vertical_m "
          f"{cl['v41']['radius_vertical_m']} -> {cl['v42']['radius_vertical_m']}, height_m {cl['v41']['height_m']} -> {cl['v42']['height_m']} (look_height_m "
          f"0.95 kept: the same look-down angle), inline_lift_max_m {cl['v41']['inline_lift_max_m']} -> {cl['v42']['inline_lift_max_m']}, and "
          f"`lens_distance_scale` {cl['v42']['lens_distance_scale']}: FilmCamera.Orbit sets Fov = CoupleFov(|eye - look| x lens_distance_scale) (the lens of "
          "the v4.1 distance) before fov_scale, FitSpan and the 9:16 vertical band, so the dancers fill 1.25 x more of the frame. FitSpan and the band may "
          "still widen the lens (never narrower) and back the eye off at the lens limit - the check reports where. The ways need nothing new: they blend "
          "the live orbit states (C2); a crane's ground end is the closer orbit, the overheads stay.", "",
          "The ease (keeps the phone ways v4.1's): a block next to a phone way carries `closer_in_s` [a, b] (after a phone way out) and / or `closer_out_s` "
          "[c, d] (before a phone way in), film seconds. w(t) = Ramp(t, a, b) x (1 - Ramp(t, c, d)) (smootherstep, a missing window counts as 1); the "
          "factor on the block's eye offset about the look point is f = 1 + (1 / k - 1)(1 - w), k = `v4.closer.scale` (FlowMath.CloserFactor): radius x f, "
          "height_m = look_height_m + (height_m - look_height_m) x f, the inline lift's cap x f, and the lens CoupleFov(|eye - look| x lens_distance_scale / f) "
          "(v4.1's lens throughout). At a phone way f = 1.25: the way's 3D end is v4.1's orbit.", "",
          f"The framing bands: 9:16 `vertical_band` {json.dumps(V42_BAND_V)} (v4.1: 0.31-0.77, under the always-on legend), and new in 16:9 "
          f"`horizontal_band` {json.dumps(V42_BAND_H)}: FilmCamera.Orbit applies FitBand(ref p, couple floor point, top_m, band_lo, band_hi, max(24, fov), "
          "55) after FitSpan in 16:9, as the 9:16 band (66). With the dancers 1.25 x larger the v4.1 bands put their feet onto the caption block.", ""]
    L += _md_table(["orbit part", "ease", "film s"], [[e["segment"], e["side"], f"{e['film'][0]}-{e['film'][1]}"] for e in res["closer"].get("eases", [])])
    L += ["",
          "**J. The face sidecar.** Per dancer and frame: `layer` = \"splat\" (that dancer's splats are drawn: SplatDancers on, its splat avatar loaded, "
          "its avatar slot visible), \"avatar\" (the mesh avatar is drawn instead), \"skeleton\" (neither: lines only) or \"none\" (no head pose); `la` = "
          "that layer's opacity now (the splats' composite opacity: the phone-video fade). dancecap.film_faceswap swaps a face only on layer \"splat\", "
          "with weight x la x (1 - video_alpha), never behind a frustum surface.", "",
          "Everything below (v4.1's contract) still holds, except where H-J replace it (v4.1 B: the legend's place in 9:16 is kept, its 'always on' is not).", ""]
    return "\n".join(L) + "\n"


def write_all_v42(res: dict, out: Path) -> None:
    doc, tl = res["doc"], res["tl"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "narration").mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    (out / "direction.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    caps.write_timeline(out / "narration" / "timeline.json", tl)
    ch = []
    for i, c in enumerate(doc["chapters"]):
        ch.append({"title": c["title"], "startS": c["film_s"], "endS": doc["chapters"][i + 1]["film_s"] if i + 1 < len(doc["chapters"]) else doc["timeline"]["film_duration_s"],
                   "segment": c["segment"]})
    (out / "chapters.json").write_text(json.dumps({"schema": "dancecap.film_recap/1/chapters", "film_duration_s": doc["timeline"]["film_duration_s"], "chapters": ch},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")
    cols = [("v4.1", res["metrics41"]), ("v4.2", res["metrics"])]
    ba = {"schema": "dancecap.film_recap_flow/v4.2/before_after", "columns": ["v4.1", "v4.2"], "v42_rows": v42_rows(res),
          "rows": before_after_rows_n(cols, res["targets"]), "metrics": {lab: _metrics_public(m) for lab, m in cols},
          "metrics_16x9": {"v4.1": _metrics_public(res["metrics41_16x9"]), "v4.2": _metrics_public(res["metrics_16x9"])}, "closer": res["closer"],
          "meter": doc["flow"]["meter"], "check_targets": res["checks"]["targets"]}
    (out / "before_after.json").write_text(json.dumps(ba, indent=1, default=str), encoding="utf-8")
    rep = {"schema": "dancecap.film_recap_flow/2.2/report", "checks": res["checks"], "targets": res["targets"], "metrics": _metrics_public(res["metrics"]),
           "metrics_16x9": _metrics_public(res["metrics_16x9"]), "metrics_v41": _metrics_public(res["metrics41"]), "closer": res["closer"], "meter": doc["flow"]["meter"],
           "runs": [{k: v for k, v in r.items() if k != "q"} | {"quality": r.get("q")} for r in res["runs"]], "ways": res["sched"]["ways"],
           "orbits": res["sched"]["plans"], "dp": res["dp"], "clips": res["clips"], "pivots": res["pivots"], "replay": res["replay"],
           "physics": [{k: r[k] for k in ("id", "film", "mode", "on", "why")} for r in res["phys_rows"]], "physics_timeline": physics_timeline(doc)}
    (out / "direction_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "clips.json").write_text(json.dumps({"what": "per phone: the master-second ranges of video the v4.2 direction needs (phone holds + frustum ways in / out, "
                                                        "padded 0.5 s; = v4.1's)", "phones": res["clips"]}, indent=1), encoding="utf-8")
    (out / "review_v4.2.md").write_text(review_v42_md(res), encoding="utf-8")
    res["ghost"] = ghost_overlap(doc, res["metrics"], res["geo4"], res["cfg4"])
    r4 = dict(res, cfg4=dataclasses_replace(res["cfg4"], version="4.1"))         # v4.1's sections, with v4.2's blocks and numbers
    base = contract_v41_md(r4) + "\n" + contract_md(r4).split("\n", 1)[1]
    (out / "UNITY_CONTRACT_v4.md").write_text(contract_v42_md(res) + "\n" + base.split("\n", 1)[1], encoding="utf-8")
    (out / "audio" / "make_film_mix.py").write_text(MIX_SCRIPT.replace("the v3 direction", "the v4.2 direction"), encoding="utf-8")
    m, c4 = res["metrics"], res["cfg4"]
    sheet_v4(m, doc, res["geo4"], out / "flow_sheet.png",
             f"v4.2 camera flow (the v4.1 plan, orbits 0.8 x closer): {m['moves']['total']} moves, {m['phone_holds']['n']} phone holds (mean {m['phone_holds']['mean_s']} s), "
             f"slow orbits {', '.join(str(x) for x in m['three_d']['drift_rates_deg_s'])} deg/s; physics meter {len(doc['flow']['meter']['windows'])} windows",
             band=c4.orbit_rate_deg_s, cap=c4.cap_az_deg_s, label="v4.2")


# ======================================================================================================== v4.3 (owner 2026-10-23)
# "when we show initial counter balances we need to orbit but maintain a two shot in profile so we can see the physics of the counter balance.
# otherwise the transitions to pov look good". v4.3 = v4.2 (every move, way, phone hold, orbit rate law, lens, band, meter window, chapter, caption
# and the time map) + ONE planner-written azimuth track for the slow orbit that carries chapters 1-3 (run 2, between the way out of phone 00 and
# the way into phone 06). The track holds the + profile (the eye about 90 deg off the couple's lead -> follow pelvis line on the lead's left: the
# joined hands toward the camera) through the first counterbalance, crosses the couple's line ONCE in the stretch without a counterbalance
# interval (the two phones the owner likes stand on opposite sides of the line: one crossing is forced), and holds the - profile (the lead's
# right: his right hand on her back toward the camera) through the both-hands / one-hand demonstrations. It leaves and rejoins the v4.2 law C2
# inside the orbit (phi = law + w (E - law), w = the closer ease weight of the block: 34.746-40.746 / 93.499-99.499), so both phone ways are
# v4.2's frame for frame. Unity plays the table (FilmCamera.WorldAzimuth through FlowMath.TrackAt; UNITY_CONTRACT_v4.md 'v4.3').

OUT_V42 = OUT_V4 / "v4.2"
V42_DIRECTION = OUT_V42 / "direction.json"
OUT_V43_FAST = OUT_V4 / "v4.3_fastpass"            # the fast-pass variant (an owner decision; not what the recorder reads)
V43_OWNER = ("owner 2026-10-23, after the v4.2 films: \"when we show initial counter balances we need to orbit but maintain a two shot in profile so we "
             "can see the physics of the counter balance. otherwise the transitions to pov look good\"")
V43_TRACK = {
    "schema": "hm film azimuth track v1", "revision": "4.3",
    "runs": [2],                  # the slow orbit(s) whose counterbalances are 'initial' (chapters 1-3 lie in run 2)
    "chapters": [1, 3],           # doc["chapters"] indices (0 = the opening jokes): the initial counterbalances
    "hz": 30.0,                   # the table's sample rate (the film's frame rate: the recorder's frames land on the samples)
    "knot_s": 1.0,                # the solver's cubic B-spline knot spacing
    "rate_min_deg_s": 1.0,        # never reversing: >= 1 deg/s the orbit's own way (v4.1 E)
    "rate_max_deg_s": 7.0,        # the crossing's ceiling (owner-approved exception to v4.1's 2.5-6 deg/s orbit band; the ways' caps are 12)
    "rate_max_outside_deg_s": 6.0,   # ... everywhere else on the track v4.1's band top (the exception is the crossing only)
    "accel_max_deg_s2": 2.0, "margin": 0.02,
    "profile_tol_deg": 15.0,      # a window's median view angle to the couple's line within this of 90 ...
    "share_tol_deg": 25.0, "share_min": 0.9,   # ... and >= 90 % of its frames within 25 of 90
    "bound_tol_deg": 1.0,         # a window that no legal track can bring within profile_tol (its own best, solved alone) must be within this of that best
    "w_interval": 2.0, "w_segment": 1.0, "min_part_s": 1.0, "lam_accel": 2e-3, "lam_jerk": 2e-3, "rounds": 3,
    "lift_top_m": 2.4,            # the ground band's top: no lift may take the eye above it
    # the crossing's eye lift: "raised" (the brief) = up to lift_top_m during the crossing; "none" = no lift there. Fix round 1 measured both in the
    # editor (stills every 0.5 s over 68-75 s, Jev "her face can be seen" + the calibrated estimate below): the lift shows part of her face ~0.5 s
    # longer at the start of the hidden stretch (her head rises beside / above his), but not at its core (~70.5-72 s: her chin would clear his head
    # only from a ~3.4 m eye; the band's top is 2.4). Face hidden: with the lift ~2.0 s, without ~2.5 s. So the brief's lift stays
    "crossing_lift": "raised",
    # an OPTIONAL faster pass through the line (owner decision, not decided: v4.3 plays without it). None = off; else {rate_max_deg_s, half_s,
    # w_gap}: within +-half_s of the line crossing the turn may reach rate_max_deg_s (the ways' cap is 12), and the gap's cb-layer segments are
    # scored on their own side of that moment (weight w_gap), so the track lingers in profile and runs through the line fast
    "fast_pass": None,
}
V43_FAST_PASS = {"rate_max_deg_s": 12.0, "half_s": 4.0, "w_gap": 0.5}      # the variant the verifier sketched (fix round 1, M1 option b)
FLOW_SCHEMA_V43 = ("; v4.3 (flow.revision '4.3', UNITY_CONTRACT_v4.md 'v4.3'): flow.azimuth_tracks {id: {film, t0, dt, n, deg[], lift_m[], law, ...}} and "
                   "camera.azimuth_track (the id) - inside film[0] < t < film[1] the eye's world azimuth (deg, unwrapped, atan2(z, x)) and its inline lift (m) "
                   "come from the table (linear between the samples), outside it the block's law world_azimuth_deg + drift_deg_s x (film - F0)")


def _dance_vec(doc: dict, f) -> np.ndarray:
    """film -> master seconds for an array of film times (each segment's own map, as _dance_at_film; outside the film: the edge segment's map)"""
    segs = doc["segments"]
    f = np.asarray(f, float)
    f0 = np.array([float(s["film"][0]) for s in segs])
    d0 = np.array([float(s["dance"][0]) for s in segs])
    sp = np.array([float(s["speed"]) for s in segs])
    i = np.clip(np.searchsorted(f0, f, side="right") - 1, 0, len(segs) - 1)
    return d0[i] + (f - f0[i]) * sp[i]


class PelvisLine:
    """the couple's lead -> follow line on the floor from the INSTALLED motion: joint 0 (the pelvis) of the capture's smplx_lead.bin /
    smplx_follow.bin, Unity capture coordinates (what FilmCamera.SideAzimuth / InlineLift / FitSpan read; DanceOrigin's offset cancels in a
    direction and is the same for the eye's centre), on the capture's clock. Also the joints the head-occlusion estimate and the side test use."""

    PELVIS, SPINE3, NECK, HEAD, L_SH, R_SH = 0, 9, 12, 15, 16, 17

    def __init__(self, cap_dir: Path = CAP_DIR):
        import hashlib
        from . import export_unity as eu
        j0 = 3 + 55 * 4
        self.sha, P = {}, {}
        for role in ("lead", "follow"):
            p = Path(cap_dir) / f"smplx_{role}.bin"
            raw = p.read_bytes()
            self.sha[role] = hashlib.sha256(raw).hexdigest()
            h, m = eu.read_motion(p)
            self.t = np.linspace(float(h["t0"]), float(h["t1"]), int(h["frames"]))
            P[role] = m[:, j0:j0 + 55 * 3].reshape(-1, 55, 3).astype(float)
        self.lead, self.follow = P["lead"], P["follow"]
        ln = self.follow[:, 0] - self.lead[:, 0]
        self.theta = np.degrees(np.unwrap(np.arctan2(ln[:, 2], ln[:, 0])))       # world deg, unwrapped (the viewer's atan2(z, x))
        self.sep = np.hypot(ln[:, 0], ln[:, 2])

    def theta_at(self, d) -> np.ndarray:
        return np.interp(np.asarray(d, float), self.t, self.theta)

    def joint_at(self, role: str, j: int, d) -> np.ndarray:
        a = self.lead if role == "lead" else self.follow
        d = np.atleast_1d(np.asarray(d, float))
        return np.stack([np.interp(d, self.t, a[:, j, k]) for k in range(3)], axis=1)

    def damped(self, role: str, doc: dict, film, tau: float = 0.6) -> np.ndarray:
        """FilmCamera.Follow's twin (y = 0): the weighted mean of the pelvis over 10 samples tau / 3 apart back in film time, weights exp(-i / 3)
        (the caller keeps it away from a dance-time cut)"""
        film = np.atleast_1d(np.asarray(film, float))
        dt = tau / 3.0
        acc, ws = np.zeros((len(film), 3)), 0.0
        for i in range(10):
            wgt = math.exp(-i * dt / tau)
            acc += wgt * self.joint_at(role, self.PELVIS, _dance_vec(doc, np.clip(film - i * dt, 0.0, None)))
            ws += wgt
        out = acc / ws
        out[:, 1] = 0.0
        return out


def _ss3(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """smootherstep of x (already scaled to its window) and its 1st / 2nd derivatives by x"""
    x = np.asarray(x, float)
    xc = np.clip(x, 0.0, 1.0)
    inside = (x > 0.0) & (x < 1.0)
    s = xc ** 3 * (xc * (6.0 * xc - 15.0) + 10.0)
    s1 = np.where(inside, 30.0 * xc * xc * (1.0 - xc) ** 2, 0.0)
    s2 = np.where(inside, 60.0 * xc * (1.0 - xc) * (1.0 - 2.0 * xc), 0.0)
    return s, s1, s2


def track_blend(t: np.ndarray, win_in: tuple[float, float], win_out: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """the weight w(t) = Ramp(win_in) x (1 - Ramp(win_out)) of the profile track over the law (closer_weight's law, smootherstep ramps) and its 1st /
    2nd time derivatives: 0 (the law) at and outside the window's ends, with zero 1st and 2nd derivatives there - the C2 joins"""
    (ia, ib), (oa, ob) = win_in, win_out
    ri, ri1, ri2 = _ss3((t - ia) / (ib - ia))
    ri1, ri2 = ri1 / (ib - ia), ri2 / (ib - ia) ** 2
    ro, ro1, ro2 = _ss3((t - oa) / (ob - oa))
    ro1, ro2 = ro1 / (ob - oa), ro2 / (ob - oa) ** 2
    return ri * (1.0 - ro), ri1 * (1.0 - ro) - ri * ro1, ri2 * (1.0 - ro) - 2.0 * ri1 * ro1 - ri * ro2


def view_angle_to_line(phi_deg, theta_deg) -> np.ndarray:
    """the angle (0..180 deg) between the view (the eye at azimuth phi about the couple, looking at it: phi + 180) and the lead -> follow line
    (theta): 90 = a profile two-shot, 0 / 180 = one dancer in front of the other"""
    return np.abs((np.asarray(phi_deg, float) + 180.0 - np.asarray(theta_deg, float) + 180.0) % 360.0 - 180.0)


def inline_lift_twin(phi_deg: np.ndarray, pl: np.ndarray, pf: np.ndarray) -> np.ndarray:
    """FilmCamera.InlineLift's twin (m): 1.9 m x k x apart - k = 1 - SmoothStep(|r| / 45 deg) with r the eye's angle off the lead -> follow line
    (either side; 0 = along it), apart = SmoothStep(InverseLerp(0.2, 0.7, distance)); 0 while the dancers stand within 0.2 m"""
    ln = pf - pl
    n2 = ln[:, 0] ** 2 + ln[:, 2] ** 2
    x = (np.radians(phi_deg) - np.arctan2(ln[:, 2], ln[:, 0]) + math.pi) % (2.0 * math.pi) - math.pi
    r = (x + 0.5 * math.pi) % math.pi - 0.5 * math.pi

    def hs(v):                                           # Mathf.SmoothStep(0, 1, v): the cubic Hermite
        v = np.clip(v, 0.0, 1.0)
        return v * v * (3.0 - 2.0 * v)
    out = 1.9 * (1.0 - hs(np.abs(r) / (0.25 * math.pi))) * hs((np.sqrt(n2) - 0.2) / 0.5)
    out[n2 < 0.04] = 0.0
    return out


def hann_mean(x: np.ndarray, half: int = 36) -> np.ndarray:
    """FilmCamera.Orbit's Hann window over +-half frames (w = cos(k / (half + 1) x pi / 2)^2): x carries half extra samples at both ends; returns
    len(x) - 2 half values"""
    k = np.arange(-half, half + 1)
    w = np.cos(k / (half + 1.0) * math.pi * 0.5) ** 2
    return np.convolve(x, w, mode="valid") / w.sum()


def v43_windows(doc: dict, sh: Shot, a: float, b: float, cb: dict, cfg: dict = V43_TRACK) -> dict:
    """the windows the profile track is scored on: the counterbalance intervals (counterbalance.json, kind counterbalance, mapped to film time) and
    the counterbalance-layer segments of the 'initial' chapters inside the run; the crossing = the longest stretch of [a, b] between two intervals.
    A window is scored on its part inside the track's window (a, b: outside it the phone ways, v4.2's, decide) and outside the crossing, on the
    side of the line that part lies (before the crossing: the first profile; after it: the second); a remaining part under min_part_s is reported only"""
    segs = doc["segments"]
    ch = doc["chapters"]
    c0, c1 = cfg["chapters"]
    span = (float(ch[c0]["film_s"]), float(ch[c1 + 1]["film_s"]) if c1 + 1 < len(ch) else float(doc["timeline"]["film_duration_s"]))
    lo, hi = max(span[0], sh.f0), min(span[1], sh.f1)
    ivs = []
    for iv in cb.get("intervals", []):
        if iv.get("kind") != "counterbalance":
            continue
        f0, f1 = film_of(doc, float(iv["t0"])), film_of(doc, float(iv["t1"]))
        if f1 > lo and f0 < hi:
            ivs.append((f0, f1, iv))
    inner = sorted((max(f0, a), min(f1, b)) for f0, f1, _ in ivs if min(f1, b) > max(f0, a))
    gaps = [(p[1], q[0]) for p, q in zip(inner, inner[1:]) if q[0] > p[1]]
    if not gaps:
        raise fr.RecapError(f"v4.3: run {(segs[sh.segs[0]]['camera'].get('v4') or {}).get('run')}: no stretch between two counterbalance intervals to cross the line in")
    gap = max(gaps, key=lambda g: g[1] - g[0])
    rows = [{"name": f"counterbalance {f0:.2f}-{f1:.2f}", "kind": "interval", "film": [round(f0, 3), round(f1, 3)], "confidence": iv.get("confidence"),
             "connection": ((iv.get("connection") or {}).get("pair") if isinstance(iv.get("connection"), dict) else iv.get("connection"))} for f0, f1, iv in ivs]
    for i in sh.segs:
        s = segs[i]
        if (s.get("layers") or {}).get("counterbalance") and s["film"][1] > lo and s["film"][0] < hi:
            rows.append({"name": s["id"], "kind": "segment", "film": [round(float(s["film"][0]), 3), round(float(s["film"][1]), 3)]})
    for r in rows:
        x0, x1 = r["film"]
        p0, p1 = max(x0, a), min(x1, b)
        parts, why = [], []
        if p1 - p0 <= 0:
            why.append("inside a phone way (v4.2's, unchanged)")
        else:
            if x0 < a or x1 > b:
                why.append(f"{max(0.0, a - x0) + max(0.0, x1 - b):.2f} s inside a phone way (v4.2's, unchanged)")
            if p0 < gap[0]:
                parts.append([round(p0, 3), round(min(p1, gap[0]), 3), "first"])
            if p1 > gap[1]:
                parts.append([round(max(p0, gap[1]), 3), round(p1, 3), "second"])
            cut = (min(p1, gap[1]) - max(p0, gap[0]))
            if cut > 0:
                why.append(f"{cut:.2f} s inside the crossing")
            short = [p for p in parts if p[1] - p[0] < cfg["min_part_s"] - 1e-9]
            if short:
                why.append("a remaining part under " + f"{cfg['min_part_s']:g} s: " + ", ".join(f"{p[0]}-{p[1]}" for p in short))
            parts = [p for p in parts if p[1] - p[0] >= cfg["min_part_s"] - 1e-9]
        r["scored"] = parts
        r["not_scored"] = "; ".join(why)
    rows.sort(key=lambda r: (r["film"][0], r["kind"]))
    return {"chapters_film": [round(span[0], 3), round(span[1], 3)], "crossing": [round(gap[0], 3), round(gap[1], 3)], "windows": rows}


def solve_profile_track(t: np.ndarray, law: np.ndarray, W: float, theta: np.ndarray, win_in: tuple, win_out: tuple, items: list[dict],
                        cfg: dict = V43_TRACK, rate_max: np.ndarray | None = None) -> dict:
    """the profile track as a quadratic programme: phi = law + w E (w = track_blend, E = a cubic B-spline with knots every knot_s over the track's
    window), minimising, over the scored windows (weight / frames), the squared offset of the eye's azimuth about the line from its profile
    (phi - theta - side), + lam_accel int phi''^2 + lam_jerk int phi'''^2, subject to - on every sample inside the window - rate_min + margin <=
    phi' x sign(W) <= rate_max - margin and |phi''| <= accel_max - margin (linear in E: SLSQP from E = 0, which is feasible: the law). A window
    whose target (median within profile_tol, share_min within share_tol) is missed is re-weighted (x 2, up to `rounds` times) when it alone could
    meet it; its own best (the same programme with only that window) is reported for every window that misses"""
    from scipy.interpolate import BSpline
    from scipy.optimize import minimize
    a, b = float(win_in[0]), float(win_out[1])
    dt = 1.0 / float(cfg["hz"])
    sg = math.copysign(1.0, W)
    inw = (t > a + 1e-9) & (t < b - 1e-9)
    nk = max(4, int(math.ceil((b - a) / float(cfg["knot_s"]) - 1e-9)))
    knots = np.concatenate([[a] * 3, a + (b - a) * np.arange(nk + 1) / nk, [b] * 3])
    nc = len(knots) - 4
    Bm = np.zeros((3, len(t), nc))
    for j in range(nc):
        e = np.zeros(nc)
        e[j] = 1.0
        sp = BSpline(knots, e, 3, extrapolate=False)
        Bm[0, :, j] = np.nan_to_num(sp(t))
        Bm[1, :, j] = np.nan_to_num(sp.derivative(1)(t))
        Bm[2, :, j] = np.nan_to_num(sp.derivative(2)(t))
    w, w1, w2 = track_blend(t, win_in, win_out)
    D0 = w[:, None] * Bm[0]
    D1 = w1[:, None] * Bm[0] + w[:, None] * Bm[1]
    D2 = w2[:, None] * Bm[0] + 2.0 * w1[:, None] * Bm[1] + w[:, None] * Bm[2]
    for M in (D0, D1, D2):
        M[~inw] = 0.0
    D3 = np.gradient(D2, dt, axis=0)
    m_ = float(cfg["margin"])
    rmin, amax = float(cfg["rate_min_deg_s"]), float(cfg["accel_max_deg_s2"])
    rmax = np.full(len(t), float(cfg["rate_max_deg_s"])) if rate_max is None else np.asarray(rate_max, float)      # per sample (the crossing may be faster)
    cA_full = np.vstack([sg * D1[inw], -sg * D1[inw], D2[inw], -D2[inw]])
    nin = int(inw.sum())
    cB_full = np.concatenate([np.full(nin, rmin + m_ - abs(W)), abs(W) - rmax[inw] + m_, np.full(nin, -(amax - m_)), np.full(nin, -(amax - m_))])
    # the solver first sees every second sample (15 Hz: half the work); a 30-Hz row the result breaks joins the programme and it is solved again
    # (twice at most); whatever is left (never, so far) is pulled back toward the law, which is feasible
    half_rows = np.concatenate([np.arange(0, nin, 2) + q * nin for q in range(4)])
    R0 = cfg["lam_accel"] * dt * D2[inw].T @ D2[inw] + cfg["lam_jerk"] * dt * D3[inw].T @ D3[inw]

    def qp(its, x0=None):
        t_0 = __import__("time").perf_counter()
        Q, g = R0.copy(), np.zeros(nc)
        for it in its:
            m = it["m"]
            r = law[m] - theta[m] - it["side"]
            A = D0[m]
            sc = it["wt"] / max(1, int(m.sum()))
            Q += sc * A.T @ A
            g += sc * A.T @ r
        rows = half_rows
        nit = 0
        c = np.zeros(nc) if x0 is None else x0
        for _ in range(3):
            A_, B_ = cA_full[rows], cB_full[rows]
            res = minimize(lambda c_: 0.5 * c_ @ Q @ c_ + g @ c_, c, jac=lambda c_: Q @ c_ + g, method="SLSQP",
                           constraints=[{"type": "ineq", "fun": lambda c_, A_=A_, B_=B_: A_ @ c_ - B_, "jac": lambda c_, A_=A_: A_}],
                           options={"maxiter": 800, "ftol": 1e-11})
            c = res.x
            nit += int(res.nit)
            viol = np.nonzero(cA_full @ c < cB_full - 1e-9)[0]
            if not len(viol):
                break
            rows = np.union1d(rows, viol)                # the 30-Hz rows the 15-Hz solution broke join the programme
        Ac = cA_full @ c
        bad_ = Ac < cB_full
        s = 1.0
        if bad_.any():                                   # the law (c = 0) is strictly feasible: pull back along the segment to the feasible end
            s = float(min(1.0, np.min(cB_full[bad_] / np.where(np.abs(Ac[bad_]) > 1e-12, Ac[bad_], -1e-12))))
            c = c * max(0.0, s)
        return c, {"status": int(res.status), "message": str(res.message), "iterations": nit, "constraint_rows": int(len(rows)), "pulled_back": round(1.0 - s, 6),
                   "seconds": round(__import__("time").perf_counter() - t_0, 1)}

    def stats(c):
        phi = law + D0 @ c
        ang = view_angle_to_line(phi, theta)
        for it in items:
            x = ang[it["m"]]
            it["median"], it["share"] = float(np.median(x)), float(np.mean(np.abs(x - 90.0) <= cfg["share_tol_deg"]))
        return phi

    def meets(it, internal=0.5):
        if it.get("kind") == "gap":                      # the fast pass's shaping terms: not a target
            return True
        tol = float(cfg["profile_tol_deg"])
        best = it.get("best_alone")
        if best is not None and abs(best - 90.0) > tol:
            tol = abs(best - 90.0) + float(cfg["bound_tol_deg"])
        return abs(it["median"] - 90.0) <= tol - internal and it["share"] >= cfg["share_min"] + 0.02

    for it in items:
        it["wt"] = it["w0"]
    c, info = qp(items)
    phi = stats(c)
    rounds = [{"round": 0, **info, "missed": [it["name"] for it in items if not meets(it)]}]
    for rnd in range(1, int(cfg["rounds"]) + 1):
        miss = [it for it in items if not meets(it)]
        if not miss:
            break
        for it in miss:
            if it.get("best_alone") is None:
                ca, _ = qp([dict(it, wt=1.0)], x0=c)            # from the joint solution (feasible, and near this window's own best)
                x = view_angle_to_line(law + D0 @ ca, theta)[it["m"]]
                it["best_alone"] = float(np.median(x))
                it["best_alone_share"] = float(np.mean(np.abs(x - 90.0) <= cfg["share_tol_deg"]))
        fix = [it for it in miss if not meets(it)]
        if not fix:
            break
        for it in fix:
            it["wt"] *= 2.0
        c2, info2 = qp(items)
        prev = [(it["median"], it["share"]) for it in items]
        phi2 = stats(c2)
        if sum(1 for it in items if meets(it)) < sum(1 for it, (md, shr) in zip(items, prev) if meets(dict(it, median=md, share=shr))):
            for it, (md, shr) in zip(items, prev):           # the re-weighting made it worse: keep the previous solution
                it["median"], it["share"] = md, shr
            rounds.append({"round": rnd, **info2, "kept_previous": True})
            break
        c, phi = c2, phi2
        rounds.append({"round": rnd, **info2, "missed": [it["name"] for it in items if not meets(it)]})
    rate = W + D1 @ c
    acc = D2 @ c
    return {"phi": phi, "rate": rate, "accel": acc, "coef": c, "knots": nk, "rounds": rounds, "inside": inw}


def profile_track(doc: dict, k: int, line: PelvisLine, cb: dict, cfg: dict = V43_TRACK, log=print) -> tuple[dict, dict]:
    """v4.3: the azimuth track of the slow-orbit shot k (the run's camera block): the table the director plays, and the build's report"""
    segs = doc["segments"]
    shots = shots_of(segs)
    sh = shots[k]
    cam = segs[sh.segs[0]]["camera"]
    run = (cam.get("v4") or {}).get("run")
    tr_in = cam.get("transition_in") or {}
    if k + 1 >= len(shots) or not tr_in:
        raise fr.RecapError(f"v4.3: run {run} needs a way in and a way out (the track lies between them)")
    ntr = segs[shots[k + 1].segs[0]]["camera"]["transition_in"]
    a = round(sh.f0 + float(tr_in["start_s"]) + float(tr_in["duration_s"]), 6)      # the way in has arrived
    b = round(shots[k + 1].f0 + float(ntr["start_s"]), 6)                            # the way out starts
    A0, W, F0 = float(cam["world_azimuth_deg"]), float(cam["drift_deg_s"]), float(sh.f0)
    ease = V42_CLOSER_EASE_S
    win_in = tuple(cam["closer_in_s"]) if cam.get("closer_in_s") and abs(float(cam["closer_in_s"][0]) - a) < 1e-3 else (a, a + ease)
    win_out = tuple(cam["closer_out_s"]) if cam.get("closer_out_s") and abs(float(cam["closer_out_s"][1]) - b) < 1e-3 else (b - ease, b)
    win_in, win_out = (float(win_in[0]), float(win_in[1])), (float(win_out[0]), float(win_out[1]))
    hz = float(cfg["hz"])
    i0, i1 = int(math.floor(a * hz + 1e-9)), int(math.ceil(b * hz - 1e-9))
    t = np.arange(i0, i1 + 1) / hz
    law = A0 + W * (t - F0)
    d = _dance_vec(doc, t)
    theta = line.theta_at(d)
    wins = v43_windows(doc, sh, a, b, cb, cfg)
    ja = int(np.argmin(np.abs(t - a)))
    psi_a = float(law[ja] - theta[ja])
    first = 90.0 + 180.0 * round((psi_a - 90.0) / 180.0)                  # the profile nearest the eye where the way out of the phone leaves it
    second = first + 180.0 * math.copysign(1.0, W)                        # ... and the other one, the orbit's own way round
    items = []
    for r in wins["windows"]:
        for p0, p1, side in r["scored"]:
            m = (t >= p0 - 1e-9) & (t <= p1 + 1e-9)
            items.append({"name": r["name"], "kind": r["kind"], "part": [p0, p1], "side": first if side == "first" else second, "m": m,
                          "w0": cfg["w_interval"] if r["kind"] == "interval" else cfg["w_segment"]})
    t1 = __import__("time").time()
    cross = wins["crossing"]
    rcap = np.where((t >= cross[0]) & (t <= cross[1]), float(cfg["rate_max_deg_s"]), float(cfg.get("rate_max_outside_deg_s", cfg["rate_max_deg_s"])))
    sol = solve_profile_track(t, law, W, theta, win_in, win_out, items, cfg, rate_max=rcap)
    log(f"v4.3 run {run}: profile track {a:.3f}-{b:.3f} s ({len(t)} samples, {sol['knots']} knot spans) solved in {__import__('time').time() - t1:.1f} s, "
        f"rounds {[(x['round'], x.get('missed')) for x in sol['rounds']]}")
    ths = np.interp(d, line.t, np.convolve(np.pad(line.theta, 15, mode="edge"), np.ones(31) / 31.0, mode="valid"))
    fp = cfg.get("fast_pass")
    fast_rec = None
    if fp:
        # the optional faster pass (owner decision): around the decided track's line crossing the turn may reach fp's rate, and the gap's two
        # sides are scored on their own profile (weight w_gap), so the solver lingers in profile and runs through the line fast
        cr0 = [float(t[i + 1]) for i in np.nonzero(np.diff(np.floor((sol["phi"] - ths) / 180.0)))[0] if cross[0] <= t[i + 1] <= cross[1]]
        if len(cr0) != 1:
            raise fr.RecapError(f"v4.3 run {run}: fast pass: the decided track crosses the line {len(cr0)} times in the gap (one expected)")
        tx, hw_ = cr0[0], float(fp["half_s"])
        fwin = (round(max(cross[0], tx - hw_), 3), round(min(cross[1], tx + hw_), 3))
        rcap = np.where((t >= fwin[0]) & (t <= fwin[1]), float(fp["rate_max_deg_s"]), rcap)
        items_fp = list(items)
        for p0, p1, sd in ((cross[0], fwin[0], first), (fwin[1], cross[1], second)):
            if p1 - p0 >= float(cfg["min_part_s"]):
                items_fp.append({"name": f"the gap {'before' if sd == first else 'after'} the fast pass", "kind": "gap", "part": [round(p0, 3), round(p1, 3)],
                                 "side": sd, "m": (t >= p0 - 1e-9) & (t <= p1 + 1e-9), "w0": float(fp["w_gap"])})
        t2 = __import__("time").time()
        sol = solve_profile_track(t, law, W, theta, win_in, win_out, items_fp, cfg, rate_max=rcap)
        items = [it for it in items_fp if it["kind"] != "gap"]
        fast_rec = {"line_at_s_decided": round(tx, 3), "film": list(fwin), "rate_max_deg_s": float(fp["rate_max_deg_s"]), "w_gap": float(fp["w_gap"]),
                    "rule": "within film the turn may reach rate_max_deg_s (an owner decision beyond the decided 7 deg/s); the rest of the crossing <= "
                            "rules.rate_max_deg_s_gap"}
        log(f"v4.3 run {run}: fast pass {fwin} at <= {fp['rate_max_deg_s']:g} deg/s solved in {__import__('time').time() - t2:.1f} s, "
            f"rounds {[(x['round'], x.get('missed')) for x in sol['rounds']]}")
    phi = sol["phi"]
    # the eye lift inside the window: the inline lift's twin (the damped pelvises, the Hann window over +-1.2 s), the block's cap x the closer factor
    # (as Unity's live lift) outside the crossing; inside it cfg['crossing_lift']: "none" = no lift (fix round 1: it cannot show the dancer behind),
    # "raised" = the cap raised to the ground band's top (lift_top_m - the eye's height at full closeness; the first build, kept for the record)
    half = 36
    te = np.arange(i0 - half, i1 + half + 1) / hz
    law_e = A0 + W * (te - F0)
    phi_e = law_e.copy()
    phi_e[half:half + len(t)] = phi
    inside_e = (te > a + 1e-9) & (te < b - 1e-9)
    phi_e[~inside_e] = law_e[~inside_e]
    pl, pf = line.damped("lead", doc, te), line.damped("follow", doc, te)
    L = inline_lift_twin(phi_e, pl, pf)
    h_full = float(cam["height_m"])
    cap_raised = max(0.0, float(cfg["lift_top_m"]) - h_full)
    cap_reg = np.array([float(cam.get("inline_lift_max_m", 0.0)) * closer_factor(cam, float(x)) for x in te])
    in_cross_e = (te >= cross[0]) & (te <= cross[1])
    mode = cfg.get("crossing_lift", "raised")
    if mode not in ("none", "raised"):
        raise fr.RecapError(f"v4.3: crossing_lift {mode!r} (none | raised)")
    cap_cross = cap_raised if mode == "raised" else 0.0
    lift = hann_mean(np.minimum(L, np.where(in_cross_e, cap_cross, cap_reg)), half)
    lift_raised = hann_mean(np.minimum(L, np.where(in_cross_e, cap_raised, cap_reg)), half)
    lift_none = hann_mean(np.minimum(L, np.where(in_cross_e, 0.0, cap_reg)), half)
    for x_ in (lift, lift_raised, lift_none):
        x_[(t <= a + 1e-9) | (t >= b - 1e-9)] = 0.0
    lift_alt, alt_mode = (lift_none, "none") if mode == "raised" else (lift_raised, "raised")     # the other choice, for the review's comparison
    lift_law = hann_mean(np.minimum(inline_lift_twin(law_e, pl, pf), np.array([float(cam.get("inline_lift_max_m", 0.0)) * closer_factor(cam, float(x)) for x in te])), half)
    edge_live = float(max(np.max(hann_mean(np.minimum(inline_lift_twin(law_e, pl, pf), 9.0), half)[np.abs(t - a) <= 1.3]),
                          np.max(hann_mean(np.minimum(inline_lift_twin(law_e, pl, pf), 9.0), half)[np.abs(t - b) <= 1.3])))
    ang = view_angle_to_line(phi, theta)
    psi = phi - theta
    cr_t = [round(float(t[i]), 3) for i in np.nonzero(np.diff(np.floor((phi - ths) / 180.0)))[0]]
    rate, acc = sol["rate"], sol["accel"]
    inw = sol["inside"]
    fast = inw & (rate * math.copysign(1.0, W) > 6.0 + 1e-9)
    fast_runs = []
    for i in np.nonzero(fast)[0]:
        if fast_runs and i == fast_runs[-1][1] + 1:
            fast_runs[-1][1] = i
        else:
            fast_runs.append([i, i])
    cmask = inw & (t >= cross[0]) & (t <= cross[1])
    block = {"schema": cfg["schema"], "revision": cfg["revision"], "run": run, "segments": [segs[sh.segs[0]]["id"], segs[sh.segs[-1]]["id"]],
             "film": [a, b], "t0": i0 / hz, "dt": 1.0 / hz, "hz": hz, "i0": i0, "n": len(t), "interp": "linear",
             "deg": [round(float(x), 5) for x in phi], "lift_m": [round(float(x), 4) for x in lift],
             "law": {"world_azimuth_deg": A0, "drift_deg_s": W, "F0": F0, "rule": "phi(film) = world_azimuth_deg + drift_deg_s x (film - F0): the block's own law, "
                                                                            "played outside film[0] < film < film[1]"},
             "blend": {"in_s": list(win_in), "out_s": list(win_out), "rule": "deg = law + w (E - law), w = Ramp(in_s) x (1 - Ramp(out_s)) (smootherstep): the law "
                                                                              "with zero 1st and 2nd derivative difference at both ends (C2)"},
             "rules": {k2: cfg[k2] for k2 in ("rate_min_deg_s", "rate_max_deg_s", "rate_max_outside_deg_s", "accel_max_deg_s2", "profile_tol_deg", "share_tol_deg",
                                              "share_min", "lift_top_m", "crossing_lift")}
             | ({"exception": "the crossing turns at up to rate_max_deg_s (7): the owner-approved exception to v4.1's 2.5-6 deg/s one-rate orbit band"} if not fast_rec else
                {"rate_max_deg_s": fast_rec["rate_max_deg_s"], "rate_max_deg_s_gap": cfg["rate_max_deg_s"],
                 "exception": f"the crossing turns at up to {cfg['rate_max_deg_s']:g} deg/s (the owner-approved exception to v4.1's 2.5-6 band) and inside "
                              f"fast_pass.film at up to {fast_rec['rate_max_deg_s']:g} deg/s (the fast pass: an owner decision beyond the decided "
                              f"{cfg['rate_max_deg_s']:g})", "fast_pass": fast_rec}),
             "profiles": {"first_deg": first, "second_deg": second, "rule": "psi = the eye's azimuth about the couple minus the lead -> follow pelvis line's "
                          "(world deg, unwrapped); first = the profile nearest psi where the way in leaves the eye, second = first + 180 the orbit's way"},
             "crossing": {"film": cross, "line_at_s": cr_t, "rate_max_deg_s": round(float(np.max(np.abs(rate[cmask]))), 3) if cmask.any() else None,
                          "over_6_deg_s": [[round(float(t[i]), 3), round(float(t[j]), 3)] for i, j in fast_runs],
                          "lift": cfg.get("crossing_lift", "raised"), "lift_cap_m": round(cap_cross, 3), "lift_max_m": round(float(lift.max()), 4),
                          "raised_lift_max_m": round(float(lift_raised.max()), 4)} | ({"fast_pass": fast_rec} if fast_rec else {}),
             "line": {"source": "the installed motion: smplx_lead.bin / smplx_follow.bin joint 0 (the pelvis)", "lead_sha256": line.sha["lead"],
                      "follow_sha256": line.sha["follow"]},
             "windows": [{k2: v for k2, v in r.items()} for r in wins["windows"]],
             "lift_rule": "inside film[0] < film < film[1] the eye's inline lift = lift_m (m, linear between the samples), replacing the Hann-weighted live "
                          "InlineLift (its cap and the closer factor already applied); its python twin, inside the crossing "
                          + ("with the cap raised to lift_top_m - height_m" if cap_cross > 0 else "with no lift (crossing.lift 'none': behind the taller lead "
                             "a lift up to the band's top cannot show the follow; the eye keeps its height)")}
    for r in block["windows"]:
        its = [it for it in items if it["name"] == r["name"]]
        if its:
            r["side"] = ["first" if it["side"] == first else "second" for it in its]
            r["planned"] = [{"part": it["part"], "median_deg": round(it["median"], 2), "share_within_tol": round(it["share"], 3),
                             "best_alone_median_deg": None if it.get("best_alone") is None else round(it["best_alone"], 2)} for it in its]
    rep = {"run": run, "film": [a, b], "samples": len(t), "solver": sol["rounds"], "knot_spans": sol["knots"],
           "rate": {"min": round(float(np.min(rate[inw] * math.copysign(1.0, W))), 3), "max": round(float(np.max(rate[inw] * math.copysign(1.0, W))), 3)},
           "accel_abs_max": round(float(np.max(np.abs(acc[inw]))), 3), "crossings_at_s": cr_t, "first_profile": first, "second_profile": second,
           "psi_at_way_ends": [round(psi_a, 2), round(float(psi[-1]), 2)], "lift_max_m": round(float(lift.max()), 4), "lift_v42_law_max_m": round(float(lift_law.max()), 4),
           "live_lift_at_window_edges_m": round(edge_live, 4), "raised_lift_max_m": round(float(lift_raised.max()), 4), "fast_pass": fast_rec,
           "angle": ang, "t": t, "lift": lift, "lift_alt": lift_alt, "lift_alt_mode": alt_mode, "rate_s": rate, "accel_s": acc, "theta": theta, "law": law}
    return block, rep


def v43_doc(doc42: dict, line: PelvisLine | None = None, cap_dir: Path = CAP_DIR, cfg: dict = V43_TRACK, log=print) -> tuple[dict, dict]:
    """the v4.3 direction from the v4.2 one: every run in cfg['runs'] gets its profile track (flow.azimuth_tracks, camera.azimuth_track on each
    of its segments); nothing else changes. Returns (doc, {track id: build report})"""
    doc = copy.deepcopy(doc42)
    line = line or PelvisLine(cap_dir)
    cb = load_counterbalance(cap_dir)
    segs = doc["segments"]
    tracks, reps = {}, {}
    for run in cfg["runs"]:
        shots = shots_of(segs)
        ks = [k for k, sh in enumerate(shots) if sh.kind == "3d" and (segs[sh.segs[0]]["camera"].get("v4") or {}).get("run") == run
              and (segs[sh.segs[0]]["camera"].get("v4") or {}).get("role") == "slow_orbit"]
        if len(ks) != 1:
            raise fr.RecapError(f"v4.3: run {run} is {len(ks)} slow-orbit shots (one expected)")
        block, rep = profile_track(doc, ks[0], line, cb, cfg, log=log)
        tid = f"run{run}"
        tracks[tid] = block
        reps[tid] = rep
        for i in shots[ks[0]].segs:
            segs[i]["camera"]["azimuth_track"] = tid
    doc["flow"]["azimuth_tracks"] = tracks
    doc["flow"]["revision"] = "4.3"
    doc["flow"]["schema"] = doc["flow"]["schema"] + FLOW_SCHEMA_V43
    doc["created"] = ("generated by dancecap.film_recap_flow build --version v4.3 (the v4.2 plan + the initial counterbalances as profile two-shot orbits: "
                      "a planned azimuth track for the slow orbit of chapters 1-3) on top of dancecap.film_recap; edit the rules there")
    doc.setdefault("recap_film", {}).setdefault("v4", {})["v43"] = {"owner": V43_OWNER, "tracks": {tid: {"run": b_["run"], "film": b_["film"], "crossing": b_["crossing"]}
                                                                                                  for tid, b_ in tracks.items()}}
    return doc, reps


def v43_cfg() -> V4Cfg:
    return dataclasses_replace(v42_cfg(), version="4.3")


def _strip_tracks(doc: dict) -> dict:
    """a direction without what v4.3 adds (the tracks, the camera keys, the revision / schema / provenance and the re-measured numbers)"""
    d = copy.deepcopy(doc)
    fl_ = d.get("flow") or {}
    for k2 in ("azimuth_tracks", "revision", "schema"):
        fl_.pop(k2, None)
    (fl_.get("v4") or {}).pop("measured", None)
    d.pop("created", None)
    ((d.get("recap_film") or {}).get("v4") or {}).pop("v43", None)
    for s in d["segments"]:
        s["camera"].pop("azimuth_track", None)
    return d


def track_report(doc: dict, met: dict, met_h: dict | None, doc42: dict | None, met42: dict | None, met42_h: dict | None, line: PelvisLine | None = None,
                 reps: dict | None = None) -> dict:
    """the v4.3 numbers on the sampled plan (PoseModel, 30 Hz): per tracked run and window the view's angle to the couple's line (v4.2 / v4.3, both
    aspects), the line crossings, the rate band, the lift and the eye's height; and the frame-exact comparison with v4.2 (every frame outside the
    tracks' windows, the ways in particular) plus the direction's own diff (v4.3 = v4.2 + the track keys)"""
    tracks = (doc.get("flow") or {}).get("azimuth_tracks") or {}
    out: dict = {"tracks": {}}
    if not tracks or met.get("_sample") is None:
        return out
    line = line or PelvisLine()
    sm = met["_sample"]
    t = sm["t"]
    d = _dance_vec(doc, t)
    theta = line.theta_at(d)

    def angles(s):
        dr = s["dir"]
        return view_angle_to_line(np.degrees(np.arctan2(-dr[:, 2], -dr[:, 0])), theta)
    ang = {"9:16": angles(sm)}
    if met_h is not None and met_h.get("_sample") is not None:
        ang["16:9"] = angles(met_h["_sample"])
    ang42 = {}
    if met42 is not None and met42.get("_sample") is not None:
        ang42["9:16"] = angles(met42["_sample"])
        if met42_h is not None and met42_h.get("_sample") is not None:
            ang42["16:9"] = angles(met42_h["_sample"])
    shots = shots_of(doc["segments"])
    f0s = np.array([s.f0 for s in shots])
    shot_of = np.clip(np.searchsorted(f0s, t, side="right") - 1, 0, len(shots) - 1)
    # the lead's right (his right shoulder - his left), for the side the eye is on
    lr = line.joint_at("lead", line.R_SH, d) - line.joint_at("lead", line.L_SH, d)
    lr_az = np.arctan2(lr[:, 2], lr[:, 0])
    eye_rel = sm["eye"] - sm["c"]
    on_right = np.cos(np.arctan2(eye_rel[:, 2], eye_rel[:, 0]) - lr_az) > 0.0
    # the two heads within 4 deg of each other in the picture (one hides the other), per version and aspect
    hs = {("v43", "9:16"): head_sep_deg(line, d, sm["eye"])}
    for lab, asp, M in (("v43", "16:9", met_h), ("v42", "9:16", met42), ("v42", "16:9", met42_h)):
        if M is not None and M.get("_sample") is not None and len(M["_sample"]["t"]) == len(t):
            hs[(lab, asp)] = head_sep_deg(line, d, M["_sample"]["eye"])
    for tid, tb in tracks.items():
        a, b = float(tb["film"][0]), float(tb["film"][1])
        k = next((kk for kk, s in enumerate(shots) if doc["segments"][s.segs[0]]["camera"].get("azimuth_track") == tid), None)
        inside = (t > a) & (t < b)
        settled = (shot_of == k) & (sm["way"] < 0) & ~sm["ph_hold"] & inside
        W = float(tb["law"]["drift_deg_s"])
        sg = math.copysign(1.0, W)
        rate = sm["az_rate"]
        acc = np.gradient(np.nan_to_num(rate)) * float(1.0 / (t[1] - t[0]))
        ok = settled & np.isfinite(rate)
        ok_a = ok & np.roll(ok, 1) & np.roll(ok, -1)
        rows = []
        for w_ in tb.get("windows", []):
            x0, x1 = w_["film"]
            row = {"name": w_["name"], "kind": w_["kind"], "film": w_["film"], "scored": w_.get("scored", []), "not_scored": w_.get("not_scored", ""),
                   "best_alone": [p.get("best_alone_median_deg") for p in w_.get("planned", [])]}
            mw = (t >= x0) & (t <= x1)
            for lab, A in (("v42", ang42), ("v43", ang)):
                for asp, aa in A.items():
                    x = aa[mw]
                    if len(x):
                        row[f"{lab}_{asp}"] = {"min": round(float(x.min()), 1), "median": round(float(np.median(x)), 1), "max": round(float(x.max()), 1),
                                               "within_25": round(float(np.mean(np.abs(x - 90) <= 25)), 3)}
            sc = np.zeros(len(t), bool)
            for p0, p1, _ in w_.get("scored", []):
                sc |= (t >= p0 - 1e-9) & (t <= p1 + 1e-9)
            if sc.any():
                for asp, aa in ang.items():
                    x = aa[sc]
                    row[f"scored_{asp}"] = {"median": round(float(np.median(x)), 2), "within_tol": round(float(np.mean(np.abs(x - 90) <= tb["rules"]["share_tol_deg"])), 3),
                                            "frames": int(sc.sum())}
                for asp, aa in ang42.items():
                    x = aa[sc]
                    row[f"scored_v42_{asp}"] = {"median": round(float(np.median(x)), 2), "within_tol": round(float(np.mean(np.abs(x - 90) <= tb["rules"]["share_tol_deg"])), 3)}
            if mw.any():
                row["eye_on_lead_right"] = round(float(np.mean(on_right[mw])), 3)
                row["heads_under_4deg"] = {f"{lab}_{asp}": round(float(np.mean(x[mw] < 4.0)), 3) for (lab, asp), x in hs.items()}
            rows.append(row)
        # the line crossings of the settled orbit (the eye's azimuth about the couple's centre minus the 1-s mean line, through 0 / 180)
        ths = np.interp(d, line.t, np.convolve(np.pad(line.theta, 15, mode="edge"), np.ones(31) / 31.0, mode="valid"))
        run_m = (shot_of == k) & (sm["way"] < 0) & ~sm["ph_hold"]
        ii = np.nonzero(run_m)[0]
        psi = sm["az"][ii] - ths[ii]
        cr = [round(float(t[ii[j + 1]]), 3) for j in np.nonzero(np.diff(np.floor(psi / 180.0)))[0] if ii[j + 1] == ii[j] + 1]
        psi_raw = sm["az"][ii] - theta[ii]
        cr_raw = [round(float(t[ii[j + 1]]), 3) for j in np.nonzero(np.diff(np.floor(psi_raw / 180.0)))[0] if ii[j + 1] == ii[j] + 1]
        ivs = [w_["film"] for w_ in tb.get("windows", []) if w_["kind"] == "interval"]
        cr_in = [x for x in cr if any(p0 <= x <= p1 for p0, p1 in ivs)]
        eh = sm["eye_h"]
        row_t = {"film": [a, b], "windows": rows, "crossings_at_s": cr, "crossings_raw_line_at_s": cr_raw, "crossings_in_intervals": cr_in,
                 "rate_its_way": {"min": round(float(np.min(sg * rate[ok])), 3), "max": round(float(np.max(sg * rate[ok])), 3)} if ok.any() else None,
                 "accel_abs_max": round(float(np.max(np.abs(acc[ok_a]))), 3) if ok_a.any() else None,
                 "eye_h_max_m": round(float(np.max(eh[inside])), 3) if inside.any() else None,
                 "eye_h_max_16x9_m": round(float(np.max(met_h["_sample"]["eye_h"][inside])), 3) if (met_h is not None and inside.any()) else None,
                 "lift_max_m": round(float(np.max(tb.get("lift_m") or [0.0])), 4)}
        # the crossing: its fastest turn, the time above v4.1's 6-deg/s band, the eye's vertical speed
        cx = (t >= tb["crossing"]["film"][0]) & (t <= tb["crossing"]["film"][1]) & ok
        if cx.any():
            row_t["crossing"] = {"film": tb["crossing"]["film"], "rate_max_deg_s": round(float(np.max(sg * rate[cx])), 3),
                                 "rate_mean_deg_s": round(float(np.mean(sg * rate[cx])), 3),
                                 "over_6_deg_s_s": round(float(np.sum(sg * rate[cx] > 6.0 + 1e-9)) / 30.0, 2),
                                 "outside_it_rate_max_deg_s": round(float(np.max(sg * rate[ok & ~cx])), 3) if (ok & ~cx).any() else None,
                                 "outside_it_over_6_deg_s_s": round(float(np.sum(sg * rate[ok & ~cx] > 6.0 + 1e-9)) / 30.0, 2),
                                 "eye_vertical_speed_max_m_s": round(float(np.max(np.abs(np.gradient(eh[inside]) * 30.0))), 3) if inside.any() else None}
            fpw = ((tb.get("rules") or {}).get("fast_pass") or {}).get("film")
            if fpw:
                g_ = cx & ~((t >= fpw[0]) & (t <= fpw[1]))
                row_t["crossing"]["fast_pass"] = {"film": fpw, "rate_max_in_it_deg_s": round(float(np.max(sg * rate[cx & ~g_])), 3) if (cx & ~g_).any() else None,
                                                  "rate_max_rest_of_gap_deg_s": round(float(np.max(sg * rate[g_])), 3) if g_.any() else None,
                                                  "over_7_deg_s_s": round(float(np.sum(sg * rate[cx] > 7.0 + 1e-9)) / 30.0, 2)}
        # the far dancer's head behind the near one (plan estimates: the far head within 0.2 m of the near dancer's head / neck / spine / pelvis
        # polyline; the two heads under 3 / 4 deg apart), over the crossing: v4.2's law (its own crossing in the same orbit), v4.3, and v4.3 with the
        # first build's raised lift (when the build's reports are at hand)
        rp_ = (reps or {}).get(tid) or {}
        row_t["far_head_hidden_s"] = _hidden_heads(doc, met, met42, line, k, tb, lift_alt=rp_.get("lift_alt"), alt_mode=rp_.get("lift_alt_mode"))
        out["tracks"][tid] = row_t
    # frame-exact: every frame outside the tracks' windows = v4.2's (eye, view direction, lens), in both aspects; the ways in particular
    if met42 is not None and met42.get("_sample") is not None:
        wins = [(float(tb["film"][0]), float(tb["film"][1])) for tb in tracks.values()]
        inside_any = np.zeros(len(t), bool)
        for a, b in wins:
            inside_any |= (t > a) & (t < b)
        ex = {}
        for asp, A, B in (("9:16", sm, met42["_sample"]), ("16:9", (met_h or {}).get("_sample"), (met42_h or {}).get("_sample"))):
            if A is None or B is None or len(A["t"]) != len(B["t"]):
                continue
            dif = np.max(np.abs(A["eye"] - B["eye"]), axis=1) + np.max(np.abs(A["dir"] - B["dir"]), axis=1) + np.abs(A["fov"] - B["fov"])
            way = B["way"] >= 0
            ex[asp] = {"frames": int(len(t)), "differ": int(np.sum(dif > 0)), "differ_outside_tracks": int(np.sum((dif > 0) & ~inside_any)),
                       "way_frames": int(way.sum()), "way_frames_differ": int(np.sum((dif > 0) & way)), "max_diff_outside_tracks": float(np.max(dif[~inside_any])),
                       "first_last_differing_s": [round(float(t[dif > 0][0]), 4), round(float(t[dif > 0][-1]), 4)] if (dif > 0).any() else None}
        out["frame_exact"] = ex
    if doc42 is not None:
        out["direction_diff"] = {"same_but_the_track": json.dumps(_strip_tracks(doc), sort_keys=True) == json.dumps(_strip_tracks(doc42), sort_keys=True),
                                 "v42_revision": (doc42.get("flow") or {}).get("revision")}
    out["motion"] = {"lead_sha256": line.sha["lead"], "follow_sha256": line.sha["follow"]}
    return out


def head_sep_deg(line: PelvisLine, d, eye: np.ndarray) -> np.ndarray:
    """the angle (deg) between the two dancers' head joints seen from the eye (n x 3, capture coordinates) at dance times d: under ~3-4 deg the
    heads overlap in the picture (a head is ~0.2 m wide: ~3.5 deg at the orbit's 3-4 m)"""
    vl = line.joint_at("lead", line.HEAD, d) - eye
    vf = line.joint_at("follow", line.HEAD, d) - eye
    c = np.sum(vl * vf, axis=1) / np.maximum(1e-9, np.linalg.norm(vl, axis=1) * np.linalg.norm(vf, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


FACE_HIDDEN_R_M = 0.10     # her head centre within this of his (at his depth) = her face hidden: calibrated (fix round 1) on editor stills every
                           # 0.5 s over the crossing, judged by Jev ("the woman's face can be seen") and looked at - within ~0.5 s in both aspects


def _hidden_heads(doc: dict, met: dict, met42: dict | None, line: PelvisLine, k: int, tb: dict, r_m: float = 0.2,
                  lift_alt: np.ndarray | None = None, alt_mode: str | None = None) -> dict:
    """the dancer behind hidden by the one in front, over the crossing +-3 s (plan estimates on the PoseModel eye): (1) her FACE hidden - the far
    head's centre within FACE_HIDDEN_R_M of the near head's centre at the near depth (s, and its longest stretch); (2) the far head within r_m of the
    near dancer's head / neck / spine / pelvis polyline (s); (3) the two heads closer than 3 / 4 deg in the picture (over the crossing itself). For
    v4.2's law, v4.3 as planned and - lift_alt, the other crossing-lift choice (alt_mode) on the table's samples - v4.3 with that lift instead"""
    sm = met["_sample"]
    t = sm["t"]
    lo, hi = tb["crossing"]["film"]
    m = (t >= lo - 3.0) & (t <= hi + 3.0)
    idx = np.nonzero(m)[0]
    tw = t[idx]
    d = _dance_vec(doc, tw)
    cx = (tw >= lo) & (tw <= hi)
    J = (line.HEAD, line.NECK, line.SPINE3, line.PELVIS)
    P = {role: np.stack([line.joint_at(role, j, d) for j in J], axis=1) for role in ("lead", "follow")}   # (n, 4, 3)
    tt = float(tb["t0"]) + float(tb["dt"]) * np.arange(int(tb["n"]))
    inside = (tw > float(tb["film"][0])) & (tw < float(tb["film"][1]))
    lift = np.where(inside, np.interp(tw, tt, np.asarray(tb.get("lift_m") or [0.0] * int(tb["n"]), float)), 0.0)

    def longest(mask):
        run, best = None, None
        for i, x in enumerate(mask):
            if x:
                run = [i, i] if run is None else [run[0], i]
                if best is None or run[1] - run[0] > best[1] - best[0]:
                    best = list(run)
            else:
                run = None
        return None if best is None else [round(float(tw[best[0]]), 2), round(float(tw[best[1]]), 2), round((best[1] - best[0] + 1) / 30.0, 2)]

    def hidden(eye, look):
        poly, face = np.zeros(len(eye), bool), np.zeros(len(eye), bool)
        for i in range(len(eye)):
            fwd = _unit(look[i] - eye[i])
            right = _unit(np.cross(fwd, np.array([0.0, 1.0, 0.0])))
            up = np.cross(right, fwd)
            z = {role: (P[role][i] - eye[i]) @ fwd for role in P}
            near, far = ("lead", "follow") if z["lead"][0] < z["follow"][0] else ("follow", "lead")
            zf = z[far][0]
            if zf <= 0.1:
                continue
            q = np.array([(P[far][i, 0] - eye[i]) @ right, (P[far][i, 0] - eye[i]) @ up]) / zf
            pts = np.stack([(P[near][i] - eye[i]) @ right, (P[near][i] - eye[i]) @ up], axis=1) / z[near][:, None]
            best = 1e9
            for j in range(len(J) - 1):
                p, q2 = pts[j], pts[j + 1]
                v = q2 - p
                u = float(np.clip(np.dot(q - p, v) / max(1e-12, np.dot(v, v)), 0.0, 1.0))
                best = min(best, float(np.linalg.norm(q - (p + u * v))))
            poly[i] = best * float(np.mean(z[near])) < r_m
            face[i] = float(np.linalg.norm(q - pts[0])) * float(z[near][0]) < FACE_HIDDEN_R_M
        return {"face_hidden_s": round(float(face.sum()) / 30.0, 2), "face_hidden_longest_s": longest(face), "behind_body_s": round(float(poly.sum()) / 30.0, 2)}

    def heads(eye):
        sep_w = head_sep_deg(line, d, eye)
        sep = sep_w[cx]
        return {"heads_under_3deg_s": round(float(np.sum(sep < 3.0)) / 30.0, 2), "heads_under_4deg_s": round(float(np.sum(sep < 4.0)) / 30.0, 2),
                "heads_under_4deg_window_s": round(float(np.sum(sep_w < 4.0)) / 30.0, 2),
                "heads_sep_median_deg": round(float(np.median(sep)), 2) if len(sep) else None, "heads_sep_min_deg": round(float(np.min(sep)), 2) if len(sep) else None}

    def both(eye, look):
        return hidden(eye, look) | heads(eye)
    eye = sm["eye"][idx]
    look = sm["c"][idx] + np.array([0.0, 0.95, 0.0])
    out = {"window_s": [round(lo - 3.0, 2), round(hi + 3.0, 2)], "crossing_s": [lo, hi], "face_rule": f"her head centre within {FACE_HIDDEN_R_M} m of his "
           "(at his depth): calibrated on editor stills + Jev", "lift": tb["crossing"].get("lift", "raised"), "lift_max_m": round(float(lift.max()), 4),
           "v43": both(eye, look)}
    if lift_alt is not None:
        e1 = eye.copy()
        e1[:, 1] += np.where(inside, np.interp(tw, tt, np.asarray(lift_alt, float)), 0.0) - lift
        out["alt"] = {"lift": alt_mode, "lift_max_m": round(float(np.max(lift_alt)), 4)} | both(e1, look)
    if met42 is not None and met42.get("_sample") is not None:
        e42 = met42["_sample"]["eye"][idx]
        out["v42"] = both(e42, met42["_sample"]["c"][idx] + np.array([0.0, 0.95, 0.0]))
    return out


def v43_extra_metrics(met: dict, doc: dict, geo4: Geo4 | None, cfg4: V4Cfg, met_h: dict | None, doc42: dict | None, met42: dict | None = None,
                      met42_h: dict | None = None, line: PelvisLine | None = None, reps: dict | None = None) -> None:
    """v4.3's numbers into met['track'] (the v4.2 samples are computed from doc42 when not given; reps = the build's track reports, for the
    raised-lift comparison)"""
    if doc42 is not None and geo4 is not None and met42 is None:
        met42 = film_metrics(doc42, geo4, v42_cfg())
        met42_h = film_metrics(doc42, geo4, v42_cfg(), aspect="horizontal")
    met["track"] = track_report(doc, met, met_h, doc42, met42, met42_h, line, reps)


def _targets_v43(target, doc: dict, met: dict, cfg4: V4Cfg, notes: list[str], bad: list[str]) -> None:
    """the owner's v4.3 targets (2026-10-23): the initial counterbalances in profile, one crossing, the track's rate band, the ways and every frame
    outside the track = v4.2's"""
    tracks = (doc.get("flow") or {}).get("azimuth_tracks") or {}
    tm = (met.get("track") or {}).get("tracks") or {}
    if not tracks:
        target("10. the initial counterbalances as profile two-shot orbits (flow.azimuth_tracks)", "a track for run 2", "no track", False)
        return
    for tid, tb in tracks.items():
        r = tm.get(tid)
        if r is None:
            target(f"10. {tid}: measured", "the sampled plan", "not measured", False)
            continue
        ru = tb["rules"]
        tol, stol, smin = float(ru["profile_tol_deg"]), float(ru["share_tol_deg"]), float(ru["share_min"])
        btol = float(V43_TRACK["bound_tol_deg"])
        for w_ in r["windows"]:
            if not w_.get("scored"):
                notes.append(f"v4.3 {tid} {w_['name']} {w_['film'][0]}-{w_['film'][1]} s not scored ({w_['not_scored']}): angle to the line v4.2 median "
                             f"{(w_.get('v42_9:16') or {}).get('median')} -> v4.3 {(w_.get('v43_9:16') or {}).get('median')} deg")
                continue
            best = [x for x in w_.get("best_alone") or [] if x is not None]
            want_tol = tol
            why = ""
            if best and abs(best[0] - 90.0) > tol:
                want_tol = abs(best[0] - 90.0) + btol
                why = f" (no legal track can: its own best is {best[0]:.1f}, so within {btol:g} of that)"
            for asp in ("9:16", "16:9"):
                sc = w_.get(f"scored_{asp}")
                if sc is None:
                    continue
                v42 = w_.get(f"scored_v42_{asp}") or {}
                target(f"10. profile two-shot, {w_['name']} (scored {', '.join(f'{p0}-{p1}' for p0, p1, _ in w_['scored'])} s, {asp})",
                       f"median within {want_tol:.1f} of 90{why}; >= {100 * smin:.0f} % within {stol:g}",
                       f"median {sc['median']} deg, {100 * sc['within_tol']:.1f} % (v4.2 {v42.get('median')} deg, {100 * (v42.get('within_tol') or 0):.1f} %)",
                       abs(sc["median"] - 90.0) <= want_tol + 1e-6 and sc["within_tol"] >= smin - 1e-9)
                if why and asp == "9:16":
                    notes.append(f"v4.3 {tid} {w_['name']}: the profile target is out of reach for any track under the decided rules (rate >= "
                                 f"{ru['rate_min_deg_s']:g} deg/s its way from the way out of the phone, <= {ru['accel_max_deg_s2']:g} deg/s2): its own best "
                                 f"median is {best[0]:.1f} deg, delivered {sc['median']} deg")
        ncr = len(r["crossings_at_s"])
        target(f"10. {tid}: the couple's line crossed once, outside the counterbalance intervals", "1 crossing, 0 in an interval",
               f"{ncr} at {r['crossings_at_s']} s (raw line: {r['crossings_raw_line_at_s']}); in an interval: {r['crossings_in_intervals']}",
               ncr == 1 and not r["crossings_in_intervals"])
        rw, am = r.get("rate_its_way") or {}, r.get("accel_abs_max")
        cx = r.get("crossing") or {}
        r_out = ru.get("rate_max_outside_deg_s", ru["rate_max_deg_s"])
        target(f"10. {tid}: the track's turn about the couple (settled orbit frames, 30 Hz) - never reversing; only the crossing above v4.1's 6 deg/s "
               "(the owner-approved exception to its 2.5-6 band)", f">= {ru['rate_min_deg_s']:g} deg/s its way; <= {ru['rate_max_deg_s']:g} in the crossing, "
               f"<= {r_out:g} elsewhere; |accel| <= {ru['accel_max_deg_s2']:g} deg/s2",
               f"{rw.get('min')}..{rw.get('max')} deg/s; the crossing peaks at {cx.get('rate_max_deg_s')} deg/s ({cx.get('over_6_deg_s_s')} s above 6), "
               f"elsewhere at most {cx.get('outside_it_rate_max_deg_s')} deg/s; |accel| max {am}",
               bool(rw) and rw["min"] >= ru["rate_min_deg_s"] - 0.01 and rw["max"] <= ru["rate_max_deg_s"] + 0.01 and am is not None
               and am <= ru["accel_max_deg_s2"] + 0.02 and (cx.get("outside_it_rate_max_deg_s") or 0.0) <= r_out + 0.01)
        if ru.get("fast_pass"):
            fx = cx.get("fast_pass") or {}
            target(f"10. {tid}: the fast pass (an OWNER DECISION beyond the decided {ru.get('rate_max_deg_s_gap', 7.0):g} deg/s): faster only inside "
                   f"{ru['fast_pass']['film']} s", f"<= {ru['rate_max_deg_s']:g} deg/s in it, <= {ru.get('rate_max_deg_s_gap', 7.0):g} in the rest of the crossing",
                   f"{fx.get('rate_max_in_it_deg_s')} / {fx.get('rate_max_rest_of_gap_deg_s')} deg/s ({fx.get('over_7_deg_s_s')} s above 7)",
                   fx.get("rate_max_in_it_deg_s") is not None and fx["rate_max_in_it_deg_s"] <= ru["rate_max_deg_s"] + 0.01
                   and (fx.get("rate_max_rest_of_gap_deg_s") or 0.0) <= ru.get("rate_max_deg_s_gap", 7.0) + 0.01)
        target(f"10. {tid}: the eye inside the ground band through the crossing's lift", f"<= {ru['lift_top_m']} m",
               f"lift max {r['lift_max_m']} m; eye max {r['eye_h_max_m']} m (9:16), {r['eye_h_max_16x9_m']} m (16:9)",
               r["eye_h_max_m"] is not None and r["eye_h_max_m"] <= ru["lift_top_m"] + 1e-3 and (r["eye_h_max_16x9_m"] or 0) <= ru["lift_top_m"] + 1e-3)
        deg = np.asarray(tb["deg"], float)
        tt = float(tb["t0"]) + float(tb["dt"]) * np.arange(int(tb["n"]))
        law = lambda x: float(tb["law"]["world_azimuth_deg"]) + float(tb["law"]["drift_deg_s"]) * (x - float(tb["law"]["F0"]))   # noqa: E731
        e0 = abs(float(np.interp(tb["film"][0], tt, deg)) - law(tb["film"][0]))
        e1 = abs(float(np.interp(tb["film"][1], tt, deg)) - law(tb["film"][1]))
        lift_edge = max(abs(float(np.interp(tb["film"][0], tt, tb["lift_m"]))), abs(float(np.interp(tb["film"][1], tt, tb["lift_m"]))))
        target(f"10. {tid}: the table joins the law at its window's ends (no jump into / out of the track)", "<= 0.001 deg, lift 0 m",
               f"{e0:.6f} / {e1:.6f} deg, lift {lift_edge:.4f} m; {tb['n']} samples from {tb['t0']:.4f} s every {tb['dt']:.6f} s",
               e0 <= 1e-3 and e1 <= 1e-3 and lift_edge <= 1e-4 and len(tb["deg"]) == int(tb["n"]) == len(tb["lift_m"]))
        mo = (met.get("track") or {}).get("motion") or {}
        if mo and (mo.get("lead_sha256") != tb["line"]["lead_sha256"] or mo.get("follow_sha256") != tb["line"]["follow_sha256"]):
            notes.append(f"STALE: the installed motion changed since the {tid} track was planned (lead {tb['line']['lead_sha256'][:12]} -> {mo.get('lead_sha256', '')[:12]}, "
                         f"follow {tb['line']['follow_sha256'][:12]} -> {mo.get('follow_sha256', '')[:12]}): the pelvis is the same in every fit so far, but "
                         "re-run build --version v4 to be sure")
    ex = (met.get("track") or {}).get("frame_exact")
    if ex is None:
        target("10. the ways and every frame outside the tracks = v4.2's (frame-exact, PoseModel 30 Hz, both aspects)", "0 frames differ",
               f"no v4.2 baseline ({V42_DIRECTION} missing: run build --version v4.2)", False)
    else:
        target("10. the ways and every frame outside the tracks = v4.2's (frame-exact, PoseModel 30 Hz, both aspects)",
               "0 frames differ outside the tracks (0 way frames)",
               "; ".join(f"{asp}: {x['differ_outside_tracks']} of {x['frames']} outside ({x['way_frames_differ']} of {x['way_frames']} way frames; "
                         f"{x['differ']} differ inside, {x['first_last_differing_s']} s)" for asp, x in ex.items()),
               len(ex) == 2 and all(x["differ_outside_tracks"] == 0 and x["way_frames_differ"] == 0 for x in ex.values()))
    dd = (met.get("track") or {}).get("direction_diff")
    if dd is not None:
        target("10. the direction is v4.2's + the track (flow.azimuth_tracks, camera.azimuth_track, revision, provenance, re-measured numbers)", "same",
               "same" if dd["same_but_the_track"] else "DIFFERENT", dd["same_but_the_track"])


def v43_rows(res: dict) -> list[list]:
    """the per-window table v4.2 / v4.3 (the view's angle to the couple's line, 9:16; 16:9 next to it where it differs)"""
    out = []
    for tid, r in (res["metrics"].get("track") or {}).get("tracks", {}).items():
        for w_ in r["windows"]:
            a42, a43 = w_.get("v42_9:16") or {}, w_.get("v43_9:16") or {}
            h43 = w_.get("v43_16:9") or {}
            sc = w_.get("scored_9:16") or {}
            rows_t = [t_ for t_ in res["checks"]["targets"] if t_["target"].startswith(f"10. profile two-shot, {w_['name']} ")]
            st = "-" if not rows_t else ("OK" if all(t_["ok"] for t_ in rows_t) else "MISSED")
            out.append([w_["name"], f"{w_['film'][0]}-{w_['film'][1]}", f"{a42.get('min')} / {a42.get('median')} / {a42.get('max')}", f"{100 * (a42.get('within_25') or 0):.0f} %",
                        f"{a43.get('min')} / {a43.get('median')} / {a43.get('max')}", f"{100 * (a43.get('within_25') or 0):.0f} %",
                        "" if abs((h43.get("median") or 0) - (a43.get("median") or 0)) < 0.15 else f"16:9 median {h43.get('median')}",
                        (f"{', '.join(f'{p0}-{p1}' for p0, p1, _ in w_['scored'])}: median {sc.get('median')}" if w_.get("scored") else w_.get("not_scored", "")),
                        ", ".join(f"{x:.1f}" for x in w_.get("best_alone") or [] if x is not None) or "-", f"{100 * w_.get('eye_on_lead_right', 0):.0f} %", st])
    return out


def track_plot(res: dict, out: Path) -> Path | None:
    """run 2 over film time: the view's angle to the couple's line (v4.2 / v4.3) with the windows, the turn rate, the eye lift"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reps = res.get("track_build") or {}
    if not reps:
        return None
    tid, rp_ = next(iter(reps.items()))
    tb = res["doc"]["flow"]["azimuth_tracks"][tid]
    t = rp_["t"]
    fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1]})
    for w_ in tb["windows"]:
        col = "#e9c46a" if w_["kind"] == "interval" else "#d9e7f5"
        ax[0].axvspan(w_["film"][0], w_["film"][1], ymin=0 if w_["kind"] == "interval" else 0.5, ymax=0.5 if w_["kind"] == "interval" else 1.0, color=col, alpha=0.6, lw=0)
        ax[0].text(0.5 * (w_["film"][0] + w_["film"][1]), 182 if w_["kind"] == "segment" else -12, w_["name"].replace("counterbalance ", "CB "), fontsize=6.5, ha="center")
    ax[0].axvspan(*tb["crossing"]["film"], color="#cccccc", alpha=0.25, lw=0)
    ax[0].plot(t, view_angle_to_line(rp_["law"], rp_["theta"]), color="#9b9b9b", lw=1.0, label="v4.2 (the law)")
    ax[0].plot(t, rp_["angle"], color="#2a78d6", lw=1.5, label="v4.3 (the track)")
    for y, ls in ((90, "-"), (65, ":"), (115, ":"), (75, "--"), (105, "--")):
        ax[0].axhline(y, color="#555555", lw=0.6, ls=ls)
    ax[0].set_ylim(-20, 195)
    ax[0].set_ylabel("view to the line (deg)\n90 = profile two-shot")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title(f"v4.3 {tid}: the initial counterbalances as profile two-shot orbits (yellow: counterbalance intervals, blue: counterbalance-layer "
                    f"segments, grey: the crossing {tb['crossing']['film'][0]}-{tb['crossing']['film'][1]} s)", fontsize=9)
    ax[1].plot(t, rp_["rate_s"], color="#2a78d6", lw=1.2, label="v4.3")
    ax[1].axhline(float(tb["law"]["drift_deg_s"]), color="#9b9b9b", lw=1.0, label="v4.2")
    for y in (1.0, 7.0):
        ax[1].axhline(y, color="#c0392b", lw=0.7, ls="--")
    ax[1].axhspan(2.5, 6.0, color="#2a9d8f", alpha=0.08)
    ax[1].set_ylabel("turn about the\ncouple (deg/s)")
    ax[1].legend(loc="upper left", fontsize=8)
    ax[2].plot(t, rp_["lift"], color="#6250d6", lw=1.2)
    ax[2].set_ylabel("eye lift (m)")
    ax[2].set_xlabel("film s")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def build_v43(log=print, write: bool = True, hand_events: Path | None = None, out: Path | None = None, need_path: Path = NEED3D,
              cfg: dict = V43_TRACK) -> dict:
    """v4.3 = the v4.2 direction (build_v42, rebuilt: v4.2 is reproducible into v4/direction/v4.2/) + the profile track of the initial
    counterbalances' orbit (v43_doc; cfg = the track's rules: V43_TRACK, or with fast_pass the owner-decision variant). Measured on the plan in
    both aspects against v4.2 itself"""
    r = build_v42(log=log, write=False, hand_events=hand_events, need_path=need_path)
    cfg4 = v43_cfg()
    geo4, pov = r["geo4"], fr.load_pov(fr.Paths(), "auto")
    doc42, met42, met42_h = r["doc"], r["metrics"], r["metrics_16x9"]
    line = PelvisLine()
    doc, reps = v43_doc(doc42, line, cfg=cfg, log=log)
    met = film_metrics(doc, geo4, cfg4)
    v41_extra_metrics(met, doc, geo4, cfg4, r.get("need"), pov)
    met_h = film_metrics(doc, geo4, cfg4, aspect="horizontal")
    v42_extra_metrics(met, doc, geo4, cfg4, pov, met_h)
    v43_extra_metrics(met, doc, geo4, cfg4, met_h, doc42, met42, met42_h, line, reps)
    phq = PhoneQ41(pov, geo4, r["cfg4"]) if pov is not None else None
    checks = check_v4(doc, r["tl"], r["sp"], r["geo"], geo4, cfg4, None, met, phq)
    doc["flow"]["v4"]["measured"] = _metrics_public(met)
    res = dict(r)
    res.update({"doc": doc, "cfg4": cfg4, "metrics": met, "metrics_16x9": met_h, "checks": checks, "doc42": doc42, "metrics42": met42, "metrics42_16x9": met42_h,
                "track_build": reps, "line": line})
    if write:
        write_all_v43(res, Path(out or OUT_V4))
    return res


def review_v43_md(res: dict) -> str:
    doc, met, ch = res["doc"], res["metrics"], res["checks"]
    L = [f"# v4.3 direction review: {doc['title']} ({doc['name']})", "", V43_OWNER, "",
         "v4.3 = the v4.2 direction (every move, way, phone hold, lens, band, meter window, chapter, caption and the time map: `build_v42`, rebuilt) + ONE "
         "planned azimuth track for the slow orbit of chapters 1-3 (the initial counterbalances). v4.2 stays reproducible: `python -m dancecap.film_recap_flow "
         "build --version v4.2` -> `v4/direction/v4.2/` (direction.json byte-equal to the v4.2 that was recorded). Nothing was recorded.", "",
         ("Checks: PASS." if not ch["bad"] else f"Checks: {len(ch['bad'])} problems (see the end)."), ""]
    for tid, tb in doc["flow"]["azimuth_tracks"].items():
        r = (met.get("track") or {}).get("tracks", {}).get(tid) or {}
        rp_ = (res.get("track_build") or {}).get(tid) or {}
        cr = r.get("crossing") or {}
        blk = next(s["camera"] for s in doc["segments"] if s["camera"].get("azimuth_track") == tid)
        sides = {}
        for w_ in r.get("windows", []):
            for sd in (next((x.get("side") for x in tb["windows"] if x["name"] == w_["name"]), None) or [])[:1]:
                sides.setdefault(sd, []).append(w_.get("eye_on_lead_right", 0.5))
        side_txt = {k2: ("the lead's right" if np.mean(v) > 0.5 else "the lead's left") for k2, v in sides.items()}
        L += [f"## {tid}: film {tb['film'][0]}-{tb['film'][1]} s (segments {tb['segments'][0]} .. {tb['segments'][1]})", "",
              f"- **The law it replaces** (v4.2): world azimuth {tb['law']['world_azimuth_deg']} deg + {tb['law']['drift_deg_s']} deg/s x (film - "
              f"{tb['law']['F0']}): the couple stands almost still, so the orbit swept the eye about 215 deg round them and filmed the first counterbalance "
              "nearly along their line (one dancer behind the other).",
              f"- **The track**: the eye's world azimuth as a table ({tb['n']} samples at {tb['hz']:g} Hz from {tb['t0']:.4f} s; inside {tb['film'][0]} < film < "
              f"{tb['film'][1]}, linear between the samples). It holds the first profile (psi {tb['profiles']['first_deg']:g}: the eye on "
              f"{side_txt.get('first', '?')}) to the first counterbalance's end, crosses the couple's line ONCE in {tb['crossing']['film'][0]}-"
              f"{tb['crossing']['film'][1]} s (no counterbalance interval there; along the line at {r.get('crossings_at_s')} s), and holds the second "
              f"profile (psi {tb['profiles']['second_deg']:g}: the eye on {side_txt.get('second', '?')}) through the both-hands / one-hand demonstrations, then "
              f"rejoins the law. It leaves / rejoins the law by the blend w(t) over {tb['blend']['in_s']} / {tb['blend']['out_s']} (the closer ease windows; C2): "
              "both phone ways are v4.2's frame for frame. (The lead's left = the joined hands toward the camera; his right = his right hand on her back.)",
              f"- **The line**: the lead -> follow pelvis line of the installed motion (smplx_lead.bin {tb['line']['lead_sha256'][:12]}..., smplx_follow.bin "
              f"{tb['line']['follow_sha256'][:12]}...; the pelvis is the same in the input / v3 / v4 / v43 fits).",
              f"- **Turn rate** (settled frames, 30 Hz): {(r.get('rate_its_way') or {}).get('min')}..{(r.get('rate_its_way') or {}).get('max')} deg/s its way, "
              f"|accel| <= {r.get('accel_abs_max')} deg/s2. The crossing: peak {cr.get('rate_max_deg_s')} deg/s, mean {cr.get('rate_mean_deg_s')} deg/s over "
              f"{tb['crossing']['film'][0]}-{tb['crossing']['film'][1]} s, {cr.get('over_6_deg_s_s')} s of it above 6 deg/s; outside the crossing the track "
              f"turns at most {cr.get('outside_it_rate_max_deg_s')} deg/s ({cr.get('outside_it_over_6_deg_s_s')} s above 6: the return onto the law in the "
              f"blend before the way in; all stretches above 6: {tb['crossing']['over_6_deg_s']}). The crossing is the "
              "owner-approved exception to v4.1's 2.5-6 deg/s orbit band (the ways' caps are 12 deg/s).",
              _lift_bullet(tb, r, blk, cr),
              f"- **Solver**: {rp_.get('knot_spans')} cubic B-spline spans, SLSQP rounds {[(x['round'], x.get('missed')) for x in rp_.get('solver', [])]}.", "",
              "### The view's angle to the couple's line, per window (min / median / max deg, 90 = profile; PoseModel 30 Hz, 9:16)", ""]
    L += _md_table(["window", "film s", "v4.2", "v4.2 within 25", "v4.3", "v4.3 within 25", "16:9", "scored", "its own best", "eye on the lead's right", "target"], v43_rows(res))
    L += ["", "Scored = the window's part inside the track (outside the phone ways) and outside the crossing; the target: median within 15 deg of 90 and "
          ">= 90 % of the frames within 25 - or, where no legal track can reach 15 (its own best, the same programme with only that window), within 1 deg "
          "of that best. Not scored: the crossing itself, and the parts inside the way into phone 06 (v4.2's, unchanged by decision).", ""]
    ex = (met.get("track") or {}).get("frame_exact") or {}
    L += ["## The ways and everything else: v4.2's", ""]
    for asp, x in ex.items():
        L += [f"- {asp}: {x['differ']} of {x['frames']} frames differ, all inside the track's window ({x['first_last_differing_s']} s); outside it "
              f"{x['differ_outside_tracks']} (max difference {x['max_diff_outside_tracks']}); way frames differing {x['way_frames_differ']} of {x['way_frames']}."]
    dd = (met.get("track") or {}).get("direction_diff") or {}
    L += [f"- The direction: {'v4.2 + the track keys, nothing else' if dd.get('same_but_the_track') else 'DIFFERS from v4.2 beyond the track'} "
          "(flow.azimuth_tracks, camera.azimuth_track on the run's segments, flow.revision 4.3, the schema line, `created`, recap_film.v4.v43, the "
          "re-measured flow.v4.measured).", ""]
    gh, gh42 = res.get("ghost") or {}, res.get("ghost42") or {}
    if gh and gh42:
        L += ["Faint frustums against the dancers (ghost_overlap, plan estimate, s): " + "; ".join(
            f"{c}: over the couple {gh42['per_phone'][c]['over_couple_s']} -> {gh['per_phone'][c]['over_couple_s']}, apex within 1 m of the eye "
            f"{gh42['per_phone'][c]['near_lens_s']} -> {gh['per_phone'][c]['near_lens_s']}" for c in sorted(gh["per_phone"])) + " (the keep-out in "
            "flow.ghost_frustums covers these).", ""]
    L += owner_costs_md(res)
    L += ["## Checks", ""]
    L += [f"- {'OK' if t_['ok'] else 'MISSED'}: {t_['target']}: want {t_['want']}, measured {t_['measured']}" for t_ in ch["targets"]]
    L += [f"- PROBLEM: {b_}" for b_ in ch["bad"]] + [f"- {n}" for n in ch["notes"]]
    return "\n".join(L) + "\n"


def _lift_bullet(tb: dict, r: dict, blk: dict, cr: dict) -> str:
    hh = r.get("far_head_hidden_s") or {}
    h43, h42, alt = hh.get("v43") or {}, hh.get("v42") or {}, hh.get("alt") or {}
    eye = f"eye max {r.get('eye_h_max_m')} m (9:16) / {r.get('eye_h_max_16x9_m')} m (16:9)"
    fl_ = h43.get("face_hidden_longest_s") or [None, None, None]
    if tb["crossing"].get("lift", "raised") == "none":
        txt = (f"- **No lift in the crossing**: the eye keeps the orbit's own height ({blk.get('height_m')} m at full closeness; {eye}). ")
    else:
        txt = (f"- **The lift**: during the crossing the eye rises up to {tb['rules']['lift_top_m']} m (cap {tb['crossing']['lift_cap_m']} m above the block's "
               f"{blk.get('height_m')} m eye height at full closeness; v4.2's cap {blk.get('inline_lift_max_m')} m): lift max {r.get('lift_max_m')} m, {eye}, "
               f"its vertical speed <= {cr.get('eye_vertical_speed_max_m_s')} m/s. It shows part of the follow's face for longer as the camera comes "
               "behind the lead (her head rises beside / above his), but not at the core of the crossing: there her chin would clear the top of his head "
               "only from an eye at ~3.4 m (the band's top is 2.4). ")
    txt += (f"Plan estimates over {hh.get('window_s')} s: her face hidden behind his head {h43.get('face_hidden_s')} s ({fl_[0]}-{fl_[1]} s; {hh.get('face_rule')})"
            + (f", with the lift '{alt.get('lift')}' instead {alt.get('face_hidden_s')} s" if alt else "") + f", v4.2 {h42.get('face_hidden_s')} s; "
            f"the two heads under 3 / 4 deg apart over the crossing v4.3 {h43.get('heads_under_3deg_s')} / {h43.get('heads_under_4deg_s')} s"
            + (f" ('{alt.get('lift')}': {alt.get('heads_under_3deg_s')} / {alt.get('heads_under_4deg_s')} s)" if alt else "")
            + f", v4.2 {h42.get('heads_under_3deg_s')} / {h42.get('heads_under_4deg_s')} s.")
    return txt


def owner_costs_md(res: dict) -> list[str]:
    """v4.3 (fix round 1): what the decided rules cost, in plain words for the owner - the verifier's findings M1 / m1 / m2 / m4"""
    doc, met = res["doc"], res["metrics"]
    L = ["## For the owner: what the decided rules cost", ""]
    for tid, tb in doc["flow"]["azimuth_tracks"].items():
        r = (met.get("track") or {}).get("tracks", {}).get(tid) or {}
        hh = r.get("far_head_hidden_s") or {}
        h43, h42 = hh.get("v43") or {}, hh.get("v42") or {}
        win = {w_["name"]: w_ for w_ in r.get("windows", [])}
        run = h43.get("face_hidden_longest_s")
        cross_rows = [w_ for w_ in r.get("windows", []) if w_["kind"] == "segment" and run and w_["film"][0] < run[1] and w_["film"][1] > run[0]]
        fp = (tb["rules"] or {}).get("fast_pass")
        rmax = tb["rules"]["rate_max_deg_s"] if not fp else tb["rules"].get("rate_max_deg_s_gap", 7.0)
        if run:
            seg_txt = ", ".join(f"{w_['name']} {w_['film'][0]}-{w_['film'][1]} s (angle to the line median {(w_.get('v42_9:16') or {}).get('median')} -> "
                                f"{(w_.get('v43_9:16') or {}).get('median')} deg; heads under 4 deg {100 * (w_.get('heads_under_4deg') or {}).get('v42_9:16', 0):.0f} -> "
                                f"{100 * (w_.get('heads_under_4deg') or {}).get('v43_9:16', 0):.0f} % of the frames)" for w_ in cross_rows)
            L += [f"1. **{run[0]:.1f}-{run[1]:.1f} s: the follow's face hidden behind the lead for about {run[2]:.1f} s** (9:16 plan estimate, calibrated on "
                  f"editor stills; her head is partly behind his for longer: the heads under 4 deg apart for {h43.get('heads_under_4deg_s')} s of the crossing "
                  f"{tb['crossing']['film']} s). This is the one line crossing the two phone ways "
                  f"force (phone 00 and phone 06 film the couple from opposite sides). It lands at {r.get('crossings_at_s')} s in the gap without a detected "
                  f"counterbalance: {seg_txt}" + (" - the counterbalance layer is on there (Kadu explaining the \"hip forward\" mistake)." if cross_rows else ".")
                  + (f" At the decided <= {rmax:g} deg/s the camera needs the whole gap for the ~145 deg between the two profiles, so it passes behind the "
                     f"lead at ~{rmax:g} deg/s; the lift shows her a little longer, but at the core no eye inside the 2.4 m band can." if not fp else
                     f" The fast pass (an owner decision) runs through the line at up to {fp['rate_max_deg_s']:g} deg/s in {fp['film']} s.")
                  + f" Over {hh.get('window_s')} s (the crossing +-3 s) her face is hidden {h42.get('face_hidden_s')} s in v4.2 (its own crossing, at the "
                  f"first counterbalance's end) and {h43.get('face_hidden_s')} s in v4.3." + ("" if fp else " The option, if the owner wants her visible there: a short faster pass through the line (up to 12 deg/s, the "
                                        "ways' turn cap, for about 3 s; `build --version v4 --fast-pass` builds it into v4/direction/v4.3_fastpass/ "
                                        "for review, without touching this direction)."), ""]
        w2 = win.get("ch02_02_weight")
        if w2 and (w2.get("best_alone") or [None])[0] is not None:
            L += [f"2. **ch02_02_weight ({w2['film'][0]}-{w2['film'][1]} s, counterbalance layer): median {(w2.get('v43_9:16') or {}).get('median')} deg to the "
                  f"line, not within 15 of 90 (75 or more) as the target asks.** No track can reach it under the decided rules: leaving phone 00 at "
                  f"{abs(float(tb['law']['drift_deg_s'])):g} deg/s, the camera may slow at <= {tb['rules']['accel_max_deg_s2']:g} deg/s2 only to "
                  f"{tb['rules']['rate_min_deg_s']:g} deg/s while the couple turns slightly against it; its own best is {w2['best_alone'][0]:.1f} deg. Reaching "
                  f"75 would need a slower floor there (~0.7 deg/s at 43-53 s). v4.2 had {(w2.get('v42_9:16') or {}).get('median')} deg.", ""]
        tt = float(tb["t0"]) + float(tb["dt"]) * np.arange(int(tb["n"]))
        lawv = float(tb["law"]["world_azimuth_deg"]) + float(tb["law"]["drift_deg_s"]) * (tt - float(tb["law"]["F0"]))
        o0, o1 = tb["blend"]["out_s"]
        mo = (tt >= o0) & (tt <= o1)
        dv = np.abs(np.asarray(tb["deg"], float) - lawv)
        ex = (met.get("track") or {}).get("frame_exact") or {}
        nway = ", ".join(f"{asp} {x['way_frames_differ']} of {x['way_frames']}" for asp, x in ex.items())
        cb96 = next((w_ for w_ in r.get("windows", []) if w_["kind"] == "interval" and o0 <= w_["film"][0] < o1), None)
        if mo.any():
            L += [f"3. **The way into phone 06 is v4.2's frame for frame from {o1} s (the way itself; way frames differing: {nway}).** Over {o0}-{o1} s, "
                  f"v4.2's closer ease just before it, the track blends back onto v4.2's orbit and differs from it by up to {float(dv[mo].max()):.2f} deg "
                  f"(at {float(tt[mo][np.argmax(dv[mo])]):.1f} s), 0 at {o1} s: the price of holding "
                  + (f"the {cb96['film'][0]}-{cb96['film'][1]} s counterbalance in profile ({(cb96.get('v42_9:16') or {}).get('median')} -> "
                     f"{(cb96.get('v43_9:16') or {}).get('median')} deg)." if cb96 else "the last counterbalances in profile."), ""]
        frozen = [w_ for w_ in r.get("windows", []) if not w_.get("scored") and "phone way" in (w_.get("not_scored") or "")]
        if frozen:
            L += ["4. **Not in profile: " + "; ".join(f"{w_['name']}{'' if w_['kind'] == 'interval' else ' ' + str(w_['film'][0]) + '-' + str(w_['film'][1])} s "
                                                     f"at {(w_.get('v43_9:16') or {}).get('median')} deg"
                                                     for w_ in frozen)
                  + "** - v4.2's own view (30-32 deg off profile): they lie inside the way into phone 06, which stays v4.2's by decision. The only "
                  "counterbalance of chapters 1-3 that is not filmed in profile.", ""]
    return L


def contract_v43_md(res: dict) -> str:
    """the v4.3 section that heads UNITY_CONTRACT_v4.md (generic words only: lead / follow / the recap capture)"""
    doc = res["doc"]
    tid, tb = next(iter(doc["flow"]["azimuth_tracks"].items()))
    r = ((res["metrics"].get("track") or {}).get("tracks") or {}).get(tid) or {}
    L = ["# Unity contract: the v4 class-recap direction (`hm film flow v2`, revision 4.3)", "",
         "Source: `dancecap/work/larissa_kadu_recap/v4/direction/direction.json` (`flow.revision` \"4.3\"; v4.2 in `v4/direction/v4.2/`, v4.1 in "
         "`v4/direction/v4.1/`, v4.0 in `v4/direction/v4.0/`). Built by `python -m dancecap.film_recap_flow build --version v4 --mix`, checked by `... check "
         "--version v4` (v4.3's own targets, revision gate: an unknown revision is an error). Write no names into head-movement: say lead / follow / "
         "\"the recap capture\".", "",
         "## v4.3: what changes for the Unity director (on top of v4.2 H-J, v4.1 A-G and v4.0 1-9 below)", "",
         "| # | what | where | v4.2 |", "|---|---|---|---|",
         "| K | one slow orbit plays a planned azimuth track: the eye's world azimuth (and its inline lift) from a table inside the track's window | "
         "`flow.azimuth_tracks.<id>`, `camera.azimuth_track` = \"<id>\" | world_azimuth_deg + drift_deg_s x (film - F0) everywhere |",
         "| L | the orbit check knows tracked shots: their own rate band (1-7 deg/s its way), not 2.5-6 | FilmFlowCheck orbit rule | 2.5-6 +- 0.25 |", "",
         "**K. The azimuth track.** `flow.azimuth_tracks` is an object: id -> track. This direction has one, "
         f"`\"{tid}\"` (run {tb['run']}, segments {tb['segments'][0]} .. {tb['segments'][1]}); every segment of that run carries `camera.azimuth_track` = "
         f"`\"{tid}\"` (the blocks stay identical, so they still form ONE FilmShot, F0 = {tb['law']['F0']}). The track's keys:", "",
         "| key | type | meaning |", "|---|---|---|",
         "| `film` | [a, b] film s | the window: the table applies for a < film < b (open); at and outside it the block's law, exactly as today |",
         f"| `t0`, `dt`, `n` | film s, s, int | sample i (0 .. n-1) is at film t0 + i x dt (here t0 = {tb['t0']!r} = {tb['i0']} / {tb['hz']:g}, dt = 1 / "
         f"{tb['hz']:g}, n = {tb['n']}; the samples cover [a, b]: t0 <= a, t0 + (n - 1) dt >= b) |",
         "| `deg` | n floats, deg | the eye's WORLD azimuth about the orbit centre, the viewer's atan2(z, x), unwrapped (continuous with world_azimuth_deg + "
         f"drift_deg_s x (film - F0): {tb['deg'][0]} at t0, {tb['deg'][-1]} at the last sample) |",
         "| `lift_m` | n floats, m | the eye's inline lift inside the window (0 at both ends) |",
         "| `law` | {world_azimuth_deg, drift_deg_s, F0} | the block's law, for checks (the block's own keys are unchanged) |",
         "| `interp` | \"linear\" | the interpolation |",
         "| `hz`, `i0` | 30, int | t0 = i0 / hz exactly (documentation; use t0 / dt) |",
         "| `blend`, `rules`, `profiles`, `crossing`, `line`, `windows`, `lift_rule` | | documentation and the python checks' inputs; Unity needs none of them |", "",
         "Interpolation (pure, `HmFilm.FlowMath.TrackAt(float t0, float dt, float[] v, float t)`): x = (t - t0) / dt; i = clamp(floor(x), 0, n - 2); u = "
         "clamp(x - i, 0, 1); return v[i] + (v[i + 1] - v[i]) x u. (The recorder's frames at k / 30 s land on the samples; between them it is linear - the "
         "underlying curve is C2 and its curvature over 1/30 s is below 0.001 deg.)", "",
         "One table for both aspects (the azimuth law was always aspect-free; 9:16 and 16:9 differ only in radius_vertical_m / radius_m and the bands, "
         "unchanged). Floats are fine: t0 as a float is off by < 1e-6 s, the degrees by < 1e-5.", "",
         "Parse (FlowTimeline.Build): `flow.azimuth_tracks` -> a dictionary id -> {A = film[0], B = film[1], T0, Dt, Deg float[], Lift float[]}; a track "
         "whose arrays are not n long, or whose window is not inside its samples, is dropped with a Warning (the shot then plays its law).", "",
         "Callers (FilmCamera.cs): **`WorldAzimuth(s, film, out phi)`** is the one place: if `s.Camera[\"azimuth_track\"]` names a track and A < film < B, "
         "phi = TrackAt(T0, Dt, Deg, film) x Deg2Rad, else the law as now. Every caller inherits it: Orbit (:491), the Hann lift loop's samples (:513), "
         "Rhythm (:629), Physics3Q (:658), FeatureZoom (:904), FeetCloseup (:1008), the overhead's up_policy orbit (:1154) - in this direction only Orbit of "
         "that one shot reads a track. The ways (WayV4) blend the live poses, so they are unchanged wherever the track is not played - and it is not played "
         "on any way frame (the window is the orbit between its way in and its way out; python: 0 way frames differ).", "",
         "**The lift** (FilmCamera.Orbit, the `avoid_inline` block): inside the window (A < film < B) of the shot's track, lift = TrackAt(T0, Dt, Lift, film) "
         "- it REPLACES the Hann-weighted live InlineLift (the cap and the closer factor are already in it); outside the window the live lift as now. Then "
         "height += lift as before (the ground_level clamp to 0.5-2.4 m still applies). The table is the live lift's python twin (the damped pelvises, "
         "the +-36-frame Hann window, inline_lift_max_m x closer) "
         + (f"with the cap raised to {tb['rules']['lift_top_m']} m - height_m = {tb['crossing']['lift_cap_m']} m inside the crossing {tb['crossing']['film']} s. "
            if tb["crossing"].get("lift", "raised") != "none" else
            f"with NO lift inside the crossing {tb['crossing']['film']} s (crossing.lift \"none\": behind the taller lead a lift up to the band's top cannot "
            f"show the follow); in this direction lift_m is {float(np.max(tb['lift_m'])):g} m everywhere, so inside the window the eye has no inline lift. ")
         + f"Within 1.3 s of both window ends the live lift (uncapped) is {((res.get('track_build') or {}).get(tid) or {}).get('live_lift_at_window_edges_m')} m "
         "(the eye is 60-90 deg off the line there): the switch is invisible. The live lift of a frame OUTSIDE the window samples the block's LAW "
         "at its +-1.2 s Hann samples, never the table (FilmCamera.WorldAzimuth(..., useTrack: false)), so every frame outside the window is exactly "
         "the frame without the track.", "",
         "Numbers to check in Play mode (`hm_film_flow --action check`, both aspects): the tracked shot's settled orbit frames turn "
         f"{tb['rules']['rate_min_deg_s']:g}-{tb['rules']['rate_max_deg_s']:g} deg/s about the couple its way, above "
         f"{tb['rules'].get('rate_max_outside_deg_s', tb['rules']['rate_max_deg_s']):g} only inside the crossing {tb['crossing']['film']} s (python: "
         f"{(r.get('rate_its_way') or {}).get('min')}..{(r.get('rate_its_way') or {}).get('max')}, outside the crossing <= "
         f"{(r.get('crossing') or {}).get('outside_it_rate_max_deg_s')}; Unity's live centre is damped, so allow its usual +-0.25); the eye stays <= {tb['rules']['lift_top_m']} m; jumps / stutters / eye out of band / heads out of frame "
         "= 0; every way's peaks (turn / rotation / speed) equal v4.2's sweep (`v4/direction/stills_v4.2/flowcheck_v42_*.json`), the two ways next to the "
         "track (phone out 30.246-34.746, phone in 99.499-105.499) frame for frame.", "",
         "**L. FilmFlowCheck** (Editor/FilmFlowCheck.cs, the orbit rule ~:450-485): for a shot with `camera.azimuth_track`, a settled orbit frame is in "
         f"band when its turn about the couple is within [{tb['rules']['rate_min_deg_s']:g} - 0.25, {tb['rules']['rate_max_deg_s']:g} + 0.25] deg/s inside "
         f"the track's `crossing.film` and [{tb['rules']['rate_min_deg_s']:g} - 0.25, {tb['rules'].get('rate_max_outside_deg_s', 6.0):g} + 0.25] elsewhere, "
         "in the sign of drift_deg_s (the track's own `rules`), not 2.5-6; the turning-back rule is unchanged (sign of drift_deg_s). Report the tracked frames "
         "separately (orbitTrackedFrames, their min / max rate). The FlowV4Tests test that reads the v4 direction gates on `tl.Revision != \"4.2\"` "
         "(FlowV4Tests.cs:470): let it run for \"4.3\" too, and add: TrackAt at samples / midpoints / clamped ends; on the direction's own track: the "
         "value at film = A and B equals the law within 1e-3 deg, every 1/30-s step turns 1-7 deg/s its way, the accel <= 2 deg/s2, lift 0 at both ends "
         "and <= 2.4 m - height_m.", "",
         "Everything below (v4.2's contract, which carries v4.1's and v4.0's) still holds.", "", "---", "",
         "# The v4.2 contract (unchanged; v4.3 adds K and L above)", ""]
    return "\n".join(L) + "\n"


def write_all_v43(res: dict, out: Path) -> None:
    doc, tl = res["doc"], res["tl"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "narration").mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    (out / "direction.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    caps.write_timeline(out / "narration" / "timeline.json", tl)
    ch = []
    for i, c in enumerate(doc["chapters"]):
        ch.append({"title": c["title"], "startS": c["film_s"], "endS": doc["chapters"][i + 1]["film_s"] if i + 1 < len(doc["chapters"]) else doc["timeline"]["film_duration_s"],
                   "segment": c["segment"]})
    (out / "chapters.json").write_text(json.dumps({"schema": "dancecap.film_recap/1/chapters", "film_duration_s": doc["timeline"]["film_duration_s"], "chapters": ch},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")
    res["ghost"] = ghost_overlap(doc, res["metrics"], res["geo4"], res["cfg4"])
    res["ghost42"] = ghost_overlap(res["doc42"], res["metrics42"], res["geo4"], v42_cfg())
    tr_pub = {k: {k2: v for k2, v in r.items() if k2 not in ("angle", "t", "lift", "rate_s", "accel_s", "theta", "law")} for k, r in (res.get("track_build") or {}).items()}
    cols = [("v4.2", res["metrics42"]), ("v4.3", res["metrics"])]
    ba = {"schema": "dancecap.film_recap_flow/v4.3/before_after", "columns": ["v4.2", "v4.3"], "v43_rows": v43_rows(res),
          "rows": before_after_rows_n(cols, res["targets"]), "metrics": {lab: _metrics_public(m) for lab, m in cols},
          "metrics_16x9": {"v4.2": _metrics_public(res["metrics42_16x9"]), "v4.3": _metrics_public(res["metrics_16x9"])}, "track": res["metrics"].get("track"),
          "track_build": tr_pub, "check_targets": res["checks"]["targets"], "ghost": {"v4.2": res["ghost42"], "v4.3": res["ghost"]}}
    (out / "before_after.json").write_text(json.dumps(ba, indent=1, default=str), encoding="utf-8")
    rep = {"schema": "dancecap.film_recap_flow/2.3/report", "checks": res["checks"], "targets": res["targets"], "metrics": _metrics_public(res["metrics"]),
           "metrics_16x9": _metrics_public(res["metrics_16x9"]), "metrics_v42": _metrics_public(res["metrics42"]), "track_build": tr_pub,
           "closer": res["closer"], "meter": doc["flow"]["meter"], "runs": [{k: v for k, v in r.items() if k != "q"} | {"quality": r.get("q")} for r in res["runs"]],
           "ways": res["sched"]["ways"], "orbits": res["sched"]["plans"], "dp": res["dp"], "clips": res["clips"], "pivots": res["pivots"], "replay": res["replay"],
           "ghost_overlap": res["ghost"], "physics": [{k: r[k] for k in ("id", "film", "mode", "on", "why")} for r in res["phys_rows"]], "physics_timeline": physics_timeline(doc)}
    (out / "direction_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "clips.json").write_text(json.dumps({"what": "per phone: the master-second ranges of video the v4.3 direction needs (phone holds + frustum ways in / out, "
                                                        "padded 0.5 s; = v4.2's)", "phones": res["clips"]}, indent=1), encoding="utf-8")
    (out / "review_v4.3.md").write_text(review_v43_md(res), encoding="utf-8")
    r42 = dict(res, doc=res["doc42"], metrics=res["metrics42"], metrics_16x9=res["metrics42_16x9"], cfg4=v42_cfg())
    r4 = dict(r42, cfg4=dataclasses_replace(v42_cfg(), version="4.1"))
    base = contract_v41_md(r4) + "\n" + contract_md(r4).split("\n", 1)[1]
    v42_part = contract_v42_md(r42) + "\n" + base.split("\n", 1)[1]
    (out / "UNITY_CONTRACT_v4.md").write_text(contract_v43_md(res) + v42_part.split("\n", 1)[1].lstrip("\n"), encoding="utf-8")
    (out / "audio" / "make_film_mix.py").write_text(MIX_SCRIPT.replace("the v3 direction", "the v4.3 direction"), encoding="utf-8")
    m, c4 = res["metrics"], res["cfg4"]
    sheet_v4(m, doc, res["geo4"], out / "flow_sheet.png",
             f"v4.3 camera flow (the v4.2 plan + run 2's profile track): {m['moves']['total']} moves, {m['phone_holds']['n']} phone holds (mean {m['phone_holds']['mean_s']} s), "
             f"slow orbits {', '.join(str(x) for x in m['three_d']['drift_rates_deg_s'])} deg/s (run 2: its track 1-7 deg/s)", band=c4.orbit_rate_deg_s, cap=c4.cap_az_deg_s, label="v4.3")
    track_plot(res, out / "track_v4.3.png")


VERSIONS = ("v3", "v4", "v4.0", "v4.1", "v4.2", "v4.3")       # v4 = the latest v4 (v4.3)


def _v4_out(version: str) -> Path:
    return {"v4.0": OUT_V40, "v4.1": OUT_V41, "v4.2": OUT_V42}.get(version, OUT_V4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.film_recap_flow", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--version", choices=VERSIONS, default="v3", help="v3 (default: v3/direction/, reproducible as it was); v4.0 (owner 2026-10-22 first: "
                   "fewer, longer shots, slow orbits, C2 glides, faint frustums -> v4/direction/v4.0/); v4.1 (owner 2026-10-22 after the v3 films: "
                   "holds >= 20 s placed by Jev's need_3d, orbits <= 6 deg/s, ways >= 4 s under 12 deg/s, the legend in the 9:16 top band -> v4/direction/v4.1/); "
                   "v4.2 (owner 2026-10-22 after v4.1: the v4.1 plan, the physics meter only for a few seconds at each switch-on, the slow orbits 0.8 x "
                   "closer -> v4/direction/v4.2/); v4 = v4.3 (owner 2026-10-23: the v4.2 plan + the initial counterbalances as profile two-shot orbits: a "
                   "planned azimuth track for the orbit of chapters 1-3 -> v4/direction/, what the recorder reads)")
    b.add_argument("--plan", choices=("meaning", "share"), default="meaning", help="v4.1: meaning-first (the most phone time whose holds stay clearly under "
                   "the 3D views' Jev need) or share-first (the least multiplier that fills the 45 %% phone share, or the most phone time when none does)")
    b.add_argument("--out", default=None, help="write here instead of the version's folder")
    b.add_argument("--alt", default=None, help="v4.1: a built alternative to compare with in the review (default: v4/direction/v4.1_share when it exists)")
    b.add_argument("--mix", action="store_true", help="also render the film soundtrack into <out>/audio/ (the class mix of film/audio, the same time map)")
    b.add_argument("--hand-events", default=None, help=f"the hand-contact events the 'connected' rule reads (default {HAND_EVENTS.relative_to(PACKAGE_ROOT)})")
    b.add_argument("--fast-pass", action="store_true", help="v4 / v4.3 only: the OWNER-DECISION variant (not decided): run 2 runs through the couple's "
                   f"line at up to {V43_FAST_PASS['rate_max_deg_s']:g} deg/s for a few seconds (the decided track: <= {V43_TRACK['rate_max_deg_s']:g}) -> "
                   "v4/direction/v4.3_fastpass/ unless --out (never the recorder's v4/direction/ by default)")
    c = sub.add_parser("check")
    c.add_argument("--direction", default=None, help="default: v3/direction/direction.json (v4/direction/ with --version v4 / v4.3, v4/direction/v4.<n>/ "
                   "with v4.0 / v4.1 / v4.2); a flow-v2 direction gets the v4 checks + the targets of its flow.revision (4.1, 4.2, 4.3; an unknown one is an error)")
    c.add_argument("--version", choices=VERSIONS, default="v3")
    k = sub.add_parser("clips", help="UNITY STAGE ONLY: export the phones' video the direction needs into the capture (camera_video, under the Unity lock)")
    k.add_argument("--no-install", action="store_true", help="write the master only (work/<take>/camera_video/<capture>); mirror later with camera_video --install-only")
    k.add_argument("--priority", choices=("below_normal", "normal"), default="normal")
    k.add_argument("--version", choices=VERSIONS, default="v3")
    sh_ = sub.add_parser("sheet", help="redraw flow_sheet.png from a fresh build (nothing else written)")
    sh_.add_argument("--version", choices=VERSIONS, default="v3")
    a = ap.parse_args(argv)
    if a.cmd == "clips":
        return cmd_clips(a.no_install, a.priority, OUT if a.version == "v3" else _v4_out(a.version))
    if a.cmd == "sheet":
        if a.version != "v3":
            r = (build_v4(write=False) if a.version == "v4.0" else build_v41(write=False) if a.version == "v4.1" else build_v42(write=False)
                 if a.version == "v4.2" else build_v43(write=False))
            m, c4 = r["metrics"], r["cfg4"]
            print(sheet_v4(m, r["doc"], r["geo4"], _v4_out(a.version) / "flow_sheet.png", f"{a.version} camera flow: {m['moves']['total']} moves, "
                           f"{m['phone_holds']['n']} phone holds", band=c4.orbit_rate_deg_s, cap=c4.cap_az_deg_s, label=a.version))
            return 0
        r = build(write=False)
        print(sheet(r, OUT / "flow_sheet.png"))
        return 0
    if a.cmd == "build" and a.version != "v3":
        if a.fast_pass and a.version not in ("v4", "v4.3"):
            ap.error("--fast-pass is a v4 / v4.3 option")
        out = Path(a.out) if a.out else (OUT_V43_FAST if a.fast_pass else _v4_out(a.version))
        if a.fast_pass:
            r = build_v43(hand_events=a.hand_events, out=out, cfg=dict(V43_TRACK, fast_pass=dict(V43_FAST_PASS)))
        elif a.version == "v4.0":
            r = build_v4(hand_events=a.hand_events, out=out)
        elif a.version == "v4.2":
            r = build_v42(hand_events=a.hand_events, out=out)
        elif a.version in ("v4", "v4.3"):
            r = build_v43(hand_events=a.hand_events, out=out)
        else:
            cfg4 = dataclasses_replace(v41_cfg(), plan=a.plan)
            alt = Path(a.alt) if a.alt else (OUT_V4 / "v4.1_share" if a.plan == "meaning" else None)
            r = build_v41(cfg4=cfg4, hand_events=a.hand_events, out=out, alt_dir=alt if (alt and alt.resolve() != out.resolve()) else None)
        ch = r["checks"]
        print(f"{a.version} direction -> {out}: {len(r['doc']['segments'])} segments, {ch['shots']} shots, {ch['boundaries']} ways {ch['transitions']}, checks "
              f"{'PASS' if not ch['bad'] else str(len(ch['bad'])) + ' problems'}")
        for row in ch["targets"]:
            print(f"  {'OK    ' if row['ok'] else 'MISSED'} {row['target']}: want {row['want']}, measured {row['measured']}")
        for n in ch["notes"]:
            print("  " + n)
        if a.mix:
            from . import film_recap_audio as fa
            mx = fa.film_mix(out / "direction.json", fa.soundtrack_class_mix(WORK / "film" / "audio"))
            print("film mix:", {k: v for k, v in mx.items() if k != "slow"})
        return 1 if ch["bad"] else 0
    if a.cmd == "build":
        r = build(hand_events=a.hand_events)
        ch = r["checks"]
        print(f"v3 direction: {len(r['doc']['segments'])} segments, {ch['shots']} shots, {ch['boundaries']} flights {ch['transitions']}, joins {len(r['v3']['flow']['merges'])}, "
              f"checks {'PASS' if not ch['bad'] else ch['bad']}")
        for n in ch["notes"]:
            print("  " + n)
        if a.mix:
            from . import film_recap_audio as fa
            m = fa.film_mix(OUT / "direction.json", fa.soundtrack_class_mix(WORK / "film" / "audio"))
            print("film mix:", {k: v for k, v in m.items() if k != "slow"})
        return 1 if ch["bad"] else 0
    return cmd_check(Path(a.direction) if a.direction else (OUT if a.version == "v3" else _v4_out(a.version)) / "direction.json")


if __name__ == "__main__":
    raise SystemExit(main())
