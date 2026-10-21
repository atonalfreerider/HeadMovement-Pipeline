"""v3 of the class-recap direction (owner 2026-10-21): the counterbalance pivot for "back to the axis", the physics layer on every
demonstration and every connected shot, and NO jump cut anywhere (every change of view is a continuous ground-level flight; a phone's point
of view is entered and left through the phone's frustum with its 2D video surface standing in the 3D world).

    python -m dancecap.film_recap_flow build      # -> work/larissa_kadu_recap/v3/direction/ (direction.json + review; film/direction.json is NOT touched)
    python -m dancecap.film_recap_flow check [--direction PATH]
    python -m dancecap.film_recap_flow sheet      # the camera-flow sheet (azimuth / height / transitions over the film) for review
    python -m dancecap.film_recap_flow build --mix  # ... and the film soundtrack into v3/direction/audio/ (sample-identical to film/'s: the same time map)
    python -m dancecap.film_recap_flow clips      # UNITY STAGE ONLY: the phones' video for the frustum surfaces (writes the capture's cameras/)

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
     phones' frustum envelopes); a shot too short to hold its view between its way in and its way out is merged into a neighbour;
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
PREV_V30 = WORK / "v3" / "direction_prev_v3.0" / "direction.json"     # v3.0 (2026-10-21, before the owner's v3.1 decisions), for the review
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
    depart_before_s: float = 0.1       # the way out of a phone starts this long before the next segment (the phone keeps its video until then; v3.0: 0.3)
    hold_s: dict = field(default_factory=lambda: {"3d": 1.0, "phone": 1.4, "overhead": 2.2})       # phone v3.0: 2.0
    # duration rule T = base + per180 * |d azimuth| / 180 + per_m_r * |d radius| + per_m_h * |d height|, clipped to [lo, hi], never shorter than the
    # arc floor (the arc's smootherstep peak <= MAX_ARC_DEG_S); squeezable to `min` (and never below the arc floor)
    rule: dict = field(default_factory=lambda: {
        "flight": {"base": 0.9, "per180": 1.4, "per_m_r": 0.25, "per_m_h": 0.3, "lo": 0.9, "hi": 2.4, "min": 0.8},
        "crane_up": {"base": 1.6, "per180": 0.6, "per_m_r": 0.0, "per_m_h": 0.0, "lo": 1.6, "hi": 2.2, "min": 1.4},
        "crane_down": {"base": 1.6, "per180": 0.6, "per_m_r": 0.0, "per_m_h": 0.0, "lo": 1.6, "hi": 2.2, "min": 1.4},
        # v3.0: phone in / out 1.3 + 1.8 x |daz| / 180 (1.3-3.2 s, min 1.2), phone to phone 1.9 + 1.8 x (1.9-3.6, min 1.7)
        "phone_in": {"base": 0.8, "per180": 1.2, "per_m_r": 0.1, "per_m_h": 0.1, "lo": 0.8, "hi": 2.2, "min": 0.75},
        "phone_out": {"base": 0.8, "per180": 1.2, "per_m_r": 0.1, "per_m_h": 0.1, "lo": 0.8, "hi": 2.2, "min": 0.75},
        "phone_to_phone": {"base": 1.0, "per180": 1.1, "per_m_r": 0.1, "per_m_h": 0.1, "lo": 1.0, "hi": 2.4, "min": 0.95},
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
        st = min(st, word - 0.1 - T)
        late = min(late, max(st, word - 0.1 - T + 0.3))
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
    cam = copy.deepcopy(segs[surv.segs[0]]["camera"])
    lay = copy.deepcopy(segs[surv.segs[0]]["layers"])
    view = segs[surv.segs[0]]["view_state"]
    treat = copy.deepcopy(segs[surv.segs[0]].get("treatment") or {})
    lo, hi = min(surv.d0, gone.d0), max(surv.d1, gone.d1)
    if surv.kind == "phone":
        if pov is None:
            return False
        ph = pov.phones.get(surv.phone)
        if ph is None or rp.window_quality(ph, lo, hi, pov.bad.get(surv.phone, []), pov.cfg) is None:
            return False
        cam["crop"] = rp.crop_track(ph, lo, hi, pov.cfg)
        if cam.get("mode") == "camera_tour":
            cam["cameras"] = [{"camera": surv.phone, "dance_t": [round(lo, 3), round(hi, 3)]}]
    for i in sorted(set(surv.segs + gone.segs)):
        sg = segs[i]
        if i in gone.segs:
            sg["v3_merged_from"] = sg["camera"].get("mode")
            sg["treatment"] = dict(treat, merged=True)
            if segs[surv.segs[0]].get("pivot"):
                sg["pivot"] = segs[surv.segs[0]]["pivot"]
            else:
                sg.pop("pivot", None)
        sg["camera"] = copy.deepcopy(cam)
        sg["layers"] = copy.deepcopy(lay)
        if surv.kind != "phone":
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


def fix_boundary(segs: list[dict], shots: list[Shot], k: int, pov: director.PovSource | None, log: list[str]) -> bool:
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


def plan_flow(segs: list[dict], geo: Geo, cfg: FlowCfg, pov: director.PovSource | None) -> tuple[list[Shot], dict]:
    """join too-short shots until every view holds, then the azimuth plan and the transitions"""
    log: list[str] = []
    for _ in range(cfg.max_merge_rounds):
        shots = shots_of(segs)
        plan_azimuths(shots, segs, geo, cfg)
        probs = schedule(shots, segs, geo, cfg)
        if not probs:
            break
        done = False
        for pr in sorted(probs, key=lambda x: -x["deficit_s"]):
            if fix_boundary(segs, shots, pr["k"], pov, log) or fix_boundary(segs, shots, pr["boundary"], pov, log):
                done = True
                break
        if not done:
            log.append(f"could not fix {len(probs)} squeezed boundaries: {probs}")
            break
    shots = shots_of(segs)
    plan_azimuths(shots, segs, geo, cfg)
    probs = schedule(shots, segs, geo, cfg)
    return shots, {"merges": log, "problems": probs}


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
    shots, flow_rep = plan_flow(segs, geo, cfg, pov)
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
                                "max_arc_deg_s": MAX_ARC_DEG_S}}
    doc["created"] = "generated by dancecap.film_recap_flow build (v3: pivot, physics, flights) on top of dancecap.film_recap; edit the rules there, not this file"
    doc["recap_film"]["v3"] = {"back_to_the_axis": len(phrases), "axis_events_dropped": [e["id"] for e in dropped_ev], "joins": len(flow_rep["merges"]),
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
    prev = None
    if PREV_V30.exists():
        pd = json.loads(PREV_V30.read_text(encoding="utf-8"))
        prev = flow_stats(pd, cfg_from_doc(pd))
    L += ["## (5) the transitions", "", "Owner decisions 2026-10-21 (v3.1): **overheads** - \"Keep overheads, crane up\": the 4 true top-down views stay, each entered and "
          "left by the smooth crane (the only ways that leave ground level); **phone share** - \"Shorter flights, ~50% phone\": quicker ways into and out of the "
          "phones (one easing, smootherstep; no arc faster than " + f"{MAX_ARC_DEG_S:.0f} deg/s at its peak) and a shorter minimum phone hold, so the phones' full-frame "
          "video comes back toward half the film. Every phone entry / exit still shows the frustum surface standing in the world, at least "
          f"{res['cfg'].frustum_readable_s} s readable (alpha >= 0.5, the eye not yet / no longer on the phone's optical axis).", "",
          "| | film/direction.json (cuts) | v3.0 | v3.1 (this) |", "|---|---|---|---|"]
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
    L += ["", "Durations of all transitions (v3.1): " + ", ".join(f"{k}: {v}" for k, v in fs["histogram"].items()) + ".", "",
          f"Why not 50 %: the phone shots are back at the cut film's share ({100 * pa['phone_shots_share']:.1f} % vs {100 * pb['phone_shots_share']:.1f} %), but every "
          f"phone way overlaps the phone's own time a little (a phone_to_phone, centred on its boundary, takes all of its {fs['by_type'].get('phone_to_phone', {}).get('sum_s', 0)} s "
          f"from the two phones). The director's balance step stops at {100 * (res['report'].get('director') or {}).get('pov_share', 0):.1f} % (no calm shot or second "
          f"zoom is left to hand to a phone), and the {len(v3['pivots'])} pivots, the {sum(1 for x in shots if x.kind == 'overhead')} overheads, the physics views "
          f"and the replay keep their 3D time. A what-if without ANY physics-call view (DirCfg.phys_calls off: half the physics_3q time; measured 2026-10-21) "
          f"reached only 47.3 %, so 50 % would need fewer pivots / overheads / physics views - not done.", "",
          f"Counts: {ch['transitions']}; shortest hold {ch['holds_min']} s. Joins (a shot too short to hold between its way in "
          f"and out joined a neighbour, {len(v3['flow']['merges'])}):", ""] + [f"- {m}" for m in v3["flow"]["merges"]]
    sq = [(sh, segs[sh.segs[0]]["camera"]["transition_in"]) for sh in shots[1:] if segs[sh.segs[0]]["camera"]["transition_in"].get("squeezed")]
    L += ["", f"Squeezed ({len(sq)}; the rule allows {res['cfg'].squeeze_max_s} s shorter and an arrival up to {res['cfg'].arrive_late_max_s} s after the "
          f"segment start): " + "; ".join(f"{segs[sh.segs[0]]['id']} {t['type']} ({', '.join(t['squeezed'])})" for sh, t in sq), ""]
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
    fig.suptitle(f"v3.1 camera flow: {len(shots)} shots, {len(shots) - 1} flights, no cut (blue 3D, orange phone, purple overhead)", fontsize=14)
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
    r = check(doc, tl, sp, geo, cfg_from_doc(doc))
    print("OK" if not r["bad"] else "\n".join(r["bad"]))
    for n in r["notes"]:
        print(n)
    print(f"{r['shots']} shots, {r['boundaries']} flights {r['transitions']}, shortest hold {r['holds_min']} s")
    return 1 if r["bad"] else 0


def cmd_clips(no_install: bool, priority: str) -> int:
    """UNITY STAGE ONLY (it writes the capture's cameras/ folder, under the Unity lock): decode the phones' video the v3 direction needs (clips.json: the
    phone shots and the frustum surfaces' ways in and out) together with the clips the capture has now (so film/direction.json still plays)"""
    from . import camera_video as cv
    need = json.loads((OUT / "clips.json").read_text(encoding="utf-8"))["phones"]
    geo = Geo()
    clips = {}
    for cam in sorted(set(need) | set(geo.phones)):
        rs = [tuple(x) for x in need.get(cam, {}).get("need", [])] + [tuple(x) for x in geo.phones.get(cam, {}).get("clips", [])]
        if rs:
            clips[cam] = sorted(rs)
    cv.OPTIONS["offsets_file"] = rp.write_offsets()
    cv.set_priority(priority)
    print("clips (s per phone):", {c: round(sum(y - x for x, y in cv.merge_ranges(v)), 1) for c, v in clips.items()})
    man = cv.run([TAKE], CAPTURE, None, [str(OUT / "direction.json")], False, list(rp.PHONES), None, None, not no_install, False, False, 1280, 85, 256, 78, clips=clips)
    print("pov_ok:", man.get("pov_ok"))
    return 0 if man.get("pov_ok") else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.film_recap_flow", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--mix", action="store_true", help="also render the film soundtrack into v3/direction/audio/ (the class mix of film/audio, the same time map)")
    b.add_argument("--hand-events", default=None, help=f"the hand-contact events the 'connected' rule reads (default {HAND_EVENTS.relative_to(PACKAGE_ROOT)})")
    c = sub.add_parser("check")
    c.add_argument("--direction", default=str(OUT / "direction.json"))
    k = sub.add_parser("clips", help="UNITY STAGE ONLY: export the phones' video the v3 direction needs into the capture (camera_video, under the Unity lock)")
    k.add_argument("--no-install", action="store_true", help="write the master only (work/<take>/camera_video/<capture>); mirror later with camera_video --install-only")
    k.add_argument("--priority", choices=("below_normal", "normal"), default="normal")
    sub.add_parser("sheet", help="redraw flow_sheet.png from a fresh build (nothing else written)")
    a = ap.parse_args(argv)
    if a.cmd == "clips":
        return cmd_clips(a.no_install, a.priority)
    if a.cmd == "sheet":
        r = build(write=False)
        print(sheet(r, OUT / "flow_sheet.png"))
        return 0
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
    return cmd_check(Path(a.direction))


if __name__ == "__main__":
    raise SystemExit(main())
