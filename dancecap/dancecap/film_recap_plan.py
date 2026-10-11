"""The directing logic of the class-recap film: what the camera and the teaching overlays do, second by second, as a function of the speech.

Pure functions over plain data (no Unity, no file I/O except the loaders at the bottom), so every rule is testable:

  speech (final transcript utterances, teaching events, chapters)  +  activity (are the teachers dancing or standing?)  +  layers (which
  analysis data the capture has)   ->   beats (one camera treatment each)   ->   inserts (slow motion / replays, with a sound policy)
  ->   segments (the film direction's segments with annotations, chapter cards, film times)   ->   a time map (master seconds <-> film seconds)
  the captions use.

Rules (docs/RECAP_FILM.md has the rationale):
  * The SPEECH drives the film. A shot starts LEAD_S before the sentence it illustrates and holds until the sentence ends; the cut is a
    camera blend of BLEND_S that is settled when the words begin. No shot change inside a sentence unless a long sentence carries two
    different teaching points (then the move happens at a clause break).
  * Treatments follow the teaching events: weight / counterbalance -> physics view; axis, exit, rotation -> overhead with the leader's T;
    path, direction, space -> overhead of the floor; feet, steps -> feet close-up; timing -> medium-close with touchdown rings; body parts
    -> a closer orbit with a call-out; everything else (the teachers just talking) -> a calm wide orbit that follows the speaker.
    Overlays that depict MOTION (feet, axis, path, timing) are used only when the teachers actually move there (activity); a body-part or
    weight call-out may also illustrate a still posture.
  * An overlay whose analysis data is absent is dropped and listed (`dropped`), the treatment falls back to a calmer one.
  * Slow motion and replays only when a teacher ASKS to watch ('see', 'watch', 'look at this', 'slowly', 'one more time', 'again' while
    dancing) and only over a demonstration (a burst of motion). The teachers' voices are only ever heard at 1.0x locked to their picture:
    a slow passage is either placed in a speech-free gap (`gap_slow`) or inserted at a sentence end while the speech is parked
    (`replay`); both play the room-tone bed instead of the class sound.
"""
from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------ tunables


@dataclass
class Cfg:
    lead_s: float = 0.3              # a shot starts this long before the words it illustrates
    blend_s: float = 0.7             # camera blend between shots (settled when the words begin)
    min_shot_s: float = 3.2
    talk_radius_m: float = 3.6       # the calm orbit's radius when the teachers stand together (T3: the 9:16 frame was 60 % empty at 4.2 m; fit_span widens it when they part)
    max_run_s: float = 12.5          # a non-talk treatment holds at most this long, then the camera changes angle (a variant)
    max_unit_s: float = 11.0         # a sentence longer than this may be split at a clause break between two different teaching points
    min_split_part_s: float = 3.5
    score_threshold: float = 1.8     # below this a unit is just talk
    motion_speed: float = 0.25       # m/s: a teacher moving faster than this (median over the window) is "dancing"
    motion_frac: float = 0.2         # ... for at least this share of the window
    apart_m: float = 0.95            # the teachers further apart than this are standing apart (no hold)
    burst_speed: float = 0.45        # m/s: a demonstration burst
    burst_min_s: float = 1.6
    # slow motion / replays
    max_inserts: int = 4
    max_added_s: float = 24.0
    replay_speed: float = 0.5
    core_max_s: float = 3.6
    insert_spacing_s: float = 25.0
    allow_slowing_cue: bool = False  # 'slowing down' is dance advice, not a request to slow the picture
    allow_again_cue: bool = False    # a plain 'again' describes the structure of the dance ('starting here again'), it does not ask to watch
    weight_motion_frac: float = 0.05 # a weight call-out needs at least a little movement (a lean) in the window
    gap_min_s: float = 1.8           # a speech-free gap this long can be slowed in place (the slowed window is gap - 2 x gap_pad_s - 0.1)
    gap_pad_s: float = 0.35          # keep this far from the neighbouring words
    # call-outs
    arrow_min_conf: float = 0.5
    arrow_s: float = 2.0
    arrow_min_s: float = 1.3         # an arrow squeezed by its neighbours is never shorter than this
    arrow_max_s: float = 2.4
    label_repeat_s: float = 40.0
    max_arrows_per_10s: int = 3
    card_s: float = 2.6
    # D4 (owner 2026-10-11): "more arrows" - one arrow per attention call that has something to point at (film_recap_arrows), at most this many at once
    arrows_every_call: bool = True
    vector_min_speed: float = 0.1    # m/s: a centre of mass slower than this has no direction of travel to show (the viewer hides a velocity arrow below 0.05)
    max_arrows_at_once: int = 2
    arrow_same_label_s: float = 6.0   # the same label on the same target within this many seconds is one arrow (its window is extended)
    # D4 (owner 2026-10-11, "when avatars are in view, we need to set to 75% transparency"): ONE value for every avatar the film draws (3D shots,
    # physics, overhead, the flight into a phone). 0.25 opacity = 75 % transparent, read literally; 0.75 would be the other reading. The direction
    # carries it as the top-level key `avatar_opacity` (the viewer applies it to every segment's `avatars` layer).
    avatar_opacity: float = 0.25
    avatar_on_video: bool = False    # the 3D avatars also over a phone's own video (layer `avatars_over_video`); off: only the skeleton lines sit on the video
    # overhead framing (D2): the couple fills this share of the frame width at the tightest; never below min_fill
    overhead_fill: float = 0.66
    overhead_min_fill: float = 0.46
    # R10 (2026-10-11): the 16:9 render. The shots were designed for 9:16; the camera blocks carry an opt-in `horizontal` block (the viewer reads it only
    # while rendering 16:9, so the 9:16 render is byte-identical with or without it). `horizontal_overrides` = False omits it (the old 16:9 framing).
    horizontal_overrides: bool = True
    h_overhead_fill: float = 0.7     # 16:9 overhead: the couple's joints + margin fill this share of the WIDTH at the tightest (the height binds in a wide frame)
    h_overhead_min_fill: float = 0.66
    h_overhead_margin_m: float = 0.16
    h_overhead_never: float = 0.97
    h_overhead_follow_s: float = 0.45  # the camera's damping of the couple's centre (9:16: 0.8 s; a looser look point inflates the extent the frame must hold)
    h_physics_distance_m: float = 3.4
    h_rhythm_band: tuple = (0.2, 0.9)
    h_physics_band: tuple = (0.21, 0.88)   # the couple's floor-to-top span between the captions (bottom 10 %) and the top safe margin, 70 % of the height
    h_orbit_fov_scale: float = 0.8   # 16:9 talk / body orbits: the lens narrowed to this share (fit_span keeps both teachers inside 90 % of the width)
    h_feet_distance_m: float = 3.0
    h_feet_fov_deg: float = 52.0
    h_portrait_zoom: float = 1.3     # 16:9: factor on the crop zoom of a portrait phone clip (the viewer caps the result at 2.1x: the clips are 480-720 px wide)
    phone_side_fill: float = 0.42     # 16:9: brightness of the blurred copy of a portrait phone clip behind / beside the centred clip (0 = plain black)


# ------------------------------------------------------------------------------------------------ activity


@dataclass
class Activity:
    """30 Hz measurements of the two teachers (lead, follow): `sep` = their distance on the floor (m), `speed` = the faster one's floor speed (m/s)"""
    hz: float = 30.0
    sep: np.ndarray | None = None
    speed: np.ndarray | None = None
    source: str = "none"

    @property
    def known(self) -> bool:
        return self.sep is not None and len(self.sep) > 0

    def _sl(self, t0: float, t1: float) -> slice:
        n = len(self.sep)
        return slice(max(0, int(round(t0 * self.hz))), min(n, max(int(round(t1 * self.hz)), int(round(t0 * self.hz)) + 1)))

    def window(self, t0: float, t1: float, cfg: Cfg) -> dict:
        if not self.known:
            return {"known": False}
        s = self._sl(t0, t1)
        sep, spd = self.sep[s], self.speed[s]
        if sep.size == 0:
            return {"known": False}
        return {"known": True, "sep_med": float(np.median(sep)), "sep_max": float(np.max(sep)), "speed_med": float(np.median(spd)),
                "speed_p90": float(np.percentile(spd, 90)), "moving_frac": float(np.mean(spd > cfg.motion_speed)),
                "apart": bool(np.median(sep) > cfg.apart_m)}

    def moving(self, t0: float, t1: float, cfg: Cfg) -> bool | None:
        """True when the teachers dance / move in the window, False when they stand, None when unknown"""
        w = self.window(t0, t1, cfg)
        if not w["known"]:
            return None
        return w["moving_frac"] >= cfg.motion_frac

    def bursts(self, t0: float, t1: float, cfg: Cfg) -> list[dict]:
        """demonstration bursts in [t0, t1]: stretches above burst_speed (gaps up to 0.8 s bridged), longer than burst_min_s"""
        if not self.known:
            return []
        s = self._sl(t0, t1)
        spd = self.speed[s]
        on = spd >= cfg.burst_speed
        out, i, n = [], 0, len(on)
        gap = int(0.8 * self.hz)
        while i < n:
            if not on[i]:
                i += 1
                continue
            j = i
            last = i
            while j < n and (on[j] or j - last <= gap):
                if on[j]:
                    last = j
                j += 1
            a, b = i, last + 1
            if (b - a) / self.hz >= cfg.burst_min_s:
                seg = spd[a:b]
                out.append({"t0": s.start / self.hz + a / self.hz, "t1": s.start / self.hz + b / self.hz, "mean": float(seg.mean()),
                            "peak_t": s.start / self.hz + (a + int(np.argmax(seg))) / self.hz})
            i = max(j, i + 1)
        return out

    @classmethod
    def from_track3d(cls, track_npz: Path, roles_npz: Path | None = None, hz: float = 30.0) -> Activity:
        """the teachers' triangulated 3D tracks (people/track3d.npz) and the role decision (roles.npz state 0 = slot A is the lead)"""
        z = np.load(track_npz, allow_pickle=True)
        x = z["X"].astype(np.float64)                                   # (T, 2, 17, 3) COCO joints, y up
        t = x.shape[0]
        state = np.zeros(t, dtype=int)
        if roles_npz is not None and Path(roles_npz).exists():
            r = np.load(roles_npz)
            state = np.asarray(r["state"])[:t]
            if len(state) < t:
                state = np.pad(state, (0, t - len(state)), mode="edge")
        pel = (x[:, :, 11, :] + x[:, :, 12, :]) / 2.0
        lead = np.where((state == 0)[:, None], pel[:, 0], pel[:, 1])
        fol = np.where((state == 0)[:, None], pel[:, 1], pel[:, 0])
        return cls.from_positions(lead, fol, hz, "people/track3d.npz")

    @classmethod
    def from_positions(cls, lead: np.ndarray, fol: np.ndarray, hz: float = 30.0, source: str = "positions") -> Activity:
        """lead / follow positions (T, 3) with y up (NaN = not tracked): distance on the floor and the faster one's floor speed"""
        def fill(a: np.ndarray) -> np.ndarray:
            a = a.copy()
            idx = np.arange(len(a))
            for c in range(a.shape[1]):
                ok = np.isfinite(a[:, c])
                if ok.sum() > 2:
                    a[:, c] = np.interp(idx, idx[ok], a[ok, c])
                else:
                    a[:, c] = 0.0
            return a

        lead, fol = fill(lead), fill(fol)
        k = max(1, int(round(0.3 * hz)))
        ker = np.ones(k) / k

        def smooth(v: np.ndarray) -> np.ndarray:
            return np.convolve(np.pad(v, (k, k), mode="edge"), ker, mode="same")[k:-k]

        def speed(p: np.ndarray) -> np.ndarray:
            sx, sz = smooth(p[:, 0]), smooth(p[:, 2])
            return np.hypot(np.gradient(sx), np.gradient(sz)) * hz

        sep = np.hypot(lead[:, 0] - fol[:, 0], lead[:, 2] - fol[:, 2])
        return cls(hz=hz, sep=sep, speed=np.maximum(speed(lead), speed(fol)), source=source)

    @classmethod
    def from_floorcraft(cls, doc: dict) -> Activity:
        """the capture's floorcraft.json `track` (the T axis origin's speed): a coarse fallback, separation unknown (0)"""
        tr = doc.get("track") or {}
        spd = tr.get("speed_m_s")
        if not spd:
            return cls()
        step = float(tr.get("step_s") or 0.0667)
        hz = 1.0 / step
        arr = np.asarray(spd, dtype=float)
        return cls(hz=hz, sep=np.zeros(len(arr)), speed=arr, source="floorcraft.json track")


# ------------------------------------------------------------------------------------------------ layers (which analysis data exists)

ANALYSIS_LAYERS = ("physics", "counterbalance", "floorcraft", "timing", "graph")
FILES = {"physics": "physics.json", "counterbalance": "counterbalance.json", "floorcraft": "floorcraft.json", "timing": "timing.json",
         "graph": "moves/graph.json"}
PLANNED = ("physics", "counterbalance", "floorcraft", "timing")      # what the analysis agents are making for the recap (no graph, no beats)


@dataclass
class Layers:
    """which analysis layers the capture has; `assumed` = the capture does not exist yet, the planned set is assumed"""
    available: dict[str, bool] = field(default_factory=dict)
    assumed: bool = False
    phones: bool = False                       # the source-phone videos are exported (cameras/NN.track.json)
    note: str = ""

    def has(self, *names: str) -> bool:
        return all(self.available.get(n, False) for n in names)

    def any(self, *names: str) -> bool:
        return any(self.available.get(n, False) for n in names)

    @classmethod
    def probe(cls, capture_dir: Path | None, assume_planned: bool = True) -> Layers:
        if capture_dir is not None and Path(capture_dir).is_dir():
            avail = {k: (Path(capture_dir) / f).exists() for k, f in FILES.items()}
            phones = any(Path(capture_dir, "cameras").glob("*.track.json")) if Path(capture_dir, "cameras").is_dir() else False
            return cls(avail, False, phones, f"probed {Path(capture_dir).name}")
        if assume_planned:
            return cls({k: k in PLANNED for k in ANALYSIS_LAYERS}, True, False,
                       "the capture does not exist yet: the planned analysis set (physics, counterbalance, floorcraft, timing) is ASSUMED")
        return cls({k: False for k in ANALYSIS_LAYERS}, False, False, "no capture, nothing assumed")

    @classmethod
    def all_on(cls) -> Layers:
        return cls({k: True for k in ANALYSIS_LAYERS}, False, True, "all layers on (test)")


# ------------------------------------------------------------------------------------------------ treatments

# family -> what it needs (any of) and whether it needs motion
FAMILIES = {
    "talk": {"needs": (), "motion": False},
    "body": {"needs": (), "motion": False},
    "feet": {"needs": (), "motion": True},
    "weight": {"needs": ("physics", "counterbalance"), "motion": False},
    "axis": {"needs": ("floorcraft",), "motion": True},
    "path": {"needs": ("floorcraft",), "motion": True},
    "timing": {"needs": ("timing", "floorcraft"), "motion": True},
    "graph": {"needs": ("graph",), "motion": False},
}
FAMILY_ORDER = ["weight", "axis", "feet", "path", "timing", "body", "graph"]      # tie-break
TAG_FAMILY = {
    "overlay_weight_change": ("weight", 3.0),
    "overlay_T_axis": ("axis", 3.0),
    "overlay_path/floorcraft": ("path", 2.4),
    "overlay_footsteps/physics": ("feet", 2.8),
    "overlay_timing": ("timing", 2.4),
    "graph_moves": ("graph", 1.6),
}
FEET_WHERE = {"feet", "legs"}
FEET_WORDS_SET = FEET_WHERE | {"feet"}

# layer presets (the vocabulary of the demo film's layers: FilmDirector.ReadLayers)
# D2 (2026-10-10): the audit read the 0.55 / 0.42 avatars under the skeleton overlay as translucent grey ghosts (her white top and light jeans
# vanish): the class film shows the dancers themselves, the overlays sit on top
L_TALK = {"avatars": 0.95, "skeleton_mode": "rhythm", "floor": True, "floorcraft": "off", "counterbalance": False, "traces": False,
          "neck": False, "moves": False, "graph_inset": False, "hud": False}
L_BODY = {"avatars": 0.92, "skeleton_mode": "rhythm", "floor": True, "floorcraft": "off", "counterbalance": False, "traces": False,
          "neck": False, "spine_beads": True, "moves": False, "graph_inset": False, "hud": False}
L_FEET = {"avatars": 0.85, "skeleton_mode": "rhythm", "floorcraft": "off", "counterbalance": False, "footprints": "recent", "traces": False,
          "moves": False, "graph_inset": False}
L_WEIGHT = {"avatars": 0.5, "skeleton_mode": "physics", "balance": True, "counterbalance": True, "physics": True,
            "legend": "Skeleton load - ESTIMATED from motion", "floorcraft": "off", "moves": False, "graph_inset": False}
L_AXIS = {"avatars": 0.6, "skeleton_mode": "rhythm", "floorcraft": "current_t", "counterbalance": True, "footprints": "recent",
          "moves": False, "graph_inset": False, "grid": False, "t_width_m": 0.05, "t_brightness": 1.0}
# T3: "full_record" drew every T of the take so far (36 by the end: a bright tangle over a couple 1.9 m wide in 9:16); the faint record stays under current_t
L_PATH = {"avatars": 0.6, "skeleton_mode": "rhythm", "floorcraft": "current_t", "counterbalance": False, "footprints": "recent",
          "moves": False, "graph_inset": False, "grid": False}
L_TIMING = {"avatars": 0.85, "skeleton_mode": "rhythm", "footprints": "touchdown_rings_neutral", "floorcraft": "off", "counterbalance": False,
            "moves": False, "graph_inset": False, "spine_beads": True}
# a phone's point of view: the dancers are in the video, the 3D layer is only the skeleton on top (the demo's POV used 0.35 avatars; no floor under the video)
L_POV = {"avatars": 0.95, "avatars_fade_with_video": True, "skeleton_mode": "rhythm", "floor": False, "grid": False, "floorcraft": "off", "counterbalance": False, "traces": False,
         "neck": False, "moves": False, "graph_inset": False, "hud": False, "hand_contacts": False}
L_GRAPH = {"dance_full_size": 0.0, "graph": "full", "miniature_couple": True, "path_trail": True, "moves": True, "graph_inset": False,
           "floorcraft": "off", "hand_contacts": False}

VIEW_STATE = {"pov": "camera_tour", "talk": "orbit", "body": "orbit", "feet": "orbit", "weight": "physics", "axis": "overhead", "path": "overhead",
              "timing": "orbit", "graph": "dance_graph"}


def treatment_layers(family: str, layers: Layers, dropped: list[dict] | None = None, where: str = "", variant: int = 0,
                     avatar_opacity: float | None = None, avatar_on_video: bool = False) -> dict:
    """the layer dict of a family, minus the keys whose data is absent (recorded in `dropped`). `avatar_opacity` (D4, the film's ONE avatar value)
    replaces every positive `avatars` value of the preset; the skeleton lines and the overlays keep their own strength."""
    base = {"pov": L_POV, "talk": L_TALK, "body": L_BODY, "feet": L_FEET, "weight": L_WEIGHT, "axis": L_AXIS, "path": L_PATH, "timing": L_TIMING,
            "graph": L_GRAPH}[family]
    out = deepcopy(base)
    if avatar_opacity is not None and float(out.get("avatars", 0.0)) > 0.0:
        out["avatars"] = round(float(avatar_opacity), 3)
    if family == "pov" and avatar_on_video:
        out["avatars_over_video"] = True
    if family == "axis" and variant % 2 == 1:
        out = deepcopy(L_PATH)                    # the floor view of the alternation: every T and the dotted path
        out["counterbalance"] = True
        if avatar_opacity is not None:
            out["avatars"] = round(float(avatar_opacity), 3)

    def drop(key: str, needs: str, replacement=None):
        if key in out:
            if dropped is not None:
                dropped.append({"where": where, "what": f"layer {key}={out[key]!r}", "needs": needs})
            if replacement is None:
                del out[key]
            else:
                out[key] = replacement

    if not layers.has("physics"):
        if out.get("skeleton_mode") == "physics":
            drop("skeleton_mode", "physics.json", "rhythm")
        for k in ("balance", "physics", "legend"):
            if k in out:
                drop(k, "physics.json")
    if not layers.has("counterbalance") and out.get("counterbalance"):
        drop("counterbalance", "counterbalance.json", False)
    if not layers.has("floorcraft"):
        if out.get("floorcraft") not in (None, "off"):
            drop("floorcraft", "floorcraft.json", "off")
        if out.get("footprints") in ("recent", "all") and not layers.has("timing"):
            drop("footprints", "floorcraft.json / timing.json")
    if not layers.any("timing", "floorcraft") and str(out.get("footprints", "")).startswith("touchdown"):
        drop("footprints", "timing.json / floorcraft.json")
    if not layers.has("graph") and out.get("graph") not in (None, "off"):
        drop("graph", "moves/graph.json", "off")
    return out


def with_horizontal(cam: dict, cfg: "Cfg | None" = None) -> dict:
    """the camera block plus its opt-in `horizontal` overrides (the keys the viewer swaps in while rendering 16:9). Phone shots (their frame is the
    clip's, cropped by `crop`) and the modes that already fit themselves (orbit `fit_span`, rhythm, feature zoom) are left alone."""
    cfg = cfg or Cfg()
    out = dict(cam)
    if not cfg.horizontal_overrides or "horizontal" in out:
        return out
    mode = out.get("mode")
    h: dict = {}
    if mode == "physics_3q":
        # fit_band's band is a 9:16 one (couple between the physics legend and the caption block: 34 %..74 % of the height); in a wide frame the
        # couple took 40 % of the height and 8 % of the width. 17 %..87 % keeps the heads under the top margin and the feet above the captions.
        h = {"distance_m": cfg.h_physics_distance_m, "band_lo": cfg.h_physics_band[0], "band_hi": cfg.h_physics_band[1]}
    elif mode == "overhead_floorcraft" and out.get("framing") == "couple" and out.get("fill"):
        fill = float(out["fill"])
        # the 9:16 fills (0.60 / 0.66) are shares of a narrow frame's width; a wide frame is bound by its HEIGHT (a couple's footprint is about square):
        # `parallax` counts the head and raised hands as nearer the lens, so nothing is cropped; fill = the width share at the tightest
        h = {"parallax": True, "fill": round(cfg.h_overhead_fill - (0.04 if fill < cfg.overhead_fill - 0.01 else 0.0), 3), "min_fill": cfg.h_overhead_min_fill,
             "margin_m": cfg.h_overhead_margin_m, "never": cfg.h_overhead_never, "follow_s": cfg.h_overhead_follow_s}
    elif mode == "orbit" and out.get("fit_span"):
        # the calm talk / body orbit: a lens that makes a couple 2.3 m tall fill 72 % of the height leaves two standing teachers a third of a wide frame;
        # narrowed (fit_span widens it again whenever they stand apart: both always stay inside 90 % of the width)
        h = {"fov_scale": cfg.h_orbit_fov_scale}
    elif mode == "rhythm":
        # the finale's orbit fits a 2.2 m column between 30 % and 94 % of the height: a 1.9 m couple fills half of a wide frame's height
        h = {"band_lo": cfg.h_rhythm_band[0], "band_hi": cfg.h_rhythm_band[1]}
    elif mode == "feet_closeup":
        h = {"distance_m": cfg.h_feet_distance_m, "fov_deg": cfg.h_feet_fov_deg}
    if h:
        out["horizontal"] = h
    return out


def camera_for(family: str, *, variant: int = 0, focus: str = "couple", focus_weight: float = 0.0, apart: bool = False, cfg: Cfg | None = None) -> dict:
    """the camera block of a family. `variant` alternates the direction of travel between uses; every block is deterministic."""
    cfg = cfg or Cfg()
    sign = 1 if variant % 2 == 0 else -1
    blend = round(cfg.blend_s, 2)
    if family == "talk":
        # avoid_inline (D2): the orbit never looks along the line from one dancer to the other (the closing orbit of the first render hid Kadu
        # behind Larissa's back for 4 s)
        c = {"mode": "orbit", "radius_m": 4.8 if apart else cfg.talk_radius_m, "height_m": 1.75, "look_height_m": 1.0, "period_s": sign * 80,
             "fit_span": True, "avoid_inline": True, "blend_s": blend}
        if focus in ("lead", "follow") and focus_weight > 0:
            c.update({"focus": focus, "focus_weight": round(focus_weight, 2)})
        return c
    if family == "body":
        return {"mode": "orbit", "radius_m": 3.1, "height_m": 1.45, "look_height_m": 1.1, "period_s": sign * 60, "fit_span": True, "avoid_inline": True,
                "focus": focus if focus in ("lead", "follow") else "couple", "focus_weight": 0.45 if focus in ("lead", "follow") else 0.0,
                "blend_s": blend}
    if family == "feet":
        # T3 (2026-10-09, after looking at the stills): the 9:16 frame is narrow, so a frame that holds both dancers' feet is 2.2 m tall at 2.7 m
        # and cut the heads at the neck / put the partners' hands in the picture. A low, wide shot instead (both dancers whole, the feet in the
        # lower half where the footprints and the 'right foot' arrow are); the director's `need` keeps both pairs of feet inside the width.
        return {"mode": "feet_closeup", "dancer": "lead", "foot": "both", "height_m": 0.45 if sign > 0 else 0.75, "distance_m": 3.6,
                "look_height_m": 0.62 if sign > 0 else 0.68, "fov_deg": 60, "azimuth_offset_deg": 25 * sign, "side": "outside_turn", "blend_s": blend}
    if family == "weight":
        return {"mode": "physics_3q", "azimuth_deg": sign * 35, "drift_deg_s": sign * 4, "blend_s": blend}
    if family == "axis":
        # alternate: the couple from above with the current T / the whole floor with every T and the path
        # T3: the 'floor' framing of the viewer fits the whole take's foot area (about 25 m up in this class: the couple was a speck); the couple
        # framing with a wider footprint (`across_m`, the viewer widens it further when the dancers' feet are apart) keeps the T and the path readable
        # D2 (owner 2026-10-10, "for overhead views get closer with the camera"): `fill` = the share of the frame WIDTH the couple's four limbs' extent
        # fills at the tightest (the audit measured about a quarter); the viewer follows the dancers' spread per frame (a pivot or an exit is never
        # cropped: the instantaneous extent always fits at `min_fill`-plus margin), `across_m` stays for viewers that predate `fill`
        return {"mode": "overhead_floorcraft", "framing": "couple", "across_m": 2.2 if sign > 0 else 2.8, "fill": cfg.overhead_fill if sign > 0 else cfg.overhead_fill - 0.06,
                "min_fill": cfg.overhead_min_fill, "blend_s": blend}
    if family == "path":
        return {"mode": "overhead_floorcraft", "framing": "couple", "across_m": 2.8, "fill": cfg.overhead_fill - 0.06, "min_fill": cfg.overhead_min_fill,
                "blend_s": blend}
    if family == "timing":
        return {"mode": "rhythm", "blend_s": blend}
    if family == "graph":
        return {"mode": "graph_wide", "distance_m": 8, "height_m": 3.6, "period_s": 90, "graph_focus_distance_scale": 0.5, "blend_s": blend}
    raise KeyError(family)


# ------------------------------------------------------------------------------------------------ speech helpers

SPEAKER_KEYS = {"Larissa": "speaker_larissa", "Kadu": "speaker_kadu"}


def event_families(ev: dict) -> list[tuple[str, float]]:
    """(family, weight) pairs of one teaching event (its tags, the body part for camera_focus)"""
    out = []
    where = (ev.get("focus") or {}).get("where")
    for tag in ev.get("tags", []):
        if tag in TAG_FAMILY:
            fam, w = TAG_FAMILY[tag]
            if fam == "feet" and where not in FEET_WHERE and "footwork" not in ev.get("kinds", []):
                fam, w = "weight", 2.2          # 'footsteps/physics' without a foot is the physics of the move
            out.append((fam, w))
        elif tag == "camera_focus":
            out.append(("feet", 2.4) if where in FEET_WHERE else ("body", 2.0 if where else 1.2))
    return out


def event_conf_factor(ev: dict) -> float:
    c = float(ev.get("conf") or 0.0)
    f = 0.4 + 0.6 * max(0.0, min(1.0, c))
    return f * (0.85 if ev.get("needs_check") else 1.0)


def score_events(events: list[dict]) -> dict[str, float]:
    sc: dict[str, float] = {}
    for ev in events:
        k = event_conf_factor(ev)
        for fam, w in event_families(ev):
            sc[fam] = sc.get(fam, 0.0) + w * k
    return sc


def pick_family(scores: dict[str, float], cfg: Cfg) -> tuple[str, float]:
    best, best_s = "talk", 0.0
    for fam in FAMILY_ORDER:
        s = scores.get(fam, 0.0)
        if s > best_s + 1e-9:
            best, best_s = fam, s
    return (best, best_s) if best_s >= cfg.score_threshold else ("talk", best_s)


@dataclass
class Beat:
    id: str
    t0: float
    t1: float
    family: str
    chapter: int
    speaker: str | None = None
    focus: str = "couple"
    score: float = 0.0
    why: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    chapter_start: bool = False
    utterances: list[str] = field(default_factory=list)
    variant: int = 0
    activity: dict = field(default_factory=dict)
    demanded: bool = False                    # a shot change inside the sentence was demanded by the events
    cuts: list[float] = field(default_factory=list)    # unit boundaries inside the beat (where it may be split)
    starts_inner: bool = False                # the beat starts inside a sentence (a clause split demanded by two teaching points)
    inner_cuts: list[float] = field(default_factory=list)   # clause boundaries inside the beat (never used to split a run)
    # director mode (film_recap_director): the shot's own camera block / treatment instead of camera_for(family)
    camera: dict | None = None
    treatment: dict = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


def chapter_of(t: float, chapters: list[dict]) -> int:
    idx = 0
    for i, c in enumerate(chapters):
        if t >= c["start"] - 1e-6:
            idx = i
    return idx


def role_of(speaker: str | None, roles: dict) -> str:
    r = roles.get(speaker or "", "")
    return "lead" if r == "leader" else "follow" if r == "follower" else "couple"


def decide_unit(unit: dict, events: list[dict], act: Activity, layers: Layers, cfg: Cfg, dropped: list[dict], notes: list[str]) -> tuple[str, float, list[str], list[str]]:
    """family for one unit (a sentence or part of one): (family, score, reasons, event ids)"""
    evs = [e for e in events if unit["t0"] - 0.2 <= e["start"] <= unit["t1"] + 0.2]
    scores = score_events(evs)
    fam, sc = pick_family(scores, cfg)
    reasons = []
    ids = [e["id"] for e in evs]
    if evs:
        reasons.append("events " + ", ".join(f"{e['id']}:{'/'.join(e.get('triggers', []))}" for e in evs[:5]) + (" ..." if len(evs) > 5 else ""))
    w = act.window(unit["t0"] - 0.5, unit["t1"] + 1.5, cfg)
    unit_act = w
    # a family that depicts motion needs motion
    tries = [fam] + [f for f, _ in sorted(scores.items(), key=lambda kv: -kv[1]) if f != fam]
    for cand in tries:
        if cand == "talk":
            fam = "talk"
            break
        spec = FAMILIES[cand]
        if scores.get(cand, 0.0) < cfg.score_threshold:
            continue
        if spec["motion"] and w["known"] and w["moving_frac"] < cfg.motion_frac and w["speed_p90"] < cfg.burst_speed * 0.6:
            reasons.append(f"{cand} dropped: the teachers stand still here (moving {w['moving_frac']:.0%})")
            continue
        if cand == "weight" and w["known"] and w["moving_frac"] < cfg.weight_motion_frac and w["speed_p90"] < 0.18:
            reasons.append(f"weight dropped: no movement at all here (p90 {w['speed_p90']:.2f} m/s)")
            continue
        if spec["needs"] and not layers.any(*spec["needs"]):
            dropped.append({"where": unit["id"], "what": f"treatment {cand}", "needs": " or ".join(FILES[n] for n in spec["needs"])})
            reasons.append(f"{cand} dropped: no {' / '.join(spec['needs'])} data")
            continue
        fam = cand
        sc = scores[cand]
        break
    else:
        fam = "talk"
    if fam == "talk":
        sc = 0.0
    if w["known"] and w.get("apart") and fam in ("body", "feet", "weight", "axis", "timing"):
        reasons.append(f"{fam} -> talk: the teachers stand {w['sep_med']:.1f} m apart")
        fam, sc = "talk", 0.0
    return fam, sc, reasons, ids


def split_long_unit(u: dict, events: list[dict], act: Activity, layers: Layers, cfg: Cfg, dropped: list[dict], notes: list[str]) -> list[dict]:
    """A sentence longer than max_unit_s that carries two different teaching points becomes two units at the clause break between them."""
    if u["t1"] - u["t0"] <= cfg.max_unit_s:
        return [u]
    evs = [e for e in events if u["t0"] <= e["start"] <= u["t1"] and event_conf_factor(e) >= 0.8]
    words = u.get("words") or []
    if len(evs) < 2 or len(words) < 6:
        return [u]
    fams = [(e, pick_family(score_events([e]), cfg)[0]) for e in evs]
    cut = None
    for (e1, f1), (e2, f2) in zip(fams, fams[1:]):
        if f1 == f2 or e2["start"] - e1["end"] < 3.0:
            continue
        mid = 0.5 * (e1["end"] + e2["start"])
        # clause breaks: gaps between words >= 0.12 s or a word ending in punctuation
        cands = []
        for a, b in zip(words, words[1:]):
            gap = b["start"] - a["end"]
            if a["end"] > e1["end"] + 0.3 and b["start"] < e2["start"] - 0.3 and (gap >= 0.12 or str(a.get("punct", "")).strip() or a["text"].endswith((",", ".", "?", "!"))):
                cands.append((abs(0.5 * (a["end"] + b["start"]) - mid), 0.5 * (a["end"] + b["start"])))
        if cands:
            t = min(cands)[1]
            if t - u["t0"] >= cfg.min_split_part_s and u["t1"] - t >= cfg.min_split_part_s:
                cut = t
                break
    if cut is None:
        return [u]
    a = {**u, "id": u["id"], "t1": cut, "demanded_split": True, "words": [w for w in words if w["end"] <= cut + 1e-6]}
    b = {**u, "id": u["id"] + "'", "t0": cut, "demanded_split": True, "words": [w for w in words if w["start"] >= cut - 1e-6]}
    notes.append(f"{u['id']} ({u['t1'] - u['t0']:.1f} s) split at {cut:.2f} s: {fams[0][1]} -> {fams[-1][1]}")
    return split_long_unit(a, events, act, layers, cfg, dropped, notes) + split_long_unit(b, events, act, layers, cfg, dropped, notes)


def plan_beats(utterances: list[dict], events: list[dict], chapters: list[dict], act: Activity, layers: Layers, roles: dict, duration: float,
               cfg: Cfg | None = None) -> tuple[list[Beat], dict]:
    """beats (contiguous from 0 to `duration`, chapter starts are beat starts) and a report {dropped, notes}"""
    cfg = cfg or Cfg()
    dropped: list[dict] = []
    notes: list[str] = []
    units: list[dict] = []
    for u in sorted(utterances, key=lambda x: x["start"]):
        if u["end"] <= 0:
            continue
        uu = {**u, "t0": max(0.0, u["start"]), "t1": min(duration, u["end"])}
        if uu["t1"] - uu["t0"] <= 0.05:
            continue
        units.extend(split_long_unit(uu, events, act, layers, cfg, dropped, notes))
    # chapter starts must start a unit: a unit that straddles one is split there (starts were snapped to sentence starts, so this is rare)
    cstarts = [c["start"] for c in chapters if c["start"] > 0]
    fixed: list[dict] = []
    for u in units:
        cut = [c for c in cstarts if u["t0"] + 0.4 < c < u["t1"] - 0.4]
        if cut:
            notes.append(f"{u['id']} straddles the chapter start {cut[0]:.2f} s: split there")
            fixed.append({**u, "t1": cut[0]})
            fixed.append({**u, "id": u["id"] + "~", "t0": cut[0]})
        else:
            fixed.append(u)
    units = fixed

    decided = []
    for u in units:
        fam, sc, why, ids = decide_unit(u, events, act, layers, cfg, dropped, notes)
        decided.append({**u, "family": fam, "score": sc, "why": why, "event_ids": ids})

    # shot boundaries between consecutive units
    n = len(decided)
    bounds = [0.0]
    for i in range(1, n):
        prev, cur = decided[i - 1], decided[i]
        gap = cur["t0"] - prev["t1"]
        if gap >= cfg.lead_s:
            b = cur["t0"] - cfg.lead_s
        elif gap > 0:
            b = prev["t1"] + gap / 2
        else:
            b = cur["t0"]
        bounds.append(max(bounds[-1], b))
    bounds.append(duration)

    beats: list[Beat] = []
    for i, u in enumerate(decided):
        t0, t1 = bounds[i], bounds[i + 1]
        spk = u.get("speaker")
        ch = chapter_of(u["t0"], chapters)
        b = Beat(id="", t0=t0, t1=t1, family=u["family"], chapter=ch, speaker=spk, score=u["score"], why=list(u["why"]), events=list(u["event_ids"]),
                 utterances=[u["id"]], activity=act.window(t0, t1, cfg), demanded=bool(u.get("demanded_split")))
        b.starts_inner = i > 0 and decided[i - 1]["id"].rstrip("'~") == u["id"].rstrip("'~")
        beats.append(b)

    # the film starts at 0: the lead-in before the first words belongs to the first beat (already: bounds[0] = 0)
    beats = merge_beats(beats, chapters, cfg, notes)
    beats = split_runs(beats, act, cfg, notes)
    assign_variation(beats, chapters, act, roles, cfg)
    return beats, {"dropped": dropped, "notes": notes}


def _join(p: Beat, b: Beat, at: float) -> None:
    """record the boundary `at` between p and b (and b's own cuts) in p; a boundary inside a sentence is kept apart"""
    (p.inner_cuts if b.starts_inner else p.cuts).append(at)
    p.cuts += b.cuts
    p.inner_cuts += b.inner_cuts


def merge_beats(beats: list[Beat], chapters: list[dict], cfg: Cfg, notes: list[str]) -> list[Beat]:
    """same-family neighbours of one chapter join; a beat shorter than min_shot_s joins the neighbour with the higher score (a chapter start
    is never merged backwards)"""
    cstart = {round(c["start"], 2) for c in chapters}

    def starts_chapter(b: Beat) -> bool:
        return any(abs(b.t0 - (c["start"] - cfg.lead_s)) < 0.45 or abs(b.t0 - c["start"]) < 0.45 for c in chapters) or b.t0 <= 1e-6

    out: list[Beat] = []
    for b in beats:
        b.chapter_start = starts_chapter(b) and (not out or b.chapter != out[-1].chapter)
        if out and out[-1].family == b.family and not b.chapter_start and out[-1].chapter == b.chapter:
            p = out[-1]
            _join(p, b, b.t0)
            p.t1 = b.t1
            p.score = max(p.score, b.score)
            p.why += b.why
            p.events += b.events
            p.utterances += b.utterances
            p.demanded = p.demanded or b.demanded
            if b.speaker != p.speaker:
                p.speaker = p.speaker if (p.t1 - p.t0) >= 2 * (b.t1 - b.t0) else b.speaker
            continue
        out.append(b)
    changed = True
    while changed:
        changed = False
        for i, b in enumerate(out):
            if b.dur >= cfg.min_shot_s or len(out) == 1:
                continue
            prev = out[i - 1] if i > 0 and not b.chapter_start else None
            nxt = out[i + 1] if i + 1 < len(out) and not out[i + 1].chapter_start else None
            if prev is None and nxt is None:
                continue
            target = prev if nxt is None else nxt if prev is None else (prev if prev.score >= nxt.score else nxt)
            if target is prev:
                _join(prev, b, b.t0)
                prev.t1 = b.t1
            else:
                (nxt.inner_cuts if nxt.starts_inner else nxt.cuts).append(b.t1)
                nxt.cuts += b.cuts
                nxt.inner_cuts += b.inner_cuts
                nxt.starts_inner = b.starts_inner
                nxt.t0 = b.t0
                nxt.chapter_start = nxt.chapter_start or b.chapter_start
            target.events = sorted(set(target.events + b.events))
            target.utterances = target.utterances + b.utterances if target is prev else b.utterances + target.utterances
            target.why.append(f"absorbed a {b.dur:.1f} s {b.family} beat")
            notes.append(f"beat {b.t0:.1f}-{b.t1:.1f} ({b.family}) shorter than {cfg.min_shot_s} s joined the {target.family} neighbour")
            del out[i]
            changed = True
            break
    # same-family neighbours may have become adjacent
    merged: list[Beat] = []
    for b in out:
        if merged and merged[-1].family == b.family and not b.chapter_start and merged[-1].chapter == b.chapter:
            _join(merged[-1], b, b.t0)
            merged[-1].t1 = b.t1
            merged[-1].events = sorted(set(merged[-1].events + b.events))
            merged[-1].utterances += b.utterances
            merged[-1].score = max(merged[-1].score, b.score)
            continue
        merged.append(b)
    for i, b in enumerate(merged):
        if i + 1 < len(merged):
            merged[i + 1].t0 = merged[i].t1
    return merged


def split_runs(beats: list[Beat], act: Activity, cfg: Cfg, notes: list[str]) -> list[Beat]:
    """A treatment other than talk holds at most max_run_s: a longer beat is cut at its inner unit boundaries (sentence starts) into near-equal
    parts, each part a different camera angle (the variant alternates), so a long explanation does not sit on one picture. Talk is a slow orbit
    and may run on."""
    out: list[Beat] = []
    for b in beats:
        if b.family == "talk" or b.dur <= cfg.max_run_s:
            out.append(b)
            continue
        n = int(-(-b.dur // cfg.max_run_s))
        cuts = sorted(c for c in set(round(x, 3) for x in b.cuts) if b.t0 + cfg.min_shot_s <= c <= b.t1 - cfg.min_shot_s)
        chosen: list[float] = []
        for k in range(1, n):
            target = b.t0 + k * b.dur / n
            prev = chosen[-1] if chosen else b.t0
            ok = [c for c in cuts if c - prev >= cfg.min_shot_s and b.t1 - c >= cfg.min_shot_s]
            if not ok:
                break
            within = [c for c in ok if c - prev <= cfg.max_run_s]          # a piece never longer than max_run_s when a cut allows it
            chosen.append(min(within or ok, key=lambda c: abs(c - target)))
        if not chosen:
            out.append(b)
            continue
        edges = [b.t0] + chosen + [b.t1]
        notes.append(f"{b.family} beat {b.t0:.1f}-{b.t1:.1f} ({b.dur:.1f} s) cut into {len(edges) - 1} angles at {', '.join(f'{c:.1f}' for c in chosen)} s")
        for i in range(len(edges) - 1):
            piece = Beat(id="", t0=edges[i], t1=edges[i + 1], family=b.family, chapter=b.chapter, speaker=b.speaker, score=b.score,
                         why=list(b.why) + [f"angle {i + 1}/{len(edges) - 1} of a long {b.family} explanation"], events=list(b.events),
                         chapter_start=b.chapter_start and i == 0, utterances=list(b.utterances), activity=act.window(edges[i], edges[i + 1], cfg),
                         cuts=[c for c in b.cuts if edges[i] < c < edges[i + 1]],
                         inner_cuts=[c for c in b.inner_cuts if edges[i] < c < edges[i + 1]])
            out.append(piece)
    return out


def assign_variation(beats: list[Beat], chapters: list[dict], act: Activity, roles: dict, cfg: Cfg) -> None:
    """ids, the direction of travel alternates between uses of a family, the camera focuses on who is talking (or who the event names)"""
    used: dict[str, int] = {}
    per_chapter: dict[int, int] = {}
    for b in beats:
        k = per_chapter.get(b.chapter, 0)
        per_chapter[b.chapter] = k + 1
        # the first beat of a chapter has a stable id (the library's chapter marks anchor at it, whatever treatment a rebuild picks)
        b.id = f"ch{b.chapter:02d}_{k + 1:02d}_open" if k == 0 else f"ch{b.chapter:02d}_{k + 1:02d}_{b.family}"
        b.variant = used.get(b.family, 0)
        used[b.family] = b.variant + 1
        b.focus = role_of(b.speaker, roles)


# ------------------------------------------------------------------------------------------------ slow motion and replays

CUE_STRONG = re.compile(r"\b(see|watch(?:ing)?|look(?:ing)? at (?:it|this|that|me|him|her|how)|notice|slowly|slow motion|in slow)\b", re.I)
CUE_AGAIN = re.compile(r"\bagain\b", re.I)
CUE_MORE = re.compile(r"\b(one more time|once more|one more)\b", re.I)
CUE_SLOWING = re.compile(r"\bslow(?:ing)? down\b", re.I)
NOT_A_REQUEST = re.compile(r"\b(look at my|looking for|you see\b|see right|see,? okay|i see)\b", re.I)


def find_cues(utterances: list[dict], cfg: Cfg) -> list[dict]:
    """the sentences in which a teacher ASKS to look / repeat: {utt, t, strength, cue}"""
    out = []
    for u in utterances:
        if u["end"] <= 0:
            continue
        text = u["text"]
        if NOT_A_REQUEST.search(text) and not CUE_STRONG.search(NOT_A_REQUEST.sub("", text)):
            continue
        for rx, strength, name in ((CUE_STRONG, 3.0, "watch"), (CUE_MORE, 2.2, "one more time"),
                                   (CUE_AGAIN, 1.6 if cfg.allow_again_cue else 0.0, "again"), (CUE_SLOWING, 1.2 if cfg.allow_slowing_cue else 0.0, "slowing")):
            if strength <= 0:
                continue
            m = rx.search(text)
            if not m:
                continue
            t = u["start"]
            first = m.group(0).split()[0].lower().strip(",.?!")
            for w in u.get("words", []):
                if re.sub(r"[^a-z']", "", w["text"].lower()) == re.sub(r"[^a-z']", "", first):
                    t = w["start"]
                    break
            out.append({"utt": u["id"], "t": max(0.0, t), "strength": strength, "cue": name, "text": m.group(0), "u_start": u["start"], "u_end": u["end"]})
            break
    return out


def speech_spans(utterances: list[dict], join_s: float = 0.35) -> list[list[float]]:
    ws = sorted((w["start"], w["end"]) for u in utterances for w in u.get("words", []) if w.get("end", 0) > 0 and w["start"] >= 0)
    spans: list[list[float]] = []
    for s, e in ws:
        if spans and s - spans[-1][1] < join_s:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return spans


def speech_free(spans: list[list[float]], a: float, b: float, pad: float = 0.0) -> bool:
    return all(e + pad <= a or s - pad >= b for s, e in spans)


def plan_inserts(utterances: list[dict], act: Activity, duration: float, cfg: Cfg | None = None) -> tuple[list[dict], list[dict]]:
    """Slow-motion inserts. Returns (inserts, rejected) - each insert {kind 'gap_slow' | 'replay', cue, core [a, b] (master s), speed, park (master s,
    replay only), added_s, why}. Sound policy: the speech is never slowed; `gap_slow` lives inside a speech-free gap, `replay` is inserted at a
    sentence end with the speech parked; both play the room-tone bed. Needs a known activity (a demonstration burst) after a cue."""
    cfg = cfg or Cfg()
    rejected: list[dict] = []
    if not act.known:
        return [], [{"why": "no activity data: nothing is slowed without proof of a demonstration"}]
    spans = speech_spans(utterances)
    utts = sorted(utterances, key=lambda u: u["start"])
    ends = sorted({u["end"] for u in utts if u["end"] > 0})
    cands = []
    for cue in find_cues(utterances, cfg):
        w0, w1 = cue["t"] - 0.8, cue["t"] + 6.5
        bursts = [b for b in act.bursts(max(0.0, w0), min(duration, w1), cfg)]
        if not bursts:
            rejected.append({"cue": cue["text"], "t": round(cue["t"], 2), "why": "no demonstration burst within 6.5 s after the cue"})
            continue
        best = max(bursts, key=lambda b: (b["t1"] - b["t0"]) * b["mean"])
        # the core: up to core_max_s around the burst's peak
        c0 = max(best["t0"], best["peak_t"] - cfg.core_max_s / 2)
        c1 = min(best["t1"], c0 + cfg.core_max_s)
        c0 = max(best["t0"], c1 - cfg.core_max_s)
        if c1 - c0 < cfg.burst_min_s:
            rejected.append({"cue": cue["text"], "t": round(cue["t"], 2), "why": "burst too short"})
            continue
        # gap_slow: a speech-free gap right after the cue, inside the burst or just after it
        gap_ins = None
        for ga, gb in zip([s[1] for s in spans], [s[0] for s in spans[1:]]):
            if gb - ga < cfg.gap_min_s or ga < cue["t"] - 0.2 or ga > cue["t"] + 6.0:
                continue
            a, b = ga + cfg.gap_pad_s, gb - cfg.gap_pad_s - 0.1
            if b - a < 0.8:
                continue
            wg = act.window(a, b, cfg)
            if wg["known"] and wg["speed_p90"] >= cfg.burst_speed:
                gap_ins = {"kind": "gap_slow", "core": [round(a, 3), round(b, 3)], "speed": cfg.replay_speed,
                           "added_s": round((b - a) / cfg.replay_speed - (b - a), 3)}
                break
        if gap_ins:
            cands.append({**gap_ins, "cue": cue, "score": cue["strength"] * 1.3 + best["mean"], "why": f"{cue['cue']} '{cue['text']}' at {cue['t']:.1f} s, motion in the speech-free gap"})
            continue
        # replay: park at a sentence end >= max(core end, the cue sentence's end) within 8 s
        t_min = max(c1, cue["u_end"])
        park = None
        for e in ends:
            if e >= t_min - 0.05 and e <= t_min + 8.0:
                # a pause long enough, or the sentence ends: the next speech must start after the park point
                park = e + 0.12
                break
        if park is None or park >= duration - 1.0:
            rejected.append({"cue": cue["text"], "t": round(cue["t"], 2), "why": "no sentence end within 8 s after the demonstration"})
            continue
        if not any(abs(park - 0.12 - e) < 0.01 for e in ends):
            continue
        cands.append({"kind": "replay", "core": [round(c0, 3), round(c1, 3)], "speed": cfg.replay_speed, "park": round(park, 3),
                      "added_s": round((c1 - c0) / cfg.replay_speed, 3), "cue": cue, "score": cue["strength"] + best["mean"],
                      "why": f"{cue['cue']} '{cue['text']}' at {cue['t']:.1f} s, demonstration {best['t0']:.1f}-{best['t1']:.1f} s replayed after the sentence"})
    cands.sort(key=lambda c: -c["score"])
    chosen: list[dict] = []
    added = 0.0
    for c in cands:
        at = c.get("park", c["core"][0])
        if len(chosen) >= cfg.max_inserts:
            rejected.append({"cue": c["cue"]["text"], "t": round(c["cue"]["t"], 2), "why": "over the insert budget"})
            continue
        if added + c["added_s"] > cfg.max_added_s:
            rejected.append({"cue": c["cue"]["text"], "t": round(c["cue"]["t"], 2), "why": f"over the {cfg.max_added_s} s budget"})
            continue
        if any(abs(at - x.get("park", x["core"][0])) < cfg.insert_spacing_s for x in chosen):
            rejected.append({"cue": c["cue"]["text"], "t": round(c["cue"]["t"], 2), "why": "too close to another insert"})
            continue
        chosen.append(c)
        added += c["added_s"]
    chosen.sort(key=lambda c: c.get("park", c["core"][0]))
    for i, c in enumerate(chosen):
        c["id"] = f"slow{i + 1}"
    return chosen, rejected


# ------------------------------------------------------------------------------------------------ the time map


class TimeMap:
    """master (dance) seconds <-> film seconds. Locked parts map with a constant offset (the sum of the inserts before them); a
    `gap_slow` window stretches by its added seconds; a `replay` parks the speech (the offset grows at its park point)."""

    def __init__(self, inserts: list[dict]):
        pts = []        # (master time where the extra starts, extra seconds, kind)
        for ins in inserts:
            if ins["kind"] == "replay":
                pts.append((ins["park"], ins["added_s"]))
            else:
                pts.append((ins["core"][1], ins["added_s"]))        # the whole stretch is added once the window has passed
        self.pts = sorted(pts)

    def extra(self, t: float) -> float:
        return sum(a for p, a in self.pts if p <= t + 1e-9)

    def film(self, t: float) -> float:
        return t + self.extra(t)

    @property
    def total_extra(self) -> float:
        return sum(a for _, a in self.pts)


# ------------------------------------------------------------------------------------------------ call-outs

BODY_ARROWS = {
    # where: {who: (dancer, joint)}; "*" = any
    "frame": {"*": ("follow", "left_shoulder")},
    "shoulders": {"*": ("follow", "left_shoulder")},
    "arm": {"leader": ("lead", "right_elbow"), "follower": ("follow", "left_elbow"), "*": ("lead", "right_elbow")},
    "hands": {"*": ("lead", "right_wrist")},
    "hip": {"*": ("follow", "pelvis")},
    "chest": {"*": ("follow", "spine3")},
    "upper body": {"*": ("follow", "spine3")},
    "connection": {"*": ("lead", "right_wrist")},
    "legs": {"*": ("lead", "left_knee")},
}
LABELS = {
    ("legs",): "legs engaged", ("leg",): "leg", ("frame",): "frame", ("hip",): "hip", ("arm",): "arm", ("hand",): "hands", ("hands",): "hands",
    ("connection",): "connection", ("connected",): "connection", ("open chest",): "open chest", ("upper body",): "upper body",
    ("chest",): "chest", ("thighs",): "thighs", ("shoulder", "blades"): "shoulder blades", ("shoulders",): "shoulders",
}
WEIGHT_LABELS = {"counterbalance": "counterbalance", "weight": "weight", "weights": "weight", "heavy": "heavy", "balance": "balance",
                 "shift": "weight shift", "suspension": "suspension"}
MOVE_TERMS = {   # the teacher's word -> the label (docs/ZOUK_TERMINOLOGY.md: canonical pt-BR name where one is decided)
    "lateral": "Corredor", "simple turn": "Giro Simples", "bonus": "Bônus", "gostosinho": "Gostosinho", "sacada": "Sacada", "lunge": "Lunge",
    "basic": "Basic step", "rotate": "Rotation", "rotation": "Rotation",
}
ORDINALS = {"first": "First", "second": "Second", "third": "Third", "fourth": "Fourth", "last": "Last"}


def label_for(ev: dict) -> str | None:
    trig = tuple(t.lower() for t in ev.get("triggers", []))
    if trig in LABELS:
        return LABELS[trig]
    for t in trig:
        if (t,) in LABELS:
            return LABELS[(t,)]
    return None


def enumeration_text(ev: dict) -> str | None:
    """'second exit' / 'first option' / 'last one' -> 'Second exit' (the teacher's words)"""
    for t in ev.get("triggers", []):
        w = t.lower().split()
        if len(w) == 2 and w[0] in ORDINALS and w[1] in ("option", "exit", "part", "one", "step"):
            return f"{ORDINALS[w[0]]} {w[1] if w[1] != 'one' else 'one'}"
    return None


def make_annotations(beat: Beat, events: list[dict], layers: Layers, cfg: Cfg, dropped: list[dict], state: dict) -> list[dict]:
    """The call-outs of one beat (dance_t windows in master seconds). `state` carries the last label times across beats."""
    out: list[dict] = []
    evs = [e for e in events if e["id"] in set(beat.events)]
    arrows_recent: list[float] = state.setdefault("arrow_times", [])
    last_label: dict[str, float] = state.setdefault("last_label", {})
    n = 0

    def window(e: dict, length: float) -> tuple[float, float]:
        a = max(beat.t0 + 0.05, e["start"] - 0.35)
        b = min(beat.t1 - 0.05, a + length)
        return round(a, 3), round(b, 3)

    def need(ids: str, anno_id: str, *layer_names: str) -> bool:
        if layers.any(*layer_names):
            return True
        dropped.append({"where": beat.id, "what": f"annotation {anno_id}", "needs": " or ".join(FILES.get(x, x) for x in layer_names)})
        return False

    for e in evs:
        conf = float(e.get("conf") or 0.0)
        a0 = e["start"]
        if not (beat.t0 <= a0 <= beat.t1):
            continue
        tags = set(e.get("tags", []))
        kinds = set(e.get("kinds", []))
        enum = enumeration_text(e) if "enumeration" in kinds else None
        if enum and a0 - state.get("last_enum", -99) > 3.0:
            wa, wb = window(e, 2.8)
            if wb - wa >= 0.8:
                out.append({"id": f"{beat.id}_e{n}", "kind": "readout", "dance_t": [wa, wb], "text": enum})
                state["last_enum"] = a0
                n += 1
            continue
        if "move_name" in kinds:
            trig = " ".join(e.get("triggers", [])).lower()
            term = MOVE_TERMS.get(trig)
            if term and conf >= 0.6 and a0 - state.get("last_move", -99) > 3.0:
                wa, wb = window(e, 2.6)
                if wb - wa >= 0.8:
                    out.append({"id": f"{beat.id}_m{n}", "kind": "readout", "dance_t": [wa, wb], "text": term})
                    state["last_move"] = a0
                    n += 1
            continue
        if cfg.arrows_every_call:
            continue                      # D4: the arrows come from film_recap_arrows (one per attention call)
        if conf < cfg.arrow_min_conf or e.get("needs_check"):
            continue
        recent = [t for t in arrows_recent if a0 - 10.0 < t <= a0]
        if len(recent) >= cfg.max_arrows_per_10s or any(a0 - t < cfg.arrow_s for t in arrows_recent):
            continue
        # body parts / feet call-outs
        where = (e.get("focus") or {}).get("where")
        who = (e.get("focus") or {}).get("who", "couple")
        if beat.family == "feet" and where in FEET_WHERE | {"feet"} and "camera_focus" in tags | {"overlay_footsteps/physics"}:
            lab = "right foot" if "right foot" in [t.lower() for t in e.get("triggers", [])] else "left foot" if "left foot" in [t.lower() for t in e.get("triggers", [])] else None
            if lab:
                side = lab.split()[0]
                tgt = {"type": "foot", "dancer": "follow" if who == "follower" else "lead", "side": side}
                key = lab + tgt["dancer"]
                if a0 - last_label.get(key, -999) >= cfg.label_repeat_s:
                    wa, wb = window(e, cfg.arrow_s)
                    out.append({"id": f"{beat.id}_a{n}", "kind": "arrow3d", "target": tgt, "text": lab, "dance_t": [wa, wb]})
                    last_label[key] = a0
                    arrows_recent.append(a0)
                    n += 1
            continue
        if beat.family in ("body", "talk") and "camera_focus" in tags and where in BODY_ARROWS:
            lab = label_for(e)
            if not lab:
                continue
            dancer, joint = BODY_ARROWS[where].get(who, BODY_ARROWS[where]["*"])
            if lab in last_label and a0 - last_label[lab] < cfg.label_repeat_s:
                continue
            wa, wb = window(e, cfg.arrow_s)
            out.append({"id": f"{beat.id}_a{n}", "kind": "arrow3d", "target": {"type": "joint", "dancer": dancer, "joint": joint}, "text": lab,
                        "dance_t": [wa, wb]})
            last_label[lab] = a0
            arrows_recent.append(a0)
            n += 1
            continue
        if beat.family == "weight" and "overlay_weight_change" in tags:
            trig = [t.lower() for t in e.get("triggers", [])]
            lab = next((WEIGHT_LABELS[t] for t in trig if t in WEIGHT_LABELS), None)
            if lab and lab not in last_label or (lab and a0 - last_label[lab] >= cfg.label_repeat_s * 0.5):
                if need("w", f"{beat.id}_w{n}", "physics"):
                    wa, wb = window(e, cfg.arrow_s + 0.4)
                    out.append({"id": f"{beat.id}_w{n}", "kind": "arrow3d", "target": {"type": "couple_com"}, "text": lab, "dance_t": [wa, wb]})
                    last_label[lab] = a0
                    arrows_recent.append(a0)
                    n += 1
    if beat.family == "weight" and layers.any("physics") and beat.dur >= 2.5:
        out.append({"id": f"{beat.id}_coms", "kind": "com_marker", "dance_t": [round(beat.t0 + 0.4, 3), round(beat.t1 - 0.2, 3)],
                    "targets": [{"type": "com", "dancer": "lead"}, {"type": "com", "dancer": "follow"}, {"type": "couple_com"}]})
    elif beat.family == "weight" and not layers.any("physics"):
        dropped.append({"where": beat.id, "what": "annotation com_marker", "needs": "physics.json"})
    return out
