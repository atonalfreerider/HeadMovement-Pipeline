"""Floor craft record of a take: the leader's floor axis over the dance, segmented into PLANTS, MOVES and ROTATIONS,
with the counterbalance PIVOTS in the same time-ordered record - what the viewer accumulates as a fading history on
the floor (head-movement docs/VIEWER_SPEC.md 3.5 floor craft, 3.6 leader floor axis, 3.9 counterbalance).

    python -m dancecap.floorcraft <take> [--export-capture <name>] [--smplx-dir <dir>] [--no-review]
                                         [--review-dir <dir>]
        -> work/<take>/floorcraft/floorcraft.json         reference seconds, world frame (contract below)
           work/<take>/review/floorcraft_map.png           overhead map of the whole take
           work/<take>/review/floorcraft_timeline.png      the segmentation over time
           work/<take>/review/taxis_map.png, taxis_timeline.png   the T axis, dials, turn verdicts (version 3)
           --export-capture: <StreamingAssets>/<name>/floorcraft.json (+ work/<take>/export/<name>/ when present):
           capture seconds, Unity coordinates; capture.json gets "floorcraft": "floorcraft.json"
    python -m dancecap.floorcraft <take> --export-only --export-capture <name>     (no re-analysis)

Inputs: work/<take>/smplx/{lead,follow}.npz (contract: Y-up right-handed world, metres, floor y = 0, `t` = reference
seconds; the follower's fit for the balance axis / dial centres / couple COM),
the beat grid (as dancecap.counterbalance: timing/timing.json or beats.json; one beat = two zouk eighths) and
work/<take>/counterbalance/counterbalance.json (re-detected first when missing or computed on other fits).

LEADER FLOOR AXIS (VIEWER_SPEC 3.6), per frame:
  origin   the leader's chest (SMPL-X spine3 joint) projected to the floor (y = 0);
  long     the chest's left-right line projected to the floor = the +x axis of spine3's GLOBAL rotation (global_orient
           x spine1 x spine2 x spine3 local rotations; no body model needed). It follows the shoulder joint line within
           ~9 deg (p90) on larissa_kadu_38 but does not swing with a reaching or shrugging shoulder. Fallback without
           pose parameters: the shoulder joints (16 - 17);
  forward  the floor perpendicular to `long` on the chest's front side (not the projected chest normal, so a forward
           lean does not shorten or flip it);
  yaw      angle of `forward` on the floor from +x towards -z: ccw seen from above = positive about +Y (right-handed
           world; the counterbalance arc_start_deg convention), unwrapped over time.
Origin and the long-axis unit vector are smoothed by the zero-phase spline of physics_spline at cutoff_hz (2 Hz keeps
every step - beats are 1.3 Hz at 76 BPM - and removes the fit jitter); yaw comes from the smoothed vector.

SEGMENTATION (FCConfig; durations in beats follow the tempo). The track is first CUT at every counterbalance pivot:
the pivot owns the leader's motion inside it (he circles her anchored foot and his axis winds round with him), so no
plant, move or rotation overlaps a pivot; that motion is recorded on the pivot (leader_axis). Then, outside the pivots:
  PLANT     the leader holds the dance in one place and establishes an axis: a maximal span of >= plant_min_beats
            (1 beat, and >= plant_min_s) in which the axis origin stays within plant_radius_m (0.20 m) of the span's
            centroid AND its net drift - the slope of a least-squares line through the span - is <= plant_drift_m_s
            (0.15 m/s). Greedy from the earliest frame: the longest span from there that passes, then on after it;
            then neighbouring plants that are ONE stand are merged (merge_plants: their union passes the same test,
            or they touch - no travel between them - with centroids within plant_radius_m: union trimmed to pass),
            and the STEP between two touching stands that are not one place is cut out of their edges (expose_steps).
            The greedy search's running radius is not monotone, and it lets each plant absorb up to 0.20 m of a
            transit, so on its own it cut one stand in two (larissa_kadu_38 HUD 10.4-14.4: two plants 0.15 m apart
            whose union has radius 0.16 m) and split single steps between the two stands they join (a foot steps
            0.3-0.45 m, the chest follows at 0.6-0.8 m/s right at the junction) leaving 1-3 frame 'moves'.
            Why these numbers (larissa_kadu_38, leader chest origin): median speed 0.50 m/s, p10 0.10-0.13 m/s; a
            travelling step carries the chest 0.3-0.5 m in a beat (0.4-0.6 m/s); a weight change / basic in place
            sways it +-10-15 cm with ~0 net drift. 0.20 m holds one in-place weight change but not two travelling
            steps; 0.15 m/s is < 12 cm net per beat, under a third of a travelling step, so a slow travel is not
            chopped into a row of plants (the radius alone admits 0.5 m/s over one beat).
            Record: centroid origin, radius (max / p90 distance), drift, yaw at its start / end (0.2 s means), the
            net turn inside it, and the axis it ESTABLISHED: yaw_deg = the circular-mean yaw over its longest stretch
            without a rotation in progress (`held`; yaw_source 'held'), the plain circular mean (yaw_mean_deg) when no
            rotation interrupts it or rotations fill it ('mean ...') - the mean over a plant with a 130 deg turn in it
            is neither the heading he came in with nor the one he settled on.
  MOVE      the axis travels from one plant to the next (before the first / after the last plant: from the take edge,
            or from / to a pivot: from_pivot / to_pivot). kind 'step': between two plants, the origin's own path
            (origin_path_m) < step_max_m (0.5 m) within step_max_beats (1 beat) - one step to the next stand (the
            step window of expose_steps; when the two plants only pass the plant test with the step's halves in
            them, the step sits at their junction: a 1-3 frame record joining the two crosses); kind 'travel':
            everything else. Record: path = from (plant origin) -> the origin sampled at path_step_s -> to (plant
            origin), so a renderer drawing `path` alone joins the plant crosses; its length (length_m), the
            plant-to-plant displacement and heading, and the net yaw change from the previous plant's end to the
            next plant's start (unwrapped, so a circling move keeps its full turns).
  ROTATION  the axis yaw changes by >= rotation_deg (25 deg): the swings of a zig-zag (hysteresis) filter on the
            smoothed, unwrapped yaw - every monotone excursion of >= 25 deg between two turning points; smaller
            sway (contra-body motion of a basic, +-10-15 deg) is not one. Edges trimmed to where the yaw leaves the
            start value / reaches the end value within min(5 deg, 10 % of the swing) (plateaus before / after do not
            count). Split where a swing crosses from a plant group into a travel move or back, so each rotation
            belongs to one context. Plants chained by single steps are one group (a turn while stepping to the next
            stand stays whole), owned by the plant or step holding most of its angle: context 'plant' (`plant`;
            centre = its origin) or 'move' (`move` = that step or the travel move; centre = the axis origin
            mid-rotation); `plants` = every plant it touches.
  PIVOT     every counterbalance pivot of counterbalance.json (pivot point = the follower's anchored foot, the
            leader's circling path and swept angle, beat ticks, confidence) and every counterbalance interval
            (combined-COM floor point), copied with the same ids into this record; leader_axis = his chest path on the
            floor and the net turn of his axis over the pivot (the motion cut out of the moves / rotations).
`events` lists every record by start time (type, id, t0, t1; moves also kind) for the viewer's accumulating floor record.

VERSION 3 (user 2026-10-07: "the arrow T should find a stable position ... an averaged position that represents the
parallel vector and the normal vector to the leader's chest orientation"; schema
work/larissa_kadu_38/review/floorcraft2/SCHEMA_V2.md, CONTRACTS.md):
  T AXIS    t_episodes / t_axis_records: the axis track above segmented into T episodes, each with ONE averaged
            origin and heading. A T FORMS over a window of t_form_beats (its grow-in on screen); its origin / heading
            are then AVERAGED over all its active frames inside its region (median chest on the floor, mean heading;
            the end re-checked against the averaged T, t_refine_iter) - the averaged position he occupied, not the
            first beat (review 2026-10-07: the forming window alone left the long Ts 0.28-0.33 m from where he danced
            on them; now 0.05-0.07 m). It holds while his chest stays in the ALLOWED REGION (a plus: the stem corridor
            along forward, |side| <= t_half_width_m and |forward| <= t_forward_m - walking back and forth; the crossbar
            corridor along parallel, |forward| <= t_half_width_m and |side| <= t_side_m - side to side) with his
            heading within t_pivot_deg. A run of >= t_sustain_beats outside the region ('exit') or turned by more
            than t_pivot_deg ('rotation') ends it. A T that would hold for < t_min_hold_beats after forming (it would
            only flash: grow in and hand over at once - 6 of 13 Ts before this rule) is DROPPED; its heading stays a
            WAYPOINT of the axis pivots. Consecutive Ts are joined by TRANSITIONS (the dotted line). AXIS PIVOTS =
            every heading change > t_pivot_deg between two Ts (through the waypoints, so a turn out and back while no
            T holds is two pivots), before the first T (lead_in), into and out of a counterbalance pivot (its dial
            owns only the circling inside it) and after the last T (lead_out); in place when the new T forms within
            t_same_place_m of the old (same_place_as, origin copied), else at his chest mid-turn. No T forms inside a
            counterbalance pivot.
            Why these numbers (larissa_kadu_38, smplx_rf2_noslip): every T of this take ends by ROTATION (the couple
            turns almost continuously, about -2400 deg in 38 s), so the corridor size does not change its
            segmentation anywhere in 0.20-0.40 m half-width / 0.9-1.5 m length (8 Ts, the same breaks); 0.30 m is an
            engineering choice: the chest sways +-10-15 cm sideways in a forward-back basic (plus fit jitter) while a
            diagonal step carries it 0.2-0.35 m across (the synthetic tests: one step out and back keeps the T, 1 m
            diagonally makes a new one). 1.2 m = 2-3 travelling steps from the centre. 30 deg = the user's threshold
            (25 / 35 move one break), twice the +-10-15 deg contra-body sway of a basic. Sustain 1 beat: a single step
            out and back or a contra-body swing does not end a T (0.75 beat: the same Ts; 1.25: one break moves).
            Forming over 1 beat with every heading within 20 deg (< 30, so a formed T is not broken at once; 15 deg
            moves no T) and a cross-axis net drift <= 0.20 m/s (a travelling step drifts 0.4-0.6 m/s): no T forms
            while he is still arriving diagonally. Minimum hold 1 beat (0.75: the same Ts; 1.5 drops two more).
  DIALS     find_dials: counterbalance pivots (centre = her anchored foot) and repeated turning about one fixed point:
            chained same-sign axis pivots summing to >= dial_min_deg (180) while the leader's chest ('leader_spin'),
            the couple's combined COM ('couple_turn') or the follower's chest ('leader_around_follower') - tried from
            the least moving - stays within dial_place_m (p90) AND, for the last two, the leader really goes ROUND it
            (his orbit about it has the turn's sign and >= dial_orbit_ratio of its degrees; review 2026-10-07: at HUD
            16.1-17.3 s the couple COM stayed put while the couple swapped places +132 deg and he spun -286 deg under
            the arms: no couple turn). A chain that fails as a whole is tried in its contiguous sub-chains. The
            turning into / out of a counterbalance is never a dial (its own dial is right there).
  BALANCE   balance_axis / couple_axis: per frame and dancer, the axis chest (spine3) -> support (nearest point of the
            stance foot's heel-toe line to the COM), its COM offset / tilt and a 0..1 score (the physics-view line's
            brightness); the couple's combined COM over her anchored foot in a counterbalance pivot.
            Why these numbers: a foot is ~9-10 cm wide, so a COM within 4 cm of the sole's centre line is over the
            foot (score 1) and 12 cm is clearly off it (0); 3 / 9 deg of tilt = 6 / 19 cm at chest height (1.2-1.3
            m). On larissa_kadu_38 in single support the COM is 2 / 10 / 21 cm (p10 / p50 / p90) from the stance
            foot's sole line for the lead (2 / 9 / 26 cm for the follow: walking moves the COM between the feet), and
            inside the user-labelled counterbalance the combined COM is 1-4 cm from her anchored foot while it is
            established (HUD 32.6-35.2 s) and 15-27 cm while entering / leaving it (her own COM then hangs 25-32 cm
            off her foot: the counterbalance).
  EVALUATION evaluate_turn: share of the turn's frames (widened to >= bal_eval_min_beats) with score >= 0.5 (COM <= 8
            cm off, tilt <= 6 deg): good >= 60 %, ok >= 35 %, else needs work; for spins the leader, for a
            counterbalance the couple (combined COM over her anchored foot), for couple turns both (the worse).
            Reorientations (axis pivots, rotations) are CLASSIFIED first (classify_turn: 'one_foot' pivot / turned on
            'both_feet' / 'stepped'); the user asked whether the axis is over 'the pivot point of the foot', so only a
            pivot on one foot is graded (over the frames with his weight on it) - the others say 'not graded' and keep
            the COM numbers (balance_verdict). Dials are always graded. ESTIMATES from the fitted bodies.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import counterbalance as cb
from .take import PACKAGE_ROOT

VERSION = 3   # 3 (2026-10-07): + T axis (t_axes, transitions, axis_pivots), dials, balance axis, turn evaluations
REVISION = "3.1"   # 3.1 (review fixes 2026-10-07): averaged Ts, no flashing Ts (waypoints), pivots around the
#                   counterbalance / before the first T, dial orbit check + sub-chains, turn kinds (only one-foot
#                   pivots graded). Additive: every 3.0 key keeps its meaning; from_t / to_t may be null.
#             2 (2026-10-07): one stand = one plant, steps cut out, pivots cut out of moves / rotations, held yaw
SPINE_CHAIN = (3, 6, 9)        # spine1, spine2, spine3 (parents 0 -> 3 -> 6 -> 9)
CHEST, L_SHOULDER, R_SHOULDER = 9, 16, 17
DEFAULT_CAPTURES = dict(cb.DEFAULT_CAPTURES)


@dataclass
class FCConfig:
    cutoff_hz: float = 2.0                 # axis smoothing (origin + long-axis vector), zero-phase spline
    plant_radius_m: float = 0.20           # max distance of the origin to the plant centroid
    plant_drift_m_s: float = 0.15          # max net drift speed (least-squares line through the plant)
    plant_min_beats: float = 1.0
    plant_min_s: float = 0.5
    rotation_deg: float = 25.0             # zig-zag hysteresis: a rotation is a monotone yaw swing >= this
    rotation_edge_deg: float = 5.0         # edge trim tolerance (also <= 10 % of the swing)
    end_window_s: float = 0.2              # plant start / end yaw = mean over this long
    move_min_s: float = 0.1                # lead-in / trailing moves shorter than this are dropped
    touch_path_m: float = 0.10             # two plants whose gap has an own origin path shorter than this ...
    touch_min_s: float = 0.15              # ... or lasts less than this TOUCH: the step between them is cut out
    step_max_m: float = 0.50               # a move between two plants with an own path < this lasting <= ...
    step_max_beats: float = 1.0            # ... this many beats is ONE STEP (kind 'step'), else 'travel'
    path_step_s: float = 1 / 15            # move paths and the axis track are sampled at this step (15 Hz)
    default_bpm: float = 76.4              # when the take has no beat grid
    gap_frames: float = 1.5                # time steps > this x the median step split the grid into segments
    long_axis_m: float = 2.0               # drawing size of the live axis (VIEWER_SPEC 3.6), recorded for the viewer
    short_axis_m: float = 0.8
    # --- T axis (version 3): the leader's stable T and its reorientations (module docstring "T AXIS")
    t_half_width_m: float = 0.30           # corridor half-width about the stem (forward) and the crossbar (parallel)
    t_forward_m: float = 1.20              # the stem corridor reaches +- this along forward
    t_side_m: float = 1.20                 # the crossbar corridor reaches +- this along parallel
    t_pivot_deg: float = 30.0              # a heading > this from the T's (sustained) = a reorientation: a new T
    t_sustain_beats: float = 1.0           # an exit / rotation must last this long (a single step out and back not)
    t_form_beats: float = 1.0              # forming window: origin / heading averaged over this long, then fixed
    t_form_yaw_deg: float = 20.0           # forming window: every heading within this of the window mean
    t_form_drift_m_s: float = 0.20         # forming window: net drift ACROSS both T axes <= this (not arriving
                                           # diagonally; drifting along the stem or the crossbar is dancing on it)
    t_same_place_m: float = 0.25           # a new T whose origin is this close to the previous one is at its place
    t_pivot_edge_deg: float = 15.0         # pivot extent: heading leaves / reaches the T headings within this
    t_draw_forward_m: float = 0.8          # drawing size of a T (recorded for the viewer)
    t_draw_side_m: float = 1.0
    t_refine_iter: int = 2                 # the T's origin / heading = median / mean over ALL its active in-region
                                           # frames (not only the forming window), its end re-checked against it
    t_min_hold_beats: float = 1.0          # a T must hold this long after forming, else it is dropped (it would
                                           # only flash: grow in and hand over at once); the take's last T is kept
    dial_min_deg: float = 180.0            # repeated turning about one point of >= this is a DIAL
    dial_place_m: float = 0.30             # the dial's centre point moves <= this (p90 distance to its median)
    dial_max_hold_beats: float = 2.0       # consecutive pivots of one dial are <= this apart
    dial_orbit_ratio: float = 0.6          # a couple / around-the-follower dial: the leader's orbit about the centre
                                           # has the sign of the turn and >= this x its degrees (he really goes round
                                           # it; else the turn is his own spin while the couple swaps places)
    # --- balance axis (physics view) and turn evaluation
    com_cutoff_hz: float = 5.0             # COM smoothing (as physics / counterbalance)
    bal_contact_h_m: tuple = (0.04, 0.09)  # foot (lower of ankle / ball) above the dancer's floor: contact .. none
    bal_contact_v_m_s: tuple = (0.4, 1.0)  # foot (slower of ankle / ball) horizontal speed: planted .. swinging
    bal_sole_s: tuple = (-0.35, 1.35)      # the sole line = ankle + s (ball - ankle): heel .. toe tip
    bal_good_offset_m: float = 0.04        # COM within this of the sole line: score 1 (half a foot's width)
    bal_bad_offset_m: float = 0.12         # ... beyond this: score 0 (clearly off the foot)
    bal_good_tilt_deg: float = 3.0         # chest -> support line within this of vertical: score 1
    bal_bad_tilt_deg: float = 9.0          # ... beyond this: score 0
    bal_aligned_score: float = 0.5         # a frame is ALIGNED at score >= this (COM <= 8 cm off, tilt <= 6 deg)
    bal_good_share: float = 0.6            # verdict 'good': aligned for >= this share of the turn
    bal_ok_share: float = 0.35             # verdict 'ok': >= this; else 'needs work'
    bal_eval_min_beats: float = 0.5        # a turn is evaluated over >= this long (centred on it): a 0.1 s pivot
    # --- turn kinds: only a turn ON ONE FOOT (and a dial) is graded good / ok / needs work (classify_turn)
    turn_double_share: float = 0.7         # 'both_feet': both feet in contact for >= this share of the turn
    turn_pivot_contact: float = 0.8        # 'one_foot': the pivot foot in contact >= this share of the turn,
    turn_pivot_stance: float = 0.7         # the weight on it (stance) on >= this share of the frames with a stance,
    turn_pivot_travel_m: float = 0.10      # and its ball within this of its median position (planted, it turns on it)


# ----------------------------------------------------------------------------------------------- helpers

def wrap180(a):
    """Degrees -> [-180, 180)."""
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def _r(x, nd=4):
    return cb._r(x, nd)


def yaw_of(fwd: np.ndarray) -> np.ndarray:
    """Floor angle (deg) of (..., 2) [x, z] vectors from +x towards -z (ccw from above)."""
    return np.degrees(np.arctan2(-fwd[..., 1], fwd[..., 0]))


def fwd_of(yaw_deg) -> np.ndarray:
    """(..., 2) [x, z] unit vectors of floor angles (deg, from +x towards -z)."""
    a = np.radians(np.asarray(yaw_deg, float))
    return np.stack([np.cos(a), -np.sin(a)], -1)


def long_of_fwd(f: np.ndarray) -> np.ndarray:
    """The dancer's LEFT on the floor from the forward vector ([x, z]): forward = (-left_z, left_x) inverted."""
    return np.stack([f[..., 1], -f[..., 0]], -1)


def xyz(p2) -> list:
    """[x, z] -> [x, 0, z] (floor point)."""
    p2 = np.asarray(p2, float)
    return _r(np.stack([p2[..., 0], np.zeros(p2.shape[:-1]), p2[..., 1]], -1), 4)


def drift_speed(t: np.ndarray, p: np.ndarray) -> float:
    """|slope| (m/s) of the least-squares straight line through the points p (n, 2) over t (sway averages out)."""
    if len(t) < 3:
        return float("inf")
    tc = t - t.mean()
    den = float((tc ** 2).sum())
    return float(np.linalg.norm((tc[:, None] * (p - p.mean(0))).sum(0) / max(den, 1e-12)))


def drift_vec(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Slope (m/s, [x, z]) of the least-squares straight line through the points p (n, 2) over t."""
    if len(t) < 3:
        return np.full(2, np.inf)
    tc = t - t.mean()
    return (tc[:, None] * (p - p.mean(0))).sum(0) / max(float((tc ** 2).sum()), 1e-12)


def circ_mean_deg(a) -> float:
    r = np.radians(np.asarray(a, float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


# ----------------------------------------------------------------------------------------------- axis

def chest_rotation(npz: dict) -> np.ndarray | None:
    """(T, 3, 3) global rotation of spine3 from the pose parameters (global_orient x spine1 x spine2 x spine3), or
    None when they are missing / not finite."""
    if "global_orient" not in npz or "body_pose" not in npz:
        return None
    from .physics_body import axis_angle_to_matrix
    T = len(npz["t"])
    go = np.asarray(npz["global_orient"], float).reshape(T, 3)
    bp = np.asarray(npz["body_pose"], float).reshape(T, -1, 3)
    if bp.shape[1] < 9:
        return None
    R = axis_angle_to_matrix(go)
    for j in SPINE_CHAIN:
        R = R @ axis_angle_to_matrix(bp[:, j - 1])
    return R


def raw_axis(npz: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """(origin (T, 2) [x, z], left (T, 2) unnormalised [x, z], source) of the leader's floor axis, unsmoothed."""
    t = np.asarray(npz["t"], float)
    if "joints" not in npz:
        raise ValueError("floorcraft: the lead npz has no joints (contract: joints (T, 55, 3))")
    J = cb._bridge(np.asarray(npz["joints"], float)[:, :55], t)
    origin = J[:, CHEST][:, [0, 2]]
    R = chest_rotation(npz)
    if R is not None and np.isfinite(R).mean() > 0.5:
        R = cb._bridge(R, t)
        return origin, R[:, :, 0][:, [0, 2]], "spine3 global rotation (+x = the chest's left, projected to the floor)"
    return origin, (J[:, L_SHOULDER] - J[:, R_SHOULDER])[:, [0, 2]], "shoulder joints 16 - 17 (no pose parameters)"


def smooth_axis(t: np.ndarray, origin: np.ndarray, left: np.ndarray, cutoff_hz: float) -> dict:
    """Smoothed origin, unit long axis (the left), forward, unwrapped yaw (deg), origin speed (m/s) on t."""
    from .physics_spline import Smoother
    if len(t) >= 7:
        S = Smoother(t, cutoff_hz)
        o, L, v = S(origin), S(left), S(origin, 1)
    else:
        o, L, v = origin.copy(), left.copy(), np.gradient(origin, t, axis=0) if len(t) > 1 else np.zeros_like(origin)
    L = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-9)
    f = np.stack([-L[:, 1], L[:, 0]], 1)      # left = +x -> forward = +z (SMPL-X rest pose)
    return {"t": t, "origin": o, "long": L, "forward": f, "yaw": np.degrees(np.unwrap(np.radians(yaw_of(f)))),
            "speed": np.linalg.norm(v, axis=1)}


# ----------------------------------------------------------------------------------------------- segmentation

def plant_ok(t: np.ndarray, o: np.ndarray, i0: int, i1: int, cfg: FCConfig) -> bool:
    """The plant test on frames i0..i1: origin within plant_radius_m of their centroid, net drift <= plant_drift_m_s."""
    P = o[i0:i1 + 1]
    return (float(np.linalg.norm(P - P.mean(0), axis=1).max()) <= cfg.plant_radius_m + 1e-9
            and drift_speed(t[i0:i1 + 1], P) <= cfg.plant_drift_m_s + 1e-9)


def find_plants(t: np.ndarray, o: np.ndarray, beat: float, cfg: FCConfig) -> list[tuple[int, int]]:
    """[(i0, i1 inclusive)] plants: greedy longest spans with the origin within plant_radius_m of their centroid,
    net drift <= plant_drift_m_s and duration >= max(plant_min_s, plant_min_beats x beat); then neighbours that are one
    stand are merged (merge_plants)."""
    T = len(t)
    min_dur = max(cfg.plant_min_s, cfg.plant_min_beats * beat)
    out, i = [], 0
    while i < T - 1:
        k = i
        while k + 1 < T:
            P = o[i:k + 2]
            if float(np.linalg.norm(P - P.mean(0), axis=1).max()) > cfg.plant_radius_m:
                break
            k += 1
        found = None
        m = k
        while m > i and t[m] - t[i] >= min_dur - 1e-9:
            if drift_speed(t[i:m + 1], o[i:m + 1]) <= cfg.plant_drift_m_s:
                found = m
                break
            m -= 1
        if found is None:
            i += 1
            continue
        out.append((i, found))
        i = found + 1
    return expose_steps(t, o, merge_plants(t, o, out, cfg, min_dur), cfg, min_dur, beat)


def degenerate_gap(t: np.ndarray, o: np.ndarray, i0: int, i1: int, cfg: FCConfig) -> bool:
    """True when the origin does not travel between frames i0 and i1 (the end of one plant, the start of the next):
    its own path is < touch_path_m or the gap lasts < touch_min_s: the two plants TOUCH."""
    own = float(np.linalg.norm(np.diff(o[i0:i1 + 1], axis=0), axis=1).sum()) if i1 > i0 else 0.0
    return own < cfg.touch_path_m or float(t[i1] - t[i0]) < cfg.touch_min_s


def trim_to_plant(t: np.ndarray, o: np.ndarray, i0: int, i1: int, cfg: FCConfig, min_dur: float) -> tuple | None:
    """The span i0..i1 shrunk from whichever end lies farther from the centroid until it passes the plant test, or
    None when it falls below min_dur first."""
    while t[i1] - t[i0] >= min_dur - 1e-9 and i1 - i0 >= 2:
        if plant_ok(t, o, i0, i1, cfg):
            return i0, i1
        c = o[i0:i1 + 1].mean(0)
        if np.linalg.norm(o[i0] - c) >= np.linalg.norm(o[i1] - c):
            i0 += 1
        else:
            i1 -= 1
    return None


def merge_plants(t: np.ndarray, o: np.ndarray, spans: list[tuple[int, int]], cfg: FCConfig,
                 min_dur: float = 0.0) -> list[tuple[int, int]]:
    """Merge neighbouring plants that are ONE stand. The greedy search stops a plant where its RUNNING radius first
    exceeds the limit, but the radius of a span is not monotone in its length (the centroid moves as frames are added),
    so one stand can come out as two plants joined by a 1-3 frame 'move':
      - their union (with the frames between) passes the plant test itself -> one plant (larissa_kadu_38 HUD 10.4-14.4:
        two plants 0.15 m apart whose union has radius 0.16 m);
      - or nothing travels between them (degenerate_gap) and their centroids are within plant_radius_m of each other:
        one place, but the earlier plant began in the slow end of the approach (or the later one runs into the slow
        start of the departure) -> the union trimmed at its far end(s) until it passes (trim_to_plant; the trimmed
        frames go to the neighbouring move), kept when it holds at least as long as the longer of the two.
    Two stands farther apart than plant_radius_m with no gap between them are two places: expose_steps cuts the step
    between them out of the plants."""
    out = list(spans)
    k = 0
    while k < len(out) - 1:
        (a0, a1), (b0, b1) = out[k], out[k + 1]
        span = None
        if plant_ok(t, o, a0, b1, cfg):
            span = (a0, b1)
        elif degenerate_gap(t, o, a1, b0, cfg) and \
                np.linalg.norm(o[a0:a1 + 1].mean(0) - o[b0:b1 + 1].mean(0)) <= cfg.plant_radius_m:
            r = trim_to_plant(t, o, a0, b1, cfg, min_dur)
            if r is not None and t[r[1]] - t[r[0]] >= max(t[a1] - t[a0], t[b1] - t[b0]) - 1e-9:
                span = r
        if span is not None:
            out[k:k + 2] = [span]
            k = max(0, k - 1)
        else:
            k += 1
    return out


def expose_steps(t: np.ndarray, o: np.ndarray, spans: list[tuple[int, int]], cfg: FCConfig, min_dur: float,
                 beat: float) -> list[tuple[int, int]]:
    """Cut the step out of two neighbouring plants that touch (degenerate_gap) without being one stand. The greedy
    search lets each plant absorb up to plant_radius_m of a transit, so a single step from one stand to the next
    (larissa_kadu_38: a foot steps 0.3-0.45 m, then the chest speeds up to 0.6-0.8 m/s right at the junction and
    covers 0.3-0.45 m in a beat) is split between the two plants with a 1-3 frame gap left over. The step = the
    frames around the origin's speed peak within one beat of the junction where the speed stays >= half the peak,
    narrowed (the plants grow back into it) until both plants still pass the full plant test and keep >= min_dur. When
    no such window remains (the plants only pass WITH the step's halves, e.g. a stand entered from one side and left
    back towards it) they keep their spans and the step sits at their junction: segment still records it as a
    'step' from cross to cross."""
    if len(spans) < 2:
        return spans
    spd = np.linalg.norm(np.gradient(o, t, axis=0), axis=1)
    idx = np.arange(len(t))
    out = list(spans)
    for k in range(len(out) - 1):
        (a0, a1), (b0, b1) = out[k], out[k + 1]
        if not degenerate_gap(t, o, a1, b0, cfg):
            continue
        tj = 0.5 * (t[a1] + t[b0])
        w = np.flatnonzero((t >= tj - beat) & (t <= tj + beat) & (idx > a0) & (idx < b1))
        if len(w) == 0:
            continue
        pk = int(w[np.argmax(spd[w])])
        thr = 0.5 * spd[pk]
        s0 = s1 = pk
        while s0 - 1 > a0 and spd[s0 - 1] >= thr:
            s0 -= 1
        while s1 + 1 < b1 and spd[s1 + 1] >= thr:
            s1 += 1
        s0 = max(s0, int(np.searchsorted(t, t[a0] + min_dur - 1e-9)))                 # the first plant keeps min_dur
        s1 = min(s1, int(np.searchsorted(t, t[b1] - min_dur + 1e-9, "right")) - 1)     # the second too
        while s0 < pk and not plant_ok(t, o, a0, s0, cfg):
            s0 += 1
        while s1 > pk and not plant_ok(t, o, s1, b1, cfg):
            s1 -= 1
        if s1 - s0 < 2 or not plant_ok(t, o, a0, s0, cfg) or not plant_ok(t, o, s1, b1, cfg) \
                or degenerate_gap(t, o, s0, s1, cfg):
            continue
        out[k], out[k + 1] = (a0, s0), (s1, b1)
    return out


def zigzag(y: np.ndarray, thr: float) -> list[int]:
    """Turning points of y with hysteresis thr: alternating extrema, every swing between two of them >= thr."""
    n = len(y)
    if n < 2:
        return []
    hi = lo = 0
    trend, ext, piv = 0, 0, []
    for i in range(1, n):
        if trend == 0:
            if y[i] > y[hi]:
                hi = i
            if y[i] < y[lo]:
                lo = i
            if y[hi] - y[lo] >= thr:
                trend, ext = (1, hi) if hi > lo else (-1, lo)
                piv.append(lo if hi > lo else hi)
        elif trend == 1:
            if y[i] > y[ext]:
                ext = i
            elif y[ext] - y[i] >= thr:
                piv.append(ext)
                trend, ext = -1, i
        else:
            if y[i] < y[ext]:
                ext = i
            elif y[i] - y[ext] >= thr:
                piv.append(ext)
                trend, ext = 1, i
    if trend and abs(y[ext] - y[piv[-1]]) >= thr:
        piv.append(ext)
    return piv


def swing_edges(y: np.ndarray, a: int, b: int, tol: float) -> tuple[int, int]:
    """Trim a swing a -> b to where y leaves y[a] / reaches y[b] within tol (plateaus at either end excluded)."""
    mid_v = 0.5 * (y[a] + y[b])
    seg = y[a:b + 1]
    up = y[b] > y[a]
    cross = np.flatnonzero(seg >= mid_v if up else seg <= mid_v)
    mid = a + (int(cross[0]) if len(cross) else (b - a) // 2)
    pre = np.flatnonzero(np.abs(y[a:mid + 1] - y[a]) <= tol)
    post = np.flatnonzero(np.abs(y[mid:b + 1] - y[b]) <= tol)
    return a + int(pre[-1]) if len(pre) else a, mid + int(post[0]) if len(post) else b


def _longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    """(start, end inclusive) of the longest run of True in mask, or None."""
    best, start = None, None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - 1 - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    return best


def _slice(ax: dict, s: int, e: int) -> dict:
    return {k: (v[s:e + 1] if isinstance(v, np.ndarray) else v) for k, v in ax.items()}


def segment(ax: dict, beat: float, cfg: FCConfig, base: dict | None = None, before: dict | None = None,
            after: dict | None = None) -> dict:
    """Plants / moves / rotations of one contiguous piece of axis track (ax from smooth_axis) that no counterbalance
    pivot overlaps (analyse cuts the track at the pivots; `before` / `after` = the pivot the piece starts after / ends
    at, recorded on its lead-in / trailing move). Ids start at base[kind] (+ base['swing'] for the swing index)."""
    base = {"plant": 0, "move": 0, "rotation": 0, "swing": 0, **(base or {})}
    t, o, yaw = ax["t"], ax["origin"], ax["yaw"]
    T = len(t)
    dt = float(np.median(np.diff(t))) if T > 1 else 1 / 30
    ne = max(1, int(round(cfg.end_window_s / dt)))

    plants = []
    for n, (i0, i1) in enumerate(find_plants(t, o, beat, cfg) if T > 1 else []):
        P = o[i0:i1 + 1]
        c = P.mean(0)
        d = np.linalg.norm(P - c, axis=1)
        y = yaw[i0:i1 + 1]
        ys, ye = float(y[:ne].mean()), float(y[-ne:].mean())
        plants.append({
            "id": base["plant"] + n, "t0": round(float(t[i0]), 4), "t1": round(float(t[i1]), 4),
            "duration_s": round(float(t[i1] - t[i0]), 4), "duration_beats": round(float(t[i1] - t[i0]) / beat, 3),
            "origin": xyz(c), "radius_m": round(float(d.max()), 4), "radius_p90_m": round(float(np.percentile(d, 90)), 4),
            "drift_m_s": round(drift_speed(t[i0:i1 + 1], P), 4),
            "yaw_mean_deg": round(circ_mean_deg(y), 2),
            "yaw_start_deg": round(float(wrap180(ys)), 2), "yaw_end_deg": round(float(wrap180(ye)), 2),
            "turn_deg": round(ye - ys, 2), "_i": (i0, i1), "_c": c})

    def path(i0: int, i1: int, a: np.ndarray, b: np.ndarray) -> tuple[dict, float]:
        """The origin from i0 to i1 at path_step_s with the end points a (start) and b (end) added when they differ
        from it (the plant origins: the drawn path runs cross to cross); its polyline length."""
        step = max(1, int(round(cfg.path_step_s / dt)))
        idx = np.unique(np.concatenate([np.arange(i0, i1 + 1, step), [i1]]))
        tt, pp = [float(x) for x in t[idx]], [p for p in o[idx]]
        if np.linalg.norm(a - pp[0]) > 1e-3:
            tt.insert(0, float(t[i0]))
            pp.insert(0, a)
        if np.linalg.norm(b - pp[-1]) > 1e-3:
            tt.append(float(t[i1]))
            pp.append(b)
        P = np.asarray(pp, float)
        return ({"t": _r(np.asarray(tt), 4), "pos": xyz(P)},
                float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0)

    moves = []
    spans = []
    if plants:
        if plants[0]["_i"][0] > 0:
            spans.append((None, plants[0], 0, plants[0]["_i"][0]))
        for p, q in zip(plants[:-1], plants[1:]):
            spans.append((p, q, p["_i"][1], q["_i"][0]))
        if plants[-1]["_i"][1] < T - 1:
            spans.append((plants[-1], None, plants[-1]["_i"][1], T - 1))
    elif T > 1:
        spans.append((None, None, 0, T - 1))
    for p, q, i0, i1 in spans:
        dur = float(t[i1] - t[i0])
        if (p is None or q is None) and dur < cfg.move_min_s:
            continue
        a = p["_c"] if p else o[i0]
        b = q["_c"] if q else o[i1]
        own = float(np.linalg.norm(np.diff(o[i0:i1 + 1], axis=0), axis=1).sum()) if i1 > i0 else 0.0
        kind = "step" if p is not None and q is not None and own < cfg.step_max_m \
            and dur <= cfg.step_max_beats * beat + 1e-9 else "travel"
        pth, plen = path(i0, i1, a, b)
        disp = b - a
        y0 = float(yaw[min(T - 1, i0):min(T, i0 + ne)].mean()) if p is None else \
            float(yaw[max(0, p["_i"][1] - ne + 1):p["_i"][1] + 1].mean())
        y1 = float(yaw[max(0, i1 - ne + 1):i1 + 1].mean()) if q is None else \
            float(yaw[q["_i"][0]:q["_i"][0] + ne].mean())
        moves.append({
            "id": base["move"] + len(moves), "kind": kind, "t0": round(float(t[i0]), 4), "t1": round(float(t[i1]), 4),
            "duration_s": round(dur, 4), "from_plant": p["id"] if p else None, "to_plant": q["id"] if q else None,
            "from_pivot": before["id"] if p is None and before else None,
            "to_pivot": after["id"] if q is None and after else None,
            "from": xyz(a), "to": xyz(b), "path": pth, "length_m": round(plen, 4), "origin_path_m": round(own, 4),
            "displacement_m": round(float(np.linalg.norm(disp)), 4),
            "heading_deg": round(float(yaw_of(disp)), 2) if np.linalg.norm(disp) > 1e-6 else None,
            "from_yaw_deg": round(float(wrap180(y0)), 2), "to_yaw_deg": round(float(wrap180(y1)), 2),
            "yaw_change_deg": round(y1 - y0, 2), "_i": (i0, i1)})

    # rotation contexts: every travel move, and every GROUP of plants chained by single steps (a turn while stepping
    # from one stand to the next stays ONE rotation; it is owned by the plant or step holding most of its angle); a
    # travel move is a context of its own (the turn happens in another place)
    ctx: list[dict] = []
    for kind, rec in sorted([("plant", p) for p in plants] + [("move", m) for m in moves], key=lambda x: x[1]["_i"][0]):
        chain = kind == "move" and rec["kind"] == "step"
        if (chain or (kind == "plant" and ctx and ctx[-1].get("open"))) and ctx and ctx[-1]["kind"] == "plant":
            ctx[-1]["members"].append((kind, rec))
            ctx[-1]["_i"] = (ctx[-1]["_i"][0], rec["_i"][1])
            ctx[-1]["open"] = chain
        elif kind == "plant":
            ctx.append({"kind": "plant", "members": [(kind, rec)], "_i": rec["_i"], "open": False})
        else:
            ctx.append({"kind": "move", "rec": rec, "_i": rec["_i"]})

    def owner(cx: dict, p0: int, p1: int) -> tuple[str, dict, list[int]]:
        """The member (plant or step) of a plant group holding most of the yaw change of frames p0..p1, and the ids
        of the plants it touches."""
        best, best_v, touched = cx["members"][0], -1.0, []
        for kind, rec in cx["members"]:
            a0, a1 = max(p0, rec["_i"][0]), min(p1, rec["_i"][1])
            if a1 <= a0:
                continue
            if kind == "plant":
                touched.append(rec["id"])
            v = abs(float(yaw[a1] - yaw[a0]))
            if v > best_v:
                best, best_v = (kind, rec), v
        return best[0], best[1], touched

    tol_rot = cfg.rotation_edge_deg
    rotations = []
    tp = zigzag(yaw, cfg.rotation_deg)
    n_sw = 0
    for a, b in zip(tp[:-1], tp[1:]):
        sw = float(yaw[b] - yaw[a])
        if abs(sw) < cfg.rotation_deg:
            continue
        e0, e1 = swing_edges(yaw, a, b, min(tol_rot, 0.1 * abs(sw)))
        # split the swing where it crosses a context boundary (plant group | travel move), so every rotation belongs
        # to ONE plant (group) or ONE move: pieces of >= rotation_deg are kept and a smaller piece joins its kept
        # neighbour (the earlier one when it has two); when no piece reaches it, the whole swing is one rotation in the
        # context that holds most of its angle (a turn straddling a plant edge stays whole). Times are the trimmed
        # edges; the angle of an outer end is measured from the swing's turning point itself (the trim must not
        # shave the angle).
        pieces = []
        for cx in ctx:
            s0, s1 = cx["_i"]
            p0, p1 = max(e0, s0), min(e1, s1)
            if p1 > p0:
                pieces.append([p0, p1, cx, float(yaw[p1] - yaw[p0])])
        if not pieces:
            continue
        ang_of = lambda p0, p1: float(yaw[b if p1 == e1 else p1] - yaw[a if p0 == e0 else p0])   # noqa: E731
        keep = [abs(pc[3]) >= cfg.rotation_deg for pc in pieces]
        merged = []
        if any(keep):
            pending = None                  # small pieces before the first kept one
            for pc, k in zip(pieces, keep):
                if k:
                    merged.append([pc[0] if pending is None else pending, pc[1], pc[2]])
                    pending = None
                elif merged:
                    merged[-1][1] = pc[1]
                elif pending is None:
                    pending = pc[0]
            merged = [m for m in merged if abs(ang_of(m[0], m[1])) >= cfg.rotation_deg]
        if not merged:
            big = max(pieces, key=lambda pc: abs(pc[3]))
            merged = [[e0, e1, big[2]]]
        for p0, p1, cx in merged:
            t0, t1 = float(t[p0]), float(t[p1])
            ang = ang_of(p0, p1)
            y0 = float(yaw[a if p0 == e0 else p0])
            kind, rec, plant_ids = owner(cx, p0, p1) if cx["kind"] == "plant" else ("move", cx["rec"], [])
            if kind == "plant":
                centre, plant_id, move_id = rec["_c"], rec["id"], None
            else:
                centre, plant_id, move_id = o[(p0 + p1) // 2], None, rec["id"]
            rotations.append({
                "id": base["rotation"] + len(rotations), "t0": round(t0, 4), "t1": round(t1, 4),
                "duration_s": round(t1 - t0, 4), "from_yaw_deg": round(float(wrap180(y0)), 2),
                "to_yaw_deg": round(float(wrap180(y0 + ang)), 2), "signed_deg": round(ang, 2),
                "direction": "ccw" if ang > 0 else "cw", "rate_deg_s": round(abs(ang) / max(t1 - t0, 1e-9), 1),
                "context": kind, "plant": plant_id, "plants": plant_ids, "move": move_id,
                "centre": xyz(centre),
                "swing": {"index": base["swing"] + n_sw, "deg": round(sw, 2),
                          "t": [round(float(t[e0]), 4), round(float(t[e1]), 4)], "parts": len(merged)},
                "_i": (p0, p1)})
        n_sw += 1

    # the axis a plant ESTABLISHED: the circular-mean yaw over its longest stretch without a rotation in progress (the
    # mean over a plant that contains a 130 deg turn is neither heading); the plain mean when rotations fill it
    for p in plants:
        i0, i1 = p["_i"]
        free = np.ones(i1 - i0 + 1, bool)
        for r in rotations:
            r0, r1 = r["_i"]
            lo, hi = max(r0 + 1, i0), min(r1 - 1, i1)
            if hi >= lo:
                free[lo - i0:hi - i0 + 1] = False
        run = _longest_run(free)
        if run is not None and run[1] - run[0] + 1 >= max(3, ne) and not free.all():
            ym, src = circ_mean_deg(yaw[i0 + run[0]:i0 + run[1] + 1]), "held"
            p["held"] = [round(float(t[i0 + run[0]]), 4), round(float(t[i0 + run[1]]), 4)]
        else:
            ym, src = p["yaw_mean_deg"], "mean" if free.all() else "mean (turning throughout)"
            p["held"] = [p["t0"], p["t1"]]
        f = fwd_of(ym)
        p.update({"yaw_deg": round(float(wrap180(ym)), 2), "yaw_source": src, "forward": xyz(f),
                  "long_axis": xyz(long_of_fwd(f))})
    for rec in plants + moves + rotations:
        rec.pop("_i", None)
        rec.pop("_c", None)
    return {"plants": plants, "moves": moves, "rotations": rotations, "swings": n_sw}


def leader_axis_in(ax: dict, k0: int, k1: int, cfg: FCConfig) -> dict:
    """The leader's axis over track frames k0..k1 (a counterbalance pivot, which owns this motion): his chest path on
    the floor, its length and the net (unwrapped) turn of his axis."""
    t, o, yaw = ax["t"], ax["origin"], ax["yaw"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    step = max(1, int(round(cfg.path_step_s / dt)))
    idx = np.unique(np.concatenate([np.arange(k0, k1 + 1, step), [k1]]))
    turn = float(yaw[k1] - yaw[k0])
    return {"t0": round(float(t[k0]), 4), "t1": round(float(t[k1]), 4),
            "path": {"t": _r(t[idx], 4), "pos": xyz(o[idx])},
            "length_m": round(float(np.linalg.norm(np.diff(o[k0:k1 + 1], axis=0), axis=1).sum()), 4),
            "from_yaw_deg": round(float(wrap180(yaw[k0])), 2), "to_yaw_deg": round(float(wrap180(yaw[k1])), 2),
            "turn_deg": round(turn, 2), "direction": "ccw" if turn > 0 else "cw"}


# ----------------------------------------------------------------------------------------------- T axis (version 3)

def t_frame(Y) -> tuple[np.ndarray, np.ndarray]:
    """(forward, parallel) unit [x, z] vectors of a T with heading Y (deg); parallel = the leader's LEFT."""
    F = fwd_of(Y)
    return F, long_of_fwd(F)


def t_offsets(p, O, Y) -> tuple[np.ndarray, np.ndarray]:
    """(along forward, along parallel) offsets (m) of floor points p (..., 2) [x, z] from a T at O with heading Y."""
    F, P = t_frame(Y)
    d = np.asarray(p, float) - np.asarray(O, float)
    return d @ F, d @ P


def t_inside(p, O, Y, cfg: FCConfig) -> np.ndarray:
    """True where floor points p lie in the T's ALLOWED REGION: the stem corridor (|parallel offset| <= t_half_width_m,
    |forward offset| <= t_forward_m: walking back and forth) or the crossbar corridor (|forward offset| <=
    t_half_width_m, |parallel offset| <= t_side_m: side to side)."""
    a, b = t_offsets(p, O, Y)
    w = cfg.t_half_width_m
    return ((np.abs(b) <= w) & (np.abs(a) <= cfg.t_forward_m)) | ((np.abs(a) <= w) & (np.abs(b) <= cfg.t_side_m))


def t_episodes(ax: dict, beat: float, cfg: FCConfig, blocked: np.ndarray | None = None,
               brief: list | None = None) -> list[dict]:
    """The T episodes of one contiguous axis track (ax from smooth_axis; blocked = frames a counterbalance pivot owns).

    SEEK: the earliest window of t_form_beats (>= 3 frames) with no blocked frame that is SETTLED - with O = the
    median origin and Y = the mean (unwrapped) heading of the window, every frame of it lies in the allowed region of
    (O, Y) and within t_form_yaw_deg of Y, and its net drift (least-squares line, as the plants) is <=
    t_form_drift_m_s along the stem OR along the crossbar: drifting along one T axis is dancing on the T, drifting
    across both (a diagonal arrival) means he is still arriving. Frames before it are a transition (travelling /
    turning, no T).
    The window FORMS the T (its grow-in, t0..t_formed). A window whose O is within t_same_place_m of the previous T
    (nothing blocked in between) and whose Y is within t_pivot_deg of it CONTINUES that T (he stepped out and came
    back: an excursion); within t_same_place_m but turned by more than t_pivot_deg it is a ROTATED T AT THE SAME PLACE
    (O copied from the previous T: same).
    ACTIVE: from the end of the forming window the T holds while he is in its region with his heading within
    t_pivot_deg of Y. It ENDS at the first frame of a run of >= t_sustain_beats outside the region ('exit') or turned
    by more than t_pivot_deg ('rotation'; both: 'exit+rotation') - a single step out and back, or a contra-body swing,
    does not end it - or at a blocked frame / the track end.
    AVERAGED (review 2026-10-07: the forming window alone left the long Ts 0.28-0.33 m from where he danced on them):
    O / Y are then refined to the median origin / mean heading over ALL the T's active frames inside its region and
    within t_pivot_deg (a rotated-in-place T keeps the copied origin, heading only), and its end is re-checked
    against the refined T (up to t_refine_iter times), so the stored T is the averaged position he occupied and its
    boundaries are valid for it.
    MINIMUM HOLD: a T that holds for less than t_min_hold_beats after forming (he started turning / leaving right
    after it formed) is DROPPED - on screen it would grow in and hand over at once; its frames join the transition
    to the next T (the track's last T, cut by the end of the take, is kept). brief (optional list) collects the
    dropped episodes: headings he did hold for a moment, the WAYPOINTS of the axis pivots between the kept Ts (a turn
    out and back while no T holds is still two reorientations).
    -> [{i0, f (last forming frame), i1 (exclusive end), O [x, z], Y (unwrapped deg), end, same (index of the
    previous episode it is a rotated copy of, or None), excursions, dropped (short Ts dropped since the previous
    episode), O_form / Y_form (the forming window's values)}] (frame indices of the track)."""
    t, o, y = ax["t"], ax["origin"], ax["yaw"]
    T = len(t)
    if T < 3:
        return []
    blocked = np.zeros(T, bool) if blocked is None else np.asarray(blocked, bool)
    bcum = np.concatenate([[0], np.cumsum(blocked)])
    dt = float(np.median(np.diff(t)))
    nF = max(3, int(round(cfg.t_form_beats * beat / dt)))
    nS = max(1, int(round(cfg.t_sustain_beats * beat / dt)))
    min_hold = cfg.t_min_hold_beats * beat

    def settled(j: int):
        if j + nF > T or bcum[j + nF] - bcum[j] > 0:
            return None
        O = np.median(o[j:j + nF], 0)
        Y = float(y[j:j + nF].mean())
        if np.abs(y[j:j + nF] - Y).max() > cfg.t_form_yaw_deg or not t_inside(o[j:j + nF], O, Y, cfg).all():
            return None
        v = drift_vec(t[j:j + nF], o[j:j + nF])
        F, P = t_frame(Y)
        if min(abs(float(v @ F)), abs(float(v @ P))) > cfg.t_form_drift_m_s:
            return None
        return O, Y

    def hold(k: int, O, Y) -> tuple[int, str]:
        """(exclusive end, reason): the first frame of a sustained exit / rotation run from frame k on."""
        out, rot, end = 0, 0, "end"
        while k < T:
            if blocked[k]:
                return k, "blocked"
            out = 0 if bool(t_inside(o[k], O, Y, cfg)) else out + 1
            rot = rot + 1 if abs(float(y[k]) - Y) > cfg.t_pivot_deg else 0
            if out >= nS or rot >= nS:
                runs = [(n, why) for n, why in ((out, "exit"), (rot, "rotation")) if n >= nS]
                return k - max(n for n, _ in runs) + 1, "+".join(why for _, why in runs)
            k += 1
        return T, end

    def averaged(i0: int, i1: int, O, Y, keep_origin: bool):
        """(O, Y) over the active frames i0..i1-1 that lie in the region of (O, Y) with the heading within
        t_pivot_deg (median origin, mean heading); unchanged with fewer than 3 such frames."""
        k = np.arange(i0, max(i0 + 1, i1))
        sel = k[t_inside(o[k], O, Y, cfg) & (np.abs(y[k] - Y) <= cfg.t_pivot_deg)]
        if len(sel) < 3:
            return O, Y
        return (O if keep_origin else np.median(o[sel], 0)), float(y[sel].mean())

    eps: list[dict] = []
    dropped = 0
    i = 0
    while i < T:
        if blocked[i]:
            i += 1
            continue
        j, st = i, None
        while j < T and not blocked[j]:
            st = settled(j)
            if st is not None:
                break
            j += 1
        if st is None:
            i = j
            continue
        O, Y = st
        rec = {"i0": j, "f": j + nF - 1, "O": O, "Y": Y, "same": None, "excursions": 0, "O_form": O.copy(),
               "Y_form": Y, "dropped": dropped}
        prev = eps[-1] if eps else None
        if prev is not None and bcum[j] - bcum[prev["i1"]] == 0:        # no counterbalance pivot in between
            d = float(np.linalg.norm(O - prev["O"]))
            if d <= cfg.t_same_place_m and abs(Y - prev["Y"]) <= cfg.t_pivot_deg:
                eps.pop()                       # back on the same T: it continues (the frames between are its own)
                rec = {**prev, "excursions": prev["excursions"] + 1, "dropped": prev["dropped"] + dropped}
            elif d <= cfg.t_same_place_m:
                rec["O"], rec["same"] = prev["O"].copy(), len(eps) - 1
        O, Y = rec["O"], rec["Y"]
        i1, end = hold(j + nF, O, Y)
        for _ in range(cfg.t_refine_iter):
            O2, Y2 = averaged(rec["i0"], i1, O, Y, rec["same"] is not None)
            if float(np.linalg.norm(O2 - O)) < 1e-3 and abs(Y2 - Y) < 0.05:
                break
            O, Y = O2, Y2
            i1, end = hold(j + nF, O, Y)
        rec.update({"O": O, "Y": Y, "i1": i1, "end": end})
        if end != "end" and float(t[min(i1, T - 1)] - t[rec["f"]]) < min_hold - 1e-9:
            dropped += 1                        # it would only flash: its frames join the transition
            if rec["excursions"] and prev is not None and rec["i0"] == prev["i0"]:
                eps.append(prev)                # (a continued T keeps its earlier span)
            elif brief is not None:
                brief.append(rec)
            i = max(i1, j + 1)
            continue
        dropped = 0
        eps.append(rec)
        i = rec["i1"]
    return eps


def _first(mask: np.ndarray) -> int | None:
    k = np.flatnonzero(mask)
    return int(k[0]) if len(k) else None


def _last(mask: np.ndarray) -> int | None:
    k = np.flatnonzero(mask)
    return int(k[-1]) if len(k) else None


REF_KIND = {"T": "t_axis", "brief": "brief_t", "start": "track_start", "end": "track_end", "cb": "counterbalance"}


def _pivot_extent(y: np.ndarray, ya: float, yb: float, r0: int, r1: int, cfg: FCConfig) -> tuple[int, int, int]:
    """(p0, p1, pm) track frames of a heading change ya -> yb (unwrapped deg) searched within r0..r1: p0 = the last
    frame still within tol of ya, p1 = the first frame after it within tol of yb (tol = min(t_pivot_edge_deg, 25 % of
    the change)), pm = the frame half-way through the change."""
    dY = yb - ya
    tol = min(cfg.t_pivot_edge_deg, 0.25 * abs(dY))
    r = np.arange(r0, min(max(r1, r0 + 1), len(y) - 1) + 1)
    k = _last(np.abs(y[r] - ya) <= tol)
    p0 = int(r[k]) if k is not None else int(r[0])
    rr = np.arange(p0, int(r[-1]) + 1)
    k = _first(np.abs(y[rr] - yb) <= tol)
    p1 = int(rr[k]) if k is not None else int(r[-1])
    p1 = min(max(p1, p0 + 1), len(y) - 1)
    frac = (y[p0:p1 + 1] - ya) / dY if abs(dY) > 1e-9 else np.ones(p1 - p0 + 1)
    return p0, p1, p0 + (_first(frac >= 0.5) or 0)


def t_axis_records(segs: list[dict], cfg: FCConfig, cb_pivots: list[dict]) -> tuple[list, list, list]:
    """(t_axes, transitions, axis_pivots) from the per-segment episodes. segs = [{"ax": axis track, "eps": t_episodes,
    "off": global frame index of the track's first frame}] in time order. Global frame indices are kept in "_g"
    (popped by strip_private) for the balance evaluation.
    AXIS PIVOTS (every reorientation of his axis by more than t_pivot_deg that no dial owns):
      context 'between_ts'            the heading change from one T to the next (from_t -> to_t);
      context 'lead_in'               before the first T: his heading at the start of the track -> the first T's
                                      (from_t null);
      context 'into_counterbalance'   a T -> his heading where a counterbalance pivot begins (to_t null): the turning
                                      on the way into the counterbalance, which its dial does not cover;
      context 'out_of_counterbalance' his heading where the pivot ends -> the next T (from_t null);
      context 'lead_out'              the last T -> his heading at the end of the track when that T ended earlier.
    (review 2026-10-07: ~700 deg of turning around the counterbalance and before the first T was recorded nowhere.)"""
    t_axes = []
    for s, sg in enumerate(segs):
        ax, eps = sg["ax"], sg["eps"]
        t, o, y = ax["t"], ax["origin"], ax["yaw"]
        T = len(t)
        for n, e in enumerate(eps):
            i0, i1 = e["i0"], e["i1"]
            act = np.arange(i0, max(i0 + 1, i1))
            a, b = t_offsets(o[act], e["O"], e["Y"])
            F, P = t_frame(e["Y"])
            if n == 0:
                start = "start" if s == 0 else "gap"
                if s == 0 and i0 > 0 and np.any([pv["t1"] <= t[i0] and pv["t0"] >= t[0] for pv in cb_pivots]):
                    start = "counterbalance"
            else:
                start = eps[n - 1]["end"] if eps[n - 1]["end"] != "blocked" else "counterbalance"
            end = {"blocked": "counterbalance", "end": "end" if s == len(segs) - 1 else "gap"}.get(e["end"], e["end"])
            rec = {
                "id": len(t_axes), "t0": round(float(t[i0]), 4), "t1": round(float(t[min(i1, T - 1)]), 4),
                "t_formed": round(float(t[e["f"]]), 4), "origin": xyz(e["O"]),
                "yaw_deg": round(float(wrap180(e["Y"])), 2), "yaw_unwrapped_deg": round(float(e["Y"]), 2),
                "forward": xyz(F), "parallel": xyz(P), "start_reason": start, "end_reason": end,
                "same_place_as": None, "prev": None, "next": None, "pivot_in": None, "pivot_out": None, "dial": None,
                "excursions": e["excursions"], "dropped_before": int(e.get("dropped", 0)),
                "formed_origin": xyz(e.get("O_form", e["O"])),
                "formed_yaw_deg": round(float(wrap180(e.get("Y_form", e["Y"]))), 2),
                "occupancy": {"forward_m": _r([a.min(), a.max()], 3), "side_m": _r([b.min(), b.max()], 3),
                              "in_region_share": round(float(t_inside(o[act], e["O"], e["Y"], cfg).mean()), 3),
                              "yaw_dev_p90_deg": round(float(np.percentile(np.abs(y[act] - e["Y"]), 90)), 1),
                              "centre_offset_m": round(float(np.linalg.norm(np.median(o[act], 0) - e["O"])), 3)},
                "_g": (sg["off"] + i0, sg["off"] + e["f"], sg["off"] + i1), "_seg": s, "_ep": e}
            if e["same"] is not None:
                rec["same_place_as"] = len(t_axes) - (n - e["same"])
            t_axes.append(rec)
    for k in range(len(t_axes) - 1):
        t_axes[k]["next"], t_axes[k + 1]["prev"] = k + 1, k

    axis_pivots, transitions = [], []

    def add_pivot(sg_i: int, cur: tuple, nxt: tuple, context: str, cbp: int | None) -> int | None:
        """an axis pivot cur -> nxt (anchors (Y unwrapped deg, track frame, ref, T record | None) on segment sg_i's
        track; the change is searched within their frames), if it exceeds t_pivot_deg"""
        ya, r0, ref_a, A = cur
        yb, r1, ref_b, B = nxt
        dY = float(yb - ya)
        if abs(dY) <= cfg.t_pivot_deg:
            return None
        ax, off = segs[sg_i]["ax"], segs[sg_i]["off"]
        t, o, y = ax["t"], ax["origin"], ax["yaw"]
        p0, p1, pm = _pivot_extent(y, ya, yb, r0, r1, cfg)
        stayed = bool(A is not None and B is not None and B["same_place_as"] == A["id"])
        pid = len(axis_pivots)
        axis_pivots.append({
            "id": pid, "t0": round(float(t[p0]), 4), "t1": round(float(t[p1]), 4),
            "at": xyz(np.asarray(A["origin"], float)[[0, 2]] if stayed else o[pm]),
            "from_t": None if A is None else A["id"], "to_t": None if B is None else B["id"],
            "from_yaw_deg": round(float(wrap180(ya)), 2), "to_yaw_deg": round(float(wrap180(yb)), 2),
            "signed_deg": round(dY, 2), "abs_deg": round(abs(dY), 2), "direction": "ccw" if dY > 0 else "cw",
            "stayed": stayed, "context": context, "via_brief": ref_a[0] == "brief" or ref_b[0] == "brief",
            "from_kind": REF_KIND[ref_a[0]], "to_kind": REF_KIND[ref_b[0]],
            "counterbalance_pivot": cbp, "dial": None, "_g": (off + p0, off + p1), "_from": ref_a, "_to": ref_b})
        if A is not None and A["pivot_out"] is None:
            A["pivot_out"] = pid
        if B is not None:
            B["pivot_in"] = pid
        return pid

    def chain(sg_i: int, start: tuple, end: tuple, context: str, cbp: int | None = None) -> list[int]:
        """axis pivots from anchor start to anchor end through the WAYPOINTS between them (the brief Ts dropped by
        t_episodes: headings he held for a moment); a waypoint within t_pivot_deg of the current heading is passed
        over"""
        way = sorted((e for e in segs[sg_i].get("brief", []) if start[1] < e["f"] < end[1]), key=lambda e: e["f"])
        seq = [start] + [(e["Y"], e["f"], ("brief", sg_i, e["i0"]), None) for e in way] + [end]
        cur, ids = seq[0], []
        for n, nxt in enumerate(seq[1:]):
            if abs(nxt[0] - cur[0]) <= cfg.t_pivot_deg and n < len(seq) - 2:
                continue
            pid = add_pivot(sg_i, cur, nxt, context, cbp)
            if pid is not None:
                ids.append(pid)
            cur = nxt
        return ids

    def frame_of(ax: dict, tq: float, side: str) -> int:
        tt = ax["t"]
        if side == "before":                                 # last frame <= tq
            return int(np.clip(np.searchsorted(tt, tq + 1e-6, "right") - 1, 0, len(tt) - 1))
        return int(np.clip(np.searchsorted(tt, tq - 1e-6), 0, len(tt) - 1))     # first frame >= tq

    def anchor(ta: dict) -> tuple:
        return ta["_ep"]["Y"], ta["_ep"]["f"], ("T", ta["id"]), ta

    # lead-in: his turning before the first T of the take (and of a track after a gap)
    for s, sg in enumerate(segs):
        firsts = [ta for ta in t_axes if ta["_seg"] == s]
        if not firsts or firsts[0]["start_reason"] == "counterbalance":
            continue
        chain(s, (float(sg["ax"]["yaw"][0]), 0, ("start", s), None), anchor(firsts[0]), "lead_in")
    for k in range(len(t_axes)):
        A = t_axes[k]
        B = t_axes[k + 1] if k + 1 < len(t_axes) else None
        if B is None or B["_seg"] != A["_seg"]:
            if A["end_reason"] not in ("end", "gap", "counterbalance"):      # the track goes on without a T
                ax = segs[A["_seg"]]["ax"]
                n_ = len(ax["t"]) - 1
                chain(A["_seg"], anchor(A), (float(ax["yaw"][-1]), n_, ("end", A["_seg"]), None), "lead_out")
            if B is None:
                continue
        cbs = [pv for pv in cb_pivots if pv["t0"] < B["t0"] + 1e-6 and pv["t1"] > A["t1"] - 1e-6]
        cbp = cbs[0]["id"] if cbs else None
        ids: list[int] = []
        if A["_seg"] == B["_seg"]:
            ax = segs[A["_seg"]]["ax"]
            y = ax["yaw"]
            if cbp is None:
                ids = chain(A["_seg"], anchor(A), anchor(B), "between_ts")
            else:
                # around the counterbalance: into it (A -> his heading where the pivot begins) and out of it (his
                # heading where it ends -> B); the counterbalance dial owns the circling in between
                k0 = frame_of(ax, cbs[0]["t0"], "before")
                k1 = frame_of(ax, cbs[-1]["t1"], "after")
                if k0 > A["_ep"]["f"]:
                    ids += chain(A["_seg"], anchor(A), (float(y[k0]), k0, ("cb", cbs[0]["id"]), None),
                                 "into_counterbalance", cbs[0]["id"])
                if B["_ep"]["f"] > k1:
                    ids += chain(A["_seg"], (float(y[k1]), k1, ("cb", cbs[-1]["id"]), None), anchor(B),
                                 "out_of_counterbalance", cbs[-1]["id"])
        pa, pb = np.asarray(A["origin"], float)[[0, 2]], np.asarray(B["origin"], float)[[0, 2]]
        dist = float(np.linalg.norm(pb - pa))
        path = None
        if A["_seg"] == B["_seg"]:
            ax = segs[A["_seg"]]["ax"]
            i0, i1 = A["_ep"]["i1"], B["_ep"]["i0"]
            i0 = min(i0, len(ax["t"]) - 1)
            if i1 > i0:
                step = max(1, int(round(cfg.path_step_s / float(np.median(np.diff(ax["t"]))))))
                idx = np.unique(np.concatenate([np.arange(i0, i1 + 1, step), [i1]]))
                path = {"t": _r(ax["t"][idx], 4), "pos": xyz(ax["origin"][idx])}
        transitions.append({
            "id": len(transitions), "from_t": A["id"], "to_t": B["id"], "t0": A["t1"], "t1": B["t0"],
            "from": A["origin"], "to": B["origin"], "distance_m": round(dist, 4),
            "same_place": bool(B["same_place_as"] == A["id"]), "reason": B["start_reason"],
            "axis_pivot": ids[0] if ids and cbp is None else None, "axis_pivots": ids, "dial": None,
            "counterbalance_pivot": cbp,
            "dropped_ts": B["dropped_before"], "path": path})
    # ids in time order (the lead-in pivots were made first)
    axis_pivots.sort(key=lambda r: r["t0"])
    remap = {r["id"]: n for n, r in enumerate(axis_pivots)}
    for n, r in enumerate(axis_pivots):
        r["id"] = n
    for ta in t_axes:
        for key in ("pivot_in", "pivot_out"):
            if ta[key] is not None:
                ta[key] = remap[ta[key]]
    for tr in transitions:
        if tr["axis_pivot"] is not None:
            tr["axis_pivot"] = remap[tr["axis_pivot"]]
        tr["axis_pivots"] = sorted(remap[i] for i in tr["axis_pivots"])
    return t_axes, transitions, axis_pivots


# ----------------------------------------------------------------------------------------------- balance axis

def dancer_com(npz: dict, mass: float | None = None) -> tuple[np.ndarray, float | None, str]:
    """(COM (T, 3) world, mass kg or None, source): the whole-body COM from the shaped SMPL-X mass elements
    (counterbalance.body_com = physics_body, the physics stage's COM), else the de Leva-style joint COM."""
    try:
        com, m, src, _ = cb.body_com(npz, mass, cb.CBConfig())
        return np.asarray(com, float), float(m), f"SMPL-X mass elements ({src})"
    except Exception as e:  # noqa: BLE001 - no body model / parameters: the joint estimate
        J = np.asarray(npz["joints"], float)[:, :55]
        return cb.joints_com(J), mass, ("joints (de Leva-style segment weights; body model unavailable: "
                                        f"{type(e).__name__})")


def _smooth_on_segments(t: np.ndarray, x: np.ndarray, cutoff_hz: float, gap_frames: float) -> np.ndarray:
    """Zero-phase spline smoothing of x (T, ...) per contiguous piece of the time grid (short pieces kept raw)."""
    from .physics_spline import Smoother
    x = np.asarray(x, float)
    out = x.copy()
    for a, b in cb.contiguous_segments(t, gap_frames):
        if b - a + 1 >= 7:
            shp = x[a:b + 1].shape
            out[a:b + 1] = Smoother(t[a:b + 1], cutoff_hz)(x[a:b + 1].reshape(b - a + 1, -1)).reshape(shp)
    return out


def sole_nearest(p: np.ndarray, ankle: np.ndarray, ball: np.ndarray, s_range: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(distance (T,), nearest point (T, 2)) of floor points p (T, 2) to the sole line ankle + s (ball - ankle),
    s in s_range (heel .. toe tip), all [x, z]."""
    d = ball - ankle
    L2 = np.maximum((d ** 2).sum(-1, keepdims=True), 1e-9)
    s = np.clip(((p - ankle) * d).sum(-1, keepdims=True) / L2, s_range[0], s_range[1])
    q = ankle + s * d
    return np.linalg.norm(p - q, axis=-1), q


def balance_axis(t: np.ndarray, J: np.ndarray, com: np.ndarray, cfg: FCConfig) -> dict:
    """Per-frame balance axis of one dancer (world, metres; J (T, >=22, 3) joints, com (T, 3), both smoothed).
    Contact weight per foot = (lower of ankle / ball height above the dancer's floor estimate, bal_contact_h_m ramp) x
    (slower of ankle / ball horizontal speed, bal_contact_v_m_s ramp); a foot is IN CONTACT at weight >= 0.5. The
    STANCE foot = the foot in contact whose sole line (ankle + s (ball - ankle), s in bal_sole_s = heel .. toe tip) is
    nearest the COM's floor point - the weight-bearing foot; SUPPORT = that nearest sole point (the COM over any part
    of the sole is over the foot). com_offset_m = horizontal COM -> support distance; tilt_deg = angle of the line
    chest (spine3) -> support from vertical; score = min(offset ramp bal_good..bal_bad_offset_m, tilt ramp
    bal_good..bal_bad_tilt_deg) in 0..1 (0 without a stance foot)."""
    T = len(t)
    fl_a = np.percentile(np.minimum(J[:, 7, 1], J[:, 8, 1]), 5.0)
    fl_b = np.percentile(np.minimum(J[:, 10, 1], J[:, 11, 1]), 5.0)
    p = com[:, [0, 2]]
    dist, near, w, ball = {}, {}, {}, {}
    for f, (ia, ib) in cb.FEET.items():
        A, B = J[:, ia][:, [0, 2]], J[:, ib][:, [0, 2]]
        ball[f] = B
        h = np.minimum(J[:, ia, 1] - fl_a, J[:, ib, 1] - fl_b)
        if T > 1:
            v = np.minimum(np.linalg.norm(np.gradient(A, t, axis=0), axis=1),
                           np.linalg.norm(np.gradient(B, t, axis=0), axis=1))
        else:
            v = np.zeros(T)
        w[f] = cb._down(h, *cfg.bal_contact_h_m) * cb._down(v, *cfg.bal_contact_v_m_s)
        dist[f], near[f] = sole_nearest(p, A, B, cfg.bal_sole_s)
    dl = np.where(w["left"] >= 0.5, dist["left"], np.inf)
    dr = np.where(w["right"] >= 0.5, dist["right"], np.inf)
    stance = np.where(np.isfinite(dl) | np.isfinite(dr), np.where(dl <= dr, 1, 2), 0)
    support = np.where((stance == 1)[:, None], near["left"], near["right"])
    support[stance == 0] = np.nan
    off = np.where(stance > 0, np.minimum(dl, dr), np.nan)
    chest = J[:, CHEST]
    tilt = np.degrees(np.arctan2(np.linalg.norm(chest[:, [0, 2]] - support, axis=1), np.maximum(chest[:, 1], 0.1)))
    score = np.minimum(cb._down(off, cfg.bal_good_offset_m, cfg.bal_bad_offset_m),
                       cb._down(tilt, cfg.bal_good_tilt_deg, cfg.bal_bad_tilt_deg))
    score = np.where(stance > 0, np.nan_to_num(score), 0.0)
    return {"stance": stance.astype(int), "double": ((w["left"] >= 0.5) & (w["right"] >= 0.5)).astype(int),
            "support": support, "chest": chest, "com": com, "com_offset_m": off, "tilt_deg": np.where(stance > 0, tilt,
                                                                                                         np.nan),
            "score": score, "contact": w, "ball": ball}


def couple_axis(t: np.ndarray, Jf: np.ndarray, com_c: np.ndarray, cb_pivots: list[dict], cfg: FCConfig) -> dict:
    """The couple's balance axis inside every counterbalance pivot: the combined COM (com_c (T, 3)) over the
    follower's ANCHORED foot (pivots[].pivot_foot; its sole line, nearest point = support); offset / tilt (of the line
    support -> combined COM) / score as balance_axis; NaN (pivot -1) outside the pivots."""
    T = len(t)
    out = {"com": np.full((T, 3), np.nan), "support": np.full((T, 2), np.nan), "com_offset_m": np.full(T, np.nan),
           "tilt_deg": np.full(T, np.nan), "score": np.full(T, np.nan), "pivot": np.full(T, -1)}
    for pv in cb_pivots:
        k = (t >= pv["t0"] - 1e-6) & (t <= pv["t1"] + 1e-6)
        if not k.any():
            continue
        ia, ib = cb.FEET.get(pv.get("pivot_foot") or "left", cb.FEET["left"])
        d, q = sole_nearest(com_c[k][:, [0, 2]], Jf[k][:, ia][:, [0, 2]], Jf[k][:, ib][:, [0, 2]], cfg.bal_sole_s)
        tilt = np.degrees(np.arctan2(d, np.maximum(com_c[k][:, 1], 0.1)))
        out["com"][k], out["support"][k], out["com_offset_m"][k], out["tilt_deg"][k] = com_c[k], q, d, tilt
        out["score"][k] = np.minimum(cb._down(d, cfg.bal_good_offset_m, cfg.bal_bad_offset_m),
                                     cb._down(tilt, cfg.bal_good_tilt_deg, cfg.bal_bad_tilt_deg))
        out["pivot"][k] = pv["id"]
    return out


def evaluate_turn(bal: dict, who: str, g0: int, g1: int, cfg: FCConfig, mask: np.ndarray | None = None) -> dict:
    """Balance-axis evaluation of frames g0..g1 (inclusive, global) for who = 'lead' | 'follow' | 'couple' (the
    combined COM over the follower's anchored foot) | 'both' (each dancer over their own stance foot; the verdict is
    the worse one's). mask (g1 - g0 + 1 booleans, optional): only those frames count for the subject (a one-foot
    pivot: the frames with the weight on the pivot foot)."""
    def one(key: str, use_mask: bool) -> dict:
        b = bal.get(key)
        if b is None:
            return {"frames": 0, "verdict": "unknown"}
        sl = slice(g0, g1 + 1)
        ok = np.isfinite(b["com_offset_m"][sl]) & (b["stance"][sl] > 0 if "stance" in b else True)
        if use_mask and mask is not None:
            ok = ok & np.asarray(mask, bool)
        n = int(ok.sum())
        if n == 0:
            return {"frames": 0, "verdict": "unknown"}
        off, tilt, sc = b["com_offset_m"][sl][ok], b["tilt_deg"][sl][ok], b["score"][sl][ok]
        share = float((sc >= cfg.bal_aligned_score).mean())
        verdict = "good" if share >= cfg.bal_good_share else ("ok" if share >= cfg.bal_ok_share else "needs work")
        return {"frames": n, "aligned_share": round(share, 3), "mean_offset_m": round(float(off.mean()), 4),
                "median_offset_m": round(float(np.median(off)), 4),
                "p90_offset_m": round(float(np.percentile(off, 90)), 4),
                "mean_tilt_deg": round(float(tilt.mean()), 2), "mean_score": round(float(sc.mean()), 3),
                "verdict": verdict}

    rank = {"good": 0, "ok": 1, "needs work": 2, "unknown": 3}
    per = {r: one(r, r == who) for r in ("lead", "follow") if bal.get(r) is not None}
    if who == "both":
        cands = [per[r] for r in per if per[r]["verdict"] != "unknown"]
        main = max(cands, key=lambda d: rank[d["verdict"]]) if cands else {"frames": 0, "verdict": "unknown"}
        measure = "each dancer's COM over their own stance foot (verdict: the worse of the two)"
    elif who == "couple":
        main = one("couple", True)
        measure = "combined COM of the couple over the follower's anchored foot"
    else:
        main = per.get(who, {"frames": 0, "verdict": "unknown"})
        measure = f"{who} COM over {'his' if who == 'lead' else 'her'} stance foot"
    ev = {"subject": who, "measure": measure, **main}
    if main["verdict"] == "unknown":
        ev["text"] = "unknown: no foot in contact during the turn"
    elif who == "both":
        worse = next(r for r in per if per[r] is main)
        ev["worse"] = worse
        ev["text"] = (f"{main['verdict']}: COM over the stance foot - " + ", ".join(
            f"{r} {per[r]['aligned_share']:.0%}" for r in per if per[r]["verdict"] != "unknown")
            + f" of the turn ({worse}: mean {100 * main['mean_offset_m']:.1f} cm off, tilt "
              f"{main['mean_tilt_deg']:.1f} deg)")
    else:
        where = "the anchored foot" if who == "couple" else "the stance foot"
        ev["text"] = (f"{main['verdict']}: COM over {where} {main['aligned_share']:.0%} of the turn "
                      f"(mean {100 * main['mean_offset_m']:.1f} cm off, tilt {main['mean_tilt_deg']:.1f} deg)")
    ev["per_dancer"] = per
    ev["note"] = "estimate from the fitted bodies (SMPL-X COM; foot contact from joint heights / speeds)"
    return ev


TURN_KIND_TEXT = {"one_foot": "a pivot on one foot", "both_feet": "turned on both feet",
                  "stepped": "a stepped turn (the weight changed feet / the foot travelled)",
                  "airborne": "no foot in contact"}
TURN_KIND_LABEL = {"one_foot": "one foot", "both_feet": "both feet", "stepped": "stepped", "airborne": "airborne"}


def classify_turn(b: dict, g0: int, g1: int, cfg: FCConfig) -> dict:
    """How one dancer's feet carried a turn over frames g0..g1 (balance_axis dict b; global frames):
      'both_feet'  both feet in contact for >= turn_double_share of the turn (a torso turn standing on both feet);
      'one_foot'   a PIVOT: one foot in contact >= turn_pivot_contact of the turn, the weight (stance) on it for
                   >= turn_pivot_stance of the frames with a stance and its ball within turn_pivot_travel_m of its
                   median position (it turns on the spot);
      'stepped'    anything else: the weight changed feet or the foot travelled;
      'airborne'   no foot in contact for most of it.
    Only a one-foot pivot asks 'is the axis over the pivot foot' (the user, 2026-10-07: 'the axis alignment of the
    pivot point of the foot'), so only it is graded (with dials). -> {kind, pivot_foot, double_share,
    contact_share {left, right}, ball_travel_m {left, right}, stance_share, stance_changes}."""
    sl = slice(g0, g1 + 1)
    w = {f: np.asarray(b["contact"][f][sl]) >= 0.5 for f in ("left", "right")}
    n = max(1, len(w["left"]))
    st = np.asarray(b["stance"][sl])
    stf = st[st > 0]
    share = {f: float(w[f].mean()) for f in w}
    travel = {}
    for f in w:
        B = np.asarray(b["ball"][f][sl], float)
        travel[f] = float(np.linalg.norm(B - np.median(B, 0), axis=1).max()) if len(B) else float("nan")
    pf = max(share, key=share.get)
    code = 1 if pf == "left" else 2
    stance_share = float((stf == code).mean()) if len(stf) else 0.0
    changes = int((np.diff(stf) != 0).sum()) if len(stf) > 1 else 0
    double = float((w["left"] & w["right"]).mean())
    if float((w["left"] | w["right"]).mean()) < 0.5:
        kind = "airborne"
    elif double >= cfg.turn_double_share:
        kind = "both_feet"
    elif (share[pf] >= cfg.turn_pivot_contact and stance_share >= cfg.turn_pivot_stance
          and travel[pf] <= cfg.turn_pivot_travel_m):
        kind = "one_foot"
    else:
        kind = "stepped"
    return {"kind": kind, "pivot_foot": pf if kind == "one_foot" else None, "frames": int(n),
            "double_share": round(double, 3), "contact_share": {f: round(v, 3) for f, v in share.items()},
            "ball_travel_m": {f: round(v, 3) for f, v in travel.items()}, "stance_share": round(stance_share, 3),
            "stance_changes": changes}


def graded_turn(bal: dict, who: str, g0: int, g1: int, cfg: FCConfig) -> dict:
    """evaluate_turn for a REORIENTATION of one dancer (axis pivots, rotations): classify_turn first - only a pivot on
    one foot gets good / ok / needs work (over the frames with the weight on that foot); a turn on both feet or a
    stepped turn keeps its numbers but verdict 'not graded' (balance_verdict = what the COM test alone would say)."""
    b = bal.get(who)
    if b is None or "contact" not in b:
        ev = evaluate_turn(bal, who, g0, g1, cfg)
        ev.update({"graded": ev["verdict"] != "unknown", "turn_kind": "unknown"})
        return ev
    tk = classify_turn(b, g0, g1, cfg)
    if tk["kind"] == "one_foot":
        code = 1 if tk["pivot_foot"] == "left" else 2
        ev = evaluate_turn(bal, who, g0, g1, cfg, mask=np.asarray(b["stance"][g0:g1 + 1]) == code)
        ev["text"] = f"{ev['text']} - a pivot on the {tk['pivot_foot']} foot"
        ev["measure"] = f"{who} COM over {'his' if who == 'lead' else 'her'} pivot foot ({tk['pivot_foot']})"
        ev["graded"] = ev["verdict"] != "unknown"
    else:
        ev = evaluate_turn(bal, who, g0, g1, cfg)
        ev["balance_verdict"] = ev["verdict"]
        ev["verdict"] = "unknown" if tk["kind"] == "airborne" else "not graded"
        ev["graded"] = False
        extra = "" if ev["balance_verdict"] == "unknown" else (
            f" (COM over the stance foot {ev.get('aligned_share', 0):.0%} of it, mean "
            f"{100 * ev.get('mean_offset_m', 0):.1f} cm off)")
        ev["text"] = f"not graded: {TURN_KIND_TEXT[tk['kind']]}, not a pivot on one foot" + extra
    ev["turn_kind"] = tk["kind"]
    ev["turn_kind_label"] = TURN_KIND_LABEL[tk["kind"]]
    ev["feet"] = tk
    return ev


# ----------------------------------------------------------------------------------------------- dials

def _spread(p: np.ndarray) -> tuple[np.ndarray, float]:
    """(median point, p90 distance to it) of floor points p (n, 2)."""
    c = np.median(p, 0)
    return c, float(np.percentile(np.linalg.norm(p - c, axis=1), 90)) if len(p) else float("inf")


def _orbit_deg(L: np.ndarray, c: np.ndarray) -> float:
    """Signed angle (deg, ccw from above +) the floor points L (n, 2) sweep about c (unwrapped end - start)."""
    rel = L - c
    ang = np.degrees(np.unwrap(np.arctan2(-rel[:, 1], rel[:, 0])))
    return float(ang[-1] - ang[0]) if len(ang) > 1 else 0.0


def dial_centre(g: list, full: dict, cfg: FCConfig) -> tuple[dict | None, dict]:
    """The point a chain of same-sign axis pivots g turns about, or None -> (choice, log entry).
    Candidates, tried from the least moving: the leader's chest ('leader_spin': he turns on his own axis), the couple's
    combined COM ('couple_turn') and the follower's chest ('leader_around_follower'); a candidate must stay within
    dial_place_m (p90 distance to its median) AND, for the two orbit kinds, the leader must really go ROUND it: his
    orbit about it has the sign of the turn and >= dial_orbit_ratio x its degrees (review 2026-10-07: at HUD 16.1-17.3
    s the couple COM stayed put while the couple swapped places +132 deg ccw and he spun -286 deg cw under the joined
    arms - his own spin, not a couple turn about that point)."""
    total = float(sum(ap["signed_deg"] for ap in g))
    g0, g1 = g[0]["_g"][0], g[-1]["_g"][1]
    L = full["lead"][g0:g1 + 1]
    Lf = L[np.isfinite(L).all(1)]
    cands = []
    for kind, key in (("leader_spin", "lead"), ("couple_turn", "couple"), ("leader_around_follower", "follow")):
        P = full.get(key)
        if P is None:
            continue
        P = P[g0:g1 + 1]
        P = P[np.isfinite(P).all(1)]
        if len(P) < 2:
            continue
        c, sp = _spread(P)
        orbit = None if kind == "leader_spin" or len(Lf) < 2 else _orbit_deg(Lf, c)
        cands.append({"kind": kind, "key": key, "centre": c, "spread": sp, "orbit": orbit})
    cands.sort(key=lambda x: x["spread"])
    log = {"axis_pivots": [ap["id"] for ap in g], "degrees": round(total, 1), "t": [g[0]["t0"], g[-1]["t1"]],
           "centre_spread_m": {c["kind"]: round(c["spread"], 3) for c in cands},
           "orbit_deg": {c["kind"]: round(c["orbit"], 1) for c in cands if c["orbit"] is not None},
           "dial": False, "kind": None, "rejected": {}}
    for c in cands:
        if c["spread"] > cfg.dial_place_m:
            log["rejected"][c["kind"]] = f"moves {c['spread']:.2f} m > {cfg.dial_place_m:.2f} m"
            continue
        if c["orbit"] is not None and not (np.sign(c["orbit"]) == np.sign(total)
                                          and abs(c["orbit"]) >= cfg.dial_orbit_ratio * abs(total)):
            log["rejected"][c["kind"]] = (f"he does not go round it: orbit {c['orbit']:+.0f} deg vs turn "
                                          f"{total:+.0f} deg")
            continue
        log.update({"dial": True, "kind": c["kind"]})
        return c, log
    return None, log


CB_CONTEXTS = ("into_counterbalance", "out_of_counterbalance")


def find_dials(t_axes: list, axis_pivots: list, cb_pivots: list, full: dict, beat: float, cfg: FCConfig,
               log: list | None = None) -> list:
    """DIALS: repeated turning about one point.
    counterbalance  every counterbalance pivot (the leader circles the follower's anchored foot): centre = the pivot
                    point, degrees = its swept angle (counterbalance.json), subject 'couple'.
    turns           consecutive axis pivots of one sign chained through Ts (each next one within dial_max_hold_beats
                    of the previous), summed: >= dial_min_deg is a dial when one point stays put over the turning and
                    the motion really turns about it (dial_centre: his chest = 'leader_spin', subject 'lead'; the
                    couple's combined COM = 'couple_turn' or the follower's chest = 'leader_around_follower' - his
                    orbit about it agrees with the turn -, subject 'both'). A chain that fails as a whole is tried in
                    its contiguous sub-chains of >= dial_min_deg (longest first; the parts either side are tried
                    again), so one spin on the spot inside a longer travelling sequence still gets its dial. A turn
                    while travelling (no point stays) is no dial: its axis pivot(s) record it. Pivots chain through a
                    T or a waypoint (a brief T); the turning into / out of a counterbalance is never a dial (the
                    counterbalance dial is right there).
    degrees = the T heading change (sum of the pivots; ccw +); orbit_deg = the leader's angle swept about the centre
    (non-spin kinds). full = {"t", "lead" (T, 2) chest on the floor, "follow" (T, 2) | None, "couple" (T, 2) | None,
    "yaw" (T,) unwrapped}. log (optional) collects every chain tried (>= dial_min_deg) with the spreads / orbits of
    the candidate centres, the rejections and whether it became a dial."""
    dials = []
    tt = full["t"]
    for pv in cb_pivots:
        k = np.flatnonzero((tt >= pv["t0"]) & (tt <= pv["t1"]))
        before = [ta for ta in t_axes if ta["t1"] <= pv["t0"] + 1e-6]
        la = pv.get("leader_axis") or {}
        dials.append({
            "id": len(dials), "kind": "counterbalance", "t0": pv["t0"], "t1": pv["t1"], "centre": list(pv["pivot"]),
            "degrees": round(float(pv["swept_signed_deg"]), 2), "abs_deg": round(abs(float(pv["swept_signed_deg"])), 2),
            "turns": round(abs(float(pv["swept_signed_deg"])) / 360.0, 3), "direction": pv["direction"],
            "who": f"lead circles the follow's anchored {pv.get('pivot_foot') or ''} foot".replace("  ", " "),
            "subject": "couple", "radius_m": pv["leader_radius_m"], "arc_start_deg": pv["arc_start_deg"],
            "heading_deg": la.get("turn_deg"), "orbit_deg": round(float(pv["swept_signed_deg"]), 2),
            "counterbalance_pivot": pv["id"], "axis_pivots": [], "t_axis": before[-1]["id"] if before else None,
            "_g": (int(k[0]), int(k[-1])) if len(k) else None})
    groups, cur = [], []
    for ap in axis_pivots:
        if ap.get("context") in CB_CONTEXTS:
            # the turning into / out of a counterbalance: its dial is right there (two dials ~15 cm apart would read
            # as one) - an axis pivot only
            if cur:
                groups.append(cur)
                cur = []
            continue
        if cur:
            last = cur[-1]
            chained = (ap.get("_from", ("T", ap["from_t"])) == last.get("_to", ("T", last["to_t"]))
                       and np.sign(ap["signed_deg"]) == np.sign(last["signed_deg"])
                       and ap["t0"] - last["t1"] <= cfg.dial_max_hold_beats * beat + 1e-6)
            if not chained:
                groups.append(cur)
                cur = []
        cur.append(ap)
    if cur:
        groups.append(cur)

    def make(g: list, c: dict) -> None:
        total = float(sum(ap["signed_deg"] for ap in g))
        g0, g1 = g[0]["_g"][0], g[-1]["_g"][1]
        L = full["lead"][g0:g1 + 1]
        L = L[np.isfinite(L).all(1)]
        kind = c["kind"]
        if kind == "leader_spin":
            radius, start, orbit = 0.25, g[0]["from_yaw_deg"], None
            who, subject = "lead turns on his own axis", "lead"
        else:
            rel = L - c["centre"]
            ang = np.degrees(np.unwrap(np.arctan2(-rel[:, 1], rel[:, 0])))
            radius, start, orbit = float(np.median(np.linalg.norm(rel, axis=1))), float(wrap180(ang[0])), \
                float(ang[-1] - ang[0])
            who = ("the couple turns about a shared point (their combined COM)" if kind == "couple_turn"
                   else "lead turns around the follow")
            subject = "both"
        did = len(dials)
        for ap in g:
            ap["dial"] = did
        inner = {ap["to_t"] for ap in g[:-1] if ap["to_t"] is not None}      # the Ts the chain passes through
        for ta in t_axes:
            if ta["id"] in inner:
                ta["dial"] = did
        dials.append({
            "id": did, "kind": kind, "t0": g[0]["t0"], "t1": g[-1]["t1"], "centre": xyz(c["centre"]),
            "degrees": round(total, 2), "abs_deg": round(abs(total), 2), "turns": round(abs(total) / 360.0, 3),
            "direction": "ccw" if total > 0 else "cw", "who": who, "subject": subject, "radius_m": round(radius, 4),
            "arc_start_deg": round(float(wrap180(start)), 2), "heading_deg": round(total, 2),
            "orbit_deg": None if orbit is None else round(orbit, 2), "centre_spread_m": round(c["spread"], 4),
            "counterbalance_pivot": None, "axis_pivots": [ap["id"] for ap in g], "t_axis": g[0]["from_t"],
            "_g": (g0, g1)})

    def try_chain(g: list) -> None:
        """the whole chain, else its longest passing contiguous sub-chain (then the parts either side)"""
        if not g:
            return
        for n in range(len(g), 0, -1):
            for s in range(0, len(g) - n + 1):
                sub = g[s:s + n]
                if abs(sum(ap["signed_deg"] for ap in sub)) < cfg.dial_min_deg:
                    continue
                c, entry = dial_centre(sub, full, cfg)
                if log is not None and (n == len(g) or c is not None):
                    log.append(entry | ({"sub_chain_of": [ap["id"] for ap in g]} if n < len(g) else {}))
                if c is not None:
                    make(sub, c)
                    try_chain(g[:s])
                    try_chain(g[s + n:])
                    return

    for g in groups:
        try_chain(g)
    dials.sort(key=lambda d: d["t0"])
    remap = {d["id"]: n for n, d in enumerate(dials)}
    for n, d in enumerate(dials):
        d["id"] = n
    for rec in axis_pivots + t_axes:
        if rec.get("dial") is not None:
            rec["dial"] = remap[rec["dial"]]
    return dials


def balance_json(t: np.ndarray, bal: dict, cfg: FCConfig) -> dict:
    """The per-frame balance tracks for the JSON (positions mm-rounded, NaN -> null)."""
    out = {"t": _r(t, 4), "frame_step_s": round(float(np.median(np.diff(t))), 5) if len(t) > 1 else None,
           "stance_codes": {"0": "none (no foot in contact)", "1": "left", "2": "right"},
           "com_source": bal.get("_com_source")}
    for role in ("lead", "follow"):
        b = bal.get(role)
        if b is None:
            continue
        out[role] = {"stance": b["stance"].tolist(), "double": b["double"].tolist(),
                     "support": [None if not np.isfinite(q).all() else
                                 [round(float(q[0]), 3), 0.0, round(float(q[1]), 3)] for q in b["support"]],
                     "chest": _r(b["chest"], 3), "com": _r(b["com"], 3), "com_offset_m": _r(b["com_offset_m"], 3),
                     "tilt_deg": _r(b["tilt_deg"], 2), "score": _r(b["score"], 3)}
    c = bal.get("couple")
    if c is not None:
        inside = c["pivot"] >= 0
        out["couple"] = {
            "com": [_r(c["com"][i], 3) if inside[i] else None for i in range(len(t))],
            "support": [[round(float(c["support"][i, 0]), 3), 0.0, round(float(c["support"][i, 1]), 3)]
                        if inside[i] else None for i in range(len(t))],
            "com_offset_m": _r(c["com_offset_m"], 3), "tilt_deg": _r(c["tilt_deg"], 2), "score": _r(c["score"], 3),
            "pivot": [int(p) if p >= 0 else None for p in c["pivot"]]}
    return out


def strip_private(recs: list) -> list:
    for r in recs:
        for k in [k for k in r if k.startswith("_")]:
            r.pop(k)
    return recs


# ----------------------------------------------------------------------------------------------- counterbalance

def pivots_from(cbdoc: dict | None) -> tuple[list, list]:
    """(pivots, counterbalances) records from a counterbalance.json (world frame, reference seconds)."""
    if not cbdoc:
        return [], []
    pivots, cbs = [], []
    for n, iv in enumerate(cbdoc.get("intervals", [])):
        tr = iv.get("com_track") or {}
        tt, cc = np.asarray(tr.get("t") or [], float), np.asarray(tr.get("com") or [], float)
        k = (tt >= iv["t0"]) & (tt <= iv["t1"]) if len(tt) else np.zeros(0, bool)
        dot = np.median(cc[k], 0) if k.any() else None
        cbs.append({"id": n, "kind": iv["kind"], "t0": iv["t0"], "t1": iv["t1"], "confidence": iv.get("confidence"),
                    "com_floor": None if dot is None else [round(float(dot[0]), 4), 0.0, round(float(dot[2]), 4)],
                    "connection": (iv.get("connection") or {}).get("pair"), "pivots": []})
        for pv in iv.get("pivots", []):
            rec = {"id": len(pivots), "counterbalance": n, "t0": pv["t0"], "t1": pv["t1"],
                   "counterbalance_t": [iv["t0"], iv["t1"]], "pivot": pv["pivot"], "pivot_foot": pv["pivot_foot"],
                   "leader_radius_m": pv["leader_radius_m"], "swept_deg": pv["swept_deg"],
                   "swept_signed_deg": pv["swept_signed_deg"], "direction": pv["direction"],
                   "arc_start_deg": pv["arc_start_deg"], "confidence": pv["confidence"],
                   "leader_path": pv.get("leader_path"), "beat_ticks": pv.get("beat_ticks", []),
                   "leader_axis": None}
            cbs[-1]["pivots"].append(rec["id"])
            pivots.append(rec)
    return pivots, cbs


def ensure_counterbalance(take: str, smplx_dir: Path, lead_sha1: str) -> tuple[dict | None, str]:
    """counterbalance.json of the take, re-detected when missing or computed on other fits (lead sha1), another
    beat grid (beats_sha1) or another detector (code_sha1)."""
    p = cb.work_dir(take) / "counterbalance" / "counterbalance.json"
    if p.exists():
        doc = json.loads(p.read_text())
        src = doc.get("source", {})
        if src.get("lead_sha1") == lead_sha1 and src.get("beats_sha1") == cb.grid_sha1(cb.work_dir(take)) \
                and src.get("code_sha1") == cb.code_sha1():
            return doc, "counterbalance.json (same fits, beat grid and detector)"
    try:
        res = cb.run(take, smplx_dir, review=False)
    except Exception as e:  # noqa: BLE001 - floor craft without pivots is still useful
        return None, f"counterbalance detection failed: {type(e).__name__}: {e}"
    return res["_json"], "counterbalance re-detected (missing, or other fits / beat grid)"


# ----------------------------------------------------------------------------------------------- analysis

def analyse(npz: dict, beat_s: float | None = None, cfg: FCConfig | None = None, cbdoc: dict | None = None,
            follow: dict | None = None, coms: dict | None = None, masses: dict | None = None) -> dict:
    """Floor craft of a lead npz dict (t, joints, pose parameters) -> {"track", "plants", "moves", "rotations",
    "pivots", "counterbalances", "t_axes", "transitions", "axis_pivots", "dials", "balance", "events", "info"} in the
    world frame, reference seconds. follow = the follower's npz (same time grid; balance axis, dial centres and the
    couple COM need it); coms = {role: (T, 3) COM, role_source: str} (run() passes the SMPL-X mass-element COM;
    default: the joint estimate); masses = {role: kg} for the combined COM (default: equal weights)."""
    cfg = cfg or FCConfig()
    t = np.asarray(npz["t"], float)
    beat = float(beat_s) if beat_s else 60.0 / cfg.default_bpm
    origin, left, src = raw_axis(npz)
    pivots, cbs = pivots_from(cbdoc)
    out = {"plants": [], "moves": [], "rotations": []}
    tracks, segs = [], []
    n_swings = 0
    for a, b in cb.contiguous_segments(t, cfg.gap_frames):
        if b - a + 1 < 2:
            continue
        ax = smooth_axis(t[a:b + 1], origin[a:b + 1], left[a:b + 1], cfg.cutoff_hz)
        tracks.append(ax)
        # cut the track at every counterbalance pivot: the pivot owns the leader's motion inside it (his circling, his
        # axis winding round), recorded on the pivot as leader_axis; plants / moves / rotations live outside them, and
        # no T forms inside it (blocked)
        tt, n = ax["t"], len(ax["t"])
        cuts = []
        for pv in pivots:
            k0 = max(0, int(np.searchsorted(tt, pv["t0"], "right")) - 1)     # last frame <= t0
            k1 = min(n - 1, int(np.searchsorted(tt, pv["t1"])))               # first frame >= t1
            if pv["t0"] < tt[-1] and pv["t1"] > tt[0] and k1 > k0:
                cuts.append((k0, k1, pv))
        start, before, pieces = 0, None, []
        blocked = np.zeros(n, bool)
        for k0, k1, pv in sorted(cuts, key=lambda c: c[0]):
            k0 = max(k0, start)
            if k1 <= k0:
                continue
            pieces.append((start, k0, before, pv))
            pv["leader_axis"] = leader_axis_in(ax, k0, k1, cfg)
            blocked[k0 + 1:k1] = True
            start, before = k1, pv
        pieces.append((start, n - 1, before, None))
        for s0, s1, bf, af in pieces:
            if s1 - s0 + 1 < 2:
                continue
            base = {k: len(out[k + "s"]) for k in ("plant", "move", "rotation")} | {"swing": n_swings}
            seg = segment(_slice(ax, s0, s1), beat, cfg, base, bf, af)
            n_swings += seg["swings"]
            for k in out:
                out[k] += seg[k]
        brief: list = []
        segs.append({"ax": ax, "eps": t_episodes(ax, beat, cfg, blocked, brief), "brief": brief, "off": a})
    t_axes, transitions, axis_pivots = t_axis_records(segs, cfg, pivots)

    # balance axis per dancer (+ the couple inside the counterbalance pivots), on the full frame grid
    T = len(t)
    coms = dict(coms or {})
    com_src: dict = {}
    J, bal = {}, {}
    for role, d in (("lead", npz), ("follow", follow)):
        if d is None:
            continue
        Jr = cb._bridge(np.asarray(d["joints"], float)[:, :55], t)
        J[role] = _smooth_on_segments(t, Jr, cfg.com_cutoff_hz, cfg.gap_frames)
        if coms.get(role) is not None:
            c, com_src[role] = np.asarray(coms[role], float), coms.get(f"{role}_source", "given")
        else:
            c, com_src[role] = cb.joints_com(Jr), "joints (de Leva-style segment weights)"
        c = _smooth_on_segments(t, cb._bridge(c, t), cfg.com_cutoff_hz, cfg.gap_frames)
        bal[role] = balance_axis(t, J[role], c, cfg)
    com_c = None
    if "lead" in bal and "follow" in bal:
        m = masses or {}
        wl, wf = float(m.get("lead") or 1.0), float(m.get("follow") or 1.0)
        com_c = (wl * bal["lead"]["com"] + wf * bal["follow"]["com"]) / (wl + wf)
        bal["couple"] = couple_axis(t, J["follow"], com_c, pivots, cfg)
        com_src["couple"] = f"mass-weighted (lead {wl:.1f}, follow {wf:.1f})" if masses else "equal weights"
    bal["_com_source"] = com_src

    # dials (counterbalance pivots + repeated turning about one point) and the turn evaluations
    lead_floor = np.full((T, 2), np.nan)
    for sg in segs:
        lead_floor[sg["off"]:sg["off"] + len(sg["ax"]["t"])] = sg["ax"]["origin"]
    full = {"t": t, "lead": lead_floor,
            "follow": J["follow"][:, CHEST][:, [0, 2]] if "follow" in J else None,
            "couple": com_c[:, [0, 2]] if com_c is not None else None}
    dial_log: list = []
    dials = find_dials(t_axes, axis_pivots, pivots, full, beat, cfg, dial_log)
    n_min = max(1, int(round(cfg.bal_eval_min_beats * beat / (float(np.median(np.diff(t))) if T > 1 else 1 / 30))))

    def ev(who: str, g0: int, g1: int, turn: bool = False) -> dict:
        """evaluate_turn over g0..g1 widened (centred) to >= bal_eval_min_beats; the window is recorded. turn: a
        reorientation of one dancer (graded_turn: only a pivot on one foot is graded)."""
        if g1 - g0 + 1 < n_min:
            c = (g0 + g1) // 2
            g0, g1 = max(0, c - n_min // 2), min(T - 1, c - n_min // 2 + n_min - 1)
        if turn:
            e = graded_turn(bal, who, g0, g1, cfg)
        else:
            e = evaluate_turn(bal, who, g0, g1, cfg)
            e["graded"] = e["verdict"] != "unknown"
        return {**e, "window": [round(float(t[g0]), 4), round(float(t[g1]), 4)]}

    for ap in axis_pivots:
        ap["evaluation"] = ev("lead", *ap["_g"], turn=True)
    for d in dials:
        if d["_g"] is None:
            d["evaluation"] = {"subject": d["subject"], "frames": 0, "verdict": "unknown", "graded": False,
                               "text": "unknown: outside the analysed frames"}
        else:
            d["evaluation"] = ev(d["subject"], *d["_g"])
            if "lead" in bal and "contact" in bal["lead"]:      # informative: how his feet carried it
                tk = classify_turn(bal["lead"], *d["_g"], cfg)
                d["evaluation"].update({"turn_kind": tk["kind"], "turn_kind_label": TURN_KIND_LABEL[tk["kind"]],
                                        "feet": tk})
        if d["counterbalance_pivot"] is not None:
            pivots[d["counterbalance_pivot"]]["evaluation"] = d["evaluation"]
            pivots[d["counterbalance_pivot"]]["dial"] = d["id"]
    for tr in transitions:
        tr["dial"] = next((d["id"] for d in dials if d["counterbalance_pivot"] is not None
                           and d["counterbalance_pivot"] == tr["counterbalance_pivot"]), None)
        if tr["dial"] is None and tr["axis_pivot"] is not None:
            tr["dial"] = axis_pivots[tr["axis_pivot"]]["dial"]
    for r in out["rotations"]:
        g0 = int(np.clip(np.searchsorted(t, r["t0"] - 1e-6), 0, T - 1))
        g1 = int(np.clip(np.searchsorted(t, r["t1"] + 1e-6, "right") - 1, g0, T - 1))
        r["evaluation"] = ev("lead", g0, g1, turn=True)
    for recs in (t_axes, axis_pivots, dials):
        strip_private(recs)

    events = sorted([{"type": k[:-1], "id": r["id"], "t0": r["t0"], "t1": r["t1"]} | (
                        {"kind": r["kind"]} if k == "moves" else {})
                     for k in ("plants", "moves", "rotations") for r in out[k]]
                    + [{"type": "pivot", "id": p["id"], "t0": p["t0"], "t1": p["t1"]} for p in pivots]
                    + [{"type": "counterbalance", "id": c["id"], "t0": c["t0"], "t1": c["t1"]} for c in cbs]
                    + [{"type": "t_axis", "id": r["id"], "t0": r["t0"], "t1": r["t1"]} for r in t_axes]
                    + [{"type": "transition", "id": r["id"], "t0": r["t0"], "t1": r["t1"]} for r in transitions]
                    + [{"type": "axis_pivot", "id": r["id"], "t0": r["t0"], "t1": r["t1"]} for r in axis_pivots]
                    + [{"type": "dial", "id": r["id"], "t0": r["t0"], "t1": r["t1"], "kind": r["kind"]}
                       for r in dials],
                    key=lambda e: (e["t0"], e["type"]))
    step = cfg.path_step_s
    tt = np.concatenate([ax["t"] for ax in tracks])
    oo = np.concatenate([ax["origin"] for ax in tracks])
    yy = np.concatenate([ax["yaw"] for ax in tracks])
    ss = np.concatenate([ax["speed"] for ax in tracks])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    idx = np.unique(np.concatenate([np.arange(0, len(tt), max(1, int(round(step / dt)))), [len(tt) - 1]]))
    plant_s = sum(p["t1"] - p["t0"] for p in out["plants"])
    t_s = sum(r["t1"] - r["t0"] for r in t_axes)
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    return {**out, "pivots": pivots, "counterbalances": cbs, "events": events,
            "t_axes": t_axes, "transitions": transitions, "axis_pivots": axis_pivots, "dials": dials,
            "balance": balance_json(t, bal, cfg), "_bal": bal, "_t": t,
            "track": {"t": _r(tt[idx], 4), "origin": xyz(oo[idx]), "yaw_deg": _r(yy[idx], 2),
                      "speed_m_s": _r(ss[idx], 3), "step_s": round(step, 5)},
            "_full": {"t": tt, "origin": oo, "yaw": yy, "speed": ss},
            "info": {"axis_source": src, "beat_s": round(beat, 5), "frames": int(len(t)),
                     "coverage": [[round(float(tracks[k]["t"][0]), 4), round(float(tracks[k]["t"][-1]), 4)]
                                  for k in range(len(tracks))],
                     "counts": {k: len(v) for k, v in out.items()} | {
                         "pivots": len(pivots), "steps": sum(m["kind"] == "step" for m in out["moves"]),
                         "travel_moves": sum(m["kind"] == "travel" for m in out["moves"]),
                         "t_axes": len(t_axes), "transitions": len(transitions), "axis_pivots": len(axis_pivots),
                         "dials": len(dials),
                         "dropped_short_ts": sum(ta["dropped_before"] for ta in t_axes),
                         "axis_pivots_by_context": {c: sum(ap["context"] == c for ap in axis_pivots) for c in
                                                    sorted({ap["context"] for ap in axis_pivots})},
                         "axis_pivots_by_turn_kind": {k: sum(ap["evaluation"].get("turn_kind") == k
                                                             for ap in axis_pivots) for k in
                                                      sorted({ap["evaluation"].get("turn_kind") or "unknown"
                                                              for ap in axis_pivots})},
                         "axis_pivots_graded": sum(bool(ap["evaluation"].get("graded")) for ap in axis_pivots)},
                     "plant_share": round(plant_s / span, 3) if span else None,
                     "t_axis_share": round(t_s / span, 3) if span else None,
                     "com_source": com_src, "turn_groups": dial_log,
                     "origin_speed_m_s": {"p10": round(float(np.percentile(ss, 10)), 3),
                                          "p50": round(float(np.median(ss)), 3),
                                          "p90": round(float(np.percentile(ss, 90)), 3)}}}

NOTES = [
    "World: right-handed Y-up metres, floor y = 0; Unity = (x, y, -z). Times = reference seconds (npz t).",
    "Leader floor axis: origin = his chest (spine3 joint) on the floor; long axis = the chest's left-right line "
    "(spine3 global +x, projected); forward = its floor perpendicular on the chest's front. yaw_deg = angle of "
    "forward from +x towards -z (ccw seen from above = positive about +Y); forward / long_axis are unit vectors "
    "[x, 0, z]; long_axis points to the leader's LEFT. axis.long_m / short_m = the live drawing size (VIEWER_SPEC 3.6).",
    "No plant, move or rotation overlaps a counterbalance pivot: the track is cut at the pivots, which own the leader's "
    "motion inside them (pivots[].leader_axis). Draw every record as it is - nothing needs skipping.",
    "plants: the leader holds the dance in one place (origin within params.plant_radius_m of the centroid, net drift "
    "<= params.plant_drift_m_s, >= params.plant_min_beats; adjacent plants whose union passes are merged); origin = "
    "centroid; yaw_deg / forward / long_axis = the axis he ESTABLISHED (circular mean over `held`, its longest stretch "
    "without a rotation in progress; yaw_source); yaw_mean_deg = circular mean over the whole plant; turn_deg = "
    "yaw_end - yaw_start (0.2 s means at its ends; + = ccw).",
    "moves: the axis between two plants (from_plant / to_plant null at the take edges or next to a pivot: from_pivot / "
    "to_pivot). kind 'step' = one step from a plant to the next (origin_path_m < params.step_max_m within "
    "params.step_max_beats; a step whose plants touch sits at their junction, 1-3 frames), kind 'travel' = the rest. "
    "path = from -> the origin sampled at params.path_step_s -> to, so it runs plant cross to plant cross (length_m = "
    "its length; "
    "origin_path_m = the origin's own path between the plants); displacement / heading = plant origin to plant "
    "origin; yaw_change_deg = unwrapped yaw from the previous plant's end to the next plant's start (full turns kept).",
    "rotations: zig-zag swings of the axis yaw >= params.rotation_deg, split at plant (group) / travel-move "
    "boundaries (plants chained by steps are one group: a turn while stepping stays whole); signed_deg = to - from "
    "(unwrapped, + = ccw); context 'plant' (plant = the plant holding most of the angle) or 'move' (move = the step "
    "or travel move holding it); plants = every plant it touches; centre = that plant's origin, else the axis origin "
    "mid-rotation; swing ties the pieces of one swing.",
    "pivots: counterbalance.json pivots (follower's anchored foot = pivot, the leader's circling path, swept angle, "
    "beat ticks; arc_start_deg = angle of the leader about the pivot at t0 from +x towards -z) + leader_axis (his "
    "chest path on the floor over the pivot, its length and the net turn of his axis, unwrapped); counterbalances: "
    "its intervals with the median combined-COM floor point. Ids match the record order of counterbalance.json.",
    "events: every record by start time - the viewer adds each to the fading floor record once the playhead passes "
    "its t0 (plants / rotations / pivots) and draws moves progressively; track = the smoothed axis at "
    "params.path_step_s for the live axis and reviews.",
    "VERSION 3 (revision 3.1; schema: work/larissa_kadu_38/review/floorcraft2/SCHEMA_V2.md, CONTRACTS.md). t_axes: "
    "the leader's STABLE T - one origin (median chest on the floor) and heading (mean) per episode, AVERAGED over "
    "the frames he spent on it (formed_origin / formed_yaw_deg = the forming window's values), growing in over "
    "[t0, t_formed] and held while he stays in the allowed region (stem corridor along forward: |side| <= "
    "params.t_axis.half_width_m, |forward| <= forward_m; crossbar corridor along parallel: |forward| <= "
    "half_width_m, |side| <= side_m) with his heading within pivot_deg; a sustained (sustain_beats) exit or rotation "
    "ends it (end_reason). Every T holds >= params.t_axis.min_hold_beats after forming (shorter ones are dropped: "
    "dropped_before counts them; the take's last T may be cut by its end). parallel = his LEFT (the crossbar), "
    "forward = the stem. same_place_as: a rotated T at the previous T's place (origin copied). Draw the current T "
    "bright, old ones faded teal; nothing follows his chest.",
    "transitions: consecutive T origins (from -> to) for the dotted line, t0 = the old T's end, t1 = the new T's "
    "start; same_place (distance 0) for a T rotated in place; axis_pivots = the reorientations on the way. "
    "axis_pivots: every reorientation of his axis > params.t_axis.pivot_deg (signed_deg unwrapped, + = ccw; stayed = "
    "same place; at = old origin, else his chest mid-turn; t0..t1 = the heading leaving the old / reaching the new "
    "heading); context between_ts (T -> T, through the headings of dropped brief Ts: via_brief, from_kind / to_kind "
    "'brief_t'), lead_in (track start -> first T), into_counterbalance / out_of_counterbalance (T -> where a "
    "counterbalance pivot begins / where it ends -> T; its dial owns only the circling inside it), lead_out. from_t / "
    "to_t are null at a non-T end (from_kind / to_kind say what it is).",
    "dials: repeated turning about one point - kind 'counterbalance' (the counterbalance pivots: centre = the "
    "follower's anchored foot, degrees = the leader's swept angle) or consecutive same-sign axis pivots summing to >= "
    "params.dial.min_deg while one point stays put (p90 <= place_m): 'leader_spin' (his chest), 'couple_turn' (the "
    "combined COM), 'leader_around_follower' (her chest) - the last two only when his orbit about the centre agrees "
    "with the turn (orbit_ratio). degrees signed (+ = ccw, can exceed 360), radius_m / arc_start_deg for drawing the "
    "needle. params.turn_groups lists every chain tried with the centre spreads, orbits and rejections.",
    "balance: per frame and dancer, the balance axis chest (spine3) -> support (the nearest point of the STANCE "
    "foot's heel-toe line to the COM; stance = the foot in contact nearest the COM: 1 left, 2 right, 0 none); "
    "com_offset_m = horizontal COM -> support; tilt_deg = chest -> support line from vertical; score 0..1 = "
    "min(offset ramp params.balance.good_offset_m..bad_offset_m, tilt ramp good_tilt_deg..bad_tilt_deg) = the line's "
    "brightness. couple: inside counterbalance pivots only, the combined COM over the follower's anchored foot.",
    "evaluation (axis_pivots, dials, pivots, rotations): share of the turn's frames (window: the turn, widened to >= "
    "params.balance.eval_min_beats) with score >= aligned_score -> verdict good (>= good_share) / ok (>= ok_share) / "
    "needs work; subject lead (his pivots, spins), couple (counterbalance: combined COM over her anchored foot) or "
    "both (couple turns: the worse dancer). Reorientations carry turn_kind (one_foot | both_feet | stepped | "
    "airborne; feet = the contact numbers): only a pivot on ONE foot is graded (graded true; over the frames with "
    "the weight on that foot) - otherwise verdict 'not graded' (or 'unknown'), balance_verdict = the COM test alone; "
    "show no verdict icon for them. Dials are always graded. ESTIMATES from the fitted bodies: label them so.",
]


def grouped_params(cfg: FCConfig) -> dict:
    """The version-3 thresholds grouped for the viewer (params.t_axis / dial / balance; also flat in params)."""
    c = asdict(cfg)
    return {"t_axis": {"half_width_m": cfg.t_half_width_m, "forward_m": cfg.t_forward_m, "side_m": cfg.t_side_m,
                       "pivot_deg": cfg.t_pivot_deg, "sustain_beats": cfg.t_sustain_beats,
                       "form_beats": cfg.t_form_beats, "form_yaw_deg": cfg.t_form_yaw_deg,
                       "form_drift_m_s": cfg.t_form_drift_m_s,
                       "same_place_m": cfg.t_same_place_m, "pivot_edge_deg": cfg.t_pivot_edge_deg,
                       "draw_forward_m": cfg.t_draw_forward_m, "draw_side_m": cfg.t_draw_side_m,
                       "min_hold_beats": cfg.t_min_hold_beats, "refine_iter": cfg.t_refine_iter},
            "dial": {"min_deg": cfg.dial_min_deg, "place_m": cfg.dial_place_m,
                     "max_hold_beats": cfg.dial_max_hold_beats, "orbit_ratio": cfg.dial_orbit_ratio},
            "turn": {"double_share": cfg.turn_double_share, "pivot_contact": cfg.turn_pivot_contact,
                     "pivot_stance": cfg.turn_pivot_stance, "pivot_travel_m": cfg.turn_pivot_travel_m},
            "balance": {k[4:]: c[k] for k in c if k.startswith("bal_")} | {"com_cutoff_hz": cfg.com_cutoff_hz}}


def to_json(res: dict, take: str, cfg: FCConfig, source: dict) -> dict:
    return cb._clean({
        "version": VERSION, "revision": REVISION, "kind": "floorcraft", "take": take, "frame": "world",
        "time_base": "reference",
        "coverage": res["info"]["coverage"],
        "params": {**asdict(cfg), **grouped_params(cfg), **{k: v for k, v in res["info"].items() if k != "coverage"}},
        "axis": {"long_m": cfg.long_axis_m, "short_m": cfg.short_axis_m, "origin": "spine3 on the floor",
                 "source": res["info"]["axis_source"]},
        "source": source,
        "t_axes": res["t_axes"], "transitions": res["transitions"], "axis_pivots": res["axis_pivots"],
        "dials": res["dials"],
        "plants": res["plants"], "moves": res["moves"], "rotations": res["rotations"],
        "pivots": res["pivots"], "counterbalances": res["counterbalances"], "events": res["events"],
        "track": res["track"], "balance": res["balance"], "notes": NOTES,
    })

def code_sha1() -> str:
    """sha1 of this module's source: a changed definition / threshold makes every floorcraft.json stale."""
    return cb._sha1(Path(__file__))


def dumps(o, ind: int = 0) -> str:
    """JSON with one key per line but every array of scalars / of scalar arrays (points, per-frame tracks) on ONE
    line - the per-frame balance tracks would otherwise take a line per number."""
    sp = " " * ind
    if isinstance(o, dict):
        if not o:
            return "{}"
        return "{\n" + ",\n".join(f"{sp} {json.dumps(str(k))}: {dumps(v, ind + 1)}" for k, v in o.items()) + \
            "\n" + sp + "}"
    if isinstance(o, list):
        flat = all(not isinstance(v, (dict, list)) or (isinstance(v, list) and
                                                       all(not isinstance(x, (dict, list)) for x in v)) for v in o)
        if flat:
            return json.dumps(o, separators=(",", ":"))
        return "[\n" + ",\n".join(f"{sp} {dumps(v, ind + 1)}" for v in o) + "\n" + sp + "]"
    return json.dumps(o)


def take_masses(take: str) -> dict:
    """{role: kg} from takes/<take>.toml [dancers.<role>] mass_kg (as counterbalance.run)."""
    dancers = cb.take_cfg(take).get("dancers", {})
    return {r: float(dancers[r]["mass_kg"]) for r in cb.ROLES if dancers.get(r, {}).get("mass_kg")}


def run(take: str, smplx_dir: str | Path | None = None, cfg: FCConfig | None = None, review: bool = True,
        out_path: Path | None = None, review_dir: Path | None = None) -> dict:
    """Analyse a take -> work/<take>/floorcraft/floorcraft.json (+ review images in review_dir, default
    work/<take>/review/)."""
    cfg = cfg or FCConfig()
    work = cb.work_dir(take)
    d = cb.find_smplx_dir(take, smplx_dir)
    npzs = cb.load_npzs(d)
    npz = npzs["lead"]
    grid = cb.load_grid(work)
    beat, _, _ = cb.beats_from_grid(grid)
    sha = cb._sha1(d / "lead.npz")
    cbdoc, cbnote = ensure_counterbalance(take, d, sha)
    cbp = work / "counterbalance" / "counterbalance.json"
    masses = take_masses(take)
    coms: dict = {}
    for r in cb.ROLES:
        c, m, s = dancer_com(npzs[r], masses.get(r))
        coms[r], coms[f"{r}_source"] = c, s
        if r not in masses and m:
            masses[r] = m
    res = analyse(npz, beat, cfg, cbdoc, follow=npzs["follow"], coms=coms, masses=masses)
    source = {"smplx_dir": str(d.relative_to(PACKAGE_ROOT)) if d.is_relative_to(PACKAGE_ROOT) else str(d),
              "world_frame": cb.world_frame_note(d), "lead_sha1": sha, "follow_sha1": cb._sha1(d / "follow.npz"),
              "lead_probe": cb.lead_probe(npz), "t_range": _r([npz["t"][0], npz["t"][-1]], 4),
              "frames": int(len(npz["t"])),
              "beats": "take grid" if beat else f"none (default {cfg.default_bpm} BPM)",
              "beats_sha1": cb.grid_sha1(work),
              "counterbalance": cbnote,
              "counterbalance_sha1": cb._sha1(cbp) if cbdoc is not None and cbp.exists() else None,
              "masses_kg": {r: round(float(m), 2) for r, m in masses.items()},
              "code_sha1": code_sha1()}
    doc = to_json(res, take, cfg, source)
    out_path = out_path or work / "floorcraft" / "floorcraft.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dumps(doc))
    res["_json"], res["_path"] = doc, out_path
    if review:
        rd = Path(review_dir) if review_dir else work / "review"
        t_hud0 = float(npz["t"][0])
        res["_review"] = [plot_map(doc, res, rd / "floorcraft_map.png", take, t_hud0),
                          plot_timeline(doc, res, rd / "floorcraft_timeline.png", take, t_hud0, grid),
                          plot_taxis_map(doc, res, rd / "taxis_map.png", take, t_hud0),
                          plot_taxis_timeline(doc, res, rd / "taxis_timeline.png", take, t_hud0, grid,
                                              (cb.load_truth(take) or {}).get("counterbalance"))]
    return res


def is_current(take: str, smplx_dir: str | Path | None = None) -> bool:
    """True when work/<take>/floorcraft/floorcraft.json was computed on the current fits (lead and follow sha1), beat
    grid (beats_sha1), counterbalance.json (its sha1) and this module's code (code_sha1), so a re-export needs no
    re-analysis."""
    p = cb.work_dir(take) / "floorcraft" / "floorcraft.json"
    if not p.exists():
        return False
    doc = json.loads(p.read_text())
    src = doc.get("source", {})
    d = cb.find_smplx_dir(take, smplx_dir)
    cbp = cb.work_dir(take) / "counterbalance" / "counterbalance.json"
    return (doc.get("version") == VERSION
            and src.get("lead_sha1") == cb._sha1(d / "lead.npz")
            and src.get("follow_sha1") == cb._sha1(d / "follow.npz")
            and src.get("beats_sha1") == cb.grid_sha1(cb.work_dir(take))
            and src.get("counterbalance_sha1") == (cb._sha1(cbp) if cbp.exists() else None)
            and src.get("code_sha1") == code_sha1())

# ----------------------------------------------------------------------------------------------- Unity copy

def unity_euler_y(yaw_deg):
    """Unity Euler Y (deg) of a floor yaw: Quaternion.Euler(0, y, 0) * Vector3.forward = the forward axis in Unity
    coordinates (x, -z). World yaw psi (from +x towards -z) -> y = 90 - psi, in [-180, 180)."""
    return None if yaw_deg is None else round(float(wrap180(90.0 - float(yaw_deg))), 2)


def unity_doc(doc: dict, time_to_audio: float, capture: str, motion_check: str) -> dict:
    """floorcraft.json in the capture's time base (capture = reference + time_to_audio) and Unity coordinates
    (x, y, -z); every converted time keeps its reference value in <key>_reference."""
    from .export_unity import to_unity_points
    tta = float(time_to_audio)
    U = lambda p: None if p is None else _r(to_unity_points(np.asarray(p, float)), 4)   # noqa: E731
    T = lambda x: None if x is None else _r(np.asarray(x, float) + tta, 4)              # noqa: E731

    def times(d: dict, keys=("t0", "t1")) -> None:
        for k in keys:
            if d.get(k) is not None:
                d[f"{k}_reference"] = d[k]
                d[k] = round(d[k] + tta, 4)

    def path(p: dict | None) -> dict | None:
        return None if not p else {"t": T(p["t"]), "pos": U(p["pos"])}

    out = json.loads(json.dumps(doc))
    out.update({"time_base": "capture", "frame": "unity", "capture": capture, "time_to_audio": tta,
                "time_offset_s": tta, "motion_check": motion_check,
                "coverage_reference": doc.get("coverage"), "coverage": [T(c) for c in doc.get("coverage", [])]})
    for p in out["plants"]:
        times(p)
        p["origin"], p["forward"], p["long_axis"] = U(p["origin"]), U(p["forward"]), U(p["long_axis"])
        for k in ("yaw_deg", "yaw_mean_deg", "yaw_start_deg", "yaw_end_deg"):
            p[k.replace("yaw", "euler_y")] = unity_euler_y(p.get(k))
        if p.get("held"):
            p["held_reference"] = p["held"]
            p["held"] = T(p["held"])
        p["turn_deg_unity_euler_y"] = -p["turn_deg"]
    for m in out["moves"]:
        times(m)
        m["from"], m["to"], m["path"] = U(m["from"]), U(m["to"]), path(m["path"])
        m["heading_euler_y_deg"] = unity_euler_y(m["heading_deg"])
        m["from_euler_y_deg"], m["to_euler_y_deg"] = unity_euler_y(m["from_yaw_deg"]), unity_euler_y(m["to_yaw_deg"])
        m["yaw_change_deg_unity_euler_y"] = -m["yaw_change_deg"]
    for r in out["rotations"]:
        times(r)
        r["centre"] = U(r["centre"])
        r["from_euler_y_deg"], r["to_euler_y_deg"] = unity_euler_y(r["from_yaw_deg"]), unity_euler_y(r["to_yaw_deg"])
        r["signed_deg_unity_euler_y"] = -r["signed_deg"]
    for pv in out["pivots"]:
        times(pv)
        pv["counterbalance_t_reference"] = pv["counterbalance_t"]
        pv["counterbalance_t"] = T(pv["counterbalance_t"])
        pv["pivot"] = U(pv["pivot"])
        pv["leader_path"] = path(pv.get("leader_path"))
        la = pv.get("leader_axis")
        if la:
            times(la)
            la["path"] = path(la.get("path"))
            la["from_euler_y_deg"] = unity_euler_y(la["from_yaw_deg"])
            la["to_euler_y_deg"] = unity_euler_y(la["to_yaw_deg"])
            la["turn_deg_unity_euler_y"] = -la["turn_deg"]
        for tk in pv.get("beat_ticks") or []:
            tk["t"], tk["pos"] = round(tk["t"] + tta, 4), U(tk["pos"])
        sw = pv.get("swept_signed_deg")
        pv["swept_signed_deg_unity_up"] = None if sw is None else -sw
        a = pv.get("arc_start_deg")
        pv["arc_start_deg_unity"] = None if a is None else round(float(wrap180(a)), 2)
    for c in out["counterbalances"]:
        times(c)
        c["com_floor"] = U(c["com_floor"])
    for e in out["events"]:
        times(e)
    tr = out["track"]
    tr["t"], tr["origin"] = T(tr["t"]), U(tr["origin"])
    tr["euler_y_deg"] = _r(90.0 - np.asarray(tr["yaw_deg"], float), 2)   # unwrapped, like yaw_deg

    def window(rec: dict) -> None:
        ev = rec.get("evaluation")
        if ev and ev.get("window"):
            ev["window_reference"] = ev["window"]
            ev["window"] = T(ev["window"])

    for r in out["rotations"] + out["pivots"]:
        window(r)
    for ta in out.get("t_axes", []):
        times(ta, ("t0", "t1", "t_formed"))
        ta["origin"], ta["forward"], ta["parallel"] = U(ta["origin"]), U(ta["forward"]), U(ta["parallel"])
        ta["euler_y_deg"] = unity_euler_y(ta["yaw_deg"])
        ta["euler_y_unwrapped_deg"] = round(90.0 - ta["yaw_unwrapped_deg"], 2)
    for x in out.get("transitions", []):
        times(x)
        x["from"], x["to"], x["path"] = U(x["from"]), U(x["to"]), path(x.get("path"))
    for ap in out.get("axis_pivots", []):
        times(ap)
        ap["at"] = U(ap["at"])
        ap["from_euler_y_deg"] = unity_euler_y(ap["from_yaw_deg"])
        ap["to_euler_y_deg"] = unity_euler_y(ap["to_yaw_deg"])
        ap["signed_deg_unity_euler_y"] = -ap["signed_deg"]
        window(ap)
    for d in out.get("dials", []):
        times(d)
        d["centre"] = U(d["centre"])
        d["degrees_unity_euler_y"] = -d["degrees"]
        d["arc_start_deg_unity"] = d["arc_start_deg"]       # angle from +x towards Unity +z = world -z: same number
        window(d)
    bal = out.get("balance")
    if bal:
        UL = lambda pts: [None if p is None else U(p) for p in pts]   # noqa: E731
        bal["t_reference"] = bal["t"]
        bal["t"] = T(bal["t"])
        for role in ("lead", "follow"):
            if role in bal:
                b = bal[role]
                b["support"], b["chest"], b["com"] = UL(b["support"]), U(b["chest"]), U(b["com"])
        if "couple" in bal:
            bal["couple"]["com"], bal["couple"]["support"] = UL(bal["couple"]["com"]), UL(bal["couple"]["support"])
    out["notes"] = out["notes"] + [
        f"UNITY COPY of work/{doc['take']}/floorcraft/floorcraft.json: positions and vectors (x, y, -z) in the capture's "
        "Unity frame BEFORE the viewer's dance-origin offset (capture.json 'origin', VIEWER_SPEC 3.0) - apply it like "
        "every other capture position; times = capture time = reference + time_to_audio (capture.json; the clock of "
        "CaptureTimeline.AudioTimes, as counterbalance.json and moves/*.json); *_reference keep the reference seconds.",
        "Angles: *_euler_y_deg = Unity Euler Y of that yaw (Quaternion.Euler(0, y, 0) * Vector3.forward = the forward "
        "axis; = 90 - yaw_deg); Euler Y grows clockwise seen from above, so signed changes flip sign "
        "(*_unity_euler_y = -signed). direction 'ccw' / 'cw' is as seen from above and unchanged by the mirror. "
        "track.euler_y_deg is unwrapped like track.yaw_deg. Pivots keep counterbalance.json's Unity conventions "
        "(swept_signed_deg_unity_up, arc_start_deg_unity: angle in the Unity floor plane from +x towards +z).",
    ]
    return out


def export(take: str, capture: str | None = None, streaming: Path = cb.UNITY_STREAMING, force: bool = False,
           manifest: bool = True, src: Path | None = None, mirror: bool = True) -> dict:
    """Write <streaming>/<capture>/floorcraft.json (+ work/<take>/export/<capture>/ when present and mirror).
    Refuses (ValueError) when the analysed range does not overlap the capture or the capture's lead motion differs
    from the fits the analysis ran on (force: write anyway, motion_check 'MISMATCH (forced)')."""
    capture = capture or DEFAULT_CAPTURES.get(take, take)
    src = Path(src) if src else cb.work_dir(take) / "floorcraft" / "floorcraft.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: run `python -m dancecap.floorcraft {take}`")
    doc = json.loads(src.read_text())
    cov = doc.get("coverage") or [doc["source"]["t_range"]]
    report = {"capture": capture, "written": []}
    dirs = [Path(streaming) / capture] + ([cb.work_dir(take) / "export" / capture] if mirror else [])
    for cap_dir in dirs:
        if not (cap_dir / "capture.json").exists():
            continue
        clock = cb.capture_clock(cap_dir)
        rr = clock.range_reference
        if rr is not None and cb._overlap_s(cov, rr[0], rr[1]) <= 0:
            raise ValueError(f"{cap_dir.name}: the analysed range {cov} (reference s) does not overlap the capture's "
                             f"frames {list(rr)} - wrong take or window")
        try:
            mc = cb.check_capture_motion(cap_dir, doc)
        except ValueError:
            if not force:
                raise
            mc = "MISMATCH (forced)"
        out = unity_doc(doc, clock.time_to_audio, capture, mc)
        out["time_offset_source"] = "capture.json time_to_audio (moves.labels.CaptureClock = CaptureTimeline.AudioTimes)"
        out["capture_range_reference"] = None if rr is None else [round(rr[0], 4), round(rr[1], 4)]
        out["capture_range"] = None if rr is None else [round(clock.to_capture(rr[0]), 4),
                                                         round(clock.to_capture(rr[1]), 4)]
        (cap_dir / "floorcraft.json").write_text(dumps(out))
        report["written"].append(str(cap_dir / "floorcraft.json"))
        report["motion_check"] = mc
        report["time_to_audio"] = clock.time_to_audio
        cj = cap_dir / "capture.json"
        if manifest:   # additive: tell the viewer the layer exists (capture.json v4 key)
            m = json.loads(cj.read_text())
            if m.get("floorcraft") != "floorcraft.json":
                m["floorcraft"] = "floorcraft.json"
                cj.write_text(json.dumps(m, indent=2))
    if not report["written"]:
        raise FileNotFoundError(f"capture {capture} not found in {streaming} or work/{take}/export/")
    return report


# ----------------------------------------------------------------------------------------------- review

INK, MUTED, GRID, TEAL, CB_C = "#0b0b0b", "#8a8984", "#e6e5e0", "#19c3d6", "#eda100"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]   # sequential blue, steps 250 -> 700 (light = early, dark = late)


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("time", BLUE_RAMP)


def _draw_map(ax, doc: dict, res: dict, t_hud0: float, norm, cmap, lim: tuple, window: tuple | None = None,
              small: bool = False) -> None:
    """One overhead map into ax: the records whose [t0, t1] overlaps window (reference s; None = all)."""
    from matplotlib.patches import Circle, FancyArrowPatch
    full = res["_full"]
    tt, oo = full["t"], full["origin"]
    col = lambda t_ref: cmap(norm(t_ref - t_hud0))   # noqa: E731
    sel = (lambda r: True) if window is None else (lambda r: r["t1"] >= window[0] and r["t0"] <= window[1])  # noqa: E731
    fs = 0.85 if small else 1.0
    x0, x1, y0, y1 = lim
    for gx in range(math.floor(x0), math.ceil(x1) + 1):          # the viewer's 1 m teal crosses
        for gy in range(math.floor(y0), math.ceil(y1) + 1):
            ax.plot([gx - 0.04, gx + 0.04], [gy, gy], color=TEAL, lw=1.2, zorder=0)
            ax.plot([gx, gx], [gy - 0.04, gy + 0.04], color=TEAL, lw=1.2, zorder=0)
    ax.plot(oo[:, 0], -oo[:, 1], color=GRID, lw=0.8, zorder=1)       # view from above: x right, -z up
    if window is not None:
        k = (tt >= window[0]) & (tt <= window[1])
        ax.plot(oo[k, 0], -oo[k, 1], color=MUTED, lw=0.6, alpha=0.6, zorder=1)
    for m in filter(sel, doc["moves"]):
        P = np.asarray(m["path"]["pos"], float)
        if len(P) < 2:
            continue
        c = col(0.5 * (m["t0"] + m["t1"]))
        ax.plot(P[:, 0], -P[:, 2], color=c, lw=2.0 if m.get("kind") != "step" else 1.4, zorder=2,
                solid_capstyle="round")
        if np.linalg.norm(P[-1] - P[-2]) > 1e-4:
            ax.add_patch(FancyArrowPatch((P[-2, 0], -P[-2, 2]), (P[-1, 0], -P[-1, 2]), arrowstyle="-|>",
                                         mutation_scale=11, color=c, lw=0, zorder=3))
    for p in filter(sel, doc["plants"]):
        c = col(0.5 * (p["t0"] + p["t1"]))
        o = np.array(p["origin"])
        L = np.array(p["long_axis"]) * 0.30
        F = np.array(p["forward"]) * 0.22
        ax.add_patch(Circle((o[0], -o[2]), p["radius_m"], fill=False, color=c, lw=0.6, alpha=0.5, zorder=3))
        for lw_, c_ in ((5, "white"), (2.6, c)):
            ax.plot([o[0] - L[0], o[0] + L[0]], [-o[2] + L[2], -o[2] - L[2]], color=c_, lw=lw_, zorder=4,
                    solid_capstyle="round")
        ax.add_patch(FancyArrowPatch((o[0], -o[2]), (o[0] + F[0], -o[2] - F[2]), arrowstyle="-|>",
                                     mutation_scale=13, color=c, lw=2.2, zorder=5))
        ax.text(o[0] - F[0] * 0.9, -o[2] + F[2] * 0.9, f"P{p['id']}", fontsize=7.5 * fs, color=INK, ha="center",
                va="center", zorder=6)
    n_in_plant: dict = {}
    for r in filter(sel, doc["rotations"]):
        c = col(0.5 * (r["t0"] + r["t1"]))
        key = r["plant"] if r["context"] == "plant" else None
        kk = n_in_plant.get(key, 0) if key is not None else 0
        if key is not None:
            n_in_plant[key] = kk + 1
        ce = np.array(r["centre"])
        rad = 0.12 + 0.045 * kk
        a0, sw = r["from_yaw_deg"], r["signed_deg"]
        aa = np.radians(a0 + np.linspace(0, sw, 60))
        ax.plot(ce[0] + rad * np.cos(aa), -ce[2] + rad * np.sin(aa), color=c, lw=1.6,
                ls="-" if r["context"] == "plant" else (0, (2, 1.5)), zorder=6)
        ax.add_patch(FancyArrowPatch((ce[0] + rad * np.cos(aa[-4]), -ce[2] + rad * np.sin(aa[-4])),
                                     (ce[0] + rad * np.cos(aa[-1]), -ce[2] + rad * np.sin(aa[-1])),
                                     arrowstyle="-|>", mutation_scale=8, color=c, lw=0, zorder=6))
        am = aa[len(aa) // 2]
        ax.text(ce[0] + (rad + 0.05) * np.cos(am), -ce[2] + (rad + 0.05) * np.sin(am), f"{sw:+.0f}°",
                fontsize=6.5 * fs, color=INK, ha="center", va="center", zorder=7)
    for pv in filter(sel, doc["pivots"]):
        px, pz = pv["pivot"][0], pv["pivot"][2]
        ax.add_patch(Circle((px, -pz), 0.08, fill=False, color=CB_C, lw=3, zorder=7))
        ax.add_patch(Circle((px, -pz), 0.08, fill=False, color=INK, lw=0.6, zorder=8))
        r0 = 1.08 * pv["leader_radius_m"]
        a0, sw = math.radians(pv["arc_start_deg"]), math.radians(pv["swept_signed_deg"])
        aa = a0 + np.linspace(0, sw, 300)
        rr = r0 * (1 + 0.05 * np.linspace(0, 1, 300) * abs(sw) / (2 * math.pi))     # spiral out per turn
        ax.plot(px + rr * np.cos(aa), -pz + rr * np.sin(aa), color=CB_C, lw=3.0, alpha=0.9, zorder=7)
        ax.add_patch(FancyArrowPatch((px + rr[-6] * np.cos(aa[-6]), -pz + rr[-6] * np.sin(aa[-6])),
                                     (px + rr[-1] * np.cos(aa[-1]), -pz + rr[-1] * np.sin(aa[-1])),
                                     arrowstyle="-|>", mutation_scale=12, color=INK, lw=0, zorder=8))
        for tk in pv.get("beat_ticks", []):
            ax.plot(tk["pos"][0], -tk["pos"][2], "|", color=INK, ms=8, mew=1.2, zorder=8)
        ax.text(px, -pz - rr.max() - 0.05, f"pivot {pv['id']}: {pv['swept_deg']:.0f}° {pv['direction']}, "
                f"{pv['t0'] - t_hud0:.2f}-{pv['t1'] - t_hud0:.2f} s", fontsize=8 * fs, ha="center", va="top",
                color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=CB_C, lw=1), zorder=9)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor("#fcfcfb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_map(doc: dict, res: dict, path: Path, take: str, t_hud0: float) -> Path:
    """Overhead floor map of the whole take (top) and of three time slices (bottom): plants = axis crosses (chest
    line + forward arrow), moves = paths with an arrowhead, rotations = arcs with their signed angle, pivots = yellow
    rings + the leader's circling arc; time by the blue ramp (light = early)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    full = res["_full"]
    tt, oo = full["t"], full["origin"]
    norm = Normalize(tt[0] - t_hud0, tt[-1] - t_hud0)
    cmap = _cmap()
    pad = 0.45
    xs = [float(oo[:, 0].min()), float(oo[:, 0].max())]
    ys = [float(-oo[:, 1].max()), float(-oo[:, 1].min())]
    for pv in doc["pivots"]:
        r = 1.08 * pv["leader_radius_m"] * (1 + 0.05 * abs(pv["swept_signed_deg"]) / 360) + 0.35
        xs += [pv["pivot"][0] - r, pv["pivot"][0] + r]
        ys += [-pv["pivot"][2] - r, -pv["pivot"][2] + r]
    lim = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
    fig = plt.figure(figsize=(16, 18.5), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1], hspace=0.1, wspace=0.06)
    ax = fig.add_subplot(gs[0, :])
    _draw_map(ax, doc, res, t_hud0, norm, cmap, lim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("-z (m)   [view from above = Unity z]")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label(f"time from the first frame (s) = viewer HUD   [reference = HUD + {t_hud0:.3f}]")
    handles = [Line2D([], [], color=BLUE_RAMP[5], lw=2.6, label="plant: chest line + forward arrow (circle = radius)"),
               Line2D([], [], color=BLUE_RAMP[5], lw=2.0, label="move: plant -> axis origin path -> plant"),
               Line2D([], [], color=BLUE_RAMP[5], lw=1.4, label="step: one step to the next stand"),
               Line2D([], [], color=BLUE_RAMP[5], lw=1.6, label="rotation >= 25° in a plant (dotted: in a move)"),
               Line2D([], [], color=CB_C, lw=3.0, label="counterbalance pivot: ring + leader arc, beat ticks "
                      "(owns his motion: no move / rotation inside)"),
               Line2D([], [], color=GRID, lw=1.0, label="axis origin (smoothed chest on the floor)"),
               Line2D([], [], color=TEAL, lw=1.2, label="1 m floor crosses")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.92)
    c = doc["params"]["counts"]
    ax.set_title(f"{take}: leader floor axis record - {c['plants']} plants ({doc['params']['plant_share']:.0%} of "
                 f"the time), {c['moves']} moves ({c.get('steps', 0)} single steps), {c['rotations']} rotations, "
                 f"{c['pivots']} pivot(s)", fontsize=11, loc="left")
    edges = np.linspace(tt[0], tt[-1], 4)
    for n in range(3):
        a = fig.add_subplot(gs[1, n])
        w = (edges[n], edges[n + 1])
        k = (tt >= w[0]) & (tt <= w[1])
        px = [float(oo[k, 0].min()), float(oo[k, 0].max())]
        py = [float(-oo[k, 1].max()), float(-oo[k, 1].min())]
        for pv in doc["pivots"]:
            if pv["t1"] >= w[0] and pv["t0"] <= w[1]:
                r = 1.08 * pv["leader_radius_m"] * (1 + 0.05 * abs(pv["swept_signed_deg"]) / 360) + 0.35
                px += [pv["pivot"][0] - r, pv["pivot"][0] + r]
                py += [-pv["pivot"][2] - r, -pv["pivot"][2] + r]
        cx, cy = 0.5 * (min(px) + max(px)), 0.5 * (min(py) + max(py))
        h = 0.5 * max(max(px) - min(px), max(py) - min(py), 1.2) + 0.3
        _draw_map(a, doc, res, t_hud0, norm, cmap, (cx - h, cx + h, cy - h, cy + h), w, small=True)
        a.set_title(f"HUD {edges[n] - t_hud0:.1f}-{edges[n + 1] - t_hud0:.1f} s (records overlapping it)",
                    fontsize=9, loc="left")
        a.tick_params(labelsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_timeline(doc: dict, res: dict, path: Path, take: str, t_hud0: float, grid=None) -> Path:
    """The segmentation over time: record lanes (plants, moves, rotations, counterbalance / pivots), the axis origin
    speed and the axis yaw, on the beat grid (x = time from the first frame = the viewer HUD)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = res["_full"]
    tt = full["t"] - t_hud0
    _, beat_t, meas_t = cb.beats_from_grid(grid)
    fig, axs = plt.subplots(3, 1, figsize=(18, 9.5), dpi=100, sharex=True,
                            gridspec_kw=dict(height_ratios=[1.3, 1, 1.3], hspace=0.08))
    cfg = doc["params"]

    def grid_lines(a):
        if beat_t is not None:
            for b in beat_t[(beat_t >= full["t"][0]) & (beat_t <= full["t"][-1])]:
                a.axvline(b - t_hud0, color="#f0efec", lw=0.8, zorder=0)
        if meas_t is not None:
            for b in meas_t[(meas_t >= full["t"][0]) & (meas_t <= full["t"][-1])]:
                a.axvline(b - t_hud0, color="#d6d5cf", lw=1.0, zorder=0)

    a = axs[0]
    grid_lines(a)
    lanes = {"plants": 3, "moves": 2, "rotations": 1, "pivots": 0}
    for p in doc["plants"]:
        a.barh(3, p["t1"] - p["t0"], left=p["t0"] - t_hud0, height=0.6, color=BLUE_RAMP[4], edgecolor="white", lw=1)
        a.text(0.5 * (p["t0"] + p["t1"]) - t_hud0, 3, f"P{p['id']}", fontsize=6.5, color="white", ha="center",
               va="center")
    for m in doc["moves"]:
        if m["t1"] - m["t0"] < 0.15:            # a step at the junction of two touching plants
            x = 0.5 * (m["t0"] + m["t1"]) - t_hud0
            a.plot([x, x], [1.72, 2.28], color=BLUE_RAMP[7], lw=2.0, solid_capstyle="butt")
            continue
        a.barh(2, m["t1"] - m["t0"], left=m["t0"] - t_hud0, height=0.6, edgecolor="white", lw=1,
               color=BLUE_RAMP[4] if m.get("kind") == "step" else BLUE_RAMP[1])
        if m["displacement_m"] >= 0.3:
            a.text(0.5 * (m["t0"] + m["t1"]) - t_hud0, 2, f"{m['displacement_m']:.1f}m", fontsize=6, color=INK,
                   ha="center", va="center")
    for r in doc["rotations"]:
        c = "#eb6834" if r["direction"] == "ccw" else "#4a3aa7"
        a.barh(1, r["t1"] - r["t0"], left=r["t0"] - t_hud0, height=0.6, color=c, edgecolor="white", lw=1,
               alpha=0.55 if r["context"] == "move" else 1.0)
        a.text(0.5 * (r["t0"] + r["t1"]) - t_hud0, 1.42, f"{r['signed_deg']:+.0f}", fontsize=6, color=INK,
               ha="center", va="bottom")
    for c in doc["counterbalances"]:
        a.barh(0, c["t1"] - c["t0"], left=c["t0"] - t_hud0, height=0.6, color=CB_C, alpha=0.45, edgecolor="none")
    for pv in doc["pivots"]:
        a.barh(0, pv["t1"] - pv["t0"], left=pv["t0"] - t_hud0, height=0.36, color=CB_C, edgecolor=INK, lw=0.6)
        a.text(0.5 * (pv["t0"] + pv["t1"]) - t_hud0, 0, f"pivot {pv['swept_deg']:.0f}° {pv['direction']}", fontsize=7,
               ha="center", va="center", color=INK)
    a.set_yticks(list(lanes.values()))
    a.set_yticklabels(["plants", "moves\n(dark = single step)", "rotations\n(orange ccw, violet cw;\nfaint = in a move)",
                       "counterbalance\n/ pivot"], fontsize=8)
    a.set_ylim(-0.6, 3.6)
    a.set_title(f"{take}: floor craft segmentation (plants: origin within {cfg['plant_radius_m']} m, drift <= "
                f"{cfg['plant_drift_m_s']} m/s, >= {cfg['plant_min_beats']} beat; rotations >= {cfg['rotation_deg']}°)",
                fontsize=10, loc="left")
    a = axs[1]
    grid_lines(a)
    for p in doc["plants"]:
        a.axvspan(p["t0"] - t_hud0, p["t1"] - t_hud0, color=BLUE_RAMP[0], alpha=0.35, lw=0)
    a.plot(tt, full["speed"], color=INK, lw=1.2)
    a.axhline(cfg["plant_drift_m_s"], color=MUTED, lw=0.8, ls="--")
    a.set_ylabel("axis origin\nspeed (m/s)", fontsize=9)
    a.set_ylim(0, max(1.0, float(np.percentile(full["speed"], 99)) * 1.1))
    a = axs[2]
    grid_lines(a)
    for r in doc["rotations"]:
        a.axvspan(r["t0"] - t_hud0, r["t1"] - t_hud0, color="#eb6834" if r["direction"] == "ccw" else "#4a3aa7",
                  alpha=0.18, lw=0)
    a.plot(tt, full["yaw"], color=INK, lw=1.2)
    a.set_ylabel("axis yaw (deg, unwrapped;\n+ = ccw from above)", fontsize=9)
    a.set_xlabel(f"time from the first frame (s) = viewer HUD   [reference = HUD + {t_hud0:.3f}]   grid: light = "
                 "beats, dark = measure starts")
    for a in axs:
        a.tick_params(labelsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


SIDE_C = "#eb6834"                            # crossbar offsets (timeline)
LEAD_C = "#d6452a"                            # the lead's red-orange (current T)
VERDICT_C = {"good": "#0ca30c", "ok": "#fab219", "needs work": "#d03b3b", "unknown": "#8a8984",   # status palette
             "not graded": "#b9b8b2"}
VERDICT_MARK = {"good": "o", "ok": "s", "needs work": "X", "unknown": ".", "not graded": "_"}   # never colour alone


def _crosses(ax, lim: tuple) -> None:
    """The viewer's 1 m floor crosses (small and faint, user 2026-10-07)."""
    x0, x1, y0, y1 = lim
    for gx in range(math.floor(x0), math.ceil(x1) + 1):
        for gy in range(math.floor(y0), math.ceil(y1) + 1):
            ax.plot([gx - 0.035, gx + 0.035], [gy, gy], color=TEAL, lw=0.7, alpha=0.45, zorder=0)
            ax.plot([gx, gx], [gy - 0.035, gy + 0.035], color=TEAL, lw=0.7, alpha=0.45, zorder=0)


def _draw_t(ax, ta: dict, current: bool, cfg: dict, label: bool = True, fs: float = 1.0) -> None:
    """One T on the overhead map (x right, -z up): crossbar along parallel, stem + arrowhead along forward."""
    from matplotlib.patches import FancyArrowPatch
    o, F, P = (np.array(ta[k], float) for k in ("origin", "forward", "parallel"))
    hs, fl = 0.5 * cfg["draw_side_m"], cfg["draw_forward_m"]
    c, lw, al = (LEAD_C, 2.6, 1.0) if current else (TEAL, 1.6, 0.35)
    ax.plot([o[0] - P[0] * hs, o[0] + P[0] * hs], [-o[2] + P[2] * hs, -o[2] - P[2] * hs], color=c, lw=lw, alpha=al,
            solid_capstyle="round", zorder=5 if current else 3)
    ax.add_patch(FancyArrowPatch((o[0] - F[0] * 0.15, -o[2] + F[2] * 0.15), (o[0] + F[0] * fl, -o[2] - F[2] * fl),
                                 arrowstyle="-|>", mutation_scale=12 if current else 9, color=c, lw=lw, alpha=al,
                                 zorder=5 if current else 3))
    if label:
        ax.text(o[0] - F[0] * 0.28, -o[2] + F[2] * 0.28, f"T{ta['id']}", fontsize=7.5 * fs, ha="center", va="center",
                color=INK if current else MUTED, zorder=7)


def _draw_dial(ax, d: dict, fs: float = 1.0, upto: float | None = None) -> None:
    """A dial: ring at the centre + the swept angle as an outward spiral, labelled with the degrees and verdict."""
    from matplotlib.patches import Circle, FancyArrowPatch
    cx, cz = d["centre"][0], d["centre"][2]
    frac = 1.0 if upto is None else float(np.clip((upto - d["t0"]) / max(d["t1"] - d["t0"], 1e-6), 0, 1))
    sw = math.radians(d["degrees"]) * frac
    r0 = max(0.18, min(0.45, d["radius_m"]))
    ax.add_patch(Circle((cx, -cz), 0.06, fill=False, color=CB_C, lw=2.4, zorder=6))
    if abs(sw) > 1e-3:
        aa = math.radians(d["arc_start_deg"]) + np.linspace(0, sw, 240)
        rr = r0 * (1 + 0.12 * np.linspace(0, 1, 240) * abs(sw) / (2 * math.pi))
        ax.plot(cx + rr * np.cos(aa), -cz + rr * np.sin(aa), color=CB_C, lw=2.2, alpha=0.95, zorder=6)
        ax.add_patch(FancyArrowPatch((cx + rr[-5] * np.cos(aa[-5]), -cz + rr[-5] * np.sin(aa[-5])),
                                     (cx + rr[-1] * np.cos(aa[-1]), -cz + rr[-1] * np.sin(aa[-1])),
                                     arrowstyle="-|>", mutation_scale=11, color=INK, lw=0, zorder=7))
    ev = d.get("evaluation") or {}
    v = ev.get("verdict", "unknown")
    ax.text(cx, -cz, f"{d['degrees'] * frac:+.0f}°", fontsize=8.5 * fs, ha="center", va="center", color=INK,
            fontweight="bold", zorder=8, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=CB_C, lw=0.8, alpha=0.9))
    if upto is None or frac >= 1.0:
        ax.text(cx, -cz - r0 * (1 + 0.12 * abs(sw) / (2 * math.pi)) - 0.06,
                f"dial {d['id']} {d['kind'].replace('_', ' ')}: {v}\n{ev.get('text', '')[len(v) + 2:]}",
                fontsize=6.6 * fs, ha="center", va="top", color=INK, zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=VERDICT_C[v], lw=1.2))


def _draw_axis_pivot(ax, ap: dict, k: int, fs: float = 1.0) -> None:
    from matplotlib.patches import FancyArrowPatch
    if ap.get("dial") is not None:
        return
    a = np.array(ap["at"], float)
    rad = 0.13 + 0.04 * k
    aa = np.radians(ap["from_yaw_deg"] + np.linspace(0, ap["signed_deg"], 60))
    v = (ap.get("evaluation") or {}).get("verdict", "unknown")
    ax.plot(a[0] + rad * np.cos(aa), -a[2] + rad * np.sin(aa), color=INK, lw=1.0, alpha=0.8, zorder=6)
    ax.add_patch(FancyArrowPatch((a[0] + rad * np.cos(aa[-4]), -a[2] + rad * np.sin(aa[-4])),
                                 (a[0] + rad * np.cos(aa[-1]), -a[2] + rad * np.sin(aa[-1])), arrowstyle="-|>",
                                 mutation_scale=7, color=INK, lw=0, zorder=6))
    am = aa[len(aa) // 2]
    px, py = a[0] + (rad + 0.07) * np.cos(am), -a[2] + (rad + 0.07) * np.sin(am)
    ax.plot(px - 0.055, py, VERDICT_MARK[v], color=VERDICT_C[v], ms=5.5 * fs, mec="white", mew=0.6, zorder=8)
    ax.text(px + 0.0, py, f"{ap['signed_deg']:+.0f}°", fontsize=6.5 * fs, ha="left", va="center", color=INK, zorder=8)


def _draw_taxis_state(ax, doc: dict, full_o: np.ndarray, full_t: np.ndarray, now: float, lim: tuple,
                      fs: float = 1.0, current_id: int | None = None) -> None:
    """The viewer's floor at time `now` (reference s): current T bright, older Ts faded teal, dotted transitions so
    far, axis pivots and dials so far (a running dial shows its degrees so far)."""
    cfg = doc["params"]["t_axis"]
    _crosses(ax, lim)
    k = full_t <= now
    ax.plot(full_o[:, 0], -full_o[:, 1], color=GRID, lw=0.7, zorder=1)
    ax.plot(full_o[k, 0], -full_o[k, 1], color=MUTED, lw=0.6, alpha=0.55, zorder=1)
    if k.any():
        ax.plot(full_o[k][-1, 0], -full_o[k][-1, 1], "o", color=LEAD_C, ms=4, zorder=9)
    for x in doc["transitions"]:
        if x["t0"] > now or x["same_place"]:
            continue
        a, b = np.array(x["from"]), np.array(x["to"])
        ax.plot([a[0], b[0]], [-a[2], -b[2]], ls=(0, (1.2, 2.2)), color=TEAL if x["t1"] < now else LEAD_C,
                lw=1.8, alpha=0.8, zorder=2, dash_capstyle="round")
    for ta in doc["t_axes"]:
        if ta["t0"] <= now:
            _draw_t(ax, ta, ta["t0"] <= now <= ta["t1"] or ta["id"] == current_id, cfg, fs=fs)
    seen: dict = {}
    for ap in doc["axis_pivots"]:
        if ap["t0"] <= now:
            key = tuple(np.round(ap["at"], 2))
            _draw_axis_pivot(ax, ap, seen.get(key, 0), fs)
            seen[key] = seen.get(key, 0) + 1
    for d in doc["dials"]:
        if d["t0"] <= now:
            _draw_dial(ax, d, fs, upto=now)
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_aspect("equal")
    ax.set_facecolor("#fcfcfb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_taxis_map(doc: dict, res: dict, path: Path, take: str, t_hud0: float) -> Path:
    """The T-axis record from above: the whole take (all Ts, the last one bright, older ones faded teal; dotted
    transitions; axis pivots with their angle and verdict mark; dials with degrees and verdict) and the viewer's floor
    at six instants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    full = res["_full"]
    tt, oo = full["t"], full["origin"]
    pad = 0.6
    xs = [float(oo[:, 0].min()), float(oo[:, 0].max())] + [d["centre"][0] for d in doc["dials"]]
    ys = [float(-oo[:, 1].max()), float(-oo[:, 1].min())] + [-d["centre"][2] for d in doc["dials"]]
    lim = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
    fig = plt.figure(figsize=(16, 21), dpi=100)
    gs = fig.add_gridspec(3, 3, height_ratios=[2.0, 1, 1], hspace=0.12, wspace=0.06)
    ax = fig.add_subplot(gs[0, :])
    _draw_taxis_state(ax, doc, oo, tt, float(tt[-1]), lim,
                      current_id=doc["t_axes"][-1]["id"] if doc["t_axes"] else None)
    for x in doc["transitions"]:
        if x["same_place"]:
            continue
        a, b = np.array(x["from"]), np.array(x["to"])
        m = 0.5 * (a + b)
        ax.text(m[0], -m[2], f"{x['distance_m']:.1f} m", fontsize=6.5, color=MUTED, ha="center", va="bottom")
    c = doc["params"]["counts"]
    ta_c = doc["params"]["t_axis"]
    ax.set_title(f"{take}: the leader's T axis - {c['t_axes']} Ts ({doc['params']['t_axis_share']:.0%} of the time "
                 f"on a T), {c['axis_pivots']} axis pivots (> {ta_c['pivot_deg']:.0f}°), {c['dials']} dial(s).  "
                 f"Corridors: ±{ta_c['half_width_m']:.2f} m wide, ±{ta_c['forward_m']:.1f} m long; sustain "
                 f"{ta_c['sustain_beats']:.1f} beat", fontsize=10.5, loc="left")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("-z (m)   [view from above = Unity z]")
    handles = [Line2D([], [], color=LEAD_C, lw=2.6, label="current T (here: the last one): crossbar = his chest line, "
                      "arrow = forward"),
               Line2D([], [], color=TEAL, lw=1.6, alpha=0.5, label="old Ts (faded teal)"),
               Line2D([], [], color=TEAL, lw=1.8, ls=(0, (1.2, 2.2)), label="transition T -> T (dotted)"),
               Line2D([], [], color=INK, lw=1.0, label="axis pivot: arc + signed angle (+ = ccw from above)"),
               Line2D([], [], color=CB_C, lw=2.2, label="dial: repeated turn about one point, degrees at the centre"),
               Line2D([], [], color=GRID, lw=1.0, label="leader's chest on the floor (smoothed)")]
    handles += [Line2D([], [], ls="", marker=VERDICT_MARK[v], color=VERDICT_C[v], ms=7, label=f"turn verdict: {v}")
                for v in ("good", "ok", "needs work")]
    handles.append(Line2D([], [], ls="", marker=VERDICT_MARK["not graded"], color=VERDICT_C["not graded"], ms=7,
                          label="not graded (turned on both feet / stepped: not a one-foot pivot)"))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.92)
    # six instants: the middle of six Ts spread over the take (or evenly)
    tas = doc["t_axes"]
    picks = [tas[int(round(k))] for k in np.linspace(0, len(tas) - 1, min(6, len(tas)))] if tas else []
    insts = [0.5 * (ta["t_formed"] + ta["t1"]) for ta in picks]
    for d in doc["dials"]:
        if len(insts) < 6 or d["kind"] == "counterbalance":
            insts.append(0.5 * (d["t0"] + d["t1"]))
    insts = sorted(set(round(x, 3) for x in insts))
    while len(insts) > 6:
        gaps = np.diff(insts)
        insts.pop(int(np.argmin(gaps)) + 1)
    for n, now in enumerate(insts[:6]):
        a = fig.add_subplot(gs[1 + n // 3, n % 3])
        _draw_taxis_state(a, doc, oo, tt, now, lim, fs=0.85)
        cur = [ta["id"] for ta in tas if ta["t0"] <= now <= ta["t1"]]
        run = [d["id"] for d in doc["dials"] if d["t0"] <= now <= d["t1"]]
        a.set_title(f"viewer at HUD {now - t_hud0:.1f} s: current " + (f"T{cur[0]}" if cur else "none (between Ts)")
                    + (f", dial {run[0]} running" if run else ""), fontsize=9, loc="left")
        a.tick_params(labelsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_taxis_timeline(doc: dict, res: dict, path: Path, take: str, t_hud0: float, grid=None,
                        truth: list | None = None) -> Path:
    """The T axis over time: record lanes (Ts, axis pivots with verdicts, dials / counterbalance + the user's label),
    the chest heading vs the held T heading, his offsets from the current T (with the corridor limits), and the
    balance-axis scores (lead, follow, the couple over her anchored foot)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = res["_full"]
    tt, oo, yy = full["t"], full["origin"], full["yaw"]
    x = tt - t_hud0
    _, beat_t, meas_t = cb.beats_from_grid(grid)
    cfg = doc["params"]["t_axis"]
    fig, axs = plt.subplots(5, 1, figsize=(18, 15), dpi=100, sharex=True,
                            gridspec_kw=dict(height_ratios=[1.4, 1.3, 1.0, 1.0, 1.1], hspace=0.09))

    def grid_lines(a):
        if beat_t is not None:
            for b in beat_t[(beat_t >= tt[0]) & (beat_t <= tt[-1])]:
                a.axvline(b - t_hud0, color="#f0efec", lw=0.8, zorder=0)
        if meas_t is not None:
            for b in meas_t[(meas_t >= tt[0]) & (meas_t <= tt[-1])]:
                a.axvline(b - t_hud0, color="#d6d5cf", lw=1.0, zorder=0)

    a = axs[0]
    grid_lines(a)
    for ta in doc["t_axes"]:
        a.barh(3, ta["t_formed"] - ta["t0"], left=ta["t0"] - t_hud0, height=0.6, color=BLUE_RAMP[1],
               edgecolor="white", lw=1)
        a.barh(3, ta["t1"] - ta["t_formed"], left=ta["t_formed"] - t_hud0, height=0.6, color=BLUE_RAMP[5],
               edgecolor="white", lw=1)
        a.text(0.5 * (ta["t0"] + ta["t1"]) - t_hud0, 3, f"T{ta['id']}" + ("*" if ta["same_place_as"] is not None
                                                                         else ""),
               fontsize=7, color="white", ha="center", va="center")
    for ap in doc["axis_pivots"]:
        v = ap["evaluation"]["verdict"]
        a.barh(2, max(ap["t1"] - ap["t0"], 0.04), left=ap["t0"] - t_hud0, height=0.5, color=MUTED, edgecolor="none")
        xm = 0.5 * (ap["t0"] + ap["t1"]) - t_hud0
        a.plot(xm, 2.42, VERDICT_MARK[v], color=VERDICT_C[v], ms=6, mec="white", mew=0.6)
        a.text(xm, 1.62, f"{ap['signed_deg']:+.0f}°\n{ap['evaluation'].get('turn_kind_label', '')}", fontsize=6.0,
               color=INK, ha="center", va="top")
    for d in doc["dials"]:
        v = d["evaluation"]["verdict"]
        a.barh(1, d["t1"] - d["t0"], left=d["t0"] - t_hud0, height=0.5, color=CB_C, edgecolor=INK, lw=0.6)
        a.text(0.5 * (d["t0"] + d["t1"]) - t_hud0, 1.3, f"dial {d['degrees']:+.0f}° {d['kind'].replace('_', ' ')}",
               fontsize=7, ha="center", va="bottom", color=INK)
        a.plot(d["t1"] - t_hud0 + 0.15, 1, VERDICT_MARK[v], color=VERDICT_C[v], ms=7, mec="white", mew=0.6)
        a.text(d["t1"] - t_hud0 + 0.32, 1, v, fontsize=7, ha="left", va="center", color=INK)
    for c in doc["counterbalances"]:
        a.barh(0, c["t1"] - c["t0"], left=c["t0"] - t_hud0, height=0.36, color=CB_C, alpha=0.5, edgecolor="none")
    for tr in truth or []:
        a.barh(-0.35, tr["t1"] - tr["t0"], left=tr["t0"] - t_hud0, height=0.2, color="none", edgecolor=INK, lw=1.0,
               hatch="///")
        a.text(tr["t0"] - t_hud0, -0.62, f"user label HUD {tr['t0'] - t_hud0:.2f}-{tr['t1'] - t_hud0:.2f} s",
               fontsize=7, ha="left", va="top", color=INK)
    a.set_yticks([3, 2, 1, 0])
    a.set_yticklabels(["T axes (light = forming;\n* = rotated in place)", "axis pivots\n(mark = verdict)",
                       "dials", "counterbalance\n(detected; hatched = user)"], fontsize=8)
    a.set_ylim(-0.9, 3.6)
    a.set_title(f"{take}: T-axis segmentation (corridors ±{cfg['half_width_m']:.2f} m x ±{cfg['forward_m']:.1f} m; "
                f"new T after a sustained ({cfg['sustain_beats']:.1f} beat) exit or > {cfg['pivot_deg']:.0f}° "
                f"rotation; forming {cfg['form_beats']:.1f} beat)   verdicts (one-foot pivots and dials only): o good, "
                f"■ ok, X needs work, _ not graded",
                fontsize=10, loc="left")
    a = axs[1]
    grid_lines(a)
    a.plot(x, yy, color=MUTED, lw=1.0, label="chest heading (smoothed, unwrapped)")
    for ta in doc["t_axes"]:
        Y = ta["yaw_unwrapped_deg"]
        a.plot([ta["t0"] - t_hud0, ta["t1"] - t_hud0], [Y, Y], color=LEAD_C, lw=2.2, solid_capstyle="butt")
        a.fill_between([ta["t0"] - t_hud0, ta["t1"] - t_hud0], Y - cfg["pivot_deg"], Y + cfg["pivot_deg"],
                       color=LEAD_C, alpha=0.08, lw=0)
    a.plot([], [], color=LEAD_C, lw=2.2, label=f"T heading held (band ±{cfg['pivot_deg']:.0f}°)")
    a.set_ylabel("heading (deg, + = ccw\nfrom above, unwrapped)", fontsize=9)
    a.legend(loc="lower left", fontsize=8, framealpha=0.9)
    a = axs[2]
    grid_lines(a)
    fo = np.full(len(tt), np.nan)
    so = np.full(len(tt), np.nan)
    for ta in doc["t_axes"]:
        k = (tt >= ta["t0"]) & (tt <= ta["t1"])
        O = np.array(ta["origin"], float)[[0, 2]]
        Y = ta["yaw_unwrapped_deg"]
        fo[k], so[k] = t_offsets(oo[k], O, Y)
    a.plot(x, fo, color=BLUE_RAMP[6], lw=1.2, label="along the stem (forward)")
    a.plot(x, so, color=SIDE_C, lw=1.2, label="along the crossbar (his left)")
    for v in (cfg["half_width_m"], -cfg["half_width_m"]):
        a.axhline(v, color=MUTED, lw=0.8, ls="--")
    a.set_ylabel("chest offset from\nthe current T (m)", fontsize=9)
    a.legend(loc="lower left", fontsize=8, framealpha=0.9, title=f"dashed: ±{cfg['half_width_m']:.2f} m corridor "
             "half-width", title_fontsize=7.5)
    bal = doc.get("balance") or {}
    bt = np.asarray(bal.get("t") or [], float) - t_hud0
    for a, role, col in ((axs[3], "lead", LEAD_C), (axs[4], "follow", "#2a78d6")):
        grid_lines(a)
        b = bal.get(role)
        if b is None or not len(bt):
            continue
        sc = np.asarray([np.nan if v is None else v for v in b["score"]], float)
        st = np.asarray(b["stance"])
        sc[st == 0] = np.nan
        a.plot(bt, sc, color=col, lw=1.0, label=f"{role}: balance-axis score (chest -> stance foot)")
        if role == "follow" and "couple" in bal:
            cs = np.asarray([np.nan if v is None else v for v in bal["couple"]["score"]], float)
            a.plot(bt, cs, color=CB_C, lw=1.6, label="couple: combined COM over her anchored foot (counterbalance)")
        a.axhline(doc["params"]["balance"]["aligned_score"], color=MUTED, lw=0.8, ls="--")
        a.set_ylim(-0.05, 1.05)
        a.set_ylabel(f"{role} score\n(1 = COM over the foot)", fontsize=9)
        a.legend(loc="lower left", fontsize=8, framealpha=0.9)
    axs[-1].set_xlabel(f"time from the first frame (s) = viewer HUD   [reference = HUD + {t_hud0:.3f}]   grid: light = "
                       "beats, dark = measure starts;  dashed = aligned threshold; gaps = no foot in contact")
    for a in axs:
        a.tick_params(labelsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------- CLI

def summary(doc: dict) -> str:
    p = doc["params"]
    lines = [f"{doc['take']}: {p['counts']['plants']} plants ({p['plant_share']:.0%} of the time), "
             f"{p['counts']['moves']} moves ({p['counts'].get('steps', 0)} single steps), {p['counts']['rotations']} "
             f"rotations, {p['counts']['pivots']} pivot(s); "
             f"axis origin speed p10/p50/p90 {p['origin_speed_m_s']['p10']}/{p['origin_speed_m_s']['p50']}/"
             f"{p['origin_speed_m_s']['p90']} m/s  [{doc['source']['smplx_dir']}; {doc['source']['counterbalance']}]"]
    for pl in doc["plants"]:
        lines.append(f"  plant {pl['id']:>2} {pl['t0']:.2f}-{pl['t1']:.2f} s ({pl['duration_beats']:.1f} beats) at "
                     f"({pl['origin'][0]:+.2f}, {pl['origin'][2]:+.2f}) r {pl['radius_m']:.2f} m yaw {pl['yaw_deg']:+.0f} "
                     f"({pl.get('yaw_source', 'mean')}) turn {pl['turn_deg']:+.0f}")
    rot = doc["rotations"]
    if rot:
        lines.append(f"  rotations: {sum(r['context'] == 'plant' for r in rot)} in plants, "
                     f"{sum(r['context'] == 'move' for r in rot)} in moves; "
                     f"|angle| median {np.median([abs(r['signed_deg']) for r in rot]):.0f} deg")
    for pv in doc["pivots"]:
        la = pv.get("leader_axis") or {}
        lines.append(f"  pivot {pv['id']} {pv['t0']:.2f}-{pv['t1']:.2f} s {pv['swept_deg']:.0f} deg {pv['direction']} "
                     f"r {pv['leader_radius_m']:.2f} m conf {pv['confidence']:.2f}; leader axis turn "
                     f"{la.get('turn_deg', float('nan')):+.0f} deg over {la.get('length_m', float('nan')):.1f} m")
    if "t_axes" in doc:
        h0 = doc["source"]["t_range"][0] if doc.get("source", {}).get("t_range") else 0.0
        lines.append(f"  T axis: {p['counts']['t_axes']} Ts ({p.get('t_axis_share') or 0:.0%} of the time), "
                     f"{p['counts']['axis_pivots']} axis pivots, {p['counts']['dials']} dial(s)  [HUD = reference - "
                     f"{h0:.3f}]")
        for ta in doc["t_axes"]:
            o = ta["origin"]
            lines.append(f"    T{ta['id']:<2} HUD {ta['t0'] - h0:6.2f}-{ta['t1'] - h0:6.2f} at ({o[0]:+.2f}, "
                         f"{o[2]:+.2f}) yaw {ta['yaw_deg']:+6.1f}  {ta['start_reason']} -> {ta['end_reason']}"
                         + (f"  (rotated in place of T{ta['same_place_as']})" if ta["same_place_as"] is not None
                            else ""))
        for ap in doc["axis_pivots"]:
            short = {"track_start": "start", "track_end": "end", "brief_t": "brief", "counterbalance": "CB"}
            fr = f"T{ap['from_t']}" if ap["from_t"] is not None else short.get(ap.get("from_kind"), "?")
            to = f"T{ap['to_t']}" if ap["to_t"] is not None else short.get(ap.get("to_kind"), "?")
            lines.append(f"    pivot P{ap['id']:<2} HUD {ap['t0'] - h0:6.2f}-{ap['t1'] - h0:6.2f} {fr}->{to} "
                         f"{ap['signed_deg']:+7.1f} deg {'in place' if ap['stayed'] else 'moving'} "
                         f"[{ap['evaluation'].get('turn_kind_label', '?')}]  {ap['evaluation']['text']}")
        for d in doc["dials"]:
            c = d["centre"]
            lines.append(f"    dial {d['id']} {d['kind']} HUD {d['t0'] - h0:6.2f}-{d['t1'] - h0:6.2f} "
                         f"{d['degrees']:+.0f} deg at ({c[0]:+.2f}, {c[2]:+.2f}): {d['who']}; "
                         f"{d['evaluation']['text']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.floorcraft", description=__doc__.split("\n\n")[0])
    ap.add_argument("take")
    ap.add_argument("--export-capture", default=None, help="also write the Unity copy into this capture")
    ap.add_argument("--export-only", action="store_true", help="only the Unity copy of the existing floorcraft.json")
    ap.add_argument("--smplx-dir", default=None, help="default: work/<take>/smplx, else smplx_v1")
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--review-dir", default=None, help="review images here (default work/<take>/review/)")
    ap.add_argument("--force", action="store_true", help="export even if the capture motion differs from the fits")
    a = ap.parse_args(argv)
    if not a.export_only:
        res = run(a.take, a.smplx_dir, review=not a.no_review, review_dir=a.review_dir)
        print(summary(res["_json"]))
        print(f"[floorcraft] -> {res['_path']}")
        for p in res.get("_review") or []:
            print(f"[floorcraft] review -> {p}")
    if a.export_capture or a.export_only:
        print("[floorcraft] unity:", json.dumps(export(a.take, a.export_capture, force=a.force), indent=1))


if __name__ == "__main__":
    main()
