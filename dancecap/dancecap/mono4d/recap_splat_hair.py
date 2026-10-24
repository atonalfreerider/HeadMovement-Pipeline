r"""Swinging hair for the recap follow's splat avatar, driven by the UNITY hair simulation (2026-10-20; owner: "I need the
larissa kadu recap to look LHM++ quality with swinging hair"; route "Splats inside Unity").

The viewer already simulates Larissa's hair: 160 guide strands x 17 points (head-movement Assets/Hair/HairStrands.cs,
from the capture's hair_groom.json), with the body capsules, the face box and the comb. The mesh avatar draws hair cards
on those guides. The splat dancer (Assets/Splats/Avatar) skins her LHM++ hair Gaussians rigidly with LHM++'s own LBS
weights (head / neck / upper back): the hair never swings. This stage binds the FREE hair Gaussians (the hanging hair
below the scalp) to the Unity guides, offline and deterministically, by filling the binding slot of the shared format
(docs/SPLAT_AVATAR_FORMAT.md section 6); the Unity splat layer then moves them with the simulated strand segments every
frame (Assets/Splats/Avatar/Core/Shaders/HM_SkinSplats.compute kernels HairSegments + SkinHair; the CPU reference is
Assets/Splats/Avatar/Core/SplatHairBinding.cs; the full rule is docs/SPLAT_AVATAR_FORMAT.md section 6).

    python -m dancecap.mono4d.recap_splat_hair bind     [--take T] [--dump unity_hair_dump_frame0.json] [--install]
    python -m dancecap.mono4d.recap_splat_hair select   [--take T]   # LOOK before any dump: rigid / free / fringe counts + sheet
    python -m dancecap.mono4d.recap_splat_hair sheet    [--take T]   # LOOK: binding sheet (strand colours + guides), posed previews
    python -m dancecap.mono4d.recap_splat_hair verify   [--take T]   # re-read the installed file: records, sha, binding round trip
    python -m dancecap.mono4d.recap_splat_hair finish   --take T --dump <hm_hair dump.json>   # ONE command after the Unity dump:
        copy the dump to <out>/unity_hair_dump_frame0.json, bind + install, verify, sheet, and write the Quest re-distill
        commands of the bound follow (<out>/finish.json; the Quest sets must carry the binding: quest_distill prepare --roles follow)

TAKE PROFILES (2026-10-23, HAIR_PROFILES): no --take = the class recap (Larissa's RED hair: the GREY fringe rule below);
--take carlos_aline = Aline's long near-black hair (her top is red, the recap's red / grey rules mis-select): the BLACK
rule: a free Gaussian keeps its LBS when it is lighter than BRIGHT_L (the white fringe) or NOT hair-coloured by
splat_hair.select_hair's learned colour model (crown vs face + torso front, Mahalanobis in Lab: the red top, her skin
under the hair, edges of the white crop stay on the body). Everything else (binding, posing, file) is take-agnostic.

Inputs: work/<take>/lhmpp/follow.splatav + .json (the recap_splatav export, unbound; never modified here) and
a dump of the Unity guides (`unity command hm_hair --action dump` in Play mode with the take's capture loaded:
`rest` = the settled rest shape in HEAD-BONE space, i.e. relative to the avatar skin's head joint 15, axes = Unity axes in
the T-pose). Output: work/<take>/splat_hair_unity/follow.splatav + .json (bound), sheets, report.json;
`--install` copies the pair to head-movement/Assets/StreamingAssets/<capture>/splats/ (git-ignored likeness data).

SELECTION (which Gaussians swing): splat_hair.select_hair, style "loose" (the other hair workflow's selection, reused):
RIGID = scalp-level hair (within 3 cm of the head capsule, above the head joint) keeps LHM++'s LBS (head weight >= 0.91,
mean 0.996 on this export: rigid with the head); FREE = the hanging hair + the hair layer painted on the body under it.
The hair byte = exporter flag OR rigid OR free (all hair-looking Gaussians); bound (strand >= 0) = FREE minus the white
fringe (L* > BRIGHT_L) and the grey Gaussians (a* < GREY_A, L* > GREY_DARK_L: the mic pack at her lower back, edges of the
top), which keep their LBS.

BINDING (Unity coordinates, head-bone space; c = canonical centre - J_15 of the splat skeleton):
  nearest rest segment (Euclidean, ties to the lower guide / segment index) of the Unity guides;
  segment frame F_s = [X Y Z] (columns): Y = unit tangent point s -> s+1, Z = her forward (Unity -Z, = SMPL-X +Z of the
  head joint) made orthogonal to Y (Unity +Y when |Y . fwd| > 0.999), X = Y x Z (det +1);
  local_offset = F_s^T (c - R_s), local_rotation = q(F_s)^-1 * q_canonical (xyzw).
POSING (what the Unity runtime does; pose_reference() below is its CPU reference):
  per segment, the sim points re-anchored to the splat skeleton's head (P' = h_now + H_now H_sim^-1 (P - h_sim)), so the
  hair sits on the head the body splats show (also between capture frames: the film's sub-frame poses);
  D_s = minrot(H_now t_rest_s, t_sim_s) H_now; the offset c - R_s split along / across the rest tangent (a, p): posed
  from segment s: P'_s + D_s a + K_s p, K_s = nlerp(H_now, D_s, k), k = 1 - smoothstep(lever.near_m, lever.far_m, |p|)
  (hair on its strand turns with it; hair far from every guide follows the guide's displacement, no lever arm);
  smooth along the strand: blended with the neighbour segment N (s+1 when the projection u > 0.5, s-1 when u < 0.5) by
  w = clamp(|u - 0.5|, 0, 0.5) (continuous across the segment joints: at a joint both sides blend 50/50);
  the simulation's face box (Unity only: pose_reference below leaves it out) pushes bound Gaussians off the face;
  attached at the top: lerp(head-rigid S_15 mean, strand pose, a), a = smoothstep over the head-bone height
  (attach.y_top_m .. attach.y_top_m - attach.width_m), so the free hair grows out of the rigid scalp hair without a seam.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE
from . import splat_hair as SH

TAKE = "larissa_kadu_recap"
CAPTURE = "06_LarissaKaduRecap"
WORK = PACKAGE_ROOT / "work" / TAKE
SRC = WORK / "lhmpp"
OUT = WORK / "splat_hair_unity"
UNITY_DIR = WORKSPACE / "head-movement" / "Assets" / "StreamingAssets" / CAPTURE / "splats"
GROOM = WORKSPACE / "head-movement" / "Assets" / "StreamingAssets" / CAPTURE / "hair_groom.json"
HEAD = 15
FORWARD_UNITY = np.array([0.0, 0.0, -1.0])   # her forward in Unity coordinates (SMPL-X +Z mirrored)
UP_UNITY = np.array([0.0, 1.0, 0.0])
ATTACH = {"y_top_m": 0.0, "width_m": 0.05}     # head-bone height (0 = the head joint = splat_hair's loose hang level)
LEVER = {"near_m": 0.02, "far_m": 0.07}        # the across-strand offset turns with the segment up to near_m, with the head
                                               # beyond far_m (no lever arm: side hair 5-12 cm from its guide stays off the face)
BRIGHT_L = 55.0   # free Gaussians lighter than this (CIE L*, hair Lab mean ~33) are the white fringe LHM++ puts around the
                  # hair (the top / the photo background bleeding in; mean opacity ~0.02): they keep their LBS (where LHM++ had
                  # them, over the white top) instead of flying out with the hair as a grey haze
GREY_A = 15.0     # ... and free Gaussians with CIE a* below this that are not near-black (L* > GREY_DARK_L): grey, not red
GREY_DARK_L = 15.0  # hair (a* 5th percentile 18 on this export): the mic pack clipped at her lower back under the hair tips
                  # and the edges of the white top; they stay on the body (LBS) instead of flying out as grey spikes


HAIR_LOOK_BLACK = {
    # (2026-10-23 fix round 1; Aline's near-black hair; see apply_look): LHM++ drew her hair with few large opaque Gaussians,
    # often 2-8 near-copies stacked within 2 mm (composite opacity up to 0.99); a strand that swings out alone carried them as
    # round black discs onto his white shirt / her red top. The bound hair becomes strand-shaped and translucent per clump:
    "needle_across_m": 0.003,   # bound hair: longest axis along its rest segment, both across sigmas <= 3 mm ...
    "needle_along_max_m": 0.05,  # ... the along sigma grown to keep the projected area (sigma_max * sigma_mid / across), <= 5 cm
    "stack_cap": 0.25,          # bound hair: composite opacity of a 2 mm stack of near-copies capped at 0.25
    "stack_link_m": 0.002,
    "paint_fade": 0.03,         # unbound hair PAINT on the torso back (a layer ~1.5 cm over the top's own surface, dark and
    "paint_ytop_m": 0.03,       # achromatic): opacity x 0.03, spine1 .. neck + 3 cm (the top's red shows when the hair swings)
    "under_tint": {"max_chroma": 15.0, "l_range": [12.0, 35.0], "near_paint_m": 0.025, "dark_l": 12.0, "dark_chroma": 20.0,
                   "red_chroma": 25.0, "red_hue_deg": [15.0, 40.0], "radii_m": [0.03, 0.05, 0.08, 0.12], "min_red": 8,
                   "l_ratio_clip": [0.7, 1.3],
                   "tint_paint": True},
    # (fix round 2) what the faded paint exposed: LHM++ painted the top's own back GREY under the hair (Lab ~23 / 13 / 4,
    # opacity ~0.45, at the top's depth) - a pale blotch whenever the hair swings off her back - plus dark brown smudges
    # (L* < 12, C* < 20) beside the shoulder blades. Each group takes the local colour of the top (under_mask / under_tint:
    # the opacity-weighted median of the clearly red Gaussians nearby: C* >= red_chroma = the top's 10th percentile, hue
    # in red_hue_deg = the top's 28-31 deg, not her skin's ~47; the group's own shading kept); the faded paint is tinted the same way (it stays faded: opaque red paint 1.5 cm over the
    # top drew red fog through the hair in fix round 1).
    "crown": {"el_from": 10.0, "el_full": 30.0, "face_az": 50.0, "face_above_hairline": 6.0, "max_chroma": 15.0, "shell_m": 0.03},
    # unbound scalp hair above the crown, outside the face box: luminance toward the hanging hair's (LHM++ guessed a grey top
    # the head-height cameras never saw); scale = opacity-weighted mean L* ratio, clamped at the hanging hair's p90, hue kept.
    # Candidates: the hair byte, and any head Gaussian within shell_m of the head capsule (the white fringe / highlights on the
    # crown that select_hair did not flag as hair); the chroma limit keeps skin out
}


@dataclass(frozen=True)
class HairProfile:
    """The take-dependent part of the hair stage: paths and the fringe rule ("grey" = the recap's red hair, "black");
    look = the black-hair look fixes (apply_look; None = none, the recap)."""
    take: str
    capture: str
    fringe: str = "grey"
    out_dir: str | None = None        # override of work/<take>/splat_hair_unity (dry runs)
    look: dict | None = None

    @property
    def work(self) -> Path:
        return PACKAGE_ROOT / "work" / self.take

    @property
    def src(self) -> Path:
        return self.work / "lhmpp"

    @property
    def out(self) -> Path:
        return Path(self.out_dir) if self.out_dir else self.work / "splat_hair_unity"

    @property
    def unity_dir(self) -> Path:
        return WORKSPACE / "head-movement" / "Assets" / "StreamingAssets" / self.capture / "splats"

    @property
    def groom(self) -> Path:
        return WORKSPACE / "head-movement" / "Assets" / "StreamingAssets" / self.capture / "hair_groom.json"


HAIR_PROFILES = {"larissa_kadu_recap": HairProfile(TAKE, CAPTURE, "grey"),
                 "carlos_aline": HairProfile("carlos_aline", "07_CarlosAline", "black", look=HAIR_LOOK_BLACK)}
HAIR_PROFILES["recap"] = HAIR_PROFILES["larissa_kadu_recap"]
HP: HairProfile = HAIR_PROFILES[TAKE]


def use(take: str | None = None, **over) -> HairProfile:
    """Make a take's hair profile the active one (None = the class recap)."""
    global HP
    key = take or TAKE
    if key not in HAIR_PROFILES:
        raise SystemExit(f"recap_splat_hair: no take profile {key!r} (known: {', '.join(sorted(HAIR_PROFILES))})")
    HP = dataclasses.replace(HAIR_PROFILES[key], **over) if over else HAIR_PROFILES[key]
    return HP


def fringe_mask(lab: np.ndarray, free: np.ndarray, hair_colour, rule: str) -> tuple[np.ndarray, dict]:
    """Free Gaussians that keep their LBS instead of being bound (module doc): the white fringe (L* > BRIGHT_L) plus
    "grey": CIE a* < GREY_A and L* > GREY_DARK_L (not red: the mic pack, the top's edges; red hair only), or
    "black": not hair-coloured by select_hair's learned crown model (hair_colour False). -> (mask, counts)."""
    light = lab[:, 0] > BRIGHT_L
    if rule == "grey":
        other = (lab[:, 1] < GREY_A) & (lab[:, 0] > GREY_DARK_L)
        name = "free_grey_kept_lbs"
    elif rule == "black":
        if hair_colour is None:
            raise ValueError("fringe rule 'black' needs select_hair's hair_colour mask")
        other = ~np.asarray(hair_colour, bool)
        name = "free_not_hair_coloured_kept_lbs"
    else:
        raise ValueError(f"unknown fringe rule {rule!r} (grey | black)")
    m = free & (light | other)
    return m, {"free_light_fringe_kept_lbs": int((free & light).sum()), name: int((free & other & ~light).sum())}


RECORD = np.dtype([("mean", "<f4", 3), ("rotation", "<f4", 4), ("log_scale", "<f4", 3), ("opacity", "<f4"),
                   ("sh0", "<f4", 3), ("joints", "<u2", 4), ("weights", "<f4", 4), ("hair", "u1"), ("reserved", "u1", 3),
                   ("strand", "<i4"), ("segment", "<i4"), ("local_offset", "<f4", 3), ("local_rotation", "<f4", 4)])
assert RECORD.itemsize == 120


def log(msg: str) -> None:
    print(f"[recap_splat_hair] {msg}", flush=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _ws(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return str(p)


# ================================================================================================ quaternions (xyzw)
def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of x y z w quaternions (b applied first), broadcasting."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, float), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def q_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float).copy()
    q[..., :3] *= -1
    return q


def q_from_mat(R: np.ndarray) -> np.ndarray:
    w = SH.mat_to_quat(R)                      # w x y z, w >= 0
    return np.concatenate([w[..., 1:], w[..., :1]], -1)


def q_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    return SH.quat_to_mat(np.concatenate([q[..., 3:], q[..., :3]], -1))


def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", q_to_mat(q), v)


# ================================================================================================ files
def read_splatav(path: Path) -> tuple[bytes, np.ndarray, bytes]:
    raw = Path(path).read_bytes()
    assert raw[:4] == b"SPAV", f"{path}: bad magic"
    hdr = np.frombuffer(raw[4:64], "<u4")
    version, n, rb, sh_rest, flags, hb = (int(x) for x in hdr[:6])
    assert version == 1 and rb == 120 and hb == 64 and flags & 1, (version, rb, hb, flags)
    rec = np.frombuffer(raw[hb:hb + n * rb], RECORD).copy()
    return raw[:hb], rec, raw[hb + n * rb:]


def load_dump(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    G, P = int(d["guides"]), int(d["points"])
    rest = np.asarray(d["rest"], float).reshape(G, P, 3)
    pos = np.asarray(d["pos"], float).reshape(G, P, 3) if d.get("pos") else None
    return {"guides": G, "points": P, "rest": rest, "pos": pos, "head_pos": np.asarray(d["headPos"], float),
            "head_rot": np.asarray(d["headRot"], float), "raw": d}


def rest_sha(rest: np.ndarray) -> str:
    return hashlib.sha256(np.round(np.asarray(rest, float), 6).astype("<f8").tobytes()).hexdigest()


# ================================================================================================ binding
def segment_frames(rest: np.ndarray) -> np.ndarray:
    """(G, P, 3) rest points (head-bone space, Unity coords) -> (G, P-1, 3, 3) frames, columns X, Y, Z (module doc)."""
    Y = np.diff(rest, axis=-2)
    Y /= np.maximum(np.linalg.norm(Y, axis=-1, keepdims=True), 1e-12)
    ref = np.where((np.abs(Y @ FORWARD_UNITY) > 0.999)[..., None], UP_UNITY, FORWARD_UNITY)
    Z = ref - (ref * Y).sum(-1, keepdims=True) * Y
    Z /= np.maximum(np.linalg.norm(Z, axis=-1, keepdims=True), 1e-12)
    X = np.cross(Y, Z)
    return np.stack([X, Y, Z], -1)


def bind(c: np.ndarray, q: np.ndarray, rest: np.ndarray, chunk: int = 2048) -> dict:
    """Head-bone centres c (n, 3) and canonical rotations q (n, 4 xyzw) -> nearest rest segment binding."""
    G, P, _ = rest.shape
    A = rest[:, :-1].reshape(-1, 3)
    B = rest[:, 1:].reshape(-1, 3)
    best = np.empty(len(c), np.int64)
    dist = np.empty(len(c))
    for i0 in range(0, len(c), chunk):
        p = c[i0:i0 + chunk, None, :]
        cp, _ = SH.segment_closest(p, A[None], B[None])
        d = np.linalg.norm(p - cp, axis=-1)
        best[i0:i0 + chunk] = d.argmin(1)            # argmin: first minimum = lowest guide / segment
        dist[i0:i0 + chunk] = d.min(1)
    strand, seg = best // (P - 1), best % (P - 1)
    F = segment_frames(rest)[strand, seg]
    off = np.einsum("nji,nj->ni", F, c - rest[strand, seg])
    lrot = q_mul(q_conj(q_from_mat(F)), q)
    L = np.linalg.norm(rest[strand, seg + 1] - rest[strand, seg], axis=1)
    u = off[:, 1] / np.maximum(L, 1e-9)
    return {"strand": strand.astype(np.int32), "segment": seg.astype(np.int32), "local_offset": off,
            "local_rotation": lrot, "distance": dist, "u": u}


def attach_weight(c_y: np.ndarray, attach: dict = ATTACH) -> np.ndarray:
    """0 = rigid with the head (at / above y_top), 1 = moved by the strand (width below it); smoothstep."""
    x = np.clip((attach["y_top_m"] - np.asarray(c_y, float)) / max(attach["width_m"], 1e-6), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def neighbour(seg: np.ndarray, u: np.ndarray, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour segment index and blend weight of the along-strand smoothing (module doc)."""
    up = u >= 0.5
    nb = np.where(up, seg + 1, seg - 1)
    w = np.clip(np.abs(u - 0.5), 0.0, 0.5)
    ok = (nb >= 0) & (nb < segments)
    return np.where(ok, nb, seg).astype(np.int32), np.where(ok, w, 0.0)


def pose_reference(c: np.ndarray, q: np.ndarray, b: dict, rest_bind: np.ndarray, rest_live: np.ndarray,
                   sim: np.ndarray, sim_head_pos: np.ndarray, sim_head_rot: np.ndarray, head_pos: np.ndarray,
                   head_rot: np.ndarray, attach: dict = ATTACH) -> tuple[np.ndarray, np.ndarray]:
    """CPU reference of the Unity hair pass for bound Gaussians (all in one frame: Unity world or avatar space).
    c / q: canonical head-bone centres / rotations (used only for the head-rigid end of the attach blend);
    b: binding (strand, segment, local_offset, local_rotation); rest_bind: the rest the binding was made on;
    rest_live: the simulation's own rest; sim: simulated points; sim_head_*: the head pose the simulation used;
    head_*: the head pose shown now (the splat skeleton's head: position of J_15, rotation)."""
    G, P, _ = rest_live.shape
    S = P - 1
    Hn = np.asarray(head_rot, float)
    A = q_mul(Hn, q_conj(sim_head_rot))                       # sim head -> shown head
    Pn = head_pos + q_rotate(A, sim.reshape(-1, 3) - sim_head_pos).reshape(G, P, 3)
    t_rest = np.diff(rest_live, axis=1)
    t_rest /= np.linalg.norm(t_rest, axis=-1, keepdims=True)
    t_sim = np.diff(Pn, axis=1)
    t_sim /= np.linalg.norm(t_sim, axis=-1, keepdims=True)
    t_car = q_rotate(np.broadcast_to(Hn, t_rest.shape[:-1] + (4,)), t_rest)
    D = q_mul(q_from_mat(SH.min_rotation(t_car, t_sim)), Hn)  # (G, S, 4)
    s, k = b["strand"], b["segment"]
    F = segment_frames(rest_bind)[s, k]
    ch = rest_bind[s, k] + np.einsum("nij,nj->ni", F, b["local_offset"])      # canonical head-bone centre
    qc = q_mul(q_from_mat(F), b["local_rotation"])
    u = b["local_offset"][:, 1] / np.linalg.norm(rest_bind[s, k + 1] - rest_bind[s, k], axis=1)
    nb, w = neighbour(k, u, S)
    def nlerp(qa, qb, t):
        qb = np.where(((qa * qb).sum(-1) < 0)[:, None], -qb, qb)
        r = (1 - t)[:, None] * qa + t[:, None] * qb
        return r / np.linalg.norm(r, axis=-1, keepdims=True)

    def split(o, tt):
        al = (o * tt).sum(-1, keepdims=True) * tt
        return al, o - al

    Hb = np.broadcast_to(Hn, (len(s), 4))
    oa, on = ch - rest_live[s, k], ch - rest_live[s, nb]
    ala, pea = split(oa, t_rest[s, k])
    aln, pen = split(on, t_rest[s, nb])
    x = np.clip((np.linalg.norm(pea, axis=1) - LEVER["near_m"]) / (LEVER["far_m"] - LEVER["near_m"]), 0, 1)
    kk = 1 - x * x * (3 - 2 * x)
    ka, kn = nlerp(Hb, D[s, k], kk), nlerp(Hb, D[s, nb], kk)
    pa = Pn[s, k] + q_rotate(D[s, k], ala) + q_rotate(ka, pea)
    pb = Pn[s, nb] + q_rotate(D[s, nb], aln) + q_rotate(kn, pen)
    pos = (1 - w)[:, None] * pa + w[:, None] * pb
    rot = q_mul(nlerp(ka, kn, w), qc)
    a = attach_weight(ch[:, 1], attach)
    rig_p = head_pos + q_rotate(np.broadcast_to(Hn, ch.shape[:-1] + (4,)), ch)
    rig_q = q_mul(np.broadcast_to(Hn, qc.shape), qc)
    pos = (1 - a)[:, None] * rig_p + a[:, None] * pos
    rig_q = np.where(((rig_q * rot).sum(-1) < 0)[:, None], -rig_q, rig_q)
    rot = (1 - a)[:, None] * rig_q + a[:, None] * rot
    rot /= np.linalg.norm(rot, axis=-1, keepdims=True)
    return pos, rot


# ================================================================================================ black-hair look (fix round 1)
def needle_shape(q: np.ndarray, log_scale: np.ndarray, tangent: np.ndarray, across: float,
                 along_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Strand-shaped bound hair. q (n, 4) xyzw canonical rotations, log_scale (n, 3), tangent (n, 3) rest tangents of the
    bound segments (same axes) -> new (q, log_scale): local x = the tangent, local y = the old middle axis made orthogonal
    to it (an arbitrary orthogonal axis when parallel), z = x cross y; sigmas (along, min(mid, across), min(min, across)),
    along = max(sigma_max, min(sigma_max * sigma_mid / min(mid, across), along_max)): the largest face's projected area is
    kept (a round clump becomes a streak along its strand, never shorter than it was)."""
    q = np.asarray(q, float)
    t = np.asarray(tangent, float)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    S = np.exp(np.asarray(log_scale, float))
    n = len(q)
    ar = np.arange(n)
    order = np.argsort(S, 1, kind="stable")
    s_min, s_mid, s_max = S[ar, order[:, 0]], S[ar, order[:, 1]], S[ar, order[:, 2]]
    mid_axis = q_to_mat(q)[ar, :, order[:, 1]]                       # columns of R = the local axes
    y = mid_axis - (mid_axis * t).sum(1, keepdims=True) * t
    parallel = np.linalg.norm(y, axis=1) < 1e-6
    ref = np.where((np.abs(t[:, 1]) > 0.9)[:, None], np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    y = np.where(parallel[:, None], np.cross(t, ref), y)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    z = np.cross(t, y)
    c_mid, c_min = np.minimum(s_mid, across), np.minimum(s_min, across)
    along = np.maximum(s_max, np.minimum(s_max * s_mid / c_mid, along_max))
    return q_from_mat(np.stack([t, y, z], -1)), np.log(np.stack([along, c_mid, c_min], 1))


def stack_cap(centres: np.ndarray, opacity: np.ndarray, cap: float, link: float) -> tuple[np.ndarray, dict]:
    """Groups of near-copies (centres connected within link metres) keep a composite opacity 1 - prod(1 - a) <= cap: each
    member's ln(1 - a) is scaled by ln(1 - cap) / ln(1 - composite) where the composite is above cap (members keep their
    ratios; groups under the cap are unchanged). -> (opacity, counts)"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    c = np.asarray(centres, float)
    a = np.clip(np.asarray(opacity, float), 0.0, 1.0 - 1e-7)
    n = len(c)
    pr = cKDTree(c).query_pairs(link, output_type="ndarray") if n > 1 else np.zeros((0, 2), int)
    k, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)), directed=False)
    nl = np.log1p(-a)
    tot = np.bincount(lab, weights=nl, minlength=k)
    over = -np.expm1(tot) > cap
    s = np.ones(k)
    s[over] = np.log1p(-cap) / tot[over]
    return -np.expm1(nl * s[lab]), {"groups": int(k), "groups_capped": int(over.sum()),
                                    "gaussians_capped": int(over[lab].sum()), "in_groups_ge2": int((np.bincount(lab)[lab] >= 2).sum())}


def paint_mask(mean_unity: np.ndarray, J: np.ndarray, normal_z_world: np.ndarray, lab: np.ndarray, unbound: np.ndarray,
               rigid: np.ndarray, hair_colour: np.ndarray, ytop: float) -> np.ndarray:
    """Hair PAINT on the torso back: unbound, not the rigid scalp, facing back (body normal z < -0.2 in the dancecap world,
    where the front is +z), between spine1 (J3) and the neck (J12) + ytop, within 22 cm of the spine sideways, achromatic
    (C* < 12) and dark (L* < 20) or hair-coloured. Mean / J in Unity coordinates (x, y as the dancecap world)."""
    m = np.asarray(mean_unity, float)
    C = np.hypot(lab[:, 1], lab[:, 2])
    zone = ((np.asarray(normal_z_world) < -0.2) & (m[:, 1] > J[3, 1]) & (m[:, 1] < J[12, 1] + ytop)
            & (np.abs(m[:, 0] - J[9, 0]) < 0.22))
    return zone & unbound & ~rigid & (C < 12) & ((lab[:, 0] < 20) | hair_colour)


def back_zone(mean_unity: np.ndarray, J: np.ndarray, normal_z_world: np.ndarray, ytop: float) -> np.ndarray:
    """The torso back of paint_mask: facing back (body normal z < -0.2 in the dancecap world), spine1 (J3) .. neck (J12) +
    ytop, within 22 cm of the spine sideways."""
    m = np.asarray(mean_unity, float)
    return ((np.asarray(normal_z_world) < -0.2) & (m[:, 1] > J[3, 1]) & (m[:, 1] < J[12, 1] + ytop)
            & (np.abs(m[:, 0] - J[9, 0]) < 0.22))


def under_mask(mean_unity: np.ndarray, J: np.ndarray, normal_z_world: np.ndarray, lab: np.ndarray, unbound: np.ndarray,
               rigid: np.ndarray, hair_byte: np.ndarray, paint: np.ndarray, ytop: float, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """What the faded paint leaves showing on the torso back (back_zone, unbound, not rigid, not paint, within near_paint_m
    of a paint Gaussian): GREY = not hair-flagged, C* < max_chroma, L* in l_range (LHM++'s guess of the top under the hair);
    DARK = L* < dark_l and C* < dark_chroma (brown-black streaks beside the shoulder blades, hair-flagged or not).
    -> (grey, dark) masks."""
    from scipy.spatial import cKDTree
    m = np.asarray(mean_unity, float)
    n = len(m)
    L, C = lab[:, 0], np.hypot(lab[:, 1], lab[:, 2])
    if not np.any(paint):
        return np.zeros(n, bool), np.zeros(n, bool)
    d, _ = cKDTree(m[paint]).query(m, k=1)
    base = back_zone(m, J, normal_z_world, ytop) & unbound & ~rigid & ~paint & (d <= cfg["near_paint_m"])
    grey = base & ~np.asarray(hair_byte, bool) & (C < cfg["max_chroma"]) & (L >= cfg["l_range"][0]) & (L <= cfg["l_range"][1])
    dark = base & ~grey & (L < cfg["dark_l"]) & (C < cfg["dark_chroma"])
    return grey, dark


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o])
    return float(v[o][min(int(np.searchsorted(cw, 0.5 * cw[-1])), len(v) - 1)])


def under_tint(mean: np.ndarray, lab: np.ndarray, opacity: np.ndarray, targets: np.ndarray, red: np.ndarray,
               cfg: dict) -> tuple[np.ndarray, dict]:
    """The local colour of the top for each target: the red Gaussians (mask `red`, targets excluded) within the first of
    cfg radii_m that holds >= min_red of them; L*, a*, b* = their opacity-weighted medians (Lr, ar, br); the target keeps
    its shading: L' = Lr * clip(L / Lt, l_ratio_clip), Lt = the targets' opacity-weighted median L*; a*, b* scaled by L'/Lr
    (the red's hue and saturation at that lightness). Targets with fewer red neighbours at the largest radius keep their
    colour. -> (lab with the targets recoloured, info)"""
    from scipy.spatial import cKDTree
    lab = np.asarray(lab, float)
    out = lab.copy()
    ti = np.nonzero(np.asarray(targets, bool))[0]
    ri = np.nonzero(np.asarray(red, bool) & ~np.asarray(targets, bool))[0]
    info = {"targets": int(len(ti)), "red_reference": int(len(ri)), "tinted": 0, "kept_no_red": 0,
            "radius_used_m": {str(r): 0 for r in cfg["radii_m"]}}
    if not len(ti) or not len(ri):
        info["kept_no_red"] = int(len(ti))
        return out, info
    m = np.asarray(mean, float)
    op = np.asarray(opacity, float)
    tree = cKDTree(m[ri])
    Lt = _wmedian(lab[ti, 0], np.maximum(op[ti], 1e-6))
    lo, hi = cfg["l_ratio_clip"]
    for i in ti:
        nb = []
        for r in cfg["radii_m"]:
            nb = tree.query_ball_point(m[i], r)
            if len(nb) >= cfg["min_red"]:
                info["radius_used_m"][str(r)] += 1
                break
        if len(nb) < cfg["min_red"]:
            info["kept_no_red"] += 1
            continue
        ii = ri[np.asarray(nb, int)]
        w = np.maximum(op[ii], 1e-6)
        Lr, ar, br = (_wmedian(lab[ii, c], w) for c in range(3))
        Ln = Lr * float(np.clip(lab[i, 0] / max(Lt, 1e-6), lo, hi))
        f = Ln / max(Lr, 1e-6)
        out[i] = (Ln, ar * f, br * f)
        info["tinted"] += 1
    t = out[ti]
    info["L_target_wmedian_before"] = round(Lt, 2)
    info["lab_median_before"] = [round(float(v), 2) for v in np.median(lab[ti], 0)]
    info["lab_median_after"] = [round(float(v), 2) for v in np.median(t, 0)]
    info["chroma_median_after"] = round(float(np.median(np.hypot(t[:, 1], t[:, 2]))), 2)
    return out, info


def crown_tone(lab: np.ndarray, opacity: np.ndarray, el: np.ndarray, az: np.ndarray, hairline: np.ndarray,
               candidates: np.ndarray, bound: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """The crown's luminance toward the hanging (bound) hair's. candidates = unbound scalp hair (hair byte, head joints);
    the crown = candidates with elevation > el_from, outside the face box (|az| < face_az and el < hairline +
    face_above_hairline) and chroma < max_chroma; weight w = smoothstep(el_from .. el_full). L' = min(L * s, p90 of the
    bound hair's L*), s = opacity-weighted mean L* of the bound hair / of the full-weight crown; L = (1 - w) L + w min(L, L');
    a*, b* scaled with L (hue kept). -> (lab, w, info)"""
    lab = np.asarray(lab, float)
    L, C = lab[:, 0], np.hypot(lab[:, 1], lab[:, 2])
    face = (np.abs(az) < cfg["face_az"]) & (el < hairline + cfg["face_above_hairline"])
    crown = candidates & (el > cfg["el_from"]) & ~face & (C < cfg["max_chroma"])
    x = np.clip((el - cfg["el_from"]) / (cfg["el_full"] - cfg["el_from"]), 0.0, 1.0)
    w = np.where(crown, x * x * (3 - 2 * x), 0.0)
    full = crown & (w >= 1.0)
    ob = opacity[bound]
    lb_mean = float((L[bound] * ob).sum() / max(ob.sum(), 1e-9))
    lb_p90 = float(np.percentile(L[bound], 90)) if bound.any() else float("nan")
    of = opacity[full]
    lc_mean = float((L[full] * of).sum() / max(of.sum(), 1e-9)) if full.any() else float("nan")
    s = lb_mean / lc_mean if full.any() and lc_mean > 0 else 1.0
    Ln = np.where(crown, (1 - w) * L + w * np.minimum(L, np.minimum(L * s, lb_p90)), L)
    f = np.where(L > 1e-6, Ln / np.maximum(L, 1e-6), 1.0)
    out = lab.copy()
    out[:, 0] = Ln
    out[:, 1:] *= f[:, None]
    oc = opacity[crown]
    info = {"crown": int(crown.sum()), "full_weight": int(full.sum()), "scale": round(s, 4),
            "bound_L_wmean": round(lb_mean, 2), "bound_L_p90": round(lb_p90, 2), "crown_L_wmean_full": round(lc_mean, 2),
            "crown_L_wmean_before": round(float((L[crown] * oc).sum() / max(oc.sum(), 1e-9)), 2) if crown.any() else None,
            "crown_L_wmean_after": round(float((Ln[crown] * oc).sum() / max(oc.sum(), 1e-9)), 2) if crown.any() else None}
    return out, w, info


def lab_to_sh0(lab: np.ndarray) -> np.ndarray:
    """CIE Lab (cv2 float convention, L* 0..100) -> the record's sh0 (DC term: rgb = 0.5 + C0 sh0, sRGB 0..1)."""
    import cv2
    if len(lab) == 0:
        return np.zeros((0, 3))
    rgb = cv2.cvtColor(np.asarray(lab, np.float32)[None], cv2.COLOR_LAB2RGB)[0].astype(np.float64)
    return (np.clip(rgb, 0.0, 1.0) - 0.5) / SH.SH_C0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def apply_look(out: np.ndarray, idx: np.ndarray, b: dict, rest: np.ndarray, J: np.ndarray, lab: np.ndarray,
               rigid: np.ndarray, sel: dict, look: dict) -> tuple[np.ndarray, dict]:
    """The black-hair look fixes (HAIR_LOOK_BLACK) on the bound records `out` (in place; the binding itself - strand,
    segment, local offset - is unchanged, so every posed centre is the same as without them):
    1. needle_shape on the bound hair (rotation, log scale; local_rotation re-expressed in its segment frame);
    2. stack_cap on the bound hair's opacity;
    3. paint_mask (torso-back hair paint) opacity x paint_fade;
    3b. under_tint (fix round 2): the grey / dark layer the faded paint exposes, and the paint itself, take the top's local
        red (sh0; opacity unchanged);
    4. crown_tone on the unbound scalp hair (sh0).
    Only the changed records are written. -> (canonical rotations of the bound Gaussians, info)"""
    from .. import hair_groom as HG
    info: dict = {"params": look}
    op0 = _sigmoid(out["opacity"])
    S0 = np.exp(out["log_scale"].astype(np.float64))
    big0 = (S0.max(1) > 0.008) & (op0 > 0.5)
    s_, k_ = b["strand"], b["segment"]
    F = segment_frames(rest)[s_, k_]
    q0 = out["rotation"][idx].astype(np.float64)
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
    q2, ls2 = needle_shape(q0, out["log_scale"][idx], rest[s_, k_ + 1] - rest[s_, k_], look["needle_across_m"],
                           look["needle_along_max_m"])
    out["rotation"][idx] = q2.astype(np.float32)
    out["log_scale"][idx] = ls2.astype(np.float32)
    out["local_rotation"][idx] = q_mul(q_conj(q_from_mat(F)), q2).astype(np.float32)
    S2 = np.exp(ls2)
    info["needle"] = {"bound": int(len(idx))}
    if len(idx):
        info["needle"].update({"along_mm_p50_p90": [round(float(v) * 1000, 1) for v in np.percentile(S2[:, 0], [50, 90])],
                               "across_mm_p50": round(float(np.median(S2[:, 1])) * 1000, 2),
                               "sigma_max_mm_p50_before": round(float(np.median(S0[idx].max(1))) * 1000, 1)})
    op2, ci = stack_cap(out["mean"][idx], op0[idx], look["stack_cap"], look["stack_link_m"])
    ch = ~np.isclose(op2, op0[idx], rtol=0, atol=1e-9)
    out["opacity"][idx[ch]] = _logit(op2[ch]).astype(np.float32)
    info["stack_cap"] = {**ci, "cap": look["stack_cap"], "link_m": look["stack_link_m"]}
    unbound = out["strand"] < 0
    f = sel["features"]
    pm = paint_mask(out["mean"], J, f["normal"][:, 2], lab, unbound, rigid, sel["hair_colour"], look["paint_ytop_m"])
    out["opacity"][pm] = _logit(op0[pm] * look["paint_fade"]).astype(np.float32)
    info["paint_fade"] = {"n": int(pm.sum()), "opacity_gt_0.3_before": int((pm & (op0 > 0.3)).sum()), "factor": look["paint_fade"],
                          "body_sd_mm_p50": round(float(np.median(f["sd"][pm])) * 1000, 1) if pm.any() else None}
    shell = f["dh"] <= look["crown"].get("shell_m", 0.0)
    cand = ((out["hair"] == 1) | shell) & unbound & np.isin(f["dom"], SH.HEAD_JOINTS)
    lab2, w, cinfo = crown_tone(lab, op0, f["el"], f["az"], HG.hairline_el(f["az"]), cand, ~unbound, look["crown"])
    cm = w > 0
    out["sh0"][cm] = lab_to_sh0(lab2[cm]).astype(np.float32)
    cinfo["not_hair_flagged"] = int((cm & (out["hair"] == 0)).sum())
    info["crown"] = cinfo
    ut = look.get("under_tint")
    if ut:
        grey, dark = under_mask(out["mean"], J, f["normal"][:, 2], lab, unbound, rigid, out["hair"] == 1, pm,
                                look["paint_ytop_m"], ut)
        grey &= ~cm
        dark &= ~cm
        C, hue = np.hypot(lab[:, 1], lab[:, 2]), np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
        red = (unbound & ~rigid & (out["hair"] == 0) & (C >= ut["red_chroma"]) & (hue >= ut["red_hue_deg"][0])
               & (hue <= ut["red_hue_deg"][1]) & ~pm & ~grey & ~dark & ~cm)
        groups = [("grey", grey), ("dark", dark)] + ([("paint", pm & ~cm)] if ut.get("tint_paint") else [])
        tinfo = {"red_reference_rule": f"unbound, not hair-flagged, C* >= {ut['red_chroma']}, hue {ut['red_hue_deg']} deg"}
        for name, mk in groups:
            lab3, ti = under_tint(out["mean"], lab, op0, mk, red, ut)
            done = mk & np.any(lab3 != lab, axis=1)
            if done.any():
                out["sh0"][done] = lab_to_sh0(lab3[done]).astype(np.float32)
            ti["opacity_median"] = round(float(np.median(op0[mk])), 3) if mk.any() else None
            tinfo[name] = ti
        info["under_tint"] = tinfo
    op3 = _sigmoid(out["opacity"])
    S3 = np.exp(out["log_scale"].astype(np.float64))
    bm = np.zeros(len(out), bool)
    bm[idx] = True
    info["large_opaque_bound"] = {"before": int((big0 & bm).sum()),
                                  "after": int(((S3.max(1) > 0.008) & (op3 > 0.5) & bm).sum()),
                                  "rule": "sigma_max > 8 mm and opacity > 0.5"}
    return q2, info


# ================================================================================================ stage
def select(src_splatav: Path, header_json: dict) -> dict:
    g = SH.load_gaussians(src_splatav)
    rest = SH.RestBody(header_json["betas"])
    sel = SH.select_hair(g, rest, "loose")
    return {"g": g, "rest": rest, "sel": sel}


def run_bind(dump_path: Path, install: bool = False, src: Path | None = None) -> dict:
    t0 = time.time()
    OUT, UNITY_DIR, GROOM, WORK = HP.out, HP.unity_dir, HP.groom, HP.work
    src = Path(src) if src else HP.src / "follow.splatav"
    OUT.mkdir(parents=True, exist_ok=True)
    hjson = json.loads(src.with_suffix(".json").read_text())
    assert hjson.get("role") == "follow", hjson.get("role")
    head, rec, tail = read_splatav(src)
    assert (rec["strand"] < 0).all(), "the source is already bound: bind from the recap_splatav export (lhmpp/)"
    dump = load_dump(dump_path)
    rest = dump["rest"]
    J = np.asarray(hjson["rest_joints"], float)
    s = select(src, hjson)
    sel = s["sel"]
    free, rigid = sel["free"], sel["rigid"]
    lab = SH._lab(s["g"]["rgb"])
    fringe, fcount = fringe_mask(lab, free, sel.get("hair_colour"), HP.fringe)
    # reuse check: the other hair workflow's saved selection of the same file, if present, must agree
    prev = WORK / "splat_hair" / "follow_hair.npz"
    reuse = None
    if prev.exists():
        z = np.load(prev)
        reuse = {"file": _ws(prev), "free_equal": bool(np.array_equal(np.sort(z["free"]), np.nonzero(free)[0])),
                 "rigid_equal": bool(np.array_equal(np.sort(z["rigid"]), np.nonzero(rigid)[0]))}
    idx = np.nonzero(free & ~fringe)[0]
    c = rec["mean"][idx].astype(np.float64) - J[HEAD]
    q = rec["rotation"][idx].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    b = bind(c, q, rest)
    exporter_hair = rec["hair"] == 1
    out = rec.copy()
    out["hair"] = (exporter_hair | free | rigid).astype(np.uint8)
    out["strand"] = -1
    out["segment"] = -1
    out["local_offset"] = 0
    out["local_rotation"] = 0
    out["strand"][idx] = b["strand"]
    out["segment"][idx] = b["segment"]
    out["local_offset"][idx] = b["local_offset"].astype(np.float32)
    out["local_rotation"][idx] = b["local_rotation"].astype(np.float32)
    look_info = None
    if HP.look:      # the black-hair look fixes (apply_look): rotations / scales / opacities / crown colour; binding unchanged
        q, look_info = apply_look(out, idx, b, rest, J, lab, rigid, sel, HP.look)
    # round trip in float32 (what Unity reads): canonical centre / rotation back from the binding
    F = segment_frames(rest)[b["strand"], b["segment"]]
    c_back = rest[b["strand"], b["segment"]] + np.einsum("nij,nj->ni", F, out["local_offset"][idx].astype(np.float64))
    q_back = q_mul(q_from_mat(F), out["local_rotation"][idx].astype(np.float64))
    rt_mm = float(np.abs(c_back - c).max() * 1000)
    rt_deg = float(np.degrees(2 * np.arccos(np.clip(np.abs((q_back * q).sum(1)), 0, 1))).max())
    dst = OUT / "follow.splatav"
    dst.write_bytes(head + out.tobytes() + tail)
    seg_n = rest.shape[1] - 1
    per_guide = np.bincount(b["strand"], minlength=rest.shape[0])
    a = attach_weight(c[:, 1])
    d_mm = b["distance"] * 1000
    groom_sha = _sha256(GROOM) if GROOM.exists() else None
    info = {
        "selection": "dancecap.mono4d.splat_hair.select_hair style 'loose' (scalp RIGID keeps LHM++ LBS; FREE = hanging "
                     "hair + the hair layer painted on the body under it) on this file; hair byte = exporter flag OR rigid OR free",
        "exporter_selection": hjson.get("hair"),
        "select_info": sel["info"],
        "reuse_check": reuse,
        "hair_count": int(out["hair"].sum()), "bound": int(len(idx)), "rigid_scalp": int(rigid.sum()),
        "free_selected": int(free.sum()), **fcount,
        "bright_l": BRIGHT_L, **({"grey_a": GREY_A, "grey_dark_l": GREY_DARK_L} if HP.fringe == "grey" else
                                 {"fringe_rule": HP.fringe,
                                  "hair_colour_model": "splat_hair.select_hair hair_colour (crown vs face + torso front, "
                                                       "Mahalanobis in Lab)"}),
        "exporter_flag": int(exporter_hair.sum()), "exporter_only": int((exporter_hair & ~free & ~rigid).sum()),
        "binding": "unity_hair_strands",
        "binding_info": {
            "tool": "dancecap.mono4d.recap_splat_hair bind",
            "source": f"head-movement HairStrands rest guides (unity command hm_hair --action dump, Play mode, {HP.capture})",
            "dump": _ws(dump_path), "groom": _ws(GROOM), "groom_sha256": groom_sha,
            "guides": int(rest.shape[0]), "points": int(rest.shape[1]), "rest_sha256": rest_sha(rest),
            "rest_space": "head-bone space: Unity coordinates relative to the head joint (15), T-pose axes; the Gaussian "
                          "side is the canonical mean minus rest_joints[15] of this file",
            "segment_frame": "columns X, Y, Z: Y = unit tangent point s -> s+1, Z = Unity -Z (her forward; +Z of the "
                             "SMPL-X head joint mirrored) made orthogonal to Y (Unity +Y when |Y . fwd| > 0.999), X = Y x Z",
            "posing": "docs/SPLAT_AVATAR_FORMAT.md section 6 (re-anchored to the splat head, along-strand neighbour "
                      "blend, attach blend at the top)",
            "attach": ATTACH,
            "lever": LEVER,
            "rest_points": np.round(rest.reshape(-1, 3), 6).tolist(),
            "distance_mm": {"p50": round(float(np.median(d_mm)), 2), "p90": round(float(np.percentile(d_mm, 90)), 2),
                            "p99": round(float(np.percentile(d_mm, 99)), 2), "max": round(float(d_mm.max()), 2)},
            "segments_used": int(len(np.unique(b["strand"] * seg_n + b["segment"]))),
            "guides_used": int((per_guide > 0).sum()), "per_guide_max": int(per_guide.max()),
            "segment_histogram": np.bincount(b["segment"], minlength=seg_n).tolist(),
            "u_range": [round(float(b["u"].min()), 3), round(float(b["u"].max()), 3)],
            "attach_weight": {"rigid_0": int((a <= 0).sum()), "partial": int(((a > 0) & (a < 1)).sum()), "full_1": int((a >= 1).sum())},
            "round_trip_f32": {"centre_mm_max": round(rt_mm, 5), "rotation_deg_max": round(rt_deg, 5)},
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    if look_info:
        info["look"] = look_info
    js = dict(hjson)
    js["hair"] = info
    js["hair_count"] = int(out["hair"].sum())
    js["data_bytes"] = dst.stat().st_size
    js["data_sha256"] = _sha256(dst)
    js.setdefault("provenance", {})["hair_stage"] = {"tool": "dancecap.mono4d.recap_splat_hair", "source_splatav": _ws(src),
                                                     "source_sha256": _sha256(src), "created": info["binding_info"]["created"]}
    (OUT / "follow.json").write_text(json.dumps(js, indent=1))
    np.savez_compressed(OUT / "binding.npz", index=idx.astype(np.int32), c=c, q=q, distance=b["distance"], u=b["u"],
                        rigid=np.nonzero(rigid)[0].astype(np.int32), rest=rest)
    rep = {k: v for k, v in info.items() if k not in ("exporter_selection",)}
    rep["binding_info"] = {k: v for k, v in info["binding_info"].items() if k != "rest_points"}
    rep["seconds"] = round(time.time() - t0, 1)
    rep["out"] = [_ws(dst), _ws(OUT / "follow.json")]
    if install:
        UNITY_DIR.mkdir(parents=True, exist_ok=True)
        cur = UNITY_DIR / "follow.splatav"
        if cur.exists() and _sha256(cur) == _sha256(src):
            log("StreamingAssets held the unbound export (same sha as lhmpp/): replacing it")
        shutil.copyfile(dst, UNITY_DIR / "follow.splatav")
        shutil.copyfile(OUT / "follow.json", UNITY_DIR / "follow.json")
        rep["installed"] = [_ws(UNITY_DIR / "follow.splatav"), _ws(UNITY_DIR / "follow.json")]
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    log(f"bound {len(idx)} free hair Gaussians to {info['binding_info']['segments_used']} segments of "
        f"{info['binding_info']['guides_used']} guides; distance mm {info['binding_info']['distance_mm']}; "
        f"round trip {info['binding_info']['round_trip_f32']}; reuse {reuse}")
    return rep


# ================================================================================================ LOOK
def _palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(7)                    # fixed seed: colours only
    return rng.uniform(0.15, 1.0, (n, 3))


def sheet(dumps: list[Path] | None = None, device: str = "cpu") -> list[Path]:
    """binding_sheet.jpg: canonical hair, (1) as is, (2) bound Gaussians coloured by guide, (3) + the guides, 5 azimuths;
    posed_<t>.jpg per Unity dump: head + hair (rigid scalp + bound hair posed by pose_reference with the simulated
    guides of that frame, the head held fixed: the swing relative to the head), rigid (LHM++ LBS stand-in: head-rigid)
    vs swinging, 4 azimuths around the head."""
    import cv2
    OUT = HP.out
    hjson = json.loads((OUT / "follow.json").read_text())
    _, rec, _ = read_splatav(OUT / "follow.splatav")
    z = np.load(OUT / "binding.npz")
    idx, rest = z["index"], z["rest"]
    J = np.asarray(hjson["rest_joints"], float)
    g = SH.load_gaussians(OUT / "follow.splatav")           # dancecap world (z mirrored)
    S = np.array([1.0, 1.0, -1.0])
    guides_w = (J[HEAD] + rest) * S                          # canonical guides in the dancecap world
    ctr = (J[HEAD] + np.array([0, -0.12, 0])) * S
    W, H = 400, 560
    K = np.array([[1350.0, 0, W / 2], [0, 1350.0, H / 2], [0, 0, 1]])
    pal = _palette(rest.shape[0])
    tint = g["rgb"].copy()
    tint[idx] = pal[rec["strand"][idx]]
    out = []
    rows = []
    for mode in ("plain", "guide colour", "guides"):
        cols = []
        for az in (0, 90, 150, 180, 210, 270):
            C = ctr + 1.6 * np.array([math.sin(math.radians(az)), 0.12, math.cos(math.radians(az))])
            R, t = SH.look_at(C, ctr)
            m = np.ones(len(rec), bool)
            im = SH.render_check(g["xyz"][m], g["quat"][m], g["scale"][m], g["opacity"][m],
                                 (g["rgb"] if mode == "plain" else tint)[m], R, t, K, W, H, device=device)
            im = np.ascontiguousarray(im)
            if mode == "guides":
                SH.draw_strands(im, guides_w, R, t, K, colour=(20, 20, 20), thick=1)
            cv2.putText(im, f"{mode} az{az}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cols.append(im)
        rows.append(np.concatenate(cols, 1))
    p = OUT / "binding_sheet.jpg"
    cv2.imwrite(str(p), np.concatenate(rows, 0)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    out.append(p)
    # posed previews (head + hair only, head held fixed in its canonical place)
    if dumps:
        rig = z["rigid"]
        head_sel = np.nonzero((rec["joints"][:, 0] == HEAD) | (rec["joints"][:, 0] >= 22) & (rec["joints"][:, 0] <= 24))[0]
        keep = np.union1d(head_sel, rig)
        keep = np.setdiff1d(keep, idx)
        b = {"strand": rec["strand"][idx], "segment": rec["segment"][idx],
             "local_offset": rec["local_offset"][idx].astype(np.float64), "local_rotation": rec["local_rotation"][idx].astype(np.float64)}
        for dp in dumps:
            d = load_dump(dp)
            Hs = d["head_rot"] / np.linalg.norm(d["head_rot"])
            # shown head = canonical: position J_15, rotation identity -> the sim seen from the head
            pos, rot = pose_reference(z["c"], z["q"], b, rest, d["rest"], d["pos"], d["head_pos"], Hs, J[HEAD],
                                      np.array([0.0, 0, 0, 1]))
            xyz = g["xyz"].copy()
            quat = g["quat"].copy()
            xyz[idx] = pos * S
            qq = rot.copy()
            quat[idx] = np.stack([qq[:, 3], -qq[:, 0], -qq[:, 1], qq[:, 2]], 1)   # Unity xyzw -> world wxyz
            sim_w = (J[HEAD] + q_rotate(q_conj(Hs), d["pos"].reshape(-1, 3) - d["head_pos"]).reshape(d["pos"].shape)) * S
            rows = []
            for label, X, Q, m in (("head-rigid (LBS stand-in)", g["xyz"], g["quat"], np.union1d(keep, idx)),
                                   ("swinging (Unity guides)", xyz, quat, np.union1d(keep, idx)),
                                   ("swinging + sim guides", xyz, quat, np.union1d(keep, idx))):
                cols = []
                for az in (0, 90, 180, 270):
                    C = ctr + 1.9 * np.array([math.sin(math.radians(az)), 0.12, math.cos(math.radians(az))])
                    R, t = SH.look_at(C, ctr)
                    im = np.ascontiguousarray(SH.render_check(X[m], Q[m], g["scale"][m], g["opacity"][m], g["rgb"][m], R, t, K, W, H,
                                                              device=device))
                    if label.endswith("guides"):
                        SH.draw_strands(im, sim_w, R, t, K, colour=(20, 120, 255), thick=1)
                    cv2.putText(im, f"{label} az{az}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    cols.append(im)
                rows.append(np.concatenate(cols, 1))
            p = OUT / f"posed_{Path(dp).stem.replace('unity_hair_dump_', '')}.jpg"
            cv2.imwrite(str(p), np.concatenate(rows, 0)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
            out.append(p)
    for p in out:
        log(f"wrote {_ws(p)}")
    return out


def verify(path: Path | None = None) -> dict:
    path = Path(path) if path else HP.unity_dir / "follow.splatav"
    js = json.loads(path.with_suffix(".json").read_text())
    _, rec, _ = read_splatav(path)
    ok_sha = _sha256(path) == js["data_sha256"]
    b = rec["strand"] >= 0
    bi = js["hair"]["binding_info"]
    G, P = bi["guides"], bi["points"]
    res = {"file": _ws(path), "sha256_ok": ok_sha, "count": int(len(rec)), "hair": int(rec["hair"].sum()),
           "bound": int(b.sum()), "bound_all_hair": bool((rec["hair"][b] == 1).all()),
           "strand_range": [int(rec["strand"][b].min()), int(rec["strand"][b].max())],
           "segment_range": [int(rec["segment"][b].min()), int(rec["segment"][b].max())],
           "in_range": bool((rec["strand"][b] < G).all() and (rec["segment"][b] < P - 1).all()),
           "local_rotation_unit": round(float(np.abs(np.linalg.norm(rec["local_rotation"][b], axis=1) - 1).max()), 6),
           "unbound_zero": bool((rec["segment"][~b] == -1).all())}
    log(json.dumps(res))
    return res


def _disc_view(xyz, rgb, scale, opac, R, t, K, W, H) -> np.ndarray:
    """Quick CPU LOOK render: every Gaussian with opacity >= 0.3 as an opaque disc of 1.5 sigma, far to near (not the
    splat look; recap_splatav.splat_view's rule with per-Gaussian colours)."""
    import cv2
    Xc = xyz @ R.T + t
    z = Xc[:, 2]
    keep = (z > 0.05) & (opac >= 0.3)
    uv = Xc[:, :2] / np.maximum(z, 1e-6)[:, None] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    rad = np.maximum(1, np.round(1.5 * scale.max(1) * K[0, 0] / np.maximum(z, 1e-3))).astype(int)
    img = np.full((H, W, 3), 255, np.uint8)
    col = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    for k in np.nonzero(keep)[0][np.argsort(-z[keep])]:
        c = col[k]
        cv2.circle(img, (int(uv[k, 0]), int(uv[k, 1])), int(min(rad[k], 12)), (int(c[2]), int(c[1]), int(c[0])), -1)
    return img


def selection_sheet(g: dict, J: np.ndarray, bound: np.ndarray, fringe: np.ndarray, rigid: np.ndarray, path: Path) -> Path:
    """LOOK at a hair selection (canonical pose, head and upper body only): rows = as is / bound to the strands GREEN, kept
    on LBS (fringe rule) RED, rigid scalp BLUE / the bound Gaussians alone; columns = 4 azimuths around the head."""
    import cv2
    x = g["xyz"]
    head = J[HEAD] * np.array([1.0, 1.0, -1.0])
    near = np.linalg.norm(x - head, axis=1) < 0.75
    ctr = head + np.array([0.0, -0.22, 0.0])
    W, H = 360, 480
    K = np.array([[900.0, 0, W / 2], [0, 900.0, H / 2], [0, 0, 1]])
    tint = g["rgb"].copy()
    for m, c in ((bound, (0.1, 0.85, 0.1)), (fringe, (0.95, 0.1, 0.1)), (rigid, (0.1, 0.3, 0.95))):
        tint[m] = 0.3 * tint[m] + 0.7 * np.array(c)
    rows = []
    for mode in ("plain", "tint", "bound"):
        m = near & (bound if mode == "bound" else np.ones(len(x), bool))
        cols = []
        for az in (0, 90, 180, 270):
            C = ctr + 1.6 * np.array([math.sin(math.radians(az)), 0.1, math.cos(math.radians(az))])
            R, t = SH.look_at(C, ctr)
            im = _disc_view(x[m], (tint if mode == "tint" else g["rgb"])[m], g["scale"][m], g["opacity"][m], R, t, K, W, H)
            cv2.putText(im, f"{mode} az{az}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cols.append(im)
        rows.append(np.concatenate(cols, 1))
    path = Path(path)
    cv2.imwrite(str(path), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])     # _disc_view draws BGR
    return path


def select_report(src: Path | None = None) -> dict:
    """Before any Unity dump: the hair selection of the export (rigid scalp / free / fringe kept LBS by the profile's
    rule) with counts -> <out>/select_report.json, and a quick LOOK sheet select_sheet.jpg (selection_sheet)."""
    src = Path(src) if src else HP.src / "follow.splatav"
    hjson = json.loads(src.with_suffix(".json").read_text())
    s = select(src, hjson)
    sel = s["sel"]
    lab = SH._lab(s["g"]["rgb"])
    fringe, fcount = fringe_mask(lab, sel["free"], sel.get("hair_colour"), HP.fringe)
    HP.out.mkdir(parents=True, exist_ok=True)
    bound = sel["free"] & ~fringe
    rep = {"take": HP.take, "source": _ws(src), "fringe_rule": HP.fringe, "rigid_scalp": int(sel["rigid"].sum()),
           "free_selected": int(sel["free"].sum()), **fcount, "would_bind": int(bound.sum()),
           "hair_byte": int((sel["free"] | sel["rigid"]).sum()),
           "bound_lab_mean": [round(float(v), 2) for v in lab[bound].mean(0)] if bound.any() else None,
           "fringe_lab_mean": [round(float(v), 2) for v in lab[fringe].mean(0)] if fringe.any() else None,
           "select_info": sel["info"]}
    (HP.out / "select_report.json").write_text(json.dumps(rep, indent=1, default=float))
    try:
        p = selection_sheet(s["g"], np.asarray(hjson["rest_joints"], float), bound, fringe, sel["rigid"],
                            HP.out / "select_sheet.jpg")
        rep["sheet"] = _ws(p)
    except Exception as e:  # noqa: BLE001 - a sheet is a LOOK aid; the counts stand without it
        rep["sheet_error"] = f"{type(e).__name__}: {e}"
    (HP.out / "select_report.json").write_text(json.dumps(rep, indent=1, default=float))
    log(f"select: rigid {rep['rigid_scalp']}, free {rep['free_selected']}, would bind {rep['would_bind']}, {fcount}")
    return rep


QUEST_WORK = {"carlos_aline": "work/quest_distill_ca_hair"}     # the follow's Quest re-distill run folder after the bind
QUEST_LESSON = {"carlos_aline": "carlos_aline", "larissa_kadu_recap": "recap"}


def finish(dump: Path, quest_max_cost: float = 1.0, device: str = "cpu") -> dict:
    """After the Unity stage's `unity command hm_hair --action dump` (Play mode, the take's capture loaded): the whole hair
    stage in one call. The dump is copied to <out>/unity_hair_dump_frame0.json (kept with the binding), then bind +
    install into StreamingAssets/<capture>/splats/follow.*, verify (records, sha, binding round trip) and the binding
    sheet (LOOK). Returns / writes <out>/finish.json with the follow's Quest re-distill commands (the distilled sets copy
    the strand binding from the bound full avatar: they must be made again from the bound file)."""
    dump = Path(dump)
    HP.out.mkdir(parents=True, exist_ok=True)
    keep = HP.out / "unity_hair_dump_frame0.json"
    if dump.resolve() != keep.resolve():
        shutil.copyfile(dump, keep)
    rep = run_bind(keep, install=True)
    ver = verify()
    sheets = [_ws(p) for p in sheet(sorted(HP.out.glob("unity_hair_dump_t*.json")), device=device)]
    lesson = QUEST_LESSON.get(HP.take, HP.take)
    qw = QUEST_WORK.get(HP.take, f"work/quest_distill_{HP.take}_hair")
    nxt = [f"python -m dancecap.mono4d.quest_distill prepare --lessons {lesson} --roles follow --work {qw}",
           f"python -m dancecap remote run --jobs {qw}/pod_job.json --max-cost {quest_max_cost}",
           f"python -m dancecap.mono4d.quest_distill import --work {qw}"]
    out = {"take": HP.take, "dump": _ws(dump), "kept_dump": _ws(keep), "bind": {k: rep[k] for k in ("bound", "out", "installed")
                                                                                 if k in rep},
           "binding_info": rep.get("binding_info"), "verify": ver, "sheets": sheets,
           "ok": bool(ver["sha256_ok"] and ver["bound"] > 0 and ver["in_range"] and ver["unbound_zero"]),
           "next_quest_redistill": nxt}
    (HP.out / "finish.json").write_text(json.dumps(out, indent=1, default=str))
    log(f"finish: ok {out['ok']}, bound {ver['bound']}; next (from dancecap/):\n  " + "\n  ".join(nxt))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.recap_splat_hair")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--take", default=None, help="hair profile (default: the class recap)")
    common.add_argument("--out-dir", default=None, help="write work/<take>/splat_hair_unity outputs here instead (dry runs)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("bind", parents=[common])
    a.add_argument("--dump", default=None, help="default <out>/unity_hair_dump_frame0.json")
    a.add_argument("--install", action="store_true")
    a = sub.add_parser("sheet", parents=[common])
    a.add_argument("--dumps", nargs="*", default=None)
    a.add_argument("--device", default="cpu", help="torch device of the LOOK renders (cuda: seconds instead of minutes)")
    sub.add_parser("verify", parents=[common])
    sub.add_parser("select", parents=[common])
    a = sub.add_parser("finish", parents=[common])
    a.add_argument("--dump", required=True, help="the hm_hair --action dump file (its returned path)")
    a.add_argument("--quest-max-cost", type=float, default=1.0,
                   help="the printed re-distill session cap (the worst case of a 68-min A40 session is about $0.67)")
    a.add_argument("--device", default="cpu", help="torch device of the binding sheet renders (cuda: seconds instead of minutes)")
    args = ap.parse_args(argv)
    use(args.take, **({"out_dir": args.out_dir} if args.out_dir else {}))
    if args.cmd == "bind":
        run_bind(Path(args.dump) if args.dump else HP.out / "unity_hair_dump_frame0.json", install=args.install)
    elif args.cmd == "sheet":
        d = args.dumps if args.dumps is not None else sorted(str(p) for p in HP.out.glob("unity_hair_dump_t*.json"))
        sheet([Path(p) for p in d], device=args.device)
    elif args.cmd == "verify":
        verify()
    elif args.cmd == "select":
        select_report()
    elif args.cmd == "finish":
        finish(Path(args.dump), args.quest_max_cost, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
