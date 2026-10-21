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

    python -m dancecap.mono4d.recap_splat_hair bind     [--dump unity_hair_dump_frame0.json] [--install]
    python -m dancecap.mono4d.recap_splat_hair sheet    # LOOK: binding sheet (strand colours + guides) and posed previews
    python -m dancecap.mono4d.recap_splat_hair verify   # re-read the installed file: records, sha, binding round trip

Inputs: work/larissa_kadu_recap/lhmpp/follow.splatav + .json (the recap-avatars export, unbound; never modified here) and
a dump of the Unity guides (`unity command hm_hair --action dump` in Play mode with 06_LarissaKaduRecap loaded:
`rest` = the settled rest shape in HEAD-BONE space, i.e. relative to the avatar skin's head joint 15, axes = Unity axes in
the T-pose). Output: work/larissa_kadu_recap/splat_hair_unity/follow.splatav + .json (bound), sheets, report.json;
`--install` copies the pair to head-movement/Assets/StreamingAssets/06_LarissaKaduRecap/splats/ (git-ignored likeness data).

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
import hashlib
import json
import math
import shutil
import time
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


# ================================================================================================ stage
def select(src_splatav: Path, header_json: dict) -> dict:
    g = SH.load_gaussians(src_splatav)
    rest = SH.RestBody(header_json["betas"])
    sel = SH.select_hair(g, rest, "loose")
    return {"g": g, "rest": rest, "sel": sel}


def run_bind(dump_path: Path, install: bool = False, src: Path = SRC / "follow.splatav") -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    hjson = json.loads(src.with_suffix(".json").read_text())
    assert hjson.get("role") == "follow", hjson.get("role")
    head, rec, tail = read_splatav(src)
    assert (rec["strand"] < 0).all(), "the source is already bound: bind from the recap-avatars export (lhmpp/)"
    dump = load_dump(dump_path)
    rest = dump["rest"]
    J = np.asarray(hjson["rest_joints"], float)
    s = select(src, hjson)
    sel = s["sel"]
    free, rigid = sel["free"], sel["rigid"]
    lab = SH._lab(s["g"]["rgb"])
    light = lab[:, 0] > BRIGHT_L
    grey = (lab[:, 1] < GREY_A) & (lab[:, 0] > GREY_DARK_L)
    fringe = free & (light | grey)
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
        "free_selected": int(free.sum()), "free_light_fringe_kept_lbs": int((free & light).sum()),
        "free_grey_kept_lbs": int((free & grey & ~light).sum()), "bright_l": BRIGHT_L, "grey_a": GREY_A, "grey_dark_l": GREY_DARK_L,
        "exporter_flag": int(exporter_hair.sum()), "exporter_only": int((exporter_hair & ~free & ~rigid).sum()),
        "binding": "unity_hair_strands",
        "binding_info": {
            "tool": "dancecap.mono4d.recap_splat_hair bind",
            "source": "head-movement HairStrands rest guides (unity command hm_hair --action dump, Play mode, 06_LarissaKaduRecap)",
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


def sheet(dumps: list[Path] | None = None) -> list[Path]:
    """binding_sheet.jpg: canonical hair, (1) as is, (2) bound Gaussians coloured by guide, (3) + the guides, 5 azimuths;
    posed_<t>.jpg per Unity dump: head + hair (rigid scalp + bound hair posed by pose_reference with the simulated
    guides of that frame, the head held fixed: the swing relative to the head), rigid (LHM++ LBS stand-in: head-rigid)
    vs swinging, 4 azimuths around the head."""
    import cv2
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
                                 (g["rgb"] if mode == "plain" else tint)[m], R, t, K, W, H)
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
                    im = np.ascontiguousarray(SH.render_check(X[m], Q[m], g["scale"][m], g["opacity"][m], g["rgb"][m], R, t, K, W, H))
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


def verify(path: Path = UNITY_DIR / "follow.splatav") -> dict:
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.recap_splat_hair")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("bind")
    a.add_argument("--dump", default=str(OUT / "unity_hair_dump_frame0.json"))
    a.add_argument("--install", action="store_true")
    a = sub.add_parser("sheet")
    a.add_argument("--dumps", nargs="*", default=None)
    sub.add_parser("verify")
    args = ap.parse_args(argv)
    if args.cmd == "bind":
        run_bind(Path(args.dump), install=args.install)
    elif args.cmd == "sheet":
        d = args.dumps if args.dumps is not None else sorted(str(p) for p in OUT.glob("unity_hair_dump_t*.json"))
        sheet([Path(p) for p in d])
    elif args.cmd == "verify":
        verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
