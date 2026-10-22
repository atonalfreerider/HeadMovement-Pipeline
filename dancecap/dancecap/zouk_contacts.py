"""Zouk contact points and collisions of the class recap (larissa_kadu_recap), v4 (2026-10-22): which hold is on, where the hands must be, and
where the arms and hands pass through each other - above all the LEADER'S RIGHT HAND.

Owner (2026-10-22): "For the hand on shoulder contact, specifically, the leader is usually in Brazilian Zouk placing their right hand on the
follower's upper left shoulder or wrapped behind their back. We need to create a system for contact points and collision detection, because a
lot of the hand errors are happening between the leader's right hand and the right forearm. We need a dance-specific solution to that that
involves collision detection."

A NEW module next to dancecap.hand_contact (the v3 contact refinement): it reuses hand_contact's fit loading, 2D observations (obs), COCO
judge, surface queries, the arm Actor of the optimiser, the speed audit and the curation, and adds what v3 did not have:

  1. a contact-point library (SITES): named sites on each body - palm, fingers, back of the hand, wrist, forearm, elbow, upper arm, top / front
     of the shoulder, shoulder blade, upper / mid back, the far side of the back, waist / side, chest, neck - each a point + outward normal in the
     frame of the joint that carries it (found on the dancer's own shaped rest mesh by ray casts), and the holds of Brazilian zouk (HOLDS) with
     the sites they put together (the leader's right palm on her upper-left back in the closed hold, wrapped behind her back in a close embrace,
     on her back from behind when she faces away, on top of her left shoulder; her left forearm on his right shoulder / upper arm; his left
     hand with her right hand);
  2. collision bodies (CAPSULES): per dancer ~70 capsules fitted to the shaped SMPL-X rest mesh (hands with every phalanx, palms, forearms,
     upper arms, shoulder girdle, torso slabs, neck, head, legs): per frame the signed clearance of every capsule pair between the dancers and
     inside one body (the hand against its own forearm with the wrist area excluded), plus the wrist's flexion / extension / deviation / twist
     against anatomical limits ("wrist folded into the forearm");
  3. the hold of the leader's right hand per frame from the evidence (hand_contact's events, the 6 phones' 2D: MAMMA's hand landmarks and
     their visibility in the phones that see her back / her side, the COCO wrists, the distance and facing of the dancers), Viterbi with a
     switching cost and a minimum length (hysteresis);
  4. the resolution: per hold window an optimisation of the leader's right arm (shoulder, elbow, wrist, fingers; hand_contact.Actor) that puts
     the palm on the hold's sites, removes the capsule collisions (hand vs every forearm / upper arm / hand / head, hand vs own forearm, the
     arm vs her torso), keeps the wrist inside its limits, with hand_contact's speed / whip / smoothness terms and its guards (MAMMA hand points,
     the independent COCO wrist / elbow - visibility-aware here, added speed and whips, exact new penetration, the curation).

    python -m dancecap.zouk_contacts capsules         # fit the capsule bodies + their error against the mesh      -> v4/contacts/capsules.json
    python -m dancecap.zouk_contacts sites            # the contact sites on each dancer's rest mesh                -> v4/contacts/sites.json
    python -m dancecap.zouk_contacts measure --fit input,v3,v4   # capsule collisions + wrist limits, every frame   -> measure_<fit>.npz
    python -m dancecap.zouk_contacts holds            # the leader's right-hand hold per frame (evidence + Viterbi) -> holds.json / holds.npz
    python -m dancecap.zouk_contacts report           # the before tables + the worst frames                        -> report_measure.json
    python -m dancecap.zouk_contacts look --t 40,106.8 # the 6 phones at those times (MAMMA hand points, COCO wrist, sites, both fits)
    python -m dancecap.zouk_contacts exact --fit v4   # (pod, GPU) exact mesh check: deepest hand point inside her per part -> exact_<fit>.npz
    python -m dancecap.zouk_contacts plan [--base v4 --out plan_pass2.json]   # the windows to refine            -> plan*.json
    python -m dancecap.zouk_contacts refine --parts 5 --tag all2 --write [--cfg '{"base": "v4", "plan": ..., "wrist_only": true}']
                                                      # (pod) the resolution, parts in parallel on one GPU, merged -> runs/<tag>/, v4/smplx_contact/
    python -m dancecap.zouk_contacts curate --tag pass2   # curation.json's vetoes (time ranges) put back to the installed motion
    python -m dancecap.zouk_contacts validate         # before / after over the whole film (input, v3, v4)          -> validate.json
    python -m dancecap.zouk_contacts before-after     # sheets: the worst leader-right-hand frames + #118 #47 #155  -> review/before_after_*.jpg
    python -m dancecap.zouk_contacts unity-motion     # v4/smplx_contact through export_unity into a SIDE capture (never installed)
                                                      #   -> work/<take>/export/06_LarissaKaduRecap_v4contacts/, v4/contacts/unity_motion/
                                                      #      (+ sha256 for its install_motion.sh, read-only comparison with the installed capture)

The 2026-10-22 run (pod jobs in v4/contacts/jobs/): pass 1 = plan.json from the installed motion (runs/all2), pass 2 = plan_pass2.json on the
pass-1 result with the 2D / COCO elbow weighted up (runs/pass2), the curation (curation.json), pass 3 = the folded wrists only
(plan_pass3.json, wrist_only, runs/pass3); every window adopted only when its guards pass.

Everything is written under work/larissa_kadu_recap/v4/ (git-ignored likeness data); the input fits are never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import hand_contact as hc

TAKE = hc.TAKE
WORK = hc.WORK
V4 = WORK / "v4"
OUT = V4 / "contacts"
FIT_OUT = V4 / "smplx_contact"
REVIEW = OUT / "review"
ROLES = hc.ROLES
SIDES = hc.SIDES
HZ = hc.HZ
SX = {"left": "L", "right": "R"}
FITS = {"input": hc.SRC_FIT, "v3": hc.FIT_OUT, "v4": FIT_OUT}

# SMPL-X joints
PELVIS, SPINE1, SPINE2, SPINE3, NECK, HEAD = 0, 3, 6, 9, 12, 15
HIP = {"left": 1, "right": 2}
KNEE = {"left": 4, "right": 5}
ANKLE = {"left": 7, "right": 8}
FOOT = {"left": 10, "right": 11}
COLLAR = {"left": 13, "right": 14}
SHOULDER = {"left": 16, "right": 17}
ELBOW = {"left": 18, "right": 19}
WRIST = {"left": 20, "right": 21}
FINGERS = ("index", "middle", "pinky", "ring", "thumb")             # SMPL-X order of the 15 finger joints per hand
TIP_OF = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}  # smplx extra-joint tips: left 0..4, right 5..9


def finger_joint(side: str, finger: str, q: int) -> int:
    """SMPL-X joint of phalanx q (1..3) of a finger"""
    return (25 if side == "left" else 40) + 3 * FINGERS.index(finger) + (q - 1)


def tip_index(side: str, finger: str) -> int:
    return TIP_OF[finger] + (0 if side == "left" else 5)


def log(*a) -> None:
    print("[zouk_contacts]", *a, flush=True)


write_json = hc.write_json
py_json = hc.py_json


def load_fit(name_or_dir) -> dict:
    d = FITS.get(name_or_dir, name_or_dir)
    return hc.load_fit(Path(d))


# ------------------------------------------------------------------------------------------------ kinematics (numpy / torch alike)

def posed(fit_r: dict, ks: np.ndarray | None = None, pose: np.ndarray | None = None, device: str = "cpu", batch: int = 1024) -> dict:
    """FK of one dancer (hand_kin.SmplxKin, float64): {"J": (n,55,3), "R": (n,55,3,3) global rotations, "tips": (n,10,3)} at grid frames ks"""
    import torch
    kin = hc.kin_for(fit_r["betas"], device)
    tr = np.asarray(fit_r["transl"], float)
    if pose is None:
        P = hc.full_pose(fit_r)
        if ks is not None:
            P = P[ks]
    else:
        P = np.asarray(pose, float)                     # already the frames ks (or all frames when ks is None)
    if ks is not None:
        tr = tr[ks]
    n = len(P)
    J, R, tips = np.zeros((n, 55, 3)), np.zeros((n, 55, 3, 3)), np.zeros((n, 10, 3))
    with torch.no_grad():
        for a in range(0, n, batch):
            o = kin.forward(torch.tensor(P[a:a + batch], dtype=torch.float64, device=device),
                            torch.tensor(tr[a:a + batch], dtype=torch.float64, device=device))
            J[a:a + batch] = o["joints"].cpu().numpy()
            R[a:a + batch] = o["R"].cpu().numpy()
            tips[a:a + batch] = o["tips"].cpu().numpy()
    return {"J": J, "R": R, "tips": tips}


def rest_body(betas: np.ndarray) -> dict:
    """the shaped rest pose (pose 0 = T pose, flat hands, palms down): verts (V,3), joints (55,3), tips (10,3), faces, dominant joint (V,)"""
    import torch
    kin = hc.full_kin(betas, "cpu")
    with torch.no_grad():
        o = kin.forward(torch.zeros((1, 55, 3), dtype=torch.float64), torch.zeros((1, 3), dtype=torch.float64), want_verts=True)
    return {"V": o["verts"][0].numpy(), "J": o["joints"][0].numpy(), "tips": o["tips"][0].numpy(), "faces": kin.faces,
            "dom": kin.weights_np.argmax(1)}


# ------------------------------------------------------------------------------------------------ capsules
#
# A capsule = two points + a radius. A point is a joint ("j", j), a point between two joints ("lerp", a, b, f), a fingertip ("tip", i), or a
# point in a joint's frame ("loc", j, (x, y, z)) - the torso slabs and the head, whose surface is not along a bone. Radii (and the local points)
# are fitted to each dancer's shaped rest mesh: the vertices a capsule stands for (by their dominant skinning joint) -> the median distance to
# its axis. Rest frame: +x = the dancer's left, +y = up, +z = front.

PART_GROUPS = ("hand", "forearm", "upper_arm", "shoulder", "torso", "neck_head", "leg")


def _arm_specs(s: str) -> list:
    x = SX[s]
    sh, el, wr, co = SHOULDER[s], ELBOW[s], WRIST[s], COLLAR[s]
    out = [
        (f"girdle_{x}", ("lerp", co, sh, 0.15), ("lerp", co, sh, 0.92), f"shoulder_{x}", [co]),
        (f"upper_arm_prox_{x}", ("lerp", sh, el, 0.10), ("lerp", sh, el, 0.50), f"upper_arm_{x}", [sh]),
        (f"upper_arm_dist_{x}", ("lerp", sh, el, 0.50), ("lerp", sh, el, 0.95), f"upper_arm_{x}", [sh]),
        (f"forearm_prox_{x}", ("lerp", el, wr, 0.04), ("lerp", el, wr, 0.50), f"forearm_{x}", [el]),
        (f"forearm_dist_{x}", ("lerp", el, wr, 0.50), ("lerp", el, wr, 0.96), f"forearm_{x}", [el]),
    ]
    for f in ("index", "middle", "ring", "pinky"):
        m = finger_joint(s, f, 1)
        out.append((f"palm_{f}_{x}", ("lerp", wr, m, 0.15), ("lerp", wr, m, 0.98), f"hand_{x}", [wr]))
    for f in FINGERS:
        j1, j2, j3 = (finger_joint(s, f, q) for q in (1, 2, 3))
        out.append((f"{f}1_{x}", ("j", j1), ("j", j2), f"hand_{x}", [j1]))
        out.append((f"{f}2_{x}", ("j", j2), ("j", j3), f"hand_{x}", [j2]))
        out.append((f"{f}3_{x}", ("j", j3), ("tip", tip_index(s, f)), f"hand_{x}", [j3]))
    return out


def _leg_specs(s: str) -> list:
    x = SX[s]
    return [(f"thigh_{x}", ("lerp", HIP[s], KNEE[s], 0.12), ("lerp", HIP[s], KNEE[s], 0.95), f"leg_{x}", [HIP[s]]),
            (f"shin_{x}", ("lerp", KNEE[s], ANKLE[s], 0.05), ("lerp", KNEE[s], ANKLE[s], 0.95), f"leg_{x}", [KNEE[s]]),
            (f"foot_{x}", ("j", ANKLE[s]), ("j", FOOT[s]), f"leg_{x}", [ANKLE[s], FOOT[s]])]


TORSO_SLABS = (("pelvis", PELVIS), ("belly", SPINE1), ("ribs", SPINE2), ("chest", SPINE3))


def capsule_specs() -> list:
    """[(name, point A, point B, part, [joints whose dominant vertices it stands for])]; torso slabs / head are ("loc", ...) placeholders
    fitted per dancer (fit_capsules)"""
    out = []
    for s in SIDES:
        out += _arm_specs(s)
    for nm, j in TORSO_SLABS:
        out.append((nm, ("loc", j, None), ("loc", j, None), "torso", [j]))
    out.append(("neck", ("lerp", NECK, HEAD, 0.0), ("lerp", NECK, HEAD, 0.75), "neck_head", [NECK]))
    out.append(("head", ("loc", HEAD, None), ("loc", HEAD, None), "neck_head", [HEAD, 22, 23, 24]))
    for s in SIDES:
        out += _leg_specs(s)
    return out


def _point_rest(p, J: np.ndarray, tips: np.ndarray) -> np.ndarray:
    k = p[0]
    if k == "j":
        return J[p[1]]
    if k == "lerp":
        return J[p[1]] + p[3] * (J[p[2]] - J[p[1]])
    if k == "tip":
        return tips[p[1]]
    if k == "loc":
        return J[p[1]] + np.asarray(p[2], float)
    raise ValueError(p)


def seg_dist_np(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """points P (...,3) to segment A-B (3,) -> distance (...), parameter t (...)"""
    d = B - A
    L2 = float(d @ d)
    t = np.clip(((P - A) @ d) / max(L2, 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (A + t[..., None] * d), axis=-1), t


def _fit_slab(Vs: np.ndarray, Jj: np.ndarray, horizontal: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """a horizontal (along x) capsule standing for a torso slab's vertices, in the joint's rest frame -> (p0, p1 local, r)"""
    L = Vs - Jj
    x0, x1 = np.percentile(L[:, 0], [4, 96])
    z0, z1 = np.percentile(L[:, 2], [4, 96])
    r = 0.5 * (z1 - z0)
    cz, cy = 0.5 * (z0 + z1), float(np.median(L[:, 1]))
    a, b = x0 + r, x1 - r
    if b < a:
        a = b = 0.5 * (x0 + x1)
    p0, p1 = np.array([a, cy, cz]), np.array([b, cy, cz])
    # radius: the median distance of the slab's vertices to the axis (the vertical spread of the slab makes it a little larger than r)
    dd, _ = seg_dist_np(L, p0, p1)
    return p0, p1, float(np.median(dd))


def fit_capsules(betas: np.ndarray) -> dict:
    """the capsule body of one dancer: {"names", "parts", "A": [point spec], "B": [...], "r": (n,), "fit": per capsule error}"""
    rb = rest_body(betas)
    V, J, tips, dom = rb["V"], rb["J"], rb["tips"], rb["dom"]
    specs = capsule_specs()
    names, parts, A, B, r, fit = [], [], [], [], [], {}
    # vertices -> capsule: among the capsules standing for the vertex's dominant joint, the nearest one
    by_joint: dict = {}
    for i, sp in enumerate(specs):
        for j in sp[4]:
            by_joint.setdefault(j, []).append(i)
    # local (torso / head) capsules first
    loc = {}
    for i, (nm, pa, pb, part, js) in enumerate(specs):
        if pa[0] != "loc":
            continue
        j = pa[1]
        vs = V[np.isin(dom, js)]
        if nm == "head":
            # a vertical capsule over the head's bounding box (the face's dense inner vertices - eyes, mouth - would bias a median)
            L = vs - J[j]
            lo, hi = np.percentile(L, 3, axis=0), np.percentile(L, 97, axis=0)
            c = 0.5 * (lo + hi)
            rr = 0.25 * ((hi[0] - lo[0]) + (hi[2] - lo[2])) * 0.92
            h = max(0.0, 0.5 * (hi[1] - lo[1]) - rr)
            p0, p1 = c - np.array([0, h, 0]), c + np.array([0, h, 0])
        else:
            p0, p1, rr = _fit_slab(vs, J[j])
        loc[i] = (("loc", j, tuple(np.round(p0, 5))), ("loc", j, tuple(np.round(p1, 5))), rr)
    ends = []
    for i, sp in enumerate(specs):
        if i in loc:
            pa, pb, _ = loc[i]
        else:
            pa, pb = sp[1], sp[2]
        ends.append((pa, pb, _point_rest(pa, J, tips), _point_rest(pb, J, tips)))
    assign = np.full(len(V), -1)
    for j, cands in by_joint.items():
        vid = np.flatnonzero(dom == j)
        if not len(vid):
            continue
        D = np.stack([seg_dist_np(V[vid], ends[i][2], ends[i][3])[0] for i in cands], 1)
        assign[vid] = np.asarray(cands)[D.argmin(1)]
    for i, (nm, pa0, pb0, part, js) in enumerate(specs):
        pa, pb, a3, b3 = ends[i]
        vid = np.flatnonzero(assign == i)
        if i in loc:
            rr = loc[i][2]
            d, _ = seg_dist_np(V[vid], a3, b3)
        elif nm.startswith("girdle_"):
            # the shoulder girdle: its TOP surface (trapezius / acromion), by rays cast down onto the mesh at 30-80 % of the axis; below it
            # the chest slab and the upper arm stand for the body
            hs = []
            for tt in np.linspace(0.3, 0.8, 6):
                q = a3 + tt * (b3 - a3)
                hit, _ = _ray_hit(V, rb["faces"], q + np.array([0, 0.35, 0]), np.array([0, -1.0, 0]))
                if hit is not None:
                    hs.append(hit[1] - q[1])
            d, t = seg_dist_np(V[vid], a3, b3)
            rr = float(np.median(hs)) if hs else float(np.median(d))
        else:
            d, t = seg_dist_np(V[vid], a3, b3)
            mid = (t > 0.04) & (t < 0.96)
            rr = float(np.median(d[mid] if mid.sum() >= 8 else d)) if len(d) else 0.01
        err = np.abs(d - rr) if len(vid) else np.zeros(1)
        names.append(nm)
        parts.append(part)
        A.append(pa)
        B.append(pb)
        r.append(rr)
        fit[nm] = {"verts": int(len(vid)), "r_cm": round(rr * 100, 2), "abs_err_cm_median": round(float(np.median(err)) * 100, 2),
                   "abs_err_cm_p90": round(float(np.percentile(err, 90)) * 100, 2), "length_cm": round(float(np.linalg.norm(b3 - a3)) * 100, 1)}
    return {"names": names, "parts": parts, "A": A, "B": B, "r": np.asarray(r), "fit": fit}


def _jsonable_caps(c: dict) -> dict:
    return {"names": c["names"], "parts": c["parts"], "A": [list(p) for p in c["A"]], "B": [list(p) for p in c["B"]], "r": c["r"].tolist(),
            "fit": c["fit"]}


def _caps_from_json(d: dict) -> dict:
    def tup(p):
        p = list(p)
        if p[0] == "loc":
            return ("loc", int(p[1]), tuple(float(x) for x in p[2]))
        if p[0] == "lerp":
            return ("lerp", int(p[1]), int(p[2]), float(p[3]))
        return (p[0], int(p[1]))
    return {"names": list(d["names"]), "parts": list(d["parts"]), "A": [tup(p) for p in d["A"]], "B": [tup(p) for p in d["B"]],
            "r": np.asarray(d["r"], float), "fit": d.get("fit", {})}


def capsule_bodies(fit: dict | None = None, force: bool = False) -> dict:
    """{role: capsule body} (cached in v4/contacts/capsules.json, keyed by the betas)"""
    p = OUT / "capsules.json"
    fit = fit or load_fit("input")
    key = {r: [round(float(x), 6) for x in np.asarray(fit[r]["betas"]).ravel()] for r in ROLES}
    if p.exists() and not force:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("betas") == key:
            return {r: _caps_from_json(d["bodies"][r]) for r in ROLES}
    bodies = {r: fit_capsules(fit[r]["betas"]) for r in ROLES}
    write_json(p, {"what": "capsule bodies fitted to each dancer's shaped SMPL-X rest mesh (zouk_contacts.fit_capsules): point specs, radii, "
                           "and per capsule the error of its surface against the vertices it stands for", "betas": key,
                   "bodies": {r: _jsonable_caps(b) for r, b in bodies.items()}})
    return bodies


def capsule_points(caps: dict, J, R, tips, xp=np):
    """(n,55,3) joints, (n,55,3,3) global rotations, (n,10,3) tips -> endpoints A, B (n,C,3) (numpy or torch: xp = np | torch)"""
    def pt(p):
        k = p[0]
        if k == "j":
            return J[:, p[1]]
        if k == "lerp":
            return J[:, p[1]] + p[3] * (J[:, p[2]] - J[:, p[1]])
        if k == "tip":
            return tips[:, p[1]]
        v = xp.asarray(p[2]) if xp is np else xp.tensor(p[2], dtype=J.dtype, device=J.device)
        return J[:, p[1]] + (R[:, p[1]] @ v[:, None])[..., 0]
    A = xp.stack([pt(p) for p in caps["A"]], 1)
    B = xp.stack([pt(p) for p in caps["B"]], 1)
    return A, B


def seg_seg(p1, q1, p2, q2, xp=np):
    """closest distance between segments p1-q1 and p2-q2 (broadcast over leading dims; numpy or torch, differentiable) -> (dist, s, t)"""
    eps = 1e-12
    if xp is np:
        clip = np.clip
        where = np.where
        sqrt = np.sqrt
    else:
        clip = xp.clamp
        where = xp.where
        sqrt = xp.sqrt
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    if xp is np:
        a_s, e_s = np.maximum(a, eps), np.maximum(e, eps)
        den = np.maximum(a * e - b * b, eps)
    else:
        a_s, e_s = a.clamp_min(eps), e.clamp_min(eps)
        den = (a * e - b * b).clamp_min(eps)
    s = clip((b * f - c * e) / den, 0.0, 1.0)
    s = where(a * e - b * b > 1e-10, s, s * 0.0)
    t = (b * s + f) / e_s
    s = where(t < 0, clip(-c / a_s, 0.0, 1.0), where(t > 1, clip((b - c) / a_s, 0.0, 1.0), s))
    t = clip(t, 0.0, 1.0)
    s = where(a > eps, s, s * 0.0)
    t = where(e > eps, t, t * 0.0)
    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t[..., None]
    dd = ((c1 - c2) ** 2).sum(-1)
    return sqrt(dd + 1e-14), s, t


def clearance(capsA: dict, AA, AB, capsB: dict, BA, BB, ia, ib, xp=np):
    """signed clearance (n, len(ia), len(ib)) = axis distance - rA - rB (< 0: the capsules overlap by that depth)"""
    p1, q1 = AA[:, ia][:, :, None], AB[:, ia][:, :, None]
    p2, q2 = BA[:, ib][:, None], BB[:, ib][:, None]
    d, _, _ = seg_seg(p1, q1, p2, q2, xp)
    ra = capsA["r"][ia]
    rb = capsB["r"][ib]
    if xp is not np:
        ra = xp.tensor(ra, dtype=d.dtype, device=d.device)
        rb = xp.tensor(rb, dtype=d.dtype, device=d.device)
    return d - ra[:, None] - rb[None, :]


def caps_index(caps: dict, part: str | None = None, prefix: str | None = None, names=None) -> np.ndarray:
    return np.array([i for i, (n, p) in enumerate(zip(caps["names"], caps["parts"]))
                     if (part is None or p == part) and (prefix is None or n.startswith(prefix)) and (names is None or n in names)], int)


def hand_distal_index(caps: dict, s: str) -> np.ndarray:
    """the hand capsules that can fold into their own forearm: every phalanx + the distal 60 % of the palm (palm capsules are cut below)"""
    x = SX[s]
    return np.array([i for i, n in enumerate(caps["names"]) if n.endswith(f"_{x}") and caps["parts"][i] == f"hand_{x}"], int)


# ------------------------------------------------------------------------------------------------ wrist / elbow limits

# active ranges of motion of the wrist (deg; clinical norms ~ flexion 80, extension 70, radial 20, ulnar 30) with a small allowance
WRIST_LIMITS = {"flexion": 80.0, "extension": 70.0, "radial": 25.0, "ulnar": 40.0, "twist": 90.0, "bend": 85.0}
ELBOW_LIMITS = {"flexion": 150.0, "hyperextension": 10.0, "off_hinge": 35.0}


def _rest_axes(side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rest-frame (T pose, palms down) axes of a hand: a = forearm / hand direction (elbow -> wrist), n = volar (palm side), t = radial (thumb
    side)"""
    a = np.array([1.0, 0, 0]) if side == "left" else np.array([-1.0, 0, 0])
    return a, np.array([0, -1.0, 0]), np.array([0, 0, 1.0])


def rotvec_to_mat(rv: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Ro
    return Ro.from_rotvec(np.asarray(rv, float).reshape(-1, 3)).as_matrix().reshape(np.shape(rv)[:-1] + (3, 3))


def wrist_angles(pose: np.ndarray, side: str) -> dict:
    """(n,55,3) axis-angle -> per frame degrees of the wrist's local rotation R (relative to the forearm), split the anatomical way: the
    forearm's twist (pronation / supination, about the forearm axis a) comes first, then the hand bends in its OWN frame - R = T(a, twist) F.
    The forearm axis seen from the hand, a_h = R^T a = F^T a, gives flexion (+ toward the palm, - = extension) and deviation (+ radial /
    toward the thumb, - ulnar) independently of the twist; twist = the angle of T = R F^T about a."""
    a, n, t = _rest_axes(side)
    R = rotvec_to_mat(pose[:, WRIST[side]])
    ah = np.einsum("kji,j->ki", R, a)                                # R^T a
    # the swing as a vector (bend x its direction in the flexion / deviation plane): continuous at any bend (an atan2 per axis flips past
    # 90 deg and reads a 95-deg flexion as 170 deg of deviation)
    bend_r = np.arccos(np.clip(ah @ a, -1, 1))
    sb = np.sqrt((ah @ n) ** 2 + (ah @ t) ** 2)
    kk = np.where(sb > 1e-9, bend_r / np.maximum(sb, 1e-12), 1.0)
    flex = np.degrees(-kk * (ah @ n))
    dev = np.degrees(-kk * (ah @ t))
    # F = the minimal rotation taking a_h onto a (no twist of its own); T = R F^T is a rotation about a
    ax = np.cross(ah, a)
    sn = np.linalg.norm(ax, axis=1)
    cs = ah @ a
    ang = np.arctan2(sn, cs)
    axu = ax / np.maximum(sn, 1e-12)[:, None]
    F = rotvec_to_mat(axu * ang[:, None])
    Tm = R @ np.transpose(F, (0, 2, 1))
    Tn = Tm @ n
    tw = np.degrees(np.arctan2(Tn @ np.cross(a, n), Tn @ n))
    # bend = the whole angle between the hand's axis and the forearm (flexion and deviation together); past ~90 deg the split above is
    # meaningless (a hand folded back onto its forearm reads as +-180 deg deviation)
    bend = np.degrees(ang)
    return {"flexion": flex, "deviation": dev, "twist": tw, "bend": bend}


def elbow_angles(pose: np.ndarray, side: str) -> dict:
    """the elbow's local rotation: flexion (deg, + = bending: the forearm toward the front of the upper arm in the rest frame), and the
    off-hinge part (the angle the forearm leaves the elbow's flexion plane)"""
    a, _, _ = _rest_axes(side)
    Re = rotvec_to_mat(pose[:, ELBOW[side]])
    h = Re @ a
    fwd = np.array([0, 0, 1.0])
    flex = np.degrees(np.arctan2(h @ fwd, h @ a))                 # rest: arm straight (0); bending brings the forearm forward (+z)
    up = np.cross(a, fwd)
    off = np.degrees(np.arcsin(np.clip(h @ up, -1, 1)))
    return {"flexion": flex, "off_hinge": off}


def limit_excess(ang: dict) -> dict:
    """degrees beyond the wrist limits per frame (0 inside): flexion / extension / radial / ulnar (the swing vector's parts), the whole
    bend, the twist"""
    L = WRIST_LIMITS
    out = {"flexion": np.maximum(0, ang["flexion"] - L["flexion"]), "extension": np.maximum(0, -ang["flexion"] - L["extension"]),
           "radial": np.maximum(0, ang["deviation"] - L["radial"]), "ulnar": np.maximum(0, -ang["deviation"] - L["ulnar"]),
           "twist": np.maximum(0, np.abs(ang["twist"]) - L["twist"])}
    if "bend" in ang:
        out["bend"] = np.maximum(0, ang["bend"] - L["bend"])
    return out


# ------------------------------------------------------------------------------------------------ contact sites
#
# A site = a point on the dancer's surface + its outward normal, in the frame of the joint that carries it (found on the shaped rest mesh by a
# ray cast from outside), or a point computed from the hand joints (palm, back of hand, fingers). `radius_m` = how far from the point a hand
# still counts as "on the site" (the patch).
#   ray sites: (name, joint, origin offset from that joint (rest frame), ray direction, radius)

def _ray_site_specs() -> list:
    out = []
    for s in SIDES:
        x = SX[s]
        sg = 1.0 if s == "left" else -1.0
        out += [
            (f"shoulder_top_{x}", COLLAR[s], ("shoulder_x", 0.75, "shoulder_y", 0.3, "shoulder_z", 0.0), (0, -1, 0), 0.06),
            (f"shoulder_front_{x}", COLLAR[s], ("shoulder_x", 0.80, "shoulder_y", -0.04, None, 0.5), (0, 0, -1), 0.06),
            (f"shoulder_back_{x}", COLLAR[s], ("shoulder_x", 0.80, "shoulder_y", -0.03, None, -0.5), (0, 0, 1), 0.06),
            # the shoulder blade: the upper back 10 cm off the spine, 6 cm above the spine-3 joint (hand_contact's #118 anchor)
            (f"shoulder_blade_{x}", SPINE3, (None, sg * 0.10, None, 0.06, None, -0.5), (0, 0, 1), 0.08),
            # the far side of the back (a hand wrapped behind her): mid back, 13 cm off the spine, at the spine-2 joint
            (f"back_side_{x}", SPINE2, (None, sg * 0.13, None, 0.02, None, -0.5), (0, 0, 1), 0.08),
            (f"waist_{x}", SPINE1, (None, sg * 0.5, None, 0.02, None, -0.01), (-sg, 0, 0), 0.07),
            (f"side_{x}", SPINE2, (None, sg * 0.5, None, 0.02, None, -0.01), (-sg, 0, 0), 0.07),
        ]
    # her back as a grid (x her left +, levels: shoulder blades, below the blades, the waist), for the hand-on-back holds
    for xs, xv in (("L2", 1.0), ("L1", 0.5), ("C", 0.0), ("R1", -0.5), ("R2", -1.0)):
        for lv, j, yv, wx in (("hi", SPINE3, 0.06, 0.13), ("mid", SPINE3, -0.05, 0.115), ("lo", SPINE2, -0.03, 0.085)):
            out.append((f"back_{xs}_{lv}", j, (None, xv * wx, None, yv, None, -0.5), (0, 0, 1), 0.05))
    out += [
        ("upper_back", SPINE3, (None, 0.0, None, 0.03, None, -0.5), (0, 0, 1), 0.08),
        ("mid_back", SPINE2, (None, 0.0, None, 0.0, None, -0.5), (0, 0, 1), 0.09),
        ("lower_back", SPINE1, (None, 0.0, None, -0.02, None, -0.5), (0, 0, 1), 0.09),
        ("chest", SPINE3, (None, 0.0, None, 0.0, None, 0.5), (0, 0, -1), 0.09),
        ("chest_L", SPINE3, (None, 0.08, None, 0.0, None, 0.5), (0, 0, -1), 0.07),
        ("chest_R", SPINE3, (None, -0.08, None, 0.0, None, 0.5), (0, 0, -1), 0.07),
        ("neck_back", NECK, (None, 0.0, None, 0.03, None, -0.4), (0, 0, 1), 0.05),
    ]
    return out


def _ray_hit(V: np.ndarray, F: np.ndarray, O: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """first hit of one ray with the mesh (numpy Moeller-Trumbore over all faces) -> (point, outward normal) or (None, None)"""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = B - A, C - A
    p = np.cross(D, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = O - A
    u = (tv * p).sum(1) * inv
    q = np.cross(tv, e1)
    v = (q @ D) * inv
    t = (e2 * q).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    if not hit.any():
        return None, None
    i = np.flatnonzero(hit)[np.argmin(t[hit])]
    n = np.cross(e1[i], e2[i])
    n /= np.linalg.norm(n)
    if n @ D > 0:
        n = -n
    return O + t[i] * D, n


def fit_sites(betas: np.ndarray) -> dict:
    """{name: {"joint", "p" (local, rest frame), "n" (local outward normal), "radius_m"}} for the ray sites of one dancer"""
    rb = rest_body(betas)
    V, J, F = rb["V"], rb["J"], rb["faces"]
    out = {}
    for name, j, spec, d, rad in _ray_site_specs():
        side = "left" if name.endswith("_L") else "right"
        sh = J[SHOULDER[side]]
        xs, xv, ys, yv, zs, zv = spec
        base = J[j]
        def comp(kind, val, axis):
            if kind == "shoulder_x":
                return sh[0] * val
            if kind == "shoulder_y":
                return sh[1] + val
            if kind == "shoulder_z":
                return sh[2] + val
            return base[axis] + val
        O = np.array([comp(xs, xv, 0), comp(ys, yv, 1), comp(zs, zv, 2)])
        D = np.asarray(d, float)
        P, n = _ray_hit(V, F, O - 0.0 * D, D)
        if P is None:
            continue
        out[name] = {"joint": int(j), "p": (P - J[j]).tolist(), "n": n.tolist(), "radius_m": rad, "rest_point": P.tolist()}
    return out


def site_bodies(fit: dict | None = None, force: bool = False) -> dict:
    p = OUT / "sites.json"
    fit = fit or load_fit("input")
    key = {r: [round(float(x), 6) for x in np.asarray(fit[r]["betas"]).ravel()] for r in ROLES}
    if p.exists() and not force:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("betas") == key:
            return d["sites"]
    sites = {r: fit_sites(fit[r]["betas"]) for r in ROLES}
    write_json(p, {"what": "contact sites (zouk_contacts.fit_sites): point + outward normal in the frame of the joint that carries it (rest frame "
                           "x = the dancer's left, y = up, z = front), found by a ray cast on the shaped rest mesh; radius = the patch",
                   "betas": key, "sites": sites})
    return sites


def site_world(site: dict, J, R, xp=np):
    """posed site point + normal (n,3) (numpy or torch)"""
    j = site["joint"]
    if xp is np:
        p, n = np.asarray(site["p"]), np.asarray(site["n"])
    else:
        p = xp.tensor(site["p"], dtype=J.dtype, device=J.device)
        n = xp.tensor(site["n"], dtype=J.dtype, device=J.device)
    return J[:, j] + (R[:, j] @ p[:, None])[..., 0], (R[:, j] @ n[:, None])[..., 0]


def hand_site(J, side: str, name: str):
    """sites computed from the hand joints (numpy): palm (centre + volar normal), back_of_hand, fingers (middle of the middle finger), wrist"""
    pc = hc.palm_centre(J, side)
    pn = hc.palm_normal(J, side)
    if name == "palm":
        return pc, pn
    if name == "back_of_hand":
        return pc - 0.02 * pn, -pn
    if name == "fingers":
        return J[:, finger_joint(side, "middle", 2)], pn
    if name == "wrist":
        return J[:, WRIST[side]], pn
    raise KeyError(name)


# ------------------------------------------------------------------------------------------------ zouk holds
#
# The holds of Brazilian zouk that put a hand on the partner, with the sites they join. The leader's right hand is the owner's case:
#   upper_back   closed / semi-closed hold, face to face: his right palm on her upper-left back (left shoulder blade, back of her left shoulder),
#                his arm passing her left flank; her left forearm on his right shoulder / upper arm (or round his neck); his left hand with her
#                right hand
#   wrap_back    close embrace with his right arm wrapped behind her back: the palm on her mid back / the far (right) side of her back
#   behind_back  she faces away from him (he is behind her): his right palm on her upper back / shoulder blade
#   shoulder_top open / side positions: his right hand on top / the front of her left shoulder
#   chest        his hand on her sternum / chest (a real but rare zouk contact: kept only with clear evidence)
#   hand_hold    his right hand holds one of her hands (hand_contact's hand-hand events)
#   arm          his right hand on her upper arm / forearm
#   free         no contact

HOLDS = {
    "upper_back": {"facing": (-1.0, -0.35), "sites": ["shoulder_blade_L", "shoulder_back_L", "upper_back"], "palm_to": "surface",
                   "desc": "closed / semi-closed hold: his right palm on her upper-left back (left shoulder blade / back of her left shoulder)",
                   "other_contacts": [("follow", "left", ["shoulder_top_R", "shoulder_back_R", "neck_back"], "lead"), ("lead", "left", ["hand_R"], "follow")]},
    "wrap_back": {"facing": (-1.0, -0.35), "sites": ["mid_back", "back_side_R", "shoulder_blade_R"], "palm_to": "surface",
                  "desc": "close embrace, his right arm wrapped behind her back: palm on her mid back / the far (right) side of her back"},
    "behind_back": {"facing": (0.35, 1.0), "sites": ["shoulder_blade_L", "upper_back", "shoulder_blade_R", "mid_back"], "palm_to": "surface",
                    "desc": "she faces away from him: his right palm on her upper back / shoulder blade from behind"},
    "shoulder_top": {"facing": (-1.0, 1.0), "sites": ["shoulder_top_L", "shoulder_front_L"], "palm_to": "surface",
                     "desc": "open / side positions: his right hand on top / the front of her left shoulder"},
    "chest": {"facing": (-1.0, -0.2), "sites": ["chest", "chest_L", "chest_R"], "palm_to": "surface",
              "desc": "his right palm on her sternum / chest (rare; only with clear 2D evidence)"},
    "hand_hold": {"facing": (-1.0, 1.0), "sites": [], "palm_to": "hand", "desc": "his right hand holds one of her hands"},
    "arm": {"facing": (-1.0, 1.0), "sites": [], "palm_to": "arm", "desc": "his right hand on her upper arm / forearm"},
    "free": {"facing": (-1.0, 1.0), "sites": [], "palm_to": None, "desc": "no contact"},
}
TORSO_HOLDS = ("upper_back", "wrap_back", "behind_back", "shoulder_top", "chest")
HOLD_NAMES = tuple(HOLDS)


# ------------------------------------------------------------------------------------------------ dancer frames

def torso_frame(J: np.ndarray) -> np.ndarray:
    """(n,3,3) columns = the dancer's left (horizontal), up, front, at the spine-3 joint"""
    x = J[:, SHOULDER["left"]] - J[:, SHOULDER["right"]]
    x[:, 1] = 0
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    up = np.tile(np.array([0, 1.0, 0]), (len(J), 1))
    z = np.cross(x, up)
    return np.stack([x, up, z], 2)


def in_frame(P: np.ndarray, J: np.ndarray) -> np.ndarray:
    """world points (n,3) -> the dancer's torso frame at spine 3 (x her left, y up, z her front)"""
    Fr = torso_frame(J)
    return np.einsum("kji,kj->ki", Fr, P - J[:, SPINE3])


def couple_geometry(JL: np.ndarray, JF: np.ndarray) -> dict:
    """facing (dot of the two torso fronts: -1 face to face, +1 same way) and the horizontal distance of the torso centres"""
    fl, ff = torso_frame(JL)[:, :, 2], torso_frame(JF)[:, :, 2]
    cl, cf = JL[:, [SPINE1, SPINE2, SPINE3]].mean(1), JF[:, [SPINE1, SPINE2, SPINE3]].mean(1)
    return {"facing": (fl * ff).sum(1), "dist_m": np.linalg.norm((cl - cf)[:, [0, 2]], axis=1)}


# ------------------------------------------------------------------------------------------------ measurement (capsules, every frame)

def _pair_groups(caps: dict) -> dict:
    """capsule index groups of one body: hand_L/R, hand_distal_L/R, forearm_L/R, forearm_self_L/R (the forearm minus its distal 15 %: what
    the own hand can fold into), upper_arm_L/R, shoulder_L/R, torso, neck_head, leg_L/R"""
    g = {}
    for s in SIDES:
        x = SX[s]
        g[f"hand_{x}"] = caps_index(caps, f"hand_{x}")
        g[f"forearm_{x}"] = caps_index(caps, f"forearm_{x}")
        g[f"upper_arm_{x}"] = caps_index(caps, f"upper_arm_{x}")
        g[f"shoulder_{x}"] = caps_index(caps, f"shoulder_{x}")
        g[f"leg_{x}"] = caps_index(caps, f"leg_{x}")
        # the hand against its own forearm: the fingers and the distal palm only (the palm capsules start at 15 % of wrist -> knuckle)
        g[f"fingers_{x}"] = np.array([i for i in g[f"hand_{x}"] if not caps["names"][i].startswith("palm_")], int)
    g["torso"] = caps_index(caps, "torso")
    g["neck_head"] = caps_index(caps, "neck_head")
    return g


# the partner parts a hand is checked against, and the own parts (self-collision; the own forearm only for the fingers / distal palm)
PARTNER_PARTS = ("hand_L", "hand_R", "forearm_L", "forearm_R", "upper_arm_L", "upper_arm_R", "shoulder_L", "shoulder_R", "torso", "neck_head",
                 "leg_L", "leg_R")
SELF_PARTS = ("forearm_own", "upper_arm_own", "torso", "neck_head", "hand_other", "forearm_other", "leg_L", "leg_R")


def self_forearm_clearance(caps: dict, A, B, s: str, xp=np):
    """the hand folding into its OWN forearm: fingers + the palm's distal 60 % vs the forearm's proximal 80 % (n,) min clearance (m)"""
    x = SX[s]
    hand = caps_index(caps, f"hand_{x}")
    fa = caps_index(caps, f"forearm_{x}")
    # shortened copies: palm capsules from 40 % of their length, the forearm up to 80 % of elbow -> wrist (the distal capsule's first 60 %)
    names = caps["names"]
    if xp is not np:
        hand_i, fa_i = xp.as_tensor(hand, device=A.device), xp.as_tensor(fa, device=A.device)
    else:
        hand_i, fa_i = hand, fa
    Ah, Bh = A[:, hand_i], B[:, hand_i]
    is_palm = np.array([names[i].startswith("palm_") for i in hand])
    if is_palm.any():
        if xp is np:
            m = is_palm[None, :, None]
            Ah = np.where(m, Ah + 0.4 * (Bh - Ah), Ah)
        else:
            m = xp.tensor(is_palm, device=A.device)[None, :, None]
            Ah = xp.where(m, Ah + 0.4 * (Bh - Ah), Ah)
    Af, Bf = A[:, fa_i], B[:, fa_i]
    is_dist = np.array([names[i].startswith("forearm_dist") for i in fa])
    if xp is np:
        Bf = np.where(is_dist[None, :, None], Af + 0.6 * (Bf - Af), Bf)
    else:
        Bf = xp.where(xp.tensor(is_dist, device=A.device)[None, :, None], Af + 0.6 * (Bf - Af), Bf)
    d, _, _ = seg_seg(Ah[:, :, None], Bh[:, :, None], Af[:, None], Bf[:, None], xp)
    rh, rf = caps["r"][hand], caps["r"][fa]
    if xp is not np:
        rh = xp.tensor(rh, dtype=d.dtype, device=d.device)
        rf = xp.tensor(rf, dtype=d.dtype, device=d.device)
    c = d - rh[:, None] - rf[None, :]
    return c.reshape(c.shape[0], -1).min(1) if xp is np else c.reshape(c.shape[0], -1).min(1).values


def measure_fit(fit: dict, bodies: dict | None = None, batch: int = 1500) -> dict:
    """Per frame, every hand (4): the min capsule clearance (m, < 0 = depth inside) to each partner part and each own part, the capsule pair
    that is deepest per part; the wrist / elbow angles; the arm (forearm + upper arm) against the partner's torso; the couple's facing and
    distance; the palm in the partner's torso frame. -> dict of arrays (T, ...)"""
    bodies = bodies or capsule_bodies(fit)
    T = len(fit["lead"]["t"])
    out = {}
    pz = {}
    for r in ROLES:
        P = hc.full_pose(fit[r])
        k = posed(fit[r])
        pz[r] = (P, k)
    geo = couple_geometry(pz["lead"][1]["J"], pz["follow"][1]["J"])
    out["facing"], out["dist_m"] = geo["facing"], geo["dist_m"]
    groups = {r: _pair_groups(bodies[r]) for r in ROLES}
    ends = {r: capsule_points(bodies[r], pz[r][1]["J"], pz[r][1]["R"], pz[r][1]["tips"]) for r in ROLES}
    for r in ROLES:
        o = hc.OTHER[r]
        cr, co = bodies[r], bodies[o]
        Ar, Br = ends[r]
        Ao, Bo = ends[o]
        for s in SIDES:
            x = SX[s]
            key = f"{r}_{x}"
            hi = groups[r][f"hand_{x}"]
            for part in PARTNER_PARTS:
                ib = groups[o][part]
                c = np.empty(T)
                arg = np.empty((T, 2), int)
                for a in range(0, T, batch):
                    sl = slice(a, a + batch)
                    cc = clearance(cr, Ar[sl], Br[sl], co, Ao[sl], Bo[sl], hi, ib)
                    flat = cc.reshape(len(cc), -1)
                    m = flat.argmin(1)
                    c[sl] = flat[np.arange(len(m)), m]
                    arg[sl] = np.stack([hi[m // len(ib)], ib[m % len(ib)]], 1)
                out[f"{key}_vs_{part}"] = c
                out[f"{key}_vs_{part}_pair"] = arg
            # self: the own forearm (folding), the own torso / head / other arm / legs (fingers + palm)
            out[f"{key}_self_forearm"] = np.concatenate([self_forearm_clearance(cr, Ar[a:a + batch], Br[a:a + batch], s)
                                                         for a in range(0, T, batch)])
            ox = SX["right" if s == "left" else "left"]
            for part, ib in (("self_torso", groups[r]["torso"]), ("self_neck_head", groups[r]["neck_head"]),
                             ("self_hand_other", groups[r][f"hand_{ox}"]), ("self_forearm_other", groups[r][f"forearm_{ox}"])):
                c = np.empty(T)
                for a in range(0, T, batch):
                    sl = slice(a, a + batch)
                    cc = clearance(cr, Ar[sl], Br[sl], cr, Ar[sl], Br[sl], hi, ib)
                    c[sl] = cc.reshape(len(cc), -1).min(1)
                out[f"{key}_{part}"] = c
            # the arm (forearm + upper arm) of this side against the partner's torso (an arm through her body)
            ia = np.concatenate([groups[r][f"forearm_{x}"], groups[r][f"upper_arm_{x}"]])
            c = np.empty(T)
            for a in range(0, T, batch):
                sl = slice(a, a + batch)
                cc = clearance(cr, Ar[sl], Br[sl], co, Ao[sl], Bo[sl], ia, groups[o]["torso"])
                c[sl] = cc.reshape(len(cc), -1).min(1)
            out[f"{key}_arm_vs_torso"] = c
            P, k = pz[r]
            wa = wrist_angles(P, s)
            ea = elbow_angles(P, s)
            for nm, v in wa.items():
                out[f"{key}_wrist_{nm}"] = v
            for nm, v in ea.items():
                out[f"{key}_elbow_{nm}"] = v
            pc, pn = hand_site(k["J"], s, "palm")
            out[f"{key}_palm_in_partner"] = in_frame(pc, pz[o][1]["J"])
            out[f"{key}_elbow_in_partner"] = in_frame(k["J"][:, ELBOW[s]], pz[o][1]["J"])
            out[f"{key}_palm_normal_in_partner"] = np.einsum("kji,kj->ki", torso_frame(pz[o][1]["J"]), pn)
    return out


def _runs(m: np.ndarray, merge: int = 3) -> list[tuple[int, int]]:
    """True runs of a mask as (first, last), runs separated by at most `merge` False frames joined"""
    idx = np.flatnonzero(m)
    if not len(idx):
        return []
    out = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - out[-1][1] <= merge + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(int(a), int(b)) for a, b in out]


def run_measure(name: str, fit_dir: Path | None = None) -> dict:
    """measure_fit on a named fit (input / v3 / v4 or a directory) -> v4/contacts/measure_<name>.npz"""
    t0 = time.time()
    fit = load_fit(fit_dir or name)
    bodies = capsule_bodies(load_fit("input"))
    m = measure_fit(fit, bodies)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"measure_{name}.npz", **m)
    log(f"measure {name}: {len(m)} series in {time.time() - t0:.0f} s -> {OUT / f'measure_{name}.npz'}")
    return m


def load_measure(name: str) -> dict:
    with np.load(OUT / f"measure_{name}.npz") as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------ the before tables

HAND_KEYS = ("lead_R", "lead_L", "follow_L", "follow_R")
PEN_LEVELS_CM = (0.5, 1.0, 2.0)


def _depth_stats(c: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """clearance series (m) -> frames / seconds below -0.5 / -1 / -2 cm, incidents (runs below -0.5 cm), depth p50 / p90 / max (cm)"""
    c = np.asarray(c, float)
    if mask is not None:
        c = np.where(mask, c, np.inf)
    pen = -c * 100
    m = pen > PEN_LEVELS_CM[0]
    out = {f"frames_gt_{lv}cm": int((pen > lv).sum()) for lv in PEN_LEVELS_CM}
    out["seconds_gt_0.5cm"] = round(float(m.sum()) / HZ, 1)
    out["incidents"] = len(_runs(m, 3))
    if m.any():
        out.update({"depth_cm_p50": round(float(np.median(pen[m])), 2), "depth_cm_p90": round(float(np.percentile(pen[m], 90)), 2),
                    "depth_cm_max": round(float(pen[m].max()), 2)})
    return out


def hand_collision_table(m: dict, key: str, mask: np.ndarray | None = None, caps: dict | None = None) -> dict:
    """per partner part + the own forearm (folding) / torso / other hand / other forearm: _depth_stats; the deepest capsule pairs"""
    out = {}
    for part in PARTNER_PARTS:
        c = m[f"{key}_vs_{part}"]
        st = _depth_stats(c, mask)
        if caps is not None and st["frames_gt_0.5cm"]:
            pr = m[f"{key}_vs_{part}_pair"]
            sel = (c < -0.005) & (mask if mask is not None else True)
            from collections import Counter
            cnt = Counter(f"{caps['self']['names'][a]} x {caps['other']['names'][b]}" for a, b in pr[sel])
            st["pairs"] = dict(cnt.most_common(4))
        out[f"partner_{part}"] = st
    for part in ("self_forearm", "self_torso", "self_neck_head", "self_hand_other", "self_forearm_other"):
        out[part] = _depth_stats(m[f"{key}_{part}"], mask)
    out["arm_vs_partner_torso"] = _depth_stats(m[f"{key}_arm_vs_torso"], mask)
    return out


def wrist_table(m: dict, key: str, mask: np.ndarray | None = None) -> dict:
    ang = {nm: m[f"{key}_wrist_{nm}"] for nm in ("flexion", "deviation", "twist", "bend")}
    ex = limit_excess(ang)
    sel = np.ones(len(ang["flexion"]), bool) if mask is None else mask
    out = {}
    for k, v in ex.items():
        vv = v[sel]
        out[k] = {"frames_gt_0": int((vv > 0).sum()), "frames_gt_10deg": int((vv > 10).sum()), "incidents": len(_runs(v * sel > 0, 3)),
                  "max_deg": round(float(vv.max()), 1) if len(vv) else 0.0}
    allx = np.max(np.stack(list(ex.values()), 1), 1)
    out["any"] = {"frames_gt_0": int((allx[sel] > 0).sum()), "frames_gt_10deg": int((allx[sel] > 10).sum()),
                  "seconds_gt_0": round(float((allx[sel] > 0).sum()) / HZ, 1)}
    q = {k: np.percentile(v[sel], [1, 50, 99]).round(1).tolist() for k, v in ang.items()}
    out["angles_p1_p50_p99_deg"] = q
    return out


def worst_frames(m: dict, key: str = "lead_R", n: int = 20, parts=None, min_gap_s: float = 1.5, mask: np.ndarray | None = None) -> list:
    """the n deepest collision incidents of a hand (one frame each, at least min_gap_s apart): against every forearm (hers, his own:
    folding), her upper arms / hands / head / torso, and the wrist beyond its limits (1 cm per 10 deg)"""
    parts = parts or ["partner_" + p for p in PARTNER_PARTS] + ["self_forearm", "wrist"]
    T = len(m["facing"])
    D = {}
    for p in parts:
        if p == "wrist":
            ex = limit_excess({nm: m[f"{key}_wrist_{nm}"] for nm in ("flexion", "deviation", "twist", "bend")})
            D[p] = np.minimum(np.max(np.stack(list(ex.values()), 1), 1) / 10.0, 10.0)
        elif p.startswith("partner_"):
            D[p] = -m[f"{key}_vs_{p[8:]}"] * 100
        else:
            D[p] = -m[f"{key}_{p}"] * 100
    M = np.stack([D[p] for p in parts], 1)
    if mask is not None:
        M = np.where(mask[:, None], M, -np.inf)
    best = M.max(1)
    which = np.asarray(parts)[M.argmax(1)]
    order = np.argsort(-best)
    out, taken = [], []
    for k in order:
        if best[k] <= 0.5 or len(out) >= n:
            break
        if any(abs(k - j) < min_gap_s * HZ for j in taken):
            continue
        taken.append(int(k))
        out.append({"frame": int(k), "t": round(k / HZ, 2), "what": str(which[k]), "depth_cm": round(float(best[k]), 2),
                    "all_cm": {p: round(float(D[p][k]), 2) for p in parts if D[p][k] > 0.3}})
    return out


# ------------------------------------------------------------------------------------------------ the hold of the leader's right hand
#
# hand_contact's events say WHEN his right hand touches her (mesh gap + the 6 phones' 2D); where it is, the fit often has wrong: in the closed
# hold MAMMA's fit puts the hand INSIDE her chest while the video shows it on her back (v3 review: #29, #31, #118; the v3 refinement could not
# move it: MAMMA's own hand points, hallucinated for a hidden hand, vetoed every solve). Here each torso-contact frame chooses a SITE on her
# body (shoulder blades, upper / mid back, the far side of the back, top / front / back of the shoulders, chest, sides, her arms) by what the
# phones see: per phone, is the site hidden behind her (or his) body from that phone (a ray against the capsule bodies), does MAMMA see his
# right hand there (>= min_lm of its 63 hand points with visibility > 0.5) and where (the median of those points against the site's
# projection, scaled by the hand's projected length), and the owner's COCO wrist likewise; plus the zouk prior of the holds for how the
# couple stands (facing) and a weak pull to the fitted hand. A Viterbi over the frames (a cost per change of site / of hold) is the hysteresis.

@dataclass
class HoldCfg:
    torso_near_m: float = 0.12       # the fitted palm within this (capsule clearance) of her torso / shoulders: a torso-contact frame
    couple_max_m: float = 0.85        # ... and the torso centres within this
    min_lm: int = 6                   # a phone sees his hand: >= this many of the 63 MAMMA hand points with visibility > 0.5
    vis: float = 0.5
    sigma_rel: float = 0.45           # 2D: Cauchy scale = this x the hand's projected length (the median of the visible hand points vs the site)
    p_hallucinate: float = 0.2        # MAMMA reports the hand where the site is hidden from that phone
    p_miss: float = 0.35              # MAMMA does not see a hand the site shows
    w_coco: float = 0.5               # the owner's COCO wrist (conf >= 0.5) as a second, weaker detector (scale x 1.8)
    occl_margin_m: float = 0.035      # the ray from the phone stops this short of the site (the hand lies on her surface)
    occl_r_scale: float = 0.92        # occluder capsules a little thinner than fitted (their error is ~1 cm)
    hand_len_m: float = 0.19
    palm_lift_m: float = 0.015        # the palm centre stands this far off the site (the hand's half thickness)
    reach_m: float = 0.70             # a site further than this from his right shoulder cannot hold his palm
    sigma_fit_m: float = 0.18         # weak pull to the fitted palm (Cauchy), weight w_fit
    w_fit: float = 0.6
    switch_site: float = 2.5          # Viterbi: cost of a change of site within a hold ...
    switch_hold: float = 12.0         # ... and of hold
    min_hold_s: float = 1.0
    free_rel: float = 1.3             # the 'free' state (his hand not on her): a phone that sees it scores it as this many hand lengths off
    free_unseen: float = 0.5          # ... a phone that does not see it
    free_prior: float = 0.12
    free_fit: float = -0.5
    face_to_face: float = -0.35       # facing below: face to face; above +0.35: she faces away (he is behind her)


# per facing class: the prior of each hold (shared by its sites)
HOLD_PRIOR = {"face": {"upper_back": 0.45, "wrap_back": 0.18, "shoulder_top": 0.10, "chest": 0.07, "arm": 0.10, "hand_hold": 0.10},
              "away": {"behind_back": 0.50, "shoulder_top": 0.12, "side": 0.12, "chest": 0.06, "arm": 0.10, "hand_hold": 0.10},
              "side": {"behind_back": 0.28, "upper_back": 0.15, "shoulder_top": 0.15, "side": 0.12, "chest": 0.08, "arm": 0.11,
                       "hand_hold": 0.11}}
HOLDS["side"] = {"facing": (-1.0, 1.0), "sites": ["side_L", "side_R", "waist_L", "waist_R"], "palm_to": "surface",
                 "desc": "his right hand on her side / waist (turns, open positions)"}
HOLD_NAMES = tuple(HOLDS)
# candidate sites of his right palm on her body
BACK_GRID = tuple(f"back_{x}_{lv}" for x in ("L2", "L1", "C", "R1", "R2") for lv in ("hi", "mid", "lo"))
CAND_SITES = BACK_GRID + ("shoulder_back_L", "shoulder_back_R", "shoulder_top_L", "shoulder_front_L", "shoulder_top_R", "chest_L", "chest",
                          "chest_R", "side_L", "side_R", "upper_arm_L", "upper_arm_R", "forearm_L", "forearm_R", "hand_L", "hand_R")
UPPER_LEFT_BACK = ("back_L2_hi", "back_L1_hi", "back_C_hi", "shoulder_back_L")


def site_hold(site: str, fclass: str) -> str:
    """the hold a palm on this site means, for how the couple stands (face / away / side)"""
    if site.startswith(("upper_arm", "forearm")):
        return "arm"
    if site.startswith("hand_"):
        return "hand_hold"
    if site.startswith(("shoulder_top", "shoulder_front")):
        return "shoulder_top"
    if site.startswith("chest"):
        return "chest"
    if site.startswith(("side_", "waist_")):
        if fclass == "face":                     # face to face, his right arm round her left flank: the hand at her side is the same hold
            return "upper_back" if site.endswith("_L") else "wrap_back"
        return "side"
    if fclass == "face":
        return "upper_back" if site in UPPER_LEFT_BACK else "wrap_back"
    if fclass == "side" and site in UPPER_LEFT_BACK:
        return "upper_back"
    return "behind_back"


def facing_class(f: np.ndarray, cfg: HoldCfg) -> np.ndarray:
    return np.where(f < cfg.face_to_face, "face", np.where(f > -cfg.face_to_face, "away", "side"))


def candidate_points(kz: dict, sites_f: dict, bodies_f: dict) -> dict:
    """{site: (point (n,3), normal (n,3))} on her posed body for the candidate sites (arm sites: the upper arm's axis middle + its radius
    toward the outside)"""
    out = {}
    J, R = kz["J"], kz["R"]
    for nm in CAND_SITES:
        if nm.startswith(("upper_arm", "forearm")):
            s = "left" if nm.endswith("_L") else "right"
            a_, b_ = (SHOULDER[s], ELBOW[s]) if nm.startswith("upper") else (ELBOW[s], WRIST[s])
            c = 0.5 * (J[:, a_] + J[:, b_])
            out_dir = c - J[:, SPINE3]
            out_dir[:, 1] = 0
            out_dir /= np.maximum(np.linalg.norm(out_dir, axis=1, keepdims=True), 1e-9)
            cap = f"upper_arm_dist_{SX[s]}" if nm.startswith("upper") else f"forearm_dist_{SX[s]}"
            r = float(bodies_f["r"][bodies_f["names"].index(cap)])
            out[nm] = (c + r * out_dir, out_dir)
        elif nm.startswith("hand_"):
            s = "left" if nm.endswith("_L") else "right"
            out[nm] = hand_site(J, s, "palm")
        else:
            out[nm] = site_world(sites_f[nm], J, R)
    return out


def ray_occluded(C: np.ndarray, X: np.ndarray, A: np.ndarray, B: np.ndarray, r: np.ndarray, margin) -> np.ndarray:
    """phone centre C (n,3), points X (n,m,3), occluder capsules A, B (n,c,3), radii (c,) -> (n,m) the segment C -> X (stopped `margin`
    (scalar or (m,)) short of X) passes through a capsule"""
    d = X - C[:, None]
    L = np.linalg.norm(d, axis=-1, keepdims=True)
    mg = np.broadcast_to(np.asarray(margin, float), X.shape[1:2])[None, :, None]
    Xs = C[:, None] + d * np.maximum(L - mg, 0) / np.maximum(L, 1e-9)
    dist, _, _ = seg_seg(C[:, None, None], Xs[:, :, None], A[:, None], B[:, None])
    return (dist < r[None, None]).any(-1)


def occluders(bodies: dict, kl: dict, kf: dict, actor: str = "lead", side: str = "right") -> tuple:
    """capsules that can hide the actor's hand from a phone: her torso, head, arms; his torso, head and his OTHER arm"""
    Af, Bf = capsule_points(bodies["follow"], kf["J"], kf["R"], kf["tips"])
    Al, Bl = capsule_points(bodies["lead"], kl["J"], kl["R"], kl["tips"])
    ox = "L" if side == "right" else "R"
    of = [i for i, p in enumerate(bodies["follow"]["parts"]) if p in ("torso", "neck_head", "upper_arm_L", "upper_arm_R", "forearm_L",
                                                                       "forearm_R", "shoulder_L", "shoulder_R")]
    ol = [i for i, p in enumerate(bodies["lead"]["parts"]) if p in ("torso", "neck_head", f"upper_arm_{ox}", f"forearm_{ox}", f"shoulder_{ox}")]
    OA = np.concatenate([Af[:, of], Al[:, ol]], 1)
    OB = np.concatenate([Bf[:, of], Bl[:, ol]], 1)
    Orad = np.concatenate([bodies["follow"]["r"][of], bodies["lead"]["r"][ol]])
    return OA, OB, Orad


def site_margins(cfg: HoldCfg, bodies: dict) -> np.ndarray:
    """per candidate site: how far short of it the phone's ray stops (a limb / hand site stands on her own limb's capsule: the ray must not
    count that limb as hiding it)"""
    out = []
    for nm in CAND_SITES:
        if nm.startswith(("upper_arm", "forearm")):
            cap = ("upper_arm_dist_" if nm.startswith("upper") else "forearm_dist_") + nm[-1]
            out.append(cfg.occl_margin_m + float(bodies["follow"]["r"][bodies["follow"]["names"].index(cap)]) + 0.015)
        elif nm.startswith("hand_"):
            out.append(cfg.occl_margin_m + 0.03)
        else:
            out.append(cfg.occl_margin_m)
    return np.asarray(out)


def hand_2d(obs, c: str, ks: np.ndarray, cfg: HoldCfg) -> tuple[np.ndarray, np.ndarray]:
    """MAMMA's view of his right hand in phone c: (sees (n,), the median of the visible hand points (n,2))"""
    z = obs.z[c]
    hid = hc.landmark_parts()["right_hand"]
    lm = z["lm"][ks, hc.PIDX["lead"]][:, hid]
    vv = (z["vis"][ks, hc.PIDX["lead"]][:, hid] > cfg.vis) & (np.abs(lm).sum(-1) > 0)
    see = z["ok"][ks] & (vv.sum(1) >= cfg.min_lm)
    med = np.stack([np.where(vv, lm[..., d], np.nan) for d in range(2)], -1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        u = np.nanmedian(med, axis=1)
    return see, u


def hold_evidence(fit: dict, obs, ks: np.ndarray, cfg: HoldCfg, bodies: dict, sites: dict, judge=None) -> dict:
    """per frame of ks and candidate site: the log-likelihood of the 6 phones' 2D (+ COCO) for his right palm on that site, the 3D terms.
    -> {"ll2d" (n,S), "llfit" (n,S), "reach_ok" (n,S), "sees" (n,C) phones that see the hand, "hidden" (n,S,C) site hidden from the phone}"""
    kl = posed(fit["lead"], ks)
    kf = posed(fit["follow"], ks)
    cands = candidate_points(kf, sites["follow"], bodies["follow"])
    S = len(CAND_SITES)
    X = np.stack([cands[nm][0] + cfg.palm_lift_m * cands[nm][1] for nm in CAND_SITES], 1)          # (n,S,3)
    n = len(ks)
    OA, OB, Orad = occluders(bodies, kl, kf)
    Orad = Orad * cfg.occl_r_scale
    C = len(obs.cams)
    ll = np.zeros((n, S))
    sees = np.zeros((n, C), bool)
    hidden = np.zeros((n, S, C), bool)
    for ci, c in enumerate(obs.cams):
        z = obs.z[c]
        ok = z["ok"][ks]
        Rc, tc = z["R"][ks], z["t"][ks]
        Cc = -np.einsum("kji,kj->ki", Rc, tc)
        occ = ray_occluded(Cc, X, OA, OB, Orad, site_margins(cfg, bodies))                       # (n,S)
        uv, dep = obs.project(c, X, ks)                                                           # (n,S,2)
        f = z["K"][ks][:, 0, 0]
        Lpx = cfg.hand_len_m * f[:, None] / np.maximum(dep, 0.1)
        see, u_obs = hand_2d(obs, c, ks, cfg)
        e = np.linalg.norm(uv - u_obs[:, None], axis=-1) / np.maximum(Lpx, 1.0)
        vis_site = ~occ & (dep > 0.1)
        l_see = np.where(vis_site, -np.log1p((e / cfg.sigma_rel) ** 2), np.log(cfg.p_hallucinate))
        l_not = np.where(vis_site, np.log(cfg.p_miss), 0.0)
        lv = np.where(see[:, None], np.nan_to_num(l_see, nan=np.log(cfg.p_hallucinate)), l_not)
        ll += np.where(ok[:, None], lv, 0.0)
        sees[:, ci] = see
        hidden[:, :, ci] = occ
        if judge is not None and cfg.w_coco > 0:
            lc = np.zeros((n, S))
            for i, k in enumerate(ks):
                if not ok[i]:
                    continue
                u, good = judge.joint(c, "lead", "right", int(k), "wrist")
                if good:
                    ee = np.linalg.norm(uv[i] - u[None], axis=-1) / np.maximum(Lpx[i], 1.0)
                    lc[i] = np.where(vis_site[i], -np.log1p((ee / (1.8 * cfg.sigma_rel)) ** 2), np.log(cfg.p_hallucinate))
            ll += cfg.w_coco * lc
    pc, _ = hand_site(kl["J"], "right", "palm")
    dfit = np.linalg.norm(X - pc[:, None], axis=-1)
    llfit = -cfg.w_fit * np.log1p((dfit / cfg.sigma_fit_m) ** 2)
    reach = np.linalg.norm(X - kl["J"][:, None, SHOULDER["right"]], axis=-1) < cfg.reach_m
    return {"ll2d": ll, "llfit": llfit, "reach_ok": reach, "sees": sees, "hidden": hidden, "dfit": dfit, "X": X}


def viterbi(score: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """score (n,S) log-likelihoods, trans (S,S) costs -> best state path (n,)"""
    n, S = score.shape
    dp = score[0].copy()
    bp = np.zeros((n, S), np.int32)
    for i in range(1, n):
        cand = dp[:, None] - trans                       # from state a (rows) to b (cols)
        bp[i] = cand.argmax(0)
        dp = cand.max(0) + score[i]
    path = np.zeros(n, np.int32)
    path[-1] = int(dp.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    return path


def _min_runs(lab: np.ndarray, n: int, keep=()) -> np.ndarray:
    """runs shorter than n frames take the label of the longer neighbouring run (labels in `keep` are never changed)"""
    lab = lab.copy()
    while True:
        runs = []
        i = 0
        while i < len(lab):
            j = i
            while j + 1 < len(lab) and lab[j + 1] == lab[i]:
                j += 1
            runs.append((i, j, lab[i]))
            i = j + 1
        short = [(ri, r_) for ri, r_ in enumerate(runs) if r_[1] - r_[0] + 1 < n and r_[2] not in keep and len(runs) > 1]
        if not short:
            return lab
        ri, (a, b, v) = min(short, key=lambda x: x[1][1] - x[1][0])
        nb = [runs[x] for x in (ri - 1, ri + 1) if 0 <= x < len(runs)]
        best = max(nb, key=lambda x: x[1] - x[0])
        lab[a:b + 1] = best[2]


def lrh_contact_frames(fit: dict, m: dict, cfg: HoldCfg) -> dict:
    """which frames the leader's right hand is on her torso / shoulders / arms (hand_contact's events, or the fitted palm close to her torso
    capsules), and which frames it holds her hand (hand_contact's hand-hand events) -> masks (T,)"""
    T = len(fit["lead"]["t"])
    torso = np.zeros(T, bool)
    hand = np.zeros(T, bool)
    arm = np.zeros(T, bool)
    for e in hc.load_events():
        if e["role"] != "lead" or e["side"] != "right":
            continue
        sl = slice(e["i0"], e["i1"] + 1)
        if e["region"] == "hand":
            hand[sl] = True
        elif e["region"] in ("forearm", "upper_arm"):
            arm[sl] = True
        elif e["region"] in ("chest", "shoulder_blade", "back", "waist", "shoulder", "neck_head", "hip"):
            torso[sl] = True
    close = m["dist_m"] < cfg.couple_max_m
    near = np.minimum.reduce([m["lead_R_vs_torso"], m["lead_R_vs_shoulder_L"], m["lead_R_vs_shoulder_R"]]) < cfg.torso_near_m
    near_arm = np.minimum(m["lead_R_vs_upper_arm_L"], m["lead_R_vs_upper_arm_R"]) < 0.03
    torso = (torso | arm | ((near | near_arm) & close)) & ~hand
    return {"torso": torso, "hand": hand}


def detect_holds(fit_name: str = "input", cfg: HoldCfg | None = None, use_coco: bool = True) -> dict:
    """the hold of the leader's right hand on every frame -> v4/contacts/holds.json (+ holds.npz: per frame hold id, site id, the evidence).
    Two levels: the HOLD (per facing class, the log-sum of its sites' scores + a 'free' state = his hand is not on her although the fit is
    close: MAMMA sees it, but away from every site) by a Viterbi with a switching cost and a minimum length (hysteresis); then within the
    hold the site per frame (a lighter Viterbi): the solver's starting target, the 2D refines it."""
    cfg = cfg or HoldCfg()
    t0 = time.time()
    fit = load_fit(fit_name)
    m = load_measure(fit_name)
    bodies = capsule_bodies(load_fit("input"))
    sites = site_bodies(load_fit("input"))
    T = len(fit["lead"]["t"])
    masks = lrh_contact_frames(fit, m, cfg)
    ks = np.flatnonzero(masks["torso"])
    obs = hc.Obs()
    judge = hc.CocoJudge() if use_coco else None
    ev = {"ll2d": [], "llfit": [], "reach_ok": [], "sees": [], "hidden": [], "dfit": []}
    for a in range(0, len(ks), 1500):
        e = hold_evidence(fit, obs, ks[a:a + 1500], cfg, bodies, sites, judge)
        for k_ in ev:
            ev[k_].append(e[k_])
        log(f"holds: evidence {min(a + 1500, len(ks))}/{len(ks)} frames ({time.time() - t0:.0f} s)")
    ev = {k_: (np.concatenate(v) if v else np.zeros((0, len(CAND_SITES)))) for k_, v in ev.items()}
    fcls = facing_class(m["facing"], cfg)
    S = len(CAND_SITES)
    hold_names = list(HOLD_NAMES)
    H = len(hold_names)
    FREE = hold_names.index("free")
    site_id = np.full(T, -1, np.int16)
    hold_id = np.full(T, FREE, np.int16)
    hold_id[masks["hand"]] = hold_names.index("hand_hold")
    margin = np.full(T, np.nan)
    hold_score = np.full((T, H), np.nan, np.float32)
    nview_sees = ev["sees"].sum(1) if len(ev["sees"]) else np.zeros(0)
    pos = np.full(T, -1)
    pos[ks] = np.arange(len(ks))
    nsite = {fc: {h: sum(site_hold(s2, fc) == h for s2 in CAND_SITES) for h in HOLDS} for fc in ("face", "away", "side")}
    for a, b in _runs(masks["torso"], 0):
        kk = np.arange(a, b + 1)
        idx = pos[kk]
        n = len(kk)
        sc = ev["ll2d"][idx] + ev["llfit"][idx] + np.where(ev["reach_ok"][idx], 0.0, -20.0)
        holds_of = np.array([[site_hold(s, fcls[k]) for s in CAND_SITES] for k in kk])
        pri = np.array([[np.log(max(HOLD_PRIOR[fcls[k]].get(h, 0.01), 1e-3) / max(1, nsite[fcls[k]][h])) for h in holds_of[i]]
                        for i, k in enumerate(kk)])
        sc_site = sc + pri
        # the free state: each phone that sees his hand scores it as if the hand stood free_rel hand lengths from the nearest site
        l_free = (nview_sees[idx] * -np.log1p((cfg.free_rel / cfg.sigma_rel) ** 2)
                  + (len(obs.cams) - nview_sees[idx]) * np.log(cfg.free_unseen) + np.log(cfg.free_prior) + cfg.free_fit)
        # hold scores: log-sum-exp over the sites of the hold (for the frame's facing class)
        hsc = np.full((n, H), -1e9)
        for hi_, h in enumerate(hold_names):
            msk = holds_of == h
            if not msk.any():
                continue
            v = np.where(msk, sc_site, -1e9)
            mx = v.max(1, keepdims=True)
            hsc[:, hi_] = np.where(msk.any(1), (mx + np.log(np.exp(v - mx).sum(1, keepdims=True)))[:, 0], -1e9)
        hsc[:, FREE] = l_free
        tr = np.full((H, H), cfg.switch_hold)
        np.fill_diagonal(tr, 0.0)
        hp = viterbi(hsc, tr)
        hp = _min_runs(hp, int(round(cfg.min_hold_s * HZ)))
        hold_id[kk] = hp
        hold_score[kk] = hsc
        srt = np.sort(hsc, 1)
        margin[kk] = hsc[np.arange(n), hp] - np.where(srt[:, -1] == hsc[np.arange(n), hp], srt[:, -2], srt[:, -1])
        # sites within each hold run
        for h in np.unique(hp):
            if h == FREE:
                continue
            for c0, c1 in _runs(hp == h, 0):
                rr = np.arange(c0, c1 + 1)
                msk = holds_of[rr] == hold_names[h]
                v = np.where(msk, sc_site[rr], -1e9)
                tr2 = np.full((S, S), cfg.switch_site)
                np.fill_diagonal(tr2, 0.0)
                site_id[kk[rr]] = viterbi(v, tr2)
    names = np.array(hold_names)[hold_id]
    segs = []
    for h in hold_names:
        for a, b in _runs(names == h, 0):
            sid = site_id[a:b + 1]
            sid = sid[sid >= 0]
            segs.append({"hold": h, "i0": a, "i1": b, "t0": round(a / HZ, 2), "t1": round(b / HZ, 2), "seconds": round((b - a + 1) / HZ, 2),
                         "site": CAND_SITES[int(np.bincount(sid).argmax())] if len(sid) else None,
                         "facing_median": round(float(np.median(m["facing"][a:b + 1])), 2),
                         "dist_m_median": round(float(np.median(m["dist_m"][a:b + 1])), 2),
                         "margin_median": round(float(np.nanmedian(margin[a:b + 1])), 2) if np.isfinite(margin[a:b + 1]).any() else None,
                         "phones_seeing_hand_median": float(np.median(nview_sees[pos[a:b + 1]])) if (pos[a:b + 1] >= 0).all() else None})
    segs.sort(key=lambda s: s["i0"])
    summ = {h: round(float((names == h).sum()) / HZ, 1) for h in hold_names}
    np.savez_compressed(OUT / "holds.npz", hold_id=hold_id, site_id=site_id, margin=margin, hold_score=hold_score, ks=ks, ll2d=ev["ll2d"],
                        llfit=ev["llfit"], sees=ev["sees"], hidden=ev["hidden"], dfit=ev["dfit"], holds=np.array(hold_names),
                        sites=np.array(CAND_SITES), torso_contact=masks["torso"], hand_hold=masks["hand"])
    rep = {"what": "the hold of the leader's right hand per frame (zouk_contacts.detect_holds): hand_contact's events + the 6 phones' 2D "
                   "(MAMMA hand points + visibility, COCO wrist) per candidate site with occlusion by the capsule bodies, the zouk prior per "
                   "facing, a 'free' state, Viterbi over holds (hysteresis) then over sites", "fit": fit_name, "config": asdict(cfg),
           "prior": HOLD_PRIOR, "seconds_per_hold": summ, "segments": segs, "runtime_s": round(time.time() - t0, 1)}
    write_json(OUT / "holds.json", rep)
    log(f"holds: {summ} ({time.time() - t0:.0f} s)")
    return rep


def load_holds() -> dict:
    with np.load(OUT / "holds.npz") as z:
        d = {k: z[k] for k in z.files}
    d["names"] = np.asarray(d["holds"])[d["hold_id"]]
    return d


# ------------------------------------------------------------------------------------------------ looking: the phones at one frame
#
# A tile per phone: the undistorted crop around the couple, MAMMA's visible right-hand points of the leader (yellow), the owner's COCO right
# wrist (cyan cross), the candidate sites (green = the phone sees it, red = hidden behind a body), the fitted palms (white = first fit,
# magenta = second fit). Used to calibrate the hold evidence and for the review sheets.

SITE_COL = {"upper_back": (60, 220, 60), "wrap_back": (40, 160, 255), "behind_back": (60, 220, 60), "shoulder_top": (255, 200, 40),
            "chest": (40, 40, 255), "side": (200, 120, 255), "arm": (255, 255, 255)}


def look_tiles(k: int, fits: list[tuple[str, dict]], obs, phones, bodies: dict, sites: dict, cams=None, tile: int = 330,
               cfg: HoldCfg | None = None, show_sites: bool = True, extent_m: float = 1.15, label: str = "") -> list[np.ndarray]:
    import cv2
    cfg = cfg or HoldCfg()
    cams = cams or list(obs.cams)
    fit0 = fits[0][1]
    ks = np.array([k])
    kl = posed(fit0["lead"], ks)
    kf = posed(fit0["follow"], ks)
    cands = candidate_points(kf, sites["follow"], bodies["follow"])
    X = np.stack([cands[nm][0] + cfg.palm_lift_m * cands[nm][1] for nm in CAND_SITES], 1)
    OA, OB, Orad = occluders(bodies, kl, kf)
    centre = 0.5 * (kl["J"][0, SPINE3] + kf["J"][0, SPINE3])
    palms = []
    for name, f in fits:
        kk = posed(f["lead"], ks)
        palms.append((name, hand_site(kk["J"], "right", "palm")[0][0], kk["J"][0, ELBOW["right"]], kk["J"][0, WRIST["right"]]))
    fcl = facing_class(couple_geometry(kl["J"], kf["J"])["facing"], cfg)[0]
    judge = hc.CocoJudge(cams) if not hasattr(look_tiles, "_judge") else look_tiles._judge
    look_tiles._judge = judge
    out = []
    for c in cams:
        z = obs.z[c]
        if not z["ok"][k]:
            out.append(np.full((tile, tile, 3), 30, np.uint8))
            continue
        img, k1, terr = phones.frame(c, k)
        K, R, t = z["K"][k], z["R"][k], z["t"][k]
        uv, d = obs.project(c, centre[None, None], ks)
        cx, cy = uv[0, 0]
        side = max(160.0, extent_m * K[0, 0] / max(d[0, 0], 0.3))
        raw = hc.crop_img(img, cx, cy, side, tile, K, k1) if img is not None else np.full((tile, tile, 3), 40, np.uint8)
        Kc = hc.crop_K(K, cx, cy, side, tile)
        s = tile / side

        def to_tile(u):
            return (int(round((u[0] - (cx - side / 2)) * s)), int(round((u[1] - (cy - side / 2)) * s)))
        Cc = -R.T @ t
        if show_sites:
            occ = ray_occluded(Cc[None], X, OA, OB, Orad * cfg.occl_r_scale, site_margins(cfg, bodies))[0]
            pu, _ = obs.project(c, X, ks)
            for i, nm in enumerate(CAND_SITES):
                h = site_hold(nm, fcl)
                col = SITE_COL.get(h, (255, 255, 255))
                p = to_tile(pu[0, i])
                cv2.circle(raw, p, 5, col, 2 if not occ[i] else 1, cv2.LINE_AA)
                if occ[i]:
                    cv2.line(raw, (p[0] - 4, p[1] - 4), (p[0] + 4, p[1] + 4), col, 1, cv2.LINE_AA)
        see, u_med = hand_2d(obs, c, ks, cfg)
        hid = hc.landmark_parts()["right_hand"]
        lm = z["lm"][k, hc.PIDX["lead"]][hid]
        vv = z["vis"][k, hc.PIDX["lead"]][hid] > cfg.vis
        for q, v in zip(lm, vv):
            if v:
                cv2.circle(raw, to_tile(q), 1, (0, 230, 255), -1)
        u, good = judge.joint(c, "lead", "right", int(k), "wrist")
        if good:
            p = to_tile(u)
            cv2.drawMarker(raw, p, (255, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 2)
        ue, goode = judge.joint(c, "lead", "right", int(k), "elbow")
        if goode:
            cv2.drawMarker(raw, to_tile(ue), (255, 255, 0), cv2.MARKER_SQUARE, 9, 1)
        for (name, pc, el, wr), col in zip(palms, [(255, 255, 255), (255, 0, 255), (0, 255, 0)]):
            pu, _ = obs.project(c, np.stack([pc, el, wr])[None], ks)
            a, b_, w_ = to_tile(pu[0, 0]), to_tile(pu[0, 1]), to_tile(pu[0, 2])
            cv2.line(raw, b_, w_, col, 1, cv2.LINE_AA)
            cv2.line(raw, w_, a, col, 1, cv2.LINE_AA)
            cv2.circle(raw, a, 4, col, -1, cv2.LINE_AA)
        Cf = -R.T @ t - kf["J"][0, SPINE3]
        Cf[1] = 0
        fv = float(Cf @ torso_frame(kf["J"])[0, :, 2] / max(np.linalg.norm(Cf), 1e-9))
        lines = [f"cam {c} t {k / HZ:.2f} {label}", f"her front->phone {fv:+.1f}  sees {'Y' if see[0] else 'n'}"]
        out.append(hc.label_tile(raw, lines, 0.4))
    return out


def look(times: list[float], fits=("input", "v3"), path: Path | None = None, cams=None, tile: int = 300, show_sites: bool = True,
         labels: dict | None = None, extent_m: float = 1.15) -> Path:
    """one row per time: the 6 phones (look_tiles) -> a jpg"""
    import cv2
    obs = hc.Obs()
    phones = hc.Phones()
    bodies = capsule_bodies(load_fit("input"))
    sites = site_bodies(load_fit("input"))
    fl = [(f, load_fit(f)) if isinstance(f, str) else f for f in fits]
    rows = []
    for t in times:
        k = int(round(t * HZ))
        tl = look_tiles(k, fl, obs, phones, bodies, sites, cams, tile, show_sites=show_sites,
                        label=(labels or {}).get(t, ""), extent_m=extent_m)
        rows.append(np.concatenate(tl, 1))
    sheet = np.concatenate(rows, 0)
    path = path or (REVIEW / f"look_{int(times[0] * 10):05d}.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


# ------------------------------------------------------------------------------------------------ the measurement report (before)

def film_time_of(t_dance: float) -> float | None:
    """dance (master) seconds -> film seconds of the current direction (v3/direction/direction.json, 1.0x segments), None outside them"""
    if not hasattr(film_time_of, "_segs"):
        p = WORK / "v3" / "direction" / "direction.json"
        segs = []
        if p.exists():
            for s in json.loads(p.read_text(encoding="utf-8"))["segments"]:
                if abs(float(s.get("speed", 1.0)) - 1.0) < 1e-6 and s.get("dance") and s.get("film") and not str(s["id"]).endswith("_r"):
                    segs.append((float(s["dance"][0]), float(s["dance"][1]), float(s["film"][0])))
        film_time_of._segs = segs
    for a, b, f0 in film_time_of._segs:
        if a <= t_dance <= b:
            return round(f0 + (t_dance - a), 2)
    return None


WORST_GROUPS = (("her torso (hand inside her body)", ["partner_torso", "partner_shoulder_L", "partner_shoulder_R"], 6),
                ("his own right forearm (wrist folded past its limits)", ["self_forearm", "wrist"], 7),
                ("her arms / hands", ["partner_forearm_L", "partner_forearm_R", "partner_upper_arm_L", "partner_upper_arm_R", "partner_hand_L",
                                      "partner_hand_R", "partner_neck_head"], 7))


def worst_frames_mixed(m: dict, key: str = "lead_R", min_gap_s: float = 1.5) -> list:
    """the worst frames per kind of collision (WORST_GROUPS: her torso, his own forearm / the wrist, her arms and hands), distinct incidents"""
    out, taken = [], []
    for title, parts, n in WORST_GROUPS:
        cand = worst_frames(m, key, 200, parts, min_gap_s)
        got = 0
        for w in cand:
            if got >= n:
                break
            if any(abs(w["frame"] - j) < min_gap_s * HZ for j in taken):
                continue
            w["group"] = title
            out.append(w)
            taken.append(w["frame"])
            got += 1
    return out


def measure_report(fits=("input", "v3"), n_worst: int = 20) -> dict:
    """the before tables: per fit and hand, the collisions per partner part / own part, the wrist limits; for the leader's right hand the
    same per hold (holds.npz) and the worst frames; the hand_contact events those frames fall in -> v4/contacts/report_measure.json"""
    bodies = capsule_bodies(load_fit("input"))
    H = load_holds() if (OUT / "holds.npz").exists() else None
    events = hc.load_events()
    out = {"what": "capsule collisions (zouk_contacts.measure_fit) + wrist limits, per fit; clearance < -0.5 cm = penetration; depth in cm",
           "wrist_limits_deg": WRIST_LIMITS, "fits": {}}
    for name in fits:
        m = load_measure(name)
        fr = {}
        for key in HAND_KEYS:
            r = "lead" if key.startswith("lead") else "follow"
            caps = {"self": bodies[r], "other": bodies[hc.OTHER[r]]}
            fr[key] = {"collisions": hand_collision_table(m, key, caps=caps), "wrist": wrist_table(m, key)}
        if H is not None:
            per = {}
            for h in HOLD_NAMES:
                msk = H["names"] == h
                if msk.sum() < 15:
                    continue
                t = hand_collision_table(m, "lead_R", msk)
                per[h] = {"seconds": round(float(msk.sum()) / HZ, 1),
                          "collisions": {k: v for k, v in t.items() if v["frames_gt_0.5cm"]},
                          "wrist_any_beyond_limit_s": wrist_table(m, "lead_R", msk)["any"]["seconds_gt_0"]}
            fr["lead_R_by_hold"] = per
        wf = worst_frames_mixed(m, "lead_R")
        for w in wf:
            k = w["frame"]
            w["film_s"] = film_time_of(k / HZ)
            if H is not None:
                w["hold"] = str(H["names"][k])
                w["site"] = str(H["sites"][H["site_id"][k]]) if H["site_id"][k] >= 0 else None
            w["events"] = [e["id"] for e in events if e["i0"] - 6 <= k <= e["i1"] + 6 and e["role"] == "lead" and e["side"] == "right"]
            w["wrist_deg"] = {nm: round(float(m[f"lead_R_wrist_{nm}"][k]), 1) for nm in ("bend", "flexion", "deviation", "twist")}
            w["facing"] = round(float(m["facing"][k]), 2)
        fr["lead_R_worst"] = wf
        out["fits"][name] = fr
    write_json(OUT / "report_measure.json", out)
    return out


def worst_sheet(fit_names=("input", "v3"), n: int = 20, path_stem: str = "worst", per_sheet: int = 5, cams_per_row: int = 3,
                frames: list | None = None, title: str = "") -> list:
    """phone crops (look_tiles: 3 phones that see the hand area) + both fits at the worst leader-right-hand frames"""
    import cv2
    rep = json.loads((OUT / "report_measure.json").read_text(encoding="utf-8"))
    wf = frames or rep["fits"][fit_names[-1]]["lead_R_worst"][:n]
    obs = hc.Obs()
    phones = hc.Phones()
    bodies = capsule_bodies(load_fit("input"))
    sites = site_bodies(load_fit("input"))
    fl = [(f, load_fit(f)) for f in fit_names]
    paths = []
    for si in range(0, len(wf), per_sheet):
        rows = []
        for w in wf[si:si + per_sheet]:
            k = int(w["frame"])
            # the phones: those whose view of the palm area is widest, preferring phones that see the hand
            pc = hand_site(posed(fl[-1][1]["lead"], np.array([k]))["J"], "right", "palm")[0]
            cams = hc.pick_cams(obs, None, None, k, pc[0], n=cams_per_row, exclude_bad=False) or list(obs.cams)[:cams_per_row]
            lab = f"{w.get('what', '')} {w.get('depth_cm', '')}cm {w.get('hold', '')}"
            tl = look_tiles(k, fl, obs, phones, bodies, sites, cams, 330, show_sites=False, label="", extent_m=0.85)
            head = np.full((330, 250, 3), 245, np.uint8)
            lines = [f"t {k / HZ:.2f} s (film {w.get('film_s')})", f"frame {k}", f"worst: {w.get('what')}", f"depth {w.get('depth_cm')} cm",
                     f"hold: {w.get('hold')}", f"site: {w.get('site')}", f"events: {w.get('events')}", f"wrist {w.get('wrist_deg')}",
                     f"facing {w.get('facing')}"]
            for extra in w.get("_lines", []):
                lines.append(extra)
            for kk, v in (w.get("all_cm") or {}).items():
                lines.append(f"  {kk}: {v}")
            y = 18
            for ln in lines[:17]:
                cv2.putText(head, str(ln)[:44], (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
                y += 18
            rows.append(np.concatenate([head] + tl, 1))
        Wd = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 4), (0, Wd - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        top = np.full((30, Wd, 3), 255, np.uint8)
        cv2.putText(top, title or f"leader's right hand, worst collision frames: phone crop | white = {fit_names[0]}, magenta = {fit_names[-1]} "
                    "(elbow - wrist - palm); yellow = MAMMA's visible hand points, cyan x = COCO wrist", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        p = REVIEW / f"{path_stem}_{si // per_sheet + 1}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.concatenate([top] + rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 86])
        paths.append(p)
        log(f"sheet -> {p}")
    return paths


# ------------------------------------------------------------------------------------------------ the resolution
#
# Per window (a run of frames where the leader's right hand is in a hold and off its site, collides, or bends the wrist past its limits) the
# leader's right arm (shoulder, elbow, wrist, fingers: hand_contact.Actor, deltas on a smooth temporal basis, blended in / out by raised-cosine
# ramps that are a pure blend of the core's correction) is optimised from the INSTALLED motion (v3/smplx_contact):
#   stage 1 (holds with a site): IK - the palm to the hold's per-frame site on her body (smoothed in her frame) facing her surface, the elbow
#            to the multi-view COCO elbow; no contact / penetration term (the arm may have to pass round her);
#   stage 2: the palm into the hold's REGION (soft, region_slack) and onto her surface (palm patch, palm facing it, the finger template of the
#            hold), no penetration of the hand / forearm / upper arm into her mesh (closest-point planes, exact winding-number sign in the
#            push-out), the capsule collisions (his right hand vs her forearms, upper arms, hands, head, shoulder girdles; his right forearm /
#            upper arm vs her arms; his fingers / distal palm vs his OWN right forearm), the wrist inside its limits (whole bend, flexion /
#            extension, radial / ulnar), MAMMA's hand points only in the phones that see the hand where the hold's site is visible
#            (visibility-aware 2D), the COCO elbow, hand_contact's terms on the added motion (free up to 0.3 m/s), the speed cap, smoothness,
#            minimal rotation change;
#   guards against the INSTALLED motion and the INPUT fit: no new capsule collision of the hand (any pair, +0.3 cm), no new exact penetration,
#   the wrist inside its limits, the speed audit (added jump 0.55 m/s, whip = input p99 + 0.25 m/s), the visibility-aware MAMMA hand points
#   (+3 px) and the independent COCO elbow (+4 px) in the phones that see them; a window that fails keeps the installed motion.

@dataclass
class ZoukCfg:
    base: str = "v3"                     # the motion the windows start from (the installed v3 hand-contact motion; "v4" for a second pass)
    plan: str = "plan.json"              # the windows (plan_windows' output in v4/contacts/)
    problems: tuple = ()                 # plan only these kinds of problem (empty: all; ("wrist",): the folded wrists)
    wrist_only: bool = False             # the wrist pass: no hold target / surface / site terms, the palm kept where it is (keep_slack_m), the
    w_keep: float = 3.0e3                #   fingers as they are; only the wrist limits, the collisions, the 2D, the elbow, speed and smoothness
    keep_slack_m: float = 0.025
    pad_s: float = 0.3                   # a window = a run of problem frames + this on each side (its core)
    merge_s: float = 0.6                 # problem runs closer than this are one window
    max_core_s: float = 12.0             # longer runs are solved as consecutive chunks (each from the previous result)
    min_core_s: float = 0.25
    ramp_min: int = 12
    v_ramp_mps: float = 0.35
    ramp_max: int = 45
    # problems that open a window (installed motion, capsules)
    pen_cm: float = 0.5                  # hand vs her limbs / hands / head / girdle, hand vs own forearm
    torso_pen_cm: float = 1.0            # hand / arm vs her torso slabs
    wrist_excess_deg: float = 3.0
    off_site_m: float = 0.07             # in a hold with a site: the palm further than this from the hold's region (and not on it)
    # terms
    w_site_ik: float = 1.0e3             # stage 1: per m^2, the palm to the smoothed site track (an attractor: the 2D and the elbow lead)
    w_region: float = 3.0e3              # stage 2: per m^2 beyond region_slack_m (soft-min distance to the hold's sites)
    region_slack_m: float = 0.06
    region_tau_m: float = 0.02
    w_cap: float = 2.0e4                 # per m^2 of capsule overlap beyond cap_margin_m (mean over frames of the summed squared overlaps)
    cap_margin_m: float = 0.003
    arm_arm_allow_m: float = 0.025       # his forearm / upper arm may sink this far into her arms' capsules (her forearm rests on his upper arm in
                                         #   the closed hold; the capsule radii are ~1 cm off)
    w_self: float = 2.0e4
    w_twist: float = 0.0                 # 2026-10-22 review: per rad^2 of wrist twist beyond max(twist_soft_deg, |start| + 5) (candy wrappers)
    twist_soft_deg: float = 85.0
    w_elbow_lim: float = 0.0             # per rad^2 of elbow hyperextension beyond min(-5, start) and off-hinge beyond max(35, |start| + 5)
    gate_guards: bool = False            # the post-hoc gate's tests as window guards (twist, elbow, elbow / fingertip whips, wrist flick, COCO wrist)
    w_bend: float = 300.0                # per rad^2 beyond bend_soft_deg (the whole wrist bend) ...
    bend_soft_deg: float = 78.0
    w_comp: float = 150.0                # ... and per rad^2 of flexion / extension / radial / ulnar beyond their limits - 3 deg
    vis_min_lm: int = 6
    w_2d: float = 3.0                    # MAMMA's hand points in the usable phones (both stages), robust with sigma_2d_px
    sigma_2d_px: float = 14.0
    w_elbow: float = 8.0                 # COCO elbow (triangulated, bias-corrected, smoothed), robust: the elbow is seen, the hand is not
    w_elbow2d: float = 3.0               # ... and per phone: every confident COCO elbow detection (single views too), robust sigma_elbow2d_px
    sigma_elbow2d_px: float = 15.0
    site_seen_scale: float = 0.15        # the site attractor where a usable phone sees the hand (the 2D leads there)
    w_fingers_up: float = 40.0           # back holds: fingers pointing up her back more than 30 deg above her horizontal cost this (zouk: the
                                         #   leader's hand on her back lies across it, the fingers toward her spine / round her far side)
    w_speed: float = 2000.0              # the added motion beyond 0.3 m/s (hand_contact's term, 600 there: the moves here are larger)
    agree_px: float = 25.0               # a usable phone agrees with the hand when its median hand-point error is below this
    iters1: int = 120
    iters2: int = 110
    iters_polish: int = 12
    iters_push: int = 40                 # the push-out with the exact (winding-number) inside test
    exact_check: bool = True             # the exact before / after penetration measure (winding numbers; slow on a CPU)
    # guards
    guard_cap_slack_cm: float = 0.3      # capsule collisions: p95 over the core may not grow by more, frames > 2 cm by more than 3 + 2 %
    guard_agree_drop: float = 0.30       # the phones that agree with the hand (agree_px), summed over the core, may not drop by more ...
    guard_ma2d_slack_px: float = 6.0     # ... and the median hand-point error over the usable phones may not grow by more
    guard_coco_slack_px: float = 4.0
    guard_pen_slack_cm: float = 0.2      # exact penetration p90 (0.5 where the hand was not on her before: it now lies on her surface)


TORSO_HOLD_SITES = {
    "upper_back": UPPER_LEFT_BACK + ("side_L", "back_L2_mid", "back_L1_mid", "shoulder_top_L"),
    "wrap_back": tuple(s for s in BACK_GRID if s not in UPPER_LEFT_BACK) + ("side_R", "shoulder_back_R"),
    "behind_back": BACK_GRID + ("shoulder_back_L", "shoulder_back_R"),
    "shoulder_top": ("shoulder_top_L", "shoulder_front_L", "shoulder_back_L", "shoulder_top_R"),
    "chest": ("chest_L", "chest", "chest_R"),
    "side": ("side_L", "side_R"),
    "arm": ("upper_arm_L", "upper_arm_R", "forearm_L", "forearm_R"),
}
TEMPLATE_OF_ZHOLD = {"upper_back": "open", "wrap_back": "open", "behind_back": "open", "side": "open", "chest": "open",
                     "shoulder_top": "relaxed", "arm": "wrap"}


def problem_frames(m: dict, H: dict, cfg: ZoukCfg, cands_dist: np.ndarray | None = None) -> dict:
    """per frame: why the leader's right hand needs the resolution in the installed motion (capsule collisions, wrist limits, off its hold's
    sites) -> {"any": (T,), reasons...}"""
    key = "lead_R"
    T = len(m["facing"])
    r = {}
    lim = [f"{key}_vs_{p}" for p in ("hand_L", "hand_R", "forearm_L", "forearm_R", "upper_arm_L", "upper_arm_R", "shoulder_L", "shoulder_R",
                                     "neck_head")]
    hh = H["names"] == "hand_hold"
    limb = np.min(np.stack([np.where(hh & ("hand_" in nm), np.inf, m[nm]) for nm in lim], 1), 1)
    r["limb_collision"] = limb < -cfg.pen_cm / 100
    r["hand_hold_collision"] = hh & (np.minimum(m[f"{key}_vs_hand_L"], m[f"{key}_vs_hand_R"]) < -0.015)
    r["self_forearm"] = m[f"{key}_self_forearm"] < -cfg.pen_cm / 100
    r["torso"] = (m[f"{key}_vs_torso"] < -cfg.torso_pen_cm / 100) | (m[f"{key}_arm_vs_torso"] < -cfg.torso_pen_cm / 100)
    ex = limit_excess({nm: m[f"{key}_wrist_{nm}"] for nm in ("flexion", "deviation", "twist", "bend")})
    r["wrist"] = np.max(np.stack([ex[k] for k in ("flexion", "extension", "radial", "ulnar", "bend")], 1), 1) > cfg.wrist_excess_deg
    r["off_site"] = np.zeros(T, bool) if cands_dist is None else (np.isfinite(cands_dist) & (cands_dist > cfg.off_site_m))
    r["any"] = np.any(np.stack(list(r.values()), 1), 1)
    return r


def region_distance(fit: dict, H: dict, ks: np.ndarray, bodies: dict, sites: dict) -> np.ndarray:
    """per frame of ks: the distance of the fit's right palm to the nearest site of its hold's region (inf where the hold has no region)"""
    kl = posed(fit["lead"], ks)
    kf = posed(fit["follow"], ks)
    cands = candidate_points(kf, sites["follow"], bodies["follow"])
    pc, _ = hand_site(kl["J"], "right", "palm")
    out = np.full(len(ks), np.inf)
    names = H["names"][ks]
    for h, sl in TORSO_HOLD_SITES.items():
        msk = names == h
        if not msk.any():
            continue
        P = np.stack([cands[s][0][msk] for s in sl if s in cands], 1)
        out[msk] = np.linalg.norm(P - pc[msk, None], axis=-1).min(1)
    return out


def plan_windows(cfg: ZoukCfg | None = None) -> dict:
    """the windows the resolution refines (installed motion, holds.npz) -> v4/contacts/plan.json"""
    cfg = cfg or ZoukCfg()
    fit = load_fit(cfg.base)
    m = load_measure(cfg.base)
    H = load_holds()
    bodies = capsule_bodies(load_fit("input"))
    sites = site_bodies(load_fit("input"))
    T = len(m["facing"])
    rd = region_distance(fit, H, np.arange(T), bodies, sites)
    P = problem_frames(m, H, cfg, rd)
    if cfg.problems:
        P["any"] = np.any(np.stack([P[k] for k in cfg.problems], 1), 1)
    runs0 = _runs(P["any"], int(round(cfg.merge_s * HZ)))
    has_region = np.isin(H["names"], list(TORSO_HOLD_SITES))
    runs = []
    for a, b in runs0:
        cut = [a] + [k for k in range(a + 1, b + 1) if has_region[k] != has_region[k - 1]] + [b + 1]
        for x0, x1 in zip(cut[:-1], cut[1:]):
            if P["any"][x0:x1].any():
                q = np.flatnonzero(P["any"][x0:x1])
                runs.append((x0 + int(q[0]), x0 + int(q[-1])))
    pad = int(round(cfg.pad_s * HZ))
    wins = []
    for a, b in runs:
        a, b = max(0, a - pad), min(T - 1, b + pad)
        if (b - a + 1) / HZ < cfg.min_core_s:
            continue
        # chunks of at most max_core_s (consecutive, overlapping by ramp_min frames: each solved from the previous chunk's result)
        n = int(np.ceil((b - a + 1) / (cfg.max_core_s * HZ)))
        edges = np.linspace(a, b + 1, n + 1).round().astype(int)
        for ci in range(n):
            c0, c1 = int(edges[ci]), int(edges[ci + 1]) - 1
            if ci > 0:
                c0 = max(a, c0 - cfg.ramp_min)
            names = H["names"][c0:c1 + 1]
            why = {k: int(v[c0:c1 + 1].sum()) for k, v in P.items() if k != "any" and v[c0:c1 + 1].any()}
            # the blend ramps: long enough for the move the hand makes at the run's ends (its distance to the hold's region there)
            ends = [x for x, at_end in ((c0, ci == 0), (c1, ci == n - 1)) if at_end]
            move = max([min(0.3, float(rd[x])) if np.isfinite(rd[x]) else 0.0 for x in ends] + [0.0])
            ramp = hc.ramp_for(move, cfg.v_ramp_mps, cfg.ramp_min, cfg.ramp_max)
            wins.append({"core": [c0, c1], "t": [round(c0 / HZ, 2), round(c1 / HZ, 2)], "chunk": [ci, n], "run": [a, b], "ramp": int(ramp),
                         "holds": {h: int((names == h).sum()) for h in np.unique(names)}, "why_frames": why})
    # windows of one run are solved in order in one process; runs are spread over the parts
    rep = {"what": "windows of the leader's right hand to refine (zouk_contacts.plan_windows) in the motion " + cfg.base,
           "config": asdict(cfg), "frames_with_problems": {k: int(v.sum()) for k, v in P.items()},
           "seconds_with_problems": round(float(P["any"].sum()) / HZ, 1), "windows": wins,
           "window_seconds": round(sum(w["core"][1] - w["core"][0] + 1 for w in wins) / HZ, 1)}
    write_json(OUT / cfg.plan, rep)
    np.savez_compressed(OUT / f"problems_{cfg.base}.npz", **{k: v for k, v in P.items()}, region_dist=rd)
    log(f"plan: {len(wins)} windows, {rep['window_seconds']} s; problems {rep['frames_with_problems']}")
    return rep


def _sub_caps(caps: dict, idx) -> dict:
    idx = list(idx)
    return {"names": [caps["names"][i] for i in idx], "parts": [caps["parts"][i] for i in idx], "A": [caps["A"][i] for i in idx],
            "B": [caps["B"][i] for i in idx], "r": caps["r"][idx]}


def _torch_clearance(torch, A1, B1, r1, A2, B2, r2):
    """(W,n1,3) x (W,n2,3) capsules -> (W,n1,n2) clearance"""
    d, _, _ = seg_seg(A1[:, :, None], B1[:, :, None], A2[:, None], B2[:, None], torch)
    return d - r1[None, :, None] - r2[None, None, :]


def wrist_terms_t(torch, Rw, side: str, with_twist: bool = False):
    """torch: wrist local rotation (W,3,3) -> bend, flexion, deviation (rad) [, twist (rad): wrist_angles' twist, R = T(a, twist) F with F
    the minimal rotation taking R^T a onto a; F^T v = v c + u x v + u (u.v) / (1 + c), u = a x ah, c = a.ah (defined below a 180-deg bend)]"""
    a, n, t = (torch.tensor(v, dtype=Rw.dtype, device=Rw.device) for v in _rest_axes(side))
    ah = torch.einsum("wji,j->wi", Rw, a)
    ca = (ah * a).sum(-1)
    bend = torch.acos(ca.clamp(-1 + 1e-6, 1 - 1e-6))
    an, at = (ah * n).sum(-1), (ah * t).sum(-1)
    kk = bend / torch.sqrt(an ** 2 + at ** 2 + 1e-12)                # the swing vector (wrist_angles)
    if not with_twist:
        return bend, -kk * an, -kk * at
    u = torch.cross(a.expand_as(ah), ah, dim=-1)
    nn = n.expand_as(ah)
    v = nn * ca[:, None] + torch.cross(u, nn, dim=-1) + u * ((u * nn).sum(-1) / (1 + ca).clamp_min(1e-4))[:, None]
    Tn = torch.einsum("wij,wj->wi", Rw, v)
    axn = torch.cross(a, n, dim=-1)
    tw = torch.atan2((Tn * axn).sum(-1), (Tn * nn).sum(-1))
    return bend, -kk * an, -kk * at, tw


def elbow_terms_t(torch, Re, side: str):
    """torch: elbow local rotation (W,3,3) -> flexion, off-hinge (rad), as elbow_angles"""
    a, _, _ = (torch.tensor(v, dtype=Re.dtype, device=Re.device) for v in _rest_axes(side))
    fwd = torch.tensor([0.0, 0.0, 1.0], dtype=Re.dtype, device=Re.device)
    h = torch.einsum("wij,j->wi", Re, a)
    flex = torch.atan2((h * fwd).sum(-1), (h * a).sum(-1))
    up = torch.cross(a, fwd, dim=-1)
    off = torch.asin((h * up).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
    return flex, off


class VisTwoD(hc.TwoD):
    """hand_contact.TwoD restricted per frame and phone to the phones that SEE his hand (>= vis_min_lm visible MAMMA hand points) and from
    which the hand's place in the hold is not hidden (usable (W, n_cams) bool, the obs camera order)"""

    def __init__(self, torch, obs, actor, device, cams, usable: np.ndarray):
        super().__init__(torch, obs, actor, device, cams)
        ci = {c: i for i, c in enumerate(obs.cams)}
        for it in self.items:
            u = torch.as_tensor(usable[:, ci[it["cam"]]], device=device)
            it["vis"] = it["vis"] & u[:, None]
        self.items = [it for it in self.items if bool(it["vis"].any())]


class CocoElbow2D:
    """the owner's COCO elbow detections of the leader's right arm in every phone (conf >= 0.5, undistorted) over the window: a robust 2D
    term on the bias-corrected SMPL-X elbow (single-view detections constrain the elbow where no phone pair triangulates it)"""

    def __init__(self, torch, obs, judge, ks: np.ndarray, bias: dict, device):
        self.torch = torch
        C = len(obs.cams)
        W = len(ks)
        U = np.zeros((W, C, 2), np.float32)
        M = np.zeros((W, C), np.float32)
        for ci, c in enumerate(obs.cams):
            z = obs.z[c]
            for i, k in enumerate(ks):
                if not z["ok"][k]:
                    continue
                u, ok = judge.joint(c, "lead", "right", int(k), "elbow")
                if ok:
                    U[i, ci], M[i, ci] = u, 1.0
        f32 = dict(dtype=torch.float32, device=device)
        self.U, self.M = torch.tensor(U, **f32), torch.tensor(M, **f32)
        self.K = torch.tensor(np.stack([obs.z[c]["K"][ks] for c in obs.cams], 1), **f32)       # (W,C,3,3)
        self.R = torch.tensor(np.stack([obs.z[c]["R"][ks] for c in obs.cams], 1), **f32)
        self.t = torch.tensor(np.stack([obs.z[c]["t"][ks] for c in obs.cams], 1), **f32)
        self.b = torch.tensor(np.asarray(bias.get("lead_right_elbow", [0, 0, 0]), float), **f32)
        self.n = float(M.sum())

    def point(self, J):
        torch = self.torch
        ax = J[:, WRIST["right"]] - J[:, ELBOW["right"]]
        ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nrm = hc._palm_normal_t(torch, J, "right")
        nrm = nrm - (nrm * ax).sum(-1, keepdim=True) * ax
        nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cr = torch.cross(ax, nrm, dim=-1)
        return J[:, ELBOW["right"]] + ax * self.b[0] + nrm * self.b[1] + cr * self.b[2]

    def residual(self, J):
        torch = self.torch
        X = self.point(J)                                                   # (W,3)
        Xc = torch.einsum("wcij,wj->wci", self.R, X) + self.t
        uv = torch.einsum("wcij,wcj->wci", self.K, Xc)
        uv = uv[..., :2] / uv[..., 2:].clamp_min(1e-6)
        ok = self.M * (Xc[..., 2] > 0.1).float()
        return (uv - self.U).norm(dim=-1), ok

    def loss(self, J, sigma, w):
        e, ok = self.residual(J)
        r = sigma ** 2 * self.torch.log1p(e ** 2 / sigma ** 2) / sigma ** 2
        n = ok.sum(1).clamp_min(1)
        return (((r * ok).sum(1) / n) * w).sum()


def window_inputs(fit_base: dict, H: dict, ks: np.ndarray, bodies: dict, sites: dict, obs, cfg: ZoukCfg, hcfg: HoldCfg) -> dict:
    """per window frame: the hold, the site track (her frame, smoothed), the region sites, the phones usable for the 2D, her capsules"""
    from scipy.ndimage import gaussian_filter1d
    kl = posed(fit_base["lead"], ks)
    kf = posed(fit_base["follow"], ks)
    cands = candidate_points(kf, sites["follow"], bodies["follow"])
    names = H["names"][ks]
    sid = H["site_id"][ks]
    W = len(ks)
    site_pt = np.full((W, 3), np.nan)
    site_n = np.full((W, 3), np.nan)
    for i in range(W):
        if sid[i] >= 0 and names[i] in TORSO_HOLD_SITES:
            nm = CAND_SITES[sid[i]]
            site_pt[i], site_n[i] = cands[nm][0][i], cands[nm][1][i]
    has_site = np.isfinite(site_pt).all(1)
    # smooth the site track in her spine-3 frame (a change of site is not a jump of the target)
    if has_site.sum() >= 3:
        Fr = torso_frame(kf["J"])
        loc = np.einsum("kji,kj->ki", Fr, site_pt - kf["J"][:, SPINE3])
        nl = np.einsum("kji,kj->ki", Fr, site_n)
        idx = np.flatnonzero(has_site)
        for arr in (loc, nl):
            for d in range(3):
                arr[:, d] = np.interp(np.arange(W), idx, arr[idx, d])
        loc = gaussian_filter1d(loc, 8.0, axis=0, mode="nearest")
        nl = gaussian_filter1d(nl, 8.0, axis=0, mode="nearest")
        site_pt = np.einsum("kij,kj->ki", Fr, loc) + kf["J"][:, SPINE3]
        site_n = np.einsum("kij,kj->ki", Fr, nl)
        site_n /= np.maximum(np.linalg.norm(site_n, axis=1, keepdims=True), 1e-9)
    # the region: per frame the sites of its hold (padded to the longest list with the first one)
    L = max(len(v) for v in TORSO_HOLD_SITES.values())
    reg = np.zeros((W, L, 3))
    reg_ok = np.zeros((W, L), bool)
    for i in range(W):
        sl = TORSO_HOLD_SITES.get(names[i])
        if not sl:
            continue
        for j, s in enumerate(sl):
            if s in cands:
                reg[i, j] = cands[s][0][i] + hcfg.palm_lift_m * cands[s][1][i]
                reg_ok[i, j] = True
    # the phones usable for the 2D: MAMMA sees the hand, and (in a hold with a site) the site is not hidden from that phone
    C = len(obs.cams)
    usable = np.zeros((W, C), bool)
    OA, OB, Orad = occluders(bodies, kl, kf)
    for ci, c in enumerate(obs.cams):
        see, _ = hand_2d(obs, c, ks, hcfg)
        z = obs.z[c]
        Cc = -np.einsum("kji,kj->ki", z["R"][ks], z["t"][ks])
        X = np.where(has_site[:, None], site_pt, hand_site(kl["J"], "right", "palm")[0])[:, None]
        occ = ray_occluded(Cc, X, OA, OB, Orad * hcfg.occl_r_scale, hcfg.occl_margin_m)[:, 0]
        usable[:, ci] = see & (~occ | ~has_site)                 # no site (free / hand hold): every phone that sees the hand
    Af, Bf = capsule_points(bodies["follow"], kf["J"], kf["R"], kf["tips"])
    return {"names": names, "site_pt": site_pt, "site_n": site_n, "has_site": has_site, "region": reg, "region_ok": reg_ok, "usable": usable,
            "her_A": Af, "her_B": Bf, "kf": kf, "kl": kl}


def _window_gate(fit_in: dict, ks: np.ndarray, M0: dict, M1: dict, core_np: np.ndarray, ref: dict, cfg) -> tuple[list, int]:
    """the post-hoc gate's tests on one window (start M0 -> result M1, the input at ks): NEW twist / elbow hyperextension / off-hinge, added
    elbow / fingertip speed, a faster wrist rotation -> (guard messages, the ramp the elbow / fingertip moves need). Speed messages start with
    'whip' so run_refine retries with a longer ramp."""
    g = GATE
    why = []
    P_in = hc.full_pose(fit_in["lead"])[ks]
    P0, P1 = M0["pose"][:, :55], M1["pose"][:, :55]
    W0, W1, Wi = wrist_angles(P0, "right"), wrist_angles(P1, "right"), wrist_angles(P_in, "right")
    E0, E1, Ei = elbow_angles(P0, "right"), elbow_angles(P1, "right"), elbow_angles(P_in, "right")
    tw = (np.abs(W1["twist"]) > g["twist_deg"]) & (W1["bend"] < g["twist_bend_max_deg"]) & \
         (np.abs(W1["twist"]) > np.maximum(np.abs(W0["twist"]), np.abs(Wi["twist"])) + g["twist_new_deg"])
    if tw.any():
        why.append(f"wrist twist {np.abs(W1['twist'][tw]).max():.0f} deg on {int(tw.sum())} frames (new)")
    hy = (E1["flexion"] < g["hyper_deg"]) & (E0["flexion"] >= g["hyper_deg"])
    if hy.any():
        why.append(f"elbow hyperextension {E1['flexion'][hy].min():.0f} deg on {int(hy.sum())} frames (new)")
    oh = (np.abs(E1["off_hinge"]) > g["off_hinge_deg"]) & \
         (np.abs(E1["off_hinge"]) > np.maximum(np.abs(E0["off_hinge"]), np.abs(Ei["off_hinge"])) + g["off_hinge_new_deg"])
    if oh.any():
        why.append(f"elbow off its hinge {np.abs(E1['off_hinge'][oh]).max():.0f} deg on {int(oh.sum())} frames (new)")
    o_in = posed(fit_in["lead"], ks, P_in)
    pts = {"elbow": (o_in["J"][:, ELBOW["right"]], M0["J"][:, ELBOW["right"]], M1["J"][:, ELBOW["right"]])}
    for i, fi in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        pts[f"tip_{fi}"] = (o_in["tips"][:, 5 + i], M0["tips"][:, i], M1["tips"][:, i])
    ramp_need = 0
    ce = np.flatnonzero(core_np)
    J_all = np.asarray(fit_in["lead"]["joints"], float)
    p99_elbow = float(np.percentile(hc.speeds(J_all[:, ELBOW["right"]]), 99))
    for k, (xi, x0, x1) in pts.items():
        p99 = p99_elbow if k == "elbow" else max(float(ref.get("palm", 2.2)), 1.5)
        a0, a1 = hc.speeds(x0 - xi), hc.speeds(x1 - xi)
        v0, v1, vi = hc.speeds(x0), hc.speeds(x1), hc.speeds(xi)
        bad = ((a1 > g["add_mps"]) & (a1 > a0 + g["add_new_mps"])) | (v1 > np.maximum.reduce([np.full_like(v1, p99), vi, v0]) + g["fast_margin_mps"])
        if bad.any():
            j = int(np.flatnonzero(bad)[np.argmax(a1[bad])])
            why.append(f"whip {k} at {ks[j] / HZ:.2f} s added {a1[j]:.2f} m/s (speed {v1[j]:.2f})")
        if len(ce):
            ends = np.concatenate([ce[:3], ce[-3:]])
            ramp_need = max(ramp_need, hc.ramp_for(float(np.linalg.norm(x1[ends] - x0[ends], axis=1).max()), cfg.v_ramp_mps, cfg.ramp_min, cfg.ramp_max))
    r0, r1, ri = _local_rot_speed(P0, WRIST["right"]), _local_rot_speed(P1, WRIST["right"]), _local_rot_speed(P_in, WRIST["right"])
    fl = r1 > np.maximum(r0, ri) + g["wrist_rot_new_deg_frame"]
    if fl.any():
        j = int(np.flatnonzero(fl)[np.argmax(r1[fl])])
        why.append(f"whip wrist rotation at {ks[j] / HZ:.2f} s: {r1[j]:.0f} deg/frame (start {r0[j]:.0f}, input {ri[j]:.0f})")
    return why, ramp_need


def refine_window(fit_in: dict, poses: dict, win: dict, H: dict, obs, judge, bias: dict, surf, bodies: dict, sites: dict, cfg: ZoukCfg,
                  hcfg: HoldCfg, device: str = "cuda", speed_ref: dict | None = None) -> dict:
    """one window of the leader's right hand (see the section's comment). poses = {role: (T,55,3)} current poses (the installed motion with
    the windows adopted so far); fit_in = the INPUT fit (the speed guards compare with it)."""
    import torch
    from .hand_kin import WRIST as HW, ELBOW as HE
    T = len(fit_in["lead"]["t"])
    c0, c1 = win["core"]
    ramp = int(win.get("ramp", cfg.ramp_min))
    a0, a1 = max(0, c0 - ramp), min(T - 1, c1 + ramp)
    ks = np.arange(a0, a1 + 1)
    full = hc.ramp_weights(c1 - c0 + 1, ramp)
    w_np = full[(a0 - (c0 - ramp)):(a0 - (c0 - ramp)) + len(ks)]
    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    wsum = w.sum()
    f32 = dict(dtype=torch.float32, device=device)
    wi = window_inputs({r: {**fit_in[r], **_pose_fields(poses[r])} for r in ROLES}, H, ks, bodies, sites, obs, cfg, hcfg)
    names = wi["names"]
    hold_core = names[(ks >= c0) & (ks <= c1)]
    main_hold = max(set(hold_core.tolist()), key=hold_core.tolist().count) if len(hold_core) else "free"
    tmpl = TEMPLATE_OF_ZHOLD.get(main_hold) if (main_hold in TORSO_HOLD_SITES and not cfg.wrist_only) else None
    actor = hc.Actor(fit_in["lead"], "lead", "right", ks, poses["lead"], device, tmpl)
    _ce = np.flatnonzero(w_np > 0.99)
    actor.core = (int(_ce[0]), int(_ce[-1])) if len(_ce) else None
    usable = wi["usable"]
    twod = VisTwoD(torch, obs, actor, device, list(obs.cams), usable)
    # her mesh (fixed) for the planes + exact checks
    bF = hc.Body(fit_in["follow"], device)
    VO = bF.forward_np(poses["follow"][ks], np.asarray(fit_in["follow"]["transl"], float)[ks], verts=True, batch=256)["verts"]
    VOt = torch.as_tensor(VO, device=device)
    planes = hc.Planes(torch, surf)
    # capsules: his right hand / forearm / upper arm (from the actor's FK); hers (fixed)
    capL = bodies["lead"]
    i_hand = caps_index(capL, "hand_R")
    i_farm = caps_index(capL, "forearm_R")
    i_uarm = caps_index(capL, "upper_arm_R")
    capF = bodies["follow"]
    tgt_parts = ["hand_L", "hand_R", "forearm_L", "forearm_R", "upper_arm_L", "upper_arm_R", "shoulder_L", "shoulder_R", "neck_head"]
    hold_hand = (names == "hand_hold")
    j_tgt = np.concatenate([caps_index(capF, p) for p in tgt_parts])
    j_arms = np.concatenate([caps_index(capF, p) for p in ("forearm_L", "forearm_R", "upper_arm_L", "upper_arm_R")])
    j_is_hand = np.isin(j_tgt, np.concatenate([caps_index(capF, "hand_L"), caps_index(capF, "hand_R")]))
    HA = torch.tensor(wi["her_A"], **f32)
    HB = torch.tensor(wi["her_B"], **f32)
    j_tgt_t = torch.as_tensor(j_tgt, device=device)
    j_arms_t = torch.as_tensor(j_arms, device=device)
    rF = torch.tensor(capF["r"], **f32)
    rL = torch.tensor(capL["r"], **f32)
    ihand_t = torch.as_tensor(i_hand, device=device)
    # hand-hold frames: her hands are the partner of the hold (contact wanted): allow 1.5 cm there
    allow_hand = torch.tensor(np.where(hold_hand, 0.015, 0.0), **f32)
    capL_sub = _sub_caps(capL, np.concatenate([i_hand, i_farm]))
    # targets
    site_t = torch.tensor(np.nan_to_num(wi["site_pt"] + hcfg.palm_lift_m * np.nan_to_num(wi["site_n"])), **f32)
    back_t = torch.tensor(np.isin(names, ("upper_back", "wrap_back", "behind_back")).astype(np.float32) * (0.0 if cfg.wrist_only else 1.0),
                          device=device)
    her_up = torch.tensor(torso_frame(wi["kf"]["J"])[:, :, 1], **f32)
    site_n_t = torch.tensor(np.nan_to_num(wi["site_n"]), **f32)
    has_site_t = torch.tensor(wi["has_site"].astype(np.float32), device=device)
    reg_t = torch.tensor(wi["region"], **f32)
    reg_ok_t = torch.tensor(wi["region_ok"], device=device)
    hr = wi["region_ok"].any(1).astype(float) * (0.0 if cfg.wrist_only else 1.0)
    if len(hr) > 3:
        from scipy.ndimage import uniform_filter1d
        hr = np.minimum(hr, uniform_filter1d(hr, 13, mode="nearest") ** 2)          # fades over ~0.2 s on each side of a hold change
    has_reg = torch.tensor(hr.astype(np.float32), device=device)
    ct = hc.coco_targets(fit_in["lead"], "lead", "right", ks, judge, obs, list(obs.cams), bias)
    Xe, ve, nve = ct["elbow"]
    e0 = posed(fit_in["lead"], ks, poses["lead"][ks])["J"][:, HE["right"]]
    ve = ve & (np.linalg.norm(np.nan_to_num(Xe) - e0, axis=1) < 0.25) & (nve >= 2)
    Xe_t = torch.tensor(np.nan_to_num(Xe), **f32)
    ve_t = torch.tensor(ve.astype(np.float32), device=device)
    el2d = CocoElbow2D(torch, obs, judge, ks, bias, device)
    # the site attractor is weak where a usable phone sees the hand (the 2D leads there)
    seen = usable.any(1)
    site_w_t = torch.tensor(np.where(seen, cfg.site_seen_scale, 1.0) * wi["has_site"], **f32)
    # speed caps (hand_contact's): the input's p99 of the hand or its own speed there, + 0.1
    J_in = np.asarray(fit_in["lead"]["joints"], float)[ks]
    ref = (speed_ref or {}).get("lead_right", {"wrist": 2.0, "palm": 2.2})
    tr_in = hc.hand_tracks(J_in, "right")
    caps_sp = [torch.tensor(np.maximum(ref[k], hc.speeds(tr_in[k])) + 0.1, **f32) for k in ("wrist", "palm")]
    wr_rot = torch.tensor((3.0, 3.0, 3.0) if main_hold in TORSO_HOLD_SITES else (8.0, 8.0, 4.0), **f32)
    rc = hc.RefineCfg()

    def capsule_losses(o):
        J, R, tips = o["joints"], o["R"], o["tips"]
        A, B = capsule_points(capL_sub, J, R, tips, xp=torch)
        nh = len(i_hand)
        Ah, Bh = A[:, :nh], B[:, :nh]
        Af_, Bf_ = A[:, nh:], B[:, nh:]
        # his hand vs her limbs / hands / head / girdle
        c = _torch_clearance(torch, Ah, Bh, rL[ihand_t], HA[:, j_tgt_t], HB[:, j_tgt_t], rF[j_tgt_t])
        allow = torch.where(torch.as_tensor(j_is_hand, device=device)[None, None, :], allow_hand[:, None, None], torch.zeros_like(c))
        ov = torch.relu(cfg.cap_margin_m - c - allow)
        l_hand = (ov ** 2).sum((1, 2))
        # his forearm / upper arm vs her arms (a forearm resting on an upper arm is allowed arm_arm_allow_m)
        Aa, Ba = capsule_points(_sub_caps(capL, np.concatenate([i_farm, i_uarm])), J, R, tips, xp=torch)
        ca = _torch_clearance(torch, Aa, Ba, rL[torch.as_tensor(np.concatenate([i_farm, i_uarm]), device=device)], HA[:, j_arms_t],
                              HB[:, j_arms_t], rF[j_arms_t])
        l_arm = (torch.relu(-cfg.arm_arm_allow_m - ca) ** 2).sum((1, 2))
        # his fingers / distal palm vs his own right forearm
        cs = self_forearm_clearance(capL_sub, A, B, "right", xp=torch)
        l_self = torch.relu(cfg.cap_margin_m - cs) ** 2
        return l_hand, l_arm, l_self, c, cs

    # the starting motion's own twist / elbow (the limits never fight what the installed motion already does)
    _wa0 = wrist_angles(poses["lead"][ks][:, :55], "right")
    _ea0 = elbow_angles(poses["lead"][ks][:, :55], "right")
    tw_lim_t = torch.tensor(np.radians(np.maximum(cfg.twist_soft_deg, np.abs(_wa0["twist"]) + 5.0)), **f32)
    hyp_lim_t = torch.tensor(np.radians(np.minimum(-5.0, _ea0["flexion"])), **f32)
    off_lim_t = torch.tensor(np.radians(np.maximum(35.0, np.abs(_ea0["off_hinge"]) + 5.0)), **f32)

    def wrist_loss(wt):
        Rc = actor.chain_R(wt)
        Rw = Rc[:, 2]
        bend, flex, dev, tw = wrist_terms_t(torch, Rw, "right", with_twist=True)
        L = WRIST_LIMITS
        rad = np.pi / 180
        l = cfg.w_bend * torch.relu(bend - cfg.bend_soft_deg * rad) ** 2
        comp = (torch.relu(flex - (L["flexion"] - 3) * rad) ** 2 + torch.relu(-flex - (L["extension"] - 3) * rad) ** 2
                + torch.relu(dev - (L["radial"] - 3) * rad) ** 2 + torch.relu(-dev - (L["ulnar"] - 3) * rad) ** 2)
        l = l + cfg.w_comp * comp
        if cfg.w_twist > 0:
            # the twist is defined while the hand is not folded back onto the forearm: the term fades out from 90 to 150 deg of bend
            fade = ((150 * rad - bend) / (60 * rad)).clamp(0, 1)
            l = l + cfg.w_twist * fade * torch.relu(tw.abs() - tw_lim_t) ** 2
        if cfg.w_elbow_lim > 0:
            ef, eo = elbow_terms_t(torch, Rc[:, 1], "right")
            l = l + cfg.w_elbow_lim * (torch.relu(hyp_lim_t - ef) ** 2 + torch.relu(eo.abs() - off_lim_t) ** 2)
        return l, bend

    terms_last = {}

    def total(o, stage):
        J = o["joints"]
        terms = {}
        loss = 0.0
        pc = o["palm"].mean(1)
        npalm = hc._palm_normal_t(torch, J, "right")
        # stage 1: IK to the site track (+ facing it)
        if stage == 1:
            d2 = ((pc - site_t) ** 2).sum(-1)
            l1 = cfg.w_site_ik * (d2 * site_w_t * w).sum() / wsum
            l1 = l1 + rc.w_normal_flat * ((1 + (npalm * site_n_t).sum(-1)) * site_w_t * w).sum() / wsum
            loss = loss + l1
            terms["site"] = l1.detach()
            if twod.items:
                l2 = cfg.w_2d * twod.loss(o["lm"], cfg.sigma_2d_px, w) / wsum
                loss = loss + l2
                terms["2d"] = l2.detach()
        else:
            # the region: soft-min distance to the hold's sites, beyond the slack
            dd = (pc[:, None] - reg_t).norm(dim=-1)
            dd = torch.where(reg_ok_t, dd, torch.full_like(dd, 9.0))
            soft = -cfg.region_tau_m * torch.logsumexp(-dd / cfg.region_tau_m, dim=1)
            lr = cfg.w_region * ((torch.relu(soft - cfg.region_slack_m) ** 2) * has_reg * w).sum() / wsum
            loss = loss + lr
            terms["region"] = lr.detach()
            # her surface: hand / arm points vs the planes (penetration), the palm patch on the surface for the holds with a region
            P = torch.cat([o["hand"], o["arm"], o["upper"], o["tips5"], o["palm"]], 1)
            sd = planes.sd(P)
            nh, na, nu = actor.nh, actor.na, actor.nu
            hand_sd = torch.cat([sd[:, :nh], sd[:, nh + na + nu:]], 1)
            arm_sd = sd[:, nh:nh + na + nu]
            dh = torch.relu(rc.margin_m - hand_sd) ** 2
            da = torch.relu(-arm_sd) ** 2
            pen = dh.mean(1) + dh.topk(min(12, dh.shape[1]), dim=1).values.mean(1) + 0.5 * (da.mean(1) + da.topk(min(8, da.shape[1]), dim=1).values.mean(1))
            lp = rc.w_pen * (pen * w).sum() / wsum
            loss = loss + lp
            terms["pen"] = lp.detach()
            pd = sd[:, nh + na + nu + 5:]
            near = (planes.near[:, nh + na + nu + 5:].sum(1) > 0).float() * has_reg
            p8 = pd.topk(min(8, pd.shape[1]), dim=1, largest=False).values.mean(1)
            lf = rc.w_flat * (((p8 - rc.palm_offset_m) ** 2) * near * w).sum() / wsum
            lf = lf + rc.w_flat_patch * (((torch.relu(pd - rc.flat_patch_m) ** 2).mean(1)) * near * w).sum() / wsum * (1.0 if tmpl == "open" else 0.3)
            nsurf = (planes.n[:, nh + na + nu + 5:] * planes.near[:, nh + na + nu + 5:, None]).sum(1)
            nsurf = nsurf / nsurf.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            lf = lf + rc.w_normal_flat * ((1 + (npalm * nsurf).sum(-1)) * near * w).sum() / wsum
            loss = loss + lf
            terms["surface"] = lf.detach()
            if actor.u is not None:
                st = sd[:, nh + na + nu:nh + na + nu + 5]
                pull = torch.relu(st - 0.004) * (st < 0.05).float() * has_reg[:, None]
                c_ = actor.curl()
                lt = rc.w_tip * ((pull ** 2).sum(1) * w).sum() / wsum + rc.w_curl * ((c_ - actor.c0) ** 2).sum(-1).mean()
                if len(ks) > 2:
                    lt = lt + rc.w_curl_smooth * ((c_[2:] - 2 * c_[1:-1] + c_[:-2]) ** 2).sum(-1).mean()
                loss = loss + lt
            l_hand, l_arm, l_self, _, _ = capsule_losses(o)
            lc = cfg.w_cap * ((l_hand + l_arm) * w).sum() / wsum + cfg.w_self * (l_self * w).sum() / wsum
            loss = loss + lc
            terms["caps"] = lc.detach()
            fd = J[:, hc.mcp_joints("right")[1]] - J[:, HW["right"]]
            fd = fd / fd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            lfu = cfg.w_fingers_up * ((torch.relu((fd * her_up).sum(-1) - 0.5) ** 2) * back_t * w).sum() / wsum
            loss = loss + lfu
            terms["fingers_up"] = lfu.detach()
            if twod.items:
                l2 = cfg.w_2d * twod.loss(o["lm"], cfg.sigma_2d_px, w) / wsum
                loss = loss + l2
                terms["2d"] = l2.detach()
        lw, _ = wrist_loss(w)
        lw = (lw * w).sum() / wsum
        loss = loss + lw
        terms["wrist"] = lw.detach()
        le = cfg.w_elbow * (hc._rho(torch, ((J[:, HE["right"]] - Xe_t) ** 2).sum(-1), rc.sigma_elbow_m) * ve_t * w).sum() / wsum
        if el2d.n > 0:
            le = le + cfg.w_elbow2d * el2d.loss(J, cfg.sigma_elbow2d_px, w) / wsum
        loss = loss + le
        terms["elbow"] = le.detach()
        if len(ks) > 2:
            lsp = 0.0
            for (X, X0), cap in zip(((J[:, HW["right"]], actor.wrist0), (hc._palm_centre_t(J, "right"), actor.palm_c0)), caps_sp):
                dX = X - X0
                dv = (dX[1:] - dX[:-1]) * HZ
                lsp = lsp + cfg.w_speed * (torch.relu(torch.sqrt((dv ** 2).sum(-1) + 1e-10) - rc.v_free_mps) ** 2).sum() / HZ
                v = (X[1:] - X[:-1]) * HZ
                lsp = lsp + cfg.w_speed * (torch.relu(torch.sqrt((v ** 2).sum(-1) + 1e-10) - cap) ** 2).sum() / HZ
            loss = loss + lsp
            terms["speed"] = lsp.detach()
            acc = actor.delta[2:] - 2 * actor.delta[1:-1] + actor.delta[:-2]
            lsm = rc.w_smooth * (acc ** 2).sum((1, 2)).mean()
            for X in (J[:, HW["right"]], pc):
                ax_ = X[2:] - 2 * X[1:-1] + X[:-2]
                lsm = lsm + rc.w_smooth_pos * (ax_ ** 2).sum(-1).mean()
            loss = loss + lsm
            terms["smooth"] = lsm.detach()
        lrot = ((actor.delta ** 2).sum(-1) * wr_rot).sum(-1).mean()
        loss = loss + lrot
        terms["rot"] = lrot.detach()
        if cfg.wrist_only:
            lk = cfg.w_keep * ((torch.relu((pc - actor.palm_c0).norm(dim=-1) - cfg.keep_slack_m) ** 2) * w).sum() / wsum
            loss = loss + lk
            terms["keep"] = lk.detach()
        terms_last.clear()
        terms_last.update(terms)
        return loss

    def refresh(o, exact=False):
        P = torch.cat([o["hand"], o["arm"], o["upper"], o["tips5"], o["palm"]], 1).detach()
        planes.refresh(P, VOt, inside_test=exact)

    def optimise(stage, iters, lr, every=15, exact=False):
        params = actor.params(stage if stage == 2 else 1)
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(iters):
            o = actor.forward(w)
            if stage == 2 and it % every == 0:
                refresh(o, exact)
            loss = total(o, stage)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            o = actor.forward(w)
            if stage == 2:
                refresh(o, exact)
            return float(total(o, stage)), {k: float(v) for k, v in terms_last.items()}

    # ------------------------------------------------------------------ measures before
    def measures(o_np_pose) -> dict:
        """capsule clearances of the hand (min over her parts, per frame), own forearm, wrist, palm-region distance, on the window"""
        kl_ = posed(fit_in["lead"], ks, o_np_pose)
        A, B = capsule_points(capL, kl_["J"], kl_["R"], kl_["tips"])
        cl = clearance(capL, A, B, capF, wi["her_A"], wi["her_B"], i_hand, j_tgt)
        cl = np.where(j_is_hand[None, None, :] & hold_hand[:, None, None], cl + 0.015, cl)
        ia = np.concatenate([i_farm, i_uarm])
        ca = clearance(capL, A, B, capF, wi["her_A"], wi["her_B"], ia, j_arms) + cfg.arm_arm_allow_m
        cs = self_forearm_clearance(capL, A, B, "right")
        wa = wrist_angles(o_np_pose[:, :55], "right")
        ex = limit_excess(wa)
        wex = np.max(np.stack([ex[k] for k in ("flexion", "extension", "radial", "ulnar", "bend")], 1), 1)
        pc, _ = hand_site(kl_["J"], "right", "palm")
        dreg = np.linalg.norm(wi["region"] - pc[:, None], axis=-1)
        dreg = np.where(wi["region_ok"], dreg, np.inf).min(1)
        return {"hand_limbs_cm": -np.minimum(cl.reshape(len(ks), -1).min(1), 0) * 100, "arm_arms_cm": -np.minimum(ca.reshape(len(ks), -1).min(1), 0) * 100,
                "self_cm": -np.minimum(cs, 0) * 100, "wrist_excess_deg": wex, "bend": wa["bend"], "region_m": dreg, "J": kl_["J"],
                "tips": kl_["tips"][:, 5:10], "pose": o_np_pose}

    core_np = w_np > 0.99
    pose0 = poses["lead"][ks].copy()
    M0 = measures(pose0)
    with torch.no_grad():
        o0 = actor.forward(w)
    res = {"core": [int(c0), int(c1)], "frames": [int(a0), int(a1)], "ramp": ramp, "hold": main_hold, "template": tmpl,
           "holds": {h: int((hold_core == h).sum()) for h in np.unique(hold_core)}, "elbow_coco_share": float(ve.mean()),
           "phones_usable_2d_mean": float(usable.sum(1).mean())}
    # stage 1 (holds with a site)
    if wi["has_site"][core_np].mean() > 0.5 and not cfg.wrist_only:
        actor.fingers_on = False
        l1, t1 = optimise(1, cfg.iters1, rc.lr)
        res["stage1"] = {"loss": l1, **t1}
        # a palm facing away from her surface after the IK: also try the forearm twisted (hand_contact.twist_start)
        with torch.no_grad():
            o1 = actor.forward(w)
        n1 = hc.palm_normal(o1["joints"].cpu().numpy(), "right")
        ang = np.degrees(np.arccos(np.clip((n1 * -np.nan_to_num(wi["site_n"])).sum(1), -1, 1)))
        res["palm_angle_deg_stage1"] = float(np.median(ang[core_np & wi["has_site"]])) if (core_np & wi["has_site"]).any() else None
    starts = [actor.snapshot()]
    if res.get("palm_angle_deg_stage1") is not None and res["palm_angle_deg_stage1"] > rc.flip_deg:
        hc.twist_start(actor, w, np.nan_to_num(wi["site_n"]))
        starts.append(actor.snapshot())
    actor.fingers_on = actor.u is not None
    tried = []
    for st in starts:
        actor.restore(st)
        l2, t2 = optimise(2, cfg.iters2, rc.lr2)
        tried.append((l2, t2, actor.snapshot()))
    best = int(np.argmin([t[0] for t in tried]))
    actor.restore(tried[best][2])
    lp, tp = optimise(2, cfg.iters_polish, rc.lr2 / 3, every=3)
    # push-out with the exact inside test where her mesh still holds a hand / arm point
    if cfg.iters_push:
        lp, tp = optimise(2, cfg.iters_push, rc.lr2 / 3, every=2, exact=True)
    res["stage2"] = {"loss": lp, **tp, "starts": len(starts), "chosen": best}
    with torch.no_grad():
        o1 = actor.forward(w)
    pose1 = actor.pose_np(w_np)
    M1 = measures(pose1)

    def pct(a, msk):
        return float(np.percentile(a[msk], 90)) if msk.any() else 0.0
    def stats(M, k):
        v = M[k][core_np] if core_np.any() else np.zeros(1)
        return {"p90": round(float(np.percentile(v, 90)), 2), "p95": round(float(np.percentile(v, 95)), 2), "max": round(float(v.max()), 2),
                "frames_gt_2": int((v > 2.0).sum()), "frames_gt_10": int((v > 10.0).sum())}
    res["before"] = {k: stats(M0, k) for k in ("hand_limbs_cm", "arm_arms_cm", "self_cm", "wrist_excess_deg")}
    res["after"] = {k: stats(M1, k) for k in ("hand_limbs_cm", "arm_arms_cm", "self_cm", "wrist_excess_deg")}
    fin = np.isfinite(M0["region_m"]) & core_np
    res["region_m_median"] = [round(float(np.median(M0["region_m"][fin])), 3) if fin.any() else None,
                              round(float(np.median(M1["region_m"][fin])), 3) if fin.any() else None]
    # exact: her mesh vs his hand (deepest hand point inside her, winding numbers) before / after
    with torch.no_grad():
        def exact_pen(o):
            P = torch.cat([o["hand"], o["palm"]], 1).detach().float()
            q = hc.query_batched(surf, P, VOt, bs=8, wn_within=0.03, inside_test=cfg.exact_check)
            return torch.relu(-q["sd"].min(1).values).cpu().numpy().astype(float) * 100
        pe0, pe1 = exact_pen(o0), exact_pen(o1)
    res["exact_pen_cm"] = {"before_p90": round(pct(pe0, core_np), 2), "after_p90": round(pct(pe1, core_np), 2),
                           "before_max": round(float(pe0[core_np].max()), 2) if core_np.any() else 0.0,
                           "after_max": round(float(pe1[core_np].max()), 2) if core_np.any() else 0.0}
    # 2D, visibility-aware (MAMMA hand points in the usable phones) and the independent COCO elbow
    res["ma2d_vis_px"] = [twod.core_median(o0["lm"], core_np, actor.n_hand_lm) if twod.items else None,
                          twod.core_median(o1["lm"], core_np, actor.n_hand_lm) if twod.items else None]

    def agreeing(o) -> tuple[int, dict]:
        """phones x core frames whose median hand-point error is below agree_px (the usable phones); the per-phone medians"""
        n, per = 0, {}
        with torch.no_grad():
            for cam, e, v in twod.errors(o["lm"]):
                e, v = e[:, :actor.n_hand_lm], v[:, :actor.n_hand_lm]
                ev = torch.where(v, e, torch.full_like(e, float("nan")))
                med = torch.nanmedian(ev, dim=1).values.cpu().numpy()
                ok = (v.sum(1).cpu().numpy() >= 5) & core_np
                n += int((ok & (med < cfg.agree_px)).sum())
                if ok.any():
                    per[cam] = round(float(np.nanmedian(med[ok])), 1)
        return n, per
    na0, pc0 = agreeing(o0)
    na1, pc1 = agreeing(o1)
    res["agreeing_phone_frames"] = [na0, na1]
    res["ma2d_per_phone_px"] = {"before": pc0, "after": pc1}

    def coco_elbow_px(J):
        b = np.asarray(bias.get("lead_right_elbow", [0, 0, 0]), float)
        X = J[:, HE["right"]] + np.einsum("kij,j->ki", hc._forearm_frame(J, "right"), b)
        e = hc._coco_errors(judge, obs, X[core_np], ks[core_np], "lead", "right", "elbow", obs.cams)
        return float(np.median(e)) if len(e) else None
    res["coco_elbow_px"] = [coco_elbow_px(M0["J"]), coco_elbow_px(M1["J"])]
    # guards
    why = []
    ncore = int(core_np.sum())
    for k in ("hand_limbs_cm", "arm_arms_cm", "self_cm"):
        b_, a_ = res["before"][k], res["after"][k]
        sl = cfg.guard_cap_slack_cm if k != "arm_arms_cm" else 1.0
        if a_["p95"] > max(b_["p95"], cfg.pen_cm) + sl or a_["frames_gt_2"] > b_["frames_gt_2"] + 3 + 0.02 * ncore:
            why.append(f"new capsule collision {k}: p95 {b_['p95']:.2f} -> {a_['p95']:.2f} cm, frames > 2 cm {b_['frames_gt_2']} -> {a_['frames_gt_2']}")
    inside = (M1["wrist_excess_deg"][core_np] <= 3.0).mean() if core_np.any() else 1.0
    res["wrist_inside_share"] = [round(float((M0["wrist_excess_deg"][core_np] <= 3.0).mean()), 3) if core_np.any() else 1.0, round(float(inside), 3)]
    if inside < res["wrist_inside_share"][0] - 0.02 or res["after"]["wrist_excess_deg"]["p95"] > res["before"]["wrist_excess_deg"]["p95"] + 2.0:
        why.append(f"wrist further beyond its limits: inside {res['wrist_inside_share']}, p95 excess "
                   f"{res['before']['wrist_excess_deg']['p95']} -> {res['after']['wrist_excess_deg']['p95']} deg")
    was_off = res["region_m_median"][0] is not None and res["region_m_median"][0] > ON_TARGET_M
    slack = 0.5 if was_off else cfg.guard_pen_slack_cm
    if res["exact_pen_cm"]["after_p90"] > max(res["exact_pen_cm"]["before_p90"], 0.3) + slack:
        why.append(f"new penetration (exact) p90 {res['exact_pen_cm']['before_p90']} -> {res['exact_pen_cm']['after_p90']} cm")
    if na0 > 0 and na1 < (1 - cfg.guard_agree_drop) * na0:
        why.append(f"fewer phones agree with the hand (MAMMA points < {cfg.agree_px:.0f} px): {na0} -> {na1} phone-frames")
    b2, a2 = res["ma2d_vis_px"]
    if b2 is not None and a2 is not None and np.isfinite(b2) and np.isfinite(a2) and a2 > b2 + cfg.guard_ma2d_slack_px:
        why.append(f"MAMMA hand points (usable phones) {b2:.1f} -> {a2:.1f} px")
    b3, a3 = res["coco_elbow_px"]
    if b3 is not None and a3 is not None and a3 > b3 + cfg.guard_coco_slack_px:
        why.append(f"COCO elbow {b3:.1f} -> {a3:.1f} px")
    # speed against the INPUT fit (hand_contact's audit rule)
    J_new = M1["J"]
    res["speed"] = {}
    ramp_need = ramp
    for k in ("wrist", "palm"):
        ti, tn = hc.hand_tracks(J_in, "right")[k], hc.hand_tracks(J_new, "right")[k]
        sv = hc.speed_violations(ti, tn, ref[k], rc.add_speed_max_mps, rc.speed_margin_mps)
        res["speed"][k] = {"max_added_mps": round(sv["max_add"], 3), "jumps": int(len(sv["jump"])), "fast": int(len(sv["fast"]))}
        bad = np.union1d(sv["jump"], sv["fast"])
        if len(bad):
            j = int(bad[np.argmax(sv["add"][bad])])
            why.append(f"{'jump' if len(sv['jump']) else 'whip'} {k} at {ks[j] / HZ:.2f} s added {sv['add'][j]:.2f} m/s (speed {sv['v_new'][j]:.2f})")
        tb = hc.hand_tracks(M0["J"], "right")[k]
        ce = np.flatnonzero(core_np)
        if len(ce):
            ends = np.concatenate([ce[:3], ce[-3:]])
            ramp_need = max(ramp_need, hc.ramp_for(float(np.linalg.norm(tn[ends] - tb[ends], axis=1).max()), cfg.v_ramp_mps, cfg.ramp_min, cfg.ramp_max))
    if cfg.gate_guards:
        g_why, g_ramp = _window_gate(fit_in, ks, M0, M1, core_np, ref, cfg)
        why += g_why
        ramp_need = max(ramp_need, g_ramp)

        def coco_wrist_px(J):
            b = np.asarray(bias.get("lead_right_wrist", [0, 0, 0]), float)
            X = J[:, HW["right"]] + np.einsum("kij,j->ki", hc._forearm_frame(J, "right"), b)
            e = hc._coco_errors(judge, obs, X[core_np], ks[core_np], "lead", "right", "wrist", obs.cams)
            return (float(np.median(e)), len(e)) if len(e) else (None, 0)
        (cw0, n0), (cw1, n1) = coco_wrist_px(M0["J"]), coco_wrist_px(M1["J"])
        res["coco_wrist_px"] = [cw0, cw1]
        if cw0 is not None and cw1 is not None and min(n0, n1) >= GATE["coco_min_n"] and cw1 > cw0 + GATE["coco_worse_px"]:
            why.append(f"COCO wrist {cw0:.1f} -> {cw1:.1f} px")
    res["ramp_needed"] = int(ramp_need)
    res["adopted"] = not why
    res["guard"] = why
    res["new_pose"] = (ks, pose1, list(actor.chain) + list(actor.fj))
    return res


def _pose_fields(P: np.ndarray) -> dict:
    T = len(P)
    return {"global_orient": P[:, 0], "body_pose": P[:, 1:22].reshape(T, 63), "left_hand_pose": P[:, 25:40].reshape(T, 45),
            "right_hand_pose": P[:, 40:55].reshape(T, 45)}


# ------------------------------------------------------------------------------------------------ running the windows (pod), merging

def split_runs(wins: list, n: int) -> list[list[int]]:
    """window indices -> n parts: the chunks of one run stay together and in order; runs dealt out by frames, largest first"""
    runs: dict = {}
    order = sorted(range(len(wins)), key=lambda i: wins[i]["core"][0])
    key, last_end = None, -10 ** 9
    for i in order:
        w = wins[i]
        if key is None or w["core"][0] - 2 * 45 - 5 > last_end:          # 2 x ramp_max apart: independent
            key = i
        runs.setdefault(key, []).append(i)
        last_end = max(last_end, w["core"][1])
    parts: list[list[int]] = [[] for _ in range(n)]
    load = [0] * n

    def work(ix):
        return sum(wins[i]["core"][1] - wins[i]["core"][0] + 80 for i in ix)
    for ix in sorted(runs.values(), key=work, reverse=True):
        j = int(np.argmin(load))
        parts[j] += ix
        load[j] += work(ix)
    return [sorted(p, key=lambda i: wins[i]["core"][0]) for p in parts]


def run_refine(part: tuple[int, int] | None = None, device: str = "cuda", only: list[int] | None = None, cfg: ZoukCfg | None = None,
               tag: str = "all") -> dict:
    """the windows of plan.json (this part of them) -> v4/contacts/refine_<tag>[_p<i>].json + poses_<tag>[_p<i>].npz (the lead's poses)"""
    import torch
    from . import gpu
    cfg = cfg or ZoukCfg()
    hcfg = HoldCfg()
    plan = json.loads((OUT / cfg.plan).read_text(encoding="utf-8"))
    wins = plan["windows"]
    idx = list(range(len(wins)))
    if part is not None:
        idx = split_runs(wins, part[1])[part[0]]
    if only is not None:
        idx = [i for i in idx if i in set(only)]
    fit_in = load_fit("input")
    base = load_fit(cfg.base)
    poses = {r: hc.full_pose(base[r]).copy() for r in ROLES}
    H = load_holds()
    bodies = capsule_bodies(fit_in)
    sites = site_bodies(fit_in)
    obs = hc.Obs()
    judge = hc.CocoJudge()
    bias = hc.calibrate_coco_bias(fit_in, obs, judge)
    speed_ref = hc.input_speed_ref(fit_in)
    sfx = f"_p{part[0]}" if part is not None else ""
    rd = OUT / "runs" / tag
    rd.mkdir(parents=True, exist_ok=True)
    results = []
    tried = poses["lead"].copy()                     # review only: every solve applied, adopted or not
    t0 = time.time()
    with gpu.exclusive(f"zouk_contacts refine {tag}{sfx}"):
        faces = hc.Body(fit_in["lead"], "cpu").faces
        surf = hc.GpuSurface(faces, len(hc.vertex_joint()), device)
        for n_, i in enumerate(idx):
            w = dict(wins[i])
            te = time.time()
            attempts = []
            res = None
            for attempt in range(3):
                try:
                    res = refine_window(fit_in, poses, w, H, obs, judge, bias, surf, bodies, sites, cfg, hcfg, device, speed_ref)
                except Exception as ex:  # noqa: BLE001 - one window must not stop the run; reported
                    import traceback
                    res = {"adopted": False, "guard": [f"{type(ex).__name__}: {ex}"], "traceback": traceback.format_exc()[-2000:]}
                attempts.append({"ramp": w.get("ramp", cfg.ramp_min), "adopted": bool(res.get("adopted")), "guard": res.get("guard")})
                if res.get("adopted"):
                    break
                g = res.get("guard") or []
                if any(x.startswith(("jump", "whip")) for x in g) and w.get("ramp", cfg.ramp_min) < cfg.ramp_max:
                    w["ramp"] = int(min(cfg.ramp_max, max(res.get("ramp_needed", 0), round(1.6 * w.get("ramp", cfg.ramp_min)))))
                    continue
                break
            if res.get("new_pose"):
                ks, P, joints = res["new_pose"]
                tried[np.ix_(ks, joints)] = P[:, joints]
                if res.get("adopted"):
                    poses["lead"][np.ix_(ks, joints)] = P[:, joints]
            rec = {k: v for k, v in res.items() if k != "new_pose"}
            rec.update({"window": i, "t": w["t"], "attempts": attempts, "seconds": round(time.time() - te, 1)})
            results.append(rec)
            log(f"refine [{tag}{sfx}] {n_ + 1}/{len(idx)}: window {i} {w['t']} {res.get('hold')} adopted={res.get('adopted')} "
                f"{res.get('guard') or ''} ({rec['seconds']} s, total {time.time() - t0:.0f} s)")
            if device != "cpu":
                torch.cuda.empty_cache()
    summ = {"windows": len(idx), "adopted": sum(bool(r.get("adopted")) for r in results), "seconds": round(time.time() - t0, 1)}
    rep = {"what": "zouk_contacts.refine_window over plan.json", "config": asdict(cfg), "part": list(part) if part else None, "summary": summ,
           "results": results}
    write_json(rd / f"refine{sfx}.json", rep)
    np.savez_compressed(rd / f"poses{sfx}.npz", lead=poses["lead"])
    np.savez_compressed(rd / f"poses_tried{sfx}.npz", lead=tried)
    log(f"[{tag}{sfx}] {summ}")
    return rep


def run_parallel(n: int = 4, tag: str = "all", only: list[int] | None = None, cfg_over: dict | None = None, write: bool = False) -> dict:
    """run_refine in n processes (disjoint runs of windows; the GPU is launch-bound), then merge"""
    import os
    import subprocess
    t0 = time.time()
    env = {**os.environ, "DANCECAP_GPU_SHARED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    rd = OUT / "runs" / tag
    rd.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(n):
        lf = open(rd / f"refine_p{i}.log", "w", encoding="utf-8")
        procs.append((subprocess.Popen([sys.executable, "-m", "dancecap.zouk_contacts", "refine-part", str(i), str(n), "--tag", tag]
                                       + (["--only", ",".join(str(x) for x in only)] if only else []) + (["--cfg", json.dumps(cfg_over)] if cfg_over else []),
                                       cwd=str(hc.PACKAGE_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
    for pr, lf in procs:
        pr.wait()
        lf.close()
    return merge_parts(n, tag, round(time.time() - t0, 1), write=write, cfg_over=cfg_over)


def merge_parts(n: int, tag: str = "all", seconds: float | None = None, write: bool = True, cfg_over: dict | None = None) -> dict:
    """the parts' poses (the lead's arm, disjoint windows) over the installed motion -> v4/smplx_contact/ (the follower is the installed one)"""
    cfg = ZoukCfg(**(cfg_over or {}))
    base = load_fit(cfg.base)
    P0 = hc.full_pose(base["lead"])
    P = P0.copy()
    results, conflicts = [], 0
    owner = np.full(P.shape[:2], -1)
    rd = OUT / "runs" / tag
    Pt = P0.copy()
    for i in range(n):
        if not (rd / f"refine_p{i}.json").exists():
            continue
        rp = json.loads((rd / f"refine_p{i}.json").read_text(encoding="utf-8"))
        results += rp["results"]
        with np.load(rd / f"poses_p{i}.npz") as z:
            ch = np.abs(z["lead"] - P0).max(2) > 1e-9
            conflicts += int((owner[ch] >= 0).sum())
            owner[ch] = i
            P[ch] = z["lead"][ch]
        if (rd / f"poses_tried_p{i}.npz").exists():
            with np.load(rd / f"poses_tried_p{i}.npz") as z:
                ch = np.abs(z["lead"] - P0).max(2) > 1e-9
                Pt[ch] = z["lead"][ch]
    results.sort(key=lambda r: r["t"][0])
    summ = {"windows": len(results), "adopted": sum(bool(r.get("adopted")) for r in results), "seconds": seconds, "parts": n,
            "joint_frame_conflicts_between_parts": conflicts,
            "adopted_seconds": round(sum(r["core"][1] - r["core"][0] + 1 for r in results if r.get("adopted")) / HZ, 1)}
    rep = {"what": f"zouk_contacts resolution over the motion {cfg.base} (plan {cfg.plan})", "config": asdict(cfg), "summary": summ,
           "results": results}
    np.savez_compressed(rd / "poses_merged.npz", lead=P)
    np.savez_compressed(rd / "poses_tried_merged.npz", lead=Pt)
    if write:
        rep["written"] = write_fit_v4(base, {"lead": P, "follow": hc.full_pose(base["follow"])}, FIT_OUT)
    write_json(rd / "refine.json", rep)
    log(f"merged {summ}")
    return rep


def write_fit_v4(base: dict, poses: dict, out_dir: Path) -> dict:
    """the contract npz: the installed fit's keys with the refined poses and the FK joints recomputed on the changed frames"""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for r in ROLES:
        d = dict(base[r])
        P = poses[r]
        T = len(P)
        changed = np.flatnonzero(np.abs(P - hc.full_pose(base[r])).max((1, 2)) > 0)
        changed_v3 = np.flatnonzero(np.abs(P - hc.full_pose(load_fit("v3")[r])).max((1, 2)) > 0)
        d["global_orient"] = P[:, 0].copy()
        d["body_pose"] = P[:, 1:22].reshape(T, 63).copy()
        d["left_hand_pose"] = P[:, 25:40].reshape(T, 45).copy()
        d["right_hand_pose"] = P[:, 40:55].reshape(T, 45).copy()
        J = np.asarray(base[r]["joints"], float).copy()
        if len(changed):
            J[changed] = posed(base[r], changed, P[changed])["J"]
        d["joints"] = J
        src = str(base[r].get("source", ""))
        if "v4 zouk contacts" not in src:
            src += " | v4 zouk contacts + collisions (dancecap.zouk_contacts, 2026-10-22) of v3/smplx_contact"
        d["source"] = np.array(src)
        np.savez(out_dir / f"{r}.npz", **d)
        info[r] = {"frames_changed_vs_installed": int(len(changed_v3)), "frames_changed_this_pass": int(len(changed)), "file": str(out_dir / f"{r}.npz")}
    info["fk_check_mm"] = fk_check_v4(out_dir)
    write_json(out_dir / "zouk_contacts.json", {"what": "v3/smplx_contact (installed) + the v4 zouk contact / collision resolution of the leader's "
                                                        "right hand (dancecap.zouk_contacts)", "base": str(hc.FIT_OUT), "info": info})
    return info


def fk_check_v4(d: Path, n: int = 400) -> dict:
    """max |smplx FK joints - stored joints| (mm) on a sample of changed + random frames, with the smplx package (the exporter's contract)"""
    from . import mamma_dyn as md
    out = {}
    for r in ROLES:
        z = dict(np.load(d / f"{r}.npz", allow_pickle=True))
        src = dict(np.load(hc.FIT_OUT / f"{r}.npz", allow_pickle=True))
        P, P0 = hc.full_pose(z), hc.full_pose(src)
        ch = np.flatnonzero(np.abs(P - P0).max((1, 2)) > 0)
        rng = np.random.default_rng(1)
        ks = np.unique(np.concatenate([ch[np.linspace(0, len(ch) - 1, min(n, len(ch))).astype(int)] if len(ch) else np.zeros(0, int),
                                       rng.choice(len(P), 100, replace=False)]))
        pose = np.concatenate([z["global_orient"], z["body_pose"], z["jaw_pose"], z["leye_pose"], z["reye_pose"], z["left_hand_pose"],
                               z["right_hand_pose"]], 1)[ks]
        _, J = md._forward(pose, z["transl"][ks], z["betas"], bool(z["flat_hand_mean"]))
        out[r] = round(float(np.abs(J - z["joints"][ks]).max()) * 1000, 3)
    return out


# ------------------------------------------------------------------------------------------------ exact check of the capsule numbers (pod)
#
# The capsules are fitted to the rest mesh (forearms / hands within ~1 cm, torso slabs 1-3 cm). Here the leader's right hand (and forearm)
# against HER posed SMPL-X mesh, exactly: closest points + generalized winding numbers (hand_contact.GpuSurface), per frame the deepest hand
# point and, per part of her body (by the dominant skinning joint of the closest vertex), the deepest hand point whose closest point lies on
# that part. The own forearm (folding) has no closed surface: capsules + the wrist angles only.

EXACT_PARTS = ("torso", "forearm_L", "forearm_R", "upper_arm_L", "upper_arm_R", "shoulder_L", "shoulder_R", "hand_L", "hand_R", "neck_head", "leg")


def _part_of_joint() -> np.ndarray:
    m = np.full(55, EXACT_PARTS.index("torso"), np.int8)
    for s in SIDES:
        x = SX[s]
        m[ELBOW[s]] = EXACT_PARTS.index(f"forearm_{x}")
        m[SHOULDER[s]] = EXACT_PARTS.index(f"upper_arm_{x}")
        m[COLLAR[s]] = EXACT_PARTS.index(f"shoulder_{x}")
        m[WRIST[s]] = EXACT_PARTS.index(f"hand_{x}")
        m[list(range(25, 40) if s == "left" else range(40, 55))] = EXACT_PARTS.index(f"hand_{x}")
        for j in (HIP[s], KNEE[s], ANKLE[s], FOOT[s]):
            m[j] = EXACT_PARTS.index("leg")
    for j in (NECK, HEAD, 22, 23, 24):
        m[j] = EXACT_PARTS.index("neck_head")
    return m


def exact_measure(name: str, device: str = "cuda", batch: int = 48, ks: np.ndarray | None = None) -> dict:
    """the leader's right hand (and forearm) vs her mesh, every frame (or ks) -> v4/contacts/exact_<name>.npz: hand_pen (T,) deepest hand
    point inside her (m, >= 0), hand_pen_part (T, P), arm_pen (T,) deepest forearm point, hand_gap (T,) signed min distance"""
    import torch
    t0 = time.time()
    fit = load_fit(name)
    T = len(fit["lead"]["t"])
    ks = np.arange(T) if ks is None else np.asarray(ks)
    ids = hc.rest_hand_ids(fit["lead"]["betas"])["right"]
    pts = np.concatenate([ids.hand, ids.palm, ids.arm])
    nh, npm = len(ids.hand), len(ids.palm)
    bl, bf = hc.Body(fit["lead"], device), hc.Body(fit["follow"], device)
    surf = hc.GpuSurface(bl.faces, len(hc.vertex_joint()), device)
    pj = _part_of_joint()[hc.vertex_joint()]
    n = len(ks)
    out = {"hand_pen": np.zeros(n), "arm_pen": np.zeros(n), "hand_gap": np.zeros(n), "hand_pen_part": np.zeros((n, len(EXACT_PARTS)))}
    PL, PF = hc.full_pose(fit["lead"]), hc.full_pose(fit["follow"])
    trl, trf = np.asarray(fit["lead"]["transl"], float), np.asarray(fit["follow"]["transl"], float)
    with torch.no_grad():
        for a in range(0, n, batch):
            kk = ks[a:a + batch]
            VL = bl.forward_np(PL[kk], trl[kk], verts=True, batch=batch, ids=pts)["verts"]
            VF = bf.forward_np(PF[kk], trf[kk], verts=True, batch=batch)["verts"]
            q = surf.query(torch.as_tensor(VL, device=device), torch.as_tensor(VF, device=device), wn_within=0.03)
            sd = q["sd"].cpu().numpy()
            part = pj[q["vert"].cpu().numpy()]
            sl = slice(a, a + len(kk))
            hsd = sd[:, :nh + npm]
            out["hand_pen"][sl] = np.maximum(0, -hsd.min(1))
            out["hand_gap"][sl] = hsd.min(1)
            out["arm_pen"][sl] = np.maximum(0, -sd[:, nh + npm:].min(1))
            hp = part[:, :nh + npm]
            for pi in range(len(EXACT_PARTS)):
                v = np.where(hp == pi, hsd, np.inf).min(1)
                out["hand_pen_part"][sl, pi] = np.maximum(0, -np.where(np.isfinite(v), v, 0))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"exact_{name}.npz", ks=ks, parts=np.array(EXACT_PARTS), **out)
    log(f"exact {name}: {n} frames in {time.time() - t0:.0f} s")
    return out


def exact_table(name: str, H: dict | None = None) -> dict:
    """frames / seconds / depth of the exact penetration of the leader's right hand per part of her body (+ per hold)"""
    with np.load(OUT / f"exact_{name}.npz") as z:
        d = {k: z[k] for k in z.files}
    T = len(d["hand_pen"])
    out = {"hand_any": _depth_stats(-d["hand_pen"]), "forearm_any": _depth_stats(-d["arm_pen"])}
    for i, p in enumerate(EXACT_PARTS):
        st = _depth_stats(-d["hand_pen_part"][:, i])
        if st["frames_gt_0.5cm"]:
            out[f"hand_vs_{p}"] = st
    if H is not None and len(H["names"]) == T:
        out["by_hold"] = {h: {"hand": _depth_stats(-d["hand_pen"], H["names"] == h), "forearm": _depth_stats(-d["arm_pen"], H["names"] == h)}
                          for h in HOLD_NAMES if (H["names"] == h).sum() >= 15}
    return out


# ------------------------------------------------------------------------------------------------ validation (before / after over the film)

ON_TARGET_M = 0.06          # a palm within this of its hold's region (the sites of the hold on her posed body) ...
ON_SURFACE_M = 0.03         # ... at most this far from her surface (exact) and not inside her (exact depth < INSIDE_OK_M)
INSIDE_OK_M = 0.01


def on_target(fit: dict, H: dict, ks: np.ndarray, bodies: dict, sites: dict, exact: dict | None) -> np.ndarray:
    """per frame of ks (in a hold with a region): the palm on its hold's region and on her surface"""
    rd = region_distance(fit, H, ks, bodies, sites)
    ok = rd <= ON_TARGET_M
    if exact is not None:
        pos = {int(k): i for i, k in enumerate(exact["ks"])}
        ii = np.array([pos.get(int(k), -1) for k in ks])
        have = ii >= 0
        pen = np.where(have, exact["hand_pen"][np.maximum(ii, 0)], 0.0)
        gap = np.where(have, exact["hand_gap"][np.maximum(ii, 0)], 0.0)
        ok &= (pen < INSIDE_OK_M) & (gap < ON_SURFACE_M)
    return ok


def hand_2d_errors(fit: dict, ks: np.ndarray, obs, hcfg: HoldCfg) -> np.ndarray:
    """per frame of ks: the median px error of the fit's right-hand MAMMA points against MAMMA's VISIBLE hand points, over the phones that
    see the hand (>= min_lm visible points); NaN where no phone sees it"""
    parts = hc.landmark_parts()
    hid = parts["right_hand"]
    out = np.full((len(ks), len(obs.cams)), np.nan)
    for a in range(0, len(ks), 2000):
        kk = ks[a:a + 2000]
        hp = hc.hand_points_np(fit["lead"], hc.full_pose(fit["lead"]), kk, "right")
        for ci, c in enumerate(obs.cams):
            z = obs.z[c]
            see, _ = hand_2d(obs, c, kk, hcfg)
            uv, dep = obs.project(c, hp["lm"], kk)
            o = z["lm"][kk, hc.PIDX["lead"]][:, hid]
            vv = (z["vis"][kk, hc.PIDX["lead"]][:, hid] > hcfg.vis) & (np.abs(o).sum(-1) > 0) & (dep > 0.1)
            e = np.where(vv, np.linalg.norm(uv - o, axis=-1), np.nan)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(e, 1)
            out[a:a + len(kk), ci] = np.where(see, med, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(out, 1)


def coco_px(fit: dict, ks: np.ndarray, obs, judge, bias: dict, which: str) -> np.ndarray:
    """all COCO detections (px errors) of the leader's right wrist / elbow (bias-corrected joint) over ks, every phone"""
    J = posed(fit["lead"], ks, hc.full_pose(fit["lead"])[ks])["J"]
    jj = WRIST["right"] if which == "wrist" else ELBOW["right"]
    b = np.asarray(bias.get(f"lead_right_{which}", [0, 0, 0]), float)
    X = J[:, jj] + np.einsum("kij,j->ki", hc._forearm_frame(J, "right"), b)
    return hc._coco_errors(judge, obs, X, ks, "lead", "right", which, obs.cams)


def validate(fits=("input", "v3", "v4")) -> dict:
    """before / after over the whole film (input fit, the installed v3 motion, the v4 result): capsule collisions of the leader's right
    hand per pair, the exact penetration (when exact_<fit>.npz exist), the wrist limits, the on-target rate per hold, MAMMA's hand points
    (phones that see the hand) and the COCO wrist / elbow on the changed frames, the speed audit against the input -> validate.json"""
    t0 = time.time()
    H = load_holds()
    fl = {f: load_fit(f) for f in fits}
    bodies = capsule_bodies(fl["input"])
    sites = site_bodies(fl["input"])
    obs = hc.Obs()
    judge = hc.CocoJudge()
    bias = hc.calibrate_coco_bias(fl["input"], obs, judge)
    hcfg = HoldCfg()
    T = len(H["names"])
    P3, P4 = hc.full_pose(fl["v3"]["lead"]), hc.full_pose(fl["v4"]["lead"]) if "v4" in fl else None
    changed = np.flatnonzero(np.abs(P4 - P3).max((1, 2)) > 0) if P4 is not None else np.zeros(0, int)
    rep = {"what": "zouk_contacts validation: input = smplx_rf2_noslip, v3 = installed (v3/smplx_contact), v4 = v4/smplx_contact; the leader's "
                   "right hand", "frames_changed_v4_vs_v3": int(len(changed)), "seconds_changed": round(len(changed) / HZ, 1), "fits": {}}
    torso_holds = [h for h in TORSO_HOLD_SITES if (H["names"] == h).sum() >= 15]
    hold_frames = np.flatnonzero(np.isin(H["names"], torso_holds))
    for f, fit in fl.items():
        m = load_measure(f) if (OUT / f"measure_{f}.npz").exists() else run_measure(f)
        ex = None
        if (OUT / f"exact_{f}.npz").exists():
            with np.load(OUT / f"exact_{f}.npz") as z:
                ex = {k: z[k] for k in z.files}
        r = {"collisions": hand_collision_table(m, "lead_R", caps={"self": bodies["lead"], "other": bodies["follow"]}),
             "wrist": wrist_table(m, "lead_R"), "exact": exact_table(f, H) if ex is not None else None}
        ot = on_target(fit, H, hold_frames, bodies, sites, ex)
        r["on_target"] = {h: {"seconds": round(float((H["names"][hold_frames] == h).sum()) / HZ, 1),
                              "rate": round(float(ot[H["names"][hold_frames] == h].mean()), 3)} for h in torso_holds}
        r["on_target"]["all_torso_holds"] = {"seconds": round(len(hold_frames) / HZ, 1), "rate": round(float(ot.mean()), 3)}
        e2 = hand_2d_errors(fit, hold_frames, obs, hcfg)
        r["mamma_hand_px_hold_frames"] = {"median": round(float(np.nanmedian(e2)), 2) if np.isfinite(e2).any() else None,
                                          "frames_seen": int(np.isfinite(e2).sum())}
        if len(changed):
            e2c = hand_2d_errors(fit, changed, obs, hcfg)
            r["mamma_hand_px_changed_frames"] = {"median": round(float(np.nanmedian(e2c)), 2) if np.isfinite(e2c).any() else None,
                                                 "frames_seen": int(np.isfinite(e2c).sum())}
            for which in ("elbow", "wrist"):
                ce = coco_px(fit, changed, obs, judge, bias, which)
                r[f"coco_{which}_px_changed_frames"] = {"median": round(float(np.median(ce)), 2) if len(ce) else None, "n": int(len(ce)),
                                                        "pck20": round(float((ce < 20).mean()), 3) if len(ce) else None}
        if f != "input":
            au = hc.speed_audit(fl["input"], fit)
            r["speed_audit_vs_input"] = {"n_incidents": au["n_incidents"],
                                         "lead_right": au["lead_right"]["incidents"][:12]}
        # nothing but the leader's right arm may change against the installed motion
        if f == "v4":
            from .hand_kin import SHOULDER as HS, ELBOW as HE, WRIST as HW, FINGER_JOINTS
            arm = [HS["right"], HE["right"], HW["right"], *FINGER_JOINTS["right"]]
            other = [j for j in range(55) if j not in arm]
            r["max_change_other_joints_rad"] = {"lead": float(np.abs(P4[:, other] - P3[:, other]).max()),
                                                "follow": float(np.abs(hc.full_pose(fit["follow"]) - hc.full_pose(fl["v3"]["follow"])).max())}
        rep["fits"][f] = r
        log(f"validate {f} ({time.time() - t0:.0f} s)")
    rep["table"] = validate_table(rep)
    write_json(OUT / "validate.json", rep)
    return rep


def validate_table(rep: dict) -> list:
    """the before / after rows (plain): measure | input | v3 | v4"""
    rows = []
    fits = list(rep["fits"])

    def g(f, *path):
        x = rep["fits"][f]
        for p in path:
            if x is None:
                return None
            x = x.get(p) if isinstance(x, dict) else None
        return x
    for part in ("partner_forearm_L", "partner_forearm_R", "partner_upper_arm_L", "partner_upper_arm_R", "partner_hand_L", "partner_hand_R",
                 "partner_shoulder_L", "partner_shoulder_R", "partner_neck_head", "partner_torso", "self_forearm", "arm_vs_partner_torso"):
        what = ("his right forearm / upper arm vs her torso" if part == "arm_vs_partner_torso" else
                "his right hand vs " + part.replace("partner_", "her ").replace("self_forearm", "his own right forearm"))
        rows.append([f"capsules: {what} - frames > 0.5 cm / > 2 cm / depth p90 cm"]
                    + [f"{g(f, 'collisions', part, 'frames_gt_0.5cm')} / {g(f, 'collisions', part, 'frames_gt_2.0cm')} / "
                       f"{g(f, 'collisions', part, 'depth_cm_p90')}" for f in fits])
    rows.append(["wrist beyond its limits (bend 85, flex 80, ext 70, radial 25, ulnar 40) - seconds / frames > 10 deg"]
                + [f"{g(f, 'wrist', 'any', 'seconds_gt_0')} / {g(f, 'wrist', 'any', 'frames_gt_10deg')}" for f in fits])
    rows.append(["wrist bend beyond 85 deg (hand folded toward the forearm) - frames"] + [f"{g(f, 'wrist', 'bend', 'frames_gt_0')}" for f in fits])
    for nm in ("hand_any", "forearm_any"):
        rows.append([f"exact: his right {'hand' if nm == 'hand_any' else 'forearm'} inside her mesh - frames > 0.5 cm / > 2 cm / p90 cm"]
                    + [f"{g(f, 'exact', nm, 'frames_gt_0.5cm')} / {g(f, 'exact', nm, 'frames_gt_2.0cm')} / {g(f, 'exact', nm, 'depth_cm_p90')}"
                       for f in fits])
    for h in list(rep["fits"][fits[0]]["on_target"]):
        rows.append([f"on target: {h} ({g(fits[0], 'on_target', h, 'seconds')} s)"] + [f"{g(f, 'on_target', h, 'rate')}" for f in fits])
    rows.append(["MAMMA hand points, phones that see the hand, hold frames - median px"]
                + [f"{g(f, 'mamma_hand_px_hold_frames', 'median')}" for f in fits])
    rows.append(["MAMMA hand points, changed frames - median px"] + [f"{g(f, 'mamma_hand_px_changed_frames', 'median')}" for f in fits])
    rows.append(["COCO elbow (independent), changed frames - median px"] + [f"{g(f, 'coco_elbow_px_changed_frames', 'median')}" for f in fits])
    rows.append(["COCO wrist, changed frames - median px"] + [f"{g(f, 'coco_wrist_px_changed_frames', 'median')}" for f in fits])
    rows.append(["speed audit vs the input (added jumps > 0.55 m/s, whips)"] + [f"{g(f, 'speed_audit_vs_input', 'n_incidents')}" for f in fits])
    return [["measure"] + fits] + rows


def table_text(rows: list) -> str:
    w0 = max(len(r[0]) for r in rows)
    ws = [max(len(str(r[i])) for r in rows) for i in range(1, len(rows[0]))]
    return "\n".join(r[0].ljust(w0) + " | " + " | ".join(str(c).ljust(w) for c, w in zip(r[1:], ws)) for r in rows)


# ------------------------------------------------------------------------------------------------ before / after sheets

KEY_EVENT_TIMES = {118: (226.6, 232.0, 237.4), 47: (133.5, 140.2), 155: (325.8, 327.2)}


def before_after_sheets(n: int = 20, extra: list | None = None, fit_names=("input", "v3", "v4"), per_sheet: int = 5) -> list:
    """the n worst leader-right-hand frames (of the installed motion) + the key events (#118, #47, #155 at the frames the v3 review
    looked at): per frame 2 phones, each the raw crop and the meshes of input / v3 / v4 (the acting hand yellow) -> review/before_after_*.jpg"""
    import cv2
    rep = json.loads((OUT / "report_measure.json").read_text(encoding="utf-8"))
    wf = [dict(w) for w in rep["fits"]["v3"]["lead_R_worst"][:n]]
    for e in (extra or []):
        for t in KEY_EVENT_TIMES.get(int(e), ()):
            wf.append({"frame": int(round(t * HZ)), "t": t, "what": f"event #{e}", "depth_cm": None, "key_event": int(e)})
    fl = [(f, load_fit(f)) for f in fit_names]
    obs = hc.Obs()
    phones = hc.Phones()
    H = load_holds()
    ms = {f: load_measure(f) for f in fit_names if (OUT / f"measure_{f}.npz").exists()}
    paths = []
    tile = 236
    for si in range(0, len(wf), per_sheet):
        rows = []
        for w in wf[si:si + per_sheet]:
            k = int(w["frame"])
            pc = hand_site(posed(fl[-1][1]["lead"], np.array([k]))["J"], "right", "palm")[0][0]
            cams = review_cams(obs, posed(fl[0][1]["follow"], np.array([k]))["J"][0], k, pc, 2)
            tl = mesh_tiles(k, fl, obs, phones, cams, tile, point=pc)
            cells = [x for row in tl for x in row]
            head = np.full((tile, 250, 3), 245, np.uint8)
            lines = [f"t {k / HZ:.2f} s  film {film_time_of(k / HZ)}", str(w.get("group") or w.get("what"))[:44],
                     f"{w.get('what')}  {w.get('depth_cm') or ''} cm", f"hold {H['names'][k]}"]
            for f, m in ms.items():
                cl = min(m[f"lead_R_vs_{p}"][k] for p in ("forearm_L", "forearm_R", "upper_arm_L", "upper_arm_R", "hand_L", "hand_R"))
                lines.append(f"{f}: limbs {max(0, -cl) * 100:.1f} torso {max(0, -m['lead_R_vs_torso'][k]) * 100:.1f}")
                lines.append(f"   own fa {max(0, -m['lead_R_self_forearm'][k]) * 100:.1f} bend {m['lead_R_wrist_bend'][k]:.0f} deg")
            y = 16
            for ln in lines[:13]:
                cv2.putText(head, ln[:44], (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1, cv2.LINE_AA)
                y += 17
            rows.append(np.concatenate([head] + cells, 1))
        Wd = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 4), (0, Wd - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        top = np.full((30, Wd, 3), 255, np.uint8)
        cv2.putText(top, "leader's right hand: per phone the crop | input | v3 (installed) | v4 (zouk contacts) - lead blue, follow orange, "
                    "his right hand yellow", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        p = REVIEW / f"before_after_{si // per_sheet + 1}.jpg"
        cv2.imwrite(str(p), np.concatenate([top] + rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 86])
        paths.append(p)
        log(f"sheet -> {p}")
    return paths


# ------------------------------------------------------------------------------------------------ the side export (never installed)

SIDE_CAPTURE = "06_LarissaKaduRecap_v4contacts"
UNITY_MOTION = OUT / "unity_motion"


def export_motion() -> dict:
    """v4/smplx_contact through export_unity into the SIDE capture work/<take>/export/06_LarissaKaduRecap_v4contacts/ (install=False: the
    Unity project is not touched), its four motion files copied to v4/contacts/unity_motion/ with their sha256 (install_motion.sh checks
    them), and a read-only comparison with the installed capture"""
    import hashlib
    import shutil
    from . import export_unity as eu
    rep = eu.export(TAKE, SIDE_CAPTURE, install=False, room=False, textures=False, faces=False, shape_edits=True, allow_no_audio=True,
                    smplx_dir="v4/smplx_contact")
    side = WORK / "export" / SIDE_CAPTURE
    UNITY_MOTION.mkdir(parents=True, exist_ok=True)
    for f in hc.MOTION_FILES:
        shutil.copy2(side / f, UNITY_MOTION / f)
    out = {"side_export": str(side), "files": [str(UNITY_MOTION / f) for f in hc.MOTION_FILES],
           "fk_max_err_mm": {r: d["fk_max_err_mm"] for r, d in rep["dancers"].items()}, "warnings": rep["warnings"],
           "sha256": {f: hashlib.sha256((UNITY_MOTION / f).read_bytes()).hexdigest() for f in hc.MOTION_FILES},
           "fit": str(FIT_OUT), "exported": time.strftime("%Y-%m-%d %H:%M:%S"), "compare_installed": {}}
    inst = eu.UNITY_STREAMING / hc.UNITY_CAPTURE
    if not (inst / "smplx_lead.bin").exists():
        inst = WORK / "export" / hc.UNITY_CAPTURE
    out["compared_with"] = str(inst)
    f3, f4 = load_fit("v3"), load_fit("v4")
    from .hand_kin import SHOULDER as HS, ELBOW as HE, WRIST as HW, FINGER_JOINTS
    for r, f in (("lead", "smplx_lead.bin"), ("follow", "smplx_follow.bin")):
        h0, m0 = eu.read_motion(inst / f)
        h1, m1 = eu.read_motion(UNITY_MOTION / f)
        if m0.shape != m1.shape:
            out["compare_installed"][r] = f"shape differs {m0.shape} vs {m1.shape}"
            continue
        changed = np.flatnonzero(np.abs(hc.full_pose(f4[r]) - hc.full_pose(f3[r])).max((1, 2)) > 0)
        same = np.setdiff1d(np.arange(len(m0)), changed)
        pos0 = m0[:, 3 + 55 * 4:].reshape(len(m0), 55, 3)
        pos1 = m1[:, 3 + 55 * 4:].reshape(len(m1), 55, 3)
        d = np.linalg.norm(pos1 - pos0, axis=-1)
        arm = sorted({HS["right"], HE["right"], HW["right"], *FINGER_JOINTS["right"]})
        other = [j for j in range(55) if j not in arm]
        out["compare_installed"][r] = {"frames": int(len(m0)), "frames_changed": int(len(changed)),
                                       "max_abs_diff_unchanged_frames": float(np.abs(m1[same] - m0[same]).max()) if len(same) else None,
                                       "max_joint_move_cm_other_joints": float(d[:, other].max() * 100),
                                       "max_joint_move_cm_right_arm_hand": float(d[:, arm].max() * 100)}
    write_json(UNITY_MOTION / "unity_motion.json", out)
    _write_install_script()
    return out


def _write_install_script() -> Path:
    """v4/contacts/unity_motion/install_motion.sh: the v3 script with this folder as the source (checks the sha256 recorded by the export
    and the frame layout, backs the installed files up once, --check, --restore). For the Unity stage only; never run here."""
    src = (hc.V3 / "unity_motion" / "install_motion.sh").read_text(encoding="utf-8")
    s = src.replace('SRC="$WS/dancecap/work/larissa_kadu_recap/v3/unity_motion"', 'SRC="$WS/dancecap/work/larissa_kadu_recap/v4/contacts/unity_motion"')
    s = s.replace('WS="$(cd "$(dirname "$0")/../../../../.." && pwd)"', 'WS="$(cd "$(dirname "$0")/../../../../../.." && pwd)"')
    s = s.replace("# For the UNITY STAGE (not run by the hand_contact workflow): put the v3 hand-contact motion into the installed recap capture.",
                  "# For the UNITY STAGE (not run by the zouk_contacts workflow): put the v4 zouk-contact motion into the installed recap capture.")
    s = s.replace("dancecap/work/larissa_kadu_recap/v3/unity_motion/install_motion.sh", "dancecap/work/larissa_kadu_recap/v4/contacts/unity_motion/install_motion.sh")
    s = s.replace("(run: python -m dancecap.hand_contact unity-motion)", "(run: python -m dancecap.zouk_contacts unity-motion)")
    s = s.replace('echo "installed the v3 hand-contact motion (2026-10-21 safety pass) into $CAP ($FILES)"',
                  'echo "installed the v4 zouk-contact motion (dancecap.zouk_contacts) into $CAP ($FILES)"')
    s = s.replace("# put the backed-up (pre-v3) files back", "# put the backed-up (installed before v4) files back")
    s = s.replace("# Version: the 2026-10-21 safety pass (dancecap.hand_contact; decisions in v3/hand_contact/curation.json, numbers in\n"
                  "# v3/hand_contact/validate.json, the export and the sha256 of each file in v3/unity_motion/unity_motion.json).",
                  "# Version: the v4 zouk contacts (dancecap.zouk_contacts; decisions in v4/contacts/curation.json + the post-hoc gate\n"
                  "# v4/contacts/gate.json, numbers in v4/contacts/validate.json, the export and the sha256 of each file in\n"
                  "# v4/contacts/unity_motion/unity_motion.json).")
    s = s.replace("# never overwrite the first backup (that is the pre-v3 motion)",
                  "# never overwrite the first backup (that is the motion installed before v4: the v3 hand-contact motion)")
    p = UNITY_MOTION / "install_motion.sh"
    p.write_bytes(s.replace("\r\n", "\n").encode("utf-8"))
    return p


# ------------------------------------------------------------------------------------------------ review tiles with the hand outlined

def mesh_tiles(k: int, fits: list[tuple[str, dict]], obs, phones, cams: list[str], tile: int = 260, point: np.ndarray | None = None,
               span_m: float = 0.75) -> list[np.ndarray]:
    """per phone: the undistorted crop around `point` + per fit the two meshes over it (lead blue, follow orange, his right hand + forearm
    yellow) and the OUTLINE of his right hand drawn on top (yellow line: a hand hidden behind her still shows where it is)"""
    import cv2
    from .hand_kin import WRIST as HW, FINGER_JOINTS
    vj = hc.vertex_joint()
    posed_fits = []
    for name, fit in fits:
        b = {r: hc.Body(fit[r], "cpu") for r in ROLES}
        fw = {r: b[r].forward_np(hc.full_pose(fit[r])[k:k + 1], np.asarray(fit[r]["transl"], float)[k:k + 1]) for r in ROLES}
        posed_fits.append((name, fw, b["lead"].faces))
    faces = posed_fits[0][2]
    hand_v = np.isin(vj, [HW["right"], *FINGER_JOINTS["right"]])
    arm_v = hand_v | (vj == ELBOW["right"])
    hand_f = hand_v[faces].all(1)
    if point is None:
        point = hand_site(posed_fits[0][1]["lead"]["joints"], "right", "palm")[0][0]
    out = []
    for c in cams:
        z = obs.z[c]
        img, k1, terr = phones.frame(c, k)
        K, R, t = z["K"][k], z["R"][k], z["t"][k]
        uv, d = obs.project(c, point[None, None], np.array([k]))
        cx, cy = uv[0, 0]
        side = max(120.0, span_m * K[0, 0] / max(d[0, 0], 0.3))
        raw = hc.crop_img(img, cx, cy, side, tile, K, k1) if img is not None else np.full((tile, tile, 3), 40, np.uint8)
        Kc = hc.crop_K(K, cx, cy, side, tile)
        row = [hc.label_tile(raw, [f"cam {c} t {k / HZ:.2f}"])]
        for name, fw, fc in posed_fits:
            VL, VF = fw["lead"]["verts"][0].astype(float), fw["follow"]["verts"][0].astype(float)
            nv = len(VL)
            col_l = np.tile(np.asarray(hc.ROLE_RGB["lead"]), (len(fc), 1))
            col_l[arm_v[fc].all(1)] = (1.0, 0.85, 0.1)
            col_f = np.tile(np.asarray(hc.ROLE_RGB["follow"]), (len(fc), 1))
            rend, cov = hc.render_faces(np.concatenate([VL, VF]), np.concatenate([fc, fc + nv]), np.concatenate([col_l, col_f]), Kc, R, t,
                                        tile, tile)
            tl = hc.overlay_tile(raw, rend, cov, 0.55)
            _, hcov = hc.render_faces(VL, fc[hand_f], np.tile([1.0, 1, 0], (int(hand_f.sum()), 1)), Kc, R, t, tile, tile)
            cnts, _ = cv2.findContours(hcov.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(tl, cnts, -1, (0, 255, 255), 1, cv2.LINE_AA)
            row.append(hc.label_tile(tl, [name]))
        out.append(row)
    return out


def review_cams(obs, kf_J: np.ndarray, k: int, point: np.ndarray, n: int = 2) -> list[str]:
    """phones for a hand-on-her-back review: one that sees her BACK (her front away from it), then the widest other view"""
    cands = []
    fwd = torso_frame(kf_J[None])[0, :, 2]
    for c in obs.cams:
        z = obs.z[c]
        if not z["ok"][k]:
            continue
        uv, d = obs.project(c, point[None, None], np.array([k]))
        Wd, Hd = [int(x) for x in z["size"]]
        u, v = uv[0, 0]
        if d[0, 0] <= 0.2 or not (0.05 * Wd < u < 0.95 * Wd and 0.03 * Hd < v < 0.97 * Hd):
            continue
        C = -z["R"][k].T @ z["t"][k]
        tc = C - kf_J[SPINE3]
        tc[1] = 0
        fv = float(tc @ fwd / max(np.linalg.norm(tc), 1e-9))
        cands.append((fv, z["K"][k][0, 0] / d[0, 0], c))
    if not cands:
        return list(obs.cams)[:n]
    back = sorted(cands, key=lambda x: x[0])[0]
    rest = sorted([x for x in cands if x[2] != back[2]], key=lambda x: -x[1])
    return [back[2]] + [x[2] for x in rest[:n - 1]]


# ------------------------------------------------------------------------------------------------ curation (decisions taken by looking)

CURATION = OUT / "curation.json"


def load_curation() -> dict:
    """v4/contacts/curation.json: {"veto": {"<pass>:<window>": {"core_s": [a, b], "why": ...}}, "blend_s": s} - windows put back to the
    installed motion after looking at their phone crops (each with its reason); the v3 curation (hand_contact) is not applied here: the v4
    windows start from the installed motion, which already carries it"""
    if CURATION.exists():
        c = json.loads(CURATION.read_text(encoding="utf-8"))
    else:
        c = {}
    return {"veto": {str(k): v for k, v in (c.get("veto") or {}).items()}, "blend_s": float(c.get("blend_s", 0.7)),
            "notes": c.get("notes", {}), "despike": list(c.get("despike") or [])}


def _slerp_rotvec(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """per row: rotation a (n,3) -> b (n,3) by t (n,) along the shortest arc (axis-angle in, axis-angle out)"""
    from scipy.spatial.transform import Rotation as Ro
    qa, qb = Ro.from_rotvec(a).as_quat(), Ro.from_rotvec(b).as_quat()
    dot = (qa * qb).sum(1)
    qb = np.where(dot[:, None] < 0, -qb, qb)
    dot = np.abs(dot).clip(-1, 1)
    om = np.arccos(dot)
    so = np.sin(om)
    small = so < 1e-6
    wa = np.where(small, 1 - t, np.sin((1 - t) * om) / np.where(small, 1, so))
    wb = np.where(small, t, np.sin(t * om) / np.where(small, 1, so))
    q = wa[:, None] * qa + wb[:, None] * qb
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Ro.from_quat(q).as_rotvec()


def apply_curation(tag: str = "pass2", write: bool = True, base_tag: str | None = None) -> dict:
    """the adopted result (runs/<tag>/poses_merged.npz) with the vetoed windows put back to the installed motion: on each vetoed window's
    core the installed pose, blended in / out over blend_s by a raised cosine (slerp per joint of the leader's right arm and fingers, so the
    neighbouring windows keep their correction and nothing jumps) -> v4/smplx_contact/"""
    from .hand_kin import SHOULDER as HS, ELBOW as HE, WRIST as HW, FINGER_JOINTS
    cur = load_curation()
    base = load_fit("v3")                                   # the installed motion
    P3 = hc.full_pose(base["lead"])
    P = np.load(OUT / "runs" / tag / "poses_merged.npz")["lead"].copy()
    T = len(P)
    m = np.zeros(T)
    nb = int(round(cur["blend_s"] * HZ))
    for wi, v in cur["veto"].items():
        c0, c1 = (int(round(float(x) * HZ)) for x in v["core_s"])
        a0, a1 = max(0, c0 - nb), min(T - 1, c1 + nb)
        x = np.arange(a0, a1 + 1)
        r = np.ones(len(x))
        up = x < c0
        r[up] = 0.5 - 0.5 * np.cos(np.pi * (x[up] - a0 + 1) / (c0 - a0 + 1))
        dn = x > c1
        r[dn] = 0.5 + 0.5 * np.cos(np.pi * (x[dn] - c1) / (a1 - c1 + 1))
        m[x] = np.maximum(m[x], r)
    joints = [HS["right"], HE["right"], HW["right"], *FINGER_JOINTS["right"]]
    ks = np.flatnonzero(m > 0)
    for j in joints:
        P[ks, j] = _slerp_rotvec(P[ks, j], P3[ks, j], m[ks])
    full = m >= 1.0
    P[full] = P3[full]
    # despike (curation "despike": [[a_s, b_s, why], ...]): 2-4-frame glitches carried over from the input fit (the leader's right palm at
    # 2-3 m/s for 3 frames and back): the arm and finger rotations on [a, b] are slerped between the frames just outside it
    desp = []
    for d in cur.get("despike", []):
        a, b = int(round(float(d[0]) * HZ)), int(round(float(d[1]) * HZ))
        k0, k1 = a - 1, b + 1
        if k0 < 0 or k1 >= T or b < a:
            continue
        x = np.arange(a, b + 1)
        tt = (x - k0) / float(k1 - k0)
        for j in joints:
            P[x, j] = _slerp_rotvec(np.repeat(P[k0:k0 + 1, j], len(x), 0), np.repeat(P[k1:k1 + 1, j], len(x), 0), tt)
        desp.append([round(a / HZ, 3), round(b / HZ, 3)])
    rep = {"vetoed_windows": cur["veto"], "frames_put_back": int(full.sum()), "frames_blended": int(((m > 0) & ~full).sum()),
           "despiked": desp}
    if write:
        rep["written"] = write_fit_v4(base, {"lead": P, "follow": hc.full_pose(base["follow"])}, FIT_OUT)
    write_json(OUT / "runs" / tag / "curated.json", rep)
    log(f"curation: {rep['frames_put_back']} frames back to the installed motion, {rep['frames_blended']} blended")
    return rep


# ------------------------------------------------------------------------------------------------ the post-hoc gate (2026-10-22 review)
#
# The independent check of the v4 result found errors the window guards did not test: the wrist TWIST left free (candy-wrapper wrists), a
# backward-bent elbow, whips of the elbow and the fingertips (the speed audit covers only the wrist and the palm), a wrist that flicks in 0.3 s
# at a window edge, and windows where the hold prior moved the hand away from where the independent COCO wrist sees it. Until a re-solve has
# those terms, the gate measures each of them per frame on the curated result against the installed (v3) motion and the input fit, and vetoes
# every adopted window whose span (core + ramps) holds a NEW problem; curation is applied again until no flagged frame is left.

GATE = {"twist_deg": 90.0, "twist_new_deg": 10.0, "twist_bend_max_deg": 60.0, "hyper_deg": -5.0, "off_hinge_deg": 35.0, "off_hinge_new_deg": 10.0,
        "add_mps": 0.55, "add_new_mps": 0.15, "fast_margin_mps": 0.25, "wrist_rot_new_deg_frame": 8.0, "coco_worse_px": 3.0, "coco_min_n": 8}
GATE_TAGS = ("all2", "pass2", "pass3")


def _local_rot_speed(P: np.ndarray, j: int) -> np.ndarray:
    """(T-1,) degrees per frame the local rotation of joint j turns between consecutive frames"""
    from scipy.spatial.transform import Rotation as Ro
    R = Ro.from_rotvec(P[:, j])
    return np.degrees((R[:-1].inv() * R[1:]).magnitude())


def gate_flags(fits: dict | None = None) -> dict:
    """per frame of the leader's right arm: the problems v4 has that neither v3 (installed) nor the input has -> {name: bool (T,)}"""
    fits = fits or {f: load_fit(f) for f in ("input", "v3", "v4")}
    g = GATE
    P = {f: hc.full_pose(fits[f]["lead"]) for f in ("input", "v3", "v4")}
    T = len(P["v4"])
    changed = np.abs(P["v4"] - P["v3"]).max((1, 2)) > 0
    W = {f: wrist_angles(P[f], "right") for f in P}
    E = {f: elbow_angles(P[f], "right") for f in P}
    flags = {}
    tw = {f: np.abs(W[f]["twist"]) for f in P}
    flags["twist"] = (tw["v4"] > g["twist_deg"]) & (W["v4"]["bend"] < g["twist_bend_max_deg"]) & \
                     (tw["v4"] > np.maximum(tw["v3"], tw["input"]) + g["twist_new_deg"])
    flags["elbow_hyperextension"] = (E["v4"]["flexion"] < g["hyper_deg"]) & (E["v3"]["flexion"] >= g["hyper_deg"])
    off = {f: np.abs(E[f]["off_hinge"]) for f in P}
    flags["elbow_off_hinge"] = (off["v4"] > g["off_hinge_deg"]) & (off["v4"] > np.maximum(off["v3"], off["input"]) + g["off_hinge_new_deg"])
    # speeds: the elbow, the wrist, the palm centre and the five fingertips (FK of the stored poses)
    ks = np.flatnonzero(changed)
    pts = {}
    for f in P:
        o = posed(fits[f]["lead"], None, P[f])
        pts[f] = {"elbow": o["J"][:, ELBOW["right"]], "wrist": o["J"][:, WRIST["right"]], "palm": hc.palm_centre(o["J"], "right")}
        for fi in FINGERS:
            pts[f][f"tip_{fi}"] = o["tips"][:, tip_index("right", fi)]
    sp = np.zeros(T, bool)
    worst = {}
    for k in pts["v4"]:
        xin, x3, x4 = pts["input"][k], pts["v3"][k], pts["v4"][k]
        p99 = float(np.percentile(hc.speeds(xin), 99))
        a3, a4 = hc.speeds(x3 - xin), hc.speeds(x4 - xin)
        v_in, v3, v4 = hc.speeds(xin), hc.speeds(x3), hc.speeds(x4)
        bad = ((a4 > g["add_mps"]) & (a4 > a3 + g["add_new_mps"])) | (v4 > np.maximum.reduce([np.full_like(v4, p99), v_in, v3]) + g["fast_margin_mps"])
        b = np.zeros(T, bool)
        b[:-1] |= bad
        b[1:] |= bad
        sp |= b
        worst[k] = int(b.sum())
    flags["whip"] = sp
    w_in, w3, w4 = (_local_rot_speed(P[f], WRIST["right"]) for f in ("input", "v3", "v4"))
    rb = w4 > np.maximum(w3, w_in) + g["wrist_rot_new_deg_frame"]
    fr = np.zeros(T, bool)
    fr[:-1] |= rb
    fr[1:] |= rb
    flags["wrist_flick"] = fr
    for k in flags:
        flags[k] &= changed            # only frames v4 changed can carry a new problem
    return {"flags": flags, "changed": changed, "whip_frames_per_point": worst}


def adopted_windows(tags=None) -> list[dict]:
    """every adopted window of the passes: {"id": "<tag>:<i>", "core": [a, b] frames, "span": [a - ramp, b + ramp], "hold": ...}"""
    out = []
    for tag in (tags or GATE_TAGS):
        p = OUT / "runs" / tag / "refine.json"
        if not p.exists():
            continue
        for i, r in enumerate(json.loads(p.read_text(encoding="utf-8"))["results"]):
            if not r.get("adopted"):
                continue
            c0, c1 = int(r["core"][0]), int(r["core"][1])
            ramp = int(r.get("ramp", 12) or 12)
            out.append({"id": f"{tag}:{r.get('window', i)}", "core": [c0, c1], "span": [max(0, c0 - ramp), c1 + ramp], "hold": r.get("hold"),
                        "t": [round(c0 / HZ, 2), round(c1 / HZ, 2)],
                        # a vetoed window goes back to the installed motion over its whole span (core + its ramps): a veto of the core alone
                        # kept a blend of the rejected solve in the ramps
                        "t_span": [round(max(0, c0 - ramp) / HZ, 2), round((c1 + ramp) / HZ, 2)]})
    return out


def coco_window_check(fits: dict, wins: list[dict]) -> dict:
    """per window: the independent COCO wrist (median px over the core, every phone with a detection) of v3 and v4"""
    obs = hc.Obs()
    judge = hc.CocoJudge()
    bias = hc.calibrate_coco_bias(fits["input"], obs, judge)
    out = {}
    for w in wins:
        ks = np.arange(w["core"][0], w["core"][1] + 1)
        e3, e4 = coco_px(fits["v3"], ks, obs, judge, bias, "wrist"), coco_px(fits["v4"], ks, obs, judge, bias, "wrist")
        out[w["id"]] = {"n": int(min(len(e3), len(e4))), "v3_px": round(float(np.median(e3)), 2) if len(e3) else None,
                        "v4_px": round(float(np.median(e4)), 2) if len(e4) else None}
    return out


def run_gate(tag: str = "pass3", rounds: int = 4, coco: bool = True, extra_veto: dict | None = None, tags=None) -> dict:
    """apply the curation, flag NEW problems frame by frame, veto every adopted window whose span holds one (and, with coco, every window whose
    independent COCO wrist got more than GATE["coco_worse_px"] worse than v3), apply again; repeat until nothing is flagged -> gate.json,
    curation.json (the gate's vetoes marked "gate"), v4/smplx_contact/"""
    cur_raw = json.loads(CURATION.read_text(encoding="utf-8")) if CURATION.exists() else {}
    cur_raw.setdefault("veto", {})
    for k, v in (extra_veto or {}).items():
        cur_raw["veto"][k] = v
    CURATION.write_text(json.dumps(cur_raw, indent=1, ensure_ascii=False), encoding="utf-8")
    wins = adopted_windows(tags)
    hist = []
    coco_rep = None
    for rnd in range(rounds):
        apply_curation(tag)
        fits = {f: load_fit(f) for f in ("input", "v3", "v4")}
        gf = gate_flags(fits)
        fl = gf["flags"]
        anyf = np.zeros(len(gf["changed"]), bool)
        for v in fl.values():
            anyf |= v
        vetoed = {k for k in cur_raw["veto"]}
        new = {}
        for w in wins:
            if w["id"] in vetoed:
                continue
            a, b = w["span"]
            hit = {k: int(v[a:b + 1].sum()) for k, v in fl.items() if v[a:b + 1].any()}
            if hit:
                new[w["id"]] = {"core_s": w["t_span"], "why": f"gate round {rnd + 1}: new problems in its span: {hit}", "gate": True}
        if coco and rnd == 0:
            live = [w for w in wins if w["id"] not in vetoed and w["id"] not in new]
            coco_rep = coco_window_check(fits, live)
            for w in live:
                c = coco_rep[w["id"]]
                if c["n"] >= GATE["coco_min_n"] and c["v3_px"] is not None and c["v4_px"] - c["v3_px"] > GATE["coco_worse_px"]:
                    new[w["id"]] = {"core_s": w["t_span"], "why": f"gate: the independent COCO wrist {c['v3_px']} -> {c['v4_px']} px (n {c['n']})", "gate": True}
        hist.append({"round": rnd + 1, "flagged_frames": {k: int(v.sum()) for k, v in fl.items()}, "flagged_any": int(anyf.sum()),
                     "flagged_times_s": [round(x / HZ, 2) for x in np.flatnonzero(anyf)[:60]], "new_vetoes": sorted(new),
                     "frames_changed_vs_v3": int(gf["changed"].sum())})
        log(f"gate round {rnd + 1}: flagged {hist[-1]['flagged_frames']}, new vetoes {sorted(new)}")
        if not new:
            break
        cur_raw["veto"].update(new)
        CURATION.write_text(json.dumps(cur_raw, indent=1, ensure_ascii=False), encoding="utf-8")
    rep = {"what": "zouk_contacts post-hoc gate: NEW problems of v4 vs the installed v3 and the input (twist, elbow hyperextension / off-hinge, "
                   "elbow / wrist / palm / fingertip whips, wrist flicks) and the independent COCO wrist per window", "gate": GATE,
           "windows_adopted": len(wins), "windows_vetoed": sorted(k for k in cur_raw["veto"]), "rounds": hist, "coco_per_window": coco_rep,
           "windows_kept": [w for w in wins if w["id"] not in cur_raw["veto"]]}
    write_json(OUT / "gate.json", rep)
    return rep


# ------------------------------------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.zouk_contacts", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capsules")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("sites")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("measure")
    p.add_argument("--fit", default="input,v3", help="comma list of input / v3 / v4")
    p = sub.add_parser("holds")
    p.add_argument("--fit", default="input")
    sub.add_parser("report")
    p = sub.add_parser("sheet")
    p.add_argument("--worst", type=int, default=20)
    p.add_argument("--fits", default="input,v3")
    p = sub.add_parser("look")
    p.add_argument("--t", required=True, help="comma list of seconds")
    p.add_argument("--fits", default="input,v3")
    p.add_argument("--cams", default=None)
    p.add_argument("--out", default=None)
    p = sub.add_parser("exact")
    p.add_argument("--fit", default="input,v3")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("plan")
    p.add_argument("--base", default="v3")
    p.add_argument("--out", default="plan.json")
    p.add_argument("--problems", default=None, help="comma list: plan only these kinds (e.g. wrist,self_forearm)")
    for nm in ("refine", "refine-part"):
        p = sub.add_parser(nm)
        if nm == "refine-part":
            p.add_argument("i", type=int)
            p.add_argument("n", type=int)
        else:
            p.add_argument("--parts", type=int, default=4)
            p.add_argument("--write", action="store_true")
        p.add_argument("--tag", default="all")
        p.add_argument("--only", default=None)
        p.add_argument("--cfg", default=None, help="json of ZoukCfg fields")
        p.add_argument("--device", default="cuda")
    p = sub.add_parser("merge")
    p.add_argument("n", type=int)
    p.add_argument("--tag", default="all")
    p.add_argument("--write", action="store_true")
    p.add_argument("--cfg", default=None)
    sub.add_parser("validate")
    p = sub.add_parser("curate")
    p.add_argument("--tag", default="all")
    p = sub.add_parser("gate")
    p.add_argument("--tag", default="pass3")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--no-coco", action="store_true")
    p.add_argument("--tags", default=None, help="comma list of the passes whose adopted windows the gate tests (default all2,pass2,pass3)")
    p = sub.add_parser("before-after")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--extra", default="118,47,155")
    sub.add_parser("unity-motion")
    a = ap.parse_args(argv)
    if a.cmd == "capsules":
        b = capsule_bodies(force=a.force)
        print(json.dumps(py_json({r: v["fit"] for r, v in b.items()}), indent=1)[:4000])
    elif a.cmd == "sites":
        print(json.dumps(py_json(site_bodies(force=a.force)), indent=1)[:4000])
    elif a.cmd == "measure":
        for f in a.fit.split(","):
            run_measure(f)
    elif a.cmd == "holds":
        print(json.dumps(py_json(detect_holds(a.fit)["seconds_per_hold"]), indent=1))
    elif a.cmd == "report":
        r = measure_report()
        print(json.dumps(py_json({f: v["lead_R"] for f, v in r["fits"].items()}), indent=1)[:6000])
    elif a.cmd == "sheet":
        worst_sheet(tuple(a.fits.split(",")), a.worst)
    elif a.cmd == "look":
        print(look([float(x) for x in a.t.split(",")], tuple(a.fits.split(",")), Path(a.out) if a.out else None,
                   a.cams.split(",") if a.cams else None))
    elif a.cmd == "exact":
        for f in a.fit.split(","):
            exact_measure(f, a.device)
    elif a.cmd == "plan":
        plan_windows(ZoukCfg(base=a.base, plan=a.out, problems=tuple(a.problems.split(",")) if a.problems else ()))
    elif a.cmd in ("refine", "refine-part"):
        cfg = ZoukCfg(**json.loads(a.cfg)) if a.cfg else ZoukCfg()
        only = [int(x) for x in a.only.split(",")] if a.only else None
        if a.cmd == "refine-part":
            run_refine(part=(a.i, a.n), device=a.device, only=only, cfg=cfg, tag=a.tag)
        elif a.parts > 1:
            run_parallel(a.parts, a.tag, only, json.loads(a.cfg) if a.cfg else None, a.write)
        else:
            run_refine(None, a.device, only, cfg, a.tag)
    elif a.cmd == "merge":
        merge_parts(a.n, a.tag, write=a.write, cfg_over=json.loads(a.cfg) if a.cfg else None)
    elif a.cmd == "curate":
        print(json.dumps(py_json(apply_curation(a.tag)), indent=1)[:3000])
    elif a.cmd == "gate":
        r = run_gate(a.tag, a.rounds, not a.no_coco, tags=a.tags.split(",") if a.tags else None)
        print(json.dumps(py_json({"vetoed": r["windows_vetoed"], "rounds": r["rounds"]}), indent=1)[:6000])
    elif a.cmd == "validate":
        v = validate()
        print(json.dumps(py_json(v["table"]), indent=1)[:8000])
    elif a.cmd == "before-after":
        before_after_sheets(a.n, [int(x) for x in a.extra.split(",")] if a.extra else [])
    elif a.cmd == "unity-motion":
        print(json.dumps(py_json(export_motion()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
