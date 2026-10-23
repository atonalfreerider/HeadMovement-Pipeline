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
(plan_pass3.json, wrist_only, runs/pass3); every window adopted only when its guards pass. After the independent check
(review_check-hands.json): the re-solve r1-r3 with twist / elbow limits and the post-hoc gate (gate.json) -> installed (smplx_lead.bin
5ccf839e..., kept as v4/smplx_contact_installed + runs/inst).

The final pass (follow the phones over the hold prior; jobs/zc_final.json):
    python -m dancecap.zouk_contacts plan --base inst --out plan_r4.json --problems exact --follow-phones --max-core-s 5 --merge-s 0.5
    python -m dancecap.zouk_contacts refine --parts 5 --tag r4 --cfg '{"base": "inst", "plan": "plan_r4.json", "follow_phones": true, ...}'  (pod)
    python -m dancecap.zouk_contacts gate --tag r4 --tags r4 --base-tag inst --exact    # curation.json["stages"]["r4"], gate_r4.json
    python -m dancecap.zouk_contacts measure --fit v4 && python -m dancecap.zouk_contacts validate-final   # -> validate_final.md

The closing pass (after the check of the final pass, final_check/; jobs/zc_close.json; see the section "the closing pass"):
    python -m dancecap.zouk_contacts make-base --src inst2 --out fin2        # the installed d034e0b7 motion + Hermite despikes
    python -m dancecap.zouk_contacts exact --fit fin2 --changed-vs inst2 && python -m dancecap.zouk_contacts measure --fit fin2
    python -m dancecap.zouk_contacts plan --base fin2 --out plan_r6.json --problems exact,fold,off_phone,float --follow-phones --max-core-s 5 --merge-s 0.5
    python -m dancecap.zouk_contacts refine --parts 5 --tag r6 --cfg '{"base": "fin2", "plan": "plan_r6.json", "follow_phones": true,
        "ray_mode": "hit", "unfold": true, "wrist_sigma": 2.0, "vis_guards": true, "gate_guards": true, ...}'          (pod)
    python -m dancecap.zouk_contacts gate --tag r6 --tags r6 --base-tag fin2 --exact --vis    # curation.json["stages"]["r6"], gate_r6.json
    python -m dancecap.zouk_contacts validate-final --fits input,v3,inst2,v4

The last pass (before the one render; see the section "the last pass"; runs locally, her surface queries on the GPU):
    python -m dancecap.zouk_contacts last-pass                    # the installed ec8e7101 -> v4/smplx_contact, exact_v4.npz, last/report.json
    python -m dancecap.zouk_contacts measure --fit v4 && python -m dancecap.zouk_contacts validate-final --fits input,v3,ec8e,v4
    python -m dancecap.zouk_contacts unity-motion                 # then install_motion.sh (--check first)

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
FIT_INST = V4 / "smplx_contact_installed"           # the v4 motion installed 2026-10-22 (smplx_lead.bin 5ccf839e...): the final pass's base
FIT_FIN1 = V4 / "smplx_contact_fin1"                 # the final pass r4 (curated + gated): the base of its second round r5
FITS = {"input": hc.SRC_FIT, "v3": hc.FIT_OUT, "v4": FIT_OUT, "inst": FIT_INST, "fin1": FIT_FIN1}

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
    freeze_arm_unfold: bool = False      # unfold_only windows: the shoulder and elbow frozen (hc.Actor.freeze_arm), only the wrist turns
    keep_joint: str = "palm"             # the wrist pass keeps this where it is: "palm" (pass 3) or "wrist" (the closing pass's unfold_only windows)
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
    # the final pass (2026-10-22, after the independent check: "follow the phones over the hold prior"): no hold sites / region; where the
    # palm is inside her, its target is the nearest point of her torso surface on the side the window's ray phone sees (the phone that sees
    # the hand and agrees with the motion), a move across that phone's ray costing more than one along it (visible_side_targets); elsewhere
    # the palm stays where it is; the fingers are kept, no forearm-twist start
    follow_phones: bool = False
    ray_alpha: float = 2.0               # the target: a move across the ray phone's ray costs (1 + ray_alpha) x a move along it
    ray_region_scale: float = 10.0       # w_region x this in the final pass (the target leads)
    ray_slack_m: float = 0.02            # stage 2: the palm within this of its target (pushed frames: on her surface; others: where it is)
    ray_sigma_frames: float = 3.0        # the push (target - palm) smoothed over time
    ray_rate_mps: float = 0.0            # > 0 (the closing pass): ... and its change limited to this speed (both ways in time: it rises before)
    exact_pen_cm: float = 1.0            # plan problem "exact": the motion's hand inside her torso / shoulders (exact_<base>.npz) deeper than this
    despike: tuple = ()                  # [[a_s, b_s], ...]: input glitches; his right arm + fingers slerped across them in the base before solving
    # the closing pass (2026-10-22, after the check of the final pass; see the section "the closing pass")
    ray_mode: str = "side"               # "side": visible_side_targets; "hit": the first hit of the ray phone's ray on her torso (ray_hit_targets)
    max_slide_m: float = 0.25            # "hit": the farthest the palm may slide along the ray out of her (or toward the observed hand)
    float_pull_m: float = 0.06           # "hit": a palm this close in front of her back (back holds) along the ray is laid onto it
    float_from_m: float = 0.035          # ... when its centre is further than this from her surface (the palm's half thickness + 2 cm)
    spike_interp: bool = False           # round r8 spike windows: the palm interpolated in her frame (spike_targets) instead of the ray hit
    unfold: bool = False                 # folded wrists start unfolded (unfold_wrist) and the 2D places the hand
    unfold_knee_deg: float = 80.0        # ... bends above this compressed (slope 0.1)
    wrist_sigma: float = 0.0             # ... and the start's wrist rotation low-passed over this many frames (smooth_wrist; with unfold)
    vis_guards: bool = False             # the COCO wrist / elbow guards count only the phones that see the candidate's joint (joint_visible)
    fold_plan_deg: float = 120.0         # plan problem "fold": the wrist bend past this
    off_phone_m: float = 0.08            # plan problem "off_phone": the back-view phone shows the hand further than this (image plane)
    float_plan_cm: float = 2.0           # plan problem "float": back holds with the hand further than this off her (exact gap)
    w_bend: float = 300.0                # per rad^2 beyond bend_soft_deg (the whole wrist bend) ...
    bend_soft_deg: float = 78.0
    w_comp: float = 150.0                # ... and per rad^2 of flexion / extension / radial / ulnar beyond their limits - 3 deg
    vis_min_lm: int = 6
    w_2d: float = 3.0                    # MAMMA's hand points in the usable phones (both stages), robust with sigma_2d_px
    sigma_2d_px: float = 14.0
    w_elbow: float = 8.0                 # COCO elbow (triangulated, bias-corrected, smoothed), robust: the elbow is seen, the hand is not
    w_elbow2d: float = 3.0               # ... and per phone: every confident COCO elbow detection (single views too), robust sigma_elbow2d_px
    sigma_elbow2d_px: float = 15.0
    w_wrist2d: float = 0.0               # the closing pass: the COCO wrist per phone that sees it (visibility + swap filtered), robust
    site_seen_scale: float = 0.15        # the site attractor where a usable phone sees the hand (the 2D leads there)
    w_fingers_up: float = 40.0           # back holds: fingers pointing up her back more than 30 deg above her horizontal cost this (zouk: the
                                         #   leader's hand on her back lies across it, the fingers toward her spine / round her far side)
    w_speed: float = 2000.0              # the added motion beyond 0.3 m/s (hand_contact's term, 600 there: the moves here are larger)
    w_speed_tips: float = 0.0            # the same for the elbow and the five fingertips (the final pass: the gate's whip test)
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
    guard_coco_rel: float = 0.0          # the closing pass: a COCO guard allows max(slack, this x the motion's median px before) - where the
                                         #   motion already misses the detector by 25-40 px (his hand behind her: the detector guesses) a 3-4 px
                                         #   change is noise
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


# ------------------------------------------------------------------------------------------------ the final pass: follow the phones
#
# The independent check (review_check-hands.json) found that the hold prior (the zouk sites) moved the hand to where the hold says, not where
# the phones show it (87.8, 133.5, 177.5, 180.5 s), and that the clearest closed-hold frames (106.6 s: cam 06 shows the hand flat on her back)
# stayed ~10 cm inside her. The motion installed then (v3 there) has the hand where the phones show it in 2D, only too deep. The final pass
# keeps that 2D as far as it can: per window ONE phone (the ray phone: MAMMA sees the hand there and agrees with the motion), and on each
# frame where the palm centre is inside her, the target is the nearest point of her torso surface whose outward side faces that phone (a
# move across its ray costs 3 x a move along it), lifted by the hand's half thickness: the hand comes out of her onto her back (or front)
# close to where that phone shows it. The hold detector's sites are not used.

TORSO_JOINTS = (PELVIS, SPINE1, SPINE2, SPINE3, NECK, COLLAR["left"], COLLAR["right"])


def exact_torso_pen(name: str) -> np.ndarray:
    """(T,) deepest point of his right hand inside her torso / shoulders (m, exact_<name>.npz)"""
    with np.load(OUT / f"exact_{name}.npz") as z:
        parts = [str(x) for x in z["parts"]]
        hp = z["hand_pen_part"]
        ks = z["ks"]
        T = int(ks.max()) + 1
        out = np.zeros(max(T, len(ks)))
        out[ks] = hp[:, [parts.index(p) for p in ("torso", "shoulder_L", "shoulder_R")]].max(1)
    return out


RAY_AGREE_PX = 35.0


def cam_agreement(fit: dict, ks: np.ndarray, obs, hcfg: HoldCfg, cams=None) -> dict:
    """per phone: (MAMMA sees his right hand (n,), the median px of the motion's hand points against MAMMA's visible ones (n,), NaN unseen)"""
    import warnings
    hp = hc.hand_points_np(fit["lead"], hc.full_pose(fit["lead"]), ks, "right")
    hid = hc.landmark_parts()["right_hand"]
    out = {}
    for c in (cams or obs.cams):
        z = obs.z[c]
        see, _ = hand_2d(obs, c, ks, hcfg)
        uv, dep = obs.project(c, hp["lm"], ks)
        o = z["lm"][ks, hc.PIDX["lead"]][:, hid]
        vv = (z["vis"][ks, hc.PIDX["lead"]][:, hid] > hcfg.vis) & (np.abs(o).sum(-1) > 0) & (dep > 0.1)
        e = np.where(vv, np.linalg.norm(uv - o, axis=-1), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(e, 1)
        out[c] = (see & np.isfinite(med), med)
    return out


def choose_ray_cam(fit: dict, ks: np.ndarray, obs, hcfg: HoldCfg, agree_px: float = RAY_AGREE_PX) -> dict:
    """the window's ray phone: most frames where MAMMA sees his right hand AND the motion's hand points agree with it (median < agree_px),
    + a quarter of the frames it merely sees it; no phone at all -> the phone most behind her (her back toward it)"""
    score = {}
    for c, (ok, med) in cam_agreement(fit, ks, obs, hcfg).items():
        score[c] = {"seen": int(ok.sum()), "agree": int((ok & (med < agree_px)).sum()),
                    "median_px": round(float(np.median(med[ok])), 1) if ok.any() else None}
    best = max(obs.cams, key=lambda c: (score[c]["agree"] + 0.25 * score[c]["seen"], -(score[c]["median_px"] or 1e9)))
    basis = "phones"
    if score[best]["agree"] + score[best]["seen"] == 0:
        kf = posed(fit["follow"], ks[len(ks) // 2:len(ks) // 2 + 1])
        fwd = torso_frame(kf["J"])[0, :, 2]
        back = []
        k = int(ks[len(ks) // 2])
        for c in obs.cams:
            z = obs.z[c]
            if not z["ok"][k]:
                continue
            C = -z["R"][k].T @ z["t"][k] - kf["J"][0, SPINE3]
            C[1] = 0
            back.append((float(C @ fwd / max(np.linalg.norm(C), 1e-9)), c))
        best = min(back)[1] if back else best
        basis = "her back (no phone sees the hand)"
    return {"ray_cam": best, "ray_cam_basis": basis, "ray_cam_scores": score}


def visible_side_targets(surf, V, pc0: np.ndarray, C: np.ndarray, vmask: np.ndarray, lift_m: float, alpha: float = 2.0,
                         min_facing: float = 0.3, ok: np.ndarray | None = None, bs: int = 16) -> dict:
    """per frame where the palm centre pc0 (W,3) is INSIDE her (winding numbers, her posed mesh V (W,nv,3) torch): the point of her torso
    surface (vertices vmask) on the side the ray phone (camera centre C (W,3)) sees - its outward normal toward the phone - nearest to the
    palm, where a move across that phone's ray (which changes the hand's 2D there) costs (1 + alpha) x a move along it; the target = that
    point lifted lift_m along its normal (the palm lies on her surface, close to where that phone shows it); elsewhere the palm itself.
    2026-10-22: the first version slid the palm along the ray to where it enters her; from a phone at her side that is 30-40 cm away and
    the arm could not follow (106.07-107.43 s)."""
    torch = surf.torch
    dev = surf.dev
    W = len(pc0)
    P = torch.as_tensor(np.asarray(pc0, np.float32)[:, None], device=dev)
    q = hc.query_batched(surf, P, V, bs=8, wn_within=0.03)
    sd = q["sd"][:, 0].cpu().numpy().astype(float)
    tgt = np.asarray(pc0, float).copy()
    nrm = np.zeros((W, 3))
    cost = np.full(W, np.inf)
    vm = torch.as_tensor(np.asarray(vmask, bool), device=dev)
    Ct = torch.as_tensor(np.asarray(C, np.float32), device=dev)
    pct = torch.as_tensor(np.asarray(pc0, np.float32), device=dev)
    with torch.no_grad():
        for a in range(0, W, bs):
            Vb = V[a:a + bs].float()
            Nb = surf.vertex_normals(Vb)
            toC = Ct[a:a + bs, None] - Vb
            facing = (Nb * toC).sum(-1) / toC.norm(dim=-1).clamp_min(1e-6) > min_facing
            u = pct[a:a + bs] - Ct[a:a + bs]
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            d = Vb - pct[a:a + bs, None]
            along = (d * u[:, None]).sum(-1)
            perp = (d - along[..., None] * u[:, None]).norm(dim=-1)
            c = torch.sqrt(along ** 2 + ((1 + alpha) * perp) ** 2)
            c = torch.where(facing & vm[None], c, torch.full_like(c, float("inf")))
            cm, i = c.min(1)
            ar = torch.arange(len(Vb), device=dev)
            X, n = Vb[ar, i], Nb[ar, i]
            tgt[a:a + bs] = (X + lift_m * n).cpu().numpy()
            nrm[a:a + bs] = n.cpu().numpy()
            cost[a:a + bs] = cm.cpu().numpy()
    push = (sd < 0) & np.isfinite(cost) & (np.ones(W, bool) if ok is None else np.asarray(ok, bool))
    tgt = np.where(push[:, None], tgt, pc0)
    nrm = np.where(push[:, None], nrm, 0.0)
    return {"target": tgt, "normal": nrm, "pushed": push, "palm_sd_m": sd, "push_m": np.linalg.norm(tgt - pc0, axis=1)}


def despike_poses(P: np.ndarray, ranges, joints=None) -> tuple[np.ndarray, list]:
    """input glitches (2-4 frames where his right palm runs at 2-3 m/s and back): his right arm + finger rotations on [a, b] (seconds)
    slerped between the frames just outside -> (a copy of P, the ranges applied as [a_s, b_s])"""
    from .hand_kin import SHOULDER as HS, ELBOW as HE, WRIST as HW, FINGER_JOINTS
    joints = joints or [HS["right"], HE["right"], HW["right"], *FINGER_JOINTS["right"]]
    P = P.copy()
    T = len(P)
    done = []
    for d in ranges or ():
        a, b = int(round(float(d[0]) * HZ)), int(round(float(d[1]) * HZ))
        k0, k1 = a - 1, b + 1
        if k0 < 0 or k1 >= T or b < a:
            continue
        x = np.arange(a, b + 1)
        tt = (x - k0) / float(k1 - k0)
        for j in joints:
            P[x, j] = _slerp_rotvec(np.repeat(P[k0:k0 + 1, j], len(x), 0), np.repeat(P[k1:k1 + 1, j], len(x), 0), tt)
        done.append([round(a / HZ, 3), round(b / HZ, 3)])
    return P, done


# ------------------------------------------------------------------------------------------------ the closing pass (second independent check)
#
# The check of the final pass (final_check/, 2026-10-22) found: his hand still ~10 cm inside her on the clearest closed holds (74.5, 106.6 s:
# cam 06 shows the hand flat on her back and the mesh hand sits on the same pixels - only its depth along cam 06's ray is wrong; cam 01, her
# left profile, puts the mesh's back edge on her real back edge, so her body is not too deep), the wrist folded back 150-179 deg onto his
# forearm (12.3 s of the film, untouched), the hand 11-25 cm from where the back-view phone shows it (153.8, 244.0, 310.7 s), and two
# despikes that stop dead (165.73 s: the burst is in his pelvis / spine too) or interpolate straight through her (347.43-347.63 s).
# The earlier passes' 2D guards vetoed the depth fixes in phones that cannot see his right hand: at 106.6 s the COCO "right wrist" of cam 03 is
# the clasped hand-hold and cam 02's MAMMA hand is at her shoulder - his right hand is behind her back there. The closing pass:
#   - the target: the FIRST HIT of the ray phone's ray through the palm on her torso (a pure slide along that ray: the hand stays on its
#     pixels in that phone) - out of her where it is inside, onto her where it floats a few cm off her back, and along the ray through MAMMA's
#     observed hand where the back-view phone shows the hand elsewhere (off_phone);
#   - folded wrists start unfolded (the wrist's rotation compressed above 75 deg) and the 2D of the ray phone places the hand;
#   - the despikes are cubic Hermite curves matching the motion's velocity at both ends, his whole body at 165.73-166.00 s;
#   - every COCO / MAMMA guard (window and post-hoc gate) counts only the phones from which the candidate's joint is not hidden behind her
#     body or his own (capsule rays): a detector cannot vouch for a hand it cannot see.

FIT_INST2 = V4 / "smplx_contact_installed_d034"      # the final pass installed 2026-10-22 (smplx_lead.bin d034e0b7...): the closing pass's base
FIT_FIN2 = V4 / "smplx_contact_fin2"                 # ... with the Hermite despikes (the closing pass r6 starts from it)
FIT_FIN3 = V4 / "smplx_contact_fin3"                 # ... round r6 curated + gated (round r7 starts from it)
FIT_FIN4 = V4 / "smplx_contact_fin4"                 # ... r6 + r7 gated together against fin2 (stage r67; round r8 starts from it)
FIT_FIN5 = V4 / "smplx_contact_fin5"                 # ... + round r8 (the final assembly; round r9, the folds, starts from it)
FITS.update({"inst2": FIT_INST2, "fin2": FIT_FIN2, "fin3": FIT_FIN3, "fin4": FIT_FIN4, "fin5": FIT_FIN5})
BACK_HOLDS = ("upper_back", "wrap_back", "behind_back")


def despike_hermite(P: np.ndarray, tr: np.ndarray, ranges, h: int = 3) -> tuple[np.ndarray, np.ndarray, list]:
    """input glitches: on each range [a_s, b_s, kind] the rotations (kind "arm": his right shoulder / elbow / wrist + fingers; "body": every
    joint and the root translation) become a cubic Hermite curve between the frames just outside it whose end velocities are the motion's own
    (h-frame differences outside the range): no dead stop at the ends (2026-10-22 check: the slerp stopped the palm at 165.73 s); "body"
    keeps the legs' global rotations (the feet follow the hips only, <= 2 cm, instead of a reshaped step); kind
    "arm0": the arm with zero end velocities (where the wrist outside the range itself flails, its velocity makes the curve overshoot).
    Rotations are interpolated in the tangent space of the first outside frame. -> (P copy, transl copy, the ranges applied)"""
    from scipy.spatial.transform import Rotation as Ro
    from .hand_kin import SHOULDER as HS, ELBOW as HE, WRIST as HW, FINGER_JOINTS
    arm = [HS["right"], HE["right"], HW["right"], *FINGER_JOINTS["right"]]
    P, tr = P.copy(), np.asarray(tr, float).copy()
    P_orig = P.copy()
    T = len(P)
    done = []
    for d in ranges or ():
        a, b = int(round(float(d[0]) * HZ)), int(round(float(d[1]) * HZ))
        kind = str(d[2]) if len(d) > 2 and str(d[2]) in ("arm", "body", "arm0") else "arm"
        vel = 0.0 if kind == "arm0" else 1.0             # "arm0": zero end velocities (a smoothstep) where the motion outside flails
        k0, k1 = a - 1, b + 1
        if k0 - h < 0 or k1 + h >= T or b < a:
            continue
        n = float(k1 - k0)
        s = (np.arange(a, b + 1) - k0) / n
        h10, h01, h11 = s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        joints = list(range(P.shape[1])) if kind == "body" else arm
        for j in joints:
            R0 = Ro.from_rotvec(P[k0, j])
            loc = lambda k: (R0.inv() * Ro.from_rotvec(P[k, j])).as_rotvec()  # noqa: E731
            r1 = loc(k1)
            v0 = -loc(k0 - h) / h * n * vel                             # loc(k0) = 0
            v1 = (loc(k1 + h) - r1) / h * n * vel
            r = h10[:, None] * v0 + h01[:, None] * r1 + h11[:, None] * v1
            P[a:b + 1, j] = (R0 * Ro.from_rotvec(r)).as_rotvec()
        if kind == "body":
            # the legs keep their ORIGINAL global rotations (a cubic through a step's leg rotations moved the stepping foot 7 cm, the
            # 2026-10-22 export check): the hip's local rotation re-expressed under the despiked pelvis, knee / ankle / foot unchanged
            x = np.arange(a, b + 1)
            Rp_new, Rp_old = Ro.from_rotvec(P[x, PELVIS]), Ro.from_rotvec(P_orig[x, PELVIS])
            for s_ in SIDES:
                hj = HIP[s_]
                P[x, hj] = (Rp_new.inv() * Rp_old * Ro.from_rotvec(P_orig[x, hj])).as_rotvec()
                for jj in (KNEE[s_], ANKLE[s_], FOOT[s_]):
                    P[x, jj] = P_orig[x, jj]
            p0, p1 = tr[k0], tr[k1]
            v0, v1 = (tr[k0] - tr[k0 - h]) / h * n, (tr[k1 + h] - tr[k1]) / h * n
            tr[a:b + 1] = h00[:, None] * p0 + h10[:, None] * v0 + h01[:, None] * p1 + h11[:, None] * v1
        done.append([round(a / HZ, 3), round(b / HZ, 3), kind])
    return P, tr, done


def make_base(src: str = "inst2", out: str = "fin2", ranges=None) -> dict:
    """the closing pass's base: the fit `src` with the Hermite despikes -> FITS[out] (+ runs/<out>/poses_merged.npz for the curation)"""
    base = load_fit(src)
    PL = hc.full_pose(base["lead"])
    P, tr, done = despike_hermite(PL, base["lead"]["transl"], ranges or ())
    b2 = {r: dict(base[r]) for r in ROLES}
    b2["lead"]["transl"] = tr
    ch = np.flatnonzero((np.abs(P - PL).max((1, 2)) > 0) | (np.abs(tr - np.asarray(base["lead"]["transl"], float)).max(1) > 0))
    out_dir = FITS[out]
    out_dir.mkdir(parents=True, exist_ok=True)
    d = dict(b2["lead"])
    T = len(P)
    d["global_orient"] = P[:, 0].copy()
    d["body_pose"] = P[:, 1:22].reshape(T, 63).copy()
    d["left_hand_pose"] = P[:, 25:40].reshape(T, 45).copy()
    d["right_hand_pose"] = P[:, 40:55].reshape(T, 45).copy()
    J = np.asarray(base["lead"]["joints"], float).copy()
    if len(ch):
        J[ch] = posed(b2["lead"], ch, P[ch])["J"]
    d["joints"] = J
    np.savez(out_dir / "lead.npz", **d)
    np.savez(out_dir / "follow.npz", **dict(base["follow"]))
    (OUT / "runs" / out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "runs" / out / "poses_merged.npz", lead=P)
    rep = {"what": f"{src} + Hermite despikes (zouk_contacts.despike_hermite)", "src": str(FITS.get(src, src)), "despiked": done,
           "frames_changed": int(len(ch)), "fk_check_mm": fk_check_v4(out_dir)}
    write_json(out_dir / "zouk_contacts.json", rep)
    log(f"base {out}: {rep}")
    return rep


def joint_visible(obs, ks: np.ndarray, X: np.ndarray, bodies: dict, kl: dict, kf: dict, margin: float = 0.035,
                  r_scale: float = 0.92) -> np.ndarray:
    """(W, C) phone c sees the point X[i] (W,3) of his right arm at frame ks[i]: the phone has the frame, the point is in front of it, and the
    ray from the phone (stopped `margin` short of the point) passes through no capsule of her torso / head / arms / shoulders or of his torso /
    head / left arm (occluders)"""
    OA, OB, Orad = occluders(bodies, kl, kf)
    out = np.zeros((len(ks), len(obs.cams)), bool)
    for ci, c in enumerate(obs.cams):
        z = obs.z[c]
        Cc = -np.einsum("kji,kj->ki", z["R"][ks], z["t"][ks])
        occ = ray_occluded(Cc, X[:, None], OA, OB, Orad * r_scale, margin)[:, 0]
        _, dep = obs.project(c, X[:, None], ks)
        out[:, ci] = z["ok"][ks] & (dep[:, 0] > 0.1) & ~occ
    return out


def coco_errors_vis(judge, obs, X: np.ndarray, ks: np.ndarray, which: str, vis: np.ndarray, X_left: np.ndarray | None = None,
                    X_alt: np.ndarray | None = None) -> np.ndarray:
    """hc._coco_errors restricted to the phones that see the point (vis (W, C), joint_visible). X_left (his LEFT wrist / elbow): a detection
    nearer to it than to the right joint (X, or X_alt: the other motion compared, so that both keep the same detections) is the detector
    swapping his arms - in the closed hold his left hand holds hers in front of her while the right one is behind her back (2026-10-22:
    the COCO 'right wrist' of cams 01 / 03 at 106.6 and 293.3 s sits on the clasped hands) - and is not counted"""
    es = []
    for ci, c in enumerate(obs.cams):
        z = obs.z[c]
        uv, d = obs.project(c, X[:, None], ks)
        ul = obs.project(c, X_left[:, None], ks)[0] if X_left is not None else None
        ua = obs.project(c, X_alt[:, None], ks)[0] if X_alt is not None else None
        for i, k in enumerate(ks):
            if not vis[i, ci]:
                continue
            u, ok = judge.joint(c, "lead", "right", int(k), which)
            if not (ok and z["ok"][k] and d[i, 0] > 0.1):
                continue
            e = float(np.linalg.norm(uv[i, 0] - u))
            if ul is not None:
                dl = float(np.linalg.norm(ul[i, 0] - u))
                da = float(np.linalg.norm(ua[i, 0] - u)) if ua is not None else e
                if dl < min(e, da):
                    continue
            es.append(e)
    return np.array(es)


def mamma_swapped(obs, c: str, ks: np.ndarray, lm_right: list, lm_left: np.ndarray, hcfg: HoldCfg) -> np.ndarray:
    """(W,) MAMMA's 'right hand' in phone c is nearer (median of its visible points) to the motion's LEFT hand than to its right hand (any of
    lm_right: the motions compared) - MAMMA swapped his hands there"""
    import warnings
    z = obs.z[c]
    hid = hc.landmark_parts()["right_hand"]
    o = z["lm"][ks, hc.PIDX["lead"]][:, hid]
    vv = (z["vis"][ks, hc.PIDX["lead"]][:, hid] > hcfg.vis) & (np.abs(o).sum(-1) > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mo = np.stack([np.nanmedian(np.where(vv, o[..., d], np.nan), 1) for d in range(2)], -1)
        ml = np.nanmedian(obs.project(c, lm_left, ks)[0], 1)
        dl = np.linalg.norm(mo - ml, axis=-1)
        dr = np.min([np.linalg.norm(mo - np.nanmedian(obs.project(c, L, ks)[0], 1), axis=-1) for L in lm_right], 0)
    return np.nan_to_num(dl, nan=np.inf) < np.nan_to_num(dr, nan=np.inf)


def hand_2d_errors_vis(fit: dict, ks: np.ndarray, obs, hcfg: HoldCfg, vis: np.ndarray | None, swap_with: list | None = None) -> np.ndarray:
    """hand_2d_errors (per frame: the median px of the fit's right-hand MAMMA points against MAMMA's visible ones over the phones that see the
    hand) counting only the phones with vis[i, c] (the candidate's palm not hidden from them; None: every phone); swap_with (the motions
    compared, fits): a phone-frame where MAMMA's right hand is nearer to his left hand than to the right hand of every one of them is skipped
    (mamma_swapped)"""
    import warnings
    hid = hc.landmark_parts()["right_hand"]
    hp = hc.hand_points_np(fit["lead"], hc.full_pose(fit["lead"]), ks, "right")
    if swap_with:
        lm_r = [hc.hand_points_np(f["lead"], hc.full_pose(f["lead"]), ks, "right")["lm"] for f in swap_with]
        lm_l = hc.hand_points_np(fit["lead"], hc.full_pose(fit["lead"]), ks, "left")["lm"]
    out = np.full((len(ks), len(obs.cams)), np.nan)
    for ci, c in enumerate(obs.cams):
        z = obs.z[c]
        see, _ = hand_2d(obs, c, ks, hcfg)
        uv, dep = obs.project(c, hp["lm"], ks)
        o = z["lm"][ks, hc.PIDX["lead"]][:, hid]
        vv = (z["vis"][ks, hc.PIDX["lead"]][:, hid] > hcfg.vis) & (np.abs(o).sum(-1) > 0) & (dep > 0.1)
        e = np.where(vv, np.linalg.norm(uv - o, axis=-1), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(e, 1)
        ok = see if vis is None else (see & vis[:, ci])
        if swap_with:
            ok = ok & ~mamma_swapped(obs, c, ks, lm_r, lm_l, hcfg)
        out[:, ci] = np.where(ok, med, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(out, 1)


def face_normals_np(V: np.ndarray, F: np.ndarray, fi: np.ndarray) -> np.ndarray:
    """(W,3) unit normals of face fi[i] of mesh V[i] (W,nv,3)"""
    ar = np.arange(len(fi))
    A, B, C = V[ar, F[fi, 0]], V[ar, F[fi, 1]], V[ar, F[fi, 2]]
    n = np.cross(B - A, C - A)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


RAY_KEEP, RAY_INSIDE, RAY_FLOAT, RAY_OBS = 0, 1, 2, 3


def ray_hit_targets(V: np.ndarray, faces: np.ndarray, C: np.ndarray, thru: np.ndarray, pc0: np.ndarray, mode: np.ndarray, lift_m: float,
                    max_slide_m: float = 0.25, float_pull_m: float = 0.06, device: str = "cpu", obs_free: bool = False) -> dict:
    """per frame: the ray from the phone centre C (W,3) through `thru` (W,3: the palm, or the palm moved to MAMMA's observed hand) and its
    FIRST hit on her torso (V (W,nv,3) posed, faces: her torso faces); mode per frame:
      RAY_INSIDE (the palm inside her): the hit before the palm, within max_slide_m of it -> target = hit + lift_m x normal (a slide along the
                 ray: the hand keeps its pixels in that phone and lies on the side of her the phone sees);
      RAY_FLOAT  (the palm a few cm off her back): the hit at most float_pull_m BEYOND the palm -> target on her surface;
      RAY_OBS    (the phone shows the hand elsewhere): the hit within max_slide_m of the palm -> on her surface (obs_free: else `thru`
                 itself, the 2D fix at the palm's depth, when within max_slide_m; off by default - a back hold's hand lies on her);
      RAY_KEEP   the palm.
    -> {"target", "normal", "pushed" (target differs), "on_surface" (target on her surface), "push_m", "t_hit", "L"}"""
    W = len(C)
    D = thru - C
    L = np.linalg.norm(D, axis=1)
    D = D / np.maximum(L, 1e-9)[:, None]
    t, fi = hc.ray_first_hit(C, D, V, faces, device)
    hit = np.isfinite(t) & (fi >= 0)
    X = C + np.where(hit, t, 0.0)[:, None] * D
    n = np.zeros((W, 3))
    if hit.any():
        n[hit] = face_normals_np(np.asarray(V[hit], float), np.asarray(faces), fi[hit])
        flip = (n * D).sum(1) > 0
        n[flip] *= -1                                     # toward the phone (the side of her it sees)
    Xl = X + lift_m * n
    dist = np.linalg.norm(Xl - pc0, axis=1)
    tgt = pc0.copy()
    on = np.zeros(W, bool)
    m_in = (mode == RAY_INSIDE) & hit & (t < L + 0.01) & (dist <= max_slide_m)
    m_fl = (mode == RAY_FLOAT) & hit & (t > L) & (t <= L + float_pull_m)
    m_ob = (mode == RAY_OBS) & hit & (dist <= max_slide_m) & (t <= L + float_pull_m)
    m_ob2 = (mode == RAY_OBS) & ~m_ob & (np.linalg.norm(thru - pc0, axis=1) <= max_slide_m) & bool(obs_free)
    surf = m_in | m_fl | m_ob
    tgt[surf] = Xl[surf]
    tgt[m_ob2] = thru[m_ob2]
    on[surf] = True
    pushed = surf | m_ob2
    return {"target": tgt, "normal": np.where(on[:, None], n, 0.0), "pushed": pushed, "on_surface": on,
            "push_m": np.linalg.norm(tgt - pc0, axis=1), "t_hit": t, "L": L}


def observed_thru(obs, cam: str, ks: np.ndarray, lm_model: np.ndarray, pc: np.ndarray, frames: np.ndarray, hcfg: HoldCfg,
                  min_lm: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """the palm moved to where phone `cam` shows the hand, at the palm's depth: per frame of ks in `frames` (bool) where MAMMA sees >= min_lm of
    the hand points, the shift (median of MAMMA's visible points - median of the motion's same points, px; median-filtered and smoothed over
    time, interpolated across short gaps) applied to the palm's projection and back-projected at its depth -> (points (W,3), used (W,))"""
    from scipy.ndimage import gaussian_filter1d, median_filter
    z = obs.z[cam]
    hid = hc.landmark_parts()["right_hand"]
    o = z["lm"][ks, hc.PIDX["lead"]][:, hid]
    vv = (z["vis"][ks, hc.PIDX["lead"]][:, hid] > hcfg.vis) & (np.abs(o).sum(-1) > 0)
    uv, dep = obs.project(cam, lm_model, ks)
    ok = z["ok"][ks] & (vv.sum(1) >= min_lm) & frames
    W = len(ks)
    du = np.zeros((W, 2))
    for i in np.flatnonzero(ok):
        du[i] = np.median(o[i][vv[i]], 0) - np.median(uv[i][vv[i]], 0)
    if ok.sum() < 3:
        return pc.copy(), np.zeros(W, bool)
    idx = np.flatnonzero(ok)
    for d in range(2):
        du[:, d] = np.interp(np.arange(W), idx, du[idx, d])
    du = gaussian_filter1d(median_filter(du, size=(5, 1), mode="nearest"), 2.0, axis=0, mode="nearest")
    used = frames & (np.arange(W) >= idx[0]) & (np.arange(W) <= idx[-1])
    R, t, K = z["R"][ks], z["t"][ks], z["K"][ks]
    Xc = np.einsum("kij,kj->ki", R, pc) + t
    u = np.einsum("kij,kj->ki", K, Xc)
    u = u[:, :2] / u[:, 2:] + du
    ray = np.einsum("kij,kj->ki", np.linalg.inv(K), np.concatenate([u, np.ones((W, 1))], 1))
    Xn = ray * (Xc[:, 2] / ray[:, 2])[:, None]
    X = np.einsum("kji,kj->ki", R, Xn - t)
    return np.where(used[:, None], X, pc), used


def window_ray_hits(win: dict, ks: np.ndarray, pc: np.ndarray, Ccam: np.ndarray, side_t: dict, VO: np.ndarray, faces: np.ndarray,
                    names: np.ndarray, cfg, cam_ok: np.ndarray, device: str, thru_obs: tuple | None = None) -> dict:
    """the closing pass's targets of one window (ray_mode "hit"): per frame the mode (inside her -> RAY_INSIDE, a back hold with the palm more
    than float_from_m off her -> RAY_FLOAT, the window's off_phone frames -> RAY_OBS through MAMMA's observed hand (thru_obs)), the first hit
    of the ray phone's ray on her torso (ray_hit_targets); an inside frame whose ray finds no hit within max_slide_m falls back to
    visible_side_targets (the nearest point of her surface facing the phone)"""
    sd = np.asarray(side_t["palm_sd_m"], float)
    W = len(ks)
    mode = np.full(W, RAY_KEEP)
    mode[np.isin(names, BACK_HOLDS) & (sd > cfg.float_from_m)] = RAY_FLOAT
    mode[sd < 0] = RAY_INSIDE
    thru = pc.copy()
    if thru_obs is not None:
        X, used = thru_obs
        mode[used] = RAY_OBS
        thru = np.where(used[:, None], X, pc)
    mode[~np.asarray(cam_ok, bool)] = RAY_KEEP
    tf = faces[np.isin(hc.vertex_joint(), TORSO_JOINTS)[faces].all(1)]
    r = ray_hit_targets(np.asarray(VO, np.float32), tf, Ccam, thru, pc, mode, HoldCfg().palm_lift_m, cfg.max_slide_m, cfg.float_pull_m, device)
    fb = (mode == RAY_INSIDE) & ~r["pushed"] & np.asarray(side_t["pushed"], bool)
    if fb.any():
        r["target"][fb] = side_t["target"][fb]
        r["normal"][fb] = side_t["normal"][fb]
        r["pushed"][fb] = True
        r["on_surface"][fb] = True
        r["push_m"] = np.linalg.norm(r["target"] - pc, axis=1)
    r["modes"] = {nm: int((mode == v).sum()) for nm, v in (("inside", RAY_INSIDE), ("float", RAY_FLOAT), ("obs", RAY_OBS))}
    r["fallback_side"] = int(fb.sum())
    r["palm_sd_m"] = sd
    return r


def rate_limit(d: np.ndarray, step: float, iters: int = 3) -> np.ndarray:
    """a (W,3) displacement track whose frame-to-frame change is at most `step` (m): the larger pushes are kept and spread over the
    neighbouring frames (each frame takes the largest of its own push and its neighbours' pushes reduced by step per frame of distance, in
    the push's direction)"""
    d = np.asarray(d, float).copy()
    W = len(d)
    if W < 2:
        return d
    for _ in range(iters):
        for rng_ in (range(1, W), range(W - 2, -1, -1)):
            for i in rng_:
                pv = d[i - 1] if rng_.step == 1 else d[i + 1]
                diff = d[i] - pv
                n = np.linalg.norm(diff)
                if n > step:
                    # keep the larger of the two pushes: shrink toward the neighbour only if this frame's push is the smaller one
                    if np.linalg.norm(d[i]) >= np.linalg.norm(pv):
                        continue
                    d[i] = pv + diff * (step / n)
    # a final pass guaranteeing the bound
    for i in range(1, W):
        diff = d[i] - d[i - 1]
        n = np.linalg.norm(diff)
        if n > step:
            d[i] = d[i - 1] + diff * (step / n)
    return d


def smooth_wrist(P: np.ndarray, sigma: float = 2.0, weight: np.ndarray | None = None) -> np.ndarray:
    """the right wrist's local rotation low-passed over time (quaternions kept in one hemisphere, gaussian sigma frames, renormalised), blended
    in by weight: the input's wrist flails 50-100 deg within 0.2-0.3 s in the closed holds (69-75, 94-97, 106.3-107.0 s)"""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation as Ro
    P = P.copy()
    j = WRIST["right"]
    if len(P) < 3:
        return P
    q = Ro.from_rotvec(P[:, j]).as_quat()
    for i in range(1, len(q)):
        if (q[i] * q[i - 1]).sum() < 0:
            q[i] *= -1
    qs = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    wq = np.ones(len(P)) if weight is None else np.asarray(weight, float)
    P[:, j] = _slerp_rotvec(P[:, j], Ro.from_quat(qs).as_rotvec(), wq)
    return P


def spike_targets(win: dict, ks: np.ndarray, pc: np.ndarray, Jf: np.ndarray, surf, VOt, Ccam: np.ndarray, hcfg: HoldCfg,
                  ok: np.ndarray) -> dict:
    """the closing pass, round r8, a spike window (the input's hand passing through her for 2-8 frames in a fast transition): on the run's
    frames the palm's target is its position relative to HER torso (spine-3 frame) interpolated between the frames just outside the run
    (the hand stays where it was before and after, as the phones show), laid onto her surface where that is still inside her
    (visible_side_targets); elsewhere the palm"""
    a, b = (int(x) for x in win["run"])
    i0, i1 = int(np.searchsorted(ks, a - 1)), int(np.searchsorted(ks, b + 1))
    tgt = pc.copy()
    pushed = np.zeros(len(ks), bool)
    if 0 <= i0 < len(ks) and i1 < len(ks) and ks[i0] == a - 1 and ks[i1] == b + 1:
        Fr = torso_frame(Jf)
        loc = np.einsum("kji,kj->ki", Fr, pc - Jf[:, SPINE3])
        for i in range(i0 + 1, i1):
            u = (i - i0) / float(i1 - i0)
            l = (1 - u) * loc[i0] + u * loc[i1]
            tgt[i] = Fr[i] @ l + Jf[i, SPINE3]
            pushed[i] = True
    side = visible_side_targets(surf, VOt, tgt, Ccam, np.isin(hc.vertex_joint(), TORSO_JOINTS), hcfg.palm_lift_m, 0.0, ok=ok)
    on = np.asarray(side["pushed"], bool)
    tgt = np.where(on[:, None], side["target"], tgt)
    return {"target": tgt, "normal": np.where(on[:, None], side["normal"], 0.0), "pushed": pushed | on, "on_surface": on,
            "push_m": np.linalg.norm(tgt - pc, axis=1), "palm_sd_m": side["palm_sd_m"], "modes": {"spike": int(pushed.sum())},
            "fallback_side": int(on.sum())}


def unfold_wrist(P: np.ndarray, knee_deg: float = 75.0, slope: float = 0.1, weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """a start for folded wrists (2026-10-22 check: 150-179 deg bends, the hand folded back onto the forearm): the right wrist's local rotation
    scaled toward the identity so that a bend b above knee_deg becomes knee_deg + slope (b - knee_deg) (continuous; below the knee untouched),
    times weight (the window's blend) -> (P copy, the bend before (deg))"""
    from scipy.spatial.transform import Rotation as Ro
    P = P.copy()
    j = WRIST["right"]
    b = wrist_angles(P[:, :55], "right")["bend"]
    tgt = np.where(b > knee_deg, knee_deg + slope * (b - knee_deg), b)
    s = np.where(b > 1e-6, 1.0 - tgt / np.maximum(b, 1e-6), 0.0)
    if weight is not None:
        s = s * np.asarray(weight, float)
    if (s > 0).any():
        # the rotation scaled along a path CONTINUOUS in time: quaternions kept in one hemisphere frame to frame (a fold near 180 deg flips
        # the shortest arc to the identity between frames: a 177 deg/frame wrist flick, 2026-10-22 smoke test at 348.1 s)
        q = Ro.from_rotvec(P[:, j]).as_quat()
        if q[0, 3] < 0:
            q[0] *= -1
        for i in range(1, len(q)):
            if (q[i] * q[i - 1]).sum() < 0:
                q[i] *= -1
        xyz = q[:, :3]
        nx = np.linalg.norm(xyz, axis=1)
        ang = 2 * np.arctan2(nx, q[:, 3])                             # [0, 2 pi): continuous through 180 deg
        axis = xyz / np.maximum(nx, 1e-12)[:, None]
        sel = (s > 0) & (nx > 1e-9)
        P[sel, j] = axis[sel] * (ang[sel] * (1.0 - s[sel]))[:, None]
    return P, b


def off_phone_frames(fit: dict, H: dict, obs, hcfg: HoldCfg, min_m: float = 0.08, min_lm: int = 6, back_fv: float = 0.1,
                     min_run: int = 8) -> dict:
    """the frames where the phone that sees most of HER BACK (her front away from it: the most negative facing below back_fv - not a phone in
    front of her, where MAMMA reports the hand-hold hand for a right hand hidden behind her) and sees his right hand (>= min_lm MAMMA hand
    points) shows the hand more than min_m (in its image plane) from where the motion puts it, in the back holds, runs of >= min_run frames
    -> {"mask" (T,), "cam" (T,) str, "err_m" (T,)}"""
    T = len(H["names"])
    ks = np.arange(T)
    kf_J = posed(fit["follow"], ks)["J"]
    fwd = torso_frame(kf_J)[:, :, 2]
    pc = hand_site(posed(fit["lead"], ks)["J"], "right", "palm")[0]
    hid = hc.landmark_parts()["right_hand"]
    best = np.full(T, 1e9)
    cam = np.array([""] * T, dtype=object)
    flag = np.zeros(T, bool)
    err_m = np.full(T, np.nan)
    back = np.isin(H["names"], BACK_HOLDS)
    for a in range(0, T, 2000):
        kk = ks[a:a + 2000]
        hp = hc.hand_points_np(fit["lead"], hc.full_pose(fit["lead"]), kk, "right")
        hl = hc.hand_points_np(fit["lead"], hc.full_pose(fit["lead"]), kk, "left")["lm"]
        for c in obs.cams:
            z = obs.z[c]
            swapped = mamma_swapped(obs, c, kk, [hp["lm"]], hl, hcfg)
            Cc = -np.einsum("kji,kj->ki", z["R"][kk], z["t"][kk]) - kf_J[kk, SPINE3]
            Cc[:, 1] = 0
            fv = (Cc * fwd[kk]).sum(1) / np.maximum(np.linalg.norm(Cc, axis=1), 1e-9)
            o = z["lm"][kk, hc.PIDX["lead"]][:, hid]
            vv = (z["vis"][kk, hc.PIDX["lead"]][:, hid] > hcfg.vis) & (np.abs(o).sum(-1) > 0)
            uv, dep = obs.project(c, hp["lm"], kk)
            e = np.where(vv, np.linalg.norm(uv - o, axis=-1), np.nan)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(e, 1)
            _, dpc = obs.project(c, pc[kk][:, None], kk)
            px_m = z["K"][kk][:, 0, 0] / np.maximum(dpc[:, 0], 0.3)
            ok = z["ok"][kk] & (vv.sum(1) >= min_lm) & (fv < back_fv) & back[kk] & ~swapped
            better = ok & (fv < best[kk])
            best[kk] = np.where(better, fv, best[kk])
            cam[kk] = np.where(better, c, cam[kk])
            flag[kk] = np.where(better, np.nan_to_num(med / px_m, nan=0.0) > min_m, flag[kk])
            err_m[kk] = np.where(better, med / px_m, err_m[kk])
    runs = _runs(flag, 3)
    mask = np.zeros(T, bool)
    for a, b in runs:
        if b - a + 1 >= min_run:
            mask[a:b + 1] = True
    return {"mask": mask & (cam != ""), "cam": cam, "err_m": err_m}


def plan_spikes_folds(base: str, out: str = "plan_r8.json", spike_cm: float = 4.0, spike_max_frames: int = 8, fold_deg: float = 120.0,
                      pad: int = 3, ramp: int = 12, min_fold_frames: int = 3) -> dict:
    """the closing pass, round r8: two kinds of short windows in the motion `base` (exact_<base>.npz):
      spikes - runs of at most spike_max_frames frames with his hand > spike_cm inside her (the input's fast transitions through her body:
               87.87, 310.33, 347.53, 280.57 s), core = the run +- pad, the ray-hit solve;
      folds  - runs of the wrist bent past fold_deg (unfold_only: the wrist and elbow stay, the hand unfolds, the ray phone's 2D turns it),
               trimmed so that no fold window overlaps a spike window's span
    -> v4/contacts/<out> (plan_windows' format; windows carry "kind" and, for the folds, "unfold_only")"""
    fit = load_fit(base)
    T = len(fit["lead"]["t"])
    with np.load(OUT / f"exact_{base}.npz") as z:
        pen = np.zeros(T)
        pen[z["ks"]] = z["hand_pen"]
    obs = hc.Obs()
    hcfg = HoldCfg()
    H = load_holds()
    wins = []
    taken = np.zeros(T, bool)
    from scipy.ndimage import median_filter
    base_pen = median_filter(pen, size=21, mode="nearest")          # a spike rises > 3 cm above the depth around it (310.33 s: 2.5 -> 10.4 -> 4)
    for a, b in _runs((pen > spike_cm / 100) & (pen - base_pen > 0.03), 2):
        if b - a + 1 > spike_max_frames:
            continue
        c0, c1 = max(0, a - pad), min(T - 1, b + pad)
        w = {"core": [c0, c1], "t": [round(c0 / HZ, 2), round(c1 / HZ, 2)], "chunk": [0, 1], "run": [a, b], "ramp": ramp, "kind": "spike",
             "why_frames": {"spike": int(b - a + 1)}, "pen_cm_max": round(float(pen[a:b + 1].max()) * 100, 1),
             "holds": {h: int((H["names"][c0:c1 + 1] == h).sum()) for h in np.unique(H["names"][c0:c1 + 1])}}
        w.update(choose_ray_cam(fit, np.arange(c0, c1 + 1), obs, hcfg))
        wins.append(w)
        taken[max(0, c0 - 2 * ramp):c1 + 2 * ramp + 1] = True
    bend = wrist_angles(hc.full_pose(fit["lead"]), "right")["bend"]
    for a, b in _runs(bend > fold_deg, 3):
        c0, c1 = max(0, a - pad), min(T - 1, b + pad)
        free = np.flatnonzero(~taken[c0:c1 + 1])
        if len(free) < min_fold_frames + 2 * pad or not (bend[c0:c1 + 1][free] > fold_deg).sum() >= min_fold_frames:
            continue
        for x0, x1 in _runs(~taken[c0:c1 + 1], 0):
            x0, x1 = c0 + x0, c0 + x1
            if (bend[x0:x1 + 1] > fold_deg).sum() < min_fold_frames:
                continue
            w = {"core": [x0, x1], "t": [round(x0 / HZ, 2), round(x1 / HZ, 2)], "chunk": [0, 1], "run": [a, b], "ramp": ramp, "kind": "fold",
                 "unfold_only": True, "why_frames": {"fold": int((bend[x0:x1 + 1] > fold_deg).sum())},
                 "bend_max": round(float(bend[x0:x1 + 1].max()), 1),
                 "holds": {h: int((H["names"][x0:x1 + 1] == h).sum()) for h in np.unique(H["names"][x0:x1 + 1])}}
            w.update(choose_ray_cam(fit, np.arange(x0, x1 + 1), obs, hcfg))
            wins.append(w)
            taken[max(0, x0 - 2 * ramp):x1 + 2 * ramp + 1] = True
    wins.sort(key=lambda w: w["core"][0])
    rep = {"what": f"closing pass round r8: short spike windows + unfold-only fold windows in the motion {base}", "base": base,
           "windows": wins, "window_seconds": round(sum(w["core"][1] - w["core"][0] + 1 for w in wins) / HZ, 1),
           "kinds": {k: sum(1 for w in wins if w["kind"] == k) for k in ("spike", "fold")}}
    write_json(OUT / out, rep)
    log(f"plan r8: {rep['kinds']}, {rep['window_seconds']} s")
    return rep


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
    if "exact" in cfg.problems:
        P["exact"] = exact_torso_pen(cfg.base) > cfg.exact_pen_cm / 100
    off = None
    if "fold" in cfg.problems:                          # the closing pass: the wrist folded back onto his forearm
        P["fold"] = wrist_angles(hc.full_pose(fit["lead"]), "right")["bend"] > cfg.fold_plan_deg
    if "float" in cfg.problems:                         # ... the hand a few cm off her back in a back hold (exact gap)
        with np.load(OUT / f"exact_{cfg.base}.npz") as z:
            gap = np.zeros(T)
            gap[z["ks"]] = z["hand_gap"]
        P["float"] = (gap > cfg.float_plan_cm / 100) & np.isin(H["names"], BACK_HOLDS)
    if "off_phone" in cfg.problems:                     # ... the back-view phone shows the hand elsewhere
        off = off_phone_frames(fit, H, hc.Obs(), HoldCfg(), cfg.off_phone_m)
        P["off_phone"] = off["mask"]
    if cfg.problems:
        P["any"] = np.any(np.stack([P[k] for k in cfg.problems], 1), 1)
    runs0 = _runs(P["any"], int(round(cfg.merge_s * HZ)))
    has_region = np.isin(H["names"], list(TORSO_HOLD_SITES))
    if cfg.follow_phones:
        has_region = np.zeros(T, bool)                  # the final pass ignores the holds: one window per run of problem frames
        pen_base = exact_torso_pen(cfg.base)
        obs = hc.Obs()
        hcfg = HoldCfg()
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
            if cfg.follow_phones:                       # the move at an end = how deep the hand is inside her there (the push)
                move = max([min(0.3, float(pen_base[x]) + 0.02) for x in ends] + [0.0])
            else:
                move = max([min(0.3, float(rd[x])) if np.isfinite(rd[x]) else 0.0 for x in ends] + [0.0])
            ramp = hc.ramp_for(move, cfg.v_ramp_mps, cfg.ramp_min, cfg.ramp_max)
            win = {"core": [c0, c1], "t": [round(c0 / HZ, 2), round(c1 / HZ, 2)], "chunk": [ci, n], "run": [a, b], "ramp": int(ramp),
                   "holds": {h: int((names == h).sum()) for h in np.unique(names)}, "why_frames": why}
            if cfg.follow_phones:
                win.update(choose_ray_cam(fit, np.arange(c0, c1 + 1), obs, hcfg))
                if off is not None and off["mask"][c0:c1 + 1].sum() >= 8:
                    # on these frames the back-view phone that shows the hand elsewhere leads (its ray through MAMMA's observed hand, its 2D)
                    fr = np.flatnonzero(off["mask"][c0:c1 + 1]) + c0
                    win["off_phone_frames"] = [int(x) for x in fr]
                    win["off_phone_cams"] = [str(off["cam"][x]) for x in fr]
            wins.append(win)
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

    def __init__(self, torch, obs, judge, ks: np.ndarray, bias: dict, device, keep: np.ndarray | None = None,
                 X_right: np.ndarray | None = None, X_left: np.ndarray | None = None, which: str = "elbow"):
        """keep (W, C) (the closing pass): only these phone-frames (the phones that see the elbow); X_right / X_left (W,3): a detection
        nearer to his left elbow than to the right one is the detector swapping his arms and is dropped"""
        self.torch = torch
        C = len(obs.cams)
        W = len(ks)
        U = np.zeros((W, C, 2), np.float32)
        M = np.zeros((W, C), np.float32)
        for ci, c in enumerate(obs.cams):
            z = obs.z[c]
            ur = obs.project(c, X_right[:, None], ks)[0][:, 0] if X_right is not None else None
            ul = obs.project(c, X_left[:, None], ks)[0][:, 0] if X_left is not None else None
            for i, k in enumerate(ks):
                if not z["ok"][k] or (keep is not None and not keep[i, ci]):
                    continue
                u, ok = judge.joint(c, "lead", "right", int(k), which)
                if ok and ur is not None and np.linalg.norm(u - ul[i]) < np.linalg.norm(u - ur[i]):
                    ok = False
                if ok:
                    U[i, ci], M[i, ci] = u, 1.0
        f32 = dict(dtype=torch.float32, device=device)
        self.U, self.M = torch.tensor(U, **f32), torch.tensor(M, **f32)
        self.K = torch.tensor(np.stack([obs.z[c]["K"][ks] for c in obs.cams], 1), **f32)       # (W,C,3,3)
        self.R = torch.tensor(np.stack([obs.z[c]["R"][ks] for c in obs.cams], 1), **f32)
        self.t = torch.tensor(np.stack([obs.z[c]["t"][ks] for c in obs.cams], 1), **f32)
        self.b = torch.tensor(np.asarray(bias.get(f"lead_right_{which}", [0, 0, 0]), float), **f32)
        self.j = WRIST["right"] if which == "wrist" else ELBOW["right"]          # the closing pass also uses the COCO wrist (CocoJoint2D)
        self.n = float(M.sum())

    def point(self, J):
        torch = self.torch
        ax = J[:, WRIST["right"]] - J[:, ELBOW["right"]]
        ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nrm = hc._palm_normal_t(torch, J, "right")
        nrm = nrm - (nrm * ax).sum(-1, keepdim=True) * ax
        nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cr = torch.cross(ax, nrm, dim=-1)
        return J[:, self.j] + ax * self.b[0] + nrm * self.b[1] + cr * self.b[2]

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


def window_inputs(fit_base: dict, H: dict, ks: np.ndarray, bodies: dict, sites: dict, obs, cfg: ZoukCfg, hcfg: HoldCfg,
                  ray: dict | None = None) -> dict:
    """per window frame: the hold, the site track (her frame, smoothed), the region sites, the phones usable for the 2D, her capsules.
    ray (the final pass, cfg.follow_phones): {"cam", "target", "normal", "pushed"} (visible_side_targets) -> the site track and the region
    are those targets (smoothed) instead of the hold's sites"""
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
    occ_test = has_site
    ray_info = None
    if ray is not None:
        pc0 = hand_site(kl["J"], "right", "palm")[0]
        rt = ray
        d = gaussian_filter1d(rt["target"] - pc0, cfg.ray_sigma_frames, axis=0, mode="nearest") if W > 3 else rt["target"] - pc0
        if getattr(cfg, "ray_rate_mps", 0.0) > 0:
            d = rate_limit(d, cfg.ray_rate_mps / HZ)
        site_pt = pc0 + d
        nl = gaussian_filter1d(rt["normal"], cfg.ray_sigma_frames, axis=0, mode="nearest") if W > 3 else rt["normal"]
        nn_ = np.linalg.norm(nl, axis=1, keepdims=True)
        site_n = np.where(nn_ > 0.3, nl / np.maximum(nn_, 1e-9), 0.0)
        has_site = np.ones(W, bool)
        reg = site_pt[:, None].copy()
        reg_ok = np.ones((W, 1), bool)
        occ_test = np.linalg.norm(d, axis=1) > 0.01             # the phones' view of the target is tested where the hand was pushed
        ray_info = {"cam": ray["cam"], "pushed": rt["pushed"], "push_m": np.linalg.norm(d, axis=1),
                    "on_surface": np.asarray(rt.get("on_surface", np.linalg.norm(d, axis=1) > 0.005), bool)}
    # the phones usable for the 2D: MAMMA sees the hand, and (in a hold with a site) the site is not hidden from that phone
    C = len(obs.cams)
    usable = np.zeros((W, C), bool)
    OA, OB, Orad = occluders(bodies, kl, kf)
    lm_sw = None
    if getattr(cfg, "vis_guards", False):              # the closing pass: phone-frames where MAMMA swapped his hands are not used
        PL = hc.full_pose(fit_base["lead"])
        lm_sw = (hc.hand_points_np(fit_base["lead"], PL, ks, "right")["lm"], hc.hand_points_np(fit_base["lead"], PL, ks, "left")["lm"])
    for ci, c in enumerate(obs.cams):
        see, _ = hand_2d(obs, c, ks, hcfg)
        if lm_sw is not None:
            see = see & ~mamma_swapped(obs, c, ks, [lm_sw[0]], lm_sw[1], hcfg)
        z = obs.z[c]
        Cc = -np.einsum("kji,kj->ki", z["R"][ks], z["t"][ks])
        X = np.where(has_site[:, None], site_pt, hand_site(kl["J"], "right", "palm")[0])[:, None]
        occ = ray_occluded(Cc, X, OA, OB, Orad * hcfg.occl_r_scale, hcfg.occl_margin_m)[:, 0]
        usable[:, ci] = see & (~occ | ~occ_test)                 # no site (free / hand hold): every phone that sees the hand
    Af, Bf = capsule_points(bodies["follow"], kf["J"], kf["R"], kf["tips"])
    return {"names": names, "site_pt": site_pt, "site_n": site_n, "has_site": has_site, "region": reg, "region_ok": reg_ok, "usable": usable,
            "her_A": Af, "her_B": Bf, "kf": kf, "kl": kl, "ray": ray_info}


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
        bad = whip_frames_added(a0, a1, hc.speeds(x1 - x0), vi, v0, v1, p99) if getattr(cfg, "vis_guards", False) else             whip_frames(a0, a1, vi, v0, v1, p99)
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
    from dataclasses import replace as _replace
    from .hand_kin import WRIST as HW, ELBOW as HE
    if win.get("unfold_only"):
        # the closing pass, round r8: a folded wrist alone - the wrist and the elbow stay where they are (keep_joint), the hand unfolds and
        # the ray phone's 2D turns it; no target / surface terms (wrist_only)
        cfg = _replace(cfg, wrist_only=True, keep_joint="wrist", unfold=True)
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
    # her mesh (fixed) for the planes + exact checks (+ the ray targets of the final pass)
    bF = hc.Body(fit_in["follow"], device)
    VO = bF.forward_np(poses["follow"][ks], np.asarray(fit_in["follow"]["transl"], float)[ks], verts=True, batch=256)["verts"]
    VOt = torch.as_tensor(VO, device=device)
    poses_in = poses
    unf = None
    if cfg.unfold:
        # the closing pass: folded wrists start unfolded (the guards below still compare with the motion as it was, poses_in)
        P_u, bend_in = unfold_wrist(poses["lead"][ks], knee_deg=cfg.unfold_knee_deg, weight=w_np)
        if cfg.wrist_sigma > 0:
            P_u = smooth_wrist(P_u, cfg.wrist_sigma, w_np)
        if (bend_in[w_np > 0] > 85.0).any() or cfg.wrist_sigma > 0:
            poses = dict(poses)
            poses["lead"] = poses["lead"].copy()
            poses["lead"][ks] = P_u
            unf = {"frames_unfolded": int(((bend_in > 85.0) & (w_np > 0)).sum()), "bend_max_before": round(float(bend_in.max()), 1)}
    ray = None
    if cfg.follow_phones:
        cam = win.get("ray_cam") or choose_ray_cam({r: {**fit_in[r], **_pose_fields(poses[r])} for r in ROLES}, ks, obs, hcfg)["ray_cam"]
        # the ray phone per frame: the window's, or on its off_phone frames the back-view phone that shows the hand elsewhere
        cam_f = np.array([cam] * len(ks), dtype=object)
        offf = dict(zip(win.get("off_phone_frames") or [], win.get("off_phone_cams") or []))
        if cfg.ray_mode == "hit":
            for i, k in enumerate(ks):
                if int(k) in offf and offf[int(k)]:
                    cam_f[i] = offf[int(k)]
        Ccam = np.zeros((len(ks), 3))
        ok_f = np.zeros(len(ks), bool)
        for c in set(cam_f.tolist()):
            ii = np.flatnonzero(cam_f == c)
            zc_ = obs.z[c]
            Ccam[ii] = -np.einsum("kji,kj->ki", zc_["R"][ks[ii]], zc_["t"][ks[ii]])
            ok_f[ii] = zc_["ok"][ks[ii]]
        kl0 = posed(fit_in["lead"], ks, poses["lead"][ks])
        pc_start = hand_site(kl0["J"], "right", "palm")[0]
        side_t = visible_side_targets(surf, VOt, pc_start, Ccam, np.isin(hc.vertex_joint(), TORSO_JOINTS), hcfg.palm_lift_m, cfg.ray_alpha,
                                      ok=ok_f)
        if cfg.ray_mode == "hit" and win.get("kind") == "spike" and cfg.spike_interp:
            ray = {"cam": cam, "cam_f": cam_f, **spike_targets(win, ks, pc_start, posed(fit_in["follow"], ks, poses["follow"][ks])["J"], surf, VOt,
                                                               Ccam, hcfg, ok_f)}
        elif cfg.ray_mode == "hit":
            thru_obs = None
            if offf:
                lm_m = hc.hand_points_np(fit_in["lead"], poses["lead"], ks, "right")["lm"]
                X_o, used_o = pc_start.copy(), np.zeros(len(ks), bool)
                for c in set(offf.values()):
                    fr = np.array([offf.get(int(k)) == c for k in ks])
                    X_c, u_c = observed_thru(obs, c, ks, lm_m, pc_start, fr, hcfg)
                    X_o[u_c], used_o[u_c] = X_c[u_c], True
                thru_obs = (X_o, used_o)
            ray = {"cam": cam, "cam_f": cam_f, **window_ray_hits(win, ks, pc_start, Ccam, side_t, VO, bF.faces, H["names"][ks], cfg, ok_f,
                                                                 device, thru_obs)}
        else:
            ray = {"cam": cam, "cam_f": cam_f, **side_t}
    wi = window_inputs({r: {**fit_in[r], **_pose_fields(poses[r])} for r in ROLES}, H, ks, bodies, sites, obs, cfg, hcfg, ray)
    names = wi["names"]
    hold_core = names[(ks >= c0) & (ks <= c1)]
    main_hold = max(set(hold_core.tolist()), key=hold_core.tolist().count) if len(hold_core) else "free"
    tmpl = TEMPLATE_OF_ZHOLD.get(main_hold) if (main_hold in TORSO_HOLD_SITES and not cfg.wrist_only and not cfg.follow_phones) else None
    actor = hc.Actor(fit_in["lead"], "lead", "right", ks, poses["lead"], device, tmpl)
    if win.get("unfold_only") and cfg.freeze_arm_unfold:
        actor.freeze_arm()                              # round r9: only the wrist turns (the shoulder / elbow deltas stay 0)
    _ce = np.flatnonzero(w_np > 0.99)
    actor.core = (int(_ce[0]), int(_ce[-1])) if len(_ce) else None
    usable = wi["usable"]
    if cfg.follow_phones:
        # MAMMA's hand in the other phones is often inconsistent with the ray phone (the input fit is already their compromise): the 2D term
        # keeps the ray phone's view only; the independent COCO elbow (every phone) still holds the arm
        usable = usable & (np.asarray(obs.cams, dtype=object)[None, :] == np.asarray(ray["cam_f"], dtype=object)[:, None])
    twod = VisTwoD(torch, obs, actor, device, list(obs.cams), usable)
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
    surf_on = has_reg
    w_region = cfg.w_region
    if cfg.follow_phones:
        # the palm flat on her surface only where it was pushed out of her (else it stays where it is: the region = the palm's own place);
        # the target outweighs the push-out to her NEAREST surface (which may be her side, away from where the ray phone shows the hand)
        from scipy.ndimage import uniform_filter1d
        so = np.asarray(wi["ray"].get("on_surface", wi["ray"]["push_m"] > 0.005), float)
        surf_on = torch.tensor(np.minimum(1.0, uniform_filter1d(so, 7, mode="nearest") * 2).astype(np.float32), device=device)
        w_region = cfg.w_region * cfg.ray_region_scale
    ct = hc.coco_targets(fit_in["lead"], "lead", "right", ks, judge, obs, list(obs.cams), bias)
    Xe, ve, nve = ct["elbow"]
    e0 = posed(fit_in["lead"], ks, poses["lead"][ks])["J"][:, HE["right"]]
    ve = ve & (np.linalg.norm(np.nan_to_num(Xe) - e0, axis=1) < 0.25) & (nve >= 2)
    Xe_t = torch.tensor(np.nan_to_num(Xe), **f32)
    ve_t = torch.tensor(ve.astype(np.float32), device=device)
    if cfg.vis_guards:
        kls = posed(fit_in["lead"], ks, poses["lead"][ks])
        keep_e = joint_visible(obs, ks, kls["J"][:, HE["right"]], bodies, kls, wi["kf"])
        el2d = CocoElbow2D(torch, obs, judge, ks, bias, device, keep_e, kls["J"][:, HE["right"]], kls["J"][:, ELBOW["left"]])
        wr2d = None
        if cfg.w_wrist2d > 0:
            # ... and the COCO wrist in the phones that see it (the depth fix may not drag the wrist off what a phone that sees it shows)
            keep_w = joint_visible(obs, ks, kls["J"][:, HW["right"]], bodies, kls, wi["kf"])
            wr2d = CocoElbow2D(torch, obs, judge, ks, bias, device, keep_w, kls["J"][:, HW["right"]], kls["J"][:, WRIST["left"]], which="wrist")
    else:
        el2d = CocoElbow2D(torch, obs, judge, ks, bias, device)
        wr2d = None
    # the site attractor is weak where a usable phone sees the hand (the 2D leads there)
    seen = usable.any(1)
    site_w_t = torch.tensor(np.where(seen & (not cfg.follow_phones), cfg.site_seen_scale, 1.0) * wi["has_site"], **f32)
    reg_slack = cfg.ray_slack_m if cfg.follow_phones else cfg.region_slack_m
    # speed caps (hand_contact's): the input's p99 of the hand or its own speed there, + 0.1
    J_in = np.asarray(fit_in["lead"]["joints"], float)[ks]
    ref = (speed_ref or {}).get("lead_right", {"wrist": 2.0, "palm": 2.2})
    tr_in = hc.hand_tracks(J_in, "right")
    caps_sp = [torch.tensor(np.maximum(ref[k], hc.speeds(tr_in[k])) + 0.1, **f32) for k in ("wrist", "palm")]
    wr_rot = torch.tensor((3.0, 3.0, 3.0) if main_hold in TORSO_HOLD_SITES else (8.0, 8.0, 4.0), **f32)
    rc = hc.RefineCfg()
    with torch.no_grad():
        tips0 = actor.forward(torch.zeros_like(w))["tips5"].clone()       # the start's fingertips (w = 0: the start pose)

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
    # the closing pass (vis_guards): the limits in the loss with a margin inside the gate's (-5 / 35 deg): round r6 lost 13 windows to
    # elbows at -5.5..-7 deg / 35-36 deg off the hinge
    hyp_m, off_m, off_add = (-2.0, 31.0, 2.0) if cfg.vis_guards else (-5.0, 35.0, 5.0)
    hyp_lim_t = torch.tensor(np.radians(np.minimum(hyp_m, _ea0["flexion"])), **f32)
    off_lim_t = torch.tensor(np.radians(np.maximum(off_m, np.abs(_ea0["off_hinge"]) + off_add)), **f32)

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
            lr = w_region * ((torch.relu(soft - reg_slack) ** 2) * has_reg * w).sum() / wsum
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
            near = (planes.near[:, nh + na + nu + 5:].sum(1) > 0).float() * surf_on
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
        if wr2d is not None and wr2d.n > 0:
            le = le + cfg.w_wrist2d * wr2d.loss(J, cfg.sigma_elbow2d_px, w) / wsum
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
            if cfg.w_speed_tips > 0:
                # the elbow and the five fingertips too (the gate's whip test): their added motion against the start beyond v_free_mps
                for X, X0 in ((o["tips5"], tips0), (J[:, HE["right"]][:, None], actor.elbow0[:, None])):
                    dv = ((X - X0)[1:] - (X - X0)[:-1]) * HZ
                    lsp = lsp + cfg.w_speed_tips * (torch.relu(torch.sqrt((dv ** 2).sum(-1) + 1e-10) - rc.v_free_mps) ** 2).sum() / HZ
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
            if cfg.keep_joint == "wrist":
                lk = cfg.w_keep * ((torch.relu((J[:, HW["right"]] - actor.wrist0).norm(dim=-1) - cfg.keep_slack_m) ** 2) * w).sum() / wsum
            else:
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
    core_np_ = core_np
    pose0 = poses_in["lead"][ks].copy()                 # the motion as it was (before an unfolded start)
    M0 = measures(pose0)
    with torch.no_grad():
        o0 = actor.forward(w)
        if unf is not None:                             # the 'before' measures (exact, 2D) of the motion as it was, not the unfolded start
            a0_ = hc.Actor(fit_in["lead"], "lead", "right", ks, poses_in["lead"], device, None)
            o0 = a0_.forward(torch.zeros_like(w))
            del a0_
    res = {"core": [int(c0), int(c1)], "frames": [int(a0), int(a1)], "ramp": ramp, "hold": main_hold, "template": tmpl, "unfold": unf,
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
    if wi["ray"] is not None:
        rr = wi["ray"]
        res["ray"] = {"cam": rr["cam"], "pushed_core_frames": int((rr["pushed"] & core_np_).sum()),
                      "push_cm_max": round(float(rr["push_m"][core_np_].max() * 100), 1) if core_np_.any() else 0.0,
                      "push_cm_median_pushed": round(float(np.median(rr["push_m"][rr["pushed"]]) * 100), 1) if rr["pushed"].any() else 0.0}
        if ray is not None and "modes" in ray:
            res["ray"].update({"modes": ray["modes"], "fallback_side": ray["fallback_side"], "on_surface_core": int((ray["on_surface"] & core_np_).sum())})
    if res.get("palm_angle_deg_stage1") is not None and res["palm_angle_deg_stage1"] > rc.flip_deg and not cfg.follow_phones:
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

    vis_j = {}
    if cfg.vis_guards:
        # the closing pass: a detector vouches only for a joint the phone can see - the CANDIDATE's wrist / elbow not hidden behind her
        # body or his own (capsule rays); the same phones for the motion before
        kl1 = posed(fit_in["lead"], ks, pose1)
        for nm_, jj_ in (("elbow", HE["right"]), ("wrist", HW["right"])):
            vis_j[nm_] = joint_visible(obs, ks[core_np], kl1["J"][core_np, jj_],
                                       bodies, {k_: v_[core_np] for k_, v_ in kl1.items()}, {k_: v_[core_np] for k_, v_ in wi["kf"].items()})
        res["vis_phone_frames"] = {k_: int(v_.sum()) for k_, v_ in vis_j.items()}

    def coco_joint_px(J, which):
        jj_ = HE["right"] if which == "elbow" else HW["right"]
        b = np.asarray(bias.get(f"lead_right_{which}", [0, 0, 0]), float)
        X = J[:, jj_] + np.einsum("kij,j->ki", hc._forearm_frame(J, "right"), b)
        if which in vis_j:
            Jo = M1["J"] if J is M0["J"] else M0["J"]
            Xo = Jo[:, jj_] + np.einsum("kij,j->ki", hc._forearm_frame(Jo, "right"), b)
            Xl = M0["J"][:, WRIST["left"] if which == "wrist" else ELBOW["left"]]
            e = coco_errors_vis(judge, obs, X[core_np], ks[core_np], which, vis_j[which], Xl[core_np], Xo[core_np])
        else:
            e = hc._coco_errors(judge, obs, X[core_np], ks[core_np], "lead", "right", which, obs.cams)
        return (float(np.median(e)), len(e)) if len(e) else (None, 0)

    def coco_elbow_px(J):
        return coco_joint_px(J, "elbow")[0]
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
    if b3 is not None and a3 is not None and a3 > b3 + max(cfg.guard_coco_slack_px, cfg.guard_coco_rel * b3):
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
        tb = hc.hand_tracks(M0["J"], "right")[k]
        if cfg.gate_guards:
            # against the START too (the final pass starts from a despiked motion: taking an input glitch out is 'added' speed against the
            # input, not a jump): the gate's whip rule (whip_frames) with the start as the reference
            bad = np.flatnonzero(whip_frames_added(hc.speeds(tb - ti), sv["add"], hc.speeds(tn - tb), sv["v_in"], hc.speeds(tb), sv["v_new"],
                                                   ref[k]))
        if len(bad):
            j = int(bad[np.argmax(sv["add"][bad])])
            why.append(f"{'jump' if len(sv['jump']) else 'whip'} {k} at {ks[j] / HZ:.2f} s added {sv['add'][j]:.2f} m/s (speed {sv['v_new'][j]:.2f})")
        ce = np.flatnonzero(core_np)
        if len(ce):
            ends = np.concatenate([ce[:3], ce[-3:]])
            ramp_need = max(ramp_need, hc.ramp_for(float(np.linalg.norm(tn[ends] - tb[ends], axis=1).max()), cfg.v_ramp_mps, cfg.ramp_min, cfg.ramp_max))
    if cfg.gate_guards:
        g_why, g_ramp = _window_gate(fit_in, ks, M0, M1, core_np, ref, cfg)
        why += g_why
        ramp_need = max(ramp_need, g_ramp)

        (cw0, n0), (cw1, n1) = coco_joint_px(M0["J"], "wrist"), coco_joint_px(M1["J"], "wrist")
        res["coco_wrist_px"] = [cw0, cw1]
        if cw0 is not None and cw1 is not None and min(n0, n1) >= GATE["coco_min_n"] and                 cw1 > cw0 + max(GATE["coco_worse_px"], cfg.guard_coco_rel * cw0):
            why.append(f"COCO wrist {cw0:.1f} -> {cw1:.1f} px")
    res["ramp_needed"] = int(ramp_need)
    import os as _os
    if _os.environ.get("DANCECAP_ZC_DEBUG") and wi.get("ray") is not None:
        dd = OUT / "runs" / "_debug"
        dd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dd / f"win_{c0}_{c1}.npz", ks=ks, w=w_np, site_pt=wi["site_pt"], pc0=hand_site(posed(fit_in["lead"], ks, poses["lead"][ks])["J"], "right", "palm")[0],
                            pc1=hand_site(M1["J"], "right", "palm")[0], pushed=wi["ray"]["pushed"], on_surface=wi["ray"]["on_surface"],
                            push_m=wi["ray"]["push_m"], cam_f=np.asarray(ray["cam_f"], str) if ray is not None and "cam_f" in ray else np.zeros(0),
                            tips0=M0["tips"], tips1=M1["tips"], pe0=pe0, pe1=pe1)
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
    if cfg.despike:                                  # input glitches out of the base before solving (the final pass)
        poses["lead"], _ = despike_poses(poses["lead"], cfg.despike)
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
    if cfg.despike:                     # every part solved from the despiked base: a part's change = against THAT (else the last part's
        P0, _ = despike_poses(P0, cfg.despike)      # despiked-but-unsolved frames would overwrite another part's solved ones)
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
    t0 = time.time()
    fit = load_fit(name)
    T = len(fit["lead"]["t"])
    ks = np.arange(T) if ks is None else np.asarray(ks)
    out = exact_of(fit, ks, device, batch)
    n = len(ks)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"exact_{name}.npz", ks=ks, parts=np.array(EXACT_PARTS), **out)
    log(f"exact {name}: {n} frames in {time.time() - t0:.0f} s")
    return out


def exact_of(fit: dict, ks: np.ndarray, device: str = "cuda", batch: int = 48) -> dict:
    """exact_measure's numbers for a fit held in memory (frames ks): hand_pen, arm_pen, hand_gap, hand_pen_part"""
    import torch
    ks = np.asarray(ks)
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


def arm_speed_checks(fit_in: dict, fit: dict, P: np.ndarray | None = None) -> dict:
    """his right elbow, wrist, palm centre and five fingertips against the input fit, the whole film: frames whose ADDED speed (the motion's
    own change against the input) exceeds 0.55 m/s, frames faster than max(input p99, input there) + 0.25 m/s, and frames where the wrist's
    local rotation turns more than 8 deg / frame faster than the input's"""
    Pin = hc.full_pose(fit_in["lead"])
    P = hc.full_pose(fit["lead"]) if P is None else P
    oi, o1 = posed(fit_in["lead"], None, Pin), posed(fit["lead"], None, P)
    pts = {"elbow": (oi["J"][:, ELBOW["right"]], o1["J"][:, ELBOW["right"]]), "wrist": (oi["J"][:, WRIST["right"]], o1["J"][:, WRIST["right"]]),
           "palm": (hc.palm_centre(oi["J"], "right"), hc.palm_centre(o1["J"], "right"))}
    for fi in FINGERS:
        pts[f"tip_{fi}"] = (oi["tips"][:, tip_index("right", fi)], o1["tips"][:, tip_index("right", fi)])
    out = {}
    for k, (xi, x1) in pts.items():
        a = hc.speeds(x1 - xi)
        v, vi = hc.speeds(x1), hc.speeds(xi)
        p99 = float(np.percentile(vi, 99))
        out[k] = {"added_gt_0.55": int((a > 0.55).sum()), "fast": int((v > np.maximum(p99, vi) + 0.25).sum()), "max_added_mps": round(float(a.max()), 2)}
    out["fingertips_added_gt_0.55"] = int(np.max([out[f"tip_{fi}"]["added_gt_0.55"] for fi in FINGERS]))
    out["wrist_rotation_faster_than_input_8deg"] = int((_local_rot_speed(P, WRIST["right"]) > _local_rot_speed(Pin, WRIST["right"]) + 8.0).sum())
    return out


def validate_final(fits=("input", "v3", "v4"), changed_vs: str = "v3", path: Path | None = None, notes: dict | None = None) -> dict:
    """the final numbers over the whole film, input | v3 | final (v4/smplx_contact): exact penetration of his right hand / forearm into her,
    wrist folds, wrist twist, elbow hyperextension / off-hinge, the hold on-target rate, MAMMA's hand points and the COCO wrist / elbow (all
    frames and the frames the final changed against v3), the speed audit (hand_contact's, + the elbow, fingertips and wrist rotation)
    -> v4/contacts/validate_final.json + validate_final.md (the table)"""
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
    P = {f: hc.full_pose(fl[f]["lead"]) for f in fits}
    last = fits[-1]
    changed = np.flatnonzero(np.abs(P[last] - P[changed_vs]).max((1, 2)) > 0)
    torso_holds = [h for h in TORSO_HOLD_SITES if (H["names"] == h).sum() >= 15]
    hold_frames = np.flatnonzero(np.isin(H["names"], torso_holds))
    allk = np.arange(T)
    rep = {"what": "zouk_contacts final validation, the leader's right hand over the whole film (input = smplx_rf2_noslip, v3 = the v3 hand-"
                   "contact motion, final = v4/smplx_contact)", "frames": T, "frames_changed_final_vs_v3": int(len(changed)), "fits": {},
           "notes": dict(notes or {})}
    for f in fits:
        fit = fl[f]
        r = {}
        with np.load(OUT / f"exact_{f}.npz") as z:
            hp, ap, gap = z["hand_pen"], z["arm_pen"], z["hand_gap"]
            parts = [str(x) for x in z["parts"]]
            hpp = z["hand_pen_part"]
        r["exact_hand"] = [int((hp > 0.005).sum()), int((hp > 0.02).sum()), round(float(np.percentile(hp[hp > 0.005], 90)) * 100, 2) if (hp > 0.005).any() else 0.0]
        r["exact_hand_gt5_gt8"] = [int((hp > 0.05).sum()), int((hp > 0.08).sum())]
        tp = hpp[:, [parts.index(p) for p in ("torso", "shoulder_L", "shoulder_R")]].max(1)
        r["exact_hand_torso"] = [int((tp > 0.005).sum()), int((tp > 0.02).sum())]
        r["exact_hand_neck_head"] = int((hpp[:, parts.index("neck_head")] > 0.005).sum())
        r["exact_forearm"] = [int((ap > 0.005).sum()), int((ap > 0.02).sum())]
        back = np.isin(H["names"], ("upper_back", "wrap_back", "behind_back"))
        r["floating_back_holds_gt2cm"] = int(((gap > 0.02) & back).sum())
        Wa, Ea = wrist_angles(P[f], "right"), elbow_angles(P[f], "right")
        lo = Wa["bend"] < 60
        r["wrist_fold_gt85"] = int((Wa["bend"] > 85).sum())
        r["twist_gt90_gt120_bend_lt60"] = [int(((np.abs(Wa["twist"]) > 90) & lo).sum()), int(((np.abs(Wa["twist"]) > 120) & lo).sum())]
        r["twist_gt90_all"] = int((np.abs(Wa["twist"]) > 90).sum())
        r["twist_gt120_any_bend"] = int((np.abs(Wa["twist"]) > 120).sum())
        r["bend_gt120_150_170"] = [int((Wa["bend"] > 120).sum()), int((Wa["bend"] > 150).sum()), int((Wa["bend"] > 170).sum())]
        r["bend120_and_twist120"] = int(((Wa["bend"] > 120) & (np.abs(Wa["twist"]) > 120)).sum())
        op = off_phone_frames(fit, H, obs, hcfg, 0.08, min_run=1)
        r["off_phone_gt8cm"] = int(op["mask"].sum())
        r["elbow_hyper_lt_m5_m10"] = [int((Ea["flexion"] < -5).sum()), int((Ea["flexion"] < -10).sum())]
        r["elbow_off_hinge_gt35"] = int((np.abs(Ea["off_hinge"]) > 35).sum())
        m = load_measure(f) if (OUT / f"measure_{f}.npz").exists() else run_measure(f)
        r["capsule_hand_vs_own_forearm"] = _depth_stats(m["lead_R_self_forearm"])
        ex = {"ks": allk, "hand_pen": hp, "hand_gap": gap}
        ot = on_target(fit, H, hold_frames, bodies, sites, ex)
        r["on_target_torso_holds"] = round(float(ot.mean()), 3)
        r["on_target_upper_back"] = round(float(ot[H["names"][hold_frames] == "upper_back"].mean()), 3)
        e_all = hand_2d_errors(fit, allk, obs, hcfg)
        r["mamma_hand_px_all"] = [round(float(np.nanmedian(e_all)), 2), int(np.isfinite(e_all).sum())]
        r["mamma_hand_px_changed"] = [round(float(np.nanmedian(e_all[changed])), 2), int(np.isfinite(e_all[changed]).sum())] if len(changed) else None
        for which in ("wrist", "elbow"):
            ce = coco_px(fit, allk, obs, judge, bias, which)
            cc = coco_px(fit, changed, obs, judge, bias, which) if len(changed) else np.zeros(0)
            r[f"coco_{which}_px_all"] = [round(float(np.median(ce)), 2), int(len(ce))]
            r[f"coco_{which}_px_changed"] = [round(float(np.median(cc)), 2), int(len(cc))] if len(cc) else None
            # the phones that see the joint in THIS motion (joint_visible: not hidden behind her body or his own)
            kl_, kf_ = posed(fit["lead"], allk), posed(fit["follow"], allk)
            jj = WRIST["right"] if which == "wrist" else ELBOW["right"]
            vis = joint_visible(obs, allk, kl_["J"][:, jj], bodies, kl_, kf_)
            b_ = np.asarray(bias.get(f"lead_right_{which}", [0, 0, 0]), float)
            X_ = kl_["J"][:, jj] + np.einsum("kij,j->ki", hc._forearm_frame(kl_["J"], "right"), b_)
            ev = coco_errors_vis(judge, obs, X_, allk, which, vis)
            r[f"coco_{which}_px_all_vis"] = [round(float(np.median(ev)), 2), int(len(ev))]
        if f != "input":
            au = hc.speed_audit(fl["input"], fit)
            r["speed_audit_incidents"] = au["n_incidents"]
            r["speed_audit_lead_right"] = au["lead_right"]["incidents"][:12]
            r["arm_speed"] = arm_speed_checks(fl["input"], fit, P[f])
        rep["fits"][f] = r
        log(f"validate_final {f} ({time.time() - t0:.0f} s)")
    rep["table"] = validate_final_table(rep)
    write_json(OUT / "validate_final.json", rep)
    (path or (OUT / "validate_final.md")).write_text(validate_final_md(rep), encoding="utf-8")
    return rep


def validate_final_table(rep: dict) -> list:
    fits = list(rep["fits"])
    F = rep["fits"]

    def col(fn):
        out = []
        for f in fits:
            try:
                out.append(fn(F[f]))
            except (KeyError, TypeError, IndexError):
                out.append("-")
        return out
    rows = [["measure"] + [{"v4": "final", "ec8e": "installed before (ec8e7101)", "inst2": "installed (d034e0b7)", "inst": "installed (5ccf839e)"}.get(f, f)
                           for f in fits]]
    rows.append(["his right hand inside her (exact, her mesh): frames > 0.5 cm / > 2 cm / p90 depth cm"]
                + col(lambda r: f"{r['exact_hand'][0]} / {r['exact_hand'][1]} / {r['exact_hand'][2]}"))
    rows.append(["... frames > 5 cm / > 8 cm"] + col(lambda r: f"{r['exact_hand_gt5_gt8'][0]} / {r['exact_hand_gt5_gt8'][1]}"))
    rows.append(["... inside her torso / shoulders: frames > 0.5 cm / > 2 cm"] + col(lambda r: f"{r['exact_hand_torso'][0]} / {r['exact_hand_torso'][1]}"))
    rows.append(["... inside her neck / head: frames > 0.5 cm"] + col(lambda r: f"{r['exact_hand_neck_head']}"))
    rows.append(["his right forearm inside her (exact): frames > 0.5 cm / > 2 cm"] + col(lambda r: f"{r['exact_forearm'][0]} / {r['exact_forearm'][1]}"))
    rows.append(["hand more than 2 cm off her in the back holds (exact gap): frames"] + col(lambda r: f"{r['floating_back_holds_gt2cm']}"))
    rows.append(["wrist folded past 85 deg: frames"] + col(lambda r: f"{r['wrist_fold_gt85']}"))
    rows.append(["wrist bent past 120 / 150 / 170 deg (folded back onto his forearm): frames"]
                + col(lambda r: f"{r['bend_gt120_150_170'][0]} / {r['bend_gt120_150_170'][1]} / {r['bend_gt120_150_170'][2]}"))
    rows.append(["wrist |twist| > 120 deg at any bend / together with a bend past 120 deg: frames"]
                + col(lambda r: f"{r['twist_gt120_any_bend']} / {r['bend120_and_twist120']}"))
    rows.append(["fingers into his own forearm (capsules): frames > 0.5 cm / > 2 cm"]
                + col(lambda r: f"{r['capsule_hand_vs_own_forearm']['frames_gt_0.5cm']} / {r['capsule_hand_vs_own_forearm']['frames_gt_2.0cm']}"))
    rows.append(["wrist |twist| > 90 / > 120 deg (frames with bend < 60 deg)"]
                + col(lambda r: f"{r['twist_gt90_gt120_bend_lt60'][0]} / {r['twist_gt90_gt120_bend_lt60'][1]}"))
    rows.append(["elbow hyperextension: frames bent backward > 5 / > 10 deg"] + col(lambda r: f"{r['elbow_hyper_lt_m5_m10'][0]} / {r['elbow_hyper_lt_m5_m10'][1]}"))
    rows.append(["elbow off its hinge > 35 deg: frames"] + col(lambda r: f"{r['elbow_off_hinge_gt35']}"))
    rows.append(["hold on-target rate, all torso holds / upper_back (circular: the hold detector's own sites)"]
                + col(lambda r: f"{r['on_target_torso_holds']} / {r['on_target_upper_back']}"))
    rows.append(["MAMMA hand points (phones that see the hand), median px: all frames"] + col(lambda r: f"{r['mamma_hand_px_all'][0]}"))
    rows.append([f"MAMMA hand points, median px: the {rep['frames_changed_final_vs_v3']} frames the final changed vs v3"]
                + col(lambda r: f"{r['mamma_hand_px_changed'][0]}"))
    rows.append(["the back-view phone shows the hand > 8 cm (image plane) from the motion's, back holds: frames"]
                + col(lambda r: f"{r['off_phone_gt8cm']}"))
    for which in ("wrist", "elbow"):
        rows.append([f"COCO {which} (independent detector), median px: all frames"] + col(lambda r, w=which: f"{r[f'coco_{w}_px_all'][0]}"))
        rows.append([f"COCO {which}, median px: changed frames"] + col(lambda r, w=which: f"{r[f'coco_{w}_px_changed'][0]}"))
        rows.append([f"COCO {which}, median px: all frames, only the phones that see the joint in that motion (n detections)"]
                    + col(lambda r, w=which: f"{r[f'coco_{w}_px_all_vis'][0]} ({r[f'coco_{w}_px_all_vis'][1]})"))
    rows.append(["speed audit vs the input (hand_contact: wrist / palm jumps > 0.55 m/s, whips): incidents"]
                + col(lambda r: f"{r['speed_audit_incidents']}"))
    rows.append(["added speed vs the input > 0.55 m/s: elbow / fingertips (worst finger) frames"]
                + col(lambda r: f"{r['arm_speed']['elbow']['added_gt_0.55']} / {r['arm_speed']['fingertips_added_gt_0.55']}"))
    rows.append(["wrist turning > 8 deg / frame faster than the input: frames"] + col(lambda r: f"{r['arm_speed']['wrist_rotation_faster_than_input_8deg']}"))
    return rows


def validate_final_md(rep: dict) -> str:
    rows = rep["table"]
    out = ["# Leader's right hand: final numbers (" + " | ".join(str(c) for c in rows[0][1:]) + ")", "",
           f"Whole film, {rep['frames']} frames at 30 Hz. The final motion changed {rep['frames_changed_final_vs_v3']} frames against v3. "
           "Exact = his hand / forearm against her posed SMPL-X mesh (closest points + winding numbers). MAMMA = the hand landmarks the fit "
           "used, in the phones that see the hand; COCO = an independent detector the fit never saw (bias-corrected joint).", "",
           "| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    last = list(rep["fits"])[-1]
    inc = rep["fits"][last].get("speed_audit_lead_right") or []
    if inc:
        out += ["", f"Speed audit incidents of the {'final' if last == 'v4' else last} motion (hand_contact's rule: the motion's own change against "
                    "the input faster than 0.55 m/s):", ""]
        for x in inc:
            slower = x["speed_mps"] < x["input_mps"]
            out.append(f"- {x['t']} s, {x['track']}: added {x['added_mps']} m/s; the hand moves at {x['speed_mps']} m/s where the input moves at "
                       f"{x['input_mps']} m/s" + (" (slower than the input: a despiked input glitch, not a whip)" if slower else ""))
    for k, v in (rep.get("notes") or {}).items():
        out += ["", f"{k}: {v}"]
    return "\n".join(out) + "\n"


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
                  "# Version: the v4 zouk contacts with the final pass, the closing pass and the last pass (dancecap.zouk_contacts; decisions\n"
                  "# in v4/contacts/curation.json incl. its stages r4 / r5 / r6 / r7 / r67 / r8 + the post-hoc gates v4/contacts/gate.json,\n"
                  "# gate_r4.json, gate_r5.json, gate_r6.json, gate_r7.json, gate_r67.json, gate_r8.json (the final assembly:\n"
                  "# v4/contacts/runs/r8/final_assembly.json); the last pass: v4/contacts/last/report.json), numbers in\n"
                  "# v4/contacts/validate_final.md, the export and the sha256 of each file in\n"
                  "# v4/contacts/unity_motion/unity_motion.json). Earlier installs are kept: v4/contacts/unity_motion_installed_5ccf/\n"
                  "# (smplx_lead.bin 5ccf839e..., before the final pass), v4/contacts/unity_motion_installed_d034/ (d034e0b7..., before\n"
                  "# the closing pass) and v4/contacts/unity_motion_installed_ec8e/ (ec8e7101..., before the last pass).")
    s = s.replace("# never overwrite the first backup (that is the pre-v3 motion)",
                  "# never overwrite the first backup (that is the motion installed before v4: the v3 hand-contact motion; --restore puts that\n"
                  "                                                              # back - the 5ccf839e / d034e0b7 / ec8e7101 motions are in\n"
                  "                                                              # ../unity_motion_installed_5ccf/, _d034/ and _ec8e/)")
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
            "notes": c.get("notes", {}), "despike": list(c.get("despike") or []), "stages": c.get("stages") or {}}


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
    neighbouring windows keep their correction and nothing jumps) -> v4/smplx_contact/.
    base_tag (a STAGE, the final pass): the pass `tag` was solved from runs/<base_tag>/poses_merged.npz (e.g. "inst", the motion installed
    2026-10-22); only curation.json["stages"][tag] applies (its vetoes go back to that motion, its despike ranges are taken out of it), the
    earlier vetoes are already inside it"""
    from .hand_kin import SHOULDER as HS, ELBOW as HE, WRIST as HW, FINGER_JOINTS
    cur = load_curation()
    desp_back = []
    if base_tag:
        st = cur["stages"].get(tag) or {}
        cur = {"veto": {str(k): v for k, v in (st.get("veto") or {}).items()}, "blend_s": float(st.get("blend_s", cur["blend_s"])),
               "despike": []}
        base = load_fit(base_tag) if base_tag in FITS else load_fit("v3")
        P3, desp_back = despike_poses(np.load(OUT / "runs" / base_tag / "poses_merged.npz")["lead"], st.get("despike") or [])
    else:
        base = load_fit("v3")                               # the installed motion
        P3 = hc.full_pose(base["lead"])
    P = np.load(OUT / "runs" / tag / "poses_merged.npz")["lead"].copy()
    if base_tag:
        # the frames no adopted window of the pass changed take the back motion (with THIS stage's despike: the pass may have been solved
        # with other despike ranges)
        rc_ = json.loads((OUT / "runs" / tag / "refine.json").read_text(encoding="utf-8")).get("config") or {}
        Pb_run, _ = despike_poses(np.load(OUT / "runs" / base_tag / "poses_merged.npz")["lead"], rc_.get("despike") or [])
        unsolved = np.abs(P - Pb_run).max((1, 2)) == 0
        P[unsolved] = P3[unsolved]
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
           "despiked": desp or desp_back, "base_tag": base_tag}
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
        "add_mps": 0.55, "add_new_mps": 0.15, "fast_margin_mps": 0.25, "wrist_rot_new_deg_frame": 8.0, "coco_worse_px": 3.0, "coco_min_n": 8,
        # the final pass (2026-10-22) adds: a NEW wrist fold, the COCO elbow and MAMMA's hand points per window, and (exact, her mesh) new
        # penetration and a hand that now floats off her where it touched her
        "fold_deg": 85.0, "fold_new_deg": 10.0, "coco_elbow_worse_px": 4.0, "mamma_worse_px": 6.0, "mamma_min_n": 8,
        "pen_new_cm": 1.0, "pen_new_frames": 3, "float_cm": 2.5, "float_ref_cm": 0.5, "float_frames": 10, "whip_slower_mps": -0.1,
        # the closing pass: a whip moves at least this fast (whip_frames_added); a COCO check allows max(px, this x the reference's px)
        "whip_min_mps": 0.8, "coco_rel_vis": 0.3}
GATE_TAGS = ("all2", "pass2", "pass3")


def new_fold(bend_cand: np.ndarray, bend_ref: np.ndarray, bend_in: np.ndarray) -> np.ndarray:
    """a NEW wrist fold: the candidate's whole wrist bend past fold_deg and fold_new_deg more than both the reference and the input"""
    g = GATE
    return (bend_cand > g["fold_deg"]) & (bend_cand > np.maximum(bend_ref, bend_in) + g["fold_new_deg"])


def whip_frames(a0: np.ndarray, a1: np.ndarray, v_in: np.ndarray, v0: np.ndarray, v1: np.ndarray, p99: float) -> np.ndarray:
    """(T-1,) bool: a point of his arm whips in the candidate. a = the speed of the motion's own change against the input (added), v = the
    speed; 0 = the reference (installed) motion, 1 = the candidate. A whip: the candidate's added speed is over add_mps, add_new_mps more than
    the reference's, AND the candidate is faster there than the input / the reference (+ whip_slower_mps) - a fix that only calms a flick of
    the input (the fingertips at 2.5 m/s, 106.4-106.7 s) adds 'speed' against it without whipping; or the candidate is faster than
    max(p99 of the input, the input, the reference) + fast_margin_mps"""
    g = GATE
    added = (a1 > g["add_mps"]) & (a1 > a0 + g["add_new_mps"]) & (v1 > np.maximum(v_in, v0) + g["whip_slower_mps"])
    return added | (v1 > np.maximum.reduce([np.full_like(v1, p99), v_in, v0]) + g["fast_margin_mps"])


def whip_frames_added(a0: np.ndarray, a1: np.ndarray, a_start: np.ndarray, v_in: np.ndarray, v0: np.ndarray, v1: np.ndarray,
                      p99: float) -> np.ndarray:
    """whip_frames where the candidate also ADDS more than add_mps against its own start (a_start = the speed of candidate - start): a
    despiked base already differs from the input by a glitch's speed there (165.73-166.00 s), which is not the window's whip; the
    'faster than everything' rule stays"""
    g = GATE
    fast = v1 > np.maximum.reduce([np.full_like(v1, p99), v_in, v0]) + g["fast_margin_mps"]
    # and the point itself moves fast enough to be seen as a whip: where the input flails (106.3-107.0 s: the wrist 101 -> 18 -> 112 deg in
    # 0.3 s, fingertips 3 m/s) a calm candidate differs from it by > 0.55 m/s at frames where it moves at 0.4-0.5 m/s
    return (whip_frames(a0, a1, v_in, v0, v1, p99) & (a_start > g["add_mps"]) & (v1 > g["whip_min_mps"])) | fast


def _local_rot_speed(P: np.ndarray, j: int) -> np.ndarray:
    """(T-1,) degrees per frame the local rotation of joint j turns between consecutive frames"""
    from scipy.spatial.transform import Rotation as Ro
    R = Ro.from_rotvec(P[:, j])
    return np.degrees((R[:-1].inv() * R[1:]).magnitude())


def gate_flags(fits: dict | None = None, ref: str = "v3", cand: str = "v4", added_vs_ref: bool = False) -> dict:
    """per frame of the leader's right arm: the problems the candidate (fits[cand], v4) has that neither the reference (fits[ref]: the installed
    v3, or the motion installed before the final pass) nor the input has -> {name: bool (T,)}"""
    fits = fits or {f: load_fit(f) for f in ("input", "v3", "v4")}
    fits = {"input": fits["input"], "v3": fits[ref], "v4": fits[cand]}              # below "v3" = the reference, "v4" = the candidate
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
    flags["wrist_fold"] = new_fold(W["v4"]["bend"], W["v3"]["bend"], W["input"]["bend"])
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
        # the closing pass (added_vs_ref): a whip must also ADD > add_mps against the reference (its despikes are not the candidate's)
        bad = whip_frames_added(a3, a4, hc.speeds(x4 - x3), v_in, v3, v4, p99) if added_vs_ref else whip_frames(a3, a4, v_in, v3, v4, p99)
        if added_vs_ref and k in ("wrist", "palm"):
            # the wrist and the palm keep hand_contact's speed-audit rule (the owner-facing one: added > 0.55 m/s against the input while
            # faster than the input) wherever the candidate itself added it against the reference (not the reference's despikes)
            bad = bad | ((a4 > g["add_mps"]) & (hc.speeds(x4 - x3) > g["add_mps"]) & (v4 > v_in))
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


def coco_window_check(fits: dict, wins: list[dict], ref: str = "v3", cand: str = "v4", mamma: bool = False, vis: bool = False) -> dict:
    """per window: the independent COCO wrist (median px over the core, every phone with a detection) of the reference and the candidate
    (keys v3_px / v4_px); the final pass (mamma=True) adds the COCO elbow and MAMMA's hand points (phones that see the hand) per window.
    vis (the closing pass): every check counts only the phones that see the CANDIDATE's joint / palm (joint_visible: not hidden behind her body
    or his own), the same phones for the reference"""
    obs = hc.Obs()
    judge = hc.CocoJudge()
    bias = hc.calibrate_coco_bias(fits["input"], obs, judge)
    hcfg = HoldCfg()
    out = {}
    if vis:
        bodies = capsule_bodies(fits["input"])

        def joint_x(fit, ks, which):
            J = posed(fit["lead"], ks)["J"]
            jj = WRIST["right"] if which == "wrist" else ELBOW["right"]
            b = np.asarray(bias.get(f"lead_right_{which}", [0, 0, 0]), float)
            return J[:, jj] + np.einsum("kij,j->ki", hc._forearm_frame(J, "right"), b)

        def med(a):
            a = np.asarray(a, float)
            a = a[np.isfinite(a)]
            return (round(float(np.median(a)), 2) if len(a) else None), int(len(a))
        for w in wins:
            ks = np.arange(w["core"][0], w["core"][1] + 1)
            kl, kf = posed(fits[cand]["lead"], ks), posed(fits[cand]["follow"], ks)
            r = {}
            for which, key in (("wrist", ""), ("elbow", "elbow_")):
                v = joint_visible(obs, ks, kl["J"][:, WRIST["right"] if which == "wrist" else ELBOW["right"]], bodies, kl, kf)
                Xl = kl["J"][:, WRIST["left"] if which == "wrist" else ELBOW["left"]]
                X3, X4 = joint_x(fits[ref], ks, which), joint_x(fits[cand], ks, which)
                a3, n3 = med(coco_errors_vis(judge, obs, X3, ks, which, v, Xl, X4))
                a4, n4 = med(coco_errors_vis(judge, obs, X4, ks, which, v, Xl, X3))
                if which == "wrist":
                    r.update({"n": min(n3, n4), "v3_px": a3, "v4_px": a4})
                else:
                    r.update({"elbow_n": min(n3, n4), "elbow_v3_px": a3, "elbow_v4_px": a4})
            if mamma:
                vp = joint_visible(obs, ks, hand_site(kl["J"], "right", "palm")[0], bodies, kl, kf)
                (m3, k3), (m4, k4) = (med(hand_2d_errors_vis(fits[f], ks, obs, hcfg, vp, [fits[ref], fits[cand]])) for f in (ref, cand))
                r.update({"mamma_n": min(k3, k4), "mamma_v3_px": m3, "mamma_v4_px": m4})
            r["vis"] = True
            out[w["id"]] = r
        return out

    def med(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return (round(float(np.median(a)), 2) if len(a) else None), int(len(a))
    for w in wins:
        ks = np.arange(w["core"][0], w["core"][1] + 1)
        e3, e4 = coco_px(fits[ref], ks, obs, judge, bias, "wrist"), coco_px(fits[cand], ks, obs, judge, bias, "wrist")
        out[w["id"]] = {"n": int(min(len(e3), len(e4))), "v3_px": round(float(np.median(e3)), 2) if len(e3) else None,
                        "v4_px": round(float(np.median(e4)), 2) if len(e4) else None}
        if mamma:
            (a3, n3), (a4, n4) = med(coco_px(fits[ref], ks, obs, judge, bias, "elbow")), med(coco_px(fits[cand], ks, obs, judge, bias, "elbow"))
            out[w["id"]].update({"elbow_n": min(n3, n4), "elbow_v3_px": a3, "elbow_v4_px": a4})
            (m3, k3), (m4, k4) = med(hand_2d_errors(fits[ref], ks, obs, hcfg)), med(hand_2d_errors(fits[cand], ks, obs, hcfg))
            out[w["id"]].update({"mamma_n": min(k3, k4), "mamma_v3_px": m3, "mamma_v4_px": m4})
    return out


def window_check_vetoes(chk: dict, wins: list[dict], skip=(), exempt: dict | None = None, coco_rel: float = 0.0) -> dict:
    """the per-window 2D checks -> vetoes: the COCO wrist more than coco_worse_px worse, the COCO elbow more than coco_elbow_worse_px, MAMMA's
    hand points more than mamma_worse_px (each with enough detections). exempt {window id: ["mamma" | "coco_wrist" | "coco_elbow"]}: checks a
    curation "keep" (decided by looking at the phone crops, with its reason) waives for that window"""
    g = GATE
    out = {}
    exempt = exempt or {}
    for w in wins:
        c = chk.get(w["id"])
        if c is None or w["id"] in skip:
            continue
        why = []
        ex = set(exempt.get(w["id"], ()))
        if "coco_wrist" not in ex and c["n"] >= g["coco_min_n"] and c["v3_px"] is not None and c["v4_px"] is not None and                 c["v4_px"] - c["v3_px"] > max(g["coco_worse_px"], coco_rel * c["v3_px"]):
            why.append(f"the independent COCO wrist {c['v3_px']} -> {c['v4_px']} px (n {c['n']})")
        if "coco_elbow" not in ex and c.get("elbow_n", 0) >= g["coco_min_n"] and c.get("elbow_v4_px") is not None and c.get("elbow_v3_px") is not None and \
                c["elbow_v4_px"] - c["elbow_v3_px"] > max(g["coco_elbow_worse_px"], coco_rel * c["elbow_v3_px"]):
            why.append(f"the COCO elbow {c['elbow_v3_px']} -> {c['elbow_v4_px']} px (n {c['elbow_n']})")
        if "mamma" not in ex and c.get("mamma_n", 0) >= g["mamma_min_n"] and c.get("mamma_v4_px") is not None and c.get("mamma_v3_px") is not None and \
                c["mamma_v4_px"] - c["mamma_v3_px"] > g["mamma_worse_px"]:
            why.append(f"MAMMA hand points {c['mamma_v3_px']} -> {c['mamma_v4_px']} px (n {c['mamma_n']})")
        if why:
            out[w["id"]] = {"core_s": w["t_span"], "why": "gate: " + "; ".join(why), "gate": True}
    return out


def exact_update(name: str = "v4", ref: str = "inst", device: str = "cuda") -> dict:
    """exact_<name>.npz over the whole film, computed only where the leader's pose differs from the reference fit (the rest copied from
    exact_<ref>.npz: the follower and his other frames are the reference's)"""
    f1, f0 = load_fit(name), load_fit(ref)
    if np.abs(hc.full_pose(f1["follow"]) - hc.full_pose(f0["follow"])).max() > 0:
        raise ValueError("the follower differs from the reference: run exact on every frame")
    ks = np.flatnonzero(np.abs(hc.full_pose(f1["lead"]) - hc.full_pose(f0["lead"])).max((1, 2)) > 0)
    with np.load(OUT / f"exact_{ref}.npz") as z:
        d = {k: z[k].copy() for k in z.files}
    if len(ks):
        new = exact_measure(name, device, ks=ks)
        pos = np.searchsorted(d["ks"], ks)
        for k in ("hand_pen", "arm_pen", "hand_gap", "hand_pen_part"):
            d[k][pos] = new[k]
    np.savez_compressed(OUT / f"exact_{name}.npz", **d)
    log(f"exact {name}: {len(ks)} frames recomputed against {ref}")
    return {"frames": int(len(ks))}


def exact_window_vetoes(name: str, ref: str, wins: list[dict], H: dict, skip=()) -> tuple[dict, dict]:
    """per window span, from exact_<name>.npz vs exact_<ref>.npz: NEW penetration (deeper by > pen_new_cm on >= pen_new_frames frames) and a
    hand that now floats off her (gap > float_cm where the reference touched or was inside her, gap <= float_ref_cm, >= float_frames frames,
    torso holds) -> (vetoes, per-window numbers)"""
    with np.load(OUT / f"exact_{name}.npz") as z1, np.load(OUT / f"exact_{ref}.npz") as z0:
        p1, p0, g1, g0 = z1["hand_pen"], z0["hand_pen"], z1["hand_gap"], z0["hand_gap"]
    return exact_window_checks(p0, p1, g0, g1, H["names"], wins, skip)


def exact_window_checks(p0: np.ndarray, p1: np.ndarray, g0: np.ndarray, g1: np.ndarray, names: np.ndarray, wins: list[dict],
                        skip=()) -> tuple[dict, dict]:
    """exact_window_vetoes on arrays (reference p0 / g0, candidate p1 / g1: depth inside her >= 0 and signed gap, m; names: the hold per frame)"""
    g = GATE
    back = np.isin(names, ("upper_back", "wrap_back", "behind_back", "shoulder_top", "chest", "side"))
    out, nums = {}, {}
    for w in wins:
        a, b = w["span"]
        sl = slice(a, b + 1)
        deeper = int((p1[sl] > p0[sl] + g["pen_new_cm"] / 100).sum())
        floats = int(((g1[sl] > g["float_cm"] / 100) & (g0[sl] <= g["float_ref_cm"] / 100) & back[sl]).sum())
        nums[w["id"]] = {"deeper_frames": deeper, "float_frames": floats,
                         "pen_cm_p90": [round(float(np.percentile(p0[sl], 90)) * 100, 2), round(float(np.percentile(p1[sl], 90)) * 100, 2)],
                         "pen_frames_gt2cm": [int((p0[sl] > 0.02).sum()), int((p1[sl] > 0.02).sum())]}
        if w["id"] in skip:
            continue
        why = []
        if deeper >= g["pen_new_frames"]:
            why.append(f"new penetration (exact): {deeper} frames deeper by > {g['pen_new_cm']} cm")
        if floats >= g["float_frames"]:
            why.append(f"the hand now floats off her (exact gap > {g['float_cm']} cm) on {floats} frames")
        if why:
            out[w["id"]] = {"core_s": w.get("t_span", w.get("t")), "why": "gate: " + "; ".join(why), "gate": True}
    return out, nums


def run_gate(tag: str = "pass3", rounds: int = 4, coco: bool = True, extra_veto: dict | None = None, tags=None, base_tag: str | None = None,
             exact: bool = False, device: str = "cuda", vis: bool = False) -> dict:
    """apply the curation, flag NEW problems frame by frame, veto every adopted window whose span holds one (and, with coco, every window whose
    independent COCO wrist got more than GATE["coco_worse_px"] worse than v3), apply again; repeat until nothing is flagged -> gate.json,
    curation.json (the gate's vetoes marked "gate"), v4/smplx_contact/.
    base_tag (the final pass, a stage): the reference is that motion (runs/<base_tag>/, e.g. "inst", the motion installed before it), the vetoes
    go to curation.json["stages"][tag]; the window checks add the COCO elbow and MAMMA's hand points, and with exact the exact mesh check of
    the candidate (exact_update on the changed frames) adds new penetration / a newly floating hand -> gate_<tag>.json"""
    cur_raw = json.loads(CURATION.read_text(encoding="utf-8")) if CURATION.exists() else {}
    cur_raw.setdefault("veto", {})
    if base_tag:
        stage = cur_raw.setdefault("stages", {}).setdefault(tag, {})
        stage["base_tag"] = base_tag
        vetoes = stage.setdefault("veto", {})
    else:
        vetoes = cur_raw["veto"]
    for k, v in (extra_veto or {}).items():
        vetoes[k] = v
    CURATION.write_text(json.dumps(cur_raw, indent=1, ensure_ascii=False), encoding="utf-8")
    wins = adopted_windows(tags or ([tag] if base_tag else None))
    ref = base_tag or "v3"
    hist = []
    coco_rep = None
    exact_rep = None
    exact_done = not exact
    H = load_holds()
    for rnd in range(rounds):
        apply_curation(tag, base_tag=base_tag)
        fits = {"input": load_fit("input"), "ref": load_fit(ref), "cand": load_fit("v4")}
        gf = gate_flags(fits, ref="ref", cand="cand", added_vs_ref=vis)
        fl = gf["flags"]
        anyf = np.zeros(len(gf["changed"]), bool)
        for v in fl.values():
            anyf |= v
        vetoed = set(vetoes)
        new = {}
        keep_ex = {k: set(v.get("exempt", [])) for k, v in ((cur_raw.get("stages", {}).get(tag, {}).get("keep") or {}) if base_tag else {}).items()}
        for w in wins:
            if w["id"] in vetoed:
                continue
            a, b = w["span"]
            # a curation keep (decided by looking, with its reason) may waive a named frame check for that window
            hit = {k: int(v[a:b + 1].sum()) for k, v in fl.items() if v[a:b + 1].any() and k not in keep_ex.get(w["id"], ())}
            if hit:
                new[w["id"]] = {"core_s": w["t_span"], "why": f"gate round {rnd + 1}: new problems in its span: {hit}", "gate": True}
        if coco and coco_rep is None:
            live = [w for w in wins if w["id"] not in vetoed and w["id"] not in new]
            coco_rep = coco_window_check(fits, live, ref="ref", cand="cand", mamma=bool(base_tag), vis=vis)
            keep = (cur_raw.get("stages", {}).get(tag, {}).get("keep") or {}) if base_tag else {}
            new.update(window_check_vetoes(coco_rep, live, exempt={k: v.get("exempt", []) for k, v in keep.items()},
                                           coco_rel=GATE["coco_rel_vis"] if vis else 0.0))
        if not new and not exact_done:
            exact_update("v4", ref, device)
            live = [w for w in wins if w["id"] not in vetoed]
            keep_ = (cur_raw.get("stages", {}).get(tag, {}).get("keep") or {}) if base_tag else {}
            ev, exact_rep = exact_window_vetoes("v4", ref, live, H, skip=[k for k, v in keep_.items() if "exact" in v.get("exempt", [])])
            new.update(ev)
            exact_done = True
        hist.append({"round": rnd + 1, "flagged_frames": {k: int(v.sum()) for k, v in fl.items()}, "flagged_any": int(anyf.sum()),
                     "flagged_times_s": [round(x / HZ, 2) for x in np.flatnonzero(anyf)[:60]], "new_vetoes": sorted(new),
                     "frames_changed_vs_reference": int(gf["changed"].sum())})
        log(f"gate round {rnd + 1}: flagged {hist[-1]['flagged_frames']}, new vetoes {sorted(new)}")
        if not new:
            break
        vetoes.update(new)
        CURATION.write_text(json.dumps(cur_raw, indent=1, ensure_ascii=False), encoding="utf-8")
    rep = {"what": "zouk_contacts post-hoc gate: NEW problems of the candidate vs the reference and the input (twist, elbow hyperextension / "
                   "off-hinge, wrist fold, elbow / wrist / palm / fingertip whips, wrist flicks) and per window the independent COCO wrist "
                   "(+ COCO elbow, MAMMA hand points, exact new penetration / floating in the final pass)", "gate": GATE, "tag": tag,
           "reference": ref, "windows_adopted": len(wins), "windows_vetoed": sorted(k for k in vetoes), "rounds": hist,
           "coco_per_window": coco_rep, "exact_per_window": exact_rep, "windows_kept": [w for w in wins if w["id"] not in vetoes]}
    write_json(OUT / ("gate.json" if not base_tag else f"gate_{tag}.json"), rep)
    return rep


# ------------------------------------------------------------------------------------------------ the last pass (before the one render)
#
# Owner (2026-10-22): the hand fixes come before the recap is re-recorded, and it is rendered once. What a viewer SEES matters now - no hand
# inside her body, no wrist folded back onto his forearm, no 1-3-frame spikes - with a small, bounded change of the installed motion
# (smplx_lead.bin ec8e7101..., kept as v4/smplx_contact_installed_ec8e). The earlier passes could not close these because the 2D guards (MAMMA's
# hand points, the COCO wrist / elbow of the phone that sees the hand) vetoed every move, and on 1-3-frame spikes and deep folds that 2D is
# itself unreliable. The last pass:
#   1. spikes - runs of at most 5 frames with his hand > 5 cm inside her, or a short fold past 120 deg: his right arm (shoulder, elbow,
#      wrist, fingers; the whole body where the pelvis / spine / root spike too, the legs keeping their global rotations) across the run
#      becomes a C2 cubic spline through the poses just outside it (4 frames each side, fewer where that makes it faster than its
#      neighbours); no 2D veto, the 2D change is reported;
#   2. frames still > 5 cm inside her (with the whole episode > 2 cm around them): the hand pushed out to <= 1 cm - per frame the smallest
#      rigid translation that brings it out of her (exit_translations: every direction searched, the exits of neighbouring frames one
#      steady path), then the arm solve (ArmWindow / solve_arm_window: the shoulder's rotation, the elbow on its hinge, the forearm's own
#      twist, the hand's orientation) against her exact surface re-linearised each round - inside the twist / hinge / bend limits, at
#      most 10 cm of hand movement, blended in and out over 0.6 s, with the speed audit's rules as limits (no new added-speed or fast
#      frame, no new wrist flick); the 2D guard reports and vetoes a window only when the COCO elbow, in the phones that see it, gets
#      more than 25 px worse;
#   3. wrists bent past 120 deg: unfolded to <= 100 deg over the run bent past 100 deg, moving the forearm as needed (same cap, blends and
#      limits), the palm kept where it touched her;
#   4. no new floating: a hand that touched her keeps the contact in the solve, floating back-hold frames in a window are pulled onto her;
#   then a post-hoc check against the installed motion: a window with a new gate flag (whip, flick, twist, hyperextension, fold) or new
#   penetration is scaled back (x 0.75 / 0.5 / 0.25 / 0), and while the film's elbow / fingertip added-speed frames or its floating frames
#   exceed the installed motion's, the windows adding most to them are scaled back.
#
#     python -m dancecap.zouk_contacts last-pass            # ec8e -> v4/smplx_contact (+ v4/contacts/last/report.json, exact_v4.npz)

FIT_EC8E = V4 / "smplx_contact_installed_ec8e"       # the closing pass installed 2026-10-22 (smplx_lead.bin ec8e7101...): the last pass's base
FITS["ec8e"] = FIT_EC8E
LAST_DIR = OUT / "last"
ARM_R = (SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55))     # his right arm: shoulder, elbow, wrist, the 15 finger joints


@dataclass
class LastCfg:
    spike_cm: float = 5.0            # a spike: a run of at most spike_max_frames frames with his hand deeper than this inside her ...
    spike_max_frames: int = 5
    spike_merge: int = 1             # ... (runs this few frames apart are one run)
    fold_deg: float = 120.0          # ... or with the wrist bent past this (a short fold)
    knots: int = 4                   # the C2 spline goes through this many frames on each side of a spike
    speed_slack: float = 1.1         # ... and must not move faster than this x its neighbours (else fewer knots)
    body_ratio: float = 1.5          # the whole body is despiked where the pelvis / spine-3 / root move this much faster inside the run than
    body_min_deg: float = 3.0        # within 12 frames around it and at least this fast (deg / frame; the root: body_min_mps)
    body_min_mps: float = 0.5
    push_from_cm: float = 5.0        # frames deeper than this after the despikes are pushed out ...
    push_to_cm: float = 1.0          # ... to at most this, with the whole episode deeper than push_run_cm around them
    push_run_cm: float = 2.0
    fold_to_deg: float = 100.0       # wrists bent past fold_deg are unfolded to at most this over their run bent past it
    pad: int = 3                     # a window's core: the frames to push / unfold +- this
    cap_m: float = 0.10              # the palm centre, the wrist and each fingertip move at most this much
    ramp: int = 18                   # the blend in / out (frames, raised cosine): 0.6 s
    merge_gap: int = 30              # cores closer than this (frames) are one window
    touch_m: float = 0.02            # the hand counts as touching her where its signed gap was below this
    elbow_guard_px: float = 25.0     # a window is vetoed when the COCO elbow (phones that see it) gets this much worse (median px)
    guard_min_n: int = 8
    rounds: int = 4                  # re-linearisations of her surface per window
    iters: int = 100                 # L-BFGS iterations per round
    device: str = "cuda"             # her surface queries (the solve itself runs on the CPU, float64)


def angular_speed(Rg: np.ndarray) -> np.ndarray:
    """(T,3,3) rotations -> (T,) degrees turned from the previous frame (0 at frame 0)"""
    from scipy.spatial.transform import Rotation as Ro
    r = Ro.from_matrix(np.asarray(Rg, float))
    return np.r_[0.0, np.degrees((r[:-1].inv() * r[1:]).magnitude())]


def spike_ranges(pen: np.ndarray, bend: np.ndarray, speeds: dict, cfg: LastCfg | None = None) -> list[dict]:
    """step 1's spikes: runs of at most spike_max_frames frames with the hand deeper than spike_cm inside her (pen, m) or the wrist bent past
    fold_deg (bend, deg), runs at most spike_merge frames apart joined (a joined run longer than spike_max_frames is not a spike: steps 2 / 3
    take it); kind "body" where the pelvis or spine-3 (deg / frame) or the root (m/s) - speeds {"pelvis", "spine3", "root"}, (T,) the speed
    into each frame - moves body_ratio x faster inside the run (its frames and the one after) than within 12 frames around it, and at least
    body_min_*; else "arm" -> [{"a", "b", "kind", "why", "body_why", "pen_cm", "bend_deg"}] sorted, apart"""
    cfg = cfg or LastCfg()
    T = len(pen)
    cand = []
    for m, why in ((np.asarray(pen) > cfg.spike_cm / 100, "deep"), (np.asarray(bend) > cfg.fold_deg, "fold")):
        cand += [[a, b, {why}] for a, b in _runs(m, cfg.spike_merge)]
    cand.sort(key=lambda c: c[0])
    merged = []
    for a, b, w in cand:
        if merged and a <= merged[-1][1] + cfg.spike_merge + 1:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2] |= w
        else:
            merged.append([a, b, set(w)])
    out = []
    for a, b, w in merged:
        if b - a + 1 > cfg.spike_max_frames:
            continue
        nb = np.r_[max(0, a - 12):max(0, a - 1), min(T, b + 2):min(T, b + 13)].astype(int)
        inside = np.arange(a, min(T, b + 2))
        body = []
        for k, floor in (("pelvis", cfg.body_min_deg), ("spine3", cfg.body_min_deg), ("root", cfg.body_min_mps)):
            s = np.asarray(speeds[k], float)
            vi, vn = float(s[inside].max()), (float(s[nb].max()) if len(nb) else 0.0)
            if vi > floor and vi > cfg.body_ratio * vn:
                body.append(k)
        out.append({"a": int(a), "b": int(b), "t": [round(a / HZ, 2), round(b / HZ, 2)], "kind": "body" if body else "arm",
                    "why": sorted(w), "body_why": body, "pen_cm": round(float(np.max(pen[a:b + 1])) * 100, 1),
                    "bend_deg": round(float(np.max(bend[a:b + 1])), 1)})
    return out


def spike_knots(a: int, b: int, T: int, n: int, blocked: np.ndarray | None = None) -> np.ndarray | None:
    """the n frames on each side of the run [a, b] that the spline goes through (not inside another run: blocked), or None when the frames
    just outside it are not free"""
    blocked = np.zeros(T, bool) if blocked is None else blocked
    left = [k for k in range(a - 1, max(-1, a - 1 - 3 * n), -1) if not blocked[k]][:n][::-1]
    right = [k for k in range(b + 1, min(T, b + 1 + 3 * n)) if not blocked[k]][:n]
    if len(left) < 2 or len(right) < 2 or left[-1] != a - 1 or right[0] != b + 1:
        return None
    return np.array(left + right)


def despike_c2(P: np.ndarray, tr: np.ndarray, a: int, b: int, K: np.ndarray, kind: str = "arm") -> tuple[np.ndarray, np.ndarray]:
    """one spike [a, b]: the rotations (kind "arm": his right shoulder, elbow, wrist + fingers; "body": every joint and the root translation,
    the legs keeping their global rotations as in despike_hermite) on the run become a C2 cubic spline (scipy CubicSpline, not-a-knot)
    through the knot frames K (outside the run, unchanged), in the tangent space of the frame before the run -> (P copy, transl copy)"""
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation as Ro
    P, tr = np.asarray(P, float).copy(), np.asarray(tr, float).copy()
    P0 = P.copy()
    x = np.arange(a, b + 1)
    K = np.asarray(K)
    joints = range(P.shape[1]) if kind == "body" else ARM_R
    for j in joints:
        R0 = Ro.from_rotvec(P0[a - 1, j])
        loc = (R0.inv() * Ro.from_rotvec(P0[K, j])).as_rotvec()
        P[x, j] = (R0 * Ro.from_rotvec(CubicSpline(K, loc, axis=0)(x))).as_rotvec()
    if kind == "body":
        Rp_new, Rp_old = Ro.from_rotvec(P[x, PELVIS]), Ro.from_rotvec(P0[x, PELVIS])
        for s_ in SIDES:
            P[x, HIP[s_]] = (Rp_new.inv() * Rp_old * Ro.from_rotvec(P0[x, HIP[s_]])).as_rotvec()
            for jj in (KNEE[s_], ANKLE[s_], FOOT[s_]):
                P[x, jj] = P0[x, jj]
        tr[x] = CubicSpline(K, tr[K], axis=0)(x)
    return P, tr


def _arm_points(betas: np.ndarray) -> dict:
    """his right hand (surface + palm patch) and forearm points (hand_contact.rest_hand_ids, the exact check's points) with their skinning
    weights on the shoulder, the elbow and the hand (wrist + fingers)"""
    ids = hc.rest_hand_ids(betas)["right"]
    pts = np.concatenate([ids.hand, ids.palm, ids.arm])
    W = hc.skin_weights()[pts]
    nh, npm = len(ids.hand), len(ids.palm)
    return {"ids": pts, "nh": nh, "npm": npm, "w17": W[:, SHOULDER["right"]], "w19": W[:, ELBOW["right"]],
            "wh": W[:, [WRIST["right"], *range(40, 55)]].sum(1)}


def _surface_planes(surf, X: np.ndarray, VF: np.ndarray, device: str, bs: int = 12) -> dict:
    """signed distance, closest point and outward normal of her posed surface (VF (W,nv,3)) at points X (W,n,3): exact (winding numbers)"""
    import torch
    W, n = X.shape[:2]
    sd, q, nq = np.zeros((W, n)), np.zeros((W, n, 3)), np.zeros((W, n, 3))
    with torch.no_grad():
        for a in range(0, W, bs):
            r = surf.query(torch.as_tensor(X[a:a + bs], dtype=torch.float32, device=device),
                           torch.as_tensor(VF[a:a + bs], dtype=torch.float32, device=device), wn_within=0.03)
            sd[a:a + bs] = r["sd"].cpu().numpy()
            q[a:a + bs] = r["q"].cpu().numpy()
            nq[a:a + bs] = r["nq"].cpu().numpy()
    return {"sd": sd, "q": q, "n": nq}


def sphere_dirs(n: int = 48) -> np.ndarray:
    """(n,3) unit directions spread evenly over the sphere (Fibonacci)"""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)


def pick_exit(t: np.ndarray, dirs: np.ndarray, depth_at_max: np.ndarray, tmax: float, w_jump: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """per frame of a run, from t (F, D) - the smallest translation along each direction that brings the hand out (inf: none within tmax) -
    the exits as one path through time (dynamic programming): each frame's cost is its translation's length (a frame with no exit: tmax +
    10 x the depth left at tmax, depth_at_max (F, D)), each step's cost w_jump x how far the translation jumps between the two frames (the
    exit direction deep in her swings 90 deg between neighbouring frames: following the shortest exit frame by frame whips the hand)
    -> (delta (F,3), ok (F,) an exit within tmax)"""
    t = np.asarray(t, float)
    F, D = t.shape
    ok_fd = np.isfinite(t)
    tl = np.where(ok_fd, t, tmax)
    unary = np.where(ok_fd, t, tmax + 10.0 * np.asarray(depth_at_max, float))
    vec = tl[:, :, None] * dirs[None]                                   # (F, D, 3)
    cost = unary[0].copy()
    bp = np.zeros((F, D), np.int64)
    for i in range(1, F):
        jump = np.linalg.norm(vec[i - 1][:, None] - vec[i][None], axis=-1)   # (D from, D to)
        c = cost[:, None] + w_jump * jump
        bp[i] = c.argmin(0)
        cost = c.min(0) + unary[i]
    path = np.zeros(F, np.int64)
    path[-1] = int(cost.argmin())
    for i in range(F - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    ar = np.arange(F)
    return vec[ar, path], ok_fd[ar, path]


def exit_translations(X: np.ndarray, VF: np.ndarray, surf, device: str, tmax: float = 0.095, depth_ok: float = 0.008,
                      n_dirs: int = 48, steps: int = 7, bs: int = 64, max_up: float = 0.35) -> dict:
    """step 2's push: per frame the smallest rigid translation of his hand points X (F,n,3) that leaves no point deeper than depth_ok inside
    her (VF (F,nv,3)), over n_dirs directions: a bisection (`steps` halvings of [0, tmax]) with the fast inside test (the side of the closest
    point's normal), the chosen exit (pick_exit) then checked exactly (winding numbers) and lengthened 5 mm at a time until it holds (or the
    cap) -> {"delta" (F,3), "ok" (F,), "t" (F,), "depth_left" (F,) exact, at the chosen translation}; the exits of neighbouring frames
    form one steady path (pick_exit), and X should be one run of consecutive frames. Directions rising more steeply than max_up (the
    world's +y component) are not searched: the shortest way out of her upper back is often over her shoulder, and the review of the
    first run saw the hand pushed up onto the back of her neck (289.0 s, cam 01)"""
    import torch
    F, n = X.shape[:2]
    D = sphere_dirs(n_dirs)
    D = D[D[:, 1] <= max_up]
    n_dirs = len(D)
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    Vt = torch.as_tensor(np.asarray(VF, np.float32), device=device)
    Dt = torch.as_tensor(D, dtype=torch.float32, device=device)
    fi, di = np.meshgrid(np.arange(F), np.arange(n_dirs), indexing="ij")
    fi, di = fi.ravel(), di.ravel()

    def depth(tvals: np.ndarray, f_idx: np.ndarray, dvec: torch.Tensor, exact: bool) -> np.ndarray:
        out = np.zeros(len(f_idx))
        tv = torch.as_tensor(np.asarray(tvals, np.float32), device=device)
        with torch.no_grad():
            for a in range(0, len(f_idx), bs):
                f_ = torch.as_tensor(f_idx[a:a + bs], device=device)
                P = Xt[f_] + (tv[a:a + bs, None] * dvec[a:a + bs])[:, None, :]
                q = surf.query(P, Vt[f_], wn_within=0.03, inside_test=exact)
                out[a:a + bs] = np.maximum(0, -q["sd"].min(1).values.cpu().numpy())
        return out
    Dall = Dt[torch.as_tensor(di, device=device)]
    hi = np.full(len(fi), tmax)
    d_hi = depth(hi, fi, Dall, False)
    ok = d_hi <= depth_ok
    lo = np.zeros(len(fi))
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        dm = depth(np.where(ok, mid, tmax), fi, Dall, False)
        good = ok & (dm <= depth_ok)
        hi = np.where(good, mid, hi)
        lo = np.where(ok & ~good, mid, lo)
    t = np.where(ok, hi, np.inf).reshape(F, n_dirs)
    delta, okf = pick_exit(t, D, d_hi.reshape(F, n_dirs), tmax)
    tt = np.linalg.norm(delta, axis=1)
    u = delta / np.maximum(tt, 1e-9)[:, None]
    ut = torch.as_tensor(u, dtype=torch.float32, device=device)
    left = depth(tt, np.arange(F), ut, True)
    for _ in range(int(np.ceil(tmax / 0.005))):                        # the exact check: lengthen the exit until it holds
        more = (left > depth_ok + 0.001) & (tt < tmax - 1e-6)
        if not more.any():
            break
        tt = np.where(more, np.minimum(tt + 0.005, tmax), tt)
        left = np.where(more, depth(tt, np.arange(F), ut, True), left)
    okf = left <= depth_ok + 0.001
    return {"delta": u * tt[:, None], "ok": okf, "t": tt, "depth_left": left}


def envelope(n_left: int, n_core: int, n_right: int) -> np.ndarray:
    """a window's blend: raised-cosine ramps (0 -> 1 over n_left frames, 1 -> 0 over n_right), 1 on the core"""
    up = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n_left + 1) / (n_left + 1))
    dn = 0.5 + 0.5 * np.cos(np.pi * np.arange(1, n_right + 1) / (n_right + 1))
    return np.r_[up, np.ones(n_core), dn]


def _skew_t(torch, v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1), torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


class ArmWindow:
    """one window of the last pass: his right arm re-posed by 8 numbers per frame - a world rotation of the arm about the shoulder (3), the
    elbow's flexion about its hinge (1, the off-hinge part kept), the forearm's own twist about its bone (1, pronation / supination: no joint
    moves), a world rotation of the hand (3) - applied to the base motion's frames
    ks; the hand / forearm points follow by linear blend skinning (the base's posed points moved by the changed joints' transforms). The
    numbers are free on every frame and multiplied by the window's envelope (1 on the core, raised-cosine ramps to exactly 0 at the
    window's ends: the change blends in and out, and the ramps' own limits - bend, twist, speed - hold there too)."""

    NV = 8

    def __init__(self, fit: dict, P: np.ndarray, ks: np.ndarray, n_left: int, n_right: int, rest_J: np.ndarray, pts: dict):
        import torch
        self.torch = torch
        self.ks = np.asarray(ks)
        W = len(ks)
        self.nl, self.nr, self.nc = n_left, n_right, W - n_left - n_right
        lead = fit["lead"]
        o = posed(lead, self.ks, P[self.ks])
        J, R = o["J"], o["R"]
        t64 = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
        self.G14, self.G17, self.G19, self.G21 = (t64(R[:, j]) for j in (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"]))
        self.S, self.E, self.Wr = t64(J[:, SHOULDER["right"]]), t64(J[:, ELBOW["right"]]), t64(J[:, WRIST["right"]])
        self.R19 = t64(rotvec_to_mat(P[self.ks, ELBOW["right"]]))
        self.Pw = np.asarray(P[self.ks], float).copy()
        self.b1 = t64(rest_J[ELBOW["right"]] - rest_J[SHOULDER["right"]])
        self.b2 = t64(rest_J[WRIST["right"]] - rest_J[ELBOW["right"]])
        V = hc.Body(lead, "cpu").forward_np(P[self.ks], np.asarray(lead["transl"], float)[self.ks], ids=pts["ids"])["verts"].astype(float)
        self.V0 = V
        Vt = t64(V)
        self.A17 = torch.einsum("wji,wnj->wni", self.G17, Vt - self.S[:, None])
        self.A19 = torch.einsum("wji,wnj->wni", self.G19, Vt - self.E[:, None])
        self.A21 = torch.einsum("wji,wnj->wni", self.G21, Vt - self.Wr[:, None])
        self.w17, self.w19, self.wh = (t64(pts[k])[None, :, None] for k in ("w17", "w19", "wh"))
        self.Vt = Vt
        C = hc.palm_centre(J, "right")
        tips = o["tips"][:, 5:10]
        self.C0, self.tips0 = t64(C), t64(tips)
        self.Cl = torch.einsum("wji,wj->wi", self.G21, self.C0 - self.Wr)
        self.Tl = torch.einsum("wji,wnj->wni", self.G21, self.tips0 - self.Wr[:, None])
        self.env = t64(envelope(n_left, self.nc, n_right))
        self.ey = t64([0.0, 1.0, 0.0])
        self.ea = self.b2 / self.b2.norm()                                  # the forearm bone's axis in the elbow's frame

    def set_input(self, fit_in_lead: dict, P_in: np.ndarray, p99: dict) -> None:
        """the INPUT motion at this window's frames (the speed audit's reference): its elbow, wrist, palm centre, fingertips, the wrist's
        local rotation, and p99 {point: m/s} of each point's speed over the film"""
        torch = self.torch
        o = posed(fit_in_lead, self.ks, P_in[self.ks])
        t64 = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
        self.X_in = {"E": t64(o["J"][:, ELBOW["right"]]), "W": t64(o["J"][:, WRIST["right"]]), "C": t64(hc.palm_centre(o["J"], "right")),
                     "tips": t64(o["tips"][:, 5:10])}
        self.R21_in = t64(rotvec_to_mat(P_in[self.ks, WRIST["right"]]))
        self.p99 = dict(p99)

    def params(self, v):
        """the free numbers (W, NV) -> every frame's change: times the envelope"""
        return v * self.env[:, None]

    def forward(self, v) -> dict:
        torch = self.torch
        p = self.params(v)
        ds, th, te, dh = p[:, 0:3], p[:, 3], p[:, 4], p[:, 5:8]
        Q17 = torch.linalg.matrix_exp(_skew_t(torch, ds))
        G17 = Q17 @ self.G17
        Ry = torch.linalg.matrix_exp(_skew_t(torch, th[:, None] * self.ey))
        R19 = Ry @ self.R19 @ torch.linalg.matrix_exp(_skew_t(torch, te[:, None] * self.ea))
        G19 = G17 @ R19
        G21 = torch.linalg.matrix_exp(_skew_t(torch, dh)) @ self.G21
        E = self.S + torch.einsum("wij,j->wi", G17, self.b1)
        Wr = E + torch.einsum("wij,j->wi", G19, self.b2)
        X17 = torch.einsum("wij,wnj->wni", G17, self.A17) + self.S[:, None]
        X19 = torch.einsum("wij,wnj->wni", G19, self.A19) + E[:, None]
        X21 = torch.einsum("wij,wnj->wni", G21, self.A21) + Wr[:, None]
        rest = 1.0 - self.w17 - self.w19 - self.wh
        X = rest * self.Vt + self.w17 * X17 + self.w19 * X19 + self.wh * X21
        C = torch.einsum("wij,wj->wi", G21, self.Cl) + Wr
        tips = torch.einsum("wij,wnj->wni", G21, self.Tl) + Wr[:, None]
        R21 = G19.transpose(1, 2) @ G21
        return {"p": p, "G17": G17, "R19": R19, "G19": G19, "G21": G21, "R21": R21, "E": E, "W": Wr, "X": X, "C": C, "tips": tips}

    def poses(self, v, P: np.ndarray) -> np.ndarray:
        """the base pose P (T,55,3) with this window's frames re-posed by the numbers v -> a copy"""
        from scipy.spatial.transform import Rotation as Ro
        with self.torch.no_grad():
            f = self.forward(v)
        P = P.copy()
        G14 = self.G14.numpy()
        P[self.ks, SHOULDER["right"]] = Ro.from_matrix(np.einsum("wji,wjk->wik", G14, f["G17"].numpy())).as_rotvec()
        P[self.ks, ELBOW["right"]] = Ro.from_matrix(f["R19"].numpy()).as_rotvec()
        P[self.ks, WRIST["right"]] = Ro.from_matrix(f["R21"].numpy()).as_rotvec()
        return P


def solve_arm_window(win: ArmWindow, VF: np.ndarray, surf, flags: dict, cfg: LastCfg, pts: dict, log_to: list | None = None):
    """the last pass's solve of one window (steps 2 and 3 together): per frame flags (W,) bool - "push" (the hand to <= push_to_cm inside
    her), "unfold" (the wrist bend to <= fold_to_deg), "touch" (the hand touched her: it keeps the contact; an unfolded one keeps its palm
    centre) - with the smallest arm change (elbow / wrist / palm movement, the hand's rotation), the bend / twist / elbow limits never worse
    than the base, at most cap_m of palm / wrist / fingertip movement and a smooth, bounded change frame to frame. Her surface (exact closest
    points + normals, _surface_planes) is re-linearised `rounds` times -> (core numbers v, report)"""
    torch = win.torch
    nh, npm = pts["nh"], pts["npm"]
    nhand = nh + npm
    W = len(win.ks)
    f64 = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    push, unfold, touch = (f64(flags[k].astype(float)) for k in ("push", "unfold", "touch"))
    pull = f64(flags.get("pull", np.zeros(W, bool)).astype(float))
    with torch.no_grad():
        f0 = win.forward(torch.zeros((W, win.NV), dtype=torch.float64))
        b0, _, _, tw0 = wrist_terms_t(torch, f0["R21"], "right", with_twist=True)
        fl0, _ = elbow_terms_t(torch, f0["R19"], "right")
    pl0 = _surface_planes(surf, win.V0, VF, cfg.device)
    sd_orig = f64(pl0["sd"])
    deg = np.pi / 180
    # per frame the deepest hand point (a soft minimum over the points, tau 2 mm): push frames to push_to_cm (3 mm margin), the others never
    # deeper than the base (or 1 cm); the forearm never deeper than the base (or 1 cm)
    tau = 0.002
    smin_h0 = -tau * torch.logsumexp(-sd_orig[:, :nhand] / tau, 1)
    smin_a0 = -tau * torch.logsumexp(-sd_orig[:, nhand:] / tau, 1)
    lo_h = torch.where(push > 0, torch.full_like(smin_h0, -(cfg.push_to_cm / 100 - 0.003)),
                       torch.minimum(smin_h0, torch.full_like(smin_h0, -0.01)) - 0.002)
    lo_a = torch.minimum(smin_a0, torch.full_like(smin_a0, -0.01)) - 0.002
    # an unfolded wrist keeps its palm centre where the hand touched her (and was not deep inside her)
    keep = touch * unfold * (smin_h0 > -0.015).double()
    bend_hi = torch.where(unfold > 0, torch.full_like(b0, (cfg.fold_to_deg - 3) * deg), torch.maximum(b0, torch.full_like(b0, 85 * deg)))
    # the twist never beyond the base's (or 90 deg); on a wrist folded past 150 deg the base's twist is meaningless: 90 deg there
    tw_hi = torch.where(b0 < 150 * deg, torch.maximum(tw0.abs(), torch.full_like(tw0, 85 * deg)), torch.full_like(tw0, 85 * deg))
    fl_lo = torch.minimum(fl0, torch.full_like(fl0, -3 * deg))
    # the speed audit's rules (zouk_contacts.gate_flags, hand_contact.speed_audit) as limits, per point (elbow, wrist, palm, fingertips) and
    # step: the motion's change against the INPUT moves at most max(0.5 m/s, the base's) - no new added-speed frame; the point itself at most
    # max(the input's p99, the input there, the base there) + 0.2 m/s - no new fast frame; the wrist's local rotation turns at most
    # max(the input's, the base's) + 6 deg / frame - no new flick
    lim = {}
    if getattr(win, "X_in", None) is not None:
        for k in ("E", "W", "C", "tips"):
            xin, xb = win.X_in[k], f0[k]
            d0 = ((xb - xin)[1:] - (xb - xin)[:-1]).norm(dim=-1)
            vin, vb = (xin[1:] - xin[:-1]).norm(dim=-1), (xb[1:] - xb[:-1]).norm(dim=-1)
            p99 = win.p99[k] / HZ
            lim[k] = (torch.clamp(d0 + 0.001, min=0.5 / HZ), torch.maximum(torch.maximum(vin, vb), torch.full_like(vin, p99)) + 0.2 / HZ)

        def rot_step(R):
            c = ((R[:-1].transpose(1, 2) @ R[1:]).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
            return c.clamp(-1, 1)
        c_in, c_b = rot_step(win.R21_in), rot_step(f0["R21"])
        a_lim = torch.maximum(torch.acos(c_in), torch.acos(c_b)) + 6 * deg
    fl_hi = torch.maximum(fl0, torch.full_like(fl0, 150 * deg))
    # the start: zero, except that folded wrists start unfolded along a path continuous in time (unfold_wrist: the wrist's local rotation
    # compressed above fold_to_deg - 15 deg) - at a bend near 180 deg the direction to unfold is ambiguous frame by frame (a free start spun
    # the hand 270 deg about the forearm in 0.2 s at 273.6 s)
    v0 = np.zeros((W, win.NV))
    if flags["unfold"].any():
        from scipy.spatial.transform import Rotation as Ro
        core = slice(0, W)
        Pu, _ = unfold_wrist(win.Pw[core], knee_deg=cfg.fold_to_deg - 15, slope=0.1)
        G19c, G21c = win.G19.numpy()[core], win.G21.numpy()[core]
        G21u = np.einsum("wij,wjk->wik", G19c, rotvec_to_mat(Pu[:, WRIST["right"]]))
        v0[:, 5:8] = Ro.from_matrix(np.einsum("wij,wkj->wik", G21u, G21c)).as_rotvec()
    v = torch.tensor(v0, dtype=torch.float64, requires_grad=True)
    # push frames: the smallest rigid translation of the hand that brings it out (exit_translations, every direction searched: deep in her,
    # the closest-point normals of the hand's points point every way and one plane per point pulls the hand sideways) - the palm's target,
    # and the hand's points there are where her surface is first linearised
    X_lin, pl = win.V0, pl0
    tgt = torch.zeros((W, 3), dtype=torch.float64)
    has_tgt = torch.zeros(W, dtype=torch.float64)
    exit_rep = None
    pi_ = np.flatnonzero(flags["push"])
    if len(pi_):
        X_lin = win.V0.copy()
        ts, found = [], 0
        for a_, b_ in _runs(flags["push"], 0):                          # one steady exit path per run of push frames
            x = np.arange(a_, b_ + 1)
            ex = exit_translations(win.V0[x, :nhand:2], VF[x], surf, cfg.device, tmax=cfg.cap_m - 0.005,
                                   depth_ok=cfg.push_to_cm / 100 - 0.002)
            X_lin[x, :nhand] += ex["delta"][:, None]
            tgt[x] = f64(ex["delta"])
            ts.append(ex["t"])
            found += int(ex["ok"].sum())
        pl = _surface_planes(surf, X_lin, VF, cfg.device)
        has_tgt[pi_] = 1.0
        ts = np.concatenate(ts)
        exit_rep = {"frames": int(len(pi_)), "found": found, "t_cm_max": round(float(ts.max()) * 100, 2),
                    "t_cm_median": round(float(np.median(ts)) * 100, 2)}
    hist = []
    relu = torch.relu
    cap = cfg.cap_m - 0.005

    def d2(x):
        return x[2:] - 2 * x[1:-1] + x[:-2]
    for rnd in range(cfg.rounds):
        sd_lin, nq = f64(pl["sd"]), f64(pl["n"])
        Xl = f64(X_lin)
        far = (sd_lin >= 0.05).double()                  # the plane of a point far off her says nothing: left out of the minimum

        def loss_fn():
            f = win.forward(v)
            sdp = sd_lin + ((f["X"] - Xl) * nq).sum(-1) + far
            smin = -tau * torch.logsumexp(-sdp[:, :nhand] / tau, 1)
            smin_a = -tau * torch.logsumexp(-sdp[:, nhand:] / tau, 1)
            L_pen = 3e5 * ((relu(lo_h - smin) ** 2).sum() + (relu(lo_a - smin_a) ** 2).sum())
            # the contact of a touching hand
            # (a touching hand stays within 1.5 cm of her: the gate counts > 2 cm off her as floating; a floating one in a back hold is
            # pulled onto her, more gently)
            L_touch = 1e6 * (touch * relu(smin - 0.015) ** 2).sum() + 3e4 * (pull * relu(smin - 0.015) ** 2).sum()
            dE, dW, dC = f["E"] - f0["E"], f["W"] - f0["W"], f["C"] - f0["C"]
            dT = f["tips"] - f0["tips"]
            L_min = 50 * ((dE ** 2).sum() + (dW ** 2).sum() + (dC ** 2).sum()) + 2.0 * (f["p"][:, 4:8] ** 2).sum() + 0.5 * (f["p"][:, :3] ** 2).sum()
            L_keep = 1000 * (keep * (dC ** 2).sum(-1)).sum()
            L_tgt = 2000 * (has_tgt * ((dC - tgt) ** 2).sum(-1)).sum()
            mv = torch.cat([dW.norm(dim=-1)[:, None], dC.norm(dim=-1)[:, None], dT.norm(dim=-1)], 1)
            L_cap = 1e8 * (relu(mv - cap) ** 2).sum()
            bend, _, _, tw = wrist_terms_t(torch, f["R21"], "right", with_twist=True)
            L_bend = 3e4 * (relu(bend - bend_hi) ** 2).sum()
            tw_ok = (bend.detach() < 150 * deg).double()                 # the twist is defined only below a full fold
            L_tw = 3e4 * (tw_ok * relu(tw.abs() - tw_hi) ** 2).sum()
            fl, _ = elbow_terms_t(torch, f["R19"], "right")
            L_el = 300 * ((relu(fl_lo - fl) ** 2) + relu(fl - fl_hi) ** 2).sum()
            L_sm = 3e4 * sum((d2(x) ** 2).sum() for x in (dE, dW, dC)) + 300 * (d2(f["p"]) ** 2).sum()
            # a bounded change frame to frame (no whip): the added movement of the elbow, the wrist, the palm, the fingertips <= 12 mm / frame
            # (0.36 m/s; the speed audit's limit is 0.55)
            L_v = 1e6 * sum((relu((x[1:] - x[:-1]).norm(dim=-1) - 0.012) ** 2).sum() for x in (dE, dW, dC, dT))
            L_sp = 0.0
            if lim:
                for k, x in (("E", f["E"]), ("W", f["W"]), ("C", f["C"]), ("tips", f["tips"])):
                    xin = win.X_in[k]
                    dd = ((x - xin)[1:] - (x - xin)[:-1]).norm(dim=-1)
                    vv = (x[1:] - x[:-1]).norm(dim=-1)
                    L_sp = L_sp + 1e8 * ((relu(dd - lim[k][0]) ** 2).sum() + (relu(vv - lim[k][1]) ** 2).sum())
                L_sp = L_sp + 1e6 * (relu(torch.acos(rot_step(f["R21"]).clamp(-1 + 1e-7, 1 - 1e-7)) - a_lim) ** 2).sum()
            return L_sp + L_pen + L_touch + L_min + L_keep + L_tgt + L_cap + L_bend + L_tw + L_el + L_sm + L_v
        opt = torch.optim.LBFGS([v], lr=1.0, max_iter=cfg.iters, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            L = loss_fn()
            L.backward()
            return L
        opt.step(closure)
        with torch.no_grad():
            f = win.forward(v)
            X_lin = f["X"].numpy()
        pl = _surface_planes(surf, X_lin, VF, cfg.device)
        hand_depth = np.maximum(0, -pl["sd"][:, :nhand].min(1))
        hist.append({"round": rnd + 1, "loss": round(float(loss_fn().detach()), 4),
                     "push_frames_depth_cm_max": round(float(hand_depth[flags["push"]].max() * 100), 2) if flags["push"].any() else None})
    with torch.no_grad():
        f = win.forward(v)
        bend, _, _, tw = wrist_terms_t(torch, f["R21"], "right", with_twist=True)
        mv = torch.cat([(f["W"] - f0["W"]).norm(dim=-1)[:, None], (f["C"] - f0["C"]).norm(dim=-1)[:, None],
                        (f["tips"] - f0["tips"]).norm(dim=-1)], 1).max(1).values
    hand_depth0 = np.maximum(0, -pl0["sd"][:, :nhand].min(1))
    hand_depth = np.maximum(0, -pl["sd"][:, :nhand].min(1))
    rep = {"rounds": hist, "frames": W, "core": [int(win.ks[win.nl]), int(win.ks[win.nl + win.nc - 1])],
           "hand_move_cm_max": round(float(mv.max()) * 100, 2),
           "elbow_move_cm_max": round(float((f["E"] - f0["E"]).norm(dim=-1).max()) * 100, 2),
           "depth_cm_max_before_after": [round(float(hand_depth0.max()) * 100, 2), round(float(hand_depth.max()) * 100, 2)],
           "bend_deg_max_before_after": [round(float(b0.max() / deg), 1), round(float(bend.max() / deg), 1)],
           "push_frames": int(flags["push"].sum()), "unfold_frames": int(flags["unfold"].sum()), "touch_frames": int(flags["touch"].sum()),
           "pull_frames": int(flags.get("pull", np.zeros(W, bool)).sum()),
           "exit": exit_rep}
    if flags["push"].any():
        rep["push_frames_depth_cm_max_before_after"] = [round(float(hand_depth0[flags["push"]].max()) * 100, 2),
                                                        round(float(hand_depth[flags["push"]].max()) * 100, 2)]
    if flags["unfold"].any():
        rep["unfold_frames_bend_max_before_after"] = [round(float(b0[flags["unfold"]].max() / deg), 1),
                                                      round(float(bend[torch.as_tensor(flags["unfold"])].max() / deg), 1)]
    if log_to is not None:
        log_to.append(rep)
    return v.detach(), rep


def last_windows(pen: np.ndarray, gap: np.ndarray, bend: np.ndarray, cfg: LastCfg | None = None, back: np.ndarray | None = None) -> list[dict]:
    """steps 2 / 3: the windows of the despiked motion - push frames (deeper than push_from_cm), unfold frames (the runs bent past fold_to_deg
    that reach fold_deg); cores = those frames +- pad, merged when closer than merge_gap; ramps of `ramp` frames each side
    -> [{"core": [c0, c1], "span": [s0, s1], "push", "unfold", "touch", "pull": frame lists (absolute)}] (push: the episode deeper than
    push_run_cm around each frame deeper than push_from_cm; touch: the hand's signed gap (gap: the installed motion's) below touch_m;
    pull: the back holds (back (T,) bool) where the hand floated 2-5 cm off her)"""
    cfg = cfg or LastCfg()
    T = len(pen)
    pen = np.asarray(pen)
    # push frames: the whole episode deeper than push_run_cm around a frame deeper than push_from_cm (leaving the neighbours of a pushed
    # frame 3-4 cm inside her would need the hand to jump out and back in within a frame or two)
    push = np.zeros(T, bool)
    for a, b in _runs(pen > cfg.push_run_cm / 100, 2):
        if (pen[a:b + 1] > cfg.push_from_cm / 100).any():
            push[a:b + 1] = True
    unf = np.zeros(T, bool)
    for a, b in _runs(np.asarray(bend) > cfg.fold_to_deg, 0):
        if (np.asarray(bend)[a:b + 1] > cfg.fold_deg).any():
            unf[a:b + 1] = True
    need = push | unf
    core = np.zeros(T, bool)
    for a, b in _runs(need, 0):
        core[max(0, a - cfg.pad):min(T, b + cfg.pad + 1)] = True
    out = []
    for c0, c1 in _runs(core, cfg.merge_gap):
        s0, s1 = max(0, c0 - cfg.ramp), min(T - 1, c1 + cfg.ramp)
        x = np.arange(s0, s1 + 1)
        out.append({"core": [int(c0), int(c1)], "span": [int(s0), int(s1)], "t": [round(c0 / HZ, 2), round(c1 / HZ, 2)],
                    "push": x[push[x] & (x >= c0) & (x <= c1)].tolist(), "unfold": x[unf[x] & (x >= c0) & (x <= c1)].tolist(),
                    "touch": x[np.asarray(gap)[x] < cfg.touch_m].tolist(),
                    "pull": (x[(np.asarray(gap)[x] >= cfg.touch_m) & (np.asarray(gap)[x] <= 0.05) & np.asarray(back)[x]].tolist()
                             if back is not None else [])})
    return out


def _elbow_guard(fits: dict, ks: np.ndarray, obs, judge, bias: dict, bodies: dict, hcfg: HoldCfg) -> dict:
    """the 2D of a window, base ("before") against the candidate ("after"): the COCO elbow and wrist (median px over the phone-frames that
    see the candidate's joint and have a detection, the same detections for both) and MAMMA's hand points (phones that see the palm)"""
    out = {}
    kl, kf = posed(fits["after"]["lead"], ks), posed(fits["after"]["follow"], ks)

    def med(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return (round(float(np.median(a)), 2) if len(a) else None), int(len(a))

    def joint_x(fit, which):
        J = posed(fit["lead"], ks)["J"]
        jj = WRIST["right"] if which == "wrist" else ELBOW["right"]
        b = np.asarray(bias.get(f"lead_right_{which}", [0, 0, 0]), float)
        return J[:, jj] + np.einsum("kij,j->ki", hc._forearm_frame(J, "right"), b), J[:, jj]
    for which in ("elbow", "wrist"):
        jj = WRIST["right"] if which == "wrist" else ELBOW["right"]
        vis = joint_visible(obs, ks, kl["J"][:, jj], bodies, kl, kf)
        Xl = kl["J"][:, WRIST["left"] if which == "wrist" else ELBOW["left"]]
        (X0, J0), (X1, J1) = joint_x(fits["before"], which), joint_x(fits["after"], which)
        e0, n0 = med(coco_errors_vis(judge, obs, X0, ks, which, vis, Xl, X1))
        e1, n1 = med(coco_errors_vis(judge, obs, X1, ks, which, vis, Xl, X0))
        mv = []
        for ci, c in enumerate(obs.cams):
            u0, _ = obs.project(c, J0[:, None], ks)
            u1, _ = obs.project(c, J1[:, None], ks)
            d = np.linalg.norm(u1[:, 0] - u0[:, 0], axis=-1)
            mv += list(d[vis[:, ci]])
        out[which] = {"before_px": e0, "after_px": e1, "n": min(n0, n1),
                      "moved_px_visible_max": round(float(np.max(mv)), 1) if mv else None}
    vp = joint_visible(obs, ks, hand_site(kl["J"], "right", "palm")[0], bodies, kl, kf)
    (m0, k0), (m1, k1) = (med(hand_2d_errors_vis(fits[f], ks, obs, hcfg, vp, [fits["before"], fits["after"]])) for f in ("before", "after"))
    out["mamma"] = {"before_px": m0, "after_px": m1, "n": min(k0, k1)}
    return out


def _with_pose(fit: dict, P: np.ndarray, tr: np.ndarray | None = None) -> dict:
    """the fit with the leader's pose (and root translation) replaced, in memory"""
    T = len(P)
    d = dict(fit["lead"])
    d["global_orient"] = P[:, 0].copy()
    d["body_pose"] = P[:, 1:22].reshape(T, 63).copy()
    d["left_hand_pose"] = P[:, 25:40].reshape(T, 45).copy()
    d["right_hand_pose"] = P[:, 40:55].reshape(T, 45).copy()
    if tr is not None:
        d["transl"] = np.asarray(tr, float).copy()
    return {"lead": d, "follow": fit["follow"]}


def _segment_speeds(P: np.ndarray, fit_lead: dict, ks: np.ndarray) -> dict:
    """(len(ks)-1,) m/s of his right elbow, wrist, palm centre and the fastest fingertip between consecutive frames ks"""
    o = posed(fit_lead, ks, P[ks])
    pts = {"elbow": o["J"][:, ELBOW["right"]], "wrist": o["J"][:, WRIST["right"]], "palm": hc.palm_centre(o["J"], "right")}
    tips = o["tips"][:, 5:10]
    out = {k: np.linalg.norm(np.diff(x, axis=0), axis=-1) * HZ for k, x in pts.items()}
    out["tip"] = np.linalg.norm(np.diff(tips, axis=0), axis=-1).max(1) * HZ
    return out


def last_sheets(items: list, fit_names=("ec8e", "v4"), stem: str = "last", per_sheet: int = 5, tile: int = 236) -> list:
    """review sheets of the last pass: per item ({"t": s, "what": str}) 2 phones (one that sees her back, the widest other), each the raw
    crop and the meshes of every fit (his right hand + forearm yellow, the hand outlined), the exact depth / gap and the wrist bend of each
    fit in the side panel -> v4/contacts/last/<stem>_<i>.jpg"""
    import cv2
    fl = [(f, load_fit(f)) for f in fit_names]
    ex = {}
    for f in fit_names:
        with np.load(OUT / f"exact_{f}.npz") as z:
            ex[f] = (z["hand_pen"].copy(), z["hand_gap"].copy())
    bends = {f: wrist_angles(hc.full_pose(fit["lead"]), "right")["bend"] for f, fit in fl}
    obs = hc.Obs()
    phones = hc.Phones()
    H = load_holds()
    LAST_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for si in range(0, len(items), per_sheet):
        rows = []
        for it in items[si:si + per_sheet]:
            k = int(round(float(it["t"]) * HZ))
            pc = hand_site(posed(fl[-1][1]["lead"], np.array([k]))["J"], "right", "palm")[0][0]
            cams = review_cams(obs, posed(fl[0][1]["follow"], np.array([k]))["J"][0], k, pc, 2)
            tl = mesh_tiles(k, fl, obs, phones, cams, tile, point=pc)
            cells = [x for row in tl for x in row]
            head = np.full((tile, 250, 3), 245, np.uint8)
            lines = [f"t {k / HZ:.2f} s  film {film_time_of(k / HZ)}", str(it.get("what", ""))[:44], f"hold {H['names'][k]}"]
            for f in fit_names:
                lines.append(f"{f}: in her {ex[f][0][k] * 100:.1f} cm gap {ex[f][1][k] * 100:.1f}")
                lines.append(f"    wrist bend {bends[f][k]:.0f} deg")
            y = 16
            for ln in lines[:13]:
                cv2.putText(head, ln[:44], (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
                y += 18
            rows.append(np.concatenate([head] + cells, 1))
        Wd = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 4), (0, Wd - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        top = np.full((30, Wd, 3), 255, np.uint8)
        cv2.putText(top, "last pass: per phone the crop | " + " | ".join(fit_names) + " - lead blue, follow orange, his right hand yellow",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        p = LAST_DIR / f"{stem}_{si // per_sheet + 1}.jpg"
        cv2.imwrite(str(p), np.concatenate([top] + rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 86])
        paths.append(p)
        log(f"sheet -> {p}")
    return paths


def added_speed_max(fit_lead: dict, P_ref: np.ndarray, P_new: np.ndarray, ks: np.ndarray) -> dict:
    """the fastest ADDED movement (m/s, the change P_new - P_ref between consecutive frames ks) of his right elbow, palm centre and
    fingertips"""
    o0, o1 = posed(fit_lead, ks, P_ref[ks]), posed(fit_lead, ks, P_new[ks])
    d = {"elbow": o1["J"][:, ELBOW["right"]] - o0["J"][:, ELBOW["right"]],
         "palm": hc.palm_centre(o1["J"], "right") - hc.palm_centre(o0["J"], "right"), "tip": o1["tips"][:, 5:10] - o0["tips"][:, 5:10]}
    return {k: round(float(np.linalg.norm(np.diff(v, axis=0), axis=-1).max()) * HZ, 2) if len(ks) > 1 else 0.0 for k, v in d.items()}


def _added_fast(fit_in_lead: dict, P_in: np.ndarray, fit_lead: dict, P: np.ndarray) -> dict:
    """(T,) bool per point (elbow, tip0..tip4): the motion's change against the input moves faster than 0.55 m/s from the frame to the next
    (arm_speed_checks' "added_gt_0.55" counts these)"""
    oi, o1 = posed(fit_in_lead, None, P_in), posed(fit_lead, None, P)
    out = {"elbow": hc.speeds(o1["J"][:, ELBOW["right"]] - oi["J"][:, ELBOW["right"]]) > 0.55}
    for i in range(5):
        out[f"tip{i}"] = hc.speeds(o1["tips"][:, 5 + i] - oi["tips"][:, 5 + i]) > 0.55
    return {k: np.r_[x, False] for k, x in out.items()}             # (T,): the step into the next frame (arm_speed_checks' count)


def last_pass(base: str = "ec8e", out: str = "v4", cfg: LastCfg | None = None, write: bool = True, only: list | None = None) -> dict:
    """the last pass (see the section): the base motion (the installed ec8e7101) -> steps 1-3 -> FITS[out] (v4/smplx_contact), exact_<out>.npz
    (recomputed where the leader changed), v4/contacts/last/report.json (every spike / window with its numbers and 2D)"""
    from scipy.spatial.transform import Rotation as Ro  # noqa: F401
    cfg = cfg or LastCfg()
    t0 = time.time()
    fit = load_fit(base)
    T = len(fit["lead"]["t"])
    P0 = hc.full_pose(fit["lead"])
    tr0 = np.asarray(fit["lead"]["transl"], float)
    with np.load(OUT / f"exact_{base}.npz") as z:
        ex0 = {k: z[k].copy() for k in z.files}
    rep = {"what": "the last pass (zouk_contacts.last_pass): spikes C2-despiked, deep frames pushed out, folds unfolded", "base": base,
           "cfg": asdict(cfg), "spikes": [], "windows": []}
    obs = hc.Obs()
    judge = hc.CocoJudge()
    bias = hc.calibrate_coco_bias(load_fit("input"), obs, judge)
    bodies = capsule_bodies(load_fit("input"))
    hcfg = HoldCfg()
    # ---- step 1: spikes
    o = posed(fit["lead"], None, P0)
    speeds = {"pelvis": angular_speed(o["R"][:, PELVIS]), "spine3": angular_speed(o["R"][:, SPINE3]),
              "root": np.r_[0.0, np.linalg.norm(np.diff(tr0, axis=0), axis=1) * HZ]}
    bend0 = wrist_angles(P0, "right")["bend"]
    spikes = spike_ranges(ex0["hand_pen"], bend0, speeds, cfg)
    blocked = np.zeros(T, bool)
    for s in spikes:
        blocked[s["a"]:s["b"] + 1] = True
    P1, tr1 = P0.copy(), tr0.copy()
    for s in spikes:
        a, b = s["a"], s["b"]
        best = None
        for nk in range(cfg.knots, 1, -1):
            K = spike_knots(a, b, T, nk, blocked)
            if K is None:
                continue
            Pc, trc = despike_c2(P1, tr1, a, b, K, s["kind"])
            seg = np.arange(max(0, a - 12), min(T, b + 13))              # the run and 12 frames each side (its neighbours)
            sp = _segment_speeds(Pc, _with_pose(fit, Pc, trc)["lead"], seg)
            gi = np.arange(a - 1 - seg[0], b + 1 - seg[0])                # the steps into / inside / out of the run
            nbi = np.setdiff1d(np.arange(len(seg) - 1), gi)
            ratio = max(float(sp[k][gi].max()) / max(float(sp[k][nbi].max()), 0.05) for k in sp)
            if best is None or ratio < best[0] - 1e-9:
                best = (ratio, nk, Pc, trc)
            if ratio <= cfg.speed_slack:
                break
        if best is None:
            s["skipped"] = "no free frames around it"
            continue
        s["knots"], s["speed_ratio_vs_neighbours"] = best[1], round(best[0], 2)
        P1, tr1 = best[2], best[3]
    ch1 = np.flatnonzero((np.abs(P1 - P0).max((1, 2)) > 0) | (np.abs(tr1 - tr0).max(1) > 0))
    fit1 = _with_pose(fit, P1, tr1)
    ex1 = {k: v.copy() for k, v in ex0.items()}
    if len(ch1):
        e = exact_of(fit1, ch1, cfg.device, 24)
        for k in ("hand_pen", "arm_pen", "hand_gap", "hand_pen_part"):
            ex1[k][ch1] = e[k]
    for s in spikes:
        if "skipped" in s:
            continue
        x = np.arange(s["a"], s["b"] + 1)
        s["pen_cm_after"] = round(float(ex1["hand_pen"][x].max()) * 100, 1)
        s["bend_deg_after"] = round(float(wrist_angles(P1[x], "right")["bend"].max()), 1)
        s["2d"] = _elbow_guard({"before": fit, "after": fit1}, x, obs, judge, bias, bodies, hcfg)
    rep["spikes"] = spikes
    rep["step1_frames_changed"] = int(len(ch1))
    log(f"last pass step 1: {len(spikes)} spikes, {len(ch1)} frames ({time.time() - t0:.0f} s)")
    # ---- steps 2 / 3: push out, unfold
    bend1 = wrist_angles(P1, "right")["bend"]
    # the contact to keep: where the INSTALLED motion's hand touched her (a despike may have lifted it); the back holds' floating frames
    # in a window are pulled onto her (step 4)
    wins = last_windows(ex1["hand_pen"], ex0["hand_gap"], bend1, cfg, back=np.isin(load_holds()["names"], BACK_HOLDS))
    rest_J = rest_body(fit["lead"]["betas"])["J"]
    pts = _arm_points(fit["lead"]["betas"])
    fit_in = load_fit("input")
    P_in = hc.full_pose(fit_in["lead"])
    o_in = posed(fit_in["lead"], None, P_in)
    p99_in = {k: float(np.percentile(hc.speeds(x), 99)) for k, x in
              (("E", o_in["J"][:, ELBOW["right"]]), ("W", o_in["J"][:, WRIST["right"]]), ("C", hc.palm_centre(o_in["J"], "right")))}
    p99_in["tips"] = float(min(np.percentile(hc.speeds(o_in["tips"][:, 5 + i]), 99) for i in range(5)))
    surf = hc.GpuSurface(hc.Body(fit["lead"], "cpu").faces, len(hc.vertex_joint()), cfg.device)
    bf = hc.Body(fit["follow"], cfg.device)
    PF, trF = hc.full_pose(fit["follow"]), np.asarray(fit["follow"]["transl"], float)
    P2 = P1.copy()
    solved = {}
    for wi, w in enumerate(wins):
        s0, s1 = w["span"]
        c0, c1 = w["core"]
        if only and not any(s0 / HZ <= float(t) <= s1 / HZ for t in only):     # a trial on some windows (only: seconds)
            continue
        ks = np.arange(s0, s1 + 1)
        win = ArmWindow(fit1, P1, ks, c0 - s0, s1 - c1, rest_J, pts)
        win.set_input(fit_in["lead"], P_in, p99_in)
        VF = bf.forward_np(PF[ks], trF[ks], verts=True, batch=64)["verts"]
        flags = {k: np.isin(ks, w[k]) for k in ("push", "unfold", "touch", "pull")}
        v, r = solve_arm_window(win, VF, surf, flags, cfg, pts)
        Pw = win.poses(v, P1)
        fit_w = _with_pose(fit1, Pw, tr1)
        g = _elbow_guard({"before": fit1, "after": fit_w}, np.arange(c0, c1 + 1), obs, judge, bias, bodies, hcfg)
        r.update({"id": f"last:{wi}", "t": w["t"], "t_span": [round(s0 / HZ, 2), round(s1 / HZ, 2)], "2d": g})
        el = g["elbow"]
        worse = (el["after_px"] - el["before_px"]) if (el["after_px"] is not None and el["before_px"] is not None) else 0.0
        r["vetoed"] = bool(el["n"] >= cfg.guard_min_n and worse > cfg.elbow_guard_px)
        if r["vetoed"]:
            r["why"] = f"the COCO elbow (phones that see it) {el['before_px']} -> {el['after_px']} px (n {el['n']})"
        else:
            P2[ks] = Pw[ks]
            solved[wi] = {"win": win, "v": v, "span": (s0, s1), "r": r}
        rep["windows"].append(r)
        log(f"last pass window {wi + 1}/{len(wins)} {w['t']}: {r['depth_cm_max_before_after']} cm, bend {r['bend_deg_max_before_after']}, "
            f"move {r['hand_move_cm_max']} cm, elbow 2D {el['before_px']} -> {el['after_px']} px{' VETO' if r['vetoed'] else ''} "
            f"({time.time() - t0:.0f} s)")
    # ---- the post-hoc check of every window against the installed motion (the base): no new gate flag in its span (zouk_contacts.gate_flags:
    # whip, wrist flick, twist, elbow hyperextension / off-hinge, fold) and no new penetration (> 2 cm and 1 cm deeper than the base) - a
    # window that has one is scaled back (its numbers x 0.75, 0.5, 0.25,
    # then 0 = vetoed); and over the film no more frames whose change against the input moves faster than 0.55 m/s (the elbow; each
    # fingertip, counted as arm_speed_checks does: the worst finger) and no more floating frames in the back holds than the base - while a
    # budget is exceeded, the windows adding the most to it are scaled back
    back = np.isin(load_holds()["names"], BACK_HOLDS)
    fit_base = fit
    steps_s = [1.0, 0.75, 0.5, 0.25, 0.0]
    scale = {wi: 0 for wi in solved}
    base_add = _added_fast(fit_in["lead"], P_in, fit_base["lead"], P0)
    float0 = (ex0["hand_gap"] > 0.02) & back
    tips = [f"tip{i}" for i in range(5)]
    for rnd in range(16):
        P2 = P1.copy()
        for wi, sv in solved.items():
            P2 = sv["win"].poses(sv["v"] * steps_s[scale[wi]], P2) if steps_s[scale[wi]] > 0 else P2
        fit2 = _with_pose(fit, P2, tr1)
        ch2 = np.flatnonzero((np.abs(P2 - P0).max((1, 2)) > 0) | (np.abs(tr1 - tr0).max(1) > 0))
        ex2 = {k: v.copy() for k, v in ex0.items()}
        if len(ch2):
            e = exact_of(fit2, ch2, cfg.device, 24)
            for k in ("hand_pen", "arm_pen", "hand_gap", "hand_pen_part"):
                ex2[k][ch2] = e[k]
        gf = gate_flags({"input": fit_in, "ref": fit_base, "cand": fit2}, ref="ref", cand="cand", added_vs_ref=True)
        add2 = _added_fast(fit_in["lead"], P_in, fit2["lead"], P2)
        float2 = (ex2["hand_gap"] > 0.02) & back
        live = [wi for wi in solved if steps_s[scale[wi]] > 0]
        moved = []
        for wi in live:
            a_, b_ = solved[wi]["span"]
            sl = slice(a_, b_ + 1)
            why = [f"{k} {int(v_[sl].sum())}" for k, v_ in gf["flags"].items() if v_[sl].any()]
            # ... nor a hand pushed (or unfolded) INTO her: frames deeper than 2 cm and 1 cm deeper than the base (exact)
            deeper = int(((ex2["hand_pen"][sl] > 0.02) & (ex2["hand_pen"][sl] > ex0["hand_pen"][sl] + 0.01)).sum())
            if deeper:
                why.append(f"new penetration {deeper}")
            if why:
                moved.append((wi, why))
        if not moved:
            over = []
            if add2["elbow"].sum() > base_add["elbow"].sum():
                over.append("elbow")
            if max(add2[k].sum() for k in tips) > max(base_add[k].sum() for k in tips):
                over += tips
            if float2.sum() > float0.sum():
                over.append("float")
            if over:
                def contrib(wi):
                    a_, b_ = solved[wi]["span"]
                    sl = slice(a_, b_ + 1)
                    c = sum(int(add2[k][sl].sum()) - int(base_add[k][sl].sum()) for k in over if k != "float")
                    if "float" in over:
                        c += int(float2[sl].sum()) - int(float0[sl].sum())
                    return c
                cs = sorted(((contrib(wi), wi) for wi in live), reverse=True)
                for c, wi in cs[:2]:
                    if c > 0:
                        moved.append((wi, [f"over the film's budget ({', '.join(over)}): this window adds {c}"]))
        for wi, why in moved:
            scale[wi] += 1
            solved[wi]["r"].setdefault("scaled_back", []).append({"to": steps_s[scale[wi]], "why": why})
        n_e, n_t = int(add2["elbow"].sum()), max(int(add2[k].sum()) for k in tips)
        b_e, b_t = int(base_add["elbow"].sum()), max(int(base_add[k].sum()) for k in tips)
        log(f"last pass check round {rnd + 1}: scales {[steps_s[scale[wi]] for wi in solved]}; film: elbow added {n_e} (base {b_e}), "
            f"worst fingertip {n_t} (base {b_t}), floating {int(float2.sum())} (base {int(float0.sum())})")
        if not moved:
            break
    rep["check"] = {"rounds": rnd + 1, "elbow_added": [b_e, n_e], "fingertip_added_worst": [b_t, n_t],
                    "floating_back": [int(float0.sum()), int(float2.sum())],
                    "new_gate_flags": {k: int(v_.sum()) for k, v_ in gf["flags"].items()}}
    for wi, sv in solved.items():
        sv["r"]["scale"] = steps_s[scale[wi]]
        if steps_s[scale[wi]] == 0:
            sv["r"]["vetoed"] = True
            sv["r"]["why"] = "the post-hoc check: " + "; ".join(sv["r"]["scaled_back"][-1]["why"])
    for r in rep["windows"]:
        xs = np.arange(int(round(r["t_span"][0] * HZ)), int(round(r["t_span"][1] * HZ)) + 1)
        r["float_back_gt2cm_before_after"] = [int(((ex0["hand_gap"][xs] > 0.02) & back[xs]).sum()),
                                              int(((ex2["hand_gap"][xs] > 0.02) & back[xs]).sum())]
        r["added_speed_mps_max"] = added_speed_max(fit2["lead"], P1, P2, xs)
        x = np.arange(r["core"][0], r["core"][1] + 1)
        r["exact_core_depth_cm_max_after"] = round(float(ex2["hand_pen"][x].max()) * 100, 2)
        r["exact_core_frames_gt5cm_after"] = int((ex2["hand_pen"][x] > 0.05).sum())
        r["bend_deg_max_after"] = round(float(wrist_angles(P2[x], "right")["bend"].max()), 1)
    rep["frames_changed_vs_base"] = int(len(ch2))
    rep["seconds"] = round(time.time() - t0, 1)
    LAST_DIR.mkdir(parents=True, exist_ok=True)
    if only:
        write_json(LAST_DIR / "trial.json", rep)
        np.savez_compressed(LAST_DIR / "trial_poses.npz", lead=P2, transl=tr1)
        return rep
    np.savez_compressed(LAST_DIR / "poses_step1.npz", lead=P1, transl=tr1)
    np.savez_compressed(LAST_DIR / "poses_final.npz", lead=P2, transl=tr1)
    if write:
        b2 = {r: dict(fit[r]) for r in ROLES}
        b2["lead"]["transl"] = tr1
        rep["written"] = write_fit_v4(b2, {"lead": P2, "follow": hc.full_pose(fit["follow"])}, FITS[out])
        np.savez_compressed(OUT / f"exact_{out}.npz", ks=np.arange(T), parts=ex0["parts"],
                            **{k: ex2[k] for k in ("hand_pen", "arm_pen", "hand_gap", "hand_pen_part")})
    write_json(LAST_DIR / "report.json", rep)
    log(f"last pass: {len(ch2)} frames changed vs {base} ({time.time() - t0:.0f} s)")
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
    p.add_argument("--changed-vs", default=None, help="only the frames where the fit's leader differs from this fit (the rest copied)")
    p = sub.add_parser("plan")
    p.add_argument("--base", default="v3")
    p.add_argument("--out", default="plan.json")
    p.add_argument("--problems", default=None, help="comma list: plan only these kinds (e.g. wrist,self_forearm; exact = the hand inside "
                                                    "her torso / shoulders in exact_<base>.npz)")
    p.add_argument("--follow-phones", action="store_true", help="the final pass: one window per run, each with its ray phone")
    p.add_argument("--max-core-s", type=float, default=None)
    p.add_argument("--merge-s", type=float, default=None)
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
    p.add_argument("--base-tag", default=None, help="a stage: curation.json['stages'][tag], back to runs/<base-tag>/")
    p = sub.add_parser("gate")
    p.add_argument("--tag", default="pass3")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--no-coco", action="store_true")
    p.add_argument("--tags", default=None, help="comma list of the passes whose adopted windows the gate tests (default all2,pass2,pass3)")
    p.add_argument("--base-tag", default=None, help="the final pass: the reference motion runs/<base-tag>/ (e.g. inst); vetoes in stages")
    p.add_argument("--exact", action="store_true", help="also the exact mesh check of the candidate (GPU)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vis", action="store_true", help="the closing pass: the 2D checks count only the phones that see the candidate's joint")
    p = sub.add_parser("make-base")
    p.add_argument("--src", default="inst2")
    p.add_argument("--out", default="fin2")
    p.add_argument("--despike", default="[[165.73, 166.0, \"body\"], [347.43, 347.63, \"arm\"]]", help="json [[a_s, b_s, arm|body], ...]")
    p = sub.add_parser("validate-final")
    p.add_argument("--fits", default="input,v3,v4")
    p.add_argument("--changed-vs", default="v3")
    p = sub.add_parser("before-after")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--extra", default="118,47,155")
    sub.add_parser("unity-motion")
    p = sub.add_parser("last-pass", help="the last pass: spikes, deep frames, folds of the installed motion -> v4/smplx_contact")
    p.add_argument("--base", default="ec8e")
    p.add_argument("--out", default="v4")
    p.add_argument("--cfg", default=None, help="json of LastCfg fields")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--only", default=None, help="a trial: only the windows around these seconds (comma list), report in last/trial.json")
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
            if a.changed_vs:
                exact_update(f, a.changed_vs, a.device)
            else:
                exact_measure(f, a.device)
    elif a.cmd == "plan":
        kw = {"base": a.base, "plan": a.out, "problems": tuple(a.problems.split(",")) if a.problems else (), "follow_phones": a.follow_phones}
        if a.max_core_s:
            kw["max_core_s"] = a.max_core_s
        if a.merge_s is not None:
            kw["merge_s"] = a.merge_s
        plan_windows(ZoukCfg(**kw))
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
        print(json.dumps(py_json(apply_curation(a.tag, base_tag=a.base_tag)), indent=1)[:3000])
    elif a.cmd == "gate":
        r = run_gate(a.tag, a.rounds, not a.no_coco, tags=a.tags.split(",") if a.tags else None, base_tag=a.base_tag, exact=a.exact,
                     device=a.device, vis=a.vis)
        print(json.dumps(py_json({"vetoed": r["windows_vetoed"], "rounds": r["rounds"]}), indent=1)[:6000])
    elif a.cmd == "make-base":
        print(json.dumps(py_json(make_base(a.src, a.out, json.loads(a.despike))), indent=1))
    elif a.cmd == "validate-final":
        v = validate_final(tuple(a.fits.split(",")), a.changed_vs)
        print(validate_final_md(v))
    elif a.cmd == "validate":
        v = validate()
        print(json.dumps(py_json(v["table"]), indent=1)[:8000])
    elif a.cmd == "before-after":
        before_after_sheets(a.n, [int(x) for x in a.extra.split(",")] if a.extra else [])
    elif a.cmd == "unity-motion":
        print(json.dumps(py_json(export_motion()), indent=1))
    elif a.cmd == "last-pass":
        r = last_pass(a.base, a.out, LastCfg(**json.loads(a.cfg)) if a.cfg else None, write=not a.no_write,
                      only=[float(x) for x in a.only.split(",")] if a.only else None)
        print(json.dumps(py_json({k: r[k] for k in ("step1_frames_changed", "frames_changed_vs_base", "seconds")}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
